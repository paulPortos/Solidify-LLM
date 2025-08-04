import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re
from datetime import datetime

class SolidityVulnerabilityTester:
    def __init__(self, model_path, base_model="Qwen/Qwen2.5-0.5B-Instruct"):
        """Initialize the vulnerability tester with the fine-tuned model"""
        self.model_path = model_path
        self.base_model = base_model
        self.tokenizer = None
        self.model = None
        
        # Verify model path exists
        self.verify_model_path()
        
        # Test cases with known vulnerabilities
        self.test_cases = [
            {
                "name": "Reentrancy Attack",
                "code": """
pragma solidity ^0.8.0;

contract VulnerableBank {
    mapping(address => uint256) public balances;
    
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
    
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
        
        balances[msg.sender] -= amount;
    }
}
""",
                "expected_vulnerabilities": ["reentrancy", "reentrancy attack", "state change after external call"],
                "vulnerability_type": "Reentrancy",
                "severity": "High"
            },
            
            {
                "name": "Integer Overflow",
                "code": """
pragma solidity ^0.7.0;

contract VulnerableToken {
    mapping(address => uint256) public balances;
    uint256 public totalSupply;
    
    function mint(address to, uint256 amount) public {
        balances[to] += amount;
        totalSupply += amount;
    }
    
    function transfer(address to, uint256 amount) public {
        require(balances[msg.sender] >= amount);
        balances[msg.sender] -= amount;
        balances[to] += amount;
    }
}
""",
                "expected_vulnerabilities": ["integer overflow", "overflow", "arithmetic overflow", "unchecked math"],
                "vulnerability_type": "Integer Overflow",
                "severity": "High"
            },
            
            {
                "name": "Price Manipulation",
                "code": """
pragma solidity ^0.8.0;

interface IUniswapV2Pair {
    function getReserves() external view returns (uint112, uint112, uint32);
}

contract VulnerableDEX {
    IUniswapV2Pair public pair;
    
    function getPrice() public view returns (uint256) {
        (uint112 reserve0, uint112 reserve1,) = pair.getReserves();
        return (reserve1 * 1e18) / reserve0;
    }
    
    function calculateReward(uint256 amount) public view returns (uint256) {
        uint256 price = getPrice();
        return amount * price / 1e18;
    }
}
""",
                "expected_vulnerabilities": ["price manipulation", "oracle manipulation", "flash loan", "spot price"],
                "vulnerability_type": "Price Manipulation",
                "severity": "High"
            },
            
            {
                "name": "Access Control Missing",
                "code": """
pragma solidity ^0.8.0;

contract VulnerableContract {
    address public owner;
    uint256 public criticalValue;
    
    constructor() {
        owner = msg.sender;
    }
    
    function setCriticalValue(uint256 _value) public {
        criticalValue = _value;
    }
    
    function emergencyWithdraw() public {
        payable(msg.sender).transfer(address(this).balance);
    }
}
""",
                "expected_vulnerabilities": ["access control", "missing modifier", "unauthorized access", "no owner check"],
                "vulnerability_type": "Access Control",
                "severity": "Critical"
            },
            
            {
                "name": "Denial of Service",
                "code": """
pragma solidity ^0.8.0;

contract VulnerableAuction {
    address public highestBidder;
    uint256 public highestBid;
    
    function bid() public payable {
        require(msg.value > highestBid, "Bid too low");
        
        if (highestBidder != address(0)) {
            payable(highestBidder).transfer(highestBid);
        }
        
        highestBidder = msg.sender;
        highestBid = msg.value;
    }
}
""",
                "expected_vulnerabilities": ["denial of service", "dos", "failed transfer", "external call failure"],
                "vulnerability_type": "Denial of Service",
                "severity": "Medium"
            },
            
            {
                "name": "Timestamp Dependence",
                "code": """
pragma solidity ^0.8.0;

contract VulnerableTimeLock {
    mapping(address => uint256) public lockTime;
    mapping(address => uint256) public balances;
    
    function deposit() public payable {
        balances[msg.sender] += msg.value;
        lockTime[msg.sender] = block.timestamp + 1 days;
    }
    
    function withdraw() public {
        require(block.timestamp > lockTime[msg.sender], "Still locked");
        require(balances[msg.sender] > 0, "No balance");
        
        uint256 amount = balances[msg.sender];
        balances[msg.sender] = 0;
        payable(msg.sender).transfer(amount);
    }
}
""",
                "expected_vulnerabilities": ["timestamp dependence", "block.timestamp", "miner manipulation", "temporal vulnerability"],
                "vulnerability_type": "Timestamp Dependence",
                "severity": "Low"
            },
            
            {
                "name": "Unchecked Return Value",
                "code": """
pragma solidity ^0.8.0;

interface IERC20 {
    function transfer(address to, uint256 amount) external returns (bool);
}

contract VulnerableTransfer {
    IERC20 public token;
    
    function distributeTokens(address[] memory recipients, uint256 amount) public {
        for (uint i = 0; i < recipients.length; i++) {
            token.transfer(recipients[i], amount);
        }
    }
}
""",
                "expected_vulnerabilities": ["unchecked return", "failed transfer", "silent failure", "return value"],
                "vulnerability_type": "Unchecked Return Value",
                "severity": "Medium"
            },
            
            {
                "name": "Front Running",
                "code": """
pragma solidity ^0.8.0;

contract VulnerableCommitReveal {
    mapping(address => bytes32) public commits;
    mapping(address => uint256) public reveals;
    
    function commit(bytes32 commitment) public {
        commits[msg.sender] = commitment;
    }
    
    function reveal(uint256 value, uint256 nonce) public {
        bytes32 hash = keccak256(abi.encodePacked(value, nonce));
        require(commits[msg.sender] == hash, "Invalid reveal");
        reveals[msg.sender] = value;
    }
}
""",
                "expected_vulnerabilities": ["front running", "mev", "transaction ordering", "mempool"],
                "vulnerability_type": "Front Running",
                "severity": "Medium"
            }
        ]
    
    def verify_model_path(self):
        """Verify that the fine-tuned model path and files exist"""
        if not os.path.exists(self.model_path):
            print(f"❌ Model directory not found: {self.model_path}")
            return False
            
        # Check for essential LoRA adapter files
        required_files = [
            "adapter_config.json",
            "adapter_model.safetensors"
        ]
        
        missing_files = []
        for file in required_files:
            file_path = os.path.join(self.model_path, file)
            if not os.path.exists(file_path):
                missing_files.append(file)
        
        if missing_files:
            print(f"❌ Missing required model files: {missing_files}")
            print(f"📁 In directory: {os.path.abspath(self.model_path)}")
            return False
            
        print(f"✅ Fine-tuned model files verified in: {os.path.abspath(self.model_path)}")
        return True
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        try:
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            
            print("Loading base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            print(f"Loading fine-tuned adapter from: {self.model_path}")
            # Load the fine-tuned LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model.eval()
            
            print("✅ Fine-tuned model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print(f"❌ Make sure the model path exists: {self.model_path}")
            return False
    
    def analyze_code(self, code, max_length=512):
        """Analyze Solidity code for vulnerabilities"""
        if not self.model or not self.tokenizer:
            return "Model not loaded"
        
        # Create instruction prompt
        instruction = f"Analyze this Solidity code for vulnerabilities:\n\n{code}\n\nIdentify any security vulnerabilities present."
        
        # Format as chat template
        messages = [{"role": "user", "content": instruction}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        
        # Move inputs to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()
        elif "assistant" in response:
            response = response.split("assistant")[-1].strip()
        
        return response
    
    def evaluate_response(self, response, expected_vulnerabilities):
        """Evaluate if the model response contains expected vulnerabilities"""
        response_lower = response.lower()
        
        # Check for exact matches and partial matches
        found_vulnerabilities = []
        for vuln in expected_vulnerabilities:
            if vuln.lower() in response_lower:
                found_vulnerabilities.append(vuln)
        
        accuracy = len(found_vulnerabilities) / len(expected_vulnerabilities) if expected_vulnerabilities else 0
        return accuracy, found_vulnerabilities
    
    def run_comprehensive_test(self):
        """Run comprehensive vulnerability detection test"""
        if not self.load_model():
            return
        
        print("\n" + "="*80)
        print("🔍 SOLIDITY VULNERABILITY DETECTION TEST")
        print("="*80)
        
        results = {
            "total_tests": len(self.test_cases),
            "correct_detections": 0,
            "partial_detections": 0,
            "missed_detections": 0,
            "detailed_results": [],
            "accuracy_by_type": {},
            "overall_accuracy": 0
        }
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n📋 Test {i}/{len(self.test_cases)}: {test_case['name']}")
            print(f"🎯 Expected: {test_case['vulnerability_type']} ({test_case['severity']} severity)")
            print("-" * 50)
            
            try:
                # Analyze the code
                response = self.analyze_code(test_case['code'])
                
                # Evaluate the response
                accuracy, found_vulns = self.evaluate_response(response, test_case['expected_vulnerabilities'])
                
                # Categorize result
                if accuracy >= 0.8:
                    result_category = "✅ CORRECT"
                    results["correct_detections"] += 1
                elif accuracy >= 0.3:
                    result_category = "⚠️ PARTIAL"
                    results["partial_detections"] += 1
                else:
                    result_category = "❌ MISSED"
                    results["missed_detections"] += 1
                
                # Store detailed results
                detailed_result = {
                    "test_name": test_case['name'],
                    "vulnerability_type": test_case['vulnerability_type'],
                    "severity": test_case['severity'],
                    "accuracy": accuracy,
                    "found_vulnerabilities": found_vulns,
                    "expected_vulnerabilities": test_case['expected_vulnerabilities'],
                    "model_response": response[:200] + "..." if len(response) > 200 else response,
                    "result_category": result_category
                }
                results["detailed_results"].append(detailed_result)
                
                # Update accuracy by type
                vuln_type = test_case['vulnerability_type']
                if vuln_type not in results["accuracy_by_type"]:
                    results["accuracy_by_type"][vuln_type] = []
                results["accuracy_by_type"][vuln_type].append(accuracy)
                
                # Print results
                print(f"🤖 Model Response: {response[:150]}{'...' if len(response) > 150 else ''}")
                print(f"🎯 Found Vulnerabilities: {found_vulns}")
                print(f"📊 Accuracy: {accuracy:.2%}")
                print(f"📝 Result: {result_category}")
                
            except Exception as e:
                print(f"❌ Error testing {test_case['name']}: {e}")
                results["detailed_results"].append({
                    "test_name": test_case['name'],
                    "error": str(e),
                    "result_category": "❌ ERROR"
                })
        
        # Calculate overall metrics
        total_accuracy = sum(result['accuracy'] for result in results["detailed_results"] if 'accuracy' in result)
        results["overall_accuracy"] = total_accuracy / len(self.test_cases) if self.test_cases else 0
        
        # Calculate accuracy by vulnerability type
        for vuln_type, accuracies in results["accuracy_by_type"].items():
            results["accuracy_by_type"][vuln_type] = sum(accuracies) / len(accuracies)
        
        self.print_summary(results)
        self.save_results(results)
        
        return results
    
    def print_summary(self, results):
        """Print comprehensive test summary"""
        print("\n" + "="*80)
        print("📊 TEST SUMMARY REPORT")
        print("="*80)
        
        # Overall metrics
        print(f"📈 Overall Accuracy: {results['overall_accuracy']:.2%}")
        print(f"✅ Correct Detections: {results['correct_detections']}/{results['total_tests']} ({results['correct_detections']/results['total_tests']:.1%})")
        print(f"⚠️ Partial Detections: {results['partial_detections']}/{results['total_tests']} ({results['partial_detections']/results['total_tests']:.1%})")
        print(f"❌ Missed Detections: {results['missed_detections']}/{results['total_tests']} ({results['missed_detections']/results['total_tests']:.1%})")
        
        # Accuracy by vulnerability type
        print(f"\n📋 Accuracy by Vulnerability Type:")
        print("-" * 50)
        for vuln_type, accuracy in results["accuracy_by_type"].items():
            print(f"  {vuln_type}: {accuracy:.2%}")
        
        # Performance rating
        overall_acc = results['overall_accuracy']
        if overall_acc >= 0.8:
            rating = "🌟 EXCELLENT"
            color = "🟢"
        elif overall_acc >= 0.6:
            rating = "👍 GOOD"
            color = "🟡"
        elif overall_acc >= 0.4:
            rating = "⚠️ FAIR"
            color = "🟠"
        else:
            rating = "❌ POOR"
            color = "🔴"
        
        print(f"\n{color} Model Performance Rating: {rating}")
        print(f"📊 Score: {overall_acc:.1%}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if overall_acc < 0.5:
            print("  • Consider increasing training epochs")
            print("  • Improve data quality and formatting")
            print("  • Add more diverse vulnerability examples")
        elif overall_acc < 0.7:
            print("  • Fine-tune hyperparameters")
            print("  • Add more training data")
        else:
            print("  • Model performs well!")
            print("  • Consider testing on more complex vulnerabilities")
    
    def save_results(self, results):
        """Save detailed results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vulnerability_test_results_{timestamp}.json"
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {filename}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")

def main():
    """Main function to run the vulnerability detection test"""
    # Correct path to your fine-tuned model
    model_path = "../qwen-solidity-vulnerabilities"
    
    # Check if model path exists
    if not os.path.exists(model_path):
        print(f"❌ Model path not found: {model_path}")
        print("📁 Please ensure your fine-tuned model is in the correct location:")
        print("   Expected: version1_0/qwen-solidity-vulnerabilities/")
        print("   Files should include: adapter_config.json, adapter_model.safetensors, etc.")
        return
    
    print("🚀 Starting Solidity Vulnerability Detection Test...")
    print(f"📁 Model Path: {os.path.abspath(model_path)}")
    
    tester = SolidityVulnerabilityTester(model_path)
    results = tester.run_comprehensive_test()
    
    print("\n🎉 Testing completed!")

if __name__ == "__main__":
    main()
