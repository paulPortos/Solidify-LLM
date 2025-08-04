import torch
import gc
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
)
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer
from transformers.utils.quantization_config import BitsAndBytesConfig
from transformers.data.data_collator import DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables and authenticate
load_dotenv()
hf_token = os.getenv('HUGGINGFACE_API_KEY')
if hf_token:
    login(token=hf_token)
    print("Authenticated with Hugging Face")
else:
    print("Warning: HUGGINGFACE_API_KEY not found in .env file")

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "seyyedaliayati/solidity-defi-vulnerabilities"
OUTPUT_DIR = "./version1_0/qwen-solidity-vulnerabilities"
# Reduced from 512 for memory safety
MAX_SEQ_LENGTH = 384
# Reduced from 2 for 4GB VRAM
BATCH_SIZE = 1
EPOCHS = 3
LEARNING_RATE = 2e-4

print("Setting up 4-bit quantization config...")
# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading model in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

print("Preparing model for k-bit training...")
model = prepare_model_for_kbit_training(model)

print("Setting up LoRA config...")
# LoRA configuration with rank, alpha scaling, target modules, dropout, bias and task type
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

print("Applying LoRA to model...")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("Loading dataset...")
# Load and prepare dataset
dataset = load_dataset(DATASET_NAME)
print(f"Dataset loaded. Train samples: {len(dataset['train'])}") # type: ignore

def format_prompt(example):
    """Format the dataset into instruction-response format"""
    vulnerability = example.get('attack_title', '')
    code = example.get('testcase', '')
    description = example.get('attack_explain', '')
    title = example.get('title', '')
    
    # Create instruction prompt
    instruction = f"Analyze this Solidity code for vulnerabilities:\n\n{code}\n\nIdentify any security vulnerabilities present."
    
    # Create response
    response = f"Vulnerability: {vulnerability}\n\nTarget: {title}\n\nDescription: {description}"
    
    # Format as chat template
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response}
    ]
    
    return {"messages": messages}

def tokenize_function(examples):
    """Tokenize the formatted prompts"""
    texts = []
    for messages in examples["messages"]:
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(text)
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=MAX_SEQ_LENGTH,
        return_tensors=None
    )
    
    # Set labels to input_ids for causal language modeling
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

print("Formatting dataset...")
# Format and tokenize dataset
formatted_dataset = dataset.map(format_prompt, remove_columns=dataset["train"].column_names) # type: ignore
tokenized_dataset = formatted_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=formatted_dataset["train"].column_names # type: ignore
)

print("Setting up training arguments...")
# Training arguments - Fixed evaluation strategy issue
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    # Increased to maintain effective batch size of 8
    gradient_accumulation_steps=8,
    optim="paged_adamw_8bit",
    save_steps=500,
    logging_steps=25,
    learning_rate=LEARNING_RATE,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    save_total_limit=3,
    # Fixed: Disabled since we don't have eval dataset
    load_best_model_at_end=False,
)

print("Setting up trainer...")
# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"], # type: ignore
    data_collator=data_collator,
)

print("Starting training...")
print(f"Training for {EPOCHS} epochs with batch size {BATCH_SIZE}")
print(f"Effective batch size: {BATCH_SIZE * training_args.gradient_accumulation_steps}")

# Start training
trainer.train()

print("Saving final model...")
# Save the final model
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training completed!")
print(f"Model saved to: {OUTPUT_DIR}")

# Test the fine-tuned model
print("\n" + "="*50)
print("Testing the fine-tuned model...")

# Clean up memory before testing
del trainer
torch.cuda.empty_cache()
gc.collect()

# Load the fine-tuned model for testing
print("Loading base model for testing...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

print("Loading fine-tuned LoRA weights...")
model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
model.eval()

# Test with a sample Solidity code
test_code = """
pragma solidity ^0.8.0;

interface IERC20 {
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

interface IUniswapV2Pair {
    function getReserves() external view returns (uint112 reserve0, uint112 reserve1, uint32 blockTimestampLast);
    function token0() external view returns (address);
    function token1() external view returns (address);
}

contract VulnerableYieldFarm {
    IERC20 public rewardToken;
    IERC20 public stakingToken;
    IUniswapV2Pair public pricePair;
    
    mapping(address => uint256) public stakedBalance;
    mapping(address => uint256) public lastRewardTime;
    
    uint256 public rewardRate = 100; // 100 tokens per second per staked token
    uint256 public totalStaked;
    
    constructor(address _rewardToken, address _stakingToken, address _pricePair) {
        rewardToken = IERC20(_rewardToken);
        stakingToken = IERC20(_stakingToken);
        pricePair = IUniswapV2Pair(_pricePair);
    }
    
    function stake(uint256 amount) external {
        require(amount > 0, "Cannot stake 0");
        
        stakingToken.transferFrom(msg.sender, address(this), amount);
        
        if (stakedBalance[msg.sender] > 0) {
            claimRewards();
        }
        
        stakedBalance[msg.sender] += amount;
        totalStaked += amount;
        lastRewardTime[msg.sender] = block.timestamp;
    }
    
    function unstake(uint256 amount) external {
        require(stakedBalance[msg.sender] >= amount, "Insufficient staked balance");
        
        claimRewards();
        
        stakedBalance[msg.sender] -= amount;
        totalStaked -= amount;
        stakingToken.transfer(msg.sender, amount);
        lastRewardTime[msg.sender] = block.timestamp;
    }
    
    function claimRewards() public {
        uint256 reward = calculateReward(msg.sender);
        if (reward > 0) {
            lastRewardTime[msg.sender] = block.timestamp;
            rewardToken.transfer(msg.sender, reward);
        }
    }
    
    function calculateReward(address user) public view returns (uint256) {
        if (stakedBalance[user] == 0) return 0;
        
        uint256 timeStaked = block.timestamp - lastRewardTime[user];
        uint256 baseReward = stakedBalance[user] * timeStaked * rewardRate;
        
        // VULNERABILITY: Using spot price from DEX for reward calculation
        uint256 multiplier = getPriceMultiplier();
        
        return baseReward * multiplier / 1000;
    }
    
    function getPriceMultiplier() public view returns (uint256) {
        (uint112 reserve0, uint112 reserve1,) = pricePair.getReserves();
        
        // VULNERABILITY: Direct use of reserves without TWAP or oracle
        // This can be manipulated with flash loans
        uint256 price;
        if (pricePair.token0() == address(stakingToken)) {
            price = (reserve1 * 1e18) / reserve0;
        } else {
            price = (reserve0 * 1e18) / reserve1;
        }
        
        // Higher price = higher multiplier (up to 5x)
        if (price > 2e18) return 5000; // 5x multiplier
        if (price > 1.5e18) return 3000; // 3x multiplier
        if (price > 1e18) return 2000; // 2x multiplier
        return 1000; // 1x multiplier (base case)
    }
    
    function emergencyWithdraw() external {
        uint256 amount = stakedBalance[msg.sender];
        require(amount > 0, "No staked balance");
        
        stakedBalance[msg.sender] = 0;
        totalStaked -= amount;
        stakingToken.transfer(msg.sender, amount);
    }
    
    function updateRewardRate(uint256 newRate) external {
        // VULNERABILITY: No access control
        rewardRate = newRate;
    }
}
"""

messages = [
    {"role": "user", "content": f"Analyze this Solidity code for vulnerabilities:\n\n{test_code}\n\nIdentify any security vulnerabilities present."}
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

print("Generating response with fine-tuned model...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.5,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print("Fine-tuned model response:")
print(response)
