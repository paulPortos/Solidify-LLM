# Solidity Vulnerability Identifier - Model Performance Goals

## Project Overview
This project aims to fine-tune a language model to identify security vulnerabilities in Solidity smart contracts. The model will serve as a co-pilot assistant, not a final decision maker.

## Target Performance Metrics by Use Case

### 1. Developer Assistant Tool (Copilot-style helper)
**Target Accuracy: 70-80%**

**Rationale:** 
- Smart suggestions to speed up development and auditing
- False positives are acceptable as long as critical issues are flagged
- Focus on developer productivity enhancement

**Characteristics:**
- ✅ Tolerant to: Minor mistakes and false positives
- 🚫 Not suitable for: Mission-critical auditing without human oversight

### 2. Security Scanner / Auditing Aid
**Target Accuracy: 85-90%+ on high-impact vulnerabilities**

**Key Focus Areas:**
- Reentrancy attacks
- Oracle manipulation
- Integer overflow/underflow
- Access control issues

**Performance Requirements:**
- **High Recall (Sensitivity):** Catch as many true vulnerabilities as possible
- **Reasonable Precision:** Minimize noise and false alarms for users

### 3. Autonomous Auto-Patch or Deployer Blocker
**Target Accuracy: 95%+ with very low false positives**

**Critical Requirements:**
- Extremely low false positive rate
- High confidence in vulnerability detection
- Minimal risk of blocking safe deployments

**Risk Mitigation:**
- Any wrong judgment can break the contract
- False negatives could allow critical bugs to reach mainnet
- False positives could block perfectly safe deployments

## Important Note
**The LLM will serve as a co-pilot, not a final decision maker.** Human oversight and validation remain essential for all security-critical decisions.

## Success Criteria
- Achieve target accuracy metrics for each use case
- Demonstrate reliable performance on common vulnerability patterns
- Maintain low false positive rates to ensure practical usability
- Provide clear, actionable vulnerability