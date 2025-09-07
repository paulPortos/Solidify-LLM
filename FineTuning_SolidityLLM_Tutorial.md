# Comprehensive Guide: Fine-Tuning a Qwen LLM for Solidity Vulnerability Detection

## Table of Contents
1. Introduction
2. Environment Setup
3. Model and Tokenizer Configuration
4. BitsAndBytes (bnb) Quantization
5. LoRA (Low-Rank Adaptation) Configuration
6. Dataset Preparation and `format_prompt`
7. Tokenization and Data Pipeline
8. TrainingArguments Explained
9. Trainer Setup and Training Loop
10. Model Saving and Testing
11. Hardware Tips (4GB/6GB/8GB VRAM)

---

## 1. Introduction
This guide walks you through fine-tuning a Qwen LLM (Qwen2.5-Coder-0.5B) for Solidity smart contract vulnerability detection using your IIO (Instruction, Input, Output) dataset. It explains every config and function in your code, including quantization, LoRA, and prompt formatting.

---

## 2. Environment Setup
- **Python 3.8+**
- **PyTorch** (with CUDA for GPU)
- **transformers, peft, datasets, huggingface_hub, python-dotenv, bitsandbytes**
- Activate your virtual environment and install requirements:
  ```bash
  pip install -r requirements.txt
  ```
- Place your Hugging Face token in a `.env` file:
  ```env
  HUGGINGFACE_API_KEY=hf_xxx
  ```

---

## 3. Model and Tokenizer Configuration
```python
MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```
- **MODEL_NAME**: The base model to fine-tune.
- **Tokenizer**: Handles text-to-token conversion, padding, and special tokens.

---

## 4. BitsAndBytes (bnb) Quantization
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
```
- **4-bit quantization**: Reduces VRAM usage, enabling larger models on smaller GPUs.
- **Double quantization**: Further compresses weights.
- **nf4**: NormalFloat4, a quantization type for better accuracy.
- **torch.float16**: Computation in half-precision for speed and memory savings.

---

## 5. LoRA (Low-Rank Adaptation) Configuration
```python
lora_config = LoraConfig(
    r=16,              # LoRA rank (lower for less VRAM)
    lora_alpha=32,     # Scaling factor (usually 2x rank)
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,  # Dropout for regularization
    bias="none",
    task_type="CAUSAL_LM"
)
```
- **r**: Controls LoRA adapter capacity (higher = more learning, more VRAM).
- **lora_alpha**: Scales LoRA updates (usually 2x r).
- **target_modules**: Which layers to adapt.
- **lora_dropout**: Prevents overfitting.

---

## 6. Dataset Preparation and `format_prompt`
Your dataset uses the IIO format:
```json
{"instruction": "Fix the bug...", "input": "contract code...", "output": "fixed code and explanation..."}
```
**Prompt formatting:**
```python
def format_prompt(example):
    instruction = example.get('instruction', '').strip()
    input_code = example.get('input', '').strip()
    expected_output = example.get('output', '').strip()
    if input_code:
        full_instruction = f"{instruction}\n\n```solidity\n{input_code}\n```"
    else:
        full_instruction = instruction
    if not instruction and not input_code:
        full_instruction = "Analyze this Solidity code for security vulnerabilities."
    if not expected_output:
        expected_output = "No specific vulnerabilities identified in this code."
    messages = [
        {"role": "user", "content": full_instruction},
        {"role": "assistant", "content": expected_output}
    ]
    return {"messages": messages}
```
- **Purpose**: Teaches the LLM to map instructions + code to a solution/explanation.

---

## 7. Tokenization and Data Pipeline
```python
formatted_dataset = dataset.map(format_prompt, remove_columns=dataset["train"].column_names)
tokenized_dataset = formatted_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=formatted_dataset["train"].column_names
)
```
- **Tokenization**: Converts formatted prompts to model-ready tokens.
- **Batched mapping**: Efficiently processes the dataset.

---

## 8. TrainingArguments Explained
```python
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=20,                # Number of full passes over the dataset
    per_device_train_batch_size=1,      # Small for 4GB VRAM
    gradient_accumulation_steps=8,      # Simulates larger batch size
    optim="paged_adamw_8bit",
    save_steps=250,                     # Frequent checkpoints
    logging_steps=10,                   # Frequent logs
    learning_rate=1e-4,                 # Stable learning
    weight_decay=0.01,                  # Regularization
    fp16=True,                          # Memory efficiency
    bf16=False,
    max_grad_norm=0.3,                  # Prevents exploding gradients
    max_steps=-1,
    warmup_ratio=0.1,                   # Gradual LR increase
    group_by_length=True,
    lr_scheduler_type="cosine",        # Smooth LR decay
    report_to="tensorboard",
    save_total_limit=3,
    load_best_model_at_end=False,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    remove_unused_columns=True,
    dataloader_drop_last=True,
)
```
- **Key settings**: batch size, epochs, learning rate, fp16, gradient accumulation, checkpointing, logging, scheduler.

---

## 9. Trainer Setup and Training Loop
```python
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)
trainer.train()
```
- **Trainer**: Handles batching, optimization, checkpointing, and logging.
- **Data collator**: Prepares batches for language modeling.

---

## 10. Model Saving and Testing
```python
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)
# ...
# Load model and run a test prompt to verify
```
- **Saves**: Final model and tokenizer for later use.
- **Testing**: Run a sample prompt to check model quality.

---

## 11. Hardware Tips (4GB/6GB/8GB VRAM)
- **4GB VRAM**: Use batch size 1, max_seq_length 512–800, LoRA r=16, fp16=True.
- **6GB VRAM**: Batch size 2, max_seq_length 896, LoRA r=32, fp16=True.
- **8GB VRAM**: Batch size 3, max_seq_length 1024, LoRA r=32–64, fp16=True.
- **Always monitor VRAM usage** with `nvidia-smi`.

---

## 12. Troubleshooting and Best Practices
- **OOM errors**: Lower batch size, max_seq_length, or LoRA rank.
- **Slow training**: Use fp16, increase batch size if possible.
- **Loss not decreasing**: Lower learning rate, check data formatting.
- **Model not learning vulnerabilities**: Ensure dataset is diverse and well-formatted.
- **Heat issues**: Train in a cool environment, clean GPU fans, use fp16.

---

**This guide should help you understand and optimize every part of your fine-tuning pipeline for Solidity vulnerability detection!**
