# 1. Fine tuning 

import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments, TextStreamer
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported

def fine_tune_model():
    # Load model
    max_seq_length = 4096  # Increased for longer conversations
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )
    
    # Prepare model for PEFT
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
        use_rslora=True,
        use_gradient_checkpointing="unsloth"
    )
    print(model.print_trainable_parameters())
    
    # Set Qwen2.5 chat template
    QWEN_CHAT_TEMPLATE = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = QWEN_CHAT_TEMPLATE
    
    # Load and prepare preference dataset for SFT
    import json
    from datasets import Dataset
    
    def load_preference_dataset_for_sft(file_path):
        """Load preference dataset and convert to SFT format using chosen responses"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                # Create conversation format from prompt and chosen response
                conversation = [
                    {"role": "user", "content": item["prompt"]},
                    {"role": "assistant", "content": item["chosen"]}
                ]
                data.append({"messages": conversation})
        return Dataset.from_list(data)
    
    def apply_template(examples):
        messages = examples["messages"]
        text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
        return {"text": text}
    
    print("Loading preference dataset for SFT...")
    dataset = load_preference_dataset_for_sft("preference_pairs_with_placeholders.jsonl")
    
    # Filter for high quality examples
    print("Filtering for high-quality examples...")
    # We'll take a subset for SFT to avoid overfitting
    dataset = dataset.select(range(min(10000, len(dataset))))  # Max 10k examples for SFT
    
    print(f"Using {len(dataset)} examples for SFT")
    dataset = dataset.map(apply_template, batched=True)
    
    trainer=SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=True,
        args=TrainingArguments(
            learning_rate=2e-5,  # Lower LR for instruct model fine-tuning
            lr_scheduler_type="cosine",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,  # Increased for better gradients
            num_train_epochs=1,
            max_steps=500,  # Limit steps to avoid overfitting
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            warmup_ratio=0.1,
            output_dir="sft_output",
            save_steps=100,
            save_total_limit=3,
            seed=42,
            report_to="tensorboard",
        ),
    )
    
    #@title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    
    trainer_stats = trainer.train()
    
    #@title Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory         /max_memory*100, 3)
    lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
    
    # Load model for inference
    model = FastLanguageModel.for_inference(model)
    
    messages = [
        {"from": "human", "value": "Is 9.11 larger than 9.9?"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    
    text_streamer = TextStreamer(tokenizer)
    _ = model.generate(input_ids=inputs, streamer=text_streamer, max_new_tokens=512, use_cache=True)
    
    # Save the SFT model for DPO training
    print("Saving SFT model...")
    model.save_pretrained_merged("Qwen2.5-7B-Instruct-SFT", tokenizer, save_method="merged_16bit")
    print("SFT model saved as 'Qwen2.5-7B-Instruct-SFT'")
    print("Ready for DPO training! Run: python src/dpo_training.py")
    
    # Uncomment to push to HuggingFace Hub
    # model.push_to_hub_merged("your_username/Qwen2.5-7B-Instruct-SFT", tokenizer, save_method="merged_16bit", token = "your_huggingface_token")
    
if __name__ == "__main__":
    fine_tune_model()
    
    