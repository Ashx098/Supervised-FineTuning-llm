import torch
import yaml
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training

# Hugging Face token for accessing gated models
HF_TOKEN = os.environ.get("HF_TOKEN", "")

def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), '../config/fine_tune_config.yaml')
    if not os.path.exists(config_path):
        print(f"❌ Error: Config file {config_path} does not exist.")
        return None
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Handle dtype conversion
    if "quantization_args" in config and "bnb_4bit_compute_dtype" in config["quantization_args"]:
        dtype_str = config["quantization_args"]["bnb_4bit_compute_dtype"]
        if dtype_str == "bfloat16":
            config["quantization_args"]["bnb_4bit_compute_dtype"] = torch.bfloat16
        elif dtype_str == "float16":
            config["quantization_args"]["bnb_4bit_compute_dtype"] = torch.float16
        else:
            raise ValueError(f"Unsupported compute_dtype: {dtype_str}")

    return config

def format_chat_template(batch, tokenizer):
    """Format the data using Gemma chat template"""
    samples = []
    
    # Extract messages from the batch
    messages_list = batch["messages"]
    
    for messages in messages_list:
        # Apply the Gemma chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        samples.append(text)
    
    return {"text": samples}

def main():
    # Load configuration
    config = load_config()
    if config is None:
        return
    
    # Get paths and configuration
    model_id = config["model_id"]
    output_dir = os.path.join(os.path.dirname(__file__), '..', config["output_dir"])
    
    # Use the test single data file
    test_data_path = os.path.join(os.path.dirname(__file__), '../data/fine_tuning_data_test_single.json')
    
    print(f"🔧 Configuration:")
    print(f"Model ID: {model_id}")
    print(f"Output Directory: {output_dir}")
    print(f"Data File: {test_data_path}")
    print(f"🧪 TEST MODE: Using single sample")
    print(f"----------------------------")
    
    # Load dataset
    if not os.path.exists(test_data_path):
        print(f"❌ Error: Data file {test_data_path} does not exist.")
        return
    
    dataset = load_dataset("json", data_files=test_data_path, split='train')
    print(f"📊 Dataset loaded with {len(dataset)} samples")
    print(f"Sample data: {dataset[0]}")
    
    # Load tokenizer
    print(f"📝 Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        trust_remote_code=True,
        token=HF_TOKEN,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Set Gemma chat template
    chat_template = """{% for message in messages %}{% if message['role'] == 'user' %}<start_of_turn>user
{{ message['content'] }}<end_of_turn>
{% elif message['role'] == 'model' %}<start_of_turn>model
{{ message['content'] }}<end_of_turn>
{% else %}{{ raise_exception('Unknown role: ' ~ message['role']) }}{% endif %}{% endfor %}{% if add_generation_prompt %}<start_of_turn>model
{% endif %}"""
    tokenizer.chat_template = chat_template
    print("✅ Gemma chat template applied to tokenizer.")
    
    # Format dataset
    print("🔄 Formatting dataset with chat template...")
    train_dataset = dataset.map(
        lambda x: format_chat_template(x, tokenizer), 
        num_proc=4, 
        batched=True, 
        batch_size=1
    )
    print(f"Formatted sample: {train_dataset[0]['text'][:200]}...")
    
    # Load quantization config
    quant_config = BitsAndBytesConfig(**config["quantization_args"])
    
    # Load model
    print(f"🤖 Loading model: {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quant_config,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    print("✅ Model loaded successfully.")
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    # PEFT config
    peft_config = LoraConfig(**config["lora_args"])
    
    # Training arguments from config
    training_args = config["training_args"].copy()
    if isinstance(training_args["learning_rate"], str):
        training_args["learning_rate"] = float(training_args["learning_rate"])
    
    # Create SFT config
    sft_config = SFTConfig(
        output_dir=output_dir,
        **training_args
    )
    
    # Initialize trainer
    print("🚀 Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=sft_config,
        peft_config=peft_config,
        max_seq_length=config["dataset_config"]["max_seq_length"],
        tokenizer=tokenizer,
        packing=True,
    )
    
    print("🎯 Starting training...")
    trainer.train()
    
    print(f"💾 Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    trainer.model.save_pretrained(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(output_dir)
    
    print("✅ Training completed successfully!")
    print(f"📁 Model saved to: {output_dir}")
    print("🧪 Test training complete!")

if __name__ == "__main__":
    main()
