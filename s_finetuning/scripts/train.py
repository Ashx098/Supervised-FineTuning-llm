import torch
import yaml
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset
import json
from getpass import getpass

def load_single_sample_for_testing(processed_data_path):
    """Load only one data point for testing"""
    print("🧪 Loading single sample for testing...")
    
    if not os.path.exists(processed_data_path):
        print(f"❌ Error: {processed_data_path} does not exist.")
        return None
    
    try:
        with open(processed_data_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Failed to parse {processed_data_path}: {e}")
        return None
    
    if not data:
        print(f"❌ Error: {processed_data_path} is empty.")
        return None
    
    # Take only the first sample
    single_sample = [data[0]]
    
    # Create a temporary file with single sample
    test_data_path = processed_data_path.replace('.json', '_test_single.json')
    with open(test_data_path, 'w') as f:
        json.dump(single_sample, f, indent=2)
    
    print(f"✅ Created test dataset with 1 sample: {test_data_path}")
    return test_data_path

def main():
    # Prompt for Hugging Face token
    print("🔐 Please enter your Hugging Face token (input will be hidden):")
    hf_token = getpass("Hugging Face Token: ")
    if not hf_token:
        print("❌ Error: No Hugging Face token provided. Exiting.")
        return

    config_path = os.path.join(os.path.dirname(__file__), '../config/fine_tune_config.yaml')
    if not os.path.exists(config_path):
        print(f"❌ Error: Config file {config_path} does not exist.")
        return
    
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

    # Configuration
    model_id = config["model_id"]
    output_dir = os.path.join(os.path.dirname(__file__), '..', config["output_dir"])
    new_model_name = config["new_model_name"]

    bnb_config = BitsAndBytesConfig(**config["quantization_args"])
    lora_config = LoraConfig(**config["lora_args"])

    if isinstance(config["training_args"]["learning_rate"], str):
        config["training_args"]["learning_rate"] = float(config["training_args"]["learning_rate"])

    training_args = TrainingArguments(
        output_dir=output_dir,
        **config["training_args"]
    )

    # 🧪 Load single sample for testing
    processed_data_path = os.path.join(os.path.dirname(__file__), '..', config["dataset_config"]["processed_data_output"])
    test_data_path = load_single_sample_for_testing(processed_data_path)
    if test_data_path is None:
        print("❌ Failed to load test dataset. Exiting.")
        return
    
    max_seq_length = config["dataset_config"]["max_seq_length"]

    print(f"--- Configuration Loaded ---")
    print(f"Model ID: {model_id}")
    print(f"Output Directory: {output_dir}")
    print(f"Max Sequence Length: {max_seq_length}")
    print(f"🧪 TEST MODE: Using single sample")
    print(f"🚫 Push to Hub: DISABLED")
    print(f"----------------------------")

    print(f"Loading base model from Hugging Face: {model_id}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            token=hf_token,
        )
        print(f"✅ Model loaded successfully from {model_id}.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Ensure you have an internet connection and access to Hugging Face Hub.")
        print(f"💡 Check if '{model_id}' is accessible or if the provided token is valid.")
        return

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    print("\n📊 Trainable parameters after PEFT setup:")
    model.print_trainable_parameters()

    print(f"Loading tokenizer for {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # Gemma chat template
        chat_template = """{% for message in messages %}{% if message['role'] == 'user' %}<start_of_turn>user
{{ message['content'] }}<end_of_turn>
{% elif message['role'] == 'model' %}<start_of_turn>model
{{ message['content'] }}<end_of_turn>
{% else %}{{ raise_exception('Unknown role: ' ~ message['role']) }}{% endif %}{% endfor %}{% if add_generation_prompt %}<start_of_turn>model
{% endif %}"""
        tokenizer.chat_template = chat_template
        print("✅ Custom chat template applied to tokenizer.")
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return

    print(f"📂 Loading test dataset from: {test_data_path}...")
    try:
        dataset = load_dataset("json", data_files=test_data_path, split="train")
        print(f"📊 Dataset size: {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    print("🚀 Initializing SFTTrainer...")
    try:
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=lora_config,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            args=training_args,
            packing=True,
        )
    except Exception as e:
        print(f"❌ Error initializing SFTTrainer: {e}")
        return

    print("🎯 Starting training (TEST MODE - limited steps)...")
    try:
        trainer.train()
        print("✅ Training finished successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return

    print(f"💾 Saving model and tokenizer to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("✅ Fine-tuning test complete!")
    print(f"📁 Model saved to: {output_dir}")
    print("🧪 Test successful! You can now run with full dataset.")
    
    # Clean up test file
    if os.path.exists(test_data_path):
        os.remove(test_data_path)
        print("🧹 Cleaned up test data file")

if __name__ == "__main__":
    main()