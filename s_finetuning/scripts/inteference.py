import torch
import yaml
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from .constants import SYSTEM_PROMPT

def main():
    config_path = os.path.join(os.path.dirname(__file__), '../config/fine_tune_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_id = config["model_id"]
    output_dir = os.path.join(os.path.dirname(__file__), '..', config["output_dir"])

    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    print(f"Loading base model: {model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=compute_dtype, 
        device_map="auto",
    )

    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading PEFT adapters from: {output_dir}...")
    model_for_inference = PeftModel.from_pretrained(base_model, output_dir)

    model_to_use = model_for_inference

    print("Creating text generation pipeline...")
    pipe = pipeline(
        "text-generation",
        model=model_to_use,
        tokenizer=tokenizer,
        torch_dtype=compute_dtype,
        device_map="auto",
    )

    print("\n--- Testing Inference ---")
    test_queries = [
        "Hi there! How's your day going?",
        "Show me all emails from alice@example.com about project X from last month.",
        "What's the best way to cook pasta?",
        "Find all sales reports for Q2 2024.",
        "Can you help me plan a trip to Japan next spring?"
    ]

    for query in test_queries:
        prompt = f"{SYSTEM_PROMPT}\\n\\nUser Query: {query}\\n\\n### Assistant:"
        print(f"\\n--- Query ---\\n{query}")
        print(f"--- Prompt Sent to Model ---\\n{prompt}")

        stop_token_ids = [tokenizer.eos_token_id]

        outputs = pipe(
            prompt,
            max_new_tokens=512, 
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            eos_token_id=stop_token_ids,
            pad_token_id=tokenizer.pad_token_id,
        )

        generated_text = outputs[0]['generated_text']
        assistant_response_start = generated_text.find("### Assistant:") + len("### Assistant:")
        raw_json_output = generated_text[assistant_response_start:].strip()

        print(f"\\n--- Raw Generated Response ---\\n{raw_json_output}")

        try:
            parsed_json = json.loads(raw_json_output)
            print("\\n--- Parsed JSON Output (Success) ---")
            print(json.dumps(parsed_json, indent=2))
        except json.JSONDecodeError as e:
            print(f"\\n--- JSON Parsing Failed: {e} ---")
            print("The model did not generate valid JSON.")

if __name__ == "__main__":
    main()
