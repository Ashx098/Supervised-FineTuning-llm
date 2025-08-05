# Gemma 27B QLoRA Fine-tuning for Structured JSON Output

This project demonstrates how to fine-tune the Google Gemma 2 27B model using **QLoRA (Quantized Low-Rank Adaptation)** with Hugging Face Transformers, PEFT, bitsandbytes, and TRL. The goal is to train the model to parse natural language queries and generate structured JSON responses, including multi-class/multi-label classification within that JSON.

## 🚀 Key Features

* **QLoRA Fine-tuning:** Efficiently fine-tunes Gemma 27B by quantizing the base model to 4-bit and training only lightweight LoRA adapters, drastically reducing VRAM usage.
* **Structured JSON Output:** Trains the model to generate a specific JSON format containing:
    * The original query.
    * A `Classification Router` (multi-class, derived from the input data's `type`).
    * A `Search Query Prompt` (a rephrased instruction for a downstream system).
    * The original `data` object, preserving its structure and any multi-label `filters.intent` information.
* **RTX 4090 Optimization:** Configured with `bfloat16` and `gradient_accumulation_steps` to leverage the 24GB VRAM of an RTX 4090 effectively.
* **Modular Design:** Separates concerns into distinct scripts for data preparation, training, and inference, with an external YAML configuration.
* **Hugging Face Ecosystem:** Leverages popular Hugging Face libraries (`transformers`, `peft`, `bitsandbytes`, `trl`, `datasets`) for streamlined development.
