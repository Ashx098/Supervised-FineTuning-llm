# Project Overview: Fine-Tuning Gemma for Structured JSON Output

## 1. Project Goal

The primary objective of this project is to fine-tune a large language model (e.g., `google/gemma-3-27b-it`) to function as a sophisticated query analysis engine. When given a user's query, the model is trained to classify it, extract key information, and output a structured JSON object that conforms to a strict, predefined schema.

This process transforms a general-purpose language model into a specialized, reliable tool that can understand user intent and provide machine-readable output for downstream processing.

---

## 2. The Fine-Tuning Pipeline: How It Works

The project is structured as a robust, three-stage pipeline:

**Stage 1: Data Formatting (`prepare_data.py`)**
Your source data (`data/raw_data.json`) is a high-quality, hand-labeled dataset where each query is paired with the perfect JSON output. This script performs one simple but essential task: it converts your dataset into the specific `messages` chat format that the `SFTTrainer` requires for fine-tuning. It combines your query and its corresponding JSON into a single training example, prefixed with the master `SYSTEM_PROMPT` imported from `constants.py`.

**Stage 2: Fine-Tuning (`train.py`)**
This script takes the powerful base Gemma model and trains it on your formatted dataset. Using a memory-efficient technique called QLoRA, it teaches the model to mimic the patterns in your data. Specifically, it learns that when it sees the `SYSTEM_PROMPT` followed by a user query, it should generate a JSON object that perfectly matches the hand-labeled examples you provided.

**Stage 3: Inference (`inteference.py`)**
This script is for testing the final, fine-tuned model. It loads the trained model and presents it with new, unseen user queries. Crucially, it uses the **exact same `SYSTEM_PROMPT`** imported from `constants.py`, ensuring perfect consistency between training and testing.

---

## 3. File Breakdown

- **`experiments/finetuning/scripts/constants.py`**:
  - **Purpose**: To ensure prompt consistency.
  - **Function**: This file contains the master `SYSTEM_PROMPT` as a single source of truth. Centralizing the prompt here prevents accidental mismatches between the training and inference scripts, which is critical for model performance.

- **`experiments/finetuning/scripts/prepare_data.py`**:
  - **Purpose**: To format your existing dataset for training.
  - **Function**: Imports the `SYSTEM_PROMPT` from `constants.py`. It then reads the `query` and `data` pairs from `data/raw_data.json` and converts them into the `messages` chat format required by the trainer, saving the result to `data/fine_tuning_data.jsonl`.

- **`experiments/finetuning/scripts/train.py`**:
  - **Purpose**: To fine-tune the Gemma model.
  - **Function**: Loads the base model, quantizes it for memory efficiency, and then uses the `SFTTrainer` to train it on your formatted data. It saves the final, trained model adapters to the `fine_tuned_model/` directory.

- **`experiments/finetuning/scripts/inteference.py`**:
  - **Purpose**: To test the fine-tuned model.
  - **Function**: Imports the `SYSTEM_PROMPT` from `constants.py`. It loads the base model, merges it with the trained adapters from `fine_tuned_model/`, and runs test queries to demonstrate performance.

- **`experiments/finetuning/config/fine_tune_config.yaml`**:
  - **Purpose**: To manage all project settings.
  - **Function**: This file contains all important parameters, including the model ID, file paths, and QLoRA/training configurations. It is perfectly structured and optimized for an RTX 4090.

- **`requirements.txt`**:
  - **Purpose**: To define all necessary Python packages.
  - **Function**: Ensures a reproducible environment by listing all dependencies.

- **`data/raw_data.json`**:
  - **Purpose**: The source of your high-quality, hand-labeled training data.
  - **Function**: This file contains the 1000+ examples of user queries and their corresponding perfect JSON outputs that the model will be trained on.

---

## 4. How to Run the Project

Execute the scripts in the following order from your terminal:

1.  **Install Dependencies**:
    ```bash
    pip install -r experiments/finetuning/requirements.txt
    ```

2.  **Format the Data**:
    ```bash
    python experiments/finetuning/scripts/prepare_data.py
    ```
    *This will convert your `raw_data.json` into the `fine_tuning_data.jsonl` file required for training.*

3.  **Run the Fine-Tuning**:
    ```bash
    python experiments/finetuning/scripts/train.py
    ```
    *This will start the training process and save the resulting model in the `fine_tuned_model/` directory. This will take a significant amount of time.*

4.  **Test the Model**:
    ```bash
    python experiments/finetuning/scripts/inteference.py
    ```
    *This will load your newly fine-tuned model and show you its performance on a set of test queries.*

---

## 5. How the Model Learns

This project uses **instruction fine-tuning**. The model learns to replicate the patterns in your hand-labeled data.

- **The Role of the `SYSTEM_PROMPT`**: The prompt, defined in `constants.py`, is the master instruction. During training, every example is prefixed with this prompt. The model learns that "when I see this block of text, my job is to convert the user query that follows into the JSON format I have seen in the examples."

- **The Role of Your Data**: Your `raw_data.json` is the "teacher." The model's neural network adjusts its weights to minimize the difference between its own generated JSON and the "correct" JSON you provided. After seeing 1000+ high-quality examples, it becomes an expert at this specific task.

---

## 6. Expected Outcome and Success Rate

- **Expected Outcome**: The fine-tuned model should be highly proficient at generating a JSON object that correctly classifies the query and extracts relevant information, as defined by your hand-labeled examples.

- **Success Rate**: With a high-quality, consistent dataset of 1000+ examples, the success rate should be **very high** for queries that are similar to those in the training set. The model's performance is a direct reflection of the quality of your data.

- **How to Improve**: If the model makes mistakes on certain types of queries, the best way to improve it is to add more examples of those specific cases to your `raw_data.json` and retrain.

---

## 7. Important Notes

- **Prompt Consistency is Guaranteed**: By centralizing the `SYSTEM_PROMPT` in `constants.py`, we have eliminated the risk of a mismatch between training and inference, ensuring robust and reliable model performance.
- **Hardware**: This project is configured for a high-end GPU (like an RTX 4090) with at least 24GB of VRAM. The use of 4-bit quantization and gradient checkpointing is essential.
- **Data Quality is Key**: The success of this project hinges on the quality and consistency of your hand-labeled `raw_data.json` file.
