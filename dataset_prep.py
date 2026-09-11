import json
from datasets import Dataset
from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

def prepare_dataset(input_jsonl="finetuning_dataset.jsonl", output_dir="./processed_dataset"):
    print("[+] Chargement et préparation du dataset...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # 1. Lecture du JSONL
    data = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                
    # 2. Conversion en HuggingFace Dataset
    hf_dataset = Dataset.from_list(data)
    
    # 3. Formatage des messages avec le Chat Template de Qwen
    def apply_template(example):
        text = tokenizer.apply_chat_template(
            example["messages"], 
            tokenize=False, 
            add_generation_prompt=False
        )
        return {"text": text}
    
    formatted_dataset = hf_dataset.map(apply_template, remove_columns=["messages"])
    
    # 4. Split Train / Validation (90% / 10%)
    split_ds = formatted_dataset.train_test_split(test_size=0.1, seed=42)
    split_ds.save_to_disk(output_dir)
    
    print(f"[✓] Dataset préparé avec succès ({len(split_ds['train'])} train, {len(split_ds['test'])} val). Sauvegardé dans '{output_dir}'.")

if __name__ == "__main__":
    prepare_dataset()