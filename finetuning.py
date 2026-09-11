import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_DIR = "./processed_dataset"
OUTPUT_DIR = "./qwen_pharma_lora"

def run_finetuning():
    print("[+] Configuration de la quantification 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    print("[+] Chargement du modèle et du tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    print("[+] Configuration de LoRA...")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    dataset = load_from_disk(DATASET_DIR)

    sft_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=3,
        fp16=True,
        save_strategy="epoch",
        optim="paged_adamw_8bit",
        report_to="none",
        dataset_text_field="text"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        args=sft_args,
        max_seq_length=512
    )

    print("[+] Démarrage de l'entraînement...")
    trainer.train()

    print("[+] Sauvegarde de l'adaptateur LoRA...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[✓] Fine-tuning terminé avec succès. Adaptateur sauvegardé dans '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    run_finetuning()