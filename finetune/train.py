import json
from pathlib import Path
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments

# ── Config ─────────────────────────────────────────────────────────────────

MODEL_NAME     = "unsloth/Phi-3-mini-4k-instruct"
MAX_SEQ_LEN    = 2048
LOAD_IN_4BIT   = True

LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

TRAIN_FILE     = Path("finetune/data/train.jsonl")
EVAL_FILE      = Path("finetune/data/eval.jsonl")
OUTPUT_DIR     = Path("finetune/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS         = 3
BATCH_SIZE     = 2
GRAD_ACCUM     = 4      # effective batch = 8
LR             = 2e-4
WARMUP_STEPS   = 20
SAVE_STEPS     = 100


# ── Load model ─────────────────────────────────────────────────────────────

def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=LOAD_IN_4BIT,
        dtype=None,  # auto
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    return model, tokenizer


# ── Load dataset ───────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def format_example(example: dict, tokenizer) -> str:
    """Convert ChatML dict to Phi-3 chat template string."""
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )


def build_dataset(path: Path, tokenizer) -> Dataset:
    raw = load_jsonl(path)
    texts = [format_example(ex, tokenizer) for ex in raw]
    return Dataset.from_dict({"text": texts})


# ── Train ──────────────────────────────────────────────────────────────────

def train():
    print("Loading model...")
    model, tokenizer = load_model()

    print("Loading dataset...")
    train_ds = build_dataset(TRAIN_FILE, tokenizer)
    eval_ds  = build_dataset(EVAL_FILE, tokenizer)
    print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=SFTConfig(
            output_dir=str(OUTPUT_DIR),
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LR,
            warmup_steps=WARMUP_STEPS,
            save_steps=SAVE_STEPS,
            eval_strategy="steps",
            eval_steps=SAVE_STEPS,
            logging_steps=10,
            fp16=False,
            bf16=True,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            report_to="none",
            load_best_model_at_end=True,
        ),
    )

    print("Training...")
    trainer.train()

    print("Saving adapter...")
    model.save_pretrained(str(OUTPUT_DIR / "adapter"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "adapter"))
    print(f"Done. Adapter saved to {OUTPUT_DIR / 'adapter'}")


if __name__ == "__main__":
    train()