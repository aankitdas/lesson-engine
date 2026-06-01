from unsloth import FastLanguageModel

print("Loading adapter...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="finetune/output/adapter",
    max_seq_length=2048,
    load_in_4bit=True,
)

print("Saving as GGUF (Q4_K_M)...")
model.save_pretrained_gguf(
    "finetune/output/gguf",
    tokenizer,
    quantization_method="q8_0",
)
print("Done → finetune/output/gguf/")