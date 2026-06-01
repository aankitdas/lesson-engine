from fastapi import FastAPI
from pydantic import BaseModel
from unsloth import FastLanguageModel
import uvicorn

app = FastAPI()

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="finetune/output/adapter",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
print("Model ready.")


class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]
    temperature: float = 0.2
    max_tokens: int = 2048


@app.post("/v1/chat/completions")
def chat(request: ChatRequest):
    inputs = tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in request.messages],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        do_sample=request.temperature > 0,
        repetition_penalty=1.1,
    )

    text = tokenizer.decode(
        outputs[0][inputs.shape[1]:],
        skip_special_tokens=True
    )

    return {
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {
            "prompt_tokens":     int(inputs.shape[1]),
            "completion_tokens": int(outputs.shape[1] - inputs.shape[1]),
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)