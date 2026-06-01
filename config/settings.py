from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

GEMINI_MODEL    = "gemini-2.5-flash"
DEEPSEEK_MODEL  = "deepseek-chat"

DB_PATH         = "chalk.db"
VECTOR_STORE_PATH = "vectorstore/index.faiss"

# Pricing per 1M tokens — update here when rates change
PRICING = {
    "gemini-2.5-flash": {
        "input":  0.15,
        "output": 0.60,
    },
    "deepseek-chat": {
        "input":  0.27,
        "output": 1.10,
    },
}

def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICING.get(model, {"input": 0.0, "output": 0.0})
    return round(
        (input_tokens * p["input"] / 1_000_000) +
        (output_tokens * p["output"] / 1_000_000),
        6
    )