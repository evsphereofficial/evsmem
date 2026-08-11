from llama_cpp import Llama
from pathlib import Path

MODEL = r"D:\Programming\AiProjects\EvAgent\evsmem\models\LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
INPUT = Path("input.txt")

llm = Llama(
    model_path=MODEL,
    n_ctx=128192,
    n_gpu_layers=-1,
    verbose=False,
)

print("Model loaded.")
print("Put your prompt in input.txt and save it.")
print("Press ENTER here when ready.")
print("Press Ctrl+C to quit.\n")

while True:
    input("Press ENTER to run input.txt... ")

    prompt = INPUT.read_text(encoding="utf-8")

    if not prompt.strip():
        print("input.txt is empty.")
        continue

    print("\nGenerating...\n")

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.3,
        max_tokens=2048,
    )

    answer = response["choices"][0]["message"]["content"]

    print("\n" + "=" * 80)
    print(answer)
    print("=" * 80 + "\n")