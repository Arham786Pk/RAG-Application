from llama_cpp import Llama
import sys
import os
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("langchain").setLevel(logging.ERROR)

sys.stderr = open(os.devnull, "w")

llm = Llama(
    model_path="D:\\Models\\Tiny-LLama\\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    n_ctx=4096,
    n_threads=4,
    n_gpu_layers=0,
    use_mmap=True,
    use_mlock=False,
    verbose=False
)

response = llm(
    "Q: What is the capital of France?\nA:",
    max_tokens=50,
    stop=["\n"]
)

print(response["choices"][0]["text"].strip())
