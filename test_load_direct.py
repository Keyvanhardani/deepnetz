import os, sys, traceback

models_dir = r"D:\models"
files = []
for root, dirs, fnames in os.walk(models_dir):
    for f in fnames:
        if f.endswith(".gguf") and "mmproj" not in f.lower():
            files.append(os.path.join(root, f))

print("Found %d GGUF files (excluding mmproj)" % len(files))

# Test with llama_cpp directly
from llama_cpp import Llama

for path in files[:5]:  # test first 5
    name = os.path.basename(path)
    size_mb = os.path.getsize(path) // (1024*1024)
    print("\nTesting: %s (%d MB)" % (name, size_mb))
    try:
        llm = Llama(model_path=path, n_ctx=512, n_gpu_layers=0, verbose=False)
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )
        reply = out["choices"][0]["message"]["content"]
        print("  OK -> %s" % repr(reply[:50]))
        del llm
    except Exception as e:
        print("  FAIL -> %s" % str(e)[:100])
        traceback.print_exc()
