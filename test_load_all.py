import os, sys, traceback, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

models_dir = r"D:\models"
files = []
for root, dirs, fnames in os.walk(models_dir):
    for f in fnames:
        if f.endswith(".gguf") and "mmproj" not in f.lower():
            files.append(os.path.join(root, f))

print("Found %d GGUF files" % len(files))

from llama_cpp import Llama

results = []
for path in files:
    name = os.path.basename(path)
    size_mb = os.path.getsize(path) // (1024*1024)
    print("\nTesting: %s (%d MB)" % (name, size_mb))
    try:
        llm = Llama(model_path=path, n_ctx=512, n_gpu_layers=0, verbose=False)
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": "Say hi"}],
            max_tokens=5,
        )
        reply = out["choices"][0]["message"]["content"]
        print("  OK -> %s" % repr(reply[:50]))
        results.append((name, "OK", reply[:30]))
        del llm
    except Exception as e:
        msg = str(e)[:80]
        print("  FAIL -> %s" % msg)
        results.append((name, "FAIL", msg))

print("\n\n========== FULL REPORT ==========")
ok = [r for r in results if r[1] == "OK"]
fail = [r for r in results if r[1] == "FAIL"]
print("OK: %d / %d" % (len(ok), len(results)))
print("\nWorking models:")
for name, _, reply in ok:
    print("  + %s -> %s" % (name, repr(reply)))
print("\nFailed models:")
for name, _, err in fail:
    print("  - %s -> %s" % (name, err))
