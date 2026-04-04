import json, urllib.request, time, sys, traceback

BASE = "http://localhost:8081"

def api_get(path):
    return json.loads(urllib.request.urlopen(BASE + path, timeout=10).read())

def api_post(path, data):
    req = urllib.request.Request(BASE + path, data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    resp = urllib.request.urlopen(req, timeout=180)
    return json.loads(resp.read())

models = api_get("/v1/models").get("data", [])
print("=== LOCAL MODELS (%d) ===" % len(models))
for m in models:
    print("  %-40s | %-10s | %8d MB | %s" % (m["id"], m.get("backend","?"), m.get("size_mb",0), m.get("path","")))

print("\n=== LOAD + CHAT TESTS ===")
results = []
for m in models:
    path = m.get("path", "")
    name = m.get("id", "")
    if not path:
        print("  SKIP  %s (no path)" % name)
        results.append((name, "SKIP", "no path"))
        continue
    try:
        t0 = time.time()
        r = api_post("/v1/models/load", {"model": path, "backend": "native"})
        dt = time.time() - t0
        if r.get("status") == "error":
            err = r.get("error", "unknown")
            print("  FAIL  %s -> %s" % (name, err[:80]))
            results.append((name, "FAIL", err[:80]))
            continue
    except Exception as e:
        print("  FAIL  %s -> %s" % (name, str(e)[:80]))
        results.append((name, "FAIL", str(e)[:80]))
        continue

    try:
        cr = api_post("/v1/chat/completions", {
            "model": "deepnetz",
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "max_tokens": 10, "temperature": 0.1,
        })
        reply = cr.get("choices", [{}])[0].get("message", {}).get("content", "")
        print("  OK    %s (%.1fs) -> %s" % (name, dt, repr(reply[:50])))
        results.append((name, "OK", reply[:50]))
    except Exception as e:
        print("  CHAT  %s loaded but chat failed: %s" % (name, str(e)[:80]))
        results.append((name, "CHAT_FAIL", str(e)[:80]))

    try:
        api_post("/v1/models/unload", {})
        time.sleep(2)
    except:
        pass

print("\n=== HUB SEARCH ===")
for q in ["qwen", "llama", "mistral", "gemma", ""]:
    cards = api_get("/v1/cards/search?q=%s&limit=3" % q).get("cards", [])
    names = [c["name"] for c in cards]
    label = q if q else "(all)"
    print("  %-10s -> %d cards: %s" % (label, len(cards), ", ".join(names)))

print("\n=== SUMMARY ===")
ok = sum(1 for _, s, _ in results if s == "OK")
fail = sum(1 for _, s, _ in results if s in ("FAIL", "CHAT_FAIL"))
print("  Total: %d | OK: %d | FAIL: %d" % (len(results), ok, fail))
for name, status, detail in results:
    if status != "OK":
        print("  %-10s %s: %s" % (status, name, detail))
