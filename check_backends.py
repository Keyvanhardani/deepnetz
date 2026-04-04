from deepnetz.backends.discovery import discover_backends
bs = discover_backends()
for b in bs:
    d = b.detect()
    print("%s: available=%s, version=%s, models=%d" % (b.name, d.available, d.version, d.models_count))
