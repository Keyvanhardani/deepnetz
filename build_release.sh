#!/bin/bash
# Build DeepNetz release — compiled .so files, no Python source.
#
# Usage: ./build_release.sh
# Output: dist/deepnetz-*.whl (pip installable, no source)

set -e

echo "=== DeepNetz Release Build ==="
echo ""

# 1. Clean
echo "[1/5] Cleaning..."
rm -rf build/ dist/ *.egg-info deepnetz/*.c deepnetz/engine/*.c
find . -name "*.so" -delete
find . -name "*.pyd" -delete

# 2. Compile with Cython
echo "[2/5] Compiling with Cython..."
python3 setup.py build_ext --inplace 2>&1 | grep -E "Compiling|Building|running"

# 3. Verify .so files exist
echo "[3/5] Verifying compiled modules..."
SO_COUNT=$(find deepnetz -name "*.so" | wc -l)
echo "  Found $SO_COUNT compiled modules"

if [ "$SO_COUNT" -lt 5 ]; then
    echo "  ERROR: Expected at least 5 .so files!"
    exit 1
fi

# 4. Build wheel (without source .py files)
echo "[4/5] Building wheel..."

# Create a temporary MANIFEST.in that excludes source
cat > MANIFEST.in << 'EOF'
recursive-include deepnetz *.so *.pyd
include deepnetz/__init__.py
include deepnetz/cache/__init__.py
include deepnetz/offload/__init__.py
include deepnetz/engine/__init__.py
recursive-exclude deepnetz *.py
recursive-exclude deepnetz *.c
include README.md
include pyproject.toml
EOF

python3 -m build --wheel 2>&1 | grep -E "Successfully|wheel"

# 5. Verify wheel contents
echo "[5/5] Checking wheel contents..."
WHL=$(ls dist/*.whl 2>/dev/null | head -1)
if [ -n "$WHL" ]; then
    echo "  Wheel: $WHL"
    echo "  Contents (should have .so, NOT .py for modules):"
    python3 -c "
import zipfile, sys
with zipfile.ZipFile('$WHL') as z:
    for name in sorted(z.namelist()):
        if 'deepnetz' in name:
            print(f'    {name}')
" 2>/dev/null
fi

# Clean intermediate .c files
rm -f deepnetz/*.c deepnetz/engine/*.c MANIFEST.in

echo ""
echo "=== Done! ==="
echo "Install with: pip install $WHL"
echo ""
