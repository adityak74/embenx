#!/bin/bash
set -e

# Usage: ./publish.sh [patch|minor|major]
TYPE=${1:-patch}

echo "🚀 Starting release process for type: $TYPE..."

# 1. Clean old distributions
echo "🧹 Cleaning old distributions..."
rm -rf dist/ build/ *.egg-info

# 2. Bump version (using bumpversion config in pyproject.toml)
# Note: If bumpversion isn't in your path, we use python -m bumpversion
echo "🔢 Bumping version..."
uv run bump-my-version bump $TYPE

# 3. Build the package
echo "📦 Building package..."
python3 -m build

# 4. Check distributions
echo "🔍 Checking distributions with twine..."
twine check dist/*

# 5. Upload to TestPyPI
echo "🧪 Uploading to TestPyPI..."
python3 -m twine upload --repository testpypi dist/*

echo "✅ TestPyPI upload complete!"
echo "🔗 View at: https://test.pypi.org/project/embenx/"

# 6. Confirmation for Production
read -p "🚀 Upload to PRODUCTION PyPI? (y/N) " CONFIRM
if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "🌟 Uploading to Production PyPI..."
    # Uses ~/.pypirc or environment variables
    python3 -m twine upload dist/*
    echo "🎉 Production release complete!"
    echo "🔗 View at: https://pypi.org/project/embenx/"
else
    echo "🛑 Production upload skipped."
fi
