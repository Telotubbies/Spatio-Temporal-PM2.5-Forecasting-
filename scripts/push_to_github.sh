#!/bin/bash
# Push to GitHub using git CLI
# Repository: Telotubbies/Spatio-Temporal-PM2.5-Forecasting-

set -e

cd "$(dirname "$0")"

echo "🚀 Pushing to GitHub"
echo "===================="

# Check branch
CURRENT_BRANCH=$(git branch --show-current)
echo "📌 Current branch: $CURRENT_BRANCH"

# Check remote
if ! git remote | grep -q origin; then
    echo "❌ No remote 'origin' found"
    echo "   Run: git remote add origin https://github.com/Telotubbies/Spatio-Temporal-PM2.5-Forecasting-.git"
    exit 1
fi

REMOTE_URL=$(git remote get-url origin)
echo "📡 Remote: $REMOTE_URL"

# Check if branch exists on remote
if git ls-remote --heads origin "$CURRENT_BRANCH" | grep -q "$CURRENT_BRANCH"; then
    echo "✅ Branch exists on remote"
    echo "📤 Pushing updates..."
    git push origin "$CURRENT_BRANCH"
else
    echo "🌿 Creating new branch on remote..."
    echo "📤 Pushing branch..."
    git push -u origin "$CURRENT_BRANCH"
fi

echo ""
echo "✅ Push completed!"
echo ""
echo "🔗 Create Pull Request:"
echo "   https://github.com/Telotubbies/Spatio-Temporal-PM2.5-Forecasting-/compare/$CURRENT_BRANCH"

