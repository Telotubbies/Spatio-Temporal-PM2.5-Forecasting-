#!/bin/bash
# Prepare and push to GitHub repository
# Repository: Telotubbies/Spatio-Temporal-PM2.5-Forecasting-

set -e

cd "$(dirname "$0")"

echo "🚀 Preparing PM2.5 Forecasting Pipeline for GitHub"
echo "=================================================="

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "❌ Git not initialized. Run: git init"
    exit 1
fi

# Check current branch
CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "master")
echo "📌 Current branch: $CURRENT_BRANCH"

# Create feature branch if not exists
if [ "$CURRENT_BRANCH" != "feature/production-pipeline" ]; then
    echo "🌿 Creating feature branch..."
    git checkout -b feature/production-pipeline 2>/dev/null || git checkout feature/production-pipeline
fi

# Add all files (except venv, data, __pycache__)
echo "📦 Staging files..."
git add -A
git reset HEAD venv/ data/raw/ data/processed/ __pycache__/ .pytest_cache/ 2>/dev/null || true

# Show what will be committed
echo ""
echo "📋 Files to be committed:"
git status --short | head -30

# Create commit
echo ""
echo "💾 Creating commit..."
git commit -m "feat: Add production-ready PM2.5 forecasting pipeline

✨ Features:
- Multi-source data collection (Air4Thai, Open-Meteo, NASA FIRMS)
- Historical data from 2010 (16+ years coverage)
- AI Engineering standards (error handling, validation, logging)
- AMD GPU support (ROCm for 7800XT)
- Modular architecture with clean separation of concerns
- Comprehensive documentation and quick start guides

📊 Data Sources:
- PM2.5: Air4Thai API (82 stations in Bangkok)
- Weather: Open-Meteo Historical API (2010-present)
- Fire: NASA FIRMS (placeholder)
- Static: WorldCover, WorldPop (optional)

🏗️ Architecture:
- Custom exception hierarchy
- Data validation at boundaries
- Structured logging
- Type hints throughout
- Production-ready error handling

📚 Documentation:
- README.md - Full documentation
- QUICKSTART.md - Quick start guide
- AI_ENGINEERING_CHECKLIST.md - Standards compliance
- HISTORICAL_DATA.md - Historical data collection guide
- COMPARISON_WITH_REFERENCE.md - Comparison with reference repo

🔧 Technical:
- Python 3.11+
- Parquet storage (partitioned by year/month/station_id)
- ROCm support for AMD 7800XT
- Clean module separation (collectors, features, pipeline)" || echo "⚠️  No changes to commit"

echo ""
echo "✅ Commit created!"
echo ""
echo "📤 Next steps:"
echo "1. Push to GitHub:"
echo "   git push -u origin feature/production-pipeline"
echo ""
echo "2. Or use MCP GitHub (if credentials configured):"
echo "   See PUSH_INSTRUCTIONS.md"
echo ""
echo "3. Create Pull Request:"
echo "   Visit: https://github.com/Telotubbies/Spatio-Temporal-PM2.5-Forecasting-/compare/feature/production-pipeline"

