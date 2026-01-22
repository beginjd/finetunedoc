#!/bin/bash
# Script to initialize git repo and push to GitHub
# Usage: ./push_to_github.sh <github-repo-url>
# Example: ./push_to_github.sh https://github.com/username/cobol-compiler-finetuning.git

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <github-repo-url>"
    echo "Example: $0 https://github.com/username/cobol-compiler-finetuning.git"
    echo ""
    echo "Or if you want to create a new repo, first create it on GitHub, then run:"
    echo "  $0 https://github.com/username/repo-name.git"
    exit 1
fi

REPO_URL="$1"

echo "=== Setting up Git repository ==="
echo ""

# Initialize git if not already initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    echo "✓ Git repository initialized"
else
    echo "✓ Git repository already exists"
fi

# Check if remote already exists
if git remote get-url origin > /dev/null 2>&1; then
    echo "Remote 'origin' already exists. Updating..."
    git remote set-url origin "$REPO_URL"
else
    echo "Adding remote 'origin'..."
    git remote add origin "$REPO_URL"
fi
echo "✓ Remote configured: $REPO_URL"
echo ""

# Add files
echo "Adding files to git..."
git add runpod/
git add scripts/
git add pyproject.toml
git add .gitignore
git add README.md 2>/dev/null || true

# Check if data files should be added
if [ -f "data/cobol_dataset_train.jsonl" ]; then
    echo ""
    read -p "Add dataset files (data/*.jsonl) to git? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add data/*.jsonl
        echo "✓ Dataset files added"
    else
        echo "⊘ Dataset files skipped (they're in .gitignore)"
    fi
fi

echo ""
echo "Files staged. Current status:"
git status --short

echo ""
read -p "Commit and push to GitHub? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted. Run 'git commit' and 'git push' manually when ready."
    exit 0
fi

# Commit
echo ""
echo "Creating commit..."
git commit -m "Add RunPod fine-tuning setup and scripts

- Add QLoRA fine-tuning script for Mistral 7B
- Add RunPod startup script
- Add PDF extraction and dataset preparation scripts
- Add configuration files"

echo "✓ Commit created"
echo ""

# Push
echo "Pushing to GitHub..."
BRANCH=$(git branch --show-current 2>/dev/null || echo "main")

# Try to push, create branch if needed
if git push -u origin "$BRANCH" 2>&1 | grep -q "no upstream branch"; then
    echo "Creating branch '$BRANCH' on remote..."
    git push -u origin "$BRANCH"
else
    git push -u origin "$BRANCH"
fi

echo ""
echo "=== Success! ==="
echo "Repository pushed to: $REPO_URL"
echo ""
echo "Next steps:"
echo "1. Update RunPod start.sh to clone this repo:"
echo "   git clone $REPO_URL /workspace/repo"
echo "   cp -r /workspace/repo/runpod /workspace/"
echo "   cp -r /workspace/repo/data /workspace/"
