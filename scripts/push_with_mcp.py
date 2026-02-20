#!/usr/bin/env python3
"""
Script to prepare files for MCP GitHub push.
"""
import os
from pathlib import Path

def get_files_to_push():
    """Get all files to push."""
    files = []
    exclude_dirs = {'venv', '__pycache__', '.git', 'data', 'logs', '.ipynb_checkpoints', 'node_modules'}
    exclude_files = {'.DS_Store', 'pipeline.pid', 'pipeline_output.log'}
    
    for root, dirs, filenames in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for filename in filenames:
            if filename in exclude_files:
                continue
                
            filepath = Path(root) / filename
            rel_path = filepath.relative_to('.')
            
            # Only include relevant files
            if any(filepath.suffix in ext for ext in ['.py', '.md', '.txt', '.sh', '.ipynb', '.yaml', '.yml', '.json', '.gitignore']):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    files.append({
                        'path': str(rel_path),
                        'content': content
                    })
                except Exception as e:
                    print(f"⚠️  Skipping {rel_path}: {e}")
    
    return files

if __name__ == "__main__":
    files = get_files_to_push()
    print(f"📦 Total files: {len(files)}")
    print("\n📋 Files:")
    for f in files[:30]:
        print(f"   {f['path']}")
    if len(files) > 30:
        print(f"   ... and {len(files) - 30} more")
