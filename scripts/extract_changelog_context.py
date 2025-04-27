#!/usr/bin/env python3
"""
Script to extract repository context for generating changelog entries.
Gathers README content, module docstrings, project structure, and changelog history.
"""
import os
import ast
import re
import sys
import json
from pathlib import Path


def extract_module_info():
    """Extract docstrings from Python modules"""
    module_info = {}
    python_files = []
    
    # Find all Python files
    for root, _, files in os.walk('.'):
        if "__pycache__" in root:
            continue
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                python_files.append(filepath)
    
    # Extract docstrings
    for filepath in sorted(python_files):
        try:
            with open(filepath, "r", encoding="utf-8") as mf:
                module_content = mf.read()
            try:
                tree = ast.parse(module_content)
                docstring = ast.get_docstring(tree)
                if docstring:
                    module_info[filepath] = docstring.strip()
            except SyntaxError:
                continue
        except Exception:
            continue
    
    # Write to file
    with open("module_info.txt", "w", encoding="utf-8") as f:
        for module, doc in module_info.items():
            if len(doc) > 100:  # Truncate long docstrings
                doc = doc[:100] + "..."
            f.write(f"{module}:\n{doc}\n\n")


def extract_project_structure():
    """Extract project directory structure"""
    project_structure = []
    
    # Read gitignore patterns if available
    gitignore_patterns = []
    if os.path.exists('.gitignore'):
        with open('.gitignore', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    gitignore_patterns.append(line)
    
    def is_ignored(path):
        """Check if a path matches any gitignore pattern"""
        # Convert Windows paths to forward slashes for consistent matching
        path = path.replace('\\', '/')
        if path.startswith('./'):
            path = path[2:]
            
        for pattern in gitignore_patterns:
            # Handle directory-specific patterns (ending with /)
            if pattern.endswith('/'):
                if path.startswith(pattern) or path.startswith(f"./{pattern}"):
                    return True
            # Handle file patterns with wildcards
            elif '*' in pattern:
                parts = pattern.split('*')
                if len(parts) == 2:
                    if path.startswith(parts[0]) and path.endswith(parts[1]):
                        return True
            # Handle exact matches
            elif path == pattern or path.endswith(f"/{pattern}"):
                return True
        return False
    
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories, venvs and cache dirs
        dirs[:] = [d for d in dirs if not d.startswith('.') and 
                  d != 'venv' and d != '__pycache__' and not is_ignored(os.path.join(root, d))]
        
        if root != '.' and not is_ignored(root):
            project_structure.append(root)
    
    # Write to file
    with open("project_structure.txt", "w", encoding="utf-8") as f:
        for directory in sorted(project_structure):
            f.write(f"{directory}\n")


def extract_changelog_history():
    """Extract past changelog entries"""
    if not os.path.exists('CHANGELOG.md'):
        with open("changelog_history.txt", "w") as f:
            f.write("No CHANGELOG.md found")
        return
    
    try:
        # Read the full changelog
        with open("CHANGELOG.md", "r") as f:
            content = f.read()
        
        # Find all version headers
        version_headers = list(re.finditer(r"## \[\d+\.\d+\.\d+\]", content))
        
        # Extract the latest 3 complete entries (or all if fewer)
        entries_to_keep = min(3, len(version_headers))
        
        if entries_to_keep > 0:
            # Get the position of the last header we want to include
            if entries_to_keep < len(version_headers):
                end_pos = version_headers[entries_to_keep].start()
            else:
                end_pos = len(content)
                
            # Get content from beginning to the end position
            changelog_sample = content[:end_pos].strip()
            
            with open("changelog_history.txt", "w") as f:
                f.write(changelog_sample)
        else:
            with open("changelog_history.txt", "w") as f:
                f.write("No changelog entries found")
    except Exception as e:
        with open("changelog_history.txt", "w") as f:
            f.write(f"Error extracting changelog: {str(e)}")


def extract_readme():
    """Extract README content"""
    if os.path.exists('README.md'):
        with open('README.md', 'r') as f:
            readme_content = f.read()
        
        with open('readme_content.txt', 'w') as f:
            f.write(readme_content)
    else:
        with open('readme_content.txt', 'w') as f:
            f.write("No README.md found")


def analyze_conventional_commits():
    """Analyze conventional commits to categorize changes"""
    if not os.path.exists('pr_commits.txt'):
        with open('commit_categories.json', 'w') as f:
            f.write('{}')
        return
        
    try:
        with open('pr_commits.txt', 'r') as f:
            commits = f.readlines()
            
        categories = {
            'feat': {'title': 'Features', 'items': []},
            'fix': {'title': 'Bug Fixes', 'items': []},
            'docs': {'title': 'Documentation', 'items': []},
            'style': {'title': 'Styling', 'items': []},
            'refactor': {'title': 'Code Refactoring', 'items': []},
            'perf': {'title': 'Performance', 'items': []},
            'test': {'title': 'Tests', 'items': []},
            'build': {'title': 'Build System', 'items': []},
            'ci': {'title': 'CI', 'items': []},
            'chore': {'title': 'Chores', 'items': []}
        }
        
        other_commits = []
        
        for commit in commits:
            commit = commit.strip()
            matched = False
            
            for prefix in categories.keys():
                # Match prefix: or prefix(scope):
                if re.match(f'^{prefix}(\\(.*\\))?:', commit):
                    categories[prefix]['items'].append(commit)
                    matched = True
                    break
            
            if not matched:
                other_commits.append(commit)
        
        # Add uncategorized commits
        if other_commits:
            categories['other'] = {'title': 'Other Changes', 'items': other_commits}
        
        # Generate summary of categories
        with open('commit_categories.txt', 'w') as f:
            for category, data in categories.items():
                if data['items']:
                    f.write(f"## {data['title']} ({len(data['items'])})\n")
                    for item in data['items']:
                        f.write(f"- {item}\n")
                    f.write("\n")
        
        # Save full data as JSON
        with open('commit_categories.json', 'w') as f:
            json.dump(categories, f)
            
    except Exception as e:
        with open('commit_categories.json', 'w') as f:
            json.dump({"error": str(e)}, f)


def analyze_code_impact():
    """Analyze which components are most affected by the changes"""
    if not os.path.exists('pr_files_changed.txt'):
        with open('impact_analysis.txt', 'w') as f:
            f.write("No files changed information available")
        return
        
    try:
        with open('pr_files_changed.txt', 'r') as f:
            files = f.readlines()
        
        components = {}
        extensions = {}
        file_types = {
            'py': 'Python',
            'js': 'JavaScript',
            'ts': 'TypeScript',
            'jsx': 'React',
            'tsx': 'React TypeScript',
            'go': 'Go',
            'java': 'Java',
            'c': 'C',
            'cpp': 'C++',
            'h': 'Headers',
            'md': 'Documentation',
            'yml': 'Configuration',
            'yaml': 'Configuration',
            'json': 'Configuration',
            'toml': 'Configuration',
            'sql': 'Database',
            'html': 'Frontend',
            'css': 'Frontend',
            'scss': 'Frontend'
        }
        
        # Component analysis
        for file_line in files:
            # Strip status code (M, A, D, etc.) if present
            file_path = file_line.strip().split()[-1]
            
            # Skip deleted files
            if file_line.startswith('D'):
                continue
                
            parts = file_path.split('/')
            if len(parts) > 1:
                component = parts[0]
                components[component] = components.get(component, 0) + 1
            
            # Extension analysis
            ext = file_path.split('.')[-1] if '.' in file_path else 'unknown'
            file_type = file_types.get(ext, ext)
            extensions[file_type] = extensions.get(file_type, 0) + 1
        
        # Write impact analysis
        with open('impact_analysis.txt', 'w') as f:
            f.write("# Component Impact Analysis\n\n")
            
            f.write("## Components Changed\n")
            for component, count in sorted(components.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{component}: {count} files\n")
            
            f.write("\n## File Types\n")
            for ext, count in sorted(extensions.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{ext}: {count} files\n")
        
        # Save as JSON for programmatic use
        with open('impact_analysis.json', 'w') as f:
            json.dump({
                'components': components,
                'file_types': extensions
            }, f)
            
    except Exception as e:
        with open('impact_analysis.txt', 'w') as f:
            f.write(f"Error analyzing impact: {str(e)}")


def analyze_pr_labels():
    """Analyze PR labels if available"""
    if not os.path.exists('pr_labels.txt'):
        return
        
    try:
        with open('pr_labels.txt', 'r') as f:
            labels = [label.strip() for label in f.readlines() if label.strip()]
        
        label_categories = {
            'bug': 'Bug Fix',
            'enhancement': 'Enhancement',
            'feature': 'New Feature',
            'breaking': 'Breaking Change',
            'documentation': 'Documentation',
            'refactor': 'Refactoring',
            'test': 'Testing',
            'performance': 'Performance',
            'dependency': 'Dependencies',
            'security': 'Security'
        }
        
        categorized_labels = {}
        for label in labels:
            for key, category in label_categories.items():
                if key in label.lower():
                    categorized_labels[label] = category
                    break
            if label not in categorized_labels:
                categorized_labels[label] = 'Other'
        
        with open('label_analysis.json', 'w') as f:
            json.dump({
                'raw_labels': labels,
                'categorized_labels': categorized_labels
            }, f)
            
    except Exception as e:
        with open('label_analysis.json', 'w') as f:
            json.dump({"error": str(e)}, f)


def analyze_test_coverage():
    """Analyze test coverage impact"""
    if not os.path.exists('pr_files_changed.txt'):
        return
        
    try:
        with open('pr_files_changed.txt', 'r') as f:
            files = [line.strip().split()[-1] for line in f.readlines()]
        
        source_files = [f for f in files if not f.startswith('test') and 
                       not f.startswith('tests') and f.endswith('.py')]
        test_files = [f for f in files if (f.startswith('test') or 
                     f.startswith('tests')) and f.endswith('.py')]
        
        coverage_ratio = len(test_files)/len(source_files) if len(source_files) > 0 else 0
        
        with open('test_coverage_analysis.txt', 'w') as f:
            f.write(f'Source files modified: {len(source_files)}\n')
            f.write(f'Test files modified: {len(test_files)}\n')
            f.write(f'Test coverage ratio: {coverage_ratio:.2f}\n')
            
            if coverage_ratio < 0.5 and len(source_files) > 0:
                f.write('\nWarning: Test coverage might be insufficient.\n')
                f.write('The following source files have changes:\n')
                for file in source_files:
                    f.write(f'- {file}\n')
            
    except Exception as e:
        with open('test_coverage_analysis.txt', 'w') as f:
            f.write(f"Error analyzing test coverage: {str(e)}")


def main():
    """Main entry point"""
    print("Extracting repository context...")
    extract_readme()
    extract_module_info()
    extract_project_structure()
    extract_changelog_history()
    analyze_conventional_commits()
    analyze_code_impact()
    analyze_pr_labels()
    analyze_test_coverage()
    print("Repository context extraction complete.")


if __name__ == "__main__":
    main() 