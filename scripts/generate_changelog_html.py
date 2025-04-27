#!/usr/bin/env python3
"""
Script to generate a visual HTML representation of the changelog.
Creates a standalone HTML file with charts and formatted changelog content.
"""
import os
import re
import json
import sys
from datetime import datetime

# Check for required packages and notify if missing
try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False
    print("Warning: markdown package not available. Install with: pip install markdown")

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import base64
    from io import BytesIO
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib package not available. Install with: pip install matplotlib")

# HTML template for the visual changelog
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #3498db;
            margin-top: 30px;
        }}
        h3 {{
            color: #2980b9;
        }}
        .version {{
            background-color: #f1f8ff;
            border-left: 4px solid #0366d6;
            padding: 10px 15px;
            margin-bottom: 20px;
        }}
        .charts {{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-around;
            margin: 20px 0;
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
        }}
        .chart {{
            margin: 10px;
            text-align: center;
        }}
        .chart img {{
            max-width: 100%;
            height: auto;
        }}
        .metadata {{
            background-color: #f6f8fa;
            padding: 10px 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            font-size: 0.9em;
        }}
        pre {{
            background-color: #f6f8fa;
            padding: 15px;
            border-radius: 3px;
            overflow-x: auto;
        }}
        code {{
            font-family: Consolas, Monaco, 'Andale Mono', monospace;
            font-size: 0.9em;
        }}
        ul {{
            padding-left: 20px;
        }}
        li {{
            margin-bottom: 5px;
        }}
        .highlight {{
            background-color: #fffbdd;
            padding: 2px 5px;
            border-radius: 3px;
        }}
        .impact {{
            margin-top: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    
    <div class="metadata">
        <strong>Release Date:</strong> {date}<br>
        <strong>PR:</strong> #{pr_number} - {pr_title}<br>
        <strong>Repository:</strong> {repo_name}
    </div>
    
    <div class="charts">
        {charts_html}
    </div>
    
    <div class="version">
        {changelog_html}
    </div>
    
    <div class="impact">
        <h2>Impact Analysis</h2>
        {impact_html}
    </div>
</body>
</html>
"""

def figure_to_base64(fig):
    """Convert a matplotlib figure to a base64 string for embedding in HTML"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('ascii')
    return f"data:image/png;base64,{img_str}"


def create_commit_type_chart():
    """Create a pie chart showing commit types"""
    if not MATPLOTLIB_AVAILABLE or not os.path.exists('commit_categories.json'):
        return None, "No commit type data available"
        
    try:
        with open('commit_categories.json', 'r') as f:
            categories = json.load(f)
            
        # Extract counts and labels
        labels = []
        counts = []
        
        for category, data in categories.items():
            if data.get('items', []):
                labels.append(data.get('title', category))
                counts.append(len(data.get('items', [])))
        
        if not counts:
            return None, "No commit categories found"
            
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Commit Types')
        
        return fig, "Distribution of Commit Types"
    except Exception as e:
        return None, f"Error creating commit type chart: {str(e)}"


def create_file_impact_chart():
    """Create a bar chart showing file impact by component"""
    if not MATPLOTLIB_AVAILABLE or not os.path.exists('impact_analysis.json'):
        return None, "No impact analysis data available"
        
    try:
        with open('impact_analysis.json', 'r') as f:
            impact = json.load(f)
            
        components = impact.get('components', {})
        
        if not components:
            return None, "No component data found"
            
        # Sort by count and take top 10
        sorted_items = sorted(components.items(), key=lambda x: x[1], reverse=True)[:10]
        labels = [item[0] for item in sorted_items]
        counts = [item[1] for item in sorted_items]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(labels, counts)
        ax.set_xlabel('Number of Files')
        ax.set_title('Files Changed by Component')
        
        # Add value labels
        for i, v in enumerate(counts):
            ax.text(v + 0.1, i, str(v), va='center')
        
        plt.tight_layout()
        
        return fig, "Files Changed by Component"
    except Exception as e:
        return None, f"Error creating file impact chart: {str(e)}"


def create_file_type_chart():
    """Create a bar chart showing file types changed"""
    if not MATPLOTLIB_AVAILABLE or not os.path.exists('impact_analysis.json'):
        return None, "No file type data available"
        
    try:
        with open('impact_analysis.json', 'r') as f:
            impact = json.load(f)
            
        file_types = impact.get('file_types', {})
        
        if not file_types:
            return None, "No file type data found"
            
        # Sort by count and take top 8
        sorted_items = sorted(file_types.items(), key=lambda x: x[1], reverse=True)[:8]
        labels = [item[0] for item in sorted_items]
        counts = [item[1] for item in sorted_items]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, counts)
        ax.set_ylabel('Number of Files')
        ax.set_title('Files Changed by Type')
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        return fig, "Files Changed by Type"
    except Exception as e:
        return None, f"Error creating file type chart: {str(e)}"


def generate_html_changelog(pr_info):
    """Generate HTML changelog with visual elements"""
    if not MARKDOWN_AVAILABLE:
        print("Error: markdown package required but not available.")
        return
        
    # Get changelog content
    changelog_content = ""
    if os.path.exists('pr_changelog_entry.md'):
        with open('pr_changelog_entry.md', 'r') as f:
            changelog_content = f.read()
    else:
        print("Error: No changelog entry found at pr_changelog_entry.md")
        return
    
    # Extract version and date
    version_match = re.search(r'## \[(\d+\.\d+\.\d+)\]', changelog_content)
    version = version_match.group(1) if version_match else "new-version"
    
    # Convert markdown to HTML
    changelog_html = markdown.markdown(changelog_content)
    
    # Generate charts if matplotlib is available
    charts_html = ""
    if MATPLOTLIB_AVAILABLE:
        # Create charts
        commit_chart, commit_title = create_commit_type_chart()
        impact_chart, impact_title = create_file_impact_chart()
        file_type_chart, file_type_title = create_file_type_chart()
        
        # Add charts if available
        if commit_chart:
            img_data = figure_to_base64(commit_chart)
            charts_html += f'<div class="chart"><img src="{img_data}" alt="{commit_title}"><p>{commit_title}</p></div>'
            plt.close(commit_chart)
            
        if impact_chart:
            img_data = figure_to_base64(impact_chart)
            charts_html += f'<div class="chart"><img src="{img_data}" alt="{impact_title}"><p>{impact_title}</p></div>'
            plt.close(impact_chart)
            
        if file_type_chart:
            img_data = figure_to_base64(file_type_chart)
            charts_html += f'<div class="chart"><img src="{img_data}" alt="{file_type_title}"><p>{file_type_title}</p></div>'
            plt.close(file_type_chart)
    
    if not charts_html:
        charts_html = "<p>No charts available. Install matplotlib for visual charts.</p>"
    
    # Get impact analysis content
    impact_html = ""
    if os.path.exists('impact_analysis.txt'):
        with open('impact_analysis.txt', 'r') as f:
            impact_content = f.read()
        impact_html = markdown.markdown(impact_content)
    else:
        impact_html = "<p>No impact analysis available.</p>"
        
    # Test coverage analysis
    test_coverage_html = ""
    if os.path.exists('test_coverage_analysis.txt'):
        with open('test_coverage_analysis.txt', 'r') as f:
            test_coverage = f.read()
        test_coverage_html = f"<h3>Test Coverage</h3><pre>{test_coverage}</pre>"
        impact_html += test_coverage_html
    
    # Extract PR info
    pr_number = pr_info.get('pr_number', '0')
    pr_title = pr_info.get('pr_title', 'No title')
    repo_name = pr_info.get('repo_name', 'Unknown')
    current_date = pr_info.get('current_date', datetime.now().strftime('%Y-%m-%d'))
    
    # Generate HTML
    html = HTML_TEMPLATE.format(
        title=f"Changelog for v{version}",
        date=current_date,
        pr_number=pr_number,
        pr_title=pr_title,
        repo_name=repo_name,
        charts_html=charts_html,
        changelog_html=changelog_html,
        impact_html=impact_html
    )
    
    # Write HTML to file
    with open('changelog_visual.html', 'w') as f:
        f.write(html)
    
    print(f"Visual changelog generated in changelog_visual.html")


def main():
    """Main entry point"""
    # Get PR info from environment or default values
    pr_info = {
        'pr_number': os.getenv('PR_NUMBER', '0'),
        'pr_title': os.getenv('PR_TITLE', 'No title provided'),
        'repo_name': os.getenv('REPO_NAME', 'Unknown repository'),
        'current_date': datetime.now().strftime('%Y-%m-%d')
    }
    
    generate_html_changelog(pr_info)


if __name__ == "__main__":
    main() 