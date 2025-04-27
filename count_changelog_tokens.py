import os
import tiktoken
import json
from pathlib import Path

def count_tokens_in_file(file_path, max_chars=None):
    """Count tokens in a file using tiktoken, optionally truncating content"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Truncate if requested
        if max_chars and len(content) > max_chars:
            content = content[:max_chars] + "..."
        
        # Use cl100k_base tokenizer (used by GPT-4 and other recent models)
        encoder = tiktoken.get_encoding("cl100k_base")
        tokens = encoder.encode(content)
        return len(tokens), content
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0, ""

def count_changelog_context_tokens():
    """Count tokens in files used as context for changelog generation"""
    # Files used as context in the changelog generation process with maximum character limits
    # Based on the actual code in generate_changelog_entry.py
    context_files = [
        ('readme_content.txt', 800),          # README content (truncated to 800 chars)
        ('module_info.txt', 1000),            # Module information (truncated to 1000 chars)
        ('project_structure.txt', 500),       # Project structure (truncated to 500 chars)
        ('changelog_history.txt', 500),       # Previous changelog entries (truncated to 500 chars)
        ('commit_categories.txt', None),      # Conventional commits analysis (no truncation)
        ('impact_analysis.txt', None),        # Code impact analysis (no truncation)
        ('pr_commits.txt', None),             # PR commits (no truncation)
        ('pr_files_changed.txt', None),       # Files changed in PR (no truncation)
        ('pr_diff_stats.txt', None),          # Diff statistics (no truncation)
    ]
    
    # Initialize totals
    total_tokens = 0
    file_token_counts = {}
    truncated_contents = {}
    
    # Count tokens in each context file
    for file_name, max_chars in context_files:
        if os.path.exists(file_name):
            tokens, truncated_content = count_tokens_in_file(file_name, max_chars)
            file_token_counts[file_name] = tokens
            truncated_contents[file_name] = truncated_content
            total_tokens += tokens
            truncation_note = f" (truncated to {max_chars} chars)" if max_chars else ""
            print(f"{file_name}: {tokens:,} tokens{truncation_note}")
        else:
            print(f"{file_name}: File not found")
    
    # PR metadata (environment variables)
    pr_content = {
        'PR_NUMBER': '123',
        'PR_TITLE': 'Sample PR title',
        'PR_BODY': 'This is a sample PR description that might be several sentences long.',
        'REPO_NAME': 'username/repository-name'
    }
    
    pr_content_str = "\n".join([f"{k}: {v}" for k, v in pr_content.items()])
    encoder = tiktoken.get_encoding("cl100k_base")
    pr_content_tokens = len(encoder.encode(pr_content_str))
    
    print(f"PR metadata (env vars): ~{pr_content_tokens:,} tokens (simulated)")
    total_tokens += pr_content_tokens
    file_token_counts['PR metadata (env vars)'] = pr_content_tokens
    
    # Generate the actual prompt format used in generate_changelog_entry.py
    project_context = f"""
    # Library Context
    Repository: {pr_content['REPO_NAME']}
    
    ## Project Structure (directories only):
    {truncated_contents.get('project_structure.txt', '(Sample project structure would be here)')}
    
    ## Brief README Summary:
    {truncated_contents.get('readme_content.txt', '(Sample README content would be here)')}
    
    ## Key Modules:
    {truncated_contents.get('module_info.txt', '(Sample module info would be here)')}
    
    ## Changelog Format and History:
    {truncated_contents.get('changelog_history.txt', '(Sample changelog history would be here)')}
    """
    
    # Conventional commits and impact analysis are conditionally included
    conventional_commit_info = ""
    if os.path.exists('commit_categories.txt'):
        conventional_commit_info = f"""
        ## Conventional Commits Analysis:
        {truncated_contents.get('commit_categories.txt', '')}
        """
        
    impact_analysis_info = ""
    if os.path.exists('impact_analysis.txt'):
        impact_analysis_info = f"""
        ## Impact Analysis:
        {truncated_contents.get('impact_analysis.txt', '')}
        """
    
    # Build the complete prompt as used in the actual workflow
    sample_prompt = f"""
    Based on the following PR information, generate a concise changelog entry in markdown format for a new version.
    
    PR #{pr_content['PR_NUMBER']}: {pr_content['PR_TITLE']}
    
    PR Description:
    {pr_content['PR_BODY']}
    
    PR Labels: enhancement, bug
    
    Commits in this PR:
    {truncated_contents.get('pr_commits.txt', '(Sample commit data would be here)')}
    
    Files changed:
    {truncated_contents.get('pr_files_changed.txt', '(Sample files changed would be here)')}
    
    Diff statistics:
    {truncated_contents.get('pr_diff_stats.txt', '(Sample diff stats would be here)')}
    {conventional_commit_info}
    {impact_analysis_info}
    
    {project_context}
    
    Based on the analysis, the suggested semantic version bump would be minor (from 1.0.0).
    The current date is: 2023-12-01.
    
    Follow this exact format for the changelog:
    ## [VERSION] - DATE
    
    ### Category1
    - Description of changes:
      - Detailed point 1
      - Detailed point 2
    
    ### Category2
    - Description of other changes
    """
    
    prompt_tokens = len(encoder.encode(sample_prompt))
    print(f"Actual prompt with truncated content: {prompt_tokens:,} tokens")
    
    # Update total tokens based on the actual prompt
    total_tokens = prompt_tokens  # Reset to just the prompt tokens
    file_token_counts['Complete prompt'] = prompt_tokens
    
    print(f"\nTotal tokens for realistic changelog context: {total_tokens:,}")
    print(f"Estimated token usage for OpenAI API: {total_tokens:,}")
    
    # Calculate cost estimates (as of current OpenAI pricing)
    gpt4_input_price = 0.01 / 1000  # $0.01 per 1K tokens for GPT-4
    gpt4_output_price = 0.03 / 1000  # $0.03 per 1K tokens for GPT-4
    estimated_output_tokens = 500   # Estimate for changelog entry generation
    
    input_cost = (total_tokens * gpt4_input_price)
    output_cost = (estimated_output_tokens * gpt4_output_price)
    total_cost = input_cost + output_cost
    
    print(f"\nEstimated cost per API call (GPT-4):")
    print(f"Input tokens: {total_tokens:,} tokens = ${input_cost:.4f}")
    print(f"Output tokens: ~{estimated_output_tokens:,} tokens = ${output_cost:.4f}")
    print(f"Total estimated cost: ${total_cost:.4f} per changelog generation")
    
    # Save to JSON file
    with open('changelog_token_analysis.json', 'w') as f:
        json.dump({
            'total_tokens': total_tokens,
            'file_token_counts': file_token_counts,
            'estimated_output_tokens': estimated_output_tokens,
            'estimated_cost': {
                'input_cost': input_cost,
                'output_cost': output_cost,
                'total_cost': total_cost
            }
        }, f, indent=2)
    
    print("\nResults saved to changelog_token_analysis.json")

if __name__ == "__main__":
    # Check if tiktoken is installed, install if not
    try:
        import tiktoken
    except ImportError:
        print("Installing tiktoken...")
        import subprocess
        subprocess.check_call(["pip", "install", "tiktoken"])
        import tiktoken
    
    count_changelog_context_tokens() 