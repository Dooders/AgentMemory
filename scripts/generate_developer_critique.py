"""
Generate developer critique for pull requests based on code changes.

This script analyzes PR changes and generates an expert developer's critique
focusing on code quality, potential issues, and architectural impact.
"""

import json
import os
import sys
from typing import Any, Dict

import openai
import tiktoken

# Set up OpenAI API
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY environment variable is required")
    sys.exit(1)

openai.api_key = OPENAI_API_KEY


def count_tokens(text: str) -> int:
    """Count tokens in a text string."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def load_embeddings(file_path: str = "changelog_embeddings.json") -> Dict[str, Any]:
    """Load embeddings from JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Embeddings file {file_path} not found")
        return {"embeddings": {}, "token_counts": {}}


def load_pr_details() -> Dict[str, str]:
    """Load PR details from files."""
    details = {}

    # Load PR title and description
    details["pr_title"] = os.environ.get("PR_TITLE", "Untitled PR")
    details["pr_number"] = os.environ.get("PR_NUMBER", "Unknown")
    details["repo_name"] = os.environ.get("REPO_NAME", "Unknown")
    details["pr_description"] = os.environ.get("PR_BODY", "")

    # Load files changed
    try:
        with open("pr_files_changed.txt", "r") as f:
            details["files_changed"] = f.read()
    except FileNotFoundError:
        details["files_changed"] = "No files changed information available"

    # Load commit messages
    try:
        with open("pr_commits.txt", "r") as f:
            details["commits"] = f.read()
    except FileNotFoundError:
        details["commits"] = "No commit information available"

    # Load diff stats
    try:
        with open("pr_diff_stats.txt", "r") as f:
            details["diff_stats"] = f.read()
    except FileNotFoundError:
        details["diff_stats"] = "No diff statistics available"

    # Load PR labels
    try:
        with open("pr_labels.txt", "r") as f:
            details["labels"] = f.read()
    except FileNotFoundError:
        details["labels"] = "No labels available"

    return details


def generate_developer_critique(
    pr_details: Dict[str, str], embeddings_data: Dict[str, Any]
) -> str:
    """Generate developer critique using GPT-4o and embeddings."""

    # Create system prompt
    system_prompt = """You are an experienced senior developer tasked with reviewing code changes.
    Analyze the PR from multiple angles:
    
    1. Code Quality: Assess style, readability, maintainability
    2. Architecture: Evaluate design decisions, patterns, coupling
    3. Performance: Identify potential bottlenecks or optimizations
    4. Security: Spot potential vulnerabilities
    5. Testing: Assess test coverage and quality
    6. Documentation: Check for adequate documentation
    7. Future-proofing: Consider extensibility and scalability
    
    For each area:
    - Highlight strengths with specific examples
    - Identify concerns or areas for improvement with specific examples
    - Suggest alternatives or solutions where appropriate
    
    Be thorough but concise. Focus on substantial issues rather than minor style quibbles.
    Provide specific code references and clear, actionable recommendations.
    Use markdown formatting for readability."""

    # Construct the input for the model
    context = f"""
    PR #{pr_details['pr_number']} in {pr_details['repo_name']}
    Title: {pr_details['pr_title']}
    
    Description:
    {pr_details['pr_description']}
    
    Files Changed:
    {pr_details['files_changed']}
    
    Commit Messages:
    {pr_details['commits']}
    
    Diff Statistics:
    {pr_details['diff_stats']}
    
    Labels:
    {pr_details['labels']}
    """

    # Add relevant code context from embeddings if available
    if "content_samples" in embeddings_data:
        context += "\n\nCodebase Context:\n"
        for filename, content in embeddings_data.get("content_samples", {}).items():
            context += f"\n--- {filename} ---\n{content[:1000]}...\n"

    user_content = f"Analyze this PR and provide a detailed developer critique. Look at code quality, architecture, performance, security, testing, documentation, and future maintenance aspects.\n\nPR Details:\n{context}"

    # Create the payload for the API call
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    try:
        # Make the API call without embeddings context
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(model="gpt-4o", messages=messages)

        critique = response.choices[0].message.content
        return critique
    except Exception as e:
        print(f"Error generating critique: {e}")
        return "Failed to generate developer critique due to API errors."


def main():
    """Main function to generate developer critique."""
    # Load PR details and embeddings
    pr_details = load_pr_details()
    embeddings_data = load_embeddings()

    # Generate developer critique
    critique = generate_developer_critique(pr_details, embeddings_data)

    # Write critique to file
    with open("developer_critique.md", "w") as f:
        f.write(critique)

    print("Developer critique generated successfully!")

    # Print a preview
    preview_length = min(500, len(critique))
    print(f"\nPreview:\n{critique[:preview_length]}...")


if __name__ == "__main__":
    main()
