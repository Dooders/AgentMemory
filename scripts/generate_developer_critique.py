"""
Generate developer critique for pull requests based on code changes.

This script analyzes PR changes and generates an expert developer's critique
focusing on code quality, potential issues, and architectural impact.
"""

import json
import os
import sys
import traceback
from typing import Any, Dict, Optional

import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger('developer_critique')

try:
    import openai
    import tiktoken
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Missing dependencies: {e}")
    DEPENDENCIES_AVAILABLE = False

# Set up OpenAI API
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY environment variable is not set")


def count_tokens(text: str) -> int:
    """Count tokens in a text string."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception as e:
        logger.warning(f"Error counting tokens: {e}")
        # Estimate tokens as 1 token ≈ 4 characters as fallback
        return len(text) // 4


def load_embeddings(file_path: str = "changelog_embeddings.json") -> Dict[str, Any]:
    """Load embeddings from JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Embeddings file {file_path} not found")
        return {"embeddings": {}, "token_counts": {}}
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in embeddings file {file_path}")
        return {"embeddings": {}, "token_counts": {}}
    except Exception as e:
        logger.error(f"Error loading embeddings: {e}")
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
        logger.warning("pr_files_changed.txt not found")
        details["files_changed"] = "No files changed information available"
    except Exception as e:
        logger.error(f"Error reading pr_files_changed.txt: {e}")
        details["files_changed"] = "Error loading files changed information"

    # Load commit messages
    try:
        with open("pr_commits.txt", "r") as f:
            details["commits"] = f.read()
    except FileNotFoundError:
        logger.warning("pr_commits.txt not found")
        details["commits"] = "No commit information available"
    except Exception as e:
        logger.error(f"Error reading pr_commits.txt: {e}")
        details["commits"] = "Error loading commit information"

    # Load diff stats
    try:
        with open("pr_diff_stats.txt", "r") as f:
            details["diff_stats"] = f.read()
    except FileNotFoundError:
        logger.warning("pr_diff_stats.txt not found")
        details["diff_stats"] = "No diff statistics available"
    except Exception as e:
        logger.error(f"Error reading pr_diff_stats.txt: {e}")
        details["diff_stats"] = "Error loading diff statistics"

    # Load PR labels
    try:
        with open("pr_labels.txt", "r") as f:
            details["labels"] = f.read()
    except FileNotFoundError:
        logger.warning("pr_labels.txt not found")
        details["labels"] = "No labels available"
    except Exception as e:
        logger.error(f"Error reading pr_labels.txt: {e}")
        details["labels"] = "Error loading PR labels"

    return details


def generate_developer_critique(
    pr_details: Dict[str, str], embeddings_data: Dict[str, Any]
) -> str:
    """Generate developer critique using GPT-4o and embeddings."""
    if not DEPENDENCIES_AVAILABLE:
        logger.error("Required dependencies (openai, tiktoken) are missing")
        return create_fallback_critique(pr_details)
        
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not set")
        return create_fallback_critique(pr_details)

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
        
        # Set max retries
        MAX_RETRIES = 3
        retry_count = 0
        last_error = None
        
        while retry_count < MAX_RETRIES:
            try:
                logger.info(f"Attempt {retry_count + 1}/{MAX_RETRIES} to generate developer critique")
                response = client.chat.completions.create(model="gpt-4o", messages=messages)
                critique = response.choices[0].message.content
                
                if not critique or len(critique.strip()) == 0:
                    logger.warning("API returned empty response")
                    retry_count += 1
                    continue
                
                logger.info("Successfully generated developer critique")
                return critique
            except Exception as e:
                last_error = e
                logger.warning(f"API call attempt {retry_count + 1} failed: {e}")
                retry_count += 1
                if retry_count < MAX_RETRIES:
                    logger.info("Retrying in 5 seconds...")
                    import time
                    time.sleep(5)
        
        logger.error(f"All API call attempts failed: {last_error}")
        return create_fallback_critique(pr_details)
    except Exception as e:
        logger.error(f"Error generating critique: {e}")
        logger.error(traceback.format_exc())
        return create_fallback_critique(pr_details)


def create_fallback_critique(pr_details: Dict[str, str]) -> str:
    """Create a basic fallback critique when API calls fail."""
    logger.info("Creating fallback developer critique")
    
    pr_number = pr_details.get("pr_number", "Unknown")
    pr_title = pr_details.get("pr_title", "Untitled PR")
    
    critique = f"""# Developer Critique for PR #{pr_number}

## Summary
This is an automatically generated fallback critique for PR: "{pr_title}"

## Code Quality
- Unable to analyze code quality due to API limitations
- Manual review recommended

## Testing
- Ensure adequate test coverage for changes
- Consider adding integration tests

## Security
- Review for potential security implications
- Follow secure coding best practices

## Documentation
- Ensure changes are well-documented
- Update README or relevant docs if needed

## Note
This is a fallback critique generated due to API limitations. A more detailed analysis could not be provided automatically.
"""
    return critique


def main():
    """Main function to generate developer critique."""
    try:
        logger.info("Starting developer critique generation")
        
        # Load PR details and embeddings
        pr_details = load_pr_details()
        embeddings_data = load_embeddings()

        # Generate developer critique
        critique = generate_developer_critique(pr_details, embeddings_data)

        # Write critique to file
        try:
            with open("developer_critique.md", "w") as f:
                f.write(critique)
            logger.info("Developer critique written to developer_critique.md")
        except Exception as e:
            logger.error(f"Error writing critique to file: {e}")
            logger.error(traceback.format_exc())
            
            # Try alternate location as a fallback
            try:
                with open("fallback_developer_critique.md", "w") as f:
                    f.write(critique)
                # Copy to expected filename
                import shutil
                shutil.copy("fallback_developer_critique.md", "developer_critique.md")
                logger.info("Used fallback method to write developer critique")
            except Exception as e2:
                logger.error(f"Critical error writing developer critique: {e2}")
                sys.exit(1)

        # Print a preview
        preview_length = min(500, len(critique))
        logger.info(f"Developer critique preview (first {preview_length} chars):")
        logger.info(critique[:preview_length] + "...")
        
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        logger.error(traceback.format_exc())
        
        # Create emergency critique
        emergency_critique = create_fallback_critique({
            "pr_number": os.environ.get("PR_NUMBER", "Unknown"),
            "pr_title": os.environ.get("PR_TITLE", "Untitled PR")
        })
        
        try:
            with open("developer_critique.md", "w") as f:
                f.write(emergency_critique)
            logger.info("Emergency critique saved")
        except:
            logger.error("Failed to write emergency critique")
            sys.exit(1)


if __name__ == "__main__":
    main()
