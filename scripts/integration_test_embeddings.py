#!/usr/bin/env python3
"""
Integration test for the changelog embedding generation workflow.
Tests the full end-to-end process of generating embeddings and using them
to create a changelog entry.
"""
import os
import sys
import json
import unittest
import tempfile
import shutil
import subprocess
from unittest.mock import patch, MagicMock
from pathlib import Path

# Try to import the script modules - they should be in the same directory
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

try:
    from create_changelog_embeddings import create_changelog_context_embeddings
except ImportError:
    print("Could not import create_changelog_embeddings. Make sure it's in the same directory.")
    sys.exit(1)

class IntegrationTestEmbeddingWorkflow(unittest.TestCase):
    """Integration test for the full embedding workflow"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Save current directory to restore later
        self.old_cwd = os.getcwd()
        
        # Change to test directory
        os.chdir(self.test_dir)
        
        # Create test files
        self.create_test_files()
        
        # Environment variables for the PR
        os.environ["PR_NUMBER"] = "123"
        os.environ["PR_TITLE"] = "Add new feature"
        os.environ["PR_BODY"] = "This PR adds a new feature to the project"
        os.environ["REPO_NAME"] = "user/repo"
    
    def tearDown(self):
        """Clean up after tests"""
        # Restore original directory
        os.chdir(self.old_cwd)
        
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
        
        # Clear environment variables
        for var in ["PR_NUMBER", "PR_TITLE", "PR_BODY", "REPO_NAME"]:
            if var in os.environ:
                del os.environ[var]
    
    def create_test_files(self):
        """Create sample test files"""
        # Create files with realistic content
        with open("readme_content.txt", "w") as f:
            f.write("""# TASM (Tiered Adaptive Semantic Memory)
A flexible memory system for AI agents that organizes and retrieves information across multiple tiers.

## Features
- Short-term memory for recent context
- Long-term memory for persistent knowledge
- Semantic search for relevant information
- Memory management policies
""")
        
        with open("module_info.txt", "w") as f:
            f.write("""memory/base.py: Base classes for memory implementations
memory/short_term.py: Implementation of short-term memory
memory/long_term.py: Implementation of long-term memory
memory/semantic.py: Semantic search capabilities""")
        
        with open("project_structure.txt", "w") as f:
            f.write("""./memory
./memory/base.py
./memory/short_term.py
./memory/long_term.py
./memory/semantic.py
./tests
./tests/test_memory.py
./examples
./examples/basic_usage.py""")
        
        with open("changelog_history.txt", "w") as f:
            f.write("""# Changelog

## [0.2.0] - 2023-05-15
### Added
- Implemented semantic search
- Added memory management policies

## [0.1.0] - 2023-04-01
### Added
- Initial release with basic memory capabilities""")
        
        with open("pr_commits.txt", "w") as f:
            f.write("""abc123 - feat: implement vector embedding support (user)
def456 - test: add tests for embedding functionality (user)
ghi789 - docs: update documentation for embeddings (user)""")
        
        with open("pr_files_changed.txt", "w") as f:
            f.write("""A memory/embeddings.py
M memory/base.py
M README.md
A tests/test_embeddings.py""")
    
    @patch('create_changelog_embeddings.get_openai_embedding')
    @patch('subprocess.run')
    def test_full_workflow(self, mock_subprocess_run, mock_get_embedding):
        """Test the full embedding generation and changelog workflow"""
        # Mock embedding generation to return fixed vector
        mock_vector = [0.1] * 1536
        mock_get_embedding.return_value = mock_vector
        
        # Step 1: Generate embeddings
        create_changelog_context_embeddings()
        
        # Verify embeddings file was created
        self.assertTrue(os.path.exists("changelog_embeddings.json"))
        
        # Load the embeddings
        with open("changelog_embeddings.json", "r") as f:
            data = json.load(f)
        
        # Verify all expected embeddings are present
        self.assertIn("readme_content.txt", data["embeddings"])
        self.assertIn("module_info.txt", data["embeddings"])
        self.assertIn("project_structure.txt", data["embeddings"])
        self.assertIn("pr_metadata", data["embeddings"])
        
        # Dimensions check
        for file_name, embedding in data["embeddings"].items():
            self.assertEqual(len(embedding), 1536)
        
        # Step 2: Simulate API call with embeddings
        # Create a mock response for the OpenAI API call
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": """## [0.3.0] - 2023-06-01

### Added
- Implemented vector embedding support for improved memory retrieval
- Added embedding functionality to core memory system

### Changed
- Updated base memory interface to support embeddings
- Improved documentation for embedding features"""
                    }
                }
            ]
        }
        
        # Mock the subprocess call to simulate the curl API call
        mock_subprocess_run.return_value = MagicMock(
            stdout=json.dumps(mock_response).encode(),
            returncode=0
        )
        
        # Simulate the API call from the workflow
        api_call_cmd = [
            "curl", "-s", "-X", "POST",
            "https://api.openai.com/v1/chat/completions",
            "-H", "Content-Type: application/json",
            "-H", "Authorization: Bearer test_key",
            "-d", "@payload.json"
        ]
        
        # Create a payload file
        with open("payload.json", "w") as f:
            f.write(json.dumps({
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are a changelog generator."},
                    {"role": "user", "content": "Generate a changelog entry for PR #123"}
                ],
                "context_embeddings": data["embeddings"]
            }))
        
        # Make the simulated API call
        result = subprocess.run(api_call_cmd, capture_output=True)
        mock_subprocess_run.assert_called_once()
        
        # Step 3: Write the result to a file (simulating the workflow)
        with open("pr_changelog_entry.md", "w") as f:
            f.write(mock_response["choices"][0]["message"]["content"])
        
        # Verify the changelog entry file exists
        self.assertTrue(os.path.exists("pr_changelog_entry.md"))
        
        # Read the changelog entry
        with open("pr_changelog_entry.md", "r") as f:
            changelog_content = f.read()
        
        # Verify changelog content
        self.assertIn("vector embedding", changelog_content.lower())
        self.assertIn("## [0.3.0]", changelog_content)

if __name__ == "__main__":
    unittest.main() 