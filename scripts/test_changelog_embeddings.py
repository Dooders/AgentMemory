#!/usr/bin/env python3
"""
Unit tests for create_changelog_embeddings.py
"""
import os
import json
import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from pathlib import Path

# Import functions from the script to test
from create_changelog_embeddings import (
    count_tokens,
    truncate_text,
    get_openai_embedding,
    create_changelog_context_embeddings
)

class TestChangelogEmbeddings(unittest.TestCase):
    """Test cases for the changelog embeddings script"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Save current directory to restore later
        self.old_cwd = os.getcwd()
        
        # Change to test directory
        os.chdir(self.test_dir)
        
        # Create sample test files
        self.create_test_files()
    
    def tearDown(self):
        """Clean up after tests"""
        # Restore original directory
        os.chdir(self.old_cwd)
        
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def create_test_files(self):
        """Create sample test files for embedding generation"""
        # Create readme file
        with open("readme_content.txt", "w") as f:
            f.write("# Test Project\nThis is a sample README for testing embeddings.")
        
        # Create module info file
        with open("module_info.txt", "w") as f:
            f.write("module.py: This is a sample module docstring.")
        
        # Create project structure file
        with open("project_structure.txt", "w") as f:
            f.write("./src\n./tests\n./docs")
        
        # Create changelog history file
        with open("changelog_history.txt", "w") as f:
            f.write("# Changelog\n\n## [1.0.0]\n- Initial release")
        
        # Create PR commits file
        with open("pr_commits.txt", "w") as f:
            f.write("abcd123 - Add new feature (User)")
    
    def test_count_tokens(self):
        """Test token counting function"""
        text = "This is a sample text with 10 tokens."
        token_count = count_tokens(text)
        self.assertGreaterEqual(token_count, 8)  # At least 8 tokens (may vary by tokenizer)
    
    def test_truncate_text(self):
        """Test text truncation function"""
        text = "This is a sample text that will be truncated."
        # Truncate to 10 chars
        truncated = truncate_text(text, 10)
        self.assertEqual(truncated, "This is a ...")
        
        # No truncation needed
        short_text = "Short"
        truncated = truncate_text(short_text, 10)
        self.assertEqual(truncated, short_text)
        
        # No max_chars provided
        truncated = truncate_text(text, None)
        self.assertEqual(truncated, text)
    
    @patch('create_changelog_embeddings.get_openai_embedding')
    def test_embedding_generation(self, mock_get_embedding):
        """Test embedding generation for files"""
        # Mock the embedding function to return a fixed vector
        mock_vector = [0.1] * 1536  # 1536-dimensional mock vector
        mock_get_embedding.return_value = mock_vector
        
        # Run the embedding generation
        create_changelog_context_embeddings()
        
        # Check that the output file was created
        self.assertTrue(os.path.exists("changelog_embeddings.json"))
        
        # Load the output file
        with open("changelog_embeddings.json", "r") as f:
            data = json.load(f)
        
        # Check that embeddings were created for all test files
        self.assertIn("readme_content.txt", data["embeddings"])
        self.assertIn("module_info.txt", data["embeddings"])
        self.assertIn("project_structure.txt", data["embeddings"])
        self.assertIn("changelog_history.txt", data["embeddings"])
        self.assertIn("pr_commits.txt", data["embeddings"])
        
        # Check that token counts were calculated
        self.assertIn("readme_content.txt", data["token_counts"])
        self.assertGreater(data["token_counts"]["readme_content.txt"], 0)
        
        # Check that content samples were created
        self.assertIn("readme_content.txt", data["content_samples"])
        self.assertIn("# Test Project", data["content_samples"]["readme_content.txt"])
        
        # Check embedding dimensions
        self.assertEqual(len(data["embeddings"]["readme_content.txt"]), 1536)
    
    @patch('create_changelog_embeddings.get_openai_embedding')
    @patch.dict(os.environ, {
        "PR_NUMBER": "123",
        "PR_TITLE": "Test PR",
        "PR_BODY": "This is a test PR",
        "REPO_NAME": "user/repo"
    })
    def test_pr_metadata_embedding(self, mock_get_embedding):
        """Test PR metadata embedding generation"""
        # Mock the embedding function to return a fixed vector
        mock_vector = [0.1] * 1536  # 1536-dimensional mock vector
        mock_get_embedding.return_value = mock_vector
        
        # Run the embedding generation
        create_changelog_context_embeddings()
        
        # Load the output file
        with open("changelog_embeddings.json", "r") as f:
            data = json.load(f)
        
        # Check that PR metadata embedding was created
        self.assertIn("pr_metadata", data["embeddings"])
        self.assertIn("pr_metadata", data["token_counts"])
        self.assertIn("pr_metadata", data["content_samples"])
        
        # Check content sample contains PR info
        self.assertIn("PR_NUMBER: 123", data["content_samples"]["pr_metadata"])
        self.assertIn("PR_TITLE: Test PR", data["content_samples"]["pr_metadata"])
    
    @patch('create_changelog_embeddings.OpenAI')
    def test_openai_api_call(self, mock_openai):
        """Test actual OpenAI API call format"""
        # Create a mock for the OpenAI client
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Create a mock for the embeddings response
        mock_response = MagicMock()
        mock_embeddings_create = mock_client.embeddings.create
        mock_embeddings_create.return_value = mock_response
        
        # Set up the mock response
        mock_data = MagicMock()
        mock_data.embedding = [0.1] * 1536
        mock_response.data = [mock_data]
        
        # Call the function
        result = get_openai_embedding("Test text")
        
        # Verify the API was called correctly
        mock_embeddings_create.assert_called_once()
        call_args = mock_embeddings_create.call_args[1]
        self.assertEqual(call_args["input"], "Test text")
        self.assertEqual(call_args["model"], "text-embedding-3-small")
        
        # Verify the result
        self.assertEqual(len(result), 1536)
    
    def test_file_truncation(self):
        """Test file content truncation"""
        # Create a long file
        long_content = "A" * 1000
        with open("long_file.txt", "w") as f:
            f.write(long_content)
        
        # Add to the test files list
        with patch('create_changelog_embeddings.get_openai_embedding') as mock_get_embedding:
            # Mock the embedding function
            mock_get_embedding.return_value = [0.1] * 1536
            
            # Add the long file to the list of files to process
            with patch('create_changelog_embeddings.context_files', 
                      [('long_file.txt', 100)]):  # Truncate to 100 chars
                
                # Run embedding generation
                create_changelog_context_embeddings()
                
                # Check the result
                with open("changelog_embeddings.json", "r") as f:
                    data = json.load(f)
                
                # Verify truncation
                self.assertIn("long_file.txt", data["content_samples"])
                # Should be 100 chars + "..."
                self.assertEqual(len(data["content_samples"]["long_file.txt"]), 103)

if __name__ == "__main__":
    unittest.main() 