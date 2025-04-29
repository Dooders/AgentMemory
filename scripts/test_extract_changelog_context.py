#!/usr/bin/env python3
"""
Unit tests for extract_changelog_context.py script
"""
import os
import sys
import unittest
import tempfile
import shutil
from unittest.mock import patch, mock_open

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from scripts.extract_changelog_context import (
    extract_module_info,
    extract_project_structure,
    extract_changelog_history,
    extract_readme,
    analyze_conventional_commits,
    analyze_code_impact,
    analyze_pr_labels,
    analyze_test_coverage
)


class TestExtractChangelogContext(unittest.TestCase):
    """Test cases for extract_changelog_context.py functions"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.old_dir = os.getcwd()
        os.chdir(self.test_dir)
        
    def tearDown(self):
        """Clean up test environment"""
        os.chdir(self.old_dir)
        shutil.rmtree(self.test_dir)
    
    def test_extract_module_info(self):
        """Test extract_module_info function"""
        # Create a test Python file with a docstring
        os.makedirs("test_module")
        with open("test_module/test_file.py", "w") as f:
            f.write('"""Test docstring for module"""')
        
        # Run the function
        extract_module_info()
        
        # Check that the output file was created and contains the docstring
        self.assertTrue(os.path.exists("module_info.txt"))
        with open("module_info.txt", "r") as f:
            content = f.read()
        
        # Check for the file path - handle Windows paths (backslashes)
        file_path = content.split(':', 1)[0]
        self.assertTrue(file_path.endswith('test_module\\test_file.py') or 
                       file_path.endswith('test_module/test_file.py'))
        self.assertIn("Test docstring for module", content)
    
    def test_extract_project_structure(self):
        """Test extract_project_structure function"""
        # Create some test directories
        os.makedirs("dir1/subdir1")
        os.makedirs("dir2")
        
        # Run the function
        extract_project_structure()
        
        # Check that the output file was created and contains the directories
        self.assertTrue(os.path.exists("project_structure.txt"))
        with open("project_structure.txt", "r") as f:
            content = f.read()
        
        # Windows paths use backslashes, adjust test accordingly
        if os.sep == '\\':
            self.assertIn(".\\dir1", content)
            self.assertIn(".\\dir1\\subdir1", content)
            self.assertIn(".\\dir2", content)
        else:
            self.assertIn("./dir1", content)
            self.assertIn("./dir1/subdir1", content)
            self.assertIn("./dir2", content)
    
    def test_extract_changelog_history(self):
        """Test extract_changelog_history function"""
        # Create mock CHANGELOG.md with proper format matching regex r"## \[\d+\.\d+\.\d+\]"
        with open("CHANGELOG.md", "w") as f:
            f.write("# Changelog\n\n## [1.0.0]\nChanges in 1.0.0\n\n## [0.9.0]\nChanges in 0.9.0\n\n## [0.8.0]\nChanges in 0.8.0\n\n## [0.7.0]\nChanges in 0.7.0")
        
        # Run the function
        extract_changelog_history()
        
        # Check that output file was created
        self.assertTrue(os.path.exists("changelog_history.txt"))
        
        # Read the output file
        with open("changelog_history.txt", "r") as f:
            content = f.read()
            
        # The first three versions should be included
        self.assertIn("## [1.0.0]", content)
        self.assertIn("## [0.9.0]", content)
        self.assertIn("## [0.8.0]", content)
        # The fourth version should not be included
        self.assertNotIn("## [0.7.0]", content)
    
    def test_extract_readme(self):
        """Test extract_readme function"""
        # Create a mock README.md
        with open("README.md", "w") as f:
            f.write("# Test Project\nThis is a test readme.")
        
        # Run the function
        extract_readme()
        
        # Check that the output contains the README content
        self.assertTrue(os.path.exists("readme_content.txt"))
        with open("readme_content.txt", "r") as f:
            content = f.read()
        self.assertEqual("# Test Project\nThis is a test readme.", content)
    
    def test_analyze_conventional_commits(self):
        """Test analyze_conventional_commits function"""
        # Create mock pr_commits.txt
        with open("pr_commits.txt", "w") as f:
            f.write("feat: add new feature\nfix: fix bug\ndocs: update docs")
        
        # Run the function
        analyze_conventional_commits()
        
        # Check that commits are categorized correctly
        self.assertTrue(os.path.exists("commit_categories.json"))
        self.assertTrue(os.path.exists("commit_categories.txt"))
        
        import json
        with open("commit_categories.json", "r") as f:
            categories = json.load(f)
        
        self.assertEqual(len(categories["feat"]["items"]), 1)
        self.assertEqual(len(categories["fix"]["items"]), 1)
        self.assertEqual(len(categories["docs"]["items"]), 1)
    
    def test_analyze_code_impact(self):
        """Test analyze_code_impact function"""
        # Create mock pr_files_changed.txt
        with open("pr_files_changed.txt", "w") as f:
            f.write("M src/module1.py\nA src/module2.py\nM tests/test_file.py")
        
        # Run the function
        analyze_code_impact()
        
        # Check that files are categorized correctly
        self.assertTrue(os.path.exists("impact_analysis.txt"))
        self.assertTrue(os.path.exists("impact_analysis.json"))
        
        import json
        with open("impact_analysis.json", "r") as f:
            impact = json.load(f)
        
        self.assertEqual(impact["components"]["src"], 2)
        self.assertEqual(impact["components"]["tests"], 1)
        self.assertEqual(impact["file_types"]["Python"], 3)
    
    def test_analyze_pr_labels(self):
        """Test analyze_pr_labels function"""
        # Create mock pr_labels.txt
        with open("pr_labels.txt", "w") as f:
            f.write("bug\nenhancement\ndocumentation")
        
        # Run the function
        analyze_pr_labels()
        
        # Check that labels are categorized correctly
        self.assertTrue(os.path.exists("label_analysis.json"))
        
        import json
        with open("label_analysis.json", "r") as f:
            labels = json.load(f)
        
        self.assertEqual(len(labels["raw_labels"]), 3)
        self.assertEqual(labels["categorized_labels"]["bug"], "Bug Fix")
        self.assertEqual(labels["categorized_labels"]["enhancement"], "Enhancement")
        self.assertEqual(labels["categorized_labels"]["documentation"], "Documentation")
    
    def test_analyze_test_coverage(self):
        """Test analyze_test_coverage function"""
        # Create mock pr_files_changed.txt
        with open("pr_files_changed.txt", "w") as f:
            f.write("M src/module1.py\nM src/module2.py\nM tests/test_module1.py")
        
        # Run the function
        analyze_test_coverage()
        
        # Check that test coverage is calculated correctly
        self.assertTrue(os.path.exists("test_coverage_analysis.txt"))
        
        with open("test_coverage_analysis.txt", "r") as f:
            content = f.read()
        
        self.assertIn("Source files modified: 2", content)
        self.assertIn("Test files modified: 1", content)
        self.assertIn("Test coverage ratio: 0.50", content)


if __name__ == "__main__":
    unittest.main() 