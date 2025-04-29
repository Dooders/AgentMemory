#!/usr/bin/env python3
"""
Integration tests for the changelog generation pipeline.
Tests all three scripts in sequence, similar to the GitHub workflow.
"""
import os
import sys
import shutil
import unittest
import tempfile
import subprocess
from unittest.mock import patch, MagicMock
import json
from datetime import datetime

# Add parent directory to path to import the modules
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


class TestChangelogPipeline(unittest.TestCase):
    """Integration tests for the complete changelog pipeline"""
    
    def setUp(self):
        """Set up test environment with a mock repository structure"""
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.old_dir = os.getcwd()
        
        # Get the project root directory (assuming we're in scripts/)
        self.project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        
        # Create a scripts directory in the test dir
        os.makedirs(os.path.join(self.test_dir, "scripts"))
        
        # Copy the necessary script files to the test directory
        script_files = [
            "extract_changelog_context.py",
            "generate_changelog_entry.py",
            "generate_changelog_html.py"
        ]
        
        for script in script_files:
            src_path = os.path.join(self.project_root, "scripts", script)
            dst_path = os.path.join(self.test_dir, "scripts", script)
            shutil.copy2(src_path, dst_path)
        
        # Change to the test directory
        os.chdir(self.test_dir)
        
        # Create a basic project structure
        os.makedirs("src/core")
        os.makedirs("src/utils")
        os.makedirs("tests")
        os.makedirs("docs")
        
        # Create a sample Python file with docstring
        with open("src/core/main.py", "w") as f:
            f.write('"""\nCore module for main functionality\n"""\n\ndef main():\n    """Main function"""\n    print("Hello World")\n')
        
        # Create a README.md
        with open("README.md", "w") as f:
            f.write("# Test Project\n\nThis is a test project for changelog generation.\n")
        
        # Create a sample CHANGELOG.md with previous entries
        with open("CHANGELOG.md", "w") as f:
            f.write("# Changelog\n\n## [1.2.3] - 2023-04-01\n\n### Features\n- Previous feature\n\n### Bug Fixes\n- Previous bug fix\n")
        
        # Create PR files that simulate GitHub workflow inputs
        with open("pr_commits.txt", "w") as f:
            f.write("feat: add new feature\nfix: fix important bug\ndocs: update documentation\n")
        
        with open("pr_files_changed.txt", "w") as f:
            f.write("M src/core/main.py\nA src/utils/new_feature.py\nM tests/test_main.py\n")
        
        with open("pr_diff_stats.txt", "w") as f:
            f.write(" src/core/main.py | 10 ++++++++--\n src/utils/new_feature.py | 25 +++++++++++++++++++++++++\n tests/test_main.py | 5 +++--\n")
        
        with open("pr_labels.txt", "w") as f:
            f.write("enhancement\nbug\n")
            
        # Create a mock tests file (mentioned in pr_files_changed.txt)
        with open("tests/test_main.py", "w") as f:
            f.write('"""Tests for main module"""\nimport unittest\n\nclass TestMain(unittest.TestCase):\n    def test_main(self):\n        self.assertTrue(True)\n')
            
        # Create a mock new feature file (mentioned in pr_files_changed.txt)
        with open("src/utils/new_feature.py", "w") as f:
            f.write('"""New feature module"""\n\ndef new_feature():\n    """Implements a new feature"""\n    return "New Feature"\n')
    
    def tearDown(self):
        """Clean up test environment"""
        os.chdir(self.old_dir)
        shutil.rmtree(self.test_dir)
    
    def run_script(self, script_name, env_vars=None):
        """Run a Python script and return its output"""
        # Set environment variables
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)
        
        # Use sys.executable to ensure we use the same Python interpreter
        # Make sure to use the local path in the test directory
        script_path = os.path.join(".", script_name)
        cmd = [sys.executable, script_path]
        result = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result
    
    def test_full_pipeline(self):
        """Test the full changelog generation pipeline"""
        # Set mock PR info
        pr_info = {
            "PR_TITLE": "Add new feature and fix bug",
            "PR_BODY": "This PR adds a new feature and fixes an important bug",
            "PR_NUMBER": "123",
            "REPO_NAME": "test/repo"
        }
        
        try:
            # Import and monkey patch to disable OpenAI
            orig_sys_path = sys.path.copy()
            sys.path.insert(0, self.test_dir)
            
            # We need to ensure the imports use our test files
            # Create an __init__.py in the scripts directory to make it a package
            with open(os.path.join("scripts", "__init__.py"), "w") as f:
                f.write("")
                
            try:
                # Override OpenAI availability for testing
                import scripts.generate_changelog_entry
                scripts.generate_changelog_entry.OPENAI_AVAILABLE = False
            except ImportError:
                # If we can't import directly, we'll use the @patch decorator during test
                pass
                
            # Restore original sys.path
            sys.path = orig_sys_path
            
            # Step 1: Extract repository context
            extract_result = self.run_script("scripts/extract_changelog_context.py")
            self.assertEqual(extract_result.returncode, 0, f"Extract context failed: {extract_result.stderr}")
            
            # Verify context files were created
            self.assertTrue(os.path.exists("module_info.txt"), "module_info.txt not created")
            self.assertTrue(os.path.exists("project_structure.txt"), "project_structure.txt not created")
            self.assertTrue(os.path.exists("readme_content.txt"), "readme_content.txt not created")
            self.assertTrue(os.path.exists("changelog_history.txt"), "changelog_history.txt not created")
            self.assertTrue(os.path.exists("commit_categories.txt"), "commit_categories.txt not created")
            self.assertTrue(os.path.exists("commit_categories.json"), "commit_categories.json not created")
            self.assertTrue(os.path.exists("impact_analysis.txt"), "impact_analysis.txt not created")
            self.assertTrue(os.path.exists("impact_analysis.json"), "impact_analysis.json not created")
            self.assertTrue(os.path.exists("test_coverage_analysis.txt"), "test_coverage_analysis.txt not created")
            
            # Step 2: Generate changelog entry
            entry_result = self.run_script("scripts/generate_changelog_entry.py", pr_info)
            self.assertEqual(entry_result.returncode, 0, f"Generate changelog entry failed: {entry_result.stderr}")
            
            # Verify changelog files were created
            self.assertTrue(os.path.exists("pr_changelog_entry.md"), "pr_changelog_entry.md not created")
            self.assertTrue(os.path.exists("RELEASE_NOTES.md"), "RELEASE_NOTES.md not created")
            
            # Check content of changelog entry
            with open("pr_changelog_entry.md", "r") as f:
                changelog_content = f.read()
                self.assertIn("## [", changelog_content, "Version heading missing in changelog")
                # Check for presence of categories from conventional commits
                self.assertIn("### Features", changelog_content, "Features section missing in changelog")
                self.assertIn("### Bug Fixes", changelog_content, "Bug Fixes section missing in changelog")
            
            # If matplotlib and markdown are available, test the HTML generation
            try:
                import matplotlib
                import markdown
                
                # Step 3: Generate HTML changelog
                html_result = self.run_script("scripts/generate_changelog_html.py", pr_info)
                self.assertEqual(html_result.returncode, 0, f"Generate HTML changelog failed: {html_result.stderr}")
                
                # Verify HTML file was created
                self.assertTrue(os.path.exists("changelog_visual.html"), "changelog_visual.html not created")
                
                # Check basic content of HTML file
                with open("changelog_visual.html", "r") as f:
                    html_content = f.read()
                    self.assertIn("<html", html_content, "HTML tag missing in changelog HTML")
                    self.assertIn("<body", html_content, "Body tag missing in changelog HTML")
                    self.assertIn("Changelog", html_content, "Changelog heading missing in HTML")
            
            except ImportError:
                print("Skipping HTML generation test as matplotlib or markdown is not available")
            
            # Step 4: Simulate updating the CHANGELOG.md (like in the GitHub workflow)
            with open("CHANGELOG.md", "r") as f:
                existing_content = f.read()
            
            with open("pr_changelog_entry.md", "r") as f:
                new_entry = f.read()
            
            with open("CHANGELOG.md", "w") as f:
                f.write(f"{new_entry}\n\n{existing_content}")
            
            # Verify CHANGELOG.md was updated
            with open("CHANGELOG.md", "r") as f:
                updated_content = f.read()
                # New entry should be at the beginning
                self.assertTrue(updated_content.startswith(new_entry.strip()), "New entry not added to CHANGELOG.md")
                # Old content should still be there
                self.assertIn("## [1.2.3]", updated_content, "Old content missing from updated CHANGELOG.md")
            
            print("All pipeline tests passed successfully")
            
        except Exception as e:
            self.fail(f"Pipeline test failed with error: {str(e)}")


if __name__ == "__main__":
    unittest.main() 