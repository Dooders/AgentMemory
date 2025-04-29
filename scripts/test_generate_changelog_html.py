#!/usr/bin/env python3
"""
Unit tests for generate_changelog_html.py script
"""
import os
import sys
import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock, mock_open

# Mock matplotlib modules before they're imported anywhere
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['matplotlib.figure'] = MagicMock()
sys.modules['matplotlib.use'] = MagicMock()

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


class TestGenerateChangelogHtml(unittest.TestCase):
    """Test cases for generate_changelog_html.py functions"""
    
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
    
    @patch('builtins.print')
    def test_no_markdown_available(self, mock_print):
        """Test when markdown is not available"""
        # Force a reload of the module if it's already loaded
        if 'scripts.generate_changelog_html' in sys.modules:
            del sys.modules['scripts.generate_changelog_html']
            
        # Mock the modules used in generate_changelog_html.py
        with patch.dict('sys.modules', {
            'markdown': None,
            'matplotlib': MagicMock(),
            'matplotlib.pyplot': MagicMock(),
            'matplotlib.figure': MagicMock(),
            'io': MagicMock(),
            'base64': MagicMock()
        }):
            # Import and patch the module
            from scripts.generate_changelog_html import generate_html_changelog
            
            # Test
            pr_info = {'pr_number': '123', 'pr_title': 'Test PR'}
            generate_html_changelog(pr_info)
            mock_print.assert_called_with("Error: markdown package required but not available.")
    
    @patch('builtins.print')
    def test_basic_functionality(self, mock_print):
        """Test basic functionality with simplified mocks"""
        # Force a reload of the module
        if 'scripts.generate_changelog_html' in sys.modules:
            del sys.modules['scripts.generate_changelog_html']
            
        # Create patch context with all dependencies mocked
        mock_markdown = MagicMock()
        mock_markdown.markdown.return_value = "<h2>Test</h2>"
        
        with patch.dict('sys.modules', {
            'markdown': mock_markdown,
            'matplotlib': MagicMock(),
            'matplotlib.pyplot': MagicMock(),
            'matplotlib.figure': MagicMock(),
            'io': MagicMock(),
            'base64': MagicMock()
        }):
            # Import module with mocks in place
            import scripts.generate_changelog_html as html_gen
            
            # Manually set availability flags
            html_gen.MARKDOWN_AVAILABLE = True
            html_gen.MATPLOTLIB_AVAILABLE = False  # Simplify by not using charts
            
            # Create test file
            with open('pr_changelog_entry.md', 'w') as f:
                f.write("## [1.2.3]\nTest changes")
            
            # Test with real file but mocked functions
            pr_info = {
                'pr_number': '123',
                'pr_title': 'Test PR',
                'repo_name': 'test/repo',
                'current_date': '2023-05-15'
            }
            
            # Run with file operations intercepted
            with patch('builtins.open', mock_open(read_data="## [1.2.3]\nTest changes")) as m:
                m.return_value.__enter__.return_value.read.return_value = "## [1.2.3]\nTest changes"
                html_gen.generate_html_changelog(pr_info)
            
            # Verify
            mock_print.assert_called_with("Visual changelog generated in changelog_visual.html")


if __name__ == "__main__":
    unittest.main() 