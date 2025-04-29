#!/usr/bin/env python3
"""
Unit tests for generate_changelog_entry.py script
"""
import os
import sys
import unittest
import tempfile
import shutil
import json
from unittest.mock import patch, mock_open
from datetime import datetime

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from scripts.generate_changelog_entry import (
    get_latest_version,
    bump_major_version,
    bump_minor_version,
    increment_patch_version,
    determine_version_bump,
    read_file_or_default,
    read_labels,
    generate_basic_entry,
    generate_release_notes
)


class TestGenerateChangelogEntry(unittest.TestCase):
    """Test cases for generate_changelog_entry.py functions"""
    
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
    
    def test_get_latest_version(self):
        """Test get_latest_version function"""
        # Test with no CHANGELOG.md
        self.assertEqual(get_latest_version(), '0.0.0')
        
        # Test with CHANGELOG.md
        with open("CHANGELOG.md", "w") as f:
            f.write("# Changelog\n\n## [1.2.3]\nSome changes\n\n## [1.1.0]\nOlder changes")
        
        self.assertEqual(get_latest_version(), '1.2.3')
    
    def test_bump_major_version(self):
        """Test bump_major_version function"""
        self.assertEqual(bump_major_version('1.2.3'), '2.0.0')
        self.assertEqual(bump_major_version('0.9.9'), '1.0.0')
    
    def test_bump_minor_version(self):
        """Test bump_minor_version function"""
        self.assertEqual(bump_minor_version('1.2.3'), '1.3.0')
        self.assertEqual(bump_minor_version('1.9.9'), '1.10.0')
    
    def test_increment_patch_version(self):
        """Test increment_patch_version function"""
        self.assertEqual(increment_patch_version('1.2.3'), '1.2.4')
        self.assertEqual(increment_patch_version('1.2.9'), '1.2.10')
    
    def test_determine_version_bump_major(self):
        """Test determine_version_bump for major version changes"""
        # Test with breaking changes in text
        self.assertEqual(
            determine_version_bump('This is a breaking change', '1.2.3'),
            '2.0.0'
        )
        
        # Test with breaking labels
        self.assertEqual(
            determine_version_bump('Some change', '1.2.3', ['breaking-change']),
            '2.0.0'
        )
        
        # Test with conventional commit with breaking change
        with open('commit_categories.json', 'w') as f:
            json.dump({
                'feat': {
                    'title': 'Features',
                    'items': ['feat!: breaking feature change']
                }
            }, f)
        
        self.assertEqual(
            determine_version_bump('Some change', '1.2.3'),
            '2.0.0'
        )
    
    def test_determine_version_bump_minor(self):
        """Test determine_version_bump for minor version changes"""
        # Test with feature changes in text
        self.assertEqual(
            determine_version_bump('Added a new feature', '1.2.3'),
            '1.3.0'
        )
        
        # Test with feature labels
        self.assertEqual(
            determine_version_bump('Some change', '1.2.3', ['feature']),
            '1.3.0'
        )
        
        # Test with conventional commit with feature
        with open('commit_categories.json', 'w') as f:
            json.dump({
                'feat': {
                    'title': 'Features',
                    'items': ['feat: add new feature']
                }
            }, f)
        
        self.assertEqual(
            determine_version_bump('Some change', '1.2.3'),
            '1.3.0'
        )
    
    def test_determine_version_bump_patch(self):
        """Test determine_version_bump for patch version changes"""
        # Test with bug fix changes
        self.assertEqual(
            determine_version_bump('Fixed a bug', '1.2.3'),
            '1.2.4'
        )
        
        # Default is patch
        self.assertEqual(
            determine_version_bump('Some change', '1.2.3'),
            '1.2.4'
        )
    
    def test_read_file_or_default(self):
        """Test read_file_or_default function"""
        # Test with non-existent file
        self.assertEqual(
            read_file_or_default('nonexistent.txt', 'Default message'),
            'Default message'
        )
        
        # Test with existing file
        with open('test.txt', 'w') as f:
            f.write('File content')
        
        self.assertEqual(
            read_file_or_default('test.txt', 'Default message'),
            'File content'
        )
    
    def test_read_labels(self):
        """Test read_labels function"""
        # Test with no labels file
        self.assertEqual(read_labels(), [])
        
        # Test with labels file
        with open('pr_labels.txt', 'w') as f:
            f.write('bug\nenhancement\ndocumentation')
        
        self.assertEqual(read_labels(), ['bug', 'enhancement', 'documentation'])
    
    @patch('os.path.exists')
    def test_generate_basic_entry(self, mock_exists):
        """Test generate_basic_entry function"""
        # Mock the commit categories existence
        mock_exists.return_value = True
        
        # Create test commit categories
        with open('commit_categories.json', 'w') as f:
            json.dump({
                'feat': {
                    'title': 'Features',
                    'items': ['feat: add new feature']
                },
                'fix': {
                    'title': 'Bug Fixes',
                    'items': ['fix: fix a bug']
                }
            }, f)
        
        # Create PR info
        pr_info = {
            'pr_number': '123',
            'pr_title': 'Test PR',
            'pr_body': 'Test PR description',
            'repo_name': 'test/repo',
            'latest_version': '1.2.3',
            'current_date': '2023-05-15',
            'commits': 'feat: add new feature\nfix: fix a bug',
            'files_changed': 'src/file1.py\nsrc/file2.py',
            'diff_stats': '+10 -5',
            'labels': ['enhancement']
        }
        
        # Generate changelog
        changelog = generate_basic_entry(pr_info)
        
        # Check result contains expected content
        self.assertIn('## [1.3.0] - 2023-05-15', changelog)
        self.assertIn('### Features', changelog)
        self.assertIn('- add new feature', changelog)
        self.assertIn('### Bug Fixes', changelog)
        self.assertIn('- fix a bug', changelog)
    
    def test_generate_release_notes(self):
        """Test generate_release_notes function"""
        # Create test PR info
        pr_info = {
            'pr_number': '123',
            'pr_title': 'Test PR',
            'pr_body': 'Test PR description',
            'repo_name': 'test/repo',
            'latest_version': '1.2.3',
            'current_date': '2023-05-15'
        }
        
        # Create test changelog entry
        changelog_entry = "## [1.3.0] - 2023-05-15\n\n### Features\n- New feature\n\n### Bug Fixes\n- Fixed bug"
        
        # Create impact analysis
        with open('impact_analysis.txt', 'w') as f:
            f.write("# Impact Analysis\nSome module changes")
        
        # Create test coverage
        with open('test_coverage_analysis.txt', 'w') as f:
            f.write("Test coverage: 80%")
        
        # Generate release notes
        generate_release_notes(changelog_entry, pr_info)
        
        # Check release notes file
        self.assertTrue(os.path.exists('RELEASE_NOTES.md'))
        with open('RELEASE_NOTES.md', 'r') as f:
            content = f.read()
        
        # Verify content
        self.assertIn('# Release Notes for repo v1.3.0', content)
        self.assertIn('## [1.3.0] - 2023-05-15', content)
        self.assertIn('### Features', content)
        self.assertIn('- New feature', content)
        self.assertIn('### Bug Fixes', content)
        self.assertIn('- Fixed bug', content)
        self.assertIn('## Installation', content)
        self.assertIn('pip install repo', content)
        self.assertIn('## Impact Analysis', content)
        self.assertIn('Some module changes', content)
        self.assertIn('## Test Coverage', content)
        self.assertIn('Test coverage: 80%', content)


if __name__ == "__main__":
    unittest.main() 