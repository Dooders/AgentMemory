#!/usr/bin/env python3
"""
Script to generate a changelog entry for a PR.
Uses OpenAI if an API key is available, or falls back to a basic entry.
"""
import os
import re
import sys
import json
import traceback
from datetime import datetime

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Configure logging for debugging
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger('changelog_generator')

def get_latest_version():
    """Extract the latest version from CHANGELOG.md"""
    latest_version = '0.0.0'
    try:
        with open('CHANGELOG.md', 'r') as f:
            content = f.read()
            version_match = re.search(r'## \[(\d+\.\d+\.\d+)\]', content)
            if version_match:
                latest_version = version_match.group(1)
    except FileNotFoundError:
        logger.warning("CHANGELOG.md not found, using default version 0.0.0")
    except Exception as e:
        logger.error(f"Error reading CHANGELOG.md: {str(e)}")
    return latest_version


def bump_major_version(version):
    """Increment the major version (x.0.0)"""
    major, _, _ = version.split('.')
    return f"{int(major) + 1}.0.0"


def bump_minor_version(version):
    """Increment the minor version (0.x.0)"""
    major, minor, _ = version.split('.')
    return f"{major}.{int(minor) + 1}.0"


def increment_patch_version(version):
    """Increment the patch version (0.0.x)"""
    major, minor, patch = version.split('.')
    return f"{major}.{minor}.{int(patch) + 1}"


def determine_version_bump(changes, current_version, labels=None):
    """Determine if changes warrant major, minor, or patch bump"""
    try:
        change_text = changes.lower()
        
        # Check for breaking changes based on labels
        if labels and any(('breaking' in label.lower() or 'major' in label.lower()) for label in labels):
            return bump_major_version(current_version)
        
        # Check for breaking changes based on conventional commits
        if os.path.exists('commit_categories.json'):
            try:
                with open('commit_categories.json', 'r') as f:
                    categories = json.load(f)
                    
                # Look for breaking changes indicator in any commit
                for category, data in categories.items():
                    for item in data.get('items', []):
                        if 'BREAKING CHANGE' in item or '!' in item.split(':')[0]:
                            return bump_major_version(current_version)
            except Exception as e:
                logger.warning(f"Error checking commit categories: {str(e)}")
        
        # Check for breaking changes in the content
        if any(keyword in change_text for keyword in 
            ['break', 'breaking', 'incompatible', 'major update', 'not backward compatible']):
            return bump_major_version(current_version)
        
        # Check for new features based on labels
        if labels and any(('feature' in label.lower() or 'enhancement' in label.lower()) for label in labels):
            return bump_minor_version(current_version)
        
        # Check for new features based on conventional commits
        if os.path.exists('commit_categories.json'):
            try:
                with open('commit_categories.json', 'r') as f:
                    categories = json.load(f)
                    
                # If there are feature commits, it's a minor update
                if categories.get('feat', {}).get('items', []):
                    return bump_minor_version(current_version)
            except Exception as e:
                logger.warning(f"Error checking commit categories for features: {str(e)}")
        
        # Check for new features in the content
        if any(keyword in change_text for keyword in 
            ['feat', 'feature', 'add', 'new', 'implement', 'support for']):
            return bump_minor_version(current_version)
        
        # Default to patch
        return increment_patch_version(current_version)
    except Exception as e:
        logger.error(f"Error determining version bump: {str(e)}")
        # Fall back to patch increment if anything fails
        return increment_patch_version(current_version)


def read_file_or_default(filepath, default="No content available"):
    """Read a file or return a default message if it doesn't exist"""
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except Exception as e:
        logger.warning(f"Error reading file {filepath}: {str(e)}")
        return default


def read_labels():
    """Read PR labels if available"""
    labels = []
    if os.path.exists('pr_labels.txt'):
        try:
            with open('pr_labels.txt', 'r') as f:
                labels = [label.strip() for label in f.readlines() if label.strip()]
        except Exception as e:
            logger.warning(f"Error reading PR labels: {str(e)}")
    return labels


def generate_with_openai(api_key, pr_info):
    """Generate a changelog entry using OpenAI API"""
    try:
        logger.info("Attempting to generate changelog with OpenAI API")
        
        if not api_key:
            logger.warning("No OpenAI API key provided")
            return generate_basic_entry(pr_info)
            
        client = OpenAI(api_key=api_key)
        
        # Create project context section
        project_context = f"""
        # Library Context
        Repository: {pr_info['repo_name']}
        
        ## Project Structure (directories only):
        {pr_info['project_structure'][:500] + '...' if len(pr_info['project_structure']) > 500 else pr_info['project_structure']}
        
        ## Brief README Summary:
        {pr_info['readme'][:800] + '...' if len(pr_info['readme']) > 800 else pr_info['readme']}
        
        ## Key Modules:
        {pr_info['module_info'][:1000] + '...' if len(pr_info['module_info']) > 1000 else pr_info['module_info']}
        
        ## Changelog Format and History:
        {pr_info['changelog_history'][:500] + '...' if len(pr_info['changelog_history']) > 500 else pr_info['changelog_history']}
        """
        
        # Add conventional commits analysis if available
        conventional_commit_info = ""
        if os.path.exists('commit_categories.txt'):
            conventional_commit_info = f"""
            ## Conventional Commits Analysis:
            {pr_info['commit_categories']}
            """
            
        # Add impact analysis if available
        impact_analysis_info = ""
        if os.path.exists('impact_analysis.txt'):
            impact_analysis_info = f"""
            ## Impact Analysis:
            {pr_info['impact_analysis']}
            """
            
        # Add test coverage analysis if available
        test_coverage_info = ""
        if os.path.exists('test_coverage_analysis.txt'):
            test_coverage_info = f"""
            ## Test Coverage Analysis:
            {pr_info['test_coverage']}
            """
        
        # Suggest version based on changes
        suggested_version = determine_version_bump(
            pr_info['pr_title'] + ' ' + pr_info['pr_body'] + ' ' + pr_info['commits'],
            pr_info['latest_version'],
            pr_info['labels']
        )
        
        prompt = f"""
        Based on the following PR information, generate a concise changelog entry in markdown format for a new version.
        
        PR #{pr_info['pr_number']}: {pr_info['pr_title']}
        
        PR Description:
        {pr_info['pr_body']}
        
        PR Labels: {', '.join(pr_info['labels']) if pr_info['labels'] else 'None'}
        
        Commits in this PR:
        {pr_info['commits']}
        
        Files changed:
        {pr_info['files_changed']}
        
        Diff statistics:
        {pr_info['diff_stats']}
        {conventional_commit_info}
        {impact_analysis_info}
        {test_coverage_info}
        
        {project_context}
        
        Based on the analysis, the suggested semantic version bump would be {suggested_version} (from {pr_info['latest_version']}).
        The current date is: {pr_info['current_date']}.
        
        Follow this exact format for the changelog:
        ## [VERSION] - DATE
        
        ### Category1
        - Description of changes:
          - Detailed point 1
          - Detailed point 2
        
        ### Category2
        - Description of other changes
        """
        
        logger.debug("Sending request to OpenAI API")
        
        response = client.chat.completions.create(
            model='gpt-4o',
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=800
        )
        
        changelog_entry = response.choices[0].message.content
        
        if not changelog_entry or len(changelog_entry.strip()) == 0:
            logger.warning("OpenAI returned empty response")
            return generate_basic_entry(pr_info)
            
        logger.info(f"Generated changelog entry with OpenAI for PR #{pr_info['pr_number']}")
        return changelog_entry
        
    except Exception as e:
        logger.error(f"Error generating changelog with OpenAI: {str(e)}")
        logger.error(traceback.format_exc())
        return generate_basic_entry(pr_info)


def generate_basic_entry(pr_info):
    """Generate a basic changelog entry when OpenAI is not available"""
    try:
        logger.info("Generating basic changelog entry.")
        
        # Try to intelligently determine version bump
        new_version = determine_version_bump(
            pr_info['pr_title'] + ' ' + pr_info['pr_body'] + ' ' + pr_info['commits'],
            pr_info['latest_version'],
            pr_info['labels']
        )
        
        entry = f"## [{new_version}] - {pr_info['current_date']}\n\n"
        
        # Try to categorize based on conventional commits
        if os.path.exists('commit_categories.json'):
            try:
                with open('commit_categories.json', 'r') as f:
                    categories = json.load(f)
                    
                has_categories = False
                for category, data in categories.items():
                    if data.get('items', []):
                        has_categories = True
                        entry += f"### {data['title']}\n"
                        for item in data['items']:
                            # Extract the actual message without the prefix
                            message = item.split(':', 1)[1].strip() if ':' in item else item
                            entry += f"- {message}\n"
                        entry += "\n"
                
                if has_categories:
                    logger.info("Generated changelog entry based on conventional commits")
                    return entry
            except Exception as e:
                logger.warning(f"Error processing conventional commits: {str(e)}")
        
        # Fallback if no conventional commits
        entry += "### Changes\n"
        entry += f"- {pr_info['pr_title']} (#{pr_info['pr_number']})\n\n"
        
        logger.info("Generated basic changelog entry as fallback")
        return entry
    except Exception as e:
        logger.error(f"Error in basic changelog generation: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Ultra fallback - ensure we always return something
        date = datetime.now().strftime('%Y-%m-%d')
        return f"## [0.0.1] - {date}\n\n### Changes\n- PR #{pr_info['pr_number']}: {pr_info['pr_title']}\n"


def generate_release_notes(changelog_entry, pr_info):
    """Generate more detailed release notes from changelog"""
    try:
        repo_name = pr_info['repo_name'].split('/')[-1] if '/' in pr_info['repo_name'] else pr_info['repo_name']
        
        # Extract version from changelog
        version_match = re.search(r'## \[(\d+\.\d+\.\d+)\]', changelog_entry)
        version = version_match.group(1) if version_match else 'new-version'
        
        # Build release notes
        release_notes = f"# Release Notes for {repo_name} v{version}\n\n"
        release_notes += changelog_entry
        
        # Add installation section
        release_notes += "\n\n## Installation\n\n"
        release_notes += f"```\npip install {repo_name}\n```\n\n"
        
        # Add impact analysis if available
        if os.path.exists('impact_analysis.txt'):
            impact_analysis = read_file_or_default('impact_analysis.txt')
            release_notes += f"\n\n## Impact Analysis\n\n{impact_analysis}\n\n"
        
        # Add test coverage information if available
        if os.path.exists('test_coverage_analysis.txt'):
            test_coverage = read_file_or_default('test_coverage_analysis.txt')
            release_notes += f"\n\n## Test Coverage\n\n{test_coverage}\n\n"
        
        # Add additional information
        release_notes += "## Additional Information\n\n"
        release_notes += f"For more details, see the [full changelog](https://github.com/{pr_info['repo_name']}/blob/main/CHANGELOG.md).\n"
        
        # Save release notes
        with open('RELEASE_NOTES.md', 'w') as f:
            f.write(release_notes)
            
        logger.info(f"Release notes generated in RELEASE_NOTES.md")
    except Exception as e:
        logger.error(f"Error generating release notes: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Create basic release notes to prevent failure
        try:
            with open('RELEASE_NOTES.md', 'w') as f:
                f.write(f"# Release Notes\n\n{changelog_entry}\n")
            logger.info("Basic release notes created as fallback")
        except Exception:
            logger.error("Failed to create even basic release notes")


def main():
    """Main entry point"""
    try:
        # Get environment variables
        pr_number = os.getenv('PR_NUMBER', '0')
        pr_title = os.getenv('PR_TITLE', 'No title provided')
        pr_body = os.getenv('PR_BODY', 'No description provided')
        repo_name = os.getenv('REPO_NAME', 'Unknown repository')
        openai_api_key = os.getenv('OPENAI_API_KEY', '')
        
        logger.info(f"Generating changelog for PR #{pr_number}: {pr_title}")
        
        # Get current date
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Get latest version
        latest_version = get_latest_version()
        
        # Get PR labels
        labels = read_labels()
        
        # Read files created by the context script
        commits = read_file_or_default('pr_commits.txt')
        files_changed = read_file_or_default('pr_files_changed.txt')
        diff_stats = read_file_or_default('pr_diff_stats.txt')
        readme = read_file_or_default('readme_content.txt')
        module_info = read_file_or_default('module_info.txt')
        changelog_history = read_file_or_default('changelog_history.txt')
        project_structure = read_file_or_default('project_structure.txt')
        commit_categories = read_file_or_default('commit_categories.txt')
        impact_analysis = read_file_or_default('impact_analysis.txt')
        test_coverage = read_file_or_default('test_coverage_analysis.txt')
        
        # Prepare PR info
        pr_info = {
            'pr_number': pr_number,
            'pr_title': pr_title,
            'pr_body': pr_body,
            'repo_name': repo_name,
            'latest_version': latest_version,
            'current_date': current_date,
            'commits': commits,
            'files_changed': files_changed,
            'diff_stats': diff_stats,
            'readme': readme,
            'module_info': module_info,
            'changelog_history': changelog_history,
            'project_structure': project_structure,
            'commit_categories': commit_categories,
            'impact_analysis': impact_analysis,
            'test_coverage': test_coverage,
            'labels': labels
        }
        
        # Generate changelog
        if OPENAI_AVAILABLE and openai_api_key:
            changelog_entry = generate_with_openai(openai_api_key, pr_info)
        else:
            logger.warning("OpenAI not available or no API key provided, using basic entry generation")
            changelog_entry = generate_basic_entry(pr_info)
        
        # Ensure we have valid content
        if not changelog_entry or len(changelog_entry.strip()) == 0:
            logger.error("Generated changelog entry is empty, using emergency fallback")
            changelog_entry = f"## [0.0.1] - {current_date}\n\n### Changes\n- PR #{pr_number}: {pr_title}\n"
        
        # Write to file
        try:
            with open('pr_changelog_entry.md', 'w') as f:
                f.write(changelog_entry)
            logger.info("Changelog entry saved to 'pr_changelog_entry.md'")
        except Exception as e:
            logger.error(f"Error writing to pr_changelog_entry.md: {str(e)}")
            logger.error(traceback.format_exc())
            # Last resort - try writing to current directory with a different name
            try:
                with open('fallback_changelog_entry.md', 'w') as f:
                    f.write(changelog_entry)
                # Copy to the expected filename
                import shutil
                shutil.copy('fallback_changelog_entry.md', 'pr_changelog_entry.md')
                logger.info("Used fallback method to write changelog entry")
            except Exception as e2:
                logger.error(f"Critical error writing changelog entry: {str(e2)}")
                sys.exit(1)
        
        # Generate release notes
        generate_release_notes(changelog_entry, pr_info)
        
        # Print to stdout for GitHub workflow
        print(changelog_entry)
        
        # Debug output
        logger.info(f"Changelog file exists: {os.path.exists('pr_changelog_entry.md')}")
        if os.path.exists('pr_changelog_entry.md'):
            logger.info(f"Changelog file size: {os.stat('pr_changelog_entry.md').st_size} bytes")
    
    except Exception as e:
        logger.error(f"Unhandled exception in main: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Create an emergency changelog entry
        emergency_entry = f"## [0.0.1] - {datetime.now().strftime('%Y-%m-%d')}\n\n### Emergency Entry\n- PR #{os.getenv('PR_NUMBER', '0')}: {os.getenv('PR_TITLE', 'No title')}\n- Note: This is an emergency fallback entry due to generation failure\n"
        
        try:
            with open('pr_changelog_entry.md', 'w') as f:
                f.write(emergency_entry)
            print(emergency_entry)
        except:
            # If we can't even write the emergency entry, we're out of options
            sys.exit(1)


if __name__ == "__main__":
    main()
    
    # Final validation
    try:
        if not os.path.exists('pr_changelog_entry.md') or os.stat('pr_changelog_entry.md').st_size == 0:
            logger.error("Error: Failed to generate a valid changelog entry.")
            
            # Last ditch effort - create a minimal valid entry
            emergency_entry = f"## [0.0.1] - {datetime.now().strftime('%Y-%m-%d')}\n\n### Emergency Entry\n- Generated as fallback\n"
            with open('pr_changelog_entry.md', 'w') as f:
                f.write(emergency_entry)
                
            # Check if it worked
            if not os.path.exists('pr_changelog_entry.md') or os.stat('pr_changelog_entry.md').st_size == 0:
                sys.exit(1)
    except Exception as e:
        logger.error(f"Error in final validation: {str(e)}")
        sys.exit(1) 