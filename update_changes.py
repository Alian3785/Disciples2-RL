#!/usr/bin/env python3
"""
Script to automatically update changes.md with recent git commits
"""

import subprocess
import re
from datetime import datetime
from pathlib import Path

def get_recent_commits(since_days=7):
    """Get recent commits from git log"""
    try:
        # Get commits from the last N days
        result = subprocess.run([
            'git', 'log',
            f'--since="{since_days} days ago"',
            '--oneline',
            '--no-merges'
        ], capture_output=True, text=True, check=True)

        commits = []
        for line in result.stdout.strip().split('\n'):
            if line:
                # Parse format: hash message
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    commit_hash = parts[0][:8]
                    message = parts[1]
                    commits.append({
                        'hash': commit_hash,
                        'message': message,
                        'author': 'Unknown',
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'refs': ''
                    })

        return commits
    except subprocess.CalledProcessError as e:
        print(f"Error: Git command failed - {e}")
        return []

def update_changes_md(commits):
    """Update changes.md with recent commits"""
    changes_file = Path('changes.md')

    if not changes_file.exists():
        print("changes.md not found")
        return

    # Read current content
    content = changes_file.read_text()

    # Group commits by category
    categories = {"Added": [], "Changed": [], "Fixed": [], "Removed": []}

    for commit in commits:
        message = commit['message'].lower()

        # Categorize commit message
        if any(word in message for word in ['add', 'new', 'create', 'implement', 'feat']):
            categories["Added"].append(commit)
        elif any(word in message for word in ['fix', 'bug', 'error', 'issue', 'resolve']):
            categories["Fixed"].append(commit)
        elif any(word in message for word in ['remove', 'delete', 'drop', 'obsolete']):
            categories["Removed"].append(commit)
        else:
            categories["Changed"].append(commit)

    # Create new entries grouped by category
    new_entries = []
    for category, commits_in_category in categories.items():
        if commits_in_category:
            new_entries.append(f"### {category}")
            for commit in commits_in_category:
                new_entries.append(f"- {commit['message']} ({commit['hash']})")

    if new_entries:
        # Find the unreleased section and add entries
        lines = content.split('\n')
        updated_lines = []
        in_unreleased = False
        added_entries = False

        i = 0
        while i < len(lines):
            line = lines[i]

            if line.strip() == '## [Unreleased]':
                in_unreleased = True
                updated_lines.append(line)
                i += 1
                continue

            if in_unreleased and line.strip().startswith('### Added'):
                # Add our entries before the Added section
                if not added_entries:
                    updated_lines.extend(new_entries)
                    added_entries = True
                updated_lines.append(line)
                i += 1
                continue
            elif in_unreleased and line.strip().startswith('###'):
                # We've passed all the category sections
                break

            updated_lines.append(line)
            i += 1

        # Add entries at the end if we didn't find the Added section
        if not added_entries and new_entries:
            updated_lines.extend(new_entries)

        # Write back the updated content
        changes_file.write_text('\n'.join(updated_lines))
        print(f"Added {len(commits)} commit entries to changes.md")
    else:
        print("No recent commits found")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Update changes.md with recent commits')
    parser.add_argument('--days', type=int, default=7, help='Number of days to look back')
    args = parser.parse_args()

    commits = get_recent_commits(args.days)
    if commits:
        print(f"Found {len(commits)} recent commits:")
        for commit in commits[:5]:  # Show first 5
            print(f"  {commit['hash']}: {commit['message']}")

        update_changes_md(commits)
    else:
        print("No commits found in the specified period")
