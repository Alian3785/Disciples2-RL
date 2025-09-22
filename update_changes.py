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
            f'--since={since_days} days ago',
            '--oneline',
            '--no-merges',
            '--pretty=format:%H|%s|%an|%ad|%D'
        ], capture_output=True, text=True, check=True)

        commits = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split('|', 4)
                if len(parts) >= 4:
                    commit_hash, message, author, date = parts[:4]
                    # Parse date (format: Thu Sep 19 10:30:00 2024 +0300)
                    try:
                        date_obj = datetime.strptime(date.split(' ', 3)[2], '%b %d %H:%M:%S %Y')
                        commits.append({
                            'hash': commit_hash[:8],
                            'message': message,
                            'author': author,
                            'date': date_obj.strftime('%Y-%m-%d'),
                            'refs': parts[4] if len(parts) > 4 else ''
                        })
                    except (IndexError, ValueError):
                        continue

        return commits
    except subprocess.CalledProcessError:
        print("Error: Not a git repository or git command failed")
        return []

def update_changes_md(commits):
    """Update changes.md with recent commits"""
    changes_file = Path('changes.md')

    if not changes_file.exists():
        print("changes.md not found")
        return

    # Read current content
    content = changes_file.read_text()

    # Add new commits to unreleased section
    new_entries = []
    for commit in commits:
        # Categorize commit message
        message = commit['message'].lower()

        category = "Changed"  # default
        if any(word in message for word in ['add', 'new', 'create', 'implement']):
            category = "Added"
        elif any(word in message for word in ['fix', 'bug', 'error', 'issue']):
            category = "Fixed"
        elif any(word in message for word in ['remove', 'delete', 'drop']):
            category = "Removed"

        new_entries.append(f"- {commit['message']} ({commit['hash']})")

    if new_entries:
        # Find the unreleased section and add entries
        lines = content.split('\n')
        updated_lines = []
        in_unreleased = False
        added_entries = False

        for line in lines:
            if line.strip() == '## [Unreleased]':
                in_unreleased = True
                updated_lines.append(line)
                continue

            if in_unreleased and line.strip().startswith('### Added'):
                # Add our entries before the Added section
                if not added_entries:
                    updated_lines.extend([f"### {category}"] + new_entries)
                    added_entries = True
                updated_lines.append(line)
                continue
            elif in_unreleased and line.strip().startswith('###'):
                # We've passed all the category sections
                break

            updated_lines.append(line)

        # Write back the updated content
        changes_file.write_text('\n'.join(updated_lines))
        print(f"Added {len(new_entries)} commit entries to changes.md")
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
