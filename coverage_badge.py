#!/usr/bin/env python3
"""
Generate coverage badge for README.
"""

import xml.etree.ElementTree as ET
import json
import os


def generate_coverage_badge():
    """Generate coverage badge from coverage.xml."""
    try:
        tree = ET.parse('coverage.xml')
        root = tree.getroot()

        # Get overall coverage percentage
        line_rate = float(root.get('line-rate', 0)) * 100
        branch_rate = float(root.get('branch-rate', 0)) * 100

        # Determine badge color
        if line_rate >= 90:
            color = "brightgreen"
        elif line_rate >= 80:
            color = "green"
        elif line_rate >= 70:
            color = "yellow"
        elif line_rate >= 60:
            color = "orange"
        else:
            color = "red"

        # Create badge URLs
        line_badge = f"https://img.shields.io/badge/coverage-{line_rate:.1f}%25-{color}"
        branch_badge = f"https://img.shields.io/badge/branch-{branch_rate:.1f}%25-{color}"

        # Update README if it exists
        readme_path = "README.md"
        if os.path.exists(readme_path):
            with open(readme_path, 'r') as f:
                content = f.read()

            # Replace coverage badges
            import re
            line_pattern = r'!\[Coverage\]\(https://img\.shields\.io/badge/coverage-\d+\.?\d*%25-\w+\)'
            branch_pattern = r'!\[Branch Coverage\]\(https://img\.shields\.io/badge/branch-\d+\.?\d*%25-\w+\)'

            content = re.sub(line_pattern, f'![Coverage]({line_badge})', content)
            content = re.sub(branch_pattern, f'![Branch Coverage]({branch_badge})', content)

            with open(readme_path, 'w') as f:
                f.write(content)

            print(f"Updated README with coverage: {line_rate:.1f}%")

        return line_rate

    except Exception as e:
        print(f"Could not generate coverage badge: {e}")
        return 0


if __name__ == "__main__":
    generate_coverage_badge()