#!/usr/bin/env python3
"""
Test runner script for RAG Orchestrator.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_tests(test_type: str = "all", coverage: bool = True, verbose: bool = False):
    """
    Run tests based on type.

    Args:
        test_type: Type of tests to run (all, unit, integration, e2e, performance)
        coverage: Whether to generate coverage reports
        verbose: Whether to show verbose output
    """
    test_paths = {
        "all": "tests/",
        "unit": "tests/unit/",
        "integration": "tests/integration/",
        "e2e": "tests/e2e/",
        "performance": "tests/performance/",
        "api": "tests/integration/test_api_endpoints.py",
    }

    if test_type not in test_paths:
        print(f"Unknown test type: {test_type}")
        print(f"Available types: {', '.join(test_paths.keys())}")
        return 1

    test_path = test_paths[test_type]

    # Build pytest command
    cmd = ["pytest", test_path]

    if coverage:
        cmd.extend(
            [
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html:coverage_html",
                "--cov-report=xml:coverage.xml",
            ]
        )

    if verbose:
        cmd.append("-v")

    if test_type == "performance" or test_type == "e2e":
        cmd.append("-m")
        cmd.append("not slow")  # Skip slow tests by default

    # Add markers for specific test types
    if test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "e2e":
        cmd.extend(["-m", "e2e"])
    elif test_type == "performance":
        cmd.extend(["-m", "performance"])

    print(f"Running {test_type} tests...")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Tests failed with exit code: {e.returncode}")
        return e.returncode


def run_linting():
    """Run code linting and formatting checks."""
    print("Running code linting...")

    commands = [
        ["black", "--check", "src/", "tests/"],
        ["isort", "--check-only", "src/", "tests/"],
        ["ruff", "check", "src/", "tests/"],
        ["mypy", "src/"],
    ]

    all_passed = True

    for cmd in commands:
        print(f"\nRunning: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print("✅ Passed")
        except subprocess.CalledProcessError:
            print("❌ Failed")
            all_passed = False

    return 0 if all_passed else 1


def run_specific_test(test_name: str):
    """Run a specific test by name."""
    cmd = ["pytest", "-v", "-k", test_name]
    print(f"Running specific test: {test_name}")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        return e.returncode


def generate_coverage_report():
    """Generate HTML coverage report."""
    print("Generating coverage report...")
    cmd = ["coverage", "html", "-d", "coverage_html"]
    subprocess.run(cmd, check=True)
    print(f"Coverage report generated: file://{Path.cwd()}/coverage_html/index.html")
    return 0


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="RAG Orchestrator Test Runner")
    parser.add_argument(
        "test_type",
        nargs="?",
        default="all",
        choices=[
            "all",
            "unit",
            "integration",
            "e2e",
            "performance",
            "api",
            "lint",
            "coverage",
            "specific",
        ],
        help="Type of tests to run",
    )
    parser.add_argument(
        "--no-coverage", action="store_true", help="Disable coverage reporting"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--test-name", help="Specific test to run (use with test_type=specific)"
    )
    parser.add_argument(
        "--include-slow", action="store_true", help="Include slow tests"
    )

    args = parser.parse_args()

    if args.test_type == "lint":
        return run_linting()
    elif args.test_type == "coverage":
        return generate_coverage_report()
    elif args.test_type == "specific":
        if not args.test_name:
            print("Error: --test-name is required for specific test runs")
            return 1
        return run_specific_test(args.test_name)
    else:
        coverage = not args.no_coverage
        return run_tests(
            test_type=args.test_type, coverage=coverage, verbose=args.verbose
        )


if __name__ == "__main__":
    sys.exit(main())
