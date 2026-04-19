#!/usr/bin/env python3
"""
Control script for algorithmic trading strategies project.
Provides command-line interface for project management tasks.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def get_venv_pylint(venv_path: Path) -> Path:
    """Get the path to pylint in the virtual environment."""
    if sys.platform == "win32":
        return venv_path / "Scripts" / "pylint.exe"
    else:
        return venv_path / "bin" / "pylint"


def get_venv_python(venv_path: Path) -> Path:
    """Get the path to the Python interpreter in the virtual environment."""
    if sys.platform == "win32":
        return venv_path / "Scripts" / "python.exe"
    else:
        return venv_path / "bin" / "python"


def verify_code(args):
    """
    Run pylint on all Python code in experiments and lib directories, then run tests.
    """
    project_root = Path(__file__).resolve().parent.parent
    venv_path = project_root / "venv"

    # Check if virtual environment exists
    if not venv_path.exists():
        print("Error: Virtual environment not found. Run 'python bin/init.py' first.")
        sys.exit(1)

    # Get pylint path
    pylint_path = get_venv_pylint(venv_path)

    # Directories to check
    directories = ["lib"]
    targets = []

    for dir_name in directories:
        dir_path = project_root / dir_name
        if dir_path.exists():
            # Find all Python files recursively
            python_files = list(dir_path.rglob("*.py"))
            targets.extend(python_files)
        else:
            print(f"Warning: Directory '{dir_name}' does not exist, skipping...")

    pylint_failed = False
    if not targets:
        print("No Python files found in the project.")
    else:
        # Run pylint (use project config if present)
        rcfile = project_root / "etc" / "pylintrc"
        cmd = [str(pylint_path)]
        if rcfile.exists():
            cmd.extend(["--rcfile", str(rcfile)])
        cmd.extend([str(f) for f in targets])
        if args.args:
            cmd.extend(args.args)
        print(f"Running pylint on {len(targets)} file(s)...")
        try:
            result = subprocess.run(cmd, cwd=str(project_root))
            pylint_failed = result.returncode != 0
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            sys.exit(130)
        except Exception as e:
            print(f"Error running pylint: {e}")
            sys.exit(1)

    # Run tests (full suite)
    print()
    test_args = argparse.Namespace(skip_unit=False, skip_integration=False)
    test_failed = run_tests(test_args) != 0
    sys.exit(1 if (pylint_failed or test_failed) else 0)


def run_tests(args):
    """
    Run unit and/or integration tests via unittest discover.
    Returns exit code (0 on success, 1 on failure).
    """
    project_root = Path(__file__).resolve().parent.parent
    venv_path = project_root / "venv"

    if not venv_path.exists():
        print("Error: Virtual environment not found. Run 'python bin/init.py' first.")
        return 1

    python_path = get_venv_python(venv_path)
    run_unit = not args.skip_unit
    run_integration = not args.skip_integration

    if not run_unit and not run_integration:
        print("Nothing to run (both test suites skipped).")
        return 0

    if run_unit and run_integration:
        # Run all tests as one suite (discover recurses into tests/unit and tests/integration)
        result = subprocess.run(
            [str(python_path), "-m", "unittest", "discover", "-s", str(project_root / "tests"), "-p", "test_*.py"],
            cwd=str(project_root),
        )
        return result.returncode

    # Run only one suite
    start_dir = str(project_root / "tests" / ("unit" if run_unit else "integration"))
    result = subprocess.run(
        [str(python_path), "-m", "unittest", "discover", "-s", start_dir, "-p", "test_*.py"],
        cwd=str(project_root),
    )
    return result.returncode


def main():
    """Main entry point for the control script."""
    parser = argparse.ArgumentParser(
        description="Control script for algorithmic trading strategies project",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Verify command
    verify_parser = subparsers.add_parser(
        "verify",
        help="Run pylint on all code in experiments and lib directories"
    )
    verify_parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Additional arguments to pass to pylint"
    )
    verify_parser.set_defaults(func=verify_code)

    # Test command
    test_parser = subparsers.add_parser(
        "test",
        help="Run unit and/or integration tests",
    )
    test_parser.add_argument(
        "-su",
        "--skip-unit",
        action="store_true",
        help="Skip unit tests",
    )
    test_parser.add_argument(
        "-si",
        "--skip-integration",
        action="store_true",
        help="Skip integration tests",
    )
    test_parser.set_defaults(func=run_tests)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute the command (test returns exit code; others may sys.exit)
    exit_code = args.func(args)
    if exit_code is not None:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()

