"""Test suite for Milestone 1 Delivery Verification Gate."""
from pathlib import Path
import pytest

from src.config import PROJECT_ROOT


def test_verify_milestone_script_exists():
    """Verify that scripts/verify_milestone_1.py exists."""
    script_path = PROJECT_ROOT / "scripts" / "verify_milestone_1.py"
    assert script_path.exists(), f"Verification script not found at {script_path}"


def test_run_verification_succeeds():
    """Verify that run_verification() executes without error and returns 0."""
    from scripts.verify_milestone_1 import run_verification

    exit_code = run_verification()
    assert exit_code == 0, f"run_verification() failed with exit code {exit_code}"
