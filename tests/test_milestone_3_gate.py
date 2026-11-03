"""Test suite for Milestone 3 Delivery Verification Gate."""
from pathlib import Path
import pytest

from src.config import PROJECT_ROOT


def test_verify_milestone_3_script_exists():
    """Verify that scripts/verify_milestone_3.py exists."""
    script_path = PROJECT_ROOT / "scripts" / "verify_milestone_3.py"
    assert script_path.exists(), f"Verification script not found at {script_path}"


def test_run_milestone_3_verification_succeeds():
    """Verify that run_milestone_3_verification() executes without error and returns 0."""
    from scripts.verify_milestone_3 import run_milestone_3_verification

    exit_code = run_milestone_3_verification()
    assert exit_code == 0, f"run_milestone_3_verification() failed with exit code {exit_code}"
