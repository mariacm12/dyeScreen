"""
Unit and regression test for the dyeScreen package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import dyeScreen


def test_dyeScreen_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "dyeScreen" in sys.modules
