"""Smoke tests verifying that remaining unbuilt components raise NotImplementedError per Rule 2.2."""

import pytest

from src.models.calibrate import compute_ece


def test_compute_ece_stub_raises():
    with pytest.raises(NotImplementedError):
        compute_ece(None, None)
