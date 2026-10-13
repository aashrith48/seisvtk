"""Shared fixtures: locate the bundled F3 demo data, write small synthetic tables."""
from pathlib import Path

import pytest

F3 = Path(__file__).resolve().parent.parent / "Testing" / "F3_Demo_2023"


@pytest.fixture(scope="session")
def f3() -> Path:
    if not F3.exists():
        pytest.skip(f"bundled F3 data not present at {F3}")
    return F3


@pytest.fixture
def f3_las(f3) -> Path:
    path = f3 / "wells" / "logs" / "F02-1_logs.las"
    if not path.exists():
        pytest.skip(f"missing {path}")
    return path


@pytest.fixture
def f3_survey(f3) -> Path:
    path = f3 / "meta" / "survey.txt"
    if not path.exists():
        pytest.skip(f"missing {path}")
    return path
