import os

import pytest

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "structures_small.json")


@pytest.fixture
def structures_path():
    return FIXTURE


def has_lewis_solver() -> bool:
    try:
        import lewis_geometry_solver  # noqa: F401
        import openbabel  # noqa: F401
        import rdkit  # noqa: F401
    except ImportError:
        return False
    return True


needs_lewis = pytest.mark.skipif(not has_lewis_solver(), reason="lewis_geometry_solver / rdkit / openbabel not installed")
