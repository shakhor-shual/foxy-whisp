import pytest
import os
import sys
from pathlib import Path

# More robust project root detection
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for all tests"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    yield
    # Cleanup after each test
    logging.getLogger().handlers.clear()
