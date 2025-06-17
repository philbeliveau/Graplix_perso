# Pytest configuration for PII Extraction System
# Agent 5: DevOps & CI/CD Specialist

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from unittest.mock import MagicMock

# Test configuration
os.environ["TESTING"] = "true"
os.environ["LOG_LEVEL"] = "DEBUG"


@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture for test data directory."""
    return Path(__file__).parent / "fixtures" / "data"


@pytest.fixture(scope="session") 
def sample_documents(test_data_dir):
    """Fixture providing sample test documents."""
    return {
        "pdf_with_pii": test_data_dir / "sample_resume.pdf",
        "docx_with_pii": test_data_dir / "sample_form.docx", 
        "image_with_text": test_data_dir / "sample_id.jpg",
        "clean_document": test_data_dir / "clean_text.txt"
    }


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Fixture providing temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_aws_client():
    """Mock AWS S3 client for testing."""
    mock_client = MagicMock()
    mock_client.upload_file.return_value = None
    mock_client.download_file.return_value = None
    mock_client.list_objects_v2.return_value = {
        'Contents': [
            {'Key': 'test-file.pdf', 'Size': 1024},
            {'Key': 'test-file.docx', 'Size': 2048}
        ]
    }
    return mock_client


@pytest.fixture
def sample_pii_data():
    """Sample PII data for testing extractors."""
    return {
        "emails": ["john.doe@email.com", "jane.smith@company.org"],
        "phones": ["+1-555-123-4567", "(555) 987-6543"],
        "ssns": ["123-45-6789", "987-65-4321"],
        "names": ["John Doe", "Jane Smith"],
        "addresses": ["123 Main St, City, ST 12345"],
        "dates": ["01/01/1990", "December 25, 1985"]
    }


@pytest.fixture
def mock_ocr_result():
    """Mock OCR result for testing."""
    return {
        "text": "John Doe\nEmail: john.doe@email.com\nPhone: +1-555-123-4567",
        "confidence": 0.95,
        "bounding_boxes": [
            {"text": "John Doe", "bbox": [10, 10, 100, 30]},
            {"text": "john.doe@email.com", "bbox": [10, 40, 200, 60]}
        ]
    }


@pytest.fixture
def test_config():
    """Test configuration dictionary."""
    return {
        "storage": {
            "type": "local",
            "local_path": "/tmp/test_data"
        },
        "processing": {
            "ocr_engine": "tesseract",
            "languages": ["en", "fr"]
        },
        "extractors": {
            "rule_based": {
                "enabled": True,
                "confidence_threshold": 0.8
            }
        },
        "logging": {
            "level": "DEBUG",
            "format": "simple"
        }
    }


# Pytest markers for different test types
pytest_plugins = ["pytest_asyncio"]

# Custom markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests that test individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests that test component interactions"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests that test complete workflows"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and benchmark tests"
    )
    config.addinivalue_line(
        "markers", "security: Security-focused tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take more than 5 seconds to run"
    )


# Test collection customization
def pytest_collection_modifyitems(config, items):
    """Modify collected test items."""
    # Add slow marker to tests that take a long time
    for item in items:
        if "e2e" in item.keywords or "integration" in item.keywords:
            item.add_marker(pytest.mark.slow)
            
        # Skip slow tests in fast test runs
        if config.getoption("--fast"):
            if "slow" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Skipping slow test in fast mode"))


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--fast", 
        action="store_true", 
        default=False, 
        help="Run fast tests only"
    )
    parser.addoption(
        "--performance", 
        action="store_true", 
        default=False, 
        help="Run performance tests"
    )


# Cleanup after tests
@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """Automatically cleanup test artifacts after each test."""
    yield
    # Cleanup logic here
    temp_files = Path("/tmp").glob("pii_test_*")
    for temp_file in temp_files:
        if temp_file.exists():
            if temp_file.is_file():
                temp_file.unlink()
            elif temp_file.is_dir():
                import shutil
                shutil.rmtree(temp_file)