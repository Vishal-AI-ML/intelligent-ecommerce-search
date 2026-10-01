import pytest

from ecommerce_search.config import Settings

TEST_PASSWORD = "unit-test-password"  # noqa: S105 - test fixture value, not a real credential


@pytest.fixture
def make_settings():
    def _make(**overrides) -> Settings:
        values = {"postgres_password": TEST_PASSWORD, "app_env": "test"}
        values.update(overrides)
        # _env_file=None keeps a developer's local .env from leaking into tests.
        return Settings(_env_file=None, **values)

    return _make


_BASE_RAW = {
    "laptop": {
        "product_id": "T-LAP-1",
        "seller_id": "test-seller",
        "title": "HP Testline 15.6-inch Laptop - Intel Core i5, 16GB RAM, 512GB SSD Model T1",
        "description": "Test laptop.",
        "category": "laptop",
        "subcategory": "business",
        "brand": "HP",
        "price": 55000,
        "currency": "INR",
        "rating": 4.2,
        "review_count": 10,
        "availability": "in_stock",
        "source_type": "synthetic",
        "is_synthetic": True,
        "ram_gb": 16,
        "storage_gb": 512,
        "storage_type": "SSD",
        "storage_interface": "NVME",
        "processor": "Intel Core i5",
        "gpu": None,
        "screen_size_inches": 15.6,
        "operating_system": "Windows 11",
        "weight_kg": 1.7,
    },
    "phone": {
        "product_id": "T-PHN-1",
        "seller_id": "test-seller",
        "title": "Samsung Testphone (8GB RAM, 128GB Storage) Model T2",
        "description": "Test phone.",
        "category": "phone",
        "subcategory": "smartphone",
        "brand": "Samsung",
        "price": 20999,
        "currency": "INR",
        "rating": 4.0,
        "review_count": 5,
        "availability": "in_stock",
        "source_type": "synthetic",
        "is_synthetic": True,
        "ram_gb": 8,
        "storage_gb": 128,
        "camera": "50 MP dual camera",
        "battery_mah": 5000,
        "screen_size_inches": 6.6,
        "operating_system": "Android",
    },
    "shoes": {
        "product_id": "T-SHO-1",
        "seller_id": "test-seller",
        "title": "Nike Teststride Running Shoes for Men - Black Model T3",
        "description": "Test shoes.",
        "category": "shoes",
        "subcategory": "running",
        "brand": "Nike",
        "price": 4999,
        "currency": "INR",
        "rating": 4.4,
        "review_count": 20,
        "availability": "in_stock",
        "source_type": "synthetic",
        "is_synthetic": True,
        "size": 9.0,
        "size_system": "UK",
        "color": "Black",
        "material": "Mesh",
        "gender": "men",
    },
    "headphones": {
        "product_id": "T-HDP-1",
        "seller_id": "test-seller",
        "title": "Sony Testwave Over-ear Wireless Headphones Model T4",
        "description": "Test headphones.",
        "category": "headphones",
        "subcategory": "over-ear",
        "brand": "Sony",
        "price": 9999,
        "currency": "INR",
        "rating": 4.1,
        "review_count": 8,
        "availability": "in_stock",
        "source_type": "synthetic",
        "is_synthetic": True,
        "wireless": True,
        "anc": False,
        "battery_life_hours": 30,
        "connectivity": "bluetooth",
    },
}


@pytest.fixture
def raw_record():
    """Factory for a valid raw record: raw_record("laptop", ram_gb="16 GB", drop=["gpu"])."""

    def _make(kind: str = "laptop", *, drop=(), **overrides) -> dict:
        record = dict(_BASE_RAW[kind])
        record.update(overrides)
        for key in drop:
            record.pop(key, None)
        return record

    return _make


@pytest.fixture
def parsed(raw_record):
    from ecommerce_search.catalog.schemas import parse_raw

    def _make(kind: str = "laptop", **kwargs):
        return parse_raw(raw_record(kind, **kwargs))

    return _make
