"""SQLAlchemy models. Importing this package registers every table on `Base.metadata`."""

from ecommerce_search.models.catalog import (
    CatalogDataset,
    CatalogReview,
    HeadphoneSpec,
    LaptopSpec,
    PhoneSpec,
    Product,
    RawCatalogRecord,
    ShoeSpec,
)

__all__ = [
    "CatalogDataset",
    "CatalogReview",
    "HeadphoneSpec",
    "LaptopSpec",
    "PhoneSpec",
    "Product",
    "RawCatalogRecord",
    "ShoeSpec",
]
