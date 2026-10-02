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
from ecommerce_search.models.embeddings import ProductEmbedding
from ecommerce_search.models.search import ProductSearchDocument

__all__ = [
    "CatalogDataset",
    "CatalogReview",
    "HeadphoneSpec",
    "LaptopSpec",
    "PhoneSpec",
    "Product",
    "ProductEmbedding",
    "ProductSearchDocument",
    "RawCatalogRecord",
    "ShoeSpec",
]
