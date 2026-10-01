"""Local dense-embedding support (Milestone 4).

Nothing in this package imports torch or sentence-transformers at import time. The provider
module imports them lazily, on first load, and `fetch` is the only module that uses the network.
"""
