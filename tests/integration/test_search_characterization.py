"""Recorded PostgreSQL behavior of the `simple` full-text configuration (PostgreSQL 16).

These tests pin what the server actually does with the technical terms this project cares about.
They are characterization, not a quality claim: the document builder was shaped by them, and a
change in server behavior must fail here rather than silently change search results.
"""

import pytest
from sqlalchemy import text

pytestmark = pytest.mark.integration

# input text -> `to_tsvector('simple', text)::text` observed on PostgreSQL 16.15
TOKENIZATION = [
    ("8GB", "'8gb':1"),
    ("8 GB", "'8':1 'gb':2"),
    ("256GB", "'256gb':1"),
    ("14.0-inch", "'14.0':1 'inch':2"),
    ("i5", "'i5':1"),
    ("LP101", "'lp101':1"),
    ("In-ear", "'ear':3 'in':2 'in-ear':1"),
    ("wireless_2_4ghz", "'2':2 '4ghz':3 'wireless':1"),
    ("2.4GHz", "'2.4':1 'ghz':2"),
    ("2.4ghz wireless", "'2.4':1 'ghz':2 'wireless':3"),
    ("shoe", "'shoe':1"),
    ("shoes", "'shoes':1"),
    ("headphone", "'headphone':1"),
    ("headphones", "'headphones':1"),
    ("WH-1000XM5", "'1000xm5':3 'wh':2 'wh-1000xm5':1"),
    ("WH1000XM5", "'wh1000xm5':1"),
    ("NVMe", "'nvme':1"),
    ("1TB", "'1tb':1"),
    ("5000mAh", "'5000mah':1"),
    ("ke liye", "'ke':1 'liye':2"),
]


@pytest.mark.parametrize(("source", "expected"), TOKENIZATION)
def test_simple_configuration_tokenization(seeded_engine, source, expected):
    with seeded_engine.connect() as conn:
        got = conn.execute(
            text("SELECT CAST(to_tsvector('simple', CAST(:s AS text)) AS text)"), {"s": source}
        ).scalar_one()
    assert got == expected


def test_plural_forms_are_not_conflated_by_simple(seeded_engine):
    """No stemming: this is why V0 indexes explicit singular/plural forms of known taxonomy
    values (documented limitation: other inflections are not matched)."""
    with seeded_engine.connect() as conn:
        for singular, plural in (("shoe", "shoes"), ("headphone", "headphones")):
            matched = conn.execute(
                text(
                    "SELECT to_tsvector('simple', CAST(:p AS text)) "
                    "@@ plainto_tsquery('simple', CAST(:s AS text))"
                ),
                {"p": plural, "s": singular},
            ).scalar_one()
            assert matched is False


@pytest.mark.parametrize(
    ("query", "tsquery"),
    [
        ("8gb laptop", "'8gb' & 'laptop'"),
        ("8 gb laptop", "'8' & 'gb' & 'laptop'"),
        ("nike -shoes", "'nike' & 'shoes'"),
        ('"nike', "'nike'"),
        ("hp-laptop", "'hp-laptop' & 'hp' & 'laptop'"),
        ("256gb ssd laptop", "'256gb' & 'ssd' & 'laptop'"),
        ("!!!", ""),
    ],
)
def test_plainto_tsquery_behavior(seeded_engine, query, tsquery):
    with seeded_engine.connect() as conn:
        got = conn.execute(
            text("SELECT CAST(plainto_tsquery('simple', CAST(:q AS text)) AS text)"), {"q": query}
        ).scalar_one()
    assert got == tsquery
