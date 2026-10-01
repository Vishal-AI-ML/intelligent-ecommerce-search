"""Helpers for the M3 integration tests (scratch databases only).

The lexical oracle below is deliberately independent of the code under test: it never calls the
production document builder or any PostgreSQL full-text function. It rebuilds each product's
searchable words from the normalized catalog columns with its own explicit rules and tokenizes
with a plain regular expression.
"""

import re

from sqlalchemy import text

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Written independently of ecommerce_search.search.documents on purpose.
ORACLE_CATEGORY_WORDS = {
    "laptop": ["laptop", "laptops"],
    "phone": ["phone", "phones"],
    "shoes": ["shoe", "shoes"],
    "headphones": ["headphone", "headphones"],
}
ORACLE_SUBCATEGORY_WORDS = {"smartphone": ["smartphones"], "ultrabook": ["ultrabooks"]}
ORACLE_CONNECTIVITY = {
    "bluetooth": "bluetooth",
    "wired": "wired",
    "usb": "usb",
    "wireless_2_4ghz": "2.4ghz wireless",
}

SPEC_QUERY = {
    "laptop": "SELECT * FROM laptop_specs",
    "phone": "SELECT * FROM phone_specs",
    "shoes": "SELECT * FROM shoe_specs",
    "headphones": "SELECT * FROM headphone_specs",
}


def query_terms(query: str) -> list[str]:
    return TOKEN_RE.findall(query.lower())


def _capacity(gb) -> str:
    words = f"{gb}gb"
    if gb >= 1024 and gb % 1024 == 0:
        words += f" {gb // 1024}tb"
    return words


def _attribute_words(category: str, spec: dict) -> str:
    parts: list[str] = []
    if category in ("laptop", "phone"):
        if spec["ram_gb"] is not None:
            parts.append(f"{spec['ram_gb']}gb ram")
        if spec["storage_gb"] is not None:
            parts.append(_capacity(spec["storage_gb"]))
    if category == "laptop":
        for key in ("storage_type", "storage_interface"):
            if spec[key]:
                parts.append(spec[key].lower())
        parts += [spec[k] for k in ("processor", "gpu", "operating_system") if spec[k]]
    elif category == "phone":
        parts += [spec[k] for k in ("camera", "operating_system") if spec[k]]
        if spec["battery_mah"] is not None:
            parts.append(f"{spec['battery_mah']}mah")
    elif category == "shoes":
        parts += [spec[k] for k in ("color", "material", "gender") if spec[k]]
    else:
        if spec["wireless"] is True:
            parts.append("wireless")
        if spec["connectivity"]:
            parts.append(ORACLE_CONNECTIVITY[spec["connectivity"]])
        if spec["anc"] is True:
            parts.append("anc")
    return " ".join(parts)


def oracle_tokens(engine) -> dict[str, set[str]]:
    """product_id -> the set of tokens a lexical match may use (own rules, no FTS)."""
    with engine.connect() as conn:
        products = {
            row["product_id"]: dict(row)
            for row in conn.execute(text("SELECT * FROM products")).mappings()
        }
        specs: dict[str, dict] = {}
        for sql in SPEC_QUERY.values():
            for row in conn.execute(text(sql)).mappings():
                specs[row["product_id"]] = dict(row)
    out: dict[str, set[str]] = {}
    for pid, product in products.items():
        category = product["category"]
        words = [
            product["title"],
            product["brand"],
            *ORACLE_CATEGORY_WORDS[category],
            product["subcategory"] or "",
            *ORACLE_SUBCATEGORY_WORDS.get(product["subcategory"] or "", []),
            _attribute_words(category, specs[pid]),
            product["description"] or "",
        ]
        out[pid] = set(TOKEN_RE.findall(" ".join(words).lower()))
    return out


def oracle_matches(tokens: dict[str, set[str]], query: str) -> set[str]:
    """Products containing every query term (strict AND)."""
    terms = set(query_terms(query))
    if not terms:
        return set()
    return {pid for pid, bag in tokens.items() if terms <= bag}


def assert_ranked(hits) -> None:
    """Scores non-negative and non-increasing; ties in ascending product_id; no duplicates."""
    ids = [h.product_id for h in hits]
    assert len(ids) == len(set(ids)), "duplicate product ids"
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    for hit in hits:
        assert hit.lexical_score >= 0
    for before, after in zip(hits, hits[1:], strict=False):
        assert before.lexical_score >= after.lexical_score
        if before.lexical_score == after.lexical_score:
            assert before.product_id < after.product_id


def plan_node_types(plan_json) -> list[str]:
    """Every 'Node Type' (with the index name when present) in an EXPLAIN (FORMAT JSON) plan."""
    found: list[str] = []

    def walk(node: dict) -> None:
        label = node["Node Type"]
        if "Index Name" in node:
            label += f" [{node['Index Name']}]"
        found.append(label)
        for child in node.get("Plans", []):
            walk(child)

    walk(plan_json[0]["Plan"])
    return found
