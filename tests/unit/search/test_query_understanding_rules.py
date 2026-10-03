"""Milestone 6 rule table: exact fields, evidence, ambiguities, conflicts and unresolved spans.

Every expectation comes from the approved M6 plan (§5) and the MASTER_PLAN/spec examples, not
from the implementation. Evidence is written as (rule-or-reason, canonical value, span text) in
the declared contract order.
"""

from dataclasses import dataclass, field

import pytest

from ecommerce_search.query_understanding import QueryUnderstanding, parse_normalized_query

pytestmark = pytest.mark.usefixtures("no_socket_connect")

_FIELDS = (
    "category",
    "brand",
    "ram_gb",
    "storage_gb",
    "storage_type",
    "min_price",
    "max_price",
    "semantic_intent",
)


@dataclass(frozen=True)
class Case:
    query: str
    fields: dict[str, object] = field(default_factory=dict)
    terms: tuple[tuple[str, str, str], ...] = ()
    ambiguities: tuple[tuple[str, str | None, str], ...] = ()
    conflicts: tuple[tuple[str, str, tuple[str, ...]], ...] = ()
    unresolved: tuple[str, ...] = ()


def summarize(understanding: QueryUnderstanding) -> Case:
    data = understanding.model_dump(mode="json")
    fields = {name: data[name] for name in _FIELDS if data[name] is not None}
    fields |= {name: value for name, value in data["attributes"].items() if value is not None}
    return Case(
        query=understanding.raw_query,
        fields=fields,
        terms=tuple((t.rule.value, t.value, t.span.text) for t in understanding.matched_terms),
        ambiguities=tuple(
            (a.reason.value, a.value, a.span.text) for a in understanding.ambiguities
        ),
        conflicts=tuple(
            (c.field.value, c.reason.value, tuple(t.span.text for t in c.candidates))
            for c in understanding.conflicts
        ),
        unresolved=tuple(span.text for span in understanding.unresolved),
    )


SSD = ("storage_type_term", "SSD", "ssd")

# MASTER_PLAN §8/§9 and spec FR-QU-3, §7.1 and §14 (Q5, Q6, Q9) examples.
SPEC_CASES = [
    Case(
        "hp laptop 8gb 256 ssd under 40k",
        {
            "category": "laptop",
            "brand": "HP",
            "ram_gb": 8,
            "storage_gb": 256,
            "storage_type": "SSD",
            "max_price": "40000.00",
        },
        terms=(
            ("brand_term", "HP", "hp"),
            ("category_term", "laptop", "laptop"),
            ("ram_capacity_paired", "8", "8gb"),
            ("storage_capacity_implicit_gb", "256", "256 ssd"),
            SSD,
            ("price_upper_bound", "40000.00", "under 40k"),
        ),
    ),
    Case(
        "laptop 8gb 256 ssd",
        {"category": "laptop", "ram_gb": 8, "storage_gb": 256, "storage_type": "SSD"},
        terms=(
            ("category_term", "laptop", "laptop"),
            ("ram_capacity_paired", "8", "8gb"),
            ("storage_capacity_implicit_gb", "256", "256 ssd"),
            SSD,
        ),
    ),
    Case(
        "coding laptop",
        {"category": "laptop", "semantic_intent": "coding"},
        terms=(("intent_term", "coding", "coding"), ("category_term", "laptop", "laptop")),
    ),
    Case("8gb", ambiguities=(("bare_capacity", "8", "8gb"),)),
    Case("8 gb", ambiguities=(("bare_capacity", "8", "8 gb"),)),
    Case("16gb", ambiguities=(("bare_capacity", "16", "16gb"),)),
    Case("256gb", ambiguities=(("bare_capacity", "256", "256gb"),)),
    Case("512gb", ambiguities=(("bare_capacity", "512", "512gb"),)),
    Case("1tb", ambiguities=(("bare_capacity", "1024", "1tb"),)),
    Case("40k", ambiguities=(("price_without_bound", "40000.00", "40k"),)),
    Case("₹40000", ambiguities=(("price_without_bound", "40000.00", "₹40000"),)),
    Case("rs 40000", ambiguities=(("price_without_bound", "40000.00", "rs 40000"),)),
    Case(
        "50k ke andar",
        {"max_price": "50000.00"},
        terms=(("price_upper_bound", "50000.00", "50k ke andar"),),
    ),
    Case(
        "phone 20k ke under",
        {"category": "phone", "max_price": "20000.00"},
        terms=(
            ("category_term", "phone", "phone"),
            ("price_upper_bound", "20000.00", "20k ke under"),
        ),
    ),
    Case(
        "coding ke liye laptop",
        {"category": "laptop", "semantic_intent": "coding"},
        terms=(
            ("intent_term", "coding", "coding ke liye"),
            ("category_term", "laptop", "laptop"),
        ),
    ),
    Case(
        "gaming ke liye laptop",
        {"category": "laptop", "semantic_intent": "gaming"},
        terms=(
            ("intent_term", "gaming", "gaming ke liye"),
            ("category_term", "laptop", "laptop"),
        ),
    ),
    Case(
        "sasta phone",
        {"category": "phone", "price_preference": "low"},
        terms=(("price_preference_term", "low", "sasta"), ("category_term", "phone", "phone")),
    ),
    Case(
        "laptop under 50k",
        {"category": "laptop", "max_price": "50000.00"},
        terms=(
            ("category_term", "laptop", "laptop"),
            ("price_upper_bound", "50000.00", "under 50k"),
        ),
    ),
    # Q9: an `ssd` query sets the medium only; NVMe is an interface on SSDs and is not excluded.
    Case(
        "ssd laptop",
        {"category": "laptop", "storage_type": "SSD"},
        terms=(SSD, ("category_term", "laptop", "laptop")),
    ),
    Case(
        "nvme laptop",
        {"category": "laptop", "storage_type": "SSD", "storage_interface": "NVME"},
        terms=(
            ("storage_interface_term", "NVME", "nvme"),
            ("storage_type_implied_by_interface", "SSD", "nvme"),
            ("category_term", "laptop", "laptop"),
        ),
    ),
]

LEXICON_CASES = [
    Case(
        "smartphones mobile",
        {"category": "phone"},
        terms=(
            ("category_term", "phone", "smartphones"),
            ("category_term", "phone", "mobile"),
        ),
    ),
    Case(
        "earbuds earphones headphone",
        {"category": "headphones"},
        terms=(
            ("category_term", "headphones", "earbuds"),
            ("category_term", "headphones", "earphones"),
            ("category_term", "headphones", "headphone"),
        ),
    ),
    Case(
        "shoe shoes",
        {"category": "shoes"},
        terms=(("category_term", "shoes", "shoe"), ("category_term", "shoes", "shoes")),
    ),
    Case(
        "boat earbuds",
        {"brand": "boAt", "category": "headphones"},
        terms=(("brand_term", "boAt", "boat"), ("category_term", "headphones", "earbuds")),
    ),
    Case(
        "one plus phone",
        {"brand": "OnePlus", "category": "phone"},
        terms=(("brand_term", "OnePlus", "one plus"), ("category_term", "phone", "phone")),
    ),
    Case("hewlett-packard", {"brand": "HP"}, terms=(("brand_term", "HP", "hewlett-packard"),)),
    Case("hewlett packard", unresolved=("hewlett", "packard")),
    Case("notebook", unresolved=("notebook",)),
    Case("iphone 15", unresolved=("iphone", "15")),
    Case(
        "for gaming laptop",
        {"category": "laptop", "semantic_intent": "gaming"},
        terms=(("intent_term", "gaming", "for gaming"), ("category_term", "laptop", "laptop")),
    ),
    Case("student", {"semantic_intent": "student"}, terms=(("intent_term", "student", "student"),)),
    Case(
        "lightweight",
        {"semantic_intent": "lightweight"},
        terms=(("intent_term", "lightweight", "lightweight"),),
    ),
    Case("premium", {"semantic_intent": "premium"}, terms=(("intent_term", "premium", "premium"),)),
    Case("comfort", {"semantic_intent": "comfort"}, terms=(("intent_term", "comfort", "comfort"),)),
    Case("travel", {"semantic_intent": "travel"}, terms=(("intent_term", "travel", "travel"),)),
    Case("office", {"semantic_intent": "office"}, terms=(("intent_term", "office", "office"),)),
    Case(
        "laptop ke liye",
        {"category": "laptop"},
        terms=(("category_term", "laptop", "laptop"),),
        unresolved=("ke", "liye"),
    ),
    Case(
        "for laptop",
        {"category": "laptop"},
        terms=(("category_term", "laptop", "laptop"),),
        unresolved=("for",),
    ),
    Case("hdd", {"storage_type": "HDD"}, terms=(("storage_type_term", "HDD", "hdd"),)),
    Case("ssd", {"storage_type": "SSD"}, terms=(SSD,)),
    Case("ram", unresolved=("ram",)),
    Case("memory storage", unresolved=("memory", "storage")),
]

# A14 capacity roles and the R6 pairing convention.
CAPACITY_CASES = [
    Case(
        "8gb ram 256gb ssd",
        {"ram_gb": 8, "storage_gb": 256, "storage_type": "SSD"},
        terms=(
            ("ram_capacity", "8", "8gb ram"),
            ("storage_capacity", "256", "256gb ssd"),
            SSD,
        ),
    ),
    Case("ram 8gb", {"ram_gb": 8}, terms=(("ram_capacity", "8", "ram 8gb"),)),
    Case(
        "8gb ssd",
        {"storage_gb": 8, "storage_type": "SSD"},
        terms=(("storage_capacity", "8", "8gb ssd"), SSD),
    ),
    Case(
        "ram 8gb ssd",
        {"storage_type": "SSD"},
        terms=(SSD,),
        ambiguities=(("capacity_multiple_roles", "8", "ram 8gb ssd"),),
    ),
    Case("8gb ram storage", ambiguities=(("capacity_multiple_roles", "8", "8gb ram storage"),)),
    Case(
        "storage 512gb ssd",
        {"storage_gb": 512, "storage_type": "SSD"},
        terms=(("storage_capacity", "512", "storage 512gb ssd"), SSD),
    ),
    Case("memory 8gb", ambiguities=(("ambiguous_memory_capacity", "8", "memory 8gb"),)),
    Case(
        "8gb memory 256gb ssd",
        {"storage_gb": 256, "storage_type": "SSD"},
        terms=(("storage_capacity", "256", "256gb ssd"), SSD),
        ambiguities=(("ambiguous_memory_capacity", "8", "8gb memory"),),
    ),
    Case(
        "8gb ram 16gb",
        {"ram_gb": 8},
        terms=(("ram_capacity", "8", "8gb ram"),),
        ambiguities=(("bare_capacity", "16", "16gb"),),
    ),
    Case(
        "8gb-ram 1tb storage",
        {"ram_gb": 8, "storage_gb": 1024},
        terms=(("ram_capacity", "8", "8gb-ram"), ("storage_capacity", "1024", "1tb storage")),
    ),
    Case(
        "laptop 16gb",
        {"category": "laptop"},
        terms=(("category_term", "laptop", "laptop"),),
        ambiguities=(("bare_capacity", "16", "16gb"),),
    ),
    # R6 positive.
    Case(
        "8gb 256gb ssd",
        {"ram_gb": 8, "storage_gb": 256, "storage_type": "SSD"},
        terms=(
            ("ram_capacity_paired", "8", "8gb"),
            ("storage_capacity", "256", "256gb ssd"),
            SSD,
        ),
    ),
    # R6 negatives: no storage, larger leftover, two leftovers, RAM present, memory or
    # multiple-roles ambiguity present, conflicted storage.
    Case(
        "8gb laptop",
        {"category": "laptop"},
        terms=(("category_term", "laptop", "laptop"),),
        ambiguities=(("bare_capacity", "8", "8gb"),),
    ),
    Case(
        "512gb 256gb ssd",
        {"storage_gb": 256, "storage_type": "SSD"},
        terms=(("storage_capacity", "256", "256gb ssd"), SSD),
        ambiguities=(("bare_capacity", "512", "512gb"),),
    ),
    Case(
        "8gb 16gb 256gb ssd",
        {"storage_gb": 256, "storage_type": "SSD"},
        terms=(("storage_capacity", "256", "256gb ssd"), SSD),
        ambiguities=(("bare_capacity", "8", "8gb"), ("bare_capacity", "16", "16gb")),
    ),
    Case(
        "8gb ram 16gb 256gb ssd",
        {"ram_gb": 8, "storage_gb": 256, "storage_type": "SSD"},
        terms=(
            ("ram_capacity", "8", "8gb ram"),
            ("storage_capacity", "256", "256gb ssd"),
            SSD,
        ),
        ambiguities=(("bare_capacity", "16", "16gb"),),
    ),
    Case(
        "8gb memory 16gb 256gb ssd",
        {"storage_gb": 256, "storage_type": "SSD"},
        terms=(("storage_capacity", "256", "256gb ssd"), SSD),
        ambiguities=(
            ("ambiguous_memory_capacity", "8", "8gb memory"),
            ("bare_capacity", "16", "16gb"),
        ),
    ),
    Case(
        "8gb ram storage 16gb 256gb ssd",
        {"storage_gb": 256, "storage_type": "SSD"},
        terms=(("storage_capacity", "256", "256gb ssd"), SSD),
        ambiguities=(
            ("capacity_multiple_roles", "8", "8gb ram storage"),
            ("bare_capacity", "16", "16gb"),
        ),
    ),
    Case(
        "8gb 256gb ssd 512gb ssd",
        {"storage_type": "SSD"},
        terms=(SSD, SSD),
        ambiguities=(("bare_capacity", "8", "8gb"),),
        conflicts=(("storage_gb", "repeated_different_values", ("256gb ssd", "512gb ssd")),),
    ),
    # S3b implicit GB guards.
    Case("256 ram", unresolved=("256", "ram")),
    Case("256 memory", unresolved=("256", "memory")),
    Case("i5 ssd", {"storage_type": "SSD"}, terms=(SSD,), unresolved=("i5",)),
    Case(
        "512 ssd",
        {"storage_gb": 512, "storage_type": "SSD"},
        terms=(("storage_capacity_implicit_gb", "512", "512 ssd"), SSD),
    ),
    Case(
        "₹256 ssd",
        {"storage_type": "SSD"},
        terms=(SSD,),
        ambiguities=(("price_without_bound", "256.00", "₹256"),),
    ),
]

# A13 / U1 unsupported numeric complexes: one exact covering span, never a partial field.
UNSUPPORTED_NUMERIC_CASES = [
    Case("₹40,000", ambiguities=(("unsupported_grouped_number", None, "₹40,000"),)),
    Case("rs 40,000", ambiguities=(("unsupported_grouped_number", None, "rs 40,000"),)),
    Case("inr 40,000", ambiguities=(("unsupported_grouped_number", None, "inr 40,000"),)),
    Case("under 40,000", ambiguities=(("unsupported_grouped_number", None, "under 40,000"),)),
    Case("₹40, 000", ambiguities=(("unsupported_grouped_number", None, "₹40, 000"),)),
    Case("40,000", ambiguities=(("unsupported_grouped_number", None, "40,000"),)),
    Case("rs 40 000", ambiguities=(("unsupported_numeric_compound", None, "rs 40 000"),)),
    Case("20k-40k", ambiguities=(("unsupported_range", None, "20k-40k"),)),
    Case("under 20k-40k", ambiguities=(("unsupported_range", None, "under 20k-40k"),)),
    Case("20k – 40k", ambiguities=(("unsupported_range", None, "20k – 40k"),)),
    Case("20k~40k", ambiguities=(("unsupported_range", None, "20k~40k"),)),
    Case("between 20k and 40k", ambiguities=(("unsupported_range", None, "between 20k and 40k"),)),
    Case("from 20k to 40k", ambiguities=(("unsupported_range", None, "from 20k to 40k"),)),
    Case("8/256gb", ambiguities=(("unsupported_numeric_compound", None, "8/256gb"),)),
    Case(
        "8 256 ssd",
        {"storage_type": "SSD"},
        terms=(SSD,),
        ambiguities=(("unsupported_numeric_compound", None, "8 256"),),
    ),
    Case(
        "₹ 40,000 laptop under 50k",
        {"category": "laptop", "max_price": "50000.00"},
        terms=(
            ("category_term", "laptop", "laptop"),
            ("price_upper_bound", "50000.00", "under 50k"),
        ),
        ambiguities=(("unsupported_grouped_number", None, "₹ 40,000"),),
    ),
    # I8: the space-grouping rule needs a bare 3-digit right operand.
    Case(
        "₹40000 8gb laptop",
        {"category": "laptop"},
        terms=(("category_term", "laptop", "laptop"),),
        ambiguities=(
            ("price_without_bound", "40000.00", "₹40000"),
            ("bare_capacity", "8", "8gb"),
        ),
    ),
    Case(
        "20k-40k ke andar",
        ambiguities=(("unsupported_range", None, "20k-40k"),),
        unresolved=("ke", "andar"),
    ),
    Case(
        "between 20k",
        ambiguities=(("price_without_bound", "20000.00", "20k"),),
        unresolved=("between",),
    ),
    Case(
        "20k and 40k",
        ambiguities=(
            ("price_without_bound", "20000.00", "20k"),
            ("price_without_bound", "40000.00", "40k"),
        ),
        unresolved=("and",),
    ),
    Case(
        "laptop 8gb 16gb ssd",
        {"category": "laptop", "ram_gb": 8, "storage_gb": 16, "storage_type": "SSD"},
        terms=(
            ("category_term", "laptop", "laptop"),
            ("ram_capacity_paired", "8", "8gb"),
            ("storage_capacity", "16", "16gb ssd"),
            SSD,
        ),
    ),
]

# False positives that must set no field (A13).
FALSE_POSITIVE_CASES = [
    Case("8gbssd", unresolved=("8", "gbssd")),
    # A `D L` run whose word is not a GLUE unit (`gb`, `tb`, `k`) is unsupported as a whole: the
    # glued word is never an atom and the number is never a value.
    Case("8ssd", unresolved=("8", "ssd")),
    Case("ram 8ssd", unresolved=("ram", "8", "ssd")),
    Case(
        "16gaming laptop",
        {"category": "laptop"},
        terms=(("category_term", "laptop", "laptop"),),
        unresolved=("16", "gaming"),
    ),
    Case("5hp", unresolved=("5", "hp")),
    Case(
        "50wala phone",
        {"category": "phone"},
        terms=(("category_term", "phone", "phone"),),
        unresolved=("50", "wala"),
    ),
    Case("under ₹40kg", unresolved=("under", "₹", "40", "kg")),
    Case("rs 500mb", unresolved=("rs", "500", "mb")),
    Case("from20k to40k", unresolved=("from20k", "to40k")),
    Case("rs-40000", unresolved=("rs", "40000")),
    Case("₹-40000", unresolved=("₹", "40000")),
    Case(
        "under-40k",
        ambiguities=(("price_without_bound", "40000.00", "40k"),),
        unresolved=("under",),
    ),
]

# A7 price grammar.
PRICE_CASES = [
    Case("inr 40000", ambiguities=(("price_without_bound", "40000.00", "inr 40000"),)),
    Case("rs.40000", ambiguities=(("price_without_bound", "40000.00", "rs.40000"),)),
    Case("rs. 40000", ambiguities=(("price_without_bound", "40000.00", "rs. 40000"),)),
    Case("₹ 40000", ambiguities=(("price_without_bound", "40000.00", "₹ 40000"),)),
    Case("₹40k", ambiguities=(("price_without_bound", "40000.00", "₹40k"),)),
    Case("4k", ambiguities=(("price_without_bound", "4000.00", "4k"),)),
    Case("rs", unresolved=("rs",)),
    Case("inr", unresolved=("inr",)),
    Case(
        "from 20k",
        {"min_price": "20000.00"},
        terms=(("price_lower_bound", "20000.00", "from 20k"),),
    ),
    Case(
        "above 20k",
        {"min_price": "20000.00"},
        terms=(("price_lower_bound", "20000.00", "above 20k"),),
    ),
    Case(
        "over ₹20000",
        {"min_price": "20000.00"},
        terms=(("price_lower_bound", "20000.00", "over ₹20000"),),
    ),
    Case(
        "below 30k",
        {"max_price": "30000.00"},
        terms=(("price_upper_bound", "30000.00", "below 30k"),),
    ),
    Case(
        "upto 30k",
        {"max_price": "30000.00"},
        terms=(("price_upper_bound", "30000.00", "upto 30k"),),
    ),
    Case(
        "up to 30k",
        {"max_price": "30000.00"},
        terms=(("price_upper_bound", "30000.00", "up to 30k"),),
    ),
    Case(
        "within rs 30000",
        {"max_price": "30000.00"},
        terms=(("price_upper_bound", "30000.00", "within rs 30000"),),
    ),
    Case(
        "under 50k ke andar",
        {"max_price": "50000.00"},
        terms=(("price_upper_bound", "50000.00", "under 50k ke andar"),),
    ),
    Case("from hp", {"brand": "HP"}, terms=(("brand_term", "HP", "hp"),), unresolved=("from",)),
    Case("over ear", unresolved=("over", "ear")),
    Case(
        "under laptop",
        {"category": "laptop"},
        terms=(("category_term", "laptop", "laptop"),),
        unresolved=("under",),
    ),
    Case("ke andar", unresolved=("ke", "andar")),
    Case("under 40000", unresolved=("under", "40000")),
    Case("40000", unresolved=("40000",)),
    Case(
        "above 20k under 40k",
        {"min_price": "20000.00", "max_price": "40000.00"},
        terms=(
            ("price_lower_bound", "20000.00", "above 20k"),
            ("price_upper_bound", "40000.00", "under 40k"),
        ),
    ),
    Case(
        "above 40k under 40k",
        {"min_price": "40000.00", "max_price": "40000.00"},
        terms=(
            ("price_lower_bound", "40000.00", "above 40k"),
            ("price_upper_bound", "40000.00", "under 40k"),
        ),
    ),
    Case(
        "above 50k under 40k",
        conflicts=(
            ("max_price", "min_price_exceeds_max_price", ("under 40k",)),
            ("min_price", "min_price_exceeds_max_price", ("above 50k",)),
        ),
    ),
    Case(
        "under 30k under 40k",
        conflicts=(("max_price", "repeated_different_values", ("under 30k", "under 40k")),),
    ),
    # Two independently explicit markers around one price: `from 20k` is a lower bound and
    # `20k ke andar` an upper bound, so both bounds are 20000 and nothing is discarded.
    Case(
        "from 20k ke andar",
        {"min_price": "20000.00", "max_price": "20000.00"},
        terms=(
            ("price_lower_bound", "20000.00", "from 20k"),
            ("price_upper_bound", "20000.00", "20k ke andar"),
        ),
    ),
    Case(
        "laptop above ₹30000 ke under",
        {"category": "laptop", "min_price": "30000.00", "max_price": "30000.00"},
        terms=(
            ("category_term", "laptop", "laptop"),
            ("price_lower_bound", "30000.00", "above ₹30000"),
            ("price_upper_bound", "30000.00", "₹30000 ke under"),
        ),
    ),
    Case(
        "under 40k 40k ke andar",
        {"max_price": "40000.00"},
        terms=(
            ("price_upper_bound", "40000.00", "under 40k"),
            ("price_upper_bound", "40000.00", "40k ke andar"),
        ),
    ),
]

# A16 dependent storage interface and B4' repeated values.
CONFLICT_CASES = [
    Case(
        "nvme hdd",
        conflicts=(
            ("storage_interface", "dependent_on_conflicted_storage_type", ("nvme",)),
            ("storage_type", "repeated_different_values", ("nvme", "hdd")),
        ),
    ),
    Case(
        "nvme nvme",
        {"storage_type": "SSD", "storage_interface": "NVME"},
        terms=(
            ("storage_interface_term", "NVME", "nvme"),
            ("storage_type_implied_by_interface", "SSD", "nvme"),
            ("storage_interface_term", "NVME", "nvme"),
            ("storage_type_implied_by_interface", "SSD", "nvme"),
        ),
    ),
    Case(
        "nvme ssd",
        {"storage_type": "SSD", "storage_interface": "NVME"},
        terms=(
            ("storage_interface_term", "NVME", "nvme"),
            ("storage_type_implied_by_interface", "SSD", "nvme"),
            SSD,
        ),
    ),
    Case("ssd hdd", conflicts=(("storage_type", "repeated_different_values", ("ssd", "hdd")),)),
    Case(
        "hp dell laptop",
        {"category": "laptop"},
        terms=(("category_term", "laptop", "laptop"),),
        conflicts=(("brand", "repeated_different_values", ("hp", "dell")),),
    ),
    Case(
        "hp hewlett-packard",
        {"brand": "HP"},
        terms=(("brand_term", "HP", "hp"), ("brand_term", "HP", "hewlett-packard")),
    ),
    Case(
        "gaming coding laptop",
        {"category": "laptop"},
        terms=(("category_term", "laptop", "laptop"),),
        conflicts=(("semantic_intent", "repeated_different_values", ("gaming", "coding")),),
    ),
    Case(
        "laptop phone",
        conflicts=(("category", "repeated_different_values", ("laptop", "phone")),),
    ),
    Case(
        "nike laptop",
        {"category": "laptop", "brand": "Nike"},
        terms=(("brand_term", "Nike", "nike"), ("category_term", "laptop", "laptop")),
    ),
]

# A6 connectives.
CONNECTIVE_CASES = [
    Case("ka", unresolved=("ka",)),
    Case("wala", unresolved=("wala",)),
    Case("xyz wala", unresolved=("xyz", "wala")),
    Case(
        "50k wala phone",
        {"category": "phone"},
        terms=(("category_term", "phone", "phone"),),
        ambiguities=(("price_without_bound", "50000.00", "50k wala"),),
    ),
    Case(
        "gaming wala laptop",
        {"category": "laptop", "semantic_intent": "gaming"},
        terms=(("intent_term", "gaming", "gaming wala"), ("category_term", "laptop", "laptop")),
    ),
    Case(
        "hp ka laptop",
        {"brand": "HP", "category": "laptop"},
        terms=(("brand_term", "HP", "hp ka"), ("category_term", "laptop", "laptop")),
    ),
    Case(
        "256gb ssd wala",
        {"storage_gb": 256, "storage_type": "SSD"},
        terms=(
            ("storage_capacity", "256", "256gb ssd wala"),
            ("storage_type_term", "SSD", "ssd wala"),
        ),
    ),
    Case(
        "nvme wala",
        {"storage_type": "SSD", "storage_interface": "NVME"},
        terms=(
            ("storage_interface_term", "NVME", "nvme wala"),
            ("storage_type_implied_by_interface", "SSD", "nvme wala"),
        ),
    ),
    Case(
        "ssd hdd wala",
        conflicts=(("storage_type", "repeated_different_values", ("ssd", "hdd wala")),),
    ),
    Case(
        "wala laptop",
        {"category": "laptop"},
        terms=(("category_term", "laptop", "laptop"),),
        unresolved=("wala",),
    ),
    Case(
        "laptop-wala",
        {"category": "laptop"},
        terms=(("category_term", "laptop", "laptop"),),
        unresolved=("wala",),
    ),
]

# Unsupported in v1: each sets no partial value.
UNSUPPORTED_V1_CASES = [
    Case("2 lakh", unresolved=("2", "lakh")),
    Case("1 crore", unresolved=("1", "crore")),
    Case("50 hazar", unresolved=("50", "hazar")),
    Case("40000rs", unresolved=("40000", "rs")),
    Case("128gb rom", ambiguities=(("bare_capacity", "128", "128gb"),), unresolved=("rom",)),
    Case("cheap", unresolved=("cheap",)),
    Case("sasti", unresolved=("sasti",)),
    Case("saste", unresolved=("saste",)),
    Case("galaxy s24", unresolved=("galaxy", "s24")),
    Case(
        "anc headphones",
        {"category": "headphones"},
        terms=(("category_term", "headphones", "headphones"),),
        unresolved=("anc",),
    ),
    Case(
        "red shoes size 9",
        {"category": "shoes"},
        terms=(("category_term", "shoes", "shoes"),),
        unresolved=("red", "size", "9"),
    ),
    Case("8gb ka ram", ambiguities=(("bare_capacity", "8", "8gb ka"),), unresolved=("ram",)),
]

ALL_RULE_CASES = (
    SPEC_CASES
    + LEXICON_CASES
    + CAPACITY_CASES
    + UNSUPPORTED_NUMERIC_CASES
    + FALSE_POSITIVE_CASES
    + PRICE_CASES
    + CONFLICT_CASES
    + CONNECTIVE_CASES
    + UNSUPPORTED_V1_CASES
)


@pytest.mark.parametrize("case", ALL_RULE_CASES, ids=lambda case: case.query)
def test_rule_table(case: Case) -> None:
    assert summarize(parse_normalized_query(case.query)) == case


def test_rule_table_queries_are_unique() -> None:
    queries = [case.query for case in ALL_RULE_CASES]
    assert len(queries) == len(set(queries))


@pytest.mark.parametrize(
    "case", UNSUPPORTED_NUMERIC_CASES + FALSE_POSITIVE_CASES, ids=lambda case: case.query
)
def test_unsupported_numeric_expressions_never_set_a_partial_numeric_field(case: Case) -> None:
    understanding = parse_normalized_query(case.query)
    unsupported = {
        "unsupported_grouped_number",
        "unsupported_range",
        "unsupported_numeric_compound",
    }
    complexes = [a.span for a in understanding.ambiguities if a.reason.value in unsupported]
    numeric = {"ram_gb", "storage_gb", "min_price", "max_price"}
    for term in understanding.matched_terms:
        if term.field.value in numeric:
            assert all(
                term.span.end <= span.start or span.end <= term.span.start for span in complexes
            )


def test_no_dependent_or_conflicted_field_is_null_without_a_conflict() -> None:
    for case in ALL_RULE_CASES:
        understanding = parse_normalized_query(case.query)
        data = understanding.model_dump(mode="json")
        values = {**data, **data["attributes"]}
        conflicted = {conflict.field.value for conflict in understanding.conflicts}
        for conflict in understanding.conflicts:
            assert values[conflict.field.value] is None
        stated = {term.field.value for term in understanding.matched_terms}
        for name in stated:
            assert values[name] is not None or name in conflicted
