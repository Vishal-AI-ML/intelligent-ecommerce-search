"""Closed, versioned v1 lexicon and the exact allowed-gap sets.

Phrases are matched over WORD tokens, longest first, at most `MAX_PHRASE_TOKENS` tokens. Each
phrase carries the exact gaps required between its words (separators are never ignored
globally). Any change to an entry, a role or a gap set bumps `LEXICON_VERSION`.
"""

from dataclasses import dataclass
from enum import StrEnum

from ecommerce_search.catalog.taxonomy import BRAND_LOOKUP, Category
from ecommerce_search.query_understanding.models import Intent

LEXICON_VERSION = "1"
MAX_PHRASE_TOKENS = 3

# Exact allowed gaps between two tokens, per construction.
GLUE = frozenset({"", " "})  # NUMBER + unit; `₹` + NUMBER
SPACE = frozenset({" "})  # phrases, markers, connectives, bare integer + storage term
LINK = frozenset({" ", "-", ",", ", "})  # capacity <-> capacity keyword, keyword <-> keyword
CUR = frozenset({" ", ".", ". "})  # `rs` / `inr` + NUMBER

CAPACITY_UNITS: dict[str, int] = {"gb": 1, "tb": 1024}  # multiplier to GB
THOUSANDS_UNIT = "k"
UNIT_WORDS = frozenset(CAPACITY_UNITS) | {THOUSANDS_UNIT}
CURRENCY_WORDS = frozenset({"rs", "inr"})

# Words used only by the unsupported-numeric-complex stage (U1).
RANGE_WORD_TO = "to"
RANGE_WORD_AND = "and"
RANGE_WORD_BETWEEN = "between"
RANGE_DASHES = ("-", "–", "—", "~")


class Role(StrEnum):
    CATEGORY = "category"
    BRAND = "brand"
    INTENT = "intent"
    STORAGE_TYPE = "storage_type"
    STORAGE_INTERFACE = "storage_interface"
    PRICE_PREFERENCE = "price_preference"
    CAPACITY_KEYWORD = "capacity_keyword"
    UPPER_MARKER = "upper_marker"
    LOWER_MARKER = "lower_marker"
    UPPER_POSTFIX_MARKER = "upper_postfix_marker"
    CONNECTIVE = "connective"


# Capacity roles named by a keyword (S3).
ROLE_RAM = "RAM"
ROLE_MEMORY = "MEMORY"
ROLE_STORAGE = "STORAGE"


@dataclass(frozen=True, slots=True)
class Phrase:
    words: tuple[str, ...]
    gaps: tuple[str, ...]  # gaps[i] sits between words[i] and words[i + 1]
    role: Role
    value: str


def _single(word: str, role: Role, value: str) -> Phrase:
    return Phrase((word,), (), role, value)


def _spaced(text: str, role: Role, value: str) -> Phrase:
    words = tuple(text.split(" "))
    return Phrase(words, (" ",) * (len(words) - 1), role, value)


def _exact(key: str, role: Role, value: str) -> Phrase:
    """Split a dictionary key into ASCII-letter words and the exact gaps between them."""
    words: list[str] = []
    gaps: list[str] = []
    word = ""
    gap = ""
    for char in key:
        if "a" <= char <= "z":
            if gap and words:
                gaps.append(gap)
            gap = ""
            word += char
        else:
            if word:
                words.append(word)
            word = ""
            gap += char
    if word:
        words.append(word)
    return Phrase(tuple(words), tuple(gaps), role, value)


CATEGORY_TERMS: dict[str, Category] = {
    "laptop": Category.LAPTOP,
    "laptops": Category.LAPTOP,
    "phone": Category.PHONE,
    "phones": Category.PHONE,
    "smartphone": Category.PHONE,
    "smartphones": Category.PHONE,
    "mobile": Category.PHONE,
    "mobiles": Category.PHONE,
    "shoe": Category.SHOES,
    "shoes": Category.SHOES,
    "headphone": Category.HEADPHONES,
    "headphones": Category.HEADPHONES,
    "earphone": Category.HEADPHONES,
    "earphones": Category.HEADPHONES,
    "earbuds": Category.HEADPHONES,
}
STORAGE_TYPE_TERMS = {"ssd": "SSD", "hdd": "HDD"}
STORAGE_INTERFACE_TERMS = {"nvme": "NVME"}
PRICE_PREFERENCE_TERMS = {"sasta": "low"}
CAPACITY_KEYWORDS = {"ram": ROLE_RAM, "memory": ROLE_MEMORY, "storage": ROLE_STORAGE}
UPPER_MARKERS = ("under", "below", "upto", "up to", "within")
LOWER_MARKERS = ("above", "over", "from")
UPPER_POSTFIX_MARKERS = ("ke andar", "ke under")
CONNECTIVES = ("wala", "ka")


def _build_phrases() -> tuple[Phrase, ...]:
    phrases: list[Phrase] = []
    phrases += [_single(word, Role.CATEGORY, cat.value) for word, cat in CATEGORY_TERMS.items()]
    phrases += [_exact(key, Role.BRAND, brand) for key, brand in BRAND_LOOKUP.items()]
    for intent in Intent:
        phrases.append(_single(intent.value, Role.INTENT, intent.value))
        phrases.append(_spaced(f"{intent.value} ke liye", Role.INTENT, intent.value))
        phrases.append(_spaced(f"for {intent.value}", Role.INTENT, intent.value))
    phrases += [_single(w, Role.STORAGE_TYPE, v) for w, v in STORAGE_TYPE_TERMS.items()]
    phrases += [_single(w, Role.STORAGE_INTERFACE, v) for w, v in STORAGE_INTERFACE_TERMS.items()]
    phrases += [_single(w, Role.PRICE_PREFERENCE, v) for w, v in PRICE_PREFERENCE_TERMS.items()]
    phrases += [_single(w, Role.CAPACITY_KEYWORD, r) for w, r in CAPACITY_KEYWORDS.items()]
    phrases += [_spaced(text, Role.UPPER_MARKER, "upper") for text in UPPER_MARKERS]
    phrases += [_spaced(text, Role.LOWER_MARKER, "lower") for text in LOWER_MARKERS]
    phrases += [_spaced(text, Role.UPPER_POSTFIX_MARKER, "upper") for text in UPPER_POSTFIX_MARKERS]
    phrases += [_single(word, Role.CONNECTIVE, word) for word in CONNECTIVES]
    return tuple(phrases)


PHRASES: tuple[Phrase, ...] = _build_phrases()

# First word -> phrases starting with it, longest first (ties keep declaration order).
PHRASE_INDEX: dict[str, tuple[Phrase, ...]] = {}
for _phrase in PHRASES:
    PHRASE_INDEX[_phrase.words[0]] = (*PHRASE_INDEX.get(_phrase.words[0], ()), _phrase)
PHRASE_INDEX = {
    word: tuple(sorted(group, key=lambda phrase: -len(phrase.words)))
    for word, group in PHRASE_INDEX.items()
}
del _phrase

# Markers U1 absorbs when they immediately precede an unsupported numeric complex.
ABSORBABLE_MARKERS: frozenset[tuple[str, ...]] = frozenset(
    tuple(text.split(" ")) for text in (*UPPER_MARKERS, *LOWER_MARKERS, RANGE_WORD_BETWEEN)
)
