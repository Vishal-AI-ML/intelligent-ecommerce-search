"""Deterministic query-understanding parser (Milestone 6, PARSER_VERSION "qu-1").

Pure: no I/O, logging, settings, clock, randomness or model. The input must already be the
output of `normalize_query`; this module never normalizes and never enforces the API length
limit. Every stage is one left-to-right pass over the tokens with constant lookahead.

Stages, in fixed order (see the M6 plan §5):

* U1  unsupported numeric complexes (grouped numbers, ranges, compounds) -> closed ambiguities;
* S1  lexicon phrases and atoms;
* S2  capacity values (`8gb`, `1tb`);
* S3  capacity role claims from keyword chains; S3b implicit GB (`256 ssd`);
* S4  the documented R6 pairing convention; S4b leftover capacities -> `bare_capacity`;
* S5  price expressions and bounds (inclusive); out-of-range values -> `out_of_range_value`;
      a lower prefix marker and an upper postfix marker around one price bind both bounds;
* S6  connectives (`wala`, `ka`) extend the evidence that ends right before them;
* S7  resolution: repeated different values and `min_price > max_price` become conflicts.

Numbers are converted only here, only from tokens already validated by the tokenizer, and only
with `int` and `Decimal`, so NaN, Infinity, exponents, signs and locale forms are impossible.
"""

from dataclasses import dataclass
from decimal import Decimal

from ecommerce_search.catalog.taxonomy import StorageType
from ecommerce_search.query_understanding.lexicon import (
    ABSORBABLE_MARKERS,
    CAPACITY_UNITS,
    CUR,
    CURRENCY_WORDS,
    GLUE,
    LINK,
    PHRASE_INDEX,
    RANGE_DASHES,
    RANGE_WORD_AND,
    RANGE_WORD_BETWEEN,
    RANGE_WORD_TO,
    ROLE_MEMORY,
    ROLE_RAM,
    ROLE_STORAGE,
    SPACE,
    THOUSANDS_UNIT,
    UNIT_WORDS,
    Phrase,
    Role,
)
from ecommerce_search.query_understanding.models import (
    CENT,
    DECIMAL_CONTEXT,
    MAX_PRICE,
    MAX_RAM_GB,
    MAX_STORAGE_GB,
    MIN_CAPACITY_GB,
    Ambiguity,
    AmbiguityReason,
    Conflict,
    ConflictReason,
    MatchedTerm,
    MatchRule,
    QueryAttributes,
    QueryUnderstanding,
    SourceSpan,
    UnderstandingField,
)
from ecommerce_search.query_understanding.tokenizer import Token, TokenKind, tokenize

PARSER_VERSION = "qu-1"

_PRECONDITION_MESSAGE = "query understanding requires a non-empty, whitespace-normalized query"
_THOUSAND = Decimal(1000)
_ZERO = Decimal(0)

_STORAGE_TYPE_ROLES = (Role.STORAGE_TYPE, Role.STORAGE_INTERFACE)
_PREFIX_KEYWORD_ROLES = frozenset({ROLE_RAM, ROLE_MEMORY, ROLE_STORAGE})

# Unsupported-complex join kinds; when a chain mixes kinds the reported reason is the first of
# range, compound, grouped present in the chain.
_JOIN_REASONS = (
    AmbiguityReason.UNSUPPORTED_RANGE,
    AmbiguityReason.UNSUPPORTED_NUMERIC_COMPOUND,
    AmbiguityReason.UNSUPPORTED_GROUPED_NUMBER,
)


@dataclass(slots=True)
class _Item:
    """Internal evidence: a candidate matched term or an ambiguity over tokens first..last."""

    first: int
    last: int
    value: str | None
    rule: MatchRule | None = None
    field: UnderstandingField | None = None
    reason: AmbiguityReason | None = None
    number: int | Decimal | None = None


@dataclass(frozen=True, slots=True)
class _Atom:
    role: Role
    value: str
    first: int
    last: int


@dataclass(frozen=True, slots=True)
class _Expression:
    """`[currency] NUMBER [glued unit]` over tokens first..last (U1)."""

    first: int
    number: int
    last: int
    currency: bool
    unit: bool


@dataclass(slots=True)
class _Capacity:
    first: int
    last: int
    gb: int
    roles: frozenset[str] = frozenset()


class _State:
    def __init__(self, tokens: tuple[Token, ...]) -> None:
        self.tokens = tokens
        self.owned = [False] * len(tokens)
        self.items: list[_Item] = []
        self.atoms_by_first: dict[int, _Atom] = {}
        self.atoms_by_last: dict[int, _Atom] = {}
        self.keyword_roles: dict[int, str] = {}  # token -> RAM / MEMORY / STORAGE
        self.storage_type_tokens: set[int] = set()  # shareable storage-type anchors
        self.claimed: set[int] = set()  # keywords and anchors already in a capacity chain

    def own(self, first: int, last: int) -> None:
        for index in range(first, last + 1):
            self.owned[index] = True

    def term(
        self,
        first: int,
        last: int,
        rule: MatchRule,
        field: UnderstandingField,
        value: str,
        number: int | Decimal | None = None,
    ) -> None:
        self.items.append(_Item(first, last, value, rule=rule, field=field, number=number))

    def ambiguity(self, first: int, last: int, reason: AmbiguityReason, value: str | None) -> None:
        self.items.append(_Item(first, last, value, reason=reason))


def _canonical_price(value: Decimal) -> str:
    return format(value.quantize(CENT, context=DECIMAL_CONTEXT), "f")


def _gap(tokens: tuple[Token, ...], index: int) -> str:
    return tokens[index].gap_before


def _is_word(tokens: tuple[Token, ...], index: int, word: str) -> bool:
    return (
        0 <= index < len(tokens)
        and tokens[index].kind is TokenKind.WORD
        and tokens[index].folded == word
    )


def _glued_word(tokens: tuple[Token, ...], index: int) -> bool:
    """Token `index` is the WORD half of a `D L` run (`8gb`, `8ssd`, `40kg`).

    Such a word is usable only as the unit of an approved GLUE construction; it is never a
    lexicon atom, and a run whose word is not that unit binds no value at all.
    """
    return (
        0 < index < len(tokens)
        and tokens[index].kind is TokenKind.WORD
        and tokens[index].gap_before == ""
        and tokens[index - 1].kind is TokenKind.NUMBER
    )


def _currency_number(tokens: tuple[Token, ...], index: int) -> bool:
    """Token `index` is a currency prefix immediately followed (allowed gap) by a NUMBER."""
    after = index + 1
    if after >= len(tokens) or tokens[after].kind is not TokenKind.NUMBER:
        return False
    token = tokens[index]
    if token.kind is TokenKind.CURRENCY:
        return _gap(tokens, after) in GLUE
    return (
        token.kind is TokenKind.WORD
        and token.folded in CURRENCY_WORDS
        and (_gap(tokens, after) in CUR)
    )


# ---------------------------------------------------------------------------------------------
# U1: unsupported numeric complexes


def _expression(tokens: tuple[Token, ...], index: int) -> _Expression | None:
    if index >= len(tokens):
        return None
    currency = _currency_number(tokens, index)
    number = index + 1 if currency else index
    if tokens[number].kind is not TokenKind.NUMBER:
        return None
    unit_index = number + 1
    unit = (
        unit_index < len(tokens)
        and tokens[unit_index].kind is TokenKind.WORD
        and tokens[unit_index].folded in UNIT_WORDS
        and _gap(tokens, unit_index) in GLUE
    )
    return _Expression(index, number, unit_index if unit else number, currency, unit)


def _three_digit_bare(tokens: tuple[Token, ...], expression: _Expression) -> bool:
    folded = tokens[expression.number].folded
    plain = not expression.currency and not expression.unit
    return plain and len(folded) == 3 and "." not in folded


def _join(
    tokens: tuple[Token, ...], left: _Expression, allow_and: bool
) -> tuple[AmbiguityReason, _Expression, bool] | None:
    """The join between `left` and the next expression, or None. Returns (kind, right, used_and)."""
    index = left.last + 1
    if index >= len(tokens):
        return None
    gap = _gap(tokens, index)
    right = _expression(tokens, index)
    if right is not None:
        if "," in gap and not left.unit:
            return AmbiguityReason.UNSUPPORTED_GROUPED_NUMBER, right, False
        if any(dash in gap for dash in RANGE_DASHES):
            return AmbiguityReason.UNSUPPORTED_RANGE, right, False
        if gap != " ":
            return AmbiguityReason.UNSUPPORTED_NUMERIC_COMPOUND, right, False
        if not left.unit and _three_digit_bare(tokens, right):
            return AmbiguityReason.UNSUPPORTED_NUMERIC_COMPOUND, right, False
        return None
    words = (RANGE_WORD_TO, RANGE_WORD_AND) if allow_and else (RANGE_WORD_TO,)
    for word in words:
        if _is_word(tokens, index, word) and gap == " " and index + 1 < len(tokens):
            right = _expression(tokens, index + 1)
            if right is not None and _gap(tokens, index + 1) == " ":
                return AmbiguityReason.UNSUPPORTED_RANGE, right, word == RANGE_WORD_AND
    return None


def _absorbed_marker_first(tokens: tuple[Token, ...], first: int, owned: list[bool]) -> int:
    if first == 0 or _gap(tokens, first) not in SPACE:
        return first
    for length in (2, 1):
        start = first - length
        if start < 0 or any(owned[start:first]):
            continue
        if any(tokens[i].kind is not TokenKind.WORD for i in range(start, first)):
            continue
        if any(_gap(tokens, i) not in SPACE for i in range(start + 1, first)):
            continue
        if tuple(tokens[i].folded for i in range(start, first)) in ABSORBABLE_MARKERS:
            return start
    return first


def _stage_unsupported_complexes(state: _State) -> None:
    tokens = state.tokens
    index = 0
    while index < len(tokens):
        left = _expression(tokens, index)
        if left is None:
            index += 1
            continue
        # `E and E` is a range only right after `between`.
        allow_and = _is_word(tokens, index - 1, RANGE_WORD_BETWEEN) and _gap(tokens, index) == " "
        kinds: set[AmbiguityReason] = set()
        while (joined := _join(tokens, left, allow_and)) is not None:
            kind, left, used_and = joined
            kinds.add(kind)
            allow_and = allow_and and not used_and
        if kinds:
            first = _absorbed_marker_first(tokens, index, state.owned)
            reason = next(reason for reason in _JOIN_REASONS if reason in kinds)
            state.own(first, left.last)
            state.ambiguity(first, left.last, reason, None)
        index = left.last + 1


# ---------------------------------------------------------------------------------------------
# S1: phrases and atoms


def _match_phrase(state: _State, index: int) -> Phrase | None:
    tokens = state.tokens
    for phrase in PHRASE_INDEX.get(tokens[index].folded, ()):
        last = index + len(phrase.words) - 1
        if last >= len(tokens):
            continue
        if all(
            tokens[index + offset].kind is TokenKind.WORD
            and not state.owned[index + offset]
            and tokens[index + offset].folded == word
            and (offset == 0 or _gap(tokens, index + offset) == phrase.gaps[offset - 1])
            for offset, word in enumerate(phrase.words)
        ):
            return phrase
    return None


def _stage_phrases(state: _State) -> None:
    tokens = state.tokens
    index = 0
    while index < len(tokens):
        if (
            state.owned[index]
            or tokens[index].kind is not TokenKind.WORD
            or _glued_word(tokens, index)
        ):
            index += 1
            continue
        phrase = _match_phrase(state, index)
        if phrase is None:
            index += 1
            continue
        last = index + len(phrase.words) - 1
        role = phrase.role
        if role is Role.CATEGORY:
            state.term(
                index, last, MatchRule.CATEGORY_TERM, UnderstandingField.CATEGORY, phrase.value
            )
        elif role is Role.BRAND:
            state.term(index, last, MatchRule.BRAND_TERM, UnderstandingField.BRAND, phrase.value)
        elif role is Role.INTENT:
            state.term(
                index, last, MatchRule.INTENT_TERM, UnderstandingField.SEMANTIC_INTENT, phrase.value
            )
        elif role is Role.STORAGE_TYPE:
            state.term(
                index,
                last,
                MatchRule.STORAGE_TYPE_TERM,
                UnderstandingField.STORAGE_TYPE,
                phrase.value,
            )
        elif role is Role.STORAGE_INTERFACE:
            state.term(
                index,
                last,
                MatchRule.STORAGE_INTERFACE_TERM,
                UnderstandingField.STORAGE_INTERFACE,
                phrase.value,
            )
            state.term(
                index,
                last,
                MatchRule.STORAGE_TYPE_IMPLIED_BY_INTERFACE,
                UnderstandingField.STORAGE_TYPE,
                StorageType.SSD.value,
            )
        elif role is Role.PRICE_PREFERENCE:
            state.term(
                index,
                last,
                MatchRule.PRICE_PREFERENCE_TERM,
                UnderstandingField.PRICE_PREFERENCE,
                phrase.value,
            )
        else:
            atom = _Atom(role, phrase.value, index, last)
            state.atoms_by_first[index] = atom
            state.atoms_by_last[last] = atom
            if role is Role.CAPACITY_KEYWORD:
                state.keyword_roles[index] = phrase.value
        if role in _STORAGE_TYPE_ROLES:
            state.storage_type_tokens.add(index)
            state.keyword_roles[index] = ROLE_STORAGE
        if role not in (
            Role.CAPACITY_KEYWORD,
            Role.UPPER_MARKER,
            Role.LOWER_MARKER,
            Role.UPPER_POSTFIX_MARKER,
            Role.CONNECTIVE,
        ):
            state.own(index, last)
        index = last + 1


# ---------------------------------------------------------------------------------------------
# S2 / S3 / S3b / S4: capacities


def _stage_capacity_values(state: _State) -> list[_Capacity]:
    tokens = state.tokens
    capacities: list[_Capacity] = []
    for index in range(len(tokens) - 1):
        number, unit = tokens[index], tokens[index + 1]
        if (
            number.kind is TokenKind.NUMBER
            and "." not in number.folded
            and not state.owned[index]
            and not state.owned[index + 1]
            and unit.kind is TokenKind.WORD
            and unit.folded in CAPACITY_UNITS
            and unit.gap_before in GLUE
        ):
            gb = int(number.folded) * CAPACITY_UNITS[unit.folded]
            capacities.append(_Capacity(index, index + 1, gb))
            state.own(index, index + 1)
    return capacities


def _keyword_available(state: _State, index: int) -> bool:
    if index in state.storage_type_tokens:
        return index not in state.claimed  # a storage-type token anchors at most one capacity
    return index in state.keyword_roles and not state.owned[index] and index not in state.claimed


def _postfix_chain(state: _State, last: int, first_gaps: frozenset[str]) -> list[int]:
    tokens = state.tokens
    chain: list[int] = []
    index = last + 1
    gaps = first_gaps
    while index < len(tokens) and _gap(tokens, index) in gaps and _keyword_available(state, index):
        chain.append(index)
        index += 1
        gaps = LINK
    return chain


def _prefix_chain(state: _State, first: int) -> list[int]:
    tokens = state.tokens
    chain: list[int] = []
    index = first - 1
    while (
        index >= 0
        and _gap(tokens, index + 1) in LINK
        and index not in state.storage_type_tokens
        and state.keyword_roles.get(index) in _PREFIX_KEYWORD_ROLES
        and _keyword_available(state, index)
    ):
        chain.append(index)
        index -= 1
    chain.reverse()
    return chain


def _in_range(value: int, maximum: int) -> bool:
    return MIN_CAPACITY_GB <= value <= maximum


def _stage_capacity_roles(state: _State, capacities: list[_Capacity]) -> None:
    for capacity in capacities:
        prefix = _prefix_chain(state, capacity.first)
        postfix = _postfix_chain(state, capacity.last, LINK)
        chain = prefix + postfix
        state.claimed.update(chain)
        capacity.roles = frozenset(state.keyword_roles[index] for index in chain)
        if not chain:
            continue
        first = prefix[0] if prefix else capacity.first
        last = postfix[-1] if postfix else capacity.last
        value = str(capacity.gb)
        if capacity.roles == {ROLE_RAM}:
            rule, field, maximum = MatchRule.RAM_CAPACITY, UnderstandingField.RAM_GB, MAX_RAM_GB
        elif capacity.roles == {ROLE_STORAGE}:
            rule, field, maximum = (
                MatchRule.STORAGE_CAPACITY,
                UnderstandingField.STORAGE_GB,
                MAX_STORAGE_GB,
            )
        else:
            rule, field, maximum = None, None, MAX_STORAGE_GB
        if not _in_range(capacity.gb, maximum):
            # No partial binding: the capacity alone is reported, its keywords stay unresolved.
            state.ambiguity(
                capacity.first, capacity.last, AmbiguityReason.OUT_OF_RANGE_VALUE, value
            )
            continue
        if rule is not None and field is not None:
            state.term(first, last, rule, field, value, capacity.gb)
        elif capacity.roles == {ROLE_MEMORY}:
            state.ambiguity(first, last, AmbiguityReason.AMBIGUOUS_MEMORY_CAPACITY, value)
        else:
            state.ambiguity(first, last, AmbiguityReason.CAPACITY_MULTIPLE_ROLES, value)
        state.own(first, last)


def _stage_implicit_gb(state: _State) -> None:
    tokens = state.tokens
    for index, token in enumerate(tokens):
        if (
            token.kind is not TokenKind.NUMBER
            or "." in token.folded
            or state.owned[index]
            or (index > 0 and _currency_number(tokens, index - 1))
        ):
            continue
        chain = _postfix_chain(state, index, SPACE)
        if not chain or any(state.keyword_roles[k] != ROLE_STORAGE for k in chain):
            continue
        gb = int(token.folded)
        if not _in_range(gb, MAX_STORAGE_GB):
            state.ambiguity(index, index, AmbiguityReason.OUT_OF_RANGE_VALUE, str(gb))
            state.own(index, index)
            continue
        state.claimed.update(chain)
        state.term(
            index,
            chain[-1],
            MatchRule.STORAGE_CAPACITY_IMPLICIT_GB,
            UnderstandingField.STORAGE_GB,
            str(gb),
            gb,
        )
        state.own(index, chain[-1])


def _stage_pairing_and_leftovers(state: _State, capacities: list[_Capacity]) -> None:
    leftovers = [capacity for capacity in capacities if not capacity.roles]
    storage = {item.number for item in state.items if item.field is UnderstandingField.STORAGE_GB}
    has_ram = any(item.field is UnderstandingField.RAM_GB for item in state.items)
    blocking = {AmbiguityReason.CAPACITY_MULTIPLE_ROLES, AmbiguityReason.AMBIGUOUS_MEMORY_CAPACITY}
    has_blocking = any(item.reason in blocking for item in state.items)
    if len(leftovers) == 1 and len(storage) == 1 and not has_ram and not has_blocking:
        leftover = leftovers[0]
        (storage_gb,) = storage
        if isinstance(storage_gb, int) and leftover.gb < storage_gb:
            value = str(leftover.gb)
            if _in_range(leftover.gb, MAX_RAM_GB):
                state.term(
                    leftover.first,
                    leftover.last,
                    MatchRule.RAM_CAPACITY_PAIRED,
                    UnderstandingField.RAM_GB,
                    value,
                    leftover.gb,
                )
            else:
                state.ambiguity(
                    leftover.first, leftover.last, AmbiguityReason.OUT_OF_RANGE_VALUE, value
                )
            return
    for leftover in leftovers:
        value = str(leftover.gb)
        reason = (
            AmbiguityReason.BARE_CAPACITY
            if _in_range(leftover.gb, MAX_STORAGE_GB)
            else AmbiguityReason.OUT_OF_RANGE_VALUE
        )
        state.ambiguity(leftover.first, leftover.last, reason, value)


# ---------------------------------------------------------------------------------------------
# S5: prices


def _price_expression(state: _State, index: int) -> tuple[int, int, bool] | None:
    """(number index, last index, thousands) of a price expression starting at `index`."""
    tokens = state.tokens
    if state.owned[index]:
        return None
    token = tokens[index]
    if token.kind is TokenKind.NUMBER:
        number = index
    elif _currency_number(tokens, index):
        number = index + 1
    else:
        return None
    if state.owned[number]:
        return None
    unit = number + 1
    thousands = (
        unit < len(tokens)
        and not state.owned[unit]
        and _is_word(tokens, unit, THOUSANDS_UNIT)
        and _gap(tokens, unit) in GLUE
    )
    if number == index and not thousands:
        return None  # a bare NUMBER is never a price
    if not thousands and _glued_word(tokens, unit):
        return None  # `₹40kg`: an unsupported `D L` run sets no price
    return number, unit if thousands else number, thousands


def _marker(state: _State, atom: _Atom | None, roles: tuple[Role, ...]) -> _Atom | None:
    if atom is None or atom.role not in roles:
        return None
    if any(state.owned[atom.first : atom.last + 1]):
        return None
    return atom


def _stage_prices(state: _State) -> None:
    tokens = state.tokens
    index = 0
    while index < len(tokens):
        expression = _price_expression(state, index)
        if expression is None:
            index += 1
            continue
        number, last, thousands = expression
        value = Decimal(tokens[number].folded)
        if thousands:
            value = DECIMAL_CONTEXT.multiply(value, _THOUSAND)
        canonical = _canonical_price(value)
        if not _ZERO < value <= MAX_PRICE:
            state.ambiguity(index, last, AmbiguityReason.OUT_OF_RANGE_VALUE, canonical)
            state.own(index, last)
            index = last + 1
            continue
        prefix = None
        if _gap(tokens, index) in SPACE:
            prefix = _marker(
                state, state.atoms_by_last.get(index - 1), (Role.UPPER_MARKER, Role.LOWER_MARKER)
            )
        postfix = None
        if last + 1 < len(tokens) and _gap(tokens, last + 1) in SPACE:
            postfix = _marker(
                state, state.atoms_by_first.get(last + 1), (Role.UPPER_POSTFIX_MARKER,)
            )
        first = index
        end = last
        if prefix is not None and prefix.role is Role.LOWER_MARKER:
            # `from 20k ke andar`: both explicit markers bind the same price, each with its own
            # truthful span; min == max, so no marker is silently discarded.
            first = prefix.first
            state.term(
                first,
                last,
                MatchRule.PRICE_LOWER_BOUND,
                UnderstandingField.MIN_PRICE,
                canonical,
                value,
            )
            if postfix is not None:
                end = postfix.last
                state.term(
                    index,
                    end,
                    MatchRule.PRICE_UPPER_BOUND,
                    UnderstandingField.MAX_PRICE,
                    canonical,
                    value,
                )
        elif prefix is not None or postfix is not None:
            first = prefix.first if prefix is not None else index
            end = postfix.last if postfix is not None else last
            state.term(
                first,
                end,
                MatchRule.PRICE_UPPER_BOUND,
                UnderstandingField.MAX_PRICE,
                canonical,
                value,
            )
        else:
            state.ambiguity(index, last, AmbiguityReason.PRICE_WITHOUT_BOUND, canonical)
        state.own(first, end)
        index = end + 1


# ---------------------------------------------------------------------------------------------
# S6: connectives


def _stage_connectives(state: _State) -> None:
    ends: dict[int, list[_Item]] = {}
    for item in state.items:
        ends.setdefault(item.last, []).append(item)
    for index in sorted(state.atoms_by_first):
        atom = state.atoms_by_first[index]
        if atom.role is not Role.CONNECTIVE or state.owned[index]:
            continue
        previous = index - 1
        if previous < 0 or not state.owned[previous] or _gap(state.tokens, index) not in SPACE:
            continue
        extended = ends.pop(previous, [])
        if not extended:
            continue
        for item in extended:
            item.last = index
        ends[index] = extended
        state.owned[index] = True


# ---------------------------------------------------------------------------------------------
# S7: resolution and output


def _span(raw: str, tokens: tuple[Token, ...], first: int, last: int) -> SourceSpan:
    start, end = tokens[first].start, tokens[last].end
    return SourceSpan(start=start, end=end, text=raw[start:end])


def _term_key(term: MatchedTerm) -> tuple[int, int, str, str]:
    return term.span.start, term.span.end, term.field.value, term.rule.value


def parse_normalized_query(normalized_query: str) -> QueryUnderstanding:
    """Parse an already-normalized query (the output of `normalize_query`)."""
    if not normalized_query or normalized_query != " ".join(normalized_query.split()):
        raise ValueError(_PRECONDITION_MESSAGE)
    tokens = tokenize(normalized_query)
    state = _State(tokens)
    _stage_unsupported_complexes(state)
    _stage_phrases(state)
    capacities = _stage_capacity_values(state)
    _stage_capacity_roles(state, capacities)
    _stage_implicit_gb(state)
    _stage_pairing_and_leftovers(state, capacities)
    _stage_prices(state)
    _stage_connectives(state)
    return _resolve(normalized_query, state)


def _resolve(raw: str, state: _State) -> QueryUnderstanding:
    tokens = state.tokens
    terms: dict[UnderstandingField, list[tuple[MatchedTerm, _Item]]] = {}
    ambiguities: list[Ambiguity] = []
    for item in state.items:
        span = _span(raw, tokens, item.first, item.last)
        if item.reason is not None:
            ambiguities.append(Ambiguity(reason=item.reason, span=span, value=item.value))
        elif item.rule is not None and item.field is not None and item.value is not None:
            term = MatchedTerm(rule=item.rule, field=item.field, value=item.value, span=span)
            terms.setdefault(item.field, []).append((term, item))

    resolved: dict[UnderstandingField, _Item] = {}
    conflicts: list[Conflict] = []
    conflicted: set[UnderstandingField] = set()

    def conflict(field: UnderstandingField, reason: ConflictReason) -> None:
        candidates = tuple(sorted((term for term, _ in terms[field]), key=_term_key))
        conflicts.append(Conflict(field=field, reason=reason, candidates=candidates))
        conflicted.add(field)
        resolved.pop(field, None)

    for field in UnderstandingField:
        candidates = terms.get(field, [])
        if not candidates:
            continue
        if len({term.value for term, _ in candidates}) == 1:
            resolved[field] = candidates[0][1]
        else:
            conflict(field, ConflictReason.REPEATED_DIFFERENT_VALUES)

    low = resolved.get(UnderstandingField.MIN_PRICE)
    high = resolved.get(UnderstandingField.MAX_PRICE)
    if low is not None and high is not None and low.number > high.number:
        conflict(UnderstandingField.MAX_PRICE, ConflictReason.MIN_PRICE_EXCEEDS_MAX_PRICE)
        conflict(UnderstandingField.MIN_PRICE, ConflictReason.MIN_PRICE_EXCEEDS_MAX_PRICE)

    if (
        UnderstandingField.STORAGE_TYPE in conflicted
        and UnderstandingField.STORAGE_INTERFACE in resolved
    ):
        conflict(
            UnderstandingField.STORAGE_INTERFACE,
            ConflictReason.DEPENDENT_ON_CONFLICTED_STORAGE_TYPE,
        )

    # Enumerations are passed as their canonical values; numbers as the converted int/Decimal.
    values = {field: item.value for field, item in resolved.items()}
    numbers = {field: item.number for field, item in resolved.items()}
    matched = [
        term for field, pairs in terms.items() if field not in conflicted for term, _ in pairs
    ]
    return QueryUnderstanding(
        raw_query=raw,
        category=values.get(UnderstandingField.CATEGORY),
        brand=values.get(UnderstandingField.BRAND),
        ram_gb=numbers.get(UnderstandingField.RAM_GB),
        storage_gb=numbers.get(UnderstandingField.STORAGE_GB),
        storage_type=values.get(UnderstandingField.STORAGE_TYPE),
        min_price=numbers.get(UnderstandingField.MIN_PRICE),
        max_price=numbers.get(UnderstandingField.MAX_PRICE),
        semantic_intent=values.get(UnderstandingField.SEMANTIC_INTENT),
        attributes=QueryAttributes(
            storage_interface=values.get(UnderstandingField.STORAGE_INTERFACE),
            price_preference=values.get(UnderstandingField.PRICE_PREFERENCE),
        ),
        matched_terms=tuple(sorted(matched, key=_term_key)),
        ambiguities=tuple(
            sorted(ambiguities, key=lambda a: (a.span.start, a.span.end, a.reason.value))
        ),
        conflicts=tuple(sorted(conflicts, key=lambda c: (c.field.value, c.reason.value))),
        unresolved=tuple(
            _span(raw, tokens, index, index)
            for index in range(len(tokens))
            if not state.owned[index]
        ),
    )


__all__ = ["PARSER_VERSION", "parse_normalized_query"]
