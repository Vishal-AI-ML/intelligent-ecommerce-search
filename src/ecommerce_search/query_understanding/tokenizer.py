"""Hand-written linear tokenizer for query understanding (no regular expressions).

Character folding is strictly 1:1 (one source character -> one folded character), so every
token keeps exact code-point offsets into the raw query:

* ASCII digits and fullwidth digits -> DIGIT;
* ASCII letters and fullwidth Latin letters -> LETTER (lowercased);
* `.` and fullwidth `．` -> DOT (part of a run only between two digits, otherwise a separator);
* `₹` -> CURRENCY (always its own token);
* any other letter, mark or number character, ZWNJ and ZWJ -> OTHER (kept visible);
* everything else (space, punctuation, other symbols, emoji, private-use, unassigned) -> SEP.

A run is a maximal sequence of non-SEP characters. Run shapes: digits (at most one internal dot)
-> NUMBER; letters -> WORD; digits then letters -> NUMBER + WORD (gap ""). Any other shape, and
any NUMBER that fails `[0-9]{1,10}(\\.[0-9]{1,2})?`, becomes one UNRESOLVED token for the run.
"""

import unicodedata
from dataclasses import dataclass
from enum import StrEnum

_DIGIT = 0
_LETTER = 1
_DOT = 2
_CURRENCY = 3
_OTHER = 4
_SEP = 5

CURRENCY_SYMBOL = "₹"
_JOINERS = frozenset({"‌", "‍"})  # ZWNJ, ZWJ
_FULLWIDTH_OFFSET = 0xFEE0
MAX_NUMBER_INTEGER_DIGITS = 10
MAX_NUMBER_FRACTION_DIGITS = 2


class TokenKind(StrEnum):
    NUMBER = "NUMBER"
    WORD = "WORD"
    CURRENCY = "CURRENCY"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True, slots=True)
class Token:
    kind: TokenKind
    start: int
    end: int
    text: str  # exact source text, raw_query[start:end]
    folded: str  # 1:1 folded ASCII text (meaningful for NUMBER, WORD and CURRENCY)
    gap_before: str  # exact source text between the previous token and this one


def _classify(char: str) -> tuple[int, str]:
    code = ord(char)
    if 0x30 <= code <= 0x39:
        return _DIGIT, char
    if 0xFF10 <= code <= 0xFF19:
        return _DIGIT, chr(code - _FULLWIDTH_OFFSET)
    if 0x61 <= code <= 0x7A:
        return _LETTER, char
    if 0x41 <= code <= 0x5A:
        return _LETTER, chr(code + 0x20)
    if 0xFF41 <= code <= 0xFF5A:
        return _LETTER, chr(code - _FULLWIDTH_OFFSET)
    if 0xFF21 <= code <= 0xFF3A:
        return _LETTER, chr(code - _FULLWIDTH_OFFSET + 0x20)
    if char in (".", "．"):
        return _DOT, "."
    if char == CURRENCY_SYMBOL:
        return _CURRENCY, char
    if char in _JOINERS or unicodedata.category(char)[0] in "LMN":
        return _OTHER, char
    return _SEP, char


def is_valid_number(folded: str) -> bool:
    """`[0-9]{1,10}(\\.[0-9]{1,2})?` over folded ASCII text, checked without a regex."""
    integer, dot, fraction = folded.partition(".")
    if not (1 <= len(integer) <= MAX_NUMBER_INTEGER_DIGITS and _ascii_digits(integer)):
        return False
    if dot:
        return 1 <= len(fraction) <= MAX_NUMBER_FRACTION_DIGITS and _ascii_digits(fraction)
    return True


def _ascii_digits(text: str) -> bool:
    return all("0" <= char <= "9" for char in text)


def _run_tokens(
    start: int, classes: list[int], folded: list[str]
) -> list[tuple[TokenKind, int, int, str]]:
    """Split one run into (kind, start, end, folded) pieces according to its shape."""
    length = len(classes)
    end = start + length
    split = 0
    dots = 0
    while split < length and classes[split] in (_DIGIT, _DOT):
        dots += classes[split] == _DOT
        split += 1
    rest_ok = all(cls == _LETTER for cls in classes[split:])
    if not rest_ok or dots > 1:
        return [(TokenKind.UNRESOLVED, start, end, "".join(folded))]
    if split == 0:
        return [(TokenKind.WORD, start, end, "".join(folded))]
    number = "".join(folded[:split])
    if not is_valid_number(number):
        return [(TokenKind.UNRESOLVED, start, end, "".join(folded))]
    if split == length:
        return [(TokenKind.NUMBER, start, end, number)]
    return [
        (TokenKind.NUMBER, start, start + split, number),
        (TokenKind.WORD, start + split, end, "".join(folded[split:])),
    ]


def tokenize(raw: str) -> tuple[Token, ...]:
    """Tokenize in one left-to-right pass; a dot looks one character back and one ahead."""
    classified = [_classify(char) for char in raw]
    length = len(classified)
    pieces: list[tuple[TokenKind, int, int, str]] = []
    run_start = -1
    classes: list[int] = []
    folded: list[str] = []
    for index, (cls, fold) in enumerate(classified):
        if cls == _DOT:
            between_digits = (
                0 < index < length - 1
                and classified[index - 1][0] == _DIGIT
                and classified[index + 1][0] == _DIGIT
            )
            cls = _DOT if between_digits else _SEP
        if cls in (_SEP, _CURRENCY):
            if classes:
                pieces.extend(_run_tokens(run_start, classes, folded))
                classes, folded = [], []
            if cls == _CURRENCY:
                pieces.append((TokenKind.CURRENCY, index, index + 1, fold))
            continue
        if not classes:
            run_start = index
        classes.append(cls)
        folded.append(fold)
    if classes:
        pieces.extend(_run_tokens(run_start, classes, folded))

    tokens: list[Token] = []
    previous_end = 0
    for kind, start, end, fold in pieces:
        tokens.append(Token(kind, start, end, raw[start:end], fold, raw[previous_end:start]))
        previous_end = end
    return tuple(tokens)
