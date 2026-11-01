"""Read the case lists of an OCaml test file, for the generators' `--check`.

A generator prints OCaml fragments, `case ... expected;` statements, that are
pasted over a case list of a test file, and ocamlformat then lays them out.
`--check` reads both through the same lexer, so layout, comments and the
spelling of a string literal (`"é"`, `"\\xC3\\xA9"`, `"\\u{e9}"`) do not count,
and compares the cases by their arguments: a case is a statement whose first
word is its head, whose last `width` arguments are the expectation and whose
other words are its key. A check fails on a generated case the test lacks or
states otherwise, and, unless the list also holds hand-written cases, on a
case the generator does not give.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

Token = tuple[str, object]

OPEN = {"(": ")", "[": "]", "[|": "|]", "{": "}"}
CLOSE = set(OPEN.values())
OPERATOR = set("!$%&*+-./:<=>?@^|~#")
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_']*")
NUMBER = re.compile(r"-?[0-9][0-9_]*(\.[0-9_]*)?([eE][-+]?[0-9]+)?")
QUOTED = re.compile(r"\{([a-z_]*)\|")
CHAR = re.compile(r"'(\\(x[0-9A-Fa-f]{2}|[0-9]{3}|o[0-7]{3}|.)|[^\\'])'")
ESCAPES = {"n": b"\n", "t": b"\t", "b": b"\b", "r": b"\r", " ": b" "}


def string(source: str, pos: int) -> tuple[bytes, int]:
    """The bytes of the string literal opening at `source[pos]`, and the
    position after it."""
    out = bytearray()
    pos += 1
    while source[pos] != '"':
        char = source[pos]
        if char != "\\":
            out += char.encode()
            pos += 1
            continue
        escape = source[pos + 1]
        if escape == "\n":
            pos += 2
            while source[pos] in " \t":
                pos += 1
        elif escape in ESCAPES:
            out += ESCAPES[escape]
            pos += 2
        elif escape == "x":
            out.append(int(source[pos + 2 : pos + 4], 16))
            pos += 4
        elif escape == "o":
            out.append(int(source[pos + 2 : pos + 5], 8))
            pos += 5
        elif escape.isdigit():
            out.append(int(source[pos + 1 : pos + 4]))
            pos += 4
        elif escape == "u":
            stop = source.index("}", pos)
            out += chr(int(source[pos + 3 : stop], 16)).encode()
            pos = stop + 1
        else:
            out += escape.encode()
            pos += 2
    return bytes(out), pos + 1


def comment(source: str, pos: int) -> int:
    """The position after the comment opening at `source[pos]`."""
    depth = 0
    while True:
        if source.startswith("(*", pos):
            depth += 1
            pos += 2
        elif source.startswith("*)", pos):
            depth -= 1
            pos += 2
            if depth == 0:
                return pos
        elif source[pos] == '"':
            pos = string(source, pos)[1]
        else:
            pos += 1


def tokens(source: str) -> list[Token]:
    """The tokens of `source`: ("str", bytes) for a string literal, ("word",
    text) for anything else; comments dropped, and so is a `;` before a closing
    bracket, which ocamlformat adds or removes with the layout."""
    out: list[Token] = []
    pos = 0
    while pos < len(source):
        char = source[pos]
        if char.isspace():
            pos += 1
        elif source.startswith("(*", pos):
            pos = comment(source, pos)
        elif char == '"':
            value, pos = string(source, pos)
            out.append(("str", value))
        elif quoted := QUOTED.match(source, pos):
            stop = source.index("|" + quoted.group(1) + "}", quoted.end())
            out.append(("str", source[quoted.end() : stop].encode()))
            pos = stop + len(quoted.group(1)) + 2
        elif source.startswith("[|", pos) or source.startswith("|]", pos):
            out.append(("word", source[pos : pos + 2]))
            pos += 2
        elif char == "'" and (literal := CHAR.match(source, pos)):
            out.append(("word", literal.group(0)))
            pos = literal.end()
        elif char in "`~?" and (ident := IDENT.match(source, pos + 1)):
            stop = ident.end()
            if char != "`" and source.startswith(":", stop):
                stop += 1
            out.append(("word", source[pos:stop]))
            pos = stop
        elif ident := IDENT.match(source, pos):
            out.append(("word", ident.group(0)))
            pos = ident.end()
        elif number := NUMBER.match(source, pos):
            out.append(("word", number.group(0)))
            pos = number.end()
        elif char in OPERATOR:
            stop = pos
            while (
                stop < len(source)
                and source[stop] in OPERATOR
                and not source.startswith("|]", stop)
            ):
                stop += 1
            out.append(("word", source[pos:stop]))
            pos = stop
        else:
            out.append(("word", char))
            pos += 1
    return [
        token
        for i, token in enumerate(out)
        if not (
            token == ("word", ";")
            and i + 1 < len(out)
            and out[i + 1][1] in CLOSE
        )
    ]


def definitions(source: str) -> dict[str, list[Token]]:
    """The tokens of each top-level `let`, by name, from after its `=`."""
    out = {}
    starts = [m.start() for m in re.finditer(r"^\S", source, re.MULTILINE)]
    for start, stop in zip(starts, [*starts[1:], len(source)]):
        header = re.match(r"(let|and)\s+(rec\s+)?([a-z_][A-Za-z0-9_']*)",
                          source[start:stop])
        if not header:
            continue
        body = tokens(source[start:stop])
        depth = 0
        for i, token in enumerate(body):
            depth += token[1] in OPEN
            depth -= token[1] in CLOSE
            if token == ("word", "=") and depth == 0:
                out[header.group(3)] = body[i + 1 :]
                break
    return out


def statements(body: list[Token]) -> list[tuple[list[Token], list[Token]]]:
    """The statements of a sequence, each as the `let ... in` bindings before it
    and the expression after them."""
    out = []
    chunk: list[Token] = []
    depth = 0
    for token in [*body, ("word", ";")]:
        if token == ("word", ";") and depth == 0:
            ins = [i for i, t in enumerate(chunk) if t == ("word", "in")]
            split = ins[-1] + 1 if ins else 0
            out.append((chunk[:split], chunk[split:]))
            chunk = []
            continue
        depth += token[1] in OPEN
        depth -= token[1] in CLOSE
        chunk.append(token)
    return out


def arguments(statement: list[Token]) -> list[list[Token]]:
    """The words of an application, a bracketed group being one word."""
    out = []
    i = 0
    while i < len(statement):
        depth, start = 0, i
        while i < len(statement):
            depth += statement[i][1] in OPEN
            depth -= statement[i][1] in CLOSE
            i += 1
            if depth <= 0:
                break
        out.append(statement[start:i])
    return out


Case = tuple[tuple, tuple]


def case(words: list[list[Token]], width: int) -> Case:
    """The key and the expectation of a case's words."""
    key = tuple(tuple(word) for word in words[1:-width])
    expected = tuple(tuple(word) for word in words[-width:])
    return key, expected


def cases(body: list[Token], head: str, width: int) -> list[Case]:
    """The cases of a sequence whose first word is `head`."""
    out = []
    for _, statement in statements(body):
        words = arguments(statement)
        if words and words[0] == [("word", head)]:
            out.append(case(words, width))
    return out


def render(words) -> str:
    """OCaml text for a key or an expectation."""
    out = ""
    for kind, value in (token for word in words for token in word):
        if kind == "str":
            value = '"' + "".join(
                chr(b) if 0x20 <= b < 0x7F and b not in b'"\\'
                else "\\x%02X" % b
                for b in value
            ) + '"'
        glued = out.endswith((":", "(", "[", "[|")) or value in (";", ",")
        out += value if glued or not out else " " + value
    return out


def differences(
    group: str, generated: list[Case], recorded: list[Case], *, partial: bool
) -> list[str]:
    """What differs between the generated cases of `group` and the recorded
    ones. With `partial`, the recorded list may hold hand-written cases whose
    keys the generator does not give."""
    out = []
    found: dict[tuple, list[tuple]] = {}
    for key, expected in recorded:
        found.setdefault(key, []).append(expected)
    keys = {key for key, _ in generated}
    for key, expected in generated:
        if key not in found:
            out.append(f"{group}: missing case {render(key)}")
            continue
        for value in found[key]:
            if value != expected:
                out.append(
                    f"{group}: case {render(key)}\n"
                    f"    test:      {render(value)}\n"
                    f"    reference: {render(expected)}"
                )
    if not partial:
        for key in found:
            if key not in keys:
                out.append(f"{group}: case not generated {render(key)}")
    order = [key for key, _ in recorded if key in keys]
    if not out and order != [key for key, _ in generated]:
        out.append(f"{group}: cases in another order")
    return out


def report(problems: list[str]) -> int:
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")
