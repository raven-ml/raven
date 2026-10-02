"""Fail if a dune rule runs a benchmark outside the bench aliases.

`dune build` builds every rule target, and `dune runtest` every runtest
rule; a benchmark run by either can take minutes or exhaust memory. A
benchmark runs only through an alias named `bench` or `bench-<name>`, from a
rule that declares or infers no targets.

A benchmark is an executable whose name starts with `bench` or that links
thumper. The script reads every dune file under the given roots (default:
the current directory) and prints each offending stanza.

    python3 scripts/check_bench_rules.py [ROOT...]
"""

from __future__ import annotations

import os
import re
import sys

BENCH_ALIAS = re.compile(r"bench(-[A-Za-z0-9_-]+)?$")
TEST_STANZAS = {"test", "tests", "cram"}


# S-expressions: atoms are str, lists are list.


def parse(text: str, path: str) -> list[tuple[int, list]]:
    """The top-level lists of [text], each with its first line."""
    pos = 0
    n = len(text)
    stack: list[list] = [[]]
    lines: list[int] = []
    skip_next = []  # depths at which a `#;` comment drops the next datum

    def push(datum):
        depth = len(stack)
        if skip_next and skip_next[-1] == depth:
            skip_next.pop()
            if depth == 1 and isinstance(datum, list):
                lines.pop()
            return
        stack[-1].append(datum)

    while pos < n:
        c = text[pos]
        if c.isspace():
            pos += 1
        elif c == ";":
            while pos < n and text[pos] != "\n":
                pos += 1
        elif text.startswith("#|", pos):
            end = text.find("|#", pos + 2)
            if end < 0:
                sys.exit(f"{path}: unterminated block comment")
            pos = end + 2
        elif text.startswith("#;", pos):
            skip_next.append(len(stack))
            pos += 2
        elif c == "(":
            if len(stack) == 1:
                lines.append(text.count("\n", 0, pos) + 1)
            stack.append([])
            pos += 1
        elif c == ")":
            if len(stack) == 1:
                sys.exit(f"{path}: unbalanced ')'")
            done = stack.pop()
            pos += 1
            push(done)
        elif c == '"':
            start = pos
            pos += 1
            while pos < n and text[pos] != '"':
                pos += 2 if text[pos] == "\\" else 1
            pos += 1
            push(text[start:pos])
        else:
            start = pos
            while pos < n and not text[pos].isspace() and text[pos] not in '();"':
                pos += 1
            push(text[start:pos])
    if len(stack) != 1:
        sys.exit(f"{path}: unbalanced '('")
    top = [x for x in stack[0] if isinstance(x, list)]
    return [(line, x) for line, x in zip(lines, top) if x]


def atoms(sexp):
    if isinstance(sexp, str):
        yield sexp
        return
    for x in sexp:
        yield from atoms(x)


def fields(stanza: list) -> dict[str, list]:
    out: dict[str, list] = {}
    for f in stanza[1:]:
        if isinstance(f, list) and f and isinstance(f[0], str):
            out.setdefault(f[0], []).append(f[1:])
    return out


# Benchmarks


def bench_names(stanzas: list[list]) -> set[str]:
    names = set()
    for s in stanzas:
        if not s or s[0] not in ("executable", "executables"):
            continue
        f = fields(s)
        declared = [a for v in f.get("name", []) + f.get("names", []) for a in atoms(v)]
        links_thumper = any(a == "thumper" for v in f.get("libraries", []) for a in atoms(v))
        for name in declared:
            if links_thumper or name.startswith("bench"):
                names.add(name)
    return names


def mentions(sexp, names: set[str]) -> list[str]:
    if not names:
        return []
    exe = re.compile(
        r"(?:^|[/:{])(" + "|".join(map(re.escape, sorted(names))) + r")(?:\.bc)?\.exe\b"
    )
    found = set()
    for a in atoms(sexp):
        for m in exe.finditer(a):
            found.add(m.group(1))
    return sorted(found)


# Rules


def aliases(stanza: list) -> list[str]:
    f = fields(stanza)
    out = [a for v in f.get("alias", []) + f.get("aliases", []) for a in atoms(v)]
    if stanza[0] == "alias":
        out += [a for v in f.get("name", []) for a in atoms(v)]
    return out


OUTPUT_ACTIONS = {"with-stdout-to", "with-stderr-to", "with-outputs-to"}


def heads(sexp):
    """Every list in [sexp] that starts with an atom, with that atom."""
    if isinstance(sexp, list):
        if sexp and isinstance(sexp[0], str):
            yield sexp[0], sexp
        for x in sexp:
            yield from heads(x)


def has_targets(stanza: list) -> bool:
    """Whether a rule declares or infers targets. dune infers the files an
    action writes with [with-stdout-to] and its kin, except the second file of
    a [diff?], which the action may leave unwritten."""
    f = fields(stanza)
    if "target" in f or "targets" in f:
        return True
    action = f.get("action", [])
    written = {x[1] for h, x in heads(action) if h in OUTPUT_ACTIONS and len(x) > 1}
    optional = {x[2] for h, x in heads(action) if h == "diff?" and len(x) > 2}
    return bool(written - optional)


def problem(stanza: list, names: set[str]) -> str | None:
    kind = stanza[0]
    used = mentions(stanza, names)
    if kind in TEST_STANZAS:
        f = fields(stanza)
        if any(a == "thumper" for v in f.get("libraries", []) for a in atoms(v)):
            return "a test links thumper and runs in runtest"
        if used:
            return f"runs {', '.join(used)} in runtest"
        return None
    if kind not in ("rule", "alias") or not used:
        return None
    exes = ", ".join(used)
    names_ = aliases(stanza)
    if not names_:
        return f"runs {exes} with no alias, so in @default"
    wrong = [a for a in names_ if not BENCH_ALIAS.match(a)]
    if wrong:
        return f"runs {exes} in @{', @'.join(wrong)}"
    if has_targets(stanza):
        return f"runs {exes} to build a target, which @default builds"
    return None


def dune_files(roots: list[str]):
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(
                d for d in dirnames if not d.startswith(("_", ".")) and d != "dune.lock"
            )
            if "dune" in filenames:
                yield os.path.join(dirpath, "dune")


def main() -> int:
    roots = sys.argv[1:] or ["."]
    files = {}
    for path in dune_files(roots):
        with open(path, encoding="utf-8") as fh:
            files[path] = parse(fh.read(), path)

    # A rule may run a benchmark built in another directory.
    names = set()
    for stanzas in files.values():
        names |= bench_names([s for _, s in stanzas])

    failures = 0
    for path, stanzas in files.items():
        for line, s in stanzas:
            why = problem(s, names)
            if why:
                failures += 1
                print(f"{path}:{line}: ({s[0]} ...) {why}")
    if failures:
        print(
            f"{failures} stanza(s) run a benchmark outside @bench; attach them "
            "to `bench` or `bench-<name>` with no targets.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
