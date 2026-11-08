"""Declare the goldens of a generator file.

A generator file `gen/<path>.py` declares each golden with one of the
decorators below. The golden is named after its function, and generate.py
writes it to `test/<path>/<name>.golden`.
"""

import os
import sys
import traceback

GOLDENS = []

# The programs tinygrad refused, as (name, reason).
REFUSED = []


class Refused(Exception):
    """What a program tinygrad refused raised, in the child that computed it."""


def reason(error):
    """One line naming the error and where it was raised."""
    frames = traceback.extract_tb(error.__traceback__)
    where = f" at {frames[-1].filename}:{frames[-1].lineno}" if frames else ""
    return "".join(traceback.format_exception_only(error)).strip().replace("\n", " ") + where


def forked(fn):
    """`fn()`, computed in a child process, so that the generator's state stays
    the one each golden starts from. Raises `Refused` if `fn` raises, in the
    parent: the child exits either way, and never runs the rest of the module. A
    child killed by a signal ends the run instead: running out of memory is no
    refusal by tinygrad, and the golden must not read as removed."""
    read, write = os.pipe()
    if (pid := os.fork()) == 0:
        status = 1
        try:
            os.close(read)
            try:
                text, status = repr(fn()), 0
            except BaseException as e:
                text = reason(e)
            with os.fdopen(write, "w") as out: out.write(text)
        finally:
            os._exit(status)
    os.close(write)
    with os.fdopen(read) as out: text = out.read()
    status = os.waitpid(pid, 0)[1]
    if os.WIFSIGNALED(status): sys.exit(f"a generator's child died of signal {os.WTERMSIG(status)}")
    if status != 0: raise Refused(text or f"the child failed: status {status}")
    return eval(text)


def attempt(name, fn, default=None):
    """`fn()`, or, recording the refusal under `name`, `default` if tinygrad
    refuses the program: one refused program leaves its goldens unwritten and
    costs the run nothing else."""
    try:
        return fn()
    except Exception as e:
        REFUSED.append((name, str(e) if isinstance(e, Refused) else reason(e)))
        return default


def cell(value):
    """A table cell: a string as it is, any other value as its repr."""
    text = value if isinstance(value, str) else repr(value)
    if "\t" in text or "\n" in text:
        raise ValueError(f"cell {text!r} holds a tab or a newline")
    return text


def table(fn):
    """Declare a table golden: `fn()` returns `(columns, rows)`."""
    def body():
        columns, rows = fn()
        if not rows:
            raise ValueError(f"table {fn.__name__} has no row")
        for row in rows:
            if len(row) != len(columns):
                raise ValueError(f"table {fn.__name__}: {row!r} has {len(row)} cells for {len(columns)} columns")
        return "\n".join("\t".join(map(cell, line)) for line in [columns, *rows])
    GOLDENS.append((fn.__name__, body))
    return fn


def text(fn):
    """Declare a text golden: `fn()` returns the text."""
    GOLDENS.append((fn.__name__, fn))
    return fn


def graph(fn):
    """Declare a graph golden: `fn()` returns a sink UOp, written in the graph
    format of graph.py."""
    def body():
        from graph import write
        return write(fn())
    GOLDENS.append((fn.__name__, body))
    return fn


def listing(uops):
    """The listing tinygrad prints for `uops`, a list of UOps in order (`render_uir`)."""
    from tinygrad.uop.render import render_uir
    return render_uir(uops) + "\n"
