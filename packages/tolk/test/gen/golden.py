"""Declare the goldens of a generator file.

A generator file `gen/<path>.py` declares each golden with one of the
decorators below. The golden is named after its function, and generate.py
writes it to `test/<path>/<name>.golden`.
"""


GOLDENS = []


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
