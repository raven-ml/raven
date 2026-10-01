"""Goldens of tinygrad/runtime/support/c.py: DLL.findlib on trees of files.

Each row is a directory tree, an environment and a search, and the file
findlib finds in it. The suite builds the same tree in a temporary directory
and compares. A row holds the platform whose search it runs: findlib is run
with the platform set to it, so the rows of macOS and Linux are generated on
either.

The columns:
- `tree`: entries separated by spaces, each `path=kind` under the tree's
  root: `elf` a file that starts as an ELF file does, `text` a file that does
  not (as a linker script), `dir` a directory, and `@target` a symbolic link
  to `target`, which may not exist.
- `name_path`: the value of the variable NAME_PATH, a path under the root;
  `-` unset, and `""` set to the empty string.
- `ld_library_path`: the value of LD_LIBRARY_PATH, its directories under the
  root separated by `:`, an empty entry left empty; `-` unset.
- `paths` and `extra_paths`: separated by spaces; a path that starts with `/`
  is absolute, under the root. `-` is the empty list.
- `found`: the file found, under the root, or `None`.

A directory never holds two candidates of one search that are both ELF files:
findlib takes the first that the directory lists, in an order the file system
chooses.
"""

import os
import sys
import sysconfig
import tempfile
from pathlib import Path

from golden import table
from tinygrad.helpers import getenv
from tinygrad.runtime.support import c

ELF = b"\x7fELF" + bytes(60)
LINKER_SCRIPT = b"INPUT(-lc)\n"

# findlib reads MULTIARCH, which sysconfig loads for the running platform: load
# it before findlib runs as another.
sysconfig.get_config_vars()

LINUX = [
    ("name_path_file", "tk-a", "x/custom.bin=text e/libtk-a.so=elf", "x/custom.bin", "-", "tk-a", "e"),
    ("name_path_missing_file", "tkb", "e/libtkb.so=elf", "x/none.so", "-", "tkb", "e"),
    ("name_path_empty", "tkc", "e/libtkc.so=elf", '""', "-", "tkc", "e"),
    ("name_path_dir_first", "tkd", "n/libtkd.so=elf l/libtkd.so=elf e/libtkd.so=elf", "n", "l", "tkd", "e"),
    ("name_path_dir_each_path", "tke", "n/libtkeb.so=elf", "n", "-", "tkea tkeb", "-"),
    ("ld_library_path_before_extra", "tkf", "l/libtkf.so=elf e/libtkf.so=elf", "-", "l", "tkf", "e"),
    ("ld_library_path_order", "tkg", "l1/libtkg.so=elf l2/libtkg.so=elf", "-", "l2:l1", "tkg", "-"),
    ("ld_library_path_empty_entries", "tkh", "l/libtkh.so=elf", "-", "::l:", "tkh", "-"),
    ("extra_paths_order", "tki", "e1/libtki.so=elf e2/libtki.so=elf", "-", "-", "tki", "e2 e1"),
    ("missing_dirs_skipped", "tkj", "e/libtkj.so=elf", "-", "gone", "tkj", "gone2 e"),
    ("linker_script_skipped", "tkk", "e/libtkk.so=text f/libtkk.so=elf", "-", "-", "tkk", "e f"),
    ("linker_script_beside_version", "tkl", "e/libtkl.so=text e/libtkl.so.2=elf", "-", "-", "tkl", "e"),
    ("version_digits_and_dots", "tkm", "e/libtkm.so.1.2.3=elf", "-", "-", "tkm", "e"),
    ("version_trailing_dot", "tkn", "e/libtkn.so.=elf", "-", "-", "tkn", "e"),
    ("version_letters_rejected", "tko", "e/libtko.so.1a=elf e/libtko.so.x=elf e/libtko.so.1.gz=elf", "-", "-", "tko", "e"),
    ("other_names_rejected", "tkp", "e/libtkp2.so=elf e/tkp.so=elf e/libtkp.dylib=elf e/tkp=elf e/libtkp.a=elf",
     "-", "-", "tkp", "e"),
    ("directory_named_as_library", "tkq", "e/libtkq.so=dir f/libtkq.so=elf", "-", "-", "tkq", "e f"),
    ("symbolic_link_to_elf", "tkr", "t/libtkr.so.1=elf e/libtkr.so=@../t/libtkr.so.1", "-", "-", "tkr", "e"),
    ("dangling_symbolic_link", "tks", "e/libtks.so=@nowhere f/libtks.so=elf", "-", "-", "tks", "e f"),
    ("absolute_file", "tkt", "a/whatever.txt=text e/libtkt.so=elf", "-", "-", "/a/whatever.txt tkt", "e"),
    ("absolute_missing_then_name", "tku", "e/libtku.so=elf", "-", "-", "/a/none.so tku", "e"),
    ("absolute_directory_skipped", "tkv", "a/sub=dir e/libtkv.so=elf", "-", "-", "/a/sub tkv", "e"),
    ("paths_before_directories", "tkw", "l/libtkwb.so=elf e/libtkwa.so=elf", "-", "l", "tkwa tkwb", "e"),
    ("not_found", "tkx", "e/other=text", "-", "-", "tkx", "e"),
    ("no_paths", "tky", "e/libtky.so=elf", "-", "-", "-", "e"),
]

DARWIN = [
    ("name_path_file", "td-a", "x/custom=text e/libtd-a.dylib=text", "x/custom", "-", "td-a", "e"),
    ("name_path_dir_first", "tdb", "n/libtdb.dylib=text l/libtdb.dylib=text e/libtdb.dylib=text", "n", "l", "tdb", "e"),
    ("ld_library_path_before_extra", "tdc", "l/libtdc.dylib=text e/libtdc.dylib=text", "-", "l", "tdc", "e"),
    ("ld_library_path_order", "tdp", "l1/libtdp.dylib=text l2/libtdp.dylib=text", "-", "l2:l1", "tdp", "-"),
    ("lib_dylib", "tdd", "e/libtdd.dylib=text", "-", "-", "tdd", "e"),
    ("plain_dylib", "tde", "e/tde.dylib=text", "-", "-", "tde", "e"),
    ("bare_name", "tdf", "e/tdf=text", "-", "-", "tdf", "e"),
    ("lib_dylib_first", "tdg", "e/libtdg.dylib=text e/tdg.dylib=text e/tdg=text", "-", "-", "tdg", "e"),
    ("plain_dylib_before_bare", "tdh", "e/tdh.dylib=text e/tdh=text", "-", "-", "tdh", "e"),
    ("directory_before_name", "tdi", "e1/tdi=text e2/libtdi.dylib=text", "-", "-", "tdi", "e1 e2"),
    ("elf_so_ignored", "tdj", "e/libtdj.so=elf e/libtdj.so.1=elf", "-", "-", "tdj", "e"),
    ("framework_dangling_link", "tdk", "e/tdk.framework/tdk=@Versions/A/tdk", "-", "-", "tdk", "e/tdk.framework"),
    ("dangling_link_outside_framework", "tdl", "e/tdl=@nowhere f/tdl=text", "-", "-", "tdl", "e f"),
    ("directory_named_as_library", "tdm", "e/libtdm.dylib=dir e/tdm.dylib=text", "-", "-", "tdm", "e"),
    ("absolute_file", "tdn", "a/whatever.txt=text e/libtdn.dylib=text", "-", "-", "/a/whatever.txt tdn", "e"),
    ("not_found", "tdo", "e/other=text", "-", "-", "tdo", "e"),
]


def build(root, tree):
    for entry in tree.split():
        path, kind = entry.split("=")
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if kind == "dir":
            target.mkdir()
        elif kind.startswith("@"):
            target.symlink_to(kind[1:])
        else:
            target.write_bytes(ELF if kind == "elf" else LINKER_SCRIPT)


def words(cell):
    return [] if cell == "-" else cell.split()


def findlib(platform, lib, tree, name_path, ld_library_path, paths, extra_paths):
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch).resolve()
        build(root, tree)
        var = lib.replace("-", "_").upper() + "_PATH"
        os.environ.pop(var, None)
        os.environ.pop("LD_LIBRARY_PATH", None)
        if name_path != "-":
            os.environ[var] = "" if name_path == '""' else str(root / name_path)
        if ld_library_path != "-":
            os.environ["LD_LIBRARY_PATH"] = ":".join(str(root / d) if d else "" for d in ld_library_path.split(":"))
        getenv.cache_clear()
        sys.platform, c.OSX = platform, platform == "darwin"
        found = c.DLL.findlib(lib, [str(root) + p if p.startswith("/") else p for p in words(paths)],
                              [str(root / d) for d in words(extra_paths)])
        if found is None:
            return "None"
        return str(Path(found).relative_to(root)) if found.startswith(str(root)) else found


@table
def findlib_trees():
    rows = [(platform, case, lib, tree, name_path, ld, paths, extra,
             findlib(platform, lib, tree, name_path, ld, paths, extra))
            for platform, cases in (("linux", LINUX), ("darwin", DARWIN))
            for case, lib, tree, name_path, ld, paths, extra in cases]
    return ["platform", "case", "lib", "tree", "name_path", "ld_library_path", "paths", "extra_paths", "found"], rows
