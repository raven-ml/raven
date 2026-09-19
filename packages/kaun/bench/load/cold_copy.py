"""Copies a file so that the copy's pages are not left in the file cache (macOS).

usage: cold_copy.py SRC DST

For a cold row of the load benchmark, copy a cached repository's files under a
scratch cache root with this script, point RAVEN_CACHE_ROOT at that root, and
run bench_load.exe once: the second run is warm.
"""
import fcntl, os, sys
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(os.path.dirname(dst), exist_ok=True)
fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
fcntl.fcntl(fd, 48, 1)  # F_NOCACHE
with open(src, "rb", buffering=0) as f:
    while True:
        b = f.read(1 << 24)
        if not b: break
        os.write(fd, b)
os.fsync(fd); os.close(fd)
