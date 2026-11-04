# /// script
# requires-python = ">=3.11"
# dependencies = ["tokenizers==0.23.1"]
# ///
"""Generate the Unigram scores as HuggingFace `tokenizers` reads them.

    uv run packages/brot/test/gen/gen_unigram_scores.py [--check]

With `--check` it writes nothing and fails if the fixture differs from what it
would write.

`tokenizers` parses JSON numbers with serde_json without its `float_roundtrip`
feature: the digits as an integer, then one multiplication or division by a
power of ten, which can land one unit in the last place from the double the
text names. A Unigram score read so can break an exact tie between two
segmentations, so brot reads scores the same way, and `test_unigram.ml` checks
it against `support/parity/unigram_scores.txt`, which this writes. Each value
is what `tokenizers` reads off the exact text of the row, in a model of its
own, written back through `to_str` and parsed by Python, which is exact.

Rows:

    READ <text> <value>     tokenizers reads <text> as <value> (hexadecimal)
    ERROR <text>            tokenizers refuses <text> as out of range
    SAVED <x> <text> <value>  brot writes the double <x> as <text>, which
                            tokenizers reads as <value>

The texts are drawn from a fixed seed to reach every branch of the reading and
its neighbours: scores as SentencePiece stores them (single precision), doubles
of every exponent, powers of two and their neighbours, 15-digit decimals below
10^-7, whole numbers about 10^15 and 10^16, the largest double, values past
10^300, subnormals, and each as Python prints it (the shortest decimal, as
`tokenizers` writes too) and as brot writes it (`%.17g`); then the texts other
writers make: integers past 2^64, exponents past 308 and past 2^31, fractions
with leading zeros, and the neighbours of each bound the reading keeps. `SAVED`
rows take brot's `%.17g`, which `test_unigram.ml` checks against the file brot
writes.
"""

import json
import math
import random
import struct
import sys
from pathlib import Path

from tokenizers import Tokenizer

TEST = Path(__file__).resolve().parent.parent
OUT = TEST / "support/parity/unigram_scores.txt"
SEED = 20261104


def single(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bits(b: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", b))[0]


def doubles(rng: random.Random) -> list[float]:
    xs = [-single(rng.uniform(0, 30)) for _ in range(800)]
    while len(xs) < 1300:
        x = bits(rng.getrandbits(64))
        if math.isfinite(x) and x != 0:
            xs.append(x)
    for e in rng.sample(range(-1074, 1024), 150):
        p = -math.ldexp(1.0, e)
        xs += [p, math.nextafter(p, 0), math.nextafter(p, -math.inf)]
    # 15 digits below 10^-7, where the power of ten passes 10^22.
    xs += [
        -float(f"{rng.randrange(10**14, 10**15)}e{rng.randrange(-23, -21)}")
        for _ in range(100)
    ]
    # Whole numbers, written out in full under 10^16.
    xs += [-float(rng.randrange(9 * 10**14, 10**15)) for _ in range(60)]
    xs += [-float(rng.randrange(10**15, 10**16)) for _ in range(60)]
    xs += [-float(rng.randrange(10**16, 10**17)) for _ in range(60)]
    big = sys.float_info.max
    below = math.nextafter(big, 0)
    xs += [big, -big, below, math.nextafter(below, 0)]
    xs += [
        rng.choice((1, -1)) * float(f"{rng.random():.17f}e{rng.randrange(300, 309)}")
        for _ in range(60)
    ]
    xs += [-bits(rng.randrange(1, 2**52)) for _ in range(60)]
    xs += [-0.5, -1.25, -3.0, -0.1, -1e-5, -1e-22, -1e-23, -5e-324, 0.0, -0.0]
    return [x for x in xs if math.isfinite(x)]


def lexemes(rng: random.Random) -> list[str]:
    out = []
    for _ in range(60):
        digits = str(rng.randrange(10**19, 10**40))
        out.append(rng.choice(("", "-")) + digits)
        out.append("-" + digits + "." + str(rng.randrange(10**30)))
    for _ in range(60):
        e = rng.choice((rng.randrange(300, 400), rng.randrange(-400, -300)))
        out.append(f"-{rng.randrange(1, 10**17)}e{e}")
        out.append(f"-0.{'0' * rng.randrange(1, 30)}{rng.randrange(1, 10**18)}")
        whole, part = rng.randrange(1, 10**9), rng.randrange(10**9)
        out.append(f"{whole}.{part}E+{rng.randrange(0, 290)}")
    out += [
        "0", "-0", "0.0", "-0.0", "1e-400", "-1e-400", "1E+0", "123456789e-330",
        "0.0001e-310", "1e-2147483648", "1e-2147483649", "-1e-99999999999",
        "0e999999999999", "-0e+99999999999", "5e400", "-1e309", "1e2147483648",
        "1.7976931348623157e308", "1.7976931348623158e308",
        "1.7976931348623159e308", "17976931348623157e292",
        "179769313486231570000000000000000000000e270",
        "18446744073709551615", "18446744073709551616", "18446744073709551614.5",
        "1844674407370955161.5", "1844674407370955161.6",
        "0.00000000000000000000000000000012345678901234567890123",
        "-4.5750198364257812", "-12.130167007446289",
        # The bit past 2^63 rounds the integer up; the last digit of the
        # fraction is the largest that still fits; zero keeps its sign under
        # an exponent past 308 but within 32 bits.
        "-9223372036854776833", "0.18446744073709551615",
        "0.18446744073709551616", "-0e1000000", "-0E2147483646",
    ]
    return out


def model(entries: list[str]) -> str:
    vocab = ",".join(f'["t{i}",{text}]' for i, text in enumerate(entries))
    head = '{"version":"1.0","model":{"type":"Unigram","unk_id":0,"vocab":'
    return head + '[["<unk>",0.0],' + vocab + "]}}"


def reads(text: str) -> float | None:
    try:
        tokenizer = Tokenizer.from_str(model([text]))
    except Exception:
        return None
    return json.loads(tokenizer.to_str())["model"]["vocab"][1][1]


def main() -> int:
    rng = random.Random(SEED)
    xs = doubles(rng)
    lines = [f"# tokenizers==0.23.1, seed {SEED}"]
    texts = [repr(x) for x in xs] + ["%.17g" % x for x in xs] + lexemes(rng)
    seen = set()
    for text in texts:
        if text in seen:
            continue
        seen.add(text)
        value = reads(text)
        if value is None:
            lines.append(f"ERROR {text}")
        else:
            lines.append(f"READ {text} {value.hex()}")
    for x in xs:
        text = "%.17g" % x
        value = reads(text)
        if value is not None:
            lines.append(f"SAVED {x!r} {text} {value.hex()}")
    out = "\n".join(lines) + "\n"
    if "--check" in sys.argv[1:]:
        if OUT.read_text() != out:
            print(f"{OUT} differs from what this would write", file=sys.stderr)
            return 1
        return 0
    OUT.write_text(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
