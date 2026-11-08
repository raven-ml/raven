# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Generate nx's special-function tables.

    uv run packages/nx/test/gen/special_tables.py [--check]

Writes `packages/nx/lib/special_tables.ml`, the coefficients, region
boundaries and counts of nx's special functions per compute dtype, and its
interface `special_tables.mli`, which documents each table. With --check
nothing is written, and the run fails if a file would change: the tables and
the proofs they carry disagree.

Coefficients come from their sources, checked against what each prints,
or are fitted here. Each count or degree is the least that meets its bound
against oracle.py's references, mpmath's functions, through a transcription
of nx's own composition in exact arithmetic; the goldens that hold nx to
those references are special.py's.
"""

import argparse
import difflib
import functools
import math
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import mpmath
from mpmath import mp, mpf

import oracle
from oracle import F32, F64, bits, of_bits, round_to

HERE = Path(__file__).resolve().parent
NX = HERE.parents[1]
ROOT = NX.parents[1]
TABLES = NX / "lib" / "special_tables.ml"
INTERFACE = NX / "lib" / "special_tables.mli"


# erfinv's guess, Giles' polynomials (Approximating the erfinv function, GPU
# Computing Gems Jade, 2011; erfinv_SP_1.cu and erfinv_DP_1.cu): a central
# polynomial in w - shift below a split in w, and tail polynomials in sqrt w -
# shift, the far one from a second split, as the source writes them, highest
# degree first. Single precision has one tail.

GILES = {
    F32: {
        "central": (5.0, 2.5, ["2.81022636e-08", "3.43273939e-07", "-3.5233877e-06", "-4.39150654e-06",
                               "0.00021858087", "-0.00125372503", "-0.00417768164", "0.246640727",
                               "1.50140941"]),
        "near": (3.0, ["-0.000200214257", "0.000100950558", "0.00134934322", "-0.00367342844",
                       "0.00573950773", "-0.0076224613", "0.00943887047", "1.00167406", "2.83297682"]),
        "far": None,
    },
    F64: {
        "central": (6.25, 3.125, [
            "-3.6444120640178196996e-21", "-1.685059138182016589e-19", "1.2858480715256400167e-18",
            "1.115787767802518096e-17", "-1.333171662854620906e-16", "2.0972767875968561637e-17",
            "6.6376381343583238325e-15", "-4.0545662729752068639e-14", "-8.1519341976054721522e-14",
            "2.6335093153082322977e-12", "-1.2975133253453532498e-11", "-5.4154120542946279317e-11",
            "1.051212273321532285e-09", "-4.1126339803469836976e-09", "-2.9070369957882005086e-08",
            "4.2347877827932403518e-07", "-1.3654692000834678645e-06", "-1.3882523362786468719e-05",
            "0.0001867342080340571352", "-0.00074070253416626697512", "-0.0060336708714301490533",
            "0.24015818242558961693", "1.6536545626831027356"]),
        "near": (3.25, [
            "2.2137376921775787049e-09", "9.0756561938885390979e-08", "-2.7517406297064545428e-07",
            "1.8239629214389227755e-08", "1.5027403968909827627e-06", "-4.013867526981545969e-06",
            "2.9234449089955446044e-06", "1.2475304481671778723e-05", "-4.7318229009055733981e-05",
            "6.8284851459573175448e-05", "2.4031110387097893999e-05", "-0.0003550375203628474796",
            "0.00095328937973738049703", "-0.0016882755560235047313", "0.0024914420961078508066",
            "-0.0037512085075692412107", "0.005370914553590063617", "1.0052589676941592334",
            "3.0838856104922207635"]),
        "far": (16.0, 5.0, [
            "-2.7109920616438573243e-11", "-2.5556418169965252055e-10", "1.5076572693500548083e-09",
            "-3.7894654401267369937e-09", "7.6157012080783393804e-09", "-1.4960026627149240478e-08",
            "2.9147953450901080826e-08", "-6.7711997758452339498e-08", "2.2900482228026654717e-07",
            "-9.9298272942317002539e-07", "4.5260625972231537039e-06", "-1.9681778105531670567e-05",
            "7.5995277030017761139e-05", "-0.00021503011930044477347", "-0.00013871931833623122026",
            "1.0103004648645343977", "4.8499064014085844221"]),
    },
}

# Where erfinv's Newton step changes residual: on erf below |y| = 1.25, where
# erfc_parts gives erf, on erfc above.
ERFINV_NEWTON_SPLIT = 1.25


def erfinv_split(fmt):
    """|p| at each split of w, and erf 1.25, at 100 bits."""
    g = GILES[fmt]
    ws = [g["central"][0]] + ([g["far"][0]] if g["far"] else [])
    with mp.workprec(100):
        return [mpmath.sqrt(1 - mpmath.exp(-mpf(w))) for w in ws] + [mpmath.erf(mpf(ERFINV_NEWTON_SPLIT))]


def erfinv_tables(fmt):
    """erfinv's per-dtype fields: the three polynomials padded with leading
    zeros to one length, so one Horner runs them with selected coefficients."""
    g = GILES[fmt]
    below, c_shift, central = g["central"]
    n_shift, near = g["near"]
    far_from, f_shift, far = g["far"] if g["far"] else (math.inf, n_shift, near)
    with mp.workprec(200):
        polys = [[round_to(fmt, mpf(d)) for d in cs] for cs in (central, near, far)]
        e125 = round_to(fmt, mpmath.erf(mpf(ERFINV_NEWTON_SPLIT)))
    n = max(len(cs) for cs in polys)
    polys = [[0.0] * (n - len(cs)) + cs for cs in polys]
    return [
        ("erfinv_central", ocaml_array(polys[0])),
        ("erfinv_central_below", ocaml_float(below)),
        ("erfinv_central_shift", ocaml_float(c_shift)),
        ("erfinv_near", ocaml_array(polys[1])),
        ("erfinv_near_shift", ocaml_float(n_shift)),
        ("erfinv_far", ocaml_array(polys[2])),
        ("erfinv_far_from", "infinity" if math.isinf(far_from) else ocaml_float(far_from)),
        ("erfinv_far_shift", ocaml_float(f_shift)),
        ("erfinv_newton_split", ocaml_float(e125)),
    ]
# Sources
#
# Coefficients as their sources print them: a decimal and, for fdlibm, the
# two words of its double, which the run checks against each other.

FDLIBM_NOTICE = ("Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved. Developed "
                 "at SunSoft, a Sun Microsystems, Inc. business. Permission to use, copy, "
                 "modify, and distribute this software is freely granted, provided that this "
                 "notice is preserved.")

# fdlibm s_erf.c, version 1.3 95/01/18.
S_ERF = {
    "erx": ("8.45062911510467529297e-01", 0x3FEB0AC160000000),
    "pp0": ("1.28379167095512558561e-01", 0x3FC06EBA8214DB68),
    "pp1": ("-3.25042107247001499370e-01", 0xBFD4CD7D691CB913),
    "pp2": ("-2.84817495755985104766e-02", 0xBF9D2A51DBD7194F),
    "pp3": ("-5.77027029648944159157e-03", 0xBF77A291236668E4),
    "pp4": ("-2.37630166566501626084e-05", 0xBEF8EAD6120016AC),
    "qq1": ("3.97917223959155352819e-01", 0x3FD97779CDDADC09),
    "qq2": ("6.50222499887672944485e-02", 0x3FB0A54C5536CEBA),
    "qq3": ("5.08130628187576562776e-03", 0x3F74D022C4D36B0F),
    "qq4": ("1.32494738004321644526e-04", 0x3F215DC9221C1A10),
    "qq5": ("-3.96022827877536812320e-06", 0xBED09C4342A26120),
    "pa0": ("-2.36211856075265944077e-03", 0xBF6359B8BEF77538),
    "pa1": ("4.14856118683748331666e-01", 0x3FDA8D00AD92B34D),
    "pa2": ("-3.72207876035701323847e-01", 0xBFD7D240FBB8C3F1),
    "pa3": ("3.18346619901161753674e-01", 0x3FD45FCA805120E4),
    "pa4": ("-1.10894694282396677476e-01", 0xBFBC63983D3E28EC),
    "pa5": ("3.54783043256182359371e-02", 0x3FA22A36599795EB),
    "pa6": ("-2.16637559486879084300e-03", 0xBF61BF380A96073F),
    "qa1": ("1.06420880400844228286e-01", 0x3FBB3E6618EEE323),
    "qa2": ("5.40397917702171048937e-01", 0x3FE14AF092EB6F33),
    "qa3": ("7.18286544141962662868e-02", 0x3FB2635CD99FE9A7),
    "qa4": ("1.26171219808761642112e-01", 0x3FC02660E763351F),
    "qa5": ("1.36370839120290507362e-02", 0x3F8BEDC26B51DD1C),
    "qa6": ("1.19844998467991074170e-02", 0x3F888B545735151D),
    "ra0": ("-9.86494403484714822705e-03", 0xBF843412600D6435),
    "ra1": ("-6.93858572707181764372e-01", 0xBFE63416E4BA7360),
    "ra2": ("-1.05586262253232909814e+01", 0xC0251E0441B0E726),
    "ra3": ("-6.23753324503260060396e+01", 0xC04F300AE4CBA38D),
    "ra4": ("-1.62396669462573470355e+02", 0xC0644CB184282266),
    "ra5": ("-1.84605092906711035994e+02", 0xC067135CEBCCABB2),
    "ra6": ("-8.12874355063065934246e+01", 0xC054526557E4D2F2),
    "ra7": ("-9.81432934416914548592e+00", 0xC023A0EFC69AC25C),
    "sa1": ("1.96512716674392571292e+01", 0x4033A6B9BD707687),
    "sa2": ("1.37657754143519042600e+02", 0x4061350C526AE721),
    "sa3": ("4.34565877475229228821e+02", 0x407B290DD58A1A71),
    "sa4": ("6.45387271733267880336e+02", 0x40842B1921EC2868),
    "sa5": ("4.29008140027567833386e+02", 0x407AD02157700314),
    "sa6": ("1.08635005541779435134e+02", 0x405B28A3EE48AE2C),
    "sa7": ("6.57024977031928170135e+00", 0x401A47EF8E484A93),
    "sa8": ("-6.04244152148580987438e-02", 0xBFAEEFF2EE749A62),
    "rb0": ("-9.86494292470009928597e-03", 0xBF84341239E86F4A),
    "rb1": ("-7.99283237680523006574e-01", 0xBFE993BA70C285DE),
    "rb2": ("-1.77579549177547519889e+01", 0xC031C209555F995A),
    "rb3": ("-1.60636384855821916062e+02", 0xC064145D43C5ED98),
    "rb4": ("-6.37566443368389627722e+02", 0xC083EC881375F228),
    "rb5": ("-1.02509513161107724954e+03", 0xC09004616A2E5992),
    "rb6": ("-4.83519191608651397019e+02", 0xC07E384E9BDC383F),
    "sb1": ("3.03380607434824582924e+01", 0x403E568B261D5190),
    "sb2": ("3.25792512996573918826e+02", 0x40745CAE221B9F0A),
    "sb3": ("1.53672958608443695994e+03", 0x409802EB189D5118),
    "sb4": ("3.19985821950859553908e+03", 0x40A8FFB7688C246A),
    "sb5": ("2.55305040643316442583e+03", 0x40A3F219CEDF3BE6),
    "sb6": ("4.74528541206955367215e+02", 0x407DA874E79FE763),
    "sb7": ("-2.24409524465858183362e+01", 0xC03670E242712D62),
}

# fdlibm e_lgamma_r.c, version 1.3 95/01/18.
E_LGAMMA = {
    "a0": ("7.72156649015328655494e-02", 0x3FB3C467E37DB0C8),
    "a1": ("3.22467033424113591611e-01", 0x3FD4A34CC4A60FAD),
    "a2": ("6.73523010531292681824e-02", 0x3FB13E001A5562A7),
    "a3": ("2.05808084325167332806e-02", 0x3F951322AC92547B),
    "a4": ("7.38555086081402883957e-03", 0x3F7E404FB68FEFE8),
    "a5": ("2.89051383673415629091e-03", 0x3F67ADD8CCB7926B),
    "a6": ("1.19270763183362067845e-03", 0x3F538A94116F3F5D),
    "a7": ("5.10069792153511336608e-04", 0x3F40B6C689B99C00),
    "a8": ("2.20862790713908385557e-04", 0x3F2CF2ECED10E54D),
    "a9": ("1.08011567247583939954e-04", 0x3F1C5088987DFB07),
    "a10": ("2.52144565451257326939e-05", 0x3EFA7074428CFA52),
    "a11": ("4.48640949618915160150e-05", 0x3F07858E90A45837),
    "tc": ("1.46163214496836224576e+00", 0x3FF762D86356BE3F),
    "tf": ("-1.21486290535849611461e-01", 0xBFBF19B9BCC38A42),
    "tt": ("-3.63867699703950536541e-18", 0xBC50C7CAA48A971F),
    "t0": ("4.83836122723810047042e-01", 0x3FDEF72BC8EE38A2),
    "t1": ("-1.47587722994593911752e-01", 0xBFC2E4278DC6C509),
    "t2": ("6.46249402391333854778e-02", 0x3FB08B4294D5419B),
    "t3": ("-3.27885410759859649565e-02", 0xBFA0C9A8DF35B713),
    "t4": ("1.79706750811820387126e-02", 0x3F9266E7970AF9EC),
    "t5": ("-1.03142241298341437450e-02", 0xBF851F9FBA91EC6A),
    "t6": ("6.10053870246291332635e-03", 0x3F78FCE0E370E344),
    "t7": ("-3.68452016781138256760e-03", 0xBF6E2EFFB3E914D7),
    "t8": ("2.25964780900612472250e-03", 0x3F6282D32E15C915),
    "t9": ("-1.40346469989232843813e-03", 0xBF56FE8EBF2D1AF1),
    "t10": ("8.81081882437654011382e-04", 0x3F4CDF0CEF61A8E9),
    "t11": ("-5.38595305356740546715e-04", 0xBF41A6109C73E0EC),
    "t12": ("3.15632070903625950361e-04", 0x3F34AF6D6C0EBBF7),
    "t13": ("-3.12754168375120860518e-04", 0xBF347F24ECC38C38),
    "t14": ("3.35529192635519073543e-04", 0x3F35FD3EE8C2D3F4),
    "u0": ("-7.72156649015328655494e-02", 0xBFB3C467E37DB0C8),
    "u1": ("6.32827064025093366517e-01", 0x3FE4401E8B005DFF),
    "u2": ("1.45492250137234768737e+00", 0x3FF7475CD119BD6F),
    "u3": ("9.77717527963372745603e-01", 0x3FEF497644EA8450),
    "u4": ("2.28963728064692451092e-01", 0x3FCD4EAEF6010924),
    "u5": ("1.33810918536787660377e-02", 0x3F8B678BBF2BAB09),
    "v1": ("2.45597793713041134822e+00", 0x4003A5D7C2BD619C),
    "v2": ("2.12848976379893395361e+00", 0x40010725A42B18F5),
    "v3": ("7.69285150456672783825e-01", 0x3FE89DFBE45050AF),
    "v4": ("1.04222645593369134254e-01", 0x3FBAAE55D6537C88),
    "v5": ("3.21709242282423911810e-03", 0x3F6A5ABB57D0CF61),
    "s0": ("-7.72156649015328655494e-02", 0xBFB3C467E37DB0C8),
    "s1": ("2.14982415960608852501e-01", 0x3FCB848B36E20878),
    "s2": ("3.25778796408930981787e-01", 0x3FD4D98F4F139F59),
    "s3": ("1.46350472652464452805e-01", 0x3FC2BB9CBEE5F2F7),
    "s4": ("2.66422703033638609560e-02", 0x3F9B481C7E939961),
    "s5": ("1.84028451407337715652e-03", 0x3F5E26B67368F239),
    "s6": ("3.19475326584100867617e-05", 0x3F00BFECDD17E945),
    "r1": ("1.39200533467621045958e+00", 0x3FF645A762C4AB74),
    "r2": ("7.21935547567138069525e-01", 0x3FE71A1893D3DCDC),
    "r3": ("1.71933865632803078993e-01", 0x3FC601EDCCFBDF27),
    "r4": ("1.86459191715652901344e-02", 0x3F9317EA742ED475),
    "r5": ("7.77942496381893596434e-04", 0x3F497DDACA41A95B),
    "r6": ("7.32668430744625636189e-06", 0x3EDEBAF7A5B38140),
    "w0": ("4.18938533204672725052e-01", 0x3FDACFE390C97D69),
    "w1": ("8.33333333333329678849e-02", 0x3FB555555555553B),
    "w2": ("-2.77777777728775536470e-03", 0xBF66C16C16B02E5C),
    "w3": ("7.93650558643019558500e-04", 0x3F4A019F98CF38B6),
    "w4": ("-5.95187557450339963135e-04", 0xBF4380CB8C0FE741),
    "w5": ("8.36339918996282139126e-04", 0x3F4B67BA4CDAD5D1),
    "w6": ("-1.63092934096575273989e-03", 0xBF5AB89D0B9E43E4),
}


def fdlibm(table, names):
    """The doubles of `names` in `table`, each checked against its words."""
    out = []
    for n in names:
        decimal, word = table[n]
        x = float(decimal)
        if bits(F64, x) != word:
            sys.exit(f"fdlibm {n}: {decimal} is not 0x{word:016X}")
        out.append(x)
    return out


def series(prefix, lo, hi):
    return [f"{prefix}{i}" for i in range(hi, lo - 1, -1)]


# Wichura, Algorithm AS 241, Applied Statistics 37 (1988), 477-484: PPND7
# and PPND16, coefficients lowest degree first, and the sums of their
# mantissas the paper prints for checking a transcription.
AS241 = {
    "PPND7": {
        "a": ["3.38713 27179E+00", "5.04342 71938E+01", "1.59291 13202E+02", "5.91093 74720E+01"],
        "b": ["1.78951 69469E+01", "7.87577 57664E+01", "6.71875 63600E+01"],
        "c": ["1.42343 72777E+00", "2.75681 53900E+00", "1.30672 84816E+00", "1.70238 21103E-01"],
        "d": ["7.37001 64250E-01", "1.20211 32975E-01"],
        "e": ["6.65790 51150E+00", "3.08122 63860E+00", "4.28682 94337E-01", "1.73372 03997E-02"],
        "f": ["2.41978 94225E-01", "1.22582 02635E-02"],
        "sums": {"ab": "32.31845 77772", "cd": "15.76149 29821", "ef": "19.40529 10204"},
    },
    "PPND16": {
        "a": ["3.38713 28727 96366 6080D0", "1.33141 66789 17843 7745D+2", "1.97159 09503 06551 4427D+3",
              "1.37316 93765 50946 1125D+4", "4.59219 53931 54987 1457D+4", "6.72657 70927 00870 0853D+4",
              "3.34305 75583 58812 8105D+4", "2.50908 09287 30122 6727D+3"],
        "b": ["4.23133 30701 60091 1252D+1", "6.87187 00749 20579 0830D+2", "5.39419 60214 24751 1077D+3",
              "2.12137 94301 58659 5867D+4", "3.93078 95800 09271 0610D+4", "2.87290 85735 72194 2674D+4",
              "5.22649 52788 52854 5610D+3"],
        "c": ["1.42343 71107 49683 57734D0", "4.63033 78461 56545 29590D0", "5.76949 72214 60691 40550D0",
              "3.64784 83247 63204 60504D0", "1.27045 82524 52368 38258D0", "2.41780 72517 74506 11770D-1",
              "2.27238 44989 26918 45833D-2", "7.74545 01427 83414 07640D-4"],
        "d": ["2.05319 16266 37758 82187D0", "1.67638 48301 83803 84940D0", "6.89767 33498 51000 04550D-1",
              "1.48103 97642 74800 74590D-1", "1.51986 66563 61645 71966D-2", "5.47593 80849 95344 94600D-4",
              "1.05075 00716 44416 84324D-9"],
        "e": ["6.65790 46435 01103 77720D0", "5.46378 49111 64114 36990D0", "1.78482 65399 17291 33580D0",
              "2.96560 57182 85048 91230D-1", "2.65321 89526 57612 30930D-2", "1.24266 09473 88078 43860D-3",
              "2.71155 55687 43487 57815D-5", "2.01033 43992 92288 13265D-7"],
        "f": ["5.99832 20655 58879 37690D-1", "1.36929 88092 27358 05310D-1", "1.48753 61290 85061 48525D-2",
              "7.86869 13114 56132 59100D-4", "1.84631 83175 10054 68180D-5", "1.42151 17583 16445 88870D-7",
              "2.04426 31033 89939 78564D-15"],
        "sums": {"ab": "55.88319 28806 14901 4439", "cd": "49.33206 50330 16102 89036",
                 "ef": "47.52583 31754 92896 71629"},
    },
}


def fortran(d):
    return mpf(d.replace(" ", "").replace("D", "E"))


def as241(name):
    """PPND7 or PPND16's coefficients as mpfs, its hash sums checked: the
    sum of each pair of polynomials' mantissas in [1, 10)."""
    t = AS241[name]
    mantissa = lambda d: Decimal(d.replace(" ", "").replace("D", "E").split("E")[0])
    for pair, total in t["sums"].items():
        got = sum(mantissa(d) for k in pair for d in t[k])
        if got != Decimal(total.replace(" ", "")):
            sys.exit(f"AS 241 {name} {pair}: mantissas sum to {got}, the paper prints {total}")
    with mp.workprec(200):
        return {k: [fortran(d) for d in t[k]] for k in "abcdef"}


# Proofs
#
# A count or a degree is the least that meets its bound, measured against
# mpmath on a dense grid of its region: the value within an eighth of the
# dtype's unit roundoff relative, and the derivative within the same,
# leaving the rest of each function's bound to rounding.

GRID = 400


def grid(lo, hi, n=GRID):
    return [lo + (hi - lo) * mpf(i) / (n - 1) for i in range(n)]


def worst(xs, approx, exact):
    return max(abs(approx(x) / exact(x) - 1) for x in xs)


def least(counts, ok):
    for n in counts:
        if ok(n):
            return n
    sys.exit("no count meets its bound")


# log_ndtr below its threshold: log Phi(x) = -x^2/2 - log(-x) - log(2 pi)/2
# + log (1 + sum_k (-1)^k (2k-1)!! / x^(2k)).

LOG_NDTR_BELOW = {F32: -10.0, F64: -20.0}


def log_ndtr_asymptotic(x, n):
    s = sum((-1) ** k * mpmath.fac2(2 * k - 1) / x ** (2 * k) for k in range(1, n + 1))
    return -x * x / 2 - mpmath.log(-x) - mpmath.log(2 * mpmath.pi) / 2 + mpmath.log1p(s)


def log_ndtr_terms(fmt):
    lo = mpf(LOG_NDTR_BELOW[fmt])
    xs = [lo * mpf(2) ** (i / 8) for i in range(0, 40)]
    with mp.workprec(200):
        def ok(n):
            value = worst(xs, lambda x: log_ndtr_asymptotic(x, n), lambda x: mpmath.log(mpmath.ncdf(x)))
            slope = worst(xs, lambda x: mpmath.diff(lambda y: log_ndtr_asymptotic(y, n), x),
                          lambda x: mpmath.npdf(x) / mpmath.ncdf(x))
            return value <= fmt.u / 8 and slope <= fmt.u / 8
        return least(range(1, 40), ok)


# digamma: on [1, 2], (x - x0) g(x - 3/2), x0 its root, g a polynomial; from
# DIGAMMA_FROM up, log x - 1/(2x) - sum_k B_2k / (2k x^2k).

DIGAMMA_FROM = {F32: 6.0, F64: 10.0}


def digamma_root():
    with mp.workprec(300):
        return mpmath.findroot(mpmath.digamma, mpf("1.4616321449683623"))


def split(fmt, x):
    hi = round_to(fmt, x)
    return hi, round_to(fmt, x - mpf(hi))


def digamma_core(fmt):
    """The coefficients of g, highest degree first, rounded to `fmt`: the
    least degree whose rounded coefficients put g within one unit roundoff
    and the derivative of (x - x0) g within sixteen."""
    with mp.workprec(300):
        x0 = digamma_root()
        g = lambda s: mpmath.digamma(s + mpf(3) / 2) / (s + mpf(3) / 2 - x0)
        xs = grid(mpf(-1) / 2, mpf(1) / 2)
        exact = [g(s) for s in xs]
        slopes = [mpmath.polygamma(1, s + mpf(3) / 2) for s in xs]
        for n in range(4, 60):
            coeffs = [mpf(round_to(fmt, c)) for c in mpmath.chebyfit(g, [-0.5, 0.5], n)]
            approx = [mpmath.polyval(coeffs, s, derivative=True) for s in xs]
            value = max(abs(a / e - 1) for (a, _), e in zip(approx, exact))
            if value > fmt.u:
                continue
            slope = max(abs((a + (s + mpf(3) / 2 - x0) * da) / d - 1)
                        for (a, da), s, d in zip(approx, xs, slopes))
            if slope <= 16 * fmt.u:
                return [float(c) for c in coeffs]
    sys.exit("no degree meets digamma's bound")


def digamma_asymptotic(x, n):
    return (mpmath.log(x) - 1 / (2 * x)
            - sum(mpmath.bernoulli(2 * k) / (2 * k * x ** (2 * k)) for k in range(1, n + 1)))


def digamma_terms(fmt):
    lo = mpf(DIGAMMA_FROM[fmt])
    xs = [lo * mpf(2) ** (i / 8) for i in range(0, 40)]
    with mp.workprec(200):
        def ok(n):
            value = worst(xs, lambda x: digamma_asymptotic(x, n), mpmath.digamma)
            slope = worst(xs, lambda x: mpmath.diff(lambda y: digamma_asymptotic(y, n), x),
                          lambda x: mpmath.polygamma(1, x))
            return value <= fmt.u / 8 and slope <= fmt.u / 8
        return least(range(1, 40), ok)


# sin (pi x) on [0, 1/2] as x S(x^2), S a polynomial: the least degree whose
# rounded coefficients put sin (pi x) within one unit roundoff and its
# derivative within sixteen of its largest, pi.


def sinpi_poly(fmt):
    with mp.workprec(200):
        g = lambda t: mpmath.sin(mpmath.pi * mpmath.sqrt(t)) / mpmath.sqrt(t) if t > 0 else mpmath.pi
        xs = grid(mpf(1) / 10 ** 6, mpf(1) / 2)
        exact = [mpmath.sin(mpmath.pi * x) for x in xs]
        slopes = [mpmath.pi * mpmath.cos(mpmath.pi * x) for x in xs]
        for n in range(3, 30):
            coeffs = [mpf(round_to(fmt, c)) for c in mpmath.chebyfit(g, [0, mpf(1) / 4], n)]
            approx = [mpmath.polyval(coeffs, x * x, derivative=True) for x in xs]
            value = max(abs(x * a / e - 1) for x, (a, _), e in zip(xs, approx, exact))
            if value > fmt.u:
                continue
            slope = max(abs(a + 2 * x * x * da - d) for x, (a, da), d in zip(xs, approx, slopes))
            if slope <= 16 * fmt.u * mpmath.pi:
                return [float(c) for c in coeffs]
    sys.exit("no degree meets sinpi's bound")


# Incomplete gamma
#
# P(a, x) and Q(a, x) = 1 - P, computed as the logarithm of one tail, after
# DiDonato and Morris (Computation of the incomplete gamma function ratios
# and their inverse, ACM TOMS 12 (1986); Algorithm 654, TOMS 13 (1987)).
# Every region runs on every element:
#
# - the corner, a < 1 and x < 1.1: P from x^a / Gamma(1 + a) (1 - j) and Q
#   from expm1 of its exponent, j a series in x;
# - the series for P, x below max(a, 1.1);
# - the continued fraction for Q, x from max(a, 1.1);
# - Temme's uniform expansion, a >= 20 and |x/a - 1| <= 0.4.
#
# The prefactor x^a e^-x / Gamma(a + 1) is exp of -bd0(a, x) - log (2 pi
# a)/2 - s(a) from a = 1 (Loader, Fast and accurate computation of binomial
# probabilities, 2000), s Stirling's correction, and of a log x - x - log
# Gamma(1 + a) below. bd0 = a (mu - log1p mu), mu = x/a - 1, is, in v = (x -
# a)/(x + a), 2 a v^2/(1 - v) - 2 a v^3 S(v^2), S(w) = sum_j w^(j-1)/(2j + 1),
# up to |v| = IGAMMA_NEAR_V and x - a - a log (x/a) beyond. The same S gives
# s(y) - s(y + 1) = z^2 S(z^2), z = 1/(2y + 1), which shifts s from [1, 8)
# past 8, where fdlibm's polynomial holds.

IGAMMA_CORNER = 1.1
IGAMMA_TEMME_FROM = 20.0
IGAMMA_TEMME_WIDTH = 0.4
IGAMMA_NEAR_V = 0.5
IGAMMA_SHIFTS = 7

# The largest shape the proofs and the goldens reach.
IGAMMA_A_MAX = 2.0 ** 20


def igamma_corner_j(a, x, n):
    """The corner's j / (a x), sum_k (-x)^(k-1) / (k! (a + k)) to n terms,
    nested backward as nx runs it."""
    s = mpf(0)
    for k in range(n, 0, -1):
        s = 1 / (mpmath.factorial(k) * (a + k)) - x * s
    return s


def igamma_series(a, x, n):
    """sum_k x^k / ((a + 1) ... (a + k)) to n terms."""
    s = mpf(1)
    for k in range(n, 0, -1):
        s = 1 + s * x / (a + k)
    return s


def igamma_cf(a, x, n):
    """Legendre's continued fraction for Q e^x x^-a Gamma(a), evaluated
    backward from depth n."""
    t = mpf(0)
    for k in range(n, 0, -1):
        t = k * (k - a) / (x + 2 * k + 1 - a - t)
    return 1 / (x + 1 - a - t)


# A derivative is held to its budget: in x, 16 times the value's; in a,
# 2^-44 at float64 and 2^-20 at float32, a sixteenth of rune's.
IGAMMA_DA = {F32: mpf(2) ** -20, F64: mpf(2) ** -44}


def igamma_count(fmt, f, exact, points):
    """The least n at which f is within u/8 of `exact`, relatively, at each
    point, and its derivatives in a and x within their budgets of exact's."""
    tol = mpf(fmt.u) / 8
    with mp.workprec(160):
        refs = []
        for a, x in points:
            a, x = mpf(a), mpf(x)
            slopes = (mpmath.diff(lambda s: exact(s, x), a), mpmath.diff(lambda s: exact(a, s), x))
            refs.append((a, x, exact(a, x), slopes))

        def ok(n):
            for a, x, r, (da, dx) in refs:
                if abs(f(a, x, n) / r - 1) > tol:
                    return False
                dv = mpmath.diff(lambda s: f(s, x, n), a)
                if abs(dv - da) > IGAMMA_DA[fmt] * max(abs(da), abs(r)):
                    return False
                dv = mpmath.diff(lambda s: f(a, s, n), x)
                if abs(dv - dx) > 16 * tol * max(abs(dx), abs(r)):
                    return False
            return True
        return least(range(1, 400), ok)


def lead_bits(a, x):
    """Guard bits for exp of a sum of terms of the size of a log x, x and log
    Gamma(a + 1)."""
    return oracle.guard(a * mpmath.log(x), x, mpmath.loggamma(a + 1))


def corner_exact(a, x):
    """The corner's sum from P: P Gamma(1 + a) x^-a = 1 - a x j, with bits
    for the difference, which is about a x."""
    with mp.workprec(mp.prec + lead_bits(a, x) + max(0, int(mpmath.log(1 / (a * x), 2))) + 8):
        lp, _ = oracle.igamma(a, x)
        return +((1 - mpmath.exp(lp + mpmath.loggamma(a + 1) - a * mpmath.log(x))) / (a * x))


def series_exact(a, x):
    """P's series from P: P Gamma(a + 1) e^x x^-a."""
    with mp.workprec(mp.prec + lead_bits(a, x)):
        lp, _ = oracle.igamma(a, x)
        return +mpmath.exp(lp + mpmath.loggamma(a + 1) + x - a * mpmath.log(x))


def cf_exact(a, x):
    """Q's continued fraction from Q: Q Gamma(a) e^x x^-a."""
    with mp.workprec(mp.prec + lead_bits(a, x)):
        _, lq = oracle.igamma(a, x)
        return +mpmath.exp(lq + mpmath.loggamma(a) + x - a * mpmath.log(x))


def igamma_corner_terms(fmt):
    """The corner's worst points are at x = 1.1."""
    x = mpf(IGAMMA_CORNER) * (1 - mpf(2) ** -40)
    points = [(a, x) for a in (1e-10, 1e-3, 0.1, 0.5, 0.9, 1 - 2.0 ** -30)]
    return igamma_count(fmt, igamma_corner_j, corner_exact, points)


def igamma_series_terms(fmt):
    """The series' worst points are at x = max(a, 1.1) below 20 and at x =
    0.6 a, Temme's lower edge, above."""
    below = [(a, max(a, IGAMMA_CORNER) * (1 - 2.0 ** -40)) for a in (1, 1.05, 1.5, 2, 3, 5, 8, 12, 16, 19.99)]
    above = [(a, (1 - IGAMMA_TEMME_WIDTH) * a) for a in (20, 50, 200, 1e3, 1e4, 2.0 ** 20)]
    return igamma_count(fmt, igamma_series, series_exact, below + above)


def igamma_cf_depth(fmt):
    """The continued fraction's worst points are at x = 1.1 below a = 1.1,
    at x = a up to 20, and at x = 1.4 a, Temme's upper edge, above."""
    low = [(a, IGAMMA_CORNER) for a in (1e-10, 0.01, 0.3, 0.7, 0.999, 1.05, 1.1)]
    mid = [(a, a) for a in (1.2, 1.5, 2, 3, 5.5, 8.3, 12.7, 19.99)]
    high = [(a, (1 + IGAMMA_TEMME_WIDTH) * a) for a in (20, 50, 200, 2.0 ** 20)]
    return igamma_count(fmt, igamma_cf, cf_exact, low + mid + high)


def atanh_s(w, n):
    """S(w) = sum_j w^(j-1) / (2j + 1) to n terms."""
    return sum(w ** (j - 1) / (2 * j + 1) for j in range(1, n + 1))


def igamma_atanh_terms(fmt):
    """S's count for bd0 up to |v| = IGAMMA_NEAR_V: mu - log1p mu = 2 v^2/(1 -
    v) - 2 v^3 S(v^2) within u/8 relatively, and its derivative in v."""
    vs = [v for v in grid(-mpf(IGAMMA_NEAR_V), mpf(IGAMMA_NEAR_V), 101) if v != 0]
    tol = mpf(fmt.u) / 8
    with mp.workprec(160):
        def phi(v, n):
            return 2 * v * v / (1 - v) - 2 * v ** 3 * atanh_s(v * v, n)

        def exact(v):
            mu = 2 * v / (1 - v)
            return mu - mpmath.log1p(mu)

        def ok(n):
            for v in vs:
                if abs(phi(v, n) / exact(v) - 1) > tol:
                    return False
                d = mpmath.diff(lambda s: phi(s, n), v)
                if abs(d / mpmath.diff(exact, v) - 1) > tol:
                    return False
            return True
        return least(range(1, 80), ok)


def igamma_shift_terms(fmt):
    """S's count in each of the shift's terms s(y) - s(y + 1) = z^2 S(z^2),
    y = a + k >= 1 + k: within u/64 absolutely, with its derivative, so that
    the seven are within u/8 together."""
    tol = mpf(fmt.u) / 64
    counts = []
    with mp.workprec(160):
        for k in range(IGAMMA_SHIFTS):
            def delta(y, n):
                z = 1 / (2 * y + 1)
                return z * z * atanh_s(z * z, n)

            def exact(y):
                return (y + mpf(1) / 2) * mpmath.log1p(1 / y) - 1

            ys = [mpf(1 + k) + mpf(i) / 8 for i in range(0, 8 * (8 - k - 1) + 1)]

            def ok(n):
                return all(abs(delta(y, n) - exact(y)) <= tol
                           and abs(mpmath.diff(lambda s: delta(s, n), y) - mpmath.diff(exact, y)) <= tol
                           for y in ys)
            counts.append(least(range(1, 80), ok))
    return counts


# Temme's expansion: Q = erfc (eta sqrt (a/2))/2 + e^(-y)/sqrt (2 pi a) T, P
# = erfc (-eta sqrt (a/2))/2 - e^(-y)/sqrt (2 pi a) T, y = a eta^2/2 = bd0,
# T = sum_k c_k(eta) a^-k. The coefficients d_kn of c_k = sum_n d_kn eta^n
# follow from c_0 = 1/mu - 1/eta and d_kn = (n + 2) d_(k-1)(n+2) + (-1)^k
# g_k d_0n (DLMF 8.12.12), g_k Stirling's coefficients of Gamma*. DiDonato
# and Morris print d_kn to 15 digits (TOMS 654's GRATIO, D0 to D70), which
# the run checks.

TEMME_ORDER = 72
TOMS654_D = {
    0: ["-.333333333333333", ".833333333333333E-01", "-.148148148148148E-01", ".115740740740741E-02",
        ".352733686067019E-03", "-.178755144032922E-03", ".391926317852244E-04", "-.218544851067999E-05",
        "-.185406221071516E-05", ".829671134095309E-06", "-.176659527368261E-06", ".670785354340150E-08",
        ".102618097842403E-07", "-.438203601845335E-08"],
    1: ["-.185185185185185E-02", "-.347222222222222E-02", ".264550264550265E-02", "-.990226337448560E-03",
        ".205761316872428E-03", "-.401877572016461E-06", "-.180985503344900E-04", ".764916091608111E-05",
        "-.161209008945634E-05", ".464712780280743E-08", ".137863344691572E-06", "-.575254560351770E-07",
        ".119516285997781E-07"],
    2: [".413359788359788E-02", "-.268132716049383E-02", ".771604938271605E-03", ".200938786008230E-05",
        "-.107366532263652E-03", ".529234488291201E-04", "-.127606351886187E-04", ".342357873409614E-07",
        ".137219573090629E-05", "-.629899213838006E-06", ".142806142060642E-06"],
    3: [".649434156378601E-03", ".229472093621399E-03", "-.469189494395256E-03", ".267720632062839E-03",
        "-.756180167188398E-04", "-.239650511386730E-06", ".110826541153473E-04", "-.567495282699160E-05",
        ".142309007324359E-05"],
    4: ["-.861888290916712E-03", ".784039221720067E-03", "-.299072480303190E-03", "-.146384525788434E-05",
        ".664149821546512E-04", "-.396836504717943E-04", ".113757269706784E-04"],
    5: ["-.336798553366358E-03", "-.697281375836586E-04", ".277275324495939E-03", "-.199325705161888E-03",
        ".679778047793721E-04"],
    6: [".531307936463992E-03", "-.592166437353694E-03", ".270878209671804E-03"],
    7: [".344367606892378E-03"],
}


def series_mul(p, q):
    out = [mpf(0)] * TEMME_ORDER
    for i, x in enumerate(p):
        if x:
            for j in range(TEMME_ORDER - i):
                out[i + j] += x * q[j]
    return out


def temme_coefficients():
    """d_kn for k < 16, n < TEMME_ORDER - 2k, at 400 bits, checked against
    TOMS 654's table."""
    with mp.workprec(400):
        n = TEMME_ORDER
        # eta = mu q(mu), q = sqrt (2 (mu - log1p mu)/mu^2).
        two_psi = [mpf(2) * (-1) ** i / (i + 2) for i in range(n)]
        q = [mpf(0)] * n
        q[0] = mpmath.sqrt(two_psi[0])
        for i in range(1, n):
            q[i] = (two_psi[i] - sum(q[j] * q[i - j] for j in range(1, i))) / (2 * q[0])
        f = [mpf(0)] + q[:n - 1]
        # mu = g(eta), the reversion of f, by fixed-point iteration.
        g = [mpf(0), mpf(1)] + [mpf(0)] * (n - 2)
        for _ in range(n):
            comp, power = [mpf(0)] * n, [mpf(1)] + [mpf(0)] * (n - 1)
            for k in range(1, n):
                power = series_mul(power, g)
                comp = [c + f[k] * pk for c, pk in zip(comp, power)]
            g = [gi - (ci - (1 if i == 1 else 0)) for i, (gi, ci) in enumerate(zip(g, comp))]
        h = g[1:] + [mpf(0)]
        inv = [mpf(0)] * n
        inv[0] = 1 / h[0]
        for i in range(1, n):
            inv[i] = -sum(h[j] * inv[i - j] for j in range(1, i + 1)) / h[0]
        d0 = inv[1:] + [mpf(0)]
        # Gamma*(a) ~ sum g_k a^-k = exp (sum_j B_2j / (2j (2j - 1) a^(2j - 1))).
        m = 40
        lg = [mpf(0)] * m
        for j in range(1, m // 2):
            lg[2 * j - 1] = mpmath.bernoulli(2 * j) / (2 * j * (2 * j - 1))
        gs = [mpf(1)] + [mpf(0)] * (m - 1)
        for i in range(1, m):
            gs[i] = sum(k * lg[k] * gs[i - k] for k in range(1, i + 1)) / i
        d = [d0]
        for k in range(1, 16):
            prev = d[-1]
            d.append([(i + 2) * prev[i + 2] + (-1) ** k * gs[k] * d0[i] if i + 2 < n - 2 * k else mpf(0)
                      for i in range(n)])
        for k, printed in TOMS654_D.items():
            for i, p in enumerate(printed):
                # 15 significant digits: within half a unit of the 15th.
                if abs(d[k][i] - mpf(p)) > abs(mpf(p)) * mpf(10) ** -14 / 2:
                    sys.exit(f"Temme d_{k}{i}: {mpmath.nstr(d[k][i], 17)}, TOMS 654 prints {p}")
        return d


TEMME = temme_coefficients()


def temme_eta(lam):
    mu = lam - 1
    e = mpmath.sqrt(2 * (mu - mpmath.log1p(mu)))
    return e if mu >= 0 else -e


def temme_eta_max():
    return max(abs(temme_eta(1 - mpf(IGAMMA_TEMME_WIDTH))), abs(temme_eta(1 + mpf(IGAMMA_TEMME_WIDTH))))


def temme_tail(a, x, degrees):
    """The smaller tail by Temme's expansion with c_k truncated to the given
    degrees, as nx computes it."""
    lam = x / a
    eta = temme_eta(lam)
    t = sum(sum(TEMME[k][n] * eta ** n for n in range(deg + 1)) / a ** k for k, deg in enumerate(degrees))
    pref = mpmath.exp(-a * eta * eta / 2) / mpmath.sqrt(2 * mpmath.pi * a)
    if lam >= 1:
        return mpmath.erfc(eta * mpmath.sqrt(a / 2)) / 2 + pref * t
    return mpmath.erfc(-eta * mpmath.sqrt(a / 2)) / 2 - pref * t


def temme_exact(a, x):
    lp, lq = oracle.igamma(a, x)
    return mpmath.exp(lq if x >= a else lp)


def igamma_temme(fmt):
    """Temme's count K and each c_k's degree: the least degree whose
    truncation, and that of its derivative, is within u/64 at a = 20 and
    |eta| at the window's edges, of the scaled tail sqrt (2 pi a) e^y tail,
    at least 2; then the least K within u/8 of the exact tail at a = 20,
    with its derivatives within their budgets."""
    tol = mpf(fmt.u)
    with mp.workprec(200):
        em = temme_eta_max()
        a0 = mpf(IGAMMA_TEMME_FROM)

        def degree(k):
            d = TEMME[k]
            for deg in range(0, TEMME_ORDER - 2 * k - 2):
                rest = sum(abs(d[n]) * em ** n for n in range(deg + 1, TEMME_ORDER - 2 * k - 2))
                slope = sum(n * abs(d[n]) * em ** (n - 1) for n in range(deg + 1, TEMME_ORDER - 2 * k - 2))
                if max(rest, slope) / a0 ** k <= 2 * tol / 64:
                    return deg
            sys.exit(f"no degree meets Temme's c_{k}")
        lams = [1 - mpf(IGAMMA_TEMME_WIDTH) + mpf(i) / 20 for i in range(17)]
        lams = [l for l in lams if l != 1] + [mpf(1)]
        points = [(a, a * l) for a in (a0, a0 * 2, a0 * 8) for l in lams]

        def ok(k_count):
            degrees = [degree(k) for k in range(k_count)]
            for a, x in points:
                v, r = temme_tail(a, x, degrees), temme_exact(a, x)
                if abs(v / r - 1) > tol / 8:
                    return False
                da = mpmath.diff(lambda s: temme_tail(s, x, degrees), a) - mpmath.diff(lambda s: temme_exact(s, x), a)
                dx = mpmath.diff(lambda s: temme_tail(a, s, degrees), x) - mpmath.diff(lambda s: temme_exact(a, s), x)
                scale_a = max(abs(mpmath.diff(lambda s: temme_exact(s, x), a)), r)
                scale_x = max(abs(mpmath.diff(lambda s: temme_exact(a, s), x)), r)
                if abs(da) > IGAMMA_DA[fmt] * scale_a or abs(dx) > 2 * tol * scale_x:
                    return False
            return True
        k_count = least(range(1, 16), ok)
        return [degree(k) for k in range(k_count)]


# erfc x e^(x^2) beyond fdlibm's last region, from ERFC_FAR_FROM: (1/(x
# sqrt pi)) sum_k (-1)^k (2k - 1)!! / (2 x^2)^k, its least count within u/8
# with its derivative.


def erfcx_far_terms(fmt):
    tol = mpf(fmt.u) / 8
    xs = [mpf(ERFC_FAR_FROM) * mpf(2) ** (i / 4) for i in range(40)]
    with mp.workprec(200):
        def approx(x, n):
            return sum((-1) ** k * mpmath.fac2(2 * k - 1) / (2 * x * x) ** k for k in range(n)) / (x * mpmath.sqrt(mpmath.pi))

        def exact(x):
            return mpmath.exp(x * x) * mpmath.erfc(x)
        return least(range(1, 40), lambda n: worst(xs, lambda x: approx(x, n), exact) <= tol
                     and worst(xs, lambda x: mpmath.diff(lambda s: approx(s, n), x),
                               lambda x: mpmath.diff(exact, x)) <= tol)


# GAMINV's normal quantile on the smaller tail q, as TOMS 654 prints it: t -
# A(t)/B(t) for t = sqrt (-2 log q), A and B lowest degree first, B's
# constant 1.
GAMINV_A = [3.31125922108741, 11.6616720288968, 4.28342155967104, .213623493715853]
GAMINV_B = [6.61053765625462, 6.40691597760039, 1.27364489782223, .036117081018842]

# GAMINV sums P's series at its guess until a term is below 1e-4, at a guess
# below 0.7 (a + 1): a term ratio below 0.7, so 26 terms.
GAMINV_TERMS = math.ceil(math.log(1e-4) / math.log(0.7))
EULER = mpf("0.577215664901532860606512090082402431")


def gaminv_asymptotic(a, y):
    """GAMINV's large-x form from y = -log (q Gamma(a))."""
    s = 1 - a
    c1 = -s * mpmath.log(y)
    c2 = -s * (1 + c1)
    c3 = s * ((c1 / 2 + (2 - a)) * c1 + (mpf(5) / 2 - mpf(3) / 2 * a))
    c4 = -s * (((c1 / 3 + (mpf(5) / 2 - mpf(3) / 2 * a)) * c1 + ((a - 6) * a + 7)) * c1
               + ((11 * a - 46) * a + 47) / 6)
    c5 = -s * ((((-c1 / 4 + (11 * a - 17) / 6) * c1 + ((-3 * a + 13) * a - 13)) * c1
                + (((2 * a - 25) * a + 72) * a - 61) / 2) * c1
               + (((25 * a - 195) * a + 477) * a - 379) / 12)
    return ((((c5 / y + c4) / y + c3) / y + c2) / y + c1) + y


def gaminv_guess(a, l, q_tail):
    """nx's GAMINV guess for log x, branch for branch: a > 0, l the smaller
    tail's logarithm, q_tail whether that tail is Q."""
    lc = mpmath.log1p(-mpmath.exp(l))
    lq, lp = (l, lc) if q_tail else (lc, l)
    if a < 1:
        lg1 = mpmath.loggamma(1 + a)
        y = -(lq + lg1 - mpmath.log(a))
        b = mpmath.exp(-y)
        if b < mpf("0.45"):
            s = 1 - a
            tt = y - s * mpmath.log(y)
            if a < mpf("0.3") and b >= mpf("0.35"):
                e = -(b + EULER)
                return e + mpmath.exp(e) * mpmath.exp(mpmath.exp(e))
            if b >= mpf("0.15"):
                return mpmath.log(y - s * mpmath.log(tt) - mpmath.log1p(s / (tt + 1)))
            if b > mpf("0.01"):
                u = ((tt + 2 * (3 - a)) * tt + (2 - a) * (3 - a)) / ((tt + (5 - a)) * tt + 2)
                return mpmath.log(y - s * mpmath.log(tt) - mpmath.log(u))
            return mpmath.log(gaminv_asymptotic(a, y))
        if lq - y < mpmath.log(mpf("1e-8")):
            power = -(mpmath.exp(lq) / a + EULER)
        else:
            power = (lp + lg1) / a
        ratio = min(mpmath.exp(power) / (a + 1), mpf("0.9"))
        return power - mpmath.log1p(-ratio)
    lg1 = mpmath.loggamma(a + 1)
    y = -(lq + lg1 - mpmath.log(a))
    r = mpmath.sqrt(-2 * l)
    n = r - mpmath.polyval(list(reversed(GAMINV_A)), r) / mpmath.polyval(list(reversed(GAMINV_B)) + [1], r)
    n = n if q_tail else -n
    ra, n2 = mpmath.sqrt(a), n * n
    xc = (a + n * ra + (n2 - 1) / 3 + n * (n2 - 7) / (36 * ra)
          + n * ((9 * n2 + 256) * n2 - 433) / (38880 * a * ra) - ((3 * n2 + 7) * n2 - 16) / (810 * a))
    xc = xc if xc > 0 else mpf("1e-30")
    ap1 = a + 1
    if q_tail and xc >= 3 * a:
        if y >= 0 and y >= mpmath.log(10) * max(a * (a - 1), mpf(2)):
            return mpmath.log(gaminv_asymptotic(a, y))
        x = xc
        for _ in range(2):
            x = y + (a - 1) * mpmath.log(x) - mpmath.log1p(-(a - 1) / (x + 1))
        return mpmath.log(x)
    if q_tail or xc > mpf("0.7") * ap1:
        return mpmath.log(xc)
    w = lp + lg1
    ap2, ap3 = a + 2, a + 3
    fixed = lambda x: (w + x - mpmath.log1p((x / ap1) * (1 + x / ap2))) / a
    x = mpmath.exp(fixed(mpmath.exp(fixed(mpmath.exp(w / a)))))
    y4 = (w + x - mpmath.log1p((x / ap1) * (1 + (x / ap2) * (1 + x / ap3)))) / a
    x4 = mpmath.exp(y4)
    tiny = xc <= mpf("0.15") * ap1
    if tiny and x4 <= mpf("0.01") * ap1:
        return y4
    xs = min(x4 if tiny else xc, mpf("0.7") * ap1)
    term, total = mpf(1), mpf(1)
    for k in range(1, GAMINV_TERMS + 1):
        term *= xs / (a + k)
        total += term
    tw = w - mpmath.log(total)
    x5 = mpmath.exp((xs + tw) / a)
    d = a - x5
    step = (a * mpmath.log(x5) - x5 - tw) / (d if abs(d) > mpf("1e-3") else 1)
    return mpmath.log(x5) + mpmath.log1p(-(step if abs(step) < mpf("0.5") else 0))


def last_step(f, s, lam, l1, l2):
    """nx's last step of a quantile and its first two derivatives in the
    target l, whose residual F = f has the derivative -1 in l: the root h of
    F(t + h) to fifth order in Newton's step d = -f / s, from F' = s and the
    derivatives lam, l1 and l2 of g, the logarithm of the tail's derivative
    in t, with s = +-e^(g - log T); Newton's step where Halley's correction
    c2 d passes 1/2."""
    d = -f / s
    u = lam - s
    u1 = l1 - s * u
    u2 = l2 - s * (u * u + u1)
    c2, c3, c4 = u / 2, (u * u + u1) / 6, (u ** 3 + 3 * u * u1 + u2) / 24
    if abs(c2 * d) >= mpf(1) / 2:
        return d, 1 / s, mpf(0)
    k3, k4 = 2 * c2 ** 2 - c3, 5 * c2 * (c3 - c2 ** 2) - c4
    h = d - c2 * d ** 2 + k3 * d ** 3 + k4 * d ** 4
    hl = (1 - 2 * c2 * d + 3 * k3 * d ** 2 + 4 * k4 * d ** 3) / s
    hll = (-2 * c2 + 6 * k3 * d + 12 * k4 * d ** 2) / s ** 2
    return h, hl, hll


def quantile_held(fmt, got, exact, budget, normal):
    """Whether nx's last step `got`, x and its first two derivatives in l,
    meets the quantile `exact`, x, its derivatives and the scale of the
    second's terms: x within u/8, relatively, or within an eighth of the
    least subnormal where the quantile is subnormal and nx skips the step;
    and where rune.mli's budgets hold, `normal`, the first derivative within
    an eighth of the `budget` in ulps and the second within sqrt(eps) / 8 of
    its terms' scale."""
    x, dx, d2x = got
    xs, dxs, d2xs, terms = exact
    if xs < fmt.min_normal:
        return abs(x - xs) <= mpf(fmt.tiny) / 8
    if abs(x / xs - 1) > mpf(fmt.u) / 8:
        return False
    eps = mpf(fmt.eps)
    return not normal or (abs(dx / dxs - 1) <= budget * eps / 8
                          and abs(d2x - d2xs) <= mpmath.sqrt(eps) / 8 * terms)


def igamma_slope(a, x, q_tail, l):
    """x, log T - l and the slope of log T in log x at x."""
    lp, lq = oracle.igamma(a, x)
    lt = lq if q_tail else lp
    slope = mpmath.exp(a * mpmath.log(x) - x - mpmath.loggamma(a) - lt) * (-1 if q_tail else 1)
    return lt - l, slope


def gaminv_steps(a, l, q_tail, y, count):
    """y = log x after `count` of nx's Halley steps in log x, in exact
    arithmetic."""
    for _ in range(count):
        x = mpmath.exp(y)
        g, slope = igamma_slope(a, x, q_tail, l)
        d = g / slope
        den = 1 - g * (a - x - slope) / (2 * slope)
        d = d / den if den > mpf("0.5") else d
        y -= max(min(d, 1), -1)
    return y


def gaminv_last(fmt, a, l, q_tail, y):
    """x after nx's last step from y = log x, x e^h by its series through
    h^4, and its first two derivatives in l; where x is not normal or |h| is
    at least 1/2 nx skips the step, and x is e^y, constant."""
    x = mpmath.exp(y)
    if x < fmt.min_normal:
        return x, mpf(0), mpf(0)
    f, slope = igamma_slope(a, x, q_tail, l)
    h, hl, hll = last_step(f, slope, a - x, -x, -x)
    if abs(h) >= mpf(1) / 2:
        return x, mpf(0), mpf(0)
    e2 = 1 + h + h ** 2 / 2
    e1 = e2 + h ** 3 / 6
    return x * (e1 + h ** 4 / 24), x * e1 * hl, x * (e2 * hl ** 2 + e1 * hll)


def igamma_halley_steps(fmt):
    """The least Halley count after which nx's last step meets the quantile
    as `quantile_held` states, the first derivative within rune.mli's budget
    in p, 64 + 16 kappa (32 (1 + |log m|) + 4) ulps for kappa = m / (x f(x))
    = 1 / |s|, f the density, from the guess at every point of a grid of a
    up to 2^20 and of the smaller tail down to the least subnormal, either
    tail. At the quantile the slope s of log T in y = log x
    gives dx/dl = x / s and d2x/dl2 = x (1 - u) / s^2, u = a - x - s."""
    least_log = math.log(fmt.tiny)
    tails = [least_log * mpf(2) ** -k for k in (0, 1, 2, 4, 6, 8, 10)] + [mpf("-0.6931"), mpf(-1.5), mpf(-3), mpf(-7)]
    with mp.workprec(128):
        cases = [(mpf(10) ** (mpf(e) / 2), l, q) for e in range(-8, 13) for l in tails for q in (False, True)]
        cases += [(mpf(IGAMMA_A_MAX), l, q) for l in tails for q in (False, True)]
        exact = []
        for a, l, q in cases:
            xs = oracle.igamma_quantile(a, mpmath.exp(l), q)
            if not xs > 0:
                continue
            _, s = igamma_slope(a, xs, q, l)
            u = a - xs - s
            budget = 64 + 16 * (32 * (1 + abs(l)) + 4) / abs(s)
            exact.append(((a, l, q), (xs, xs / s, xs * (1 - u) / s ** 2, xs * (1 + abs(u)) / s ** 2), budget))
        for count in range(1, 6):
            def ok(case):
                (a, l, q), ex, budget = case
                y = gaminv_steps(a, l, q, gaminv_guess(a, l, q), count)
                return quantile_held(fmt, gaminv_last(fmt, a, l, q, y), ex, budget, True)
            if all(ok(c) for c in exact):
                return count
    sys.exit("no Halley count meets the inverse's bound")


def igamma_tables(fmt):
    """The incomplete gamma's per-dtype fields."""
    j = igamma_atanh_terms(fmt)
    shifts = igamma_shift_terms(fmt)
    assert max(shifts) <= j
    degrees = igamma_temme(fmt)
    with mp.workprec(200):
        temme = [round_to(fmt, TEMME[k][n]) for k, deg in enumerate(degrees) for n in range(deg, -1, -1)]
        atanh = [round_to(fmt, mpf(1) / (2 * i + 1)) for i in range(j, 0, -1)]
    return [
        ("igamma_corner_terms", ocaml_float(float(igamma_corner_terms(fmt)))),
        ("igamma_series_terms", ocaml_float(float(igamma_series_terms(fmt)))),
        ("igamma_cf_depth", ocaml_float(float(igamma_cf_depth(fmt)))),
        ("igamma_atanh", ocaml_array(atanh)),
        ("igamma_shift_terms", ocaml_array([float(n) for n in shifts])),
        ("igamma_temme", ocaml_array(temme)),
        ("igamma_temme_degrees", ocaml_array([float(d) for d in degrees])),
        ("igamma_halley_steps", ocaml_float(float(igamma_halley_steps(fmt)))),
        ("beta_halley_steps", ocaml_float(float(beta_halley_steps(fmt)))),
        ("erfcx_series", ocaml_array([float((-1) ** k * mpmath.fac2(2 * k - 1))
                                      for k in range(erfcx_far_terms(fmt) - 1, -1, -1)])),
    ]


# The per-dtype record

def per_dtype(fmt):
    """Fields of the per-dtype record and their OCaml values, with the
    comment each carries."""
    ppnd = as241("PPND7" if fmt is F32 else "PPND16")
    high_first = lambda cs: [round_to(fmt, c) for c in reversed(cs)]
    one_last = lambda cs: [round_to(fmt, c) for c in reversed(cs)] + [1.0]
    k_ndtr = log_ndtr_terms(fmt)
    k_psi = digamma_terms(fmt)
    x0 = digamma_root()
    hi, lo = split(fmt, x0)
    core = digamma_core(fmt)
    with mp.workprec(200):
        c_hi, c_lo = split(fmt, 1 / mpmath.sqrt(2))
    return erfinv_tables(fmt) + [
        ("sqrt1_2_hi", ocaml_float(c_hi)),
        ("sqrt1_2_lo", ocaml_float(c_lo)),
        ("ndtri_central_p", ocaml_array(high_first(ppnd["a"]))),
        ("ndtri_central_q", ocaml_array(one_last(ppnd["b"]))),
        ("ndtri_near_p", ocaml_array(high_first(ppnd["c"]))),
        ("ndtri_near_q", ocaml_array(one_last(ppnd["d"]))),
        ("ndtri_far_p", ocaml_array(high_first(ppnd["e"]))),
        ("ndtri_far_q", ocaml_array(one_last(ppnd["f"]))),
        ("log_ndtr_below", ocaml_float(LOG_NDTR_BELOW[fmt])),
        ("log_ndtr_series", ocaml_array([float((-1) ** k * mpmath.fac2(2 * k - 1)) for k in range(k_ndtr, 0, -1)])),
        ("digamma_root_hi", ocaml_float(hi)),
        ("digamma_root_lo", ocaml_float(lo)),
        ("digamma_core", ocaml_array(core)),
        ("digamma_from", ocaml_float(DIGAMMA_FROM[fmt])),
        ("digamma_series", ocaml_array([round_to(fmt, mpmath.bernoulli(2 * k) / (2 * k)) for k in range(k_psi, 0, -1)])),
        ("sinpi", ocaml_array(sinpi_poly(fmt))),
    ] + bessel_tables(fmt) + igamma_tables(fmt) + beta_tables(fmt)


# The tables both dtypes share, highest degree first; a dtype rounds each
# coefficient to itself once.

def word(w):
    return of_bits(F64, w << 32)


# lgamma's regions on (0, 2), as fdlibm compares the high word of x.
LGAMMA_SMALL = word(0x3FECCCCD)   # x <= 0.9: lgamma (x + 1) - log x
LGAMMA_C = word(0x3FE76944)       # [0.7316, 0.9] about 1, in 1 - x
LGAMMA_B = word(0x3FCDA661)       # [0.2316, 0.7316) about the minimum
LGAMMA_F = word(0x3FFBB4C3)       # [1.7316, 2) about 2, in 2 - x
LGAMMA_E = word(0x3FF3B4C4)       # [1.2316, 1.7316) about the minimum


# erfc's regions in |x|, as fdlibm splits them: erf's P/Q below 0.84375, two
# forms of 1 - erf on either side of 1/4, P/Q about 1 to 1.25, the tails' R/S
# below and from 1/0.35, and 0 or 2 from 28.
ERFC_SMALL_SPLIT = 0.25
ERF_SMALL_BELOW = 0.84375
ERF_NEAR_BELOW = 1.25
ERFC_MID_BELOW = 1 / 0.35
ERFC_FAR_FROM = 28.0

# ndtri's regions, as AS 241 splits them: |p - 1/2| <= 0.425 central, in
# 0.180625 - (p - 1/2)^2; then r = sqrt (-log q), q the nearer tail, shifted by
# 1.6 up to 5 and by 5 above.
NDTRI_CENTRAL_BELOW = 0.425
NDTRI_CENTRAL_R = 0.180625
NDTRI_NEAR_BELOW = 5.0
NDTRI_NEAR_SHIFT = 1.6
NDTRI_FAR_SHIFT = 5.0

# lgamma's regions from 2, as fdlibm splits them: [2, 8) shifted down to
# [2, 3), Stirling's series from 8.
LGAMMA_SHIFT_FROM = 2.0
LGAMMA_STIRLING_FROM = 8.0


def shared_tables():
    erf = lambda names: fdlibm(S_ERF, names)
    lg = lambda names: fdlibm(E_LGAMMA, names)
    one = [1.0]
    return [
        ("erx", ocaml_float(erf(["erx"])[0])),
        ("erf_small_p", ocaml_array(erf(series("pp", 0, 4)))),
        ("erf_small_q", ocaml_array(erf(series("qq", 1, 5)) + one)),
        ("erf_near_p", ocaml_array(erf(series("pa", 0, 6)))),
        ("erf_near_q", ocaml_array(erf(series("qa", 1, 6)) + one)),
        ("erfc_mid_p", ocaml_array(erf(series("ra", 0, 7)))),
        ("erfc_mid_q", ocaml_array(erf(series("sa", 1, 8)) + one)),
        ("erfc_far_p", ocaml_array(erf(series("rb", 0, 6)))),
        ("erfc_far_q", ocaml_array(erf(series("sb", 1, 7)) + one)),
        ("erfc_small_split", ocaml_float(ERFC_SMALL_SPLIT)),
        ("erf_small_below", ocaml_float(ERF_SMALL_BELOW)),
        ("erf_near_below", ocaml_float(ERF_NEAR_BELOW)),
        ("erfc_mid_below", ocaml_float(ERFC_MID_BELOW)),
        ("erfc_far_from", ocaml_float(ERFC_FAR_FROM)),
        ("ndtri_central_below", ocaml_float(NDTRI_CENTRAL_BELOW)),
        ("ndtri_central_r", ocaml_float(NDTRI_CENTRAL_R)),
        ("ndtri_near_below", ocaml_float(NDTRI_NEAR_BELOW)),
        ("ndtri_near_shift", ocaml_float(NDTRI_NEAR_SHIFT)),
        ("ndtri_far_shift", ocaml_float(NDTRI_FAR_SHIFT)),
        ("lgamma_tc", ocaml_float(lg(["tc"])[0])),
        ("lgamma_tf", ocaml_float(lg(["tf"])[0])),
        ("lgamma_tt", ocaml_float(lg(["tt"])[0])),
        ("lgamma_a", ocaml_array(lg(series("a", 0, 11)))),
        ("lgamma_t", ocaml_array(lg(series("t", 0, 14)))),
        ("lgamma_u", ocaml_array(lg(series("u", 0, 5)))),
        ("lgamma_v", ocaml_array(lg(series("v", 1, 5)) + one)),
        ("lgamma_s", ocaml_array(lg(series("s", 0, 6)))),
        ("lgamma_r", ocaml_array(lg(series("r", 1, 6)) + one)),
        ("lgamma_w0", ocaml_float(lg(["w0"])[0])),
        ("lgamma_w", ocaml_array(lg(series("w", 1, 6)))),
        ("lgamma_small", ocaml_float(LGAMMA_SMALL)),
        ("lgamma_c", ocaml_float(LGAMMA_C)),
        ("lgamma_b", ocaml_float(LGAMMA_B)),
        ("lgamma_f", ocaml_float(LGAMMA_F)),
        ("lgamma_e", ocaml_float(LGAMMA_E)),
        ("lgamma_shift_from", ocaml_float(LGAMMA_SHIFT_FROM)),
        ("lgamma_stirling_from", ocaml_float(LGAMMA_STIRLING_FROM)),
        ("bessel_split", ocaml_float(BESSEL_SPLIT)),
        ("bessel_weight", ocaml_float(BESSEL_WEIGHT)),
        ("igamma_corner", ocaml_float(IGAMMA_CORNER)),
        ("igamma_temme_from", ocaml_float(IGAMMA_TEMME_FROM)),
        ("igamma_temme_width", ocaml_float(IGAMMA_TEMME_WIDTH)),
        ("igamma_near_v", ocaml_float(IGAMMA_NEAR_V)),
        ("igamma_guess_p", ocaml_array([float(c) for c in reversed(GAMINV_A)])),
        ("igamma_guess_q", ocaml_array([float(c) for c in reversed(GAMINV_B)] + [1.0])),
        ("igamma_guess_terms", ocaml_float(float(GAMINV_TERMS))),
        ("beta_bpser_bx", ocaml_float(BETA_BPSER_BX)),
        ("beta_bup_x", ocaml_float(BETA_BUP_X)),
        ("beta_power_bx", ocaml_float(BETA_POWER_BX)),
        ("beta_power_x", ocaml_float(BETA_POWER_X)),
        ("beta_a_small", ocaml_float(BETA_A_SMALL)),
        ("beta_x_far", ocaml_float(BETA_X_FAR)),
        ("beta_x_near", ocaml_float(BETA_X_NEAR)),
        ("beta_bgrat_b", ocaml_float(BETA_BGRAT_B)),
        ("beta_b_small", ocaml_float(BETA_B_SMALL)),
        ("beta_frac_a", ocaml_float(BETA_FRAC_A)),
        ("beta_frac_lambda", ocaml_float(BETA_FRAC_LAMBDA)),
        ("beta_bup_terms", str(BETA_BUP_TERMS)),
        ("beta_bup_most", str(BETA_BUP_MOST)),
        ("beta_large", ocaml_float(BETA_LARGE)),
        ("rlog1_series", ocaml_float(RLOG1_SERIES)),
        ("beta_guess_terms", str(BETA_GUESS_TERMS)),
        ("beta_guess_newton", str(BETA_GUESS_NEWTON)),
        ("beta_guess_gap", ocaml_float(BETA_GUESS_GAP)),
        ("beta_guess_bx", ocaml_float(BETA_GUESS_BX)),
        ("beta_guess_ratio", ocaml_float(BETA_GUESS_RATIO)),
        ("beta_guess_temme_from", ocaml_float(BETA_GUESS_TEMME_FROM)),
        ("beta_guess_gamma_from", ocaml_float(BETA_GUESS_GAMMA_FROM)),
    ] + beta_temme_tables()

# i0e and i1e: e^-|x| I0 x and e^-|x| I1 x. On [0, BESSEL_SPLIT], i0e x h =
# 1 + x q and i1e x / x h^2 for h = 1 + BESSEL_WEIGHT x, q and the latter
# Chebyshev series in x (2 / BESSEL_SPLIT) - 1: i0e is 1 at 0 by
# construction, and the weight flattens each, so that the series' terms, and
# their rounding, stay of the order of its value. Above,
# sqrt x i0e x and sqrt x i1e x in 2 BESSEL_SPLIT / x - 1.

BESSEL_SPLIT = 8.0
BESSEL_WEIGHT = 0.5


def chebyshev(f, n):
    """The degree-n Chebyshev interpolant of f on [-1, 1], lowest degree
    first."""
    m = n + 1
    angles = [mpmath.pi * (j + mpf(1) / 2) / m for j in range(m)]
    values = [f(mpmath.cos(a)) for a in angles]
    cs = [2 * sum(v * mpmath.cos(k * a) for v, a in zip(values, angles)) / m for k in range(m)]
    cs[0] /= 2
    return cs


def clenshaw(cs, t):
    """sum_k cs[k] T_k t and its derivative in t."""
    b1 = b2 = d1 = d2 = mpf(0)
    for c in reversed(cs[1:]):
        b1, b2, d1, d2 = c + 2 * t * b1 - b2, b1, 2 * b1 + 2 * t * d1 - d2, d1
    return cs[0] + t * b1 - b2, b1 + t * d1 - d2


def bessel_series(fmt, f, lift=lambda t, v: v, lift_slope=lambda t, v, dv: dv, units=1):
    """The coefficients of f's series on [-1, 1], rounded to `fmt`: the least
    degree whose rounded coefficients put the function the series gives,
    lift(t, series), within `units` unit roundoffs of its value and its
    derivative within sixteen of the derivative's largest magnitude."""
    with mp.workprec(200):
        ts = grid(mpf(-1) + mpf(10) ** -30, mpf(1), 200)
        exact = [(lift(t, f(t)), lift_slope(t, f(t), mpmath.diff(f, t))) for t in ts]
        steepest = max(abs(d) for _, d in exact)
        for n in range(3, 60):
            cs = [mpf(round_to(fmt, c)) for c in chebyshev(f, n)]
            approx = [(lift(t, v), lift_slope(t, v, dv)) for t in ts for v, dv in [clenshaw(cs, t)]]
            if max(abs(a / e - 1) for (a, _), (e, _) in zip(approx, exact)) > units * fmt.u:
                continue
            if max(abs(da - d) for (_, da), (_, d) in zip(approx, exact)) <= 16 * fmt.u * steepest:
                return [float(c) for c in cs]
    sys.exit("no degree meets the Bessel series' bound")


def bessel_tables(fmt):
    split, w = mpf(BESSEL_SPLIT), mpf(BESSEL_WEIGHT)
    near = lambda t: split / 2 * (t + 1)
    far = lambda t: 2 * split / (t + 1)
    i1_near = lambda x: oracle.i1e(x) / x if x > 0 else mpf(1) / 2
    # q = (i0e x h - 1) / x, its limit -1 + w at 0; the series checked as 1 +
    # x q, the function i0e h is, and its slope as q + x q'.
    q = lambda t: (oracle.i0e(near(t)) * (1 + w * near(t)) - 1) / near(t) if t > -1 else w - 1
    lift = lambda t, v: 1 + near(t) * v
    lift_slope = lambda t, v, dv: split / 2 * v + near(t) * dv
    i1 = lambda t: i1_near(near(t)) * (1 + w * near(t)) ** 2
    with mp.workprec(200):
        # The weight keeps each series' terms of the order of its value: x q
        # stays below 1 + x q, and i1e's coefficients sum within twice its
        # least value.
        ts = grid(mpf(-1), mpf(1), 100)
        assert max(abs(near(t) * q(t) / lift(t, q(t))) for t in ts) < 1
        assert sum(abs(c) for c in chebyshev(i1, 40)) < 2 * min(abs(i1(t)) for t in ts)
    return [
        # x q's coefficients, each rounded, leave 1 + x q 2u off at float64.
        ("i0e_near", ocaml_array(bessel_series(fmt, q, lift, lift_slope, units=4))),
        ("i1e_near", ocaml_array(bessel_series(fmt, i1))),
        ("i0e_far", ocaml_array(bessel_series(fmt, lambda t: mpmath.sqrt(far(t)) * oracle.i0e(far(t))))),
        ("i1e_far", ocaml_array(bessel_series(fmt, lambda t: mpmath.sqrt(far(t)) * oracle.i1e(far(t))))),
    ]

def beta_region(fmt, a, b, x):
    """The TOMS 708 method special.ml selects for finite a, b > 0 and x in
    (0, 1), as it selects it."""
    eps = BETA_EPS[fmt]
    small = a <= 1 or b <= 1
    lam = a - (a + b) * x
    swap = x > 0.5 if small else lam < 0
    a0, b0, x0 = (b, a, 1 - x) if swap else (a, b, x)
    lam0 = -lam if swap else lam
    if small:
        if b0 < eps and b0 < eps * a0:
            return "bpser"
        if a0 < eps and a0 < eps * b0 and b0 * x0 <= 1:
            return "bpser_tiny_a"
        both = a0 <= 1 and b0 <= 1
        far = x0 >= BETA_X_FAR
        if both:
            if a0 >= BETA_A_SMALL or a0 >= b0 or a0 * math.log(x0) <= math.log(BETA_POWER_X):
                return "bpser"
            return "bpser_y" if far else "bup_bgrat"
        near_pow = x0 < BETA_X_NEAR and a0 * math.log(x0 * b0) <= math.log(BETA_POWER_BX)
        if b0 <= 1 or (not far and near_pow):
            return "bpser"
        if far:
            return "bpser_y"
        return "bgrat" if b0 > BETA_BGRAT_B else "bup_bgrat"
    if b0 < BETA_B_SMALL:
        if b0 * x0 <= BETA_BPSER_BX:
            return "bpser"
        if x0 <= BETA_BUP_X:
            return "bup_bpser"
        return "bup_bup_bgrat" if a0 <= BETA_BGRAT_B else "bup_bgrat_large"
    low = min(a0, b0)
    if low <= BETA_FRAC_A or lam0 > BETA_FRAC_LAMBDA * low:
        return "bfrac"
    return "basym"


BETA_REGIONS = ["bpser", "bpser_tiny_a", "bpser_y", "bup_bgrat", "bgrat", "bup_bpser", "bup_bup_bgrat",
                "bup_bgrat_large", "bfrac", "basym"]

# TOMS 708's regions (DiDonato and Morris, Algorithm 708, ACM TOMS 18 (1992)):
# the boundaries the composition switches on and the goldens' points
# surround.

BETA_BPSER_BX = 0.7      # both > 1: BPSER while b0 x0 <= 0.7 and b0 < BETA_B_SMALL
BETA_BUP_X = 0.7         # b0 < BETA_B_SMALL: BPSER after BUP while x0 <= 0.7, else BGRAT
BETA_POWER_BX = 0.7      # (x0 b0)^a0 <= 0.7 with x0 < BETA_X_NEAR: BPSER
BETA_POWER_X = 0.9       # x0^a0 <= 0.9: BPSER
BETA_A_SMALL = 0.2       # a0 >= min(0.2, b0): BPSER
BETA_X_FAR = 0.3         # x0 >= 0.3: BPSER on the complement
BETA_X_NEAR = 0.1
BETA_BGRAT_B = 15.0      # b0 > 15: BGRAT alone; a0 <= 15: BUP of 20 terms first
BETA_B_SMALL = 40.0
BETA_FRAC_A = 100.0      # BFRAC up to 100, or where lambda > 0.2 of the smaller
BETA_FRAC_LAMBDA = 0.2   # TOMS 708 hands over at 0.03, where BFRAC needs twice the depth
BETA_BUP_TERMS = 20
BETA_BUP_MOST = 39       # the largest masked count, floor b0 - 1 or so below 40
BETA_LARGE = 8.0         # x^a y^b / B(a, b) by its large-argument form from 8
RLOG1_SERIES = 0.6       # e - log1p e by its series where |e| <= 0.6
BETA_EPS = {F32: 2.0 ** -23, F64: 1e-15}

# The beta's quantile: its first guess's choices and counts. Its Halley
# steps are proven below, after the guess, which beta_guess mirrors.

BETA_GUESS_TERMS = 3       # terms of 2F1 in the power forms, and their steps
BETA_GUESS_NEWTON = 8      # Newton steps on Temme's equation for x
BETA_GUESS_GAP = 0.05      # a power form's neglected terms below which it is taken
BETA_GUESS_BX = 0.7        # and the b x up to which its series is BPSER's
BETA_GUESS_RATIO = 4.0     # the gamma's form where one shape exceeds the other 4 times,
BETA_GUESS_TEMME_FROM = 20.0   # the smaller is below 20,
BETA_GUESS_GAMMA_FROM = 3.0    # and the larger at least 3

BOOST_NOTICE = ("Copyright John Maddock 2006. Copyright Paul A. Bristow 2007. Use, "
                "modification and distribution are subject to the Boost Software License, "
                "Version 1.0: Permission is hereby granted, free of charge, to any person or "
                "organization obtaining a copy of the software and accompanying documentation "
                "covered by this license (the \"Software\") to use, reproduce, display, "
                "distribute, execute, and transmit the Software, and to prepare derivative "
                "works of the Software, and to permit third-parties to whom the Software is "
                "furnished to do so, all subject to the following: The copyright notices in "
                "the Software and this entire statement, including the above license grant, "
                "this restriction and the following disclaimer, must be included in all "
                "copies of the Software, in whole or in part, and all derivative works of the "
                "Software, unless such copies or derivative works are solely in the form of "
                "machine-executable object code generated by a source language processor. "
                "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR "
                "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, "
                "FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT "
                "SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE FOR "
                "ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, "
                "ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER "
                "DEALINGS IN THE SOFTWARE.")

# Temme's corrections e1, e2 and e3 of eta = eta0 + e1/r + e2/r^2 + e3/r^3
# (Temme, Asymptotic inversion of the incomplete beta function, J. Comput.
# Appl. Math. 41 (1992), p. 151), as Boost's ibeta_inverse.hpp
# (temme_method_2_ibeta_inverse) transcribes them: e_k is sum_j c_kj(s^2)
# eta0^j / (d_kj (s c)^n_kj), s^2 = a / (a + b) and c^2 = 1 - s^2, each
# entry (d, n, c ascending in s^2). beta_temme_check measures them against
# the exact eta.

BETA_TEMME = [
    [(3, 1, [-1, 2]),
     (-36, 2, [-1, -5, 5]),
     (1620, 3, [1, 21, -69, 46]),
     (-6480, 4, [7, -2, 33, -62, 31]),
     (90720, 5, [25, -52, -17, 88, -115, 46])],
    [(-405, 3, [7, 12, -78, 52]),
     (2592, 4, [-7, 2, 183, -370, 185]),
     (-204120, 5, [-533, 776, -1835, 10240, -13525, 5410]),
     (-2099520, 6, [-1579, 3747, -3372, -15821, 45588, -45213, 15071])],
    [(102060, 5, [449, -1259, -769, 6686, -9260, 3704]),
     (-20995200, 6, [63149, -151557, 140052, -727469, 2239932, -2251437, 750479]),
     (36741600, 7, [29233, -78755, 105222, 146879, -1602610, 3195183, -2554139, 729754])],
]


def beta_temme_eta(x0, eta0, r, k=3):
    """eta0 + e1/r + ... + ek/r^k at s^2 = x0."""
    sc = mpmath.sqrt(x0 * (1 - x0))
    e = [sum(mpmath.polyval(list(reversed(c)), x0) * eta0 ** j / (d * sc ** n)
             for j, (d, n, c) in enumerate(row)) for row in BETA_TEMME]
    return eta0 + sum(e[i] / r ** (i + 1) for i in range(k))


def beta_temme_check():
    """Each correction lowers the error against the exact eta, at r = 400 and
    4000 on both sides of the mean: e1 about r-fold, as Temme's order says;
    e2 and e3 by less than their order would, so the transcription is held
    to no order past e1, and the guess only by the Halley count's proof."""
    with mp.workprec(200):
        for x0 in (mpf("0.3"), mpf("0.5"), mpf("0.8")):
            for eta0 in (mpf(-1) / 2, mpf(1) / 4):
                for r in (mpf(400), mpf(4000)):
                    a, b = x0 * r, (1 - x0) * r
                    if eta0 < 0:
                        z = oracle.beta_quantile_logit(a, b, mpmath.ncdf(eta0 * mpmath.sqrt(r)), False)
                    else:
                        z = oracle.beta_quantile_logit(a, b, mpmath.ncdf(-eta0 * mpmath.sqrt(r)), True)
                    x = 1 / (1 + mpmath.exp(-z))
                    f = x0 * mpmath.log(x / x0) + (1 - x0) * mpmath.log((1 - x) / (1 - x0))
                    eta = mpmath.sign(x - x0) * mpmath.sqrt(-2 * f)
                    errors = [abs(eta - beta_temme_eta(x0, eta0, r, k)) for k in range(4)]
                    assert errors[1] < errors[0] * 20 / r, ("e1", x0, eta0, r)
                    assert errors[3] < errors[1], ("e2, e3", x0, eta0, r)


def beta_temme_tables():
    beta_temme_check()
    rows = [(d, n, c) for row in BETA_TEMME for d, n, c in row]
    return [
        ("beta_temme", ocaml_matrix([[float(v) for v in reversed(c)] for _, _, c in rows])),
        ("beta_temme_scale", ocaml_array([float(d) for d, _, _ in rows])),
        ("beta_temme_power", ocaml_array([float(n) for _, n, _ in rows])),
        ("beta_temme_terms", ocaml_array([float(len(row)) for row in BETA_TEMME])),
    ]


# nx's guess for the beta's quantile, branch for branch, in exact arithmetic:
# z = log (x / (1 - x)) from a, b > 0, the smaller tail's logarithm l and
# whether that tail is Q.

def beta_softplus(z):
    return max(z, 0) + mpmath.log1p(mpmath.exp(-abs(z)))


def beta_sigmoid(z):
    return mpmath.exp(-beta_softplus(-z))


def beta_power_guess(a, b, lp, la, z_lo, z_hi):
    def series(z):
        x = beta_sigmoid(z)
        term, total = mpf(1), mpf(0)
        for k in range(1, BETA_GUESS_TERMS + 1):
            term *= (k - b) / k * x
            total += term / (a + k)
        k = BETA_GUESS_TERMS + 1
        if x > mpf(1) / 2 or b * x > BETA_GUESS_BX or a * total <= mpf(-1) / 2:
            return total, mpmath.inf
        return total, abs(term * (k - b) / k * x / (a + k))
    z = z_lo
    for _ in range(BETA_GUESS_TERMS):
        total, _ = series(z)
        if a * total <= -1:
            break
        lx = min((lp + la - mpmath.log1p(a * total)) / a, -mpf(2) ** -52)
        z = min(max(lx - mpmath.log(-mpmath.expm1(lx)), z_lo), z_hi)
    return z, series(z)[1]


def beta_gamma_guess(a, b, l, q_tail):
    nu = b + (a - 1) / 2
    u = mpmath.exp(gaminv_guess(a, l, q_tail) - mpmath.log(nu))
    return mpmath.log(-mpmath.expm1(-u)) + u


def beta_temme_guess(a, b, m, l, q_tail):
    r = a + b
    x0 = a / r
    n = -oracle.ndtri(m) if q_tail else oracle.ndtri(m)
    eta = beta_temme_eta(x0, n / mpmath.sqrt(r), r)
    z0 = mpmath.log(x0) - mpmath.log1p(-x0)
    target = -eta * eta / 2 + x0 * mpmath.log(x0) + (1 - x0) * mpmath.log1p(-x0)
    z = z0 + eta / mpmath.sqrt(x0 * (1 - x0))
    for _ in range(BETA_GUESS_NEWTON):
        x = beta_sigmoid(z)
        f = x0 * -beta_softplus(-z) + (1 - x0) * -beta_softplus(z) - target
        slope = x0 - x
        nxt = z if slope == 0 else z - f / slope
        z = z0 + (z - z0) / 2 if (nxt - z0) * eta < 0 else nxt
    return z


def beta_guess(fmt, a, b, l, q_tail):
    m = mpmath.exp(l)
    lc = mpmath.log1p(-m)
    lq, lp = (l, lc) if q_tail else (lc, l)
    lB = mpmath.loggamma(a) + mpmath.loggamma(b) - mpmath.loggamma(a + b)
    la, lb = mpmath.log(a) + lB, mpmath.log(b) + lB
    far = -2 * mpmath.log(fmt.tiny)
    finite = lambda z: min(max(z, -far), far)
    z_lo, z_hi = finite((lp + la) / a), finite(-(lq + lb) / b)
    z_p, gap_p = beta_power_guess(a, b, lp, la, z_lo, z_hi)
    z_q, gap_q = beta_power_guess(b, a, lq, lb, -z_hi, -z_lo)
    lo, hi = min(a, b), max(a, b)
    gamma = lo < BETA_GUESS_TEMME_FROM and hi >= BETA_GUESS_GAMMA_FROM and hi > BETA_GUESS_RATIO * lo
    small = not gamma and hi < BETA_GUESS_GAMMA_FROM and lo < 1
    if gap_p < BETA_GUESS_GAP:
        z = z_p
    elif gap_q < BETA_GUESS_GAP:
        z = -z_q
    elif gamma:
        z = beta_gamma_guess(a, b, l, q_tail) if a <= b else -beta_gamma_guess(b, a, l, not q_tail)
    elif small:
        z = z_p if gap_p <= gap_q else -z_q
    else:
        z = beta_temme_guess(a, b, m, l, q_tail)
    return min(max(z, z_lo), z_hi), z_lo, z_hi


def beta_halley(a, b, l, q_tail, z, z_lo, z_hi):
    """nx's Halley step in z on the smaller tail's logarithm, as the step in
    z and the new z within [z_lo, z_hi]."""
    lp, lq = oracle.beta_logit(a, b, z)
    lt = lq if q_tail else lp
    lx, ly = -beta_softplus(-z), -beta_softplus(z)
    lB = mpmath.loggamma(a) + mpmath.loggamma(b) - mpmath.loggamma(a + b)
    slope = mpmath.exp(a * lx + b * ly - lB - lt) * (-1 if q_tail else 1)
    lam = a - (a + b) * beta_sigmoid(z)
    d = (lt - l) / slope
    den = 1 - d * (lam - slope) / 2
    d = d / den if mpf(1) / 2 < den < mpf(3) / 2 else d
    return d, min(max(z - d, z_lo), z_hi)


def beta_switch_neighbours(tails):
    """Problems on both sides of each switch of the guess: shapes beside
    BETA_GUESS_GAMMA_FROM, BETA_GUESS_TEMME_FROM and 1, pairs beside the
    ratio BETA_GUESS_RATIO, and quantiles beside b x = BETA_GUESS_BX and
    beside the power form's gap of BETA_GUESS_GAP, each as (a, b, l, q)."""
    near = lambda v: [mpf(v) * (1 - mpf(2) ** -20), mpf(v) * (1 + mpf(2) ** -20)]
    edges = near(BETA_GUESS_GAMMA_FROM) + near(BETA_GUESS_TEMME_FROM) + near(1)
    others = [mpf(v) for v in ("0.3", "0.9", "4.5", "30", "1e3")]
    out = [(a, b, l, q) for e in edges for o in others for a, b in ((e, o), (o, e))
           for l in tails for q in (False, True)]
    for lo in (mpf("0.3"), mpf(2), mpf(6), mpf(15)):
        for hi in near(BETA_GUESS_RATIO * lo):
            out += [(a, b, l, q) for a, b in ((lo, hi), (hi, lo)) for l in tails for q in (False, True)]

    def at(a, b, x):
        """Both tails' problems at the quantile x."""
        z = mpmath.log(x) - mpmath.log1p(-x)
        lp, lq = oracle.beta_logit(a, b, z)
        return [(a, b, lp, False) if lp <= lq else (a, b, lq, True)]

    def gap(a, b, x):
        term = mpf(1)
        for k in range(1, BETA_GUESS_TERMS + 1):
            term *= (k - b) / k * x
        k = BETA_GUESS_TERMS + 1
        return abs(term * (k - b) / k * x / (a + k))

    for a in (mpf("0.5"), mpf(3), mpf(40)):
        for b in (mpf(2), mpf(10), mpf(300)):
            for x in near(BETA_GUESS_BX / b):
                out += at(a, b, x) + [(b2, a2, l, not q) for a2, b2, l, q in at(b, a, 1 - x)]
            lo, hi = mpf(0), mpf(1) / 2
            if gap(a, b, hi) > BETA_GUESS_GAP:
                for _ in range(60):
                    mid = (lo + hi) / 2
                    lo, hi = (mid, hi) if gap(a, b, mid) < BETA_GUESS_GAP else (lo, mid)
                for x in near(lo):
                    out += at(a, b, x)
    return out


def beta_slope(a, b, z, l, q_tail):
    """log T - l and the slope of log T in z at the logit z."""
    lp, lq = oracle.beta_logit(a, b, z)
    lt = lq if q_tail else lp
    lx, ly = -beta_softplus(-z), -beta_softplus(z)
    lB = mpmath.loggamma(a) + mpmath.loggamma(b) - mpmath.loggamma(a + b)
    return lt - l, mpmath.exp(a * lx + b * ly - lB - lt) * (-1 if q_tail else 1)


def beta_last_step(fmt, a, b, l, q_tail, z):
    """x after nx's last step of the beta's quantile from z, x(z + h) by its
    Taylor series through h^4, and its first two derivatives in l; where x
    or 1 - x is not normal or |h| is at least 1/2 nx skips the step, and x
    is that of z, constant."""
    x, y = beta_sigmoid(z), beta_sigmoid(-z)
    if min(x, y) < fmt.min_normal:
        return x, mpf(0), mpf(0)
    f, slope = beta_slope(a, b, z, l, q_tail)
    w = x * y
    l1 = -(a + b) * w
    h, hl, hll = last_step(f, slope, a - (a + b) * x, l1, l1 * (y - x))
    if abs(h) >= mpf(1) / 2:
        return x, mpf(0), mpf(0)
    p1, p2, p3 = (y - x) / 2, (1 - 6 * w) / 6, (y - x) * (1 - 12 * w) / 24
    x1 = w * (1 + 2 * p1 * h + 3 * p2 * h ** 2 + 4 * p3 * h ** 3)
    x2 = w * (2 * p1 + 6 * p2 * h + 12 * p3 * h ** 2)
    return (x + w * h * (1 + h * (p1 + h * (p2 + h * p3))), x1 * hl,
            x2 * hl ** 2 + x1 * hll)


def beta_halley_steps(fmt):
    """The least Halley count after which nx's last step meets the quantile
    as `quantile_held` states, the first derivative within rune.mli's budget
    in p, 64 (1 + |log (m / kappa)|) + 16 kappa (64 (1 + |log m|) + 4) ulps
    for kappa = m / (x f(x)) = y / |s|, f the density: from the guess at every point of a grid of shapes from 1e-30 to 2^20 and
    of the smaller tail down to the least subnormal, either tail, and on both
    sides of each of the guess's switches, where neither x nor 1 - x
    underflows. At the quantile the slope s of log T in z gives dx/dl = x y
    / s and d2x/dl2 = x y (y - x - u) / s^2, u = a - (a + b) x - s."""
    least_log = math.log(fmt.tiny)
    tails = [least_log * mpf(2) ** -k for k in (0, 2, 4, 6, 8)] + [mpf("-0.6931"), mpf(-1.5), mpf(-4)]
    shapes = [mpf(v) for v in ("1e-30", "1e-12", "1e-3", "0.05", "0.5", "1", "2.5", "10", "60", "1e3", "1e5")]
    shapes.append(mpf(2) ** 20)
    with mp.workprec(128):
        problems = [(a, b, l, q) for a in shapes for b in shapes for l in tails for q in (False, True)]
        problems += beta_switch_neighbours(tails)
        cases = []
        for a, b, l, q in problems:
            z = oracle.beta_quantile_logit(a, b, mpmath.exp(l), q)
            xs, ys = beta_sigmoid(z), beta_sigmoid(-z)
            if min(xs, ys) < fmt.tiny:
                continue
            _, s = beta_slope(a, b, z, l, q)
            u, w = a - (a + b) * xs - s, xs * ys
            # kappa = m / (x f(x)), and s = x y f(x) / m.
            kappa = ys / abs(s)
            budget = 64 * (1 + abs(l - mpmath.log(kappa))) + 16 * kappa * (64 * (1 + abs(l)) + 4)
            exact = (xs, w / s, w * (ys - xs - u) / s ** 2, w * (1 + abs(u)) / s ** 2)
            normal = min(xs, ys) >= fmt.min_normal
            cases.append((a, b, l, q, exact, budget, normal, beta_guess(fmt, a, b, l, q)))
        for count in range(1, 6):
            def ok(case):
                a, b, l, q, exact, budget, normal, (z, z_lo, z_hi) = case
                for _ in range(count):
                    _, z = beta_halley(a, b, l, q, z, z_lo, z_hi)
                got = beta_last_step(fmt, a, b, l, q, z)
                return quantile_held(fmt, got, exact, budget, normal)
            if all(ok(c) for c in cases):
                return count
    sys.exit("no Halley count meets the beta's inverse's bound")

# The counts of the incomplete beta's series, continued fractions and
# expansions, each the least that meets its tolerance at its region's worst
# points: the value within u/8 of its limit, relative to the quantity it
# enters, and each argument's derivative within u. Counts are proven at 120
# bits against the limit that oracle.py's tail gives, through the factor
# the composition multiplies the quantity by.


def least_count(fmt, points, run, exact, start, stop):
    """The least n in [start, stop] for which run(n, *p) meets the tolerance
    at every point p against exact(*p), its value and its derivative in
    each argument."""
    with mp.workprec(120):
        refs = []
        for p in points:
            p = [mpf(v) for v in p]
            full = exact(*p)
            slopes = [mpmath.diff(lambda v, i=i: exact(*(p[:i] + [v] + p[i + 1:])), p[i]) for i in range(len(p))]
            refs.append((p, full, slopes))

        def ok(n):
            for p, full, slopes in refs:
                if abs(run(n, *p) - full) > fmt.u / 8 * abs(full):
                    return False
                for i, d in enumerate(slopes):
                    dn = mpmath.diff(lambda v: run(n, *(p[:i] + [v] + p[i + 1:])), p[i])
                    if abs(dn - d) * abs(p[i]) > fmt.u * (abs(full) + abs(d * p[i])):
                        return False
            return True

        # The tolerance holds from some count on: bisect for it.
        lo, hi = start, stop
        if not ok(hi):
            sys.exit("no count meets its tolerance")
        while lo < hi:
            mid = (lo + hi) // 2
            if ok(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo


def rlog1_terms(fmt):
    """(e - log1p e) / e^2 = 1/(e + 2) - 2 e P(r^2) / (e + 2)^3 over |e| <=
    RLOG1_SERIES, P truncated to n terms."""
    def run(n, e):
        r = e / (e + 2)
        return 1 / (e + 2) - 2 * e / (e + 2) ** 3 * sum(r ** (2 * k) / (2 * k + 3) for k in range(n))
    def exact(e):
        with mp.workprec(mp.prec + 16):
            return +((e - mpmath.log1p(e)) / (e * e))
    points = [(-RLOG1_SERIES,), (RLOG1_SERIES,), (-0.3,)]
    return least_count(fmt, points, run, exact, 1, 60)


def beta_lead_bits(a, b, x):
    """Guard bits for exp of a sum of terms of the size of a log x, b log (1 -
    x) and log B(a, b)."""
    return oracle.guard(a * mpmath.log(x), b * mpmath.log1p(-x), mpmath.loggamma(a), mpmath.loggamma(b),
                        mpmath.loggamma(a + b))


def log_beta(a, b):
    return mpmath.loggamma(a) + mpmath.loggamma(b) - mpmath.loggamma(a + b)


def bpser_terms(fmt):
    """1 + a sum_n c_n / (a + n) at x up to BETA_BUP_X with b at most 1, and at
    x = BETA_BPSER_BX / b above. Below beta_eps in a, where b x <= 1 and x <=
    1/2, the tail's logarithm is O(a) and holds a s whole: there the sum s =
    sum_n c_n / n itself is held, relative to its own size."""
    def run(n, a, b, x):
        c, s = mpf(1), mpf(0)
        for k in range(1, n + 1):
            c *= (1 - b / k) * x
            s += c / (a + k)
        return 1 + a * s
    x = BETA_BUP_X
    points = [(a, b, x) for a in [1e-3, 1.0, 1e3, 1e6] for b in [1e-6, 0.5, 1.0]]
    points += [(a, b, BETA_BPSER_BX / b) for a in [1.0, 1e3, 1e6] for b in [1.001, 2.0, 39.0]]

    def tiny(n, b, x):
        c, s = mpf(1), mpf(0)
        for k in range(1, n + 1):
            c *= (1 - b / k) * x
            s += c / k
        return s
    def exact(a, b, x):
        """I_x(a, b) a B(a, b) / x^a."""
        with mp.workprec(mp.prec + beta_lead_bits(a, b, x)):
            li, _ = oracle.betainc(a, b, x)
            return +mpmath.exp(li + mpmath.log(a) + log_beta(a, b) - a * mpmath.log(x))

    def tiny_exact(b, x):
        """sum_n (1 - b)_n x^n / (n! n), the integral of ((1 - t)^(b - 1) -
        1) / t from 0 to x."""
        with mp.workprec(mp.prec + 16):
            return +mpmath.quad(lambda t: mpmath.expm1((b - 1) * mpmath.log1p(-t)) / t, [0, x])
    edge = [(b, min(0.5, 1 / b)) for b in [1e-9, 0.5, 1.5, 2.5, 10.0, 1e3, 1e6]]
    return max(least_count(fmt, points, run, exact, 5, 400), least_count(fmt, edge, tiny, tiny_exact, 2, 400))


def bfrac_cf(n, a, b, x):
    """BFRAC's 1 / (beta_0 + alpha_1 / (beta_1 + ...)), n deep, lam = a - (a + b)
    x."""
    lam = a - (a + b) * x
    y = 1 - x
    c, c0, c1, yp1 = 1 + lam, b / a, 1 + 1 / a, y + 1
    w = lambda k: k * (b - k) * x

    def alpha(k):
        e = a / (a + 2 * k - 1)
        p = 1 + (k - 1) / a
        return p * (p + c0) * e * e * w(k) * x

    def beta(k):
        if k == 0:
            return c / c1
        t = k / a
        return k + w(k) / (a + 2 * k - 1) + (1 + t) / (c1 + 2 * t) * (c + k * yp1)

    f = beta(n)
    for k in range(n, 0, -1):
        f = beta(k - 1) + alpha(k) / f
    return 1 / f


def bfrac_depth(fmt):
    """BFRAC on its region: b0 at least BETA_B_SMALL, a0 to BETA_FRAC_A near
    the mean, and beyond it where lam passes BETA_FRAC_LAMBDA of the smaller."""
    points = []
    for low in [BETA_B_SMALL, BETA_FRAC_A, 101.0, 1e3, 1e4, 2.0 ** 20]:
        for ratio in [1.0, 10.0, 1e3]:
            for a, b in [(low, low * ratio), (low * ratio, low)]:
                if max(a, b) > 2.0 ** 20 or b < BETA_B_SMALL:
                    continue
                lams = [0.0] if min(a, b) <= BETA_FRAC_A else []
                lams.append(BETA_FRAC_LAMBDA * min(a, b) * 1.001)
                for lam in lams:
                    points.append((a, b, (a - lam) / (a + b)))

    def exact(a, b, x):
        """I_x(a, b) B(a, b) / (x^a y^b), the factor BRCOMP's."""
        y = oracle.one_minus(x)
        with mp.workprec(mp.prec + beta_lead_bits(a, b, x)):
            li, _ = oracle.beta_tails(a, b, x, y)
            return +mpmath.exp(li + log_beta(a, b) - a * mpmath.log(x) - b * mpmath.log(y))
    return least_count(fmt, points, bfrac_cf, exact, 5, 400)


def basym_sum(n, a, b, lam):
    """BASYM's sum J0 + sum_i d_i w0^i J_i through d_(n+1), in exact arithmetic."""
    d = basym_coefficients(30)
    d = [[mpf(c.numerator) / c.denominator for c in poly] for poly in d]
    p, q = min(a, b), max(a, b)
    h = p / q
    sign = 1 if b >= a else -1
    w0 = sign / mpmath.sqrt(p * (1 + h))
    rl = lambda e: e - mpmath.log1p(e)
    f = a * rl(-lam / a) + b * rl(lam / b)
    z0 = mpmath.sqrt(f)
    e0, e1 = 2 / mpmath.sqrt(mpmath.pi), mpf(1) / (2 * mpmath.sqrt(2))
    poly = lambda i: mpmath.polyval(d[i - 1][::-1], h) / (1 + h) ** (i // 2)
    j0 = 0.5 / e0 * mpmath.exp(f) * mpmath.erfc(z0)
    j1 = e1
    s = j0 + poly(1) * w0 * j1
    znm1, zn, z2 = z0 * mpmath.sqrt(2), 2 * f, 2 * f
    w = w0
    for k in range(2, n + 1, 2):
        j0 = e1 * znm1 + (k - 1) * j0
        j1 = e1 * zn + k * j1
        znm1, zn = z2 * znm1, z2 * zn
        w *= w0
        t0 = poly(k) * w * j0
        w *= w0
        t1 = poly(k + 1) * w * j1
        s += t0 + t1
    return s


def basym_terms(fmt):
    """BASYM where both arguments pass BETA_FRAC_A and lam is within
    BETA_FRAC_LAMBDA of the smaller; an even count."""
    points = []
    for low in [BETA_FRAC_A * 1.001, 300.0, 1e4]:
        for ratio in [1.0, 3.0, 100.0]:
            for a, b in [(low, low * ratio), (low * ratio, low)]:
                for frac in [0.0, BETA_FRAC_LAMBDA / 2, BETA_FRAC_LAMBDA]:
                    points.append((a, b, frac * min(a, b)))

    def exact(a, b, lam):
        """I_x(a, b) e^(f + bcorr(a, b)) / e0 at x = (a - lam) / (a + b), with
        bcorr Stirling's corrections, del(a) + del(b) - del(a + b)."""
        x, y = (a - lam) / (a + b), (b + lam) / (a + b)
        stirling = lambda v: mpmath.loggamma(v) - (v - 0.5) * mpmath.log(v) + v - mpmath.log(2 * mpmath.pi) / 2
        with mp.workprec(mp.prec + beta_lead_bits(a, b, x)):
            li, _ = oracle.beta_tails(a, b, x, y)
            rl = lambda e: e - mpmath.log1p(e)
            f = a * rl(-lam / a) + b * rl(lam / b)
            bcorr = stirling(a) + stirling(b) - stirling(a + b)
            return +mpmath.exp(li + f + bcorr - mpmath.log(2 / mpmath.sqrt(mpmath.pi)))
    n = least_count(fmt, points, basym_sum, exact, 2, 29)
    return n + n % 2


def bgrat_sum(n, a, b, x):
    """BGRAT's J0 + sum_k d_k(b) J_k through n terms, z = -nu log x."""
    d = bgrat_coefficients(64)
    nu = a + (b - 1) / 2
    lx = mpmath.log(x)
    z = -nu * lx
    j = mpmath.gammainc(b, z, mpmath.inf, regularized=True) / (mpmath.exp(-z) * z ** b / mpmath.gamma(b))
    s, t = j, mpf(1)
    v, t2 = 0.25 / nu ** 2, 0.25 * lx * lx
    for i in range(n):
        bp2n = b + 2 * i
        j = (bp2n * (bp2n + 1) * j + (z + bp2n + 1) * t) * v
        t *= t2
        s += mpmath.polyval([mpf(c.numerator) / c.denominator for c in d[i][::-1]], b) * j
    return s


def bgrat_terms(fmt):
    """BGRAT for a from BETA_BGRAT_B, b at most 1, x from BETA_BUP_X."""
    points = [(BETA_BGRAT_B, b, BETA_BUP_X) for b in [1e-6, 0.5, 1.0]]
    points += [(BETA_BGRAT_B + BETA_BUP_TERMS, b, 1 - BETA_X_FAR) for b in [1e-6, 0.5, 1.0]]

    def exact(a, b, x):
        """I_x(a, b) / u, u = e^-z z^b Gamma(a + b) / (Gamma(a) Gamma(b) nu^b)."""
        nu = a + (b - 1) / 2
        with mp.workprec(mp.prec + beta_lead_bits(a, b, x)):
            li, _ = oracle.betainc(a, b, x)
            z = -nu * mpmath.log(x)
            lu = -z + b * mpmath.log(z) - log_beta(a, b) - b * mpmath.log(nu)
            return +mpmath.exp(li - lu)
    return least_count(fmt, points, bgrat_sum, exact, 1, 40)


def poly_add(p, q):
    n = max(len(p), len(q))
    return [(p[i] if i < len(p) else 0) + (q[i] if i < len(q) else 0) for i in range(n)]


def poly_mul(p, q):
    r = [Fraction(0)] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            r[i + j] += a * b
    return r


@functools.lru_cache(maxsize=None)
def basym_coefficients(count):
    """BASYM's d_i(h) for i <= count + 1, with s = sign (b - a): d_i = s^i
    G_i(h) / (1 + h)^(i div 2), G_i a polynomial, lowest degree first. Its
    recurrences in h's rational functions, as DiDonato and Morris's code runs
    them per element."""
    one_plus, one_minus = [Fraction(1), Fraction(1)], [Fraction(1), Fraction(-1)]

    def radd(x, y):
        (p, k), (q, l) = x, y
        for _ in range(max(k, l) - k):
            p = poly_mul(p, one_plus)
        for _ in range(max(k, l) - l):
            q = poly_mul(q, one_plus)
        return poly_add(p, q), max(k, l)

    rmul = lambda x, y: (poly_mul(x[0], y[0]), x[1] + y[1])
    rscale = lambda x, c: ([c * v for v in x[0]], x[1])
    zero = ([Fraction(0)], 0)
    a0, d, c = {}, {}, {}
    a0[1] = ([Fraction(2, 3) * v for v in one_minus], 0)
    c[1] = rscale(a0[1], Fraction(-1, 2))
    d[1] = rscale(c[1], -1)
    s = [Fraction(1)]
    for n in range(2, count + 1, 2):
        a0[n] = ([Fraction(2, n + 2) * v for v in poly_add([Fraction(1)], [Fraction(0)] * (n + 1) + [Fraction(1)])], 1)
        s = poly_add(s, [Fraction(0)] * n + [Fraction(1)])
        a0[n + 1] = ([Fraction(2, n + 3) * v for v in poly_mul(one_minus, s)], 0)
        for i in (n, n + 1):
            r = Fraction(-(i + 1), 2)
            b0 = {1: rscale(a0[1], r)}
            for m in range(2, i + 1):
                bsum = zero
                for j in range(1, m):
                    bsum = radd(bsum, rscale(rmul(a0[j], b0[m - j]), j * r - (m - j)))
                b0[m] = radd(rscale(a0[m], r), rscale(bsum, Fraction(1, m)))
            c[i] = rscale(b0[i], Fraction(1, i + 1))
            dsum = zero
            for j in range(1, i):
                dsum = radd(dsum, rmul(d[i - j], c[j]))
            d[i] = rscale(radd(dsum, c[i]), -1)
    out = []
    for i in range(1, count + 2):
        poly, k = d[i]
        assert k == i // 2, (i, k)
        while len(poly) > 1 and poly[-1] == 0:
            poly = poly[:-1]
        out.append(poly)
    return out


@functools.lru_cache(maxsize=None)
def bgrat_coefficients(count):
    """BGRAT's d_n(b), n = 1..count, lowest degree first: c_n = prod_k 1 /
    (2k (2k + 1)), d_n = (b - 1) c_n + (1/n) sum_i (i b - n) c_i d_(n-i)."""
    cs, ds = {}, {}
    cn = Fraction(1)
    for n in range(1, count + 1):
        cn /= (2 * n) * (2 * n + 1)
        cs[n] = cn
        acc = [-cn, cn]
        for i in range(1, n):
            acc = poly_add(acc, [Fraction(1, n) * v for v in poly_mul([Fraction(-n) * cs[i], Fraction(i) * cs[i]], ds[n - i])])
        ds[n] = acc
    return [ds[n] for n in range(1, count + 1)]


def beta_tables(fmt):
    counts = dict(rlog1=rlog1_terms(fmt), bpser=bpser_terms(fmt),
                  bfrac=bfrac_depth(fmt), basym=basym_terms(fmt), bgrat=bgrat_terms(fmt))
    high = lambda poly: [round_to(fmt, mpf(v.numerator) / v.denominator) for v in reversed(poly)]
    return [
        ("beta_eps", ocaml_float(BETA_EPS[fmt])),
        ("rlog1", ocaml_array(high([Fraction(1, 2 * k + 3) for k in range(counts["rlog1"])]))),
        ("bpser_terms", str(counts["bpser"])),
        ("bfrac_depth", str(counts["bfrac"])),
        ("basym", ocaml_matrix([high(p) for p in basym_coefficients(counts["basym"])])),
        ("bgrat", ocaml_matrix([high(p) for p in bgrat_coefficients(counts["bgrat"])])),
    ]

# Tables
#
# Each table is an OCaml binding, with a comment that names its source in
# the implementation and its doc in the interface. Coefficients are written
# as hexadecimal floats, exact.


def ocaml_float(x):
    if x == 0:
        return "0."
    mantissa, exponent = x.hex().split("p")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}p{exponent}"


def ocaml_array(xs):
    return "[|\n" + "".join(f"    {ocaml_float(x)};\n" for x in xs) + "  |]"


def ocaml_matrix(rows):
    return "[|\n" + "".join(
        "    [|\n" + "".join(f"      {ocaml_float(x)};\n" for x in row) + "    |];\n" for row in rows) + "  |]"


def ocaml_kind(value):
    if value.startswith("[|\n    [|"):
        return "float array array"
    if value.startswith("[|"):
        return "float array"
    return "int" if value.isdigit() else "float"


def wrap(words, first, rest, close, width=80):
    """`words` filled into lines of `width` columns, the first opening with
    `first`, the others with `rest`, the last ending in `close`, on a line of
    its own where it does not fit, as ocamlformat fills a comment."""
    lines, line = [], []
    prefix = first
    for w in words:
        if line and len(prefix) + len(" ".join(line + [w])) > width:
            lines.append(prefix + " ".join(line))
            line, prefix = [], rest
        line.append(w)
    last = prefix + " ".join(line)
    if close and len(last) + len(close) > width:
        lines.append(last)
        last, close = rest, close.lstrip()
    lines.append(last + close)
    return lines


def comment(text):
    """`text` as an OCaml comment wrapped at 80 columns."""
    return "\n".join(wrap(text.split(), "(* ", "   ", " *)"))


def doc(text, indent=0):
    """`text` as an odoc comment at `indent`, its paragraphs split on blank
    lines, wrapped at 80 columns."""
    pad, paras = " " * indent, text.split("\n\n")
    out = []
    for i, para in enumerate(paras):
        first = pad + ("(** " if i == 0 else "    ")
        close = " *)" if i == len(paras) - 1 else ""
        out += wrap(para.split(), first, pad + "    ", close)
        if close == "":
            out.append("")
    return "\n".join(out)


LICENCE = """\
(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)
"""

TABLES_HEADER = """\
(* Generated by packages/nx/test/gen/special_tables.py from mpmath and the
   sources its comments name; do not edit. Regenerate it from the repository
   root with [uv run packages/nx/test/gen/special_tables.py]. *)
"""

INTERFACE_HEADER = """\
(* Generated by packages/nx/test/gen/special_tables.py with special_tables.ml;
   do not edit. Regenerate it from the repository root with
   [uv run packages/nx/test/gen/special_tables.py]. *)
"""

# The implementation's comments: where each group of shared tables comes from.

SOURCES = {
    "erx": "fdlibm's s_erf.c (1.3 95/01/18), its coefficients checked against the words "
           "of their doubles: " + FDLIBM_NOTICE,
    "ndtri_central_below": "Wichura, Algorithm AS 241, Applied Statistics 37 (1988).",
    "lgamma_tc": "fdlibm's e_lgamma_r.c (1.3 95/01/18), its coefficients checked against "
                 "the words of their doubles, under the same notice.",
    "igamma_corner": "DiDonato and Morris, Computation of the incomplete gamma function "
                     "ratios and their inverse, ACM TOMS 12 (1986).",
    "igamma_guess_p": "DiDonato and Morris's GAMINV, Algorithm 654, ACM TOMS 13 (1987).",
    "beta_bpser_bx": "DiDonato and Morris, Algorithm 708, ACM TOMS 18 (1992).",
    "beta_guess_terms": "Temme, Asymptotic inversion of the incomplete beta function, "
                        "J. Comput. Appl. Math. 41 (1992), sections 3 and 4.",
    "beta_temme": "Temme's corrections e1 to e3 (p. 151) as Boost's ibeta_inverse.hpp "
                  "transcribes them; against the exact eta, e1 holds to Temme's order "
                  "and e2 and e3 lower the error by less than theirs would: "
                  + BOOST_NOTICE,
}

PER_DTYPE_SOURCES = (
    "The tables whose values differ between dtypes. erfinv's are Giles' polynomials "
    "(Approximating the erfinv function, GPU Computing Gems Jade, 2011; erfinv_SP_1.cu and "
    "erfinv_DP_1.cu), ndtri's AS 241's PPND7 and PPND16, whose printed hash sums the "
    "generator checks, and [beta_eps] TOMS 708's tolerance. The generator computes the "
    "rest with mpmath.")

# The interface. Each section opens before the shared table named.

PREAMBLE = """\
The coefficients, region bounds and term counts of the special functions \
of [Special].

A {e shared} table is one value for both compute dtypes: a double that a \
float32 function rounds to float32 where it uses it. The tables whose values \
differ between dtypes are the fields of {!t}, one record per compute dtype. \
A polynomial's coefficients come highest degree first, for Horner's rule, \
unless its doc says otherwise, and the denominator Q of a rational P/Q ends \
in its constant term 1. A region bound [_below] ends a region and [_from] \
starts one.

{e u} is the unit roundoff of a record's dtype, 2^-24 at float32 and 2^-53 \
at float64. A count or degree {e within} a bound of a value is the least at \
which the approximation, its coefficients as the table holds them, is within \
that bound of the value, relatively, against mpmath at points of its region. \
A bound of the derivative bounds the derivative's relative error the same \
way. A doc that measures otherwise says how."""

SECTIONS = {
    "erx": ("erf", "The error function",
            "The rationals of erf and erfc by region of a = |x|. Below [erf_small_below], "
            "erf x = x + x P/Q, P/Q being [erf_small_p]/[erf_small_q] in x^2, and erfc x is "
            "1 - erf x for x below [erfc_small_split] and 1/2 - (x P/Q + x - 1/2) from it. "
            "Below "
            "[erf_near_below], erf a = [erx] + P/Q, P/Q being [erf_near_p]/[erf_near_q] in "
            "a - 1, where [erx] is erf 1 rounded to 24 bits. Up to [erfc_far_from], erfc a = "
            "exp (-a^2 - 0.5625 + R/S) / a, R/S being [erfc_mid_p]/[erfc_mid_q] in 1/a^2 "
            "below [erfc_mid_below] and [erfc_far_p]/[erfc_far_q] from it. From "
            "[erfc_far_from], erfc x is 0, or 2 where x is negative."),
    "ndtri_central_below": ("ndtri", "The normal quantile",
                            "The regions of ndtri's first guess at p, as AS 241 splits them. "
                            "The central one, |p - 1/2| <= [ndtri_central_below], gives "
                            "(p - 1/2) P(r)/Q(r) at r = [ndtri_central_r] - (p - 1/2)^2, "
                            "P/Q the record's [ndtri_central_p]/[ndtri_central_q]. Each "
                            "tail takes s = sqrt (-log q), q the smaller of p and 1 - p, and "
                            "gives -P(s')/Q(s') on the lower tail: s' = s - "
                            "[ndtri_near_shift] with the record's [ndtri_near] pair up to "
                            "s = [ndtri_near_below], and s' = s - [ndtri_far_shift] with its "
                            "[ndtri_far] pair above."),
    "lgamma_tc": ("lgamma", "The log-gamma function",
                  "lgamma of x > 0 by region. On (0, 2) three polynomials run in t, each "
                  "relative to the zero it starts from: lgamma (2 - t) = t A(t) - t/2 about 1 "
                  "and 2, A [lgamma_a]; lgamma ([lgamma_tc] + t) = [lgamma_tf] + t^2 T(t) - "
                  "[lgamma_tt] about lgamma's minimum [lgamma_tc], T [lgamma_t], where "
                  "[lgamma_tf] is lgamma at the minimum rounded to a double and -[lgamma_tt] "
                  "its rounding error; lgamma (1 + t) = t U(t)/V(t) - t/2 beside 1, U/V "
                  "[lgamma_u]/[lgamma_v]. Below [lgamma_small] they give lgamma (1 + x), "
                  "less log x, the one about 2 from x = [lgamma_c], the one about the "
                  "minimum from [lgamma_b] and the one beside 1 below it. From "
                  "[lgamma_small] to 2, the one about 2 runs from [lgamma_f], the one "
                  "about the minimum from [lgamma_e] and the one beside 1 below it.\n\n"
                  "From [lgamma_shift_from], x is shifted down to 2 + f, f in \\[0, 1), where "
                  "lgamma (2 + f) = f/2 + f S(f)/R(f), S/R [lgamma_s]/[lgamma_r]. From "
                  "[lgamma_stirling_from], lgamma x = (x - 1/2)(log x - 1) + [lgamma_w0] + z "
                  "W(z^2), z = 1/x, W [lgamma_w]."),
    "bessel_split": ("bessel", "Scaled Bessel functions",
                     "The regions of i0e and i1e: Chebyshev series in x (2 / [bessel_split]) "
                     "- 1 up to [bessel_split], weighted by h = 1 + [bessel_weight] x, and in "
                     "2 [bessel_split] / x - 1 above. The record's [i0e_near] to [i1e_far] "
                     "hold the series."),
    "igamma_corner": ("igamma", "The incomplete gamma function",
                      "The regions of the incomplete gamma functions P(a, x) and Q(a, x): "
                      "the corner below a = 1 and x = [igamma_corner]; Temme's expansion "
                      "from a = [igamma_temme_from] where |x/a - 1| <= "
                      "[igamma_temme_width]; and the series of bd0, the prefactor's "
                      "exponent, to |v| = [igamma_near_v], v = (x - a)/(x + a). erfcx w = "
                      "e^(w^2) erfc w, which Temme's expansion and BASYM take, leaves erfc's "
                      "rationals for its asymptotic series (the record's [erfcx_series]) "
                      "from [erfc_far_from].\n\n"
                      "The inverses' first guess is a normal quantile t - P(t)/Q(t) at t = "
                      "sqrt (-2 log q), q the smaller tail, P/Q "
                      "[igamma_guess_p]/[igamma_guess_q]. For a small x it sums the series "
                      "1 + sum_k x^k / ((a + 1) ... (a + k)) to [igamma_guess_terms] terms, "
                      "each ratio x / (a + k) at most 0.7, so that the last term is below "
                      "10^-4."),
    "beta_bpser_bx": ("beta", "The incomplete beta function",
                      "The regions of the incomplete beta function I_x(a, b), as TOMS 708 "
                      "splits them for its arguments a0, b0, x0 after the swap, by its "
                      "methods: BPSER, the power series; BUP, the finite sum that shifts "
                      "a0; BGRAT, the expansion for a large b0; BFRAC, the continued "
                      "fraction; and BASYM, the expansion for a large a0 and b0.\n\n"
                      "Both above 1: BPSER while b0 x0 <= [beta_bpser_bx] and b0 < "
                      "[beta_b_small]; below that b0, BUP then BPSER while x0 <= "
                      "[beta_bup_x], else BGRAT, after BUP of [beta_bup_terms] terms where "
                      "a0 <= [beta_bgrat_b]; above it BFRAC up to [beta_frac_a] or where "
                      "lambda > [beta_frac_lambda] of the smaller, else BASYM. Either at "
                      "most 1: BPSER where a0 >= min ([beta_a_small], b0), x0^a0 <= "
                      "[beta_power_x] or (x0 b0)^a0 <= [beta_power_bx] below x0 = "
                      "[beta_x_near]; BPSER on the complement from x0 = [beta_x_far]; BGRAT "
                      "alone for b0 > [beta_bgrat_b]. A BUP whose count is masked sums at "
                      "most [beta_bup_most] terms. x^a y^b / B(a, b) takes its "
                      "large-argument form from [beta_large], and e - log1p e its series "
                      "where |e| <= [rlog1_series]."),
    "beta_guess_terms": ("beta_inverse", "The incomplete beta function's inverse",
                         "The first guess of the beta's quantile, in z = log (x / (1 - x)). "
                         "A tail's power form solves BPSER's x^a / (a B(a, b)) (1 + a S(x)) "
                         "= P for z, S(x) = sum_k (1 - b)_k x^k / (k! (a + k)) to "
                         "[beta_guess_terms] terms, by as many fixed-point steps, and is "
                         "taken where the first term it leaves is below [beta_guess_gap], "
                         "x at most 1/2 and b x at most [beta_guess_bx]; "
                         "the other tail's likewise. Else the gamma's form, P(a, -nu log (1 - x)) with nu = "
                         "b + (a - 1)/2, where the larger shape exceeds the smaller "
                         "[beta_guess_ratio] times, is at least [beta_guess_gamma_from] and "
                         "the smaller is below [beta_guess_temme_from]; else, where the "
                         "smaller is below 1 and the larger below [beta_guess_gamma_from], "
                         "the power form whose terms left are fewer; else Temme's "
                         "asymptotic form in 1/(a "
                         "+ b) to its third term, its equation for x solved by "
                         "[beta_guess_newton] Newton steps. Its corrections e1, e2 and e3 "
                         "are sums over j of c_kj(s^2) eta0^j / (d_kj (s c)^n_kj): "
                         "[beta_temme] holds the c_kj one after the other, highest "
                         "degree first, [beta_temme_scale] the d_kj and "
                         "[beta_temme_power] the n_kj, [beta_temme_terms] the count of "
                         "each e_k's terms."),
}

# The record's field docs, each on the first field of its group.

FIELD_DOCS = {
    "erfinv_central": "erfinv's first guess at p, p E(s) with w = -log ((1 - p)(1 + p)): E is "
                      "[erfinv_central] at s = w - [erfinv_central_shift] below "
                      "[erfinv_central_below], [erfinv_near] at s = sqrt w - "
                      "[erfinv_near_shift], and [erfinv_far] at s = sqrt w - "
                      "[erfinv_far_shift] from [erfinv_far_from], which is infinite at "
                      "float32. The three are padded with leading zeros to one length. Its "
                      "Newton step runs on erf below |p| = [erfinv_newton_split], erf 1.25, "
                      "and on erfc above.",
    "sqrt1_2_hi": "1/sqrt 2 as two floats, whose sum is it to twice the dtype's precision.",
    "ndtri_central_p": "ndtri's rationals ({!section-ndtri}): PPND7's at float32 and PPND16's "
                       "at float64.",
    "log_ndtr_below": "log_ndtr's asymptotic series below [log_ndtr_below], log (1 + sum_k "
                      "(-1)^k (2k - 1)!! / x^(2k)): [log_ndtr_series] holds (-1)^k (2k - 1)!!, "
                      "its least count within u/8 of the value and of the derivative.",
    "digamma_root_hi": "digamma's root near 1.4616 as two floats, and on [1, 2] the "
                       "polynomial g in x - 3/2 with digamma x = (x - root) g: the least "
                       "degree within u of g and 16u of the derivative. From "
                       "[digamma_from], [digamma_series] holds B_2k / 2k of the asymptotic "
                       "series in 1/x^2, its least count within u/8 of the value and of the "
                       "derivative with each B_2k / 2k exact.",
    "sinpi": "sin (pi x) on [0, 1/2] as x S(x^2): the least degree of S within u of the "
             "value and, absolutely, 16u of the derivative's largest magnitude, pi.",
    "i0e_near": "The scaled Bessel functions' Chebyshev series ({!section-bessel}), lowest "
                "degree first: [i0e_near] is q, with i0e x h = 1 + x q, and [i1e_near] is "
                "h^2 i1e x / x, on [0, bessel_split]; [i0e_far] and [i1e_far] are sqrt x "
                "i0e x and sqrt x i1e x above. Each has the least degree within u of the "
                "value (4u of 1 + x q for [i0e_near]) and 16u of the derivative's largest "
                "magnitude.",
    "igamma_corner_terms": "The incomplete gamma's counts of the corner's series, P's series "
                           "and Q's continued fraction, each the least whose sum is within "
                           "u/8 of the same sum run to 2n + 40 terms, relatively, and whose "
                           "derivatives are within 2u in x and 2^-44 (2^-20 at float32) in a "
                           "of that sum's, relative to the larger of the derivative and the "
                           "value. [igamma_atanh] is S(w) = sum_j w^(j-1)/(2j + 1), highest "
                           "degree first, to the least count within u/8 of bd0's series, mu "
                           "- log1p mu, to |v| = [igamma_near_v] and of its derivative. The "
                           "shift of Stirling's correction from \\[1, 8) runs the last "
                           "[igamma_shift_terms.(k)] of S in its k-th term, each the least "
                           "within u/64, absolutely, of the term and of its derivative. "
                           "[igamma_temme] holds Temme's c_k(eta), k < K, each to the degree "
                           "in [igamma_temme_degrees], highest first, one after the other: "
                           "each degree the least whose truncation of the scaled tail, and "
                           "of its derivative, is at most u/32 at a = 20 over Temme's "
                           "window, and K the least within u/8 of the tail with its "
                           "derivatives within the counts' bounds. [igamma_halley_steps] is "
                           "the inverses' Halley steps after the first guess, the least after "
                           "which the last step, of fifth order, puts x within u/8 of the "
                           "quantile, relatively, its first derivative in the tail's "
                           "logarithm within an eighth of rune.mli's budget and its second "
                           "within sqrt(eps)/8 of its terms, for a up to 2^20 and either "
                           "tail down to the least subnormal. [beta_halley_steps] is the "
                           "beta's quantile's Halley steps after its first guess, the least "
                           "after which its last step does as much, for shapes from 1e-30 "
                           "to 2^20 and either tail down to the least subnormal. "
                           "[erfcx_series] is erfc x e^(x^2) x sqrt pi "
                           "from [erfc_far_from], (-1)^k (2k - 1)!! in 1/(2x^2), its least "
                           "count within u/8 of the value and of the derivative.",
    "beta_eps": "TOMS 708's tolerance, the dtype's epsilon but 1e-15 at float64, below which "
                "an argument is negligible; the series of e - log1p e in r^2, r = e / (e + 2); "
                "the incomplete beta's term counts and depths, each the least at which its "
                "run is within u/8 of the same run far past convergence, relative to the "
                "value f the run computes, and its derivative d in each argument p within "
                "u (|f| + |d p|) / |p| of that run's, BASYM's count rounded up to even; BASYM's coefficients G_i(h), d_i = sign (b - a)^i G_i(h) / "
                "(1 + h)^(i div 2), and BGRAT's d_n(b).",
}


def tables(shared, rows):
    out = [LICENCE, TABLES_HEADER]
    for name, value in shared:
        text = comment(SOURCES[name]) + "\n" if name in SOURCES else ""
        out.append(f"{text}let {name} = {value}\n" if not value.startswith("[|")
                   else f"{text}let {name} =\n  {value}\n")
    decl = [comment(PER_DTYPE_SOURCES), "type t = {"]
    decl += [f"  {name} : {ocaml_kind(value)};" for name, value in rows[F64]]
    decl.append("}\n")
    out.append("\n".join(decl))
    for fmt, label in [(F32, "float32"), (F64, "float64")]:
        body = [f"let {label} =", "  {"]
        for name, value in rows[fmt]:
            value = value.replace("\n", "\n    ")
            body.append(f"    {name} = {value};")
        body.append("  }\n")
        out.append("\n".join(body))
    return "\n".join(out)


def interface(shared, rows):
    out = [LICENCE, INTERFACE_HEADER, doc(PREAMBLE) + "\n"]
    for name, value in shared:
        if name in SECTIONS:
            label, title, text = SECTIONS[name]
            out.append(doc(f"{{1:{label} {title}}}\n\n{text}") + "\n")
        out.append(f"val {name} : {ocaml_kind(value)}\n")
    out.append(doc("{1:dtypes Tables per dtype}") + "\n")
    decl = ["type t = {"]
    for name, value in rows[F64]:
        decl.append(f"  {name} : {ocaml_kind(value)};")
        if name in FIELD_DOCS:
            decl.append(doc(FIELD_DOCS[name], indent=6))
    decl.append("}")
    decl.append(doc("The type for the tables whose values differ between dtypes."))
    out.append("\n".join(decl) + "\n")
    out.append("val float32 : t\nval float64 : t\n")
    return "\n".join(out)


# Driver


def outputs():
    oracle.check()
    shared = shared_tables()
    rows = {F32: per_dtype(F32), F64: per_dtype(F64)}
    return {TABLES: tables(shared, rows), INTERFACE: interface(shared, rows)}


def read(path):
    return path.read_text() if path.exists() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="write nothing; fail if a file would change")
    args = parser.parse_args()
    files = outputs()
    changed = {path: body for path, body in files.items() if read(path) != body}
    if args.check:
        for path, body in changed.items():
            rel = path.relative_to(ROOT)
            sys.stdout.writelines(difflib.unified_diff(
                (read(path) or "").splitlines(keepends=True), body.splitlines(keepends=True),
                fromfile=f"committed/{rel}", tofile=f"generated/{rel}"))
        if changed:
            sys.exit(1)
        print(f"{len(files)} files match")
        return
    for path, body in changed.items():
        path.write_text(body)
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
