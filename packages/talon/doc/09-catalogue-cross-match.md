# Catalogue Cut and Cross-Match

A sky survey catalogue holds one row per detected source. To find the objects
a second survey also saw, this page cuts the first catalogue to well-measured
sources, matches each to the second catalogue within a small radius on the
sphere, keeps the nearest candidate and reports what stays unmatched. It uses
the terms of [Expressions and Frames](02-expressions-and-frames.md) and
[Joins](03-joins.md).

The fixtures hold six sources of an optical catalogue, `gaia`, and six of an
infrared one, `wise`. Positions `ra` and `dec` are in degrees. `parallax` and
its error are in milliarcseconds, `g`, `bp` and `rp` are optical magnitudes,
and `w1` is an infrared magnitude. The right side's columns have their own
names, since a name on both sides of a join is a problem:

```ocaml
open Talon

let show q = Format.printf "%a@." Talon.pp (Error.get_ok (Query.run q))
let floats a = Column.v Type.float64 a
let strings a = Column.v Type.string a

let gaia =
  Talon.v
    [
      ("source_id", strings [| "s1"; "s2"; "s3"; "s4"; "s5"; "s6" |]);
      ("ra", floats [| 100.0; 100.1; 100.2; 100.3; 100.4; 359.99995 |]);
      ("dec", floats [| -20.0; -20.1; -20.2; -20.3; -20.4; 5.0 |]);
      ("parallax", floats [| 10.; 5.; 2.; 8.; 20.; 4. |]);
      ("parallax_error", floats [| 0.2; 0.1; 0.5; 0.1; 0.1; 0.1 |]);
      ("ruwe", floats [| 1.0; 1.1; 1.2; 2.0; 0.9; 1.0 |]);
      ("g", floats [| 12.0; 14.5; 15.0; 13.0; 11.0; 16.0 |]);
      ("bp", floats [| 12.4; 15.0; 15.6; 13.4; 11.3; 16.3 |]);
      ("rp", floats [| 11.5; 13.9; 14.5; 12.5; 10.6; 15.5 |]);
    ]

let wise =
  Talon.v
    [
      ("wise_id", strings [| "w1"; "w2"; "w3"; "w4"; "w5"; "w6" |]);
      ( "ra_b",
        floats [| 100.0001; 100.0002; 100.1; 100.40035; 0.00005; 200. |] );
      ("dec_b", floats [| -20.0; -20.0001; -20.1002; -20.4; 5.0; 30. |]);
      ("w1", floats [| 11.2; 11.9; 13.1; 9.8; 15.1; 8. |]);
    ]
```

## Cuts and Derived Columns

Cuts are filters on measured columns. A parallax much larger than its error
gives a reliable distance, and a small `ruwe` means the astrometric fit was
good. The absolute magnitude is `g + 5 log10(parallax) - 10` for a parallax in
milliarcseconds, and a colour is a difference of two magnitudes. `Expr.nx`
applies an elementwise nx function to a column, and `log10` follows from
`Nx.log`:

```ocaml
let plx = Col.float "parallax"
let g = Col.float "g"
let ln_10 = Float.log 10.
let log10 x = Expr.(nx { f = Nx.log } x /. float ln_10)

let selected =
  Query.(
    of_table gaia
    |> filter
         Expr.(
           plx /. Col.float "parallax_error" > float 10.
           && Col.float "ruwe" < float 1.4)
    |> derive
         Expr.[
           "abs_g" := g +. (float 5. *. log10 plx) -. float 10.;
           "bp_rp" := Col.float "bp" -. Col.float "rp";
         ])
```

## The Sky Join

Two sources match when their angular separation is within a radius. A join's
conditions compare columns, so a range join narrows the candidates and a
filter then tests the separation. `dec_lo` and `dec_hi` bound a band of one
radius on each side of a source, and the join reads only the pairs in the
band:

```ocaml
let radius_arcsec = 1.
let radius_deg = radius_arcsec /. 3600.

let bands =
  Query.derive
    Expr.[
      "dec_lo" := Col.float "dec" -. float radius_deg;
      "dec_hi" := Col.float "dec" +. float radius_deg;
    ]
    selected

let candidates =
  Query.(
    bands
    |> join ~on:Join.(le "dec_lo" "dec_b" && ge "dec_hi" "dec_b") (of_table wise))
```

The separation uses the haversine formula in radians, which stays accurate at
small angles:

```ocaml
let rad_per_deg = Float.pi /. 180.
let rad_per_arcsec = rad_per_deg /. 3600.
let rad x = Expr.(x *. float rad_per_deg)
let sin_ x = Expr.nx { f = Nx.sin } x
let cos_ x = Expr.nx { f = Nx.cos } x

let separation ra1 dec1 ra2 dec2 =
  Expr.(
    let half_ra = sin_ (rad (ra2 -. ra1) /. float 2.) in
    let half_dec = sin_ (rad (dec2 -. dec1) /. float 2.) in
    let a =
      (half_dec *. half_dec)
      +. (cos_ (rad dec1) *. cos_ (rad dec2) *. half_ra *. half_ra)
    in
    nx { f = Nx.asin } (nx { f = Nx.sqrt } a) *. float 2.)

let sep_arcsec =
  Expr.(
    separation (Col.float "ra") (Col.float "dec") (Col.float "ra_b")
      (Col.float "dec_b")
    /. float rad_per_arcsec)
```

Right ascension wraps at 360 degrees, and the band is on declination, which
does not wrap. `half_ra` squared repeats every 360 degrees, so a source at
`359.99995` lies `0.0001` degrees of right ascension from a candidate at
`0.00005`. The filter keeps the pairs within the radius:

```ocaml
let in_radius =
  Query.(
    candidates
    |> derive Expr.[ "sep_arcsec" := sep_arcsec ]
    |> filter Expr.(Col.float "sep_arcsec" <= float radius_arcsec))
```

## The Nearest Candidate

A source may have several candidates inside the radius, and `s1` has two.
`over ~by` computes each source's minimum separation, and the filter keeps the
rows that reach it. Candidates at exactly equal separation both stay:

```ocaml
let sep = Col.float "sep_arcsec"

let matches =
  Query.(
    in_radius
    |> filter Expr.(sep = over ~by:[ "source_id" ] (min sep))
    |> select
         Expr.[
           keep
             Sel.(
               names
                 [ "source_id"; "wise_id"; "sep_arcsec"; "abs_g"; "bp_rp"; "w1" ]);
         ])
```

```ocaml
# let () = show matches;;
table 3 rows × 6 columns
 source_id  wise_id  sep_arcsec  abs_g    bp_rp     w1
 string     string   float64     float64  float64   float64
 s1         w1         0.338289  7.00000  0.900000  11.2000
 s2         w3         0.720000  7.99485  1.100000  13.1000
 s6         w5         0.358630  9.01030  0.800000  15.1000
```

## The Unmatched

An `Anti` join keeps each selected source with no row in `matches`:

```ocaml
let unmatched =
  Query.(
    selected
    |> join ~kind:Anti ~on:(Join.keys [ "source_id" ]) matches
    |> select Expr.[ keep Sel.(names [ "source_id"; "ra"; "dec" ]) ])
```

```ocaml
# let () = show unmatched;;
table 1 row × 3 columns
 source_id  ra       dec
 string     float64  float64
 s5         100.400  -20.4000
```

`w4` has `s5`'s declination and lies 1.18 arcsec away, so the filter removes
the pair. The cuts remove `s3` and `s4`, so they appear in neither table.

## Contracts

In [`talon.mli`](../lib/talon.mli):

- `Expr.nx`, the Elementwise operations section of `Expr`
- join columns and name clashes, the preamble of `Join`; comparison atoms, its
  Conditions section; the band a range join reads, its Range joins section;
  `Anti`, its Kinds and counts section
- `over`, the Frames section of `Expr`
