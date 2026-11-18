# Sentinel-2 metadata

A Sentinel-2 product states, in its `MTD_MSIL1C.xml` at Level-1C or its
`MTD_MSIL2A.xml` at Level-2A, the values that decode its DNs: the
quantification value, the special values NODATA and SATURATED, and from
processing baseline 04.00 an offset for each band (the Reflectance
section of [`sentinel2.mli`](../lib/sentinel2.mli)). `jord.safe` reads them
from the file's bytes into `Sentinel2.metadata`, by `Jord_safe.l1c` and
`Jord_safe.l2a` ([`jord_safe.mli`](../lib/safe/jord_safe.mli)). Add
`jord.safe` to an executable's `libraries`.

```ocaml
# open Jord;;
```

## From the file to reflectance

This program reads the metadata of a product of tile 33UUU, processed at
baseline 05.10, which jord's tests hold:

```ocaml
# let metadata =
    let xml =
      In_channel.with_open_bin "../test/safe/mtd_n0510.xml" In_channel.input_all
    in
    Result.get_ok (Jord_safe.l2a xml);;
val metadata : Sentinel2.metadata =
  {Sentinel2.baseline = "05.10"; quantification = 10000.;
   offsets =
    [(Sentinel2.B01, -1000); (Sentinel2.B02, -1000); (Sentinel2.B03, -1000);
     (Sentinel2.B04, -1000); (Sentinel2.B05, -1000); (Sentinel2.B06, -1000);
     (Sentinel2.B07, -1000); (Sentinel2.B08, -1000); (Sentinel2.B8A, -1000);
     (Sentinel2.B09, -1000); (Sentinel2.B10, -1000); (Sentinel2.B11, -1000);
     (Sentinel2.B12, -1000)];
   nodata = 0; saturated = 65535;
   wavelengths =
    [(Sentinel2.B01, {Sentinel2.central = 442.7; min = 412.; max = 456.});
     (Sentinel2.B02, {Sentinel2.central = 492.7; min = 456.; max = 533.});
     (Sentinel2.B03, {Sentinel2.central = 559.8; min = 538.; max = 583.});
     (Sentinel2.B04, {Sentinel2.central = 664.6; min = 646.; max = 684.});
     (Sentinel2.B05, {Sentinel2.central = 704.1; min = 695.; max = 714.});
     (Sentinel2.B06, {Sentinel2.central = 740.5; min = 731.; max = 749.});
     (Sentinel2.B07, {Sentinel2.central = 782.8; min = 769.; max = 797.});
     (Sentinel2.B08, {Sentinel2.central = 832.8; min = 760.; max = 907.});
     (Sentinel2.B8A, {Sentinel2.central = 864.7; min = 837.; max = 881.});
     (Sentinel2.B09, {Sentinel2.central = 945.1; min = 932.; max = 958.});
     (Sentinel2.B10, {Sentinel2.central = 1373.5; min = 1337.; max = 1412.});
     (Sentinel2.B11, {Sentinel2.central = 1613.7; min = 1539.; max = 1682.});
     (Sentinel2.B12, {Sentinel2.central = 2202.4; min = 2078.; max = 2320.})]}
```

`Sentinel2.reflectance` makes each band's decode from these values, and
`Band.decode` applies it to the band's DNs:

```ocaml
# let () =
    let red = Result.get_ok (Sentinel2.reflectance metadata B04) in
    let dn = Nx.create Nx.uint16 [| 4 |] [| 0; 1000; 1500; 65535 |] in
    let v, ok = Band.decode red Nx.float32 dn in
    Nx.print v;
    Nx.print ok;;
float32 [4] [0, 0, 0.05, 0]
bool [4] [false, true, true, false]
```

DN 1000 is reflectance 0, since the product shifts its range by 1000 to
keep negative reflectances. DN 0 is NODATA and 65535 SATURATED, so both
are masked. A product processed before baseline 04.00 states no offsets,
and `Sentinel2.offset` is 0 for each of its bands (the Reflectance section
of [`sentinel2.mli`](../lib/sentinel2.mli)).

The file also states each band's wavelengths in nanometres, those of its
satellite's instrument, `CENTRAL`, `MIN` and `MAX` (`Sentinel2.wavelength`
in the Bands section of [`sentinel2.mli`](../lib/sentinel2.mli)). In this
product, each band's `CENTRAL` lies within 0.05 nm of the mean of its
tabulated response weighted by it, and its `MIN` and `MAX` are the ends of
that response. B04's are 38 nm apart, and its response is 30.6 nm wide at
half its maximum. `Sentinel2.reflectance` gives them to the band in
micrometres, `CENTRAL` as its centre wavelength and `MAX` - `MIN` as its
width:

```ocaml
# let red = Result.get_ok (Sentinel2.reflectance metadata B04) in
  (red.center_wavelength, red.full_width_half_max);;
- : float option * float option = (Some 0.6646, Some 0.038)
```

These are Sentinel-2A's. A Sentinel-2B product states its own, as 665 nm
for B04's `CENTRAL`.

## One tile's angles

`Jord_safe.tile` reads a tile's sensing time and its angle arrays of the
Sun's and the view's angles (the Tiles section of
[`jord_safe.mli`](../lib/safe/jord_safe.mli)). The angle arrays of tile
33UUU of 2024-06-29 show what that section's rules are for. The figures
below are measured on the tile's whole `MTD_TL.xml`, whose 91 views are
the 13 bands at detectors 5 to 11. jord's tests hold a cut of that file
(`test/safe/mtd_tl_n0510.xml`, whose dune file gives its source) with four
of the views, B04's and B10's at detectors 7 and 8.

- At a node that two detectors of one band both see, their angles differ
  by up to 0.25° in zenith (B10, detectors 6 and 7) and 179° in azimuth
  (B12, detectors 6 and 7), with the azimuth's difference taken from 0° to
  180°. B04's differ by up to 0.23° and 172°, and in the cut file, where
  only detectors 7 and 8 meet, by 0.11° and 61°. So an angle is read from
  one detector's angle arrays at a time.
- Near nadir the azimuth turns fast. B04's detector 6 has azimuths from
  21° to 80° over zeniths of 0.97° to 2.25°, so a view is read by its
  direction.
- Predicting each node of the 91 views from its two neighbours 10 km away,
  the direction errs by up to 0.0005° in zenith and 0.03° in azimuth, the
  file's printing, and the angles, the azimuth by its sine and cosine, by
  0.086° and 10.2°.
- Each of B04's 10980 × 10980 pixels is read from its own detector's view,
  the detector that B04's footprint mask gives it. The mask is
  `GRANULE/L1C_T33UUU_A047109_20240629T102556/QI_DATA/MSK_DETFOO_B04.jp2`
  of the Level-1C product whose `MTD_MSIL1C.xml` is
  `test/safe/mtd_l1c_n0510.xml`, and jord does not read it. One call over
  all of the band's views, without `valid`, gives NaN at a fifth of the
  pixels (20.6%), where a node around the pixel is NaN in that view. Read
  one detector at a time with `valid`, 2.9% of the pixels, near the seams
  between detectors, get no value, since their containing node cell is NaN
  in that detector's view.
- Extrapolated one node past each detector's view, linearly from the two
  nearest finite nodes along a row or a column, the angles err by up to
  0.17° in zenith and 20° in azimuth, and the direction by 0.0011° and
  0.06°.
- The view's azimuths east of the satellite's track point back west to it,
  as the Sun's azimuth points to the Sun.

This program computes the cut file's figures for B04, the largest
difference between detectors 7 and 8 at a node that both see:

```ocaml
# let () =
    let xml =
      In_channel.with_open_bin "../test/safe/mtd_tl_n0510.xml"
        In_channel.input_all
    in
    let t = Result.get_ok (Jord_safe.tile xml) in
    let z, a = t.view in
    let at x d = Nx.slice Nx.[ A; A; I (Sentinel2.index B04); I (d - 1) ] x in
    let both = Nx.logical_and (Nx.isfinite (at z 7)) (Nx.isfinite (at z 8)) in
    let largest d = Nx.item [] (Nx.max (Nx.where both d (Nx.zeros_like d))) in
    let dz = Nx.abs (Nx.sub (at z 7) (at z 8)) in
    let da = Nx.abs (Nx.sub (at a 7) (at a 8)) in
    let da = Nx.minimum da (Nx.sub (Nx.full_like da 360.) da) in
    Printf.printf "%.2f in zenith, %.0f in azimuth\n" (largest dz)
      (largest da);;
0.11 in zenith, 61 in azimuth
```

The azimuth's difference is taken from 0° to 180°, as above, and a node
one detector does not see is NaN in its angle arrays, which `both` leaves
out.
