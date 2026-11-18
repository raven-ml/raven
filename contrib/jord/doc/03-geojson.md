# GeoJSON

`jord.geojson` reads GeoJSON's polygons into `Polygons.t`
([`jord_geojson.mli`](../lib/geojson/jord_geojson.mli)). It takes the
`Jsont.json` the caller decoded, so it reads text from a file, a network
reply or a string alike. Add `jord.geojson`, `jsont` and, to decode text,
`jsont.bytesrw` to an executable's `libraries`. The toplevel prints each
system as `Crs.pp` does, by its EPSG name:

```ocaml
# open Jord;;
# #install_printer Crs.pp;;
```

## Fields as features

This collection holds one field, a polygon about 70 m by 110 m at 15°E,
52°N, with a label in its properties:

```ocaml
let text =
  {|{"type": "FeatureCollection", "features": [
      {"type": "Feature", "properties": {"label": "wheat"},
       "geometry": {"type": "Polygon", "coordinates": [[
         [15.0, 52.0], [15.001, 52.0], [15.001, 52.001], [15.0, 52.001],
         [15.0, 52.0]]]}}]}|}
```

The program decodes the text, then reads its features:

```ocaml
# let fields, properties =
    let json = Result.get_ok (Jsont_bytesrw.decode_string Jsont.json text) in
    Result.get_ok (Jord_geojson.polygons json);;
val fields : Polygons.t =
  {Polygons.crs = WGS 84; points = <abstr>; rings = [|0; 5|];
   features = [|0; 1|]}
val properties : Jsont.json array =
  [|Jsont.Object
     ([(("label", <abstr>), Jsont.String ("wheat", <abstr>))], <abstr>)|]
```

GeoJSON's positions are longitude and latitude, so the features are in
`Crs.lonlat` (the preamble of
[`jord_geojson.mli`](../lib/geojson/jord_geojson.mli)), and each ring
keeps its closing position (`polygons`).

## Cells of a grid

To rasterize the field on a 10 m UTM grid, the program moves its points to
the grid's system first, since a rule raises if the features' system is not
the grid's (the Rules section of [`rasterize.mli`](../lib/rasterize.mli)):

```ocaml
# let () =
    let utm = Crs.utm_at ~lon:15. ~lat:52. in
    let fields = Polygons.to_crs utm fields in
    let grid =
      Grid.v utm ~rows:40 ~cols:20 ~origin:(499950., 5761250.)
        ~row_step:(0., -10.) ~col_step:(10., 0.)
    in
    let feature, _ = Rasterize.centres grid fields in
    Printf.printf "%d cells\n" (Nx.shape feature).(0);;
77 cells
```

`Polygons.to_crs` moves the vertices and keeps the edges straight, so an
edge along a parallel becomes a chord of the parallel's image (`to_crs` in
[`polygons.mli`](../lib/polygons.mli)). The field's 70 m by 110 m hold
about 7 × 11 cells of 10 m.
