# Shapefiles

`jord.shapefile` reads an ESRI Shapefile's polygons and its dBASE
attributes from their bytes
([`jord_shapefile.mli`](../lib/shapefile/jord_shapefile.mli)). It depends
on nothing beyond `jord`. Add `jord.shapefile` to an executable's
`libraries`.

```ocaml
# open Jord;;
```

## Reading the files beside each other

A shapefile is several files with one base name. The `.prj` file states the
system in WKT 1, which `Crs.of_wkt` reads, and the `.shp` and `.dbf` files
hold the shapes and their attributes. This function reads all three:

```ocaml
# let read_shapefile base =
    let read ext = In_channel.with_open_bin (base ^ ext) In_channel.input_all in
    let ( let* ) = Result.bind in
    let* crs = Crs.of_wkt (read ".prj") in
    Jord_shapefile.read crs ~shp:(read ".shp") ~dbf:(read ".dbf");;
val read_shapefile :
  string -> (Polygons.t * (string * string array) list, string) result =
  <fun>
```

Each feature is one shape, in the file's order, and each attribute is a
field's name with one text per feature (`read` in
[`jord_shapefile.mli`](../lib/shapefile/jord_shapefile.mli)).
`Crs.of_wkt` reads WKT 1 in Esri's and OGC's trees for WGS 84 and its UTM
zones, as GDAL and PROJ write them, in Esri's for NAD83 and the UTM zones
`Crs.of_epsg` reads on it, a shapefile's usual `.prj` in the United
States, and in GDAL's tree for MODIS's sinusoid. It reads WKT2 by its EPSG
identifiers, and MODIS's sinusoid as PROJ writes it without them. Any
other text is an `Error` (the Reading WKT section of
[`crs.mli`](../lib/crs.mli)), such as a file in a European grid:

```ocaml
# Crs.of_wkt {|PROJCS["ETRS89_LAEA_Europe",GEOGCS["GCS_ETRS_1989"]]|};;
- : (Crs.t, string) result =
Error
 "PROJCS[\"ETRS89_LAEA_Europe\"] is not the WKT of WGS 84 or NAD83, of a UTM zone on them that Crs.of_epsg reads, or of MODIS's sinusoid"
```
