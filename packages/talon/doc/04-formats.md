# Formats

Files enter a query as **sources**: data whose columns are known when the
source is built and whose rows arrive in batches when the query runs. This
page covers CSV and Parquet, the errors they report, and writing a source of
your own.

## Sources

`Query.of_source s` makes a query of a source. Running the query asks the
source for the columns it needs, offers it the filters it may apply, and
reads its rows batch by batch. Each place in a plan that reads a source is a
read of its own, and a read of a file opens it again.

```ocaml
open Talon

let ( let* ) = Result.bind

let late flights =
  Query.(of_source flights |> filter Expr.(Col.int "dep_delay" > int 15))

let late_flights () =
  let* flights = Talon_csv.file ~nulls:[ "NA" ] "flights.csv" in
  Query.run (late flights)
```

## CSV

`Talon_csv.file path` returns the source of a CSV file. Without a format, it
**sniffs** one from the file's first 16,384 records:

- the separator: `,`, a tab, `;` or `|`, whichever splits every record into
  the same number of fields, and into the most;
- the header: the first record names the columns, unless, in every column
  whose values are not text, its field reads as that column's type;
- the types: `bool`, `int64`, `float64`, `date`, `datetime[us]` (or
  `datetime[us, UTC]` with offsets) when every value reads as one, and
  `string` otherwise.

A number with a leading zero, such as `007`, is text, since it writes an
identifier. Sniffing never infers a categorical.

An unquoted empty field is null, and so is one equal to a null token given
with `~nulls`. A quoted field is never null: `""` is the empty string.
Fields follow RFC 4180, strictly: a quoted field may hold separators, line
breaks and doubled quotes.

To see or change a format, sniff it from a reader:

```ocaml
let csv =
  String.concat "\n"
    [
      "station,day,temp,note";
      "oslo,2024-03-01,-3.5,";
      {|oslo,2024-03-02,NA,"sensor ""B"" offline"|};
      {|lima,2024-03-01,22,"humid, calm"|};
    ]

let reader () = Bytesrw.Bytes.Reader.of_string csv
let format = Error.get_ok (Talon_csv.sniff ~nulls:[ "NA" ] (reader ()))
let () = Format.printf "%a@." Talon_csv.pp_format format
```

```text
csv (4 columns, separator ',', quote '"', header, nulls ["NA"]), types sniffed from 3 rows
  station string
  day date
  temp float64
  note string
```

`Talon_csv.with_type` declares another type for a column, and
`Talon_csv.format` states a whole format. `Talon_csv.decode` reads a reader
into a table, and `Talon_csv.file ~format` reads a file as the format says:

```ocaml
let format = Talon_csv.with_type "temp" (Type.Any Type.float32) format
let readings = Error.get_ok (Talon_csv.decode format (reader ()))
```

```text
table 3 rows × 4 columns
 station  day         temp      note
 string   date        float32   string
 oslo     2024-03-01  -3.50000  ∅
 oslo     2024-03-02         ∅  sensor "B" offline
 lima     2024-03-01  22.00000  humid, calm
```

A field that does not read as its column's type fails the read. The error
gives the line, the column, the text and the fix, and says when the type was
sniffed:

```text
line 3, column 6: "warm": column "temp": cannot read as float64: not a number. Declare the null token (~nulls) or another type (with_type).
```

`Talon_csv.encode` runs a query and writes its rows in a format, which
`decode` reads back. A field is quoted when it holds the separator, the
quote or a line break, or when it is empty text:

```ocaml
let buf = Buffer.create 256

let () =
  Error.get_ok
    (Talon_csv.encode format (Query.of_table readings)
       (Bytesrw.Bytes.Writer.of_buffer buf))
```

```text
station,day,temp,note
oslo,2024-03-01,-3.5,
oslo,2024-03-02,,"sensor ""B"" offline"
lima,2024-03-01,22,"humid, calm"
```

## Parquet

`Talon_parquet.file path` maps a Parquet file in memory and reads its
footer. The source reads only the column chunks a query needs, reads
uncompressed, Snappy, gzip, Zstandard and `LZ4_RAW` chunks, and states its
number of rows.

Each Parquet column reads as the talon type that holds its values:
`boolean` as `bool`, `int32` and `int64` as their annotated integer type,
`float` and `double` as `float32` and `float64`, `DATE` as `date`,
`TIMESTAMP` as `datetime` of its unit (in `UTC` when adjusted to UTC),
strings as `string`, and other byte arrays as `binary`.
`Talon_parquet.sniff` reads the format from a file's bytes, which
`Nx_device.Buffer.of_file` maps:

```ocaml
let carriers () =
  let path = "carriers.parquet" in
  let* bytes = Nx_device.Buffer.of_file path |> Result.map_error (Error.v ~file:path) in
  let* format = Talon_parquet.sniff bytes in
  Format.printf "%a@." Talon_parquet.pp_format format;
  Ok (Talon_parquet.source format bytes)
```

```text
parquet (2 columns)
  carrier string ← optional binary (STRING)
  name string    ← optional binary (STRING)
```

A `DECIMAL` column has no talon type of its own: declare it with
`Talon_parquet.with_type` as `float64`, or, up to 18 digits, as `int64` for
its unscaled integer. Talon refuses encrypted files, nested columns and the
codecs it does not read when it sniffs the footer.

A Parquet source uses filters to skip row groups. A comparison of a column
with a literal goes to the source, which skips each row group whose minimum
and maximum show that no row passes; the filter stays in the plan for the
rows of the groups it reads. `Query.optimize` shows it:

```text
query → carrier string, name string
filter (carrier >= "F")
└ parquet (2 columns) ~filters:[carrier >= "F"]
```

## Errors

Failures in files are `Talon.Error.t` values. An error holds a message and
the places it was found: the file, the line and column of its text, the
Parquet row group, the byte range, and the raw text that failed. `Error.pp`
prints them coarsest first, as in `flights.csv:48213:12: "NA": …`, and
`Error.get_ok` raises `Failure` with that text.

## Your Own Source

`Source.v` makes a source of anything that yields tables: a format talon does
not ship, a database cursor, a simulation. It takes a name, a schema, and a
function from a request to the source's parts, each of which opens a reader
of batches. This source counts from `0` in parts of 1,000 rows:

```ocaml
let counter n =
  let schema = Schema.v [ ("i", Type.Any Type.int64) ] in
  let batch (req : Source.request) lo hi =
    if req.columns = [] then Talon.v ~rows:(hi - lo) []
    else Talon.v [ ("i", Column.v Type.int64 (Array.init (hi - lo) (( + ) lo))) ]
  in
  let part req lo hi =
    let open_ () =
      let read = ref false in
      let next () =
        if !read then Ok None
        else (
          read := true;
          Ok (Some (batch req lo hi)))
      in
      Ok { Source.next; close = ignore }
    in
    { Source.rows = Some (hi - lo); open_ }
  in
  let parts req =
    Ok (List.init ((n + 999) / 1000) (fun k -> part req (k * 1000) (min n ((k + 1) * 1000))))
  in
  Source.v ~name:"counter" ~schema ~rows:n ~sorted:[ Order.asc "i" ] parts

let total =
  Query.(of_source (counter 2500) |> aggregate ~by:[] Expr.[ "sum" := sum (Col.int "i") ])
```

- A batch has the request's columns, in schema order. A query that only
  counts rows asks for none.
- `~sorted` claims an order, which talon checks as rows arrive.
- `~pushdown` answers, for each filter, whether the source applies it
  (`Exact`), uses it to skip rows (`Inexact`), or ignores it
  (`Unsupported`). It defaults to `Unsupported`.

A reader's `next` returns an `Error` for a failure in the data, and talon
calls `close` once it needs no more of the part.

The example [05-formats](../examples/05-formats/README.md) reads and writes
this page's CSV and reads a Parquet file.
