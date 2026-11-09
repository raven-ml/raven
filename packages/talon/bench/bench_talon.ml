(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Filter, group, join and sort over the 40k-row fixture, the same work as
   bench_talon.py does in pandas; then the operations that read validities, over
   a generated table with nulls, a read of float texts, a read of a CSV text of
   mixed columns, and reads of a Parquet file with nulls, of one of
   dictionary-encoded text, unfiltered and filtered, and of one of sorted
   keys. *)

open Talon

(* Resolve fixtures next to the executable, not the working directory: the bench
   rule runs with the bench dir as cwd, dune exec with the project root — the
   exe path is the one stable anchor in both. *)
let data_dir = Filename.concat (Filename.dirname Sys.executable_name) "data"

let start_dir = Sys.getcwd ()

(* [in_data f] is [f ()] run from [data_dir], so that a row names a file by its
   bare name. A source's name holds the path it reads, and the words a row
   allocates must not depend on how deep the checkout lies. *)
let in_data f =
  Sys.chdir data_dir;
  let r = f () in
  Sys.chdir start_dir;
  r

let customers_csv, transactions_csv = Fixtures.generate ()

(* [load text columns categories] reads the CSV [text] as [columns], its text
   columns [categories] as categoricals, as pandas reads them. *)
let load text columns categories =
  let format = Talon_csv.format columns in
  let table =
    Error.get_ok (Talon_csv.decode format (Bytesrw.Bytes.Reader.of_string text))
  in
  let q = Error.get_ok (Kit.categorize categories (Query.of_table table)) in
  Error.get_ok (Query.run q)

let string = Type.Any Type.string
let int32 = Type.Any Type.int32
let float64 = Type.Any Type.float64

let transactions =
  load transactions_csv
    [
      ("transaction_id", int32);
      ("customer_id", int32);
      ("region", string);
      ("category", string);
      ("channel", string);
      ("amount", float64);
      ("quantity", int32);
      ("discount", float64);
      ("promo", string);
      ("event_date", string);
    ]
    [ "region"; "category"; "channel"; "promo" ]

let customers =
  load customers_csv
    [
      ("customer_id", int32);
      ("segment", string);
      ("region", string);
      ("status", string);
      ("loyalty_score", float64);
      ("tenure_years", int32);
    ]
    [ "segment"; "region"; "status" ]

let amount = Col.float "amount"

(* [total q] is the sum of [q]'s amounts. *)
let total q =
  let sums = Query.aggregate ~by:[] Expr.[ "amount" := sum amount ] q in
  (Column.values Kind.float (column (Error.get_ok (Query.run sums)) "amount")).(0)

let filtered () =
  Query.(
    of_table transactions
    |> filter
         Expr.(
           amount > float 120.
           && Col.int "quantity" >= int 3
           && Col.string "region" = string "EMEA"))

let grouped () =
  Query.(
    of_table transactions
    |> aggregate ~by:[ "category"; "region" ] Expr.[ "amount" := sum amount ])

(* Both tables have a [region]; talon adds no suffixes, so the customers' is
   renamed. *)
let joined () =
  Query.(
    of_table transactions
    |> join ~kind:Left
         ~on:(Join.keys [ "customer_id" ])
         (of_table customers |> Kit.rename [ ("region", "customer_region") ]))

(* The same join to every customer twice, a right whose keys repeat. *)
let joined_twice () =
  let customers =
    Query.of_table customers |> Kit.rename [ ("region", "customer_region") ]
  in
  Query.(
    of_table transactions
    |> join ~kind:Left
         ~on:(Join.keys [ "customer_id" ])
         (append customers customers))

(* An eighth of the customers to their transactions: a right whose keys repeat
   and whose rows mostly match no left row, as a few orders find their lines
   among all of them. *)
let few_customers () =
  let few = rows customers / 8 in
  Query.(
    of_table customers
    |> slice ~offset:0 ~length:few
    |> join
         ~on:(Join.keys [ "customer_id" ])
         (of_table transactions
         |> Kit.rename [ ("region", "transaction_region") ]))

let sorted () = Query.(of_table transactions |> sort [ Order.desc "amount" ])

let first_amount q =
  (Column.values Kind.float (column (Error.get_ok (Query.run q)) "amount")).(0)

(* Checks *)

let check name ~rows:expected_rows ~amount:expected q value =
  let got_rows = rows (Error.get_ok (Query.run q)) in
  let close = Float.abs (value -. expected) <= 1e-9 *. Float.abs expected in
  if got_rows <> expected_rows || not close then
    failwith
      (Printf.sprintf "%s: %d rows and %.17g, expected %d and %.17g" name
         got_rows value expected_rows expected)

(* The rows and amounts that bench_talon.py's pandas computes. *)
let () =
  check "filter" ~rows:514 ~amount:78520.97 (filtered ()) (total (filtered ()));
  check "group" ~rows:24 ~amount:2608626.10 (grouped ()) (total (grouped ()));
  check "join" ~rows:40000 ~amount:2608626.10 (joined ()) (total (joined ()));
  check "sort" ~rows:40000 ~amount:427.44 (sorted ()) (first_amount (sorted ()))

(* Nulls *)

(* A table of 2^20 rows drawn from a seeded key: an int64 [key] in [0, 1000)
   with 1% nulls, a float64 [f] and a string [s] with 10% nulls each, two bool
   columns [a] and [b] with 10% nulls each, and the columns [i] (int64) and [g]
   (float64) without nulls. *)
let n = 1 lsl 20
let keys = Nx.Rng.split ~n:13 (Nx.Rng.key 22)

let valid k p =
  Nx.greater_equal_s (Nx.Rng.uniform ~key:keys.(k) Nx.float64 [| n |]) p

let ints k high = Nx.cast Nx.int64 (Nx.Rng.randint ~key:keys.(k) ~high [| n |])
let bools k = Nx.Rng.bernoulli ~key:keys.(k) (Nx.full Nx.float64 [| n |] 0.5)

let nullable ?valid x =
  Column.of_tensor ?validity:(Option.map (Nx.cast Nx.bit) valid) x

let key_values = ints 0 1000
let key_valid = valid 1 0.01

let strings =
  let values = Nx.to_array (ints 2 100_000) in
  let valid = Nx.to_array (valid 3 0.1) in
  Column.of_options Type.string
    (Array.init n (fun r ->
         if valid.(r) then Some ("s" ^ Int64.to_string values.(r)) else None))

let nulls =
  Talon.v
    [
      ("key", nullable ~valid:key_valid key_values);
      ( "f",
        nullable ~valid:(valid 4 0.1)
          (Nx.Rng.uniform ~key:keys.(5) Nx.float64 [| n |]) );
      ("s", strings);
      ("a", nullable ~valid:(valid 6 0.1) (bools 7));
      ("b", nullable ~valid:(valid 8 0.1) (bools 9));
      ("i", nullable (ints 10 1_000_000));
      ("g", nullable (Nx.Rng.uniform ~key:keys.(11) Nx.float64 [| n |]));
    ]

let run q = Error.get_ok (Query.run q)

(* Groups: 2^20 rows of a text key [t] of 100 values, ["id001"] to
   ["id100"], an int32 key [k] of 100 values and an int32 measure [v] in
   [1, 5], drawn from a seeded key, as H2O's group-by table holds. *)
let groups =
  let keys = Nx.Rng.split ~n:3 (Nx.Rng.key 25) in
  let draw k high = Nx.Rng.randint ~key:keys.(k) ~high [| n |] in
  let text i = Some (Printf.sprintf "id%03ld" (Int32.succ i)) in
  Talon.v
    [
      ( "t",
        Column.of_options Type.string
          (Array.map text (Nx.to_array (draw 0 100))) );
      ("k", nullable (draw 1 100));
      ("v", nullable (Nx.add_s (draw 2 5) 1l));
    ]

(* [by key] sums [v] and takes its mean in each group of [key]. *)
let by key =
  run
    Query.(
      of_table groups
      |> aggregate ~by:[ key ]
           Expr.[ "s" := sum (Col.int "v"); "m" := mean (Col.int "v") ])

(* [over_k os] is [os] in each group of [k]. *)
let over_k os = run Query.(of_table groups |> aggregate ~by:[ "k" ] os)

(* [batched t] is [t]'s rows in batches of 2^14 rows. *)
let batched t =
  let n = rows t and size = 1 lsl 14 in
  let batch i =
    let stop = Stdlib.min n ((i + 1) * size) in
    Talon.take (Nx.arange Nx.int64 (i * size) stop 1) t
  in
  Talon.of_batches (List.init ((n + size - 1) / size) batch)

(* Streamed aggregates of 2^20 rows in batches: over the [groups] keys, few
   groups, and over a key that repeats every 4 rows, a quarter as many groups
   as rows. *)
let streamed_few = lazy (batched groups)

let many_groups =
  lazy
    (let n = 1 lsl 20 in
     Talon.v
       [
         ("k", Column.of_tensor (Nx.repeat ~axis:0 4 (Nx.arange Nx.int64 0 n 4)));
         ("v", Column.of_tensor (Nx.arange Nx.int64 0 n 1));
       ])

let streamed_many = lazy (batched (Lazy.force many_groups))

let streamed t =
  run
    Query.(
      of_table (Lazy.force t)
      |> aggregate ~by:[ "k" ]
           Expr.[ "s" := sum (Col.int "v"); "c" := count (Col.int "v") ])

let streamed_moments t =
  run
    Query.(
      of_table (Lazy.force t)
      |> aggregate ~by:[ "k" ]
           Expr.[ "v" := var (Col.int "v"); "d" := std (Col.int "v") ])

(* [filter th] keeps the rows whose [key] is below [th]: [th / 10] percent of
   them. *)
let filter th =
  run Query.(of_table nulls |> filter Expr.(Col.int "key" < int th))

let kleene e = run Query.(of_table nulls |> select Expr.[ "r" := e ])

let take_random =
  let indices = ints 12 n in
  let t = Talon.v [ ("key", nullable ~valid:key_valid key_values) ] in
  fun () -> Talon.take indices t

(* The text column [s] at 2^18 random rows: each row's bytes are copied
   once. *)
let take_text =
  let indices = Nx.slice [ R (0, 1 lsl 18) ] (ints 12 n) in
  let t = Talon.v [ ("s", strings) ] in
  fun () -> Talon.take indices t

(* [nested ()] is 2^18 rows of lists of zero to three records, each an int64
   [k] and a list of zero to three texts [tags], a tenth of the rows null. It
   is made in the case's worker. *)
let nested () =
  let tag i = "tag" ^ string_of_int (i * 7919 mod 1000) in
  let record i =
    Record.(
      empty
      |> add Kind.int "k" (Some i)
      |> add (Kind.list Kind.string) "tags" (Some (Array.init (i mod 4) tag)))
  in
  let row r =
    if r mod 10 = 3 then None
    else Some (Array.init (r * 31 mod 4) (fun j -> record ((r * 4) + j)))
  in
  let ty =
    Type.list
      (Type.record
         [
           ("k", Type.Any Type.int64);
           ("tags", Type.Any (Type.list Type.string));
         ])
  in
  Talon.v [ ("x", Column.of_options ty (Array.init (1 lsl 18) row)) ]

(* The nested column at 2^16 random rows: at each depth every row's elements
   are copied as one run. *)
let take_nested =
  let indices = Nx.slice [ R (0, 1 lsl 16) ] (ints 12 (1 lsl 18)) in
  fun t -> Talon.take indices t

(* [text_layout ()] is the layout of 2^20 ASCII texts, ["s0"] to ["s99999"],
   which [Column.of_layout] checks as a text column: offsets and UTF-8. It is
   made in the case's worker, so that no other case's process holds it. *)
let text_layout () =
  let text i = "s" ^ string_of_int (i * 7919 mod 100_000) in
  let texts = Array.init n text in
  let bytes = String.concat "" (Array.to_list texts) in
  let offsets = Array.make (n + 1) 0L in
  Array.iteri
    (fun i t ->
      offsets.(i + 1) <- Int64.add offsets.(i) (Int64.of_int (String.length t)))
    texts;
  Column.Varsize
    {
      validity = None;
      offsets = Nx.create Nx.int64 [| n + 1 |] offsets;
      child =
        Column.of_tensor
          (Nx.init Nx.uint8 [| String.length bytes |] (fun i ->
               Char.code bytes.[i.(0)]));
    }

(* Three parts of odd lengths, the second and third starting inside a byte of
   the validity. *)
let concat_3 =
  let bits = Nx.cast Nx.bit key_valid in
  let part offset length =
    let values = Nx.slice [ Nx.R (offset, offset + length) ] key_values in
    let validity = Nx.slice [ Nx.R (offset, offset + length) ] bits in
    Talon.v [ ("key", Column.of_tensor ~validity values) ]
  in
  let third = n / 3 in
  let t =
    Talon.of_batches
      [
        part 0 (third + 5);
        part (third + 5) (third - 3);
        part ((2 * third) + 2) (n - (2 * third) - 2);
      ]
  in
  fun () -> Talon.column t "key"

let joined_nulls =
  let right =
    Talon.v
      [
        ("key", Column.v Type.int64 (Array.init 500 (fun k -> 2 * k)));
        ("w", Column.v Type.float64 (Array.init 500 float_of_int));
      ]
  in
  Query.(
    of_table nulls |> join ~kind:Left ~on:(Join.keys [ "key" ]) (of_table right))

(* Spikes in trials, a peri-event analysis: 2^20 spike times in order over an
   hour, and 1000 trials of one second every 3.6 s on the left, each with the
   spikes it holds, about 290k pairs. A recording holds about 10^7 spikes;
   2^20 keeps the suite within its time. *)
let spikes =
  let key = Nx.Rng.key 26 in
  let t, _ =
    Nx.sort (Nx.mul_s (Nx.Rng.uniform ~key Nx.float64 [| n |]) 3600.)
  in
  Talon.v [ ("t", nullable t) ]

let trials =
  let start = Nx.mul_s (Nx.arange Nx.float64 0 1000 1) 3.6 in
  Talon.v
    [
      ("start", nullable start);
      ("stop", nullable (Nx.add_s start 1.));
      ("trial", nullable (Nx.arange Nx.int64 0 1000 1));
    ]

let spikes_in_trials () =
  run
    Query.(
      of_table trials
      |> join ~on:Join.(le "start" "t" && gt "stop" "t") (of_table spikes))

(* An inequality within keys: 2^16 left rows and 2^10 right ones in 100
   groups, each left [x] with the right rows of its group whose [y] is not
   above it, about 330k pairs. Neither side is in order. *)
let below =
  let keys = Nx.Rng.split ~n:4 (Nx.Rng.key 27) in
  let side k rows x =
    Talon.v
      [
        ( "g",
          nullable
            (Nx.cast Nx.int64
               (Nx.Rng.randint ~key:keys.(k) ~high:100 [| rows |])) );
        (x, nullable (Nx.Rng.uniform ~key:keys.(k + 1) Nx.float64 [| rows |]));
      ]
  in
  let left = side 0 (1 lsl 16) "x" and right = side 2 (1 lsl 10) "y" in
  fun () ->
    run
      Query.(
        of_table left
        |> join ~on:Join.(keys [ "g" ] && ge "x" "y") (of_table right))

let parquet = "nulls.parquet"

(* Texts *)

(* 2^16 float64 texts of 17 significant digits, as a writer that keeps every
   bit of a float writes them, read as float64. *)
let texts =
  let xs =
    Nx.to_array (Nx.Rng.uniform ~key:(Nx.Rng.key 23) Nx.float64 [| 1 lsl 16 |])
  in
  Column.v Type.string (Array.map (fun x -> Printf.sprintf "%.17g" (1000. *. x)) xs)

let parse_floats () = Result.get_ok (Column.parse float64 texts)

let read_file file =
  in_data (fun () ->
      run (Query.of_source (Error.get_ok (Talon_parquet.file file))))

let read_parquet () = read_file parquet

(* text.parquet's 2^18 dictionary-encoded words of 8 to 27 bytes: the read
   gathers each row's bytes from the dictionary. *)
let text_parquet = "text.parquet"
let read_text () = read_file text_parquet

(* The same words, filtered by one of them as the file is read: the source
   compares each word of a dictionary once. *)
let filter_text () =
  in_data (fun () ->
      run
        Query.(
          of_source (Error.get_ok (Talon_parquet.file text_parquet))
          |> filter Expr.(Col.string "s" = string "word 42 --")))

(* keys.parquet's 2^17 ascending int64 keys, each repeated 1 to 7 times, in
   a dictionary-encoded column and a PLAIN one: snappy pages whose copies
   repeat a key's high bytes from 8 bytes back, and bit-packed indices. *)
let keys_parquet = "keys.parquet"
let read_keys () = read_file keys_parquet

(* Writing the tables the reads make, to a sink: nulls compacted in the level
   and value streams, and dictionary-encoded text. *)
let write_table t =
  let q = Query.of_table t in
  Error.get_ok
    (Talon_parquet.encode
       (Talon_parquet.format (Query.schema q))
       q
       (Bytesrw.Bytes.Writer.ignore ()))

let nulls_table = lazy (read_parquet ())
let text_table = lazy (read_text ())
let write_nulls () = write_table (Lazy.force nulls_table)
let write_text () = write_table (Lazy.force text_table)

(* The nullable table as an Arrow file: a read maps its buffers. *)
let arrow_bytes =
  lazy
    (let q = Query.of_table (Lazy.force nulls_table) in
     let b = Buffer.create (1 lsl 24) in
     let f = Talon_arrow.format (Query.schema q) in
     let w = Bytesrw.Bytes.Writer.of_buffer b in
     Error.get_ok (Talon_arrow.encode f q w);
     Buffer.contents b)

let read_arrow () =
  let b = Rig.Buffer.of_string (Lazy.force arrow_bytes) in
  let f = Error.get_ok (Talon_arrow.sniff b) in
  run (Query.of_source (Talon_arrow.source f b))

let write_arrow () =
  let q = Query.of_table (Lazy.force nulls_table) in
  Error.get_ok
    (Talon_arrow.encode
       (Talon_arrow.format (Query.schema q))
       q
       (Bytesrw.Bytes.Writer.ignore ()))

(* CSV *)

(* The text of 2^16 records, about 1.7 MB, so that a read crosses one of the
   reader's one-MiB batches: an int64 [id] in [[-10^6, 10^6)], a float64
   [price] of two decimals with 5% nulls, and a string [label], 10% of it
   quoted around a separator and doubled quotes. The bench reads it from
   memory. *)
let csv_text =
  let n = 1 lsl 16 in
  let keys = Nx.Rng.split ~n:4 (Nx.Rng.key 24) in
  let uniform k =
    Nx.to_array (Nx.Rng.uniform ~key:keys.(k) Nx.float64 [| n |])
  in
  let id =
    Nx.to_array
      (Nx.Rng.randint ~key:keys.(0) ~low:(-1_000_000) ~high:1_000_000 [| n |])
  in
  let price = uniform 1 and null = uniform 2 and quoted = uniform 3 in
  let b = Buffer.create (1 lsl 21) in
  Buffer.add_string b "id,price,label\n";
  for r = 0 to n - 1 do
    Buffer.add_string b (Int32.to_string id.(r));
    Buffer.add_char b ',';
    if null.(r) >= 0.05 then
      Buffer.add_string b (Printf.sprintf "%.2f" (1000. *. price.(r)));
    Buffer.add_char b ',';
    if quoted.(r) < 0.1 then
      Buffer.add_string b (Printf.sprintf "\"item %d, \"\"boxed\"\"\"" r)
    else Buffer.add_string b (Printf.sprintf "item-%d" r);
    Buffer.add_char b '\n'
  done;
  Buffer.contents b

let csv_format =
  Talon_csv.format
    [ ("id", Type.Any Type.int64); ("price", float64); ("label", string) ]

let read_csv () =
  Error.get_ok
    (Talon_csv.decode csv_format (Bytesrw.Bytes.Reader.of_string csv_text))

(* JSON *)

(* The rows of the CSV above as NDJSON, about 3 MB: an int64 [id], a float64
   [price] with nulls, and a string [label], some with escapes. Words count
   what a read allocates: the scanner, the builders and the columns. *)
let json_text =
  let n = 1 lsl 16 in
  let keys = Nx.Rng.split ~n:3 (Nx.Rng.key 24) in
  let uniform k =
    Nx.to_array (Nx.Rng.uniform ~key:keys.(k) Nx.float64 [| n |])
  in
  let id =
    Nx.to_array
      (Nx.Rng.randint ~key:keys.(0) ~low:(-1_000_000) ~high:1_000_000 [| n |])
  in
  let price = uniform 1 and null = uniform 2 in
  let b = Buffer.create (1 lsl 21) in
  for r = 0 to n - 1 do
    Buffer.add_string b (Printf.sprintf "{\"id\":%ld,\"price\":" id.(r));
    if null.(r) >= 0.05 then
      Buffer.add_string b (Printf.sprintf "%.2f" (1000. *. price.(r)))
    else Buffer.add_string b "null";
    Buffer.add_string b
      (if r mod 10 = 0 then Printf.sprintf ",\"label\":\"item \\\"%d\\\"\"}\n" r
       else Printf.sprintf ",\"label\":\"item-%d\"}\n" r)
  done;
  Buffer.contents b

let json_format =
  Talon_json.format
    [ ("id", Type.Any Type.int64); ("price", float64); ("label", string) ]

let read_json () =
  Error.get_ok
    (Talon_json.decode json_format (Bytesrw.Bytes.Reader.of_string json_text))

(* As-of: 2^20 left rows [x] and 2^16 right rows [y] in 100 groups, in no
   order, each left row with the right row of its group at or before it. The
   same picks as a range join, each right row holding up to its group's next
   [y], show the cost of the range mechanism beside the window's. *)
let asof =
  let keys = Nx.Rng.split ~n:4 (Nx.Rng.key 29) in
  let side k rows x =
    Query.of_table
      (Talon.v
         [
           ( "g",
             nullable
               (Nx.cast Nx.int64
                  (Nx.Rng.randint ~key:keys.(k) ~high:100 [| rows |])) );
           (x, nullable (Nx.Rng.uniform ~key:keys.(k + 1) Nx.float64 [| rows |]));
         ])
  in
  (side 0 (1 lsl 20) "x", side 2 (1 lsl 16) "y")

let asof_window () =
  let left, right = asof in
  run (Kit.join_asof ~by:[ "g" ] ~on:("x", "y") right left)

let asof_range () =
  let left, right = asof in
  let y = Col.float "y" in
  let next = Expr.(over ~by:[ "g" ] ~order:[ Order.asc "y" ] (shift (-1) y)) in
  let right =
    Query.derive Expr.[ "y_next" := coalesce [ next; float infinity ] ] right
  in
  run
    Query.(
      left
      |> join ~kind:Left
           ~on:Join.(keys [ "g" ] && ge "x" "y" && lt "x" "y_next")
           right)

(* [window e] is [e] over [nulls]'s 2^20 rows: [f] has 10% nulls and [i] ties
   over a million values, in no order. *)
let window e = run Query.(of_table nulls |> select Expr.[ "w" := e ])
let hash_keys e = window Expr.(hash ~seed:1 e)

(* A rolling deviation of [w] rows over the first 2^16 rows of [nulls]. *)
let rolling_std w () =
  run
    Query.(
      of_table nulls
      |> slice ~offset:0 ~length:65536
      |> select Expr.[ "w" := window ~before:(int w) (std (Col.float "f")) ])

(* Lists: 2^16 rows of 0 to 64 float64 elements, about 32 each, reduced with
   the row's factor [w]. *)
let lists =
  let keys = Nx.Rng.split ~n:3 (Nx.Rng.key 31) in
  let rows = 1 lsl 16 in
  let lengths =
    Nx.cast Nx.int64 (Nx.Rng.randint ~key:keys.(0) ~high:65 [| rows |])
  in
  let offsets =
    Nx.concatenate ~axis:0 [ Nx.zeros Nx.int64 [| 1 |]; Nx.cumsum lengths ]
  in
  let child =
    Column.of_tensor
      (Nx.Rng.uniform ~key:keys.(1) Nx.float64
         [| Int64.to_int (Nx.item [ rows ] offsets) |])
  in
  let layout = Column.Varsize { validity = None; offsets; child } in
  let xs =
    Result.get_ok (Column.of_layout (Any (Type.list Type.float64)) layout)
  in
  Talon.v
    [
      ("xs", xs);
      ( "w",
        Column.of_tensor (Nx.Rng.uniform ~key:keys.(2) Nx.float64 [| rows |]) );
    ]

let per_list_sum () =
  let xs = Col.v Kind.(list float) "xs"
  and x = Col.float "x"
  and w = Col.float "w" in
  run
    Query.(
      of_table lists
      |> select Expr.[ "s" := per_list [ "x" := xs ] (sum (x *. w)) ])

let unnest_keep () =
  run
    Query.(
      of_table lists |> unnest Expr.[ keep (Sel.names [ "xs" ]) ])

let () =
  Thumper.run "talon"
    [
      Thumper.group "Talon"
        [
          Thumper.bench "Filter/high_value" (fun () -> total (filtered ()));
          Thumper.bench "Group/category_region" (fun () -> total (grouped ()));
          Thumper.bench "Join/customer_lookup" (fun () -> total (joined ()));
          Thumper.bench "Join/repeated_keys" (fun () ->
              total (joined_twice ()));
          Thumper.bench "Join/few_of_repeated_keys" (fun () ->
              total (few_customers ()));
          Thumper.bench "Sort/amount_desc" (fun () -> first_amount (sorted ()));
          Thumper.bench "Nulls/filter-1pct" (fun () -> filter 10);
          Thumper.bench "Nulls/filter-50pct" (fun () -> filter 500);
          Thumper.bench "Nulls/filter-90pct" (fun () -> filter 900);
          Thumper.bench "Nulls/kleene-and" (fun () ->
              kleene Expr.(Col.bool "a" && Col.bool "b"));
          Thumper.bench "Nulls/kleene-or" (fun () ->
              kleene Expr.(Col.bool "a" || Col.bool "b"));
          Thumper.bench "Nulls/take-random" take_random;
          Thumper.bench "Nulls/take-random-text" take_text;
          Thumper.bench_with_setup ~setup:nested "Nulls/take-random-nested"
            take_nested;
          Thumper.bench "Nulls/concat-3" concat_3;
          Thumper.bench_with_setup ~setup:text_layout "Layout/text-1M" (fun l ->
              Column.of_layout (Type.Any Type.string) l);
          Thumper.bench "Nulls/sort" (fun () ->
              run Query.(of_table nulls |> sort [ Order.asc "key" ]));
          Thumper.bench "Nulls/group-sum" (fun () ->
              run
                Query.(
                  of_table nulls
                  |> aggregate ~by:[ "key" ] Expr.[ "f" := sum (Col.float "f") ]));
          Thumper.bench "Nulls/join-left" (fun () -> run joined_nulls);
          Thumper.bench "Join/spikes_in_trials" spikes_in_trials;
          Thumper.bench "Join/keyed_inequality" below;
          Thumper.bench "List/per-list-sum-64k-32" per_list_sum;
          Thumper.bench "List/unnest-keep-64k-32" unnest_keep;
          Thumper.bench "Join/asof-1M-64k" asof_window;
          Thumper.bench "Join/asof-as-range-1M-64k" asof_range;
          Thumper.bench "Group/text-100-1M" (fun () -> by "t");
          Thumper.bench "Group/int32-100-1M" (fun () -> by "k");
          Thumper.bench "Group/minmax-int32-100-1M" (fun () ->
              let v = Col.int "v" in
              over_k Expr.[ "r" := max v - min v ]);
          Thumper.bench "Group/streamed-few-groups" (fun () ->
              streamed streamed_few);
          Thumper.bench "Group/streamed-many-groups" (fun () ->
              streamed streamed_many);
          Thumper.bench "Group/var-few-groups" (fun () ->
              streamed_moments (Lazy.from_val groups));
          Thumper.bench "Group/var-many-groups" (fun () ->
              streamed_moments many_groups);
          Thumper.bench "Group/streamed-var-few-groups" (fun () ->
              streamed_moments streamed_few);
          Thumper.bench "Group/streamed-var-many-groups" (fun () ->
              streamed_moments streamed_many);
          Thumper.bench "Group/mean-std-int32-100-1M" (fun () ->
              let v = Col.int "v" in
              over_k Expr.[ "m" := mean v; "d" := std v ]);
          Thumper.bench "Nulls/filter-text-eq" (fun () ->
              run
                Query.(
                  of_table nulls
                  |> filter Expr.(Col.string "s" = string "s42")));
          Thumper.bench "Nulls/filter-text-lt" (fun () ->
              run
                Query.(
                  of_table nulls
                  |> filter Expr.(Col.string "s" < string "s5")));
          Thumper.bench "Nulls/is_in-3-texts" (fun () ->
              let texts = [ "s1"; "s22"; "s333" ] in
              run
                Query.(
                  of_table nulls
                  |> filter Expr.(is_in texts (Col.string "s"))));
          Thumper.bench "Hash/int-1M" (fun () -> hash_keys (Col.int "i"));
          Thumper.bench "Hash/text-1M" (fun () -> hash_keys (Col.string "s"));
          Thumper.bench "Window/rolling-mean-28-1M" (fun () ->
              window Expr.(window ~before:(int 28) (mean (Col.float "f"))));
          Thumper.bench "Window/rolling-sum-on-1M" (fun () ->
              window
                Expr.(
                  window ~on:(Col.int "i") ~before:(int 1000)
                    (sum (Col.float "f"))));
          Thumper.bench "Window/rolling-std-28-64k" (rolling_std 28);
          Thumper.bench "Window/rolling-std-365-64k" (rolling_std 365);
          Thumper.bench "Window/running-std-1M" (fun () ->
              window Expr.(window (std (Col.float "f"))));
          Thumper.bench "Window/running-sum-1M" (fun () ->
              window Expr.(window (sum (Col.float "f"))));
          Thumper.bench "Window/running-max-1M" (fun () ->
              window Expr.(window (max (Col.float "f"))));
        ];
      Thumper.group "Talon_text"
        [ Thumper.bench "parse-float64-17-digits" parse_floats ];
      Thumper.group "Talon_csv" [ Thumper.bench "read-mixed" read_csv ];
      Thumper.group "Talon_json" [ Thumper.bench "read-ndjson" read_json ];
      Thumper.group "Talon_arrow"
        [
          Thumper.bench "read-nullable" read_arrow;
          Thumper.bench "write-nullable" write_arrow;
        ];
      Thumper.group "Talon_parquet"
        [
          Thumper.bench "read-nullable" read_parquet;
          Thumper.bench "read-text" read_text;
          Thumper.bench "filter-text" filter_text;
          Thumper.bench "read-keys" read_keys;
          Thumper.bench "write-nullable" write_nulls;
          Thumper.bench "write-text" write_text;
        ];
    ]
  |> exit
