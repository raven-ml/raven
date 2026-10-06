(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
module A1 = Bigarray.Array1

type bigbytes = Meta.bigbytes
type int64s = (int64, Bigarray.int64_elt, Bigarray.c_layout) A1.t

type t =
  | Fixed of { valid : Nx.bit_t option; values : Nx.packed }
  | Varsize of {
      valid : Nx.bit_t option;
      offsets : Nx.int64_t;
      data : Nx.uint8_t;
    }

external hybrid :
  bigbytes ->
  int ->
  int ->
  int ->
  int ->
  ('a, 'b, Bigarray.c_layout) A1.t ->
  int ->
  int ->
  int = "talon_parquet_hybrid_byte" "talon_parquet_hybrid"
[@@noalloc]

external hybrid_bits :
  bigbytes -> int -> int -> int -> int -> bigbytes -> int -> int
  = "talon_parquet_hybrid_bits_byte" "talon_parquet_hybrid_bits"
[@@noalloc]

external plain_bits : bigbytes -> int -> int -> int -> bigbytes -> int -> int
  = "talon_parquet_plain_bits_byte" "talon_parquet_plain_bits"
[@@noalloc]

external count_bits : bigbytes -> int -> int -> int = "talon_parquet_count_bits"
[@@noalloc]

external spread_fixed : bigbytes -> bigbytes -> int -> int -> bigbytes -> int
  = "talon_parquet_spread_fixed"
[@@noalloc]

external spread_offsets : bigbytes -> int64s -> int -> int64s -> int
  = "talon_parquet_spread_offsets"
[@@noalloc]

external delta_binary_packed :
  bigbytes ->
  int ->
  int ->
  int ->
  ('a, 'b, Bigarray.c_layout) A1.t ->
  int ->
  int ->
  int
  = "talon_parquet_delta_binary_packed_byte" "talon_parquet_delta_binary_packed"
[@@noalloc]

external assemble :
  bigbytes ->
  int ->
  int ->
  int ->
  int64s ->
  int64s ->
  int64s ->
  int ->
  bigbytes ->
  int = "talon_parquet_assemble_byte" "talon_parquet_assemble"
[@@noalloc]

external plain_byte_array :
  bigbytes -> int -> int -> int -> int64s -> int -> bigbytes -> int
  = "talon_parquet_plain_byte_array_byte" "talon_parquet_plain_byte_array"
[@@noalloc]

external gather_byte_arrays :
  int64s -> bigbytes -> int64s -> int64s -> bigbytes -> int
  = "talon_parquet_gather_byte_arrays"
[@@noalloc]

let bytes n = A1.create Bigarray.int8_unsigned Bigarray.c_layout n
let int64s n = A1.create Bigarray.int64 Bigarray.c_layout n
let tensor a = Nx.of_bigarray (Bigarray.genarray_of_array1 a)

(* [zeros_for_bits n] is the bytes of a bit array of [n] elements, all clear,
   which its writers fill bit by bit. *)
let zeros_for_bits n =
  let b = bytes ((n + 7) / 8) in
  A1.fill b 0;
  b

(* [bits b n] is the first [n] elements of the bit array [b]. *)
let bits b n =
  Nx.slice [ Nx.R (0, n) ] (Nx.reshape [| -1 |] (Nx.bitcast Nx.bit (tensor b)))

(* [reading x f] is [f b] for [b] the bytes of [x]'s elements in C order, under
   a read claim. *)
let reading x f =
  let b = Nx.Op.eval (Read { by = "Talon_parquet.source"; x }) in
  Nx_device.Buffer.Claim.read b;
  Fun.protect
    ~finally:(fun () -> Nx_device.Buffer.Claim.release b)
    (fun () -> f (Nx_device.Buffer.bigarray Bigarray.int8_unsigned b))

let type_name (Type.Any t) = Format.asprintf "%a" Type.pp t

let first_page (cm : Meta.column_meta) =
  match cm.dictionary_page with
  | Some d when d < cm.data_page -> d
  | _ -> cm.data_page

let check (m : Meta.file) ~row_group i (l : Leaf.t) =
  let c = m.row_groups.(row_group).chunks.(i) in
  let fail ?bytes fmt = Meta.fail ~row_group ?bytes fmt in
  if c.encrypted then
    fail "column %S is encrypted, which talon does not read" l.name;
  Option.iter
    (fail "column %S is stored in %S, which talon does not read" l.name)
    c.file_path;
  let cm =
    match c.meta with
    | Some cm -> cm
    | None -> fail "column %S has no metadata" l.name
  in
  if cm.path <> [ l.name ] then
    fail "column %S: its chunk is of the column %S" l.name
      (String.concat "." cm.path);
  if cm.physical <> l.physical then
    fail "column %S: its chunk's physical type is not its schema's" l.name;
  let rows = m.row_groups.(row_group).rows in
  if cm.values <> rows then
    fail "column %S holds %d values for %d rows" l.name cm.values rows;
  let first = first_page cm in
  let size = min cm.compressed_size (max_int - first) in
  (* A chunk of no values has no pages to read, and writers leave its offsets at
     zero. *)
  if rows > 0 && (first < 4 || size > m.footer - first) then
    fail
      ~bytes:(first, first + max 0 (size - 1))
      "column %S: its pages overrun the file, whose footer starts at byte %d"
      l.name m.footer;
  match cm.codec with
  | Uncompressed | Snappy | Gzip | Zstd | Lz4_raw -> ()
  | Lzo ->
      fail "column %S is compressed with LZO, which talon does not read" l.name
  | Brotli ->
      fail "column %S is compressed with Brotli, which talon does not read"
        l.name
  | Lz4 ->
      fail
        "column %S is compressed with LZ4 in Hadoop's framing, which talon \
         does not read; LZ4_RAW is read"
        l.name
  | Unknown_codec n ->
      fail
        "column %S is compressed with codec %d, which Parquet does not define"
        l.name n

(* Parquet values

   A table holds the chunk's Parquet values, the dictionary's first: [n] values
   of [width] bytes in [bytes], [n] booleans as a bit array ([width = 1] on a
   boolean leaf), or, for byte arrays ([width = 0]), [n] byte strings in [bytes]
   whose ends are [offsets.{1}] to [offsets.{n}]. Widths and the counts of pages
   are [i32]s, so their products fit an [int]. *)

type table = {
  width : int;
  mutable bytes : bigbytes;
  offsets : int64s;
  mutable n : int;
}

let width (l : Leaf.t) =
  match l.physical with
  | Boolean -> 1
  | Int32 | Float -> 4
  | Int64 | Double -> 8
  | Int96 -> 12
  | Fixed_len_byte_array -> l.length
  | Byte_array -> 0

let table width ~values ~data =
  let offsets = int64s (if width = 0 then values + 1 else 0) in
  if width = 0 then offsets.{0} <- 0L;
  {
    width;
    bytes = bytes (if width = 0 then data else values * width);
    offsets;
    n = 0;
  }

(* [leaf_table l ~values ~data] is the table of [values] values of the leaf [l],
   booleans as a bit array. *)
let leaf_table (l : Leaf.t) ~values ~data =
  match l.physical with
  | Boolean ->
      { width = 1; bytes = zeros_for_bits values; offsets = int64s 0; n = 0 }
  | _ -> table (width l) ~values ~data

let span t i =
  if t.width = 0 then
    let first = Int64.to_int t.offsets.{i} in
    (first, Int64.to_int t.offsets.{i + 1} - first)
  else (i * t.width, t.width)

let string_at t i =
  let first, len = span t i in
  String.init len (fun k -> Char.unsafe_chr t.bytes.{first + k})

(* [ensure t n] makes room for [n] more bytes of byte strings in [t]. *)
let ensure t n =
  let used = Int64.to_int t.offsets.{t.n} in
  if A1.dim t.bytes - used < n then begin
    let b = bytes (max (used + n) (2 * A1.dim t.bytes)) in
    A1.blit (A1.sub t.bytes 0 used) (A1.sub b 0 used);
    t.bytes <- b
  end

(* Storage *)

(* The values of a chunk in the storage of a talon type: tensors of fixed
   widths, or byte strings as [offsets], one more than the values, and
   [data]. *)
type stored =
  | Values of Nx.packed
  | Strings of { offsets : int64s; data : bigbytes }

let le b pos n =
  let v = ref 0L in
  for k = n - 1 downto 0 do
    v := Int64.logor (Int64.shift_left !v 8) (Int64.of_int b.{pos + k})
  done;
  !v

let ns_per (u : Type.unit_) =
  match u with
  | S -> 1_000_000_000L
  | Ms -> 1_000_000L
  | Us -> 1_000L
  | Ns -> 1L

(* [add a b] and [mul a b], or [None] past int64. *)
let add a b =
  let s = Int64.add a b in
  if
    Int64.compare a 0L >= 0 = (Int64.compare b 0L >= 0)
    && Int64.compare s 0L >= 0 <> (Int64.compare a 0L >= 0)
  then None
  else Some s

let mul a b =
  if a <> 0L && Int64.compare (Int64.abs a) (Int64.div Int64.max_int b) > 0 then
    None
  else Some (Int64.mul a b)

(* [int96 ~row_group t ty u] is the int96 timestamps of [t], a Julian day after
   the nanoseconds of that day, as ticks of [u] since 1970-01-01. *)
let int96 ~row_group t ty (u : Type.unit_) =
  let out = int64s t.n in
  let f = ns_per u in
  let per_day = Int64.div 86_400_000_000_000L f in
  for i = 0 to t.n - 1 do
    let nanos = le t.bytes (12 * i) 8 in
    let day =
      Int64.shift_right (Int64.shift_left (le t.bytes ((12 * i) + 8) 4) 32) 32
    in
    if Int64.rem nanos f <> 0L then
      Meta.fail ~row_group
        "an int96 timestamp is not a whole number of %s. Read the column as a \
         datetime of a finer unit (with_type)."
        (match u with
        | S -> "seconds"
        | Ms -> "milliseconds"
        | Us -> "microseconds"
        | Ns -> "nanoseconds");
    match
      Option.bind
        (mul (Int64.sub day 2_440_588L) per_day)
        (add (Int64.div nanos f))
    with
    | Some v -> out.{i} <- v
    | None ->
        Meta.fail ~row_group
          "an int96 timestamp is outside the range of %s. Read the column as a \
           datetime of a coarser unit (with_type)."
          (type_name ty)
  done;
  Values (P (tensor out))

(* Decimals

   A decimal is an integer, unscaled, and a scale: an [int32] or [int64], or a
   big-endian two's complement byte string. It reads as its unscaled [int64]
   integer, or as the [float64] nearest its value, which the float parser rounds
   once from the exact text [unscaled]e-[scale]. *)

(* [unscaled ~row_group t] is the byte strings of [t] as [int64]s. *)
let unscaled ~row_group t =
  let out = int64s t.n in
  for i = 0 to t.n - 1 do
    let first, len = span t i in
    if len = 0 then Meta.fail ~row_group "a decimal value has no bytes";
    let v = ref (Int64.of_int ((t.bytes.{first} lxor 0x80) - 0x80)) in
    for k = 1 to len - 1 do
      if
        Int64.compare !v 0x7F_FFFF_FFFF_FFFFL > 0
        || Int64.compare !v (-0x80_0000_0000_0000L) < 0
      then Meta.fail ~row_group "a decimal value is outside the range of int64";
      v :=
        Int64.logor (Int64.shift_left !v 8) (Int64.of_int t.bytes.{first + k})
    done;
    out.{i} <- !v
  done;
  Values (P (tensor out))

let billion = 1_000_000_000

(* [big_digits t i] is the decimal text of the byte string [i] of [t], a
   big-endian two's complement integer of any length. Its magnitude is divided
   by 10^9 until it is zero, each remainder giving nine digits. *)
let big_digits ~row_group t i =
  let first, len = span t i in
  if len = 0 then Meta.fail ~row_group "a decimal value has no bytes";
  let negative = t.bytes.{first} land 0x80 <> 0 in
  let m =
    Array.init len (fun k ->
        if negative then t.bytes.{first + k} lxor 0xFF else t.bytes.{first + k})
  in
  if negative then begin
    let k = ref (len - 1) in
    while !k >= 0 && m.(!k) = 0xFF do
      m.(!k) <- 0;
      decr k
    done;
    if !k >= 0 then m.(!k) <- m.(!k) + 1
  end;
  let groups = ref [] in
  while Array.exists (fun d -> d <> 0) m do
    let r = ref 0 in
    for k = 0 to len - 1 do
      let x = (!r lsl 8) lor m.(k) in
      m.(k) <- x / billion;
      r := x mod billion
    done;
    groups := !r :: !groups
  done;
  match !groups with
  | [] -> "0"
  | g :: gs ->
      String.concat ""
        ((if negative then "-" else "")
        :: string_of_int g
        :: List.map (Printf.sprintf "%09d") gs)

(* [floats ~row_group (l : Leaf.t) scale t] is the decimals of [t], of scale
   [scale], as the nearest [float64]s. *)
let floats ~row_group (l : Leaf.t) scale t =
  let digits i =
    match l.physical with
    | Int32 -> Int32.to_string (Int64.to_int32 (le t.bytes (4 * i) 4))
    | Int64 -> Int64.to_string (le t.bytes (8 * i) 8)
    | _ -> big_digits ~row_group t i
  in
  let text i = Printf.sprintf "%se-%d" (digits i) scale in
  let texts = Column.v Type.string (Array.init t.n text) in
  match Column.parse (Any Type.float64) texts with
  | Ok c -> Values (P (Column.to_tensor Nx.float64 c))
  | Error (_, why) -> Meta.fail ~row_group "a decimal value %s" why

let categorical ~row_group t dict =
  let codes = Hashtbl.create (Iarray.length dict) in
  Iarray.iteri (fun i s -> Hashtbl.replace codes s (Int32.of_int i)) dict;
  let out = A1.create Bigarray.int32 Bigarray.c_layout t.n in
  for i = 0 to t.n - 1 do
    let s = string_at t i in
    match Hashtbl.find_opt codes s with
    | Some c -> out.{i} <- c
    | None ->
        Meta.fail ~row_group ~text:s
          "the value is not in the categorical's dictionary"
  done;
  Values (P (tensor out))

let strings t =
  if t.width = 0 then
    Strings
      {
        offsets = A1.sub t.offsets 0 (t.n + 1);
        data = A1.sub t.bytes 0 (Int64.to_int t.offsets.{t.n});
      }
  else
    Strings
      {
        offsets =
          A1.init Bigarray.int64 Bigarray.c_layout (t.n + 1) (fun i ->
              Int64.of_int (i * t.width));
        data = A1.sub t.bytes 0 (t.n * t.width);
      }

(* [fixed t dt] is the values of [t] read as [dt], whose width is [t]'s. *)
let fixed t dt =
  tensor (A1.sub t.bytes 0 (t.n * t.width))
  |> Nx.reshape [| t.n; t.width |]
  |> Nx.bitcast dt

let convert (type a) ~row_group (l : Leaf.t) (ty : a Type.t) t =
  let any = Type.Any ty in
  let values dt = Values (P (fixed t dt)) in
  match (l.physical, l.annotation, ty) with
  | _, Some (Decimal { scale; _ }), Float64 -> floats ~row_group l scale t
  | (Byte_array | Fixed_len_byte_array), _, Int64 -> unscaled ~row_group t
  | Boolean, _, _ -> Values (P (bits t.bytes t.n))
  | Int32, _, Uint32 -> values Nx.uint32
  | Int32, _, (Clock _ | Int64) ->
      Values (P (Nx.cast Nx.int64 (fixed t Nx.int32)))
  | Int32, _, _ -> values Nx.int32
  | Int64, _, Uint64 -> values Nx.uint64
  | Int64, _, _ -> values Nx.int64
  | Float, _, _ -> values Nx.float32
  | Double, _, _ -> values Nx.float64
  | Int96, _, Datetime { unit_; _ } -> int96 ~row_group t any unit_
  | Fixed_len_byte_array, _, Float16 -> values Nx.float16
  | _, _, Categorical dict -> categorical ~row_group t dict
  | _ -> strings t

(* [gather s idx] is the values of [s] at the positions [idx]. *)
let gather s (idx : int64s) =
  match s with
  | Values (P v) -> Values (P (Nx.take ~indices:(tensor idx) v))
  | Strings { offsets; data } ->
      let m = A1.dim idx in
      let out = int64s (m + 1) in
      out.{0} <- 0L;
      for i = 0 to m - 1 do
        let j = Int64.to_int idx.{i} in
        out.{i + 1} <- Int64.add out.{i} (Int64.sub offsets.{j + 1} offsets.{j})
      done;
      let d = bytes (Int64.to_int out.{m}) in
      (* The indices were checked against the dictionary as they decoded. *)
      let r = gather_byte_arrays offsets data idx out d in
      assert (r = 0);
      Strings { offsets = out; data = d }

let kernel_error = function
  | -1 -> "the data ends inside a value"
  | -2 -> "a value is out of its bounds"
  | -3 -> "a bit width is wider than its values"
  | -4 -> "a header is malformed"
  | -5 -> "a header counts other values than the page holds"
  | _ -> "an output is too small"

(* [spread_values v ~valid ~rows] is the values [v] at the rows the bit array
   [valid] marks, zero under the others, and the number of values it took. *)
let spread_values (type a b) (v : (a, b) Nx.t) ~valid ~rows =
  let dt = Nx.dtype v in
  let s = Nx_dtype.Scalar.of_dtype dt in
  let out = Nx_device.Buffer.create Nx_device.host s rows in
  let dst = Nx_device.Buffer.bigarray Bigarray.int8_unsigned out in
  let taken =
    reading v (fun src ->
        spread_fixed valid src (Nx_dtype.Scalar.bitsize s) rows dst)
  in
  (Nx.of_buffer dt [| rows |] out, taken)

(* [spread s ~valid ~rows ~present] puts the [present] values of [s] at the rows
   the bit array [valid] marks, zero under the others. *)
let spread ~row_group s ~valid ~rows ~present =
  if present = rows then
    match s with
    | Values values -> Fixed { valid = None; values }
    | Strings { offsets; data } ->
        Varsize { valid = None; offsets = tensor offsets; data = tensor data }
  else
    let mask = bits valid rows in
    let took n =
      if n < 0 then Meta.fail ~row_group "the spread: %s" (kernel_error n);
      assert (n = present)
    in
    match s with
    | Values (P v) ->
        let values, n = spread_values v ~valid ~rows in
        took n;
        Fixed { valid = Some mask; values = P values }
    | Strings { offsets; data } ->
        let o = int64s (rows + 1) in
        took (spread_offsets valid offsets rows o);
        Varsize { valid = Some mask; offsets = tensor o; data = tensor data }

(* [narrow ~row_group ty c] is [c], whose values are int32s, in the storage of
   [ty] when [ty] is an integer type narrower than int32. *)
let narrow (type a) ~row_group (ty : a Type.t) c =
  let into dt lo hi =
    match c with
    | Varsize _ -> assert false
    | Fixed { valid; values = P v } ->
        let v = Nx.cast Nx.int32 v in
        let bad = Nx.logical_or (Nx.less_s v lo) (Nx.greater_s v hi) in
        let at = Nx.positions bad in
        if Nx.numel at > 0 then begin
          let row = Int64.to_int (Nx.item [ 0 ] at) in
          Meta.fail ~row_group "the value %ld of row %d does not fit %s"
            (Nx.item [ row ] v) row (type_name (Type.Any ty))
        end;
        Fixed { valid; values = P (Nx.cast dt v) }
  in
  match ty with
  | Int8 -> into Nx.int8 (-128l) 127l
  | Int16 -> into Nx.int16 (-32768l) 32767l
  | Uint8 -> into Nx.uint8 0l 255l
  | Uint16 -> into Nx.uint16 0l 65535l
  | _ -> c

(* Reading *)

let encoding_name : Meta.encoding -> string = function
  | Plain -> "PLAIN"
  | Plain_dictionary -> "PLAIN_DICTIONARY"
  | Rle -> "RLE"
  | Bit_packed -> "BIT_PACKED"
  | Delta_binary_packed -> "DELTA_BINARY_PACKED"
  | Delta_length_byte_array -> "DELTA_LENGTH_BYTE_ARRAY"
  | Delta_byte_array -> "DELTA_BYTE_ARRAY"
  | Rle_dictionary -> "RLE_DICTIONARY"
  | Byte_stream_split -> "BYTE_STREAM_SPLIT"
  | Unknown_encoding n -> Printf.sprintf "encoding %d" n

let u32 b pos = Int64.to_int (le b pos 4)

let read b (m : Meta.file) ~row_group i (l : Leaf.t) (Type.Any ty) =
  let rows = m.row_groups.(row_group).rows in
  let cm = Option.get m.row_groups.(row_group).chunks.(i).meta in
  let here = ref None in
  let fail fmt = Meta.fail ~row_group ?bytes:!here fmt in
  let exactly what used len =
    if used <> len then fail "%s take %d bytes of the page's %d" what used len
  in
  let kernel what r =
    if r < 0 then fail "%s: %s" what (kernel_error r) else r
  in
  (* [consumes what len r] checks that a kernel that returned [r] decoded [what]
     from exactly [len] bytes. *)
  let consumes what len r = exactly what (kernel what r) len in
  let valid = zeros_for_bits (if l.optional then rows else 0) in
  let tbl = ref None and idx = ref None and dict = ref 0 in
  let row = ref 0 and present = ref 0 in
  (* [chunk_table ~dictionary] is the chunk's table, made on the first page for
     the chunk's values after [dictionary] values. *)
  let chunk_table ~dictionary =
    match !tbl with
    | Some t -> t
    | None ->
        let w = width l in
        if cm.values > (max_int / max 1 w) - dictionary - 1 then
          fail "the chunk's %d values do not fit in memory" cm.values;
        let values = dictionary + cm.values in
        let t = leaf_table l ~values ~data:cm.uncompressed_size in
        tbl := Some t;
        t
  in
  let scratch = ref (bytes 0) in
  let decompress src pos len n : bigbytes * int * int =
    if cm.codec = Uncompressed then (src, pos, len)
    else if n = 0 then (src, pos, 0)
    else begin
      if n > cm.uncompressed_size then
        fail "the page decompresses to %d bytes, more than its chunk's %d" n
          cm.uncompressed_size;
      if A1.dim !scratch < n then scratch := bytes n;
      let dst = A1.sub !scratch 0 n in
      let src = A1.sub src pos len in
      let r =
        match cm.codec with
        | Snappy -> Compress_snappy.decompress_into src dst
        | Gzip -> Compress_deflate.Gzip.decompress_into src dst
        | Zstd -> Compress_zstd.decompress_into src dst
        | Lz4_raw -> Compress_lz4.Block.decompress_into src dst
        | _ -> invalid_arg "Chunk.read: a chunk that did not pass Chunk.check"
      in
      match r with
      | Ok () -> (dst, 0, n)
      | Error e -> fail "the page does not decompress: %s" e
    end
  in
  let levels src pos len values =
    consumes "the definition levels" len
      (hybrid_bits src pos len 1 values valid !row);
    kernel "the definition levels" (count_bits valid !row values)
  in
  let plain t src pos len n =
    match l.physical with
    | Boolean ->
        consumes "the PLAIN booleans" len (plain_bits src pos len n t.bytes t.n)
    | Byte_array ->
        ensure t len;
        consumes "the PLAIN byte arrays" len
          (plain_byte_array src pos len n t.offsets t.n t.bytes)
    | _ ->
        exactly "the PLAIN values" (n * t.width) len;
        A1.blit (A1.sub src pos len) (A1.sub t.bytes (t.n * t.width) len)
  in
  let delta_strings ~prefixed t src pos len n =
    let lengths p =
      let a = int64s n in
      ( a,
        kernel "the lengths"
          (delta_binary_packed src p (len - (p - pos)) n a 0 8) )
    in
    let prefixes, c1 =
      if prefixed then lengths pos
      else
        let z = int64s n in
        A1.fill z 0L;
        (z, 0)
    in
    let suffixes, c2 = lengths (pos + c1) in
    let start = pos + c1 + c2 in
    let rec go () =
      match
        assemble src start
          (len - c1 - c2)
          n prefixes suffixes t.offsets t.n t.bytes
      with
      | -6 (* the data is too small *) ->
          ensure t (1 + A1.dim t.bytes);
          go ()
      | r -> c1 + c2 + kernel "the byte arrays" r
    in
    exactly "the byte arrays" (go ()) len
  in
  let values t encoding src pos len n =
    match ((encoding : Meta.encoding), l.physical) with
    | Plain, _ -> plain t src pos len n
    | Rle, Boolean ->
        if len < 4 || u32 src pos <> len - 4 then
          fail "the RLE booleans do not fill their page";
        consumes "the RLE booleans" (len - 4)
          (hybrid_bits src (pos + 4) (len - 4) 1 n t.bytes t.n)
    | Delta_binary_packed, (Int32 | Int64) ->
        consumes "the DELTA_BINARY_PACKED values" len
          (delta_binary_packed src pos len n t.bytes t.n t.width)
    | Delta_length_byte_array, Byte_array ->
        delta_strings ~prefixed:false t src pos len n
    | Delta_byte_array, Byte_array ->
        delta_strings ~prefixed:true t src pos len n
    | Delta_byte_array, Fixed_len_byte_array ->
        let strings = table 0 ~values:n ~data:(n * t.width) in
        delta_strings ~prefixed:true strings src pos len n;
        for k = 1 to n do
          if Int64.to_int strings.offsets.{k} <> k * t.width then
            fail "a DELTA_BYTE_ARRAY value is not %d bytes long" t.width
        done;
        A1.blit
          (A1.sub strings.bytes 0 (n * t.width))
          (A1.sub t.bytes (t.n * t.width) (n * t.width))
    | Byte_stream_split, (Float | Double | Int32 | Int64 | Fixed_len_byte_array)
      ->
        exactly "the BYTE_STREAM_SPLIT values" (n * t.width) len;
        for k = 0 to n - 1 do
          for j = 0 to t.width - 1 do
            t.bytes.{((t.n + k) * t.width) + j} <- src.{pos + (j * n) + k}
          done
        done
    | Unknown_encoding n, _ ->
        fail "the values are in encoding %d, which Parquet does not define" n
    | e, _ ->
        fail
          "the values are in %s, which talon does not read on this column's \
           physical type"
          (encoding_name e)
  in
  (* [data_page encoding src pos len ~rows n] decodes a data page of [rows]
     rows, [n] of which hold the values that [len] bytes at [pos] encode. *)
  let data_page encoding src pos len ~rows n =
    let t = chunk_table ~dictionary:0 in
    (match ((encoding : Meta.encoding), !idx) with
    | (Plain_dictionary | Rle_dictionary), None ->
        fail "a dictionary-encoded page has no dictionary page before it"
    | (Plain_dictionary | Rle_dictionary), Some idx ->
        if len < 1 then fail "the dictionary indices have no bit width";
        consumes "the dictionary indices" (len - 1)
          (hybrid src (pos + 1) (len - 1) src.{pos} n idx !present !dict)
    | _, idx ->
        let first = t.n in
        values t encoding src pos len n;
        t.n <- t.n + n;
        Option.iter
          (fun idx ->
            for k = 0 to n - 1 do
              idx.{!present + k} <- Int64.of_int (first + k)
            done)
          idx);
    row := !row + rows;
    present := !present + n
  in
  let rows_left values =
    if values > rows - !row then
      fail "the page holds %d rows, past the row group's %d" values rows
  in
  let rec pages pos stop =
    if pos < stop then begin
      here := Some (pos, stop - 1);
      let h, data =
        try Meta.page_header b ~pos ~limit:stop
        with Thrift.Error (p, msg) ->
          Meta.fail ~row_group ~bytes:(p, p) "a page header is malformed: %s"
            msg
      in
      if h.compressed_size > stop - data then
        fail "the page overruns its column chunk";
      let page_end = data + h.compressed_size in
      here := Some (pos, page_end - 1);
      Option.iter
        (fun crc ->
          if
            Compress_deflate.Crc32.bigbytes (A1.sub b data h.compressed_size)
            <> crc
          then fail "the page fails its CRC-32 checksum")
        h.crc;
      (match h.page with
      | Dictionary { values; encoding } ->
          if Option.is_some !tbl then
            fail "a dictionary page is not the chunk's first page";
          if encoding <> Plain && encoding <> Plain_dictionary then
            fail "the dictionary page is in %s, which talon does not read"
              (encoding_name encoding);
          let src, p, len =
            decompress b data h.compressed_size h.uncompressed_size
          in
          if values > 8 * len then
            fail "the dictionary page's %d values overrun it" values;
          let t = chunk_table ~dictionary:values in
          plain t src p len values;
          t.n <- values;
          dict := values;
          idx := Some (int64s rows)
      | Data { values; encoding; levels = encoding_levels } ->
          rows_left values;
          let src, p, len =
            decompress b data h.compressed_size h.uncompressed_size
          in
          let p, len, n =
            if not l.optional then (p, len, values)
            else begin
              if encoding_levels <> Rle then
                fail "definition levels in %s are not read"
                  (encoding_name encoding_levels);
              if len < 4 then
                fail "the definition levels' length overruns the page";
              let n = u32 src p in
              if n > len - 4 then fail "the definition levels overrun the page";
              (p + 4 + n, len - 4 - n, levels src (p + 4) n values)
            end
          in
          data_page encoding src p len ~rows:values n
      | Data_v2
          {
            values;
            nulls;
            rows = page_rows;
            encoding;
            definition_bytes;
            repetition_bytes;
            compressed;
          } ->
          rows_left values;
          if page_rows <> values then
            fail "the page holds %d values in %d rows" values page_rows;
          let size = min h.compressed_size h.uncompressed_size in
          if
            repetition_bytes > size
            || definition_bytes > size - repetition_bytes
          then fail "the levels overrun the page";
          let levels_bytes = repetition_bytes + definition_bytes in
          if (not l.optional) && definition_bytes <> 0 then
            fail
              "the page holds definition levels, which a required column has \
               none of";
          (* A flat column's repetition levels are all zero: some writers write
             them, and they are skipped. *)
          let n =
            if l.optional then
              levels b (data + repetition_bytes) definition_bytes values
            else values
          in
          if nulls <> values - n then
            fail "the page counts %d nulls, and its levels %d" nulls (values - n);
          let pos = data + levels_bytes in
          let len = h.compressed_size - levels_bytes in
          let src, p, len =
            if compressed then
              decompress b pos len (h.uncompressed_size - levels_bytes)
            else (b, pos, len)
          in
          data_page encoding src p len ~rows:values n
      | Other -> ());
      pages page_end stop
    end
  in
  let first = first_page cm in
  if rows > 0 then pages first (first + cm.compressed_size);
  here := None;
  if !row <> rows then
    fail "the pages hold %d rows, and the row group %d" !row rows;
  let s = convert ~row_group l ty (chunk_table ~dictionary:0) in
  let s =
    match !idx with None -> s | Some idx -> gather s (A1.sub idx 0 !present)
  in
  narrow ~row_group ty (spread ~row_group s ~valid ~rows ~present:!present)
