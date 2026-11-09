(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
module A1 = Bigarray.Array1

type bigbytes = Meta.bigbytes
type int64s = (int64, Bigarray.int64_elt, Bigarray.c_layout) A1.t

type t =
  | Fixed of { valid : Nx.host Nx.bit_t option; values : Nx.host Nx.packed }
  | Varsize of {
      valid : Nx.host Nx.bit_t option;
      offsets : Nx.host Nx.int64_t;
      data : Nx.host Nx.uint8_t;
    }
  | Coded of {
      valid : Nx.host Nx.bit_t option;
      codes : Nx.host Nx.int64_t;
      offsets : Nx.host Nx.int64_t;
      data : Nx.host Nx.uint8_t;
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

external level_count :
  ('a, 'b, Bigarray.c_layout) A1.t -> int -> int -> int -> int
  = "talon_parquet_level_count"
[@@noalloc]

external level_walk :
  ('a, 'b, Bigarray.c_layout) A1.t ->
  ('a, 'b, Bigarray.c_layout) A1.t ->
  int ->
  int ->
  int ->
  int ->
  int ->
  int ->
  bigbytes ->
  int64s ->
  int = "talon_parquet_level_walk_byte" "talon_parquet_level_walk"
[@@noalloc]

external gather_ends : int64s -> int64s -> int64s -> int
  = "talon_parquet_gather_ends"
[@@noalloc]

external gather_byte_arrays :
  int64s -> bigbytes -> int64s -> int64s -> bigbytes -> int
  = "talon_parquet_gather_byte_arrays"
[@@noalloc]

(* Host arrays

   The arrays that become a column's values are host arrays of nx.array
   ({!Nx_array.host}): rig's host memory, which paces the collector by the
   program's memory and is reused once collected. The decoder fills them under
   write claims, and they become values with no copy. What no value holds,
   the pages it decompresses and the lengths of delta-encoded byte strings,
   lies in bigarrays a read reuses across its chunks ({!type-scratch}). *)

let u8 = Bigarray.int8_unsigned
let i64 = Bigarray.int64
let bytes n = A1.create u8 Bigarray.c_layout n
let int64s n = A1.create i64 Bigarray.c_layout n
let no_bytes = bytes 0
let no_ints = int64s 0

(* [fresh dt n] is a host array of [n] elements of [dt]. *)
let fresh dt n = Nx_array.host dt [| n |]

(* [writing a f] and [peek a f] are [f b] for [b] the elements of the host
   vector [a], under a write or a read claim. *)
let writing a f = Nx_array.write a (fun g -> f (Bigarray.array1_of_genarray g))
let peek a f = Nx_array.read a (fun g -> f (Bigarray.array1_of_genarray g))

(* [value a] is the host value over [a]'s memory, which the decoder filled and
   gives up. *)
let value a = Nx.Repr.unsafe_of_array Nx.Host.v a

(* [zeros_for_bits n] is the bytes of a bit array of [n] elements, all clear,
   which its writers fill bit by bit. *)
let zeros_for_bits n =
  let a = fresh Nx.uint8 ((n + 7) / 8) in
  writing a (fun b -> A1.fill b 0);
  a

(* [retype dt n x] is the first [n] elements of [dt] whose bytes are those of
   the [uint8] vector [x]. *)
let retype dt n x =
  let w = Nx.Dtype.bits dt in
  if w = 8 then Nx.bitcast dt x
  else if w > 8 then Nx.bitcast dt (Nx.reshape [| n; w / 8 |] x)
  else Nx.slice [ Nx.R (0, n) ] (Nx.reshape [| -1 |] (Nx.bitcast dt x))

(* [bits a n] is the first [n] elements of the bit array [a]. *)
let bits a n = retype Nx.bit n (value a)

(* [reading k x f] is [f b] for [b] [x]'s elements in C order as [k], whose
   width divides theirs, under a read claim: a view of its buffer where they
   lie there in order from a byte, a copy's otherwise. *)
let reading k x f =
  Nx.Repr.read x @@ fun a ->
  let l = Nx_array.layout a and dt = Nx_array.dtype a in
  let n = Nx_array.Dtype.bytes dt (Nx_array.Layout.numel l) in
  let first = Nx_array.Layout.offset l * Nx_array.Dtype.bits dt in
  let a, first =
    if Nx_array.Layout.is_contiguous l && first land 7 = 0 then (a, first / 8)
    else (Nx_array.copy a, 0)
  in
  let w = Bigarray.kind_size_in_bytes k in
  Rig.Buffer.read (Nx_array.buffer a) k (fun b ->
      f (A1.sub b (first / w) (n / w)))

(* Scratch

   The bigarrays a read reuses across chunks: the page it decompresses, and
   the prefix and suffix lengths of delta-encoded byte strings. Each grows to
   the largest that a chunk asks for. *)

type scratch = {
  mutable page : bigbytes;
  mutable prefixes : int64s;
  mutable suffixes : int64s;
}

let scratch () = { page = bytes 0; prefixes = int64s 0; suffixes = int64s 0 }

let type_name (Type.Any t) = Format.asprintf "%a" Type.pp t

let first_page (cm : Meta.column_meta) =
  match cm.dictionary_page with
  | Some d when d < cm.data_page -> d
  | _ -> cm.data_page

let check (m : Meta.file) ~row_group i (l : Leaf.t) =
  let c = m.row_groups.(row_group).chunks.(i) in
  let fail ?bytes fmt = Meta.fail ~row_group ?bytes fmt in
  if c.encrypted then
    fail "column %S is encrypted, which talon does not read" (Leaf.name l);
  (match c.file_path with
  | Some f ->
      fail "column %S is stored in %S, which talon does not read"
        (Leaf.name l) f
  | None -> ());
  let cm =
    match c.meta with
    | Some cm -> cm
    | None -> fail "column %S has no metadata" (Leaf.name l)
  in
  if not (List.equal String.equal (List.rev cm.path) l.rev_path) then
    fail "column %S: its chunk is of the column %S" (Leaf.name l)
      (String.concat "." cm.path);
  if cm.physical <> l.physical then
    fail "column %S: its chunk's physical type is not its schema's"
      (Leaf.name l);
  let rows = m.row_groups.(row_group).rows in
  (* Only a repeated column holds more entries than rows. *)
  if rows > Nx_array.Layout.max_numel || cm.values > Nx_array.Layout.max_numel
  then
    fail "column %S holds %d values for %d rows, past what a layout holds"
      (Leaf.name l) cm.values rows;
  if (if l.rep = 0 then cm.values <> rows else cm.values < rows) then
    fail "column %S holds %d values for %d rows" (Leaf.name l) cm.values rows;
  if l.rep > 0 && rows = 0 && cm.values <> 0 then
    fail "column %S holds %d values for no rows" (Leaf.name l) cm.values;
  let first = first_page cm in
  let size = min cm.compressed_size (max_int - first) in
  (* A chunk of no values has no pages to read, and writers leave its offsets at
     zero. *)
  if rows > 0 && (first < 4 || size > m.footer - first) then
    fail
      ~bytes:(first, first + max 0 (size - 1))
      "column %S: its pages overrun the file, whose footer starts at byte %d"
      (Leaf.name l) m.footer;
  match cm.codec with
  | Uncompressed | Snappy | Gzip | Zstd | Lz4_raw -> ()
  | Lzo ->
      fail "column %S is compressed with LZO, which talon does not read"
        (Leaf.name l)
  | Brotli ->
      fail "column %S is compressed with Brotli, which talon does not read"
        (Leaf.name l)
  | Lz4 ->
      fail
        "column %S is compressed with LZ4 in Hadoop's framing, which talon \
         does not read; LZ4_RAW is read"
        (Leaf.name l)
  | Unknown_codec n ->
      fail
        "column %S is compressed with codec %d, which Parquet does not define"
        (Leaf.name l) n

(* Parquet values

   A table holds the chunk's Parquet values, the dictionary's first: [n] values
   of [width] bytes in [bytes], [n] booleans as a bit array ([width = 1] on a
   boolean leaf), or, for byte arrays ([width = 0]), [n] byte strings in [bytes]
   whose ends are [offsets.{1}] to [offsets.{n}]. Widths and the counts of pages
   are [i32]s, so their products fit an [int]. *)

type table = {
  width : int;
  mutable bytes : (int, Bigarray.int8_unsigned_elt) Nx_array.t;
  offsets : (int64, Bigarray.int64_elt) Nx_array.t;
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

(* The offsets of every table of fixed-width values, which has none: an
   array of no element, which no claim holds exclusive. *)
let no_offsets = fresh Nx.int64 0

let table width ~values ~data =
  let offsets =
    if width > 0 then no_offsets
    else
      let a = fresh Nx.int64 (values + 1) in
      writing a (fun o -> o.{0} <- 0L);
      a
  in
  {
    width;
    bytes = fresh Nx.uint8 (if width = 0 then data else values * width);
    offsets;
    n = 0;
  }

(* [leaf_table l ~values ~data] is the table of [values] values of the leaf [l],
   booleans as a bit array. *)
let leaf_table (l : Leaf.t) ~values ~data =
  match l.physical with
  | Boolean ->
      {
        width = 1;
        bytes = zeros_for_bits values;
        offsets = no_offsets;
        n = 0;
      }
  | _ -> table (width l) ~values ~data

(* [filling t f] is [f b o] for [b] and [o] the bytes and offsets of [t],
   under write claims, and [reading_table t f] the same under read claims. *)
let filling t f = writing t.bytes (fun b -> writing t.offsets (f b))
let reading_table t f = peek t.bytes (fun b -> peek t.offsets (f b))

(* [span t o i] is the first byte and the length of the value [i] of [t],
   whose offsets are [o]. *)
let span t o i =
  if t.width = 0 then
    let first = Int64.to_int o.{i} in
    (first, Int64.to_int o.{i + 1} - first)
  else (i * t.width, t.width)

let string_at t b o i =
  let first, len = span t o i in
  String.init len (fun k -> Char.unsafe_chr b.{first + k})

let length a = Nx_array.Layout.numel (Nx_array.layout a)

(* [ensure t n] makes room for [n] more bytes of byte strings in [t]. *)
let ensure t n =
  let used = peek t.offsets (fun o -> Int64.to_int o.{t.n}) in
  let size = length t.bytes in
  if size - used < n then begin
    let a = fresh Nx.uint8 (max (used + n) (2 * size)) in
    writing a (fun dst ->
        peek t.bytes (fun src ->
            A1.blit (A1.sub src 0 used) (A1.sub dst 0 used)));
    t.bytes <- a
  end

(* Storage *)

(* The values of a chunk in the storage of a talon type: tensors of fixed
   widths, or byte strings as [offsets], one more than the values, and
   [data]. *)
type stored =
  | Values of Nx.host Nx.packed
  | Strings of { offsets : Nx.host Nx.int64_t; data : Nx.host Nx.uint8_t }

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
  let a = fresh Nx.int64 t.n in
  let f = ns_per u in
  let per_day = Int64.div 86_400_000_000_000L f in
  peek t.bytes @@ fun b ->
  writing a @@ fun out ->
  for i = 0 to t.n - 1 do
    let nanos = le b (12 * i) 8 in
    let day =
      Int64.shift_right (Int64.shift_left (le b ((12 * i) + 8) 4) 32) 32
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
  Values (P (value a))

(* Decimals

   A decimal is an integer, unscaled, and a scale: an [int32] or [int64], or a
   big-endian two's complement byte string. It reads as its unscaled [int64]
   integer, or as the [float64] nearest its value, which the float parser rounds
   once from the exact text [unscaled]e-[scale]. *)

(* [unscaled ~row_group t] is the byte strings of [t] as [int64]s. *)
let unscaled ~row_group t =
  let a = fresh Nx.int64 t.n in
  reading_table t @@ fun b o ->
  writing a @@ fun out ->
  for i = 0 to t.n - 1 do
    let first, len = span t o i in
    if len = 0 then Meta.fail ~row_group "a decimal value has no bytes";
    let v = ref (Int64.of_int ((b.{first} lxor 0x80) - 0x80)) in
    for k = 1 to len - 1 do
      if
        Int64.compare !v 0x7F_FFFF_FFFF_FFFFL > 0
        || Int64.compare !v (-0x80_0000_0000_0000L) < 0
      then Meta.fail ~row_group "a decimal value is outside the range of int64";
      v := Int64.logor (Int64.shift_left !v 8) (Int64.of_int b.{first + k})
    done;
    out.{i} <- !v
  done;
  Values (P (value a))

let billion = 1_000_000_000

(* [big_digits t b o i] is the decimal text of the byte string [i] of [t], a
   big-endian two's complement integer of any length. Its magnitude is divided
   by 10^9 until it is zero, each remainder giving nine digits. *)
let big_digits ~row_group t b o i =
  let first, len = span t o i in
  if len = 0 then Meta.fail ~row_group "a decimal value has no bytes";
  let negative = b.{first} land 0x80 <> 0 in
  let m =
    Array.init len (fun k ->
        if negative then b.{first + k} lxor 0xFF else b.{first + k})
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
  let digits b o i =
    match l.physical with
    | Int32 -> Int32.to_string (Int64.to_int32 (le b (4 * i) 4))
    | Int64 -> Int64.to_string (le b (8 * i) 8)
    | _ -> big_digits ~row_group t b o i
  in
  let texts =
    reading_table t @@ fun b o ->
    Array.init t.n (fun i -> Printf.sprintf "%se-%d" (digits b o i) scale)
  in
  let texts = Column.v Type.string texts in
  match Column.parse (Any Type.float64) texts with
  | Ok c -> Values (P (Column.to_tensor Nx.float64 c))
  | Error (_, why) -> Meta.fail ~row_group "a decimal value %s" why

let categorical ~row_group t dict =
  let codes = Hashtbl.create (Iarray.length dict) in
  Iarray.iteri (fun i s -> Hashtbl.replace codes s (Int32.of_int i)) dict;
  let a = fresh Nx.int32 t.n in
  reading_table t (fun b o ->
      writing a @@ fun out ->
      for i = 0 to t.n - 1 do
        let s = string_at t b o i in
        match Hashtbl.find_opt codes s with
        | Some c -> out.{i} <- c
        | None ->
            Meta.fail ~row_group ~text:s
              "the value is not in the categorical's dictionary"
      done);
  Values (P (value a))

(* [leading n a] is the value of the first [n] elements of [a]. *)
let leading n a =
  if n = Nx_array.Layout.numel (Nx_array.layout a) then value a
  else Nx.slice [ Nx.R (0, n) ] (value a)

let strings t =
  if t.width = 0 then
    let used = peek t.offsets (fun o -> Int64.to_int o.{t.n}) in
    Strings
      { offsets = leading (t.n + 1) t.offsets; data = leading used t.bytes }
  else
    let offsets = fresh Nx.int64 (t.n + 1) in
    writing offsets (fun o ->
        for i = 0 to t.n do
          o.{i} <- Int64.of_int (i * t.width)
        done);
    Strings { offsets = value offsets; data = leading (t.n * t.width) t.bytes }

(* [fixed t dt] is the values of [t] read as [dt], whose width is [t]'s. *)
let fixed t dt = retype dt t.n (leading (t.n * t.width) t.bytes)

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
let gather s idx =
  match s with
  | Values (P v) -> Values (P (Nx.take idx v))
  | Strings { offsets; data } ->
      let out, d =
        reading i64 idx @@ fun idx ->
        reading i64 offsets @@ fun offsets ->
        let m = A1.dim idx in
        let out = fresh Nx.int64 (m + 1) in
        let total = writing out (gather_ends offsets idx) in
        assert (total >= 0);
        let d = fresh Nx.uint8 total in
        reading u8 data (fun data ->
            peek out @@ fun o ->
            writing d @@ fun dst ->
            let r = gather_byte_arrays offsets data idx o dst in
            assert (r = 0));
        (out, d)
      in
      Strings { offsets = value out; data = value d }

let kernel_error = function
  | -1 -> "the data ends inside a value"
  | -2 -> "a value is out of its bounds"
  | -3 -> "a bit width is wider than its values"
  | -4 -> "a header is malformed"
  | -5 -> "a header counts other values than the page holds"
  | -8 -> "the first entry does not start a row"
  | -9 -> "an entry repeats a field that is not defined"
  | _ -> "an output is too small"

(* [spread_values v ~valid ~rows] is the values [v] at the rows the bit array
   [valid] marks, zero under the others, and the number of values it took. *)
let spread_values (type a b) (v : (a, b, Nx.host) Nx.t) ~valid ~rows =
  let dt = Nx.dtype v in
  let out = fresh Nx.uint8 (Nx.Dtype.bytes dt rows) in
  let taken =
    writing out @@ fun dst ->
    peek valid @@ fun valid ->
    reading u8 v (fun src -> spread_fixed valid src (Nx.Dtype.bits dt) rows dst)
  in
  (retype dt rows (value out), taken)

(* [spread s ~valid ~rows ~present] puts the [present] values of [s] at the rows
   the bit array [valid] marks, zero under the others. *)
let spread ~row_group s ~valid ~rows ~present =
  if present = rows then
    match s with
    | Values values -> Fixed { valid = None; values }
    | Strings { offsets; data } -> Varsize { valid = None; offsets; data }
  else
    let valid = Option.get valid in
    (* [valid] becomes the mask's memory only once the kernels have read it. *)
    let took n =
      if n < 0 then Meta.fail ~row_group "the spread: %s" (kernel_error n);
      assert (n = present)
    in
    match s with
    | Values (P v) ->
        let values, n = spread_values v ~valid ~rows in
        took n;
        Fixed { valid = Some (bits valid rows); values = P values }
    | Strings { offsets; data } ->
        let o = fresh Nx.int64 (rows + 1) in
        took
          ( writing o @@ fun o ->
            peek valid @@ fun valid ->
            reading i64 offsets (fun offsets ->
                spread_offsets valid offsets rows o) );
        let mask = bits valid rows in
        Varsize { valid = Some mask; offsets = value o; data }

(* [narrow ~row_group ty c] is [c], whose values are int32s, in the storage of
   [ty] when [ty] is an integer type narrower than int32. *)
let narrow (type a) ~row_group (ty : a Type.t) c =
  let into dt lo hi =
    match c with
    | Varsize _ | Coded _ -> assert false
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

(* [coded ~row_group idx ~valid ~rows ~offsets ~data] is the rows of a
   dictionary-encoded chunk of byte strings, whose words are those [offsets]
   and [data] hold, and whose present rows, those the bit array [valid] marks,
   read the words [idx]. The words gain an empty one, which the null rows
   read. *)
let coded ~row_group idx ~valid ~rows ~offsets ~data =
  let n = Nx.dim 0 offsets - 1 in
  let words = fresh Nx.int64 (n + 2) in
  writing words (fun w ->
      reading i64 offsets (fun o ->
          A1.blit o (A1.sub w 0 (n + 1));
          w.{n + 1} <- o.{n}));
  let offsets = value words in
  if Nx.dim 0 idx = rows then Coded { valid = None; codes = idx; offsets; data }
  else
    let valid = Option.get valid in
    let codes, taken = spread_values idx ~valid ~rows in
    if taken < 0 then
      Meta.fail ~row_group "the spread: %s" (kernel_error taken);
    assert (taken = Nx.dim 0 idx);
    let valid = bits valid rows in
    let empty = Nx.scalar Nx.int64 (Int64.of_int n) in
    let codes = Nx.where (Nx.cast Nx.bool valid) codes empty in
    Coded { valid = Some valid; codes; offsets; data }

(* Levels

   A nested leaf's level entries, its definition and repetition levels one per
   entry, as bytes when its levels fit them, as most do, and int64s otherwise:
   no schema depth is refused. Only the kernels read them. *)

type levels =
  | Levels : {
      def : ('a, 'b, Bigarray.c_layout) A1.t;
      rep : ('a, 'b, Bigarray.c_layout) A1.t;
      n : int;
    }
      -> levels

let byte_levels = 255

(* [level_arrays l n] is room for [n] entries of the leaf [l]. A level the leaf
   does not have takes no array: the walk reads it as zero. *)
let level_arrays (l : Leaf.t) n =
  let make kind =
    let a present =
      A1.create kind Bigarray.c_layout (if present then n else 0)
    in
    Levels { def = a (l.def > 0); rep = a (l.rep > 0); n = 0 }
  in
  if max l.def l.rep <= byte_levels then make u8 else make i64

let with_n (Levels lv) n = Levels { lv with n }

(* [bit_width max] is the bits a level of at most [max] takes. *)
let bit_width max =
  let rec go w = if max lsr w = 0 then w else go (w + 1) in
  go 0

(* [walk ~row_group lv s ~elem] is the validity bytes of the node [s] selects
   among the entries of [lv], its offsets when [elem] names a list's elements,
   and its slots. The validity is [None] when the node is never null. *)
let walk ~row_group (Levels lv) (s : Leaf.slots) ~elem =
  let er, ed = Option.value elem ~default:(-1, 0) in
  let run valid offsets =
    let r =
      level_walk lv.def lv.rep lv.n s.rep s.def s.nonnull er ed valid offsets
    in
    if r < 0 then Meta.fail ~row_group "the levels: %s" (kernel_error r);
    r
  in
  let slots = run no_bytes no_ints in
  let mask = if s.nonnull > s.def then Some (zeros_for_bits slots) else None in
  let offsets = Option.map (fun _ -> fresh Nx.int64 (slots + 1)) elem in
  let with_mask f =
    match mask with Some m -> writing m f | None -> f no_bytes
  in
  let with_offsets f =
    match offsets with Some o -> writing o f | None -> f no_ints
  in
  ignore (with_mask (fun v -> with_offsets (run v)));
  (mask, offsets, slots)

(* [named name f] is [f ()], whose errors name the column [name]. *)
let named name f =
  try f ()
  with Meta.Error r ->
    let msg = Printf.sprintf "column %S: %s" name r.msg in
    raise (Meta.Error { r with msg })

let structure ~row_group ~name lv s ~elem =
  named name @@ fun () ->
  let mask, offsets, slots = walk ~row_group lv s ~elem in
  (Option.map (fun m -> bits m slots) mask, Option.map value offsets, slots)

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

let read_chunk ~scratch b (m : Meta.file) ~row_group i (l : Leaf.t)
    (Type.Any ty) =
  let rows = m.row_groups.(row_group).rows in
  let cm = Option.get m.row_groups.(row_group).chunks.(i).meta in
  (* A flat column has one entry per row; a repeated one, [cm.values] entries,
     which the pages must add up to. *)
  let entries = cm.values in
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
  let valid =
    if l.optional && not l.nested then Some (zeros_for_bits rows) else None
  in
  let lv = ref None in
  (* The bigarrays of the validity, of the table's bytes when its values are
     fixed-width, and of the dictionary indices, under write claims held from
     the second page to the last, so that a page takes no claim of its own. *)
  let held_valid = ref None and held_bytes = ref None and held_idx = ref None in
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
  let decompress src pos len n : bigbytes * int * int =
    if cm.codec = Uncompressed then (src, pos, len)
    else if n = 0 then (src, pos, 0)
    else begin
      if n > cm.uncompressed_size then
        fail "the page decompresses to %d bytes, more than its chunk's %d" n
          cm.uncompressed_size;
      if A1.dim scratch.page < n then scratch.page <- bytes n;
      let dst = A1.sub scratch.page 0 n in
      let src = A1.sub src pos len in
      let r =
        match cm.codec with
        | Snappy -> Lz.Snappy.decode src dst
        | Gzip -> Lz.Gzip.decode src dst
        | Zstd -> Lz.Zstd.decode src dst
        | Lz4_raw -> Lz.Lz4.Block.decode src dst
        | _ -> invalid_arg "Chunk.read: a chunk that did not pass Chunk.check"
      in
      match r with
      | Error e ->
          fail "the page does not decompress: %s"
            (Format.asprintf "%a" Lz.pp_error e)
      | Ok { produced; _ } when produced <> n ->
          fail "the page decompresses to %d bytes, not its header's %d" produced
            n
      | Ok _ -> (dst, 0, n)
    end
  in
  (* [with_bytes t f] is [f b], [b] the bytes of [t], the chunk's table, when
     its values are fixed-width: its held claim's, or a claim of their own.
     A table of byte strings, which grows, takes claims as it writes. *)
  let with_bytes t f =
    match !held_bytes with
    | Some b -> f b
    | None when t.width > 0 -> writing t.bytes f
    | None -> f no_bytes
  in
  let levels src pos len values =
    let valid = Option.get !held_valid in
    consumes "the definition levels" len
      (hybrid_bits src pos len 1 values valid !row);
    kernel "the definition levels" (count_bits valid !row values)
  in
  (* [nested_levels src rp rl dp dl values] decodes the levels of [values]
     entries of a nested leaf into [lv], and is how many hold a value. *)
  let nested_levels src rp rl dp dl values =
    let (Levels lv) = Option.get !lv in
    if l.rep > 0 then
      consumes "the repetition levels" rl
        (hybrid src rp rl (bit_width l.rep) values lv.rep !row (l.rep + 1));
    if l.def = 0 then values
    else begin
      consumes "the definition levels" dl
        (hybrid src dp dl (bit_width l.def) values lv.def !row (l.def + 1));
      kernel "the definition levels" (level_count lv.def !row values l.def)
    end
  in
  (* [prefixed what enc src p len] is the span of the levels [what] that a
     length prefixes at [p], and the bytes of the page after them. *)
  let prefixed what (enc : Meta.encoding) src p len =
    if enc <> Rle then
      fail "%s in %s are not read" what (encoding_name enc);
    if len < 4 then fail "the %s' length overruns the page" what;
    let n = u32 src p in
    if n > len - 4 then fail "the %s overrun the page" what;
    (p + 4, n, p + 4 + n, len - 4 - n)
  in
  let plain t b src pos len n =
    match l.physical with
    | Boolean ->
        consumes "the PLAIN booleans" len (plain_bits src pos len n b t.n)
    | Byte_array ->
        ensure t len;
        filling t @@ fun b o ->
        consumes "the PLAIN byte arrays" len
          (plain_byte_array src pos len n o t.n b)
    | _ ->
        exactly "the PLAIN values" (n * t.width) len;
        A1.blit (A1.sub src pos len) (A1.sub b (t.n * t.width) len)
  in
  (* [lengths a p] is the [n] lengths delta-encoded at [p] in [a]'s first
     [n], and the bytes they took. *)
  let lengths a src pos len n p =
    let a = A1.sub a 0 n in
    ( a,
      kernel "the lengths" (delta_binary_packed src p (len - (p - pos)) n a 0 8)
    )
  in
  let delta_strings ~prefixed t src pos len n =
    if A1.dim scratch.prefixes < n then begin
      scratch.prefixes <- int64s n;
      scratch.suffixes <- int64s n
    end;
    let prefixes, c1 =
      if prefixed then lengths scratch.prefixes src pos len n pos
      else
        let z = A1.sub scratch.prefixes 0 n in
        A1.fill z 0L;
        (z, 0)
    in
    let suffixes, c2 = lengths scratch.suffixes src pos len n (pos + c1) in
    let start = pos + c1 + c2 in
    let rec go () =
      match
        filling t @@ fun b o ->
        assemble src start (len - c1 - c2) n prefixes suffixes o t.n b
      with
      | -6 (* the data is too small *) ->
          ensure t (1 + length t.bytes);
          go ()
      | r -> c1 + c2 + kernel "the byte arrays" r
    in
    exactly "the byte arrays" (go ()) len
  in
  let values t b encoding src pos len n =
    match ((encoding : Meta.encoding), l.physical) with
    | Plain, _ -> plain t b src pos len n
    | Rle, Boolean ->
        if len < 4 || u32 src pos <> len - 4 then
          fail "the RLE booleans do not fill their page";
        consumes "the RLE booleans" (len - 4)
          (hybrid_bits src (pos + 4) (len - 4) 1 n b t.n)
    | Delta_binary_packed, (Int32 | Int64) ->
        consumes "the DELTA_BINARY_PACKED values" len
          (delta_binary_packed src pos len n b t.n t.width)
    | Delta_length_byte_array, Byte_array ->
        delta_strings ~prefixed:false t src pos len n
    | Delta_byte_array, Byte_array ->
        delta_strings ~prefixed:true t src pos len n
    | Delta_byte_array, Fixed_len_byte_array ->
        let strings = table 0 ~values:n ~data:(n * t.width) in
        delta_strings ~prefixed:true strings src pos len n;
        reading_table strings @@ fun src o ->
        for k = 1 to n do
          if Int64.to_int o.{k} <> k * t.width then
            fail "a DELTA_BYTE_ARRAY value is not %d bytes long" t.width
        done;
        A1.blit
          (A1.sub src 0 (n * t.width))
          (A1.sub b (t.n * t.width) (n * t.width))
    | Byte_stream_split, (Float | Double | Int32 | Int64 | Fixed_len_byte_array)
      ->
        exactly "the BYTE_STREAM_SPLIT values" (n * t.width) len;
        for k = 0 to n - 1 do
          for j = 0 to t.width - 1 do
            b.{((t.n + k) * t.width) + j} <- src.{pos + (j * n) + k}
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
    | (Plain_dictionary | Rle_dictionary), Some _ ->
        if len < 1 then fail "the dictionary indices have no bit width";
        let idx = Option.get !held_idx in
        consumes "the dictionary indices" (len - 1)
          (hybrid src (pos + 1) (len - 1) src.{pos} n idx !present !dict)
    | _, idx ->
        let first = t.n in
        (match !held_bytes with
        | Some b -> values t b encoding src pos len n
        | None -> with_bytes t (fun b -> values t b encoding src pos len n));
        t.n <- t.n + n;
        Option.iter
          (fun _ ->
            let idx = Option.get !held_idx in
            for k = 0 to n - 1 do
              idx.{!present + k} <- Int64.of_int (first + k)
            done)
          idx);
    row := !row + rows;
    present := !present + n
  in
  let rows_left values =
    if values > entries - !row then
      fail "the page holds %d rows, past the row group's %d" values rows
  in
  (* [header pos stop] is the page header at [pos] and where its bytes start,
     which it checks lie before [stop]. *)
  let header pos stop =
    here := Some (pos, stop - 1);
    let h, data =
      try Meta.page_header b ~pos ~limit:stop
      with Thrift.Error (p, msg) ->
        Meta.fail ~row_group ~bytes:(p, p) "a page header is malformed: %s"
          msg
    in
    if h.compressed_size > stop - data then
      fail "the page overruns its column chunk";
    (h, data)
  in
  (* [page pos stop] decodes the page at [pos], and is where it ends. *)
  let page pos stop =
    let h, data = header pos stop in
    let page_end = data + h.compressed_size in
    here := Some (pos, page_end - 1);
    Option.iter
      (fun crc ->
        if Lz.Crc32.bigbytes (A1.sub b data h.compressed_size) <> crc then
          fail "the page fails its CRC-32 checksum")
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
        with_bytes t (fun b -> plain t b src p len values);
        t.n <- values;
        dict := values;
        idx := Some (fresh Nx.int64 entries)
    | Data { values; encoding; levels = def_encoding; repetition } ->
        rows_left values;
        let src, p, len =
          decompress b data h.compressed_size h.uncompressed_size
        in
        let p, len, n =
          if l.nested then begin
            let rp, rl, p, len =
              if l.rep = 0 then (p, 0, p, len)
              else
                match repetition with
                | None -> fail "the page names no repetition level encoding"
                | Some enc -> prefixed "repetition levels" enc src p len
            in
            let dp, dl, p, len =
              if l.def = 0 then (p, 0, p, len)
              else prefixed "definition levels" def_encoding src p len
            in
            (p, len, nested_levels src rp rl dp dl values)
          end
          else if not l.optional then (p, len, values)
          else
            let dp, dl, p, len =
              prefixed "definition levels" def_encoding src p len
            in
            (p, len, levels src dp dl values)
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
        if (not l.nested) && page_rows <> values then
          fail "the page holds %d values in %d rows" values page_rows;
        let size = min h.compressed_size h.uncompressed_size in
        if
          repetition_bytes > size
          || definition_bytes > size - repetition_bytes
        then fail "the levels overrun the page";
        let levels_bytes = repetition_bytes + definition_bytes in
        if
          (if l.nested then l.def = 0 else not l.optional)
          && definition_bytes <> 0
        then
          fail
            "the page holds definition levels, which a required column has \
             none of";
        let dp = data + repetition_bytes in
        (* A flat column's repetition levels are all zero: some writers write
           them, and they are skipped. *)
        let n =
          if l.nested then
            nested_levels b data repetition_bytes dp definition_bytes values
          else if l.optional then levels b dp definition_bytes values
          else values
        in
        (* Writers count a nested leaf's nulls in their own ways, so the
           levels alone decide its values. *)
        if (not l.nested) && nulls <> values - n then
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
    page_end
  in
  let rec rest pos stop = if pos < stop then rest (page pos stop) stop in
  (* [hold f] is [f ()] under write claims of the table's bytes, when its
     values are fixed-width, and of the dictionary indices. *)
  let hold f =
    let with_idx () =
      match !idx with
      | Some a ->
          writing a (fun x ->
              held_idx := Some x;
              f ())
      | None -> f ()
    in
    match !tbl with
    | Some t when t.width > 0 ->
        writing t.bytes (fun x ->
            held_bytes := Some x;
            with_idx ())
    | _ -> with_idx ()
  in
  (* The first page that is read makes the table, and the dictionary indices
     if it is the dictionary's: the claims are taken after it. *)
  let rec pages pos stop =
    if pos < stop then begin
      let pos = page pos stop in
      if Option.is_none !tbl then pages pos stop
      else begin
        if pos < stop then hold (fun () -> rest pos stop);
        held_bytes := None;
        held_idx := None
      end
    end
  in
  (* [claimed pos stop] is the entries the pages' headers add up to, without
     reading their data: nothing else bounds what a nested leaf's chunk says
     it holds, and its levels are allocated from it. *)
  let rec claimed pos stop acc =
    if pos >= stop then acc
    else
      let h, data = header pos stop in
      let acc =
        match h.page with
        | Data { values; _ } | Data_v2 { values; _ } -> acc + values
        | Dictionary _ | Other -> acc
      in
      claimed (data + h.compressed_size) stop acc
  in
  let first = first_page cm in
  let stop = first + cm.compressed_size in
  let decode () =
    if l.nested && rows > 0 then begin
      let n = claimed first stop 0 in
      if n <> entries then
        fail "the pages hold %d values, and the column chunk %d" n entries
    end;
    if l.nested then lv := Some (level_arrays l entries);
    if rows > 0 then pages first stop
  in
  (match valid with
  | Some v ->
      writing v (fun x ->
          held_valid := Some x;
          decode ())
  | None -> decode ());
  held_valid := None;
  here := None;
  if !row <> entries then
    fail "the pages hold %d rows, and the row group %d" !row rows;
  (* A nested leaf's values spread onto its own slots, which its levels give. *)
  let lv = Option.map (fun lv -> with_n lv !row) !lv in
  let valid, rows =
    match lv with
    | None -> (valid, rows)
    | Some lv ->
        let slots : Leaf.slots =
          { rep = l.rep; def = l.slot_def; nonnull = l.def }
        in
        let mask, _, n = walk ~row_group lv slots ~elem:None in
        (mask, n)
  in
  let s = convert ~row_group l ty (chunk_table ~dictionary:0) in
  let c =
    match (!idx, s) with
    | Some idx, Strings { offsets; data } ->
        coded ~row_group (leading !present idx) ~valid ~rows ~offsets ~data
    | Some idx, Values _ ->
        let s = gather s (leading !present idx) in
        narrow ~row_group ty
          (spread ~row_group s ~valid ~rows ~present:!present)
    | None, _ ->
        narrow ~row_group ty
          (spread ~row_group s ~valid ~rows ~present:!present)
  in
  (c, lv)

let read ~scratch b m ~row_group i (l : Leaf.t) ty =
  if l.nested then
    named (Leaf.name l) (fun () -> read_chunk ~scratch b m ~row_group i l ty)
  else read_chunk ~scratch b m ~row_group i l ty

let flat_coded = function
  | Coded { valid; codes; offsets; data } -> (
      match gather (Strings { offsets; data }) codes with
      | Strings { offsets; data } -> Varsize { valid; offsets; data }
      | Values _ -> assert false)
  | (Fixed _ | Varsize _) as c -> c

(* [word_length offsets] is the length of every word of [offsets], the empty
   word after them aside, when they all have one length. *)
let word_length offsets =
  let n = Nx.dim 0 offsets - 2 in
  if n <= 0 then None
  else
    let lengths =
      Nx.sub
        (Nx.slice [ Nx.R (1, n + 1) ] offsets)
        (Nx.slice [ Nx.R (0, n) ] offsets)
    in
    let w = Nx.item [ 0 ] lengths in
    if Nx.item [] (Nx.all (Nx.equal_s lengths w)) then Some (Int64.to_int w)
    else None

(* [gather_words codes data ~n w] is the [n] words of [w] bytes that [data]
   holds at [codes], and their offsets: a gather of rows of [w] bytes, against
   the row-by-row copy of words of mixed lengths. *)
let gather_words codes data ~n w =
  let m = Nx.dim 0 codes in
  let words () =
    if w = 0 then Nx.zeros Nx.uint8 [| 0 |]
    else
      let d = Nx.slice [ Nx.R (0, n * w) ] data in
      if w = 1 then Nx.take codes d
      else
        Nx.reshape [| m * w |]
          (Nx.take ~axis:0 codes (Nx.reshape [| n; w |] d))
  in
  let offsets =
    if w = 0 then Nx.zeros Nx.int64 [| m + 1 |]
    else Nx.arange Nx.int64 0 ((m + 1) * w) w
  in
  (offsets, words ())

let flat = function
  | Coded { valid = None; codes; offsets; data } as c -> (
      match word_length offsets with
      | Some w ->
          let offsets, data =
            gather_words codes data ~n:(Nx.dim 0 offsets - 2) w
          in
          Varsize { valid = None; offsets; data }
      | None -> flat_coded c)
  | Coded _ as c -> flat_coded c
  | (Fixed _ | Varsize _) as c -> c
