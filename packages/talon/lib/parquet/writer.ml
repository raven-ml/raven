(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
module A1 = Bigarray.Array1
module W = Bytesrw.Bytes.Writer

type bigbytes = Meta.bigbytes
type u16s = (int, Bigarray.int16_unsigned_elt, Bigarray.c_layout) A1.t
type int64s = (int64, Bigarray.int64_elt, Bigarray.c_layout) A1.t

exception Failed of string

let fail fmt = Printf.ksprintf (fun msg -> raise (Failed msg)) fmt

external rle_encode :
  ('a, 'b, Bigarray.c_layout) A1.t -> int -> int -> int -> bigbytes -> int
  = "talon_parquet_rle_encode"
[@@noalloc]

external plain_byte_arrays :
  int64s -> bigbytes -> int -> int -> bigbytes -> int
  = "talon_parquet_plain_byte_array_encode"
[@@noalloc]

external pack_bits : bigbytes -> int -> int -> bigbytes -> int
  = "talon_parquet_pack_bits"
[@@noalloc]

external to_bytes : bigbytes -> int -> Bytes.t -> int -> int
  = "talon_parquet_to_bytes"
[@@noalloc]

(* Constants *)

(* [created_by]: the application, and the version of this writer's output,
   raised when a reader must tell its files apart. It does not follow talon's
   releases, so that a release leaves files' bytes alone. A value without
   " version " makes parquet-java ignore a file's byte-array statistics. *)
let created_by = "talon version 1.0.0"
let i32_max = 0x7FFF_FFFF

(* Pages are the unit a reader decodes or skips: about 1 MiB before
   compression, as Parquet's reference writers cut them. *)
let page_target = 1 lsl 20

(* A longer min or max byte-string statistic is truncated, and marked inexact,
   which keeps long values out of the footer. *)
let stat_limit = 64

(* Schema

   A talon type's Parquet type. Every field is optional: a schema does not say
   whether a column holds nulls, and a required field cannot hold a later
   batch's. Clocks and datetimes of seconds write as milliseconds, durations
   as their [int64] ticks, categoricals as text and extensions as their
   storage. *)

let unit_written : Type.unit_ -> Type.unit_ = function S -> Ms | u -> u

(* [primitive t] is the physical type, length and annotation of a type that is
   a Parquet primitive, or [None] for a nested one. *)
let primitive (Type.Any t) : (Meta.physical * int * Meta.logical option) option
    =
  let int (p : Meta.physical) bits signed =
    Some (p, 0, Some (Meta.Integer { bits; signed }))
  in
  match t with
  | Bool -> Some (Boolean, 0, None)
  | Int8 -> int Int32 8 true
  | Int16 -> int Int32 16 true
  | Int32 -> Some (Int32, 0, None)
  | Int64 | Duration _ -> Some (Int64, 0, None)
  | Uint8 -> int Int32 8 false
  | Uint16 -> int Int32 16 false
  | Uint32 -> int Int32 32 false
  | Uint64 -> int Int64 64 false
  | Float16 -> Some (Fixed_len_byte_array, 2, Some Float16)
  | Float32 -> Some (Float, 0, None)
  | Float64 -> Some (Double, 0, None)
  | String | Categorical _ -> Some (Byte_array, 0, Some String)
  | Binary -> Some (Byte_array, 0, None)
  | Date -> Some (Int32, 0, Some Date)
  | Clock u ->
      let unit_ = unit_written u in
      let p : Meta.physical = if unit_ = Ms then Int32 else Int64 in
      Some (p, 0, Some (Time { unit_; utc = false }))
  | Datetime { unit_; clock } ->
      let utc = match clock with Time.Utc -> true | Time.Local -> false in
      Some (Int64, 0, Some (Timestamp { unit_ = unit_written unit_; utc }))
  | List _ | Record _ | Tensor _ | Ext _ -> None

let element_type dt =
  match Column.of_tensor (Nx.zeros dt [| 0 |]) with
  | c -> Column.type_ c
  | exception Invalid_argument _ ->
      invalid_arg
        (Format.asprintf
           "Talon_parquet.format: a tensor of %a elements has no Parquet type"
           Nx.Dtype.pp dt)

(* [tensor_list dt shape] is the type of the nested lists a tensor of [shape]
   over [dt]'s elements writes as. *)
let tensor_list dt shape =
  let wrap _ (Type.Any t) = Type.Any (Type.list t) in
  Array.fold_right wrap (Iarray.to_array shape) (element_type dt)

(* [written t] is the type that a column of [t] writes as, and reads back as
   when it is flat. *)
let rec written (Type.Any t as ty) : Type.any =
  match t with
  | Categorical _ -> Any Type.string
  | Clock S -> Any (Type.clock Ms)
  | Datetime { unit_ = S; clock = Time.Utc } -> Any (Type.instant Ms)
  | Datetime { unit_ = S; clock = Time.Local } -> Any (Type.wall Ms)
  | Duration _ -> Any Type.int64
  | Ext { storage; _ } -> written (Any storage)
  | List e ->
      let (Type.Any e) = written (Any e) in
      Any (Type.list e)
  | Record [] ->
      invalid_arg
        "Talon_parquet.format: a record without fields has no Parquet type"
  | Record fields ->
      Any (Type.record (List.map (fun (f, t) -> (f, written t)) fields))
  | Tensor (dt, shape) -> written (tensor_list dt shape)
  | _ -> ty

(* A level is held in a [u16], so a column's highest level must fit one. Parquet
   sets no such bound: this is the range of the writer's own level buffers. *)
let max_level = 0xFFFF

(* [top_level t] is the highest definition level of a column of written type [t]:
   an optional list takes two levels, for itself and its repeated element, an
   optional record one and an optional value one. *)
let rec top_level (Type.Any t) =
  match t with
  | List e -> 2 + top_level (Any e)
  | Record fields ->
      1 + List.fold_left (fun m (_, t) -> Int.max m (top_level t)) 0 fields
  | _ -> 1

let column name ty =
  let w = written ty in
  let top = top_level w in
  if top > max_level then
    invalid_arg
      (Printf.sprintf
         "Talon_parquet: column %S reaches definition level %d, past the %d a \
          level holds"
         name top max_level);
  (name, w)

let element name physical length logical children (repetition : Meta.repetition)
    : Meta.element =
  {
    name;
    physical;
    length;
    repetition = Some repetition;
    children;
    converted = None;
    precision = 0;
    scale = 0;
    logical;
  }

(* [elements name t] is the column [name] of type [t] as schema elements, depth
   first, a list as Parquet's three levels. *)
let rec elements name (Type.Any t as ty) : Meta.element list =
  match t with
  | Ext { storage; _ } -> elements name (Any storage)
  | Tensor (dt, shape) -> elements name (tensor_list dt shape)
  | List e ->
      element name None 0 (Some List) 1 Optional
      :: element "list" None 0 None 1 Repeated
      :: elements "element" (Any e)
  | Record fields ->
      element name None 0 None (List.length fields) Optional
      :: List.concat_map (fun (f, t) -> elements f t) fields
  | _ ->
      let p, len, log = Option.get (primitive ty) in
      [ element name (Some p) len log 0 Optional ]

let schema_elements columns =
  let body = List.concat_map (fun (n, t) -> elements n t) columns in
  let root =
    {
      (element "schema" None 0 None (List.length columns) Required) with
      repetition = None;
    }
  in
  Array.of_list (root :: body)

(* Values

   A leaf's non-null values as they write: bits, fixed-width values in the dtype
   of the leaf's physical type, or byte strings. *)

type values =
  | Bits of Nx.host Nx.bit_t
  | Fixed_width of Nx.host Nx.packed * int
  | Byte_strings of (int, Nx.int8_unsigned_elt, Nx.host) Nx_ragged.t

let u8 = Bigarray.int8_unsigned
let i64 = Bigarray.int64
let width (Nx.P x) = (Nx.Dtype.bits (Nx.dtype x) + 7) / 8

(* [physical t x] is [x], values of [t], in the dtype of its physical type. *)
let physical (Type.Any t) (Nx.P x) : Nx.host Nx.packed =
  match t with
  | Int8 | Int16 | Uint8 | Uint16 -> P (Nx.cast Nx.int32 x)
  | Uint32 -> P (Nx.bitcast Nx.int32 x)
  | Uint64 -> P (Nx.bitcast Nx.int64 x)
  | Clock Ms -> P (Nx.cast Nx.int32 x)
  | Clock S -> P (Nx.cast Nx.int32 (Nx.mul_s (Nx.cast Nx.int64 x) 1000L))
  | Datetime { unit_ = S; _ } ->
      let x = Nx.cast Nx.int64 x in
      let limit = 9_223_372_036_854_775L in
      let over =
        Nx.logical_or (Nx.greater_s x limit) (Nx.less_s x (Int64.neg limit))
      in
      if Nx.dim 0 x > 0 && Nx.item [] (Nx.any over) then
        fail "a datetime in seconds does not fit Parquet's milliseconds";
      P (Nx.mul_s x 1000L)
  | _ -> P x

(* [compacted t c at] is the values of the flat column [c] of type [t] at the
   rows [at], or all its rows, in their written form. *)
let compacted (Type.Any t as ty) c at =
  let pick x = match at with None -> x | Some i -> Nx.take i x in
  match (t, Column.layout c) with
  | Bool, Fixed { values = Nx.P x; _ } -> Bits (Nx.cast Nx.bit (pick x))
  | (String | Binary), Varsize { offsets; child; _ } ->
      let bytes =
        match Column.layout child with
        | Fixed { values = Nx.P x; _ } ->
            Nx_ragged.v ~offsets (Nx.cast Nx.uint8 x)
        | _ -> assert false
      in
      Byte_strings
        (match at with
        | None -> bytes
        | Some i -> Nx_ragged.take ~indices:i bytes)
  | Categorical dict, Fixed { values = Nx.P x; _ } ->
      let words = Nx_ragged.of_strings (Iarray.to_array dict) in
      Byte_strings (Nx_ragged.take ~indices:(Nx.cast Nx.int64 (pick x)) words)
  | _, Fixed { values = Nx.P x; _ } ->
      let p = physical ty (P (pick x)) in
      Fixed_width (p, width p)
  | _ -> assert false

let size = function
  | Bits x -> (Nx.dim 0 x + 7) / 8
  | Fixed_width (Nx.P x, w) -> Nx.dim 0 x * w
  | Byte_strings r ->
      let o = Nx_ragged.offsets r and n = Nx_ragged.length r in
      if n = 0 then 0
      else
        Int64.to_int (Int64.sub (Nx.item [ n ] o) (Nx.item [ 0 ] o)) + (4 * n)

let count = function
  | Bits x -> Nx.dim 0 x
  | Fixed_width (Nx.P x, _) -> Nx.dim 0 x
  | Byte_strings r -> Nx_ragged.length r

(* Leaves *)

type leaf = {
  path : string list;
  ty : Type.any;
  physical : Meta.physical;
  values : values;
  defs : u16s;  (* one per slot *)
  reps : u16s option;  (* [None] outside a repeated field *)
  max_def : int;
  max_rep : int;
  nulls : int;
}

let u16s n = A1.create Bigarray.int16_unsigned Bigarray.c_layout n

let physical_of ty =
  match primitive ty with Some (p, _, _) -> p | None -> assert false

let rec storage_type (Type.Any t as any) =
  match t with Ext { storage; _ } -> storage_type (Any storage) | _ -> any

(* [flat name ty c n] is the leaf of a flat column of [n] rows: definition level
   [1] where a row holds a value, read from its validity. *)
let flat name ty c n =
  let ty = storage_type ty in
  let defs = u16s n in
  A1.fill defs 1;
  let nulls = Column.null_count c in
  let at =
    match Column.validity c with
    | Some v when nulls > 0 ->
        Chunk.reading u8 (Nx.cast Nx.uint8 v) (fun b ->
            for i = 0 to n - 1 do
              A1.unsafe_set defs i (A1.unsafe_get b i)
            done);
        Some (Nx.positions v)
    | _ -> None
  in
  {
    path = [ name ];
    ty;
    physical = physical_of ty;
    values = compacted ty c at;
    defs;
    reps = None;
    max_def = 1;
    max_rep = 0;
    nulls;
  }

(* A tensor column as the lists it writes as: the rows' validity on the
   outermost list, offsets from the innermost dimension out. *)
let tensor_as_lists dt shape validity (Nx.P x) =
  let shape = Iarray.to_array shape in
  let inner = Column.of_tensor (Nx.reshape [| -1 |] (Nx.cast dt x)) in
  let n = Nx.dim 0 x in
  (* [rows] of the list at each depth, outermost first: its row count is the
     product of the dimensions above it. *)
  let build (Type.Any t, child) d =
    let rows = Array.fold_left ( * ) n (Array.sub shape 0 d) in
    let offsets =
      Nx.mul_s (Nx.arange Nx.int64 0 (rows + 1) 1) (Int64.of_int shape.(d))
    in
    let any = Type.Any (Type.list t) in
    let validity = if d = 0 then validity else None in
    match Column.of_layout any (Varsize { validity; offsets; child }) with
    | Ok c -> (any, c)
    | Error (_, why) -> fail "a tensor column: %s" why
  in
  let d = Array.length shape in
  let depths = List.init d (fun i -> d - 1 - i) in
  List.fold_left build (Column.type_ inner, inner) depths

(* Slots run down the schema, as Dremel's striping: [rep] and [def] are the
   levels reached so far, and [rows] the row of the next column down, or -1 when
   no value lies below it. A list makes one slot per element, one for an empty
   or a null list; a record keeps its parent's. *)
type slots = { rows : int array; rep : int array; def : int array }

let valid_at = function
  | None -> fun _ -> true
  | Some v ->
      let bits = Nx.to_array (Nx.cast Nx.uint8 v) in
      fun i -> bits.(i) <> 0

let rec shred (Type.Any t as ty) c slots ~path ~max_def ~max_rep out =
  match t with
  | Ext { storage; _ } ->
      shred (Any storage) c slots ~path ~max_def ~max_rep out
  | Tensor (dt, shape) -> (
      match Column.layout c with
      | Fixed { validity; values } ->
          let ty, c = tensor_as_lists dt shape validity values in
          shred ty c slots ~path ~max_def ~max_rep out
      | _ -> assert false)
  | List e -> (
      match Column.layout c with
      | Varsize { validity; offsets; child } ->
          let valid = valid_at validity and off = Nx.to_array offsets in
          let len r = Int64.(to_int (sub off.(r + 1) off.(r))) in
          let m = Array.length slots.rows in
          let total = ref 0 in
          for s = 0 to m - 1 do
            let r = slots.rows.(s) in
            total := !total + if r >= 0 && valid r then Int.max 1 (len r) else 1
          done;
          let rows = Array.make !total (-1) in
          let rep = Array.make !total 0 and def = Array.make !total 0 in
          let k = ref 0 in
          let push row rp df =
            rows.(!k) <- row;
            rep.(!k) <- rp;
            def.(!k) <- df;
            incr k
          in
          for s = 0 to m - 1 do
            let r = slots.rows.(s)
            and rp = slots.rep.(s)
            and df = slots.def.(s) in
            if r < 0 || not (valid r) then push (-1) rp df
            else if len r = 0 then push (-1) rp (df + 1)
            else
              let first = Int64.to_int off.(r) in
              for j = first to first + len r - 1 do
                push j (if j = first then rp else max_rep + 1) (df + 2)
              done
          done;
          shred (Any e) child { rows; rep; def }
            ~path:("element" :: "list" :: path)
            ~max_def:(max_def + 2) ~max_rep:(max_rep + 1) out
      | _ -> assert false)
  | Record fields -> (
      match Column.layout c with
      | Children { validity; fields = columns; _ } ->
          let valid = valid_at validity in
          let m = Array.length slots.rows in
          let rows = Array.make m (-1) and def = Array.copy slots.def in
          for s = 0 to m - 1 do
            let r = slots.rows.(s) in
            if r >= 0 && valid r then begin
              rows.(s) <- r;
              def.(s) <- slots.def.(s) + 1
            end
          done;
          let child (name, ft) (_, col) =
            shred ft col { rows; rep = slots.rep; def } ~path:(name :: path)
              ~max_def:(max_def + 1) ~max_rep out
          in
          List.iter2 child fields columns
      | _ -> assert false)
  | _ ->
      let valid = valid_at (Column.validity c) in
      let m = Array.length slots.rows in
      let defs = u16s m and reps = u16s m in
      let at = Array.make m 0 and present = ref 0 in
      for s = 0 to m - 1 do
        let r = slots.rows.(s) in
        A1.unsafe_set reps s slots.rep.(s);
        if r >= 0 && valid r then begin
          A1.unsafe_set defs s (max_def + 1);
          at.(!present) <- r;
          incr present
        end
        else A1.unsafe_set defs s slots.def.(s)
      done;
      let idx =
        Nx.create Nx.int64 [| !present |]
          (Array.init !present (fun i -> Int64.of_int at.(i)))
      in
      out :=
        {
          path = List.rev path;
          ty;
          physical = physical_of ty;
          values = compacted ty c (Some idx);
          defs;
          reps = (if max_rep > 0 then Some reps else None);
          max_def = max_def + 1;
          max_rep;
          nulls = m - !present;
        }
        :: !out

let leaves name ty c rows =
  match primitive (storage_type ty) with
  | Some _ -> [ flat name ty c rows ]
  | None ->
      let slots =
        {
          rows = Array.init rows Fun.id;
          rep = Array.make rows 0;
          def = Array.make rows 0;
        }
      in
      let out = ref [] in
      shred ty c slots ~path:[ name ] ~max_def:0 ~max_rep:0 out;
      List.rev !out

(* Statistics *)

type encoding =
  | Plain
  | Dictionary of { words : values; ids : Nx.host Nx.int64_t; bits : int }

let is_text (Type.Any t) =
  match t with String | Categorical _ -> true | _ -> false

(* [boundary s n] is the length of the longest prefix of the UTF-8 [s] of at
   most [n] bytes that ends between two scalar values. *)
let boundary s n =
  let k = ref (Int.min n (String.length s)) in
  while !k > 0 && !k < String.length s && Char.code s.[!k] land 0xC0 = 0x80 do
    decr k
  done;
  !k

(* [next s ~text] is the smallest value above every value with the prefix [s],
   valid UTF-8 if [text]; [None] if there is none. *)
let rec next s ~text =
  if s = "" then None
  else if text then begin
    let i = ref (String.length s - 1) in
    while !i > 0 && Char.code s.[!i] land 0xC0 = 0x80 do
      decr i
    done;
    let u = Uchar.utf_decode_uchar (String.get_utf_8_uchar s !i) in
    if Uchar.equal u Uchar.max then next (String.sub s 0 !i) ~text
    else begin
      let b = Buffer.create (!i + 4) in
      Buffer.add_substring b s 0 !i;
      Buffer.add_utf_8_uchar b (Uchar.succ u);
      Some (Buffer.contents b)
    end
  end
  else
    let n = String.length s - 1 in
    if s.[n] = '\xff' then next (String.sub s 0 n) ~text
    else
      Some (String.sub s 0 n ^ String.make 1 (Char.chr (Char.code s.[n] + 1)))

(* [truncated s ~text] is the prefix of a value past {!stat_limit}, cut at a
   scalar value's end if [text]. A minimum becomes its prefix; a maximum the
   next value after its prefix, when one exists. *)
let truncated s ~text =
  String.sub s 0 (if text then boundary s stat_limit else stat_limit)

let truncate_min s ~text =
  if String.length s <= stat_limit then (s, true)
  else (truncated s ~text, false)

let truncate_max s ~text =
  if String.length s <= stat_limit then Some (s, true)
  else Option.map (fun v -> (v, false)) (next (truncated s ~text) ~text)

let bytes_of (Nx.P x) =
  Chunk.reading u8
    (Nx.reshape [| 1 |] x)
    (fun b -> String.init (A1.dim b) (fun i -> Char.chr b.{i}))

let row (o : int64s) (d : bigbytes) i =
  let a = Int64.to_int o.{i} and e = Int64.to_int o.{i + 1} in
  String.init (e - a) (fun j -> Char.chr d.{a + j})

(* [compare_rows o d i j] orders rows [i] and [j] as Parquet orders byte
   arrays: unsigned, a prefix first. *)
let compare_rows (o : int64s) (d : bigbytes) i j =
  let ai = Int64.to_int o.{i} and ei = Int64.to_int o.{i + 1} in
  let aj = Int64.to_int o.{j} and ej = Int64.to_int o.{j + 1} in
  let n = Int.min (ei - ai) (ej - aj) in
  let rec go k =
    if k = n then Int.compare (ei - ai) (ej - aj)
    else
      let c = Int.compare d.{ai + k} d.{aj + k} in
      if c <> 0 then c else go (k + 1)
  in
  go 0

(* [stats leaf enc] is the chunk's statistics. Byte strings are compared over
   the dictionary's words, when [enc] has them, as their rows repeat them. *)
let stats leaf enc : Meta.stats =
  let base =
    {
      Meta.nulls = Some leaf.nulls;
      nans = None;
      min = None;
      max = None;
      min_exact = None;
      max_exact = None;
    }
  in
  let bounded s lo hi =
    {
      s with
      Meta.min = Some lo;
      max = Some hi;
      min_exact = Some true;
      max_exact = Some true;
    }
  in
  let extremes x = (bytes_of (P (Nx.min x)), bytes_of (P (Nx.max x))) in
  let of_extremes (lo, hi) = bounded base lo hi in
  let (Type.Any t) = leaf.ty in
  (* The bounds hold over the non-NaN values; a zero minimum is [-0.] and a
     zero maximum [+0.], as Parquet's column order asks. *)
  let float_stats dt x =
    let nan = Nx.isnan x in
    let nans = Int64.to_int (Nx.item [] (Nx.count nan)) in
    let s = { base with nans = Some nans } in
    if nans = Nx.dim 0 x then s
    else
      let lo = Nx.min (Nx.where nan (Nx.scalar dt Float.infinity) x) in
      let hi = Nx.max (Nx.where nan (Nx.scalar dt Float.neg_infinity) x) in
      let lo = if Nx.item [] lo = 0. then Nx.scalar dt (-0.) else lo in
      let hi = if Nx.item [] hi = 0. then Nx.scalar dt 0. else hi in
      bounded s (bytes_of (P lo)) (bytes_of (P hi))
  in
  match (leaf.values, t) with
  | v, (Float16 | Float32 | Float64) when count v = 0 ->
      { base with nans = Some 0 }
  | v, _ when count v = 0 -> base
  | Bits x, _ -> of_extremes (extremes (Nx.cast Nx.uint8 x))
  | Byte_strings r, _ ->
      let r =
        match enc with Dictionary { words = Byte_strings w; _ } -> w | _ -> r
      in
      let text = is_text leaf.ty in
      Chunk.reading i64 (Nx_ragged.offsets r) @@ fun o ->
      Chunk.reading u8 (Nx_ragged.values r) @@ fun d ->
      let lo_row = ref 0 and hi_row = ref 0 in
      for i = 1 to Nx_ragged.length r - 1 do
        if compare_rows o d i !lo_row < 0 then lo_row := i;
        if compare_rows o d i !hi_row > 0 then hi_row := i
      done;
      let lo, min_exact = truncate_min (row o d !lo_row) ~text in
      let hi = truncate_max (row o d !hi_row) ~text in
      {
        base with
        min = Some lo;
        min_exact = Some min_exact;
        max = Option.map fst hi;
        max_exact = Option.map snd hi;
      }
  | Fixed_width (Nx.P x, _), Float16 ->
      float_stats Nx.float16 (Nx.cast Nx.float16 x)
  | Fixed_width (Nx.P x, _), Float32 ->
      float_stats Nx.float32 (Nx.cast Nx.float32 x)
  | Fixed_width (Nx.P x, _), Float64 ->
      float_stats Nx.float64 (Nx.cast Nx.float64 x)
  | Fixed_width (Nx.P x, _), Uint32 ->
      of_extremes (extremes (Nx.bitcast Nx.uint32 x))
  | Fixed_width (Nx.P x, _), Uint64 ->
      of_extremes (extremes (Nx.bitcast Nx.uint64 x))
  | Fixed_width (Nx.P x, _), _ -> of_extremes (extremes x)

(* Encodings

   Every leaf but a boolean is dictionary-encoded when the dictionary and its
   indices take fewer bytes than the values. Values are the same key when their
   bits are, so a NaN keeps its payload and a zero its sign. *)

let bit_width k =
  let rec go w = if (k - 1) lsr w = 0 then w else go (w + 1) in
  Int.max 1 (go 0)

let dictionary_of leaf =
  match leaf.values with
  | Byte_strings r ->
      let g = Nx_ragged.unique r in
      Some (Byte_strings (Nx_ragged.take ~indices:g.first r), g.ids)
  | Fixed_width (Nx.P x, w) ->
      let g =
        match w with
        | 2 -> Nx.unique (Nx.bitcast Nx.int16 x)
        | 4 -> Nx.unique (Nx.bitcast Nx.int32 x)
        | _ -> Nx.unique (Nx.bitcast Nx.int64 x)
      in
      Some (Fixed_width (P (Nx.take g.first x), w), g.ids)
  | Bits _ -> None

let encoding_of leaf =
  let n = count leaf.values in
  match if n = 0 then None else dictionary_of leaf with
  | None -> Plain
  | Some (words, ids) ->
      let bits = bit_width (count words) in
      let dict = size words and indices = ((n * bits) + 7) / 8 in
      if dict <= i32_max && dict + indices < size leaf.values then
        Dictionary { words; ids; bits }
      else Plain

(* Pages *)

type out = {
  w : W.t;
  codec : Meta.codec;
  mutable pos : int;  (* the bytes written so far *)
  mutable page : bigbytes;  (* the page being assembled *)
  mutable zipped : bigbytes;  (* its compressed form *)
}

let new_bytes n = A1.create u8 Bigarray.c_layout n

let write_string out s =
  if s <> "" then begin
    W.write_string out.w s;
    out.pos <- out.pos + String.length s
  end

let write_bigbytes out b n =
  if n > 0 then begin
    let b' = Bytes.create n in
    if to_bytes b 0 b' n < 0 then fail "a page overruns its buffer";
    W.write_bytes out.w b';
    out.pos <- out.pos + n
  end

let need out n =
  if A1.dim out.page < n then
    out.page <- new_bytes (Int.max n (2 * A1.dim out.page))

let zip out n =
  let via bound encode =
    let m = bound n in
    if A1.dim out.zipped < m then out.zipped <- new_bytes m;
    match encode (A1.sub out.page 0 n) (A1.sub out.zipped 0 m) with
    | Some k -> (out.zipped, k)
    | None -> fail "a page does not fit its compression bound"
  in
  match out.codec with
  | Uncompressed -> (out.page, n)
  | Snappy -> via Lz.Snappy.bound Lz.Snappy.encode
  | Gzip -> via Lz.Gzip.bound (Lz.Gzip.encode ?level:None)
  | Zstd -> via Lz.Zstd.bound (Lz.Zstd.encode ?level:None)
  | Lz4_raw -> via Lz.Lz4.Block.bound Lz.Lz4.Block.encode
  | _ -> assert false

let hybrid_bound n bits = 16 + (n * (((bits + 7) / 8) + 1))

let level_bits level =
  let rec go w = if level lsr w = 0 then w else go (w + 1) in
  go 1

(* [add_levels out pos levels a b bits] writes at [pos] of the page the levels
   [a, b) as a 4-byte length and their hybrid, and is the next position. *)
let add_levels out pos (levels : u16s) a b bits =
  let dst = A1.sub out.page (pos + 4) (A1.dim out.page - pos - 4) in
  let k = rle_encode (A1.sub levels a (b - a)) 2 (b - a) bits dst in
  if k < 0 then fail "levels overrun their page";
  for i = 0 to 3 do
    A1.set out.page (pos + i) ((k lsr (8 * i)) land 0xFF)
  done;
  pos + 4 + k

(* [plain_size v a b] and [add_plain v a b dst] are the [PLAIN] bytes of the
   values [a, b) of [v], which [add_plain] writes to [dst]. *)
let plain_size v a b =
  match v with
  | Bits _ -> (b - a + 7) / 8
  | Fixed_width (_, w) -> (b - a) * w
  | Byte_strings r ->
      Chunk.reading i64 (Nx_ragged.offsets r) (fun o ->
          Int64.to_int (Int64.sub o.{b} o.{a}) + (4 * (b - a)))

let add_plain v a b dst =
  match v with
  | Bits x -> Chunk.reading u8 x (fun src -> pack_bits src a (b - a) dst)
  | Fixed_width (Nx.P x, w) ->
      Chunk.reading u8 x (fun src ->
          let n = (b - a) * w in
          A1.blit (A1.sub src (a * w) n) (A1.sub dst 0 n);
          n)
  | Byte_strings r ->
      Chunk.reading i64 (Nx_ragged.offsets r) (fun o ->
          Chunk.reading u8 (Nx_ragged.values r) (fun d ->
              plain_byte_arrays o d a b dst))

(* [with_cost leaf enc f] is [f cost], [cost v] the bits value [v] takes in a
   page. *)
let with_cost leaf enc f =
  match (enc, leaf.values) with
  | Dictionary { bits; _ }, _ -> f (fun _ -> bits)
  | Plain, Bits _ -> f (fun _ -> 1)
  | Plain, Fixed_width (_, w) -> f (fun _ -> 8 * w)
  | Plain, Byte_strings r ->
      Chunk.reading i64 (Nx_ragged.offsets r) (fun (o : int64s) ->
          f (fun v -> 8 * (4 + Int64.to_int (Int64.sub o.{v + 1} o.{v}))))

(* [page_end defs top cost a va] is the end of the page that starts at slot [a]
   and value [va], and the value after it: at the first value that would take
   it past {!page_target}, unless it is the page's first, or at
   {!page_target} slots, which bounds its levels. *)
let page_end (defs : u16s) top cost a va =
  let slots = A1.dim defs and target = 8 * page_target in
  let rec scan b v bits =
    if b = slots || b - a >= page_target then (b, v)
    else if A1.unsafe_get defs b <> top then scan (b + 1) v bits
    else
      let c = cost v in
      if bits > 0 && bits + c > target then (b, v)
      else scan (b + 1) (v + 1) (bits + c)
  in
  scan a va 0

(* [chunk out leaf] writes the column chunk of [leaf], a dictionary page if its
   encoding has one, then data pages of at most {!page_target} bytes of values:
   the levels of a page's slots, its values and, for a dictionary, its indices'
   bit width then their hybrid. *)
let chunk out leaf : Meta.column_out =
  let start = out.pos and enc = encoding_of leaf in
  let raw = ref 0 in
  let page header size =
    if size > i32_max then
      fail "a page of %d bytes is past Parquet's limit" size;
    let data, zipped = zip out size in
    if zipped > i32_max then
      fail "a compressed page of %d bytes is past Parquet's limit" zipped;
    let h = header ~uncompressed:size ~compressed:zipped in
    write_string out h;
    write_bigbytes out data zipped;
    raw := !raw + String.length h + size
  in
  let dictionary_page =
    match enc with
    | Plain -> None
    | Dictionary { words; _ } ->
        let at = out.pos and n = size words in
        need out n;
        if n > 0 && add_plain words 0 (count words) (A1.sub out.page 0 n) < 0
        then fail "the dictionary overruns its page";
        page (Meta.dictionary_page_bytes ~values:(count words)) n;
        Some at
  in
  let data_page = out.pos in
  let slots = A1.dim leaf.defs in
  let def_bits = level_bits leaf.max_def in
  let rep_bits = level_bits leaf.max_rep in
  let write_page a b va vb =
    let nv = vb - va in
    let values_size =
      match enc with
      | Plain -> plain_size leaf.values va vb
      | Dictionary { bits; _ } -> 1 + hybrid_bound nv bits
    in
    let n = b - a in
    need out
      (values_size + 8 + hybrid_bound n def_bits + hybrid_bound n rep_bits);
    let p = ref 0 in
    Option.iter
      (fun reps -> p := add_levels out !p reps a b rep_bits)
      leaf.reps;
    p := add_levels out !p leaf.defs a b def_bits;
    let rest () = A1.sub out.page !p (A1.dim out.page - !p) in
    (match enc with
    | Plain ->
        if nv > 0 then begin
          let k = add_plain leaf.values va vb (rest ()) in
          if k < 0 then fail "a value overruns Parquet's 2 GiB limit";
          p := !p + k
        end
    | Dictionary { ids; bits; _ } ->
        A1.set out.page !p bits;
        incr p;
        let k =
          Chunk.reading i64 ids (fun o ->
              rle_encode (A1.sub o va nv) 8 nv bits (rest ()))
        in
        if k < 0 then fail "dictionary indices overrun their page";
        p := !p + k);
    let encoding : Meta.encoding =
      match enc with Plain -> Plain | Dictionary _ -> Rle_dictionary
    in
    page (Meta.data_page_bytes ~values:n ~encoding) !p
  in
  with_cost leaf enc (fun cost ->
      let rec pages a va =
        if a < slots then begin
          let b, vb = page_end leaf.defs leaf.max_def cost a va in
          write_page a b va vb;
          pages b vb
        end
      in
      pages 0 0);
  {
    column =
      {
        physical = leaf.physical;
        path = leaf.path;
        codec = out.codec;
        values = slots;
        uncompressed_size = !raw;
        compressed_size = out.pos - start;
        data_page;
        dictionary_page;
        stats = stats leaf enc;
      };
    encodings =
      (match enc with
      | Plain -> [ Plain; Rle ]
      | Dictionary _ -> [ Plain; Rle; Rle_dictionary ]);
  }

(* Files *)

let encode ~codec ~row_group ~columns q w =
  if row_group < 1 then invalid_arg "Talon_parquet: row_group must be positive";
  let out =
    { w; codec; pos = 0; page = new_bytes 65536; zipped = new_bytes 0 }
  in
  let groups = ref [] and total = ref 0 in
  let group t =
    let n = Talon.rows t in
    let leaf_list (name, ty) =
      List.map (chunk out) (leaves name ty (Talon.column t name) n)
    in
    groups := (List.concat_map leaf_list columns, n) :: !groups;
    total := !total + n
  in
  let pending = ref [] and held = ref 0 in
  (* Takes every whole row group out of the rows held, once: a batch much
     larger than [row_group] is taken in one pass over it. *)
  let cut () =
    if !held >= row_group then begin
      let all = Talon.of_batches (List.rev !pending) in
      let whole = !held / row_group and rest = !held mod row_group in
      if whole = 1 && rest = 0 then group all
      else begin
        let one =
          Talon.v ~rows:!held
            (List.map (fun (n, _) -> (n, Talon.column all n)) columns)
        in
        let take a b = Talon.take (Nx.arange Nx.int64 a b 1) one in
        for g = 0 to whole - 1 do
          group (take (g * row_group) ((g + 1) * row_group))
        done;
        pending := if rest > 0 then [ take (whole * row_group) !held ] else []
      end;
      if rest = 0 then pending := [];
      held := rest
    end
  in
  let batch () t =
    if Talon.rows t > 0 then begin
      pending := t :: !pending;
      held := !held + Talon.rows t;
      cut ()
    end
  in
  let finish () =
    if !held > 0 then group (Talon.of_batches (List.rev !pending));
    let schema = schema_elements columns in
    let leaf_count =
      Array.fold_left
        (fun n (e : Meta.element) -> if e.physical = None then n else n + 1)
        0 schema
    in
    let footer =
      Meta.footer_bytes ~schema ~row_groups:(List.rev !groups) ~rows:!total
        ~leaves:leaf_count ~created_by
    in
    write_string out footer;
    let n = String.length footer in
    write_string out
      (String.init 4 (fun i -> Char.chr ((n lsr (8 * i)) land 0xFF)));
    write_string out "PAR1";
    W.write_eod w;
    Ok ()
  in
  try
    write_string out "PAR1";
    match Query.fold q ~init:() batch with
    | Error _ as e -> e
    | Ok () -> finish ()
  with
  | Failed msg -> Error (Error.v msg)
  | Bytesrw.Bytes.Stream.Error e ->
      Error (Error.v (Bytesrw.Bytes.Stream.error_message e))
  | Sys_error msg -> Error (Error.v msg)
