(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Types
module A1 = Bigarray.Array1
module I = Ipc
module IMap = Map.Make (Int)
module W = Bytesrw.Bytes.Writer

(* Reading *)

let fail = I.fail
let max_numel = Nx_array.Layout.max_numel

(* [with_bytes ?file ?at host f] is [f] of [host]'s bytes for the call: the one
   place where what malformed bytes make [f] raise becomes [Error]. *)
let with_bytes ?file ?(at = "") host f =
  Rig.Buffer.read host Bigarray.int8_unsigned @@ fun bytes ->
  let error msg = Error (Error.v ?file (at ^ msg)) in
  match f bytes with
  | v -> Ok v
  | exception (I.Malformed m | Invalid_argument m) -> error m
  | exception
      (Failure _ | Not_found | Out_of_memory | Stack_overflow
      | Rig.Out_of_memory _) ->
      error "the metadata is malformed or states more than memory holds"

(* An id's dictionary: its chunks, newest first, one per batch since the last
   replacement, so that a delta costs one list cell and earlier states share
   their chunks. [stable] holds while every dictionary the file states for the id
   is a prefix of the next: a delta appends, and a replacement that only extends
   leaves earlier indices meaning what they meant. *)
type dictionary = { chunks : Column.t list; stable : bool }

let chunk c = Talon.v [ ("c", c) ]
let table d = Talon.of_batches (List.rev_map chunk d.chunks)
let whole d = Talon.column (table d) "c"

(* A cursor walks a batch's field nodes and buffers in schema order. A buffer
   is bytes of the mapped body, or of a fresh allocation when compressed. [dicts]
   holds the dictionaries that stand for this batch. *)

type slice = { ba : I.bigbytes; first : int; len : int; rig : Rig.Buffer.t }

type cursor = {
  bytes : I.bigbytes;
  host : Rig.Buffer.t;
  dicts : dictionary IMap.t;
  batch : batch;
  body : int;
  body_len : int;
  mutable node : int;
  mutable buf : int;
  mutable variadic : int list;
}

let cursor bytes host dicts (m : message) batch =
  {
    bytes;
    host;
    dicts;
    batch;
    body = m.body;
    body_len = m.body_len;
    node = 0;
    buf = 0;
    variadic = batch.variadic;
  }

let next_node c =
  if c.node >= Array.length c.batch.nodes then
    fail "a record batch has too few field nodes";
  let ((n, _) as r) = c.batch.nodes.(c.node) in
  c.node <- c.node + 1;
  if n < 0 || n > max_numel then fail "a field node's length is out of bounds";
  r

let fresh n =
  let a = Nx_array.host Nx.uint8 [| n |] in
  (a, Bigarray.array1_of_genarray (Nx_array.bigarray a))

(* A Zstandard block regenerates at most 128 KiB (RFC 8878, Block_Maximum_Size)
   from at least four bytes, and LZ4's length bytes add at most 255 each, so a
   length past 128 KiB per byte is malformed and allocates nothing. *)
let max_expansion = 131_072

let decompress c s n =
  if n < 0 || n / max_expansion > s.len then
    fail "a buffer's uncompressed length is out of bounds";
  let a, dst = fresh n in
  let src = A1.sub s.ba s.first s.len in
  let result =
    match c.batch.compression with
    | Some 0 -> Lz.Lz4.Frame.decode src dst
    | Some 1 -> Lz.Zstd.decode src dst
    | _ -> fail "a batch's compression names an unknown codec"
  in
  match result with
  | Error e -> fail "a buffer does not decompress: %s"
        (Format.asprintf "%a" Lz.pp_error e)
  | Ok { produced; _ } when produced <> n ->
      fail "a buffer decompresses to %d bytes, not %d" produced n
  | Ok _ -> { ba = dst; first = 0; len = n; rig = Nx_array.buffer a }

let next_buffer c =
  if c.buf >= Array.length c.batch.buffers then
    fail "a record batch has too few buffers";
  let offset, length = c.batch.buffers.(c.buf) in
  c.buf <- c.buf + 1;
  if offset < 0 || length < 0 || length > c.body_len - offset then
    fail "a buffer lies outside its message's body";
  let s =
    { ba = c.bytes; first = c.body + offset; len = length; rig = c.host }
  in
  if length = 0 || c.batch.compression = None then s
  else begin
    if length < 8 then fail "a compressed buffer has no length prefix";
    let n = I.long s.ba s.first in
    let data = { s with first = s.first + 8; len = length - 8 } in
    if n = -1 then data else decompress c data n
  end

let need s bytes =
  if s.len < bytes then fail "a buffer has %d bytes of %d" s.len bytes

(* [view s dt shape] is [dt]'s elements of [s], copied only when misaligned. *)
let view s dt shape =
  let bytes = Nx_array.Dtype.bytes dt (Array.fold_left ( * ) 1 shape) in
  need s bytes;
  let layout = Nx_array.Layout.contiguous (Iarray.of_array shape) in
  let a =
    match
      Nx_array.v dt layout (Rig.Buffer.view s.rig ~first:s.first ~length:bytes)
    with
    | a -> a
    | exception Invalid_argument _ ->
        let a, ba = fresh bytes in
        A1.blit (A1.sub s.ba s.first bytes) ba;
        Nx_array.v dt layout (Nx_array.buffer a)
  in
  Nx.Repr.of_array Nx.Host.v a

(* An array of no rows may have no offsets at all. *)
let offsets s n large =
  if n = 0 && s.len = 0 then Nx.zeros Nx.int64 [| 1 |]
  else if large then view s Nx.int64 [| n + 1 |]
  else Nx.cast Nx.int64 (view s Nx.int32 [| n + 1 |])


(* [times n w] is [n * w], malformed past what a layout holds. *)
let times n w =
  if w < 0 || (w > 0 && n > max_numel / w) then
    fail "a fixed size is out of bounds";
  n * w

(* [stepped n w] is the offsets of [n] rows of [w > 0] elements. The elements
   have been read by now, and their bytes bound [n]. *)
let stepped n w = Nx.arange Nx.int64 0 (times n w + 1) w

(* [empty_cell f] is the shape of the cells of [f], when a row of [f] holds no
   bytes at all whatever the number of rows, and [None] otherwise. Such a column
   reads as a tensor of no elements, which takes no memory per row: its row count
   is the one thing the file states about it, so it must not be what allocates. *)
let rec empty_cell (f : field) =
  match (f.encoding, f.arrow, f.children) with
  | None, Fixed_binary 0, _ -> Some [ 0 ]
  | None, Fixed_list 0, _ -> Some [ 0 ]
  | None, Fixed_list w, [ c ] when w > 0 ->
      Option.map (List.cons w) (empty_cell c)
  | None, Struct, cs when List.for_all (fun c -> Option.is_some (empty_cell c)) cs
    ->
      Some [ 0 ]
  | _ -> None

let next_variadic c =
  match c.variadic with
  | k :: rest ->
      c.variadic <- rest;
      if k < 0 || k > Array.length c.batch.buffers then
        fail "a view field has too many data buffers";
      k
  | [] -> fail "a view field has no count of its data buffers"

(* [views c s n] is the offsets and bytes of the [n] views of [s], copied. *)
let views c path s n =
  need s (16 * n);
  let data = Array.init (next_variadic c) (fun _ -> next_buffer c) in
  let at i = s.first + (16 * i) in
  let length i =
    let l = I.word s.ba (at i) in
    if l < 0 then fail "column %S: row %d: a view's length is negative" path i;
    if l > 12 then begin
      let b = I.word s.ba (at i + 8) and off = I.word s.ba (at i + 12) in
      if b < 0 || b >= Array.length data || off < 0 || off + l > data.(b).len
      then fail "column %S: row %d: a view lies outside its data" path i
    end;
    l
  in
  let total = ref 0 in
  for i = 0 to n - 1 do
    total := !total + length i
  done;
  let a, out = fresh !total in
  let d = Nx_array.host Nx.int64 [| n + 1 |] in
  Nx_array.write d (fun g ->
      let off = Bigarray.array1_of_genarray g and o = ref 0 in
      for i = 0 to n - 1 do
        A1.unsafe_set off i (Int64.of_int !o);
        let l = I.word s.ba (at i) in
        let src, from =
          if l <= 12 then (s.ba, at i + 4)
          else
            let b = data.(I.word s.ba (at i + 8)) in
            (b.ba, b.first + I.word s.ba (at i + 12))
        in
        A1.blit (A1.sub src from l) (A1.sub out !o l);
        o := !o + l
      done;
      A1.unsafe_set off n (Int64.of_int !o));
  ( Nx.Repr.of_array Nx.Host.v d,
    Column.of_tensor (Nx.Repr.of_array Nx.Host.v a) )

(* [decimals path ~scale ~width t ba first n valid] is the [n] little-endian
   decimals of [scale], of [width] bytes each, in [ba] from [first], as [t]
   reads them: a [float64] or an [int64]. A row where [valid] is false is not
   read. *)
let decimals (type a) path ~scale ~width (t : a Type.t) ba first n valid :
    Nx.host Nx.packed =
  let w = width in
  let at i = first + (i * w) in
  match t with
  | Float64 -> (
      let digits i =
        if valid i then Decimal.digits ba ~first:(at i) ~len:w ~big:false
        else "0"
      in
      match Decimal.floats n ~scale digits with
      | Ok x -> P x
      | Error why -> fail "column %S: a decimal value %s" path why)
  | Int64 ->
      let a = Nx_array.host Nx.int64 [| n |] in
      Nx_array.write a (fun g ->
          let out = Bigarray.array1_of_genarray g in
          for i = 0 to n - 1 do
            if valid i then begin
              if not (Decimal.unscaled ba ~first:(at i) ~len:w ~big:false out i)
              then
                fail "column %S: row %d: a decimal value is outside the range \
                      of int64"
                  path i
            end
            else A1.unsafe_set out i 0L
          done);
      P (Nx.Repr.of_array Nx.Host.v a)
  | _ -> fail "column %S does not read as its format's type" path

let finish ty path l =
  match Column.of_layout ty l with
  | Ok c -> c
  | Error (row, why) -> fail "column %S: row %d: %s" path row why

let rec skip c (f : field) =
  c.node <- c.node + 1;
  match (f.encoding, f.arrow) with
  | Some _, _ -> c.buf <- c.buf + 2
  | None, (Utf8_view | Binary_view) -> c.buf <- c.buf + 2 + next_variadic c
  | None, (Utf8 | Binary | Large_utf8 | Large_binary) -> c.buf <- c.buf + 3
  | None, (List | Large_list | Map) ->
      c.buf <- c.buf + 2;
      List.iter (skip c) f.children
  | None, (Fixed_list _ | Struct) ->
      c.buf <- c.buf + 1;
      List.iter (skip c) f.children
  | None, _ -> c.buf <- c.buf + 2

let slice_bit s i = s.ba.{s.first + (i / 8)} land (1 lsl (i land 7)) <> 0

(* [refuse_where path v bad why] fails at the first valid row where [bad]. *)
let refuse_where path v bad why =
  let bad =
    match v with None -> bad | Some v -> Nx.logical_and bad (Nx.cast Nx.bool v)
  in
  if Nx.dim 0 bad > 0 && Nx.item [] (Nx.any bad) then
    let row = Int64.to_int (Nx.item [ 0 ] (Nx.positions bad)) in
    fail "column %S: row %d: %s" path row why

(* [no_null_below c path f ~rows valid row] consumes the nodes and buffers of the
   zero-width element [f] of [rows] rows, which hold nothing but their validity,
   and fails at a null element where [valid] holds. [row] names its row. *)
let rec no_null_below c path (f : field) ~rows valid row =
  let m, nulls = next_node c in
  let vs = next_buffer c in
  if m <> rows then fail "column %S: a list's child has the wrong length" path;
  if nulls <> 0 then need vs ((m + 7) / 8);
  let here i = nulls = 0 || slice_bit vs i in
  if nulls <> 0 then
    for i = 0 to m - 1 do
      if valid i && not (here i) then
        fail "column %S: row %d: a tensor's element is null" path (row i)
    done;
  let below i = valid i && here i in
  match (f.arrow, f.children) with
  | Fixed_binary _, _ -> ignore (next_buffer c)
  | Fixed_list 0, _ -> List.iter (skip c) f.children
  | Fixed_list w, [ ch ] ->
      no_null_below c path ch ~rows:(times m w)
        (fun i -> below (i / w))
        (fun i -> row (i / w))
  | _, cs -> List.iter (fun ch -> no_null_below c path ch ~rows:m below row) cs

(* [outside w lo hi] is where the [int64]s of [w] leave [lo] to [hi - 1]. *)
let outside w lo hi =
  Nx.logical_or (Nx.less_s w lo) (Nx.greater_equal_s w hi)

(* [indices path s n e] is the [n] dictionary indices of [s], as [int64]. An
   unsigned index past [int64] wraps negative, outside every dictionary. *)
let indices path s n (e : encoding) =
  match I.find_scalar (Int { bits = e.index_bits; signed = e.index_signed }) with
  | None -> fail "column %S: dictionary indices of %d bits" path e.index_bits
  | Some (Scalar (_, _, dt)) -> Nx.cast Nx.int64 (view s dt [| n |])

(* [codes path s n v e] is the [int32] codes of a categorical. A categorical's
   storage holds [int32] codes, so no index of one wraps to a code. *)
let codes path s n v (e : encoding) =
  if e.index_bits = 32 && e.index_signed then view s Nx.int32 [| n |]
  else
    let w = indices path s n e in
    refuse_where path v (outside w 0L 0x8000_0000L) "an index too large";
    Nx.cast Nx.int32 w

(* [under ty v c] is [c], null also where [v] is clear. *)
let under ty v c =
  let meet o = Some (match o with None -> v | Some o -> Nx.logical_and v o) in
  let l : Column.layout =
    match Column.layout c with
    | Fixed f -> Fixed { f with validity = meet f.validity }
    | Varsize f -> Varsize { f with validity = meet f.validity }
    | Children f -> Children { f with validity = meet f.validity }
  in
  match Column.of_layout ty l with
  | Ok c -> c
  | Error (_, why) -> invalid_arg ("Talon_arrow: " ^ why)

(* [decoded c path ty e n v s] is the rows of a column read as its dictionary's
   values, [ty]: [n] indices of [s] looked up in the dictionary that stands for
   this batch, null where [v] is clear or the dictionary's entry is. A stream may
   state a dictionary only after batches of nulls, so an absent dictionary is an
   empty one, and a valid row of either fails as out of range. *)
let decoded c path (Type.Any t as ty) (e : encoding) n v s =
  let dictionary = Option.map table (IMap.find_opt e.id c.dicts) in
  let size = Option.fold ~none:0 ~some:Talon.rows dictionary in
  let index = indices path s n e in
  refuse_where path v
    (outside index 0L (Int64.of_int size))
    "a dictionary index out of range";
  match dictionary with
  | Some d when size > 0 ->
      (* A null row may hold any index, so it reads the dictionary's first. *)
      let idx =
        match v with
        | None -> index
        | Some v -> Nx.where (Nx.cast Nx.bool v) index (Nx.scalar Nx.int64 0L)
      in
      let taken = Talon.column (Talon.take idx d) "c" in
      Option.fold ~none:taken ~some:(fun v -> under ty v taken) v
  | _ -> Column.of_options t (Array.make n None)

let rec read c path (f : field) (Type.Any t as ty) : Column.t =
  match t with
  | Ext { storage; _ } ->
      let col = read c path { f with extension = None } (Any storage) in
      finish ty path (Column.layout col)
  | _ -> (
      let sub (ch : field) = read c (path ^ "." ^ ch.name) ch in
      let n, nulls = next_node c in
      let vs = next_buffer c in
      let validity =
        if nulls = 0 then None else Some (view vs Nx.bit [| n |])
      in
      let fixed values = finish ty path (Fixed { validity; values }) in
      let flat dt = fixed (P (view (next_buffer c) dt [| n |])) in
      let varsize offsets child =
        finish ty path (Varsize { validity; offsets; child })
      in
      let child () = List.hd f.children in
      match (f.encoding, f.arrow, t) with
      | Some e, _, Categorical _ ->
          fixed (P (codes path (next_buffer c) n validity e))
      | Some e, Decimal { scale; bits; _ }, _ ->
          let raw = decoded c path (Any Type.binary) e n validity (next_buffer c) in
          let cells = Column.options Kind.binary raw and w = bits / 8 in
          let _, out = fresh (times n w) in
          Array.iteri
            (fun i o ->
              match o with
              | None -> ()
              | Some (b : Binary.t) ->
                  let s = (b :> string) in
                  for k = 0 to w - 1 do
                    out.{(i * w) + k} <- Char.code s.[k]
                  done)
            cells;
          let valid i = Option.is_some cells.(i) in
          finish ty path
            (Fixed
               {
                 validity = Column.validity raw;
                 values = decimals path ~scale ~width:w t out 0 n valid;
               })
      | Some e, _, _ -> decoded c path ty e n validity (next_buffer c)
      | None, Decimal { scale; bits; _ }, _ ->
          let s = next_buffer c and w = bits / 8 in
          need s (times n w);
          let valid i = nulls = 0 || slice_bit vs i in
          fixed (decimals path ~scale ~width:w t s.ba s.first n valid)
      | None, (Bool | Int _ | Float _), _ ->
          let (Scalar (_, _, dt)) = Option.get (I.find_scalar f.arrow) in
          flat dt
      | None, Date_day, _ -> flat Nx.int32
      | None, Date_ms, _ ->
          let ms = view (next_buffer c) Nx.int64 [| n |] in
          (* [div_s] truncates: a negative ms past a midnight rounds down. *)
          let q = Nx.div_s ms 86_400_000L in
          let past = Nx.not_equal (Nx.mul_s q 86_400_000L) ms in
          let late = Nx.logical_and (Nx.less_s ms 0L) past in
          let days = Nx.where late (Nx.sub_s q 1L) q in
          refuse_where path validity
            (outside days (-0x8000_0000L) 0x8000_0000L)
            "a date past int32 days";
          fixed (P (Nx.cast Nx.int32 days))
      | None, Time (S | Ms), _ ->
          fixed (P (Nx.cast Nx.int64 (view (next_buffer c) Nx.int32 [| n |])))
      | None, (Time _ | Timestamp _ | Duration _), _ -> flat Nx.int64
      | None, (Utf8 | Binary | Large_utf8 | Large_binary), _ ->
          let s = next_buffer c in
          let large = f.arrow = Large_utf8 || f.arrow = Large_binary in
          let offs = offsets s n large in
          let b = next_buffer c in
          varsize offs (Column.of_tensor (view b Nx.uint8 [| b.len |]))
      | None, (Utf8_view | Binary_view), _ ->
          let offs, bytes = views c path (next_buffer c) n in
          varsize offs bytes
      | None, (Fixed_binary _ | Fixed_list _), Tensor (dt, shape)
        when Option.is_some (empty_cell f) ->
          (match (f.arrow, f.children) with
          | Fixed_binary _, _ -> ignore (next_buffer c)
          | Fixed_list w, [ ch ] when w > 0 ->
              let valid r = nulls = 0 || slice_bit vs r in
              no_null_below c path ch ~rows:(times n w)
                (fun i -> valid (i / w))
                (fun i -> i / w)
          | _ -> List.iter (skip c) f.children);
          fixed (P (Nx.zeros dt (Array.append [| n |] (Iarray.to_array shape))))
      | None, Fixed_binary w, _ ->
          let child = view (next_buffer c) Nx.uint8 [| times n w |] in
          varsize (stepped n w) (Column.of_tensor child)
      | None, (List | Large_list | Map), List e ->
          let offs = offsets (next_buffer c) n (f.arrow = Large_list) in
          varsize offs (sub (child ()) (Any e))
      | None, Fixed_list w, List e ->
          let elements = sub (child ()) (Any e) in
          if Column.length elements <> times n w then
            fail "column %S: a list's child has the wrong length" path;
          varsize (stepped n w) elements
      | None, Struct, Record fs ->
          let field (ch : field) (name, ty) = (name, sub ch ty) in
          let fields = List.map2 field f.children fs in
          finish ty path (Children { validity; length = n; fields })
      | None, Fixed_list w, Tensor (dt, shape) ->
          (* An element may be null only under a null row. *)
          let rows, element_nulls = next_node c in
          let s = next_buffer c in
          if rows <> times n w then
            fail "column %S: a list's child has the wrong length" path;
          if element_nulls <> 0 then begin
            need s ((rows + 7) / 8);
            for i = 0 to rows - 1 do
              let row_valid = nulls = 0 || slice_bit vs (i / w) in
              if (not (slice_bit s i)) && row_valid then
                fail
                  "column %S: row %d: a tensor's element is null. Read it as \
                   a list with with_type."
                  path (i / w)
            done
          end;
          let shape = Array.append [| n |] (Iarray.to_array shape) in
          fixed (P (view (next_buffer c) dt shape))
      | None, _, _ -> fail "column %S does not read as its format's type" path)

type doc = {
  host : Rig.Buffer.t;
  is_file : bool;
  schema : schema;
  batches : (int * int * dictionary IMap.t) list;
  categoricals : string array IMap.t;
}

let rows d = List.fold_left (fun acc (_, n, _) -> acc + n) 0 d.batches

let magic = "ARROW1"

(* [messages bytes] is the framing, the schema and the positioned messages. *)
let messages bytes =
  let n = A1.dim bytes and m = String.length magic in
  let at p = String.init m (fun i -> Char.chr bytes.{p + i}) in
  if n >= (2 * m) + 6 && at 0 = magic && at (n - m) = magic then begin
    let last = n - m - 4 in
    let size = I.word bytes last land 0xFFFF_FFFF in
    if size = 0 || last - size < m then
      fail "the footer's size is out of bounds";
    let schema, dicts, batches = I.footer bytes ~first:(last - size) ~last in
    let read pos =
      if pos < 0 || pos >= n then fail "a block lies outside the file";
      match I.message bytes pos n with
      | Some (m, _) -> (pos, m)
      | None -> fail "a block holds no message"
    in
    (true, schema, List.map read (dicts @ batches))
  end
  else
    let rec walk pos acc =
      match I.message bytes pos n with
      | None -> List.rev acc
      | Some (m, next) -> walk next ((pos, m) :: acc)
    in
    match walk 0 [] with
    | (_, { header = Some (Schema s); _ }) :: rest -> (false, s, rest)
    | _ -> fail "the bytes are neither an Arrow file nor an Arrow stream"

let is_text (f : field) =
  match f.arrow with Utf8 | Large_utf8 | Utf8_view -> true | _ -> false

let tensor_of (arrow : arrow) shape : Type.any option =
  match I.find_scalar arrow with
  | Some (Scalar (a, _, dt)) when a <> Bool ->
      Some (Any (Type.tensor dt shape))
  | _ -> None

(* [json_ints json key] is the integer array at [key] of the object [json]. *)
let json_ints json key =
  let blank c = String.contains " \t\r\n" c in
  let s = String.of_seq (Seq.filter (Fun.negate blank) (String.to_seq json)) in
  let prefix = Printf.sprintf "%S:[" key and n = String.length s in
  let k = String.length prefix in
  let rec find i =
    if i + k > n then None
    else if String.sub s i k <> prefix then find (i + 1)
    else
      let close = String.index_from s (i + k) ']' in
      Some
        (List.map int_of_string
           (List.filter
              (( <> ) "")
              (String.split_on_char ','
                 (String.sub s (i + k) (close - i - k)))))
  in
  find 0

(* [canonical f meta] is the tensor of [f], unless a permutation reorders it. *)
let canonical (f : field) meta =
  let identity shape = function
    | None -> true
    | Some p -> p = List.init (List.length shape) Fun.id
  in
  match (f.arrow, f.children, json_ints meta "shape") with
  | Fixed_list n, [ c ], Some shape
    when c.encoding = None && c.extension = None
         && List.fold_left ( * ) 1 shape = n
         && identity shape (json_ints meta "permutation") ->
      tensor_of c.arrow (Array.of_list shape)
  | _ -> None

(* [default_of cats path f] is the type [f] reads as, [cats] holding the
   dictionary ids that read as categoricals. Below a dictionary, every nested
   dictionary reads as its values, so a dictionary's values are read before the
   categoricals are known. *)
let rec default_of cats path (f : field) : Type.any option =
  let ext name meta =
    Option.map
      (fun (Type.Any s) -> Type.Any (Type.ext ~name ~metadata:meta s))
      (default_of cats path { f with extension = None })
  in
  let dictionary (e : encoding) : Type.any option =
    match IMap.find_opt e.id cats with
    | Some a -> Some (Any (Type.categorical a))
    | None -> plain IMap.empty path { f with encoding = None }
  in
  match (f.extension, f.encoding) with
  | Some (("arrow.fixed_shape_tensor" as n), meta), None -> (
      match canonical f meta with Some t -> Some t | None -> ext n meta)
  | Some (name, meta), _ -> ext name meta
  | None, Some e -> dictionary e
  | None, None -> plain cats path f

and plain cats path (f : field) : Type.any option =
  let child (c : field) = default_of cats (path ^ "." ^ c.name) c in
  let list c = Option.map (fun (Type.Any e) -> Type.Any (Type.list e)) (child c) in
  let empty shape = Type.Any (Type.tensor Nx.uint8 (Array.of_list shape)) in
  match (f.arrow, f.children) with
  | Decimal _, _ -> None
  | Fixed_binary 0, _ -> Some (empty [ 0 ])
  | (Utf8 | Large_utf8 | Utf8_view), _ -> Some (Any Type.string)
  | (Binary | Large_binary | Binary_view | Fixed_binary _), _ ->
      Some (Any Type.binary)
  | (Date_day | Date_ms), _ -> Some (Any Type.date)
  | Time u, _ -> Some (Any (Type.clock u))
  | Timestamp (u, ""), _ -> Some (Any (Type.wall u))
  | Timestamp (u, _), _ -> Some (Any (Type.instant u))
  | Duration u, _ -> Some (Any (Type.duration u))
  | Unsupported s, _ -> fail "column %S is a %s, which talon cannot hold" path s
  | Fixed_list n, [ c ] -> (
      if n < 0 then fail "column %S has a negative size" path;
      let numeric = c.encoding = None && c.extension = None in
      match if numeric then tensor_of c.arrow [| n |] else None with
      | Some t -> Some t
      | None -> (
          match empty_cell f with Some s -> Some (empty s) | None -> list c))
  | (List | Large_list | Map), [ c ] -> list c
  | (List | Large_list | Map | Fixed_list _), _ ->
      fail "column %S: a list has no element" path
  | Struct, cs -> (
      let fields = List.map (fun (c : field) -> (c.name, child c)) cs in
      if List.exists (fun (_, t) -> Option.is_none t) fields then None
      else
        let fields = List.map (fun (n, t) -> (n, Option.get t)) fields in
        try Some (Any (Type.record fields))
        with Invalid_argument why -> fail "column %S: %s" path why)
  | _ ->
      let (Scalar (_, t, _)) = Option.get (I.find_scalar f.arrow) in
      Some (Any t)

(* [encoded path acc f] adds [f] and its descendants that are dictionary-encoded,
   each with its path. *)
let rec encoded path acc (f : field) =
  let acc =
    match f.encoding with Some e -> (e.id, (path, f)) :: acc | None -> acc
  in
  List.fold_left
    (fun acc (c : field) -> encoded (path ^ "." ^ c.name) acc c)
    acc f.children

(* [shape f] is [f] without what does not change how it lays out or writes. *)
let rec shape (f : field) : field =
  {
    f with
    metadata = [];
    encoding = Option.map (fun (e : encoding) -> { e with id = 0 }) f.encoding;
    children = List.map shape f.children;
  }

(* [by_id fields] is each dictionary id's field. Fields may share a dictionary
   ("dictionaries can even be used for multiple fields"), which then needs one
   value type. *)
let by_id fields =
  let value (f : field) =
    { (shape f) with name = ""; encoding = None; extension = None }
  in
  let add acc (id, (path, f)) =
    match IMap.find_opt id acc with
    | Some (other, g) when value g <> value f ->
        fail "columns %S and %S share a dictionary and differ in type" other path
    | Some _ -> acc
    | None -> IMap.add id (path, f) acc
  in
  List.fold_left add IMap.empty fields

(* [prefix a b] is whether the strings of [a] are the first of [b]'s. *)
let prefix a b =
  let a = Column.options Kind.string a and b = Column.options Kind.string b in
  Array.length a <= Array.length b && Array.sub b 0 (Array.length a) = a

(* [dictionaries ...] walks the messages in order: a batch reads under the
   dictionaries that stand when it is reached, a persistent map per batch. In a
   file every dictionary comes before every batch, as its footer lists them, and
   replacing one is malformed, by the IPC format: "only one non-delta dictionary
   batch can be emitted". *)
let dictionaries bytes host ~is_file fields messages =
  let load held id delta m batch =
    match IMap.find_opt id fields with
    | None -> held
    | Some (path, (f : field)) ->
        let value = { f with encoding = None; extension = None } in
        (* A decimal dictionary is held as its bytes: its type is the
           reader's to declare, and each row converts once taken. *)
        let value =
          match value.arrow with
          | Decimal { bits; _ } -> { value with arrow = Fixed_binary (bits / 8) }
          | _ -> value
        in
        let c = cursor bytes host held m batch in
        let ty =
          match plain IMap.empty path value with
          | Some ty -> ty
          | None -> fail "column %S: a dictionary's values hold a decimal" path
        in
        let col = read c path value ty in
        let old = IMap.find_opt id held in
        if is_file && (not delta) && Option.is_some old then
          fail "column %S: a file replaces a dictionary" path;
        let entry =
          match old with
          | None -> { chunks = [ col ]; stable = true }
          | Some o when delta -> { o with chunks = col :: o.chunks }
          | Some o ->
              let stable = is_text f && o.stable && prefix (whole o) col in
              { chunks = [ col ]; stable }
        in
        IMap.add id entry held
  in
  let step (held, batches) (pos, (m : message)) =
    match m.header with
    | Some (Batch b) -> (held, (pos, b.length, held) :: batches)
    | Some (Dictionary { id; delta; batch }) ->
        (load held id delta m batch, batches)
    | _ -> (held, batches)
  in
  let held, batches = List.fold_left step (IMap.empty, []) messages in
  (held, List.rev batches)

(* [categoricals fields held] is the strings of each text dictionary that can be
   a categorical: stable, without a null or a repeat. An id without a dictionary
   has no rows to read, and is a categorical of no strings. *)
let categoricals fields held =
  let add id (_, (f : field)) acc =
    if not (is_text f) then acc
    else
      match IMap.find_opt id held with
      | None -> IMap.add id [||] acc
      | Some d when not d.stable -> acc
      | Some d ->
          let values = whole d in
          if Column.null_count values > 0 then acc
          else
            let a = Column.values Kind.string values in
            let distinct = List.sort_uniq compare (Array.to_list a) in
            if List.compare_length_with distinct (Array.length a) = 0 then
              IMap.add id a acc
            else acc
  in
  IMap.fold add fields IMap.empty

let open_ host bytes =
  let is_file, schema, messages = messages bytes in
  if not schema.little then
    fail "the data is big-endian, which talon does not read";
  let fields =
    by_id
      (List.fold_left
         (fun acc (f : field) -> encoded f.name acc f)
         [] schema.fields)
  in
  let held, batches = dictionaries bytes host ~is_file fields messages in
  { host; is_file; schema; batches; categoricals = categoricals fields held }

let default d (f : field) = default_of d.categoricals f.name f

(* [holding_reads_as f t] is whether [t] declares [f], a field that holds a
   decimal or a child of one: a decimal as {!Decimal.reads_as} says, and a
   list or record of what its children read as. Below it, a text dictionary
   reads as its value type, [string]. *)
let rec holding_reads_as (f : field) (Type.Any t as ty) =
  match default_of IMap.empty f.name f with
  | Some (Any d) -> Type.equal d t
  | None -> (
      match (f.extension, f.arrow, f.children, t) with
      | Some (name, meta), _, _, Ext { name = n; metadata; storage } ->
          String.equal name n && String.equal meta metadata
          && holding_reads_as { f with extension = None } (Any storage)
      | Some _, _, _, _ -> false
      | None, Decimal { precision; _ }, _, _ -> Decimal.reads_as ~precision ty
      | None, (List | Large_list | Map | Fixed_list _), [ c ], List e ->
          holding_reads_as c (Any e)
      | None, Struct, cs, Record fs ->
          List.compare_lengths cs fs = 0
          && List.for_all2
               (fun (c : field) (n, t) ->
                 String.equal c.name n && holding_reads_as c t)
               cs fs
      | _ -> false)

let rec reads_as (f : field) current (Type.Any t as ty) =
  match current with
  | None -> holding_reads_as f ty
  | Some (Type.Any current) -> (
      Type.equal current t
      ||
      match (f.encoding, f.extension, f.arrow, f.children, current, t) with
      | None, None, Fixed_list w, [ c ], Tensor _, List e when w > 0 -> (
          match I.find_scalar c.arrow with
          | Some (Scalar (_, s, _)) -> reads_as c (Some (Any s)) (Any e)
          | None -> false)
      | _ -> false)

(* [same_scales f g t] is [false] iff a leaf of the format's field [f], read as
   [t] from the file's field [g], would be an [int64] of another decimal scale. *)
let rec same_scales (f : field) (g : field) (Type.Any t as ty) =
  let scale (f : field) =
    match f.arrow with Decimal { scale; _ } -> Some scale | _ -> None
  in
  match (t, f.children, g.children) with
  | Ext { storage; _ }, _, _ -> same_scales f g (Any storage)
  | List e, [ a ], [ b ] -> same_scales a b (Any e)
  | Record ts, a, b
    when List.compare_lengths a ts = 0 && List.compare_lengths b ts = 0 ->
      List.for_all2 Fun.id
        (List.map2 (fun a b (_, t) -> same_scales a b t) a b)
        ts
  | _ -> Decimal.scales_agree ty ~format:(scale f) ~file:(scale g)

(* [pp_reads ppf f] formats the types the field [f] reads as, for a field that
   holds a decimal or a child of one. *)
let rec pp_reads ppf (f : field) =
  match default_of IMap.empty f.name f with
  | Some (Any t) -> Type.pp ppf t
  | None -> (
      match (f.arrow, f.children) with
      | Decimal { precision; _ }, _ ->
          Format.pp_print_string ppf (Decimal.reads_text ~precision)
      | (List | Large_list | Map | Fixed_list _), [ c ] ->
          Format.fprintf ppf "list[%a]" pp_reads c
      | Struct, cs ->
          let pp_field ppf (c : field) =
            Format.fprintf ppf "%a %a" Type.pp_name c.name pp_reads c
          in
          Format.fprintf ppf "record[%a]"
            (Format.pp_print_list
               ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ", ")
               pp_field)
            cs
      | _ -> Format.pp_print_string ppf "undeclared")

let parts ?file d columns =
  let part i (pos, length, dicts) : Source.part =
    let at = Printf.sprintf "record batch %d: " i in
    let decode bytes =
      let m, b =
        match I.message bytes pos (A1.dim bytes) with
        | Some (({ header = Some (Batch b); _ } as m), _) -> (m, b)
        | _ -> fail "a record batch is not where it was"
      in
      let c = cursor bytes d.host dicts m b in
      let one (f : field) =
        match List.find_opt (fun (g, _) -> g == f) columns with
        | Some (_, ty) -> Some (f.name, read c f.name f ty)
        | None ->
            skip c f;
            None
      in
      Some (Talon.v ~rows:b.length (List.filter_map one d.schema.fields))
    in
    let unread = ref true in
    let next () =
      if not !unread then Ok None
      else begin
        unread := false;
        with_bytes ?file ~at d.host decode
      end
    in
    { rows = Some length; open_ = (fun () -> Ok { next; close = ignore }) }
  in
  List.mapi part d.batches

(* Writing *)

let plain name arrow : field =
  {
    name;
    arrow;
    encoding = None;
    children = [];
    metadata = [];
    extension = None;
  }

let list_of name arrow child = { (plain name arrow) with children = [ child ] }

(* Strings, lists and dictionaries write with talon's 64-bit offsets. *)

let tensor_field name dt shape =
  let shape = Iarray.to_list shape in
  let item =
    List.find_map
      (fun (Scalar (a, _, d)) ->
        if a <> Bool && Nx_array.Dtype.equal d dt then Some (plain "item" a)
        else None)
      I.scalars
  in
  match item with
  | None ->
      invalid_arg
        (Format.asprintf "Talon_arrow.format: a tensor of %a has no Arrow type"
           Nx.Dtype.pp dt)
  | Some item ->
      let size = List.fold_left ( * ) 1 shape in
      let text = String.concat "," (List.map string_of_int shape) in
      {
        (list_of name (Fixed_list size) item) with
        extension =
          Some ("arrow.fixed_shape_tensor", "{\"shape\":[" ^ text ^ "]}");
      }

let rec field_of next name (Type.Any t) : field =
  let same (Scalar (_, s, _)) = Type.equal s t in
  match (List.find_opt same I.scalars, t) with
  | Some (Scalar (a, _, _)), _ -> plain name a
  | None, Categorical _ ->
      let id = !next in
      incr next;
      {
        (plain name Large_utf8) with
        encoding = Some { id; index_bits = 32; index_signed = true };
      }
  | None, String -> plain name Large_utf8
  | None, Binary -> plain name Large_binary
  | None, Date -> plain name Date_day
  | None, Clock u -> plain name (Time u)
  | None, Duration u -> plain name (Duration u)
  | None, Datetime { unit_; clock = Utc } ->
      plain name (Timestamp (unit_, "UTC"))
  | None, Datetime { unit_; clock = Local } ->
      plain name (Timestamp (unit_, ""))
  | None, List e -> list_of name Large_list (field_of next "item" (Any e))
  | None, Record fs ->
      let field (n, t) = field_of next n t in
      { (plain name Struct) with children = List.map field fs }
  | None, Tensor (dt, shape) -> tensor_field name dt shape
  | None, Ext { storage = Tensor _; _ } ->
      invalid_arg
        "Talon_arrow.format: a tensor under another extension has no Arrow type"
  | None, Ext { name = ext; metadata; storage } ->
      {
        (field_of next name (Any storage)) with
        extension = Some (ext, metadata);
      }
  | None, _ -> invalid_arg "Talon_arrow: a type without an Arrow type"

(* The categoricals of a field, with their dictionary ids. *)
let rec categoricals acc (f : field) (Type.Any t) =
  match (t, f.encoding) with
  | Categorical d, Some e -> (e.id, Iarray.to_array d) :: acc
  | Ext { storage; _ }, _ -> categoricals acc f (Any storage)
  | List e, _ -> categoricals acc (List.hd f.children) (Any e)
  | Record fs, _ ->
      List.fold_left2
        (fun acc c (_, ty) -> categoricals acc c ty)
        acc f.children fs
  | _ -> acc

(* A part is one buffer of a batch's body, from a value or built for it. *)

type part = Absent | Held of I.bigbytes | Values of Nx.host Nx.packed

let reading (Nx.P x) f =
  Nx.Repr.read x @@ fun a ->
  let l = Nx_array.layout a and dt = Nx_array.dtype a in
  let n = Nx_array.Dtype.bytes dt (Nx_array.Layout.numel l) in
  let first = Nx_array.Layout.offset l * Nx_array.Dtype.bits dt / 8 in
  Rig.Buffer.read (Nx_array.buffer a) Bigarray.int8_unsigned (fun b ->
      f (A1.sub b first n))

let new_bytes n = A1.create Bigarray.int8_unsigned Bigarray.c_layout n

let copy x =
  reading x (fun b ->
      let c = new_bytes (A1.dim b) in
      A1.blit b c;
      c)

let part_len = function
  | Absent -> 0
  | Held b -> A1.dim b
  | Values (P x) -> Nx_array.Dtype.bytes (Nx.dtype x) (Nx.numel x)

let with_part p f =
  match p with
  | Absent -> f (new_bytes 0)
  | Held b -> f b
  | Values x -> reading x f

(* [compress codec p] is [p]'s frame after its uncompressed length. *)
let compress codec p =
  if part_len p = 0 then Absent
  else
    with_part p @@ fun src ->
    let n = A1.dim src in
    let bound = if codec = 0 then Lz.Lz4.Frame.bound n else Lz.Zstd.bound n in
    let dst = new_bytes (8 + bound) in
    let z = A1.sub dst 8 bound in
    let frame =
      if codec = 0 then Lz.Lz4.Frame.encode src z else Lz.Zstd.encode src z
    in
    for i = 0 to 7 do
      dst.{i} <- (n lsr (8 * i)) land 0xFF
    done;
    Held (A1.sub dst 0 (8 + Option.get frame))

(* Equal rows write equal bytes: a value under a null is zero, a null row covers
   no element, and a record's fields under its nulls are null. *)

let bit v i = v.{i lsr 3} land (1 lsl (i land 7)) <> 0

(* [dropped offsets v child n] empties a varsize column's null rows. *)
let dropped offsets validity child n =
  let o = copy (Nx.P offsets) in
  let at i = I.long o (8 * i) in
  let covers i = (not (bit validity i)) && at (i + 1) > at i in
  if not (Seq.exists covers (Seq.init n Fun.id)) then (offsets, child)
  else begin
    let kept = ref [] and next = Array.make (n + 1) 0L and m = ref 0 in
    for i = 0 to n - 1 do
      if bit validity i then
        for k = at i to at (i + 1) - 1 do
          kept := Int64.of_int k :: !kept;
          incr m
        done;
      next.(i + 1) <- Int64.of_int !m
    done;
    let idx = Nx.create Nx.int64 [| !m |] (Array.of_list (List.rev !kept)) in
    ( Nx.create Nx.int64 [| n + 1 |] next,
      Talon.column (Talon.take idx (Talon.v [ ("c", child) ])) "c" )
  end

type body = { mutable nodes : (int * int) list; mutable parts : part list }

let node b n nulls = b.nodes <- (n, nulls) :: b.nodes
let push b p = b.parts <- p :: b.parts
let some_nulls v nulls = if nulls > 0 then v else None
let validity_part = function None -> Absent | Some v -> Values (Nx.P v)

(* [fixed b v x n] pushes the validity and [n] rows of [x], zero under nulls. *)
let fixed b v (Nx.P x) n =
  match v with
  | None ->
      push b Absent;
      push b (Values (P x))
  | Some valid ->
      let shape = Array.make (Nx.ndim x) 1 in
      shape.(0) <- n;
      let keep = Nx.reshape shape (Nx.cast Nx.bool valid) in
      push b (Values (P valid));
      push b (Values (P (Nx.where keep x (Nx.zeros_like x))))

(* [varsize b v offsets child n] pushes those two; it is the child, emptied. *)
let varsize b v offsets child n =
  let offsets, child =
    match v with
    | Some valid -> dropped offsets (copy (Nx.P valid)) child n
    | None -> (offsets, child)
  in
  push b (validity_part v);
  push b (Values (Nx.P offsets));
  child

let rec column b (Type.Any t) (f : field) c =
  let n = Column.length c and nulls = Column.null_count c in
  node b n nulls;
  match (t, Column.layout c) with
  | Ext { storage; _ }, _ ->
      b.nodes <- List.tl b.nodes;
      column b (Any storage) f c
  | Tensor _, Fixed { validity; values } ->
      fixed b (some_nulls validity nulls) values n;
      (* A tensor's cells are its one child: a node, no validity, the values. *)
      let cells = List.hd b.parts in
      b.parts <- List.tl b.parts;
      let (Nx.P x) = values in
      node b (Nx.numel x) 0;
      push b Absent;
      push b cells
  | Clock (S | Ms), Fixed { validity; values = P x } ->
      fixed b (some_nulls validity nulls) (Nx.P (Nx.cast Nx.int32 x)) n
  | _, Fixed { validity; values } ->
      fixed b (some_nulls validity nulls) values n
  | (String | Binary), Varsize { validity; offsets; child } -> (
      let child = varsize b (some_nulls validity nulls) offsets child n in
      match Column.layout child with
      | Fixed { values; _ } -> push b (Values values)
      | _ -> invalid_arg "Talon_arrow: a byte string's child is not flat")
  | List e, Varsize { validity; offsets; child } ->
      let child = varsize b (some_nulls validity nulls) offsets child n in
      column b (Any e) (List.hd f.children) child
  | Record fs, Children { validity; fields; _ } ->
      let v = some_nulls validity nulls in
      push b (validity_part v);
      let write (cf : field) (name, ty) =
        let child = List.assoc name fields in
        column b ty cf (match v with Some v -> under ty v child | None -> child)
      in
      List.iter2 write f.children fs
  | _ -> invalid_arg "Talon_arrow: a column does not lay out as its type"

let eos = "\xFF\xFF\xFF\xFF\000\000\000\000"
let magic = "ARROW1"

external get64 : I.bigbytes -> int -> int64 = "%caml_bigstring_get64"

type out = { w : W.t; mutable pos : int }

let write_string o s =
  W.write_string o.w s;
  o.pos <- o.pos + String.length s

(* [write_bytes o b] writes [b] in slices of at most 64 KiB. *)
let write_bytes o b =
  let n = A1.dim b and i = ref 0 and chunk = 65536 in
  while !i < n do
    let len = min chunk (n - !i) in
    let out = Bytes.create len in
    let j = ref 0 in
    while !j + 8 <= len do
      Bytes.set_int64_ne out !j (get64 b (!i + !j));
      j := !j + 8
    done;
    while !j < len do
      Bytes.unsafe_set out !j (Char.unsafe_chr (A1.unsafe_get b (!i + !j)));
      incr j
    done;
    W.write_bytes o.w out;
    i := !i + len
  done;
  o.pos <- o.pos + n

let padded l = (l + 7) land lnot 7

(* [message o compression ~dictionary ~length body] is the block it writes. *)
let message o compression ~dictionary ~length body =
  let parts =
    match compression with
    | None -> List.rev body.parts
    | Some c -> List.rev_map (compress c) body.parts
  in
  let end_, buffers =
    List.fold_left_map
      (fun off p -> (off + padded (part_len p), (off, part_len p)))
      0 parts
  in
  let m =
    I.batch_message ~dictionary ~length ~nodes:(List.rev body.nodes) ~buffers
      ~compression ~body:end_
  in
  let start = o.pos in
  write_string o m;
  let write p =
    with_part p (write_bytes o);
    let n = part_len p in
    if padded n <> n then write_string o (String.make (padded n - n) '\000')
  in
  List.iter write parts;
  (start, String.length m, end_)

let write_query ~file ~compression ~batch ~metadata columns q w =
  let o = { w; pos = 0 } and dict_blocks = ref [] and batch_blocks = ref [] in
  let schema : schema =
    { fields = List.map fst columns; metadata; little = true }
  in
  let write_dict (id, strings) =
    let b = { nodes = []; parts = [] } in
    let values = Column.v Type.string strings in
    column b (Any Type.string) (plain "" Large_utf8) values;
    dict_blocks :=
      message o compression ~dictionary:(Some id)
        ~length:(Array.length strings) b
      :: !dict_blocks
  in
  let flush t =
    let b = { nodes = []; parts = [] } in
    let write ((f : field), ty) = column b ty f (Talon.column t f.name) in
    List.iter write columns;
    batch_blocks :=
      message o compression ~dictionary:None ~length:(Talon.rows t) b
      :: !batch_blocks
  in
  let pending = ref [] and held = ref 0 in
  (* [gathered ()] is the rows held as one canonical batch. *)
  let gathered () =
    let all = Talon.of_batches (List.rev !pending) in
    let col ((f : field), _) = (f.name, Talon.column all f.name) in
    Talon.v ~rows:!held (List.map col columns)
  in
  let cut () =
    if !held >= batch then begin
      let one = gathered () in
      let take a z = Talon.take (Nx.arange Nx.int64 a z 1) one in
      let whole = !held / batch and rest = !held mod batch in
      for g = 0 to whole - 1 do
        flush (take (g * batch) ((g + 1) * batch))
      done;
      pending := (if rest > 0 then [ take (whole * batch) !held ] else []);
      held := rest
    end
  in
  let step () t =
    if Talon.rows t > 0 then begin
      pending := t :: !pending;
      held := !held + Talon.rows t;
      cut ()
    end
  in
  let finish () =
    if !held > 0 then flush (gathered ());
    write_string o eos;
    if file then begin
      let ft =
        I.footer_bytes schema ~dictionaries:(List.rev !dict_blocks)
          ~batches:(List.rev !batch_blocks)
      in
      let size = Bytes.create 4 in
      Bytes.set_int32_le size 0 (Int32.of_int (String.length ft));
      write_string o (ft ^ Bytes.to_string size ^ magic)
    end;
    W.write_eod w;
    Ok ()
  in
  try
    if file then write_string o (magic ^ "\000\000");
    write_string o (I.schema_message schema);
    let add acc (f, ty) = categoricals acc f ty in
    List.iter write_dict (List.sort compare (List.fold_left add [] columns));
    match Query.fold q ~init:() step with
    | Error _ as e -> e
    | Ok () -> finish ()
  with
  | Bytesrw.Bytes.Stream.Error e ->
      Error (Error.v (Bytesrw.Bytes.Stream.error_message e))
  | Sys_error msg -> Error (Error.v msg)

type framing = File | Stream
type compression = Uncompressed | Lz4_frame | Zstd
(* [sniffed] and [ty] are [None] for a decimal that no [with_type] declared. *)
type column = { field : field; sniffed : Type.any option; ty : Type.any option }

type format = {
  framing : framing;
  compression : compression;
  batch : int;
  metadata : (string * string) list;
  columns : column array;
}

let default_batch = 65_536

let opened ?file b f =
  match Rig.Buffer.borrow Rig.host b with
  | exception Sys_error why -> Error (Error.v ?file why)
  | None -> Error (Error.v ?file "the host cannot map the bytes")
  | Some host -> with_bytes ?file host (f host)

(* [declared fn c] is [c]'s name and type, which a decimal must have been
   given. *)
let declared fn c =
  match c.ty with
  | Some ty -> (c.field.name, ty)
  | None ->
      Decimal.undeclared ~fn:("Talon_arrow." ^ fn) ~column:c.field.name
        ~text:(I.field_text c.field)
        ~flat:(c.field.children = [])

let columns_of fn f = Array.to_list (Array.map (declared fn) f.columns)

let sniffed d =
  let s = d.schema in
  let column f =
    let ty = default d f in
    { field = f; sniffed = ty; ty }
  in
  {
    framing = (if d.is_file then File else Stream);
    compression = Uncompressed;
    batch = default_batch;
    metadata = s.metadata;
    columns = Array.of_list (List.map column s.fields);
  }
  |> fun f ->
  (* Only the names matter here: an undeclared decimal has no type yet. *)
  let names = Array.map (fun c -> (c.field.name, Type.Any Type.bool)) f.columns in
  (try ignore (Schema.v (Array.to_list names))
   with Invalid_argument why -> I.fail "%s" why);
  f

let sniff b = opened b (fun host bytes -> sniffed (open_ host bytes))

let index fn f name =
  let same c = String.equal c.field.name name in
  match Array.find_index same f.columns with
  | Some i -> i
  | None ->
      invalid_arg (Printf.sprintf "Talon_arrow.%s: no column %S." fn name)

let type_name (Type.Any t) = Format.asprintf "%a" Type.pp t
let arrow_name = I.field_text

(* [reads_text c] states the types the column [c] reads as, for errors. *)
let reads_text c =
  match c.sniffed with
  | Some ty -> type_name ty
  | None -> Format.asprintf "%a" pp_reads c.field

let replaced f i c =
  let columns = Array.copy f.columns in
  columns.(i) <- c;
  { f with columns }

let with_type name ty f =
  let i = index "with_type" f name in
  let c = f.columns.(i) in
  if not (reads_as c.field c.sniffed ty) then
    invalid_arg
      (Printf.sprintf
         "Talon_arrow.with_type: column %S (%s) reads as %s, not as %s. Cast \
          it once read (Expr.cast)."
         name (arrow_name c.field) (reads_text c) (type_name ty));
  replaced f i { c with ty = Some ty }

let metadata ?column f =
  match column with
  | None -> f.metadata
  | Some name -> f.columns.(index "metadata" f name).field.metadata

let check_keys fn m =
  let reserved (k, _) = String.starts_with ~prefix:"ARROW:extension:" k in
  match List.find_opt reserved m with
  | None -> ()
  | Some (k, _) ->
      invalid_arg (Printf.sprintf "Talon_arrow.%s: reserved key %S." fn k)

let with_metadata ?column m f =
  check_keys "with_metadata" m;
  match column with
  | None -> { f with metadata = m }
  | Some name ->
      let i = index "with_metadata" f name in
      let c = f.columns.(i) in
      replaced f i { c with field = { c.field with metadata = m } }

let pp_format ppf f =
  let n = Array.length f.columns in
  Format.fprintf ppf "@[<v>arrow %s (%d column%s)"
    (match f.framing with File -> "file" | Stream -> "stream")
    n
    (if n = 1 then "" else "s");
  let line c =
    (match c.ty with
    | Some ty ->
        Format.fprintf ppf "@,  %a" Schema.pp (Schema.v [ (c.field.name, ty) ])
    | None -> Format.fprintf ppf "@,  %a undeclared" Type.pp_name c.field.name);
    Format.fprintf ppf " ← %s" (I.field_text c.field)
  in
  Array.iter line f.columns;
  Format.fprintf ppf "@]"

(* A part reads its record batch when opened, so it maps the bytes then. *)
let parts_of ?file host f (r : Source.request) bytes =
  let d = open_ host bytes in
  let s = d.schema in
  let requested name =
    let c = f.columns.(index "source" f name) in
    let _, ty = declared "source" c in
    match List.find_opt (fun (g : field) -> g.name = name) s.fields with
    | None -> I.fail "the file has no column %S" name
    | Some g when not (reads_as g (default d g) ty) ->
        I.fail "column %S reads as %s, not as the format's %s" name
          (match default d g with
          | Some t -> type_name t
          | None -> Format.asprintf "%a" pp_reads g)
          (type_name ty)
    | Some g when not (same_scales c.field g ty) ->
        I.fail "column %S %s" name Decimal.other_scale
    | Some g -> (g, ty)
  in
  parts ?file d (List.map requested r.columns)

let make fn ?file ?rows f b =
  let name =
    match file with
    | None -> "arrow"
    | Some p -> Format.asprintf "arrow %a" Type.pp_quoted p
  in
  Source.v ~name ~schema:(Schema.v (columns_of fn f)) ?rows (fun r ->
      opened ?file b (fun host bytes -> parts_of ?file host f r bytes))

let source f b = make "source" f b

let file ?format path =
  match Rig_disk.of_file path with
  | Error why -> Error (Error.v why)
  | Ok b ->
      (* [make] raises for an undeclared decimal: outside [opened], which turns
         [Invalid_argument] from malformed bytes into [Error]. *)
      Result.map
        (fun (f, rows) -> make "file" ~file:path ~rows f b)
        (opened ~file:path b (fun host bytes ->
             let d = open_ host bytes in
             (Option.value format ~default:(sniffed d), rows d)))

let format ?(framing = File) ?(compression = Uncompressed)
    ?(batch = default_batch) ?(metadata = []) schema =
  if batch < 1 then invalid_arg "Talon_arrow.format: batch must be positive";
  check_keys "format" metadata;
  let next = ref 0 in
  let column (name, ty) =
    { field = field_of next name ty; sniffed = Some ty; ty = Some ty }
  in
  let columns = Array.of_list (List.map column (Schema.columns schema)) in
  { framing; compression; batch; metadata; columns }

let codec = function
  | Uncompressed -> None
  | Lz4_frame -> Some 0
  | Zstd -> Some 1

let encode f q w =
  let wanted = Schema.v (columns_of "encode" f) in
  if not (Schema.equal (Query.schema q) wanted) then
    invalid_arg
      (Format.asprintf
         "Talon_arrow.encode: the query's columns do not write as the \
          format's: %a, against %a"
         Schema.pp (Query.schema q) Schema.pp wanted);
  let check c =
    let _, ty = declared "encode" c in
    let written = field_of (ref 0) c.field.name ty in
    if shape written <> shape c.field then
      invalid_arg
        (Printf.sprintf
           "Talon_arrow.encode: column %S is %s in the format and writes as \
            %s. State the file with format."
           c.field.name (arrow_name c.field) (arrow_name written))
  in
  let columns = Array.to_list f.columns in
  List.iter check columns;
  write_query ~file:(f.framing = File) ~compression:(codec f.compression)
    ~batch:f.batch ~metadata:f.metadata
    (List.map (fun c -> (c.field, snd (declared "encode" c))) columns)
    q w
