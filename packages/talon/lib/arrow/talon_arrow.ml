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

(* A cursor walks a batch's field nodes and buffers in schema order. A buffer
   is bytes of the mapped body, or of a fresh allocation when compressed. *)

type slice = { ba : I.bigbytes; first : int; len : int; rig : Rig.Buffer.t }

type cursor = {
  bytes : I.bigbytes;
  host : Rig.Buffer.t;
  batch : batch;
  body : int;
  body_len : int;
  mutable node : int;
  mutable buf : int;
  mutable variadic : int list;
}

let cursor bytes host (m : message) batch =
  {
    bytes;
    host;
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

(* Neither codec expands one byte past a block's 128 KiB, so a length beyond
   that is malformed and allocates nothing. *)
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

(* [stepped c n w] is the offsets of [n] rows of [w] elements. Only here do the
   rows lack a bit of the body each: zero-width elements bound nothing. *)
let stepped c n w =
  if n / 8 > c.body_len then fail "a list states more rows than its body holds";
  if w = 0 then Nx.zeros Nx.int64 [| n + 1 |]
  else Nx.arange Nx.int64 0 (times n w + 1) w

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

(* [outside w lo hi] is where the [int64]s of [w] leave [lo] to [hi - 1]. *)
let outside w lo hi =
  Nx.logical_or (Nx.less_s w lo) (Nx.greater_equal_s w hi)

(* [codes path s n v e] is the [int32] codes of [s]. None wraps to a code. *)
let codes path s n v (e : encoding) =
  if e.index_bits = 32 && e.index_signed then view s Nx.int32 [| n |]
  else
    let index = Int { bits = e.index_bits; signed = e.index_signed } in
    match I.find_scalar index with
    | None -> fail "column %S: dictionary indices of %d bits" path e.index_bits
    | Some (Scalar (_, _, dt)) ->
        let w = Nx.cast Nx.int64 (view s dt [| n |]) in
        refuse_where path v (outside w 0L 0x8000_0000L) "an index too large";
        Nx.cast Nx.int32 w

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
      | Some _, _, _ -> fail "column %S does not read as its format's" path
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
      | None, Fixed_binary w, _ ->
          let s = next_buffer c in
          let child = view s Nx.uint8 [| times n w |] in
          varsize (stepped c n w) (Column.of_tensor child)
      | None, (List | Large_list | Map), List e ->
          let offs = offsets (next_buffer c) n (f.arrow = Large_list) in
          varsize offs (sub (child ()) (Any e))
      | None, Fixed_list w, List e ->
          varsize (stepped c n w) (sub (child ()) (Any e))
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

type dictionary = Strings of string array | Refused of string

type doc = {
  host : Rig.Buffer.t;
  is_file : bool;
  schema : schema;
  messages : (int * message) list;
  dictionaries : dictionary IMap.t;
}

let rows d =
  let count acc (_, (m : message)) =
    match m.header with Some (Batch b) -> acc + b.length | _ -> acc
  in
  List.fold_left count 0 d.messages

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

let rec encoded acc (f : field) =
  match f.encoding with
  | Some e -> (e.id, f) :: acc
  | None -> List.fold_left encoded acc f.children

let is_text (f : field) =
  match f.arrow with Utf8 | Large_utf8 | Utf8_view -> true | _ -> false

(* [dictionaries ...] is each id's dictionary, or why talon cannot hold it. *)
let dictionaries bytes host fields messages =
  let add acc (_, (m : message)) =
    match m.header with
    | Some (Dictionary { id; delta; batch }) -> (
        match List.assoc_opt id fields with
        | Some f when is_text f ->
            let value = { f with encoding = None; extension = None } in
            let c = cursor bytes host m batch in
            let col = read c "dictionary" value (Any Type.string) in
            let next =
              match IMap.find_opt id acc with
              | _ when Column.null_count col > 0 -> Refused "holds a null"
              | Some (Refused _ as r) -> r
              | None -> Strings (Column.values Kind.string col)
              | Some (Strings a) ->
                  let b = Column.values Kind.string col in
                  if delta then Strings (Array.append a b)
                  else if a = b then Strings a
                  else Refused "changes between batches"
            in
            IMap.add id next acc
        | _ -> acc)
    | _ -> acc
  in
  let repeats a =
    List.compare_lengths (List.sort_uniq compare a) a <> 0
  in
  let distinct = function
    | Strings a when repeats (Array.to_list a) -> Refused "holds a string twice"
    | d -> d
  in
  IMap.map distinct (List.fold_left add IMap.empty messages)

let open_ host bytes =
  let is_file, schema, messages = messages bytes in
  if not schema.little then
    fail "the data is big-endian, which talon does not read";
  let fields = List.fold_left encoded [] schema.fields in
  let dictionaries = dictionaries bytes host fields messages in
  { host; is_file; schema; messages; dictionaries }

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

let rec default_of d path (f : field) : Type.any =
  let ext name meta =
    let (Type.Any s) = default_of d path { f with extension = None } in
    Type.Any (Type.ext ~name ~metadata:meta s)
  in
  let dictionary (e : encoding) : Type.any =
    if not (is_text f) then
      fail "column %S is dictionary-encoded over %s, which talon cannot hold"
        path (I.field_text { f with encoding = None });
    match IMap.find_opt e.id d.dictionaries with
    | Some (Strings a) -> Any (Type.categorical a)
    | Some (Refused why) ->
        fail "column %S: its dictionary %s, which talon cannot hold" path why
    | None -> Any (Type.categorical [||])
  in
  match (f.extension, f.encoding) with
  | Some (("arrow.fixed_shape_tensor" as n), meta), None -> (
      match canonical f meta with Some t -> t | None -> ext n meta)
  | Some (name, meta), _ -> ext name meta
  | None, Some e -> dictionary e
  | None, None -> plain d path f

and plain d path (f : field) : Type.any =
  let child (c : field) = default_of d (path ^ "." ^ c.name) c in
  let list c =
    let (Any e) = child c in
    Type.Any (Type.list e)
  in
  match (f.arrow, f.children) with
  | (Utf8 | Large_utf8 | Utf8_view), _ -> Any Type.string
  | (Binary | Large_binary | Binary_view | Fixed_binary _), _ -> Any Type.binary
  | (Date_day | Date_ms), _ -> Any Type.date
  | Time u, _ -> Any (Type.clock u)
  | Timestamp (u, ""), _ -> Any (Type.wall u)
  | Timestamp (u, _), _ -> Any (Type.instant u)
  | Duration u, _ -> Any (Type.duration u)
  | Unsupported s, _ -> fail "column %S is a %s, which talon cannot hold" path s
  | Fixed_list n, [ c ] -> (
      if n < 0 then fail "column %S has a negative size" path;
      let numeric = c.encoding = None && c.extension = None in
      match if numeric then tensor_of c.arrow [| n |] else None with
      | Some t -> t
      | None -> list c)
  | (List | Large_list | Map), [ c ] -> list c
  | (List | Large_list | Map | Fixed_list _), _ ->
      fail "column %S: a list has no element" path
  | Struct, cs ->
      let field (c : field) = (c.name, child c) in
      (try Any (Type.record (List.map field cs))
       with Invalid_argument why -> fail "column %S: %s" path why)
  | _ ->
      let (Scalar (_, t, _)) = Option.get (I.find_scalar f.arrow) in
      Any t

let default d (f : field) = default_of d f.name f

let rec reads_as (f : field) (Type.Any current) (Type.Any t) =
  Type.equal current t
  ||
  match (f.encoding, f.extension, f.arrow, f.children, current, t) with
  | None, None, Fixed_list _, [ c ], Tensor _, List e ->
      let (Scalar (_, s, _)) = Option.get (I.find_scalar c.arrow) in
      reads_as c (Any s) (Any e)
  | _ -> false

let parts ?file d columns =
  let batches =
    List.filter_map
      (fun (pos, (m : message)) ->
        match m.header with Some (Batch b) -> Some (pos, b.length) | _ -> None)
      d.messages
  in
  let part i (pos, length) : Source.part =
    let at = Printf.sprintf "record batch %d: " i in
    let decode bytes =
      let m, b =
        match I.message bytes pos (A1.dim bytes) with
        | Some (({ header = Some (Batch b); _ } as m), _) -> (m, b)
        | _ -> fail "a record batch is not where it was"
      in
      let c = cursor bytes d.host m b in
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
  List.mapi part batches

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

(* [under ty v c] is [c], null also where the record above it is. *)
and under ty v c =
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
type column = { field : field; sniffed : Type.any; ty : Type.any }

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

let columns_of f =
  Array.to_list (Array.map (fun c -> (c.field.name, c.ty)) f.columns)

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
  (try ignore (Schema.v (columns_of f))
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
         name (arrow_name c.field) (type_name c.ty) (type_name ty));
  replaced f i { c with ty }

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
    Format.fprintf ppf "@,  %a ← %s" Schema.pp
      (Schema.v [ (c.field.name, c.ty) ])
      (I.field_text c.field)
  in
  Array.iter line f.columns;
  Format.fprintf ppf "@]"

(* A part reads its record batch when opened, so it maps the bytes then. *)
let parts_of ?file host f (r : Source.request) bytes =
  let d = open_ host bytes in
  let s = d.schema in
  let requested name =
    let ty = f.columns.(index "source" f name).ty in
    match List.find_opt (fun (g : field) -> g.name = name) s.fields with
    | None -> I.fail "the file has no column %S" name
    | Some g when not (reads_as g (default d g) ty) ->
        I.fail "column %S reads as %s, not as the format's %s" name
          (type_name (default d g))
          (type_name ty)
    | Some g -> (g, ty)
  in
  parts ?file d (List.map requested r.columns)

let make ?file ?rows f b =
  let name =
    match file with
    | None -> "arrow"
    | Some p -> Format.asprintf "arrow %a" Type.pp_quoted p
  in
  Source.v ~name ~schema:(Schema.v (columns_of f)) ?rows (fun r ->
      opened ?file b (fun host bytes -> parts_of ?file host f r bytes))

let source f b = make f b

let file ?format path =
  match Rig_disk.of_file path with
  | Error why -> Error (Error.v why)
  | Ok b ->
      opened ~file:path b (fun host bytes ->
          let d = open_ host bytes in
          let f = Option.value format ~default:(sniffed d) in
          make ~file:path ~rows:(rows d) f b)

let format ?(framing = File) ?(compression = Uncompressed)
    ?(batch = default_batch) ?(metadata = []) schema =
  if batch < 1 then invalid_arg "Talon_arrow.format: batch must be positive";
  check_keys "format" metadata;
  let next = ref 0 in
  let column (name, ty) =
    { field = field_of next name ty; sniffed = ty; ty }
  in
  let columns = Array.of_list (List.map column (Schema.columns schema)) in
  { framing; compression; batch; metadata; columns }

let codec = function
  | Uncompressed -> None
  | Lz4_frame -> Some 0
  | Zstd -> Some 1

(* [shape f] is [f] without what does not change how it writes. *)
let rec shape (f : field) : field =
  {
    f with
    metadata = [];
    encoding = Option.map (fun (e : encoding) -> { e with id = 0 }) f.encoding;
    children = List.map shape f.children;
  }

let encode f q w =
  let wanted = Schema.v (columns_of f) in
  if not (Schema.equal (Query.schema q) wanted) then
    invalid_arg
      (Format.asprintf
         "Talon_arrow.encode: the query's columns do not write as the \
          format's: %a, against %a"
         Schema.pp (Query.schema q) Schema.pp wanted);
  let check c =
    let written = field_of (ref 0) c.field.name c.ty in
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
    (List.map (fun c -> (c.field, c.ty)) columns)
    q w
