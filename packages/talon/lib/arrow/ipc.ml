(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Types

(* FlatBuffers *)

type bigbytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

exception Malformed of string

let fail fmt = Printf.ksprintf (fun m -> raise (Malformed m)) fmt

external get16 : bigbytes -> int -> int = "%caml_bigstring_get16"
external get32 : bigbytes -> int -> int32 = "%caml_bigstring_get32"
external get64 : bigbytes -> int -> int64 = "%caml_bigstring_get64"

let word b p = Int32.to_int (get32 b p)

(* OCaml's [int] holds 63 bits: a value past it must not wrap to a valid one. *)
let long b p =
  let v = get64 b p in
  if Int64.of_int (Int64.to_int v) <> v then fail "a 64-bit value overflows";
  Int64.to_int v

(* A table's position in the buffer's bytes [first] to [last - 1]. *)
type t = { b : bigbytes; first : int; last : int; pos : int }

let check t p n =
  if p < t.first || p + n > t.last then fail "a flatbuffer offset is outside it"

let long_at t p =
  check t p 8;
  long t.b p

let u32 t p =
  check t p 4;
  word t.b p land 0xFFFF_FFFF

let target t p =
  let v = u32 t p in
  if v = 0 then fail "a flatbuffer offset is zero";
  p + v

let root b ~first ~last =
  let t = { b; first; last; pos = first } in
  { t with pos = target t first }

(* [slot t id] is the position of field [id] of [t], or [-1] if absent. *)
let slot t id =
  check t t.pos 4;
  let vtable = t.pos - word t.b t.pos in
  let entry = 4 + (2 * id) in
  check t vtable 2;
  if entry + 2 > get16 t.b vtable then -1
  else
    let off = (check t (vtable + entry) 2; get16 t.b (vtable + entry)) in
    if off = 0 then -1 else t.pos + off

let scalar get t id default =
  let p = slot t id in
  if p < 0 then default else get t p

let int8 = scalar (fun t p -> check t p 1; t.b.{p})
let int16 = scalar (fun t p -> check t p 2; get16 t.b p)
let int32 = scalar (fun t p -> check t p 4; word t.b p)
let int64 = scalar long_at
let bool t id d = int8 t id (if d then 1 else 0) <> 0

let field t id =
  let p = slot t id in
  if p < 0 then None else Some (target t p)

let table t id = Option.map (fun pos -> { t with pos }) (field t id)

let vector t p width =
  let n = u32 t p in
  check t (p + 4) (n * width);
  (p + 4, n)

let string t id =
  Option.map
    (fun p ->
      let first, n = vector t p 1 in
      String.init n (fun i -> Char.chr t.b.{first + i}))
    (field t id)

let tables t id =
  match field t id with
  | None -> []
  | Some p ->
      let first, n = vector t p 4 in
      List.init n (fun i -> { t with pos = target t (first + (4 * i)) })

let structs t id ~size =
  match field t id with None -> (0, 0) | Some p -> vector t p size

(* Writing: a table lies at a multiple of 8, its fields in id order at multiples
   of their width, and its children after it, their offsets patched. *)

type v =
  | Absent
  | Byte of int
  | Short of int
  | Int of int
  | Long of int
  | Str of string
  | Tab of v list
  | Tabs of v list list
  | Structs of string list

let width = function
  | Absent -> 0
  | Byte _ -> 1
  | Short _ -> 2
  | Long _ -> 8
  | Int _ | Str _ | Tab _ | Tabs _ | Structs _ -> 4

let round x w = (x + w - 1) land lnot (w - 1)

let emit root =
  let b = Buffer.create 256 and patches = ref [] in
  let word n = Buffer.add_int32_le b (Int32.of_int n) in
  let goto n = while Buffer.length b < n do Buffer.add_char b '\000' done in
  (* [align w after] places the next write so that [after] bytes later the
     buffer is at a multiple of [w]. *)
  let align w after = goto (round (Buffer.length b + after) w - after) in
  let offset at pos = patches := (at, pos) :: !patches in
  let rec table fields =
    let vsize = 4 + (2 * List.length fields) in
    align 8 vsize;
    let vt = Buffer.length b and cur = ref 4 in
    let place f =
      if width f = 0 then 0
      else begin
        let at = round !cur (width f) in
        cur := at + width f;
        at
      end
    in
    let offs = List.map place fields in
    Buffer.add_uint16_le b vsize;
    Buffer.add_uint16_le b !cur;
    List.iter (Buffer.add_uint16_le b) offs;
    let start = Buffer.length b in
    word (start - vt);
    let children = ref [] in
    let put f at =
      if at > 0 then begin
        goto (start + at);
        match f with
        | Absent -> ()
        | Byte v -> Buffer.add_uint8 b (v land 0xFF)
        | Short v -> Buffer.add_int16_le b v
        | Int v -> word v
        | Long v -> Buffer.add_int64_le b (Int64.of_int v)
        | Str _ | Tab _ | Tabs _ | Structs _ ->
            children := (start + at, f) :: !children;
            word 0
      end
    in
    List.iter2 put fields offs;
    goto (start + !cur);
    List.iter (fun (at, f) -> offset at (node f)) (List.rev !children);
    start
  and node v =
    let header n =
      let p = Buffer.length b in
      word n;
      p
    in
    match v with
    | Str s ->
        align 4 0;
        let p = header (String.length s) in
        Buffer.add_string b s;
        Buffer.add_char b '\000';
        p
    | Tab fields -> table fields
    | Structs items ->
        align 8 4;
        let p = header (List.length items) in
        List.iter (Buffer.add_string b) items;
        p
    | Tabs ts ->
        align 4 0;
        let p = header (List.length ts) in
        let base = Buffer.length b in
        List.iter (fun _ -> word 0) ts;
        List.iteri (fun i t -> offset (base + (4 * i)) (table t)) ts;
        p
    | Absent | Byte _ | Short _ | Int _ | Long _ -> assert false
  in
  word 0;
  offset 0 (table root);
  align 8 0;
  let out = Buffer.to_bytes b in
  let patch (at, pos) = Bytes.set_int32_le out at (Int32.of_int (pos - at)) in
  List.iter patch !patches;
  Bytes.unsafe_to_string out

(* Arrow *)

let scalars =
  let int bits signed : arrow = Int { bits; signed } in
  [
    Scalar (Bool, Type.bool, Nx.bit);
    Scalar (int 8 true, Type.int8, Nx.int8);
    Scalar (int 16 true, Type.int16, Nx.int16);
    Scalar (int 32 true, Type.int32, Nx.int32);
    Scalar (int 64 true, Type.int64, Nx.int64);
    Scalar (int 8 false, Type.uint8, Nx.uint8);
    Scalar (int 16 false, Type.uint16, Nx.uint16);
    Scalar (int 32 false, Type.uint32, Nx.uint32);
    Scalar (int 64 false, Type.uint64, Nx.uint64);
    Scalar (Float 16, Type.float16, Nx.float16);
    Scalar (Float 32, Type.float32, Nx.float32);
    Scalar (Float 64, Type.float64, Nx.float64);
  ]

let find_scalar a = List.find_opt (fun (Scalar (b, _, _)) -> a = b) scalars

(* The tags of the union [Type] of Schema.fbs that need no parameter, with the
   name [type_text] prints. Those talon refuses are named so. *)
let nullary =
  [
    (1, Unsupported "null", "null");
    (7, Unsupported "decimal", "decimal");
    (11, Unsupported "interval", "interval");
    (14, Unsupported "union", "union");
    (22, Unsupported "run-end encoded", "run-end encoded");
    (25, Unsupported "list view", "list view");
    (26, Unsupported "large list view", "large list view");
    (4, Binary, "binary");
    (5, Utf8, "utf8");
    (6, Bool, "bool");
    (12, List, "list");
    (13, Struct, "struct");
    (19, Large_binary, "large_binary");
    (20, Large_utf8, "large_utf8");
    (21, Large_list, "large_list");
    (23, Binary_view, "binary_view");
    (24, Utf8_view, "utf8_view");
  ]

(* Reading *)

let str kv id = Option.value (string kv id) ~default:""
let units = [ Type.S; Ms; Us; Ns ]

let unit_of c =
  match List.nth_opt units c with
  | Some u -> u
  | None -> fail "the time unit %d is not one of Arrow's" c
  | exception Invalid_argument _ -> fail "a negative time unit"

let unit_code u = Option.get (List.find_index (( = ) u) units)

let type_of tag t : arrow =
  let t = match t with Some t -> t | None -> fail "a type is absent" in
  match tag with
  | 2 ->
      let bits = int32 t 0 0 in
      if not (List.mem bits [ 8; 16; 32; 64 ]) then
        fail "an integer type has %d bits" bits;
      Int { bits; signed = bool t 1 false }
  | 3 -> (
      match int16 t 0 0 with
      | (0 | 1 | 2) as p -> Float (16 lsl p)
      | p -> fail "the float precision %d is not one of Arrow's" p)
  | 8 -> if int16 t 0 1 = 0 then Date_day else Date_ms
  | 9 ->
      let u = unit_of (int16 t 0 1) and bits = int32 t 1 32 in
      let want = match u with S | Ms -> 32 | Us | Ns -> 64 in
      if bits <> want then fail "a time type has %d bits, not %d" bits want;
      Time u
  | 10 ->
      Timestamp (unit_of (int16 t 0 0), str t 1)
  | 15 -> Fixed_binary (int32 t 0 0)
  | 16 -> Fixed_list (int32 t 0 0)
  | 17 -> Map
  | 18 -> Duration (unit_of (int16 t 0 1))
  | _ -> (
      match List.find_opt (fun (n, _, _) -> n = tag) nullary with
      | Some (_, a, _) -> a
      | None -> Unsupported (Printf.sprintf "type %d" tag))

let extension_name = "ARROW:extension:name"
let extension_metadata = "ARROW:extension:metadata"
let kv_pairs t id = List.map (fun kv -> (str kv 0, str kv 1)) (tables t id)

let encoding_of d =
  let index_bits, index_signed =
    match table d 1 with
    | None -> (32, true)
    | Some i -> (int32 i 0 0, bool i 1 false)
  in
  { id = int64 d 0 0; index_bits; index_signed }

(* [field budget t]: a children vector may repeat an entry, so that a few bytes
   describe a huge tree. Each field takes 4 bytes of a tree, which [budget]
   counts down from the size of the metadata. *)
let rec field budget t =
  if !budget = 0 then fail "the schema has more fields than bytes";
  decr budget;
  let raw = kv_pairs t 6 in
  let meta name = List.assoc_opt name raw in
  let reserved (k, _) = k = extension_name || k = extension_metadata in
  {
    name = str t 0;
    arrow = type_of (int8 t 2 0) (table t 3);
    encoding = Option.map encoding_of (table t 4);
    children = List.map (field budget) (tables t 5);
    metadata = List.filter (fun kv -> not (reserved kv)) raw;
    extension =
      Option.map
        (fun name ->
          (name, Option.value (meta extension_metadata) ~default:""))
        (meta extension_name);
  }

let schema t =
  let budget = ref ((t.last - t.first) / 4) in
  {
    fields = List.map (field budget) (tables t 1);
    metadata = kv_pairs t 2;
    little = int16 t 0 0 = 0;
  }

let batch t =
  let pairs id =
    let p, n = structs t id ~size:16 in
    Array.init n (fun i ->
        let q = p + (16 * i) in
        (long_at t q, long_at t (q + 8)))
  in
  let p, n = structs t 4 ~size:8 in
  let length = int64 t 0 0 in
  if length < 0 then fail "a record batch's length is negative";
  {
    length;
    nodes = pairs 1;
    buffers = pairs 2;
    compression = Option.map (fun c -> int8 c 0 0) (table t 3);
    variadic = List.init n (fun i -> long_at t (p + (8 * i)));
  }

let parse b ~first ~last =
  let t = root b ~first ~last in
  if int16 t 0 0 < 3 then fail "the metadata version is older than V4";
  let h () =
    match table t 2 with
    | Some h -> h
    | None -> fail "a message has no header"
  in
  let dictionary d =
    match table d 1 with
    | None -> fail "a dictionary batch has no data"
    | Some r ->
        Dictionary
          { id = int64 d 0 0; delta = bool d 2 false; batch = batch r }
  in
  let header =
    match int8 t 1 0 with
    | 1 -> Some (Schema (schema (h ())))
    | 2 -> Some (dictionary (h ()))
    | 3 -> Some (Batch (batch (h ())))
    | 4 | 5 -> None
    | k -> fail "the message type %d is not one of Arrow's" k
  in
  let body_len = int64 t 3 0 in
  if body_len < 0 then fail "a message's body length is negative";
  { header; body = last; body_len }

let message b pos limit =
  if pos + 8 > limit then None
  else if word b pos <> -1 then fail "a message has no continuation word"
  else
    let size = word b (pos + 4) land 0xFFFF_FFFF and first = pos + 8 in
    if size = 0 then None
    else begin
      if first + size > limit then fail "a message's metadata is cut short";
      let m = parse b ~first ~last:(first + size) in
      if m.body_len > limit - m.body then fail "a message's body is cut short";
      Some (m, m.body + m.body_len)
    end

let footer b ~first ~last =
  let t = root b ~first ~last in
  let s =
    match table t 1 with
    | Some s -> schema s
    | None -> fail "a file has no schema"
  in
  let blocks id =
    let p, n = structs t id ~size:24 in
    List.init n (fun i -> long_at t (p + (24 * i)))
  in
  (s, blocks 2, blocks 3)

(* Formatting *)

let unit_text (u : Type.unit_) =
  match u with S -> "s" | Ms -> "ms" | Us -> "us" | Ns -> "ns"

let rec type_text f =
  let child () =
    let c = List.hd f.children in
    c.name ^ ": " ^ field_text c
  in
  match f.arrow with
  | Int { bits; signed } ->
      Printf.sprintf "%sint%d" (if signed then "" else "u") bits
  | Float b -> Printf.sprintf "float%d" b
  | Fixed_binary n -> Printf.sprintf "fixed_size_binary[%d]" n
  | Date_day -> "date32"
  | Date_ms -> "date64"
  | Time u ->
      let width = if u = S || u = Ms then 32 else 64 in
      Printf.sprintf "time%d[%s]" width (unit_text u)
  | Timestamp (u, "") -> Printf.sprintf "timestamp[%s]" (unit_text u)
  | Timestamp (u, z) -> Printf.sprintf "timestamp[%s, %s]" (unit_text u) z
  | Duration u -> Printf.sprintf "duration[%s]" (unit_text u)
  | List -> "list<" ^ child () ^ ">"
  | Large_list -> "large_list<" ^ child () ^ ">"
  | Map -> "map<" ^ child () ^ ">"
  | Fixed_list n -> Printf.sprintf "fixed_size_list<%s>[%d]" (child ()) n
  | Struct ->
      let c (c : field) = c.name ^ ": " ^ field_text c in
      "struct<" ^ String.concat ", " (List.map c f.children) ^ ">"
  | Unsupported s -> s
  | a ->
      let _, _, name = List.find (fun (_, b, _) -> a = b) nullary in
      name

and field_text f =
  match (f.encoding, f.extension) with
  | Some e, _ ->
      Printf.sprintf "dictionary<%s, %sint%d>" (type_text f)
        (if e.index_signed then "" else "u")
        e.index_bits
  | None, Some (n, _) -> Printf.sprintf "extension<%s, %s>" n (type_text f)
  | None, None -> type_text f

(* Writing *)

let type_v (a : arrow) : int * v =
  let u x = Short (unit_code x) in
  match a with
  | Int { bits; signed } -> (2, Tab [ Int bits; Byte (Bool.to_int signed) ])
  | Float b -> (3, Tab [ Short (match b with 16 -> 0 | 32 -> 1 | _ -> 2) ])
  | Fixed_binary n -> (15, Tab [ Int n ])
  | Date_day -> (8, Tab [ Short 0 ])
  | Date_ms -> (8, Tab [ Short 1 ])
  | Time x -> (9, Tab [ u x; Int (match x with S | Ms -> 32 | _ -> 64) ])
  | Timestamp (x, z) -> (10, Tab [ u x; (if z = "" then Absent else Str z) ])
  | Duration x -> (18, Tab [ u x ])
  | Fixed_list n -> (16, Tab [ Int n ])
  | Map -> (17, Tab [ Byte 0 ])
  | a ->
      let tag, _, _ = List.find (fun (_, b, _) -> a = b) nullary in
      (tag, Tab [])

let kvs l : v =
  match l with
  | [] -> Absent
  | _ -> Tabs (List.map (fun (k, v) -> [ Str k; Str v ]) l)

let rec field_v f : v list =
  let tag, table = type_v f.arrow in
  let meta =
    match f.extension with
    | None -> f.metadata
    | Some (n, "") -> (extension_name, n) :: f.metadata
    | Some (n, m) ->
        (extension_name, n) :: (extension_metadata, m) :: f.metadata
  in
  let dict : v =
    match f.encoding with
    | None -> Absent
    | Some e ->
        Tab
          [
            Long e.id;
            Tab [ Int e.index_bits; Byte (Bool.to_int e.index_signed) ];
          ]
  in
  [
    Str f.name;
    Byte 1;
    Byte tag;
    table;
    dict;
    Tabs (List.map field_v f.children);
    kvs meta;
  ]

let schema_v s : v list =
  [ Short 0; Tabs (List.map field_v s.fields); kvs s.metadata ]

(* Metadata V5, the current version. *)
let version = 4

let framed tag header ~body =
  let fb = emit [ Short version; Byte tag; header; Long body ] in
  let padded = (String.length fb + 7) land lnot 7 in
  let out = Bytes.make (8 + padded) '\000' in
  Bytes.set_int32_le out 0 0xFFFF_FFFFl;
  Bytes.set_int32_le out 4 (Int32.of_int padded);
  Bytes.blit_string fb 0 out 8 (String.length fb);
  Bytes.unsafe_to_string out

let schema_message s = framed 1 (Tab (schema_v s)) ~body:0

let struct2 (a, b) =
  let s = Bytes.create 16 in
  Bytes.set_int64_le s 0 (Int64.of_int a);
  Bytes.set_int64_le s 8 (Int64.of_int b);
  Bytes.unsafe_to_string s

let batch_message ~dictionary ~length ~nodes ~buffers ~compression ~body =
  let rb : v =
    Tab
      [
        Long length;
        Structs (List.map struct2 nodes);
        Structs (List.map struct2 buffers);
        (match compression with
        | None -> Absent
        | Some c -> Tab [ Byte c; Byte 0 ]);
      ]
  in
  match dictionary with
  | None -> framed 3 rb ~body
  | Some id -> framed 2 (Tab [ Long id; rb; Byte 0 ]) ~body

let footer_bytes s ~dictionaries ~batches =
  let block (offset, meta, body) =
    let b = Bytes.make 24 '\000' in
    Bytes.set_int64_le b 0 (Int64.of_int offset);
    Bytes.set_int32_le b 8 (Int32.of_int meta);
    Bytes.set_int64_le b 16 (Int64.of_int body);
    Bytes.unsafe_to_string b
  in
  emit
    [
      Short version;
      Tab (schema_v s);
      Structs (List.map block dictionaries);
      Structs (List.map block batches);
    ]
