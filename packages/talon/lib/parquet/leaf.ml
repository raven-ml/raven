(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon

type t = {
  rev_path : string list;
  index : int;
  nested : bool;
  optional : bool;
  physical : Meta.physical;
  length : int;
  annotation : Meta.logical option;
  rep : int;
  def : int;
  slot_def : int;
}

type slots = { rep : int; def : int; nonnull : int }

type node =
  | Prim of t
  | Struct of { slots : slots; fields : (string * node) list }
  | List of {
      slots : slots;
      elem_rep : int;
      elem_def : int;
      element : node;
    }

type column = { name : string; node : node; text : string }
type schema = { leaves : t array; columns : column array }

(* Annotations *)

let of_converted (e : Meta.element) : Meta.logical option =
  let integer bits signed = Some (Meta.Integer { bits; signed }) in
  match e.converted with
  | Some 0 -> Some String
  | Some 4 -> Some Enum
  | Some 5 -> Some (Decimal { precision = e.precision; scale = e.scale })
  | Some 6 -> Some Date
  | Some 7 -> Some (Time { unit_ = Ms; utc = true })
  | Some 8 -> Some (Time { unit_ = Us; utc = true })
  | Some 9 -> Some (Timestamp { unit_ = Ms; utc = true })
  | Some 10 -> Some (Timestamp { unit_ = Us; utc = true })
  | Some 11 -> integer 8 false
  | Some 12 -> integer 16 false
  | Some 13 -> integer 32 false
  | Some 14 -> integer 64 false
  | Some 15 -> integer 8 true
  | Some 16 -> integer 16 true
  | Some 17 -> integer 32 true
  | Some 18 -> integer 64 true
  | Some 19 -> Some Json
  | Some 20 -> Some Bson
  | _ -> None

let applies (p : Meta.physical) length : Meta.logical -> bool = function
  | String | Enum | Json | Bson | Geometry | Geography -> p = Byte_array
  | Uuid -> p = Fixed_len_byte_array && length = 16
  | Float16 -> p = Fixed_len_byte_array && length = 2
  | Date | Time { unit_ = Ms; _ } -> p = Int32
  | Time { unit_ = Us | Ns; _ } | Timestamp _ -> p = Int64
  | Integer { bits = 8 | 16 | 32; _ } -> p = Int32
  | Integer { bits = 64; _ } -> p = Int64
  | Decimal { precision; scale } -> (
      1 <= precision && 0 <= scale && scale <= precision
      &&
      match p with
      | Int32 -> precision <= 9
      | Int64 -> precision <= 18
      | Byte_array | Fixed_len_byte_array -> true
      | _ -> false)
  | _ -> false

let annotation (e : Meta.element) physical =
  let l =
    match e.logical with None | Some (Other _) -> of_converted e | l -> l
  in
  match l with Some a when applies physical e.length a -> l | _ -> None

(* Parquet's schema language *)

let pp_unit ppf (u : Type.unit_) =
  Format.pp_print_string ppf
    (match u with
    | S -> "SECONDS"
    | Ms -> "MILLIS"
    | Us -> "MICROS"
    | Ns -> "NANOS")

let pp_annotation ppf : Meta.logical -> unit = function
  | String -> Format.pp_print_string ppf "STRING"
  | Enum -> Format.pp_print_string ppf "ENUM"
  | Json -> Format.pp_print_string ppf "JSON"
  | Bson -> Format.pp_print_string ppf "BSON"
  | Uuid -> Format.pp_print_string ppf "UUID"
  | Float16 -> Format.pp_print_string ppf "FLOAT16"
  | Date -> Format.pp_print_string ppf "DATE"
  | Geometry -> Format.pp_print_string ppf "GEOMETRY"
  | Geography -> Format.pp_print_string ppf "GEOGRAPHY"
  | Time { unit_; utc } -> Format.fprintf ppf "TIME(%a,%b)" pp_unit unit_ utc
  | Timestamp { unit_; utc } ->
      Format.fprintf ppf "TIMESTAMP(%a,%b)" pp_unit unit_ utc
  | Integer { bits; signed } -> Format.fprintf ppf "INTEGER(%d,%b)" bits signed
  | Decimal { precision; scale } ->
      Format.fprintf ppf "DECIMAL(%d,%d)" precision scale
  | Map | List | Null | Variant | Other _ -> assert false

let physical_name (p : Meta.physical) length =
  match p with
  | Boolean -> "boolean"
  | Int32 -> "int32"
  | Int64 -> "int64"
  | Int96 -> "int96"
  | Float -> "float"
  | Double -> "double"
  | Byte_array -> "binary"
  | Fixed_len_byte_array -> Printf.sprintf "fixed_len_byte_array(%d)" length

(* Schemas

   A schema's elements become a tree whose nodes are the columns talon holds:
   leaves, records and lists. Each carries where its values lie among a leaf's
   level entries, in the terms of Dremel's striping: an entry's repetition level
   says which repeated field repeated, and its definition level how many
   optional or repeated fields on its path are present. *)

type raw = { e : Meta.element; kids : raw list }

let parse (es : Meta.element array) =
  let n = Array.length es in
  let rec node i =
    let e = es.(i) in
    match e.physical with
    | Some _ -> ({ e; kids = [] }, i + 1)
    | None ->
        let rec kids k i acc =
          if k = 0 then (List.rev acc, i)
          else if i >= n then
            Meta.fail
              "the schema is malformed: group %S has %d children past its end"
              e.name e.children
          else
            let r, i = node i in
            kids (k - 1) i (r :: acc)
        in
        let ks, next = kids e.children (i + 1) [] in
        ({ e; kids = ks }, next)
  in
  node

let repetition (e : Meta.element) =
  match e.repetition with
  | Some r -> r
  | None ->
      Meta.fail "the schema is malformed: column %S has no repetition" e.name

(* The shape that a group's annotation gives it. A legacy converted type counts
   as its logical type: 3 is LIST, 1 MAP and 2 MAP_KEY_VALUE. *)
let shape (e : Meta.element) =
  match (e.logical, e.converted) with
  | Some List, _ | _, Some 3 -> `List
  | Some Map, _ | _, Some (1 | 2) -> `Map
  | Some Variant, _ -> `Variant
  | _ -> `Record

(* Where a field being converted stands: the levels of the fields above it and
   its own, its path, last name first and sharing its tail with its parent's so
   that a deep schema costs no more than its size, and the definition level of
   the slots its values fill. *)
type ctx = { c_rep : int; c_def : int; c_slot : int; c_path : string list }

let dotted rev_path = String.concat "." (List.rev rev_path)
let name l = dotted l.rev_path

let descend ctx (e : Meta.element) =
  if not (String.is_valid_utf_8 e.name) then
    Meta.fail ~text:e.name "a column's name is not valid UTF-8";
  let ctx = { ctx with c_path = e.name :: ctx.c_path } in
  match repetition e with
  | Required -> ctx
  | Optional -> { ctx with c_def = ctx.c_def + 1 }
  | Repeated -> { ctx with c_rep = ctx.c_rep + 1; c_def = ctx.c_def + 1 }

(* The context of a repeated field's elements, whose slots are its entries. *)
let elements ctx = { ctx with c_slot = ctx.c_def }

let leaf ctx index (e : Meta.element) =
  let physical = Option.get e.physical in
  if e.children > 0 then
    Meta.fail "the schema is malformed: column %S has a type and children"
      (dotted ctx.c_path);
  if e.converted = Some 21 then
    Meta.fail "column %S is an interval, which talon does not read"
      (dotted ctx.c_path);
  let length = if physical = Fixed_len_byte_array then e.length else 0 in
  if physical = Fixed_len_byte_array && length = 0 then
    Meta.fail
      "the schema is malformed: column %S is a fixed_len_byte_array of length 0"
      (dotted ctx.c_path);
  {
    rev_path = ctx.c_path;
    index;
    nested =
      (ctx.c_rep > 0 || match ctx.c_path with _ :: _ :: _ -> true | _ -> false);
    optional = repetition e = Optional;
    physical;
    length;
    annotation = annotation e physical;
    rep = ctx.c_rep;
    def = ctx.c_def;
    slot_def = ctx.c_slot;
  }

(* [distinct names dup] calls [dup n] on the first name [n] of [names] seen
   twice. *)
let distinct names dup =
  let seen = Hashtbl.create 8 in
  List.iter
    (fun n ->
      if Hashtbl.mem seen n then dup n;
      Hashtbl.add seen n ())
    names

(* [nonnull ctx e] is the definition level from which a group [e] of [ctx] holds
   a value: its own when it is optional, and otherwise its slots', since a
   required field has no null of its own. *)
let nonnull ctx (e : Meta.element) =
  if repetition e = Optional then ctx.c_def else ctx.c_slot

let group_annotation (e : Meta.element) =
  match (e.logical, e.converted) with
  | Some List, _ | _, Some 3 -> " (LIST)"
  | Some Map, _ | _, Some 1 -> " (MAP)"
  | _, Some 2 -> " (MAP_KEY_VALUE)"
  | _ -> ""

(* [text b ?name r] adds [r] to [b] in Parquet's schema language, [name] after
   its type. *)
let rec text b ?name r =
  let add = Buffer.add_string b in
  add
    (match repetition r.e with
    | Required -> "required"
    | Optional -> "optional"
    | Repeated -> "repeated");
  let add_name () = Option.iter (fun n -> add (" " ^ n)) name in
  match r.e.physical with
  | Some p ->
      add (" " ^ physical_name p r.e.length);
      add_name ();
      Option.iter
        (fun a -> add (Format.asprintf " (%a)" pp_annotation a))
        (annotation r.e p);
      if name <> None then add ";"
  | None ->
      add " group";
      add_name ();
      add (group_annotation r.e);
      add " {";
      List.iter
        (fun k ->
          add " ";
          text b ~name:k.e.name k)
        r.kids;
      add " }"

let of_schema (es : Meta.element array) =
  let n = Array.length es in
  if n = 0 || es.(0).physical <> None then
    Meta.fail "the schema is malformed: its root is not a group";
  let leaves = ref [] and count = ref 0 in
  let add ctx e =
    let l = leaf ctx !count e in
    incr count;
    leaves := l :: !leaves;
    l
  in
  (* A required or optional field: [ctx] already counts its levels. *)
  let rec body ctx r =
    match r.e.physical with
    | Some _ -> Prim (add ctx r.e)
    | None -> (
        match shape r.e with
        | `Variant ->
            Meta.fail "column %S is a variant, which talon does not read"
              (dotted ctx.c_path)
        | `List -> list_group ctx r
        | `Map -> map_group ctx r
        | `Record -> record ctx r)
  (* A field of a record, or a repeated field read as a required list of
     required elements, as Parquet says of one that LIST or MAP does not
     contain. *)
  and field ctx r =
    let c = descend ctx r.e in
    match repetition r.e with
    | Repeated ->
        if r.e.physical = None && shape r.e <> `Record then
          Meta.fail
            "column %S is a repeated list or map, which Parquet allows only as \
             the element of another list"
            (dotted c.c_path);
        list_of ~nonnull:ctx.c_slot ctx c (body (elements c) r)
    | Required | Optional -> body c r
  and list_of ~nonnull ctx inner element =
    let slots = { rep = ctx.c_rep; def = ctx.c_slot; nonnull } in
    List { slots; elem_rep = inner.c_rep; elem_def = inner.c_def; element }
  and record ctx r =
    if r.kids = [] then
      Meta.fail "the schema is malformed: column %S is a group of no fields"
        (dotted ctx.c_path);
    distinct
      (List.map (fun k -> k.e.name) r.kids)
      (fun n ->
        Meta.fail "column %S has two fields named %S" (dotted ctx.c_path) n);
    let fields = List.map (fun k -> (k.e.name, field ctx k)) r.kids in
    let nonnull = nonnull ctx r.e in
    let slots = { rep = ctx.c_rep; def = ctx.c_slot; nonnull } in
    Struct { slots; fields }
  (* [repeated_child ctx r what] is the one repeated field that a LIST or MAP
     group [r] must hold. *)
  and repeated_child ctx r what =
    match r.kids with
    | [ k ] when repetition k.e = Repeated -> k
    | _ ->
        Meta.fail "column %S is a %s of %d fields, not of one repeated field"
          (dotted ctx.c_path) what (List.length r.kids)
  (* Parquet's rules for the elements of a LIST, in its LogicalTypes.md. *)
  and list_group ctx r =
    let k = repeated_child ctx r "list" in
    let inner = descend ctx k.e in
    let whole () = body (elements inner) k in
    let element =
      match k.kids with
      | [] -> whole () (* a primitive: rule 1 *)
      | _ :: _ :: _ -> whole () (* several fields: rule 2 *)
      | [ only ] when repetition only.e = Repeated -> whole () (* rule 3 *)
      | [ _ ] when k.e.name = "array" || k.e.name = r.e.name ^ "_tuple" ->
          whole () (* rule 4 *)
      | [ only ] -> field (elements inner) only (* rule 5 *)
    in
    list_of ~nonnull:(nonnull ctx r.e) ctx inner element
  (* A map is the list of its key_value group's records, however many of the
     group's fields a file holds and whatever its annotation. *)
  and map_group ctx r =
    let k = repeated_child ctx r "map" in
    if k.e.physical <> None then
      Meta.fail "column %S is a map whose entries are not a group"
        (dotted ctx.c_path);
    let inner = descend ctx k.e in
    let element = record (elements inner) k in
    list_of ~nonnull:(nonnull ctx r.e) ctx inner element
  in
  let children_mismatch () =
    Meta.fail
      "the schema is malformed: its root has %d children and %d other elements"
      es.(0).children (n - 1)
  in
  let root = { c_rep = 0; c_def = 0; c_slot = 0; c_path = [] } in
  let rec top k i acc =
    if k = 0 then (List.rev acc, i)
    else if i >= n then children_mismatch ()
    else
      let r, i = parse es i in
      top (k - 1) i (r :: acc)
  in
  let raws, next = top es.(0).children 1 [] in
  if next <> n then children_mismatch ();
  distinct
    (List.map (fun r -> r.e.name) raws)
    (Meta.fail "two columns are named %S");
  let column r =
    let b = Buffer.create 64 in
    text b r;
    { name = r.e.name; node = field root r; text = Buffer.contents b }
  in
  let columns = List.map column raws in
  { leaves = Array.of_list (List.rev !leaves); columns = Array.of_list columns }

(* Types *)

let holds_bytes l =
  match (l.physical, l.annotation) with
  | _, Some (Float16 | Decimal _) -> false
  | (Byte_array | Fixed_len_byte_array), _ -> true
  | _ -> false

let default l : Type.any option =
  match (l.annotation, l.physical) with
  | Some (Decimal _), _ -> None
  | Some (String | Enum | Json), _ -> Some (Any Type.string)
  | Some Date, _ -> Some (Any Type.date)
  | Some (Time { unit_; _ }), _ -> Some (Any (Type.clock unit_))
  | Some (Timestamp { unit_; utc }), _ ->
      Some (if utc then Any (Type.instant unit_) else Any (Type.wall unit_))
  | Some (Integer { bits; signed }), _ -> (
      match (bits, signed) with
      | 8, true -> Some (Any Type.int8)
      | 16, true -> Some (Any Type.int16)
      | 32, true -> Some (Any Type.int32)
      | 64, true -> Some (Any Type.int64)
      | 8, false -> Some (Any Type.uint8)
      | 16, false -> Some (Any Type.uint16)
      | 32, false -> Some (Any Type.uint32)
      | _ -> Some (Any Type.uint64))
  | Some Float16, _ -> Some (Any Type.float16)
  | _, Boolean -> Some (Any Type.bool)
  | _, Int32 -> Some (Any Type.int32)
  | _, Int64 -> Some (Any Type.int64)
  | _, Int96 -> Some (Any (Type.wall Ns))
  | _, Float -> Some (Any Type.float32)
  | _, Double -> Some (Any Type.float64)
  | _, (Byte_array | Fixed_len_byte_array) -> Some (Any Type.binary)

let reads_as l (Type.Any t) =
  match (l.annotation, t) with
  | Some (Decimal { precision; _ }), _ -> Decimal.reads_as ~precision (Any t)
  | _ -> (
      (match default l with Some (Any d) -> Type.equal d t | None -> false)
      || (holds_bytes l
         && match t with String | Binary | Categorical _ -> true | _ -> false)
      || l.physical = Int96
         && match t with Datetime { clock = Local; _ } -> true | _ -> false)

let rec scales_agree format file (Type.Any t as ty) =
  let scale = function
    | Prim { annotation = Some (Decimal { scale; _ }); _ } -> Some scale
    | _ -> None
  in
  match (format, file, t) with
  | List { element = a; _ }, List { element = b; _ }, List e ->
      scales_agree a b (Any e)
  | Struct { fields = a; _ }, Struct { fields = b; _ }, Record ts
    when List.compare_lengths a b = 0 && List.compare_lengths a ts = 0 ->
      List.for_all2 Fun.id
        (List.map2 (fun (_, a) (_, b) (_, t) -> scales_agree a b t) a b)
        ts
  | _ -> Decimal.scales_agree ty ~format:(scale format) ~file:(scale file)

let rec default_node = function
  | Prim l -> default l
  | List { element; _ } ->
      Option.map
        (fun (Type.Any e) -> Type.Any (Type.list e))
        (default_node element)
  | Struct { fields; _ } ->
      let field (n, node) = Option.map (fun t -> (n, t)) (default_node node) in
      let fields = List.map field fields in
      if List.for_all Option.is_some fields then
        Some (Type.Any (Type.record (List.map Option.get fields)))
      else None

let rec node_reads_as node (Type.Any t) =
  match (node, t) with
  | Prim l, _ -> reads_as l (Any t)
  | List { element; _ }, List e -> node_reads_as element (Any e)
  | Struct { fields; _ }, Record ts ->
      List.compare_lengths fields ts = 0
      && List.for_all2
           (fun (n, node) (m, t) -> String.equal n m && node_reads_as node t)
           fields ts
  | _ -> false

(* Formatting *)

let pp_reads ppf l =
  let str = Format.pp_print_string ppf in
  match (default l, l.annotation) with
  | None, Some (Decimal { precision; _ }) ->
      str (Decimal.reads_text ~precision)
  | None, _ -> str "float64"
  | Some _, _ when holds_bytes l -> str "string, binary or a categorical"
  | Some _, _ when l.physical = Int96 ->
      str "wall-clock datetime of any unit"
  | Some (Any t), _ -> Type.pp ppf t

let rec pp_node_reads ppf = function
  | Prim l -> pp_reads ppf l
  | List { element; _ } -> Format.fprintf ppf "list[%a]" pp_node_reads element
  | Struct { fields; _ } ->
      let pp_field ppf (n, node) =
        Format.fprintf ppf "%a %a" Type.pp_name n pp_node_reads node
      in
      Format.fprintf ppf "record[%a]"
        (Format.pp_print_list
           ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ", ")
           pp_field)
        fields
