(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon_next

type t = {
  name : string;
  optional : bool;
  physical : Meta.physical;
  length : int;
  annotation : Meta.logical option;
}

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

(* Schemas *)

let group (e : Meta.element) =
  match (e.logical, e.converted) with
  | Some List, _ | _, Some 3 -> "a list"
  | Some Map, _ | _, Some (1 | 2) -> "a map"
  | Some Variant, _ -> "a variant"
  | _ -> "a record"

let leaf (e : Meta.element) =
  let name = e.name in
  if not (String.is_valid_utf_8 name) then
    Meta.fail ~text:name "a column's name is not valid UTF-8";
  let repetition =
    match e.repetition with
    | Some r -> r
    | None ->
        Meta.fail "the schema is malformed: column %S has no repetition" name
  in
  let physical =
    match e.physical with
    | Some p -> p
    | None ->
        Meta.fail "column %S is %s: talon reads flat Parquet files only" name
          (group e)
  in
  if e.children > 0 then
    Meta.fail "the schema is malformed: column %S has a type and children" name;
  if repetition = Repeated then
    Meta.fail
      "column %S is a repeated field: talon reads flat Parquet files only" name;
  if e.converted = Some 21 then
    Meta.fail "column %S is an interval, which talon does not read" name;
  let length = if physical = Fixed_len_byte_array then e.length else 0 in
  if physical = Fixed_len_byte_array && length = 0 then
    Meta.fail
      "the schema is malformed: column %S is a fixed_len_byte_array of length 0"
      name;
  {
    name;
    optional = repetition = Optional;
    physical;
    length;
    annotation = annotation e physical;
  }

let of_schema (es : Meta.element array) =
  let n = Array.length es in
  if n = 0 || es.(0).physical <> None then
    Meta.fail "the schema is malformed: its root is not a group";
  if es.(0).children <> n - 1 then begin
    (* A group among the root's children would explain the count: refuse it
       first, by its name. *)
    Array.iteri (fun i e -> if i > 0 then ignore (leaf e)) es;
    Meta.fail
      "the schema is malformed: its root has %d children and %d other elements"
      es.(0).children (n - 1)
  end;
  let names = Hashtbl.create n in
  Array.init (n - 1) (fun i ->
      let l = leaf es.(i + 1) in
      if Hashtbl.mem names l.name then
        Meta.fail "two columns are named %S" l.name;
      Hashtbl.add names l.name ();
      l)

(* Types *)

let holds_bytes l =
  match (l.physical, l.annotation) with
  | _, Some (Float16 | Decimal _) -> false
  | (Byte_array | Fixed_len_byte_array), _ -> true
  | _ -> false

let int64_digits = 18

let default l : Type.any option =
  match (l.annotation, l.physical) with
  | Some (Decimal _), _ -> None
  | Some (String | Enum | Json), _ -> Some (Any Type.string)
  | Some Date, _ -> Some (Any Type.date)
  | Some (Time { unit_; _ }), _ -> Some (Any (Type.clock unit_))
  | Some (Timestamp { unit_; utc }), _ ->
      Some (Any (Type.datetime ?zone:(if utc then Some "UTC" else None) unit_))
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
  | _, Int96 -> Some (Any (Type.datetime Ns))
  | _, Float -> Some (Any Type.float32)
  | _, Double -> Some (Any Type.float64)
  | _, (Byte_array | Fixed_len_byte_array) -> Some (Any Type.binary)

let reads_as l (Type.Any t) =
  match (l.annotation, t) with
  | Some (Decimal _), Float64 -> true
  | Some (Decimal { precision; _ }), Int64 -> precision <= int64_digits
  | Some (Decimal _), _ -> false
  | _ -> (
      (match default l with Some (Any d) -> Type.equal d t | None -> false)
      || (holds_bytes l
         && match t with String | Binary | Categorical _ -> true | _ -> false)
      || l.physical = Int96
         && match t with Datetime { zone = None; _ } -> true | _ -> false)

(* Formatting *)

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

let pp ppf l =
  Format.pp_print_string ppf (if l.optional then "optional " else "required ");
  (match l.physical with
  | Boolean -> Format.pp_print_string ppf "boolean"
  | Int32 -> Format.pp_print_string ppf "int32"
  | Int64 -> Format.pp_print_string ppf "int64"
  | Int96 -> Format.pp_print_string ppf "int96"
  | Float -> Format.pp_print_string ppf "float"
  | Double -> Format.pp_print_string ppf "double"
  | Byte_array -> Format.pp_print_string ppf "binary"
  | Fixed_len_byte_array ->
      Format.fprintf ppf "fixed_len_byte_array(%d)" l.length);
  Option.iter (Format.fprintf ppf " (%a)" pp_annotation) l.annotation

let pp_reads ppf l =
  let str = Format.pp_print_string ppf in
  match (default l, l.annotation) with
  | None, Some (Decimal { precision; _ }) when precision <= int64_digits ->
      str "float64 or int64"
  | None, _ -> str "float64"
  | Some _, _ when holds_bytes l -> str "string, binary or a categorical"
  | Some _, _ when l.physical = Int96 ->
      str "datetime of any unit, without a zone"
  | Some (Any t), _ -> Type.pp ppf t
