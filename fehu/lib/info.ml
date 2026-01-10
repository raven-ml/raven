(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module String_map = Map.Make (String)

type value =
  | Null
  | Bool of bool
  | Int of int
  | Float of float
  | Int_array of int array
  | Float_array of float array
  | Bool_array of bool array
  | String of string
  | List of value list
  | Dict of (string * value) list

type t = value String_map.t

let empty = String_map.empty
let is_empty = String_map.is_empty
let singleton key value = String_map.singleton key value
let set key value info = String_map.add key value info

let update key f info =
  let current = String_map.find_opt key info in
  match f current with
  | None -> String_map.remove key info
  | Some value -> String_map.add key value info

let find key info = String_map.find_opt key info

let get_exn key info =
  match find key info with
  | Some value -> value
  | None -> invalid_arg (Printf.sprintf "Info key '%s' not present" key)

let merge lhs rhs =
  String_map.union (fun _key _left right -> Some right) lhs rhs

let to_list info = String_map.bindings info

let of_list kvs =
  List.fold_left (fun acc (k, v) -> String_map.add k v acc) String_map.empty kvs

let null = Null
let bool b = Bool b
let int i = Int i
let float f = Float f
let int_array arr = Int_array (Array.copy arr)
let float_array arr = Float_array (Array.copy arr)
let bool_array arr = Bool_array (Array.copy arr)
let string s = String s
let list l = List l
let dict d = Dict d

let parse_float_opt lit =
  try Some (float_of_string lit) with Failure _ -> None

let parse_int_opt lit = try Some (int_of_string lit) with Failure _ -> None
let float_token = "__float__"
let float_array_token = "__float_array__"
let int_array_token = "__int_array__"
let bool_array_token = "__bool_array__"

let encode_special_float = function
  | f when classify_float f = FP_nan -> `String "NaN"
  | f when classify_float f = FP_infinite && f > 0. -> `String "Infinity"
  | f when classify_float f = FP_infinite && f < 0. -> `String "-Infinity"
  | _ -> assert false

let float_to_yojson f =
  match classify_float f with
  | FP_normal | FP_subnormal | FP_zero -> `Float f
  | _ -> `Assoc [ (float_token, encode_special_float f) ]

let rec value_to_yojson = function
  | Null -> `Null
  | Bool b -> `Bool b
  | Int i -> `Int i
  | Float f -> float_to_yojson f
  | Int_array arr ->
      `Assoc
        [
          ( int_array_token,
            `List (Array.to_list arr |> List.map (fun i -> `Int i)) );
        ]
  | Float_array arr ->
      `Assoc
        [
          ( float_array_token,
            `List
              (Array.to_list arr
              |> List.map (fun f ->
                     match classify_float f with
                     | FP_normal | FP_subnormal | FP_zero -> `Float f
                     | _ -> encode_special_float f)) );
        ]
  | Bool_array arr ->
      `Assoc
        [
          ( bool_array_token,
            `List (Array.to_list arr |> List.map (fun b -> `Bool b)) );
        ]
  | String s -> `String s
  | List values -> `List (List.map value_to_yojson values)
  | Dict fields ->
      let sorted = List.sort (fun (a, _) (b, _) -> String.compare a b) fields in
      `Assoc (List.map (fun (k, v) -> (k, value_to_yojson v)) sorted)

let to_yojson info =
  let fields = to_list info in
  let sorted = List.sort (fun (a, _) (b, _) -> String.compare a b) fields in
  `Assoc (List.map (fun (k, v) -> (k, value_to_yojson v)) sorted)

let rec value_of_yojson = function
  | `Null -> Ok Null
  | `Bool b -> Ok (Bool b)
  | `Int i -> Ok (Int i)
  | `Float f -> Ok (Float f)
  | `Assoc [ (token, payload) ] when String.equal token float_token -> (
      match payload with
      | `String "NaN" -> Ok (Float Float.nan)
      | `String "Infinity" -> Ok (Float Float.infinity)
      | `String "-Infinity" -> Ok (Float Float.neg_infinity)
      | json ->
          Error
            (Format.asprintf
               "Info.value_of_yojson: invalid special float representation %a"
               Yojson.Safe.pp json))
  | `Assoc [ (token, `List elems) ] when String.equal token float_array_token ->
      let rec loop acc = function
        | [] -> Ok (Float_array (Array.of_list (List.rev acc)))
        | `Float f :: rest -> loop (f :: acc) rest
        | `Int i :: rest -> loop (float_of_int i :: acc) rest
        | `Intlit lit :: rest -> (
            match parse_float_opt lit with
            | Some f -> loop (f :: acc) rest
            | None ->
                Error
                  (Format.asprintf
                     "Info.value_of_yojson: invalid float literal %s" lit))
        | `String "NaN" :: rest -> loop (Float.nan :: acc) rest
        | `String "Infinity" :: rest -> loop (Float.infinity :: acc) rest
        | `String "-Infinity" :: rest -> loop (Float.neg_infinity :: acc) rest
        | json :: _ ->
            Error
              (Format.asprintf
                 "Info.value_of_yojson: invalid float array element %a"
                 Yojson.Safe.pp json)
      in
      loop [] elems
  | `Assoc [ (token, `List elems) ] when String.equal token int_array_token ->
      let rec loop acc = function
        | [] -> Ok (Int_array (Array.of_list (List.rev acc)))
        | `Int i :: rest -> loop (i :: acc) rest
        | `Intlit lit :: rest -> (
            match parse_int_opt lit with
            | Some i -> loop (i :: acc) rest
            | None ->
                Error
                  (Format.asprintf
                     "Info.value_of_yojson: invalid int literal %s" lit))
        | json :: _ ->
            Error
              (Format.asprintf
                 "Info.value_of_yojson: invalid int array element %a"
                 Yojson.Safe.pp json)
      in
      loop [] elems
  | `Assoc [ (token, `List elems) ] when String.equal token bool_array_token ->
      let rec loop acc = function
        | [] -> Ok (Bool_array (Array.of_list (List.rev acc)))
        | `Bool b :: rest -> loop (b :: acc) rest
        | json :: _ ->
            Error
              (Format.asprintf
                 "Info.value_of_yojson: invalid bool array element %a"
                 Yojson.Safe.pp json)
      in
      loop [] elems
  | `String s -> Ok (String s)
  | `List values ->
      let rec loop acc = function
        | [] -> Ok (List (List.rev acc))
        | x :: xs -> (
            match value_of_yojson x with
            | Ok v -> loop (v :: acc) xs
            | Error _ as err -> err)
      in
      loop [] values
  | `Assoc fields ->
      let rec loop acc = function
        | [] -> Ok (Dict (List.rev acc))
        | (k, v) :: xs -> (
            match value_of_yojson v with
            | Ok vv -> loop ((k, vv) :: acc) xs
            | Error _ as err -> err)
      in
      loop [] fields
  | json ->
      Error
        (Format.asprintf "Info.value_of_yojson: unsupported JSON value %a"
           Yojson.Safe.pp json)

let of_yojson = function
  | `Assoc fields ->
      let rec loop acc = function
        | [] -> Ok (of_list (List.rev acc))
        | (k, v) :: xs -> (
            match value_of_yojson v with
            | Ok vv -> loop ((k, vv) :: acc) xs
            | Error _ as err -> err)
      in
      loop [] fields
  | json ->
      Error
        (Format.asprintf "Info.of_yojson: expected an object, received %a"
           Yojson.Safe.pp json)
