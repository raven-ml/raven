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

let json_obj pairs =
  Jsont.Json.object' (List.map (fun (k, v) -> (Jsont.Json.name k, v)) pairs)

let encode_special_float = function
  | f when classify_float f = FP_nan -> Jsont.Json.string "NaN"
  | f when classify_float f = FP_infinite && f > 0. -> Jsont.Json.string "Infinity"
  | f when classify_float f = FP_infinite && f < 0. -> Jsont.Json.string "-Infinity"
  | _ -> assert false

let float_to_json f =
  match classify_float f with
  | FP_normal | FP_subnormal | FP_zero -> Jsont.Json.number f
  | _ -> json_obj [ (float_token, encode_special_float f) ]

let rec value_to_json = function
  | Null -> Jsont.Json.null ()
  | Bool b -> Jsont.Json.bool b
  | Int i -> Jsont.Json.int i
  | Float f -> float_to_json f
  | Int_array arr ->
      json_obj
        [
          ( int_array_token,
            Jsont.Json.list
              (Array.to_list arr |> List.map (fun i -> Jsont.Json.int i)) );
        ]
  | Float_array arr ->
      json_obj
        [
          ( float_array_token,
            Jsont.Json.list
              (Array.to_list arr
              |> List.map (fun f ->
                  match classify_float f with
                  | FP_normal | FP_subnormal | FP_zero -> Jsont.Json.number f
                  | _ -> encode_special_float f)) );
        ]
  | Bool_array arr ->
      json_obj
        [
          ( bool_array_token,
            Jsont.Json.list
              (Array.to_list arr |> List.map (fun b -> Jsont.Json.bool b)) );
        ]
  | String s -> Jsont.Json.string s
  | List values -> Jsont.Json.list (List.map value_to_json values)
  | Dict fields ->
      let sorted = List.sort (fun (a, _) (b, _) -> String.compare a b) fields in
      json_obj (List.map (fun (k, v) -> (k, value_to_json v)) sorted)

let to_json info =
  let fields = to_list info in
  let sorted = List.sort (fun (a, _) (b, _) -> String.compare a b) fields in
  json_obj (List.map (fun (k, v) -> (k, value_to_json v)) sorted)

let json_assoc = function
  | Jsont.Object (mems, _) -> List.map (fun ((n, _), v) -> (n, v)) mems
  | _ -> []

let pp_json json = Format.asprintf "%a" Jsont.Json.pp json

let rec value_of_json = function
  | Jsont.Null _ -> Ok Null
  | Jsont.Bool (b, _) -> Ok (Bool b)
  | Jsont.Number (f, _) ->
      if Float.is_integer f && Float.abs f < 4503599627370496.0 then
        Ok (Int (int_of_float f))
      else Ok (Float f)
  | Jsont.Object (mems, _) as json -> (
      let fields = List.map (fun ((n, _), v) -> (n, v)) mems in
      match fields with
      | [ (token, payload) ] when String.equal token float_token -> (
          match payload with
          | Jsont.String ("NaN", _) -> Ok (Float Float.nan)
          | Jsont.String ("Infinity", _) -> Ok (Float Float.infinity)
          | Jsont.String ("-Infinity", _) -> Ok (Float Float.neg_infinity)
          | j ->
              Error
                (Format.asprintf
                   "Info.value_of_json: invalid special float representation %s"
                   (pp_json j)))
      | [ (token, Jsont.Array (elems, _)) ]
        when String.equal token float_array_token ->
          let rec loop acc = function
            | [] -> Ok (Float_array (Array.of_list (List.rev acc)))
            | Jsont.Number (f, _) :: rest -> loop (f :: acc) rest
            | Jsont.String ("NaN", _) :: rest -> loop (Float.nan :: acc) rest
            | Jsont.String ("Infinity", _) :: rest ->
                loop (Float.infinity :: acc) rest
            | Jsont.String ("-Infinity", _) :: rest ->
                loop (Float.neg_infinity :: acc) rest
            | Jsont.String (lit, _) :: rest -> (
                match parse_float_opt lit with
                | Some f -> loop (f :: acc) rest
                | None ->
                    Error
                      (Format.asprintf
                         "Info.value_of_json: invalid float literal %s" lit))
            | j :: _ ->
                Error
                  (Format.asprintf
                     "Info.value_of_json: invalid float array element %s"
                     (pp_json j))
          in
          loop [] elems
      | [ (token, Jsont.Array (elems, _)) ]
        when String.equal token int_array_token ->
          let rec loop acc = function
            | [] -> Ok (Int_array (Array.of_list (List.rev acc)))
            | Jsont.Number (f, _) :: rest -> loop (int_of_float f :: acc) rest
            | Jsont.String (lit, _) :: rest -> (
                match parse_int_opt lit with
                | Some i -> loop (i :: acc) rest
                | None ->
                    Error
                      (Format.asprintf
                         "Info.value_of_json: invalid int literal %s" lit))
            | j :: _ ->
                Error
                  (Format.asprintf
                     "Info.value_of_json: invalid int array element %s"
                     (pp_json j))
          in
          loop [] elems
      | [ (token, Jsont.Array (elems, _)) ]
        when String.equal token bool_array_token ->
          let rec loop acc = function
            | [] -> Ok (Bool_array (Array.of_list (List.rev acc)))
            | Jsont.Bool (b, _) :: rest -> loop (b :: acc) rest
            | j :: _ ->
                Error
                  (Format.asprintf
                     "Info.value_of_json: invalid bool array element %s"
                     (pp_json j))
          in
          loop [] elems
      | _ ->
          let rec loop acc = function
            | [] -> Ok (Dict (List.rev acc))
            | (k, v) :: xs -> (
                match value_of_json v with
                | Ok vv -> loop ((k, vv) :: acc) xs
                | Error _ as err -> err)
          in
          loop [] (json_assoc json))
  | Jsont.String (s, _) -> Ok (String s)
  | Jsont.Array (values, _) ->
      let rec loop acc = function
        | [] -> Ok (List (List.rev acc))
        | x :: xs -> (
            match value_of_json x with
            | Ok v -> loop (v :: acc) xs
            | Error _ as err -> err)
      in
      loop [] values

let of_json = function
  | Jsont.Object _ as json -> (
      let fields = json_assoc json in
      let rec loop acc = function
        | [] -> Ok (of_list (List.rev acc))
        | (k, v) :: xs -> (
            match value_of_json v with
            | Ok vv -> loop ((k, vv) :: acc) xs
            | Error _ as err -> err)
      in
      loop [] fields)
  | json ->
      Error
        (Format.asprintf "Info.of_json: expected an object, received %s"
           (pp_json json))
