(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Record = Map.Make (String)

type tensor = Pack : ('a, 'layout) Rune.t -> tensor

type scalar =
  | Bool of bool
  | Int of int
  | Float of float
  | String of string
  | Json of Yojson.Basic.t

type t =
  | Tensor of tensor
  | Scalar of scalar
  | List of t list
  | Record of t Record.t

let tensor value = Tensor (Pack value)
let scalar value = Scalar value
let bool value = scalar (Bool value)
let int value = scalar (Int value)
let float value = scalar (Float value)
let string value = scalar (String value)
let json value = scalar (Json value)
let list items = List items
let rng key = int (Rune.Rng.to_int key)

let record entries =
  let add_unique map (key, value) =
    if Record.mem key map then
      invalid_arg ("Snapshot.record: duplicate key " ^ key)
    else Record.add key value map
  in
  let record = List.fold_left add_unique Record.empty entries in
  Record record

let is_tensor = function Tensor _ -> true | _ -> false
let is_scalar = function Scalar _ -> true | _ -> false
let is_list = function List _ -> true | _ -> false
let is_record = function Record _ -> true | _ -> false
let get_tensor = function Tensor pack -> Some pack | _ -> None
let get_scalar = function Scalar s -> Some s | _ -> None
let get_list = function List l -> Some l | _ -> None
let get_record = function Record r -> Some r | _ -> None

let iter ?on_tensor ?on_scalar tree =
  let rec aux path = function
    | Tensor tensor -> Option.iter (fun f -> f tensor) on_tensor
    | Scalar scalar -> Option.iter (fun f -> f scalar) on_scalar
    | List items ->
        List.iteri
          (fun idx item -> aux (path @ [ string_of_int idx ]) item)
          items
    | Record record -> Record.iter (fun _key value -> aux path value) record
  in
  aux [] tree

let map_tensors f =
  let rec aux = function
    | Tensor tensor -> Tensor (f tensor)
    | Scalar scalar -> Scalar scalar
    | List items -> List (List.map aux items)
    | Record record -> Record (Record.map aux record)
  in
  aux

let map_scalars f =
  let rec aux = function
    | Tensor tensor -> Tensor tensor
    | Scalar scalar -> Scalar (f scalar)
    | List items -> List (List.map aux items)
    | Record record -> Record (Record.map aux record)
  in
  aux

let fold_tensors f init tree =
  let rec aux acc = function
    | Tensor tensor -> f acc tensor
    | Scalar _ -> acc
    | List items -> List.fold_left aux acc items
    | Record record ->
        Record.fold (fun _key value acc -> aux acc value) record acc
  in
  aux init tree

let fold_scalars f init tree =
  let rec aux acc = function
    | Tensor _ -> acc
    | Scalar scalar -> f acc scalar
    | List items -> List.fold_left aux acc items
    | Record record ->
        Record.fold (fun _key value acc -> aux acc value) record acc
  in
  aux init tree

let rec flatten_tensors ?(prefix = "") = function
  | Tensor tensor -> [ (prefix, tensor) ]
  | Scalar _ -> []
  | List items ->
      List.mapi
        (fun idx item ->
          let key =
            if prefix = "" then Printf.sprintf "[%d]" idx
            else Printf.sprintf "%s[%d]" prefix idx
          in
          flatten_tensors ~prefix:key item)
        items
      |> List.concat
  | Record record ->
      Record.bindings record
      |> List.map (fun (key, value) ->
             let path = if prefix = "" then key else prefix ^ "." ^ key in
             flatten_tensors ~prefix:path value)
      |> List.concat

let rec flatten_scalars ?(prefix = "") = function
  | Tensor _ -> []
  | Scalar scalar -> [ (prefix, scalar) ]
  | List items ->
      List.mapi
        (fun idx item ->
          let key =
            if prefix = "" then Printf.sprintf "[%d]" idx
            else Printf.sprintf "%s[%d]" prefix idx
          in
          flatten_scalars ~prefix:key item)
        items
      |> List.concat
  | Record record ->
      Record.bindings record
      |> List.map (fun (key, value) ->
             let path = if prefix = "" then key else prefix ^ "." ^ key in
             flatten_scalars ~prefix:path value)
      |> List.concat

let scalar_to_yojson = function
  | Bool b -> `Assoc [ ("type", `String "bool"); ("value", `Bool b) ]
  | Int i -> `Assoc [ ("type", `String "int"); ("value", `Int i) ]
  | Float f -> `Assoc [ ("type", `String "float"); ("value", `Float f) ]
  | String s -> `Assoc [ ("type", `String "string"); ("value", `String s) ]
  | Json json -> `Assoc [ ("type", `String "json"); ("value", json) ]

let scalar_of_yojson = function
  | `Assoc fields -> (
      match List.assoc_opt "type" fields with
      | Some (`String "bool") -> (
          match List.assoc_opt "value" fields with
          | Some (`Bool b) -> Bool b
          | _ -> failwith "Snapshot.scalar_of_yojson: invalid bool payload")
      | Some (`String "int") -> (
          match List.assoc_opt "value" fields with
          | Some (`Int i) -> Int i
          | _ -> failwith "Snapshot.scalar_of_yojson: invalid int payload")
      | Some (`String "float") -> (
          match List.assoc_opt "value" fields with
          | Some (`Float f) -> Float f
          | Some (`Int i) -> Float (float_of_int i)
          | _ -> failwith "Snapshot.scalar_of_yojson: invalid float payload")
      | Some (`String "string") -> (
          match List.assoc_opt "value" fields with
          | Some (`String s) -> String s
          | _ -> failwith "Snapshot.scalar_of_yojson: invalid string payload")
      | Some (`String "json") -> (
          match List.assoc_opt "value" fields with
          | Some json -> Json json
          | None -> failwith "Snapshot.scalar_of_yojson: missing json value")
      | _ -> failwith "Snapshot.scalar_of_yojson: missing type field")
  | _ -> failwith "Snapshot.scalar_of_yojson: expected object"

let rec ptree = function
  | Ptree.Tensor (Ptree.P tensor) -> Tensor (Pack tensor)
  | Ptree.List items -> List (List.map ptree items)
  | Ptree.Dict bindings ->
      let record =
        List.fold_left
          (fun acc (key, value) -> Record.add key (ptree value) acc)
          Record.empty bindings
      in
      Record record

let rec to_ptree snapshot =
  let open Result in
  match snapshot with
  | Tensor (Pack tensor) -> Ok (Ptree.tensor tensor)
  | Scalar _ -> Error "Snapshot.to_ptree: encountered scalar node"
  | List items ->
      let rec build acc = function
        | [] -> Ok (List.rev acc)
        | hd :: tl -> (
            match to_ptree hd with
            | Ok pt -> build (pt :: acc) tl
            | Error _ as err -> err)
      in
      build [] items |> Result.map Ptree.list
  | Record record ->
      Record.bindings record
      |> List.fold_left
           (fun acc (key, value) ->
             match acc with
             | Error _ -> acc
             | Ok bindings -> (
                 match to_ptree value with
                 | Ok pt -> Ok ((key, pt) :: bindings)
                 | Error _ as err -> err))
           (Ok [])
      |> Result.map (fun bindings -> Ptree.dict (List.rev bindings))
