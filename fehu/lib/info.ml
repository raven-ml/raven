module String_map = Map.Make (String)

type value =
  | Null
  | Bool of bool
  | Int of int
  | Float of float
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
let string s = String s
let list l = List l
let dict d = Dict d

let rec value_to_yojson = function
  | Null -> `Null
  | Bool b -> `Bool b
  | Int i -> `Int i
  | Float f -> `Float f
  | String s -> `String s
  | List values -> `List (List.map value_to_yojson values)
  | Dict fields ->
      `Assoc (List.map (fun (k, v) -> (k, value_to_yojson v)) fields)

let to_yojson info =
  `Assoc (to_list info |> List.map (fun (k, v) -> (k, value_to_yojson v)))

let rec value_of_yojson = function
  | `Null -> Ok Null
  | `Bool b -> Ok (Bool b)
  | `Int i -> Ok (Int i)
  | `Float f -> Ok (Float f)
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
