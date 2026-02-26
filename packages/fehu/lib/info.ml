module String_map = Map.Make (String)

type t = Value.t String_map.t

let empty = String_map.empty
let is_empty = String_map.is_empty
let set key value info = String_map.add key value info
let find key info = String_map.find_opt key info

let find_exn key info =
  match String_map.find_opt key info with
  | Some v -> v
  | None -> invalid_arg (Printf.sprintf "Info.find_exn: key %S not present" key)

let remove key info = String_map.remove key info
let merge a b = String_map.union (fun _key _left right -> Some right) a b
let to_list info = String_map.bindings info

let of_list kvs =
  List.fold_left (fun acc (k, v) -> String_map.add k v acc) String_map.empty kvs

let to_value info = Value.Dict (String_map.bindings info)

let pp ppf t =
  let bindings = String_map.bindings t in
  Format.fprintf ppf "{";
  List.iteri
    (fun i (k, v) ->
      if i > 0 then Format.fprintf ppf "; ";
      Format.fprintf ppf "%s: %a" k Value.pp v)
    bindings;
  Format.fprintf ppf "}"

(* Convenience constructors *)

let null = Value.Null
let bool b = Value.Bool b
let int i = Value.Int i
let float f = Value.Float f
let string s = Value.String s
let int_array arr = Value.Int_array (Array.copy arr)
let float_array arr = Value.Float_array (Array.copy arr)
let bool_array arr = Value.Bool_array (Array.copy arr)
