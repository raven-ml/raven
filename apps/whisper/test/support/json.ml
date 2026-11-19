(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let of_string s =
  match Jsont_bytesrw.decode_string Jsont.json s with
  | Ok v -> v
  | Error e -> failwith e

let read path = of_string (In_channel.with_open_bin path In_channel.input_all)

let mems = function
  | Jsont.Object (mems, _) -> List.map (fun ((n, _), v) -> (n, v)) mems
  | _ -> failwith "Json.mems: not an object"

let mem name v =
  match List.assoc_opt name (mems v) with
  | Some v -> v
  | None -> failwith ("Json.mem: no member " ^ name)

let is_null = function Jsont.Null _ -> true | _ -> false

let list = function
  | Jsont.Array (l, _) -> l
  | _ -> failwith "Json.list: not an array"

let float = function
  | Jsont.Number (f, _) -> f
  | _ -> failwith "Json.float: not a number"

let int v =
  let f = float v in
  if Float.is_integer f then int_of_float f
  else failwith "Json.int: not an integer"

let bool = function
  | Jsont.Bool (b, _) -> b
  | _ -> failwith "Json.bool: not a bool"

let string = function
  | Jsont.String (s, _) -> s
  | _ -> failwith "Json.string: not a string"

let ints v = Array.of_list (List.map int (list v))
