(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Id_map = Map.Make (String)

type t = {
  order : Cell.id list;
  by_id : Cell.t Id_map.t;
  metadata : (string * string) list;
}

let empty () = { order = []; by_id = Id_map.empty; metadata = [] }

let of_cells ?(metadata = []) cs =
  let by_id =
    List.fold_left (fun m c -> Id_map.add (Cell.id c) c m) Id_map.empty cs
  in
  let order = List.map Cell.id cs in
  { order; by_id; metadata }

(* ───── Accessors ───── *)

let cells d = List.filter_map (fun id -> Id_map.find_opt id d.by_id) d.order
let length d = List.length d.order
let metadata d = d.metadata
let set_metadata metadata d = { d with metadata }

let nth i d =
  let rec loop j = function
    | [] -> None
    | id :: rest ->
        if j = i then Id_map.find_opt id d.by_id else loop (j + 1) rest
  in
  if i < 0 then None else loop 0 d.order

let find id d = Id_map.find_opt id d.by_id

let find_index id d =
  let rec loop i = function
    | [] -> None
    | hd :: rest -> if String.equal hd id then Some i else loop (i + 1) rest
  in
  loop 0 d.order

(* ───── Modifications ───── *)

let insert ~pos cell d =
  let id = Cell.id cell in
  let by_id = Id_map.add id cell d.by_id in
  let n = List.length d.order in
  let pos = Int.max 0 (Int.min pos n) in
  let rec build i = function
    | rest when i = pos -> id :: rest
    | hd :: rest -> hd :: build (i + 1) rest
    | [] -> [ id ]
  in
  { d with order = build 0 d.order; by_id }

let remove id d =
  if not (Id_map.mem id d.by_id) then d
  else
    let by_id = Id_map.remove id d.by_id in
    let order = List.filter (fun i -> not (String.equal i id)) d.order in
    { d with order; by_id }

let replace id cell d =
  if not (Id_map.mem id d.by_id) then d
  else
    let new_id = Cell.id cell in
    let by_id = Id_map.remove id d.by_id in
    let by_id = Id_map.add new_id cell by_id in
    let order =
      if String.equal id new_id then d.order
      else List.map (fun i -> if String.equal i id then new_id else i) d.order
    in
    { d with order; by_id }

let move id ~pos d =
  match find_index id d with
  | None -> d
  | Some i ->
      let cell = Id_map.find id d.by_id in
      let order = List.filter (fun i -> not (String.equal i id)) d.order in
      let pos = if pos > i then pos - 1 else pos in
      let n = List.length order in
      let pos = Int.max 0 (Int.min pos n) in
      let rec build j = function
        | rest when j = pos -> id :: rest
        | hd :: rest -> hd :: build (j + 1) rest
        | [] -> [ id ]
      in
      { d with order = build 0 order; by_id = Id_map.add id cell d.by_id }

let update id f d =
  match find id d with None -> d | Some c -> replace id (f c) d

let clear_all_outputs d =
  let by_id = Id_map.map (fun c -> Cell.clear_outputs c) d.by_id in
  { d with by_id }
