(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Id_map = Map.Make (String)

(* ───── Types ───── *)

type cell_status = Idle | Queued | Running

(* ───── History ───── *)

type history = {
  past : Doc.t list;
  future : Doc.t list;
  count : int;
  capacity : int;
}

let empty_history capacity = { past = []; future = []; count = 0; capacity }

let take n xs =
  let rec loop acc n = function
    | [] -> List.rev acc
    | _ when n <= 0 -> List.rev acc
    | x :: rest -> loop (x :: acc) (n - 1) rest
  in
  loop [] n xs

let push_history doc h =
  let past = doc :: h.past in
  if h.count >= h.capacity then
    { h with past = take h.capacity past; future = []; count = h.capacity }
  else { h with past; future = []; count = h.count + 1 }

(* ───── Session ───── *)

type t = {
  doc : Doc.t;
  last_checkpoint : Doc.t;
  statuses : cell_status Id_map.t;
  history : history;
}

let create ?(history_capacity = 100) doc =
  {
    doc;
    last_checkpoint = doc;
    statuses = Id_map.empty;
    history = empty_history history_capacity;
  }

let doc s = s.doc

let cell_status id s =
  match Id_map.find_opt id s.statuses with Some st -> st | None -> Idle

let can_undo s = s.history.past <> []
let can_redo s = s.history.future <> []

(* ───── Document operations ───── *)

let update_source cell_id source s =
  match Doc.find cell_id s.doc with
  | Some c ->
      let doc = Doc.replace cell_id (Cell.set_source source c) s.doc in
      { s with doc }
  | None -> s

let checkpoint s =
  if s.doc == s.last_checkpoint then s
  else
    let history = push_history s.last_checkpoint s.history in
    { s with last_checkpoint = s.doc; history }

let with_history_push f s =
  let s = checkpoint s in
  let history = push_history s.doc s.history in
  let doc = f s.doc in
  { s with doc; last_checkpoint = doc; history }

let insert_cell ~pos cell s = with_history_push (Doc.insert ~pos cell) s
let remove_cell cell_id s = with_history_push (Doc.remove cell_id) s
let move_cell cell_id ~pos s = with_history_push (Doc.move cell_id ~pos) s

let clear_outputs cell_id s =
  let doc = Doc.update cell_id Cell.clear_outputs s.doc in
  { s with doc }

let clear_all_outputs s =
  let doc = Doc.clear_all_outputs s.doc in
  { s with doc }

let set_cell_kind cell_id kind s =
  match Doc.find cell_id s.doc with
  | Some c ->
      with_history_push
        (fun doc ->
          let src = Cell.source c in
          let id = Cell.id c in
          let attrs = Cell.attrs c in
          let c' =
            match kind with
            | `Code -> Cell.code ~id ~attrs src
            | `Text -> Cell.text ~id ~attrs src
          in
          Doc.replace cell_id c' doc)
        s
  | None -> s

let set_cell_attrs cell_id attrs s =
  match Doc.find cell_id s.doc with
  | Some c ->
      with_history_push
        (fun doc -> Doc.replace cell_id (Cell.set_attrs attrs c) doc)
        s
  | None -> s

(* ───── Execution state ───── *)

let mark_running cell_id s =
  { s with statuses = Id_map.add cell_id Running s.statuses }

let mark_queued cell_id s =
  { s with statuses = Id_map.add cell_id Queued s.statuses }

let mark_idle cell_id s =
  { s with statuses = Id_map.add cell_id Idle s.statuses }

let apply_output cell_id output s =
  let doc = Doc.update cell_id (Cell.append_output output) s.doc in
  { s with doc }

let finish_execution cell_id ~success:_ s =
  let doc = Doc.update cell_id Cell.increment_execution_count s.doc in
  { s with doc; statuses = Id_map.add cell_id Idle s.statuses }

(* ───── History ───── *)

let undo s =
  match s.history.past with
  | prev :: rest ->
      let history =
        {
          s.history with
          past = rest;
          future = s.doc :: s.history.future;
          count = s.history.count - 1;
        }
      in
      { s with doc = prev; last_checkpoint = prev; history }
  | [] -> s

let redo s =
  match s.history.future with
  | next :: rest ->
      let history =
        {
          s.history with
          past = s.doc :: s.history.past;
          future = rest;
          count = s.history.count + 1;
        }
      in
      { s with doc = next; last_checkpoint = next; history }
  | [] -> s

(* ───── Reload ───── *)

let reload doc s =
  {
    doc;
    last_checkpoint = doc;
    statuses = Id_map.empty;
    history = empty_history s.history.capacity;
  }
