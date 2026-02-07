(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun_runlog

type metric = { step : int; epoch : int option; value : float }
type history_point = { step : int; value : float }

type t = {
  table : (string, metric) Hashtbl.t;
  history : (string, history_point list) Hashtbl.t;
  mutable max_epoch : int option;
}

let create ?(initial_size = 32) () =
  {
    table = Hashtbl.create initial_size;
    history = Hashtbl.create initial_size;
    max_epoch = None;
  }

let clear s =
  Hashtbl.clear s.table;
  Hashtbl.clear s.history;
  s.max_epoch <- None

let update_epoch s (epoch : int option) =
  match epoch with
  | None -> ()
  | Some e ->
      s.max_epoch <-
        Some
          (match s.max_epoch with
          | None -> e
          | Some prev -> max prev e)

let should_replace ~(prev : metric) ~(next : metric) =
  (* Prefer higher step. If equal step, prefer higher epoch when present. *)
  if next.step > prev.step then true
  else if next.step < prev.step then false
  else
    match (prev.epoch, next.epoch) with
    | None, None -> false
    | None, Some _ -> true
    | Some _, None -> false
    | Some a, Some b -> b > a

let update store (events : Kaun_runlog.Event.t list) =
  List.iter
    (fun (Event.Scalar s) ->
      update_epoch store s.epoch;
      let next = { step = s.step; epoch = s.epoch; value = s.value } in
      (* Update latest value *)
      (match Hashtbl.find_opt store.table s.tag with
      | None -> Hashtbl.replace store.table s.tag next
      | Some prev ->
          if should_replace ~prev ~next then
            Hashtbl.replace store.table s.tag next);
      (* Append to history *)
      let history_point = { step = s.step; value = s.value } in
      match Hashtbl.find_opt store.history s.tag with
      | None -> Hashtbl.replace store.history s.tag [ history_point ]
      | Some hist ->
          Hashtbl.replace store.history s.tag (hist @ [ history_point ]))
    events

let latest_epoch store = store.max_epoch

let latest_metrics store =
  Hashtbl.fold (fun tag metric acc -> (tag, metric) :: acc) store.table []
  |> List.sort (fun (a, _) (b, _) -> String.compare a b)

let history_for_tag store tag =
  Hashtbl.find_opt store.history tag
  |> Option.value ~default:[]
  |> List.map (fun p -> (p.step, p.value))
