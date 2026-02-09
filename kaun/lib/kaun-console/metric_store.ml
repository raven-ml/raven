(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun_runlog

type metric = { step : int; epoch : int option; value : float }
type history_point = { step : int; value : float }
type best_value = { step : int; value : float }

type t = {
  table : (string, metric) Hashtbl.t;
  history : (string, history_point list) Hashtbl.t;
  best_min : (string, best_value) Hashtbl.t;
  best_max : (string, best_value) Hashtbl.t;
  mutable max_epoch : int option;
}

let create ?(initial_size = 32) () =
  {
    table = Hashtbl.create initial_size;
    history = Hashtbl.create initial_size;
    best_min = Hashtbl.create initial_size;
    best_max = Hashtbl.create initial_size;
    max_epoch = None;
  }

let clear s =
  Hashtbl.clear s.table;
  Hashtbl.clear s.history;
  Hashtbl.clear s.best_min;
  Hashtbl.clear s.best_max;
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

let update_best_values store ~tag ~step ~value =
  let new_best = { step; value } in
  (* Update minimum *)
  (match Hashtbl.find_opt store.best_min tag with
  | None -> Hashtbl.replace store.best_min tag new_best
  | Some prev ->
      if value < prev.value then Hashtbl.replace store.best_min tag new_best);
  (* Update maximum *)
  match Hashtbl.find_opt store.best_max tag with
  | None -> Hashtbl.replace store.best_max tag new_best
  | Some prev ->
      if value > prev.value then Hashtbl.replace store.best_max tag new_best

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
      (* Update best values *)
      update_best_values store ~tag:s.tag ~step:s.step ~value:s.value;
      (* Append to history *)
      let hp : history_point = { step = s.step; value = s.value } in
      match Hashtbl.find_opt store.history s.tag with
      | None -> Hashtbl.replace store.history s.tag [ hp ]
      | Some hist -> Hashtbl.replace store.history s.tag (hist @ [ hp ]))
    events

let latest_epoch store = store.max_epoch

let latest_metrics store =
  Hashtbl.fold (fun tag metric acc -> (tag, metric) :: acc) store.table []
  |> List.sort (fun (a, _) (b, _) -> String.compare a b)

let history_for_tag store tag =
  Hashtbl.find_opt store.history tag
  |> Option.value ~default:([] : history_point list)
  |> List.map (fun (p : history_point) -> (p.step, p.value))

(* Check if needle is a substring of haystack *)
let contains_substring haystack needle =
  let hlen = String.length haystack in
  let nlen = String.length needle in
  if nlen > hlen then false
  else
    let rec check i =
      if i > hlen - nlen then false
      else if String.sub haystack i nlen = needle then true
      else check (i + 1)
    in
    check 0

(* Determine if a metric prefers lower values (loss-like) or higher values *)
let prefers_lower tag =
  let tag_lower = String.lowercase_ascii tag in
  contains_substring tag_lower "loss" || contains_substring tag_lower "error"

let best_for_tag store tag =
  if prefers_lower tag then Hashtbl.find_opt store.best_min tag
  else Hashtbl.find_opt store.best_max tag

let best_metrics store =
  Hashtbl.fold
    (fun tag _ acc ->
      match best_for_tag store tag with
      | None -> acc
      | Some best -> (tag, best) :: acc)
    store.table []
  |> List.sort (fun (a, _) (b, _) -> String.compare a b)
