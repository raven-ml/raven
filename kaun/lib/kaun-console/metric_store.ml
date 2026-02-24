(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun_runlog

type metric = { step : int; epoch : int option; value : float }
type history_point = { step : int; value : float }
type best_value = { step : int; value : float }

type tag_data = {
  latest : metric;
  history : history_point list;
  best_min : best_value option;
  best_max : best_value option;
}

type t = {
  by_tag : (string, tag_data) Hashtbl.t;
  mutable max_epoch : int option;
}

let create ?(initial_size = 32) () =
  { by_tag = Hashtbl.create initial_size; max_epoch = None }

let clear s =
  Hashtbl.clear s.by_tag;
  s.max_epoch <- None

let update_epoch s (epoch : int option) =
  match epoch with
  | None -> ()
  | Some e ->
      s.max_epoch <-
        Some (match s.max_epoch with None -> e | Some prev -> max prev e)

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

let update_best (best : best_value option) ~step ~value ~compare =
  let new_best = { step; value } in
  match best with
  | None -> Some new_best
  | Some prev -> if compare value prev.value then Some new_best else Some prev

let update store (events : Kaun_runlog.Event.t list) =
  List.iter
    (fun (Event.Scalar s) ->
      update_epoch store s.epoch;
      let next = { step = s.step; epoch = s.epoch; value = s.value } in
      let hp : history_point = { step = s.step; value = s.value } in
      let data =
        match Hashtbl.find_opt store.by_tag s.tag with
        | None ->
            {
              latest = next;
              history = [ hp ];
              best_min = Some { step = s.step; value = s.value };
              best_max = Some { step = s.step; value = s.value };
            }
        | Some d ->
            let latest =
              if should_replace ~prev:d.latest ~next then next else d.latest
            in
            let best_min =
              update_best d.best_min ~step:s.step ~value:s.value ~compare:( < )
            in
            let best_max =
              update_best d.best_max ~step:s.step ~value:s.value ~compare:( > )
            in
            { latest; history = d.history @ [ hp ]; best_min; best_max }
      in
      Hashtbl.replace store.by_tag s.tag data)
    events

let latest_epoch store = store.max_epoch

let latest_metrics store =
  Hashtbl.fold (fun tag d acc -> (tag, d.latest) :: acc) store.by_tag []
  |> List.sort (fun (a, _) (b, _) -> String.compare a b)

let history_for_tag store tag =
  (match Hashtbl.find_opt store.by_tag tag with
    | None -> []
    | Some d -> d.history)
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
  match Hashtbl.find_opt store.by_tag tag with
  | None -> None
  | Some d -> if prefers_lower tag then d.best_min else d.best_max

let best_metrics store =
  Hashtbl.fold
    (fun tag d acc ->
      match if prefers_lower tag then d.best_min else d.best_max with
      | None -> acc
      | Some best -> (tag, best) :: acc)
    store.by_tag []
  |> List.sort (fun (a, _) (b, _) -> String.compare a b)
