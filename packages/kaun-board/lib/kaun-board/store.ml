(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type metric = { step : int; epoch : int option; value : float }
type history_point = { step : int; value : float }
type best_value = { step : int; value : float }

type tag_data = {
  latest : metric;
  history : history_point list;
  best : best_value option;
  minimize : bool;
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

let update_epoch s = function
  | None -> ()
  | Some e ->
      s.max_epoch <-
        Some (match s.max_epoch with None -> e | Some prev -> max prev e)

let should_replace ~(prev : metric) ~(next : metric) =
  if next.step > prev.step then true
  else if next.step < prev.step then false
  else
    match (prev.epoch, next.epoch) with
    | None, None -> false
    | None, Some _ -> true
    | Some _, None -> false
    | Some a, Some b -> b > a

let update_best best ~step ~value ~compare =
  let candidate = { step; value } in
  match best with
  | None -> Some candidate
  | Some prev -> if compare value prev.value then Some candidate else Some prev

let update store events =
  List.iter
    (fun (Event.Scalar s) ->
      update_epoch store s.epoch;
      let next = { step = s.step; epoch = s.epoch; value = s.value } in
      let hp : history_point = { step = s.step; value = s.value } in
      let compare = if s.minimize then ( < ) else ( > ) in
      let data =
        match Hashtbl.find_opt store.by_tag s.tag with
        | None ->
            {
              latest = next;
              history = [ hp ];
              best = Some { step = s.step; value = s.value };
              minimize = s.minimize;
            }
        | Some d ->
            let latest =
              if should_replace ~prev:d.latest ~next then next else d.latest
            in
            let best =
              update_best d.best ~step:s.step ~value:s.value ~compare
            in
            { latest; history = d.history @ [ hp ]; best; minimize = d.minimize }
      in
      Hashtbl.replace store.by_tag s.tag data)
    events

let latest_epoch store = store.max_epoch

let latest_metrics store =
  Hashtbl.fold (fun tag d acc -> (tag, d.latest) :: acc) store.by_tag []
  |> List.sort (fun (a, _) (b, _) -> String.compare a b)

let history_for_tag store tag =
  match Hashtbl.find_opt store.by_tag tag with
  | None -> []
  | Some d -> List.map (fun (p : history_point) -> (p.step, p.value)) d.history

let best_for_tag store tag =
  match Hashtbl.find_opt store.by_tag tag with
  | None -> None
  | Some d -> d.best

let best_metrics store =
  Hashtbl.fold
    (fun tag d acc ->
      match d.best with None -> acc | Some best -> (tag, best) :: acc)
    store.by_tag []
  |> List.sort (fun (a, _) (b, _) -> String.compare a b)
