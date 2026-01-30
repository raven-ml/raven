(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun_runlog

type metric = {
  step : int;
  epoch : int option;
  value : float;
}

type t = {
  table : (string, metric) Hashtbl.t;
  mutable max_epoch : int option;
}

let create ?(initial_size = 32) () =
  { table = Hashtbl.create initial_size; max_epoch = None }

let clear s =
  Hashtbl.clear s.table;
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

let should_replace ~prev ~next =
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
      match Hashtbl.find_opt store.table s.tag with
      | None -> Hashtbl.replace store.table s.tag next
      | Some prev ->
          if should_replace ~prev ~next then
            Hashtbl.replace store.table s.tag next)
    events

let latest_epoch store = store.max_epoch

let latest_metrics store =
  Hashtbl.fold (fun tag metric acc -> (tag, metric) :: acc) store.table []
  |> List.sort (fun (a, _) (b, _) -> String.compare a b)
