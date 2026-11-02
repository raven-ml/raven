(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = Fold.t

let read ?limit ?series run = Fold.read ?limit ?keep:series run
let follow = Fold.follow
let restarted = Fold.restarted
let metrics = Fold.metrics
let readings = Fold.readings
let goal = Fold.goal
let step_metric = Fold.step_metric
let series = Fold.series
let orphans = Fold.orphans
let stats = Fold.stats

let best h k =
  match (Fold.goal h k, Fold.stats h k) with
  | Some g, Some st -> Metric.best g st
  | _ -> None

let summary = Fold.summary

type file = Fold.file = {
  key : string;
  step : int;
  time : float;
  path : string;
}

let files = Fold.files
let uses = Fold.uses

type resume = Fold.resume = {
  step : int;
  time : float;
  provenance : Provenance.t;
}

let resumes = Fold.resumes
let damaged = Fold.damaged

type key_stats = Hint.key = {
  kind : [ `Metric | `Reading ];
  goal : Metric.goal option;
  stats : Metric.stats;
}

let hint_path dir = Filename.concat dir "stats.json"

let stats_of run =
  let (l : Listing.t) = run in
  let from_hint =
    match File.read_opt (hint_path l.dir) with
    | exception Sys_error _ -> None
    | None -> None
    | Some s -> (
        match Hint.decode s with
        | Some (length, keys) -> (
            match File.stamp (Listing.log l.dir) with
            | Some (_, size, _) when size = length -> Some keys
            | _ -> None)
        | None -> None)
  in
  match from_hint with
  | Some keys -> keys
  | None -> Fold.hint_keys (Fold.read ~keep:[] l)

let run h = Fold.run h
