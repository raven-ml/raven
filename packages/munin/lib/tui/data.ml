(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Munin
module Smap = Map.Make (String)

type keys = All | Keys of string list

type entry = {
  history : History.t;
  keys : keys;
  watch : Watch.t;
  rate : float option;
  pauses : (float * float) list;
}

type chart = {
  focus : string;
  marked : (char * string) list;
  view : Chart.view;
  spec : Chart.spec;
  kept : bool;
      (* kept, once its run has ended, after the screen stops asking for it *)
}

type trend = {
  run : string;
  key : string;
  axis : Chart.axis;
  width : int;
  domain : (float * float) option;
}

type zoom = {
  focus : string;
  others : string list;
  key : string;
  axis : Chart.axis;
  index : int option;
}

type cursor = {
  at : zoom;
  x : float option;
  values : (string * float option) list;
}

module Cmap = Map.Make (struct
  type t = chart

  let compare = compare
end)

module Tmap = Map.Make (struct
  type t = trend

  let compare = compare
end)

(* A prepared chart, and whether its focus's history had read the run's end. *)
type made = { chart : Chart.prepared; whole : bool }

type t = {
  listing : Run.t list;
  entries : entry Smap.t;
  hints : (string * History.key_stats) list Smap.t; (* ended runs' *)
  charts : made Cmap.t;
  trends : Chart.trend Tmap.t;
  cursor : cursor option;
}

let empty =
  {
    listing = [];
    entries = Smap.empty;
    hints = Smap.empty;
    charts = Cmap.empty;
    trends = Tmap.empty;
    cursor = None;
  }

let listing d = d.listing
let entry d id = Smap.find_opt id d.entries

let run d id =
  match Smap.find_opt id d.entries with
  | Some e -> Some (History.run e.history)
  | None -> List.find_opt (fun r -> Run.id r = id) d.listing

let key_stats h =
  let of_key kind k =
    Option.map
      (fun stats -> (k, { History.kind; goal = History.goal h k; stats }))
      (History.stats h k)
  in
  List.sort
    (fun (a, _) (b, _) -> String.compare a b)
    (List.filter_map (of_key `Metric) (History.metrics h)
    @ List.filter_map (of_key `Reading) (History.readings h))

let stats d r =
  match Smap.find_opt (Run.id r) d.entries with
  | Some e -> Some (key_stats e.history)
  | None -> Smap.find_opt (Run.id r) d.hints

let word d ~now r =
  let w = Option.map (fun e -> e.watch) (Smap.find_opt (Run.id r) d.entries) in
  Watch.word (Run.status r) w ~now

let made d (c : chart) =
  match Cmap.find_opt c d.charts with
  | Some m -> Some m
  | None ->
      (* One prepared for the same focus and region under another view or marks,
         as while a key changes them: never one of another region, whose plot is
         not this one's. *)
      Cmap.fold
        (fun (c' : chart) m acc ->
          if c'.focus = c.focus && c'.spec = c.spec then Some m else acc)
        d.charts None

let chart d c = Option.map (fun m -> m.chart) (made d c)

let whole d c =
  match Cmap.find_opt c d.charts with Some m -> m.whole | None -> false

let trend d t = Option.map Chart.text (Tmap.find_opt t d.trends)
let cursor d = d.cursor

(* Reads *)

type request = {
  time : float;
  list : bool;
  prev : Run.t list;
  follow : (string * keys) list;
  keep : string list;
  held : entry Smap.t;
  hinted : string list;
  want_charts : chart list;
  old_charts : made Cmap.t;
  want_trends : trend list;
  old_trends : Chart.trend Tmap.t;
  want_zoom : zoom option;
}

let request d ~time ~list ~follow ~keep ~charts ~trends ~zoom =
  {
    time;
    list;
    prev = d.listing;
    follow;
    keep;
    held = d.entries;
    hinted = List.map fst (Smap.bindings d.hints);
    want_charts = charts;
    old_charts = d.charts;
    want_trends = trends;
    old_trends = d.trends;
    want_zoom = zoom;
  }

type result = {
  at : float;
  wanted : string list; (* the runs followed or kept *)
  want_charts : chart list;
  want_trends : trend list;
  listing : Run.t list option;
  read : (string * History.t * keys * float option * (float * float) list) list;
  hints : (string * (string * History.key_stats) list) list;
  charts : made Cmap.t;
  trends : Chart.trend Tmap.t;
  cursor : cursor option;
  errors : (string * string) list;
}

let errors (r : result) = r.errors

(* [covers have want] is [true] iff a history that keeps [have] keeps every key
   of [want]. *)
let covers have want =
  match (have, want) with
  | All, _ -> true
  | Keys _, All -> false
  | Keys h, Keys w -> List.for_all (fun k -> List.mem k h) w

let union a b =
  match (a, b) with
  | All, _ | _, All -> All
  | Keys a, Keys b -> Keys (List.sort_uniq String.compare (a @ b))

(* The rate over the last 10 minutes, or the last tenth of the active time, from
   the series with the most samples, once that window holds 60 s. *)
let rate h ~active =
  let best =
    List.fold_left
      (fun acc k ->
        match History.series h k with
        | s when Metric.length s > match acc with Some (_, n) -> n | None -> 1
          ->
            Some (s, Metric.length s)
        | _ -> acc
        | exception Invalid_argument _ -> acc)
      None (History.metrics h)
  in
  match best with
  | None -> None
  | Some (s, n) ->
      let span = Float.max 600. (0.1 *. active) in
      let t1 = Metric.time s (n - 1) in
      let k = ref (n - 1) in
      while !k > 0 && t1 -. Metric.time s (!k - 1) <= span do
        decr k
      done;
      let dt = t1 -. Metric.time s !k in
      let ds = Metric.step s (n - 1) - Metric.step s !k in
      if dt < 60. || ds <= 0 then None else Some (Float.of_int ds /. dt)

let read store (q : request) =
  let errors = ref [] in
  let guard id f =
    try Some (f ())
    with Sys_error e | Failure e | Invalid_argument e ->
      errors := (id, e) :: !errors;
      None
  in
  let listing =
    if q.list then
      guard "" (fun () ->
          if q.prev = [] then Store.runs store
          else Store.runs ~prev:q.prev store)
    else None
  in
  let runs = Option.value listing ~default:q.prev in
  let find id =
    match List.find_opt (fun r -> Run.id r = id) runs with
    | Some r -> Some r
    | None -> Option.join (guard id (fun () -> Store.run store id))
  in
  (* A held history that keeps the keys asked for is followed; otherwise the log
     is read from its start with the keys both ask for. *)
  let follow (id, want) =
    let fresh have =
      Option.bind (find id) (fun r ->
          let series = match have with All -> None | Keys k -> Some k in
          guard id (fun () -> (History.read ?series r, have)))
    in
    let got =
      match Smap.find_opt id q.held with
      | Some e when covers e.keys want ->
          Option.map
            (fun h -> (h, e.keys))
            (guard id (fun () -> History.follow e.history))
      | Some e -> fresh (union e.keys want)
      | None -> fresh want
    in
    Option.map
      (fun (h, keys) ->
        let r = History.run h in
        let pauses, rate =
          match keys with
          | All ->
              (* The pauses change only with the resumes, which a follow that
                 reads one marks as a restart or a longer list. *)
              let pauses =
                match Smap.find_opt id q.held with
                | Some { keys = All; pauses; history; _ }
                  when (not (History.restarted h))
                       && List.length (History.resumes history)
                          = List.length (History.resumes h) ->
                    pauses
                | _ -> Fact.pauses h
              in
              let until =
                match (Run.ended_at r, Run.status r) with
                | Some e, _ -> e
                | None, (`Running | `Silent) -> q.time
                | None, _ -> Run.updated_at r
              in
              ( pauses,
                rate h ~active:(Fact.active_at pauses (Run.started_at r) until)
              )
          | Keys _ -> ([], None)
        in
        (id, h, keys, rate, pauses))
      got
  in
  let read = List.filter_map follow q.follow in
  let history id =
    match List.find_opt (fun (id', _, _, _, _) -> id' = id) read with
    | Some (_, h, _, _, _) -> Some h
    | None -> Option.map (fun e -> e.history) (Smap.find_opt id q.held)
  in
  let hints =
    if q.list then
      List.filter_map
        (fun r ->
          let id = Run.id r in
          if
            Run.ended_at r = None
            || List.mem id q.hinted || Smap.mem id q.held
            || List.mem_assoc id q.follow
          then None
          else
            Option.map
              (fun s -> (id, s))
              (guard id (fun () -> History.stats_of r)))
        runs
    else []
  in
  let charts =
    List.fold_left
      (fun m (c : chart) ->
        match
          ( history c.focus,
            List.filter_map
              (fun (l, id) -> Option.map (fun h -> (l, h)) (history id))
              c.marked )
        with
        | Some focus, marked -> (
            let prev =
              Option.map (fun m -> m.chart) (Cmap.find_opt c q.old_charts)
            in
            match
              guard c.focus (fun () ->
                  Chart.prepare ?prev c.view c.spec focus marked)
            with
            | Some chart ->
                let whole = Run.ended_at (History.run focus) <> None in
                Cmap.add c { chart; whole } m
            | None -> m)
        | None, _ -> m)
      Cmap.empty q.want_charts
  in
  let trends =
    List.fold_left
      (fun m (t : trend) ->
        match history t.run with
        | None -> m
        | Some h -> (
            let prev = Tmap.find_opt t q.old_trends in
            match
              Option.join
                (guard t.run (fun () ->
                     Chart.trend ?prev ?domain:t.domain t.axis ~width:t.width h
                       t.key))
            with
            | Some tr -> Tmap.add t tr m
            | None -> m))
      Tmap.empty q.want_trends
  in
  (* The values of each run at the step cursor's x. *)
  let cursor =
    Option.bind q.want_zoom (fun (z : zoom) ->
        Option.map
          (fun focus ->
            let series h =
              match History.series h z.key with
              | s -> Some s
              | exception Invalid_argument _ -> None
            in
            let x =
              Option.bind (series focus) (fun s ->
                  let n = Metric.length s in
                  if n = 0 then None
                  else
                    let i =
                      match z.index with
                      | Some i -> max 0 (min (n - 1) i)
                      | None -> n - 1
                    in
                    let x = Chart.x_of z.axis focus z.key s i in
                    if Float.is_nan x then None else Some x)
            in
            let value_at h =
              match (x, series h) with
              | Some at, Some s ->
                  let xs = Chart.x_of z.axis h z.key s in
                  let r = ref None in
                  for i = 0 to Metric.length s - 1 do
                    if xs i <= at then r := Some (Metric.value s i)
                  done;
                  !r
              | _ -> None
            in
            let ids = z.focus :: z.others in
            {
              at = z;
              x;
              values =
                List.filter_map
                  (fun id ->
                    Option.map (fun h -> (id, value_at h)) (history id))
                  ids;
            })
          (history z.focus))
  in
  {
    at = q.time;
    wanted = List.map fst q.follow @ q.keep;
    want_charts = q.want_charts;
    want_trends = q.want_trends;
    listing;
    read;
    hints;
    charts;
    trends;
    cursor;
    errors = !errors;
  }

let samples h =
  List.fold_left
    (fun n k ->
      match History.stats h k with Some s -> n + s.count | None -> n)
    0 (History.metrics h)

let apply (d : t) (r : result) =
  let listing = Option.value r.listing ~default:d.listing in
  let in_store id =
    match r.listing with
    | None -> true
    | Some l -> List.exists (fun x -> Run.id x = id) l
  in
  (* What was read replaces what was held; what was held and is still wanted
     stays, read or not. *)
  let wanted = r.wanted in
  let entries =
    Smap.filter (fun id _ -> in_store id && List.mem id wanted) d.entries
  in
  let entries =
    List.fold_left
      (fun m (id, history, keys, rate, pauses) ->
        let samples = samples history in
        let watch =
          match Smap.find_opt id d.entries with
          | Some e -> Watch.observe e.watch ~now:r.at ~samples
          | None -> Watch.start ~now:r.at ~samples
        in
        Smap.add id { history; keys; watch; rate; pauses } m)
      entries r.read
  in
  let hints =
    List.fold_left (fun m (id, s) -> Smap.add id s m) d.hints r.hints
  in
  let hints =
    match r.listing with
    | None -> hints
    | Some _ -> Smap.filter (fun id _ -> in_store id) hints
  in
  (* A chart asked for and not prepared, as when its run could not be read,
     keeps its last preparation. *)
  let charts =
    List.fold_left
      (fun m c ->
        match (Cmap.find_opt c r.charts, made d c) with
        | Some p, _ | None, Some p -> Cmap.add c p m
        | None, None -> m)
      Cmap.empty r.want_charts
  in
  (* An ended run's kept chart stays if it was prepared from the run's whole
     log, unless a newer one of the run took its place. A chart prepared before
     the end lacks the samples written after it. *)
  let ended id =
    List.exists (fun x -> Run.id x = id && Run.ended_at x <> None) listing
  in
  let charts =
    Cmap.fold
      (fun (c : chart) p m ->
        if
          c.kept && p.whole && ended c.focus
          && not
               (Cmap.exists
                  (fun (c' : chart) _ -> c'.kept && c'.focus = c.focus)
                  m)
        then Cmap.add c p m
        else m)
      d.charts charts
  in
  let trends =
    List.fold_left
      (fun m t ->
        match (Tmap.find_opt t r.trends, Tmap.find_opt t d.trends) with
        | Some p, _ | None, Some p -> Tmap.add t p m
        | None, None -> m)
      Tmap.empty r.want_trends
  in
  {
    listing;
    entries;
    hints;
    charts;
    trends;
    cursor = (match r.cursor with Some _ as c -> c | None -> d.cursor);
  }
