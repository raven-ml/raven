(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Munin
module Smap = Map.Make (String)
module Sset = Runs.Sset

type slot = Large | Small of int | Strip | Trend | Rows | Strip_rows | Centre
type at = Latest | Step of int

type depth =
  | Charts
  | Strips of { strip : string option }
  | Zoom of { metric : string; at : at; narrow : int }

type layer = Screen | Filtering of Mosaic.Editor.t | Info of Mosaic.Scroll.t | Help
type pane = Auto | Shown | Hidden
type cursor = Unchosen | Waiting | On of string
type poll = Idle | Reading | Again

module Overlays = struct
  type t = { a : string option; b : string option; c : string option; d : string option }

  let empty = { a = None; b = None; c = None; d = None }
  let slots o = [ ('a', o.a); ('b', o.b); ('c', o.c); ('d', o.d) ]
  let runs o = List.filter_map (fun (l, s) -> Option.map (fun id -> (l, id)) s) (slots o)

  let toggle id o =
    let off s = if s = Some id then None else s in
    if List.exists (fun (_, s) -> s = Some id) (slots o) then
      Some { a = off o.a; b = off o.b; c = off o.c; d = off o.d }
    else
      match o with
      | { a = None; _ } -> Some { o with a = Some id }
      | { b = None; _ } -> Some { o with b = Some id }
      | { c = None; _ } -> Some { o with c = Some id }
      | { d = None; _ } -> Some { o with d = Some id }
      | _ -> None
end

type t = {
  store : Store.t;
  clock : unit -> float;
  size : (int * int) option;
  slots : (slot * (int * int)) list;
  look : Theme.look;
  data : Data.t;
  poll : poll;
  cursor : cursor;
  metric : string option;
  depth : depth;
  layer : layer;
  overlays : Overlays.t;
  colours : int Smap.t;
  others : bool;
  smooth : bool;
  scales : Sset.t;
  filter : Runs.filter;
  order : Runs.order;
  folds : Runs.folds;
  runs_pane : pane;
  run_pane : pane;
  top : int;
  strips_top : int;
  notice : string option;
}

let make ~store ~clock ~scheme ~cursor =
  {
    store;
    clock;
    size = None;
    slots = [];
    look = { scheme; depth = `Plain; background = None };
    data = Data.empty;
    poll = Idle;
    cursor;
    metric = None;
    depth = Charts;
    layer = Screen;
    overlays = Overlays.empty;
    colours = Smap.empty;
    others = false;
    smooth = true;
    scales = Sset.empty;
    filter = Runs.no_filter;
    order = Runs.Started;
    folds = Runs.Unset;
    runs_pane = Auto;
    run_pane = Auto;
    top = 0;
    strips_top = 0;
    notice = None;
  }

(* The screen's shape *)

type side = Beside of int | Over of int | Off
type hero = Big of int | Line

type panes = { runs : int option; run : side; nav : bool; hero : hero; smalls : int }

type frame = Unsized | Tiny | Reading | Empty | Waiting_run | Panes of panes

(* The terminal sizes at which the screen changes. *)
let wide = 150
let medium = 100
let least_width = 60
let least_height = 16
let hero_height = 30
let three_smalls_height = 44

let frame m =
  match m.size with
  | None -> Unsized
  | Some (w, h) when w < least_width || h < least_height -> Tiny
  | Some _ when not (Data.listed m.data) -> Reading
  | Some _ when Data.runs m.data = [] -> Empty
  | Some _ when m.cursor = Waiting -> Waiting_run
  | Some (w, h) ->
      let runs_w = if w >= wide then 42 else 34 in
      let run_w = if w >= wide then 34 else 28 in
      (* The filter's line is in the runs pane, which shows while it is
         typed. *)
      let runs =
        match (m.layer, m.runs_pane) with
        | Filtering _, _ | _, Shown -> Some runs_w
        | _, Hidden -> None
        | _, Auto -> if w >= medium then Some runs_w else None
      in
      let run =
        match m.run_pane with
        | Hidden -> Off
        | Shown -> if w >= medium then Beside run_w else Over run_w
        | Auto ->
            if w >= wide then Beside run_w
            else if w >= medium && m.depth = Charts then Beside run_w
            else Off
      in
      let hero =
        if w >= medium && h >= hero_height then Big (if w >= wide then 4 else 3)
        else Line
      in
      let smalls =
        if w >= wide && h >= three_smalls_height then 3
        else if w >= medium && h >= hero_height then 2
        else 0
      in
      Panes { runs; run; nav = w < medium; hero; smalls }

(* The runs *)

let id (f : Facts.run) = Run.id f.run
let experiment (f : Facts.run) = Run.experiment f.run

let key m e =
  Option.bind (Data.experiment m.data e) (fun (x : Facts.experiment) -> x.key)

let filter m =
  match m.layer with Filtering e -> Runs.filter (Mosaic.Editor.text e) | _ -> m.filter

let kept m = Runs.keep (filter m) m.order ~keys:(key m) (Data.runs m.data)
let folded m = match m.folds with Runs.Unset -> Sset.empty | Folded s -> s
let lines m = Runs.lines (kept m) (folded m)

let focus m =
  let k = kept m in
  let chosen =
    match m.cursor with
    | On c -> List.find_opt (fun f -> id f = c) k
    | Unchosen | Waiting -> None
  in
  match (chosen, m.cursor) with
  | Some f, _ -> Some f
  | None, Waiting -> None
  | None, _ -> Option.bind (Runs.first (Runs.lines k (folded m))) (Data.run m.data)

let metrics m = match focus m with Some f -> f.metrics | None -> []

let metric m =
  match focus m with
  | None -> None
  | Some f -> (
      match m.metric with
      | Some k when List.mem k f.metrics -> Some k
      | _ -> (
          match key m (experiment f) with
          | Some k when List.mem k f.metrics -> Some k
          | _ -> List.nth_opt f.metrics 0))

let goal_metrics (f : Facts.run) =
  List.filter (fun k -> Facts.goal f k <> None) f.metrics

(* The machine's readings a run's strips end with, those the run pane
   gives. *)
let machine (f : Facts.run) =
  let has k = Facts.stats f k <> None in
  let device =
    List.find_map
      (fun (k, (s : History.key_stats)) ->
        if s.kind = `Reading && String.starts_with ~prefix:"dev/" k then
          Option.map (fun j -> String.sub k 4 (j - 4)) (String.index_from_opt k 4 '/')
        else None)
      f.stats
  in
  let dev =
    match device with
    | Some d -> [ "dev/" ^ d ^ "/busy_pct"; "dev/" ^ d ^ "/used_bytes" ]
    | None -> []
  in
  List.filter has (dev @ [ "sys/cpu_busy_pct"; "sys/mem_used_bytes" ])

type kind = Metric | Machine

let strips m =
  match focus m with
  | None -> []
  | Some f ->
      List.map (fun k -> (k, Metric)) f.metrics @ List.map (fun k -> (k, Machine)) (machine f)

let strip m =
  match m.depth with
  | Strips { strip = Some k } when List.mem_assoc k (strips m) -> Some k
  | _ -> metric m

let colour m id = Smap.find_opt id m.colours

let drawn m =
  match focus m with
  | None -> []
  | Some f ->
      let me = id f in
      let listed o = o <> me && Data.run m.data o <> None in
      let overlays = List.filter (fun (_, o) -> listed o) (Overlays.runs m.overlays) in
      let context =
        if not m.others then []
        else
          match Data.experiment m.data (experiment f) with
          | None -> []
          | Some e ->
              List.filter (fun o -> o <> me && not (List.exists (fun (_, x) -> x = o) overlays)) e.runs
      in
      List.map (fun o -> (Plot.Context, o)) context
      @ List.map (fun (l, o) -> (Plot.Overlay l, o)) overlays
      @ [ (Plot.Cursor, me) ]

(* Names *)

(* The fewest characters of an id's end that tell a run apart. *)
let suffix_least = 4

let suffixes m =
  let named =
    List.filter_map
      (fun (r, o) -> if r = Plot.Context then None else Data.run m.data o)
      (drawn m)
  in
  let runs = List.sort_uniq (fun a b -> String.compare (id a) (id b)) (kept m @ named) in
  let by_name = Hashtbl.create 16 in
  List.iter
    (fun f ->
      let n = Facts.name f.Facts.run in
      Hashtbl.replace by_name n (id f :: Option.value (Hashtbl.find_opt by_name n) ~default:[]))
    runs;
  let tail k s = if k >= String.length s then s else String.sub s (String.length s - k) k in
  Hashtbl.fold
    (fun _ ids acc ->
      match ids with
      | [] | [ _ ] -> acc
      | _ ->
          (* The least [k] whose ends differ, else the whole ids. *)
          let longest = List.fold_left (fun w i -> max w (String.length i)) 0 ids in
          let rec find k =
            let ends = List.map (tail k) ids in
            if k >= longest || List.length (List.sort_uniq String.compare ends) = List.length ids
            then k
            else find (k + 1)
          in
          let k = find suffix_least in
          List.fold_left (fun acc i -> Smap.add i (tail k i) acc) acc ids)
    by_name Smap.empty

let label m f =
  let n = Facts.name f.Facts.run in
  match Smap.find_opt (id f) (suffixes m) with Some s -> n ^ " " ^ s | None -> n

let label_in m f w =
  let n = Facts.name f.Facts.run in
  match Smap.find_opt (id f) (suffixes m) with
  | None -> Numbers.cut w n
  | Some s ->
      let room = w - 1 - Numbers.width s in
      if room >= 1 then Numbers.cut room n ^ " " ^ s else Numbers.cut w s

(* The charts *)

type chart = {
  slot : slot;
  key : string;
  kind : kind;
  x : float * float;
  runs : (Plot.role * string) list;
  smooth : bool;
  log : bool;
  size : (int * int) option;
}

let facts m runs = List.filter_map (fun (_, o) -> Data.run m.data o) runs

(* The x domain of a chart of [key] over [runs]. *)
let domain m key runs = Plot.domain ~key (facts m runs)

let chart m slot key kind x runs =
  let log = Plot.log (facts m runs) key <> Sset.mem key m.scales in
  let smooth = m.smooth && Plot.smoothable key in
  { slot; key; kind; x; runs; smooth; log; size = List.assoc_opt slot m.slots }

let marked runs = List.filter (fun (r, _) -> r <> Plot.Context) runs

let zoom_step m =
  match (m.depth, focus m) with
  | Zoom { at = Step s; _ }, _ -> Some s
  | Zoom { at = Latest; _ }, Some f -> f.latest
  | _ -> None

let zoom_window m =
  match m.depth with
  | Zoom { narrow; metric; _ } ->
      let at = Float.of_int (Option.value (zoom_step m) ~default:0) in
      Some (Plot.window (domain m metric (drawn m)) ~at ~narrow)
  | Charts | Strips _ -> None

let charts m =
  match (frame m, focus m) with
  | Panes p, Some f -> (
      let drawn = drawn m in
      match m.depth with
      | Charts -> (
          match metric m with
          | None -> []
          | Some k ->
              let others = List.filter (( <> ) k) (goal_metrics f) in
              let smalls = List.filteri (fun i _ -> i < p.smalls) others in
              let small i s = chart m (Small i) s Metric (domain m s (marked drawn)) (marked drawn) in
              chart m Large k Metric (domain m k drawn) drawn :: List.mapi small smalls)
      | Strips _ ->
          let first = match marked drawn with ((Plot.Overlay _, _) as o) :: _ -> [ o ] | _ -> [] in
          let runs = first @ [ (Plot.Cursor, id f) ] in
          (* One axis for every strip, so that a column is one moment in each. *)
          let shared = Plot.domain (facts m runs) in
          List.map (fun (k, kind) -> chart m Strip k kind shared runs) (strips m)
      | Zoom { metric; _ } -> (
          let kind = Option.value (List.assoc_opt metric (strips m)) ~default:Metric in
          match zoom_window m with
          | Some w -> [ chart m Large metric kind w drawn ]
          | None -> []))
  | _ -> []

(* A run's trend is in its colour while the charts draw it, else faint. *)
let trend m (f : Facts.run) =
  let me = id f in
  let role =
    if List.exists (fun (_, o) -> o = me) (marked (drawn m)) then Plot.Cursor else Plot.Context
  in
  Option.map (fun k -> chart m Trend k Metric (0., f.x_end) [ (role, me) ]) (key m (experiment f))

(* The runs pane's lines shown. *)
let shown_lines m =
  match (frame m, List.assoc_opt Rows m.slots) with
  | Panes { runs = Some _; _ }, Some (_, h) ->
      List.filteri (fun i _ -> i >= m.top && i < m.top + h) (lines m)
  | _ -> []

let trends m =
  List.filter_map
    (function Runs.Run f -> Option.map (fun c -> (id f, c)) (trend m f) | Header _ -> None)
    (shown_lines m)

(* What a chart draws but its lines. *)
let frame_of m (c : chart) : Plot.t =
  let zoom = match m.depth with Zoom _ -> c.slot = Large | Charts | Strips _ -> false in
  let now =
    match (focus m, c.slot) with
    | Some f, (Large | Small _) when Watch.live f.state ->
        Option.map Float.of_int f.latest
    | _ -> None
  in
  let guides =
    match c.slot with Large -> `Both | Small _ -> `Y | Strip | Trend | Rows | Strip_rows | Centre -> `None
  in
  {
    Plot.x = c.x;
    log = c.log;
    smooth = c.smooth;
    guides;
    now;
    cursor = (if zoom then Option.map Float.of_int (zoom_step m) else None);
    extents = zoom;
    fill = c.kind = Machine && not zoom;
    lines = [];
  }

(* A chart's dot columns at its slot's size. *)
let dots m c =
  match c.size with
  | None -> None
  | Some (width, height) -> Plot.dots (Theme.make m.look) (frame_of m c) ~width ~height

(* A reduction keeps smoothed means where smoothing may draw them, so that
   [s] waits for no poll. *)
let data_plot (c : chart) dots run =
  { Data.run; key = c.key; x = c.x; dots; smooth = Plot.smoothable c.key }

(* A chart to draw: the frame and each run's reductions, whatever their dot
   counts, since the draw lays out the plot that picks one. *)
let plot m (c : chart) =
  let line (role, run) =
    let reductions = Data.reductions m.data ~run ~key:c.key ~x:c.x in
    { Plot.role; colour = Option.value (colour m run) ~default:0; reductions }
  in
  { (frame_of m c) with lines = List.map line c.runs }

let data_plots m c =
  match dots m c with
  | None -> []
  | Some (n, dot) -> List.map (fun (_, run) -> (data_plot c n run, dot)) c.runs

(* The values the screen prints *)

let legend m =
  let drawn = drawn m in
  let runs = List.map snd (marked drawn) in
  let at key at x = { Data.key; runs; at; tau = Plot.tau x } in
  match (m.depth, focus m) with
  | Charts, Some f -> Option.map (fun k -> at k (Data.Latest (id f)) (domain m k drawn)) (metric m)
  | Zoom { metric; _ }, Some _ -> (
      match (zoom_step m, zoom_window m) with
      | Some s, Some w -> Some (at metric (Data.At s) w)
      | _ -> None)
  | _ -> None

let hero m =
  match (m.depth, focus m) with
  | Zoom _, _ | _, None -> []
  | _, Some f ->
      (* Smoothed as the chart that draws it, or as a small chart would. *)
      let shown = charts m in
      let x key =
        match List.find_opt (fun (c : chart) -> c.key = key) shown with
        | Some c -> c.x
        | None -> domain m key (marked (drawn m))
      in
      List.filter_map
        (fun key ->
          if m.smooth && Plot.smoothable key then
            Some { Data.key; runs = [ id f ]; at = Latest (id f); tau = Plot.tau (x key) }
          else None)
        (goal_metrics f)

let readouts m = Option.to_list (legend m) @ hero m

let compare m =
  match (m.depth, focus m) with
  | Strips _, Some f -> (
      match marked (drawn m) with
      | (Plot.Overlay _, o) :: _ -> Some { Data.focus = id f; other = o; keys = f.metrics }
      | _ -> None)
  | _ -> None

let wants m =
  {
    Data.focus = Option.map id (focus m);
    every = (match m.layer with Info _ -> true | _ -> false);
    plots = List.concat_map (data_plots m) (charts m @ List.map snd (trends m));
    readouts = readouts m;
    compare = compare m;
  }

(* Settling: what follows every change *)

(* A run drawn without a colour takes the colour least used by the runs
   drawn, then by every run so far, then the first. *)
let assign m =
  let drawn = List.map snd (marked (drawn m)) in
  let count ids c = List.length (List.filter (fun o -> Smap.find_opt o m.colours = Some c) ids) in
  List.fold_left
    (fun m o ->
      if Smap.mem o m.colours then m
      else
        let all = Smap.bindings m.colours |> List.map fst in
        let rank c = (count drawn c, count all c, c) in
        let best =
          List.fold_left
            (fun b c -> if Stdlib.compare (rank c) (rank b) < 0 then c else b)
            0
            (List.init Theme.runs Fun.id)
        in
        { m with colours = Smap.add o best m.colours })
    m drawn

(* The runs pane's first line, moved the least to show the cursor's. *)
let reveal m =
  match (List.assoc_opt Rows m.slots, focus m) with
  | Some (w, h), Some f -> (
      let ls = lines m in
      match Runs.at ls (id f) with
      | None -> m
      | Some line ->
          let scroll = Mosaic.Scroll.reveal ~line (Mosaic.Scroll.at m.top) in
          let _, top =
            Mosaic.Scroll.resolve scroll ~width:w ~height:h ~content_width:w
              ~content_height:(List.length ls)
          in
          { m with top })
  | _ -> m

(* The strips' lines: each strip's first, then their count. *)
let strip_height = 2
let strip_gap = 1

let strip_lines m =
  let step = strip_height + strip_gap in
  let metrics, machine = List.partition (fun (_, k) -> k = Metric) (strips m) in
  let first = List.mapi (fun i (k, _) -> (k, step * i)) metrics in
  let after = step * List.length metrics in
  let rest = List.mapi (fun i (k, _) -> (k, after + 1 + (step * i))) machine in
  (first @ rest, if machine = [] then after else after + 1 + (step * List.length machine))

(* The strips' first line, moved the least to show the cursor's strip whole
   from the lines shown. *)
let reveal_strip m =
  match (m.depth, List.assoc_opt Strip_rows m.slots, strip m) with
  | Strips _, Some (w, h), Some k -> (
      let at, n = strip_lines m in
      match List.assoc_opt k at with
      | None -> m
      | Some line ->
          let resolve s =
            snd (Mosaic.Scroll.resolve s ~width:w ~height:h ~content_width:w ~content_height:n)
          in
          let show line top = resolve (Mosaic.Scroll.reveal ~line (Mosaic.Scroll.at top)) in
          { m with strips_top = show line (show (line + strip_height - 1) m.strips_top) })
  | _ -> m

let settle m = reveal_strip (reveal (assign m))

(* Transitions *)

let polled d m =
  let m = { m with data = d } in
  let runs = Data.runs d in
  let m =
    match m.cursor with
    | Unchosen when runs <> [] -> { m with cursor = On (id (List.hd runs)) }
    | Waiting -> (
        let waits (f : Facts.run) = Run.ended_at f.run = None && f.state <> Watch.Crashed in
        match List.find_opt waits runs with
        | Some f -> { m with cursor = On (id f) }
        | None -> m)
    | _ -> m
  in
  let m =
    match m.folds with
    | Runs.Unset when Data.listed d ->
        let cursor = match m.cursor with On c -> Some c | _ -> None in
        { m with folds = Folded (Runs.first_folds runs ~cursor) }
    | _ -> m
  in
  settle m

let resized w h m = settle { m with size = Some (w, h) }

let sized s w h m =
  let slots = (s, (w, h)) :: List.remove_assoc s m.slots in
  settle { m with slots }

let scrolled_rows s m =
  match List.assoc_opt Rows m.slots with
  | Some (w, h) ->
      let n = List.length (lines m) in
      let _, top = Mosaic.Scroll.resolve s ~width:w ~height:h ~content_width:w ~content_height:n in
      { m with top }
  | None -> m

let scrolled_strips s m =
  match List.assoc_opt Strip_rows m.slots with
  | Some (w, h) ->
      let _, n = strip_lines m in
      let _, top = Mosaic.Scroll.resolve s ~width:w ~height:h ~content_width:w ~content_height:n in
      { m with strips_top = top }
  | None -> m

let edited e m = settle { m with layer = Filtering e }
let scrolled_info s m = { m with layer = Info s }
let palette look m = { m with look }
let styled depth m = { m with look = { m.look with depth } }

type action =
  | Vertical of int
  | Horizontal of int
  | Run of int
  | First
  | Last
  | Deeper
  | Back
  | Overlay
  | Others
  | Clear
  | Smooth
  | Scale
  | Narrow
  | Widen
  | Filter
  | Keep_filter
  | Clear_filter
  | Order
  | Fold
  | Info
  | Toggle_runs
  | Toggle_run
  | Help
  | Close
  | Quit

type msg =
  | Key of action
  | Tick
  | Polled of Data.t
  | Resized of int * int
  | Sized of slot * int * int
  | Edited of Mosaic.Editor.t
  | Submitted
  | Info_scrolled of Mosaic.Scroll.t
  | Rows_scrolled of Mosaic.Scroll.t
  | Strips_scrolled of Mosaic.Scroll.t
  | Palette of Mosaic.Ansi.Palette.t
  | Styling of Mosaic.Ansi.Styling.t

let to_run m = function Some r -> { m with cursor = On r } | None -> m

let move_run d m =
  match focus m with
  | None -> m
  | Some f -> to_run m (Runs.step (lines m) (id f) d)

let clamp n i = max 0 (min (n - 1) i)

let next_in l cur d =
  match List.find_index (( = ) cur) l with
  | None -> List.nth_opt l 0
  | Some i -> List.nth_opt l (clamp (List.length l) (i + d))

(* Zoom's cursor moved by [d] of the samples of the focus's metric that its
   chart draws: those no resume abandoned. *)
let move_step d metric at m =
  match focus m with
  | None -> None
  | Some f -> (
      let series =
        Option.bind (Data.history m.data (id f)) (fun h ->
            match History.surviving h metric with
            | s -> Some s
            | exception Invalid_argument _ -> None)
      in
      match series with
      | None -> None
      | Some s when Metric.length s = 0 -> None
      | Some s ->
          let n = Metric.length s in
          let cur = match at with Step c -> c | Latest -> Option.value f.latest ~default:0 in
          let rec last_at i best =
            if i = n then best else last_at (i + 1) (if Metric.step s i <= cur then i else best)
          in
          let i = clamp n (last_at 0 0 + d) in
          Some (Metric.step s i))

let shown_run m =
  match frame m with Panes { run = Beside _ | Over _; _ } -> true | _ -> false

let shown_runs m = match frame m with Panes { runs = Some _; _ } -> true | _ -> false
let toggle shown = if shown then Hidden else Shown

let back m =
  match m.depth with
  | Zoom { metric; _ } -> { m with depth = Strips { strip = Some metric } }
  | Strips _ -> { m with depth = Charts }
  | Charts ->
      if Overlays.runs m.overlays <> [] then { m with overlays = Overlays.empty }
      else if Runs.text m.filter <> "" then { m with filter = Runs.no_filter }
      else if m.others then { m with others = false }
      else m

let deeper m =
  let open_folded =
    match focus m with
    | Some f when Sset.mem (experiment f) (folded m) ->
        Some { m with folds = Folded (Sset.remove (experiment f) (folded m)) }
    | _ -> None
  in
  match open_folded with
  | Some m -> m
  | None -> (
      match m.depth with
      | Charts when focus m <> None -> { m with depth = Strips { strip = None } }
      | Strips _ -> (
          match strip m with
          | Some k -> { m with depth = Zoom { metric = k; at = Latest; narrow = 0 } }
          | None -> m)
      | Charts | Zoom _ -> m)

let scale_key m =
  match m.depth with Charts -> metric m | Strips _ -> strip m | Zoom { metric; _ } -> Some metric

let full = "four overlays at most: c clears them"

let step m = function
  | Vertical d -> (
      match m.depth with
      | Charts -> move_run d m
      | Strips _ -> (
          let keys = List.map fst (strips m) in
          match Option.bind (strip m) (fun k -> next_in keys k d) with
          | Some k -> { m with depth = Strips { strip = Some k } }
          | None -> m)
      | Zoom _ -> m)
  | Horizontal d -> (
      match m.depth with
      | Charts -> (
          match Option.bind (metric m) (fun k -> next_in (metrics m) k d) with
          | Some k -> { m with metric = Some k }
          | None -> m)
      | Zoom z -> (
          match move_step d z.metric z.at m with
          | Some s -> { m with depth = Zoom { z with at = Step s } }
          | None -> m)
      | Strips _ -> m)
  | Run d -> move_run d m
  | First -> to_run m (Runs.first (lines m))
  | Last -> to_run m (Runs.last (lines m))
  | Deeper -> deeper m
  | Back -> back m
  | Overlay -> (
      match focus m with
      | None -> m
      | Some f -> (
          match Overlays.toggle (id f) m.overlays with
          | Some overlays -> { m with overlays }
          | None -> { m with notice = Some full }))
  | Others -> { m with others = not m.others }
  | Clear -> { m with overlays = Overlays.empty }
  | Smooth -> { m with smooth = not m.smooth }
  | Scale -> (
      match scale_key m with
      | Some k ->
          let flip = if Sset.mem k m.scales then Sset.remove k else Sset.add k in
          { m with scales = flip m.scales }
      | None -> m)
  | Narrow -> (
      match (m.depth, zoom_window m) with
      | Zoom z, Some (lo, hi) when (hi -. lo) /. 2. >= 1. ->
          { m with depth = Zoom { z with narrow = z.narrow + 1 } }
      | _ -> m)
  | Widen -> (
      match m.depth with
      | Zoom z when z.narrow > 0 -> { m with depth = Zoom { z with narrow = z.narrow - 1 } }
      | _ -> m)
  | Filter -> { m with layer = Filtering (Mosaic.Editor.of_string (Runs.text m.filter)) }
  | Keep_filter -> (
      match m.layer with
      | Filtering e -> { m with filter = Runs.filter (Mosaic.Editor.text e); layer = Screen }
      | _ -> m)
  | Clear_filter -> { m with filter = Runs.no_filter; layer = Screen }
  | Order -> { m with order = Runs.next m.order }
  | Fold -> (
      match (focus m, m.folds) with
      | Some f, Folded s ->
          let e = experiment f in
          let s = if Sset.mem e s then Sset.remove e s else Sset.add e s in
          { m with folds = Folded s }
      | _ -> m)
  | Info -> (
      match (m.layer, focus m) with
      | Info _, _ -> { m with layer = Screen }
      | _, Some _ -> { m with layer = Info (Mosaic.Scroll.at 0) }
      | _, None -> m)
  | Toggle_runs -> { m with runs_pane = toggle (shown_runs m) }
  | Toggle_run -> { m with run_pane = toggle (shown_run m) }
  | Help -> { m with layer = (match m.layer with Help -> Screen | _ -> Help) }
  | Close -> { m with layer = Screen }
  | Quit -> m

let act a m = settle (step { m with notice = None } a)
