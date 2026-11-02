(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Munin

type view = {
  data : Data.t;
  now : float;
  focus : Run.t;
  entry : Data.entry option; (* the focus's, with every key *)
  marked : (char * Run.t) list;
  metric : string option;
  chart : Chart.view;
  zoom_at : int option;
  filter : string;
  light : bool;
}

let name = Runs_screen.name
let word v r = Data.word v.data ~now:v.now r
let history v = Option.map (fun (e : Data.entry) -> e.history) v.entry

let watch v r =
  Option.map (fun (e : Data.entry) -> e.watch) (Data.entry v.data (Run.id r))

let keep ~filter keys =
  let words = Runs_screen.words filter in
  List.filter
    (fun k -> List.for_all (fun w -> Runs_screen.contains ~sub:w k) words)
    keys

let metrics v =
  match history v with
  | Some h -> keep ~filter:v.filter (History.metrics h)
  | None -> []

let stats v = Option.value (Data.stats v.data v.focus) ~default:[]
let latest_step v = Fact.latest_step (stats v)

let active v =
  let r = v.focus in
  let until =
    match (Run.ended_at r, Run.status r) with
    | Some e, _ -> e
    | None, (`Running | `Silent) -> v.now
    | None, _ -> Run.updated_at r
  in
  let pauses = match v.entry with Some e -> e.pauses | None -> [] in
  Fact.active_at pauses (Run.started_at r) until

(* The first line *)

let ago v t = Text.duration (v.now -. t) ^ " ago"

let state_line v ~wide =
  let r = v.focus in
  let w = word v r in
  let step = latest_step v in
  let steps = Run.steps r in
  let watch = watch v r in
  let arrival = Option.bind watch Watch.last_arrival in
  let progress ?bar ~wide () =
    match Fact.progress ?bar ~wide ~steps step with
    | "" -> []
    | p -> [ "step " ^ p ]
  in
  let watching () =
    match watch with
    | Some w ->
        [
          Printf.sprintf "watching %s, no sample yet"
            (Text.duration (v.now -. Watch.since w));
        ]
    | None -> []
  in
  let quiet () =
    match (Option.bind watch Watch.usual_gap, arrival) with
    | Some g, Some t ->
        [
          Printf.sprintf "no sample for %s, usually every %s"
            (Text.duration (v.now -. t))
            (Text.duration g);
        ]
    | _ -> []
  in
  let last_sample () =
    match arrival with Some t -> [ "last sample " ^ ago v t ] | None -> []
  in
  let last_write () = [ "last write " ^ ago v (Run.updated_at r) ] in
  let running () =
    if wide then [ "running " ^ Text.duration (active v) ] else []
  in
  let pace () =
    match Option.bind v.entry (fun e -> e.rate) with
    | Some x ->
        (Text.value x ^ if wide then " step/s" else "/s")
        ::
        (match (step, steps) with
        | Some s, Some t when s < t ->
            [ "~" ^ Text.duration (Float.of_int (t - s) /. x) ^ " left" ]
        | _ -> [])
    | None -> []
  in
  let ended () =
    match Run.ended_at r with
    | Some e ->
        [ Printf.sprintf "ended %s, %s" (Text.clock ~now:v.now e) (ago v e) ]
    | None -> []
  in
  let at_step () = match step with Some s -> Text.count s | None -> "0" in
  let facts =
    match w with
    | "starting" ->
        [
          Printf.sprintf "started %s, no sample yet" (ago v (Run.started_at r));
        ]
    | "running" -> progress ~wide () @ running () @ last_write () @ watching ()
    | "live" -> progress ~wide () @ pace () @ running () @ last_sample ()
    | "stalled" ->
        progress ~bar:false ~wide:true ()
        @ quiet () @ running () @ last_sample ()
    | "writing" ->
        progress ~bar:false ~wide:true ()
        @ pace () @ last_sample () @ [ "lock not probed" ]
    | "silent" ->
        progress ~bar:false ~wide:true ()
        @ (if arrival = None then last_write () @ watching () else quiet ())
        @ [ "lock not probed" ]
    | "crashed" ->
        progress ~bar:false ~wide:true ()
        @ [
            "writer exited without an end";
            Printf.sprintf "last write %s, %s"
              (Text.clock ~now:v.now (Run.updated_at r))
              (ago v (Run.updated_at r));
            "ran " ^ Text.duration (active v);
          ]
    | "finished" ->
        [
          Printf.sprintf "%s steps in %s" (at_step ())
            (Text.duration (active v));
        ]
        @ ended ()
    | "failed" ->
        let why = match Run.status r with `Failed e -> ": " ^ e | _ -> "" in
        [
          Printf.sprintf "failed at step %s after %s%s" (at_step ())
            (Text.duration (active v))
            why;
        ]
        @ ended ()
    | "killed" ->
        [
          Printf.sprintf "killed at step %s after %s" (at_step ())
            (Text.duration (active v));
        ]
        @ ended ()
    | _ -> []
  in
  if wide then
    String.concat "  "
      (Printf.sprintf "%-9s %-12s %-14s" w (name r) (Run.experiment r) :: facts)
  else String.concat "  " ((w ^ "  " ^ name r) :: facts)

(* The attention line *)

let non_finite_note ~focus_key label r stats =
  let firsts =
    List.filter_map
      (fun (k, (s : History.key_stats)) ->
        match s.stats.non_finite with
        | Some (_, f) when s.kind = `Metric -> Some (k, f)
        | _ -> None)
      stats
  in
  match List.partition (fun (k, _) -> Some k = focus_key) firsts with
  | [], [] -> None
  | (k, (f : Metric.sample)) :: _, rest | [], ((k, f) :: _ as rest) ->
      let rest = List.filter (fun (k', _) -> k' <> k) rest in
      let others =
        match
          List.sort
            (fun (_, (a : Metric.sample)) (_, b) -> compare a.step b.step)
            rest
        with
        | [] -> ""
        | (_, first) :: _ ->
            Printf.sprintf ", %d other metric%s from %s" (List.length rest)
              (if List.length rest = 1 then "" else "s")
              (Text.count first.step)
      in
      Some
        (Printf.sprintf "%s%s: %s %s from step %s%s" label (name r) k
           (Text.non_finite f.value) (Text.count f.step) others)

(* Without a timed arrival for longer than three of the floor's gaps. *)
let unseen = 30.

let attention v =
  let notes = ref [] in
  let add s = notes := s :: !notes in
  let r = v.focus in
  let stats r = Option.value (Data.stats v.data r) ~default:[] in
  Option.iter add (non_finite_note ~focus_key:v.metric "" r (stats r));
  List.iter
    (fun (l, mr) ->
      Option.iter add
        (non_finite_note ~focus_key:v.metric (Printf.sprintf "%c " l) mr
           (stats mr)))
    v.marked;
  (match (word v r, watch v r) with
  | "stalled", _ -> add (name r ^ " stalled")
  | "crashed", _ -> add (name r ^ " crashed")
  | ("running" | "starting"), Some w when v.now -. Watch.since w > unseen ->
      add
        (Printf.sprintf "no sample from %s in the %s since watching began"
           (name r)
           (Text.duration (v.now -. Watch.since w)))
  | _ -> ());
  List.iter
    (fun (l, mr) ->
      match word v mr with
      | ("crashed" | "stalled") as w ->
          add (Printf.sprintf "%c %s %s" l (name mr) w)
      | _ -> ())
    v.marked;
  (match history v with
  | Some h when History.damaged h > 0 ->
      add
        (Printf.sprintf "%s: %d damaged lines in its log" (name r)
           (History.damaged h))
  | _ -> ());
  match List.rev !notes with
  | [] -> None
  | l -> Some ("! " ^ String.concat " · " l)

(* Layout

   Run is a function of the terminal's size and the metrics: a grid of charts
   when every metric fits one, three to a line at most, each 40 columns by 8
   lines or more, with the system rows below; else one chart above one row per
   metric. *)

let chart_rows = 8
let chart_cols ~width = max 1 (min 3 (width / 40))

let devices v =
  match history v with
  | None -> []
  | Some h ->
      List.sort_uniq String.compare
        (List.filter_map
           (fun k ->
             if String.length k > 4 && String.sub k 0 4 = "dev/" then
               Option.map
                 (fun j -> String.sub k 4 (j - 4))
                 (String.index_from_opt k 4 '/')
             else None)
           (History.readings h))

(* The USE rows: a header, CPU, memory, two per device, disk. *)
let use_count v = 4 + (2 * List.length (devices v))

type layout =
  | Grid of { cols : int; rows : int; cw : int }
  | One of { chart_h : int; mrows : int; half : int; system : int }

let layout v ~width ~height =
  let n = List.length (metrics v) in
  let cols = chart_cols ~width in
  let rows = (n + cols - 1) / cols in
  if
    width >= 120 && n > 0
    && 4 + (rows * chart_rows) + 2 + 1 + use_count v + 2 <= height
  then Grid { cols; rows; cw = (width - (cols - 1)) / cols }
  else
    let system = 1 + List.length (devices v) in
    let mrows = (n + 1) / 2 in
    One
      {
        chart_h = height - 3 - mrows - system - 2;
        mrows;
        half = (width - 1) / 2;
        system;
      }

let grid_fits v ~width ~height =
  match layout v ~width ~height with Grid _ -> true | One _ -> false

let marked_ids v = List.map (fun (l, r) -> (l, Run.id r)) v.marked

let chart_request v spec =
  {
    Data.focus = Run.id v.focus;
    marked = marked_ids v;
    view = v.chart;
    spec;
    kept = false;
  }

let cursor v =
  let keys = metrics v in
  match v.metric with
  | Some k when List.mem k keys -> Some k
  | _ -> ( match keys with k :: _ -> Some k | [] -> None)

let charts v ~width ~height =
  let keys = metrics v in
  match layout v ~width ~height with
  | Grid { cols; rows; cw } ->
      List.mapi
        (fun i key ->
          let bottom = i / cols = rows - 1 in
          ( i,
            {
              Chart.key;
              width = cw;
              height = (chart_rows - 1 + if bottom then 2 else 0);
              x_axis = bottom;
            } ))
        keys
  | One { chart_h; _ } -> (
      match cursor v with
      | Some key when chart_h >= 4 ->
          [ (0, { Chart.key; width; height = chart_h; x_axis = true }) ]
      | _ -> [])

let trend_width = 4
let use_width = 44

let reading_keys v d =
  let k f = "dev/" ^ d ^ "/" ^ f in
  [ (k "busy_pct", Some (0., 100.)); (k "used_bytes", None) ]
  |> List.filter (fun (key, _) ->
      match history v with
      | Some h -> List.mem key (History.readings h)
      | None -> false)

let use_trends v =
  [
    ("sys/cpu_busy_pct", Some (0., 100.));
    ("sys/mem_used_bytes", None);
    ("sys/disk_busy_pct", Some (0., 100.));
  ]
  @ List.concat_map (reading_keys v) (devices v)

let trends v ~width ~height =
  let run = Run.id v.focus in
  match layout v ~width ~height with
  | Grid _ ->
      List.map
        (fun (key, domain) ->
          { Data.run; key; axis = Time; width = use_width; domain })
        (use_trends v)
  | One _ ->
      List.map
        (fun key ->
          {
            Data.run;
            key;
            axis = v.chart.axis;
            width = trend_width;
            domain = None;
          })
        (metrics v)

let zoom_request v =
  Option.map
    (fun key ->
      {
        Data.focus = Run.id v.focus;
        others = List.map snd (marked_ids v);
        key;
        axis = v.chart.axis;
        index = v.zoom_at;
      })
    (cursor v)

(* The keys of the charts drawn, which the marked runs keep. *)
let shown v ~width ~height ~zoom =
  if zoom then Option.to_list (cursor v)
  else List.map (fun (_, (s : Chart.spec)) -> s.key) (charts v ~width ~height)

let zoom_spec ~width ~height ~table =
  { Chart.key = ""; width; height = height - 4 - table - 2; x_axis = true }

let requests v ~width ~height ~zoom =
  if zoom then
    let table = if width >= 120 then 1 + 1 + List.length v.marked else 2 in
    let charts =
      match cursor v with
      | Some key ->
          [ chart_request v { (zoom_spec ~width ~height ~table) with key } ]
      | None -> []
    in
    (charts, [], zoom_request v)
  else
    ( List.map (fun (_, s) -> chart_request v s) (charts v ~width ~height),
      trends v ~width ~height,
      None )

(* Titles *)

let axis_name v key =
  match v.chart.axis with
  | Chart.Step -> "step"
  | Step_metric -> (
      match Option.bind (history v) (fun h -> History.step_metric h key) with
      | Some m -> m
      | None -> "step")
  | Time -> "time"

let legend_settings v ~key p =
  let smooth =
    match Option.bind p Chart.tau with
    | None -> "smoothing off"
    | Some t ->
        Printf.sprintf "τ %s %s" (Text.value t)
          (if v.chart.axis = Time then "s" else axis_name v key ^ "s")
  in
  let window =
    match v.chart.window with
    | None -> "window all"
    | Some (lo, hi) ->
        Printf.sprintf "window %s-%s" (Text.value lo) (Text.value hi)
  in
  Printf.sprintf "x %s · %s · %s" (axis_name v key) smooth window

let key_stats v key = List.assoc_opt key (stats v)

let chart_title v key ~cursor =
  let st = key_stats v key in
  let goal = Option.bind st (fun (s : History.key_stats) -> s.goal) in
  let latest =
    match st with Some s -> Text.value s.stats.last.value | None -> "—"
  in
  let best =
    match
      Option.bind st (fun (s : History.key_stats) ->
          Option.bind s.goal (fun g -> Metric.best g s.stats))
    with
    | Some b ->
        Printf.sprintf "  best %s @%s" (Text.value b.value) (Text.short b.step)
    | None -> ""
  in
  let notes =
    List.filter_map
      (fun (l, mr) ->
        match Option.bind (Data.stats v.data mr) (List.assoc_opt key) with
        | Some { History.stats = { non_finite = Some (_, f); _ }; _ } ->
            Some
              (Printf.sprintf "  %c %s %s" l (Text.non_finite f.value)
                 (Text.short f.step))
        | _ -> None)
      v.marked
  in
  let arrow = match Fact.arrow goal with "" -> "" | a -> " " ^ a in
  (if key = cursor then "▸" else " ")
  ^ key ^ arrow ^ " " ^ latest ^ best ^ String.concat "" notes

(* What a drawn chart adds to its title: the scale it fell back to, the samples
   it could not place, and the range a short plot cannot tick. *)
let drawn_notes v p (d : Chart.drawn) =
  (if v.chart.log then if d.log then "  log y" else "  linear: values <= 0"
   else "")
  ^ (match Chart.unpaired p with
    | 0 -> ""
    | n -> Printf.sprintf "  %s unpaired" (Text.count n))
  ^
  match d.range with
  | Some (lo, hi) ->
      Printf.sprintf "  y %s to %s" (Text.value lo) (Text.value hi)
  | None -> ""

let draw_chart ?cursor v spec g ~x ~y =
  Option.map
    (fun p -> (p, Chart.draw ?cursor ~light:v.light p g ~x ~y))
    (Data.chart v.data (chart_request v spec))

(* One row per metric: name, latest, trend, best. *)
let metric_row v key ~cursor ~width =
  let st = key_stats v key in
  let latest =
    match st with Some s -> Text.value s.stats.last.value | None -> "—"
  in
  let tr =
    Option.value
      ~default:(String.make trend_width ' ')
      (Data.trend v.data
         {
           run = Run.id v.focus;
           key;
           axis = v.chart.axis;
           width = trend_width;
           domain = None;
         })
  in
  let best =
    match st with
    | Some { goal = Some g; stats; _ } -> (
        match Metric.best g stats with
        | Some b -> Fact.arrow (Some g) ^ Text.value b.value
        | None -> "")
    | _ -> ""
  in
  let name_w = width - 26 in
  (if key = cursor then "▸" else " ")
  ^ Text.pad name_w (Text.elide name_w key)
  ^ " " ^ Text.pad_left 9 latest ^ " " ^ tr ^ " " ^ Text.pad 9 best

(* System rows *)

let reading v k =
  match key_stats v k with
  | Some s when Float.is_finite s.stats.last.value -> Some s.stats.last.value
  | _ -> None

let gibs v a b =
  match (reading v a, reading v b) with
  | Some a, Some b ->
      Some (Printf.sprintf "%s of %s GiB" (Text.gib a) (Text.gib b))
  | _ -> None

let load l = Munin_float_text.Float_text.g ~precision:2 l

let system_lines v ~wide ~width =
  let pct k = Option.map Text.percent (reading v k) in
  let join l = String.concat "  " (List.filter_map Fun.id l) in
  let cpu =
    join
      [
        Option.map (fun p -> "CPU " ^ p ^ " busy") (pct "sys/cpu_busy_pct");
        Option.map
          (fun l -> "load " ^ load l ^ "/core")
          (reading v "sys/load_per_core");
        Option.map
          (fun m -> "MEM " ^ m)
          (gibs v "sys/mem_used_bytes" "sys/mem_total_bytes");
        Option.map
          (fun s -> "swap " ^ Text.gib s ^ " GiB")
          (reading v "sys/swap_used_bytes");
      ]
  in
  let device d =
    let k f = "dev/" ^ d ^ "/" ^ f in
    join
      [
        Option.map (fun p -> d ^ " " ^ p ^ " busy") (pct (k "busy_pct"));
        Option.map
          (fun m -> "used " ^ m)
          (gibs v (k "used_bytes") (k "memory_bytes"));
        Option.map
          (fun a -> "alloc " ^ a ^ " budget")
          (gibs v (k "allocated_bytes") (k "budget_bytes"));
        (if wide then
           Option.map
             (fun t -> Printf.sprintf "%.0f °C" t)
             (reading v (k "temperature_c"))
         else None);
        (if wide then
           Option.map
             (fun t -> Printf.sprintf "%.0f W" t)
             (reading v (k "power_w"))
         else None);
      ]
  in
  List.map (Text.cut width)
    (List.filter (( <> ) "") (cpu :: List.map device (devices v)))

let use_rows v ~width =
  let tw = use_width in
  let trend ?domain k =
    Option.value ~default:""
      (Data.trend v.data
         { run = Run.id v.focus; key = k; axis = Time; width = tw; domain })
  in
  let row label u s t =
    let left = Text.pad 8 label ^ Text.pad 48 u ^ s in
    Text.pad (width - tw - 2) (Text.cut (width - tw - 2) left) ^ "  " ^ t
  in
  let bar p = Text.bar 16 (p /. 100.) in
  let opt f = function Some x -> f x | None -> "" in
  let spaced l = String.concat "   " (List.filter (( <> ) "") l) in
  let get = reading v in
  let interval =
    opt
      (fun s -> Printf.sprintf "readings every %ss" (Text.value s))
      (get "sys/interval_s")
  in
  let host =
    Option.value (Run.provenance v.focus).hostname ~default:"this machine"
  in
  let mem =
    match (get "sys/mem_used_bytes", get "sys/mem_total_bytes") with
    | Some u, Some t -> Some (u, t)
    | _ -> None
  in
  [
    Text.pad
      (width - tw - 2)
      (Printf.sprintf "SYSTEM  %s · %s · errors not logged" host interval)
    ^ "  over the run, "
    ^ Text.duration (active v);
    row "CPU"
      (opt
         (fun c -> Printf.sprintf "busy %s %4s" (bar c) (Text.percent c))
         (get "sys/cpu_busy_pct"))
      (spaced
         [
           opt
             (fun l -> "load " ^ load l ^ " per core")
             (get "sys/load_per_core");
           opt
             (fun c -> "process " ^ load c ^ " cores")
             (get "sys/proc_cpu_cores");
         ])
      (trend ~domain:(0., 100.) "sys/cpu_busy_pct");
    row "MEM"
      (opt
         (fun (u, t) ->
           Printf.sprintf "used %s %4s  %s of %s GiB"
             (bar (100. *. u /. t))
             (Text.percent (100. *. u /. t))
             (Text.gib u) (Text.gib t))
         mem)
      (spaced
         [
           opt
             (fun s -> "swap " ^ Text.gib s ^ " GiB")
             (get "sys/swap_used_bytes");
           opt
             (fun s -> "process " ^ Text.gib s ^ " GiB")
             (get "sys/proc_rss_bytes");
         ])
      (trend "sys/mem_used_bytes");
  ]
  @ List.concat_map
      (fun d ->
        let k f = "dev/" ^ d ^ "/" ^ f in
        let used =
          match (get (k "used_bytes"), get (k "memory_bytes")) with
          | Some u, Some m -> Some (u, m)
          | _ -> None
        in
        [
          row d
            (opt
               (fun b -> Printf.sprintf "busy %s %4s" (bar b) (Text.percent b))
               (get (k "busy_pct")))
            (spaced
               [
                 opt
                   (fun (u, m) ->
                     Printf.sprintf "used %s of %s GiB %s" (Text.gib u)
                       (Text.gib m)
                       (Text.percent (100. *. u /. m)))
                   used;
                 opt
                   (fun t -> Printf.sprintf "%.0f °C" t)
                   (get (k "temperature_c"));
                 opt (fun w -> Printf.sprintf "%.0f W" w) (get (k "power_w"));
               ])
            (trend ~domain:(0., 100.) (k "busy_pct"));
          row ""
            (match (get (k "allocated_bytes"), get (k "budget_bytes")) with
            | Some a, Some b ->
                Printf.sprintf "     alloc %s of %s GiB budget" (Text.gib a)
                  (Text.gib b)
            | Some a, None -> Printf.sprintf "     alloc %s GiB" (Text.gib a)
            | _ -> "")
            ""
            (trend (k "used_bytes"));
        ])
      (devices v)
  @ [
      row "DISK"
        (opt
           (fun b -> Printf.sprintf "busy %s %4s" (bar b) (Text.percent b))
           (get "sys/disk_busy_pct"))
        (spaced
           [
             opt
               (fun r -> "read " ^ Text.mbs r ^ " MB/s")
               (get "sys/disk_read_bytes_per_s");
             opt
               (fun w -> "write " ^ Text.mbs w ^ " MB/s")
               (get "sys/disk_write_bytes_per_s");
           ])
        (trend ~domain:(0., 100.) "sys/disk_busy_pct");
    ]

let legend v ~key p =
  let params r = Run.params r in
  let differing mr =
    List.filter_map
      (fun (k, x) ->
        if List.assoc_opt k (params v.focus) = Some x then None
        else Some (k ^ " " ^ Runs_screen.value_text x))
      (params mr)
  in
  ( String.concat "    "
      (Printf.sprintf "━ %s (focus)" (name v.focus)
      :: List.map
           (fun (l, mr) ->
             String.concat " "
               (Printf.sprintf "%c %s" l (name mr) :: differing mr))
           v.marked),
    legend_settings v ~key p )

let hint ~wide =
  if wide then
    "↑↓←→ metric  enter zoom  [ ] run  space mark  s smooth  x axis  L log  + \
     - window  < > pan  i info  y copy id  esc runs  ? help  q quit"
  else
    "↑↓←→ metric  enter zoom  [ ] run  space mark  s smooth  esc runs  ? help"

let paint v g ~width ~height =
  let wide = width >= 120 in
  Paint.text ~style:Paint.bold g ~x:0 ~y:0 ~w:width (state_line v ~wide);
  Option.iter
    (fun a -> Paint.text ~style:Paint.alert g ~x:0 ~y:1 ~w:width a)
    (attention v);
  Paint.text g ~x:0 ~y:(height - 1) ~w:width (hint ~wide);
  let keys = metrics v in
  match (history v, cursor v) with
  | None, _ -> Paint.text g ~x:0 ~y:3 ~w:width "reading the run"
  | Some _, None -> Paint.text g ~x:0 ~y:3 ~w:width "no metric sample yet"
  | Some _, Some cursor -> (
      let charts = charts v ~width ~height in
      match layout v ~width ~height with
      | Grid { cols; cw; _ } ->
          let first =
            Option.bind (List.nth_opt charts 0) (fun (_, s) ->
                Data.chart v.data (chart_request v s))
          in
          let names, settings_text = legend v ~key:cursor first in
          Paint.text g ~x:0 ~y:2 ~w:width
            (Text.pad (width - Text.width settings_text) names ^ settings_text);
          List.iter
            (fun (i, (spec : Chart.spec)) ->
              let x = i mod cols * (cw + 1)
              and y = 4 + (i / cols * chart_rows) in
              let notes =
                match draw_chart v spec g ~x ~y:(y + 1) with
                | Some (p, d) -> drawn_notes v p d
                | None -> "  reading"
              in
              Paint.text g ~x ~y ~w:cw (chart_title v spec.key ~cursor ^ notes))
            charts;
          let rows = (List.length keys + cols - 1) / cols in
          let y = 4 + (rows * chart_rows) + 3 in
          List.iteri
            (fun i l -> Paint.text g ~x:0 ~y:(y + i) ~w:width l)
            (use_rows v ~width)
      | One { chart_h; mrows; half; _ } ->
          let notes =
            match charts with
            | (_, spec) :: _ -> (
                match draw_chart v spec g ~x:0 ~y:3 with
                | Some (p, d) -> drawn_notes v p d
                | None -> "  reading")
            | [] -> ""
          in
          let title = chart_title v cursor ~cursor:"" ^ notes in
          let title = String.sub title 1 (String.length title - 1) in
          let labels =
            String.concat "  "
              (List.map
                 (fun (l, mr) -> Printf.sprintf "%c %s" l (name mr))
                 v.marked)
          in
          (* The axis is named when the title leaves room for it. *)
          let axis = "  x " ^ axis_name v cursor in
          let right =
            if
              Text.width title + Text.width labels + Text.width axis + 2
              <= width
            then labels ^ axis
            else labels
          in
          let room = width - Text.width right - 2 in
          Paint.text g ~x:0 ~y:2 ~w:width (Text.fit room title ^ "  " ^ right);
          let y0 = 3 + max 0 chart_h in
          for r = 0 to mrows - 1 do
            let a = List.nth keys r in
            Paint.text g ~x:0 ~y:(y0 + r) ~w:half
              (metric_row v a ~cursor ~width:half);
            Option.iter
              (fun b ->
                Paint.text g ~x:(half + 1) ~y:(y0 + r) ~w:half
                  (metric_row v b ~cursor ~width:half))
              (List.nth_opt keys (r + mrows))
          done;
          List.iteri
            (fun i l -> Paint.text g ~x:0 ~y:(y0 + mrows + i) ~w:width l)
            (system_lines v ~wide ~width))

(* Zoom *)

let paint_zoom v g ~width ~height =
  let wide = width >= 120 in
  Paint.text ~style:Paint.bold g ~x:0 ~y:0 ~w:width (state_line v ~wide);
  Paint.text g ~x:0 ~y:(height - 1) ~w:width
    (if wide then
       "↑↓ metric  ←→ cursor  g G first last  s smooth  x axis  L log  + - \
        window  < > pan  0 reset  [ ] run  space mark  i info  esc back  ? \
        help  q quit"
     else
       "↑↓ metric  ←→ cursor  s smooth  x axis  L log  + - window  esc back  ? \
        help");
  match cursor v with
  | None -> Paint.text g ~x:0 ~y:2 ~w:width "no metric to zoom"
  | Some key ->
      let cur =
        match Data.cursor v.data with
        | Some c when c.at.key = key && c.at.focus = Run.id v.focus -> Some c
        | _ -> None
      in
      let at = Option.bind cur (fun (c : Data.cursor) -> c.x) in
      let value id =
        Option.bind cur (fun (c : Data.cursor) ->
            Option.join (List.assoc_opt id c.values))
      in
      let at_text =
        match at with Some a -> Text.count (Float.to_int a) | None -> "—"
      in
      let goal = Option.bind (key_stats v key) (fun s -> s.goal) in
      let all = ('-', v.focus) :: v.marked in
      let mark l = if l = '-' then "━" else String.make 1 l in
      let best r =
        match Option.bind (Data.stats v.data r) (List.assoc_opt key) with
        | Some { History.goal = Some g; stats; _ } -> Metric.best g stats
        | _ -> None
      in
      let table =
        if wide then
          Printf.sprintf "  %s %-14s %10s  %10s  %8s  %16s  %8s  %10s  %7s"
            (Text.pad 3 "") "RUN" "LATEST" "BEST" "AT"
            ("AT " ^ String.uppercase_ascii (axis_name v key) ^ " " ^ at_text)
            "SAMPLES" "NONFINITE" "ORPHANS"
          :: List.map
               (fun (l, r) ->
                 let st =
                   Option.bind (Data.stats v.data r) (List.assoc_opt key)
                 in
                 let count, nonfinite =
                   match st with
                   | Some s ->
                       ( s.stats.count,
                         match s.stats.non_finite with
                         | Some (n, _) -> n
                         | None -> 0 )
                   | None -> (0, 0)
                 in
                 let orphans =
                   match Data.entry v.data (Run.id r) with
                   | Some e -> (
                       match History.orphans e.history key with
                       | o -> Metric.length o
                       | exception Invalid_argument _ -> 0)
                   | None -> 0
                 in
                 Printf.sprintf "  %s %s %10s  %10s  %8s  %16s  %8d  %10d  %7d"
                   (Text.pad 3 (mark l))
                   (Text.fit 14 (name r))
                   (match st with
                   | Some s -> Text.value s.stats.last.value
                   | None -> "—")
                   (match best r with
                   | Some b -> Text.value b.value
                   | None -> "—")
                   (match best r with
                   | Some b -> Text.short b.step
                   | None -> "—")
                   (Option.fold ~none:"—" ~some:Text.value (value (Run.id r)))
                   count nonfinite orphans)
               all
        else
          [
            Printf.sprintf "at %s %s:  %s" (axis_name v key) at_text
              (String.concat "  "
                 (List.map
                    (fun (l, r) ->
                      mark l ^ " "
                      ^ Option.fold ~none:"—" ~some:Text.value
                          (value (Run.id r)))
                    all));
            "best:  "
            ^ String.concat "  "
                (List.map
                   (fun (l, r) ->
                     mark l ^ " "
                     ^
                     match best r with
                     | Some b -> Text.value b.value ^ " @" ^ Text.short b.step
                     | None -> "—")
                   all);
          ]
      in
      let charts, _, _ = requests v ~width ~height ~zoom:true in
      let p = match charts with c :: _ -> Data.chart v.data c | [] -> None in
      let notes =
        match (charts, p) with
        | c :: _, Some p when c.spec.height >= 4 ->
            drawn_notes v p (Chart.draw ?cursor:at ~light:v.light p g ~x:0 ~y:3)
        | _ -> ""
      in
      let head =
        let goal =
          match goal with
          | Some `Minimize -> "↓ minimize"
          | Some `Maximize -> "↑ maximize"
          | None -> "no goal"
        in
        if wide then
          Printf.sprintf "%s  %s · %s · cursor at %s %s" key goal
            (legend_settings v ~key p) (axis_name v key) at_text
        else
          Printf.sprintf "%s %s  %s" key
            (Fact.arrow (Option.bind (key_stats v key) (fun s -> s.goal)))
            (legend_settings v ~key p)
      in
      Paint.text g ~x:0 ~y:1 ~w:width (head ^ notes);
      let chart_h = match charts with c :: _ -> c.spec.height | [] -> 0 in
      List.iteri
        (fun i l -> Paint.text g ~x:0 ~y:(3 + chart_h + 1 + i) ~w:width l)
        table
