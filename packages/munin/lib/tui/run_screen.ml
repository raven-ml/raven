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

let stats_of v r = Option.value (Data.stats v.data r) ~default:[]
let stats v = stats_of v v.focus
let key_stats v key = List.assoc_opt key (stats v)
let latest_step v = Fact.latest_step (stats v)

let cursor v =
  let keys = metrics v in
  match v.metric with
  | Some k when List.mem k keys -> Some k
  | _ -> ( match keys with k :: _ -> Some k | [] -> None)

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

let margin ~width = if width >= 100 then 2 else 1

(* The first line: the run in words. Its parts leave whole when the line is too
   long, the rate and the time left first; the progress and the last sample or
   write stay. *)

let ago v t = Text.duration (v.now -. t) ^ " ago"

let sentence v ~width =
  let r = v.focus in
  let w = word v r in
  let step = latest_step v in
  let steps = Run.steps r in
  let watch = watch v r in
  let arrival = Option.bind watch Watch.last_arrival in
  let wide = width >= 100 in
  let plain s = (Paint.plain, s) and dim s = (Paint.dim, s) in
  let progress () =
    match (step, steps) with
    | Some s, Some t when t > 0 && s < t ->
        let pct = Text.percent_of s t in
        if wide then
          [
            [ plain ("step " ^ Text.count s); dim (" of " ^ Text.count t) ];
            [ plain pct ];
          ]
        else
          [
            [ plain (Text.count s); dim ("/" ^ Text.count t ^ " "); plain pct ];
          ]
    | Some s, _ -> [ [ plain ("step " ^ Text.count s) ] ]
    | None, _ -> []
  in
  let watching () =
    match watch with
    | Some w ->
        [
          [
            dim "watching ";
            plain (Text.duration (v.now -. Watch.since w));
            dim ", no sample yet";
          ];
        ]
    | None -> []
  in
  let quiet () =
    match (Option.bind watch Watch.usual_gap, arrival) with
    | Some g, Some t ->
        [
          [
            dim "no sample for ";
            plain (Text.duration (v.now -. t));
            dim ", usually every ";
            plain (Text.duration g);
          ];
        ]
    | _ -> []
  in
  let last_sample () =
    match arrival with
    | Some t -> [ [ dim "last sample "; plain (ago v t) ] ]
    | None -> []
  in
  let last_write () =
    [ [ dim "last write "; plain (ago v (Run.updated_at r)) ] ]
  in
  let ran () =
    if width >= 140 then [ [ dim "ran "; plain (Text.duration (active v)) ] ]
    else []
  in
  let pace () =
    match Option.bind v.entry (fun e -> e.rate) with
    | Some x -> (
        [
          [
            plain (Munin_float_text.Float_print.g ~precision:2 x);
            dim (if wide then " steps/s" else "/s");
          ];
        ]
        @
        match (step, steps) with
        | Some s, Some t when s < t ->
            [
              [
                plain ("~" ^ Text.duration (Float.of_int (t - s) /. x));
                dim " left";
              ];
            ]
        | _ -> [])
    | None -> []
  in
  let ended () =
    match Run.ended_at r with
    | Some e ->
        [
          [
            dim "ended "; plain (Text.clock ~now:v.now e); dim (", " ^ ago v e);
          ];
        ]
    | None -> []
  in
  let at_step = match step with Some s -> Text.count s | None -> "0" in
  let unprobed = [ [ dim "lock not probed" ] ] in
  let rate = pace () in
  let facts =
    match w with
    | "starting" ->
        [
          [
            dim "started ";
            plain (ago v (Run.started_at r));
            dim ", no sample yet";
          ];
        ]
    | "running" -> progress () @ ran () @ last_write () @ watching ()
    | "live" -> progress () @ rate @ ran () @ last_sample ()
    | "stalled" -> progress () @ quiet () @ ran () @ last_sample ()
    | "writing" -> progress () @ rate @ last_sample () @ unprobed
    | "silent" ->
        progress ()
        @ (if arrival = None then last_write () @ watching () else quiet ())
        @ unprobed
    | "crashed" ->
        progress ()
        @ [ [ plain "writer exited without an end" ] ]
        @ [
            [
              dim "last write ";
              plain (Text.clock ~now:v.now (Run.updated_at r));
              dim (", " ^ ago v (Run.updated_at r));
            ];
          ]
        @ [ [ dim "ran "; plain (Text.duration (active v)) ] ]
    | "finished" ->
        [
          [
            plain (at_step ^ " steps");
            dim " in ";
            plain (Text.duration (active v));
          ];
        ]
        @ ended ()
    | "failed" ->
        let why =
          match Run.status r with `Failed e -> [ [ plain e ] ] | _ -> []
        in
        [
          [
            dim "failed at step ";
            plain at_step;
            dim " after ";
            plain (Text.duration (active v));
          ];
        ]
        @ why @ ended ()
    | "killed" ->
        [
          [
            dim "killed at step ";
            plain at_step;
            dim " after ";
            plain (Text.duration (active v));
          ];
        ]
        @ ended ()
    | _ -> []
  in
  let head =
    Paint.word w
    @ [ plain "  "; (Paint.bold, name r) ]
    @ (if wide then [ dim ("  " ^ Run.experiment r) ] else [])
    @ [ plain "   " ]
  in
  let drop =
    match (w, rate) with
    | ("live" | "writing"), _ :: _ ->
        let p = List.length (progress ()) in
        List.init (List.length rate) (fun i -> p + i)
    | _ -> []
  in
  Paint.fit ~drop ~w:(width - (2 * margin ~width)) head facts

(* The attention line: what needs attention, in words. Nothing when nothing
   does. *)

(* Without a timed arrival for longer than three of the floor's gaps. *)
let unseen = 30.

let non_finite_note ~cursor label stats =
  let firsts =
    List.filter_map
      (fun (k, (s : History.key_stats)) ->
        match s.stats.non_finite with
        | Some (_, f) when s.kind = `Metric -> Some (k, f)
        | _ -> None)
      stats
  in
  match
    List.sort
      (fun (ka, (a : Metric.sample)) (kb, b) ->
        compare (a.step, Some ka <> cursor) (b.step, Some kb <> cursor))
      firsts
  with
  | [] -> None
  | (k, f) :: rest ->
      let others = List.length rest in
      Some
        (Printf.sprintf "%s%s %s from step %s%s" label k
           (Text.non_finite f.value) (Text.count f.step)
           (if others = 0 then ""
            else
              Printf.sprintf " (and %d other metric%s)" others
                (if others = 1 then "" else "s")))

let attention v =
  let notes = ref [] in
  let add s = notes := s :: !notes in
  let r = v.focus in
  let cursor = cursor v in
  Option.iter add (non_finite_note ~cursor "" (stats v));
  List.iter
    (fun (l, mr) ->
      Option.iter add
        (non_finite_note ~cursor
           (Printf.sprintf "%c %s: " l (name mr))
           (stats_of v mr)))
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
  List.rev !notes

(* Layout

   Run, top to bottom: the sentence, the attention line, the metric's title and
   the compared runs' legend, the chart, a pair of difference rows per marked
   run, a strip per metric aligned under the chart's plot and grouped by prefix,
   the machine, the keys. The chart keeps three fifths of the rows the rest
   leaves, four fifths below 30 rows, where it carries the comparison. *)

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

let roomy ~height = height >= 30
let large ~width ~height = width >= 140 && height >= 44

let sys_rows v ~height =
  match history v with
  | Some h when History.readings h <> [] ->
      if roomy ~height then 1 + 2 + (2 * List.length (devices v)) + 1 else 1
  | _ -> 0

type layout = {
  title_y : int;
  legend : bool;
  chart_y : int;
  chart_h : int;
  caption : bool;
  versus_y : int;
  name_w : int; (* the label column, which ends where the plot starts *)
  value_w : int;
  strips_y : int;
  strip_rows : (int * string) list;
      (* each strip shown: its row offset and key *)
  strip_h : int;
  hidden : (int * int) option;
      (* the first and last strip shown, when not all *)
  sys_y : int;
  sys_rows : int;
}

let groups keys =
  List.fold_left
    (fun acc k ->
      let p, _ = Paint.key k in
      match List.rev acc with
      | (p', ks) :: rest when p' = p -> List.rev ((p', ks @ [ k ]) :: rest)
      | _ -> acc @ [ (p, [ k ]) ])
    [] keys

(* The strips' rows: two rows each with a blank row between groups when all fit
   so, else one row with the blanks, else one row without, else a window of
   one-row strips around the cursor. *)
let strip_plan keys ~room ~roomy ~large ~cursor =
  let gs = groups keys in
  let lay ~h ~gaps =
    let _, rows =
      List.fold_left
        (fun (y, acc) (i, (_, ks)) ->
          let y = if gaps && i > 0 then y + 1 else y in
          List.fold_left (fun (y, acc) k -> (y + h, (y, k) :: acc)) (y, acc) ks)
        (0, [])
        (List.mapi (fun i g -> (i, g)) gs)
    in
    let rows = List.rev rows in
    (rows, match List.rev rows with (y, _) :: _ -> y + h | [] -> 0)
  in
  let options =
    (if large then [ (2, true) ] else [])
    @ (if roomy then [ (1, true) ] else [])
    @ [ (1, false) ]
  in
  match
    List.find_map
      (fun (h, gaps) ->
        let rows, total = lay ~h ~gaps in
        if total <= room then Some (rows, h) else None)
      options
  with
  | Some (rows, h) -> (rows, h, None)
  | None ->
      let n = List.length keys and shown = max 0 room in
      let ci = Option.value (List.find_index (( = ) cursor) keys) ~default:0 in
      let first = max 0 (min (n - shown) (ci - (shown / 2))) in
      let ks = List.filteri (fun i _ -> i >= first && i < first + shown) keys in
      ( List.mapi (fun i k -> (i, k)) ks,
        1,
        if ks = [] then None else Some (first, first + List.length ks - 1) )

let layout v ~width ~height =
  let keys = metrics v in
  let roomy = roomy ~height in
  let marks = List.length v.marked in
  let title_y = if roomy then 3 else 2 in
  let chart_y = title_y + 1 + if marks > 0 then 1 else 0 in
  let caption = roomy && marks > 0 in
  let versus = (if caption then 1 else 0) + (2 * marks) in
  let sys = sys_rows v ~height in
  let footer = height - 1 in
  let sys_y = footer - (if roomy then 1 else 0) - sys in
  let below = if sys > 0 then sys_y - 1 else footer - 1 in
  let room = below - chart_y - versus in
  let keep = if roomy then room * 3 / 5 else room * 4 / 5 in
  let strip_room =
    if List.length keys <= 1 then 0 else max 0 (room - max 6 keep - 1)
  in
  let strip_rows, strip_h, hidden =
    strip_plan keys ~room:strip_room ~roomy ~large:(large ~width ~height)
      ~cursor:(Option.value (cursor v) ~default:"")
  in
  let strip_rows = if List.length keys <= 1 then [] else strip_rows in
  let strips_h =
    List.fold_left (fun m (y, _) -> max m (y + strip_h)) 0 strip_rows
  in
  let chart_h = room - if strips_h > 0 then strips_h + 1 else 0 in
  (* The label column holds the strips' names and the difference rows' marked
     runs, and below 30 lines what a difference row is. *)
  let longest =
    List.fold_left
      (fun a k ->
        let p, rest = Paint.key k in
        max a (Text.width p + 1 + Text.width rest))
      0 keys
  in
  let longest =
    List.fold_left
      (fun a (_, r) -> max a (Text.width (name r) + 2))
      longest v.marked
  in
  let longest = if marks > 0 && not caption then max longest 11 else longest in
  {
    title_y;
    legend = marks > 0;
    chart_y;
    chart_h;
    caption;
    versus_y = (chart_y + chart_h + if caption then 1 else 0);
    name_w = max Chart.left (min 22 (longest + 3));
    (* Wide enough, with marked runs, for a verdict and its step. *)
    value_w = (if marks > 0 then 17 else if width >= 120 then 12 else 10);
    strips_y = chart_y + chart_h + versus + 1;
    strip_rows;
    strip_h;
    hidden;
    sys_y;
    sys_rows = sys;
  }

(* The chart's region starts so that its label margin ends where the labels of
   the strips do, and leaves their value column at the right. *)
let chart_x ~width l = margin ~width + l.name_w - Chart.left

let chart_w ~width l =
  width - (2 * margin ~width) - (l.name_w - Chart.left) - l.value_w

let chart_request v ~marks spec =
  {
    Data.focus = Run.id v.focus;
    marked =
      (if marks then List.map (fun (l, r) -> (l, Run.id r)) v.marked else []);
    view = v.chart;
    spec;
    kept = false;
  }

let main_spec ~width l key =
  { Chart.key; width = chart_w ~width l; height = l.chart_h; x_axis = true }

(* A strip is a chart of [strip_h] plot rows, drawn inline: its marks go in its
   top row's blank cells. *)
let strip_spec ~width l key =
  {
    Chart.key;
    width = chart_w ~width l;
    height = l.strip_h + 1;
    x_axis = false;
  }

let trend_w ~width = if width >= 140 then 40 else 0

let use_keys v =
  [ ("sys/cpu_busy_pct", Some (0., 100.)); ("sys/mem_used_bytes", None) ]
  @ List.concat_map
      (fun d ->
        [
          ("dev/" ^ d ^ "/busy_pct", Some (0., 100.));
          ("dev/" ^ d ^ "/used_bytes", None);
        ])
      (devices v)

let zoom_spec v ~width ~height key =
  {
    Chart.key;
    width = width - (2 * margin ~width);
    height = height - 4 - (2 + List.length v.marked) - 2;
    x_axis = true;
  }

let zoom_request v =
  Option.map
    (fun key ->
      {
        Data.focus = Run.id v.focus;
        others = List.map (fun (_, r) -> Run.id r) v.marked;
        key;
        axis = v.chart.axis;
        index = v.zoom_at;
      })
    (cursor v)

(* The marked runs are drawn in the cursor's chart alone, on Run and on Zoom. *)
let shown v = Option.to_list (cursor v)

let requests v ~width ~height ~zoom =
  match cursor v with
  | None -> ([], [], None)
  | Some key when zoom ->
      ( [ chart_request v ~marks:true (zoom_spec v ~width ~height key) ],
        [],
        zoom_request v )
  | Some key ->
      let l = layout v ~width ~height in
      let strips =
        List.map
          (fun (_, k) -> chart_request v ~marks:false (strip_spec ~width l k))
          l.strip_rows
      in
      let trends =
        if trend_w ~width = 0 || l.sys_rows <= 1 then []
        else
          List.map
            (fun (k, domain) ->
              {
                Data.run = Run.id v.focus;
                key = k;
                axis = Time;
                width = trend_w ~width;
                domain;
              })
            (use_keys v)
      in
      ( chart_request v ~marks:true (main_spec ~width l key) :: strips,
        trends,
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

let arrow (s : History.key_stats option) =
  match Fact.arrow (Option.bind s (fun (s : History.key_stats) -> s.goal)) with
  | "" -> ""
  | a -> " " ^ a

let title v key p =
  let st = key_stats v key in
  Paint.key_spans key
  @ [ (Paint.dim, arrow st); (Paint.plain, "   ") ]
  @ (match st with
    | Some s -> (
        let x = s.stats.last.value in
        [
          ((if Float.is_finite x then Paint.bold else Paint.alert), Text.value x);
        ]
        @
        match Option.bind s.goal (fun g -> Metric.best g s.stats) with
        | Some b ->
            [
              (Paint.dim, "   best ");
              (Paint.plain, Text.value b.value);
              (Paint.dim, " at " ^ Text.count b.step);
            ]
        | None -> [])
    | None -> [ (Paint.dim, "—") ])
  @ (match Option.bind p Chart.tau with
    | Some t ->
        [
          ( Paint.dim,
            Printf.sprintf "   τ %s %s" (Text.value t)
              (if v.chart.axis = Time then "s" else axis_name v key ^ "s") );
        ]
    | None -> [])
  @
  match Option.map Chart.unpaired p with
  | Some n when n > 0 ->
      [ (Paint.dim, Printf.sprintf "   %s unpaired" (Text.count n)) ]
  | _ -> []

(* The settings that differ from the defaults; defaults are not written. A log y
   asked for over values not all above 0 says it stays linear. *)
let settings v key p =
  String.concat " · "
    ((match v.chart.axis with
       | Step -> []
       | Step_metric -> [ "x " ^ axis_name v key ]
       | Time -> [ "x active time" ])
    @ (if v.filter <> "" then [ Printf.sprintf "metrics with \"%s\"" v.filter ]
       else [])
    @ (if v.chart.smooth > 0 then [ "smoothed" ] else [])
    @ (if v.chart.log then
         [
           (match p with
           | Some p when not (Chart.log_y p) -> "linear: values <= 0"
           | _ -> "log y");
         ]
       else [])
    @
    match v.chart.window with
    | Some (lo, hi) ->
        [
          Printf.sprintf "%ss %s–%s" (axis_name v key)
            (Text.count (Float.to_int lo))
            (Text.count (Float.to_int hi));
        ]
    | None -> [])

(* The compared runs: each letter, its name, and the params in which it differs
   from the focus. Letters and words: braille draws no dash. *)
let legend v =
  let differing mr =
    List.filter_map
      (fun (k, x) ->
        if List.assoc_opt k (Run.params v.focus) = Some x then None
        else Some (k ^ " " ^ Runs_screen.value_text x))
      (Run.params mr)
  in
  [ (Paint.plain, "━ "); (Paint.dim, name v.focus) ]
  @ List.concat_map
      (fun (l, mr) ->
        [
          (Paint.plain, "   ");
          (Paint.mark l, String.make 1 l ^ " ");
          (Paint.dim, name mr);
        ]
        @
        match List.filteri (fun i _ -> i < 2) (differing mr) with
        | [] -> []
        | ps -> [ (Paint.plain, "  " ^ String.concat "  " ps) ])
      v.marked

(* Machine *)

let reading v k =
  match key_stats v k with
  | Some s when Float.is_finite s.stats.last.value -> Some s.stats.last.value
  | _ -> None

let use_parts v =
  let opt f k = Option.map f (reading v k) in
  let gibs a b =
    match (reading v a, reading v b) with
    | Some a, Some b -> Some (Text.gib a ^ " of " ^ Text.gib b ^ " GiB")
    | _ -> None
  in
  let load l = Munin_float_text.Float_print.g ~precision:2 l in
  let somes = List.filter_map Fun.id in
  let cpu =
    somes
      [
        opt (fun p -> Text.percent p ^ " busy") "sys/cpu_busy_pct";
        opt (fun l -> "load " ^ load l ^ " per core") "sys/load_per_core";
        opt (fun c -> "process " ^ load c ^ " cores") "sys/proc_cpu_cores";
      ]
  in
  let mem =
    somes
      [
        gibs "sys/mem_used_bytes" "sys/mem_total_bytes";
        opt (fun s -> "swap " ^ Text.gib s ^ " GiB") "sys/swap_used_bytes";
        opt (fun s -> "process " ^ Text.gib s ^ " GiB") "sys/proc_rss_bytes";
      ]
  in
  let disk =
    somes
      [
        opt
          (fun x -> "read " ^ Text.mbs x ^ " MB/s")
          "sys/disk_read_bytes_per_s";
        opt
          (fun x -> "write " ^ Text.mbs x ^ " MB/s")
          "sys/disk_write_bytes_per_s";
        opt (fun p -> Text.percent p ^ " busy") "sys/disk_busy_pct";
      ]
  in
  let device d =
    let k f = "dev/" ^ d ^ "/" ^ f in
    ( somes
        [
          opt (fun p -> Text.percent p ^ " busy") (k "busy_pct");
          opt
            (fun t -> Munin_float_text.Float_print.f 0 t ^ " °C")
            (k "temperature_c");
          opt (fun t -> Munin_float_text.Float_print.f 0 t ^ " W") (k "power_w");
        ],
      somes
        [
          Option.map
            (fun s -> "used " ^ s)
            (gibs (k "used_bytes") (k "memory_bytes"));
          Option.map
            (fun s -> "allocated " ^ s ^ " budget")
            (gibs (k "allocated_bytes") (k "budget_bytes"));
        ] )
  in
  (cpu, mem, disk, List.map (fun d -> (d, device d)) (devices v))

(* When the readings are not of now: the time of the last one, for a run that is
   not live. *)
let read_at v =
  match word v v.focus with
  | "live" | "running" | "writing" | "starting" -> None
  | _ ->
      Option.map
        (fun (s : History.key_stats) -> s.stats.last.time)
        (key_stats v "sys/cpu_busy_pct")

let paint_use v g l ~width =
  let mx = margin ~width in
  let w = width - (2 * mx) in
  let cpu, mem, disk, devs = use_parts v in
  let text l = String.concat " · " l in
  let past = read_at v in
  if l.sys_rows = 1 then begin
    let first l = List.filteri (fun i _ -> i = 0) l in
    let head =
      match past with
      | Some t ->
          [
            (Paint.dim, "read ");
            (Paint.plain, Text.duration (v.now -. t) ^ " ago");
            (Paint.dim, ": ");
          ]
      | None -> []
    in
    let parts =
      [
        [ (Paint.dim, "CPU "); (Paint.plain, text (first cpu)) ];
        [ (Paint.dim, "MEM "); (Paint.plain, text (first mem)) ];
      ]
      @ List.concat_map
          (fun (d, (u, mm)) ->
            [
              [ (Paint.dim, d ^ " "); (Paint.plain, text (first u)) ];
              [ (Paint.plain, text (first mm)) ];
            ])
          devs
    in
    Paint.spans g ~x:mx ~y:l.sys_y ~w (Paint.fit ~w head parts)
  end
  else begin
    let host =
      Option.value (Run.provenance v.focus).hostname ~default:"this machine"
    in
    let tw = trend_w ~width in
    let head =
      [ (Paint.dim, "machine "); (Paint.plain, host); Paint.sep ]
      @ (match past with
        | Some t ->
            [
              ( Paint.plain,
                Printf.sprintf "last read %s, %s" (Text.clock ~now:v.now t)
                  (ago v t) );
            ]
        | None -> (
            match reading v "sys/interval_s" with
            | Some s ->
                [
                  ( Paint.dim,
                    Printf.sprintf "read every %ss"
                      (Munin_float_text.Float_print.g ~precision:3 s) );
                ]
            | None -> []))
      @ [ Paint.sep; (Paint.dim, "errors not logged") ]
    in
    Paint.spans g ~x:mx ~y:l.sys_y ~w head;
    if tw > 0 then
      Paint.right g ~x:mx ~y:l.sys_y ~w [ (Paint.dim, "over the run, by time") ];
    let label = 10 in
    let row y name parts trend =
      Paint.text ~style:Paint.dim g ~x:mx ~y ~w:label name;
      Paint.spans g ~x:(mx + label) ~y
        ~w:(w - label - tw - 2)
        (Paint.join (List.map (fun s -> [ (Paint.plain, s) ]) parts));
      match trend with
      | Some (k, domain) when tw > 0 -> (
          match
            Data.trend v.data
              {
                Data.run = Run.id v.focus;
                key = k;
                axis = Time;
                width = tw;
                domain;
              }
          with
          | Some t -> Paint.text ~style:Paint.dim g ~x:(mx + w - tw) ~y ~w:tw t
          | None -> ())
      | _ -> ()
    in
    let y = l.sys_y + 1 in
    row y "CPU" cpu (Some ("sys/cpu_busy_pct", Some (0., 100.)));
    row (y + 1) "memory" mem (Some ("sys/mem_used_bytes", None));
    List.iteri
      (fun i (d, (u, mm)) ->
        let y = y + 2 + (2 * i) in
        row y d u (Some ("dev/" ^ d ^ "/busy_pct", Some (0., 100.)));
        row (y + 1) "" mm (Some ("dev/" ^ d ^ "/used_bytes", None)))
      devs;
    row (y + 2 + (2 * List.length devs)) "disk" disk None
  end

(* Strips *)

let paint_strip v g l key ~width ~y ~cursor ~first =
  let mx = margin ~width in
  let sel = cursor = key in
  let spec = strip_spec ~width l key in
  (* The cursor's strip in the foreground, the others dim. *)
  (match Data.chart v.data (chart_request v ~marks:false spec) with
  | Some c ->
      ignore
        (Chart.draw ~inline:true ~faint:(not sel) ~light:v.light c g
           ~x:(chart_x ~width l) ~y)
  | None -> ());
  let p, rest = Paint.key key in
  Paint.spans g ~x:mx ~y ~w:(l.name_w - 1)
    [
      (Paint.plain, if sel then "› " else "  ");
      (Paint.dim, if first then p else String.make (Text.width p) ' ');
      ( (if sel then Paint.bold else Paint.plain),
        Text.cut (max 1 (l.name_w - 3 - Text.width p)) rest );
    ];
  let value =
    match key_stats v key with
    | Some s ->
        let x = s.stats.last.value in
        [
          ( (if not (Float.is_finite x) then Paint.bad
             else if sel then Paint.bold
             else Paint.plain),
            Text.value x );
        ]
    | None -> [ (Paint.dim, "—") ]
  in
  Paint.right g ~x:(mx + width - (2 * mx) - l.value_w) ~y ~w:l.value_w value

(* Under the chart, each marked run against the focus: what the overlay hides
   where the two lines share cells. *)
let paint_versus v g l c (placed : Chart.placed) key ~width =
  let mx = margin ~width in
  let w = width - (2 * mx) in
  let goal = Option.bind (key_stats v key) (fun s -> s.goal) in
  if l.caption then
    Paint.spans g ~x:mx ~y:(l.versus_y - 1) ~w
      [ (Paint.dim, "focus minus each marked run, at the same step") ];
  List.iteri
    (fun i (m, mr) ->
      let y = l.versus_y + (2 * i) in
      Paint.spans g ~x:mx ~y ~w:(l.name_w - 1)
        [
          (Paint.plain, "  ");
          (Paint.mark m, String.make 1 m ^ " ");
          (Paint.dim, name mr);
        ];
      match Chart.draw_versus ~light:v.light c m g ~under:placed ~y ~rows:2 with
      | None -> ()
      | Some d ->
          let worse =
            match goal with
            | Some `Minimize -> d.last > 0.
            | Some `Maximize -> d.last < 0.
            | None -> false
          in
          let verdict =
            match goal with
            | None -> ""
            | Some _ ->
                if d.last = 0. then "level"
                else if worse then "worse"
                else "better"
          in
          (* The row's meaning where no caption states it, and its scale: the
             difference its edges stand for. *)
          let what = Printf.sprintf "focus − %c" m
          and scale = "±" ^ Text.value d.scale in
          let room = l.name_w - 5 in
          let label =
            if l.caption then scale
            else if Text.width what + 2 + Text.width scale <= room then
              what ^ "  " ^ scale
            else what
          in
          Paint.text ~style:Paint.dim g ~x:(mx + 4) ~y:(y + 1) ~w:(l.name_w - 5)
            label;
          (* A marked run that stops before the focus's latest step says where,
             after the end of its row, so its value reads as of that step. *)
          let early =
            match List.assoc_opt key (stats_of v mr) with
            | Some { History.stats = { non_finite = Some (_, f); _ }; _ } ->
                Some
                  (Printf.sprintf "%c %s from step %s" m
                     (Text.non_finite f.value) (Text.count f.step))
            | Some s when Some s.stats.last.step < latest_step v ->
                Some
                  (Printf.sprintf "%c ends at step %s" m
                     (Text.count s.stats.last.step))
            | _ -> None
          in
          (match early with
          | Some note ->
              let ex =
                placed.px + 2
                + Float.to_int
                    ((d.at -. Chart.lo c)
                    /. (Chart.hi c -. Chart.lo c)
                    *. Float.of_int placed.pw)
              in
              let room = placed.px + placed.pw - ex in
              let ny = if d.zero_row = y then y + 1 else y in
              if room >= Text.width note then
                Paint.text ~style:Paint.dim g ~x:ex ~y:ny ~w:room note
          | None -> ());
          (* The latest difference, and its verdict at its step, the latest both
             runs logged. *)
          Paint.right g
            ~x:(mx + w - l.value_w)
            ~y ~w:l.value_w
            [ (Paint.plain, Text.signed d.last) ];
          Paint.right g
            ~x:(mx + w - l.value_w)
            ~y:(y + 1) ~w:l.value_w
            [
              ((if worse then Paint.bad else Paint.dim), verdict);
              ( Paint.dim,
                (if verdict = "" then "at " else " at ") ^ Text.count d.step );
            ])
    v.marked

let hints =
  [
    ("↑↓", "metric");
    ("enter", "zoom");
    ("space", "mark");
    ("[ ]", "run");
    ("i", "details");
    ("s", "smooth");
    ("esc", "runs");
  ]

let paint v g ~width ~height =
  let mx = margin ~width in
  let w = width - (2 * mx) in
  Paint.spans g ~x:mx ~y:0 ~w (sentence v ~width);
  (match attention v with
  | [] -> ()
  | notes ->
      Paint.spans g ~x:mx ~y:1 ~w
        ((Paint.alert, "! ")
        :: Paint.join (List.map (fun n -> [ (Paint.bad, n) ]) notes)));
  Paint.hints g ~x:mx ~y:(height - 1) ~w hints;
  let l = layout v ~width ~height in
  match (history v, cursor v) with
  | None, _ ->
      Paint.text ~style:Paint.dim g ~x:mx ~y:l.title_y ~w "reading the run…"
  | Some _, None ->
      Paint.text ~style:Paint.dim g ~x:mx ~y:l.title_y ~w
        "no metric sample yet: the chart appears with the first"
  | Some _, Some key ->
      let p =
        Data.chart v.data (chart_request v ~marks:true (main_spec ~width l key))
      in
      let tx = if width >= 100 then chart_x ~width l + Chart.left else mx in
      Paint.spans g ~x:tx ~y:l.title_y ~w:(mx + w - tx) (title v key p);
      let s = settings v key p in
      if s <> "" then Paint.right g ~x:mx ~y:l.title_y ~w [ (Paint.accent, s) ];
      if l.legend then
        Paint.spans g ~x:tx ~y:(l.title_y + 1) ~w:(mx + w - tx) (legend v);
      (match p with
      | Some p -> (
          match
            Chart.draw ~light:v.light p g ~x:(chart_x ~width l) ~y:l.chart_y
          with
          | Some placed -> paint_versus v g l p placed key ~width
          | None -> ())
      | None ->
          Paint.text ~style:Paint.dim g ~x:(mx + l.name_w) ~y:(l.chart_y + 2) ~w
            "reading…");
      ignore
        (List.fold_left
           (fun prev (dy, k) ->
             let p, _ = Paint.key k in
             paint_strip v g l k ~width ~y:(l.strips_y + dy) ~cursor:key
               ~first:(p <> prev);
             p)
           "" l.strip_rows);
      (match l.hidden with
      | Some (a, b) ->
          Paint.right g
            ~x:(mx + w - l.value_w)
            ~y:(l.strips_y - 1) ~w:l.value_w
            [
              ( Paint.dim,
                Printf.sprintf "%d–%d of %d" (a + 1) (b + 1)
                  (List.length (metrics v)) );
            ]
      | None -> ());
      if l.sys_rows > 0 then paint_use v g l ~width

(* Zoom *)

let paint_zoom v g ~width ~height =
  let mx = margin ~width in
  let w = width - (2 * mx) in
  Paint.spans g ~x:mx ~y:0 ~w
    (Paint.word (word v v.focus)
    @ [ (Paint.plain, "  "); (Paint.bold, name v.focus) ]);
  Paint.right g ~x:mx ~y:0 ~w
    [ (Paint.plain, "esc"); (Paint.dim, " back to the run") ];
  Paint.hints g ~x:mx ~y:(height - 1) ~w
    [
      ("esc", "run");
      ("←→", "cursor");
      ("↑↓", "metric");
      ("+ -", "window");
      ("< >", "pan");
      ("0", "reset");
      ("L", "log y");
      ("x", "axis");
    ];
  match cursor v with
  | None -> Paint.text ~style:Paint.dim g ~x:mx ~y:2 ~w "no metric to zoom"
  | Some key ->
      let spec = zoom_spec v ~width ~height key in
      let p = Data.chart v.data (chart_request v ~marks:true spec) in
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
        match at with
        | Some x -> (
            match v.chart.axis with
            | Time -> "at " ^ Text.duration x
            | Step -> "at step " ^ Text.count (Float.to_int x)
            | Step_metric -> "at " ^ axis_name v key ^ " " ^ Text.value x)
        | None -> "at the cursor"
      in
      let t = title v key p in
      let s = settings v key p in
      let note =
        [ (Paint.dim, "cursor "); (Paint.plain, at_text) ]
        @ if s = "" then [] else [ Paint.sep; (Paint.accent, s) ]
      in
      Paint.spans g ~x:(mx + Chart.left) ~y:2 ~w:(w - Chart.left) t;
      if Chart.left + Paint.width t + 3 + Paint.width note <= w then
        Paint.right g ~x:mx ~y:2 ~w note
      else if s <> "" then Paint.right g ~x:mx ~y:1 ~w [ (Paint.accent, s) ];
      (match p with
      | Some p -> ignore (Chart.draw ?cursor:at ~light:v.light p g ~x:mx ~y:3)
      | None -> ());
      let ty = 3 + spec.height + 1 in
      let cols =
        [
          ("run", 18);
          (at_text, 20);
          ("latest", 12);
          ("best", 22);
          ("samples", 10);
          ("not finite", 12);
          ("orphans", 9);
        ]
      in
      let cols =
        if width >= 100 then cols
        else List.filteri (fun i _ -> i <= 1 || i = 3) cols
      in
      ignore
        (List.fold_left
           (fun x (h, cw) ->
             Paint.text ~style:Paint.dim g ~x ~y:ty ~w:(cw - 1) h;
             x + cw)
           (mx + Chart.left) cols);
      List.iteri
        (fun i (c, r) ->
          let y = ty + 1 + i in
          let st = List.assoc_opt key (stats_of v r) in
          let cell = function
            | "run" -> (
                match c with
                | None -> [ (Paint.plain, "━ "); (Paint.bold, name r) ]
                | Some c ->
                    [
                      (Paint.mark c, String.make 1 c ^ " ");
                      (Paint.plain, name r);
                    ])
            | "latest" -> (
                match st with
                | Some s -> [ (Paint.plain, Text.value s.stats.last.value) ]
                | None -> [ (Paint.dim, "—") ])
            | "best" -> (
                match
                  Option.bind st (fun s ->
                      Option.bind s.History.goal (fun g ->
                          Metric.best g s.stats))
                with
                | Some b ->
                    [
                      (Paint.plain, Text.value b.value);
                      (Paint.dim, " at " ^ Text.count b.step);
                    ]
                | None -> [ (Paint.dim, "—") ])
            | "samples" -> (
                match st with
                | Some s -> [ (Paint.plain, Text.count s.stats.count) ]
                | None -> [ (Paint.dim, "—") ])
            | "not finite" -> (
                match Option.bind st (fun s -> s.History.stats.non_finite) with
                | Some (n, _) -> [ (Paint.bad, Text.count n) ]
                | None -> [ (Paint.dim, "0") ])
            | "orphans" ->
                let n =
                  match Data.entry v.data (Run.id r) with
                  | Some e -> (
                      match History.orphans e.history key with
                      | o -> Metric.length o
                      | exception Invalid_argument _ -> 0)
                  | None -> 0
                in
                [ ((if n > 0 then Paint.plain else Paint.dim), Text.count n) ]
            | _ ->
                [
                  ( Paint.bold,
                    Option.fold ~none:"—" ~some:Text.value (value (Run.id r)) );
                ]
          in
          ignore
            (List.fold_left
               (fun x (h, cw) ->
                 Paint.spans g ~x ~y ~w:(cw - 1) (cell h);
                 x + cw)
               (mx + Chart.left) cols))
        ((None, v.focus) :: List.map (fun (c, r) -> (Some c, r)) v.marked)
