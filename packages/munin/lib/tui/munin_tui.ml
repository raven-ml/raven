(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Munin
module K = Matrix.Input.Keystroke

type screen = Runs | Run of string option

(* The screens of the stack. Help is drawn over whichever is on top. *)
type kind = Runs_k | Run_k | Zoom_k | Details_k

type model = {
  store : Store.t;
  clock : unit -> float;
  now : float;
  width : int;
  height : int;
  stack : kind list; (* never empty; its head is shown, its last is Runs *)
  help : bool;
  help_at : int; (* Help's first line shown *)
  data : Data.t;
  table : Runs_screen.table; (* Runs, as of the last change it shows *)
  reading : bool;
  again : bool; (* a key asked for a read while one ran *)
  waiting : bool; (* Run started without an id and no run chosen yet *)
  focus : string option;
  detail : string option; (* the run Details shows *)
  cursor : string option; (* Runs's cursor *)
  marks : (char * string) list;
  metric : string option;
  filter : string;
  metric_filter : string;
  typing : bool;
  order : Runs_screen.order;
  reverse : bool;
  chart : Chart.view;
  zoom_at : int option;
  scroll : int;
  light : bool;
  notice : string option; (* a key's refusal, until the next key *)
  error : string option; (* a run the last poll could not read *)
}

type msg =
  | Tick
  | Read of Data.result
  | Key of K.t
  | Resize of int * int
  | Scheme of [ `Dark | `Light ]

let top m = List.hd m.stack

(* Views of the model *)

let runs_view m =
  {
    Runs_screen.data = m.data;
    now = m.now;
    root = Store.root m.store;
    filter = m.filter;
    metric = m.metric;
    order = m.order;
    reverse = m.reverse;
    light = m.light;
  }

(* Runs's table is built when what it shows changes, and only while Runs is
   shown or about to be. *)
let tabulate m =
  {
    m with
    table =
      Runs_screen.table (runs_view m) ~width:m.width
        ~typing:(m.typing && top m = Runs_k);
  }

let row_ids m = Runs_screen.row_ids m.table

let full_entry m id =
  match Data.entry m.data id with
  | Some ({ keys = All; _ } as e) -> Some e
  | _ -> None

let run_view m id =
  Option.map
    (fun r ->
      {
        Run_screen.data = m.data;
        now = m.now;
        focus = r;
        entry = full_entry m id;
        marked =
          List.filter_map
            (fun (l, mid) ->
              if mid = id then None
              else Option.map (fun mr -> (l, mr)) (Data.run m.data mid))
            m.marks;
        metric = m.metric;
        chart = m.chart;
        zoom_at = m.zoom_at;
        filter = m.metric_filter;
        light = m.light;
      })
    (Data.run m.data id)

let focus_view m = Option.bind m.focus (run_view m)

(* Reading *)

(* What a screen of the stack needs read: each run with the keys it keeps. *)
let needs m kind =
  match kind with
  | Runs_k -> (
      List.filter_map
        (fun r ->
          if Run.ended_at r = None then Some (Run.id r, Data.Keys []) else None)
        (Data.listing m.data)
      @
      (* The cursor's run, unless its pane's chart holds the run's end. *)
      match (Option.bind m.cursor (Data.run m.data), m.metric) with
      | Some r, Some key -> (
          match Runs_screen.pane_request m.table ~height:m.height r with
          | Some c when not (Data.whole m.data c) ->
              [ (Run.id r, Data.Keys [ key ]) ]
          | _ -> [])
      | _ -> [])
  | Run_k | Zoom_k -> (
      match focus_view m with
      | Some v ->
          let shown = Run_screen.shown v in
          (Run.id v.focus, Data.All)
          :: List.map (fun (_, r) -> (Run.id r, Data.Keys shown)) v.marked
      | None -> Option.to_list (Option.map (fun id -> (id, Data.All)) m.focus))
  | Details_k -> Option.to_list (Option.map (fun id -> (id, Data.All)) m.detail)

let request m =
  let shown = top m in
  let follow = needs m shown in
  let keep =
    List.concat_map (fun k -> List.map fst (needs m k)) (List.tl m.stack)
  in
  let charts, trends, zoom =
    match (shown, focus_view m) with
    | (Run_k | Zoom_k), Some v ->
        Run_screen.requests v ~width:m.width ~height:m.height
          ~zoom:(shown = Zoom_k)
    | Runs_k, _ -> (
        match Option.bind m.cursor (Data.run m.data) with
        | Some r ->
            ( Option.to_list
                (Runs_screen.pane_request m.table ~height:m.height r),
              [],
              None )
        | None -> ([], [], None))
    | _ -> ([], [], None)
  in
  let list = shown = Runs_k || Data.listing m.data = [] || m.waiting in
  Data.request m.data ~time:m.now ~list ~follow ~keep ~charts ~trends ~zoom

let read m =
  let q = request m in
  let store = m.store in
  ( { m with reading = true; again = false },
    Mosaic.Cmd.perform (fun dispatch -> dispatch (Read (Data.read store q))) )

(* A key that changes what the screen shows reads at once, or after the read in
   progress. *)
let read_soon (m, cmd) =
  if m.reading then ({ m with again = true }, cmd)
  else
    let m, r = read m in
    (m, Mosaic.Cmd.batch [ cmd; r ])

(* The newest run without an end that has not crashed, else the newest. *)
let chosen m =
  let l = Data.listing m.data in
  match
    List.find_opt (fun r -> Run.ended_at r = None && Run.status r <> `Crashed) l
  with
  | Some r -> Some (Run.id r)
  | None -> Option.map Run.id (List.nth_opt l 0)

let settle m =
  let m =
    if m.waiting then
      match chosen m with
      | Some id -> { m with waiting = false; focus = Some id; cursor = Some id }
      | None -> m
    else m
  in
  let m =
    match m.metric with
    | Some _ -> m
    | None ->
        {
          m with
          metric =
            Runs_screen.default_metric m.data (Runs_screen.rows (runs_view m));
        }
  in
  let m = if top m = Runs_k then tabulate m else m in
  let ids = row_ids m in
  match m.cursor with
  | Some id when List.mem id ids -> m
  | _ ->
      { m with cursor = (match ids with id :: _ -> Some id | [] -> m.cursor) }

let char (k : K.t) =
  match k.key with K.Char u -> Some (Uchar.to_int u) | _ -> None

let is c k = char k = Some (Char.code c) && not k.mods.ctrl

let move_in l x d =
  match l with
  | [] -> None
  | _ ->
      let n = List.length l in
      let i =
        match x with
        | Some x -> Option.value (List.find_index (( = ) x) l) ~default:0
        | None -> 0
      in
      Some (List.nth l (max 0 (min (n - 1) (i + d))))

let cursor_move m d = { m with cursor = move_in (row_ids m) m.cursor d }

let metrics_of_focus m =
  match focus_view m with Some v -> Run_screen.metrics v | None -> []

let metric_move m d =
  match top m with
  | Runs_k ->
      {
        m with
        metric =
          move_in
            (Runs_screen.metrics m.data (Runs_screen.rows (runs_view m)))
            m.metric d;
      }
  | _ ->
      let keys = metrics_of_focus m in
      let cur = Option.bind (focus_view m) Run_screen.cursor in
      { m with metric = move_in keys cur d; zoom_at = None }

let focus_move m d =
  let ids = row_ids m in
  let id = match top m with Details_k -> m.detail | _ -> m.focus in
  match move_in ids id d with
  | Some id ->
      { m with focus = Some id; detail = Some id; zoom_at = None; scroll = 0 }
  | None -> m

let toggle_mark m id =
  if List.exists (fun (_, x) -> x = id) m.marks then
    {
      m with
      marks = List.filter (fun (_, x) -> x <> id) m.marks;
      notice = None;
    }
  else
    match
      List.find_opt (fun l -> not (List.mem_assoc l m.marks)) [ 'a'; 'b' ]
    with
    | Some l ->
        { m with marks = List.sort compare ((l, id) :: m.marks); notice = None }
    | None ->
        {
          m with
          notice = Some "2 runs marked: unmark one (space) to mark another";
        }

let zoom_move m d =
  match focus_view m with
  | Some v -> (
      match (v.entry, Run_screen.cursor v) with
      | Some e, Some key -> (
          match History.series e.history key with
          | s when Metric.length s > 0 ->
              let n = Metric.length s in
              let i = Option.value m.zoom_at ~default:(n - 1) in
              { m with zoom_at = Some (max 0 (min (n - 1) (i + d))) }
          | _ -> m
          | exception Invalid_argument _ -> m)
      | _ -> m)
  | None -> m

(* The x window narrows and widens with its right end at the latest x, and pans
   by half its span within the domain, read from the cursor's chart as last
   prepared. *)
let window m how =
  let prepared =
    Option.bind (focus_view m) (fun v ->
        let charts, _, _ =
          Run_screen.requests v ~width:m.width ~height:m.height
            ~zoom:(top m = Zoom_k)
        in
        let key = Run_screen.cursor v in
        List.find_map
          (fun (c : Data.chart) ->
            if Some c.spec.key = key then Data.chart m.data c else None)
          charts)
  in
  match prepared with
  | None -> m
  | Some p ->
      let lo, hi = (Chart.lo p, Chart.hi p) in
      let dlo, dhi =
        match m.chart.window with
        | None -> (lo, hi)
        | Some _ ->
            (0., Float.max hi (Option.value (Chart.latest p) ~default:hi))
      in
      let span = hi -. lo in
      let latest = Option.value (Chart.latest p) ~default:dhi in
      let at_latest span =
        if span >= dhi -. dlo then None
        else
          let lo = Float.max dlo (latest -. span) in
          Some (lo, lo +. span)
      in
      let pan d =
        let lo =
          Float.max dlo (Float.min (dhi -. span) (lo +. (d *. span /. 2.)))
        in
        Some (lo, lo +. span)
      in
      let window =
        match how with
        | `Narrow -> at_latest (span /. 2.)
        | `Widen -> at_latest (span *. 2.)
        | `Reset -> None
        | `Left -> if m.chart.window = None then None else pan (-1.)
        | `Right -> if m.chart.window = None then None else pan 1.
      in
      { m with chart = { m.chart with window } }

let open_screen m kind = { m with stack = kind :: m.stack; help = false }

let back m =
  match m.stack with
  | [ _ ] -> if m.filter <> "" then { m with filter = "" } else m
  | _ :: rest ->
      tabulate
        { m with stack = rest; zoom_at = None; scroll = 0; metric_filter = "" }
  | [] -> m

let typing_key m (k : K.t) =
  let set f =
    match top m with
    | Runs_k -> { m with filter = f }
    | _ -> { m with metric_filter = f }
  in
  let text = match top m with Runs_k -> m.filter | _ -> m.metric_filter in
  match k.key with
  | K.Enter -> { m with typing = false }
  | K.Escape -> { (set "") with typing = false }
  | K.Backspace ->
      let n = String.length text in
      if n = 0 then m
      else
        (* Drop the last UTF-8 sequence. *)
        let i = ref (n - 1) in
        while !i > 0 && Char.code text.[!i] land 0xC0 = 0x80 do
          decr i
        done;
        set (String.sub text 0 !i)
  | _ when k.text <> "" && not k.mods.ctrl -> set (text ^ k.text)
  | _ -> m

let key m (k : K.t) =
  if m.typing then (settle (typing_key m k), Mosaic.Cmd.none)
  else if m.help then
    if is '?' k || k.key = K.Escape then
      ({ m with help = false; help_at = 0 }, Mosaic.Cmd.none)
    else if is 'q' k then (m, Mosaic.Cmd.quit)
    else if k.key = K.Down || is 'j' k then
      ({ m with help_at = m.help_at + 1 }, Mosaic.Cmd.none)
    else if k.key = K.Up || is 'k' k then
      ({ m with help_at = max 0 (m.help_at - 1) }, Mosaic.Cmd.none)
    else (m, Mosaic.Cmd.none)
  else
    let none m = (m, Mosaic.Cmd.none) in
    let kind = top m in
    let up = k.key = K.Up || is 'k' k and down = k.key = K.Down || is 'j' k in
    let left = k.key = K.Left || is 'h' k
    and right = k.key = K.Right || is 'l' k in
    let charts = kind = Run_k || kind = Zoom_k in
    if is 'q' k then (m, Mosaic.Cmd.quit)
    else if is '?' k then none { m with help = true }
    else if k.key = K.Escape then none (back m)
    else if is '/' k && (kind = Runs_k || kind = Run_k) then
      none { m with typing = true }
    else if up || down then
      let d = if up then -1 else 1 in
      none
        (match kind with
        | Runs_k -> cursor_move m d
        | Run_k | Zoom_k -> metric_move m d
        | Details_k -> { m with scroll = max 0 (m.scroll + d) })
    else if left || right then
      let d = if left then -1 else 1 in
      none
        (match kind with
        | Runs_k -> metric_move m d
        | Run_k -> m
        | Zoom_k -> zoom_move m d
        | Details_k -> m)
    else if is 'g' k || is 'G' k then
      let far = if is 'g' k then -max_int / 2 else max_int / 2 in
      none
        (match kind with
        | Runs_k -> cursor_move m far
        | Run_k -> metric_move m far
        | Zoom_k -> zoom_move m far
        | Details_k -> { m with scroll = (if far < 0 then 0 else max_int / 2) })
    else if k.key = K.Enter then
      none
        (match kind with
        | Runs_k -> (
            match m.cursor with
            | Some id ->
                open_screen
                  {
                    m with
                    focus = Some id;
                    chart = { m.chart with window = None };
                  }
                  Run_k
            | None -> m)
        | Run_k -> open_screen { m with zoom_at = None } Zoom_k
        | _ -> m)
    else if is ' ' k then
      none
        (match kind with
        | Runs_k -> Option.fold ~none:m ~some:(toggle_mark m) m.cursor
        | _ -> Option.fold ~none:m ~some:(toggle_mark m) m.focus)
    else if (is '[' k || is ']' k) && kind <> Runs_k then
      none (focus_move m (if is '[' k then -1 else 1))
    else if (is 'o' k || is 'O' k) && kind = Runs_k then
      if is 'O' k then none { m with reverse = not m.reverse }
      else
        let order : Runs_screen.order =
          match m.order with
          | Started -> Value
          | Value -> State
          | State -> Name
          | Name -> Started
        in
        none { m with order; reverse = false }
    else if is 's' k && charts then
      none
        { m with chart = { m.chart with smooth = (m.chart.smooth + 1) mod 4 } }
    else if is 'x' k && charts then
      let axis : Chart.axis =
        match m.chart.axis with
        | Step -> Step_metric
        | Step_metric -> Time
        | Time -> Step
      in
      none
        { m with chart = { m.chart with axis; window = None }; zoom_at = None }
    else if is 'L' k && charts then
      none { m with chart = { m.chart with log = not m.chart.log } }
    else if is '+' k && charts then none (window m `Narrow)
    else if is '-' k && charts then none (window m `Widen)
    else if is '0' k && charts then none (window m `Reset)
    else if is '<' k && charts then none (window m `Left)
    else if is '>' k && charts then none (window m `Right)
    else if is 'i' k && kind <> Details_k then
      let id = match kind with Runs_k -> m.cursor | _ -> m.focus in
      none
        (match id with
        | Some _ -> open_screen { m with detail = id; scroll = 0 } Details_k
        | None -> m)
    else if is 'y' k then
      let id =
        match kind with
        | Runs_k -> m.cursor
        | Details_k -> m.detail
        | _ -> m.focus
      in
      match id with
      | Some id -> (m, Mosaic.Cmd.copy_to_clipboard id)
      | None -> none m
    else none m

(* The program *)

let init ~clock store screen () =
  let now = clock () in
  let stack, focus, waiting =
    match screen with
    | Runs -> ([ Runs_k ], None, false)
    | Run (Some id) -> ([ Run_k; Runs_k ], Some id, false)
    | Run None -> ([ Run_k; Runs_k ], None, true)
  in
  let m =
    {
      store;
      clock;
      now;
      width = 80;
      height = 24;
      stack;
      help = false;
      help_at = 0;
      data = Data.empty;
      table =
        Runs_screen.table
          {
            Runs_screen.data = Data.empty;
            now;
            root = Store.root store;
            filter = "";
            metric = None;
            order = Started;
            reverse = false;
            light = false;
          }
          ~width:80 ~typing:false;
      reading = false;
      again = false;
      waiting;
      focus;
      detail = None;
      cursor = focus;
      marks = [];
      metric = None;
      filter = "";
      metric_filter = "";
      typing = false;
      order = Started;
      reverse = false;
      chart = { axis = Step; smooth = 0; log = false; window = None };
      zoom_at = None;
      scroll = 0;
      light = false;
      notice = None;
      error = None;
    }
  in
  let m, read = read m in
  (m, Mosaic.Cmd.batch [ read; Mosaic.Cmd.query_color_scheme ])

let update msg m =
  match msg with
  | Tick ->
      let m = { m with now = m.clock () } in
      if m.reading then (m, Mosaic.Cmd.none) else read m
  | Read r ->
      (* A run that could not be read is named on the line above the keys until
         a poll reads every run. *)
      let error =
        match Data.errors r with
        | [] -> None
        | ("", why) :: _ -> Some ("cannot list the store: " ^ why)
        | (id, why) :: _ -> Some (Printf.sprintf "cannot read %s: %s" id why)
      in
      let m =
        settle { m with data = Data.apply m.data r; reading = false; error }
      in
      if m.again then read m else (m, Mosaic.Cmd.none)
  | Key k ->
      let m', cmd = key { m with notice = None } k in
      (* The table changes with what it shows, never with the cursor. *)
      let shows m = (m.filter, m.order, m.reverse, m.metric, m.typing, top m) in
      let m' =
        if top m' = Runs_k && shows m' <> shows m then tabulate m' else m'
      in
      read_soon (m', cmd)
  | Resize (width, height) ->
      read_soon (settle { m with width; height }, Mosaic.Cmd.none)
  | Scheme s ->
      let m = { m with light = s = `Light } in
      ((if top m = Runs_k then tabulate m else m), Mosaic.Cmd.none)

let subscriptions _ =
  Mosaic.Sub.batch
    [
      Mosaic.Sub.every ~id:"poll" 1. (fun _ -> Tick);
      Mosaic.Sub.on_key (fun e -> Some (Key (Mosaic.Event.Key.data e)));
      Mosaic.Sub.on_resize (fun ~width ~height -> Resize (width, height));
      Mosaic.Sub.on_color_scheme (fun s -> Some (Scheme s));
    ]

(* Painting *)

(* A terminal too small: the gist, centred, and the size needed. Lines leave
   from the end when it is shorter still. *)
let small m g =
  let gist =
    match (top m, focus_view m) with
    | (Run_k | Zoom_k | Details_k), Some v -> (
        let r = v.focus in
        let step = Run_screen.latest_step v in
        [
          Paint.word (Data.word m.data ~now:m.now r)
          @ [ (Paint.plain, "  "); (Paint.bold, Runs_screen.name r) ];
        ]
        @ (match (step, Run.steps r) with
          | Some s, Some t when t > 0 && s < t ->
              [
                [
                  (Paint.plain, Text.count s);
                  (Paint.dim, " of " ^ Text.count t ^ " · ");
                  (Paint.plain, Text.percent_of s t);
                ];
              ]
          | Some s, _ ->
              [ [ (Paint.dim, "step "); (Paint.plain, Text.count s) ] ]
          | None, _ -> [])
        @
        match
          Option.bind (Run_screen.cursor v) (fun k ->
              Option.map
                (fun s -> (k, s))
                (List.assoc_opt k
                   (Option.value (Data.stats m.data r) ~default:[])))
        with
        | Some (k, (s : History.key_stats)) ->
            [
              [
                (Paint.dim, k ^ " "); (Paint.bold, Text.value s.stats.last.value);
              ];
            ]
        | None -> [])
    | _ ->
        let all = Data.listing m.data in
        let count w =
          List.length
            (List.filter
               (fun r -> List.mem (Data.word m.data ~now:m.now r) w)
               all)
        in
        let live = count [ "live"; "running"; "writing"; "starting" ]
        and broken = count [ "failed"; "crashed" ] in
        [ [ (Paint.bold, Printf.sprintf "%d runs" (List.length all)) ] ]
        @ (if live > 0 then
             [
               [
                 (Paint.ok, "● "); (Paint.plain, Printf.sprintf "%d live" live);
               ];
             ]
           else [])
        @
        if broken > 0 then
          [
            [
              (Paint.bad, "✗ ");
              (Paint.plain, Printf.sprintf "%d failed or crashed" broken);
            ];
          ]
        else []
  in
  let need =
    [ (Paint.dim, Printf.sprintf "needs 40×10, has %d×%d" m.width m.height) ]
  in
  let gist = List.filteri (fun i _ -> i < m.height - 2) gist in
  let lines =
    gist @ if m.height > List.length gist + 1 then [ []; need ] else [ need ]
  in
  let top_y = max 0 ((m.height - List.length lines) / 2) in
  List.iteri (fun i l -> Paint.centre g ~y:(top_y + i) ~w:m.width l) lines

let paint m g =
  let width = m.width and height = m.height in
  if width < 40 || height < 10 then small m g
  else begin
    let mx = if width >= 100 then 2 else 1 in
    (match top m with
    | Runs_k ->
        Runs_screen.paint m.table g ~height ~cursor:m.cursor ~marks:m.marks
    | Run_k -> (
        match focus_view m with
        | Some v -> Run_screen.paint v g ~width ~height
        | None ->
            Paint.text ~style:Paint.dim g ~x:mx ~y:0
              ~w:(width - (2 * mx))
              (if m.waiting then
                 "waiting for a run to start in "
                 ^ Runs_screen.tilde (Store.root m.store)
               else "reading the run…"))
    | Zoom_k ->
        Option.iter
          (fun v -> Run_screen.paint_zoom v g ~width ~height)
          (focus_view m)
    | Details_k -> (
        match Option.bind m.detail (Data.run m.data) with
        | Some r ->
            let marked =
              List.filter_map
                (fun (l, id) ->
                  if Some id = m.detail then None
                  else Option.map (fun r -> (l, r)) (Data.run m.data id))
                m.marks
            in
            Details.paint
              ~word:(Data.word m.data ~now:m.now r)
              ~scroll:m.scroll r
              ~stats:(Option.value (Data.stats m.data r) ~default:[])
              (Option.map
                 (fun (e : Data.entry) -> e.history)
                 (Option.bind m.detail (full_entry m)))
              ~marked g ~width ~height
        | None -> ()));
    (match (m.typing, top m) with
    | true, Run_k ->
        let y = height - 1 in
        Paint.fill g ~x:0 ~y ~w:width Paint.plain;
        Paint.spans g ~x:mx ~y
          ~w:(width - (2 * mx))
          [
            (Paint.accent, "/ ");
            (Paint.plain, m.metric_filter);
            (Paint.reverse, " ");
          ]
    | _ -> ());
    (* A key's refusal replaces the key line until the next key; a run that
       could not be read is named above it until a poll reads every run. *)
    (match m.notice with
    | Some n ->
        Paint.fill g ~x:0 ~y:(height - 1) ~w:width Paint.plain;
        Paint.text ~style:Paint.alert g ~x:mx ~y:(height - 1)
          ~w:(width - (2 * mx))
          n
    | None -> ());
    Option.iter
      (fun e ->
        Paint.fill g ~x:0 ~y:(height - 2) ~w:width Paint.plain;
        Paint.text ~style:Paint.alert g ~x:mx ~y:(height - 2)
          ~w:(width - (2 * mx))
          e)
      m.error;
    if m.help then
      Help.paint
        ~screen:
          (match top m with
          | Runs_k -> `Runs
          | Run_k -> `Run
          | Zoom_k -> `Zoom
          | Details_k -> `Details)
        ~light:m.light ~scroll:m.help_at g ~width ~height
  end

let view m =
  Mosaic.canvas
    ~size:{ width = Mosaic.pct 100; height = Mosaic.pct 100 }
    (fun c ~dt:_ ->
      let g = Mosaic.Canvas.grid c in
      Matrix_grid.clear g;
      paint m g)

let app ~clock store screen =
  { Mosaic.init = init ~clock store screen; update; view; subscriptions }

let run store screen =
  match screen with
  | Run (Some id) when Store.run store id = None ->
      Error (Printf.sprintf "no run %s in %s" id (Store.root store))
  | _ when not (Unix.isatty Unix.stdin && Unix.isatty Unix.stdout) ->
      Error "the dashboard needs a terminal"
  | _ ->
      let config = Matrix.config ~mouse:`Off ~fps:10. () in
      Mosaic.run ~config (app ~clock:Unix.gettimeofday store screen);
      Ok ()

(* For the tests *)

module Curve = Curve

module Chart = struct
  let draw ~width ~height ~lo ~hi ~key focus marked =
    let g = Matrix_grid.create ~width ~height () in
    let view =
      { Chart.axis = Step; smooth = 0; log = false; window = Some (lo, hi) }
    in
    let spec = { Chart.key; width; height; x_axis = true } in
    ignore
      (Chart.draw ~light:false
         (Chart.prepare view spec focus marked)
         g ~x:0 ~y:0);
    g

  let paired h key = Chart.x_of Step_metric h key (History.series h key)
  let domain h key = Chart.domain Step ~key [ h ]
end

module Watch = Watch

let set_depth = function
  | `Mono -> Paint.set_depth Mono
  | `Ansi16 -> Paint.set_depth Ansi16
  | `Ansi256 -> Paint.set_depth Ansi256
  | `Truecolor -> Paint.set_depth Truecolor

let key k = Key k
let resize ~width ~height = Resize (width, height)

let frame m =
  let g = Matrix_grid.create ~width:m.width ~height:m.height () in
  paint m g;
  g
