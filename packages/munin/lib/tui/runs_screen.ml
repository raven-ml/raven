(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Munin

type order = Started | Value | State | Name

type view = {
  data : Data.t;
  now : float;
  root : string;
  filter : string;
  metric : string option;
  order : order;
  reverse : bool;
  light : bool;
}

(* Facts of a row *)

let name r = Option.value (Run.name r) ~default:(Run.id r)

let value_text : Value.t -> string = function
  | `Float x -> Text.given x
  | `Int n -> string_of_int n
  | `String s -> s
  | `Bool b -> string_of_bool b

let host r = (Run.provenance r).hostname

let metric_stats d r key =
  match Data.stats d r with
  | Some l -> (
      match List.assoc_opt key l with
      | Some ({ kind = `Metric; _ } as k) -> Some k
      | _ -> None)
  | None -> None

(* The value of [key] in a row: its best by goal, else its latest, and whether
   it is a best. *)
let value d r key =
  Option.map
    (fun (k : History.key_stats) ->
      match Option.bind k.goal (fun g -> Metric.best g k.stats) with
      | Some s -> (s, true)
      | None -> (k.stats.last, false))
    (metric_stats d r key)

let goal d rows key =
  List.find_map
    (fun r ->
      Option.bind (metric_stats d r key) (fun (k : History.key_stats) -> k.goal))
    rows

let latest_step d r = Option.bind (Data.stats d r) Fact.latest_step

(* The first sample of a metric of [r] that is not finite, by step. *)
let first_non_finite d r =
  match Data.stats d r with
  | None -> None
  | Some l ->
      List.fold_left
        (fun acc (k, (s : History.key_stats)) ->
          match (s.kind, s.stats.non_finite, acc) with
          | `Metric, Some (_, f), Some (_, (b : Metric.sample))
            when f.step >= b.step ->
              acc
          | `Metric, Some (_, f), _ -> Some (k, f)
          | _ -> acc)
        None l

let has_non_finite d r = first_non_finite d r <> None
let live w = List.mem w [ "live"; "running"; "writing"; "starting" ]
let broken w = w = "failed" || w = "crashed"

(* Metrics *)

let metric_keys d r =
  match Data.stats d r with
  | Some l ->
      List.filter_map
        (fun (k, (s : History.key_stats)) ->
          if s.kind = `Metric then Some k else None)
        l
  | None -> []

let metrics d rows =
  List.sort_uniq String.compare (List.concat_map (metric_keys d) rows)

let default_metric d rows =
  let counts = Hashtbl.create 16 in
  List.iter
    (fun r ->
      match Data.stats d r with
      | Some l ->
          List.iter
            (fun (k, (s : History.key_stats)) ->
              if s.kind = `Metric && s.goal <> None then
                Hashtbl.replace counts k
                  (1 + Option.value (Hashtbl.find_opt counts k) ~default:0))
            l
      | None -> ())
    rows;
  Hashtbl.fold
    (fun k n best ->
      match best with
      | Some (bk, bn) when bn > n || (bn = n && bk < k) -> best
      | _ -> Some (k, n))
    counts None
  |> Option.map fst

(* Rows *)

(* [contains ~sub s] is [true] iff [sub], in lowercase, is part of [s], ignoring
   case. *)
let contains ~sub s =
  let n = String.length sub and m = String.length s in
  let at i =
    let rec go j =
      j = n || (Char.lowercase_ascii s.[i + j] = sub.[j] && go (j + 1))
    in
    go 0
  in
  let rec from i = i + n <= m && (at i || from (i + 1)) in
  from 0

let words s =
  List.filter (( <> ) "") (String.split_on_char ' ' (String.lowercase_ascii s))

let matches v r w =
  contains ~sub:w (name r)
  || contains ~sub:w (Run.id r)
  || contains ~sub:w (Run.experiment r)
  || contains ~sub:w (Option.value (Run.parent r) ~default:"")
  || contains ~sub:w (Data.word v.data ~now:v.now r)
  || List.exists (contains ~sub:w) (Run.tags r)
  || List.exists
       (fun (k, x) -> contains ~sub:w (k ^ "=" ^ value_text x))
       (Run.params r)

(* The experiments of [rows], in the order of their first row. *)
let experiments rows =
  List.rev
    (List.fold_left
       (fun acc r ->
         if List.mem (Run.experiment r) acc then acc
         else Run.experiment r :: acc)
       [] rows)

let rows v =
  let ws = words v.filter in
  let shown =
    List.filter (fun r -> List.for_all (matches v r) ws) (Data.listing v.data)
  in
  let cmp =
    match v.order with
    | Started -> fun _ _ -> 0
    | Name -> fun a b -> String.compare (name a) (name b)
    | State ->
        fun a b ->
          String.compare
            (Data.word v.data ~now:v.now a)
            (Data.word v.data ~now:v.now b)
    | Value -> (
        match v.metric with
        | None -> fun _ _ -> 0
        | Some key -> (
            let sign =
              match goal v.data shown key with Some `Maximize -> -1 | _ -> 1
            in
            let get r =
              match value v.data r key with
              | Some (s, _) when Float.is_finite s.value -> Some s.value
              | _ -> None
            in
            fun a b ->
              match (get a, get b) with
              | Some x, Some y -> sign * Float.compare x y
              | Some _, None -> -1
              | None, Some _ -> 1
              | None, None -> 0))
  in
  let sorted = List.stable_sort cmp shown in
  let sorted = if v.reverse then List.rev sorted else sorted in
  (* Grouped by experiment, the groups in the order of their first row, so that
     each experiment's rows sit under its header row. *)
  List.concat_map
    (fun e -> List.filter (fun r -> Run.experiment r = e) sorted)
    (experiments sorted)

(* The best row of each experiment among [rows], by the metric's goal. *)
let stars d rows key =
  let best = Hashtbl.create 4 in
  List.iter
    (fun r ->
      match
        (value d r key, Option.bind (metric_stats d r key) (fun k -> k.goal))
      with
      | Some (s, true), Some g -> (
          let e = Run.experiment r in
          match Hashtbl.find_opt best e with
          | Some (_, b)
            when (g = `Minimize && b <= s.Metric.value)
                 || (g = `Maximize && b >= s.value) ->
              ()
          | _ -> Hashtbl.replace best e (Run.id r, s.value))
      | _ -> ())
    rows;
  Hashtbl.fold (fun _ (id, _) l -> id :: l) best []

(* Columns *)

type column = {
  id : string; (* what an order names *)
  head : string;
  cells : Paint.spans array; (* one per row *)
  right : bool;
}

let differs f rows =
  match List.sort_uniq compare (List.map f rows) with
  | [] | [ _ ] -> false
  | _ -> true

let columns v rows =
  let d = v.data in
  let column ?(right = false) id head cell =
    { id; head; cells = Array.of_list (List.map cell rows); right }
  in
  let when_differs f c = if differs f rows then [ Lazy.force c ] else [] in
  let state =
    column "state" "state" (fun r -> Paint.word (Data.word d ~now:v.now r))
  in
  let run = column "run" "run" (fun r -> [ (Paint.plain, name r) ]) in
  let parent =
    when_differs Run.parent
      (lazy
        (column "parent" "parent" (fun r ->
             [ (Paint.dim, Option.value (Run.parent r) ~default:"—") ])))
  in
  let started =
    column ~right:true "started" "started" (fun r ->
        [ (Paint.dim, Text.clock ~now:v.now (Run.started_at r)) ])
  in
  let ran =
    column ~right:true "ran" "ran" (fun r ->
        [ (Paint.dim, Text.duration (Fact.wall r)) ])
  in
  let step =
    column ~right:true "step" "step" (fun r ->
        match (latest_step d r, Run.ended_at r, Run.steps r) with
        | None, _, _ -> []
        | Some s, None, Some t when t > 0 && s < t ->
            [
              (Paint.plain, Text.count s);
              ( Paint.dim,
                Printf.sprintf " %3.0f%%"
                  (100. *. Float.of_int s /. Float.of_int t) );
            ]
        | Some s, _, _ -> [ (Paint.plain, Text.count s) ])
  in
  let metric =
    match v.metric with
    | None -> []
    | Some key ->
        let starred = stars d rows key in
        [
          column ~right:true "value"
            (key
            ^ match Fact.arrow (goal d rows key) with "" -> "" | a -> " " ^ a)
            (fun r ->
              match value d r key with
              | None -> [ (Paint.dim, "—") ]
              | Some (s, _) ->
                  [
                    ( (if Float.is_finite s.value then Paint.plain else Paint.bad),
                      Text.value s.value );
                    ( Paint.accent,
                      if List.mem (Run.id r) starred then "*" else " " );
                  ]);
        ]
  in
  let nonfinite =
    if List.exists (has_non_finite d) rows then
      [
        column "nonfinite" "not finite" (fun r ->
            match first_non_finite d r with
            | Some (_, f) ->
                [
                  ( Paint.bad,
                    Printf.sprintf "× %s at %s" (Text.non_finite f.value)
                      (Text.count f.step) );
                ]
            | None -> []);
      ]
    else []
  in
  (* Params whose values differ among the rows, one meaning per column, [—]
     where a run has none. One that differs within an experiment tells its runs
     apart; one that differs only between experiments repeats the header rows.
     Within, the param with the most values first: a sweep's. *)
  let params =
    let keys =
      List.sort_uniq compare
        (List.concat_map (fun r -> List.map fst (Run.params r)) rows)
    in
    let v k r = Option.map value_text (List.assoc_opt k (Run.params r)) in
    let exps = experiments rows in
    let within k =
      List.exists
        (fun e ->
          differs (v k) (List.filter (fun r -> Run.experiment r = e) rows))
        exps
    in
    let distinct k =
      List.length (List.sort_uniq compare (List.map (v k) rows))
    in
    let col k =
      column ~right:true ("param " ^ k) k (fun r ->
          match v k r with
          | Some s -> [ (Paint.plain, s) ]
          | None -> [ (Paint.dim, "—") ])
    in
    let differing = List.filter (fun k -> differs (v k) rows) keys in
    let inner =
      List.stable_sort
        (fun a b -> compare (distinct b) (distinct a))
        (List.filter within differing)
    in
    let outer = List.filter (fun k -> not (within k)) differing in
    (List.map col inner, List.map col outer)
  in
  let inner, outer = params in
  let host_col =
    when_differs host
      (lazy
        (column "host" "host" (fun r ->
             [ (Paint.dim, Option.value (host r) ~default:"—") ])))
  in
  let commit =
    when_differs
      (fun r -> Fact.commit (Run.provenance r))
      (lazy
        (column "commit" "commit" (fun r ->
             [
               ( Paint.dim,
                 Option.value (Fact.commit (Run.provenance r)) ~default:"—" );
             ])))
  in
  let all =
    [ state; run ] @ parent @ [ started; ran; step ] @ metric @ nonfinite
    @ inner @ outer @ host_col @ commit
  in
  (* Columns that do not fit leave whole, quietest first: the commit, params
     that repeat the header rows, the host, the time a run ran, its start, the
     params past the first, the parent. The state, name, step, value and
     non-finite flag stay. *)
  let drops =
    commit @ List.rev outer @ host_col @ [ ran; started ]
    @ List.rev (match inner with _ :: rest -> rest | [] -> [])
    @ parent
  in
  (all, drops)

(* Tables *)

type line = Group of Paint.spans | Row of int

type table = {
  view : view;
  width : int;
  rows : Run.t array;
  lines : line array;
  head : Paint.spans;
  cells : Paint.spans array; (* each row's line, without the gutter *)
  sentence : Paint.spans;
  typing : bool;
  run_x : int; (* the name's column in a row's line *)
}

let gutter = 3
let margin width = if width >= 100 then 2 else 1

let tilde root =
  match Sys.getenv_opt "HOME" with
  | Some h
    when h <> ""
         && String.length root >= String.length h
         && String.sub root 0 (String.length h) = h ->
      "~"
      ^ String.sub root (String.length h) (String.length root - String.length h)
  | _ -> root

(* The first line: the runs shown, live, failed and crashed, not finite, each
   counting the rows that show its glyph, and what every row shares. *)
let sentence v rows ~width =
  let d = v.data in
  let words = List.map (fun r -> (r, Data.word d ~now:v.now r)) rows in
  let count f = List.length (List.filter f words) in
  let total = List.length (Data.listing d) in
  let n = List.length rows in
  let head =
    if n = total then Printf.sprintf "%d run%s" n (if n = 1 then "" else "s")
    else Printf.sprintf "%d of %d runs" n total
  in
  let part glyph style k label =
    if k = 0 then []
    else
      [
        [ (style, glyph ^ " "); (Paint.plain, Printf.sprintf "%d %s" k label) ];
      ]
  in
  let failed = count (fun (_, w) -> w = "failed")
  and crashed = count (fun (_, w) -> w = "crashed") in
  let nan = count (fun (r, _) -> has_non_finite d r) in
  let parts =
    part "●" Paint.ok (count (fun (_, w) -> live w)) "live"
    @ part "✗" Paint.bad failed "failed"
    @ part "✗" Paint.bad crashed "crashed"
    @ part "×" Paint.bad nan "not finite"
  in
  let quiet =
    if failed + crashed + nan = 0 && rows <> [] then
      [ [ (Paint.dim, "nothing needs attention") ] ]
    else []
  in
  let shared f label =
    match rows with
    | r :: _ when not (differs f rows) ->
        Option.to_list (Option.map (fun x -> [ (Paint.dim, label x) ]) (f r))
    | _ -> []
  in
  let experiment =
    if v.filter <> "" then shared (fun r -> Some (Run.experiment r)) Fun.id
    else []
  in
  (* What every row shares leaves first when the line is too long. *)
  let n = List.length parts + List.length quiet in
  let tail = experiment @ shared host (fun h -> "on " ^ h) in
  Paint.fit
    ~drop:(List.rev (List.init (List.length tail) (fun i -> 1 + n + i)))
    ~w:(width - 4) []
    (([ (Paint.bold, head) ] :: parts) @ quiet @ tail)

let group_header v exp rows =
  let d = v.data in
  let words = List.map (Data.word d ~now:v.now) rows in
  let live = List.length (List.filter live words)
  and broken = List.length (List.filter broken words) in
  let n = List.length rows in
  [
    (Paint.bold, exp);
    (Paint.dim, Printf.sprintf "  %d run%s" n (if n = 1 then "" else "s"));
  ]
  @ (if live > 0 then
       [
         Paint.sep; (Paint.ok, "● "); (Paint.dim, Printf.sprintf "%d live" live);
       ]
     else [])
  @
  if broken > 0 then
    [
      Paint.sep;
      (Paint.bad, "✗ ");
      (Paint.dim, Printf.sprintf "%d failed or crashed" broken);
    ]
  else []

let table v ~width ~typing =
  let rows = rows v in
  let all, drops = columns v rows in
  let ordered =
    match v.order with
    | Started -> "started"
    | Value -> "value"
    | State -> "state"
    | Name -> "run"
  in
  let head_text (c : column) =
    c.head ^ if c.id = ordered then if v.reverse then " ▴" else " ▾" else ""
  in
  let width_of (c : column) =
    Array.fold_left
      (fun m s -> max m (Paint.width s))
      (Text.width (head_text c))
      c.cells
  in
  let room = width - (2 * margin width) - gutter in
  let total cols = List.fold_left (fun a c -> a + width_of c + 2) (-2) cols in
  let rec fit cols = function
    | _ when total cols <= room -> cols
    | d :: rest -> fit (List.filter (fun c -> c != d) cols) rest
    | [] -> cols
  in
  let cols = List.map (fun c -> (c, width_of c)) (fit all drops) in
  let line cell =
    let _, spans =
      List.fold_left
        (fun (first, acc) ((c : column), w) ->
          let s = cell c in
          let n = Paint.width s in
          let pad k =
            if k > 0 then [ (Paint.plain, String.make k ' ') ] else []
          in
          let s = if c.right then pad (w - n) @ s else s @ pad (w - n) in
          (false, acc @ (if first then [] else [ (Paint.plain, "  ") ]) @ s))
        (true, []) cols
    in
    spans
  in
  let run_x =
    let rec go x = function
      | [] -> 0
      | ((c : column), w) :: rest ->
          if c.id = "run" then x else go (x + w + 2) rest
    in
    go 0 cols
  in
  let rows_a = Array.of_list rows in
  let index = Hashtbl.create 64 in
  Array.iteri (fun i r -> Hashtbl.replace index (Run.id r) i) rows_a;
  let lines =
    List.concat_map
      (fun e ->
        let mine = List.filter (fun r -> Run.experiment r = e) rows in
        Group (group_header v e mine)
        :: List.map (fun r -> Row (Hashtbl.find index (Run.id r))) mine)
      (experiments rows)
  in
  {
    view = v;
    width;
    rows = rows_a;
    lines = Array.of_list lines;
    head =
      List.map
        (fun (s, t) -> (Paint.with_ s Paint.dim, t))
        (line (fun c -> [ (Paint.plain, head_text c) ]));
    cells = Array.mapi (fun i _ -> line (fun c -> c.cells.(i))) rows_a;
    sentence = sentence v rows ~width:(width - (2 * margin width));
    typing;
    run_x;
  }

let row_ids t = Array.to_list (Array.map Run.id t.rows)

(* The pane: the cursor run's chart of the focus metric in the rows the table
   leaves, once they are [pane_rows] or more, and at [wide] columns a block of
   its metrics beside the chart. *)
let pane_rows = 7
let wide = 140
let block_w = 46
let pane_top t = 3 + Array.length t.lines + 1
let pane_height t ~height = height - 2 - pane_top t
let head_rows t = if t.width >= 100 then 2 else 1
let block t = if t.width >= wide then block_w else 0

let pane_chart t ~height =
  if Array.length t.rows > 0 && pane_height t ~height >= pane_rows then
    Option.map
      (fun key ->
        let inner = t.width - (2 * margin t.width) in
        {
          Chart.key;
          width = (inner - if block t > 0 then block t + 3 else 0);
          height = pane_height t ~height - head_rows t;
          x_axis = true;
        })
      t.view.metric
  else None

let pane_view = { Chart.axis = Step; smooth = 0; log = false; window = None }

(* The pane's chart of an ended run is kept when the cursor leaves it, so that
   coming back reads nothing. *)
let pane_request t ~height r =
  Option.map
    (fun spec ->
      {
        Data.focus = Run.id r;
        marked = [];
        view = pane_view;
        spec;
        kept = true;
      })
    (pane_chart t ~height)

let metric_line v r key =
  let st = metric_stats v.data r key in
  Paint.key_spans key
  @ [
      ( Paint.dim,
        match Fact.arrow (Option.bind st (fun s -> s.goal)) with
        | "" -> ""
        | a -> " " ^ a );
      (Paint.plain, "  ");
    ]
  @
  match st with
  | Some s -> (
      [
        (Paint.bold, Text.value s.stats.last.value);
        (Paint.dim, " at " ^ Text.count s.stats.last.step);
      ]
      @
      match Option.bind s.goal (fun g -> Metric.best g s.stats) with
      | Some b ->
          [
            Paint.sep;
            (Paint.dim, "best ");
            (Paint.plain, Text.value b.value);
            (Paint.dim, " at " ^ Text.count b.step);
          ]
      | None -> [])
  | None -> [ (Paint.dim, "no " ^ key) ]

(* The cursor run's metrics, latest and best, grouped by prefix. *)
let metrics_block v r g ~x ~y ~w ~h =
  let st =
    match Data.stats v.data r with
    | Some l ->
        List.filter (fun (_, (s : History.key_stats)) -> s.kind = `Metric) l
    | None -> []
  in
  let name_w = w - 22 in
  Paint.spans g ~x ~y ~w
    [
      (Paint.dim, Text.pad name_w "metrics");
      (Paint.dim, Text.pad_left 10 "latest");
      (Paint.dim, Text.pad_left 12 "best");
    ];
  ignore
    (List.fold_left
       (fun (i, prev) (k, (s : History.key_stats)) ->
         let p, rest = Paint.key k in
         if i < h - 1 then begin
           let y = y + 1 + i in
           let shown = if p = prev then String.make (Text.width p) ' ' else p in
           Paint.spans g ~x ~y ~w:(name_w - 1)
             [
               (Paint.dim, shown);
               ((if Some k = v.metric then Paint.bold else Paint.plain), rest);
             ];
           let value = s.stats.last.value in
           Paint.right g ~x:(x + name_w) ~y ~w:10
             [
               ( (if Float.is_finite value then Paint.plain else Paint.bad),
                 Text.value value );
             ];
           match Option.bind s.goal (fun gl -> Metric.best gl s.stats) with
           | Some b ->
               Paint.right g
                 ~x:(x + name_w + 10)
                 ~y ~w:12
                 [ (Paint.dim, Text.value b.value) ]
           | None -> ()
         end;
         (i + 1, p))
       (0, "") st)

let pane t r g ~height =
  let v = t.view in
  let mx = margin t.width in
  let w = t.width - (2 * mx) in
  let y0 = pane_top t in
  let key = Option.value v.metric ~default:"" in
  let facts =
    [ (Paint.bold, name r); (Paint.plain, "  ") ]
    @ Paint.word (Data.word v.data ~now:v.now r)
    @ List.concat_map
        (fun (k, x) ->
          [ Paint.sep; (Paint.dim, k ^ " "); (Paint.plain, value_text x) ])
        (Run.params r)
  in
  let y_chart =
    if head_rows t = 2 then begin
      Paint.spans g ~x:mx ~y:y0 ~w facts;
      Paint.spans g ~x:mx ~y:(y0 + 1) ~w (metric_line v r key);
      y0 + 2
    end
    else begin
      Paint.spans g ~x:mx ~y:y0 ~w
        ([ (Paint.bold, name r); (Paint.plain, "  ") ]
        @ Paint.word (Data.word v.data ~now:v.now r)
        @ [ Paint.sep ] @ metric_line v r key);
      y0 + 1
    end
  in
  (match Option.bind (pane_request t ~height r) (Data.chart v.data) with
  | Some p -> ignore (Chart.draw ~light:v.light p g ~x:mx ~y:y_chart)
  | None ->
      Paint.text ~style:Paint.dim g ~x:(mx + Chart.left) ~y:(y_chart + 2) ~w
        "reading…");
  if block t > 0 then
    metrics_block v r g
      ~x:(mx + w - block t)
      ~y:y_chart ~w:(block t)
      ~h:(height - 1 - y_chart - 1)

let hints t =
  (if t.view.filter <> "" then [ ("esc", "clear filter") ] else [])
  @ [
      ("↑↓", "run");
      ("enter", "open");
      ("space", "mark");
      ("/", "filter");
      ("←→", "metric");
      ("o", "order");
      ("i", "details");
    ]
  @ if t.view.filter <> "" then [] else [ ("q", "quit") ]

(* [paint t g ~height ~cursor ~marks] draws Runs from [t], whatever the store's
   size, in time linear in the lines on screen. *)
let paint t g ~height ~cursor ~marks =
  let v = t.view in
  let width = t.width in
  let mx = margin width in
  let w = width - (2 * mx) in
  Paint.spans g ~x:mx ~y:0 ~w t.sentence;
  let root = tilde v.root in
  let room = w - Paint.width t.sentence - 4 in
  if room > 12 then
    Paint.right g ~x:mx ~y:0 ~w [ (Paint.dim, Paint.path room root) ];
  if Data.listing v.data = [] then begin
    let cy = (height / 2) - 2 in
    Paint.centre g ~y:cy ~w:width [ (Paint.bold, "No runs yet") ];
    Paint.centre g ~y:(cy + 2) ~w:width
      [ (Paint.dim, "A run appears here when a program starts one:") ];
    Paint.centre g ~y:(cy + 3) ~w:width
      [ (Paint.plain, "Munin.Session.start ~experiment:\"…\" ()") ];
    Paint.centre g ~y:(cy + 5) ~w:width
      [ (Paint.dim, "store " ^ Paint.path (width - 10) root) ];
    Paint.hints g ~x:mx ~y:(height - 1) ~w [ ("q", "quit") ]
  end
  else begin
    let n = Array.length t.lines in
    if Array.length t.rows > 0 then
      Paint.spans g ~x:(mx + gutter) ~y:2 ~w:(w - gutter) t.head;
    let shows_pane = pane_chart t ~height <> None in
    let room = if shows_pane then n else height - 3 - 3 in
    let here =
      match cursor with
      | Some id ->
          let rec find i =
            if i >= n then 0
            else
              match t.lines.(i) with
              | Row k when Run.id t.rows.(k) = id -> i
              | _ -> find (i + 1)
          in
          find 0
      | None -> 0
    in
    let first = if here < room then 0 else here - room + 1 in
    for row = 0 to min room (n - first) - 1 do
      let y = 3 + row in
      match t.lines.(first + row) with
      | Group spans -> Paint.spans g ~x:(mx + 1) ~y ~w:(w - 1) spans
      | Row k ->
          let r = t.rows.(k) in
          let sel = Some (Run.id r) = cursor in
          let base =
            if sel then Paint.selected ~light:v.light else Paint.plain
          in
          if sel then Paint.fill g ~x:mx ~y ~w base;
          (match List.find_opt (fun (_, id) -> id = Run.id r) marks with
          | Some (l, _) ->
              Paint.text
                ~style:(Paint.with_ base (Paint.mark l))
                g ~x:(mx + 1) ~y ~w:1 (String.make 1 l)
          | None -> ());
          let spans = t.cells.(k) in
          (* Under reverse video a coloured part would turn into a coloured
             block: the cursor row is drawn in one style, its name bold. *)
          let spans =
            if sel && not Paint.tinted then
              List.map (fun (_, s) -> (Paint.plain, s)) spans
            else spans
          in
          if sel && not Paint.tinted then
            Paint.text ~style:base g ~x:mx ~y ~w:1 "›";
          Paint.spans ~base g ~x:(mx + gutter) ~y ~w:(w - gutter) spans;
          if sel then
            Paint.text
              ~style:(Paint.with_ base Paint.bold)
              g
              ~x:(mx + gutter + t.run_x)
              ~y
              ~w:(Text.width (name r))
              (name r)
    done;
    if Array.length t.rows = 0 then
      Paint.text ~style:Paint.dim g ~x:(mx + gutter) ~y:3 ~w
        (Printf.sprintf "no run matches \"%s\"" v.filter);
    (match cursor with
    | Some id -> (
        match Array.find_opt (fun r -> Run.id r = id) t.rows with
        | Some r when shows_pane -> pane t r g ~height
        | Some r when height >= 12 ->
            Paint.spans g ~x:mx ~y:(height - 3) ~w
              ([ (Paint.bold, name r); (Paint.plain, "  ") ]
              @ Paint.word (Data.word v.data ~now:v.now r)
              @
              match
                Option.bind v.metric (fun k ->
                    Option.map (fun s -> (k, s)) (value v.data r k))
              with
              | Some (k, (s, _)) ->
                  [
                    Paint.sep;
                    (Paint.dim, k ^ " ");
                    (Paint.plain, Text.value s.value);
                  ]
              | None -> [])
        | _ -> ())
    | None -> ());
    if t.typing then begin
      let y = height - 1 in
      Paint.fill g ~x:0 ~y ~w:width Paint.plain;
      Paint.spans g ~x:mx ~y ~w
        [ (Paint.accent, "/ "); (Paint.plain, v.filter); (Paint.reverse, " ") ];
      Paint.right g ~x:mx ~y ~w
        [
          ( Paint.plain,
            Printf.sprintf "%d of %d" (Array.length t.rows)
              (List.length (Data.listing v.data)) );
          (Paint.dim, "   enter keep · esc clear");
        ]
    end
    else Paint.hints g ~x:mx ~y:(height - 1) ~w (hints t)
  end
