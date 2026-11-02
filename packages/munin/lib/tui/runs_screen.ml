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
}

(* Facts of a row *)

let name r = Option.value (Run.name r) ~default:(Run.id r)

let value_text : Value.t -> string = function
  | `Float x -> Text.value x
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

let non_finite d r key =
  match metric_stats d r key with
  | Some { stats = { non_finite = Some (_, f); _ }; _ } -> Some f
  | _ -> None

let has_non_finite d r =
  match Data.stats d r with
  | Some l ->
      List.exists
        (fun (_, (k : History.key_stats)) ->
          k.kind = `Metric && k.stats.non_finite <> None)
        l
  | None -> false

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

let rows v =
  let ws = words v.filter in
  let shown =
    List.filter (fun r -> List.for_all (matches v r) ws) (Data.listing v.data)
  in
  let by f a b = f a b in
  let cmp =
    match v.order with
    | Started -> fun _ _ -> 0
    | Name -> by (fun a b -> String.compare (name a) (name b))
    | State ->
        by (fun a b ->
            String.compare
              (Data.word v.data ~now:v.now a)
              (Data.word v.data ~now:v.now b))
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
  if v.reverse then List.rev sorted else sorted

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
  head : string;
  cells : string array; (* one per row *)
  right : bool;
  keep : int; (* the lower, the longer it stays *)
}

let differs f rows =
  match List.sort_uniq compare (List.map f rows) with
  | [] | [ _ ] -> false
  | _ -> true

let columns v ~wide rows =
  let d = v.data in
  let column ?(right = false) head keep cell =
    { head; cells = Array.of_list (List.map cell rows); right; keep }
  in
  let sorted head o =
    if v.order = o && o <> Started then head ^ if v.reverse then "▲" else "▼"
    else head
  in
  let when_differs f c = if differs f rows then [ Lazy.force c ] else [] in
  let metric =
    match v.metric with
    | None -> []
    | Some key ->
        let values = List.map (fun r -> value d r key) rows in
        let texts =
          Text.values
            (List.filter_map
               (Option.map (fun ((s : Metric.sample), _) -> s.value))
               values)
        in
        let starred = stars d rows key in
        let rec cells rows values texts =
          match (rows, values, texts) with
          | r :: rows, Some _ :: values, t :: texts ->
              (t ^ if List.mem (Run.id r) starred then "*" else " ")
              :: cells rows values texts
          | _ :: rows, None :: values, texts -> "—" :: cells rows values texts
          | _ -> []
        in
        let arrow = Fact.arrow (goal d rows key) in
        [
          {
            head = sorted (key ^ arrow) Value;
            cells = Array.of_list (cells rows values texts);
            right = true;
            keep = 0;
          };
          column ~right:true "AT" 22 (fun r ->
              match value d r key with
              | Some (s, true) -> Text.short s.step
              | _ -> "");
          column "NONFINITE" 2 (fun r ->
              match non_finite d r key with
              | Some f ->
                  Text.non_finite f.value ^ " "
                  ^ if wide then Text.count f.step else Text.short f.step
              | None -> "");
        ]
  in
  let param_keys =
    let seen = Hashtbl.create 8 in
    List.filter
      (fun k ->
        if Hashtbl.mem seen k then false
        else begin
          Hashtbl.add seen k ();
          differs
            (fun r -> Option.map value_text (List.assoc_opt k (Run.params r)))
            rows
        end)
      (List.concat_map (fun r -> List.map fst (Run.params r)) rows)
  in
  [
    column (sorted "STATE" State) 0 (Data.word d ~now:v.now);
    column (sorted "RUN" Name) 0 name;
  ]
  @ when_differs Run.experiment (lazy (column "EXPERIMENT" 3 Run.experiment))
  @ when_differs Run.parent
      (lazy
        (column "PARENT" 4 (fun r -> Option.value (Run.parent r) ~default:"—")))
  @ [
      column ~right:true "STARTED" 20 (fun r ->
          Text.clock ~now:v.now (Run.started_at r));
      column ~right:true "RAN" 21 (fun r -> Text.duration (Fact.wall r));
      column ~right:true "LAST" 8 (fun r ->
          Text.age (v.now -. Run.updated_at r));
      column ~right:true "STEP" 1 (fun r ->
          match latest_step d r with
          | None -> ""
          | Some s -> (
              (* Short of its steps, padded so that the percentages align. *)
              let n = if wide then Text.count else Text.short in
              match Run.steps r with
              | Some t when s < t ->
                  Printf.sprintf "%s/%s %3.0f%%" (n s) (n t)
                    (100. *. Float.of_int s /. Float.of_int t)
              | _ -> n s));
    ]
  @ metric
  @ when_differs host
      (lazy (column "HOST" 5 (fun r -> Option.value (host r) ~default:"—")))
  @ List.mapi
      (fun j k ->
        column ~right:true k
          (if j < 2 then 6 + j else 10 + j)
          (fun r ->
            match List.assoc_opt k (Run.params r) with
            | Some x -> value_text x
            | None -> "—"))
      param_keys
  @ when_differs
      (fun r -> Fact.commit (Run.provenance r))
      (lazy
        (column "COMMIT" 30 (fun r ->
             Option.value (Fact.commit (Run.provenance r)) ~default:"—")))

let width_of c =
  Array.fold_left (fun m s -> max m (Text.width s)) (Text.width c.head) c.cells

(* [fit w cols] leaves out the columns of greatest [keep] until the rest fit in
   [w], measuring each column once. *)
let fit w cols =
  let cols = List.map (fun c -> (c, width_of c)) cols in
  let total cols = 2 + List.fold_left (fun a (_, cw) -> a + cw + 2) (-2) cols in
  let rec go cols =
    if total cols <= w || List.for_all (fun (c, _) -> c.keep = 0) cols then cols
    else
      let worst = List.fold_left (fun m (c, _) -> max m c.keep) 0 cols in
      let rec drop = function
        | [] -> []
        | ((c, _) as x) :: rest ->
            if c.keep = worst then rest else x :: drop rest
      in
      go (List.rev (drop (List.rev cols)))
  in
  go cols

(* Tables *)

type table = {
  view : view;
  width : int;
  rows : Run.t array;
  head : string;
  lines : string array; (* each row's cells, without the gutter *)
  title : string;
}

let tilde root =
  match Sys.getenv_opt "HOME" with
  | Some h
    when h <> ""
         && String.length root >= String.length h
         && String.sub root 0 (String.length h) = h ->
      "~"
      ^ String.sub root (String.length h) (String.length root - String.length h)
  | _ -> root

let title v ~wide ~typing rows =
  let d = v.data in
  let count p = List.length (List.filter p rows) in
  let word w r = Data.word d ~now:v.now r = w in
  let parts =
    List.filter_map
      (fun (n, label) ->
        if n > 0 then Some (Printf.sprintf "%d %s" n label) else None)
      [
        (count (word "live"), "live");
        (count (word "failed"), "failed");
        (count (word "crashed"), "crashed");
        (count (has_non_finite d), "not finite");
      ]
  in
  let total = List.length (Data.listing d) in
  let n = List.length rows in
  let runs =
    if n = total then Printf.sprintf "%d run%s" n (if n = 1 then "" else "s")
    else Printf.sprintf "%d of %d runs" n total
  in
  let summary =
    if parts = [] then runs else runs ^ ": " ^ String.concat ", " parts
  in
  let common f label =
    match rows with
    | r :: _ when not (differs f rows) -> Option.map label (f r)
    | _ -> None
  in
  let shared =
    List.filter_map Fun.id
      [
        common (fun r -> Some (Run.experiment r)) Fun.id;
        common host (fun h -> "on " ^ h);
      ]
  in
  let filter =
    if v.filter <> "" && not typing then [ "filter " ^ v.filter ] else []
  in
  String.concat " · "
    ((if wide then [ "munin  " ^ tilde v.root; summary ]
      else [ "munin  " ^ summary ])
    @ shared @ filter)

let table v ~width ~typing =
  let wide = width >= 120 in
  let rows = rows v in
  let cols = fit width (columns v ~wide rows) in
  let line i =
    String.concat "  "
      (List.map
         (fun ((c : column), w) ->
           let s = if i < 0 then c.head else c.cells.(i) in
           if c.right then Text.pad_left w s else Text.pad w s)
         cols)
  in
  {
    view = v;
    width;
    rows = Array.of_list rows;
    head = "  " ^ line (-1);
    lines = Array.init (List.length rows) line;
    title = title v ~wide ~typing rows;
  }

let row_ids t = Array.to_list (Array.map Run.id t.rows)

(* The cursor's run *)

let sentence v r =
  let d = v.data in
  let facts =
    [ Data.word d ~now:v.now r ]
    @ (match (latest_step d r, Run.steps r) with
      | Some s, Some t ->
          [ Printf.sprintf "step %s of %s" (Text.count s) (Text.count t) ]
      | Some s, None -> [ "step " ^ Text.count s ]
      | None, _ -> [])
    @ [
        Option.value
          (Fact.commit ~dirty:" dirty" (Run.provenance r))
          ~default:"no commit";
      ]
  in
  name r ^ "  " ^ String.concat " · " facts

(* The pane, below the rows from 120 columns when they leave it [pane_rows]
   lines: the cursor's run's facts on the left, its focus metric over its steps
   on the right. *)
let pane_rows = 14

let pane_chart t ~height =
  let below = height - 4 - Array.length t.rows in
  if t.width >= 120 && below >= pane_rows then
    let half = (t.width - 2) / 2 in
    Option.map
      (fun key ->
        { Chart.key; width = half; height = min 10 (below - 3); x_axis = true })
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

let pane t r g ~y ~height ~screen ~light =
  let v = t.view in
  let width = t.width in
  let half = (width - 2) / 2 in
  let p = Run.provenance r in
  let d = v.data in
  let left =
    [
      Printf.sprintf "%s  %s  %s  %s" (name r) (Data.word d ~now:v.now r)
        (Run.experiment r) (Run.id r);
      Printf.sprintf "started %s · ran %s%s"
        (Text.date (Run.started_at r))
        (Text.duration (Fact.wall r))
        (match (latest_step d r, Run.steps r) with
        | Some s, Some n ->
            Printf.sprintf " · step %s of %s" (Text.count s) (Text.count n)
        | Some s, None -> " · step " ^ Text.count s
        | None, _ -> "");
      "";
      "params      "
      ^ String.concat "  "
          (List.map (fun (k, x) -> k ^ " " ^ value_text x) (Run.params r));
      "command     " ^ String.concat " " p.command;
      Printf.sprintf "source      %s · %s · pid %d"
        (Option.value (Fact.commit ~dirty:" dirty" p) ~default:"—")
        (Option.value p.hostname ~default:"—")
        p.pid;
      ("tags        "
      ^ match Run.tags r with [] -> "—" | t -> String.concat " " t);
      "notes       " ^ Option.value (Run.notes r) ~default:"—";
    ]
  in
  let metric_rows =
    match Data.stats d r with
    | None -> []
    | Some l ->
        ""
        :: Printf.sprintf "%-16s %10s  %10s  %8s  %8s  %s" "metric" "latest"
             "best" "at" "samples" "non-finite"
        :: List.filter_map
             (fun (k, (s : History.key_stats)) ->
               if s.kind <> `Metric then None
               else
                 let b = Option.bind s.goal (fun g -> Metric.best g s.stats) in
                 Some
                   (Printf.sprintf "%s %10s  %10s  %8s  %8d  %s"
                      (Text.pad 16 (Text.elide 16 k))
                      (Text.value s.stats.last.value)
                      (match b with
                      | Some b -> Text.value b.value
                      | None -> "—")
                      (match b with Some b -> Text.short b.step | None -> "")
                      s.stats.count
                      (match s.stats.non_finite with
                      | Some (n, f) ->
                          Printf.sprintf "%d from %s" n (Text.count f.step)
                      | None -> "")))
             l
  in
  List.iteri
    (fun i l -> if i < height then Paint.text g ~x:0 ~y:(y + i) ~w:half l)
    (left @ metric_rows);
  match pane_request t ~height:screen r with
  | Some c -> (
      let spec = c.spec in
      match Data.chart d c with
      | Some p ->
          Paint.text g ~x:(half + 2) ~y ~w:half (spec.key ^ " over steps");
          ignore (Chart.draw ~light p g ~x:(half + 2) ~y:(y + 1))
      | None -> ())
  | None -> ()

let hint ~wide =
  if wide then
    "↑↓ run  ←→ metric  enter open  space mark  / filter  o O order  i info  y \
     copy id  ? help  q quit"
  else
    "↑↓ run  ←→ metric  enter open  space mark  / filter  o order  ? help  q \
     quit"

(* [paint t g ~height ~cursor ~marks ~typing ~light] draws Runs from [t],
   whatever the store's size, in time linear in the rows on screen. *)
let paint t g ~height ~cursor ~marks ~typing ~light =
  let width = t.width in
  let wide = width >= 120 in
  let n = Array.length t.rows in
  Paint.text g ~x:0 ~y:0 ~w:width t.title;
  if n > 0 then Paint.text ~style:Paint.bold g ~x:0 ~y:2 ~w:width t.head;
  let below = height - 4 - n in
  let bottom = if wide && below >= pane_rows then below else 2 in
  let room = max 0 (height - 3 - bottom) in
  let here =
    match cursor with
    | Some id ->
        let rec find i =
          if i >= n then 0
          else if Run.id t.rows.(i) = id then i
          else find (i + 1)
        in
        find 0
    | None -> 0
  in
  let first = if here < room then 0 else here - room + 1 in
  for row = 0 to min room (n - first) - 1 do
    let i = first + row in
    let r = t.rows.(i) in
    let mark =
      match List.find_opt (fun (_, id) -> id = Run.id r) marks with
      | Some (l, _) -> String.make 1 l
      | None -> " "
    in
    let text = (if i = here then "▸" else " ") ^ mark ^ t.lines.(i) in
    let style = if i = here then Paint.reverse else Paint.plain in
    Paint.text ~style g ~x:0 ~y:(3 + row) ~w:width text
  done;
  if n = 0 then
    Paint.text g ~x:0 ~y:3 ~w:width
      (if Data.listing t.view.data = [] then
         Printf.sprintf "No run in %s yet. Runs appear here as they start."
           (tilde t.view.root)
       else "No run matches the filter.");
  (if n > 0 then
     let r = t.rows.(here) in
     if bottom > 2 then begin
       Paint.text g ~x:0 ~y:(height - bottom) ~w:width
         (String.concat "" (List.init width (fun _ -> "─")));
       pane t r g
         ~y:(height - bottom + 1)
         ~height:(bottom - 2) ~screen:height ~light
     end
     else Paint.text g ~x:0 ~y:(height - 2) ~w:width (sentence t.view r));
  if typing then begin
    let field = "/" ^ t.view.filter in
    Paint.text g ~x:0 ~y:(height - 1) ~w:width field;
    Paint.text ~style:Paint.reverse g ~x:(Text.width field) ~y:(height - 1) ~w:1
      " ";
    let keys = "enter keep  esc clear" in
    Paint.text g
      ~x:(width - Text.width keys)
      ~y:(height - 1) ~w:(Text.width keys) keys
  end
  else Paint.text g ~x:0 ~y:(height - 1) ~w:width (hint ~wide)
