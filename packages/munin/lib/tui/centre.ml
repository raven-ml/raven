(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Munin
module Th = Theme
open Mosaic
module F = Text.Fragment
module M = Model

(* Text *)

let frag ?bold th tok s = F.text ~style:(Th.fg ?bold th tok) s
let shrinks = size_wh (px 0) (px 1)

(* A line takes one row: in a column, its growth would add rows. *)
let line ?(grow = true) frags =
  rich_text ~truncate:true ~wrap:`None ~min_size:shrinks
    ~max_size:(size_wh auto (px 1))
    ~flex_grow:(if grow then 1. else 0.)
    ~flex_shrink:1. frags

let cell w frags =
  rich_text ~truncate:true ~wrap:`None ~size:(size ~width:w ~height:1) ~flex_shrink:0. frags

(* [frags] at the right end of [w] cells. *)
let rcell w frags =
  box ~flex_direction:Row ~justify_content:Flex_end ~size:(size ~width:w ~height:1)
    ~flex_shrink:0.
    [ rich_text ~wrap:`None ~truncate:true frags ]

let right frags =
  box ~flex_direction:Row ~justify_content:Flex_end ~flex_grow:1. ~min_size:shrinks
    [ rich_text ~wrap:`None ~truncate:true ~flex_shrink:1. frags ]

let split left right =
  box ~flex_direction:Row ~flex_grow:1. ~flex_shrink:0. ~max_size:(size_wh auto (px 1))
    ~gap:(gap 1)
    [ line left; rich_text ~wrap:`None ~flex_shrink:0. right ]

(* An item wider than the line has a line of its own and wraps there. *)
let groups items =
  box ~flex_direction:Row ~flex_wrap:Wrap ~gap:(gap_xy 1 0) ~flex_shrink:1.
    ~min_size:(size_wh (px 0) auto)
    (List.map
       (fun frags -> rich_text ~wrap:`Word ~flex_shrink:1. ~min_size:(size_wh (px 0) auto) frags)
       items)

let label w frags =
  box ~size:(size ~width:w ~height:1) ~flex_shrink:0. [ cell (max 0 (w - 1)) frags ]

let rule th =
  box ~flex_grow:1. ~min_size:shrinks ~size:(size_wh auto (px 1)) ~border:true
    ~border_sides:[ `Top ] ?border_color:(Th.bg th Line) []

let section th title =
  box ~flex_direction:Row ~gap:(gap 1) ~flex_shrink:0.
    [ line ~grow:false [ frag ~bold:true th Mid title ]; rule th ]

let state_token (f : Facts.run) : Th.token =
  if Watch.broken f.state then Fail
  else if f.non_finite <> None && f.state <> Killed then Fail
  else
    match f.state with
    | Live | Writing | Running | Starting -> Live
    | Stalled | Silent -> Warn
    | Finished | Killed | Crashed | Failed -> Dim

let glyph th f = frag ~bold:true th (state_token f) (Facts.glyph f)

let arrow = function Some `Minimize -> "↓" | Some `Maximize -> "↑" | None -> ""

(* Values *)

let id (f : Facts.run) = Run.id f.run
let name (f : Facts.run) = Facts.name f.run

let value_tok x tok : Th.token = if Float.is_finite x then tok else Fail

let latest (f : Facts.run) k =
  Option.map (fun (s : History.key_stats) -> s.stats.last.value) (Facts.stats f k)

(* The hero's value of [k]: smoothed where the hero smooths it. *)
let hero_value (m : M.t) f k =
  let smoothed =
    List.find_map
      (fun (r : Data.readout) ->
        if r.key <> k then None
        else match Data.values m.data r with [ (_, _, s) ] -> s | _ -> None)
      (M.hero m)
  in
  match smoothed with
  | Some s -> Some (s, true)
  | None -> Option.map (fun x -> (x, false)) (latest f k)

(* The legend's value of a run: the drawn line's, at the legend's step. *)
let legend_value (m : M.t) ~smooth run =
  match M.legend m with
  | None -> None
  | Some r -> (
      match List.find_opt (fun (i, _, _) -> i = run) (Data.values m.data r) with
      | Some (_, raw, smoothed) -> if smooth && smoothed <> None then smoothed else raw
      | None -> None)

let goal_metrics (f : Facts.run) =
  List.filter (fun k -> Facts.goal f k <> None) f.metrics

let hero_key (m : M.t) (f : Facts.run) =
  match M.key m (Run.experiment f.run) with
  | Some k when List.mem k f.metrics -> Some k
  | _ -> List.nth_opt (goal_metrics f) 0

(* Banner *)

(* An error's first line, with [ …] when more follow: [i] gives them. *)
let first_line e =
  match String.index_opt e '\n' with
  | Some k when String.trim (String.sub e k (String.length e - k)) <> "" -> String.sub e 0 k ^ " …"
  | Some k -> String.sub e 0 k
  | None -> e

(* The banner wraps at its words, so that it says the error whole. *)
let banner th (f : Facts.run) =
  let at = Numbers.count (Option.value f.latest ~default:0) in
  let says =
    match (f.state, Run.status f.run) with
    | Failed, `Failed e -> Some ("✗ failed at step " ^ at ^ ": ", first_line e)
    | Crashed, _ -> Some ("✗ crashed at step " ^ at ^ ": ", "the writer exited without an end")
    | _ -> (
        match f.non_finite with
        | Some (k, s, n) ->
            Some
              ( Printf.sprintf "× %s %s from step %s" k (Numbers.non_finite s.value)
                  (Numbers.count s.step),
                if n > 1 then Printf.sprintf "; %d metrics went NaN" n else "" )
        | None -> None)
  in
  match says with
  | None -> empty
  | Some (head, rest) ->
      box ~flex_shrink:0. ?background:(Th.tint th) ~padding:(padding_xy 1 0)
        ~margin:(margin_lrtb 0 0 0 1)
        [
          rich_text ~wrap:`Word ~min_size:(size_wh (px 0) auto)
            [ frag ~bold:true th Fail head; frag th Fail rest ];
        ]

(* Hero *)

(* Glyphs three rows high, in half blocks; every row of a glyph is as wide. *)
let big_glyph = function
  | '0' -> Some [| "█▀█"; "█ █"; "▀▀▀" |]
  | '1' -> Some [| "▀█ "; " █ "; "▀▀▀" |]
  | '2' -> Some [| "▀▀█"; "█▀▀"; "▀▀▀" |]
  | '3' -> Some [| "▀▀█"; " ▀█"; "▀▀▀" |]
  | '4' -> Some [| "█ █"; "▀▀█"; "  ▀" |]
  | '5' -> Some [| "█▀▀"; "▀▀█"; "▀▀▀" |]
  | '6' -> Some [| "█▀▀"; "█▀█"; "▀▀▀" |]
  | '7' -> Some [| "▀▀█"; "  █"; "  ▀" |]
  | '8' -> Some [| "█▀█"; "█▀█"; "▀▀▀" |]
  | '9' -> Some [| "█▀█"; "▀▀█"; "▀▀▀" |]
  | '.' | ',' -> Some [| " "; " "; "▀" |]
  | '-' -> Some [| "  "; "▀▀"; "  " |]
  | '+' -> Some [| "   "; "▄█▄"; " ▀ " |]
  | 'N' -> Some [| "█▄ █"; "█ ▀█"; "▀  ▀" |]
  | 'a' -> Some [| "   "; "▄▄▄"; "█▄█" |]
  | 'e' -> Some [| "   "; "█▀█"; "▀▀▀" |]
  | 'i' -> Some [| "▀"; "█"; "▀" |]
  | 'k' -> Some [| "█ ▄"; "█▀▄"; "▀ ▀" |]
  | 'n' -> Some [| "   "; "█▀▄"; "▀ ▀" |]
  | 'f' -> Some [| "▄▀"; "█▀"; "▀ " |]
  | ' ' -> Some [| "  "; "  "; "  " |]
  | '%' -> Some [| "  "; "  "; "% " |]
  | _ -> None

(* [s] as three rows of big glyphs, or [s] on the middle row when a
   character has none. *)
let big s =
  let rec go i acc =
    if i = String.length s then Some (List.rev acc)
    else match big_glyph s.[i] with Some g -> go (i + 1) (g :: acc) | None -> None
  in
  match go 0 [] with
  | Some gs -> List.init 3 (fun r -> String.concat " " (List.map (fun g -> g.(r)) gs))
  | None -> [ ""; s; "" ]

(* The other goal metrics: labels and values in columns as wide as the
   widest; none when they do not fit [width] whole. *)
let hero_lines th ~width lines =
  let label k = Numbers.elide 15 k in
  let widest f = List.fold_left (fun w l -> max w (Numbers.width (f l))) 0 lines in
  let lw = widest (fun (k, _) -> label k) + 2 in
  let vw = widest (fun (k, (v, _)) -> Numbers.metric k v) in
  let note = if List.exists (fun (_, (_, s)) -> s) lines then 10 else 0 in
  if lw + vw + note > width then []
  else
    List.map
    (fun (k, (v, smoothed)) ->
      box ~flex_direction:Row ~flex_shrink:0. ~min_size:shrinks
        [
          cell lw [ frag th Dim (label k) ];
          rcell vw [ frag ~bold:true th (value_tok v Hi) (Numbers.metric k v) ];
          line (if smoothed then [ frag th Dim "  smoothed" ] else []);
        ])
    lines

(* The digits' column. *)
let digits_w = 24
let hero_gap = 3

let hero th (m : M.t) (f : Facts.run) n ~width =
  match hero_key m f with
  | None -> empty
  | Some k ->
      let colour = Option.value (M.colour m (id f)) ~default:0 in
      let digits =
        match hero_value m f k with
        | Some (v, _) ->
            let style =
              if Float.is_finite v then Th.run ~bold:true th colour
              else Th.fg ~bold:true th Fail
            in
            List.map (fun r -> line [ F.text ~style r ]) (big (Numbers.metric k v))
        | None -> [ line []; line [ frag th Dim "—" ]; line [] ]
      in
      let best =
        match Facts.best f k with
        | Some b ->
            [
              frag th Dim "best ";
              frag th Fg (Numbers.metric k b.value);
              frag th Dim (" at " ^ Numbers.count b.step);
            ]
        | None -> []
      in
      let others = List.filteri (fun i _ -> i < n) (List.filter (( <> ) k) (goal_metrics f)) in
      let others =
        hero_lines th ~width:(width - digits_w - hero_gap)
          (List.filter_map (fun o -> Option.map (fun v -> (o, v)) (hero_value m f o)) others)
      in
      let key_line =
        frag th Dim (match arrow (Facts.goal f k) with "" -> k | a -> k ^ " " ^ a)
      in
      box ~flex_direction:Row ~flex_shrink:0. ~margin:(margin_lrtb 0 0 1 1) ~gap:(gap hero_gap)
        [
          box ~flex_direction:Column ~flex_shrink:0. ~min_size:(size_wh (px digits_w) auto)
            ((line [ key_line ] :: digits) @ [ line best ]);
          box ~flex_direction:Column ~flex_grow:1. ~min_size:(size_wh (px 0) auto)
            ~overflow:{ x = Hidden; y = Hidden } ~margin:(margin_lrtb 0 0 1 0) others;
        ]

(* The blank columns between the line's values. *)
let part_gap = 3

let facts_line th (m : M.t) (f : Facts.run) ~width =
  let part k =
    Option.map
      (fun (v, _) ->
        let s = Numbers.metric k v in
        ( [ frag th Dim (k ^ " "); frag ~bold:true th (value_tok v Hi) s ],
          Numbers.width k + 1 + Numbers.width s ))
      (hero_value m f k)
  in
  let keys = List.filteri (fun i _ -> i < 4) (goal_metrics f) in
  (* The parts that fit whole, leaving from the last. *)
  let rec fit room = function
    | (frags, w) :: rest when w <= room -> frags :: fit (room - w - part_gap) rest
    | _ -> []
  in
  match fit width (List.filter_map part keys) with
  | [] -> box ~flex_shrink:0. ~margin:(margin_lrtb 0 0 1 0) []
  | parts ->
      let gap = frag th Dim (String.make part_gap ' ') in
      box ~flex_shrink:0. ~margin:(margin_lrtb 0 0 1 1)
        [ line (List.concat (List.mapi (fun i p -> if i = 0 then p else gap :: p) parts)) ]

(* Charts *)

let sized slot ~w ~h = Some (M.Sized (slot, w, h))

(* A chart: its size reported, so that a poll reduces its series for it. *)
let draw th p cv ~dt:_ =
  Canvas.clear cv;
  Plot.draw th p (Canvas.grid cv)

let plot th (m : M.t) (c : M.chart) =
  canvas ~flex_grow:1. ~min_size:(size_wh (px 0) (px 0)) ~on_size:(sized c.slot)
    (draw th (M.plot m c))

let tab_label k = Numbers.elide 20 k

(* The tabs from [a] to [b], with the gaps between them. *)
let tabs_width ms a b =
  List.fold_left ( + ) 0
    (List.filteri (fun j _ -> j >= a && j <= b) ms
    |> List.map (fun k -> Numbers.width (tab_label k) + 3))

(* The widest window of tabs about the one at [i] that fits [w]. *)
let tabs_window ms i w =
  let n = List.length ms in
  let fits a b = tabs_width ms a b <= w - 4 in
  let rec grow a b =
    if b + 1 < n && fits a (b + 1) then grow a (b + 1)
    else if a > 0 && fits (a - 1) b then grow (a - 1) b
    else (a, b)
  in
  grow i i

let tabs th (m : M.t) (f : Facts.run) ~width key =
  let ms = f.metrics in
  let i = Option.value (List.find_index (( = ) key) ms) ~default:0 in
  let a, b = tabs_window ms i width in
  let colour = Option.value (M.colour m (id f)) ~default:0 in
  let tab k =
    let s = tab_label k in
    let on = k = key in
    box ~flex_direction:Column ~flex_shrink:0.
      [
        line ~grow:false [ frag ~bold:on th (if on then Hi else Dim) s ];
        line ~grow:false
          (if on then [ F.text ~style:(Th.run th colour) (String.concat "" (List.init (Numbers.width s) (fun _ -> "▔"))) ]
           else [ F.text "" ]);
      ]
  in
  let shown = List.filteri (fun j _ -> j >= a && j <= b) ms in
  box ~flex_direction:Row ~gap:(gap 3) ~flex_shrink:0.
    ((if a > 0 then [ line ~grow:false [ frag th Dim "‹" ] ] else [])
    @ List.map tab shown
    @ if b < List.length ms - 1 then [ line ~grow:false [ frag th Dim "›" ] ] else [])

let legend th (m : M.t) (c : M.chart) =
  let mono = not (Th.coloured th) in
  let item (role, run) =
    match Data.run m.data run with
    | None -> None
    | Some f ->
        let here = role = Plot.Cursor in
        let colour = Option.value (M.colour m run) ~default:0 in
        let mark =
          match role with
          | Plot.Overlay l when mono -> F.text ~style:(Th.fg ~bold:true th Hi) (String.make 1 l ^ " ")
          | _ when mono -> F.text ~style:(Th.fg ~bold:true th Hi) "━ "
          | _ -> F.text ~style:(Th.run ~bold:true th colour) "━ "
        in
        let tok = if here then Th.Hi else Fg in
        let value =
          match legend_value m ~smooth:c.smooth run with
          | Some v -> frag ~bold:here th (value_tok v tok) (Numbers.metric c.key v)
          | None -> frag ~bold:here th tok "—"
        in
        Some
          (rich_text ~wrap:`None ~flex_shrink:0.
             [ mark; frag ~bold:here th (if here then Hi else Mid) (name f ^ " "); value ])
  in
  let marked = List.filter (fun (r, _) -> r <> Plot.Context) c.runs in
  let context = List.length c.runs - List.length marked in
  let tail =
    if m.others then
      rich_text ~wrap:`None ~flex_shrink:0.
        [ F.text ~style:(Th.faint th Dim) "━ "; frag th Dim (Printf.sprintf "%d others" context) ]
    else rich_text ~wrap:`None ~flex_shrink:0. [ frag ~bold:true th Mid "r "; frag th Dim "others" ]
  in
  (* The cursor's run first, then the overlays. *)
  let order = List.rev marked in
  box ~flex_direction:Row ~flex_wrap:Wrap ~gap:(gap_xy 3 0) ~flex_shrink:0.
    ~margin:(margin_lrtb 0 0 1 1)
    (List.filter_map item order @ [ tail ])

let charts th (m : M.t) (f : Facts.run) ~width =
  match M.charts m with
  | [] ->
      (* On the hero's row, which a run without a sample lacks. *)
      let ended = match f.state with Finished | Killed | Crashed | Failed -> true | _ -> false in
      box ~flex_shrink:0. ~margin:(margin_lrtb 0 0 1 0)
        [
          rich_text ~wrap:`Word ~min_size:(size_wh (px 0) auto)
            [ frag th Dim (if ended then "the run logged no metric" else "no metric sample yet") ];
        ]
  | large :: smalls ->
      let small (c : M.chart) =
        box ~flex_direction:Column ~flex_grow:1. ~flex_basis:(px 0)
          [ line [ frag ~bold:true th Mid (Numbers.elide 30 c.key) ]; plot th m c ]
      in
      box ~flex_direction:Column ~flex_grow:1.
        ([ tabs th m f ~width large.key; legend th m large; plot th m large ]
        @
        if smalls = [] then []
        else
          [
            box ~flex_direction:Row ~gap:(gap 2) ~size:(size_wh auto (px 8)) ~flex_shrink:0.
              ~margin:(margin_lrtb 0 0 1 0) (List.map small smalls);
          ])

(* Strips *)

let strip_label = 16

let goal_word = function
  | Some `Minimize -> "↓ min"
  | Some `Maximize -> "↑ max"
  | None -> ""

let machine_label k =
  if String.ends_with ~suffix:"busy_pct" k then
    if String.starts_with ~prefix:"sys/" k then "CPU busy" else "GPU busy"
  else if String.ends_with ~suffix:"used_bytes" k then "VRAM"
  else k

let machine_value k x =
  if String.ends_with ~suffix:"used_bytes" k then Numbers.gib x ^ " GiB" else Numbers.percent x

(* The strip's difference to the first overlay, coloured by the goal. *)
let difference th (m : M.t) (f : Facts.run) key =
  match M.compare m with
  | None -> []
  | Some c -> (
      match Data.difference m.data c key with
      | None -> []
      | Some (_, value, other) -> (
          match Numbers.difference key ~value ~other with
          | None -> [ frag th Dim "same" ]
          | Some s ->
              let tok : Th.token =
                match Facts.goal f key with
                | Some `Minimize -> if value < other then Live else Fail
                | Some `Maximize -> if value > other then Live else Fail
                | None -> Dim
              in
              [ frag th tok s ]))

let strips th (m : M.t) (f : Facts.run) ~width =
  let first =
    List.find_map
      (fun (r, o) -> match r with Plot.Overlay _ -> Data.run m.data o | _ -> None)
      (M.drawn m)
  in
  let vw = if first <> None && width >= 60 then 22 else 10 in
  let colour = Option.value (M.colour m (id f)) ~default:0 in
  let header =
    split
      ([ F.text ~style:(Th.run th colour) "━ "; frag ~bold:true th Hi (name f) ]
      @
      match first with
      | Some o ->
          [
            F.text ~style:(Th.run th (Option.value (M.colour m (id o)) ~default:0)) "   ━ ";
            frag th Mid (name o);
          ]
      | None -> [])
      [
        frag th Dim
          ("latest"
          ^ match first with Some o when vw = 22 -> "   vs " ^ Numbers.elide 8 (name o) | _ -> "");
      ]
  in
  let on = M.strip m in
  let charts = M.charts m in
  let strip (c : M.chart) =
    let here = Some c.key = on in
    let machine = c.axis = Data.Time in
    let surface = if here then Some Theme.Cursor else None in
    let label =
      if machine then [ line [ frag th Fg (machine_label c.key) ] ]
      else
        [
          line [ frag ~bold:here th (if here then Hi else Fg) (Numbers.elide (strip_label - 1) c.key) ];
          line [ frag th Dim (goal_word (Facts.goal f c.key)) ];
        ]
    in
    let value =
      match latest f c.key with
      | Some v ->
          let s = if machine then machine_value c.key v else Numbers.metric c.key v in
          [ frag ~bold:(not machine) th (value_tok v (if here then Hi else Fg)) s ]
      | None -> []
    in
    box ~flex_direction:Row ~size:(size_wh auto (px M.strip_height)) ~flex_shrink:0. ?surface
      ~margin:(margin_lrtb 0 0 0 M.strip_gap)
      [
        cell 1 (if here then [ F.text ~style:(Th.run th colour) "▌" ] else []);
        box ~flex_direction:Column ~size:(size_wh (px strip_label) (px M.strip_height))
          ~flex_shrink:0. label;
        plot th m c;
        box ~flex_direction:Row ~size:(size_wh (px vw) (px 1)) ~flex_shrink:0.
          ([ rcell 10 value ]
          @ if vw = 22 && not machine then [ rcell 12 (difference th m f c.key) ] else []);
      ]
  in
  let metrics = List.filter (fun (c : M.chart) -> c.axis = Data.Step) charts in
  let machine = List.filter (fun (c : M.chart) -> c.axis = Data.Time) charts in
  (* A scroll box shrinks below its content: the room it is given is the
     viewport it reports. *)
  box ~flex_direction:Column ~flex_grow:1. ~min_size:(size_wh (px 0) (px 0))
    [
      header;
      scroll_box ~flex_grow:1. ~min_size:(size_wh (px 0) (px 0)) ~margin:(margin_lrtb 0 0 1 0)
        ~show_scrollbars:false
        ~scroll:(Scroll.at m.strips_top)
        ~on_scroll:(fun s -> M.Strips_scrolled s)
        ~on_size:(fun ~w ~h -> Some (M.Sized (Strip_rows, w, h)))
        (List.map strip metrics
        @ if machine = [] then [] else section th "machine" :: List.map strip machine);
    ]

(* Zoom *)

let zoom th (m : M.t) (f : Facts.run) =
  match M.charts m with
  | [] -> line [ frag th Dim "no metric to zoom" ]
  | c :: _ ->
      let notes =
        List.filter (( <> ) "")
          [
            (match Facts.goal f c.key with
            | Some `Minimize -> "lower is better"
            | Some `Maximize -> "higher is better"
            | None -> "");
            (if c.log then "log" else "");
            (if c.smooth then "raw behind the smoothed line" else "");
          ]
      in
      let title =
        line [ frag ~bold:true th Hi c.key; frag th Dim ("   " ^ String.concat " · " notes) ]
      in
      let values = match M.legend m with Some r -> Data.values m.data r | None -> [] in
      let run (role, o) =
        match Data.run m.data o with
        | None -> []
        | Some x ->
            let here = role = Plot.Cursor in
            let colour = Option.value (M.colour m o) ~default:0 in
            let raw, smoothed =
              match List.find_opt (fun (i, _, _) -> i = o) values with
              | Some (_, r, s) -> (r, s)
              | None -> (None, None)
            in
            let shown = if c.smooth && smoothed <> None then smoothed else raw in
            [
              line
                [
                  F.text ~style:(Th.run th colour) "━ ";
                  frag ~bold:here th (if here then Hi else Fg) (name x);
                ];
              line
                ([
                   frag th Dim "  ";
                   (let tok = if here then Th.Hi else Fg in
                    match shown with
                    | Some v -> frag ~bold:true th (value_tok v tok) (Numbers.metric c.key v)
                    | None -> frag ~bold:true th tok "—");
                 ]
                @
                match raw with
                | Some v when c.smooth ->
                    [ frag th Dim "  raw "; frag th (value_tok v Dim) (Numbers.metric c.key v) ]
                | _ -> []);
              line [];
            ]
      in
      let marked = List.rev (List.filter (fun (r, _) -> r <> Plot.Context) c.runs) in
      let at =
        match M.zoom_step m with
        | Some s -> "at step " ^ Numbers.count s
        | None -> "at the cursor"
      in
      let showing =
        let lo, hi = c.x in
        Printf.sprintf "showing %s – %s" (Numbers.count (Float.to_int lo))
          (Numbers.count (Float.to_int hi))
      in
      box ~flex_direction:Column ~flex_grow:1. ~margin:(margin_lrtb 0 0 1 0)
        [
          title;
          legend th m c;
          box ~flex_direction:Row ~flex_grow:1.
            [
              plot th m c;
              box ~flex_direction:Column ~size:(size_wh (px 25) (pct 100)) ~flex_shrink:0.
                ~border:true ~border_sides:[ `Left ] ?border_color:(Th.bg th Line)
                ~padding:(padding_lrtb 1 0 0 0)
                ([ line [ frag ~bold:true th Mid at ]; line [] ]
                @ List.concat_map run marked
                @ [ box ~flex_grow:1. []; line [ frag th Dim showing ] ]);
            ];
        ]

(* The centre *)

let view th (m : M.t) (p : M.panes) f =
  (* What fits is decided from the width the content last reported, none
     before its first report. *)
  let width = match List.assoc_opt M.Centre m.slots with Some (w, _) -> w | None -> 0 in
  let head =
    match (m.depth, p.hero) with
    | Zoom _, _ -> empty
    | _, Big n -> hero th m f n ~width
    | _, Line -> facts_line th m f ~width
  in
  let body =
    match m.depth with
    | Charts -> charts th m f ~width
    | Strips _ -> strips th m f ~width
    | Zoom _ -> zoom th m f
  in
  box ~flex_direction:Column ~flex_grow:1. ~min_size:(size_wh (px 0) (px 0))
    ~padding:(padding_xy 1 0) ~overflow:{ x = Hidden; y = Hidden }
    [
      box ~flex_direction:Column ~flex_grow:1. ~min_size:(size_wh (px 0) (px 0))
        ~on_size:(fun ~w ~h -> Some (M.Sized (Centre, w, h)))
        [ banner th f; head; body ];
    ]
