(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Mc = Matrix_charts
module Th = Theme

type role = Context | Overlay of char | Cursor

let smoothable key = String.starts_with ~prefix:"train/" key

let domain runs =
  match runs with
  | [] -> (0., 1.)
  | _ -> (0., List.fold_left (fun e (f : Facts.run) -> Float.max e f.x_end) 0. runs)

let tau (lo, hi) = (hi -. lo) /. 30.

let window (lo, hi) ~at ~narrow =
  let tiles = Float.ldexp 1. (max 0 narrow) in
  let w = (hi -. lo) /. tiles in
  let at = Float.min hi (Float.max lo at) in
  (* [at = hi] is in the last tile. *)
  let k = Float.min (tiles -. 1.) (Float.floor ((at -. lo) /. w)) in
  let start = lo +. (k *. w) in
  (start, start +. w)

(* Two decades between the least and greatest finite samples. *)
let decades = 100.

let log runs key =
  let extent =
    List.fold_left
      (fun acc f ->
        match Facts.stats f key with
        | Some { stats = { finite = Some { least; greatest; _ }; _ }; _ } -> (
            match acc with
            | None -> Some (least.value, greatest.value)
            | Some (lo, hi) -> Some (Float.min lo least.value, Float.max hi greatest.value))
        | _ -> acc)
      None runs
  in
  match extent with
  | Some (lo, hi) -> lo > 0. && hi >= decades *. lo
  | None -> false

(* Charts *)

type guides = [ `Both | `Y | `None ]
type line = { role : role; colour : int; reductions : Reduce.t list }

type t = {
  x : float * float;
  log : bool;
  smooth : bool;
  guides : guides;
  now : float option;
  cursor : float option;
  extents : bool;
  fill : bool;
  lines : line list;
}

(* The y labels' columns: every label is padded to them, so that the plot
   does not move with the values. *)
let label_width = 6

(* A raw sample further than this many dot rows from the line drawn is
   drawn on its own. *)
let spike = 3

let hatch = [ "╱   "; "   ╱"; "  ╱ "; " ╱  " ]

let theme th : Mc.Theme.t =
  let d = Mc.Theme.dark in
  {
    d with
    palette = [ Ansi.Color.default_fg ];
    background = None;
    axis = Th.fg th Line2;
    label = Th.fg th Dim;
    glyphs = { d.glyphs with dotted = ("┈", "┊") };
    dots = `Braille;
    tick_length = 0;
    label_padding = 0;
  }

(* Labels are chosen to fit the columns ({!y_ticks}), never cut. *)
let pad s = String.make (max 0 (label_width - Numbers.width s)) ' ' ^ s

(* The rows between two y ticks, at the least. *)
let tick_rows = 3

(* The bound on a quotient by a tick step, 2^52. A quotient errs by up to
   its magnitude times 2^-52 multiples, so below it a quotient places a
   multiple within one, and [n * m] stays an int. Past it the step is
   about a float's spacing, and several multiples round to one float. *)
let quotient_bound = Float.ldexp 1. 52

let y_ticks ~log ~most (lo, hi) =
  if log then
    let d = Numbers.decade lo in
    let a = if Numbers.decimal 1 d >= lo then d else d + 1 and b = Numbers.decade hi in
    let all = List.init (max 0 (b - a + 1)) (fun i -> Numbers.decimal 1 (a + i)) in
    let k = max 1 ((List.length all + most - 1) / most) in
    let ts = List.filteri (fun i _ -> i mod k = 0) all in
    if Numbers.tick ts = None then [] else ts
  else
    (* Halves, so that a domain as wide as the floats has a finite width. *)
    let half = (hi /. 2.) -. (lo /. 2.) in
    (* The multiples of [m] times [10^e] in the domain, each the float
       nearest its decimal; [None] when they are more than [most]. The
       quotients by the step place the first and last within one multiple,
       and the decimals decide. *)
    let multiples m e =
      let step = Numbers.decimal m e in
      let value n = Numbers.decimal (n * m) e in
      let a0 = Float.ceil (lo /. step) and b0 = Float.floor (hi /. step) in
      (* A step past the floats has one multiple, 0. *)
      if step = Float.infinity then Some (if lo <= 0. && hi >= 0. then [ 0. ] else [])
      else if
        not (step > 0. && Float.abs a0 < quotient_bound && Float.abs b0 < quotient_bound
             && b0 -. a0 <= Float.of_int (most + 1))
      then None
      else
        let a = Float.to_int a0 and b = Float.to_int b0 in
        let a = if value (a - 1) >= lo then a - 1 else if value a < lo then a + 1 else a in
        let b = if value (b + 1) <= hi then b + 1 else if value b > hi then b - 1 else b in
        if b - a >= most then None else Some (List.init (max 0 (b - a + 1)) (fun i -> value (a + i)))
    in
    (* From a decade below the step that gives [most], each step coarser;
       a step past the domain's values has no multiple in it but 0. *)
    let target = half *. (2. /. Float.of_int most) in
    let e0 = Numbers.decade (if target > 0. then target else half) - 1 in
    let rec search e = function
      | [] -> search (e + 1) [ 1; 2; 5 ]
      | m :: rest -> (
          match multiples m e with
          | Some ts when ts <> [] && Numbers.tick ts <> None -> ts
          | Some [] -> []
          | _ -> search e rest)
    in
    if half > 0. then search e0 [ 1; 2; 5 ] else []

let cursor_style th =
  let s = Th.fg ~bold:true th Hi in
  match Th.bg th Raise with Some c -> Ansi.Style.bg c s | None -> s

(* The figure the plot is laid out from: the scales, the guides with their
   labels, and the marks. A figure without marks has the plot of the figure
   with them, since every position scale has a given domain and the y
   labels have one width. *)
let figure th c ~y ~log ?(ticks = []) marks =
  let open Mc in
  let x_axis =
    match c.guides with
    | `Both ->
        let step = (snd c.x, Ansi.span "step") in
        let at =
          match c.cursor with
          | Some v -> [ (v, Ansi.span ~style:(cursor_style th) (Numbers.count (Float.to_int v))) ]
          | None -> []
        in
        axis ~format:(fun _ v -> Numbers.compact v) ~labels:(step :: at) "x"
    | `Y | `None -> axis ~show:false "x"
  in
  let y_axis =
    match c.guides with
    | `Both | `Y ->
        (* The ticks are labelled in one notation ({!y_ticks}). *)
        let label =
          match Numbers.tick ticks with
          | Some l -> l
          | None -> invalid_arg "Plot.figure: ticks without labels"
        in
        axis ~format:(fun _ v -> pad (label v)) "y"
    | `None -> axis ~show:false "y"
  in
  let ys = if log then Scale.log ~domain:y ~ticks () else Scale.linear ~domain:y ~ticks () in
  (* A mark without rows reads both scales, which a figure must. *)
  let base = line ~x:fst ~y:snd [||] in
  (* The y labels' columns and the axis line stay whatever the ticks. *)
  let left = match c.guides with `Both | `Y -> label_width + 1 | `None -> 0 in
  layer ((base :: marks) @ [ x_axis; y_axis ])
  |> scale "y" ys
  |> scale "x" (Scale.linear ~domain:c.x ())
  |> margins ~left

let layout th c ~width ~height ~y ~log ?ticks marks =
  Mc.layout ~theme:(theme th) ~width ~height (figure th c ~y ~log ?ticks marks)

let placeholder ~log = if log then (1., 10.) else (0., 1.)

(* The figure without its lines, which places the plot. *)
let skeleton th c ~width ~height =
  let log = c.log in
  let sk = layout th c ~width ~height ~y:(placeholder ~log) ~log [] in
  if (Mc.Layout.plot sk).width < 1 then None
  else
    let dot v = Mc.Layout.dot sk "x" `Braille (Num v) in
    Some (sk, dot (snd c.x) + 1, dot)

let dots th c ~width ~height =
  Option.map (fun (_, n, dot) -> (n, dot)) (skeleton th c ~width ~height)

(* The y domain *)

let valid ~log (a, b) =
  Float.is_finite a && Float.is_finite b && a < b && ((not log) || a > 0.)

(* A domain of one value, widened to the floats about it. *)
let about v =
  let a = Float.pred v and b = Float.succ v in
  if Float.is_finite a && Float.is_finite b then (a, b)
  else if Float.is_finite b then (v, b)
  else (a, v)

(* The values' extent, a twentieth wider each way; on a log scale down to
   the power of ten below when it is near, so that the plot's bottom is
   labelled, and a quarter above. *)
let widen ~log (lo, hi) =
  let wide =
    if log then
      let p = Numbers.decimal 1 (Numbers.decade lo) in
      ((if lo /. p < 2. then p else lo /. 1.25), hi *. 1.25)
    else
      let d = (hi -. lo) *. 0.05 in
      let d = if d > 0. && Float.is_finite d then d else Float.max 0.5 (Float.abs lo *. 0.05) in
      (lo -. d, hi +. d)
  in
  if valid ~log wide then wide else if valid ~log (lo, hi) then (lo, hi) else about lo

(* Lines *)

let fits c dots r =
  let g = Reduce.grid r in
  g.lo = fst c.x && g.hi = snd c.x && g.dots = dots

(* The x a mark draws in dot column [k]: the column's middle. *)
let column (g : Reduce.grid) k =
  g.lo +. ((Float.of_int k +. 0.5) *. (g.hi -. g.lo) /. Float.of_int g.dots)

(* The extent of the raw samples of [rs] in the plot. *)
let extent rs =
  let lo = ref Float.infinity and hi = ref Float.neg_infinity in
  List.iter
    (fun r ->
      for k = 0 to (Reduce.grid r).dots - 1 do
        match Reduce.extent r k with
        | Some (a, b) ->
            lo := Float.min !lo a;
            hi := Float.max !hi b
        | None -> ()
      done)
    rs;
  if !lo <= !hi then Some (!lo, !hi) else None

(* A line's points: the nearest sample left of the plot, each column's
   mean, the nearest right of it. *)
let points ~smooth ~log r =
  let g = Reduce.grid r in
  let left, right = Reduce.edges r ~smooth in
  let cols = ref [] in
  for k = g.dots - 1 downto 0 do
    let y = Reduce.mean r ~smooth ~log k in
    if not (Float.is_nan y) then cols := (column g k, y) :: !cols
  done;
  Array.of_list (Option.to_list left @ !cols @ Option.to_list right)

let last_in (lo, hi) pts =
  Array.fold_left (fun acc (x, y) -> if x >= lo && x <= hi then Some (x, y) else acc) None pts

(* The raw extremes of each column further than [spike] dot rows from the
   line drawn, smoothed or not. *)
let spikes ~smooth ~log ~ydot r =
  let g = Reduce.grid r in
  let drawable v = Float.is_finite v && ((not log) || v > 0.) in
  let acc = ref [] in
  for k = 0 to g.dots - 1 do
    let s = Reduce.mean r ~smooth ~log k in
    match Reduce.extent r k with
    | Some (a, b) when drawable s ->
        let at = ydot s in
        List.iter
          (fun v -> if drawable v && abs (ydot v - at) > spike then acc := (column g k, v) :: !acc)
          (if a = b then [ a ] else [ a; b ])
    | _ -> ()
  done;
  Array.of_list !acc

let bands r =
  let g = Reduce.grid r in
  let acc = ref [] in
  for k = g.dots - 1 downto 0 do
    match Reduce.extent r k with Some (a, b) -> acc := (column g k, a, b) :: !acc | None -> ()
  done;
  Array.of_list !acc

(* The first column of [r] with a sample that is not finite. *)
let first_event r =
  let g = Reduce.grid r in
  let rec go k =
    if k = g.dots then None
    else match Reduce.event r k with Some e -> Some (column g k, e) | None -> go (k + 1)
  in
  go 0

let event_label = function
  | `Nan -> "× " ^ Numbers.non_finite Float.nan
  | `Pos_inf -> "× " ^ Numbers.non_finite Float.infinity
  | `Neg_inf -> "× " ^ Numbers.non_finite Float.neg_infinity

(* Marks *)

let paint (s : Ansi.Style.t) mark =
  Mc.attrs s.attrs (mark (Mc.const (Option.value s.fg ~default:Ansi.Color.default_fg)))

(* A label one cell right of the point [(x, y)], moved inside the plot by
   matrix where the plot's edge would cut it. *)
let text s ~x ~y label =
  paint s (fun color ->
      Mc.text ~color ~offset:(1, 0) ~x:fst ~y:snd ~text:(fun _ -> label) [| (x, y) |])

let curve s pts = paint s (fun color -> Mc.line ~dots:`Braille ~color ~x:fst ~y:snd pts)

let filled s ~floor pts =
  paint s (fun color ->
      Mc.area ~dots:`Braille ~color ~x:fst ~y:snd ~y2:(fun _ -> floor) pts)

let style th l =
  match l.role with Context -> Th.faint th Dim | Overlay _ | Cursor -> Th.run th l.colour

(* The labels, over every line: the non-finite sample's and [now] at the
   plot's top, and, without colour, each overlay's letter. *)
let labels th c ~top ~event ~now lines =
  let at_top s x label = text s ~x ~y:top label in
  let letter (l, _, pts) =
    match (l.role, last_in c.x pts) with
    | Overlay letter, Some (x, y) ->
        Some (text (Th.fg ~bold:true th Hi) ~x ~y (String.make 1 letter))
    | _ -> None
  in
  Option.to_list
    (Option.map (fun (x, e) -> at_top (Th.fg ~bold:true th Fail) x (event_label e)) event)
  @ Option.to_list (Option.map (fun v -> at_top (Th.fg th Dim) v "now") now)
  @ if Th.coloured th then [] else List.filter_map letter lines

let draw th c g =
  let width = Matrix_grid.width g and height = Matrix_grid.height g in
  match skeleton th c ~width ~height with
  | None -> ()
  | Some (sk, n, _) ->
      let matched =
        List.filter_map
          (fun l ->
            Option.map (fun r -> (l, r)) (List.find_opt (fits c n) l.reductions))
          c.lines
      in
      let raw = extent (List.map snd matched) in
      let log = c.log && match raw with Some (lo, _) -> lo > 0. | None -> true in
      let y = match raw with Some e -> widen ~log e | None -> placeholder ~log in
      let lo, hi = c.x and top = snd y in
      let smooth = c.smooth in
      let line_points = List.map (fun (l, r) -> (l, r, points ~smooth ~log r)) matched in
      let cursor = List.find_opt (fun (l, _, _) -> l.role = Cursor) line_points in
      let in_x v = v >= lo && v <= hi in
      let event =
        if c.guides = `None then None
        else Option.bind cursor (fun (_, r, _) -> first_event r)
      in
      let hatched =
        match event with
        | Some (x, _) ->
            let span color = Mc.rect ~color ~pattern:hatch ~x:Fun.id ~x2:(fun _ -> hi) [| x |] in
            [ paint (Th.faint th Fail) span ]
        | None -> []
      in
      let rule (s : Ansi.Style.t) dash v = Mc.attrs s.attrs (Mc.vline ~dash ?color:s.fg v) in
      let now = match c.now with Some v when in_x v -> Some v | _ -> None in
      let rules =
        Option.to_list (Option.map (rule (Th.fg th Line2) `Dotted) now)
        @ Option.to_list (Option.map (rule (Th.fg th Mid) `Solid) c.cursor)
      in
      let spiky =
        match cursor with
        | Some (l, r, _) when not c.extents ->
            (* A spike is a sample more than three dot rows from the line in
               the final y domain, so finding spikes takes a third layout. *)
            let ys = layout th c ~width ~height ~y ~log [] in
            let ydot v = Mc.Layout.dot ys "y" `Braille (Num v) in
            [ paint (Th.faint_run th l.colour) (fun color ->
                  Mc.dot ~dots:`Braille ~color ~x:fst ~y:snd (spikes ~smooth ~log ~ydot r)) ]
        | _ -> []
      in
      let banded =
        match cursor with
        | Some (l, r, _) when c.extents ->
            [ paint (Th.faint_run th l.colour) (fun color ->
                  Mc.area ~dots:`Braille ~color
                    ~x:(fun (x, _, _) -> x) ~y:(fun (_, _, b) -> b) ~y2:(fun (_, a, _) -> a)
                    (bands r)) ]
        | None | Some _ -> []
      in
      let draw_line l pts =
        if c.fill then filled (style th l) ~floor:(fst y) pts else curve (style th l) pts
      in
      let of_role p =
        List.filter_map
          (fun (l, _, pts) -> if p l.role then Some (draw_line l pts) else None)
          line_points
      in
      let context = of_role (( = ) Context) in
      let overlays = of_role (function Overlay _ -> true | _ -> false) in
      let mine = of_role (( = ) Cursor) in
      let labels = labels th c ~top ~event ~now line_points in
      let marks = hatched @ rules @ spiky @ context @ banded @ overlays @ mine @ labels in
      let most = max 2 (1 + (((Mc.Layout.plot sk).height - 1) / tick_rows)) in
      let ticks = y_ticks ~log ~most y in
      Mc.draw (layout th c ~width ~height ~y ~log ~ticks marks) g ~x:0 ~y:0
