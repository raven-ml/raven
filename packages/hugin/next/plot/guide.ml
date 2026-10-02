(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module P2 = Hugin_next_gg.P2
module Box2 = Hugin_next_gg.Box2
module Text = Hugin_next_text.Text
module Scale = Hugin_next_kit.Scale
module Ticks = Hugin_next_kit.Ticks
open Common
open Figure
open Resolved

(* Derived lengths, in em *)

let label_em = 0.9 (* Tick labels, legend entries and facet headers. *)
let head_em = 1.2 (* Figure titles, in bold. *)
let tick_em = 0.35
let pad_em = 0.25 (* From ticks, panels and swatches to their texts. *)
let inside_bar_em = 8.
let clear_em = 0.5 (* Between the labels of one axis or legend. *)
let title_gap_em = 0.5 (* Between a title and what it titles. *)
let gap_em = 1.
let swatch_em = 0.8 (* Square swatches, and the height of line swatches. *)
let line_em = 1.5 (* The length of line swatches. *)
let row_gap_em = 0.4 (* Between the swatches of legend rows. *)
let bar_em = 1. (* The width of colour bars. *)
let x_spacing_em = 5. (* The spacing ticks aim for across. *)
let y_spacing_em = 3.5 (* Up. *)

(* A continuous legend of entries chooses its ticks as a guide this many labels
   long would, which aims at half as many ticks: four entries, whether or not a
   legend of areas leaves out an end of area 0. *)
let legend_span = 9.

type sides = { left : float; right : float; top : float; bottom : float }

let no_sides = { left = 0.; right = 0.; top = 0.; bottom = 0. }

(* Elements *)

type placed = { text : Text.t; set : Text.Layout.t; at : P2.t; data : bool }

let text_box p =
  let b = Text.Layout.box p.set in
  Box2.v
    (P2.x p.at +. Box2.minx b)
    (P2.y p.at +. Box2.miny b)
    (Box2.w b) (Box2.h b)

type element =
  | Text of placed
  | Label of placed
  | Rules of (P2.t * P2.t) list
  | Grid_lines of (P2.t * P2.t) list
  | Bar of { box : Box2.t; scale : int; vertical : bool }
  | Swatch of {
      box : Box2.t;
      scale : int;
      entry : int;
      entries : int;
      u : float;
    }

let shift_box dx dy b =
  Box2.v (Box2.minx b +. dx) (Box2.miny b +. dy) (Box2.w b) (Box2.h b)

let shift dx dy e =
  let pt p = P2.v (P2.x p +. dx) (P2.y p +. dy) in
  let box = shift_box dx dy and seg (p, q) = (pt p, pt q) in
  match e with
  | Text p -> Text { p with at = pt p.at }
  | Label p -> Label { p with at = pt p.at }
  | Rules l -> Rules (List.map seg l)
  | Grid_lines l -> Grid_lines (List.map seg l)
  | Bar b -> Bar { b with box = box b.box }
  | Swatch s -> Swatch { s with box = box s.box }

(* [boxes e] is the boxes of the ink of [e] that count in its guide's bounds:
   grid lines lie in the data area. *)
let boxes = function
  | Text p | Label p -> [ text_box p ]
  | Rules l -> List.map (fun (p, q) -> Box2.of_pts p q) l
  | Grid_lines _ -> []
  | Bar { box; _ } | Swatch { box; _ } -> [ box ]

let hull = function
  | [] -> None
  | b :: bs -> Some (List.fold_left Box2.union b bs)

(* Specifications *)

type axis_part =
  | Ticks of { labelled : bool }
  | Header of int
  | Scale_title of side

type kind =
  | Axis of { guide : guide; scale : int; part : axis_part }
  | Legend of { guide : guide; scale : int }
  | Title of { align : Text.Layout.halign; text : Text.t }

type tier = Proper | Headers | Scale_titles | Legends | Figure_titles

let tier = function
  | Axis { part = Ticks _; _ } -> Proper
  | Axis { part = Header _; _ } -> Headers
  | Axis { part = Scale_title _; _ } -> Scale_titles
  | Legend _ -> Legends
  | Title _ -> Figure_titles

type spec = { id : id; side : side; kind : kind }

let inside g =
  match g.kind with
  | Legend { guide = { side = Some (`Inside c); _ }; _ } -> Some c
  | Legend _ | Axis _ | Title _ -> None

type t = {
  spec : spec;
  length : float;
  bounds : Box2.t option;
  least : float;
  wrap : float;
  elements : element list;
}

(* Measuring text *)

(* Text measurements, by size, alignment and text. *)
module Measures = Map.Make (struct
  type t = float * Text.Layout.halign * Text.Layout.valign * Text.t

  let rank_h : Text.Layout.halign -> int = function
    | `Left -> 0
    | `Center -> 1
    | `Right -> 2

  let rank_v : Text.Layout.valign -> int = function
    | `Top -> 0
    | `Cap -> 1
    | `Middle -> 2
    | `Baseline -> 3
    | `Bottom -> 4

  let compare (s, h, v, t) (s', h', v', t') =
    let c = Float.compare s s' in
    if c <> 0 then c
    else
      let c = Int.compare (rank_h h) (rank_h h') in
      if c <> 0 then c
      else
        let c = Int.compare (rank_v v) (rank_v v') in
        if c <> 0 then c else Text.compare t t'
end)

type cx = {
  theme : Theme.t;
  measures : Text.Layout.t Measures.t ref; (* Those this layout set. *)
  reused : Text.Layout.t Measures.t; (* Those of the previous layout. *)
  scales : fitted array;
  ticks : Ticks.t array;
}

let cx ?reused theme scales =
  let reused =
    match reused with
    | Some r when Theme.equal r.theme theme -> !(r.measures)
    | _ -> Measures.empty
  in
  { theme; measures = ref Measures.empty; reused; scales; ticks = [||] }

let with_ticks ticks cx = { cx with ticks }
let em cx k = k *. Theme.size cx.theme

let set cx ?(halign = `Left) ?(valign = `Baseline) k text =
  let size = em cx k in
  let key = (size, halign, valign, text) in
  match Measures.find_opt key !(cx.measures) with
  | Some l -> l
  | None ->
      (* A layout keeps the measurements it uses, so a chain of layouts each
         given the previous one holds no more than one does. *)
      let l =
        match Measures.find_opt key cx.reused with
        | Some l -> l
        | None ->
            Text.Layout.v ~halign ~valign ~fonts:(Theme.fonts cx.theme) ~size
              text
      in
      cx.measures := Measures.add key l !(cx.measures);
      l

let place cx ?halign ?valign ~data k text at =
  { text; set = set cx ?halign ?valign k text; at; data }

let width l = Box2.w (Text.Layout.box l)
let height l = Box2.h (Text.Layout.box l)
let longest f l = List.fold_left (fun m x -> Float.max m (f x)) 0. l

(* Texts *)

let tick_text (t : Ticks.tick) =
  match t.context with
  | None -> Text.v t.label
  | Some c -> Text.v (t.label ^ "\n" ^ c)

let categorical (F f) =
  match f.kind with Channel.Categories -> true | Channel.Quantities -> false

(* [guide_title s note] is the distinct titles of the channels reading [s] in
   the order of the figure, separated by commas, then [note]. *)
let guide_title (F f) note =
  let add acc m =
    match m.m_d.title with
    | Some t when not (List.exists (Text.equal t) acc) -> t :: acc
    | _ -> acc
  in
  let rec join = function
    | ([] | [ _ ]) as l -> l
    | t :: ts -> t :: Text.v ", " :: join ts
  in
  let titles = join (List.rev (List.fold_left add [] (by_order f.members))) in
  let note =
    match (note, titles) with
    | None, _ -> []
    | Some n, [] -> [ Text.v n ]
    | Some n, _ :: _ -> [ Text.v (" " ^ n) ]
  in
  match titles @ note with [] -> None | parts -> Some (Text.concat parts)

(* [category_text s k] is the text that shows the category of index [k] in the
   domain of the band scale [s]. *)
let category_text (F f) k =
  match (f.kind, Scale.domain f.scale) with
  | Channel.Categories, Scale.Categories (Scale.Labels l) -> Text.v l.(k)
  | Channel.Categories, Scale.Categories (Scale.Indices ix) ->
      Text.v (snd ix.(k))
  | Channel.Quantities, _ -> Text.v (string_of_int k)

(* [bar s] is [true] iff the legend of [s] is a colour bar. *)
let bar (F f as s) =
  (not (categorical s))
  && List.exists
       (fun m ->
         match m.m_use with
         | Role.Encoding { map = Color; _ } -> true
         | _ -> false)
       f.members

(* [sized s] is [true] iff the legend of [s] shows an area. *)
let sized (F f) =
  List.exists
    (fun m ->
      match m.m_use with
      | Role.Encoding { map = Area; _ } -> true
      | Role.Encoding _ | Position _ | Facet _ | Value -> false)
    f.members

(* [stroked s] is [true] iff every channel the legend of [s] shows is a [stroke]
   or a [dash], which lines and rules draw: its swatches are line swatches. *)
let stroked (F f) =
  List.for_all
    (fun m ->
      Role.shown_on m.m_use <> Some `Legend
      || String.equal m.m_role Role.stroke.name
      || String.equal m.m_role Role.dash.name)
    f.members

(* The frame of a side *)

let horizontal (side : side) =
  match side with `Top | `Bottom -> true | `Left | `Right -> false

(* [pt side a d] is the point [a] along [side] and [d] beyond it. *)
let pt (side : side) a d =
  match side with
  | `Top -> P2.v a (-.d)
  | `Bottom -> P2.v a d
  | `Left -> P2.v (-.d) a
  | `Right -> P2.v d a

(* [along side length u] is where the normalised position [u] lies along [side]:
   rightward on the top and bottom, upward on the left and right. *)
let along side length u =
  if horizontal side then u *. length else (1. -. u) *. length

(* The alignment of the texts beyond a side: tick labels, headers and titles. *)
let label_align : side -> Text.Layout.halign * Text.Layout.valign = function
  | `Bottom -> (`Center, `Top)
  | `Top -> (`Center, `Bottom)
  | `Left -> (`Right, `Middle)
  | `Right -> (`Left, `Middle)

(* [thin cx side labels] keeps, in order, each label that clears the last one
   kept with their clearance: a label is dropped only where it would meet a kept
   one, so an uneven set of explicit ticks keeps its lone labels. *)
let thin cx side labels =
  let centre p = if horizontal side then P2.x p.at else P2.y p.at in
  let extent p =
    (if horizontal side then width p.set else height p.set) +. em cx clear_em
  in
  let clears a b =
    Float.abs (centre b -. centre a) >= (extent a +. extent b) /. 2.
  in
  let keep acc p =
    match acc with last :: _ when not (clears last p) -> acc | _ -> p :: acc
  in
  List.rev (List.fold_left keep [] labels)

(* [ends side length labels] is the longest of [labels] along [side] moved to
   each end of the side: a label centred on a tick may lie there at another
   length, so a guide's reach past the ends does not depend on its length. *)
let ends side length labels =
  let along p =
    let b = text_box p in
    if horizontal side then Box2.w b else Box2.h b
  in
  match labels with
  | [] -> []
  | p :: ps ->
      let p =
        List.fold_left (fun p q -> if along q > along p then q else p) p ps
      in
      let a = if horizontal side then P2.x p.at else P2.y p.at in
      let moved a' =
        if horizontal side then
          { p with at = P2.v (P2.x p.at -. a +. a') (P2.y p.at) }
        else { p with at = P2.v (P2.x p.at) (P2.y p.at -. a +. a') }
      in
      [ moved 0.; moved length ]

(* [laid g length ~wrap ~least ~counted elements] is [g] drawing [elements],
   bounded by those and the elements [counted] that thinning removed. *)
let laid g length ~wrap ~least ~counted elements =
  let bounds = hull (List.concat_map boxes (counted @ elements)) in
  { spec = g; length; bounds; least; wrap; elements }

(* Laying out *)

let axis cx g ~length ~wrap ~across ~labelled ~grid s (t : Ticks.t) =
  let side = g.side and tick = em cx tick_em in
  let at (tk : Ticks.tick) = along side length tk.position in
  let rule (tk : Ticks.tick) d = (pt side (at tk) 0., pt side (at tk) d) in
  let rules =
    (pt side 0. 0., pt side length 0.)
    :: List.map (fun tk -> rule tk tick) t.major
  in
  let lines =
    if grid then
      [ Grid_lines (List.map (fun tk -> rule tk (-.across)) t.major) ]
    else []
  in
  let labels =
    if not labelled then []
    else
      let halign, valign = label_align side in
      let d = tick +. em cx pad_em in
      List.map
        (fun tk ->
          place cx ~halign ~valign ~data:(categorical s) label_em (tick_text tk)
            (pt side (at tk) d))
        t.major
  in
  let label p = Label p in
  laid g length ~wrap ~least:0.
    ~counted:(List.map label (labels @ ends side length labels))
    ((Rules rules :: lines) @ List.map label (thin cx side labels))

let header cx g ~length ~wrap s cat =
  let halign, valign = label_align g.side in
  let p =
    place cx ~halign ~valign ~data:true label_em (category_text s cat)
      (pt g.side (length /. 2.) (em cx pad_em))
  in
  let b = text_box p in
  laid g length ~wrap ~counted:[] [ Text p ]
    ~least:(if horizontal g.side then Box2.w b else Box2.h b)

(* [title cx g ~length ~wrap ~span align k text] sets [text] at [k] em, a gap
   beyond the side, aligned with [span]. A title centred or aligned with the end
   stays within the side and needs its length; one aligned with a start within
   the side needs the length from there; one starting past the end reaches past
   it by what anchors it there. So no title reaches further at another
   length. *)
let title cx g ~length ~wrap ~span:(s0, s1) align k text =
  let along l = if horizontal g.side then width l else height l in
  let w = along (set cx ~halign:align k text) in
  let a, least =
    match align with
    | `Left when s0 >= length -> (s0, 0.)
    | `Left -> (s0, Float.max 0. (s0 +. w))
    | `Center ->
        let mid = (s0 +. s1) /. 2. in
        (Float.min (length -. (w /. 2.)) (Float.max (w /. 2.) mid), w)
    | `Right -> (Float.min length s1, w)
  in
  let h, valign = label_align g.side in
  let halign = if horizontal g.side then align else h in
  let p =
    place cx ~halign ~valign ~data:false k text
      (pt g.side a (em cx title_gap_em))
  in
  laid g length ~wrap ~least ~counted:[] [ Text p ]

(* A legend is set as a block, its title first, with its top-left corner at the
   origin and its swatches or bar along the side from the origin; the block then
   moves a gap beyond the side. A vertical legend's title goes above the side's
   start. *)

(* [bar_body cx g ~length ~top s scale t] is the colour bar of a legend, its
   ticks, its labels and those it shows, [top] below the block's top. *)
let bar_body cx g ~length ~top s scale (t : Ticks.t) =
  let vertical = not (horizontal g.side) in
  let pad = em cx pad_em and tick = em cx tick_em and bw = em cx bar_em in
  (* Ticks and labels beyond the bar's far edge, as beyond a side. *)
  let at (tk : Ticks.tick) = along g.side length tk.position in
  let beyond a d = if vertical then P2.v d a else P2.v a (top +. d) in
  let halign, valign = label_align (if vertical then `Right else `Bottom) in
  let labels =
    List.map
      (fun tk ->
        place cx ~halign ~valign ~data:(categorical s) label_em (tick_text tk)
          (beyond (at tk) (bw +. tick +. pad)))
      t.major
  in
  let rules =
    List.map
      (fun tk -> (beyond (at tk) bw, beyond (at tk) (bw +. tick)))
      t.major
  in
  let box =
    if vertical then Box2.v 0. 0. bw length else Box2.v 0. top length bw
  in
  ( [ Bar { box; scale; vertical }; Rules rules ],
    labels @ ends (if vertical then `Left else `Top) length labels,
    thin cx (if vertical then `Left else `Top) labels,
    0. )

(* [entries_body cx g ~wrap ~top s scale t] is the swatches of a legend, its
   labels, and the length of its rows: stacked if vertical, else in rows of
   columns as wide as the widest entry, as many as fit in [wrap]. A legend of
   areas leaves out an entry of area 0, and its swatches hold the circle of the
   largest. *)
let entries_body cx g ~wrap ~top s scale (t : Ticks.t) =
  let area =
    Read.area
      { Read.theme = cx.theme; scales = cx.scales; frozen = cx.ticks }
      scale
  in
  let major, circle =
    if not (sized s) then (t.major, 0.)
    else
      let major =
        List.filter (fun (tk : Ticks.tick) -> area tk.position > 0.) t.major
      in
      let across (tk : Ticks.tick) =
        2. *. Float.sqrt (area tk.position /. Float.pi)
      in
      (major, longest across major)
  in
  let pad = em cx pad_em and sh = Float.max (em cx swatch_em) circle in
  let sw = if stroked s then Float.max (em cx line_em) sh else sh in
  let texts = List.map (fun tk -> set cx label_em (tick_text tk)) major in
  let n = List.length major in
  let row = Float.max (sh +. em cx row_gap_em) (longest height texts) in
  let entry = sw +. pad +. longest width texts in
  let col = entry +. em cx clear_em in
  let per =
    if not (horizontal g.side) then 1
    else max 1 (min n (Float.to_int ((wrap +. em cx clear_em) /. col)))
  in
  let one k (tk : Ticks.tick) =
    let x = float (k mod per) *. col and y = top +. (float (k / per) *. row) in
    let box = Box2.v x (y +. ((row -. sh) /. 2.)) sw sh in
    let at = P2.v (x +. sw +. pad) (y +. (row /. 2.)) in
    ( Swatch { box; scale; entry = k; entries = n; u = tk.position },
      place cx ~valign:`Middle ~data:(categorical s) label_em (tick_text tk) at
    )
  in
  let entries = List.mapi one major in
  let labels = List.map snd entries in
  let least = if horizontal g.side then entry else float n *. row in
  (List.map fst entries, labels, labels, least)

let legend cx g ~length ~wrap scale (t : Ticks.t) =
  let s = cx.scales.(scale) and vertical = not (horizontal g.side) in
  let pad = em cx pad_em in
  let reach =
    if not (bar s) then 0.
    else
      longest (fun tk -> height (set cx label_em (tick_text tk))) t.major /. 2.
  in
  let title =
    Option.map
      (fun text ->
        if vertical then
          place cx ~valign:`Bottom ~data:false 1. text
            (P2.v 0. (-.(pad +. reach)))
        else place cx ~valign:`Top ~data:false 1. text (P2.v 0. 0.))
      (guide_title s t.note)
  in
  let top =
    match title with
    | Some p when not vertical -> Box2.h (text_box p) +. pad
    | _ -> 0.
  in
  let body, counted, shown, least =
    if bar s then bar_body cx g ~length ~top s scale t
    else entries_body cx g ~wrap ~top s scale t
  in
  let title_w =
    Option.fold ~none:0. ~some:(fun p -> Box2.w (text_box p)) title
  in
  let least = if vertical then least else Float.max title_w least in
  let elements =
    Option.fold ~none:[] ~some:(fun p -> [ Text p ]) title
    @ body
    @ List.map (fun p -> Label p) shown
  in
  let counted = List.map (fun p -> Label p) counted in
  match hull (List.concat_map boxes (counted @ elements)) with
  | None -> laid g length ~wrap ~least:0. ~counted:[] []
  | Some block ->
      let gap = em cx gap_em in
      let dx, dy =
        match g.side with
        | `Right -> (gap, 0.)
        | `Left -> (-.(gap +. Box2.maxx block), 0.)
        | `Bottom -> (0., gap)
        | `Top -> (0., -.(gap +. Box2.maxy block))
      in
      laid g length ~wrap ~least
        ~counted:(List.map (shift dx dy) counted)
        (List.map (shift dx dy) elements)

let lay cx g ~length ~across ~wrap ~span =
  let empty =
    { spec = g; length; bounds = None; least = 0.; wrap; elements = [] }
  in
  match g.kind with
  | Title { align; text } ->
      title cx g ~length ~wrap ~span align head_em (Text.bold text)
  | Axis { guide = { show = false; _ }; _ }
  | Legend { guide = { show = false; _ }; _ } ->
      empty
  | Axis { guide; scale; part = Ticks { labelled } } ->
      let grid = match guide.kind with Axis a -> a.grid | Legend -> false in
      axis cx g ~length ~wrap ~across ~labelled ~grid cx.scales.(scale)
        cx.ticks.(scale)
  | Axis { scale; part = Header cat; _ } ->
      header cx g ~length ~wrap cx.scales.(scale) cat
  | Axis { scale; part = Scale_title axis; _ } -> (
      let s = cx.scales.(scale) in
      match guide_title s cx.ticks.(scale).note with
      | None -> empty
      | Some text ->
          let align = if horizontal axis then `Center else `Left in
          title cx g ~length ~wrap ~span align 1. text)
  | Legend { scale; _ } -> legend cx g ~length ~wrap scale cx.ticks.(scale)

let protrusion g =
  match g.bounds with
  | None -> no_sides
  | Some b -> (
      let pos x = Float.max 0. x in
      let lo, hi =
        if horizontal g.spec.side then (Box2.minx b, Box2.maxx b)
        else (Box2.miny b, Box2.maxy b)
      in
      let start = pos (-.lo) and past = pos (hi -. g.length) in
      let across = { no_sides with top = start; bottom = past } in
      let along = { no_sides with left = start; right = past } in
      match g.spec.side with
      | `Left -> { across with left = pos (-.Box2.minx b) }
      | `Right -> { across with right = pos (Box2.maxx b) }
      | `Top -> { along with top = pos (-.Box2.miny b) }
      | `Bottom -> { along with bottom = pos (Box2.maxy b) })

let move p g =
  let dx = P2.x p and dy = P2.y p in
  let bounds = Option.map (shift_box dx dy) g.bounds in
  { g with bounds; elements = List.map (shift dx dy) g.elements }

(* Choosing ticks *)

let all_ticks locale (F f) =
  match f.kind with
  | Channel.Categories ->
      Ticks.of_values ~locale f.scale (Array.of_list (category_names f.scale))
  | Channel.Quantities -> Ticks.of_values ~locale f.scale [||]

let notation (F f) =
  match f.kind with
  | Channel.Quantities -> Scale.notation f.scale
  | Channel.Categories -> None

(* A scale's explicit ticks are its guide values. Otherwise, a label's extent,
   and the spacing of ticks, are the greatest fractions of a guide's length they
   take, so that no two labels overlap on any guide. *)
let choose cx gs =
  let locale = Theme.locale cx.theme in
  let clear = em cx clear_em in
  (* Entries never overlap their labels, which have rows or columns of their
     own. *)
  let measure g length t =
    match g.kind with
    | Legend { scale; _ } when not (bar cx.scales.(scale)) -> 1. /. legend_span
    | Axis _ | Legend _ | Title _ ->
        let l = set cx label_em (Text.v t) in
        ((if horizontal g.side then width l else height l) +. clear) /. length
  in
  let spacing g =
    match g.kind with
    | Legend { scale; _ } when not (bar cx.scales.(scale)) -> 0.
    | Axis _ | Legend _ | Title _ ->
        em cx (if horizontal g.side then x_spacing_em else y_spacing_em)
  in
  Array.mapi
    (fun i (F f as s) ->
      let shows (g, _) =
        match g.kind with
        | Axis { scale; part = Ticks _ | Header _; _ } | Legend { scale; _ } ->
            scale = i
        | Axis { part = Scale_title _; _ } | Title _ -> false
      in
      let mine = List.filter shows gs in
      let every =
        List.exists
          (fun (g, _) ->
            match g.kind with
            | Axis { part = Header _; _ } -> true
            | Legend _ -> categorical s
            | Axis _ | Title _ -> false)
          mine
      in
      let guides =
        List.filter
          (fun (g, l) ->
            Float.is_finite l && l > 0.
            &&
            match g.kind with
            | Axis { part = Ticks _; _ } | Legend _ -> true
            | _ -> false)
          mine
      in
      let notation = notation s in
      match (Scale.ticks f.scale, guides) with
      | Some vs, _ -> Ticks.of_values ~locale ?notation f.scale vs
      | None, _ when every -> all_ticks locale s
      | None, [] -> Ticks.of_values ~locale f.scale [||]
      | None, guides ->
          let measure t = longest (fun (g, l) -> measure g l t) guides in
          let spacing = longest (fun (g, l) -> spacing g /. l) guides in
          Ticks.choose ~locale ?notation ~spacing ~length:1. ~measure f.scale)
    cx.scales

(* Comparing and formatting *)

let equal_placed p p' =
  Text.equal p.text p'.text
  && Text.Layout.equal p.set p'.set
  && P2.equal p.at p'.at && Bool.equal p.data p'.data

let equal_segs =
  List.equal (fun (p, q) (p', q') -> P2.equal p p' && P2.equal q q')

let equal_element e e' =
  match (e, e') with
  | Text p, Text p' | Label p, Label p' -> equal_placed p p'
  | Rules l, Rules l' | Grid_lines l, Grid_lines l' -> equal_segs l l'
  | Bar b, Bar b' ->
      Box2.equal b.box b'.box && Int.equal b.scale b'.scale
      && Bool.equal b.vertical b'.vertical
  | Swatch s, Swatch s' ->
      Box2.equal s.box s'.box && Int.equal s.scale s'.scale
      && Int.equal s.entry s'.entry
      && Int.equal s.entries s'.entries
      && Float.equal s.u s'.u
  | (Text _ | Label _ | Rules _ | Grid_lines _ | Bar _ | Swatch _), _ -> false

(* Guides are compared in layouts of equal resolved figures, where their order
   determines their kinds. *)
let equal g g' =
  Nx.Ptree.Path.equal g.spec.id g'.spec.id
  && equal_side g.spec.side g'.spec.side
  && Float.equal g.length g'.length
  && Option.equal Box2.equal g.bounds g'.bounds
  && Float.equal g.least g'.least
  && List.equal equal_element g.elements g'.elements

let pp_box ppf b =
  Format.fprintf ppf "[(%g, %g) (%g, %g)]" (Box2.minx b) (Box2.miny b)
    (Box2.maxx b) (Box2.maxy b)

(* A text and its box are on one line, whatever its length. *)
let pp_placed ppf p =
  let b = Buffer.create 64 in
  let line = Format.formatter_of_buffer b in
  Format.pp_set_geometry line ~max_indent:999_999 ~margin:1_000_000;
  Format.fprintf line "%a@?" Text.pp p.text;
  Format.fprintf ppf "%s %a" (Buffer.contents b) pp_box (text_box p)

let pp_element ppf = function
  | Text p -> Format.fprintf ppf "text %a" pp_placed p
  | Label p -> Format.fprintf ppf "label %a" pp_placed p
  | Rules l -> Format.fprintf ppf "rules %d" (List.length l)
  | Grid_lines l -> Format.fprintf ppf "grid %d" (List.length l)
  | Bar { box; _ } -> Format.fprintf ppf "bar %a" pp_box box
  | Swatch { box; u; _ } -> Format.fprintf ppf "swatch %g %a" u pp_box box

let pp ppf g =
  let kind =
    match g.spec.kind with
    | Axis { part = Ticks _; _ } -> "axis"
    | Axis { part = Header _; _ } -> "header"
    | Axis { part = Scale_title _; _ } -> "scale title"
    | Legend _ -> "legend"
    | Title _ -> "title"
  in
  Format.fprintf ppf "@[<v 2>%s %a %a" kind pp_id g.spec.id pp_side g.spec.side;
  List.iter (Format.fprintf ppf "@,%a" pp_element) g.elements;
  Format.fprintf ppf "@]"
