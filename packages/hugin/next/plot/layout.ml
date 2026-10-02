(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Laying out

   [layout] turns a resolved figure into a tree of items: a leaf per panel, a
   grid per grid and per arrangement of facet panels, and around a block with
   legends of its scope or titles, a grid holding the block in its middle track
   and each legend and title in a track of its own. An item protrudes past its
   box by its guides and needs some lengths at least. A grid makes each gap the
   protrusions that meet it plus the theme's gap, gives each track the length
   its cells need, and shares the rest among its flexible tracks by weight, so
   the data areas of a column share their edges and those of a row their tops
   and bottoms.

   Ticks are chosen against the lengths of a solve without guides, then of a
   solve with the first ticks' guides, and the second choice is frozen for the
   final solve. *)

module P2 = Hugin_next_gg.P2
module Box2 = Hugin_next_gg.Box2
module Text = Hugin_next_text.Text
module Scale = Hugin_next_kit.Scale
module Ticks = Hugin_next_kit.Ticks
open Common
open Figure
open Resolved
open Items

(* Derived lengths, in em *)

let label_em = 0.9 (* Tick labels, legend entries and facet headers. *)
let tick_em = 0.35
let pad_em = 0.25 (* Between ticks, labels and titles, swatches and labels. *)
let clear_em = 0.5 (* Between the labels of one axis or legend. *)
let gap_em = 1.
let swatch_em = 1. (* Swatches and the width of colour bars. *)

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

(* A text set at a point of the page, upright or turned a quarter turn
   counterclockwise. [data] tells category labels, drawn whatever glyphs they
   lack, from figure text, which must have every glyph. *)
type placed = {
  text : Text.t;
  set : Text.Layout.t;
  at : P2.t;
  turned : bool;
  data : bool;
}

let placed_box p =
  let b = Text.Layout.box p.set and x = P2.x p.at and y = P2.y p.at in
  if not p.turned then
    Box2.v (x +. Box2.minx b) (y +. Box2.miny b) (Box2.w b) (Box2.h b)
  else
    (* The quarter turn takes (u, v) to (v, -u). *)
    Box2.v (x +. Box2.miny b) (y -. Box2.maxx b) (Box2.h b) (Box2.w b)

let equal_placed p p' =
  Text.equal p.text p'.text
  && Text.Layout.equal p.set p'.set
  && P2.equal p.at p'.at
  && Bool.equal p.turned p'.turned
  && Bool.equal p.data p'.data

(* Laid-out guides *)

type panel = { id : id; box : Box2.t; projection : Coord.projection }

type axis_out = {
  ax_id : id;
  ax_panel : id;
  ax_scale : int; (* Its index in the resolved figure's scales. *)
  ax_side : side;
  ax_grid : bool;
  ax_labels : placed list; (* Those drawn. *)
  ax_title : placed option;
}

type header_out = { hd_id : id; hd_panel : id; hd_label : placed }
type legend_entry = { u : float; swatch : Box2.t; label : placed }

type legend_body =
  | Bar of { bar : Box2.t; labels : placed list }
  | Entries of legend_entry list

type legend_out = {
  lg_id : id;
  lg_scale : int;
  lg_side : side;
  lg_title : placed option;
  lg_body : legend_body;
}

(* Measuring text *)

type cx = {
  theme : Theme.t;
  measures : Text.Layout.t Measures.t ref; (* Those this layout set. *)
  reused : Text.Layout.t Measures.t; (* Those of the previous layout. *)
  scales : fitted array;
  uses : use list array; (* Per scale. *)
  ticks : Ticks.t option array; (* None before the first choice. *)
  lengths : float list array; (* Per use, those of the previous pass. *)
  final : bool;
  notes : warning list ref; (* The category labels that lack glyphs. *)
}

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

let width l = Box2.w (Text.Layout.box l)
let height l = Box2.h (Text.Layout.box l)
let longest f l = List.fold_left (fun m x -> Float.max m (f x)) 0. l

let horizontal (side : side) =
  match side with `Top | `Bottom -> true | `Left | `Right -> false

(* [across side ~turned l] is the extent of [l] away from a panel's [side]. *)
let across side ~turned l =
  if horizontal side <> turned then height l else width l

(* The alignment of tick labels, and of the headers and titles beyond them. *)
let label_align : side -> Text.Layout.halign * Text.Layout.valign = function
  | `Bottom -> (`Center, `Top)
  | `Top -> (`Center, `Bottom)
  | `Left -> (`Right, `Middle)
  | `Right -> (`Left, `Middle)

let outer_align : side -> Text.Layout.halign * Text.Layout.valign * bool =
  function
  | `Bottom -> (`Center, `Top, false)
  | `Top -> (`Center, `Bottom, false)
  | `Left -> (`Center, `Bottom, true)
  | `Right -> (`Center, `Top, true)

(* [check cx owner p] raises if the figure text [p] lacks a glyph, and warns
   about a category label that does, in the final pass. *)
let check cx owner p =
  if cx.final then
    match Text.Layout.missing p.set with
    | [] -> ()
    | us ->
        let chars =
          String.concat ", "
            (List.map (fun u -> Printf.sprintf "U+%04X" (Uchar.to_int u)) us)
        in
        if p.data then
          cx.notes :=
            ( owner,
              Format.asprintf
                "the label %a has %s, which no face of the theme has" Text.pp
                p.text chars )
            :: !(cx.notes)
        else
          err "layout" "%a: %a holds %s, which no face of the theme has" pp_id
            owner Text.pp p.text chars

let place_text cx owner ?halign ?valign ~turned ~data k text at =
  let p = { text; set = set cx ?halign ?valign k text; at; turned; data } in
  check cx owner p;
  p

(* Protrusions *)

type sides = { left : float; right : float; top : float; bottom : float }

let no_sides = { left = 0.; right = 0.; top = 0.; bottom = 0. }

let add_side (side : side) d p =
  match side with
  | `Left -> { p with left = p.left +. d }
  | `Right -> { p with right = p.right +. d }
  | `Top -> { p with top = p.top +. d }
  | `Bottom -> { p with bottom = p.bottom +. d }

let axis_labels cx a (t : Ticks.t) =
  let halign, valign = label_align a.a_side in
  List.map (fun tk -> set cx ~halign ~valign label_em (tick_text tk)) t.major

let axis_title cx a (t : Ticks.t) =
  let halign, valign, _ = outer_align a.a_side in
  Option.map
    (set cx ~halign ~valign 1.)
    (guide_title cx.scales.(a.a_scale) t.note)

let header cx a cat =
  let halign, valign, _ = outer_align a.a_side in
  set cx ~halign ~valign label_em (category_text cx.scales.(a.a_scale) cat)

(* [depth cx a] is how far the guide [a] reaches from its panel's side. *)
let depth cx a =
  match (a.a_show, cx.ticks.(a.a_scale)) with
  | false, _ | _, None -> 0.
  | true, Some t -> (
      let _, _, turned = outer_align a.a_side in
      match a.a_role with
      | Gfx | Gfy -> (
          match a.a_category with
          | Some c when a.a_labelled ->
              em cx pad_em +. across a.a_side ~turned (header cx a c)
          | _ -> 0.)
      | Gx | Gy ->
          let tick = em cx tick_em in
          if not a.a_labelled then tick
          else
            let labels =
              longest (across a.a_side ~turned:false) (axis_labels cx a t)
            in
            let title =
              match axis_title cx a t with
              | None -> 0.
              | Some l -> em cx pad_em +. across a.a_side ~turned l
            in
            tick +. em cx pad_em +. labels +. title)

let vertical (side : side) = not (horizontal side)

(* The extents of a legend: across its track, along it at least, and above its
   start, where the title of a vertical legend goes. *)
type dims = { thick : float; least : float; above : float }

let legend_title cx ls (t : Ticks.t) =
  Option.map
    (set cx ~valign:`Bottom 1.)
    (guide_title cx.scales.(ls.ls_scale) t.note)

let entry_labels cx (t : Ticks.t) =
  List.map (fun tk -> set cx ~valign:`Middle label_em (tick_text tk)) t.major

(* [per_row cx ls labels] is how many entries of a horizontal legend share a
   row, as many as fit along its length in [cx], each as wide as the widest. *)
let per_row cx ls labels =
  let n = List.length labels in
  let cell =
    em cx swatch_em +. em cx pad_em +. longest width labels +. em cx clear_em
  in
  let length =
    List.fold_left2
      (fun l u l' -> match u with Legend_of _ -> l' | _ -> l)
      Float.nan cx.uses.(ls.ls_scale) cx.lengths.(ls.ls_scale)
  in
  if not (Float.is_finite length) then max 1 n
  else max 1 (min n (Float.to_int ((length +. em cx clear_em) /. cell)))

let dims cx ls =
  match cx.ticks.(ls.ls_scale) with
  | None -> { thick = 0.; least = 0.; above = 0. }
  | Some t ->
      let sw = em cx swatch_em and pad = em cx pad_em in
      let title = legend_title cx ls t in
      let title_w = Option.fold ~none:0. ~some:width title in
      let title_h =
        Option.fold ~none:0. ~some:(fun l -> height l +. pad) title
      in
      let labels = entry_labels cx t in
      let most f = longest f labels in
      let vert = vertical ls.ls_side in
      if ls.ls_bar then
        let ext = if vert then most width else most height in
        let thick = sw +. em cx tick_em +. pad +. ext in
        (* The labels at the ends of a vertical bar reach half their height past
           it, and its title goes above them. *)
        if vert then
          {
            thick = Float.max title_w thick;
            least = 0.;
            above = title_h +. (most height /. 2.);
          }
        else { thick = title_h +. thick; least = title_w; above = 0. }
      else
        let row = Float.max sw (most height) in
        let entry l = sw +. pad +. width l in
        if vert then
          {
            thick = Float.max title_w (most entry);
            least = float (List.length labels) *. row;
            above = title_h;
          }
        else
          let n = List.length labels and per = per_row cx ls labels in
          let rows = (n + per - 1) / per in
          {
            thick = title_h +. (float rows *. row);
            least = Float.max title_w (most entry);
            above = 0.;
          }

(* [ends side m] is the reach of a guide on [side] whose labels are at most [m]
   long along it: half of [m] past each end, where a label centred on a tick at
   an end reaches. *)
let ends (side : side) m =
  let m = m /. 2. in
  if horizontal side then { no_sides with left = m; right = m }
  else { no_sides with bottom = m; top = m }

let widest p q =
  {
    left = Float.max p.left q.left;
    right = Float.max p.right q.right;
    top = Float.max p.top q.top;
    bottom = Float.max p.bottom q.bottom;
  }

(* [reach cx a] is how far the tick labels of [a] reach past the ends of its
   panel's side. *)
let reach cx a =
  match (a.a_show, a.a_labelled, a.a_role, cx.ticks.(a.a_scale)) with
  | true, true, (Gx | Gy), Some t ->
      let along = if horizontal a.a_side then width else height in
      ends a.a_side (longest along (axis_labels cx a t))
  | _ -> no_sides

(* [spans cx a] is the length along its side that the title or header of [a]
   needs, which its panel's track gives it. *)
let spans cx a =
  match (a.a_show, a.a_labelled, cx.ticks.(a.a_scale)) with
  | true, true, Some t -> (
      let _, _, turned = outer_align a.a_side in
      let along l =
        if horizontal a.a_side <> turned then width l else height l
      in
      match (a.a_role, a.a_category) with
      | (Gfx | Gfy), Some c -> along (header cx a c)
      | (Gfx | Gfy), None -> 0.
      | (Gx | Gy), _ -> Option.fold ~none:0. ~some:along (axis_title cx a t))
  | _ -> 0.

let legend_prot cx ls =
  let above = { no_sides with top = (dims cx ls).above } in
  match cx.ticks.(ls.ls_scale) with
  | Some t when ls.ls_bar ->
      let vert = vertical ls.ls_side in
      let m = longest (if vert then height else width) (entry_labels cx t) in
      (* A vertical bar runs along the panels' side, a horizontal one across
         their bottom or top. *)
      widest above (ends (if vert then `Left else `Top) m)
  | Some _ | None -> above

let rec prot cx = function
  | Leaf l ->
      let deep =
        List.fold_left
          (fun p a -> add_side a.a_side (depth cx a) p)
          no_sides l.l_axes
      in
      List.fold_left (fun p a -> widest p (reach cx a)) deep l.l_axes
  | Grid g -> grid_prot cx g
  | Heading _ -> no_sides
  | Legend ls -> legend_prot cx ls

and grid_prot cx g =
  let nc = Array.length g.gcols and nr = Array.length g.grows in
  List.fold_left
    (fun p c ->
      let q = prot cx c.it in
      {
        left = (if c.c0 = 0 then Float.max p.left q.left else p.left);
        right =
          (if c.c0 + c.nc = nc then Float.max p.right q.right else p.right);
        top = (if c.r0 = 0 then Float.max p.top q.top else p.top);
        bottom =
          (if c.r0 + c.nr = nr then Float.max p.bottom q.bottom else p.bottom);
      })
    no_sides g.gcells

(* Solving grids *)

let sum a = Array.fold_left ( +. ) 0. a

(* [least tracks unit cells gaps] is the least length of each track: its weight
   times [unit.(i)] if it is flexible and at least what each cell it alone holds
   needs, with each cell spanning tracks given what it needs beyond them, by
   weight among its flexible tracks or else evenly. [cells] are the start, the
   number of tracks and the need of each cell. *)
let least tracks unit cells gaps =
  let m =
    Array.mapi
      (fun i t -> match t with Flex k -> k *. unit.(i) | Fixed -> 0.)
      tracks
  in
  List.iter (fun (s, n, l) -> if n = 1 then m.(s) <- Float.max m.(s) l) cells;
  List.iter
    (fun (s, n, l) ->
      if n > 1 then begin
        let have = ref 0. in
        for i = s to s + n - 1 do
          have := !have +. m.(i)
        done;
        for i = s to s + n - 2 do
          have := !have +. gaps.(i)
        done;
        let excess = l -. !have in
        if excess > 0. then begin
          let weight = ref 0. in
          for i = s to s + n - 1 do
            match tracks.(i) with
            | Flex k -> weight := !weight +. k
            | Fixed -> ()
          done;
          for i = s to s + n - 1 do
            let share =
              if !weight > 0. then
                match tracks.(i) with
                | Flex k -> excess *. k /. !weight
                | Fixed -> 0.
              else excess /. float n
            in
            m.(i) <- m.(i) +. share
          done
        end
      end)
    cells;
  m

(* [spread tracks pinned m avail] is [m] with the excess of [avail] over it
   shared by weight among the flexible tracks that are not [pinned], and the
   excess that no track takes. *)
let spread tracks pinned m avail =
  let l = Array.copy m in
  let excess = avail -. sum l in
  let weight = ref 0. in
  Array.iteri
    (fun i t ->
      match t with
      | Flex k when not pinned.(i) -> weight := !weight +. k
      | _ -> ())
    tracks;
  if excess <= 0. || !weight = 0. then (l, Float.max 0. excess)
  else begin
    Array.iteri
      (fun i t ->
        match t with
        | Flex k when not pinned.(i) ->
            l.(i) <- l.(i) +. (excess *. k /. !weight)
        | _ -> ())
      tracks;
    (l, 0.)
  end

(* [shrink base need target] is the greatest [s] in \[[0];[1]\] such that the
   rows of lengths [max base.(i) (s *. need.(i))] fit in [target]. *)
let shrink base need target =
  let n = Array.length base in
  let rec go active k =
    let fixed = ref 0. and needs = ref 0. in
    for i = 0 to n - 1 do
      if active.(i) then needs := !needs +. need.(i)
      else fixed := !fixed +. base.(i)
    done;
    let s =
      if !needs = 0. then 1.
      else Float.min 1. (Float.max 0. ((target -. !fixed) /. !needs))
    in
    let next = Array.mapi (fun i a -> a && s *. need.(i) > base.(i)) active in
    (* With no row left above its base, [s] is the scale that fits. *)
    if
      k = 0
      || Array.for_all2 Bool.equal next active
      || not (Array.exists Fun.id next)
    then s
    else go next (k - 1)
  in
  go (Array.mapi (fun i b -> need.(i) > b) base) n

type measured = {
  cgaps : float array;
  rgaps : float array;
  cols_least : float array;
  rows_least : float array; (* Before aspects. *)
  aspects : (gcell * float) list; (* Single cells holding panels with one. *)
  nrows : int;
}

type tracks = {
  col_len : float array;
  row_len : float array;
  col_gap : float array;
  row_gap : float array;
  x_off : float; (* What no track takes, split evenly about the tracks. *)
  y_off : float;
}

(* [natural cx unit item] is the least width and height of [item], [unit] being
   the data area a flexible track of weight [1.] has at least. *)
let rec natural cx unit = function
  | Leaf l -> (
      let w, h =
        List.fold_left
          (fun (w, h) a ->
            let n = spans cx a in
            if horizontal a.a_side then (Float.max w n, h)
            else (w, Float.max h n))
          (0., 0.) l.l_axes
      in
      (* A panel with an aspect fits its box in its cell, so the box holds what
         the cell must hold only if both lengths ask for it. *)
      match l.l_ratio with
      | None -> (w, h)
      | Some r -> (Float.max w (h /. r), Float.max h (r *. w)))
  | Grid g ->
      let m = measure_grid cx unit g in
      let rows = aspect_rows m m.cols_least in
      (sum m.cols_least +. sum m.cgaps, sum rows +. sum m.rgaps)
  | Heading { head; hside; _ } ->
      let l = set cx 1. head in
      if vertical hside then (height l, width l) else (width l, height l)
  | Legend ls ->
      let d = dims cx ls in
      if vertical ls.ls_side then (d.thick, d.least) else (d.least, d.thick)

and measure_grid cx (uw, uh) g =
  let cells =
    List.map (fun c -> (c, prot cx c.it, natural cx (uw, uh) c.it)) g.gcells
  in
  let gaps n first last before after =
    Array.init
      (max 0 (n - 1))
      (fun j ->
        let most f sel =
          List.fold_left
            (fun d (c, p, _) -> if sel c then Float.max d (f p) else d)
            0. cells
        in
        (* A title sits a pad from what it titles, other cells a gap apart. *)
        let meets c = last c = j || first c = j + 1 in
        let titles (c, _, _) =
          meets c && match c.it with Heading _ -> true | _ -> false
        in
        let sep = if List.exists titles cells then pad_em else gap_em in
        most before (fun c -> last c = j)
        +. most after (fun c -> first c = j + 1)
        +. em cx sep)
  in
  let cgaps =
    gaps (Array.length g.gcols)
      (fun c -> c.c0)
      (fun c -> c.c0 + c.nc - 1)
      (fun p -> p.right)
      (fun p -> p.left)
  in
  let rgaps =
    gaps (Array.length g.grows)
      (fun c -> c.r0)
      (fun c -> c.r0 + c.nr - 1)
      (fun p -> p.bottom)
      (fun p -> p.top)
  in
  let aspects =
    List.filter_map
      (fun (c, _, _) ->
        match c.it with
        | Leaf { l_ratio = Some r; _ } when c.nr = 1 && c.nc = 1 -> Some (c, r)
        | _ -> None)
      cells
  in
  let cols_least =
    least g.gcols
      (Array.make (Array.length g.gcols) uw)
      (List.map (fun (c, _, (w, _)) -> (c.c0, c.nc, w)) cells)
      cgaps
  in
  (* A row holding a panel with an aspect is not flexible: its height follows
     its columns. *)
  let rows_least =
    let unit = Array.make (Array.length g.grows) uh in
    List.iter (fun (c, _) -> unit.(c.r0) <- 0.) aspects;
    least g.grows unit
      (List.map (fun (c, _, (_, h)) -> (c.r0, c.nr, h)) cells)
      rgaps
  in
  {
    cgaps;
    rgaps;
    cols_least;
    rows_least;
    aspects;
    nrows = Array.length g.grows;
  }

(* [aspect_rows m cols] is the least length of each row, a row holding a panel
   with an aspect needing that panel's height at the width of its column. *)
and aspect_rows m cols =
  let need = aspect_need m cols in
  Array.mapi (fun i b -> Float.max b need.(i)) m.rows_least

and aspect_need m cols =
  let need = Array.make m.nrows 0. in
  List.iter
    (fun (c, r) -> need.(c.r0) <- Float.max need.(c.r0) (r *. cols.(c.c0)))
    m.aspects;
  need

and solve_grid cx unit g w h =
  let m = measure_grid cx unit g in
  let avail_w = w -. sum m.cgaps and avail_h = h -. sum m.rgaps in
  let nc = Array.length g.gcols in
  let cols, slack_x =
    spread g.gcols (Array.make nc false) m.cols_least avail_w
  in
  let is_aspect = Array.make m.nrows false in
  List.iter (fun (c, _) -> is_aspect.(c.r0) <- true) m.aspects;
  let rows = aspect_rows m cols in
  let done_ cols slack_x rows =
    let rows, slack_y = spread g.grows is_aspect rows avail_h in
    {
      col_len = cols;
      row_len = rows;
      col_gap = m.cgaps;
      row_gap = m.rgaps;
      x_off = slack_x /. 2.;
      y_off = slack_y /. 2.;
    }
  in
  if sum rows <= avail_h || m.aspects = [] then done_ cols slack_x rows
  else begin
    (* Too short for its panels with an aspect: they shrink by one factor, which
       pins the widths of their columns. *)
    let need = aspect_need m cols in
    let s = shrink m.rows_least need avail_h in
    let rows =
      Array.mapi (fun i b -> Float.max b (s *. need.(i))) m.rows_least
    in
    let pinned = Array.make nc false and least = Array.copy m.cols_least in
    List.iter
      (fun (c, _) ->
        pinned.(c.c0) <- true;
        least.(c.c0) <- Float.max least.(c.c0) (s *. cols.(c.c0)))
      m.aspects;
    let cols, slack_x = spread g.gcols pinned least avail_w in
    done_ cols slack_x rows
  end

(* Placing *)

type acc = {
  mutable panels : (panel * Coord.t) list;
  mutable axes : axis_out list;
  mutable headers : header_out list;
  mutable legends : legend_out list;
  mutable titles : placed list;
  mutable spans : (id * Box2.t) list; (* The hull of each block's panels. *)
  mutable along : (int * guide_role * float) list;
      (* Each axis's scale, role and length. *)
}

let fit_aspect r box =
  let w = Box2.w box and h = Box2.h box in
  let w', h' = if h >= r *. w then (w, r *. w) else (h /. r, h) in
  Box2.v
    (Box2.minx box +. ((w -. w') /. 2.))
    (Box2.miny box +. ((h -. h') /. 2.))
    w' h'

(* [anchor box side c d] is the point [d] beyond the [side] of [box], at [c]
   along it. *)
let anchor box (side : side) c d =
  match side with
  | `Bottom -> P2.v c (Box2.maxy box +. d)
  | `Top -> P2.v c (Box2.miny box -. d)
  | `Left -> P2.v (Box2.minx box -. d) c
  | `Right -> P2.v (Box2.maxx box +. d) c

let middle box side =
  if horizontal side then P2.x (Box2.mid box) else P2.y (Box2.mid box)

(* [thin cx side labels] drops alternate labels until no two adjacent ones
   overlap with their clearance. *)
let thin cx side labels =
  let centre p = if horizontal side then P2.x p.at else P2.y p.at in
  let extent p =
    (if horizontal side then width p.set else height p.set) +. em cx clear_em
  in
  let rec overlap = function
    | a :: (b :: _ as rest) ->
        Float.abs (centre b -. centre a) < (extent a +. extent b) /. 2.
        || overlap rest
    | _ -> false
  in
  let rec alternate = function
    | a :: _ :: rest -> a :: alternate rest
    | l -> l
  in
  let rec go l = if overlap l then go (alternate l) else l in
  go labels

let place_axis cx acc l proj box a offset =
  match (a.a_show, cx.ticks.(a.a_scale)) with
  | false, _ | _, None -> ()
  | true, Some t -> (
      let pad = em cx pad_em and s = cx.scales.(a.a_scale) in
      let halign, valign, turned = outer_align a.a_side in
      match a.a_role with
      | Gfx | Gfy -> (
          match a.a_category with
          | Some c when a.a_labelled ->
              let at =
                anchor box a.a_side (middle box a.a_side) (offset +. pad)
              in
              let label =
                place_text cx a.a_id ~halign ~valign ~turned ~data:true label_em
                  (category_text s c) at
              in
              acc.headers <-
                { hd_id = a.a_id; hd_panel = l.l_id; hd_label = label }
                :: acc.headers
          | _ -> ())
      | Gx | Gy ->
          let along u =
            match a.a_role with
            | Gx -> P2.x (Coord.point proj u 0.)
            | Gy | Gfx | Gfy -> P2.y (Coord.point proj 0. u)
          in
          let labels, title =
            if not a.a_labelled then ([], None)
            else
              let d = offset +. em cx tick_em +. pad in
              let lh, lv = label_align a.a_side in
              let label (tk : Ticks.tick) =
                place_text cx a.a_id ~halign:lh ~valign:lv ~turned:false
                  ~data:(categorical s) label_em (tick_text tk)
                  (anchor box a.a_side (along tk.position) d)
              in
              let labels = List.map label t.major in
              let deep =
                longest (fun p -> across a.a_side ~turned:false p.set) labels
              in
              let title text =
                place_text cx a.a_id ~halign ~valign ~turned ~data:false 1. text
                  (anchor box a.a_side (middle box a.a_side) (d +. deep +. pad))
              in
              (thin cx a.a_side labels, Option.map title (guide_title s t.note))
          in
          acc.axes <-
            {
              ax_id = a.a_id;
              ax_panel = l.l_id;
              ax_scale = a.a_scale;
              ax_side = a.a_side;
              ax_grid = a.a_grid;
              ax_labels = labels;
              ax_title = title;
            }
            :: acc.axes)

let place_leaf cx acc l box =
  let box = match l.l_ratio with None -> box | Some r -> fit_aspect r box in
  let proj = Coord.project l.l_coord box in
  acc.panels <-
    ({ id = l.l_id; box; projection = proj }, l.l_coord) :: acc.panels;
  acc.spans <- (l.l_id, box) :: acc.spans;
  let reached = ref no_sides in
  List.iter
    (fun a ->
      (match a.a_role with
      | Gx -> acc.along <- (a.a_scale, Gx, Box2.w box) :: acc.along
      | Gy -> acc.along <- (a.a_scale, Gy, Box2.h box) :: acc.along
      | Gfx | Gfy -> ());
      let offset =
        match a.a_side with
        | `Left -> !reached.left
        | `Right -> !reached.right
        | `Top -> !reached.top
        | `Bottom -> !reached.bottom
      in
      place_axis cx acc l proj box a offset;
      reached := add_side a.a_side (depth cx a) !reached)
    l.l_axes;
  box

let place_legend cx acc ls cell span =
  match cx.ticks.(ls.ls_scale) with
  | None -> ()
  | Some t ->
      let sw = em cx swatch_em and pad = em cx pad_em in
      let s = cx.scales.(ls.ls_scale) and vert = vertical ls.ls_side in
      let span = Option.value span ~default:cell in
      let x0 = if vert then Box2.minx cell else Box2.minx span in
      let y0 = if vert then Box2.miny span else Box2.miny cell in
      let title text =
        let reach =
          if ls.ls_bar then longest height (entry_labels cx t) /. 2. else 0.
        in
        let at, valign =
          if vert then (P2.v x0 (y0 -. pad -. reach), `Bottom)
          else (P2.v x0 y0, `Top)
        in
        place_text cx ls.ls_id ~valign ~turned:false ~data:false 1. text at
      in
      let title = Option.map title (guide_title s t.note) in
      let y0 =
        match title with
        | Some p when not vert -> y0 +. height p.set +. pad
        | _ -> y0
      in
      let label ?halign ~valign (tk : Ticks.tick) at =
        place_text cx ls.ls_id ?halign ~valign ~turned:false
          ~data:(categorical s) label_em (tick_text tk) at
      in
      let body =
        if ls.ls_bar then
          let off = sw +. em cx tick_em +. pad in
          if vert then
            let h = Box2.h span in
            let at (tk : Ticks.tick) =
              P2.v (x0 +. off) (Box2.maxy span -. (tk.position *. h))
            in
            Bar
              {
                bar = Box2.v x0 y0 sw h;
                labels =
                  thin cx `Right
                    (List.map
                       (fun tk -> label ~valign:`Middle tk (at tk))
                       t.major);
              }
          else
            let w = Box2.w span in
            let at (tk : Ticks.tick) =
              P2.v (x0 +. (tk.position *. w)) (y0 +. off)
            in
            Bar
              {
                bar = Box2.v x0 y0 w sw;
                labels =
                  thin cx `Bottom
                    (List.map
                       (fun tk -> label ~halign:`Center ~valign:`Top tk (at tk))
                       t.major);
              }
        else
          let labels = entry_labels cx t in
          let row = Float.max sw (longest height labels) in
          (* A vertical legend stacks its entries; a horizontal one sets them in
             rows of columns as wide as the widest entry. *)
          let per = if vert then 1 else per_row cx ls labels in
          let col = sw +. pad +. longest width labels +. em cx clear_em in
          let entry k (tk : Ticks.tick) =
            let x = x0 +. (float (k mod per) *. col) in
            let y = y0 +. (float (k / per) *. row) in
            let swatch = Box2.v x (y +. ((row -. sw) /. 2.)) sw sw in
            let label =
              label ~valign:`Middle tk
                (P2.v (x +. sw +. pad) (y +. (row /. 2.)))
            in
            { u = tk.position; swatch; label }
          in
          Entries (List.mapi entry t.major)
      in
      acc.legends <-
        {
          lg_id = ls.ls_id;
          lg_scale = ls.ls_scale;
          lg_side = ls.ls_side;
          lg_title = title;
          lg_body = body;
        }
        :: acc.legends

let place_heading cx ~owner ~align ~head ~side cell span outer =
  let span = Option.value span ~default:cell in
  let _, valign, turned = outer_align side in
  if turned then
    let x = match side with `Left -> Box2.maxx cell | _ -> Box2.minx cell in
    place_text cx owner ~halign:`Center ~valign ~turned ~data:false 1. head
      (P2.v x (P2.y (Box2.mid span)))
  else
    let x =
      match align with
      | `Center ->
          (* Centred on the data areas, but within the figure it titles, which
             its track makes at least as wide as itself. *)
          let half = width (set cx ~halign:align ~valign 1. head) /. 2. in
          let lo = Box2.minx outer +. half and hi = Box2.maxx outer -. half in
          Float.max lo (Float.min hi (P2.x (Box2.mid span)))
      | `Left -> Box2.minx outer
      | `Right -> Box2.maxx outer
    in
    let y = match side with `Bottom -> Box2.miny cell | _ -> Box2.maxy cell in
    place_text cx owner ~halign:align ~valign ~turned ~data:false 1. head
      (P2.v x y)

let union h h' =
  match (h, h') with
  | None, h | h, None -> h
  | Some b, Some b' -> Some (Box2.union b b')

(* [place cx acc unit item box ~span ~outer] places [item] in [box] and is the
   hull of its panels, [span] being that of the panels a heading or legend
   stands beside, and [outer] the box with protrusions of their grid. *)
let rec place cx acc unit item box ~span ~outer =
  match item with
  | Leaf l -> Some (place_leaf cx acc l box)
  | Grid g -> place_grid cx acc unit g box
  | Heading { owner; align; head; hside } ->
      acc.titles <-
        place_heading cx ~owner ~align ~head ~side:hside box span outer
        :: acc.titles;
      None
  | Legend ls ->
      place_legend cx acc ls box span;
      None

and place_grid cx acc unit g box =
  let t = solve_grid cx unit g (Box2.w box) (Box2.h box) in
  let starts len gap o =
    let a = Array.make (Array.length len) o in
    for i = 1 to Array.length len - 1 do
      a.(i) <- a.(i - 1) +. len.(i - 1) +. gap.(i - 1)
    done;
    a
  in
  let xs = starts t.col_len t.col_gap (Box2.minx box +. t.x_off) in
  let ys = starts t.row_len t.row_gap (Box2.miny box +. t.y_off) in
  let extent len gap s n =
    let e = ref 0. in
    for i = s to s + n - 1 do
      e := !e +. len.(i)
    done;
    for i = s to s + n - 2 do
      e := !e +. gap.(i)
    done;
    !e
  in
  let cell_box c =
    Box2.v xs.(c.c0) ys.(c.r0)
      (extent t.col_len t.col_gap c.c0 c.nc)
      (extent t.row_len t.row_gap c.r0 c.nr)
  in
  let p = grid_prot cx g in
  let outer =
    Box2.v
      (Box2.minx box -. p.left)
      (Box2.miny box -. p.top)
      (Box2.w box +. p.left +. p.right)
      (Box2.h box +. p.top +. p.bottom)
  in
  let is_body k = Option.equal Int.equal (Some k) g.gbody in
  let body =
    match g.gbody with
    | None -> None
    | Some k ->
        let c = List.nth g.gcells k in
        place cx acc unit c.it (cell_box c) ~span:None ~outer
  in
  let _, hull =
    List.fold_left
      (fun (k, h) c ->
        if is_body k then (k + 1, h)
        else
          ( k + 1,
            union h (place cx acc unit c.it (cell_box c) ~span:body ~outer) ))
      (0, body) g.gcells
  in
  Option.iter (fun h -> acc.spans <- (g.gid, h) :: acc.spans) hull;
  hull

(* Choosing ticks *)

let all_ticks locale (F f) =
  match f.kind with
  | Scale.Categorical ->
      Ticks.of_values ~locale f.scale (Array.of_list (category_names f.scale))
  | Scale.Quantitative | Scale.Temporal -> Ticks.of_values ~locale f.scale [||]

(* [lengths cx acc] is, for each scale, the length in [acc] of each guide that
   shows it: the shortest of its axes along one direction, or the span of the
   panels its legend stands beside. *)
let lengths cx acc =
  Array.mapi
    (fun i us ->
      List.map
        (fun u ->
          match u with
          | Header_of -> Float.nan
          | Axis_of role ->
              List.fold_left
                (fun m (j, r, l) ->
                  if j = i && r = role then Float.min m l else m)
                Float.infinity acc.along
          | Legend_of { side; block; _ } -> (
              match find_path block acc.spans with
              | None -> 0.
              | Some b -> if vertical side then Box2.h b else Box2.w b))
        us)
    cx.uses

(* [choose cx lengths] is the ticks of each scale. A facet scale and a
   categorical one with a legend show every category. Otherwise the ticks are
   chosen once against every guide that shows them, at its length: a label's
   extent is the greatest fraction of a guide's length it takes, so that no two
   labels overlap on any of them. *)
let choose cx lengths =
  let locale = Theme.locale cx.theme in
  let label t = set cx label_em (Text.v t) in
  let clear = em cx clear_em and sw = em cx swatch_em in
  let measure u t =
    let l = label t in
    match u with
    | Axis_of Gx -> width l +. clear
    | Axis_of (Gy | Gfx | Gfy) -> height l +. clear
    | Legend_of { bar; side; _ } -> (
        match (bar, vertical side) with
        | true, true -> height l +. clear
        | true, false -> width l +. clear
        | false, true -> Float.max sw (height l)
        | false, false -> sw +. em cx pad_em +. width l +. clear)
    | Header_of -> 0.
  in
  Array.mapi
    (fun i (F f as s) ->
      let us = cx.uses.(i) in
      let every =
        List.exists
          (function
            | Header_of -> true
            | Legend_of _ -> categorical s
            | Axis_of _ -> false)
          us
      in
      let guides =
        List.filter
          (fun (_, l) -> Float.is_finite l && l > 0.)
          (List.combine us lengths.(i))
      in
      if every then all_ticks locale s
      else
        match guides with
        | [] -> Ticks.of_values ~locale f.scale [||]
        | guides ->
            let measure t = longest (fun (u, l) -> measure u t /. l) guides in
            Ticks.choose ~locale ~length:1. ~measure f.scale)
    cx.scales

(* Laid-out figures *)

type t = {
  resolved : Resolved.t;
  theme : Theme.t;
  page : float * float;
  lpanels : (panel * Coord.t) list;
  frozen : Ticks.t array; (* Per scale of the resolved figure. *)
  axes : axis_out list;
  headers : header_out list;
  legends : legend_out list;
  titles : placed list;
  lwarnings : warning list;
  measures : Text.Layout.t Measures.t;
}

(* [pass cx unit size root] places [root] at [size] and is what it placed and
   the page's size. *)
let pass cx unit size root =
  let acc =
    {
      panels = [];
      axes = [];
      headers = [];
      legends = [];
      titles = [];
      spans = [];
      along = [];
    }
  in
  let p = prot cx root in
  let nw, nh = natural cx unit root in
  let page, cw, ch =
    match size with
    | Size.Panels _ ->
        ((p.left +. nw +. p.right, p.top +. nh +. p.bottom), nw, nh)
    | Size.Figure (w, h) ->
        let cw = w -. p.left -. p.right and ch = h -. p.top -. p.bottom in
        if cx.final && (cw < nw || ch < nh) then begin
          let up x = Float.ceil (x *. 100.) /. 100. in
          err "layout" "the figure needs %g × %g pt, more than %a"
            (up (p.left +. nw +. p.right))
            (up (p.top +. nh +. p.bottom))
            Size.pp size
        end;
        ((w, h), Float.max cw nw, Float.max ch nh)
  in
  let box = Box2.v p.left p.top cw ch in
  ignore (place cx acc unit root box ~span:None ~outer:box);
  (acc, page)

let layout ?prev ?(theme = Theme.default) size (r : Resolved.t) =
  let scales = Array.of_list r.scales in
  let uses = uses_of r (blocks r) in
  (* The root sits in a grid of one flexible cell, which gives a panel at the
     root the data area of [Size.panels]. *)
  let root =
    Grid
      {
        gid = Nx.Ptree.Path.root;
        gcols = [| Flex 1. |];
        grows = [| Flex 1. |];
        gcells =
          [ { r0 = 0; c0 = 0; nr = 1; nc = 1; it = build r scales uses } ];
        gbody = None;
      }
  in
  let reused =
    match prev with
    | Some l when Theme.equal l.theme theme -> l.measures
    | _ -> Measures.empty
  in
  let unit =
    match size with Size.Panels (w, h) -> (w, h) | Size.Figure _ -> (0., 0.)
  in
  let cx =
    {
      theme;
      measures = ref Measures.empty;
      reused;
      scales;
      uses;
      ticks = Array.make (Array.length scales) None;
      lengths = Array.map (List.map (fun _ -> Float.nan)) uses;
      final = false;
      notes = ref [];
    }
  in
  (* Without guides, then with the first choice's; the second is frozen. *)
  let next cx acc =
    let lengths = lengths cx acc in
    { cx with ticks = Array.map Option.some (choose cx lengths); lengths }
  in
  let acc, _ = pass cx unit size root in
  let cx = next cx acc in
  let acc, _ = pass cx unit size root in
  let cx = next cx acc in
  (* Horizontal legends wrap at the lengths of a solve with the frozen ticks,
     which their rows do not change. *)
  let acc, _ = pass cx unit size root in
  let cx = { cx with lengths = lengths cx acc; final = true } in
  let frozen = Array.map Option.get cx.ticks in
  let acc, page = pass cx unit size root in
  {
    resolved = r;
    theme;
    page;
    lpanels = List.rev acc.panels;
    frozen;
    axes = List.rev acc.axes;
    headers = List.rev acc.headers;
    legends = List.rev acc.legends;
    titles = List.rev acc.titles;
    lwarnings = r.warnings @ dedupe (List.rev !(cx.notes));
    measures = !(cx.measures);
  }

(* Observing and comparing *)

let size l = l.page
let panels l = List.map fst l.lpanels
let warnings l = l.lwarnings

let equal_panel (p, c) (p', c') =
  Nx.Ptree.Path.equal p.id p'.id && Box2.equal p.box p'.box && Coord.equal c c'

let equal_axis a a' =
  Nx.Ptree.Path.equal a.ax_id a'.ax_id
  && Nx.Ptree.Path.equal a.ax_panel a'.ax_panel
  && Int.equal a.ax_scale a'.ax_scale
  && equal_side a.ax_side a'.ax_side
  && Bool.equal a.ax_grid a'.ax_grid
  && List.equal equal_placed a.ax_labels a'.ax_labels
  && Option.equal equal_placed a.ax_title a'.ax_title

let equal_header h h' =
  Nx.Ptree.Path.equal h.hd_id h'.hd_id
  && Nx.Ptree.Path.equal h.hd_panel h'.hd_panel
  && equal_placed h.hd_label h'.hd_label

let equal_entry e e' =
  Float.equal e.u e'.u
  && Box2.equal e.swatch e'.swatch
  && equal_placed e.label e'.label

let equal_body b b' =
  match (b, b') with
  | Bar b, Bar b' ->
      Box2.equal b.bar b'.bar && List.equal equal_placed b.labels b'.labels
  | Entries es, Entries es' -> List.equal equal_entry es es'
  | Bar _, Entries _ | Entries _, Bar _ -> false

let equal_legend g g' =
  Nx.Ptree.Path.equal g.lg_id g'.lg_id
  && Int.equal g.lg_scale g'.lg_scale
  && equal_side g.lg_side g'.lg_side
  && Option.equal equal_placed g.lg_title g'.lg_title
  && equal_body g.lg_body g'.lg_body

let equal l l' =
  Resolved.equal l.resolved l'.resolved
  && Theme.equal l.theme l'.theme
  && Float.equal (fst l.page) (fst l'.page)
  && Float.equal (snd l.page) (snd l'.page)
  && List.equal equal_panel l.lpanels l'.lpanels
  && Array.length l.frozen = Array.length l'.frozen
  && Array.for_all2 Ticks.equal l.frozen l'.frozen
  && List.equal equal_axis l.axes l'.axes
  && List.equal equal_header l.headers l'.headers
  && List.equal equal_legend l.legends l'.legends
  && List.equal equal_placed l.titles l'.titles
  && List.equal Resolved.equal_warning l.lwarnings l'.lwarnings

(* Formatting *)

let pp_box ppf b =
  Format.fprintf ppf "[(%g, %g) (%g, %g)]" (Box2.minx b) (Box2.miny b)
    (Box2.maxx b) (Box2.maxy b)

(* A text and its box are on one line, whatever its length. *)
let pp_placed ppf p =
  let b = Buffer.create 64 in
  let line = Format.formatter_of_buffer b in
  Format.pp_set_geometry line ~max_indent:999_999 ~margin:1_000_000;
  Format.fprintf line "%a@?" Text.pp p.text;
  Format.fprintf ppf "%s %a%s" (Buffer.contents b) pp_box (placed_box p)
    (if p.turned then " turned" else "")

let pp_axis ppf a =
  Format.fprintf ppf "@[<v 2>axis %a %a%s" pp_id a.ax_id pp_side a.ax_side
    (if a.ax_grid then " grid" else "");
  List.iter (Format.fprintf ppf "@,label %a" pp_placed) a.ax_labels;
  Option.iter (Format.fprintf ppf "@,title %a" pp_placed) a.ax_title;
  Format.fprintf ppf "@]"

let pp_legend ppf g =
  Format.fprintf ppf "@[<v 2>legend %a %a" pp_id g.lg_id pp_side g.lg_side;
  Option.iter (Format.fprintf ppf "@,title %a" pp_placed) g.lg_title;
  (match g.lg_body with
  | Bar { bar; labels } ->
      Format.fprintf ppf "@,bar %a" pp_box bar;
      List.iter (Format.fprintf ppf "@,label %a" pp_placed) labels
  | Entries es ->
      List.iter
        (fun e ->
          Format.fprintf ppf "@,entry %g %a %a" e.u pp_box e.swatch pp_placed
            e.label)
        es);
  Format.fprintf ppf "@]"

let pp_ticks ppf (F f, t) =
  Format.fprintf ppf "@[<hov 2>%S %a%a@ %a@]" f.sid.sname pp_tag (tag f.kind)
    (Format.pp_print_option (fun ppf p -> Format.fprintf ppf " in %a" pp_id p))
    (Resolved.panel_of f.key) Ticks.pp t

let pp ppf l =
  let w, h = l.page in
  Format.fprintf ppf "@[<v>layout %g × %g" w h;
  List.iter
    (fun (p, c) ->
      Format.fprintf ppf "@,@[<v 2>panel %a %a %a" pp_id p.id pp_box p.box
        Coord.pp c;
      List.iter
        (fun a ->
          if Nx.Ptree.Path.equal a.ax_panel p.id then
            Format.fprintf ppf "@,%a" pp_axis a)
        l.axes;
      List.iter
        (fun hd ->
          if Nx.Ptree.Path.equal hd.hd_panel p.id then
            Format.fprintf ppf "@,header %a %a" pp_id hd.hd_id pp_placed
              hd.hd_label)
        l.headers;
      Format.fprintf ppf "@]")
    l.lpanels;
  List.iter (Format.fprintf ppf "@,%a" pp_legend) l.legends;
  List.iter (Format.fprintf ppf "@,title %a" pp_placed) l.titles;
  Format.fprintf ppf "@,@[<v 2>ticks";
  List.iteri
    (fun i s -> Format.fprintf ppf "@,%a" pp_ticks (s, l.frozen.(i)))
    l.resolved.scales;
  Format.fprintf ppf "@]";
  if l.lwarnings <> [] then begin
    Format.fprintf ppf "@,@[<v 2>warnings";
    List.iter (Format.fprintf ppf "@,%a" pp_warning) l.lwarnings;
    Format.fprintf ppf "@]"
  end;
  Format.fprintf ppf "@]"
