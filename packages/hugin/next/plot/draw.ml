(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Drawing

   [draw] paints a laid-out figure: the paper, then each panel's marks over its
   grid lines, clipped to its data area, then the axes, headers, legends and
   titles. A mark is drawn one cell at a time: its facet channels put each of
   its rows in a panel, and its draw function draws each panel's rows. *)

module P2 = Hugin_next_gg.P2
module Box2 = Hugin_next_gg.Box2
module Affine = Hugin_next_gg.Affine
module Path = Hugin_next_gg.Path
module Stroke = Hugin_next_gg.Stroke
module Color = Hugin_next_gg.Color
module Text = Hugin_next_text.Text
module Picture = Hugin_next_vg.Picture
module Renderable = Hugin_next_vg.Renderable
module Scale = Hugin_next_kit.Scale
module Ticks = Hugin_next_kit.Ticks
open Common
open Channel
open Figure
open Arrange
open Resolved

(* Derived lengths, in em *)

let rule_em = 0.08 (* Axis lines and ticks. *)
let grid_em = 0.06
let grid_alpha = 0.2

(* Panels *)

type key = {
  kid : id;
  kbox : Box2.t;
  kcoord : Coord.t;
  kmarks : (id * mark) list;
  kscales : (fitted * Ticks.t) list;
  ktheme : Theme.t;
  kdensity : float;
}

let equal_fitted (F f) (F f') =
  String.equal f.sid.sname f'.sid.sname
  &&
  match Scale.equal_kind f.kind f'.kind with
  | Some Type.Equal -> Scale.equal f.scale f'.scale
  | None -> false

let equal_key k k' =
  Nx.Ptree.Path.equal k.kid k'.kid
  && Box2.equal k.kbox k'.kbox
  && Coord.equal k.kcoord k'.kcoord
  && List.equal
       (fun (id, m) (id', m') -> Nx.Ptree.Path.equal id id' && equal_mark m m')
       k.kmarks k'.kmarks
  && List.equal
       (fun (s, t) (s', t') -> equal_fitted s s' && Ticks.equal t t')
       k.kscales k'.kscales
  && Theme.equal k.ktheme k'.ktheme
  && Float.equal k.kdensity k'.kdensity

type drawn = { key : key; picture : Picture.t; notes : warning list }

type t = {
  renderable : Renderable.t;
  warnings : warning list;
  drawn : drawn list;
}

(* Context *)

type cx = { ctx : Read.ctx; layout : Layout.t; resolved : Resolved.t }

let em cx k = k *. Theme.size cx.ctx.theme
let ink cx = Theme.ink cx.ctx.theme

(* [scale_of cx occ pid pnid] is the index of the scale each binding of [occ]
   reads in the panel [pnid] of the cell [pid]. *)
let scale_of cx occ pid pnid =
  let scales = cx.ctx.scales in
  let found = Array.make (List.length occ.mark.bindings) None in
  Array.iteri
    (fun i (F f) ->
      let here =
        match f.key with
        | Panel (_, p) -> Nx.Ptree.Path.equal p pnid
        | _ -> true
      in
      if here then
        List.iter
          (fun m ->
            if
              Nx.Ptree.Path.equal m.m_occ.mid occ.mid
              && Nx.Ptree.Path.equal m.m_pid pid
              && Option.is_none found.(m.m_index)
            then found.(m.m_index) <- Some i)
          f.members)
    scales;
  fun index -> found.(index)

(* [reading mid f] is [f ()], which reads the tensors of the mark [mid], with
   the errors of reading them naming the mark. *)
let reading mid f =
  try f () with Invalid_argument msg -> err "draw" "%a: %s" pp_id mid msg

(* Membership *)

let rec const_value : type d r. (d, r) Channel.t -> r option = function
  | Const v -> Some v
  | Map (f, c) -> Option.map f (const_value c)
  | Data _ -> None

let facet_const m role : string option =
  match find_binding role m.bindings with
  | None -> None
  | Some (B b) -> (
      match Role.equal_range b.role.range Role.Panels with
      | Some Type.Equal -> (const_value b.ch : string option)
      | None -> None)

(* [members rd m fps] is the rows of [m] in each facet panel of [fps], [None]
   where it has none. Rows are put in panels by the identities of their facet
   categories, each category's name read once. *)
let members rd m fps =
  let gate p =
    let ok role cat =
      match facet_const m role with
      | None -> true
      | Some v -> Option.equal String.equal (Some v) cat
    in
    ok "fx" p.pfx && ok "fy" p.pfy
  in
  match (Read.facet rd "fx", Read.facet rd "fy") with
  | None, None -> List.map (fun p -> if gate p then Some Read.All else None) fps
  | fx, fy ->
      let n = Array.fold_left ( * ) 1 m.shape in
      (* [slots f] numbers the categories of the facet [f] in order of first
         row, and is the slot of each row, [-1] where it is missing. *)
      let slots = function
        | None -> (Array.make n 0, [| None |])
        | Some (ids, name) ->
            let names = ref [] and count = ref 0 in
            let number id =
              names := Some (name id) :: !names;
              incr count;
              !count - 1
            in
            let miss = Array.map (fun id -> id = min_int) ids in
            let slot = Read.memo miss ids number in
            let s =
              Array.mapi (fun j id -> if miss.(j) then -1 else slot id) ids
            in
            (s, Array.of_list (List.rev !names))
      in
      let sx, nx = slots fx and sy, ny = slots fy in
      let width = Array.length nx in
      let count = Array.make (width * Array.length ny) 0 in
      let slot i =
        if sx.(i) < 0 || sy.(i) < 0 then -1 else (sy.(i) * width) + sx.(i)
      in
      for i = 0 to n - 1 do
        let k = slot i in
        if k >= 0 then count.(k) <- count.(k) + 1
      done;
      let rows = Array.map (fun c -> Array.make c 0) count in
      let fill = Array.make (Array.length count) 0 in
      for i = 0 to n - 1 do
        let k = slot i in
        if k >= 0 then begin
          rows.(k).(fill.(k)) <- i;
          fill.(k) <- fill.(k) + 1
        end
      done;
      (* A facet the mark leaves unbound matches every panel. *)
      let find names f cat =
        match f with
        | None -> Some 0
        | Some _ ->
            let rec go k =
              if k >= Array.length names then None
              else if Option.equal String.equal names.(k) cat then Some k
              else go (k + 1)
            in
            go 0
      in
      List.map
        (fun p ->
          match (find nx fx p.pfx, find ny fy p.pfy) with
          | Some x, Some y
            when gate p && Array.length rows.((y * width) + x) > 0 ->
              Some (Read.Rows rows.((y * width) + x))
          | _ -> None)
        fps

(* Tagging *)

let clip_to box p = Picture.clip (Path.rect box) p

(* [tagged id index box p] is [p] clipped to [box] and tagged with [id] and the
   rows [index], instance by instance if it is a stamp of one instance per
   row. *)
let tagged id index box p =
  let tag = { Picture.id; rows = Picture.Rows index } in
  match p with
  | Picture.Empty -> Picture.empty
  | Picture.Stamp { xs; _ } when Array.length xs = Array.length index ->
      clip_to box (Picture.tag tag p)
  | _ -> Picture.tag tag (clip_to box p)

(* Images *)

let byte v = Float.to_int (Float.round (255. *. Float.min 1. (Float.max 0. v)))

(* [image h w at] is the RGBA image whose pixel [(i, j)] is [at i j]. *)
let image h w at =
  let a = Array.make (h * w * 4) 0 in
  for i = 0 to h - 1 do
    for j = 0 to w - 1 do
      let c = at i j and o = ((i * w) + j) * 4 in
      a.(o) <- byte (Color.r c);
      a.(o + 1) <- byte (Color.g c);
      a.(o + 2) <- byte (Color.b c);
      a.(o + 3) <- byte (Color.alpha c)
    done
  done;
  Nx.create Nx.uint8 [| h; w; 4 |] a

(* Marks *)

(* A panel being drawn: its facet panel, layout and the pictures and warnings of
   its marks so far, latest first. *)
type target = {
  fp : facet_panel;
  panel : Layout.panel;
  mutable pictures : Picture.t list;
  mutable notes : warning list;
}

(* [draw_occ cx pid occ targets] draws [occ], a mark of the cell [pid], in each
   panel of [targets] it has rows in. *)
let draw_occ cx pid occ targets =
  let m = occ.mark in
  let rd = Read.reader ~whole:true m in
  let sels =
    reading occ.mid (fun () -> members rd m (List.map (fun t -> t.fp) targets))
  in
  let draw t sel =
    let warn msg = t.notes <- (occ.mid, msg) :: t.notes in
    let scale_of = scale_of cx occ pid t.fp.pnid in
    let r =
      reading occ.mid (fun () ->
          Read.rows cx.ctx rd ~id:occ.mid t.panel.projection ~warn scale_of sel)
    in
    tagged occ.mid r.index t.panel.box (m.draw r)
  in
  List.iter2
    (fun t sel ->
      match sel with
      | None -> ()
      | Some sel -> t.pictures <- draw t sel :: t.pictures)
    targets sels

(* Panels *)

(* [key cx pid content fp panel coord] is the reuse key of a panel. *)
let key cx pid content fp (panel : Layout.panel) coord =
  let frozen = Layout.frozen cx.layout in
  let scales =
    List.concat_map
      (fun occ ->
        let at = scale_of cx occ pid fp.pnid in
        List.filter_map
          (fun i ->
            Option.map (fun s -> (cx.ctx.scales.(s), frozen.(s))) (at i))
          (List.init (List.length occ.mark.bindings) Fun.id))
      content.occs
  in
  {
    kid = panel.id;
    kbox = panel.box;
    kcoord = coord;
    kmarks = List.map (fun o -> (o.mid, o.mark)) content.occs;
    kscales = scales;
    ktheme = cx.ctx.theme;
    kdensity = cx.ctx.density;
  }

(* [panels cx prev] is the drawing of each panel of the layout, in its order,
   reused from [prev] where the key is equal. *)
let panels cx prev =
  let r = cx.resolved in
  let coords = Layout.coords cx.layout in
  let laid id =
    List.find_opt
      (fun ((p : Layout.panel), _) -> Nx.Ptree.Path.equal p.id id)
      coords
  in
  let reused = match prev with None -> [] | Some d -> d.drawn in
  let drawn =
    List.concat_map
      (fun (pid, content) ->
        let fps = Option.value ~default:[] (find_path pid r.facets) in
        let found =
          List.filter_map
            (fun fp ->
              Option.map
                (fun (panel, coord) ->
                  (fp, panel, coord, key cx pid content fp panel coord))
                (laid fp.pnid))
            fps
        in
        let old =
          List.map
            (fun (_, _, _, k) ->
              List.find_opt (fun d -> equal_key d.key k) reused)
            found
        in
        let targets =
          List.filter_map
            (fun ((fp, panel, _, _), o) ->
              match o with
              | Some _ -> None
              | None -> Some { fp; panel; pictures = []; notes = [] })
            (List.combine found old)
        in
        (match targets with
        | [] -> ()
        | _ :: _ ->
            List.iter (fun occ -> draw_occ cx pid occ targets) content.occs);
        List.map2
          (fun (fp, (panel : Layout.panel), _, k) o ->
            match o with
            | Some d -> d
            | None ->
                let t =
                  List.find
                    (fun t -> Nx.Ptree.Path.equal t.fp.pnid fp.pnid)
                    targets
                in
                let picture =
                  Picture.tag
                    { Picture.id = panel.id; rows = Picture.Rows [||] }
                    (Picture.group (List.rev t.pictures))
                in
                { key = k; picture; notes = List.rev t.notes })
          found old)
      (Arrange.panels Nx.Ptree.Path.root r.shaped)
  in
  List.filter_map
    (fun ((p : Layout.panel), _) ->
      List.find_opt (fun d -> Nx.Ptree.Path.equal d.key.kid p.id) drawn)
    coords

(* Texts *)

(* A quarter turn counterclockwise on the page: (u, v) to (v, -u). *)
let quarter = { Affine.xx = 0.; yx = -1.; xy = 1.; yy = 0.; x0 = 0.; y0 = 0. }

let placed cx (p : Layout.placed) =
  let origin = if p.turned then P2.v 0. 0. else p.at in
  let run acc colour o run =
    let c = Option.value colour ~default:(ink cx) in
    Picture.glyphs c (P2.v (P2.x origin +. P2.x o) (P2.y origin +. P2.y o)) run
    :: acc
  in
  let glyphs = Picture.group (List.rev (Text.Layout.fold run [] p.set)) in
  if not p.turned then glyphs
  else
    Picture.transform
      Affine.(translate (P2.x p.at) (P2.y p.at) * quarter)
      glyphs

let tag_node id p = Picture.tag { Picture.id; rows = Picture.Rows [||] } p

(* Axes *)

let ticks_of cx s =
  List.map
    (fun (t : Ticks.tick) -> t.position)
    (Layout.frozen cx.layout).(s).major

let panel_of cx id =
  List.find
    (fun ((p : Layout.panel), _) -> Nx.Ptree.Path.equal p.id id)
    (Layout.coords cx.layout)
  |> fst

let segments pairs =
  List.fold_left
    (fun p ((x0, y0), (x1, y1)) ->
      Path.line_to (P2.v x1 y1) (Path.move_to (P2.v x0 y0) p))
    Path.empty pairs

let draw_axis cx (a : Layout.axis_out) =
  let panel = panel_of cx a.ax_panel in
  let b = panel.box and proj = panel.projection in
  let tick = em cx Layout.tick_em and o = a.ax_offset in
  let along u =
    match a.ax_side with
    | `Top | `Bottom -> P2.x (Coord.point proj u 0.)
    | `Left | `Right -> P2.y (Coord.point proj 0. u)
  in
  let lines =
    let us = ticks_of cx a.ax_scale in
    match a.ax_side with
    | `Bottom ->
        let y = Box2.maxy b +. o in
        ((Box2.minx b, y), (Box2.maxx b, y))
        :: List.map (fun u -> ((along u, y), (along u, y +. tick))) us
    | `Top ->
        let y = Box2.miny b -. o in
        ((Box2.minx b, y), (Box2.maxx b, y))
        :: List.map (fun u -> ((along u, y), (along u, y -. tick))) us
    | `Left ->
        let x = Box2.minx b -. o in
        ((x, Box2.miny b), (x, Box2.maxy b))
        :: List.map (fun u -> ((x, along u), (x -. tick, along u))) us
    | `Right ->
        let x = Box2.maxx b +. o in
        ((x, Box2.miny b), (x, Box2.maxy b))
        :: List.map (fun u -> ((x, along u), (x +. tick, along u))) us
  in
  let pen = Stroke.v ~cap:`Butt (em cx rule_em) in
  tag_node a.ax_id
    (Picture.group
       (Picture.stroke pen (ink cx) (segments lines)
        :: List.map (placed cx) a.ax_labels
       @ Option.to_list (Option.map (placed cx) a.ax_title)))

let draw_grid cx (a : Layout.axis_out) =
  if not a.ax_grid then Picture.empty
  else
    let panel = panel_of cx a.ax_panel in
    let b = panel.box and proj = panel.projection in
    let lines =
      List.map
        (fun u ->
          match a.ax_side with
          | `Top | `Bottom ->
              let x = P2.x (Coord.point proj u 0.) in
              ((x, Box2.miny b), (x, Box2.maxy b))
          | `Left | `Right ->
              let y = P2.y (Coord.point proj 0. u) in
              ((Box2.minx b, y), (Box2.maxx b, y)))
        (ticks_of cx a.ax_scale)
    in
    let c = Color.with_alpha (Color.alpha (ink cx) *. grid_alpha) (ink cx) in
    tag_node a.ax_id
      (Picture.stroke (Stroke.v ~cap:`Butt (em cx grid_em)) c (segments lines))

(* Legends *)

let bar_steps = 256

(* [readers cx s] is each mark that reads the scale [s] through a role with a
   legend, with the indices of its bindings that read it, in the order of the
   figure. *)
let readers cx s =
  let (F f) = cx.ctx.scales.(s) in
  let add acc m =
    let same (o, _) = Nx.Ptree.Path.equal o.mid m.m_occ.mid in
    if Arrange.positional (axis_role m.m_role) then acc
    else if List.exists same acc then
      List.map
        (fun (o, is) -> if same (o, is) then (o, m.m_index :: is) else (o, is))
        acc
    else (m.m_occ, [ m.m_index ]) :: acc
  in
  List.rev (List.fold_left add [] (by_order f.members))

let draw_legend cx notes (g : Layout.legend_out) =
  let warn msg = notes := (g.lg_id, msg) :: !notes in
  let body =
    match g.lg_body with
    | Bar { bar; labels } ->
        let colour = Read.colors cx.ctx cx.ctx.scales.(g.lg_scale) in
        let at k = colour ((float k +. 0.5) /. float bar_steps) in
        let vertical =
          match g.lg_side with
          | `Left | `Right -> true
          | `Top | `Bottom -> false
        in
        let px =
          if vertical then image bar_steps 1 (fun i _ -> at (bar_steps - 1 - i))
          else image 1 bar_steps (fun _ j -> at j)
        in
        let tick = em cx Layout.tick_em in
        let lines =
          List.map
            (fun u ->
              if vertical then
                let y = Box2.maxy bar -. (u *. Box2.h bar) in
                ((Box2.maxx bar, y), (Box2.maxx bar +. tick, y))
              else
                let x = Box2.minx bar +. (u *. Box2.w bar) in
                ((x, Box2.maxy bar), (x, Box2.maxy bar +. tick)))
            (ticks_of cx g.lg_scale)
        in
        Picture.image bar px
        :: Picture.stroke
             (Stroke.v ~cap:`Butt (em cx rule_em))
             (ink cx) (segments lines)
        :: List.map (placed cx) labels
    | Entries es ->
        let n = List.length es in
        let marks = readers cx g.lg_scale in
        List.concat
          (List.mapi
             (fun k (e : Layout.legend_entry) ->
               let proj = Coord.project (Coord.cartesian ()) e.swatch in
               let swatch (occ, is) =
                 let rows =
                   Read.swatch cx.ctx occ.mark ~id:g.lg_id proj ~warn
                     ~scale:g.lg_scale
                     ~reads:(fun i -> List.mem i is)
                     ~n ~k e.u
                 in
                 let draw =
                   Option.value occ.mark.swatch ~default:occ.mark.draw
                 in
                 draw rows
               in
               [
                 Picture.tag
                   { Picture.id = g.lg_id; rows = Picture.Rows [| k |] }
                   (Picture.group (List.map swatch marks));
                 placed cx e.label;
               ])
             es)
  in
  tag_node g.lg_id
    (Picture.group (Option.to_list (Option.map (placed cx) g.lg_title) @ body))

(* Drawing *)

let draw ?prev ~density l =
  if not (is_pos density) then
    err "draw" "density %g is not finite and positive" density;
  let r = Layout.resolved l in
  let ctx =
    {
      Read.theme = Layout.theme l;
      density;
      scales = Array.of_list r.scales;
      frozen = Layout.frozen l;
    }
  in
  let cx = { ctx; layout = l; resolved = r } in
  let drawn = panels cx prev in
  let notes = ref [] in
  let legends = List.map (draw_legend cx notes) (Layout.legends l) in
  let w, h = Layout.size l in
  let paper = Theme.paper ctx.theme in
  let paper =
    if Color.alpha paper = 0. then Picture.empty
    else Picture.fill paper (Path.rect (Box2.v 0. 0. w h))
  in
  let axes = Layout.axes l in
  let picture =
    Picture.group
      ([ paper ]
      @ List.map (draw_grid cx) axes
      @ List.map (fun d -> d.picture) drawn
      @ List.map (draw_axis cx) axes
      @ List.map
          (fun (hd : Layout.header_out) ->
            tag_node hd.hd_id (placed cx hd.hd_label))
          (Layout.headers l)
      @ legends
      @ List.map (placed cx) (Layout.titles l))
  in
  let warnings =
    dedupe
      (Layout.warnings l
      @ List.concat_map (fun (d : drawn) -> d.notes) drawn
      @ List.rev !notes)
  in
  { renderable = Renderable.v w h picture; warnings; drawn }

(* Comparing and formatting *)

let renderable d = d.renderable
let warnings d = d.warnings

let equal_rows (r : Picture.rows) (r' : Picture.rows) =
  match (r, r') with
  | Rows a, Rows a' -> Array.equal Int.equal a a'
  | Cells c, Cells c' ->
      Box2.equal c.box c'.box && c.width = c'.width && c.height = c'.height
  | Rows _, Cells _ | Cells _, Rows _ -> false

let equal_rule (r : Picture.rule) (r' : Picture.rule) =
  match (r, r') with
  | `Nonzero, `Nonzero | `Even_odd, `Even_odd -> true
  | `Nonzero, `Even_odd | `Even_odd, `Nonzero -> false

let floats = Array.equal Float.equal

(* Pictures compare as [Picture.equal] compares them, except image tensors, by
   shape and elements. *)
let rec equal_picture (p : Picture.t) (q : Picture.t) =
  match (p, q) with
  | Image a, Image b ->
      Box2.equal a.box b.box
      && Nx.shape a.pixels = Nx.shape b.pixels
      && Array.equal Int.equal (Nx.to_array a.pixels) (Nx.to_array b.pixels)
  | Group ps, Group qs -> List.equal equal_picture ps qs
  | Clip a, Clip b ->
      equal_rule a.rule b.rule && Path.equal a.path b.path
      && equal_picture a.picture b.picture
  | Transform a, Transform b ->
      Affine.equal a.m b.m && equal_picture a.picture b.picture
  | Opacity a, Opacity b ->
      Float.equal a.opacity b.opacity && equal_picture a.picture b.picture
  | Stamp a, Stamp b ->
      floats a.xs b.xs && floats a.ys b.ys
      && Option.equal floats a.scales b.scales
      && Option.equal (Array.equal Color.equal) a.fills b.fills
      && Option.equal (Array.equal Color.equal) a.strokes b.strokes
      && equal_picture a.picture b.picture
  | Tag a, Tag b ->
      Nx.Ptree.Path.equal a.tag.id b.tag.id
      && equal_rows a.tag.rows b.tag.rows
      && equal_picture a.picture b.picture
  | (Empty | Fill _ | Stroke _ | Glyphs _), _ -> Picture.equal p q
  | (Image _ | Group _ | Clip _ | Transform _ | Opacity _ | Stamp _ | Tag _), _
    ->
      false

let equal d d' =
  List.equal equal_warning d.warnings d'.warnings
  && Float.equal (Renderable.w d.renderable) (Renderable.w d'.renderable)
  && Float.equal (Renderable.h d.renderable) (Renderable.h d'.renderable)
  && equal_picture
       (Renderable.picture d.renderable)
       (Renderable.picture d'.renderable)

let pp ppf d =
  Format.fprintf ppf "@[<v>%a" Renderable.pp d.renderable;
  if d.warnings <> [] then begin
    Format.fprintf ppf "@,@[<v 2>warnings";
    List.iter (Format.fprintf ppf "@,%a" pp_warning) d.warnings;
    Format.fprintf ppf "@]"
  end;
  Format.fprintf ppf "@]"
