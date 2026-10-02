(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Drawing

   [draw] paints a laid-out figure: the paper, then each panel's marks over its
   grid lines, then the axes, headers, legends and titles. Marks clip their
   positions to the panel's domain, not their ink, so the stage clips nothing. A
   mark is drawn one cell at a time: its facet channels put each of its rows in
   a panel, and in each panel its reducer may draw the rows in its stead,
   reading only what it draws; its draw function draws the others. *)

module P2 = Hugin_next_gg.P2
module Box2 = Hugin_next_gg.Box2
module Affine = Hugin_next_gg.Affine
module Path = Hugin_next_gg.Path
module Stroke = Hugin_next_gg.Stroke
module Color = Hugin_next_gg.Color
module Text = Hugin_next_text.Text
module Picture = Hugin_next_vg.Picture
module Renderable = Hugin_next_vg.Renderable
module Raster = Hugin_next_vg_raster
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

(* Opacities of the ink *)

let label_alpha = 0.75 (* Tick and legend labels. *)
let rule_alpha = 0.6 (* Axis lines and ticks. *)
let grid_alpha = 0.2

(* Reducer thresholds *)

let raster_rows = 20_000
let m4_rows = 4 (* Rows per device-pixel column. *)

(* Panels *)

type t = {
  renderable : Renderable.t;
  warnings : warning list;
  layout : Layout.t;
  density : float;
}

(* Context *)

type cx = { ctx : Read.ctx; layout : Layout.t; resolved : Resolved.t }

let em cx k = k *. Theme.size cx.ctx.theme
let ink cx = Theme.ink cx.ctx.theme

(* [faded cx a] is the ink at [a] times its opacity. *)
let faded cx a =
  let ink = ink cx in
  Color.with_alpha (Color.alpha ink *. a) ink

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

let facet_const m role : string option =
  match find_binding role m.bindings with
  | None -> None
  | Some (B b) -> (
      match Role.equal_range b.role.range Role.Panels with
      | Some Type.Equal -> (constant b.ch : string option)
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
    ok Role.fx p.pfx && ok Role.fy p.pfy
  in
  match (Read.facet rd Role.fx, Read.facet rd Role.fy) with
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

(* [tagged id index p] is [p] tagged with [id] and the rows [index], instance by
   instance if it is a stamp of one instance per row. *)
let tagged id index p =
  let tag = { Picture.id; rows = Picture.Rows index } in
  match p with Picture.Empty -> Picture.empty | _ -> Picture.tag tag p

(* Reducers *)

let device_pixels cx box =
  let d = cx.ctx.density in
  Box2.w box *. d *. Box2.h box *. d

(* [aligned cx box] is the smallest box of whole device pixels holding [box]. *)
let aligned cx box =
  let d = cx.ctx.density in
  let lo v = Float.floor (v *. d) /. d and hi v = Float.ceil (v *. d) /. d in
  let x0 = lo (Box2.minx box) and y0 = lo (Box2.miny box) in
  Box2.v x0 y0 (hi (Box2.maxx box) -. x0) (hi (Box2.maxy box) -. y0)

(* [rasterised cx p] is [p] as one image painted by the raster renderer at the
   density, over the device pixels its bounds reach. *)
let rasterised cx p =
  match Picture.bounds p with
  | None -> Picture.empty
  | Some b ->
      let window = aligned cx b in
      let w = Box2.w window and h = Box2.h window in
      if w <= 0. || h <= 0. then Picture.empty
      else
        let moved =
          Picture.transform
            (Affine.translate (-.Box2.minx window) (-.Box2.miny window))
            p
        in
        let px =
          Raster.render ~density:cx.ctx.density (Renderable.v w h moved)
        in
        Picture.image window px

(* [band_cells cx scale_of index] is the number of categories of the band scale
   that the binding [index] reads, if it has no padding. *)
let band_cells cx scale_of index =
  match scale_of index with
  | None -> None
  | Some i -> (
      let (F f) = cx.ctx.scales.(i) in
      match f.kind with
      | Categories ->
          let n = List.length (category_names f.scale) in
          if n > 0 && Float.equal (Scale.bandwidth f.scale) (1. /. float n) then
            Some n
          else None
      | Quantities -> None)

let binding_index m (role : _ Role.t) =
  let rec go i = function
    | [] -> None
    | B b :: rest ->
        if String.equal b.role.name role.name then Some i else go (i + 1) rest
  in
  go 0 m.bindings

let binds m role = Option.is_some (find_binding role m.bindings)
let row_of = function Read.All -> Fun.id | Read.Rows r -> fun k -> r.(k)

(* [colours cx rows] is the colour each of [rows] paints its cell with:
   transparent where it is dropped. *)
let colours cx (rows : Rows.t) =
  let n = Rows.length rows in
  let fills =
    match Rows.get rows Role.fill with
    | Some cs -> cs
    | None -> Array.make n (Theme.accent cx.ctx.theme)
  in
  let os = Rows.get rows Role.opacity in
  Array.mapi
    (fun i c ->
      if rows.Rows.dropped.(i) then Color.transparent
      else
        match os with
        | None -> c
        | Some os -> Color.with_alpha (Color.alpha c *. os.(i)) c)
    fills

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

(* [cells cx m panel scale_of ~rows ~full ~few sel] is the image whose pixels
   are the cells of the rows [sel], if they draw as one: [x] and [y] read band
   scales without padding, the mark binds no [x2], [y2] or [stroke], and no two
   rows share a cell. Past 4 × 4 cells per device pixel, only the rows of the
   cells that raster output samples are read. *)
let cells cx m (panel : Layout.panel) scale_of ~rows ~full ~few sel =
  match (binding_index m Role.x, binding_index m Role.y) with
  | Some xi, Some yi
    when not (binds m Role.x2 || binds m Role.y2 || binds m Role.stroke) -> (
      match (band_cells cx scale_of xi, band_cells cx scale_of yi) with
      | Some nx, Some ny -> (
          let pos = rows (Some [ Role.x.name; Role.y.name ]) full sel in
          let paint = Some [ Role.fill.name; Role.opacity.name ] in
          let us = Option.get (Rows.normalized pos Role.x)
          and vs = Option.get (Rows.normalized pos Role.y) in
          let cell = Array.make (nx * ny) (-1) and shared = ref false in
          let clamp n v = Int.max 0 (Int.min (n - 1) v) in
          Array.iteri
            (fun k u ->
              let v = vs.(k) in
              if
                (not pos.Rows.dropped.(k))
                && Float.is_finite u && Float.is_finite v
              then begin
                let j = clamp nx (Float.to_int (Float.floor (u *. float nx)))
                and i =
                  clamp ny (Float.to_int (Float.floor ((1. -. v) *. float ny)))
                in
                let c = (i * nx) + j in
                if cell.(c) >= 0 then shared := true else cell.(c) <- k
              end)
            us;
          if !shared then None
          else
            let proj = panel.projection in
            let box =
              Box2.of_pts (Coord.point proj 0. 1.) (Coord.point proj 1. 0.)
            in
            let tag =
              {
                Picture.id = pos.Rows.id;
                rows = Picture.Cells { box; width = nx; height = ny };
              }
            in
            let at_cell cs i j =
              let k = cell.((i * nx) + j) in
              if k < 0 then Color.transparent else cs k
            in
            match Pixels.plan ~density:cx.ctx.density box ~rows:ny ~cols:nx with
            | None ->
                let cs = colours cx (rows paint full sel) in
                let px = image ny nx (at_cell (fun k -> cs.(k))) in
                Some (Picture.tag tag (Picture.image box px))
            | Some plan ->
                (* The rows of the sampled cells, read where they live. *)
                let wanted = Hashtbl.create 1024 in
                Array.iter
                  (fun i ->
                    if i >= 0 then
                      Array.iter
                        (fun j ->
                          if j >= 0 then
                            let k = cell.((i * nx) + j) in
                            if k >= 0 then Hashtbl.replace wanted k ())
                        plan.cols)
                  plan.rows;
                let ks =
                  Array.of_list
                    (List.sort Int.compare
                       (Hashtbl.fold (fun k () acc -> k :: acc) wanted []))
                in
                let row = row_of sel in
                let cs =
                  colours cx (rows paint few (Read.Rows (Array.map row ks)))
                in
                let at = Hashtbl.create (Array.length ks) in
                Array.iteri (fun p k -> Hashtbl.replace at k cs.(p)) ks;
                let colour i j =
                  let i = plan.rows.(i) and j = plan.cols.(j) in
                  if i < 0 || j < 0 then Color.transparent
                  else at_cell (Hashtbl.find at) i j
                in
                let px =
                  image (Array.length plan.rows) (Array.length plan.cols) colour
                in
                Some (Picture.tag tag (Picture.image plan.window px)))
      | _ -> None)
  | _ -> None

(* [quantities m index] is the quantities the binding [index] reads, as a tensor
   or an axis index, if no element of them is masked. *)
let quantities m index =
  let (B b) = List.nth m.bindings index in
  match data b.ch with
  | Some { lift = Num { x; valid = None }; _ } ->
      Some (`Num (Nx.cast Nx.float64 x))
  | Some { lift = Index k; _ } -> Some (`Index k)
  | _ -> None

(* [m4 cx m panel scale_of] is the rows of [m] that M4 keeps in [panel], if it
   applies: of each series, each device-pixel column's first, last, lowest and
   highest rows, when the series have more than [m4_rows] rows per column, their
   [x] is monotone and their other channels constant along them, and no value of
   [x] or [y] is missing. Columns are found where the data lives, as the bins
   that the values of [x] at the columns' edges make, with one more bin on each
   side for the rows outside the panel. *)
let m4 cx m (panel : Layout.panel) scale_of =
  let shape = m.shape in
  let rank = Array.length shape in
  let w = Float.to_int (Float.ceil (Box2.w panel.box *. cx.ctx.density)) in
  let last = if rank = 0 then 0 else shape.(rank - 1) in
  let constant_along (B b) =
    match (b.role.use, data b.ch) with
    | Position { far = false; _ }, _ | _, None -> true
    | _, Some d -> (
        match d.lift with
        | Index k | Dim { axis = k; _ } -> axis_of shape k <> Some (rank - 1)
        | Num _ | Cat _ | Strings _ | Scalar _ -> (
            match lift_shape d.lift with
            | Some s when Array.length s > 0 -> s.(Array.length s - 1) = 1
            | _ -> true))
  in
  let float_scale index : float Scale.t option =
    Option.bind (scale_of index) (fun i : float Scale.t option ->
        let (F f) = cx.ctx.scales.(i) in
        match f.kind with Quantities -> Some f.scale | Categories -> None)
  in
  let series = if last = 0 then 0 else Array.fold_left ( * ) 1 shape / last in
  let flat t = Nx.reshape [| series; last |] (Nx.broadcast_to shape t) in
  let missing s t = Nx.item [] (Nx.any (Scale.missing s t)) in
  let monotone t =
    let d =
      Nx.sub (Nx.slice [ A; R (1, last) ] t) (Nx.slice [ A; R (0, last - 1) ] t)
    in
    let all p = Nx.all ~axes:[ 1 ] p in
    Nx.item []
      (Nx.all
         (Nx.logical_or
            (all (Nx.greater_equal_s d 0.))
            (all (Nx.less_equal_s d 0.))))
  in
  let applies =
    rank > 0 && w > 0
    && last > m4_rows * w
    && List.for_all constant_along m.bindings
  in
  let xi = binding_index m Role.x and yi = binding_index m Role.y in
  match (applies, xi, yi) with
  | true, Some xi, Some yi -> (
      match
        (float_scale xi, float_scale yi, quantities m xi, quantities m yi)
      with
      | Some sx, Some sy, Some qx, Some (`Num y) -> (
          let yt = flat y in
          let xt =
            match qx with
            | `Index k when axis_of shape k = Some (rank - 1) ->
                Some (flat (Nx.cast Nx.float64 (Nx.arange Nx.int32 0 last 1)))
            | `Index _ -> None
            | `Num x ->
                let xt = flat x in
                if missing sx xt || not (monotone xt) then None else Some xt
          in
          let edges =
            List.init (w + 1) (fun c -> Scale.invert sx (float c /. float w))
          in
          match xt with
          | Some xt
            when (not (missing sy yt)) && List.for_all Option.is_some edges ->
              let edges =
                Array.of_list
                  (List.sort Float.compare (List.map Option.get edges))
              in
              let bins = w + 2 in
              let col =
                Nx.searchsorted ~side:`Right
                  (Nx.create Nx.float64 [| w + 1 |] edges)
                  xt
              in
              let base =
                Nx.mul_s
                  (Nx.reshape [| series; 1 |] (Nx.arange Nx.int64 0 series 1))
                  (Int64.of_int bins)
              in
              let key = Nx.flatten (Nx.add col base) in
              let j =
                Nx.flatten
                  (Nx.broadcast_to [| series; last |]
                     (Nx.reshape [| 1; last |] (Nx.arange Nx.int64 0 last 1)))
              in
              let yv = Nx.flatten yt and k = series * bins in
              let scatter mode values init =
                Nx.scatter ~mode ~axis:0 ~indices:key ~values init
              in
              let none = Nx.full Nx.int64 [| k |] Int64.max_int in
              let first = scatter `Min j none
              and final = scatter `Max j (Nx.full Nx.int64 [| k |] (-1L)) in
              let at extreme init =
                let e = scatter extreme yv (Nx.full Nx.float64 [| k |] init) in
                let hit = Nx.equal yv (Nx.take ~indices:key e) in
                scatter `Min
                  (Nx.where hit j (Nx.full_like j Int64.max_int))
                  none
              in
              let high = at `Max Float.neg_infinity
              and low = at `Min Float.infinity in
              let kept = ref [] in
              List.iter
                (fun t ->
                  Array.iteri
                    (fun s v ->
                      if v >= 0L && v < Int64.max_int then
                        kept := ((s / bins * last) + Int64.to_int v) :: !kept)
                    (Nx.to_array t))
                [ first; final; high; low ];
              Some
                (Read.Rows (Array.of_list (List.sort_uniq Int.compare !kept)))
          | _ -> None)
      | _ -> None)
  | _ -> None

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
  let full = Read.reader ~whole:true m and few = Read.reader ~whole:false m in
  let sels =
    reading occ.mid (fun () ->
        members full m (List.map (fun t -> t.fp) targets))
  in
  let draw t sel =
    let warn msg = t.notes <- (occ.mid, msg) :: t.notes in
    let scale_of = scale_of cx occ pid t.fp.pnid in
    let box = t.panel.box and proj = t.panel.projection in
    let rows only rd sel =
      reading occ.mid (fun () ->
          Read.rows ?only cx.ctx rd ~id:occ.mid proj ~warn scale_of sel)
    in
    let drawn rd sel =
      let r = rows None rd sel in
      tagged occ.mid r.index (m.draw r)
    in
    let n =
      match sel with
      | Read.All -> Array.fold_left ( * ) 1 m.shape
      | Read.Rows r -> Array.length r
    in
    match m.reduce with
    | Some Cells -> (
        match cells cx m t.panel scale_of ~rows ~full ~few sel with
        | Some p -> p
        | None -> drawn full sel)
    | Some Raster when n > raster_rows || float n > device_pixels cx box ->
        let r = rows None full sel in
        tagged occ.mid r.index (rasterised cx (m.draw r))
    | Some M4 -> (
        match sel with
        | Read.Rows _ -> drawn full sel
        | Read.All -> (
            match reading occ.mid (fun () -> m4 cx m t.panel scale_of) with
            | Some kept -> drawn few kept
            | None -> drawn full sel))
    | Some Raster | None -> drawn full sel
  in
  List.iter2
    (fun t sel ->
      match sel with
      | None -> ()
      | Some sel -> t.pictures <- draw t sel :: t.pictures)
    targets sels

(* Panels *)

(* [panels cx] is the picture of each panel of the layout, in its order, with
   the warnings its marks gave. *)
let panels cx =
  let r = cx.resolved in
  let laid = Layout.panels cx.layout in
  let targets =
    List.concat_map
      (fun (pid, content) ->
        let fps = Option.value ~default:[] (find_path pid r.facets) in
        let targets =
          List.filter_map
            (fun fp ->
              List.find_opt
                (fun (p : Layout.panel) -> Nx.Ptree.Path.equal p.id fp.pnid)
                laid
              |> Option.map (fun panel ->
                  { fp; panel; pictures = []; notes = [] }))
            fps
        in
        (match targets with
        | [] -> ()
        | _ :: _ ->
            List.iter (fun occ -> draw_occ cx pid occ targets) content.occs);
        targets)
      (Arrange.panels r.shaped)
  in
  let drawn t =
    let tag = { Picture.id = t.panel.id; rows = Picture.Rows [||] } in
    (Picture.tag tag (Picture.group (List.rev t.pictures)), List.rev t.notes)
  in
  List.filter_map
    (fun (p : Layout.panel) ->
      List.find_opt (fun t -> Nx.Ptree.Path.equal t.panel.id p.id) targets
      |> Option.map drawn)
    laid

(* Texts *)

(* A quarter turn counterclockwise on the page: (u, v) to (v, -u). *)
let quarter = { Affine.xx = 0.; yx = -1.; xy = 1.; yy = 0.; x0 = 0.; y0 = 0. }

(* [placed c p] draws [p], in [c] where its text sets no colour. *)
let placed c (p : Layout.placed) =
  let origin = if p.turned then P2.v 0. 0. else p.at in
  let run acc colour o run =
    let c = Option.value colour ~default:c in
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
       (Picture.stroke pen (faded cx rule_alpha) (segments lines)
        :: List.map (placed (faded cx label_alpha)) a.ax_labels
       @ Option.to_list (Option.map (placed (ink cx)) a.ax_title)))

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
    tag_node a.ax_id
      (Picture.stroke
         (Stroke.v ~cap:`Butt (em cx grid_em))
         (faded cx grid_alpha) (segments lines))

(* Legends *)

let bar_steps = 256

(* [readers cx s] is each mark that reads the scale [s] through a role with a
   legend, with the indices of its bindings that read it, in the order of the
   figure. *)
let readers cx s =
  let (F f) = cx.ctx.scales.(s) in
  let add acc m =
    let same (o, _) = Nx.Ptree.Path.equal o.mid m.m_occ.mid in
    if Role.shown_on m.m_use <> Some `Legend then acc
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
             (faded cx rule_alpha) (segments lines)
        :: List.map (placed (faded cx label_alpha)) labels
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
                 placed (faded cx label_alpha) e.label;
               ])
             es)
  in
  tag_node g.lg_id
    (Picture.group
       (Option.to_list (Option.map (placed (ink cx)) g.lg_title) @ body))

(* Drawing *)

let afresh ~density l =
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
  let drawn = panels cx in
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
      @ List.map fst drawn
      @ List.map (draw_axis cx) axes
      @ List.map
          (fun (hd : Layout.header_out) ->
            tag_node hd.hd_id (placed (ink cx) hd.hd_label))
          (Layout.headers l)
      @ legends
      @ List.map (placed (ink cx)) (Layout.titles l))
  in
  let warnings =
    dedupe (Layout.warnings l @ List.concat_map snd drawn @ List.rev !notes)
  in
  { renderable = Renderable.v w h picture; warnings; layout = l; density }

(* A layout equal to the one [prev] drew, at an equal density, draws to
   [prev]. *)
let draw ?prev ~density l =
  match prev with
  | Some d when Float.equal d.density density && Layout.equal d.layout l -> d
  | _ -> afresh ~density l

(* Comparing and formatting *)

let renderable d = d.renderable
let warnings d = d.warnings

let equal d d' =
  List.equal equal_warning d.warnings d'.warnings
  && Renderable.equal d.renderable d'.renderable

let pp ppf d =
  Format.fprintf ppf "@[<v>%a" Renderable.pp d.renderable;
  if d.warnings <> [] then begin
    Format.fprintf ppf "@,@[<v 2>warnings";
    List.iter (Format.fprintf ppf "@,%a" pp_warning) d.warnings;
    Format.fprintf ppf "@]"
  end;
  Format.fprintf ppf "@]"
