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
   reading only what it draws; its draw function draws the others, and the large
   images of what it draws are gathered. *)

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

(* [reading mid f] is [f ()], which reads the tensors of the mark [mid], with
   the errors of reading them naming the mark. *)
let reading mid f =
  try f () with Invalid_argument msg -> err "draw" "%a: %s" pp_id mid msg

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

(* [band_cells cx reads index] is the number of categories of the band scale
   that the binding [index] reads, if it has no padding. *)
let band_cells cx reads index =
  match reads.(index) with
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

(* [cells cx m panel reads ~rows ~full ~few sel] is the image whose pixels are
   the cells of the rows [sel], if they draw as one: [x] and [y] read band
   scales without padding, the mark binds no [x2], [y2] or [stroke], and no two
   rows share a cell. Past 4 × 4 cells per device pixel, only the rows of the
   cells that raster output samples are read. *)
let cells cx m (panel : Layout.panel) reads ~rows ~full ~few sel =
  match (binding_index m Role.x, binding_index m Role.y) with
  | Some xi, Some yi
    when not (binds m Role.x2 || binds m Role.y2 || binds m Role.stroke) -> (
      match (band_cells cx reads xi, band_cells cx reads yi) with
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

(* [quantities shape ~role l s] is the quantities of the lift [l] in a mark of
   shape [shape], where they live, its missing rows, if any, and the fitted
   scale that reads them, if [s] reads quantities. *)
let quantities : type d.
    int array ->
    role:string ->
    d lift ->
    fitted ->
    (Nx.float64_t * Nx.bool_t option * float Scale.t) option =
 fun shape ~role l (F f) ->
  match (kind l, f.kind) with
  | Quantities, Quantities ->
      let (Lift.Quantities q) = Lift.eval shape ~role l f.spec in
      Some (q.values, q.miss.rows, f.scale)
  | _ -> None

(* [runs edges ~inside ~dropped x y] is the increasing positions of the rows M4
   keeps among those of the series, the rows of the [[|s; n|]] tensors [x] and
   [y], that [inside] puts in the panel, all by default, if they have more than
   [m4_rows] rows per run: of each run of consecutive rows in one column, its
   first, last, lowest and highest rows, and of each run of rows that [dropped]
   drops, none by default, its first. A row's column is the bin the [edges] put
   its [x] in, found where the data lives. Runs are numbered by a running count
   of their starts, so that [s * n] rows cost [O (s * n)] whatever their gaps.
   With fewer rows per run, as where [x] is in no order, keeping four of each
   would keep most rows. *)
let runs edges ~inside ~dropped xt yt =
  let s = (Nx.shape xt).(0) and last = (Nx.shape xt).(1) in
  let n = s * last in
  let p = Nx.arange Nx.int64 0 n 1 in
  let none = Nx.full Nx.int64 [| n |] (-1L) in
  let only k = match inside with None -> k | Some i -> Nx.logical_and k i in
  let valid =
    match dropped with
    | None -> inside
    | Some d -> Some (only (Nx.logical_not d))
  in
  (* A dropped row's column is [-1], so that a run of them is one bin. *)
  let col = Nx.flatten (Nx.searchsorted ~side:`Right edges xt) in
  let col = match dropped with None -> col | Some d -> Nx.where d none col in
  (* Each row's predecessor's column, [-2] for the first row of its series in
     the panel. *)
  let shifted t =
    Nx.flatten
      (Nx.concatenate ~axis:1
         [
           Nx.full Nx.int64 [| s; 1 |] (-2L);
           Nx.slice [ A; R (0, last - 1) ] (Nx.reshape [| s; last |] t);
         ])
  in
  let prev =
    match inside with
    | None -> shifted col
    | Some inside ->
        let before =
          shifted
            (Nx.cummax ~axis:1
               (Nx.reshape [| s; last |] (Nx.where inside p none)))
        in
        Nx.where (Nx.less_s before 0L)
          (Nx.full Nx.int64 [| n |] (-2L))
          (Nx.take ~indices:(Nx.maximum_s before 0L) col)
  in
  let starts = only (Nx.not_equal col prev) in
  let run = Nx.sub_s (Nx.cumsum (Nx.cast Nx.int64 starts)) 1L in
  let bins = Int64.to_int (Nx.item [ n - 1 ] run) + 1 in
  let rows =
    match inside with
    | None -> n
    | Some i -> Int64.to_int (Nx.item [] (Nx.sum (Nx.cast Nx.int64 i)))
  in
  if rows <= m4_rows * bins then None
  else
    let within = function None -> run | Some m -> Nx.where m run none in
    let any_bin = within inside and valid_bin = within valid in
    let scatter mode bin values init =
      Nx.scatter ~mode ~axis:0 ~indices:bin ~values init
    in
    let unset = Nx.full Nx.int64 [| bins |] Int64.max_int in
    let yv = Nx.flatten yt in
    let extreme mode init =
      let e = scatter mode valid_bin yv (Nx.full Nx.float64 [| bins |] init) in
      let hit = Nx.equal yv (Nx.take ~indices:valid_bin e) in
      let hit =
        match valid with None -> hit | Some v -> Nx.logical_and v hit
      in
      scatter `Min valid_bin
        (Nx.where hit p (Nx.full_like p Int64.max_int))
        unset
    in
    (* The kept rows are marked where the data lives, the unset ends of bins
       ([-1] and [max_int]) falling outside and being dropped, and read once, in
       order. *)
    let kept =
      List.fold_left
        (fun marks rows ->
          Nx.scatter ~axis:0 ~indices:rows
            ~values:(Nx.full Nx.bool [| bins |] true)
            marks)
        (Nx.full Nx.bool [| n |] false)
        [
          scatter `Min any_bin p unset;
          scatter `Max valid_bin p (Nx.full Nx.int64 [| bins |] (-1L));
          extreme `Max Float.neg_infinity;
          extreme `Min Float.infinity;
        ]
    in
    Some (Array.map Int64.to_int (Nx.to_array (Nx.nonzero kept).(0)))

(* [m4 cx m panel reads mask] is the rows of [m] that M4 keeps of those that
   [mask] puts in [panel], as [runs] finds them, if it applies: the series have
   more than [m4_rows] rows per device-pixel column, and their channels other
   than positions and facets are constant along them. A column is cut at the
   domain's ends, and the rows beyond the outer edges make one more bin on each
   side. When the facets are constant along the series, only the panel's series
   are read; otherwise the rows of other panels are skipped, so that they
   neither join nor split a run, as they are absent from the panel's drawing. *)
let m4 cx m (panel : Layout.panel) reads mask =
  let shape = m.shape in
  let rank = Array.length shape in
  let last = if rank = 0 then 0 else shape.(rank - 1) in
  let d = cx.ctx.density and box = panel.box in
  let p0 = Float.floor (Box2.minx box *. d) in
  let w = Float.to_int (Float.ceil (Box2.maxx box *. d) -. p0) in
  let constant (B b) =
    match b.role.use with
    | Position { far = false; _ } | Facet _ -> true
    | _ -> not (Channel.varies shape b.ch (-1))
  in
  let facet_varies (B b) =
    match b.role.use with
    | Facet _ -> Channel.varies shape b.ch (-1)
    | _ -> false
  in
  let read role =
    match binding_index m role with
    | None -> None
    | Some index -> (
        match (List.nth m.bindings index, reads.(index)) with
        | B b, Some i ->
            Option.bind (data b.ch) (fun dt ->
                quantities shape ~role:b.role.name dt.lift cx.ctx.scales.(i))
        | B _, None -> None)
  in
  (* The values of [x] at the device-pixel edges and at the domain's ends, where
     [Mark.project] cuts the path, so that no column holds rows on both sides of
     a cut. *)
  let edges sx =
    let device c =
      let x = (p0 +. float c) /. d in
      Option.bind
        (Coord.invert panel.projection (P2.v x (Box2.miny box)))
        (fun (u, _) -> Scale.invert sx u)
    in
    (* A row on an edge lies in the bin after it, so the upper end is moved up
       by an ulp, for the domain's ends to lie in it. *)
    let ends =
      match (Scale.invert sx 0., Scale.invert sx 1.) with
      | Some a, Some b ->
          [ Some (Float.min a b); Some (Float.succ (Float.max a b)) ]
      | _ -> [ None ]
    in
    let all = ends @ List.init (w + 1) device in
    if List.exists Option.is_none all then None
    else
      let all = List.sort Float.compare (List.map Option.get all) in
      Some (Nx.create Nx.float64 [| w + 3 |] (Array.of_list all))
  in
  let applies =
    rank > 0 && w > 0 && last > m4_rows * w && List.for_all constant m.bindings
  in
  match (applies, read Role.x, read Role.y) with
  | true, Some (x, mx, sx), Some (y, my, _) -> (
      match edges sx with
      | None -> None
      | Some edges -> (
          let series = Array.fold_left ( * ) 1 shape / last in
          let grid t =
            Nx.reshape [| series; last |] (Nx.broadcast_to shape t)
          in
          (* The rows missing [x] or [y], none if no row is. *)
          let missing =
            match (mx, my) with
            | None, m | m, None -> m
            | Some a, Some b -> Some (Nx.logical_or a b)
          in
          let missing =
            Option.bind missing (fun d ->
                if Nx.item [] (Nx.any d) then Some d else None)
          in
          let rows = Option.map (fun r -> Read.Rows r) in
          let dropped pick =
            Option.map (fun d -> Nx.flatten (pick d)) missing
          in
          match mask with
          | `All ->
              rows
                (runs edges ~inside:None ~dropped:(dropped grid) (grid x)
                   (grid y))
          | `Mask k when List.exists facet_varies m.bindings ->
              let inside = Some (Nx.flatten (grid k)) in
              rows
                (runs edges ~inside ~dropped:(dropped grid) (grid x) (grid y))
          | `Mask k ->
              let ends = Array.copy shape in
              ends.(rank - 1) <- 1;
              let ks = Nx.reshape [| series |] (Nx.broadcast_to ends k) in
              let ids = (Nx.nonzero ks).(0) in
              if Nx.numel ids = 0 then Some (Read.Rows [||])
              else
                let pick t = Nx.take ~axis:0 ~indices:ids (grid t) in
                let kept =
                  runs edges ~inside:None ~dropped:(dropped pick) (pick x)
                    (pick y)
                in
                let ids = Nx.to_array ids in
                let row q =
                  (Int64.to_int ids.(q / last) * last) + (q mod last)
                in
                rows (Option.map (Array.map row) kept)))
  | _ -> None

(* Marks *)

(* A panel being drawn: its facet panel, layout and the pictures and warnings of
   its marks so far, latest first. *)
type target = {
  fp : Resolved.panel;
  panel : Layout.panel;
  mutable pictures : Picture.t list;
  mutable notes : warning list;
}

(* [selection m mask] is the rows of [m] that [mask] selects, [None] where it
   selects none. The mask is read only along the leading axes it varies along:
   along the others it selects blocks of consecutive rows, as a facet on a
   leading axis does. *)
let selection m = function
  | `All -> Some Read.All
  | `Mask k -> (
      let shape = m.shape and ks = Nx.shape k in
      let rank = Array.length shape and r = Array.length ks in
      let varies a = a >= rank - r && ks.(a - rank + r) <> 1 in
      let lead = ref rank in
      while !lead > 0 && not (varies (!lead - 1)) do
        decr lead
      done;
      let lead = !lead in
      let block =
        Array.fold_left ( * ) 1 (Array.sub shape lead (rank - lead))
      in
      let ends = Array.mapi (fun a d -> if a < lead then d else 1) shape in
      let k = Nx.flatten (Nx.broadcast_to ends k) in
      match Nx.to_array (Nx.nonzero k).(0) with
      | [||] -> None
      | ids ->
          let rows = Array.make (Array.length ids * block) 0 in
          Array.iteri
            (fun b id ->
              let first = Int64.to_int id * block in
              for i = 0 to block - 1 do
                rows.((b * block) + i) <- first + i
              done)
            ids;
          Some (Read.Rows rows))

(* [draw_occ cx occ part targets] draws [occ], whose rows go among the panels of
   its cell by [part], in each panel of [targets] it has rows in. *)
let draw_occ cx occ part targets =
  let m = occ.mark in
  let full = Read.reader ~whole:true m and few = Read.reader ~whole:false m in
  let draw t mask =
    let warn msg = t.notes <- (occ.mid, msg) :: t.notes in
    let reads = Option.get (find_path occ.mid t.fp.reads) in
    let box = t.panel.box and proj = t.panel.projection in
    let rows only rd sel =
      reading occ.mid (fun () ->
          Read.rows ?only cx.ctx rd ~id:occ.mid proj ~warn reads sel)
    in
    let drawn rd sel =
      let r = rows None rd sel in
      tagged occ.mid r.index
        (Pixels.gathered ~density:cx.ctx.density (m.draw r))
    in
    let selected f =
      Option.map f (reading occ.mid (fun () -> selection m mask))
    in
    match m.reduce with
    | Some M4 -> (
        match reading occ.mid (fun () -> m4 cx m t.panel reads mask) with
        | Some (Read.Rows [||]) -> None
        | Some kept -> Some (drawn few kept)
        | None -> selected (drawn full))
    | Some Cells ->
        selected (fun sel ->
            match cells cx m t.panel reads ~rows ~full ~few sel with
            | Some p -> p
            | None -> drawn full sel)
    | Some Raster ->
        selected (fun sel ->
            let n =
              match sel with
              | Read.All -> Array.fold_left ( * ) 1 m.shape
              | Read.Rows r -> Array.length r
            in
            if n > raster_rows || float n > device_pixels cx box then
              let r = rows None full sel in
              tagged occ.mid r.index (rasterised cx (m.draw r))
            else drawn full sel)
    | None -> selected (drawn full)
  in
  List.iter
    (fun t ->
      match reading occ.mid (fun () -> Resolved.mask part t.fp) with
      | `None -> ()
      | (`All | `Mask _) as mask ->
          Option.iter (fun p -> t.pictures <- p :: t.pictures) (draw t mask))
    targets

(* Panels *)

(* [panels cx] is the picture of each panel of the layout, in its order, with
   the warnings its marks gave. *)
let panels cx =
  let r = cx.resolved in
  let laid = Layout.panels cx.layout in
  let targets =
    List.concat_map
      (fun (_, (c : Resolved.cell)) ->
        let targets =
          List.filter_map
            (fun (fp : Resolved.panel) ->
              List.find_opt
                (fun (p : Layout.panel) -> Nx.Ptree.Path.equal p.id fp.pnid)
                laid
              |> Option.map (fun panel ->
                  { fp; panel; pictures = []; notes = [] }))
            c.panels
        in
        (match targets with
        | [] -> ()
        | _ :: _ ->
            List.iter
              (fun occ ->
                draw_occ cx occ (Option.get (find_path occ.mid c.parts)) targets)
              c.content.occs);
        targets)
      r.cells
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
  if not p.turned then Rows.glyphs c p.at p.set
  else
    Picture.transform
      Affine.(translate (P2.x p.at) (P2.y p.at) * quarter)
      (Rows.glyphs c (P2.v 0. 0.) p.set)

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
