(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

[@@@ocamlformat "wrap-comments=false"]

(* Laying out

   [layout] turns a resolved figure into a tree of nodes: a panel per data area,
   a grid per grid and per arrangement of facet panels. Only data areas take
   tracks. Axes, headers, legends and titles are guides on a node's side, laid
   out along it and stacked outward in tiers:

   {v
   figure titles  ─┐
   legends         │ bands beyond the node's side, each as deep as its
   scale titles    │ deepest guide; guides of one tier share a band
   headers         │ where they lie apart along the side
   axis proper    ─┘
   ──────────────── the side of the hull of the node's data areas
   content          the protrusions of a grid's boundary cells
   v}

   A guide serving several panels is on the smallest node holding them. A grid
   makes each gap the protrusions that meet it plus a gap, gives each track what
   its cells and guides need at least, and shares the rest among its tracks by
   weight, so the data areas of a column share their edges and those of a row
   their tops and bottoms.

   Once its ticks and rows are frozen, a guide's depth, least length and reach
   past the ends of its side do not depend on the side's length, and which
   guides share a band is decided once. So the layout is four solves: without
   guides; with the guides of the first choice of ticks; with those of the
   frozen choice, each in a band of its own and each horizontal legend one entry
   a row, which gives the shortest sides the figure can take, at whose lengths
   rows wrap and bands are shared; and a last one, whose sides are no shorter,
   so guides laid out at them fit what it reserved. *)

module P2 = Hugin_next_gg.P2
module Box2 = Hugin_next_gg.Box2
module Text = Hugin_next_text.Text
module Scale = Hugin_next_kit.Scale
module Ticks = Hugin_next_kit.Ticks
open Common
open Figure
open Arrange
open Resolved

let margin_em = 0.5 (* Around the page. *)
let horizontal = Guide.horizontal
let longest = Guide.longest

(* Nodes *)

(* A node holds guides of type ['g]: their specifications, then the guides each
   pass lays out. *)
type 'g node =
  | Panel of {
      id : id;
      coord : Coord.t;
      ratio : float option; (* The height of its data area over its width. *)
      guides : 'g list;
    }
  | Grid of 'g grid

and 'g grid = {
  id : id;
  widths : float array; (* The weights of its tracks. *)
  heights : float array;
  cells : 'g cell list;
  guides : 'g list;
}

and 'g cell = { r0 : int; c0 : int; nr : int; nc : int; node : 'g node }

(* A laid-out guide, in the band of its side and tier counted outward. *)
type laid = { guide : Guide.t; band : int }

let guides_of = function Panel p -> p.guides | Grid g -> g.guides
let path id segs = Nx.Ptree.Path.(v (segments id @ segs))

let rec holds node pid =
  match node with
  | Panel p -> Nx.Ptree.Path.equal p.id pid
  | Grid g -> List.exists (fun c -> holds c.node pid) g.cells

(* [attach pids spec node] is [node] with the guide [spec id] on the smallest
   node [id] of it that holds the panels [pids]. *)
let rec attach pids spec node =
  match node with
  | Panel p -> Panel { p with guides = p.guides @ [ spec p.id ] }
  | Grid g -> (
      match
        List.find_opt (fun c -> List.for_all (holds c.node) pids) g.cells
      with
      | None -> Grid { g with guides = g.guides @ [ spec g.id ] }
      | Some c ->
          let cell c' =
            if c' == c then { c with node = attach pids spec c.node } else c'
          in
          Grid { g with cells = List.map cell g.cells })

(* [fold f acc node geo] folds [f] over the nodes of [node] with their data
   hulls in [geo], the cells of a grid before it. *)
type geo = { hull : Box2.t; kids : geo list }

let rec fold f acc node geo =
  let acc =
    match node with
    | Panel _ -> acc
    | Grid g ->
        List.fold_left2
          (fun acc c k -> fold f acc c.node k)
          acc g.cells geo.kids
  in
  f acc node geo.hull

(* Building nodes *)

(* [units s] is the length of the domain of [s] in units of its transform, or
   one if it spans none. *)
let units (F f) =
  let u = Scale.length f.scale in
  if Float.is_finite u && u > 0. then u else 1.

let default_side : Role.shown -> side = function
  | `Axis Role.X -> `Bottom
  | `Axis Role.Y -> `Left
  | `Header Role.X -> `Top
  | `Header Role.Y | `Legend -> `Right

(* [panel scales c p] is the facet panel [p] of a cell holding [c], with its
   axes and headers: explicit, or else the default. *)
let panel scales (c : content) p =
  let axis (on : Role.shown) =
    Option.bind (shown c p.reads on) (fun i ->
        let (F f) = scales.(i) in
        let explicit (_, g, _) =
          if is_axis g && String.equal g.scale f.name then Some g else None
        in
        let guide =
          match List.find_map explicit c.guides with
          | Some g -> g
          | None ->
              let show = Option.value f.guide ~default:true in
              {
                kind = Axis { grid = false };
                scale = f.name;
                side = None;
                show;
              }
        in
        let part =
          match on with
          | `Axis _ -> Some (Guide.Ticks { labelled = true })
          | `Header Role.X -> Option.map (fun k -> Guide.Header k) p.pfx
          | `Header Role.Y -> Option.map (fun k -> Guide.Header k) p.pfy
          | `Legend -> None
        in
        Option.map
          (fun part ->
            {
              Guide.id = path p.pnid [ Field "axis"; Field f.name ];
              side = Option.value guide.side ~default:(default_side on);
              kind = Axis { guide; scale = i; part };
            })
          part)
  in
  let coord =
    match c.coords with
    | (_, k) :: _ -> k
    | [] -> (
        match List.find_map (fun o -> o.mark.coord) c.occs with
        | Some k -> k
        | None -> Coord.cartesian ())
  in
  let units on =
    Option.fold ~none:1. ~some:(fun i -> units scales.(i)) (shown c p.reads on)
  in
  let (Coord.Cartesian { aspect }) = coord in
  let ratio =
    Option.map
      (fun k -> k *. units (`Axis Role.Y) /. units (`Axis Role.X))
      aspect
  in
  let guides =
    List.filter_map axis
      [ `Axis Role.X; `Axis Role.Y; `Header Role.X; `Header Role.Y ]
  in
  Panel { id = p.pnid; coord; ratio; guides }

(* [content r scales pid c] is the panel of the cell [pid], or the grid of its
   facet panels: one column per category of its fx scale and one row per
   category of its fy scale, or rows of [wrap] panels. *)
let content (r : Resolved.t) scales pid c =
  let cell = Option.get (find_path pid r.cells) in
  match cell.panels with
  | [ p ] when Nx.Ptree.Path.equal p.pnid pid -> panel scales c p
  | ps ->
      let scale i = Option.map (fun i -> scales.(i)) i in
      let count i =
        match scale i with
        | Some (F { kind = Channel.Categories; scale; _ }) ->
            List.length (category_names scale)
        | Some (F { kind = Channel.Quantities; _ }) | None -> 0
      in
      let nx = max 1 (count cell.fx) and ny = max 1 (count cell.fy) in
      let pos = Option.value ~default:0 in
      let ncols, nrows, at =
        match scale cell.fx with
        | Some (F { kind = Channel.Categories; scale; _ })
          when Option.is_some (Scale.wrap scale) ->
            let w = min (Option.get (Scale.wrap scale)) nx in
            (w, (nx + w - 1) / w, fun p -> (pos p.pfx / w, pos p.pfx mod w))
        | _ -> (nx, ny, fun p -> (pos p.pfy, pos p.pfx))
      in
      let cell p =
        let r0, c0 = at p in
        { r0; c0; nr = 1; nc = 1; node = panel scales c p }
      in
      let widths = Array.make ncols 1. and heights = Array.make nrows 1. in
      Grid { id = pid; widths; heights; cells = List.map cell ps; guides = [] }

let weights n = function None -> Array.make n 1. | Some ws -> Array.of_list ws

(* [of_shaped r scales id s] is the node of [s], titled by its figure titles,
   the innermost first. *)
let rec of_shaped r scales id s =
  let titles =
    List.rev_map
      (fun (align, text) ->
        { Guide.id; side = `Top; kind = Title { align; text } })
      s.titles
  in
  match s.body with
  | Single c -> (
      match content r scales id c with
      | Panel p -> Panel { p with guides = p.guides @ titles }
      | Grid g -> Grid { g with guides = titles })
  | Arr a ->
      let cell (c : Arrange.cell) =
        let node = of_shaped r scales c.cid c.s in
        { r0 = c.row; c0 = c.col; nr = c.rows; nc = c.cols; node }
      in
      let widths = weights a.ncols a.widths in
      let heights = weights a.nrows a.heights in
      Grid
        { id; widths; heights; cells = List.map cell a.cells; guides = titles }

(* [shows g] is the scale, side and category of [g] if it is a shown axis or
   header. *)
let shows (g : Guide.spec) =
  match g.kind with
  | Axis { guide = { show = true; _ }; scale; part = Ticks _ } ->
      Some (scale, g.side, None)
  | Axis { guide = { show = true; _ }; scale; part = Header c } ->
      Some (scale, g.side, Some c)
  | Axis _ | Legend _ | Title _ -> None

(* [labelled node] is [node] with each axis of a panel unlabelled, and each
   header dropped, where the next cell on its side holds a panel showing it. *)
let rec labelled = function
  | Panel _ as n -> n
  | Grid g ->
      let cells =
        List.map (fun c -> { c with node = labelled c.node }) g.cells
      in
      let next c c' (side : side) =
        match side with
        | `Bottom -> c'.c0 = c.c0 && c'.nc = c.nc && c'.r0 = c.r0 + c.nr
        | `Top -> c'.c0 = c.c0 && c'.nc = c.nc && c'.r0 + c'.nr = c.r0
        | `Left -> c'.r0 = c.r0 && c'.nr = c.nr && c'.c0 + c'.nc = c.c0
        | `Right -> c'.r0 = c.r0 && c'.nr = c.nr && c'.c0 = c.c0 + c.nc
      in
      let beside c (g : Guide.spec) =
        let same g' = Option.is_some (shows g) && shows g' = shows g in
        List.exists
          (fun c' ->
            next c c' g.side
            &&
            match c'.node with
            | Panel p -> List.exists same p.guides
            | Grid _ -> false)
          cells
      in
      let relabel c (g : Guide.spec) =
        match g.kind with
        | Axis ({ part = Ticks _; _ } as a) when beside c g ->
            Some
              {
                g with
                kind = Axis { a with part = Ticks { labelled = false } };
              }
        | Axis { part = Header _; _ } when beside c g -> None
        | Axis _ | Legend _ | Title _ -> Some g
      in
      let cell c =
        match c.node with
        | Panel p ->
            {
              c with
              node =
                Panel { p with guides = List.filter_map (relabel c) p.guides };
            }
        | Grid _ -> c
      in
      Grid { g with cells = List.map cell cells }

(* [showing node] is each scale and side of a shown axis or header of [node],
   with its guide and the panels showing it, headers first. *)
let showing node =
  let rec go acc = function
    | Grid g -> List.fold_left (fun acc c -> go acc c.node) acc g.cells
    | Panel p ->
        let add acc (g : Guide.spec) =
          match (shows g, g.kind) with
          | Some (scale, side, cat), Axis { guide; _ } ->
              (Option.is_none cat, (scale, side), guide, p.id) :: acc
          | _ -> acc
        in
        List.fold_left add acc p.guides
  in
  let group groups (_, k, g, pid) =
    match List.assoc_opt k groups with
    | None -> groups @ [ (k, (g, [ pid ])) ]
    | Some (g, pids) ->
        List.map
          (fun (k', v) -> if k' = k then (k, (g, pids @ [ pid ])) else (k', v))
          groups
  in
  let headers (a, _, _, _) (b, _, _, _) = Bool.compare a b in
  List.fold_left group [] (List.stable_sort headers (List.rev (go [] node)))

(* [build r scales] is the node of [r]: the title of an axis or of headers is on
   the smallest node holding the panels showing them, and a legend on the
   smallest node holding the panels its readers lie in. A categorical legend
   whose readers each have the data of a position channel of their mark, whose
   axis is shown in their panels, is left out: the axis names its categories.
   An explicit legend, or a mark's, keeps it. *)
let build (r : Resolved.t) scales =
  let tree = of_shaped r scales Nx.Ptree.Path.root r.shaped in
  let titled tree ((scale, side), (guide, pids)) =
    let (F f) = scales.(scale) in
    attach pids
      (fun id ->
        {
          Guide.id = path id [ Field "axis"; Field f.name ];
          side = (if horizontal side then side else `Top);
          kind = Axis { guide; scale; part = Scale_title side };
        })
      tree
  in
  let shown = showing tree in
  let tree = List.fold_left titled (labelled tree) shown in
  let explicit = Arrange.legends r.shaped in
  let panels pid =
    match find_path pid r.cells with
    | Some c -> List.map (fun p -> p.pnid) c.panels
    | None -> [ pid ]
  in
  let on_axis pid i =
    List.exists
      (fun ((s, _), (_, pids)) ->
        s = i && List.exists (Nx.Ptree.Path.equal pid) pids)
      shown
  in
  (* [named pids m] is [true] iff a position channel of the mark of [m], with
     the data of [m], reads a scale shown by an axis in each of the panels
     [pids] that [m] lies in. *)
  let named pids m =
    let position j (B b) =
      match (b.role.use, Channel.data b.ch) with
      | Role.Position _, Some d
        when Option.is_some
               (Channel.equal_lift d.Channel.lift m.m_d.Channel.lift) ->
          Some j
      | _ -> None
    in
    let positions =
      List.filter_map Fun.id (List.mapi position m.m_occ.mark.bindings)
    in
    let shows (p : Resolved.panel) =
      match find_path m.m_occ.mid p.reads with
      | None -> false
      | Some reads ->
          List.exists
            (fun j -> Option.fold ~none:false ~some:(on_axis p.pnid) reads.(j))
            positions
    in
    let cell = Option.get (find_path m.m_pid r.cells) in
    List.for_all
      (fun (p : Resolved.panel) ->
        (not (List.exists (Nx.Ptree.Path.equal p.pnid) pids)) || shows p)
      cell.panels
  in
  let legend (tree, i) (F f) =
    let readers =
      List.filter (fun m -> Role.shown_on m.m_use = Some `Legend) f.members
    in
    let pids =
      match panel_of f.key with
      | Some p -> [ p ]
      | None -> List.concat_map (fun m -> panels m.m_pid) readers
    in
    let mine (_, (g : guide), key) =
      if String.equal g.scale f.name && equal_key key f.key then Some g
      else None
    in
    let explicit = List.find_map mine explicit in
    let named () =
      match (f.kind, explicit, f.guide) with
      | Channel.Categories, None, None -> List.for_all (named pids) readers
      | (Channel.Categories | Channel.Quantities), _, _ -> false
    in
    if readers = [] || named () then (tree, i + 1)
    else
      let guide =
        match explicit with
        | Some g -> g
        | None ->
            let show = Option.value f.guide ~default:f.legend in
            { kind = Legend; scale = f.name; side = None; show }
      in
      let kind =
        match f.kind with
        | Channel.Quantities -> "num"
        | Channel.Categories -> "cat"
      in
      let spec id =
        {
          Guide.id = path id [ Field "legend"; Field f.name; Field kind ];
          side = Option.value guide.side ~default:`Right;
          kind = Legend { guide; scale = i };
        }
      in
      (attach pids spec tree, i + 1)
  in
  fst (List.fold_left legend (tree, 0) r.scales)

(* Bands *)

let on (p : Guide.sides) : side -> float = function
  | `Left -> p.left
  | `Right -> p.right
  | `Top -> p.top
  | `Bottom -> p.bottom

let rank : Guide.tier -> int = function
  | Proper -> 0
  | Headers -> 1
  | Scale_titles -> 2
  | Legends -> 3
  | Figure_titles -> 4

let tier (l : laid) = rank (Guide.tier l.guide.spec.kind)

(* [along g] is the interval along its side that [g] covers. *)
let along (g : Guide.t) =
  let b = Option.get g.bounds in
  if horizontal g.spec.side then (Box2.minx b, Box2.maxx b)
  else (Box2.miny b, Box2.maxy b)

(* [bands em node] is the offset of each laid guide of [node] beyond its side of
   the hull of the node's data areas, and the protrusions of [node]. On a side
   the node protrudes by its content, then by each tier of guides, innermost
   first, in its bands, each as deep as its deepest guide.

   A guide within its side cannot meet the guides of the adjacent sides, which
   lie beyond the hull; one reaching past an end can. Lower tiers own the
   corner: past the axis proper, a tier with a guide reaching past an end starts
   beyond the reach there of the lower tiers' guides on the adjacent side. The
   node protrudes by its bands, or by every such reach if larger. *)
let rec bands em node =
  let gs =
    List.filter (fun l -> Option.is_some l.guide.bounds) (guides_of node)
  in
  let on_side s (l : laid) = equal_side l.guide.spec.side s in
  let reach s adjacent below =
    longest
      (fun l ->
        if on_side adjacent l && tier l < below then
          on (Guide.protrusion l.guide) s
        else 0.)
      gs
  in
  let depth (l : laid) = on (Guide.protrusion l.guide) l.guide.spec.side in
  let side s =
    let first, last =
      if horizontal s then (`Left, `Right) else (`Top, `Bottom)
    in
    let tier_bands (off, acc) k =
      match List.filter (fun l -> on_side s l && tier l = k) gs with
      | [] -> (off, acc)
      | here ->
          let clear adjacent past =
            if k > 0 && List.exists past here then reach s adjacent k else 0.
          in
          let past_start l = fst (along l.guide) < 0. in
          let past_end l = snd (along l.guide) > l.guide.length in
          let start =
            Float.max off
              (Float.max (clear first past_start) (clear last past_end))
          in
          let n = 1 + List.fold_left (fun m l -> max m l.band) 0 here in
          let offs = Array.make (n + 1) start in
          for b = 0 to n - 1 do
            let d =
              longest (fun l -> if l.band = b then depth l else 0.) here
            in
            offs.(b + 1) <- offs.(b) +. d
          done;
          (offs.(n), List.map (fun l -> (l, offs.(l.band))) here @ acc)
    in
    let off, acc =
      List.fold_left tier_bands (on (content em node) s, []) [ 0; 1; 2; 3; 4 ]
    in
    (Float.max off (Float.max (reach s first 5) (reach s last 5)), acc)
  in
  let l, gl = side `Left and r, gr = side `Right in
  let t, gt = side `Top and b, gb = side `Bottom in
  let offs = gl @ gr @ gt @ gb in
  ( List.filter_map
      (fun g -> Option.map (fun o -> (g.guide, o)) (List.assq_opt g offs))
      gs,
    { Guide.left = l; right = r; top = t; bottom = b } )

(* [content em node] is the protrusion of the boundary cells of [node]. *)
and content em = function
  | Panel _ -> Guide.no_sides
  | Grid g ->
      let nc = Array.length g.widths and nr = Array.length g.heights in
      List.fold_left
        (fun (p : Guide.sides) c ->
          let (q : Guide.sides) = prot em c.node in
          let most edge a b = if edge then Float.max a b else a in
          {
            left = most (c.c0 = 0) p.left q.left;
            right = most (c.c0 + c.nc = nc) p.right q.right;
            top = most (c.r0 = 0) p.top q.top;
            bottom = most (c.r0 + c.nr = nr) p.bottom q.bottom;
          })
        Guide.no_sides g.cells

and prot em node = snd (bands em node)

(* [part em node] is [node] with each guide in the first band of its side and
   tier whose guides it lies a gap apart from. Titles are anchored at the start,
   the middle or the end of their side, so guides apart along a side stay apart
   along a longer one: parted at the shortest lengths a figure can take, its
   guides stay parted at the final ones. *)
let rec part em node =
  let gap = em *. Guide.gap_em in
  let apart (a : laid) (b : laid) =
    let a0, a1 = along a.guide and b0, b1 = along b.guide in
    a1 +. gap <= b0 || b1 +. gap <= a0
  in
  let put placed (l : laid) =
    let mate b (m : laid) =
      m.band = b
      && Option.is_some m.guide.bounds
      && equal_side m.guide.spec.side l.guide.spec.side
      && tier m = tier l
    in
    let rec first b =
      if List.for_all (apart l) (List.filter (mate b) placed) then b
      else first (b + 1)
    in
    let band = if Option.is_some l.guide.bounds then first 0 else 0 in
    placed @ [ { l with band } ]
  in
  let guides = List.fold_left put [] (guides_of node) in
  match node with
  | Panel p -> Panel { p with guides }
  | Grid g ->
      let cells =
        List.map (fun c -> { c with node = part em c.node }) g.cells
      in
      Grid { g with cells; guides }

(* Solving grids *)

let sum a = Array.fold_left ( +. ) 0. a

(* [least weights unit cells gaps] is the least length of each track: its weight
   times [unit.(i)], and at least what each cell it alone holds needs, with each
   cell spanning tracks given what it needs beyond them, by weight. [cells] are
   the start, the number of tracks and the need of each cell. *)
let least weights unit cells gaps =
  let m = Array.mapi (fun i k -> k *. unit.(i)) weights in
  List.iter (fun (s, n, l) -> if n = 1 then m.(s) <- Float.max m.(s) l) cells;
  List.iter
    (fun (s, n, l) ->
      if n > 1 then begin
        let have = ref 0. and weight = ref 0. in
        for i = s to s + n - 1 do
          have := !have +. m.(i);
          weight := !weight +. weights.(i)
        done;
        for i = s to s + n - 2 do
          have := !have +. gaps.(i)
        done;
        let excess = l -. !have in
        if excess > 0. then
          for i = s to s + n - 1 do
            let share =
              if !weight > 0. then excess *. weights.(i) /. !weight
              else excess /. float n
            in
            m.(i) <- m.(i) +. share
          done
      end)
    cells;
  m

(* [spread weights pinned m avail] is the lengths of tracks of [weights] in
   [avail], and the excess that no track takes. A track that is not [pinned] has
   the greater of its least length in [m] and its weight's share of what the
   other tracks leave; a pinned one has its least length. The share is found by
   water-filling: a track whose least length is above its share keeps it, and
   the others share again what it leaves, until none is left below its least
   length. *)
let spread weights pinned m avail =
  let n = Array.length weights in
  (* [share free] is the length per weight that the [free] tracks share. *)
  let share free =
    let rest = ref avail and total = ref 0. in
    for i = 0 to n - 1 do
      if free.(i) then total := !total +. weights.(i)
      else rest := !rest -. m.(i)
    done;
    if !total > 0. then !rest /. !total else 0.
  in
  let rec fill free =
    let u = share free in
    let next = Array.mapi (fun i f -> f && weights.(i) *. u >= m.(i)) free in
    if Array.for_all2 Bool.equal next free then (free, u) else fill next
  in
  let free, u = fill (Array.map not pinned) in
  let l = Array.mapi (fun i b -> if free.(i) then weights.(i) *. u else b) m in
  if Array.exists Fun.id free then (l, 0.)
  else (l, Float.max 0. (avail -. sum l))

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
  aspects : (laid cell * float) list; (* Single cells holding panels with one. *)
}

type tracks = {
  col_len : float array;
  row_len : float array;
  col_gap : float array;
  row_gap : float array;
  x_off : float; (* What no track takes, split evenly about the tracks. *)
  y_off : float;
}

(* [needs guides] is the width and the height that [guides] need. *)
let needs guides =
  List.fold_left
    (fun (w, h) { guide = g; _ } ->
      if horizontal g.spec.side then (Float.max w g.least, h)
      else (w, Float.max h g.least))
    (0., 0.) guides

(* [natural em unit node] is the least width and height of [node], [unit] being
   the data area a track of weight [1.] has at least. *)
let rec natural em unit = function
  | Panel p -> (
      let w, h = needs p.guides in
      (* A panel with an aspect fits its box in its cell, so the box holds what
         the cell must hold only if both lengths ask for it. *)
      match p.ratio with
      | None -> (w, h)
      | Some r -> (Float.max w (h /. r), Float.max h (r *. w)))
  | Grid g ->
      let m = measure em unit g in
      let rows = aspect_rows m m.cols_least in
      (sum m.cols_least +. sum m.cgaps, sum rows +. sum m.rgaps)

and measure em (uw, uh) g =
  let cells =
    List.map (fun c -> (c, prot em c.node, natural em (uw, uh) c.node)) g.cells
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
        most before (fun c -> last c = j)
        +. most after (fun c -> first c = j + 1)
        +. (em *. Guide.gap_em))
  in
  let nc = Array.length g.widths and nr = Array.length g.heights in
  let cgaps =
    gaps nc
      (fun c -> c.c0)
      (fun c -> c.c0 + c.nc - 1)
      (fun (p : Guide.sides) -> p.right)
      (fun p -> p.left)
  in
  let rgaps =
    gaps nr
      (fun c -> c.r0)
      (fun c -> c.r0 + c.nr - 1)
      (fun (p : Guide.sides) -> p.bottom)
      (fun p -> p.top)
  in
  let aspects =
    List.filter_map
      (fun (c, _, _) ->
        match c.node with
        | Panel { ratio = Some r; _ } when c.nr = 1 && c.nc = 1 -> Some (c, r)
        | _ -> None)
      cells
  in
  (* The node's own guides need lengths that span all its tracks. *)
  let w, h = needs g.guides in
  let cols_least =
    least g.widths (Array.make nc uw)
      ((0, nc, w) :: List.map (fun (c, _, (w, _)) -> (c.c0, c.nc, w)) cells)
      cgaps
  in
  (* A row holding a panel with an aspect takes its height from its column. *)
  let rows_least =
    let unit = Array.make nr uh in
    List.iter (fun (c, _) -> unit.(c.r0) <- 0.) aspects;
    least g.heights unit
      ((0, nr, h) :: List.map (fun (c, _, (_, h)) -> (c.r0, c.nr, h)) cells)
      rgaps
  in
  { cgaps; rgaps; cols_least; rows_least; aspects }

(* [aspect_rows m cols] is the least length of each row, a row holding a panel
   with an aspect needing that panel's height at the width of its column. *)
and aspect_rows m cols =
  let need = aspect_need m cols in
  Array.mapi (fun i b -> Float.max b need.(i)) m.rows_least

and aspect_need m cols =
  let need = Array.make (Array.length m.rows_least) 0. in
  List.iter
    (fun (c, r) -> need.(c.r0) <- Float.max need.(c.r0) (r *. cols.(c.c0)))
    m.aspects;
  need

let solve_grid em unit g w h =
  let m = measure em unit g in
  let avail_w = w -. sum m.cgaps and avail_h = h -. sum m.rgaps in
  let nc = Array.length g.widths in
  let cols, slack_x =
    spread g.widths (Array.make nc false) m.cols_least avail_w
  in
  let is_aspect = Array.make (Array.length g.heights) false in
  List.iter (fun (c, _) -> is_aspect.(c.r0) <- true) m.aspects;
  let rows = aspect_rows m cols in
  let done_ cols slack_x rows =
    let rows, slack_y = spread g.heights is_aspect rows avail_h in
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
    let cols, slack_x = spread g.widths pinned least avail_w in
    done_ cols slack_x rows
  end

(* Placing *)

(* [fit_aspect r box] is the largest box of ratio [r] centred in [box]. A cell's
   height is exact only to the rounding of its place on the page, so a height
   within it of [r] times the width fits that width: else a flat panel would
   shrink by its rounding over a tiny [r]. *)
let fit_aspect r box =
  let w = Box2.w box and h = Box2.h box in
  let ulp = 4. *. Float.epsilon *. Float.abs (Box2.maxy box) in
  let w', h' = if h +. ulp >= r *. w then (w, r *. w) else (h /. r, h) in
  Box2.v
    (Box2.minx box +. ((w -. w') /. 2.))
    (Box2.miny box +. ((h -. h') /. 2.))
    w' h'

(* [geometry em unit node box] is the data hulls of [node] placed in [box]. *)
let rec geometry em unit node box =
  match node with
  | Panel { ratio = None; _ } -> { hull = box; kids = [] }
  | Panel { ratio = Some r; _ } -> { hull = fit_aspect r box; kids = [] }
  | Grid g ->
      let t = solve_grid em unit g (Box2.w box) (Box2.h box) in
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
      let cell c =
        Box2.v xs.(c.c0) ys.(c.r0)
          (extent t.col_len t.col_gap c.c0 c.nc)
          (extent t.row_len t.row_gap c.r0 c.nr)
      in
      let kids = List.map (fun c -> geometry em unit c.node (cell c)) g.cells in
      let hull =
        match kids with
        | [] -> box
        | k :: ks -> List.fold_left (fun h k -> Box2.union h k.hull) k.hull ks
      in
      { hull; kids }

(* [origin hull side o] is the origin of the frame of [side] of [hull], [o]
   beyond it. *)
let origin hull (side : side) o =
  match side with
  | `Top -> P2.v (Box2.minx hull) (Box2.miny hull -. o)
  | `Bottom -> P2.v (Box2.minx hull) (Box2.maxy hull +. o)
  | `Left -> P2.v (Box2.minx hull -. o) (Box2.miny hull)
  | `Right -> P2.v (Box2.maxx hull +. o) (Box2.miny hull)

(* [placed em node geo] is each laid guide of [node] with the origin of its
   frame on the page, those of its cells first. *)
let placed em node geo =
  let add hull acc ((g : Guide.t), o) = (g, origin hull g.spec.side o) :: acc in
  List.rev
    (fold
       (fun acc n hull -> List.fold_left (add hull) acc (fst (bands em n)))
       [] node geo)

let side_length hull side = if horizontal side then Box2.w hull else Box2.h hull
let across hull side = if horizontal side then Box2.h hull else Box2.w hull

(* Laying out guides *)

(* [span shown hull g length] is the interval along its side that [g] aligns to:
   for the title of axes or headers, the hull of the panels it titles, or, on a
   vertical side, starting at the left edge of their texts, of the labelled axes
   and headers [shown]. *)
let span shown hull (g : Guide.spec) length =
  match g.kind with
  | Axis { scale; part = Scale_title side; _ } -> (
      let serves ((g' : Guide.t), _) =
        equal_side g'.spec.side side
        &&
        match g'.spec.kind with
        | Axis { scale = i; part = Ticks { labelled = true } | Header _; _ } ->
            i = scale
        | Axis _ | Legend _ | Title _ -> false
      in
      let extent ((g' : Guide.t), o) =
        if horizontal side then [ (P2.x o, P2.x o +. g'.length) ]
        else
          List.filter_map
            (function
              | Guide.Text p | Label p ->
                  let b = Guide.text_box p in
                  Some (P2.x o +. Box2.minx b, P2.x o +. Box2.maxx b)
              | Rules _ | Grid_lines _ | Bar _ | Swatch _ -> None)
            g'.elements
      in
      match List.concat_map extent (List.filter serves shown) with
      | [] -> (0., length)
      | e :: es ->
          let lo, hi =
            List.fold_left
              (fun (lo, hi) (a, b) -> (Float.min lo a, Float.max hi b))
              e es
          in
          (lo -. Box2.minx hull, hi -. Box2.minx hull))
  | Axis _ | Legend _ | Title _ -> (0., length)

(* [map f node geo] is [node] with the guides [gs] of each of its nodes
   replaced by [f hull gs], [hull] the node's data hull in [geo]. *)
let rec map f node geo =
  match node with
  | Panel p -> Panel { p with guides = f geo.hull p.guides }
  | Grid g ->
      let cell c k = { c with node = map f c.node k } in
      let cells = List.map2 cell g.cells geo.kids in
      Grid { g with cells; guides = f geo.hull g.guides }

let rec strip = function
  | Panel p -> Panel { p with guides = [] }
  | Grid g ->
      let cell c = { c with node = strip c.node } in
      Grid { g with cells = List.map cell g.cells; guides = [] }

(* [lay em cx slot node geo] is [node] with its guides laid out at the lengths
   of [geo]: axes and headers first, then the titles, which align to their
   labels, and legends. [slot x] is the specification of the guide [x], the
   length its rows wrap at if frozen, and its band if parted; a guide not
   parted takes a band of its own, numbered by its place. *)
let lay em cx slot node geo =
  let one hull shown k x =
    let spec, wrap, band = slot x in
    let length = side_length hull spec.Guide.side in
    let wrap = Option.value wrap ~default:length in
    let across = across hull spec.side in
    let span = span shown hull spec length in
    let guide = Guide.lay cx spec ~length ~across ~wrap ~span in
    { guide; band = Option.value band ~default:k }
  in
  let axis hull k x =
    let spec, _, _ = slot x in
    let l =
      if rank (Guide.tier spec.kind) > 1 then None else Some (one hull [] k x)
    in
    (k, x, l)
  in
  let axes = map (fun hull xs -> List.mapi (axis hull) xs) node geo in
  let shown =
    placed em
      (map (fun _ xs -> List.filter_map (fun (_, _, l) -> l) xs) axes geo)
      geo
  in
  let guide hull (k, x, l) =
    match l with Some l -> l | None -> one hull shown k x
  in
  map (fun hull xs -> List.map (guide hull) xs) axes geo

(* Laid-out figures *)

type panel = { id : id; box : Box2.t; projection : Coord.projection }

type t = {
  resolved : Resolved.t;
  theme : Theme.t;
  cx : Guide.cx; (* Its measurements. *)
  page : float * float;
  lpanels : (panel * Coord.t) list;
  frozen : Ticks.t array; (* Per scale of the resolved figure. *)
  guides : Guide.t list; (* On the page, in drawing order. *)
  lwarnings : warning list;
}

exception Needs of float * float

(* [solve em unit size ~final root] is the data hulls of [root] at [size] and
   the page's size. In the [final] solve, a figure too small for [root] raises
   [Needs] with a larger size. *)
let solve em unit size ~final root =
  let p = prot em root and nw, nh = natural em unit root in
  let m = em *. margin_em in
  let ow = m +. p.left +. p.right +. m and oh = m +. p.top +. p.bottom +. m in
  let page, cw, ch =
    match size with
    | Size.Panels _ -> ((ow +. nw, oh +. nh), nw, nh)
    | Size.Figure (w, h) ->
        let cw = w -. ow and ch = h -. oh in
        if final && (w < ow +. nw || h < oh +. nh) then begin
          let up x = Float.ceil (x *. 100.) /. 100. in
          raise
            (Needs (Float.max w (up (ow +. nw)), Float.max h (up (oh +. nh))))
        end;
        ((w, h), Float.max cw nw, Float.max ch nh)
  in
  (geometry em unit root (Box2.v (m +. p.left) (m +. p.top) cw ch), page)

(* [check guides] raises if figure text of [guides] lacks a glyph, and is the
   warnings of category labels that do. *)
let check guides =
  let note (g : Guide.t) acc = function
    | Guide.Text p | Label p -> (
        match Text.Layout.missing p.set with
        | [] -> acc
        | us ->
            let chars =
              String.concat ", "
                (List.map
                   (fun u -> Printf.sprintf "U+%04X" (Uchar.to_int u))
                   us)
            in
            if not p.data then
              err "layout" "%a: %a holds %s, which no face of the theme has"
                pp_id g.spec.id Text.pp p.text chars;
            ( g.spec.id,
              Format.asprintf
                "the label %a has %s, which no face of the theme has" Text.pp
                p.text chars )
            :: acc)
    | Rules _ | Grid_lines _ | Bar _ | Swatch _ -> acc
  in
  let notes =
    List.fold_left
      (fun acc (g : Guide.t) -> List.fold_left (note g) acc g.elements)
      [] guides
  in
  dedupe (List.rev notes)

let attempt ?prev theme size (r : Resolved.t) =
  let scales = Array.of_list r.scales in
  (* The root sits in a grid of one cell, which gives a panel at the root the
     data area of [Size.panels]. *)
  let root =
    let cell = { r0 = 0; c0 = 0; nr = 1; nc = 1; node = build r scales } in
    let one = [| 1. |] in
    Grid
      {
        id = Nx.Ptree.Path.root;
        widths = one;
        heights = one;
        cells = [ cell ];
        guides = [];
      }
  in
  let cx = Guide.cx ?reused:(Option.map (fun l -> l.cx) prev) theme scales in
  let em = Theme.size theme in
  let unit =
    match size with Size.Panels (w, h) -> (w, h) | Size.Figure _ -> (0., 0.)
  in
  let solve ~final node = solve em unit size ~final node in
  let lengths spec node geo =
    let add hull acc x =
      let (g : Guide.spec) = spec x in
      (g, side_length hull g.side) :: acc
    in
    fold
      (fun acc n hull -> List.fold_left (add hull) acc (guides_of n))
      [] node geo
  in
  let spec (l : laid) = l.guide.spec in
  let geo, _ = solve ~final:false (strip root) in
  let cx = Guide.with_ticks (Guide.choose cx (lengths Fun.id root geo)) cx in
  let laid = lay em cx (fun g -> (g, None, None)) root geo in
  let geo, _ = solve ~final:false laid in
  let frozen = Guide.choose cx (lengths spec laid geo) in
  let cx = Guide.with_ticks frozen cx in
  (* With the frozen ticks, every guide in a band of its own and every
     horizontal legend one entry a row, a solve gives the shortest sides the
     figure can take: rows wrapped and bands shared at those lengths only
     lengthen them in the final solve, where they stay wrapped and shared. *)
  let laid = lay em cx (fun l -> (spec l, Some 0., None)) laid geo in
  let geo, _ = solve ~final:false laid in
  let laid = part em (lay em cx (fun l -> (spec l, None, None)) laid geo) in
  let geo, page = solve ~final:true laid in
  let laid =
    lay em cx (fun l -> (spec l, Some l.guide.wrap, Some l.band)) laid geo
  in
  let by_tier (g : Guide.t) (g' : Guide.t) =
    Int.compare (rank (Guide.tier g.spec.kind)) (rank (Guide.tier g'.spec.kind))
  in
  let guides =
    List.stable_sort by_tier
      (List.map (fun (g, o) -> Guide.move o g) (placed em laid geo))
  in
  let panel acc n box =
    match n with
    | Panel { id; coord; _ } ->
        ({ id; box; projection = Coord.project coord box }, coord) :: acc
    | Grid _ -> acc
  in
  let notes = check guides in
  {
    resolved = r;
    theme;
    cx;
    page;
    lpanels = List.rev (fold panel [] laid geo);
    frozen;
    guides;
    lwarnings = r.warnings @ notes;
  }

let layout ?prev ?(theme = Theme.default) size r =
  match attempt ?prev theme size r with
  | l -> l
  | exception Needs (w, h) ->
      (* The size a figure too small names is one that lays it out. Each named
         size is strictly larger than the one tried, and an attempt whose
         choices of ticks, rows and bands are those of the attempt before lays
         out at the size that attempt named. So the attempts are at most one
         more than the changes of choice up to the widest need: at most three
         retries over 15,000 random figures. *)
      let rec fits w h =
        match attempt theme (Size.figure w h) r with
        | _ -> (w, h)
        | exception Needs (w, h) -> fits w h
      in
      let w, h = fits w h in
      err "layout" "the figure needs %.2f × %.2f pt, more than %a" w h Size.pp
        size

(* Observing and comparing *)

let size l = l.page
let panels l = List.map fst l.lpanels
let warnings l = l.lwarnings
let resolved l = l.resolved
let theme l = l.theme
let coords l = l.lpanels
let frozen l = l.frozen
let guides l = l.guides

let equal_panel (p, c) (p', c') =
  Nx.Ptree.Path.equal p.id p'.id && Box2.equal p.box p'.box && Coord.equal c c'

let equal l l' =
  Resolved.equal l.resolved l'.resolved
  && Theme.equal l.theme l'.theme
  && Float.equal (fst l.page) (fst l'.page)
  && Float.equal (snd l.page) (snd l'.page)
  && List.equal equal_panel l.lpanels l'.lpanels
  && Array.length l.frozen = Array.length l'.frozen
  && Array.for_all2 Ticks.equal l.frozen l'.frozen
  && List.equal Guide.equal l.guides l'.guides
  && List.equal Resolved.equal_warning l.lwarnings l'.lwarnings

(* Formatting *)

let pp_ticks ppf (F f, t) =
  Format.fprintf ppf "@[<hov 2>%S %a%a@ %a@]" f.name Channel.pp_kind f.kind
    (Format.pp_print_option (fun ppf p -> Format.fprintf ppf " in %a" pp_id p))
    (Resolved.panel_of f.key) Ticks.pp t

let pp ppf l =
  let w, h = l.page in
  Format.fprintf ppf "@[<v>layout %g × %g" w h;
  List.iter
    (fun (p, c) ->
      Format.fprintf ppf "@,panel %a %a %a" pp_id p.id Guide.pp_box p.box
        Coord.pp c)
    l.lpanels;
  List.iter (Format.fprintf ppf "@,%a" Guide.pp) l.guides;
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
