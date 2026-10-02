(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Text = Hugin_next_text.Text
module Scale = Hugin_next_kit.Scale
module Ticks = Hugin_next_kit.Ticks
open Common
open Figure
open Arrange
open Resolved

(* Items *)

type axis_spec = {
  a_id : id;
  a_scale : int;
  a_on : [ `Axis of Role.axis | `Header of Role.axis ];
  a_side : side;
  a_guide : guide; (* Explicit, or else the default. *)
  a_labelled : bool; (* False where the next panel on its side labels it. *)
  a_category : string option; (* The panel's category, for a header. *)
}

type leaf = {
  l_id : id;
  l_coord : Coord.t;
  l_ratio : float option; (* The height of its data area over its width. *)
  l_axes : axis_spec list;
}

type legend_spec = { ls_id : id; ls_scale : int; ls_side : side; ls_bar : bool }
type track = Flex of float | Fixed

type item =
  | Leaf of leaf
  | Grid of grid
  | Heading of {
      owner : id; (* The titled node, or a facet scale's axis. *)
      align : Text.Layout.halign;
      head : Text.t;
      hside : side;
    }
  | Legend of legend_spec

and grid = {
  gid : id;
  gcols : track array;
  grows : track array;
  gcells : gcell list;
  gbody : int option; (* The cell headings align on. *)
}

and gcell = { r0 : int; c0 : int; nr : int; nc : int; it : item }

(* The guides that show a scale's ticks. *)
type use =
  | Axis_of of Role.axis
  | Header_of
  | Legend_of of { bar : bool; side : side; block : id; show : bool }

(* Guide texts *)

let tick_text (t : Ticks.tick) =
  match t.context with
  | None -> Text.v t.label
  | Some c -> Text.v (t.label ^ "\n" ^ c)

let categorical (F f) =
  match f.kind with Channel.Categories -> true | Channel.Quantities -> false

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

let category_text (F f) name =
  match Scale.domain f.scale with
  | Scale.Categories (Scale.Indices ix) -> (
      match Array.find_opt (fun (i, _) -> string_of_int i = name) ix with
      | Some (_, s) -> Text.v s
      | None -> Text.v name)
  | _ -> Text.v name

(* Building items *)

let names_of (F f) =
  match f.kind with
  | Channel.Categories -> category_names f.scale
  | Channel.Quantities -> []

(* [units s] is the length of the domain of [s] in units of its transform, or
   one if it spans none. *)
let units (F f) =
  let u = Scale.length f.scale in
  if Float.is_finite u && u > 0. then u else 1.

(* [scale_index r pid pnid on] is the scale that the guide [on] shows in the
   facet panel [pnid] of the cell [pid]. *)
let scale_index (r : Resolved.t) pid pnid on =
  let rec go i = function
    | [] -> None
    | F f :: rest ->
        let reads =
          List.exists
            (fun m ->
              Nx.Ptree.Path.equal m.m_pid pid && Role.shown_on m.m_use = Some on)
            f.members
        in
        let here =
          match f.key with
          | Panel (_, p) -> Nx.Ptree.Path.equal p pnid
          | _ -> true
        in
        if reads && here then Some i else go (i + 1) rest
  in
  go 0 r.scales

(* [guide kind (F f) ~show explicit] is the explicit guide of [f], or else the
   guide of [kind] that its marks imply or else shows by [show]. *)
let guide kind (F f) ~show = function
  | Some g -> g
  | None ->
      let show = Option.value f.guide ~default:show in
      { kind; scale = f.name; side = None; show }

let axis_guide c (F f as s) =
  guide
    (Axis { grid = false })
    s ~show:true
    (List.find_map
       (fun (_, g, _) ->
         if is_axis g && String.equal g.scale f.name then Some g else None)
       c.guides)

let default_side : [ `Axis of Role.axis | `Header of Role.axis ] -> side =
  function
  | `Axis Role.X -> `Bottom
  | `Axis Role.Y -> `Left
  | `Header Role.X -> `Top
  | `Header Role.Y -> `Right

let axis_spec r scales c pid p on =
  Option.map
    (fun i ->
      let (F f as s) = scales.(i) in
      let g = axis_guide c s in
      {
        a_id = Nx.Ptree.Path.(add (Field f.name) (add (Field "axis") p.pnid));
        a_scale = i;
        a_on = on;
        a_side = Option.value g.side ~default:(default_side on);
        a_guide = g;
        a_labelled = true;
        a_category =
          (match on with
          | `Header Role.X -> p.pfx
          | `Header Role.Y -> p.pfy
          | `Axis _ -> None);
      })
    (scale_index r pid p.pnid (on :> Role.shown))

let leaf r scales pid c p =
  let axes =
    List.filter_map
      (axis_spec r scales c pid p)
      [ `Axis Role.X; `Axis Role.Y; `Header Role.X; `Header Role.Y ]
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
    match List.find_opt (fun a -> a.a_on = on) axes with
    | Some a -> units scales.(a.a_scale)
    | None -> 1.
  in
  let (Coord.Cartesian { aspect }) = coord in
  {
    l_id = p.pnid;
    l_coord = coord;
    l_ratio =
      Option.map
        (fun k -> k *. units (`Axis Role.Y) /. units (`Axis Role.X))
        aspect;
    l_axes = axes;
  }

(* [labelled cells] is [cells] with each axis of a panel unlabelled where the
   next cell on its side holds a panel showing that axis on that side, of the
   same scale and, for a header, the same category. *)
let labelled cells =
  let next c c' (side : side) =
    match side with
    | `Bottom -> c'.c0 = c.c0 && c'.nc = c.nc && c'.r0 = c.r0 + c.nr
    | `Top -> c'.c0 = c.c0 && c'.nc = c.nc && c'.r0 + c'.nr = c.r0
    | `Left -> c'.r0 = c.r0 && c'.nr = c.nr && c'.c0 + c'.nc = c.c0
    | `Right -> c'.r0 = c.r0 && c'.nr = c.nr && c'.c0 = c.c0 + c.nc
  in
  let shared c a =
    List.exists
      (fun c' ->
        match c'.it with
        | Leaf l' when next c c' a.a_side ->
            List.exists
              (fun a' ->
                a'.a_guide.show && a'.a_scale = a.a_scale
                && equal_side a'.a_side a.a_side
                && Option.equal String.equal a'.a_category a.a_category)
              l'.l_axes
        | _ -> false)
      cells
  in
  List.map
    (fun c ->
      match c.it with
      | Leaf l ->
          let axes =
            List.map
              (fun a -> if shared c a then { a with a_labelled = false } else a)
              l.l_axes
          in
          { c with it = Leaf { l with l_axes = axes } }
      | Grid _ | Heading _ | Legend _ -> c)
    cells

(* [wrap id around body] is [body] in the middle track of a grid whose other
   tracks hold the items of [around] on their sides, the first nearest. *)
let wrap id around body =
  match around with
  | [] -> body
  | _ :: _ ->
      let on s =
        List.filter_map
          (fun (s', it) -> if equal_side s s' then Some it else None)
          around
      in
      let left = List.rev (on `Left) and right = on `Right in
      let top = List.rev (on `Top) and bottom = on `Bottom in
      let nl = List.length left and nt = List.length top in
      let cell r0 c0 it = { r0; c0; nr = 1; nc = 1; it } in
      let gcells =
        (cell nt nl body :: List.mapi (fun i it -> cell nt i it) left)
        @ List.mapi (fun i it -> cell nt (nl + 1 + i) it) right
        @ List.mapi (fun i it -> cell i nl it) top
        @ List.mapi (fun i it -> cell (nt + 1 + i) nl it) bottom
      in
      let fixed n = Array.make n Fixed in
      Grid
        {
          gid = id;
          gcols =
            Array.concat [ fixed nl; [| Flex 1. |]; fixed (List.length right) ];
          grows =
            Array.concat [ fixed nt; [| Flex 1. |]; fixed (List.length bottom) ];
          gcells;
          gbody = Some 0;
        }

let weights n = function
  | None -> Array.make n (Flex 1.)
  | Some ws -> Array.of_list (List.map (fun k -> Flex k) ws)

let blocks r =
  let rec go id s =
    match s.body with
    | Single _ ->
        let ps =
          match find_path id r.facets with
          | Some ps -> List.map (fun p -> p.pnid) ps
          | None -> [ id ]
        in
        (id, ps)
        :: List.filter_map
             (fun p ->
               if Nx.Ptree.Path.equal p id then None else Some (p, [ p ]))
             ps
    | Arr a ->
        let subs = List.map (fun c -> go c.cid c.s) a.cells in
        (id, List.concat_map (fun b -> snd (List.hd b)) subs)
        :: List.concat subs
  in
  go Nx.Ptree.Path.root r.shaped

(* [block_of r blocks key] is the block that a scale of the scope [key] has its
   legend beside: the scope's node, or the innermost block holding the cells
   that node lies in. *)
let block_of r blocks key =
  let root = Nx.Ptree.Path.root in
  match key with
  | Figure -> root
  | Cell c | Panel (_, c) | Panels_of (_, c) -> c
  | Node id when Option.is_some (find_path id blocks) -> id
  | Node id ->
      let pids =
        Option.fold ~none:[] ~some:(List.map fst) (find_path id r.nodes)
      in
      let held =
        List.concat_map
          (fun p -> Option.value ~default:[] (find_path p blocks))
          pids
      in
      let holds ps =
        List.for_all (fun p -> List.exists (Nx.Ptree.Path.equal p) ps) held
      in
      let best =
        List.fold_left
          (fun best (b, ps) ->
            if not (holds ps) then best
            else
              match best with
              | Some (_, n) when n < List.length ps -> best
              | _ -> Some (b, List.length ps))
          None blocks
      in
      Option.fold ~none:root ~some:fst best

let uses_of (r : Resolved.t) blocks =
  let legends = legends r.shaped in
  let uses (F f as s) =
    let on = List.filter_map (fun m -> Role.shown_on m.m_use) f.members in
    let legend () =
      let g =
        guide Legend s ~show:f.legend
          (List.find_map
             (fun (_, (g : guide), key) ->
               if String.equal g.scale f.name && equal_key key f.key then Some g
               else None)
             legends)
      in
      let colour =
        List.exists
          (fun m ->
            match m.m_use with
            | Encoding { map = Color; _ } -> true
            | _ -> false)
          f.members
      in
      Legend_of
        {
          bar = colour && not (categorical s);
          side = Option.value g.side ~default:`Right;
          block = block_of r blocks f.key;
          show = g.show;
        }
    in
    let when_ b u = if b then [ u () ] else [] in
    List.concat
      [
        when_ (List.mem (`Axis Role.X) on) (fun () -> Axis_of X);
        when_ (List.mem (`Axis Role.Y) on) (fun () -> Axis_of Y);
        when_
          (List.exists (function `Header _ -> true | _ -> false) on)
          (fun () -> Header_of);
        when_ (List.mem `Legend on) legend;
      ]
  in
  Array.of_list (List.map uses r.scales)

let build r scales uses =
  let legend i = function
    | Legend_of { side; block; show = true; bar } ->
        let (F f) = scales.(i) in
        let kind =
          match f.kind with
          | Channel.Quantities -> "num"
          | Channel.Categories -> "cat"
        in
        let ls_id =
          Nx.Ptree.Path.(
            v (segments block @ [ Field "legend"; Field f.name; Field kind ]))
        in
        Some
          ( block,
            (side, Legend { ls_id; ls_scale = i; ls_side = side; ls_bar = bar })
          )
    | Legend_of { show = false; _ } | Axis_of _ | Header_of -> None
  in
  let legends =
    List.concat
      (List.mapi
         (fun i us -> List.filter_map (legend i) us)
         (Array.to_list uses))
  in
  let legends_at id =
    List.filter_map
      (fun (b, l) -> if Nx.Ptree.Path.equal b id then Some l else None)
      legends
  in
  let heading owner side (align, head) =
    (side, Heading { owner; align; head; hside = side })
  in
  (* The title of a facet scale goes beside its headers. *)
  let facet_title pid c (i, on) =
    let (F f as s) = scales.(i) in
    let g = axis_guide c s in
    let side = Option.value g.side ~default:(default_side on) in
    match guide_title s None with
    | Some t when g.show ->
        let owner =
          Nx.Ptree.Path.(add (Field f.name) (add (Field "axis") pid))
        in
        [ heading owner side (`Center, t) ]
    | _ -> []
  in
  let content pid c =
    let one p = Leaf (leaf r scales pid c p) in
    match find_path pid r.facets with
    | None -> one { pnid = pid; pfy = None; pfx = None }
    | Some [ p ] when Nx.Ptree.Path.equal p.pnid pid -> one p
    | Some ps ->
        (* A facet panel is a block of its own, beside which the legends of its
           own scales go. *)
        let one p = wrap p.pnid (legends_at p.pnid) (one p) in
        let fx = scale_index r pid pid (`Header Role.X)
        and fy = scale_index r pid pid (`Header Role.Y) in
        let names = Option.fold ~none:[] ~some:(fun i -> names_of scales.(i)) in
        let xs = names fx and ys = names fy in
        let pos l = function
          | None -> 0
          | Some c ->
              let rec find k = function
                | [] -> 0
                | c' :: rest ->
                    if String.equal c c' then k else find (k + 1) rest
              in
              find 0 l
        in
        let nx = max 1 (List.length xs) and ny = max 1 (List.length ys) in
        let wrap_at =
          Option.bind fx (fun i ->
              let (F f) = scales.(i) in
              match f.kind with
              | Channel.Categories -> Scale.wrap f.scale
              | Channel.Quantities -> None)
        in
        let ncols, nrows, at =
          match wrap_at with
          | Some w ->
              let w = min w nx in
              ( w,
                (nx + w - 1) / w,
                fun p -> (pos xs p.pfx / w, pos xs p.pfx mod w) )
          | None -> (nx, ny, fun p -> (pos ys p.pfy, pos xs p.pfx))
        in
        let cell p =
          let r0, c0 = at p in
          { r0; c0; nr = 1; nc = 1; it = one p }
        in
        let facets = [ (fx, `Header Role.X); (fy, `Header Role.Y) ] in
        let titles =
          List.concat_map
            (fun (i, on) ->
              Option.fold ~none:[] ~some:(fun i -> facet_title pid c (i, on)) i)
            facets
        in
        wrap pid titles
          (Grid
             {
               gid = pid;
               gcols = Array.make ncols (Flex 1.);
               grows = Array.make nrows (Flex 1.);
               gcells = labelled (List.map cell ps);
               gbody = None;
             })
  in
  let rec block id s =
    let body =
      match s.body with
      | Single c -> content id c
      | Arr a ->
          let cell (c : cell) =
            {
              r0 = c.row;
              c0 = c.col;
              nr = c.rows;
              nc = c.cols;
              it = block c.cid c.s;
            }
          in
          Grid
            {
              gid = a.aid;
              gcols = weights a.ncols a.widths;
              grows = weights a.nrows a.heights;
              gcells = labelled (List.map cell a.cells);
              gbody = None;
            }
    in
    let titles = List.rev_map (heading id `Top) s.titles in
    wrap id titles (wrap id (legends_at id) body)
  in
  block Nx.Ptree.Path.root r.shaped
