(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Scale = Hugin_next_kit.Scale
module Text = Hugin_next_text.Text
open Common
open Channel
open Figure
open Expand

(* Scopes *)

type key =
  | Figure
  | Node of id
  | Cell of id
  | Panels_of of id * id (* Per panel of the mark, in the cell. *)
  | Panel of id * id (* The mark, the facet panel. *)

let equal_key k k' =
  match (k, k') with
  | Figure, Figure -> true
  | Node a, Node b | Cell a, Cell b -> Nx.Ptree.Path.equal a b
  | Panels_of (a, b), Panels_of (a', b') | Panel (a, b), Panel (a', b') ->
      Nx.Ptree.Path.equal a a' && Nx.Ptree.Path.equal b b'
  | _ -> false

type shares = (string * key) list

type env = {
  shares : shares; (* Innermost first. *)
  pending : string list; (* Independent names awaiting the node's children. *)
  cell : id; (* The innermost grid cell, or the root. *)
}

let key_of (env : env) name =
  match List.assoc_opt name env.shares with
  | Some k -> k
  | None -> if Role.by_cell name then Cell env.cell else Figure

(* Contents *)

type occ = {
  mid : id;
  mark : mark;
  order : int;
  shares : shares;
  per_panel : string list;
}

type axis_item = {
  gid : id;
  side : side option;
  grid : bool;
  show : bool;
  scale : string;
}

type legend_item = {
  lid : id;
  lside : side option;
  lshow : bool;
  lscale : string;
  lkey : key; (* The scope of the scales it stands for. *)
}

type guide_item = G_axis of axis_item | G_legend of legend_item

let equal_axis a a' =
  Option.equal equal_side a.side a'.side
  && Bool.equal a.grid a'.grid && Bool.equal a.show a'.show
  && String.equal a.scale a'.scale

let equal_legend l l' =
  Option.equal equal_side l.lside l'.lside
  && Bool.equal l.lshow l'.lshow
  && String.equal l.lscale l'.lscale

type content = {
  occs : occ list;
  guides : guide_item list;
  coords : (id * Coord.t) list;
  held : (id * shares) list;
      (* The nodes that lie in the content, each with the scopes a channel there
         reads. *)
}

type shaped = { titles : (Text.Layout.halign * Text.t) list; body : body }
and body = Single of content | Arr of arr

and arr = {
  aid : id;
  nrows : int;
  ncols : int;
  cells : cell list;
  widths : float list option;
  heights : float list option;
}

and cell = {
  row : int;
  col : int;
  rows : int;
  cols : int;
  cid : id;
  s : shaped;
}

let single content = { titles = []; body = Single content }
let no_content = { occs = []; guides = []; coords = []; held = [] }

let rec map_contents f s =
  match s.body with
  | Single c -> { s with body = Single (f c) }
  | Arr a ->
      let cells =
        List.map (fun cell -> { cell with s = map_contents f cell.s }) a.cells
      in
      { s with body = Arr { a with cells } }

let add_coord c = map_contents (fun ct -> { ct with coords = c :: ct.coords })

(* [hold node s] places [node] in every content of [s]. *)
let hold node = map_contents (fun c -> { c with held = node :: c.held })

(* [rename old cid s] is [s] with the scopes of the cell [old] those of the cell
   [cid], where a layer broadcasts [old] into [cid]. *)
let rename old cid =
  let key = function
    | Cell c when Nx.Ptree.Path.equal c old -> Cell cid
    | k -> k
  in
  let shares = List.map (fun (n, k) -> (n, key k)) in
  let guide = function
    | G_legend l -> G_legend { l with lkey = key l.lkey }
    | G_axis _ as g -> g
  in
  map_contents (fun c ->
      {
        c with
        occs =
          List.map (fun (o : occ) -> { o with shares = shares o.shares }) c.occs;
        guides = List.map guide c.guides;
        held = List.map (fun (id, s) -> (id, shares s)) c.held;
      })

(* Arranging *)

let rec core n =
  match n.n with
  | E_title (_, _, f) | E_coord (_, f) | E_share (_, f) | E_span { f; _ } ->
      core f
  | E_mark _ | E_layer _ | E_grid _ | E_axis _ | E_legend _ -> n

let rec span_of n =
  match n.n with
  | E_span s -> (s.rows, s.cols)
  | E_title (_, _, f) | E_coord (_, f) | E_share (_, f) -> span_of f
  | E_mark _ | E_layer _ | E_grid _ | E_axis _ | E_legend _ -> (1, 1)

(* [scale_name b] is the name of the scale the channel of [b] reads, if any. *)
let scale_name (B b) =
  match (data b.ch, Role.scale b.role.use) with
  | Some d, Some default ->
      Some (Option.value ~default (Option.bind d.spec Scale.name))
  | _ -> None

(* [reads n] is the scales the marks under [n] read, each with whether a
   position or facet role reads it. *)
let rec reads n =
  match n.n with
  | E_mark { mark; _ } ->
      let placing (B b) =
        match b.role.use with
        | Position _ | Facet _ -> true
        | Encoding _ | Value -> false
      in
      List.filter_map
        (fun bd -> Option.map (fun s -> (s, placing bd)) (scale_name bd))
        mark.bindings
  | E_layer cs -> List.concat_map reads cs
  | E_grid g -> List.concat_map (List.concat_map reads) g.rows
  | E_span { f; _ } | E_share (_, f) | E_title (_, _, f) | E_coord (_, f) ->
      reads f
  | E_axis _ | E_legend _ -> []

let check_share node pairs f =
  let reads = reads f in
  List.iter
    (fun (name, (s : sharing)) ->
      if not (List.mem_assoc name reads) then
        err "resolve" "%a shares the scale %S, which nothing under it reads"
          pp_id node.id name;
      match (s, (core f).n) with
      | `Independent, E_layer _
        when List.exists (fun (n, p) -> String.equal n name && p) reads ->
          err "resolve"
            "%a makes the scale %S independent per layer child, but a position \
             or facet reads it"
            pp_id node.id name
      | _ -> ())
    pairs

let share_env (env : env) node pairs =
  List.fold_left
    (fun (env : env) (name, (s : sharing)) ->
      match s with
      | `Shared ->
          {
            env with
            shares = (name, Node node.id) :: env.shares;
            pending =
              List.filter (fun n -> not (String.equal n name)) env.pending;
          }
      | `Independent -> { env with pending = name :: env.pending })
    env pairs

let child_env (env : env) ~cell id =
  let key name = if cell && Role.by_cell name then Cell id else Node id in
  let shares = List.map (fun n -> (n, key n)) env.pending @ env.shares in
  { shares; pending = []; cell = (if cell then id else env.cell) }

let equal_title (a, t) (a', t') = equal_halign a a' && Text.equal t t'

let concat cs =
  List.fold_right
    (fun c acc ->
      {
        occs = c.occs @ acc.occs;
        guides = c.guides @ acc.guides;
        coords = c.coords @ acc.coords;
        held = c.held @ acc.held;
      })
    cs no_content

(* [layer_shaped lid shares children] is the layer [lid] of [children], [shares]
   the scopes of the layer. *)
let rec layer_shaped lid shares (children : shaped list) =
  (* Titles lift one at a time, outermost first: equal ones lift as one. *)
  let rec merge ts ts' =
    match (ts, ts') with
    | [], ts | ts, [] -> ts
    | t :: ts, t' :: ts' ->
        if equal_title t t' then t :: merge ts ts'
        else err "resolve" "the children of %a have different titles" pp_id lid
  in
  let titles = List.fold_left (fun acc s -> merge acc s.titles) [] children in
  let children = List.map (fun s -> { s with titles = [] }) children in
  let arrs =
    List.filter_map
      (fun s -> match s.body with Arr a -> Some a | Single _ -> None)
      children
  in
  match arrs with
  | [] ->
      let contents =
        List.filter_map
          (fun s -> match s.body with Single c -> Some c | Arr _ -> None)
          children
      in
      { titles; body = Single (concat contents) }
  | first :: _ ->
      { titles; body = Arr (broadcast lid shares children arrs first) }

and broadcast lid shares children arrs first =
  let dim d d' =
    if d = d' || d' = 1 then d
    else if d = 1 then d'
    else
      err "resolve" "the arrangements of the children of %a do not broadcast"
        pp_id lid
  in
  let nrows = List.fold_left (fun d a -> dim d a.nrows) 1 arrs in
  let ncols = List.fold_left (fun d a -> dim d a.ncols) 1 arrs in
  (* Under nx's rule a figure of one cell repeats into none beside a grid of
     none, which would drop it. *)
  let has_cells s =
    match s.body with Single _ -> true | Arr a -> a.nrows * a.ncols > 0
  in
  if nrows * ncols = 0 && List.exists has_cells children then
    err "resolve" "a child of %a with cells is layered with a grid of none"
      pp_id lid;
  let layout a = List.map (fun c -> (c.row, c.col, c.rows, c.cols)) a.cells in
  let spanned a = List.exists (fun c -> c.rows > 1 || c.cols > 1) a.cells in
  let template =
    match List.find_opt spanned arrs with
    | Some a ->
        List.iter
          (fun a' ->
            if layout a' <> layout a then
              err "resolve"
                "a grid with spans under %a broadcasts with a grid of other \
                 cells"
                pp_id lid)
          arrs;
        a
    | None -> (
        match
          List.find_opt (fun a -> a.nrows = nrows && a.ncols = ncols) arrs
        with
        | Some a -> a
        | None -> first)
  in
  let cell_at a r c =
    let r = if a.nrows = 1 then 0 else r and c = if a.ncols = 1 then 0 else c in
    List.find (fun cell -> cell.row = r && cell.col = c) a.cells
  in
  let positions =
    if spanned template then layout template
    else
      List.concat
        (List.init nrows (fun r -> List.init ncols (fun c -> (r, c, 1, 1))))
  in
  let cells =
    List.mapi
      (fun k (row, col, rows, cols) ->
        let cid = Nx.Ptree.Path.(add (Index k) (add (Field "cell") lid)) in
        let parts =
          List.map
            (fun s ->
              match s.body with
              | Single _ -> s
              | Arr a ->
                  let cell = cell_at a row col in
                  rename cell.cid cid cell.s)
            children
        in
        let s = hold (cid, shares) (layer_shaped cid shares parts) in
        { row; col; rows; cols; cid; s })
      positions
  in
  let same = template.nrows = nrows && template.ncols = ncols in
  {
    aid = lid;
    nrows;
    ncols;
    cells;
    widths = (if same then template.widths else None);
    heights = (if same then template.heights else None);
  }

let rec arrange nodes (env : env) ~in_cell n =
  let record () = nodes := n.id :: !nodes in
  let here = [ (n.id, env.shares) ] in
  match n.n with
  | E_mark { mark; order } ->
      record ();
      let occ =
        {
          mid = n.id;
          mark;
          order;
          shares = env.shares;
          per_panel = env.pending;
        }
      in
      single { no_content with occs = [ occ ]; held = here }
  | E_axis { side; grid; show; scale } ->
      record ();
      let a = { gid = n.id; side; grid; show; scale } in
      single { no_content with guides = [ G_axis a ]; held = here }
  | E_legend { side; show; scale } ->
      record ();
      let l =
        {
          lid = n.id;
          lside = side;
          lshow = show;
          lscale = scale;
          lkey = key_of env scale;
        }
      in
      single { no_content with guides = [ G_legend l ]; held = here }
  | E_title (align, t, f) ->
      let s = arrange nodes env ~in_cell f in
      { s with titles = (align, t) :: s.titles }
  | E_coord (c, f) -> add_coord (n.id, c) (arrange nodes env ~in_cell f)
  | E_share (pairs, f) ->
      check_share n pairs f;
      arrange nodes (share_env env n pairs) ~in_cell f
  | E_span { f; _ } ->
      if not in_cell then
        err "resolve" "%a spans cells outside a grid" pp_id n.id;
      arrange nodes env ~in_cell f
  | E_layer [] ->
      record ();
      single { no_content with held = here }
  | E_layer cs ->
      record ();
      (* A layer lies where its children lie, reading the scopes a channel of
         each child reads there. *)
      layer_shaped n.id env.shares
        (List.map
           (fun c ->
             let env = child_env env ~cell:false c.id in
             arrange nodes env ~in_cell:false c |> hold (n.id, env.shares))
           cs)
  | E_grid g ->
      record ();
      arrange_grid nodes env n g.rows g.widths g.heights

and arrange_grid nodes (env : env) n rows widths heights =
  let nrows = List.length rows in
  let covered = Hashtbl.create 16 in
  let cells = ref [] in
  List.iteri
    (fun r row ->
      let col = ref 0 in
      List.iter
        (fun c ->
          while Hashtbl.mem covered (r, !col) do
            incr col
          done;
          let rs, cs = span_of c in
          if r + rs > nrows then
            err "resolve" "the span %a reaches past the last row" pp_id c.id;
          for i = r to r + rs - 1 do
            for j = !col to !col + cs - 1 do
              if Hashtbl.mem covered (i, j) then
                err "resolve" "the span %a covers another cell" pp_id c.id;
              Hashtbl.add covered (i, j) ()
            done
          done;
          let env = child_env env ~cell:true c.id in
          let s =
            arrange nodes env ~in_cell:true c |> hold (n.id, env.shares)
          in
          cells :=
            { row = r; col = !col; rows = rs; cols = cs; cid = c.id; s }
            :: !cells;
          col := !col + cs)
        row)
    rows;
  let width r =
    let rec count j = if Hashtbl.mem covered (r, j) then count (j + 1) else j in
    count 0
  in
  let ncols = if nrows = 0 then 0 else width 0 in
  for r = 0 to nrows - 1 do
    if width r <> ncols then
      err "resolve" "the rows of %a cover different numbers of columns" pp_id
        n.id
  done;
  if Hashtbl.length covered <> nrows * ncols then
    err "resolve" "the rows of %a cover different numbers of columns" pp_id n.id;
  let check what ws k =
    match ws with
    | Some ws when List.length ws <> k ->
        err "resolve" "%a has %d %s for %d tracks" pp_id n.id (List.length ws)
          what k
    | _ -> ()
  in
  check "widths" widths ncols;
  check "heights" heights nrows;
  {
    titles = [];
    body =
      Arr { aid = n.id; nrows; ncols; cells = List.rev !cells; widths; heights };
  }

let panels root s =
  let rec go acc pid s =
    match s.body with
    | Single c -> (pid, c) :: acc
    | Arr a ->
        List.fold_left (fun acc cell -> go acc cell.cid cell.s) acc a.cells
  in
  List.rev (go [] root s)
