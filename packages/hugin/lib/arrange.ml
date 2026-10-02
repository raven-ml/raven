(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Scale = Hugin_kit.Scale
module Text = Hugin_text.Text
open Common
open Channel
open Figure

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

type content = {
  occs : occ list;
  guides : (id * guide * key) list; (* With the scope of the scale it names. *)
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
  map_contents (fun c ->
      {
        c with
        occs =
          List.map (fun (o : occ) -> { o with shares = shares o.shares }) c.occs;
        guides = List.map (fun (id, g, k) -> (id, g, key k)) c.guides;
        held = List.map (fun (id, s) -> (id, shares s)) c.held;
      })

(* Arranging *)

type read = Read : View.ident * 'a View.sort -> read

(* The state of a walk: the view binds read, the keys they read, the marks met
   and every node met, latest first. *)
type st = {
  view : View.t;
  mutable reads : read list;
  mutable marks : int;
  mutable nodes : id list;
}

let read_key st (k : _ View.key) =
  match
    List.find_opt (fun (Read (i, _)) -> View.equal_ident i k.ident) st.reads
  with
  | None -> st.reads <- Read (k.ident, k.sort) :: st.reads
  | Some (Read (_, s)) -> (
      match View.equal_sort s k.sort with
      | Some _ -> ()
      | None ->
          err "resolve" "the figure reads two keys %a of different sorts"
            View.pp_ident k.ident)

(* [force st f] is [f] with the binds at its head evaluated, through
   wrappers. *)
let rec force st = function
  | Bind (k, fn) ->
      read_key st k;
      force st (fn (View.get k st.view))
  | Title t -> Title { t with f = force st t.f }
  | Coord_sys (c, f) -> Coord_sys (c, force st f)
  | Share (p, f) -> Share (p, force st f)
  | Span s -> Span { s with f = force st s.f }
  | Name (s, f) -> Name (s, force st f)
  | (Mark _ | Layer _ | Grid _ | Guide _) as f -> f

(* [names f] is the names the wrappers at the head of the forced [f] give. *)
let rec names = function
  | Name (s, f) -> s :: names f
  | Title { f; _ } | Coord_sys (_, f) | Share (_, f) | Span { f; _ } -> names f
  | Mark _ | Layer _ | Grid _ | Bind _ | Guide _ -> []

(* [children st parent fs] is [fs] forced, each with its id. *)
let children st parent fs =
  let named =
    List.mapi
      (fun i f ->
        let f = force st f in
        match names f with
        | [] -> (Nx.Ptree.Path.add (Index i) parent, f)
        | [ s ] -> (Nx.Ptree.Path.add (Field s) parent, f)
        | _ -> err "resolve" "the child %d of %a has two names" i pp_id parent)
      fs
  in
  let rec distinct = function
    | [] -> ()
    | (id, _) :: rest ->
        if List.exists (fun (id', _) -> Nx.Ptree.Path.equal id id') rest then
          err "resolve" "two children of %a are named %a" pp_id parent pp_id id;
        distinct rest
  in
  distinct named;
  named

let rec span_of = function
  | Span s -> (s.rows, s.cols)
  | Title { f; _ } | Coord_sys (_, f) | Share (_, f) | Name (_, f) -> span_of f
  | Mark _ | Layer _ | Grid _ | Bind _ | Guide _ -> (1, 1)

let rec core = function
  | Title { f; _ }
  | Coord_sys (_, f)
  | Share (_, f)
  | Span { f; _ }
  | Name (_, f) ->
      core f
  | (Mark _ | Layer _ | Grid _ | Bind _ | Guide _) as f -> f

(* [scale_name b] is the name of the scale the channel of [b] reads, if any. *)
let scale_name (B b) =
  match (data b.ch, Role.scale b.role.use) with
  | Some d, Some default ->
      Some (Option.value ~default (Option.bind d.spec Scale.name))
  | _ -> None

let panels s =
  let rec go acc pid s =
    match s.body with
    | Single c -> (pid, c) :: acc
    | Arr a ->
        List.fold_left (fun acc cell -> go acc cell.cid cell.s) acc a.cells
  in
  List.rev (go [] Nx.Ptree.Path.root s)

let legends s =
  List.concat_map
    (fun (_, c) -> List.filter (fun (_, g, _) -> not (is_axis g)) c.guides)
    (panels s)

(* [reads s] is the scales the marks of [s] read, each with whether a position
   or facet role reads it. *)
let reads s =
  let read (B b as bd) =
    let placing = Role.shown_on b.role.use <> Some `Legend in
    Option.map (fun s -> (s, placing)) (scale_name bd)
  in
  List.concat_map
    (fun (_, c) ->
      List.concat_map (fun o -> List.filter_map read o.mark.bindings) c.occs)
    (panels s)

(* [check_share id pairs f s] checks the shares [pairs] of the node [id] over
   [f], arranged as [s]. *)
let check_share id pairs f s =
  let reads = reads s in
  List.iter
    (fun (name, (sh : sharing)) ->
      if not (List.mem_assoc name reads) then
        err "resolve" "%a shares the scale %S, which nothing under it reads"
          pp_id id name;
      match sh with
      | `Independent
        when (match core f with Layer _ -> true | _ -> false)
             && List.exists (fun (n, p) -> String.equal n name && p) reads ->
          err "resolve"
            "%a makes the scale %S independent per layer child, but a position \
             or facet reads it"
            pp_id id name
      | _ -> ())
    pairs

(* A position or facet shared by a grid cell is that cell's scope: the scope its
   grid gives the cell when it keeps the name per cell, and the one a layer
   broadcasting the grid renames. *)
let share_env (env : env) id pairs =
  let key name =
    if Role.by_cell name && Nx.Ptree.Path.equal env.cell id then Cell id
    else Node id
  in
  List.fold_left
    (fun (env : env) (name, (s : sharing)) ->
      match s with
      | `Shared ->
          {
            env with
            shares = (name, key name) :: env.shares;
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

(* [walk st env ~in_cell id f] is the forced [f] of id [id] arranged in [env];
   wrappers have their child's id. *)
let rec walk st (env : env) ~in_cell id f =
  let record () = st.nodes <- id :: st.nodes in
  let here = [ (id, env.shares) ] in
  match f with
  | Mark mark ->
      record ();
      st.marks <- st.marks + 1;
      let occ =
        {
          mid = id;
          mark;
          order = st.marks;
          shares = env.shares;
          per_panel = env.pending;
        }
      in
      single { no_content with occs = [ occ ]; held = here }
  | Guide g ->
      record ();
      let guides = [ (id, g, key_of env g.scale) ] in
      single { no_content with guides; held = here }
  | Title t ->
      let s = walk st env ~in_cell id t.f in
      { s with titles = (t.align, t.text) :: s.titles }
  | Coord_sys (c, f) -> add_coord (id, c) (walk st env ~in_cell id f)
  | Share (pairs, f) ->
      let s = walk st (share_env env id pairs) ~in_cell id f in
      check_share id pairs f s;
      s
  | Span { f; _ } ->
      if not in_cell then err "resolve" "%a spans cells outside a grid" pp_id id;
      walk st env ~in_cell id f
  | Name (_, f) -> walk st env ~in_cell id f
  | Bind _ -> walk st env ~in_cell id (force st f)
  | Layer [] ->
      record ();
      single { no_content with held = here }
  | Layer fs ->
      record ();
      (* A layer lies where its children lie, reading the scopes a channel of
         each child reads there. *)
      layer_shaped id env.shares
        (List.map
           (fun (cid, f) ->
             let env = child_env env ~cell:false cid in
             walk st env ~in_cell:false cid f |> hold (id, env.shares))
           (children st id fs))
  | Grid g ->
      record ();
      walk_grid st env id g.rows g.widths g.heights

and walk_grid st (env : env) id rows widths heights =
  let nrows = List.length rows in
  let covered = Hashtbl.create 16 in
  let cells = ref [] in
  let rec split n l =
    match l with
    | x :: l when n > 0 ->
        let a, b = split (n - 1) l in
        (x :: a, b)
    | _ -> ([], l)
  in
  let rec regroup rows cs =
    match rows with
    | [] -> []
    | row :: rows ->
        let row, cs = split (List.length row) cs in
        row :: regroup rows cs
  in
  let rows = regroup rows (children st id (List.concat rows)) in
  List.iteri
    (fun r row ->
      let col = ref 0 in
      List.iter
        (fun (cid, c) ->
          while Hashtbl.mem covered (r, !col) do
            incr col
          done;
          let rs, cs = span_of c in
          if r + rs > nrows then
            err "resolve" "the span %a reaches past the last row" pp_id cid;
          for i = r to r + rs - 1 do
            for j = !col to !col + cs - 1 do
              if Hashtbl.mem covered (i, j) then
                err "resolve" "the span %a covers another cell" pp_id cid;
              Hashtbl.add covered (i, j) ()
            done
          done;
          let env = child_env env ~cell:true cid in
          let s = walk st env ~in_cell:true cid c |> hold (id, env.shares) in
          cells :=
            { row = r; col = !col; rows = rs; cols = cs; cid; s } :: !cells;
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
      err "resolve" "the rows of %a cover different numbers of columns" pp_id id
  done;
  if Hashtbl.length covered <> nrows * ncols then
    err "resolve" "the rows of %a cover different numbers of columns" pp_id id;
  let check what ws k =
    match ws with
    | Some ws when List.length ws <> k ->
        err "resolve" "%a has %d %s for %d tracks" pp_id id (List.length ws)
          what k
    | _ -> ()
  in
  check "widths" widths ncols;
  check "heights" heights nrows;
  {
    titles = [];
    body =
      Arr { aid = id; nrows; ncols; cells = List.rev !cells; widths; heights };
  }

type t = { shaped : shaped; order : id list; reads : read list }

let arrange view figure =
  let st = { view; reads = []; marks = 0; nodes = [] } in
  let f = force st figure in
  (match names f with
  | _ :: _ :: _ -> err "resolve" "the root has two names"
  | _ -> ());
  let root = Nx.Ptree.Path.root in
  let shaped =
    walk st { shares = []; pending = []; cell = root } ~in_cell:false root f
  in
  { shaped; order = List.rev st.nodes; reads = st.reads }
