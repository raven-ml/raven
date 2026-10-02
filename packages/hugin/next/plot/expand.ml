(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Text = Hugin_next_text.Text
open Common
open Figure

type node = { id : id; n : enode }

and enode =
  | E_mark of { mark : mark; order : int }
  | E_layer of node list
  | E_grid of {
      rows : node list list;
      widths : float list option;
      heights : float list option;
    }
  | E_span of { rows : int; cols : int; f : node }
  | E_share of (string * sharing) list * node
  | E_title of Text.Layout.halign * Text.t * node
  | E_coord of Coord.t * node
  | E_axis of { side : side option; grid : bool; show : bool; scale : string }
  | E_legend of { side : side option; show : bool; scale : string }

type read = Read : View.ident * 'a View.sort -> read

type expansion = {
  view : View.t;
  mutable reads : read list;
  mutable marks : int;
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

let rec force st = function
  | Bind (k, fn) ->
      read_key st k;
      force st (fn (View.get k st.view))
  | Title t -> Title { t with f = force st t.f }
  | Coord_sys (c, f) -> Coord_sys (c, force st f)
  | Share (p, f) -> Share (p, force st f)
  | Span s -> Span { s with f = force st s.f }
  | Name (s, f) -> Name (s, force st f)
  | (Mark _ | Layer _ | Grid _ | Axis _ | Legend _) as f -> f

let rec names = function
  | Name (s, f) -> s :: names f
  | Title { f; _ } | Coord_sys (_, f) | Share (_, f) | Span { f; _ } -> names f
  | Mark _ | Layer _ | Grid _ | Bind _ | Axis _ | Legend _ -> []

let rec split n l =
  match l with
  | x :: l when n > 0 ->
      let a, b = split (n - 1) l in
      (x :: a, b)
  | _ -> ([], l)

let rec expand st id f =
  let n =
    match f with
    | Mark mark ->
        st.marks <- st.marks + 1;
        E_mark { mark; order = st.marks }
    | Layer fs -> E_layer (children st id fs)
    | Grid g ->
        let rec regroup rows cells =
          match rows with
          | [] -> []
          | row :: rows ->
              let row, cells = split (List.length row) cells in
              row :: regroup rows cells
        in
        let cells = children st id (List.concat g.rows) in
        E_grid
          {
            rows = regroup g.rows cells;
            widths = g.widths;
            heights = g.heights;
          }
    | Span s -> E_span { rows = s.rows; cols = s.cols; f = expand st id s.f }
    | Share (p, f) -> E_share (p, expand st id f)
    | Title t -> E_title (t.align, t.text, expand st id t.f)
    | Coord_sys (c, f) -> E_coord (c, expand st id f)
    | Name (_, f) -> (expand st id f).n
    | Bind _ -> (expand st id (force st f)).n
    | Axis { side; grid; show; scale } -> E_axis { side; grid; show; scale }
    | Legend { side; show; scale } -> E_legend { side; show; scale }
  in
  { id; n }

and children st parent fs =
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
  List.map (fun (id, f) -> expand st id f) named
