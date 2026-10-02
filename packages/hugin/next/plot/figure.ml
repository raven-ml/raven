(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Text = Hugin_next_text.Text
module Picture = Hugin_next_vg.Picture
module Scale = Hugin_next_kit.Scale
open Common
open Channel

(* Marks and figures *)

type rows = Rows.t
type reducer = M4 | Cells | Raster

type binding =
  | B : {
      role : ('d, 'r) Role.t;
      ch : ('d, 'r) Channel.t;
      imply : 'd Scale.t option;
      guide : bool option;
    }
      -> binding

type mark = {
  kind : string;
  reduce : reducer option;
  coord : Coord.t option;
  swatch : (rows -> Picture.t) option;
  bindings : binding list;
  draw : rows -> Picture.t;
  shape : int array;
}

type sharing = [ `Shared | `Independent ]
type side = [ `Left | `Right | `Top | `Bottom ]
type guide_kind = Axis of { grid : bool } | Legend

type guide = {
  kind : guide_kind;
  scale : string;
  side : side option;
  show : bool;
}

type t =
  | Mark of mark
  | Layer of t list
  | Grid of {
      rows : t list list;
      widths : float list option;
      heights : float list option;
    }
  | Span of { rows : int; cols : int; f : t }
  | Share of (string * sharing) list * t
  | Title of { align : Text.Layout.halign; text : Text.t; f : t }
  | Coord_sys of Coord.t * t
  | Name of string * t
  | Bind : 'a View.key * ('a -> t) -> t
  | Guide of guide

(* Two bindings of one role may read scales of two kinds. *)
let equal_imply : type d e. d Scale.t option -> e Scale.t option -> bool =
 fun i i' ->
  match (i, i') with
  | None, None -> true
  | Some s, Some s' -> (
      match Scale.equal_kind (Scale.kind s) (Scale.kind s') with
      | Some Type.Equal -> Scale.equal s s'
      | None -> false)
  | None, Some _ | Some _, None -> false

let equal_binding (B b) (B b') =
  String.equal b.role.name b'.role.name
  &&
  match Role.equal_range b.role.range b'.role.range with
  | None -> false
  | Some Type.Equal ->
      Channel.equal b.role.range b.ch b'.ch
      && equal_imply b.imply b'.imply
      && Option.equal Bool.equal b.guide b'.guide

let equal_mark (m : mark) (m' : mark) =
  String.equal m.kind m'.kind
  && Option.equal ( = ) m.reduce m'.reduce
  && Option.equal Coord.equal m.coord m'.coord
  && Array.equal Int.equal m.shape m'.shape
  && Option.equal ( == ) m.swatch m'.swatch
  && m.draw == m'.draw
  && List.equal equal_binding m.bindings m'.bindings

let equal_sharing (s : sharing) (s' : sharing) =
  match (s, s') with
  | `Shared, `Shared | `Independent, `Independent -> true
  | `Shared, `Independent | `Independent, `Shared -> false

let equal_side (s : side) (s' : side) =
  match (s, s') with
  | `Left, `Left | `Right, `Right | `Top, `Top | `Bottom, `Bottom -> true
  | _ -> false

let pp_side ppf (side : side) =
  Format.pp_print_string ppf
    (match side with
    | `Left -> "left"
    | `Right -> "right"
    | `Top -> "top"
    | `Bottom -> "bottom")

let equal_guide g g' =
  (match (g.kind, g'.kind) with
    | Axis a, Axis a' -> Bool.equal a.grid a'.grid
    | Legend, Legend -> true
    | Axis _, Legend | Legend, Axis _ -> false)
  && String.equal g.scale g'.scale
  && Option.equal equal_side g.side g'.side
  && Bool.equal g.show g'.show

let is_axis g = match g.kind with Axis _ -> true | Legend -> false

let pp_guide ppf g =
  Format.fprintf ppf "%s %S%a%s%s"
    (match g.kind with Axis _ -> "axis" | Legend -> "legend")
    g.scale
    (Format.pp_print_option (fun ppf s -> Format.fprintf ppf " %a" pp_side s))
    g.side
    (match g.kind with Axis { grid = true } -> " grid" | _ -> "")
    (if g.show then "" else " hidden")

let equal_halign (a : Text.Layout.halign) (a' : Text.Layout.halign) =
  match (a, a') with
  | `Left, `Left | `Center, `Center | `Right, `Right -> true
  | _ -> false

let rec equal f g =
  match (f, g) with
  | Mark m, Mark m' -> equal_mark m m'
  | Layer fs, Layer gs -> List.equal equal fs gs
  | Grid a, Grid b ->
      List.equal (List.equal equal) a.rows b.rows
      && Option.equal (List.equal Float.equal) a.widths b.widths
      && Option.equal (List.equal Float.equal) a.heights b.heights
  | Span a, Span b -> a.rows = b.rows && a.cols = b.cols && equal a.f b.f
  | Share (p, f), Share (p', g) ->
      List.equal
        (fun (n, s) (n', s') -> String.equal n n' && equal_sharing s s')
        p p'
      && equal f g
  | Title a, Title b ->
      equal_halign a.align b.align && Text.equal a.text b.text && equal a.f b.f
  | Coord_sys (c, f), Coord_sys (c', g) -> Coord.equal c c' && equal f g
  | Name (s, f), Name (s', g) -> String.equal s s' && equal f g
  | Bind (k, fn), Bind (k', fn') -> (
      match View.equal_key k k' with
      | Some Type.Equal -> fn == fn'
      | None -> false)
  | Guide g, Guide g' -> equal_guide g g'
  | ( ( Mark _ | Layer _ | Grid _ | Span _ | Share _ | Title _ | Coord_sys _
      | Name _ | Bind _ | Guide _ ),
      _ ) ->
      false

(* Making marks *)

(* [broadcast s s'] is the shape [s] and [s'] broadcast to, under nx's rule. *)
let broadcast s s' =
  let n = Array.length s and n' = Array.length s' in
  let m = Int.max n n' in
  let dim s n i = if i < m - n then 1 else s.(i - m + n) in
  let out = Array.make m 1 in
  let ok = ref true in
  for i = 0 to m - 1 do
    let d = dim s n i and d' = dim s' n' i in
    if d = d' || d' = 1 then out.(i) <- d
    else if d = 1 then out.(i) <- d'
    else ok := false
  done;
  if !ok then Some out else None

let role_name (B b) = b.role.name

let find_binding (role : _ Role.t) bindings =
  List.find_opt (fun b -> String.equal (role_name b) role.name) bindings

let check_ends fn bindings (a : _ Role.t) (b : _ Role.t) =
  match (find_binding a bindings, find_binding b bindings) with
  | None, Some _ -> err fn "%s is bound without %s" b.name a.name
  | Some (B ba), Some (B bb) -> (
      match (data ba.ch, data bb.ch) with
      | Some d, Some d'
        when Option.is_none (equal_kind (kind d.lift) (kind d'.lift)) ->
          err fn
            "%s and %s hold one channel of quantities and one of categories"
            a.name b.name
      | _ -> ())
  | _, None -> ()

let check_binding fn shape (B b) =
  match data b.ch with
  | None -> ()
  | Some d -> (
      (match (Role.scale b.role.use, d.spec, d.title) with
      | None, Some _, _ ->
          err fn "the role %s reads no scale but has a scale" b.role.name
      | None, _, Some _ ->
          err fn "the role %s reads no scale but has a title" b.role.name
      | _ -> ());
      let check_axis k =
        match axis_of shape k with
        | Some a -> a
        | None ->
            err fn "the role %s reads axis %d of the shape %a, which has none"
              b.role.name k pp_shape shape
      in
      match d.lift with
      | Index k -> ignore (check_axis k)
      | Dim { axis; labels; _ } -> (
          let a = check_axis axis in
          match labels with
          | Some l when Array.length l <> shape.(a) ->
              err fn "the role %s has %d labels for an axis of length %d"
                b.role.name (Array.length l) shape.(a)
          | _ -> ())
      | Num _ | Floats _ | Cat _ | Strings _ -> ())

let mark_shape fn ?(shape = [||]) bindings =
  if Array.exists (fun d -> d < 0) shape then
    err fn "the shape %a has a negative dimension" pp_shape shape;
  let shape =
    List.fold_left
      (fun shape (B b) ->
        match Option.bind (data b.ch) (fun d -> lift_shape d.lift) with
        | None -> shape
        | Some s -> (
            match broadcast shape s with
            | Some shape -> shape
            | None ->
                err fn
                  "the channel of %s, of shape %a, does not broadcast with the \
                   shape %a"
                  b.role.name pp_shape s pp_shape shape))
      (Array.copy shape) bindings
  in
  List.iter (check_binding fn shape) bindings;
  shape

let make_mark fn ~name ?reduce ?coord ?shape ?swatch bindings draw =
  let rec distinct = function
    | [] -> ()
    | b :: rest ->
        if
          List.exists (fun b' -> String.equal (role_name b') (role_name b)) rest
        then err fn "the role %s is bound twice" (role_name b);
        distinct rest
  in
  distinct bindings;
  check_ends fn bindings Role.x Role.x2;
  check_ends fn bindings Role.y Role.y2;
  let shape = mark_shape fn ?shape bindings in
  { kind = name; reduce; coord; swatch; bindings; draw; shape }

(* Composing *)

let layer fs = Layer fs

let check_weights fn = function
  | None -> ()
  | Some ws ->
      List.iter
        (fun w ->
          if not (is_pos w) then
            err fn "the weight %g is not finite and positive" w)
        ws

let grid ?widths ?heights rows =
  check_weights "grid" widths;
  check_weights "grid" heights;
  Grid { rows; widths; heights }

let span ?(rows = 1) ?(cols = 1) f =
  if rows < 1 || cols < 1 then err "span" "%d × %d is less than 1 × 1" rows cols;
  Span { rows; cols; f }

let share pairs f =
  let rec distinct = function
    | [] -> ()
    | (n, _) :: rest ->
        if List.mem_assoc n rest then
          err "share" "the scale %S is named twice" n;
        distinct rest
  in
  distinct pairs;
  Share (pairs, f)

let title ?(align = `Left) text f = Title { align; text; f }
let coord c f = Coord_sys (c, f)

let name s f =
  match s with
  | "axis" | "legend" | "panel" | "cell" ->
      err "name" "%S is the segment of generated nodes" s
  | _ -> Name (s, f)

let bind k fn = Bind (k, fn)

let axis ?side ?(grid = false) ?(show = true) scale =
  Guide { kind = Axis { grid }; scale; side; show }

let legend ?side ?(show = true) scale =
  Guide { kind = Legend; scale; side; show }
