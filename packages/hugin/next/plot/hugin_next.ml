(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module P2 = Hugin_next_gg.P2
module Box2 = Hugin_next_gg.Box2
module Affine = Hugin_next_gg.Affine
module Path = Hugin_next_gg.Path
module Stroke = Hugin_next_gg.Stroke
module Color = Hugin_next_gg.Color
module Font = Hugin_next_font.Font
module Text = Hugin_next_text.Text
module Picture = Hugin_next_vg.Picture
module Renderable = Hugin_next_vg.Renderable
module Locale = Hugin_next_kit.Locale
module Scale = Hugin_next_kit.Scale
module Scheme = Hugin_next_kit.Scheme
module Symbol = Hugin_next_kit.Symbol
module Curve = Hugin_next_kit.Curve
module Time = Hugin_next_kit.Time
module Ticks = Hugin_next_kit.Ticks

let err fn fmt =
  Format.kasprintf (fun s -> invalid_arg ("Hugin_next." ^ fn ^ ": " ^ s)) fmt

(* The stages after [resolve] come with the layout and drawing of figures. *)
let unimplemented fn = failwith ("Hugin_next." ^ fn ^ ": not implemented")
let is_pos x = Float.is_finite x && x > 0.

let pp_shape ppf s =
  Format.fprintf ppf "[%a]"
    (Format.pp_print_seq
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       Format.pp_print_int)
    (Array.to_seq s)

(* Ids and warnings *)

type id = Nx.Ptree.Path.t
type warning = id * string

let pp_id ppf id =
  match Nx.Ptree.Path.segments id with
  | [] -> Format.pp_print_string ppf "root"
  | _ -> Nx.Ptree.Path.pp ppf id

let pp_warning ppf (id, msg) = Format.fprintf ppf "@[%a: %s@]" pp_id id msg

let compare_seg (s : Nx.Ptree.Path.seg) (s' : Nx.Ptree.Path.seg) =
  match (s, s') with
  | Field a, Field b -> String.compare a b
  | Field _, Index _ -> -1
  | Index _, Field _ -> 1
  | Index i, Index j -> Int.compare i j

let compare_id id id' =
  List.compare compare_seg
    (Nx.Ptree.Path.segments id)
    (Nx.Ptree.Path.segments id')

(* Tensors compare physically, whatever their dtypes. *)
let equal_tensor : type a b c d. (a, b) Nx.t -> (c, d) Nx.t -> bool =
 fun x y ->
  match Nx_dtype.equal_witness (Nx.dtype x) (Nx.dtype y) with
  | Some Type.Equal -> x == y
  | None -> false

let equal_strings = Array.equal String.equal

(* Coordinate systems *)

module Coord = struct
  type t = Cartesian of { aspect : float option }

  let cartesian ?aspect () =
    (match aspect with
    | Some a when not (is_pos a) ->
        err "Coord.cartesian" "aspect %g is not finite and positive" a
    | _ -> ());
    Cartesian { aspect }

  let equal (Cartesian c) (Cartesian c') =
    Option.equal Float.equal c.aspect c'.aspect

  let pp ppf (Cartesian { aspect }) =
    match aspect with
    | None -> Format.pp_print_string ppf "cartesian"
    | Some a -> Format.fprintf ppf "cartesian ~aspect:%g" a

  (* A cartesian projection maps the unit square onto its box, y up. *)
  type projection = Box of Box2.t

  let project (Cartesian _) box = Box box

  let point (Box b) x y =
    P2.v (Box2.minx b +. (x *. Box2.w b)) (Box2.maxy b -. (y *. Box2.h b))

  let invert (Box b) pt =
    let w = Box2.w b and h = Box2.h b in
    if w = 0. || h = 0. then None
    else Some ((P2.x pt -. Box2.minx b) /. w, (Box2.maxy b -. P2.y pt) /. h)
end

(* Views *)

module View = struct
  type _ sort =
    | Number : float sort
    | Choice : string sort
    | Interval : (float * float) option sort
    | Zoom : 'd Scale.kind -> ('d * 'd) option sort

  (* A zoom key is named by its scale and node, so no user key names it. *)
  type ident = User of string | Zoom_of of { scale : string; at : id }
  type 'a key = { ident : ident; sort : 'a sort; init : 'a }

  let compare_ident i i' =
    match (i, i') with
    | User n, User n' -> String.compare n n'
    | User _, Zoom_of _ -> -1
    | Zoom_of _, User _ -> 1
    | Zoom_of z, Zoom_of z' ->
        let c = String.compare z.scale z'.scale in
        if c <> 0 then c else compare_id z.at z'.at

  let equal_ident i i' = compare_ident i i' = 0

  let pp_ident ppf = function
    | User n -> Format.fprintf ppf "%S" n
    | Zoom_of { scale; at } -> Format.fprintf ppf "zoom %S at %a" scale pp_id at

  let equal_sort : type a b. a sort -> b sort -> (a, b) Type.eq option =
   fun s s' ->
    match (s, s') with
    | Number, Number -> Some Type.Equal
    | Choice, Choice -> Some Type.Equal
    | Interval, Interval -> Some Type.Equal
    | Zoom k, Zoom k' -> (
        match Scale.equal_kind k k' with
        | Some Type.Equal -> Some Type.Equal
        | None -> None)
    | _ -> None

  let equal_pair eq (a, b) (a', b') = eq a a' && eq b b'

  let equal_value : type a. a sort -> a -> a -> bool =
   fun s v v' ->
    match s with
    | Number -> Float.equal v v'
    | Choice -> String.equal v v'
    | Interval -> Option.equal (equal_pair Float.equal) v v'
    | Zoom Scale.Quantitative -> Option.equal (equal_pair Float.equal) v v'
    | Zoom Scale.Temporal -> Option.equal (equal_pair Time.equal) v v'
    | Zoom Scale.Categorical -> Option.equal (equal_pair String.equal) v v'

  let pp_value : type a. a sort -> Format.formatter -> a -> unit =
   fun s ppf v ->
    let pp_pair pp ppf (a, b) = Format.fprintf ppf "(%a, %a)" pp a pp b in
    let pp_opt pp ppf = function
      | None -> Format.pp_print_string ppf "none"
      | Some v -> pp ppf v
    in
    let pp_float ppf x = Format.fprintf ppf "%g" x in
    match s with
    | Number -> pp_float ppf v
    | Choice -> Format.fprintf ppf "%S" v
    | Interval -> pp_opt (pp_pair pp_float) ppf v
    | Zoom Scale.Quantitative -> pp_opt (pp_pair pp_float) ppf v
    | Zoom Scale.Temporal -> pp_opt (pp_pair Time.pp) ppf v
    | Zoom Scale.Categorical -> pp_opt (pp_pair Format.pp_print_string) ppf v

  let equal_key : type a b. a key -> b key -> (a, b) Type.eq option =
   fun k k' ->
    if not (equal_ident k.ident k'.ident) then None
    else
      match equal_sort k.sort k'.sort with
      | Some Type.Equal as eq ->
          if equal_value k.sort k.init k'.init then eq else None
      | None -> None

  let check_interval fn = function
    | None -> ()
    | Some (lo, hi) ->
        if not (Float.is_finite lo && Float.is_finite hi && lo <= hi) then
          err fn "the interval (%g, %g) is not finite or not ordered" lo hi

  let number name ~init = { ident = User name; sort = Number; init }
  let choice name ~init = { ident = User name; sort = Choice; init }

  let interval name ~init =
    check_interval "View.interval" init;
    { ident = User name; sort = Interval; init }

  let zoom : type d. ?at:id -> d Scale.t -> (d * d) option key =
   fun ?(at = Nx.Ptree.Path.root) s ->
    let kind = Scale.kind s in
    (match kind with
    | Scale.Categorical -> err "View.zoom" "the scale is categorical"
    | Scale.Quantitative | Scale.Temporal -> ());
    match Scale.name s with
    | None -> err "View.zoom" "the scale is unnamed"
    | Some scale ->
        { ident = Zoom_of { scale; at }; sort = Zoom kind; init = None }

  type value = V : 'a sort * 'a -> value

  (* Bindings in increasing order of their idents. *)
  type t = (ident * value) list

  let empty = []

  let set : type a. a key -> a -> t -> t =
   fun k v view ->
    (match k.sort with Interval -> check_interval "View.set" v | _ -> ());
    let rec add = function
      | [] -> [ (k.ident, V (k.sort, v)) ]
      | ((i, _) as b) :: rest ->
          let c = compare_ident k.ident i in
          if c < 0 then (k.ident, V (k.sort, v)) :: b :: rest
          else if c = 0 then (k.ident, V (k.sort, v)) :: rest
          else b :: add rest
    in
    add view

  let get : type a. a key -> t -> a =
   fun k view ->
    match List.find_opt (fun (i, _) -> equal_ident k.ident i) view with
    | None -> k.init
    | Some (_, V (s, v)) -> (
        match equal_sort k.sort s with Some Type.Equal -> v | None -> k.init)

  let equal (view : t) (view' : t) =
    List.equal
      (fun (i, V (s, v)) (i', V (s', v')) ->
        equal_ident i i'
        &&
        match equal_sort s s' with
        | Some Type.Equal -> equal_value s v v'
        | None -> false)
      view view'

  let pp ppf (view : t) =
    let pp_binding ppf (i, V (s, v)) =
      Format.fprintf ppf "@[%a = %a@]" pp_ident i (pp_value s) v
    in
    Format.fprintf ppf "@[<v>%a@]" (Format.pp_print_list pp_binding) view
end

(* Sizes and themes *)

module Size = struct
  type t = Figure of float * float | Panels of float * float

  let check fn w h =
    if not (is_pos w && is_pos h) then
      err fn "%g × %g is not finite and positive" w h

  let figure w h =
    check "Size.figure" w h;
    Figure (w, h)

  let panels w h =
    check "Size.panels" w h;
    Panels (w, h)

  let mm l = l *. 72. /. 25.4
  let dpi d = d /. 72.

  let equal s s' =
    match (s, s') with
    | Figure (w, h), Figure (w', h') | Panels (w, h), Panels (w', h') ->
        Float.equal w w' && Float.equal h h'
    | Figure _, Panels _ | Panels _, Figure _ -> false

  let pp ppf = function
    | Figure (w, h) -> Format.fprintf ppf "figure %g × %g pt" w h
    | Panels (w, h) -> Format.fprintf ppf "panels %g × %g pt" w h
end

module Theme = struct
  type t = {
    ink : Color.t;
    paper : Color.t;
    accent : Color.t;
    size : float;
    fonts : Font.t list;
    palette : Scheme.t;
    scheme : Scheme.t;
    locale : Locale.t;
  }

  let v ?(ink = Color.gray 0.1) ?(paper = Color.white) ?accent ?(size = 10.)
      ?(fonts = [ Font.regular; Font.bold ]) ?(palette = Scheme.tableau10)
      ?(scheme = Scheme.viridis) ?(locale = Locale.default) () =
    if not (is_pos size) then
      err "Theme.v" "size %g is not finite and positive" size;
    (match fonts with [] -> err "Theme.v" "fonts is empty" | _ :: _ -> ());
    let accent =
      match accent with Some c -> c | None -> (Scheme.colors 1 palette).(0)
    in
    { ink; paper; accent; size; fonts; palette; scheme; locale }

  let default = v ()
  let ink th = th.ink
  let paper th = th.paper
  let accent th = th.accent
  let size th = th.size
  let fonts th = th.fonts
  let palette th = th.palette
  let scheme th = th.scheme
  let locale th = th.locale

  let equal th th' =
    Color.equal th.ink th'.ink
    && Color.equal th.paper th'.paper
    && Color.equal th.accent th'.accent
    && Float.equal th.size th'.size
    && List.equal Font.equal th.fonts th'.fonts
    && Scheme.equal th.palette th'.palette
    && Scheme.equal th.scheme th'.scheme
    && Locale.equal th.locale th'.locale

  let pp ppf th =
    Format.fprintf ppf
      "@[<v 1>(theme@ ink %a@ paper %a@ accent %a@ size %g@ fonts %a@ palette \
       %a@ scheme %a@ locale %a)@]"
      Color.pp th.ink Color.pp th.paper Color.pp th.accent th.size
      (Format.pp_print_list ~pp_sep:Format.pp_print_space Font.pp)
      th.fonts Scheme.pp th.palette Scheme.pp th.scheme Locale.pp th.locale
end

(* Roles *)

(* The ranges roles map into. [Curves] and [Pixels] are the ranges of roles the
   built-in marks keep their parameters in. *)
type _ range =
  | Floats : float range
  | Colors : Color.t range
  | Symbols : Symbol.t range
  | Texts : Text.t range
  | Panels : string range
  | Curves : Curve.t range
  | Pixels : Nx.packed range

let equal_range : type r s. r range -> s range -> (r, s) Type.eq option =
 fun r s ->
  match (r, s) with
  | Floats, Floats -> Some Type.Equal
  | Colors, Colors -> Some Type.Equal
  | Symbols, Symbols -> Some Type.Equal
  | Texts, Texts -> Some Type.Equal
  | Panels, Panels -> Some Type.Equal
  | Curves, Curves -> Some Type.Equal
  | Pixels, Pixels -> Some Type.Equal
  | _ -> None

let equal_in : type r. r range -> r -> r -> bool =
 fun r v v' ->
  match r with
  | Floats -> Float.equal v v'
  | Colors -> Color.equal v v'
  | Symbols -> Symbol.equal v v'
  | Texts -> Text.equal v v'
  | Panels -> String.equal v v'
  | Curves -> Curve.equal v v'
  | Pixels ->
      let (Nx.P px) = v in
      let (Nx.P px') = v' in
      equal_tensor px px'

module Role = struct
  (* [scale] is the name of the scale the role reads by default, [None] for a
     role that reads none. *)
  type ('d, 'r) t = { name : string; range : 'r range; scale : string option }

  let x = { name = "x"; range = Floats; scale = Some "x" }
  let x2 = { name = "x2"; range = Floats; scale = Some "x" }
  let y = { name = "y"; range = Floats; scale = Some "y" }
  let y2 = { name = "y2"; range = Floats; scale = Some "y" }
  let fill = { name = "fill"; range = Colors; scale = Some "color" }
  let stroke = { name = "stroke"; range = Colors; scale = Some "color" }
  let opacity = { name = "opacity"; range = Floats; scale = Some "opacity" }
  let size = { name = "size"; range = Floats; scale = Some "size" }
  let width = { name = "width"; range = Floats; scale = Some "width" }
  let symbol = { name = "symbol"; range = Symbols; scale = Some "symbol" }
  let text = { name = "text"; range = Texts; scale = None }
  let fx = { name = "fx"; range = Panels; scale = Some "fx" }
  let fy = { name = "fy"; range = Panels; scale = Some "fy" }

  let names =
    [
      "x";
      "x2";
      "y";
      "y2";
      "fill";
      "stroke";
      "opacity";
      "size";
      "width";
      "symbol";
      "text";
      "fx";
      "fy";
    ]

  let value ~name =
    if name = "" then err "Role.value" "the name is empty";
    if List.mem name names then err "Role.value" "%S names a built-in role" name;
    { name; range = Floats; scale = None }

  (* The parameters of built-in marks. *)
  let curve = { name = "curve"; range = Curves; scale = None }
  let dx = { name = "dx"; range = Floats; scale = None }
  let dy = { name = "dy"; range = Floats; scale = None }
  let pixels = { name = "pixels"; range = Pixels; scale = None }
end

(* Channels *)

(* [Scalar v] is the quantity [v] for every row: it contributes [v] to its
   scale's domain without taking part in broadcasting. *)
type _ lift =
  | Num : { x : ('a, 'b) Nx.t; valid : Nx.bool_t option } -> float lift
  | Index : int -> float lift
  | Scalar : float -> float lift
  | Cat : {
      codes : ('a, 'b) Nx.t;
      valid : Nx.bool_t option;
      labels : string array option;
    }
      -> string lift
  | Strings : string array -> string lift
  | Dim : {
      axis : int;
      valid : Nx.bool_t option;
      labels : string array option;
    }
      -> string lift

type ('d, 'r) channel =
  | Const : 'r -> ('d, 'r) channel
  | Data : {
      lift : 'd lift;
      scale : 'd Scale.t option;
      title : Text.t option;
    }
      -> ('d, 'r) channel
  | Map : ('r -> 'r) * ('d, 'r) channel -> ('d, 'r) channel

let lift_kind : type d. d lift -> d Scale.kind = function
  | Num _ -> Scale.Quantitative
  | Index _ -> Scale.Quantitative
  | Scalar _ -> Scale.Quantitative
  | Cat _ -> Scale.Categorical
  | Strings _ -> Scale.Categorical
  | Dim _ -> Scale.Categorical

let equal_lift : type d e. d lift -> e lift -> (d, e) Type.eq option =
 fun l l' ->
  let ok b = if b then Some Type.Equal else None in
  match (l, l') with
  | Num a, Num b ->
      ok (equal_tensor a.x b.x && Option.equal ( == ) a.valid b.valid)
  | Index k, Index k' -> ok (Int.equal k k')
  | Scalar v, Scalar v' -> ok (Float.equal v v')
  | Cat a, Cat b ->
      ok
        (equal_tensor a.codes b.codes
        && Option.equal ( == ) a.valid b.valid
        && Option.equal equal_strings a.labels b.labels)
  | Strings a, Strings b -> ok (equal_strings a b)
  | Dim a, Dim b ->
      ok
        (Int.equal a.axis b.axis
        && Option.equal ( == ) a.valid b.valid
        && Option.equal equal_strings a.labels b.labels)
  | _ -> None

let rec equal_channel : type d e r.
    r range -> (d, r) channel -> (e, r) channel -> bool =
 fun r c c' ->
  match (c, c') with
  | Const v, Const v' -> equal_in r v v'
  | Data a, Data b -> (
      match equal_lift a.lift b.lift with
      | Some Type.Equal ->
          Option.equal Scale.equal a.scale b.scale
          && Option.equal Text.equal a.title b.title
      | None -> false)
  | Map (f, c), Map (f', c') -> f == f' && equal_channel r c c'
  | _ -> false

(* [data c] is the lift, specification and title of [c], if it holds data. *)
type 'd data = {
  lift : 'd lift;
  spec : 'd Scale.t option;
  title : Text.t option;
}

let rec data : type d r. (d, r) channel -> d data option = function
  | Const _ -> None
  | Data { lift; scale; title } -> Some { lift; spec = scale; title }
  | Map (_, c) -> data c

(* Marks and figures *)

type rows = |
type reducer = M4 | Cells | Raster

type binding =
  | B : {
      role : ('d, 'r) Role.t;
      ch : ('d, 'r) channel;
      imply : float Scale.t option;
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
  | Axis of { side : side option; grid : bool; show : bool; scale : string }
  | Legend of { side : side option; show : bool; scale : string }

let equal_binding (B b) (B b') =
  String.equal b.role.name b'.role.name
  &&
  match equal_range b.role.range b'.role.range with
  | None -> false
  | Some Type.Equal ->
      equal_channel b.role.range b.ch b'.ch
      && Option.equal Scale.equal b.imply b'.imply
      && Option.equal Bool.equal b.guide b'.guide

let equal_mark m m' =
  String.equal m.kind m'.kind
  && Option.equal ( = ) m.reduce m'.reduce
  && Option.equal Coord.equal m.coord m'.coord
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
  | Axis a, Axis b ->
      Option.equal equal_side a.side b.side
      && Bool.equal a.grid b.grid && Bool.equal a.show b.show
      && String.equal a.scale b.scale
  | Legend a, Legend b ->
      Option.equal equal_side a.side b.side
      && Bool.equal a.show b.show
      && String.equal a.scale b.scale
  | ( ( Mark _ | Layer _ | Grid _ | Span _ | Share _ | Title _ | Coord_sys _
      | Name _ | Bind _ | Axis _ | Legend _ ),
      _ ) ->
      false

(* Lifts *)

let is_real : type a b. (a, b) Nx.dtype -> bool = function
  | Nx.Complex64 | Nx.Complex128 | Nx.Bool -> false
  | _ -> true

let is_integer : type a b. (a, b) Nx.dtype -> bool = function
  | Nx.Int4 | Nx.UInt4 | Nx.Int8 | Nx.UInt8 | Nx.Int16 | Nx.UInt16 | Nx.Int32
  | Nx.UInt32 | Nx.Int64 | Nx.UInt64 ->
      true
  | _ -> false

(* [fits s s'] is [true] iff [s] broadcasts to [s'] without growing it. *)
let fits s s' =
  let n = Array.length s and n' = Array.length s' in
  n <= n'
  &&
  let ok = ref true in
  for i = 0 to n - 1 do
    let d = s.(i) in
    if d <> 1 && d <> s'.(n' - n + i) then ok := false
  done;
  !ok

let check_valid fn shape = function
  | None -> ()
  | Some v ->
      if not (fits (Nx.shape v) shape) then
        err fn "valid, of shape %a, does not broadcast to the shape %a" pp_shape
          (Nx.shape v) pp_shape shape

let check_distinct fn labels =
  let seen = Hashtbl.create (Array.length labels) in
  Array.iter
    (fun l ->
      if Hashtbl.mem seen l then err fn "the label %S is repeated" l;
      Hashtbl.add seen l ())
    labels

let num ?scale ?valid ?title x =
  if not (is_real (Nx.dtype x)) then
    err "num" "the dtype %s is not real" (Nx_dtype.to_string (Nx.dtype x));
  check_valid "num" (Nx.shape x) valid;
  Data { lift = Num { x; valid }; scale; title }

let cat ?scale ?valid ?title ?labels codes =
  if not (is_integer (Nx.dtype codes)) then
    err "cat" "the dtype %s is not an integer dtype"
      (Nx_dtype.to_string (Nx.dtype codes));
  check_valid "cat" (Nx.shape codes) valid;
  Option.iter (check_distinct "cat") labels;
  let labels = Option.map Array.copy labels in
  Data { lift = Cat { codes; valid; labels }; scale; title }

let strings ?scale ?title a =
  Data { lift = Strings (Array.copy a); scale; title }

let dim ?scale ?valid ?title ?labels axis =
  let labels = Option.map Array.copy labels in
  Data { lift = Dim { axis; valid; labels }; scale; title }

let index ?scale ?title axis = Data { lift = Index axis; scale; title }
let const v = Const v
let map_range f c = Map (f, c)

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

(* [axis_of shape k] is the axis [k] of [shape], counting from the last for a
   negative [k]. *)
let axis_of shape k =
  let n = Array.length shape in
  let a = if k < 0 then n + k else k in
  if a < 0 || a >= n then None else Some a

(* [lift_shape l] is the shape [l] takes part in broadcasting with, if any. *)
let lift_shape : type d. d lift -> int array option = function
  | Num { x; _ } -> Some (Nx.shape x)
  | Cat { codes; _ } -> Some (Nx.shape codes)
  | Strings a -> Some [| Array.length a |]
  | Dim { valid = Some v; _ } -> Some (Nx.shape v)
  | Dim { valid = None; _ } | Index _ | Scalar _ -> None

let role_name (B b) = b.role.name
let binds name bindings = List.exists (fun b -> role_name b = name) bindings

let find_binding name bindings =
  List.find_opt (fun b -> String.equal (role_name b) name) bindings

let binding_kind (B b) : [ `Quantities | `Categories ] option =
  match data b.ch with
  | None -> None
  | Some d -> (
      match lift_kind d.lift with
      | Scale.Quantitative | Scale.Temporal -> Some `Quantities
      | Scale.Categorical -> Some `Categories)

let check_ends fn bindings a b =
  match (find_binding a bindings, find_binding b bindings) with
  | None, Some _ -> err fn "%s is bound without %s" b a
  | Some ba, Some bb -> (
      match (binding_kind ba, binding_kind bb) with
      | Some k, Some k' when k <> k' ->
          err fn
            "%s and %s hold one channel of quantities and one of categories" a b
      | _ -> ())
  | _, None -> ()

let check_binding fn shape (B b) =
  match data b.ch with
  | None -> ()
  | Some d -> (
      (match (b.role.scale, d.spec, d.title) with
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
      | Num _ | Scalar _ | Cat _ | Strings _ -> ())

let make_mark fn ~name ?reduce ?coord ?swatch ?(base = [||]) bindings draw =
  let rec distinct = function
    | [] -> ()
    | b :: rest ->
        if binds (role_name b) rest then
          err fn "the role %s is bound twice" (role_name b);
        distinct rest
  in
  distinct bindings;
  check_ends fn bindings "x" "x2";
  check_ends fn bindings "y" "y2";
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
      base bindings
  in
  List.iter (check_binding fn shape) bindings;
  { kind = name; reduce; coord; swatch; bindings; draw; shape }

(* Built-in marks *)

let on ?imply ?guide role ch = B { role; ch; imply; guide }
let opt role = Option.map (on role)

(* Rows are inhabited once figures are drawn: until then no draw function is
   called. *)
let not_drawn (r : rows) : Picture.t = match r with _ -> .

(* [length ch] implies [zero] on the scale of [ch] when it holds quantities: a
   position without its other end is a length. *)
let length : type d r. (d, r) Role.t -> (d, r) channel -> binding =
 fun role ch ->
  match data ch with
  | Some { lift; _ } -> (
      match lift_kind lift with
      | Scale.Quantitative -> on ~imply:(Scale.linear ~zero:true ()) role ch
      | Scale.Temporal | Scale.Categorical -> on role ch)
  | None -> on role ch

let position role ~alone = function
  | None -> None
  | Some ch -> Some (if alone then length role ch else on role ch)

let facets fx fy = [ opt Role.fx fx; opt Role.fy fy ]

let make fn ?reduce ?coord ?base l =
  make_mark fn ~name:fn ?reduce ?coord ?base (List.filter_map Fun.id l)
    not_drawn

let dot ?fill ?stroke ?opacity ?size ?symbol ?fx ?fy ~x ~y () =
  Mark
    (make "dot" ~reduce:Raster
       ([
          Some (on Role.x x);
          Some (on Role.y y);
          opt Role.fill fill;
          opt Role.stroke stroke;
          opt Role.opacity opacity;
          opt Role.size size;
          opt Role.symbol symbol;
        ]
       @ facets fx fy))

let line ?x ?stroke ?fill ?width ?opacity ?(curve = Curve.linear) ?fx ?fy ~y ()
    =
  let x =
    match x with Some x -> on Role.x x | None -> on Role.x (index (-1))
  in
  Mark
    (make "line" ~reduce:M4
       ([
          Some x;
          Some (on Role.y y);
          opt Role.stroke stroke;
          opt Role.fill fill;
          opt Role.width width;
          opt Role.opacity opacity;
          Some (on Role.curve (const curve));
        ]
       @ facets fx fy))

let rect ?x ?x2 ?y ?y2 ?fill ?stroke ?opacity ?fx ?fy () =
  Mark
    (make "rect" ~reduce:Cells
       ([
          position Role.x ~alone:(Option.is_none x2) x;
          opt Role.x2 x2;
          position Role.y ~alone:(Option.is_none y2) y;
          opt Role.y2 y2;
          opt Role.fill fill;
          opt Role.stroke stroke;
          opt Role.opacity opacity;
        ]
       @ facets fx fy))

let rule ?x ?x2 ?y ?y2 ?stroke ?width ?opacity ?fx ?fy () =
  let has = Option.is_some in
  let positions =
    if has x && not (has x2) then
      [
        position Role.x ~alone:false x;
        position Role.y ~alone:(not (has y2)) y;
        opt Role.y2 y2;
      ]
    else if has y && not (has y2) then
      [
        position Role.y ~alone:false y;
        position Role.x ~alone:(not (has x2)) x;
        opt Role.x2 x2;
      ]
    else if has x && has x2 && has y && has y2 then
      [ opt Role.x x; opt Role.x2 x2; opt Role.y y; opt Role.y2 y2 ]
    else
      err "rule"
        "the channels match no case: give x without x2, y without y2, or x, \
         x2, y and y2"
  in
  Mark
    (make "rule"
       (positions
       @ [
           opt Role.stroke stroke;
           opt Role.width width;
           opt Role.opacity opacity;
         ]
       @ facets fx fy))

let text ?fill ?opacity ?(dx = 0.) ?(dy = 0.) ?fx ?fy ~x ~y ~text () =
  Mark
    (make "text"
       ([
          Some (on Role.x x);
          Some (on Role.y y);
          Some (on Role.text text);
          opt Role.fill fill;
          opt Role.opacity opacity;
          Some (on Role.dx (const dx));
          Some (on Role.dy (const dy));
        ]
       @ facets fx fy))

let is_pixel : type a b. (a, b) Nx.dtype -> bool = function
  | Nx.UInt8 | Nx.Float16 | Nx.Float32 | Nx.Float64 | Nx.BFloat16
  | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      true
  | _ -> false

let image ?fx ?fy px =
  let shape = Nx.shape px in
  let rank = Array.length shape in
  let dtype = Nx.dtype px in
  if not (is_pixel dtype) then
    err "image" "the dtype %s is neither uint8 nor floating point"
      (Nx_dtype.to_string dtype);
  if rank < 2 then
    err "image" "the shape %a has fewer than two axes" pp_shape shape;
  let lead, h, w =
    if rank = 2 then ([||], shape.(0), shape.(1))
    else
      match shape.(rank - 1) with
      | 1 | 3 | 4 ->
          (Array.sub shape 0 (rank - 3), shape.(rank - 3), shape.(rank - 2))
      | c -> err "image" "the last axis has %d channels, not 1, 3 or 4" c
  in
  let fixed = Scale.linear ~nice:false () in
  let x = Data { lift = Scalar 0.; scale = None; title = None } in
  let x2 = Data { lift = Scalar (float w); scale = None; title = None } in
  let y = Data { lift = Scalar 0.; scale = None; title = None } in
  let y2 = Data { lift = Scalar (float h); scale = None; title = None } in
  Mark
    (make "image"
       ~coord:(Coord.cartesian ~aspect:1. ())
       ~base:lead
       ([
          Some (on ~imply:fixed ~guide:false Role.x x);
          Some (on Role.x2 x2);
          Some
            (on
               ~imply:(Scale.linear ~nice:false ~reverse:true ())
               ~guide:false Role.y y);
          Some (on Role.y2 y2);
          Some (on Role.pixels (const (Nx.P px)));
        ]
       @ facets fx fy))

(* [varies shape b a] is [true] iff the channel of [b] can vary along axis [a]
   of [shape]. *)
let varies shape (B b) a =
  let rank = Array.length shape in
  let along s =
    let off = rank - Array.length s in
    a >= off && s.(a - off) > 1
  in
  match data b.ch with
  | None -> false
  | Some d -> (
      match d.lift with
      | Num { x; _ } -> along (Nx.shape x)
      | Cat { codes; _ } -> along (Nx.shape codes)
      | Strings s -> along [| Array.length s |]
      | Index k | Dim { axis = k; _ } -> axis_of shape k = Some a
      | Scalar _ -> false)

let contour ?x ?y ?opacity ?fx ?fy ~fill () =
  let x =
    match x with Some x -> on Role.x x | None -> on Role.x (index (-1))
  in
  let y =
    match y with Some y -> on Role.y y | None -> on Role.y (index (-2))
  in
  if Option.is_none (data fill) then err "contour" "fill is a constant";
  let m =
    make "contour"
      ([ Some x; Some y; Some (on Role.fill fill); opt Role.opacity opacity ]
      @ facets fx fy)
  in
  let shape = m.shape in
  let rank = Array.length shape in
  if rank < 2 then
    err "contour" "the shape %a has fewer than two axes" pp_shape shape;
  if varies shape x (rank - 2) then
    err "contour" "x can vary along the rows of the grid";
  if varies shape y (rank - 1) then
    err "contour" "y can vary along the columns of the grid";
  Mark m

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

let title ?(align = `Center) text f = Title { align; text; f }
let coord c f = Coord_sys (c, f)

let name s f =
  match s with
  | "axis" | "legend" | "panel" | "cell" ->
      err "name" "%S is the segment of generated nodes" s
  | _ -> Name (s, f)

let bind k fn = Bind (k, fn)

let axis ?side ?(grid = false) ?(show = true) scale =
  Axis { side; grid; show; scale }

let legend ?side ?(show = true) scale = Legend { side; show; scale }

(* Extending *)

module Mark = struct
  type nonrec binding = binding

  let bind = on

  type nonrec rows = rows

  let id (r : rows) = match r with _ -> .
  let length (r : rows) = match r with _ -> .
  let shape (r : rows) = match r with _ -> .
  let index (r : rows) = match r with _ -> .
  let get (r : rows) _ = match r with _ -> .
  let normalized (r : rows) _ = match r with _ -> .
  let range (r : rows) _ = match r with _ -> .
  let ticks (r : rows) _ = match r with _ -> .
  let points (r : rows) = match r with _ -> .
  let extent (r : rows) _ = match r with _ -> .
  let projection (r : rows) = match r with _ -> .
  let project (r : rows) _ = match r with _ -> .
  let series (r : rows) = match r with _ -> .
  let theme (r : rows) = match r with _ -> .
  let text ?halign:_ ?valign:_ (r : rows) _ _ _ = match r with _ -> .
  let warn (r : rows) _ = match r with _ -> .

  type nonrec reducer = reducer

  let m4 = M4
  let cells = Cells
  let raster = Raster

  let v ~name ?reduce ?coord ?swatch bindings draw =
    Mark (make_mark "Mark.v" ~name ?reduce ?coord ?swatch bindings draw)
end

(* Resolving

   [resolve] expands the figure, evaluating binds and assigning ids; arranges
   it, forming scopes and broadcasting layers over grids; reads the channels of
   each occurrence and merges the specifications of each scale; summarises each
   occurrence's data where it lives; and fits the scales, categorical ones first
   since facets make panels. An occurrence is a mark in one cell: a mark that a
   layer broadcasts over a grid occurs once per cell, and reads the scales of
   each. *)

let positional = function "x" | "y" | "fx" | "fy" -> true | _ -> false

(* The default scales of roles other than positions and facets are told apart by
   their kinds too. *)
let kind_scoped = function
  | "color" | "opacity" | "size" | "width" | "symbol" -> true
  | _ -> false

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

type kind_tag = Q | T | C

let tag : type d. d Scale.kind -> kind_tag = function
  | Scale.Quantitative -> Q
  | Scale.Temporal -> T
  | Scale.Categorical -> C

let pp_tag ppf t =
  Format.pp_print_string ppf
    (match t with Q -> "quantitative" | T -> "temporal" | C -> "categorical")

(* A scale's identity within a scope: its name, and its kind for the default
   scales [kind_scoped] names. *)
type sid = { sname : string; skind : kind_tag option }

let sid name t =
  { sname = name; skind = (if kind_scoped name then Some t else None) }

let equal_sid s s' =
  String.equal s.sname s'.sname && Option.equal ( = ) s.skind s'.skind

(* Expanding: binds evaluated, ids assigned. Wrappers have their child's id. *)

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

(* [force st f] evaluates the binds at the head of [f], through wrappers. *)
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

(* Arranging: shares give scopes, layers broadcast over grids, and every panel's
   content becomes a list of occurrences. *)

type env = {
  shares : (string * key) list; (* Innermost first. *)
  pending : string list; (* Independent names awaiting the node's children. *)
  cell : id; (* The innermost grid cell, or the root. *)
}

type shares = (string * key) list

let key_of (env : env) name =
  match List.assoc_opt name env.shares with
  | Some k -> k
  | None -> if positional name then Cell env.cell else Figure

type occ = {
  mid : id;
  mark : mark;
  order : int;
  shares : (string * key) list;
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
  match (data b.ch, b.role.scale) with
  | Some d, Some default ->
      Some (Option.value ~default (Option.bind d.spec Scale.name))
  | _ -> None

(* [reads n] is the scales the marks under [n] read, each with whether a
   position or facet role reads it. *)
let rec reads n =
  match n.n with
  | E_mark { mark; _ } ->
      List.filter_map
        (fun (B b as bd) ->
          Option.map
            (fun s -> (s, positional (Option.get b.role.scale)))
            (scale_name bd))
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
  let key name = if cell && positional name then Cell id else Node id in
  let shares = List.map (fun n -> (n, key n)) env.pending @ env.shares in
  { shares; pending = []; cell = (if cell then id else env.cell) }

let equal_titles =
  List.equal (fun (a, t) (a', t') -> equal_halign a a' && Text.equal t t')

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
  let titles =
    List.fold_left
      (fun acc s ->
        match (s.titles, acc) with
        | [], _ -> acc
        | t, None -> Some t
        | t, Some t' ->
            if equal_titles t t' then acc
            else
              err "resolve" "the children of %a have different titles" pp_id lid)
      None children
  in
  let titles = Option.value ~default:[] titles in
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

(* [arrange nodes env ~in_cell n] is [n] arranged, with the id of each core node
   added to [nodes]. *)
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
  | E_layer cs ->
      record ();
      layer_shaped n.id env.shares
        (List.map
           (fun c ->
             arrange nodes (child_env env ~cell:false c.id) ~in_cell:false c)
           cs)
      |> hold (n.id, env.shares)
  | E_grid g ->
      record ();
      arrange_grid nodes env n g.rows g.widths g.heights
      |> hold (n.id, env.shares)

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
          let s =
            arrange nodes (child_env env ~cell:true c.id) ~in_cell:true c
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

(* [panels s] is the cells of [s] that hold content, with their ids, in reading
   order. *)
let panels root s =
  let rec go acc pid s =
    match s.body with
    | Single c -> (pid, c) :: acc
    | Arr a ->
        List.fold_left (fun acc cell -> go acc cell.cid cell.s) acc a.cells
  in
  List.rev (go [] root s)

(* Readings: the channels that read scales, each with its scale's identity and
   scope. *)

type reading =
  | R : {
      occ : occ;
      pid : id;
      index : int;
      role : string;
      d : 'd data;
      kind : 'd Scale.kind;
      imply : float Scale.t option;
      guide : bool option;
      mapped : bool;
      sid : sid;
      key : key;
    }
      -> reading

let is_map : type d r. (d, r) channel -> bool = function
  | Map _ -> true
  | Const _ | Data _ -> false

let readings_of pid occ =
  let env = { shares = occ.shares; pending = []; cell = pid } in
  List.concat
    (List.mapi
       (fun index (B b) ->
         match (data b.ch, b.role.scale) with
         | Some d, Some default ->
             let name = Option.value ~default (Option.bind d.spec Scale.name) in
             let kind = lift_kind d.lift in
             let key =
               if not (List.mem name occ.per_panel) then key_of env name
               else if b.role.name = "fx" || b.role.name = "fy" then
                 err "resolve"
                   "%a makes its facet scale %S independent per panel" pp_id
                   occ.mid name
               else Panels_of (occ.mid, pid)
             in
             [
               R
                 {
                   occ;
                   pid;
                   index;
                   role = b.role.name;
                   d;
                   kind;
                   imply = b.imply;
                   guide = b.guide;
                   mapped = is_map b.ch;
                   sid = sid name (tag kind);
                   key;
                 };
             ]
         | _ -> [])
       occ.mark.bindings)

type 'd member = {
  m_occ : occ;
  m_pid : id;
  m_index : int;
  m_role : string;
  m_d : 'd data;
  m_imply : float Scale.t option;
  m_guide : bool option;
}

type group =
  | G : {
      sid : sid;
      key : key;
      kind : 'd Scale.kind;
      members : 'd member list; (* In figure order. *)
      legend : bool;
          (* A role other than a position or facet reads it without
             map_range. *)
    }
      -> group

let axis_role = function "x" | "x2" -> "x" | "y" | "y2" -> "y" | r -> r

let group readings =
  let add groups (R r) =
    let member =
      {
        m_occ = r.occ;
        m_pid = r.pid;
        m_index = r.index;
        m_role = r.role;
        m_d = r.d;
        m_imply = r.imply;
        m_guide = r.guide;
      }
    in
    let legend = (not (positional (axis_role r.role))) && not r.mapped in
    let rec go = function
      | [] ->
          [
            G
              {
                sid = r.sid;
                key = r.key;
                kind = r.kind;
                members = [ member ];
                legend;
              };
          ]
      | (G g as gr) :: rest -> (
          if not (equal_sid g.sid r.sid && equal_key g.key r.key) then
            gr :: go rest
          else
            match Scale.equal_kind g.kind r.kind with
            | Some Type.Equal ->
                G
                  {
                    g with
                    members = member :: g.members;
                    legend = g.legend || legend;
                  }
                :: rest
            | None ->
                let m = List.hd g.members in
                err "resolve" "the scale %S is read as %a by %a and as %a by %a"
                  r.sid.sname pp_tag (tag g.kind) pp_id m.m_occ.mid pp_tag
                  (tag r.kind) pp_id r.occ.mid)
    in
    go groups
  in
  List.fold_left add [] readings
  |> List.rev_map (fun (G g) -> G { g with members = List.rev g.members })
  |> List.rev

(* In a panel, the channels on x read one scale, and likewise y, fx and fy. *)
let check_panel_scales readings =
  let rec go seen = function
    | [] -> ()
    | (R r as rd) :: rest ->
        let axis = axis_role r.role in
        (if positional axis then
           match
             List.find_opt
               (fun (R r') ->
                 Nx.Ptree.Path.equal r'.pid r.pid
                 && String.equal (axis_role r'.role) axis)
               seen
           with
           | Some (R r')
             when not (equal_sid r'.sid r.sid && equal_key r'.key r.key) ->
               err "resolve" "%a and %a read two %s scales in the panel %a"
                 pp_id r'.occ.mid pp_id r.occ.mid axis pp_id r.pid
           | _ -> ());
        go (rd :: seen) rest
  in
  go [] readings

let check_coords cells =
  List.iter
    (fun (pid, c) ->
      match c.coords with
      | (id, k) :: rest -> (
          match List.find_opt (fun (_, k') -> not (Coord.equal k k')) rest with
          | Some (id', _) ->
              err "resolve"
                "the panel %a lies under two coordinate systems, at %a and %a"
                pp_id pid pp_id id pp_id id'
          | None -> ())
      | [] -> (
          let implied =
            List.filter_map
              (fun o -> Option.map (fun k -> (o.mid, k)) o.mark.coord)
              c.occs
          in
          match implied with
          | (id, k) :: rest -> (
              match
                List.find_opt (fun (_, k') -> not (Coord.equal k k')) rest
              with
              | Some (id', _) ->
                  err "resolve"
                    "%a and %a imply two coordinate systems in the panel %a"
                    pp_id id pp_id id' pp_id pid
              | None -> ())
          | [] -> ()))
    cells

let check_axes cells readings =
  List.iter
    (fun (pid, c) ->
      let axes =
        List.filter_map
          (function G_axis a -> Some a | G_legend _ -> None)
          c.guides
      in
      List.iter
        (fun a ->
          List.iter
            (fun a' ->
              if String.equal a.scale a'.scale && not (equal_axis a a') then
                err "resolve" "the panel %a holds two different axes for %S"
                  pp_id pid a.scale)
            axes;
          let ok =
            List.exists
              (fun (R r) ->
                Nx.Ptree.Path.equal r.pid pid
                && positional (axis_role r.role)
                && String.equal r.sid.sname a.scale)
              readings
          in
          if not ok then
            err "resolve"
              "the axis %a names %S, no position or facet scale of its panel"
              pp_id a.gid a.scale)
        axes)
    cells

(* Merging specifications *)

let explicit_domain : type d. d Scale.t -> d Scale.domain option =
 fun s ->
  (* A set property differs from the same property unset, so [s] sets its domain
     iff setting it again changes nothing. *)
  let d = Scale.domain s in
  match Scale.with_domain d s with
  | s' -> if Scale.equal s s' then Some d else None
  | exception Invalid_argument _ -> None

let labelled : type d. d lift -> bool = function
  | Cat { labels = Some _; _ } | Strings _ -> true
  | Cat { labels = None; _ } | Dim _ -> false
  | Num _ | Index _ | Scalar _ -> false

let merge_level sid level specs =
  let rec go acc prior = function
    | [] -> acc
    | (m, s) :: rest ->
        let acc =
          match acc with
          | None -> Some s
          | Some a -> (
              match Scale.merge a s with
              | Ok a -> Some a
              | Error p ->
                  let m0 =
                    match
                      List.find_opt
                        (fun (_, s0) -> Result.is_error (Scale.merge s0 s))
                        prior
                    with
                    | Some (m0, _) -> m0
                    | None -> m
                  in
                  err "resolve"
                    "%a and %a give the scale %S two %s values of %a" pp_id
                    m0.m_occ.mid pp_id m.m_occ.mid sid.sname level
                    Scale.pp_property p)
        in
        go acc ((m, s) :: prior) rest
  in
  go None [] specs

(* [merged kind sid ms] is the specification of the scale [ms] read: their
   explicit specifications merged, then implied ones under them. *)
let merged : type d. d Scale.kind -> sid -> d member list -> d Scale.t =
 fun kind sid ms ->
  let base : d Scale.t =
    match kind with
    | Scale.Quantitative -> Scale.linear ()
    | Scale.Temporal -> Scale.time ()
    | Scale.Categorical -> Scale.band ()
  in
  let explicit =
    List.filter_map (fun m -> Option.map (fun s -> (m, s)) m.m_d.spec) ms
  in
  let implied : (d member * d Scale.t) list =
    match kind with
    | Scale.Quantitative ->
        List.concat_map
          (fun m ->
            let own =
              match m.m_imply with Some i -> [ (m, i) ] | None -> []
            in
            if String.equal m.m_role "size" then
              own @ [ (m, Scale.linear ~zero:true ()) ]
            else own)
          ms
    | Scale.Categorical ->
        List.filter_map
          (fun m ->
            match m.m_role with
            | "y" | "y2" -> Some (m, Scale.band ~reverse:true ())
            | _ -> None)
          ms
    | Scale.Temporal -> []
  in
  (* [imply] keeps the name and transform of the explicit specification, so an
     implied one gives only its other properties, and only those can
     conflict. *)
  let implied = List.map (fun (m, i) -> (m, Scale.imply i base)) implied in
  let explicit =
    Option.value ~default:base (merge_level sid "explicit" explicit)
  in
  let spec =
    match merge_level sid "implied" implied with
    | None -> explicit
    | Some i -> Scale.imply i explicit
  in
  (* Labelled and indexed categories identify categories differently. *)
  (match kind with
  | Scale.Categorical -> (
      let sort m = labelled m.m_d.lift in
      (match ms with
      | m :: rest -> (
          match List.find_opt (fun m' -> sort m' <> sort m) rest with
          | Some m' ->
              err "resolve"
                "%a and %a read labelled and indexed categories on the scale %S"
                pp_id m.m_occ.mid pp_id m'.m_occ.mid sid.sname
          | None -> ())
      | [] -> ());
      match (explicit_domain spec, ms) with
      | Some (Scale.Categories c), m :: _ ->
          let domain_labelled =
            match c with Scale.Labels _ -> true | Scale.Indices _ -> false
          in
          if domain_labelled <> sort m then
            err "resolve"
              "the domain of the scale %S and %a identify categories \
               differently"
              sid.sname pp_id m.m_occ.mid
      | _ -> ())
  | Scale.Quantitative | Scale.Temporal -> ());
  spec

let merged_guide sid ms =
  let rec go acc = function
    | [] -> Option.map snd acc
    | m :: rest -> (
        match (m.m_guide, acc) with
        | None, _ -> go acc rest
        | Some g, None -> go (Some (m, g)) rest
        | Some g, Some (m0, g0) ->
            if Bool.equal g g0 then go acc rest
            else
              err "resolve" "%a and %a imply different guides for the scale %S"
                pp_id m0.m_occ.mid pp_id m.m_occ.mid sid.sname)
  in
  go None ms

(* Summaries: what fitting needs of an occurrence's data, computed where it
   lives and read to the host at once. *)

type input =
  | In : {
      index : int;
      role : string;
      colour : bool; (* A missing value keeps its row. *)
      lift : 'd lift;
      spec : 'd Scale.t; (* Finds the missing values. *)
      fitted : bool; (* Its hull or kept codes are summarised. *)
    }
      -> input

type summary = {
  hulls : (int * (float * float)) list;
      (* Per binding index, when some value is kept. *)
  codes : (int * int list) list; (* Per binding index, increasing. *)
  notes : string list; (* Problems with the data, for warnings. *)
}

type probe = {
  miss : Nx.bool_t option; (* Broadcasts to the mark's shape. *)
  values : Nx.float64_t option;
  icodes : Nx.int64_t option;
  counts : (Nx.float64_t * (int -> string)) list;
}

let count m = Nx.sum (Nx.cast Nx.float64 m)

(* [counted noun ppf k] formats [k noun] with its verb, such as [1 code is]. *)
let counted noun ppf k =
  if k = 1 then Format.fprintf ppf "1 %s is" noun
  else Format.fprintf ppf "%d %ss are" k noun

let ( ||| ) m m' =
  match (m, m') with
  | None, m | m, None -> m
  | Some a, Some b -> Some (Nx.logical_or a b)

let invalid valid = Option.map Nx.logical_not valid

(* [along shape a t] is the vector [t] of the length of axis [a], shaped to
   broadcast along that axis of [shape]. *)
let along shape a t =
  Nx.reshape
    (Array.init
       (Array.length shape - a)
       (fun i -> if i = 0 then shape.(a) else 1))
    t

let beyond_int : type a b. (a, b) Nx.dtype -> Nx.int64_t -> Nx.bool_t option =
 fun dtype c ->
  let max = Int64.of_int max_int and min = Int64.of_int min_int in
  match dtype with
  | Nx.Int64 -> Some (Nx.logical_or (Nx.greater_s c max) (Nx.less_s c min))
  | Nx.UInt64 -> Some (Nx.logical_or (Nx.less_s c 0L) (Nx.greater_s c max))
  | _ -> None

(* [absent ints c] is [true] where the code [c] is none of the increasing
   [ints]. *)
let absent ints c =
  let k = Array.length ints in
  if k = 0 then Nx.full_like (Nx.cast Nx.bool c) true
  else
    let d = Nx.create Nx.int64 [| k |] (Array.map Int64.of_int ints) in
    let pos =
      Nx.clamp ~max:(Int64.of_int (k - 1)) (Nx.searchsorted ~side:`Left d c)
    in
    Nx.not_equal (Nx.take ~indices:pos d) c

let quantities ?valid role spec v =
  let m = Scale.missing spec v in
  let undefined = Nx.logical_and m (Nx.isfinite v) in
  let undefined =
    match valid with
    | None -> undefined
    | Some ok -> Nx.logical_and undefined ok
  in
  let note k =
    Format.asprintf "%s: %a missing for its scale" role (counted "finite value")
      k
  in
  {
    miss = Some m ||| invalid valid;
    values = Some v;
    icodes = None;
    counts = [ (count undefined, note) ];
  }

let probe : type d. int array -> string -> d lift -> d Scale.t -> probe =
 fun shape role lift spec ->
  match lift with
  | Num { x; valid } -> quantities ?valid role spec (Nx.cast Nx.float64 x)
  | Index k ->
      let a = Option.get (axis_of shape k) in
      let n = shape.(a) in
      quantities role spec
        (along shape a (Nx.cast Nx.float64 (Nx.arange Nx.int32 0 n 1)))
  | Scalar x -> quantities role spec (Nx.scalar Nx.float64 x)
  | Cat { codes; valid; labels = Some labels } ->
      let c = Nx.cast Nx.int64 codes in
      let n = Array.length labels in
      let out =
        Nx.logical_or (Nx.less_s c 0L) (Nx.greater_equal_s c (Int64.of_int n))
      in
      let outside =
        match valid with None -> out | Some ok -> Nx.logical_and out ok
      in
      let m =
        match explicit_domain spec with
        | Some (Scale.Categories (Scale.Labels domain)) ->
            let kept l = Array.exists (String.equal l) domain in
            let allowed = Nx.create Nx.bool [| n |] (Array.map kept labels) in
            Nx.logical_or out (Nx.logical_not (Nx.take ~indices:c allowed))
        | _ -> out
      in
      let note k =
        Format.asprintf "%s: %a outside its %d labels" role (counted "code") k n
      in
      {
        miss = Some m ||| invalid valid;
        values = None;
        icodes = None;
        counts = [ (count outside, note) ];
      }
  | Cat { codes; valid; labels = None } ->
      let c = Nx.cast Nx.int64 codes in
      let beyond = beyond_int (Nx.dtype codes) c in
      let m = beyond ||| invalid valid in
      let m =
        match explicit_domain spec with
        | Some (Scale.Categories (Scale.Indices ix)) ->
            m ||| Some (absent (Array.map fst ix) c)
        | _ -> m
      in
      let counts =
        match beyond with
        | None -> []
        | Some b ->
            let b =
              match valid with None -> b | Some ok -> Nx.logical_and b ok
            in
            let note k =
              Format.asprintf "%s: %a beyond the range of int" role
                (counted "code") k
            in
            [ (count b, note) ]
      in
      { miss = m; values = None; icodes = Some c; counts }
  | Strings a ->
      let m =
        match explicit_domain spec with
        | Some (Scale.Categories (Scale.Labels domain)) ->
            let out s = not (Array.exists (String.equal s) domain) in
            Some (Nx.create Nx.bool [| Array.length a |] (Array.map out a))
        | _ -> None
      in
      { miss = m; values = None; icodes = None; counts = [] }
  | Dim { axis; valid; _ } ->
      let a = Option.get (axis_of shape axis) in
      let m =
        match explicit_domain spec with
        | Some (Scale.Categories (Scale.Indices ix)) ->
            let out i = not (Array.exists (fun (j, _) -> j = i) ix) in
            let n = shape.(a) in
            Some (along shape a (Nx.create Nx.bool [| n |] (Array.init n out)))
        | _ -> None
      in
      { miss = m ||| invalid valid; values = None; icodes = None; counts = [] }

(* [kept_codes keep c] is the distinct codes of [c] where [keep] holds, in
   increasing order. *)
let kept_codes keep c =
  let keys =
    Nx.stack ~axis:1 [ Nx.flatten (Nx.cast Nx.int64 keep); Nx.flatten c ]
  in
  let groups = Nx.unique keys in
  let rows = Nx.to_array (Nx.take ~axis:0 ~indices:groups.first keys) in
  let codes = ref [] in
  for i = 0 to (Array.length rows / 2) - 1 do
    if rows.(2 * i) = 1L then codes := Int64.to_int rows.((2 * i) + 1) :: !codes
  done;
  List.sort_uniq Int.compare !codes

let summarise shape inputs filter =
  let probes =
    List.map
      (fun (In i as input) -> (input, probe shape i.role i.lift i.spec))
      inputs
  in
  let numel = Array.fold_left ( * ) 1 shape in
  let full t = Nx.broadcast_to shape t in
  let dropped =
    List.fold_left
      (fun acc (In i, p) -> if i.colour then acc else acc ||| p.miss)
      None probes
  in
  let keep =
    match dropped with
    | None -> Nx.full Nx.bool shape true
    | Some d -> Nx.logical_not (full d)
  in
  let keep =
    match filter with None -> keep | Some f -> Nx.logical_and keep (full f)
  in
  let kept p =
    match p.miss with
    | None -> keep
    | Some m -> Nx.logical_and keep (Nx.logical_not (full m))
  in
  let hulled =
    if numel = 0 then []
    else
      List.filter_map
        (fun (In i, p) ->
          match p.values with
          | Some v when i.fitted ->
              let k = kept p and v = full v in
              let lo = Nx.min (Nx.where k v (Nx.full_like v Float.infinity)) in
              let hi =
                Nx.max (Nx.where k v (Nx.full_like v Float.neg_infinity))
              in
              Some (i.index, lo, hi)
          | _ -> None)
        probes
  in
  let counts = List.concat_map (fun (_, p) -> p.counts) probes in
  let scalars =
    List.concat_map (fun (_, lo, hi) -> [ lo; hi ]) hulled @ List.map fst counts
  in
  let host = match scalars with [] -> [||] | l -> Nx.to_array (Nx.stack l) in
  let hulls =
    List.mapi
      (fun k (index, _, _) -> (index, (host.(2 * k), host.((2 * k) + 1))))
      hulled
    |> List.filter (fun (_, (lo, hi)) -> lo <= hi)
  in
  let off = 2 * List.length hulled in
  let notes =
    List.concat
      (List.mapi
         (fun k (_, note) ->
           let n = int_of_float host.(off + k) in
           if n > 0 then [ note n ] else [])
         counts)
  in
  let codes =
    if numel = 0 then []
    else
      List.filter_map
        (fun (In i, p) ->
          match p.icodes with
          | Some c when i.fitted -> Some (i.index, kept_codes (kept p) (full c))
          | _ -> None)
        probes
  in
  { hulls; codes; notes }

(* Fitting *)

let by_order ms =
  List.stable_sort
    (fun m m' ->
      let c = Int.compare m.m_occ.order m'.m_occ.order in
      if c <> 0 then c else Int.compare m.m_index m'.m_index)
    ms

let categories sid (ms : string member list) summary_of =
  match ms with
  | m :: _ when labelled m.m_d.lift ->
      let seen = Hashtbl.create 16 and labels = ref [] in
      let add l =
        if not (Hashtbl.mem seen l) then (
          Hashtbl.add seen l ();
          labels := l :: !labels)
      in
      List.iter
        (fun m ->
          match m.m_d.lift with
          | Cat { labels = Some l; _ } | Strings l -> Array.iter add l
          | Cat { labels = None; _ } | Dim _ -> ())
        (by_order ms);
      Scale.Labels (Array.of_list (List.rev !labels))
  | _ ->
      let ints = Hashtbl.create 16 and texts = Hashtbl.create 16 in
      let text m i s =
        match Hashtbl.find_opt texts i with
        | Some (s0, m0) when not (String.equal s s0) ->
            err "resolve"
              "%a and %a give the category %d of the scale %S two texts, %S \
               and %S"
              pp_id m0.m_occ.mid pp_id m.m_occ.mid i sid.sname s0 s
        | Some _ -> ()
        | None -> Hashtbl.add texts i (s, m)
      in
      List.iter
        (fun m ->
          match m.m_d.lift with
          | Dim { axis; labels; _ } ->
              let shape = m.m_occ.mark.shape in
              let a = Option.get (axis_of shape axis) in
              for i = 0 to shape.(a) - 1 do
                Hashtbl.replace ints i ();
                Option.iter (fun l -> text m i l.(i)) labels
              done
          | Cat { labels = None; _ } ->
              Option.iter
                (List.iter (fun i -> Hashtbl.replace ints i ()))
                (List.assoc_opt m.m_index (summary_of m.m_occ m.m_pid).codes)
          | Cat { labels = Some _; _ } | Strings _ -> ())
        (by_order ms);
      let ints =
        List.sort Int.compare (Hashtbl.fold (fun i () acc -> i :: acc) ints [])
      in
      let shown i =
        match Hashtbl.find_opt texts i with
        | Some (s, _) -> s
        | None -> string_of_int i
      in
      Scale.Indices (Array.of_list (List.map (fun i -> (i, shown i)) ints))

let fit_scale : type d.
    d Scale.kind ->
    sid ->
    d member list ->
    d Scale.t ->
    (occ -> id -> summary) ->
    d Scale.t =
 fun kind sid ms spec summary_of ->
  match kind with
  | Scale.Quantitative ->
      let hull acc m =
        match
          (List.assoc_opt m.m_index (summary_of m.m_occ m.m_pid).hulls, acc)
        with
        | None, acc -> acc
        | Some h, None -> Some h
        | Some (lo, hi), Some (a, b) -> Some (Float.min a lo, Float.max b hi)
      in
      let observed = List.fold_left hull None ms in
      Scale.fit
        (Option.map (fun (lo, hi) -> Scale.Floats (lo, hi)) observed)
        spec
  | Scale.Categorical ->
      Scale.fit (Some (Scale.Categories (categories sid ms summary_of))) spec
  | Scale.Temporal -> Scale.fit None spec

type fitted =
  | F : {
      sid : sid;
      key : key;
      kind : 'd Scale.kind;
      members : 'd member list;
      legend : bool;
      guide : bool option;
      spec : 'd Scale.t; (* Merged. *)
      scale : 'd Scale.t; (* Fitted, then zoomed. *)
    }
      -> fitted

let category_names (s : string Scale.t) =
  match Scale.domain s with
  | Scale.Categories (Scale.Labels l) -> Array.to_list l
  | Scale.Categories (Scale.Indices ix) ->
      List.map (fun (i, _) -> string_of_int i) (Array.to_list ix)

(* Facets *)

type facet_panel = { pnid : id; pfy : string option; pfx : string option }
type presence = Everywhere | Nowhere | Rows of Nx.bool_t

let rec const_value : type d r. (d, r) channel -> r option = function
  | Const v -> Some v
  | Map (f, c) -> Option.map f (const_value c)
  | Data _ -> None

(* [presence shape mark role cat] is where the rows of [mark] are in the panels
   of the category [cat] of the facet [role]. *)
let presence shape mark role cat =
  match find_binding role mark.bindings with
  | None -> Everywhere
  | Some (B b) -> (
      match equal_range b.role.range Panels with
      | None -> Everywhere
      | Some Type.Equal -> (
          match (const_value b.ch, data b.ch) with
          | Some v, _ -> if String.equal v cat then Everywhere else Nowhere
          | None, None -> Everywhere
          | None, Some d -> (
              let index = int_of_string_opt cat in
              let equal_code codes i =
                Rows (Nx.equal_s (Nx.cast Nx.int64 codes) (Int64.of_int i))
              in
              match d.lift with
              | Dim { axis; _ } ->
                  let a = Option.get (axis_of shape axis) in
                  let n = shape.(a) in
                  let rows = Array.init n (fun i -> index = Some i) in
                  Rows (along shape a (Nx.create Nx.bool [| n |] rows))
              | Cat { codes; labels = Some l; _ } -> (
                  let rec find i =
                    if i >= Array.length l then None
                    else if String.equal l.(i) cat then Some i
                    else find (i + 1)
                  in
                  match find 0 with
                  | Some i -> equal_code codes i
                  | None -> Nowhere)
              | Cat { codes; labels = None; _ } -> (
                  match index with
                  | Some i -> equal_code codes i
                  | None -> Nowhere)
              | Strings a ->
                  Rows
                    (Nx.create Nx.bool
                       [| Array.length a |]
                       (Array.map (String.equal cat) a))
              | Num _ | Index _ | Scalar _ -> Everywhere)))

let both p p' =
  match (p, p') with
  | Nowhere, _ | _, Nowhere -> Nowhere
  | Everywhere, p | p, Everywhere -> p
  | Rows r, Rows r' -> Rows (Nx.logical_and r r')

(* The scale of the facet [role] read in the cell [pid]. *)
let facet_scale fitted pid role : string Scale.t option =
  List.find_map
    (fun (F f) : string Scale.t option ->
      match Scale.equal_kind f.kind Scale.Categorical with
      | Some Type.Equal ->
          if
            List.exists
              (fun m ->
                Nx.Ptree.Path.equal m.m_pid pid && String.equal m.m_role role)
              f.members
          then Some f.scale
          else None
      | None -> None)
    fitted

let facet_panels fitted pid =
  let fx = facet_scale fitted pid "fx" and fy = facet_scale fitted pid "fy" in
  (match (fx, fy) with
  | Some s, Some _ when Option.is_some (Scale.wrap s) ->
      err "resolve" "the fx scale of %a wraps, but the cell has an fy scale"
        pp_id pid
  | _ -> ());
  let cats = function
    | None -> [ None ]
    | Some s -> List.map Option.some (category_names s)
  in
  match (fx, fy) with
  | None, None -> [ { pnid = pid; pfy = None; pfx = None } ]
  | _ ->
      List.concat_map
        (fun pfy ->
          List.map
            (fun pfx ->
              let add c id =
                match c with
                | None -> id
                | Some c -> Nx.Ptree.Path.add (Field c) id
              in
              let pnid =
                Nx.Ptree.Path.add (Field "panel") pid |> add pfy |> add pfx
              in
              { pnid; pfy; pfx })
            (cats fx))
        (cats fy)

(* Resolved figures *)

type spec = Sp : 'd Scale.t -> spec

type entry = {
  e_mark : mark;
  e_inputs : input list;
  e_filter : (string option * string option) option;
  e_summary : summary;
}

type resolved = {
  figure : t;
  view : View.t;
  shaped : shaped;
  facets : (id * facet_panel list) list;
  scales : fitted list; (* In the order of their first readers. *)
  nodes : (id * (id * shares) list) list;
      (* Each node with the cells it lies in and the scopes it reads there. *)
  warnings : warning list;
  cache : entry list;
}

let equal_input (In i) (In i') =
  Int.equal i.index i'.index
  && Bool.equal i.fitted i'.fitted
  && Bool.equal i.colour i'.colour
  &&
  match Scale.equal_kind (Scale.kind i.spec) (Scale.kind i'.spec) with
  | Some Type.Equal -> Scale.equal i.spec i'.spec
  | None -> false

let equal_filter =
  Option.equal (fun (a, b) (a', b') ->
      Option.equal String.equal a a' && Option.equal String.equal b b')

let default_spec : type d. d Scale.kind -> d Scale.t = function
  | Scale.Quantitative -> Scale.linear ()
  | Scale.Temporal -> Scale.time ()
  | Scale.Categorical -> Scale.band ()

let inputs_of specs occ pid =
  List.concat
    (List.mapi
       (fun index (B b) ->
         match data b.ch with
         | None -> []
         | Some d ->
             let colour =
               match b.role.range with Colors -> true | _ -> false
             in
             let kind = lift_kind d.lift in
             let found =
               List.find_map
                 (fun ((mid, pid', i), sp) ->
                   if
                     Int.equal i index
                     && Nx.Ptree.Path.equal mid occ.mid
                     && Nx.Ptree.Path.equal pid pid'
                   then Some sp
                   else None)
                 specs
             in
             let spec, fitted =
               match found with
               | None -> (default_spec kind, false)
               | Some (Sp s) -> (
                   match Scale.equal_kind (Scale.kind s) kind with
                   | Some Type.Equal -> (s, true)
                   | None ->
                       assert
                         false (* A reading's scale has the reading's kind. *))
             in
             [
               In
                 {
                   index;
                   role = b.role.name;
                   colour;
                   lift = d.lift;
                   spec;
                   fitted;
                 };
             ])
       occ.mark.bindings)

let find_path id l =
  List.find_map
    (fun (id', v) -> if Nx.Ptree.Path.equal id id' then Some v else None)
    l

type lookup =
  | No_node
  | No_scope (* No one scope of the name holds the node. *)
  | Found of fitted option

(* [scope_of places name] is the scope of [name] that holds a node lying in
   [places], if one scope holds it in every cell. *)
let scope_of places name =
  let key (pid, shares) = key_of { shares; pending = []; cell = pid } name in
  match List.map key places with
  | k :: ks when List.for_all (equal_key k) ks -> Some k
  | _ -> None

(* [find_scale scales nodes ~at name t] is the scale [name] of the kind [t] in
   the scope holding [at]. A scale that a mark makes independent per panel is
   held by the mark, and by the facet panel it is in: [Panel (mid, p)] is in a
   facet panel iff [p] is not the cell its readers are in. *)
let find_scale scales nodes ~at name t =
  let sid = sid name t in
  let held (F f) =
    equal_sid f.sid sid
    &&
    match f.key with
    | Panel (mid, p) ->
        Nx.Ptree.Path.equal mid at
        || Nx.Ptree.Path.equal p at
           && not
                (List.exists (fun m -> Nx.Ptree.Path.equal m.m_pid p) f.members)
    | _ -> false
  in
  match List.filter held scales with
  | [ f ] -> Found (Some f)
  | _ :: _ :: _ -> No_scope
  | [] -> (
      match find_path at nodes with
      | None -> No_node
      | Some places -> (
          match scope_of places name with
          | None -> No_scope
          | Some key ->
              Found
                (List.find_opt
                   (fun (F f) -> equal_sid f.sid sid && equal_key f.key key)
                   scales)))

let zoom_domain : type d. d Scale.kind -> d * d -> d Scale.domain =
 fun kind (a, b) ->
  match kind with
  | Scale.Quantitative -> Scale.Floats (a, b)
  | Scale.Temporal -> Scale.Instants (a, b)
  | Scale.Categorical ->
      assert false (* View.zoom refuses categorical scales. *)

type zoom =
  | Z : {
      at : id;
      name : string;
      sid : sid;
      key : key;
      kind : 'd Scale.kind;
      domain : 'd Scale.domain;
    }
      -> zoom

(* [zoom view nodes scales] is [scales] with the zooms of [view] applied, and
   the warnings of the zooms it ignores. *)
let zoom view nodes scales =
  let warnings = ref [] in
  let warn at fmt =
    Format.kasprintf (fun s -> warnings := (at, s) :: !warnings) fmt
  in
  let target (ident, View.V (sort, v)) =
    match (ident, sort) with
    | View.Zoom_of { scale = name; at }, View.Zoom kind -> (
        match v with
        | None -> None
        | Some ends -> (
            match find_scale scales nodes ~at name (tag kind) with
            | Found (Some (F f))
              when Option.is_some (Scale.equal_kind f.kind kind) ->
                let domain = zoom_domain kind ends in
                Some (Z { at; name; sid = f.sid; key = f.key; kind; domain })
            | No_scope ->
                warn at "no scope of the scale %S holds the zoom's node" name;
                None
            | No_node | Found _ ->
                warn at "the zoom of the scale %S applies to no scale" name;
                None))
    | _ -> None
  in
  let zooms = List.filter_map target view in
  let apply (F f) =
    match
      List.filter
        (fun (Z z) -> equal_sid z.sid f.sid && equal_key z.key f.key)
        zooms
    with
    | [] -> F f
    | [ Z z ] -> (
        match Scale.equal_kind z.kind f.kind with
        | None -> F f
        | Some Type.Equal -> (
            match Scale.with_domain z.domain f.scale with
            | scale -> F { f with scale }
            | exception Invalid_argument _ ->
                warn z.at
                  "the zoom of the scale %S sets a domain it cannot take" z.name;
                F f))
    | zs ->
        List.iter
          (fun (Z z) ->
            warn z.at
              "the scale %S is zoomed several times; its zooms are ignored"
              z.name)
          zs;
        F f
  in
  let scales = List.map apply scales in
  (scales, List.rev !warnings)

let unread view reads =
  List.filter_map
    (fun (ident, View.V (sort, _)) ->
      match ident with
      | View.User name ->
          let read (Read (i, s)) =
            View.equal_ident i ident && Option.is_some (View.equal_sort s sort)
          in
          if List.exists read reads then None
          else
            Some
              ( Nx.Ptree.Path.root,
                Format.asprintf
                  "the view sets %S, which no key of its sort reads" name )
      | View.Zoom_of _ -> None)
    view

let check_legends cells scales =
  let legends =
    List.concat_map
      (fun (_, c) ->
        List.filter_map
          (function G_legend l -> Some l | G_axis _ -> None)
          c.guides)
      cells
  in
  List.iter
    (fun l ->
      let stands (F f) =
        String.equal f.sid.sname l.lscale && equal_key f.key l.lkey && f.legend
      in
      if not (List.exists stands scales) then
        err "resolve"
          "the legend %a names %S, no scale with a legend in its scope" pp_id
          l.lid l.lscale;
      List.iter
        (fun l' ->
          if
            String.equal l.lscale l'.lscale
            && equal_key l.lkey l'.lkey
            && not (equal_legend l l')
          then
            err "resolve" "%a and %a are two different legends for %S" pp_id
              l.lid pp_id l'.lid l.lscale)
        legends)
    legends

let dedupe ws =
  let seen (id, s) =
    List.exists (fun (id', s') ->
        Nx.Ptree.Path.equal id id' && String.equal s s')
  in
  List.rev
    (List.fold_left (fun acc w -> if seen w acc then acc else w :: acc) [] ws)

let resolve ?prev ?(view = View.empty) figure =
  let st = { view; reads = []; marks = 0 } in
  let f = force st figure in
  (match names f with
  | _ :: _ :: _ -> err "resolve" "the root has two names"
  | _ -> ());
  let root = Nx.Ptree.Path.root in
  let tree = expand st root f in
  let nodes = ref [] in
  let shaped =
    arrange nodes { shares = []; pending = []; cell = root } ~in_cell:false tree
  in
  let cells = panels root shaped in
  check_coords cells;
  let occs =
    List.concat_map (fun (pid, c) -> List.map (fun o -> (pid, o)) c.occs) cells
  in
  let readings = List.concat_map (fun (pid, o) -> readings_of pid o) occs in
  check_panel_scales readings;
  check_axes cells readings;
  let unfitted =
    List.map
      (fun (G g) ->
        let spec = merged g.kind g.sid g.members in
        let guide = merged_guide g.sid g.members in
        F
          {
            sid = g.sid;
            key = g.key;
            kind = g.kind;
            members = g.members;
            legend = g.legend;
            guide;
            spec;
            scale = spec;
          })
      (group readings)
  in
  let specs =
    List.concat_map
      (fun (F f) ->
        List.map
          (fun m -> ((m.m_occ.mid, m.m_pid, m.m_index), Sp f.spec))
          f.members)
      unfitted
  in
  (* Summaries, reused from [prev] where their inputs are the same. *)
  let old = match prev with None -> [] | Some r -> r.cache in
  let fresh = ref [] in
  let summary occ pid filter =
    let inputs = inputs_of specs occ pid in
    let fkey = Option.map fst filter in
    let hit e =
      equal_mark e.e_mark occ.mark
      && List.equal equal_input e.e_inputs inputs
      && equal_filter e.e_filter fkey
    in
    match List.find_opt hit !fresh with
    | Some e -> e.e_summary
    | None ->
        let e =
          match List.find_opt hit old with
          | Some e -> e
          | None ->
              let mask = Option.bind filter snd in
              {
                e_mark = occ.mark;
                e_inputs = inputs;
                e_filter = fkey;
                e_summary = summarise occ.mark.shape inputs mask;
              }
        in
        fresh := e :: !fresh;
        e.e_summary
  in
  let base =
    List.map (fun (pid, o) -> ((o.mid, pid), summary o pid None)) occs
  in
  let summary_of (occ : occ) pid =
    snd
      (List.find
         (fun ((mid, pid'), _) ->
           Nx.Ptree.Path.equal mid occ.mid && Nx.Ptree.Path.equal pid pid')
         base)
  in
  let notes =
    List.concat_map
      (fun ((mid, _), s) -> List.map (fun n -> (mid, n)) s.notes)
      base
  in
  let fit summary_of (F f) =
    F { f with scale = fit_scale f.kind f.sid f.members f.spec summary_of }
  in
  let per_panel (F f) = match f.key with Panels_of _ -> true | _ -> false in
  (* Categorical scales first, since facets make panels. *)
  let fitted =
    List.map
      (fun (F f as s) ->
        match f.kind with
        | Scale.Categorical when not (per_panel s) -> fit summary_of s
        | _ -> s)
      unfitted
  in
  let facets =
    List.map (fun (pid, _) -> (pid, facet_panels fitted pid)) cells
  in
  let constants =
    List.concat_map
      (fun (pid, c) ->
        let check o role =
          match find_binding role o.mark.bindings with
          | None -> None
          | Some (B b) -> (
              match equal_range b.role.range Panels with
              | None -> None
              | Some Type.Equal -> (
                  match const_value b.ch with
                  | None -> None
                  | Some v ->
                      let cats =
                        Option.fold ~none:[] ~some:category_names
                          (facet_scale fitted pid role)
                      in
                      if List.mem v cats then None
                      else
                        Some
                          ( o.mid,
                            Format.asprintf
                              "the facet constant %S of %s names no panel" v
                              role )))
        in
        List.concat_map
          (fun o -> List.filter_map (check o) [ "fx"; "fy" ])
          c.occs)
      cells
  in
  let panel_scales (F f as s) =
    match f.key with
    | Panels_of (mid, pid) ->
        let occ = (List.hd f.members).m_occ in
        let at c role =
          match c with
          | None -> Everywhere
          | Some c -> presence occ.mark.shape occ.mark role c
        in
        List.filter_map
          (fun p ->
            match both (at p.pfy "fy") (at p.pfx "fx") with
            | Nowhere -> None
            | rows ->
                let mask =
                  match rows with
                  | Rows r -> Some r
                  | Everywhere | Nowhere -> None
                in
                let summary_of _ _ =
                  summary occ pid (Some ((p.pfy, p.pfx), mask))
                in
                Some (fit summary_of (F { f with key = Panel (mid, p.pnid) })))
          (Option.value ~default:[] (find_path pid facets))
    | _ -> [ s ]
  in
  let fitted = List.concat_map panel_scales fitted in
  let fitted =
    List.map
      (fun (F f as s) ->
        match (f.kind, f.key) with
        | Scale.Categorical, _ | _, Panel _ -> s
        | _ -> fit summary_of s)
      fitted
  in
  (* Each node with the cells it lies in; a facet panel lies in its cell and
     reads the scopes of the cell. *)
  let nodes =
    let held =
      List.concat_map
        (fun (pid, c) -> List.map (fun (id, sh) -> (id, (pid, sh))) c.held)
        cells
    in
    let panels =
      List.concat_map
        (fun (pid, ps) ->
          let shares =
            Option.value ~default:[]
              (Option.bind (find_path pid cells) (fun c -> find_path pid c.held))
          in
          List.filter_map
            (fun p ->
              if Nx.Ptree.Path.equal p.pnid pid then None
              else Some (p.pnid, (pid, shares)))
            ps)
        facets
    in
    let places = held @ panels in
    let ids =
      List.fold_left
        (fun acc id ->
          if List.exists (Nx.Ptree.Path.equal id) acc then acc else id :: acc)
        []
        (List.map fst places @ List.rev !nodes)
    in
    List.rev_map
      (fun id ->
        ( id,
          List.filter_map
            (fun (id', p) ->
              if Nx.Ptree.Path.equal id id' then Some p else None)
            places ))
      ids
  in
  let scales, zooms = zoom view nodes fitted in
  check_legends cells scales;
  let warnings = dedupe (notes @ constants @ zooms @ unread view st.reads) in
  {
    figure;
    view;
    shaped;
    facets;
    scales;
    nodes;
    warnings;
    cache = List.rev !fresh;
  }

module Resolved = struct
  type t = resolved

  let scale : type d. ?at:id -> t -> d Scale.t -> d Scale.t =
   fun ?(at = Nx.Ptree.Path.root) r s ->
    let name =
      match Scale.name s with
      | Some n -> n
      | None -> err "Resolved.scale" "the scale is unnamed"
    in
    let kind = Scale.kind s in
    let absent () =
      err "Resolved.scale" "the scope of %a has no %a scale %S" pp_id at pp_tag
        (tag kind) name
    in
    match find_scale r.scales r.nodes ~at name (tag kind) with
    | No_node -> err "Resolved.scale" "no node has the id %a" pp_id at
    | No_scope ->
        err "Resolved.scale" "no scope of the scale %S holds %a" name pp_id at
    | Found None -> absent ()
    | Found (Some (F f)) -> (
        match Scale.equal_kind f.kind kind with
        | Some Type.Equal -> f.scale
        | None -> absent ())

  let warnings r = r.warnings
  let panel_of = function Panel (_, p) -> Some p | _ -> None

  let equal_member m m' =
    Nx.Ptree.Path.equal m.m_occ.mid m'.m_occ.mid
    && String.equal m.m_role m'.m_role
    && Nx.Ptree.Path.equal m.m_pid m'.m_pid

  let equal_scale (F f) (F f') =
    equal_sid f.sid f'.sid
    && Option.equal Nx.Ptree.Path.equal (panel_of f.key) (panel_of f'.key)
    && Option.equal Bool.equal f.guide f'.guide
    &&
    match Scale.equal_kind f.kind f'.kind with
    | Some Type.Equal ->
        List.equal equal_member f.members f'.members
        && Scale.equal f.scale f'.scale
    | None -> false

  let equal_warning (id, s) (id', s') =
    Nx.Ptree.Path.equal id id' && String.equal s s'

  let equal r r' =
    equal r.figure r'.figure && View.equal r.view r'.view
    && List.equal equal_scale r.scales r'.scales
    && List.equal equal_warning r.warnings r'.warnings

  (* Formatting *)

  let pp_side ppf (side : side) =
    Format.pp_print_string ppf
      (match side with
      | `Left -> "left"
      | `Right -> "right"
      | `Top -> "top"
      | `Bottom -> "bottom")

  let pp_guide ppf = function
    | G_axis a ->
        Format.fprintf ppf "axis %S%a%s%s" a.scale
          (Format.pp_print_option (fun ppf s ->
               Format.fprintf ppf " %a" pp_side s))
          a.side
          (if a.grid then " grid" else "")
          (if a.show then "" else " hidden")
    | G_legend l ->
        Format.fprintf ppf "legend %S%a%s" l.lscale
          (Format.pp_print_option (fun ppf s ->
               Format.fprintf ppf " %a" pp_side s))
          l.lside
          (if l.lshow then "" else " hidden")

  let pp_title ppf ((align : Text.Layout.halign), t) =
    Format.fprintf ppf "title %a%s" Text.pp t
      (match align with `Center -> "" | `Left -> " left" | `Right -> " right")

  let pp_weights name ppf = function
    | None -> ()
    | Some ws ->
        Format.fprintf ppf ", %s %a" name
          (Format.pp_print_list
             ~pp_sep:(fun ppf () -> Format.pp_print_string ppf " ")
             (fun ppf w -> Format.fprintf ppf "%g" w))
          ws

  let pp_ids =
    Format.pp_print_list ~pp_sep:(fun ppf () -> Format.fprintf ppf ",@ ") pp_id

  let rec pp_shaped facets ppf (pid, s) =
    match s.body with
    | Single c ->
        Format.fprintf ppf "@[<v 2>panel %a" pp_id pid;
        List.iter (Format.fprintf ppf "@,%a" pp_title) s.titles;
        (match c.coords with
        | (_, k) :: _ -> Format.fprintf ppf "@,coord %a" Coord.pp k
        | [] -> ());
        List.iter
          (fun o -> Format.fprintf ppf "@,%s %a" o.mark.kind pp_id o.mid)
          c.occs;
        List.iter (Format.fprintf ppf "@,%a" pp_guide) c.guides;
        (match find_path pid facets with
        | Some [ p ] when Nx.Ptree.Path.equal p.pnid pid -> ()
        | Some ps ->
            Format.fprintf ppf "@,@[<hov 2>facets %a@]" pp_ids
              (List.map (fun p -> p.pnid) ps)
        | None -> ());
        Format.fprintf ppf "@]"
    | Arr a ->
        Format.fprintf ppf "@[<v 2>grid %a, %d × %d%a%a" pp_id a.aid a.nrows
          a.ncols (pp_weights "widths") a.widths (pp_weights "heights")
          a.heights;
        List.iter (Format.fprintf ppf "@,%a" pp_title) s.titles;
        List.iter
          (fun cell ->
            Format.fprintf ppf "@,@[<v 2>cell (%d, %d)%s@,%a@]" cell.row
              cell.col
              (if cell.rows = 1 && cell.cols = 1 then ""
               else Format.asprintf ", spanning %d × %d" cell.rows cell.cols)
              (pp_shaped facets) (cell.cid, cell.s))
          a.cells;
        Format.fprintf ppf "@]"

  let pp_scale ppf (F f) =
    let readers =
      List.fold_left
        (fun acc m ->
          let r = Format.asprintf "%a:%s" pp_id m.m_occ.mid m.m_role in
          if List.mem r acc then acc else r :: acc)
        [] f.members
      |> List.rev
    in
    Format.fprintf ppf "@[<v 2>%S %a%a, read by @[<hov>%a@]@,%a%a@]" f.sid.sname
      pp_tag (tag f.kind)
      (Format.pp_print_option (fun ppf p -> Format.fprintf ppf " in %a" pp_id p))
      (panel_of f.key)
      (Format.pp_print_list
         ~pp_sep:(fun ppf () -> Format.fprintf ppf ",@ ")
         Format.pp_print_string)
      readers Scale.pp f.scale
      (Format.pp_print_option (fun ppf g -> Format.fprintf ppf "@,guide %b" g))
      f.guide

  let pp ppf r =
    Format.fprintf ppf "@[<v>@[<v 2>figure@,%a@]" (pp_shaped r.facets)
      (Nx.Ptree.Path.root, r.shaped);
    Format.fprintf ppf "@,@[<v 2>scales";
    List.iter (Format.fprintf ppf "@,%a" pp_scale) r.scales;
    Format.fprintf ppf "@]";
    if r.warnings <> [] then (
      Format.fprintf ppf "@,@[<v 2>warnings";
      List.iter (Format.fprintf ppf "@,%a" pp_warning) r.warnings;
      Format.fprintf ppf "@]");
    Format.fprintf ppf "@]"
end

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

type panel_out = { id : id; box : Box2.t; projection : Coord.projection }

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

(* Items *)

type guide_role = Gx | Gy | Gfx | Gfy

type axis_spec = {
  a_id : id;
  a_scale : int;
  a_role : guide_role;
  a_side : side;
  a_grid : bool;
  a_show : bool;
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
  | Heading of { align : Text.Layout.halign; head : Text.t; hside : side }
  | Legend of legend_spec

and grid = {
  gid : id;
  gcols : track array;
  grows : track array;
  gcells : gcell list;
  gbody : int option; (* The cell headings align on. *)
}

and gcell = { r0 : int; c0 : int; nr : int; nc : int; it : item }

(* How layout chooses a scale's ticks. *)
type use =
  | Axis_of of guide_role
  | Header_of
  | Legend_of of { bar : bool; side : side; block : id; show : bool }

type cx = {
  theme : Theme.t;
  measures : Text.Layout.t Measures.t ref;
  scales : fitted array;
  uses : use array;
  ticks : Ticks.t option array; (* None before the first choice. *)
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
      let l =
        Text.Layout.v ~halign ~valign ~fonts:(Theme.fonts cx.theme) ~size text
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
          err "layout" "%a holds %s, which no face of the theme has" Text.pp
            p.text chars

let place_text cx owner ?halign ?valign ~turned ~data k text at =
  let p = { text; set = set cx ?halign ?valign k text; at; turned; data } in
  check cx owner p;
  p

(* Guide texts *)

let tick_text (t : Ticks.tick) =
  match t.context with
  | None -> Text.v t.label
  | Some c -> Text.v (t.label ^ "\n" ^ c)

let categorical (F f) =
  match f.kind with
  | Scale.Categorical -> true
  | Scale.Quantitative | Scale.Temporal -> false

(* [guide_title s note] is the distinct titles of the channels reading [s] in
   the order of the figure, separated by commas, then [note]. *)
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
        else { thick = title_h +. thick; least = 0.; above = 0. }
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
          let n = List.length labels in
          let total = List.fold_left (fun s l -> s +. entry l) 0. labels in
          {
            thick = title_h +. row;
            least = total +. (float (max 0 (n - 1)) *. em cx clear_em);
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
   times [unit] if it is flexible and at least what each cell it alone holds
   needs, with each cell spanning tracks given what it needs beyond them, by
   weight among its flexible tracks or else evenly. [cells] are the start, the
   number of tracks and the need of each cell. *)
let least tracks unit cells gaps =
  let m = Array.map (function Flex k -> k *. unit | Fixed -> 0.) tracks in
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
    if k = 0 || Array.for_all2 Bool.equal next active then s else go next (k - 1)
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
  | Leaf l ->
      List.fold_left
        (fun (w, h) a ->
          let n = spans cx a in
          if horizontal a.a_side then (Float.max w n, h) else (w, Float.max h n))
        (0., 0.) l.l_axes
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
        most before (fun c -> last c = j)
        +. most after (fun c -> first c = j + 1)
        +. em cx gap_em)
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
  let cols_least =
    least g.gcols uw
      (List.map (fun (c, _, (w, _)) -> (c.c0, c.nc, w)) cells)
      cgaps
  in
  let rows_least =
    least g.grows uh
      (List.map (fun (c, _, (_, h)) -> (c.r0, c.nr, h)) cells)
      rgaps
  in
  let aspects =
    List.filter_map
      (fun (c, _, _) ->
        match c.it with
        | Leaf { l_ratio = Some r; _ } when c.nr = 1 && c.nc = 1 -> Some (c, r)
        | _ -> None)
      cells
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
  mutable panels : (panel_out * Coord.t) list;
  mutable axes : axis_out list;
  mutable headers : header_out list;
  mutable legends : legend_out list;
  mutable titles : placed list;
  mutable spans : (id * Box2.t) list; (* The hull of each block's panels. *)
  mutable along : (int * float) list; (* Each axis's scale and length. *)
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
      | Gx -> acc.along <- (a.a_scale, Box2.w box) :: acc.along
      | Gy -> acc.along <- (a.a_scale, Box2.h box) :: acc.along
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
                  List.map (fun tk -> label ~valign:`Middle tk (at tk)) t.major;
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
                  List.map
                    (fun tk -> label ~halign:`Center ~valign:`Top tk (at tk))
                    t.major;
              }
        else
          let row = Float.max sw (longest height (entry_labels cx t)) in
          let entry (x, y, es) (tk : Ticks.tick) =
            let swatch = Box2.v x (y +. ((row -. sw) /. 2.)) sw sw in
            let label =
              label ~valign:`Middle tk
                (P2.v (x +. sw +. pad) (y +. (row /. 2.)))
            in
            let e = { u = tk.position; swatch; label } in
            if vert then (x, y +. row, e :: es)
            else
              (x +. sw +. pad +. width label.set +. em cx clear_em, y, e :: es)
          in
          let _, _, es = List.fold_left entry (x0, y0, []) t.major in
          Entries (List.rev es)
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

let place_heading cx ~align ~head ~side cell span outer =
  let owner = Nx.Ptree.Path.root in
  let span = Option.value span ~default:cell in
  let _, valign, turned = outer_align side in
  if turned then
    let x = match side with `Left -> Box2.maxx cell | _ -> Box2.minx cell in
    place_text cx owner ~halign:`Center ~valign ~turned ~data:false 1. head
      (P2.v x (P2.y (Box2.mid span)))
  else
    let x =
      match align with
      | `Center -> P2.x (Box2.mid span)
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
  | Heading { align; head; hside } ->
      acc.titles <-
        place_heading cx ~align ~head ~side:hside box span outer :: acc.titles;
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

(* Building items *)

let names_of (F f) =
  match f.kind with
  | Scale.Categorical -> category_names f.scale
  | Scale.Quantitative | Scale.Temporal -> []

(* [units s] is the length of the domain of [s] in units of its transform, or
   one if it spans none. *)
let units (F f) =
  let u = Scale.length f.scale in
  if Float.is_finite u && u > 0. then u else 1.

(* [scale_index r pid pnid role] is the scale that [role] reads in the facet
   panel [pnid] of the cell [pid]. *)
let scale_index (r : resolved) pid pnid role =
  let rec go i = function
    | [] -> None
    | F f :: rest ->
        let reads =
          List.exists
            (fun m ->
              Nx.Ptree.Path.equal m.m_pid pid
              && String.equal (axis_role m.m_role) role)
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

let explicit_axis c name =
  List.find_map
    (function G_axis a when String.equal a.scale name -> Some a | _ -> None)
    c.guides

let default_side : guide_role -> side = function
  | Gx -> `Bottom
  | Gy -> `Left
  | Gfx -> `Top
  | Gfy -> `Right

let axis_spec r scales c pid p role =
  let name =
    match role with Gx -> "x" | Gy -> "y" | Gfx -> "fx" | Gfy -> "fy"
  in
  Option.map
    (fun i ->
      let (F f) = scales.(i) in
      let sname = f.sid.sname in
      let explicit = explicit_axis c sname in
      let side =
        match explicit with
        | Some { side = Some s; _ } -> s
        | _ -> default_side role
      in
      (match (role, horizontal side) with
      | Gx, false | Gy, true ->
          err "layout" "the axis of %S of the panel %a is on its %a side" sname
            pp_id p.pnid Resolved.pp_side side
      | _ -> ());
      {
        a_id = Nx.Ptree.Path.(add (Field sname) (add (Field "axis") p.pnid));
        a_scale = i;
        a_role = role;
        a_side = side;
        a_grid = (match explicit with Some a -> a.grid | None -> false);
        a_show =
          (match explicit with
          | Some a -> a.show
          | None -> Option.value f.guide ~default:true);
        a_labelled = true;
        a_category =
          (match role with Gfx -> p.pfx | Gfy -> p.pfy | Gx | Gy -> None);
      })
    (scale_index r pid p.pnid name)

let leaf r scales pid c p =
  let axes =
    List.filter_map (axis_spec r scales c pid p) [ Gx; Gy; Gfx; Gfy ]
  in
  let coord =
    match c.coords with
    | (_, k) :: _ -> k
    | [] -> (
        match List.find_map (fun o -> o.mark.coord) c.occs with
        | Some k -> k
        | None -> Coord.cartesian ())
  in
  let units role =
    match List.find_opt (fun a -> a.a_role = role) axes with
    | Some a -> units scales.(a.a_scale)
    | None -> 1.
  in
  let (Coord.Cartesian { aspect }) = coord in
  {
    l_id = p.pnid;
    l_coord = coord;
    l_ratio = Option.map (fun k -> k *. units Gy /. units Gx) aspect;
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
                a'.a_show && a'.a_scale = a.a_scale
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

(* [blocks r] is each node of [r] that layout places as a whole, with the panels
   it holds, outer nodes first. *)
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

let uses_of (r : resolved) blocks =
  let legends =
    List.concat_map
      (fun (_, c) ->
        List.filter_map
          (function G_legend l -> Some l | G_axis _ -> None)
          c.guides)
      (panels Nx.Ptree.Path.root r.shaped)
  in
  let use (F f as s) =
    let roles = List.map (fun m -> axis_role m.m_role) f.members in
    let is n = List.exists (String.equal n) roles in
    if is "x" then Axis_of Gx
    else if is "y" then Axis_of Gy
    else if is "fx" || is "fy" then Header_of
    else
      let explicit =
        List.find_opt
          (fun l -> String.equal l.lscale f.sid.sname && equal_key l.lkey f.key)
          legends
      in
      let show =
        match explicit with
        | Some l -> l.lshow
        | None -> Option.value f.guide ~default:f.legend
      in
      let side =
        match explicit with Some { lside = Some s; _ } -> s | _ -> `Right
      in
      let colour =
        List.exists
          (fun m ->
            String.equal m.m_role "fill" || String.equal m.m_role "stroke")
          f.members
      in
      Legend_of
        {
          bar = colour && not (categorical s);
          side;
          block = block_of r blocks f.key;
          show;
        }
  in
  Array.of_list (List.map use r.scales)

let build r scales uses =
  let legends_at id =
    List.concat
      (List.mapi
         (fun i u ->
           match u with
           | Legend_of { side; block; show = true; bar }
             when Nx.Ptree.Path.equal block id ->
               let (F f) = scales.(i) in
               let ls_id =
                 Nx.Ptree.Path.(
                   add (Field f.sid.sname) (add (Field "legend") id))
               in
               [
                 ( side,
                   Legend { ls_id; ls_scale = i; ls_side = side; ls_bar = bar }
                 );
               ]
           | _ -> [])
         (Array.to_list uses))
  in
  let heading side (align, head) =
    (side, Heading { align; head; hside = side })
  in
  (* The title of a facet scale goes beside its headers. *)
  let facet_title c (i, role) =
    let (F f as s) = scales.(i) in
    let explicit = explicit_axis c f.sid.sname in
    let show =
      match explicit with
      | Some a -> a.show
      | None -> Option.value f.guide ~default:true
    in
    let side =
      match explicit with
      | Some { side = Some s; _ } -> s
      | _ -> default_side role
    in
    match guide_title s None with
    | Some t when show -> [ heading side (`Center, t) ]
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
        let fx = scale_index r pid pid "fx"
        and fy = scale_index r pid pid "fy" in
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
              | Scale.Categorical -> Scale.wrap f.scale
              | Scale.Quantitative | Scale.Temporal -> None)
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
        let facets = [ (fx, Gfx); (fy, Gfy) ] in
        let titles =
          List.concat_map
            (fun (i, role) ->
              Option.fold ~none:[] ~some:(fun i -> facet_title c (i, role)) i)
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
    let titles = List.rev_map (heading `Top) s.titles in
    wrap id titles (wrap id (legends_at id) body)
  in
  block Nx.Ptree.Path.root r.shaped

(* Choosing ticks *)

let all_ticks locale (F f) =
  match f.kind with
  | Scale.Categorical ->
      Ticks.of_values ~locale f.scale (Array.of_list (category_names f.scale))
  | Scale.Quantitative | Scale.Temporal -> Ticks.of_values ~locale f.scale [||]

(* [lengths cx acc] is the length of the guides of each scale in [acc]: the
   shortest of its axes, or the span of the panels its legend stands beside. *)
let lengths cx acc =
  Array.mapi
    (fun i u ->
      match u with
      | Header_of -> Float.nan
      | Axis_of _ ->
          List.fold_left
            (fun m (j, l) -> if j = i then Float.min m l else m)
            Float.infinity acc.along
      | Legend_of { side; block; _ } -> (
          match find_path block acc.spans with
          | None -> 0.
          | Some b -> if vertical side then Box2.h b else Box2.w b))
    cx.uses

(* [choose cx lengths] is the ticks of each scale at its length. *)
let choose cx lengths =
  let locale = Theme.locale cx.theme in
  let label t = set cx label_em (Text.v t) in
  let clear = em cx clear_em and sw = em cx swatch_em in
  Array.mapi
    (fun i (F f as s) ->
      let length = lengths.(i) in
      let pick measure =
        if Float.is_finite length && length > 0. then
          Ticks.choose ~locale ~length ~measure f.scale
        else Ticks.of_values ~locale f.scale [||]
      in
      match cx.uses.(i) with
      | Header_of -> all_ticks locale s
      | Axis_of role ->
          let measure t =
            (match role with Gx -> width (label t) | _ -> height (label t))
            +. clear
          in
          pick measure
      | Legend_of _ when categorical s -> all_ticks locale s
      | Legend_of { bar; side; _ } ->
          let vert = vertical side in
          let measure t =
            let l = label t in
            match (bar, vert) with
            | true, true -> height l +. clear
            | true, false -> width l +. clear
            | false, true -> Float.max sw (height l)
            | false, false -> sw +. em cx pad_em +. width l +. clear
          in
          pick measure)
    cx.scales

(* Laid-out figures *)

type laid = {
  resolved : resolved;
  theme : Theme.t;
  page : float * float;
  lpanels : (panel_out * Coord.t) list;
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

let layout ?prev ?(theme = Theme.default) size (r : resolved) =
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
  let measures =
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
      measures = ref measures;
      scales;
      uses;
      ticks = Array.make (Array.length scales) None;
      final = false;
      notes = ref [];
    }
  in
  (* Without guides, then with the first choice's; the second is frozen. *)
  let next cx acc =
    { cx with ticks = Array.map Option.some (choose cx (lengths cx acc)) }
  in
  let acc, _ = pass cx unit size root in
  let cx = next cx acc in
  let acc, _ = pass cx unit size root in
  let cx = { (next cx acc) with final = true } in
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

module Layout = struct
  type t = laid

  type panel = panel_out = {
    id : id;
    box : Box2.t;
    projection : Coord.projection;
  }

  let size l = l.page
  let panels l = List.map fst l.lpanels
  let warnings l = l.lwarnings

  let equal_panel (p, c) (p', c') =
    Nx.Ptree.Path.equal p.id p'.id
    && Box2.equal p.box p'.box && Coord.equal c c'

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

  (* Boxes are on one line, so each guide's text and box are. *)
  let pp_box ppf b =
    Format.fprintf ppf "[(%g, %g) (%g, %g)]" (Box2.minx b) (Box2.miny b)
      (Box2.maxx b) (Box2.maxy b)

  let pp_placed ppf p =
    Format.fprintf ppf "%a %a%s" Text.pp p.text pp_box (placed_box p)
      (if p.turned then " turned" else "")

  let pp_axis ppf a =
    Format.fprintf ppf "@[<v 2>axis %a %a%s" pp_id a.ax_id Resolved.pp_side
      a.ax_side
      (if a.ax_grid then " grid" else "");
    List.iter (Format.fprintf ppf "@,label %a" pp_placed) a.ax_labels;
    Option.iter (Format.fprintf ppf "@,title %a" pp_placed) a.ax_title;
    Format.fprintf ppf "@]"

  let pp_legend ppf g =
    Format.fprintf ppf "@[<v 2>legend %a %a" pp_id g.lg_id Resolved.pp_side
      g.lg_side;
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
end

(* Drawing *)

module Drawing = struct
  type t = |

  let renderable (d : t) = match d with _ -> .
  let warnings (d : t) = match d with _ -> .
  let equal (d : t) _ = match d with _ -> .
  let pp _ (d : t) = match d with _ -> .
end

let draw ?prev:_ ~density:_ _ = unimplemented "draw"

let render ?view ?theme ?(density = 2.) size f =
  draw ~density (layout ?theme size (resolve ?view f))

let save ?warn:_ ?view:_ ?theme:_ ?size:_ ?density:_ _ _ = unimplemented "save"
let pp _ _ = unimplemented "pp"
