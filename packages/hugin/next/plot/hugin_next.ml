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

  type projection = |

  let point (p : projection) _ _ = match p with _ -> .
  let invert (p : projection) _ = match p with _ -> .
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
  | "axis" | "legend" | "panel" ->
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

(* Stages *)

module Resolved = struct
  type t = |

  let scale ?at:_ (r : t) _ = match r with _ -> .
  let warnings (r : t) = match r with _ -> .
  let equal (r : t) _ = match r with _ -> .
  let pp _ (r : t) = match r with _ -> .
end

module Layout = struct
  type t = |
  type panel = { id : id; box : Box2.t; projection : Coord.projection }

  let size (l : t) = match l with _ -> .
  let panels (l : t) = match l with _ -> .
  let warnings (l : t) = match l with _ -> .
  let equal (l : t) _ = match l with _ -> .
  let pp _ (l : t) = match l with _ -> .
end

module Drawing = struct
  type t = |

  let renderable (d : t) = match d with _ -> .
  let warnings (d : t) = match d with _ -> .
  let equal (d : t) _ = match d with _ -> .
  let pp _ (d : t) = match d with _ -> .
end

let resolve ?prev:_ ?view:_ _ = unimplemented "resolve"
let layout ?prev:_ ?theme:_ _ _ = unimplemented "layout"
let draw ?prev:_ ~density:_ _ = unimplemented "draw"

let render ?view ?theme ?(density = 2.) size f =
  draw ~density (layout ?theme size (resolve ?view f))

let save ?warn:_ ?view:_ ?theme:_ ?size:_ ?density:_ _ _ = unimplemented "save"
let pp _ _ = unimplemented "pp"
