(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module P2 = Hugin_next_gg.P2
module Box2 = Hugin_next_gg.Box2
module Path = Hugin_next_gg.Path
module Stroke = Hugin_next_gg.Stroke
module Color = Hugin_next_gg.Color
module Field2 = Hugin_next_gg_kit.Field2
module Pgon2 = Hugin_next_gg_kit.Pgon2
module Text = Hugin_next_text.Text
module Picture = Hugin_next_vg.Picture
module Scale = Hugin_next_kit.Scale
module Symbol = Hugin_next_kit.Symbol
module Curve = Hugin_next_kit.Curve
open Common
open Channel
open Figure

let on = Mark.bind
let opt role = Option.map (on role)

(* Derived lengths, in em *)

let line_em = 0.15
let outline_em = 0.08
let dot_em = 0.5 (* The diameter of a dot's circle. *)
let em rows k = k *. Theme.size (Mark.theme rows)
let dot_area rows = Float.pi *. Float.pow (em rows dot_em /. 2.) 2.

(* Reading rows *)

let first_value rows role =
  match Mark.get rows role with
  | Some vs when Array.length vs > 0 -> Some vs.(0)
  | _ -> None

let or_const rows role v =
  match Mark.get rows role with
  | Some vs -> vs
  | None -> Array.make (Mark.length rows) v

(* [fade o c] is [c] at the opacity [o], or [c] if [o] is missing. *)
let fade o c =
  if Float.is_nan o then c
  else Color.with_alpha (Color.alpha c *. Float.min 1. (Float.max 0. o)) c

(* [faded rows cs] is [cs] at each row's opacity. *)
let faded rows cs =
  match Mark.get rows Role.opacity with
  | None -> cs
  | Some os -> Array.mapi (fun i c -> fade os.(i) c) cs

let finite = Float.is_finite

(* [fills rows ~default] is each row's fill, if the mark paints one: its [fill],
   or [default] unless [stroke] alone is bound. *)
let fills rows ~default =
  match (Mark.get rows Role.fill, Mark.get rows Role.stroke) with
  | Some fs, _ -> Some (faded rows fs)
  | None, Some _ -> None
  | None, None -> Some (faded rows (Array.make (Mark.length rows) default))

let strokes rows = Option.map (faded rows) (Mark.get rows Role.stroke)

(* [box_path rows (x0, x1) (y0, y1) i] is the rectangle of row [i]'s extents on
   the page. *)
let box_path rows (x0, x1) (y0, y1) i =
  let b = Box2.of_pts (P2.v x0.(i) y0.(i)) (P2.v x1.(i) y1.(i)) in
  Mark.project rows (Path.rect b)

(* A bar's band position stands apart from its neighbours. *)
let bar_padding = 0.2 (* The fraction of a step between bars. *)
let bar_band = Scale.band ~padding:bar_padding ()

(* [length ~band role ch] binds [ch], a position without its other end, to
   [role]. Quantities are a length, which implies [zero] on their scale, and
   categories a band, which implies [band] if given. *)
let length : type d r.
    ?band:string Scale.t -> (d, r) Role.t -> (d, r) Channel.t -> binding =
 fun ?band role ch ->
  match data ch with
  | Some { lift; _ } -> (
      match lift_kind lift with
      | Scale.Quantitative -> on ~imply:(Scale.linear ~zero:true ()) role ch
      | Scale.Categorical -> on ?imply:band role ch
      | Scale.Temporal -> on role ch)
  | None -> on role ch

let position ?band role ~alone = function
  | None -> None
  | Some ch -> Some (if alone then length ?band role ch else on role ch)

(* [continuous ch] is [true] iff [ch] holds data read by a continuous scale. *)
let continuous : type d r. (d, r) Channel.t option -> bool = function
  | None -> false
  | Some ch -> (
      match data ch with
      | Some { lift; _ } -> (
          match lift_kind lift with
          | Scale.Quantitative | Scale.Temporal -> true
          | Scale.Categorical -> false)
      | None -> false)

let facets fx fy = [ opt Role.fx fx; opt Role.fy fy ]

let make fn ?reduce ?coord ?base ?swatch l draw =
  make_mark fn ~name:fn ?reduce ?coord ?swatch ?base (List.filter_map Fun.id l)
    draw

(* Dots *)

let draw_dot rows =
  let th = Mark.theme rows in
  let xs, ys = Mark.points rows in
  let fills = fills rows ~default:(Theme.accent th)
  and strokes = strokes rows in
  let paint = match fills with Some _ -> `Fill | None -> `Stroke in
  let scales = Array.map Float.sqrt (or_const rows Role.size (dot_area rows)) in
  let pen = Stroke.v (em rows outline_em) in
  let glyph symbol =
    let path = Symbol.path paint 1. symbol in
    Picture.group
      [
        (match fills with
        | Some _ -> Picture.fill Color.black path
        | None -> Picture.empty);
        (match strokes with
        | Some _ -> Picture.stroke pen Color.black path
        | None -> Picture.empty);
      ]
  in
  let stamp ?fills ?strokes ~scales xs ys symbol =
    Picture.stamp ?fills ?strokes ~scales xs ys (glyph symbol)
  in
  match Mark.get rows Role.symbol with
  | None -> stamp ?fills ?strokes ~scales xs ys Symbol.circle
  | Some symbols ->
      (* One stamp per symbol, each tagged with its rows. *)
      let groups = ref [] in
      Array.iteri
        (fun i s ->
          match List.find_opt (fun (s', _) -> Symbol.equal s s') !groups with
          | Some (_, ks) -> ks := i :: !ks
          | None -> groups := (s, ref [ i ]) :: !groups)
        symbols;
      let index = Mark.index rows in
      Picture.group
        (List.rev_map
           (fun (s, ks) ->
             let ks = Array.of_list (List.rev !ks) in
             let sub a = Array.map (fun k -> a.(k)) ks in
             Picture.tag
               { Picture.id = Mark.id rows; rows = Picture.Rows (sub index) }
               (stamp ?fills:(Option.map sub fills)
                  ?strokes:(Option.map sub strokes) ~scales:(sub scales)
                  (sub xs) (sub ys) s))
           !groups)

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
       @ facets fx fy)
       draw_dot)

(* Lines *)

(* [style rows xs role ~default equal] is the value of [role] of the first row
   of the series [rows] that is not dropped, a row at [xs.(i)] being dropped
   where that is [nan], with a warning if it varies along the series. *)
let style rows xs role ~default equal =
  match Mark.get rows role with
  | None -> default
  | Some vs ->
      let first = ref (-1) and varies = ref false in
      Array.iteri
        (fun i v ->
          if finite xs.(i) then
            if !first < 0 then first := i
            else if not (equal v vs.(!first)) then varies := true)
        vs;
      if !varies then
        Mark.warn rows
          (Printf.sprintf "the %s of a line varies along a series"
             role.Role.name);
      if !first < 0 then default else vs.(!first)

(* [closed p] is [p] with every subpath closed. *)
let closed path =
  let close (p, open_) = if open_ then Path.close p else p in
  close
    (Path.fold
       ~move:(fun acc x y -> (Path.move_to (P2.v x y) (close acc), false))
       ~line:(fun (p, _) x y -> (Path.line_to (P2.v x y) p, true))
       ~cubic:(fun (p, _) a b c d x y ->
         (Path.cubic_to (P2.v a b) (P2.v c d) (P2.v x y) p, true))
       ~close:(fun (p, _) -> (Path.close p, false))
       (Path.empty, false) path)

(* [draw_series rows xs path] paints the series [rows] along [path], given in
   normalised positions, a row at [xs.(i)] being dropped where that is [nan]. A
   fill fills the region [path] closes within the domain, and a stroke strokes
   [path] within it. *)
let draw_series rows xs path =
  let th = Mark.theme rows in
  let o = style rows xs Role.opacity ~default:1. Float.equal in
  let width = style rows xs Role.width ~default:(em rows line_em) Float.equal in
  let stroke =
    Option.map
      (fun _ ->
        style rows xs Role.stroke ~default:Color.transparent Color.equal)
      (Mark.get rows Role.stroke)
  in
  match Mark.get rows Role.fill with
  | Some _ ->
      let fill =
        style rows xs Role.fill ~default:Color.transparent Color.equal
      in
      Picture.group
        [
          Picture.fill ~rule:`Even_odd (fade o fill)
            (Mark.project rows (closed path));
          (match stroke with
          | Some c ->
              Picture.stroke (Stroke.v width) (fade o c)
                (Mark.project rows path)
          | None -> Picture.empty);
        ]
  | None ->
      let c = Option.value stroke ~default:(Theme.accent th) in
      Picture.stroke (Stroke.v width) (fade o c) (Mark.project rows path)

let draw_line rows =
  let curve =
    Option.value (first_value rows Role.curve) ~default:Curve.linear
  in
  let series s =
    let us, vs = Mark.positions s in
    draw_series s us (Curve.path curve us vs)
  in
  Picture.group (List.map series (Mark.series rows))

(* A segment across the swatch's box, or the box filled. *)
let swatch_line rows =
  let x0, x1 = Mark.extent rows `X in
  match Mark.get rows Role.fill with
  | Some _ ->
      let y0, y1 = Mark.extent rows `Y in
      let box = Box2.of_pts (P2.v x0.(0) y0.(0)) (P2.v x1.(0) y1.(0)) in
      draw_series rows x0 (Path.rect box)
  | None ->
      draw_series rows x0 (Path.polyline [| x0.(0); x1.(0) |] [| 0.5; 0.5 |])

let line ?x ?stroke ?fill ?width ?opacity ?(curve = Curve.linear) ?fx ?fy ~y ()
    =
  let x =
    match x with Some x -> on Role.x x | None -> on Role.x (index (-1))
  in
  Mark
    (make "line" ~reduce:M4 ~swatch:swatch_line
       ([
          Some x;
          Some (on Role.y y);
          opt Role.stroke stroke;
          opt Role.fill fill;
          opt Role.width width;
          opt Role.opacity opacity;
          Some (on Role.curve (const curve));
        ]
       @ facets fx fy)
       draw_line)

(* Rects *)

let draw_rect rows =
  let n = Mark.length rows in
  let xe = Mark.extent rows `X and ye = Mark.extent rows `Y in
  let fills = fills rows ~default:(Theme.accent (Mark.theme rows)) in
  let strokes = strokes rows in
  let pen = Stroke.v (em rows outline_em) in
  let cell i =
    if not (finite (fst xe).(i) && finite (fst ye).(i)) then Picture.empty
    else
      let path = box_path rows xe ye i in
      Picture.group
        [
          (match fills with
          | Some cs -> Picture.fill cs.(i) path
          | None -> Picture.empty);
          (match strokes with
          | Some cs -> Picture.stroke pen cs.(i) path
          | None -> Picture.empty);
        ]
  in
  Picture.group (List.init n cell)

let rect ?x ?x2 ?y ?y2 ?fill ?stroke ?opacity ?fx ?fy () =
  (* A band position across a continuous one is a bar. *)
  let bars other = if continuous other then Some bar_band else None in
  Mark
    (make "rect" ~reduce:Cells
       ([
          position Role.x ?band:(bars y) ~alone:(Option.is_none x2) x;
          opt Role.x2 x2;
          position Role.y ?band:(bars x) ~alone:(Option.is_none y2) y;
          opt Role.y2 y2;
          opt Role.fill fill;
          opt Role.stroke stroke;
          opt Role.opacity opacity;
        ]
       @ facets fx fy)
       draw_rect)

(* Rules *)

(* [segments rows] is each row's segment, in normalised positions, [nan] where
   the row is dropped. *)
let segments rows =
  let get r = Mark.get rows r in
  match (get Role.x, get Role.x2, get Role.y, get Role.y2) with
  | Some xs, None, _, _ ->
      let y0, y1 = Mark.extent rows `Y in
      (xs, y0, xs, y1)
  | _, _, Some ys, None ->
      let x0, x1 = Mark.extent rows `X in
      (x0, ys, x1, ys)
  | Some _, Some x2, Some _, Some y2 ->
      let x, y = Mark.positions rows in
      (x, y, x2, y2)
  | _ ->
      (* The swatch of a rule binds no position: a segment across its box. *)
      let x0, x1 = Mark.extent rows `X in
      ( x0,
        Array.make (Mark.length rows) 0.5,
        x1,
        Array.make (Mark.length rows) 0.5 )

let draw_rule rows =
  let n = Mark.length rows and th = Mark.theme rows in
  let ax, ay, bx, by = segments rows in
  let colours = faded rows (or_const rows Role.stroke (Theme.ink th)) in
  let widths = or_const rows Role.width (em rows line_em) in
  let segment i =
    let path =
      Mark.project rows
        (Path.polyline [| ax.(i); bx.(i) |] [| ay.(i); by.(i) |])
    in
    if Option.is_none (Path.bounds path) then Picture.empty
    else Picture.stroke (Stroke.v ~cap:`Butt widths.(i)) colours.(i) path
  in
  Picture.group (List.init n segment)

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
       @ facets fx fy)
       draw_rule)

(* Texts *)

(* [draw_texts rows texts] sets [texts.(i)] at the point of each row. *)
let draw_texts rows texts =
  let xs, ys = Mark.points rows in
  let dx = Option.value (first_value rows Role.dx) ~default:0.
  and dy = Option.value (first_value rows Role.dy) ~default:0. in
  let fills =
    faded rows (or_const rows Role.fill (Theme.ink (Mark.theme rows)))
  in
  let label i =
    if not (finite xs.(i)) then Picture.empty
    else Mark.text rows fills.(i) (P2.v (xs.(i) +. dx) (ys.(i) -. dy)) texts.(i)
  in
  Picture.group (List.init (Mark.length rows) label)

let draw_text rows = draw_texts rows (or_const rows Role.text (Text.v ""))

(* A swatch shows the colour of its entry on a letter. *)
let swatch_text rows = draw_texts rows [| Text.v "a" |]

let text ?fill ?opacity ?(dx = 0.) ?(dy = 0.) ?fx ?fy ~x ~y ~text () =
  Mark
    (make "text" ~swatch:swatch_text
       ([
          Some (on Role.x x);
          Some (on Role.y y);
          Some (on Role.text text);
          opt Role.fill fill;
          opt Role.opacity opacity;
          Some (on Role.dx (const dx));
          Some (on Role.dy (const dy));
        ]
       @ facets fx fy)
       draw_text)

(* Images *)

let is_pixel : type a b. (a, b) Nx.dtype -> bool = function
  | Nx.UInt8 | Nx.Float16 | Nx.Float32 | Nx.Float64 | Nx.BFloat16
  | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      true
  | _ -> false

(* [datum px lead k] is the image of the datum [k] of the leading axes [lead] of
   [px]. *)
let datum px lead k =
  let index = Array.make (Array.length lead) 0 and r = ref k in
  for a = Array.length lead - 1 downto 0 do
    index.(a) <- !r mod lead.(a);
    r := !r / lead.(a)
  done;
  Nx.slice (Array.to_list (Array.map (fun i -> Nx.I i) index)) px

let unit_square = Box2.v 0. 0. 1. 1.

(* An image is placed by its positions, which a zoom can take beyond the domain,
   and then clipped to the domain: it has no ink beyond its box. *)
let draw_image rows =
  match first_value rows Role.pixels with
  | None -> Picture.empty
  | Some (Nx.P px) ->
      let at = Coord.point (Mark.projection rows) in
      let get role = Option.get (Mark.get rows role) in
      let x0 = get Role.x and x1 = get Role.x2 in
      let y0 = get Role.y and y1 = get Role.y2 in
      let index = Mark.index rows and lead = Mark.shape rows in
      let within u = 0. <= u && u <= 1. in
      let image i =
        if not (finite x0.(i) && finite y0.(i)) then Picture.empty
        else
          let box = Box2.of_pts (at x0.(i) y0.(i)) (at x1.(i) y1.(i)) in
          let px = Pixels.rgba (datum px lead index.(i)) in
          let shape = Nx.shape px in
          let picture =
            match
              Pixels.plan ~density:rows.Rows.density box ~rows:shape.(0)
                ~cols:shape.(1)
            with
            | None -> Picture.image box px
            | Some plan -> Picture.image plan.window (Pixels.gather plan px)
          in
          if List.for_all within [ x0.(i); x1.(i); y0.(i); y1.(i) ] then picture
          else Picture.clip (Mark.project rows (Path.rect unit_square)) picture
      in
      Picture.group (List.init (Mark.length rows) image)

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
       @ facets fx fy)
       draw_image)

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

(* Contours *)

let strictly_monotone a =
  let n = Array.length a in
  let rec go i s =
    i >= n - 1
    || (finite a.(i + 1) && Float.compare a.(i + 1) a.(i) = s && go (i + 1) s)
  in
  n < 2
  || Array.for_all finite a
     &&
     let s = Float.compare a.(1) a.(0) in
     s <> 0 && go 0 s

let draw_contour rows =
  let shape = Mark.shape rows in
  let rank = Array.length shape in
  let n = shape.(rank - 2) and m = shape.(rank - 1) in
  let norm role = Option.get (Mark.normalized rows role) in
  let us = norm Role.fill and xs = norm Role.x and ys = norm Role.y in
  let colour = Option.get (Mark.range rows Role.fill) in
  let o = Option.value (first_value rows Role.opacity) ~default:1. in
  let levels =
    0. :: 1. :: Array.to_list (Option.get (Mark.ticks rows Role.fill))
    |> List.sort_uniq Float.compare
    |> Array.of_list
  in
  let field k =
    let first = k * n * m in
    let cols = Array.init m (fun j -> xs.(first + j))
    and rows_y = Array.init n (fun i -> ys.(first + (i * m))) in
    if not (strictly_monotone cols && strictly_monotone rows_y) then begin
      Mark.warn rows "a field whose positions are not strictly monotone";
      Picture.empty
    end
    else
      let z = Nx.create Nx.float64 [| n; m |] (Array.sub us first (n * m)) in
      let f = Field2.v ~xs:cols ~ys:rows_y z in
      let band l =
        let lo = levels.(l) and hi = levels.(l + 1) in
        let path = Pgon2.to_path (Field2.isoband ~lo ~hi f) in
        Picture.fill
          (fade o (colour ((lo +. hi) /. 2.)))
          (Mark.project rows path)
      in
      Picture.group (List.init (Array.length levels - 1) band)
  in
  Picture.group (List.init (Mark.length rows / (n * m)) field)

(* [grid role ch] binds [ch], a position of a sampled field, to [role]: the
   field's extent is its grid, so it implies [nice] off. *)
let grid : type d r. (d, r) Role.t -> (d, r) Channel.t -> binding =
 fun role ch ->
  match data ch with
  | Some { lift; _ } -> (
      match lift_kind lift with
      | Scale.Quantitative -> on ~imply:(Scale.linear ~nice:false ()) role ch
      | Scale.Temporal -> on ~imply:(Scale.time ~nice:false ()) role ch
      | Scale.Categorical -> on role ch)
  | None -> on role ch

let contour ?x ?y ?opacity ?fx ?fy ~fill () =
  let x =
    match x with Some x -> grid Role.x x | None -> grid Role.x (index (-1))
  in
  let y =
    match y with Some y -> grid Role.y y | None -> grid Role.y (index (-2))
  in
  if Option.is_none (data fill) then err "contour" "fill is a constant";
  let fx = opt Role.fx fx and fy = opt Role.fy fy in
  let m =
    make "contour"
      [
        Some x;
        Some y;
        Some (on Role.fill fill);
        opt Role.opacity opacity;
        fx;
        fy;
      ]
      draw_contour
  in
  let shape = m.shape in
  let rank = Array.length shape in
  if rank < 2 then
    err "contour" "the shape %a has fewer than two axes" pp_shape shape;
  if varies shape x (rank - 2) then
    err "contour" "x can vary along the rows of the grid";
  if varies shape y (rank - 1) then
    err "contour" "y can vary along the columns of the grid";
  List.iter
    (fun (name, b) ->
      match b with
      | Some b when varies shape b (rank - 2) || varies shape b (rank - 1) ->
          err "contour" "%s can vary along the grid" name
      | _ -> ())
    [ ("fx", fx); ("fy", fy) ];
  Mark m
