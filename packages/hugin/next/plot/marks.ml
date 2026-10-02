(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The built-in marks name nothing of the library but [Api], the public names
   they use, so that a user can write each of them. The test suite compiles a
   copy of this file against the public interface. *)

open Api
module Field2 = Hugin_next_gg_kit.Field2
module Pgon2 = Hugin_next_gg_kit.Pgon2

let err fn fmt =
  Format.kasprintf (fun s -> invalid_arg ("Hugin_next." ^ fn ^ ": " ^ s)) fmt

let pp_shape ppf s =
  Format.fprintf ppf "[%a]"
    (Format.pp_print_seq
       ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
       Format.pp_print_int)
    (Array.to_seq s)

let on = Mark.bind
let opt role = Option.map (on role)

(* Parameters, made once so that a mark rebuilt binds the same roles. *)

let curve_param = Role.param ~name:"curve" ~equal:Curve.equal
let dx_param = Role.param ~name:"dx" ~equal:Float.equal
let dy_param = Role.param ~name:"dy" ~equal:Float.equal

let same_tensor (Nx.P a) (Nx.P b) =
  match Nx_dtype.equal_witness (Nx.dtype a) (Nx.dtype b) with
  | Some Type.Equal -> a == b
  | None -> false

let pixels_param = Role.param ~name:"pixels" ~equal:same_tensor

(* Derived lengths, in em *)

let line_em = 0.15
let outline_em = 0.08
let dot_em = 0.6 (* The diameter of a dot's circle. *)
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

(* [pen ~cap width dash] strokes at [width] with [dash], whose lengths are in
   multiples of the width. *)
let pen ?cap width dash =
  let scaled l = l *. width in
  let dash = if width > 0. then List.map scaled (Dash.lengths dash) else [] in
  Stroke.v ?cap ~dash width

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
    ?band:string Scale.t -> (d, r) Role.t -> (d, r) channel -> Mark.binding =
 fun ?band role ch ->
  match kind ch with
  | Some Scale.Quantitative -> on ~imply:(Scale.linear ~zero:true ()) role ch
  | Some Scale.Categorical -> on ?imply:band role ch
  | Some Scale.Temporal | None -> on role ch

let position ?band role ~alone = function
  | None -> None
  | Some ch -> Some (if alone then length ?band role ch else on role ch)

(* [continuous ch] is [true] iff [ch] holds data read by a continuous scale. *)
let continuous : type d r. (d, r) channel option -> bool = function
  | None -> false
  | Some ch -> (
      match kind ch with
      | Some (Scale.Quantitative | Scale.Temporal) -> true
      | Some Scale.Categorical | None -> false)

let facets fx fy = [ opt Role.fx fx; opt Role.fy fy ]

let make name ?reduce ?coord ?shape ?swatch l draw =
  Mark.v ~name ?reduce ?coord ?shape ?swatch (List.filter_map Fun.id l) draw

(* Dots *)

let draw_dot rows =
  let th = Mark.theme rows in
  let xs, ys = Mark.points rows in
  let fills = fills rows ~default:(Theme.accent th)
  and strokes = strokes rows in
  let paint = match fills with Some _ -> `Fill | None -> `Stroke in
  (* Dots of one size draw a glyph of that size, so that the stamp's instances
     are copies of it, which renderers draw fastest; dots of several sizes scale
     a glyph of unit area. *)
  let area, scales =
    match Mark.get rows Role.size with
    | Some areas
      when Array.exists (fun a -> not (Float.equal a areas.(0))) areas ->
        (1., Some (Array.map Float.sqrt areas))
    | Some areas when Array.length areas > 0 -> (areas.(0), None)
    | Some _ | None -> (dot_area rows, None)
  in
  let pen = Stroke.v (em rows outline_em) in
  let glyph symbol =
    let path = Symbol.path paint area symbol in
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
  let stamp ?fills ?strokes ?scales xs ys symbol =
    Picture.stamp ?fills ?strokes ?scales xs ys (glyph symbol)
  in
  match Mark.get rows Role.symbol with
  | None -> stamp ?fills ?strokes ?scales xs ys Symbol.circle
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
                  ?strokes:(Option.map sub strokes)
                  ?scales:(Option.map sub scales) (sub xs) (sub ys) s))
           !groups)

let dot ?fill ?stroke ?opacity ?size ?symbol ?fx ?fy ~x ~y () =
  make "dot" ~reduce:Mark.raster
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
    draw_dot

(* Lines *)

(* [style rows xs (name, role) ~default equal] is the value of [role] of the
   first row of the series [rows] that is not dropped, a row at [xs.(i)] being
   dropped where that is [nan], with a warning naming [name] if it varies along
   the series. *)
let style rows xs (name, role) ~default equal =
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
        Mark.warn rows (Printf.sprintf "the %s varies along a series" name);
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
  let o = style rows xs ("opacity", Role.opacity) ~default:1. Float.equal in
  let width =
    style rows xs ("width", Role.width) ~default:(em rows line_em) Float.equal
  in
  let dash = style rows xs ("dash", Role.dash) ~default:Dash.solid Dash.equal in
  let stroke =
    Option.map
      (fun _ ->
        style rows xs ("stroke", Role.stroke) ~default:Color.transparent
          Color.equal)
      (Mark.get rows Role.stroke)
  in
  match Mark.get rows Role.fill with
  | Some _ ->
      let fill =
        style rows xs ("fill", Role.fill) ~default:Color.transparent Color.equal
      in
      Picture.group
        [
          Picture.fill ~rule:`Even_odd (fade o fill)
            (Mark.project rows (closed path));
          (match stroke with
          | Some c ->
              Picture.stroke (pen width dash) (fade o c)
                (Mark.project rows path)
          | None -> Picture.empty);
        ]
  | None ->
      let c = Option.value stroke ~default:(Theme.accent th) in
      Picture.stroke (pen width dash) (fade o c) (Mark.project rows path)

let draw_line rows =
  let curve =
    Option.value (first_value rows curve_param) ~default:Curve.linear
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

(* The curves whose piece between two points depends on those two points alone
   and stays within their bounding box, which M4 draws as the whole curve
   does. *)
let boxed = Curve.[ linear; step_after; step_before; step_mid ]

let line ?x ?stroke ?fill ?width ?dash ?opacity ?(curve = Curve.linear) ?fx ?fy
    ~y () =
  let x =
    match x with Some x -> on Role.x x | None -> on Role.x (index (-1))
  in
  let reduce =
    match fill with
    | None when List.exists (Curve.equal curve) boxed -> Some Mark.m4
    | _ -> None
  in
  make "line" ?reduce ~swatch:swatch_line
    ([
       Some x;
       Some (on Role.y y);
       opt Role.stroke stroke;
       opt Role.fill fill;
       opt Role.width width;
       opt Role.dash dash;
       opt Role.opacity opacity;
       Some (on curve_param (const curve));
     ]
    @ facets fx fy)
    draw_line

(* Areas *)

(* [paint_area rows xs path] fills [path], given in normalised positions, with
   the fill of the series [rows], a row at [xs.(i)] being dropped where that is
   [nan]. *)
let paint_area rows xs path =
  let accent = Theme.accent (Mark.theme rows) in
  let o = style rows xs ("opacity", Role.opacity) ~default:1. Float.equal in
  let fill = style rows xs ("fill", Role.fill) ~default:accent Color.equal in
  Picture.fill (fade o fill) (Mark.project rows path)

(* Each series is the region between its curve and its baseline: [y2], or else
   the start of [y]'s length. *)
let draw_area rows =
  let curve =
    Option.value (first_value rows curve_param) ~default:Curve.linear
  in
  let series s =
    let us, vs = Mark.positions s in
    let base =
      match Mark.get s Role.y2 with
      | Some y2 -> y2
      | None -> fst (Mark.extent s `Y)
    in
    paint_area s us (Curve.area curve ~x0:us ~y0:base us vs)
  in
  Picture.group (List.map series (Mark.series rows))

(* The swatch's box filled. *)
let swatch_area rows =
  let x0, x1 = Mark.extent rows `X and y0, y1 = Mark.extent rows `Y in
  let box = Box2.of_pts (P2.v x0.(0) y0.(0)) (P2.v x1.(0) y1.(0)) in
  paint_area rows x0 (Path.rect box)

let area ?x ?y2 ?fill ?opacity ?(curve = Curve.linear) ?fx ?fy ~y () =
  let x =
    match x with Some x -> on Role.x x | None -> on Role.x (index (-1))
  in
  make "area" ~swatch:swatch_area
    ([
       Some x;
       position Role.y ~alone:(Option.is_none y2) (Some y);
       opt Role.y2 y2;
       opt Role.fill fill;
       opt Role.opacity opacity;
       Some (on curve_param (const curve));
     ]
    @ facets fx fy)
    draw_area

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
  make "rect" ~reduce:Mark.cells
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
    draw_rect

(* Frames *)

let unit_square = Box2.v 0. 0. 1. 1.

let draw_frame rows =
  let ink = Theme.ink (Mark.theme rows) in
  let colours = faded rows (or_const rows Role.stroke ink) in
  let pen = Stroke.v (em rows outline_em) in
  let path = Mark.project rows (Path.rect unit_square) in
  let edge i = Picture.stroke pen colours.(i) path in
  Picture.group (List.init (Mark.length rows) edge)

let frame ?stroke ?opacity ?fx ?fy () =
  make "frame"
    ([ opt Role.stroke stroke; opt Role.opacity opacity ] @ facets fx fy)
    draw_frame

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
  let dashes = or_const rows Role.dash Dash.solid in
  let segment i =
    let path =
      Mark.project rows
        (Path.polyline [| ax.(i); bx.(i) |] [| ay.(i); by.(i) |])
    in
    if Option.is_none (Path.bounds path) then Picture.empty
    else Picture.stroke (pen ~cap:`Butt widths.(i) dashes.(i)) colours.(i) path
  in
  Picture.group (List.init n segment)

let rule ?x ?x2 ?y ?y2 ?stroke ?width ?dash ?opacity ?fx ?fy () =
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
  make "rule"
    (positions
    @ [
        opt Role.stroke stroke;
        opt Role.width width;
        opt Role.dash dash;
        opt Role.opacity opacity;
      ]
    @ facets fx fy)
    draw_rule

(* Lines of equations *)

let slope_role = Role.value ~name:"slope"
let intercept_role = Role.value ~name:"intercept"

(* [within m c (a, b) (ya, yb)] is the interval of the x in \[[a];[b]\] whose [y
   = m x + c] lies in \[[ya];[yb]\], if any: y is monotone in x, so these x form
   one interval. *)
let within m c (a, b) (ya, yb) =
  let lo, hi =
    if m = 0. then
      if ya <= c && c <= yb then (a, b) else (Float.infinity, Float.neg_infinity)
    else
      let x0 = (ya -. c) /. m and x1 = (yb -. c) /. m in
      (Float.max a (Float.min x0 x1), Float.min b (Float.max x0 x1))
  in
  if lo <= hi then Some (lo, hi) else None

(* [equation_path rows sx sy m c] is the line y = m x + c across the x domain of
   [sx], cut in data to the y domain of [sy], in normalised positions: a segment
   between its ends if [sx] and [sy] are linear, and otherwise a curve through
   points one point of the page apart along x. Cutting before normalising keeps
   a clamped scale from clamping it. *)
let equation_path rows sx sy m c =
  let (Scale.Floats (a, b)) = Scale.domain sx
  and (Scale.Floats (ya, yb)) = Scale.domain sy in
  match within m c (a, b) (ya, yb) with
  | None -> Path.empty
  | Some (lo, hi) -> (
      let nx = Scale.normalize sx and ny = Scale.normalize sy in
      (* Inside the interval, y leaves the domain by rounding alone. *)
      let y x = ny (Float.min yb (Float.max ya ((m *. x) +. c))) in
      match (Scale.transform sx, Scale.transform sy) with
      | Scale.Linear, Scale.Linear ->
          Path.polyline [| nx lo; nx hi |] [| y lo; y hi |]
      | _ ->
          let u0 = nx lo and u1 = nx hi in
          let at u = Coord.point (Mark.projection rows) u 0. in
          let w = Float.abs (P2.x (at u1) -. P2.x (at u0)) in
          let n = Int.max 1 (Float.to_int (Float.ceil w)) in
          let x j =
            if j = 0 then lo
            else if j = n then hi
            else
              let u = u0 +. (Float.of_int j /. Float.of_int n *. (u1 -. u0)) in
              Option.value (Scale.invert sx u) ~default:Float.nan
          in
          let xs = Array.init (n + 1) x in
          Curve.path Curve.linear (Array.map nx xs) (Array.map y xs))

let draw_abline rows =
  match
    ( Mark.scale rows `X Scale.Quantitative,
      Mark.scale rows `Y Scale.Quantitative )
  with
  | Some sx, Some sy ->
      let th = Mark.theme rows in
      let slopes = or_const rows slope_role Float.nan
      and intercepts = or_const rows intercept_role Float.nan in
      let colours = faded rows (or_const rows Role.stroke (Theme.ink th)) in
      let widths = or_const rows Role.width (em rows line_em) in
      let dashes = or_const rows Role.dash Dash.solid in
      let line i =
        let m = slopes.(i) and c = intercepts.(i) in
        if not (finite m && finite c) then Picture.empty
        else
          let path = Mark.project rows (equation_path rows sx sy m c) in
          if Option.is_none (Path.bounds path) then Picture.empty
          else
            Picture.stroke
              (pen ~cap:`Butt widths.(i) dashes.(i))
              colours.(i) path
      in
      Picture.group (List.init (Mark.length rows) line)
  | _ ->
      Mark.warn rows "an abline needs quantitative x and y scales";
      Picture.empty

let abline ?stroke ?width ?dash ?opacity ?fx ?fy ~slope ~intercept () =
  make "abline" ~swatch:draw_rule
    ([
       Some (on slope_role slope);
       Some (on intercept_role intercept);
       opt Role.stroke stroke;
       opt Role.width width;
       opt Role.dash dash;
       opt Role.opacity opacity;
     ]
    @ facets fx fy)
    draw_abline

(* Texts *)

(* [draw_texts rows texts] sets [texts.(i)] at the point of each row. *)
let draw_texts rows texts =
  let xs, ys = Mark.points rows in
  let dx = Option.value (first_value rows dx_param) ~default:0.
  and dy = Option.value (first_value rows dy_param) ~default:0. in
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
  make "text" ~swatch:swatch_text
    ([
       Some (on Role.x x);
       Some (on Role.y y);
       Some (on Role.text text);
       opt Role.fill fill;
       opt Role.opacity opacity;
       Some (on dx_param (const dx));
       Some (on dy_param (const dy));
     ]
    @ facets fx fy)
    draw_text

(* Images *)

let is_pixel : type a b. (a, b) Nx.dtype -> bool = function
  | Nx.UInt8 | Nx.Float16 | Nx.Float32 | Nx.Float64 | Nx.BFloat16
  | Nx.Float8_e4m3 | Nx.Float8_e5m2 ->
      true
  | _ -> false

(* [rgba px] is the image [px], of shape [[|h; w|]] or [[|h; w; c|]], as
   [Picture.image] takes it: [uint8] values as they are, and floating-point
   values clamped into [0;1] and scaled to [0;255], a pixel with a NaN component
   transparent. *)
let rgba : type a b. (a, b) Nx.t -> Nx.uint8_t =
 fun px ->
  let px =
    match Nx.shape px with [| h; w |] -> Nx.reshape [| h; w; 1 |] px | _ -> px
  in
  match Nx.dtype px with
  | Nx.UInt8 -> px
  | _ ->
      let f = Nx.cast Nx.float32 px in
      let c = (Nx.shape f).(2) in
      let hole = Nx.any ~axes:[ 2 ] ~keepdims:true (Nx.isnan f) in
      let v = Nx.round (Nx.mul_s (Nx.clamp ~min:0. ~max:1. f) 255.) in
      let v =
        Nx.where (Nx.broadcast_to (Nx.shape v) hole) (Nx.zeros_like v) v
      in
      let channel k = Nx.slice [ A; A; R (k, k + 1) ] v in
      let rgb, alpha =
        match c with
        | 1 ->
            ([ v; v; v ], Nx.where hole (Nx.zeros_like v) (Nx.full_like v 255.))
        | 3 ->
            ( [ v ],
              Nx.where hole
                (Nx.zeros_like (channel 0))
                (Nx.full_like (channel 0) 255.) )
        | _ -> ([ Nx.slice [ A; A; R (0, 3) ] v ], channel 3)
      in
      Nx.cast Nx.uint8 (Nx.concatenate ~axis:2 (rgb @ [ alpha ]))

(* [lead px] is the leading axes of the image tensor [px], its datum axes. *)
let lead px =
  let shape = Nx.shape px in
  let rank = Array.length shape in
  if rank = 2 then [||] else Array.sub shape 0 (rank - 3)

(* [datum px lead k] is the image of the datum [k] of the leading axes [lead] of
   [px]. *)
let datum px lead k =
  let index = Array.make (Array.length lead) 0 and r = ref k in
  for a = Array.length lead - 1 downto 0 do
    index.(a) <- !r mod lead.(a);
    r := !r / lead.(a)
  done;
  Nx.slice (Array.to_list (Array.map (fun i -> Nx.I i) index)) px

(* [inside (a, b) i] is [true] iff the extent of row [i] lies in the domain. *)
let inside (a, b) i = Float.min a.(i) b.(i) >= 0. && Float.max a.(i) b.(i) <= 1.

(* An image is placed by its extents, which a zoom can take beyond the domain.
   Pixels have no geometry to crop, so an image reaching beyond the domain is
   clipped to it, the one clip of ink. *)
let draw_image rows =
  match first_value rows pixels_param with
  | None -> Picture.empty
  | Some (Nx.P px) ->
      let at = Coord.point (Mark.projection rows) in
      let ((x0, x1) as xe) = Mark.extent rows `X
      and ((y0, y1) as ye) = Mark.extent rows `Y in
      let index = Mark.index rows and lead = lead px in
      let image i =
        if not (finite x0.(i) && finite y0.(i)) then Picture.empty
        else
          let box = Box2.of_pts (at x0.(i) y0.(i)) (at x1.(i) y1.(i)) in
          let picture = Picture.image box (rgba (datum px lead index.(i))) in
          if inside xe i && inside ye i then picture
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
  let at v = floats [| v |] in
  make "image"
    ~coord:(Coord.cartesian ~aspect:1. ())
    ~shape:lead
    ([
       Some (on ~imply:fixed ~guide:false Role.x (at 0.));
       Some (on Role.x2 (at (float w)));
       Some
         (on
            ~imply:(Scale.linear ~nice:false ~reverse:true ())
            ~guide:false Role.y (at 0.));
       Some (on Role.y2 (at (float h)));
       Some (on pixels_param (const (Nx.P px)));
     ]
    @ facets fx fy)
    draw_image

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
   field's extent is its grid, so it implies [nice] off on quantities. *)
let grid : type d r. (d, r) Role.t -> (d, r) channel -> Mark.binding =
 fun role ch ->
  match kind ch with
  | Some Scale.Quantitative -> on ~imply:(Scale.linear ~nice:false ()) role ch
  | Some (Scale.Categorical | Scale.Temporal) | None -> on role ch

let contour ?x ?y ?opacity ?fx ?fy ~fill () =
  if Option.is_none (kind fill) then err "contour" "fill is a constant";
  let on_x =
    match x with Some x -> grid Role.x x | None -> grid Role.x (index (-1))
  and on_y =
    match y with Some y -> grid Role.y y | None -> grid Role.y (index (-2))
  in
  let bindings =
    List.filter_map Fun.id
      [
        Some on_x;
        Some on_y;
        Some (on ~imply:(Scale.linear ~stepped:true ()) Role.fill fill);
        opt Role.opacity opacity;
        opt Role.fx fx;
        opt Role.fy fy;
      ]
  in
  let shape = Mark.broadcast bindings in
  let rank = Array.length shape in
  if rank < 2 then
    err "contour" "the shape %a has fewer than two axes" pp_shape shape;
  (* The defaults, [index (-1)] and [index (-2)], vary along their own axes. *)
  let can_vary c a =
    match c with Some c -> varies shape c a | None -> false
  in
  if can_vary x (-2) then err "contour" "x can vary along the rows of the grid";
  if can_vary y (-1) then
    err "contour" "y can vary along the columns of the grid";
  if can_vary fx (-2) || can_vary fx (-1) then
    err "contour" "fx can vary along the grid";
  if can_vary fy (-2) || can_vary fy (-1) then
    err "contour" "fy can vary along the grid";
  Mark.v ~name:"contour" bindings draw_contour
