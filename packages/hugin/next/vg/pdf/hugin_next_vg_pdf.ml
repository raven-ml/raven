(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin_next_gg
open Hugin_next_font
open Hugin_next_vg

(* Numbers *)

let add_num b d v = Vector.add_fixed b d v

(* [rounded d v] is [v] as written with [d] decimals. *)
let rounded d v =
  let p = 10. ** float d in
  Float.round (v *. p) /. p

(* [is_zero d v] is [true] iff [v] is written [0] with [d] decimals. *)
let is_zero d v = rounded d v = 0.

(* Objects *)

type font = {
  font : Font.t;
  name : string;  (** Its name in the resources. *)
  number : int;  (** Its object, written once the page is. *)
  text : (int, string) Hashtbl.t;  (** The text each glyph renders. *)
  used : (int, unit) Hashtbl.t;
}

type doc = {
  bodies : string Dynarray.t;  (** Indexed by object number minus one. *)
  resources : int;  (** The resource dictionary every content stream shares. *)
  mutable fonts : font list;  (** In reverse order of first use. *)
  mutable xobjects : (string * int) list;  (** In reverse order of first use. *)
  forms : (string, string) Hashtbl.t;
      (** Form names by dictionary and content. *)
  mutable states : (string * string) list;
      (** Graphics states: name and dictionary, in reverse order. *)
  state_names : (string, string) Hashtbl.t;  (** Graphics state names by key. *)
  mutable images : (Nx.uint8_t * (int * int * int * int) * string) list;
}

let add doc body =
  Dynarray.add_last doc.bodies body;
  Dynarray.length doc.bodies

let set doc n body = Dynarray.set doc.bodies (n - 1) body

let stream ?(dict = "") data =
  let data =
    Bytesrw.Bytes.Writer.filter_string
      [ Compress_deflate.Zlib.compress_writes () ]
      data
  in
  Printf.sprintf
    "<< /Length %d /Filter /FlateDecode%s >>\nstream\n%s\nendstream"
    (String.length data) dict data

let xobject doc number =
  let name = Printf.sprintf "X%d" (List.length doc.xobjects + 1) in
  doc.xobjects <- (name, number) :: doc.xobjects;
  name

(* [state doc key dict] is the name of the graphics state [dict], named by
   [key]. *)
let state doc key dict =
  match Hashtbl.find_opt doc.state_names key with
  | Some name -> name
  | None ->
      let name = Printf.sprintf "G%d" (Hashtbl.length doc.state_names + 1) in
      Hashtbl.add doc.state_names key name;
      doc.states <- (name, dict) :: doc.states;
      name

let alpha_key a =
  let b = Buffer.create 8 in
  add_num b 3 a;
  Buffer.contents b

(* [add_alpha doc b op a] sets the alpha constant [op], [ca] or [CA], to [a]. *)
let add_alpha doc b op a =
  let a = alpha_key a in
  Printf.bprintf b "/%s gs\n"
    (state doc (op ^ a) (Printf.sprintf "<< /Type /ExtGState /%s %s >>" op a))

(* Writing in a frame *)

(* What paints a leaf of a kind: its own colour, the one an enclosing instance
   sets, which it then inherits, or a colour an instance written in full
   gives. *)
type paint = Own | Inherit | Fixed of Color.t

type ctx = {
  m : Affine.t;  (** From the picture's coordinates to the frame's. *)
  cut : Box2.t;  (** Where geometry is cut, as {!Vector} says. *)
  mag : float;  (** How much the page magnifies the frame's numbers. *)
  d : int;  (** Decimals of the frame's numbers. *)
  fills : paint;
  strokes : paint;
  fill_set : bool;  (** An enclosing instance sets the fill colour and [ca]. *)
  stroke_set : bool;
  pen : float;  (** What the lengths of pens are multiplied by. *)
  pen_set : bool;  (** An enclosing instance sets the width and dashes. *)
}

(* [add_color doc b ~stroking ~set c] sets the colour and alpha of fills, or of
   strokes if [stroking]. An alpha of [1.] is set only if an enclosing instance
   may have set another. *)
let add_color doc b ~stroking ~set c =
  List.iter
    (fun v ->
      add_num b 3 v;
      Buffer.add_char b ' ')
    [ Color.r c; Color.g c; Color.b c ];
  Buffer.add_string b (if stroking then "RG\n" else "rg\n");
  if Color.alpha c < 1. || set then
    add_alpha doc b (if stroking then "CA" else "ca") (Color.alpha c)

let add_paint doc b ~stroking ~set paint own =
  match paint with
  | Own -> add_color doc b ~stroking ~set own
  | Fixed c -> add_color doc b ~stroking ~set c
  | Inherit -> ()

(* [ops b d] is a sink writing path construction operators with [d] decimals. *)
let ops b d =
  let pt x y =
    add_num b d x;
    Buffer.add_char b ' ';
    add_num b d y;
    Buffer.add_char b ' '
  in
  {
    Vector.move =
      (fun x y ->
        pt x y;
        Buffer.add_string b "m\n");
    line =
      (fun x y ->
        pt x y;
        Buffer.add_string b "l\n");
    cubic =
      (fun x1 y1 x2 y2 x y ->
        pt x1 y1;
        pt x2 y2;
        pt x y;
        Buffer.add_string b "c\n");
    close = (fun () -> Buffer.add_string b "h\n");
  }

let area_ops ctx path =
  let b = Buffer.create 64 in
  Vector.area ctx.m ctx.cut path (ops b ctx.d);
  Buffer.contents b

(* Leaves *)

let fill doc ctx b rule color path =
  let p = area_ops ctx path in
  if p <> "" then begin
    Buffer.add_string b "q\n";
    add_paint doc b ~stroking:false ~set:ctx.fill_set ctx.fills color;
    Buffer.add_string b p;
    Buffer.add_string b (if rule = `Even_odd then "f*\nQ\n" else "f\nQ\n")
  end

(* [dash_decimals d] is the decimals of dash lengths and phases in a frame of
   [d] decimals: three more, since the errors of the lengths add up along a
   subpath, so that a thousand of them stay within the accuracy. *)
let dash_decimals d = Vector.finer 3 d

(* [is_dashed d pen] is [true] iff a length of the dashes of [pen] is written
   non-zero in a frame of [d] decimals: a pattern that is not is written
   solid. *)
let is_dashed d (pen : Vector.pen) =
  let d = dash_decimals d in
  List.exists (fun v -> not (is_zero d v)) pen.dash

let add_dash b d (pen : Vector.pen) ~offset =
  let d = dash_decimals d in
  Buffer.add_char b '[';
  List.iteri
    (fun i v ->
      if i > 0 then Buffer.add_char b ' ';
      add_num b d v)
    pen.dash;
  Buffer.add_string b "] ";
  let period = List.fold_left ( +. ) 0. pen.dash in
  let period =
    if List.length pen.dash mod 2 = 1 then 2. *. period else period
  in
  add_num b d (Float.rem (pen.offset +. offset) period);
  Buffer.add_string b " d\n"

let add_pen b d (pen : Vector.pen) =
  add_num b d pen.width;
  Buffer.add_string b " w\n";
  if is_dashed d pen then add_dash b d pen ~offset:0.

let stroke doc ctx b s color path =
  let lin = Vector.linear ctx.m in
  let pen = Vector.pen ctx.m ctx.pen s in
  (* A pen the frame stretches unevenly is stroked under [u], the frame's linear
     part scaled to stretch nothing more than the page does, as written, and its
     points under the inverse of [u]. *)
  let u =
    if Vector.is_similar lin then None
    else
      let u = Vector.unit lin in
      Some
        {
          u with
          xx = rounded 15 u.xx;
          yx = rounded 15 u.yx;
          xy = rounded 15 u.xy;
          yy = rounded 15 u.yy;
        }
  in
  let out = Option.map Affine.invert u in
  (* A width written as zero would be a viewer's thinnest line, and a [u]
     singular as written cannot carry the pen. *)
  let thin = (not ctx.pen_set) && is_zero ctx.d pen.width in
  match out with
  | Some None -> ()
  | _ when thin -> ()
  | _ ->
      let out = Option.join out in
      let reach = Vector.reach pen s in
      let cut = Vector.grown ctx.cut reach in
      let dashed = is_dashed ctx.d pen in
      (* The dashes are set before the first piece, and again before each piece
         that starts at another offset, which is cut from within a subpath and
         stroked on its own. *)
      let body = Buffer.create 64 and offset = ref Float.nan in
      let started = ref false in
      Vector.outline ?out ctx.m cut
        ~points:(Stroke.cap s <> `Square)
        ~dashed path
        ~piece:(fun d ->
          if dashed && (not ctx.pen_set) && not (Float.equal d !offset) then begin
            if !started then Buffer.add_string body "S\n";
            add_dash body ctx.d pen ~offset:d;
            offset := d
          end;
          started := true)
        (ops body ctx.d);
      if !started then begin
        Buffer.add_string b "q\n";
        add_paint doc b ~stroking:true ~set:ctx.stroke_set ctx.strokes color;
        if not ctx.pen_set then begin
          add_num b ctx.d pen.width;
          Buffer.add_string b " w\n"
        end;
        (match Stroke.cap s with
        | `Butt -> ()
        | `Round -> Buffer.add_string b "1 J\n"
        | `Square -> Buffer.add_string b "2 J\n");
        (match Stroke.join s with
        | `Miter ->
            if Stroke.miter_limit s <> 10. then begin
              add_num b 3 (Stroke.miter_limit s);
              Buffer.add_string b " M\n"
            end
        | `Round -> Buffer.add_string b "1 j\n"
        | `Bevel -> Buffer.add_string b "2 j\n");
        Option.iter
          (fun (u : Affine.t) ->
            List.iter
              (fun v ->
                add_num b 15 v;
                Buffer.add_char b ' ')
              [ u.xx; u.yx; u.xy; u.yy ];
            Buffer.add_string b "0 0 cm\n")
          u;
        Buffer.add_buffer b body;
        Buffer.add_string b "S\nQ\n"
      end

(* Text *)

let font_of doc f =
  match
    List.find_opt (fun u -> u.font == f || Font.equal u.font f) doc.fonts
  with
  | Some u -> u
  | None ->
      let u =
        {
          font = f;
          name = Printf.sprintf "F%d" (List.length doc.fonts + 1);
          number = add doc "";
          text = Hashtbl.create 64;
          used = Hashtbl.create 64;
        }
      in
      doc.fonts <- u :: doc.fonts;
      u

(* [width f g] is the width of [g] as written in the font's widths, in
   thousandths of an em. *)
let width f g = Float.round (Font.advance f g *. 1e6) /. 1e3

(* [mapped u run] is [true] iff the map of [u] gives each glyph of [run] the one
   character it renders. It enters in the map each glyph of [run] that renders
   one character and that the map does not hold yet. *)
let mapped u run =
  let text = Run.text run and n = Run.length run in
  let all = ref true in
  for i = 0 to n - 1 do
    let g = Run.glyph run i and start = Run.cluster run i in
    let stop =
      if i + 1 < n then Run.cluster run (i + 1) else String.length text
    in
    let one =
      g <> 0
      && start < String.length text
      && start + Uchar.utf_decode_length (String.get_utf_8_uchar text start)
         = stop
    in
    let maps =
      one
      &&
      let s = String.sub text start (stop - start) in
      match Hashtbl.find_opt u.text g with
      | Some s' -> String.equal s s'
      | None ->
          Hashtbl.add u.text g s;
          true
    in
    if not maps then all := false
  done;
  !all

(* [add_utf16 b s] adds the UTF-16BE code units of [s] in hexadecimal. *)
let add_utf16 b s =
  let rec loop i =
    if i < String.length s then begin
      let d = String.get_utf_8_uchar s i in
      let c = Uchar.to_int (Uchar.utf_decode_uchar d) in
      if c < 0x10000 then Printf.bprintf b "%04X" c
      else begin
        let v = c - 0x10000 in
        Printf.bprintf b "%04X%04X"
          (0xD800 lor (v lsr 10))
          (0xDC00 lor (v land 0x3FF))
      end;
      loop (i + Uchar.utf_decode_length d)
    end
  in
  loop 0

let glyphs doc ctx b color at run =
  let m = Affine.(ctx.m * translate (P2.x at) (P2.y at)) in
  let lin = Vector.linear m in
  let s = Vector.stretch lin in
  let f = Run.font run and n = Run.length run in
  (* How far, in ems, the outline of a glyph reaches from its origin in each
     direction. *)
  let e =
    let bx = Font.bounds f in
    Float.max
      (Float.max (Float.abs (Box2.minx bx)) (Float.abs (Box2.maxx bx)))
      (Float.max (Float.abs (Box2.miny bx)) (Float.abs (Box2.maxy bx)))
  in
  (* The size is written with the decimals that keep outlines within a tenth of
     the accuracy, and glyphs are placed by the size as written. *)
  let ds = Vector.finer 1 (Vector.decimals (ctx.mag *. e)) in
  let size = rounded ds (Run.size run *. s) in
  let origin i = P2.transform m (P2.v (Run.x run i) (Run.y run i)) in
  (* A glyph is shown if the font's box around its origin meets the cut. *)
  let ink = size *. e *. Float.sqrt 2. in
  let shown =
    Array.init n (fun i ->
        let o = origin i in
        Vector.overlaps ctx.cut
          (P2.x o -. ink)
          (P2.y o -. ink)
          (P2.x o +. ink)
          (P2.y o +. ink))
  in
  (* A size written as zero paints nothing a viewer can show. *)
  if size <> 0. && Array.exists Fun.id shown then begin
    let u = font_of doc f in
    let t = Vector.unit lin in
    (* TJ adjustments are in thousandths of the size: the decimals that keep
       them within a tenth of the accuracy. *)
    let dn =
      Int.max 0
        (Int.min 17
           (4
           + int_of_float (Float.ceil (Float.log10 (ctx.mag *. size /. 1000.)))
           ))
    in
    let mapped = mapped u run in
    let actual = (not mapped) || Array.exists not shown in
    Buffer.add_string b "q\n";
    if actual then begin
      Buffer.add_string b "/Span << /ActualText <FEFF";
      add_utf16 b (Run.text run);
      Buffer.add_string b "> >> BDC\n"
    end;
    add_paint doc b ~stroking:false ~set:ctx.fill_set ctx.fills color;
    Printf.bprintf b "BT\n/%s " u.name;
    add_num b ds size;
    Buffer.add_string b " Tf\n";
    let i = ref 0 in
    while !i < n do
      if not shown.(!i) then incr i
      else begin
        (* A line of glyphs on one baseline, placed by one matrix, its text
           space y-up so that glyphs stand upright in the frame. *)
        let first = !i in
        let x0 = Run.x run first and y = Run.y run first in
        let o = origin first in
        List.iter
          (fun v ->
            add_num b 15 v;
            Buffer.add_char b ' ')
          [ t.xx; t.yx; -.t.xy; -.t.yy ];
        add_num b ctx.d (P2.x o);
        Buffer.add_char b ' ';
        add_num b ctx.d (P2.y o);
        Buffer.add_string b " Tm\n[";
        let pen = ref 0. and continue = ref true in
        while !continue do
          let g = Run.glyph run !i in
          Hashtbl.replace u.used g ();
          Printf.bprintf b "<%04X>" g;
          pen := !pen +. (size *. width f g /. 1000.);
          incr i;
          if !i < n && shown.(!i) && Run.y run !i = y then begin
            let target = s *. (Run.x run !i -. x0) in
            let adj = rounded dn ((!pen -. target) *. 1000. /. size) in
            if adj <> 0. then begin
              Buffer.add_char b ' ';
              add_num b dn adj;
              Buffer.add_char b ' ';
              pen := !pen -. (adj *. size /. 1000.)
            end
          end
          else continue := false
        done;
        Buffer.add_string b "] TJ\n"
      end
    done;
    Buffer.add_string b "ET\n";
    if actual then Buffer.add_string b "EMC\n";
    Buffer.add_string b "Q\n"
  end

(* Images *)

(* [plane data ~pixels ~chans ~first ~count] is the [count] channels from
   [first] of each of the [pixels] pixels of [chans] channels of [data]. *)
let plane data ~pixels ~chans ~first ~count =
  let out = Bytes.create (pixels * count) in
  for p = 0 to pixels - 1 do
    for c = 0 to count - 1 do
      (* [p * chans + first + c] is below [pixels * chans], the length of
         [data]. *)
      Bytes.unsafe_set out
        ((p * count) + c)
        (Char.unsafe_chr
           (Bigarray.Array1.unsafe_get data ((p * chans) + first + c)))
    done
  done;
  Bytes.unsafe_to_string out

let image_object doc pixels =
  let shape = Nx.shape pixels in
  let rows = shape.(0) and cols = shape.(1) and chans = shape.(2) in
  let data = Bigarray.reshape_1 (Nx.to_bigarray pixels) (rows * cols * chans) in
  let plane = plane data ~pixels:(rows * cols) ~chans in
  let image space extra samples =
    add doc
      (stream
         ~dict:
           (Printf.sprintf
              " /Type /XObject /Subtype /Image /Width %d /Height %d \
               /ColorSpace /%s /BitsPerComponent 8 /Interpolate false%s"
              cols rows space extra)
         samples)
  in
  match chans with
  | 1 -> image "DeviceGray" "" (plane ~first:0 ~count:1)
  | 3 -> image "DeviceRGB" "" (plane ~first:0 ~count:3)
  | _ ->
      let mask = image "DeviceGray" "" (plane ~first:3 ~count:1) in
      image "DeviceRGB"
        (Printf.sprintf " /SMask %d 0 R" mask)
        (plane ~first:0 ~count:3)

let image doc ctx b box pixels =
  let shape = Nx.shape pixels in
  let h = shape.(0) and w = shape.(1) in
  match Vector.crop ctx.cut ctx.m box w h with
  | None -> ()
  | Some ((c0, r0, c1, r1) as window) ->
      let cw = Box2.w box /. float w and ch = Box2.h box /. float h in
      let x = Box2.minx box +. (float c0 *. cw)
      and y = Box2.miny box +. (float r0 *. ch) in
      let bw = float (c1 - c0) *. cw and bh = float (r1 - r0) *. ch in
      (* The unit square, its first row at the top, onto the box: corners within
         a tenth of the accuracy. *)
      let m = Affine.(ctx.m * translate x (y +. bh) * scale bw (-.bh)) in
      let d = ctx.d + 1 in
      let xx = rounded d m.xx and yx = rounded d m.yx in
      let xy = rounded d m.xy and yy = rounded d m.yy in
      let det = (xx *. yy) -. (xy *. yx) in
      (* A matrix that flattens the image as written paints nothing. *)
      if Float.is_finite det && det <> 0. then begin
        let name =
          match
            List.find_opt
              (fun (px, win, _) -> px == pixels && win = window)
              doc.images
          with
          | Some (_, _, name) -> name
          | None ->
              let px = Nx.slice [ R (r0, r1); R (c0, c1); A ] pixels in
              let name = xobject doc (image_object doc px) in
              doc.images <- (pixels, window, name) :: doc.images;
              name
        in
        Buffer.add_string b "q\n";
        (* Images are painted with the alpha of fills, which an enclosing
           instance may have set. *)
        if ctx.fill_set then add_alpha doc b "ca" 1.;
        List.iter
          (fun v ->
            add_num b d v;
            Buffer.add_char b ' ')
          [ xx; yx; xy; yy ];
        add_num b ctx.d m.x0;
        Buffer.add_char b ' ';
        add_num b ctx.d m.y0;
        Printf.bprintf b " cm\n/%s Do\nQ\n" name
      end

(* Forms *)

(* [reach m k p] is how far the pens of the strokes of [p] reach beyond their
   paths under [m], their lengths multiplied by [k]. Stamps keep pens, so their
   instances do not change it. *)
let reach m k p =
  let r = ref 0. in
  let rec walk m (p : Picture.t) =
    match p with
    | Stroke { stroke; _ } ->
        r := Float.max !r (Vector.reach (Vector.pen m k stroke) stroke)
    | Empty | Fill _ | Glyphs _ | Image _ -> ()
    | Group ps -> List.iter (walk m) ps
    | Transform { m = m'; picture } -> walk Affine.(m * m') picture
    | Clip { picture; _ }
    | Opacity { picture; _ }
    | Tag { picture; _ }
    | Stamp { picture; _ } ->
        walk m picture
  in
  walk (Vector.linear m) p;
  !r

(* [meet a b] is the intersection of [a] and [b], if they overlap. Boxes that
   touch meet on a line, where nothing paints, so the comparisons may be strict
   or not. *)
let meet a b =
  let minx = Float.max (Box2.minx a) (Box2.minx b)
  and miny = Float.max (Box2.miny a) (Box2.miny b) in
  let maxx = Float.min (Box2.maxx a) (Box2.maxx b)
  and maxy = Float.min (Box2.maxy a) (Box2.maxy b) in
  if minx <= maxx && miny <= maxy then
    Some (Box2.of_pts (P2.v minx miny) (P2.v maxx maxy))
  else None

(* [bbox ctx p ~pen] is a box of the frame holding what [p] paints there with
   the lengths of its pens multiplied by [pen], within the cut, or [None] if it
   paints nothing there. *)
let bbox ctx p ~pen =
  match Picture.bounds (Picture.transform ctx.m p) with
  | None -> None
  | Some bx -> meet (Vector.grown bx (reach ctx.m pen p +. 1.)) ctx.cut
  | exception Invalid_argument _ -> Some ctx.cut

(* [form doc ~group bbox content] is the name of a form XObject of [content]
   over [bbox], a transparency group if [group]. Equal forms are written
   once. *)
let form doc ~group bbox content =
  let b = Buffer.create 64 in
  Buffer.add_string b " /Type /XObject /Subtype /Form /BBox [";
  List.iteri
    (fun i v ->
      if i > 0 then Buffer.add_char b ' ';
      add_num b 3 v)
    [ Box2.minx bbox; Box2.miny bbox; Box2.maxx bbox; Box2.maxy bbox ];
  Printf.bprintf b "] /Resources %d 0 R" doc.resources;
  if group then Buffer.add_string b " /Group << /S /Transparency /I true >>";
  let dict = Buffer.contents b in
  let key = dict ^ content in
  match Hashtbl.find_opt doc.forms key with
  | Some name -> name
  | None ->
      let name = xobject doc (add doc (stream ~dict content)) in
      Hashtbl.add doc.forms key name;
      name

let rec holds_stroke (p : Picture.t) =
  match p with
  | Stroke _ -> true
  | Empty | Fill _ | Glyphs _ | Image _ -> false
  | Group ps -> List.exists holds_stroke ps
  | Transform { picture; _ }
  | Clip { picture; _ }
  | Opacity { picture; _ }
  | Tag { picture; _ }
  | Stamp { picture; _ } ->
      holds_stroke picture

(* [holds_opacity ~over p] is [true] iff [p] holds an opacity whose picture
   satisfies [over], which defaults to any. *)
let rec holds_opacity ?(over = fun _ -> true) (p : Picture.t) =
  match p with
  | Opacity { picture; _ } -> over picture
  | Empty | Fill _ | Stroke _ | Glyphs _ | Image _ -> false
  | Group ps -> List.exists (holds_opacity ~over) ps
  | Transform { picture; _ }
  | Clip { picture; _ }
  | Tag { picture; _ }
  | Stamp { picture; _ } ->
      holds_opacity ~over picture

(* Pictures *)

let rec picture doc ctx b (p : Picture.t) =
  match p with
  | Empty -> ()
  | Fill { rule; color; path } -> fill doc ctx b rule color path
  | Stroke { stroke = s; color; path } -> stroke doc ctx b s color path
  | Glyphs { color; at; run } -> glyphs doc ctx b color at run
  | Image { box; pixels } -> image doc ctx b box pixels
  | Group ps -> List.iter (picture doc ctx b) ps
  | Clip { rule; path; picture = p } ->
      (* A clip of no area within the cut shows nothing of its picture. *)
      let q = area_ops ctx path in
      if q <> "" then begin
        Buffer.add_string b "q\n";
        Buffer.add_string b q;
        Buffer.add_string b (if rule = `Even_odd then "W* n\n" else "W n\n");
        picture doc ctx b p;
        Buffer.add_string b "Q\n"
      end
  | Transform { m; picture = p } ->
      (* Transforms that compose beyond the range of floats, or to a map with no
         inverse, leave nothing to write. *)
      let m = Affine.(ctx.m * m) in
      if Affine.invert m <> None then picture doc { ctx with m } b p
  | Opacity { opacity; picture = p } -> (
      match bbox ctx p ~pen:ctx.pen with
      | Some bx when opacity > 0. ->
          let content = Buffer.create 256 in
          picture doc ctx content p;
          let name = form doc ~group:true bx (Buffer.contents content) in
          let a = alpha_key opacity in
          Printf.bprintf b "q\n/%s gs\n/%s Do\nQ\n"
            (state doc ("a" ^ a)
               (Printf.sprintf "<< /Type /ExtGState /ca %s /CA %s >>" a a))
            name
      | _ -> ())
  | Tag { picture = p; _ } -> picture doc ctx b p
  | Stamp { picture = p; xs; ys; scales; fills; strokes } ->
      stamp doc ctx b p xs ys scales fills strokes

and stamp doc ctx b p xs ys scales fills strokes =
  let n = Array.length xs in
  let lin = Vector.linear ctx.m in
  let scale i = match scales with None -> 1. | Some a -> a.(i) in
  let shown i =
    Float.is_finite xs.(i)
    && Float.is_finite ys.(i)
    && Float.is_finite (scale i)
  in
  let pick a i outer = match a with Some a -> Fixed a.(i) | None -> outer in
  (* Instances keep the pens of [p], which reach this far around them. *)
  let kept = reach ctx.m ctx.pen p in
  (* [whole i] paints instance [i] in full: its picture mapped to the frame,
     with its own colours and pens. *)
  let whole i =
    let s = scale i in
    let m = Affine.(ctx.m * translate xs.(i) ys.(i) * scale s s) in
    let meets =
      match
        Option.map
          (fun bx -> Vector.grown bx kept)
          (Picture.bounds (Picture.transform m p))
      with
      | Some bx -> Vector.meets ctx.cut bx
      | None -> false
      | exception Invalid_argument _ -> true
    in
    if meets then
      picture doc
        {
          ctx with
          m;
          fills = pick fills i ctx.fills;
          strokes = pick strokes i ctx.strokes;
          pen = ctx.pen /. s;
        }
        b p
  in
  let pens = if scales = None then [] else Vector.pens lin ctx.pen p in
  let translucent =
    List.exists
      (Array.exists (fun c -> Color.alpha c < 1.))
      (List.filter_map Fun.id [ fills; strokes ])
  in
  (* A transparency group resets alphas, and the pen an instance sets reaches
     the strokes of the group by the specification but not in Poppler, which
     resets it there too. *)
  if
    scales <> None
    && (Vector.scales_within p
       || List.length pens > 1
       || holds_opacity ~over:holds_stroke p)
    || (translucent && holds_opacity p)
  then
    for i = 0 to n - 1 do
      if shown i then whole i
    done
  else begin
    let extent =
      match Picture.bounds (Picture.transform lin p) with
      | bx -> bx
      | exception Invalid_argument _ -> None
    in
    (* An instance places the form's points by its translation, written to the
       accuracy, and by its scale; the form's numbers, its scale and the pen it
       sets get the decimals that keep each within a tenth of it. Scales written
       with [ds] decimals move the points of the picture, which lie within
       [extent] of an instance's position, by that much. *)
    let ds =
      match extent with
      | None -> 17
      | Some e ->
          let far =
            Float.max
              (Float.max (Float.abs (Box2.minx e)) (Float.abs (Box2.maxx e)))
              (Float.max (Float.abs (Box2.miny e)) (Float.abs (Box2.maxy e)))
          in
          Vector.finer 1 (Vector.decimals (ctx.mag *. far))
    in
    let pen_decimals sw = Vector.finer 1 (Vector.decimals (ctx.mag *. sw)) in
    (* The scale each instance writes, or [0.] for one the form cannot carry:
       one written as zero, one that magnifies the form beyond the decimals of
       numbers, or one whose pen is written as zero. Pens are divided by the
       scale written, so that viewers stroke them at their width. *)
    let written =
      Array.init n (fun i ->
          if not (shown i) then 0.
          else
            let s = scale i in
            let sw = rounded ds s in
            if ctx.mag *. s >= 1e14 then 0.
            else
              match pens with
              | [ pen ] when is_zero (pen_decimals sw) (pen.width /. sw) -> 0.
              | _ -> sw)
    in
    let largest = ref 1. and smallest = ref 1. in
    Array.iter
      (fun s ->
        if s > 0. then begin
          largest := Float.max !largest s;
          smallest := Float.min !smallest s
        end)
      written;
    let mag = ctx.mag *. !largest in
    let dctx =
      {
        m = lin;
        cut = Vector.all;
        mag;
        d = Vector.finer 1 (Vector.decimals mag);
        fills = (if fills = None then ctx.fills else Inherit);
        strokes = (if strokes = None then ctx.strokes else Inherit);
        fill_set = ctx.fill_set || fills <> None;
        stroke_set = ctx.stroke_set || strokes <> None;
        (* The widest pen an instance sets: pens kept while instances shrink
           reach beyond the picture's box, and the boxes of the form and of the
           groups within it hold them. *)
        pen = ctx.pen /. !smallest;
        pen_set = ctx.pen_set || scales <> None;
      }
    in
    let form =
      if not (Array.exists (fun s -> s > 0.) written) then None
      else
        match bbox dctx p ~pen:dctx.pen with
        | None -> None
        | Some bx ->
            let content = Buffer.create 256 in
            picture doc dctx content p;
            Some (form doc ~group:false bx (Buffer.contents content))
    in
    for i = 0 to n - 1 do
      let sw = written.(i) in
      if sw = 0. then (if shown i then whole i)
      else
        match form with
        | None -> ()
        | Some name ->
            let at = P2.transform ctx.m (P2.v xs.(i) ys.(i)) in
            let x = P2.x at and y = P2.y at in
            let shows =
              match extent with
              | None -> true
              | Some e ->
                  Vector.overlaps ctx.cut
                    (x +. (sw *. Box2.minx e) -. kept)
                    (y +. (sw *. Box2.miny e) -. kept)
                    (x +. (sw *. Box2.maxx e) +. kept)
                    (y +. (sw *. Box2.maxy e) +. kept)
            in
            if shows then begin
              Buffer.add_string b "q\n";
              Option.iter
                (fun a ->
                  add_color doc b ~stroking:false ~set:ctx.fill_set a.(i))
                fills;
              Option.iter
                (fun a ->
                  add_color doc b ~stroking:true ~set:ctx.stroke_set a.(i))
                strokes;
              (match (scales, pens) with
              | Some _, [ pen ] ->
                  let k = 1. /. sw in
                  add_pen b (pen_decimals sw)
                    {
                      width = pen.width *. k;
                      dash = List.map (fun d -> d *. k) pen.dash;
                      offset = pen.offset *. k;
                    }
              | _ -> ());
              add_num b ds sw;
              Buffer.add_string b " 0 0 ";
              add_num b ds sw;
              Buffer.add_char b ' ';
              add_num b ctx.d x;
              Buffer.add_char b ' ';
              add_num b ctx.d y;
              Buffer.add_string b " cm\n/";
              Buffer.add_string b name;
              Buffer.add_string b " Do\nQ\n"
            end
    done
  end

(* Fonts *)

let pdf_name s =
  let b = Buffer.create (String.length s) in
  String.iter
    (fun c ->
      match c with
      | '!' .. '~' when not (String.contains "()<>[]{}/%#" c) ->
          Buffer.add_char b c
      | c -> Printf.bprintf b "#%02X" (Char.code c))
    s;
  if Buffer.length b = 0 then "Font" else Buffer.contents b

let add_font doc u =
  let f = u.font in
  let base = pdf_name (Font.postscript_name f) in
  let bytes = Font.bytes f in
  let file =
    add doc
      (stream ~dict:(Printf.sprintf " /Length1 %d" (String.length bytes)) bytes)
  in
  let num v =
    let b = Buffer.create 8 in
    add_num b 3 v;
    Buffer.contents b
  in
  let em v = num (v *. 1000.) in
  let bounds = Font.bounds f in
  (* An estimate of the dominant stem's width from the weight class. *)
  let stem = 10. +. (220. *. (float (Font.weight f) -. 50.) /. 900.) in
  let flags = 4 lor if Font.slant f = `Normal then 0 else 64 in
  let descriptor =
    add doc
      (Printf.sprintf
         "<< /Type /FontDescriptor /FontName /%s /Flags %d /FontBBox [%s %s %s \
          %s] /ItalicAngle %s /Ascent %s /Descent %s /CapHeight %s /StemV %s \
          /FontFile2 %d 0 R >>"
         base flags
         (em (Box2.minx bounds))
         (em (-.Box2.maxy bounds))
         (em (Box2.maxx bounds))
         (em (-.Box2.miny bounds))
         (num (Font.italic_angle f *. 180. /. Float.pi))
         (em (Font.ascent f))
         (em (-.Font.descent f))
         (em (Font.cap_height f))
         (num (Float.round stem))
         file)
  in
  let glyphs =
    Hashtbl.fold (fun g () acc -> g :: acc) u.used [] |> List.sort Int.compare
  in
  let widths = Buffer.create 256 in
  List.iteri
    (fun i g ->
      if i > 0 then Buffer.add_char widths ' ';
      Printf.bprintf widths "%d [" g;
      add_num widths 3 (width f g);
      Buffer.add_char widths ']')
    glyphs;
  let cid =
    add doc
      (Printf.sprintf
         "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /%s /CIDSystemInfo \
          << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> \
          /FontDescriptor %d 0 R /DW 1000 /W [%s] /CIDToGIDMap /Identity >>"
         base descriptor (Buffer.contents widths))
  in
  let cmap = Buffer.create 1024 in
  Buffer.add_string cmap
    "/CIDInit /ProcSet findresource begin\n\
     12 dict begin\n\
     begincmap\n\
     /CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n\
     /CMapName /Adobe-Identity-UCS def\n\
     /CMapType 2 def\n\
     1 begincodespacerange\n\
     <0000> <FFFF>\n\
     endcodespacerange\n";
  let mapped =
    List.filter_map
      (fun g -> Option.map (fun s -> (g, s)) (Hashtbl.find_opt u.text g))
      glyphs
  in
  (* A CMap's bfchar blocks hold at most 100 entries. *)
  let rec blocks = function
    | [] -> ()
    | l ->
        let block = List.filteri (fun i _ -> i < 100) l in
        Printf.bprintf cmap "%d beginbfchar\n" (List.length block);
        List.iter
          (fun (g, s) ->
            Printf.bprintf cmap "<%04X> <" g;
            add_utf16 cmap s;
            Buffer.add_string cmap ">\n")
          block;
        Buffer.add_string cmap "endbfchar\n";
        blocks (List.filteri (fun i _ -> i >= 100) l)
  in
  blocks mapped;
  Buffer.add_string cmap
    "endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend";
  let to_unicode = add doc (stream (Buffer.contents cmap)) in
  set doc u.number
    (Printf.sprintf
       "<< /Type /Font /Subtype /Type0 /BaseFont /%s /Encoding /Identity-H \
        /DescendantFonts [%d 0 R] /ToUnicode %d 0 R >>"
       base cid to_unicode)

(* Documents *)

let render r =
  let w = Renderable.w r and h = Renderable.h r in
  (* The document's structure takes the first objects, then come those the
     picture uses, in the order it first uses them. *)
  let catalog = 1 and pages = 2 and page = 3 and contents = 4 in
  let doc =
    {
      bodies = Dynarray.create ();
      resources = 5;
      fonts = [];
      xobjects = [];
      forms = Hashtbl.create 16;
      states = [];
      state_names = Hashtbl.create 16;
      images = [];
    }
  in
  for _ = 1 to 5 do
    ignore (add doc "")
  done;
  let size v =
    let b = Buffer.create 16 in
    add_num b 9 v;
    Buffer.contents b
  in
  let margin = Float.max w h in
  let ctx =
    {
      m = Affine.id;
      cut =
        Box2.v (-.margin) (-.margin) (w +. (2. *. margin)) (h +. (2. *. margin));
      mag = 1.;
      d = Vector.decimals 1.;
      fills = Own;
      strokes = Own;
      fill_set = false;
      stroke_set = false;
      pen = 1.;
      pen_set = false;
    }
  in
  let body = Buffer.create 4096 in
  (* The page's y-down coordinates. *)
  Printf.bprintf body "1 0 0 -1 0 %s cm\n" (size h);
  picture doc ctx body (Renderable.picture r);
  List.iter (add_font doc) (List.rev doc.fonts);
  let resources = Buffer.create 256 in
  let category name entries =
    if entries <> [] then
      Printf.bprintf resources " /%s << %s >>" name (String.concat " " entries)
  in
  category "Font"
    (List.rev_map
       (fun u -> Printf.sprintf "/%s %d 0 R" u.name u.number)
       doc.fonts);
  category "XObject"
    (List.rev_map
       (fun (name, n) -> Printf.sprintf "/%s %d 0 R" name n)
       doc.xobjects);
  category "ExtGState"
    (List.rev_map
       (fun (name, dict) -> Printf.sprintf "/%s %s" name dict)
       doc.states);
  set doc doc.resources (Printf.sprintf "<<%s >>" (Buffer.contents resources));
  set doc contents (stream (Buffer.contents body));
  set doc page
    (Printf.sprintf
       "<< /Type /Page /Parent %d 0 R /MediaBox [0 0 %s %s] /Resources %d 0 R \
        /Contents %d 0 R >>"
       pages (size w) (size h) doc.resources contents);
  set doc pages
    (Printf.sprintf "<< /Type /Pages /Kids [%d 0 R] /Count 1 >>" page);
  set doc catalog (Printf.sprintf "<< /Type /Catalog /Pages %d 0 R >>" pages);
  let out = Buffer.create 65536 in
  Buffer.add_string out "%PDF-1.7\n%\xe2\xe3\xcf\xd3\n";
  let count = Dynarray.length doc.bodies in
  let offsets =
    Array.init count (fun i ->
        let off = Buffer.length out in
        Printf.bprintf out "%d 0 obj\n%s\nendobj\n" (i + 1)
          (Dynarray.get doc.bodies i);
        off)
  in
  let xref = Buffer.length out in
  Printf.bprintf out "xref\n0 %d\n0000000000 65535 f \n" (count + 1);
  Array.iter (fun off -> Printf.bprintf out "%010d 00000 n \n" off) offsets;
  Printf.bprintf out
    "trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" (count + 1)
    catalog xref;
  Buffer.contents out
