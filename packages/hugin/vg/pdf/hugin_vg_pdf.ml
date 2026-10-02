(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin_gg
open Hugin_font
open Hugin_vg

(* Numbers *)

let add_num b d v = Vector.add_fixed b d v
let rounded = Vector.rounded

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
  let data = Compress_deflate.Zlib.compress data in
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

let add_paint doc b ~stroking ~set (paint : Vector.paint) own =
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

let area_ops (ctx : Vector.ctx) path =
  let b = Buffer.create 64 in
  Vector.area ctx.m ctx.cut path (ops b ctx.d);
  Buffer.contents b

(* Leaves *)

let fill doc b (ctx : Vector.ctx) rule color path =
  let p = area_ops ctx path in
  if p <> "" then begin
    Buffer.add_string b "q\n";
    add_paint doc b ~stroking:false ~set:ctx.fill_set ctx.fills color;
    Buffer.add_string b p;
    Buffer.add_string b (if rule = `Even_odd then "f*\nQ\n" else "f\nQ\n")
  end

let add_dash b d (pen : Vector.pen) ~offset =
  let d = Vector.dash_decimals d in
  Buffer.add_char b '[';
  List.iteri
    (fun i v ->
      if i > 0 then Buffer.add_char b ' ';
      add_num b d v)
    pen.dash;
  Buffer.add_string b "] ";
  add_num b d (Vector.phase pen offset);
  Buffer.add_string b " d\n"

let add_pen b d (pen : Vector.pen) =
  add_num b d pen.width;
  Buffer.add_string b " w\n";
  if Vector.is_dashed d pen then add_dash b d pen ~offset:0.

(* A width written as zero would be a viewer's thinnest line. *)
let stroke doc b (ctx : Vector.ctx) (st : Vector.stroke) color outline =
  if ctx.pen_set || not (is_zero ctx.d st.pen.width) then begin
    (* The dashes are set before the first piece, and again before each piece
       that starts at another offset, which is cut from within a subpath and
       stroked on its own. *)
    let body = Buffer.create 64 and offset = ref Float.nan in
    let started = ref false in
    outline
      ~piece:(fun d ->
        if st.dashed && (not ctx.pen_set) && not (Float.equal d !offset) then begin
          if !started then Buffer.add_string body "S\n";
          add_dash body ctx.d st.pen ~offset:d;
          offset := d
        end;
        started := true)
      (ops body ctx.d);
    if !started then begin
      let s = st.style in
      Buffer.add_string b "q\n";
      add_paint doc b ~stroking:true ~set:ctx.stroke_set ctx.strokes color;
      if not ctx.pen_set then begin
        add_num b ctx.d st.pen.width;
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
        st.frame;
      Buffer.add_buffer b body;
      Buffer.add_string b "S\nQ\n"
    end
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

let glyphs doc b (ctx : Vector.ctx) color at run =
  let m = Affine.(ctx.m * translate (P2.x at) (P2.y at)) in
  let lin = Affine.linear m in
  let s = Affine.stretch lin in
  let f = Run.font run and n = Run.length run in
  let e = Vector.ems f in
  (* The size is written with the decimals that keep outlines within a tenth of
     the accuracy, and glyphs are placed by the size as written. *)
  let ds = Vector.scale_decimals ctx e in
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

(* Equal pictures write equal bytes, so an image is written once for tensors
   that are equal as [Picture.equal] compares them. *)
let same_pixels a b = a == b || Nx.item [] (Nx.array_equal a b)

let image doc b (ctx : Vector.ctx) (im : Vector.image) =
  let c0, r0, c1, r1 = im.window and x = im.x and y = im.y in
  let bw = im.w and bh = im.h in
  (* The unit square, its first row at the top, onto the box: corners within a
     tenth of the accuracy. *)
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
          (fun (px, win, _) -> win = im.window && same_pixels px im.pixels)
          doc.images
      with
      | Some (_, _, name) -> name
      | None ->
          let px = Nx.slice [ R (r0, r1); R (c0, c1); A ] im.pixels in
          let name = xobject doc (image_object doc px) in
          doc.images <- (im.pixels, im.window, name) :: doc.images;
          name
    in
    Buffer.add_string b "q\n";
    (* Images are painted with the alpha of fills, which an enclosing instance
       may have set. *)
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

let holds_stroke = Instances.exists (function Stroke _ -> true | _ -> false)

(* [holds_opacity ~over p] is [true] iff [p] holds an opacity whose picture
   satisfies [over], which defaults to any. *)
let holds_opacity ?(over = fun _ -> true) =
  Instances.exists (function
    | Opacity { picture; _ } -> over picture
    | _ -> false)

(* Nodes *)

let clip b (ctx : Vector.ctx) rule path k =
  (* A clip of no area within the cut shows nothing of its picture. *)
  let q = area_ops ctx path in
  if q <> "" then begin
    Buffer.add_string b "q\n";
    Buffer.add_string b q;
    Buffer.add_string b (if rule = `Even_odd then "W* n\n" else "W n\n");
    k b;
    Buffer.add_string b "Q\n"
  end

let opacity doc b ctx opacity p k =
  match Vector.box ctx p with
  | Some bx when opacity > 0. ->
      let content = Buffer.create 256 in
      k content;
      let name = form doc ~group:true bx (Buffer.contents content) in
      let a = alpha_key opacity in
      Printf.bprintf b "q\n/%s gs\n/%s Do\nQ\n"
        (state doc ("a" ^ a)
           (Printf.sprintf "<< /Type /ExtGState /ca %s /CA %s >>" a a))
        name
  | _ -> ()

(* Stamps: a form that each instance paints under its own matrix, its colours
   and pen set before. *)

let pen_decimals = Vector.scale_decimals

(* A form cannot carry a scale written as zero, one that magnifies it beyond the
   decimals of numbers, or one whose pen is written as zero. It carries none
   where a transparency group resets alphas, and where the pen an instance sets
   reaches the strokes of a group: it does by the specification but not in
   Poppler, which resets it there too. *)
let carry (ctx : Vector.ctx) (st : Vector.stamp) =
  let full =
    (st.pens <> [] && holds_opacity ~over:holds_stroke st.picture)
    || (st.translucent && holds_opacity st.picture)
  in
  fun s ->
    let sw = rounded st.decimals s in
    if full || sw = 0. || ctx.mag *. s >= 1e14 then Float.nan
    else
      match st.pens with
      | [ pen ] when is_zero (pen_decimals ctx sw) (pen.width /. sw) ->
          Float.nan
      | _ -> sw

let define doc _ bbox k =
  let content = Buffer.create 256 in
  k content;
  form doc ~group:false bbox (Buffer.contents content)

let use doc b (ctx : Vector.ctx) (st : Vector.stamp) name i at sw =
  Buffer.add_string b "q\n";
  Option.iter
    (fun a -> add_color doc b ~stroking:false ~set:ctx.fill_set a.(i))
    st.fills;
  Option.iter
    (fun a -> add_color doc b ~stroking:true ~set:ctx.stroke_set a.(i))
    st.strokes;
  (match st.pens with
  | [ pen ] -> add_pen b (pen_decimals ctx sw) (Vector.scale_pen (1. /. sw) pen)
  | _ -> ());
  add_num b st.decimals sw;
  Buffer.add_string b " 0 0 ";
  add_num b st.decimals sw;
  Buffer.add_char b ' ';
  add_num b ctx.d (P2.x at);
  Buffer.add_char b ' ';
  add_num b ctx.d (P2.y at);
  Buffer.add_string b " cm\n/";
  Buffer.add_string b name;
  Buffer.add_string b " Do\nQ\n"

let target doc =
  {
    Vector.fill = fill doc;
    stroke = stroke doc;
    glyphs = glyphs doc;
    image = image doc;
    clip;
    opacity = opacity doc;
    tag = (fun b _ _ k -> k b);
    carry;
    define = define doc;
    use = use doc;
    instance = (fun b _ _ k -> k b);
  }

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

(* [tag bytes] is the six capital letters that name a font subset, drawn from a
   digest of its [bytes] so that different subsets get different names. *)
let tag bytes =
  let d = Digest.string bytes in
  String.init 6 (fun i -> Char.chr (Char.code 'A' + (Char.code d.[i] mod 26)))

let add_font doc u =
  let f = u.font in
  let glyphs =
    Hashtbl.fold (fun g () acc -> g :: acc) u.used [] |> List.sort Int.compare
  in
  let bytes = Font.subset f glyphs in
  let base = tag bytes ^ "+" ^ pdf_name (Font.postscript_name f) in
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
  let body = Buffer.create 4096 in
  (* The page's y-down coordinates. *)
  Printf.bprintf body "1 0 0 -1 0 %s cm\n" (size h);
  Vector.walk (target doc) (Vector.page r) body (Renderable.picture r);
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
