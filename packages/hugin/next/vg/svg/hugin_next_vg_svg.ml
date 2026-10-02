(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin_next_gg
open Hugin_next_font
open Hugin_next_vg

(* Text *)

(* [is_held c] is [true] iff XML holds the character [c] as itself in text that
   viewers draw, [c] being no surrogate: tabs and line ends are not, since
   viewers draw them as spaces. *)
let is_held c = c >= 0x20 && c < 0xFFFE

(* [add_text b s] adds the UTF-8 string [s] as XML text or attribute value:
   markup characters escaped, tabs and line ends as character references, and
   what XML cannot hold, invalid bytes included, as U+FFFD. *)
let add_text b s =
  let rec loop i =
    if i < String.length s then begin
      let d = String.get_utf_8_uchar s i in
      let u = Uchar.utf_decode_uchar d in
      (match Uchar.to_int u with
      | 0x3C -> Buffer.add_string b "&lt;"
      | 0x3E -> Buffer.add_string b "&gt;"
      | 0x26 -> Buffer.add_string b "&amp;"
      | 0x22 -> Buffer.add_string b "&quot;"
      | 0x27 -> Buffer.add_string b "&apos;"
      | (0x9 | 0xA | 0xD) as c -> Printf.bprintf b "&#%d;" c
      | c when is_held c || c > 0xFFFF -> Buffer.add_utf_8_uchar b u
      | _ -> Buffer.add_utf_8_uchar b Uchar.rep);
      loop (i + Uchar.utf_decode_length d)
    end
  in
  loop 0

let base64_alphabet =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

let add_base64 b s =
  let len = String.length s in
  let byte i = if i < len then Char.code s.[i] else 0 in
  let c k = Buffer.add_char b base64_alphabet.[k] in
  let i = ref 0 in
  while !i < len do
    let b0 = byte !i and b1 = byte (!i + 1) and b2 = byte (!i + 2) in
    c (b0 lsr 2);
    c (((b0 land 3) lsl 4) lor (b1 lsr 4));
    if !i + 1 < len then c (((b1 land 0xf) lsl 2) lor (b2 lsr 6))
    else Buffer.add_char b '=';
    if !i + 2 < len then c (b2 land 0x3f) else Buffer.add_char b '=';
    i := !i + 3
  done

(* [id kind content] is the id of a definition of [content]. *)
let id kind content =
  "h" ^ kind ^ String.sub (Digest.to_hex (Digest.string content)) 0 16

(* Documents *)

type doc = {
  defs : Buffer.t;  (** Definitions, in the order of first use. *)
  defined : (string, unit) Hashtbl.t;
  mutable families : (Font.t * string) list;
}

let define doc id content =
  if not (Hashtbl.mem doc.defined id) then begin
    Hashtbl.add doc.defined id ();
    Buffer.add_string doc.defs content
  end

(* [family doc font] is the family under which [doc] embeds [font]. *)
let family doc font =
  match List.assq_opt font doc.families with
  | Some f -> f
  | None ->
      let bytes = Font.bytes font in
      let f = id "f" bytes in
      let b = Buffer.create ((String.length bytes * 4 / 3) + 128) in
      Printf.bprintf b
        "<style>@font-face{font-family:%s;src:url(data:font/ttf;base64," f;
      add_base64 b bytes;
      Buffer.add_string b ")}</style>\n";
      define doc f (Buffer.contents b);
      doc.families <- (font, f) :: doc.families;
      f

(* Writing in a frame *)

let add_num b d v = Vector.add_fixed b d v

let add_attr_num b name d v =
  Printf.bprintf b " %s=\"" name;
  add_num b d v;
  Buffer.add_char b '"'

let add_color b name ~set c =
  let level v = int_of_float (Float.round (v *. 255.)) in
  Printf.bprintf b " %s=\"#%02x%02x%02x\"" name
    (level (Color.r c))
    (level (Color.g c))
    (level (Color.b c));
  if Color.alpha c < 1. || set then
    add_attr_num b (name ^ "-opacity") 3 (Color.alpha c)

let add_paint b name ~set (paint : Vector.paint) own =
  match paint with
  | Own -> add_color b name ~set own
  | Fixed c -> add_color b name ~set c
  | Inherit -> ()

let add_matrix b (u : Affine.t) ox oy d =
  Buffer.add_string b " transform=\"matrix(";
  List.iter
    (fun v ->
      add_num b 15 v;
      Buffer.add_char b ' ')
    [ u.xx; u.yx; u.xy; u.yy ];
  add_num b d ox;
  Buffer.add_char b ' ';
  add_num b d oy;
  Buffer.add_string b ")\""

(* [data d] is a sink writing path data with [d] decimals into a buffer. *)
let data b d =
  let pt x y =
    add_num b d x;
    Buffer.add_char b ' ';
    add_num b d y
  in
  {
    Vector.move =
      (fun x y ->
        Buffer.add_char b 'M';
        pt x y);
    line =
      (fun x y ->
        Buffer.add_char b 'L';
        pt x y);
    cubic =
      (fun x1 y1 x2 y2 x y ->
        Buffer.add_char b 'C';
        pt x1 y1;
        Buffer.add_char b ' ';
        pt x2 y2;
        Buffer.add_char b ' ';
        pt x y);
    close = (fun () -> Buffer.add_char b 'Z');
  }

let area_data (ctx : Vector.ctx) m path =
  let b = Buffer.create 64 in
  Vector.area m ctx.cut path (data b ctx.d);
  Buffer.contents b

(* Leaves *)

let fill b (ctx : Vector.ctx) rule color path =
  let d = area_data ctx ctx.m path in
  if d <> "" then begin
    Printf.bprintf b "<path d=\"%s\"" d;
    add_paint b "fill" ~set:ctx.fill_set ctx.fills color;
    if rule = `Even_odd then Buffer.add_string b " fill-rule=\"evenodd\"";
    if ctx.stroke_set then Buffer.add_string b " stroke=\"none\"";
    Buffer.add_string b "/>\n"
  end

(* [add_pen b d pen ~offset] adds the attributes of [pen] in a frame of [d]
   decimals, its dashes starting [offset] further. *)
let add_pen b d (pen : Vector.pen) ~offset =
  add_attr_num b "stroke-width" d pen.width;
  if Vector.is_dashed d pen then begin
    let dd = Vector.dash_decimals d in
    Buffer.add_string b " stroke-dasharray=\"";
    List.iteri
      (fun i v ->
        if i > 0 then Buffer.add_char b ' ';
        add_num b dd v)
      pen.dash;
    Buffer.add_char b '"';
    let offset = Vector.phase pen offset in
    if offset <> 0. then add_attr_num b "stroke-dashoffset" dd offset
  end

let stroke b (ctx : Vector.ctx) (st : Vector.stroke) color outline =
  let pieces = ref [] and current = Buffer.create 64 and offset = ref 0. in
  let flush () =
    if Buffer.length current > 0 then begin
      pieces := (!offset, Buffer.contents current) :: !pieces;
      Buffer.clear current
    end
  in
  outline
    ~piece:(fun d ->
      if d <> !offset then begin
        flush ();
        offset := d
      end)
    (data current ctx.d);
  flush ();
  let element (offset, d) =
    Buffer.add_string b "<path";
    Option.iter (fun u -> add_matrix b u 0. 0. ctx.d) st.frame;
    Printf.bprintf b " d=\"%s\" fill=\"none\"" d;
    add_paint b "stroke" ~set:ctx.stroke_set ctx.strokes color;
    if not ctx.pen_set then add_pen b ctx.d st.pen ~offset;
    let s = st.style in
    (match Stroke.cap s with
    | `Butt -> ()
    | `Round -> Buffer.add_string b " stroke-linecap=\"round\""
    | `Square -> Buffer.add_string b " stroke-linecap=\"square\"");
    (match Stroke.join s with
    | `Miter ->
        if Stroke.miter_limit s <> 4. then
          add_attr_num b "stroke-miterlimit" 3 (Stroke.miter_limit s)
    | `Round -> Buffer.add_string b " stroke-linejoin=\"round\""
    | `Bevel -> Buffer.add_string b " stroke-linejoin=\"bevel\"");
    Buffer.add_string b "/>\n"
  in
  (* Dashed pieces cut from within a subpath start at their own offsets; all the
     others go in one element. *)
  let pieces = List.rev !pieces in
  let starts, others = List.partition (fun (d, _) -> d = 0.) pieces in
  if starts <> [] then element (0., String.concat "" (List.map snd starts));
  List.iter element others

(* [text_of r] is the text of [r] if each of its glyphs renders one character of
   the Basic Multilingual Plane that XML holds and that its font maps to it,
   glyph [0] standing for the characters the font lacks. *)
let text_of run =
  let text = Run.text run and n = Run.length run and font = Run.font run in
  let b = Buffer.create (String.length text) in
  let rec loop i =
    if i = n then Some (Buffer.contents b)
    else
      let start = Run.cluster run i in
      let stop =
        if i + 1 < n then Run.cluster run (i + 1) else String.length text
      in
      if start >= String.length text then None
      else
        let d = String.get_utf_8_uchar text start in
        let u = Uchar.utf_decode_uchar d in
        if
          start + Uchar.utf_decode_length d = stop
          && is_held (Uchar.to_int u)
          && Run.glyph run i <> 0
          && Font.glyph font u = Run.glyph run i
        then begin
          Buffer.add_utf_8_uchar b u;
          loop (i + 1)
        end
        else None
  in
  loop 0

(* Turns off the viewer's shaping, so that it draws the glyphs the cmap gives
   where the run puts them. *)
let unshaped =
  "font-variant-ligatures:none;font-kerning:none;font-feature-settings:'liga' \
   0,'clig' 0,'calt' 0,'rlig' 0,'kern' 0,'ccmp' 0,'locl' 0,'mark' 0,'mkmk' 0"

let glyphs doc b (ctx : Vector.ctx) color at run =
  let m = Affine.(ctx.m * translate (P2.x at) (P2.y at)) in
  if Vector.run_meets ctx.cut m run then
    match text_of run with
    | Some text ->
        let n = Run.length run in
        let lin = Affine.linear m in
        let s = Affine.stretch lin in
        (* Glyph positions in the frame the text is written in, the frame's own
           if it only scales evenly, else one turned with the run. *)
        let frame =
          if Vector.is_even lin then None else Some (Vector.unit lin)
        in
        let px i =
          match frame with
          | None -> P2.x (P2.transform m (P2.v (Run.x run i) (Run.y run i)))
          | Some _ -> s *. Run.x run i
        in
        let py i =
          match frame with
          | None -> P2.y (P2.transform m (P2.v (Run.x run i) (Run.y run i)))
          | Some _ -> s *. Run.y run i
        in
        (* Positions under a matrix add their errors to its origin's. *)
        let d = if frame = None then ctx.d else Vector.finer 1 ctx.d in
        Buffer.add_string b "<text";
        Option.iter (fun u -> add_matrix b u m.x0 m.y0 ctx.d) frame;
        Buffer.add_string b " x=\"";
        for i = 0 to n - 1 do
          if i > 0 then Buffer.add_char b ' ';
          add_num b d (px i)
        done;
        Buffer.add_string b "\" y=\"";
        let level = ref true in
        for i = 1 to n - 1 do
          if py i <> py 0 then level := false
        done;
        for i = 0 to if !level then 0 else n - 1 do
          if i > 0 then Buffer.add_char b ' ';
          add_num b d (py i)
        done;
        Printf.bprintf b "\" font-family=\"%s\"" (family doc (Run.font run));
        (* The size gets the decimals that keep outlines, which reach [e] ems
           from their origin, within a tenth of the accuracy. *)
        add_attr_num b "font-size"
          (Vector.scale_decimals ctx (Vector.ems (Run.font run)))
          (Run.size run *. s);
        add_paint b "fill" ~set:ctx.fill_set ctx.fills color;
        if ctx.stroke_set then Buffer.add_string b " stroke=\"none\"";
        Printf.bprintf b " style=\"%s\" xml:space=\"preserve\">" unshaped;
        add_text b text;
        Buffer.add_string b "</text>\n"
    | None ->
        Buffer.add_string b "<g aria-label=\"";
        add_text b (Run.text run);
        Buffer.add_char b '"';
        add_paint b "fill" ~set:ctx.fill_set ctx.fills color;
        if ctx.stroke_set then Buffer.add_string b " stroke=\"none\"";
        Buffer.add_string b ">\n";
        let font = Run.font run and size = Run.size run in
        for i = 0 to Run.length run - 1 do
          let g =
            Affine.(m * translate (Run.x run i) (Run.y run i) * scale size size)
          in
          let d = area_data ctx g (Font.outline font (Run.glyph run i)) in
          if d <> "" then Printf.bprintf b "<path d=\"%s\"/>\n" d
        done;
        Buffer.add_string b "</g>\n"

let image b (ctx : Vector.ctx) (im : Vector.image) =
  let c0, r0, c1, r1 = im.window in
  let shape = Nx.shape im.pixels in
  let pixels =
    if c0 = 0 && r0 = 0 && c1 = shape.(1) && r1 = shape.(0) then im.pixels
    else Nx.slice [ R (r0, r1); R (c0, c1); A ] im.pixels
  in
  let x = im.x and y = im.y and bw = im.w and bh = im.h in
  let lin = Affine.linear ctx.m in
  Buffer.add_string b "<image";
  if Vector.is_axial lin then begin
    let p = P2.transform ctx.m (P2.v x y)
    and q = P2.transform ctx.m (P2.v (x +. bw) (y +. bh)) in
    (* The corners rounded, so that each edge is within the accuracy. *)
    let round = Vector.rounded ctx.d in
    add_attr_num b "x" ctx.d (P2.x p);
    add_attr_num b "y" ctx.d (P2.y p);
    add_attr_num b "width" ctx.d (round (P2.x q) -. round (P2.x p));
    add_attr_num b "height" ctx.d (round (P2.y q) -. round (P2.y p))
  end
  else begin
    let s = Affine.stretch lin in
    let o = P2.transform ctx.m (P2.v x y) in
    add_matrix b (Vector.unit lin) (P2.x o) (P2.y o) ctx.d;
    (* Lengths under a matrix add their errors to its origin's. *)
    add_attr_num b "width" (Vector.finer 1 ctx.d) (s *. bw);
    add_attr_num b "height" (Vector.finer 1 ctx.d) (s *. bh)
  end;
  Buffer.add_string b
    " preserveAspectRatio=\"none\" style=\"image-rendering:pixelated\" \
     href=\"data:image/png;base64,";
  add_base64 b (Nx_io.encode_png pixels);
  Buffer.add_string b "\"/>\n"

(* Tags *)

(* [add_json_string b s] adds [s] as a JSON string, escaping what XML cannot
   hold, so that the string survives the attribute. *)
let add_json_string b s =
  Buffer.add_char b '"';
  let rec loop i =
    if i < String.length s then begin
      let d = String.get_utf_8_uchar s i in
      let u =
        if Uchar.utf_decode_is_valid d then Uchar.utf_decode_uchar d
        else Uchar.rep
      in
      (match Uchar.to_int u with
      | 0x22 -> Buffer.add_string b "\\\""
      | 0x5C -> Buffer.add_string b "\\\\"
      | c when c < 0x20 || c = 0xFFFE || c = 0xFFFF ->
          Printf.bprintf b "\\u%04x" c
      | _ -> Buffer.add_utf_8_uchar b u);
      loop (i + Uchar.utf_decode_length d)
    end
  in
  loop 0;
  Buffer.add_char b '"'

let add_tag_id b id =
  let json = Buffer.create 32 in
  Buffer.add_char json '[';
  List.iteri
    (fun i seg ->
      if i > 0 then Buffer.add_char json ',';
      match (seg : Nx.Ptree.Path.seg) with
      | Index k -> Buffer.add_string json (string_of_int k)
      | Field f -> add_json_string json f)
    (Nx.Ptree.Path.segments id);
  Buffer.add_char json ']';
  Buffer.add_string b " data-id=\"";
  add_text b (Buffer.contents json);
  Buffer.add_char b '"'

let add_rows b a =
  if Array.length a > 0 then begin
    Buffer.add_string b " data-rows=\"";
    Array.iteri
      (fun i r ->
        if i > 0 then Buffer.add_char b ' ';
        Buffer.add_string b (string_of_int r))
      a;
    Buffer.add_char b '"'
  end

let add_exacts b name vs =
  Printf.bprintf b " %s=\"" name;
  List.iteri
    (fun i v ->
      if i > 0 then Buffer.add_char b ' ';
      Vector.add_exact b v)
    vs;
  Buffer.add_char b '"'

(* Nodes *)

let clip doc b (ctx : Vector.ctx) rule path k =
  let clip = Buffer.create 64 in
  Printf.bprintf clip "<path d=\"%s\"%s/>" (area_data ctx ctx.m path)
    (if rule = `Even_odd then " clip-rule=\"evenodd\"" else "");
  let content = Buffer.contents clip in
  let id = id "c" content in
  define doc id
    (Printf.sprintf "<clipPath id=\"%s\">%s</clipPath>\n" id content);
  Printf.bprintf b "<g clip-path=\"url(#%s)\">\n" id;
  k b;
  Buffer.add_string b "</g>\n"

let opacity b _ opacity _ k =
  Buffer.add_string b "<g";
  add_attr_num b "opacity" 3 opacity;
  Buffer.add_string b ">\n";
  k b;
  Buffer.add_string b "</g>\n"

let tag b (ctx : Vector.ctx) ({ id; rows } : Picture.tag) k =
  Buffer.add_string b "<g";
  add_tag_id b id;
  (match rows with
  | Rows a -> add_rows b a
  | Cells { box; width; height } ->
      add_exacts b "data-cells"
        [
          Box2.minx box;
          Box2.miny box;
          Box2.w box;
          Box2.h box;
          float width;
          float height;
        ];
      if not (Affine.equal ctx.m Affine.id) then
        add_exacts b "data-matrix"
          [ ctx.m.xx; ctx.m.yx; ctx.m.xy; ctx.m.yy; ctx.m.x0; ctx.m.y0 ]);
  Buffer.add_string b ">\n";
  k b;
  Buffer.add_string b "</g>\n"

(* Stamps: a definition in [<defs>] that each instance [<use>]s, its pen,
   colours and row set on the [<use>]. *)

let define doc _ _ k =
  let body = Buffer.create 256 in
  k body;
  let content = Buffer.contents body in
  let def = id "s" content in
  define doc def (Printf.sprintf "<g id=\"%s\">\n%s</g>\n" def content);
  def

let add_row b (st : Vector.stamp) i =
  match st.rows with
  | None -> ()
  | Some a -> Printf.bprintf b " data-row=\"%d\"" a.(i)

let use b (ctx : Vector.ctx) (st : Vector.stamp) def i at s =
  let x = P2.x at and y = P2.y at in
  Printf.bprintf b "<use href=\"#%s\"" def;
  if not st.scaled then begin
    add_attr_num b "x" ctx.d x;
    add_attr_num b "y" ctx.d y
  end
  else begin
    Buffer.add_string b " transform=\"translate(";
    add_num b ctx.d x;
    Buffer.add_char b ' ';
    add_num b ctx.d y;
    Buffer.add_string b ") scale(";
    Vector.add_exact b s;
    Buffer.add_string b ")\"";
    match st.pens with
    | [ pen ] ->
        add_pen b
          (Vector.scale_decimals ctx s)
          (Vector.scale_pen (1. /. s) pen)
          ~offset:0.
    | _ -> ()
  end;
  Option.iter (fun a -> add_color b "fill" ~set:ctx.fill_set a.(i)) st.fills;
  Option.iter
    (fun a -> add_color b "stroke" ~set:ctx.stroke_set a.(i))
    st.strokes;
  add_row b st i;
  Buffer.add_string b "/>\n"

let instance b (st : Vector.stamp) i k =
  match st.rows with
  | None -> k b
  | Some _ ->
      Buffer.add_string b "<g";
      add_row b st i;
      Buffer.add_string b ">\n";
      k b;
      Buffer.add_string b "</g>\n"

let target doc =
  {
    Vector.fill;
    stroke;
    glyphs = glyphs doc;
    image;
    clip = clip doc;
    opacity;
    tag;
    carry = (fun _ _ s -> s);
    define = define doc;
    use;
    instance;
  }

(* Documents *)

let render r =
  let w = Renderable.w r and h = Renderable.h r in
  let doc =
    { defs = Buffer.create 1024; defined = Hashtbl.create 16; families = [] }
  in
  let body = Buffer.create 4096 in
  Vector.walk (target doc) (Vector.page r) body (Renderable.picture r);
  let out = Buffer.create (Buffer.length body + Buffer.length doc.defs + 256) in
  Buffer.add_string out
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
     <svg xmlns=\"http://www.w3.org/2000/svg\" width=\"";
  Vector.add_exact out w;
  Buffer.add_string out "pt\" height=\"";
  Vector.add_exact out h;
  Buffer.add_string out "pt\" viewBox=\"0 0 ";
  Vector.add_exact out w;
  Buffer.add_char out ' ';
  Vector.add_exact out h;
  Buffer.add_string out "\">\n";
  if Buffer.length doc.defs > 0 then begin
    Buffer.add_string out "<defs>\n";
    Buffer.add_buffer out doc.defs;
    Buffer.add_string out "</defs>\n"
  end;
  Buffer.add_buffer out body;
  Buffer.add_string out "</svg>\n";
  Buffer.contents out
