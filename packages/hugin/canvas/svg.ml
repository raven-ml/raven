(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Numbers and strings *)

(* Two decimals is a hundredth of a pixel; trailing zeros are dropped. *)
let num v =
  if Float.is_integer v && Float.abs v < 1e15 then
    Printf.sprintf "%d" (int_of_float v)
  else begin
    let s = Printf.sprintf "%.2f" v in
    let n = ref (String.length s) in
    while !n > 0 && s.[!n - 1] = '0' do
      decr n
    done;
    if !n > 0 && s.[!n - 1] = '.' then decr n;
    String.sub s 0 !n
  end

let escape s =
  let b = Buffer.create (String.length s) in
  String.iter
    (function
      | '<' -> Buffer.add_string b "&lt;"
      | '>' -> Buffer.add_string b "&gt;"
      | '&' -> Buffer.add_string b "&amp;"
      | '"' -> Buffer.add_string b "&quot;"
      | c -> Buffer.add_char b c)
    s;
  Buffer.contents b

let base64_alphabet =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

let base64 s =
  let len = String.length s in
  let out = Buffer.create ((len + 2) / 3 * 4) in
  let byte i = if i < len then Char.code (String.unsafe_get s i) else 0 in
  let i = ref 0 in
  while !i < len do
    let b0 = byte !i and b1 = byte (!i + 1) and b2 = byte (!i + 2) in
    let c k = Buffer.add_char out base64_alphabet.[k] in
    c (b0 lsr 2);
    c (((b0 land 3) lsl 4) lor (b1 lsr 4));
    if !i + 1 < len then c (((b1 land 0xf) lsl 2) lor (b2 lsr 6))
    else Buffer.add_char out '=';
    if !i + 2 < len then c (b2 land 0x3f) else Buffer.add_char out '=';
    i := !i + 3
  done;
  Buffer.contents out

(* Attributes *)

let rgb (c : Color.t) =
  let ch v =
    int_of_float (Float.round (Float.min 1. (Float.max 0. v) *. 255.))
  in
  Printf.sprintf "rgb(%d,%d,%d)" (ch c.r) (ch c.g) (ch c.b)

let paint b attr (c : Color.t) =
  Printf.bprintf b " %s=\"%s\"" attr (rgb c);
  if c.a < 1. then
    Printf.bprintf b " %s-opacity=\"%s\"" attr (num (Float.max 0. c.a))

let path_data p =
  let b = Buffer.create 256 in
  let sep () = if Buffer.length b > 0 then Buffer.add_char b ' ' in
  Path.fold
    ~move:(fun () x y ->
      sep ();
      Printf.bprintf b "M%s %s" (num x) (num y))
    ~line:(fun () x y ->
      sep ();
      Printf.bprintf b "L%s %s" (num x) (num y))
    ~curve:(fun () c1x c1y c2x c2y x y ->
      sep ();
      Printf.bprintf b "C%s %s %s %s %s %s" (num c1x) (num c1y) (num c2x)
        (num c2y) (num x) (num y))
    ~close:(fun () ->
      sep ();
      Buffer.add_char b 'Z')
    () p;
  Buffer.contents b

let matrix (m : Affine.t) =
  Printf.sprintf "matrix(%s %s %s %s %s %s)" (num m.xx) (num m.yx) (num m.xy)
    (num m.yy) (num m.x0) (num m.y0)

(* Rendering *)

type state = {
  defs : Buffer.t;
  mutable next_id : int;
  mutable fonts : Font.t list;  (** Faces embedded so far. *)
}

let fresh st prefix =
  st.next_id <- st.next_id + 1;
  Printf.sprintf "%s%d" prefix st.next_id

let embed_font st font =
  if not (List.memq font st.fonts) then begin
    st.fonts <- font :: st.fonts;
    Printf.bprintf st.defs
      "<style>@font-face{font-family:\"%s\";font-weight:%d;src:url(data:font/ttf;base64,%s)}</style>\n"
      (escape (Font.family font))
      (Font.weight font)
      (base64 (Font.bytes font))
  end

let rec picture st b (p : Picture.t) =
  match p with
  | Empty -> ()
  | Fill { rule; color; path } ->
      Printf.bprintf b "<path d=\"%s\"" (path_data path);
      paint b "fill" color;
      if rule = `Evenodd then Buffer.add_string b " fill-rule=\"evenodd\"";
      Buffer.add_string b "/>\n"
  | Stroke { stroke; color; path } ->
      Printf.bprintf b "<path d=\"%s\" fill=\"none\"" (path_data path);
      paint b "stroke" color;
      Printf.bprintf b " stroke-width=\"%s\"" (num stroke.width);
      Printf.bprintf b " stroke-linecap=\"%s\""
        (match stroke.cap with
        | `Butt -> "butt"
        | `Round -> "round"
        | `Square -> "square");
      Printf.bprintf b " stroke-linejoin=\"%s\""
        (match stroke.join with
        | `Miter -> "miter"
        | `Round -> "round"
        | `Bevel -> "bevel");
      if stroke.join = `Miter then
        Printf.bprintf b " stroke-miterlimit=\"%s\"" (num stroke.miter_limit);
      if Array.length stroke.dash > 0 then
        Printf.bprintf b " stroke-dasharray=\"%s\""
          (String.concat " " (Array.to_list (Array.map num stroke.dash)));
      Buffer.add_string b "/>\n"
  | Text { font; size; color; x; y; text } ->
      embed_font st font;
      Printf.bprintf b
        "<text x=\"%s\" y=\"%s\" font-family=\"%s\" font-size=\"%s\" \
         font-weight=\"%d\""
        (num x) (num y)
        (escape (Font.family font))
        (num size) (Font.weight font);
      paint b "fill" color;
      Printf.bprintf b ">%s</text>\n" (escape text)
  | Image { x; y; w; h; data } ->
      Printf.bprintf b
        "<image x=\"%s\" y=\"%s\" width=\"%s\" height=\"%s\" \
         preserveAspectRatio=\"none\" style=\"image-rendering:pixelated\" \
         href=\"data:image/png;base64,%s\"/>\n"
        (num x) (num y) (num w) (num h)
        (base64 (Nx_io.encode_png data))
  | Group ps -> List.iter (picture st b) ps
  | Clip { path; picture = inner } ->
      let id = fresh st "clip" in
      Printf.bprintf st.defs "<clipPath id=\"%s\"><path d=\"%s\"/></clipPath>\n"
        id (path_data path);
      Printf.bprintf b "<g clip-path=\"url(#%s)\">\n" id;
      picture st b inner;
      Buffer.add_string b "</g>\n"
  | Transform { m; picture = inner } ->
      Printf.bprintf b "<g transform=\"%s\">\n" (matrix m);
      picture st b inner;
      Buffer.add_string b "</g>\n"
  | Stamp { picture = inner; xs; ys } ->
      let id = fresh st "stamp" in
      Printf.bprintf st.defs "<g id=\"%s\">\n" id;
      picture st st.defs inner;
      Buffer.add_string st.defs "</g>\n";
      Array.iteri
        (fun i x ->
          let y = ys.(i) in
          if Float.is_finite x && Float.is_finite y then
            Printf.bprintf b "<use href=\"#%s\" x=\"%s\" y=\"%s\"/>\n" id
              (num x) (num y))
        xs

let render ~width ~height p =
  let st = { defs = Buffer.create 1024; next_id = 0; fonts = [] } in
  let body = Buffer.create 4096 in
  picture st body p;
  let out = Buffer.create (Buffer.length body + Buffer.length st.defs + 256) in
  Printf.bprintf out
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
     <svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%s\" height=\"%s\" \
     viewBox=\"0 0 %s %s\">\n"
    (num width) (num height) (num width) (num height);
  if Buffer.length st.defs > 0 then begin
    Buffer.add_string out "<defs>\n";
    Buffer.add_buffer out st.defs;
    Buffer.add_string out "</defs>\n"
  end;
  Buffer.add_buffer out body;
  Buffer.add_string out "</svg>\n";
  Buffer.contents out
