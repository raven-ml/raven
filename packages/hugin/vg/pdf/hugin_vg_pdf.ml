(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin_vg

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

(* Objects *)

type doc = {
  mutable objects : string list;  (** Bodies in reverse order of number. *)
  mutable count : int;
}

let add doc body =
  doc.count <- doc.count + 1;
  doc.objects <- body :: doc.objects;
  doc.count

(* [reserve doc] allocates a number whose body {!set} fills in later. *)
let reserve doc = add doc ""

let set doc n body =
  let i = doc.count - n in
  doc.objects <- List.mapi (fun j b -> if j = i then body else b) doc.objects

(* Streams are deflated unless their data already is. *)
let stream ?(compress = true) dict data =
  let dict, data =
    if compress then ("/Filter /FlateDecode " ^ dict, Nx_io.deflate data)
    else (dict, data)
  in
  Printf.sprintf "<< /Length %d %s>>\nstream\n%s\nendstream"
    (String.length data) dict data

(* Resources gathered while walking the picture. *)
type font_use = {
  font : Font.t;
  name : string;
  number : int;
  glyphs : (int, int * float) Hashtbl.t;
      (** Used glyph ids to their code point and advance per 1000 em. *)
}

type state = {
  doc : doc;
  resources : int;  (** Object number of the shared resource dictionary. *)
  mutable fonts : font_use list;
  mutable xobjects : (string * int) list;
  mutable alphas : (float * string) list;
  mutable next : int;
}

let fresh st prefix =
  st.next <- st.next + 1;
  Printf.sprintf "%s%d" prefix st.next

let font_use st font =
  match List.find_opt (fun u -> u.font == font) st.fonts with
  | Some u -> u
  | None ->
      let u =
        {
          font;
          name = fresh st "F";
          number = reserve st.doc;
          glyphs = Hashtbl.create 64;
        }
      in
      st.fonts <- u :: st.fonts;
      u

let alpha_state st a =
  let a = Float.round (a *. 1000.) /. 1000. in
  match List.assoc_opt a st.alphas with
  | Some name -> name
  | None ->
      let name = fresh st "GS" in
      st.alphas <- (a, name) :: st.alphas;
      name

(* Content operators *)

let color_op b (c : Color.t) op =
  let ch v = num (Float.min 1. (Float.max 0. v)) in
  Printf.bprintf b "%s %s %s %s\n" (ch c.r) (ch c.g) (ch c.b) op

(* [with_alpha st b c k] runs [k] with the constant alpha of [c] applied. *)
let with_alpha st b (c : Color.t) k =
  if c.a < 1. then begin
    Printf.bprintf b "q /%s gs\n" (alpha_state st (Float.max 0. c.a));
    k ();
    Buffer.add_string b "Q\n"
  end
  else k ()

let path_ops b p =
  Path.fold
    ~move:(fun () x y -> Printf.bprintf b "%s %s m\n" (num x) (num y))
    ~line:(fun () x y -> Printf.bprintf b "%s %s l\n" (num x) (num y))
    ~curve:(fun () c1x c1y c2x c2y x y ->
      Printf.bprintf b "%s %s %s %s %s %s c\n" (num c1x) (num c1y) (num c2x)
        (num c2y) (num x) (num y))
    ~close:(fun () -> Buffer.add_string b "h\n")
    () p

let matrix_op b (m : Affine.t) =
  Printf.bprintf b "%s %s %s %s %s %s cm\n" (num m.xx) (num m.yx) (num m.xy)
    (num m.yy) (num m.x0) (num m.y0)

(* Each glyph is shown by id; the TJ adjustments reproduce kerning. *)
let text_ops st b ~font ~size ~x ~y text =
  let u = font_use st font in
  let glyphs = Font.glyphs font ~size text in
  let codes =
    let rec go i acc =
      if i >= String.length text then List.rev acc
      else begin
        let d = String.get_utf_8_uchar text i in
        go (i + Uchar.utf_decode_length d) (Uchar.utf_decode_uchar d :: acc)
      end
    in
    go 0 []
  in
  List.iter2
    (fun (g : Font.glyph) u' ->
      Hashtbl.replace u.glyphs g.id (Uchar.to_int u', g.advance *. 1000. /. size))
    glyphs codes;
  Printf.bprintf b "BT /%s %s Tf 1 0 0 -1 %s %s Tm [" u.name (num size) (num x)
    (num y);
  let pen = ref 0. in
  List.iter
    (fun (g : Font.glyph) ->
      let shift = g.x -. !pen in
      if Float.abs shift > 1e-3 then
        Printf.bprintf b " %s" (num (-.shift *. 1000. /. size));
      Printf.bprintf b " <%04x>" g.id;
      pen := g.x +. g.advance)
    glyphs;
  Buffer.add_string b " ] TJ ET\n"

(* Images *)

(* [idat png] is the concatenated IDAT payload of [png]: a zlib stream of
   PNG-predicted scanlines, which PDF decodes with predictor 15. *)
let idat png =
  let b = Buffer.create (String.length png) in
  let pos = ref 8 in
  while !pos + 8 <= String.length png do
    let len =
      (Char.code png.[!pos] lsl 24)
      lor (Char.code png.[!pos + 1] lsl 16)
      lor (Char.code png.[!pos + 2] lsl 8)
      lor Char.code png.[!pos + 3]
    in
    if String.sub png (!pos + 4) 4 = "IDAT" then
      Buffer.add_string b (String.sub png (!pos + 8) len);
    pos := !pos + 12 + len
  done;
  Buffer.contents b

let image_object st ?smask data ~rows ~cols ~channels =
  let colorspace = if channels = 1 then "/DeviceGray" else "/DeviceRGB" in
  let dict =
    Printf.sprintf
      "/Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace %s \
       /BitsPerComponent 8 /Filter /FlateDecode /DecodeParms << /Predictor 15 \
       /Colors %d /BitsPerComponent 8 /Columns %d >> %s"
      cols rows colorspace channels cols
      (match smask with
      | Some n -> Printf.sprintf "/SMask %d 0 R " n
      | None -> "")
  in
  add st.doc (stream ~compress:false dict (idat (Nx_io.encode_png data)))

(* [split_alpha data] is the RGB and alpha planes of an RGBA image. *)
let split_alpha data ~rows ~cols =
  let data = Nx.contiguous data in
  let buf = Nx.data data and off = Nx.offset data in
  let rgb =
    Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout
      (rows * cols * 3)
  in
  let alpha =
    Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout (rows * cols)
  in
  for i = 0 to (rows * cols) - 1 do
    Bigarray.Array1.unsafe_set rgb (3 * i)
      (Nx_buffer.unsafe_get buf (off + (4 * i)));
    Bigarray.Array1.unsafe_set rgb
      ((3 * i) + 1)
      (Nx_buffer.unsafe_get buf (off + (4 * i) + 1));
    Bigarray.Array1.unsafe_set rgb
      ((3 * i) + 2)
      (Nx_buffer.unsafe_get buf (off + (4 * i) + 2));
    Bigarray.Array1.unsafe_set alpha i
      (Nx_buffer.unsafe_get buf (off + (4 * i) + 3))
  done;
  let tensor a shape =
    Nx.of_bigarray (Bigarray.reshape (Bigarray.genarray_of_array1 a) shape)
  in
  (tensor rgb [| rows; cols; 3 |], tensor alpha [| rows; cols |])

let image_xobject st data =
  let shape = Nx.shape data in
  let rows = shape.(0) and cols = shape.(1) in
  let channels = if Array.length shape = 3 then shape.(2) else 1 in
  let number =
    if channels = 4 then begin
      let rgb, alpha = split_alpha data ~rows ~cols in
      let smask = image_object st alpha ~rows ~cols ~channels:1 in
      image_object st ~smask rgb ~rows ~cols ~channels:3
    end
    else image_object st data ~rows ~cols ~channels
  in
  let name = fresh st "Im" in
  st.xobjects <- (name, number) :: st.xobjects;
  name

(* Pictures *)

let rec picture st b (p : Picture.t) =
  match p with
  | Empty -> ()
  | Fill { rule; color; path } ->
      with_alpha st b color (fun () ->
          color_op b color "rg";
          path_ops b path;
          Buffer.add_string b (if rule = `Evenodd then "f*\n" else "f\n"))
  | Stroke { stroke; color; path } ->
      with_alpha st b color (fun () ->
          Buffer.add_string b "q\n";
          color_op b color "RG";
          Printf.bprintf b "%s w %d J %d j %s M\n" (num stroke.width)
            (match stroke.cap with `Butt -> 0 | `Round -> 1 | `Square -> 2)
            (match stroke.join with `Miter -> 0 | `Round -> 1 | `Bevel -> 2)
            (num stroke.miter_limit);
          if Array.length stroke.dash > 0 then
            Printf.bprintf b "[%s] 0 d\n"
              (String.concat " " (Array.to_list (Array.map num stroke.dash)));
          path_ops b path;
          Buffer.add_string b "S\nQ\n")
  | Text { font; size; color; x; y; text } ->
      with_alpha st b color (fun () ->
          color_op b color "rg";
          text_ops st b ~font ~size ~x ~y text)
  | Image { x; y; w; h; data } ->
      let name = image_xobject st data in
      Printf.bprintf b "q %s 0 0 %s %s %s cm /%s Do Q\n" (num w) (num (-.h))
        (num x)
        (num (y +. h))
        name
  | Group ps -> List.iter (picture st b) ps
  | Clip { path; picture = inner } ->
      Buffer.add_string b "q\n";
      path_ops b path;
      Buffer.add_string b "W n\n";
      picture st b inner;
      Buffer.add_string b "Q\n"
  | Transform { m; picture = inner } ->
      Buffer.add_string b "q\n";
      matrix_op b m;
      picture st b inner;
      Buffer.add_string b "Q\n"
  | Stamp { picture = inner; xs; ys } ->
      let body = Buffer.create 256 in
      picture st body inner;
      let dict =
        Printf.sprintf
          "/Type /XObject /Subtype /Form /BBox [-1000000 -1000000 1000000 \
           1000000] /Resources %d 0 R"
          st.resources
      in
      let number = add st.doc (stream dict (Buffer.contents body)) in
      let name = fresh st "Fm" in
      st.xobjects <- (name, number) :: st.xobjects;
      Array.iteri
        (fun i x ->
          let y = ys.(i) in
          if Float.is_finite x && Float.is_finite y then
            Printf.bprintf b "q 1 0 0 1 %s %s cm /%s Do Q\n" (num x) (num y)
              name)
        xs

(* Fonts *)

let utf16be_hex u =
  let b = Buffer.create 8 in
  let add16 v = Printf.bprintf b "%04x" v in
  if u < 0x10000 then add16 u
  else begin
    let v = u - 0x10000 in
    add16 (0xD800 lor (v lsr 10));
    add16 (0xDC00 lor (v land 0x3FF))
  end;
  Buffer.contents b

let font_objects st (u : font_use) =
  let font = u.font in
  let base =
    let family =
      String.concat "" (String.split_on_char ' ' (Font.family font))
    in
    let family = if family = "" then "Font" else family in
    if Font.weight font >= 600 then family ^ "-Bold" else family
  in
  let file =
    add st.doc
      (stream
         (Printf.sprintf "/Length1 %d " (String.length (Font.bytes font)))
         (Font.bytes font))
  in
  let ascent = Font.ascent font ~size:1000.
  and descent = Font.descent font ~size:1000. in
  let descriptor =
    add st.doc
      (Printf.sprintf
         "<< /Type /FontDescriptor /FontName /%s /Flags 4 /FontBBox [-1000 %s \
          2000 %s] /ItalicAngle 0 /Ascent %s /Descent %s /CapHeight %s /StemV \
          %d /FontFile2 %d 0 R >>"
         base (num (-.descent)) (num ascent) (num ascent) (num (-.descent))
         (num (0.7 *. ascent))
         (if Font.weight font >= 600 then 140 else 80)
         file)
  in
  let gids =
    Hashtbl.fold (fun g u acc -> (g, u) :: acc) u.glyphs [] |> List.sort compare
  in
  let widths =
    String.concat " "
      (List.map
         (fun (g, (_, advance)) -> Printf.sprintf "%d [%s]" g (num advance))
         gids)
  in
  let cid =
    add st.doc
      (Printf.sprintf
         "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /%s /CIDSystemInfo \
          << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> \
          /FontDescriptor %d 0 R /DW 1000 /W [%s] /CIDToGIDMap /Identity >>"
         base descriptor widths)
  in
  let to_unicode =
    let b = Buffer.create 1024 in
    Buffer.add_string b
      "/CIDInit /ProcSet findresource begin\n\
       12 dict begin\n\
       begincmap\n\
       /CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n\
       /CMapName /Adobe-Identity-UCS def\n\
       /CMapType 2 def\n\
       1 begincodespacerange\n\
       <0000> <ffff>\n\
       endcodespacerange\n";
    let rec blocks = function
      | [] -> ()
      | l ->
          let n = Int.min 100 (List.length l) in
          Printf.bprintf b "%d beginbfchar\n" n;
          List.iteri
            (fun i (g, (u, _)) ->
              if i < n then Printf.bprintf b "<%04x> <%s>\n" g (utf16be_hex u))
            l;
          Buffer.add_string b "endbfchar\n";
          blocks (List.filteri (fun i _ -> i >= n) l)
    in
    blocks gids;
    Buffer.add_string b
      "endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend";
    add st.doc (stream "" (Buffer.contents b))
  in
  set st.doc u.number
    (Printf.sprintf
       "<< /Type /Font /Subtype /Type0 /BaseFont /%s /Encoding /Identity-H \
        /DescendantFonts [%d 0 R] /ToUnicode %d 0 R >>"
       base cid to_unicode)

(* Document *)

let render ~width ~height p =
  let doc = { objects = []; count = 0 } in
  let resources = reserve doc in
  let st =
    { doc; resources; fonts = []; xobjects = []; alphas = []; next = 0 }
  in
  let body = Buffer.create 4096 in
  (* Flip to the canvas's y-down coordinates. *)
  Printf.bprintf body "1 0 0 -1 0 %s cm\n" (num height);
  picture st body p;
  List.iter (font_objects st) st.fonts;
  let dict_of entries =
    String.concat " "
      (List.map (fun (name, n) -> Printf.sprintf "/%s %d 0 R" name n) entries)
  in
  set doc resources
    (Printf.sprintf "<< /Font << %s >> /XObject << %s >> /ExtGState << %s >> >>"
       (dict_of (List.map (fun u -> (u.name, u.number)) st.fonts))
       (dict_of st.xobjects)
       (String.concat " "
          (List.map
             (fun (a, name) ->
               Printf.sprintf "/%s << /Type /ExtGState /ca %s /CA %s >>" name
                 (num a) (num a))
             st.alphas)));
  let contents = add doc (stream "" (Buffer.contents body)) in
  let pages = reserve doc in
  let page =
    add doc
      (Printf.sprintf
         "<< /Type /Page /Parent %d 0 R /MediaBox [0 0 %s %s] /Resources %d 0 \
          R /Contents %d 0 R >>"
         pages (num width) (num height) resources contents)
  in
  set doc pages
    (Printf.sprintf "<< /Type /Pages /Kids [%d 0 R] /Count 1 >>" page);
  let catalog =
    add doc (Printf.sprintf "<< /Type /Catalog /Pages %d 0 R >>" pages)
  in
  let out = Buffer.create 65536 in
  Buffer.add_string out "%PDF-1.7\n%\xe2\xe3\xcf\xd3\n";
  let offsets =
    List.mapi
      (fun i body ->
        let off = Buffer.length out in
        Printf.bprintf out "%d 0 obj\n%s\nendobj\n" (i + 1) body;
        off)
      (List.rev doc.objects)
  in
  let xref = Buffer.length out in
  Printf.bprintf out "xref\n0 %d\n0000000000 65535 f \n" (doc.count + 1);
  List.iter (fun off -> Printf.bprintf out "%010d 00000 n \n" off) offsets;
  Printf.bprintf out
    "trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n"
    (doc.count + 1) catalog xref;
  Buffer.contents out
