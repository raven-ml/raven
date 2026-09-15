(*---------------------------------------------------------------------------
  Tests for the vg PDF renderer.
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_vg

let red = Color.v 1. 0. 0.
let render p = Hugin_vg_pdf.render ~width:100. ~height:50. p

(* Non-overlapping occurrences of [sub] in [s]. *)
let count ~sub s =
  let n = String.length sub in
  let rec go i acc =
    if i + n > String.length s then acc
    else if String.sub s i n = sub then go (i + n) (acc + 1)
    else go (i + 1) acc
  in
  go 0 0

(* The cross-reference table must point every object at its header. *)
let check_xref pdf =
  let startxref =
    ignore
      (Str.search_backward
         (Str.regexp "startxref\n\\([0-9]+\\)")
         pdf
         (String.length pdf - 1));
    int_of_string (Str.matched_group 1 pdf)
  in
  starts_with ~msg:"startxref points at the table" ~affix:"xref\n0 "
    (String.sub pdf startxref 7);
  let count =
    Scanf.sscanf (String.sub pdf startxref 20) "xref\n0 %d" (fun n -> n)
  in
  let header =
    startxref + String.length (Printf.sprintf "xref\n0 %d\n" count)
  in
  for i = 1 to count - 1 do
    let entry = String.sub pdf (header + (20 * i)) 20 in
    let off = int_of_string (String.sub entry 0 10) in
    starts_with
      ~msg:(Printf.sprintf "object %d offset" i)
      ~affix:(Printf.sprintf "%d 0 obj\n" i)
      (String.sub pdf off 16)
  done

let test_document () =
  let pdf = render Picture.empty in
  starts_with ~affix:"%PDF-1.7\n" pdf;
  ends_with ~affix:"%%EOF\n" pdf;
  contains ~sub:"/Type /Catalog" pdf;
  contains ~sub:"/MediaBox [0 0 100 50]" pdf;
  contains ~msg:"canvas coordinates are flipped once" ~sub:"1 0 0 -1 0 50 cm"
    pdf;
  check_xref pdf

let test_fill_and_stroke () =
  let pdf = render (Picture.fill red (Path.rect 1. 2. 3. 4.)) in
  contains ~sub:"1 0 0 rg\n1 2 m\n4 2 l\n4 6 l\n1 6 l\nh\nf\n" pdf;
  let pdf = render (Picture.fill ~rule:`Evenodd red (Path.rect 0. 0. 1. 1.)) in
  contains ~sub:"h\nf*\n" pdf;
  let s =
    Stroke.v ~cap:`Square ~join:`Bevel ~dash:[| 3.; 1. |] ~miter_limit:2. 1.5
  in
  let pdf =
    render (Picture.stroke s red (Path.polyline [| 0.; 10. |] [| 0.; 5. |]))
  in
  contains ~sub:"1 0 0 RG\n1.5 w 2 J 2 j 2 M\n[3 1] 0 d\n0 0 m\n10 5 l\nS\n" pdf

let test_alpha_uses_ext_gstate () =
  let pdf =
    render (Picture.fill (Color.v ~a:0.25 1. 0. 0.) (Path.rect 0. 0. 1. 1.))
  in
  contains ~sub:"q /GS1 gs\n1 0 0 rg\n" pdf;
  contains ~sub:"/GS1 << /Type /ExtGState /ca 0.25 /CA 0.25 >>" pdf;
  let pdf =
    render
      (Picture.group
         [
           Picture.fill (Color.v ~a:0.5 1. 0. 0.) (Path.rect 0. 0. 1. 1.);
           Picture.fill (Color.v ~a:0.5 0. 1. 0.) (Path.rect 0. 0. 1. 1.);
         ])
  in
  equal ~msg:"one state per alpha" int 1 (count ~sub:"/Type /ExtGState" pdf)

let test_text_embeds_font () =
  let pdf = render (Picture.text Font.bold ~size:12. red ~x:3. ~y:20. "AV") in
  contains ~sub:"BT /F1 12 Tf 1 0 0 -1 3 20 Tm [ <" pdf;
  contains ~msg:"kerning as a TJ adjustment" ~sub:"> " pdf;
  contains ~sub:"/Subtype /Type0 /BaseFont /Inter-Bold /Encoding /Identity-H"
    pdf;
  contains ~sub:"/Subtype /CIDFontType2" pdf;
  contains ~sub:"/CIDToGIDMap /Identity" pdf;
  contains ~sub:"/FontFile2" pdf;
  contains ~sub:"/Length1 " pdf;
  contains ~sub:"beginbfchar" pdf;
  equal ~msg:"one embedded font file" int 1 (count ~sub:"/Length1 " pdf);
  (* The glyph run for AV is two glyph ids with a negative kern between. *)
  let tj =
    Str.regexp "\\[ <\\([0-9a-f]+\\)> \\(-?[0-9.]+\\) <\\([0-9a-f]+\\)> \\] TJ"
  in
  is_true ~msg:"two glyphs with an adjustment"
    (try
       ignore (Str.search_forward tj pdf 0);
       float_of_string (Str.matched_group 2 pdf) > 0.
     with Not_found -> false);
  check_xref pdf

let test_image () =
  let data = Nx.create Nx.uint8 [| 1; 2; 3 |] [| 255; 0; 0; 0; 0; 255 |] in
  let pdf = render (Picture.image ~x:1. ~y:2. ~w:20. ~h:10. data) in
  contains ~sub:"q 20 0 0 -10 1 12 cm /Im1 Do Q" pdf;
  contains
    ~sub:
      "/Subtype /Image /Width 2 /Height 1 /ColorSpace /DeviceRGB \
       /BitsPerComponent 8 /Filter /FlateDecode /DecodeParms << /Predictor 15 \
       /Colors 3 /BitsPerComponent 8 /Columns 2 >>"
    pdf;
  let rgba = Nx.create Nx.uint8 [| 1; 1; 4 |] [| 255; 0; 0; 128 |] in
  let pdf = render (Picture.image ~x:0. ~y:0. ~w:1. ~h:1. rgba) in
  contains ~sub:"/ColorSpace /DeviceGray" pdf;
  contains ~sub:"/SMask " pdf;
  let gray = Nx.create Nx.uint8 [| 1; 1 |] [| 7 |] in
  let pdf = render (Picture.image ~x:0. ~y:0. ~w:1. ~h:1. gray) in
  contains ~sub:"/ColorSpace /DeviceGray" pdf;
  not_contains ~sub:"/SMask" pdf;
  check_xref pdf

let test_clip_transform_stamp () =
  let inner = Picture.fill red (Path.rect 0. 0. 1. 1.) in
  let pdf = render (Picture.clip (Path.rect 0. 0. 10. 10.) inner) in
  contains ~sub:"q\n0 0 m\n10 0 l\n10 10 l\n0 10 l\nh\nW n\n1 0 0 rg\n" pdf;
  let pdf =
    render (Picture.transform Affine.(translate 5. 6. * scale 2. 3.) inner)
  in
  contains ~sub:"q\n2 0 0 3 5 6 cm\n" pdf;
  let pdf = render (Picture.stamp inner [| 1.; nan; 3. |] [| 2.; 2.; 4. |]) in
  contains ~sub:"/Subtype /Form" pdf;
  contains ~sub:"q 1 0 0 1 1 2 cm /Fm1 Do Q\nq 1 0 0 1 3 4 cm /Fm1 Do Q\n" pdf;
  equal ~msg:"non-finite positions are skipped" int 2 (count ~sub:"/Fm1 Do" pdf);
  check_xref pdf

let () =
  run "Vg pdf"
    [
      test "document" test_document;
      test "fill and stroke" test_fill_and_stroke;
      test "alpha" test_alpha_uses_ext_gstate;
      test "text embeds font" test_text_embeds_font;
      test "image" test_image;
      test "clip, transform and stamp" test_clip_transform_stamp;
    ]
