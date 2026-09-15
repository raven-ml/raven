(*---------------------------------------------------------------------------
  Tests for the canvas SVG renderer.
  ---------------------------------------------------------------------------*)

open Windtrap
open Hugin_canvas

let red = Color.v 1. 0. 0.
let render p = Svg.render ~width:100. ~height:50. p

(* Non-overlapping occurrences of [sub] in [s]. *)
let count ~sub s =
  let n = String.length sub in
  let rec go i acc =
    if i + n > String.length s then acc
    else if String.sub s i n = sub then go (i + n) (acc + 1)
    else go (i + 1) acc
  in
  go 0 0

let test_document () =
  let svg = render Picture.empty in
  starts_with ~affix:"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<svg " svg;
  contains ~sub:"width=\"100\" height=\"50\" viewBox=\"0 0 100 50\"" svg;
  ends_with ~affix:"</svg>\n" svg;
  not_contains ~msg:"no defs without content" ~sub:"<defs>" svg

let test_fill () =
  let svg = render (Picture.fill red (Path.rect 1. 2. 3. 4.5)) in
  contains ~sub:"<path d=\"M1 2 L4 2 L4 6.5 L1 6.5 Z\" fill=\"rgb(255,0,0)\"/>"
    svg;
  let svg =
    render
      (Picture.fill ~rule:`Evenodd (Color.v ~a:0.5 0. 0. 1.)
         (Path.rect 0. 0. 1. 1.))
  in
  contains
    ~sub:"fill=\"rgb(0,0,255)\" fill-opacity=\"0.5\" fill-rule=\"evenodd\"" svg

let test_curves_and_numbers () =
  let p =
    Path.empty |> Path.move_to 0.126 0. |> Path.curve_to 1. 1. 2. 2. 3. 3.
  in
  let svg = render (Picture.fill red p) in
  contains ~msg:"two decimals, trailing zeros dropped"
    ~sub:"M0.13 0 C1 1 2 2 3 3" svg;
  let svg = render (Picture.fill red (Path.rect (-1.5) 1e6 2. 2.)) in
  contains ~sub:"M-1.5 1000000" svg

let test_stroke () =
  let s =
    Stroke.v ~cap:`Square ~join:`Miter ~dash:[| 4.; 2. |] ~miter_limit:3. 1.5
  in
  let svg =
    render (Picture.stroke s red (Path.polyline [| 0.; 10. |] [| 0.; 5. |]))
  in
  contains ~sub:"<path d=\"M0 0 L10 5\" fill=\"none\" stroke=\"rgb(255,0,0)\""
    svg;
  contains ~sub:"stroke-width=\"1.5\"" svg;
  contains
    ~sub:
      "stroke-linecap=\"square\" stroke-linejoin=\"miter\" \
       stroke-miterlimit=\"3\""
    svg;
  contains ~sub:"stroke-dasharray=\"4 2\"" svg;
  let svg = render (Picture.stroke (Stroke.v 1.) red (Path.rect 0. 0. 1. 1.)) in
  contains ~sub:"stroke-linecap=\"round\" stroke-linejoin=\"round\"/>" svg;
  not_contains ~sub:"miterlimit" svg;
  not_contains ~sub:"dasharray" svg

let test_text_embeds_font () =
  let svg =
    render (Picture.text Font.bold ~size:12. red ~x:3. ~y:20. "a < b & \"c\"")
  in
  contains
    ~sub:
      "<text x=\"3\" y=\"20\" font-family=\"Inter\" font-size=\"12\" \
       font-weight=\"700\" fill=\"rgb(255,0,0)\">a &lt; b &amp; \
       &quot;c&quot;</text>"
    svg;
  contains
    ~sub:
      "<defs>\n\
       <style>@font-face{font-family:\"Inter\";font-weight:700;src:url(data:font/ttf;base64,"
    svg;
  not_contains ~msg:"only the used face is embedded" ~sub:"font-weight:400;src"
    svg;
  (* Using a face twice embeds it once. *)
  let svg =
    render
      (Picture.group
         [
           Picture.text Font.regular ~size:12. red ~x:0. ~y:0. "a";
           Picture.text Font.regular ~size:12. red ~x:0. ~y:0. "b";
         ])
  in
  equal ~msg:"one font-face rule" int 1 (count ~sub:"@font-face" svg)

let test_image () =
  let data = Nx.create Nx.uint8 [| 1; 2; 3 |] [| 255; 0; 0; 0; 0; 255 |] in
  let svg = render (Picture.image ~x:1. ~y:2. ~w:20. ~h:10. data) in
  contains
    ~sub:
      "<image x=\"1\" y=\"2\" width=\"20\" height=\"10\" \
       preserveAspectRatio=\"none\" style=\"image-rendering:pixelated\" \
       href=\"data:image/png;base64,iVBORw0KGgo"
    svg

let test_clip_transform_stamp () =
  let inner = Picture.fill red (Path.rect 0. 0. 1. 1.) in
  let svg = render (Picture.clip (Path.rect 0. 0. 10. 10.) inner) in
  contains
    ~sub:
      "<defs>\n\
       <clipPath id=\"clip1\"><path d=\"M0 0 L10 0 L10 10 L0 10 \
       Z\"/></clipPath>"
    svg;
  contains
    ~sub:
      "<g clip-path=\"url(#clip1)\">\n\
       <path d=\"M0 0 L1 0 L1 1 L0 1 Z\" fill=\"rgb(255,0,0)\"/>\n\
       </g>"
    svg;
  let svg =
    render (Picture.transform Affine.(translate 5. 6. * scale 2. 3.) inner)
  in
  contains ~sub:"<g transform=\"matrix(2 0 0 3 5 6)\">" svg;
  let svg = render (Picture.stamp inner [| 1.; nan; 3. |] [| 2.; 2.; 4. |]) in
  contains
    ~sub:
      "<g id=\"stamp1\">\n\
       <path d=\"M0 0 L1 0 L1 1 L0 1 Z\" fill=\"rgb(255,0,0)\"/>\n\
       </g>"
    svg;
  contains
    ~sub:
      "<use href=\"#stamp1\" x=\"1\" y=\"2\"/>\n\
       <use href=\"#stamp1\" x=\"3\" y=\"4\"/>"
    svg;
  equal ~msg:"non-finite positions are skipped" int 2 (count ~sub:"<use " svg)

let () =
  run "Canvas svg"
    [
      test "document" test_document;
      test "fill" test_fill;
      test "curves and numbers" test_curves_and_numbers;
      test "stroke" test_stroke;
      test "text embeds font" test_text_embeds_font;
      test "image" test_image;
      test "clip, transform and stamp" test_clip_transform_stamp;
    ]
