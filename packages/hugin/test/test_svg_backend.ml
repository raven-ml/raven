(*---------------------------------------------------------------------------
  Tests for the SVG backend — rendered through Hugin.render_svg. We verify SVG
  structure, XML escaping, and content correctness.
  ---------------------------------------------------------------------------*)

open Hugin
open Windtrap

let contains s sub =
  let len_s = String.length s and len_sub = String.length sub in
  if len_sub > len_s then false
  else
    let found = ref false in
    for i = 0 to len_s - len_sub do
      if (not !found) && String.sub s i len_sub = sub then found := true
    done;
    !found

let ends_with s suffix = String.ends_with ~suffix s

let render ?(width = 400.) ?(height = 300.) spec =
  let tmp = Filename.temp_file "hugin_test" ".svg" in
  Hugin.render_svg ~width ~height tmp spec;
  let ic = open_in tmp in
  let n = in_channel_length ic in
  let s = really_input_string ic n in
  close_in ic;
  Sys.remove tmp;
  s

let sample_x = Nx.init Float32 [| 5 |] (fun i -> float_of_int i.(0))
let sample_y = Nx.init Float32 [| 5 |] (fun i -> float_of_int i.(0))

(* SVG structure *)

let test_svg_envelope () =
  let svg = render (Hugin.line ~x:sample_x ~y:sample_y ()) in
  is_true ~msg:"starts with xml"
    (String.length svg > 5 && String.sub svg 0 5 = "<?xml");
  is_true ~msg:"ends with svg" (ends_with svg "</svg>\n")

let test_svg_dimensions () =
  let svg =
    render ~width:800. ~height:600. (Hugin.line ~x:sample_x ~y:sample_y ())
  in
  is_true ~msg:"has width" (contains svg "width=\"800\"");
  is_true ~msg:"has height" (contains svg "height=\"600\"")

(* XML escaping through text marks *)

let test_xml_escaping () =
  let svg = render (Hugin.text ~x:1. ~y:1. "a & b < c" ()) in
  is_true ~msg:"ampersand escaped" (contains svg "&amp;");
  is_true ~msg:"less-than escaped" (contains svg "&lt;")

let test_xml_escaping_quotes () =
  let svg = render (Hugin.text ~x:1. ~y:1. "say \"hello\"" ()) in
  is_true ~msg:"quotes escaped" (contains svg "&quot;")

(* Clip regions *)

let test_clip_region () =
  (* A line plot should produce a clip region for the data area *)
  let svg = render (Hugin.line ~x:sample_x ~y:sample_y ()) in
  is_true ~msg:"has clipPath" (contains svg "<clipPath")

(* Dash patterns *)

let test_dashed_line () =
  let svg =
    render (Hugin.line ~x:sample_x ~y:sample_y ~line_style:`Dashed ())
  in
  is_true ~msg:"has dasharray" (contains svg "stroke-dasharray")

let test_dotted_line () =
  let svg =
    render (Hugin.line ~x:sample_x ~y:sample_y ~line_style:`Dotted ())
  in
  is_true ~msg:"has dasharray" (contains svg "stroke-dasharray")

(* Marker rendering *)

let test_markers_in_svg () =
  let svg = render (Hugin.line ~x:sample_x ~y:sample_y ~marker:Circle ()) in
  (* Markers use <defs><symbol>...<use> pattern *)
  is_true ~msg:"has use elements" (contains svg "<use ")

let test_scatter_markers () =
  let svg = render (Hugin.point ~x:sample_x ~y:sample_y ()) in
  is_true ~msg:"has marker paths" (contains svg "<path d=\"M")

let () =
  run "Svg_backend"
    [
      group "SVG structure"
        [
          test "XML envelope" test_svg_envelope;
          test "dimensions" test_svg_dimensions;
          test "clip region" test_clip_region;
        ];
      group "XML escaping"
        [
          test "ampersand and less-than" test_xml_escaping;
          test "quotes" test_xml_escaping_quotes;
        ];
      group "line styles"
        [ test "dashed" test_dashed_line; test "dotted" test_dotted_line ];
      group "markers"
        [
          test "line markers" test_markers_in_svg;
          test "scatter markers" test_scatter_markers;
        ];
    ]
