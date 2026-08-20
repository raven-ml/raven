(*---------------------------------------------------------------------------
  Tests for the SVG backend — rendered through Hugin.render_svg. We verify SVG
  structure, XML escaping, and content correctness.
  ---------------------------------------------------------------------------*)

open Hugin
open Windtrap

let render ?(width = 400.) ?(height = 300.) spec =
  let tmp = temp_file ~suffix:".svg" () in
  Hugin.render_svg ~width ~height tmp spec;
  let ic = open_in tmp in
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () -> really_input_string ic (in_channel_length ic))

(* Non-overlapping occurrence count of [sub] in [s]. A count through [equal]
   reports the number it rejected, where a presence check could only say
   "false". *)
let count_substring s sub =
  let len_s = String.length s and len_sub = String.length sub in
  if len_sub = 0 || len_sub > len_s then 0
  else begin
    let count = ref 0 and i = ref 0 in
    while !i <= len_s - len_sub do
      if String.sub s !i len_sub = sub then begin
        incr count;
        i := !i + len_sub
      end
      else incr i
    done;
    !count
  end

let sample_x = Nx.init Float32 [| 5 |] (fun i -> float_of_int i.(0))
let sample_y = Nx.init Float32 [| 5 |] (fun i -> float_of_int i.(0))
let line () = Hugin.line ~x:sample_x ~y:sample_y ()

(* SVG structure *)

let test_svg_envelope () =
  let svg = render (line ()) in
  starts_with ~affix:"<?xml" svg;
  ends_with ~affix:"</svg>\n" svg

let test_svg_dimensions () =
  let svg = render ~width:800. ~height:600. (line ()) in
  contains ~sub:"width=\"800\"" svg;
  contains ~sub:"height=\"600\"" svg

(* XML escaping through text marks. The expected renderings come from the XML
   spec, not from the backend: escaping the whole string pins where each entity
   lands, where asserting that "&amp;" occurs somewhere in an 8KB document
   passes even if the two entities are swapped or the text is mangled. *)

let test_xml_escaping () =
  let svg = render (Hugin.text ~x:1. ~y:1. "a & b < c" ()) in
  contains ~sub:"a &amp; b &lt; c" svg

let test_xml_escaping_quotes () =
  let svg = render (Hugin.text ~x:1. ~y:1. "say \"hello\"" ()) in
  contains ~sub:"say &quot;hello&quot;" svg

let test_xml_no_raw_delimiters () =
  (* The escaped document must not leak a raw delimiter from the text. *)
  let svg = render (Hugin.text ~x:1. ~y:1. "a & b" ()) in
  not_contains ~sub:"a & b" svg

(* Clip regions *)

let test_clip_region () =
  (* A line plot should produce a clip region for the data area, and reference
     it — a <clipPath> nothing points at would clip nothing. *)
  let svg = render (line ()) in
  in_order ~subs:[ "<clipPath"; "clip-path=" ] svg

(* Dash patterns *)

let dasharray = "stroke-dasharray"

let test_solid_line_has_no_dasharray () =
  let svg = render (Hugin.line ~x:sample_x ~y:sample_y ~line_style:`Solid ()) in
  not_contains ~sub:dasharray svg

let test_dashed_line () =
  let svg =
    render (Hugin.line ~x:sample_x ~y:sample_y ~line_style:`Dashed ())
  in
  contains ~sub:dasharray svg

let test_dotted_line () =
  let svg =
    render (Hugin.line ~x:sample_x ~y:sample_y ~line_style:`Dotted ())
  in
  contains ~sub:dasharray svg

let test_dashed_and_dotted_differ () =
  (* Both styles set a dash array, so "has a dash array" is an oracle that
     cannot tell them apart. The distinguishing claim is that they differ. *)
  let render_style s =
    render (Hugin.line ~x:sample_x ~y:sample_y ~line_style:s ())
  in
  not_equal text (render_style `Dashed) (render_style `Dotted)

(* Marker rendering *)

let test_markers_in_svg () =
  (* Markers use <defs><symbol>...<use> — the <use> must follow the symbol it
     instantiates, which three separate contains calls would not check. *)
  let svg = render (Hugin.line ~x:sample_x ~y:sample_y ~marker:Circle ()) in
  in_order ~subs:[ "<defs"; "<symbol"; "<use " ] svg

let test_marker_count_matches_points () =
  (* One <use> per sample point: a count, not a presence check. *)
  let svg = render (Hugin.line ~x:sample_x ~y:sample_y ~marker:Circle ()) in
  equal int (Nx.numel sample_x) (count_substring svg "<use ")

let test_scatter_markers () =
  let svg = render (Hugin.point ~x:sample_x ~y:sample_y ()) in
  contains ~sub:"<path d=\"M" svg

let () =
  run "Svg_backend"
    [
      group "SVG structure"
        [
          test "XML envelope" test_svg_envelope;
          test "dimensions" test_svg_dimensions;
          test "clip region is referenced" test_clip_region;
        ];
      group "XML escaping"
        [
          test "ampersand and less-than" test_xml_escaping;
          test "quotes" test_xml_escaping_quotes;
          test "no raw delimiter survives" test_xml_no_raw_delimiters;
        ];
      group "line styles"
        [
          test "solid has no dash array" test_solid_line_has_no_dasharray;
          test "dashed" test_dashed_line;
          test "dotted" test_dotted_line;
          test "dashed and dotted differ" test_dashed_and_dotted_differ;
        ];
      group "markers"
        [
          test "line markers instantiate a symbol" test_markers_in_svg;
          test "one marker per point" test_marker_count_matches_points;
          test "scatter markers" test_scatter_markers;
        ];
    ]
