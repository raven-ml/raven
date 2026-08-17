(*---------------------------------------------------------------------------
  Tests for Ticks — exercised indirectly through SVG output. We verify tick
  label formatting, count, and presence in rendered SVGs.
  ---------------------------------------------------------------------------*)

open Hugin
open Windtrap

(* Non-overlapping occurrence count, matching windtrap's [contains ~count]. *)
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

let render spec =
  let tmp = temp_file ~suffix:".svg" () in
  Hugin.render_svg ~width:400. ~height:300. tmp spec;
  let ic = open_in tmp in
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () -> really_input_string ic (in_channel_length ic))

let vec l = Nx.create Float32 [| List.length l |] (Array.of_list l)
let labels svg = count_substring svg "<text "

(* The text content of every <text> element, i.e. the rendered tick labels.
   Scoping assertions to these keeps a needle like "e-" from matching attribute
   names such as "stroke-width". *)
let label_texts svg =
  let rec collect acc i =
    match String.index_from_opt svg i '<' with
    | None -> List.rev acc
    | Some lt ->
        if lt + 6 <= String.length svg && String.sub svg lt 6 = "<text " then
          match String.index_from_opt svg lt '>' with
          | None -> List.rev acc
          | Some gt -> (
              match String.index_from_opt svg gt '<' with
              | None -> List.rev acc
              | Some close ->
                  collect
                    (String.sub svg (gt + 1) (close - gt - 1) :: acc)
                    close)
        else collect acc (lt + 1)
  in
  collect [] 0

let scientific_labels svg =
  List.filter
    (fun l -> count_substring l "e-" > 0 || count_substring l "e+" > 0)
    (label_texts svg)

(* linear tick formatting *)

let test_zero_label () =
  let x = vec [ -10.; -5.; 0.; 5.; 10. ] in
  let svg = render (Hugin.line ~x ~y:x ()) in
  (* The zero tick should show "0" not "1e-15" or similar. Floating-point tick
     generation is exactly where a near-zero value leaks out, so name the shapes
     that must not appear. *)
  contains ~sub:">0<" svg;
  equal ~msg:"no tick is in scientific notation" (list string) []
    (scientific_labels svg);
  not_contains ~sub:"NaN" svg

let test_reasonable_count () =
  let x = Nx.init Float32 [| 101 |] (fun i -> float_of_int i.(0)) in
  let svg = render (Hugin.line ~x ~y:x ()) in
  (* Each tick generates a <text> element, so the count is x ticks + y ticks.
     Two separate bounds report the number they rejected; one boolean AND of
     both reports only "false". *)
  greater ~msg:"more than a couple of labels" int ~than:2 (labels svg);
  less ~msg:"not an absurd number of labels" int ~than:40 (labels svg)

(* log tick formatting *)

let test_log_tick_labels () =
  let x = vec [ 0.01; 0.1; 1.; 10.; 100. ] in
  let y = vec [ 1.; 2.; 3.; 4.; 5. ] in
  let svg = render (Hugin.line ~x ~y () |> Hugin.xscale `Log) in
  (* Log ticks should be powers of 10, formatted as 10^k, ascending. *)
  in_order ~subs:[ ">10^-2<"; ">10^-1<"; ">1<"; ">10<"; ">10^2<" ] svg

(* large range doesn't explode *)

let test_large_range () =
  let x = vec [ 0.; 1e6 ] in
  let y = vec [ 0.; 1. ] in
  let svg = render (Hugin.line ~x ~y ()) in
  less ~msg:"tick count bounded" int ~than:40 (labels svg);
  greater ~msg:"a million-wide axis still gets labelled" int ~than:2
    (labels svg)

(* small fractional range *)

let test_fractional_range () =
  let x = vec [ 0.; 0.001 ] in
  let y = vec [ 0.; 1. ] in
  let svg = render (Hugin.line ~x ~y ()) in
  starts_with ~affix:"<?xml" svg;
  ends_with ~affix:"</svg>\n" svg;
  (* A range this narrow is where tick stepping divides by a tiny number. *)
  not_contains ~sub:"NaN" svg;
  greater ~msg:"the axis is still labelled" int ~than:2 (labels svg)

let () =
  run "Ticks"
    [
      group "linear formatting"
        [
          test "zero label" test_zero_label;
          test "reasonable count" test_reasonable_count;
          test "large range bounded" test_large_range;
          test "fractional range" test_fractional_range;
        ];
      group "log formatting" [ test "log tick labels" test_log_tick_labels ];
    ]
