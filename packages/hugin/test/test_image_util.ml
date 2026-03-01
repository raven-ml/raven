(*---------------------------------------------------------------------------
  Tests for Image_util.base64_encode — exercised indirectly through Hugin.pp
  which calls base64_encode on PNG buffer data. We also test the base64 logic
  through the pp data URI output.
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

let is_base64_char = function
  | 'A' .. 'Z' | 'a' .. 'z' | '0' .. '9' | '+' | '/' | '=' -> true
  | _ -> false

let sample_x = Nx.init Float32 [| 5 |] (fun i -> float_of_int i.(0))
let sample_y = Nx.init Float32 [| 5 |] (fun i -> float_of_int i.(0))

(* pp produces a data URI with base64 encoded PNG *)

let test_pp_data_uri () =
  let spec = Hugin.line ~x:sample_x ~y:sample_y () in
  let buf = Buffer.create 256 in
  let fmt = Format.formatter_of_buffer buf in
  Hugin.pp fmt spec;
  Format.pp_print_flush fmt ();
  let output = Buffer.contents buf in
  is_true ~msg:"starts with image markdown"
    (contains output "![figure](data:image/png;base64,");
  (* Base64 output should only contain valid base64 chars *)
  let b64_start = "base64," in
  let start_idx =
    let rec find i =
      if i > String.length output - String.length b64_start then -1
      else if String.sub output i (String.length b64_start) = b64_start then
        i + String.length b64_start
      else find (i + 1)
    in
    find 0
  in
  is_true ~msg:"found base64 data" (start_idx > 0);
  if start_idx > 0 then begin
    let end_idx = String.length output - 1 in
    let b64 = String.sub output start_idx (end_idx - start_idx) in
    is_true ~msg:"all chars are valid base64"
      (String.to_seq b64 |> Seq.for_all is_base64_char)
  end

(* render_to_buffer produces non-empty data *)

let test_render_to_buffer () =
  let spec = Hugin.line ~x:sample_x ~y:sample_y () in
  let buf = Hugin.render_to_buffer spec in
  is_true ~msg:"non-empty" (String.length buf > 0);
  (* PNG magic bytes: 0x89 P N G *)
  is_true ~msg:"PNG magic byte" (Char.code (String.get buf 0) = 0x89);
  is_true ~msg:"PNG P" (String.get buf 1 = 'P');
  is_true ~msg:"PNG N" (String.get buf 2 = 'N');
  is_true ~msg:"PNG G" (String.get buf 3 = 'G')

let () =
  run "Image_util"
    [
      group "pp data URI"
        [ test "produces valid base64 data URI" test_pp_data_uri ];
      group "render_to_buffer"
        [ test "produces valid PNG" test_render_to_buffer ];
    ]
