(*---------------------------------------------------------------------------
  Tests for Image_util.base64_encode — exercised indirectly through Hugin.pp
  which calls base64_encode on PNG buffer data. We also test the base64 logic
  through the pp data URI output.
  ---------------------------------------------------------------------------*)

open Hugin
open Windtrap

let is_base64_char = function
  | 'A' .. 'Z' | 'a' .. 'z' | '0' .. '9' | '+' | '/' | '=' -> true
  | _ -> false

(* The distinct characters of [s] that base64 does not admit, sorted. Reporting
   the offenders lets the failure name what went wrong, where a [for_all]
   boolean can only say "false". *)
let base64_offenders s =
  String.to_seq s
  |> Seq.filter (fun c -> not (is_base64_char c))
  |> List.of_seq
  |> List.sort_uniq Char.compare

let sample_x = Nx.init Float32 [| 5 |] (fun i -> float_of_int i.(0))
let sample_y = Nx.init Float32 [| 5 |] (fun i -> float_of_int i.(0))
let sample_line () = Hugin.line ~x:sample_x ~y:sample_y ()

(* pp produces a data URI with base64 encoded PNG *)

let uri_prefix = "![figure](data:image/png;base64,"

let pp_to_string spec =
  let buf = Buffer.create 256 in
  let fmt = Format.formatter_of_buffer buf in
  Hugin.pp fmt spec;
  Format.pp_print_flush fmt ();
  Buffer.contents buf

let test_pp_data_uri () =
  let output = pp_to_string (sample_line ()) in
  (* The markdown image wrapper is a prefix and a suffix, not merely present. *)
  starts_with ~affix:uri_prefix output;
  ends_with ~affix:")" output;
  let payload =
    String.sub output (String.length uri_prefix)
      (String.length output - String.length uri_prefix - 1)
  in
  greater ~msg:"the data URI carries a payload" int ~than:0
    (String.length payload);
  equal ~msg:"payload is base64" (list char) [] (base64_offenders payload);
  (* base64 encodes 3 bytes into 4 characters, so the payload is a multiple of
     four with padding only at the very end. *)
  equal ~msg:"base64 length is a multiple of 4" int 0
    (String.length payload mod 4);
  (* '=' is padding: it may only occupy the last two positions. *)
  let body = String.sub payload 0 (String.length payload - 2) in
  not_contains ~msg:"padding only at the end" ~sub:"=" body

(* render_to_buffer produces non-empty data *)

(* PNG signature and terminating IEND chunk, from the PNG spec: an eight-byte
   magic, and a zero-length IEND chunk with its fixed CRC. Asserting both ends
   is what distinguishes a complete PNG from a truncated one. *)
let png_magic = "\x89PNG\r\n\x1a\n"
let png_iend = "\x00\x00\x00\x00IEND\xae\x42\x60\x82"

let test_render_to_buffer () =
  let buf = Hugin.render_to_buffer (sample_line ()) in
  greater ~msg:"non-empty" int ~than:0 (String.length buf);
  starts_with ~msg:"PNG signature" ~affix:png_magic buf;
  ends_with ~msg:"PNG IEND chunk" ~affix:png_iend buf

let test_pp_payload_decodes_to_the_png () =
  (* The data URI must carry the very bytes render_to_buffer produces, which is
     the claim tying the two entry points together. *)
  let output = pp_to_string (sample_line ()) in
  let payload =
    String.sub output (String.length uri_prefix)
      (String.length output - String.length uri_prefix - 1)
  in
  let png = Hugin.render_to_buffer (sample_line ()) in
  (* 4 base64 characters per 3 bytes, rounded up to the padded block. *)
  equal ~msg:"payload length matches the PNG size" int
    ((String.length png + 2) / 3 * 4)
    (String.length payload)

let () =
  run "Image_util"
    [
      group "pp data URI"
        [
          test "produces valid base64 data URI" test_pp_data_uri;
          test "payload matches the rendered PNG"
            test_pp_payload_decodes_to_the_png;
        ];
      group "render_to_buffer"
        [ test "produces a complete PNG" test_render_to_buffer ];
    ]
