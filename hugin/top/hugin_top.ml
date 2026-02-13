(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let install_printer name =
  let phrase =
    Printf.sprintf "#install_printer %s;;" name
    |> Lexing.from_string
    |> !Toploop.parse_toplevel_phrase
  in
  Toploop.execute_phrase false Format.err_formatter phrase |> ignore

let base64_encode_string input =
  let alphabet =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
  in
  let len = String.length input in
  let out_len = ((len + 2) / 3) * 4 in
  let out = Bytes.create out_len in
  let rec loop i j =
    if i < len then begin
      let b0 = Char.code (String.unsafe_get input i) in
      let b1 =
        if i + 1 < len then Char.code (String.unsafe_get input (i + 1)) else 0
      in
      let b2 =
        if i + 2 < len then Char.code (String.unsafe_get input (i + 2)) else 0
      in
      Bytes.unsafe_set out j
        (String.unsafe_get alphabet (b0 lsr 2));
      Bytes.unsafe_set out (j + 1)
        (String.unsafe_get alphabet (((b0 land 3) lsl 4) lor (b1 lsr 4)));
      Bytes.unsafe_set out (j + 2)
        (if i + 1 < len
         then String.unsafe_get alphabet (((b1 land 0xf) lsl 2) lor (b2 lsr 6))
         else '=');
      Bytes.unsafe_set out (j + 3)
        (if i + 2 < len then String.unsafe_get alphabet (b2 land 0x3f)
         else '=');
      loop (i + 3) (j + 4)
    end
  in
  loop 0 0;
  Bytes.unsafe_to_string out

let pp_hugin_figure fmt figure =
  let image_data = Hugin.render figure in
  let base64_data = base64_encode_string image_data in
  Format.fprintf fmt "![figure](data:image/png;base64,%s)" base64_data

let () = install_printer "Hugin_top.pp_hugin_figure"
