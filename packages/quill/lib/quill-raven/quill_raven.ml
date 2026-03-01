(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* ───── Package configuration ───── *)

let raven_packages =
  [
    "unix";
    "threads";
    "threads.posix";
    "nx";
    "nx.core";
    "nx.backend";
    "nx.c";
    "nx.buffer";
    "nx.io";
    "rune";
    "kaun";
    "kaun.datasets";
    "hugin";
    "sowilo";
    "talon";
    "talon.csv";
    "brot";
    "fehu";
    "fehu.envs";
    "fehu.algorithms";
  ]

let raven_printers = [ "Nx.pp_data" ]

(* ───── Pretty-printers ───── *)

let base64_encode_string input =
  let alphabet =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
  in
  let len = String.length input in
  let out_len = (len + 2) / 3 * 4 in
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
      Bytes.unsafe_set out j (String.unsafe_get alphabet (b0 lsr 2));
      Bytes.unsafe_set out (j + 1)
        (String.unsafe_get alphabet (((b0 land 3) lsl 4) lor (b1 lsr 4)));
      Bytes.unsafe_set out (j + 2)
        (if i + 1 < len then
           String.unsafe_get alphabet (((b1 land 0xf) lsl 2) lor (b2 lsr 6))
         else '=');
      Bytes.unsafe_set out (j + 3)
        (if i + 2 < len then String.unsafe_get alphabet (b2 land 0x3f) else '=');
      loop (i + 3) (j + 4)
    end
  in
  loop 0 0;
  Bytes.unsafe_to_string out

let pp_figure fmt (figure : Hugin.figure) =
  let png_data = Hugin.render figure in
  let b64 = base64_encode_string png_data in
  Format.pp_open_stag fmt
    (Quill.Cell.Display_tag { mime = "image/png"; data = b64 });
  Format.fprintf fmt "<figure>";
  Format.pp_close_stag fmt ()

(* ───── Setup ───── *)

let setup () =
  Quill_top.add_packages raven_packages;
  List.iter Quill_top.install_printer raven_printers;
  Quill_top.install_printer_fn ~ty:"Hugin.figure" (fun fmt obj ->
      pp_figure fmt (Obj.obj obj))

(* ───── Kernel ───── *)

let create ~on_event = Quill_top.create ~setup ~on_event ()
