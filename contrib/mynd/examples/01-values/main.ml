(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* An image's values. A decoder gives integer codes; Mynd.convert turns them
   into the fractions of full scale they stand for, and back. *)

let shape x =
  let dims = Array.to_list (Array.map string_of_int (Nx.shape x)) in
  "[" ^ String.concat "; " dims ^ "]"

(* [pixel x] is the pixel at row 150, column 225 of [x], one string per
   channel. *)
let pixel fmt x = List.init 3 (fun c -> fmt (Nx.item [ 150; 225; c ] x))

let line name values =
  Printf.printf "  %-8s %s\n" name (String.concat " " values)

let count m = Int64.to_int (Nx.item [] (Nx.count m))

let () =
  let img = Nx_io.load_image "../chelsea.png" in
  Format.printf "chelsea.png: %s %a@." (shape img) Nx.pp_dtype (Nx.dtype img);
  let n = Nx.numel img in

  (* A uint8 code [k] stands for [k / 255], a uint16 code [k] for
     [k / 65535]: the same fraction is [257 k] in uint16. *)
  let x = Mynd.convert Nx.float32 img in
  Printf.printf "pixel (150, 225):\n";
  line "uint8" (pixel string_of_int img);
  line "float32" (pixel (Printf.sprintf "%.4f") x);
  line "uint16" (pixel string_of_int (Mynd.convert Nx.uint16 img));

  (* Every code survives a round trip through float32. *)
  let back = Mynd.convert Nx.uint8 x in
  Printf.printf "uint8 to float32 and back: %d of %d codes changed\n"
    (count (Nx.not_equal back img))
    n;

  (* convert rounds a fraction to the nearest code. Truncating [255 x], as a
     cast does, gives one code less wherever the fraction of [255 x] is at
     least a half. Under a gamma of 0.8 that is about half the codes. *)
  let gamma x = Nx.pow_s x 0.8 in
  let lower x =
    let rounded = Mynd.convert Nx.uint8 (gamma x) in
    let truncated = Nx.cast Nx.uint8 (Nx.mul_s (gamma x) 255.) in
    let d = Nx.sub (Nx.cast Nx.int16 rounded) (Nx.cast Nx.int16 truncated) in
    count (Nx.equal_s d 1)
  in
  let codes = Mynd.convert Nx.float32 (Nx.arange Nx.uint8 0 256 1) in
  Printf.printf "after a gamma of 0.8, truncation is one code below rounding\n";
  Printf.printf "  for %d of the 256 codes\n" (lower codes);
  Printf.printf "  for %d of the photograph's %d values\n" (lower x) n
