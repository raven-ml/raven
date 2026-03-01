(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = Color.t array

let eval t v =
  let v = Float.max 0. (Float.min 1. v) in
  let i = int_of_float (v *. 255.) in
  t.(min i 255)

let of_colors stops =
  let n = Array.length stops in
  if n < 2 then invalid_arg "Cmap.of_colors: need at least 2 stops";
  Array.init 256 (fun i ->
      let v = float i /. 255. in
      let scaled = v *. float (n - 1) in
      let idx = int_of_float scaled in
      let idx = min idx (n - 2) in
      let frac = scaled -. float idx in
      Color.mix frac stops.(idx) stops.(idx + 1))

(* Decode a canonical 256-entry hex-encoded colormap *)

let hex_digit c =
  match c with
  | '0' .. '9' -> Char.code c - Char.code '0'
  | 'a' .. 'f' -> 10 + Char.code c - Char.code 'a'
  | 'A' .. 'F' -> 10 + Char.code c - Char.code 'A'
  | _ -> invalid_arg (Printf.sprintf "Cmap.hex_digit: invalid hex digit %C" c)

let decode_hex_cmap hex =
  Array.init 256 (fun i ->
      let off = i * 6 in
      let byte j =
        let h = hex_digit (String.unsafe_get hex (off + (j * 2))) in
        let l = hex_digit (String.unsafe_get hex (off + (j * 2) + 1)) in
        float ((h lsl 4) lor l) /. 255.
      in
      Color.rgb ~r:(byte 0) ~g:(byte 1) ~b:(byte 2) ())

let viridis = decode_hex_cmap Cmap_data.viridis_hex
let plasma = decode_hex_cmap Cmap_data.plasma_hex
let inferno = decode_hex_cmap Cmap_data.inferno_hex
let magma = decode_hex_cmap Cmap_data.magma_hex
let cividis = decode_hex_cmap Cmap_data.cividis_hex
let coolwarm = decode_hex_cmap Cmap_data.coolwarm_hex

let gray =
  of_colors [| Color.rgb ~r:0. ~g:0. ~b:0. (); Color.rgb ~r:1. ~g:1. ~b:1. () |]

let gray_r =
  of_colors [| Color.rgb ~r:1. ~g:1. ~b:1. (); Color.rgb ~r:0. ~g:0. ~b:0. () |]

let hot =
  of_colors
    [|
      Color.rgb ~r:0. ~g:0. ~b:0. ();
      Color.rgb ~r:0.7 ~g:0. ~b:0. ();
      Color.rgb ~r:1. ~g:0.6 ~b:0. ();
      Color.rgb ~r:1. ~g:1. ~b:1. ();
    |]
