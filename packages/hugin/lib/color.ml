(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = { l : float; c : float; h : float; a : float }

(* Constructors *)

let oklch ~l ~c ~h () = { l; c; h; a = 1. }
let oklcha ~l ~c ~h ~a () = { l; c; h; a }

(* sRGB <-> linear RGB *)

let srgb_to_linear c =
  if c <= 0.04045 then c /. 12.92 else Float.pow ((c +. 0.055) /. 1.055) 2.4

let linear_to_srgb c =
  if c <= 0.0031308 then 12.92 *. c
  else (1.055 *. Float.pow c (1. /. 2.4)) -. 0.055

(* Linear RGB -> OKLab *)

let linear_rgb_to_oklab r g b =
  let l = (0.4122214708 *. r) +. (0.5363325363 *. g) +. (0.0514459929 *. b) in
  let m = (0.2119034982 *. r) +. (0.6806995451 *. g) +. (0.1073969566 *. b) in
  let s = (0.0883024619 *. r) +. (0.2164557896 *. g) +. (0.6898418685 *. b) in
  let l = Float.cbrt l and m = Float.cbrt m and s = Float.cbrt s in
  let lab_l =
    (0.2104542553 *. l) +. (0.7936177850 *. m) -. (0.0040720468 *. s)
  in
  let lab_a =
    (1.9779984951 *. l) -. (2.4285922050 *. m) +. (0.4505937099 *. s)
  in
  let lab_b =
    (0.0259040371 *. l) +. (0.7827717662 *. m) -. (0.8086757660 *. s)
  in
  (lab_l, lab_a, lab_b)

(* OKLab -> linear RGB *)

let oklab_to_linear_rgb lab_l lab_a lab_b =
  let l = lab_l +. (0.3963377774 *. lab_a) +. (0.2158037573 *. lab_b) in
  let m = lab_l -. (0.1055613458 *. lab_a) -. (0.0638541728 *. lab_b) in
  let s = lab_l -. (0.0894841775 *. lab_a) -. (1.2914855480 *. lab_b) in
  let l = l *. l *. l and m = m *. m *. m and s = s *. s *. s in
  let r = (4.0767416621 *. l) -. (3.3077115913 *. m) +. (0.2309699292 *. s) in
  let g = (-1.2684380046 *. l) +. (2.6097574011 *. m) -. (0.3413193965 *. s) in
  let b = (-0.0041960863 *. l) -. (0.7034186147 *. m) +. (1.7076147010 *. s) in
  (r, g, b)

(* OKLab <-> OKLCH *)

let oklab_to_oklch lab_l lab_a lab_b =
  let c = Float.sqrt ((lab_a *. lab_a) +. (lab_b *. lab_b)) in
  let h = Float.atan2 lab_b lab_a *. 180. /. Float.pi in
  let h = if h < 0. then h +. 360. else h in
  (lab_l, c, h)

let oklch_to_oklab l c h =
  let h_rad = h *. Float.pi /. 180. in
  (l, c *. Float.cos h_rad, c *. Float.sin h_rad)

(* sRGB -> OKLCH *)

let of_srgb r g b =
  let lr = srgb_to_linear r
  and lg = srgb_to_linear g
  and lb = srgb_to_linear b in
  let lab_l, lab_a, lab_b = linear_rgb_to_oklab lr lg lb in
  oklab_to_oklch lab_l lab_a lab_b

(* OKLCH -> sRGB *)

let to_srgb l c h =
  let lab_l, lab_a, lab_b = oklch_to_oklab l c h in
  let lr, lg, lb = oklab_to_linear_rgb lab_l lab_a lab_b in
  let clamp v = Float.max 0. (Float.min 1. v) in
  ( linear_to_srgb (clamp lr),
    linear_to_srgb (clamp lg),
    linear_to_srgb (clamp lb) )

let rgb ~r ~g ~b () =
  let l, c, h = of_srgb r g b in
  { l; c; h; a = 1. }

let rgba ~r ~g ~b ~a () =
  let l, c, h = of_srgb r g b in
  { l; c; h; a }

let hex_digit c =
  match c with
  | '0' .. '9' -> Char.code c - Char.code '0'
  | 'a' .. 'f' -> Char.code c - Char.code 'a' + 10
  | 'A' .. 'F' -> Char.code c - Char.code 'A' + 10
  | _ -> invalid_arg (Printf.sprintf "Color.hex: invalid hex digit %C" c)

let hex_byte s i =
  let hi = hex_digit (String.get s i) in
  let lo = hex_digit (String.get s (i + 1)) in
  float ((hi * 16) + lo) /. 255.

let hex s =
  let n = String.length s in
  let off = if n > 0 && String.get s 0 = '#' then 1 else 0 in
  let len = n - off in
  match len with
  | 6 ->
      let r = hex_byte s off
      and g = hex_byte s (off + 2)
      and b = hex_byte s (off + 4) in
      rgb ~r ~g ~b ()
  | 8 ->
      let r = hex_byte s off and g = hex_byte s (off + 2) in
      let b = hex_byte s (off + 4) and a = hex_byte s (off + 6) in
      rgba ~r ~g ~b ~a ()
  | _ ->
      invalid_arg
        (Printf.sprintf "Color.hex: expected 6 or 8 hex digits, got %d" len)

(* Accessors *)

let lightness t = t.l
let chroma t = t.c
let hue t = t.h
let alpha t = t.a

(* Converting *)

let to_rgba t =
  let r, g, b = to_srgb t.l t.c t.h in
  (r, g, b, t.a)

(* Operations *)

let with_alpha a t = { t with a }
let lighten amount t = { t with l = Float.min 1. (t.l +. amount) }
let darken amount t = { t with l = Float.max 0. (t.l -. amount) }

let interpolate_hue ratio h1 h2 =
  let diff = h2 -. h1 in
  let diff =
    if diff > 180. then diff -. 360.
    else if diff < -180. then diff +. 360.
    else diff
  in
  let h = h1 +. (ratio *. diff) in
  if h < 0. then h +. 360. else if h >= 360. then h -. 360. else h

let mix ratio a b =
  {
    l = a.l +. (ratio *. (b.l -. a.l));
    c = a.c +. (ratio *. (b.c -. a.c));
    h = interpolate_hue ratio a.h b.h;
    a = a.a +. (ratio *. (b.a -. a.a));
  }

(* Named colors — Okabe-Ito *)

let orange = hex "#E69F00"
let sky_blue = hex "#56B4E9"
let green = hex "#009E73"
let yellow = hex "#F0E442"
let blue = hex "#0072B2"
let vermillion = hex "#D55E00"
let purple = hex "#CC79A7"
let black = { l = 0.; c = 0.; h = 0.; a = 1. }
let white = { l = 1.; c = 0.; h = 0.; a = 1. }
let gray = oklch ~l:0.5 ~c:0. ~h:0. ()

(* Formatting *)

let pp fmt t = Format.fprintf fmt "oklch(%g %g %g / %g)" t.l t.c t.h t.a
