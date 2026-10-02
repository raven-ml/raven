(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Encoded sRGB components and straight alpha, all in [0;1]. *)
type t = { r : float; g : float; b : float; alpha : float }

let in_unit x = 0. <= x && x <= 1.

let check fn what x =
  if not (in_unit x) then
    invalid_arg (Printf.sprintf "Color.%s: %s %g not in [0, 1]" fn what x)

let make fn alpha r g b =
  check fn "red" r;
  check fn "green" g;
  check fn "blue" b;
  check fn "alpha" alpha;
  { r; g; b; alpha }

let v ?(alpha = 1.) r g b = make "v" alpha r g b
let gray ?(alpha = 1.) l = make "gray" alpha l l l
let black = v 0. 0. 0.
let white = v 1. 1. 1.
let red = v 1. 0. 0.
let green = v 0. 1. 0.
let blue = v 0. 0. 1.
let transparent = v ~alpha:0. 0. 0. 0.
let r c = c.r
let g c = c.g
let b c = c.b
let alpha c = c.alpha

let with_alpha a c =
  check "with_alpha" "alpha" a;
  { c with alpha = a }

(* sRGB transfer functions *)

let linear c =
  if c <= 0.04045 then c /. 12.92 else Float.pow ((c +. 0.055) /. 1.055) 2.4

(* Clamped so that rounding cannot leave [0;1]. *)
let encode c =
  let e =
    if c <= 0.0031308 then 12.92 *. c
    else (1.055 *. Float.pow c (1. /. 2.4)) -. 0.055
  in
  Float.min 1. (Float.max 0. e)

(* Oklab, with the matrices of CSS Color 4: linear sRGB to LMS through CIE XYZ,
   then the cube roots of LMS to Oklab, and their inverses. *)

let oklab_of_linear r g b =
  let l =
    (0.41222146947076282 *. r) +. (0.53633253726173491 *. g)
    +. (0.051445993267502196 *. b)
  in
  let m =
    (0.21190349581782511 *. r) +. (0.68069955064523435 *. g)
    +. (0.10739695353694051 *. b)
  in
  let s =
    (0.088302459190056373 *. r)
    +. (0.2817188391361215 *. g) +. (0.62997870167382219 *. b)
  in
  let l = Float.cbrt l and m = Float.cbrt m and s = Float.cbrt s in
  ( (0.21045426830931396 *. l) +. (0.7936177747023053 *. m)
    -. (0.0040720430116192585 *. s),
    (1.9779985324311686 *. l) -. (2.42859224204858 *. m)
    +. (0.450593709617411 *. s),
    (0.025904042465547734 *. l)
    +. (0.7827717124575297 *. m) -. (0.8086757549230774 *. s) )

let linear_of_oklab l a b =
  let l' = l +. (0.3963377773761749 *. a) +. (0.21580375730991364 *. b) in
  let m' = l -. (0.10556134581565857 *. a) -. (0.0638541728258133 *. b) in
  let s' = l -. (0.08948417752981186 *. a) -. (1.2914855480194092 *. b) in
  let l = l' *. l' *. l' and m = m' *. m' *. m' and s = s' *. s' *. s' in
  ( (4.0767416360759592 *. l) -. (3.3077115392580634 *. m)
    +. (0.2309699031821045 *. s),
    (-1.2684379732850315 *. l) +. (2.6097573492876882 *. m)
    -. (0.3413193760026571 *. s),
    (-0.0041960761386754851 *. l)
    -. (0.70341861793593619 *. m) +. (1.7076146940746117 *. s) )

let to_oklab c = oklab_of_linear (linear c.r) (linear c.g) (linear c.b)

(* Gamut mapping as CSS Color 4 specifies it: a binary search on the chroma, at
   constant lightness and hue, for a colour whose clipped image lies less than
   [jnd] away but no closer than [jnd -. epsilon], or until the chroma range is
   narrower than [epsilon]. *)

let jnd = 0.02
let epsilon = 0.0001
let clip x = Float.min 1. (Float.max 0. x)

let clipped alpha (r, g, b) =
  { r = encode (clip r); g = encode (clip g); b = encode (clip b); alpha }

let delta_e l a b c =
  let l', a', b' = to_oklab c in
  let dl = l -. l' and da = a -. a' and db = b -. b' in
  Float.sqrt ((dl *. dl) +. (da *. da) +. (db *. db))

let in_gamut (r, g, b) = in_unit r && in_unit g && in_unit b

(* Every chroma above about [0.33] lies outside the gamut at every lightness, so
   capping the chroma at [1.] maps to a colour of the same lightness and hue,
   and keeps the cubes of [linear_of_oklab] finite. *)
let gamut_map alpha l a b =
  if l >= 1. then { white with alpha }
  else if l <= 0. then { black with alpha }
  else
    let a, b =
      if Float.hypot a b > 1. then
        let hue = Float.atan2 b a in
        (Float.cos hue, Float.sin hue)
      else (a, b)
    in
    let rgb = linear_of_oklab l a b in
    if in_gamut rgb then clipped alpha rgb
    else
      let clip_at a b = clipped alpha (linear_of_oklab l a b) in
      let origin = clip_at a b in
      if delta_e l a b origin < jnd then origin
      else
        let chroma = Float.hypot a b and hue = Float.atan2 b a in
        let cos_h = Float.cos hue and sin_h = Float.sin hue in
        let rec search lo hi lo_in_gamut last =
          let range = hi -. lo in
          if range <= epsilon then last
          else
            let c = (lo +. hi) /. 2. in
            let a = c *. cos_h and b = c *. sin_h in
            if lo_in_gamut && in_gamut (linear_of_oklab l a b) then
              search c hi lo_in_gamut last
            else
              let clip = clip_at a b in
              let e = delta_e l a b clip in
              if e >= jnd then search lo c lo_in_gamut clip
              else if jnd -. e < epsilon then clip
              else search c hi false clip
        in
        search 0. chroma true origin

let of_oklab ?(alpha = 1.) l a b =
  if not (Float.is_finite l && Float.is_finite a && Float.is_finite b) then
    invalid_arg
      (Printf.sprintf "Color.of_oklab: coordinates (%g, %g, %g) not finite" l a
         b);
  check "of_oklab" "alpha" alpha;
  gamut_map alpha l a b

(* Below this chroma a colour is grey up to rounding: greys convert to chromas
   below [1e-15], and two distinct 8-bit channels give at least [1e-3]. *)
let powerless = 4e-6
let two_pi = 2. *. Float.pi

let to_oklch c =
  let l, a, b = to_oklab c in
  let chroma = Float.hypot a b in
  if chroma < powerless then (l, chroma, Float.nan)
  else
    let h = Float.atan2 b a in
    (* [h +. two_pi] rounds to [two_pi] for a negative [h] above [-4e-16], which
       no colour reaches in practice. *)
    let h = if h < 0. then h +. two_pi else h in
    (l, chroma, if h >= two_pi then 0. else h)

let of_oklch ?(alpha = 1.) l c h =
  if not (Float.is_finite l && Float.is_finite c && c >= 0.) then
    invalid_arg
      (Printf.sprintf "Color.of_oklch: lightness %g or chroma %g invalid" l c);
  if Float.abs h = Float.infinity then
    invalid_arg (Printf.sprintf "Color.of_oklch: hue %g infinite" h);
  check "of_oklch" "alpha" alpha;
  let h = if Float.is_nan h then 0. else h in
  gamut_map alpha l (c *. Float.cos h) (c *. Float.sin h)

let mix t c c' =
  if not (in_unit t) then
    invalid_arg (Printf.sprintf "Color.mix: %g not in [0, 1]" t);
  let w = (1. -. t) *. c.alpha and w' = t *. c'.alpha in
  let alpha = w +. w' in
  if alpha = 0. then transparent
  else
    let l, a, b = to_oklab c and l', a', b' = to_oklab c' in
    let avg x x' = ((w *. x) +. (w' *. x')) /. alpha in
    gamut_map (Float.min 1. alpha) (avg l l') (avg a a') (avg b b')

let luminance_threshold = Float.sqrt 0.0525 -. 0.05

let contrast c =
  let y =
    (0.2126 *. linear c.r) +. (0.7152 *. linear c.g) +. (0.0722 *. linear c.b)
  in
  if y >= luminance_threshold then black else white

(* Hexadecimal notation *)

let hex_digit = function
  | '0' .. '9' as c -> Char.code c - Char.code '0'
  | 'a' .. 'f' as c -> Char.code c - Char.code 'a' + 10
  | 'A' .. 'F' as c -> Char.code c - Char.code 'A' + 10
  | _ -> -1

let of_hex s =
  let n = String.length s - 1 in
  let rec bad_digit i =
    if i > n then None
    else if hex_digit s.[i] < 0 then Some i
    else bad_digit (i + 1)
  in
  if n < 0 || s.[0] <> '#' then
    Error (Printf.sprintf "%S: a hex colour starts with '#'" s)
  else if not (n = 3 || n = 4 || n = 6 || n = 8) then
    Error
      (Printf.sprintf "%S: expected 3, 4, 6 or 8 hex digits after '#', found %d"
         s n)
  else
    match bad_digit 1 with
    | Some i -> Error (Printf.sprintf "%S: %C is not a hex digit" s s.[i])
    | None ->
        let digit i = hex_digit s.[i] in
        let channel k =
          let byte =
            if n <= 4 then 17 * digit (1 + k)
            else (16 * digit (1 + (2 * k))) + digit (2 + (2 * k))
          in
          float byte /. 255.
        in
        let alpha = if n = 4 || n = 8 then channel 3 else 1. in
        Ok { r = channel 0; g = channel 1; b = channel 2; alpha }

let byte x = int_of_float (Float.round (x *. 255.))

let to_hex c =
  let a = byte c.alpha in
  if a = 255 then
    Printf.sprintf "#%02x%02x%02x" (byte c.r) (byte c.g) (byte c.b)
  else Printf.sprintf "#%02x%02x%02x%02x" (byte c.r) (byte c.g) (byte c.b) a

let equal c c' =
  Float.equal c.r c'.r && Float.equal c.g c'.g && Float.equal c.b c'.b
  && Float.equal c.alpha c'.alpha

let compare c c' =
  let k = Float.compare c.r c'.r in
  if k <> 0 then k
  else
    let k = Float.compare c.g c'.g in
    if k <> 0 then k
    else
      let k = Float.compare c.b c'.b in
      if k <> 0 then k else Float.compare c.alpha c'.alpha

let pp ppf c = Format.pp_print_string ppf (to_hex c)
