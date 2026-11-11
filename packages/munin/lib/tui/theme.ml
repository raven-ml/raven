(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module S = Ansi.Style
module C = Ansi.Color

type look = {
  scheme : [ `Dark | `Light ];
  depth : Ansi.Styling.t;
  background : (int * int * int) option;
}

type token = Raise | Sel | Line | Line2 | Dim | Mid | Fg | Hi | Live | Fail | Warn
type t = { look : look; bg : int }

let dark = function
  | Raise -> 0x161a20
  | Sel -> 0x1b2330
  | Line -> 0x22272f
  | Line2 -> 0x323944
  | Dim -> 0x5f6773
  | Mid -> 0x8f98a5
  | Fg -> 0xc9ced6
  | Hi -> 0xeef1f5
  | Live -> 0x3ddc84
  | Fail -> 0xff6b6b
  | Warn -> 0xe5b45a

let light = function
  | Raise -> 0xefefec
  | Sel -> 0xe3eaf5
  | Line -> 0xe0e0dc
  | Line2 -> 0xc9c9c4
  | Dim -> 0x8a8f98
  | Mid -> 0x5d636c
  | Fg -> 0x2e333a
  | Hi -> 0x0f1216
  | Live -> 0x12894a
  | Fail -> 0xc93636
  | Warn -> 0x9a6400

let dark_bg = 0x0e1013
let light_bg = 0xfbfbfa
let dark_runs = [| 0x6cb6ff; 0x4fd1c5; 0xc792ea; 0xf49ac2; 0xb4befe; 0x89dceb |]
let light_runs = [| 0x1f6fd1; 0x0f8f86; 0x8a4fc4; 0xc2427f; 0x5c64d6; 0x1c8fb0 |]
let runs = Array.length dark_runs

let ansi16_runs =
  [| C.blue; C.cyan; C.magenta; C.bright_magenta; C.bright_blue; C.bright_cyan |]

let make look =
  let bg =
    match (look.background, look.scheme) with
    | Some (r, g, b), _ -> (r lsl 16) lor (g lsl 8) lor b
    | None, `Dark -> dark_bg
    | None, `Light -> light_bg
  in
  { look; bg }

(* The depths the tokens are given at: 8 colours take the 16's, which the
   output reduces, and an output without style takes the attributes, which
   it drops. *)
let level t =
  match t.look.depth with
  | `Truecolor -> `Truecolor
  | `Ansi256 -> `Ansi256
  | `Ansi16 | `Ansi8 -> `Ansi16
  | `Mono | `Plain -> `Mono

let coloured t = level t <> `Mono
let hex t tok = match t.look.scheme with `Dark -> dark tok | `Light -> light tok
let rgb x = C.of_rgb ((x lsr 16) land 0xff) ((x lsr 8) land 0xff) (x land 0xff)

let mix a b f =
  let ch x s = Float.of_int ((x lsr s) land 0xff) in
  let m s = Float.to_int (Float.round ((ch a s *. (1. -. f)) +. (ch b s *. f))) in
  (m 16 lsl 16) lor (m 8 lsl 8) lor m 0

(* An RGB colour at the depth: itself with direct colour, the nearest entry
   of the cube and the ramp with 256. *)
let deep t x =
  match level t with
  | `Truecolor -> Some (rgb x)
  | `Ansi256 -> Some (C.to_depth `Ansi256 (rgb x))
  | `Ansi16 | `Mono -> None

let ansi16 = function
  | Sel | Line | Line2 -> Some C.bright_black
  | Live -> Some C.green
  | Fail -> Some C.red
  | Warn -> Some C.yellow
  | Raise | Dim | Mid | Fg | Hi -> None

let colour t tok =
  match level t with
  | `Truecolor | `Ansi256 -> deep t (hex t tok)
  | `Ansi16 -> ansi16 tok
  | `Mono -> None

(* Below 256 colours the dim tokens are faint and the brightest is bold, so
   that the weights read without the greys. *)
let weights t tok =
  match (level t, tok) with
  | (`Ansi16 | `Mono), (Dim | Line | Line2) -> (true, false)
  | (`Ansi16 | `Mono), Hi -> (false, true)
  | _ -> (false, false)

let fg ?(bold = false) t tok =
  let dim, strong = weights t tok in
  S.make ?fg:(colour t tok) ~bold:(bold || strong) ~dim ()

let bg t tok = colour t tok
let ground _ = C.default_bg

let run_hex t i =
  let i = ((i mod runs) + runs) mod runs in
  match t.look.scheme with `Dark -> dark_runs.(i) | `Light -> light_runs.(i)

let run_colour t i =
  match level t with
  | `Truecolor | `Ansi256 -> deep t (run_hex t i)
  | `Ansi16 -> Some ansi16_runs.(((i mod runs) + runs) mod runs)
  | `Mono -> None

let run ?(bold = false) t i = S.make ?fg:(run_colour t i) ~bold ()
let half = 0.5

let faint t tok =
  match level t with
  | `Truecolor | `Ansi256 -> S.make ?fg:(deep t (mix (hex t tok) t.bg half)) ()
  | `Ansi16 -> S.make ?fg:(ansi16 tok) ~dim:true ()
  | `Mono -> S.make ~dim:true ()

let faint_run t i =
  match level t with
  | `Truecolor | `Ansi256 -> S.make ?fg:(deep t (mix (run_hex t i) t.bg half)) ()
  | `Ansi16 -> S.make ?fg:(run_colour t i) ~dim:true ()
  | `Mono -> S.make ~dim:true ()

let tint t = deep t (mix (hex t Fail) t.bg 0.85)

let sel t =
  match level t with
  | `Mono -> S.make ~inverse:true ()
  | _ -> S.make ?bg:(bg t Sel) ()
