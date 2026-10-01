(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin_next_gg

(* How a scheme reads [n] classes, with its table before any reversal, never
   mutated. A ramp holds its stops, and a Brewer scheme its designed tables from
   three classes up and its table once [unreversed] has made it. *)
type kind =
  | Listed of Color.t array
  | Cyclic of Color.t array
  | Palette of Color.t array
  | Ramp of { stops : Color.t array; table : Color.t array }
  | Brewer of {
      tables : Color.t array array;
      diverging : bool;
      table : Color.t array option Atomic.t;
    }

(* Named schemes are compared by [name], made ones by their colours. *)
type t = { name : string option; kind : kind; reversed : bool }

let reversed cs =
  let n = Array.length cs in
  Array.init n (fun i -> cs.(n - 1 - i))

(* Decoding the data *)

let of_floats xs =
  Array.init
    (Array.length xs / 3)
    (fun i -> Color.v xs.(3 * i) xs.((3 * i) + 1) xs.((3 * i) + 2))

let of_hex s =
  let byte i = float (int_of_string ("0x" ^ String.sub s i 2)) /. 255. in
  Array.init
    (String.length s / 6)
    (fun i -> Color.v (byte (6 * i)) (byte ((6 * i) + 2)) (byte ((6 * i) + 4)))

(* Ramps *)

(* [ramp_color cs num den] is the colour [num / den] of the way along the ramp
   through [cs], with [num <= den × (Array.length cs - 1)]. *)
let ramp_color cs num den =
  let j = num / den and r = num mod den in
  if r = 0 then cs.(j) else Color.mix (float r /. float den) cs.(j) cs.(j + 1)

let ramp_colors n cs =
  let m = Array.length cs - 1 in
  if n = 1 then [| ramp_color cs m 2 |]
  else Array.init n (fun i -> ramp_color cs (i * m) (n - 1))

let brewer_colors n tables diverging =
  let t3 = tables.(0) and last = Array.length tables - 1 in
  match n with
  | 0 -> [||]
  | 1 -> [| t3.(1) |]
  | 2 -> if diverging then [| t3.(0); t3.(2) |] else [| t3.(1); t3.(2) |]
  | n when n - 3 <= last -> Array.copy tables.(n - 3)
  | n -> ramp_colors n tables.(last)

(* A Brewer scheme's table, 256 Oklab mixes, is made at its first reading rather
   than when the program starts, where the 27 tables cost about 1.6 ms. Domains
   that read it first together may each make it; they make equal tables. *)
let unreversed s =
  match s.kind with
  | Listed t | Cyclic t | Palette t | Ramp { table = t; _ } -> t
  | Brewer { tables; table; _ } -> (
      match Atomic.get table with
      | Some t -> t
      | None ->
          let t = ramp_colors 256 tables.(Array.length tables - 1) in
          Atomic.set table (Some t);
          t)

(* Making schemes *)

let make ?name kind = { name; kind; reversed = false }

let ramp cs =
  if Array.length cs = 0 then invalid_arg "Scheme.ramp: no colours";
  let stops = Array.copy cs in
  make (Ramp { stops; table = ramp_colors 256 stops })

let palette cs =
  if Array.length cs = 0 then invalid_arg "Scheme.palette: no colours";
  make (Palette (Array.copy cs))

let reverse s = { s with reversed = not s.reversed }

(* Readings *)

(* [at s u] is the colour of the bin of [u], which is not [nan], in the
   continuous reading of [s]. *)
let at s u =
  let t = unreversed s in
  let n = Array.length t in
  let x = float n *. u in
  let i =
    if x < 1. then 0 else if x >= float (n - 1) then n - 1 else truncate x
  in
  t.(if s.reversed then n - 1 - i else i)

let color ?(unknown = Color.transparent) s u =
  if Float.is_nan u then unknown else at s u

let table s =
  if s.reversed then reversed (unreversed s) else Array.copy (unreversed s)

let colors n s =
  if n < 0 then
    invalid_arg (Printf.sprintf "Scheme.colors: negative count %d" n);
  let in_order cs = if s.reversed then reversed cs else cs in
  match s.kind with
  | Ramp { stops; _ } -> in_order (ramp_colors n stops)
  | Brewer { tables; diverging; _ } ->
      in_order (brewer_colors n tables diverging)
  | Palette t ->
      let k = Array.length t in
      Array.init n (fun i ->
          let j = i mod k in
          t.(if s.reversed then k - 1 - j else j))
  | Cyclic _ -> Array.init n (fun i -> at s (float i /. float n))
  | Listed _ ->
      if n = 1 then [| at s 0.5 |]
      else Array.init n (fun i -> at s (float i /. float (n - 1)))

(* Named schemes *)

let listed name data = make ~name (Listed (of_floats data))

let brewer name ~diverging specs =
  let tables = Array.map of_hex specs in
  make ~name (Brewer { tables; diverging; table = Atomic.make None })

let qualitative name spec = make ~name (Palette (of_hex spec))

(* Sequential schemes *)

let viridis = listed "viridis" Scheme_data.viridis
let magma = listed "magma" Scheme_data.magma
let inferno = listed "inferno" Scheme_data.inferno
let plasma = listed "plasma" Scheme_data.plasma
let cividis = listed "cividis" Scheme_data.cividis
let turbo = listed "turbo" Scheme_data.turbo
let sequential name specs = brewer name ~diverging:false specs
let blues = sequential "blues" Scheme_data.blues
let greens = sequential "greens" Scheme_data.greens
let greys = sequential "greys" Scheme_data.greys
let oranges = sequential "oranges" Scheme_data.oranges
let purples = sequential "purples" Scheme_data.purples
let reds = sequential "reds" Scheme_data.reds
let bugn = sequential "bugn" Scheme_data.bugn
let bupu = sequential "bupu" Scheme_data.bupu
let gnbu = sequential "gnbu" Scheme_data.gnbu
let orrd = sequential "orrd" Scheme_data.orrd
let pubu = sequential "pubu" Scheme_data.pubu
let pubugn = sequential "pubugn" Scheme_data.pubugn
let purd = sequential "purd" Scheme_data.purd
let rdpu = sequential "rdpu" Scheme_data.rdpu
let ylgn = sequential "ylgn" Scheme_data.ylgn
let ylgnbu = sequential "ylgnbu" Scheme_data.ylgnbu
let ylorbr = sequential "ylorbr" Scheme_data.ylorbr
let ylorrd = sequential "ylorrd" Scheme_data.ylorrd

(* Diverging schemes *)

let diverging name specs = brewer name ~diverging:true specs
let brbg = diverging "brbg" Scheme_data.brbg
let piyg = diverging "piyg" Scheme_data.piyg
let prgn = diverging "prgn" Scheme_data.prgn
let puor = diverging "puor" Scheme_data.puor
let rdbu = diverging "rdbu" Scheme_data.rdbu
let rdgy = diverging "rdgy" Scheme_data.rdgy
let rdylbu = diverging "rdylbu" Scheme_data.rdylbu
let rdylgn = diverging "rdylgn" Scheme_data.rdylgn
let spectral = diverging "spectral" Scheme_data.spectral

(* Cyclic schemes *)

let twilight = make ~name:"twilight" (Cyclic (of_floats Scheme_data.twilight))

(* Qualitative schemes *)

let okabe_ito = qualitative "okabe_ito" Scheme_data.okabe_ito
let tableau10 = qualitative "tableau10" Scheme_data.tableau10
let accent = qualitative "accent" Scheme_data.accent
let dark2 = qualitative "dark2" Scheme_data.dark2
let paired = qualitative "paired" Scheme_data.paired
let pastel1 = qualitative "pastel1" Scheme_data.pastel1
let pastel2 = qualitative "pastel2" Scheme_data.pastel2
let set1 = qualitative "set1" Scheme_data.set1
let set2 = qualitative "set2" Scheme_data.set2
let set3 = qualitative "set3" Scheme_data.set3

(* Colour vision deficiency *)

type deficiency = Protan | Deutan | Tritan

(* The sRGB transfer functions, the encoding clamped so that rounding cannot
   leave [0;1]. *)
let to_linear c =
  if c <= 0.04045 then c /. 12.92 else Float.pow ((c +. 0.055) /. 1.055) 2.4

let of_linear c =
  let c = Float.min 1. (Float.max 0. c) in
  let e =
    if c <= 0.0031308 then 12.92 *. c
    else (1.055 *. Float.pow c (1. /. 2.4)) -. 0.055
  in
  Float.min 1. (Float.max 0. e)

let simulate ?(severity = 1.) d c =
  if not (0. <= severity && severity <= 1.) then
    invalid_arg
      (Printf.sprintf "Scheme.simulate: severity %g not in [0, 1]" severity);
  let m =
    match d with
    | Protan -> Scheme_data.protan
    | Deutan -> Scheme_data.deutan
    | Tritan -> Scheme_data.tritan
  in
  (* The matrices at the tabulated severities [k / 10] and [(k + 1) / 10] around
     [severity], weighted [1 - f] and [f]. *)
  let x = 10. *. severity in
  let k = Int.min 9 (truncate x) in
  let f = x -. float k in
  let r = to_linear (Color.r c) in
  let g = to_linear (Color.g c) in
  let b = to_linear (Color.b c) in
  let row i =
    let a j =
      let at k = m.((9 * k) + (3 * i) + j) in
      ((1. -. f) *. at k) +. (f *. at (k + 1))
    in
    of_linear ((a 0 *. r) +. (a 1 *. g) +. (a 2 *. b))
  in
  Color.v ~alpha:(Color.alpha c) (row 0) (row 1) (row 2)

(* Comparing and formatting *)

let same_colors cs cs' =
  Array.length cs = Array.length cs' && Array.for_all2 Color.equal cs cs'

let equal s s' =
  Bool.equal s.reversed s'.reversed
  &&
  match (s.name, s'.name) with
  | Some n, Some n' -> String.equal n n'
  | None, None -> (
      match (s.kind, s'.kind) with
      | Ramp { stops; _ }, Ramp { stops = stops'; _ } ->
          same_colors stops stops'
      | Palette cs, Palette cs' -> same_colors cs cs'
      | _ -> false)
  | Some _, None | None, Some _ -> false

let pp ppf s =
  let pp_made ppf (cons, cs) =
    Format.fprintf ppf "@[<1>%s(%a)@]" cons
      (Format.pp_print_array ~pp_sep:Format.pp_print_space Color.pp)
      cs
  in
  let pp_unreversed ppf s =
    match (s.name, s.kind) with
    | Some name, _ -> Format.pp_print_string ppf name
    | None, Ramp { stops; _ } -> pp_made ppf ("ramp", stops)
    | None, _ -> pp_made ppf ("palette", unreversed s)
  in
  if s.reversed then Format.fprintf ppf "@[<1>reverse(%a)@]" pp_unreversed s
  else pp_unreversed ppf s
