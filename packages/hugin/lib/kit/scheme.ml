(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Hugin_gg

(* How a scheme reads [n] classes, with its table before any reversal, never
   mutated. A ramp holds its stops, and a Brewer scheme its designed tables from
   three classes up and its table once [unreversed] has made it. *)
type kind =
  | Listed of Color.t array
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

(* [of_hex s] is the colours of the hexadecimal digits [s], six per colour. The
   tables of [Scheme_data] are well formed. *)
let of_hex s =
  let color i =
    match Color.of_hex ("#" ^ String.sub s (6 * i) 6) with
    | Ok c -> c
    | Error _ -> assert false
  in
  Array.init (String.length s / 6) color

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
  | Listed t | Palette t | Ramp { table = t; _ } -> t
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
  | Listed _ ->
      if n = 1 then [| at s 0.5 |]
      else Array.init n (fun i -> at s (float i /. float (n - 1)))

(* Named schemes *)

let listed name data = make ~name (Listed (of_hex data))

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
