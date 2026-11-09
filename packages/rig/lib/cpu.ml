(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The host's processor

   An x86-64 host's arch is the highest level of the x86-64 psABI (§3.1.1,
   table "Micro-Architecture Levels") whose features its processor reports,
   each level's and those of the levels below it, which it includes. "Levels
   x86-64-v3 and x86-64-v4 are only available if the corresponding features
   have been fully enabled", so theirs count the registers' state the system
   saves, as XCR0 states it: XMM and YMM for v3 (Intel SDM vol. 1, §14.3),
   with opmask and ZMM for v4 (§15.2). "Any program can expect that an x86-64
   processor implements the baseline features", so a host is at least
   x86-64. A processor of a level the table lacks states the highest it
   has. CPUID's bits are those the SDM's detection procedures name (vol. 1,
   chapters 5 and 12 to 15; vol. 2A, CMPXCHG8B/CMPXCHG16B).

   An arm64 host's arch is "arm64" followed by each optional feature the
   system states, after a "+" and in lexicographic order, with the names of
   ACLE's "Function Multi Versioning" ("Mapping"), whose mangling says
   "feature names are appended in lexicographic order": "arm64+fp16" on an
   M1. *)

external instruction_set_here : unit -> string = "caml_rig_instruction_set"
external arm64_fp16 : unit -> bool = "caml_rig_arm64_fp16"

type words = { ecx1 : int; ebx7 : int; ecx81 : int; xcr0 : int }

external x86_words : unit -> words = "caml_rig_x86_words"

(* A feature, by the word and bit that state it. *)
type feature = Ecx1 of int | Ebx7 of int | Ecx81 of int | Xcr0 of int

let has w = function
  | Ecx1 b -> (w.ecx1 lsr b) land 1 = 1
  | Ebx7 b -> (w.ebx7 lsr b) land 1 = 1
  | Ecx81 b -> (w.ecx81 lsr b) land 1 = 1
  | Xcr0 b -> (w.xcr0 lsr b) land 1 = 1

let baseline = "x86-64"

(* The levels above the baseline, in order, with the features each adds. *)
let table =
  [
    ( "x86-64-v2",
      [
        Ecx1 13 (* CMPXCHG16B *);
        Ecx81 0 (* LAHF-SAHF *);
        Ecx1 23 (* POPCNT *);
        Ecx1 0 (* SSE3 *);
        Ecx1 19 (* SSE4_1 *);
        Ecx1 20 (* SSE4_2 *);
        Ecx1 9 (* SSSE3 *);
      ] );
    ( "x86-64-v3",
      [
        Ecx1 28 (* AVX *);
        Ebx7 5 (* AVX2 *);
        Ebx7 3 (* BMI1 *);
        Ebx7 8 (* BMI2 *);
        Ecx1 29 (* F16C *);
        Ecx1 12 (* FMA *);
        Ecx81 5 (* LZCNT *);
        Ecx1 22 (* MOVBE *);
        Ecx1 27 (* OSXSAVE *);
        Xcr0 1 (* XMM state *);
        Xcr0 2 (* YMM state *);
      ] );
    ( "x86-64-v4",
      [
        Ebx7 16 (* AVX512F *);
        Ebx7 30 (* AVX512BW *);
        Ebx7 28 (* AVX512CD *);
        Ebx7 17 (* AVX512DQ *);
        Ebx7 31 (* AVX512VL *);
        Xcr0 5 (* opmask state *);
        Xcr0 6 (* upper halves of ZMM0-15 *);
        Xcr0 7 (* ZMM16-31 *);
      ] );
  ]

let x86_levels = baseline :: List.map fst table

let x86_level w =
  let rec highest level = function
    | (next, features) :: above when List.for_all (has w) features ->
        highest next above
    | _ -> level
  in
  highest baseline table

let instruction_set arch =
  if List.mem arch x86_levels then "x86_64"
  else if String.starts_with ~prefix:"arm64+" arch then "arm64"
  else arch

let host_arch =
  match instruction_set_here () with
  | "x86_64" -> x86_level (x86_words ())
  | "arm64" when arm64_fp16 () -> "arm64+fp16"
  | isa -> isa
