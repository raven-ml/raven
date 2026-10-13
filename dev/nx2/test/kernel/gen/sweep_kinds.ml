(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Checks every f32 transcendental kind of one operand at all 2^32 arguments and
   records each kind's largest error, its largest fractional error and worst
   arguments, stamped with the digest of nx_kinds.h and nx_kinds_real.h, in
   golden/kinds. Run it from dev/nx2/test/kernel after a change to either, or
   when test_kinds' digests of results move:

   ../../../../_build/default/dev/nx2/test/kernel/gen/sweep_kinds.exe

   It takes ten to twenty CPU-minutes, too long for a test run; test_kinds fails
   until the record matches the headers, and rechecks the worst arguments on
   every run. A kind past its bound, or whose fractional error passes
   Nx_kinds_support.f32_fractions, writes no record. *)

module K = Nx_kinds_support

let () =
  let check (kind, bound) =
    let w = K.sweep kind in
    let e = if Array.length w.errors = 0 then 0 else fst w.errors.(0) in
    let ceiling = List.assoc kind K.f32_fractions in
    Printf.printf "%-6s max %d ulp%s (bound %d), %.3f ulps (at most %.3f)\n%!"
      kind e
      (if e = 1 then "" else "s")
      bound w.frac ceiling;
    if e > bound then (
      let p = snd w.errors.(0) in
      Printf.printf "%s exceeds its bound at %h (0x%08x)\n" kind
        (Int32.float_of_bits (Int32.of_int p))
        p;
      exit 1);
    if Float.round (w.frac *. 1000.) /. 1000. > ceiling then (
      Printf.printf "%s's fractional error grew, at %h (0x%08x)\n" kind
        (Int32.float_of_bits (Int32.of_int w.frac_at))
        w.frac_at;
      exit 1);
    (kind, w)
  in
  K.write_record (List.map check K.f32_bounds);
  Printf.printf "wrote %s\n" K.record
