(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* parity.exe PART: AION-1's released weights against upstream's code, one part
   at a time, on the directory AION_PARITY names, laid out as the README's Tests
   section says. Each part's module says what it checks:

   - spectrum, image: the codecs against their rounding ensembles (Codec);

   - mmu: Survey's and Codecs' tokens of real rows (Mmu);

   - generate MAIN: main.exe's generate against the library (Generate);

   - gpu [DEVICE]: the codecs and the transformer compiled on a GPU (Gpu), under
   an alias of its own.

   A part prints its checks and exits 1 when one fails, or when an input is
   missing. *)

let () =
  match Array.to_list Sys.argv with
  | [ _; "spectrum" ] -> Codec.spectrum ()
  | [ _; "image" ] -> Codec.image ()
  | [ _; "mmu" ] -> Mmu.run ()
  | [ _; "generate"; main ] -> Generate.run main
  | [ _; "gpu" ] -> Gpu.run None
  | [ _; "gpu"; device ] -> Gpu.run (Some device)
  | _ ->
      prerr_endline
        "usage: parity.exe (spectrum | image | mmu | generate MAIN | gpu \
         [DEVICE])";
      exit 2
