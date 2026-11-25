(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Writes the PNG baselines of the corpus pages, at a density of 2, to the
   directory given on the command line. test_raster compares its rendering with
   them within one colour level, which an [expect_file] baseline cannot
   state. *)

let write path data =
  Out_channel.with_open_bin path (fun oc -> Out_channel.output_string oc data)

let () =
  let dir = Sys.argv.(1) in
  List.iter
    (fun (name, r) ->
      write
        (Filename.concat dir (name ^ ".png"))
        (Hugin_vg_raster.png ~density:2. r))
    Vg_corpus.pages
