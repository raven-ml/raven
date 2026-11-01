(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Writes the PNG baselines of the benchmark figures to the directory given on
   the command line. test_draw compares them within one colour level, which an
   [expect_file] baseline cannot state; their SVGs are [expect_file]
   baselines. *)

open Hugin_test_figures

let write path data =
  Out_channel.with_open_bin path (fun oc -> Out_channel.output_string oc data)

let () =
  let dir = Sys.argv.(1) in
  List.iter
    (fun ((name, _, _, _) as g) ->
      let r = Hugin.Drawing.renderable (Figures.drawing g) in
      write
        (Filename.concat dir (name ^ ".png"))
        (Hugin_vg_raster.png ~density:Figures.density r))
    Figures.goldens
