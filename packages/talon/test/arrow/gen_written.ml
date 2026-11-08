(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Writes the files of [Arrow_corpus] into the directory given, for
   [gen/written.py] to read with pyarrow. *)

let () =
  let dir = Sys.argv.(1) in
  List.iter
    (fun (it : Arrow_corpus.item) ->
      Out_channel.with_open_bin (Filename.concat dir it.name) (fun oc ->
          output_string oc (Arrow_corpus.bytes it)))
    Arrow_corpus.items
