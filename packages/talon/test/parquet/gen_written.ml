(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Writes the files of [Written_corpus] into the directory given, for
   [gen/written.py] to read with pyarrow: those not committed with a second
   argument, the others without. *)

let () =
  let dir = Sys.argv.(1) and large = Array.length Sys.argv > 2 in
  List.iter
    (fun (item : Written_corpus.item) ->
      if item.large = large then
      let q = Talon.Query.of_table item.table in
      let f =
        Talon_parquet.format ~compression:item.compression
          ~row_group:item.row_group (Talon.Query.schema q)
      in
      let path = Filename.concat dir (item.name ^ ".parquet") in
      Out_channel.with_open_bin path (fun oc ->
          let w = Bytesrw.Bytes.Writer.of_out_channel oc in
          Talon.Error.get_ok (Talon_parquet.encode f q w)))
    Written_corpus.items
