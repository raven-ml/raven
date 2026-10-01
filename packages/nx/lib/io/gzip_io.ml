(*--------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  --------------------------------------------------------------------------*)

open Bytesrw

let gunzip ~src ~dst =
  let input = Unix.openfile src [ Unix.O_RDONLY ] 0 in
  Fun.protect ~finally:(fun () -> Unix.close input) @@ fun () ->
  let temp = Temp_file.sibling dst in
  match
    let output = Unix.openfile temp [ Unix.O_WRONLY; Unix.O_TRUNC ] 0 in
    Fun.protect
      ~finally:(fun () -> Unix.close output)
      (fun () ->
        let r = Bytesrw_unix.bytes_reader_of_fd input in
        let w = Bytesrw_unix.bytes_writer_of_fd output in
        try
          Bytes.Writer.write_reader ~eod:true w
            (Compress_deflate.Gzip.decompress_reads () r)
        with Bytes.Stream.Error e -> failwith (Bytes.Stream.error_message e))
  with
  | () -> Temp_file.replace temp dst
  | exception exn ->
      Temp_file.remove_if_exists temp;
      raise exn
