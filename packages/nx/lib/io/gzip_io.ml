(*--------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  --------------------------------------------------------------------------*)

open Bigarray

let buffer_length = 65536

let write_run output run =
  Nx_io_codec.write_all output run ~off:0 ~len:(Array1.dim run)

(* Reads [input] into one buffer and writes each run the decoder produces to
   [output], so memory stays bounded whatever the file sizes. A read of 0 bytes
   is the input's end; once there, a step that returns [More] of the whole input
   has output still due, and the next read, of 0 bytes again, steps on. *)
let decompress input output =
  let d = Lz.Gzip.decoder () in
  let buf = Array1.create int8_unsigned c_layout buffer_length in
  let rec loop () =
    let n = Unix.read_bigarray input buf 0 buffer_length in
    let rec drain pos =
      let src = Array1.sub buf pos (n - pos) in
      match Lz.Decoder.step d ~last:(n = 0) src (write_run output) with
      | Ok (Lz.More k) when pos + k < n -> drain (pos + k)
      | Ok (Lz.More _) -> loop ()
      | Ok (Lz.Done _) -> ()
      | Error e ->
          Format.kasprintf failwith "invalid gzip file: %a" Lz.pp_error e
    in
    drain 0
  in
  loop ()

let gunzip ~src ~dst =
  let input = Unix.openfile src [ Unix.O_RDONLY ] 0 in
  Fun.protect ~finally:(fun () -> Unix.close input) @@ fun () ->
  Temp_file.write ~overwrite:true dst (decompress input)
