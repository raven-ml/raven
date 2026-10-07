(*--------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  --------------------------------------------------------------------------*)

module Decoder = Compress_deflate.Decoder

(* Reads [input] into one buffer and writes the data to [output] as the decoder
   returns it, so memory stays bounded whatever the file sizes. *)
let decompress input output =
  let d = Compress_deflate.Gzip.decoder () in
  let buf = Bytes.create 65536 in
  let rec loop () =
    match Decoder.decode d with
    | `Await ->
        Decoder.src d buf 0 (Unix.read input buf 0 (Bytes.length buf));
        loop ()
    | `Data (b, first, length) ->
        ignore (Unix.write output b first length);
        loop ()
    | `End -> ()
    | `Error msg -> failwith ("invalid gzip file: " ^ msg)
  in
  loop ()

let gunzip ~src ~dst =
  let input = Unix.openfile src [ Unix.O_RDONLY ] 0 in
  Fun.protect ~finally:(fun () -> Unix.close input) @@ fun () ->
  Temp_file.write ~overwrite:true dst (decompress input)
