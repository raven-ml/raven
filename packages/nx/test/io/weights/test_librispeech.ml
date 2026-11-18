(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* LibriSpeech files load as libsndfile reads them: for each file a list names,
   the rate, the rows and the BLAKE2b-256 of the int16 samples, little-endian in
   C order, are the list's. The dune file says how to run it. *)

open Windtrap
module Audio = Nx_io.Audio

let env k =
  match Sys.getenv_opt k with
  | Some v -> v
  | None -> failwith (k ^ " is not set: the dune file says how to run this")

let root = env "NX_IO_LIBRISPEECH"

(* The list's rows: a path under [root], its rate, rows and digest. *)
let files =
  match
    String.split_on_char '\n'
      (String.trim
         (In_channel.with_open_bin
            (env "NX_IO_LIBRISPEECH_LIST")
            In_channel.input_all))
  with
  | [] -> []
  | _ :: rows ->
      List.map
        (fun row ->
          match String.split_on_char '\t' row with
          | [ path; rate; length; digest ] ->
              (path, int_of_string rate, int_of_string length, digest)
          | _ -> failwith row)
        rows

let speech =
  cases "LibriSpeech files load as libsndfile reads them"
    ~id:(fun (p, _, _, _) -> p)
    files
    (fun (path, rate, length, digest) ->
      let a = Nx_io.load_audio (Filename.concat root path) in
      equal (triple int int int) (rate, length, 1)
        (Audio.rate a, Audio.length a, Audio.channels a);
      let x = Nx.unpack Nx.int16 (Audio.samples a) in
      let le = Bytes.create (2 * Nx.numel x) in
      Array.iteri (fun i v -> Bytes.set_int16_le le (2 * i) v) (Nx.to_array x);
      equal string digest (Digest.BLAKE256.to_hex (Digest.BLAKE256.bytes le)))

let () = exit (run "nx.io LibriSpeech" [ speech ])
