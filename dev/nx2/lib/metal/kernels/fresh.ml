(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* fresh METALLIB SOURCE...: exits 0 if build.sh made METALLIB from SOURCES,
   in build.sh's order, and it defines every kernel kernels.ml names; fails
   naming build.sh otherwise. *)

let () =
  match Array.to_list Sys.argv with
  | _ :: metallib :: sources -> (
      let m = In_channel.with_open_bin metallib In_channel.input_all in
      let kernels = Array.to_list (Array.map fst Kernels.kernels) in
      match Stamp.stale m ~digest:(Stamp.digest sources) ~kernels with
      | None -> ()
      | Some why ->
          Printf.eprintf
            "dev/nx2/lib/metal/kernels/kernels.metallib %s: rebuild it with \
             dev/nx2/lib/metal/kernels/build.sh\n"
            why;
          exit 1)
  | _ ->
      prerr_endline "usage: fresh METALLIB SOURCE...";
      exit 2
