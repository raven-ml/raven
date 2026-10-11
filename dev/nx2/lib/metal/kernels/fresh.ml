(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* fresh METALLIB SOURCE...: exits 0 if build.sh made METALLIB from SOURCES,
   in build.sh's order; fails naming build.sh otherwise. *)

let () =
  match Array.to_list Sys.argv with
  | _ :: metallib :: sources ->
      let m = In_channel.with_open_bin metallib In_channel.input_all in
      if not (Stamp.has m (Stamp.stamp (Stamp.digest sources))) then begin
        Printf.eprintf
          "%s was built from other sources: rebuild it with \
           dev/nx2/lib/metal/kernels/build.sh\n"
          metallib;
        exit 1
      end
  | _ ->
      prerr_endline "usage: fresh METALLIB SOURCE...";
      exit 2
