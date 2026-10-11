(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* fresh CUBIN SOURCE... checks the source stamp build.sh compiled into the
   cubin. Hash the files in build.sh's order, as lines of hex MD5 digests. *)

let digest files =
  let line f = Digest.to_hex (Digest.file f) ^ "\n" in
  Digest.to_hex (Digest.string (String.concat "" (List.map line files)))

let rec at s sub i j =
  j = String.length sub || (s.[i + j] = sub.[j] && at s sub i (j + 1))

let rec contains s sub i =
  i + String.length sub <= String.length s
  && (at s sub i 0 || contains s sub (i + 1))

let () =
  match Array.to_list Sys.argv with
  | _ :: cubin :: (_ :: _ as sources) ->
      let bytes = In_channel.with_open_bin cubin In_channel.input_all in
      let name = "\000nx_cuda_sources_" ^ digest sources ^ "\000" in
      if not (contains bytes name 0) then begin
        Printf.eprintf
          "%s was built from other sources: rebuild it with \
           dev/nx2/lib/cuda/kernels/build.sh\n"
          cubin;
        exit 1
      end
  | _ ->
      prerr_endline "usage: fresh CUBIN SOURCE...";
      exit 2
