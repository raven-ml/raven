(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Which sources a metallib was built from. build.sh adds to the metallib an
   empty function named after the digest of the files it compiled, so a
   metallib left from older sources lacks the current sources' name. *)

(* The digest of [files]: the MD5 of their MD5s in hex, a line each, as
   [md5 -q] prints them in build.sh. *)
let digest files =
  let line f = Digest.to_hex (Digest.file f) ^ "\n" in
  Digest.to_hex (Digest.string (String.concat "" (List.map line files)))

(* The function build.sh adds for sources of [digest]. *)
let stamp digest = "nx_metal_sources_" ^ digest

(* Whether [s] holds [sub] at [i], from its [j]th byte on. *)
let rec at s sub i j =
  j = String.length sub || (s.[i + j] = sub.[j] && at s sub i (j + 1))

(* Whether [s] holds [sub] at [i] or past it. *)
let rec contains s sub i =
  i + String.length sub <= String.length s
  && (at s sub i 0 || contains s sub (i + 1))

(* Whether [metallib] has the function [f]. Its function list names each
   function in a NAME tag: a 16-bit little-endian length, then the name and a
   NUL. *)
let has metallib f =
  let n = String.length f + 1 in
  let length = String.init 2 (fun i -> Char.chr ((n lsr (8 * i)) land 0xff)) in
  contains metallib ("NAME" ^ length ^ f ^ "\000") 0
