(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Checks a kernel library's artifacts against the digests its gen.py
   pinned, with no toolchain:

     pins.exe PINS DIR FILE...

   PINS names files by their path from dev/nx2; DIR is the directory the
   FILEs are relative to, from dev/nx2. Every FILE is pinned in PINS's
   "inputs" or "outputs" with its BLAKE2b-256 digest, and every pinned file
   is given: a source changed without regenerating, or an artifact edited,
   added or removed, fails. *)

(* The [path: digest] entries of PINS's "inputs" and "outputs" objects, one
   to a line as gen.py writes them: [  "lib/array/nx_dtype.h": "9f86...",]. *)
let pinned file =
  let entry l =
    match String.split_on_char '"' (String.trim l) with
    | [ ""; path; ": "; digest; ("" | ",") ] -> Some (path, digest)
    | _ -> None
  in
  let section = ref false in
  In_channel.with_open_text file In_channel.input_lines
  |> List.filter_map (fun l ->
         match String.trim l with
         | {|"inputs": {|} | {|"outputs": {|} ->
             section := true;
             None
         | "}" | "}," ->
             section := false;
             None
         | _ -> if !section then entry l else None)

(* [dir/f] with its [.] and [..] segments resolved. *)
let normalize dir f =
  let step acc = function
    | "" | "." -> acc
    | ".." -> ( match acc with _ :: up -> up | [] -> [])
    | s -> s :: acc
  in
  String.split_on_char '/' (dir ^ "/" ^ f)
  |> List.fold_left step [] |> List.rev |> String.concat "/"

let digest f =
  Digest.BLAKE256.to_hex
    (Digest.BLAKE256.string (In_channel.with_open_bin f In_channel.input_all))

let () =
  match Array.to_list Sys.argv with
  | _ :: pins :: dir :: files ->
      let files = List.filter (fun f -> f <> pins) files in
      let pinned = pinned pins in
      let given = List.map (fun f -> (normalize dir f, f)) files in
      let errors =
        List.filter_map
          (fun (path, f) ->
            match List.assoc_opt path pinned with
            | None -> Some (path ^ " is not pinned: regenerate with gen.py")
            | Some d when d <> digest f ->
                Some (path ^ " differs from its pin: regenerate with gen.py")
            | Some _ -> None)
          given
        @ List.filter_map
            (fun (path, _) ->
              if List.mem_assoc path given then None
              else Some (path ^ " is pinned but missing: regenerate with gen.py"))
            pinned
      in
      List.iter prerr_endline errors;
      if errors <> [] then exit 1
  | _ ->
      prerr_endline "usage: pins.exe PINS DIR FILE...";
      exit 2
