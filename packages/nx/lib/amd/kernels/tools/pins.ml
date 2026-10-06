(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Checks the kernels against the digests gen.py pinned, with no toolchain:

     pins.exe PINS FILE...

   Every file is pinned in PINS's "inputs" or "outputs" with its SHA-256, and
   every pinned file is given: a source changed without regenerating, a code
   object edited, added or removed, fails. *)

module Firmware = Nx_device_support.Firmware

(* The [path: digest] entries of PINS's "inputs" and "outputs" objects, one to a
   line as gen.py writes them: [  "src/common.h": "9f86...",]. *)
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

let () =
  match Array.to_list Sys.argv with
  | _ :: pins :: files ->
      (* dune spells a file of the rule's own directory [./gen.py]. *)
      let relative f =
        if String.starts_with ~prefix:"./" f then
          String.sub f 2 (String.length f - 2)
        else f
      in
      let files =
        List.filter_map
          (fun f -> if f = pins then None else Some (relative f))
          files
      in
      let pinned = pinned pins in
      let errors =
        List.filter_map
          (fun f ->
            match List.assoc_opt f pinned with
            | None -> Some (f ^ " is not pinned: regenerate with gen.py")
            | Some d when d <> Firmware.sha256 (In_channel.with_open_bin f In_channel.input_all) ->
                Some (f ^ " differs from its pin: regenerate with gen.py")
            | Some _ -> None)
          files
        @ List.filter_map
            (fun (f, _) ->
              if List.mem f files then None
              else Some (f ^ " is pinned but missing: regenerate with gen.py"))
            pinned
      in
      List.iter prerr_endline errors;
      if errors <> [] then exit 1
  | _ ->
      prerr_endline "usage: pins.exe PINS FILE...";
      exit 2
