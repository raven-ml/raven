(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Prints an OCaml module binding each [name file] pair on the command line to
   the file's bytes as a string literal. *)

let read_file path =
  let ic = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () -> really_input_string ic (in_channel_length ic))

let () =
  let args = List.tl (Array.to_list Sys.argv) in
  let rec bind = function
    | name :: path :: rest ->
        Printf.printf "let %s = %S\n" name (read_file path);
        bind rest
    | [] -> ()
    | [ _ ] -> failwith "embed: expected name and file pairs"
  in
  bind args
