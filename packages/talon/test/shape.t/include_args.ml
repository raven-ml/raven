(* Prints the -I flags of the libraries ../includes lists, one argument per line
   for ocamlc -args. Each line of ../includes names an interface's path in an
   install tree, then its path as dune resolved it, relative to ../ if not
   absolute. The interface is read from OCAMLPATH if it holds it, otherwise from
   its resolved path. OCaml splits and joins the paths, as the shell cannot on
   Windows, where OCAMLPATH is separated by ';', paths start with a drive and
   may use '\', and native programs read no Cygwin path. *)

let path_sep = if Sys.win32 then ';' else ':'

let ocamlpath =
  match Sys.getenv_opt "OCAMLPATH" with
  | None -> []
  | Some p -> List.filter (fun d -> d <> "") (String.split_on_char path_sep p)

let dir cmi resolved =
  let installed d = Sys.file_exists (Filename.concat d cmi) in
  match List.find_opt installed ocamlpath with
  | Some d -> Filename.dirname (Filename.concat d cmi)
  | None when Filename.is_relative resolved ->
      Filename.dirname (Filename.concat Filename.parent_dir_name resolved)
  | None -> Filename.dirname resolved

let () =
  let lines = In_channel.with_open_text "../includes" In_channel.input_lines in
  List.iter
    (fun line ->
      match String.index_opt line ' ' with
      | None -> ()
      | Some i ->
          let cmi = String.sub line 0 i in
          let resolved = String.sub line (i + 1) (String.length line - i - 1) in
          Printf.printf "-I\n%s\n" (dir cmi resolved))
    lines
