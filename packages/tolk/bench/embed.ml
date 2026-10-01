(* Writes an OCaml module that holds the goldens named on the command line:
   [all] is each one's name, its file's without the extension, and its graph's
   text. *)

let () =
  print_string "let all =\n  [\n";
  for k = 1 to Array.length Sys.argv - 1 do
    let file = Sys.argv.(k) in
    Printf.printf "    (%S, %S);\n"
      (Filename.remove_extension (Filename.basename file))
      (Golden.body file)
  done;
  print_string "  ]\n"
