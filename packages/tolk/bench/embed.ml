(* Writes an OCaml module that holds the graphs named on the command line:
   [all] is each one's name, its file's without the extension, and its graph's
   text, a golden's body or a [.graph] file's contents. *)

let () =
  print_string "let all =\n  [\n";
  for k = 1 to Array.length Sys.argv - 1 do
    let file = Sys.argv.(k) in
    Printf.printf "    (%S, %S);\n"
      (Filename.remove_extension (Filename.basename file))
      (if Filename.check_suffix file ".graph" then
         In_channel.with_open_bin file In_channel.input_all
       else Golden.body file)
  done;
  print_string "  ]\n"
