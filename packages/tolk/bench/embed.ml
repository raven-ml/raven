(* Writes an OCaml module that holds the graphs named on the command line:
   [all] is each one's name, its file's without the extension, and its graph's
   text, a [.graph] file's contents or what [Golden.body] reads for a golden
   or its [.tolk]. A golden and its [.tolk] both named make one entry. *)

let () =
  print_string "let all =\n  [\n";
  let seen = Hashtbl.create 16 in
  for k = 1 to Array.length Sys.argv - 1 do
    let file = Sys.argv.(k) in
    let stem = Filename.remove_extension file in
    let name = Filename.basename stem in
    if not (Hashtbl.mem seen name) then begin
      Hashtbl.add seen name ();
      Printf.printf "    (%S, %S);\n" name
        (if Filename.check_suffix file ".graph" then
           In_channel.with_open_bin file In_channel.input_all
         else Golden.body (stem ^ ".golden"))
    end
  done;
  print_string "  ]\n"
