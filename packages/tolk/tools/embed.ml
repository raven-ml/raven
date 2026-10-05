(* Writes an OCaml module whose [program] is the bytes of the file given on the
   command line. *)

let () =
  Printf.printf "let program = %S\n"
    (In_channel.with_open_bin Sys.argv.(1) In_channel.input_all)
