(* §7.12: a program linking lz whole (-linkall), whose size less size_base.exe's
   is lz's share of a static OCaml executable. *)

let () = print_string (string_of_int (Lz.Crc32.string ""))
