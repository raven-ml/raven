let () =
  match Sys.argv with
  | [| _; spec |] ->
      set_binary_mode_out stdout true;
      print_string (Lz_support.raw_of_spec spec)
  | _ ->
      prerr_endline "usage: raw.exe RAW";
      exit 2
