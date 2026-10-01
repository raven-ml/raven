(* Writes an OCaml module whose [digest] is the digest of the files under the
   directory given on the command line, their paths and contents, but for
   [source_digest.ml], the module it writes. *)

let rec files dir =
  Sys.readdir dir |> Array.to_list |> List.sort String.compare
  |> List.concat_map (fun name ->
      let path = Filename.concat dir name in
      if Sys.is_directory path then files path
      else if name = "source_digest.ml" then []
      else [ path ])

let () =
  let b = Buffer.create 4096 in
  List.iter
    (fun path ->
      Buffer.add_string b path;
      Buffer.add_char b '\000';
      Buffer.add_string b
        (Digest.to_hex
           (Digest.string (In_channel.with_open_bin path In_channel.input_all)));
      Buffer.add_char b '\n')
    (files Sys.argv.(1));
  Printf.printf "let digest = %S\n"
    (Digest.to_hex (Digest.string (Buffer.contents b)))
