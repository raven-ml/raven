(* Bug: Slice batch processing with mixed specs can cause bounds issues *)

let () =
  Printf.printf "Testing slice batch processing bounds bug\n";

  let x = Nx.create Nx.float32 [| 3; 4; 5 |] (Array.init 60 float_of_int) in

  Printf.printf "Input shape: [%s]\n"
    (String.concat "; " (Array.to_list (Array.map string_of_int (Nx.shape x))));

  (* Complex slice that triggers batch processing with potential bounds issues:
     - L [0; 2]: list of indices for dimension 0 - I 1: single index for
     dimension 1 - R [1; 3]: range for dimension 2 *)
  let slice_spec = [ Nx.L [ 0; 2 ]; Nx.I 1; Nx.R [ 1; 3 ] ] in

  try
    let result = Nx.slice slice_spec x in
    Printf.printf "Result shape: [%s]\n"
      (String.concat "; "
         (Array.to_list (Array.map string_of_int (Nx.shape result))));
    Printf.printf "Slice succeeded - might not trigger the bug in this case\n"
  with
  | Failure msg when msg = "Index out of bounds" ->
      Printf.printf "BUG CONFIRMED: Slice batch processing failed with: %s\n"
        msg;
      Printf.printf
        "This happens when mixed slice specs cause incorrect bounds calculations\n"
  | e ->
      Printf.printf "Unexpected exception: %s\n" (Printexc.to_string e);
      raise e
