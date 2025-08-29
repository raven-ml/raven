(* Bug: Slice batch processing with mixed specs can cause bounds issues *)

let () =
  Printf.printf "Testing slice batch processing bounds bug\n";

  let x = Nx.create Nx.float32 [| 3; 4; 5 |] (Array.init 60 float_of_int) in

  Printf.printf "Input shape: [%s]\n"
    (String.concat "; " (Array.to_list (Array.map string_of_int (Nx.shape x))));

  (* Complex slice that triggers batch processing with potential bounds issues:
     - Take indices [0; 2] from dimension 0 - Then slice index 1 from dimension
     1 - Then slice range [1, 3) from dimension 2 *)
  try
    (* First take indices [0, 2] along axis 0 *)
    let indices0 = Nx.create Nx.int32 [| 2 |] [| 0l; 2l |] in
    let step1 = Nx.take ~axis:0 indices0 x in

    (* Then slice to get only index 1 along axis 1 *)
    let step2 =
      Nx.slice [ Nx.Rs (0, 2, 1); Nx.Rs (1, 2, 1); Nx.Rs (0, 5, 1) ] step1
    in

    (* Finally slice range [1, 3) along axis 2 *)
    let result =
      Nx.slice [ Nx.Rs (0, 2, 1); Nx.Rs (0, 1, 1); Nx.Rs (1, 3, 1) ] step2
    in
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
