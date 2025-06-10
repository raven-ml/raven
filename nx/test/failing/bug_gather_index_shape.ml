(* Test for gather operation with incorrect index tensor shape *)
open Nx

let test_gather_index_shape () =
  Printf.printf "Testing gather operation index shape bug\n";
  
  (* Create a 3D tensor *)
  let data = arange float32 0 24 1 |> reshape [|2; 3; 4|] in
  Printf.printf "Data shape: [2; 3; 4]\n";
  
  (* Try to gather indices [0, 2] along axis 1 *)
  (* This should select the 0th and 2nd elements along dimension 1 *)
  try
    let result = slice [R []; L [0; 2]; R []] data in
    Printf.printf "Result shape: ";
    Array.iter (Printf.printf "%d ") (shape result);
    Printf.printf "\n";
    Printf.printf "Expected shape: [2; 2; 4]\n";
    
    (* Verify the result *)
    let expected_shape = [|2; 2; 4|] in
    if shape result = expected_shape then
      Printf.printf "PASS: Gather produced correct shape\n"
    else (
      Printf.printf "FAIL: Gather produced incorrect shape\n";
      Printf.printf "Got: [";
      Array.iter (Printf.printf "%d ") (shape result);
      Printf.printf "]\n"
    )
  with
  | Invalid_argument msg ->
      Printf.printf "FAIL: Gather raised exception: %s\n" msg;
      Printf.printf "This is the bug - gather creates multi-dimensional index tensor instead of 1D\n"
  | e ->
      Printf.printf "Unexpected exception: %s\n" (Printexc.to_string e)

let () = test_gather_index_shape ()