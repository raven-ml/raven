open Nx

let () =
  let t = create float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  Printf.printf "Original array: [";
  Array.iter (Printf.printf "%.0f ") (to_array t);
  Printf.printf "]\n\n";

  let view1 = slice [ Nx.R (0, 3) ] t in
  Printf.printf "view1 = slice [0] [3] (indices 0-2): [";
  Array.iter (Printf.printf "%.0f ") (to_array view1);
  Printf.printf "]\n";

  let view2 = slice [ Nx.R (2, 5) ] t in
  Printf.printf "view2 = slice [2] [5] (indices 2-4): [";
  Array.iter (Printf.printf "%.0f ") (to_array view2);
  Printf.printf "]\n\n";

  Printf.printf "Attempting blit view1 -> view2...\n";
  try
    blit view1 view2;
    Printf.printf "Result: [";
    Array.iter (Printf.printf "%.0f ") (to_array t);
    Printf.printf "]\n";
    Printf.printf "Expected: [1 2 1 2 3]\n"
  with e -> Printf.printf "Error: %s\n" (Printexc.to_string e)
