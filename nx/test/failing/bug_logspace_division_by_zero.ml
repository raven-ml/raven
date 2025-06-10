(* Bug: logspace with count=1 and endpoint=true causes division by zero *)

let () =
  Printf.printf "Testing logspace division by zero bug\n";

  (* When count=1 and endpoint=true, logspace computes step = (stop - start) /
     (count - 1) which results in division by zero when count=1 *)
  try
    let _ = Nx.logspace Nx.float32 ~endpoint:true 0.0 1.0 1 in
    failwith "BUG: logspace should have raised Division_by_zero exception"
  with
  | Division_by_zero ->
      Printf.printf
        "BUG CONFIRMED: logspace causes division by zero with count=1 and \
         endpoint=true\n";
      Printf.printf
        "This happens because step = (stop - start) / (count - 1) = 1.0 / 0\n"
  | e ->
      Printf.printf "Unexpected exception: %s\n" (Printexc.to_string e);
      raise e
