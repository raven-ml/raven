(* bench_native.ml - Native backend benchmarks *)

module Runner = Bench_nx.Make (Nx_native)

let () =
  Printf.printf "Nx Native Backend Benchmarking\n";
  Printf.printf "==============================\n\n";
  
  let ctx = Nx_native.create_context () in
  let _backend_name, results = Runner.run ~backend_name:"Native" ctx in
  
  (* Print some basic statistics *)
  Printf.printf "\nCompleted %d benchmarks\n" (List.length results)