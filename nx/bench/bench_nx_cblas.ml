(* bench_cblas.ml - CBLAS backend benchmarks *)

module Runner = Bench_nx.Make (Nx_cblas)

let () =
  Printf.printf "Nx CBLAS Backend Benchmarking\n";
  Printf.printf "=============================\n\n";
  
  let ctx = Nx_cblas.create_context () in
  let _backend_name, results = Runner.run ~backend_name:"CBLAS" ctx in
  
  (* Print some basic statistics *)
  Printf.printf "\nCompleted %d benchmarks\n" (List.length results)