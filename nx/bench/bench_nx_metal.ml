(* bench_metal.ml - Metal backend benchmarks *)

module Runner = Bench_nx.Make (Nx_metal)

let () =
  Printf.printf "Nx Metal Backend Benchmarking\n";
  Printf.printf "=============================\n\n";

  let ctx = Nx_metal.create_context () in
  let _backend_name, results = Runner.run ~backend_name:"Metal" ctx in

  (* Print some basic statistics *)
  Printf.printf "\nCompleted %d benchmarks\n" (List.length results)
