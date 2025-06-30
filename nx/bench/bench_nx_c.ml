(* bench_c.ml - C backend benchmarks *)

module Runner = Bench_nx.Make (Nx_c)

let () =
  Printf.printf "Nx C Backend Benchmarking\n";
  Printf.printf "=============================\n\n";

  let ctx = Nx_c.create_context () in
  let _backend_name, results = Runner.run ~backend_name:"C" ctx in

  (* Print some basic statistics *)
  Printf.printf "\nCompleted %d benchmarks\n" (List.length results)
