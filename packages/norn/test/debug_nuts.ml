let f64 = Nx.float64
let true_mean = Nx.create f64 [| 2 |] [| 3.0; -1.0 |]
let true_cov = Nx.create f64 [| 2; 2 |] [| 1.0; 0.5; 0.5; 2.0 |]
let cov_inv = Nx.inv true_cov

let log_prob x =
  let d = Nx.sub x true_mean in
  let dt = Nx.reshape [| 1; 2 |] d in
  let mahal = Nx.matmul (Nx.matmul dt cov_inv) (Nx.reshape [| 2; 1 |] d) in
  Nx.mul_s (Nx.reshape [||] mahal) (-0.5)

let compute_stats name positions n =
  let nf = Float.of_int n in
  let sum0 = ref 0.0 in
  let sum1 = ref 0.0 in
  for i = 0 to n - 1 do
    let x0, x1 = positions.(i) in
    sum0 := !sum0 +. x0;
    sum1 := !sum1 +. x1
  done;
  let mean0 = !sum0 /. nf in
  let mean1 = !sum1 /. nf in
  let var0 = ref 0.0 in
  let var1 = ref 0.0 in
  for i = 0 to n - 1 do
    let x0, x1 = positions.(i) in
    var0 := !var0 +. ((x0 -. mean0) *. (x0 -. mean0));
    var1 := !var1 +. ((x1 -. mean1) *. (x1 -. mean1))
  done;
  let var0 = !var0 /. nf in
  let var1 = !var1 /. nf in
  Printf.printf "%s:\n" name;
  Printf.printf "  mean = [%.4f, %.4f]  (true: [3.0, -1.0])\n" mean0 mean1;
  Printf.printf "  var  = [%.4f, %.4f]  (true: [1.0, 2.0])\n" var0 var1;
  Printf.printf "  var error = [%.1f%%, %.1f%%]\n"
    (100.0 *. Float.abs (var0 -. 1.0) /. 1.0)
    (100.0 *. Float.abs (var1 -. 2.0) /. 2.0)

let () =
  Printf.printf "=== Comparison: Norn vs BlackJAX ===\n\n";
  Printf.printf "Target: 2D Gaussian, mean=[3,-1], cov=[[1,0.5],[0.5,2]]\n\n";

  (* --- HMC: match BlackJAX: 1000 warmup + 3000 samples, fixed --- *)
  Nx.Rng.run ~seed:42 (fun () ->
      let metric = Norn.unit_metric 2 in
      let kernel = Norn.hmc_kernel ~step_size:0.1 ~num_leapfrog:20 ~metric () in
      let init = Nx.zeros f64 [| 2 |] in
      let state = ref (kernel.init init log_prob) in
      for _ = 1 to 1000 do
        let s, _ = kernel.step !state log_prob in
        state := s
      done;
      let positions = Array.make 3000 (0.0, 0.0) in
      for i = 0 to 2999 do
        let s, _ = kernel.step !state log_prob in
        state := s;
        positions.(i) <- (Nx.item [ 0 ] s.position, Nx.item [ 1 ] s.position)
      done;
      compute_stats "Norn HMC (fixed eps=0.1, 1000w+3000s)" positions 3000;
      Printf.printf
        "  BlackJAX: mean=[3.0048, -0.9621] var=[0.9896, 2.0819]\n\n");

  (* --- NUTS: match BlackJAX: 1000 warmup + 3000 samples, fixed --- *)
  Nx.Rng.run ~seed:42 (fun () ->
      let metric = Norn.unit_metric 2 in
      let kernel = Norn.nuts_kernel ~step_size:0.1 ~metric () in
      let init = Nx.zeros f64 [| 2 |] in
      let state = ref (kernel.init init log_prob) in
      for _ = 1 to 1000 do
        let s, _ = kernel.step !state log_prob in
        state := s
      done;
      let positions = Array.make 3000 (0.0, 0.0) in
      let total_lf = ref 0 in
      for i = 0 to 2999 do
        let s, info = kernel.step !state log_prob in
        state := s;
        total_lf := !total_lf + info.num_integration_steps;
        positions.(i) <- (Nx.item [ 0 ] s.position, Nx.item [ 1 ] s.position)
      done;
      compute_stats "Norn NUTS (fixed eps=0.1, 1000w+3000s)" positions 3000;
      Printf.printf "  avg leapfrog/step = %.1f\n"
        (Float.of_int !total_lf /. 3000.0);
      Printf.printf
        "  BlackJAX: mean=[3.0297, -0.9192] var=[1.0453, 2.1354]\n\n");

  (* --- NUTS adapted: warmup trace + 100 samples --- *)
  Nx.Rng.run ~seed:42 (fun () ->
      let report ~step (_state : Norn.state) (info : Norn.info) =
        if step < 0 then begin
          let ws = 1000 + step + 1 in
          if ws <= 10 || ws mod 100 = 0 then
            Printf.printf "  warmup %4d: accept=%.4f lf=%d\n" ws
              info.acceptance_rate info.num_integration_steps
        end
      in
      let result =
        Norn.sample ~step_size:0.1 ~target_accept:0.80 ~num_warmup:1000 ~report
          ~n:100 log_prob (Nx.zeros f64 [| 2 |]) (fun ~step_size ~metric ->
            Norn.nuts_kernel ~step_size ~metric ())
      in
      Printf.printf "\nAdapted: step_size=%.6f accept=%.4f divergent=%d\n"
        result.stats.step_size result.stats.accept_rate
        result.stats.num_divergent)
