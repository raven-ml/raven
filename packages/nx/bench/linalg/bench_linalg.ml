(* Representative public dense-factorization benchmarks. The 256x256 cases
   guard sustained work; batched 32x32 cases guard scheduling overhead. *)

let spd shape =
  let rank = Array.length shape in
  let n = shape.(rank - 1) in
  let g_shape = Array.copy shape in
  g_shape.(rank - 1) <- 2 * n;
  let g = Nx.rand Nx.Float64 g_shape in
  let axes = Array.init rank Fun.id in
  axes.(rank - 2) <- rank - 1;
  axes.(rank - 1) <- rank - 2;
  Nx.matmul g (Nx.transpose ~axes:(Array.to_list axes) g)

let () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let matrix = Nx.rand Nx.Float64 [| 256; 256 |] in
  let positive = spd [| 256; 256 |] in
  let rhs = Nx.rand Nx.Float64 [| 256; 16 |] in
  let batch = Nx.rand Nx.Float64 [| 64; 32; 32 |] in
  let positive_batch = spd [| 64; 32; 32 |] in
  Thumper.run "nx_linalg"
    ~budgets:
      [
        Thumper.Budget.no_slower_than ~metric:Thumper.Metric.wall_time 0.05;
        Thumper.Budget.no_more_alloc_than 0.01;
      ]
    [
      Thumper.group "single"
        [
          Thumper.bench "cholesky 256x256 f64" (fun () ->
              Nx.cholesky positive);
          Thumper.bench "qr 256x256 f64" (fun () -> Nx.qr matrix);
          Thumper.bench "eigh 256x256 f64" (fun () -> Nx.eigh positive);
          Thumper.bench "svd 256x256 f64" (fun () -> Nx.svd matrix);
          Thumper.bench "solve 256x256x16 f64" (fun () ->
              Nx.solve positive rhs);
        ];
      Thumper.group "batched"
        [
          Thumper.bench "cholesky 64x32x32 f64" (fun () ->
              Nx.cholesky positive_batch);
          Thumper.bench "qr 64x32x32 f64" (fun () -> Nx.qr batch);
          Thumper.bench "eigh 64x32x32 f64" (fun () ->
              Nx.eigh positive_batch);
          Thumper.bench "svd 64x32x32 f64" (fun () -> Nx.svd batch);
        ];
    ]
