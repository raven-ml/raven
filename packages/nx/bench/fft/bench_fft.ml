(* Public FFT performance regimes: native power-of-two, native mixed-radix,
   prime-size Bluestein, and the real transforms (rfft/irfft) whose last axis
   takes its own packed path. Setup and dtype conversion are outside timing. *)

let case name n =
  let input = Nx.cast Nx.Complex64 (Nx.rand Nx.Float32 [| n |]) in
  Thumper.bench name (fun () -> Nx.fft input)

let rcase name n =
  let input = Nx.rand Nx.Float64 [| n |] in
  Thumper.bench name (fun () -> Nx.rfft Nx.complex128 input)

let icase name n =
  let spectrum = Nx.rfft Nx.complex128 (Nx.rand Nx.Float64 [| n |]) in
  Thumper.bench name (fun () -> Nx.irfft Nx.float64 ~n spectrum)

let icase_batched name lines n =
  let spectrum = Nx.rfft Nx.complex128 (Nx.rand Nx.Float64 [| lines; n |]) in
  Thumper.bench name (fun () -> Nx.irfft Nx.float64 ~n spectrum)

let () =
  Nx.Rng.with_key (Nx.Rng.key 42) @@ fun () ->
  Thumper.run "nx_fft"
    ~budgets:
      [
        Thumper.Budget.no_slower_than ~metric:Thumper.Metric.wall_time 0.05;
        Thumper.Budget.no_more_alloc_than 0.01;
      ]
    [
      Thumper.group "fft"
        [
          case "c64 65536 power-of-two" 65536;
          case "c64 100000 mixed-radix" 100000;
          case "c64 65521 prime" 65521;
          case "c64 4099 bluestein-smooth-m" 4099;
        ];
      Thumper.group "rfft"
        [
          rcase "f64 65536 power-of-two" 65536;
          rcase "f64 44100 smooth" 44100;
          rcase "f64 65535 odd-control" 65535;
          rcase "f64 131042 half-prime" 131042;
        ];
      Thumper.group "irfft"
        [
          icase "c64 65536 power-of-two" 65536;
          icase_batched "c64 256x4097 batched" 256 8192;
        ];
    ]
