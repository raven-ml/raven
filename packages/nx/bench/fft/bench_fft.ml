(* Public FFT performance regimes: native power-of-two, native mixed-radix,
   and prime-size Bluestein. Setup and dtype conversion are outside timing. *)

let case name n =
  let input = Nx.cast Nx.Complex64 (Nx.rand Nx.Float32 [| n |]) in
  Thumper.bench name (fun () -> Nx.fft input)

let () =
  Nx.Rng.run ~seed:42 @@ fun () ->
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
        ];
    ]
