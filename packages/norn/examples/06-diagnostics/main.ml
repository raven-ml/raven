(* Diagnostics and findings.

   Diagnostics of draws are values of the draws' structure: one R-hat, one
   effective sample size, one Monte Carlo error per element. A summary turns
   them, and the transitions' statistics, into findings: data a program can act
   on. *)

let f64 = Nx.float64
let t = Nx.Ptree.tensor

(* Two well-separated modes at -4 and 4: chains that start in different modes
   never meet. Positions have shape [chains; 1]. *)
let bimodal x =
  let lp m = Nx.mul_s (Nx.square (Nx.sub_s x m)) (-0.5) in
  Nx.logsumexp ~axes:[ 1 ] (Nx.concatenate ~axis:1 [ lp (-4.); lp 4. ])

let run start =
  let key = Nx.Rng.key 0 in
  let s = Norn.Nuts.init t bimodal start in
  let s = Norn.Nuts.warmup t bimodal key ~steps:100 s in
  let _, draws, stats = Norn.Nuts.sample t bimodal key ~draws:200 s in
  (draws, stats)

let () =
  (* Four chains, all in one mode: they agree, but on half the target. *)
  let draws, stats = run (Nx.full f64 [| 4; 1 |] 4.) in
  Printf.printf "one mode:  R-hat %.3f, ESS %.0f, mean %.2f\n"
    (Nx.item [ 0 ] (Norn.Diag.rhat t draws))
    (Nx.item [ 0 ] (Norn.Diag.ess_bulk t draws))
    (Nx.item [] (Nx.mean (draws :> Nx.float64_t)));
  Printf.printf "           E-BFMI per chain: %s\n"
    (String.concat " "
       (List.map (Printf.sprintf "%.2f")
          (Array.to_list (Nx.to_array (Norn.Diag.ebfmi stats)))));

  (* Two chains in each mode: R-hat sees that they disagree. *)
  let start = Nx.create f64 [| 4; 1 |] [| -4.; -4.; 4.; 4. |] in
  let draws, stats = run start in
  Printf.printf "two modes: R-hat %.3f, ESS %.0f, mean %.2f\n"
    (Nx.item [ 0 ] (Norn.Diag.rhat t draws))
    (Nx.item [ 0 ] (Norn.Diag.ess_bulk t draws))
    (Nx.item [] (Nx.mean (draws :> Nx.float64_t)));

  (* The summary's findings, as values and as sentences. *)
  let summary = Norn.Summary.v t ~stats draws in
  Format.printf "%a@." Norn.Summary.pp summary;
  List.iter
    (function
      | Norn.Summary.Rhat_high { rhat; _ } ->
          Printf.printf "matched Rhat_high: %.2f\n" rhat
      | f -> Format.printf "other finding: %a@." Norn.Summary.pp_finding f)
    (Norn.Summary.findings summary)
