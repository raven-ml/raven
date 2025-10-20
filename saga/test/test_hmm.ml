open Alcotest
module Hmm = Saga_models.Hmm

let simple_hmm () =
  let init = [| 0.6; 0.4 |] in
  let transitions = [| [| 0.7; 0.3 |]; [| 0.4; 0.6 |] |] in
  let emissions = [| [| 0.5; 0.5 |]; [| 0.1; 0.9 |] |] in
  Hmm.create ~init ~transitions ~emissions

let test_forward_backward () =
  let hmm = simple_hmm () in
  let obs = [| 0; 1; 0 |] in
  let alpha = Hmm.forward hmm obs in
  let beta = Hmm.backward hmm obs in
  check int "forward length" 3 (Array.length alpha);
  check int "backward length" 3 (Array.length beta);
  Array.iter
    (fun row ->
      check bool "forward normalized" true
        (abs_float (Array.fold_left ( +. ) 0.0 row -. 1.0) < 1e-6))
    alpha;
  Array.iter
    (fun row ->
      check bool "backward normalized" true
        (abs_float (Array.fold_left ( +. ) 0.0 row -. 1.0) < 1e-6))
    beta

let test_log_likelihood () =
  let hmm = simple_hmm () in
  let obs = [| 0; 1; 0 |] in
  let ll = Hmm.log_likelihood hmm obs in
  check bool "log likelihood negative" true (ll < 0.0)

let test_viterbi () =
  let hmm = simple_hmm () in
  let obs = [| 0; 1; 1; 0 |] in
  let path = Hmm.viterbi hmm obs in
  check int "path length" (Array.length obs) (Array.length path);
  Array.iter
    (fun state ->
      check bool "valid state" true (state >= 0 && state < Hmm.num_states hmm))
    path

let test_baum_welch () =
  let init = [| 0.5; 0.5 |] in
  let transitions = [| [| 0.5; 0.5 |]; [| 0.5; 0.5 |] |] in
  let emissions = [| [| 0.5; 0.5 |]; [| 0.5; 0.5 |] |] in
  let hmm = Hmm.create ~init ~transitions ~emissions in
  let training = [ [| 0; 1; 1; 0 |]; [| 0; 1; 0 |]; [| 1; 1; 0 |] ] in
  let trained = Hmm.baum_welch hmm training in
  let ll_before =
    List.fold_left
      (fun acc seq -> acc +. Hmm.log_likelihood hmm seq)
      0.0 training
  in
  let ll_after =
    List.fold_left
      (fun acc seq -> acc +. Hmm.log_likelihood trained seq)
      0.0 training
  in
  check bool "likelihood improved" true (ll_after >= ll_before -. 1e-6)

let tests =
  [
    test_case "forward/backward" `Quick test_forward_backward;
    test_case "log-likelihood" `Quick test_log_likelihood;
    test_case "viterbi" `Quick test_viterbi;
    test_case "baum-welch" `Quick test_baum_welch;
  ]

let () = Alcotest.run "saga hmm" [ ("hmm", tests) ]
