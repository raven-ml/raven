open Alcotest
module Pcfg = Saga_models.Pcfg

let simple_grammar () =
  Pcfg.create ~start:0 ~num_nonterminals:2 ~num_terminals:2
    [
      { lhs = 0; rhs = Binary (0, 1); prob = 0.4 };
      { lhs = 0; rhs = Unary 0; prob = 0.6 };
      { lhs = 1; rhs = Unary 1; prob = 1.0 };
    ]

let test_inside_probability () =
  let grammar = simple_grammar () in
  let sentence = [| 0; 1 |] in
  let chart = Pcfg.inside grammar sentence in
  let total = chart.(0).(2).(Pcfg.start_symbol grammar) in
  check bool "probability positive" true (total > 0.0)

let test_log_probability () =
  let grammar = simple_grammar () in
  let sentence = [| 0; 1 |] in
  let ll = Pcfg.log_probability grammar sentence in
  check bool "log probability finite" true (Float.is_finite ll)

let test_viterbi () =
  let grammar = simple_grammar () in
  let sentence = [| 0; 1 |] in
  let chart = Pcfg.viterbi grammar sentence in
  let cell = chart.(0).(Array.length sentence) in
  let k, _, _ = cell.(Pcfg.start_symbol grammar) in
  check bool "valid split" true (k > 0)

let test_inside_outside () =
  let grammar = simple_grammar () in
  let sentences = [ [| 0; 1 |]; [| 0 |] ] in
  let trained = Pcfg.inside_outside grammar sentences in
  let ll_after =
    List.fold_left
      (fun acc s -> acc +. Pcfg.log_probability trained s)
      0.0 sentences
  in
  check bool "likelihood finite" true (Float.is_finite ll_after)

let tests =
  [
    test_case "inside" `Quick test_inside_probability;
    test_case "log_probability" `Quick test_log_probability;
    test_case "viterbi" `Quick test_viterbi;
    test_case "inside_outside" `Quick test_inside_outside;
  ]

let () = Alcotest.run "saga pcfg" [ ("pcfg", tests) ]
