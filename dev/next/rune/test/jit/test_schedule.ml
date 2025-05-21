(* test_schedule.ml *)
open Alcotest
open Support

let test_scheduler () =
  let a, b, c, graph = simple_add_graph () in
  let specs = Rune_jit.Debug.schedule graph in
  check int "1 kernel" 1 (List.length specs);
  let s = List.hd specs in
  check (list int) "inputs" [ a; b ] s.inputs;
  check (list int) "outputs" [ c ] s.outputs

let () =
  Alcotest.run "Scheduler"
    [ ("basic", [ test_case "simple add" `Quick test_scheduler ]) ]
