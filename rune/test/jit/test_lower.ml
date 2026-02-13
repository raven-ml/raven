(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* test_lower.ml *)
open Windtrap
open Support

let test_lower () =
  let _a, _b, _c, graph = simple_add_graph () in
  let spec = List.hd (Rune_jit.Debug.schedule graph) in
  match
    Rune_jit.Debug.lower_kernel ~kernel_spec:spec
      ~original_graph_vars_metadata:graph.vars_metadata
  with
  | Error e -> failf "lowering error: %s" e
  | Ok ll ->
      (* we expect: 1 Buffer (output), 2 Loads, 1 ALU, 1 Store = 5 instrs *)
      equal ~msg:"instr count" int 5 (List.length ll.instructions)

let () = run "Lowerer" [ group "basic" [ test "simple add" test_lower ] ]
