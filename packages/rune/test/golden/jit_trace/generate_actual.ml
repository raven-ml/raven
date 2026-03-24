(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Generates .actual files for JIT trace golden tests. Each file contains the
   rendered C source from tracing a function through the JIT capture handler via
   [Jit.trace_graph]. Dune diff rules compare .actual against .expected
   (generated from the reference tinygrad pipeline). *)

let write_actual dir name content =
  let filename = Filename.concat dir (name ^ ".actual") in
  let oc = open_out filename in
  output_string oc content;
  output_char oc '\n';
  close_out oc

let trace_source f x =
  let traced = Rune.trace_graph f x in
  String.concat "\n---\n" traced.rendered_source

(* ── Test cases ── *)

(* c = a + scalar(1.0), shape [256] *)
let trace_add_const () =
  let x = Nx.full Nx.float32 [| 256 |] 0.0 in
  trace_source (fun x -> Nx.add x (Nx.scalar Nx.float32 1.0)) x

(* c = a * a, shape [256] *)
let trace_mul_self () =
  let x = Nx.full Nx.float32 [| 256 |] 0.0 in
  trace_source (fun x -> Nx.mul x x) x

(* c = sum(a), shape [256] -> scalar *)
let trace_sum () =
  let x = Nx.full Nx.float32 [| 256 |] 0.0 in
  trace_source (fun x -> Nx.sum x) x

(* c = (a + scalar(1.0)) * scalar(2.0), shape [256] *)
let trace_chain () =
  let x = Nx.full Nx.float32 [| 256 |] 0.0 in
  trace_source
    (fun x ->
      Nx.mul (Nx.add x (Nx.scalar Nx.float32 1.0)) (Nx.scalar Nx.float32 2.0))
    x

type test_case = { name : string; generate : unit -> string }

let test_cases =
  [
    { name = "add_const"; generate = trace_add_const };
    { name = "mul_self"; generate = trace_mul_self };
    { name = "sum"; generate = trace_sum };
    { name = "chain"; generate = trace_chain };
  ]

let () =
  let dir = Sys.argv.(1) in
  let failed = ref false in
  List.iter
    (fun { name; generate } ->
      match generate () with
      | out -> write_actual dir name out
      | exception exn ->
          Printf.eprintf "FAIL %s: %s\n%!" name (Printexc.to_string exn);
          Printexc.print_backtrace stderr;
          failed := true)
    test_cases;
  if !failed then exit 1
