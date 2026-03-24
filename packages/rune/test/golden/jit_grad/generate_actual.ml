(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Generates .actual files for grad+JIT golden tests. Each file contains the
   rendered C source from tracing grad(f) through the JIT capture handler via
   [Jit.trace_graph]. Dune diff rules compare .actual against .expected
   (generated from tinygrad's backward computation). *)

let write_actual dir name content =
  let filename = Filename.concat dir (name ^ ".actual") in
  let oc = open_out filename in
  output_string oc content;
  output_char oc '\n';
  close_out oc

let grad_source f x =
  let traced = Rune.trace_graph (Rune.grad f) x in
  String.concat "\n---\n" traced.rendered_source

(* ── Test cases ── *)

(* grad(sum(x*x)) = 2*x, shape [4] *)
let grad_square () =
  let x = Nx.full Nx.float32 [| 4 |] 3.0 in
  grad_source (fun x -> Nx.sum (Nx.mul x x)) x

(* grad(sum(sin(x))) = cos(x), shape [4] *)
let grad_sin () =
  let x = Nx.full Nx.float32 [| 4 |] 1.0 in
  grad_source (fun x -> Nx.sum (Nx.sin x)) x

(* grad(sum((x+1)*x)) = 2x+1, shape [4] *)
let grad_polynomial () =
  let x = Nx.full Nx.float32 [| 4 |] 2.0 in
  grad_source
    (fun x -> Nx.sum (Nx.mul (Nx.add x (Nx.scalar Nx.float32 1.0)) x))
    x

(* grad(sum(x*x*x)) = 3*x^2, shape [4] *)
let grad_cube () =
  let x = Nx.full Nx.float32 [| 4 |] 2.0 in
  grad_source (fun x -> Nx.sum (Nx.mul (Nx.mul x x) x)) x

type test_case = { name : string; generate : unit -> string }

let test_cases =
  [
    { name = "grad_square"; generate = grad_square };
    { name = "grad_sin"; generate = grad_sin };
    { name = "grad_polynomial"; generate = grad_polynomial };
    { name = "grad_cube"; generate = grad_cube };
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
