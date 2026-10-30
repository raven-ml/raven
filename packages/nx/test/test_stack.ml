(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Operations of each of nx.cpu's kernel families, called from C stacks at
   both residues modulo 32 of a 16-byte aligned stack, give their values.
   Windows aligns a thread's stack to 16 bytes only, and a kernel compiled
   to keep 32-byte vectors on the stack at addresses it takes as 32-byte
   aligned faults at one residue. The values are exact. *)

open Windtrap

(* The depths that start nx's calls at both residues
   (Nx_stack_support.deeper). *)
let depths = [ 0; 8; 16; 24 ]

(* [f ()]'s elements, computed and read at each depth, each the same as
   [expected]. *)
let at_depths ~msg testable expected f =
  List.iter
    (fun n ->
      let got = Nx_stack_support.deeper n (fun () -> Nx.to_array (f ())) in
      equal ~msg:(Printf.sprintf "%s at depth %d" msg n) (array testable)
        expected got)
    depths

let iota n = Array.init n float_of_int
let x64 = Nx.arange Nx.float64 0 64 1
let x32 = Nx.arange Nx.float32 0 64 1

let test_elementwise () =
  at_depths ~msg:"add" float_exact
    (Array.init 64 (fun i -> float_of_int (2 * i)))
    (fun () -> Nx.add x32 x32);
  at_depths ~msg:"cast" float_exact (iota 64) (fun () ->
      Nx.cast Nx.float32 (Nx.cast Nx.float16 x32));
  at_depths ~msg:"copy" float_exact
    (Array.init 64 (fun i -> float_of_int ((i mod 8 * 8) + (i / 8))))
    (fun () -> Nx.copy (Nx.transpose (Nx.reshape [| 8; 8 |] x64)))

let test_reductions () =
  at_depths ~msg:"sum" float_exact [| 2016. |] (fun () -> Nx.sum x64);
  at_depths ~msg:"cumsum" float_exact
    (Array.init 64 (fun i -> float_of_int (i * (i + 1) / 2)))
    (fun () -> Nx.cumsum x64)

let test_moves () =
  let rev = Nx.arange Nx.float64 63 (-1) (-1) in
  at_depths ~msg:"sort" float_exact (iota 64) (fun () -> fst (Nx.sort rev));
  at_depths ~msg:"take" float_exact (iota 64) (fun () ->
      Nx.take (Nx.arange Nx.int64 63 (-1) (-1)) rev);
  at_depths ~msg:"concatenate" float_exact (iota 128) (fun () ->
      Nx.concatenate ~axis:0 [ x64; Nx.arange Nx.float64 64 128 1 ])

let test_contractions () =
  (* A ones matrix times 0 … 63 as 8 × 8: each column's sum, in each row. *)
  let a = Nx.reshape [| 8; 8 |] x32 in
  let ones = Nx.ones Nx.float32 [| 8; 8 |] in
  at_depths ~msg:"matmul" float_exact
    (Array.init 64 (fun i -> float_of_int ((8 * 28) + (8 * (i mod 8)))))
    (fun () -> Nx.matmul ones a)

let test_routines () =
  (* [[4, 2], [2, 3]] is l lᵀ for l = [[2, 0], [1, √2]] and maps [1, 1] to
     [6, 5]; its LU and its solve are exact in binary. *)
  let a = Nx.create Nx.float64 [| 2; 2 |] [| 4.; 2.; 2.; 3. |] in
  let b = Nx.create Nx.float64 [| 2 |] [| 6.; 5. |] in
  at_depths ~msg:"solve" float_exact [| 1.; 1. |] (fun () -> Nx.solve a b);
  at_depths ~msg:"cholesky" (float 1e-15) [| 2.; 0.; 1.; Float.sqrt 2. |]
    (fun () -> Nx.cholesky a);
  (* A diagonal matrix's eigenvalues, ascending. *)
  let d = Nx.create Nx.float64 [| 2; 2 |] [| 3.; 0.; 0.; 1. |] in
  at_depths ~msg:"eigh" (float 1e-15) [| 1.; 3. |] (fun () ->
      fst (Nx.eigh Nx.float64 d))

let test_fft () =
  (* An impulse's transform is ones. *)
  let impulse =
    Nx.create Nx.complex128 [| 8 |]
      (Array.init 8 (fun i -> if i = 0 then Complex.one else Complex.zero))
  in
  List.iter
    (fun n ->
      let got =
        Nx_stack_support.deeper n (fun () -> Nx.to_array (Nx.fft impulse))
      in
      equal ~msg:(Printf.sprintf "fft at depth %d" n)
        (array (pair float_exact float_exact))
        (Array.make 8 (1., 0.))
        (Array.map (fun (z : Complex.t) -> (z.re, z.im)) got))
    depths

let () =
  exit
    (run "nx stack"
       [
         group "kernels from either residue of the stack"
           [
             test "elementwise, casts and copies" test_elementwise;
             test "reductions and scans" test_reductions;
             test "sorts, gathers and joins" test_moves;
             test "products" test_contractions;
             test "factorisations and solves" test_routines;
             test "transforms" test_fft;
           ];
       ])
