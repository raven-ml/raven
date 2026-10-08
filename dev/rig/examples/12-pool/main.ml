(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The host's pool of threads.

   One pool of threads per process computes on the host: nx's kernels, host
   programs, and a library's own C code share it, so that together they never
   take more threads than the host has cores. A job is a range of units cut into
   chunks; the pool's threads claim chunks in order until none remains.

   The pool's interface is C ([rig_pool.h]). [sum.c] runs a job from an OCaml
   stub. The output prints no core counts, which differ from host to host. *)

external sum_squares :
  int ->
  int ->
  (int64, Bigarray.int64_elt, Bigarray.c_layout) Bigarray.Array1.t ->
  int64 = "caml_rig_example_sum_squares"

external cores : unit -> int * int = "caml_rig_example_cores"

let () =
  let cores, performance = cores () in
  Printf.printf "1 <= performance cores <= cores: %b\n"
    (1 <= performance && performance <= cores);

  let n = 1_000_000 in
  let x = Bigarray.(Array1.init int64 c_layout n Int64.of_int) in
  let expected = ref 0L in
  for i = 0 to n - 1 do
    expected := Int64.add !expected (Int64.mul x.{i} x.{i})
  done;
  Printf.printf "sum of squares below %d: %Ld\n" n !expected;

  (* The same job on one thread, on the performance cores, and on every core, in
     256 chunks: the sum is the same. *)
  List.iter
    (fun (name, threads) ->
      Printf.printf "  %-17s %b\n" name (sum_squares threads 256 x = !expected))
    [
      ("one thread", 1);
      ("performance cores", performance);
      ("every core", cores);
    ]
