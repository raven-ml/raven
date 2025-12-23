open Nx

(* open Core *)
module Par_array = Parallel.Arrays.Array
module Slice = Par_array.Slice
module Capsule = Await.Capsule
module Iarray = Parallel.Arrays.Iarray

(* open Array open Parallel *)
open Stdlib

(* Basic Nx Operations: Arithmetic, indexing, and transformations *)

let () =
  (* Create some sample arrays *)
  let a =
    Nx.init float64 [| 2; 3 |] (fun idx ->
        float_of_int ((idx.(0) * 3) + idx.(1) + 1)
        (* Values 1-6 in row-major order *))
  in

  let b =
    Nx.init float64 [| 2; 3 |] (fun idx ->
        float_of_int (idx.(0) + idx.(1) + 1) (* Values based on row+col+1 *))
  in

  (* Display our sample arrays *)
  Printf.printf "Array A:\n%s\n\n" (to_string a);
  Printf.printf "Array B:\n%s\n\n" (to_string b);

  (* 3. Basic arithmetic operations *)
  let t0 = Unix.gettimeofday () in
  let sum = add a b in
  let t1 = Unix.gettimeofday () in
  Printf.printf "add took %.6f seconds\n" (t1 -. t0);
  Printf.printf "A + B:\n%s\n\n" (to_string sum);

  let parallel_fork_join (par : Parallel.t) f1i f2i f1i_next f2i_next
     = let (s0, s1) = Parallel.fork_join2 par 
     (fun _par -> [|f1i +. f2i|]) 
     (fun _par -> [|f1i_next +. f2i_next|]) in
     (s0 s1)

     in

     let run_one_test ~(f : Parallel.t @ local -> 'a) : 'a = let module
     Scheduler = Parallel_scheduler in let scheduler = Scheduler.create () in
     let result = Scheduler.parallel scheduler ~f in Scheduler.stop scheduler;
     result in
  let sum_oxcaml_parallel a b =
    (* flatten both arrays *)
    let shape = Nx.shape a in
    let flat1 = Nx.to_array (Nx.flatten a) in
    let flat2 = Nx.to_array (Nx.flatten b) in

    let len1 = Array.length flat1 in
    let len2 = Array.length flat2 in
    assert (len1 = len2);

    let len = len1 in
    (* output flat buffer *)
    let out = ones float64 [| len |] in
    let t0 = Unix.gettimeofday () in
    (* iterate with stride 2 and use parallel fork/join *)
    let rec loop i =
      if i >= len then ()
        else if i = len - 1 then
        (* tail element *)
        let f1i = flat1.(i) in
        let f2i = flat2.(i) in
        Nx.set [ i ] out (Nx.scalar float64 (f1i +. f2i));
        loop (i + 1)
      else begin let f1i = flat1.(i) in let f2i = flat2.(i) in let f1i_next =
         flat1.(i+1) in let f2i_next = flat2.(i+1) in let
         test_sum_oxcaml_parallel (par @ local) = parallel_fork_join par f1i f2i
         f1i_next f2i_next in

         let (s0, s1) = run_one_test ~f:test_sum_oxcaml_parallel

         in Nx.set [i] out (Nx.scalar float64 s0.(0)); Nx.set [i + 1] out
         (Nx.scalar float64 s1.(0));

         loop (i + 2) end
    in

    loop 0;
    let t1 = Unix.gettimeofday () in
    Printf.printf "pre loop took %.6f seconds\n" (t1 -. t0);
    (* reshape back to original shape *)
    Nx.reshape shape out
  in

  let t0 = Unix.gettimeofday () in

  let res = sum_oxcaml_parallel a b in
  let t1 = Unix.gettimeofday () in
  Printf.printf "add took %.6f seconds\n" (t1 -. t0);
  Printf.printf "A + B:\n%s\n\n" (to_string res)
