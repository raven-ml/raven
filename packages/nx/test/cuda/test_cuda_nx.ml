(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx on a set whose kernels are nx.cuda's: Nx's contractions and reductions
   of values placed on CUDA's GPU 0 compute there and give the host's
   results. The values are small integers, which every association sums
   exactly. Every test skips where CUDA sees no GPU. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype

let gpu =
  let t =
    lazy
      (match Rig_cuda.count () with
      | Ok 0 | Error _ -> None
      | Ok _ -> (
          match Rig.open_ { (Rig_cuda.gpu 0) with name = "CUDA:nx" } with
          | Ok d -> Some d
          | Error e -> failwith e))
  in
  fun () ->
    match Lazy.force t with
    | Some d -> d
    | None -> skip ~reason:"CUDA sees no GPU" ()

let host dt s xs =
  Nx.Repr.of_array Nx.Host.v (A.of_array dt (Iarray.of_array s) xs)

let read x = A.to_array (Option.get (Nx.Repr.array (Nx.place Nx.Host.on x)))

(* [n] small integers from [seed], as floats. *)
let ints n seed =
  Array.init n (fun i -> Float.of_int ((((i * 7) + seed) mod 9) - 4))

let matmul () =
  let module G = (val Nx.devices ~kernels:(module Nx_cuda) [ gpu () ]) in
  let a = host D.Float32 [| 17; 33 |] (ints (17 * 33) 1) in
  let b = host D.Float32 [| 33; 9 |] (ints (33 * 9) 2) in
  let want = read (Nx.matmul a b) in
  let got = read (Nx.matmul (Nx.place G.on a) (Nx.place G.on b)) in
  equal (array float_exact) want got

let sum () =
  let module G = (val Nx.devices ~kernels:(module Nx_cuda) [ gpu () ]) in
  let x = host D.Float32 [| 6; 1000 |] (ints 6000 3) in
  let want = read (Nx.sum ~axes:[ 1 ] x) in
  let got = read (Nx.sum ~axes:[ 1 ] (Nx.place G.on x)) in
  equal (array float_exact) want got

(* nx_cuda.mli: a float sum associates as nx.cpu states, so the two give
   the same bits; over rows longer than a block of 1024 terms and a scan's
   chunk of 4096. *)
let float_sum_bits () =
  let module G = (val Nx.devices ~kernels:(module Nx_cuda) [ gpu () ]) in
  let st = Random.State.make [| 7 |] in
  let n = 3 * 10_007 in
  let xs = Array.init n (fun _ -> Random.State.float st 2. -. 1.) in
  let x = host D.Float32 [| 3; 10_007 |] xs in
  let bits a = Array.map Int32.bits_of_float a in
  equal (array int32)
    (bits (read (Nx.sum ~axes:[ 1 ] x)))
    (bits (read (Nx.sum ~axes:[ 1 ] (Nx.place G.on x))));
  equal (array int32)
    (bits (read (Nx.cumsum ~axis:1 x)))
    (bits (read (Nx.cumsum ~axis:1 (Nx.place G.on x))))

let () =
  exit
    (run "nx on nx.cuda"
       [
         group "a set of nx.cuda's kernels"
           [
             test "matmul computes on the GPU" matmul;
             test "sum computes on the GPU" sum;
             test "a float32 sum and scan have nx.cpu's bits" float_sum_bits;
           ];
       ])
