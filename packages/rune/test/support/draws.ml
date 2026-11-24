(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module R = Nx_rng_support

(* [f ()], on [set], computed with no call evaluated, read back on the host. *)
let compiled set f =
  Compiled.set set;
  let before = Rune.evaluations () in
  let y = f () in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  Nx.place Nx.Host.on y

(* A draw from a host key and host arguments compiled with its key at [whole]
   and each argument at [cut] of its rank. A draw given to a field is compiled
   once: its calls run that one compiled function. *)
let at whole cut =
  let set = Nx.Placement.devices whole in
  let key k = Nx.Rng.place whole k in
  let on x = Nx.place (cut (Array.length (Nx.shape x))) x in
  R.
    {
      run0 =
        (fun { f0 } ->
          let g = Rune.jit Nx.Ptree.(Nx.Rng.ptree @-> returns tensor) f0 in
          fun k -> compiled set (fun () -> g (key k)));
      run1 =
        (fun { f1 } ->
          let g =
            Rune.jit Nx.Ptree.(Nx.Rng.ptree @-> tensor @-> returns tensor) f1
          in
          fun k x -> compiled set (fun () -> g (key k) (on x)));
      run2 =
        (fun { f2 } ->
          let g =
            Rune.jit
              Nx.Ptree.(Nx.Rng.ptree @-> tensor @-> tensor @-> returns tensor)
              f2
          in
          fun k x y -> compiled set (fun () -> g (key k) (on x) (on y)));
    }

let law (module S : Nx.Devices) ~dtypes ~shapes k =
  R.law (at S.on (fun _ -> S.on)) ~dtypes ~shapes k

let mask_agrees (module S : Nx.Devices) =
  let x = Nx.create Nx.float32 [| 64; 64 |] (Array.init 4096 Float.of_int) in
  let f k x =
    let p = Nx.full_like x 0.9 in
    let m = Nx.cast Nx.float32 (Nx.Rng.bernoulli ~key:k p) in
    Nx.add (Nx.mul x m) (Nx.sum ~axes:[ 0 ] ~keepdims:true m)
  in
  let g = Rune.jit Nx.Ptree.(Nx.Rng.ptree @-> tensor @-> returns tensor) f in
  let k = Nx.Rng.key 5 in
  let y = Nx.place Nx.Host.on (g (Nx.Rng.place S.on k) (Nx.place S.on x)) in
  let bits a = Array.map Int32.bits_of_float (Nx.to_array a) in
  bits (f k x) = bits y
