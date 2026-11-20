(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module R = Nx_rng_support

(* [f ()] computed with no call evaluated, read back on the host. *)
let compiled f =
  let before = Rune.evaluations () in
  let y = f () in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  Nx.place Nx.Host.on y

(* A draw from a host key and host arguments compiled with its key at
   [whole] and each argument at [cut] of its rank. *)
let at whole cut =
  let key k = Nx.Rng.place whole k in
  let on x = Nx.place (cut (Array.length (Nx.shape x))) x in
  R.
    {
      run0 =
        (fun { f0 } k ->
          compiled (fun () ->
              Rune.jit Nx.Ptree.(Nx.Rng.ptree @-> returns tensor) f0 (key k)));
      run1 =
        (fun { f1 } k x ->
          compiled (fun () ->
              Rune.jit
                Nx.Ptree.(Nx.Rng.ptree @-> tensor @-> returns tensor)
                f1 (key k) (on x)));
      run2 =
        (fun { f2 } k x y ->
          compiled (fun () ->
              Rune.jit
                Nx.Ptree.(Nx.Rng.ptree @-> tensor @-> tensor @-> returns tensor)
                f2 (key k) (on x) (on y)));
    }

let law (module S : Nx.Devices) ~dtypes ~shapes k c =
  R.law (at S.on (fun _ -> S.on)) ~dtypes ~shapes k c

let mask_stores (module S : Nx.Devices) =
  let x = Nx.create Nx.float32 [| 64; 64 |] (Array.init 4096 Float.of_int) in
  let f k x =
    let p = Nx.full_like x 0.9 in
    let m = Nx.cast Nx.float32 (Nx.Rng.bernoulli ~key:k p) in
    Nx.add (Nx.mul x m) (Nx.sum ~axes:[ 0 ] ~keepdims:true m)
  in
  let g = Rune.jit Nx.Ptree.(Nx.Rng.ptree @-> tensor @-> returns tensor) f in
  let k = Nx.Rng.key 5 in
  let y, ks =
    Tolk.Realize.kernels (fun () ->
        Nx.place Nx.Host.on (g (Nx.Rng.place S.on k) (Nx.place S.on x)))
  in
  let bits a = Array.map Int32.bits_of_float (Nx.to_array a) in
  let nodes u = Tolk.Ops.toposort ~calls:Skip u in
  (* A kernel's bodies: its own, or each its launch calls. *)
  let rec bodies u =
    match List.filter (fun v -> Tolk.Ops.op v = Tolk.Op.Call) (nodes u) with
    | [] -> [ u ]
    | calls -> List.concat_map (fun c -> bodies (Tolk.Ops.body c)) calls
  in
  let parity = Tolk.Bigint.of_int 0x1BD11BDA in
  let schedules u =
    match Tolk.Ops.arg u with
    | Tolk.Ops.Const (`Int n) -> Tolk.Bigint.equal n parity
    | _ -> false
  in
  let stored k =
    List.filter_map
      (fun u ->
        if Tolk.Ops.op u = Tolk.Op.Store then
          Some (Tolk.Dtype.name (Tolk.Ops.dtype (Tolk.Ops.nth u 1)))
        else None)
      (nodes k)
  in
  let draws =
    List.filter
      (fun u -> List.exists schedules (nodes u))
      (List.concat_map (fun (k : Tolk.Realize.kernel) -> bodies k.ast) ks)
  in
  (List.concat_map stored draws, bits (f k x) = bits y)
