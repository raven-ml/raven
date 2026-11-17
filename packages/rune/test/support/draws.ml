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

(* A draw from a host key and host arguments compiled on [S], its key and
   arguments placed whole on each of [S]'s devices. *)
let runner (module S : Nx.Devices) =
  let key k = Nx.Rng.place S.on k and on x = Nx.place S.on x in
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

let law s ~dtypes ~shapes k c =
  R.law (runner s) ~rows:Bound ~dtypes ~shapes k c
