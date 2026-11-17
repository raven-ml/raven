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
