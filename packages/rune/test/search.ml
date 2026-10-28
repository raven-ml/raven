(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Calls whose search reports the cram test reads. *)

module Rune = Rune2

let x () = Nx.ones Nx.float32 [| 4 |]

(* A function compiled with a search reports each kernel it searches; another
   compiled without one reports none. *)
let () =
  let f a = Nx.add_s (Nx.mul a a) 0.625 in
  ignore (Rune.jit' ~beam:1 ~parallel:2 f (x ()));
  let g a = Nx.add_s (Nx.mul a a) 0.875 in
  ignore (Rune.jit' g (x ()))

(* An explicit width stands for BEAM: [0] searches nothing under BEAM=1, and
   a function given a width compiles nothing again under another BEAM. *)
let () =
  let g a = Nx.add_s (Nx.mul a a) 0.375 in
  Tolk.Setting.context
    [ B (Tolk.Setting.beam, 1) ]
    (fun () -> ignore (Rune.jit' ~beam:0 g (x ())));
  let f a = Nx.add_s (Nx.mul a a) 0.125 in
  let h = Rune.jit' ~beam:2 f in
  ignore (h (x ()));
  Tolk.Setting.context
    [ B (Tolk.Setting.beam, 3); B (Tolk.Setting.jitbeam, Some 3) ]
    (fun () -> ignore (h (x ())))

(* The programs derived from a function given a width search at that width. *)
let () =
  let f a = Nx.sum (Nx.mul (Nx.sin a) a) in
  ignore (Rune.grad' (Rune.jit' ~beam:1 f) (x ()))

(* A call inlined in a jit given no width compiles as that one's program: it
   searches nothing. *)
let () =
  let inner = Rune.jit' ~beam:1 (fun a -> Nx.add_s (Nx.mul a a) 0.25) in
  ignore (Rune.jit' (fun a -> Nx.neg (inner a)) (x ()))
