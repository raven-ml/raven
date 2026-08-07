(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Reproduction driver for the BEAM miscompile: the gradient of an unrolled
   recurrence gives a different answer jitted under BEAM>=1 than it does
   eagerly, and a different wrong answer on each run.

   Not part of runtest: a single BEAM=2 pass over this graph takes minutes.
   Run it directly, e.g.

     BEAM=2 IGNORE_BEAM_CACHE=1 dune exec packages/rune/test/repro_jit_beam.exe *)

let f32 = Nx.float32
let horizon = try int_of_string (Sys.getenv "HORIZON") with _ -> 10
let bs = 64
let m = 96
let n = 32

type rnn = { w : Nx.float32_t; b : Nx.float32_t }

module Rnn = struct
  type t = rnn

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { w; b } =
    { w = f w; b = f b }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { w = f p.w q.w; b = f p.b q.b }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { w; b } =
    f w;
    f b
end

(* Deterministic fill, so eager and jitted runs see identical data and repeated
   process runs are comparable. *)
let fill shape scale =
  let n = Array.fold_left ( * ) 1 shape in
  Nx.create f32 shape
    (Array.init n (fun i -> scale *. sin (float_of_int ((i * 7) + 1) *. 0.37)))

let params () =
  {
    w = fill [| n; n |] (0.5 /. sqrt (float_of_int n));
    b = fill [| m; n |] (1.0 /. sqrt (float_of_int m));
  }

let x0 = fill [| bs; n |] 1.0
let us = fill [| horizon; bs; m |] 1.0

let loss (p : rnn) =
  let x = ref x0 in
  let acc = ref (Nx.scalar f32 0.) in
  for t = 0 to horizon - 1 do
    let u = Nx.squeeze (Nx.slice [ Nx.I t ] us) in
    x := Nx.add (Nx.matmul !x p.w) (Nx.matmul u p.b);
    acc := Nx.add !acc (Nx.mean (Nx.mul !x !x))
  done;
  !acc

(* Gradient norm: a scalar summarising every entry of both gradients, so a
   miscompile anywhere in the backward graph shows up in one number. *)
let grad_norm p =
  let g = Rune.grad (module Rnn) loss p in
  Nx.add (Nx.mean (Nx.mul g.b g.b)) (Nx.mean (Nx.mul g.w g.w))

let time label f =
  let t0 = Unix.gettimeofday () in
  let r = f () in
  Printf.printf "%-22s %8.4fs  result = %.9f\n%!" label
    (Unix.gettimeofday () -. t0) r;
  r

let () =
  let device = try Sys.getenv "DEVICE" with Not_found -> "CPU" in
  let p = params () in
  let item t = Nx.item [] t in
  Printf.printf "device=%s horizon=%d BEAM=%s\n%!" device horizon
    (try Sys.getenv "BEAM" with Not_found -> "0");
  let eager = time "eager" (fun () -> item (grad_norm p)) in
  let jitted = Rune.jit ~device (module Rnn) grad_norm in
  let first = time "jit (compile+run)" (fun () -> item (jitted p)) in
  let replay = time "jit (replay)" (fun () -> item (jitted p)) in
  let ok v = Float.abs (v -. eager) <= 1e-4 *. Float.max 1.0 (Float.abs eager) in
  if ok first && ok replay then print_endline "OK: jit matches eager"
  else (
    Printf.printf "MISMATCH: eager %.9f, jit %.9f / %.9f\n" eager first replay;
    exit 1)
