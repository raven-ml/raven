(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx on a set whose kernels are nx.cuda's, on GPU 0 through CUDA: every draw of
   Nx.Rng has the host's bits (Nx_rng_support). A draw is a program of kinds,
   which nx.cuda computes by nx_kinds.h's functions, and of integer arithmetic,
   a dirichlet row's sum among it. A case draws a block of 2048 elements whose
   parameters meet every cover of Nx_rng_support, so two cases a sampler reach
   them all. Every test skips where CUDA opens no GPU. *)

open Windtrap

let set =
  lazy
    (match Nx_kernels_support.gpus () with
    | [] -> None
    | b :: _ -> Some (Nx.devices ~kernels:b.kernels [ b.device ]))

let draws s =
  prop ~count:2
    (Nx_rng_support.name s ^ " has the host's bits")
    (Nx_rng_support.gen s)
    (fun c ->
      match Lazy.force set with
      | None -> skip ~reason:"CUDA opens no GPU" ()
      | Some set ->
          Nx_rng_support.(
            law (placed set)
              ~dtypes:[ Float16; Bfloat16; Float32; Float64 ]
              ~shapes:[ Zero; One; Block; Row ] s c))

(* nx_cuda.mli: a kernel allocates a number of words its operands' extents do
   not change. An eager binomial is one map of a program too wide for a launch,
   which nx.cuda runs in parts. The budget is the words one warm draw took once
   nx.cuda's search for the parts spanned a launch's nodes, 18.2M on fudge
   (2026-11-26), with a quarter's margin; a search over the whole rest of the
   program took 45.6M. *)
let budget = 23_000_000

(* The words one draw of [n] elements allocates on the host, after one uncounted
   draw. *)
let draw_words n =
  match Lazy.force set with
  | None -> skip ~reason:"CUDA opens no GPU" ()
  | Some set ->
      let module G = (val set) in
      let t =
        Nx.place G.on
          (Nx.create Nx.int32 [| n |]
             (Array.init n (fun i -> Int32.of_int (1 + (i * 37 mod 100_000)))))
      in
      let p =
        Nx.place G.on
          (Nx.create Nx.float32 [| n |]
             (Array.init n (fun i -> Float.of_int i /. Float.of_int n)))
      in
      let key = Nx.Rng.place G.on (Nx.Rng.key 3) in
      let draw () = ignore (Sys.opaque_identity (Nx.Rng.binomial ~key t p)) in
      let words () =
        let minor, promoted, major = Gc.counters () in
        minor +. major -. promoted
      in
      draw ();
      let before = words () in
      draw ();
      int_of_float (words () -. before)

let words =
  group "words"
    [
      test "a warm eager binomial allocates within its budget" (fun () ->
          less int ~than:budget (draw_words 2048));
      (* At one element the loop has no axis, a launch's words differ, and so do
         its parts. A few hundred words of 18M move between runs: a thousandth
         is allowed. *)
      test "a warm eager binomial allocates as many words at 2 as at 2048"
        (fun () ->
          let two = draw_words 2 in
          less int ~than:(two / 1000) (abs (draw_words 2048 - two)));
    ]

let () =
  exit
    (run "Nx.Rng on nx.cuda"
       [ group "draws" (List.map draws Nx_rng_support.samplers); words ])
