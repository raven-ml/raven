(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Aion

type 'd transformer = {
  encode :
    ('d Nx.float32_t, 'd) Transformer.t ->
    'd Tokens.t list ->
    ('d Nx.float32_t, 'd) Transformer.encoding;
  decode :
    ('d Nx.float32_t, 'd) Transformer.t ->
    ('d Nx.float32_t, 'd) Transformer.encoding ->
    'd Transformer.target list ->
    'd Nx.float32_t list;
  forward :
    ('d Nx.float32_t, 'd) Transformer.t ->
    'd Tokens.t list ->
    'd Transformer.target list ->
    'd Nx.float32_t list;
  embedding : ('d Nx.float32_t, 'd) Transformer.encoding -> 'd Nx.float32_t;
}

let transformer () =
  let open Nx.Ptree in
  let t () = instantiate (module Transformer)
  and inputs () = list (v Tokens.walk)
  and targets () = list (v Transformer.walk_target)
  and encoding () = instantiate (module Transformer.Encoding) in
  {
    encode =
      Rune.jit (t () @-> inputs () @-> returns (encoding ())) Transformer.encode;
    decode =
      Rune.jit
        (t () @-> encoding () @-> targets () @-> returns (list tensor))
        Transformer.decode;
    forward =
      Rune.jit
        (t () @-> inputs () @-> targets () @-> returns (list tensor))
        Transformer.forward;
    embedding = Rune.jit (encoding () @-> returns tensor) Transformer.embedding;
  }

let host x = Nx.place Nx.Host.on x
let floats x = Nx.to_array (Nx.cast Nx.float64 (host x))
let ints x = Array.map Int64.to_int (Nx.to_array (host x))
let golden () = Fixture.load "transformer_golden"
let modalities = Modality.[ Scalar Xp_bp; Scalar Flux_g; Scalar Z ]

(* [x] within 2 τ ŝ of eager's [y], [spread] the fixture's ŝ. *)
let close name ~spread x y =
  Spread.holds name
    (Spread.values
       ~n:(Fixture.ensemble (golden ()))
       ~reference:(floats y)
       ~spread:(Array.map (fun s -> 2. *. s) (floats spread))
       (floats x))

(* [embedding]'s sums compiled and eager: two sums of [l] terms, each within
   γ(l) Σ|x| of the exact one ([l] below 256), divided by [l] with one more
   rounding. *)
let embedding_bound (e : _ Transformer.encoding) =
  let c = floats e.context and l = ints e.lengths in
  let n = Nx.dim 1 e.context and w = Nx.dim 2 e.context in
  let u = ldexp 1. (-24) in
  Array.init
    (Array.length l * w)
    (fun x ->
      let i = x / w and a = ref 0. in
      for r = 0 to l.(i) - 1 do
        a := !a +. Float.abs c.((((i * n) + r) * w) + (x mod w))
      done;
      let l = float_of_int l.(i) in
      let g = l *. u /. (1. -. (l *. u)) in
      (2. *. g *. !a /. l) +. (2. *. u *. !a /. l))

let int_rows rows =
  Nx.create Nx.int64
    [| List.length rows |]
    (Array.of_list (List.map Int64.of_int rows))

let agrees on c ~rows ~step ~perm ~zs =
  let t = Fixture.transformer () in
  let tc = Nx.Ptree.place (module Transformer) on t in
  let put = List.map (Nx.Ptree.place_fixed Tokens.walk on) in
  let encoding e = Nx.Ptree.place (module Transformer.Encoding) on e in
  let x = Fixture.inputs ~rows modalities in
  let e = Transformer.encode t x in
  let ec = c.encode tc (put x) in
  equal (array int) (ints e.lengths) (ints ec.lengths);
  close "context"
    ~spread:(Fixture.take rows (Fixture.get (golden ()) Nx.float32 "s.embed"))
    ec.context e.context;
  let emb = floats (Transformer.embedding e) in
  let d = floats (c.embedding (encoding e)) in
  Array.iteri
    (fun i b ->
      if Float.abs (d.(i) -. emb.(i)) > b then
        failf "embedding %d: %g, eager %g, bound %g" i d.(i) emb.(i) b)
    (embedding_bound e);
  (* The step's positions of each object, and their spread, in the order
     [perm]. *)
  let x = Fixture.inputs ~rows ~step modalities and k = List.length rows in
  let perm = Nx.broadcast_to [| k; 7 |] (int_rows perm) in
  let p =
    Nx.take_along_axis ~axis:1 perm
      (Fixture.take rows
         (Fixture.get
            (Fixture.load "roar_golden")
            Nx.int64 (step ^ ".positions")))
  in
  let spread =
    Nx.take_along_axis ~axis:1
      (Nx.reshape [| k; 7; 1 |] perm)
      (Fixture.take rows
         (Fixture.get (Fixture.load "roar_golden") Nx.float32 ("s." ^ step)))
  in
  let target p : _ Transformer.target =
    { modality = Scalar Xp_rp; positions = Some p }
  in
  let l = List.hd (Transformer.forward t x [ target p ]) in
  let pc = [ target (Nx.place on p) ] in
  close "forward" ~spread (List.hd (c.forward tc (put x) pc)) l;
  let ep = encoding (Transformer.encode t x) in
  close "decode" ~spread (List.hd (c.decode tc ep pc)) l;
  let x = Fixture.inputs ~rows:zs modalities in
  let z : _ Transformer.target list =
    [ { modality = Scalar Z; positions = None } ]
  in
  close "whole target"
    ~spread:
      (Fixture.take (List.map pred zs)
         (Fixture.get (golden ()) Nx.float32 "s.logits.tok_z"))
    (List.hd (c.forward tc (put x) z))
    (List.hd (Transformer.forward t x z))

let draw ?(traces = ref 0) on =
  let step =
    Rune.jit
      Nx.Ptree.(
        instantiate (module Transformer)
        @-> v Walk.int @-> v Roar.walk_draw
        @-> returns (v Roar.walk_draw))
      (fun t k d ->
        incr traces;
        Roar.step t k d)
  in
  let t =
    lazy (Nx.Ptree.place (module Transformer) on (Fixture.transformer ()))
  in
  fun ~keys ?temperature ~reveal target inputs ->
    (* [start] captures the temperature and the target. *)
    let start =
      Rune.jit
        Nx.Ptree.(
          Nx.Rng.ptree @-> list (v Tokens.walk) @-> returns (v Roar.walk_draw))
        (fun keys x -> Roar.start ~keys ?temperature x target)
    in
    let d =
      start (Nx.Rng.place on keys)
        (List.map (Nx.Ptree.place_fixed Tokens.walk on) inputs)
    in
    host
      (Roar.tokens
         (List.fold_left (fun d k -> step (Lazy.force t) k d) d reveal))
