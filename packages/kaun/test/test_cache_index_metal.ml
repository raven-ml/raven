(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Cache indices compiled for the Metal device. Compiled only on macOS. *)

open Windtrap
open Kaun

let int32s shape a = Nx.create Nx.int32 shape (Array.map Int32.of_int a)
let flat t = Nx.to_array (Nx.reshape [| -1 |] (Nx.contiguous t))

(* A toy compressed stream, as in test_attention.ml: each token stores its value
   at its position, the token that closes a block of 4 stores the sum of its
   block's values, read through a selection, and each token sums the entries it
   sees. *)

type state = {
  x : Nx.float32_t;
  y : Nx.float32_t;
  sources : Nx.float32_t;
  entries : Nx.float32_t;
}

module State = struct
  type _ t = state

  let walk c s =
    let open Nx.Ptree.Walk in
    let x = field c "x" tensor s.x in
    let y = field c "y" tensor s.y in
    let sources = field c "sources" tensor s.sources in
    let entries = field c "entries" tensor s.entries in
    { x; y; sources; entries }
end

let state = Nx.Ptree.instantiate (module State)

let stream index s =
  let batch = Cache_index.batch index and seq = Cache_index.seq index in
  let pos = Cache_index.positions index in
  let first = Nx.reshape [| batch; seq; 1 |] (Nx.sub pos (Nx.mod_s pos 4l)) in
  let block =
    Nx.add first (Nx.reshape [| 1; 1; 4 |] (Nx.arange Nx.int32 0 4 1))
  in
  let own, sources =
    Cache_index.extend (Cache_index.select block index) s.x s.sources
  in
  let blocks = Cache_index.every 4 index in
  let seen, entries =
    Cache_index.extend blocks (Nx.sum ~axes:[ 2 ] own) s.entries
  in
  let context = Cache_index.context blocks in
  let y =
    Nx.sum ~axes:[ 2 ]
      (Nx.where (Cache_index.mask blocks)
         (Nx.reshape [| batch; 1; context |] seen)
         (Nx.zeros Nx.float32 [| 1 |]))
  in
  { s with y; sources; entries }

(* A prompt of 6 positions, which splits block 1, then 6 one-token steps that
   close blocks 1 and 2, over shuffled tables. Block j holds 16 j + 10. *)
let test_stream_on_metal () =
  let step = Rune.jit_step ~device:"METAL" Cache_index.ptree state stream in
  let table = int32s [| 1; 12 |] [| 5; 11; 0; 7; 2; 9; 4; 1; 10; 3; 8; 6 |] in
  let blocks = int32s [| 1; 3 |] [| 1; 2; 0 |] in
  let call s positions =
    let n = Array.length positions in
    let pos = int32s [| 1; n |] positions in
    let index = Cache_index.make ~every:[ (4, blocks) ] ~pos ~table () in
    let x = Nx.add_s (Nx.cast Nx.float32 (Nx.reshape [| 1; n; 1 |] pos)) 1. in
    step index { s with x; y = Nx.zeros Nx.float32 [| 1; n |] }
  in
  let s =
    call
      {
        x = Nx.zeros Nx.float32 [| 1; 1; 1 |];
        y = Nx.zeros Nx.float32 [| 1; 1 |];
        sources = Cache_index.pool ~slots:12 Nx.float32 [| 1 |];
        entries = Cache_index.pool ~slots:3 Nx.float32 [| 1 |];
      }
      [| 0; 1; 2; 3; 4; 5 |]
  in
  let ys = ref (Array.to_list (flat s.y)) and s = ref s in
  for t = 6 to 11 do
    s := call !s [| t |];
    ys := !ys @ Array.to_list (flat !s.y)
  done;
  let expected =
    List.init 12 (fun t ->
        List.fold_left ( +. ) 0.
          (List.init 3 (fun j ->
               if (j + 1) * 4 <= t + 1 then float_of_int ((16 * j) + 10) else 0.)))
  in
  equal ~msg:"outputs" (list float_exact) expected !ys;
  equal ~msg:"entries, at their shuffled slots" (array float_exact)
    [| 42.; 10.; 26. |]
    (Array.sub (flat !s.entries) 0 3)

let () =
  run "kaun cache index metal"
    [
      group "select and every"
        [
          test
            "a stream of blocks compiled for Metal stores and reads the \
             expected entries"
            test_stream_on_metal;
        ];
    ]
