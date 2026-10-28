(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Cache indices compiled for the Metal device, a set nx computes only under
   [Rune.jit]. Compiled only on macOS. *)

open Windtrap
open Kaun

let int64s shape a = Nx.create Nx.int64 shape (Array.map Int64.of_int a)
let flat t = Nx.to_array (Nx.reshape [| -1 |] t)

(* A virtual Mac's paravirtual GPU makes no residency set, so Metal opens no
   device there. *)
let metal =
  lazy
    (match Rig.open_ (Rig_metal.gpu 0) with
    | Ok d -> Nx.devices [ d ]
    | Error ("Metal made no residency set" as why) -> skip ~reason:why ()
    | Error why -> fail why)

(* A toy compressed stream, as in test_attention.ml: each token stores its value
   at its position, the token that closes a block of 4 stores the sum of its
   block's values, read through a selection, and each token sums the entries it
   sees. *)

type 'd state = {
  x : 'd Nx.float32_t;
  y : 'd Nx.float32_t;
  sources : 'd Nx.float32_t;
  entries : 'd Nx.float32_t;
}

module State = struct
  type (_, 'd) t = 'd state

  let walk c s =
    let open Nx.Ptree.Walk in
    let x = field c "x" tensor s.x in
    let y = field c "y" tensor s.y in
    let sources = field c "sources" tensor s.sources in
    let entries = field c "entries" tensor s.entries in
    { x; y; sources; entries }
end

let state () = Nx.Ptree.instantiate (module State)

let stream index s =
  let batch = Cache_index.batch index and seq = Cache_index.seq index in
  let pos = Cache_index.positions index in
  let first = Nx.reshape [| batch; seq; 1 |] (Nx.sub pos (Nx.mod_s pos 4L)) in
  let block =
    Nx.add first
      (Nx.reshape [| 1; 1; 4 |]
         (Nx.arange_at (Nx.placement pos) Nx.int64 0 4 1))
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
         (Nx.scalar_like seen 0.))
  in
  { s with y; sources; entries }

(* A prompt of 6 positions, which splits block 1, then 6 one-token steps that
   close blocks 1 and 2, over shuffled tables. Block j holds 16 j + 10. *)
let test_stream_on_metal () =
  let module Metal = (val Lazy.force metal) in
  let state = state () in
  let step =
    let step =
      Rune.jit Nx.Ptree.(v Cache_index.walk @-> state @-> returns state) stream
    in
    fun index s -> step index (Nx.Ptree.donate state s)
  in
  let on x = Nx.place Metal.on x in
  let table =
    on (int64s [| 1; 12 |] [| 5; 11; 0; 7; 2; 9; 4; 1; 10; 3; 8; 6 |])
  in
  let blocks = on (int64s [| 1; 3 |] [| 1; 2; 0 |]) in
  let call s positions =
    let n = Array.length positions in
    let pos = int64s [| 1; n |] positions in
    let index =
      Cache_index.make ~every:[ (4, blocks) ] ~pos:(on pos) ~table ()
    in
    let x = Nx.add_s (Nx.cast Nx.float32 (Nx.reshape [| 1; n; 1 |] pos)) 1. in
    step index { s with x = on x; y = on (Nx.zeros Nx.float32 [| 1; n |]) }
  in
  let s =
    call
      {
        x = on (Nx.zeros Nx.float32 [| 1; 1; 1 |]);
        y = on (Nx.zeros Nx.float32 [| 1; 1 |]);
        sources = on (Nx.zeros Nx.float32 [| 12; 1 |]);
        entries = on (Nx.zeros Nx.float32 [| 3; 1 |]);
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

(* One slot, so the pool's slot axis has extent 1: a token that stores nothing
   leaves the slot as a token that stores there left it. *)
let test_one_slot_on_metal () =
  let module Metal = (val Lazy.force metal) in
  let on x = Nx.place Metal.on x in
  List.iter
    (fun (msg, pos, table, expected) ->
      let values =
        Nx.create Nx.float32 [| 1; 2; 2; 4 |]
          (Array.init 16 (fun i -> if i < 8 then 10. else 20.))
      in
      let pool = Nx.zeros Nx.float32 [| 1; 2; 4 |] in
      let f pos table values pool =
        snd (Cache_index.extend (Cache_index.make ~pos ~table ()) values pool)
      in
      let step =
        Rune.jit
          Nx.Ptree.(tensor @-> tensor @-> tensor @-> tensor @-> returns tensor)
          f
      in
      equal ~msg (array float_exact) (Array.make 8 expected)
        (flat
           (step
              (on (int64s [| 1; 2 |] pos))
              (on (int64s [| 1; Array.length table |] table))
              (on values) (on pool))))
    [
      ("padding after the token", [| 0; -1 |], [| 0 |], 10.);
      ("padding before the token", [| -1; 0 |], [| 0 |], 20.);
      ("an unallocated column before it", [| 0; 1 |], [| -1; 0 |], 20.);
    ]

let () =
  exit
    (run "kaun cache index metal"
       [
         group "select and every"
           [
             test
               "a stream of blocks compiled for Metal stores and reads the \
                expected entries"
               test_stream_on_metal;
             test "a pool of one slot keeps what its token stored"
               test_one_slot_on_metal;
           ];
       ])
