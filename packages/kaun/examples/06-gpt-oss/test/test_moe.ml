(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The mixture of experts at gpt-oss's expert-parallel placement: experts
   split along their expert axis over four memory devices give the bits the
   experts give on one device, float and MXFP4, since each block's take of
   its expert has a gather's bits wherever the experts lie. *)

open Windtrap

let memory k = Result.get_ok (Rig.memory_device (Printf.sprintf "m%d" k))

module S = (val Nx.devices (List.init 4 memory))

let experts = 4
let width = 32
let hidden = 32
let limit = 7.

(* Floats in [-0.5, 0.5) and bytes, from a pattern of the position. *)
let floats s seed =
  let n = Array.fold_left ( * ) 1 s in
  Nx.create Nx.float32 s
    (Array.init n (fun i ->
         (Float.of_int ((i + seed) * 7919 mod 97) /. 97.) -. 0.5))

let bytes s seed f =
  let n = Array.fold_left ( * ) 1 s in
  Nx.create Nx.uint8 s (Array.init n (fun i -> f ((i + seed) * 37)))

let float_weight ~inputs ~outputs seed =
  Moe.Float (floats [| experts; inputs; outputs |] seed)

let mxfp4_weight ~inputs ~outputs seed =
  Moe.Quant
    (Nx_quant.mxfp4
       ~scales:
         (bytes [| experts; outputs; inputs / 32 |] seed (fun v ->
              120 + (v mod 14)))
       (bytes [| experts; outputs; inputs / 2 |] seed (fun v -> v mod 256)))

let params weight =
  {
    Moe.gate_up = weight ~inputs:width ~outputs:(2 * hidden) 1;
    gate_up_bias = floats [| experts; 2 * hidden |] 2;
    down = weight ~inputs:hidden ~outputs:width 3;
    down_bias = floats [| experts; width |] 4;
  }

(* Each part at its role's placement: [Experts] along axis 0. *)
let place placement (p : (_, Nx.host) Moe.t) =
  let at = placement Gpt_oss.Experts ~axis:0 in
  let weight = function
    | Moe.Float w -> Moe.Float (Nx.place at w)
    | Moe.Quant w -> Moe.Quant (Nx_quant.place at w)
  in
  {
    Moe.gate_up = weight p.gate_up;
    gate_up_bias = Nx.place at p.gate_up_bias;
    down = weight p.down;
    down_bias = Nx.place at p.down_bias;
  }

let bits t = Nx.to_array (Nx.bitcast Nx.int32 t)

let placement =
  Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal

let test_split weight () =
  let p = params weight in
  let ids =
    Nx.create Nx.int64 [| 5; 2 |]
      [| 0L; 3L; 1L; 2L; 3L; 1L; 0L; 2L; 2L; 3L |]
  in
  let weights = floats [| 5; 2 |] 9 and x = floats [| 5; width |] 7 in
  let host = Moe.apply ~limit p (ids, weights) x in
  let parallel = Gpt_oss.expert_parallel S.v in
  let whole v = Nx.place (parallel Gpt_oss.Whole ~axis:0) v in
  let y =
    Moe.apply ~limit (place parallel p) (whole ids, whole weights) (whole x)
  in
  equal ~msg:"the bits" (array int32) (bits host) (bits y);
  equal ~msg:"whole on each device" placement S.on (Nx.placement y)

let () =
  exit
    (run "gpt-oss mixture of experts"
       [
         group "expert parallelism"
           [
             test "float experts split give the bits of one device"
               (test_split float_weight);
             test "MXFP4 experts split give the bits of one device"
               (test_split mxfp4_weight);
           ];
       ])
