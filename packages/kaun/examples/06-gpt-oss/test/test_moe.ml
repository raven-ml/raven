(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The mixture of experts over four memory devices. At gpt-oss's
   expert-parallel placement, experts split along their expert axis, each
   device computes its own experts' rows and [+0] in the others': the gate
   and up projection closes before the activation, adding exact zeros, and
   the down projection's output is a sum that one closing after the top-k
   combine closes, within float32's reassociation of the combine's terms of
   the experts on one device, float and MXFP4. Row parallel, each expert cut
   along its features, the down projection's output is a sum that one
   closing after the top-k combine closes, eagerly and compiled. *)

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

let placement =
  Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal

(* The largest gap between [want] and [got], each element, for a baseline:
   the devices reassociate float32 sums one device sums otherwise. Baselines
   print it with [%h], exact and the same on every system: the C library's
   [%g] writes 1.19209e-007 under mingw-w64. *)
let gap want got =
  let want = Nx.to_array want and got = Nx.to_array got in
  let g = ref 0. in
  Array.iteri (fun i w -> g := Float.max !g (Float.abs (w -. got.(i)))) want;
  !g

let close x = Nx.place (Nx.Placement.closed (Nx.placement x)) x

let test_split weight baseline () =
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
    close
      (Moe.apply ~limit (place parallel p) (whole ids, whole weights) (whole x))
  in
  expect (Printf.sprintf "largest gap %h" (gap host y)) baseline;
  equal ~msg:"whole on each device" placement S.on (Nx.placement y)

(* Row parallelism: [gate_up] cut along its outputs and [down] along its
   inputs, the hidden features, each device holding a quarter of each
   expert. [down]'s product is a sum of the devices' terms, which the top-k
   combine keeps, and one closing after it gives the experts' output. *)
let rows (p : (_, Nx.host) Moe.t) =
  (* A packed weight cut along its inputs splits its one block of scales:
     they lie whole on each device, which reads its codes where they lie. *)
  let weight axis = function
    | Moe.Float w -> Moe.Float (Nx.place (S.split ~axis) w)
    | Moe.Quant (Nx_quant.Mxfp4 { codes; scales }) when axis = 2 ->
        Moe.Quant
          (Nx_quant.mxfp4 ~scales:(Nx.place S.on scales)
             (Nx.place (S.split ~axis) codes))
    | Moe.Quant w -> Moe.Quant (Nx_quant.place (S.split ~axis) w)
  in
  let quant = match p.gate_up with Moe.Quant _ -> true | Moe.Float _ -> false in
  {
    Moe.gate_up = weight (if quant then 1 else 2) p.gate_up;
    gate_up_bias = Nx.place (S.split ~axis:1) p.gate_up_bias;
    down = weight (if quant then 2 else 1) p.down;
    down_bias = Nx.place S.on p.down_bias;
  }

(* The lines of [f]'s compiled program at [x] that close a sum: place a
   value that lies partial. *)
let places f x =
  let program = Rune.program Nx.Ptree.tensor Nx.Ptree.tensor f x in
  let lines =
    String.split_on_char '\n' (Format.asprintf "%a" Rune.pp_program program)
  in
  let has sub l =
    let k = String.length sub and n = String.length l in
    let rec go i = i + k <= n && (String.sub l i k = sub || go (i + 1)) in
    go 0
  in
  List.filter (fun l -> has "= Place (" l && has "partial" l) lines

let test_rows weight baseline () =
  let p = params weight in
  let ids =
    Nx.create Nx.int64 [| 5; 4 |]
      [| 0L; 3L; 1L; 2L; 3L; 1L; 0L; 2L; 2L; 3L; 0L; 1L; 1L; 0L; 3L; 2L;
         2L; 0L; 1L; 3L |]
  in
  let weights = floats [| 5; 4 |] 9 and x = floats [| 5; width |] 7 in
  let host = Moe.apply ~limit p (ids, weights) x in
  let on v = Nx.place S.on v in
  let p = rows p and ids = on ids and weights = on weights in
  let f x = Moe.apply ~limit p (ids, weights) x in
  let y = f (on x) in
  not_equal placement ~msg:"the experts' output is a sum" S.on
    (Nx.placement y);
  let eager = gap host (close y) in
  let step x = close (f x) in
  let compiled = gap host (Rune.jit' step (on x)) in
  expect (Printf.sprintf "eager %h, compiled %h" eager compiled) baseline;
  equal ~msg:"one closing, after the combine" int 1
    (List.length (places step (on x)))

(* At bfloat16, row parallel: each device's term of [down]'s product rounds
   into bfloat16, the combine keeps the sum at bfloat16 and the closing adds
   the four terms there. The largest gap from one device's output, a
   baseline. *)
let test_rows_bf16 () =
  let bf x = Nx.cast Nx.bfloat16 x in
  let p = params float_weight in
  let weight = function
    | Moe.Float w -> Moe.Float (bf w)
    | Moe.Quant _ -> assert false
  in
  let p =
    {
      Moe.gate_up = weight p.gate_up;
      gate_up_bias = bf p.gate_up_bias;
      down = weight p.down;
      down_bias = bf p.down_bias;
    }
  in
  let ids =
    Nx.create Nx.int64 [| 5; 4 |]
      [| 0L; 3L; 1L; 2L; 3L; 1L; 0L; 2L; 2L; 3L; 0L; 1L; 1L; 0L; 3L; 2L;
         2L; 0L; 1L; 3L |]
  in
  let weights = bf (floats [| 5; 4 |] 9) and x = bf (floats [| 5; width |] 7) in
  let host = Moe.apply ~limit p (ids, weights) x in
  let on v = Nx.place S.on v in
  let p = rows p and ids = on ids and weights = on weights in
  let step x = close (Moe.apply ~limit p (ids, weights) x) in
  expect (Printf.sprintf "largest gap %h" (gap host (step (on x))))
  @@ __POS_OF__ {|
    largest gap 0x1p-7
  |};
  equal ~msg:"one closing, compiled" int 1 (List.length (places step (on x)))

(* Expert parallelism compiled: each device computes its experts' rows, so
   the down projection's product is a sum, which the step closes once, after
   the combine. *)
let test_split_compiled weight baseline () =
  let p = params weight in
  let ids =
    Nx.create Nx.int64 [| 5; 4 |]
      [| 0L; 3L; 1L; 2L; 3L; 1L; 0L; 2L; 2L; 3L; 0L; 1L; 1L; 0L; 3L; 2L;
         2L; 0L; 1L; 3L |]
  in
  let weights = floats [| 5; 4 |] 9 and x = floats [| 5; width |] 7 in
  let host = Moe.apply ~limit p (ids, weights) x in
  let on v = Nx.place S.on v in
  let p = place (Gpt_oss.expert_parallel S.v) p in
  let ids = on ids and weights = on weights in
  let step x = close (Moe.apply ~limit p (ids, weights) x) in
  expect
    (Printf.sprintf "compiled %h" (gap host (Rune.jit' step (on x))))
    baseline;
  equal ~msg:"one closing, after the combine" int 1
    (List.length (places step (on x)))

let () =
  exit
    (run "gpt-oss mixture of experts"
       [
         group "expert parallelism"
           [
             test "float experts split are one device's, reassociated"
               (test_split float_weight @@ __POS_OF__ {|
                 largest gap 0x1p-24
               |});
             test "MXFP4 experts split are one device's, reassociated"
               (test_split mxfp4_weight @@ __POS_OF__ {|
                 largest gap 0x1p-11
               |});
           ];
         group "expert parallelism, compiled"
           [
             test "float experts split compile, closing once"
               (test_split_compiled float_weight @@ __POS_OF__ {|
                 compiled 0x1p-23
               |});
             test "MXFP4 experts split compile, closing once"
               (test_split_compiled mxfp4_weight @@ __POS_OF__ {|
                 compiled 0x1.8p-9
               |});
           ];
         group "row parallelism"
           [
             test "float experts cut along their features close once"
               (test_rows float_weight @@ __POS_OF__ {|
                 eager 0x1p-23, compiled 0x1p-23
               |});
             test "MXFP4 experts cut along their features close once"
               (test_rows mxfp4_weight @@ __POS_OF__ {|
                 eager 0x1.8p-9, compiled 0x1p-9
               |});
             test "bfloat16 experts cut along their features close once"
               test_rows_bf16;
           ];
       ])
