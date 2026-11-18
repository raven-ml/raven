(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The mixture of experts: the routing among equal logits, each token's output
   independent of the others', and expert parallelism over four software
   devices. At gpt-oss's expert-parallel placement, experts split along their
   expert axis, each device computes its own experts' rows and [+0] in the
   others': the gate and up projection closes before the activation, adding
   exact zeros, and the down projection's output is a sum that one closing
   after the top-k combine closes, within float32's reassociation of the
   combine's terms of the experts on one device, float and MXFP4. *)

open Windtrap

let memory k = Result.get_ok (Rig.software (Printf.sprintf "m%d" k))

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

(* Five tokens' routings to two of the four experts each, and inputs. *)
let ids =
  Nx.create Nx.int64 [| 5; 2 |] [| 0L; 3L; 1L; 2L; 3L; 1L; 0L; 2L; 2L; 3L |]

let weights = floats [| 5; 2 |] 9
let x = floats [| 5; width |] 7

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
  let host = Moe.apply ~limit p (ids, weights) x in
  let parallel = Gpt_oss.expert_parallel S.v in
  let whole v = Nx.place (parallel Gpt_oss.Whole ~axis:0) v in
  let y =
    close
      (Moe.apply ~limit (place parallel p) (whole ids, whole weights) (whole x))
  in
  expect (Printf.sprintf "largest gap %h" (gap host y)) baseline;
  equal ~msg:"whole on each device" placement S.on (Nx.placement y)

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

(* Expert parallelism compiled: each device computes its experts' rows, so
   the down projection's product is a sum, which the step closes once, after
   the combine. *)
let test_split_compiled weight () =
  let p = params weight in
  let host = Moe.apply ~limit p (ids, weights) x in
  let on v = Nx.place S.on v in
  let p = place (Gpt_oss.expert_parallel S.v) p in
  let ids = on ids and weights = on weights in
  let step x = close (Moe.apply ~limit p (ids, weights) x) in
  (* The devices sum their terms in another order than one device does; a
     closing in the wrong place errs by the size of the output. *)
  Fixture.close ~tol:1e-5 "compiled"
    (Nx.to_array host)
    (Nx.to_array (Rune.jit' step (on x)));
  equal ~msg:"one closing, after the combine" int 1
    (List.length (places step (on x)))

(* Among equal logits the lowest expert comes first, and exactly [k]
   distinct experts are selected. *)
let test_ties () =
  let row values = Array.init 32 (fun e -> List.assoc_opt e values) in
  let rows =
    [
      (row [], [ 0; 1; 2; 3 ]);
      (row [ (6, 1.); (13, 1.); (20, 1.); (27, 1.) ], [ 6; 13; 20; 27 ]);
      ( row [ (5, 3.); (9, 3.); (20, 1.5); (21, 1.5); (30, 1.5) ],
        [ 5; 9; 20; 21 ] );
    ]
  in
  let logits =
    Array.concat
      (List.map (fun (r, _) -> Array.map (Option.value ~default:0.) r) rows)
  in
  let ids, _ = Moe.route ~k:4 (Nx.create Nx.float32 [| 3; 32 |] logits) in
  equal ~msg:"experts" (list (list int))
    (List.map snd rows)
    (List.init 3 (fun r ->
         List.init 4 (fun j -> Int64.to_int (Nx.item [ r; j ] ids))))

(* A token's output is its own: each row of a batch is the token's output
   alone. *)
let test_independent weight () =
  let p = params weight in
  let whole = Moe.apply ~limit p (ids, weights) x in
  for t = 0 to 4 do
    let one v = Nx.slice [ R (t, t + 1) ] v in
    equal
      ~msg:(Printf.sprintf "token %d" t)
      (array (float_rel ~rel:1e-5 ~abs:1e-6))
      (Nx.to_array (one whole))
      (Nx.to_array (Moe.apply ~limit p (one ids, one weights) (one x)))
  done

let () =
  exit
    (run "gpt-oss mixture of experts"
       [
         group "routing"
           [
             test "equal logits: the lowest experts first" test_ties;
             test "float experts: each token alone"
               (test_independent float_weight);
             test "MXFP4 experts: each token alone"
               (test_independent mxfp4_weight);
           ];
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
               (test_split_compiled float_weight);
             test "MXFP4 experts split compile, closing once"
               (test_split_compiled mxfp4_weight);
           ];
       ])
