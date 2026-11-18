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

(* [bound p (ids, weights) x] is, each element, the largest gap between two
   computations of [Moe.apply ~limit p (ids, weights) x] that associate
   their float32 sums in any order, eagerly or compiled: twice the bound of
   each from the exact value, to first order in float32's unit roundoff
   [u], as rune.mli's numerics bound a compiled call, with
   [γ(n) = n u / (1 - n u)]. It is computed at float64 from the exact
   values, the decoded weights [W1] and [W2] exact there (Nx_quant.dequant).

   - [h = x W1 + b1] holds Nx.contract's bound over 32 products,
     [γ(33, 2u) = γ(66)], and rounds once at its bias:
     [e_h = γ(66) Σ|x| |W1| + u |h|].
   - The activation [a = f(g) (l + 1)], [f(g) = g σ(c g)], [c = 1.702],
     reads [h]'s features clamped, which keeps no error where both
     computations clamp: [g - e_g >= limit] or [|l| - e_l >= limit]. It
     carries [|f'(g)| |l + 1| e_g + |f(g)| e_l], [f' = σ (1 + c g (1 - σ))],
     and adds the roundings at [c g], seen through [σ' <= 1/4], at [σ],
     within [11u] (eager, 3 ulps; compiled as nx composes it, exp's 9 ulps
     through [t / (1 + t)] and [1 - q]), and at [g σ], [l + 1] and their
     product, [u |l + 1| (c g² / 4 + 11 |g| + 3 |f|)]: that is [e_a].
   - An output's terms [w_k a_j W2] pass the down projection's 32 products,
     [γ(66)] again, its bias, the weight's product, the sum over the four
     slots, [γ(3)], and the closing of four devices' terms, [γ(3)]: [γ(74)]
     in all (Higham, Accuracy and Stability of Numerical Algorithms, Lemma
     3.3), on top of the activation's error:
     [Σ_k |w_k| (Σ_j e_a |W2| + γ(74) (Σ_j |a| |W2| + |b2|))]. *)
let bound (p : (_, Nx.host) Moe.t) (ids, weights) x =
  (* The gate features are the even ones, the linear the odd. *)
  let gate t = Nx.slice [ A; A; Rs (0, 2 * hidden, 2) ] t in
  let linear t = Nx.slice [ A; A; Rs (1, 2 * hidden, 2) ] t in
  let open Nx.Infix in
  let u = 0x1p-24 and c = 1.702 in
  let gamma n = Float.of_int n *. u /. (1. -. (Float.of_int n *. u)) in
  let f64 = Nx.cast Nx.float64 in
  let matrices = function
    | Moe.Float w -> f64 w
    | Moe.Quant w -> Nx.matrix_transpose (Nx_quant.dequant Nx.float64 w)
  in
  let w1 = matrices p.gate_up and w2 = Nx.abs (matrices p.down) in
  let products = Nx.matmul_segments ids in
  let x = Nx.unsqueeze ~axes:[ -2 ] (f64 x) in
  let h = products x w1 + Nx.take ~axis:0 ids (f64 p.gate_up_bias) in
  let e_h = (products (Nx.abs x) (Nx.abs w1) *$ gamma 66) + (Nx.abs h *$ u) in

  let unclamped v e =
    Nx.where (v - e >= Nx.scalar_like v limit) (Nx.zeros_like e) e
  in
  let e_g = unclamped (gate h) (gate e_h) in
  let e_l = unclamped (Nx.abs (linear h)) (linear e_h) in
  let g = Nx.clamp ~max:limit (gate h) in
  let l1 = Nx.abs (Nx.clamp ~min:(-.limit) ~max:limit (linear h) +$ 1.) in
  let s = Nx.sigmoid (g *$ c) in
  let f = Nx.abs (g * s) in
  let f' = Nx.abs (s * (((g *$ c) * (~-s +$ 1.)) +$ 1.)) in
  let rounding = (Nx.square g *$ (c /. 4.)) + (Nx.abs g *$ 11.) + (f *$ 3.) in
  let e_a = (f' * l1 * e_g) + (f * e_l) + (l1 * rounding *$ u) in

  let b2 = Nx.abs (Nx.take ~axis:0 ids (f64 p.down_bias)) in
  let y = products e_a w2 + ((products (f * l1) w2 + b2) *$ gamma 74) in
  let w = Nx.unsqueeze ~axes:[ -1 ] (Nx.abs (f64 weights)) in
  Nx.sum ~axes:[ -2 ] (y * w) *$ 2.

(* [within bound want got] asserts that each element of [got] is within
   [bound]'s of [want]'s, reporting the element nearest its bound, a NaN as
   an infinite gap. *)
let within bound want got =
  let want = Nx.to_array want and got = Nx.to_array got in
  let bound = Nx.to_array bound in
  let gap i =
    let d = Float.abs (want.(i) -. got.(i)) in
    if Float.is_nan d then Float.infinity else d
  in
  let worst = ref 0 in
  Array.iteri
    (fun i b -> if gap i /. b > gap !worst /. bound.(!worst) then worst := i)
    bound;
  let i = !worst in
  at_most float_exact
    ~msg:(Printf.sprintf "compiled gap at element %d" i)
    ~than:bound.(i) (gap i)

(* Expert parallelism compiled: each device computes its experts' rows, so
   the down projection's product is a sum, which the step closes once, after
   the combine. *)
let test_split_compiled weight () =
  let p = params weight in
  let ids =
    Nx.create Nx.int64 [| 5; 4 |]
      [| 0L; 3L; 1L; 2L; 3L; 1L; 0L; 2L; 2L; 3L; 0L; 1L; 1L; 0L; 3L; 2L;
         2L; 0L; 1L; 3L |]
  in
  let weights = floats [| 5; 4 |] 9 and x = floats [| 5; width |] 7 in
  let host = Moe.apply ~limit p (ids, weights) x in
  let bound = bound p (ids, weights) x in
  let on v = Nx.place S.on v in
  let p = place (Gpt_oss.expert_parallel S.v) p in
  let ids = on ids and weights = on weights in
  let step x = close (Moe.apply ~limit p (ids, weights) x) in
  within bound host (Rune.jit' step (on x));
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
  let ids =
    Nx.create Nx.int64 [| 5; 2 |]
      [| 0L; 3L; 1L; 2L; 3L; 1L; 0L; 2L; 2L; 3L |]
  in
  let weights = floats [| 5; 2 |] 9 and x = floats [| 5; width |] 7 in
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
