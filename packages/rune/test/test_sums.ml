(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Sums under rune: a matrix product over an axis cut across devices is a sum
   of the devices' terms. Its gradient agrees with finite differences, and
   lies where each parameter does, no sum; a compiled call equals the eager
   one and lies where it does; a map over a leading axis keeps the sum. *)

open Windtrap

let f64 = Nx.float64
let m = Nx_support.memory

module Four = (val Nx.devices [ m 0; m 1; m 2; m 3 ])

let invalid ~sub f = raises_match (Exn.invalid_arg ~sub) f
let k_rows = Four.split ~axis:0
let k_cols = Four.split ~axis:1
let elements x = Nx.to_array (Nx.place Nx.Host.on x)
let closed x = Nx.place (Nx.Placement.closed (Nx.placement x)) x

(* A product over [k], cut across Four, asked for its sum. *)
let mm x y = Nx.matmul ~sum:true x y
let placement = Nx_support.placement

(* [x] lies at a sum: its placement is not its closed one. *)
let is_sum ~msg x =
  let p = Nx.placement x in
  not_equal (placement ()) ~msg (Nx.Placement.closed p) p

(* A case: [a] of [[m; k]] cut along [k], [b] of [[k; n]] cut along [k],
   [c] of [[n]] whole, from drawn values. *)
type case = { m : int; k : int; n : int; seed : int }

let pp_case ppf c = Format.fprintf ppf "%d x %d by %d, seed %d" c.m c.k c.n c.seed

let case =
  Gen.with_pp pp_case
    (let open Gen in
     let* m = int_range 1 3 in
     let* k = int_range 1 7 in
     let* n = int_range 1 3 in
     let+ seed = int in
     { m; k; n; seed })

let values c =
  let st = Random.State.make [| c.seed |] in
  let draw s =
    Nx.init f64 s (fun _ -> Random.State.float st 2. -. 1.)
  in
  (draw [| c.m; c.k |], draw [| c.k; c.n |], draw [| c.n |])

(* The model: a product, a bias joined on each group's first device, a
   scale, closed, then a non-linear loss. *)
let model a b c =
  let y = Nx.add (mm a b) c in
  let y = Nx.mul y (Nx.scalar_like y 0.5) in
  Nx.sum (Nx.tanh (closed y))

let place c =
  let a, b, bias = values c in
  (Nx.place k_cols a, Nx.place k_rows b, Nx.place Four.on bias)

(* Central differences of [f] at [x], a host value, along each element. *)
let differences f x =
  let h = 1e-6 in
  let xs = Nx.to_array x and s = Nx.shape x in
  Array.mapi
    (fun i _ ->
      let at d =
        let ys = Array.copy xs in
        ys.(i) <- ys.(i) +. d;
        f (Nx.create f64 s ys)
      in
      (at h -. at (-.h)) /. (2. *. h))
    xs

(* A central difference at [h] errs by [h² |f'''| / 6 + u |f| / h]: about
   [1e-12] and [1e-10] relative for the model's tanh of entries in
   [-1, 1]. [1e-6] leaves room for the model's third derivative. *)
let close_to ~msg want got =
  equal int ~msg (Array.length want) (Array.length got);
  Array.iteri
    (fun i w ->
      let tol = 1e-6 *. (1. +. Float.abs w) in
      if Float.abs (w -. got.(i)) > tol then
        failf "%s: element %d is %.17g, finite differences give %.17g" msg i
          got.(i) w)
    want

let law_grad c =
  let a, b, bias = place c in
  let loss a b bias = Nx.item [] (model a b bias) in
  let ga = Rune.grad' (fun a -> model a b bias) a in
  let gb = Rune.grad' (fun b -> model a b bias) b in
  let gc = Rune.grad' (fun bias -> model a b bias) bias in
  equal (Nx_support.placement ()) ~msg:"a's gradient lies where a does"
    (Nx.placement a) (Nx.placement ga);
  equal (Nx_support.placement ()) ~msg:"b's gradient lies where b does"
    (Nx.placement b) (Nx.placement gb);
  equal (Nx_support.placement ()) ~msg:"the bias's gradient lies where it does"
    (Nx.placement bias) (Nx.placement gc);
  let ha, hb, hc = values c in
  let on p x = Nx.place p x in
  close_to ~msg:"a's gradient"
    (differences (fun a -> loss (on k_cols a) b bias) ha)
    (elements ga);
  close_to ~msg:"b's gradient"
    (differences (fun b -> loss a (on k_rows b) bias) hb)
    (elements gb);
  close_to ~msg:"the bias's gradient"
    (differences (fun bias -> loss a b (on Four.on bias)) hc)
    (elements gc)

(* The gradient of a sum, the loss itself a sum: its seed, and each
   cotangent, is no sum. *)
let law_grad_of_a_sum c =
  let a, b, _ = place c in
  let g = Rune.grad' (fun a -> Nx.sum (mm a b)) a in
  equal (Nx_support.placement ()) ~msg:"where it lies" (Nx.placement a)
    (Nx.placement g);
  let _, hb, _ = values c in
  let want = Nx.broadcast_to [| c.m; c.k |] (Nx.reshape [| 1; c.k |] (Nx.sum ~axes:[ 1 ] hb)) in
  close_to ~msg:"the rows of b summed" (Nx.to_array want) (elements g)

(* rune.mli's classes (Numerics, Reductions): a compiled contraction and
   an eager one are each within [γ(K + 1, 2u) Σ|a b|] of the exact sum, so
   within twice that of each other. Entries lie in [-1, 1] and [K <= 7], so
   [Σ|a b| <= 7]. A function of a few such products and maps is within a
   few times that, plus its maps' ulps, relative to [1 + |y|]. *)
let gamma n u = n *. u /. (1. -. (n *. u))
let u64 = epsilon_float /. 2.
let u32 = Float.ldexp 1. (-24)
let ub = Float.ldexp 1. (-8)
let product_tol u = 2. *. 7. *. gamma 8. (2. *. u)
let model_tol u = (4. *. product_tol u) +. (16. *. u)

(* The bits of [x] closed, NaNs alike. *)
let bits x = Array.map Int64.bits_of_float (elements x)

(* A compiled call lies where the eager one does: a sum where the eager
   result is one, and agrees with it within rune.mli's classes: a
   contraction's association is its own. *)
let law_jit c =
  let a, b, bias = place c in
  let f a = Nx.add (mm a b) bias in
  let eager = f a and compiled = Rune.jit' f a in
  is_sum ~msg:"the eager result is a sum" eager;
  equal (placement ()) ~msg:"where it lies" (Nx.placement eager)
    (Nx.placement compiled);
  let eq ~msg x y =
    Array.iteri
      (fun i v ->
        let w = (elements x).(i) in
        if Float.abs (v -. w) > model_tol u64 *. (1. +. Float.abs w) then
          failf "%s: element %d is %.17g, eager gives %.17g" msg i v w)
      (elements y)
  in
  eq ~msg:"the sum, closed" eager compiled;
  let g a = model a b bias in
  eq ~msg:"the model" (g a) (Rune.jit' g a);
  eq ~msg:"the model's gradient"
    (Rune.grad' g a)
    (Rune.jit' (Rune.grad' g) a)

(* A map over a leading axis of [a] keeps the sum of each lane. *)
let law_vmap c =
  let a, b, bias = place c in
  let f a = closed (Nx.add (mm a b) bias) in
  let stacked = Nx.stack [ a; Nx.neg a ] in
  let mapped = Rune.vmap' f stacked in
  let want = Nx.stack [ f a; f (Nx.neg a) ] in
  Array.iteri
    (fun i w ->
      let v = (elements mapped).(i) in
      if Float.abs (v -. w) > model_tol u64 *. (1. +. Float.abs w) then
        failf "element %d is %.17g, the lanes give %.17g" i v w)
    (elements want)

(* Constructs over a sum *)

(* A fixed case: [a] of [[2; 7]] cut along k, [b] of [[7; 3]], [xs] three
   rows of [[2; 7]] cut along k, and their host values. *)
let fixed () =
  let st = Random.State.make [| 7 |] in
  let draw s = Nx.init f64 s (fun _ -> Random.State.float st 2. -. 1.) in
  let ha = draw [| 2; 7 |] and hb = draw [| 7; 3 |] and hx = draw [| 3; 2; 7 |] in
  ( (ha, Nx.place k_cols ha),
    (hb, Nx.place k_rows hb),
    Nx.place (Four.split ~axis:2) hx )

let within ~msg ?(tol = model_tol u64) want got =
  let want = elements want and got = elements got in
  equal int ~msg (Array.length want) (Array.length got);
  Array.iteri
    (fun i w ->
      if Float.abs (w -. got.(i)) > tol *. (1. +. Float.abs w) then
        failf "%s: element %d is %.17g, want %.17g" msg i got.(i) w)
    want

(* A scan whose carry is a sum: each step adds a row's product to it and
   outputs it closed. *)
let step b c x =
  let c = Nx.add c (mm x b) in
  (c, closed c)

let test_scan () =
  let (ha, a), (_, b), xs = fixed () in
  let init = mm a b in
  let c, ys = Rune.scan' (step b) ~init xs in
  is_sum ~msg:"the carry stays a sum" c;
  (* The loop written out. *)
  let c' = ref init and ys' = ref [] in
  for i = 0 to 2 do
    let x = Nx.slice [ I i ] xs in
    let c, y = step b !c' x in
    c' := c;
    ys' := y :: !ys'
  done;
  equal (array int64) ~msg:"the carry, as the loop written out" (bits !c')
    (bits c);
  equal (array int64) ~msg:"the outputs, as the loop written out"
    (bits (Nx.stack (List.rev !ys')))
    (bits ys);
  let f init = fst (Rune.scan' (step b) ~init xs) in
  within ~msg:"compiled" (f init) (Rune.jit' f init);
  let g a =
    let c, ys = Rune.scan' (step b) ~init:(mm a b) xs in
    Nx.add (Nx.sum (Nx.tanh ys)) (Nx.sum (Nx.tanh (closed c)))
  in
  let loss a = Nx.item [] (g (Nx.place k_cols a)) in
  close_to ~msg:"the gradient through the carry and outputs"
    (differences loss ha)
    (elements (Rune.grad' g a));
  within ~msg:"the gradient compiled" (Rune.grad' g a)
    (Rune.jit' (Rune.grad' g) a);
  let stacked = Nx.stack [ a; Nx.neg a ] in
  let f a = fst (Rune.scan' (step b) ~init:(mm a b) xs) in
  within ~msg:"a map over the scan"
    (Nx.stack [ closed (f a); closed (f (Nx.neg a)) ])
    (Rune.vmap' f stacked)

let test_iterate () =
  let (ha, a), (_, b), _ = fixed () in
  let init = mm a b in
  let s0 = Nx.item [] (Nx.sum (Nx.abs (closed init))) in
  let until c = Nx.greater_s (Nx.sum (Nx.abs (closed c))) (3.5 *. s0) in
  let run a init = Rune.iterate' ~max:10 ~until (fun c -> Nx.add c (mm a b)) init in
  let c = run a init in
  is_sum ~msg:"the carry stays a sum" c;
  let g init = run a init in
  within ~msg:"compiled" (g init) (Rune.jit' g init);
  let loss a = Nx.sum (Nx.tanh (closed (run a (mm a b)))) in
  close_to ~msg:"the gradient"
    (differences (fun a -> Nx.item [] (loss (Nx.place k_cols a))) ha)
    (elements (Rune.grad' loss a))

let test_maps_and_remat () =
  let (ha, a), (_, b), _ = fixed () in
  let bias = Nx.place Four.on (Nx.create f64 [| 3 |] [| 0.25; -0.5; 1. |]) in
  let f a = Nx.add (mm a b) bias in
  let stacked = Nx.stack [ a; Nx.neg a ] in
  let r = Rune.vmap' f stacked in
  is_sum ~msg:"a map's result stays a sum" r;
  within ~msg:"a map's result"
    (Nx.stack [ closed (f a); closed (f (Nx.neg a)) ])
    r;
  let g a = Nx.sum (Nx.tanh (closed (f a))) in
  within ~msg:"a map of the gradient"
    (Nx.stack [ Rune.grad' g a; Rune.grad' g (Nx.neg a) ])
    (Rune.vmap' (Rune.grad' g) stacked);
  let over a =
    Nx.sum
      (Rune.remat Nx.Ptree.(tensor @-> returns tensor)
         (fun a -> Nx.tanh (closed (f a)))
         a)
  and returning a =
    Nx.sum
      (Nx.tanh
         (closed (Rune.remat Nx.Ptree.(tensor @-> returns tensor) f a)))
  in
  let loss h a = Nx.item [] (h (Nx.place k_cols a)) in
  close_to ~msg:"remat over a sum" (differences (loss over) ha)
    (elements (Rune.grad' over a));
  close_to ~msg:"remat returning a sum" (differences (loss returning) ha)
    (elements (Rune.grad' returning a));
  let t = Nx.place k_cols (Nx.create f64 [| 2; 7 |] (Array.init 14 (fun i -> Float.of_int (i mod 5) -. 2.))) in
  let y, ty = Rune.jvp' f a t in
  is_sum ~msg:"the primal a sum" y;
  is_sum ~msg:"its tangent a sum" ty;
  within ~msg:"the tangent, t b" (Nx.matmul t b) ty;
  let h a t = Rune.jvp' (fun a -> Nx.tanh (closed (mm a b))) a t in
  within ~msg:"jvp compiled" (snd (h a t))
    (snd
       (Rune.jit
          Nx.Ptree.(tensor @-> tensor @-> returns (pair tensor tensor))
          h a t))

(* The cotangent of a sum lies at its placement closed: a vjp takes it
   there, and raises at the sum's own. *)
let test_cotangents () =
  let (_, a), (hb, b), _ = fixed () in
  let y, pb = Rune.vjp' (fun a -> mm a b) a in
  let ct = Nx.ones_at (Nx.Placement.closed (Nx.placement y)) f64 [| 2; 3 |] in
  let g = pb ct in
  equal (placement ()) ~msg:"where the cotangent lies" (Nx.placement a)
    (Nx.placement g);
  within ~msg:"the cotangent"
    (Nx.matmul (Nx.ones f64 [| 2; 3 |]) (Nx.transpose hb))
    g;
  let lt = Rune.linear_transpose Nx.Ptree.tensor Nx.Ptree.tensor (fun a -> mm a b) a in
  within ~msg:"a linear transpose" g (lt ct);
  let custom =
    Rune.custom_vjp Nx.Ptree.tensor Nx.Ptree.tensor
      (fun a ->
        let y = mm a b in
        (y, fun ct -> Nx.matmul ct (Nx.transpose b)))
  in
  within ~msg:"a custom pullback's cotangent, a sum, closed" g
    (Rune.grad' (fun a -> Nx.sum (custom a)) a)

(* float32, complex64 and bfloat16 sums: a compiled product agrees with
   eager within its dtype's class, and a closed float32 gradient with
   float64's. *)
let test_dtypes () =
  let (_, a), (_, b), _ = fixed () in
  let a32 = Nx.cast Nx.float32 a and b32 = Nx.cast Nx.float32 b in
  let f a = Nx.mul_s (Nx.sub (mm a b32) (Nx.scalar_like a 0.5)) 3. in
  let e = f a32 and j = Rune.jit' f a32 in
  is_sum ~msg:"a float32 sum" e;
  within ~tol:(model_tol u32) ~msg:"float32 compiled" (Nx.cast f64 (closed e))
    (Nx.cast f64 (closed j));
  let g a = Nx.sum (Nx.tanh (closed (mm a b))) in
  within ~tol:(model_tol u32) ~msg:"float32's gradient"
    (Rune.grad' g a)
    (Nx.cast f64 (Rune.grad' (fun a -> Nx.sum (Nx.tanh (closed (mm a b32)))) a32));
  let c64 x = Nx.cast Nx.complex64 x in
  let ac = c64 a32 and bc = c64 b32 in
  let fc a = Nx.add (mm a bc) (Nx.scalar_like a { Complex.re = 1.; im = -1. }) in
  let e = fc ac and j = Rune.jit' fc ac in
  is_sum ~msg:"a complex64 sum" e;
  let parts x =
    Array.concat
      (List.map
         (fun (z : Complex.t) ->
           [| Int64.bits_of_float z.re; Int64.bits_of_float z.im |])
         (Array.to_list (Nx.to_array (Nx.place Nx.Host.on (closed x)))))
  in
  let close_parts ~msg x y =
    Array.iteri
      (fun i v ->
        let w = Int64.float_of_bits x.(i) and v = Int64.float_of_bits v in
        if Float.abs (v -. w) > model_tol u32 *. (1. +. Float.abs w) then
          failf "%s: part %d is %g, eager gives %g" msg i v w)
      y
  in
  close_parts ~msg:"complex64 compiled" (parts e) (parts j);
  (* bfloat16: a compiled sum and the eager one each close within
     [γ(K + 1, 2u') Σ|a b| + γ(4, u) Σ|term|] of the exact sum,
     [Σ|term| <= Σ|a b| <= 7]. *)
  let bf x = Nx.cast Nx.bfloat16 x in
  let ab = bf a and bb = bf b in
  let fb a = mm a bb in
  let e = fb ab and j = Rune.jit' fb ab in
  is_sum ~msg:"a bfloat16 sum" e;
  let read x = Nx.to_array (Nx.cast f64 (Nx.place Nx.Host.on (closed x))) in
  let tol = 2. *. 7. *. (gamma 8. (2. *. u32) +. gamma 4. ub) in
  Array.iteri
    (fun i w ->
      let v = (read j).(i) in
      if Float.abs (v -. w) > tol then
        failf "bfloat16 compiled: element %d is %g, eager gives %g" i v w)
    (read e)

let test_refusals () =
  let a, b, _ = place { m = 2; k = 4; n = 2; seed = 0 } in
  invalid ~sub:"a sum the operation does not keep"
    (fun () -> Rune.grad' (fun a -> Nx.sum (Nx.exp (mm a b))) a);
  invalid ~sub:"a sum the operation does not keep"
    (fun () -> Rune.jit' (fun a -> Nx.exp (mm a b)) a)

(* A bfloat16 weight's gradient over a batch cut across Four: the
   transposed product contracts each device's rows into float32, adds the
   devices' partials from the first window's, in C order, and rounds once
   (nx.mli's contraction over cut summed axes), so its bits are those of
   that rule computed on the host. *)
type batch = { rows : int; ins : int; outs : int; bseed : int }

let pp_batch ppf c =
  Format.fprintf ppf "[%d; %d] by [%d; %d], seed %d" c.rows c.ins c.ins c.outs
    c.bseed

let batch =
  Gen.with_pp pp_batch
    (let open Gen in
     let* rows = int_range 1 9 in
     let* ins = int_range 1 3 in
     let* outs = int_range 1 3 in
     let+ bseed = int in
     { rows; ins; outs; bseed })

let law_bf16_grad c =
  cover "every device holds rows" (c.rows >= 4);
  let st = Random.State.make [| c.bseed |] in
  let draw s =
    Nx.cast Nx.bfloat16
      (Nx.init f64 s (fun _ -> Random.State.float st 2. -. 1.))
  in
  let x = draw [| c.rows; c.ins |] and w = draw [| c.ins; c.outs |] in
  let grad =
    Rune.grad'
      (fun w -> Nx.sum (Nx.matmul (Nx.place k_rows x) w))
      (Nx.place Four.on w)
  in
  (* Window [j] of the rows, as a split cuts them. *)
  let tile j =
    let q = c.rows / 4 and r = c.rows mod 4 in
    ((j * q) + Int.min j r, q + if j < r then 1 else 0)
  in
  let partial j =
    let lo, n = tile j in
    Nx.contract Nx.float32 (Nx.Pattern.v "b i, b o -> i o | b")
      (Nx.slice [ R (lo, lo + n); A ] x)
      (Nx.ones Nx.bfloat16 [| n; c.outs |])
  in
  let held = List.filter (fun j -> snd (tile j) > 0) [ 0; 1; 2; 3 ] in
  let want =
    match held with
    | j :: rest -> List.fold_left (fun s j -> Nx.add s (partial j)) (partial j) rest
    | [] -> assert false
  in
  let bits x =
    Array.map Int64.bits_of_float
      (Nx.to_array (Nx.cast f64 (Nx.place Nx.Host.on x)))
  in
  equal (array int64) ~msg:"the gradient's bits"
    (bits (Nx.cast Nx.bfloat16 want))
    (bits grad)

let () =
  exit
    (run "rune sums"
       [
         group "sums"
           [
             prop ~count:50 "grad through a sum agrees with finite differences"
               case law_grad;
             prop ~count:50 "the gradient of a sum is no sum" case
               law_grad_of_a_sum;
             prop ~count:8 "jit of a sum equals eager" case law_jit;
             prop ~count:30 "vmap over a sum equals its lanes" case law_vmap;
             test "an operation not linear in a sum raises" test_refusals;
             prop ~count:100
               "a bfloat16 gradient over a cut batch closes its partials in \
                float32, in window order"
               batch law_bf16_grad;
           ];
         group "constructs"
           [
             test "a scan whose carry is a sum" test_scan;
             test "an iterate whose carry is a sum" test_iterate;
             test "maps, remat and jvp over a sum" test_maps_and_remat;
             test "a sum's cotangent lies whole" test_cotangents;
             test "float32, complex64 and bfloat16 sums" test_dtypes;
           ];
       ])
