(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Each optimizer's step against the formula its documentation states, in plain
   floats, leaf by leaf, from drawn parameters, gradients, states, counters,
   rates and coefficients. A coefficient drawn as [None] is left out, and the
   formula takes the documented default. *)

open Windtrap

(* Parameters: a matrix and a vector at float64, so that every operation of a
   step is a float64 one. Adafactor factors the one and not the other, and a
   norm over a leaf (LARS, LAMB, Adafactor's clipping) differs between them. *)
module Wv = struct
  type t = { w : Nx.host Nx.float64_t; v : Nx.host Nx.float64_t }

  let walk c { w; v } =
    let open Nx.Ptree.Walk in
    let w = field c "w" tensor w in
    let v = field c "v" tensor v in
    { w; v }

  let ptree : (t, Nx.host) Nx.Ptree.t = Nx.Ptree.v walk
end

let p = Wv.ptree

(* A value of [Wv] from its leaves' elements in C order: the matrix's, of [m]
   rows and [n] columns, and the vector's, of [n]. *)
let make (m, n) (w, v) =
  {
    Wv.w = Nx.create Nx.float64 [| m; n |] w;
    v = Nx.create Nx.float64 [| n |] v;
  }

(* Each leaf by name: its drawn elements and its stepped ones. *)
let leaves =
  [
    ("w", fst, fun (x : Wv.t) -> Nx.to_array x.w);
    ("v", snd, fun (x : Wv.t) -> Nx.to_array x.v);
  ]

let counter c = Nx.scalar Nx.int32 (Int32.of_int c)

(* A rate of the counter, [a * (1 + c / 1000)] at the counter [c], exact at
   float64. A step that read it at another counter would be off by at least
   a two-thousandth of it. *)
let rate a step =
  Nx.mul_s (Nx.add_s (Nx.div_s (Nx.cast Nx.float64 step) 1000.0) 1.0) a

let rate_at a c = a *. (1.0 +. (Float.of_int c /. 1000.0))

(* [Vega.rate]'s number, the float32 nearest [x]: a default decay. *)
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* Draws *)

let element = Gen.(frequency [ (1, constant 0.0); (6, float_range (-4.0) 4.0) ])

(* Two leaves' elements, of shape [(m, n)], all non-negative with [positive], as
   a second moment's. *)
let values ?(positive = false) (m, n) =
  let open Gen in
  let one = if positive then map Float.abs element else element in
  pair (array ~size:(constant (m * n)) one) (array ~size:(constant n) one)

let shape = Gen.(pair (int_range 0 3) (int_range 0 3))

let completed =
  Gen.(frequency [ (1, constant 0); (1, int_range 1 3); (3, int_range 4 1000) ])

let factor = Gen.(frequency [ (1, constant 0.0); (4, float_range 0.0 2.0) ])

(* An optional argument: left out a third of the time. *)
let opt g = Gen.(frequency [ (1, constant None); (2, map Option.some g) ])

(* A coefficient of a moving average or a momentum, in [[0; 0.999]]. *)
let coefficient =
  Gen.(
    frequency
      [ (1, constant 0.0); (1, constant 0.999); (4, float_range 0.0 0.999) ])

let eps = Gen.of_list [ 1e-8; 1e-3; 0.5 ]

(* The breaking cases every law draws: an empty leaf, a matrix of several rows
   and columns, the first step. *)
let cover_draw (m, n) c =
  cover "an empty leaf" (m * n = 0);
  cover "a matrix of several rows and columns" (m > 1 && n > 1);
  cover "the first step" (c = 0)

let cover_defaults names =
  List.iter
    (fun (name, o) -> cover ("the default " ^ name) (Option.is_none o))
    names

(* Agreement *)

(* Both sides round a chain of under forty float64 operations, each within
   [2^-53] of its exact value. The draws bound what amplifies the rounding:
   values of magnitude at most 4 and moments at most 16; RAdam's [rho_inf / (rho
   - 4)] below 2000, as [b2 <= 0.999] gives [rho_inf <= 1999] and a rectified
   step has [rho >= 5]; and L-BFGS's pairs, whose curvature is at least a
   quarter of [|y| |s|]. A result is then within [40 * 2000 * 2^-53 < 1e-11] of
   its magnitude, or of 8 where a last subtraction cancels. *)
let close = float_rel ~rel:1e-10 ~abs:1e-10

let agree msg expected actual = equal ~msg (array close) expected actual

let advanced c (step : Nx.host Nx.int32_t) =
  equal ~msg:"step" int32 (Int32.of_int (c + 1)) (Nx.item [] step)

let map3 f a b c = Array.init (Array.length a) (fun i -> f a.(i) b.(i) c.(i))
let norm a = sqrt (Array.fold_left (fun s x -> s +. (x *. x)) 0.0 a)
let average b m x = Array.map2 (fun m x -> (b *. m) +. ((1.0 -. b) *. x)) m x
let descend lr x d = Array.map2 (fun x d -> x -. (lr *. d)) x d

(* [1 - b^t], exactly up to rounding. *)
let bias b t = -.Float.expm1 (Float.of_int t *. log b)

(* [refuses name f] checks that [f] raises [Invalid_argument] naming [name]. *)
let refuses (name, f) = raises_match (Exn.invalid_arg ~sub:name) f

(* SGD *)

let sgd_formula =
  let gen =
    let open Gen in
    let* s = shape in
    let+ x = values s
    and+ g = values s
    and+ v = values s
    and+ momentum = opt coefficient
    and+ lr = factor
    and+ c = completed in
    (s, x, g, v, momentum, lr, c)
  in
  prop "sgd_step is its formula" gen (fun (s, x, g, v, momentum, lr, c) ->
      cover_draw s c;
      cover_defaults [ ("momentum", momentum) ];
      let st = { Vega.velocity = make s v; step = counter c } in
      let x', st' =
        Vega.sgd_step p ~lr:(rate lr) ?momentum st ~params:(make s x)
          ~grads:(make s g)
      in
      let momentum = Option.value momentum ~default:0.0 in
      let lr = rate_at lr c in
      List.iter
        (fun (name, leaf, stepped) ->
          (* v' = momentum * v + g; p' = p - lr * v' *)
          let v' =
            Array.map2 (fun v g -> (momentum *. v) +. g) (leaf v) (leaf g)
          in
          agree ("velocity." ^ name) v' (stepped st'.velocity);
          agree name (descend lr (leaf x) v') (stepped x'))
        leaves;
      advanced c st'.step)

(* Plain descent does not read the velocity: one of another skeleton is no
   error. *)
let test_sgd_reads_no_velocity () =
  let p = Nx.Ptree.(list tensor) in
  let x = [ Nx.ones Nx.float64 [| 2 |]; Nx.ones Nx.float64 [| 3 |] ] in
  let st = { (Vega.sgd_init p x) with velocity = [] } in
  let x', _ = Vega.sgd_step p ~lr:(rate 0.5) st ~params:x ~grads:x in
  equal
    (list (array float_exact))
    [ [| 0.5; 0.5 |]; [| 0.5; 0.5; 0.5 |] ]
    (List.map Nx.to_array x')

let sgd_refusals =
  let x = Nx.ones Nx.float64 [| 2 |] and p = Nx.Ptree.tensor in
  let step momentum () =
    Vega.sgd_step p ~lr:(rate 0.1) ~momentum (Vega.sgd_init p x) ~params:x
      ~grads:x
  in
  cases "sgd_step refuses a momentum outside [0; 1)" ~id:fst
    [
      ("momentum = 1.", step 1.0);
      ("momentum = -0.1", step (-0.1));
      ("momentum = nan", step Float.nan);
    ]
    (fun (_, f) -> refuses ("momentum", f))

(* LARS *)

(* The trust ratio of the leaf [x] and its update [u]: [eta * |x| / |u|], or [1]
   where either norm is zero. *)
let trust eta x u =
  let x = norm x and u = norm u in
  if x > 0.0 && u > 0.0 then eta *. x /. u else 1.0

let lars_formula =
  let gen =
    let open Gen in
    let* s = shape in
    let+ x = values s
    and+ g = values s
    and+ v = values s
    and+ momentum = opt coefficient
    and+ eta = opt (float_range 1e-3 2.0)
    and+ wd = opt factor
    and+ nesterov = opt bool
    and+ lr = factor
    and+ c = completed in
    (s, x, g, v, (momentum, eta, wd, nesterov), lr, c)
  in
  (* A zero leaf takes [r = 1], the momentum SGD step at the rate; a gradient
     that cancels the decay is a zero update, which keeps its leaf. Both took
     the trust coefficient once. *)
  let zero = ([| 0.0; 0.0 |], [| 0.0 |])
  and ones = ([| 1.0; -2.0 |], [| 3.0 |]) in
  let examples =
    [
      ((2, 1), zero, ones, zero, (None, None, None, None), 0.5, 0);
      ( (2, 1),
        ([| 2.0; -4.0 |], [| 1.0 |]),
        ([| -1.0; 2.0 |], [| -0.5 |]),
        zero,
        (None, None, Some 0.5, None),
        0.5,
        0 );
    ]
  in
  prop "lars_step is its formula" ~examples gen
    (fun (s, x, g, v, (momentum, eta, wd, nesterov), lr, c) ->
      cover_draw s c;
      cover_defaults
        [
          ("momentum", momentum);
          ("trust_coefficient", eta);
          ("weight_decay", wd);
        ];
      cover "the default nesterov" (Option.is_none nesterov);
      cover "nesterov" (nesterov = Some true);
      let st = { Vega.velocity = make s v; step = counter c } in
      let x', st' =
        Vega.lars_step p ~lr:(rate lr) ?momentum ?trust_coefficient:eta
          ?weight_decay:(Option.map rate wd) ?nesterov st ~params:(make s x)
          ~grads:(make s g)
      in
      let momentum = Option.value momentum ~default:0.9 in
      let eta = Option.value eta ~default:0.001 in
      let wd = match wd with Some wd -> rate_at wd c | None -> f32 0.01 in
      let nesterov = Option.value nesterov ~default:false in
      let lr = rate_at lr c in
      List.iter
        (fun (name, leaf, stepped) ->
          let x = leaf x in
          (* u = g + wd * p; r = eta * |p| / |u|, or 1; v' = momentum * v + r
           * u; p' = p - lr * v', or p - lr * (r * u + momentum * v') *)
          let u = Array.map2 (fun g x -> g +. (wd *. x)) (leaf g) x in
          let r = trust eta x u in
          let v' =
            Array.map2 (fun v u -> (momentum *. v) +. (r *. u)) (leaf v) u
          in
          let d =
            if nesterov then
              Array.map2 (fun u v' -> (r *. u) +. (momentum *. v')) u v'
            else v'
          in
          agree ("velocity." ^ name) v' (stepped st'.velocity);
          agree name (descend lr x d) (stepped x'))
        leaves;
      advanced c st'.step)

let lars_refusals =
  let x = Nx.ones Nx.float64 [| 2 |] and p = Nx.Ptree.tensor in
  let step ?momentum ?trust_coefficient () =
    ignore
      (Vega.lars_step p ~lr:(rate 0.1) ?momentum ?trust_coefficient
         (Vega.lars_init p x) ~params:x ~grads:x)
  in
  let trust c =
    ( Printf.sprintf "trust_coefficient = %h" c,
      ("trust_coefficient", fun () -> step ~trust_coefficient:c ()) )
  in
  cases "lars_step refuses a momentum or a trust coefficient out of range"
    ~id:fst
    (("momentum = 1.", ("momentum", fun () -> step ~momentum:1.0 ()))
    :: ("momentum = nan", ("momentum", fun () -> step ~momentum:Float.nan ()))
    :: List.map trust [ 0.0; -0.0; -0.001; Float.infinity; Float.nan ])
    (fun (_, r) -> refuses r)

(* Adam's family *)

let adam_draw =
  let open Gen in
  let* s = shape in
  let+ x = values s
  and+ g = values s
  and+ mu = values s
  and+ nu = values ~positive:true s
  and+ b1 = opt coefficient
  and+ b2 = opt coefficient
  and+ eps = opt eps
  and+ lr = factor
  and+ c = completed in
  (s, x, g, (mu, nu), (b1, b2, eps), lr, c)

let adam_state s c (mu, nu) =
  { Vega.mu = make s mu; nu = make s nu; step = counter c }

(* [moments (b1, b2) t mu nu g] is Adam's [mu' = b1 * mu + (1 - b1) * g], [nu' =
   b2 * nu + (1 - b2) * g^2] and their bias corrections [mu' / (1 - b1^t)] and
   [nu' / (1 - b2^t)]. *)
let moments (b1, b2) t mu nu g =
  let mu' = average b1 mu g
  and nu' = average b2 nu (Array.map (fun g -> g *. g) g) in
  let hat b = Array.map (fun m -> m /. bias b t) in
  (mu', nu', hat b1 mu', hat b2 nu')

(* Adam's direction, [mu_hat / (sqrt nu_hat + eps)]. *)
let adam eps = Array.map2 (fun m n -> m /. (sqrt n +. eps))

(* The coefficients with their defaults. *)
let adam_defaults (b1, b2, eps) =
  ( (Option.value b1 ~default:0.9, Option.value b2 ~default:0.999),
    Option.value eps ~default:1e-8 )

let cover_adam (b1, b2, eps) =
  cover_defaults [ ("b1", b1); ("b2", b2); ("eps", eps) ]

(* [adam_check t (st', x') ~update] checks the moments of [st'] and the
   parameters [x'] against [update x mu_hat nu_hat], leaf by leaf. *)
let check_adam ~bs ~t ~x ~g ~mu ~nu (st' : (Wv.t, _) Vega.adam_state) x' update
    =
  List.iter
    (fun (name, leaf, stepped) ->
      let mu', nu', mu_hat, nu_hat =
        moments bs t (leaf mu) (leaf nu) (leaf g)
      in
      agree ("mu." ^ name) mu' (stepped st'.mu);
      agree ("nu." ^ name) nu' (stepped st'.nu);
      agree name (update (leaf x) mu_hat nu_hat) (stepped x'))
    leaves;
  advanced (t - 1) st'.step

let adam_formula =
  prop "adam_step is its formula" adam_draw
    (fun (s, x, g, ((mu, nu) as m), ((b1, b2, eps) as k), lr, c) ->
      cover_draw s c;
      cover_adam k;
      let x', st' =
        Vega.adam_step p ~lr:(rate lr) ?b1 ?b2 ?eps (adam_state s c m)
          ~params:(make s x) ~grads:(make s g)
      in
      let bs, eps = adam_defaults k and lr = rate_at lr c in
      (* p' = p - lr * mu_hat / (sqrt nu_hat + eps) *)
      check_adam ~bs ~t:(c + 1) ~x ~g ~mu ~nu st' x' (fun x mu_hat nu_hat ->
          descend lr x (adam eps mu_hat nu_hat)))

(* Each coefficient outside its range, by name. *)
let adam_refusals name step =
  let x = Nx.ones Nx.float64 [| 2 |] in
  cases
    (name ^ " refuses a coefficient out of range")
    ~id:fst
    [
      ("b1 = 1.", ("b1", fun () -> step ~b1:1.0 ~b2:0.999 ~eps:1e-8 x));
      ("b1 = -0.1", ("b1", fun () -> step ~b1:(-0.1) ~b2:0.999 ~eps:1e-8 x));
      ("b2 = 1.", ("b2", fun () -> step ~b1:0.9 ~b2:1.0 ~eps:1e-8 x));
      ("b2 = -0.1", ("b2", fun () -> step ~b1:0.9 ~b2:(-0.1) ~eps:1e-8 x));
      ("eps = 0.", ("eps", fun () -> step ~b1:0.9 ~b2:0.999 ~eps:0.0 x));
      ("b1 = nan", ("b1", fun () -> step ~b1:Float.nan ~b2:0.999 ~eps:1e-8 x));
      ("eps = nan", ("eps", fun () -> step ~b1:0.9 ~b2:0.999 ~eps:Float.nan x));
    ]
    (fun (_, r) -> refuses r)

let adam_refuses =
  adam_refusals "adam_step" (fun ~b1 ~b2 ~eps x ->
      let p = Nx.Ptree.tensor in
      ignore
        (Vega.adam_step p ~lr:(rate 0.1) ~b1 ~b2 ~eps (Vega.adam_init p x)
           ~params:x ~grads:x))

let adamw_formula =
  prop "adamw_step is its formula"
    (Gen.pair adam_draw (opt factor))
    (fun ((s, x, g, ((mu, nu) as m), ((b1, b2, eps) as k), lr, c), wd) ->
      cover_draw s c;
      cover_adam k;
      cover_defaults [ ("weight_decay", wd) ];
      let x', st' =
        Vega.adamw_step p ~lr:(rate lr) ?b1 ?b2 ?eps
          ?weight_decay:(Option.map rate wd) (adam_state s c m)
          ~params:(make s x) ~grads:(make s g)
      in
      let bs, eps = adam_defaults k and lr = rate_at lr c in
      let wd = match wd with Some wd -> rate_at wd c | None -> f32 0.01 in
      (* p' = p - lr * (d + wd * p) *)
      check_adam ~bs ~t:(c + 1) ~x ~g ~mu ~nu st' x' (fun x mu_hat nu_hat ->
          let d =
            Array.map2 (fun d x -> d +. (wd *. x)) (adam eps mu_hat nu_hat) x
          in
          descend lr x d))

let adamw_refuses =
  adam_refusals "adamw_step" (fun ~b1 ~b2 ~eps x ->
      let p = Nx.Ptree.tensor in
      ignore
        (Vega.adamw_step p ~lr:(rate 0.1) ~b1 ~b2 ~eps (Vega.adamw_init p x)
           ~params:x ~grads:x))

(* RAdam *)

let rho_inf b2 = (2.0 /. (1.0 -. b2)) -. 1.0

(* [rho b2 t] is RAdam's [rho] at the step [t]. *)
let rho b2 t =
  let tf = Float.of_int t in
  rho_inf b2 -. (2.0 *. tf *. (b2 ** tf) /. bias b2 t)

(* [rectifies b2 t] says whether [t] is at or past [t*], the first step at which
   [rho] reaches 5. *)
let rectifies b2 t =
  let rec reached k = k <= t && (rho b2 k >= 5.0 || reached (k + 1)) in
  rho_inf b2 > 5.0 && reached 1

let radam_formula =
  prop "radam_step is its formula" adam_draw
    (fun (s, x, g, ((mu, nu) as m), ((b1, b2, eps) as k), lr, c) ->
      let ((_, b2') as bs), eps' = adam_defaults k and t = c + 1 in
      cover_draw s c;
      cover_adam k;
      cover "a momentum step" (not (rectifies b2' t));
      cover "a rectified step" (rectifies b2' t);
      let x', st' =
        Vega.radam_step p ~lr:(rate lr) ?b1 ?b2 ?eps (adam_state s c m)
          ~params:(make s x) ~grads:(make s g)
      in
      let lr = rate_at lr c in
      (* r = sqrt ((rho - 4) (rho - 2) rho_inf / ((rho_inf - 4) (rho_inf - 2)
         rho)); d = r * mu_hat / (sqrt nu_hat + eps) from t*, mu_hat before *)
      let r =
        let rho = rho b2' t and inf = rho_inf b2' in
        sqrt
          ((rho -. 4.0) *. (rho -. 2.0) *. inf
          /. ((inf -. 4.0) *. (inf -. 2.0) *. rho))
      in
      check_adam ~bs ~t ~x ~g ~mu ~nu st' x' (fun x mu_hat nu_hat ->
          if rectifies b2' t then
            descend lr x (Array.map (fun d -> r *. d) (adam eps' mu_hat nu_hat))
          else descend lr x mu_hat))

(* [switch b2] is [t*], from [rho]'s float64 closed form. *)
let switch b2 =
  let rec first t = if rho b2 t >= 5.0 then t else first (t + 1) in
  first 1

(* With a constant unit gradient, Adam's bias-corrected average is one, so a
   momentum step moves a float32 parameter by exactly the rate and a rectified
   one by [r], below one. The first step that does not move by the rate is the
   switch, which is [t*] at every [b2], including those at which a float32 [rho]
   crossed 5 a step early. *)
let radam_switch =
  cases "radam_step switches at t* for a float32 leaf"
    ~id:(Printf.sprintf "b2 = %g")
    [
      0.8; 0.9; 0.99; 0.998; 0.99857854705; 0.9986040462; 0.999; 0.9999; 0.99999;
    ] (fun b2 ->
      let p = Nx.Ptree.tensor and lr = Vega.rate 1.0 in
      let grads = Nx.create Nx.float32 [| 1 |] [| 1.0 |] in
      let rec moved t params st =
        let params', st = Vega.radam_step p ~lr ~b2 st ~params ~grads in
        let by = Nx.item [ 0 ] params -. Nx.item [ 0 ] params' in
        if Float.abs (by -. 1.0) > 1e-3 || t = 12 then t
        else moved (t + 1) params' st
      in
      let params = Nx.create Nx.float32 [| 1 |] [| 0.0 |] in
      equal int (switch b2) (moved 1 params (Vega.radam_init p params)))

(* RAdam at [b2 = 0.9999] at float32 against float64, over the rectified steps 6
   to 12: [rho] is the small difference of two terms near 19999 there. Computing
   [1 - b2^t] as [1 - pow b2 t] at float32 put this displacement off by 3.5%;
   float32's own rounding of [rho], about [2e-3] absolute, bounds it near
   [1e-4]. *)
let test_radam_rectifies_at_float32 () =
  let displacement (type b) (dt : (float, b) Nx.dtype) =
    let target = Nx.create dt [| 3 |] [| 1.0; -2.0; 0.5 |] in
    let p = Nx.Ptree.tensor in
    let rec go k params st acc =
      if k = 12 then List.rev acc
      else
        let grads = Nx.mul_s (Nx.sub params target) 2.0 in
        let params, st =
          Vega.radam_step p ~lr:(Vega.rate 0.1) ~b2:0.9999 st ~params ~grads
        in
        go (k + 1) params st (Nx.to_array (Nx.cast Nx.float64 params) :: acc)
    in
    let params = Nx.zeros dt [| 3 |] in
    let path = go 0 params (Vega.radam_init p params) [] in
    Array.map2 ( -. ) (List.nth path 11) (List.nth path 5)
  in
  equal
    (array (float_rel ~rel:1e-4 ~abs:0.0))
    (displacement Nx.float64) (displacement Nx.float32)

let radam_refuses =
  adam_refusals "radam_step" (fun ~b1 ~b2 ~eps x ->
      let p = Nx.Ptree.tensor in
      ignore
        (Vega.radam_step p ~lr:(rate 0.1) ~b1 ~b2 ~eps (Vega.radam_init p x)
           ~params:x ~grads:x))

(* LAMB *)

let lamb_formula =
  prop "lamb_step is its formula"
    (Gen.pair adam_draw (opt factor))
    (fun ((s, x, g, ((mu, nu) as m), ((b1, b2, eps) as k), lr, c), wd) ->
      cover_draw s c;
      cover_adam k;
      cover_defaults [ ("weight_decay", wd) ];
      let x', st' =
        Vega.lamb_step p ~lr:(rate lr) ?b1 ?b2 ?eps
          ?weight_decay:(Option.map rate wd) (adam_state s c m)
          ~params:(make s x) ~grads:(make s g)
      in
      let bs, eps = adam_defaults k and lr = rate_at lr c in
      let wd = match wd with Some wd -> rate_at wd c | None -> f32 0.01 in
      (* u = d + wd * p; r = |p| / |u|, or 1; p' = p - lr * r * u *)
      check_adam ~bs ~t:(c + 1) ~x ~g ~mu ~nu st' x' (fun x mu_hat nu_hat ->
          let u =
            Array.map2 (fun d x -> d +. (wd *. x)) (adam eps mu_hat nu_hat) x
          in
          let r = trust 1.0 x u in
          descend lr x (Array.map (fun u -> r *. u) u)))

let lamb_refuses =
  adam_refusals "lamb_step" (fun ~b1 ~b2 ~eps x ->
      let p = Nx.Ptree.tensor in
      ignore
        (Vega.lamb_step p ~lr:(rate 0.1) ~b1 ~b2 ~eps (Vega.lamb_init p x)
           ~params:x ~grads:x))

(* RMSprop *)

let rmsprop_formula =
  let gen =
    let open Gen in
    let* s = shape in
    let+ x = values s
    and+ g = values s
    and+ nu = values ~positive:true s
    and+ v = values s
    and+ decay = opt coefficient
    and+ eps = opt eps
    and+ momentum = opt coefficient
    and+ lr = factor
    and+ c = completed in
    (s, x, g, (nu, v), (decay, eps, momentum), lr, c)
  in
  prop "rmsprop_step is its formula" gen
    (fun (s, x, g, (nu, v), (decay, eps, momentum), lr, c) ->
      cover_draw s c;
      cover_defaults [ ("decay", decay); ("eps", eps); ("momentum", momentum) ];
      let st = { Vega.nu = make s nu; velocity = make s v; step = counter c } in
      let x', st' =
        Vega.rmsprop_step p ~lr:(rate lr) ?decay ?eps ?momentum st
          ~params:(make s x) ~grads:(make s g)
      in
      let decay = Option.value decay ~default:0.9 in
      let eps = Option.value eps ~default:1e-8 in
      let momentum = Option.value momentum ~default:0.0 in
      let lr = rate_at lr c in
      List.iter
        (fun (name, leaf, stepped) ->
          let g = leaf g in
          (* nu' = decay * nu + (1 - decay) * g^2; v' = momentum * v + g / (sqrt
             nu' + eps); p' = p - lr * v' *)
          let nu' = average decay (leaf nu) (Array.map (fun g -> g *. g) g) in
          let scaled v g nu = (momentum *. v) +. (g /. (sqrt nu +. eps)) in
          let v' = map3 scaled (leaf v) g nu' in
          agree ("nu." ^ name) nu' (stepped st'.nu);
          agree ("velocity." ^ name) v' (stepped st'.velocity);
          agree name (descend lr (leaf x) v') (stepped x'))
        leaves;
      advanced c st'.step)

let rmsprop_refuses =
  let x = Nx.ones Nx.float64 [| 2 |] and p = Nx.Ptree.tensor in
  let step ?decay ?eps ?momentum () =
    Vega.rmsprop_step p ~lr:(rate 0.1) ?decay ?eps ?momentum
      (Vega.rmsprop_init p x) ~params:x ~grads:x
  in
  cases "rmsprop_step refuses a coefficient out of range" ~id:fst
    [
      ("decay = 1.", ("decay", fun () -> step ~decay:1.0 ()));
      ("decay = 1.5", ("decay", fun () -> step ~decay:1.5 ()));
      ("decay = nan", ("decay", fun () -> step ~decay:Float.nan ()));
      ("eps = 0.", ("eps", fun () -> step ~eps:0.0 ()));
      ("momentum = 1.", ("momentum", fun () -> step ~momentum:1.0 ()));
    ]
    (fun (_, r) -> refuses r)

(* Adagrad *)

let adagrad_formula =
  let gen =
    let open Gen in
    let* s = shape in
    let+ x = values s
    and+ g = values s
    and+ sum = values ~positive:true s
    and+ eps = opt eps
    and+ lr = factor
    and+ c = completed in
    (s, x, g, sum, eps, lr, c)
  in
  prop "adagrad_step is its formula" gen (fun (s, x, g, sum, eps, lr, c) ->
      cover_draw s c;
      cover_defaults [ ("eps", eps) ];
      let st = { Vega.sum_of_squares = make s sum; step = counter c } in
      let x', st' =
        Vega.adagrad_step p ~lr:(rate lr) ?eps st ~params:(make s x)
          ~grads:(make s g)
      in
      let eps = Option.value eps ~default:1e-8 and lr = rate_at lr c in
      List.iter
        (fun (name, leaf, stepped) ->
          let g = leaf g in
          (* s' = s + g^2; p' = p - lr * g / (sqrt s' + eps) *)
          let sum' = Array.map2 (fun s g -> s +. (g *. g)) (leaf sum) g in
          let d = Array.map2 (fun g s -> g /. (sqrt s +. eps)) g sum' in
          agree ("sum_of_squares." ^ name) sum' (stepped st'.sum_of_squares);
          agree name (descend lr (leaf x) d) (stepped x'))
        leaves;
      advanced c st'.step)

let adagrad_refuses =
  let x = Nx.ones Nx.float64 [| 2 |] and p = Nx.Ptree.tensor in
  let step eps () =
    Vega.adagrad_step p ~lr:(rate 0.1) ~eps (Vega.adagrad_init p x) ~params:x
      ~grads:x
  in
  cases "adagrad_step refuses an eps that is not positive" ~id:fst
    [
      ("eps = 0.", step 0.0);
      ("eps = -1.", step (-1.0));
      ("eps = nan", step Float.nan);
    ]
    (fun (_, f) -> refuses ("eps", f))

(* Adan *)

let adan_formula =
  let gen =
    let open Gen in
    let* s = shape in
    let+ (x, g), (mu, delta, nu, prev) =
      pair
        (pair (values s) (values s))
        (quad (values s) (values s) (values ~positive:true s) (values s))
    and+ k = triple (opt coefficient) (opt coefficient) (opt coefficient)
    and+ eps = opt eps
    and+ wd = opt factor
    and+ lr = factor
    and+ c = completed in
    (s, (x, g), (mu, delta, nu, prev), (k, eps, wd), lr, c)
  in
  prop "adan_step is its formula" gen
    (fun (s, (x, g), (mu, delta, nu, prev), ((b1, b2, b3), eps, wd), lr, c) ->
      cover_draw s c;
      cover_defaults
        [
          ("b1", b1); ("b2", b2); ("b3", b3); ("eps", eps); ("weight_decay", wd);
        ];
      let st =
        {
          Vega.mu = make s mu;
          delta = make s delta;
          nu = make s nu;
          prev_grads = make s prev;
          step = counter c;
        }
      in
      let x', st' =
        Vega.adan_step p ~lr:(rate lr) ?b1 ?b2 ?b3 ?eps
          ?weight_decay:(Option.map rate wd) st ~params:(make s x)
          ~grads:(make s g)
      in
      let b1 = Option.value b1 ~default:0.98
      and b2 = Option.value b2 ~default:0.92 in
      let b3 = Option.value b3 ~default:0.99
      and eps = Option.value eps ~default:1e-8 in
      let wd = match wd with Some wd -> rate_at wd c | None -> f32 0.02 in
      let lr = rate_at lr c in
      List.iter
        (fun (name, leaf, stepped) ->
          let g = leaf g and x = leaf x in
          (* dg = g - prev_grads; mu' = b1 * mu + (1 - b1) * g; delta' = b2 *
             delta + (1 - b2) * dg; nu' = b3 * nu + (1 - b3) * (g + b2 * dg)^2;
             d = (mu' + b2 * delta') / (sqrt nu' + eps) + wd * p *)
          let dg = Array.map2 ( -. ) g (leaf prev) in
          let mu' = average b1 (leaf mu) g
          and delta' = average b2 (leaf delta) dg in
          let ahead = Array.map2 (fun g dg -> g +. (b2 *. dg)) g dg in
          let nu' = average b3 (leaf nu) (Array.map (fun a -> a *. a) ahead) in
          let d m dl n x =
            ((m +. (b2 *. dl)) /. (sqrt n +. eps)) +. (wd *. x)
          in
          let d = Array.mapi (fun i x -> d mu'.(i) delta'.(i) nu'.(i) x) x in
          agree ("mu." ^ name) mu' (stepped st'.mu);
          agree ("delta." ^ name) delta' (stepped st'.delta);
          agree ("nu." ^ name) nu' (stepped st'.nu);
          agree ("prev_grads." ^ name) g (stepped st'.prev_grads);
          agree name (descend lr x d) (stepped x'))
        leaves;
      advanced c st'.step)

let adan_refuses =
  let x = Nx.ones Nx.float64 [| 2 |] and p = Nx.Ptree.tensor in
  let step ?b1 ?b2 ?b3 ?eps () =
    Vega.adan_step p ~lr:(rate 0.1) ?b1 ?b2 ?b3 ?eps (Vega.adan_init p x)
      ~params:x ~grads:x
  in
  cases "adan_step refuses a coefficient out of range" ~id:fst
    [
      ("b1 = 1.", ("b1", fun () -> step ~b1:1.0 ()));
      ("b2 = -0.1", ("b2", fun () -> step ~b2:(-0.1) ()));
      ("b3 = 1.", ("b3", fun () -> step ~b3:1.0 ()));
      ("b3 = nan", ("b3", fun () -> step ~b3:Float.nan ()));
      ("eps = 0.", ("eps", fun () -> step ~eps:0.0 ()));
    ]
    (fun (_, r) -> refuses r)

(* Lion *)

let lion_formula =
  let gen =
    let open Gen in
    let* s = shape in
    let+ x = values s
    and+ g = values s
    and+ mu = values s
    and+ b1 = opt coefficient
    and+ b2 = opt coefficient
    and+ lr = factor
    and+ c = completed in
    (s, x, g, mu, (b1, b2), lr, c)
  in
  prop "lion_step is its formula" gen (fun (s, x, g, mu, (b1, b2), lr, c) ->
      cover_draw s c;
      cover_defaults [ ("b1", b1); ("b2", b2) ];
      let st = { Vega.mu = make s mu; step = counter c } in
      let x', st' =
        Vega.lion_step p ~lr:(rate lr) ?b1 ?b2 st ~params:(make s x)
          ~grads:(make s g)
      in
      let b1 = Option.value b1 ~default:0.9
      and b2 = Option.value b2 ~default:0.99 in
      let lr = rate_at lr c in
      let sign x = if x > 0.0 then 1.0 else if x < 0.0 then -1.0 else 0.0 in
      List.iter
        (fun (name, leaf, stepped) ->
          let g = leaf g and mu = leaf mu in
          (* p' = p - lr * sign (b1 * mu + (1 - b1) * g); mu' = b2 * mu + (1 -
             b2) * g *)
          let d = Array.map sign (average b1 mu g) in
          agree ("mu." ^ name) (average b2 mu g) (stepped st'.mu);
          agree name (descend lr (leaf x) d) (stepped x'))
        leaves;
      advanced c st'.step)

let lion_refuses =
  let x = Nx.ones Nx.float64 [| 2 |] and p = Nx.Ptree.tensor in
  let step ?b1 ?b2 () =
    Vega.lion_step p ~lr:(rate 0.1) ?b1 ?b2 (Vega.lion_init p x) ~params:x
      ~grads:x
  in
  cases "lion_step refuses a coefficient out of range" ~id:fst
    [
      ("b1 = 1.", ("b1", fun () -> step ~b1:1.0 ()));
      ("b2 = -0.1", ("b2", fun () -> step ~b2:(-0.1) ()));
      ("b2 = nan", ("b2", fun () -> step ~b2:Float.nan ()));
    ]
    (fun (_, r) -> refuses r)

(* Adafactor *)

let mean a = Array.fold_left ( +. ) 0.0 a /. Float.of_int (Array.length a)
let rms a = sqrt (mean (Array.map (fun x -> x *. x) a))

(* The state of a matrix of [m] rows and [n] columns and a vector of [n]: the
   matrix factored or not. A part a leaf does not use is a scalar zero. *)
let adafactor_state (m, n) ~factored (row, col, full, vec) c =
  let leaf shape a = Nx.create Nx.float64 shape a in
  let zero = Nx.scalar Nx.float64 0.0 in
  let w f u = if factored then f else u in
  {
    Vega.nu_row = { Wv.w = w (leaf [| m; 1 |] row) zero; v = zero };
    nu_col = { w = w (leaf [| 1; n |] col) zero; v = zero };
    nu = { w = w zero (leaf [| m; n |] full); v = leaf [| n |] vec };
    step = counter c;
  }

let adafactor_formula =
  let gen =
    let open Gen in
    let* ((m, n) as s) = shape in
    let size k = array ~size:(constant k) (map Float.abs element) in
    let+ (x, g), stats =
      pair
        (pair (values s) (values s))
        (quad (size m) (size n) (size (m * n)) (size n))
    and+ factored = bool
    and+ decay_rate = opt (float_range 0.1 1.0)
    and+ eps = opt (of_list [ 1e-30; 1e-3 ])
    and+ threshold = opt (of_list [ 1.0; 0.01; infinity ])
    and+ lr = factor
    and+ c = completed in
    (s, (x, g), stats, factored, (decay_rate, eps, threshold), lr, c)
  in
  prop "adafactor_step is its formula" gen
    (fun
      ( ((m, n) as s),
        (x, g),
        ((row, col, full, vec) as stats),
        factored,
        (decay_rate, eps, threshold),
        lr,
        c )
    ->
      cover_draw s c;
      cover "a factored matrix" factored;
      cover "a clipped update" (threshold = Some 0.01);
      cover_defaults
        [
          ("decay_rate", decay_rate);
          ("eps", eps);
          ("clipping_threshold", threshold);
        ];
      let x', st' =
        Vega.adafactor_step p ~lr:(rate lr) ?decay_rate ?eps
          ?clipping_threshold:threshold
          (adafactor_state s ~factored stats c)
          ~params:(make s x) ~grads:(make s g)
      in
      let decay_rate = Option.value decay_rate ~default:0.8 in
      let eps = Option.value eps ~default:1e-30 in
      let threshold = Option.value threshold ~default:1.0 in
      let lr = rate_at lr c and t = Float.of_int (c + 1) in
      (* An average keeps [b = 1 - t^-decay_rate] of itself and takes [1 - b] of
         the new value. u = g / (sqrt nu + eps); p' = p - lr * u * min (1,
         threshold / rms u). *)
      let b = -.Float.expm1 (-.decay_rate *. log t) in
      let fresh = t ** -.decay_rate in
      let averaged m v = (b *. m) +. (fresh *. v) in
      let update x g nu =
        let u = Array.map2 (fun g nu -> g /. (sqrt nu +. eps)) g nu in
        let clip = Float.min 1.0 (threshold /. rms u) in
        descend lr x (Array.map (fun u -> u *. clip) u)
      in
      let read x = Nx.to_array x in
      (* The matrix: factored, nu_row' = b * nu_row + (1 - b) * mean_r g^2,
         nu_col' likewise over the rows, and nu = nu_row' * nu_col' / (mean_c
         nu_row' + eps); a matrix of no element keeps its statistics. *)
      let gw = fst g in
      let g2 i j = gw.((i * n) + j) *. gw.((i * n) + j) in
      (if factored then (
         let empty = m * n = 0 in
         let row' =
           if empty then row
           else
             Array.init m (fun i ->
                 averaged row.(i) (mean (Array.init n (g2 i))))
         in
         let col' =
           if empty then col
           else
             Array.init n (fun j ->
                 averaged col.(j) (mean (Array.init m (fun i -> g2 i j))))
         in
         let nu =
           Array.init (m * n) (fun e ->
               row'.(e / n) *. col'.(e mod n) /. (mean row' +. eps))
         in
         agree "nu_row.w" row' (read st'.nu_row.w);
         agree "nu_col.w" col' (read st'.nu_col.w);
         agree "nu.w" [| 0.0 |] (read st'.nu.w);
         agree "w" (update (fst x) gw nu) (read x'.w))
       else
         let nu' = Array.map2 (fun nu g -> averaged nu (g *. g)) full gw in
         agree "nu_row.w" [| 0.0 |] (read st'.nu_row.w);
         agree "nu.w" nu' (read st'.nu.w);
         agree "w" (update (fst x) gw nu') (read x'.w));
      (* The vector. *)
      let nu' = Array.map2 (fun nu g -> averaged nu (g *. g)) vec (snd g) in
      agree "nu.v" nu' (read st'.nu.v);
      agree "v" (update (snd x) (snd g) nu') (read x'.v);
      advanced c st'.step)

(* A leaf of two axes or more is factored: its statistics are of shapes [[...;
   m; 1]] and [[...; 1; n]]; a leaf of fewer, or every leaf without [factored],
   keeps [nu] of its own shape. A part a leaf does not use is a scalar zero. *)
let test_adafactor_shapes () =
  let p = Nx.Ptree.(triple tensor tensor tensor) in
  let x =
    ( Nx.ones Nx.float64 [| 2; 3 |],
      Nx.ones Nx.float64 [| 4; 2; 3 |],
      Nx.ones Nx.float64 [| 3 |] )
  in
  let shapes (a, b, c) = [ Nx.shape a; Nx.shape b; Nx.shape c ] in
  let parts (st : (_, _) Vega.adafactor_state) =
    [ shapes st.nu_row; shapes st.nu_col; shapes st.nu ]
  in
  let shape = list (list (array int)) in
  equal ~msg:"factored" shape
    [
      [ [| 2; 1 |]; [| 4; 2; 1 |]; [||] ];
      [ [| 1; 3 |]; [| 4; 1; 3 |]; [||] ];
      [ [||]; [||]; [| 3 |] ];
    ]
    (parts (Vega.adafactor_init p x));
  equal ~msg:"not factored" shape
    [
      [ [||]; [||]; [||] ];
      [ [||]; [||]; [||] ];
      [ [| 2; 3 |]; [| 4; 2; 3 |]; [| 3 |] ];
    ]
    (parts (Vega.adafactor_init p ~factored:false x))

(* A leaf of no element steps to itself and keeps its state, which init makes
   zeros: an empty factored matrix, whichever axis is empty, a stack of them,
   and a complex and a bfloat16 one. *)
(* The floats of a state part, a complex one's parts. *)
let part_floats (type a b) (x : (a, b, Nx.host) Nx.t) =
  match Nx.Dtype.kind (Nx.dtype x) with
  | Complex ->
      Array.append
        (Nx.to_array (Nx.real Nx.float64 x))
        (Nx.to_array (Nx.imag Nx.float64 x))
  | Float | Signed | Unsigned | Boolean -> Nx.to_array (Nx.cast Nx.float64 x)

let empty_leaf (type a b) (x : (a, b, Nx.host) Nx.t) () =
  let p = Nx.Ptree.tensor in
  let zeros msg (st : (_, _) Vega.adafactor_state) =
    List.iter
      (fun (name, part) ->
        let a = part_floats part in
        equal
          ~msg:(msg ^ ": " ^ name)
          (array float_exact)
          (Array.map (fun _ -> 0.0) a)
          a)
      [ ("nu_row", st.nu_row); ("nu_col", st.nu_col); ("nu", st.nu) ]
  in
  let st = Vega.adafactor_init p x in
  zeros "init" st;
  let x', st' = Vega.adafactor_step p ~lr:(rate 0.1) st ~params:x ~grads:x in
  equal ~msg:"the leaf" (array int) (Nx.shape x) (Nx.shape x');
  zeros "step" st'

let empty_factored =
  cases "a leaf of no element keeps a zero state" ~id:fst
    [
      ("float64 [2; 0]", empty_leaf (Nx.zeros Nx.float64 [| 2; 0 |]));
      ("float64 [0; 3]", empty_leaf (Nx.zeros Nx.float64 [| 0; 3 |]));
      ("float64 [0; 2; 3]", empty_leaf (Nx.zeros Nx.float64 [| 0; 2; 3 |]));
      ("complex64 [2; 0]", empty_leaf (Nx.zeros Nx.complex64 [| 2; 0 |]));
      ("bfloat16 [0; 3]", empty_leaf (Nx.zeros Nx.bfloat16 [| 0; 3 |]));
    ]
    (fun (_, check) -> check ())

(* A leaf that is neither float nor complex holds zeros of its own shape in
   every part, and the step leaves them and it unchanged. A vector leaf held a
   scalar zero in [nu_row] and [nu_col]. *)
let test_adafactor_integer_leaf () =
  let p = Nx.Ptree.(pair tensor tensor) in
  let count = Nx.create Nx.int32 [| 2 |] [| 3l; 7l |] in
  let x = (Nx.ones Nx.float64 [| 2; 3 |], count) in
  let g = (Nx.full Nx.float64 [| 2; 3 |] 0.5, Nx.zeros Nx.int32 [| 2 |]) in
  let parts (st : (_, _) Vega.adafactor_state) =
    List.map (fun (_, c) -> Nx.to_array c) [ st.nu_row; st.nu_col; st.nu ]
  in
  let zeros = [ [| 0l; 0l |]; [| 0l; 0l |]; [| 0l; 0l |] ] in
  let st = Vega.adafactor_init p x in
  equal ~msg:"init" (list (array int32)) zeros (parts st);
  let (_, count'), st' =
    Vega.adafactor_step p ~lr:(rate 0.01) st ~params:x ~grads:g
  in
  equal ~msg:"the leaf" (array int32) [| 3l; 7l |] (Nx.to_array count');
  equal ~msg:"the step" (list (array int32)) zeros (parts st')

let adafactor_refuses =
  let x = Nx.ones Nx.float64 [| 2 |] and p = Nx.Ptree.tensor in
  let step ?decay_rate ?eps ?clipping_threshold () =
    Vega.adafactor_step p ~lr:(rate 0.1) ?decay_rate ?eps ?clipping_threshold
      (Vega.adafactor_init p x) ~params:x ~grads:x
  in
  cases "adafactor_step refuses a coefficient that is not positive" ~id:fst
    [
      ("decay_rate = 0.", ("decay_rate", fun () -> step ~decay_rate:0.0 ()));
      ( "decay_rate = nan",
        ("decay_rate", fun () -> step ~decay_rate:Float.nan ()) );
      ("eps = 0.", ("eps", fun () -> step ~eps:0.0 ()));
      ( "clipping_threshold = -1.",
        ("clipping_threshold", fun () -> step ~clipping_threshold:(-1.0) ()) );
    ]
    (fun (_, r) -> refuses r)

(* L-BFGS *)

(* [init] holds the parameters, a zero gradient and empty pairs, [history] of
   them stacked on a new first axis of every leaf, and the counter 0. *)
let test_lbfgs_init () =
  let x = make (2, 3) (Array.init 6 Float.of_int, [| 1.0; 2.0; 3.0 |]) in
  let st = Vega.lbfgs_init p ~history:4 x in
  let floats (x : Wv.t) = (Nx.to_array x.w, Nx.to_array x.v) in
  let leaves = pair (array float_exact) (array float_exact) in
  equal ~msg:"params" leaves (floats x) (floats st.params);
  equal ~msg:"grads" leaves
    (Array.make 6 0.0, Array.make 3 0.0)
    (floats st.grads);
  equal ~msg:"s" leaves (Array.make 24 0.0, Array.make 12 0.0) (floats st.s);
  equal ~msg:"y's shapes"
    (pair (array int) (array int))
    ([| 4; 2; 3 |], [| 4; 3 |])
    (Nx.shape st.y.w, Nx.shape st.y.v);
  equal ~msg:"step" int32 0l (Nx.item [] st.step);
  equal ~msg:"the default history" (array int) [| 10; 3 |]
    (Nx.shape (Vega.lbfgs_init p x).s.v);
  raises (Invalid_argument "Vega.lbfgs_init: expected history >= 1, got 0")
    (fun () -> Vega.lbfgs_init p ~history:0 x)

let dot a b =
  let s = ref 0.0 in
  Array.iteri (fun i a -> s := !s +. (a *. b.(i))) a;
  !s

let axpy a x y = Array.map2 (fun x y -> (a *. x) +. y) x y

(* [two_loop g s y] is the recursion of Nocedal (1980) for the gradient [g] over
   the pairs [s.(i)], [y.(i)], newest first, a pair whose curvature [y . s] is
   not positive left out, the initial inverse Hessian scaled by [(s . y) / (y .
   y)] of the newest pair that enters, or by 1. The step moves along its
   negation. *)
let two_loop g s y =
  let h = Array.length s in
  let rho =
    Array.init h (fun i ->
        let c = dot y.(i) s.(i) in
        if c > 0.0 then 1.0 /. c else 0.0)
  in
  let alpha = Array.make h 0.0 and q = ref g in
  for i = 0 to h - 1 do
    alpha.(i) <- rho.(i) *. dot s.(i) !q;
    q := axpy (-.alpha.(i)) y.(i) !q
  done;
  let gamma = ref 1.0 in
  for i = h - 1 downto 0 do
    if rho.(i) > 0.0 then gamma := 1.0 /. (rho.(i) *. dot y.(i) y.(i))
  done;
  let r = ref (Array.map (fun q -> !gamma *. q) !q) in
  for i = h - 1 downto 0 do
    let beta = rho.(i) *. dot y.(i) !r in
    r := axpy (alpha.(i) -. beta) s.(i) !r
  done;
  !r

(* A drawn pair: [y = c s] elementwise, [c] in [[0.5; 2]], of positive curvature
   unless [s] is zero; [y = -c s], of negative curvature; or an empty slot. Its
   curvature is at least [|y| |s| / 4] in magnitude, so no rounding decides
   whether it enters. *)
let drawn_pair len =
  let open Gen in
  let+ kind = int_range 0 2
  and+ s = array ~size:(constant len) (float_range (-1.0) 1.0)
  and+ c = array ~size:(constant len) (float_range 0.5 2.0) in
  match kind with
  | 0 -> (s, Array.map2 ( *. ) c s)
  | 1 -> (s, Array.map2 (fun c s -> -.(c *. s)) c s)
  | _ -> (Array.make len 0.0, Array.make len 0.0)

let lbfgs_formula =
  let gen =
    let open Gen in
    let* ((m, n) as s) = shape in
    let* history = frequency [ (1, constant 1); (3, int_range 2 3) ] in
    let+ x = values s
    and+ prev = values s
    and+ pairs = list ~size:(constant history) (drawn_pair ((m * n) + n))
    and+ lr = factor
    and+ c = completed in
    (s, history, x, prev, Array.of_list pairs, lr, c)
  in
  prop ~count:200 "lbfgs_step is its formula" gen
    (fun (((m, n) as s), history, x, prev, pairs, lr, c) ->
      cover_draw s c;
      cover "a history of one pair" (history = 1);
      let curvature (s, y) = dot y s in
      cover "a pair of positive curvature"
        (Array.exists (fun p -> curvature p > 0.0) pairs);
      cover "a pair of negative curvature"
        (Array.exists (fun p -> curvature p < 0.0) pairs);
      cover "a rejected newest pair under an accepted one"
        (curvature pairs.(0) <= 0.0
        && Array.exists (fun p -> curvature p > 0.0) pairs);
      let len = (m * n) + n in
      let flat (w, v) = Array.append w v in
      let split a = (Array.sub a 0 (m * n), Array.sub a (m * n) n) in
      (* The gradient completes the newest pair, [y.(0) = g - prev]. *)
      let g = Array.map2 ( +. ) (flat prev) (snd pairs.(0)) in
      let ss = Array.map fst pairs and ys = Array.map snd pairs in
      ys.(0) <- Array.map2 ( -. ) g (flat prev);
      (* Pairs as a state holds them: each leaf's, newest first, stacked. *)
      let memory vs =
        let parts = Array.map split vs in
        let leaf part = Array.concat (Array.to_list (Array.map part parts)) in
        (leaf fst, leaf snd)
      in
      let stacked vs =
        let w, v = memory vs in
        {
          Wv.w = Nx.create Nx.float64 [| history; m; n |] w;
          v = Nx.create Nx.float64 [| history; n |] v;
        }
      in
      let waiting = Array.make len 0.0 in
      let st =
        {
          Vega.params = make s x;
          grads = make s prev;
          s = stacked ss;
          y = stacked (Array.mapi (fun i y -> if i = 0 then waiting else y) ys);
          step = counter c;
        }
      in
      let st' = Vega.lbfgs_step p ~lr:(rate lr) (make s (split g)) st in
      (* The move [-lr * r] lands on the new point and is the newest [s]; the
         newest [y] waits for the next gradient. Below them, a completed pair of
         positive curvature and the older pairs, the oldest dropped; a rejected
         one is replaced, and evicts nothing. *)
      let lr = rate_at lr c in
      let move = Array.map (fun r -> -.lr *. r) (two_loop g ss ys) in
      let first = if dot ys.(0) ss.(0) > 0.0 then 0 else 1 in
      let older a = Array.sub a first (history - 1) in
      let s' = Array.append [| move |] (older ss) in
      let y' = Array.append [| waiting |] (older ys) in
      let floats (x : Wv.t) =
        Array.append (Nx.to_array x.w) (Nx.to_array x.v)
      in
      agree "params" (Array.map2 ( +. ) (flat x) move) (floats st'.params);
      agree "grads" g (floats st'.grads);
      agree "s" (flat (memory s')) (floats st'.s);
      agree "y" (flat (memory y')) (floats st'.y);
      advanced c st'.step)

(* Where the newest pair has no positive curvature, the initial inverse Hessian
   is scaled by the newest pair that has: the older pair [s = e1], [y = 1000 e1]
   gives [1e-3], so the direction along [e2], which no accepted pair spans, is a
   thousandth of the gradient's. With a gradient [2 e1 + 3 e2], the two-loop
   recursion gives [r = 2e-3 e1 + 3e-3 e2]. *)
let test_lbfgs_scales_by_an_accepted_pair () =
  let s = Nx.create Nx.float64 [| 2; 2 |] [| 0.0; 1.0; 1.0; 0.0 |]
  and y = Nx.create Nx.float64 [| 2; 2 |] [| 0.0; 0.0; 1000.0; 0.0 |] in
  let st =
    {
      Vega.params = Nx.create Nx.float64 [| 2 |] [| 0.0; 0.0 |];
      grads = Nx.create Nx.float64 [| 2 |] [| 2.0; 4.0 |];
      s;
      y;
      step = Nx.scalar Nx.int32 2l;
    }
  in
  let st' =
    Vega.lbfgs_step Nx.Ptree.tensor ~lr:(Vega.rate 1.0)
      (Nx.create Nx.float64 [| 2 |] [| 2.0; 3.0 |])
      st
  in
  equal (array (float 1e-15)) [| -0.002; -0.003 |] (Nx.to_array st'.params)

(* A pair the gradient completes without positive curvature never enters the
   memory: the step from a state holding it as its pending pair is, bit for bit,
   the step from the same state whose pending slot is empty, a memory that never
   saw it. The gradient moves from [prev] by a drawn [y0], negated where its
   curvature with the pending move [s0] would be positive; zeros of either sign
   reach the recursion, which a pair it leaves out must not change. *)
let rejected_pair_unseen =
  let gen =
    let open Gen in
    let* ((m, n) as s) = shape in
    let* history = frequency [ (1, constant 1); (3, int_range 2 3) ] in
    let len = (m * n) + n in
    let signed =
      frequency
        [ (2, constant (-0.0)); (1, constant 0.0); (2, float_range (-1.0) 1.0) ]
    in
    let vector = array ~size:(constant len) signed in
    let+ x = values s
    and+ prev = vector
    and+ pending = vector
    and+ y0 = frequency [ (1, constant (Array.make len 0.0)); (4, vector) ]
    and+ older = list ~size:(constant (history - 1)) (drawn_pair len)
    and+ lr = factor
    and+ k = completed in
    let y0 = if dot y0 pending > 0.0 then Array.map Float.neg y0 else y0 in
    (s, history, x, prev, pending, y0, Array.of_list older, lr, k)
  in
  let examples =
    (* Gradients and a pending move of [-0.]: a pair multiplied by a [rho] of
       [+0.] gave the move [+0.] where the empty slot gives [-0.]. *)
    [
      ( (0, 2),
        1,
        ([||], [| 0.0; 0.0 |]),
        [| -0.0; -0.0 |],
        [| -0.0; -0.0 |],
        [| -0.0; -0.0 |],
        [||],
        0.0,
        0 );
    ]
  in
  prop ~count:300 ~examples "a rejected pair evicts nothing" gen
    (fun (((m, n) as s), history, x, prev, pending, y0, older, lr, k) ->
      let g = Array.map2 ( +. ) prev y0 in
      let completed = Array.map2 ( -. ) g prev in
      assume (dot completed pending <= 0.0);
      let curvature (s, y) = dot y s in
      let negative_zero v = v = 0.0 && Float.sign_bit v in
      cover_draw s k;
      cover "a history of one pair" (history = 1);
      cover "older pairs all of positive curvature"
        (history > 1 && Array.for_all (fun p -> curvature p > 0.0) older);
      cover "a zero curvature" (dot completed pending = 0.0);
      cover "a negative curvature" (dot completed pending < 0.0);
      cover "a gradient of -0." (Array.exists negative_zero g);
      let split a = (Array.sub a 0 (m * n), Array.sub a (m * n) n) in
      let memory vs =
        let parts = Array.map split vs in
        let leaf part = Array.concat (Array.to_list (Array.map part parts)) in
        {
          Wv.w = Nx.create Nx.float64 [| history; m; n |] (leaf fst);
          v = Nx.create Nx.float64 [| history; n |] (leaf snd);
        }
      in
      let waiting = Array.make (Array.length g) 0.0 in
      let state s0 =
        {
          Vega.params = make s x;
          grads = make s (split prev);
          s = memory (Array.append [| s0 |] (Array.map fst older));
          y = memory (Array.append [| waiting |] (Array.map snd older));
          step = counter k;
        }
      in
      let step s0 =
        Vega.lbfgs_step p ~lr:(rate lr) (make s (split g)) (state s0)
      in
      let seen = step pending and unseen = step waiting in
      let bits (st : (Wv.t, _) Vega.lbfgs_state) =
        List.map
          (fun (x : Wv.t) ->
            Array.map Int64.bits_of_float
              (Array.append (Nx.to_array x.w) (Nx.to_array x.v)))
          [ st.params; st.grads; st.s; st.y ]
      in
      equal (list (array int64)) (bits unseen) (bits seen))

(* On [|p - t|^2], whose Hessian is [2 I], a first step completes a pair whose
   curvature is the Hessian's along the move, so the second step's direction is
   Newton's and a unit rate lands on [t] from anywhere. *)
let test_lbfgs_newton () =
  let t = make (1, 2) ([| 1.5; -0.5 |], [| 2.0; 0.25 |]) in
  let grads (x : Wv.t) =
    { Wv.w = Nx.mul_s (Nx.sub x.w t.w) 2.0; v = Nx.mul_s (Nx.sub x.v t.v) 2.0 }
  in
  let step lr (st : (Wv.t, _) Vega.lbfgs_state) =
    Vega.lbfgs_step p ~lr:(Vega.rate lr) (grads st.params) st
  in
  let start = make (1, 2) ([| 5.0; -3.0 |], [| -4.0; 1.0 |]) in
  let st = step 1.0 (step 0.1 (Vega.lbfgs_init p start)) in
  let floats (x : Wv.t) = Array.append (Nx.to_array x.w) (Nx.to_array x.v) in
  equal (array (float 1e-12)) (floats t) (floats st.params)

(* The inner products are at float32 when a leaf is not float64. A float64 leaf
   and a float32 one hold [s = (1, 2^-15)], the gradient moves from [(1, 0)] to
   [(2, 3 * 2^-15)], so [y = (1, 3 * 2^-15)]. At float32 [y . s], [s . g] and [y
   . y] round to [1], [2] and [1], and the recursion gives the move [(-2,
   2^-15)] exactly; at float64 the float64 leaf moves by
   [-2.0000000055879354]. *)
let test_lbfgs_float32_products () =
  let p = Nx.Ptree.(pair tensor tensor) in
  let w x = Nx.create Nx.float64 [| 1 |] [| x |]
  and v x = Nx.create Nx.float32 [| 1 |] [| x |] in
  let memory w v = (Nx.reshape [| 1; 1 |] w, Nx.reshape [| 1; 1 |] v) in
  let st =
    {
      Vega.params = (w 0.0, v 0.0);
      grads = (w 1.0, v 0.0);
      s = memory (w 1.0) (v 0x1p-15);
      y = memory (w 0.0) (v 0.0);
      step = Nx.scalar Nx.int32 1l;
    }
  in
  let grads = (w 2.0, v (3.0 *. 0x1p-15)) in
  let w', _ = (Vega.lbfgs_step p ~lr:(Vega.rate 1.0) grads st).params in
  equal float_exact (-2.0) (Nx.item [ 0 ] w')

(* Groups follow vega.mli's sections. *)
let () =
  exit
    (run "vega steps"
       [
         group "sgd"
           [
             sgd_formula;
             test "plain descent does not read the velocity"
               test_sgd_reads_no_velocity;
             sgd_refusals;
           ];
         group "lars" [ lars_formula; lars_refusals ];
         group "adam" [ adam_formula; adam_refuses ];
         group "adamw" [ adamw_formula; adamw_refuses ];
         group "radam"
           [
             radam_formula;
             radam_switch;
             test "radam_step rectifies at float32 as at float64"
               test_radam_rectifies_at_float32;
             radam_refuses;
           ];
         group "lamb" [ lamb_formula; lamb_refuses ];
         group "rmsprop" [ rmsprop_formula; rmsprop_refuses ];
         group "adagrad" [ adagrad_formula; adagrad_refuses ];
         group "adan" [ adan_formula; adan_refuses ];
         group "lion" [ lion_formula; lion_refuses ];
         group "adafactor"
           [
             test "adafactor_init factors the leaves of two axes or more"
               test_adafactor_shapes;
             adafactor_formula;
             empty_factored;
             test "an integer leaf holds zeros of its shape in every part"
               test_adafactor_integer_leaf;
             adafactor_refuses;
           ];
         group "lbfgs"
           [
             test "lbfgs_init holds the parameters and an empty history"
               test_lbfgs_init;
             lbfgs_formula;
             rejected_pair_unseen;
             test "a second step on a quadratic bowl is Newton's"
               test_lbfgs_newton;
             test "the newest accepted pair scales the inverse Hessian"
               test_lbfgs_scales_by_an_accepted_pair;
             test "inner products are float32 when a leaf is not float64"
               test_lbfgs_float32_products;
           ];
       ])
