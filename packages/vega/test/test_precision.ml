(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tests for how a step stores float16 and bfloat16 leaves: with stochastic
   rounding, from draws keyed by the state's counter and the leaf's path, and
   the refusal of float16 leaves by the steps that average squared gradients.
   The laws compare a half step with the same step on the leaves' float32
   copies, which computes the values the half step stores. *)

open Windtrap

let p = Nx.Ptree.tensor

(* Half formats on the host *)

type format = { exponent : int; fraction : int }

let format (type b) (dt : (float, b) Nx.dtype) =
  match dt with
  | Float16 -> { exponent = 5; fraction = 10 }
  | Bfloat16 -> { exponent = 8; fraction = 7 }
  | _ -> invalid_arg "format: not a half dtype"

let bias f = (1 lsl (f.exponent - 1)) - 1

(* The value of the 16 bits [b] in [f]. *)
let decode f b =
  let sign = if b land 0x8000 <> 0 then -1.0 else 1.0 in
  let all = (1 lsl f.exponent) - 1 in
  let ones = (1 lsl f.fraction) - 1 in
  let e = (b lsr f.fraction) land all and m = b land ones in
  let scaled m e =
    sign *. Float.ldexp (Float.of_int m) (e - bias f - f.fraction)
  in
  if e = all then if m = 0 then sign *. infinity else nan
  else if e = 0 then scaled m 1
  else scaled (m lor (1 lsl f.fraction)) e

let largest f =
  let ones = (1 lsl f.fraction) - 1 in
  decode f ((((1 lsl f.exponent) - 2) lsl f.fraction) lor ones)

let min_normal f = Float.ldexp 1.0 (1 - bias f)

(* The spacing of [f]'s values in [[2^e, 2^(e+1))]. *)
let spacing_at f e = Float.ldexp 1.0 (e - f.fraction)

(* Whether [f] holds the finite value [v]: [v] is a multiple of the spacing
   of [f]'s values at its magnitude, and no larger than the largest. *)
let holds f v =
  Float.is_finite v
  && Float.abs v <= largest f
  && (v = 0.0
     ||
     let e = max (snd (Float.frexp v) - 1) (1 - bias f) in
     Float.rem v (spacing_at f e) = 0.0)

let bits (type b) (x : (float, b, _) Nx.t) =
  Nx.to_array (Nx.bitcast Nx.uint16 x)

let floats x = Nx.to_array (Nx.cast Nx.float64 x)

(* [stored f ~x ~near b] is [true] iff the bits [b] store the float32 value [x]
   as a half store does: [x] itself where [f] holds it, a zero with [x]'s sign,
   else one of the two values of [f] bracketing [x]; and [near], the bits
   [Nx.cast] gives, for a NaN, an infinity or a value beyond the largest finite
   one. *)
let stored f ~x ~near b =
  let v = decode f b in
  if Float.is_nan x || Float.abs x > largest f then b = near
  else if v = 0.0 && Float.sign_bit v <> Float.sign_bit x then false
  else if v = x then true
  else
    let toward =
      if v = 0.0 then if x > 0.0 then 1 else 0x8001
      else
        let s = b land 0x8000 and m = b land 0x7FFF in
        if Float.abs x > Float.abs v then s lor (m + 1) else s lor (m - 1)
    in
    let n = decode f toward in
    Float.min v n < x && x < Float.max v n

(* The elements of [half] that do not store [wide]'s values, as text. *)
let misstored (type b) (dt : (float, b) Nx.dtype) name
    (wide : Nx.host Nx.float32_t) (half : (float, b, Nx.host) Nx.t) =
  let f = format dt in
  let x = floats wide and b = bits half and near = bits (Nx.cast dt wide) in
  List.filter_map Fun.id
    (List.init (Array.length x) (fun i ->
         if stored f ~x:x.(i) ~near:near.(i) b.(i) then None
         else
           Some
             (Printf.sprintf "%s[%d]: %h stored as %h" name i x.(i)
                (decode f b.(i)))))

let no_misstored = list string

(* Draws *)

(* A leaf's elements, each in a regime: ordinary values; a zero gradient, whose
   result the dtype holds; values about the least normal one, whose results
   are subnormal or cross zero; the largest finite values, moved by a few
   spacings; and specials: zeros of either sign, NaN and infinite gradients. *)
let element f lr =
  let open Gen in
  let tiny = 2.0 *. min_normal f in
  let top = largest f in
  let spacing = spacing_at f (bias f) in
  let signed r = float_range (-.r) r in
  frequency
    [
      (3, pair (signed 4.0) (signed 4.0));
      (1, pair (signed 4.0) (constant 0.0));
      (2, pair (signed tiny) (map (fun g -> g /. lr) (signed tiny)));
      ( 1,
        pair
          (of_list [ top; -.top; top -. spacing; spacing -. top ])
          (signed (4.0 *. spacing)) );
      ( 1,
        pair
          (of_list [ 0.0; -0.0; 1.0 ])
          (of_list [ nan; infinity; neg_infinity; -0.0 ]) );
    ]

let leaf f =
  let open Gen in
  let* lr = of_list [ 1.0; 0.1; 1e-3; 0x1p-9 ] in
  let* n = int_range 1 8 in
  let+ xs = array ~size:(constant n) (element f lr) in
  (lr, Array.map fst xs, Array.map snd xs)

(* Covers over the values a case stores: [x0] holds the parameters before the
   step, [stores] the float32 values of every part the step stores. *)
let cover_stores f ~x0 stores =
  let x = Array.concat stores in
  let any p = Array.exists p x in
  let top = largest f -. spacing_at f (bias f) in
  cover "a result the dtype holds" (any (holds f));
  cover "a negative result" (any (fun v -> v < 0.0));
  cover "a result across zero"
    (Array.exists2
       (fun a b -> (a > 0.0 && b < 0.0) || (a < 0.0 && b > 0.0))
       x0 (List.hd stores));
  cover "a subnormal result"
    (any (fun v -> v <> 0.0 && Float.abs v < min_normal f));
  cover "a result within a spacing of the largest finite value"
    (any (fun v -> Float.is_finite v && Float.abs v >= top));
  cover "a NaN result" (any Float.is_nan);
  cover "an infinite result" (any (fun v -> Float.abs v = infinity));
  cover "a -0. leaf" (Array.exists (fun a -> a = 0.0 && Float.sign_bit a) x0)

let counter c = Nx.scalar Nx.int32 (Int32.of_int c)

(* Law 1: each store brackets the float32 value *)

let sgd_brackets (type b) (dt : (float, b) Nx.dtype) =
  let f = format dt in
  let gen = Gen.(pair (leaf f) (int_range 0 1000)) in
  prop (Nx.Dtype.name dt ^ " sgd stores bracket the float32 step") gen
    (fun ((lr, x, g), c) ->
      let n = [| Array.length x |] in
      let params = Nx.create dt n x and grads = Nx.create dt n g in
      let wide = Nx.cast Nx.float32 in
      let step params grads =
        let st = { (Vega.sgd_init p params) with step = counter c } in
        fst (Vega.sgd_step p ~lr:(Vega.lr lr) st ~params ~grads)
      in
      let x32 = step (wide params) (wide grads) in
      cover_stores f ~x0:(floats params) [ floats x32 ];
      equal no_misstored [] (misstored dt "params" x32 (step params grads)))

let lion_brackets (type b) (dt : (float, b) Nx.dtype) =
  let f = format dt in
  let gen =
    Gen.(
      let tiny = 2.0 *. min_normal f in
      let average =
        frequency
          [ (2, float_range (-1.0) 1.0); (1, float_range (-.tiny) tiny) ]
      in
      triple (leaf f) (array ~size:(constant 8) average) (int_range 0 1000))
  in
  prop (Nx.Dtype.name dt ^ " lion stores bracket the float32 step") gen
    (fun ((lr, x, g), m, c) ->
      let n = [| Array.length x |] in
      let params = Nx.create dt n x and grads = Nx.create dt n g in
      let mu = Nx.create dt n (Array.sub m 0 (Array.length x)) in
      let wide = Nx.cast Nx.float32 in
      let step params grads mu =
        Vega.lion_step p ~lr:(Vega.lr lr) { mu; step = counter c } ~params
          ~grads
      in
      let x32, st32 = step (wide params) (wide grads) (wide mu) in
      let x16, st16 = step params grads mu in
      cover_stores f ~x0:(floats params) [ floats x32; floats st32.mu ];
      equal no_misstored []
        (misstored dt "params" x32 x16 @ misstored dt "mu" st32.mu st16.mu))

let adam_brackets =
  let dt = Nx.bfloat16 in
  let f = format dt in
  let moments =
    Gen.(
      pair
        (array ~size:(constant 8) (float_range (-1.0) 1.0))
        (array ~size:(constant 8) (float_range 0.0 1.0)))
  in
  let gen = Gen.(triple (leaf f) moments (int_range 0 1000)) in
  prop "bfloat16 adam stores bracket the float32 step" gen
    (fun ((lr, x, g), (m, v), c) ->
      let n = [| Array.length x |] in
      let k = Array.length x in
      let params = Nx.create dt n x and grads = Nx.create dt n g in
      let mu = Nx.create dt n (Array.sub m 0 k)
      and nu = Nx.create dt n (Array.sub v 0 k) in
      let wide = Nx.cast Nx.float32 in
      let step params grads mu nu =
        Vega.adam_step p ~lr:(Vega.lr lr)
          { mu; nu; step = counter c }
          ~params ~grads
      in
      let x32, st32 = step (wide params) (wide grads) (wide mu) (wide nu) in
      let x16, st16 = step params grads mu nu in
      cover_stores f ~x0:(floats params)
        [ floats x32; floats st32.mu; floats st32.nu ];
      equal no_misstored []
        (misstored dt "params" x32 x16
        @ misstored dt "mu" st32.mu st16.mu
        @ misstored dt "nu" st32.nu st16.nu))

(* L-BFGS from any state, over a float16 or bfloat16 leaf beside a float32
   one: every value it stores brackets the value the step from the state's
   float32 copy stores. The pairs below the newest are drawn of positive
   curvature, so that they enter the recursion. Half the states also draw zeros
   of either sign, infinities, NaN, and the extremes of float16 and bfloat16. *)
type lbfgs_leaf = {
  x : float array;
  prev : float array;
  g : float array;
  s : float array;
  y : float array;
}

let half_extremes =
  [ 65504.0; -65504.0; 0x1p-14; 0x1p-24; 0x1.fep127; -0x1.fep127; 0x1p-133 ]

let specials = [ 0.0; -0.0; infinity; neg_infinity; nan ] @ half_extremes

let lbfgs_leaf ~wild history n =
  let open Gen in
  let value r =
    if wild then frequency [ (2, float_range (-.r) r); (1, of_list specials) ]
    else float_range (-.r) r
  in
  let values k r = array ~size:(constant k) (value r) in
  let+ x = values n 4.0
  and+ prev = values n 4.0
  and+ g = values n 4.0
  and+ s = values (history * n) 1.0
  and+ c = array ~size:(constant (history * n)) (float_range 0.5 2.0)
  and+ e = values (history * n) 0.1 in
  let y = Array.init (history * n) (fun i -> (s.(i) *. c.(i)) +. e.(i)) in
  { x; prev; g; s; y }

let lbfgs_draw =
  let open Gen in
  let* history = int_range 1 3 in
  let* n = int_range 0 4 in
  let* m = int_range 0 2 in
  let* wild = bool in
  let+ a = lbfgs_leaf ~wild history n and+ b = lbfgs_leaf ~wild history m in
  (history, a, b)

let lbfgs_brackets (type b) (dt : (float, b) Nx.dtype) =
  prop
    (Nx.Dtype.name dt ^ " beside float32 stores bracket the float32 step")
    lbfgs_draw
    (fun (history, a, b) ->
      let n = Array.length a.x and m = Array.length b.x in
      let both f h =
        ( Nx.create dt (Array.append h [| n |]) (f a),
          Nx.create Nx.float32 (Array.append h [| m |]) (f b) )
      in
      let st =
        {
          Vega.params = both (fun d -> d.x) [||];
          grads = both (fun d -> d.prev) [||];
          s = both (fun d -> d.s) [| history |];
          y = both (fun d -> d.y) [| history |];
          step = Nx.scalar Nx.int32 3l;
        }
      in
      let grads = both (fun d -> d.g) [||] in
      let widen (h, f) = (Nx.cast Nx.float32 h, f) in
      let p = Nx.Ptree.(pair tensor tensor)
      and p32 = Nx.Ptree.(pair tensor tensor) in
      cover "a history of two pairs or more" (history >= 2);
      cover "an empty leaf" (n = 0);
      let drawn = [ a.x; a.prev; a.g; a.s; a.y; b.x; b.prev; b.g; b.s; b.y ] in
      let any f = List.exists (Array.exists f) drawn in
      cover "finite values only" (not (any (fun v -> not (Float.is_finite v))));
      cover "a NaN" (any Float.is_nan);
      cover "an extreme of float16 or bfloat16"
        (any (fun v -> List.mem v half_extremes));
      let lr = Vega.lr 0.1 in
      let wide =
        Vega.lbfgs_step p32 ~lr (widen grads)
          {
            params = widen st.params;
            grads = widen st.grads;
            s = widen st.s;
            y = widen st.y;
            step = st.step;
          }
      in
      let half = Vega.lbfgs_step p ~lr grads st in
      let flat x = Nx.reshape [| -1 |] x in
      let check name (w, w') (h, h') =
        misstored dt name (flat w) (flat h)
        @
        let words x = Nx.to_array (Nx.bitcast Nx.int32 x) in
        if words w' = words h' then []
        else [ name ^ ": the float32 leaf differs" ]
      in
      equal no_misstored []
        (check "params" wide.params half.params
        @ check "s" wide.s half.s @ check "y" wide.y half.y
        @ check "grads" (widen grads) half.grads))

(* Law 2: a store is unbiased *)

(* [2^14] elements at [x] step by [-lr * g]. Where the dtype's spacing between
   [x] and its neighbour toward the step is [spacing], each element stores
   that neighbour with probability [q = lr / spacing] and [x] otherwise, and
   the mean lies within six standard errors of [x - lr * g]. The rows step
   from [1.0] down and from [-1.0] up, where the spacing is half that above,
   and from the least normal value down into the subnormals. *)
let unbiased (type b) (dt : (float, b) Nx.dtype) rows =
  cases
    (Nx.Dtype.name dt ^ " stores are unbiased")
    ~id:(fun (x, g, lr, _) -> Printf.sprintf "%h - %h * %h" x lr g)
    rows
    (fun (x, g, lr, spacing) ->
      let n = 1 lsl 14 in
      let params = Nx.full dt [| n |] x and grads = Nx.full dt [| n |] g in
      let st = Vega.sgd_init p params in
      let params', _ = Vega.sgd_step p ~lr:(Vega.lr lr) st ~params ~grads in
      let q = lr /. spacing in
      let se = spacing *. sqrt (q *. (1.0 -. q) /. Float.of_int n) in
      equal (float (6.0 *. se)) (x -. (lr *. g))
        (Nx.item [] (Nx.mean (Nx.cast Nx.float64 params'))))

(* A float32 rate, as [Vega.lr] holds it. *)
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

let unbiased_rows ~below_one ~least ~sub =
  List.concat_map
    (fun q ->
      let lr = f32 (q *. below_one) in
      [ (1.0, 1.0, lr, below_one); (-1.0, -1.0, lr, below_one) ])
    [ 0x1p-4; 0.256; 0.5; 0.75 ]
  @ List.map (fun q -> (least, 1.0, q *. sub, sub)) [ 0.25; 0.5; 0.75 ]

(* Law 3: moving averages decay *)

(* Adam's second moment of a unit gradient for 100 steps, then of zero: [1 -
   b2^100], then [b2] of it per step. Each decay, a relative [1 - b2 = 1e-3],
   is below half bfloat16's spacing, [2^-9]. A rate of [1e-9] keeps the
   parameters still. *)
let test_moments_decay () =
  let n = 4096 and b2 = 0.999 and rise = 100 in
  let marks = [ 200; 600; 2100 ] in
  let expected t =
    (1.0 -. (b2 ** Float.of_int rise)) *. (b2 ** Float.of_int (t - rise))
  in
  let w = ref (Nx.zeros Nx.bfloat16 [| n |]) in
  let st = ref (Vega.adam_init p !w) in
  let one = Nx.ones Nx.bfloat16 [| n |]
  and zero = Nx.zeros Nx.bfloat16 [| n |] in
  for t = 1 to List.fold_left max 0 marks do
    let grads = if t <= rise then one else zero in
    let w', st' =
      Vega.adam_step p ~lr:(Vega.lr 1e-9) ~b2 !st ~params:!w ~grads
    in
    w := w';
    st := st';
    if List.mem t marks then
      equal
        ~msg:(Printf.sprintf "mean nu at step %d" t)
        (float_rel ~rel:0.01 ~abs:0.0) (expected t)
        (Nx.item [] (Nx.mean (Nx.cast Nx.float64 st'.nu)))
  done

(* Law 4: a bfloat16 trajectory tracks float64 *)

let mean x = Nx.item [] (Nx.mean x)

(* Adam at [1e-3] from [w = 1] under a unit gradient: Adam's direction is
   [1 / (1 + eps)] at every step, so after 500 steps [w] is [1 - 500 lr / (1 +
   eps)], about [0.5]. Each element's error is the walk of 500 stores of a
   spacing [2^-8] or [2^-9]: its mean over 4096 elements lies within five
   standard errors of zero, and its mean magnitude below the walk's root mean
   square, [0.038]. *)
let test_tracks_a_constant_gradient () =
  let n = 4096 and steps = 500 in
  let lr = Int32.float_of_bits (Int32.bits_of_float 1e-3) in
  let reference = 1.0 -. (Float.of_int steps *. lr /. (1.0 +. 1e-8)) in
  let w = ref (Nx.ones Nx.bfloat16 [| n |]) in
  let st = ref (Vega.adam_init p !w) in
  let grads = Nx.ones Nx.bfloat16 [| n |] in
  for _ = 1 to steps do
    let w', st' = Vega.adam_step p ~lr:(Vega.lr lr) !st ~params:!w ~grads in
    w := w';
    st := st'
  done;
  let error = Nx.sub_s (Nx.cast Nx.float64 !w) reference in
  less ~msg:"|mean error|" float_exact ~than:2e-3 (Float.abs (mean error));
  less ~msg:"mean |error|" float_exact ~than:0.06 (mean (Nx.abs error))

(* Adam at [1e-3] on [|w - t|^2 / 2] under gradient noise of deviation 0.1,
   2000 steps from [w, t ~ U[-1, 1]]; the bfloat16 gradient is taken at the
   bfloat16 weights. The float64 run on the same noise ends at a mean
   [|w - t|] of about [0.04]; the bfloat16 one stays within [0.02] of it. *)
let test_tracks_a_noisy_quadratic () =
  let n = 4096 and steps = 2000 in
  let keys = Nx.Rng.split ~n:3 (Nx.Rng.key 3) in
  let uniform k =
    Nx.sub_s (Nx.mul_s (Nx.Rng.uniform ~key:k Nx.float32 [| n |]) 2.0) 1.0
  in
  let w0 = uniform keys.(0) and t = uniform keys.(1) in
  let noise i =
    Nx.mul_s
      (Nx.Rng.normal ~key:(Nx.Rng.fold_in keys.(2) i) Nx.float32 [| n |])
      0.1
  in
  let lr = Vega.lr 1e-3 in
  let run (type b) (dt : (float, b) Nx.dtype) =
    let w = ref (Nx.cast dt w0) in
    let st = ref (Vega.adam_init p !w) in
    for i = 1 to steps do
      let g = Nx.add (Nx.sub (Nx.cast Nx.float32 !w) t) (noise i) in
      let w', st' =
        Vega.adam_step p ~lr !st ~params:!w ~grads:(Nx.cast dt g)
      in
      w := w';
      st := st'
    done;
    Nx.cast Nx.float64 !w
  in
  let reference = run Nx.float64 and half = run Nx.bfloat16 in
  less ~msg:"mean |w - reference|" float_exact ~than:0.02
    (mean (Nx.abs (Nx.sub half reference)))

(* Law 5: a step is deterministic *)

(* Two leaves, a bfloat16 one under Adam's three stores and a float16 one
   under SGD's two, stepped toward a target. *)
let target = 0.25

let adam_run k (w, st) =
  let rec go k (w, st) =
    if k = 0 then (w, st)
    else
      let grads = Nx.sub_s w target in
      go (k - 1) (Vega.adam_step p ~lr:(Vega.lr 0.01) st ~params:w ~grads)
  in
  go k (w, st)

let sgd_run k (w, st) =
  let rec go k (w, st) =
    if k = 0 then (w, st)
    else
      let grads = Nx.sub_s w target in
      go (k - 1)
        (Vega.sgd_step p ~lr:(Vega.lr 0.01) ~momentum:0.9 st ~params:w ~grads)
  in
  go k (w, st)

(* [saved x] is [x] through its elements on the host, as a checkpoint holds
   it. *)
let saved (type b) (x : (float, b, Nx.host) Nx.t) =
  Nx.create (Nx.dtype x) (Nx.shape x) (Nx.to_array x)

(* The number of elements whose bits differ between [a] and [b]. *)
let differ a b =
  let a = bits a and b = bits b in
  Array.fold_left ( + ) 0 (Array.map2 (fun x y -> Bool.to_int (x <> y)) a b)

let same name a b = equal ~msg:name int 0 (differ a b)

let test_deterministic =
  let gen = Gen.(array ~size:(int_range 1 16) (float_range (-1.0) 1.0)) in
  prop "equal inputs give equal bits, a checkpoint included" gen (fun xs ->
      let n = [| Array.length xs |] in
      let w = Nx.create Nx.bfloat16 n xs in
      let fresh () = (Nx.copy w, Vega.adam_init p (Nx.copy w)) in
      let w1, st1 = adam_run 10 (fresh ())
      and w2, st2 = adam_run 10 (fresh ()) in
      same "adam: two copies" w1 w2;
      same "adam: their mu" st1.mu st2.mu;
      same "adam: their nu" st1.nu st2.nu;
      let wh, sth = adam_run 5 (fresh ()) in
      let resumed =
        ( saved wh,
          {
            Vega.mu = saved sth.mu;
            nu = saved sth.nu;
            step = Nx.scalar Nx.int32 (Nx.item [] sth.step);
          } )
      in
      let wr, str = adam_run 5 resumed in
      same "adam: resumed from a checkpoint" w1 wr;
      same "adam: its nu" st1.nu str.nu;
      let v = Nx.create Nx.float16 n xs in
      let fresh () = (Nx.copy v, Vega.sgd_init p (Nx.copy v)) in
      let v1, s1 = sgd_run 10 (fresh ()) and v2, s2 = sgd_run 10 (fresh ()) in
      same "sgd: two copies" v1 v2;
      same "sgd: their velocity" s1.velocity s2.velocity)

(* Keys *)

(* [2^14] bfloat16 elements at [1.0] stepped at [2^-9], half a spacing: each
   rounds either way with probability [1/2], so draws that differ change some
   element. *)
let n_half = 1 lsl 14
let midway = 0x1p-9

let sgd_at c params =
  let grads = Nx.ones_like params in
  let st = { (Vega.sgd_init p params) with step = counter c } in
  fst (Vega.sgd_step p ~lr:(Vega.lr midway) st ~params ~grads)

let test_counter_keys () =
  let params = Nx.ones Nx.bfloat16 [| n_half |] in
  greater ~msg:"elements that counters 0 and 1 round apart" int ~than:0
    (differ (sgd_at 0 params) (sgd_at 1 params));
  same "counter 7, twice" (sgd_at 7 params) (sgd_at 7 params)

(* Lion at [2^-9] from [1.0], with [mu = 1], [g = 1 + 2^-7] and [b2 = 1/2]:
   the parameter, [1 - 2^-9], and the average, [1 + 2^-8], both lie midway
   between their neighbours, and in both the low 16 bits are [0x8000]. Stores
   that drew the same bits would round both up or both down at every element;
   independent ones agree at half of them, within six standard errors. The
   rows lay the leaf out as a vector and as a column. *)
let stores_draw_apart =
  cases "a leaf's stores draw apart" ~id:(fun (name, _) -> name)
    [ ("a vector", [| n_half |]); ("a column", [| n_half; 1 |]) ]
    (fun (_, shape) ->
      let params = Nx.ones Nx.bfloat16 shape in
      let grads = Nx.full Nx.bfloat16 shape (1.0 +. 0x1p-7) in
      let st = { Vega.mu = Nx.ones Nx.bfloat16 shape; step = counter 3 } in
      let params', st' =
        Vega.lion_step p ~lr:(Vega.lr midway) ~b2:0.5 st ~params ~grads
      in
      let up = Array.map (fun v -> v > 0.999) (floats params')
      and mu_up = Array.map (fun v -> v > 1.003) (floats st'.mu) in
      let alike a b = Bool.to_int (a = b) in
      let agree = Array.fold_left ( + ) 0 (Array.map2 alike up mu_up) in
      let sd = 0.5 *. sqrt (Float.of_int n_half) in
      equal ~msg:"elements whose stores round alike" (float (6.0 *. sd))
        (Float.of_int n_half /. 2.0)
        (Float.of_int agree))

(* Two leaves at [1.0] of one structure, and a third leaf beside them. *)
module Two = struct
  type t = { a : Nx.host Nx.bfloat16_t; b : Nx.host Nx.bfloat16_t }

  let walk c { a; b } =
    let open Nx.Ptree.Walk in
    let a = field c "a" tensor a in
    let b = field c "b" tensor b in
    { a; b }

  let ptree : (t, Nx.host) Nx.Ptree.t = Nx.Ptree.v walk
end

module Three = struct
  type t = {
    a : Nx.host Nx.bfloat16_t;
    z : Nx.host Nx.bfloat16_t;
    b : Nx.host Nx.bfloat16_t;
  }

  let walk c { a; z; b } =
    let open Nx.Ptree.Walk in
    let a = field c "a" tensor a in
    let z = field c "z" tensor z in
    let b = field c "b" tensor b in
    { a; z; b }

  let ptree : (t, Nx.host) Nx.Ptree.t = Nx.Ptree.v walk
end

let step2 ptree params grads =
  let st = Vega.adam_init ptree params in
  Vega.adam_step ptree ~lr:(Vega.lr midway) st ~params ~grads

let test_path_keys () =
  let ones () = Nx.ones Nx.bfloat16 [| n_half |] in
  let small () = Nx.full Nx.bfloat16 [| n_half |] 1e-3 in
  let two, st2 =
    step2 Two.ptree
      { a = ones (); b = ones () }
      { a = small (); b = small () }
  in
  greater ~msg:"elements that two paths round apart" int ~than:0
    (differ two.a two.b);
  let three, st3 =
    step2 Three.ptree
      { a = ones (); z = ones (); b = ones () }
      { a = small (); z = small (); b = small () }
  in
  same "a leaf added before b leaves b" two.b three.b;
  same "nor its mu" st2.mu.b st3.mu.b;
  same "nor its nu" st2.nu.b st3.nu.b;
  same "nor a" two.a three.a

(* A bfloat16 vector alone on a device of a set of two, and a matrix split
   along either axis of the set: the draws are the same at every placement, so
   the stores are the host's. *)
let test_placement () =
  let device name = Result.get_ok (Rig.memory_device name) in
  let d0 = device "vega-precision-0" and d1 = device "vega-precision-1" in
  let module S = (val Nx.devices [ d0; d1 ]) in
  let step x =
    let p = Nx.Ptree.tensor in
    let st = Vega.sgd_init p x and lr = Vega.lr midway in
    fst (Vega.sgd_step p ~lr ~momentum:0.9 st ~params:x ~grads:x)
  in
  let flat x = Nx.reshape [| -1 |] (Nx.place Nx.Host.on x) in
  let x = Nx.ones Nx.bfloat16 [| n_half |] in
  same "a vector alone on a device" (step x)
    (flat (step (Nx.place (Nx.Placement.device S.v d0) x)));
  let m = Nx.ones Nx.bfloat16 [| 64; 256 |] in
  let host = flat (step m) in
  same "a matrix split along axis 0" host
    (flat (step (Nx.place (S.split ~axis:0) m)));
  same "a matrix split along axis 1" host
    (flat (step (Nx.place (S.split ~axis:1) m)))

(* Float16 leaves *)

(* Each step that averages squared gradients, by name, from its initial
   state at [x]. *)
let float16_steps =
  let lr = Vega.lr 0.1 in
  let step name init step = (name, fun x -> ignore (step (init x) x)) in
  [
    step "Vega.adam_step" (Vega.adam_init p) (fun st x ->
        Vega.adam_step p ~lr st ~params:x ~grads:x);
    step "Vega.adamw_step" (Vega.adamw_init p) (fun st x ->
        Vega.adamw_step p ~lr st ~params:x ~grads:x);
    step "Vega.radam_step" (Vega.radam_init p) (fun st x ->
        Vega.radam_step p ~lr st ~params:x ~grads:x);
    step "Vega.lamb_step" (Vega.lamb_init p) (fun st x ->
        Vega.lamb_step p ~lr st ~params:x ~grads:x);
    step "Vega.rmsprop_step" (Vega.rmsprop_init p) (fun st x ->
        Vega.rmsprop_step p ~lr st ~params:x ~grads:x);
    step "Vega.adagrad_step" (Vega.adagrad_init p) (fun st x ->
        Vega.adagrad_step p ~lr st ~params:x ~grads:x);
    step "Vega.adan_step" (Vega.adan_init p) (fun st x ->
        Vega.adan_step p ~lr st ~params:x ~grads:x);
    step "Vega.adafactor_step"
      (fun x -> Vega.adafactor_init p x)
      (fun st x -> Vega.adafactor_step p ~lr st ~params:x ~grads:x);
  ]

let float16_refused =
  cases "steps that average squared gradients refuse float16 leaves" ~id:fst
    float16_steps (fun (name, step) ->
      raises
        (Invalid_argument
           (name
          ^ ": the root: float16 cannot hold the average of squared \
             gradients; use bfloat16 parameters, or float32 ones cast to \
             float16 inside the objective"))
        (fun () -> step (Nx.ones Nx.float16 [| 3 |])))

let test_float16_accepted () =
  let x = Nx.ones Nx.float16 [| 3 |] and lr = Vega.lr 0.1 in
  let moved name w =
    less ~msg:name float_exact ~than:1.0 (Nx.item [ 0 ] (Nx.cast Nx.float64 w))
  in
  moved "sgd"
    (fst (Vega.sgd_step p ~lr (Vega.sgd_init p x) ~params:x ~grads:x));
  moved "lars"
    (fst (Vega.lars_step p ~lr (Vega.lars_init p x) ~params:x ~grads:x));
  moved "lion"
    (fst (Vega.lion_step p ~lr (Vega.lion_init p x) ~params:x ~grads:x));
  moved "lbfgs" (Vega.lbfgs_step p ~lr x (Vega.lbfgs_init p x)).params

(* Float32 and float64 leaves *)

(* Each step's float32 and float64 results, printed bit for bit: the baseline
   of the code before half stores rounded stochastically, which they must not
   change. *)
let test_wide_leaves_unchanged () =
  let lr = Vega.lr 0.05 in
  let show name x =
    let xs = List.map (Printf.sprintf "%h") (Array.to_list (floats x)) in
    Printf.printf "%s %s\n" name (String.concat " " xs)
  in
  let run (type b) (dt : (float, b) Nx.dtype) =
    let x = Nx.create dt [| 3 |] [| 0.7; -1.3; 2.1 |] in
    let g w = Nx.sub_s (Nx.mul w w) 0.5 in
    let rec go k f (w, st) =
      if k = 0 then (w, st) else go (k - 1) f (f st ~params:w ~grads:(g w))
    in
    let name = Nx.Dtype.name dt in
    let sgd st = Vega.sgd_step p ~lr ~momentum:0.9 st in
    let w, _ = go 3 sgd (x, Vega.sgd_init p x) in
    show (name ^ " sgd") w;
    let adam st = Vega.adam_step p ~lr st in
    let w, st = go 3 adam (x, Vega.adam_init p x) in
    show (name ^ " adam") w;
    show (name ^ " adam nu") st.nu;
    let lion st = Vega.lion_step p ~lr st in
    let w, _ = go 3 lion (x, Vega.lion_init p x) in
    show (name ^ " lion") w;
    let adafactor st = Vega.adafactor_step p ~lr st in
    let w, _ = go 3 adafactor (x, Vega.adafactor_init p x) in
    show (name ^ " adafactor") w;
    let st = ref (Vega.lbfgs_init p x) in
    for _ = 1 to 3 do
      st := Vega.lbfgs_step p ~lr (g !st.params) !st
    done;
    show (name ^ " lbfgs") !st.params
  in
  run Nx.float32;
  run Nx.float64;
  expect (output ())
  @@ __POS_OF__
       {|
    float32 sgd 0x1.67c058p-1 -0x1.ac85c4p+0 0x1.2cadecp+0
    float32 adam 0x1.5fc2ecp-1 -0x1.73437ap+0 0x1.f348fap+0
    float32 adam nu 0x1.1e7102p-18 0x1.5adc7p-8 0x1.518cd2p-5
    float32 lion 0x1.8p-1 -0x1.73333p+0 0x1.f33334p+0
    float32 adafactor 0x1.704ce2p-1 -0x1.73547cp+0 0x1.f55f64p+0
    float32 lbfgs 0x1.6722ap-1 -0x1.922a7ap+0 0x1.b2afd8p+0
    float64 sgd 0x1.67c0578525aa5p-1 -0x1.ac85c5f6a813ap+0 0x1.2cadec8409863p+0
    float64 adam 0x1.5fc2eaac0398cp-1 -0x1.73437b0c37859p+0 0x1.f348f90279184p+0
    float64 adam nu 0x1.1e70fd43acaedp-18 0x1.5adc7634754bp-8 0x1.518cd2f435fe6p-5
    float64 lion 0x1.8000000666666p-1 -0x1.7333333cccccdp+0 0x1.f33333299999ap+0
    float64 adafactor 0x1.704ce210bc15ep-1 -0x1.73547cb83be79p+0 0x1.f55f62f491493p+0
    float64 lbfgs 0x1.67229f93792bp-1 -0x1.922a8057ad417p+0 0x1.b2afd597746f8p+0
    |}

let tests =
  [
    group "stores bracket the computed value"
      [
        sgd_brackets Nx.float16;
        sgd_brackets Nx.bfloat16;
        lion_brackets Nx.float16;
        lion_brackets Nx.bfloat16;
        adam_brackets;
        lbfgs_brackets Nx.float16;
        lbfgs_brackets Nx.bfloat16;
      ];
    group "stores are unbiased"
      [
        unbiased Nx.bfloat16
          (unbiased_rows ~below_one:0x1p-8 ~least:0x1p-126 ~sub:0x1p-133);
        unbiased Nx.float16
          (unbiased_rows ~below_one:0x1p-11 ~least:0x1p-14 ~sub:0x1p-24);
      ];
    group "trajectories"
      [
        test "a bfloat16 moving average decays" test_moments_decay;
        test "bfloat16 adam tracks a constant gradient"
          test_tracks_a_constant_gradient;
        test "bfloat16 adam tracks a noisy quadratic"
          test_tracks_a_noisy_quadratic;
      ];
    group "draws"
      [
        test_deterministic;
        test "the counter keys the draws" test_counter_keys;
        stores_draw_apart;
        test "the path keys the draws" test_path_keys;
        test "the draws are the same at every placement" test_placement;
      ];
    group "float16"
      [
        float16_refused;
        test "sgd, lars, lion and lbfgs step float16 leaves"
          test_float16_accepted;
      ];
    test "float32 and float64 leaves step as before"
      test_wide_leaves_unchanged;
  ]

let () = exit (run "vega precision" tests)
