(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Schedule = Schedule

(* Helpers *)

(* A parameter structure may carry leaves that are not parameters: an RNG key
   threaded through a compiled step, a step counter, a batch of indices. Rune
   does not differentiate them — their slot in the gradient structure holds
   zeros — and an optimizer must not update them either. Adam's square root over
   an integer leaf is meaningless, and even plain descent would round its step
   into the value. Carry them instead. Float and complex leaves step. *)
let updates (type a b d) (p : (a, b, d) Nx.t) =
  match Nx.Dtype.kind (Nx.dtype p) with
  | Float | Complex -> true
  | Signed | Unsigned | Boolean -> false

(* Validation *)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* [num x] is the shortest decimal that reads as [x], the same on every system.
   [Printf]'s [%g] is the C library's, which under mingw-w64 writes three
   exponent digits: 1e+006. *)
let num x = Format.asprintf "%a" (Nx.Dtype.pp_value Nx.float64) x

let validate_positive ctx name value =
  if value <= 0.0 then
    invalid_argf "%s: expected %s > 0.0, got %s" ctx name (num value)

let validate_non_negative ctx name value =
  if value < 0.0 then
    invalid_argf "%s: expected %s >= 0.0, got %s" ctx name (num value)

let validate_unit_interval ctx name value =
  if value < 0.0 || value >= 1.0 then
    invalid_argf "%s: expected 0.0 <= %s < 1.0, got %s" ctx name (num value)

(* Optimizers over parameter structures (Nx.Ptree.t). Optimizer state is itself
   parameter-shaped, and every scalar that changes across steps — the bias
   corrections, the step counter — is a tensor leaf, so a training step is a
   pure function of (params, state) that traces under Rune.jit. *)

(* Values that share the parameters' skeleton. A step flattens the parameters
   and every other value once, checks that each has the parameters' skeleton,
   then walks the parameters, taking one tensor of each other value per leaf. *)

let describe path =
  match Nx.Ptree.Path.segments path with
  | [] -> "the root"
  | _ -> Nx.Ptree.Path.to_string path

let aligned fn p skeleton name x =
  let leaves, k = Nx.Ptree.flatten p x in
  match
    Nx.Ptree.Skeleton.diff ~this:("in " ^ name) k ~that:"in the parameters"
      skeleton
  with
  | None -> leaves
  | Some m -> invalid_arg (fn ^ ": " ^ m)

let take (type a b d) fn name path (x : (a, b, d) Nx.t) rest : (a, b, d) Nx.t =
  match !rest with
  | [] -> invalid_arg (fn ^ ": the structure's walk visited one value two ways")
  | Nx.P y :: tail -> (
      rest := tail;
      match Nx.Dtype.equal_witness (Nx.dtype x) (Nx.dtype y) with
      | Some Equal -> y
      | None ->
          invalid_argf "%s: %s: %s in %s, %s in the parameters" fn
            (describe path)
            (Nx.Dtype.name (Nx.dtype y))
            name
            (Nx.Dtype.name (Nx.dtype x)))

(* A step computes at float32, or at float64 for a float64 or complex128 leaf,
   and returns each leaf at its own dtype. Float16 and bfloat16 leaves round
   once, when the result is stored: at their dtype, Adam's [eps = 1e-8] rounds
   to zero in float16 and [b2 = 0.999] to one in bfloat16. *)
type ('w, 'd) compute = (float, 'w, 'd) Nx.t

(* Stochastic rounding *)

(* A float16 or bfloat16 store rounds the float32 value [x], between two
   neighbouring values [lo < x < hi] of the dtype, to [hi] with probability
   [(x - lo) / (hi - lo)] and to [lo] otherwise, so that it is [x] in
   expectation (Croci et al., "Stochastic rounding: implementation, error
   analysis and applications", R. Soc. Open Sci. 2022, §6.1). Rounding to
   nearest loses every update below half a spacing: a bfloat16 weight of 1
   under Adam at a rate of 1e-3 never moves, and a second moment under
   [b2 = 0.999] never decays (Zamirai et al., "Revisiting BFloat16 Training",
   2020, §3.2).

   Each store takes 16 random bits per element from {!Nx.Rng.bits}, where a
   32-bit word costs one Threefry block. A leaf's stores in one step are
   numbered, the parameter 0 and the state's part [i] then [i + 1], and store
   [k] takes the low half of word [k / 2], drawn at its shape, for an even
   [k] and the high half for an odd one: two stores share one word per
   element, and the draws are elementwise, the same at every placement. Two
   elements per word would halve SGD's blocks, but splitting a draw across
   elements needs an axis no placement cuts.

   Every word's key is the step's base key, which holds the state's counter,
   folded with a 62-bit FNV-1a hash of the leaf's path and the word: a
   function of the step's inputs, so a step is deterministic and a checkpoint
   resumes bit for bit, and of the path rather than the leaf's position, so a
   leaf added to the structure leaves the others' draws alone. *)

(* FNV-1a over 64 bits (Fowler, Noll and Vo): its constants fix the hash across
   OCaml versions, which [Hashtbl.hash] does not. *)
let fnv_offset = 0xcbf29ce484222325L
let fnv_prime = 0x100000001b3L

let fnv1a h s =
  let byte h c =
    Int64.mul (Int64.logxor h (Int64.of_int (Char.code c))) fnv_prime
  in
  String.fold_left byte h s

(* The hash of word [j] of the draws at [path], in [0, 2^62). Each segment
   ends in a zero byte, so the field ["a.b"] and the fields [a] then [b] hash
   apart. *)
let path_hash path j =
  let segment h : Nx.Ptree.Path.seg -> int64 = function
    | Field s -> fnv1a h ("f" ^ s ^ "\000")
    | Index i -> fnv1a h (Printf.sprintf "i%d\000" i)
  in
  let h = List.fold_left segment fnv_offset (Nx.Ptree.Path.segments path) in
  Int64.to_int (fnv1a h (Printf.sprintf "w%d" j)) land max_int

(* The words of the base key: a constant and the counter. *)
let base_word = 0x76656761l

(* [base counter] is a step's base key, made where the counter lies, once. *)
let base counter =
  lazy
    (Nx.Rng.of_tensor (Nx.stack [ Nx.scalar_like counter base_word; counter ]))

(* [draws base path] gives store [k] at [path] its 16 random bits per element
   of [y], the value it stores, as a uint32 below [2^16]. Word [j] is drawn
   once per shape, where [y] lies. *)
let draws base path =
  let drawn = ref [] in
  let words j at shape =
    match List.assoc_opt (j, shape) !drawn with
    | Some w -> w
    | None ->
        let key = Nx.Rng.fold_in (Lazy.force base) (path_hash path j) in
        let w = Nx.Rng.bits_at at ~key shape in
        drawn := ((j, shape), w) :: !drawn;
        w
  in
  let low w = Nx.cast Nx.uint32 (Nx.cast Nx.uint16 w)
  and high w = Nx.rshift (Nx.bitcast Nx.uint32 w) 16 in
  fun k (y : (float, _, _) Nx.t) ->
    let w = words (k / 2) (Nx.placement y) (Nx.shape y) in
    if k mod 2 = 0 then low w else high w

let bfloat16_largest = 0x1.fep127

(* Bfloat16 is float32's top 16 bits, sign and magnitude. Adding [u], 16
   random bits, to [x]'s bits and keeping the top 16 rounds the magnitude up
   to the next bfloat16 value where the sum carries into bit 16, with
   probability (x's low 16 bits) / 2^16, and down otherwise: that probability
   is [(|x| - lo) / (hi - lo)] in every binade, subnormals included. Up to the
   largest finite magnitude the carry stops below the sign bit; above it, and
   for NaN and the infinities, the store is the cast's. *)
let bfloat16_store x u =
  let sum = Nx.add (Nx.bitcast Nx.uint32 x) u in
  let rounded = Nx.bitcast Nx.bfloat16 (Nx.cast Nx.uint16 (Nx.rshift sum 16)) in
  Nx.where
    (Nx.less_equal_s (Nx.abs x) bfloat16_largest)
    rounded (Nx.cast Nx.bfloat16 x)

(* Float16's exponent is narrower than float32's, so its store picks between
   the nearest value [r] and its other neighbour [n]: [r]'s bits stepped one
   away from zero where [|x| >= |r|], toward it otherwise, in a format of sign
   and magnitude whose cast keeps [x]'s sign, zeros included. It takes [n]
   where [u / 2^16 < |x - r| / |n - r|], at most a half, exact at float32. The
   ratio is a multiple of [2^-13] for a normal [x], so the probability is
   exact, and within [2^-16] of it at the subnormals. NaN, the infinities and
   values the cast rounds to one make the ratio NaN or zero, keeping [r]. *)
let float16_store x u =
  let r = Nx.cast Nx.float16 x in
  let r32 = Nx.cast Nx.float32 r and b = Nx.bitcast Nx.int16 r in
  let away = Nx.greater_equal (Nx.abs x) (Nx.abs r32) in
  let n =
    Nx.bitcast Nx.float16 (Nx.where away (Nx.add_s b 1) (Nx.sub_s b 1))
  in
  let gap = Nx.abs (Nx.sub (Nx.cast Nx.float32 n) r32) in
  let ratio = Nx.div (Nx.abs (Nx.sub x r32)) gap in
  let u = Nx.mul_s (Nx.cast Nx.float32 u) 0x1p-16 in
  Nx.where (Nx.less u ratio) n r

(* [store draw k like y] is [y], at its compute dtype, stored at [like]'s
   dtype as store [k] of its leaf. *)
let store (type a b c e d)
    (draw : int -> (float, Nx.float32_elt, d) Nx.t -> d Nx.uint32_t) k
    (like : (a, b, d) Nx.t) (y : (c, e, d) Nx.t) : (a, b, d) Nx.t =
  match (Nx.dtype like, Nx.dtype y) with
  | Float16, Float32 -> float16_store y (draw k y)
  | Bfloat16, Float32 -> bfloat16_store y (draw k y)
  | dt, _ -> Nx.cast dt y

(* The compute dtypes, as a witness a step's arithmetic is written against. *)
type _ wide = F32 : Nx.float32_elt wide | F64 : Nx.float64_elt wide

let dtype : type w. w wide -> (float, w) Nx.dtype = function
  | F32 -> Nx.float32
  | F64 -> Nx.float64

(* A float16 leaf cannot hold an average of squared gradients: about
   [(1 - b2) g^2] after Adam's first step, it is below float16's least
   subnormal, [2^-24], for [|g|] under about [8e-3], and stores zero, where the
   step divides by its root. Rounding stochastically keeps it unbiased but
   mostly zero, which [1 / sqrt] does not average away (Li et al., "Memory
   Efficient Optimizers with 4-bit States", NeurIPS 2023, §4.1, the
   zero-point problem). Bfloat16 has float32's range. *)
let refuse_float16 fn path =
  invalid_argf
    "%s: %s: float16 cannot hold the average of squared gradients; use \
     bfloat16 parameters, or float32 ones cast to float16 inside the objective"
    fn (describe path)

(* [leafwise fn p ~counter ~params ~grads parts ~f] steps [params] leaf by
   leaf. [parts] are the state's values of the parameters' skeleton, each with
   its name for errors. At a float leaf, [f w x g s] takes the parameter, its
   gradient and its leaf of each part, in order, cast to the compute dtype
   [w], and returns the new parameter and the parts' new leaves, which are
   stored back at the leaf's dtype, a half one with draws keyed by [counter].
   A part [f] returns as it took it, or as the gradient, is the input itself,
   with nothing drawn. With [squared], the parts average squared gradients
   and a float16 leaf raises. [f]'s first argument is [decays path shape],
   asked once per float or complex leaf with the parameter's own shape, true
   for every leaf by default.

   A complex leaf steps as its real and imaginary parts, two real coordinates
   per element, stacked along a new first axis: [f] sees the leaf of real
   parameters a real optimizer would. Its gradient there is the conjugate of
   Rune's: the gradient of a real objective of [z = x + iy] is
   [∂f/∂x - i ∂f/∂y], whose parts' gradient is [(∂f/∂x, ∂f/∂y)], so the step
   descends along the conjugate. Other leaves pass through unchanged. The
   result is the new parameters and the new parts. *)
let leafwise ?(squared = false) ?(decays = fun _ _ -> true) fn p ~counter
    ~params ~grads parts
    ~(f :
       'w.
       bool ->
       'w wide ->
       ('w, 'd) compute ->
       ('w, 'd) compute ->
       ('w, 'd) compute array ->
       ('w, 'd) compute * ('w, 'd) compute array) =
  let skeleton = snd (Nx.Ptree.flatten p params) in
  let aligned name x = ref (aligned fn p skeleton name x) in
  let grads = aligned "the gradients" grads in
  let parts = Array.of_list parts in
  let leaves = Array.map (fun (name, x) -> (name, aligned name x)) parts in
  let parts' = Array.map (fun _ -> ref []) parts in
  let base = base counter in
  let real (type b w) path decays (w : w wide) (x : (float, b, _) Nx.t) g s =
    (match Nx.dtype x with
    | Float16 when squared -> refuse_float16 fn path
    | _ -> ());
    let wide = dtype w in
    let g' = Nx.cast wide g and s' = Array.map (Nx.cast wide) s in
    let x', t = f decays w (Nx.cast wide x) g' s' in
    match Nx.dtype x with
    | Float16 | Bfloat16 ->
        let draw = draws base path in
        let part i y =
          if y == s'.(i) then s.(i)
          else if y == g' then g
          else store draw (i + 1) x y
        in
        (store draw 0 x x', Array.mapi part t)
    | dt -> (Nx.cast dt x', Array.map (Nx.cast dt) t)
  in
  let complex (type b w) decays (w : w wide) (x : (Complex.t, b, _) Nx.t) g s
      =
    let dt = Nx.dtype x and wide = dtype w in
    let parts z = Nx.stack [ Nx.real wide z; Nx.imag wide z ] in
    let join r = Nx.complex dt ~re:(Nx.get [ 0 ] r) ~im:(Nx.get [ 1 ] r) in
    let g = Nx.stack [ Nx.real wide g; Nx.neg (Nx.imag wide g) ] in
    let x, s = f decays w (parts x) g (Array.map parts s) in
    (join x, Array.map join s)
  in
  let step (type a b) path (x : (a, b, _) Nx.t) (g : (a, b, _) Nx.t)
      (s : (a, b, _) Nx.t array) :
      (a, b, _) Nx.t * (a, b, _) Nx.t array =
    let dt = Nx.dtype x in
    let asked () = decays path (Nx.shape x) in
    match (Nx.Dtype.kind dt, dt) with
    | Float, Float64 -> real path (asked ()) F64 x g s
    | Float, _ -> real path (asked ()) F32 x g s
    | Complex, Complex128 -> complex (asked ()) F64 x g s
    | Complex, _ -> complex (asked ()) F32 x g s
    | (Signed | Unsigned | Boolean), _ -> (x, s)
  in
  let update path x =
    let g = take fn "the gradients" path x grads in
    let s = Array.map (fun (name, rest) -> take fn name path x rest) leaves in
    let x, s = step path x g s in
    Array.iteri (fun i y -> parts'.(i) := Nx.P y :: !(parts'.(i))) s;
    x
  in
  let params = Nx.Ptree.map p update params in
  let rebuild i (_, like) = Nx.Ptree.rebuild p ~like (List.rev !(parts'.(i))) in
  (params, Array.mapi rebuild parts)

(* A scalar of the step counter, computed once per step at each compute dtype
   that asks for it. *)
type 'd counted = {
  f32 : 'd Nx.float32_t Lazy.t;
  f64 : 'd Nx.float64_t Lazy.t;
}

let counted (f : 'w. (float, 'w) Nx.dtype -> ('w, 'd) compute) =
  { f32 = lazy (f Nx.float32); f64 = lazy (f Nx.float64) }

let at : type w d. d counted -> w wide -> (w, d) compute =
 fun c -> function F32 -> Lazy.force c.f32 | F64 -> Lazy.force c.f64

(* [one_minus_exp x] is [1 - exp x] for [x <= 0] without the cancellation of
   subtracting from one: [-expm1 x], with [expm1] written with [exp] and [log]
   as [(u - 1) x / log u] for [u = exp x], whose rounding errors cancel. It is
   [x] where [u] rounds to one, and [u - 1] below [-1], which loses nothing and
   stays defined where [u] underflows to zero. *)
let one_minus_exp x =
  let u = Nx.exp x in
  let near = Nx.div (Nx.mul (Nx.sub_s u 1.0) x) (Nx.log u) in
  let expm1 = Nx.where (Nx.equal_s u 1.0) x near in
  Nx.neg (Nx.where (Nx.less_s x (-1.0)) (Nx.sub_s u 1.0) expm1)

(* [counter fn p params] is a step counter of zero whole on every device of the
   parameters' set, so that it, and the scalars a step makes of it, stand
   beside a leaf on any of those devices. *)
let counter fn p params =
  match fst (Nx.Ptree.flatten p params) with
  | Nx.P x :: _ ->
      let every = Nx.Placement.(on (devices (Nx.placement x))) in
      Nx.scalar_at every Nx.int32 0l
  | [] -> invalid_arg (fn ^ ": the parameters hold no array")

(* Gradient transformations *)

let clip_by_global_norm p ~max_norm grads =
  validate_positive "Vega.clip_by_global_norm" "max_norm" max_norm;
  match fst (Nx.Ptree.flatten p grads) with
  | [] -> grads
  | _ :: _ ->
      (* The norm is accumulated at float32, which every device computes. *)
      let norm = Nx.Ptree.norm p Nx.float32 grads in
      let factor =
        Nx.where
          (Nx.greater_s norm max_norm)
          (Nx.rdiv_s max_norm norm) (Nx.ones_like norm)
      in
      Nx.Ptree.scale p factor grads

(* A complex gradient is clipped part by part, as two real coordinates. *)
let clip (type a b d) ~max (g : (a, b, d) Nx.t) : (a, b, d) Nx.t =
  let dt = Nx.dtype g in
  match Nx.Dtype.kind dt with
  | Complex ->
      let parts (type w) (w : w wide) =
        let part z = Nx.clamp ~min:(-.max) ~max (z (dtype w) g) in
        Nx.complex dt ~re:(part Nx.real) ~im:(part Nx.imag)
      in
      (match dt with Complex128 -> parts F64 | Complex64 -> parts F32)
  | Float | Signed | Unsigned | Boolean ->
      let of_float = Nx.Dtype.of_float dt in
      Nx.clamp ~min:(of_float (-.max)) ~max:(of_float max) g

let clip_by_value p ~max grads =
  validate_positive "Vega.clip_by_value" "max" max;
  Nx.Ptree.map p (fun _ g -> clip ~max g) grads

(* Loss scaling *)

module Loss_scale = struct
  type 'd t = { scale : 'd Nx.float32_t; good_steps : 'd Nx.int32_t }

  (* Static scales are marked by [good_steps = -1]: the mark is itself a tensor,
     so [adjust] can pass them through with [Nx.where] arithmetic instead of
     control flow — under [jit] the state is an ordinary input, not a trace-time
     constant. *)

  let make at v good_steps =
    {
      scale = Nx.scalar_at at Nx.float32 v;
      good_steps = Nx.scalar_at at Nx.int32 good_steps;
    }

  let static at v =
    validate_positive "Vega.Loss_scale.static" "scale" v;
    make at v (-1l)

  let dynamic ?(init = 32768.0) at =
    validate_positive "Vega.Loss_scale.dynamic" "init" init;
    make at init 0l

  let walk c { scale; good_steps } =
    let open Nx.Ptree.Walk in
    let scale = field c "scale" tensor scale in
    let good_steps = field c "good_steps" tensor good_steps in
    { scale; good_steps }

  let scale t x = Nx.mul x (Nx.cast (Nx.dtype x) t.scale)

  let unscale p t grads =
    Nx.Ptree.map p (fun _ g -> Nx.div g (Nx.cast (Nx.dtype g) t.scale)) grads

  let all_finite p t grads =
    Nx.Ptree.fold p
      (fun _ g acc -> Nx.logical_and acc (Nx.all (Nx.isfinite g)))
      grads
      (Nx.scalar_at (Nx.placement t.scale) Nx.bool true)

  let adjust ~growth_interval ~growth_factor ~backoff_factor t ~finite =
    let dynamic = Nx.greater_equal_s t.good_steps 0l in
    let good = Nx.add_s t.good_steps 1l in
    let grow = Nx.greater_equal_s good (Int32.of_int growth_interval) in
    let scale_fin = Nx.where grow (Nx.mul_s t.scale growth_factor) t.scale in
    let good_fin = Nx.where grow (Nx.zeros_like good) good in
    let scale' = Nx.where finite scale_fin (Nx.mul_s t.scale backoff_factor) in
    let good' = Nx.where finite good_fin (Nx.zeros_like good) in
    {
      scale = Nx.where dynamic scale' t.scale;
      good_steps = Nx.where dynamic good' t.good_steps;
    }

  (* The update is computed whatever [finite] is and selected leaf by leaf, so
     the step has no control flow and compiles once. Every tensor of [x] is
     selected, the optimizer's counters with the rest. *)
  let step ?(growth_interval = 2000) ?(growth_factor = 2.0)
      ?(backoff_factor = 0.5) p s t ~grads update x =
    if growth_interval <= 0 then
      invalid_argf "Vega.Loss_scale.step: expected growth_interval > 0, got %d"
        growth_interval;
    validate_positive "Vega.Loss_scale.step" "growth_factor" growth_factor;
    validate_positive "Vega.Loss_scale.step" "backoff_factor" backoff_factor;
    let grads = unscale p t grads in
    let finite = all_finite p t grads in
    let x =
      Nx.Ptree.map2 s (fun _ x' x -> Nx.where finite x' x) (update grads) x
    in
    (x, adjust ~growth_interval ~growth_factor ~backoff_factor t ~finite)
end

(* Rates *)

let rate v =
  validate_non_negative "Vega.rate" "v" v;
  fun step -> Nx.scalar_like (Nx.cast Nx.float32 step) v

(* Shared arithmetic of the steps *)

(* [zeros p x] is [x] with every tensor zeroed. *)
let zeros p x = Nx.Ptree.map p (fun _ t -> Nx.zeros_like t) x

(* [descend ~lr x d] is [x - lr * d], the rate cast to [x]'s dtype. *)
let descend ~lr x d = Nx.sub x (Nx.mul d (Nx.cast (Nx.dtype x) lr))

(* [decay_at ~default weight_decay counter] is the weight decay at the
   counter, [rate default] when the step was given none, at each compute
   dtype. *)
let decay_at ~default weight_decay counter =
  match weight_decay with
  | Some wd ->
      let r = wd counter in
      counted (fun dt -> Nx.cast dt r)
  | None ->
      let r = rate default counter in
      counted (fun dt -> Nx.cast dt r)

(* [decayed wd w decays x u] is [u + wd * x], the decay term added to the
   direction [u] of the leaf [x] at the compute dtype [w], for a leaf that
   [decays]. Another leaf's direction is [u] itself: the term is left out,
   since [u + 0 * x] is NaN at an infinite [x]. *)
let decayed wd w decays x u =
  if decays then Nx.add u (Nx.mul x (at wd w)) else u

(* [ema b m x] is [b * m + (1 - b) * x]: the moving average [m] after [x]. *)
let ema b m x = Nx.add (Nx.mul_s m b) (Nx.mul_s x (1.0 -. b))

(* The layer-wise trust ratio of LARS and LAMB: [|x| / (|u| + 1e-6)] for the
   leaf [x] and its update [u], or [1] when either norm is zero, as a scalar at
   [x]'s dtype. *)
let trust_ratio x u =
  let norm t = Nx.sqrt (Nx.sum (Nx.mul t t)) in
  let xn = norm x and un = norm u in
  Nx.where
    (Nx.logical_and (Nx.greater_s xn 0.0) (Nx.greater_s un 0.0))
    (Nx.div xn (Nx.add_s un 1e-6))
    (Nx.ones_like xn)

(* SGD and LARS *)

type ('p, 'd) sgd_state = { velocity : 'p; step : 'd Nx.int32_t }

module Sgd_state = struct
  type ('p, 'd) t = ('p, 'd) sgd_state

  let walk c st =
    let open Nx.Ptree.Walk in
    let velocity = field c "velocity" leaf st.velocity in
    let step = field c "step" tensor st.step in
    { velocity; step }
end

let sgd_ptree p = Nx.Ptree.nest (module Sgd_state) p

let velocity_init fn p params =
  { velocity = zeros p params; step = counter fn p params }

let sgd_init p params = velocity_init "Vega.sgd_init" p params

let sgd_step p ~lr ?(momentum = 0.0) st ~params ~grads =
  let fn = "Vega.sgd_step" in
  let lr = lr st.step in
  validate_unit_interval fn "momentum" momentum;
  let step = Nx.add_s st.step 1l in
  if momentum = 0.0 then
    (* Plain gradient descent: the velocity is exactly the gradient. Skipping
       the [momentum * v + g] arithmetic avoids touching (and, under [jit],
       capturing) the velocity tensors at all. *)
    let f _ _ x g _ = (descend ~lr x g, [||]) in
    let params, _ = leafwise fn p ~counter:st.step ~params ~grads [] ~f in
    (* The velocity a later step with momentum reads holds what [leafwise]'s
       would: a complex leaf's descent coordinates, the conjugate gradient. *)
    let velocity = Nx.Ptree.map p (fun _ g -> Nx.conjugate g) grads in
    (params, { velocity; step })
  else
    let f _ _ x g s =
      let v = Nx.add (Nx.mul_s s.(0) momentum) g in
      (descend ~lr x v, [| v |])
    in
    let params, parts =
      leafwise fn p ~counter:st.step ~params ~grads
        [ ("the velocity", st.velocity) ]
        ~f
    in
    (params, { velocity = parts.(0); step })

let lars_init p params = velocity_init "Vega.lars_init" p params

let lars_step p ~lr ?(momentum = 0.9) ?weight_decay ?decays
    ?(nesterov = false) st ~params ~grads =
  let fn = "Vega.lars_step" in
  let lr = lr st.step in
  let wd = decay_at ~default:0.01 weight_decay st.step in
  validate_unit_interval fn "momentum" momentum;
  let f decays w x g s =
    let u = decayed wd w decays x g in
    let u = Nx.mul u (trust_ratio x u) in
    let v = Nx.add (Nx.mul_s s.(0) momentum) u in
    let d = if nesterov then Nx.add u (Nx.mul_s v momentum) else v in
    (descend ~lr x d, [| v |])
  in
  let params, parts =
    leafwise ?decays fn p ~counter:st.step ~params ~grads
      [ ("the velocity", st.velocity) ]
      ~f
  in
  (params, { velocity = parts.(0); step = Nx.add_s st.step 1l })

(* The Adam family: Adam, AdamW, RAdam and LAMB *)

type ('p, 'd) adam_state = { mu : 'p; nu : 'p; step : 'd Nx.int32_t }

module Adam_state = struct
  type ('p, 'd) t = ('p, 'd) adam_state

  let walk c st =
    let open Nx.Ptree.Walk in
    let mu = field c "mu" leaf st.mu in
    let nu = field c "nu" leaf st.nu in
    let step = field c "step" tensor st.step in
    { mu; nu; step }
end

let adam_ptree p = Nx.Ptree.nest (module Adam_state) p

let moments_init fn p params =
  { mu = zeros p params; nu = zeros p params; step = counter fn p params }

let adam_init p params = moments_init "Vega.adam_init" p params

(* [bias_correction step b] is [1 - b^t] at the counter [t = step], as [-expm1
   (t ln b)]: [1 - pow b t] would lose the digits that matter to cancellation,
   since [b^t] is within [1e-3] of one for Adam's [b2] and small [t]. *)
let bias_correction step b =
  counted (fun dt ->
      one_minus_exp (Nx.mul_s (Nx.cast dt step) (Float.log b)))

(* One step of Adam's moments over every leaf, shared by the family: [apply x
   mu_hat nu_hat] is the new parameter from the old one and the bias-corrected
   moments. The corrections derive from the state's counter inside the step, so
   the whole step traces under jit and the state carries nothing the counter
   does not already determine. *)
let adam_update ?decays fn p ~b1 ~b2
    ~(apply :
       'w.
       bool ->
       'w wide ->
       ('w, 'd) compute ->
       ('w, 'd) compute ->
       ('w, 'd) compute ->
       ('w, 'd) compute) st
    ~params ~grads =
  let step = Nx.add_s st.step 1l in
  let c1 = bias_correction step b1 and c2 = bias_correction step b2 in
  let f decays w x g s =
    let m = ema b1 s.(0) g and n = ema b2 s.(1) (Nx.mul g g) in
    (apply decays w x (Nx.div m (at c1 w)) (Nx.div n (at c2 w)), [| m; n |])
  in
  let params, parts =
    leafwise ~squared:true ?decays fn p ~counter:st.step ~params ~grads
      [ ("mu", st.mu); ("nu", st.nu) ]
      ~f
  in
  (params, { mu = parts.(0); nu = parts.(1); step })

let validate_adam fn ~b1 ~b2 ~eps =
  validate_unit_interval fn "b1" b1;
  validate_unit_interval fn "b2" b2;
  validate_positive fn "eps" eps

(* Adam's direction from the bias-corrected moments. *)
let adam_direction ~eps mu_hat nu_hat =
  Nx.div mu_hat (Nx.add_s (Nx.sqrt nu_hat) eps)

let adam_step p ~lr ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-8) st ~params ~grads =
  let fn = "Vega.adam_step" in
  let lr = lr st.step in
  validate_adam fn ~b1 ~b2 ~eps;
  let apply _ _ x mu_hat nu_hat =
    descend ~lr x (adam_direction ~eps mu_hat nu_hat)
  in
  adam_update fn p ~b1 ~b2 ~apply st ~params ~grads

let adamw_init p params = moments_init "Vega.adamw_init" p params

let adamw_step p ~lr ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-8) ?weight_decay
    ?decays st ~params ~grads =
  let fn = "Vega.adamw_step" in
  let lr = lr st.step in
  let wd = decay_at ~default:0.01 weight_decay st.step in
  validate_adam fn ~b1 ~b2 ~eps;
  let apply decays w x mu_hat nu_hat =
    descend ~lr x (decayed wd w decays x (adam_direction ~eps mu_hat nu_hat))
  in
  adam_update ?decays fn p ~b1 ~b2 ~apply st ~params ~grads

let radam_init p params = moments_init "Vega.radam_init" p params

(* The first step at which [rho] reaches 5, where RAdam starts rectifying. It
   depends on [b2] alone, so it is found on the host in float64 and the step
   compares it with the integer counter, exactly whatever the leaves' dtype.
   [rho] rises with [t] towards [rho_inf], so the search ends; it never reaches
   5 when [rho_inf] does not exceed it. *)
let radam_switch b2 =
  let rho_inf = (2.0 /. (1.0 -. b2)) -. 1.0 in
  let rho t =
    let x = t *. Float.log b2 in
    rho_inf -. (2.0 *. t *. Float.exp x /. -.Float.expm1 x)
  in
  let rec first t =
    if t >= Int32.to_int Int32.max_int then Int32.max_int
    else if rho (float t) >= 5.0 then Int32.of_int t
    else first (t + 1)
  in
  if rho_inf <= 5.0 then Int32.max_int else first 1

let radam_step p ~lr ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-8) st ~params ~grads =
  let fn = "Vega.radam_step" in
  let lr = lr st.step in
  validate_adam fn ~b1 ~b2 ~eps;
  let rho_inf = (2.0 /. (1.0 -. b2)) -. 1.0 in
  let step = Nx.add_s st.step 1l in
  let rectified = Nx.greater_equal_s step (radam_switch b2) in
  let r =
    counted (fun dt ->
        let t = Nx.cast dt step in
        let x = Nx.mul_s t (Float.log b2) in
        let tail =
          Nx.div (Nx.mul (Nx.mul_s t 2.0) (Nx.exp x)) (one_minus_exp x)
        in
        let rho = Nx.rsub_s rho_inf tail in
        let k = rho_inf /. ((rho_inf -. 4.0) *. (rho_inf -. 2.0)) in
        Nx.sqrt
          (Nx.div
             (Nx.mul_s (Nx.mul (Nx.sub_s rho 4.0) (Nx.sub_s rho 2.0)) k)
             rho))
  in
  let apply _ w x mu_hat nu_hat =
    let rectification = Nx.mul (at r w) (adam_direction ~eps mu_hat nu_hat) in
    descend ~lr x (Nx.where rectified rectification mu_hat)
  in
  adam_update fn p ~b1 ~b2 ~apply st ~params ~grads

let lamb_init p params = moments_init "Vega.lamb_init" p params

let lamb_step p ~lr ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-8) ?weight_decay
    ?decays st ~params ~grads =
  let fn = "Vega.lamb_step" in
  let lr = lr st.step in
  let wd = decay_at ~default:0.01 weight_decay st.step in
  validate_adam fn ~b1 ~b2 ~eps;
  let apply decays w x mu_hat nu_hat =
    let u = decayed wd w decays x (adam_direction ~eps mu_hat nu_hat) in
    descend ~lr x (Nx.mul u (trust_ratio x u))
  in
  adam_update ?decays fn p ~b1 ~b2 ~apply st ~params ~grads

(* RMSprop *)

type ('p, 'd) rmsprop_state = { nu : 'p; velocity : 'p; step : 'd Nx.int32_t }

module Rmsprop_state = struct
  type ('p, 'd) t = ('p, 'd) rmsprop_state

  let walk c st =
    let open Nx.Ptree.Walk in
    let nu = field c "nu" leaf st.nu in
    let velocity = field c "velocity" leaf st.velocity in
    let step = field c "step" tensor st.step in
    { nu; velocity; step }
end

let rmsprop_ptree p = Nx.Ptree.nest (module Rmsprop_state) p

let rmsprop_init p params =
  {
    nu = zeros p params;
    velocity = zeros p params;
    step = counter "Vega.rmsprop_init" p params;
  }

let rmsprop_step p ~lr ?(decay = 0.9) ?(eps = 1e-8) ?(momentum = 0.0) st ~params
    ~grads =
  let fn = "Vega.rmsprop_step" in
  let lr = lr st.step in
  validate_unit_interval fn "decay" decay;
  validate_positive fn "eps" eps;
  validate_unit_interval fn "momentum" momentum;
  let f _ _ x g s =
    let nu = ema decay s.(0) (Nx.mul g g) in
    let u = Nx.div g (Nx.add_s (Nx.sqrt nu) eps) in
    let v =
      if momentum = 0.0 then u else Nx.add (Nx.mul_s s.(1) momentum) u
    in
    (descend ~lr x v, [| nu; v |])
  in
  let parts = [ ("nu", st.nu); ("the velocity", st.velocity) ] in
  let params, parts =
    leafwise ~squared:true fn p ~counter:st.step ~params ~grads parts ~f
  in
  (params, { nu = parts.(0); velocity = parts.(1); step = Nx.add_s st.step 1l })

(* Adagrad *)

type ('p, 'd) adagrad_state = { sum_of_squares : 'p; step : 'd Nx.int32_t }

module Adagrad_state = struct
  type ('p, 'd) t = ('p, 'd) adagrad_state

  let walk c st =
    let open Nx.Ptree.Walk in
    let sum_of_squares = field c "sum_of_squares" leaf st.sum_of_squares in
    let step = field c "step" tensor st.step in
    { sum_of_squares; step }
end

let adagrad_ptree p = Nx.Ptree.nest (module Adagrad_state) p

let adagrad_init p params =
  {
    sum_of_squares = zeros p params;
    step = counter "Vega.adagrad_init" p params;
  }

let adagrad_step p ~lr ?(eps = 1e-8) st ~params ~grads =
  let fn = "Vega.adagrad_step" in
  let lr = lr st.step in
  validate_positive fn "eps" eps;
  let f _ _ x g s =
    let s = Nx.add s.(0) (Nx.mul g g) in
    let d = Nx.div g (Nx.add_s (Nx.sqrt s) eps) in
    (descend ~lr x d, [| s |])
  in
  let parts = [ ("sum_of_squares", st.sum_of_squares) ] in
  let params, parts =
    leafwise ~squared:true fn p ~counter:st.step ~params ~grads parts ~f
  in
  (params, { sum_of_squares = parts.(0); step = Nx.add_s st.step 1l })

(* Adan *)

type ('p, 'd) adan_state = {
  mu : 'p;
  delta : 'p;
  nu : 'p;
  prev_grads : 'p;
  step : 'd Nx.int32_t;
}

module Adan_state = struct
  type ('p, 'd) t = ('p, 'd) adan_state

  let walk c st =
    let open Nx.Ptree.Walk in
    let mu = field c "mu" leaf st.mu in
    let delta = field c "delta" leaf st.delta in
    let nu = field c "nu" leaf st.nu in
    let prev_grads = field c "prev_grads" leaf st.prev_grads in
    let step = field c "step" tensor st.step in
    { mu; delta; nu; prev_grads; step }
end

let adan_ptree p = Nx.Ptree.nest (module Adan_state) p

let adan_init p params =
  {
    mu = zeros p params;
    delta = zeros p params;
    nu = zeros p params;
    prev_grads = zeros p params;
    step = counter "Vega.adan_init" p params;
  }

let adan_step p ~lr ?(b1 = 0.98) ?(b2 = 0.92) ?(b3 = 0.99) ?(eps = 1e-8)
    ?weight_decay ?decays st ~params ~grads =
  let fn = "Vega.adan_step" in
  let lr = lr st.step in
  let wd = decay_at ~default:0.02 weight_decay st.step in
  validate_unit_interval fn "b1" b1;
  validate_unit_interval fn "b2" b2;
  validate_unit_interval fn "b3" b3;
  validate_positive fn "eps" eps;
  let f decays w x g s =
    let dg = Nx.sub g s.(3) in
    let mu = ema b1 s.(0) g and delta = ema b2 s.(1) dg in
    let ahead = Nx.add g (Nx.mul_s dg b2) in
    let nu = ema b3 s.(2) (Nx.mul ahead ahead) in
    let d =
      Nx.div
        (Nx.add mu (Nx.mul_s delta b2))
        (Nx.add_s (Nx.sqrt nu) eps)
    in
    (descend ~lr x (decayed wd w decays x d), [| mu; delta; nu; g |])
  in
  let parts =
    [
      ("mu", st.mu);
      ("delta", st.delta);
      ("nu", st.nu);
      ("prev_grads", st.prev_grads);
    ]
  in
  let params, parts =
    leafwise ~squared:true ?decays fn p ~counter:st.step ~params ~grads parts
      ~f
  in
  let step = Nx.add_s st.step 1l in
  ( params,
    {
      mu = parts.(0);
      delta = parts.(1);
      nu = parts.(2);
      prev_grads = parts.(3);
      step;
    } )

(* Lion *)

type ('p, 'd) lion_state = { mu : 'p; step : 'd Nx.int32_t }

module Lion_state = struct
  type ('p, 'd) t = ('p, 'd) lion_state

  let walk c st =
    let open Nx.Ptree.Walk in
    let mu = field c "mu" leaf st.mu in
    let step = field c "step" tensor st.step in
    { mu; step }
end

let lion_ptree p = Nx.Ptree.nest (module Lion_state) p
let lion_init p params =
  { mu = zeros p params; step = counter "Vega.lion_init" p params }

let lion_step p ~lr ?(b1 = 0.9) ?(b2 = 0.99) st ~params ~grads =
  let fn = "Vega.lion_step" in
  let lr = lr st.step in
  validate_unit_interval fn "b1" b1;
  validate_unit_interval fn "b2" b2;
  let f _ _ x g s =
    (descend ~lr x (Nx.sign (ema b1 s.(0) g)), [| ema b2 s.(0) g |])
  in
  let params, parts =
    leafwise fn p ~counter:st.step ~params ~grads [ ("mu", st.mu) ] ~f
  in
  (params, { mu = parts.(0); step = Nx.add_s st.step 1l })

(* Adafactor *)

type ('p, 'd) adafactor_state = {
  nu_row : 'p;
  nu_col : 'p;
  nu : 'p;
  step : 'd Nx.int32_t;
}

module Adafactor_state = struct
  type ('p, 'd) t = ('p, 'd) adafactor_state

  let walk c st =
    let open Nx.Ptree.Walk in
    let nu_row = field c "nu_row" leaf st.nu_row in
    let nu_col = field c "nu_col" leaf st.nu_col in
    let nu = field c "nu" leaf st.nu in
    let step = field c "step" tensor st.step in
    { nu_row; nu_col; nu; step }
end

let adafactor_ptree p = Nx.Ptree.nest (module Adafactor_state) p

(* A factored leaf keeps its statistics in [nu_row] and [nu_col], any other in
   [nu], and the parts a leaf does not use hold a scalar zero. So the state
   records which leaves are factored: those whose [nu] has fewer axes than the
   leaf itself. A leaf the step does not update is never factored and holds
   zeros of its own shape in every part, as it does in every other state. *)
let adafactor_init p ?(factored = true) params =
  let factors x = factored && updates x && Nx.ndim x >= 2 in
  let unused x =
    if updates x then Nx.scalar_like x (Nx.Dtype.zero (Nx.dtype x))
    else Nx.zeros_like x
  in
  (* A statistic is the mean of zeros over its axis, so it lies where the
     step's means of squared gradients will. *)
  let factor axis x =
    if factors x then
      Nx.mean ~axes:[ Nx.ndim x - axis ] ~keepdims:true (Nx.zeros_like x)
    else unused x
  in
  {
    nu_row = Nx.Ptree.map p (fun _ x -> factor 1 x) params;
    nu_col = Nx.Ptree.map p (fun _ x -> factor 2 x) params;
    nu =
      Nx.Ptree.map p
        (fun _ x -> if factors x then unused x else Nx.zeros_like x)
        params;
    step = counter "Vega.adafactor_init" p params;
  }

let adafactor_step p ~lr ?(decay_rate = 0.8) ?(eps = 1e-30)
    ?(clipping_threshold = 1.0) st ~params ~grads =
  let fn = "Vega.adafactor_step" in
  let lr = lr st.step in
  validate_positive fn "decay_rate" decay_rate;
  validate_positive fn "eps" eps;
  validate_positive fn "clipping_threshold" clipping_threshold;
  let step = Nx.add_s st.step 1l in
  (* An average keeps [1 - t^-decay_rate] of its old value and takes
     [t^-decay_rate] of the new one, [exp e] for [e = -decay_rate log t]. *)
  let exponent dt =
    Nx.mul_s (Nx.log (Nx.cast dt step)) (-.decay_rate)
  in
  let keep = counted (fun dt -> one_minus_exp (exponent dt)) in
  let fresh = counted (fun dt -> Nx.exp (exponent dt)) in
  let f _ w x g s =
    let average m v = Nx.add (Nx.mul m (at keep w)) (Nx.mul v (at fresh w)) in
    let g2 = Nx.mul g g in
    let n = Nx.ndim x in
    let u, s =
      if Nx.ndim s.(2) < n then
        let row = average s.(0) (Nx.mean ~axes:[ n - 1 ] ~keepdims:true g2) in
        let col = average s.(1) (Nx.mean ~axes:[ n - 2 ] ~keepdims:true g2) in
        let row_mean = Nx.mean ~axes:[ n - 2 ] ~keepdims:true row in
        let nu = Nx.div (Nx.mul row col) (Nx.add_s row_mean eps) in
        (Nx.div g (Nx.add_s (Nx.sqrt nu) eps), [| row; col; s.(2) |])
      else
        let nu = average s.(2) g2 in
        (Nx.div g (Nx.add_s (Nx.sqrt nu) eps), [| s.(0); s.(1); nu |])
    in
    let rms = Nx.sqrt (Nx.mean (Nx.mul u u)) in
    let clip =
      Nx.minimum_s (Nx.rdiv_s clipping_threshold rms) 1.0
    in
    (descend ~lr x (Nx.mul u clip), s)
  in
  let parts = [ ("nu_row", st.nu_row); ("nu_col", st.nu_col); ("nu", st.nu) ] in
  let params, parts =
    leafwise ~squared:true fn p ~counter:st.step ~params ~grads parts ~f
  in
  (params, { nu_row = parts.(0); nu_col = parts.(1); nu = parts.(2); step })

(* L-BFGS *)

type ('p, 'd) lbfgs_state = {
  params : 'p;
  grads : 'p;
  s : 'p;
  y : 'p;
  step : 'd Nx.int32_t;
}

module Lbfgs_state = struct
  type ('p, 'd) t = ('p, 'd) lbfgs_state

  let walk c st =
    let open Nx.Ptree.Walk in
    let params = field c "params" leaf st.params in
    let grads = field c "grads" leaf st.grads in
    let s = field c "s" leaf st.s in
    let y = field c "y" leaf st.y in
    let step = field c "step" tensor st.step in
    { params; grads; s; y; step }
end

let lbfgs_ptree p = Nx.Ptree.nest (module Lbfgs_state) p

let lbfgs_init p ?(history = 10) params =
  if history < 1 then
    invalid_argf "Vega.lbfgs_init: expected history >= 1, got %d" history;
  let memory () =
    Nx.Ptree.map p
      (fun _ leaf ->
        let shape = Array.append [| history |] (Nx.shape leaf) in
        Nx.zeros_like (Nx.broadcast_to shape (Nx.unsqueeze ~axes:[ 0 ] leaf)))
      params
  in
  {
    params;
    grads = zeros p params;
    s = memory ();
    y = memory ();
    step = counter "Vega.lbfgs_init" p params;
  }

(* A leaf's memory is its pairs stacked along axis 0, newest first: [slot i]
   views the [i]-th, [push] puts a new one on top and drops the oldest, [top]
   replaces the newest. *)
let slot i memory = Nx.get [ i ] memory

let push x memory =
  let n = (Nx.shape memory).(0) in
  let x = Nx.unsqueeze ~axes:[ 0 ] x in
  if n = 1 then x
  else Nx.concatenate ~axis:0 [ x; Nx.slice [ Nx.R (0, n - 1) ] memory ]

let top x memory =
  let n = (Nx.shape memory).(0) in
  let x = Nx.unsqueeze ~axes:[ 0 ] x in
  if n = 1 then x
  else Nx.concatenate ~axis:0 [ x; Nx.slice [ Nx.R (1, n) ] memory ]

let is_float64 (type a b d) (t : (a, b, d) Nx.t) =
  match Nx.dtype t with Float64 | Complex128 -> true | _ -> false

(* L-BFGS computes each float leaf at float32, or at float64 for a float64
   leaf, and rounds once, when it stores: its vectors are lists of leaves of
   several dtypes, walked by [leaves ()]. [widen] casts a leaf to the dtype it
   computes at and [stored draw k like x] stores [x] back at [like]'s, as store
   [k] of its leaf. A complex leaf computes as it is, its parts at float32 or
   float64. *)
let widen (Nx.P x) =
  match (Nx.Dtype.kind (Nx.dtype x), Nx.dtype x) with
  | Float, (Float32 | Float64) -> Nx.P x
  | Float, _ -> Nx.P (Nx.cast Nx.float32 x)
  | (Complex | Signed | Unsigned | Boolean), _ -> Nx.P x

let stored draw k (Nx.P like) (Nx.P x) = Nx.P (store draw k like x)

(* [into f (P m) (P x)] is [f x m] for [x] stored at the dtype of the memory
   [m], which it already has. *)
let into
    (f : 'v 's. ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t)
    (Nx.P m : 'd Nx.packed) (Nx.P x : 'd Nx.packed) =
  match Nx.Dtype.equal_witness (Nx.dtype m) (Nx.dtype x) with
  | Some Equal -> Nx.P (f x m)
  | None -> Nx.P (f (Nx.cast (Nx.dtype m) x) m)

let leaves () =
  let walk c xs =
    let open Nx.Ptree.Walk in
    list (fun c (Nx.P x) -> Nx.P (tensor c x)) c xs
  in
  Nx.Ptree.v walk

(* The newest pair's step is the move the last call made; this call completes
   its gradient difference, [grads - st.grads]. The direction is the two-loop
   recursion (Nocedal, 1980) over the pairs with [rho = 1 / (y . s)], a pair
   whose curvature is not positive (an empty slot, a rejected pair) entering
   neither loop, and the initial inverse Hessian scaled by [(s . y) / (y . y)]
   of the newest pair. Its scalars are at the compute dtype every float leaf
   shares, so the step traces under jit. *)
let lbfgs_step p ~lr grads st =
  let fn = "Vega.lbfgs_step" in
  let lr = lr st.step in
  (* Every value has the parameters' skeleton and dtypes, checked before the
     leaves are widened, which would hide a part of another dtype. *)
  let skeleton = snd (Nx.Ptree.flatten p st.params) in
  let parts =
    List.map
      (fun (name, x) -> (name, ref (aligned fn p skeleton name x)))
      [ ("the gradients", grads); ("grads", st.grads); ("s", st.s); ("y", st.y) ]
  in
  Nx.Ptree.fold p
    (fun path x () ->
      List.iter (fun (name, rest) -> ignore (take fn name path x rest)) parts)
    st.params ();
  let float64 =
    Nx.Ptree.fold p
      (fun _ t acc -> acc && ((not (updates t)) || is_float64 t))
      st.params true
  in
  let flat = leaves () in
  let wide x = List.map widen (fst (Nx.Ptree.flatten p x)) in
  (* The recursion runs over the real coordinates, two per complex element,
     whose gradient is the conjugate of Rune's ([leafwise]); [Nx.Ptree.dot]
     pairs complex arrays part by part. The gradient difference is of the
     coordinates' gradients: conjugating after subtracting would turn a part
     [-0 - -0 = 0] into [-0]. *)
  let conjugate x = Nx.Ptree.map flat (fun _ g -> Nx.conjugate g) (wide x) in
  let difference =
    Nx.Ptree.map2 flat (fun _ a b -> if updates a then Nx.sub a b else a)
  in
  let g = conjugate grads in
  let diff = difference g (conjugate st.grads) in
  let y = Nx.Ptree.map2 flat (fun _ d m -> top d m) diff (wide st.y) in
  let s = wide st.s in
  let history =
    match s with
    | Nx.P t :: _ -> Nx.dim 0 t
    | [] -> invalid_arg (fn ^ ": the parameters hold no array")
  in
  let run (type w) (w : w wide) =
    let dt = dtype w in
    let dot = Nx.Ptree.dot flat dt in
    let pair i =
      let s = Nx.Ptree.map flat (fun _ m -> slot i m) s
      and y = Nx.Ptree.map flat (fun _ m -> slot i m) y in
      let ys = dot y s in
      (s, y, Nx.where (Nx.greater_s ys 0.0) (Nx.recip ys) (Nx.zeros_like ys))
    in
    let pairs = Array.init history pair in
    let alphas = Array.map (fun (_, _, rho) -> Nx.zeros_like rho) pairs in
    let q = ref g in
    for i = 0 to history - 1 do
      let s, y, rho = pairs.(i) in
      alphas.(i) <- Nx.mul rho (dot s !q);
      q := Nx.Ptree.axpy flat (Nx.neg alphas.(i)) y !q
    done;
    let _, y0, rho0 = pairs.(0) in
    let gamma =
      Nx.where (Nx.greater_s rho0 0.0)
        (Nx.recip (Nx.mul rho0 (dot y0 y0)))
        (Nx.ones_like rho0)
    in
    let r = ref (Nx.Ptree.scale flat gamma !q) in
    for i = history - 1 downto 0 do
      let s, y, rho = pairs.(i) in
      let beta = Nx.mul rho (dot y !r) in
      r := Nx.Ptree.axpy flat (Nx.sub alphas.(i) beta) s !r
    done;
    !r
  in
  let r = if float64 then run F64 else run F32 in
  (* [-lr * r], a complex [r] part by part: a complex product by the rate
     would add the product of a zero part to the other part's, losing a sign
     of zero the coordinates keep. *)
  let move = Nx.Ptree.scale flat (Nx.neg lr) r in
  let params =
    Nx.Ptree.map2 flat
      (fun _ x d -> if updates x then Nx.add x d else x)
      (wide st.params) move
  in
  (* Each leaf stores its new point, its move as the newest [s] and its
     gradient difference as the newest [y], its stores 0, 1 and 2; the older
     pairs, already at its dtype, move down unrounded. A leaf that computes at
     its own dtype stores the memory the recursion read. *)
  let leaves x = Array.of_list (fst (Nx.Ptree.flatten p x)) in
  let paths = Nx.Ptree.fold p (fun path _ acc -> path :: acc) st.params [] in
  let base = base st.step in
  let draw = Array.of_list (List.rev_map (draws base) paths) in
  let like = leaves st.params and s = leaves st.s in
  let y_half = leaves st.y and y = Array.of_list y and g = Array.of_list g in
  let point = Array.of_list params and move = Array.of_list move in
  let diff = Array.of_list diff in
  let put k xs i = stored draw.(i) k like.(i) xs.(i) in
  let rebuild like f =
    Nx.Ptree.rebuild p ~like (List.init (Array.length draw) f)
  in
  let pushed x m = push (Nx.zeros_like x) m in
  let newest i =
    let (Nx.P l) = like.(i) in
    match Nx.dtype l with
    | Float16 | Bfloat16 ->
        into (fun x m -> pushed x (top x m)) y_half.(i) (put 2 diff i)
    | _ -> stored draw.(i) 2 like.(i) (into pushed y.(i) g.(i))
  in
  {
    params = rebuild st.params (put 0 point);
    grads;
    s = rebuild st.s (fun i -> into push s.(i) (put 1 move i));
    y = rebuild st.y newest;
    step = Nx.add_s st.step 1l;
  }
