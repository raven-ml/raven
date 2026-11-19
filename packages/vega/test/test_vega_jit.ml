(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Every step compiles with its state as a donated argument: the parameters and
   the optimizer state go into [Rune.jit] and come back out, and the compiled
   trajectory matches the eager one leaf for leaf, the counter included. *)

open Windtrap

let steps = 6

(* A float32 matrix and vector: Adafactor factors the one and not the other. *)
module Wb = struct
  type t = { w : Nx.host Nx.float32_t; b : Nx.host Nx.float32_t }

  let walk c { w; b } =
    let open Nx.Ptree.Walk in
    let w = field c "w" tensor w in
    let b = field c "b" tensor b in
    { w; b }

  let ptree : (t, Nx.host) Nx.Ptree.t = Nx.Ptree.v walk
end

let fill i n = Array.init n (fun j -> sin (float_of_int ((i * 7919) + j)))

let wb i =
  {
    Wb.w = Nx.create Nx.float32 [| 4; 3 |] (fill i 12);
    b = Nx.create Nx.float32 [| 3 |] (fill (i + 1) 3);
  }

let target = lazy (wb 3)

let grads (x : Wb.t) =
  let t = Lazy.force target in
  { Wb.w = Nx.mul_s (Nx.sub x.w t.w) 2.0; b = Nx.mul_s (Nx.sub x.b t.b) 2.0 }

let floats x = Nx.to_array (Nx.cast Nx.float64 x)

let compiles state ~init ~step () =
  let both = Nx.Ptree.pair Wb.ptree (state Wb.ptree) in
  let body (params, st) = step st ~params ~grads:(grads params) in
  let run f =
    let rec go k x = if k = 0 then x else go (k - 1) (f x) in
    let params = wb 1 in
    go steps (params, init Wb.ptree params)
  in
  let eager = run body in
  let compiled =
    let f = Rune.jit Nx.Ptree.(both @-> returns both) body in
    run (fun x -> f (Nx.Ptree.donate both x))
  in
  ignore
    (Nx.Ptree.map2 both
       (fun path x y ->
         equal
           ~msg:(Format.asprintf "%a" Nx.Ptree.Path.pp path)
           (array (float 1e-5)) (floats x) (floats y);
         x)
       eager compiled)

let lr = Vega.rate 0.01

let tests =
  [
    test "sgd"
      (compiles Vega.sgd_ptree ~init:Vega.sgd_init ~step:(fun st ->
           Vega.sgd_step Wb.ptree ~lr ~momentum:0.9 st));
    test "lars"
      (compiles Vega.sgd_ptree ~init:Vega.lars_init ~step:(fun st ->
           Vega.lars_step Wb.ptree ~lr ~nesterov:true st));
    test "adam"
      (compiles Vega.adam_ptree ~init:Vega.adam_init ~step:(fun st ->
           Vega.adam_step Wb.ptree ~lr st));
    (* [b2 = 0.99] crosses to rectified steps at the sixth. *)
    test "radam"
      (compiles Vega.adam_ptree ~init:Vega.radam_init ~step:(fun st ->
           Vega.radam_step Wb.ptree ~lr ~b2:0.99 st));
    (* The decay at the counter is computed inside the program. *)
    test "adamw under a scheduled decay"
      (compiles Vega.adam_ptree ~init:Vega.adamw_init ~step:(fun st ->
           let weight_decay =
             Vega.Schedule.cosine_decay ~init_value:0.4 ~decay_steps:4 ()
           in
           Vega.adamw_step Wb.ptree ~lr ~weight_decay st));
    (* The rule's answers are constants of the program. *)
    test "lamb under a rule that decays [w] alone"
      (compiles Vega.adam_ptree ~init:Vega.lamb_init ~step:(fun st ->
           let decays path _ = Nx.Ptree.Path.to_string path = "w" in
           let weight_decay = Vega.rate 0.1 in
           Vega.lamb_step Wb.ptree ~lr ~weight_decay ~decays st));
    test "lamb"
      (compiles Vega.adam_ptree ~init:Vega.lamb_init ~step:(fun st ->
           Vega.lamb_step Wb.ptree ~lr st));
    (* The rule's answers are constants of the program. *)
    test "lars under a rule that adapts [w] alone"
      (compiles Vega.sgd_ptree ~init:Vega.lars_init ~step:(fun st ->
           let adapts path _ = Nx.Ptree.Path.to_string path = "w" in
           Vega.lars_step Wb.ptree ~lr ~adapts st));
    test "rmsprop"
      (compiles Vega.rmsprop_ptree ~init:Vega.rmsprop_init ~step:(fun st ->
           Vega.rmsprop_step Wb.ptree ~lr ~momentum:0.9 st));
    test "adagrad"
      (compiles Vega.adagrad_ptree ~init:Vega.adagrad_init ~step:(fun st ->
           Vega.adagrad_step Wb.ptree ~lr st));
    test "adan"
      (compiles Vega.adan_ptree ~init:Vega.adan_init ~step:(fun st ->
           Vega.adan_step Wb.ptree ~lr st));
    test "lion"
      (compiles Vega.lion_ptree ~init:Vega.lion_init ~step:(fun st ->
           Vega.lion_step Wb.ptree ~lr st));
    test "adafactor"
      (compiles Vega.adafactor_ptree
         ~init:(fun p x -> Vega.adafactor_init p x)
         ~step:(fun st ->
           let lr step =
             Nx.mul_s (Nx.rsqrt (Nx.cast Nx.float32 (Nx.add_s step 1l))) 1e-2
           in
           Vega.adafactor_step Wb.ptree ~lr st));
    (* Six steps fill a history of three and evict from it. *)
    test "lbfgs"
      (compiles Vega.lbfgs_ptree
         ~init:(fun p x -> Vega.lbfgs_init p ~history:3 x)
         ~step:(fun st ~params ~grads ->
           let st = Vega.lbfgs_step Wb.ptree ~lr grads { st with params } in
           (st.params, st)));
  ]

(* A loss-scaled Adam step compiles once and follows the eager steps, through a
   step whose gradients overflow: [poison] is added to every gradient, [0.] but
   at the third step, where it is an infinity. *)
let test_loss_scaled_compiles () =
  let both = Nx.Ptree.pair Wb.ptree (Vega.adam_ptree Wb.ptree) in
  let all = Nx.Ptree.pair both (Nx.Ptree.v Vega.Loss_scale.walk) in
  let body poison ((params, st), ls) =
    let g = grads params in
    let scaled x = Nx.add (Vega.Loss_scale.scale ls x) poison in
    let grads = { Wb.w = scaled g.w; b = scaled g.b } in
    Vega.Loss_scale.step ~growth_interval:2 Wb.ptree both ls ~grads
      (fun grads -> Vega.adam_step Wb.ptree ~lr st ~params ~grads)
      (params, st)
  in
  let run f =
    let rec go k x =
      if k = steps then x
      else
        let bad = if k = 2 then Float.infinity else 0.0 in
        go (k + 1) (f (Nx.scalar Nx.float32 bad) x)
    in
    let params = wb 1 in
    go 0
      ( (params, Vega.adam_init Wb.ptree params),
        Vega.Loss_scale.dynamic ~init:1024.0 Nx.Host.on )
  in
  let eager = run body in
  let compiled =
    let f = Rune.jit Nx.Ptree.(tensor @-> all @-> returns all) body in
    run (fun poison x -> f poison (Nx.Ptree.donate all x))
  in
  equal ~msg:"one backoff and two growths" float_exact 2048.0
    (Nx.item [] (snd compiled).Vega.Loss_scale.scale);
  ignore
    (Nx.Ptree.map2 all
       (fun path x y ->
         equal
           ~msg:(Format.asprintf "%a" Nx.Ptree.Path.pp path)
           (array (float 1e-5)) (floats x) (floats y);
         x)
       eager compiled)

(* RAdam at [b2 = 0.9999], compiled in float32: the rectified steps 6 to 12 move
   as far as eager float64 ones to within 1.5e-4 (2e-5 eagerly; the float32 [1 -
   pow b2 t] was off by 3.5%). The remaining error is float32's [rho = rho_inf -
   tail], a difference of two terms near 19999 whose absolute error is about
   2e-3; compiled code rounds the rest of the rectification differently ([1 -
   b^t] itself agrees with float64 to 1.5e-7 compiled). *)
let test_radam_compiled_rectification () =
  let target64 = Nx.create Nx.float64 [| 3 |] [| 1.0; -2.0; 0.5 |] in
  let target32 = Nx.cast Nx.float32 target64 in
  let p = Nx.Ptree.tensor in
  let step target (params, st) =
    let grads = Nx.mul_s (Nx.sub params target) 2.0 in
    Vega.radam_step p ~lr:(Vega.rate 0.1) ~b2:0.9999 st ~params ~grads
  in
  let path f params =
    let rec go k x acc =
      if k = 12 then List.rev acc
      else
        let x = f x in
        go (k + 1) x (Nx.to_array (Nx.cast Nx.float64 (fst x)) :: acc)
    in
    let path = go 0 (params, Vega.radam_init p params) [] in
    Array.map2 ( -. ) (List.nth path 11) (List.nth path 5)
  in
  let both = Nx.Ptree.pair p (Vega.adam_ptree p) in
  let compiled =
    let f = Rune.jit Nx.Ptree.(both @-> returns both) (step target32) in
    path (fun x -> f (Nx.Ptree.donate both x)) (Nx.zeros Nx.float32 [| 3 |])
  in
  let exact = path (step target64) (Nx.zeros Nx.float64 [| 3 |]) in
  equal (array (float_rel ~rel:3e-4 ~abs:0.0)) exact compiled

(* A compiled polynomial decay takes eager's values through the decay and after
   it, where it stays at [end_value]: its base is exactly zero there. *)
let test_polynomial_decay_compiles () =
  List.iter
    (fun power ->
      let s =
        Vega.Schedule.polynomial_decay ~init_value:1.0 ~end_value:0.1
          ~decay_steps:10 ~power ()
      in
      let g = Rune.jit' s in
      for k = 0 to 20 do
        let step = Nx.scalar Nx.int32 (Int32.of_int k) in
        equal
          ~msg:(Printf.sprintf "power %g, step %d" power k)
          (float 1e-6)
          (Nx.item [] (s step))
          (Nx.item [] (g step))
      done)
    [ 0.5; 0.7; 1.0; 2.0 ]

(* One step on bfloat16 leaves, eager and compiled from equal inputs, at a
   counter past zero. The inputs make every float32 value the step computes
   exact, whatever order the compiled program computes in: the parameters,
   gradients and averages lie on bfloat16's grid in [1, 2), the rate is a
   power of two, and the averages' weights are a half. So the stores differ
   only if the draws do, and the draws must be equal. *)
let half_n = 4096

let grid i = 1.0 +. (Float.of_int (i mod 128) /. 128.0)
let half_leaf f = Nx.create Nx.bfloat16 [| half_n |] (Array.init half_n f)

let half_agrees state ~init ~step () =
  let p = Nx.Ptree.tensor in
  let both = Nx.Ptree.pair p (state p) in
  let params = half_leaf grid and grads = half_leaf (fun i -> grid (i * 7)) in
  let body (params, st) = step st ~params ~grads in
  let x = (params, init p params) in
  let eager = body x in
  let compiled = Rune.jit Nx.Ptree.(both @-> returns both) body x in
  let found = ref [] in
  let compare (type v r) path (x : (v, r, _) Nx.t) y =
    let bits x = Nx.to_array (Nx.cast Nx.int64 (Nx.bitcast Nx.uint16 x)) in
    let ints x = Nx.to_array (Nx.cast Nx.int64 x) in
    let a, b =
      match Nx.dtype x with
      | Bfloat16 -> (bits x, bits y)
      | _ -> (ints x, ints y)
    in
    Array.iteri
      (fun i a ->
        if a <> b.(i) then
          found :=
            Format.asprintf "%a[%d]: %Lx and %Lx" Nx.Ptree.Path.pp path i a
              b.(i)
            :: !found)
      a;
    x
  in
  ignore (Nx.Ptree.map2 both compare eager compiled);
  equal (list string) [] (List.rev !found)

(* The states start at the counter 7. *)
let seven = Nx.scalar Nx.int32 7l
let lr2 = Vega.rate 0x1p-9

let half_tests =
  let p = Nx.Ptree.tensor in
  [
    test "sgd"
      (half_agrees Vega.sgd_ptree
         ~init:(fun p x -> { (Vega.sgd_init p x) with step = seven })
         ~step:(fun st -> Vega.sgd_step p ~lr:lr2 st));
    test "lion"
      (half_agrees Vega.lion_ptree
         ~init:(fun _ _ ->
           { Vega.mu = half_leaf (fun i -> grid (i * 3)); step = seven })
         ~step:(fun st -> Vega.lion_step p ~lr:lr2 ~b1:0.5 ~b2:0.5 st));
  ]


(* Schedules' numbers are data of their program. [Literal] is vega's
   schedules as they were, each number written into the program as a
   literal. The data form gives its bits at every counter, eagerly and
   compiled (compiled against eager is §numerics', not bit for bit), while
   one program serves every set of numbers of a structure. *)
type t = Nx.host Vega.Schedule.t

module type Forms = sig
  val rate : float -> t
  val linear : init_value:float -> end_value:float -> steps:int -> t

  val cosine_decay :
    init_value:float -> decay_steps:int -> ?alpha:float -> unit -> t

  val exponential_decay :
    init_value:float -> decay_rate:float -> decay_steps:int -> t

  val polynomial_decay :
    init_value:float ->
    end_value:float ->
    decay_steps:int ->
    ?power:float ->
    unit ->
    t

  val warmup_cosine :
    init_value:float -> peak_value:float -> warmup_steps:int -> t

  val warmup_cosine_decay :
    init_value:float ->
    peak_value:float ->
    warmup_steps:int ->
    decay_steps:int ->
    ?end_value:float ->
    unit ->
    t

  val cosine_decay_restarts :
    init_value:float ->
    decay_steps:int ->
    ?t_mul:float ->
    ?m_mul:float ->
    ?alpha:float ->
    unit ->
    t

  val one_cycle :
    max_value:float ->
    total_steps:int ->
    ?div_factor:float ->
    ?final_div_factor:float ->
    ?pct_start:float ->
    unit ->
    t

  val piecewise_constant : boundaries:int list -> values:float list -> t
  val join : (int * t) list -> t
end

module Data : Forms = struct
  include Vega.Schedule

  let rate = Vega.rate
end

module Literal : Forms = struct
  let f32 = Nx.float32
  let to_f32 step = Nx.cast f32 step

  (* Cosine annealing factor: 1 -> 0 as ratio goes 0 -> 1. *)
  let cosine_factor ratio =
    Nx.mul_s (Nx.add_s (Nx.cos (Nx.mul_s ratio Float.pi)) 1.0) 0.5

  (* step / steps, clamped to 1. *)
  let clamp_ratio ~steps s =
    Nx.div_s
      (Nx.minimum_s s (float_of_int steps))
      (float_of_int steps)

  let linear ~init_value ~end_value ~steps =
    if steps <= 0 then invalid_arg "Schedule.linear: steps must be positive";
    fun step ->
      let ratio = clamp_ratio ~steps (to_f32 step) in
      Nx.add_s (Nx.mul_s ratio (end_value -. init_value)) init_value

  let cosine_decay ~init_value ~decay_steps ?(alpha = 0.) () =
    if decay_steps <= 0 then
      invalid_arg "Schedule.cosine_decay: decay_steps must be positive";
    fun step ->
      let ratio = clamp_ratio ~steps:decay_steps (to_f32 step) in
      let cosine_val = cosine_factor ratio in
      Nx.mul_s (Nx.add_s (Nx.mul_s cosine_val (1.0 -. alpha)) alpha) init_value

  let exponential_decay ~init_value ~decay_rate ~decay_steps =
    if decay_steps <= 0 then
      invalid_arg "Schedule.exponential_decay: decay_steps must be positive";
    if decay_rate <= 0.0 then
      invalid_arg "Schedule.exponential_decay: decay_rate must be positive";
    fun step ->
      let ratio = Nx.div_s (to_f32 step) (float_of_int decay_steps) in
      Nx.mul_s (Nx.rpow_s decay_rate ratio) init_value

  let polynomial_decay ~init_value ~end_value ~decay_steps ?(power = 1.0) () =
    if decay_steps <= 0 then
      invalid_arg "Schedule.polynomial_decay: decay_steps must be positive";
    let steps = float_of_int decay_steps in
    fun step ->
      (* The fraction left, [(steps - s) / steps], is exactly zero from the end
         of the decay on; [1 - s / steps] can round below it. *)
      let s = Nx.minimum_s (to_f32 step) steps in
      let remaining = Nx.div_s (Nx.rsub_s steps s) steps in
      let poly = Nx.pow_s remaining power in
      Nx.add_s (Nx.mul_s poly (init_value -. end_value)) end_value

  let warmup_cosine ~init_value ~peak_value ~warmup_steps =
    if warmup_steps <= 0 then
      invalid_arg "Schedule.warmup_cosine: warmup_steps must be positive";
    fun step ->
      let ratio = clamp_ratio ~steps:warmup_steps (to_f32 step) in
      let cosine_val = Nx.rsub_s 1.0 (cosine_factor ratio) in
      Nx.add_s (Nx.mul_s cosine_val (peak_value -. init_value)) init_value

  let warmup_cosine_decay ~init_value ~peak_value ~warmup_steps ~decay_steps
      ?(end_value = 0.) () =
    if warmup_steps <= 0 then
      invalid_arg "Schedule.warmup_cosine_decay: warmup_steps must be positive";
    if decay_steps <= 0 then
      invalid_arg "Schedule.warmup_cosine_decay: decay_steps must be positive";
    fun step ->
      let s = to_f32 step in
      let warm_ratio = clamp_ratio ~steps:warmup_steps s in
      let linear_part =
        Nx.add_s (Nx.mul_s warm_ratio (peak_value -. init_value)) init_value
      in
      let decay_step =
        Nx.maximum_s (Nx.sub_s s (float_of_int warmup_steps)) 0.0
      in
      let decay_ratio = clamp_ratio ~steps:decay_steps decay_step in
      let decay_part =
        Nx.add_s
          (Nx.mul_s (cosine_factor decay_ratio) (peak_value -. end_value))
          end_value
      in
      Nx.where
        (Nx.greater_s s (float_of_int warmup_steps))
        decay_part linear_part

  let cosine_decay_restarts ~init_value ~decay_steps ?(t_mul = 1.0)
      ?(m_mul = 1.0) ?(alpha = 0.) () =
    if decay_steps <= 0 then
      invalid_arg
        "Schedule.cosine_decay_restarts: decay_steps must be positive";
    if t_mul < 1.0 then
      invalid_arg "Schedule.cosine_decay_restarts: t_mul must be at least 1";
    if m_mul <= 0.0 then
      invalid_arg "Schedule.cosine_decay_restarts: m_mul must be positive";
    let first = float_of_int decay_steps in
    let anneal cycle ratio =
      let amp = Nx.mul_s (Nx.rpow_s m_mul cycle) init_value in
      Nx.mul
        (Nx.add_s (Nx.mul_s (cosine_factor ratio) (1.0 -. alpha)) alpha)
        amp
    in
    (* Exact float comparison is intentional: 1.0 is the unmodified default. *)
    if t_mul = 1.0 then fun step ->
      let s = to_f32 step in
      let cycle = Nx.floor (Nx.div_s s first) in
      let pos = Nx.sub s (Nx.mul_s cycle first) in
      anneal cycle (Nx.div_s pos first)
    else fun step ->
      (* Geometric periods: cycle k spans [start k, start (k+1)) where
         [start k = first * (t_mul^k - 1) / (t_mul - 1)]. Inverting gives
         [k = floor (log q / log t_mul)] with [q = s*(t_mul-1)/first + 1]; the
         two [where]s absorb any float slop at the cycle boundaries. *)
      let s = to_f32 step in
      let start k =
        Nx.mul_s
          (Nx.div_s (Nx.sub_s (Nx.rpow_s t_mul k) 1.0) (t_mul -. 1.0))
          first
      in
      let q = Nx.add_s (Nx.mul_s s ((t_mul -. 1.0) /. first)) 1.0 in
      let k = Nx.floor (Nx.div_s (Nx.log q) (Stdlib.log t_mul)) in
      let k = Nx.where (Nx.less s (start k)) (Nx.sub_s k 1.0) k in
      let k =
        Nx.where
          (Nx.greater_equal s (start (Nx.add_s k 1.0)))
          (Nx.add_s k 1.0) k
      in
      let k = Nx.maximum_s k 0.0 in
      let period = Nx.mul_s (Nx.rpow_s t_mul k) first in
      let pos = Nx.sub s (start k) in
      anneal k (Nx.div pos period)

  let one_cycle ~max_value ~total_steps ?(div_factor = 25.0)
      ?(final_div_factor = 10000.0) ?(pct_start = 0.3) () =
    if total_steps <= 0 then
      invalid_arg "Schedule.one_cycle: total_steps must be positive";
    let warmup_steps = int_of_float (pct_start *. float_of_int total_steps) in
    let decay_steps = total_steps - warmup_steps in
    let init_value = max_value /. div_factor in
    let end_value = max_value /. final_div_factor in
    if warmup_steps <= 0 then fun step ->
      (* All decay: warmup is degenerate, the counter ramps straight down. *)
      let ratio = clamp_ratio ~steps:decay_steps (to_f32 step) in
      Nx.add_s
        (Nx.mul_s (cosine_factor ratio) (max_value -. end_value))
        end_value
    else if decay_steps <= 0 then fun step ->
      (* All warmup: the schedule never leaves its ramp. *)
      let ratio = clamp_ratio ~steps:warmup_steps (to_f32 step) in
      Nx.add_s (Nx.mul_s ratio (max_value -. init_value)) init_value
    else fun step ->
      let s = to_f32 step in
      let warm_ratio = clamp_ratio ~steps:warmup_steps s in
      let linear_part =
        Nx.add_s (Nx.mul_s warm_ratio (max_value -. init_value)) init_value
      in
      let decay_step =
        Nx.maximum_s (Nx.sub_s s (float_of_int warmup_steps)) 0.0
      in
      let decay_ratio = clamp_ratio ~steps:decay_steps decay_step in
      let decay_part =
        Nx.add_s
          (Nx.mul_s (cosine_factor decay_ratio) (max_value -. end_value))
          end_value
      in
      Nx.where
        (Nx.greater_s s (float_of_int warmup_steps))
        decay_part linear_part

  let piecewise_constant ~boundaries ~values =
    let n_boundaries = List.length boundaries in
    let n_values = List.length values in
    if n_values <> n_boundaries + 1 then
      invalid_arg
        (Printf.sprintf
           "Schedule.piecewise_constant: expected %d values for %d boundaries, \
            got %d"
           (n_boundaries + 1) n_boundaries n_values);
    let boundaries = Array.of_list boundaries in
    let values = Array.of_list values in
    for i = 1 to Array.length boundaries - 1 do
      if boundaries.(i) <= boundaries.(i - 1) then
        invalid_arg
          "Schedule.piecewise_constant: boundaries must be strictly increasing"
    done;
    fun step ->
      let s = to_f32 step in
      let acc = ref (Nx.scalar_like s values.(Array.length values - 1)) in
      for i = Array.length boundaries - 1 downto 0 do
        acc :=
          Nx.where
            (Nx.greater_s s (float_of_int boundaries.(i)))
            !acc
            (Nx.scalar_like s values.(i))
      done;
      !acc

  let join segments =
    if segments = [] then
      invalid_arg "Schedule.join: segments must not be empty";
    List.iter
      (fun (n, _) ->
        if n <= 0 then
          invalid_arg "Schedule.join: segment lengths must be positive")
      segments;
    let segments = Array.of_list segments in
    let n_seg = Array.length segments in
    let starts = Array.make n_seg 0 in
    for i = 1 to n_seg - 1 do
      starts.(i) <- starts.(i - 1) + fst segments.(i - 1)
    done;
    fun step ->
      (* Selected last to first: the earliest segment whose span covers the
         relative counter wins. Inactive segments are still computed (at
         out-of-range counters) and discarded by [where]. *)
      let at i = snd segments.(i) (Nx.sub_s step (Int32.of_int starts.(i))) in
      let acc = ref (at (n_seg - 1)) in
      for i = n_seg - 2 downto 0 do
        let rel = Nx.sub_s step (Int32.of_int starts.(i)) in
        acc :=
          Nx.where
            (Nx.less_equal_s rel (Int32.of_int (fst segments.(i))))
            (at i) !acc
      done;
      !acc

  let rate v step = Nx.scalar_at (Nx.placement step) Nx.float32 v
end

(* Each constructor at sets of numbers that share its structure, with the
   number of steps whose thrice the counters reach. *)
module Sets (F : Forms) = struct
  let wcd (init_value, peak_value, warmup_steps, decay_steps, end_value) =
    ( F.warmup_cosine_decay ~init_value ~peak_value ~warmup_steps ~decay_steps
        ~end_value (),
      warmup_steps + decay_steps )

  (* TESSERA's schedule of [--steps], [--lr] and [--warmup]
     (apps/tessera/bin/main.ml). *)
  let tessera (lr, warmup) steps =
    let warmup_steps =
      max 1 (Float.to_int (Float.round (warmup *. Float.of_int steps)))
    in
    F.warmup_cosine_decay ~init_value:0. ~peak_value:lr ~warmup_steps
      ~decay_steps:(max 1 (steps - warmup_steps))
      ()

  let all =
    [
      ("rate", [ (F.rate 0.1, 10); (F.rate 3e-4, 10); (F.rate 0., 10) ]);
      ( "linear",
        [
          (F.linear ~init_value:1e-3 ~end_value:1e-5 ~steps:100, 100);
          (F.linear ~init_value:0.2 ~end_value:0.7 ~steps:37, 37);
        ] );
      ( "cosine_decay",
        [
          (F.cosine_decay ~init_value:0.1 ~decay_steps:100 ~alpha:0.1 (), 100);
          (F.cosine_decay ~init_value:3e-4 ~decay_steps:777 (), 777);
        ] );
      ( "exponential_decay",
        List.map
          (fun (init_value, decay_rate, decay_steps) ->
            ( F.exponential_decay ~init_value ~decay_rate ~decay_steps,
              decay_steps ))
          [ (1e-3, 0.96, 1000); (0.1, 0.5, 100); (3e-4, 0.9, 777) ] );
      ( "polynomial_decay",
        List.map
          (fun (init_value, end_value, decay_steps, power) ->
            ( F.polynomial_decay ~init_value ~end_value ~decay_steps ~power (),
              decay_steps ))
          [
            (1e-3, 1e-5, 1000, 2.0);
            (1e-3, 0., 1000, 0.5);
            (1e-2, 1e-4, 300, 1.0);
          ] );
      ( "warmup_cosine",
        [
          ( F.warmup_cosine ~init_value:0. ~peak_value:1e-3 ~warmup_steps:50,
            50 );
          (F.warmup_cosine ~init_value:1e-5 ~peak_value:0.1 ~warmup_steps:7, 7);
        ] );
      ( "warmup_cosine_decay",
        List.map wcd
          [
            (1e-5, 1e-3, 10, 100, 1e-6);
            (0., 2e-3, 7, 33, 0.);
            (1e-5, 3e-4, 1, 1, 1e-6);
            (0., 1e-3, 100, 9900, 0.);
            (1e-7, 0.1, 500, 99500, 1e-5);
          ] );
      ( "TESSERA's distil and pretrain recipes",
        [
          (tessera (8e-4, 0.05) 200, 200);
          (tessera (2e-3, 0.1) 300, 300);
          (tessera (8e-4, 0.05) 3000, 3000);
        ] );
      ( "cosine_decay_restarts",
        [
          ( F.cosine_decay_restarts ~init_value:1e-3 ~decay_steps:100 ~m_mul:0.9
              (),
            100 );
          ( F.cosine_decay_restarts ~init_value:1e-3 ~decay_steps:250 ~m_mul:0.5
              ~alpha:0.1 (),
            250 );
        ] );
      ( "cosine_decay_restarts with geometric periods",
        [
          ( F.cosine_decay_restarts ~init_value:1.0 ~decay_steps:10 ~t_mul:2.0
              ~m_mul:0.5 (),
            100 );
          ( F.cosine_decay_restarts ~init_value:3e-4 ~decay_steps:17 ~t_mul:1.5
              ~alpha:0.1 (),
            200 );
        ] );
      ( "one_cycle",
        [
          (F.one_cycle ~max_value:1.0 ~total_steps:100 (), 100);
          ( F.one_cycle ~max_value:3e-3 ~total_steps:1000 ~div_factor:10.
              ~final_div_factor:1e3 ~pct_start:0.25 (),
            1000 );
        ] );
      ( "one_cycle without warmup",
        [
          (F.one_cycle ~max_value:1.0 ~total_steps:3 ~pct_start:0.2 (), 3);
          (F.one_cycle ~max_value:0.5 ~total_steps:4 ~pct_start:0.1 (), 4);
        ] );
      ( "one_cycle without decay",
        [
          (F.one_cycle ~max_value:1.0 ~total_steps:10 ~pct_start:1.0 (), 10);
          (F.one_cycle ~max_value:2e-3 ~total_steps:7 ~pct_start:1.5 (), 7);
        ] );
      ( "piecewise_constant",
        [
          ( F.piecewise_constant ~boundaries:[ 100; 200 ]
              ~values:[ 0.1; 0.01; 0.001 ],
            200 );
          ( F.piecewise_constant ~boundaries:[ 7; 50 ]
              ~values:[ 1.0; 0.3; 2e-4 ],
            50 );
        ] );
      ( "join",
        [
          ( F.join
              [
                (10, F.rate 1.0);
                (20, F.linear ~init_value:1.0 ~end_value:0. ~steps:20);
              ],
            30 );
          ( F.join
              [
                (30, F.rate 3e-4);
                (5, F.linear ~init_value:0.5 ~end_value:2e-3 ~steps:5);
              ],
            35 );
        ] );
    ]
end

module Data_sets = Sets (Data)
module Literal_sets = Sets (Literal)

let bits y = Array.map Int32.bits_of_float (Nx.to_array y)
let counters n = Nx.init Nx.int32 [| n |] (fun i -> Int32.of_int i.(0))
let counter k = Nx.scalar Nx.int32 (Int32.of_int k)

let print f =
  Format.asprintf "%a" Rune.pp_program
    (Rune.program Nx.Ptree.tensor Nx.Ptree.tensor f (counter 5))

(* [same f data literal c] says that [f data] and [f literal] give equal bits
   at the counters [c]. *)
let same ~msg f data literal c =
  equal ~msg (array int32) (bits (f literal c)) (bits (f data c))

(* For each set of a constructor: its bits, eager and compiled, at the
   counters 0 to thrice its steps; and its program, which is the first
   set's. *)
let schedule_law (name, data) =
  let literal = List.assoc name Literal_sets.all in
  let program = print (fst (List.hd data)) in
  List.iteri
    (fun i ((d, steps), (l, _)) ->
      let msg what = Printf.sprintf "%s, set %d: %s" name i what in
      let c = counters ((3 * steps) + 1) in
      same ~msg:(msg "eager") Fun.id d l c;
      same ~msg:(msg "compiled") Rune.jit' d l c;
      equal ~msg:(msg "one program") string program (print d))
    (List.combine data literal)

let schedule_laws =
  List.map
    (fun ((name, _) as case) -> test name (fun () -> schedule_law case))
    Data_sets.all

(* TESSERA's two recipes, compiled on a 0-d counter at every counter of
   --steps 100, 1000 and 3000, and on a vector of every counter of --steps
   100000. *)
let test_tessera_compiled () =
  List.iter
    (fun recipe ->
      List.iter
        (fun steps ->
          let d = Rune.jit' (Data_sets.tessera recipe steps)
          and l = Rune.jit' (Literal_sets.tessera recipe steps) in
          for k = 0 to steps + 1 do
            let msg = Printf.sprintf "--steps %d, counter %d" steps k in
            same ~msg (fun f c -> f c) d l (counter k)
          done)
        [ 100; 1000; 3000 ];
      let steps = 100_000 in
      same ~msg:"--steps 100000" Rune.jit'
        (Data_sets.tessera recipe steps)
        (Literal_sets.tessera recipe steps)
        (counters (steps + 2)))
    [ (8e-4, 0.05); (2e-3, 0.1) ]

(* Eager, every counter of every --steps up to 1000 of both recipes. *)
let test_tessera_eager () =
  List.iter
    (fun recipe ->
      for steps = 1 to 1000 do
        same ~msg:(Printf.sprintf "--steps %d" steps) Fun.id
          (Data_sets.tessera recipe steps)
          (Literal_sets.tessera recipe steps)
          (counters (steps + 2))
      done)
    [ (8e-4, 0.05); (2e-3, 0.1) ]

(* The program of an AdamW step [step st ~params ~grads]. *)
let step_program step =
  let both = Nx.Ptree.pair Wb.ptree (Vega.adam_ptree Wb.ptree) in
  let x =
    let params = wb 1 in
    (params, Vega.adamw_init Wb.ptree params)
  in
  let f (params, st) = step st ~params ~grads:(grads params) in
  Format.asprintf "%a" Rune.pp_program (Rune.program both both f x)

(* A TESSERA training step at two runs, --steps 200 at --lr 8e-4 and --decay
   1e-4 and --steps 300 at --lr 2e-3 and --decay 0.05: one program. *)
let test_one_step_program () =
  let program (lr, warmup, decay) steps =
    let lr = Data_sets.tessera (lr, warmup) steps in
    let weight_decay = Vega.rate decay in
    step_program (fun st -> Vega.adamw_step Wb.ptree ~lr ~weight_decay st)
  in
  equal string
    (program (8e-4, 0.05, 1e-4) 200)
    (program (2e-3, 0.1, 0.05) 300)

(* A step's coefficients are constants of its program: a sweep over [b1],
   [b2] or [eps] compiles once per value. *)
let test_coefficients_are_constants () =
  let program ?(b1 = 0.9) ?(b2 = 0.95) ?(eps = 1e-8) () =
    step_program (fun st -> Vega.adamw_step Wb.ptree ~lr ~b1 ~b2 ~eps st)
  in
  let base = program () in
  equal ~msg:"one value" string base (program ());
  not_equal ~msg:"b1" string base (program ~b1:0.8 ());
  not_equal ~msg:"b2" string base (program ~b2:0.99 ());
  not_equal ~msg:"eps" string base (program ~eps:1e-6 ())

let () =
  exit
    (run "vega jit"
       [
         group "a step compiles with its state donated" tests;
         group "a bfloat16 step draws as eager's" half_tests;
         group "precision"
           [
             test "compiled radam rectifies as in float64"
               test_radam_compiled_rectification;
             test "a polynomial decay compiles to eager's values"
               test_polynomial_decay_compiles;
           ];
         group "loss scaling"
           [
             test "a loss-scaled step compiles, overflow included"
               test_loss_scaled_compiles;
           ];
         group "a schedule's numbers are data" schedule_laws;
         group "TESSERA's schedules are data"
           [
             test "compiled at every counter" test_tessera_compiled;
             test "eager at every counter" test_tessera_eager;
             test "one step program for two runs" test_one_step_program;
           ];
         group "coefficients"
           [
             test "a sweep compiles once per value"
               test_coefficients_are_constants;
           ];
       ])
