(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The rules every function over parameters follows: structures and states,
   gradient transformations, rates, weight decay and parts. Each optimizer's
   formula is test_steps', Muon's test_muon's, half-precision stores
   test_precision's. *)

open Windtrap

let vec xs = Nx.create Nx.float64 [| Array.length xs |] xs
let rate64 v step = Nx.scalar_at (Nx.placement step) Nx.float64 v

(* A float64 vector, and a check of its elements. *)
module Vec = struct
  type t = Nx.host Nx.float64_t

  let ptree : (t, Nx.host) Nx.Ptree.t = Nx.Ptree.tensor
end

let check_vec ?(eps = 1e-9) ?msg expected actual =
  let t = if eps = 0. then float_exact else float eps in
  equal ?msg (array t) expected (Nx.to_array actual)

(* Every step *)

(* What [n] steps from a step's initial state give: the parameters they end
   with, the initial state's arrays copied before the steps and read again after
   them, and the final state's arrays. *)
type 's ran = {
  params : 's;
  initial : Nx.host Nx.packed list;
  initial_after : Nx.host Nx.packed list;
  final : Nx.host Nx.packed list;
}

(* Every step, by its call, with its function's and its init's names: [run n p x
   g] steps [n] times from the step's initial state over the parameters [x],
   each time at the gradient [g]. *)
type step = {
  name : string;
  fn : string;
  init : string;
  run : 's. int -> ('s, Nx.host) Nx.Ptree.t -> 's -> 's -> 's ran;
}

let arrays s x = fst (Nx.Ptree.flatten s x)
let copies = List.map (fun (Nx.P a) -> Nx.P (Nx.copy a))

let stepping init state step n p x g =
  let st = init p x in
  let initial = copies (arrays (state p) st) in
  let rec go k (x, st') =
    if k = 0 then (x, st') else go (k - 1) (step p st' x g)
  in
  let params, st' = go n (x, st) in
  {
    params;
    initial;
    initial_after = arrays (state p) st;
    final = arrays (state p) st';
  }

let steps =
  let lr = Vega.rate 0.01 in
  [
    {
      name = "sgd_step";
      fn = "Vega.sgd_step";
      init = "Vega.sgd_init";
      run =
        (fun n p ->
          stepping Vega.sgd_init Vega.sgd_ptree
            (fun p st x g -> Vega.sgd_step p ~lr st ~params:x ~grads:g)
            n p);
    };
    {
      name = "sgd_step ~momentum:0.9";
      fn = "Vega.sgd_step";
      init = "Vega.sgd_init";
      run =
        (fun n p ->
          stepping Vega.sgd_init Vega.sgd_ptree
            (fun p st x g ->
              Vega.sgd_step p ~lr ~momentum:0.9 st ~params:x ~grads:g)
            n p);
    };
    {
      name = "lars_step";
      fn = "Vega.lars_step";
      init = "Vega.lars_init";
      run =
        (fun n p ->
          stepping Vega.lars_init Vega.sgd_ptree
            (fun p st x g -> Vega.lars_step p ~lr st ~params:x ~grads:g)
            n p);
    };
    {
      name = "muon_step";
      fn = "Vega.muon_step";
      init = "Vega.muon_init";
      run =
        (fun n p ->
          stepping Vega.muon_init Vega.sgd_ptree
            (fun p st x g -> Vega.muon_step p ~lr st ~params:x ~grads:g)
            n p);
    };
    {
      name = "adam_step";
      fn = "Vega.adam_step";
      init = "Vega.adam_init";
      run =
        (fun n p ->
          stepping Vega.adam_init Vega.adam_ptree
            (fun p st x g -> Vega.adam_step p ~lr st ~params:x ~grads:g)
            n p);
    };
    {
      name = "adamw_step";
      fn = "Vega.adamw_step";
      init = "Vega.adamw_init";
      run =
        (fun n p ->
          stepping Vega.adamw_init Vega.adam_ptree
            (fun p st x g -> Vega.adamw_step p ~lr st ~params:x ~grads:g)
            n p);
    };
    (* [b2 = 0.9] rectifies from the sixth step. *)
    {
      name = "radam_step ~b2:0.9";
      fn = "Vega.radam_step";
      init = "Vega.radam_init";
      run =
        (fun n p ->
          stepping Vega.radam_init Vega.adam_ptree
            (fun p st x g ->
              Vega.radam_step p ~lr ~b2:0.9 st ~params:x ~grads:g)
            n p);
    };
    {
      name = "lamb_step";
      fn = "Vega.lamb_step";
      init = "Vega.lamb_init";
      run =
        (fun n p ->
          stepping Vega.lamb_init Vega.adam_ptree
            (fun p st x g -> Vega.lamb_step p ~lr st ~params:x ~grads:g)
            n p);
    };
    {
      name = "rmsprop_step ~momentum:0.9";
      fn = "Vega.rmsprop_step";
      init = "Vega.rmsprop_init";
      run =
        (fun n p ->
          stepping Vega.rmsprop_init Vega.rmsprop_ptree
            (fun p st x g ->
              Vega.rmsprop_step p ~lr ~momentum:0.9 st ~params:x ~grads:g)
            n p);
    };
    {
      name = "adagrad_step";
      fn = "Vega.adagrad_step";
      init = "Vega.adagrad_init";
      run =
        (fun n p ->
          stepping Vega.adagrad_init Vega.adagrad_ptree
            (fun p st x g -> Vega.adagrad_step p ~lr st ~params:x ~grads:g)
            n p);
    };
    {
      name = "adan_step";
      fn = "Vega.adan_step";
      init = "Vega.adan_init";
      run =
        (fun n p ->
          stepping Vega.adan_init Vega.adan_ptree
            (fun p st x g -> Vega.adan_step p ~lr st ~params:x ~grads:g)
            n p);
    };
    {
      name = "lion_step";
      fn = "Vega.lion_step";
      init = "Vega.lion_init";
      run =
        (fun n p ->
          stepping Vega.lion_init Vega.lion_ptree
            (fun p st x g -> Vega.lion_step p ~lr st ~params:x ~grads:g)
            n p);
    };
    {
      name = "adafactor_step";
      fn = "Vega.adafactor_step";
      init = "Vega.adafactor_init";
      run =
        (fun n p ->
          stepping
            (fun p x -> Vega.adafactor_init p x)
            Vega.adafactor_ptree
            (fun p st x g -> Vega.adafactor_step p ~lr st ~params:x ~grads:g)
            n p);
    };
    {
      name = "lbfgs_step";
      fn = "Vega.lbfgs_step";
      init = "Vega.lbfgs_init";
      run =
        (fun n p ->
          stepping
            (fun p x -> Vega.lbfgs_init p ~history:2 x)
            Vega.lbfgs_ptree
            (fun p st x g ->
              let st = Vega.lbfgs_step p ~lr g { st with params = x } in
              (st.params, st))
            n p);
    };
  ]

(* A float32 matrix and an [int32] counter. *)
let with_counter =
  let p = Nx.Ptree.(pair tensor tensor) in
  let x =
    ( Nx.create Nx.float32 [| 2; 3 |] [| 1.0; -2.0; 0.5; 3.0; 0.0; -1.5 |],
      Nx.create Nx.int32 [| 2 |] [| 7l; -9l |] )
  in
  let g =
    ( Nx.create Nx.float32 [| 2; 3 |] [| 0.5; 1.0; -2.0; 0.25; 4.0; 1.0 |],
      Nx.zeros Nx.int32 [| 2 |] )
  in
  (p, x, g)

(* The elements of every integer array of a list. *)
let integers xs =
  List.filter_map
    (fun (Nx.P a) ->
      match Nx.Dtype.kind (Nx.dtype a) with
      | Signed -> Some (Nx.to_array (Nx.cast Nx.int64 a))
      | Float | Complex | Unsigned | Boolean -> None)
    xs

(* Parameters and states *)

(* One skeleton *)

let one_skeleton =
  let p = Nx.Ptree.(list tensor) in
  let m = Nx.ones Nx.float32 [| 2; 3 |] in
  cases "every step names itself and the root where the gradients differ"
    ~id:(fun s -> s.name)
    steps
    (fun step ->
      raises
        (Invalid_argument
           (step.fn
          ^ ": the root: length 1 in the gradients, length 2 in the parameters"
           ))
        (fun () -> step.run 1 p [ m; m ] [ m ]))

(* A structure over packed tensors, whose leaves' dtypes differ between
   values. *)
module Any = struct
  type (_, 'd) t = 'd Nx.packed

  let walk c (Nx.P x) = Nx.P (Nx.Ptree.Walk.tensor c x)
end

(* A structure whose one leaf sits at the field ["a.b"] or at [a] then [b]: two
   paths that print alike. *)
module Split = struct
  type (_, 'd) t = bool * 'd Nx.float64_t

  let walk c (dotted, x) =
    let open Nx.Ptree.Walk in
    let x =
      if dotted then field c "a.b" tensor x
      else field c "a" (fun c -> field c "b" tensor) x
    in
    (dotted, x)
end

(* A state part, a path that prints like another, a dtype: each mismatch names
   its step, the path and both holdings. *)
let test_steps_check_the_skeleton () =
  let vecs = Nx.Ptree.list Vec.ptree in
  let params = [ vec [| 1.0 |]; vec [| 2.0 |] ] in
  let st = Vega.adam_init vecs params in
  raises
    (Invalid_argument
       "Vega.adam_step: the root: length 1 in the gradients, length 2 in the \
        parameters") (fun () ->
      ignore
        (Vega.adam_step vecs ~lr:(Vega.rate 0.1) st ~params
           ~grads:[ vec [| 1.0 |] ]));
  raises
    (Invalid_argument
       "Vega.adamw_step: the root: length 1 in mu, length 2 in the parameters")
    (fun () ->
      ignore
        (Vega.adamw_step vecs ~lr:(Vega.rate 0.1)
           { st with mu = [ vec [| 0.0 |] ] }
           ~params ~grads:params));
  raises
    (Invalid_argument
       "Vega.sgd_step: the root: length 1 in the velocity, length 2 in the \
        parameters") (fun () ->
      ignore
        (Vega.sgd_step vecs ~lr:(Vega.rate 0.1) ~momentum:0.9
           { velocity = [ vec [| 0.0 |] ]; step = st.step }
           ~params ~grads:params));
  raises
    (Invalid_argument
       "Vega.sgd_step: the root: length 1 in the gradients, length 2 in the \
        parameters") (fun () ->
      ignore
        (Vega.sgd_step vecs ~lr:(Vega.rate 0.1)
           (Vega.sgd_init vecs params)
           ~params
           ~grads:[ vec [| 1.0 |] ]));
  raises
    (Invalid_argument
       "Vega.adan_step: the root: length 1 in prev_grads, length 2 in the \
        parameters") (fun () ->
      let st = Vega.adan_init vecs params in
      ignore
        (Vega.adan_step vecs ~lr:(Vega.rate 0.1)
           { st with prev_grads = [ vec [| 0.0 |] ] }
           ~params ~grads:params));
  let split : ((_, Nx.host) Split.t, Nx.host) Nx.Ptree.t = Nx.Ptree.instantiate (module Split) in
  let params = (true, vec [| 1.0 |]) in
  raises
    (Invalid_argument
       "Vega.adam_step: [a; b]: a leaf in the gradients, a leaf at [a.b] in \
        the parameters") (fun () ->
      ignore
        (Vega.adam_step split ~lr:(Vega.rate 0.1)
           (Vega.adam_init split params)
           ~params
           ~grads:(false, vec [| 1.0 |])));
  let any : ((_, Nx.host) Any.t, Nx.host) Nx.Ptree.t = Nx.Ptree.instantiate (module Any) in
  let params = Nx.P (vec [| 1.0 |]) in
  raises
    (Invalid_argument
       "Vega.adam_step: the root: float32 in the gradients, float64 in the \
        parameters") (fun () ->
      ignore
        (Vega.adam_step any ~lr:(Vega.rate 0.1)
           (Vega.adam_init any params)
           ~params
           ~grads:(Nx.P (Nx.create Nx.float32 [| 1 |] [| 1.0 |]))))

(* An L-BFGS state's parts have the parameters' dtypes, as the gradients do: a
   part of another dtype raises at its leaf, naming the part. *)
let test_lbfgs_checks_the_state_dtypes () =
  let any : ((_, Nx.host) Any.t, Nx.host) Nx.Ptree.t =
    Nx.Ptree.instantiate (module Any)
  in
  let params = Nx.create Nx.bfloat16 [| 2 |] [| 1.0; 2.0 |] in
  let st = Vega.lbfgs_init any ~history:2 (Nx.P params) in
  let recast dt (Nx.P x) = Nx.P (Nx.cast dt x) in
  let rejects part dt st =
    raises
      (Invalid_argument
         (Printf.sprintf
            "Vega.lbfgs_step: the root: %s in %s, bfloat16 in the parameters"
            (Nx.Dtype.name dt) part))
      (fun () -> Vega.lbfgs_step any ~lr:(Vega.rate 0.1) (Nx.P params) st)
  in
  rejects "grads" Nx.float16 { st with grads = recast Nx.float16 st.grads };
  rejects "s" Nx.float32 { st with s = recast Nx.float32 st.s };
  rejects "y" Nx.float32 { st with y = recast Nx.float32 st.y }

(* A skeleton mismatch raises before any leaf is stepped: the step asks a leaf's
   rule as it steps it, and asks none. *)
let test_raises_before_stepping () =
  let p = Nx.Ptree.(list tensor) in
  let m = Nx.ones Nx.float32 [| 2; 3 |] in
  let asked = ref 0 in
  let decays _ _ =
    incr asked;
    true
  in
  raises_match Exn.invalid_arg (fun () ->
      Vega.adamw_step p ~lr:(Vega.rate 0.1) ~decays
        (Vega.adamw_init p [ m; m ])
        ~params:[ m; m ] ~grads:[ m ]);
  equal ~msg:"rules asked" int 0 !asked

(* Non-parameter leaves *)

(* An [int32] leaf comes back unchanged from every step, in the parameters and
   in every part of the state. *)
let non_parameter_leaves =
  cases
    "every step passes an integer leaf through, in the parameters and the state"
    ~id:(fun s -> s.name)
    steps
    (fun step ->
      let p, x, g = with_counter in
      let ran = step.run 2 p x g in
      equal ~msg:"parameters" (array int32)
        (Nx.to_array (snd x))
        (Nx.to_array (snd ran.params));
      (* The last array of a state is its counter, which advances. *)
      let parts l = List.filteri (fun i _ -> i < List.length l - 1) l in
      equal ~msg:"state"
        (list (array int64))
        (integers (parts ran.initial))
        (integers (parts ran.final)))

(* A step leaves the parameters, the gradients and the state it is given as they
   were. *)
let arguments_unchanged =
  cases "every step leaves its arguments as they were"
    ~id:(fun s -> s.name)
    steps
    (fun step ->
      let p, x, g = with_counter in
      let floats (a, _) = Nx.to_array a in
      let x0 = floats x and g0 = floats g in
      let ran = step.run 2 p x g in
      equal ~msg:"parameters" (array float_exact) x0 (floats x);
      equal ~msg:"gradients" (array float_exact) g0 (floats g);
      let all =
        List.map (fun (Nx.P a) -> Nx.to_array (Nx.bitcast Nx.uint8 a))
      in
      equal ~msg:"state"
        (list (array int))
        (all ran.initial) (all ran.initial_after))

(* Complex leaves *)

let cmat dt zs =
  Nx.create dt [| 2; Array.length zs / 2 |]
    (Array.map (fun (re, im) -> { Complex.re; im }) zs)

(* A complex leaf's real coordinates at the parts' format [dt], and its
   gradient there: the conjugate of Rune's gradient. *)
let coordinates dt z = Nx.stack [ Nx.real dt z; Nx.imag dt z ]
let descent dt g = Nx.stack [ Nx.real dt g; Nx.neg (Nx.imag dt g) ]

(* The float elements of an array, a complex one's as its coordinates, read
   at float64; nothing for an integer array. *)
let floats_of (type v s) (a : (v, s, _) Nx.t) =
  match Nx.Dtype.kind (Nx.dtype a) with
  | Float -> Some (Nx.to_array (Nx.cast Nx.float64 a))
  | Complex -> Some (Nx.to_array (coordinates Nx.float64 a))
  | Signed | Unsigned | Boolean -> None

let test_complex_sgd () =
  let z = cmat Nx.complex64 [| (1.0, 2.0); (-3.0, 0.5) |] in
  let g = cmat Nx.complex64 [| (0.5, -1.0); (2.0, 4.0) |] in
  let p = Nx.Ptree.tensor in
  let z', st =
    Vega.sgd_step p ~lr:(Vega.rate 0.25) (Vega.sgd_init p z) ~params:z ~grads:g
  in
  (* z - lr * conj g, exact at these binary fractions. *)
  equal ~msg:"real parts" (array float_exact) [| 0.875; -3.5 |]
    (Nx.to_array (Nx.real Nx.float64 z'));
  equal ~msg:"imaginary parts" (array float_exact) [| 1.75; 1.5 |]
    (Nx.to_array (Nx.imag Nx.float64 z'));
  equal ~msg:"the velocity is the conjugate gradient" (array float_exact)
    [| 1.0; -4.0 |]
    (Nx.to_array (Nx.imag Nx.float64 st.velocity))

(* Every init holds zeros in every part, the parameters in L-BFGS's [params],
   and the counter [0]. *)
let inits_are_zero =
  cases "every init holds zeros and the counter 0"
    ~id:(fun s -> s.name)
    steps
    (fun step ->
      let p, x, g = with_counter in
      let ran = step.run 0 p x g in
      let floats = List.filter_map (fun (Nx.P a) -> floats_of a) ran.initial in
      let floats =
        if step.fn = "Vega.lbfgs_step" then List.tl floats else floats
      in
      equal ~msg:"parts"
        (list (array float_exact))
        (List.map (Array.map (fun _ -> 0.0)) floats)
        floats;
      let (Nx.P counter) = List.nth ran.initial (List.length ran.initial - 1) in
      equal ~msg:"counter" (array int64) [| 0L |]
        (Nx.to_array (Nx.cast Nx.int64 counter)))

(* [six step x g] is the floats of the parameters and the state after six steps
   over the leaf [x] at the gradient [g], past RAdam's switch at [b2 = 0.9]. *)
let six step x g =
  let ran = step.run 6 Nx.Ptree.tensor x g in
  Option.to_list (floats_of ran.params)
  @ List.filter_map (fun (Nx.P a) -> floats_of a) ran.final

(* A complex leaf of two or more axes steps as the real leaf of its
   coordinates, stacked along a new first axis, against the descent direction
   of its coordinates: its parameters and its state, at the parts' format. A
   leaf of fewer axes steps each part as a real leaf of its own shape, which
   Adafactor factors differently, so the law is drawn over two and three axes,
   empty ones included. Parts are often zeros of either sign, whose sign a
   complex product by a real factor can lose. *)
let complex_leaf =
  let open Gen in
  let* rank = int_range 2 3 in
  let* shape = array ~size:(constant rank) (int_range 0 3) in
  let n = Array.fold_left ( * ) 1 shape in
  let part =
    frequency
      [ (1, constant 0.0); (1, constant (-0.0)); (6, float_range (-4.0) 4.0) ]
  in
  let+ z = array ~size:(constant n) (pair part part)
  and+ g = array ~size:(constant n) (pair part part) in
  (shape, z, g)

let as_coordinates dtype parts =
  prop "every step moves a complex leaf as its real coordinates"
    ~examples:[ ([| 1; 1 |], [| (0.0, -0.0) |], [| (0.0, 0.0) |]) ]
    complex_leaf
    (fun (shape, z, g) ->
      cover "an empty leaf" (Array.exists (( = ) 0) shape);
      cover "a leaf of three axes" (Array.length shape = 3);
      let negative_zero x = x = 0.0 && Float.sign_bit x in
      cover "a part of -0."
        (Array.exists
           (fun (re, im) -> negative_zero re || negative_zero im)
           (Array.append z g));
      let leaf zs =
        Nx.create dtype shape
          (Array.map (fun (re, im) -> { Complex.re; im }) zs)
      in
      let z = leaf z and g = leaf g in
      (* Adafactor's moment of a factored leaf's elements is unused, a scalar
         zero at the leaf's dtype: a complex zero's two coordinates against a
         real zero's one. L-BFGS keeps the gradient it received, Rune's on the
         complex leaf: its parameters and its memory, [s] and [y], are
         compared. The complex memory's coordinates come before its axis of
         [history] pairs, the real memory's after: [pairs_first] moves them
         there. *)
      let used name floats =
        match name with
        | "adafactor_step" -> List.filteri (fun i _ -> i < 3) floats
        | "lbfgs_step" ->
            List.filteri (fun i _ -> i = 0 || i = 3 || i = 4) floats
        | _ -> floats
      in
      let pairs_first a =
        let history = 2 in
        let n = Array.length a / (2 * history) in
        Array.init (Array.length a) (fun j ->
            let i = j / (2 * n) and c = j / n mod 2 and k = j mod n in
            a.((((c * history) + i) * n) + k))
      in
      let on_complex name floats =
        match name with
        | "lbfgs_step" ->
            List.mapi (fun i a -> if i = 0 then a else pairs_first a) floats
        | _ -> floats
      in
      (* Muon steps a complex matrix as its real form ([test_muon]). *)
      List.iter
        (fun step ->
          let name = step.name in
          equal ~msg:name
            (list (array float_exact))
            (used name (six step (coordinates parts z) (descent parts g)))
            (on_complex name (used name (six step z g))))
        (List.filter (fun s -> s.name <> "muon_step") steps))

(* States are structures *)

(* Two float32 leaves, [a] of two elements and [b] of one. *)
module Pair = struct
  type t = { a : Nx.host Nx.float32_t; b : Nx.host Nx.float32_t }

  let walk c { a; b } =
    let open Nx.Ptree.Walk in
    let a = field c "a" tensor a in
    let b = field c "b" tensor b in
    { a; b }

  let ptree : (t, Nx.host) Nx.Ptree.t = Nx.Ptree.v walk
end

let pair a b =
  {
    Pair.a = Nx.create Nx.float32 [| Array.length a |] a;
    b = Nx.create Nx.float32 [| Array.length b |] b;
  }

let visit_lines s x =
  List.map (Format.asprintf "%a" Nx.Ptree.pp_visit) (Nx.Ptree.visits s x)

(* Each state visits its fields in order, each at the parameters' paths below
   it, then [step]. *)
let state_visits =
  let p = Pair.ptree and x = pair [| 1.0; 2.0 |] [| 3.0 |] in
  let fields names =
    List.concat_map (fun f -> [ f ^ ".a: a leaf"; f ^ ".b: a leaf" ]) names
    @ [ "step: a leaf" ]
  in
  cases "each state visits its parts at the parameters' paths, then step"
    ~id:(fun (name, _, _) -> name)
    [
      ( "sgd_ptree",
        fields [ "velocity" ],
        fun () -> visit_lines (Vega.sgd_ptree p) (Vega.sgd_init p x) );
      ( "adam_ptree",
        fields [ "mu"; "nu" ],
        fun () -> visit_lines (Vega.adam_ptree p) (Vega.adam_init p x) );
      ( "rmsprop_ptree",
        fields [ "nu"; "velocity" ],
        fun () -> visit_lines (Vega.rmsprop_ptree p) (Vega.rmsprop_init p x) );
      ( "adagrad_ptree",
        fields [ "sum_of_squares" ],
        fun () -> visit_lines (Vega.adagrad_ptree p) (Vega.adagrad_init p x) );
      ( "adan_ptree",
        fields [ "mu"; "delta"; "nu"; "prev_grads" ],
        fun () -> visit_lines (Vega.adan_ptree p) (Vega.adan_init p x) );
      ( "lion_ptree",
        fields [ "mu" ],
        fun () -> visit_lines (Vega.lion_ptree p) (Vega.lion_init p x) );
      ( "adafactor_ptree",
        fields [ "nu_row"; "nu_col"; "nu" ],
        fun () -> visit_lines (Vega.adafactor_ptree p) (Vega.adafactor_init p x)
      );
      ( "lbfgs_ptree",
        fields [ "params"; "grads"; "s"; "y" ],
        fun () -> visit_lines (Vega.lbfgs_ptree p) (Vega.lbfgs_init p x) );
    ]
    (fun (_, expected, visits) -> equal (list string) expected (visits ()))

let test_state_reports () =
  let p = Nx.Ptree.(list tensor) in
  equal (list string)
    [
      "mu: length 2";
      "mu.0: a leaf";
      "mu.1: a leaf";
      "nu: length 2";
      "nu.0: a leaf";
      "nu.1: a leaf";
      "step: a leaf";
    ]
    (visit_lines (Vega.adam_ptree p)
       (Vega.adam_init p [ vec [| 1.0 |]; vec [| 2.0 |] ]))

(* A map over a state's structure reaches each of its arrays where it lies:
   doubling every array of a state after one step doubles each part and the
   counter. *)
let test_state_maps () =
  let p = Pair.ptree in
  let x = pair [| 1.0; -2.0 |] [| 3.0 |] in
  let _, st =
    Vega.adam_step p ~lr:(Vega.rate 0.1) (Vega.adam_init p x) ~params:x
      ~grads:(pair [| 0.5; 0.25 |] [| -0.5 |])
  in
  let doubled = Nx.Ptree.map (Vega.adam_ptree p) (fun _ t -> Nx.add t t) st in
  let twice t = Array.map (fun x -> 2.0 *. x) (Nx.to_array t) in
  equal ~msg:"mu.a" (array float_exact) (twice st.mu.a)
    (Nx.to_array doubled.mu.a);
  equal ~msg:"nu.b" (array float_exact) (twice st.nu.b)
    (Nx.to_array doubled.nu.b);
  equal ~msg:"step" int32 2l (Nx.item [] doubled.step)

(* Placement *)

(* Two leaves each alone on its own device of one set: every per-leaf step reads
   its scalars beside both and gives the host's values; the state's parts lie
   beside their leaves and the counter whole on both devices. *)
let test_leaves_on_two_devices () =
  let device name = Result.get_ok (Rig.software name) in
  let d0 = device "vega-test-0" and d1 = device "vega-test-1" in
  let module S = (val Nx.devices [ d0; d1 ]) in
  let host = (vec [| 1.0; -2.0 |], vec [| 3.0 |]) in
  let placed =
    ( Nx.place (Nx.Placement.device S.v d0) (fst host),
      Nx.place (Nx.Placement.device S.v d1) (snd host) )
  in
  let step p x =
    Vega.adam_step p ~lr:(Vega.rate 0.1) (Vega.adam_init p x) ~params:x ~grads:x
  in
  let (a, b), _ = step Nx.Ptree.(pair tensor tensor) host in
  let (a', b'), st = step Nx.Ptree.(pair tensor tensor) placed in
  equal ~msg:"first leaf" (array float_exact) (Nx.to_array a) (Nx.to_array a');
  equal ~msg:"second leaf" (array float_exact) (Nx.to_array b) (Nx.to_array b');
  let where x = Format.asprintf "%a" Nx.Placement.pp (Nx.placement x) in
  equal ~msg:"mu of the first leaf" string
    (where (fst placed))
    (where (fst st.mu));
  equal ~msg:"nu of the second leaf" string
    (where (snd placed))
    (where (snd st.nu));
  equal ~msg:"the counter" string
    (where (Nx.scalar_at S.on Nx.int32 0l))
    (where st.step)

let inits_refuse_no_array =
  cases "every init names itself on parameters of no array"
    ~id:(fun s -> s.name)
    steps
    (fun step ->
      raises
        (Invalid_argument (step.init ^ ": the parameters hold no array"))
        (fun () -> step.run 1 Nx.Ptree.unit () ()))

(* Gradient transformations *)

(* Gradients of two leaves, a float32 vector and a float64 one, of drawn
   magnitude, and a bound. *)
let gradients =
  let open Gen in
  let* scale = of_list [ 0.0; 1e-3; 1.0; 1e3 ] in
  let leaf = array ~size:(int_range 0 4) (float_range (-1.0) 1.0) in
  let+ a = leaf and+ b = leaf and+ max_norm = float_range 0.1 4.0 in
  let s = Array.map (fun x -> x *. scale) in
  ((Nx.create Nx.float32 [| Array.length a |] (s a), vec (s b)), max_norm)

let floats2 (a, b) = (Nx.to_array (Nx.cast Nx.float64 a), Nx.to_array b)

(* Gradients within the bound come back bit for bit. Larger ones are scaled by
   [max_norm / norm]: the norm, at float32, of at most eight elements is within
   [2^-20] of its float64 value relative, and the float32 leaf rounds once more,
   so each element is within [2^-18] of the float64 reference, relative, or a
   subnormal one within [2^-1000]. Within [2^-16] of the bound either may
   hold. *)
let clip_by_global_norm =
  prop "clip_by_global_norm scales gradients above max_norm to it" gradients
    (fun (g, max_norm) ->
      let p = Nx.Ptree.(pair tensor tensor) in
      let a, b = floats2 g in
      let norm =
        sqrt (Array.fold_left (fun s x -> s +. (x *. x)) 0.0 (Array.append a b))
      in
      let within = norm <= max_norm *. (1.0 -. 0x1p-16)
      and above = norm >= max_norm *. (1.0 +. 0x1p-16) in
      cover "all-zero gradients" (norm = 0.0);
      cover "gradients within the bound" within;
      cover "gradients above the bound" above;
      let clipped = floats2 (Vega.clip_by_global_norm p ~max_norm g) in
      if within then
        equal
          (Windtrap.pair (array float_exact) (array float_exact))
          (a, b) clipped;
      if above then
        let scaled = Array.map (fun x -> x *. max_norm /. norm) in
        equal
          (let near = array (float_rel ~rel:0x1p-18 ~abs:0x1p-1000) in
           Windtrap.pair near near)
          (scaled a, scaled b)
          clipped)

let test_clip_no_array () =
  equal unit () (Vega.clip_by_global_norm Nx.Ptree.unit ~max_norm:1.0 ())

(* Leaves of float32, float64, bfloat16 and complex64 elements, among them NaN,
   infinities and zeros of either sign, and a bound. *)
module Clipped = struct
  type t = {
    s : Nx.host Nx.float32_t;
    d : Nx.host Nx.float64_t;
    h : Nx.host Nx.bfloat16_t;
    z : Nx.host Nx.complex64_t;
  }

  let walk c { s; d; h; z } =
    let open Nx.Ptree.Walk in
    let s = field c "s" tensor s in
    let d = field c "d" tensor d in
    let h = field c "h" tensor h in
    let z = field c "z" tensor z in
    { s; d; h; z }

  let ptree : (t, Nx.host) Nx.Ptree.t = Nx.Ptree.v walk
end

let clip_draw =
  let open Gen in
  let value =
    frequency
      [
        (6, float_range (-4.0) 4.0);
        (1, of_list [ 0.0; -0.0; infinity; neg_infinity; nan ]);
      ]
  in
  let leaf = array ~size:(int_range 0 4) value in
  let+ s = leaf
  and+ d = leaf
  and+ h = leaf
  and+ re = leaf
  and+ im = leaf
  and+ max = of_list [ 0.5; 1.0; 1.005; 3.0 ] in
  let n = min (Array.length re) (Array.length im) in
  let z = Array.init n (fun i -> { Complex.re = re.(i); im = im.(i) }) in
  let v dt a = Nx.create dt [| Array.length a |] a in
  ( {
      Clipped.s = v Nx.float32 s;
      d = v Nx.float64 d;
      h = v Nx.bfloat16 h;
      z = Nx.create Nx.complex64 [| n |] z;
    },
    max )

(* The floats of a leaf, a complex one's parts. *)
let parts (x : Clipped.t) =
  let f t = Nx.to_array (Nx.cast Nx.float64 t) in
  [
    ("s", f x.s);
    ("d", f x.d);
    ("h", f x.h);
    ("z", Array.append (f (Nx.real Nx.float32 x.z)) (f (Nx.imag Nx.float32 x.z)));
  ]

(* Each element, each part of a complex one, clamped to [[-m; m]] for [m] the
   bound rounded to the leaf's dtype, or its parts'; a NaN stays NaN. *)
let clip_by_value =
  prop "clip_by_value clamps each element to the bound at its dtype" clip_draw
    (fun (g, max) ->
      let all = List.concat_map (fun (_, a) -> Array.to_list a) (parts g) in
      cover "a NaN element" (List.exists Float.is_nan all);
      cover "an infinite element"
        (List.exists (fun x -> Float.abs x = infinity) all);
      cover "an element clamped" (List.exists (fun x -> Float.abs x > max) all);
      cover "a bound bfloat16 rounds" (max = 1.005);
      let rounded dt = Nx.item [] (Nx.cast Nx.float64 (Nx.scalar dt max)) in
      let bound = function
        | "h" -> rounded Nx.bfloat16
        | "d" -> max
        | _ -> rounded Nx.float32
      in
      let clipped = Vega.clip_by_value Clipped.ptree ~max g in
      List.iter2
        (fun (name, a) (_, a') ->
          let m = bound name in
          let clamp x =
            if Float.is_nan x then x else Float.min m (Float.max (-.m) x)
          in
          equal ~msg:name (array float_exact) (Array.map clamp a) a')
        (parts g) (parts clipped))

(* Rates *)

let rate_is_constant =
  prop "Vega.rate is its number at every counter"
    Gen.(pair (float_range 0.0 10.0) (int_range 0 1_000_000))
    (fun (v, c) ->
      equal float_exact
        (Int32.float_of_bits (Int32.bits_of_float v))
        (Vega.Schedule.eval (Vega.rate v) c))

let test_rate_refuses () =
  raises (Invalid_argument "Vega.rate: expected v >= 0.0, got -0.1") (fun () ->
      Vega.rate (-0.1));
  raises (Invalid_argument "Vega.rate: expected v >= 0.0, got nan") (fun () ->
      Vega.rate Float.nan)

let test_clip_by_global_norm_carries () =
  let p = Nx.Ptree.(pair tensor tensor) in
  let grads =
    (Nx.create Nx.float32 [| 2 |] [| 3.0; 4.0 |], Nx.scalar Nx.int32 7l)
  in
  let clipped, counter = Vega.clip_by_global_norm p ~max_norm:1.0 grads in
  check_vec ~eps:1e-6 [| 0.6; 0.8 |] clipped;
  equal ~msg:"an int32 leaf is carried" int32 7l (Nx.item [] counter)

(* An integer, unsigned or boolean leaf, such as a counter or a mask carried in
   the gradients, is returned unchanged. *)
let test_clip_by_value_carries () =
  let p = Nx.Ptree.(quad tensor tensor tensor tensor) in
  let grads =
    ( Nx.create Nx.float32 [| 2 |] [| 3.0; -4.0 |],
      Nx.create Nx.int32 [| 2 |] [| 7l; -9l |],
      Nx.create Nx.uint8 [| 2 |] [| 0; 200 |],
      Nx.create Nx.bool [| 2 |] [| false; true |] )
  in
  let clipped, ints, bytes, mask = Vega.clip_by_value p ~max:1.0 grads in
  check_vec ~eps:0. [| 1.0; -1.0 |] clipped;
  equal ~msg:"int32" (array int32) [| 7l; -9l |] (Nx.to_array ints);
  equal ~msg:"uint8" (array int) [| 0; 200 |] (Nx.to_array bytes);
  equal ~msg:"bool" (array bool) [| false; true |] (Nx.to_array mask)

let test_clip_validation () =
  let grads = pair [| 1.0 |] [| 1.0 |] in
  raises
    (Invalid_argument "Vega.clip_by_global_norm: expected max_norm > 0.0, got 0")
    (fun () -> Vega.clip_by_global_norm Pair.ptree ~max_norm:0.0 grads);
  raises (Invalid_argument "Vega.clip_by_value: expected max > 0.0, got -1")
    (fun () -> Vega.clip_by_value Pair.ptree ~max:(-1.0) grads);
  (* The shortest decimal that reads back, the same on every system: the C
     library's [%g] wrote 1e+06, 1e+006 under mingw-w64. *)
  raises
    (Invalid_argument "Vega.clip_by_value: expected max > 0.0, got -1000000")
    (fun () -> Vega.clip_by_value Pair.ptree ~max:(-1e6) grads);
  raises
    (Invalid_argument "Vega.clip_by_value: expected max > 0.0, got -4.58e-72")
    (fun () -> Vega.clip_by_value Pair.ptree ~max:(-4.58e-72) grads);
  raises (Invalid_argument "Vega.clip_by_value: expected max > 0.0, got nan")
    (fun () -> Vega.clip_by_value Pair.ptree ~max:Float.nan grads)

(* Zero gradients isolate the decay: p_k = p_0 (1 - lr wd)^k. A coupled (L2)
   decay would instead be distorted by the adaptive scaling. [wd = 0.1] is not a
   float32 value, so the tolerance, far below float32's gap at [0.1], tells a
   float64 decay from [Vega.rate]'s float32 one. *)
let test_rate_cast () =
  let lr = 0.1 and wd = 0.1 in
  let p0 = [| 2.0; -4.0 |] in
  let run weight_decay =
    let params = ref (vec p0) in
    let st = ref (Vega.adamw_init Vec.ptree !params) in
    for _ = 1 to 3 do
      let params', st' =
        Vega.adamw_step Vec.ptree ~lr:(rate64 lr) ~weight_decay !st
          ~params:!params
          ~grads:(vec [| 0.0; 0.0 |])
      in
      params := params';
      st := st'
    done;
    !params
  in
  let decayed wd =
    let c = (1.0 -. (lr *. wd)) ** 3.0 in
    Array.map (fun p -> p *. c) p0
  in
  check_vec ~eps:1e-14 ~msg:"a float64 decay" (decayed wd) (run (rate64 wd));
  check_vec ~eps:1e-14 ~msg:"a float32 constant"
    (decayed (Int32.float_of_bits (Int32.bits_of_float wd)))
    (run (Vega.rate wd))

(* Weight decay *)

(* A leaf of each float dtype a step computes or stores apart. *)
module Mixed = struct
  type t = {
    h : Nx.host Nx.bfloat16_t;
    s : Nx.host Nx.float32_t;
    d : Nx.host Nx.float64_t;
    z : Nx.host Nx.complex64_t;
  }

  let walk c { h; s; d; z } =
    let open Nx.Ptree.Walk in
    let h = field c "h" tensor h in
    let s = field c "s" tensor s in
    let d = field c "d" tensor d in
    let z = field c "z" tensor z in
    { h; s; d; z }

  let ptree : (t, Nx.host) Nx.Ptree.t = Nx.Ptree.v walk
end

(* Leaves of [n] elements, among them zeros of either sign, infinities and
   NaN. *)
let mixed n =
  let open Gen in
  let value =
    frequency
      [
        (6, float_range (-4.0) 4.0);
        (1, of_list [ 0.0; -0.0; infinity; neg_infinity; nan ]);
      ]
  in
  let values = array ~size:(constant n) value in
  let+ h = values and+ s = values and+ d = values and+ re = values
  and+ im = values in
  let v dt xs = Nx.create dt [| n |] xs in
  let z = Array.map2 (fun re im -> { Complex.re; im }) re im in
  {
    Mixed.h = v Nx.bfloat16 h;
    s = v Nx.float32 s;
    d = v Nx.float64 d;
    z = Nx.create Nx.complex64 [| n |] z;
  }

let at_counter c = Nx.scalar Nx.int32 (Int32.of_int c)

(* The float leaves of [x], by path, a complex one's as its coordinates. *)
let by_path s x =
  let add path a acc =
    match floats_of a with
    | Some f -> (Nx.Ptree.Path.to_string path, f) :: acc
    | None -> acc
  in
  List.rev (Nx.Ptree.fold s add x [])

let leaves_equal = list (Windtrap.pair string (array float_exact))

(* Each step that decays, from its initial state at the counter [c], as the
   float leaves of the parameters and the state it returns. [weight_decay] is
   [None] for the step's default. *)
let decaying ?decays () =
  let p = Mixed.ptree and lr = Vega.rate 0.01 in
  let floats_of state x = by_path (Nx.Ptree.pair p (state p)) x in
  [
    ( "lars",
      fun weight_decay c x g ->
        let st = { (Vega.lars_init p x) with step = at_counter c } in
        floats_of Vega.sgd_ptree
          (Vega.lars_step p ~lr ?weight_decay ?decays st ~params:x ~grads:g) );
    ( "adamw",
      fun weight_decay c x g ->
        let st = { (Vega.adamw_init p x) with step = at_counter c } in
        floats_of Vega.adam_ptree
          (Vega.adamw_step p ~lr ?weight_decay ?decays st ~params:x ~grads:g)
    );
    ( "lamb",
      fun weight_decay c x g ->
        let st = { (Vega.lamb_init p x) with step = at_counter c } in
        floats_of Vega.adam_ptree
          (Vega.lamb_step p ~lr ?weight_decay ?decays st ~params:x ~grads:g) );
    ( "adan",
      fun weight_decay c x g ->
        let st = { (Vega.adan_init p x) with step = at_counter c } in
        floats_of Vega.adan_ptree
          (Vega.adan_step p ~lr ?weight_decay ?decays st ~params:x ~grads:g) );
  ]

(* A decay that rises over training, from 0.04 to 0.4 in 1000 steps. *)
let rising = Vega.Schedule.linear ~init_value:0.04 ~end_value:0.4 ~steps:1000

(* At every counter, a step under a scheduled decay is, bit for bit, the step
   under the constant decay of the schedule's value there. *)
let scheduled_decay =
  let gen =
    let open Gen in
    let* n = int_range 0 4 in
    triple (mixed n) (mixed n) (int_range 0 1200)
  in
  prop "a scheduled decay steps as its value at the counter" gen
    (fun (x, g, c) ->
      cover "an empty leaf" (Nx.numel x.Mixed.s = 0);
      cover "past the schedule's end" (c > 1000);
      let constant = Vega.rate (Vega.Schedule.eval rising c) in
      List.iter
        (fun (name, step) ->
          equal ~msg:name leaves_equal
            (step (Some constant) c x g)
            (step (Some rising) c x g))
        (decaying ()))

(* Each step decays by the rate it is passed: at the counter 500, the rising
   schedule's [0.22] moves the leaves apart from its start, [0.04], and a
   passed [0.3] apart from the default. *)
let passed_decay =
  let x =
    let xs = [| 1.5; -2.0; 0.75 |] in
    let v dt = Nx.create dt [| 3 |] xs in
    let z = Array.map (fun re -> { Complex.re; im = -.re }) xs in
    {
      Mixed.h = v Nx.bfloat16;
      s = v Nx.float32;
      d = v Nx.float64;
      z = Nx.create Nx.complex64 [| 3 |] z;
    }
  in
  let g = Nx.Ptree.map Mixed.ptree (fun _ a -> Nx.add a a) x in
  cases "each step decays by the rate it is passed" ~id:fst (decaying ())
    (fun (name, step) ->
      not_equal ~msg:(name ^ ": a schedule against its start") leaves_equal
        (step (Some (Vega.rate 0.04)) 500 x g)
        (step (Some rising) 500 x g);
      not_equal ~msg:(name ^ ": a passed decay against the default")
        leaves_equal (step None 500 x g)
        (step (Some (Vega.rate 0.3)) 500 x g))

let mixed_step =
  let open Gen in
  let* n = int_range 0 4 in
  triple (mixed n) (mixed n) (int_range 0 1200)

let rejects _ _ = false

(* A leaf the rule rejects steps under AdamW as under Adam, bit for bit, an
   infinite parameter included. *)
let rejected_is_adam =
  prop "a leaf that does not decay steps as adam" mixed_step (fun (x, g, c) ->
      let infinite a = Array.exists (fun v -> Float.abs v = infinity) a in
      cover "an infinite parameter"
        (List.exists (fun (_, a) -> infinite a) (by_path Mixed.ptree x));
      let p = Mixed.ptree and lr = Vega.rate 0.01 in
      let st = { (Vega.adam_init p x) with step = at_counter c } in
      let both = Nx.Ptree.pair p (Vega.adam_ptree p) in
      equal leaves_equal
        (by_path both (Vega.adam_step p ~lr st ~params:x ~grads:g))
        (by_path both
           (Vega.adamw_step p ~lr ~weight_decay:rising ~decays:rejects st
              ~params:x ~grads:g)))

(* A leaf the rule rejects does not see the decay, in a trust ratio either. *)
let rejected_ignores_the_decay =
  prop "a leaf that does not decay steps alike under any decay" mixed_step
    (fun (x, g, c) ->
      List.iter
        (fun (name, step) ->
          equal ~msg:name leaves_equal
            (step (Some (Vega.rate 0.3)) c x g)
            (step (Some rising) c x g))
        (decaying ~decays:rejects ()))

(* Under a rule that picks some of the leaves by name, each leaf of the
   parameters and the state is the leaf of the step where every leaf decays,
   or of the step where none does: the rule's answer for its parameter. The
   last field of a path names the parameter, [step] the counter. *)
let mixed_rule =
  let gen = Gen.(pair mixed_step (array ~size:(constant 4) bool)) in
  prop "a rule decays the leaves it picks" gen (fun ((x, g, c), picked) ->
      cover "some leaves decay and some do not"
        (Array.exists Fun.id picked && Array.exists not picked);
      let picks = List.combine [ "h"; "s"; "d"; "z" ] (Array.to_list picked) in
      let chosen path =
        match List.rev (String.split_on_char '.' path) with
        | name :: _ -> Option.value ~default:false (List.assoc_opt name picks)
        | [] -> false
      in
      let decays path _ = chosen (Nx.Ptree.Path.to_string path) in
      let expected all none =
        List.map2
          (fun ((path, _) as a) b -> if chosen path then a else b)
          all none
      in
      let bits =
        List.map (fun (p, a) -> (p, Array.map Int64.bits_of_float a))
      in
      List.iter2
        (fun (name, mixed) ((_, all), (_, none)) ->
          let all = all (Some rising) c x g
          and none = none (Some rising) c x g in
          cover (name ^ ": the decay moves a leaf") (bits all <> bits none);
          equal ~msg:name leaves_equal (expected all none)
            (mixed (Some rising) c x g))
        (decaying ~decays ())
        (List.combine
           (decaying ~decays:(fun _ _ -> true) ())
           (decaying ~decays:rejects ())))

(* A rule that records what it is asked, and what it was asked, sorted. *)
let counting () =
  let asked = ref [] in
  let rule path shape =
    asked := (Nx.Ptree.Path.to_string path, Array.to_list shape) :: !asked;
    true
  in
  (rule, fun () -> List.sort compare !asked)

(* Each rule of each step that takes one is asked once per float or complex
   leaf, with the parameter's own shape, alone or beside the step's other
   rule, and never for an integer leaf. A rule given alone is asked once per
   leaf: the step asks nothing in place of the rule it was not given. *)
let rules_are_asked =
  let x =
    ( {
        Mixed.h = Nx.ones Nx.bfloat16 [| 3 |];
        s = Nx.ones Nx.float32 [| 2; 3 |];
        d = Nx.ones Nx.float64 [||];
        z = Nx.ones Nx.complex64 [| 1; 3 |];
      },
      Nx.zeros Nx.int32 [| 4 |] )
  in
  let p = Nx.Ptree.pair Mixed.ptree Nx.Ptree.tensor in
  let lr = Vega.rate 0.1 in
  let lars ?decays ?adapts () =
    ignore
      (Vega.lars_step p ~lr ?decays ?adapts (Vega.lars_init p x) ~params:x
         ~grads:x)
  and lamb ?decays ?adapts () =
    ignore
      (Vega.lamb_step p ~lr ?decays ?adapts (Vega.lamb_init p x) ~params:x
         ~grads:x)
  in
  let one step =
    let rule, asked = counting () in
    step rule;
    [ asked () ]
  and both step =
    let decays, decayed = counting () and adapts, adapted = counting () in
    step ~decays ~adapts;
    [ decayed (); adapted () ]
  in
  let rows =
    [
      ("lars ~decays", fun () -> one (fun decays -> lars ~decays ()));
      ("lars ~adapts", fun () -> one (fun adapts -> lars ~adapts ()));
      ( "lars, both rules",
        fun () -> both (fun ~decays ~adapts -> lars ~decays ~adapts ()) );
      ( "adamw ~decays",
        fun () ->
          one (fun decays ->
              ignore
                (Vega.adamw_step p ~lr ~decays (Vega.adamw_init p x) ~params:x
                   ~grads:x)) );
      ("lamb ~decays", fun () -> one (fun decays -> lamb ~decays ()));
      ("lamb ~adapts", fun () -> one (fun adapts -> lamb ~adapts ()));
      ( "lamb, both rules",
        fun () -> both (fun ~decays ~adapts -> lamb ~decays ~adapts ()) );
      ( "adan ~decays",
        fun () ->
          one (fun decays ->
              ignore
                (Vega.adan_step p ~lr ~decays (Vega.adan_init p x) ~params:x
                   ~grads:x)) );
    ]
  in
  let expected =
    [ ("0.d", []); ("0.h", [ 3 ]); ("0.s", [ 2; 3 ]); ("0.z", [ 1; 3 ]) ]
  in
  cases "each rule is asked once per float or complex leaf" ~id:fst rows
    (fun (_, run) ->
      List.iter
        (equal (list (Windtrap.pair string (list int))) expected)
        (run ()))

(* The trust ratio's rule *)

(* LARS and LAMB from a state at the counter [c] whose parts are not zero, as
   the float leaves of the parameters and the state they return. *)
let adapting ?adapts ?decays () =
  let p = Mixed.ptree and lr = Vega.rate 0.01 in
  let square = Nx.Ptree.map p (fun _ a -> Nx.mul a a) in
  [
    ( "lars",
      fun c x g ->
        let st = { Vega.velocity = g; step = at_counter c } in
        by_path
          (Nx.Ptree.pair p (Vega.sgd_ptree p))
          (Vega.lars_step p ~lr ~weight_decay:rising ?decays ?adapts st
             ~params:x ~grads:g) );
    ( "lamb",
      fun c x g ->
        let st = { Vega.mu = g; nu = square g; step = at_counter c } in
        by_path
          (Nx.Ptree.pair p (Vega.adam_ptree p))
          (Vega.lamb_step p ~lr ~weight_decay:rising ?decays ?adapts st
             ~params:x ~grads:g) );
  ]

(* A leaf the rule rejects takes no trust ratio: under LAMB it steps as under
   AdamW with the same decay, and under LARS, not decayed, as under momentum
   SGD, bit for bit. *)
let rejected_is_the_base =
  prop "a leaf that does not adapt steps as the base optimizer" mixed_step
    (fun (x, g, c) ->
      let p = Mixed.ptree and lr = Vega.rate 0.01 in
      let decays path _ = Nx.Ptree.Path.to_string path <> "0.s" in
      let adamw =
        let nu = Nx.Ptree.map p (fun _ a -> Nx.mul a a) g in
        let st = { Vega.mu = g; nu; step = at_counter c } in
        by_path
          (Nx.Ptree.pair p (Vega.adam_ptree p))
          (Vega.adamw_step p ~lr ~weight_decay:rising ~decays st ~params:x
             ~grads:g)
      in
      equal ~msg:"lamb" leaves_equal adamw
        (List.assoc "lamb" (adapting ~adapts:rejects ~decays ()) c x g);
      let sgd =
        let st = { Vega.velocity = g; step = at_counter c } in
        by_path
          (Nx.Ptree.pair p (Vega.sgd_ptree p))
          (Vega.sgd_step p ~lr ~momentum:0.9 st ~params:x ~grads:g)
      in
      equal ~msg:"lars" leaves_equal sgd
        (List.assoc "lars" (adapting ~adapts:rejects ~decays:rejects ()) c x g))

(* A float32 and a float64 leaf: LARS's closed form computes at each. *)
module Sd = struct
  type t = { s : Nx.host Nx.float32_t; d : Nx.host Nx.float64_t }

  let walk c { s; d } =
    let open Nx.Ptree.Walk in
    let s = field c "s" tensor s in
    let d = field c "d" tensor d in
    { s; d }

  let ptree : (t, Nx.host) Nx.Ptree.t = Nx.Ptree.v walk
end

(* [lars_rejected ~decays ~nesterov p g v] is lars_step's closed form at
   [r = 1] (its doc's), at the leaf's dtype, with the float32 rates [lr] and
   [wd] and [momentum] at the leaf's: [u = g + wd * p] for a leaf that
   decays and [g] for one that does not, [v' = momentum * v + u], and [p' =
   p - lr * v'], or [p - lr * (u + momentum * v')] with [nesterov]. *)
let lars_rejected ~lr ~wd ~momentum ~decays ~nesterov p g v =
  let f32 x = Int32.float_of_bits (Int32.bits_of_float x) in
  let c x = Nx.scalar_like p x in
  let u = if decays then Nx.add g (Nx.mul p (c (f32 wd))) else g in
  let v' = Nx.add (Nx.mul v (c momentum)) u in
  let d = if nesterov then Nx.add u (Nx.mul v' (c momentum)) else v' in
  (Nx.sub p (Nx.mul d (c (f32 lr))), v')

(* A leaf LARS's rule rejects follows the closed form at [r = 1], decayed or
   not, with Nesterov's form or without, bit for bit: the rule for the ratio
   leaves the decay and the momentum as they are. *)
let rejected_lars_closed_form =
  let leaves n =
    let open Gen in
    let+ xs = array ~size:(constant n) (float_range (-4.0) 4.0) in
    {
      Sd.s = Nx.create Nx.float32 [| n |] xs;
      d = Nx.create Nx.float64 [| n |] xs;
    }
  in
  let gen =
    let open Gen in
    let* n = int_range 0 4 in
    let+ (x, g, v), (decays, nesterov) =
      pair (triple (leaves n) (leaves n) (leaves n)) (pair bool bool)
    in
    (x, g, v, decays, nesterov)
  in
  prop "a leaf that does not adapt follows lars's closed form at r = 1" gen
    (fun (x, g, v, decays, nesterov) ->
      cover "decayed, with nesterov" (decays && nesterov);
      cover "neither" ((not decays) && not nesterov);
      let lr = 0.01 and wd = 0.3 and momentum = 0.9 in
      let p = Sd.ptree in
      let rule = if decays then None else Some rejects in
      let st = { Vega.velocity = v; step = at_counter 3 } in
      let x', st' =
        Vega.lars_step p ~lr:(Vega.rate lr) ~momentum
          ~weight_decay:(Vega.rate wd) ?decays:rule ~adapts:rejects ~nesterov
          st ~params:x ~grads:g
      in
      let closed p g v =
        lars_rejected ~lr ~wd ~momentum ~decays ~nesterov p g v
      in
      let bits t =
        Array.map Int64.bits_of_float (Nx.to_array (Nx.cast Nx.float64 t))
      in
      let check name (y, w) (y', w') =
        equal ~msg:(name ^ ": parameter") (array int64) (bits y) (bits y');
        equal ~msg:(name ^ ": velocity") (array int64) (bits w) (bits w')
      in
      check "float32" (closed x.s g.s v.s) (x'.s, st'.velocity.s);
      check "float64" (closed x.d g.d v.d) (x'.d, st'.velocity.d))

(* Under a rule that picks some of the leaves by name, each leaf of the
   parameters and the state is the leaf of the step where every leaf adapts,
   or of the step where none does, whatever the rule for the decay. *)
let mixed_adapts =
  let gen = Gen.(pair mixed_step (array ~size:(constant 4) bool)) in
  prop "a rule adapts the leaves it picks" gen (fun ((x, g, c), picked) ->
      cover "some leaves adapt and some do not"
        (Array.exists Fun.id picked && Array.exists not picked);
      let picks = List.combine [ "h"; "s"; "d"; "z" ] (Array.to_list picked) in
      let chosen path =
        match List.rev (String.split_on_char '.' path) with
        | name :: _ -> Option.value ~default:false (List.assoc_opt name picks)
        | [] -> false
      in
      let adapts path _ = chosen (Nx.Ptree.Path.to_string path) in
      let decays path _ = Nx.Ptree.Path.to_string path <> "0.d" in
      List.iter2
        (fun (name, mixed) ((_, all), (_, none)) ->
          let all = all c x g and none = none c x g in
          let bits =
            List.map (fun (p, a) -> (p, Array.map Int64.bits_of_float a))
          in
          cover (name ^ ": the ratio moves a leaf") (bits all <> bits none);
          let expected =
            List.map2
              (fun ((path, _) as a) b -> if chosen path then a else b)
              all none
          in
          equal ~msg:name leaves_equal expected (mixed c x g))
        (adapting ~adapts ~decays ())
        (List.combine
           (adapting ~decays ())
           (adapting ~adapts:rejects ~decays ())))

(* Parts by the kind of leaf *)

(* Two matrices, two vectors and a head at the parameter's positions, and a
   fixed counter, which every part holds. *)
module Net = struct
  type ('a, 'd) t = {
    w : 'a list;
    b : 'a list;
    head : 'a;
    seen : 'd Nx.int32_t;
  }

  let walk c { w; b; head; seen } =
    let open Nx.Ptree.Walk in
    let w = field c "w" (list leaf) w in
    let b = field c "b" (list leaf) b in
    let head = field c "head" leaf head in
    let seen = field c "seen" tensor seen in
    { w; b; head; seen }
end

let net_paths = [ "w.0"; "w.1"; "b.0"; "b.1"; "head" ]
let net_shapes = [ [| 2; 3 |]; [| 3; 2 |]; [| 3 |]; [| 2 |]; [| 1; 4 |] ]

(* A [Net] of leaves at [dt] from one array of values per position. *)
let net dt values =
  match List.map2 (Nx.create dt) net_shapes values with
  | [ w0; w1; b0; b1; head ] ->
      let seen = Nx.create Nx.int32 [| 2 |] [| 7l; 11l |] in
      { Net.w = [ w0; w1 ]; b = [ b0; b1 ]; head; seen }
  | _ -> invalid_arg "net: expected an array of values per position"

let net_values =
  let open Gen in
  let leaf s =
    array ~size:(constant (Array.fold_left ( * ) 1 s)) (float_range (-4.0) 4.0)
  in
  List.fold_right
    (fun s rest ->
      let+ v = leaf s and+ vs = rest in
      v :: vs)
    net_shapes (constant [])

(* [only kind k x] is the part [k] of [x], [None] at the other positions,
   and [over part x] is [x] with [part]'s arrays laid over it. *)
let only kind k x =
  Nx.Ptree.Payload.map
    (module Net)
    (fun path a -> if kind path = k then Some a else None)
    x

let over part x =
  Nx.Ptree.Payload.map2
    (module Net)
    (fun _ a x -> Option.value a ~default:x)
    part x

(* The arrays two parts hold between them, each position's from the part
   that holds it. *)
let either a b =
  Nx.Ptree.Payload.map2
    (module Net)
    (fun path a b ->
      match (a, b) with
      | Some x, None | None, Some x -> x
      | Some _, Some _ | None, None ->
          failf "%a is in both parts or in neither" Nx.Ptree.Path.pp path)
    a b

let rec repeat k step x = if k = 0 then x else repeat (k - 1) step (step x)

(* Three steps over the model cut in two parts by a drawn function from a
   leaf's path to its part, each part with its own state, the stepped parts
   laid back over the parameters: the parameters and the states' arrays have
   the bits of three steps over the whole model. At bfloat16 the stores'
   draws agree too, since a part keeps the model's paths. A part of no
   parameter still holds the fixed counter. *)
let parts_by_kind (type b) (dt : (float, b) Nx.dtype) =
  let model :
      (((float, b, Nx.host) Nx.t, Nx.host) Net.t, Nx.host) Nx.Ptree.t =
    Nx.Ptree.instantiate (module Net)
  in
  let part :
      (((float, b, Nx.host) Nx.t option, Nx.host) Net.t, Nx.host) Nx.Ptree.t =
    Nx.Ptree.nest (module Net) Nx.Ptree.(option tensor)
  in
  let lr = Vega.rate 0.05 in
  let gen =
    let open Gen in
    let n = List.length net_paths in
    let picked =
      frequency
        [ (1, constant (Array.make n true)); (5, array ~size:(constant n) bool) ]
    in
    triple net_values net_values picked
  in
  prop "a step over two parts, merged, is the step over the model" gen
    (fun (x, g, picked) ->
      cover "each part holds a parameter"
        (Array.exists Fun.id picked && Array.exists not picked);
      cover "one part holds every parameter" (Array.for_all Fun.id picked);
      let x = net dt x and g = net dt g in
      let picks = List.combine net_paths (Array.to_list picked) in
      let kind path = List.assoc (Nx.Ptree.Path.to_string path) picks in
      let only k = only kind k in
      let same msg whole split =
        equal ~msg leaves_equal (by_path model whole) (by_path model split)
      in
      (* [split init step] steps each part with its own state. *)
      let both init = (init part (only true x), init part (only false x)) in
      let split step (x, (a, b)) =
        let xa, a = step a ~params:(only true x) ~grads:(only true g) in
        let xb, b = step b ~params:(only false x) ~grads:(only false g) in
        (over xa (over xb x), (a, b))
      in
      let whole, st =
        repeat 3
          (fun (x, st) -> Vega.adamw_step model ~lr st ~params:x ~grads:g)
          (x, Vega.adamw_init model x)
      in
      let merged, (a, b) =
        repeat 3
          (split (fun st -> Vega.adamw_step part ~lr st))
          (x, both Vega.adamw_init)
      in
      same "adamw: parameters" whole merged;
      same "adamw: mu" st.mu (either a.mu b.mu);
      same "adamw: nu" st.nu (either a.nu b.nu);
      let whole, st =
        repeat 3
          (fun (x, st) ->
            Vega.sgd_step model ~lr ~momentum:0.9 st ~params:x ~grads:g)
          (x, Vega.sgd_init model x)
      in
      let merged, (a, b) =
        repeat 3
          (split (fun st -> Vega.sgd_step part ~lr ~momentum:0.9 st))
          (x, both Vega.sgd_init)
      in
      same "sgd: parameters" whole merged;
      same "sgd: velocity" st.velocity (either a.velocity b.velocity))

(* A state made over one part and stepped over the other raises at the first
   position the two parts hold apart. *)
let test_part_state_over_another () =
  let part :
      ((Nx.host Nx.float32_t option, Nx.host) Net.t, Nx.host) Nx.Ptree.t =
    Nx.Ptree.nest (module Net) Nx.Ptree.(option tensor)
  in
  let ones s = Array.make (Array.fold_left ( * ) 1 s) 1.0 in
  let x = net Nx.float32 (List.map ones net_shapes) in
  let kind path = Nx.Ptree.Path.to_string path <> "b.0" in
  let only k = only kind k in
  let lr = Vega.rate 0.1 in
  raises
    (Invalid_argument
       "Vega.adamw_step: w.0: Some in mu, None in the parameters") (fun () ->
      Vega.adamw_step part ~lr
        (Vega.adamw_init part (only true x))
        ~params:(only false x) ~grads:(only false x));
  raises
    (Invalid_argument
       "Vega.sgd_step: w.0: None in the velocity, Some in the parameters")
    (fun () ->
      Vega.sgd_step part ~lr ~momentum:0.9
        (Vega.sgd_init part (only false x))
        ~params:(only true x) ~grads:(only true x))

let () =
  exit
    (run "vega"
       [
         group "parameters and states"
           [
             group "one skeleton"
               [
                 one_skeleton;
                 test "a state's part, a path and a dtype name the mismatch"
                   test_steps_check_the_skeleton;
                 test "an L-BFGS state's parts have the parameters' dtypes"
                   test_lbfgs_checks_the_state_dtypes;
                 test "a mismatch raises before any leaf is stepped"
                   test_raises_before_stepping;
               ];
             group "non-parameter leaves" [ non_parameter_leaves ];
             group "purity" [ arguments_unchanged ];
             group "inits" [ inits_are_zero ];
             group "complex leaves"
               [
                 test "sgd moves z to z - lr * conj g" test_complex_sgd;
                 group "complex64" [ as_coordinates Nx.complex64 Nx.float32 ];
                 group "complex128" [ as_coordinates Nx.complex128 Nx.float64 ];
               ];
             group "states are structures"
               [
                 state_visits;
                 test "a state reports what its parameters report"
                   test_state_reports;
                 test "a map over a state reaches each part where it lies"
                   test_state_maps;
               ];
             group "placement"
               [
                 test "a state lies with leaves alone on two devices"
                   test_leaves_on_two_devices;
                 inits_refuse_no_array;
               ];
           ];
         group "gradient transformations"
           [
             clip_by_global_norm;
             test "clip_by_global_norm returns a structure of no array"
               test_clip_no_array;
             test "clip_by_global_norm carries a leaf that is not a float"
               test_clip_by_global_norm_carries;
             clip_by_value;
             test "clip_by_value carries a leaf that is not a float"
               test_clip_by_value_carries;
             test "clipping refuses a bound that is not positive"
               test_clip_validation;
           ];
         group "rates"
           [
             rate_is_constant;
             test "Vega.rate refuses a negative or NaN number" test_rate_refuses;
             test "a step casts its rate to the leaf's dtype" test_rate_cast;
           ];
         group "weight decay"
           [
             scheduled_decay;
             passed_decay;
             rejected_is_adam;
             rejected_ignores_the_decay;
             mixed_rule;
             rules_are_asked;
           ];
         group "trust ratio"
           [ rejected_is_the_base; rejected_lars_closed_form; mixed_adapts ];
         group "parts by the kind of leaf"
           [
             group "float32" [ parts_by_kind Nx.float32 ];
             group "bfloat16" [ parts_by_kind Nx.bfloat16 ];
             test "a state stepped over another part raises"
               test_part_state_over_another;
           ];
       ])
