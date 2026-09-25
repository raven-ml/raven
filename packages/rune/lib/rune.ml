(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let require_scalar name y =
  if Nx.numel y <> 1 then
    invalid_arg
      (Printf.sprintf
         "%s: the objective must return a scalar tensor, got shape [%s]; use \
          vjp for non-scalar outputs"
         name
         (Structure.shape_string (Nx.shape y)))

(* Install a transformation handler for the run of [f]. The recorded depth lets
   [jit] step aside when a transformation is observing the operations. *)
let run_transform f x handler =
  Gate.with_transform (fun () -> Effect.Deep.match_with f x handler)

(* Gradients are defined with respect to real and complex leaves. A parameter
   structure may hold others — an RNG key threaded through a compiled step, a
   step counter, a batch of indices — and they ride along rather than being
   differentiated: untracked, so nothing accumulates into them, and zero in the
   gradient structure they return. An optimizer is then responsible for leaving
   them alone; see Vega. *)
let differentiable_leaf leaf =
  let dt = Nx.dtype leaf in
  Nx_dtype.is_float dt || Nx_dtype.is_complex dt

(* The single-tensor forms have no structure to carry a non-differentiable value
   in, so there the dtype is simply wrong. *)
let require_float_leaf name leaf =
  if not (differentiable_leaf leaf) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot differentiate a %s tensor; gradients are defined for real \
          and complex dtypes"
         name
         (Nx_dtype.to_string (Nx.dtype leaf)))

(* Reverse mode *)

(* The leaves of [params], aliased and tracked on a fresh tape: an alias per
   leaf makes a tensor behind two leaves two parameters, and a capture that is
   the same value as a leaf a constant. *)
let tracked_params p params =
  let params = Structure.aliases p params in
  let tape = Tape.create () in
  Nx.Ptree.fold p
    (fun _ leaf () -> if differentiable_leaf leaf then Tape.track tape leaf)
    params ();
  (tape, params)

let cotangents tape p params =
  Nx.Ptree.map p (fun _ leaf -> Tape.cotangent tape leaf) params

let value_and_grad p f params =
  let tape, params = tracked_params p params in
  let y = run_transform f params (Reverse.handler tape) in
  require_scalar "Rune.value_and_grad" y;
  Tape.accumulate tape y (Nx.ones_like y);
  Tape.backward tape;
  (y, cotangents tape p params)

let grad p f params = snd (value_and_grad p f params)

let value_and_grad_aux p f params =
  let aux = ref None in
  let f' ps =
    let y, a = f ps in
    aux := Some a;
    y
  in
  let y, grads = value_and_grad p f' params in
  match !aux with
  | Some a -> (y, grads, a)
  | None -> assert false (* [f'] completed, so [aux] was set. *)

(* Seed each result leaf with its cotangent, checking the cotangents against the
   result. *)
let seed fn tape q y cts =
  ignore
    (Structure.map2 fn q ~this:"the result" ~that:"the cotangents"
       (fun path yl ct ->
         if Nx.shape yl <> Nx.shape ct then
           invalid_arg
             (Printf.sprintf
                "%s: %s: cotangent shape [%s] does not match result shape [%s]"
                fn (Structure.describe path)
                (Structure.shape_string (Nx.shape ct))
                (Structure.shape_string (Nx.shape yl)));
         Tape.accumulate tape yl ct;
         yl)
       y cts)

let vjp p q f params cts =
  let tape, params = tracked_params p params in
  let y = run_transform f params (Reverse.handler tape) in
  seed "Rune.vjp" tape q y cts;
  Tape.backward tape;
  (y, cotangents tape p params)

let vjp_fun p q f params =
  let tape, params = tracked_params p params in
  let y = run_transform f params (Reverse.handler tape) in
  let pullback cts =
    Tape.reset_cotangents tape;
    seed "Rune.vjp_fun" tape q y cts;
    Tape.backward tape;
    cotangents tape p params
  in
  (y, pullback)

(* Forward mode *)

let output_tangent store y =
  match Tensor_map.find store y with Some dy -> dy | None -> Nx.zeros_like y

(* [f params] under the forward handler, with each leaf of [params] aliased and
   seeded with its tangent. *)
let run_forward fn p f params tangents =
  let params = Structure.aliases p params in
  let store = Tensor_map.create () in
  ignore
    (Structure.map2 fn p ~this:"the parameters" ~that:"the tangents"
       (fun path leaf tangent ->
         if Nx.shape leaf <> Nx.shape tangent then
           invalid_arg
             (Printf.sprintf
                "%s: %s: tangent shape [%s] does not match parameter shape [%s]"
                fn (Structure.describe path)
                (Structure.shape_string (Nx.shape tangent))
                (Structure.shape_string (Nx.shape leaf)));
         Tensor_map.set store leaf tangent;
         leaf)
       params tangents);
  (store, run_transform f params (Forward.handler store))

let jvp p q f params tangents =
  let store, y = run_forward "Rune.jvp" p f params tangents in
  (y, Nx.Ptree.map q (fun _ yl -> output_tangent store yl) y)

let jvp_aux p q f params tangents =
  let aux = ref None in
  let f' ps =
    let y, a = f ps in
    aux := Some a;
    y
  in
  let store, y = run_forward "Rune.jvp_aux" p f' params tangents in
  match !aux with
  | Some a -> (y, Nx.Ptree.map q (fun _ yl -> output_tangent store yl) y, a)
  | None -> assert false (* [f'] completed, so [aux] was set. *)

(* Custom differentiation rules *)

let custom_vjp = Custom.custom_vjp
let custom_jvp = Custom.custom_jvp

(* Vectorizing maps *)

let broadcast_output st y =
  if Vmap.batched st y then y else Vmap.ensure_batched st y

let vmap fn =
  let (Structure.Uncurried u) = Structure.uncurry "Rune.vmap" fn in
  fun f ->
    u.curry (fun args ->
        let batch = ref None in
        Nx.Ptree.fold u.args
          (fun path leaf () ->
            let at () = Nx.Ptree.Path.to_string path in
            match (Nx.shape leaf, !batch) with
            | [||], _ ->
                invalid_arg
                  (Printf.sprintf
                     "Rune.vmap: %s: a scalar; vmap maps axis 0 of every leaf"
                     (at ()))
            | shape, None -> batch := Some (shape.(0), at ())
            | shape, Some (n, first) ->
                if shape.(0) <> n then
                  invalid_arg
                    (Printf.sprintf
                       "Rune.vmap: %s: %d rows along axis 0, %s: %d" (at ())
                       shape.(0) first n))
          args ();
        let batch_size =
          match !batch with
          | Some (n, _) -> n
          | None -> invalid_arg "Rune.vmap: the arguments have no leaf to map"
        in
        let st = Vmap.create ~batch_size in
        let args =
          Nx.Ptree.map u.args
            (fun _ leaf ->
              let leaf = Structure.alias leaf in
              Vmap.mark st leaf;
              leaf)
            args
        in
        let y = run_transform (u.apply f) args (Vmap.handler st) in
        Nx.Ptree.map u.result (fun _ yl -> broadcast_output st yl) y)

let vmap' f x =
  if Array.length (Nx.shape x) = 0 then
    invalid_arg "Rune.vmap': cannot map a scalar";
  let x = Structure.alias x in
  let st = Vmap.create ~batch_size:(Nx.shape x).(0) in
  Vmap.mark st x;
  broadcast_output st (run_transform f x (Vmap.handler st))

(* Single-tensor variants *)

let tracked_tensor x =
  let x = Structure.alias x in
  let tape = Tape.create () in
  Tape.track tape x;
  (tape, x)

let run_reverse' f x ~seed =
  let tape, x = tracked_tensor x in
  let y = run_transform f x (Reverse.handler tape) in
  Tape.accumulate tape y (seed y);
  Tape.backward tape;
  (y, Tape.cotangent tape x)

let value_and_grad' f x =
  require_float_leaf "Rune.value_and_grad'" x;
  run_reverse' f x ~seed:(fun y ->
      require_scalar "Rune.value_and_grad'" y;
      Nx.ones_like y)

let grad' f x = snd (value_and_grad' f x)
let vjp' f x cotangent = run_reverse' f x ~seed:(fun _ -> cotangent)

let vjp_fun' f x =
  let tape, x = tracked_tensor x in
  let y = run_transform f x (Reverse.handler tape) in
  let pullback ct =
    Tape.reset_cotangents tape;
    Tape.accumulate tape y ct;
    Tape.backward tape;
    Tape.cotangent tape x
  in
  (y, pullback)

let jvp' f x tangent =
  if Nx.shape x <> Nx.shape tangent then
    invalid_arg
      (Printf.sprintf
         "Rune.jvp': tangent shape [%s] does not match parameter shape [%s]"
         (Structure.shape_string (Nx.shape tangent))
         (Structure.shape_string (Nx.shape x)));
  let x = Structure.alias x in
  let store = Tensor_map.create () in
  Tensor_map.set store x tangent;
  let y = run_transform f x (Forward.handler store) in
  (y, output_tangent store y)

(* Gradient checkpointing *)

let remat fn =
  let (Structure.Uncurried u) = Structure.uncurry "Rune.remat" fn in
  fun f ->
    u.curry (fun params ->
        Remat.run
          (Remat.Call
             {
               params_s = u.args;
               result_s = u.result;
               params;
               f = u.apply f;
               residuals = false;
             }))

(* Jacobians *)

(* [basis_like y] is a [numel y; shape y...] tensor whose k-th slice is the k-th
   standard basis element of [y]'s space. *)
let basis_like (type a b) (y : (a, b) Nx.t) : (a, b) Nx.t =
  let n = Nx.numel y in
  Nx.reshape (Array.append [| n |] (Nx.shape y)) (Nx.eye (Nx.dtype y) n)

let jacrev' (type a b c d) (f : (a, b) Nx.t -> (c, d) Nx.t) (x : (a, b) Nx.t) :
    (a, b) Nx.t =
  (* [vjp_fun'] returns the primal output and records its reusable pullback in
     the same forward pass. Derive the row basis from that output rather than
     evaluating [f] separately for its shape. *)
  let y, pullback = vjp_fun' f x in
  let rows = vmap' pullback (basis_like y) in
  Nx.reshape (Array.append (Nx.shape y) (Nx.shape x)) (Nx.contiguous rows)

let jacfwd' (type a b c d) (f : (a, b) Nx.t -> (c, d) Nx.t) (x : (a, b) Nx.t) :
    (c, d) Nx.t =
  let cols = vmap' (fun v -> snd (jvp' f x v)) (basis_like x) in
  (* [cols] is [numel x; shape y...], so it supplies the output shape without a
     separate evaluation of [f]. Move its input axis last. *)
  let cols_shape = Nx.shape cols in
  let rank = Array.length cols_shape in
  let cols = Nx.moveaxis 0 (rank - 1) cols in
  let y_shape = Array.sub cols_shape 1 (rank - 1) in
  Nx.reshape (Array.append y_shape (Nx.shape x)) (Nx.contiguous cols)

let hessian' (type a b) (f : (a, b) Nx.t -> (a, b) Nx.t) (x : (a, b) Nx.t) :
    (a, b) Nx.t =
  jacfwd' (grad' f) x

let hvp p f params v =
  let store, g = run_forward "Rune.hvp" p (grad p f) params v in
  Nx.Ptree.map p (fun _ gl -> output_tangent store gl) g

let hvp' f x v = snd (jvp' (grad' f) x v)

(* Gradient checking *)

let check_grads ?(eps = 1e-4) ?(tol = 1e-2) p f params =
  let scalar_f64 t = Nx.item [] (Nx.reshape [||] (Nx.cast Nx.float64 t)) in
  let g = grad p f params in
  (* Two deterministic directions: all-ones, and a params-derived direction so
     the two are independent for non-constant params. *)
  let directions =
    [
      ("ones", Nx.Ptree.map p (fun _ leaf -> Nx.ones_like leaf) params);
      ( "params-derived",
        Nx.Ptree.map p
          (fun _ leaf ->
            Nx.add (Nx.sin leaf) (Derivs.float_scalar_like leaf 1.1))
          params );
    ]
  in
  let check (name, v) =
    let bump s =
      Nx.Ptree.map2 p
        (fun _ leaf vl ->
          Nx.add leaf (Nx.mul vl (Derivs.float_scalar_like vl s)))
        params v
    in
    let numeric =
      (scalar_f64 (f (bump eps)) -. scalar_f64 (f (bump (-.eps))))
      /. (2.0 *. eps)
    in
    let analytic = ref 0.0 in
    ignore
      (Nx.Ptree.map2 p
         (fun _ gl vl ->
           analytic := !analytic +. scalar_f64 (Nx.sum (Nx.mul gl vl));
           gl)
         g v);
    if
      Float.abs (!analytic -. numeric)
      <= tol *. Float.max 1.0 (Float.abs numeric)
    then Ok ()
    else
      Error
        (Printf.sprintf
           "check_grads: directional derivative along %s is %g but the \
            gradient predicts %g"
           name numeric !analytic)
  in
  List.fold_left
    (fun acc d -> match acc with Ok () -> check d | e -> e)
    (Ok ()) directions

(* Control flow. [scan] attempts the staged [Scan.E_scan] effect, which jit
   compiles as a loop; when no handler claims it, the eager fold runs, observed
   by whatever transformation handlers are installed. [cond] and [while_loop]
   run eagerly. *)

let scan = Scan.scan

let scan' ~f ~init xs =
  Scan.scan Nx.Ptree.tensor Nx.Ptree.tensor Nx.Ptree.tensor ~f ~init xs

let cond (pred : (bool, Nx.bool_elt) Nx.t) ~(then_ : unit -> 'r)
    ~(else_ : unit -> 'r) : 'r =
  if Nx.item [] pred then then_ () else else_ ()

let while_loop ~cond ~body init =
  let rec go c = if Nx.item [] (cond c) then go (body c) else c in
  go init

(* Just-in-time compilation *)

exception Jit_error = Jit.Jit_error

let device = Jit.device
let devices = Jit.devices
let default_device = Jit.default_device
let jit = Jit.jit
let jit' = Jit.jit'
let pmap = Jit.pmap

type jit_stats = Jit.stats = {
  bytes_to_device : int;
  bytes_from_device : int;
  resident_bytes : int;
  reused_bytes : int;
}

let jit_stats = Jit.stats
let reset_jit_stats = Jit.reset_stats

(* Debugging *)

let with_debug = Debug.with_debug

(* Autodiff control *)

let no_grad f = Gate.without_tracing f
let detach t = Gate.without_tracing (fun () -> Nx.copy t)
