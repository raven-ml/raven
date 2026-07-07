(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Ptree = Nx.Ptree

let shape_string s =
  String.concat "," (Array.to_list (Array.map string_of_int s))

let require_scalar name y =
  if Nx.numel y <> 1 then
    invalid_arg
      (Printf.sprintf
         "%s: the objective must return a scalar tensor, got shape [%s]; use \
          vjp for non-scalar outputs"
         name
         (shape_string (Nx.shape y)))

(* Install a transformation handler for the run of [f]. The recorded depth lets
   [jit] step aside when a transformation is observing the operations. *)
let run_transform f x handler =
  Gate.with_transform (fun () -> Effect.Deep.match_with f x handler)

(* Run [f params] under the reverse handler with the leaves of [params] tracked,
   seed the output cotangent, and pull gradients back to the leaves. *)
let run_reverse (type c d) (module P : Ptree.S) (f : P.t -> (c, d) Nx.t)
    (params : P.t) ~(seed : (c, d) Nx.t -> (c, d) Nx.t) : (c, d) Nx.t * P.t =
  let tape = Tape.create () in
  P.iter (fun leaf -> Tape.track tape leaf) params;
  let y = run_transform f params (Reverse.handler tape) in
  Tape.accumulate tape y (seed y);
  Tape.backward tape;
  (y, P.map (fun leaf -> Tape.cotangent tape leaf) params)

(* Gradients are only defined with respect to real (or complex) leaves; integer
   data belongs in the closure or the auxiliary output. *)
let require_float_leaf name leaf =
  let dt = Nx.dtype leaf in
  if not (Nx_core.Dtype.is_float dt || Nx_core.Dtype.is_complex dt) then
    invalid_arg
      (Printf.sprintf
         "%s: cannot differentiate with respect to a %s leaf; hold \
          non-differentiable data in the closure or the auxiliary output"
         name
         (Nx_core.Dtype.to_string dt))

let value_and_grad (type c d) (module P : Ptree.S) (f : P.t -> (c, d) Nx.t)
    (params : P.t) : (c, d) Nx.t * P.t =
  P.iter (fun leaf -> require_float_leaf "Rune.value_and_grad" leaf) params;
  let y, grads =
    run_reverse
      (module P)
      f params
      ~seed:(fun y ->
        require_scalar "Rune.value_and_grad" y;
        Nx.ones_like y)
  in
  (y, grads)

let grad (type c d) (module P : Ptree.S) (f : P.t -> (c, d) Nx.t) (params : P.t)
    : P.t =
  snd (value_and_grad (module P) f params)

let value_and_grad_aux (type c d) (module P : Ptree.S)
    (f : P.t -> (c, d) Nx.t * 'aux) (params : P.t) : (c, d) Nx.t * P.t * 'aux =
  let aux = ref None in
  let f' ps =
    let y, a = f ps in
    aux := Some a;
    y
  in
  let y, grads = value_and_grad (module P) f' params in
  match !aux with
  | Some a -> (y, grads, a)
  | None -> assert false (* [f'] completed, so [aux] was set. *)

let vjp (type c d) (module P : Ptree.S) (f : P.t -> (c, d) Nx.t) (params : P.t)
    (cotangent : (c, d) Nx.t) : (c, d) Nx.t * P.t =
  run_reverse (module P) f params ~seed:(fun _ -> cotangent)

let err_cotangent_shape leaf cotangent =
  invalid_arg
    (Printf.sprintf
       "Rune.vjp2: cotangent shape [%s] does not match output shape [%s]"
       (shape_string (Nx.shape cotangent))
       (shape_string (Nx.shape leaf)))

let vjp2 (module P : Ptree.S) (module Q : Ptree.S) (f : P.t -> Q.t)
    (params : P.t) (cotangents : Q.t) : Q.t * P.t =
  let tape = Tape.create () in
  P.iter (fun leaf -> Tape.track tape leaf) params;
  let y = run_transform f params (Reverse.handler tape) in
  let (_ : Q.t) =
    Q.map2
      (fun yleaf ct ->
        if Nx.shape yleaf <> Nx.shape ct then err_cotangent_shape yleaf ct;
        Tape.accumulate tape yleaf ct;
        yleaf)
      y cotangents
  in
  Tape.backward tape;
  (y, P.map (fun leaf -> Tape.cotangent tape leaf) params)

let vjp_fun (type c d) (module P : Ptree.S) (f : P.t -> (c, d) Nx.t)
    (params : P.t) : (c, d) Nx.t * ((c, d) Nx.t -> P.t) =
  let tape = Tape.create () in
  P.iter (fun leaf -> Tape.track tape leaf) params;
  let y = run_transform f params (Reverse.handler tape) in
  let pullback ct =
    if Nx.shape ct <> Nx.shape y then
      invalid_arg
        (Printf.sprintf
           "Rune.vjp_fun: cotangent shape [%s] does not match output shape [%s]"
           (shape_string (Nx.shape ct))
           (shape_string (Nx.shape y)));
    Tape.reset_cotangents tape;
    Tape.accumulate tape y ct;
    Tape.backward tape;
    P.map (fun leaf -> Tape.cotangent tape leaf) params
  in
  (y, pullback)

let vjp_fun' (type a b c d) (f : (a, b) Nx.t -> (c, d) Nx.t) (x : (a, b) Nx.t) :
    (c, d) Nx.t * ((c, d) Nx.t -> (a, b) Nx.t) =
  let tape = Tape.create () in
  Tape.track tape x;
  let y = run_transform f x (Reverse.handler tape) in
  let pullback ct =
    Tape.reset_cotangents tape;
    Tape.accumulate tape y ct;
    Tape.backward tape;
    Tape.cotangent tape x
  in
  (y, pullback)

(* Forward mode *)

let err_tangent_shape name leaf tangent =
  invalid_arg
    (Printf.sprintf "%s: tangent shape [%s] does not match parameter shape [%s]"
       name
       (shape_string (Nx.shape tangent))
       (shape_string (Nx.shape leaf)))

let output_tangent store y =
  match Tensor_map.find store y with Some dy -> dy | None -> Nx.zeros_like y

let jvp (type c d) (module P : Ptree.S) (f : P.t -> (c, d) Nx.t) (params : P.t)
    (tangents : P.t) : (c, d) Nx.t * (c, d) Nx.t =
  let store = Tensor_map.create () in
  let (_ : P.t) =
    P.map2
      (fun leaf tangent ->
        if Nx.shape leaf <> Nx.shape tangent then
          err_tangent_shape "Rune.jvp" leaf tangent;
        Tensor_map.set store leaf tangent;
        leaf)
      params tangents
  in
  let y = run_transform f params (Forward.handler store) in
  (y, output_tangent store y)

let jvp_aux (type c d) (module P : Ptree.S) (f : P.t -> (c, d) Nx.t * 'aux)
    (params : P.t) (tangents : P.t) : (c, d) Nx.t * (c, d) Nx.t * 'aux =
  let aux = ref None in
  let f' ps =
    let y, a = f ps in
    aux := Some a;
    y
  in
  let y, dy = jvp (module P) f' params tangents in
  match !aux with
  | Some a -> (y, dy, a)
  | None -> assert false (* [f'] completed, so [aux] was set. *)

let jvp2 (module P : Ptree.S) (module Q : Ptree.S) (f : P.t -> Q.t)
    (params : P.t) (tangents : P.t) : Q.t * Q.t =
  let store = Tensor_map.create () in
  let (_ : P.t) =
    P.map2
      (fun leaf tangent ->
        if Nx.shape leaf <> Nx.shape tangent then
          err_tangent_shape "Rune.jvp2" leaf tangent;
        Tensor_map.set store leaf tangent;
        leaf)
      params tangents
  in
  let y = run_transform f params (Forward.handler store) in
  (y, Q.map (fun yleaf -> output_tangent store yleaf) y)

(* Custom differentiation rules *)

let custom_vjp = Custom.custom_vjp
let custom_jvp = Custom.custom_jvp

(* Vectorizing maps *)

let broadcast_output st y =
  if Vmap.batched st y then y else Vmap.ensure_batched st y

(* Validate in_axes, determine the batch size, move mapped axes to the front and
   mark those leaves: shared by [vmap] and [vmap2]. *)
let prepare_vmap ?in_axes (module P : Ptree.S) (params : P.t) : Vmap.state * P.t
    =
  let leaves = ref 0 in
  P.iter (fun _ -> incr leaves) params;
  let specs =
    match in_axes with
    | None -> List.init !leaves (fun _ -> Some 0)
    | Some l ->
        if List.length l <> !leaves then
          invalid_arg
            (Printf.sprintf
               "Rune.vmap: in_axes has %d entries but the structure has %d \
                leaves"
               (List.length l) !leaves);
        l
  in
  (* Pair specs with leaves by physical identity, in iteration order. Positional
     pairing inside [P.map] would be unsound: the callback's evaluation order is
     instance-defined (record fields evaluate right to left), while [iter] has
     an explicit sequence. *)
  let assoc = ref [] in
  let batch = ref None in
  let i = ref 0 in
  P.iter
    (fun leaf ->
      let spec = List.nth specs !i in
      incr i;
      let key = Obj.repr leaf in
      (match List.assq_opt key !assoc with
      | Some spec' when spec' <> spec ->
          invalid_arg
            "Rune.vmap: the same tensor appears as several leaves with \
             different in_axes entries"
      | _ -> ());
      assoc := (key, spec) :: !assoc;
      match spec with
      | None -> ()
      | Some ax -> (
          let s = Nx.shape leaf in
          let rank = Array.length s in
          if rank = 0 then invalid_arg "Rune.vmap: cannot map a scalar leaf";
          let ax = if ax < 0 then ax + rank else ax in
          if ax < 0 || ax >= rank then
            invalid_arg "Rune.vmap: in_axes entry is out of bounds";
          match !batch with
          | None -> batch := Some s.(ax)
          | Some b ->
              if s.(ax) <> b then
                invalid_arg
                  (Printf.sprintf
                     "Rune.vmap: mapped axis sizes disagree (%d vs %d)" b s.(ax))
          ))
    params;
  let batch_size =
    match !batch with
    | Some b -> b
    | None -> invalid_arg "Rune.vmap: in_axes maps no leaf"
  in
  (* Move mapped axes to the front and mark those leaves as batched. *)
  let st = Vmap.create ~batch_size in
  let params =
    P.map
      (fun leaf ->
        match List.assq (Obj.repr leaf) !assoc with
        | None -> leaf
        | Some ax ->
            let ax = if ax < 0 then ax + Array.length (Nx.shape leaf) else ax in
            let leaf = if ax = 0 then leaf else Nx.moveaxis ax 0 leaf in
            Vmap.mark st leaf;
            leaf)
      params
  in
  (st, params)

let finalize_vmap st out_axis y =
  let y = broadcast_output st y in
  if out_axis = 0 then y else Nx.moveaxis 0 out_axis y

let vmap (type c d) ?in_axes ?(out_axis = 0) (module P : Ptree.S)
    (f : P.t -> (c, d) Nx.t) (params : P.t) : (c, d) Nx.t =
  let st, params = prepare_vmap ?in_axes (module P) params in
  let y = run_transform f params (Vmap.handler st) in
  finalize_vmap st out_axis y

let vmap2 ?in_axes ?(out_axis = 0) (module P : Ptree.S) (module Q : Ptree.S)
    (f : P.t -> Q.t) (params : P.t) : Q.t =
  let st, params = prepare_vmap ?in_axes (module P) params in
  let y = run_transform f params (Vmap.handler st) in
  Q.map (fun yleaf -> finalize_vmap st out_axis yleaf) y

let vmap' (type a b c d) ?(in_axis = 0) ?(out_axis = 0)
    (f : (a, b) Nx.t -> (c, d) Nx.t) (x : (a, b) Nx.t) : (c, d) Nx.t =
  let rank = Array.length (Nx.shape x) in
  if rank = 0 then invalid_arg "Rune.vmap': cannot map a scalar";
  let x = if in_axis = 0 then x else Nx.moveaxis in_axis 0 x in
  let st = Vmap.create ~batch_size:(Nx.shape x).(0) in
  Vmap.mark st x;
  let y = run_transform f x (Vmap.handler st) in
  let y = broadcast_output st y in
  if out_axis = 0 then y else Nx.moveaxis 0 out_axis y

(* Single-tensor variants *)

let run_reverse' (type a b c d) (f : (a, b) Nx.t -> (c, d) Nx.t)
    (x : (a, b) Nx.t) ~(seed : (c, d) Nx.t -> (c, d) Nx.t) :
    (c, d) Nx.t * (a, b) Nx.t =
  let tape = Tape.create () in
  Tape.track tape x;
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

let jvp' (type a b c d) (f : (a, b) Nx.t -> (c, d) Nx.t) (x : (a, b) Nx.t)
    (tangent : (a, b) Nx.t) : (c, d) Nx.t * (c, d) Nx.t =
  if Nx.shape x <> Nx.shape tangent then err_tangent_shape "Rune.jvp'" x tangent;
  let store = Tensor_map.create () in
  Tensor_map.set store x tangent;
  let y = run_transform f x (Forward.handler store) in
  (y, output_tangent store y)

(* Gradient checkpointing *)

let remat (module P : Ptree.S) (f : P.t -> ('c, 'd) Nx.t) (params : P.t) :
    ('c, 'd) Nx.t =
  Custom.custom_vjp
    (module P)
    ~fwd:(fun p -> (f p, p))
    ~bwd:(fun p ct -> snd (vjp (module P) f p ct))
    params

(* Jacobians *)

(* [basis_like y] is a [numel y; shape y...] tensor whose k-th slice is the k-th
   standard basis element of [y]'s space. *)
let basis_like (type a b) (y : (a, b) Nx.t) : (a, b) Nx.t =
  let n = Nx.numel y in
  Nx.reshape (Array.append [| n |] (Nx.shape y)) (Nx.eye (Nx.dtype y) n)

let jacrev' (type a b c d) (f : (a, b) Nx.t -> (c, d) Nx.t) (x : (a, b) Nx.t) :
    (a, b) Nx.t =
  let y, pullback = vjp_fun' f x in
  let rows = vmap' pullback (basis_like y) in
  Nx.reshape (Array.append (Nx.shape y) (Nx.shape x)) (Nx.contiguous rows)

let jacfwd' (type a b c d) (f : (a, b) Nx.t -> (c, d) Nx.t) (x : (a, b) Nx.t) :
    (c, d) Nx.t =
  let cols = vmap' (fun v -> snd (jvp' f x v)) (basis_like x) in
  (* [cols] is [numel x; shape y...]; the input axis moves last. *)
  let rank = Array.length (Nx.shape cols) in
  let cols = Nx.moveaxis 0 (rank - 1) cols in
  let y_shape = Array.sub (Nx.shape cols) 0 (rank - 1) in
  Nx.reshape (Array.append y_shape (Nx.shape x)) (Nx.contiguous cols)

let hessian' (type a b) (f : (a, b) Nx.t -> (a, b) Nx.t) (x : (a, b) Nx.t) :
    (a, b) Nx.t =
  jacfwd' (grad' f) x

let hvp (module P : Ptree.S) (f : P.t -> ('c, 'd) Nx.t) (params : P.t) (v : P.t)
    : P.t =
  snd (jvp2 (module P) (module P) (grad (module P) f) params v)

let hvp' (type a b c d) (f : (a, b) Nx.t -> (c, d) Nx.t) (x : (a, b) Nx.t)
    (v : (a, b) Nx.t) : (a, b) Nx.t =
  snd (jvp' (grad' f) x v)

(* Gradient checking *)

let check_grads ?(eps = 1e-4) ?(tol = 1e-2) (module P : Ptree.S)
    (f : P.t -> ('c, 'd) Nx.t) (params : P.t) : (unit, string) result =
  let scalar_f64 t = Nx.item [] (Nx.reshape [||] (Nx.astype Nx.float64 t)) in
  let g = grad (module P) f params in
  (* Two deterministic directions: all-ones, and a params-derived direction so
     the two are independent for non-constant params. *)
  let directions =
    [
      ("ones", P.map (fun leaf -> Nx.ones_like leaf) params);
      ( "params-derived",
        P.map
          (fun leaf -> Nx.add (Nx.sin leaf) (Derivs.float_scalar_like leaf 1.1))
          params );
    ]
  in
  let check (name, v) =
    let bump s =
      P.map2
        (fun leaf vl -> Nx.add leaf (Nx.mul vl (Derivs.float_scalar_like vl s)))
        params v
    in
    let numeric =
      (scalar_f64 (f (bump eps)) -. scalar_f64 (f (bump (-.eps))))
      /. (2.0 *. eps)
    in
    let analytic = ref 0.0 in
    let (_ : P.t) =
      P.map2
        (fun gl vl ->
          analytic := !analytic +. scalar_f64 (Nx.sum (Nx.mul gl vl));
          gl)
        g v
    in
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

(* Control flow. Eager implementations with staging-ready signatures: a future
   jit stages these as structured control flow instead of unrolled traces. *)

let scan (module C : Ptree.S) ~(f : C.t -> ('a, 'b) Nx.t -> C.t * ('c, 'd) Nx.t)
    ~(init : C.t) (xs : ('a, 'b) Nx.t) : C.t * ('c, 'd) Nx.t =
  let shape = Nx.shape xs in
  if Array.length shape = 0 then
    invalid_arg "Rune.scan: xs must have a leading scan axis";
  let n = shape.(0) in
  if n = 0 then invalid_arg "Rune.scan: xs is empty along the scan axis";
  let carry = ref init in
  let ys = ref [] in
  for i = 0 to n - 1 do
    let c, y = f !carry (Nx.slice [ Nx.I i ] xs) in
    carry := c;
    ys := y :: !ys
  done;
  (!carry, Nx.stack ~axis:0 (List.rev !ys))

let cond (pred : (bool, Nx.bool_elt) Nx.t) ~(then_ : unit -> 'r)
    ~(else_ : unit -> 'r) : 'r =
  if Nx.item [] pred then then_ () else else_ ()

let while_loop (module C : Ptree.S) ~(cond : C.t -> (bool, Nx.bool_elt) Nx.t)
    ~(body : C.t -> C.t) (init : C.t) : C.t =
  let rec go c = if Nx.item [] (cond c) then go (body c) else c in
  go init

(* Just-in-time compilation *)

exception Jit_error = Jit.Jit_error

let jit = Jit.jit
let jit2 = Jit.jit2
let jit' = Jit.jit'

(* Debugging *)

let with_debug = Debug.with_debug

(* Autodiff control *)

let no_grad f = Gate.without_tracing f
let detach t = Gate.without_tracing (fun () -> Nx.copy t)
