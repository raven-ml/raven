(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module type Differentiable = Differentiable.S

module Ptree = Ptree

let require_scalar name y =
  if Nx.numel y <> 1 then
    invalid_arg
      (Printf.sprintf
         "%s: the objective must return a scalar tensor, got shape [%s]; use \
          vjp for non-scalar outputs"
         name
         (String.concat ","
            (Array.to_list (Array.map string_of_int (Nx.shape y)))))

(* Run [f params] under the reverse handler with the leaves of [params] tracked,
   seed the output cotangent, and pull gradients back to the leaves. *)
let run_reverse (type c d) (module P : Differentiable) (f : P.t -> (c, d) Nx.t)
    (params : P.t) ~(seed : (c, d) Nx.t -> (c, d) Nx.t) : (c, d) Nx.t * P.t =
  let tape = Tape.create () in
  P.iter (fun leaf -> Tape.track tape leaf) params;
  let y = Effect.Deep.match_with f params (Reverse.handler tape) in
  Tape.accumulate tape y (seed y);
  Tape.backward tape;
  (y, P.map (fun leaf -> Tape.cotangent tape leaf) params)

let value_and_grad (type c d) (module P : Differentiable)
    (f : P.t -> (c, d) Nx.t) (params : P.t) : (c, d) Nx.t * P.t =
  let y, grads =
    run_reverse
      (module P)
      f params
      ~seed:(fun y ->
        require_scalar "Rune_next.value_and_grad" y;
        Nx.ones_like y)
  in
  (y, grads)

let grad (type c d) (module P : Differentiable) (f : P.t -> (c, d) Nx.t)
    (params : P.t) : P.t =
  snd (value_and_grad (module P) f params)

let value_and_grad_aux (type c d) (module P : Differentiable)
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
  | None ->
      invalid_arg
        "Rune_next.value_and_grad_aux: objective did not produce a value"

let vjp (type c d) (module P : Differentiable) (f : P.t -> (c, d) Nx.t)
    (params : P.t) (cotangent : (c, d) Nx.t) : (c, d) Nx.t * P.t =
  run_reverse (module P) f params ~seed:(fun _ -> cotangent)

(* Forward mode *)

let err_tangent_shape name leaf tangent =
  invalid_arg
    (Printf.sprintf "%s: tangent shape [%s] does not match parameter shape [%s]"
       name
       (String.concat ","
          (Array.to_list (Array.map string_of_int (Nx.shape tangent))))
       (String.concat ","
          (Array.to_list (Array.map string_of_int (Nx.shape leaf)))))

let output_tangent store y =
  match Tensor_map.find store y with Some dy -> dy | None -> Nx.zeros_like y

let jvp (type c d) (module P : Differentiable) (f : P.t -> (c, d) Nx.t)
    (params : P.t) (tangents : P.t) : (c, d) Nx.t * (c, d) Nx.t =
  let store = Tensor_map.create () in
  let (_ : P.t) =
    P.map2
      (fun leaf tangent ->
        if Nx.shape leaf <> Nx.shape tangent then
          err_tangent_shape "Rune_next.jvp" leaf tangent;
        Tensor_map.set store leaf tangent;
        leaf)
      params tangents
  in
  let y = Effect.Deep.match_with f params (Forward.handler store) in
  (y, output_tangent store y)

let jvp_aux (type c d) (module P : Differentiable)
    (f : P.t -> (c, d) Nx.t * 'aux) (params : P.t) (tangents : P.t) :
    (c, d) Nx.t * (c, d) Nx.t * 'aux =
  let aux = ref None in
  let f' ps =
    let y, a = f ps in
    aux := Some a;
    y
  in
  let y, dy = jvp (module P) f' params tangents in
  match !aux with
  | Some a -> (y, dy, a)
  | None -> invalid_arg "Rune_next.jvp_aux: objective did not produce a value"

(* Custom differentiation rules *)

let custom_vjp = Custom.custom_vjp
let custom_jvp = Custom.custom_jvp

(* Vectorizing maps *)

let broadcast_output st y =
  if Vmap.batched st y then y else Vmap.ensure_batched st y

let vmap (type c d) ?in_axes ?(out_axis = 0) (module P : Differentiable)
    (f : P.t -> (c, d) Nx.t) (params : P.t) : (c, d) Nx.t =
  let leaves = ref 0 in
  P.iter (fun _ -> incr leaves) params;
  let specs =
    match in_axes with
    | None -> List.init !leaves (fun _ -> Some 0)
    | Some l ->
        if List.length l <> !leaves then
          invalid_arg
            (Printf.sprintf
               "Rune_next.vmap: in_axes has %d entries but the structure has \
                %d leaves"
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
            "Rune_next.vmap: the same tensor appears as several leaves with \
             different in_axes entries"
      | _ -> ());
      assoc := (key, spec) :: !assoc;
      match spec with
      | None -> ()
      | Some ax -> (
          let s = Nx.shape leaf in
          let rank = Array.length s in
          if rank = 0 then
            invalid_arg "Rune_next.vmap: cannot map a scalar leaf";
          let ax = if ax < 0 then ax + rank else ax in
          if ax < 0 || ax >= rank then
            invalid_arg "Rune_next.vmap: in_axes entry is out of bounds";
          match !batch with
          | None -> batch := Some s.(ax)
          | Some b ->
              if s.(ax) <> b then
                invalid_arg
                  (Printf.sprintf
                     "Rune_next.vmap: mapped axis sizes disagree (%d vs %d)" b
                     s.(ax))))
    params;
  let batch_size =
    match !batch with
    | Some b -> b
    | None -> invalid_arg "Rune_next.vmap: in_axes maps no leaf"
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
  let y = Effect.Deep.match_with f params (Vmap.handler st) in
  let y = broadcast_output st y in
  if out_axis = 0 then y else Nx.moveaxis 0 out_axis y

let vmap' (type a b c d) ?(in_axis = 0) ?(out_axis = 0)
    (f : (a, b) Nx.t -> (c, d) Nx.t) (x : (a, b) Nx.t) : (c, d) Nx.t =
  let rank = Array.length (Nx.shape x) in
  if rank = 0 then invalid_arg "Rune_next.vmap': cannot map a scalar";
  let x = if in_axis = 0 then x else Nx.moveaxis in_axis 0 x in
  let st = Vmap.create ~batch_size:(Nx.shape x).(0) in
  Vmap.mark st x;
  let y = Effect.Deep.match_with f x (Vmap.handler st) in
  let y = broadcast_output st y in
  if out_axis = 0 then y else Nx.moveaxis 0 out_axis y

(* Single-tensor variants *)

let run_reverse' (type a b c d) (f : (a, b) Nx.t -> (c, d) Nx.t)
    (x : (a, b) Nx.t) ~(seed : (c, d) Nx.t -> (c, d) Nx.t) :
    (c, d) Nx.t * (a, b) Nx.t =
  let tape = Tape.create () in
  Tape.track tape x;
  let y = Effect.Deep.match_with f x (Reverse.handler tape) in
  Tape.accumulate tape y (seed y);
  Tape.backward tape;
  (y, Tape.cotangent tape x)

let value_and_grad' f x =
  run_reverse' f x ~seed:(fun y ->
      require_scalar "Rune_next.value_and_grad'" y;
      Nx.ones_like y)

let grad' f x = snd (value_and_grad' f x)
let vjp' f x cotangent = run_reverse' f x ~seed:(fun _ -> cotangent)

let jvp' (type a b c d) (f : (a, b) Nx.t -> (c, d) Nx.t) (x : (a, b) Nx.t)
    (tangent : (a, b) Nx.t) : (c, d) Nx.t * (c, d) Nx.t =
  if Nx.shape x <> Nx.shape tangent then
    err_tangent_shape "Rune_next.jvp'" x tangent;
  let store = Tensor_map.create () in
  Tensor_map.set store x tangent;
  let y = Effect.Deep.match_with f x (Forward.handler store) in
  (y, output_tangent store y)

(* Autodiff control *)

let no_grad f = Gate.without_tracing f
let detach t = Gate.without_tracing (fun () -> Nx.copy t)
