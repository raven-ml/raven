(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Ptree = Nx.Ptree

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let not_yet name = invalid_arg ("Rune." ^ name ^ ": not implemented yet")

let differentiable x =
  let dt = Nx.dtype x in
  Nx_dtype.is_float dt || Nx_dtype.is_complex dt

(* Reverse mode *)

(* [linearize fn p f params] is [f] applied to [params] under a recorder and a
   forward mode whose tangents are the recorder's slots: each real or complex
   leaf of [params] becomes a dual whose tangent is an input slot. It is the
   tape, the forward mode, the leaves as [f] received them, and [f]'s result. *)
let linearize fn p f params =
  let tape = Linear.create fn in
  let i = Jvp.create ~slots:tape fn in
  let any = ref false in
  let seeded =
    Ptree.map p
      (fun _ x ->
        if differentiable x then begin
          any := true;
          Jvp.dual i x (Linear.input tape x)
        end
        else x)
      params
  in
  if not !any then
    invalid_argf "%s: the parameters hold no real or complex tensor" fn;
  let y = Linear.install tape (fun () -> Jvp.install i (fun () -> f seeded)) in
  (tape, i, seeded, y)

(* [pull p tape i seeded params seed] is the gradient of [params] once [seed]
   has added the result's cotangents: the conjugated cotangent of each leaf's
   input slot, and zeros for a leaf with none. *)
let pull p tape i seeded params seed =
  let cts = Linear.cotangents tape in
  seed cts;
  Linear.transpose cts;
  let gradient _ x s =
    match Option.bind (snd (Jvp.split i s)) (Linear.cotangent cts) with
    | Some g -> Nx.conjugate g
    | None -> Nx.zeros_like x
  in
  Ptree.map2 p gradient params seeded

let objective fn y =
  if Nx.numel y <> 1 || not (differentiable y) then
    invalid_arg
      (Format.asprintf
         "%s: the objective must return a real or complex scalar, got %a %a" fn
         Nx.pp_dtype (Nx.dtype y) Nx.pp_shape (Nx.shape y))

let value_and_grad_of fn p f params =
  let tape, i, seeded, y = linearize fn p f params in
  let y, dy = Jvp.split i y in
  objective fn y;
  let seed cts =
    Option.iter (fun dy -> Linear.add cts dy (Nx.ones_like y)) dy
  in
  (y, pull p tape i seeded params seed)

let value_and_grad p f params =
  value_and_grad_of "Rune.value_and_grad" p f params

let grad p f params = snd (value_and_grad_of "Rune.grad" p f params)

let value_and_grad_aux p x f params =
  let tape, i, seeded, (y, aux) =
    linearize "Rune.value_and_grad_aux" p f params
  in
  let y, dy = Jvp.split i y in
  objective "Rune.value_and_grad_aux" y;
  let seed cts =
    Option.iter (fun dy -> Linear.add cts dy (Nx.ones_like y)) dy
  in
  let aux = Ptree.map x (fun _ v -> fst (Jvp.split i v)) aux in
  (y, pull p tape i seeded params seed, aux)

let vjp_of fn p q f params =
  let tape, i, seeded, y = linearize fn p f params in
  let seed cts ct =
    let add _ v ct =
      Option.iter
        (fun slot -> Linear.add cts slot (Nx.conjugate ct))
        (snd (Jvp.split i v));
      v
    in
    ignore
      (Structure.map2 fn q ~this:"the result" ~that:"the cotangents" add y ct)
  in
  let pullback ct = pull p tape i seeded params (fun cts -> seed cts ct) in
  (Ptree.map q (fun _ v -> fst (Jvp.split i v)) y, pullback)

let vjp p q f params = vjp_of "Rune.vjp" p q f params

(* Forward mode *)

let jvp_of fn p q f params tangents =
  let i = Jvp.create fn in
  let any = ref false in
  let duals =
    Structure.map2 fn p ~this:"the parameters" ~that:"the tangents"
      (fun _ x dx ->
        if differentiable x then begin
          any := true;
          Jvp.dual i x dx
        end
        else x)
      params tangents
  in
  if not !any then
    invalid_argf "%s: the parameters hold no real or complex tensor" fn;
  let y = Jvp.install i (fun () -> f duals) in
  let tangent _ v =
    match Jvp.split i v with _, Some dv -> dv | v, None -> Nx.zeros_like v
  in
  (Ptree.map q (fun _ v -> fst (Jvp.split i v)) y, Ptree.map q tangent y)

let jvp p q f params tangents = jvp_of "Rune.jvp" p q f params tangents

(* Jacobians *)

let jacfwd' _ _ = not_yet "jacfwd'"
let jacrev' _ _ = not_yet "jacrev'"

(* Controlling differentiation *)

let detach x = Construct.perform (Detach x)

type move = { move : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t }

let check_grads ?(eps = 1e-4) ?(tol = 1e-2) p f params =
  let scalar t = Nx.item [] (Nx.reshape [||] (Nx.cast Nx.float64 t)) in
  let like x v = Nx.full (Nx.dtype x) [||] (Nx_dtype.of_float (Nx.dtype x) v) in
  let g = snd (value_and_grad_of "Rune.check_grads" p f params) in
  (* A direction moves the real and complex leaves; the others are carried. *)
  let direction { move } =
    Ptree.map p
      (fun _ x -> if differentiable x then move x else Nx.zeros_like x)
      params
  in
  let directions =
    [
      ("ones", direction { move = Nx.ones_like });
      ( "params-derived",
        direction { move = (fun x -> Nx.add (Nx.sin x) (like x 1.1)) } );
    ]
  in
  let check (name, v) =
    let bump s =
      Ptree.map2 p (fun _ x v -> Nx.add x (Nx.mul v (like v s))) params v
    in
    let numeric =
      (scalar (f (bump eps)) -. scalar (f (bump (-.eps)))) /. (2. *. eps)
    in
    let analytic = ref 0. in
    ignore
      (Ptree.map2 p
         (fun _ g v ->
           analytic := !analytic +. scalar (Nx.sum (Nx.mul (Nx.conjugate g) v));
           g)
         g v);
    if
      Float.abs (!analytic -. numeric)
      <= tol *. Float.max 1. (Float.abs numeric)
    then Ok ()
    else
      Error
        (Printf.sprintf
           "Rune.check_grads: the directional derivative along %s is %g, and \
            the gradient predicts %g"
           name numeric !analytic)
  in
  List.fold_left
    (fun acc d -> match acc with Ok () -> check d | Error _ -> acc)
    (Ok ()) directions

(* Custom differentiation rules *)

let custom_jvp p q rule args =
  Construct.perform (Custom (Jvp_rule { p; q; rule; args; value = None }))

let custom_vjp p q rule args =
  Construct.perform (Custom (Vjp_rule { p; q; rule; args }))

(* Gradient checkpointing *)

let remat s =
  let (Structure.Signature u) = Structure.uncurry "Rune.remat" s in
  fun f ->
    u.curry (fun args ->
        Construct.perform
          (Remat
             {
               p = u.args;
               q = u.result;
               f = u.apply f;
               args;
               recomputed = false;
             }))

(* Vectorizing maps *)

type axis = Construct.axis

let axis () = Type.Id.make ()
let vmap ?axis:_ _ _ _ = not_yet "vmap"
let lanes a x = Construct.perform (Lanes (a, x))
let lane_index ?axis () = Construct.perform (Lane_index axis)

(* Totals *)

module Total = struct
  type ('a, 'b) t = ('a, 'b) Construct.total

  let make () = Type.Id.make ()
  let add t v = Construct.perform (Add (t, v))
  let collect = Total.collect
end

(* Loops *)

let steps fn xs =
  let lead (Nx.P x) =
    match Nx.shape x with
    | [||] -> invalid_arg (fn ^ ": an xs leaf is a scalar")
    | shape -> shape.(0)
  in
  match xs with
  | [] -> invalid_arg (fn ^ ": xs has no leaf")
  | x :: rest ->
      let n = lead x in
      if List.exists (fun x -> lead x <> n) rest then
        invalid_arg (fn ^ ": the xs leaves differ in their leading length");
      if n = 0 then invalid_arg (fn ^ ": xs is empty along the scan axis")

let answer r =
  match Construct.perform (Scan r) with
  | r -> r
  | exception Scan.Not_staged -> Scan.fold r

let scan_of fn cs xs_s ys_s ~f ~init xs =
  let req_xs, _ = Ptree.flatten xs_s xs in
  steps fn req_xs;
  let first = ref None in
  let req_step c_leaves x_leaves =
    let c = Ptree.rebuild cs ~like:init c_leaves in
    let c', y = f c (Ptree.rebuild xs_s ~like:xs x_leaves) in
    Structure.check fn cs ~this:"the carry the body returned" c'
      ~that:"the carry it received" c;
    (match !first with
    | None -> first := Some y
    | Some y0 ->
        ignore
          (Structure.map2 fn ys_s ~this:"a step's outputs"
             ~that:"the first step's outputs"
             (fun _ y _ -> y)
             y y0));
    (fst (Ptree.flatten cs c'), fst (Ptree.flatten ys_s y))
  in
  let req_carry, _ = Ptree.flatten cs init in
  let r = answer { req_carry; req_xs; req_step; req_reverse = false } in
  match !first with
  | Some y0 ->
      (Ptree.rebuild cs ~like:init r.r_carry, Ptree.rebuild ys_s ~like:y0 r.r_ys)
  | None -> assert false (* Every answer runs the body at least once. *)

let scan cs xs_s ys_s ~f ~init xs = scan_of "Rune.scan" cs xs_s ys_s ~f ~init xs

(* Compilation *)

exception Jit_error = Lower.Jit_error

let jit _ _ _ = not_yet "jit"
let compiled = Compiled.backend

(* Functions of one tensor *)

let t = Ptree.tensor
let grad' f x = snd (value_and_grad_of "Rune.grad'" t f x)
let value_and_grad' f x = value_and_grad_of "Rune.value_and_grad'" t f x
let vjp' f x = vjp_of "Rune.vjp'" t t f x
let jvp' f x dx = jvp_of "Rune.jvp'" t t f x dx
let vmap' ?axis:_ _ _ = not_yet "vmap'"
let scan' ~f ~init xs = scan_of "Rune.scan'" t t t ~f ~init xs
let jit' _ _ = not_yet "jit'"
