(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Ptree = Nx.Ptree

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* [fresh x] is [x] in storage of its own, in C order from its start, when [x]
   is an eager view: a gradient or a tangent is a computation's result, and its
   layout is not the caller's to inherit. A traced value's layout is its
   interpretation's. *)
let fresh x =
  match Nx.Repr.v x with Traced _ -> x | Host _ | Placed _ -> Nx.contiguous x

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
        if Linear.differentiable x then begin
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
    | Some g -> fresh (Nx.conjugate g)
    | None -> Nx.zeros_like x
  in
  Ptree.map2 p gradient params seeded

let objective fn y =
  if Nx.numel y <> 1 || not (Linear.differentiable y) then
    invalid_arg
      (Format.asprintf
         "%s: the objective must return a real or complex scalar, got %a %a" fn
         Nx.pp_dtype (Nx.dtype y) Nx.pp_shape (Nx.shape y))

let value_and_grad_aux_of fn p x f params =
  let tape, i, seeded, (y, aux) = linearize fn p f params in
  let y, dy = Jvp.split i y in
  objective fn y;
  let seed cts =
    Option.iter (fun dy -> Linear.add cts dy (Nx.ones_like y)) dy
  in
  let aux = Ptree.map x (fun _ v -> fst (Jvp.split i v)) aux in
  (y, pull p tape i seeded params seed, aux)

let value_and_grad_of fn p f params =
  let y, g, () =
    value_and_grad_aux_of fn p Ptree.unit (fun x -> (f x, ())) params
  in
  (y, g)

let value_and_grad p f params =
  value_and_grad_of "Rune.value_and_grad" p f params

let grad p f params = snd (value_and_grad_of "Rune.grad" p f params)

let value_and_grad_aux p x f params =
  value_and_grad_aux_of "Rune.value_and_grad_aux" p x f params

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
        if Linear.differentiable x then begin
          any := true;
          Jvp.dual i x dx
        end
        else x)
      params tangents
  in
  if not !any then
    invalid_argf "%s: the parameters hold no real or complex tensor" fn;
  let y = Jvp.install i (fun () -> f duals) in
  ( Ptree.map q (fun _ v -> fst (Jvp.split i v)) y,
    Ptree.map q (fun _ v -> fresh (Jvp.tangent i v)) y )

let jvp p q f params tangents = jvp_of "Rune.jvp" p q f params tangents

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
      (fun _ x -> if Linear.differentiable x then move x else Nx.zeros_like x)
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

(* [length fn s args] is the common length of the first axis of [args]' tensors,
   which a map maps over. *)
let length fn s args =
  let first = ref None in
  Ptree.fold s
    (fun path x () ->
      let at = Ptree.Path.to_string path in
      match (Nx.shape x, !first) with
      | [||], _ -> invalid_argf "%s: %s: a scalar; a map maps axis 0" fn at
      | shape, None -> first := Some (shape.(0), at)
      | shape, Some (n, first) ->
          if shape.(0) <> n then
            invalid_argf "%s: %s: %d rows along axis 0, %s: %d" fn at shape.(0)
              first n)
    args ();
  match !first with
  | Some (n, _) -> n
  | None -> invalid_arg (fn ^ ": the arguments have no tensor to map")

let vmap_of fn ?axis s =
  let (Structure.Signature u) = Structure.uncurry fn s in
  fun f ->
    u.curry (fun args ->
        let m = Vmap.create ?axis fn (length fn u.args args) in
        let args = Ptree.map u.args (fun _ x -> Vmap.lane m x) args in
        let y = Vmap.install m (fun () -> u.apply f args) in
        Ptree.map u.result (fun _ y -> Vmap.batched m y) y)

let vmap ?axis s = vmap_of "Rune.vmap" ?axis s

(* Jacobians *)

(* A [numel x; shape x...] tensor whose [k]-th row is the [k]-th standard basis
   element of [x]'s space. *)
let basis x =
  let n = Nx.numel x in
  Nx.reshape (Array.append [| n |] (Nx.shape x)) (Nx.eye (Nx.dtype x) n)

let jacfwd' f x =
  let fn = "Rune.jacfwd'" in
  let one = Ptree.tensor in
  let cols =
    vmap_of fn
      Ptree.(one @-> returns one)
      (fun v -> snd (jvp_of fn one one f x v))
      (basis x)
  in
  let r = Nx.ndim cols in
  let y_shape = Array.sub (Nx.shape cols) 1 (r - 1) in
  Nx.reshape (Array.append y_shape (Nx.shape x)) (Nx.moveaxis 0 (r - 1) cols)

let jacrev' f x =
  let fn = "Rune.jacrev'" in
  let one = Ptree.tensor in
  let y, pullback = vjp_of fn one one f x in
  (* Row [k] of the pullback is the gradient of [Re y_k], the conjugate of row
     [k] of the Jacobian. *)
  let rows =
    vmap_of fn
      Ptree.(one @-> returns one)
      (fun e -> Nx.conjugate (pullback e))
      (basis y)
  in
  Nx.reshape (Array.append (Nx.shape y) (Nx.shape x)) rows

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

let scan_of fn cs xs_s ys_s ~f ~init xs =
  let req_xs, _ = Ptree.flatten xs_s xs in
  steps fn req_xs;
  let first = ref None in
  let req_step c_leaves x_leaves =
    let c = Ptree.rebuild cs ~like:init c_leaves in
    let c', y = f c (Ptree.rebuild xs_s ~like:xs x_leaves) in
    Structure.check fn cs ~this:"the carry the step returned" c'
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
  let req_trips = Trips.Rows { xs = req_xs; reverse = false } in
  let r = Construct.loop { req_carry; req_trips; req_step } in
  match !first with
  | Some y0 ->
      (Ptree.rebuild cs ~like:init r.r_carry, Ptree.rebuild ys_s ~like:y0 r.r_ys)
  | None -> assert false (* Every answer runs the step at least once. *)

let scan cs xs_s ys_s ~f ~init xs = scan_of "Rune.scan" cs xs_s ys_s ~f ~init xs

let iterate c ~max ~until ~f init =
  let fn = "Rune.iterate" in
  if max < 0 then invalid_argf "%s: max = %d is negative" fn max;
  let carry l = Ptree.rebuild c ~like:init l in
  let until l =
    let u = until (carry l) in
    if Nx.numel u <> 1 then
      invalid_arg
        (Format.asprintf "%s: until must return one boolean, got %a %a" fn
           Nx.pp_dtype (Nx.dtype u) Nx.pp_shape (Nx.shape u));
    Nx.reshape [||] u
  in
  let this = "the carry the step returned" and that = "the carry it received" in
  let req_step l _ =
    let x = carry l in
    let x' = Structure.map2 fn c ~this ~that (fun _ x' _ -> x') (f x) x in
    Structure.check_placements fn c ~this x' ~that x;
    (fst (Ptree.flatten c x'), [])
  in
  let failure _ =
    Printf.sprintf "%s: until is still false after max = %d steps" fn max
  in
  let req_carry, _ = Ptree.flatten c init in
  let req_trips = Trips.Until { until; max; failure } in
  carry (Construct.loop { req_carry; req_trips; req_step }).r_carry

(* Roots *)

(* [dense_in fn x dt op b] is the [v] with [op v = b]: [op]'s matrix over the
   leaves of [x], of dtype [dt], flattened into one vector, one product per
   column under one map, solved with Nx.solve. *)
let dense_in : type a c x.
    string -> x Ptree.t -> (a, c) Nx.dtype -> (x -> x) -> x -> x =
 fun fn x dt op b ->
  let leaves, _ = Ptree.flatten x b in
  let flat v : (a, c) Nx.t =
    Nx.concatenate ~axis:0
      (List.map
         (fun (Nx.P l) ->
           if not (Nx_dtype.equal dt (Nx.dtype l)) then
             invalid_arg
               (Format.asprintf
                  "%s: the default linear solve takes leaves of one dtype, got \
                   %a and %a; pass ~linear_solve"
                  fn Nx.pp_dtype dt Nx.pp_dtype (Nx.dtype l));
           let l = Nx.unpack dt (Nx.P l) in
           Nx.reshape [| Nx.numel l |] l)
         (fst (Ptree.flatten x v)))
  in
  let unflat v =
    let at = ref 0 in
    let leaf (Nx.P l) =
      let n = Nx.numel l in
      let part = Nx.reshape (Nx.shape l) (Nx.slice [ Nx.R (!at, !at + n) ] v) in
      at := !at + n;
      Nx.P (Nx.unpack (Nx.dtype l) (Nx.P part))
    in
    Ptree.rebuild x ~like:b (List.map leaf leaves)
  in
  let fb = flat b in
  let n = Nx.numel fb in
  let columns =
    vmap_of fn
      Ptree.(tensor @-> returns tensor)
      (fun e -> flat (op (unflat e)))
      (Nx.eye dt n)
  in
  let v = Nx.solve (Nx.matrix_transpose columns) (Nx.reshape [| n; 1 |] fb) in
  unflat (Nx.reshape [| n |] v)

let dense fn x op b =
  let leaves, _ = Ptree.flatten x b in
  match leaves with [] -> b | Nx.P l :: _ -> dense_in fn x (Nx.dtype l) op b

let root ?linear_solve x ~residual solve =
  let fn = "Rune.root" in
  let residual v =
    let r = residual v in
    ignore
      (Structure.map2 fn x ~this:"the residual's result" ~that:"the solution"
         (fun _ r _ -> r)
         r v);
    r
  in
  let linear_solve =
    match linear_solve with Some f -> f | None -> dense fn x
  in
  Construct.perform (Root { x; residual; solve; linear_solve })

(* Compilation *)

exception Jit_error = Lower.Jit_error

let jit ?beam ?parallel s f = Jit.jit ?beam ?parallel "Rune.jit" s f

(* Functions of one tensor *)

let t = Ptree.tensor
let grad' f x = snd (value_and_grad_of "Rune.grad'" t f x)
let value_and_grad' f x = value_and_grad_of "Rune.value_and_grad'" t f x
let vjp' f x = vjp_of "Rune.vjp'" t t f x
let jvp' f x dx = jvp_of "Rune.jvp'" t t f x dx
let vmap' ?axis f x = vmap_of "Rune.vmap'" ?axis Ptree.(t @-> returns t) f x
let scan' ~f ~init xs = scan_of "Rune.scan'" t t t ~f ~init xs
let iterate' ~max ~until ~f x = iterate t ~max ~until ~f x

let jit' ?beam ?parallel f =
  Jit.jit ?beam ?parallel "Rune.jit'" Ptree.(tensor @-> returns tensor) f
