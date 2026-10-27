(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Program
module Ptree = Nx.Ptree

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let pp_shape = Nx.pp_shape

(* Structures as values *)

let values s x =
  Array.of_list (List.map (fun (Nx.P y) -> V y) (fst (Ptree.flatten s x)))

(* [vs] as a structure like [like], each at its leaf's type. *)
let rebuild s ~like vs =
  let leaves = fst (Ptree.flatten s like) in
  Ptree.rebuild s ~like
    (List.map2
       (fun (Nx.P l) v -> Nx.P (Program.like l v))
       leaves (Array.to_list vs))

(* Outputs [i] to [j - 1]. *)
let slice a i j = Array.sub a i (j - i)
let first a = a.(0)

let path_name p =
  if Ptree.Path.segments p = [] then "the root" else Ptree.Path.to_string p

(* [x] and [y] of structure [s] alike: the same visits, and at each path the
   same dtype and shape, as [this] and [that] name them. *)
let same_leaves name s ~this x ~that y =
  let _, k = Ptree.flatten s x and _, k' = Ptree.flatten s y in
  Option.iter
    (fun msg -> invalid_argf "%s: %s" name msg)
    (Ptree.Skeleton.diff ~this k ~that k');
  let leaves x =
    List.rev (Ptree.fold s (fun p y acc -> (p, V y) :: acc) x [])
  in
  List.iter2
    (fun (p, V a) (_, V b) ->
      if not (Nx.Dtype.equal (Nx.dtype a) (Nx.dtype b)) then
        invalid_argf "%s: %s: %s %s, %s %s" name (path_name p)
          (Nx.Dtype.name (Nx.dtype a))
          this
          (Nx.Dtype.name (Nx.dtype b))
          that;
      if Nx.shape a <> Nx.shape b then
        invalid_argf "%s: %s: shape %s %s, %s %s" name (path_name p)
          (Format.asprintf "%a" pp_shape (Nx.shape a))
          this
          (Format.asprintf "%a" pp_shape (Nx.shape b))
          that)
    (leaves x) (leaves y)

(* The carry a step returned against the one it received. *)
let same_carry name s ~received returned =
  same_leaves name s ~this:"in the carry the step returned" returned
    ~that:"in the carry it received" received

(* Forward mode *)

let tangible x = Nx.Dtype.(is Float (Nx.dtype x) || is Complex (Nx.dtype x))

let jvp_named name p q f params tangents =
  same_leaves name p ~this:"in the parameters" params ~that:"in the tangents"
    tangents;
  if not (Ptree.fold p (fun _ x acc -> acc || tangible x) params false) then
    invalid_argf "%s: the parameters hold no real or complex tensor" name;
  Jvp.run ~name (fun i ->
      let x =
        Ptree.map2 p
          (fun _ x t -> if tangible x then Jvp.seed i x t else x)
          params tangents
      in
      let y = f x in
      let primals = Ptree.map q (fun _ y -> Jvp.primal i y) y in
      let tangents =
        Ptree.map q
          (fun _ y ->
            match Jvp.tangent i y with
            | Some t -> t
            | None -> Nx.zeros_like (Jvp.primal i y))
          y
      in
      (primals, tangents))

let jvp p q f params tangents = jvp_named "Rune.jvp" p q f params tangents

(* Reverse mode *)

(* The placement of [x]'s first placed leaf, which names the set of every value
   of the brand [s]'s values have. *)
let placement_of s x =
  Ptree.fold s
    (fun _ y acc -> match acc with Some _ -> acc | None -> Nx.placement y)
    x None

let placed_leaf s x =
  Ptree.fold s
    (fun _ y acc ->
      match acc with
      | Some _ -> acc
      | None -> if Option.is_some (Nx.placement y) then Some (V y) else None)
    x None

(* A cotangent at [x]'s type: a placed one beside a parameter of every set lies
   on the set [at] names. *)
let gradient (type v s d) name path (at : d Nx.Placement.t option)
    (x : (v, s, d) Nx.t) (c : Linear.ct option) : (v, s, d) Nx.t =
  match c with
  | None -> Nx.zeros_like x
  | Some { v = V c; fresh } ->
      let g =
        match (Nx.Prim.coerce (Nx.Prim.form x) c, at) with
        | Some g, _ -> g
        | None, Some p -> (
            let f = { (Nx.Prim.form x) with placement = Some p } in
            match Nx.Prim.coerce f c with Some g -> g | None -> like x (V c))
        | None, None ->
            invalid_argf
              "%s: %s: the gradient of a parameter of every set lies on %s, \
               and no parameter or result is placed"
              name (path_name path)
              (Format.asprintf "%a"
                 (Format.pp_print_option Nx.Placement.pp)
                 (Nx.placement c))
      in
      if fresh then g else Nx.copy g

(* [f params]'s primal outputs as [f]'s result [like], and the function from
   their cotangents to the gradient, a value like [params]. *)
let reverse name p f params =
  if not (Ptree.fold p (fun _ x acc -> acc || tangible x) params false) then
    invalid_argf "%s: the parameters hold no real or complex tensor" name;
  let result = ref None in
  let ys, l =
    Linear.linearize ~name (values p params) (fun vs ->
        let r, outs = f (rebuild p ~like:params vs) in
        result := Some r;
        outs)
  in
  (* [at] and [witness] name the set of the result's placed values, where no
     parameter is placed. *)
  let pullback ~at ~witness cts =
    let witness =
      match placed_leaf p params with Some _ as w -> w | None -> witness
    in
    let at = match placement_of p params with Some _ as a -> a | None -> at in
    let gs = Linear.transpose ?witness l cts in
    let k = ref 0 in
    Ptree.map p
      (fun path x ->
        if not (tangible x) then Nx.zeros_like x
        else begin
          let g = gs.(!k) in
          incr k;
          gradient name path at x g
        end)
      params
  in
  (Option.get !result, ys, pullback)

let vjp_named name p q f params =
  let like, ys, pullback =
    reverse name p
      (fun x ->
        let y = f x in
        (y, values q y))
      params
  in
  let y = rebuild q ~like ys in
  let at = placement_of q y and witness = placed_leaf q y in
  let back cts =
    same_leaves name q ~this:"in the result" y ~that:"in the cotangents" cts;
    pullback ~at ~witness
      (Array.map
         (fun (V c) -> if tangible c then Some (V c) else None)
         (values q cts))
  in
  (y, back)

let vjp p q f params = vjp_named "Rune.vjp" p q f params

let objective name y =
  if not (Nx.Dtype.is Float (Nx.dtype y) && Nx.numel y = 1) then
    invalid_argf "%s: the objective must return a real scalar, got %s %s" name
      (Nx.Dtype.name (Nx.dtype y))
      (Format.asprintf "%a" pp_shape (Nx.shape y))

let value_and_grad_aux_named name p x f params =
  let (like, aux), ys, pullback =
    reverse name p
      (fun ps ->
        let y, aux = f ps in
        ((y, aux), Array.append [| V y |] (values x aux)))
      params
  in
  let y = Program.like like ys.(0) in
  objective name y;
  let aux = rebuild x ~like:aux (Array.sub ys 1 (Array.length ys - 1)) in
  let cts = Array.make (Array.length ys) None in
  cts.(0) <- Some (V (Nx.ones_like y));
  let witness = if Option.is_some (Nx.placement y) then Some (V y) else None in
  (y, pullback ~at:(Nx.placement y) ~witness cts, aux)

let value_and_grad_named name p f params =
  let y, g, () =
    value_and_grad_aux_named name p Ptree.unit (fun x -> (f x, ())) params
  in
  (y, g)

let value_and_grad p f params =
  value_and_grad_named "Rune.value_and_grad" p f params

let value_and_grad_aux p x f params =
  value_and_grad_aux_named "Rune.value_and_grad_aux" p x f params

let grad p f params = snd (value_and_grad_named "Rune.grad" p f params)

type 'd move = { move : 'v 's. ('v, 's, 'd) Nx.t -> ('v, 's, 'd) Nx.t }

let check_grads ?(eps = 1e-4) ?(tol = 1e-2) p f params =
  let name = "Rune.check_grads" in
  let _, g = value_and_grad_named name p f params in
  let read t = Nx.item [] (Nx.reshape [||] (Nx.cast Nx.float64 t)) in
  let lit x c = Nx.scalar_like x (Nx.Dtype.of_float (Nx.dtype x) c) in
  let direction { move } =
    Ptree.map p
      (fun _ x -> if tangible x then move x else Nx.zeros_like x)
      params
  in
  let directions =
    [
      ("ones", direction { move = Nx.ones_like });
      ( "params-derived",
        direction { move = (fun x -> Nx.add (Nx.sin x) (lit x 1.1)) } );
    ]
  in
  let check (name', v) =
    let bump s =
      Ptree.map2 p (fun _ x v -> Nx.add x (Nx.mul v (lit v s))) params v
    in
    let numeric =
      (read (f (bump eps)) -. read (f (bump (-.eps)))) /. (2. *. eps)
    in
    (* [Re Σ g v], the derivative the gradient predicts along [v]. *)
    let analytic = ref 0. in
    ignore
      (Ptree.map2 p
         (fun _ g v ->
           analytic := !analytic +. read (Nx.sum (Nx.mul g v));
           g)
         g v);
    let analytic = !analytic in
    if
      Float.abs (analytic -. numeric) <= tol *. Float.max 1. (Float.abs numeric)
    then Ok ()
    else
      Error
        (Printf.sprintf
           "%s: the directional derivative along %s is %g, and the gradient \
            predicts %g"
           name name' numeric analytic)
  in
  List.fold_left
    (fun acc d -> match acc with Ok () -> check d | Error _ -> acc)
    (Ok ()) directions

(* Constructs without programs *)

let detach x = like x (first (perform (Detach x)))

type nonrec axis = axis

let axis = axis
let lanes a x = like x (first (perform (Lanes (a, x))))

(* A lane's index is of every set, so it reads back at any. *)
let lane_index a () =
  let f : (int32, Nx.int32_elt, _) Nx.Prim.form =
    {
      dtype = Nx.int32;
      layout = Nx_array.Layout.contiguous (Iarray.of_list []);
      placement = None;
    }
  in
  cast f (first (perform (Lane_index a)))

module Total = struct
  type ('v, 's, 'd) t = ('v, 's, 'd) total

  let make = total
  let add t x = ignore (perform (Add (t, x)))

  let collect (type v s d) (t : (v, s, d) t) ~(zero : (v, s, d) Nx.t) f =
    let sum = ref zero in
    let effc (type a) (e : a Effect.t) :
        ((a, _) Effect.Deep.continuation -> _) option =
      match e with
      | Construct (Add (u, x)) -> (
          match same_total u t with
          | None -> None
          | Some Type.Equal ->
              Some
                (fun k ->
                  if Nx.shape x <> Nx.shape zero then
                    Effect.Deep.discontinue k
                      (Invalid_argument
                         (Format.asprintf
                            "Rune.Total.add: shape %a, where the total's zero \
                             has %a"
                            pp_shape (Nx.shape x) pp_shape (Nx.shape zero)))
                  else begin
                    sum := Nx.add !sum x;
                    Effect.Deep.continue k [||]
                  end))
      | _ -> None
    in
    let r = Effect.Deep.match_with f () { retc = Fun.id; exnc = raise; effc } in
    (r, !sum)
end

(* Loops *)

(* The form of [x]'s rows along axis 0. *)
let row (V x) =
  let s = Nx.shape x in
  F
    {
      dtype = Nx.dtype x;
      layout =
        Nx_array.Layout.contiguous
          (Iarray.of_array (Array.sub s 1 (Array.length s - 1)));
      placement = Option.map Nx.Placement.without_leading_axis (Nx.placement x);
    }

(* Whether a carry leaf traced at [r] is traced again at [t], the form its step
   returned it at: traced of every set and returned placed. A value is read back
   by its set, so a leaf placed elsewhere on one set, or returned of every set,
   binds as traced. *)
let moved (F r) (F t) =
  match (r.placement, t.placement) with None, Some _ -> true | _ -> false

(* [trace like forms] traced until its carry, its first [nc] forms, settles.
   Each trace rebuilds the step's carry like [like]. A leaf of every set the
   step returns placed is traced again at the form it was returned at, as JAX
   traces a loop again whose weakly typed carry its step promotes: its leaf of
   [like] becomes the returned one, the others stay. A placed leaf stays placed,
   so at most [nc] traces follow the first. With the [like] and the forms it
   settled at. *)
let settling c ~nc trace like forms =
  let rec go like forms =
    let ((_, _, returned, carry') as traced) = trace like forms in
    let moves = Array.init nc (fun i -> moved forms.(i) returned.(i)) in
    if not (Array.exists Fun.id moves) then (traced, like, forms)
    else
      let forms =
        Array.mapi
          (fun i (F r as fr) ->
            if i < nc && moves.(i) then
              match returned.(i) with F t -> F { t with layout = r.layout }
            else fr)
          forms
      in
      let before = Array.of_list (fst (Ptree.flatten c like)) in
      let after = Array.of_list (fst (Ptree.flatten c carry')) in
      let leaves =
        List.init nc (fun i -> if moves.(i) then after.(i) else before.(i))
      in
      go (Ptree.rebuild c ~like leaves) forms
  in
  go like forms

let scan_named name c x y ~f ~init xs =
  let init_v = values c init and xs_v = values x xs in
  if Array.length xs_v = 0 then invalid_argf "%s: xs has no leaf" name;
  let lengths =
    List.rev
      (Ptree.fold x
         (fun p v acc ->
           match Nx.shape v with
           | [||] -> invalid_argf "%s: xs: %s is 0-d" name (path_name p)
           | s -> (path_name p, s.(0)) :: acc)
         xs [])
  in
  let first, length = List.hd lengths in
  List.iter
    (fun (p, n) ->
      if n <> length then
        invalid_argf "%s: xs: %s: %d rows, %s: %d" name p n first length)
    lengths;
  let nc = Array.length init_v and nx = Array.length xs_v in
  let ys_like = ref None in
  let trace like forms =
    let returned = ref None in
    let step, captures =
      Trace.trace Body ~name forms (fun vs ->
          let carry = rebuild c ~like (slice vs 0 nc) in
          let r = rebuild x ~like:xs (slice vs nc (nc + nx)) in
          let carry', ys = f carry r in
          same_carry name c ~received:carry carry';
          ys_like := Some ys;
          returned := Some carry';
          Array.append (values c carry') (values y ys))
    in
    let carry' = Option.get !returned in
    (step, captures, Array.map form (values c carry'), carry')
  in
  let (step, captures, _, _), like, _ =
    settling c ~nc trace init
      (Array.append (Array.map form init_v) (Array.map row xs_v))
  in
  let ys_like = Option.get !ys_like in
  let ny = Array.length (values y ys_like) in
  let r = run (Scan { step; length; init = init_v; xs = xs_v; captures }) in
  ( rebuild c ~like (slice r 0 nc),
    rebuild y ~like:ys_like (slice r nc (nc + ny)) )

let scan c x y ~f ~init xs = scan_named "Rune.scan" c x y ~f ~init xs

let iterate_named name c ~max ~until ~f init =
  if max < 0 then invalid_argf "%s: max = %d is negative" name max;
  let init_v = values c init in
  let nc = Array.length init_v in
  let trace like forms =
    let returned = ref None in
    let step, captures =
      Trace.trace Body ~name forms (fun vs ->
          let carry = rebuild c ~like vs in
          let carry' = f carry in
          same_carry name c ~received:carry carry';
          returned := Some carry';
          values c carry')
    in
    let carry' = Option.get !returned in
    (step, captures, Array.map form (values c carry'), carry')
  in
  let (step, step_captures, _, _), like, forms =
    settling c ~nc trace init (Array.map form init_v)
  in
  let until, until_captures =
    Trace.trace Body ~name forms (fun vs ->
        let stop = until (rebuild c ~like vs) in
        let s = Nx.shape stop in
        if Array.fold_left ( * ) 1 s <> 1 then
          invalid_argf "%s: until must return one boolean, got shape %s" name
            (Format.asprintf "%a" pp_shape s);
        [| V stop |])
  in
  let r =
    run
      (Iterate
         {
           name;
           step;
           until;
           max;
           init = init_v;
           step_captures;
           until_captures;
         })
  in
  rebuild c ~like (slice r 0 nc)

let iterate c ~max ~until ~f init =
  iterate_named "Rune.iterate" c ~max ~until ~f init

(* Compilation *)

(* A signature as one structure of its arguments, numbered from [0]. *)
type ('s, 'd) walker = {
  walk : 'a 'b. ('a, 'b, 'd, 'd) Ptree.Walk.cursor -> 's -> 's;
}

type ('f, 'd) spine =
  | Spine : {
      walk : ('a, 'd) walker;
      result : ('r, 'd) Ptree.t;
      apply : 'f -> 'a -> 'r;
      curry : ('a -> 'r) -> 'f;
    }
      -> ('f, 'd) spine

let rec spine : type f d. string -> int -> (f, d) Ptree.fn -> (f, d) spine =
 fun name k -> function
  | Arg (a, Returns result) ->
      let walk c x = Ptree.Walk.index c k (Ptree.Walk.structure a) x in
      Spine
        { walk = { walk }; result; apply = (fun f x -> f x); curry = Fun.id }
  | Arg (a, rest) ->
      let (Spine u) = spine name (k + 1) rest in
      let walk c (x, xs) =
        let x = Ptree.Walk.index c k (Ptree.Walk.structure a) x in
        let xs = u.walk.walk c xs in
        (x, xs)
      in
      Spine
        {
          walk = { walk };
          result = u.result;
          apply = (fun f (x, xs) -> u.apply (f x) xs);
          curry = (fun g x -> u.curry (fun xs -> g (x, xs)));
        }
  | Returns _ -> invalid_argf "%s: the signature has no argument" name

let call name a r f args =
  let args_v = values a args in
  let like = ref None in
  let body, _ =
    Trace.trace Jit ~name (Array.map form args_v) (fun vs ->
        let y = f (rebuild a ~like:args vs) in
        like := Some y;
        values r y)
  in
  let y = Option.get !like in
  let out = run (Call { body; args = args_v }) in
  rebuild r ~like:y (slice out 0 (Array.length (values r y)))

let jit_named name s =
  let (Spine u) = spine name 0 s in
  let a = Ptree.v u.walk.walk in
  fun f -> u.curry (fun args -> call name a u.result (u.apply f) args)

let jit s = jit_named "Rune.jit" s

(* Functions of one array *)

let scan' ~f ~init xs =
  scan_named "Rune.scan'" Ptree.tensor Ptree.tensor Ptree.tensor ~f ~init xs

let iterate' ~max ~until ~f x =
  iterate_named "Rune.iterate'" Ptree.tensor ~max ~until ~f x

let jit' f = jit_named "Rune.jit'" Ptree.(tensor @-> returns tensor) f
let jvp' f x t = jvp_named "Rune.jvp'" Ptree.tensor Ptree.tensor f x t

let jacfwd' f x =
  let name = "Rune.jacfwd'" in
  let sx = Nx.shape x and n = Nx.numel x in
  let iota = Nx.arange Nx.int32 0 n 1 in
  let column k =
    let e = Nx.equal iota (Nx.scalar_like iota (Int32.of_int k)) in
    snd
      (jvp_named name Ptree.tensor Ptree.tensor f x
         (Nx.reshape sx (Nx.cast (Nx.dtype x) e)))
  in
  if n = 0 then
    (* No column: one pass along the empty tangent gives the result's form, and
       raises as every column would. *)
    let t = snd (jvp_named name Ptree.tensor Ptree.tensor f x x) in
    Nx.zeros (Nx.dtype t) (Array.append (Nx.shape t) sx)
  else
    let columns = List.init n column in
    let sy = Nx.shape (List.hd columns) in
    Nx.reshape (Array.append sy sx) (Nx.stack ~axis:(Array.length sy) columns)

let grad' f x = snd (value_and_grad_named "Rune.grad'" Ptree.tensor f x)

let value_and_grad' f x =
  value_and_grad_named "Rune.value_and_grad'" Ptree.tensor f x

let vjp' f x = vjp_named "Rune.vjp'" Ptree.tensor Ptree.tensor f x

let jacrev' f x =
  let name = "Rune.jacrev'" in
  let y, pullback = vjp_named name Ptree.tensor Ptree.tensor f x in
  let complex x = Nx.Dtype.is Complex (Nx.dtype x) in
  if complex y && not (complex x) then
    invalid_argf
      "%s: the result is complex and the argument real, so the Jacobian has no \
       place in the argument's dtype; use Rune.jacfwd'"
      name;
  let sy = Nx.shape y and n = Nx.numel y in
  let iota = Nx.arange Nx.int32 0 n 1 in
  (* Row [k] is the pullback of element [k] of the result in C order. *)
  let row k =
    let e = Nx.equal iota (Nx.scalar_like iota (Int32.of_int k)) in
    pullback (Nx.reshape sy (Nx.cast (Nx.dtype y) e))
  in
  let shape = Array.append sy (Nx.shape x) in
  if n = 0 then Nx.zeros (Nx.dtype x) shape
  else Nx.reshape shape (Nx.stack ~axis:0 (List.init n row))
