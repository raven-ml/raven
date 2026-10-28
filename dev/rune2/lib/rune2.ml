(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Program
module Ptree = Nx.Ptree

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* Re-raised with the backtrace of its raise, which a plain [raise] would
   restart at the handler. *)
let reraise e = Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())

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

(* A cotangent at [x]'s type, memory of its own. *)
let gradient x (c : Linear.ct option) =
  match c with
  | None -> Nx.zeros_like x
  | Some { v; fresh } ->
      let g = like x v in
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
  let pullback cts =
    let gs = Linear.transpose l cts in
    let k = ref 0 in
    Ptree.map p
      (fun _ x ->
        if not (tangible x) then Nx.zeros_like x
        else begin
          let g = gs.(!k) in
          incr k;
          gradient x g
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
  let back cts =
    same_leaves name q ~this:"in the result" y ~that:"in the cotangents" cts;
    pullback
      (Array.map
         (fun (V c) -> if tangible c then Some (V c) else None)
         (values q cts))
  in
  (y, back)

let vjp p q f params = vjp_named "Rune.vjp" p q f params

let linearize_named name p q f params =
  if not (Ptree.fold p (fun _ x acc -> acc || tangible x) params false) then
    invalid_argf "%s: the parameters hold no real or complex tensor" name;
  let like = ref None in
  let ys, l =
    Linear.linearize ~name (values p params) (fun vs ->
        let y = f (rebuild p ~like:params vs) in
        like := Some y;
        values q y)
  in
  let y = rebuild q ~like:(Option.get !like) ys in
  let apply tangents =
    same_leaves name p ~this:"in the parameters" params ~that:"in the tangents"
      tangents;
    let ts =
      Array.of_list
        (List.filter
           (fun (V x) -> tangible x)
           (Array.to_list (values p tangents)))
    in
    let out = Linear.apply l ts in
    (* A tangent is a value of its own: the function may return a tangent it
       received, or a view of one. *)
    rebuild q ~like:y
      (Array.mapi
         (fun k (V y) ->
           match out.(k) with
           | Some (V t) -> V (Nx.copy (Program.like y (V t)))
           | None -> V (Nx.zeros_like y))
         ys)
  in
  (y, apply)

let linearize p q f params = linearize_named "Rune.linearize" p q f params

let linear_transpose_named name p q f params =
  let shapes = ref None in
  let l =
    Linear.trace ~name (values p params) (fun vs ->
        (* A transpose offers nothing. *)
        let y = Program.silenced (fun () -> f (rebuild p ~like:params vs)) in
        shapes :=
          Some (Ptree.map q (fun _ y -> Nx.zeros_like y) y);
        values q y)
  in
  let like = Option.get !shapes in
  fun cts ->
    same_leaves name q ~this:"in the result" like ~that:"in the cotangents" cts;
    let gs =
      Linear.transpose l
        (Array.map
           (fun (V c) -> if tangible c then Some (V c) else None)
           (values q cts))
    in
    let k = ref (-1) in
    Ptree.map p
      (fun _ x ->
        incr k;
        if tangible x then gradient x gs.(!k) else Nx.zeros_like x)
      params

let linear_transpose p q f params =
  linear_transpose_named "Rune.linear_transpose" p q f params

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
  (y, pullback cts, aux)

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

(* Custom rules *)

let custom_jvp p q rule args =
  let name = "Rune.custom_jvp" in
  let like = ref None in
  let ys =
    Custom.jvp_call ~name (values p args) (fun vs ->
        let x = rebuild p ~like:args vs in
        let y, map = rule x in
        like := Some y;
        let tangent ts =
          let dy = map (rebuild p ~like:x ts) in
          same_leaves name q ~this:"in the tangent map's result" dy
            ~that:"in the result" y;
          values q dy
        in
        (values q y, tangent))
  in
  rebuild q ~like:(Option.get !like) ys

let custom_vjp p q rule args =
  let name = "Rune.custom_vjp" in
  let like = ref None in
  let ys =
    Custom.vjp_call ~name (values p args) (fun vs ->
        let x = rebuild p ~like:args vs in
        let y, pb = rule x in
        like := Some y;
        let pullback cts =
          let g = pb (rebuild q ~like:y cts) in
          same_leaves name p ~this:"in the pullback's result" g
            ~that:"in the arguments" x;
          values p g
        in
        (values q y, pullback))
  in
  rebuild q ~like:(Option.get !like) ys

(* [x] and [y] of structure [s] alike in their placements too. *)
let same_placements name s ~this x ~that y =
  let placements v =
    List.rev (Ptree.fold s (fun p y acc -> (p, Nx.placement y) :: acc) v [])
  in
  List.iter2
    (fun (p, a) (_, b) ->
      if not (Nx.Placement.equal a b) then
        invalid_argf "%s: %s: placed on %s %s, on %s %s" name (path_name p)
          (Format.asprintf "%a" Nx.Placement.pp a)
          this
          (Format.asprintf "%a" Nx.Placement.pp b)
          that)
    (placements x) (placements y)

let custom_transpose p q f ~transpose args =
  let name = "Rune.custom_transpose" in
  let like = ref None in
  let ys =
    Custom.transpose_call ~name (values p args) (fun vs ->
        let x = rebuild p ~like:args vs in
        let y = f x in
        like := Some y;
        let transposed cts =
          let g = transpose (rebuild q ~like:y cts) in
          let this = "in the transpose's result"
          and that = "in the arguments" in
          same_leaves name p ~this g ~that x;
          same_placements name p ~this g ~that x;
          values p g
        in
        (values q y, transposed))
  in
  rebuild q ~like:(Option.get !like) ys

(* A draw like [x], from the key [k]: standard normal parts. *)
let draw (type v s d) k (x : (v, s, d) Nx.t) : (v, s, d) Nx.t =
  let normal k = Nx.Rng.normal ~key:(Nx.Rng.key k) Nx.float64 (Nx.shape x) in
  let v =
    if Nx.Dtype.is Complex (Nx.dtype x) then
      Nx.cast (Nx.dtype x)
        (Nx.complex Nx.complex128
           ~re:(normal (2 * k))
           ~im:(normal ((2 * k) + 1)))
    else Nx.cast (Nx.dtype x) (normal (2 * k))
  in
  Nx.place (Nx.placement x) v

let check_transpose ?(tol = 1e-5) p q f ~transpose args =
  let name = "Rune.check_transpose" in
  let read t = Nx.item [] (Nx.reshape [||] (Nx.cast Nx.float64 t)) in
  (* [Re Σ a b] over the arrays of [a] and [b], and [Σ |a b|], in complex128, so
     that no product of a narrow format overflows. A complex value cast to
     float64 is its real part. *)
  let pairing s a b =
    let dot = ref 0. and size = ref 0. in
    ignore
      (Ptree.map2 s
         (fun _ a b ->
           let wide x = Nx.cast Nx.complex128 x in
           let ab = Nx.mul (wide a) (wide b) in
           let modulus =
             Nx.sqrt (Nx.cast Nx.float64 (Nx.mul ab (Nx.conjugate ab)))
           in
           dot := !dot +. read (Nx.sum ab);
           size := !size +. read (Nx.sum modulus);
           a)
         a b);
    (!dot, !size)
  in
  let check pair =
    let leaf = ref 0 in
    let random s like =
      Ptree.map s
        (fun _ x ->
          incr leaf;
          if tangible x then draw ((1000 * pair) + !leaf) x else Nx.zeros_like x)
        like
    in
    let v = random p args in
    let lv = f v in
    let w = random q lv in
    let ltw = transpose w in
    same_leaves name q ~this:"in the result" lv ~that:"in the cotangents" w;
    same_leaves name p ~this:"in the transpose's result" ltw
      ~that:"in the arguments" args;
    let a, sa = pairing q w lv and b, sb = pairing p ltw v in
    if Float.abs (a -. b) <= tol *. Float.max 1. (Float.max sa sb) then Ok ()
    else
      Error
        (Printf.sprintf "%s: pair %d: <w, L v> is %g, and <Lᵀ w, v> is %g" name
           pair a b)
  in
  List.fold_left
    (fun acc pair -> match acc with Ok () -> check pair | Error _ -> acc)
    (Ok ()) [ 1; 2; 3 ]

(* Constructs without programs *)

let detach x = like x (first (perform (Detach x)))

type nonrec axis = axis

let axis = axis
let lanes a x = like x (first (perform (Lanes (a, x))))

(* A lane's index lies whole on each of [x]'s devices. *)
let lane_index a x =
  let placement = Nx.beside x in
  let index = first (perform (Lane_index (a, placement))) in
  cast Nx.int32 (Nx.Placement.devices placement) index

module Total = struct
  type ('v, 's, 'd) t = ('v, 's, 'd) total

  let make = total
  (* An addition the collect would refuse raises where it is made, also in a
     traced function, so that a construct that raises has delivered nothing. *)
  let add t x =
    accepts t (Nx.Prim.form x);
    add_to t x

  let collect (type v s d) (t : (v, s, d) t) ~(zero : (v, s, d) Nx.t) f =
    let sum = ref zero in
    let refused shape =
      Invalid_argument
        (Format.asprintf
           "Rune.Total.add: shape %a, where the total's zero has %a"
           pp_shape shape pp_shape (Nx.shape zero))
    in
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
                    Effect.Deep.discontinue k (refused (Nx.shape x))
                  else begin
                    sum := Nx.add !sum x;
                    Effect.Deep.continue k [||]
                  end))
      | Accepts (u, f) -> (
          match same_total u t with
          | None -> None
          | Some Type.Equal ->
              let shape = Iarray.to_array (Nx_array.Layout.shape f.layout) in
              Some
                (fun k ->
                  if shape <> Nx.shape zero then
                    Effect.Deep.discontinue k (refused shape)
                  else Effect.Deep.continue k ()))
      | Running (u, f) -> (
          match same_total u t with
          | Some Type.Equal ->
              let shape = Iarray.to_array (Nx_array.Layout.shape f.layout) in
              Some
                (fun k ->
                  Effect.Deep.continue k
                    (if shape = Nx.shape zero then Some !sum else None))
          | None -> None)
      | Replace (u, x) -> (
          match same_total u t with
          | Some Type.Equal ->
              Some
                (fun k ->
                  sum := x;
                  Effect.Deep.continue k ())
          | None -> None)
      | Where u -> (
          match same_total u t with
          | Some Type.Equal ->
              Some
                (fun k ->
                  Effect.Deep.continue k (At (Some (Nx.placement !sum))))
          | None -> None)
      | _ -> None
    in
    let r = Effect.Deep.match_with f () { retc = Fun.id; exnc = reraise; effc } in
    (r, !sum)
end

(* Probes *)

module Probe = struct
  type 'a t = 'a Program.probe

  let make s = Program.probe s

  (* Nothing is walked or allocated before the one query. *)
  let emit p v =
    if Program.opened p then begin
      let (Program.Probe { tree; _ }) = p in
      let row (Nx.P x) = V (Nx.reshape (Array.append [| 1 |] (Nx.shape x)) x) in
      let values = Array.of_list (List.map row (fst (Ptree.flatten tree v))) in
      let site = Program.fresh_site () in
      let e =
        Program.Emission
          { probe = p; like = v; site; lead = 0; values; mask = None }
      in
      ignore (Program.perform (Emit e))
    end

  (* [x] with its first [n] axes merged into one. *)
  let merged n (V x) =
    let s = Nx.shape x in
    let k = Array.fold_left ( * ) 1 (Array.sub s 0 n) in
    V (Nx.reshape (Array.append [| k |] (Array.sub s n (Array.length s - n))) x)

  (* The rows of [vs] that [mask] holds, the mask read once on the host. *)
  let compact (V mask) vs =
    let ran : bool array =
      match Nx.Dtype.equal_witness (Nx.dtype mask) Nx.bool with
      | Some Type.Equal -> Nx.to_array mask
      | None -> invalid_arg "Rune.Probe.collect: a mask that is not a boolean"
    in
    if Array.for_all Fun.id ran then vs
    else
      let rows = ref [] in
      Array.iteri (fun j b -> if b then rows := Int64.of_int j :: !rows) ran;
      let rows = Array.of_list (List.rev !rows) in
      let at = Nx.create Nx.int64 [| Array.length rows |] rows in
      Array.map
        (fun (V x) ->
          let at = Nx.place (Nx.beside x) at in
          V (Nx.take ~axis:0 at x))
        vs

  let rows (V x) = (Nx.shape x).(0)

  (* An offer's lanes and rows merged into its first axis, the rows that did not
     run dropped. *)
  let offer (type a) (p : a t) (Program.Emission e) : a option =
    match Program.same_probe e.probe p with
    | None -> invalid_arg "Rune.Probe.collect: an emission of another probe"
    | Some Type.Equal ->
        let (Program.Probe { tree; _ }) = p in
        let values = Array.map (merged (e.lead + 1)) e.values in
        let values =
          match e.mask with
          | None -> values
          | Some m -> compact (merged (e.lead + 1) m) values
        in
        (* A site that ran no row offers nothing. *)
        let ran = match values with [||] -> true | vs -> rows vs.(0) > 0 in
        if ran then Some (rebuild tree ~like:e.like values) else None

  let collect p f =
    (* A construct inside an interpretation may be answered outside the collect,
       where its emissions would pass it by. *)
    if Program.within () then
      invalid_arg
        "Rune.Probe.collect: inside a traced function, a map or a \
         differentiation";
    let offers = ref [] in
    let effc (type a) (e : a Effect.t) :
        ((a, _) Effect.Deep.continuation -> _) option =
      match e with
      | Program.Open q when Option.is_some (Program.same_probe q p) ->
          Some (fun k -> Effect.Deep.continue k true)
      | Construct (Emit (Emission em as m))
        when Option.is_some (Program.same_probe em.probe p) ->
          Some
            (fun k ->
              offers := m :: !offers;
              Effect.Deep.continue k [||])
      | _ -> None
    in
    let body () =
      match f () with
      | r -> Ok r
      | exception x -> Error (x, Printexc.get_raw_backtrace ())
    in
    let r = Effect.Deep.try_with body () { effc } in
    (r, List.filter_map (offer p) (List.rev !offers))
end

(* Loops *)


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
  let forms =
    Array.append (Array.map form init_v) (Array.map Loops.row xs_v)
  in
  let step, captures =
    Trace.trace Body ~name forms (fun vs ->
        let carry = rebuild c ~like:init (slice vs 0 nc) in
        let r = rebuild x ~like:xs (slice vs nc (nc + nx)) in
        let carry', ys = f carry r in
        same_carry name c ~received:carry carry';
        ys_like := Some ys;
        Array.append (values c carry') (values y ys))
  in
  let ys_like = Option.get !ys_like in
  let ny = Array.length (values y ys_like) in
  let r =
    run
      (Scan
         {
           step;
           length;
           trips = None;
           reverse = false;
           init = init_v;
           xs = xs_v;
           captures;
           seeds = [||];
         })
  in
  ( rebuild c ~like:init (slice r 0 nc),
    rebuild y ~like:ys_like (slice r nc (nc + ny)) )

let scan c x y ~f ~init xs = scan_named "Rune.scan" c x y ~f ~init xs

let iterate_named name c ~max ~until ~f init =
  if max < 0 then invalid_argf "%s: max = %d is negative" name max;
  let init_v = values c init in
  let nc = Array.length init_v in
  let forms = Array.map form init_v in
  let step, step_captures =
    Trace.trace Body ~name forms (fun vs ->
        let carry = rebuild c ~like:init vs in
        let carry' = f carry in
        same_carry name c ~received:carry carry';
        values c carry')
  in
  let until, until_captures =
    Trace.trace Body ~name forms (fun vs ->
        let stop = until (rebuild c ~like:init vs) in
        let s = Nx.shape stop in
        if Array.fold_left ( * ) 1 s <> 1 then
          invalid_argf "%s: until must return one boolean, got shape %s" name
            (Format.asprintf "%a" pp_shape s);
        [| V stop |])
  in
  (* Transformations test the stop before the loop and inside it apart, which
     would offer one site's tests apart. *)
  if until.emits then invalid_argf "%s: until emits to a probe; emit in f" name;
  let r =
    run
      (Iterate
         {
           name;
           step;
           until;
           max;
           rows = None;
           stopped = None;
           init = init_v;
           step_captures;
           until_captures;
           seeds = [||];
         })
  in
  rebuild c ~like:init (slice r 0 nc)

let iterate c ~max ~until ~f init =
  iterate_named "Rune.iterate" c ~max ~until ~f init

(* Branches *)

(* Every branch traced at the call, every operation staged, each over every
   branch's captures; the taken one evaluated. *)
let switch_named name s (index : _ Nx.int32_t) branches =
  let branches = Array.of_list branches in
  let n = Array.length branches in
  if n = 0 then invalid_argf "%s: no branches" name;
  if Nx.ndim index <> 0 then
    invalid_argf "%s: the index has shape %s, not []" name
      (Format.asprintf "%a" pp_shape (Nx.shape index));
  let likes = Array.make n None in
  let traced =
    Array.mapi
      (fun k b ->
        Trace.trace Staged ~name [||] (fun _ ->
            let y = b () in
            likes.(k) <- Some y;
            values s y))
      branches
  in
  let likes = Array.map Option.get likes in
  Array.iteri
    (fun k y ->
      if k > 0 then
        same_leaves name s
          ~this:(Printf.sprintf "in branch %d" k)
          y ~that:"in branch 0" likes.(0))
    likes;
  (* Each result lies where branch 0's does. *)
  let leaves = Array.map (fun y -> fst (Ptree.flatten s y)) likes in
  let paths =
    List.rev (Ptree.fold s (fun p _ acc -> path_name p :: acc) likes.(0) [])
  in
  List.iteri
    (fun j path ->
      let at k =
        let (Nx.P l) = List.nth leaves.(k) j in
        Nx.placement l
      in
      for k = 1 to n - 1 do
        if not (Nx.Placement.equal (at k) (at 0)) then
          invalid_argf "%s: %s: on %s in branch %d, on %s in branch 0" name path
            (Format.asprintf "%a" Nx.Placement.pp (at k))
            k
            (Format.asprintf "%a" Nx.Placement.pp (at 0))
      done)
    paths;
  let like = likes.(0) in
  let programs, captures = Program.unite traced ~params:0 in
  let r =
    run
      (Switch
         { name; index = V index; branches = programs; captures; seeds = [||] })
  in
  rebuild s ~like (slice r 0 (Array.length (values s like)))

let switch s index branches = switch_named "Rune.switch" s index branches
let cond_named name s p t f = switch_named name s (Nx.cast Nx.int32 p) [ f; t ]
let cond s p t f = cond_named "Rune.cond" s p t f
let cond' p t f = cond_named "Rune.cond'" Ptree.tensor p t f

(* Roots *)

let operators = Atomic.make 0

(* [p] over its [nargs] arguments then its data, of [forms], with its captures
   [caps], mapped over [lanes] maps, outermost first: the arguments hold every
   lane's along leading axes of those lengths, and a datum where its map's mask
   says. *)
let rec mapped name (p, caps) ~nargs forms lanes =
  match lanes with
  | [] -> (p, caps)
  | (n, mask) :: rest ->
      let inner, icaps = mapped name (p, caps) ~nargs forms rest in
      let laned m k = k < nargs || m.(k - nargs) in
      let stack k f =
        List.fold_right
          (fun (n, m) f -> if laned m k then Program.stacked n f else f)
          lanes f
      in
      Trace.trace Body ~name (Array.mapi stack forms) (fun bs ->
          Vmap.run ~name ~axis:(Program.axis ()) n (fun w ->
              let vs =
                Array.mapi
                  (fun k (V b as v) ->
                    if laned mask k then V (Vmap.lane w b) else v)
                  bs
              in
              let out = Program.eval inner (Array.append vs icaps) in
              Array.map (fun (V y) -> V (Vmap.batch w n y)) out))

(* The default linear solve: the operator's matrix, its columns the operator of
   the unit vectors, computed in one map over them, solved by [Nx.solve]. The
   arrays are flattened into one vector, so they share a dtype. A complex
   operator need only be real-linear: its matrix is of the real system over the
   parts, the operator of the real and the imaginary unit vectors. *)
let dense name forms op bs =
  let (V b0) = bs.(0) in
  Array.iter
    (fun (V b) ->
      if not (Nx.Dtype.equal (Nx.dtype b) (Nx.dtype b0)) then
        invalid_argf
          "%s: the default linear solve takes arrays of one dtype, not %s and \
           %s"
          name
          (Nx.Dtype.name (Nx.dtype b0))
          (Nx.Dtype.name (Nx.dtype b)))
    bs;
  let sizes = Array.map (fun (V b) -> Nx.numel b) bs in
  let n = Array.fold_left ( + ) 0 sizes in
  let flat vs =
    Nx.concatenate ~axis:0
      (Array.to_list
         (Array.map (fun v -> Nx.reshape [| -1 |] (Program.like b0 v)) vs))
  in
  let unflat v =
    let at = ref 0 in
    Array.mapi
      (fun k (F f) ->
        let piece = Nx.slice [ Nx.R (!at, !at + sizes.(k)) ] v in
        at := !at + sizes.(k);
        V (Nx.reshape (Iarray.to_array (Nx_array.Layout.shape f.layout)) piece))
      forms
  in
  (* A complex operator's real system: a vector of [n] complex numbers as its
     [2n] parts, real then imaginary. *)
  let parts (type c q d) (z0 : (Complex.t, c, d) Nx.t)
      (cdt : (Complex.t, c) Nx.dtype) (fdt : (float, q) Nx.dtype) =
    let flatc vs =
      Nx.concatenate ~axis:0
        (Array.to_list
           (Array.map (fun v -> Nx.reshape [| -1 |] (Program.like z0 v)) vs))
    in
    let real z = Nx.concatenate ~axis:0 [ Nx.real fdt z; Nx.imag fdt z ] in
    let whole r =
      Nx.complex cdt
        ~re:(Nx.slice [ Nx.R (0, n) ] r)
        ~im:(Nx.slice [ Nx.R (n, 2 * n) ] r)
    in
    let m = 2 * n in
    let iota = Nx.arange Nx.int32 0 m 1 in
    let eye =
      Nx.cast fdt
        (Nx.equal (Nx.reshape [| m; 1 |] iota) (Nx.reshape [| 1; m |] iota))
    in
    let rows =
      Vmap.run ~name ~axis:(Program.axis ()) m (fun w ->
          Vmap.batch w m (real (flatc (op (unflat (whole (Vmap.lane w eye)))))))
    in
    unflat (whole (Nx.solve (Nx.matrix_transpose rows) (real (flatc bs))))
  in
  if n = 0 then bs
  else
    match Nx.Dtype.equal_witness (Nx.dtype b0) Nx.complex64 with
    | Some Type.Equal -> parts b0 Nx.complex64 Nx.float32
    | None -> (
        match Nx.Dtype.equal_witness (Nx.dtype b0) Nx.complex128 with
        | Some Type.Equal -> parts b0 Nx.complex128 Nx.float64
        | None ->
            let iota = Nx.arange Nx.int32 0 n 1 in
            let eye =
              Nx.cast (Nx.dtype b0)
                (Nx.equal
                   (Nx.reshape [| n; 1 |] iota)
                   (Nx.reshape [| 1; n |] iota))
            in
            let rows =
              Vmap.run ~name ~axis:(Program.axis ()) n (fun w ->
                  let column = flat (op (unflat (Vmap.lane w eye))) in
                  Vmap.batch w n column)
            in
            unflat (Nx.solve (Nx.matrix_transpose rows) (flat bs)))

let root_named name ?linear_solve x ~residual solve =
  (* The solution, of [solve]'s code with no derivative: every operation staged,
     so that it computes on its captures detached. *)
  let like = ref None in
  let sprog, scaps =
    Trace.trace Staged ~name [||] (fun _ ->
        let y = solve () in
        like := Some y;
        values x y)
  in
  let y0 = Option.get !like in
  (* The space the operators act on: the solution's arrays, contiguous. *)
  let y_like = y0 in
  let scaps = Array.map (fun (V c) -> V (detach c)) scaps in
  let forms =
    Array.map
      (fun v ->
        let (F f) = form v in
        let shape = Nx_array.Layout.shape f.layout in
        F { f with layout = Nx_array.Layout.contiguous shape })
      (values x y_like)
  in
  let nx = Array.length forms in
  let point vs = rebuild x ~like:y_like vs in
  (* The residual, whose values from outside are what the root depends on. *)
  let residual_p, theta =
    Trace.trace Body ~name forms (fun vs ->
        (* The residual and the linear solve compute derivatives alone, which
           offer nothing. *)
        Program.silenced @@ fun () ->
        let r = residual (point vs) in
        same_leaves name x ~this:"in the residual's result" r
          ~that:"in the solution" (point vs);
        values x r)
  in
  if Program.reads_lanes ~index:false residual_p then
    invalid_argf
      "%s: the residual reads the lanes of a map, which joins the lanes' \
       systems"
      name;
  let nt = Array.length theta in
  let data_forms = Array.append forms (Array.map form theta) in
  let operand_forms = Array.append forms data_forms in
  let split vs =
    (Array.sub vs 0 nx, Array.sub vs nx nx, Array.sub vs (2 * nx) nt)
  in
  let closed (p, caps) =
    if caps <> [||] then invalid_argf "%s: a root's program captures" name;
    p
  in
  (* [J v], the residual's derivative along the solution, and [Jᵀ w]. *)
  let tangents i out =
    Array.init nx (fun k ->
        let (V y) = out.(k) in
        match Jvp.tangent_of i y with
        | Some t -> V t
        | None -> V (Nx.zeros_like (Jvp.primal i y)))
  in
  let jacobian =
    closed
      (* A root's derivative offers nothing. *)
      (Trace.trace Body ~name operand_forms (fun vs ->
           Program.silenced @@ fun () ->
           let v, xs, th = split vs in
           Jvp.run ~name (fun i ->
               let xd =
                 Array.map2
                   (fun (V x) v -> V (Jvp.seed i x (Program.like x v)))
                   xs v
               in
               tangents i (Program.eval residual_p (Array.append xd th)))))
  in
  let transpose =
    closed
      (Trace.trace Body ~name operand_forms (fun vs ->
           Program.silenced @@ fun () ->
           let ws, xs, th = split vs in
           let l =
             Linear.trace ~name
               (Array.map (Loops.filled ~zero:true) forms)
               (fun v -> Program.eval jacobian (Array.concat [ v; xs; th ]))
           in
           let cts = Linear.transpose l (Array.map Option.some ws) in
           Array.mapi
             (fun k c ->
               match c with
               | Some (c : Linear.ct) -> c.v
               | None -> Loops.filled ~zero:true forms.(k))
             cts))
  in
  (* The linear solve, traced once against an abstract operator. *)
  let id = Atomic.fetch_and_add operators 1 in
  let solve_p, lcaps =
    Trace.trace Body ~name operand_forms (fun vs ->
        Program.silenced @@ fun () ->
        let bs = Array.sub vs 0 nx and data = Array.sub vs nx (nx + nt) in
        let op args =
          Program.perform
            (Operator { id; transposed = false; lanes = []; args; data; forms })
        in
        match linear_solve with
        | Some solve ->
            values x (solve (fun v -> point (op (values x v))) (point bs))
        | None -> dense name forms op bs)
  in
  (* Each operator traced once per maps it lies under, as each evaluation of a
     solve, a backward loop's per row, fills it again; a compiled program
     holding the solve may run on several domains. *)
  let filled = Hashtbl.create 4 and lock = Mutex.create () in
  let operator ~transposed ~lanes =
    Mutex.protect lock (fun () ->
        let key = (transposed, lanes) in
        match Hashtbl.find_opt filled key with
        | Some p -> p
        | None ->
            let p =
              mapped name
                ((if transposed then transpose else jacobian), [||])
                ~nargs:nx operand_forms lanes
            in
            Hashtbl.add filled key p;
            p)
  in
  let system =
    {
      Program.id;
      operator;
      residual = residual_p;
      jacobian;
      transpose;
      solve = solve_p;
    }
  in
  let solution =
    Array.map2
      (fun (F f) v -> V (Program.cast_at f v))
      forms
      (Program.threaded sprog (fun totals -> Program.eval ~totals sprog scaps))
  in
  let ys = run (Root { name; system; solution; theta; captures = lcaps }) in
  rebuild x ~like:y0 ys

let root ?linear_solve x ~residual solve =
  root_named "Rune.root" ?linear_solve x ~residual solve

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

(* [traced name a r f args forms] is [f]'s program over stand-ins of [forms] for
   [args]' leaves, and the result it returned, as jit traces it. *)
let traced name a r f args forms totals =
  let like = ref None and asked = ref [] in
  let body, _ =
    Trace.trace ~totals ~asked Jit ~name forms (fun vs ->
        let y = f (rebuild a ~like:args vs) in
        like := Some y;
        values r y)
  in
  (body, Option.get !like, !asked)

(* A result's structure, rebuilt from a call's outputs, its leaves read once
   at the trace. *)
type 'y results = { rebuild : value array -> 'y }

let results r like =
  let leaves = fst (Ptree.flatten r like) in
  let rebuild vs =
    Ptree.rebuild r ~like
      (List.mapi (fun k (Nx.P l) -> Nx.P (Program.like l vs.(k))) leaves)
  in
  { rebuild }

(* A call's program comes from its key's entry: traced over stand-ins of its
   leaves the first time the key is met, then compiled. The call is a construct,
   so a transformation around it evaluates the program over its values; with
   none, its compiled form runs. *)
let call name t a r f args =
  let leaves, skeleton = Ptree.flatten a args in
  let args_v = Array.of_list (List.map (fun (Nx.P y) -> V y) leaves) in
  let paths () =
    List.rev (Ptree.fold a (fun p _ acc -> path_name p :: acc) args [])
  in
  let trace forms totals =
    let body, like, asked = traced name a r f args forms totals in
    (body, results r like, asked)
  in
  let e = Jit.find t leaves skeleton ~paths trace in
  (* A program takes a total's running value where its trace found it; a call
     that finds it elsewhere has a program of its own, traced there. *)
  let moved = ref false in
  let totals =
    Array.map
      (fun (Addition (u, into, out) as a) ->
        match Program.running u into with
        | Some v when Option.is_none (Program.coerce_at into v) ->
            moved := true;
            let at = Nx.Prim.form v in
            Addition (u, { into with placement = at.placement }, out)
        | _ -> a)
      e.body.adds
  in
  let e =
    if !moved then Jit.find ~totals t leaves skeleton ~paths trace else e
  in
  let out =
    run (Call { body = e.body; args = args_v; site = e.site; seeds = [||] })
  in
  let y = e.like.rebuild out in y

let jit_named ?beam ?parallel name s =
  let (Spine u) = spine name 0 s in
  let a = Ptree.v u.walk.walk in
  fun f ->
    let t = Jit.create ~name ?beam ?parallel () in
    u.curry (fun args -> call name t a u.result (u.apply f) args)

let jit ?beam ?parallel s = jit_named ?beam ?parallel "Rune.jit" s

(* Rematerialization *)

let remat_named name s =
  let (Spine u) = spine name 0 s in
  let a = Ptree.v u.walk.walk in
  fun f ->
    u.curry (fun args ->
        let args_v = values a args in
        let like = ref None in
        let body, captures =
          Trace.trace Staged ~name (Array.map form args_v) (fun vs ->
              let y = u.apply f (rebuild a ~like:args vs) in
              like := Some y;
              values u.result y)
        in
        let y = Option.get !like in
        let out =
          run
            (Remat
               {
                 body;
                 args = Array.append args_v captures;
                 seeds = [||];
                 differentiated = false;
               })
        in
        rebuild u.result ~like:y
          (slice out 0 (Array.length (values u.result y))))

let remat s = remat_named "Rune.remat" s

(* Maps *)

(* The common length of the arrays' axes 0. *)
let lanes_of name s args =
  let lengths =
    List.rev
      (Ptree.fold s
         (fun p v acc ->
           match Nx.shape v with
           | [||] -> invalid_argf "%s: %s is 0-d" name (path_name p)
           | s -> (path_name p, s.(0)) :: acc)
         args [])
  in
  match lengths with
  | [] -> invalid_argf "%s: the arguments hold no array" name
  | (first, n) :: _ ->
      List.iter
        (fun (p, m) ->
          if m <> n then
            invalid_argf "%s: %s: %d rows along axis 0, %s: %d" name p m first n)
        lengths;
      n

let vmap_named name ?axis s =
  let (Spine u) = spine name 0 s in
  let a = Ptree.v u.walk.walk in
  let axis = match axis with Some a -> a | None -> Program.axis () in
  fun f ->
    u.curry (fun args ->
        let n = lanes_of name a args in
        Vmap.run ~name ~axis n (fun i ->
            let y = u.apply f (Ptree.map a (fun _ x -> Vmap.lane i x) args) in
            Ptree.map u.result (fun _ y -> Vmap.batch i n y) y))

let vmap ?axis s = vmap_named "Rune.vmap" ?axis s
(* Inspecting a trace *)

type program = Program.t

(* [f]'s trace at [args], as jit's: over C-order stand-ins. *)
let staged name a r f args =
  let leaves, _ = Ptree.flatten a args in
  let body, like, _ = traced name a r f args (Jit.stand_ins leaves) [||] in
  (body, like)

let program a r f args = fst (staged "Rune.program" a r f args)
let pp_program = Program.pp
let eval_shape a r f args = snd (staged "Rune.eval_shape" a r f args)

(* Functions of one array *)

let scan' ~f ~init xs =
  scan_named "Rune.scan'" Ptree.tensor Ptree.tensor Ptree.tensor ~f ~init xs

let iterate' ~max ~until ~f x =
  iterate_named "Rune.iterate'" Ptree.tensor ~max ~until ~f x

let jit' ?beam ?parallel f =
  jit_named ?beam ?parallel "Rune.jit'" Ptree.(tensor @-> returns tensor) f
let jvp' f x t = jvp_named "Rune.jvp'" Ptree.tensor Ptree.tensor f x t

let vmap' ?axis f x =
  vmap_named "Rune.vmap'" ?axis Ptree.(tensor @-> returns tensor) f x

(* The Jacobian's columns as the lanes of one map of [jvp] over the basis of
   [x]'s elements. *)
let jacfwd' f x =
  let name = "Rune.jacfwd'" in
  let sx = Nx.shape x and n = Nx.numel x in
  let here = Nx.beside x in
  let iota = Nx.arange_at here Nx.int32 0 n 1 in
  let e = Nx.equal (Nx.reshape [| n; 1 |] iota) (Nx.reshape [| 1; n |] iota) in
  let basis = Nx.reshape (Array.append [| n |] sx) (Nx.cast (Nx.dtype x) e) in
  let column t = snd (jvp_named name Ptree.tensor Ptree.tensor f x t) in
  let ts = vmap_named name Ptree.(tensor @-> returns tensor) column basis in
  let r = Nx.ndim ts - 1 in
  let sy = Array.sub (Nx.shape ts) 1 r in
  Nx.reshape (Array.append sy sx) (Nx.moveaxis 0 r ts)

let grad' f x = snd (value_and_grad_named "Rune.grad'" Ptree.tensor f x)

let value_and_grad' f x =
  value_and_grad_named "Rune.value_and_grad'" Ptree.tensor f x

let vjp' f x = vjp_named "Rune.vjp'" Ptree.tensor Ptree.tensor f x

(* The Jacobian's rows as the lanes of one map of the pullback over the basis of
   the result's elements. *)
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
  let here = Nx.beside y in
  let iota = Nx.arange_at here Nx.int32 0 n 1 in
  let e = Nx.equal (Nx.reshape [| n; 1 |] iota) (Nx.reshape [| 1; n |] iota) in
  let basis = Nx.reshape (Array.append [| n |] sy) (Nx.cast (Nx.dtype y) e) in
  let rows = vmap_named name Ptree.(tensor @-> returns tensor) pullback basis in
  Nx.reshape (Array.append sy (Nx.shape x)) rows

(* The lowering, for its suites *)

module Lower = Lower
module Lower_arith = Lower_arith

let derivations = Program.derivations
let evaluations = Compile.evaluations
let variants = Compile.variants
