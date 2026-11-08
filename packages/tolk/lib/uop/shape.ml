(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops

(* Lists *)

(* The first [n] elements of a list, as Python's slice: a short list gives
   what it has. *)
let rec take n l =
  if n <= 0 then [] else match l with [] -> [] | x :: r -> x :: take (n - 1) r

module Value = Dtype.Value

let min_max u = (vmin u, vmax u)

(* Simplification *)

(* The rules of [simplify], [symbolic] below: they read shapes, which read
   [simplify]. The end of this module sets them. *)
let simplify_rules = ref (Pattern_matcher.v (fun () -> []))

(* The rules leave a graph of constants as it is: a constant, and a sink of
   constants and of stacks of constants, are themselves without a rewrite. *)
let simplify u =
  let constant s =
    op s = Op.Const
    || (op s = Op.Stack && List.for_all (fun c -> op c = Op.Const) (src s))
  in
  if op u = Op.Const then u
  else if op u = Op.Sink && List.for_all constant (src u) then u
  else
    graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() u
      (After_sources !simplify_rules)

let resolve ?(default = true) u =
  if not (Dtype.equal (dtype u) Dtype.Bool) then
    invalid_argf "only a boolean resolves, not a %s" (repr_dtype (dtype u));
  let lo, hi = min_max (simplify u) in
  if Value.( = ) lo hi then Value.to_bool lo else default

let sint_of_const (c : Dtype.const) =
  match c with
  | `Int z when Bigint.fits_int z -> Some (Int (Bigint.to_int z) : sint)
  | `Bool b -> Some (Int (Bool.to_int b))
  | _ -> None

let ssimplify u : sint =
  let r = simplify u in
  let known =
    match (op r, src r) with
    | Op.Cast, [ c ] when op c = Op.Const ->
        sint_of_const (Dtype.const (dtype r) (value c))
    | Op.Const, _ -> sint_of_const (value r)
    | _ -> None
  in
  match known with Some s -> s | None -> Sym r

let ssimplify_sint = function Int n -> Int n | Sym u -> ssimplify u

let eval u ~kinds ~what =
  if not (Dtype.mem (dtype u) kinds) then
    invalid_argf "a %s is not %s" (repr_dtype (dtype u)) what;
  let s = simplify u in
  let lo, hi = min_max s in
  if not (Value.( = ) lo hi) then
    invalid_argf "the value ranges from %s to %s" (repr_const lo)
      (repr_const hi);
  lo

let to_bool u =
  match eval u ~kinds:[ Dtype.Bool ] ~what:"a boolean" with
  | `Bool b -> b
  | v -> invalid_argf "%s is not a boolean" (repr_const v)

let to_z u =
  match eval u ~kinds:(Dtype.Weak_int :: Dtype.ints) ~what:"an integer" with
  | (`Int _ | `Bool _) as v -> Value.to_z v
  | v -> invalid_argf "%s is not an integer" (repr_const v)

let to_float u =
  match eval u ~kinds:(Dtype.Weak_float :: Dtype.floats) ~what:"a float" with
  | `Float x -> x
  | v -> invalid_argf "%s is not a float" (repr_const v)

(* Symbolic integers *)

module Sint = struct
  type node = t
  type t = sint

  let node : t -> node = function Int n -> int n | Sym u -> u

  (* Integers compute exactly, as Python's do, and a result past [int] raises
     rather than wraps. *)
  let of_z z =
    if Bigint.fits_int z then Int (Bigint.to_int z)
    else invalid_argf "%s is larger than an int" (Bigint.to_string z)

  let arith fz fu a b =
    match (a, b) with
    | Int x, Int y -> of_z (fz (Bigint.of_int x) (Bigint.of_int y))
    | _ -> Sym (fu (node a) (node b))

  let ( + ) = arith Bigint.add add
  let ( - ) = arith Bigint.sub sub
  let ( * ) = arith Bigint.mul mul

  let ( // ) =
    arith
      (fun x y -> Value.to_z Value.(`Int x // `Int y))
      (div ~rounding:`Floor)

  let ( % ) = arith (fun x y -> Value.to_z Value.(`Int x % `Int y)) mod_
  let neg = function Int n -> Int 0 - Int n | Sym u -> Sym (neg u)

  (* A product of integers is taken whole, so that a zero makes it zero past
     partial products that do not fit. *)
  let prod l =
    let ints = List.filter_map (function Int n -> Some n | Sym _ -> None) l in
    if List.length ints = List.length l then
      of_z (List.fold_left (fun z n -> Bigint.mul z (Bigint.of_int n)) Bigint.one ints)
    else List.fold_left ( * ) (Int 1) l

  type cond = Known of bool | Cond of node

  let compare fi fu a b =
    match (a, b) with
    | Int x, Int y -> Known (fi x y)
    | _ -> Cond (fu (node a) (node b))

  let ( < ) = compare Stdlib.( < ) lt
  let ( <= ) = compare Stdlib.( <= ) le
  let ( > ) = compare Stdlib.( > ) gt
  let ( >= ) = compare Stdlib.( >= ) ge
  let ( <> ) = compare Stdlib.( <> ) ne
  let resolve ?default = function Known b -> b | Cond u -> resolve ?default u

  (* The truth of a condition that must be decidable. *)
  let truth = function Known b -> b | Cond u -> to_bool u
  let equal = equal_sint
  let pp ppf s = Format.pp_print_string ppf (repr_sint s)
end

(* The symbolic operands, then the extreme of the integer ones. *)
let smax_smin ~name ~combine ~pick ss =
  let syms = List.filter_map (function Sym u -> Some u | Int _ -> None) ss
  and ints = List.filter_map (function Int n -> Some n | Sym _ -> None) ss in
  let extreme = function
    | [] -> []
    | n :: rest -> [ List.fold_left pick n rest ]
  in
  match (syms, extreme ints) with
  | [], [] -> invalid_argf "%s of nothing" name
  | [], n -> Int (List.hd n)
  | u :: rest, n -> ssimplify (List.fold_left combine u (rest @ List.map int n))

let smax ss = smax_smin ~name:"smax" ~combine:maximum ~pick:Stdlib.max ss
let smin ss = smax_smin ~name:"smin" ~combine:minimum ~pick:Stdlib.min ss

(* The number of elements of a shape of known sizes. *)
let size_of shape =
  if List.mem 0 shape then 0
  else
    List.fold_left
      (fun acc n ->
        if n <> 0 && abs acc > max_int / abs n then
          invalid_argf "a shape of %s elements is larger than an int"
            (Bigint.to_string
               (List.fold_left (fun z n -> Bigint.mul z (Bigint.of_int n)) Bigint.one shape))
        else acc * n)
      1 shape

(* Shapes *)

let as_shape u : sint list =
  let known s =
    match sint_of_const (value s) with
    | Some n -> n
    | None -> invalid_argf "%s is not a size" (repr_const (value s))
  in
  match op u with
  | Op.Const -> [ known u ]
  | Op.Stack ->
      List.map
        (fun s -> if op s = Op.Const then known s else ssimplify s)
        (src u)
  | _ -> [ ssimplify u ]

let compute_marg u =
  let shape_src i = as_shape (nth u i) in
  match (op u, arg u) with
  | Op.Reshape, _ -> Reshape (shape_src 1)
  | Op.Expand, _ -> Expand (shape_src 1)
  | Op.Pad, _ -> Pad (List.combine (shape_src 1) (shape_src 2))
  | Op.Shrink, _ -> Shrink (List.combine (shape_src 1) (shape_src 2))
  | Op.Permute, _ ->
      let axis = function
        | Int n -> n
        | Sym _ -> invalid_arg "a permutation of a symbolic axis"
      in
      Permute (List.map axis (shape_src 1))
  | op, _ -> invalid_argf "%s is not a movement" (Op.name op)

let marg u = memoized_movement ~compute:compute_marg u

let marg_shape u =
  match marg u with
  | Reshape s | Expand s -> s
  | _ -> invalid_argf "%s has no shape argument" (Op.name (op u))

let broadcast_axes src out =
  let nleft = List.length out - List.length src in
  if nleft < 0 then
    invalid_argf "cannot broadcast %s into %s" (repr_shape src) (repr_shape out);
  List.init nleft Fun.id
  @ List.filter_map Fun.id
      (List.mapi
         (fun i s ->
           let one = match s with Int 1 -> true | _ -> false in
           let o = List.nth out (nleft + i) in
           if one && Sint.resolve Sint.(o <> Int 1) then Some (nleft + i)
           else None)
         src)

let rec shape_opt u =
  memoized_shape ~compute:compute_shape u

and shape u =
  match shape_opt u with
  | Some s -> s
  | None -> invalid_argf "%s has no shape" (Op.name (op u))

and compute_shape u : sint list option =
  let src0 () = first (op u) (src u) in
  let void = Dtype.equal (dtype u) Dtype.Void in
  match op u with
  | Op.If | Op.Barrier | Op.Sink | Op.Endif | Op.Backedge | Op.Linear
  | Op.Program | Op.Source ->
      None
  | Op.Custom_function -> if void then None else Some (as_shape (src0 ()))
  | Op.Ins -> if void then None else Some []
  | Op.Reshape when (op (src0 ())) = Op.Noop -> Some (marg_shape u)
  | Op.Noop -> ( match src u with s :: _ -> shape_opt s | [] -> None)
  | Op.Index ->
      let buf = src0 () and idxs = drop 1 (src u) in
      Some (List.concat_map shape idxs @ drop (List.length idxs) (shape buf))
  | Op.Stack -> (
      match src u with
      | [] -> Some []
      | s :: _ -> Some (Int (List.length (src u)) :: shape s))
  | Op.Const | Op.Getaddr | Op.Range | Op.Special -> Some []
  | Op.Binary -> (
      match arg u with
      | Bytes b -> Some [ Int (String.length b) ]
      | _ -> invalid_arg "binary needs bytes")
  | Op.Buffer | Op.Alloc | Op.Param | Op.Refparam -> (
      match as_shape (first (op u) (src u)) with
      | ([] | [ _ ]) as s -> Some s
      | _ -> invalid_arg "storage has one axis")
  | Op.Stage ->
      let rs = drop 1 (src u) in
      Some
        (List.map
           (fun r -> Int (Value.to_int (Value.( + ) (vmax r) (`Int Bigint.one))))
           rs
        @ shape (src0 ()))
  | Op.Wmma -> (
      match src u with
      | [ a; b; acc ] ->
          let init s = take (List.length s - 1) s in
          let last s = List.nth s (List.length s - 1) in
          Some
            (broadcast_shape
               [ init (shape a); init (shape b); init (shape acc) ]
            @ [ last (shape acc) ])
      | _ -> invalid_arg "wmma needs three sources")
  | Op.Detach | Op.Contiguous_backward | Op.After | Op.Load | Op.Copy
  | Op.Store | Op.End | Op.Call ->
      shape_opt (src0 ())
  | Op.Bitcast -> (
      match shape_opt (src0 ()) with
      | None -> None
      | Some [] -> Some []
      | Some ps ->
          let out_sz = Dtype.itemsize (dtype u)
          and in_sz = Dtype.itemsize (dtype (src0 ())) in
          if out_sz = in_sz then Some ps
          else
            let n = List.length ps in
            let last = List.nth ps (n - 1) in
            (match last with
            | Int l when l * in_sz mod out_sz <> 0 ->
                invalid_argf "a bitcast cannot resize an axis of %d" l
            | _ -> ());
            Some
              (take (n - 1) ps
              @ [ ssimplify_sint Sint.(last * Int in_sz // Int out_sz) ]))
  | op when Op.Set.mem op Op.Set.movement || op = Op.Reduce ->
      let ps =
        match shape_opt (src0 ()) with
        | Some ps -> ps
        | None ->
            invalid_argf "%s needs a shape, and %s has none" (Op.name op)
              (Op.name (Ops.op (src0 ())))
      in
      Some (movement_shape u ps)
  | op when Op.Set.mem op Op.Set.unary || op = Op.Cast ->
      (match src u with
      | [ _ ] -> ()
      | _ -> invalid_argf "%s needs one source" (Op.name op));
      shape_opt (src0 ())
  | op when Op.Set.mem op Op.Set.broadcastable ->
      let shapes =
        List.map
          (fun s ->
            match shape_opt s with
            | Some sh -> sh
            | None -> invalid_argf "%s of a node without a shape" (Op.name op))
          (src u)
      in
      if List.is_empty shapes then invalid_argf "%s needs sources" (Op.name op);
      if
        Setting.value Setting.disallow_broadcast
        && not (Helpers.all_same equal_shape shapes)
      then
        invalid_argf "%s of shapes %s" (Op.name op)
          (String.concat ", " (List.map repr_shape shapes));
      Some (broadcast_shape shapes)
  | op -> invalid_argf "%s has no shape rule" (Op.name op)

and movement_shape u ps =
  let bad what =
    invalid_argf "invalid %s %s for %s" (Op.name (op u)) what (repr_shape ps)
  in
  let ok = Sint.resolve ?default:None in
  (* A size is a number: one that holds Invalid, such as the size of a shrink
     whose bounds carry a validity, evaluates to none. *)
  let numbers s =
    List.iter
      (function
        | Sym d when List.exists is_invalid (toposort ~calls:Skip d) ->
            invalid_argf "%s of sizes %s, one holding Invalid, which is no number"
              (Op.name (op u)) (repr_shape s)
        | _ -> ())
      s;
    s
  in
  match op u with
  | Op.Reduce -> (
      match arg u with
      | Reduce { num_axes; _ } when num_axes >= 0 && num_axes <= List.length ps
        ->
          drop num_axes ps
      | _ -> invalid_argf "invalid reduction axes for %s" (repr_shape ps))
  | _ -> (
      match marg u with
      | Reshape s ->
          if not (List.for_all (fun x -> Sint.truth Sint.(x >= Int 0)) s) then
            invalid_argf "a shape cannot hold negative sizes: %s" (repr_shape s);
          if Sint.resolve ~default:false Sint.(prod ps <> prod s) then
            invalid_argf "cannot reshape %s to %s" (repr_shape ps)
              (repr_shape s);
          numbers s
      | Expand s -> numbers s @ ps
      | Permute order ->
          if List.sort Int.compare order <> List.init (List.length ps) Fun.id
          then bad (repr_tuple (List.map string_of_int order));
          List.map (List.nth ps) order
      | Pad bounds ->
          if
            List.length ps <> List.length bounds
            || not
                 (List.for_all2
                    (fun s (o, sz) ->
                      ok Sint.(sz >= Int 0)
                      && ok Sint.(o >= Int 0)
                      && ok Sint.(o + s <= sz))
                    ps bounds)
          then bad "padding";
          numbers (List.map snd bounds)
      | Shrink bounds ->
          if
            List.length ps <> List.length bounds
            || not
                 (List.for_all2
                    (fun s (o, sz) ->
                      ok Sint.(o >= Int 0)
                      && ok Sint.(sz >= Int 0)
                      && ok Sint.(o + sz <= s))
                    ps bounds)
          then bad "bounds";
          numbers (List.map snd bounds))

let ndim u = List.length (shape u)
let numel u = Sint.prod (shape u)
let max_shape u = to_max_shape (shape u)
let max_numel u = size_of (max_shape u)

(* Movement *)

let shape_to_shape_arg (arg : sint list) =
  let src = List.map (function Int n -> int n | Sym u -> u) arg in
  List.iter
    (fun x ->
      if not (Dtype.is_int (dtype x)) then
        invalid_argf "a shape holds integers, not %s" (repr_dtype (dtype x)))
    src;
  match src with [ x ] -> x | src -> v Op.Stack ~src

let mop u (m : movement) =
  let simplified args =
    (src (simplify (sink (List.map shape_to_shape_arg args))))
  in
  let scalar_noop () =
    if not (List.is_empty (shape u)) then
      invalid_arg "an empty pad or shrink needs a scalar";
    u
  in
  match m with
  | Expand [] -> u
  | Pad [] | Shrink [] -> scalar_noop ()
  | Reshape s -> v Op.Reshape ~src:(u :: simplified [ s ])
  | Expand s -> v Op.Expand ~src:(u :: simplified [ s ])
  | Pad b -> v Op.Pad ~src:(u :: simplified [ List.map fst b; List.map snd b ])
  | Shrink b ->
      v Op.Shrink ~src:(u :: simplified [ List.map fst b; List.map snd b ])
  | Permute l ->
      v Op.Permute ~src:(u :: simplified [ List.map (fun i -> Int i) l ])

let resolve_dim ?(extra = 0) u dim =
  let total = ndim u + extra in
  let bound = max 1 total in
  if dim < -bound || dim > bound - 1 then
    invalid_argf "axis %d is out of range [%d, %d]" dim (-bound) (bound - 1);
  if dim < 0 then dim + total else dim

(* A movement that leaves the shape unchanged is no movement. *)
let unless_same u ret = if equal_shape (shape ret) (shape u) then u else ret

let reshape u new_shape =
  let old = shape u in
  let inferred = List.length (List.filter (Sint.equal (Int (-1))) new_shape) in
  if inferred > 1 then
    invalid_argf "only one size can be inferred, in %s" (repr_shape new_shape);
  let new_shape =
    if inferred = 0 then new_shape
    else
      let known = Sint.prod new_shape in
      List.map
        (fun s ->
          if Sint.equal s (Int (-1)) then Sint.(neg (prod old) // known) else s)
        new_shape
  in
  if Sint.truth Sint.(prod old <> prod new_shape) then
    invalid_argf "cannot reshape %s to %s" (repr_shape old)
      (repr_shape new_shape);
  unless_same u (mop u (Reshape new_shape))

let permute u order =
  let order = List.map (resolve_dim u) order in
  let n = ndim u in
  if List.sort Int.compare order <> List.init n Fun.id then
    invalid_argf "%s is not a permutation"
      (repr_tuple (List.map string_of_int order));
  if order = List.init n Fun.id then u else mop u (Permute order)

let broadcast_to u new_shape =
  let old = shape u in
  if equal_shape old new_shape then u
  else begin
    if List.length old > List.length new_shape then
      invalid_argf "cannot broadcast %s to fewer axes, %s" (repr_shape old)
        (repr_shape new_shape);
    let aligned = List.hd (align_left [ old; new_shape ]) in
    if
      not
        (List.for_all2
           (fun s ns -> equal_sint s ns || equal_sint s (Int 1))
           aligned new_shape)
    then
      invalid_argf "cannot broadcast %s to %s" (repr_shape old)
        (repr_shape new_shape);
    let n_left = List.length new_shape - List.length old in
    let expand_at =
      List.filter_map
        (fun i -> if i >= n_left then Some (i - n_left) else None)
        (broadcast_axes old new_shape)
    in
    let kept =
      List.filter
        (fun i -> not (List.mem i expand_at))
        (List.init (List.length old) Fun.id)
    in
    let squeezed = reshape u (List.map (List.nth old) kept) in
    let expanded =
      mop squeezed
        (Expand
           (take n_left new_shape
           @ List.map (fun i -> List.nth new_shape (n_left + i)) expand_at))
    in
    let index_of x l = Option.get (List.find_index (Int.equal x) l) in
    permute expanded
      (List.init n_left Fun.id
      @ List.init (List.length old) (fun i ->
          n_left
          +
          if List.mem i expand_at then index_of i expand_at
          else List.length expand_at + index_of i kept))
  end

let expand u new_shape =
  let aligned = align_left [ shape u; new_shape ] in
  broadcast_to u
    (List.map2
       (fun from to_ -> if Sint.equal to_ (Int (-1)) then from else to_)
       (List.nth aligned 0) (List.nth aligned 1))

let check_rank u what l =
  if ndim u <> List.length l then
    invalid_argf "%s of %d axes for a node of %d" what (List.length l) (ndim u)

(* A shrink to the same shape is its source unless a start is symbolic: an
   offset of 0 that waits on effects, as one ordered after a store, would lose
   them. *)
let shrink u bounds =
  check_rank u "bounds" bounds;
  let concrete = function Some (Sym _, _) -> false | _ -> true in
  (if List.for_all concrete bounds then unless_same u else Fun.id)
    (mop u
       (Shrink
          (List.map2
             (fun b s ->
               match b with
               | Some (lo, hi) -> (lo, Sint.(hi - lo))
               | None -> (Int 0, s))
             bounds (shape u))))

let shrink_to u new_shape =
  shrink u (List.map (Option.map (fun ns -> (Int 0, ns))) new_shape)

let movement_pad u pads =
  check_rank u "padding" pads;
  unless_same u
    (mop u
       (Pad
          (List.map2
             (fun (before, after) s -> (before, Sint.(s + before + after)))
             pads (shape u))))

(* Whether a fill's bits are all zero, which the padding's own zeros already
   are: a [-0.] fill is not, as it differs from [+0.] in its sign bit. *)
let is_zero (c : Dtype.const) =
  match c with
  | `Invalid -> false
  | `Bool b -> not b
  | `Int n -> Bigint.equal n Bigint.zero
  | `Float x -> Int64.equal (Int64.bits_of_float x) 0L

let const_like ?dtype u c =
  let ret = const ?dtype:(Some (Option.value dtype ~default:(Ops.dtype u))) c in
  match shape_opt u with
  | Some (_ :: _ as s) when not (equal_shape (shape ret) s) ->
      mop ret (Expand s)
  | _ -> ret

let pad ?(value = `Int Bigint.zero) u padding =
  let pads = List.map (Option.value ~default:(Int 0, Int 0)) padding in
  check_rank u "padding" pads;
  let has_neg =
    not
      (List.for_all
         (fun p -> Sint.resolve Sint.(p >= Int 0))
         (List.concat_map (fun (b, a) -> [ b; a ]) pads))
  in
  let x, pads =
    if not has_neg then (u, pads)
    else
      ( shrink u
          (List.map2
             (fun (b, a) s ->
               Some (Sint.neg (smin [ b; Int 0 ]), smin [ Sint.(a + s); s ]))
             pads (shape u)),
        List.map (fun (b, a) -> (smax [ b; Int 0 ], smax [ a; Int 0 ])) pads )
  in
  let padded = movement_pad x pads in
  if is_zero value then padded
  else
    where
      (movement_pad (const_like ~dtype:Dtype.Bool x (`Bool true)) pads)
      padded (const value)

let pad_to ?(value = `Int Bigint.zero) u new_shape =
  let to_pad x =
    if List.length new_shape <> ndim x then
      invalid_argf "%d sizes for a node of %d axes" (List.length new_shape)
        (ndim x);
    unless_same x
      (mop x
         (Pad
            (List.map2
               (fun s ns -> (Int 0, Option.value ns ~default:s))
               (shape x) new_shape)))
  in
  let ret = to_pad u in
  if is_zero value || ret == u then ret
  else
    where
      (to_pad (const_like ~dtype:Dtype.Bool u (`Bool true)))
      ret (const value)

let flatten ?(start = 0) ?(stop = -1) u =
  let start = resolve_dim u start and stop = resolve_dim u stop in
  let s = shape u in
  reshape u
    (take start s
    @ [ Sint.prod (take (stop - start + 1) (drop start s)) ]
    @ drop (stop + 1) s)

let unflatten u axis sizes =
  let axis = resolve_dim u axis in
  let s = shape u in
  reshape u (take axis s @ sizes @ drop (axis + 1) s)

let squeeze ?axis u =
  match axis with
  | None ->
      reshape u (List.filter (fun s -> Sint.truth Sint.(s <> Int 1)) (shape u))
  | Some axis ->
      let axis = resolve_dim u axis in
      if ndim u = 0 || Sint.truth Sint.(List.nth (shape u) axis <> Int 1) then u
      else reshape u (List.filteri (fun i _ -> i <> axis) (shape u))

let unsqueeze u axis =
  let axis = resolve_dim ~extra:1 u axis in
  reshape u (take axis (shape u) @ (Int 1 :: drop axis (shape u)))

let transpose u a b =
  let a = resolve_dim u a and b = resolve_dim u b in
  permute u
    (List.init (ndim u) (fun i -> if i = a then b else if i = b then a else i))

(* A flip of an axis of size n is made of the other movements. The axis is
   moved to the front; its n values are padded and reshaped so that value i
   lies at row i, column n - 1 - i of an m x m square; the square is transposed
   and its anti-diagonal read back, which puts value i at n - 1 - i. Index
   arithmetic folds the result to that index. An axis whose size may be below
   2 is taken as m = n + 2 and cropped back, and so is an axis of the rest
   whose size may be 0, so that every reshape keeps its elements. *)
let flip u axes =
  let axes = List.map (resolve_dim u) axes in
  if ndim u = 0 && axes <> [] then invalid_arg "a scalar has no axis to flip";
  if List.length (List.sort_uniq Int.compare axes) <> List.length axes then
    invalid_argf "an axis appears twice in %s"
      (repr_tuple (List.map string_of_int axes));
  let known c = Sint.resolve ~default:false c in
  let flip1 ret axis =
    let n = List.nth (shape ret) axis in
    if known Sint.(n <= Int 1) then ret
    else
      let x = transpose ret 0 axis in
      let rest = List.tl (shape x) in
      let whole = List.map (fun _ -> None) rest in
      let tail =
        List.map
          (fun s -> if known Sint.(s > Int 0) then s else Sint.(s + Int 1))
          rest
      in
      let m = if known Sint.(n >= Int 2) then n else Sint.(n + Int 2) in
      let tail_pad = List.map (fun s -> (Int 0, s)) tail in
      let m1 = Sint.(m - Int 1) in
      let x =
        reshape (mop x (Pad ((Int 0, m) :: tail_pad))) (m :: Int 1 :: tail)
      in
      let x =
        reshape
          (mop x (Pad ((Int 0, m) :: (Int 0, m1) :: tail_pad)))
          (Sint.(m * m1) :: tail)
      in
      let x =
        reshape (mop x (Pad ((m1, Sint.(m * m)) :: tail_pad))) (m :: m :: tail)
      in
      let x = reshape (transpose x 0 1) (Sint.(m * m) :: tail) in
      let x =
        reshape
          (mop x (Shrink ((m1, Sint.(m * m1)) :: tail_pad)))
          (m :: m1 :: tail)
      in
      let x =
        reshape (shrink x (None :: Some (Int 0, Int 1) :: whole)) (m :: tail)
      in
      let crop = List.map (fun s -> Some (Int 0, s)) rest in
      transpose (shrink x (Some (Sint.(m - n), m) :: crop)) 0 axis
  in
  List.fold_left flip1 u axes

let split ?(axis = 0) u sizes =
  let axis = resolve_dim u axis in
  let n =
    match List.nth (shape u) axis with
    | Int n -> n
    | Sym _ -> invalid_arg "a split along an axis of symbolic size"
  in
  let total = List.fold_left ( + ) 0 sizes in
  if total <> n then
    invalid_argf "sizes that sum to %d split an axis of %d elements" total n;
  let cut (lo, pieces) k =
    let bounds =
      List.init (ndim u) (fun i ->
          if i = axis then Some (Int lo, Int (lo + k)) else None)
    in
    (lo + k, shrink u bounds :: pieces)
  in
  List.rev (snd (List.fold_left cut (0, []) sizes))

let repeat u repeats =
  if List.exists (fun r -> r < 0) repeats then
    invalid_argf "repeats are not negative, in %s"
      (repr_tuple (List.map string_of_int repeats));
  let base =
    List.hd (align_left [ shape u; List.map (fun _ -> Int 1) repeats ])
  in
  let pairs = List.combine repeats base in
  let unsqueezed =
    List.concat_map (fun (r, s) -> if r = 1 then [ s ] else [ Int 1; s ]) pairs
  in
  let expanded =
    List.concat_map (fun (r, s) -> if r = 1 then [ s ] else [ Int r; s ]) pairs
  in
  reshape
    (expand (reshape u unsqueezed) expanded)
    (List.map (fun (r, s) -> Sint.(Int r * s)) pairs)

let pool ?stride ?dilation u kernel =
  let n = List.length kernel in
  let given = function Some l -> l | None -> List.init n (fun _ -> 1) in
  let stride = given stride and dilation = given dilation in
  if ndim u < n then
    invalid_argf "cannot pool %s with %d kernel axes" (repr_shape (shape u)) n;
  if List.length stride <> n || List.length dilation <> n then
    invalid_arg "one stride and one dilation per kernel axis";
  let lead = ndim u - n in
  let noop = take lead (shape u) and keep = List.init lead (fun _ -> None) in
  let axes =
    List.map2
      (fun (k, s) (d, i) -> (k, s, d, i))
      (List.combine kernel stride)
      (List.combine dilation (drop lead (shape u)))
  in
  let reach (k, _, d, _) = d * (k - 1) in
  List.iter
    (fun ((_, _, _, i) as a) ->
      let need = reach a + 1 in
      if not (Sint.resolve Sint.(Int need <= i)) then
        invalid_arg "kernel size cannot be greater than actual input size")
    axes;
  let ceildiv a b = Sint.((a + b - Int 1) // b) in
  let o =
    List.map
      (fun ((_, s, _, i) as a) -> ceildiv Sint.(i - Int (reach a)) (Int s))
      axes
  in
  (* Scales the input so that a stride can be cut from it. *)
  let f =
    List.map2
      (fun (_, s, d, i) o ->
        smax [ Int 1; ceildiv Sint.((o * Int s) - Int d) i ])
      axes o
  in
  let each g = List.map2 (fun a (o, f) -> g a o f) axes (List.combine o f) in
  let span (k, _, d, i) f = Sint.(Int k * ((i * f) + Int d)) in
  let x =
    repeat u
      (List.map (fun _ -> 1) noop
      @ each (fun ((_, _, _, i) as a) _ f ->
          match ceildiv (span a f) i with
          | Int r -> r
          | Sym _ -> invalid_arg "a symbolic pool needs a concrete repeat"))
  in
  let x = shrink_to x (keep @ each (fun a _ f -> Some (span a f))) in
  let x =
    reshape x
      (noop
      @ List.concat
          (each (fun (k, _, d, i) _ f -> [ Int k; Sint.((i * f) + Int d) ])))
  in
  let x =
    shrink_to x
      (keep
      @ List.concat
          (each (fun (k, s, _, _) o _ ->
               [ Some (Int k); Some Sint.(o * Int s) ])))
  in
  let x =
    reshape x
      (noop @ List.concat (each (fun (k, s, _, _) o _ -> [ Int k; o; Int s ])))
  in
  let x =
    shrink_to x
      (keep
      @ List.concat
          (each (fun (k, _, _, _) o _ -> [ Some (Int k); Some o; Some (Int 1) ]))
      )
  in
  let x =
    reshape x (noop @ List.concat (each (fun (k, _, _, _) o _ -> [ Int k; o ])))
  in
  permute x
    (List.init lead Fun.id
    @ List.init n (fun a -> lead + (2 * a) + 1)
    @ List.init n (fun a -> lead + (2 * a)))

let stack ?(axis = 0) us =
  match us with
  | [] -> invalid_arg "stack needs a node"
  | first :: _ ->
      let axis = resolve_dim ~extra:1 first axis in
      let s = shape first in
      if not (List.for_all (fun u -> equal_shape (shape u) s) us) then
        invalid_argf "stacked shapes differ: %s"
          (String.concat ", " (List.map (fun u -> repr_shape (shape u)) us));
      let dt = dtype_of Op.Stack us No_arg in
      let ret =
        v Op.Stack
          ~src:
            (List.map
               (fun u -> if is_invalid (base u) then u else ccast u dt)
               us)
      in
      permute ret
        (List.init axis (fun i -> i + 1)
        @ [ 0 ]
        @ List.init (ndim ret - axis - 1) (fun i -> axis + 1 + i))

let consts ?dtype cs =
  let dtype = match dtype with Some dt -> dt | None -> Dtype.of_consts cs in
  stack (List.map (const ~dtype) cs)

let valid u cond = where cond u (const_like u `Invalid)
let vconst_like u c = broadcast (const ~dtype:(dtype u) c) (max_numel u)

let rop u op axes =
  let axes = List.sort Int.compare axes in
  let s = shape u in
  let reduce_axes =
    List.filter (fun a -> Sint.resolve Sint.(List.nth s a <> Int 1)) axes
  in
  let kept = List.filteri (fun i _ -> not (List.mem i axes)) s in
  if List.is_empty reduce_axes then reshape u kept
  else
    let perm =
      reduce_axes
      @ List.filter
          (fun i -> not (List.mem i reduce_axes))
          (List.init (List.length s) Fun.id)
    in
    let ret =
      v Op.Reduce
        ~src:[ permute u perm ]
        ~arg:(Reduce { op; num_axes = List.length reduce_axes })
    in
    if axes <> reduce_axes then reshape ret kept else ret

(* Data types *)

let nbytes u =
  match numel u with
  | Int n -> n * element_size u
  | Sym s -> Bigint.to_int (to_z s) * element_size u

let cat ?(axis = 0) u rest =
  let axis = resolve_dim u axis in
  let s = shape u in
  List.iter
    (fun x ->
      let sx = shape x in
      if
        List.length sx <> List.length s
        || not
             (List.for_all2
                (fun (i, a) b -> i = axis || equal_sint a b)
                (List.mapi (fun i a -> (i, a)) s)
                sx)
      then
        invalid_argf "cannot concatenate %s and %s" (repr_shape s)
          (repr_shape sx))
    rest;
  let dim x = List.nth (shape x) axis in
  if List.for_all (fun x -> equal_sint (dim x) (dim u)) rest then
    flatten ~start:axis ~stop:(axis + 1) (stack ~axis (u :: rest))
  else
    let all = u :: rest in
    let starts =
      List.rev
        (List.fold_left
           (fun acc x -> Sint.(List.hd acc + dim x) :: acc)
           [ Int 0 ] all)
    in
    let total = List.nth starts (List.length all) in
    let padded =
      List.mapi
        (fun i x ->
          pad x
            (List.init (ndim x) (fun j ->
                 if j = axis then
                   let next = List.nth starts (i + 1) in
                   Some (List.nth starts i, Sint.(total - next))
                 else None)))
        all
    in
    usum (List.hd padded) (List.tl padded)

(* Running operations *)

(* Chunks of 16: each element reduces a window of at most 16 elements, and each
   factor of 16 in the axis's length adds a stage. *)
let split_cumalu = 16

(* [None] for each axis of [u] but [axis], which is [Some p]. *)
let at_axis u axis p =
  List.init (ndim u) (fun i -> if i = axis then Some p else None)

let running_size u axis =
  match List.nth (shape u) axis with
  | Int n -> n
  | Sym _ -> invalid_arg "a running operation along an axis of symbolic size"

(* The running [op] along the last axis of [u]: over each element's window of
   the elements up to it, the axis padded before with [op]'s identity. *)
let pooled_cumalu u op =
  let last = ndim u - 1 in
  let n = running_size u last in
  let value = identity_element op (dtype u) in
  rop
    (pool (pad ~value u (at_axis u last (Int (n - 1), Int 0))) [ n ])
    op
    [ last + 1 ]

let combiner op =
  if Op.equal op Op.Add then add
  else if Op.equal op Op.Mul then mul
  else maximum

(* The running [op] along the last axis of [t], past 16 elements: chunks of 16
   from the axis's first element, each combined with the running [op] of the
   chunks' last elements before it. Chunks start at the axis's first element so
   that an element's chunk, and the order of its operations, depend on its
   index alone. *)
let rec scan ~dense t op =
  let last = ndim t - 1 in
  let s = running_size t last in
  if s <= split_cumalu then pooled_cumalu t op
  else
    let value = identity_element op (dtype t) in
    let along x lo hi = shrink x (at_axis x last (Int lo, Int hi)) in
    let combine = combiner op in
    let q = s / split_cumalu and r = s mod split_cumalu in
    let span = q * split_cumalu in
    (* Dense, the first stage runs over the axis's full chunks alone, whose
       loads are those of whole rows, and the tail joins after; otherwise the
       last chunk is padded after, which the later stages, 16 times smaller,
       can afford. *)
    let body, n =
      if r = 0 then (t, q)
      else if dense then (along t 0 span, q)
      else
        ( pad ~value t (at_axis t last (Int 0, Int (split_cumalu - r))),
          q + 1 )
    in
    let chunks =
      pooled_cumalu (unflatten body last [ Int n; Int split_cumalu ]) op
    in
    let ends =
      squeeze ~axis:(last + 1)
        (shrink chunks
           (at_axis chunks (last + 1)
              (Int (split_cumalu - 1), Int split_cumalu)))
    in
    let sums = scan ~dense:false ends op in
    let base = pad ~value sums (at_axis sums last (Int 1, Int (-1))) in
    let whole =
      flatten ~start:last
        (combine chunks (reshape base (shape base @ [ Int 1 ])))
    in
    if r = 0 then whole
    else if not dense then along whole 0 s
    else
      (* The tail runs as a chunk of its own and joins the full chunks by [op]
         with its identity, which changes only a zero sum's sign or a
         maximum's result at a NaN. *)
      let tail = along t span s in
      let tail =
        pad ~value tail (at_axis tail last (Int 0, Int (split_cumalu - r)))
      in
      let tail =
        combine (along (pooled_cumalu tail op) 0 r) (along sums (q - 1) q)
      in
      combine
        (pad ~value whole (at_axis whole last (Int 0, Int r)))
        (pad ~value tail (at_axis tail last (Int span, Int 0)))

let cumalu u axis op =
  let axis = resolve_dim u axis and last = ndim u - 1 in
  ignore (running_size u axis);
  if List.exists (fun d -> equal_sint d (Int 0)) (shape u) then u
  else transpose (scan ~dense:true (transpose u axis last) op) axis last

let full dt c k = expand (const ~dtype:dt (`Int (Bigint.of_int c))) [ Int k ]

(* The longest run built as a pooled running sum: its pool repeats the [2n - 1]
   padded steps to [(n + 1)(2n - 1)] elements, which fit an int up to here. *)
let pooled_run = 1518500249

(* The longest arange: its rows, of [b <= 2^30] elements since [b² <= n], pad
   the last one by fewer than [2^30] within an int. *)
let longest_arange = max_int - (1 lsl 30)

(* The [n] elements [start + i * step], of type [dt]. Past [pooled_run] they are
   laid out in rows of [b] elements, [b] the power of two with
   [b² <= n < 4b²]: a row's first element is an element of the run by
   [b * step], to which the run by [step] of the offsets along the row is
   added. The first elements are elements of the arange, and the offsets,
   fewer than [b <= √n] steps, are far within [dt], so every element is
   exact in an integer type, an unsigned one wrapping as it does in the
   running sum. *)
let rec run ~start ~step dt n =
  if n <= pooled_run then
    let before = Bigint.(sub (of_int start) (of_int step)) in
    add (pooled_cumalu (full dt step n) Op.Add) (const (`Int before))
  else
    let rec side b = if b <= n / (4 * b) then side (2 * b) else b in
    let b = side 1 in
    let a = Helpers.ceildiv n b in
    let grid = [ Int a; Int b ] in
    let firsts = reshape (run ~start ~step:(b * step) dt a) [ Int a; Int 1 ] in
    let offsets = reshape (run ~start:0 ~step dt b) [ Int 1; Int b ] in
    let rows = add (expand firsts grid) (expand offsets grid) in
    shrink (reshape rows [ Sint.(Int a * Int b) ]) [ Some (Int 0, Int n) ]

let arange ?(start = 0) ?(step = 1) ?dtype stop =
  if step = 0 then invalid_arg "an arange of step 0";
  let z = Bigint.of_int in
  let count = Bigint.(cdiv (sub (z stop) (z start)) (z step)) in
  if Bigint.(gt count (z longest_arange)) then
    invalid_argf "an arange [%d, %d) by %d has more than %d elements" start
      stop step longest_arange;
  let n = if Bigint.sign count <= 0 then 0 else Bigint.to_int count in
  (* The bounds of the elements, [start] and the last; an empty arange's are
     tinygrad's, [start] and [stop - step]. *)
  let lo, hi =
    let last =
      if n > 0 then Bigint.add (z start) (Bigint.mul (z (n - 1)) (z step))
      else Bigint.sub (z stop) (z step)
    in
    if step > 0 then (z start, last) else (last, z start)
  in
  let dt =
    match dtype with Some dt -> dt | None -> Dtype.commit_int lo hi
  in
  if Value.(`Int lo < Dtype.min dt) || Value.(Dtype.max dt < `Int hi) then
    invalid_argf "arange [%d, %d) is not representable in %s" start stop
      (repr_dtype dt);
  if n = 0 then full dt 0 0
  else
    (* Rows of a float type are laid out in [Int64], which holds the integers
       exactly, so that each element is rounded once. *)
    let acc =
      if not (Dtype.is_float dt) then dt
      else if n > pooled_run then Int64
      else Dtype.least_upper [ dt; Float32 ]
    in
    cast (run ~start ~step acc n) dt

(* Variables *)

let is_variable u =
  op u = Op.Param
  && (match arg u with
    | Param { vmin_vmax = Some _; addrspace = Dtype.Alu; _ } -> true
    | _ -> false)
  && match shape_opt u with Some [] -> true | _ -> false

(* Divisibility *)

let rec divides u (n : Bigint.t) =
  if Bigint.equal n Bigint.one then Some u
  else
    match op u with
    | _ when is_literal u ->
        let x = as_value (Dtype.const (dtype u) (value u)) and n = `Int n in
        if Value.(x % n = of_int 0) then
          Some (const_like u (Value.(x // n) :> Dtype.const))
        else None
    | Op.Add -> (
        match (divides (nth u 0) n, divides (nth u 1) n) with
        | Some d0, Some d1 -> Some (add d0 d1)
        | _ -> None)
    | Op.Mul -> (
        match divides (nth u 0) n with
        | Some d0 -> Some (mul d0 (nth u 1))
        | None -> Option.map (fun d1 -> mul (nth u 0) d1) (divides (nth u 1) n))
    | _ -> None

(* Multisets of nodes, in order of first insertion. *)
let add_count k counts t =
  match List.assq_opt t counts with
  | Some _ ->
      List.map (fun (u, m) -> if u == t then (u, m + k) else (u, m)) counts
  | None -> counts @ [ (t, k) ]

let count_terms terms = List.fold_left (add_count 1) [] terms
let subtract counts terms = List.fold_left (add_count (-1)) counts terms

let elements counts =
  List.concat_map (fun (u, n) -> List.init (max 0 n) (fun _ -> u)) counts

let product start terms = List.fold_left mul start terms

let gcd us =
  match us with
  | [] -> invalid_arg "gcd of nothing"
  | first_u :: _ ->
      let popped = List.map (pop_const ~op:Op.Mul) us in
      let common =
        List.fold_left
          (fun acc (term, _) ->
            let c = count_terms (split_uop term Op.Mul) in
            List.filter_map
              (fun (u, n) ->
                match List.assq_opt u c with
                | Some m when min n m > 0 -> Some (u, min n m)
                | _ -> None)
              acc)
          (count_terms (split_uop (fst (List.hd popped)) Op.Mul))
          (List.tl popped)
      in
      let factors =
        if List.is_empty common then List.map const_factor us
        else List.map (fun (_, c) -> integer c) popped
      in
      (* The coefficient is a number, a scalar whatever the shape of [us]: a
         broadcast 1 would not read as 1, and divides no term. *)
      product
        (const ~dtype:(dtype first_u)
           (`Int (List.fold_left Bigint.gcd Bigint.zero factors)))
        (elements common)

let rec divide_exact u d =
  if u == d then Some (const_like u (`Int Bigint.one))
  else if op d = Op.Const then divides u (Value.to_z (as_value (value d)))
  else
    match op u with
    | Op.Add -> (
        match (divide_exact (nth u 0) d, divide_exact (nth u 1) d) with
        | Some s0, Some s1 -> Some (add s0 s1)
        | _ -> None)
    | Op.Mul ->
        let fac, c = pop_const ~op:Op.Mul u
        and dfac, dc = pop_const ~op:Op.Mul d in
        let c = as_value c and dc = as_value dc in
        let counts =
          subtract (count_terms (split_uop fac Op.Mul)) (split_uop dfac Op.Mul)
        in
        if
          Value.(c % dc = of_int 0)
          && List.for_all (fun (_, n) -> n >= 0) counts
        then
          Some
            (product
               (const_like u (Value.(c // dc) :> Dtype.const))
               (elements counts))
        else None
    | _ -> None

(* Evaluation *)

(* A cast in a symbolic integer converts without truncating. *)
let sym_cast dt x : Dtype.value =
  if Dtype.is_float dt then `Float (Value.to_float x)
  else if Dtype.equal dt Dtype.Bool then `Bool (Value.to_bool x)
  else `Int (Value.to_z x)

let sym_alu op dt xs =
  as_value
    (exec_alu ~truncate_output:false op dt
       (List.map (fun x -> (x :> Dtype.const)) xs))

(* The integer operations [sym_compile] computes on [int]s. Each raises
   [Inexact] where the result may not be an [int], and the operation is then
   computed exactly. *)

exception Inexact

(* Whether [x] times a value of the same bound fits an [int]. *)
let small x = x > -0x4000_0000 && x < 0x4000_0000

let int_add x y =
  let s = x + y in
  if x >= 0 = (y >= 0) && s >= 0 <> (x >= 0) then raise Inexact else s

let int_mul x y = if small x && small y then x * y else raise Inexact

(* A division by zero is zero, and its remainder the dividend. *)
let int_quotient ~toward_zero x y =
  if y = 0 then 0
  else if x = min_int then raise Inexact
  else
    let q = x / y in
    if toward_zero || x mod y = 0 || x < 0 = (y < 0) then q else q - 1

let int_remainder ~toward_zero x y =
  int_add x (-int_mul (int_quotient ~toward_zero x y) y)

let int_binary = function
  | Op.Add -> Some int_add
  | Op.Sub ->
      Some (fun x y -> int_add x (if y = min_int then raise Inexact else -y))
  | Op.Mul -> Some int_mul
  | Op.Max -> Some Int.max
  | Op.Truncdiv -> Some (int_quotient ~toward_zero:true)
  | Op.Truncmod -> Some (int_remainder ~toward_zero:true)
  | Op.Div -> Some (int_quotient ~toward_zero:false)
  | Op.Mod -> Some (int_remainder ~toward_zero:false)
  | _ -> None

(* [s]'s value in an environment, computed exactly, and the same on [int]s,
   which raises [Inexact], if [s] is an integer of integer operations. *)
let compiled (s : sint) var =
  match s with
  | Int n -> ((fun _ -> `Int (Bigint.of_int n)), Some (fun _ -> n))
  | Sym u ->
      let s = simplify u in
      let memo tbl f n =
        match Tbl.find_opt tbl n with
        | Some g -> g
        | None ->
            let g = f n in
            Tbl.replace tbl n g;
            g
      in
      (* Each node's value, computed exactly. *)
      let values = Tbl.create 16 in
      let rec compute n = memo values exact n
      and exact n =
        match op n with
        | Op.Const ->
            let v = as_value (value n) in
            fun _ -> v
        | Op.Param when addrspace n = Some Dtype.Alu ->
            let read = var n in
            fun env -> `Int (Bigint.of_int (read env))
        (* an after is its first source: the rest only order it *)
        | Op.After -> compute (first (op n) (src n))
        | Op.Cast ->
            let x = compute (first (op n) (src n)) in
            fun env -> sym_cast (dtype n) (x env)
        | Op.Bitcast ->
            let src = first (op n) (src n) in
            let x = compute src in
            fun env -> Dtype.bitcast (dtype src) (dtype n) (x env)
        | op when Op.Set.mem op Op.Set.alu ->
            let xs = List.map compute (src n) in
            fun env -> sym_alu op (dtype n) (List.map (fun x -> x env) xs)
        | op -> fun _ -> invalid_argf "%s cannot be evaluated" (Op.name op)
      in
      (* An integer node of integer sources, computed on [int]s. *)
      let ints = Tbl.create 16 in
      let rec int n = memo ints native n
      and native n =
        if not (Dtype.is_int (dtype n)) then None
        else
          match (op n, src n) with
          | Op.Const, _ -> (
              match as_value (value n) with
              | `Int z when Bigint.fits_int z ->
                  let c = Bigint.to_int z in
                  Some (fun _ -> c)
              | _ -> None)
          | Op.Param, _ when addrspace n = Some Dtype.Alu ->
              Some (var n)
          | Op.After, x :: _ -> int x
          | Op.Neg, [ x ] ->
              Option.map
                (fun x env ->
                  let v = x env in
                  if v = min_int then raise Inexact else -v)
                (int x)
          | o, [ x; y ] -> (
              match (int_binary o, int x, int y) with
              | Some f, Some x, Some y -> Some (fun env -> f (x env) (y env))
              | _ -> None)
          | _ -> None
      in
      (compute s, int s)

let sym_compile s var =
  match compiled s var with
  | exact, None -> fun env -> Value.to_int (exact env)
  | exact, Some f -> (
      fun env -> try f env with Inexact -> Value.to_int (exact env))

let sym_compile_z s var =
  match compiled s var with
  | exact, None -> fun env -> Value.to_z (exact env)
  | exact, Some f -> (
      fun env ->
        try Bigint.of_int (f env) with Inexact -> Value.to_z (exact env))

let sym_infer s vars =
  let var n vars =
    let name = expr n in
    match List.assoc_opt name vars with
    | Some x -> x
    | None -> invalid_argf "the variable %s has no value" name
  in
  sym_compile s var vars

(* Graph construction that rewrites *)

let contract u rs =
  List.iter
    (fun r ->
      if not (Axis_type.equal (axis_type r) Axis_type.Upcast) then
        invalid_arg "contracted ranges must be upcast")
    rs;
  let rec product = function
    | [] -> [ [] ]
    | r :: rest ->
        let n = Value.to_int (vmax r) + 1 in
        List.concat_map
          (fun i -> List.map (fun t -> i :: t) (product rest))
          (List.init n Fun.id)
  in
  stack
    (List.map
       (fun idx ->
         substitute ~calls:Skip ~pass:Fixed_point u
           (List.map2 (fun r i -> (r, const_like r (`Int (Bigint.of_int i)))) rs idx))
       (product rs))

(* Symbolic rules *)

module V = Dtype.Value

(* Weak constants *)

let is_weak dt = Dtype.mem dt Dtype.weaks
let weak u = is_weak (dtype u)
let is_const u = op u = Op.Const
let unchanged src u = List.equal ( == ) src (Ops.src u)

(* The committed types [u] commits its sources at: the operands' meet and [u]'s
   own derived type, [None] if either is weak. *)
let derived_dtypes u src =
  if not (Op.Set.mem (op u) Op.Set.broadcastable) then None
  else
    let meet = promo_dtype src in
    if is_weak meet then None
    else
      let result = dtype_of (op u) src (arg u) in
      if is_weak result then None else Some (meet, result)

(* Drop the cast off a committed constant where the consumer derives it anyway,
   so rules keyed on bare constants keep matching. The drop must change nothing
   the consumer derives: neither the operands' meet nor the node's own type. *)
let uncast_const u =
  (* A weak cast over a constant is not a commit: it is still resolving. The
     literal left is the constant at the width it was committed to, as a machine
     holds it. A NaN or an infinity has no integer value: its literal is left as
     it is written, and the consumer's derived type keeps the cast. *)
  let uncast s =
    if op s = Op.Cast && (not (weak s)) && is_const (nth s 0) && weak (nth s 0)
    then
      let dt = dtype s in
      match value s with
      | `Float f
        when not (Float.is_finite f || Dtype.is_float dt || Dtype.is_bool dt) ->
          nth s 0
      | c -> (
          match Dtype.const dt c with
          | #Dtype.value as v -> const (Dtype.truncate dt v :> Dtype.const)
          | `Invalid -> s)
    else s
  in
  let src = List.map uncast (Ops.src u) in
  if unchanged src u then None
  else
    match derived_dtypes u src with
    | Some (meet, result)
      when Dtype.equal meet (promo_dtype (Ops.src u))
           && Dtype.equal result (dtype u) ->
        Some (replace ~src u)
    | _ -> None

let pm_uncast_const =
  Pattern_matcher.(
    v
      (fun () -> [
        rule (Upat.v ~op:Op.Set.broadcastable ~name:"u" ()) (fun m ->
            uncast_const (m "u"));
      ]))

(* Division and remainder *)

exception Not_a_number

let number : Dtype.const -> V.t = function
  | #Dtype.value as v -> v
  | `Invalid -> raise_notrace Not_a_number

let rule p f =
  Pattern_matcher.rule p (fun m -> try f m with Not_a_number -> None)

let zero = V.of_int 0
let num u = number (value u)

(* Python's integers in node arithmetic: [lit n] is the weak literal an integer
   becomes, and [sum] starts from the integer 0, as Python's does. *)
let lit (n : V.t) = const (n :> Dtype.const)
let sum us = List.fold_left add (lit zero) us
let size u = Nodes.cardinal (backward_slice ~calls:Skip u)
let zero_like u = const_like u (zero :> Dtype.const)

(* [all_ones x n] is whether [n] is -1 as a constant of [x]'s type, every bit
   set: an unsigned type wraps it. *)
let all_ones x n =
  Dtype.equal_const
    (Dtype.const (dtype x) (`Int n))
    (Dtype.const (dtype x) (`Int Bigint.minus_one))

(* [first_rule rules] is the result of the first rule that applies. *)
let first_rule = List.find_map (fun rule -> rule ())

(* itertools.product: the first list varies slowest. *)
let rec cartesian = function
  | [] -> Seq.return []
  | xs :: rest ->
      Seq.flat_map
        (fun x -> Seq.map (List.cons x) (cartesian rest))
        (List.to_seq xs)

(* Memoised on index nodes, for as long as the node lives; compilation runs on
   domains, so the table is locked. *)
module Memo = Ephemeron.K1.Make (struct
  type nonrec t = t

  let equal = ( == )
  let hash = hash
end)

let memo = Memo.create 64
let memo_lock = Mutex.create ()

let rec fold_divmod_general d =
  match Mutex.protect memo_lock (fun () -> Memo.find_opt memo d) with
  | Some ret -> ret
  | None ->
      let ret = fold d in
      Mutex.protect memo_lock (fun () -> Memo.replace memo d ret);
      ret

and fold d =
  let x = nth d 0 and y = nth d 1 and is_mod = op d = Op.Mod in
  if V.(vmin y = zero && vmax y = zero) then raise Division_by_zero;
  (* x % y is x - q * y where x // y is a constant q *)
  let constant_quotient () =
    if not is_mod then None
    else
      let q = O.(x // y) in
      if V.(vmin q = vmax q) then Some O.(x - (lit (vmin q) * y)) else None
  in
  let x_peeled, const = pop_const x in
  let const = number const and uops_no_const = split_uop x_peeled Op.Add in
  (* Constant denominator rules, for a constant c > 0 *)
  let c = if op y = Op.Const then num y else zero in
  (* nested_div: (x % (k * c)) // c is (x // c) % k for k > 0; the mod case is
     remove_nested_mod's *)
  let nested_div () =
    if is_mod || op x <> Op.Mod then None
    else
      match divides (nth x 1) (V.to_z c) with
      | Some k when V.(vmin k > zero) -> Some O.(nth x 0 // y % k)
      | _ -> None
  in
  (* remove_nested_mod in a sum: (a % 4 + b) % 2 is (a + b) % 2 *)
  let remove_nested_mod () =
    let unnest u =
      if op u = Op.Mod && Option.is_some (divides (nth u 1) (V.to_z c))
      then nth u 0
      else u
    in
    let xs = List.map unnest uops_no_const in
    if (not is_mod) || List.equal ( == ) xs uops_no_const then None
    else Some O.((usum (List.hd xs) (List.tl xs) + lit const) % y)
  in
  (* exact division keeps the structure of x: (q * c + k) // c is q + k // c *)
  let exact_division () =
    if is_mod then None
    else
      Option.map
        (fun q -> O.(q + lit V.(const // c)))
        (divides x_peeled (V.to_z c))
  in
  let folds () =
    (* The shared decomposition: const_factor u divides u *)
    let factors = List.map (fun u -> `Int (const_factor u)) uops_no_const in
    let terms =
      List.map2
        (fun u f -> Option.get (divides u (V.to_z f)))
        uops_no_const factors
    in
    (* fold_divmod_congruence: fold if x is congruent to an expression whose
       range is within one period of c. A lone term (a binary numerator that
       crosses one period) or an exact f % c = c // 2 tie tries both signs of
       the remainder; otherwise the smaller keeps the product small. *)
    let congruence () =
      let lone = List.compare_length_with terms 1 = 0 in
      let choices f =
        (* r is in [0, c), so r is the smaller in magnitude iff r <= c - r *)
        let r = V.(f % c) in
        if V.(r * of_int 2 = c) || lone then [ r; V.(r - c) ]
        else [ (if V.(r <= c - r) then r else V.(r - c)) ]
      in
      let fold rems =
        let rem =
          O.(
            sum (List.map2 (fun r v -> lit r * v) rems terms)
            + lit V.(const % c))
        in
        let q = V.(vmin rem // c) in
        if V.(q <> vmax rem // c) then None
        else if is_mod then Some O.(rem - lit V.(q * c))
        else
          let quotient (f, r) v = O.(lit V.((f - r) // c) * v) in
          let quotients =
            List.map2 quotient (List.combine factors rems) terms
          in
          Some O.(sum quotients + lit V.(const // c) + lit q)
      in
      Seq.find_map fold (cartesian (List.map choices factors))
    in
    (* nest_by_factor: x // c is (x // f) // (c // f), and x % c is (x // f %
       (c // f)) * f + x % f, for f a term's factor dividing c or the gcd of
       every factor and c. The division holds for x of any sign; the
       remainder's reconstruction needs x >= 0 *)
    let nest_by_factor () =
      let divisor f =
        let f = V.(max f (-f)) in
        if V.(of_int 1 < f && f < c && c % f = zero) then Some f else None
      in
      let g =
        `Int
          (List.fold_left
             (fun g f -> Bigint.gcd g (V.to_z f))
             (V.to_z c) factors)
      in
      let divs =
        List.sort_uniq V.compare (List.filter_map divisor (g :: factors))
      in
      let remainder newxs div =
        let part f t =
          if V.(f % div = zero) then [] else [ O.(lit V.(f % div) * t) ]
        in
        let parts = List.concat (List.map2 part factors terms) in
        let parts =
          if V.(const % div = zero) then parts
          else parts @ [ const_like x (V.(const % div) :> Dtype.const) ]
        in
        let b = match parts with [] -> zero_like x | b :: bs -> usum b bs in
        if V.(zero <= vmin b && vmax b < div) then
          let r = O.((newxs % lit V.(c // div) * lit div) + b) in
          Some (size r, r)
        else None
      in
      let result div =
        match fold_divmod_general O.(x // lit div) with
        | Some newxs when not is_mod ->
            Some (size newxs, O.(newxs // lit V.(c // div)))
        | Some newxs when V.(vmin x >= zero && vmin newxs >= zero) ->
            remainder newxs div
        | _ -> None
      in
      let smaller (n0, r0) (n1, r1) =
        if n1 < n0 then (n1, r1) else (n0, r0)
      in
      match List.filter_map result divs with
      | [] -> None
      | r :: rs -> Some (snd (List.fold_left smaller r rs))
    in
    first_rule [ congruence; nest_by_factor ]
  in
  (* Variable denominator and fallback rules *)
  let all_uops = split_uop x Op.Add in
  (* divide_by_gcd: x // y is (x // gcd) // (y // gcd) *)
  let divide_by_gcd () =
    let g = simplify (gcd (all_uops @ [ y ])) in
    if op g = Op.Const && V.(num g = of_int 1) then None
    else
      let ret =
        alu
          (Option.get (divide_exact x g))
          (op d)
          [ Option.get (divide_exact y g) ]
      in
      Some (if is_mod then O.(ret * g) else ret)
  in
  (* factor_remainder: (d * x + y) // d is x + y // d *)
  let factor_remainder () =
    let split u (quo, rem) =
      let f = `Int (const_factor u) in
      match divide_exact u y with
      | Some q -> (q :: quo, rem)
      | None when op y = Op.Const && V.(f % c <> f) ->
          let t = Option.get (divides u (V.to_z f)) in
          let q = if is_mod then zero_like u else O.(t * lit V.(f // c)) in
          (q :: quo, O.(t * lit V.(f % c)) :: rem)
      | None -> (quo, u :: rem)
    in
    match List.fold_right split all_uops ([], []) with
    | [], _ -> None
    | quo, rem ->
        let new_x = O.(sum rem + zero_like x) in
        if V.(vmin new_x < zero) then None
        else
          Some (if is_mod then O.(new_x % y) else O.((new_x // y) + sum quo))
  in
  let fallback () =
    if V.(vmin y < zero || vmin x < zero) then None else factor_remainder ()
  in
  let constant_rules =
    if V.(c > zero) then
      [ nested_div; remove_nested_mod; exact_division; folds ]
    else []
  in
  first_rule
    ((constant_quotient :: constant_rules) @ [ divide_by_gcd; fallback ])

let floor_ops = Op.Set.of_list [ Op.Div; Op.Mod ]

let div_and_mod_symbolic =
  Pattern_matcher.v
    (fun () -> [
      (* Fast inline rules *)
      (* (x // c + a) // d is (x + a * c) // (c * d) for d > 0, on index
         arithmetic, which never wraps *)
      rule
        Upat.(((var ~dtype:[ Dtype.Weak_int ] "x" // cvar "c") + cvar "a") // cvar "d")
        (fun m ->
          let x = m "x" and c = m "c" and a = m "a" and d = m "d" in
          if V.(vmin d > zero) then Some O.((x + (a * c)) // (c * d)) else None);
      (* (x + c) // d is (x + c % d) // d + c // d, and (x + c) % d is (x + c %
         d) % d: the multiple of d leaves the constant, for any d <> 0 *)
      rule
        (Upat.v ~op:floor_ops
           ~src:
             [
               Upat.(var "x" ~dtype:[ Dtype.Weak_int ] + cvar "c");
               Upat.cvar "d";
             ]
           ~name:"n" ())
        (fun m ->
          let x = m "x" and c = num (m "c") and d = m "d" in
          let dv = num d in
          if V.(dv = zero || c % dv = c) then None
          else if op (m "n") = Op.Div then
            Some O.(((x + lit V.(c % dv)) // d) + lit V.(c // dv))
          else Some O.((x + lit V.(c % dv)) % d));
      (* Slow rules *)
      rule (Upat.v ~op:floor_ops ~dtype:[ Dtype.Weak_int ] ~name:"d" ())
        (fun m -> fold_divmod_general (m "d"));
    ])

(* Cleaning up movements *)

let is_int_const i c =
  match value c with
  | #Dtype.value as v -> Dtype.Value.(v = of_int i)
  | `Invalid -> false

let shrink_arg u =
  match marg u with Shrink arg -> arg | _ -> invalid_arg "not a shrink"

let permute_arg u =
  match marg u with Permute arg -> arg | _ -> invalid_arg "not a permute"

let mop_cleanup =
  Pattern_matcher.(
    v
      (fun () -> [
        (* Merge adjacent shrinks. *)
        rule
          (Upat.f
             (Upat.op Op.Shrink ~name:"x")
             Op.Shrink ~allow_any_len:true ~name:"s")
          (fun m ->
            let x = m "x" and s = m "s" in
            let merge (o, _) (p, n) = (Sint.(o + p), n) in
            Some
              (mop (nth x 0)
                 (Shrink (List.map2 merge (shrink_arg x) (shrink_arg s)))));
        (* Merge adjacent reshapes. *)
        rule
          (Upat.op Op.Reshape ~name:"x"
             ~src:[ Upat.op Op.Reshape ~name:"x2"; Upat.wild ])
          (fun m ->
            let x = m "x" in
            Some (replace ~src:[ nth (m "x2") 0; nth x 1 ] x));
        (* Remove no-op reshapes. *)
        rule
          (Upat.op Op.Reshape ~name:"x" ~src:[ Upat.var "x2"; Upat.wild ])
          (fun m ->
            let x2 = m "x2" in
            match shape_opt x2 with
            | Some s when List.equal Sint.equal s (shape (m "x")) -> Some x2
            | _ -> None);
        (* Merge permutes. *)
        rule
          (Upat.op Op.Permute ~name:"x"
             ~src:[ Upat.op Op.Permute ~name:"x2"; Upat.wild ])
          (fun m ->
            let x2 = m "x2" in
            let order =
              List.map (List.nth (permute_arg x2)) (permute_arg (m "x"))
            in
            Some (permute (nth x2 0) order));
        (* A stack of indexes by constants. *)
        rule
          (Upat.op Op.Stack ~name:"stk"
             ~each:(Upat.op Op.Index ~src:[ Upat.var "src"; Upat.op Op.Const ]))
          (fun m ->
            let stk = m "stk" and src = m "src" in
            let in_order i x = is_int_const i (nth x 1) in
            if
              List.equal Sint.equal (shape stk) (shape src)
              && List.for_all Fun.id (List.mapi in_order (Ops.src stk))
            then Some src
            else None);
        (* A constant index into a stack is that stack's source. *)
        rule
          (Upat.op Op.Index ~name:"idx" ~allow_any_len:true
             ~src:[ Upat.op Op.Stack ~name:"a"; Upat.cvar "i" ])
          (fun m ->
            let x = index (m "a") [ m "i" ] in
            match List.drop 2 (src (m "idx")) with
            | [] -> Some x
            | rest -> Some (index x rest));
        (* An index into a stack past its first axis indexes each lane, then
           selects. *)
        rule
          (Upat.op Op.Index ~name:"idx" ~allow_any_len:true
             ~src:[ Upat.op Op.Stack ~name:"a"; Upat.var "i" ])
          (fun m ->
            let a = m "a" in
            match List.drop 2 (src (m "idx")) with
            | [] -> None
            | rest ->
                let lane x = index x rest in
                Some (index (replace a ~src:(List.map lane (src a))) [ m "i" ]));
        (* An index of an index is one index. *)
        rule
          (Upat.op Op.Index ~name:"idx2" ~allow_any_len:true
             ~src:[ Upat.op Op.Index ~name:"idx1" ~allow_any_len:true ])
          (fun m ->
            let idx1 = m "idx1" in
            let idxs = List.tl (src idx1) @ List.tl (src (m "idx2")) in
            if List.for_all (fun x -> List.is_empty (shape x)) idxs then
              Some (index (nth idx1 0) idxs)
            else None);
        (* An index of a shaped index. *)
        rule
          (Upat.op Op.Index ~name:"idx2" ~allow_any_len:true
             ~src:
               [ Upat.op Op.Index ~src:[ Upat.var "buf"; Upat.var "idx1_arg" ] ])
          (fun m ->
            let idx1_arg = m "idx1_arg"
            and idxs = List.drop 1 (src (m "idx2")) in
            if List.compare_length_with idxs (ndim idx1_arg) = 0 then
              Some (index (m "buf") [ index idx1_arg idxs ])
            else None);
      ]))

(* Simplification rules *)

let pop_num ?op u =
  let x, c = pop_const ?op u in
  (x, number c)

let equals u v =
  match value u with #Dtype.value as x -> V.(x = v) | _ -> false

let pm = Pattern_matcher.v
let ops = Op.Set.of_list
let one = V.of_int 1
let const_v u (v : V.t) = const_like u (v :> Dtype.const)
let sum_of = function u :: us -> usum u us | [] -> invalid_arg "empty sum"

let conj = function
  | u :: us -> uprod u us
  | [] -> invalid_arg "empty conjunction"

(* A NaN or an infinity has no value in an integer type: a rule that would
   convert one does not apply. *)
let convertible dt : V.t -> bool = function
  | `Float f -> Float.is_finite f || Dtype.is_float dt || Dtype.is_bool dt
  | _ -> true

(* [v] as a machine holds it in [dt]: converted, then wrapped to [dt]'s
   width. *)
let at dt v =
  if not (convertible dt v) then raise_notrace Not_a_number;
  Dtype.truncate dt (number (Dtype.const dt v))

(* Phase 1: the most generic folding rules *)

(* The reciprocal overflows only where a power of magnitude at least 1 does, and
   [sqrt] gives -0. and NaN at -0. and -inf, where a half-integer power is +0.
   and +inf. *)
let simplify_pow x c =
  let c = num c and pow x v = pow x (lit v) in
  let h = V.(c - `Float 0.5) in
  match c with
  | `Float f when not (Float.is_finite f) -> None
  | _ when V.(c < zero) && Dtype.is_int (dtype x) -> None
  | _ when V.(c <= `Float (-1.)) -> Some (pow (reciprocal x) V.(-c))
  | _ when V.(c < zero) -> None
  | _ when V.(c = zero) -> Some (const_v x one)
  | _ when V.(h < c && `Float (Float.trunc (to_float h) +. 0.5) = c) ->
      let p = mul (pow x h) (sqrt x) in
      if not (Dtype.is_float (dtype x)) then Some p
      else
        let special v r =
          where O.(x <> float v) r (const_like x (`Float (Float.abs v)))
        in
        Some (special Float.neg_infinity (special 0. p))
  | _ when V.(`Int (to_z c) = c) ->
      let y = pow x V.(c // of_int 2) in
      Some O.(y * y * if V.(c % of_int 2 = one) then x else int 1)
  | _ -> None

let fold_bitcast root c =
  let dt = dtype c in
  if Dtype.itemsize dt <> Dtype.itemsize (dtype root) then None
  else
    (* the value is read as [dt] stores it: an integer is mathematical and may
       not fit, so it wraps to the stated width, and a NaN keeps its bits, which
       a conversion would quiet *)
    let v =
      if Dtype.is_float dt then number (Dtype.const dt (num c))
      else Dtype.truncate dt (num c)
    in
    Some (const_v root (Dtype.bitcast dt (dtype root) v))

(* A committed integer holds its type's value, so a fold reads a committed
   operand, and a weak integer operand the operation commits, at the width of
   the operation's operands, and writes a committed integer result at its width;
   floats re-round in the mint. A stack folds lane by lane. A shift by a
   negative count has no value, and does not fold. *)
let fold_const_alu a =
  let alu args = exec_alu (op a) (dtype a) args in
  let operands =
    if Op.Set.mem (op a) Op.Set.comparison then promo_dtype (src a) else dtype a
  in
  let read s =
    match (op s, value s) with
    (* a float constant is its bits: a NaN operand keeps them for a selection
       or a negation to move *)
    | Op.Cast, (#Dtype.value as v) when Dtype.is_float (dtype s) ->
        Dtype.const (dtype s) v
    | Op.Cast, (#Dtype.value as v) -> (at (dtype s) v :> Dtype.const)
    | Op.Const, (`Int _ as v)
      when Dtype.equal (dtype s) Dtype.Weak_int && Dtype.mem operands Dtype.ints
      ->
        (at operands v :> Dtype.const)
    | _, c -> c
  in
  let defined args =
    match (op a, args) with
    | (Op.Shl | Op.Shr), [ _; (#Dtype.value as n) ] -> V.(n >= zero)
    | _ -> true
  in
  let stack s = op s = Op.Stack in
  match List.filter stack (src a) with
  | [] ->
      let args = List.map read (src a) in
      if defined args then Some (const_like a (alu args)) else None
  | stacks ->
      let count =
        List.fold_left (fun n s -> max n (List.length (src s))) 0 stacks
      in
      let lane i s = read (if stack s then nth s i else s) in
      let lanes = List.init count (fun i -> List.map (lane i) (src a)) in
      if List.for_all defined lanes then
        Some (consts ~dtype:(dtype a) (List.map alu lanes))
      else None

(* [c1 op c2] for the operation [f] of a committed integer type, which reads
   a weak constant at its width: a weak result past that width is wrapped to
   it, as on int32 (x * 65536) * 65536 is x * 0, not x * 2^32. *)
let wrap_folded f c1 c2 =
  let dt = dtype f and c = alu c1 (op f) [ c2 ] in
  let weak u = op u = Op.Const && Dtype.mem (dtype u) Dtype.weaks in
  if not (Dtype.mem dt Dtype.ints && weak c1 && weak c2) then c
  else
    match (value c1, value c2) with
    | (`Int _ as v1), (`Int _ as v2) -> (
        match exec_alu (op f) Dtype.Weak_int [ v1; v2 ] with
        | `Int _ as v ->
            let w = Dtype.truncate dt v in
            if Dtype.equal_const w v then c
            else const_like c1 (w :> Dtype.const)
        | _ -> c)
    | _ -> c

(* A fold makes a signed constant only where it fits [x]'s type: wrapping it
   at the width would keep the value, but tinygrad, which takes signed
   overflow to be undefined, refuses the fold. Unsigned operations wrap. *)
let fits x (v : Dtype.value) =
  let dt = dtype x in
  (not (Dtype.mem dt Dtype.sints))
  ||
  match v with
  | `Int _ -> V.(Dtype.min dt <= v && v <= Dtype.max dt)
  | _ -> true

(* The factors like terms combine into: [c0 + c1], or [c + 1]. *)
let like_sum m = V.(vmin (m "c0") + vmin (m "c1"))
let like_succ m = V.(vmin (m "c") + of_int 1)

(* the B with q == B//div and B%div == base%div, or None. only such congruence
   is needed to recombine, and canonicalization moves consts freely: the
   quotient may be merged ((x//c + a)//div -> (x + a*c)//(c*div) for div>0) and
   shifted ((y + k*D)//D == y//D + k) *)
let quotient_base q base div =
  let (q, s), (n, a) = (pop_num q, pop_num base) in
  if op q <> Op.Div || not (is_const (nth q 1)) then None
  else
    let qd = num (nth q 1) in
    let merged =
      if V.(div > zero) && op n = Op.Div && is_const (nth n 1) then
        let c = num (nth n 1) in
        if V.(qd = c * div) then Some (nth n 0, V.(a * c), V.(c * div))
        else None
      else None
    in
    let found =
      match merged with
      | Some _ -> merged
      | None -> if V.(qd = div) then Some (n, a, div) else None
    in
    Option.bind found (fun (n, a, d) ->
        let (x, xa), (p, pa) = (pop_num n, pop_num (nth q 0)) in
        let t = V.(xa + a - pa) in
        if p != x || V.(t % d <> zero) then None
        else
          let k = V.((t // d) - s) in
          Some (if V.(k = zero) then base else sub base (lit V.(k * div))))

(* a scaled mod (base%div)*mul recombines with a partner q*(div*mul) carrying
   the quotient of a b == base (mod div): fully into b*mul when q == b//div, and
   partially into the wider mod (b%(div*d))*mul when q == (b//div)%d, for d>0 *)
let fold_add_divmod_recombine x =
  let terms = List.mapi (fun i t -> (i, t)) (split_uop x Op.Add) in
  let rest i j =
    List.filter_map
      (fun (k, t) -> if k = i || k = j then None else Some t)
      terms
  in
  terms
  |> List.find_map (fun (i, u) ->
      let md, mul = pop_num ~op:Op.Mul u in
      if op md <> Op.Mod || not (is_const (nth md 1)) then None
      else
        let base = nth md 0 and div = num (nth md 1) in
        terms
        |> List.find_map (fun (j, v) ->
            let q, scale = pop_num ~op:Op.Mul v in
            if i = j || V.(scale <> div * mul) then None
            else
              let recombine b = Some (usum O.(b * lit mul) (rest i j)) in
              match quotient_base q base div with
              | Some b -> recombine b
              | None when op q = Op.Mod && is_const (nth q 1) ->
                  let d = num (nth q 1) in
                  if V.(d <= zero) then None
                  else
                    Option.bind
                      (quotient_base (nth q 0) base div)
                      (fun b -> recombine O.(b % lit V.(div * d)))
              | None -> None))

(* Invalid poisons the value: ops move inside the gate so the Invalid reaches
   the LOAD/STORE and folds there. this needs to be before symbolic so that
   0*something_that_might_be_invalid doesnt become 0 *)
let invalid_pat = Upat.op Op.Const ~arg:(Const `Invalid) ~name:"i"
let invalid_gate = Upat.(where (var "cond") (var "x") invalid_pat)

(* the two const spellings: Invalid carries no width, so it rides bare inside
   either *)
let bare_const = Upat.(any [ op Op.Const; op Op.Stack ~each:(op Op.Const) ])

let casted_const =
  let p = Upat.(op Op.Cast ~src:[ op Op.Const ]) in
  Upat.(
    any [ p; op Op.Stack ~each:(any [ p; op Op.Const ~arg:(Const `Invalid) ]) ])

(* a REDUCE moves inside the gate clauses without its ranges: they invalidate
   every lane at once, so that gate lifts out. The clauses with its ranges
   invalidate single lanes, and an Invalid lane is the reduce's identity *)
let lift_reduce_gate red cond x i =
  match arg red with
  | Reduce { num_axes = 0; op = rop } -> (
      let ranges = List.tl (src red) in
      let in_reduce c =
        let crs = Ops.ranges c in
        List.exists
          (fun r ->
            List.exists
              (fun rr -> Nodes.mem rr crs)
              (Nodes.to_list (Ops.ranges r)))
          ranges
      in
      let keep, lift = List.partition in_reduce (split_uop cond Op.And) in
      let identity () = const_like x (identity_element rop (dtype red)) in
      let inner =
        match keep with [] -> x | _ -> where (conj keep) x (identity ())
      in
      let ret = replace red ~src:(inner :: ranges) in
      match lift with [] -> Some ret | _ -> Some (where (conj lift) ret i))
  | _ -> None

let unary_or_cast = Op.Set.union Op.Set.unary (ops [ Op.Cast; Op.Bitcast ])

let pm_data_invalid =
  pm
    (fun () -> [
      (* a stack of Invalid lanes is one Invalid, expanded to its width *)
      rule (Upat.broadcast ~name:"s" invalid_pat) (fun m ->
          match src (m "s") with
          | [] -> None
          | i :: _ -> Some (expand i (shape (m "s"))));
      rule (Upat.v ~op:unary_or_cast ~src:[ invalid_pat ] ()) (fun m ->
          Some (m "i"));
      rule (Upat.v ~op:unary_or_cast ~src:[ invalid_gate ] ~name:"op" ())
        (fun m ->
          Some (where (m "cond") (replace (m "op") ~src:[ m "x" ]) (m "i")));
      (* binary ops move inside the gate, with Invalid in the false branch *)
      rule
        (Upat.v ~op:Op.Set.binary
           ~src:[ invalid_gate; Upat.var "y" ]
           ~name:"alu" ())
        (fun m ->
          Some (where (m "cond") (alu (m "x") (op (m "alu")) [ m "y" ]) (m "i")));
      rule
        (Upat.v ~op:Op.Set.binary
           ~src:[ Upat.var "y"; invalid_gate ]
           ~name:"alu" ())
        (fun m ->
          Some (where (m "cond") (alu (m "y") (op (m "alu")) [ m "x" ]) (m "i")));
      rule
        (Upat.v
           ~op:(Op.Set.diff Op.Set.binary Op.Set.comparison)
           ~perm:[ invalid_pat; Upat.wild ] ())
        (fun m -> Some (m "i"));
      (* a multiply-add moves inside the gate of each operand in turn, and an
         Invalid operand makes it Invalid, as binary ops do *)
      rule (Upat.v ~op:(ops [ Op.Mulacc ]) ~name:"alu" ()) (fun m ->
          let a = m "alu" in
          let gated s = op s = Op.Where && is_invalid (nth s 2) in
          match List.find_opt (fun s -> is_invalid s || gated s) (src a) with
          | None -> None
          | Some s when is_invalid s -> Some s
          | Some g ->
              let inner s = if s == g then nth g 1 else s in
              Some
                (where (nth g 0)
                   (replace a ~src:(List.map inner (src a)))
                   (nth g 2)));
      rule (Upat.reduce ~name:"red" ~allow_any_len:true invalid_gate [])
        (fun m -> lift_reduce_gate (m "red") (m "cond") (m "x") (m "i"));
      (* an Invalid condition poisons the whole where; a gated Invalid condition
         lifts the gate out *)
      rule Upat.(where invalid_pat wild wild) (fun m -> Some (m "i"));
      rule
        Upat.(where invalid_gate (var "a") (var "b"))
        (fun m ->
          Some (where (m "cond") (where (m "x") (m "a") (m "b")) (m "i")));
      (* normalize where(cond, Invalid, val) -> where(~cond, val, Invalid) *)
      rule
        Upat.(where (var "cond") invalid_pat (var "val"))
        (fun m ->
          let v = m "val" and i = m "i" in
          Some (if is_invalid v then i else where (logical_not (m "cond")) v i));
      (* lift Invalid out: a.where(cond.where(x, Invalid), c) ->
         (~a|cond).where(a.where(x, c), Invalid) *)
      rule
        Upat.(where (var "a") invalid_gate (var "c"))
        (fun m ->
          let a = m "a" and c = m "c" in
          if is_invalid c then None
          else
            Some
              (where O.(logical_not a lor m "cond") (where a (m "x") c) (m "i")));
      rule
        Upat.(where (var "a") (var "b") invalid_gate)
        (fun m ->
          let a = m "a" and b = m "b" in
          if is_invalid b then None
          else Some (where O.(a lor m "cond") (where a b (m "x")) (m "i")));
      (* fold gated LOAD/STORE *)
      rule
        (Upat.op Op.Store ~name:"x"
           ~src:
             [
               Upat.or_casted
                 (Upat.index ~allow_any_len:true Upat.wild [ invalid_pat ]);
               Upat.wild;
             ])
        (fun m -> Some (after (v Op.Noop) (src (m "x"))));
      rule
        (Upat.op Op.Load ~allow_any_len:true ~name:"x"
           ~src:
             [
               Upat.or_casted
                 (Upat.index ~allow_any_len:true Upat.wild [ invalid_pat ]);
             ])
        (fun m ->
          let x = m "x" in
          Some (match src x with _ :: alt :: _ -> alt | _ -> const_v x zero));
    ])

let pm_remove_invalid =
  pm
    (fun () -> [
      rule (Upat.named "w" invalid_gate) (fun m ->
          let w = m "w" in
          Some (replace w ~src:[ m "cond"; m "x"; const_v w zero ]));
      rule (Upat.op Op.Stack ~name:"s") (fun m ->
          let s = m "s" in
          if not (List.exists is_invalid (src s)) then None
          else
            let zero_invalid x =
              if is_invalid x then const ~dtype:(dtype s) (`Int Bigint.zero) else x
            in
            Some (replace s ~src:(List.map zero_invalid (src s))));
    ])

(* [broadcast_const u] is the constant [u] is through movements that keep each
   element's value, as a constant shaped like another node is ({!const_like}). *)
let rec broadcast_const u =
  match op u with
  | Op.Const -> Some u
  | Op.Reshape | Op.Expand | Op.Permute | Op.Shrink ->
      broadcast_const (nth u 0)
  | _ -> None

(* folding a strong dtype WHERE to a weak const branch keeps the strong dtype *)
let fold_const_where gate c0 c1 w =
  let ret = if V.to_bool (num gate) then c0 else c1 in
  let weak u = Dtype.mem (dtype u) Dtype.weaks in
  if is_const ret && weak ret && not (weak w) then ccast ret (dtype w) else ret

(* A scalar computed from values ordered after effects is computed from the
   values, after the effects: a fold of the computation, such as [x * 0] to
   [0], would drop them. A selection of an invalid value stays. *)
let lift_value_after x =
  let afters = List.filter (fun s -> op s = Op.After) (src x) in
  let value a =
    Op.Set.mem (op (nth a 0)) Op.Set.alu || addrspace (nth a 0) = Some Dtype.Alu
  in
  if shape x <> [] || is_invalid (List.nth (src x) (List.length (src x) - 1))
  then None
  else if not (List.for_all value afters) then None
  else
    let unwrap s = if op s = Op.After then nth s 0 else s in
    let deps = dedup_nodes (List.concat_map (fun a -> List.tl (src a)) afters) in
    Some (after (replace x ~src:(List.map unwrap (src x))) deps)

let boolean = [ Dtype.Bool ]
let int_like = Dtype.Weak_int :: Dtype.ints
let int_or_bool = Dtype.Bool :: int_like

let symbolic_simple =
  Pattern_matcher.concat
    [
      pm_data_invalid;
      pm
        (fun () -> [
          rule
            (Upat.v
               ~op:(Op.Set.union Op.Set.alu (ops [ Op.Cast ]))
               ~early_reject:[ Op.After ] ~name:"x" ())
            (fun m -> lift_value_after (m "x"));
          (* Self folding *)
          (* a float x + 0 is x only for -0., since -0. + +0. is +0. *)
          rule
            (Upat.v
               ~op:(ops [ Op.Add; Op.Xor; Op.Or ])
               ~perm:Upat.[ var "x"; named "c" (int 0) ]
               ~name:"a" ())
            (fun m ->
              let a = m "a" in
              let negative_zero =
                match value (m "c") with
                | `Float z -> Float.sign_bit z
                | _ -> false
              in
              if op a = Op.Add && Dtype.is_float (dtype a) && not negative_zero
              then None
              else Some (m "x"));
          rule
            (Upat.v
               ~op:(ops [ Op.Shl; Op.Shr ])
               ~src:Upat.[ var "x"; int 0 ]
               ())
            (fun m -> Some (m "x"));
          rule Upat.(var "x" * int 1) (fun m -> Some (m "x"));
          (* x // x is 1 but at 0, where it is 0 *)
          rule
            Upat.(var "x" // var "x")
            (fun m ->
              let x = m "x" in
              if V.(vmin x <= zero && zero <= vmax x) then None
              else Some (const_v x one));
          rule Upat.(var "x" // int 1) (fun m -> Some (m "x"));
          rule
            Upat.(var ~dtype:(Dtype.Weak_int :: Dtype.sints) "x" // int (-1))
            (fun m -> Some (neg (m "x")));
          (* (x^y)^y -> x, for a weak y: a typed y would commit a weak x *)
          rule
            Upat.(var "x" lxor var ~dtype:[ Dtype.Weak_int ] "y" lxor var "y")
            (fun m -> Some (m "x"));
          (* (x%y)%y = -> x%y (rewritten with base for speed) *)
          rule
            Upat.(named "base" (wild % var "y") % var "y")
            (fun m -> Some (m "base"));
          (* variations of (x%c)+(x//c)*c = x *)
          rule (Upat.op Op.Add ~dtype:[ Dtype.Weak_int ] ~name:"x") (fun m ->
              fold_add_divmod_recombine (m "x"));
          rule
            Upat.(var ~dtype:boolean "x" land cvar "c")
            (fun m -> Some (if V.to_bool (num (m "c")) then m "x" else m "c"));
          rule
            Upat.(var ~dtype:boolean "x" lor cvar "c")
            (fun m -> Some (if V.to_bool (num (m "c")) then m "c" else m "x"));
          rule
            Upat.(
              eq (var ~dtype:boolean "x") (const ~dtype:boolean (`Bool true)))
            (fun m -> Some (m "x"));
          rule
            (Upat.v ~op:Op.Set.idempotent ~src:Upat.[ var "x"; var "x" ] ())
            (fun m -> Some (m "x"));
          rule
            Upat.(logical_not (logical_not (var ~dtype:boolean "x")))
            (fun m -> Some (m "x"));
          rule
            Upat.(
              where (var ~dtype:boolean "x")
                (const ~dtype:boolean (`Bool true))
                (const ~dtype:boolean (`Bool false)))
            (fun m -> Some (m "x"));
          rule
            Upat.(
              where (var ~dtype:boolean "x")
                (const ~dtype:boolean (`Bool false))
                (const ~dtype:boolean (`Bool true)))
            (fun m -> Some (logical_not (m "x")));
          (* CAST(bool -> int) == const — CAST(True)=1, CAST(False)=0, so fold
             based on const value *)
          rule
            Upat.(
              eq
                (f ~dtype:int_like (var ~dtype:boolean "x") Op.Cast)
                (cvar "c"))
            (fun m ->
              let x = m "x" and c = m "c" in
              Some
                (if equals c zero then logical_not x
                 else if equals c one then x
                 else const_like x (`Bool false)));
          rule Upat.(trunc (var ~dtype:int_or_bool "x")) (fun m -> Some (m "x"));
          (* Zero folding *)
          rule
            Upat.(var "x" < var "x")
            (fun m -> Some (const_like ~dtype:Dtype.Bool (m "x") (`Bool false)));
          rule Upat.(var "x" % var "x") (fun m -> Some (const_v (m "x") zero));
          rule
            Upat.(var "x" lxor var "x")
            (fun m -> Some (const_v (m "x") zero));
          rule Upat.(var "x" land int 0) (fun m -> Some (const_v (m "x") zero));
          (* (x&mask)>>k -> x>>k when mask only clears bits below k: all ones
             at x's type *)
          rule
            Upat.((var "x" land cvar "mask") lsr cvar "k")
            (fun m ->
              let mask = V.to_z (num (m "mask"))
              and k = V.to_int (num (m "k")) in
              if
                (k >= 0)
                [@mutate off "a shift by 0 is the x >> 0 rule's, tried first"]
                && all_ones (m "x") Bigint.(logor mask (pred (shift_left one k)))
              then Some (shr (m "x") (lit (`Int (Bigint.of_int k))))
              else None);
          (* x == x -> True (only ints) *)
          rule
            Upat.(eq (var ~dtype:int_or_bool "x") (var "x"))
            (fun m -> Some (const_like ~dtype:Dtype.Bool (m "x") (`Bool true)));
          (* Constant folding *)
          (* canonicalize casted CONST *)
          rule
            (Upat.op Op.Cast ~dtype:Dtype.all ~name:"root"
               ~src:[ Upat.cvar "c" ])
            (fun m ->
              let root = m "root" in
              match value (m "c") with
              | #Dtype.value as v when not (convertible (dtype root) v) -> None
              | c -> Some (const_like root c));
          (* collapse committed const conversions, the inner one read at its
             width *)
          rule
            (Upat.op Op.Cast ~dtype:Dtype.all ~name:"root"
               ~src:
                 [
                   Upat.op Op.Cast ~dtype:Dtype.all ~name:"inner"
                     ~src:[ Upat.op Op.Const ];
                 ])
            (fun m ->
              let root = m "root" in
              let v = at (dtype (m "inner")) (num (m "inner")) in
              if convertible (dtype root) v then Some (const_v root v) else None);
          (* one rule per spelling: bare has no width, a pair evaluates at its
             stated width, mixed commits to the promotion. THREEFRY(const,const)
             folds via its decomposition *)
          rule
            (Upat.v
               ~op:(Op.Set.diff Op.Set.alu (ops [ Op.Threefry ]))
               ~each:bare_const ~name:"a" ())
            (fun m -> fold_const_alu (m "a"));
          rule
            (Upat.v
               ~op:(Op.Set.diff Op.Set.alu (ops [ Op.Threefry ]))
               ~each:casted_const ~name:"a" ())
            (fun m -> fold_const_alu (m "a"));
          rule
            (Upat.v
               ~op:(Op.Set.diff Op.Set.binary (ops [ Op.Threefry ]))
               ~perm:[ casted_const; bare_const ]
               ~name:"a" ())
            (fun m ->
              let a = m "a" in
              let dt = promo_dtype (src a) in
              let commit s =
                if Dtype.mem (dtype s) Dtype.weaks then ccast s dt else s
              in
              if Dtype.mem dt Dtype.weaks then None
              else Some (replace a ~src:(List.map commit (src a))));
        ]
        (* Two stage ALU folding; sums, products and maxima for integers: in
           floats each step rounds, and a maximum keeps a NaN only as its first
           operand *)
        @ List.map
            (fun o ->
              let dtype =
                if o = Op.Add || o = Op.Mul || o = Op.Max then Some int_or_bool
                else None
              in
              let x = Upat.var ?dtype "x" in
              rule
                Upat.(named "f" (alu (alu x o [ cvar "c1" ]) o [ cvar "c2" ]))
                (fun m ->
                  let f = m "f" in
                  let o = op f in
                  (* a sum, a product and a bitwise operation fold the same
                     before or after wrapping; a maximum orders wrapped values:
                     on uint8, max(1, -3) is 253, so its weak constants fold at
                     its width *)
                  let at_width c =
                    if o = Op.Max && Dtype.mem (Ops.dtype c) Dtype.weaks then
                      ccast c (Ops.dtype f)
                    else c
                  in
                  let const c = (vmin (m c) :> Dtype.const) in
                  let folded = [ const "c1"; const "c2" ] in
                  let fit =
                    match exec_alu o Dtype.Weak_int folded with
                    | #Dtype.value as v -> fits (m "x") v
                    | _ -> true
                  in
                  let c1 = at_width (m "c1") and c2 = at_width (m "c2") in
                  if fit then Some (alu (m "x") o [ wrap_folded f c1 c2 ])
                  else None))
            (Op.Set.to_list Op.Set.associative)
        @ [
          (* bool MUL is AND, ADD/MAX is OR. prevents other rules to rewrite
             bool ADD/MUL incorrectly *)
          rule
            Upat.(var ~dtype:boolean "x" * var ~dtype:boolean "y")
            (fun m -> Some O.(m "x" land m "y"));
          rule
            Upat.(var ~dtype:boolean "x" + var ~dtype:boolean "y")
            (fun m -> Some O.(m "x" lor m "y"));
          rule
            Upat.(maximum (var ~dtype:boolean "x") (var ~dtype:boolean "y"))
            (fun m -> Some O.(m "x" lor m "y"));
          (* Div rules *)
          rule
            Upat.(cvar ~arg:(`Int Bigint.zero) "x" / int 0)
            (fun m -> Some (const_like (m "x") (`Float Dtype.nan)));
          (* x*0 -> 0 or 0*x -> 0, for integers: a float product by zero is NaN
             at an infinity or a NaN, and -0. at a negative x *)
          rule
            Upat.(var ~dtype:int_or_bool "x" * int 0)
            (fun m -> Some (const_v (m "x") zero));
          (* Cast/bitcast *)
          rule
            (Upat.v ~op:(ops [ Op.Cast; Op.Bitcast ]) ~name:"root" ())
            (fun m ->
              let root = m "root" in
              if Dtype.equal (dtype root) (dtype (nth root 0)) then
                Some (nth root 0)
              else None);
          (* a BITCAST reads its operand at the width it states, so a weak const
             is nonsense here: the bare arm is bool only *)
          rule
            (Upat.op Op.Bitcast ~name:"root"
               ~src:
                 [
                   Upat.(
                     any
                       [
                         op Op.Const ~dtype:boolean ~name:"c";
                         op Op.Cast ~src:[ op Op.Const ] ~name:"c";
                       ]);
                 ])
            (fun m -> fold_bitcast (m "root") (m "c"));
          (* b.cast(a).cast(b) -> b if a preserves all values in b *)
          rule
            Upat.(f ~name:"b" (f ~name:"a" (var "x") Op.Cast) Op.Cast)
            (fun m ->
              let x = m "x" and b = dtype (m "b") in
              if
                Dtype.equal (dtype x) b
                && Dtype.can_lossless_cast b (dtype (m "a"))
              then Some x
              else None);
          (* bitcast twice *)
          rule
            (Upat.op Op.Bitcast ~name:"b" ~src:[ Upat.bitcast (Upat.var "x") ])
            (fun m -> Some (bitcast (m "x") (dtype (m "b"))));
          rule
            Upat.(cast (var "x") Dtype.Bool)
            (fun m -> Some O.(m "x" <> int 0));
          (* Pow *)
          rule
            Upat.(alu (var "x") Op.Pow [ cvar "c" ])
            (fun m -> simplify_pow (m "x") (m "c"));
          (* positive const ** x, and 1 ** x, which is 1 of the power's type at
             every x, NaN included *)
          rule
            Upat.(named "p" (alu (cvar "c") Op.Pow [ var "x" ]))
            (fun m ->
              let cv = num (m "c") in
              if V.(cv = one) then Some (const_v (m "p") one)
              else if
                V.(cv > zero && cv < `Float Float.infinity)
                && Dtype.is_float (dtype (m "x"))
              then
                let log2c =
                  exec_alu ~truncate_output:false Op.Log2 Float64
                    [ (cv :> Dtype.const) ]
                in
                Some (exp2 O.(m "x" * float (V.to_float (as_value log2c))))
              else None);
          (* unpack a uint64 packed from two uint32 (threefry) *)
          rule
            Upat.(
              cast
                ((v ~dtype:[ Dtype.Uint64 ] () lsl int 32)
                lor cast (var ~dtype:[ Dtype.Uint32 ] "y") Dtype.Uint64)
                Dtype.Uint32)
            (fun m -> Some (m "y"));
          rule
            Upat.(
              ((var ~dtype:[ Dtype.Uint64 ] "x" lsl int 32)
              lor cast (v ~dtype:[ Dtype.Uint32 ] ()) Dtype.Uint64)
              lsr int 32)
            (fun m -> Some (cast (cast (m "x") Dtype.Uint32) Dtype.Uint64));
          (* Simple where folding *)
          (* a conditional with the same results either way is a noop, also fold
             const conditionals, broadcast ones included *)
          rule
            Upat.(where wild (var "val") (var "val"))
            (fun m -> Some (m "val"));
          rule
            Upat.(named "w" (where (var "gate") (var "c0") (var "c1")))
            (fun m ->
              Option.map
                (fun gate -> fold_const_where gate (m "c0") (m "c1") (m "w"))
                (broadcast_const (m "gate")));
        ]);
      mop_cleanup;
    ]

(* Phase 2: rules that match deeper *)

let lt_folding x c =
  let p, np =
    List.partition
      (fun (_, f) -> Bigint.equal f Bigint.one)
      (List.map
         (fun u ->
           let u, f = pop_const ~op:Op.Mul u in
           (u, V.to_z (as_value f)))
         (split_uop x Op.Add))
  in
  let d = List.fold_left (fun d (_, f) -> Bigint.gcd d f) Bigint.zero np in
  (* d*y+p < c <=> y < ceildiv(c-p, d), so it folds when that bound is the same
     for p's least and greatest *)
  let bound f =
    let s = List.fold_left (fun s (u, _) -> V.(s + f u)) zero p in
    Bigint.cdiv (Bigint.sub c (V.to_z s)) d
  in
  match np with
  | (n, f) :: ns when Bigint.gt d Bigint.one ->
      let t = bound vmin in
      if Bigint.equal t (bound vmax) then
        let term (u, f) = O.(u * lit (`Int (Bigint.fdiv f d))) in
        Some O.(usum (term (n, f)) (List.map term ns) < lit (`Int t))
      else None
  | _ -> None

(* (X := a0*x0 + a1*x1 + ...) > 0 is equivalent to x0 + x1 + ... > 0 if xi >= 0
   and ai > 0 for ints. returns x0 + x1 + ... in such case, or None if not *)
let canonicalize_simplex x =
  (* assumed the const is the last src of MUL *)
  let strip u =
    if op u = Op.Mul && is_const (nth u 1) && V.(num (nth u 1) > zero) then
      (true, nth u 0)
    else (false, u)
  in
  let terms = List.map strip (split_uop x Op.Add) in
  let atom (_, u) =
    Op.Set.mem (op u) Op.Set.irreducible && V.(vmin u >= zero)
  in
  if List.for_all atom terms && List.exists fst terms then
    Some (sum_of (List.map snd terms))
  else None

let commutative =
  pm
    (fun () -> [
      (* COMMUTATIVE flipping (only for index) *)
      (* NOTE: this can break merging vector math by only flipping some of them *)
      rule
        (Upat.v ~op:Op.Set.commutative ~dtype:[ Dtype.Weak_int ] ~name:"x" ())
        (fun m ->
          let x = m "x" in
          if compare_structure (nth x 1) (nth x 0) < 0 then
            Some (replace x ~src:(List.rev (src x)))
          else None);
    ])

(* The operand C leaves [u] undefined for some values of: a float converted to
   an integer, outside the integer's range. *)
let partial_operand u =
  match op u with
  | Op.Cast when Dtype.is_int (dtype u) && Dtype.is_float (dtype (nth u 0)) ->
      Some (nth u 0)
  | _ -> None

let partial_ops = Op.[ Cast ]

(* Whether [u] holds an undefined-prone operation whose operand reads [cond]: a
   selection by [cond] may be its guard. *)
let guards_partial cond u =
  op_in_backward_slice_with_self ~calls:Skip u partial_ops
  && List.exists
       (fun n ->
         match partial_operand n with
         | Some o -> reaches ~calls:Skip o cond
         | None -> false)
       (toposort ~calls:Skip u)

(* in cond.where(t, f), cond is True within t and False within f *)
let fold_where_closure cond t f =
  if not (Dtype.equal (dtype cond) Dtype.Bool) then None
  else if
    (* a constant condition, broadcast or not, assumes nothing: the same node
       is every other use of that constant *)
    is_const (base cond)
  then None
  else if
    (* INDEX gates are owned by the valid/store-coalescing machinery, leave them
       alone, and so are loads: a load of several elements is addressed by a
       shrink of its buffer, valid included, where one element's address is an
       INDEX, and a register's read by the register alone. Asked before the
       search below, which walks the branches. *)
    List.exists
      (fun u ->
        op_in_backward_slice_with_self ~calls:Skip u [ Op.Index; Op.Load ])
      [ cond; t; f ]
  then None
  else if not (reaches ~calls:Enter t cond || reaches ~calls:Enter f cond)
  then None
  else if
    (* a conversion that the selection keeps from an operand it is undefined
       for stays guarded: C computes it in every lane of a vector, and a
       selection of vectors keeps an undefined lane it discards *)
    guards_partial cond t || guards_partial cond f
  then None
  else
    let assume b u =
      substitute ~calls:Skip ~pass:Fixed_point u
        [ (cond, const_like cond (`Bool b)) ]
    in
    Some (where cond (assume true t) (assume false f))

let both_const u0 u1 = is_const u0 && is_const u1

let symbolic =
  Pattern_matcher.concat
    [
      symbolic_simple;
      commutative;
      pm
        (fun () -> [
           (* Boolean algebra *)
           rule
             Upat.(
               var ~dtype:boolean "x" lor logical_not (var ~dtype:boolean "x"))
             (fun m -> Some (const_like (m "x") (`Bool true)));
           (* Combine terms *)
           (* like terms combine for integers: in floats each product and
              sum rounds *)
           rule
             Upat.(
               (var ~dtype:int_like "x" * cvar "c0") + (var "x" * cvar "c1"))
             (fun m ->
               if fits (m "x") (like_sum m) then
                 Some O.(m "x" * (m "c0" + m "c1"))
               else None);
           rule
             Upat.(
               var "y"
               + (var ~dtype:int_like "x" * cvar "c0")
               + (var "x" * cvar "c1"))
             (fun m ->
               if fits (m "x") (like_sum m) then
                 Some O.(m "y" + (m "x" * (m "c0" + m "c1")))
               else None);
           rule
             Upat.(var ~dtype:int_like "x" + (var "x" * cvar "c"))
             (fun m ->
               if fits (m "x") (like_succ m) then
                 Some O.(m "x" * (m "c" + int 1))
               else None);
           rule
             Upat.(var "y" + var ~dtype:int_like "x" + (var "x" * cvar "c"))
             (fun m ->
               if fits (m "x") (like_succ m) then
                 Some O.(m "y" + (m "x" * (m "c" + int 1)))
               else None);
           rule
             Upat.(var "y" + (var ~dtype:int_like "x" * cvar "c") + var "x")
             (fun m ->
               if fits (m "x") (like_succ m) then
                 Some O.(m "y" + (m "x" * (m "c" + int 1)))
               else None);
           rule Upat.(var "x" + var "x") (fun m -> Some O.(m "x" * int 2));
           rule
             Upat.(named "a" (var "y" + var ~dtype:int_like "x" + var "x"))
             (fun m ->
               let a = m "a" and x = m "x" in
               let two v = V.(of_int 2 * v) in
               if fits a (two (vmin x)) && fits a (two (vmax x)) then
                 Some O.(m "y" + (x * int 2))
               else None);
           rule
             Upat.(cvar "y" * (var ~dtype:[ Dtype.Weak_int ] "x" + cvar "c"))
             (fun m ->
               let y = m "y" in
               Some O.((y * m "x") + (y * m "c")));
           (* Where folding *)
           rule
             Upat.(
               where
                 (logical_not (var ~dtype:boolean "cond"))
                 (var "t") (var "f"))
             (fun m ->
               let f = m "f" in
               if is_invalid f then None else Some (where (m "cond") f (m "t")));
           (* in cond.where(t, f), uses of cond fold to True within t and False
              within f *)
           rule
             Upat.(where (var ~dtype:boolean "cond") (var "t") (var "f"))
             (fun m -> fold_where_closure (m "cond") (m "t") (m "f"));
           rule
             Upat.(where (var "gate") (var "x") (int 0) <> int 0)
             (fun m -> Some O.(m "gate" land (m "x" <> int 0)));
           (* a.where(b.where(c, d), d) -> (a & b).where(c, d) *)
           rule
             Upat.(
               where (var "a") (where (var "b") (var "c") (var "d")) (var "d"))
             (fun m -> Some (where O.(m "a" land m "b") (m "c") (m "d")));
           (* a.where(b.where(c, d), c) -> (a & !b).where(d, c) *)
           rule
             Upat.(
               where (var "a") (where (var "b") (var "c") (var "d")) (var "c"))
             (fun m ->
               Some
                 (where O.(m "a" land logical_not (m "b")) (m "d") (m "c")));
           (* a.where(c, b.where(c, d)) -> (a | b).where(c, d) *)
           rule
             Upat.(
               where (var "a") (var "c") (where (var "b") (var "c") (var "d")))
             (fun m -> Some (where O.(m "a" lor m "b") (m "c") (m "d")));
           (* alu of two where with same conds can combine, only do if true
              branch or false branch is const *)
           rule
             (Upat.v ~op:Op.Set.binary ~name:"alu"
                ~src:
                  Upat.
                    [
                      where (var "c") (var "t") (var "f");
                      where (var "c") (var "tt") (var "ff");
                    ]
                ())
             (fun m ->
               let o = op (m "alu")
               and t = m "t"
               and tt = m "tt"
               and f = m "f"
               and ff = m "ff" in
               if both_const t tt || both_const f ff then
                 Some (where (m "c") (alu t o [ tt ]) (alu f o [ ff ]))
               else None);
           (* if its a plus we add the associative variation too, for integers:
              it reassociates the sum *)
           rule
             Upat.(
               var ~dtype:int_like "y"
               + where (var "c") (var "t") (var "f")
               + where (var "c") (var "tt") (var "ff"))
             (fun m ->
               let t = m "t" and tt = m "tt" and f = m "f" and ff = m "ff" in
               if both_const t tt || both_const f ff then
                 Some O.(m "y" + where (m "c") (t + tt) (f + ff))
               else None);
           (* complementary zero branches under the same condition select
              directly, for integers: a float t + 0 is +0. at t = -0. *)
           rule
             Upat.(
               where (var "c") (var ~dtype:int_or_bool "t") (int 0)
               + where (var "c") (int 0) (var "f"))
             (fun m -> Some (where (m "c") (m "t") (m "f")));
           (* ALU/variable min==max -> CONST *)
           rule
             (Upat.v
                ~op:
                  (ops
                     [
                       Op.Cmplt;
                       Op.Cmpeq;
                       Op.Div;
                       Op.Mod;
                       Op.Param;
                       Op.Special;
                     ])
                ~name:"x" ())
             (fun m ->
               let x = m "x" in
               if
                 (not (Dtype.equal (dtype x) Dtype.Void))
                 && V.(vmin x = vmax x)
               then Some (const_v x (vmin x))
               else None);
           rule
             (Upat.op Op.Range
                ~src:[ Upat.or_casted (Upat.op Op.Const) ]
                ~name:"x")
             (fun m ->
               let x = m "x" in
               if V.(vmin x = vmax x) then Some (const_v x (vmin x)) else None);
           (* max folding, for integers: a float selection keeps IEEE's NaN and
              signed zeros where a maximum does not *)
           rule
             Upat.(
               where
                 (cvar "a" < var ~dtype:int_or_bool "b")
                 (var "b") (cvar "c"))
             (fun m ->
               if V.(num (m "a") = num (m "c")) then
                 Some (maximum (m "a") (m "b"))
               else None);
           (* where (a < b) c (a + (c - b)) is the maximum of a + (c - b) and
              c, for a signed a whose shifted arm does not wrap *)
           rule
             Upat.(
               where
                 (var ~dtype:int_or_bool "a" < cvar "b")
                 (cvar "c") (var "y"))
             (fun m ->
               let a = m "a" and y = m "y" and c = m "c" in
               let d = V.(num c - num (m "b")) in
               if
                 (y == a && V.(d = zero))
                 || Dtype.is_int (dtype a)
                    && (not (Dtype.is_unsigned (dtype a)))
                    && exact (dtype a) V.[ d; vmin a + d; vmax a + d ]
                    && y == O.(a + lit d)
               then Some (maximum y c)
               else None);
           (* a float maximum's bounds leave out NaN and the order of zeros *)
           rule
             Upat.(named "m" (maximum (var ~dtype:int_or_bool "x") (var "y")))
             (fun m ->
               let mx = m "m" and x = m "x" in
               let lo = fst (operand_bounds mx x) in
               if V.(lo < snd (operand_bounds mx (m "y"))) then None
               (* the operand kept is committed to the maximum's type *)
               else if Dtype.mem (dtype x) Dtype.weaks then
                 Some (ccast x (dtype mx))
               else Some x);
           (* (x//c1)//c2 -> x//(c1*c2) for c2>0, where c1*c2 does not wrap *)
           rule
             Upat.(var "x" // cvar "c1" // cvar "c2")
             (fun m ->
               let c1 = vmin (m "c1") and c2 = vmin (m "c2") in
               if V.(c2 > zero) && exact (dtype (m "x")) V.[ c1; c2; c1 * c2 ]
               then Some O.(m "x" // (m "c1" * m "c2"))
               else None);
           (* Lt *)
           (* c0+x<c1 -> x < c1-c0 for signed x, where neither side wraps:
              an unsigned sum that wraps moves past c1 *)
           rule
             Upat.(cvar "c0" + var ~dtype:(Dtype.Weak_int :: Dtype.sints) "x" < cvar "c1")
             (fun m ->
               let x = m "x" and c0 = vmin (m "c0") and c1 = vmin (m "c1") in
               if
                 exact (dtype x)
                   V.[ c0; c1; vmin x + c0; vmax x + c0; c1 - c0 ]
               then Some O.(x < m "c1" - m "c0")
               else None);
           (* x//d<c -> x<c*d for d>0 *)
           rule
             Upat.(var ~dtype:[ Dtype.Weak_int ] "x" // cvar "d" < cvar "c")
             (fun m ->
               let d = num (m "d") in
               if V.(d > zero) then Some O.(m "x" < lit V.(num (m "c") * d))
               else None);
           (* Move add/mul consts to end (NOTE: this is still happening before
              constant folding), for integers: it reassociates *)
           rule
             Upat.(var ~dtype:int_like "x" + cvar "c1" + var "y")
             (fun m ->
               let y = m "y" in
               if is_const y then None else Some O.(m "x" + y + m "c1"));
           (* a float product by -1 is the negation, which commutes with the
              other product's rounding *)
           rule
             Upat.(var "x" * cvar "c1" * var "y")
             (fun m ->
               let x = m "x" and y = m "y" and c1 = m "c1" in
               if is_const y then None
               else if Dtype.mem (dtype x) int_like || V.(num c1 = of_int (-1))
               then Some O.(x * y * c1)
               else None);
           (* Rules from symbolic *)
           (* generic lt folding *)
           rule
             Upat.(var ~dtype:[ Dtype.Weak_int ] "x" < cvar "c")
             (fun m ->
               match num (m "c") with
               | `Int c -> lt_folding (m "x") c
               | _ -> None);
           (* -x < -y -> y < x, and -x < c -> -c < x, where no negation
              wraps at the width the comparison computes at: at int8,
              -(-128) is -128. A weak constant is read at that width. *)
           rule
             Upat.(
               named "cmp"
                 (var ~dtype:(Dtype.Weak_int :: Dtype.sints) "x" * int (-1)
                 < var "y" * int (-1)))
             (fun m ->
               let x = m "x" and y = m "y" in
               let dt = promo_dtype (src (m "cmp")) in
               if exact dt V.[ -vmin x; -vmin y ] then Some O.(y < x) else None);
           rule
             Upat.(
               named "cmp"
                 (var ~dtype:(Dtype.Weak_int :: Dtype.sints) "x" * int (-1)
                 < cvar "c"))
             (fun m ->
               let x = m "x" and c = m "c" in
               let dt = promo_dtype (src (m "cmp")) in
               let c_at_width =
                 if Dtype.mem dt Dtype.ints then Dtype.truncate dt (num c)
                 else num c
               in
               if V.(c_at_width <> Dtype.min dt) && exact dt V.[ -vmin x ] then
                 Some O.(neg c < x)
               else None);
           (* canonicalize a simplex with positive coefficients > 0. NOTE: not
              x < 1 means x > 0 *)
           rule
             Upat.(logical_not (var ~dtype:[ Dtype.Weak_int ] "x" < int 1))
             (fun m ->
               Option.map
                 (fun x -> logical_not O.(x < int 1))
                 (canonicalize_simplex (m "x")));
           (* a range mod its own upper bound is just the range *)
           rule
             Upat.(op Op.Range ~each:(var "end") ~name:"r" % var "end")
             (fun m -> Some (m "r"));
           rule
             Upat.(op Op.Range ~each:(var "end") ~name:"r" // var "end")
             (fun m -> Some (const_v (m "r") zero));
           (* cast/long folding *)
           (* if the intermediate cast doesnt narrow we can do it in one cast *)
           rule
             Upat.(f ~name:"b" (f ~name:"a" (var "x") Op.Cast) Op.Cast)
             (fun m ->
               let x = m "x" in
               if Dtype.can_lossless_cast (dtype x) (dtype (m "a")) then
                 Some (cast x (dtype (m "b")))
               else None);
           rule
             Upat.(
               f ~name:"b"
                 (f ~dtype:int_like ~name:"a" (var ~dtype:int_like "x") Op.Cast)
                 Op.Cast)
             (fun m ->
               let x = m "x" in
               if overflows x (dtype (m "a")) then None
               else Some (ccast x (dtype (m "b"))));
           (* try to do math in int instead of long, keep weak const weak *)
           rule
             (Upat.v ~op:Op.Set.binary ~name:"u"
                ~src:
                  Upat.
                    [
                      var ~dtype:[ Dtype.Int64; Dtype.Weak_int ] "x";
                      var ~dtype:[ Dtype.Int64; Dtype.Weak_int ] "y";
                    ]
                ())
             (fun m ->
               let u = m "u" and x = m "x" and y = m "y" in
               let narrow w =
                 if is_const w then lit (num w) else cast w Dtype.Int32
               in
               (* An int32 shift by 32 or more, and the least int32's remainder
                  by -1, have no value, where the int64 ones do. *)
               let undefined =
                 match op u with
                 | Op.Shr -> Value.(vmax y >= of_int 32)
                 | Op.Mod | Op.Truncmod ->
                     Value.(vmin x <= Dtype.min Dtype.Int32)
                     && Value.(vmin y <= of_int (-1) && of_int (-1) <= vmax y)
                 | _ -> false
               in
               if
                 List.exists
                   (fun w -> Dtype.equal (dtype w) Dtype.Int64)
                   [ x; y ]
                 && (not
                       (List.exists
                          (fun w -> overflows w Dtype.Int32)
                          [ u; x; y ]))
                 && not undefined
               then Some (cast (alu (narrow x) (op u) [ narrow y ]) (dtype u))
               else None);
           rule
             Upat.(
               f ~dtype:Dtype.sints ~name:"cast"
                 (var ~dtype:[ Dtype.Weak_int ] "x" + cvar "c")
                 Op.Cast)
             (fun m ->
               let c = m "cast" in
               Some O.(cast (m "x") (dtype c) + const_like c (value (m "c"))));
           (* an AFTER waits only on effects and the ops listed here, any
              other dep is replaced by its srcs. An AFTER of global storage
              it waits on stays: read through a join, it holds a state whose
              writers its reader waits for. *)
           rule (Upat.op Op.After ~name:"x") (fun m ->
               let x = m "x" in
               let effects =
                 Op.Set.union Op.Set.effectful
                   (ops
                      [
                        Op.Range;
                        Op.End;
                        Op.Backedge;
                        Op.Linear;
                        Op.Stage;
                        Op.Load;
                      ])
               in
               let state y =
                 op y = Op.After && addrspace y = Some Dtype.Global
               in
               (* a reduction waited on is its source, with its ranges ended *)
               let deps y =
                 if Op.Set.mem (op y) effects || state y then [ y ]
                 else if op y = Op.Reduce then
                   [ end_ (after (v Op.Noop) [ nth y 0 ]) (List.tl (src y)) ]
                 else src y
               in
               Some
                 (replace x
                    ~src:
                      (nth x 0
                      :: dedup_nodes
                           (List.concat_map deps (List.tl (src x))))));
           (* after/end with 1 src is just src[0] *)
           rule
             (Upat.v ~op:(ops [ Op.After; Op.End ]) ~src:[ Upat.var "s" ] ())
             (fun m -> Some (m "s"));
           (* a range that became a constant runs once, so an end drops it: a
              loop of one trip around a call is that call *)
           rule (Upat.op Op.End ~name:"x") (fun m ->
               let x = m "x" in
               Some
                 (replace x
                    ~src:
                      (nth x 0
                      :: List.filter
                           (fun r -> not (is_const r))
                           (List.tl (src x)))));
         ]);
      div_and_mod_symbolic;
      (* the rules above key on bare CONSTs, so a redundantly committed const
         has to be uncast in the same fixpoint *)
      pm_uncast_const;
    ]

let () = simplify_rules := symbolic

(* Storage *)

let empty ?device new_shape dt =
  weak_storage dt;
  let max_shape = to_max_shape new_shape in
  let u =
    v Op.Alloc
      ~src:[ size_src (Some (size_of max_shape)) ]
      ~arg:
        (Param
           (param_arg ~slot:(unique_num ()) ?device ~bind_on_realize:true dt))
  in
  shrink_to
    (reshape u (List.map (fun n -> Int n) max_shape))
    (List.map Option.some new_shape)

let view_as u new_shape =
  let max_shape = List.map (fun n -> Int n) (to_max_shape new_shape) in
  let ret = if new_shape = [] then u else reshape u max_shape in
  if (List.equal Sint.equal) max_shape new_shape then ret
  else shrink_to ret (List.map Option.some new_shape)

let empty_like ?dtype ?device:dev u =
  let dev = match dev with Some d -> Some d | None -> device u in
  let dtype = match dtype with Some dt -> dt | None -> commit_dtype u in
  empty ?device:dev (shape u) dtype

let clone ?device:dev u =
  let dev = match dev with Some d -> Some d | None -> device u in
  (match dev with
  | Some d when is_disk_device d ->
      invalid_arg "cannot clone a disk; store into a disk buffer instead"
  | _ -> ());
  let ret = empty_like ?device:dev u in
  let src =
    match (device u, dev) with
    | None, _ -> u
    | Some d, Some d' when String.equal d d' -> u
    | _, Some d' -> copy_to_device u d'
    | _, None -> u
  in
  after ret [ store ret (cast src (dtype ret)) ]

let alloc ?slot ?(addrspace = Dtype.Global) ?device ?vmin_vmax new_shape dt =
  let slot = match slot with Some s -> s | None -> unique_num () in
  let ret =
    v Op.Alloc
      ~src:[ size_src (Some (size_of (to_max_shape new_shape))) ]
      ~arg:
        (Param
           (param_arg ~slot ~addrspace ?device ?vmin_vmax
              (Dtype.strong dt)))
  in
  if List.is_empty new_shape then reshape ret []
  else view_as ret new_shape

let alloc_like ?slot ?addrspace u =
  alloc ?slot ?addrspace
    (List.map (fun n -> Int n) (max_shape u))
    (dtype u)

let placeholder ?slot ?(addrspace = Dtype.Global) new_shape dt =
  let dt = Dtype.strong dt in
  let slot = match slot with Some s -> s | None -> unique_num () in
  let src = [ size_src (Some (size_of new_shape)) ] in
  let ret =
    match addrspace with
    | Dtype.Global ->
        v Op.Param ~src ~arg:(Param (param_arg ~slot ~addrspace dt))
    | Dtype.Local | Dtype.Reg ->
        v Op.Buffer ~src
          ~arg:(Param (param_arg ~slot ~addrspace dt))
    | Dtype.Alu -> invalid_arg "a placeholder cannot be a scalar variable"
  in
  if List.length new_shape > 1 then
    reshape ret (List.map (fun n -> Int n) new_shape)
  else ret

let placeholder_like u slot =
  if not (List.for_all (function Int _ -> true | Sym _ -> false) (shape u))
  then invalid_arg "a placeholder needs a shape of known sizes";
  placeholder ~slot (max_shape u) (dtype u)

let param ?shape:new_shape ?device ?vmin_vmax ?name
    ?(addrspace = Dtype.Global) ?(volatile = false) ?phase ?align slot dt
    =
  weak_storage dt;
  let make size =
    v Op.Param
      ~src:[ size_src size ]
      ~arg:
        (Param
           (param_arg ?vmin_vmax ?name ~addrspace ?device
              ~volatile ?phase ?align ~slot dt))
  in
  match new_shape with
  | None | Some [] -> make None
  | Some s -> view_as (make (Some (size_of (to_max_shape s)))) s

(* Phases of storage *)

(* Where a stage's view of a buffer, through movements and bitcasts, starts if
   it can be a contiguous run of the buffer: the buffer's alignment and the byte
   its first element lies at, an index taken through each movement to its
   source's. A run's last element lies as many elements past its first as the
   view has elements less one; a view whose first or last element is padding,
   or whose ends lie otherwise, is no run. *)
type view_start =
  | Start of int * int
  | Unknown (* A size or a bound is symbolic. *)
  | No_view (* No buffer is viewed, or the view is no run of it. *)

let rec view_start u =
  let exception Stop of view_start in
  let int = function Int n -> n | Sym _ -> raise_notrace (Stop Unknown) in
  let dims u = List.map int (shape u) in
  let flat idx shape =
    List.fold_left2 (fun acc i n -> (acc * n) + i) 0 idx shape
  in
  let unflat k shape =
    fst
      (List.fold_right
         (fun n (idx, k) -> ((k mod n) :: idx, k / n))
         shape ([], k))
  in
  let rec go u idx =
    match (op u, src u) with
    | Op.Buffer, _ ->
        let p = param_arg_of u in
        (p.align, p.phase + (flat idx (dims u) * element_size u))
    | Op.Detach, x :: _ -> go x idx
    | Op.Bitcast, x :: _ ->
        let bytes = flat idx (dims u) * element_size u
        and size = element_size x in
        let align, start = go x (unflat (bytes / size) (dims x)) in
        (align, start + (bytes mod size))
    | o, x :: _ when Op.Set.mem o Op.Set.movement ->
        let src = dims x in
        let idx =
          match marg u with
          | Reshape _ -> unflat (flat idx (dims u)) src
          | Expand added -> List.drop (List.length added) idx
          | Shrink b -> List.map2 (fun i (off, _) -> i + int off) idx b
          | Pad b ->
              List.map2
                (fun (i, n) (off, _) ->
                  let j = i - int off in
                  if j < 0 || j >= n then raise_notrace (Stop No_view) else j)
                (List.combine idx src) b
          | Permute p ->
              let moved = Array.make (List.length p) 0 in
              List.iter2 (fun i a -> moved.(a) <- i) idx p;
              Array.to_list moved
        in
        go x idx
    | Op.Stage, x :: _ when view_start x <> No_view ->
        raise_notrace (Stop Unknown)
    | _ -> raise_notrace (Stop No_view)
  in
  match
    let shape = dims u in
    let n = List.fold_left ( * ) 1 shape in
    if n = 0 then raise_notrace (Stop No_view);
    let align, first = go u (List.map (fun _ -> 0) shape) in
    let _, last = go u (List.map (fun d -> d - 1) shape) in
    if last - first <> (n - 1) * element_size u then
      raise_notrace (Stop No_view);
    (align, first)
  with
  | align, start -> Start (align, start mod align)
  | exception Stop s -> s
  | exception Division_by_zero -> No_view

(* The alignment and phase of the storage [u] views: its storage's, moved by
   the bytes a shrink of storage seen whole skips, known modulo fewer bytes when
   the shrink's start is symbolic. Any other view keeps its storage's, as views
   are taken to start aligned, unless a symbolic start moves it; storage the
   graph allocates starts on a boundary. A stage of a view of a buffer is that
   view when the view is contiguous, which scheduling decides
   ([Schedule.contiguous_mops_to_view]), and storage of its own otherwise: its
   phase is what the two agree on. *)
let rec storage_phase u =
  let rec whole v =
    match (op v, src v) with
    | (Op.Buffer | Op.Param | Op.Alloc | Op.Stage), _ -> true
    | (Op.Bitcast | Op.Reshape | Op.After), v :: _ -> whole v
    | _ -> false
  in
  let ints l = List.for_all (function Int _ -> true | Sym _ -> false) l in
  match (op u, src u) with
  | _ when on_disk u -> (16, 0)
  | (Op.Buffer | Op.Param | Op.Alloc), _ ->
      let p = param_arg_of u in
      (p.align, p.phase)
  | Op.Shrink, x :: _ -> (
      let align, phase = storage_phase x in
      let bytes = element_size x in
      (* The start moved by [first] bytes and by multiples of [by] bytes: known
         modulo the largest power of two up to [align] that divides [by]. *)
      let moved by first =
        let rec known a =
          if a < align && Bigint.divisible by (Bigint.of_int (2 * a)) then known (2 * a)
          else a
        in
        let a = known 1 in
        (a, (((phase + first) mod a) + a) mod a)
      in
      match marg u with
      | Shrink bounds when whole x && ints (shape x) -> (
          (* The element the shrink starts at, by the row-major strides of [x]'s
             shape: a constant, and multiples of a constant that move. *)
          let offset =
            List.fold_left2
              (fun acc (start, _) size -> Sint.((acc * size) + start))
              (Int 0) bounds (shape x)
          in
          match offset with
          | Int n -> moved Bigint.zero (n * bytes)
          | Sym e ->
              let rest, c = pop_const (simplify e) in
              moved
                (Bigint.mul (const_factor rest) (Bigint.of_int bytes))
                (Bigint.to_int (integer c) * bytes))
      | Shrink bounds when not (ints (List.map fst bounds)) ->
          (* A symbolic start into a view that reorders or pads its storage
             moves by bytes unknown here: only the element's size holds. *)
          moved (Bigint.of_int bytes) 0
      | _ -> (align, phase))
  | (Op.Bitcast | Op.After), x :: _ -> storage_phase x
  | o, x :: _ when Op.Set.mem o Op.Set.movement -> storage_phase x
  | Op.Stage, x :: _ when op x = Op.Bitcast || Op.Set.mem (op x) Op.Set.movement
    -> (
      match view_start x with
      | No_view -> (16, 0)
      | Unknown -> (1, 0)
      | Start (align, phase) ->
          (* The largest power of two up to [align] that [phase] is a multiple
             of: storage on a boundary starts there too. *)
          let rec known a = if phase mod a = 0 then a else known (a / 2) in
          (known align, 0))
  | _ -> (16, 0)

let fits_start ~param:(align, phase) (a, p) = a >= align && p mod align = phase
let allocated_start = (16, 0)

let param_start body slot =
  List.find_map
    (fun u ->
      match (op u, arg u) with
      | Op.Param, Param p when p.addrspace = Dtype.Global && p.slot = slot ->
          Some (p.align, p.phase)
      | _ -> None)
    (toposort ~calls:Skip body)

let param_like ?name u slot =
  match op u with
  | Op.Param when addrspace u = Some Dtype.Alu ->
      let p = param_arg_of u in
      replace u ~arg:(Param { p with slot; name; bound = None })
  | _ ->
      let align, phase = storage_phase u in
      param ?shape:(shape_opt u) ?device:(device u) ?name ~phase ~align slot
        (dtype u)

(* Variables *)

let is_bound_var u =
  is_variable u
  && match arg u with Param { bound = Some _; _ } -> true | _ -> false

let bind var x =
  if (not (is_variable var)) || is_bound_var var then
    invalid_argf "only an unbound variable binds, not %s" (Op.name (op var));
  let c = const (x :> Dtype.const) in
  if not (Value.( <= ) (vmin var) (vmin c) && Value.( <= ) (vmax c) (vmax var))
  then
    invalid_argf "%s is out of [%s, %s]" (repr_const x)
      (repr_const (vmin var))
      (repr_const (vmax var));
  let p = param_arg_of var in
  replace var ~arg:(Param { p with bound = Some x })

let unbound var =
  if not (is_variable var) then
    invalid_argf "%s is not a variable" (Op.name (op var));
  replace var ~arg:(Param { (param_arg_of var) with bound = None })

let unbind var =
  match (is_bound_var var, arg var) with
  | true, Param { bound = Some x; _ } -> (unbound var, x)
  | _ -> invalid_arg "only a bound variable unbinds"

let unbind_all u =
  let bound =
    List.filter is_bound_var
      (Nodes.to_list (backward_slice_with_self ~calls:Skip u))
  in
  let pairs = List.map (fun x -> (x, unbound x)) bound in
  ( substitute ~calls:Skip ~pass:Once u pairs,
    List.map (fun (x, var) -> (var, snd (unbind x))) pairs )

let variables u =
  let found =
    dedup_nodes
      (List.filter_map
         (fun x ->
           if op x = Op.Param && addrspace x = Some Dtype.Alu then
             Some (if is_variable x then unbound x else x)
           else None)
         (Nodes.to_list (backward_slice_with_self ~calls:Skip u)))
  in
  let key x =
    let p = param_arg_of x in
    (Option.value p.name ~default:"", p.slot)
  in
  List.stable_sort (fun a b -> Stdlib.compare (key a) (key b)) found

(* Calls *)

let store_call dst src =
  call (store (param_like dst 0) (param_like src 1)) [ dst; src ]

let pm_resolve_params : (t option array, t) Pattern_matcher.t =
  Pattern_matcher.(
    v
      (fun () -> [
        rule_ctx (Upat.op Op.Param ~name:"p") (fun params m ->
            let slot = (param_arg_of (m "p")).slot in
            if slot >= 0 then params.(slot) else None);
      ]))

let call_with_outputs ?name ?(precompile = false) ?output_pos values args =
  let n = List.length args + List.length values in
  let default_dev = List.find_map device (values @ args) in
  let pos =
    match output_pos with
    | None -> List.init (List.length values) (fun i -> List.length args + i)
    | Some p -> p
  in
  if
    List.length pos <> List.length values
    || List.length (List.sort_uniq Int.compare pos) <> List.length pos
  then invalid_arg "output_pos needs one distinct position per output";
  let rec ascending = function
    | a :: (b :: _ as r) -> a < b && ascending r
    | _ -> true
  in
  if not (ascending pos) then
    invalid_arg "output_pos must be strictly ascending";
  if not (List.for_all (fun p -> p >= 0 && p < n) pos) then
    invalid_arg "output_pos must be within the argument list";
  let params = Array.make n None in
  let remaining = ref args in
  for i = 0 to n - 1 do
    if not (List.mem i pos) then
      match !remaining with
      | a :: rest ->
          params.(i) <- Some a;
          remaining := rest
      | [] -> ()
  done;
  let mint o p =
    let dev = match device o with Some d -> Some d | None -> default_dev in
    let buf = alloc (shape o) (dtype o) ?device:dev in
    let resolved =
      List.map
        (function
          | Int k -> Int k
          | Sym s ->
              Sym
                (graph_rewrite ~calls:Skip ~pass:Once ~ctx:params s
                   (After_sources pm_resolve_params)))
        (shape o)
    in
    ( alloc resolved (dtype o)
        ~slot:(param_arg_of (buf_uop buf)).slot
        ?device:dev,
      param_like buf p )
  in
  let outputs = List.map2 mint values pos in
  let body = sink (List.map2 (fun x (_, p) -> store p x) values outputs) in
  let inputs =
    ref (List.map (fun x -> if precompile then contiguous x else x) args)
  in
  let call_args =
    List.init n (fun i ->
        match List.find_index (Int.equal i) pos with
        | Some k -> fst (List.nth outputs k)
        | None -> (
            match !inputs with
            | x :: rest ->
                inputs := rest;
                x
            | [] -> invalid_arg "too few arguments"))
  in
  let c = call body call_args ?name ~precompile in
  List.map (fun (r, _) -> after r [ c ]) outputs

let call_with_output ?name ?precompile value args =
  List.hd (call_with_outputs ?name ?precompile [ value ] args)

(* The storage parameter [u] reaches through its first sources, if any. *)
let rec param_under u =
  match op u with
  | Op.Param -> if addrspace u = Some Dtype.Alu then None else Some u
  | _ -> ( match src u with x :: _ -> param_under x | [] -> None)

let rec written_params body =
  let slots = ref [] in
  let add u =
    match param_under u with
    | Some p ->
        let s = (param_arg_of p).slot in
        if s >= 0 && not (List.mem s !slots) then slots := s :: !slots
    | None -> ()
  in
  (* A call writes its arguments at the positions its body writes, or every
     argument where the body does not show them. *)
  let call c =
    let args = Array.of_list (List.tl (src c)) in
    match written_params (nth c 0) with
    | Some w ->
        List.iter (fun i -> if i < Array.length args then add args.(i)) w
    | None -> Array.iter add args
  in
  let rec entry e =
    match op e with
    | Op.Call -> call e
    | Op.Store -> add (nth e 0)
    | Op.End | Op.Backedge -> entry (nth e 0)
    | Op.Linear -> List.iter entry (src e)
    | _ -> ()
  in
  let shown () = Some (List.sort Int.compare !slots) in
  match op body with
  | Op.Sink ->
      List.iter
        (fun u ->
          match op u with
          | Op.Store -> add (nth u 0)
          | Op.Call -> call u
          | _ -> ())
        (toposort ~calls:Skip body);
      shown ()
  | Op.Linear ->
      (* A schedule's calls, or a kernel's lines: a body lowered inside a
         kernel holds its stores and calls. *)
      List.iter entry (src body);
      shown ()
  | Op.Store ->
      add (nth body 0);
      shown ()
  | _ -> None

(* Launches *)

let is_kernel_sink u =
  op u = Op.Sink && match arg u with Kernel _ -> true | _ -> false

(* A launch's calls of its members are compiled on their own. *)
let is_member_call c =
  op c = Op.Call
  && is_kernel_sink (nth c 0)
  && match arg c with Call { precompile; _ } -> precompile | _ -> false

let is_launch u =
  is_kernel_sink u && src u <> [] && List.for_all is_member_call (src u)

(* The launch's parameters are its calls' arguments, each once, storage
   first: members then read one parameter where they share storage. *)
let launch calls =
  List.iter
    (fun c ->
      if
        op c <> Op.Call
        || (not (is_kernel_sink (nth c 0)))
        || match arg c with Call { precompile; _ } -> precompile | _ -> true
      then invalid_arg "a launch's member is a call of a kernel")
    calls;
  let seen = Tbl.create 16 and order = ref [] in
  List.iter
    (fun c ->
      List.iter
        (fun a ->
          if not (Tbl.mem seen a) then begin
            Tbl.replace seen a ();
            order := a :: !order
          end)
        (src_without_body c))
    calls;
  let scalar a = addrspace a = Some Dtype.Alu in
  let order = List.rev !order in
  let args =
    List.filter (fun a -> not (scalar a)) order @ List.filter scalar order
  in
  let params = Tbl.create 16 in
  List.iteri
    (fun i a ->
      (* A kernel's parameters name no device: its call's arguments do. *)
      let p =
        if scalar a then
          if op a = Op.Param then param_like a i
          else
            param ~vmin_vmax:(vmin a, vmax a) ~addrspace:Dtype.Alu i
              (Dtype.strong (dtype a))
        else
          let p = param_like a i in
          match arg p with
          | Param q -> replace p ~arg:(Param { q with device = None })
          | _ -> p
      in
      Tbl.replace params a p)
    args;
  let member c =
    let name = match arg c with Call k -> k.name | _ -> None in
    call ?name ~precompile:true (nth c 0)
      (List.map (Tbl.find params) (src_without_body c))
  in
  call
    (sink
       ~kernel:(kernel_info ~name:"launch" ~optimized:true ())
       (List.map member calls))
    args

let launch_members c =
  if op c <> Op.Call || not (is_launch (nth c 0)) then
    invalid_arg "launch_members takes a call of a launch";
  let args = Array.of_list (src_without_body c) in
  let passed p =
    match (op p, arg p) with
    | Op.Param, Param q when q.slot >= 0 && q.slot < Array.length args ->
        args.(q.slot)
    | _ -> invalid_arg "a launch's member is passed its parameters"
  in
  List.map
    (fun m ->
      let name = match arg m with Call k -> k.name | _ -> None in
      call ?name (nth m 0) (List.map passed (src_without_body m)))
    (src (nth c 0))

let custom_kernel args f =
  (* A variable stays a variable of the kernel, unbound, in its own slot. *)
  let placeholder i s =
    if is_variable s then
      replace s ~arg:(Param { (param_arg_of s) with slot = i; bound = None })
    else placeholder_like s i
  in
  let placeholders = List.mapi placeholder args in
  let body = f placeholders in
  let kernel = call body args in
  (* An argument the kernel only reads is no state the kernel leaves. *)
  let written i =
    match written_params body with Some w -> List.mem i w | None -> true
  in
  List.mapi (fun i s -> if written i then after s [ kernel ] else s) args

(* Programs *)

(* The slot of the buffer parameter that the load or store [u] accesses, and
   how. *)
let accessed u =
  if op u <> Op.Store && op u <> Op.Load then None
  else
    let s0 = nth u 0 in
    let idx =
      if op s0 = Op.Index || op s0 = Op.Shrink then Some s0
      else if op s0 = Op.Cast && op (nth s0 0) = Op.Index then Some (nth s0 0)
      else None
    in
    match idx with
    | Some idx ->
        let buf = buf_uop (nth idx 0) in
        if op buf <> Op.Param then None
        else
          Some
            ( (if op u = Op.Store then `Write else `Read),
              (param_arg_of buf).slot )
    | None -> None

(* The slots of the buffer parameters of its caller that the call [c] writes
   and reads, through the loads and stores of its body, an instruction list
   or a sink, and of the calls the body makes. *)
let rec call_accesses c =
  let b = nth c 0 in
  let nodes =
    match op b with
    | Op.Linear -> src b
    | Op.Sink -> toposort ~calls:Skip b
    | _ -> []
  in
  let args = Array.of_list (src_without_body c) in
  let rec param_of a =
    match op a with
    | Op.Param -> if addrspace a = Some Dtype.Alu then None else Some a
    | _ -> ( match src a with x :: _ -> param_of x | [] -> None)
  in
  let passed slot =
    if slot < 0 || slot >= Array.length args then []
    else
      match param_of args.(slot) with
      | Some p -> [ (param_arg_of p).slot ]
      | None -> []
  in
  List.fold_left
    (fun (w, r) u ->
      match accessed u with
      | Some (`Write, slot) -> (passed slot @ w, r)
      | Some (`Read, slot) -> (w, passed slot @ r)
      | None when op u = Op.Call ->
          let cw, cr = call_accesses u in
          (List.concat_map passed cw @ w, List.concat_map passed cr @ r)
      | None -> (w, r))
    ([], []) nodes

let program_info_of_sink
    ?(target =
      Helpers.Target.
        { device = ""; renderer = ""; arch = ""; interface = ""; indices = "" })
    sink =
  let params = ref [] and globals = ref [] and outs = ref [] and ins = ref [] in
  let global_size = Array.make 3 (Int 1)
  and local_size = Array.make 3 (Int 1) in
  (match arg sink with
  | Kernel { split = Some s; _ } -> global_size.(0) <- s.iterations
  | _ -> ());
  List.iter
    (fun u ->
      if op u = Op.Param then begin
        params := u :: !params;
        if addrspace u <> Some Dtype.Alu then
          globals := (param_arg_of u).slot :: !globals
      end;
      (match accessed u with
      | Some (`Write, slot) -> outs := slot :: !outs
      | Some (`Read, slot) -> ins := slot :: !ins
      | None -> ());
      (* A call writes and reads its arguments where its body does. *)
      if op u = Op.Call then begin
        let w, r = call_accesses u in
        outs := w @ !outs;
        ins := r @ !ins
      end;
      if op u = Op.Special then
        match arg u with
        | String name ->
            let axis =
              Char.code name.[String.length name - 1] - Char.code '0'
            in
            let sizes = if name.[0] = 'l' then local_size else global_size in
            (* A size reads each named variable as the launch binds it, by
               name; a nameless scalar is its call's argument in its slot. *)
            let free =
              List.filter_map
                (fun v ->
                  if is_variable v && (param_arg_of v).name <> None then
                    Some
                      ( v,
                        replace v
                          ~arg:(Param { (param_arg_of v) with slot = -1 }) )
                  else None)
                (toposort ~calls:Skip (nth u 0))
            in
            sizes.(axis) <-
              ssimplify (substitute ~calls:Skip ~pass:Once (nth u 0) free)
        | _ -> invalid_arg "a hardware index needs a name")
    (toposort ~calls:Skip sink);
  let sorted l = List.sort_uniq Int.compare l in
  let outs, ins =
    if List.is_empty !outs && List.is_empty !ins then (!globals, !globals)
    else (!outs, !ins)
  in
  let params =
    List.stable_sort
      (fun a b -> Int.compare (param_arg_of a).slot (param_arg_of b).slot)
      (List.rev !params)
  in
  {
    global_size = Array.to_list global_size;
    local_size = Array.to_list local_size;
    params;
    globals = sorted !globals;
    outs = sorted outs;
    ins = sorted ins;
    target;
  }

let kernel_params prg =
  match (op prg, arg prg, src prg) with
  | Op.Program, Program info, _ :: linear :: _ ->
      let slot u = (param_arg_of u).slot in
      (* The linear holds each of the program's instructions: its parameters
         are told by their operation before the lookup. *)
      let params =
        List.filter
          (fun u -> op u = Op.Param && List.memq u info.params)
          (src linear)
      in
      let taken = List.map slot params in
      let rest =
        List.filter (fun p -> not (List.mem (slot p) taken)) info.params
      in
      List.stable_sort
        (fun a b -> Int.compare (slot a) (slot b))
        (params @ rest)
  | _ -> invalid_arg "kernel_params takes a linearized program"

let launch_dims (p : program_info) vars =
  ( List.map (fun s -> sym_infer s vars) p.global_size,
    List.map (fun s -> sym_infer s vars) p.local_size )
