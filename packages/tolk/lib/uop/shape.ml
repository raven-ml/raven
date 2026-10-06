(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops

(* Lists *)

(* The first [n] elements of a list and the rest, as Python's slices: short
   lists give what they have. *)
let rec take n l =
  if n <= 0 then [] else match l with [] -> [] | x :: r -> x :: take (n - 1) r

let rec drop n l =
  if n <= 0 then l else match l with [] -> [] | _ :: r -> drop (n - 1) r

module Value = Dtype.Value

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let repr_const c = Format.asprintf "%a" Dtype.pp_const c
let repr_dtype dt = Format.asprintf "%a" Dtype.pp dt

let repr_tuple = function
  | [ x ] -> "(" ^ x ^ ",)"
  | xs -> "(" ^ String.concat ", " xs ^ ")"

let min_max u = (vmin u, vmax u)

let equal_sint s0 s1 =
  match (s0, s1) with
  | Int n0, Int n1 -> Int.equal n0 n1
  | Sym u0, Sym u1 -> u0 == u1
  | _ -> false

let repr_sint = function
  | Int n -> string_of_int n
  | Sym u -> Format.asprintf "%a" pp u

let equal_shape = List.equal equal_sint
let repr_shape s = repr_tuple (List.map repr_sint s)

let align_left shapes =
  let n = List.fold_left (fun m s -> max m (List.length s)) 0 shapes in
  List.map (fun s -> List.init (n - List.length s) (fun _ -> Int 1) @ s) shapes

let first op = function
  | s :: _ -> s
  | [] -> invalid_argf "%s needs a source" (Op.name op)

let as_value (c : Dtype.const) : Dtype.value =
  match c with
  | `Invalid -> invalid_arg "Invalid is not a number"
  | #Dtype.value as x -> x

(* A constant that takes part in a greatest common divisor: an integer. *)
let integer c =
  match as_value c with
  | `Float x when not (Float.is_integer x) ->
      invalid_argf "%s is not an integer" (repr_const c)
  | v -> Value.to_z v

(* A recursive property is filled bottom-up over the nodes that lack it, so a
   deep graph never recurses deeply. *)
let memoized ~calls ~get ~set ~compute u =
  match get u with
  | Some x -> x
  | None ->
      (* A new node is mostly built on nodes that have the property. *)
      let srcs =
        if calls = Skip && op u = Op.Call then drop 1 (src u) else src u
      in
      if List.for_all (fun s -> Option.is_some (get s)) srcs then
        set u (compute u)
      else
        List.iter
          (fun n -> set n (compute n))
          (toposort ~calls ~gate:(fun n -> Option.is_none (get n)) u);
      Option.get (get u)

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
  if not (List.exists (Dtype.equal (dtype u)) kinds) then
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

let marg u =
  match movement_memo u with
  | Some m -> m
  | None ->
      let shape_src i = as_shape (nth u i) in
      let m =
        match (op u, arg u) with
        | Op.Reshape, _ -> Reshape (shape_src 1)
        | Op.Expand, _ -> Expand (shape_src 1)
        | Op.Pad, _ -> Pad (List.combine (shape_src 1) (shape_src 2))
        | Op.Shrink, _ -> Shrink (List.combine (shape_src 1) (shape_src 2))
        | Op.Permute, Axes l -> Permute l
        | Op.Flip, Flips l -> Flip l
        | op, _ -> invalid_argf "%s is not a movement" (Op.name op)
      in
      set_movement_memo u m;
      m

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
  memoized ~calls:Enter
    ~get:(fun n -> shape_memo n)
    ~set:(fun n s -> set_shape_memo n s)
    ~compute:compute_shape u

and shape u =
  match shape_opt u with
  | Some s -> s
  | None -> invalid_argf "%s has no shape" (Op.name (op u))

and compute_shape u : sint list option =
  let src0 () = first (op u) (src u) in
  let void = Dtype.equal (dtype u) Dtype.Void in
  match op u with
  | Op.If | Op.Barrier | Op.Sink | Op.Endif | Op.Backedge | Op.Group | Op.Linear
  | Op.Program | Op.Source | Op.Custom_function ->
      None
  | Op.Call | Op.Ins -> if void then None else Some []
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
  | Op.Buffer | Op.Alloc | Op.Param -> (
      match arg u with
      | Param { size = None; _ } -> Some []
      | Param { size = Some n; _ } -> Some [ Int n ]
      | _ -> invalid_arg "storage needs a ParamArg")
  | Op.Custom | Op.Customi -> (
      if void then None
      else
        match List.filter_map shape_opt (src u) with
        | [] -> None
        | shapes -> Some (broadcast_shape shapes))
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
  | Op.Mstack | Op.Mselect | Op.Detach | Op.Contiguous_backward | Op.After
  | Op.Load | Op.Copy | Op.Allreduce | Op.Store | Op.End ->
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
  | Op.Unshard when List.is_empty (src u) -> None
  | op when Op.Set.mem op Op.Set.movement || op = Op.Unshard || op = Op.Reduce
    ->
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
        | Sym d when List.exists is_invalid (toposort ~calls:Enter d) ->
            invalid_argf "%s of sizes %s, one holding Invalid, which is no number"
              (Op.name (op u)) (repr_shape s)
        | _ -> ())
      s;
    s
  in
  match op u with
  | Op.Unshard -> (
      match arg u with
      | Axes axes ->
          let ranges = drop 1 (src u) in
          List.mapi
            (fun a s ->
              match List.find_index (Int.equal a) axes with
              | Some i ->
                  Sint.(
                    s
                    * Int
                        (Value.to_int
                           (Value.( + ) (vmax (List.nth ranges i)) (`Int Bigint.one))))
              | None -> s)
            ps
      | _ -> invalid_arg "an unshard needs its axes")
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
          numbers (List.map (fun (_, sz) -> ssimplify_sint sz) bounds)
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
          numbers (List.map (fun (_, sz) -> ssimplify_sint sz) bounds)
      | Flip flips ->
          if List.length flips <> List.length ps then bad "axes";
          ps)

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
  | Permute l -> v Op.Permute ~src:[ u ] ~arg:(Axes l)
  | Flip l -> v Op.Flip ~src:[ u ] ~arg:(Flips l)

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

let flip u axes =
  let axes = List.map (resolve_dim u) axes in
  if List.length (List.sort_uniq Int.compare axes) <> List.length axes then
    invalid_argf "an axis appears twice in %s"
      (repr_tuple (List.map string_of_int axes));
  let flips = List.init (ndim u) (fun i -> List.mem i axes) in
  if List.exists Fun.id flips then mop u (Flip flips) else u

let check_rank u what l =
  if ndim u <> List.length l then
    invalid_argf "%s of %d axes for a node of %d" what (List.length l) (ndim u)

let shrink u bounds =
  check_rank u "bounds" bounds;
  unless_same u
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

let is_zero (c : Dtype.const) =
  match c with
  | `Invalid -> false
  | #Dtype.value as x -> Value.( = ) x (`Int Bigint.zero)

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

let split_cumalu = 256

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

let cumalu u axis op =
  let axis = resolve_dim u axis and last = ndim u - 1 in
  let s = running_size u axis and t = transpose u axis last in
  if List.exists (fun d -> equal_sint d (Int 0)) (shape u) then u
  else if s <= 2 * split_cumalu then transpose (pooled_cumalu t op) axis last
  else
    let value = identity_element op (dtype u) in
    let rounded = Helpers.round_up s split_cumalu in
    let t = pad ~value t (at_axis t last (Int (rounded - s), Int 0)) in
    let chunks =
      pooled_cumalu
        (unflatten t last [ Int (rounded / split_cumalu); Int split_cumalu ])
        op
    in
    let ends =
      squeeze ~axis:(last + 1)
        (shrink chunks
           (at_axis chunks (last + 1)
              (Int (split_cumalu - 1), Int split_cumalu)))
    in
    let base =
      pad ~value (pooled_cumalu ends op) (at_axis ends last (Int 1, Int (-1)))
    in
    let combine =
      if Op.equal op Op.Add then add
      else if Op.equal op Op.Mul then mul
      else maximum
    in
    let whole =
      flatten ~start:last
        (combine chunks (reshape base (shape base @ [ Int 1 ])))
    in
    transpose
      (shrink whole (at_axis whole last (Int (rounded - s), Int rounded)))
      axis last

let arange ?(start = 0) ?(step = 1) ?dtype stop =
  if step = 0 then invalid_arg "an arange of step 0";
  let lo, hi =
    if step > 0 then (start, stop - step) else (stop - step, start)
  in
  let dt =
    match dtype with
    | Some dt -> dt
    | None -> Dtype.commit_int (Bigint.of_int lo) (Bigint.of_int hi)
  in
  if
    Value.(`Int (Bigint.of_int lo) < Dtype.min dt)
    || Value.(Dtype.max dt < `Int (Bigint.of_int hi))
  then
    invalid_argf "arange [%d, %d) is not representable in %s" start stop
      (repr_dtype dt);
  let n = Helpers.ceildiv (stop - start) step in
  let full dt c k = expand (const ~dtype:dt (`Int (Bigint.of_int c))) [ Int k ] in
  if n <= 0 then full dt 0 0
  else
    let acc =
      if Dtype.is_float dt then Dtype.least_upper [ dt; Float32 ] else dt
    in
    cast (add (pooled_cumalu (full acc step n) Op.Add) (int (start - step))) dt

(* Several devices *)

(* Storage *)

(* Variables *)

let is_variable u =
  op u = Op.Param
  && (match arg u with
    | Param { vmin_vmax = Some _; addrspace = Some Dtype.Alu; _ } -> true
    | _ -> false)
  && match shape_opt u with Some [] -> true | _ -> false

(* Divisibility *)

let rec divides u (n : Bigint.t) =
  if Bigint.equal n Bigint.one then Some u
  else
    match op u with
    | Op.Const ->
        let x = as_value (value u) and n = `Int n in
        if Value.(x % n = of_int 0) then
          Some (const_like u (Value.(x // n) :> Dtype.const))
        else None
    | Op.Stack ->
        let srcs = List.map (fun s -> divides s n) (src u) in
        if List.exists Option.is_none srcs then None
        else Some (v Op.Stack ~src:(List.map Option.get srcs))
    | Op.Add -> (
        match (divides (nth u 0) n, divides (nth u 1) n) with
        | Some d0, Some d1 -> Some (add d0 d1)
        | _ -> None)
    | Op.Mul -> (
        match divides (nth u 0) n with
        | Some d0 -> Some (mul d0 (nth u 1))
        | None -> Option.map (fun d1 -> mul (nth u 0) d1) (divides (nth u 1) n))
    | op when Op.Set.mem op Op.Set.defines -> (
        match arg u with
        | Param { multiple_of = Some m; _ } ->
            if Bigint.equal (Bigint.rem (Bigint.of_int m) n) Bigint.zero then
              Some (div ~rounding:`Floor u (const (`Int n)))
            else None
        | _ -> None)
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

let sym_infer (s : sint) vars =
  match s with
  | Int n -> n
  | Sym u ->
      let s = simplify u in
      let cache = Tbl.create 16 in
      let get n = Tbl.find cache n in
      let eval n : Dtype.value =
        match op n with
        | Op.Const -> as_value (value n)
        | Op.Param when addrspace n = Some Dtype.Alu || is_variable n -> (
            let name = expr n in
            match List.assoc_opt name vars with
            | Some x -> `Int (Bigint.of_int x)
            | None -> invalid_argf "the variable %s has no value" name)
        | Op.Cast -> sym_cast (dtype n) (get (first (op n) (src n)))
        | Op.Bitcast ->
            Dtype.bitcast (dtype (first (op n) (src n))) (dtype n)
              (get (first (op n) (src n)))
        | op when Op.Set.mem op Op.Set.alu ->
            sym_alu op (dtype n) (List.map get (src n))
        | op -> invalid_argf "%s cannot be evaluated" (Op.name op)
      in
      Value.to_int (topovisit s eval cache)

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
  | Op.Cdiv -> Some (int_quotient ~toward_zero:true)
  | Op.Cmod -> Some (int_remainder ~toward_zero:true)
  | Op.Floordiv -> Some (int_quotient ~toward_zero:false)
  | Op.Floormod -> Some (int_remainder ~toward_zero:false)
  | _ -> None

let sym_compile (s : sint) var =
  match s with
  | Int n -> fun _ -> n
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
      (* Each node's value, as [sym_infer] computes it. *)
      let values = Tbl.create 16 in
      let rec compute n = memo values exact n
      and exact n =
        match op n with
        | Op.Const ->
            let v = as_value (value n) in
            fun _ -> v
        | Op.Param when addrspace n = Some Dtype.Alu || is_variable n ->
            let read = var n in
            fun env -> `Int (Bigint.of_int (read env))
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
          | Op.Param, _ when addrspace n = Some Dtype.Alu || is_variable n ->
              Some (var n)
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
      let exact = compute s in
      let to_int env = Value.to_int (exact env) in
      match int s with
      | None -> to_int
      | Some f -> fun env -> ( try f env with Inexact -> to_int env)

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

(* Calls *)

(* Programs *)

(* Symbolic rules *)

module V = Dtype.Value

(* Weak constants *)

let is_weak dt = List.mem dt Dtype.weaks
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

(* PARAM // c is irreducible *)
let irreducible x y =
  match arg x with
  | Param { multiple_of = Some m; _ } ->
      op x = Op.Param && op y = Op.Const && V.(of_int m % num y = zero)
  | _ -> false

let rec fold_divmod_general d =
  match Mutex.protect memo_lock (fun () -> Memo.find_opt memo d) with
  | Some ret -> ret
  | None ->
      let ret = fold d in
      Mutex.protect memo_lock (fun () -> Memo.replace memo d ret);
      ret

and fold d =
  let x = nth d 0 and y = nth d 1 and is_mod = op d = Op.Floormod in
  if V.(vmin y = zero && vmax y = zero) then raise Division_by_zero;
  let xdiv = O.(x // y) in
  let q = vmin xdiv in
  (* x // y is constant *)
  if V.(q = vmax xdiv) then
    Some
      (if is_mod then O.(x - (lit q * y))
       else const_like xdiv (q :> Dtype.const))
  else if irreducible x y then if is_mod then Some (zero_like d) else None
  else
    let x_peeled, const = pop_const x in
    let const = number const and uops_no_const = split_uop x_peeled Op.Add in
    (* Constant denominator rules, for a constant c > 0 *)
    let c = if op y = Op.Const then num y else zero in
    (* nested_div: (x % (k * c)) // c is (x // c) % k for k > 0; the mod case is
       remove_nested_mod's *)
    let nested_div () =
      if is_mod || op x <> Op.Floormod then None
      else
        match divides (nth x 1) (V.to_z c) with
        | Some k when V.(vmin k > zero) -> Some O.(nth x 0 // y % k)
        | _ -> None
    in
    (* remove_nested_mod in a sum: (a % 4 + b) % 2 is (a + b) % 2 *)
    let remove_nested_mod () =
      let unnest u =
        if op u = Op.Floormod && Option.is_some (divides (nth u 1) (V.to_z c))
        then nth u 0
        else u
      in
      let xs = List.map unnest uops_no_const in
      if (not is_mod) || List.equal ( == ) xs uops_no_const then None
      else Some O.((usum (List.hd xs) (List.tl xs) + lit const) % y)
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
      (* gcd_with_remainder: factor out the common gcd of the numerator *)
      let gcd_with_remainder () =
        let g =
          `Int
            (List.fold_left (fun g f -> Bigint.gcd g (V.to_z f)) (V.to_z c) factors)
        in
        if V.(g <= of_int 1) then None
        else
          let x_g = simplify (Option.get (divides x_peeled (V.to_z g))) in
          let new_x = O.(x_g + lit V.(const // g % (c // g))) in
          if V.(vmin new_x < zero) then None
          else if is_mod then
            Some O.((new_x % lit V.(c // g) * lit g) + lit V.(const % g))
          else Some O.((new_x // lit V.(c // g)) + lit V.(const // c))
      in
      (* nest_by_factor: x // c is (x // f) // (c // f), and x % c is (x // f %
         (c // f)) * f + x % f. The division holds for x of any sign; the
         remainder's reconstruction needs x >= 0 *)
      let nest_by_factor () =
        let divisor u f =
          let f = V.(max f (-f)) in
          if op u <> Op.Const && V.(of_int 1 < f && f < c && c % f = zero) then
            Some f
          else None
        in
        let divs =
          List.sort_uniq V.compare
            (List.filter_map Fun.id (List.map2 divisor uops_no_const factors))
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
      first_rule [ congruence; gcd_with_remainder; nest_by_factor ]
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
      if V.(c > zero) then [ nested_div; remove_nested_mod; folds ] else []
    in
    first_rule (constant_rules @ [ divide_by_gcd; fallback ])

let floor_ops = Op.Set.of_list [ Op.Floordiv; Op.Floormod ]

let div_and_mod_symbolic =
  Pattern_matcher.v
    (fun () -> [
      (* Fast inline rules *)
      (* (x // c + a) // d is (x + a * c) // (c * d) for d > 0, where
           nothing wraps *)
      rule
        Upat.(((var "x" // cvar "c") + cvar "a") // cvar "d")
        (fun m ->
          let x = m "x" and c = m "c" and a = m "a" and d = m "d" in
          let ac = V.(vmin a * vmin c) and cd = V.(vmin c * vmin d) in
          let values =
            V.[ vmin a; vmin c; vmin d; ac; cd; vmin x + ac; vmax x + ac ]
          in
          if V.(vmin d > zero) && exact (dtype x) values then
            Some O.((x + (a * c)) // (c * d))
          else None);
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
          else if op (m "n") = Op.Floordiv then
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
          (Upat.op Op.Permute ~name:"x" ~src:[ Upat.op Op.Permute ~name:"x2" ])
          (fun m ->
            let x2 = m "x2" in
            let order =
              List.map (List.nth (permute_arg x2)) (permute_arg (m "x"))
            in
            Some (replace ~arg:(Axes order) x2));
        (* Remove no-op permutes. *)
        rule (Upat.op Op.Permute ~name:"x") (fun m ->
            let x = m "x" in
            let order = permute_arg x in
            let identity = List.init (List.length order) Fun.id in
            if List.equal Int.equal order identity then Some (nth x 0) else None);
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

let dedup l =
  Helpers.dedup
    (module struct
      type t = Ops.t

      let equal = ( == )
      let hash = hash
    end)
    l

(* Phase 1: the most generic folding rules *)

(* The reciprocal overflows only where a power of magnitude at least 1 does, and
   [sqrt] gives -0. and NaN at -0. and -inf, where a half-integer power is +0.
   and +inf. *)
let simplify_pow x c =
  let c = num c and pow x v = pow x (lit v) in
  let h = V.(c - `Float 0.5) in
  match c with
  | `Float f when not (Float.is_finite f) -> None
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
    | Op.Cast, (#Dtype.value as v) -> (at (dtype s) v :> Dtype.const)
    | Op.Const, (`Int _ as v)
      when Dtype.equal (dtype s) Dtype.Weak_int && List.mem operands Dtype.ints
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

(* the B with q == B//div and B%div == base%div, or None. only such congruence
   is needed to recombine, and canonicalization moves consts freely: the
   quotient may be merged ((x//c + a)//div -> (x + a*c)//(c*div) for div>0) and
   shifted ((y + k*D)//D == y//D + k) *)
let quotient_base q base div =
  let (q, s), (n, a) = (pop_num q, pop_num base) in
  if op q <> Op.Floordiv || not (is_const (nth q 1)) then None
  else
    let qd = num (nth q 1) in
    let merged =
      if V.(div > zero) && op n = Op.Floordiv && is_const (nth n 1) then
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
      if op md <> Op.Floormod || not (is_const (nth md 1)) then None
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
              | None when op q = Op.Floormod && is_const (nth q 1) ->
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
   every lane at once, so that gate lifts out *)
let lift_reduce_gate red cond x i =
  match arg red with
  | Reduce { num_axes = 0; _ } -> (
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
      let inner = match keep with [] -> x | _ -> where (conj keep) x i in
      match lift with
      | [] -> None
      | _ -> Some (where (conj lift) (replace red ~src:(inner :: ranges)) i))
  | _ -> None

let unary_or_cast = Op.Set.union Op.Set.unary (ops [ Op.Cast; Op.Bitcast ])

(* A stack of Invalid lanes stays a stack: one Invalid in its place would drop
   the width that the lanes' movements and devectorize read. *)
let pm_data_invalid =
  pm
    (fun () -> [
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
      (* a multiply-add (D25) moves inside the gate of each operand in turn,
         and an Invalid operand makes it Invalid, as binary ops do *)
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
        (Upat.op Op.Store
           ~src:
             [
               Upat.or_casted
                 (Upat.index ~allow_any_len:true Upat.wild [ invalid_pat ]);
               Upat.wild;
             ])
        (fun _ -> Some (v Op.Noop));
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
  | Op.Reshape | Op.Expand | Op.Permute | Op.Shrink | Op.Flip ->
      broadcast_const (nth u 0)
  | _ -> None

(* folding a strong dtype WHERE to a weak const branch keeps the strong dtype *)
let fold_const_where gate c0 c1 w =
  let ret = if V.to_bool (num gate) then c0 else c1 in
  let weak u = List.mem (dtype u) Dtype.weaks in
  if is_const ret && weak ret && not (weak w) then ccast ret (dtype w) else ret

let boolean = [ Dtype.Bool ]
let int_like = Dtype.Weak_int :: Dtype.ints
let int_or_bool = Dtype.Bool :: int_like

let symbolic_simple =
  Pattern_matcher.concat
    [
      pm_data_invalid;
      pm
        (fun () -> [
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
          rule Upat.(var "x" // var "x") (fun m -> Some (const_v (m "x") one));
          rule Upat.(var "x" // int 1) (fun m -> Some (m "x"));
          rule Upat.(var "x" // int (-1)) (fun m -> Some (neg (m "x")));
          rule Upat.(var "x" lxor var "y" lxor var "y") (fun m -> Some (m "x"));
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
            Upat.(var ~dtype:boolean "x" <> const ~dtype:boolean (`Bool false))
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
          (* CAST(bool -> int) != const — CAST(True)=1, CAST(False)=0, so fold
             based on const value *)
          rule
            Upat.(
              f ~dtype:int_like (var ~dtype:boolean "x") Op.Cast <> cvar "c")
            (fun m ->
              let x = m "x" and c = m "c" in
              Some
                (if equals c zero then x
                 else if equals c one then logical_not x
                 else const_like x (`Bool true)));
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
          (* (x&mask)>>k -> x>>k when mask only clears bits below k *)
          rule
            Upat.((var "x" land cvar "mask") lsr cvar "k")
            (fun m ->
              let mask = V.to_z (num (m "mask"))
              and k = V.to_int (num (m "k")) in
              if
                (k >= 0)
                [@mutate off "a shift by 0 is the x >> 0 rule's, tried first"]
                && Bigint.(equal (logor mask (pred (shift_left one k))) minus_one)
              then Some (shr (m "x") (lit (`Int (Bigint.of_int k))))
              else None);
          rule
            Upat.(var "x" land cvar "mask" // cvar "c")
            (fun m ->
              let mask = V.to_z (num (m "mask")) and c = V.to_z (num (m "c")) in
              if
                Bigint.(
                  gt c zero
                  && equal (logand c (pred c)) zero
                  && equal (logor mask (pred c)) minus_one)
              then Some O.(m "x" // lit (`Int c))
              else None);
          (* x != x -> False (only ints) *)
          rule
            Upat.(var ~dtype:int_or_bool "x" <> var "x")
            (fun m -> Some (const_like ~dtype:Dtype.Bool (m "x") (`Bool false)));
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
                if List.mem (dtype s) Dtype.weaks then ccast s dt else s
              in
              if List.mem dt Dtype.weaks then None
              else Some (replace a ~src:(List.map commit (src a))));
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
          (* positive const ** x *)
          rule
            Upat.(alu (cvar "c") Op.Pow [ var "x" ])
            (fun m ->
              let c = m "c" in
              let cv = num c in
              if V.(cv = one) then Some c
              else if V.(cv > zero && cv < `Float Float.infinity) then
                Some (exp2 O.(m "x" * float (Float.log2 (V.to_float cv))))
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
              ((cast (var ~dtype:[ Dtype.Uint32 ] "x") Dtype.Uint64 lsl int 32)
              lor cast (v ~dtype:[ Dtype.Uint32 ] ()) Dtype.Uint64)
              lsr int 32)
            (fun m -> Some (cast (m "x") Dtype.Uint64));
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
      (fun u -> Bigint.equal (const_factor u) Bigint.one)
      (split_uop x Op.Add)
  in
  let d = List.fold_left (fun d u -> Bigint.gcd d (const_factor u)) c np in
  let sum f = List.fold_left (fun s u -> V.(s + f u)) zero p in
  match np with
  | n :: ns when Bigint.gt d Bigint.one && V.(zero <= sum vmin && sum vmax < `Int d) ->
      Some O.(Option.get (divides (usum n ns) d) < lit (`Int (Bigint.fdiv c d)))
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
       alone. Asked before the search below, which walks the branches. *)
    List.exists
      (fun u -> op_in_backward_slice_with_self ~calls:Skip u [ Op.Index ])
      [ cond; t; f ]
  then None
  else if not (reaches ~calls:Enter t cond || reaches ~calls:Enter f cond)
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
               (var ~dtype:int_or_bool "x" * cvar "c0") + (var "x" * cvar "c1"))
             (fun m -> Some O.(m "x" * (m "c0" + m "c1")));
           rule
             Upat.(
               var "y"
               + (var ~dtype:int_or_bool "x" * cvar "c0")
               + (var "x" * cvar "c1"))
             (fun m -> Some O.(m "y" + (m "x" * (m "c0" + m "c1"))));
           rule
             Upat.(var ~dtype:int_or_bool "x" + (var "x" * cvar "c"))
             (fun m -> Some O.(m "x" * (m "c" + int 1)));
           rule
             Upat.(var "y" + var ~dtype:int_or_bool "x" + (var "x" * cvar "c"))
             (fun m -> Some O.(m "y" + (m "x" * (m "c" + int 1))));
           rule
             Upat.(var "y" + (var ~dtype:int_or_bool "x" * cvar "c") + var "x")
             (fun m -> Some O.(m "y" + (m "x" * (m "c" + int 1))));
           rule Upat.(var "x" + var "x") (fun m -> Some O.(m "x" * int 2));
           rule
             Upat.(var "y" + var ~dtype:int_or_bool "x" + var "x")
             (fun m -> Some O.(m "y" + (m "x" * int 2)));
           (* -(x+c) -> -x + -c, for integers: -(x + c) is -0. at x = -c *)
           rule
             Upat.(int (-1) * (var ~dtype:int_or_bool "x" + cvar "c"))
             (fun m -> Some O.(-m "x" + -m "c"));
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
               var ~dtype:int_or_bool "y"
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
                       Op.Cmpne;
                       Op.Floordiv;
                       Op.Floormod;
                       Op.Param;
                       Op.After;
                       Op.Special;
                     ])
                ~name:"x" ())
             (fun m ->
               let x = m "x" in
               if V.(vmin x = vmax x) then Some (const_v x (vmin x)) else None);
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
           rule
             Upat.(
               where
                 (var ~dtype:int_or_bool "a" < cvar "b")
                 (cvar "c") (var "a"))
             (fun m ->
               if V.(num (m "b") = num (m "c")) then
                 Some (maximum (m "a") (m "b"))
               else None);
           (* a float maximum's bounds leave out NaN and the order of zeros *)
           rule
             Upat.(named "m" (maximum (var ~dtype:int_or_bool "x") (var "y")))
             (fun m ->
               let mx = m "m" and x = m "x" and y = m "y" in
               let (x0, x1), (y0, y1) =
                 (operand_bounds mx x, operand_bounds mx y)
               in
               (* the operand kept is committed to the maximum's type *)
               let keep u =
                 if List.mem (dtype u) Dtype.weaks then ccast u (dtype mx)
                 else u
               in
               if V.(x0 >= y1) then Some (keep x)
               else if V.(x1 <= y0) then Some (keep y)
               else None);
         ]
        (* Two stage ALU folding; sums, products and maxima for integers: in
           floats each step rounds, and a maximum keeps a NaN only as its first
           operand *)
        @ List.map
            (fun o ->
              let dtype =
                if List.mem o Op.[ Add; Mul; Max ] then Some int_or_bool
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
                    if o = Op.Max && List.mem (Ops.dtype c) Dtype.weaks then
                      ccast c (Ops.dtype f)
                    else c
                  in
                  let c = alu (at_width (m "c1")) o [ at_width (m "c2") ] in
                  Some (alu (m "x") o [ c ])))
            (Op.Set.to_list Op.Set.associative)
        @ [
            (* (x//c1)//c2 -> x//(c1*c2) for c2>0, where c1*c2 does not wrap *)
            rule
              Upat.(var "x" // cvar "c1" // cvar "c2")
              (fun m ->
                let c1 = vmin (m "c1") and c2 = vmin (m "c2") in
                if V.(c2 > zero) && exact (dtype (m "x")) V.[ c1; c2; c1 * c2 ]
                then Some O.(m "x" // (m "c1" * m "c2"))
                else None);
            (* Lt *)
            (* c0+x<c1 -> x < c1-c0, where neither side wraps *)
            rule
              Upat.(cvar "c0" + var ~dtype:int_like "x" < cvar "c1")
              (fun m ->
                let x = m "x" and c0 = vmin (m "c0") and c1 = vmin (m "c1") in
                if
                  exact (dtype x)
                    V.[ c0; c1; vmin x + c0; vmax x + c0; c1 - c0 ]
                then Some O.(x < m "c1" - m "c0")
                else None);
            (* c0*x<c1 -> sign(c0)*x < ceil(c1/abs(c0)) *)
            rule
              Upat.(cvar "c0" * var ~dtype:[ Dtype.Weak_int ] "x" < cvar "c1")
              (fun m ->
                let c0 = num (m "c0") and c1 = num (m "c1") and x = m "x" in
                let a = if V.(c0 < zero) then V.(-c0) else c0 in
                if V.(a > one) then
                  Some
                    O.((if V.(c0 > zero) then x else -x) < lit V.(-(-c1 // a)))
                else None);
            (* x//d<c -> x<c*d for d>0, and -> c*d<x for d<0 *)
            rule
              Upat.(var ~dtype:[ Dtype.Weak_int ] "x" // cvar "d" < cvar "c")
              (fun m ->
                let d = num (m "d")
                and cd = lit V.(num (m "c") * num (m "d"))
                and x = m "x" in
                if V.(d > zero) then Some O.(x < cd)
                else if V.(d < zero) then Some O.(x > cd)
                else None);
            (* Move add/mul consts to end (NOTE: this is still happening before
               constant folding), for integers: it reassociates *)
            rule
              Upat.(var ~dtype:int_or_bool "x" + cvar "c1" + var "y")
              (fun m ->
                let y = m "y" in
                if is_const y then None else Some O.(m "x" + y + m "c1"));
            rule
              Upat.(var ~dtype:int_or_bool "x" * cvar "c1" * var "y")
              (fun m ->
                let y = m "y" in
                if is_const y then None else Some O.(m "x" * y * m "c1"));
            (* Rules from symbolic *)
            (* generic lt folding *)
            rule
              Upat.(var ~dtype:[ Dtype.Weak_int ] "x" < cvar "c")
              (fun m ->
                match num (m "c") with
                | `Int c when Bigint.sign c > 0 -> lt_folding (m "x") c
                | _ -> None);
            rule
              Upat.(
                var ~dtype:[ Dtype.Weak_int ] "x" * int (-1)
                < var "y" * int (-1))
              (fun m -> Some O.(m "y" < m "x"));
            (* canonicalize a simplex with positive coefficients > 0. NOTE: not
               x < 1 means x > 0 *)
            rule
              Upat.(ne (var ~dtype:[ Dtype.Weak_int ] "x" < int 1) (bool true))
              (fun m ->
                Option.map
                  (fun x -> O.(x < int 1 <> bool true))
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
            (* a zero extension commutes with a mask and with a right shift by
               less than the operand's width: a widening cast of an unsigned
               [x land y] or [x lsr k] is the operation on the widened
               operands, so packed values unpack at the width their consumer
               computes in *)
            rule
              Upat.(
                f ~dtype:Dtype.uints ~name:"c"
                  (v ~op:(ops [ Op.And; Op.Shr ]) ~dtype:Dtype.uints ~name:"u"
                     ())
                  Op.Cast)
              (fun m ->
                let c = m "c" and u = m "u" in
                let dt = dtype c and width = Dtype.bitsize (dtype u) in
                let k = nth u 1 in
                let exact =
                  Dtype.bitsize dt > width
                  && (op u = Op.And
                     || V.(vmin k >= zero && vmax k < of_int width))
                in
                if exact then
                  Some (alu (cast (nth u 0) dt) (op u) [ cast k dt ])
                else None);
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
                if
                  List.exists
                    (fun w -> Dtype.equal (dtype w) Dtype.Int64)
                    [ x; y ]
                  && not
                       (List.exists
                          (fun w -> overflows w Dtype.Int32)
                          [ u; x; y ])
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
            (* an AFTER waits only on the effect ops listed here, any other dep
               is replaced by its srcs *)
            rule (Upat.op Op.After ~name:"x") (fun m ->
                let x = m "x" in
                let effects =
                  ops
                    [
                      Op.Range;
                      Op.Store;
                      Op.Call;
                      Op.Barrier;
                      Op.End;
                      Op.Backedge;
                      Op.Linear;
                      Op.Stage;
                    ]
                in
                let deps y =
                  if Op.Set.mem (op y) effects then [ y ] else src y
                in
                Some
                  (replace x
                     ~src:
                       (nth x 0
                       :: dedup (List.concat_map deps (List.tl (src x))))));
            (* after/end with 1 src is just src[0] *)
            rule
              (Upat.v ~op:(ops [ Op.After; Op.End ]) ~src:[ Upat.var "s" ] ())
              (fun m -> Some (m "s"));
            (* ranges can be subbed for CONSTs, remove them from ENDs. BACKEDGE
               conditions are never range selectors. *)
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
