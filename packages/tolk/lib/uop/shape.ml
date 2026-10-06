(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Uop.Node

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

let param_arg_of u =
  match u.arg with
  | Param p -> p
  | _ -> invalid_argf "%s has no ParamArg" (Op.name u.op)

let weak_storage dt =
  if List.mem dt Dtype.weaks then
    invalid_argf "a %s cannot be stored" (repr_dtype dt)

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

let dedup_nodes l =
  Helpers.dedup
    (module struct
      type nonrec t = t

      let equal = ( == )
      let hash u = u.id
    end)
    l

(* A recursive property is filled bottom-up over the nodes that lack it, so a
   deep graph never recurses deeply. *)
let memoized ~calls ~get ~set ~compute u =
  match get u with
  | Some x -> x
  | None ->
      (* A new node is mostly built on nodes that have the property. *)
      let srcs =
        if calls = Skip && u.op = Op.Call then drop 1 u.src else u.src
      in
      if List.for_all (fun s -> Option.is_some (get s)) srcs then
        set u (compute u)
      else
        List.iter
          (fun n -> set n (compute n))
          (toposort ~calls ~gate:(fun n -> Option.is_none (get n)) u);
      Option.get (get u)

(* Simplification *)

(* The symbolic rewrite, installed by [Symbolic] when the library is
   initialised. *)
let simplify_hook : (t -> t) option Atomic.t = Atomic.make None

(* The symbolic rules leave a graph of constants as it is: a sink of constants
   and of stacks of constants is itself, which lets shapes be built before the
   rules are installed. *)
let simplify u =
  let constant s =
    s.op = Op.Const
    || (s.op = Op.Stack && List.for_all (fun c -> c.op = Op.Const) s.src)
  in
  if u.op = Op.Const then u
  else if u.op = Op.Sink && List.for_all constant u.src then u
  else
    match Atomic.get simplify_hook with
    | Some rewrite -> rewrite u
    | None -> invalid_arg "the symbolic rules are not installed"

let resolve ?(default = true) u =
  if not (Dtype.equal u.dtype Dtype.Bool) then
    invalid_argf "only a boolean resolves, not a %s" (repr_dtype u.dtype);
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
    match (r.op, r.src) with
    | Op.Cast, [ ({ op = Op.Const; _ } as c) ] ->
        sint_of_const (Dtype.const r.dtype (value c))
    | Op.Const, _ -> sint_of_const (value r)
    | _ -> None
  in
  match known with Some s -> s | None -> Sym r

let ssimplify_sint = function Int n -> Int n | Sym u -> ssimplify u

let eval u ~kinds ~what =
  if not (List.exists (Dtype.equal u.dtype) kinds) then
    invalid_argf "a %s is not %s" (repr_dtype u.dtype) what;
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
  match u.op with
  | Op.Const -> [ known u ]
  | Op.Stack ->
      List.map (fun s -> if s.op = Op.Const then known s else ssimplify s) u.src
  | _ -> [ ssimplify u ]

let marg u =
  match u.memos.marg_memo with
  | Some m -> m
  | None ->
      let shape_src i = as_shape (nth u i) in
      let m =
        match (u.op, u.arg) with
        | Op.Reshape, _ -> Reshape (shape_src 1)
        | Op.Expand, _ -> Expand (shape_src 1)
        | Op.Pad, _ -> Pad (List.combine (shape_src 1) (shape_src 2))
        | Op.Shrink, _ -> Shrink (List.combine (shape_src 1) (shape_src 2))
        | Op.Permute, Axes l -> Permute l
        | Op.Flip, Flips l -> Flip l
        | op, _ -> invalid_argf "%s is not a movement" (Op.name op)
      in
      u.memos.marg_memo <- Some m;
      m

let marg_shape u =
  match marg u with
  | Reshape s | Expand s -> s
  | _ -> invalid_argf "%s has no shape argument" (Op.name u.op)

let marg_bounds u =
  match marg u with
  | Pad b | Shrink b -> b
  | _ -> invalid_argf "%s has no bounds argument" (Op.name u.op)

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
    ~get:(fun n -> n.memos.shape_memo)
    ~set:(fun n s -> n.memos.shape_memo <- Some s)
    ~compute:compute_shape u

and shape u =
  match shape_opt u with
  | Some s -> s
  | None -> invalid_argf "%s has no shape" (Op.name u.op)

and compute_shape u : sint list option =
  let src0 () = first u.op u.src in
  let void = Dtype.equal u.dtype Dtype.Void in
  match u.op with
  | Op.If | Op.Barrier | Op.Sink | Op.Endif | Op.Backedge | Op.Group | Op.Linear
  | Op.Program | Op.Source | Op.Custom_function ->
      None
  | Op.Call | Op.Ins -> if void then None else Some []
  | Op.Reshape when (src0 ()).op = Op.Noop -> Some (marg_shape u)
  | Op.Noop -> ( match u.src with s :: _ -> shape_opt s | [] -> None)
  | Op.Index ->
      let buf = src0 () and idxs = drop 1 u.src in
      Some (List.concat_map shape idxs @ drop (List.length idxs) (shape buf))
  | Op.Stack -> (
      match u.src with
      | [] -> Some []
      | s :: _ -> Some (Int (List.length u.src) :: shape s))
  | Op.Const | Op.Getaddr | Op.Range | Op.Special -> Some []
  | Op.Binary -> (
      match u.arg with
      | Bytes b -> Some [ Int (String.length b) ]
      | _ -> invalid_arg "binary needs bytes")
  | Op.Buffer | Op.Alloc | Op.Param -> (
      match u.arg with
      | Param { size = None; _ } -> Some []
      | Param { size = Some n; _ } -> Some [ Int n ]
      | _ -> invalid_arg "storage needs a ParamArg")
  | Op.Custom | Op.Customi -> (
      if void then None
      else
        match List.filter_map shape_opt u.src with
        | [] -> None
        | shapes -> Some (broadcast_shape shapes))
  | Op.Stage ->
      let rs = drop 1 u.src in
      Some
        (List.map
           (fun r -> Int (Value.to_int (Value.( + ) (vmax r) (`Int Bigint.one))))
           rs
        @ shape (src0 ()))
  | Op.Wmma -> (
      match u.src with
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
          let out_sz = Dtype.itemsize u.dtype
          and in_sz = Dtype.itemsize (src0 ()).dtype in
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
  | Op.Unshard when List.is_empty u.src -> None
  | op when Op.Set.mem op Op.Set.movement || op = Op.Unshard || op = Op.Reduce
    ->
      let ps =
        match shape_opt (src0 ()) with
        | Some ps -> ps
        | None ->
            invalid_argf "%s needs a shape, and %s has none" (Op.name op)
              (Op.name (src0 ()).op)
      in
      Some (movement_shape u ps)
  | op when Op.Set.mem op Op.Set.unary || op = Op.Cast ->
      (match u.src with
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
          u.src
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
    invalid_argf "invalid %s %s for %s" (Op.name u.op) what (repr_shape ps)
  in
  let ok = Sint.resolve ?default:None in
  (* A size is a number: one that holds Invalid, such as the size of a shrink
     whose bounds carry a validity, evaluates to none. *)
  let numbers s =
    List.iter
      (function
        | Sym d when List.exists is_invalid (toposort ~calls:Enter d) ->
            invalid_argf "%s of sizes %s, one holding Invalid, which is no number"
              (Op.name u.op) (repr_shape s)
        | _ -> ())
      s;
    s
  in
  match u.op with
  | Op.Unshard -> (
      match u.arg with
      | Axes axes ->
          let ranges = drop 1 u.src in
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
      match u.arg with
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
      if not (Dtype.is_int x.dtype) then
        invalid_argf "a shape holds integers, not %s" (repr_dtype x.dtype))
    src;
  match src with [ x ] -> x | src -> v Op.Stack ~src

let mop u (m : movement) =
  let simplified args =
    (simplify (sink (List.map shape_to_shape_arg args))).src
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
  let ret = const ?dtype:(Some (Option.value dtype ~default:u.dtype)) c in
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
let vconst_like u c = broadcast (const ~dtype:u.dtype c) (max_numel u)

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
  let value = identity_element op u.dtype in
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
    let value = identity_element op u.dtype in
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

let rec axis u =
  match u.memos.axis_memo with
  | Some a -> a
  | None ->
      let a = compute_axis u in
      u.memos.axis_memo <- Some a;
      a

and compute_axis u =
  let src0 () = first u.op u.src in
  match u.op with
  | Op.Copy | Op.Param -> None
  | Op.Unshard -> (
      match u.arg with
      | Axes [ a ] -> Some a
      | Axes l ->
          invalid_argf "the value is sharded on several axes, %s"
            (repr_tuple (List.map string_of_int l))
      | _ -> invalid_arg "an unshard needs its axes")
  | op when Op.Set.mem op Op.Set.alu || op = Op.Stack -> (
      let n = ndim u in
      let axes =
        List.filter_map
          (fun x -> Option.map (fun a -> a + n - ndim x) (axis x))
          u.src
      in
      match List.rev (Helpers.dedup (module Int) axes) with
      | [] -> None
      | last :: _ -> Some last)
  | _ when List.is_empty u.src -> None
  | op -> (
      let src_axis = axis (src0 ()) in
      match (op, src_axis) with
      | Op.Shrink, Some a ->
          let o, sz = List.nth (marg_bounds u) a in
          if
            equal_sint o (Int 0) && equal_sint sz (List.nth (shape (src0 ())) a)
          then Some a
          else None
      | Op.Reduce, a -> (
          match (a, u.arg) with
          | None, _ -> None
          | Some a, Reduce { num_axes; _ } ->
              if a < num_axes then None else Some (a - num_axes)
          | _ -> invalid_arg "a reduction needs its argument")
      | Op.Reshape, None -> None
      | Op.Reshape, Some a -> Some (reshape_axis u a)
      | Op.Permute, a -> (
          match (a, marg u) with
          | Some a, Permute order -> List.find_index (Int.equal a) order
          | _ -> None)
      | Op.Expand, a -> (
          match (a, marg u) with
          | Some a, Expand s -> Some (a + List.length s)
          | _ -> None)
      | _, a -> a)

(* The new axis is the last one before which the element count is the count
   before the source's axis, and it must not move elements between shards. *)
and reshape_axis u src_axis =
  let src = first u.op u.src in
  let new_shape = marg_shape u in
  let prefix =
    List.rev
      (List.fold_left
         (fun acc s -> Sint.(List.hd acc * s) :: acc)
         [ Int 1 ] new_shape)
  in
  let acc = List.map ssimplify_sint prefix in
  let target = ssimplify_sint (Sint.prod (take src_axis (shape src))) in
  let moved () =
    invalid_argf "a reshape of %s to %s moves elements between shards"
      (repr_shape (shape src))
      (repr_shape (shape u))
  in
  let rec last_index i best = function
    | [] -> best
    | x :: rest ->
        last_index (i + 1) (if equal_sint x target then Some i else best) rest
  in
  let new_axis =
    match last_index 0 None acc with Some i -> i | None -> moved ()
  in
  let dcount =
    match device u with
    | Some (Multi ds) -> List.length ds
    | _ -> (
        match
          List.find_opt (fun n -> n.op = Op.Unshard) (toposort ~calls:Enter src)
        with
        | Some un -> Value.to_int (vmax (nth un 1)) + 1
        | None -> moved ())
  in
  if Sint.truth Sint.(List.nth (shape u) new_axis % Int dcount <> Int 0) then
    moved ();
  new_axis

let shard_count u =
  if u.op = Op.Unshard then Value.to_int (vmax (nth u 1)) + 1
  else
    match device u with
    | Some (Multi ds) -> List.length ds
    | _ -> invalid_arg "the value is not on several devices"

let bounds u =
  match axis u with
  | None -> invalid_arg "bounds need a sharded value"
  | Some a ->
      let size = List.nth (shape (first u.op u.src)) a in
      let starts =
        List.rev
          (List.fold_left
             (fun acc _ -> Sint.(List.hd acc + size) :: acc)
             [ Int 0 ]
             (List.init (shard_count u) Fun.id))
      in
      let rec pairs = function
        | x :: (y :: _ as rest) -> (x, y) :: pairs rest
        | _ -> []
      in
      pairs starts

let shard_shape u =
  match device u with
  | Some (Multi _) -> (
      match axis u with
      | Some a ->
          let n = shard_count u in
          List.mapi
            (fun i x -> if i = a then Sint.(x // Int n) else x)
            (shape u)
      | None -> shape u)
  | _ -> shape u

let max_shard_shape u = to_max_shape (shard_shape u)

let unshard ?ranges u axes =
  let ranges =
    match ranges with
    | Some r -> r
    | None -> (
        match device u with
        | Some (Multi ds) ->
            [ range ~axis_type:Axis_type.Device (Int (List.length ds)) [ -1 ] ]
        | _ -> invalid_arg "an unshard needs a value on several devices")
  in
  if
    List.length axes <> List.length ranges
    || List.length (List.sort_uniq Int.compare axes) <> List.length axes
  then invalid_arg "an unshard needs one range per distinct axis";
  let pairs =
    List.stable_sort
      (fun (a, _) (b, _) -> Int.compare a b)
      (List.combine axes ranges)
  in
  v Op.Unshard ~src:(u :: List.map snd pairs) ~arg:(Axes (List.map fst pairs))

let shard_slice u a rng =
  match shape u with
  | [] -> u
  | s ->
      let dcount = Value.to_int (vmax rng) + 1 in
      let size = List.nth s a in
      if Sint.truth Sint.(size % Int dcount <> Int 0) then
        invalid_argf "axis %d of size %s does not split over %d devices" a
          (repr_sint size) dcount;
      let sz = Sint.(size // Int dcount) in
      let r = Sym rng in
      shrink u
        (List.mapi
           (fun i x ->
             if i = a then Some (Sint.(r * sz), Sint.((r * sz) + sz))
             else Some (Int 0, x))
           s)

let shard ?axis u devices =
  let copied = copy_to_device u (Multi devices) in
  match axis with
  | None -> copied
  | Some a ->
      let rng =
        range ~axis_type:Axis_type.Device (Int (List.length devices)) [ -1 ]
      in
      unshard (shard_slice copied a rng) [ a ]

(* Storage *)

let empty ?device new_shape dt =
  weak_storage dt;
  let max_shape = to_max_shape new_shape in
  let u =
    v Op.Alloc ~src:(device_range_src device)
      ~arg:
        (Param
           (param_arg ~slot:(unique_num ()) ~size:(size_of max_shape) ?device
              ~bind_on_realize:true dt))
  in
  shrink_to
    (reshape u (List.map (fun n -> Int n) max_shape))
    (List.map Option.some new_shape)

let view_as ?axis u new_shape =
  let max_shape = List.map (fun n -> Int n) (to_max_shape new_shape) in
  let ret = if List.length new_shape > 1 then reshape u max_shape else u in
  let ret =
    if equal_shape max_shape new_shape then ret
    else shrink_to ret (List.map Option.some new_shape)
  in
  match axis with None -> ret | Some a -> unshard ret [ a ]

let empty_like ?dtype ?device:dev u =
  let dev = match dev with Some d -> Some d | None -> device u in
  let dtype = match dtype with Some dt -> dt | None -> commit_dtype u in
  match dev with
  | Some (Multi _) when Option.is_some (axis u) ->
      unshard (empty ?device:dev (shard_shape u) dtype) [ Option.get (axis u) ]
  | _ -> empty ?device:dev (shape u) dtype

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
    | Some d, Some d' when equal_device d d' -> u
    | _, Some d' -> copy_to_device u d'
    | _, None -> u
  in
  after ret [ store ret (cast src ret.dtype) ]

let alloc ?slot ?(addrspace = Dtype.Global) ?device ?axis new_shape dt =
  let slot = match slot with Some s -> s | None -> unique_num () in
  let ret =
    v Op.Alloc ~src:(device_range_src device)
      ~arg:
        (Param
           (param_arg ~slot
              ~size:(size_of (to_max_shape new_shape))
              ~addrspace:(Some addrspace) ?device (Dtype.strong dt)))
  in
  if List.is_empty new_shape then reshape ret []
  else view_as ?axis ret new_shape

let alloc_like ?slot ?addrspace u =
  alloc ?slot ?addrspace (List.map (fun n -> Int n) (max_shard_shape u)) u.dtype

let placeholder ?slot ?(addrspace = Dtype.Global) ?device ?(volatile = false)
    ?tag new_shape dt =
  let dt = Dtype.strong dt in
  let slot = match slot with Some s -> s | None -> unique_num () in
  let name = match tag with Some (Tag.String s) -> Some s | _ -> None in
  let size = size_of new_shape in
  let ret =
    match addrspace with
    | Dtype.Global ->
        v Op.Param
          ~arg:
            (Param
               (param_arg ~slot ~size ?name ~addrspace:(Some addrspace) ?device
                  ~volatile dt))
    | Dtype.Local | Dtype.Reg ->
        if Option.is_some device then
          invalid_arg "workgroup and register storage has no device";
        v Op.Buffer
          ~arg:
            (Param (param_arg ~slot ~size ?name ~addrspace:(Some addrspace) dt))
    | Dtype.Alu -> invalid_arg "a placeholder cannot be a scalar variable"
  in
  let ret = match tag with Some g -> rtag ~tag:g ret | None -> ret in
  if List.length new_shape > 1 then
    reshape ret (List.map (fun n -> Int n) new_shape)
  else ret

let placeholder_like ?addrspace u slot =
  if not (List.for_all (function Int _ -> true | Sym _ -> false) (shape u))
  then invalid_arg "a placeholder needs a shape of known sizes";
  placeholder ~slot ?addrspace (max_shard_shape u) u.dtype

let param ?shape:new_shape ?device ?vmin_vmax ?multiple_of ?name
    ?(addrspace = Some Dtype.Global) ?(volatile = false) ?phase ?align slot dt
    =
  weak_storage dt;
  let make size =
    v Op.Param
      ~arg:
        (Param
           (param_arg ?size ?vmin_vmax ?multiple_of ?name ~addrspace ?device
              ~volatile ?phase ?align ~slot dt))
  in
  match new_shape with
  | None | Some [] -> make None
  | Some s -> view_as (make (Some (size_of (to_max_shape s)))) s

(* Variables *)

let is_variable u =
  u.op = Op.Param
  && (match u.arg with
    | Param { vmin_vmax = Some _; addrspace = Some Dtype.Alu; _ } -> true
    | _ -> false)
  && match shape_opt u with Some [] -> true | _ -> false

let is_bound_var u =
  is_variable u
  && match u.arg with Param { bound = Some _; _ } -> true | _ -> false

(* Divisibility *)

let rec divides u (n : Bigint.t) =
  if Bigint.equal n Bigint.one then Some u
  else
    match u.op with
    | Op.Const ->
        let x = as_value (value u) and n = `Int n in
        if Value.(x % n = of_int 0) then
          Some (const_like u (Value.(x // n) :> Dtype.const))
        else None
    | Op.Stack ->
        let srcs = List.map (fun s -> divides s n) u.src in
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
        match u.arg with
        | Param { multiple_of = Some m; _ } ->
            if Bigint.equal (Bigint.rem (Bigint.of_int m) n) Bigint.zero then
              Some (div ~rounding:`Floor u (const (`Int n)))
            else None
        | _ -> None)
    | _ -> None

(* Where a stage's view of a buffer, through movements and bitcasts, starts if
   it can be a contiguous run of the buffer: the buffer's alignment and the byte
   its first element lies at, an index taken through each movement to its
   source's. A run's last element lies as many elements past its first as the
   view has elements less one; a view whose first or last element is padding,
   or whose ends lie otherwise, is no run. *)
type view_start =
  | Start of int * int
  | Unknown (* A size or a bound is symbolic, or the buffer is sharded. *)
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
    match (u.op, u.src) with
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
          | Flip f ->
              List.map2
                (fun (i, n) f -> if f then n - 1 - i else i)
                (List.combine idx src) f
        in
        go x idx
    | Op.Stage, x :: _ when view_start x <> No_view ->
        raise_notrace (Stop Unknown)
    | Op.Unshard, _ -> raise_notrace (Stop Unknown)
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
    match (v.op, v.src) with
    | (Op.Buffer | Op.Param | Op.Alloc | Op.Stage), _ -> true
    | (Op.Bitcast | Op.Reshape | Op.After | Op.Mselect), v :: _ -> whole v
    | _ -> false
  in
  let ints l = List.for_all (function Int _ -> true | Sym _ -> false) l in
  match (u.op, u.src) with
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
  | (Op.Bitcast | Op.After | Op.Mselect), x :: _ -> storage_phase x
  | o, x :: _ when Op.Set.mem o Op.Set.movement -> storage_phase x
  | Op.Stage, x :: _ when x.op = Op.Bitcast || Op.Set.mem x.op Op.Set.movement
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

let param_like u slot =
  match u.op with
  | Op.Param when addrspace u = Some Dtype.Alu ->
      let p = param_arg_of u in
      v Op.Param ~arg:(Param { p with slot; name = None; bound = None })
  | _ -> (
      let a = axis u and align, phase = storage_phase u in
      match (a, device u) with
      | Some a, Some (Multi _ as d) ->
          let ss = shard_shape u in
          view_as ~axis:a
            (v Op.Param
               ~arg:
                 (Param
                    (param_arg ~slot
                       ~size:(size_of (to_max_shape ss))
                       ~device:d ~phase ~align u.dtype)))
            ss
      | _ ->
          param ?shape:(shape_opt u) ?device:(device u) ~phase ~align slot
            u.dtype)

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
        (const ~dtype:first_u.dtype
           (`Int (List.fold_left Bigint.gcd Bigint.zero factors)))
        (elements common)

let rec divide_exact u d =
  if u == d then Some (const_like u (`Int Bigint.one))
  else if d.op = Op.Const then divides u (Value.to_z (as_value (value d)))
  else
    match u.op with
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
        match n.op with
        | Op.Const -> as_value (value n)
        | Op.Param when addrspace n = Some Dtype.Alu || is_variable n -> (
            let name = expr n in
            match List.assoc_opt name vars with
            | Some x -> `Int (Bigint.of_int x)
            | None -> invalid_argf "the variable %s has no value" name)
        | Op.Cast -> sym_cast n.dtype (get (first n.op n.src))
        | Op.Bitcast ->
            Dtype.bitcast (first n.op n.src).dtype n.dtype
              (get (first n.op n.src))
        | op when Op.Set.mem op Op.Set.alu ->
            sym_alu op n.dtype (List.map get n.src)
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
        match n.op with
        | Op.Const ->
            let v = as_value (value n) in
            fun _ -> v
        | Op.Param when addrspace n = Some Dtype.Alu || is_variable n ->
            let read = var n in
            fun env -> `Int (Bigint.of_int (read env))
        | Op.Cast ->
            let x = compute (first n.op n.src) in
            fun env -> sym_cast n.dtype (x env)
        | Op.Bitcast ->
            let src = first n.op n.src in
            let x = compute src in
            fun env -> Dtype.bitcast src.dtype n.dtype (x env)
        | op when Op.Set.mem op Op.Set.alu ->
            let xs = List.map compute n.src in
            fun env -> sym_alu op n.dtype (List.map (fun x -> x env) xs)
        | op -> fun _ -> invalid_argf "%s cannot be evaluated" (Op.name op)
      in
      (* An integer node of integer sources, computed on [int]s. *)
      let ints = Tbl.create 16 in
      let rec int n = memo ints native n
      and native n =
        if not (Dtype.is_int n.dtype) then None
        else
          match (n.op, n.src) with
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

let bind var x =
  if (not (is_variable var)) || is_bound_var var then
    invalid_argf "only an unbound variable binds, not %s" (Op.name var.op);
  let c = const (x :> Dtype.const) in
  if not (Value.( <= ) (vmin var) (vmin c) && Value.( <= ) (vmax c) (vmax var))
  then
    invalid_argf "%s is out of [%s, %s]" (repr_const x)
      (repr_const (vmin var))
      (repr_const (vmax var));
  let p = param_arg_of var in
  let multiple = Option.value p.multiple_of ~default:1 in
  if Option.is_none (divides c (Bigint.of_int multiple)) then
    invalid_argf "%s is not a multiple of %d" (repr_const x) multiple;
  replace var ~arg:(Param { p with bound = Some x })

let unbound var =
  if not (is_variable var) then
    invalid_argf "%s is not a variable" (Op.name var.op);
  replace var ~arg:(Param { (param_arg_of var) with bound = None }) ~tag:None

let unbind var =
  match (is_bound_var var, var.arg) with
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
           if x.op = Op.Param && addrspace x = Some Dtype.Alu then
             Some (if is_variable x then unbound x else x)
           else if
             x.op = Op.Range && Axis_type.equal (axis_type x) Axis_type.Device
           then
             Some (variable ~dtype:x.dtype "_device_num" (`Int Bigint.zero) (vmax x))
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

let call_with_outputs ?name ?(precompile = false) ?aux ?output_pos values args =
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
    let axis = match device o with Some (Multi _) -> axis o | _ -> None in
    let buf = alloc (shard_shape o) o.dtype ?device:dev ?axis in
    let resolved =
      List.map
        (function
          | Int k -> Int k
          | Sym s ->
              Sym
                (graph_rewrite ~calls:Skip ~pass:Once ~ctx:params s
                   (After_sources pm_resolve_params)))
        (shard_shape o)
    in
    ( alloc resolved o.dtype ~slot:(param_arg_of (buf_uop buf)).slot ?device:dev
        ?axis,
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
  let c = call body call_args ?name ~precompile ?aux in
  List.map (fun (r, _) -> after r [ c ]) outputs

let call_with_output ?name ?precompile value args =
  List.hd (call_with_outputs ?name ?precompile [ value ] args)

let custom_kernel args f =
  let placeholders = List.mapi (fun i s -> placeholder_like s i) args in
  let kernel = call (f placeholders) args in
  List.map (fun s -> after s [ kernel ]) args

(* Programs *)

let program_info_of_sink
    ?(target =
      Helpers.Target.
        { device = ""; renderer = ""; arch = ""; interface = ""; indices = "" })
    sink =
  let vars = ref [] and globals = ref [] and outs = ref [] and ins = ref [] in
  let global_size = Array.make 3 (Int 1)
  and local_size = Array.make 3 (Int 1) in
  (match sink.arg with
  | Kernel { split = Some s; _ } -> global_size.(0) <- s.iterations
  | _ -> ());
  List.iter
    (fun u ->
      if u.op = Op.Param then
        if addrspace u = Some Dtype.Alu then vars := u :: !vars
        else globals := (param_arg_of u).slot :: !globals;
      if u.op = Op.Store || u.op = Op.Load then begin
        let s0 = nth u 0 in
        let idx =
          if s0.op = Op.Index || s0.op = Op.Shrink then Some s0
          else if s0.op = Op.Cast && (nth s0 0).op = Op.Index then
            Some (nth s0 0)
          else None
        in
        match idx with
        | Some idx ->
            let buf = buf_uop (nth idx 0) in
            if buf.op = Op.Param then
              let slot = (param_arg_of buf).slot in
              if u.op = Op.Store then outs := slot :: !outs
              else ins := slot :: !ins
        | None -> ()
      end;
      if u.op = Op.Special then
        match u.arg with
        | String name ->
            let axis =
              Char.code name.[String.length name - 1] - Char.code '0'
            in
            let sizes = if name.[0] = 'l' then local_size else global_size in
            sizes.(axis) <- ssimplify (nth u 0)
        | _ -> invalid_arg "a hardware index needs a name")
    (toposort ~calls:Enter sink);
  let sorted l = List.sort_uniq Int.compare l in
  let outs, ins =
    if List.is_empty !outs && List.is_empty !ins then (!globals, !globals)
    else (!outs, !ins)
  in
  let vars =
    List.stable_sort
      (fun a b -> Int.compare (param_arg_of a).slot (param_arg_of b).slot)
      (dedup_nodes (List.rev !vars))
  in
  {
    global_size = Array.to_list global_size;
    local_size = Array.to_list local_size;
    vars;
    globals = sorted !globals;
    outs = sorted outs;
    ins = sorted ins;
    target;
  }

let launch_dims (p : program_info) vars =
  ( List.map (fun s -> sym_infer s vars) p.global_size,
    List.map (fun s -> sym_infer s vars) p.local_size )

(* Late bindings *)

module Private = struct
  let set_once slot what f =
    if not (Atomic.compare_and_set slot None (Some f)) then
      invalid_argf "the %s are set already" what

  let set_symbolic pm =
    set_once simplify_hook "symbolic rules" (fun u ->
        graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() u (After_sources pm))
end
