(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Shape
module V = Dtype.Value

let strf = Printf.sprintf

(* An optimisation that does not apply to a kernel refuses with a reason;
   [Scheduler.apply_opt] returns it. *)
exception Refused of string

let check cond msg = if not cond then raise_notrace (Refused msg)

let role : Opt.target -> Axis_type.t = function
  | Upcast -> Upcast
  | Local -> Local

let split_targets : Opt.target -> Axis_type.t list = function
  | Upcast -> [ Global; Local; Weak ]
  | Local -> [ Global; Weak ]

let pp_types ppf ts =
  Format.pp_print_list
    ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ", ")
    Axis_type.pp ppf ts

(* The size of a range, its greatest value plus one, exactly: it may exceed an
   int. *)
let size r = Bigint.succ (V.to_z (vmax r))
let z n = const (`Int n)
let sint_of_z n = if Bigint.fits_int n then Int (Bigint.to_int n) else Sym (z n)

(* A size is known when it is an integer, which a constant beyond an int also
   is. *)
let known_size = function
  | Int n -> Some (Bigint.of_int n)
  | Sym u when is_literal u -> (
      match value u with `Int n -> Some n | _ -> None)
  | Sym _ -> None

let mem_ranges r u = Nodes.mem r (ranges u)
let special_name u = match arg u with String s -> s | _ -> ""

let misses r e =
  match (op e, src e) with
  | Op.Store, at :: _ -> not (mem_ranges r at)
  | _ -> not (List.exists (mem_ranges r) (src_without_body e))

(* Lanes and threads

   Which ranges take no lanes, because lanes share their thread's registers,
   and which weak ranges take no threads, because threads run unordered with
   the code outside them (the Rewriting section of postrange.mli). *)

let is_storage u =
  match (op u, arg u) with
  | (Op.Param | Op.Buffer | Op.Alloc), Param p -> p.addrspace <> Dtype.Alu
  | _ -> false

let is_reg u =
  match (op u, arg u) with
  | (Op.Param | Op.Buffer | Op.Alloc), Param p -> p.addrspace = Dtype.Reg
  | _ -> false

(* The ranges [roots] depend on in data. A register cell holds what the same
   step stored, so a load of one depends on its offsets and gate alone. *)
let rec data_ranges roots =
  let seen = Tbl.create 64 and acc = ref [] in
  let add r = if not (List.memq r !acc) then acc := r :: !acc in
  let rec go u =
    if not (Tbl.mem seen u) then begin
      Tbl.add seen u ();
      match (op u, src u) with
      | Op.Range, s ->
          add u;
          List.iter go s
      | Op.Index, _ :: offsets when is_reg (buf_uop u) -> List.iter go offsets
      | Op.After, first :: _ when not (is_group u) -> go first
      | Op.Reduce, value :: closed ->
          List.iter
            (fun r -> if not (List.memq r closed) then add r)
            (data_ranges [ value ])
      | _ -> List.iter go (src_without_body u)
    end
  in
  List.iter go roots;
  !acc

(* The ranges a register store's cells, value or gate depend on, or the
   condition that repeats a loop: none of them takes lanes, nor do those a
   loop's bound reads ({!Scheduler.read_by_bounds}). *)
let lane_bound slice =
  let roots =
    List.fold_left
      (fun acc u ->
        match (op u, src u) with
        | Op.Store, at :: rest when op at = Op.Index && is_reg (buf_uop at)
          ->
            List.tl (src at) @ rest @ acc
        | Op.Backedge, [ _; _; cond ] -> cond :: acc
        | _ -> acc)
      [] slice
  in
  data_ranges roots

(* The storage the pointer [a] points into, if it is one. *)
let pointee a =
  match op a with
  | Op.Index | Op.After | Op.Param | Op.Buffer | Op.Alloc ->
      let s = buf_uop a in
      if is_storage s then Some s else None
  | _ -> None

let is_effect u =
  match op u with Op.Store | Op.Call | Op.Custom -> true | _ -> false

(* The storage each of [effects] writes. A call writes the arguments its body
   writes ({!Shape.written_params}), code every pointer it is passed. *)
let writers effects =
  List.concat_map
    (fun u ->
      let by a = Option.map (fun s -> (s, u)) (pointee a) in
      match (op u, src u) with
      | Op.Store, at :: _ -> Option.to_list (by at)
      | Op.Call, _ ->
          let args = src_without_body u in
          let written =
            match written_params (body u) with
            | Some w -> List.filteri (fun i _ -> List.mem i w) args
            | None -> args
          in
          List.filter_map by written
      | Op.Custom, args -> List.filter_map by args
      | _ -> [])
    effects

(* The storage among [stored] that the effect [e] reads in data: through a
   load or a pointer it passes, following an [After]'s first source. *)
let reads stored e =
  let seen = Tbl.create 16 and acc = ref [] in
  let note s =
    if List.memq s stored && not (List.memq s !acc) then acc := s :: !acc
  in
  let rec go u =
    if not (Tbl.mem seen u) then begin
      Tbl.add seen u ();
      match (op u, src u) with
      | Op.Index, _ :: offsets ->
          note (buf_uop u);
          List.iter go offsets
      | Op.After, first :: _ when not (is_group u) -> go first
      | (Op.Param | Op.Buffer | Op.Alloc), _ -> if is_storage u then note u
      | _ -> List.iter go (src_without_body u)
    end
  in
  (match (op e, src e) with
  | Op.Store, at :: rest ->
      if op at = Op.Index then List.iter go (List.tl (src at));
      List.iter go rest
  | _ -> List.iter go (src_without_body e));
  !acc

(* The weak ranges among [rngs] that take no threads: an effect that depends
   on one and an effect that does not access one storage, one of them
   writing it; for a register, the one that does not reads what the other
   writes. *)
let thread_bound slice rngs =
  let effects = List.filter is_effect slice in
  (* Only a range that some effects depend on and others do not stays
     serial. *)
  let parts r =
    axis_type r = Axis_type.Weak
    && List.exists (mem_ranges r) effects
    && not (List.for_all (mem_ranges r) effects)
  in
  let weak = List.filter parts rngs in
  let writes = if weak = [] then [] else writers effects in
  if writes = [] then []
  else
    let stored = List.map fst writes in
    let read x = op x = Op.Index && List.memq (buf_uop x) stored in
    (* Effects' reads are walked only where stored storage is read. *)
    let read_in u =
      match (op u, src u) with
      | (Op.Call | Op.Custom | Op.Customi), _ -> true
      | Op.Store, _ :: rest -> List.exists read rest
      | _, s -> List.exists read s
    in
    let reads =
      if not (List.exists read_in slice) then []
      else
        List.concat_map
          (fun e -> List.map (fun s -> (s, e)) (reads stored e))
          effects
    in
    let meets r (s, w) =
      let shared = not (is_reg s) and dep = mem_ranges r w in
      let across (s', e) = s' == s && mem_ranges r e <> dep in
      (shared && List.exists across writes)
      || ((shared || dep) && List.exists across reads)
    in
    List.filter (fun r -> List.exists (meets r) writes) weak

module Scheduler = struct
  (* What the axis queries read of a kernel. The heuristic and the search ask
     them many times between two rewrites of the kernel, so they are computed
     once per kernel. *)
  type view = {
    of_ast : Ops.t;
    slice : Ops.t list;
    reduceops : Ops.t list;
    reduce_ranges : Ops.t list;
    rngs : Ops.t list;
    read_by_bounds : Ops.t list;
    opaque : Ops.t list;
    read_by_opaque : Ops.t list;
    unread_by_opaque : Ops.t list;
    lane_bound : Ops.t list;
    mutable threadless : Ops.t list option;
    mutable full_shape : sint list option;
  }

  type t = {
    mutable ast : Ops.t;
    ren : Renderer.t;
    mutable applied_opts : Opt.t list;
    mutable next_range : int;
    mutable view : view option;
  }

  (* Whether the ranges each reduction and end of [ast] closes are distinct
     ranges, which {!Simplify.pm_flatten_range} leaves as they are: checking
     costs a copy less than rewriting its kernel. *)
  let flat ast =
    let rec distinct = function
      | [] -> true
      | r :: rest ->
          op r = Op.Range && (not (List.memq r rest)) && distinct rest
    in
    let closes_ranges u =
      match op u with
      | Op.Reduce | Op.End ->
          distinct (List.drop (Option.get (range_start (op u))) (src u))
      | _ -> true
    in
    Nodes.fold
      (fun u ok -> ok && closes_ranges u)
      (backward_slice ~calls:Skip ast)
      true

  let v ast ren =
    let ast =
      if flat ast then ast
      else
        graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() ast
          (After_sources Simplify.pm_flatten_range)
    in
    let applied_opts =
      match arg ast with Kernel k -> k.applied_opts | _ -> []
    in
    let ids =
      List.filter_map
        (fun u -> if op u = Op.Range then Some (List.hd (axis_id u)) else None)
        (Nodes.to_list (backward_slice ~calls:Skip ast))
    in
    let last = match ids with [] -> 0 | i :: is -> List.fold_left max i is in
    { ast; ren; applied_opts; next_range = last + 1; view = None }

  let copy k =
    let c = v k.ast k.ren in
    c.applied_opts <- k.applied_opts;
    c

  let ast k = k.ast
  let ren k = k.ren
  let applied_opts k = k.applied_opts

  let view_of ast =
    let slice = Nodes.to_list (backward_slice ~calls:Skip ast) in
    let reduceops = List.filter (fun u -> op u = Op.Reduce) slice in
    (* The ranges the reductions close: their sources, which the scheduler
       keeps flat. *)
    let red =
      Helpers.dedup (module Ops)
        (List.concat_map (fun u -> List.tl (src u)) reduceops)
    in
    let axis u =
      op u = Op.Range
      && (not (Dtype.equal (dtype u) Dtype.Void))
      && V.(vmax u > of_int 0)
    in
    (* Keep serial/vectorized reduction axes last, deriving their role from
       REDUCE rather than the axis type. void RANGEs are loops, not opt axes.
       the DEVICE axis is launched, not an opt axis *)
    let last u = List.memq u red && List.mem (axis_type u) [ Weak; Upcast ] in
    let rngs =
      List.stable_sort
        (fun a b ->
          match Bool.compare (last a) (last b) with
          | 0 -> compare_range_arg a b
          | c -> c)
        (List.filter axis slice)
    in
    (* The ranges a loop's bound reads, as [i] in [j < x[i]]: a loop has one
       count of trips, so none of them can become lanes. *)
    let read_by_bounds =
      List.fold_left
        (fun acc u ->
          if op u = Op.Range && src u <> [] then
            Nodes.fold
              (fun r acc -> if List.memq r acc then acc else r :: acc)
              (ranges (nth u 0)) acc
          else acc)
        [] slice
    in
    (* The calls and custom code, and the ranges the sources of each read, as
       [i] in [f(&x[i])]: each takes one pointer or value for each source, so
       none of them can become lanes or take padding, nor can those of an
       expression in the target's language. Every thread runs a call or a
       statement, so a range one of them does not read cannot take threads; an
       expression has no effect, and every thread may compute it. *)
    let opaque =
      List.filter (fun u -> op u = Op.Call || op u = Op.Custom) slice
    in
    let expressions = List.filter (fun u -> op u = Op.Customi) slice in
    let reads u =
      List.fold_left
        (fun acc a ->
          Nodes.fold
            (fun r acc -> if List.memq r acc then acc else r :: acc)
            (ranges a) acc)
        [] (src_without_body u)
    in
    let read_by_opaque =
      List.fold_left
        (List.fold_left (fun acc r -> if List.memq r acc then acc else r :: acc))
        [] (List.map reads (opaque @ expressions))
    in
    let unread_by_opaque =
      if opaque = [] then []
      else
        List.filter
          (fun r -> op r = Op.Range && List.exists (misses r) opaque)
          slice
    in
    let lane_bound =
      if List.exists (fun u -> is_reg u || op u = Op.Backedge) slice then
        lane_bound slice
      else []
    in
    {
      of_ast = ast;
      slice;
      reduceops;
      reduce_ranges = red;
      rngs;
      read_by_bounds;
      opaque;
      read_by_opaque;
      unread_by_opaque;
      lane_bound;
      threadless = None;
      full_shape = None;
    }

  let view k =
    match k.view with
    | Some w when w.of_ast == k.ast -> w
    | _ ->
        let w = view_of k.ast in
        k.view <- Some w;
        w

  let slice k = (view k).slice
  let of_op k o = List.filter (fun u -> op u = o) (slice k)
  let reduceops k = (view k).reduceops
  let reduce_ranges k = (view k).reduce_ranges
  let rngs k = (view k).rngs
  let read_by_bounds k = (view k).read_by_bounds
  let opaque k = (view k).opaque
  let read_by_opaque k = (view k).read_by_opaque
  let unread_by_opaque k = (view k).unread_by_opaque
  let lane_bound k = (view k).lane_bound

  (* Computed when a split into threads asks. *)
  let threadless k =
    let w = view k in
    match w.threadless with
    | Some l -> l
    | None ->
        let l = w.unread_by_opaque @ thread_bound w.slice w.rngs in
        w.threadless <- Some l;
        l

  let shape_len k = List.length (rngs k)
  let full_shape k =
    let w = view k in
    match w.full_shape with
    | Some shape -> shape
    | None ->
        let shape = List.map (fun r -> ssimplify (nth r 0)) w.rngs in
        w.full_shape <- Some shape;
        shape
  let axis_types k = List.map axis_type (rngs k)

  let positions p l =
    List.concat (List.mapi (fun i x -> if p x then [ i ] else []) l)

  let axes_of ?reduce k types =
    let red = reduce_ranges k in
    let wanted r =
      (types = [] || List.mem (axis_type r) types)
      && match reduce with None -> true | Some b -> List.memq r red = b
    in
    positions wanted (rngs k)

  let reduceop k = match reduceops k with r :: _ -> Some r | [] -> None
  let bufs k = List.rev (of_op k Op.Index)

  let upcast_size k =
    let shape = full_shape k in
    Sint.prod (List.map (List.nth shape) (axes_of k [ Upcast ]))

  let upcastable_dims ?(reduce = false) k =
    let shape = full_shape k in
    let rngs = rngs k in
    List.filter
      (fun i ->
        let r = List.nth rngs i in
        Option.is_some (known_size (List.nth shape i))
        && (not (List.memq r (read_by_bounds k)))
        && (not (List.memq r (read_by_opaque k)))
        && not (List.memq r (lane_bound k)))
      (axes_of ~reduce k (split_targets Upcast))

  let group_for_reduces k =
    List.length (axes_of ~reduce:true k [ Warp; Local ])

  (* Grouping

     A local split of a reduce axis groups the reductions that close it: each
     thread of the workgroup reduces its part of the axis into a buffer in
     shared memory, which every thread then reduces. *)

  let closers k rng =
    List.filter (fun u -> List.memq rng (List.tl (src u))) (reduceops k)

  let grouped_reduces k =
    let thread r = List.mem (axis_type r) Axis_type.[ Warp; Local ] in
    List.filter (fun u -> List.exists thread (List.tl (src u))) (reduceops k)

  let union us vs = us @ List.filter (fun v -> not (List.memq v us)) vs

  (* The bytes of shared memory the buffers of [grouped] take after a split by
     [amt]. The sizes may exceed an int, so the product is a node. *)
  let smem k amt grouped =
    let shape = full_shape k in
    let lanes =
      List.fold_left
        (fun p a -> mul p (sint_to_uop (List.nth shape a)))
        (z amt)
        (axes_of ~reduce:false k [ Upcast ] @ axes_of k [ Warp; Local ])
    in
    let bytes =
      List.fold_left (fun n u -> n + Dtype.itemsize (dtype u)) 0 grouped
    in
    mul lanes (int bytes)

  let smem_fits k amt grouped =
    to_bool O.(smem k amt grouped <= int k.ren.shared_max)

  (* The reductions no reduce axis of the kernel encloses. *)
  let outermost k =
    let red = List.map (List.nth (rngs k)) (axes_of ~reduce:true k []) in
    List.filter
      (fun u -> not (List.exists (fun r -> mem_ranges r u) red))
      (reduceops k)

  (* [siblings k rng amt grouped] is the axes that a group of [rng] by [amt]
     splits besides [rng], so that its threads share the kernel's independent
     outermost reductions, as a check of many values combines many: without them
     each thread runs the others whole. [rng]'s reductions must be outermost.
     Each other outermost reduction gives its first reduce axis that [amt]
     divides and that only outermost reductions close, while those reductions
     read no grouped one nor are read by one, and the buffers fit in shared
     memory along with [grouped]'s. A reduction that reads another stays out:
     the two final reductions of the shared buffers would run one loop. *)
  let siblings k rng amt grouped =
    let top = outermost k in
    let red = reduce_ranges k in
    let axis_of u =
      List.find_opt
        (fun r ->
          axis_type r = Axis_type.Weak
          && List.memq r red
          && List.memq r (List.tl (src u))
          && Option.is_some (divides (nth r 0) amt)
          && List.for_all (fun c -> List.memq c top) (closers k r))
        (rngs k)
    in
    let independent grouped u =
      not
        (List.exists
           (fun g -> reaches ~calls:Skip u g || reaches ~calls:Skip g u)
           grouped)
    in
    let take (axes, grouped) u =
      match axis_of u with
      | Some r when not (List.memq u grouped || List.memq r axes) ->
          let joining = closers k r in
          let joined = union grouped joining in
          if
            List.for_all (independent grouped) joining && smem_fits k amt joined
          then (r :: axes, joined)
          else (axes, grouped)
      | _ -> (axes, grouped)
    in
    if not (List.for_all (fun u -> List.memq u top) (closers k rng)) then []
    else List.rev (fst (List.fold_left take ([], grouped) top))

  (* Printing *)

  let colored_shape k =
    List.map
      (fun r ->
        Helpers.colored
          (Axis_type.color (axis_type r))
          (strf "%4s" (Render.render (nth r 0))))
      (rngs k)
    |> String.concat " "

  (* Rewriting *)

  let convert_loop_to_global k =
    if k.ren.has_local then begin
      let red = reduce_ranges k in
      let weak =
        List.concat_map
          (fun s ->
            if op s = Op.End then
              List.filter
                (fun r ->
                  axis_type r = Weak
                  && (not (List.memq r red))
                  && known_at_launch r)
                (List.tl (src s))
            else [])
          (src k.ast)
      in
      (* exclude any output ranges from global that don't appear in all
         BUFFERIZE, or that take no threads *)
      let globalizable =
        List.fold_left
          (fun ret x -> List.filter (fun r -> mem_ranges r x) ret)
          weak (of_op k Op.Stage)
      in
      let globalizable =
        List.filter (fun r -> not (List.memq r (threadless k))) globalizable
      in
      let global r =
        (r, replace r ~arg:(Range { axis_type = Global; axis_id = axis_id r }))
      in
      k.ast <-
        substitute ~calls:Skip ~pass:Fixed_point k.ast
          (List.map global
             (List.filter (fun r -> List.memq r globalizable) (rngs k)))
    end

  let new_range k amount dtype t =
    let n = k.next_range in
    k.next_range <- n + 1;
    range ~axis_type:(role t) ~dtype (sint_of_z amount) [ n ]

  (* The size of [rng] divided by [amount], if [rng] can be split for
     [target]. *)
  let split_size k rng amount target =
    if not (List.mem (axis_type rng) (split_targets target)) then
      Error
        (Format.asprintf "a %a axis comes from a %a axis, not a %a one"
           Axis_type.pp (role target) pp_types (split_targets target)
           Axis_type.pp (axis_type rng))
    else if not (is_literal (nth rng 0) || known_at_launch rng) then
      (* Lanes and threads share one count of trips, known when the kernel
         launches: a bound that reads memory or another range, as
         [j < 4 (i + 1)], is not. *)
      Error
        (strf "the trips of %s are not known at launch: it stays serial"
           (Render.render rng))
    else if
      target = Upcast
      && (List.memq rng (read_by_bounds k) || List.memq rng (lane_bound k))
    then
      Error
        (strf
           "a register store or a loop's bound or condition depends on %s: it \
            takes no lanes"
           (Render.render rng))
    else if target = Local && List.memq rng (threadless k) then
      Error
        (strf
           "%s takes no threads: a call or custom code does not read it, or \
            code outside it shares storage with its steps"
           (Render.render rng))
    else
      match divides (nth rng 0) amount with
      | Some s -> Ok s
      | None ->
          Error
            (strf "%s does not divide %s in %s" (Bigint.to_string amount)
               (Render.render (nth rng 0))
               (colored_shape k))

  let flatten = Pattern_matcher.with_ctx Simplify.pm_flatten_range

  let shift_by ?(top = false) ?new_rng k rng amount target old_sz =
    let new_rng =
      match new_rng with
      | Some r -> r
      | None -> new_range k amount (dtype rng) target
    in
    let replaced_rng = replace rng ~src:[ old_sz ] in
    let sub_axis =
      if top then O.((new_rng * old_sz) + replaced_rng)
      else O.((replaced_rng * z amount) + new_rng)
    in
    k.ast <-
      substitute ~extra_pm:flatten ~calls:Skip ~pass:Fixed_point k.ast
        [ (rng, sub_axis) ];
    (replaced_rng, new_rng)

  let shift_to ?top ?new_rng k rng amount target =
    if
      (amount <= 1)
      [@mutate off "a split by 1 also fails, when its rewrite cycles"]
    then invalid_arg (strf "a split takes more than 1, not %d" amount);
    let amount = Bigint.of_int amount in
    match split_size k rng amount target with
    | Ok old_sz -> shift_by ?top ?new_rng k rng amount target old_sz
    | Error msg -> invalid_arg msg

  (* [shift_to], refusing where it cannot split. *)
  let split ?top ?new_rng k rng amount target =
    match split_size k rng amount target with
    | Ok old_sz -> shift_by ?top ?new_rng k rng amount target old_sz
    | Error msg -> raise_notrace (Refused msg)

  (* [tc_operand tc u] is what the core [tc] multiplies for the operand [u]: [u]
     at the core's input type, or the narrow float that [u] widens to float32
     when the core adds float32 products, since it multiplies two narrow floats
     exactly, as float32 does. *)
  let tc_operand (tc : Tc.t) u =
    if Dtype.equal (dtype u) tc.dtype_in then Some u
    else if
      op u = Op.Cast
      && Dtype.equal (dtype (nth u 0)) tc.dtype_in
      && Dtype.is_float tc.dtype_in
      && Dtype.itemsize tc.dtype_in < 4
      && Dtype.equal (dtype u) Dtype.Float32
      && Dtype.equal tc.dtype_out Dtype.Float32
    then Some (nth u 0)
    else None

  (* Whether [u] reads the range [r] only in whole tiles of [dim]: as [r / d]
     for a multiple [d] of [dim]. A tile of [r]'s values then reads one value of
     [u], as an operand reads the axis the core's other operand runs along:

       row     0 .. 15 | 16 .. 31 | ...     rows   x[row]           M
       block       0   |     1    |          weight W[owner[row / 16]]

     with [dim] 16 the core takes 16 rows of [x] against one block's matrix. *)
  let in_tiles u r dim =
    let readers =
      List.filter
        (fun x -> List.memq r (src x))
        (Nodes.to_list (backward_slice_with_self ~calls:Skip u))
    in
    let tile x =
      op x = Op.Div
      && is_literal (nth x 1)
      &&
      match value (nth x 1) with
      | `Int d -> Bigint.(equal (rem d (of_int dim)) zero)
      | _ -> false
    in
    readers <> [] && List.for_all tile readers

  let rec apply ?(append_opt = true) k (opt : Opt.t) =
    let axis_rng axis =
      check
        (0 <= axis && axis < shape_len k)
        (strf "axis %d is not one of the kernel's %d axes" axis (shape_len k));
      List.nth (rngs k) axis
    in
    let ret =
      match opt with
      | Split { axis; amount; target; top } ->
          let rng = axis_rng axis in
          check
            (amount = 0 || amount > 1)
            (strf "a split takes 0 or more than 1, not %d" amount);
          if target = Local then begin
            check k.ren.has_local "locals needed for opt";
            check
              (not (List.memq rng (unread_by_opaque k)))
              (strf "a call or custom code does not read axis %d" axis)
          end;
          let amt = if amount = 0 then size rng else Bigint.of_int amount in
          let is_reduce = List.memq rng (reduce_ranges k) in
          if target = Upcast then begin
            check
              (not (List.memq rng (read_by_bounds k)))
              (strf "a loop's bound reads axis %d" axis);
            check
              (not (List.memq rng (read_by_opaque k)))
              (strf "a call or custom code reads axis %d" axis);
            if is_reduce then
              check Bigint.(leq amt (of_int 32)) "don't unroll more than 32"
            else check Bigint.(leq amt (of_int 16)) "don't upcast more than 16"
          end;
          let groups = target = Local && is_reduce in
          (* A group's threads meet at barriers, which threads running a loop
             of a different count of trips would not all reach. *)
          if groups then
            check
              (List.is_empty (read_by_bounds k))
              "groups need loops whose bounds read no axis";
          let grouped = grouped_reduces k in
          let grouped =
            if groups then union grouped (closers k rng) else grouped
          in
          (* prevents METAL compiler hangs *)
          if groups || group_for_reduces k > 0 then
            check (smem_fits k amt grouped)
              (strf "exceeds maximum shared memory size: needs %s, max %d"
                 (Render.render (smem k amt grouped))
                 k.ren.shared_max);
          if groups then begin
            let red = reduce_ranges k in
            (* An enclosing reduce runs its iterations in turn, each storing
               and reading the shared buffer between barriers; an unrolled one
               would need a buffer per lane. *)
            check
              (not
                 (List.exists
                    (fun u -> List.memq u red && axis_type u = Axis_type.Upcast)
                    (Nodes.to_list (ranges (List.hd (closers k rng))))))
              "cannot have a group inside an unrolled reduce"
          end;
          let shared = if groups then siblings k rng amt grouped else [] in
          let replaced, new_rng = split ~top k rng amt target in
          List.iter
            (fun r -> ignore (split ~top ~new_rng k r amt target))
            shared;
          [ replaced; new_rng ]
      | Tc { axis; tc_select; tc_opt; use_tc } -> (
          check (k.applied_opts = []) "tensor core opts must be first";
          check
            (List.is_empty (read_by_bounds k))
            "tensor cores need loops whose bounds read no axis";
          check
            (List.is_empty (opaque k))
            "tensor cores need a kernel without calls or custom code";
          check
            (List.is_empty (read_by_opaque k))
            "tensor cores need custom expressions that read no axis";
          check (axis >= 0) "tensor core opts must have an axis";
          check
            (-1 <= tc_select && tc_select < List.length k.ren.tensor_cores)
            "tensor core opts must have valid tc_select";
          check
            (0 <= tc_opt && tc_opt <= 2)
            "tensor core opts must have valid tc_opt";
          check
            (0 < use_tc && use_tc <= 2)
            "use_tensor_cores value is not valid";
          match apply_tc_opt k use_tc axis tc_select tc_opt with
          | Some axes -> axes
          | None -> raise_notrace (Refused "no tensor core available"))
      | Padto { axis; amount } ->
          let rng = axis_rng axis in
          check (amount > 1) (strf "padto arg is a multiple > 1, not %d" amount);
          check (is_literal (nth rng 0)) "only pad const axes";
          (* a padded point would run the call or the code on a source of no
             element *)
          check
            (not (List.memq rng (read_by_opaque k)))
            (strf "a call or custom code reads axis %d" axis);
          check
            (not (List.mem (axis_type rng) [ Axis_type.Upcast; Warp ]))
            "cannot pad upcasted or warp";
          let amount = Bigint.of_int amount in
          let sz = size rng in
          let new_sz = Bigint.(cdiv sz amount * amount) in
          check
            Bigint.(gt sz (fdiv new_sz (of_int 4)))
            "pad adds more than quadruple the work";
          let replaced_rng =
            replace rng ~src:[ const_like (nth rng 0) (`Int new_sz) ]
          in
          let valid_ = O.(replaced_rng < z sz) in
          (* Every access along the padded range is guarded, its offsets
             aside: a counter [C[0]] ordered after the range would count the
             padding. *)
          let pad_buf b =
            let i = nth b 1 in
            if mem_ranges rng b then
              [
                ( b,
                  replace b
                    ~src:
                      [ nth b 0; valid (get_idx i) O.(valid_ land get_valid i) ]
                );
              ]
            else []
          in
          let pad_reduce r =
            if List.memq rng (List.tl (src r)) then
              let op =
                match arg r with Reduce { op; _ } -> op | _ -> Op.Add
              in
              let zero =
                const ~dtype:(dtype r) (identity_element op (dtype r))
              in
              [
                ( r,
                  replace r ~src:(where valid_ (nth r 0) zero :: List.tl (src r))
                );
              ]
            else []
          in
          (* An integer division whose divisor reads the padded range divides
             by 1 on the padding: the padding loads zeros, and a division by
             zero traps on x86 even in a lane no store keeps. *)
          let pad_division u =
            match (op u, src u) with
            | (Op.Truncdiv | Truncmod | Div | Mod), [ a; d ]
              when Dtype.is_int (dtype u) && mem_ranges rng d ->
                let one = const (Dtype.const (dtype d) (`Int Bigint.one)) in
                [ (u, replace u ~src:[ a; where valid_ d one ]) ]
            | _ -> []
          in
          k.ast <-
            substitute ~calls:Skip ~pass:Fixed_point k.ast
              (((rng, replaced_rng) :: List.concat_map pad_buf (bufs k))
              @ List.concat_map pad_reduce (reduceops k)
              @ List.concat_map pad_division (toposort ~calls:Skip k.ast));
          [ replaced_rng ]
      | Swap { axis; with_axis } ->
          let rng = axis_rng axis in
          check
            (0 <= with_axis && with_axis < shape_len k)
            (strf "axis %d is not one of the kernel's %d axes" with_axis
               (shape_len k));
          let altrng = List.nth (rngs k) with_axis in
          check
            (axis_type rng = Global && axis_type altrng = Global)
            "swap only for globals";
          let renamed r other =
            replace r
              ~arg:(Range { axis_type = axis_type r; axis_id = axis_id other })
          in
          k.ast <-
            substitute ~calls:Skip ~pass:Once k.ast
              [ (rng, renamed rng altrng); (altrng, renamed altrng rng) ];
          []
    in
    if append_opt then k.applied_opts <- k.applied_opts @ [ opt ];
    ret

  and apply_tc_opt k use_tc axis tc_select opt_level =
    let reduceop =
      match reduceops k with
      | r :: _ -> r
      | [] -> raise_notrace (Refused "no reduce ops for TensorCore")
    in
    let mul =
      let s = nth reduceop 0 in
      if op s = Op.Cast then nth s 0 else s
    in
    match arg reduceop with
    | Reduce { op = Op.Add; _ } when op mul = Op.Mul ->
        let cores =
          if tc_select = -1 then k.ren.tensor_cores
          else [ List.nth k.ren.tensor_cores tc_select ]
        in
        List.find_map (try_core k use_tc axis opt_level reduceop mul) cores
    | _ -> None

  and try_core k use_tc axis opt_level reduceop mul (tc : Tc.t) =
    let in0 = nth mul 0 and in1 = nth mul 1 in
    let device = k.ren.target.device in
    if
      (device = "CUDA" || device = "NV")
      && Dtype.equal tc.dtype_in Dtype.Float32
      && not (Setting.value Setting.allow_tf32)
    then None
    else if
      Option.is_none (tc_operand tc in0)
      || Option.is_none (tc_operand tc in1)
      || not (Dtype.equal tc.dtype_out (dtype reduceop))
    then None
    else
      (* tensor cores have three ranges. X, Y, and REDUCE *)
      let by_id_desc rs =
        List.stable_sort (fun a b -> compare_range_arg b a) rs
      in
      let n_dim, m_dim, _ = Tc.dims tc in
      let only a b dim =
        by_id_desc
          (List.filter
             (fun u -> (not (mem_ranges u b)) || in_tiles b u dim)
             (Nodes.to_list (ranges a)))
      in
      let red_ranges = by_id_desc (List.tl (src reduceop)) in
      (* pick ranges: the core's A runs along M and its B along N, tc.dims
         being (N, M, K). Either operand may be A: in0 first, as tinygrad
         takes it, then in1, whose ranges then run along M. *)
      let roles swapped a b =
        let a_ranges = only a b m_dim and b_ranges = only b a n_dim in
        if Setting.value Setting.debug >= 3 then begin
          let show rs =
            String.concat ", "
              (List.map
                 (fun r ->
                   strf "((%s), %s)"
                     (String.concat ", " (List.map string_of_int (axis_id r)))
                     (Bigint.to_string (size r)))
                 rs)
          in
          Format.eprintf "TC(%d): [%s] [%s] [%s]@." axis (show a_ranges)
            (show b_ranges) (show red_ranges)
        end;
        List.concat_map
          (fun n ->
            List.concat_map
              (fun m -> List.map (fun r -> (swapped, [| n; m; r |])) red_ranges)
              a_ranges)
          b_ranges
      in
      let choices = roles false in0 in1 @ roles true in1 in0 in
      if axis >= List.length choices then None
      else
        let swapped, axes = List.nth choices axis in
        let red = reduce_ranges k in
        check
          (not (List.memq axes.(0) red || List.memq axes.(1) red))
          "tensor core N and M can't be contracted";
        (* do optimizations and save the ranges *)
        let saved = k.ast in
        let warp = range ~axis_type:Warp (Int (Tc.threads tc)) [ -1 ] in
        let dims = match Tc.dims tc with n, m, d -> [| n; m; d |] in
        let shape () =
          Array.iteri
            (fun i a ->
              if
                not
                  (Bigint.equal
                     (Bigint.rem (size a) (Bigint.of_int dims.(i)))
                     Bigint.zero)
              then begin
                if opt_level < 2 then
                  raise_notrace (Refused "tc padding requires opt_level >= 2");
                (* PADTO might fail *)
                let axis = index_of a (rngs k) in
                axes.(i) <-
                  List.hd
                    (apply ~append_opt:false k
                       (Padto { axis; amount = dims.(i) }))
              end)
            axes;
          (* we create the warp as a whole thing, in case some of these ranges
             are moved/removed later *)
          List.map
            (fun c ->
              let d = match c with Tc.N _ -> 0 | M _ -> 1 | K _ -> 2 in
              let replaced, r =
                match index_of_bit c tc.frag_c.lanes with
                | Some j ->
                    split k axes.(d) (Bigint.of_int 2) Local
                      ~new_rng:
                        (let p = 1 lsl j in
                         O.(warp // int p % int 2))
                | None -> split k axes.(d) (Bigint.of_int 2) Upcast
              in
              axes.(d) <- replaced;
              (c, r))
            (Tc.axis_coords tc)
        in
        match shape () with
        | exception Refused _ ->
            k.ast <- saved;
            None
        | ne -> (
            match if use_tc <> 2 then use_wmma k tc ~swapped axes ne with
            | exception Refused _ ->
                k.ast <- saved;
                None
            | () -> Some (Array.to_list axes))

  and use_wmma k (tc : Tc.t) ~swapped axes ne =
    let ne_of c = snd (List.find (fun (b, _) -> Tc.equal_bit b c) ne) in
    let reduceop =
      match
        List.filter
          (fun x -> List.memq axes.(2) (List.tl (src x)))
          (reduceops k)
      with
      | [ r ] -> r
      | rs ->
          raise_notrace
            (Refused
               (strf "%d reductions run over the tensor core's K axis, not one"
                  (List.length rs)))
    in
    let r0 = nth reduceop 0 in
    let gate, mul =
      if op r0 = Op.Where then (Some (nth r0 0), nth r0 1) else (None, r0)
    in
    let mul = if op mul = Op.Cast then nth mul 0 else mul in
    (* An operand read in whole tiles of the other's axis (see [in_tiles]) is
       one value over a tile: its reads of the other's bits, which stay below
       the tile, are read at 0. A reads no N bit, and B no M bit. *)
    let own bit x =
      substitute ~calls:Skip ~pass:Fixed_point (Option.get (tc_operand tc x))
        (List.filter_map
           (fun (c, r) ->
             if bit c then Some (r, const ~dtype:(dtype r) (`Int Bigint.zero))
             else None)
           ne)
    in
    let a, b =
      match src mul with
      | [ x; y ] -> if swapped then (y, x) else (x, y)
      | _ -> invalid_arg "a multiply has two operands"
    in
    let ins =
      [
        own (function Tc.N _ -> true | _ -> false) a;
        own (function Tc.M _ -> true | _ -> false) b;
      ]
    in
    let ins =
      match gate with
      | None -> ins
      | Some g ->
          List.map
            (fun x -> where g x (const ~dtype:(dtype x) (`Int Bigint.zero)))
            ins
    in
    let relabel_a, relabel_b = Tc.relabel tc in
    let srcs =
      List.map2
        (fun x rl ->
          substitute ~calls:Skip ~pass:Once x
            (List.map (fun (a, b) -> (ne_of a, ne_of b)) rl))
        ins [ relabel_a; relabel_b ]
    in
    (* get upcast axes for the tensor cores *)
    let base_upcast_axes =
      List.map
        (fun c ->
          let r = ne_of c in
          (axis_type r, axis_id r))
        (Tc.base_upcast_axes tc)
    in
    let cnt (f : Tc.fragment) = List.length f.elements in
    let ca = cnt tc.frag_a and cb = cnt tc.frag_b and cc = cnt tc.frag_c in
    (* each operand upcasts its first upcast_cnt axes, the axes only A or B
       upcast are size 1 so the operands broadcast *)
    let upcast c =
      List.filteri (fun j _ -> j < max c (max ca cb)) base_upcast_axes
      |> List.mapi (fun j a -> (a, if j < c then 2 else 1))
    in
    let acc =
      consts ~dtype:tc.dtype_out (List.init (1 lsl cc) (fun _ -> `Float 0.))
    in
    let tc_uop =
      wmma
        ~upcast_axes:(upcast ca, upcast cb, upcast cc)
        (List.nth srcs 0) (List.nth srcs 1) ~acc ~dims:(Tc.dims tc)
        ~threads:(Tc.threads tc)
    in
    (* preserve extra reduces *)
    let k_ranges =
      List.filter_map (function Tc.K _, r -> Some r | _ -> None) ne
    in
    let reduce_ranges =
      List.filter (fun x -> not (List.memq x k_ranges)) (List.tl (src reduceop))
    in
    let tc_uop =
      match reduce_ranges with
      | [] -> tc_uop
      | rs ->
          Ops.v Op.Reduce ~src:(tc_uop :: rs)
            ~arg:(Reduce { op = Op.Add; num_axes = 0 })
    in
    k.ast <-
      substitute ~calls:Skip ~pass:Fixed_point k.ast [ (reduceop, tc_uop) ]

  and index_of x l =
    let rec go i = function
      | [] -> invalid_arg "the node is not an axis of the kernel"
      | y :: _ when y == x -> i
      | _ :: r -> go (i + 1) r
    in
    go 0 l

  and index_of_bit b l =
    let rec go i = function
      | [] -> None
      | y :: _ when Tc.equal_bit y b -> Some i
      | _ :: r -> go (i + 1) r
    in
    go 0 l

  let apply_opt ?append_opt k opt =
    match apply ?append_opt k opt with
    | axes -> Ok axes
    | exception Refused msg -> Error msg

  let get_optimized_ast ?name_override k =
    let name =
      match name_override with
      | Some n -> n
      | None ->
          let k_type = if Option.is_some (reduceop k) then "r" else "E" in
          let special_uops =
            List.stable_sort
              (fun a b -> String.compare (special_name a) (special_name b))
              (of_op k Op.Special)
          in
          (* A program's name holds no colour: colour is how a name is shown,
             and a program kept on disk must not depend on the display of the
             process that made it. *)
          let special_ops =
            List.map (fun x -> Bigint.to_string (size x)) special_uops
          in
          let axes = List.map (fun x -> Render.render (nth x 0)) (rngs k) in
          k_type ^ String.concat "_" (("" :: special_ops) @ axes)
    in
    replace k.ast
      ~arg:
        (Kernel
           (kernel_info ~name ~applied_opts:k.applied_opts ~optimized:true ()))
end

(* Whether a loop of [ast], or of a body it calls, runs as many trips as
   memory holds, as [j < x[i]]: its time depends on data a search does not
   have. *)
let bound_by_memory ast =
  List.exists
    (fun u -> op u = Op.Range && bound_loads u)
    (toposort ~calls:Enter ast)

(* Whether the builder of [ast], or of a body it calls, fixed its threads:
   workgroup memory sized for one workgroup, registers for one thread, a
   barrier every thread must reach, a matrix multiply-accumulate on a warp's
   lanes, or custom code. A split of an axis into threads or lanes would leave
   each of them as the builder made it, for threads it did not make. *)
let threads_fixed ast =
  List.exists
    (fun u ->
      match (op u, arg u) with
      | (Op.Param | Op.Buffer | Op.Alloc), Param p ->
          List.mem p.addrspace Dtype.[ Local; Reg ]
      | (Op.Barrier | Op.Wmma | Op.Custom | Op.Customi), _ -> true
      | _ -> false)
    (toposort ~calls:Enter ast)

(* Weak ranges

   A weak range is its builder's promise that two points of the kernel whose
   values of it differ never meet at a cell of global or workgroup memory that
   one of them writes ({!Ops.Axis_type.Weak}). A store breaks it where it can
   be counted: it names fewer cells than it has points along a set of its weak
   ranges, the other ranges fixed, and neither its validity nor its gate reads
   the set, so two of those points write one cell. The sets are each weak
   range alone and all of them together: [C[r + s]] names 7 cells for 16
   points of [r] and [s], 4 for 4 of each. Only a range whose bound is a
   constant or a variable is counted: the greatest value of another bound, as
   [j < i + 1] or [j < n (4 - n)], need not be reached, so its size need not
   count points. A range whose trips are not known at launch stays serial
   ([Scheduler.split_size]). *)

(* The constant factor of [t], if it is [c * x]: its values are multiples of
   it. *)
let factor t =
  let literal u =
    if is_literal u then
      match value u with `Int c -> Some (Bigint.abs c) | _ -> None
    else None
  in
  match (op t, src t) with
  | Op.Mul, [ a; b ] -> (
      match literal b with
      | Some c -> c
      | None -> Option.value (literal a) ~default:Bigint.one)
  | _ -> Bigint.one

(* The cells the offset [o] names along the ranges [set], the others fixed: at
   most the multiples of [g] between the least and the greatest sum of its
   terms that read [set], [g] the greatest common divisor of their constant
   factors. [C[16 (r // 8)]] names 2 cells, not 17. *)
let cells set o =
  let reads u = List.exists (fun r -> mem_ranges r u) set in
  let rec sum ((lo, hi, g) as acc) t =
    if op t = Op.Add then sum (sum acc (nth t 0)) (nth t 1)
    else if not (reads t) then acc
    else
      ( Bigint.add lo (V.to_z (vmin t)),
        Bigint.add hi (V.to_z (vmax t)),
        Bigint.gcd g (factor t) )
  in
  let lo, hi, g = sum (Bigint.zero, Bigint.zero, Bigint.zero) o in
  if Bigint.equal g Bigint.zero then Bigint.one
  else Bigint.(succ (div (sub hi lo) g))

let weak_store st =
  let idx = nth st 0 in
  let space = addrspace (buf_uop (nth idx 0)) in
  (* A bound that is a constant or a variable reaches its greatest value at
     some launch, every such bound at one launch: then the product of the
     ranges' sizes counts points that one launch runs. *)
  let exact r =
    let b = without_after (nth r 0) in
    is_literal b
    ||
    match (op b, arg b) with
    | Op.Param, Param p -> p.addrspace = Dtype.Alu
    | _ -> false
  in
  let weak r =
    axis_type r = Axis_type.Weak
    && (not (Dtype.equal (dtype r) Dtype.Void))
    && exact r
    && Bigint.gt (size r) Bigint.one
  in
  (* A store inside a range of no trip never runs. *)
  let runs = Nodes.fold (fun r ok -> ok && Bigint.gt (size r) Bigint.zero) in
  let weak =
    if
      (space = Some Dtype.Global || space = Some Dtype.Local)
      && runs (ranges st) true
    then
      Nodes.fold (fun r acc -> if weak r then r :: acc else acc) (ranges st) []
    else []
  in
  let offsets = List.tl (src idx) in
  let gate = match src st with [ _; _; g ] -> Some g | _ -> None in
  let refute set =
    let reads u = List.exists (fun r -> mem_ranges r u) set in
    let points =
      List.fold_left (fun p r -> Bigint.mul p (size r)) Bigint.one set
    in
    let named =
      List.fold_left
        (fun p o -> Bigint.mul p (cells set (get_idx o)))
        Bigint.one offsets
    in
    if
      (not (List.exists (fun o -> reads (get_valid o)) offsets))
      && (not (Option.fold ~none:false ~some:reads gate))
      && Bigint.lt named points
    then
      invalid_arg
        (strf
           "the store to %s names %s cells for %s points of %s: a range whose \
            points meet at a cell must not be Weak"
           (Render.render idx) (Bigint.to_string named)
           (Bigint.to_string points)
           (String.concat " and " (List.map Render.render set)))
  in
  List.iter (fun r -> refute [ r ]) weak;
  match weak with _ :: _ :: _ -> refute weak | _ -> ()

(* Raises if a store of [k] breaks a weak range's promise where it can be
   counted. *)
let refute_weak k =
  List.iter
    (fun u -> if op u = Op.Store && op (nth u 0) = Op.Index then weak_store u)
    (Scheduler.slice k)

let apply_opts ?beam ~hand_coded ast ren =
  let info = match arg ast with Kernel i -> Some i | _ -> None in
  if Option.exists (fun (i : kernel_info) -> i.optimized) info then ast
  else
    let k = Scheduler.v ast ren in
    refute_weak k;
    Scheduler.convert_loop_to_global k;
    let fixed = lazy (threads_fixed ast) in
    let k =
      match (info, beam) with
      | Some { opts_to_apply = Some opts; _ }, _ ->
          let apply o =
            match Scheduler.apply_opt k o with
            | Ok _ -> ()
            | Error msg -> invalid_arg msg
          in
          List.iter apply opts;
          k
      | _, Some search
        when not (bound_by_memory ast || Lazy.force fixed) ->
          search k
      | _
        when (not (Setting.value Setting.noopt))
             && match info with Some i -> i.applied_opts = [] | None -> true ->
          (* NOTE: hand_coded_optimizations doesn't support multiblock opts
             yet *)
          if
            op_in_backward_slice_with_self ~calls:Skip ast [ Op.Stage ]
            || Lazy.force fixed
          then k
          else hand_coded k
      | _ -> k
    in
    let name_override =
      match info with Some i when i.name <> "test" -> Some i.name | _ -> None
    in
    Scheduler.get_optimized_ast ?name_override k
