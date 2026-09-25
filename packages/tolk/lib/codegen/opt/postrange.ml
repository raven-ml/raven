(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Port of tinygrad/codegen/opt/postrange.py to the tolk_uop IR. *)

open Tolk_uop
module U = Uop

let strf = Printf.sprintf
let prod = List.fold_left ( * ) 1

exception Opt_error of string

let check cond msg = if not cond then raise (Opt_error msg)

let nth_or_error lst i msg =
  match List.nth_opt lst i with
  | Some v -> v
  | None -> raise (Opt_error msg)

(* Error strings *)

let err_locals_needed = "locals needed for opt"
let err_no_reduce_tc = "no reduce ops for TensorCore"
let err_invalid_tc_choice = "invalid tensor core choice"
let err_tc_first = "tensor core opts must be first"
let err_no_tc_available = "no tensor core available"
let err_range_missing = "range not found"

let scalar_eq (a : Dtype.t) (b : Dtype.t) = Dtype.equal a b

let scalar_is_float32 s = Dtype.equal s Dtype.float32

(* Range-view helpers *)

let range_view u =
  match U.as_range u with
  | Some v -> v
  | None -> raise (Opt_error "postrange: not a range node")

let range_size u = (range_view u).size
let range_axis u = (range_view u).axis
let range_kind u = (range_view u).kind
let is_range u = Option.is_some (U.as_range u)
let const_int_or default u =
  match U.const_int_value u with Some n -> n | None -> default

let range_int_size u = const_int_or 0 (range_size u)

let range_max_extent u = Bound.to_int (Bound.succ (U.vmax u))

(* Cached shape data recomputed after every AST mutation. *)
type shape = {
  rngs : U.t list;
  axis_types : Axis_type.t list;
  full_shape : U.t list;
}

(* Always in order by axis type. Device axes are launched, while void
   ranges are loop scopes; neither is a numeric optimization axis. *)
let compute_shape ast =
  let rngs =
    U.toposort ast
    |> List.filter (fun u ->
           is_range u && U.dtype u <> Dtype.void
           && Bound.lt Bound.zero (U.vmax u) && range_kind u <> Axis_type.Device)
    |> List.sort (fun a b ->
         compare
           (Axis_type.to_pos (range_kind a), U.axis_id a)
           (Axis_type.to_pos (range_kind b), U.axis_id b))
  in
  let axis_types = List.map range_kind rngs in
  let full_shape = List.map (fun r -> U.simplify (range_size r)) rngs in
  { rngs; axis_types; full_shape }

(* Scheduler state: wraps a kernel AST and tracks applied optimisations. *)
type t = {
  mutable ast : U.t;
  ren : Renderer.t;
  mutable applied_opts : U.Opt.t list;
  mutable tensor_core : Tc.t option;
  mutable opt_range : int;
  mutable shape : shape;
}

let refresh t = t.shape <- compute_shape t.ast

let create ast ren =
  let applied_opts =
    match U.as_kernel_info ast with
    | Some ki -> ki.applied_opts
    | None -> []
  in
  let shape = compute_shape ast in
  let max_axis =
    List.fold_left
      (fun acc r -> if is_range r then max acc (range_axis r) else acc)
      0 (U.backward_slice ast)
  in
  {
    ast; ren; applied_opts;
    tensor_core = None; opt_range = max_axis + 1; shape;
  }

(* Accessors read from the cached shape. *)
let rngs t = t.shape.rngs
let shape_len t = List.length t.shape.rngs
let full_shape t = t.shape.full_shape
let axis_types t = t.shape.axis_types
let ast t = t.ast
let ren t = t.ren
let applied_opts t = t.applied_opts
let tensor_core t = t.tensor_core

let copy t = { t with ast = t.ast }

type snapshot = {
  snap_ast : U.t;
  snap_applied_opts : U.Opt.t list;
  snap_tensor_core : Tc.t option;
  snap_opt_range : int;
  snap_shape : shape;
}

let snapshot t =
  {
    snap_ast = t.ast;
    snap_applied_opts = t.applied_opts;
    snap_tensor_core = t.tensor_core;
    snap_opt_range = t.opt_range;
    snap_shape = t.shape;
  }

let restore t s =
  t.ast <- s.snap_ast;
  t.applied_opts <- s.snap_applied_opts;
  t.tensor_core <- s.snap_tensor_core;
  t.opt_range <- s.snap_opt_range;
  t.shape <- s.snap_shape

(* Output-range and globalizable-range analyses *)

(* Ranges that appear in END nodes with non-Reduce axis type. *)
let output_rngs t =
  List.concat_map
    (fun s ->
      match U.as_end s with
      | Some { ranges; _ } ->
          List.filter
            (fun r -> is_range r && range_kind r <> Axis_type.Reduce)
            (U.ranges (U.sink ranges))
      | None -> [])
    (U.children t.ast)

(* Weak ranges eligible for promotion to Global: must appear in all
   Stage nodes' ranges. *)
let globalizable_rngs t =
  let out =
    List.filter (fun r -> range_kind r = Axis_type.Weak) (output_rngs t)
  in
  List.fold_left
    (fun acc node ->
      match U.as_stage node with
      | Some _ ->
          let stage_ranges = U.ranges node in
          List.filter (fun r -> List.memq r stage_ranges) acc
      | None -> acc)
    out (U.toposort t.ast)

(* Promote eligible Weak ranges to Global. *)
let convert_loop_to_global t =
  if Renderer.has_local t.ren then begin
    let glob = globalizable_rngs t in
    let subs =
      List.filter_map
        (fun r ->
          if List.memq r glob then
            let v = range_view r in
            Some
              ( r,
                U.range ~size:v.size ~axis:v.axis ~sub:v.sub
                  ~kind:Axis_type.Global
                  ~dtype:(U.dtype r)
                  ~parents:v.parents
                  () )
          else None)
        (rngs t)
    in
    if subs <> [] then (
      t.ast <- U.substitute subs t.ast;
      refresh t)
  end

let is_reduce u = U.op u = Ops.Reduce

let reduceop t = List.find_opt is_reduce (U.backward_slice t.ast)

(* Debug colour for the axis-type visualiser. *)
let axis_color : Axis_type.t -> string = function
  | Device -> "green"
  | Global -> "blue"
  | Local -> "cyan"
  | Warp -> "CYAN"
  | Weak | Loop -> "WHITE"
  | Upcast -> "yellow"
  | Reduce -> "red"
  | Unroll -> "magenta"
  | Placeholder -> "white"

let colors t =
  let out = output_rngs t in
  let glob = globalizable_rngs t in
  List.map2
    (fun at r ->
      if at = Axis_type.Weak && not (List.memq r out) then "BLACK"
      else if at = Axis_type.Weak && not (List.memq r glob) then "white"
      else axis_color at)
    (axis_types t) (rngs t)

let render_size sz =
  match U.const_int_value sz with
  | Some n -> string_of_int n
  | None -> Render.expr_to_string sz

let colored_shape t =
  String.concat " "
    (List.map2
       (fun rng color -> strf "%4s:%s" (render_size (range_size rng)) color)
       (rngs t) (colors t))

(* Apply [flatten_range] locally: toposort-reorder range children of
   Reduce/Store/End nodes.  Inline port of [Simplify.pm_flatten_range]
   since the tolk_uop-based simplifier lives in a different library. *)
let rec list_take n = function
  | _ when n <= 0 -> []
  | [] -> []
  | x :: xs -> x :: list_take (n - 1) xs

let rec list_drop n = function
  | l when n <= 0 -> l
  | [] -> []
  | _ :: xs -> list_drop (n - 1) xs

let ended_ranges u =
  match U.as_end u, U.as_reduce u with
  | Some v, _ -> v.ranges
  | None, Some v -> v.ranges
  | None, None -> []

let range_offset node =
  match U.op node with
  | Ops.Reduce | Ops.End -> Some 1
  | _ -> None

let reorder_range_node node =
  match range_offset node with
  | None -> None
  | Some off ->
      let ch = U.children node in
      let rngs = list_drop off ch in
      if rngs = [] then None
      else
        let new_rngs = List.filter is_range (U.toposort (U.sink rngs)) in
      if List.equal U.equal new_rngs rngs then None
      else
        Some
          (U.replace node ~src:(Array.of_list (list_take off ch @ new_rngs)) ())

let pm_flatten_range root = U.graph_rewrite reorder_range_node root

(* Kernel names depend only on the schedule, so recompilation preserves both
   source-cache keys and program identity. *)
let make_kernel_name t =
  let k_type = if reduceop t <> None then "r" else "E" in
  let special_cmp a b =
    match U.as_special a, U.as_special b with
    | Some va, Some vb -> Gpu_dim.compare_special_name va.name vb.name
    | _ -> 0
  in
  let specials =
    List.filter (fun n -> U.op n = Ops.Special) (U.toposort t.ast)
    |> List.sort special_cmp
  in
  (* Special sizes always print their upper bound, even when symbolic. *)
  let special_strs =
    List.map (fun s -> string_of_int (Bound.to_int (Bound.succ (U.vmax s)))) specials
  in
  let rng_strs = List.map (fun r -> render_size (range_size r)) (rngs t) in
  (* Reference builds ['_'.join([''] + parts)]: no separator at all when the
     kernel has no dims, so a dimensionless kernel is named plain "E"/"r". *)
  k_type
  ^ String.concat ""
      (List.map (fun s -> "_" ^ s) (special_strs @ rng_strs))

(* Finalize the kernel: generate a debug name, flatten ranges, and attach
   updated kernel_info with a tag marking it as optimized. *)
let get_optimized_ast ?name_override t =
  let name = match name_override with Some n -> n | None -> make_kernel_name t in
  t.ast <- pm_flatten_range t.ast;
  refresh t;
  let ki : U.kernel_info =
    {
      name;
      applied_opts = t.applied_opts;
      opts_to_apply = None;
      estimates = None;
      beam = 0;
    }
  in
  let srcs =
    if U.op t.ast = Ops.Sink then U.children t.ast else [ t.ast ]
  in
  U.with_tag "1" (U.sink ~kernel_info:ki srcs)

(* Split [rng] by [amount]: the original range shrinks to size/amount, a
   new range of [amount] is created with [new_kind].  When [top] is true
   the new range is the high part; otherwise it is the low part. *)
let shift_to ?(top = false) ?input_new_rng t rng amount new_kind =
  let allowed = match new_kind with
    | Axis_type.Upcast -> [ Axis_type.Global; Axis_type.Local; Axis_type.Weak ]
    | Axis_type.Unroll -> [ Axis_type.Reduce; Axis_type.Local ]
    | Axis_type.Local -> [ Axis_type.Global; Axis_type.Weak; Axis_type.Reduce ]
    | _ -> [] in
  check (List.mem (range_kind rng) allowed) "invalid split source and target kinds";
  check (amount > 0) "invalid optimization amount";
  let size = range_size rng in
  let old_sz =
    match U.divides size amount with
    | Some q -> q
    | None -> raise (Opt_error (strf "shift_to: %d can't divide range" amount))
  in
  let new_rng =
    match input_new_rng with
    | Some r -> r
    | None ->
        let axis = t.opt_range in
        t.opt_range <- t.opt_range + 1;
        U.range ~size:(U.const (Const.int (U.dtype rng) amount)) ~axis ~kind:new_kind ()
  in
  let replaced_rng = U.replace rng ~src:[| old_sz |] () in
  let open U.O in
  let sub_axis =
    if top then (new_rng * old_sz) + replaced_rng
    else (replaced_rng * U.const_int amount) + new_rng
  in
  t.ast <- U.substitute [ (rng, sub_axis) ] t.ast;
  refresh t;
  (replaced_rng, new_rng)

let ranges_of t kinds =
  List.filter (fun r -> List.mem (range_kind r) kinds) (rngs t)

let axes_of t kinds =
  axis_types t
  |> List.mapi (fun i at -> if List.mem at kinds then Some i else None)
  |> List.filter_map Fun.id

(* Axes of the given kinds whose full_shape entry is a constant > 1. *)
let const_dims t kinds =
  let fs = full_shape t in
  List.filter (fun i -> const_int_or 0 (List.nth fs i) > 1) (axes_of t kinds)

let upcast_size t =
  let fs = full_shape t in
  prod
    (List.map
       (fun a -> const_int_or 1 (List.nth fs a))
       (axes_of t [ Axis_type.Upcast; Axis_type.Unroll ]))

let upcastable_dims t =
  const_dims t [ Axis_type.Global; Axis_type.Local; Axis_type.Weak ]

let reduce_axes t =
  let reduced = U.Ref_tbl.create 16 in
  List.iter (fun u -> match U.as_reduce u with
      | None -> ()
      | Some v -> List.iter (fun r ->
          List.iter (fun r -> U.Ref_tbl.replace reduced r ()) (U.ranges r)) v.ranges)
    (U.backward_slice t.ast);
  List.mapi (fun i r -> if U.Ref_tbl.mem reduced r then Some i else None) (rngs t)
  |> List.filter_map Fun.id

let unrollable_dims t =
  let reduced = reduce_axes t in
  List.filter (fun i -> List.mem i reduced)
    (const_dims t [ Axis_type.Local; Axis_type.Reduce ])

let bufs t =
  List.rev
    (List.filter (fun x -> U.op x = Ops.Index) (U.toposort t.ast))

let upcasted t = List.length (axes_of t [ Axis_type.Upcast; Axis_type.Unroll ])

let group_for_reduces t =
  let kinds = axis_types t in
  List.fold_left (fun n i ->
      match List.nth kinds i with Axis_type.Warp | Axis_type.Local -> n + 1 | _ -> n)
    0 (reduce_axes t)

(* Split tile-coordinate bits, sharing one explicit hardware warp. *)
let apply_tc_shifts t axes (tc : Tc.t) =
  let warp = U.range ~size:(U.const_int tc.threads) ~axis:(-1) ~kind:Axis_type.Warp () in
  List.map (fun coordinate ->
      let dim = match coordinate.[0] with 'n' -> 0 | 'm' -> 1 | _ -> 2 in
      let replaced, lane =
        match List.find_index (( = ) coordinate) (fst tc.frag_c) with
        | Some bit ->
            let divisor = U.const_int (1 lsl bit) in
            let quotient = U.alu_binary ~op:Ops.Floordiv ~lhs:warp ~rhs:divisor in
            let lane = U.alu_binary ~op:Ops.Floormod ~lhs:quotient ~rhs:(U.const_int 2) in
            shift_to ~input_new_rng:lane t axes.(dim) 2 Axis_type.Local
        | None ->
            shift_to t axes.(dim) 2
              (if dim = 2 then Axis_type.Unroll else Axis_type.Upcast)
      in
      axes.(dim) <- replaced;
      coordinate, lane) (Tc.axis_coords tc)

(* The value a tensor core takes for a factor of the product: the factor at the
   core's input dtype, or the narrower float a float32 factor is widened from
   when the core outputs float32. The core multiplies it exactly, and the
   float32 product of the widened values is exact unless it leaves float32's
   normal range, which only a bfloat16 product can. *)
let tc_operand (tc : Tc.t) u =
  if scalar_eq (U.dtype u) tc.dtype_in then Some u
  else
    match U.op u, U.src u with
    | Ops.Cast, [| src |]
      when scalar_eq (U.dtype src) tc.dtype_in
           && Dtype.is_float tc.dtype_in && Dtype.itemsize tc.dtype_in < 4
           && scalar_is_float32 (U.dtype u)
           && scalar_is_float32 tc.dtype_out ->
        Some src
    | _ -> None

let build_wmma_node t (tc : Tc.t) axes coordinates =
  let reduce =
    match List.filter (fun u -> match U.as_reduce u with
        | Some r -> List.exists (fun x -> List.memq axes.(2) (U.ranges x)) r.ranges
        | None -> false) (U.backward_slice t.ast) with
    | [ reduce ] -> reduce
    | _ -> raise (Opt_error "tensor-core contraction must belong to one reduce") in
  let red = Option.get (U.as_reduce reduce) in
  let gate, mul = match U.op red.src, U.src red.src with
    | Ops.Where, [| gate; value; _ |] -> Some gate, value
    | _ -> None, red.src in
  let mul = if U.op mul = Ops.Cast then (U.src mul).(0) else mul in
  let inputs = match U.op mul, U.src mul with
    | Ops.Mul, [| a; b |] -> [ a; b ]
    | _ -> raise (Opt_error "tensor-core reduction must multiply two operands") in
  let inputs = List.map (fun input -> match tc_operand tc input with
      | Some operand -> operand
      | None -> raise (Opt_error "tensor-core operand dtype differs from the core's")) inputs in
  let inputs = List.map (fun input -> match gate with
      | None -> input
      | Some gate -> U.alu_ternary ~op:Ops.Where ~a:gate ~b:input
          ~c:(U.const (Const.zero (U.dtype input)))) inputs in
  let inputs = List.map2 (fun input relabel ->
      let mappings = List.map (fun (a, b) -> List.assoc a coordinates, List.assoc b coordinates) relabel in
      U.substitute ~walk:true mappings input) inputs (Tc.relabel tc) in
  let base_axes = List.map (fun c -> U.axis_id (List.assoc c coordinates)) (Tc.base_upcast_axes tc) in
  let counts = List.map (fun f -> List.length (snd f)) [ tc.frag_a; tc.frag_b; tc.frag_c ] in
  let a_count, b_count, c_count = match counts with
    | [ a; b; c ] -> a, b, c
    | _ -> assert false in
  let upcast_axes = List.map (fun count ->
      base_axes |> List.filteri (fun i _ -> i < max count (max a_count b_count))
      |> List.mapi (fun i axis -> axis, if i < count then 2 else 1)) counts in
  let ua, ub, uc = match upcast_axes with
    | [ a; b; c ] -> a, b, c
    | _ -> assert false in
  let info : U.wmma_info =
    { dims = tc.dims; dtype_in = tc.dtype_in;
      threads = tc.threads; tc_upcast_axes = Some (ua, ub, uc) } in
  let a, b = match inputs with [ a; b ] -> a, b | _ -> assert false in
  let wmma = U.wmma ~a ~b
      ~c:(U.const_of_dtype tc.dtype_out (U.Const_tuple
          (List.init (1 lsl c_count) (fun _ -> U.Const_scalar (`Float 0.0)))))
      ~info in
  let contracted = List.filter_map (fun (coordinate, range) ->
      if coordinate.[0] = 'k' then Some range else None) coordinates in
  let extra = U.find_nodes is_range (U.sink red.ranges)
      |> List.filter (fun r -> not (List.memq r contracted)) in
  let value = if extra = [] then wmma
    else U.reduce ~op:Ops.Add ~src:wmma ~ranges:extra in
  t.ast <- U.substitute [ reduce, value ] t.ast;
  refresh t

(* Workgroup reductions reserve shared storage for every local and vector lane. *)
let check_shared_memory t axis kind amount =
  match reduceop t with
  | Some red when (kind = Axis_type.Local && List.mem axis (reduce_axes t))
                  || group_for_reduces t > 0 ->
      let fs = full_shape t in
      let lanes = prod (List.map (fun a -> const_int_or 1 (List.nth fs a))
          (axes_of t [ Axis_type.Upcast; Axis_type.Warp; Axis_type.Local ])) in
      let needed = amount * lanes * Dtype.itemsize (U.dtype red) in
      check (needed <= Renderer.shared_max t.ren)
        (strf "exceeds shared memory: needs %d, max %d" needed (Renderer.shared_max t.ren))
  | _ -> ()

let check_reduction_split t r kind =
  if kind = Axis_type.Unroll || range_kind r = Axis_type.Reduce then begin
    let owner = List.find_opt (fun u -> match U.as_reduce u with
        | Some red -> List.exists (fun axis -> List.memq r (U.ranges axis)) red.ranges
        | None -> false) (U.backward_slice t.ast) in
    check (Option.is_some owner) "cannot split a reduction axis outside a REDUCE";
    if kind = Axis_type.Local then
      check (not (List.exists (fun u ->
          let k = range_kind u in k = Axis_type.Reduce || k = Axis_type.Unroll)
          (U.ranges (Option.get owner))))
        "cannot have a workgroup reduction inside another reduce"
  end

let round_up x n = (x + n - 1) / n * n

let is_invalid_const u =
  match U.op u, U.arg u with
  | Ops.Const, U.Arg.Value c -> Const.view c = Const.Invalid
  | _ -> false

let rec get_idx_valid u =
  match U.op u, U.src u with
  | Ops.Where, [| valid; idx; invalid |] when is_invalid_const invalid ->
      idx, valid
  | Ops.Stack, srcs ->
      let idxs, valids =
        Array.fold_right
          (fun child (idxs, valids) ->
            let idx, valid = get_idx_valid child in
            idx :: idxs, valid :: valids)
          srcs ([], [])
      in
      U.stack idxs, U.stack valids
  | _ -> u, U.const_bool true


(* Pad a range to a multiple of [amount]. *)
let apply_padto t r amount =
  check (amount > 1) "pad amount must be greater than one";
  check (Option.is_some (U.const_int_value (range_size r))) "only pad const axes";
  let rng_kind = range_kind r in
  check
    (rng_kind <> Axis_type.Upcast && rng_kind <> Axis_type.Unroll
     && rng_kind <> Axis_type.Warp)
    "cannot pad upcasted or warp";
  let old_size = range_int_size r in
  let new_sz = round_up old_size amount in
  check (old_size > new_sz / 4) "pad adds more than quadruple the work";
  let replaced_rng = U.replace r ~src:[| U.const_int new_sz |] () in
  let valid =
    U.alu_binary ~op:Ops.Cmplt ~lhs:replaced_rng ~rhs:(U.const_int old_size)
  in
  let subs =
    List.fold_left
      (fun acc b ->
        match U.as_index b with
        | Some { ptr; idxs = [ idx ] } when List.memq r (U.ranges idx) ->
            let idx, idx_valid = get_idx_valid idx in
            let cond = U.alu_binary ~op:Ops.And ~lhs:valid ~rhs:idx_valid in
            let guarded_idx = U.valid ~src:idx ~cond in
            (b, U.replace b ~src:[| ptr; guarded_idx |] ()) :: acc
        | _ -> acc)
      [ (r, replaced_rng) ] (bufs t)
  in
  let subs = List.fold_left (fun acc node ->
      match U.as_reduce node with
      | Some red when List.exists (fun axis -> List.memq r (U.ranges axis)) red.ranges ->
          let dtype = U.dtype node in
          let identity = match red.op with
            | Ops.Add -> Const.zero dtype
            | Ops.Mul -> Const.one dtype
            | Ops.Max -> Const.min_value dtype
            | _ -> raise (Opt_error "unsupported padded reduction") in
          let value = U.alu_ternary ~op:Ops.Where ~a:valid ~b:red.src
              ~c:(U.const identity) in
          (node, U.replace node ~src:(Array.of_list (value :: red.ranges)) ()) :: acc
      | _ -> acc) subs (U.backward_slice t.ast) in
  t.ast <- U.substitute subs t.ast;
  refresh t

(* Swap two global ranges' axis numbers. *)
let apply_swap t r with_axis =
  let altrng = nth_or_error (rngs t) with_axis "invalid swap axis" in
  check
    (range_kind r = Axis_type.Global && range_kind altrng = Axis_type.Global)
    "swap only for globals";
  let rv = range_view r and av = range_view altrng in
  let r' = U.replace r ~arg:(U.Arg.Range_info
      { axis = av.axis; sub = av.sub; kind = rv.kind }) () in
  let alt' = U.replace altrng ~arg:(U.Arg.Range_info
      { axis = rv.axis; sub = rv.sub; kind = av.kind }) () in
  t.ast <- U.substitute ~walk:true [ r, r'; altrng, alt' ] t.ast;
  refresh t

(* Mutual recursion: apply_opt <-> apply_tc_opt <-> pad_tc_axes *)

exception Tc_candidate_miss

(* Pad each TC axis to a multiple of tc.dims.(i).  Returns false on
   PADTO failure. *)
let rec pad_tc_axes t axes (tc : Tc.t) tc_opt =
  (try
     for i = 0 to 2 do
       let a = axes.(i) in
       let idx =
         match List.find_index (fun r -> r == a) (rngs t) with
         | Some j -> j
    | None -> raise (Opt_error err_range_missing)
       in
       let dim =
         let n, m, k = tc.dims in
         [| n; m; k |].(i)
       in
       if range_max_extent a mod dim <> 0 then begin
         if tc_opt < 2 then raise Tc_candidate_miss;
         (try
            ignore
              (apply_opt ~append_opt:false t
                 (U.Opt.Padto { axis = idx; amount = dim }))
          with Opt_error _ -> raise Tc_candidate_miss);
         axes.(i) <- List.nth (rngs t) idx
       end
     done;
     true
   with Tc_candidate_miss -> false)

(* Apply tensor core optimisation.  Returns [Some axes] on success,
   [None] if no matching TC was found. *)
and apply_tc_opt t use_tc axis tc_select tc_opt =
  let red =
    match List.find_opt is_reduce (U.toposort t.ast) with
    | Some r -> r
    | None -> raise (Opt_error err_no_reduce_tc)
  in
  let red_view = Option.get (U.as_reduce red) in
  if use_tc = 0 || red_view.op <> Ops.Add then None
  else
    let red_src = red_view.src in
    let mul =
      if U.op red_src = Ops.Cast then (U.src red_src).(0) else red_src
    in
    match U.op mul, U.src mul with
    | Ops.Mul, [| in0; in1 |] ->
        let tcs =
          if tc_select = -1 then Renderer.tensor_cores t.ren
          else
            match List.nth_opt (Renderer.tensor_cores t.ren) tc_select with
            | Some tc -> [ tc ]
            | None -> raise (Opt_error err_invalid_tc_choice)
        in
        let red_sc = U.dtype red in
        let in0_ranges = U.ranges in0 in
        let in1_ranges = U.ranges in1 in
        let red_ranges = U.ranges (U.sink red_view.ranges) in
        let sort_desc =
          List.sort (fun a b -> compare (U.axis_id b) (U.axis_id a))
        in
        let try_tc (tc : Tc.t) =
          let snap = snapshot t in
          try
            let result =
              if
                (Renderer.device t.ren = "CUDA" || Renderer.device t.ren = "NV")
                && scalar_is_float32 tc.dtype_in
                && Helpers.Context_var.get Helpers.allow_tf32 = 0
              then None
              else if
                Option.is_none (tc_operand tc in0)
                || Option.is_none (tc_operand tc in1)
                || not (scalar_eq red_sc tc.dtype_out)
              then None
              else
                let in0_r =
                  sort_desc
                    (List.filter
                       (fun u -> not (List.memq u in1_ranges))
                       in0_ranges)
                in
                let in1_r =
                  sort_desc
                    (List.filter
                       (fun u -> not (List.memq u in0_ranges))
                       in1_ranges)
                in
                let red_r = sort_desc red_ranges in
                if in0_r = [] || in1_r = [] || red_r = [] then None
                else
                  (* NOTE: tinygrad swaps in0 and in1 when building the
                     axis choice list; preserved here for parity. *)
                  let choices =
                    List.concat_map
                      (fun a ->
                        List.concat_map
                          (fun b -> List.map (fun c -> [ a; b; c ]) red_r)
                          in0_r)
                      in1_r
                  in
                  if axis >= List.length choices then None
                  else begin
                    let axes = Array.of_list (List.nth choices axis) in
                    (* The X and Y axes index the accumulator, so a reduce
                       over them would have to accumulate across the tile the
                       WMMA writes as a whole. *)
                    check
                      (not (List.exists (fun i ->
                          let r = List.nth (rngs t) i in r == axes.(0) || r == axes.(1))
                          (reduce_axes t)))
                      "tensor core X/Y axes cannot be contracted";
                    if not (pad_tc_axes t axes tc tc_opt) then None
                    else begin
                      let ne = apply_tc_shifts t axes tc in
                      if use_tc <> 2 then build_wmma_node t tc axes ne;
                      t.tensor_core <- Some tc;
                      Some (Array.to_list axes)
                    end
                end
            in
            if Option.is_none result then restore t snap;
            result
          with
          | exn ->
              restore t snap;
              raise exn
        in
        List.find_map try_tc tcs
    | _ -> None

and apply_opt ?(append_opt = true) t opt =
  let ret =
    match opt with
    | U.Opt.Tc { axis; tc_select; tc_opt; use_tc } ->
        check (axis >= 0) "invalid tensor core axis";
        check (t.applied_opts = []) err_tc_first;
        check
          (tc_select >= -1
          && tc_select < List.length (Renderer.tensor_cores t.ren))
          "invalid tc_select";
        check (tc_opt >= 0 && tc_opt <= 2) "invalid tc_opt";
        check (use_tc > 0 && use_tc <= 2) "invalid use_tc";
        let axes = apply_tc_opt t use_tc axis tc_select tc_opt in
        check (Option.is_some axes) err_no_tc_available;
        (match axes with
         | Some (a :: b :: _) -> Some (a, b)
         | _ -> None)
    | Padto { axis; amount } ->
        check (axis >= 0 && axis < shape_len t) "invalid axis";
        let r = List.nth (rngs t) axis in
        apply_padto t r amount;
        None
    | Swap { axis; with_axis } ->
        check (axis >= 0 && axis < shape_len t) "invalid axis";
        check (with_axis >= 0 && with_axis < shape_len t) "invalid swap axis";
        let r = List.nth (rngs t) axis in
        apply_swap t r with_axis;
        None
    | Split { axis; amount; kind; top } ->
        check (axis >= 0 && axis < shape_len t) "invalid axis";
        check (amount = 0 || amount > 1) "split amount must be zero or greater than one";
        let r = List.nth (rngs t) axis in
        let amount = if amount = 0 then range_max_extent r else amount in
        (match kind with
         | Axis_type.Local -> check (Renderer.has_local t.ren) err_locals_needed
         | Axis_type.Unroll -> check (amount <= 32) "don't unroll more than 32"
         | Axis_type.Upcast -> check (Renderer.device t.ren = "DSP" || amount <= 16)
             "don't upcast more than 16"
         | _ -> raise (Opt_error "split target must be Upcast, Unroll or Local"));
        check_shared_memory t axis kind amount;
        check_reduction_split t r kind;
        Some (shift_to ~top t r amount kind)
  in
  if append_opt then t.applied_opts <- t.applied_opts @ [ opt ];
  ret

(* Extract sorted Param nodes from an AST.  Returns raw Param nodes; the
   caller constructs device buffers. *)
let bufs_from_ast ast =
  let slot_of u =
    match U.as_param u with
    | Some { param; _ } when param.slot >= 0 -> Some param.slot
    | _ -> None
  in
  U.backward_slice ast
  |> List.filter_map (fun u ->
       match slot_of u with Some slot -> Some (slot, u) | None -> None)
  |> List.sort (fun (a, _) (b, _) -> compare a b)
  |> List.map snd

(* Top-level optimization dispatch.  Strategy closures are passed by the
   caller to break circular module dependencies. *)
let apply_opts ?beam_search ?hand_coded_optimizations ast ren =
  if U.node_tag ast <> None then ast
  else
    let ki = U.as_kernel_info ast in
    let k = create ast ren in
    convert_loop_to_global k;
    let has_stage =
      List.exists (fun n -> U.op n = Ops.Stage) (U.backward_slice ast)
    in
    let optimize k =
      match beam_search, hand_coded_optimizations with
      | Some bs, _ -> bs k
      | None, Some f when k.applied_opts = [] && not has_stage -> f k
      | _ -> k
    in
    let opts_to_apply = Option.bind ki (fun i -> i.opts_to_apply) in
    let k =
      match opts_to_apply with
      | Some opts ->
          List.iter (fun opt -> ignore (apply_opt k opt)) opts;
          k
      | None -> optimize k
    in
    let name_override =
      match ki with
      | Some i when i.name <> "" && i.name <> "test" -> Some i.name
      | _ -> None
    in
    get_optimized_ast ?name_override k
