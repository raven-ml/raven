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

let err_no_locals = "can't use locals"
let err_locals_needed = "locals needed for opt"
let err_no_reduce_tc = "no reduce ops for TensorCore"
let err_invalid_tc_choice = "invalid tensor core choice"
let err_tc_first = "tensor core opts must be first"
let err_no_tc_available = "no tensor core available"
let err_range_missing = "range not found"

let scalar_eq (a : Dtype.scalar) (b : Dtype.scalar) = a = b

let scalar_is_float32 s = s = Dtype.Float32

(* Range-view helpers *)

let range_view u =
  match U.as_range u with
  | Some v -> v
  | None -> raise (Opt_error "postrange: not a range node")

let range_size u = (range_view u).size
let range_axis u = (range_view u).axis
let range_sub u = (range_view u).sub
let range_kind u = (range_view u).kind
let is_range u = Option.is_some (U.as_range u)
let const_int_or default u =
  match U.const_int_value u with Some n -> n | None -> default

let range_int_size u = const_int_or 0 (range_size u)

let range_max_extent u = U.vmax u + 1

(* Cached shape data recomputed after every AST mutation. *)
type shape = {
  rngs : U.t list;
  axis_types : Axis_type.t list;
  full_shape : U.t list;
  shape_str : string list;
}

let compute_shape ast =
  let rngs =
    U.toposort ast
    |> List.filter (fun u -> is_range u && U.vmax u > 0)
    |> List.sort (fun a b ->
         compare
           (Axis_type.to_pos (range_kind a), range_axis a, range_sub a)
           (Axis_type.to_pos (range_kind b), range_axis b, range_sub b))
  in
  let axis_types = List.map range_kind rngs in
  let full_shape = List.map (fun r -> U.simplify (range_size r)) rngs in
  let cnt = Hashtbl.create 8 in
  let shape_str =
    List.map
      (fun at ->
        let n = match Hashtbl.find_opt cnt at with Some n -> n | None -> 0 in
        Hashtbl.replace cnt at (n + 1);
        strf "%s%d" (Axis_type.letter at) n)
      axis_types
  in
  { rngs; axis_types; full_shape; shape_str }

(* Scheduler state: wraps a kernel AST and tracks applied optimisations. *)
type t = {
  mutable ast : U.t;
  ren : Renderer.t;
  mutable dont_use_locals : bool;
  mutable applied_opts : U.Opt.t list;
  mutable tensor_core : Tc.t option;
  mutable opt_range : int;
  mutable shape : shape;
}

let refresh t = t.shape <- compute_shape t.ast

let create ast ren =
  let dont_use_locals, applied_opts =
    match U.as_kernel_info ast with
    | Some ki -> ki.dont_use_locals, ki.applied_opts
    | None -> false, []
  in
  let shape = compute_shape ast in
  let max_axis =
    List.fold_left (fun acc r -> max acc (range_axis r)) 0 shape.rngs
  in
  {
    ast; ren; dont_use_locals; applied_opts;
    tensor_core = None; opt_range = max_axis + 1; shape;
  }

(* Accessors read from the cached shape. *)
let rngs t = t.shape.rngs
let shape_len t = List.length t.shape.rngs
let full_shape t = t.shape.full_shape
let axis_types t = t.shape.axis_types
let shape_str t = t.shape.shape_str

let shape_str_to_axis t nms =
  List.map
    (fun nm ->
      match List.find_index (fun s -> s = nm) t.shape.shape_str with
      | Some i -> i
      | None -> raise (Opt_error (strf "shape_str_to_axis: %S not found" nm)))
    nms

let ast t = t.ast
let ren t = t.ren
let applied_opts t = t.applied_opts
let tensor_core t = t.tensor_core

let copy t = { t with ast = t.ast }

type snapshot = {
  snap_ast : U.t;
  snap_dont_use_locals : bool;
  snap_applied_opts : U.Opt.t list;
  snap_tensor_core : Tc.t option;
  snap_opt_range : int;
  snap_shape : shape;
}

let snapshot t =
  {
    snap_ast = t.ast;
    snap_dont_use_locals = t.dont_use_locals;
    snap_applied_opts = t.applied_opts;
    snap_tensor_core = t.tensor_core;
    snap_opt_range = t.opt_range;
    snap_shape = t.shape;
  }

let restore t s =
  t.ast <- s.snap_ast;
  t.dont_use_locals <- s.snap_dont_use_locals;
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

(* Loop ranges eligible for promotion to Global: must appear in all
   Stage nodes' ranges. *)
let globalizable_rngs t =
  let out =
    List.filter (fun r -> range_kind r = Axis_type.Loop) (output_rngs t)
  in
  List.fold_left
    (fun acc node ->
      match U.as_stage node with
      | Some _ ->
          let stage_ranges = U.ranges node in
          List.filter (fun r -> List.memq r stage_ranges) acc
      | None -> acc)
    out (U.toposort t.ast)

(* Promote eligible Loop ranges to Global. *)
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
                  ~dtype:(Dtype.val_of (U.dtype r))
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
  | Global -> "blue"
  | Thread -> "BLUE"
  | Local -> "cyan"
  | Warp -> "CYAN"
  | Loop -> "WHITE"
  | Upcast -> "yellow"
  | Group_reduce -> "RED"
  | Reduce -> "red"
  | Unroll -> "magenta"
  | Placeholder -> "white"

let colors t =
  let out = output_rngs t in
  let glob = globalizable_rngs t in
  List.map2
    (fun at r ->
      if t.dont_use_locals && at = Axis_type.Global then "BLUE"
      else if at = Axis_type.Loop && not (List.memq r out) then "BLACK"
      else if at = Axis_type.Loop && not (List.memq r glob) then "white"
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

(* Sanitise a kernel name to a valid identifier. *)
let to_function_name s =
  let buf = Buffer.create (String.length s) in
  String.iter
    (fun c ->
      match c with
      | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '_' -> Buffer.add_char buf c
      | _ -> Buffer.add_string buf (strf "%02X" (Char.code c)))
    s;
  Buffer.contents buf

let kernel_cnt : (string, int) Hashtbl.t = Hashtbl.create 16

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

(* Build a debug kernel name from the schedule's special and range sizes,
   deduplicating against previously generated names. *)
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
    List.map (fun s -> string_of_int (U.vmax s + 1)) specials
  in
  let rng_strs = List.map (fun r -> render_size (range_size r)) (rngs t) in
  let raw = k_type ^ "_" ^ String.concat "_" (special_strs @ rng_strs) in
  let fn = to_function_name raw in
  let cnt = 1 + Option.value ~default:0 (Hashtbl.find_opt kernel_cnt fn) in
  Hashtbl.replace kernel_cnt fn cnt;
  raw ^ if cnt > 1 then strf "n%d" (cnt - 1) else ""

(* Finalize the kernel: generate a debug name, flatten ranges, and attach
   updated kernel_info with a tag marking it as optimized. *)
let get_optimized_ast ?name_override t =
  let name = match name_override with Some n -> n | None -> make_kernel_name t in
  t.ast <- pm_flatten_range t.ast;
  refresh t;
  let ki : U.kernel_info =
    {
      name;
      axis_types = [];
      dont_use_locals = t.dont_use_locals;
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
        U.range ~size:(U.const_int amount) ~axis ~kind:new_kind ()
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
  const_dims t [ Axis_type.Global; Axis_type.Local; Axis_type.Loop ]

let unrollable_dims t =
  const_dims t [ Axis_type.Group_reduce; Axis_type.Reduce ]

let bufs t =
  List.rev
    (List.filter (fun x -> U.op x = Ops.Index) (U.toposort t.ast))

let output_shape t =
  List.map2
    (fun s at ->
      match at with
      | Axis_type.Reduce | Axis_type.Unroll | Axis_type.Group_reduce ->
          U.const_int 1
      | _ -> s)
    (full_shape t) (axis_types t)

let upcasted t = List.length (axes_of t [ Axis_type.Upcast; Axis_type.Unroll ])

let group_for_reduces t = List.length (axes_of t [ Axis_type.Group_reduce ])

(* Resolve an opt's axis to a real range index. *)
let real_axis t op axis =
  match op, axis with
  | _, None | U.Opt.Tc _, _ -> -1
  | U.Opt.Unroll _, Some a ->
      check (a >= 0) "invalid unroll axis";
      nth_or_error (unrollable_dims t) a "invalid unroll axis"
  | (U.Opt.Group _ | U.Opt.Grouptop _), Some a ->
      check (a >= 0) "invalid group axis";
      nth_or_error (axes_of t [ Axis_type.Reduce ]) a "invalid group axis"
  | _, Some a ->
      check (a >= 0 && a < shape_len t) "invalid axis";
      a

let allow_tf32 = match Sys.getenv_opt "ALLOW_TF32" with Some "1" -> true | _ -> false

let argsort perm =
  let n = List.length perm in
  let inv = Array.make n 0 in
  List.iteri (fun i x -> if x >= 0 && x < n then inv.(x) <- i) perm;
  Array.to_list inv

(* Apply TC opt/reduce splits, return the new ranges in apply order. *)
let apply_tc_shifts t axes (tc : Tc.t) =
  let warp =
    ref (U.range ~size:(U.const_int tc.threads) ~axis:(-1)
           ~kind:Axis_type.Warp ())
  in
  let ne = ref [] in
  let split dim amt kind ?input_new_rng () =
    let replaced, new_rng = shift_to t axes.(dim) amt kind ?input_new_rng in
    axes.(dim) <- replaced;
    ne := new_rng :: !ne
  in
  let apply_opt opt_str =
    let dim = Char.code opt_str.[1] - Char.code '0' in
    match opt_str.[0] with
    | 'l' ->
        let lane =
          U.alu_binary ~op:Ops.Floormod ~lhs:!warp ~rhs:(U.const_int 2)
        in
        split dim 2 Axis_type.Local ~input_new_rng:lane ();
        warp :=
          U.alu_binary ~op:Ops.Floordiv ~lhs:!warp ~rhs:(U.const_int 2)
    | 'u' -> split dim 2 Axis_type.Upcast ()
    | c -> raise (Opt_error (strf "unsupported tc opt: %c" c))
  in
  List.iter apply_opt tc.opts;
  List.iter (fun (_, amt) -> split 2 amt Axis_type.Unroll ()) (Tc.get_reduce_axes tc);
  List.rev !ne

(* Build the WMMA node and substitute it for the tagged reduce. *)
let build_wmma_node t (tc : Tc.t) ne =
  let tagged_red =
    match List.find_opt
      (fun x -> is_reduce x && U.node_tag x = Some "TC")
      (U.toposort t.ast) with
    | Some red -> red
    | None -> raise (Opt_error "tagged tensor-core reduce not found")
  in
  let tne = List.map (U.with_tag "1") ne in
  let ret = U.substitute (List.combine ne tne) tagged_red in
  let ret_src =
    match U.as_reduce ret with
    | Some rv -> rv.src
    | None -> raise (Opt_error "tensor-core node is not a reduce")
  in
  let mul_src =
    if U.op ret_src = Ops.Cast then (U.src ret_src).(0) else ret_src
  in
  let srcs = U.children mul_src in
  let perm0, perm1 =
    try Tc.permutes_for_shape_str tc (Tc.base_shape_str tc)
    with Failure msg -> raise (Opt_error msg)
  in
  let srcs =
    List.mapi
      (fun i src ->
        let p = if i = 0 then perm0 else perm1 in
        U.substitute
          (List.combine tne (List.map (fun j -> List.nth ne j) (argsort p)))
          src)
      srcs
  in
  (* Compute upcast/reduce axes *)
  let n_reduce = List.length (Tc.get_reduce_axes tc) in
  let tc_reduce_axes =
    shape_str_to_axis t (List.init n_reduce (fun i -> strf "r%d" i))
  in
  let base_ua =
    List.map (fun s -> (s, 2)) (shape_str_to_axis t (Tc.base_upcast_axes tc))
  in
  let log2 n = int_of_float (log (float_of_int n) /. log 2.0) in
  let a_ept, b_ept, c_ept = tc.elements_per_thread in
  let tc_upcast_axes =
    List.init 3 (fun i ->
        let n = log2 [| a_ept; b_ept; c_ept |].(i) in
        List.filteri (fun j _ -> j < n) base_ua)
  in
  (* Convert axes to range numbers *)
  let rngs_now = rngs t in
  let tc_upcast_axes =
    List.map
      (fun v ->
        List.map (fun (a, sz) -> (range_axis (List.nth rngs_now a), sz)) v)
      tc_upcast_axes
  in
  let tc_reduce_axes =
    List.map (fun a -> range_axis (List.nth rngs_now a)) tc_reduce_axes
  in
  (* Build the WMMA node *)
  let out_dt_val = Dtype.Val.of_scalar tc.dtype_out in
  let src0, src1 =
    match srcs with
    | a :: b :: _ -> a, b
    | _ -> raise (Opt_error "tensor-core multiply must have two operands")
  in
  let ua, ub, uc =
    match tc_upcast_axes with
    | a :: b :: c :: _ -> a, b, c
    | _ -> raise (Opt_error "tensor-core upcast axes missing")
  in
  let with_missing_tc_axes axes =
    List.fold_left
      (fun acc (rn, _) ->
        if List.mem_assoc rn acc then acc else acc @ [ (rn, 1) ])
      axes (ua @ ub)
  in
  let ua, ub, uc =
    (with_missing_tc_axes ua, with_missing_tc_axes ub, with_missing_tc_axes uc)
  in
  let wmma_info : U.wmma_info =
    {
      name = Tc.to_string tc;
      dims = tc.dims;
      dtype_in = tc.dtype_in;
      dtype_out = tc.dtype_out;
      device = Renderer.device t.ren;
      threads = tc.threads;
      upcast_axes = (ua, ub, uc);
      reduce_axes = [];
    }
  in
  let wmma =
    U.with_tag "1"
      (U.wmma
         ~a:src0 ~b:src1
         ~c:
           (U.const
              (Const.of_view (Dtype.Val.vec c_ept out_dt_val)
                 (Const.Float 0.0)))
         ~info:wmma_info
         ~dtype:out_dt_val)
  in
  (* Preserve extra reduces *)
  let red_ranges =
    match U.as_reduce tagged_red with
    | Some rv -> rv.ranges
    | None -> raise (Opt_error "tagged tensor-core node is not a reduce")
  in
  let red_range_nodes = U.find_nodes is_range (U.sink red_ranges) in
  let extra_reduces =
    List.filter
      (fun x -> not (List.mem (range_axis x) tc_reduce_axes))
      red_range_nodes
  in
  let tc_uop =
    if extra_reduces <> [] then
      U.reduce ~op:Ops.Add ~src:wmma ~ranges:extra_reduces
        ~dtype:(Dtype.val_of (U.dtype wmma))
    else wmma
  in
  t.ast <- U.substitute [ (tagged_red, tc_uop) ] t.ast;
  refresh t

(* Shared memory size check for group/reduce opts. *)
let check_shared_memory t opt amt red_opt =
  let is_group =
    match opt with U.Opt.Group _ | U.Opt.Grouptop _ -> true | _ -> false
  in
  let is_padding_skippable =
    match opt with U.Opt.Nolocals | U.Opt.Padto _ -> true | _ -> false
  in
  if red_opt <> None
     && (is_group
         || (group_for_reduces t > 0 && not is_padding_skippable))
  then begin
    let fs = full_shape t in
    let upcast_local_sz =
      prod
        (List.map
           (fun a -> const_int_or 1 (List.nth fs a))
           (axes_of t
              [
                Axis_type.Upcast;
                Axis_type.Warp;
                Axis_type.Local;
                Axis_type.Group_reduce;
              ]))
    in
    let red = Option.get red_opt in
    let smem_sz = amt * upcast_local_sz * Dtype.itemsize (U.dtype red) in
    check
      (smem_sz <= Renderer.shared_max t.ren)
      (strf "exceeds shared memory: needs %d, max %d" smem_sz
         (Renderer.shared_max t.ren))
  end

(* Check that a GROUP_REDUCE is not inside another reduce. *)
let check_no_nested_group t r red_opt =
  if red_opt <> None then begin
    let reduce_node =
      List.find_opt
        (fun u ->
          match U.as_reduce u with
          | Some v ->
              let range_nodes = U.find_nodes is_range (U.sink v.ranges) in
              List.memq r range_nodes
          | None -> false)
        (U.toposort t.ast)
    in
    match reduce_node with
    | Some red_node ->
        let enclosing = U.ranges red_node in
        check
          (not
             (List.exists
                (fun u ->
                  let k = range_kind u in
                  k = Axis_type.Reduce
                  || k = Axis_type.Unroll
                  || k = Axis_type.Group_reduce)
                enclosing))
          "cannot have a GROUP_REDUCE inside another reduce"
    | None -> ()
  end

(* Per-opt validation for shift_to opts. *)
let validate_shift_opt t opt r amt rng_kind =
  match opt with
  | U.Opt.Unroll _ ->
      check (amt <= 32) "don't unroll more than 32";
      check
        (rng_kind = Axis_type.Group_reduce || rng_kind = Axis_type.Reduce)
        "unroll is for GROUP_REDUCE/REDUCE"
  | U.Opt.Upcast _ ->
      check
        (Renderer.device t.ren = "DSP" || amt <= 16)
        "don't upcast more than 16";
      check
        (rng_kind = Axis_type.Global
        || rng_kind = Axis_type.Local
        || rng_kind = Axis_type.Loop)
        "upcast is for GLOBAL/LOCAL/LOOP"
  | U.Opt.Local _ ->
      check (not t.dont_use_locals) err_no_locals;
      check
        (rng_kind = Axis_type.Global || rng_kind = Axis_type.Loop)
        "local is for globals"
  | U.Opt.Thread _ ->
      check (Renderer.has_threads t.ren) "target does not support threads";
      (match Renderer.global_max t.ren with
       | Some (gm :: _) -> check (amt <= gm) "too many threads"
       | _ -> raise (Opt_error "too many threads"));
      check
        (List.for_all (fun at -> at <> Axis_type.Thread) (axis_types t))
        "already threaded";
      check (List.memq r (globalizable_rngs t)) "can't apply thread to this dim"
  | U.Opt.Group _ | U.Opt.Grouptop _ ->
      check
        (List.for_all
           (fun o -> match o with U.Opt.Tc _ -> false | _ -> true)
           t.applied_opts)
        "no grouping with tensor cores";
      check (not t.dont_use_locals) err_no_locals;
      check (rng_kind = Axis_type.Reduce) "group is for reduce"
  | _ -> ()

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

let invalid_for_dtype dtype =
  match dtype with
  | Dtype.Val dt -> U.invalid ~dtype:dt ()
  | Dtype.Ptr _ -> U.invalid ()

(* Pad a range to a multiple of [amount]. *)
let apply_padto t r amount _red_opt =
  check (amount > 0) "invalid pad amount";
  check (Option.is_some (U.const_int_value (range_size r))) "only pad const axes";
  let rng_kind = range_kind r in
  check
    (rng_kind <> Axis_type.Upcast && rng_kind <> Axis_type.Unroll)
    "cannot pad upcasted";
  check (rng_kind <> Axis_type.Thread) "cannot pad thread";
  let old_size = range_int_size r in
  let new_sz = round_up old_size amount in
  check (old_size > new_sz / 4) "pad adds more than quadruple the work";
  let v = range_view r in
  let replaced_rng =
    U.range ~size:(U.const_int new_sz) ~axis:v.axis ~sub:v.sub ~kind:v.kind
      ~dtype:(Dtype.val_of (U.dtype r))
      ()
  in
  let valid =
    U.alu_binary ~op:Ops.Cmplt ~lhs:replaced_rng ~rhs:(U.const_int old_size)
  in
  let store_targets =
    U.toposort t.ast
    |> List.filter_map (fun n ->
         match U.as_store n with Some { dst; _ } -> Some dst | None -> None)
  in
  let subs =
    List.fold_left
      (fun acc b ->
        match U.as_index b with
        | Some { ptr; idxs = [ idx ] } ->
            let idx, idx_valid = get_idx_valid idx in
            if not (U.in_backward_slice r idx) then acc
            else
            let idx_dtype = Dtype.val_of (U.dtype idx) in
            let valid_idx =
              U.alu_binary ~op:Ops.And ~lhs:valid ~rhs:idx_valid
            in
            let guarded_idx =
              U.alu_ternary ~op:Ops.Where ~a:valid_idx ~b:idx
                ~c:(U.invalid ~dtype:idx_dtype ())
            in
            let guarded =
              U.index ~ptr ~idxs:[guarded_idx] ~as_ptr:(Dtype.is_ptr (U.dtype b))
                ()
            in
            let replacement =
              if List.exists (fun target -> target == b) store_targets then
                guarded
              else
                U.alu_ternary ~op:Ops.Where ~a:valid ~b:guarded
                  ~c:(invalid_for_dtype (U.dtype b))
            in
            (b, replacement) :: acc
        | Some _ -> acc
        | _ -> acc)
      [ (r, replaced_rng) ] (bufs t)
  in
  t.ast <- U.substitute subs t.ast;
  refresh t

(* Swap two global ranges' axis numbers. *)
let apply_swap t r with_axis =
  let altrng = nth_or_error (rngs t) with_axis "invalid swap axis" in
  check
    (range_kind r = Axis_type.Global && range_kind altrng = Axis_type.Global)
    "swap only for globals";
  let rv = range_view r and av = range_view altrng in
  let r' =
    U.with_tag "1"
      (U.range ~size:rv.size ~sub:av.sub ~axis:av.axis ~kind:rv.kind
         ~dtype:(Dtype.val_of (U.dtype r))
         ~parents:rv.parents
         ())
  in
  let alt' =
    U.with_tag "1"
      (U.range ~size:av.size ~sub:rv.sub ~axis:rv.axis ~kind:av.kind
         ~dtype:(Dtype.val_of (U.dtype altrng))
         ~parents:av.parents
         ())
  in
  t.ast <- U.substitute [ (r, r'); (altrng, alt') ] t.ast;
  t.ast <-
    U.graph_rewrite
      (fun node ->
        match U.node_tag node with
        | Some _ -> Some (U.replace node ~node_tag:None ())
        | None -> None)
      t.ast;
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
        let in0_sc = Dtype.scalar (U.dtype in0) in
        let in1_sc = Dtype.scalar (U.dtype in1) in
        let red_sc = Dtype.scalar (U.dtype red) in
        let in0_ranges = U.ranges in0 in
        let in1_ranges = U.ranges in1 in
        let red_ranges = red_view.ranges in
        let sort_desc =
          List.sort (fun a b -> compare (range_axis b) (range_axis a))
        in
        let try_tc (tc : Tc.t) =
          let snap = snapshot t in
          try
            let result =
              if
                (Renderer.device t.ren = "CUDA" || Renderer.device t.ren = "NV")
                && scalar_is_float32 tc.dtype_in
                && not allow_tf32
              then None
              else if
                (not (scalar_eq in0_sc tc.dtype_in))
                || (not (scalar_eq in1_sc tc.dtype_in))
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
                    t.ast <-
                      U.substitute [ (red, U.with_tag "TC" red) ] t.ast;
                    refresh t;
                    if not (pad_tc_axes t axes tc tc_opt) then None
                    else begin
                      let ne = apply_tc_shifts t axes tc in
                      if use_tc <> 2 then build_wmma_node t tc ne;
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
    | U.Opt.Nolocals ->
        check
          (List.for_all
             (fun at ->
               at <> Axis_type.Warp
               && at <> Axis_type.Local
               && at <> Axis_type.Group_reduce)
             (axis_types t))
          "no locals can't have locals";
        t.dont_use_locals <- true;
        None
    | Tc { axis; tc_select; tc_opt; use_tc } ->
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
    | Padto { axis = _; amount } ->
        let ra = real_axis t opt (U.Opt.axis opt) in
        let r = List.nth (rngs t) ra in
        apply_padto t r amount (reduceop t);
        None
    | Swap { axis = _; with_axis } ->
        let ra = real_axis t opt (U.Opt.axis opt) in
        let r = List.nth (rngs t) ra in
        apply_swap t r with_axis;
        None
    | _ ->
        let ra = real_axis t opt (U.Opt.axis opt) in
        let r = List.nth (rngs t) ra in
        let red_opt = reduceop t in
        (match opt with
         | Local _ | Group _ | Grouptop _ ->
             check (Renderer.has_local t.ren) err_locals_needed
         | _ -> ());
        let new_kind =
          match opt with
          | Local _ -> Axis_type.Local
          | Upcast _ -> Axis_type.Upcast
          | Unroll _ -> Axis_type.Unroll
          | Group _ | Grouptop _ -> Axis_type.Group_reduce
          | Thread _ -> Axis_type.Thread
          | _ -> assert false
        in
        let amt =
          match U.Opt.amount opt with
          | Some 0 -> range_max_extent r
          | Some a -> a
          | None -> range_max_extent r
        in
        check (amt > 0) "invalid optimization amount";
        check_shared_memory t opt amt red_opt;
        (match opt with
         | Group _ | Grouptop _ -> check_no_nested_group t r red_opt
         | _ -> ());
        validate_shift_opt t opt r amt (range_kind r);
        let top =
          match opt with Grouptop _ | Thread _ -> true | _ -> false
        in
        Some (shift_to ~top t r amt new_kind)
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
