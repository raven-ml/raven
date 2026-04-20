(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_ir
module K = Kernel

let strf = Printf.sprintf
let prod lst = List.fold_left ( * ) 1 lst

let const_int_or default node = match K.const_arg node with
  | Some (Int n) -> Int64.to_int n | _ -> default

exception Opt_error of string

let check cond msg = if not cond then raise (Opt_error msg)

let nth_or_error lst i msg =
  match List.nth_opt lst i with Some v -> v | None -> raise (Opt_error msg)

(* Cached shape data — recomputed after every AST mutation. *)
type shape = {
  rngs : K.t list;
  axis_types : Axis_kind.t list;
  full_shape : K.t list;
  shape_str : string list;
}

let compute_shape ast =
  let rngs =
    K.toposort ast
    |> List.filter (fun u -> K.is_range u && const_int_or 0 (K.range_size u) > 1)
    |> List.sort (fun a b ->
         compare (Axis_kind.to_pos (K.range_kind a), K.range_axis a)
                 (Axis_kind.to_pos (K.range_kind b), K.range_axis b)) in
  let axis_types = List.map K.range_kind rngs in
  let full_shape = List.map K.range_size rngs in
  let cnt = Hashtbl.create 8 in
  let shape_str = List.map (fun at ->
    let n = match Hashtbl.find_opt cnt at with Some n -> n | None -> 0 in
    Hashtbl.replace cnt at (n + 1);
    strf "%s%d" (Axis_kind.letter at) n) axis_types in
  { rngs; axis_types; full_shape; shape_str }

(* Scheduler state: wraps a kernel AST and tracks applied optimisations. *)
type t = {
  mutable ast : K.t;
  ren : Renderer.t;
  mutable dont_use_locals : bool;
  mutable applied_opts : K.Opt.t list;
  mutable tensor_core : Tc.t option;
  mutable opt_range : int;
  mutable shape : shape;
}

let refresh t = t.shape <- compute_shape t.ast

let create ast ren =
  let dont_use_locals, applied_opts = match K.view ast with
    | Sink { kernel_info = Some ki; _ } -> ki.dont_use_locals, ki.applied_opts
    | _ -> false, [] in
  let all_rngs = K.find_nodes K.is_range ast in
  let max_axis = List.fold_left (fun acc r -> max acc (K.range_axis r)) 0 all_rngs in
  let shape = compute_shape ast in
  { ast; ren; dont_use_locals; applied_opts;
    tensor_core = None; opt_range = max_axis + 1; shape }

(* Accessors — read from cached shape. *)
let rngs t = t.shape.rngs
let shape_len t = List.length t.shape.rngs
let full_shape t = t.shape.full_shape
let axis_types t = t.shape.axis_types
let shape_str t = t.shape.shape_str

let shape_str_to_axis t nms =
  List.map (fun nm ->
    match List.find_index (fun s -> s = nm) t.shape.shape_str with
    | Some i -> i
    | None -> failwith (strf "shape_str_to_axis: %S not found" nm))
    nms

let ast t = t.ast
let ren t = t.ren
let applied_opts t = t.applied_opts
let tensor_core t = t.tensor_core

let copy t = { t with ast = t.ast }

(* Ranges that appear in END nodes with non-Reduce axis type. *)
let output_rngs t =
  List.concat_map (fun s ->
    match K.view s with
    | End { ranges; _ } ->
        let sink_ranges = K.find_nodes K.is_range (K.sink ranges) in
        List.filter (fun r -> K.range_kind r <> Axis_kind.Reduce) sink_ranges
    | _ -> [])
    (K.children t.ast)

(* Loop ranges eligible for promotion to Global: must appear in all
   BUFFERIZE nodes' ranges. *)
let globalizable_rngs t =
  let out = List.filter (fun r ->
    K.range_kind r = Axis_kind.Loop) (output_rngs t) in
  List.fold_left (fun acc node ->
    match K.view node with
    | Bufferize { ranges; _ } ->
        List.filter (fun r -> List.memq r ranges) acc
    | _ -> acc)
    out (K.toposort t.ast)

(* Promote eligible Loop ranges to Global. *)
let convert_loop_to_global t =
  if Renderer.has_local t.ren then begin
    let glob = globalizable_rngs t in
    let subs = List.filter_map (fun r ->
      if List.memq r glob then
        Some (r, K.range ~size:(K.range_size r)
                   ~axis:(K.range_axis r) ~sub:(K.range_sub r)
                   ~kind:Axis_kind.Global
                   ~dtype:(Dtype.val_of (K.dtype r)) ())
      else None)
      (rngs t) in
    if subs <> [] then (t.ast <- K.substitute subs t.ast; refresh t)
  end

let reduceop t =
  List.find_opt (fun x -> match K.view x with Reduce _ -> true | _ -> false)
    (K.backward_slice t.ast)

let colors t =
  let out = output_rngs t in
  let glob = globalizable_rngs t in
  List.map2 (fun at r ->
    if t.dont_use_locals && at = Axis_kind.Global then "BLUE"
    else if at = Axis_kind.Loop && not (List.memq r out) then "BLACK"
    else if at = Axis_kind.Loop && not (List.memq r glob) then "white"
    else Axis_kind.color at)
    (axis_types t) (rngs t)

let render_size sz = match K.const_arg sz with
  | Some (Int n) -> Int64.to_string n | _ -> "s"

let colored_shape t =
  String.concat " " (List.map2 (fun rng color ->
    strf "%4s:%s" (render_size (K.range_size rng)) color)
    (rngs t) (colors t))

(* Sanitise a kernel name to a valid identifier. *)
let to_function_name s =
  let buf = Buffer.create (String.length s) in
  String.iter (fun c -> match c with
    | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '_' -> Buffer.add_char buf c
    | _ -> Buffer.add_string buf (strf "%02X" (Char.code c))) s;
  Buffer.contents buf

let kernel_cnt : (string, int) Hashtbl.t = Hashtbl.create 16

(* Finalize the kernel: generate a debug name, flatten ranges,
   and attach updated kernel_info with a tag marking it as optimized. *)
let get_optimized_ast ?name_override t =
  let name = match name_override with
    | Some n -> n
    | None ->
        let k_type = if reduceop t <> None then "r" else "E" in
        let specials = List.sort (fun a b ->
            match K.view a, K.view b with
            | Special { dim = da; _ }, Special { dim = db; _ } ->
                Special_dim.compare da db
            | _ -> 0)
          (List.filter (fun n -> match K.view n with Special _ -> true | _ -> false)
             (K.toposort t.ast)) in
        let special_strs = List.map (fun s -> match K.view s with
          | Special { size; _ } -> render_size size | _ -> "?") specials in
        let rng_strs = List.map (fun rng ->
          render_size (K.range_size rng)) (rngs t) in
        let raw = k_type ^ "_" ^ String.concat "_"
          (special_strs @ rng_strs) in
        let fn = to_function_name raw in
        let cnt = 1 + (match Hashtbl.find_opt kernel_cnt fn with
          | Some c -> c | None -> 0) in
        Hashtbl.replace kernel_cnt fn cnt;
        raw ^ (if cnt > 1 then strf "n%d" (cnt - 1) else "")
  in
  t.ast <- Simplify.pm_flatten_range t.ast; refresh t;
  let ki = { K.name; axis_kinds = []; dont_use_locals = t.dont_use_locals;
             applied_opts = t.applied_opts; opts_to_apply = None;
             estimates = None } in
  K.with_tag "1"
    (K.sink ~kernel_info:ki
       (match K.view t.ast with Sink { srcs; _ } -> srcs | _ -> [t.ast]))

(* Split [rng] by [amount]: the original range shrinks to size/amount,
   a new range of [amount] is created with [new_kind].  When [top] is true
   the new range is the high part; otherwise it is the low part. *)
let shift_to ?(top = false) ?input_new_rng t rng amount new_kind =
  let size = K.range_size rng in
  let old_sz = match K.divides size amount with
    | Some q -> q
    | None -> raise (Opt_error (strf "shift_to: %d can't divide range" amount)) in
  let new_rng = match input_new_rng with
    | Some r -> r
    | None ->
        let axis = t.opt_range in
        t.opt_range <- t.opt_range + 1;
        K.range ~size:(K.const_int amount) ~axis ~kind:new_kind () in
  let replaced_rng = K.range ~size:old_sz
    ~axis:(K.range_axis rng) ~sub:(K.range_sub rng)
    ~kind:(K.range_kind rng) ~dtype:(Dtype.val_of (K.dtype rng)) () in
  let open K.O in
  let sub_axis =
    if top then new_rng * old_sz + replaced_rng
    else replaced_rng * K.const_int amount + new_rng in
  t.ast <- K.substitute [(rng, sub_axis)] t.ast; refresh t;
  (replaced_rng, new_rng)

let ranges_of t kinds =
  List.filter (fun r -> List.mem (K.range_kind r) kinds) (rngs t)

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
  prod (List.map (fun a -> const_int_or 1 (List.nth fs a))
    (axes_of t [Axis_kind.Upcast; Axis_kind.Unroll]))

let upcastable_dims t =
  const_dims t [Axis_kind.Global; Axis_kind.Local; Axis_kind.Loop]

let unrollable_dims t =
  const_dims t [Axis_kind.Group_reduce; Axis_kind.Reduce]

let bufs t =
  List.rev (List.filter (fun x ->
    match K.view x with Index _ -> true | _ -> false)
    (K.toposort t.ast))

let output_shape t =
  List.map2 (fun s at ->
    match at with
    | Axis_kind.Reduce | Axis_kind.Unroll | Axis_kind.Group_reduce ->
        K.const_int 1
    | _ -> s)
    (full_shape t) (axis_types t)

let upcasted t =
  List.length (axes_of t [Axis_kind.Upcast; Axis_kind.Unroll])

let group_for_reduces t =
  List.length (axes_of t [Axis_kind.Group_reduce])

(* Resolve an opt's axis to a real range index. *)
let real_axis t op axis =
  match op, axis with
  | _, None | K.Opt.Tc _, _ -> -1
  | K.Opt.Unroll _, Some a ->
      nth_or_error (unrollable_dims t) a "invalid unroll axis"
  | K.Opt.Group _, Some a | K.Opt.Grouptop _, Some a ->
      nth_or_error (axes_of t [Axis_kind.Reduce]) a "invalid group axis"
  | _, Some a ->
      check (a < shape_len t) "invalid axis"; a

let range_int_size rng = const_int_or 0 (K.range_size rng)

let allow_tf32 = match Sys.getenv_opt "ALLOW_TF32" with
  | Some "1" -> true | _ -> false

let argsort perm =
  let n = List.length perm in
  let inv = Array.make n 0 in
  List.iteri (fun i x -> if x >= 0 && x < n then inv.(x) <- i) perm;
  Array.to_list inv

(* Apply TC opt/reduce splits, return the new ranges (reversed). *)
let apply_tc_shifts t axes (tc : Tc.t) =
  let warp = ref (K.range ~size:(K.const_int tc.threads) ~axis:(-1)
                    ~kind:Axis_kind.Warp ()) in
  let ne = ref [] in
  List.iter (fun opt_str ->
    let dim_idx = Char.code opt_str.[1] - Char.code '0' in
    if opt_str.[0] = 'l' then begin
      let replaced, new_rng = shift_to t axes.(dim_idx) 2
        Axis_kind.Local ~input_new_rng:(K.binary ~op:`Mod
          ~lhs:!warp ~rhs:(K.const_int 2)) in
      axes.(dim_idx) <- replaced;
      warp := K.binary ~op:`Idiv ~lhs:!warp ~rhs:(K.const_int 2);
      ne := new_rng :: !ne
    end else if opt_str.[0] = 'u' then begin
      let replaced, new_rng = shift_to t axes.(dim_idx) 2
        Axis_kind.Upcast in
      axes.(dim_idx) <- replaced;
      ne := new_rng :: !ne
    end else
      failwith (strf "unsupported tc opt: %c" opt_str.[0]))
    tc.opts;
  List.iter (fun (_, amt) ->
    let replaced, new_rng = shift_to t axes.(2) amt Axis_kind.Unroll in
    axes.(2) <- replaced;
    ne := new_rng :: !ne)
    (Tc.get_reduce_axes tc);
  List.rev !ne

(* Build the WMMA node and substitute it for the tagged reduce. *)
let build_wmma_node t (tc : Tc.t) ne =
  let tagged_red = List.find (fun x ->
    match K.view x with Reduce _ -> K.tag x = Some "TC" | _ -> false)
    (K.toposort t.ast) in
  let tne = List.map (K.with_tag "1") ne in
  let ret = K.substitute (List.combine ne tne) tagged_red in
  let ret_src = match K.view ret with
    | Reduce { src; _ } -> src | _ -> assert false in
  let mul_src = match K.view ret_src with
    | Cast { src; _ } -> src | _ -> ret_src in
  let srcs = K.children mul_src in
  let perm0, perm1 = Tc.permutes_for_shape_str tc (Tc.base_shape_str tc) in
  let srcs = List.mapi (fun i src ->
    let p = if i = 0 then perm0 else perm1 in
    K.substitute (List.combine tne
      (List.map (fun j -> List.nth ne j) (argsort p))) src) srcs in
  (* Compute upcast/reduce axes *)
  let n_reduce = List.length (Tc.get_reduce_axes tc) in
  let tc_reduce_axes = shape_str_to_axis t
    (List.init n_reduce (fun i -> strf "r%d" i)) in
  let base_ua = List.map (fun s -> (s, 2))
    (shape_str_to_axis t (Tc.base_upcast_axes tc)) in
  let log2 n = int_of_float (log (float_of_int n) /. log 2.0) in
  let a_ept, b_ept, c_ept = tc.elements_per_thread in
  let tc_upcast_axes = List.init 3 (fun i ->
    let n = log2 [|a_ept; b_ept; c_ept|].(i) in
    List.filteri (fun j _ -> j < n) base_ua) in
  (* Convert axes to range numbers *)
  let rngs_now = rngs t in
  let tc_upcast_axes = List.map (fun v ->
    List.map (fun (a, sz) ->
      (K.range_axis (List.nth rngs_now a), sz)) v) tc_upcast_axes in
  let tc_reduce_axes = List.map (fun a ->
    K.range_axis (List.nth rngs_now a)) tc_reduce_axes in
  (* Build the WMMA node *)
  let out_dt = Dtype.of_scalar tc.dtype_out in
  let src0, src1 = match srcs with [a; b] -> a, b | _ -> assert false in
  let ua, ub, uc = match tc_upcast_axes with
    | [a; b; c] -> a, b, c | _ -> assert false in
  let contract_src src axes ept =
    K.with_tag "1" (K.contract ~src ~axes
      ~dtype:(Dtype.Val.vec ept (Dtype.val_of (Dtype.scalarize (K.dtype src))))) in
  let wmma = K.with_tag "1" (K.wmma
    ~name:(Tc.to_string tc)
    ~a:(contract_src src0 ua a_ept)
    ~b:(contract_src src1 ub b_ept)
    ~c:(K.broadcast (K.const (Const.float (Dtype.val_of out_dt) 0.0)) c_ept)
    ~dtype:(Dtype.Val.vec c_ept (Dtype.val_of out_dt))
    ~dims:tc.dims ~dtype_in:tc.dtype_in ~dtype_out:tc.dtype_out
    ~device:(Renderer.device t.ren) ~threads:tc.threads
    ~upcast_axes:(ua, ub, uc) ~reduce_axes:[]) in
  let tc_uop = K.with_tag "1" (K.unroll ~src:wmma ~axes:uc ~dtype:(Dtype.val_of out_dt)) in
  (* Preserve extra reduces *)
  let red_range_nodes = K.find_nodes K.is_range
    (K.sink (match K.view tagged_red with
      | Reduce { ranges; _ } -> ranges | _ -> [])) in
  let extra_reduces = List.filter (fun x ->
    not (List.mem (K.range_axis x) tc_reduce_axes)) red_range_nodes in
  let tc_uop = if extra_reduces <> [] then
    K.reduce ~op:`Add ~src:tc_uop ~ranges:extra_reduces
      ~dtype:(Dtype.val_of (K.dtype tc_uop))
  else tc_uop in
  t.ast <- K.substitute [(tagged_red, tc_uop)] t.ast; refresh t

(* Shared memory size check for group/reduce opts. *)
let check_shared_memory t opt amt red_opt =
  let is_group = match opt with K.Opt.Group _ | Grouptop _ -> true | _ -> false in
  if red_opt <> None
     && (is_group || (group_for_reduces t > 0
         && (match opt with Nolocals | Padto _ -> false | _ -> true)))
  then begin
    let fs = full_shape t in
    let upcast_local_sz = prod (List.map (fun a ->
      const_int_or 1 (List.nth fs a))
      (axes_of t [Axis_kind.Upcast; Axis_kind.Warp;
                  Axis_kind.Local; Axis_kind.Group_reduce])) in
    let red = match red_opt with Some r -> r | None -> assert false in
    let red_dt = K.dtype red in
    let smem_sz = amt * upcast_local_sz * Dtype.itemsize red_dt in
    check (smem_sz <= Renderer.shared_max t.ren)
      (strf "exceeds shared memory: needs %d, max %d"
         smem_sz (Renderer.shared_max t.ren))
  end

(* Check that a GROUP_REDUCE is not inside another reduce. *)
let check_no_nested_group t r red_opt =
  if red_opt <> None then begin
    (* Find the REDUCE whose range sources contain r *)
    let reduce_node = List.find_opt (fun u ->
      match K.view u with
      | Reduce { ranges; _ } ->
          let range_nodes = K.find_nodes K.is_range (K.sink ranges) in
          List.memq r range_nodes
      | _ -> false) (K.toposort t.ast) in
    match reduce_node with
    | Some red_node ->
        (* Check enclosing (live) ranges, not the reduce's own range inputs *)
        let live = K.live_ranges red_node in
        check (not (List.exists (fun u ->
          let k = K.range_kind u in
          k = Axis_kind.Reduce || k = Axis_kind.Unroll
          || k = Axis_kind.Group_reduce) live))
          "cannot have a GROUP_REDUCE inside another reduce"
    | None -> ()
  end

(* Per-opt validation for shift_to opts. *)
let validate_shift_opt t opt r amt rng_kind =
  match opt with
  | K.Opt.Unroll _ ->
      check (amt <= 32) "don't unroll more than 32";
      check (rng_kind = Axis_kind.Group_reduce || rng_kind = Axis_kind.Reduce)
        "unroll is for GROUP_REDUCE/REDUCE"
  | Upcast _ ->
      check (Renderer.device t.ren = "DSP" || amt <= 16)
        "don't upcast more than 16";
      check (rng_kind = Axis_kind.Global || rng_kind = Axis_kind.Local
             || rng_kind = Axis_kind.Loop)
        "upcast is for GLOBAL/LOCAL/LOOP"
  | Local _ ->
      check (not t.dont_use_locals) "can't use locals";
      check (rng_kind = Axis_kind.Global || rng_kind = Axis_kind.Loop)
        "local is for globals"
  | Thread _ ->
      check (Renderer.has_threads t.ren) "target does not support threads";
      (match Renderer.global_max t.ren with
       | Some (gm :: _) -> check (amt <= gm) "too many threads" | _ -> ());
      check (List.for_all (fun at -> at <> Axis_kind.Thread)
               (axis_types t)) "already threaded";
      check (List.memq r (globalizable_rngs t))
        "can't apply thread to this dim"
  | Group _ | Grouptop _ ->
      check (List.for_all (fun o ->
        match o with K.Opt.Tc _ -> false | _ -> true) t.applied_opts)
        "no grouping with tensor cores";
      check (not t.dont_use_locals) "can't use locals";
      check (rng_kind = Axis_kind.Reduce) "group is for reduce"
  | _ -> ()

(* Pad a range to a multiple of [amount]. *)
let apply_padto t r amount red_opt =
  check (K.const_arg (K.range_size r) <> None) "only pad const axes";
  let rng_kind = K.range_kind r in
  check (rng_kind <> Axis_kind.Upcast && rng_kind <> Axis_kind.Unroll)
    "cannot pad upcasted";
  check (rng_kind <> Axis_kind.Thread) "cannot pad thread";
  (match red_opt with
   | Some red when rng_kind = Axis_kind.Group_reduce
                 || rng_kind = Axis_kind.Reduce ->
       let red_op = match K.view red with
         | Reduce { op; _ } -> op | _ -> assert false in
       check (red_op = `Add)
         (strf "cannot pad %s" (K.view_op_name (K.view red)));
       let has_unsafe = List.exists (fun u ->
         match K.view u with
         | Unary { op = (`Recip | `Log2 | `Exp2); _ }
         | Binary { op = (`Idiv | `Pow); _ } -> true
         | _ -> false) (K.toposort red) in
       check (not has_unsafe)
         (strf "cannot pad %s" (K.view_op_name (K.view red)))
   | _ -> ());
  let old_size = range_int_size r in
  let new_sz = (old_size + amount - 1) / amount * amount in
  check (old_size > new_sz / 4) "pad adds more than quadruple the work";
  let replaced_rng = K.range ~size:(K.const_int new_sz)
    ~axis:(K.range_axis r) ~sub:(K.range_sub r)
    ~kind:(K.range_kind r) ~dtype:(Dtype.val_of (K.dtype r)) () in
  let valid = K.binary ~op:`Cmplt
    ~lhs:replaced_rng ~rhs:(K.const_int old_size) in
  let subs = [(r, replaced_rng)] in
  let subs = List.fold_left (fun acc b ->
    match K.view b with
    | Index { ptr; idxs; gate; _ } when K.in_backward_slice r b ->
        let combined_valid = match gate with
          | Some g -> K.binary ~op:`And ~lhs:valid ~rhs:g
          | None -> valid in
        let guarded_idx = K.index ~ptr ~idxs ~gate:combined_valid () in
        let where = K.ternary ~op:`Where
          ~a:combined_valid ~b:guarded_idx ~c:(K.invalid_index ()) in
        (b, where) :: acc
    | _ -> acc) subs (bufs t) in
  t.ast <- K.substitute subs t.ast; refresh t

(* Swap two global ranges' axis numbers. *)
let apply_swap t r with_axis =
  let altrng = nth_or_error (rngs t) with_axis "invalid swap axis" in
  check (K.range_kind r = Axis_kind.Global
         && K.range_kind altrng = Axis_kind.Global)
    "swap only for globals";
  let r' = K.with_tag "1" (K.range
    ~size:(K.range_size r) ~sub:(K.range_sub r)
    ~axis:(K.range_axis altrng) ~kind:(K.range_kind r)
    ~dtype:(Dtype.val_of (K.dtype r)) ()) in
  let alt' = K.with_tag "1" (K.range
    ~size:(K.range_size altrng) ~sub:(K.range_sub altrng)
    ~axis:(K.range_axis r) ~kind:(K.range_kind altrng)
    ~dtype:(Dtype.val_of (K.dtype altrng)) ()) in
  t.ast <- K.substitute [(r, r'); (altrng, alt')] t.ast;
  t.ast <- K.graph_rewrite (fun node ->
    match K.tag node with Some _ -> Some (K.replace node ()) | None -> None)
    t.ast;
  refresh t

(* Mutual recursion: apply_opt <-> apply_tc_opt <-> pad_tc_axes *)

(* Pad each TC axis to a multiple of tc.dims[i]. Returns false on failure. *)
let rec pad_tc_axes t axes (tc : Tc.t) tc_opt =
  let pad_ok = ref true in
  (try
    for i = 0 to 2 do
      let a = axes.(i) in
      let idx = match List.find_index (fun r -> r == a) (rngs t) with
        | Some j -> j | None -> raise (Opt_error "range not found") in
      let dim = let n, m, k = tc.dims in [|n; m; k|].(i) in
      if range_int_size a mod dim <> 0 then begin
        if tc_opt < 2 then raise (Opt_error "tc padding requires opt_level >= 2");
        ignore (apply_opt ~append_opt:false t
          (K.Opt.Padto { axis = idx; amount = dim }));
        axes.(i) <- List.nth (rngs t) idx
      end
    done
  with Opt_error _ -> pad_ok := false);
  !pad_ok

(* Apply tensor core optimisation.  Returns Some axes on success, None
   if no matching TC was found. *)
and apply_tc_opt t use_tc axis tc_select tc_opt =
  let red = match List.find_opt (fun x ->
    match K.view x with Reduce _ -> true | _ -> false) (K.toposort t.ast) with
    | Some r -> r | None -> raise (Opt_error "no reduce ops for TensorCore") in
  let red_op, red_src = match K.view red with
    | Reduce { op; src; _ } -> op, src | _ -> assert false in
  if use_tc = 0 || red_op <> `Add then None
  else
    let mul = match K.view red_src with Cast { src; _ } -> src | _ -> red_src in
    match K.view mul with
    | Binary { op = `Mul; lhs = in0; rhs = in1; _ } ->
        let tcs =
          if tc_select = -1 then Renderer.tensor_cores t.ren
          else match List.nth_opt (Renderer.tensor_cores t.ren) tc_select with
            | Some tc -> [tc] | None -> raise (Opt_error "invalid tensor core choice") in
        let in0_dt = Dtype.val_of (Dtype.scalarize (K.dtype in0)) in
        let in1_dt = Dtype.val_of (Dtype.scalarize (K.dtype in1)) in
        let red_dt = Dtype.val_of (Dtype.scalarize (K.dtype red)) in
        let in0_ranges = K.find_nodes K.is_range in0 in
        let in1_ranges = K.find_nodes K.is_range in1 in
        let red_ranges = match K.view red with
          Reduce { ranges; _ } -> ranges | _ -> [] in
        let sort_desc = List.sort (fun a b ->
          compare (K.range_axis b) (K.range_axis a)) in
        let try_tc (tc : Tc.t) =
          if (Renderer.device t.ren = "CUDA" || Renderer.device t.ren = "NV")
             && tc.dtype_in = Dtype.Float32 && not allow_tf32 then None
          else if Dtype.Val.scalar in0_dt <> tc.dtype_in
               || Dtype.Val.scalar in1_dt <> tc.dtype_in
               || Dtype.Val.scalar red_dt <> tc.dtype_out then None
          else
            let in0_r = sort_desc (List.filter (fun u ->
              not (List.memq u in1_ranges)) in0_ranges) in
            let in1_r = sort_desc (List.filter (fun u ->
              not (List.memq u in0_ranges)) in1_ranges) in
            let red_r = sort_desc red_ranges in
            if in0_r = [] || in1_r = [] || red_r = [] then None
            else
              let choices = List.concat_map (fun a ->
                List.concat_map (fun b ->
                  List.map (fun c -> [a; b; c]) red_r) in0_r) in1_r in
              if axis >= List.length choices then None
              else begin
                let axes = Array.of_list (List.nth choices axis) in
                t.ast <- K.substitute [(red, K.with_tag "TC" red)] t.ast;
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
        List.find_map try_tc tcs
    | _ -> None

and apply_opt ?(append_opt = true) t opt =
  let ret = match opt with
  | K.Opt.Nolocals ->
      check (List.for_all (fun at ->
        at <> Axis_kind.Warp && at <> Axis_kind.Local
        && at <> Axis_kind.Group_reduce) (axis_types t))
        "no locals can't have locals";
      if append_opt then t.applied_opts <- t.applied_opts @ [opt];
      t.dont_use_locals <- true;
      None

  | Tc { axis; tc_select; tc_opt; use_tc } ->
      check (t.applied_opts = []) "tensor core opts must be first";
      check (tc_select >= -1
             && tc_select < List.length (Renderer.tensor_cores t.ren))
        "invalid tc_select";
      check (tc_opt >= 0 && tc_opt <= 2) "invalid tc_opt";
      check (use_tc > 0 && use_tc <= 2) "invalid use_tc";
      let axes = apply_tc_opt t use_tc axis tc_select tc_opt in
      check (Option.is_some axes) "no tensor core available";
      Option.bind axes (fun l -> match l with
        | a :: b :: _ -> Some (a, b) | _ -> None)

  | Padto { axis = _; amount } ->
      let ra = real_axis t opt (K.Opt.axis opt) in
      let r = List.nth (rngs t) ra in
      apply_padto t r amount (reduceop t);
      None

  | Swap { axis = _; with_axis } ->
      let ra = real_axis t opt (K.Opt.axis opt) in
      let r = List.nth (rngs t) ra in
      apply_swap t r with_axis;
      None

  | _ ->
      let ra = real_axis t opt (K.Opt.axis opt) in
      let r = List.nth (rngs t) ra in
      let red_opt = reduceop t in
      (match opt with
       | Local _ | Group _ | Grouptop _ ->
           check (Renderer.has_local t.ren) "locals needed for opt"
       | _ -> ());
      let new_kind = match opt with
        | Local _ -> Axis_kind.Local | Upcast _ -> Axis_kind.Upcast
        | Unroll _ -> Axis_kind.Unroll
        | Group _ | Grouptop _ -> Axis_kind.Group_reduce
        | Thread _ -> Axis_kind.Thread
        | _ -> assert false in
      let amt = match K.Opt.amount opt with
        | Some 0 -> range_int_size r | Some a -> a
        | None -> range_int_size r in
      check_shared_memory t opt amt red_opt;
      (match opt with Group _ | Grouptop _ ->
        check_no_nested_group t r red_opt | _ -> ());
      validate_shift_opt t opt r amt (K.range_kind r);
      let top = match opt with Grouptop _ | Thread _ -> true | _ -> false in
      Some (shift_to ~top t r amt new_kind)
  in
  if append_opt then t.applied_opts <- t.applied_opts @ [opt];
  ret

(* Extract sorted Param nodes from an AST.  Returns raw Param nodes;
   the caller (Pipeline) constructs device buffers. *)
let bufs_from_ast ast =
  List.sort (fun a b ->
    match K.view a, K.view b with
    | Param { idx = ia; _ }, Param { idx = ib; _ } -> compare ia ib
    | _ -> 0)
    (List.filter (fun n -> match K.view n with Param _ -> true | _ -> false)
       (K.backward_slice ast))

(* Top-level optimization dispatch.  Strategy closures are passed by the
   caller (Pipeline) to break circular module dependencies. *)
let apply_opts ?beam_search ?hand_coded_optimizations ast ren =
  if K.tag ast <> None then ast
  else
    let ki = match K.view ast with
      | Sink { kernel_info = Some ki; _ } -> Some ki | _ -> None in
    let k = create ast ren in
    convert_loop_to_global k;
    let optimize k =
      match beam_search with
      | Some bs -> bs k
      | None ->
          match hand_coded_optimizations with
          | Some f when k.applied_opts = []
            && not (List.exists (fun n ->
                 match K.view n with Bufferize _ -> true | _ -> false)
                 (K.backward_slice ast)) -> f k
          | _ -> k
    in
    let k = match ki with
      | Some { opts_to_apply = Some opts; _ } ->
          List.iter (fun opt -> ignore (apply_opt k opt)) opts; k
      | _ -> optimize k
    in
    let name_override = match ki with
      | Some ki when ki.name <> "" && ki.name <> "test" -> Some ki.name
      | _ -> None in
    get_optimized_ast ?name_override k
