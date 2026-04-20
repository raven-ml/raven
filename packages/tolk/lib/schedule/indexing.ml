(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Core rangeify algorithm.

   Converts the high-level tensor graph (with movement ops, REDUCE_AXIS, etc.)
   into an indexed representation with explicit RANGE loops, BUFFERIZE nodes,
   and INDEX operations.

   The algorithm has three phases:

   1. Build the realize map: decide which nodes need their own buffer
      (realization boundary).  Realized nodes get fresh ranges and produce
      BUFFERIZE + INDEX pairs in the final graph.

   2. Backward range propagation (run_rangeify): walk the graph in reverse
      toposort.  Each node either inherits ranges from its single consumer,
      merges ranges from multiple consumers, or gets fresh ranges when
      realized.  Movement ops transform ranges (permute, reshape, etc.)
      instead of existing as nodes in the output.

   3. Apply rangeify (pm_apply_rangeify): rewrite the graph bottom-up,
      replacing REDUCE_AXIS with REDUCE, PAD with WHERE, inserting
      BUFFERIZE/INDEX/END nodes, and removing movement ops. *)

open Tolk_ir
module T = Tensor
module D = Dtype
module C = Const
module Ak = Axis_kind

(* Ops that never need realization — they produce contiguous output by
   definition, so their consumers can always index directly into them.
   tinygrad also includes DEFINE_REG and LOAD, which don't exist in the
   tensor-level IR (they are kernel-only ops). *)
let is_always_contiguous = function
  | T.Contiguous _ | T.After _ | T.Copy _ | T.Buffer _ | T.Buffer_view _
  | T.Const _ | T.Bind _ | T.Device _ | T.Mselect _ | T.Mstack _ | T.Param _
  | T.Define_local _ | T.Call _ ->
      true
  | _ -> false

(* Helpers *)

let idx n = T.const (C.int D.Val.index n) D.index
let btrue = T.const (C.bool true) D.bool
let bfalse = T.const (C.bool false) D.bool

let select_axes axes xs = List.filteri (fun i _ -> List.mem i axes) xs

let movement_src = function
  | T.Reshape { src; _ } | T.Expand { src; _ } | T.Pad { src; _ }
  | T.Shrink { src; _ } | T.Permute { src; _ } | T.Flip { src; _ } ->
      Some src
  | _ -> None

let is_movement_op v = Option.is_some (movement_src v)

(* Boolean fold: MUL for conjunction, ADD for disjunction — matching
   tinygrad's .prod() and .sum() on bool UOps. *)
let bool_reduce op identity vs =
  List.fold_left (fun acc v -> T.binary ~op ~lhs:acc ~rhs:v) identity vs

let prod_valid vs = bool_reduce `Mul btrue vs
let sum_valid vs = bool_reduce `Add bfalse vs

(* r >= s  encoded as  NOT(r < s),  matching tinygrad's
   (self < x).logical_not() → CMPNE(CMPLT(r, s), true). *)
let ge r s =
  T.binary ~op:`Cmpne
    ~lhs:(T.binary ~op:`Cmplt ~lhs:r ~rhs:(idx s)) ~rhs:btrue

(* Indexing context *)

type realize_state =
  | Marked        (* pending realization — set during realize map construction *)
  | Realized of int list  (* resolved — records which axes were realized *)

type indexing_context = {
  realize_map : (int, realize_state) Hashtbl.t;
  range_map : (int, T.t list * T.t list) Hashtbl.t;
  mutable range_idx : int;
}

let create_context () = {
  realize_map = Hashtbl.create 256;
  range_map = Hashtbl.create 256;
  range_idx = 0;
}

(* Size-1 dimensions collapse to constant 0.  [size] is concrete — tinygrad
   accepts [sint] (symbolic or int) but we only handle static shapes here. *)
let new_range ctx size ?(kind = Ak.Loop) () =
  if size = 1 then idx 0
  else begin
    let axis = ctx.range_idx in
    ctx.range_idx <- ctx.range_idx + 1;
    T.range ~size:(idx size) ~axis ~kind ()
  end

(* Context accessors — keyed by T.tag (unique per hash-consed node). *)

let realize_get ctx n = Hashtbl.find_opt ctx.realize_map (T.tag n)
let realize_set ctx n v = Hashtbl.replace ctx.realize_map (T.tag n) v
let realize_del ctx n = Hashtbl.remove ctx.realize_map (T.tag n)
let realize_mem ctx n = Hashtbl.mem ctx.realize_map (T.tag n)

let range_get ctx n = Hashtbl.find_opt ctx.range_map (T.tag n)
let range_set ctx n v = Hashtbl.replace ctx.range_map (T.tag n) v

(* Generate realize map *)

let has_store_dep deps =
  List.exists (fun d -> match T.view d with T.Store _ -> true | _ -> false) deps

(* Does [n] or any node in its backward slice match one of the
   non-injective view ops?  (RESHAPE and EXPAND are excluded —
   tinygrad only checks SHRINK, PERMUTE, FLIP, PAD here.) *)
let has_view_op_in_slice n =
  List.exists (fun x ->
    match T.view x with
    | T.Shrink _ | T.Permute _ | T.Flip _ | T.Pad _ -> true
    | _ -> false)
    (n :: T.backward_slice n)

let mark_non_contiguous_src ctx s =
  if not (is_always_contiguous (T.view (T.base s))) then
    realize_set ctx s Marked

(* Mirrors tinygrad's pm_generate_realize_map PatternMatcher.
   All four blocks fire independently per node — a PatternMatcher rule
   that returns None continues to the next rule rather than short-
   circuiting. *)
let generate_realize_map ctx root =
  let nodes = T.toposort root in
  List.iter (fun n ->
    let v = T.view n in
    (* Rule 1: always realize COPY and CONTIGUOUS *)
    (match v with
     | T.Copy _ | T.Contiguous _ -> realize_set ctx n Marked
     | _ -> ());
    (* Rule 2: realize AFTER that has a STORE dep *)
    (match v with
     | T.After { deps; _ } when has_store_dep deps -> realize_set ctx n Marked
     | _ -> ());
    (* Rule 3: realize non-contiguous sources of COPY/MSELECT/MSTACK *)
    (match v with
     | T.Copy { src; _ } | T.Mselect { src; _ } ->
         mark_non_contiguous_src ctx src
     | T.Mstack { srcs; _ } ->
         List.iter (mark_non_contiguous_src ctx) srcs
     | _ -> ());
    (* Rule 4: conditionally unrealize or re-realize the value in a
       single-dep Store+After.  Only fires when deps = [Store]. *)
    (match v with
     | T.After { deps = [d]; _ } -> begin
         match T.view d with
         | T.Store { dst; value } ->
             (* Unrealize COPY/BUFFER_VIEW when the target buffer IS the
                output and no view ops distort the destination. *)
             (match T.view value with
              | T.Copy _ | T.Buffer_view _
                when realize_mem ctx value
                     && not (has_view_op_in_slice dst) ->
                  realize_del ctx value
              | _ -> ());
             (* WAR hazard: dest's base in value's backward slice means
                the write aliases a read — force a temporary. *)
             let base = T.base dst in
             if List.exists (fun x -> x == base)
                  (value :: T.backward_slice value) then
               realize_set ctx value Marked
         | _ -> ()
       end
     | _ -> ()))
    nodes

(* Tensor ↔ Kernel conversion for symbolic simplification.
   Only index arithmetic nodes are expected — anything else is a bug. *)

module K = Kernel

let rec tensor_to_kernel n =
  match T.view n with
  | T.Const { value; _ } -> K.const value
  | T.Range { size; axis; sub; kind; dtype } ->
      K.range ~size:(tensor_to_kernel size) ~axis ~sub ~kind
        ~dtype:(D.val_of dtype) ()
  | T.Binary { op; lhs; rhs; _ } ->
      K.binary ~op ~lhs:(tensor_to_kernel lhs) ~rhs:(tensor_to_kernel rhs)
  | T.Unary { op; src; _ } ->
      K.unary ~op ~src:(tensor_to_kernel src)
  | T.Ternary { op; a; b; c; _ } ->
      K.ternary ~op ~a:(tensor_to_kernel a)
        ~b:(tensor_to_kernel b) ~c:(tensor_to_kernel c)
  | T.Invalid_index { dtype } ->
      K.const (C.int (D.val_of (D.scalarize dtype)) 0)
  | v -> failwith (Format.asprintf "tensor_to_kernel: unexpected %a" T.pp_view v)

let rec kernel_to_tensor k =
  match K.view k with
  | K.Const { value; dtype } -> T.const value (D.Val dtype)
  | K.Range { size; axis; sub; kind; dtype } ->
      T.range ~size:(kernel_to_tensor size) ~axis ~sub ~kind
        ~dtype:(D.Val dtype) ()
  | K.Binary { op; lhs; rhs; _ } ->
      T.binary ~op ~lhs:(kernel_to_tensor lhs) ~rhs:(kernel_to_tensor rhs)
  | K.Unary { op; src; _ } ->
      T.unary ~op ~src:(kernel_to_tensor src)
  | K.Ternary { op; a; b; c; _ } ->
      T.ternary ~op ~a:(kernel_to_tensor a)
        ~b:(kernel_to_tensor b) ~c:(kernel_to_tensor c)
  | _ -> failwith (Format.asprintf "kernel_to_tensor: unexpected %a" K.pp_view k)

(* Round-trip through Kernel IR to apply symbolic simplification. *)
let simplify_tensor_expr expr =
  let k = tensor_to_kernel expr in
  let k = K.graph_rewrite (K.first_match [Symbolic.sym]) k in
  kernel_to_tensor k

(* Movement ops — reshape *)

let argsort order =
  let indexed = List.mapi (fun i v -> (v, i)) order in
  List.map snd (List.sort (fun (a, _) (b, _) -> compare a b) indexed)

(* Reshape: linearize output dims into a scalar index, decompose into input
   dims via mod/div, then simplify the resulting expressions.

   A placeholder substitution trick keeps the simplifier from confusing actual
   range identities with the arithmetic it needs to reduce. *)
let apply_reshape in_shape out_shape rngs =
  let rngs = List.map simplify_tensor_expr rngs in
  (* Collect all Range nodes and create Placeholder stand-ins *)
  let all_ranges = T.ranges (T.sink rngs) in
  let sub_fwd = List.mapi (fun i r ->
    let size = match T.view r with
      | T.Range { size; _ } -> size | _ -> idx 1 in
    (r, T.range ~size ~axis:i ~kind:Ak.Placeholder ())) all_ranges in
  let sub_rev = List.map (fun (k, v) -> (v, k)) sub_fwd in
  let rngs = List.map (T.substitute sub_fwd) rngs in
  (* Linearize: weighted positional sum of output ranges *)
  let _, terms = List.fold_right (fun (s, r) (stride, ts) ->
    let t = if stride = 1 then r
      else T.binary ~op:`Mul ~lhs:(idx stride) ~rhs:r in
    (stride * s, t :: ts)) (List.combine out_shape rngs) (1, []) in
  let combined = List.fold_left (fun a t ->
    T.binary ~op:`Add ~lhs:a ~rhs:t) (idx 0) terms in
  (* Decompose: peel off input dimensions right-to-left.
     The ref + rev_map/rev processes in_shape in reverse while the ref
     accumulates the running quotient; rev_map reverses the result. *)
  let combined = ref combined in
  let axes = List.rev_map (fun s ->
    let r = T.binary ~op:`Mod ~lhs:!combined ~rhs:(idx s) in
    combined := T.binary ~op:`Idiv ~lhs:!combined ~rhs:(idx s);
    r) (List.rev in_shape) in
  (* Simplify, then restore actual ranges *)
  List.map (fun r ->
    T.substitute sub_rev (simplify_tensor_expr r)) axes

(* Transform ranges through a movement op.  Each case defines how output
   indices map to input indices — this is the inverse of the movement. *)
let apply_movement_op ~shapes v rngs =
  match v with
  | T.Shrink _ -> begin
      match T.extract_marg_pairs v with
      | Some pairs ->
          List.map2 (fun r (ss, _) ->
            if ss = 0 then r
            else T.binary ~op:`Add ~lhs:r ~rhs:(idx ss))
            rngs pairs
      | None -> rngs
    end
  | T.Permute { order; _ } ->
      List.map (fun p -> List.nth rngs p) (argsort order)
  | T.Flip { src; dims; _ } -> begin
      match shapes src with
      | Some in_shape ->
          List.map2 (fun r (f, s) ->
            if not f then r
            else T.binary ~op:`Sub ~lhs:(idx (s - 1)) ~rhs:r)
            rngs (List.combine dims in_shape)
      | None -> rngs
    end
  | T.Expand { src; shape; _ } -> begin
      match shapes src, T.extract_int_shape shape with
      | Some in_shape, Some out_shape ->
          List.map2 (fun r (in_s, out_s) ->
            if in_s = out_s then r else idx 0)
            rngs (List.combine in_shape out_shape)
      | _ -> rngs
    end
  | T.Pad { src; _ } -> begin
      match shapes src, T.extract_marg_pairs v with
      | Some in_shape, Some pairs ->
          (* The where(r-s, invalid) is intentionally outside the
             graph_rewrite so that convert_pad_to_where wraps the pad
             with only the newly added validity condition *)
          List.map2 (fun (r, sh) (s, e) ->
            if s = 0 && e = 0 then r
            else
              let valid = simplify_tensor_expr
                (T.binary ~op:`And ~lhs:(ge r s)
                   ~rhs:(T.binary ~op:`Cmplt ~lhs:r
                            ~rhs:(idx (sh + s)))) in
              T.ternary ~op:`Where ~a:valid
                ~b:(T.binary ~op:`Sub ~lhs:r ~rhs:(idx s))
                ~c:(T.invalid_index ~dtype:D.index))
            (List.combine rngs in_shape) pairs
      | _ -> rngs
    end
  | T.Reshape { src; shape; _ } -> begin
      match shapes src, T.extract_int_shape shape with
      | Some in_shape, Some out_shape ->
          apply_reshape in_shape out_shape rngs
      | _ -> rngs
    end
  | _ -> assert false

(* Apply rangeify — graph rewrite rules *)

(* Extract the index value from a possibly-gated range expression.
   where(valid, index, invalid) → index;  anything else → itself. *)
let get_idx r =
  match T.view r with
  | T.Ternary { op = `Where; b = value; c = else_; _ } ->
      (match T.view else_ with T.Invalid_index _ -> value | _ -> r)
  | _ -> r

(* Extract the validity condition from a possibly-gated range expression.
   where(valid, _, invalid) → valid;  invalid → false;  else → true. *)
let get_valid r =
  match T.view r with
  | T.Ternary { op = `Where; a = valid; c = else_; _ } ->
      (match T.view else_ with T.Invalid_index _ -> valid | _ -> btrue)
  | T.Invalid_index _ -> bfalse
  | _ -> btrue

(* Direct buffer sources: can be indexed without realization.
   Matches PARAM, BUFFER_VIEW, MSTACK, MSELECT, and AFTER nodes whose
   deps don't include STORE or END (plain scheduling barriers). *)
let is_direct_buffer = function
  | T.Param _ | T.Buffer_view _ | T.Mstack _ | T.Mselect _ -> true
  | T.After { deps; _ } ->
      not (List.exists (fun d ->
        match T.view d with T.Store _ | T.End _ -> true | _ -> false) deps)
  | _ -> false

let map_device = function
  | Some (T.Single d) -> Some (K.Device_single d)
  | Some (T.Multi ds) -> Some (K.Device_multi ds)
  | None -> None

(* REDUCE_AXIS → REDUCE with explicit range children.
   Selects the input ranges at the reduce axes. *)
let convert_reduce_axis ctx n =
  match T.view n with
  | T.Reduce_axis { src; op; axes; dtype } -> begin
      match range_get ctx n with
      | Some ((in_rngs, _) as entry) ->
          let ranges = select_axes axes in_rngs in
          let ret = T.reduce ~src ~ranges ~op ~dtype in
          range_set ctx ret entry;
          Some ret
      | None -> None
    end
  | _ -> None

(* PAD → WHERE(valid, src, 0).
   Collects validity conditions from each input range and MULs them. *)
let convert_pad_to_where ctx n =
  match range_get ctx n with
  | None -> None
  | Some ((in_rngs, _) as entry) ->
      let valid = prod_valid (List.map get_valid in_rngs) in
      let src = match T.view n with T.Pad { src; _ } -> src | _ -> assert false in
      let dtype = match T.dtype n with Some d -> d | None -> D.float32 in
      let ret = T.ternary ~op:`Where ~a:valid ~b:src
        ~c:(T.const (C.zero (D.val_of dtype)) dtype) in
      range_set ctx ret entry;
      Some ret

(* Strip movement ops — their effect is already captured in the range_map.
   Also remove when the source is an INDEX (already lowered). *)
let remove_movement_op ctx n =
  match movement_src (T.view n) with
  | Some src ->
      if Option.is_some (range_get ctx n) then Some src
      else (match T.view src with T.Index _ -> Some src | _ -> None)
  | None -> None

(* For each child of [n], insert BUFFERIZE/INDEX/END as needed:
   - Direct buffer sources get an INDEX with the consumer's input ranges.
   - Realized non-STORE sources get BUFFERIZE + INDEX.
   - Realized STORE sources get END (closing ranges).
   Returns None when no children changed. *)
let create_bufferize_and_index ctx ~devices n =
  match T.view n with
  | T.Bufferize _ | T.Index _ -> None
  | _ ->
      let parent_is_copy = match T.view n with T.Copy _ -> true | _ -> false in
      let parent_rngs = range_get ctx n in
      let children = T.children n in
      let changed = ref false in
      let new_children = List.map (fun s ->
        let sv = T.view s in
        if is_direct_buffer sv then
          match parent_rngs with
          | Some (in_rngs, _) ->
              changed := true;
              (* Strip pointer → value dtype, matching tinygrad's .dtype.base *)
              let dtype = match T.dtype s with
                | Some d -> D.Val (D.val_of d) | None -> D.index in
              T.index ~ptr:s ~idxs:in_rngs ~dtype ()
          | None -> s
        else match realize_get ctx s with
        | Some (Realized realized_axes) ->
            changed := true;
            let out_rngs = match range_get ctx s with
              | Some (_, out) -> out | None -> [] in
            let closed = select_axes realized_axes out_rngs in
            (match sv with
             | T.Store _ ->
                 let ranges = List.filter (fun r ->
                   match T.view r with T.Range _ -> true | _ -> false) closed in
                 realize_del ctx s;
                 T.end_ ~value:s ~ranges
             | _ ->
                 let removable = not parent_is_copy
                   && not (is_always_contiguous sv) in
                 let is_local =
                   List.length out_rngs <> List.length realized_axes in
                 let addrspace = if is_local then D.Local else D.Global in
                 let device = map_device (devices s) in
                 let opts : K.bufferize_opts =
                   { device; addrspace; removable } in
                 let src_dtype = match T.dtype s with
                   | Some d -> d | None -> D.float32 in
                 let buf = T.bufferize ~src:s ~ranges:closed
                   ~dtype:src_dtype ~opts in
                 match parent_rngs with
                 | Some (in_rngs, _) ->
                     let idxs = select_axes realized_axes in_rngs in
                     let idx_dtype = D.Val (D.val_of src_dtype) in
                     T.index ~ptr:buf ~idxs ~dtype:idx_dtype ()
                 | None -> buf)
        | _ -> s) children in
      if !changed then Some (T.replace n ~children:new_children ())
      else None

(* Cascading rules matching tinygrad's pm_apply_rangeify PatternMatcher.
   Rules 1–2 are op-specific; rule 3 (All) matches everything; rule 4
   matches movement ops.  On None, each falls through to the next. *)
let apply_rangeify_pass ctx ~devices root =
  T.graph_rewrite ~name:"apply rangeify"
    ~on_rebuild:(fun ~old_n ~new_n ->
      if T.tag old_n <> T.tag new_n then begin
        (match realize_get ctx old_n with
        | Some v -> realize_set ctx new_n v | None -> ());
        (match range_get ctx old_n with
        | Some v -> range_set ctx new_n v | None -> ())
      end)
    (fun n ->
      let specific = match T.view n with
        | T.Reduce_axis _ -> convert_reduce_axis ctx n
        | T.Pad _ -> convert_pad_to_where ctx n
        | _ -> None in
      match specific with
      | Some _ -> specific
      | None ->
      match create_bufferize_and_index ctx ~devices n with
      | Some _ as r -> r
      | None -> remove_movement_op ctx n) root

(* Run rangeify — backward range propagation *)

let pcontig_var = Helpers.Context_var.int ~key:"PCONTIG" ~default:0

let all_same = function
  | [] -> true
  | x :: rest -> List.for_all (fun y -> y == x) rest

let is_elementwise_or_reduce = function
  | T.Unary _ | T.Binary _ | T.Ternary _ | T.Cast _ | T.Bitcast _
  | T.Reduce_axis _ -> true
  | _ -> false

(* Only called on nodes from T.ranges, which are always Range. *)
let range_axis r = match T.view r with
  | T.Range { axis; _ } -> axis | _ -> assert false

(* Transpose a list of equal-length lists: one per consumer →
   one per axis. *)
let transpose = function
  | [] -> []
  | first :: _ as lists ->
      List.mapi (fun i _ -> List.map (fun l -> List.nth l i) lists) first

(* Check whether ended ranges force additional axes to be realized.
   Clears ending ranges and returns the (possibly updated) out_rngs. *)
let check_ending_ranges ctx ~pcontig ~get_ending ~set_ending ~out_shape x out_rngs =
  if get_ending x = [] then out_rngs
  else begin
    let existing = match realize_get ctx x with
      | Some (Realized axes) -> axes | _ -> [] in
    let realize_axis = ref existing in
    List.iteri (fun i r ->
      if not (List.mem i !realize_axis) then
        if pcontig <= 1 ||
           List.exists (fun rr ->
             List.exists (fun e ->
               range_axis rr > range_axis e) (get_ending x))
             (T.ranges r)
        then realize_axis := !realize_axis @ [i]) out_rngs;
    set_ending x [];
    if !realize_axis <> [] then begin
      realize_set ctx x (Realized !realize_axis);
      List.mapi (fun i r ->
        if List.mem i !realize_axis then
          new_range ctx (List.nth out_shape i) ()
        else r) out_rngs
    end else out_rngs
  end

(* Main backward walk.  For each node (roots-to-leaves) we determine:
   - out_rngs: one range expression per output axis
   - rngs: one range expression per input axis (= out_rngs transformed
     by movement ops, with fresh Reduce ranges for REDUCE_AXIS)

   The pair (rngs, out_rngs) is stored in range_map for use by
   apply_rangeify_pass. *)
let run_rangeify root ~shapes =
  let ctx = create_context () in
  generate_realize_map ctx root;
  let consumers = T.consumer_map root in
  let toposort = T.toposort ~enter_calls:false root in
  let ending : (int, T.t list) Hashtbl.t = Hashtbl.create 256 in
  let get_ending x =
    Option.value ~default:[] (Hashtbl.find_opt ending (T.tag x)) in
  let set_ending x v = Hashtbl.replace ending (T.tag x) v in
  let pcontig = Helpers.Context_var.get pcontig_var in

  List.iter (fun x ->
    let v = T.view x in

    (* Skip non-rangeable nodes.
       Lunique is OCaml-specific (tinygrad only skips UNIQUE). *)
    let skip = match v with
      | T.Device _ | T.Unique _ | T.Lunique _
      | T.Call _ | T.Linear _
      | T.Mstack _ | T.Mselect _ -> true
      | T.After { deps; _ } -> not (has_store_dep deps)
      | _ ->
          match T.dtype x with
          | Some dt -> D.scalar dt = D.Index
          | None -> false in
    if skip then () else begin

    (* Propagate ending ranges from consumers *)
    set_ending x (List.concat_map get_ending (consumers x));

    let out_shape = Option.value ~default:[] (shapes x) in

    (* Input ranges of consumers that already have ranges *)
    let consumer_rngs = List.filter_map (fun c ->
      match range_get ctx c with
      | Some (in_rngs, _) -> Some in_rngs
      | None -> None) (consumers x) in

    (* --- Determine output ranges --- *)
    let out_rngs =
      if realize_mem ctx x then begin
        (* 1. Realized → fresh ranges, end all, mark all axes *)
        let out = List.map (fun s -> new_range ctx s ()) out_shape in
        set_ending x [];
        assert (realize_get ctx x = Some Marked);
        realize_set ctx x
          (Realized (List.init (List.length out_shape) Fun.id));
        Some out
      end
      else match List.length consumer_rngs with
      | 0 -> None  (* no consumer has ranges → skip *)
      | 1 -> Some (List.hd consumer_rngs)
      | _ ->
          (* 3. Multiple consumers → merge per-axis *)
          let n = List.length (List.hd consumer_rngs) in
          if not (List.for_all (fun l -> List.length l = n) consumer_rngs) then begin
            (* Consumer ranges disagree on rank → realize *)
            let n_out = List.length out_shape in
            let out = List.map (fun s -> new_range ctx s ()) out_shape in
            realize_set ctx x (Realized (List.init n_out Fun.id));
            Some out
          end else
          (* Truncate to min of consumer rank and output rank, matching
             tinygrad's zip truncation behavior. *)
          let per_axis_full = transpose consumer_rngs in
          let n_out = List.length out_shape in
          let per_axis = List.filteri (fun i _ -> i < n_out) per_axis_full in
          let rngs_valids = List.map (fun axis_rngs ->
            let local = List.map get_idx axis_rngs in
            let valids = List.map get_valid axis_rngs in
            (local, valids)) per_axis in
          let all_all_same =
            List.for_all (fun (lr, _) -> all_same lr) rngs_valids in
          let out = ref [] and realize_axes = ref [] in
          List.iteri (fun i (local_rngs, valids) ->
            if all_all_same || (pcontig > 0 && all_same local_rngs) then begin
              (* Ranges agree — merge validity with OR *)
              let merged = simplify_tensor_expr
                (T.ternary ~op:`Where ~a:(sum_valid valids)
                   ~b:(List.hd local_rngs)
                   ~c:(T.invalid_index ~dtype:D.index)) in
              out := merged :: !out
            end else begin
              (* Ranges disagree — fresh range, mark axis *)
              out := new_range ctx (List.nth out_shape i) () :: !out;
              realize_axes := i :: !realize_axes
            end) rngs_valids;
          let realize_axes = List.rev !realize_axes in
          if realize_axes <> [] then
            realize_set ctx x (Realized realize_axes);
          Some (List.rev !out)
    in

    match out_rngs with
    | None -> ()
    | Some out_rngs ->

    (* --- Ending range check --- *)
    (* Elementwise/reduce ops with ended ranges may need to realize
       additional axes to prevent stale range references. *)
    let out_rngs =
      if is_elementwise_or_reduce v then
        check_ending_ranges ctx ~pcontig ~get_ending ~set_ending
          ~out_shape x out_rngs
      else out_rngs
    in

    (* --- Compute input ranges --- *)
    let rngs = out_rngs in

    (* Movement ops transform output ranges into input ranges *)
    let rngs =
      if is_movement_op v then apply_movement_op ~shapes v rngs
      else rngs in

    (* EXPAND: track ending ranges for axes that changed
       (range was replaced by const 0 for broadcasted dims).
       tinygrad guards this with all(isinstance(y,int) or y.op is not
       Ops.RANGE for y in x.shape) to skip when EXPAND injects a range
       via a symbolic shape.  With static int shapes this is always true. *)
    (match v with
     | T.Expand _ ->
         let diff = List.filter_map (fun (ri, ro) ->
           if ri != ro then Some ro else None)
           (List.combine rngs out_rngs) in
         if diff <> [] then
           set_ending x (get_ending x @ T.ranges (T.sink diff))
     | _ -> ());

    (* REDUCE_AXIS: create Reduce-kind ranges for the reduction axes *)
    let rngs = match v with
      | T.Reduce_axis { axes; src; _ } -> begin
          match shapes src with
          | Some src_shape ->
              List.mapi (fun i (r, s) ->
                if List.mem i axes
                then new_range ctx s ~kind:Ak.Reduce ()
                else r) (List.combine rngs src_shape)
          | None -> rngs
        end
      | _ -> rngs in

    range_set ctx x (rngs, out_rngs)
    end)
    (List.rev toposort);

  ctx
