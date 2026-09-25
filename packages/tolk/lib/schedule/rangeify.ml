(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Port of tinygrad/schedule/rangeify.py to the tolk_uop IR.
   Transforms a tensor-level SINK into a kernel graph with CALL nodes
   wrapping kernel ASTs.

   [find_bufs] keeps precompiled call bodies opaque: Rune's staged scan embeds
   a compiled sub-linear in CUSTOM_FUNCTION "loop", outside the caller's
   kernel scope. This loop extension has no tinygrad counterpart. Its output
   buffers depend directly on the call effect. *)

open Tolk_uop
module U = Uop

let symbolic =
  Upat.Pattern_matcher.(Symbolic.symbolic ++ Symbolic.index_pushing)

let getv = Helpers.Context_var.get

(* Helpers *)

let prod l = List.fold_left ( * ) 1 l
let int_ n = U.const_int n

let src0 u = (U.src u).(0)
let src_list u = Array.to_list (U.src u)
let src_tail u =
  let s = U.src u in
  Array.to_list (Array.sub s 1 (Array.length s - 1))

let shape_node = function [ d ] -> int_ d | ds -> U.stack (List.map int_ ds)

let movement_src u =
  match U.op u with
  | Ops.Reshape | Ops.Expand | Ops.Pad | Ops.Shrink | Ops.Permute | Ops.Flip ->
      Some (src0 u)
  | _ -> None

let is_movement u = Option.is_some (movement_src u)

let shape_expr_of = U.shape_opt

let shape_of n =
  let rec concrete = function
    | [] -> Some []
    | dim :: rest ->
        Option.bind (U.const_int_value dim) (fun value ->
            Option.map (fun dims -> value :: dims) (concrete rest))
  in
  Option.bind (U.shape_opt n) concrete

let device_max_bufs =
  function "METAL" -> 31 | "WEBGPU" -> 8 | "CPU" -> 31 | _ -> 0

let base n = U.base n

let pm_early_rangeify n =
  match U.as_index n with
  | Some { ptr; _ } when
      (let op = U.op ptr in Ops.Group.is_elementwise op || op = Ops.Const) ->
      let tail = src_tail n in
      let push s = U.replace n ~src:(Array.of_list (s :: tail)) () in
      let new_children = List.map push (U.children ptr) in
      Some (U.replace ptr ~src:(Array.of_list new_children) ())
  | _ -> None

let shape_for_store u =
  try Some (U.shape u) with Invalid_argument _ -> None

let same_shape a b =
  List.length a = List.length b && List.for_all2 U.equal a b

let pm_add_ranges_to_store ctr n =
  match U.as_store n with
  | Some { dst; value; _ } ->
      (match shape_for_store dst, shape_for_store value with
       | Some [], _ | _, Some [] -> None
       | Some _, Some _ when U.max_numel dst = 1 && U.max_numel value = 1
         ->
           None
       | Some d_sh, Some v_sh ->
           if not (same_shape d_sh v_sh) then
             invalid_arg "Rangeify.add_ranges_to_store: bad store shape";
           let idxs =
             List.map (fun size ->
                 let axis = !ctr in
                 incr ctr;
                 U.range ~size ~axis ~kind:Axis_type.Weak ())
               d_sh
           in
           let mk s = U.index ~ptr:s ~idxs () in
           Some (U.end_
                   ~value:(U.store ~dst:(mk dst) ~value:(mk value) ())
                   ~ranges:idxs)
       | _ -> None)
  | _ -> None

let early_movement_pass sink =
  let ctr = ref 1000 in
  let rules =
    [
      Prepare.movement_ops;
      pm_early_rangeify;
      pm_add_ranges_to_store ctr;
    ]
  in
  U.graph_rewrite ~bottom_up:true ~name:"early movement ops"
    (U.first_match rules)
    sink

let is_invalid = U.is_invalid_const

(* Post-rangeify *)

let is_always_run op = op = Ops.Noop

let remove_noop_stage n =
  match U.as_stage n with
  | Some { src; ranges; _ } ->
      (match U.as_index src with
       | Some _ when Array.length (U.src src) - 1 = List.length ranges ->
           let idxs = src_tail src in
           if not (List.equal ( == ) idxs ranges) then None
           else
             let ptr = src0 src in
             (match shape_of n with
              | Some sh when sh <> [] ->
                  let zeros = shape_node (List.map (fun _ -> 0) sh) in
                  Some (U.shrink ~src:ptr ~offset:zeros ~size:(shape_node sh))
              | _ -> Some ptr)
       | _ -> None)
  | _ -> None

let cleanup_dead_axes n =
  match U.as_stage n with
  | Some { src; ranges; opts; _ } when opts.removable
                                       && (not (is_always_run (U.op src)))
                                       && U.op src <> Ops.After ->
      let sh = Option.value (shape_of n) ~default:[] in
      if List.length sh <> List.length ranges then None
      else if List.exists (fun r -> match U.as_range r with
          | Some v -> U.op v.size <> Ops.Const | None -> false) ranges
      then None
      else
        let src_ranges = U.ranges src in
        let hit = ref false and new_ranges = ref [] and new_sh = ref [] in
        List.iter2 (fun s rng ->
            let dead = match U.op rng with
              | Ops.Const -> true
              | Ops.Range -> not (List.exists (U.equal rng) src_ranges)
              | _ -> false
            in
            if dead then begin new_sh := 1 :: !new_sh; hit := true end
            else begin
              new_sh := s :: !new_sh;
              new_ranges := rng :: !new_ranges
            end) sh ranges;
        if not !hit then None
        else
          let new_ranges = List.rev !new_ranges in
          let new_sh = List.rev !new_sh in
          let b = U.stage ~src ~ranges:new_ranges ~opts in
          let r = U.reshape ~src:b ~shape:(shape_node new_sh) in
          Some (U.broadcast_to ~src:r ~shape:(shape_node sh))
  | _ -> None

let is_reduce_range r =
  match U.as_range r with
  | Some v -> v.kind = Axis_type.Reduce | None -> false

let range_size_expr r =
  match U.as_range r with Some v -> v.size | None -> int_ 1

let prod_expr = function
  | [] -> int_ 1
  | x :: xs ->
      let open U.O in
      List.fold_left ( * ) x xs

let unflatten_stage_index flat ranges =
  let sizes = List.map range_size_expr ranges in
  let open U.O in
  let rec loop acc flat = function
    | [] -> List.rev acc
    | [ _ ] -> List.rev (flat :: acc)
    | _ :: rest ->
        let stride = prod_expr rest in
        let axis = flat // stride in
        loop (axis :: acc) (flat mod stride) rest
  in
  loop [] flat sizes

(* The index sources that correspond one-for-one with a stage's ranges. A
   stage flattened to a single index expression is unflattened back; any other
   arity mismatch means the index was built against a different stage, which
   the rangeify pass must never produce. *)
let stage_index_sources buf idx =
  let srcs = src_tail idx in
  let ranges = buf.U.ranges in
  let removable_range r =
    match U.op r with Ops.Range | Ops.Const -> true | _ -> false
  in
  if not (List.for_all removable_range ranges) then None
  else if List.length ranges = List.length srcs then Some srcs
  else
    match srcs, ranges with
    | [ flat ], _ :: _ :: _ -> Some (unflatten_stage_index flat ranges)
    | _ ->
        invalid_arg
          (Printf.sprintf "Rangeify: index on wrong stage, %d ranges vs %d \
                           index sources" (List.length ranges)
             (List.length srcs))

let substitute_stage_ranges mappings src =
  let mappings = List.filter (fun (k, v) -> not (U.equal k v)) mappings in
  if mappings = [] then src
  else
    let keys = List.map fst mappings in
    let rewrite u =
      match
        List.find_map
          (fun (k, v) -> if U.equal u k then Some v else None)
          mappings
      with
      | Some _ as r -> r
      | None ->
          if List.exists
               (fun key -> List.exists (U.equal key) (U.ranges u))
               keys
          then None
          else raise U.Bottom_up_gate
    in
    U.graph_rewrite ~bottom_up:true rewrite src

let remove_stage src (buf : U.stage_view) idx =
  if is_always_run (U.op src) || not buf.opts.removable then None
  else
    let accessed = U.Ref_tbl.create 8 in
    let indexes = ref [] and reduces = ref [] in
    ignore
      (U.toposort src ~gate:(fun x ->
           match U.op x with
           | Ops.Stage ->
               (match U.as_stage x with
                | Some { opts = { addrspace = Dtype.Global; _ }; _ } ->
                    U.Ref_tbl.replace accessed x (); false
                | _ -> true)
           | Ops.Mstack -> U.Ref_tbl.replace accessed x (); false
           | Ops.After -> U.Ref_tbl.replace accessed (U.buf_uop x) (); false
           | Ops.Store -> false
           | Ops.Param -> U.Ref_tbl.replace accessed x (); true
           | Ops.Index -> indexes := x :: !indexes; true
           | Ops.Reduce -> reduces := x :: !reduces; true
           | _ -> true));
    let pc = getv Helpers.pcontig in
    if U.Ref_tbl.length accessed > 3 && pc <= 2 then None
    else
      let buffer_in_reduce =
        !reduces <> [] &&
        let found = ref false in
        let srcs = List.map src0 !reduces in
        ignore
          (U.toposort (U.sink srcs) ~gate:(fun x ->
               if !found then false
               else match U.op x with
                 | Ops.Param | Ops.Stage | Ops.After -> found := true; false
                 | _ -> true));
        !found
      in
      match stage_index_sources buf idx with
      | None -> None
      | Some buf_src ->
      if buffer_in_reduce then begin
        if pc <= 2 then None
        else
          let local_indexes =
            List.filter (fun x ->
                match U.as_index x with
                | Some { ptr; _ } ->
                    (match U.as_stage ptr with
                     | Some { opts = { addrspace = Dtype.Local; _ }; _ } -> true
                     | _ -> false)
                | _ -> false) !indexes
          in
          let exclude_ranges =
            List.concat_map (fun x -> U.ranges (U.sink (src_tail x)))
              local_indexes
          in
          let subs =
            List.filter_map (fun (k, v) ->
                if U.op k = Ops.Const then None else Some (k, v))
              (List.combine buf.ranges buf_src)
          in
          let is_pcontig, is_subs =
            List.partition (fun (k, v) ->
                List.exists (U.equal k) exclude_ranges
                || List.exists is_reduce_range (U.ranges v))
              subs
          in
          if is_subs = [] then None
          else
            let ret = substitute_stage_ranges is_subs src in
            if is_pcontig = [] then Some ret
            else
              let pc_rngs = List.map fst is_pcontig in
              let pc_idxs = List.map snd is_pcontig in
              let b =
                U.stage ~src:ret ~ranges:pc_rngs
                  ~opts:{ device = None; addrspace = Dtype.Local;
                          removable = true }
              in
              Some
                (U.replace idx ~src:(Array.of_list (b :: pc_idxs)) ())
      end else
        let mappings =
          List.filter_map (fun (k, v) ->
              if U.op k = Ops.Const then None
              else match U.arg v with
                | U.Arg.Value c when Const.view c = Const.Invalid -> None
                | _ -> Some (k, v))
            (List.combine buf.ranges buf_src)
        in
        Some (substitute_stage_ranges mappings src)

let remove_stage_index n =
  match U.as_index n with
  | Some { ptr; _ } -> (
      match U.as_stage ptr with
      | Some buf -> remove_stage buf.src buf n
      | None -> None)
  | None -> None

let is_buf_boundary u =
  match U.op u with
  | Ops.Stage | Ops.After | Ops.Param | Ops.Mselect | Ops.Mstack -> true
  | _ -> false

(* Buffer-boundary nodes reachable from [root], stopping at the first boundary
   on each path. A boundary is itself; any other node is the deduped union of
   its sources' sets. Memoised into [ctx.buf_cache] (persistent across matches,
   keyed by tag) so each subtree is walked once rather than re-toposorted per
   match. *)
let reachable_bufs (ctx : Indexing.indexing_context) root =
  let visitor u =
    if is_buf_boundary u then [ u ]
    else
      let srcs = U.src u in
      if Array.length srcs = 1 then
        Option.value (Hashtbl.find_opt ctx.buf_cache (U.tag srcs.(0)))
          ~default:[]
      else begin
        let seen = U.Ref_tbl.create 16 in
        let acc = ref [] in
        Array.iter
          (fun s ->
            match Hashtbl.find_opt ctx.buf_cache (U.tag s) with
            | Some lst ->
                List.iter
                  (fun b ->
                    if not (U.Ref_tbl.mem seen b) then begin
                      U.Ref_tbl.replace seen b ();
                      acc := b :: !acc
                    end)
                  lst
            | None -> ())
          srcs;
        List.rev !acc
      end
  in
  U.topovisit visitor ctx.buf_cache root

let limit_bufs (ctx : Indexing.indexing_context) n =
  match U.op n with
  | op when Ops.Group.is_binary op || Ops.Group.is_ternary op ->
      let dname = match U.device_of n with
        | Some (Single d) -> Some (List.hd (String.split_on_char ':' d))
        | Some (Multi ds) -> Some (List.hd (String.split_on_char ':' (List.hd ds)))
        | _ -> None
      in
      Option.bind dname (fun d ->
          let max_bufs = match getv Helpers.max_kernel_buffers with
            | 0 -> device_max_bufs d | n -> n
          in
          if max_bufs = 0 then None
          else
            let bufs = reachable_bufs ctx n in
            if List.length bufs <= max_bufs - 1 then None
            else
              let children = U.children n in
              let new_children =
                List.map (fun s ->
                    if Ops.Group.is_elementwise (U.op s)
                       && U.device_of s <> None then
                      let orig = U.ranges s in
                      let renum =
                        List.map (fun x -> match U.as_range x with
                            | Some v ->
                                let axis = ctx.range_idx in
                                ctx.range_idx <- ctx.range_idx + 1;
                                U.range ~size:v.size ~axis ~sub:v.sub
                                  ~kind:Axis_type.Weak
                                  ~dtype:(U.dtype x)
                                  ~parents:v.parents ()
                            | None -> x) orig
                      in
                      let subst = U.substitute (List.combine orig renum) s in
                      let opts : U.stage_opts =
                        { device = U.device_of s; addrspace = Dtype.Global;
                          removable = true }
                      in
                      let b =
                        U.stage ~src:subst ~ranges:renum ~opts
                      in
                      U.replace s ~op:Ops.Index
                        ~src:(Array.of_list (b :: orig)) ()
                    else s) children
              in
              if List.for_all2 ( == ) children new_children then None
              else Some (U.replace n ~src:(Array.of_list new_children) ()))
  | _ -> None

(* Add buffers *)

(* How many elements an index spans: a range's size, one element for an axis
   pinned to a constant, and otherwise the bound of the expression — a stage
   flattened to a single index no longer holds a RANGE, but it still spans its
   whole extent. *)
let range_int_size r =
  match U.as_range r with
  | Some v -> Option.value (U.const_int_value v.size) ~default:(Bound.to_int (Bound.succ (U.vmax r)))
  | None -> if U.op r = Ops.Const then 1 else Bound.to_int (Bound.succ (U.vmax r))

let flat_index_of_ranges ?dims ranges =
  let range_dims ranges =
    match dims with
    | Some dims when List.length dims = List.length ranges -> dims
    | _ -> List.map range_int_size ranges
  in
  match ranges with
  | [] -> int_ 0
  | [ r ] -> r
  | ranges ->
      let dims = Array.of_list (range_dims ranges) in
      let ranges = Array.of_list ranges in
      let n_axes = Array.length ranges in
      let acc = ref ranges.(n_axes - 1) in
      let stride = ref 1 in
      for i = n_axes - 2 downto 0 do
        stride := !stride * dims.(i + 1);
        let open U.O in
        let term =
          if !stride = 0 then int_ 0
          else if !stride = 1 then ranges.(i)
          else ranges.(i) * int_ !stride
        in
        acc := !acc + term
      done;
      !acc

let flatten_stage n =
  match U.as_stage n with
  | Some { src; ranges; opts; _ }
    (* Every stage flattens to exactly one index: multiple ranges collapse
       into a flat expression, and a rank-0 stage indexes its single element
       at zero. *)
    when List.length ranges <> 1
         && List.for_all
              (fun r -> match U.op r with Ops.Range | Ops.Const -> true | _ -> false)
              ranges ->
      (* A stage's shape is its range dims followed by the source's own shape.
         Only the range dims flatten into the index; folding the source's
         shape in as well overflows the size to a negative when the source
         carries an unresolved dimension. *)
      let range_dims = List.map (fun r -> Bound.to_int (Bound.succ (U.vmax r))) ranges in
      let shape = try U.max_shape n with Invalid_argument _ -> range_dims in
      let flat_idx = flat_index_of_ranges ~dims:range_dims ranges in
      let flat = U.stage ~src ~ranges:[ flat_idx ] ~opts in
      let ret = U.reshape ~src:flat ~shape:(shape_node shape) in
      let sym_shape =
        List.map
          (fun r ->
             match U.as_range r with
             | Some v when U.op v.size <> Ops.Const -> Some v.size
             | _ -> None)
          ranges
      in
      if List.for_all Option.is_none sym_shape then Some ret
      else
        let sym =
          sym_shape
          |> List.map (function Some dim -> dim | None -> int_ 1)
        in
        let size = match sym with [ dim ] -> dim | dims -> U.stack dims in
        let zeros = shape_node (List.map (fun _ -> 0) sym_shape) in
        Some (U.shrink ~src:ret ~offset:zeros ~size)
  | _ -> None

let range_axis_cmp a b = compare (U.axis_id a) (U.axis_id b)

let stage_to_store ?(allow_locals = true) counter n =
  match U.as_stage n with
  | Some { src; ranges; opts } ->
      (* A buffer is never weak: store at a committed width and cast the
         result back, so readers see the dtype the stage had. *)
      let buf_dtype = U.commit_dtype n in
      let read_back u = U.cast ~src:u ~dtype:(U.dtype n) in
      let shape =
        match shape_of n with
        | Some _ as shape -> shape
        | None -> (try Some (U.max_shape n) with Invalid_argument _ -> None)
      in
      let range_dims = List.map range_int_size ranges in
      let stage_dims =
        match shape with
        | Some shape when shape <> [] && List.length shape = List.length ranges ->
            shape
        | _ -> range_dims
      in
      let idx_expr =
        match ranges with
        | [ idx ] -> idx
        | _ -> flat_index_of_ranges ~dims:stage_dims ranges
      in
      let idx_ranges = List.sort range_axis_cmp (U.ranges idx_expr) in
      let size =
        match stage_dims with
        | _ :: _ -> prod stage_dims
        | _ ->
            let size_ranges =
              match idx_ranges with [] -> U.ranges src | ranges -> ranges
            in
            prod (List.map range_int_size size_ranges)
      in
      if size <= 0 then None
      else
        (match U.op src with
        | Ops.After ->
            let stores =
              List.filter (fun d -> match U.as_store d with
                  | Some { dst; _ } -> U.op dst = Ops.Index
                  | _ -> false) (src_tail src)
            in
            let buf = U.buf_uop (src0 src) in
            let cmp a b =
              let c = range_axis_cmp a b in
              if c = 0 then compare (U.tag a) (U.tag b) else c
            in
            let ended_stores =
              List.filter_map
                (fun store ->
                   let { U.dst; value; gate } =
                     Option.get (U.as_store store)
                   in
                   let target =
                     match U.as_index dst with
                     | Some { ptr = p; _ } ->
                         (match U.as_stage p with
                          | Some { src = inner; _ } when U.op inner = Ops.Index ->
                              inner
                          | _ -> dst)
                     | None -> dst
                   in
                   if value == target then None
                   else
                     let ranges =
                       List.sort_uniq cmp (U.ranges target @ idx_ranges)
                     in
                     Some
                       (U.end_
                          ~value:(U.store ~dst:target ~value ?gate ())
                          ~ranges))
                stores
            in
            if ended_stores = [] then Some buf
            else Some (U.after ~src:buf ~deps:ended_stores)
        | _ when opts.addrspace = Dtype.Global ->
            let id = !counter in
            incr counter;
            let buf =
              U.alloc ~slot:id ?device:opts.device ~shape:(shape_node [ size ])
                ~dtype:buf_dtype ()
            in
            let idx = U.index ~ptr:buf ~idxs:[ idx_expr ] () in
            let ended =
              U.end_
                ~value:(U.store ~dst:idx ~value:(U.cast ~src ~dtype:buf_dtype) ())
                ~ranges:idx_ranges
            in
            Some (read_back (U.after ~src:buf ~deps:[ ended ]))
        | _ when opts.addrspace = Dtype.Local && allow_locals ->
            let id = !counter in
            incr counter;
            let buf =
              U.buffer ~slot:id ~shape:(shape_node [ size ]) ~dtype:buf_dtype
                ~addrspace:Dtype.Local ()
            in
            let idx = U.index ~ptr:buf ~idxs:[ idx_expr ] () in
            let st =
              U.end_
                ~value:(U.store ~dst:idx ~value:(U.cast ~src ~dtype:buf_dtype) ())
                ~ranges:idx_ranges
            in
            Some
              (read_back
                 (U.after ~src:buf ~deps:[ U.barrier ~srcs:[ st ] () ]))
        | _ -> None)
  | None -> None

(* Split kernels *)

type split_context = {
  mutable slot : int;
  buf_map : U.t U.Ref_tbl.t;
  mutable formals : (int * U.t) list;
  (* Scalar bindings unbound inside the kernel, most recent first. *)
  mutable vars : U.t list;
  mutable range_ctr : int;
}

let create_split_context () =
  { slot = 0; buf_map = U.Ref_tbl.create 16; formals = [];
    vars = [];
    range_ctr = 0 }

let same_split_buffer a b =
  if a == b then true
  else
    let identity n =
      let b = U.buf_uop n in
      match U.op b, U.Arg.as_param_arg (U.arg b) with
      | (Ops.Buffer | Ops.Alloc | Ops.Param), Some p -> Some (U.op b, p.slot, p.addrspace)
      | _ -> None
    in
    match identity a, identity b with
    | Some ia, Some ib -> ia = ib
    | _ -> false

let find_buf_arg ctx key =
  U.Ref_tbl.find_opt ctx.buf_map key

let replace_formal_arg ctx old_arg new_arg =
  let same_arg arg =
    same_split_buffer arg old_arg
  in
  ctx.formals <-
    List.map
      (fun (slot, arg) -> (slot, if same_arg arg then new_arg else arg))
      ctx.formals

let debuf ctx n =
  let dtype = U.dtype n in
  let shape = U.shape n in
  let max_shape = U.max_shape n in
  let size = U.max_numel n in
  let addrspace =
    match U.addrspace n with Some a -> a | None -> Dtype.Global
  in
  let slot = ctx.slot in
  ctx.slot <- ctx.slot + 1;
  let ret =
    let device = U.device_of n in
    let volatile =
      match U.Arg.as_param_arg (U.arg (U.buf_uop n)) with
      | Some param -> param.volatile
      | None -> false
    in
    let param =
      U.param ~slot ~dtype ~shape:(shape_node [ size ]) ?device ~addrspace ~volatile ()
    in
    let reshaped = U.reshape ~src:param ~shape:(shape_node max_shape) in
    (* Symbolic buffers: the param is sized for [max_shape]; shrink the
       max-sized view down to the actual [shape] when they differ. *)
    if not (List.equal U.equal (U.shape reshaped) shape) then
      U.shrink ~src:reshaped
        ~offset:(shape_node (List.map (fun _ -> 0) shape))
        ~size:(match shape with [d] -> d | dims -> U.stack dims)
    else reshaped
  in
  let arg = match find_buf_arg ctx n with
    | Some arg -> arg
    | None ->
        U.Ref_tbl.replace ctx.buf_map n n;
        n
  in
  ctx.formals <- (slot, arg) :: ctx.formals;
  Some ret

let handle_after ctx n =
  let op = U.op n in
  let is_local = U.addrspace n = Some Dtype.Local in
  if is_local then None
  else
    let buf = match op with
      | Ops.After | Ops.Mstack | Ops.Mselect -> U.buf_uop n
      | _ -> n
    in
  (match find_buf_arg ctx buf with
     | None -> U.Ref_tbl.replace ctx.buf_map buf n
     | Some existing
       when same_split_buffer existing buf && op = Ops.After && src_tail n <> [] ->
         U.Ref_tbl.replace ctx.buf_map buf n;
         replace_formal_arg ctx existing n
     | Some _ -> ());
    Some buf

let unbind_kernel ctx n =
  if not (List.exists (( == ) n) ctx.vars) then ctx.vars <- n :: ctx.vars;
  Option.map (fun (v : U.bind_view) -> v.var) (U.as_bind n)

let renumber_range ctx n =
  match U.as_range n, U.node_tag n with
  | Some v, Some "" ->
      let axis = ctx.range_ctr in
      ctx.range_ctr <- ctx.range_ctr + 1;
      Some
        (U.range ~size:v.size ~axis ~sub:v.sub ~kind:v.kind
           ~dtype:(U.dtype n) ~parents:v.parents ())
  | _ -> None

(* Ranges are numbered in the order a depth-first walk first pops them,
   where a node's sources are pushed in order and a node already waiting on
   the stack is never pushed again. A range closed by an END or a REDUCE is
   pushed next to the body it closes and so is numbered after every range
   first met inside that body; a range nothing closes, such as the device
   range, is numbered where it is first read. Call bodies are not entered. *)
let renumber_kernel_ranges root =
  let on_stack = U.Ref_tbl.create 64 in
  let stack = ref [ root ] in
  U.Ref_tbl.replace on_stack root ();
  let ranges = ref [] in
  let rec run () =
    match !stack with
    | [] -> ()
    | n :: rest ->
        stack := rest;
        if U.op n = Ops.Range then ranges := n :: !ranges;
        let srcs = U.src n in
        let first =
          match U.op n with Ops.Call -> 1 | _ -> 0
        in
        for i = Array.length srcs - 1 downto first do
          let s = srcs.(i) in
          if not (U.Ref_tbl.mem on_stack s) then begin
            U.Ref_tbl.replace on_stack s ();
            stack := s :: !stack
          end
        done;
        run ()
  in
  run ();
  let all_ranges = List.rev !ranges in
  let mappings =
    List.mapi
      (fun axis r ->
        match U.as_range r with
        | Some v ->
            let r' =
              U.range ~size:v.size ~axis ~sub:v.sub ~kind:v.kind
                ~dtype:(U.dtype r) ~parents:v.parents ()
            in
            (r, r')
        | None -> assert false)
      all_ranges
  in
  U.substitute mappings root

let find_bufs n =
  (* A base buffer read through two INDEXes whose immediate pointer has a
     different op (e.g. a raw BUFFER vs an AFTER/STAGE over it) is a
     read/write cycle within the kernel. Key on the pointer op, matching
     tinygrad's [read_from.setdefault(buf, idx.src[0].op)]. *)
  let read_from : Ops.t U.Ref_tbl.t = U.Ref_tbl.create 8 in
  List.iter (fun s ->
      match U.as_index s with
      | Some { ptr; _ } ->
          let b = U.buf_uop ptr in
          (match U.op b with
           | Ops.Buffer | Ops.Alloc | Ops.Param ->
               let ptr_op = U.op ptr in
               (match U.Ref_tbl.find_opt read_from b with
                | Some prev when not (Ops.equal prev ptr_op) ->
                    failwith "cycle detected while indexing buffer"
                | _ -> U.Ref_tbl.replace read_from b ptr_op)
           | _ -> ())
      | None -> ())
    (* [enter_calls:false]: a precompiled call's payload (e.g. a staged
       loop's body linear) is a separate program, not part of this kernel;
       its INDEX/LOAD structure must not be read as this kernel's buffer
       accesses. *)
    (U.toposort n ~enter_calls:false ~gate:(fun x -> U.op x <> Ops.After));
  None

let to_define_global ctx n =
  match U.op n with
  | Ops.Store -> find_bufs n
  | Ops.Buffer when U.is_variable n -> Some (U.replace n ~op:Ops.Param ())
  | Ops.Buffer | Ops.Alloc | Ops.Mstack | Ops.Mselect -> debuf ctx n
  | Ops.Param -> (
      match U.as_param n with
      (* A named, ranged PARAM normalises to the canonical variable so
         binding identity survives the kernel split. *)
      | Some { param = { name = Some name; vmin_vmax = Some (lo, hi); multiple_of; volatile; _ }; _ } ->
          Some
            (U.param ~slot:(-1) ~name ~dtype:(U.dtype n) ~shape:(U.stack [])
               ~vmin_vmax:(lo, hi) ?multiple_of ~addrspace:Dtype.Alu ~volatile ())
      (* A scalar storage formal also needs the flat size-one kernel view.
         The tag prevents freshly created kernel parameters from being
         renumbered again. *)
      | Some { param = { name = None; _ }; _ }
        when U.node_tag n = Some "" ->
          debuf ctx n
      | _ -> None)
  | Ops.After when U.is_bound_var n -> unbind_kernel ctx n
  | Ops.After -> handle_after ctx n
  (* ALU params are scalar symbolic values, not buffers. *)
  | Ops.Index
    when Array.length (U.src n) = 1
         && (match U.as_param (src0 n) with
            | Some { param = { addrspace = Dtype.Alu; _ }; _ } -> true
            | _ -> false) ->
      Some (src0 n)
  | Ops.Stage ->
      (match U.as_stage n with
       | Some { opts; _ } when opts.device <> None ->
           Some
             (U.replace n
                ~arg:(U.Arg.Stage_info { opts with device = None }) ())
       | _ -> None)
  | Ops.Const when Array.length (U.src n) > 0 ->
      Some (U.replace n ~src:[||] ())
  | Ops.Range -> None
  | Ops.Noop when Array.length (U.src n) > 0 -> Some (src0 n)
  | _ -> None

let compact_kernel_params ctx body =
  let add_unique xs x = if List.exists (( == ) x) xs then xs else xs @ [ x ] in
  let topo = U.toposort body in
  let params =
    List.fold_left
      (fun acc n ->
         match U.as_param n with
         | Some { param = { slot; addrspace; _ }; _ }
           when slot >= 0 && addrspace <> Dtype.Alu ->
             add_unique acc n
         | _ -> acc)
      [] topo
  in
  (* [debuf] hands out slots in the order the kernel rewrite meets the
     buffers, and that order is the numbering: the rewrite reaches an operand
     shared by two consumers at the later of them, a toposort at the earlier. *)
  let slot_of n =
    match U.as_param n with Some { param; _ } -> param.slot | None -> -1
  in
  let params =
    List.stable_sort (fun a b -> Int.compare (slot_of a) (slot_of b)) params
  in
  let buffer_slot_map = List.mapi (fun slot param -> param, slot) params in
  let find_buffer_slot old =
    List.find_map
      (fun (param, slot) -> if param == old then Some slot else None)
      buffer_slot_map
  in
  let body =
    U.graph_rewrite ~name:"compact kernel params" ~walk:true
      (fun n ->
         match U.as_param n with
         | Some { param; _ }
           when param.slot >= 0 && param.addrspace <> Dtype.Alu -> (
             match find_buffer_slot n with
             | Some slot when slot <> param.slot ->
                 Some
                   (U.replace n
                      ~arg:(U.Arg.Param_arg { param with slot })
                      ())
             | _ -> None)
         | _ -> None)
      body
  in
  let bufs =
    List.map
      (fun param_node ->
         let old_slot =
           match U.as_param param_node with
           | Some { param; _ } -> param.slot
           | None -> assert false
         in
         match
           List.find_map
             (fun (slot, arg) -> if slot = old_slot then Some arg else None)
             ctx.formals
         with
         | Some arg -> arg
         | None -> param_node)
      params
  in
  body, bufs @ List.rev ctx.vars

let is_device_range r =
  match U.as_range r with
  | Some { kind = Axis_type.Device; _ } -> true
  | _ -> false

let split_store n =
  match U.op n with
  | Ops.Store | Ops.End ->
      (* An open loop range means the store belongs to an enclosing kernel.
         An open device range is fine: it is bound per device at launch. *)
      if List.exists (fun r -> not (is_device_range r)) (U.ranges n) then None
      else
        let ctx = create_split_context () in
        (* Stop at [After]: nodes behind a buffer boundary belong to already
           split upstream kernels, which the kernel rewrite prunes anyway.
           Walking into them would rescan the whole graph history per kernel. *)
        let nodes =
          U.toposort ~enter_calls:false ~gate:(fun x -> U.op x <> Ops.After) n
        in
        ctx.slot <-
          List.fold_left
            (fun acc nd ->
               match U.as_param nd with
               | Some { param = { slot; _ }; _ } when slot >= 0 ->
                   max acc (slot + 1)
               | _ -> acc)
            ctx.slot nodes;
        let rewrite =
          U.first_match
            [ to_define_global ctx; Simplify.flatten_range; Prepare.movement_ops ]
        in
        let ret =
          U.graph_rewrite ~bottom_up:true ~name:"kernel_split" rewrite n
        in
        let ret =
          match U.as_end ret with
          | Some { value; ranges } -> (
              match U.as_store value with
              | Some { dst; value = stored; gate } -> (
                  match U.as_index dst with
                  | Some { ptr; idxs } ->
                      let is_range r = U.op r = Ops.Range in
                      let stored_ranges =
                        List.filter is_range (U.ranges stored)
                      in
                      let same_size a b = Bound.equal (U.vmax a) (U.vmax b) in
                      let flat_size ranges =
                        List.fold_left
                          (fun acc r -> acc * (Bound.to_int (Bound.succ (U.vmax r))))
                          1 ranges
                      in
                      let same_flat_size dst stored =
                        flat_size dst = flat_size stored
                      in
                      let range_mem r ranges =
                        List.exists (U.equal r) ranges
                      in
                      let same_ranges a b =
                        List.length a = List.length b
                        && List.for_all (fun r -> range_mem r b) a
                      in
                      let sort_ranges ranges =
                        List.sort_uniq
                          (fun a b ->
                             let c = range_axis_cmp a b in
                             if c = 0 then compare (U.tag a) (U.tag b) else c)
                          ranges
                      in
                      let rewrite_flat_idx idx =
                        let dst_ranges =
                          List.filter is_range (U.ranges idx)
                        in
                        match dst_ranges, stored_ranges with
                        | [ dst ], [ stored ] when same_size dst stored ->
                            Some (`Substitute [ (dst, stored) ])
                        | _ :: _, _ :: _
                          when (not (same_ranges dst_ranges stored_ranges))
                               && same_flat_size dst_ranges stored_ranges ->
                            let gate_ranges =
                              match gate with
                              | None -> []
                              | Some gate -> U.ranges gate
                            in
                            if
                              List.exists
                                (fun r -> range_mem r gate_ranges)
                                dst_ranges
                            then None
                            else
                              Some
                                (`Replace
                                  ( flat_index_of_ranges stored_ranges,
                                    dst_ranges ))
                        | _ -> None
                      in
                      (match idxs with
                       | [ idx ] -> (
                           match rewrite_flat_idx idx with
                           | Some (`Substitute subs) ->
                           let idxs = List.map (U.substitute subs) idxs in
                           let dst = U.index ~ptr ~idxs () in
                           let ranges = List.map (U.substitute subs) ranges in
                           U.end_ ~value:(U.store ~dst ~value:stored ?gate ())
                             ~ranges
                           | Some (`Replace (idx, dst_ranges)) ->
                               let dst = U.index ~ptr ~idxs:[ idx ] () in
                               let ranges =
                                 ranges
                                 |> List.filter
                                      (fun r -> not (range_mem r dst_ranges))
                                 |> fun rs -> sort_ranges (rs @ stored_ranges)
                               in
                               U.end_
                                 ~value:(U.store ~dst ~value:stored ?gate ())
                                 ~ranges
                           | None -> ret)
                       | _ -> ret)
                  | None -> ret)
              | None -> ret)
          | None -> ret
        in
        let ret = renumber_kernel_ranges ret in
        let info : U.call_info =
          {
            grad_fxn = None;
            name = None;
            precompile = false;
            precompile_backward = false;
            dtype = Dtype.void;
            aux = None;
          }
        in
        (* Buffers can be on different devices here: the scheduler
           turns a kernel that is a copy into a transfer. *)
        let body, args =
          compact_kernel_params ctx
            (U.sink ~kernel_info:{
               name = "";
               applied_opts = []; opts_to_apply = None;
               estimates = None; beam = 0 } [ ret ])
        in
        Some (U.call ~body ~args ~info)
  | _ -> None

(* WAR deps *)

let fix_war_deps root =
  let afters =
    List.filter (fun n -> U.op n = Ops.After) (U.toposort root)
  in
  if afters = [] then root
  else
    let buf_of n = match U.op n with
      | Ops.After -> U.buf_uop n | _ -> n
    in
    let kernel_assign : U.t U.Ref_tbl.t = U.Ref_tbl.create 16 in
    List.iter (fun u -> U.Ref_tbl.replace kernel_assign (buf_of u) u) afters;
    let call_of u = match U.op u with
      | Ops.After -> List.find_opt (fun d -> U.op d = Ops.Call) (src_tail u)
      | _ -> None
    in
    let assign_rep : U.t list U.Ref_tbl.t = U.Ref_tbl.create 16 in
    List.iter (fun u ->
        let u_buf = buf_of u in
        let reads = match call_of u with
          | Some c ->
              List.filter (fun a -> Ops.Group.is_define (U.op a))
                (src_tail c)
          | None -> []
        in
        List.iter (fun s ->
            if s != u_buf then
              match U.Ref_tbl.find_opt kernel_assign s with
              | Some a -> (
                  (* A WAR dep between two AFTERs is only needed across
                     different calls: within one call the ordering is the
                     call's own business. Calls are unique graph nodes, so
                     identity is the test — structural equality walks both
                     calls' whole payload DAGs (a staged loop call carries
                     its entire body program), which is pathologically slow
                     and compares nodes that should simply be [==]. *)
                  match (call_of a, call_of u) with
                  | Some ca, Some cu when ca == cu -> ()
                  | _ ->
                      let prev =
                        Option.value
                          (U.Ref_tbl.find_opt assign_rep a)
                          ~default:[]
                      in
                      if not (List.exists (( == ) u) prev) then
                        U.Ref_tbl.replace assign_rep a (u :: prev))
              | _ -> ()) reads) afters;
    if U.Ref_tbl.length assign_rep = 0 then root
    else
      U.graph_rewrite ~name:"fix_war_deps" (fun n ->
          match U.Ref_tbl.find_opt assign_rep n with
          | Some extra when U.op n = Ops.After ->
              Some (U.after ~src:(src0 n) ~deps:(src_tail n @ extra))
          | _ -> None) root

(* Main pipeline *)

let post_rangeify_rules =
  U.first_match [
    Prepare.movement_ops;
    (* The constant fold runs here and not in every symbolic rewrite: it
       commits a cast of a constant to a concrete width, which is only safe
       once the ranges are built and nothing downstream still gets to choose
       that width. *)
    Upat.Pattern_matcher.rewrite symbolic;
    Upat.Pattern_matcher.rewrite Simplify.pm_reduce_simplify;
    cleanup_dead_axes;
    remove_noop_stage;
    remove_stage_index;
    (fun n -> match U.as_stage n with
       | Some { src; _ } when U.op src = Ops.Const ->
           (match U.arg src with
            | U.Arg.Value v -> Some (U.const v)
            | _ -> None)
       | _ -> None);
    (fun n -> match U.as_index n with
       | Some { ptr; _ } when U.op ptr = Ops.Const -> Some ptr
       | _ -> None);
    (fun n -> match U.op n with
       | Ops.Noop when Array.length (U.src n) > 0
                       && U.op (src0 n) = Ops.Const ->
           Some (src0 n)
       | _ -> None);
    (fun n -> match U.as_index n with
       | Some { ptr; _ } when U.op ptr = Ops.Mstack ->
           (match U.children ptr with
            | s :: _ ->
                let b = base s in
                (match U.arg b with
                 | U.Arg.Value v when U.op b = Ops.Const -> Some (U.const v)
                 | _ -> None)
            | [] -> None)
       | _ -> None);
  ]

let add_buffers_rules ?(allow_locals = true) counter =
  U.first_match [
    Prepare.movement_ops;
    flatten_stage;
    stage_to_store ~allow_locals counter;
    (* Index the buffer under the read-back cast the rule above adds, and cast
       the loaded value instead. Without this the expander widens the whole
       cast buffer into one vector. *)
    (fun n -> match U.as_index n with
       | Some { ptr; _ } when U.op ptr = Ops.Cast && Dtype.is_weak (U.dtype ptr)
         ->
           let idxs = src_tail n in
           Some
             (U.cast ~dtype:(U.dtype n)
                ~src:(U.index ~ptr:(src0 ptr) ~idxs ()))
       | _ -> None);
    (* Tag PARAMs and RANGEs so later passes can tell an original node from
       one freshly created during the kernel split. *)
    (fun n -> match U.op n, U.node_tag n with
       | (Ops.Param | Ops.Range), None -> Some (U.with_tag "" n)
       | _ -> None);
    (* RESHAPEs through MSELECT/MSTACK *)
    (fun n -> match U.op n with
       | Ops.Mselect | Ops.Mstack ->
           let children = U.children n in
           if children <> []
              && List.for_all (fun c -> U.op c = Ops.Reshape) children
           then
             let unwrapped = List.map (fun c -> base (src0 c)) children in
             let inner = U.replace n ~src:(Array.of_list unwrapped) () in
             (* Always restore the shape view, including the rank-0 one: a
                scalar read acquires its flat 0 index by moving through the
                reshape. *)
             (match shape_of n with
              | Some sh -> Some (U.reshape ~src:inner ~shape:(shape_node sh))
              | None -> Some inner)
           else None
       | _ -> None);
    (* Strip RESHAPE on CALL args *)
    (fun n -> match U.op n with
       | Ops.Call ->
           let args = src_tail n in
           let new_args =
             List.map (fun a -> if U.op a = Ops.Reshape then src0 a else a) args
           in
           if List.for_all2 ( == ) args new_args then None
           else
             Some
               (U.replace n
                  ~src:(Array.of_list (src0 n :: new_args)) ())
       | _ -> None);
    (* Strip MOP on AFTER deps; flatten nested AFTERs *)
    (fun n -> match U.op n with
       | Ops.After ->
           let deps = src_tail n in
           let new_deps =
             List.map (fun d ->
                 Option.value (movement_src d) ~default:d) deps
           in
           let flat =
             List.concat_map (fun d -> match U.op d with
                 | Ops.After -> src_tail d | _ -> [ d ]) new_deps
           in
           if List.length flat = List.length deps
              && List.for_all2 ( == ) flat deps then None
           else Some (U.after ~src:(src0 n) ~deps:flat)
       | _ -> None);
    (* Remove invalid writes: a STORE of an Invalid constant (possibly
       through STAGE) is a NOOP. *)
    (fun n -> match U.as_store n with
       | Some { value; gate = None; _ } ->
           let value =
             match U.op value with
             | Ops.Stage when Array.length (U.src value) > 0 ->
                 src0 value
             | _ -> value
           in
           if is_invalid value then Some (U.noop ())
           else None
       | _ -> None);
    (fun n -> match U.op n with
       | Ops.After ->
           let deps = src_tail n in
           let real =
             List.filter (fun d -> match U.op d with
                 | Ops.Noop when Array.length (U.src d) = 0 -> false
                 | Ops.End ->
                     let v = src0 d in
                     not (U.op v = Ops.Noop && Array.length (U.src v) = 0)
                 | _ -> true) deps
           in
           if List.length real < List.length deps then
             match real with
             | [] -> Some (src0 n)
             | _ -> Some (U.after ~src:(src0 n) ~deps:real)
           else None
      | _ -> None);
  ]

let get_kernel_graph root =
  let root = Prepare.prepare_rangeify root in
  let rctx =
    Indexing.run_rangeify root ~shapes:shape_of ~shape_exprs:shape_expr_of
  in
  let root = Indexing.apply_rangeify_pass rctx root in
  let root =
    U.graph_rewrite ~name:"post_rangeify" post_rangeify_rules root
  in
  let root =
    U.graph_rewrite ~name:"limit_bufs" (limit_bufs rctx) root
  in
  let buffer_slot_start =
    List.fold_left (fun acc x ->
        let slot =
          match U.Arg.as_param_arg (U.arg x) with
          | Some { slot; _ } -> Some slot
          | None -> None
        in
        match slot with
        | Some slot when slot >= 0 -> max acc (slot + 1)
        | Some _ | None -> acc)
      0 (U.toposort root)
  in
  let counter = ref buffer_slot_start in
  let root =
    U.graph_rewrite ~name:"add_buffers" ~bottom_up:true
      (add_buffers_rules ~allow_locals:false counter) root
  in
  let root =
    U.graph_rewrite ~enter_calls:false ~bottom_up:true
      ~name:"split_kernels" split_store root
  in
  let root =
    U.graph_rewrite ~enter_calls:false ~bottom_up:true
      ~name:"split_kernels_fixpoint" split_store root
  in
  fix_war_deps root
