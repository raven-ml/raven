(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Schedule pipeline orchestrator.

   The main entry point is {!get_kernel_graph}, which transforms a tensor-level
   SINK into a graph with CALL nodes wrapping Kernel.t ASTs. *)

open Tolk_ir
module T = Tensor
module K = Kernel
module D = Dtype
module C = Const

(* Helpers *)

let _openpilot_hacks = Device.Context.int ~name:"OPENPILOT_HACKS" ~default:0
let dtype_or_void prog id = Option.value ~default:D.void (T.dtype prog id)
let shape_prod = List.fold_left ( * ) 1

(* Earliest_rewrites *)

(* fix_store_after_hazard: make source contiguous if it has hazardous movement
   ops on the dest buffer.

   Walks src's toposort gated at CONTIGUOUS boundaries, tracking which nodes
   reach the target's base.  If any reaching node is an unsafe movement op
   (Permute, Flip, or Shrink when target also has Shrink), inserts Contiguous
   on the value. *)
let fix_store_after_hazard _b (program : T.t) ~target ~value =
  let unsafe_set =
    let has_shrink =
      List.exists (fun dep ->
        match T.view program dep with T.Shrink _ -> true | _ -> false)
        (T.backward_slice program target)
    in
    fun v -> match v with
      | T.Permute _ | T.Flip _ -> true
      | T.Shrink _ -> has_shrink
      | _ -> false
  in
  let base = T.base program target in
  (* Walk value's backward slice, skipping past Contiguous boundaries,
     tracking reachability to base. *)
  let slice = T.backward_slice program value in
  let reaches = Hashtbl.create (List.length slice) in
  let found = ref false in
  List.iter (fun dep ->
    if not !found then begin
      let dep_v = T.view program dep in
      (* Gate: don't propagate reachability through Contiguous *)
      let is_reachable =
        dep = base ||
        (match dep_v with T.Contiguous _ -> false | _ ->
          List.exists (fun c -> Hashtbl.find_opt reaches c = Some true)
            (T.children program dep))
      in
      Hashtbl.replace reaches dep is_reachable;
      if is_reachable && unsafe_set dep_v then found := true
    end)
    slice;
  if !found then T.contiguous _b ~src:value () else value

(* normalize_store_after_target_chain: walk through AFTER chains to root
   target, insert contiguous if RHS depends on the previous assign result.
   Returns None if nothing changed. *)
let normalize_store_after_target_chain _b (program : T.t) ~target ~value
    ~extras =
  let root_target = ref target in
  while
    match T.view program !root_target with
    | After { src = t; deps = ds; _ }
      when List.exists (fun d ->
        match T.view program d with Store _ -> true | _ -> false)
      ds -> root_target := t; true
    | _ -> false
  do () done;
  let value' =
    if List.mem target (T.backward_slice program value) then
      T.contiguous _b ~src:value ()
    else value
  in
  if !root_target = target && value' = value then None
  else
    let new_id = T.assign _b ~target:!root_target ~value:value' ~extras () in
    Some (T.view (T.finish _b) new_id)

(* Multi-stage canonicalization pass run before rangeify. Handles:
   - Detach/Contiguous_backward -> After (strip autodiff markers).
   - Allreduce resolution via handle_allreduce (collective communication).
   - Copy elision when source and target device match.
   - Reshape(Reshape(x)) -> Reshape(x) flattening.
   - Assign (After+Store) normalization:
     1. Bitcast target chasing.
     2. Inner AFTER wrapping for view assigns.
     3. Hazard detection via fix_store_after_hazard.
     4. Root target chain normalization.
   - Zero-element tensor folding (reduce over empty -> identity constant,
     any zero-size op -> zero constant). *)
let earliest_rewrites (_b : T.builder) ~(shapes : int list option array)
    ~(devices : T.device option array) (program : T.t) (_emit : T.view -> T.id)
    (id : T.id) (v : T.view) : T.view option =
  match v with
  | Detach { src; _ } | Contiguous_backward { src; _ } ->
      Some (T.After { src; deps = []; dtype = dtype_or_void program src })
  | Allreduce { src = buf; device; op; dtype } ->
      let red_shape = Option.value ~default:[] shapes.(id) in
      let red_size = shape_prod red_shape in
      Allreduce.create_allreduce_function _b ~shapes ~devices ~buf ~red_op:op
        ~red_device_id:device ~red_dtype:dtype ~red_shape ~red_size ()
      |> Option.map (fun result_id -> T.view (T.finish _b) result_id)
  | Copy { src; device; _ } -> (
      match devices.(src), T.view program device with
      | Some (Single src_dev), Device { device = Single copy_dev }
        when src_dev = copy_dev ->
          Some (T.Noop { src = Some src; dtype = dtype_or_void program src })
      | _ -> None)
  | Reshape { src; shape; dtype } -> (
      match T.view program src with
      | Reshape { src = inner_src; _ } ->
          Some (T.Reshape { src = inner_src; shape; dtype })
      | _ -> None)
  | After { src = buf; deps; dtype }
    when List.exists (fun d ->
      match T.view program d with Store _ -> true | _ -> false) deps ->
      (* Extract Store dep and remaining extras *)
      let store_info =
        List.find_map (fun d ->
          match T.view program d with
          | Store { dst = target; value } -> Some (d, target, value)
          | _ -> None) deps
      in
      (match store_info with
      | None -> None
      | Some (store_id, target, value) ->
          let extras =
            List.filter (fun d -> d <> store_id) deps
          in
          (* Rule 1: Bitcast target chasing *)
          match T.view program target with
          | Bitcast { src = inner_target; _ } ->
              let dt = Option.value ~default:dtype
                  (T.dtype program inner_target) in
              let new_value = T.bitcast _b ~src:value ~dtype:dt in
              let new_id = T.assign _b ~target:inner_target ~value:new_value
                ~extras () in
              Some (T.view (T.finish _b) new_id)
          | _ ->
              (* Rule 2: Inner AFTER wrapping for view assigns.
                 When target is a view (different shape from buf), wrap the store
                 in an inner AFTER so the store gets its own ranges. *)
              let target_shape = shapes.(target) in
              let buf_shape = shapes.(buf) in
              if target_shape <> buf_shape
                 && target_shape <> None && buf_shape <> None then begin
                let inner_after = T.after _b ~src:target
                  ~deps:[ store_id ] in
                let new_id = T.after _b ~src:buf
                  ~deps:(inner_after :: extras) in
                Some (T.view (T.finish _b) new_id)
              end
              (* Rule 3: fix_store_after_hazard *)
              else
                let value' =
                  fix_store_after_hazard _b program ~target ~value in
                if value' <> value then begin
                  let new_id = T.assign _b ~target ~value:value' ~extras () in
                  Some (T.view (T.finish _b) new_id)
                end
                (* Rule 4: normalize target chain.
                   Only fires when target is itself an After (assign). *)
                else match T.view program target with
                | After _ ->
                    normalize_store_after_target_chain _b program
                      ~target ~value ~extras
                | _ -> None)
  | Reduce_axis { src; op; dtype; _ }
    when (match shapes.(src) with
         | Some s -> shape_prod s = 0 | None -> false)
      && (match shapes.(id) with
         | Some s -> shape_prod s > 0 | None -> false) ->
      Some (T.view (T.finish _b) (T.const _b (C.identity_element op dtype)))
  | Sink _ -> None
  | _ when (match shapes.(id) with
           | Some s -> shape_prod s = 0 | None -> false) ->
      let dt = dtype_or_void program id in
      let zero = T.const _b (C.zero dt) in
      Some (T.view (T.finish _b) zero)
  | _ -> None

(* Flat-index helpers *)

let range_sizes (lookup : T.id -> T.view) (rngs : T.id list) : int list =
  List.map (fun r ->
    match lookup r with
    | T.Range { size; _ } -> (
        match lookup size with
        | T.Const { value; _ } -> (
            match Const.view value with Int n -> Int64.to_int n | _ -> 1)
        | _ -> 1)
    | _ -> 1) rngs

let compute_flat_index_emit (emit : T.view -> T.id) (rngs : T.id list) (shape : int list) : T.id =
  let mk_const n = emit (T.Const { value = Const.int D.index n; dtype = D.index; srcs = [] }) in
  let n = List.length shape in
  if n = 0 then mk_const 0
  else begin
    let strides = Array.make n 1 in
    for i = n - 2 downto 0 do
      strides.(i) <- strides.(i + 1) * List.nth shape (i + 1)
    done;
    let terms =
      List.mapi
        (fun i rng ->
          let stride = strides.(i) in
          if stride = 0 then None
          else if stride = 1 then Some rng
          else
            let s = mk_const stride in
            Some (emit (T.Binary { op = `Mul; lhs = rng; rhs = s; dtype = D.index })))
        rngs
    in
    let terms = List.filter_map Fun.id terms in
    match terms with
    | [] -> mk_const 0
    | first :: rest ->
        List.fold_left
          (fun acc t -> emit (T.Binary { op = `Add; lhs = acc; rhs = t; dtype = D.index }))
          first rest
  end

(* Bufferize_to_store *)

let bufferize_size lookup ranges =
  List.fold_left (fun acc r ->
    match lookup r with
    | T.Range { size; _ } -> (
        match lookup size with
        | Const { value; _ } -> (
            match Const.view value with Int n -> acc * Int64.to_int n | _ -> acc)
        | _ -> acc)
    | _ -> acc)
    1 ranges

let bufferize_to_store ~(devices : T.device option array)
    (lunique_counter : int ref) ~(lookup : T.id -> T.view) (emit : T.view -> T.id)
    (id : T.id) (v : T.view) : T.view option =
  match v with
  | T.Bufferize { src; ranges; dtype; opts } ->
      let size = bufferize_size lookup ranges in
      if size <= 0 then None
      else begin
        let sorted_rngs =
          List.sort (fun a b ->
            match lookup a, lookup b with
            | Range { axis = a1; _ }, Range { axis = a2; _ } -> compare a1 a2
            | _ -> 0)
            ranges
        in
        let range_ids =
          List.filter (fun r -> match lookup r with Range _ -> true | _ -> false)
            sorted_rngs
        in
        (* Extract Store target+value from the After+Store assign pattern *)
        let assign_target_value () =
          match lookup src with
          | After { deps; _ } ->
              List.find_map (fun d ->
                match lookup d with
                | Store { dst; value; _ } -> Some (dst, value)
                | _ -> None) deps
          | _ -> None
        in
        match assign_target_value () with
        | Some (target, value) -> (
            let value = ref value in
            while match lookup !value with
              | Noop { src = Some s; _ } -> value := s; true
              | _ -> false
            do () done;
            let value = !value in
            match lookup target with
            | Index _ ->
                let rec find_base id = match lookup id with
                  | Reshape { src; _ } | Expand { src; _ } | Pad { src; _ }
                  | Shrink { src; _ } | Permute { src; _ } | Flip { src; _ } ->
                      find_base src
                  | _ -> id
                in
                let store = emit (T.Store { dst = target; value }) in
                let ended = emit (T.End { value = store; ranges = range_ids }) in
                let base_buf = find_base target in
                let dt = Option.value ~default:dtype
                    (T.node_dtype (lookup base_buf)) in
                Some (T.After { src = base_buf; deps = [ ended ]; dtype = dt })
            | _ -> None)
        | None ->
            let flat_idxs =
              if List.length sorted_rngs > 1 then
                let sizes = range_sizes lookup sorted_rngs in
                [ compute_flat_index_emit emit sorted_rngs sizes ]
              else sorted_rngs
            in
            if opts.addrspace = D.Global then begin
              let luniq_id = !lunique_counter in
              incr lunique_counter;
              let luniq = emit (T.Lunique { id = luniq_id }) in
              let device = match opts.device with
                | Some (Kernel.Device_single d) -> T.Single d
                | Some (Kernel.Device_multi ds) -> T.Multi ds
                | _ -> match devices.(id) with
                  | Some d -> d
                  | None -> Single "CPU"
              in
              let dev = emit (T.Device { device }) in
              let buf = emit (T.Buffer { unique = luniq; device = dev;
                                         size; dtype }) in
              let idx = emit (T.Index { ptr = buf; idxs = flat_idxs;
                                        gate = None; dtype }) in
              let store = emit (T.Store { dst = idx; value = src }) in
              let ended = emit (T.End { value = store; ranges = range_ids }) in
              Some (T.After { src = buf; deps = [ ended ]; dtype })
            end else begin
              incr lunique_counter;
              let ptr_dtype = D.ptr_of dtype ~addrspace:D.Local ~size in
              let def_local = emit (T.Define_local { size; dtype = ptr_dtype }) in
              let idx = emit (T.Index { ptr = def_local; idxs = flat_idxs;
                                        gate = None; dtype }) in
              let store = emit (T.Store { dst = idx; value = src }) in
              let ended = emit (T.End { value = store; ranges = range_ids }) in
              let bar = emit T.Barrier in
              let ended_bar = emit (T.End { value = bar; ranges = [ ended ] }) in
              Some (T.After { src = def_local; deps = [ ended_bar ]; dtype })
            end
      end
  | _ -> None

(* Split_kernels *)

let kernel_backward_slice program root =
  let n = T.length program in
  let visited = Array.make n false in
  let rec visit id =
    if not visited.(id) then begin
      visited.(id) <- true;
      match T.view program id with
      | Param _ | Buffer _ | Device _ | Unique _ | Lunique _ -> ()
      | v -> List.iter visit (T.children_of v)
    end
  in
  visit root;
  let acc = ref [] in
  for i = n - 1 downto 0 do
    if visited.(i) then acc := i :: !acc
  done;
  !acc

let tensor_subtree_to_kernel ~shapes ~(slot_map : (int, int) Hashtbl.t)
    program root =
  let slice = kernel_backward_slice program root in
  let tbl = Hashtbl.create (List.length slice) in
  let buf_map = Hashtbl.create 8 in
  let next_slot = ref (Hashtbl.length slot_map) in
  let get_slot id =
    match Hashtbl.find_opt slot_map id with
    | Some s -> s
    | None ->
        let s = !next_slot in
        incr next_slot;
        Hashtbl.replace slot_map id s;
        s
  in
  let lookup id =
    match Hashtbl.find_opt tbl id with
    | Some k -> k
    | None -> failwith (Printf.sprintf
        "tensor_subtree_to_kernel: missing id %d" id)
  in
  let param_size id =
    match shapes.(id) with
    | Some dims -> List.fold_left ( * ) 1 dims
    | None -> 1
  in
  let map = List.map lookup in
  List.iter (fun id ->
    let k = match T.view program id with
      | Const { value; _ } -> K.const value
      | Range { size; axis; sub; kind; dtype } ->
          K.range ~size:(lookup size) ~axis ~sub ~kind ~dtype ()
      | End { value; ranges } ->
          K.end_ ~value:(lookup value) ~ranges:(map ranges) ()
      | Index { ptr; idxs; gate; _ } ->
          K.index ~ptr:(lookup ptr) ~idxs:(map idxs)
            ?gate:(Option.map lookup gate) ~as_ptr:false ()
      | Store { dst; value } ->
          K.store ~dst:(lookup dst) ~value:(lookup value) ~ranges:[]
      | Reduce { src; ranges; op; dtype } ->
          K.reduce ~op ~src:(lookup src) ~ranges:(map ranges) ~dtype
      | Unary { op; src; _ } -> K.unary ~op ~src:(lookup src)
      | Binary { op; lhs; rhs; _ } ->
          K.binary ~op ~lhs:(lookup lhs) ~rhs:(lookup rhs)
      | Ternary { op; a; b; c; _ } ->
          K.ternary ~op ~a:(lookup a) ~b:(lookup b) ~c:(lookup c)
      | Cast { src; dtype } -> K.cast ~src:(lookup src) ~dtype:(D.to_any dtype)
      | Bitcast { src; dtype } -> K.bitcast ~src:(lookup src) ~dtype
      | Vectorize { srcs; _ } -> K.vectorize ~srcs:(map srcs)
      | Define_var { name; lo; hi; dtype } -> K.define_var ~name ~lo ~hi ~dtype ()
      | Define_local { size; dtype } -> K.define_local ~size ~dtype
      | Barrier -> K.barrier
      | Invalid_index _ -> K.invalid_index ()
      | Bufferize { src; ranges; dtype; opts } ->
          K.bufferize ~src:(lookup src) ~ranges:(map ranges)
            ~dtype:(D.ptr_of dtype ~addrspace:opts.addrspace ~size:1)
            ~opts
      | After { src; deps; _ } -> K.after ~src:(lookup src) ~deps:(map deps)
      | Sink { srcs; kernel_info } -> K.sink ?kernel_info (map srcs)
      | Noop { src = Some s; _ } -> lookup s
      | Noop { src = None; _ } -> K.const (C.int D.index 0)
      | Param { dtype; _ } ->
          let slot = get_slot id in
          let size = param_size id in
          K.param ~idx:slot
            ~dtype:(D.ptr_of dtype ~addrspace:Global ~size)
      | Buffer { size = buf_size; dtype; _ } ->
          let slot = get_slot id in
          Hashtbl.replace buf_map id id;
          K.param ~idx:slot
            ~dtype:(D.ptr_of dtype ~addrspace:Global ~size:buf_size)
      | Bind { var = src; _ } | Contiguous { src; _ }
      | Reshape { src; _ } | Expand { src; _ } | Pad { src; _ }
      | Shrink { src; _ } | Permute { src; _ } | Flip { src; _ } -> lookup src
      | Device _ | Unique _ | Lunique _ -> K.const (C.int D.index 0)
      | v ->
          failwith (Printf.sprintf
            "tensor_subtree_to_kernel: unexpected op at id %d: %s" id
            (Format.asprintf "%a" T.pp_view v))
    in
    Hashtbl.replace tbl id k)
    slice;
  (lookup root, buf_map)

let split_store ~shapes program ~(lookup : T.id -> T.view)
    (_emit : T.view -> T.id) (id : T.id) (v : T.view) : T.view option =
  match v with
  | End _ ->
      (* Walk to collect buffer and variable args *)
      let visited = Hashtbl.create 32 in
      let bufs = ref [] in
      let vars = ref [] in
      let rec walk nid =
        if not (Hashtbl.mem visited nid) then begin
          Hashtbl.replace visited nid ();
          let nv = lookup nid in
          (match nv with
          | T.Buffer _ | T.Param _ -> bufs := nid :: !bufs
          | T.Bind _ -> vars := nid :: !vars
          | _ -> ());
          match nv with
          | T.After _ -> ()
          | _ -> List.iter walk (T.children_of nv)
        end
      in
      List.iter walk (T.children_of v);
      let ordered_bufs = List.rev !bufs in
      (* Build slot mapping: sequential from 0 in walk encounter order *)
      let slot_map = Hashtbl.create 8 in
      List.iteri (fun i buf_id -> Hashtbl.replace slot_map buf_id i) ordered_bufs;
      let kernel_ast =
        fst (tensor_subtree_to_kernel ~shapes ~slot_map program id)
      in
      let kernel_sink =
        K.sink
          ~kernel_info:{ K.name = ""; axis_kinds = []; dont_use_locals = false;
                         applied_opts = []; opts_to_apply = None;
                         estimates = None }
          [ kernel_ast ]
      in
      let args = ordered_bufs @ List.rev !vars in
      let call_info : T.call_info =
        { grad_fxn = None; metadata = []; name = None; precompile = false }
      in
      let dtype = Option.value ~default:D.void (T.node_dtype v) in
      Some (T.Call { callee = Ast kernel_sink; args; info = call_info; dtype })
  | _ -> None

(* WAR dependency fixup *)

(* Detects write-after-read hazards between After nodes and adds explicit
   ordering dependencies to eliminate them. Builds a map from each buffer to
   its writing After node, then for each After's kernel Call, checks whether
   any read buffer is also written by another After. When a conflict is found,
   the writer After gains a dependency on the reader After, ensuring the read
   completes before the write. Cycles are detected and rejected. *)
let fix_war_deps program =
  let n = T.length program in
  let afters = ref [] in
  for i = 0 to n - 1 do
    match T.view program i with After _ -> afters := i :: !afters | _ -> ()
  done;
  let afters = List.rev !afters in
  if afters = [] then program
  else begin
    let buf_of_after id =
      match T.view program id with
      | After { src; _ } -> T.base program src
      | _ -> id
    in
    let kernel_assign = Hashtbl.create 16 in
    List.iter (fun u -> Hashtbl.replace kernel_assign (buf_of_after u) u) afters;
    let call_of_after u =
      match T.view program u with
      | After { deps; _ } ->
          List.find_opt
            (fun d -> match T.view program d with Call _ -> true | _ -> false)
            deps
      | _ -> None
    in
    let assign_rep = Hashtbl.create 16 in
    List.iter (fun u ->
      let u_buf = buf_of_after u in
      let u_call = call_of_after u in
      let reads = match u_call with
        | Some call_id -> (
            match T.view program call_id with
            | Call { args; _ } ->
                List.filter (fun a -> match T.view program a with
                  | Buffer _ | Param _ -> true | _ -> false) args
            | _ -> [])
        | None -> []
      in
      List.iter (fun s ->
        if s <> u_buf then
          match Hashtbl.find_opt kernel_assign s with
          | Some a ->
              let a_call = call_of_after a in
              if a_call <> None && a_call = u_call then ()
              else begin
                let u_after_buf = Option.value ~default:u
                    (Hashtbl.find_opt kernel_assign u_buf) in
                let has_cycle =
                  List.exists (fun dep ->
                    match T.view program dep with
                    | After _ -> T.base program dep = s
                    | _ -> false)
                    (T.backward_slice program u_after_buf)
                in
                if has_cycle then
                  failwith (Printf.sprintf
                    "cycle detected in assign graph, buffers %d and %d \
                     have circular dependency" s u_buf)
                else begin
                  let prev = Option.value ~default:[]
                      (Hashtbl.find_opt assign_rep a) in
                  if not (List.mem u prev) then begin
                    Hashtbl.replace assign_rep a (u :: prev);
                    Hashtbl.replace kernel_assign s a
                  end
                end
              end
          | None -> ())
        reads)
      afters;
    if Hashtbl.length assign_rep = 0 then program
    else
      T.rebuild (fun id v ->
        match Hashtbl.find_opt assign_rep id with
        | Some extra_deps -> (
            match v with
            | After { src; deps; dtype } ->
                Some (T.After { src; deps = deps @ extra_deps; dtype })
            | _ -> None)
        | None -> None)
        program
  end

(* Get_kernel_graph *)

let get_kernel_graph (program : T.t) : T.t =
  let shapes = T.compute_shapes program in
  let devices = T.compute_devices program in
  (* Step 1: multi_pm *)
  let program =
    let b = T.create () in
    T.rewrite_fixpoint_grow
      (fun ~lookup:_ _emit _id v ->
        Multi.multi_pm b ~shapes ~devices program v)
      program
  in
  let shapes = T.compute_shapes program in
  let devices = T.compute_devices program in
  (* Step 2: earliest_rewrites *)
  let program =
    let b = T.create () in
    T.rewrite_fixpoint_grow
      (fun ~lookup:_ emit id v ->
        earliest_rewrites b ~shapes ~devices program emit id v)
      program
  in
  let shapes = T.compute_shapes program in
  let devices = T.compute_devices program in
  (* Step 3: run_rangeify *)
  let b = T.create () in
  let ctx, program = Indexing.run_rangeify b program ~shapes in
  (* Step 4: pm_apply_rangeify *)
  let program = Indexing.apply_rangeify_pass program ctx ~shapes ~devices in
  let _shapes = T.compute_shapes program in
  let devices = T.compute_devices program in
  (* Step 5: post-rangeify optimization — const folding + noop elimination *)
  let program =
    T.rewrite_fixpoint_grow
      (fun ~lookup _emit _id v ->
        match v with
        | Bufferize { src; _ } | Index { ptr = src; _ } -> (
            match lookup src with
            | Const { value; dtype; _ } ->
                Some (T.Const { value; dtype; srcs = [] })
            | _ -> None)
        | Copy { src; dtype; _ } -> (
            match lookup src with
            | Const { value; _ } ->
                Some (T.Const { value; dtype; srcs = [] })
            | _ -> None)
        | Noop { src = Some s; _ } -> (
            match lookup s with Const _ -> Some (lookup s) | _ -> None)
        | _ -> None)
      program
  in
  (* pm_remove_bufferize *)
  let program =
    let old_program = program in
    T.rewrite_fixpoint_grow
      (fun ~lookup emit _id v ->
        match v with
        | Index { ptr; idxs; _ } -> (
            match lookup ptr with
            | Bufferize { src; opts; _ } when opts.removable ->
                if Indexing.is_always_contiguous (lookup src) then None
                else begin
                  let old_ptr = match T.view old_program _id with
                    | Index { ptr; _ } -> ptr | _ -> -1
                  in
                  if old_ptr < 0 then None
                  else
                    let old_src = match T.view old_program old_ptr with
                      | Bufferize { src; _ } -> src | _ -> -1
                    in
                    if old_src < 0 then None
                    else begin
                      let accessed_buffers = ref 0 in
                      let reduce_srcs = ref [] in
                      let visited = Hashtbl.create 32 in
                      let rec walk id =
                        if not (Hashtbl.mem visited id) then begin
                          Hashtbl.replace visited id ();
                          match T.view old_program id with
                          | Bufferize { opts = o; _ }
                            when o.addrspace = D.Global ->
                              incr accessed_buffers
                          | Mstack _ -> incr accessed_buffers
                          | Param _ ->
                              incr accessed_buffers;
                              List.iter walk (T.children old_program id)
                          | Reduce { src = rsrc; _ } ->
                              reduce_srcs := rsrc :: !reduce_srcs;
                              List.iter walk (T.children old_program id)
                          | _ ->
                              List.iter walk (T.children old_program id)
                        end
                      in
                      walk old_src;
                      if !accessed_buffers > 3 then None
                      else begin
                        let has_buf_in_reduce = ref false in
                        List.iter (fun rsrc ->
                          let visited2 = Hashtbl.create 16 in
                          let rec walk_r id =
                            if not !has_buf_in_reduce
                               && not (Hashtbl.mem visited2 id)
                            then begin
                              Hashtbl.replace visited2 id ();
                              match T.view old_program id with
                              | Param _ | Bufferize _ ->
                                  has_buf_in_reduce := true
                              | _ ->
                                  List.iter walk_r (T.children old_program id)
                            end
                          in
                          walk_r rsrc)
                          !reduce_srcs;
                        if !has_buf_in_reduce then None
                        else begin
                          let subst = Hashtbl.create 8 in
                          let ranges = match lookup ptr with
                            | Bufferize { ranges; _ } -> ranges | _ -> []
                          in
                          List.iter2 (fun br ir ->
                            match lookup br, lookup ir with
                            | Const _, _ | _, Invalid_index _ -> ()
                            | _ -> Hashtbl.replace subst br ir)
                            ranges idxs;
                          if Hashtbl.length subst = 0 then Some (lookup src)
                          else begin
                            let rebuilt = Hashtbl.create 32 in
                            let rec rebuild_node id =
                              match Hashtbl.find_opt rebuilt id with
                              | Some new_id -> new_id
                              | None ->
                                  let new_id =
                                    match Hashtbl.find_opt subst id with
                                    | Some replacement -> replacement
                                    | None ->
                                        let v = lookup id in
                                        let children = T.children_of v in
                                        let new_children =
                                          List.map rebuild_node children in
                                        if new_children = children then id
                                        else begin
                                          let tbl = Hashtbl.create 4 in
                                          List.iter2 (Hashtbl.replace tbl)
                                            children new_children;
                                          emit (T.map_children (fun c ->
                                            Option.value ~default:c
                                              (Hashtbl.find_opt tbl c)) v)
                                        end
                                  in
                                  Hashtbl.replace rebuilt id new_id;
                                  new_id
                            in
                            Some (lookup (rebuild_node src))
                          end
                        end
                      end
                    end
                end
            | _ -> None)
        | _ -> None)
      program
  in
  (* Step 6: pm_add_buffers *)
  let program =
    let lunique_counter = ref 0 in
    T.rewrite_fixpoint_grow
      (fun ~lookup emit id v ->
        match bufferize_to_store ~devices lunique_counter ~lookup emit id v with
        | Some _ as r -> r
        | None -> (
            match v with
            | Call { callee; args; info; dtype } ->
                let new_args = List.map (fun a ->
                  match lookup a with
                  | Reshape { src; _ } -> src | _ -> a) args
                in
                if new_args <> args then
                  Some (T.Call { callee; args = new_args; info; dtype })
                else None
            | After { src; deps; dtype } ->
                let new_deps = List.map (fun d ->
                  match lookup d with
                  | Reshape { src = inner; _ } | Expand { src = inner; _ }
                  | Permute { src = inner; _ } | Flip { src = inner; _ }
                  | Pad { src = inner; _ } | Shrink { src = inner; _ } -> inner
                  | _ -> d) deps
                in
                let flat_deps = List.concat_map (fun d ->
                  match lookup d with
                  | After { deps = inner_deps; _ } -> inner_deps
                  | _ -> [ d ]) new_deps
                in
                if flat_deps <> deps then
                  Some (T.After { src; deps = flat_deps; dtype })
                else None
            | _ -> None))
      program
  in
  (* Step 6.5: flatten multi-idx INDEX -> single flat index with strides. *)
  let program =
    T.rewrite_fixpoint_grow
      (fun ~lookup emit _id v ->
        match v with
        | T.Index { ptr; idxs; gate; dtype } when List.length idxs > 1 ->
            let is_range_or_const0 idx =
              match lookup idx with
              | T.Range _ -> true
              | T.Const { value; _ } -> (
                  match Const.view value with Int 0L -> true | _ -> false)
              | _ -> false
            in
            if not (List.for_all is_range_or_const0 idxs) then None
            else begin
              let sizes = range_sizes lookup idxs in
              let flat_idx = compute_flat_index_emit emit idxs sizes in
              Some (T.Index { ptr; idxs = [flat_idx]; gate; dtype })
            end
        | _ -> None)
      program
  in
  (* Step 7: split_kernels *)
  let shapes = T.compute_shapes program in
  let program =
    T.rewrite_fixpoint_grow
      (fun ~lookup emit id v -> split_store ~shapes program ~lookup emit id v)
      program
  in
  (* Step 8: WAR dependency fixup *)
  fix_war_deps program
