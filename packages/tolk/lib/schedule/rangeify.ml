(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Schedule pipeline.

   Transforms a tensor-level SINK into a kernel graph with CALL nodes
   wrapping Kernel.t ASTs. The pipeline:

    1. multi_pm — multi-device rewriting
    2. fold_moved_after — openpilot AFTER folding (when enabled)
    3. earliest_rewrites — syntactic sugar, movement ops, canonicalization
    4. run_rangeify — core range analysis (in Indexing)
    5. apply_rangeify — bottom-up rewrite with rangeify context
    6. post-rangeify — buffer folding, const folding, buffer removal
    7. limit_bufs — insert bufferize when too many input buffers
    8. add_buffers — BUFFERIZE → STORE + BUFFER
    9. split_kernels — STORE/END → CALL(kernel SINK)
   10. WAR deps — write-after-read dependency fixup *)

open Tolk_ir
module T = Tensor
module K = Kernel
module D = Dtype
module C = Const

(* Context variables *)

let openpilot_hacks_var =
  Helpers.Context_var.int ~key:"OPENPILOT_HACKS" ~default:0
let float16_var = Helpers.Context_var.int ~key:"FLOAT16" ~default:0
let split_reduceop_var =
  Helpers.Context_var.int ~key:"SPLIT_REDUCEOP" ~default:1
let split_threshold_var =
  Helpers.Context_var.int ~key:"REDUCEOP_SPLIT_THRESHOLD" ~default:32768
let split_size_var =
  Helpers.Context_var.int ~key:"REDUCEOP_SPLIT_SIZE" ~default:22
let max_kernel_buffers_var =
  Helpers.Context_var.int ~key:"MAX_KERNEL_BUFFERS" ~default:0

(* Helpers *)

let int_ n = T.const (C.int D.Val.index n) D.index
let shape_prod = List.fold_left ( * ) 1
let dtype_or_void n = match T.dtype n with Some d -> d | None -> D.void

(* Encode an int list as a shape tensor (Vectorize of Consts). *)
let shape_node dims =
  match List.map int_ dims with [d] -> d | ds -> T.vectorize ~srcs:ds

(* Follow src through AFTER nodes to the root buffer. *)
let rec root_after n =
  match T.view n with After { src; _ } -> root_after src | _ -> n

(* Extract the concrete size of each range (1 for non-range or symbolic). *)
let range_sizes rngs =
  List.map (fun r ->
    match T.view r with
    | Range { size; _ } ->
        (match T.view size with
        | Const { value; _ } ->
            (match C.view value with Int n -> Int64.to_int n | _ -> 1)
        | _ -> 1)
    | _ -> 1) rngs

(* Compute a single flat index from multi-dimensional ranges and shape
   using row-major strides. *)
let compute_flat_index rngs shape =
  let n = List.length shape in
  if n = 0 then int_ 0
  else
    let shape = Array.of_list shape in
    let strides = Array.make n 1 in
    for i = n - 2 downto 0 do
      strides.(i) <- strides.(i + 1) * shape.(i + 1)
    done;
    let terms = List.filter_map (fun (i, rng) ->
      if strides.(i) = 0 then None
      else if strides.(i) = 1 then Some rng
      else Some (T.binary ~op:`Mul ~lhs:rng ~rhs:(int_ strides.(i))))
      (List.mapi (fun i rng -> (i, rng)) rngs) in
    match terms with
    | [] -> int_ 0
    | first :: rest ->
        List.fold_left (fun acc t ->
          T.binary ~op:`Add ~lhs:acc ~rhs:t) first rest

(* lower_shaped_wmma: lowers tensor-level Shaped_wmma to kernel-level Wmma
   with CONTRACT/UNROLL. Blocked on the Tensor IR gaining a Shaped_wmma
   variant — tinygrad defines SHAPED_WMMA in uop/ops.py but our tensor.mli
   doesn't have it yet. The kernel IR (Wmma, Contract, Gep) is ready. *)

let is_elementwise = function
  | T.Unary _ | T.Binary _ | T.Ternary _ | T.Cast _ | T.Bitcast _
  | T.Const _ -> true
  | _ -> false

let is_movement = function
  | T.Reshape _ | T.Expand _ | T.Pad _ | T.Shrink _ | T.Permute _
  | T.Flip _ -> true
  | _ -> false

let movement_src = function
  | T.Reshape { src; _ } | T.Expand { src; _ } | T.Pad { src; _ }
  | T.Shrink { src; _ } | T.Permute { src; _ } | T.Flip { src; _ } -> src
  | _ -> assert false

let argsort order =
  let indexed = List.mapi (fun i o -> (o, i)) order in
  let sorted = List.sort (fun (a, _) (b, _) -> compare a b) indexed in
  List.map snd sorted

let device_max_bufs = function
  | "METAL" -> 31 | "WEBGPU" -> 8 | _ -> 0

(* Syntactic sugar *)

(* INDEX(INDEX(ptr, idxs1), idxs2) → INDEX(ptr, idxs1 @ idxs2) when the
   inner INDEX is a pointer type and the outer is not. *)
let index_concat n =
  match T.view n with
  | Index { ptr; idxs; gate; dtype } ->
      (match T.view ptr with
      | Index { ptr = inner_ptr; idxs = inner_idxs; dtype = inner_dt; _ } ->
          (match inner_dt with
          | D.Ptr _ when (match dtype with D.Ptr _ -> false | _ -> true) ->
              Some (T.index ~ptr:inner_ptr
                ~idxs:(inner_idxs @ idxs) ?gate ~dtype ())
          | _ -> None)
      | _ -> None)
  | _ -> None

(* INDEX on elementwise/const: push the INDEX into the sources. *)
let early_rangeify n =
  match T.view n with
  | Index { ptr; idxs; dtype; _ } when idxs <> [] ->
      let v = T.view ptr in
      if is_elementwise v then
        let new_children = List.map (fun s ->
          T.index ~ptr:s ~idxs ~dtype ()) (T.children ptr) in
        Some (T.replace ptr ~children:new_children ())
      else None
  | _ -> None

(* Movement ops *)

(* Push a movement op through INDEX by applying it to the ranges. *)
let mop_through_index shapes n =
  match T.view n with
  | Index { ptr; idxs; gate; dtype } when is_movement (T.view ptr) ->
      let v = T.view ptr in
      let src = movement_src v in
      (* Check len(idxs) == len(ptr.shape), matching tinygrad's
         len(idx.src[1:]) == len(r.shape) where r is the movement op.
         For Ptr-typed PARAM sources (from debuf), derive src_shape from
         the ptr dtype size when compute_shapes returns None. *)
      (let src_shape = match shapes src with
        | Some _ as s -> s
        | None -> match T.dtype src with
          | Some (D.Ptr p) -> Some [D.Ptr.size p]
          | _ -> None in
      match src_shape, shapes ptr with
      | Some _, Some ptr_shape when List.length idxs = List.length ptr_shape ->
          let shapes_with_ptr n = match shapes n with
            | Some _ as s -> s
            | None -> match T.dtype n with
              | Some (D.Ptr p) -> Some [D.Ptr.size p]
              | _ -> None in
          let new_idxs = Indexing.apply_movement_op ~shapes:shapes_with_ptr v idxs in
          Some (T.index ~ptr:src ~idxs:new_idxs ?gate ~dtype ())
      | _ -> None)
  | _ -> None

(* Move movement ops and INDEX past AFTER (but not when AFTER has a raw
   STORE with shaped children — from replace_contig_with_store_after). *)
let mop_past_after shapes n =
  let v = T.view n in
  if not (is_movement v || (match v with Index _ -> true | _ -> false))
  then None
  else
    let src = match v with
      | Index { ptr; _ } -> ptr | _ -> movement_src v in
    match T.view src with
    | After { src = after_src; deps; _ } ->
        if shapes after_src = None then None
        else if List.exists (fun d ->
          match T.view d with
          | Store { dst; _ } -> shapes dst <> None
          | _ -> false) deps
        then None
        else
          let new_after = T.after ~src:after_src ~deps in
          Some (T.replace n
            ~children:(new_after :: List.tl (T.children n)) ())
    | _ -> None

(* Strip movement ops from END: they don't affect the closed ranges. *)
let mop_past_end n =
  match T.view n with
  | End { value; ranges } when is_movement (T.view value) ->
      Some (T.end_ ~value:(movement_src (T.view value)) ~ranges)
  | _ -> None

(* Fold moved AFTERs — openpilot hack *)

(* Walk through PERMUTE/RESHAPE/WHERE+PAD on the store value to find
   the underlying source, adjusting the AFTER inverse accordingly.
   Called only when OPENPILOT_HACKS is set. *)
let found_after ctx ~after ~value =
  let x = ref value in
  let a = ref after in
  (* CAST float16 → walk through *)
  (if Helpers.Context_var.get float16_var <> 0 then
    match T.view !x with
    | Cast { src; dtype } when dtype = D.float16 ->
        x := src;
        a := T.cast ~src:!a ~dtype:D.float32
    | _ -> ());
  let continue_ = ref true in
  while !continue_ do
    match T.view !x with
    | Permute { src; order; _ } ->
        x := src;
        a := T.permute ~src:!a ~order:(argsort order)
    | Reshape { src; _ } ->
        let src_shape = T.extract_int_shape
          (List.nth (T.children !x) 1) in
        (match src_shape with
        | Some s ->
            x := src;
            a := T.reshape ~src:!a ~shape:(shape_node s)
        | None -> continue_ := false)
    | Ternary { op = `Where; b = pad_src; c = false_; _ }
      when (match T.view (T.base false_) with
            | Invalid_index _ -> true | _ -> false)
        && (match T.view pad_src with Pad _ -> true | _ -> false) ->
        let pad_inner = match T.view pad_src with
          | Pad { src; _ } -> src | _ -> assert false in
        (* XXX shrink bounds from pad.marg not yet extracted — we walk
           through the pad to its source without adjusting the AFTER.
           Tinygrad shrinks using (l, s-r) from the PAD arg and shape. *)
        x := pad_inner;
    | _ -> continue_ := false
  done;
  Hashtbl.replace ctx (T.tag !x) !a

(* Earliest rewrites *)

(* Walk AFTER chain on the store target to find the root buffer. If the
   store value depends on the target, insert contiguous to break the
   dependency cycle. *)
let normalize_store_after_target_chain ~target ~value =
  let root = root_after target in
  let value =
    if List.exists (fun n -> n == target) (T.toposort value)
    then T.contiguous ~src:value ()
    else value
  in
  T.after ~src:root ~deps:[T.store ~dst:root ~value]

(* Make the store value contiguous if it reaches the target buffer through
   hazardous movement ops. PERMUTE and FLIP reorder indices; SHRINK can
   have overlapping regions when the destination is also shrunk. *)
let fix_store_after_hazard ~buf ~target ~value =
  let is_unsafe =
    let has_shrink = List.exists (fun n ->
      match T.view n with Shrink _ -> true | _ -> false)
      (T.toposort target) in
    fun v -> match v with
    | T.Permute _ | T.Flip _ -> true
    | T.Shrink _ -> has_shrink
    | _ -> false
  in
  let base = T.base target in
  let slice = T.toposort value
    ~gate:(fun s -> match T.view s with Contiguous _ -> false | _ -> true) in
  let reaches : (int, bool) Hashtbl.t = Hashtbl.create (List.length slice) in
  let found = ref false in
  List.iter (fun s ->
    if not !found then begin
      let r = s == base ||
        List.exists (fun c ->
          Hashtbl.find_opt reaches (T.tag c) = Some true)
          (T.children s) in
      Hashtbl.replace reaches (T.tag s) r;
      if r && is_unsafe (T.view s) then found := true
    end) slice;
  if !found then
    Some (T.after ~src:buf
            ~deps:[T.store ~dst:target ~value:(T.contiguous ~src:value ())])
  else None

(* Resolve a CALL by inlining the callee: gather PARAMs from the body,
   map each to the corresponding argument by slot, and substitute.
   Kernel calls (SINK with kernel_info), precompiled calls, and Ast
   callees are never resolved — they are real invocations. *)
let resolve_call n =
  match T.view n with
  | Call { callee = Ref body; args; info; _ } ->
      let is_kernel_sink = match T.view body with
        | Sink { kernel_info = Some _; _ } -> true | _ -> false in
      if info.precompile || is_kernel_sink then None
      else
        let params =
          List.filter (fun x -> match T.view x with Param _ -> true | _ -> false)
            (T.toposort body) in
        let params = List.sort (fun a b ->
          match T.view a, T.view b with
          | Param { slot = sa; _ }, Param { slot = sb; _ } -> compare sa sb
          | _ -> 0) params in
        let mappings = List.filter_map (fun p ->
          match T.view p with
          | Param { slot; _ } when slot < List.length args ->
              Some (p, List.nth args slot)
          | _ -> None) params in
        Some (T.substitute mappings body)
  | _ -> None

(* Detect which axes of [src] are expanded (broadcast from size 1) by
   pushing ranges through the movement-op chain and seeing which survive.
   An axis whose range disappears was introduced by EXPAND.  Tinygrad
   uses index+substitute+pm_mops; we walk the chain directly since
   movement ops are linear and apply_movement_op is the same transform
   pm_mops invokes. *)
let detect_expanded shapes src =
  let src_shape = match shapes src with Some s -> s | None -> [] in
  let n = List.length src_shape in
  if n = 0 then []
  else
    let rngs = List.mapi (fun i s ->
      if s > 1 then T.range ~size:(int_ s) ~axis:i ~kind:Axis_kind.Loop ()
      else int_ 0) src_shape in
    let rec push node rngs =
      let v = T.view node in
      match v with
      | Reshape { src; _ } | Expand { src; _ } | Pad { src; _ }
      | Shrink { src; _ } | Permute { src; _ } | Flip { src; _ } ->
          push src (Indexing.apply_movement_op ~shapes v rngs)
      | _ -> rngs
    in
    let final = push src rngs in
    let live = List.concat_map (fun r ->
      List.filter_map (fun x ->
        match T.view x with Range { axis; _ } -> Some axis | _ -> None)
        (r :: T.backward_slice r)) final in
    List.init n (fun i -> not (List.mem i live))

(* Split a large reduce into two phases for better GPU occupancy. The
   dimension is factored: phase 1 reduces within each chunk, phase 2
   reduces across chunks. Only applies when the reduction ratio exceeds
   a threshold and the chosen axis is not an expanded broadcast. *)
let split_reduceop shapes n =
  match T.view n with
  | Reduce_axis { src; op; axes; _ } ->
      (match shapes n, shapes src with
      | Some red_shape, Some src_shape
        when shape_prod red_shape > 0
          && Helpers.Context_var.get split_reduceop_var <> 0
          && shape_prod src_shape / shape_prod red_shape
             >= Helpers.Context_var.get split_threshold_var ->
          let is_expanded = detect_expanded shapes src in
          let max_div = min 256
            (1 lsl Helpers.Context_var.get split_size_var
             / shape_prod red_shape) in
          let candidates = List.concat_map (fun i ->
            if List.nth is_expanded i then []
            else
              let dim = List.nth src_shape i in
              let rec try_div d acc =
                if d < 8 then List.rev acc
                else if dim mod d = 0 then try_div (d - 1) ((i, d) :: acc)
                else try_div (d - 1) acc
              in
              try_div max_div []) axes in
          (match candidates with
          | [] -> None
          | (dim, divisor) :: _ ->
              let nd = List.length src_shape in
              let split_shape = List.init (nd + 1) (fun i ->
                if i < dim then List.nth src_shape i
                else if i = dim then divisor
                else if i = dim + 1 then List.nth src_shape dim / divisor
                else List.nth src_shape (i - 1)) in
              let perm = List.init (nd + 1) (fun i ->
                if i < dim then i
                else if i < nd then i + 1
                else dim) in
              let splitted =
                T.permute ~order:perm
                  ~src:(T.reshape ~src ~shape:(shape_node split_shape)) in
              let phase1 = T.contiguous
                ~src:(T.reduce_axis ~src:splitted ~op ~axes) () in
              let phase2 = T.reduce_axis ~src:phase1 ~op
                ~axes:[List.length red_shape] in
              Some (T.reshape ~src:phase2 ~shape:(shape_node red_shape)))
      | _ -> None)
  | _ -> None

(* Post-rangeify cleanups *)

(* BUFFERIZE(INDEX(ptr, idxs), ranges) is identity when idxs = ranges.
   Remove both and return ptr, shrunk to the bufferize shape. *)
let remove_noop_bufferize ~idxs ~ranges ~ptr ~buf_shape =
  if not (List.equal (==) idxs ranges) then None
  else begin match T.view ptr with
  | Buffer_view _ -> None
  | _ ->
    match buf_shape with
    | Some shape when shape <> [] ->
        Some (T.shrink ~src:ptr
                ~before:(shape_node (List.map (fun _ -> 0) shape))
                ~after:(shape_node shape))
    | _ -> Some ptr
  end

let is_always_run = function
  | T.Contiguous _ | T.Copy _ | T.Noop _ -> true | _ -> false

(* Remove dead axes from BUFFERIZE. An axis is dead if its range is a
   constant or is not referenced by the source computation. Dead axes
   are collapsed to size 1 via reshape, then restored via expand. *)
let cleanup_dead_axes shapes n =
  match T.view n with
  | Bufferize { src; ranges; dtype; opts } ->
      let src_v = T.view src in
      (* Never touch CONTIGUOUS/COPY/NOOP sources or plain AFTERs *)
      if is_always_run src_v then None
      else if (match src_v with After _ -> true | _ -> false) then None
      else
        let shape = match shapes n with Some s -> s | None -> [] in
        if List.length shape <> List.length ranges then None
        else
          (* Bail on symbolic range sizes *)
          let has_symbolic = List.exists (fun r ->
            match T.view r with
            | Range { size; _ } ->
                (match T.view size with Const _ -> false | _ -> true)
            | _ -> false) ranges in
          if has_symbolic then None
          else
            let src_ranges = T.ranges src in
            let hit = ref false in
            let new_ranges = ref [] in
            let reshape_dims = ref [] in
            List.iter2 (fun s rng ->
              let dead = match T.view rng with
                | Const _ -> true
                | Range _ ->
                    not (List.exists (fun r -> r == rng) src_ranges)
                | _ -> false in
              if dead then begin
                reshape_dims := 1 :: !reshape_dims;
                hit := true
              end else begin
                reshape_dims := s :: !reshape_dims;
                new_ranges := rng :: !new_ranges
              end) shape ranges;
            if not !hit then None
            else
              let new_ranges = List.rev !new_ranges in
              let reshape_shape = List.rev !reshape_dims in
              let ret = T.bufferize ~src ~ranges:new_ranges ~dtype ~opts in
              let ret = T.reshape ~src:ret ~shape:(shape_node reshape_shape) in
              Some (T.expand ~src:ret ~shape:(shape_node shape))
  | _ -> None

let pcontig_var = Helpers.Context_var.int ~key:"PCONTIG" ~default:0

(* Decide whether a BUFFERIZE can be removed by re-expressing its source
   inline. The cost function counts accessed buffers and checks whether
   reduce sources reference buffers — if so, the intermediate buffer is
   needed for locality and we keep it. *)
let remove_bufferize ~src ~buf_ranges ~buf_shape ~idx_ranges ~removable =
  assert (List.length buf_ranges = List.length idx_ranges);
  let src_v = T.view src in
  if is_always_run src_v || not removable then None
  else
    (* Walk source subtree: count accessed buffers, collect indexes and
       reduces. Stop descending at global BUFFERIZE and MSTACK. *)
    let accessed = Hashtbl.create 8 in
    let indexes = ref [] in
    let reduces = ref [] in
    ignore (T.toposort src ~gate:(fun x ->
      match T.view x with
      | Bufferize { opts = { addrspace = D.Global; _ }; _ } ->
          Hashtbl.replace accessed (T.tag x) (); false
      | Mstack _ ->
          Hashtbl.replace accessed (T.tag x) (); false
      | Param _ ->
          Hashtbl.replace accessed (T.tag x) (); true
      | Index _ ->
          indexes := x :: !indexes; true
      | Reduce _ ->
          reduces := x :: !reduces; true
      | _ -> true));
    let pcontig = Helpers.Context_var.get pcontig_var in
    if Hashtbl.length accessed > 3 && pcontig <= 2 then None
    else
      (* Check if any reduce's source transitively references a buffer *)
      let buffer_in_reduce =
        if !reduces = [] then false
        else begin
          let rsrcs = List.filter_map (fun r ->
            match T.view r with
            | Reduce { src; _ } -> Some src | _ -> None) !reduces in
          let found = ref false in
          ignore (T.toposort (T.sink rsrcs) ~gate:(fun x ->
            if !found then false
            else match T.view x with
            | Param _ | Bufferize _ -> found := true; false
            | _ -> true));
          !found
        end
      in
      if buffer_in_reduce then begin
        if pcontig > 2 then begin
          (* Partial contig: keep ranges that overlap local indexes or
             are used by reduce axes, bufferize only those. *)
          let buf_size = match buf_shape with
            | Some s -> shape_prod s | None -> 1 in
          let in_size = Hashtbl.length accessed in
          let out_in_ratio =
            float_of_int (buf_size + 1) /. float_of_int (in_size + 1) in
          if out_in_ratio < 10.0 then None
          else
            let local_indexes = List.filter (fun x ->
              match T.view x with
              | Index { ptr; _ } ->
                  (match T.view ptr with
                  | Bufferize { opts = { addrspace = D.Local; _ }; _ } -> true
                  | _ -> false)
              | _ -> false) !indexes in
            let exclude_ranges =
              List.concat_map (fun x ->
                match T.view x with
                | Index { idxs; _ } -> T.ranges (T.group idxs)
                | _ -> []) local_indexes in
            let subs = List.filter_map (fun (k, v) ->
              match T.view k with Const _ -> None | _ -> Some (k, v))
              (List.combine buf_ranges idx_ranges) in
            let is_pcontig, is_subs = List.partition (fun (k, v) ->
              List.exists (fun r -> r == k) exclude_ranges ||
              List.exists (fun r ->
                match T.view r with
                | Range { kind; _ } -> kind = Axis_kind.Reduce
                | _ -> false) (T.ranges v)) subs in
            if is_subs = [] then None
            else
              let ret = T.substitute is_subs src in
              if is_pcontig = [] then Some ret
              else
                let pc_rngs = List.map fst is_pcontig in
                let pc_idxs = List.map snd is_pcontig in
                let opts : K.bufferize_opts =
                  { device = None; addrspace = D.Local; removable = true } in
                let dtype = match T.dtype src with
                  | Some d -> d | None -> D.float32 in
                let buf = T.bufferize ~src:ret ~ranges:pc_rngs ~dtype ~opts in
                Some (T.index ~ptr:buf ~idxs:pc_idxs ~dtype ())
        end else None
      end
      else
        (* Safe to remove: substitute BUFFERIZE ranges → INDEX ranges *)
        let mappings = List.filter_map (fun (k, v) ->
          match T.view k with
          | Const _ -> None
          | _ -> (match T.view v with
            | Invalid_index _ -> None
            | _ -> Some (k, v)))
          (List.combine buf_ranges idx_ranges) in
        Some (T.substitute mappings src)

(* Handle DISK/TINYFS buffer views: compute offset from the INDEX and
   create a BUFFER_VIEW node. *)
let late_buffer_view devices n =
  match T.view n with
  | Bufferize { src; ranges; _ } ->
      (match T.view src with
      | Bitcast _ | Contiguous _ ->
          let dev = devices n in
          let is_disk = match dev with
            | Some (T.Single d) ->
                String.length d >= 4
                && (String.sub d 0 4 = "DISK" || String.sub d 0 6 = "TINYFS")
            | _ -> false in
          if not is_disk then None
          else
            let shape = range_sizes ranges in
            let size = shape_prod shape in
            (* Walk up to find the INDEX *)
            let rec find_index x =
              match List.find_opt (fun u ->
                match T.view u with Index _ -> true | _ -> false)
                (T.children x) with
              | Some idx -> idx
              | None -> match T.children x with
                | c :: _ -> find_index c | [] -> x
            in
            let idx = find_index src in
            let offset = match T.view idx with
              | Index { idxs; _ } when idxs = [] -> 0
              | Index { idxs; _ } ->
                  (* XXX tinygrad uses idx.vmin (symbolic minimum) for
                     each index. We approximate with const values only;
                     symbolic indices contribute 0. This is wrong for
                     non-const offsets on DISK buffers. *)
                  List.fold_left (fun acc i ->
                    match T.view i with
                    | Const { value; _ } ->
                        (match C.view value with
                        | Int n -> acc + Int64.to_int n | _ -> acc)
                    | _ -> acc) 0 idxs
                  |> max 0
              | _ -> 0
            in
            let idx_base = T.base idx in
            let bv = T.buffer_view ~src:idx_base ~size ~offset
              ~dtype:(dtype_or_void src) in
            let rng_node = match ranges with
              | [r] -> r | _ -> List.hd ranges in
            Some (T.replace n
              ~children:[bv; rng_node] ())
      | _ -> None)
  | _ -> None

(* Insert BUFFERIZE for elementwise sources when a kernel exceeds the
   device's buffer limit. Each source gets its own ranges so it
   materializes independently. *)
let limit_bufs (ctx : Indexing.indexing_context) devices n =
  match T.view n with
  | Binary _ | Ternary _ ->
      let dev_name = match devices n with
        | Some (T.Single d) ->
            Some (List.hd (String.split_on_char ':' d))
        | Some (T.Multi ds) ->
            Some (List.hd (String.split_on_char ':' (List.hd ds)))
        | None -> None in
      Option.bind dev_name (fun dname ->
        let max_bufs =
          match Helpers.Context_var.get max_kernel_buffers_var with
          | 0 -> device_max_bufs dname
          | n -> n in
        if max_bufs = 0 then None
        else
          let bufs = Hashtbl.create 16 in
          ignore (T.toposort n ~gate:(fun u ->
            match T.view u with
            | Bufferize _ | After _ | Param _ | Mselect _
            | Mstack _ | Define_var _ ->
                Hashtbl.replace bufs (T.tag u) (); false
            | _ -> true));
          if Hashtbl.length bufs <= max_bufs - 1 then None
          else
            let children = T.children n in
            let new_children = List.map (fun s ->
              let sv = T.view s in
              if is_elementwise sv && devices s <> None then
                let orig_ranges = T.ranges s in
                let new_ranges = List.map (fun x ->
                  match T.view x with
                  | Range { size; sub; dtype; _ } ->
                      let axis = ctx.range_idx in
                      ctx.range_idx <- ctx.range_idx + 1;
                      T.range ~size ~axis ~sub
                        ~kind:Axis_kind.Loop ~dtype ()
                  | _ -> x) orig_ranges in
                let dev = match devices s with
                  | Some (T.Single d) -> Some (K.Device_single d)
                  | Some (T.Multi ds) -> Some (K.Device_multi ds)
                  | None -> None in
                let opts : K.bufferize_opts =
                  { device = dev; addrspace = D.Global;
                    removable = true } in
                let dtype = match T.dtype s with
                  | Some d -> d | None -> D.float32 in
                let subst = T.substitute
                  (List.combine orig_ranges new_ranges) s in
                let buf = T.bufferize ~src:subst ~ranges:new_ranges
                  ~dtype ~opts in
                T.index ~ptr:buf ~idxs:orig_ranges ~dtype ()
              else s) children in
            if List.for_all2 (==) children new_children then None
            else Some (T.replace n ~children:new_children ()))
  | _ -> None

(* Add buffers *)

(* Collapse multi-range BUFFERIZE into a single flat index. If the
   BUFFERIZE already has one range, nothing to do. *)
let flatten_bufferize shapes n =
  match T.view n with
  | Bufferize { src; ranges; dtype; opts } when List.length ranges > 1 ->
      let flat_idx = compute_flat_index ranges (range_sizes ranges) in
      let ret = T.bufferize ~src ~ranges:[flat_idx] ~dtype ~opts in
      let buf_shape = match shapes n with
        | Some s -> s
        | None -> range_sizes ranges
      in
      let ret = T.reshape ~src:ret ~shape:(shape_node buf_shape) in
      (* If any range has symbolic size, shrink to actual bounds *)
      let has_symbolic = List.exists (fun r ->
        match T.view r with
        | Range { size; _ } ->
            (match T.view size with Const _ -> false | _ -> true)
        | _ -> false) ranges in
      if has_symbolic then
        let sym = List.map (fun r ->
          match T.view r with
          | Range { size; _ } ->
              (match T.view size with Const _ -> int_ 1 | _ -> size)
          | _ -> int_ 1) ranges in
        let before = shape_node (List.map (fun _ -> 0) sym) in
        let after = match sym with [d] -> d | ds -> T.vectorize ~srcs:ds in
        Some (T.shrink ~src:ret ~before ~after)
      else Some ret
  | _ -> None

(* Convert BUFFERIZE to STORE + BUFFER. Three paths:
   - AFTER: the source is an assign — wrap existing store in END
   - GLOBAL: allocate a new buffer, index, store, END
   - LOCAL: like GLOBAL but with DEFINE_LOCAL and barrier (not used when
     allow_locals=false in the main pipeline) *)
let bufferize_to_store counter n =
  match T.view n with
  | Bufferize { src; ranges; dtype; opts } ->
      (* Extract Range nodes from the expression tree — after
         flatten_bufferize, ranges may contain a single flat index
         expression rather than raw Range nodes. *)
      let all_ranges = List.filter (fun r ->
        match T.view r with T.Range _ -> true | _ -> false)
        (T.toposort (T.sink ranges)) in
      let size = shape_prod (range_sizes all_ranges) in
      if size <= 0 then None
      else
        let range_nodes =
          List.sort (fun a b ->
            match T.view a, T.view b with
            | Range { axis = a1; _ }, Range { axis = a2; _ } -> compare a1 a2
            | _ -> 0) all_ranges in
        let rngs = range_nodes in
        let ptr_dt =
          D.Ptr.create (D.val_of dtype) ~addrspace:opts.addrspace ~size in
        (match T.view src with
        (* AFTER path: source is an assign (AFTER+STORE) *)
        | After { src = after_src; deps; _ } ->
            let stores = List.filter (fun d ->
              match T.view d with
              | Store { dst; _ } ->
                  (match T.view dst with Index _ -> true | _ -> false)
              | _ -> false) deps in
            let buf = T.base after_src in
            (match stores with
            | [] -> Some buf
            | store :: _ ->
                let dst, store_val = match T.view store with
                  | Store { dst; value } -> dst, value
                  | _ -> assert false in
                (* Walk through BUFFERIZE(INDEX(…)) on the store target *)
                let target = match T.view dst with
                  | Index { ptr; _ } ->
                      (match T.view ptr with
                      | Bufferize { src = inner; _ } ->
                          (match T.view inner with
                          | Index _ -> inner | _ -> dst)
                      | _ -> dst)
                  | _ -> dst in
                if store_val == target then
                  Some (T.after ~src:buf ~deps:[])
                else
                  let target_rngs = T.ranges target in
                  let all_rngs = List.sort_uniq (fun a b ->
                    match T.view a, T.view b with
                    | Range { axis = a1; _ }, Range { axis = a2; _ } ->
                        compare a1 a2
                    | _ -> compare (T.tag a) (T.tag b))
                    (target_rngs @ range_nodes) in
                  let ended = T.end_
                    ~value:(T.store
                      ~dst:(T.replace target ~dtype:(D.Ptr ptr_dt) ())
                      ~value:store_val)
                    ~ranges:all_rngs in
                  Some (T.after ~src:buf ~deps:[ended]))
        (* GLOBAL path: new buffer *)
        | _ when opts.addrspace = D.Global ->
            let luniq_id = !counter in
            incr counter;
            let dev = match opts.device with
              | Some (K.Device_single d) -> T.device (T.Single d)
              | Some (K.Device_multi ds) -> T.device (T.Multi ds)
              | Some (K.Device_index _) | None ->
                  T.device (T.Single "CPU") in
            let buf = T.buffer ~unique:(T.lunique ~id:luniq_id)
              ~device:dev ~size ~dtype in
            let idx = T.index ~ptr:buf ~idxs:rngs
              ~dtype:(D.Ptr ptr_dt) () in
            let ended = T.end_
              ~value:(T.store ~dst:idx ~value:src)
              ~ranges:range_nodes in
            Some (T.after ~src:buf ~deps:[ended])
        (* LOCAL path: DEFINE_LOCAL + barrier *)
        | _ when opts.addrspace = D.Local ->
            incr counter;
            let buf = T.define_local ~size ~dtype:ptr_dt in
            let idx = T.index ~ptr:buf ~idxs:rngs
              ~dtype:(D.Ptr ptr_dt) () in
            let ended = T.end_
              ~value:(T.store ~dst:idx ~value:src)
              ~ranges:range_nodes in
            let bar = T.barrier in
            let ended_bar = T.end_ ~value:bar ~ranges:[ended] in
            Some (T.after ~src:buf ~deps:[ended_bar])
        | _ -> None)
  | _ -> None

(* Split into kernels *)

(* Per-kernel context accumulated during the local graph rewrite that
   converts a STORE/END subtree into a CALL(kernel SINK). *)
type split_context = {
  mutable slot : int;
  buf_map : (int, T.t) Hashtbl.t;
  vars : (int, T.t) Hashtbl.t;
  mutable range_ctr : int;
  mutable opts : K.Opt.t list option;
  renumbered : (int, unit) Hashtbl.t;
  buf_shapes : (int, int list) Hashtbl.t;
}

let create_split_context () = {
  slot = 0;
  buf_map = Hashtbl.create 16;
  vars = Hashtbl.create 4;
  range_ctr = 0;
  opts = None;
  renumbered = Hashtbl.create 16;
  buf_shapes = Hashtbl.create 16;
}

(* Convert BUFFER/PARAM to a kernel PARAM with a slot index. The buffer
   is reshaped to its shape so downstream indexing sees the right layout. *)
let debuf ctx shapes n =
  let dtype = dtype_or_void n in
  let size = match T.view n with
    | Buffer { size; _ } -> size
    | _ -> (match shapes n with
      | Some dims -> List.fold_left ( * ) 1 dims
      | None -> 1)
  in
  let ptr_dt = D.Ptr.create (D.val_of dtype) ~addrspace:D.Global ~size in
  let slot = ctx.slot in
  ctx.slot <- ctx.slot + 1;
  let ret = T.param ~slot ~dtype:(D.Ptr ptr_dt) () in
  (* Use multi-dim shape from buf_shapes (precomputed from INDEX consumers)
     when available, falling back to shapes. *)
  let buf_shape = match Hashtbl.find_opt ctx.buf_shapes (T.tag n) with
    | Some s -> Some s
    | None -> shapes n in
  let ret = match buf_shape with
    | Some shape when shape <> [] ->
        T.reshape ~src:ret ~shape:(shape_node shape)
    | _ -> ret
  in
  (* XXX tinygrad distinguishes max_shape (static upper bound) from shape
     (possibly symbolic) and adds a shrink when they differ. *)
  if not (Hashtbl.mem ctx.buf_map (T.tag n)) then
    Hashtbl.replace ctx.buf_map (T.tag n) n;
  Some ret

(* Handle AFTER/MSTACK/MSELECT during kernel split: record the buffer
   mapping and return the buffer node so downstream sees BUFFER not the
   wrapper. Local-memory AFTERs are left in the kernel. *)
let handle_after ctx n =
  let v = T.view n in
  let is_local = match v with
    | After { dtype = D.Ptr p; _ } -> D.Ptr.addrspace p = D.Local
    | _ -> false in
  if is_local then None
  else
    let buf = match v with
      | After { src; _ } -> T.base src
      | Mstack _ | Mselect _ -> List.hd (T.children n)
      | _ -> n in
    let buf = match T.view buf with
      | Mstack _ | Mselect _ -> List.hd (T.children buf)
      | _ -> buf in
    assert (not (Hashtbl.mem ctx.buf_map (T.tag buf)));
    Hashtbl.replace ctx.buf_map (T.tag buf) n;
    Some buf

(* Cycle detection: verify each buffer is accessed through a single
   index path. Tinygrad compares idx.src[0].op (operation type); we
   compare node identity which is strictly more conservative — it may
   report cycles that tinygrad allows, but will never miss one. *)
let find_bufs n =
  let slice = T.toposort n
    ~gate:(fun x -> match T.view x with After _ -> false | _ -> true) in
  let read_from : (int, int) Hashtbl.t = Hashtbl.create 8 in
  List.iter (fun s ->
    match T.view s with
    | Index { ptr; _ } ->
        let buf = T.base ptr in
        (match T.view buf with
        | Buffer _ | Param _ ->
            let tag = T.tag ptr in
            (match Hashtbl.find_opt read_from (T.tag buf) with
            | Some prev when prev <> tag ->
                failwith "cycle detected while indexing buffer"
            | _ -> Hashtbl.replace read_from (T.tag buf) tag)
        | _ -> ())
    | _ -> ()) slice;
  None

(* Record a BIND node for the kernel argument list, pass through to the
   bound variable. *)
let unbind_kernel ctx n =
  Hashtbl.replace ctx.vars (T.tag n) n;
  match T.view n with Bind { var; _ } -> Some var | _ -> assert false

(* Renumber range axes starting from 0 so that kernel deduplication works
   regardless of the original axis numbering.  Tinygrad uses a tag field
   on UOps to avoid renumbering the replacement; we track the output
   nodes explicitly since the OCaml IR has no mutable tag. *)
let renumber_range ctx n =
  if Hashtbl.mem ctx.renumbered (T.tag n) then None
  else match T.view n with
  | Range { size; sub; kind; dtype; _ } ->
      let axis = ctx.range_ctr in
      ctx.range_ctr <- ctx.range_ctr + 1;
      let r = T.range ~size ~axis ~sub ~kind ~dtype () in
      Hashtbl.replace ctx.renumbered (T.tag r) ();
      Some r
  | _ -> assert false

(* Strip CONTIGUOUS, saving any Opt hints to ctx for KernelInfo. *)
let get_contiguous ctx n =
  match T.view n with
  | Contiguous { src; opts; _ } ->
      if opts <> [] then ctx.opts <- Some opts;
      Some src
  | _ -> assert false

(* Local rewrite for kernel split: convert BUFFER/PARAM to kernel PARAMs,
   record bindings, renumber ranges, strip CONTIGUOUS/NOOP/CONST srcs. *)
let to_define_global ctx shapes n =
  match T.view n with
  | Store _ -> find_bufs n
  | Buffer _ -> debuf ctx shapes n
  | Param { device = Some _; _ } -> debuf ctx shapes n
  | Bind _ -> unbind_kernel ctx n
  | After _ | Mstack _ | Mselect _ -> handle_after ctx n
  | Index { ptr; idxs = []; _ } ->
      (* INDEX(DEFINE_VAR) → DEFINE_VAR *)
      (match T.view ptr with Define_var _ -> Some ptr | _ -> None)
  | Bufferize { src; ranges; dtype; opts } ->
      (* Remove device from local BUFFERIZE *)
      if opts.device <> None then
        Some (T.bufferize ~src ~ranges ~dtype
                ~opts:{ opts with device = None })
      else None
  | Const _ ->
      (* Remove UNIQUE/DEVICE children to dedup constants *)
      if T.children n <> [] then Some (T.replace n ~children:[] ())
      else None
  | Range _ -> renumber_range ctx n
  | Contiguous _ -> get_contiguous ctx n
  | Noop { src = Some s; _ } -> Some s
  | Noop { src = None; _ } -> None
  (* XXX tinygrad's rangeify_codegen has AFTER.broadcast and AFTER.gep
     rules for DEFINE_LOCAL vectorized access. These fire when local
     buffers use vector dtypes. Not yet needed — add when vector-typed
     DEFINE_LOCAL is emitted. *)
  | _ -> None

(* Linearize multi-dim kernel index expressions into a single flat offset.
   Iterate dims right-to-left, accumulating stride, building sum(acc * src).
   The expression structure must match tinygrad's so that range renumbering
   in the split_store local rewrite assigns axis numbers in the same order. *)
let linearize_idxs idxs =
  if List.length idxs <= 1 then idxs
  else
    let dim_sizes = List.map (fun idx ->
      match K.view idx with
      | Range { size; _ } -> K.const_arg size
      | _ -> None) idxs in
    if List.exists Option.is_none dim_sizes then idxs
    else
      let dims = List.map (fun s ->
        match s with Some (Const.Int n) -> Int64.to_int n | _ -> 0)
        dim_sizes in
      (* Right-to-left: accumulate stride, build terms — matching
         tinygrad's _apply_reshape. Then simplify to canonicalize,
         matching tinygrad's graph_rewrite(combined, symbolic+...). *)
      let acc = ref 1 in
      let terms = List.rev_map (fun (s, idx) ->
        let t = if !acc = 1 then idx
          else K.binary ~op:`Mul ~lhs:(K.const_int !acc) ~rhs:idx in
        acc := !acc * s; t)
        (List.rev (List.combine dims idxs)) in
      let flat = List.fold_left (fun a t -> K.binary ~op:`Add ~lhs:a ~rhs:t)
        (K.const_int 0) terms in
      let flat = K.graph_rewrite ~name:"linearize_simplify"
        (K.first_match [Symbolic.sym]) flat in
      [flat]

(* Convert a tensor subtree (after to_define_global) into kernel IR.
   Each tensor node maps 1:1 to its kernel equivalent.  Shaped_wmma
   should already be lowered by earliest_rewrites; hitting it here
   is a pipeline bug. *)
let tensor_subtree_to_kernel root =
  let slice = T.toposort root in
  let tbl : (int, K.t) Hashtbl.t = Hashtbl.create (List.length slice) in
  let lookup n = match Hashtbl.find_opt tbl (T.tag n) with
    | Some k -> k | None -> K.const (C.int D.Val.index 0) in
  let map ns = List.map lookup ns in
  List.iter (fun n ->
    let k = match T.view n with
      | Const { value; _ } -> K.const value
      | Range { size; axis; sub; kind; dtype } ->
          K.range ~size:(lookup size) ~axis ~sub ~kind
            ~dtype:(D.val_of dtype) ()
      | End { value; ranges } ->
          K.end_ ~value:(lookup value) ~ranges:(map ranges) ()
      | Index { ptr; idxs; gate; _ } ->
          K.index ~ptr:(lookup ptr) ~idxs:(map idxs)
            ?gate:(Option.map lookup gate) ~as_ptr:false ()
      | Store { dst; value } ->
          K.store ~dst:(lookup dst) ~value:(lookup value) ~ranges:[]
      | Reduce { src; ranges; op; dtype } ->
          K.reduce ~op ~src:(lookup src) ~ranges:(map ranges)
            ~dtype:(D.val_of dtype)
      | Unary { op; src; _ } -> K.unary ~op ~src:(lookup src)
      | Binary { op; lhs; rhs; _ } ->
          K.binary ~op ~lhs:(lookup lhs) ~rhs:(lookup rhs)
      | Ternary { op; a; b; c; _ } ->
          K.ternary ~op ~a:(lookup a) ~b:(lookup b) ~c:(lookup c)
      | Cast { src; dtype } -> K.cast ~src:(lookup src) ~dtype
      | Bitcast { src; dtype } ->
          K.bitcast ~src:(lookup src) ~dtype:(D.val_of dtype)
      | Vectorize { srcs; _ } -> K.vectorize ~srcs:(map srcs)
      | Define_var { name; lo; hi; dtype } ->
          K.define_var ~name ~lo ~hi ~dtype:(D.val_of dtype) ()
      | Define_local { size; dtype } -> K.define_local ~size ~dtype
      | Barrier -> K.barrier
      | Invalid_index _ -> K.invalid_index ()
      | Bufferize { src; ranges; dtype; opts } ->
          K.bufferize ~src:(lookup src) ~ranges:(map ranges)
            ~dtype:(D.Ptr.create (D.val_of dtype)
              ~addrspace:opts.addrspace ~size:1) ~opts
      | After { src; deps; _ } ->
          K.after ~src:(lookup src) ~deps:(map deps)
      | Sink { srcs; kernel_info } -> K.sink ?kernel_info (map srcs)
      | Noop { src = Some s; _ } -> lookup s
      | Noop { src = None; _ } -> K.const (C.int D.Val.index 0)
      | Param { slot; dtype; _ } ->
          let pt = match dtype with
            | D.Ptr p -> p
            | D.Val v -> D.Ptr.create v ~addrspace:D.Global ~size:1 in
          K.param ~idx:slot ~dtype:pt
      | Bind { var; _ } -> lookup var
      | Contiguous { src; _ } | Reshape { src; _ } | Expand { src; _ }
      | Pad { src; _ } | Shrink { src; _ } | Permute { src; _ }
      | Flip { src; _ } | Detach { src; _ } -> lookup src
      | Device _ | Unique _ | Lunique _ ->
          K.const (C.int D.Val.index 0)
      | v -> failwith (Format.asprintf
          "tensor_subtree_to_kernel: unexpected %a" T.pp_view v)
    in
    Hashtbl.replace tbl (T.tag n) k) slice;
  lookup root

(* Ranges reachable from [n] that are not closed by any END or consumed
   by a REDUCE in the subtree. Tinygrad's .ranges excludes ended and
   reduce-internal ranges. *)
let open_ranges n =
  let all = T.ranges n in
  let closed = List.concat_map (fun x ->
    match T.view x with
    | End { ranges; _ } -> ranges
    | Reduce { ranges; _ } -> ranges
    | _ -> [])
    (T.toposort n) in
  List.filter (fun r -> not (List.exists (fun e -> e == r) closed)) all

(* Convert a STORE/END subtree into a CALL(kernel SINK, bufs, vars). *)
let split_store shapes n =
  match T.view n with
  | Store _ | End _ ->
      (* Don't split if there are open ranges *)
      if open_ranges n <> [] then None
      (* Raw shaped STORE should be processed through its END wrapper *)
      else if (match T.view n with
        | Store { dst; _ } -> shapes dst <> None
        | _ -> false)
      then None
      else
        let ctx = create_split_context () in
        (* Precompute multi-dim shapes for Buffer nodes from their INDEX
           consumers. *)
        List.iter (fun nd -> match T.view nd with
          | Index { ptr; idxs; _ } when List.length idxs > 1 ->
            (match T.view ptr with
             | Buffer _ ->
               let dims = List.filter_map (fun r -> match T.view r with
                 | Range { size; _ } -> (match T.view size with
                   | Const { value; _ } -> (match C.view value with
                     | Int n -> Some (Int64.to_int n) | _ -> None)
                   | _ -> None)
                 | _ -> None) idxs in
               if List.length dims = List.length idxs then
                 Hashtbl.replace ctx.buf_shapes (T.tag ptr) dims
             | _ -> ())
          | _ -> ()) (T.toposort n);
        (* Flatten range: toposort-reorder range children of End/Store.
           Tensor-level equivalent of Simplify.flatten_range. *)
        let flatten_range_t n =
          match T.view n with
          | End { value; ranges } when ranges <> [] ->
              let new_rngs = List.filter (fun r ->
                match T.view r with Range _ -> true | _ -> false)
                (T.toposort (T.sink ranges)) in
              if List.equal (==) ranges new_rngs then None
              else Some (T.end_ ~value ~ranges:new_rngs)
          | _ -> None
        in
        (* Use on-the-fly shape computation for the local rewrite,
           since debuf creates new RESHAPE nodes not in the precomputed
           shapes table. *)
        let local_shapes n = T.compute_shapes (T.sink [n]) n in
        let rewrite = T.first_match [
          to_define_global ctx shapes;
          flatten_range_t;
          mop_through_index local_shapes;
        ] in
        let ret = T.graph_rewrite ~name:"kernel_split" rewrite n in
        (* Determine callee type based on the stored value.
           If the END already wraps a CALL (the inner STORE was split
           first in the bottom-up pass), nothing more to do. *)
        let stored = match T.view ret with
          | Store { value; _ } -> Some value
          | End { value; _ } ->
              (match T.view value with
              | Store { value; _ } -> Some value
              | Call _ -> None
              | _ -> failwith "split_store: END wraps non-STORE")
          | Call _ -> None
          | _ -> failwith "split_store: unexpected result" in
        begin match stored with
        | None -> None
        | Some stored ->
        let bufs = Hashtbl.fold (fun _ v acc -> v :: acc)
          ctx.buf_map [] in
        let vars = Hashtbl.fold (fun _ v acc -> v :: acc)
          ctx.vars [] in
        let info : T.call_info = {
          grad_fxn = None; metadata = [];
          name = None; precompile = false } in
        let dtype = match T.dtype n with
          | Some d -> d | None -> D.void in
        (* COPY/BUFFER_VIEW are cross-device ops — keep as Ref *)
        let callee : T.callee = match T.view stored with
          | Copy _ | Buffer_view _ ->
              let ended = match T.view ret with
                | End { ranges; _ } -> ranges | _ -> [] in
              Ref (T.replace stored
                ~children:(T.children stored @ ended) ())
          | _ ->
              (* Normal kernel: convert tensor subtree to kernel IR *)
              let kernel_sink = T.sink
                ~kernel_info:{ K.name = ""; axis_kinds = [];
                  dont_use_locals = false; applied_opts = [];
                  opts_to_apply = ctx.opts; estimates = None }
                [ret] in
              Ast (tensor_subtree_to_kernel kernel_sink)
        in
        Some (T.call ~callee ~args:(bufs @ vars) ~info ~dtype)
        end
  | _ -> None

(* WAR dependency fixup *)

(* If kernel U reads buffer S, and S is also written by another kernel,
   S's write must complete before U runs. Add explicit ordering deps. *)
let fix_war_deps root =
  let nodes = T.toposort root in
  let afters = List.filter (fun n ->
    match T.view n with After _ -> true | _ -> false) nodes in
  if afters = [] then root
  else
    let buf_of n = match T.view n with
      | After { src; _ } -> T.base src | _ -> n in
    let kernel_assign : (int, T.t) Hashtbl.t = Hashtbl.create 16 in
    List.iter (fun u ->
      Hashtbl.replace kernel_assign (T.tag (buf_of u)) u) afters;
    let call_of u = match T.view u with
      | After { deps; _ } ->
          List.find_opt (fun d ->
            match T.view d with Call _ -> true | _ -> false) deps
      | _ -> None in
    let assign_rep : (int, T.t list) Hashtbl.t = Hashtbl.create 16 in
    List.iter (fun u ->
      let u_buf = buf_of u in
      let reads = match call_of u with
        | Some call -> (match T.view call with
          | Call { args; _ } ->
              List.filter (fun a -> match T.view a with
                | Buffer _ | Param _ -> true | _ -> false) args
          | _ -> [])
        | None -> [] in
      List.iter (fun s ->
        if s != u_buf then
          match Hashtbl.find_opt kernel_assign (T.tag s) with
          | Some a ->
              if call_of a <> None && call_of a = call_of u then ()
              else begin
                let prev = match Hashtbl.find_opt assign_rep (T.tag a) with
                  | Some l -> l | None -> [] in
                if not (List.exists (fun p -> p == u) prev) then
                  Hashtbl.replace assign_rep (T.tag a) (u :: prev)
              end
          | None -> ()) reads) afters;
    if Hashtbl.length assign_rep = 0 then root
    else
      T.graph_rewrite ~name:"fix_war_deps" (fun n ->
        match Hashtbl.find_opt assign_rep (T.tag n) with
        | Some extra_deps -> (match T.view n with
          | After { src; deps; _ } ->
              Some (T.after ~src ~deps:(deps @ extra_deps))
          | _ -> None)
        | None -> None) root

(* Main pipeline *)

let get_kernel_graph (root : T.t) : T.t =
  let shapes = T.compute_shapes root in
  let devices = T.compute_devices root in
  (* 1. multi_pm *)
  let root =
    T.graph_rewrite ~name:"multi_pm"
      (Multi.multi_pm ~shapes ~devices) root in
  let shapes = T.compute_shapes root in
  let devices = T.compute_devices root in
  (* 2. fold moved AFTERs (openpilot hack) *)
  let root =
    if Helpers.Context_var.get openpilot_hacks_var = 0 then root
    else
      let ctx : (int, T.t) Hashtbl.t = Hashtbl.create 16 in
      T.graph_rewrite ~name:"fold_moved_after" (fun n ->
        match T.view n with
        | After { deps; _ } ->
            let store = List.find_opt (fun d ->
              match T.view d with Store _ -> true | _ -> false) deps in
            (match store with
            | Some s ->
                let value = match T.view s with
                  | Store { value; _ } -> value | _ -> assert false in
                let after = n in
                (match T.view value with
                | Reshape _ | Expand _ | Pad _ | Shrink _ | Permute _
                | Flip _ | Cast _ | Ternary { op = `Where; _ } ->
                    found_after ctx ~after ~value; None
                | _ -> None)
            | None -> None)
        | Unary _ | Binary _ | Ternary _ | Cast _ | Bitcast _ ->
            let children = T.children n in
            let new_children = List.map (fun s ->
              match Hashtbl.find_opt ctx (T.tag s) with
              | Some after -> after | None -> s) children in
            if List.for_all2 (==) children new_children then None
            else Some (T.replace n ~children:new_children ())
        | _ -> None) root
  in
  (* 3. earliest_rewrites (syntactic sugar + mops + canonicalization) *)
  let root =
    T.graph_rewrite ~name:"earliest_rewrites" (T.first_match [
      index_concat;
      early_rangeify;
      mop_through_index shapes;
      mop_past_after shapes;
      mop_past_end;
      (* Merge adjacent reshapes *)
      (fun n -> match T.view n with
        | Reshape { src; shape; _ } ->
            (match T.view src with
            | Reshape { src = inner; _ } ->
                Some (T.reshape ~src:inner ~shape)
            | _ -> None)
        | _ -> None);
      resolve_call;
      (* Resolve allreduce *)
      (fun n -> match T.view n with
        | Allreduce { src = buf; device; op; dtype } ->
            let shape = match shapes n with Some s -> s | None -> [] in
            Allreduce.create_allreduce_function buf ~op ~device ~dtype
              ~shape ()
        | _ -> None);
      split_reduceop shapes;
      (* Remove DETACH/CONTIGUOUS_BACKWARD *)
      (fun n -> match T.view n with
        | Detach { src; _ } | Contiguous_backward { src; _ } -> Some src
        | _ -> None);
      (* COPY size mismatch: wrap in contiguous if movement ops changed size *)
      (fun n -> match T.view n with
        | Copy { src; _ } when is_movement (T.view src) ->
            let base_shape = shapes (T.base src) in
            let src_shape = shapes src in
            if base_shape <> src_shape then
              Some (T.replace n
                ~children:(T.contiguous ~src () :: List.tl (T.children n)) ())
            else None
        | _ -> None);
      (* Same-device COPY → NOOP *)
      (fun n -> match T.view n with
        | Copy { src; _ } ->
            (match devices src, devices n with
            | Some d1, Some d2 when d1 = d2 ->
                Some (T.noop ~src ~dtype:(dtype_or_void src) ())
            | _ -> None)
        | _ -> None);
      (* Assign rules (AFTER+STORE) *)
      (fun n -> match T.view n with
        | After { src = buf; deps; _ } ->
            let store = List.find_opt (fun d ->
              match T.view d with Store _ -> true | _ -> false) deps in
            (match store with
            | Some s ->
                let target, value = match T.view s with
                  | Store { dst; value } -> dst, value
                  | _ -> assert false in
                (* Bitcast on target → move to value *)
                (match T.view target with
                | Bitcast { src = inner; _ } ->
                    Some (T.after ~src:inner
                      ~deps:[T.store ~dst:inner
                        ~value:(T.bitcast ~src:value
                          ~dtype:(dtype_or_void inner))])
                | _ ->
                    (* View shape mismatch → wrap in inner AFTER *)
                    let target_shape = shapes target in
                    let buf_shape = shapes n in
                    if target_shape <> buf_shape
                       && target_shape <> None && buf_shape <> None then
                      let inner = T.after ~src:target ~deps:[s] in
                      let extras = List.filter (fun d -> d != s) deps in
                      Some (T.after ~src:buf ~deps:(inner :: extras))
                    else
                      match fix_store_after_hazard ~buf ~target ~value with
                      | Some _ as r -> r
                      | None ->
                        match T.view target with
                        | After _ ->
                            Some (normalize_store_after_target_chain
                              ~target ~value)
                        | _ -> None)
            | None -> None)
        | _ -> None);
      (* Size-0 reduce → identity element *)
      (fun n -> match T.view n with
        | Reduce_axis { src; op; dtype; _ }
          when (match shapes src with
            | Some s -> shape_prod s = 0 | None -> false)
            && (match shapes n with
              | Some s -> shape_prod s > 0 | None -> false) ->
            Some (T.const (C.identity_element op (D.val_of dtype)) dtype)
        | _ -> None);
      (* Size-0 → zero *)
      (fun n -> match T.view n with
        | Sink _ -> None
        | _ when (match shapes n with
            | Some s -> shape_prod s = 0 | None -> false) ->
            let dt = dtype_or_void n in
            Some (T.const (C.zero (D.val_of dt)) dt)
        | _ -> None);
    ]) root in
  let shapes = T.compute_shapes root in
  let devices = T.compute_devices root in
  (* 4. run_rangeify *)
  let ctx = Indexing.run_rangeify root ~shapes in
  (* 5. apply_rangeify *)
  let root = Indexing.apply_rangeify_pass ctx ~devices root in
  let shapes = T.compute_shapes root in
  (* 6. post-rangeify: buffer folding + buffer removal.
     Tinygrad also composes symbolic + pm_reduce_simplify here, but in
     our split IR, symbolic operates on Kernel.t and is applied during
     run_rangeify/apply_rangeify_pass via simplify_tensor_expr. *)
  let root =
    T.graph_rewrite ~name:"post_rangeify" (T.first_match [
      cleanup_dead_axes shapes;
      (* Remove noop bufferize *)
      (fun n -> match T.view n with
        | Bufferize { src; ranges; _ } ->
            (match T.view src with
            | Index { ptr; idxs; _ } ->
                remove_noop_bufferize ~idxs ~ranges ~ptr
                  ~buf_shape:(shapes n)
            | _ -> None)
        | _ -> None);
      (* No buffers for const *)
      (fun n -> match T.view n with
        | Bufferize { src; _ } ->
            (match T.view src with
            | Const { value; _ } ->
                Some (T.const value (dtype_or_void n))
            | _ -> None)
        | _ -> None);
      (* Indexing a const is a const *)
      (fun n -> match T.view n with
        | Index { ptr; _ } ->
            (match T.view ptr with Const _ -> Some ptr | _ -> None)
        | _ -> None);
      (* Copy on const is const *)
      (fun n -> match T.view n with
        | Copy { src; _ } ->
            (match T.view src with
            | Const { value; _ } ->
                Some (T.const value (dtype_or_void n))
            | _ -> None)
        | _ -> None);
      (* Noop on const *)
      (fun n -> match T.view n with
        | Noop { src = Some s; _ } ->
            (match T.view s with Const _ -> Some s | _ -> None)
        | _ -> None);
      (* MSTACK(CONST).INDEX → CONST *)
      (fun n -> match T.view n with
        | Index { ptr; _ } ->
            (match T.view ptr with
            | Mstack { srcs; _ } ->
                (match srcs with
                | s :: _ ->
                    let base = T.base s in
                    (match T.view base with
                    | Const { value; dtype; _ } ->
                        Some (T.const value dtype)
                    | _ -> None)
                | [] -> None)
            | _ -> None)
        | _ -> None);
      (* Remove bufferize with cost function *)
      (fun n -> match T.view n with
        | Index { ptr; idxs; _ } ->
            (match T.view ptr with
            | Bufferize { src; ranges; opts; _ } ->
                remove_bufferize ~src ~buf_ranges:ranges
                  ~buf_shape:(shapes ptr) ~idx_ranges:idxs
                  ~removable:opts.removable
            | _ -> None)
        | _ -> None);
    ]) root in
  (* 7. limit_bufs *)
  let root =
    let devices = T.compute_devices root in
    T.graph_rewrite ~name:"limit_bufs"
      (limit_bufs ctx devices) root in
  (* 8. add buffers (BUFFERIZE → STORE + BUFFER) *)
  let root =
    let devices = T.compute_devices root in
    let lunique_start =
      List.fold_left (fun acc x ->
        match T.view x with
        | Lunique { id; _ } -> max acc (id + 1)
        | _ -> acc) 0 (T.toposort root) in
    let counter = ref lunique_start in
    T.graph_rewrite ~name:"add_buffers" (T.first_match [
      mop_through_index shapes;
      mop_past_after shapes;
      mop_past_end;
      flatten_bufferize shapes;
      late_buffer_view devices;
      bufferize_to_store counter;
      (* Move RESHAPEs through MSELECT/MSTACK *)
      (fun n -> match T.view n with
        | Mselect _ | Mstack _ ->
            let children = T.children n in
            if List.for_all (fun c ->
              match T.view c with Reshape _ -> true | _ -> false)
              children
            then
              let unwrapped = List.map (fun c ->
                T.base (match T.view c with
                  | Reshape { src; _ } -> src | _ -> c)) children in
              let inner = T.replace n ~children:unwrapped () in
              let shape = match shapes n with
                | Some s -> s | None -> [] in
              if shape <> [] then
                Some (T.reshape ~src:inner ~shape:(shape_node shape))
              else Some inner
            else None
        | _ -> None);
      (* Remove RESHAPEs on CALL args *)
      (fun n -> match T.view n with
        | Call { callee; args; info; dtype } ->
            let new_args = List.map (fun a ->
              match T.view a with
              | Reshape { src; _ } -> src | _ -> a) args in
            if List.for_all2 (==) args new_args then None
            else Some (T.call ~callee ~args:new_args ~info ~dtype)
        | _ -> None);
      (* Remove MOP on AFTER deps, flatten nested AFTERs *)
      (fun n -> match T.view n with
        | After { src; deps; _ } ->
            let new_deps = List.map (fun d ->
              match T.view d with
              | Reshape { src; _ } | Expand { src; _ }
              | Permute { src; _ } | Flip { src; _ }
              | Pad { src; _ } | Shrink { src; _ } -> src
              | _ -> d) deps in
            let flat_deps = List.concat_map (fun d ->
              match T.view d with
              | After { deps; _ } -> deps
              | _ -> [d]) new_deps in
            if List.for_all2 (==) deps flat_deps then None
            else Some (T.after ~src ~deps:flat_deps)
        | _ -> None);
      (* Remove invalid writes *)
      (fun n -> match T.view n with
        | After { src; deps; _ } ->
            let real_deps = List.filter (fun d ->
              match T.view d with
              | Noop { src = None; _ } -> false
              | End { value; _ } ->
                  (match T.view value with
                  | Noop { src = None; _ } -> false
                  | _ -> true)
              | _ -> true) deps in
            if List.length real_deps < List.length deps then
              (match real_deps with
              | [] -> Some src
              | _ -> Some (T.after ~src ~deps:real_deps))
            else None
        | _ -> None);
    ]) root in
  (* 9. split kernels *)
  let shapes = T.compute_shapes root in
  let root =
    T.graph_rewrite ~name:"split_kernels" (split_store shapes) root in
  (* 10. WAR deps *)
  fix_war_deps root
