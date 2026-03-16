(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Late-stage Kernel IR transformations that lower high-level constructs into
   concrete operations suitable for rendering:
   - pm_reduce:       REDUCE -> accumulator loops (DEFINE_REG + Store + End)
   - pm_add_loads:    insert explicit LOAD for INDEX value consumers
   - pm_devectorize:  split wide vector ALU ops into scalar + VECTORIZE
   - pm_render:       final simplifications before rendering *)

open Tolk_ir
module K = Kernel

let prod lst = List.fold_left ( * ) 1 lst

(* Identity Elements *)

let identity_element (op : Op.reduce) (dt : Dtype.t) : Const.t =
  match op with
  | `Add -> if Dtype.is_float dt then Const.float dt 0.0 else Const.int dt 0
  | `Mul -> if Dtype.is_float dt then Const.float dt 1.0 else Const.int dt 1
  | `Max -> (
      match Dtype.min dt with
      | `Float f -> Const.float dt f
      | `SInt i -> Const.int64 dt i
      | `UInt i -> Const.int64 dt i
      | `Bool _ -> Const.bool false)

let reduce_op_to_binary : Op.reduce -> Op.binary = function
  | `Add -> `Add | `Mul -> `Mul | `Max -> `Max

(* Horizontal Reduction *)

let horizontal_reduce (inp : K.t) (out_dtype : Dtype.t) : K.t list =
  match K.dtype inp with
  | Some inp_dt when not (Dtype.equal inp_dt out_dtype) ->
      let amount = inp_dt.count / out_dtype.count in
      List.init amount (fun i ->
          K.gep_multi ~src:inp
            ~idxs:(List.init out_dtype.count (fun j -> i + (j * amount))))
  | _ -> [ inp ]

(* pm_reduce *)

type reduce_ctx = {
  range_to_ends : (K.t list, K.t list) Hashtbl.t;
}

let reduce_to_acc (ctx : reduce_ctx) (node : K.t) : K.t option =
  match K.view node with
  | Reduce { op; src = inp; ranges = reduce_range; dtype } ->
      let lst = horizontal_reduce inp dtype in
      let fold lst =
        match lst with
        | [] -> failwith "reduce_to_acc: empty horizontal reduce"
        | first :: rest ->
            List.fold_left
              (fun a x -> K.binary ~op:(reduce_op_to_binary op) ~lhs:a ~rhs:x)
              first rest
      in
      if reduce_range = [] then Some (fold lst)
      else begin
        let topo = K.toposort inp in
        let ended = K.Ref_tbl.create 16 in
        List.iter
          (fun n ->
            match K.view n with
            | End { ranges; _ } ->
                List.iter (fun r -> K.Ref_tbl.replace ended r ()) ranges
            | _ -> ())
          topo;
        let reduce_set = K.Ref_tbl.create 8 in
        List.iter (fun r -> K.Ref_tbl.replace reduce_set r ()) reduce_range;
        let input_ranges =
          List.filter
            (fun n ->
              (match K.view n with Range _ -> true | _ -> false)
              && not (K.Ref_tbl.mem reduce_set n)
              && not (K.Ref_tbl.mem ended n))
            topo
        in
        let scalar_dt = Dtype.scalar_of dtype in
        let identity = K.const (identity_element op scalar_dt) in
        let identity =
          if dtype.count > 1 then K.broadcast identity dtype.count else identity
        in
        let acc_pty = Dtype.Ptr.create dtype ~size:1 ~addrspace:Dtype.Reg () in
        let acc = K.define_reg ~size:1 ~dtype:acc_pty in
        let zero = K.const_int 0 in
        let idx ptr = K.index ~ptr ~idxs:[ zero ] () in
        let acc_after_input =
          match input_ranges with
          | [] -> acc
          | deps -> K.after ~src:acc ~deps
        in
        let acc_init = K.store ~dst:(idx acc_after_input) ~value:identity ~ranges:[] in
        let acc_in_loop = K.after ~src:acc ~deps:(acc_init :: reduce_range) in
        let ret = fold (K.load ~src:(idx acc_in_loop) () :: lst) in
        let store_back = K.store ~dst:(idx acc) ~value:ret ~ranges:[] in
        let end_node = K.end_ ~value:store_back ~ranges:reduce_range in
        let existing =
          match Hashtbl.find_opt ctx.range_to_ends reduce_range with
          | Some l -> l | None -> []
        in
        Hashtbl.replace ctx.range_to_ends reduce_range (existing @ [ end_node ]);
        Some (K.load ~src:(idx (K.after ~src:acc ~deps:[ end_node ])) ())
      end
  | _ -> None

(* Merge END nodes that share the same ranges into Group+End. *)
let merge_reduce_ends (ctx : reduce_ctx) (root : K.t) : K.t =
  let subs = K.Ref_tbl.create 8 in
  Hashtbl.iter
    (fun ranges ends ->
      if List.length ends > 1 then begin
        let stores =
          List.map
            (fun e -> match K.view e with End { value; _ } -> value
                      | _ -> failwith "merge_reduce_ends: expected End")
            ends
        in
        let merged = K.end_ ~value:(K.group stores) ~ranges in
        List.iter (fun old -> K.Ref_tbl.replace subs old merged) ends
      end)
    ctx.range_to_ends;
  if K.Ref_tbl.length subs = 0 then root
  else
    K.rebuild (fun node -> K.Ref_tbl.find_opt subs node) root

(* Tensor-core built-in accumulate: fold ADD into WMMA's c operand. *)
let wmma_accumulate (node : K.t) : K.t option =
  match K.view node with
  | Binary { op = `Add; lhs; rhs; _ } ->
      let try_fold wmma other =
        match K.view wmma with
        | Wmma { a; b; c; _ } ->
            let new_c = K.binary ~op:`Add ~lhs:c ~rhs:other in
            Some (K.replace wmma ~children:[ a; b; new_c ] ())
        | _ -> None
      in
      (match try_fold lhs rhs with
       | Some _ as r -> r
       | None -> try_fold rhs lhs)
  | _ -> None

let pm_reduce (root : K.t) : K.t =
  let ctx = { range_to_ends = Hashtbl.create 8 } in
  let rewrite = K.first_match [reduce_to_acc ctx; wmma_accumulate; Symbolic.gep_pushing] in
  merge_reduce_ends ctx (K.rewrite_fixpoint rewrite root)

(* pm_add_loads *)

let pm_add_loads (root : K.t) : K.t =
  let needs_load = K.Ref_tbl.create 16 in
  List.iter
    (fun node ->
      let value_refs =
        match K.view node with
        | Load { alt; _ } -> Option.to_list alt
        | Store { value; ranges; _ } -> value :: ranges
        | Index { idxs; gate; _ } -> idxs @ Option.to_list gate
        | _ -> K.children node
      in
      List.iter
        (fun child ->
          match K.view child with
          | Index _ -> K.Ref_tbl.replace needs_load child ()
          | _ -> ())
        value_refs)
    (K.toposort root);
  if K.Ref_tbl.length needs_load = 0 then root
  else
    let memo_ptr = K.Ref_tbl.create 128 in
    let memo_val = K.Ref_tbl.create 128 in
    let rewrite_children rewrite_v node =
      match K.view node with
      | Index { ptr; idxs; gate; _ } ->
          K.index ~ptr:(rewrite_v ~ptr:true ptr)
            ~idxs:(List.map (rewrite_v ~ptr:false) idxs)
            ?gate:(Option.map (rewrite_v ~ptr:false) gate) ()
      | Load { src; alt; _ } ->
          K.load ~src:(rewrite_v ~ptr:true src)
            ?alt:(Option.map (rewrite_v ~ptr:false) alt) ()
      | Store { dst; value; ranges } ->
          K.store ~dst:(rewrite_v ~ptr:true dst)
            ~value:(rewrite_v ~ptr:false value)
            ~ranges:(List.map (rewrite_v ~ptr:false) ranges)
      | _ ->
          K.replace node
            ~children:(List.map (rewrite_v ~ptr:false) (K.children node)) ()
    in
    let rec go ~ptr node =
      let memo = if ptr then memo_ptr else memo_val in
      match K.Ref_tbl.find_opt memo node with
      | Some r -> r
      | None ->
          let inner = rewrite_children go node in
          let result =
            if (not ptr) && K.Ref_tbl.mem needs_load node then
              match K.view inner with Index _ -> K.load ~src:inner () | _ -> inner
            else inner
          in
          K.Ref_tbl.replace memo node result;
          result
    in
    K.intern (go ~ptr:false root)

(* pm_devectorize *)

let gep_lane (s : K.t) (i : int) : K.t =
  match K.dtype s with
  | Some dt when dt.count > 1 -> K.gep ~src:s ~idx:i
  | _ -> s

let is_vectorizable (node : K.t) =
  K.is_alu node ||
  match K.view node with Cast _ | Bitcast _ -> true | _ -> false

let no_vectorized_alu (node : K.t) : K.t option =
  if not (is_vectorizable node) then None
  else
    match K.dtype node with
    | Some dt when dt.count > 1 ->
        (* Preserve WHERE(cond, val, Invalid_index) *)
        let skip =
          match K.view node with
          | Ternary { op = `Where; c; _ } -> (
              match K.view c with Invalid_index _ -> true | _ -> false)
          | _ -> false
        in
        if skip then None
        else
          let scalar_dt = Dtype.scalar_of dt in
          let children = K.children node in
          let srcs =
            List.init dt.count (fun i ->
                K.replace node
                  ~children:(List.map (fun s -> gep_lane s i) children)
                  ~dtype:scalar_dt ())
          in
          Some (K.vectorize ~srcs)
    | _ -> None

let no_vectorized_wmma (node : K.t) : K.t option =
  match K.view node with
  | Wmma { a; b; c; dtype; upcast_axes = upcast_a, upcast_b, upcast_c; _ } ->
      let out_sz = prod (List.map snd upcast_c) in
      if dtype.count <= out_sz then None
      else
        let scalar_dt = Dtype.scalar_of dtype in
        let chunked src axes =
          match K.dtype src with
          | None -> []
          | Some src_dt ->
              let sz = prod (List.map snd axes) in
              let groups = src_dt.count / sz in
              List.init groups (fun g ->
                  K.gep_multi ~src
                    ~idxs:(List.init sz (fun j -> (g * sz) + j)))
        in
        let ca = chunked a upcast_a and cb = chunked b upcast_b
        and cc = chunked c upcast_c in
        let n = List.length ca in
        if n = 0 || List.length cb <> n || List.length cc <> n then None
        else
          let wmma_dt = Dtype.vec scalar_dt out_sz in
          let wmmas =
            List.map2 (fun (a, b) c ->
                K.replace node ~children:[ a; b; c ] ~dtype:wmma_dt ())
              (List.combine ca cb) cc
          in
          let srcs =
            List.concat_map
              (fun w -> List.init out_sz (fun i -> K.gep ~src:w ~idx:i))
              wmmas
          in
          Some (K.vectorize ~srcs)
  | _ -> None

let scalarize_ptr_dtype (pty : Dtype.ptr) : Dtype.ptr =
  let lanes = max 1 pty.base.count in
  { pty with base = Dtype.scalar_of pty.base; size = pty.size * lanes }

let rec local_or_reg_base (node : K.t) : bool =
  match K.view node with
  | After { src; _ } | Cast { src; _ } | Bitcast { src; _ } ->
      local_or_reg_base src
  | Define_local _ | Define_reg _ -> true
  | _ -> false

let devec_buf mk size (dtype : Dtype.ptr) =
  let pty = scalarize_ptr_dtype dtype in
  Some (K.cast ~src:(mk ~size:(size * dtype.base.count) ~dtype:pty) ~dtype:dtype.base)

let no_vectorized_buf (node : K.t) : K.t option =
  match K.view node with
  | Define_local { size; dtype } when dtype.base.count > 1 ->
      devec_buf K.define_local size dtype
  | Define_reg { size; dtype } when dtype.base.count > 1 ->
      devec_buf K.define_reg size dtype
  | _ -> None

let no_vectorized_index (node : K.t) : K.t option =
  match K.view node with
  | Index { ptr; idxs; dtype; _ } when dtype.base.count > 1 ->
      if not (local_or_reg_base ptr) then None
      else
        let cnt = dtype.base.count in
        let idx_sum =
          match idxs with
          | [] -> K.const_int 0
          | first :: rest ->
              List.fold_left (fun a r -> K.binary ~op:`Add ~lhs:a ~rhs:r) first rest
        in
        let cnt_c = K.const (Const.int Dtype.index cnt) in
        let iota = K.vectorize ~srcs:(List.init cnt (fun i -> K.const_int i)) in
        let scaled =
          K.binary ~op:`Mul ~lhs:(K.broadcast idx_sum cnt)
            ~rhs:(K.broadcast cnt_c cnt)
        in
        Some (K.index ~ptr ~idxs:[ K.binary ~op:`Add ~lhs:scaled ~rhs:iota ] ())
  | _ -> None

let is_true_const (node : K.t) : bool =
  match K.view node with
  | Const { value; _ } -> (
      match Const.view value with Const.Bool true -> true | _ -> false)
  | _ -> false

let cast_after_after (node : K.t) : K.t option =
  match K.view node with
  | Cast { src; dtype } -> (
      match K.view src with
      | After { src = inner; deps } ->
          Some (K.after ~src:(K.cast ~src:inner ~dtype) ~deps)
      | _ -> None)
  | _ -> None

(* Drop trivially-true gates from INDEX. Placed here (not pm_render)
   because it must run before masked_load_alt / where_after_gated_load. *)
let drop_true_gate (node : K.t) : K.t option =
  match K.view node with
  | Index { ptr; idxs; gate = Some g; _ } when is_true_const g ->
      Some (K.index ~ptr ~idxs ())
  | _ -> None

let pm_devectorize (root : K.t) : K.t =
  K.rewrite_fixpoint (K.first_match [
    cast_after_after; no_vectorized_buf; no_vectorized_index;
    no_vectorized_wmma; no_vectorized_alu; drop_true_gate;
  ]) root

(* pm_correct_load_store *)

(* Extract the pointer dtype from a load/store source, looking through
   Cast and Bitcast wrappers to find the underlying INDEX. *)
let rec load_store_ptr_dtype (node : K.t) : Dtype.ptr option =
  match K.view node with
  | Index { dtype; _ } -> Some dtype
  | Cast { src; _ } | Bitcast { src; _ } -> load_store_ptr_dtype src
  | _ -> None

(* Split oversized LOAD/STORE into scalar operations when the backend does
   not support the full vector width.

   Uses K.replace to build per-lane Load/Store nodes with GEP'd pointer
   sources, bypassing the K.load smart constructor's pointer validation
   (GEP on a pointer-typed node produces a value-typed Gep, but the
   resulting scalar Load/Store is structurally valid for linearization).
   Currently splits to scalar only; intermediate-width splitting is deferred
   until the IR gains a pointer-narrowing mechanism. *)
let split_load_store (ren : Renderer.t) (node : K.t) : K.t option =
  let needs_split ptr_src count =
    if count <= 1 then false
    else
      match load_store_ptr_dtype ptr_src with
      | Some pty when pty.addrspace = Dtype.Reg -> false
      | Some pty ->
          let widths = Renderer.load_store_widths ren (Dtype.scalar_of pty.base) in
          not (List.exists (fun w -> w >= count) widths)
      | None -> false
  in
  match K.view node with
  | Load { src; alt; dtype } when dtype.count > 1 && needs_split src dtype.count ->
      let scalar_dt = Dtype.scalar_of dtype in
      let srcs =
        List.init dtype.count (fun i ->
            let gep_ptr = K.gep ~src ~idx:i in
            let gep_alt = Option.map (fun a -> K.gep ~src:a ~idx:i) alt in
            K.replace node
              ~children:(gep_ptr :: Option.to_list gep_alt)
              ~dtype:scalar_dt ())
      in
      Some (K.vectorize ~srcs)
  | Store { dst; value; ranges } -> (
      match K.dtype value with
      | Some dt when dt.count > 1 && needs_split dst dt.count ->
          let stores =
            List.init dt.count (fun i ->
                K.replace node
                  ~children:
                    ((K.gep ~src:dst ~idx:i) :: (K.gep ~src:value ~idx:i)
                   :: ranges)
                  ())
          in
          Some (K.group stores)
      | _ -> None)
  | _ -> None

let pm_correct_load_store (ren : Renderer.t) (root : K.t) : K.t =
  K.rewrite_fixpoint (split_load_store ren) root

(* pm_render *)

let expand_vector_const (node : K.t) : K.t option =
  match K.view node with
  | Const { value; dtype } when dtype.count > 1 ->
      let c = K.const value in
      Some (K.vectorize ~srcs:(List.init dtype.count (fun _ -> c)))
  | _ -> None

let trivial_gep (node : K.t) : K.t option =
  match K.view node with
  | Gep { src; idx = 0; _ } -> (
      match K.dtype src with Some dt when dt.count = 1 -> Some src | _ -> None)
  | _ -> None

let lower_cat (node : K.t) : K.t option =
  match K.view node with
  | Cat { srcs; _ } ->
      Some
        (K.vectorize
           ~srcs:
             (List.concat_map
                (fun src ->
                  match K.dtype src with
                  | Some dt when dt.count > 1 ->
                      List.init dt.count (fun i -> K.gep ~src ~idx:i)
                  | _ -> [ src ])
                srcs))
  | _ -> None

let trivial_vectorize (node : K.t) : K.t option =
  match K.view node with
  | Vectorize { srcs = [ src ]; _ } -> Some src
  | _ -> None

let masked_load_alt (node : K.t) : K.t option =
  match K.view node with
  | Load { src; alt = None; _ } ->
      let rec has_gate r =
        match K.view r with
        | Index { gate = Some _; _ } -> true
        | Cast { src; _ } | Bitcast { src; _ } -> has_gate src
        | _ -> false
      in
      if not (has_gate src) then None
      else Some (K.load ~src ~alt:(K.zero_like node) ())
  | _ -> None

(* WHERE-after-gated-Load folding *)

type cast_wrapper = No_wrap | Wrap_cast of Dtype.t | Wrap_bitcast of Dtype.t
type gate_relation = Same_gate | Negated_gate | Different_gate

let peel_wrapper (node : K.t) : cast_wrapper * K.t =
  match K.view node with
  | Cast { src; dtype } -> (Wrap_cast dtype, src)
  | Bitcast { src; dtype } -> (Wrap_bitcast dtype, src)
  | _ -> (No_wrap, node)

let apply_wrapper w src =
  match w with
  | No_wrap -> src
  | Wrap_cast dtype -> K.cast ~src ~dtype
  | Wrap_bitcast dtype -> K.bitcast ~src ~dtype

let gate_relation ~cond ~gate =
  if cond == gate then Same_gate
  else
    match K.view gate with
    | Binary { op = `Xor; lhs; rhs; dtype }
      when dtype.scalar = Dtype.Bool && dtype.count = 1 ->
        if (lhs == cond && is_true_const rhs)
           || (rhs == cond && is_true_const lhs)
        then Negated_gate
        else Different_gate
    | _ -> Different_gate

let rec find_index_gate (node : K.t) : K.t option =
  match K.view node with
  | Index { gate = Some g; _ } -> Some g
  | Cast { src; _ } | Bitcast { src; _ } -> find_index_gate src
  | _ -> None

let coerce_alt ~load_dtype alt =
  match K.dtype alt with
  | Some dt when Dtype.equal dt load_dtype -> alt
  | _ -> (
      match K.view alt with
      | Cast { src; _ } -> (
          match K.dtype src with
          | Some dt when Dtype.equal dt load_dtype -> src
          | _ -> K.cast ~src:alt ~dtype:load_dtype)
      | _ -> K.cast ~src:alt ~dtype:load_dtype)

let where_after_gated_load (node : K.t) : K.t option =
  match K.view node with
  | Ternary { op = `Where; a = cond; b = true_side; c = false_side; _ } ->
      let try_fold ~load_side ~other_side ~require_negated =
        let wrapper, load_node = peel_wrapper load_side in
        match K.view load_node with
        | Load { src; dtype = load_dtype; _ } -> (
            match find_index_gate src with
            | None -> None
            | Some gate ->
                let matches = match gate_relation ~cond ~gate with
                  | Same_gate -> not require_negated
                  | Negated_gate -> require_negated
                  | Different_gate -> false
                in
                if matches then
                  let alt = coerce_alt ~load_dtype other_side in
                  Some (apply_wrapper wrapper (K.load ~src ~alt ()))
                else None)
        | _ -> None
      in
      (match
         try_fold ~load_side:true_side ~other_side:false_side
           ~require_negated:false
       with
      | Some _ as r -> r
      | None ->
          try_fold ~load_side:false_side ~other_side:true_side
            ~require_negated:true)
  | _ -> None

let pm_render (root : K.t) : K.t =
  K.rewrite_fixpoint (K.first_match [
    expand_vector_const; trivial_gep; lower_cat;
    trivial_vectorize; where_after_gated_load; masked_load_alt;
  ]) root
