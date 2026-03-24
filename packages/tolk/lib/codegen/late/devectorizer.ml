(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_ir
module K = Kernel

let prod lst = List.fold_left ( * ) 1 lst

(* Reduce lowering *)

let identity_element = Const.identity_element

let reduce_op_to_binary : Op.reduce -> Op.binary = function
  | `Add -> `Add | `Mul -> `Mul | `Max -> `Max

let horizontal_reduce (inp : K.t) (out_dtype : Dtype.t) : K.t list =
  match K.dtype inp with
  | Some inp_dt when not (Dtype.equal inp_dt out_dtype) ->
      let amount = Dtype.count inp_dt / Dtype.count out_dtype in
      List.init amount (fun i ->
          K.gep_multi ~src:inp
            ~idxs:(List.init (Dtype.count out_dtype) (fun j -> i + (j * amount))))
  | _ -> [ inp ]

type reduce_ctx = {
  mutable acc_num : int;
}

(* Lower Reduce into an explicit register accumulator.

   For rangeless reduces, just fold the horizontal lanes.  Otherwise build:

     acc = DEFINE_REG
     acc_after_input = AFTER(acc, <outer ranges that feed inp>)
     STORE(INDEX(acc_after_input), identity)          -- init
     acc_in_loop = AFTER(acc, [init; reduce_range])   -- visible inside loop
     ret = fold(LOAD(acc_in_loop) :: horizontal_lanes)
     STORE(INDEX(acc), ret)                           -- write-back
     END(store_back, reduce_range, "mergeable")
     result = LOAD(AFTER(acc, [end]))                 -- read after loop

   The After nodes encode scheduling: acc_after_input delays init until outer
   ranges are live; acc_in_loop gates the in-body read on both init and the
   reduce range; the final AFTER(acc, [end]) reads the result after the loop
   closes. *)
let reduce_to_acc (ctx : reduce_ctx) (node : K.t) : K.t option =
  match K.view node with
  | Reduce { op; src = inp; ranges = reduce_range; dtype } ->
      let lst = horizontal_reduce inp dtype in
      let fold = function
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
        List.iter (fun n ->
          match K.view n with
          | End { ranges; _ } ->
              List.iter (fun r -> K.Ref_tbl.replace ended r ()) ranges
          | _ -> ()) topo;
        let reduce_set = K.Ref_tbl.create 8 in
        List.iter (fun r -> K.Ref_tbl.replace reduce_set r ()) reduce_range;
        let input_ranges =
          List.filter (fun n ->
            K.is_range n
            && not (K.Ref_tbl.mem reduce_set n)
            && not (K.Ref_tbl.mem ended n)) topo
        in
        let scalar_dt = Dtype.scalar_of dtype in
        let identity = K.const (identity_element op scalar_dt) in
        let identity =
          if Dtype.count dtype > 1 then K.broadcast identity (Dtype.count dtype) else identity
        in
        let acc_pty = Dtype.ptr_of dtype ~addrspace:Dtype.Reg ~size:1 in
        let slot = ctx.acc_num in
        ctx.acc_num <- ctx.acc_num + 1;
        let acc = K.define_reg ~size:1 ~dtype:acc_pty ~slot in
        let zero = K.const_int 0 in
        let idx ptr = K.index ~ptr ~idxs:[ zero ] ~as_ptr:false () in
        let acc_after_input = match input_ranges with
          | [] -> acc | deps -> K.after ~src:acc ~deps in
        let acc_init = K.store ~dst:(idx acc_after_input) ~value:identity ~ranges:[] in
        let acc_in_loop = K.after ~src:acc ~deps:(acc_init :: reduce_range) in
        let ret = fold (idx acc_in_loop :: lst) in
        let store_back = K.store ~dst:(idx acc) ~value:ret ~ranges:[] in
        let end_node = K.end_ ~value:store_back ~ranges:reduce_range ~tag:"mergeable" () in
        Some (idx (K.after ~src:acc ~deps:[ end_node ]))
      end
  | _ -> None

let merge_reduce_ends_rule (_ctx : reduce_ctx) (node : K.t) : K.t option =
  match K.view node with
  | Sink _ ->
      let by_ranges : (K.t list, K.t list) Hashtbl.t = Hashtbl.create 8 in
      List.iter (fun n ->
        match K.view n with
        | End { ranges; _ } when K.tag n = Some "mergeable" ->
            let existing =
              Option.value ~default:[] (Hashtbl.find_opt by_ranges ranges)
            in
            Hashtbl.replace by_ranges ranges (existing @ [ n ])
        | _ -> ()) (K.toposort node);
      let needs_merge =
        Hashtbl.fold (fun _ ends acc -> acc || List.length ends > 1) by_ranges false
      in
      if not needs_merge then None
      else begin
        let mappings = ref [] in
        Hashtbl.iter (fun ranges ends ->
          if List.length ends > 1 then begin
            let stores = List.map (fun e ->
              match K.view e with
              | End { value; _ } -> value
              | _ -> failwith "merge_reduce_ends: expected End") ends
            in
            let merged = K.end_ ~value:(K.group stores) ~ranges () in
            List.iter (fun old -> mappings := (old, merged) :: !mappings) ends
          end) by_ranges;
        match !mappings with
        | [] -> None
        | mappings -> Some (K.substitute mappings node)
      end
  | _ -> None

let wmma_accumulate (node : K.t) : K.t option =
  match K.view node with
  | Binary { op = `Add; lhs; rhs; _ } ->
      let try_fold wmma other =
        match K.view wmma with
        | Wmma { a; b; c; _ } ->
            Some (K.replace wmma ~children:[ a; b; K.O.( + ) c other ] ())
        | _ -> None
      in
      (match try_fold lhs rhs with
       | Some _ as r -> r | None -> try_fold rhs lhs)
  | _ -> None

let pm_reduce (root : K.t) : K.t =
  let ctx = { acc_num = 0 } in
  let rewrite = K.first_match [
    reduce_to_acc ctx;
    wmma_accumulate;
    Symbolic.gep_pushing;
    merge_reduce_ends_rule ctx;
  ] in
  K.graph_rewrite ~name:"remove_reduce" rewrite root

(* Load insertion *)

(* Inserts Load nodes for value-typed Index references and collapses
   Store(Load(x), v) into Store(x, v). *)

let add_loads_rule (node : K.t) : K.t option =
  match K.view node with
  | Index { dtype = Dtype.T _; ptr; idxs; gate; _ } ->
      (* Non-ptr Index: retype to the buffer's ptr dtype, then wrap with Load.
         The ptr dtype is recovered from the buffer definition (Param, Define_local,
         etc.); when the ptr is a Vectorize (after do_expand) we walk into the
         first element to find the underlying buffer. *)
      let rec find_buf_ptr_dtype node =
        match K.view node with
        | Param { dtype = pty; _ } | Param_image { dtype = pty; _ }
        | Define_local { dtype = pty; _ } | Define_reg { dtype = pty; _ }
        | Bufferize { dtype = pty; _ } -> Some pty
        | Vectorize { srcs = s :: _; _ } | Cat { srcs = s :: _; _ } ->
            find_buf_ptr_dtype s
        | After { src; _ } | Cast { src; _ } | Bitcast { src; _ } ->
            find_buf_ptr_dtype src
        | _ -> None
      in
      let ptr_pty = match find_buf_ptr_dtype ptr with
        | Some pty -> pty
        | None ->
            (* Fallback: shouldn't happen in well-formed graphs *)
            failwith "add_loads_rule: cannot find buffer ptr dtype"
      in
      let ptr_idx = K.index_raw ~ptr ~idxs ?gate ~dtype:(Dtype.ptr_to_any ptr_pty) () in
      Some (K.load ~src:ptr_idx ())
  | Index { dtype = Dtype.P _; _ } -> None
  | Store { dst; value; ranges } -> begin
      match K.view dst with
      | Load { src; _ } -> Some (K.store ~dst:src ~value ~ranges)
      | _ -> None end
  | _ -> None

let pm_add_loads (root : K.t) : K.t =
  K.graph_rewrite ~name:"** add loads (code)" (K.first_match [ add_loads_rule ]) root

(* Devectorization *)

let gep_lane (s : K.t) (i : int) : K.t =
  match K.dtype s with
  | Some dt when Dtype.count dt > 1 -> K.gep ~src:s ~idx:i
  | _ -> s

let is_vectorizable (node : K.t) =
  K.is_alu node ||
  match K.view node with Cast _ | Bitcast _ -> true | _ -> false

let no_vectorized_alu (node : K.t) : K.t option =
  if not (is_vectorizable node) then None
  else
    (* Skip ptr-typed Casts — their vcount is the ptr vector width (always 1),
       not the element count. *)
    match K.view node with
    | Cast { dtype = Dtype.P _; _ } -> None
    | _ ->
    match K.dtype node with
    | Some dt when Dtype.count dt > 1 -> begin
        match K.view node with
        | Ternary { op = `Where; c; _ }
          when (match K.view c with Invalid_index _ -> true | _ -> false) ->
            None
        | _ ->
            let scalar_dt = Dtype.scalar_of dt in
            let children = K.children node in
            let srcs =
              List.init (Dtype.count dt) (fun i ->
                  K.replace node
                    ~children:(List.map (fun s -> gep_lane s i) children)
                    ~dtype:scalar_dt ())
            in
            Some (K.vectorize ~srcs)
      end
    | _ -> None

let no_vectorized_wmma (node : K.t) : K.t option =
  match K.view node with
  | Wmma { a; b; c; dtype; upcast_axes = upcast_a, upcast_b, upcast_c; _ } ->
      let out_sz = prod (List.map snd upcast_c) in
      if Dtype.count dtype <= out_sz then None
      else
        let scalar_dt = Dtype.scalar_of dtype in
        let chunked src axes =
          match K.dtype src with
          | None -> []
          | Some src_dt ->
              let sz = prod (List.map snd axes) in
              let groups = Dtype.count src_dt / sz in
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
  let lanes = max 1 (Dtype.count (Dtype.base pty)) in
  Dtype.ptr_with_size (Dtype.ptr_with_base pty (Dtype.scalar_of (Dtype.base pty))) (Dtype.ptr_size pty * lanes)

let rec local_or_reg_base (node : K.t) : bool =
  match K.view node with
  | After { src; _ } | Cast { src; _ } | Bitcast { src; _ } ->
      local_or_reg_base src
  | Define_local _ | Define_reg _ -> true
  | _ -> false

let devec_buf mk size (dtype : Dtype.ptr) =
  let pty = scalarize_ptr_dtype dtype in
  Some (K.cast ~src:(mk ~size:(size * Dtype.count (Dtype.base dtype)) ~dtype:pty) ~dtype:(Dtype.ptr_to_any dtype))

let no_vectorized_buf (node : K.t) : K.t option =
  match K.view node with
  | Define_local { size; dtype } when Dtype.count (Dtype.base dtype) > 1 ->
      devec_buf K.define_local size dtype
  | Define_reg { size; dtype; slot } when Dtype.count (Dtype.base dtype) > 1 ->
      devec_buf (fun ~size ~dtype -> K.define_reg ~size ~dtype ~slot) size dtype
  | _ -> None

(* Scalarize a vector Index into local/reg memory by replacing
   Index(ptr, [i]) with Index(ptr, [i*count + <0,1,...,count-1>]),
   turning one wide access into count consecutive scalar slots. *)
let no_vectorized_index (node : K.t) : K.t option =
  match K.view node with
  | Index { ptr; idxs; dtype = Dtype.P pty; _ }
    when Dtype.count (Dtype.base pty) > 1 ->
      if not (local_or_reg_base ptr) then None
      else
        let cnt = Dtype.count (Dtype.base pty) in
        (* Idempotency guard: if idxs are already vectorized to cnt,
           this rewrite has already been applied. Prevents cycles in
           unified_rewrite where the result matches the same pattern. *)
        let already_vec = List.exists (fun idx ->
          Dtype.count (K.dtype_or Dtype.void idx) = cnt) idxs in
        if already_vec then None
        else
          let open K.O in
          let idx_sum =
            match idxs with
            | [] -> int_ 0
            | first :: rest -> List.fold_left ( + ) first rest
          in
          let iota = K.vectorize ~srcs:(List.init cnt (fun i -> int_ i)) in
          let scaled = K.broadcast idx_sum cnt * K.broadcast (int_ cnt) cnt in
          Some (K.index ~ptr ~idxs:[ scaled + iota ] ())
  | _ -> None

let is_true_const node =
  match K.view node with
  | Const { value; _ } ->
      (match Const.view value with Const.Bool true -> true | _ -> false)
  | _ -> false

let cast_after_after (node : K.t) : K.t option =
  match K.view node with
  | Cast { src; dtype } -> begin
      match K.view src with
      | After { src = inner; deps } ->
          Some (K.after ~src:(K.cast ~src:inner ~dtype) ~deps)
      | _ -> None end
  | _ -> None

let drop_true_gate (node : K.t) : K.t option =
  match K.view node with
  | Index { ptr; idxs; gate = Some g; _ } when is_true_const g ->
      Some (K.index ~ptr ~idxs ())
  | _ -> None

let pm_devectorize_rule =
  K.first_match [
    cast_after_after; no_vectorized_buf; no_vectorized_index;
    no_vectorized_wmma; no_vectorized_alu; drop_true_gate;
  ]

let pm_devectorize (root : K.t) : K.t =
  K.graph_rewrite pm_devectorize_rule root

(* Load/store width correction *)

(* Splits vectorized loads and stores into chunks whose widths the
   renderer supports. Matches Load/Store whose ptr src is a Cast(Index)
   or bare Index with a multi-element ptr dtype. Each chunk gets a fresh
   Index with a ptr-typed Cast to the appropriate vector width. *)

let split_load_store (ren : Renderer.t) (node : K.t) : K.t option =
  (* Extract (idx_node, element_count) from the ptr src of a Load/Store.
     src may be Cast(Index) (widened) or bare Index. *)
  let extract_idx_sz src =
    match K.view src with
    | Cast { src = inner; dtype = Dtype.P pty } -> begin
        match K.view inner with
        | Index _ -> Some (inner, Dtype.count (Dtype.base pty))
        | _ -> None end
    | Cast { src = inner; _ } -> begin
        match K.view inner with
        | Index { dtype = Dtype.P pty; _ } -> Some (inner, Dtype.count (Dtype.base pty))
        | _ -> None end
    | _ -> None
  in
  (* Walk through ptr to find the buffer's ptr dtype *)
  let rec find_buf_pty node =
    match K.view node with
    | Param { dtype; _ } | Define_local { dtype; _ }
    | Define_reg { dtype; _ } -> Some dtype
    | After { src; _ } | Cast { src; _ } | Bitcast { src; _ } -> find_buf_pty src
    | _ -> None
  in
  let get_buf_pty idx =
    match K.view idx with
    | Index { ptr; _ } -> find_buf_pty ptr
    | _ -> None
  in
  let process_load_or_store ~is_load ~ptr_src ~ls_dtype ~alt ~value ~ranges =
    match extract_idx_sz ptr_src with
    | None -> None
    | Some (_, sz) when sz <= 1 -> None
    | Some (idx, sz) ->
        match get_buf_pty idx with
        | None -> None
        | Some buf_pty when Dtype.addrspace buf_pty = Dtype.Reg -> None
        | Some buf_pty ->
            let base_scalar = Dtype.scalar_of (Dtype.base buf_pty) in
            let widths = Renderer.load_store_widths ren base_scalar in
            let lengths = widths @ [1] in
            let idx_ptr, idx_idxs, idx_gate = match K.view idx with
              | Index { ptr; idxs; gate; _ } -> (ptr, idxs, gate)
              | _ -> assert false
            in
            let global_offset = ref 0 in
            let ret = ref [] in
            while !global_offset < sz do
              match List.find_opt (fun fl ->
                !global_offset + fl <= sz) lengths with
              | None -> global_offset := sz (* shouldn't happen with 1 at end *)
              | Some fold_length ->
                  let new_idxs =
                    if !global_offset = 0 then idx_idxs
                    else List.map (fun i ->
                      K.binary ~op:`Add ~lhs:i
                        ~rhs:(K.const_int !global_offset)) idx_idxs
                  in
                  let base_idx =
                    K.index ~ptr:idx_ptr ~idxs:new_idxs ?gate:idx_gate ()
                  in
                  let lidx =
                    if fold_length > 1 then
                      let wide_pty = Dtype.ptr_of
                        (Dtype.vec base_scalar fold_length)
                        ~addrspace:(Dtype.addrspace buf_pty)
                        ~size:(Dtype.ptr_size buf_pty) in
                      K.cast ~src:base_idx ~dtype:(Dtype.ptr_to_any wide_pty)
                    else base_idx
                  in
                  if is_load then begin
                    let load_dt = Dtype.vec (Dtype.scalar_of ls_dtype) fold_length in
                    let children = lidx :: Option.to_list alt in
                    ret := K.replace node ~children ~dtype:load_dt () :: !ret
                  end else begin
                    let gep_idxs =
                      List.init fold_length (fun j -> !global_offset + j) in
                    let chunk_value = K.gep_multi ~src:value ~idxs:gep_idxs in
                    ret := K.store ~dst:lidx ~value:chunk_value ~ranges :: !ret
                  end;
                  global_offset := !global_offset + fold_length
            done;
            let ret = List.rev !ret in
            match ret with
            | [] | [_] -> None  (* no split needed *)
            | _ ->
                if is_load then Some (K.cat ~srcs:ret)
                else Some (K.group ret)
  in
  match K.view node with
  | Load { src; alt; dtype } ->
      process_load_or_store ~is_load:true ~ptr_src:src ~ls_dtype:dtype
        ~alt ~value:src (* unused for load *) ~ranges:[]
  | Store { dst; value; ranges } ->
      process_load_or_store ~is_load:false ~ptr_src:dst
        ~ls_dtype:Dtype.void ~alt:None ~value ~ranges
  | _ -> None

let pm_correct_load_store_rule ren = split_load_store ren

let pm_correct_load_store ren root = K.graph_rewrite (split_load_store ren) root

(* Load/store folding *)

let expand_index (node : K.t) : K.t option =
  match K.view node with
  | Index { ptr; idxs = [vec]; _ } -> (
      match K.view ptr with
      | Vectorize { srcs; _ } ->
          let count = Dtype.count (K.dtype_or Dtype.void vec) in
          if count <= 1 then None
          else begin
            let lanes = List.init count (fun i ->
              let buf = List.nth srcs (min i (List.length srcs - 1)) in
              K.index ~ptr:buf ~idxs:[K.gep ~src:vec ~idx:i] ()) in
            Some (K.vectorize ~srcs:lanes)
          end
      | _ -> None)
  | _ -> None

(* Load(GEP(x)) → GEP(Load(x, dtype=scalar.vec(src.count))). Updates the
   Load dtype to reflect the wider ptr, then applies the GEP after. *)
let gep_after_load (node : K.t) : K.t option =
  match K.view node with
  | Load { src; alt; dtype } -> begin
      match K.view src with
      | Gep { src = inner_ptr; idxs; dtype = gep_dt } ->
          (* Use GEP dtype count to widen the Load to match the vector width. *)
          let gep_count = Dtype.count gep_dt in
          let wide_dt = Dtype.vec (Dtype.scalar_of dtype) gep_count in
          let wide_load = K.replace node
            ~children:(inner_ptr :: Option.to_list alt)
            ~dtype:wide_dt () in
          Some (K.gep_multi ~src:wide_load ~idxs)
      | _ -> None end
  | _ -> None

let gep_on_store (node : K.t) : K.t option =
  match K.view node with
  | Store { dst; value; ranges } -> begin
      match K.view dst with
      | Gep { src = inner_ptr; idxs; _ } ->
          Some (K.store ~dst:inner_ptr ~value:(K.gep_multi ~src:value ~idxs) ~ranges)
      | _ -> None end
  | _ -> None

let ptrcat_after_load (node : K.t) : K.t option =
  match K.view node with
  | Load { src; alt; _ } -> begin
      match K.view src with
      | Ptrcat { srcs; _ } ->
          List.iter (fun p ->
            if not (K.is_ptr p) then begin
              let op = match K.view p with
                | Index _ -> "Index" | Cast _ -> "Cast" | Gep _ -> "Gep"
                | Param _ -> "Param" | Load _ -> "Load" | After _ -> "After"
                | Ptrcat _ -> "Ptrcat" | Vectorize _ -> "Vec"
                | Const _ -> "Const" | Binary _ -> "Bin"
                | Define_local _ -> "DefLocal" | Define_reg _ -> "DefReg"
                | _ -> "other"
              in
              Printf.eprintf "ptrcat_after_load: non-ptr op=%s dtype=%s\n%!" op
                (match K.dtype p with Some dt -> Dtype.to_string dt | None -> "none")
            end
          ) srcs;
          Some (K.cat ~srcs:(List.map (fun p -> K.load ~src:p ?alt ()) srcs))
      | _ -> None end
  | _ -> None

let cat_after_store (node : K.t) : K.t option =
  match K.view node with
  | Store { dst; value; ranges } -> begin
      match K.view dst with
      | Ptrcat { srcs; _ } ->
          let offset = ref 0 in
          let stores = List.map (fun p ->
            let count = match K.dtype p with Some d -> Dtype.count d | None -> 1 in
            let geps = List.init count (fun j -> K.gep ~src:value ~idx:(!offset + j)) in
            offset := !offset + count;
            let data = match geps with
              | [ single ] -> single | _ -> K.vectorize ~srcs:geps in
            K.store ~dst:p ~value:data ~ranges) srcs in
          Some (K.group stores)
      | _ -> None end
  | _ -> None

(* Extract (root_expression, constant_offset) from an index value.
   For ADD(expr, CONST(n)): root=expr, offset=n.
   For CONST(n): root=`Const, offset=n.
   Otherwise: root=the node itself, offset=0. *)
let decompose_idx (idx_node : K.t) : K.t option * int =
  match K.view idx_node with
  | Binary { op = `Add; lhs; rhs; _ } -> begin
      match K.view rhs with
      | Const { value; _ } -> begin
          match Const.view value with
          | Const.Int i -> (Some lhs, Int64.to_int i)
          | _ -> (Some idx_node, 0)
        end
      | _ -> begin
          match K.view lhs with
          | Const { value; _ } -> begin
              match Const.view value with
              | Const.Int i -> (Some rhs, Int64.to_int i)
              | _ -> (Some idx_node, 0)
            end
          | _ -> (Some idx_node, 0)
        end
    end
  | Const { value; _ } -> begin
      match Const.view value with
      | Const.Int i -> (None, Int64.to_int i)
      | _ -> (Some idx_node, 0)
    end
  | Invalid_index _ -> (None, -1) (* sentinel for invalid *)
  | _ -> (Some idx_node, 0)

(* Find runs of consecutive integers in a sorted list.
   E.g., [0;1;2;5;6] -> [[0;1;2]; [5;6]] *)
let group_consecutive (lst : int list) : int list list =
  match lst with
  | [] -> []
  | first :: _ ->
      let groups = ref [] in
      let current = ref [first] in
      let prev = ref first in
      List.iter (fun x ->
        if x = !prev then () (* skip first *)
        else if x = !prev + 1 then begin
          current := x :: !current;
          prev := x
        end else begin
          groups := List.rev !current :: !groups;
          current := [x];
          prev := x
        end) (List.tl lst);
      List.rev (List.rev !current :: !groups)

(* fold_expanded_index: Vectorize(Index(buf, idx0), ..., Index(buf, idxN))
   → grouped pointer-cast loads. *)
let fold_expanded_index (node : K.t) : K.t option =
  match K.view node with
  | Vectorize { srcs; _ } when srcs <> [] -> begin
      let is_index_with_same_buf () =
        match K.view (List.hd srcs) with
        | Index { ptr = buf0; _ } ->
            List.for_all (fun s ->
              match K.view s with
              | Index { ptr; _ } -> ptr == buf0
              | _ -> false) srcs
        | _ -> false
      in
      if not (is_index_with_same_buf ()) then None
      else begin
        let n = List.length srcs in
        let buf_pty = match K.view (List.hd srcs) with
          | Index { dtype = Dtype.P pty; _ } -> pty
          | _ -> Dtype.ptr_of Dtype.float32 ~addrspace:Dtype.Global ~size:(-1)
        in
        let entries = List.mapi (fun i s ->
          match K.view s with
          | Index { idxs = [idx_val]; gate; _ } ->
              let root, offset = decompose_idx idx_val in
              (gate, root, offset, i)
          | Index { idxs = []; gate; _ } ->
              (gate, None, 0, i)
          | _ -> (None, Some s, 0, i)) srcs
        in
        (* Group by (gate, root_src) using physical equality *)
        let groups : ((K.t option * K.t option) * (int * int) list) list =
          let tbl : ((K.t option * K.t option), (int * int) list) Hashtbl.t =
            Hashtbl.create 4
          in
          let key_order = ref [] in
          List.iter (fun (gate, root, offset, i) ->
            let key = (gate, root) in
            let found = ref false in
            List.iter (fun existing_key ->
              let (eg, er) = existing_key in
              if (match eg, gate with
                  | None, None -> true
                  | Some a, Some b -> a == b
                  | _ -> false)
                 && (match er, root with
                     | None, None -> true
                     | Some a, Some b -> a == b
                     | _ -> false)
              then begin
                found := true;
                let prev = Hashtbl.find tbl existing_key in
                Hashtbl.replace tbl existing_key ((offset, i) :: prev)
              end) !key_order;
            if not !found then begin
              Hashtbl.replace tbl key [(offset, i)];
              key_order := !key_order @ [key]
            end) entries;
          List.map (fun key ->
            (key, List.rev (Hashtbl.find tbl key))) !key_order
        in
        (* Build grouped indices *)
        let ret = ref [] in
        let idxs = Array.make n (-1) in
        let global_offset = ref 0 in
        List.iter (fun (_key, offset_indices) ->
          let sorted_offsets =
            List.sort (fun (a, _) (b, _) -> compare a b) offset_indices
          in
          let sorted_keys = List.map fst sorted_offsets in
          let consecutive_groups = group_consecutive sorted_keys in
          List.iter (fun grp ->
            let grp_len = List.length grp in
            let first_offset = List.hd grp in
            let first_orig_idx =
              List.assoc first_offset sorted_offsets
            in
            let lidx = List.nth srcs first_orig_idx in
            let widened =
              if grp_len > 1 then
                (* Widen the ptr dtype to load grp_len elements at once
                   by casting the Index to a wider ptr type. *)
                let wide_pty =
                  Dtype.ptr_of
                    (Dtype.vec (Dtype.scalar_of (Dtype.base buf_pty)) grp_len)
                    ~addrspace:(Dtype.addrspace buf_pty) ~size:(Dtype.ptr_size buf_pty)
                in
                K.cast ~src:lidx ~dtype:(Dtype.ptr_to_any wide_pty)
              else lidx
            in
            (* Map each element to its position in the result *)
            List.iteri (fun lane_i offset ->
              let orig_indices =
                List.filter_map (fun (o, i) ->
                  if o = offset then Some i else None) sorted_offsets
              in
              List.iter (fun oi ->
                idxs.(oi) <- !global_offset + lane_i) orig_indices)
              grp;
            ret := widened :: !ret;
            global_offset := !global_offset + grp_len)
            consecutive_groups)
          groups;
        if Array.exists (fun x -> x < 0) idxs then None
        else begin
          let ret = List.rev !ret in
          let total = !global_offset in
          if total = 0 then None
          else
            let idxs_list = Array.to_list idxs in
            let cat_pty =
              Dtype.ptr_of
                (Dtype.scalar_of (Dtype.base buf_pty))
                ~addrspace:(Dtype.addrspace buf_pty) ~size:(Dtype.ptr_size buf_pty)
            in
            let post_cat = K.ptrcat ~srcs:ret ~dtype:(Dtype.ptr_with_v cat_pty total) in
            Some (K.gep_multi ~src:post_cat ~idxs:idxs_list)
        end
      end
    end
  | _ -> None

let load_store_folding_rule =
  K.first_match [
    expand_index; fold_expanded_index;
    gep_after_load; gep_on_store;
    ptrcat_after_load; cat_after_store;
  ]

let load_store_indexing_rule = drop_true_gate

(* Render preparation *)

let expand_vector_const (node : K.t) : K.t option =
  match K.view node with
  | Const { value; dtype } when Dtype.count dtype > 1 ->
      let c = K.const value in
      Some (K.vectorize ~srcs:(List.init (Dtype.count dtype) (fun _ -> c)))
  | _ -> None

let expand_vconst (node : K.t) : K.t option =
  match K.view node with
  | Vconst { values; _ } ->
      Some (K.vectorize ~srcs:(List.map K.const values))
  | _ -> None

let trivial_gep (node : K.t) : K.t option =
  match K.view node with
  | Gep { src; idxs = [0]; _ } -> (
      match K.dtype src with Some dt when Dtype.count dt = 1 -> Some src | _ -> None)
  | _ -> None

let lower_cat (node : K.t) : K.t option =
  match K.view node with
  | Cat { srcs; _ } ->
      let lanes = List.concat_map (fun src ->
        match K.dtype src with
        | Some dt when Dtype.count dt > 1 ->
            List.init (Dtype.count dt) (fun i -> K.gep ~src ~idx:i)
        | _ -> [ src ]) srcs
      in
      Some (K.vectorize ~srcs:lanes)
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

type cast_wrapper = No_wrap | Wrap_cast of Dtype.any | Wrap_bitcast of Dtype.t
type gate_relation = Same_gate | Negated_gate | Different_gate

let peel_wrapper (node : K.t) =
  match K.view node with
  | Cast { src; dtype } -> Wrap_cast dtype, src
  | Bitcast { src; dtype } -> Wrap_bitcast dtype, src
  | _ -> No_wrap, node

let apply_wrapper w src = match w with
  | No_wrap -> src
  | Wrap_cast dtype -> K.cast ~src ~dtype
  | Wrap_bitcast dtype -> K.bitcast ~src ~dtype

let gate_relation ~cond ~gate =
  if cond == gate then Same_gate
  else match K.view gate with
  | Binary { op = `Xor; lhs; rhs; dtype }
    when Dtype.scalar dtype = Dtype.Bool && Dtype.count dtype = 1 ->
      if (lhs == cond && is_true_const rhs)
         || (rhs == cond && is_true_const lhs)
      then Negated_gate else Different_gate
  | _ -> Different_gate

let rec find_index_gate (node : K.t) : K.t option =
  match K.view node with
  | Index { gate = Some g; _ } -> Some g
  | Cast { src; _ } | Bitcast { src; _ } -> find_index_gate src
  | _ -> None

let coerce_alt ~load_dtype alt =
  match K.dtype alt with
  | Some dt when Dtype.equal dt load_dtype -> alt
  | _ ->
      match K.view alt with
      | Cast { src; _ } ->
          (match K.dtype src with
           | Some dt when Dtype.equal dt load_dtype -> src
           | _ -> K.cast ~src:alt ~dtype:(Dtype.to_any load_dtype))
      | _ -> K.cast ~src:alt ~dtype:(Dtype.to_any load_dtype)

(* Fold Where(cond, Load(gated_ptr), fallback) into Load(gated_ptr, alt=fallback)
   when the Index gate matches (or negates) cond, eliminating the redundant branch. *)
let where_after_gated_load (node : K.t) : K.t option =
  match K.view node with
  | Ternary { op = `Where; a = cond; b = true_side; c = false_side; _ } ->
      let try_fold ~load_side ~other_side ~require_negated =
        let wrapper, load_node = peel_wrapper load_side in
        match K.view load_node with
        | Load { src; dtype = load_dtype; _ } -> begin
            match find_index_gate src with
            | None -> None
            | Some gate ->
                let matches = match gate_relation ~cond ~gate with
                  | Same_gate -> not require_negated
                  | Negated_gate -> require_negated
                  | Different_gate -> false
                in
                if not matches then None
                else
                  let alt = coerce_alt ~load_dtype other_side in
                  Some (apply_wrapper wrapper (K.load ~src ~alt ())) end
        | _ -> None
      in
      (match try_fold ~load_side:true_side ~other_side:false_side
               ~require_negated:false with
       | Some _ as r -> r
       | None ->
           try_fold ~load_side:false_side ~other_side:true_side
             ~require_negated:true)
  | _ -> None

let pm_render_rule =
  K.first_match [
    expand_vector_const; expand_vconst; trivial_gep; lower_cat;
    trivial_vectorize; where_after_gated_load; masked_load_alt;
  ]

let pm_render (root : K.t) : K.t =
  K.graph_rewrite pm_render_rule root
