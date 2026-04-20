(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_ir
module K = Kernel

(* Helpers *)

(* Partition a sorted int list into runs of consecutive values.
   [0;1;2;5;6] -> [[0;1;2]; [5;6]] *)
let group_consecutive = function
  | [] -> []
  | x :: rest ->
      let rec go acc cur = function
        | [] -> List.rev (List.rev cur :: acc)
        | x :: rest ->
            match cur with
            | prev :: _ when x = prev + 1 -> go acc (x :: cur) rest
            | _ -> go (List.rev cur :: acc) [x] rest
      in
      go [] [x] rest

(* Grouping key for fold_expanded_index: physical identity for nodes,
   structural equality for constants, a sentinel for Invalid. *)
type root_key =
  | Root_node of K.t
  | Root_invalid
  | Root_const

(* Split idx into (root, constant_offset).  ADD(e, c) -> (e, c). *)
let decompose_idx idx =
  match K.view idx with
  | Binary { op = `Add; lhs; rhs; _ } -> (
      match K.const_arg rhs with
      | Some (Int c) -> Root_node lhs, Int64.to_int c
      | _ -> match K.const_arg lhs with
        | Some (Int c) -> Root_node rhs, Int64.to_int c
        | _ -> Root_node idx, 0)
  | Const { value; _ } -> (
      match Const.view value with
      | Int c -> Root_const, Int64.to_int c
      | _ -> Root_node idx, 0)
  | Invalid_index _ -> Root_invalid, 0
  | _ -> Root_node idx, 0

(* load_store_indexing *)

(* Is this node a monotonically increasing function of its inputs? *)
let rec is_increasing node = match K.view node with
  | Const _ | Define_var _ | Special _ | Range _ -> true
  | Binary { op = `Add; lhs; rhs; _ } -> is_increasing lhs && is_increasing rhs
  | Binary { op = `Mul | `Idiv; lhs; rhs; _ }
    when K.is_const rhs && K.vmin rhs >= 0 -> is_increasing lhs
  | _ -> false

(* For image indexes: determine which validity clauses can be dropped
   because the index is provably out of bounds when the clause is false. *)
let drop_valid_stmts valid idx height width =
  let open Symbolic in
  List.filter (fun stmt ->
    match parse_valid stmt with
    | None -> false
    | Some (x, is_upper, c) ->
        (* For X0 + X1 + ... >= 1, check out-of-bound when all Xi = 0. *)
        if not is_upper && c = 1
           && List.for_all (fun u -> Symbolic.is_irreducible u && K.vmin u = 0)
                (Divandmod.split_add x)
        then
          let testidx = List.fold_left (fun nowidx u ->
            K.substitute [(u, K.const_int 0)] nowidx)
            idx (Divandmod.split_add x) in
          K.vmax (K.gep ~src:testidx ~idx:0) < 0
          || K.vmax (K.gep ~src:testidx ~idx:1) < 0
        else
          (* If X <= c, check out-of-bound at X = c+1.
             If X >= c, check out-of-bound at X = c-1. *)
          let test_value = if is_upper then c + 1 else c - 1 in
          let dims = [width; height] in
          let srcs = K.children idx in
          List.exists2 (fun i b ->
            if is_increasing i then
              let rw = K.substitute [(x, K.const_int test_value)] i in
              K.vmin rw >= b || K.vmax rw < 0
            else false)
            (List.filteri (fun j _ -> j < List.length dims) srcs) dims)
    (Symbolic.split_and valid)

(* Simplify an INDEX validity gate.  For non-image buffers, runs
   uop_given_valid on the index; for Param_image buffers, also drops
   redundant validity clauses proved by image dimension bounds. *)
let simplify_valid_load buf start_idx valid =
  let idx = Symbolic.uop_given_valid valid start_idx in
  match K.view buf with
  | Param_image { width; height; _ } ->
      (* Wait for image-indexed form (2-component coords). *)
      if Dtype.count (K.dtype start_idx) <> 2 then None
      else
        let drop = drop_valid_stmts valid idx height width in
        if drop = [] && idx == start_idx then None
        else
          let remaining = List.filter (fun s ->
            not (List.exists (fun d -> d == s) drop))
            (Symbolic.split_and valid) in
          let gated_idx = match remaining with
            | [] -> idx
            | _ ->
                let new_valid = List.fold_left (fun acc s ->
                  K.binary ~op:`And ~lhs:acc ~rhs:s)
                  (List.hd remaining) (List.tl remaining) in
                K.ternary ~op:`Where ~a:new_valid ~b:idx
                  ~c:(K.invalid_index ~lanes:(Dtype.count (K.dtype idx)) ())
          in
          Some (K.index ~ptr:buf ~idxs:[gated_idx] ())
  | _ ->
      if idx == start_idx then None
      else
        let gated_idx = K.ternary ~op:`Where ~a:valid ~b:idx
          ~c:(K.invalid_index ~lanes:(Dtype.count (K.dtype idx)) ()) in
        Some (K.index ~ptr:buf ~idxs:[gated_idx] ())

(* Remove a gate that is the constant [true]. *)
let drop_true_gate (node : K.t) : K.t option =
  match K.view node with
  | Index { ptr; idxs; gate = Some g; _ }
    when K.const_arg g = Some (Bool true) ->
      Some (K.index ~ptr ~idxs ())
  | _ -> None

(* Match INDEX(buf, where(cond, x, Invalid)) or INDEX(buf, x:long, c:bool)
   and simplify the validity gate via uop_given_valid. *)
let simplify_valid_index (node : K.t) : K.t option =
  match K.view node with
  | Index { ptr; idxs = [idx]; gate = None; _ } ->
      (* Pattern 1: INDEX(buf, where(cond, x, Invalid)) *)
      (match K.view idx with
       | Ternary { op = `Where; a = cond; b = x; c = inv; _ }
         when (match K.view inv with Invalid_index _ -> true | _ -> false) ->
           simplify_valid_load ptr x cond
       | _ -> None)
  | Index { ptr; idxs = [x]; gate = Some c; _ } ->
      (* Pattern 2: INDEX(buf, x, c:bool) — after index dtype lowered *)
      (match K.dtype_opt x with
       | Some dt when Dtype.scalar dt = Dtype.Int64 ->
           simplify_valid_load ptr x c
       | _ -> None)
  | _ -> None

let load_store_indexing node =
  match simplify_valid_index node with
  | Some _ as r -> r
  | None -> drop_true_gate node

(* load/store grouping *)

(* Expand Index(Vectorize(buf,...), vec) into Vectorize of per-lane indexes. *)
let expand_index (node : K.t) : K.t option =
  match K.view node with
  | Index { ptr; idxs = [vec]; gate = None; _ } -> (
      match K.view ptr with
      | Vectorize { srcs = buf :: _; _ } ->
          let n = Dtype.count (K.dtype vec) in
          let lanes = List.init n (fun i ->
            K.index ~ptr:buf ~idxs:[K.gep ~src:vec ~idx:i] ()) in
          Some (K.vectorize ~srcs:lanes)
      | _ -> None)
  | _ -> None

(* Fold Vectorize(Index(buf,i0),...,Index(buf,iN)) back into grouped
   pointer-cast accesses: consecutive offsets share a single wide ptr. *)
let fold_expanded_index (node : K.t) : K.t option =
  match K.view node with
  | Vectorize { srcs; _ } when srcs <> [] ->
      let first = match K.view (List.hd srcs) with
        | Index { ptr; dtype = Dtype.Ptr pty; _ } -> Some (ptr, pty)
        | _ -> None in
      (match first with
       | None -> None
       | Some (buf, buf_pty) ->
      if not (List.for_all (fun s -> match K.view s with
        | Index { ptr; _ } -> ptr == buf | _ -> false) srcs) then None
      else if not (List.for_all K.is_ptr srcs) then None
      else begin
        let n = List.length srcs in
        (* Ordered map with physical-equality keys on K.t nodes. *)
        let offsets :
            (K.t option * root_key, (int, int list) Hashtbl.t) Hashtbl.t =
          Hashtbl.create 4 in
        let key_order = ref [] in
        let find_or_create valid root =
          let eq_key (v, r) =
            (match v, valid with
             | None, None -> true | Some a, Some b -> a == b | _ -> false)
            && (match r, root with
                | Root_node a, Root_node b -> a == b
                | Root_invalid, Root_invalid -> true
                | Root_const, Root_const -> true
                | _ -> false)
          in
          match List.find_opt eq_key !key_order with
          | Some k -> Hashtbl.find offsets k
          | None ->
              let k = (valid, root) in
              let tbl = Hashtbl.create 4 in
              Hashtbl.replace offsets k tbl;
              key_order := k :: !key_order;
              tbl
        in
        (* Collect per-src offsets keyed by (gate, root). *)
        List.iteri (fun i s ->
          match K.view s with
          | Index { idxs = [idx]; gate; _ } ->
              let root, arg = decompose_idx idx in
              let tbl = find_or_create gate root in
              let prev = match Hashtbl.find_opt tbl arg with
                | Some l -> l | None -> [] in
              Hashtbl.replace tbl arg (i :: prev)
          | _ -> ()) srcs;
        (* Group consecutive offsets and widen ptrs. *)
        let group_and_widen () =
          let ret = ref [] in
          let idxs = Array.make n (-1) in
          let global_offset = ref 0 in
          List.iter (fun key ->
            let tbl = Hashtbl.find offsets key in
            let sorted_args = List.sort compare
              (Hashtbl.fold (fun k _ acc -> k :: acc) tbl []) in
            List.iter (fun grp ->
              let grp_len = List.length grp in
              let first_off = List.hd grp in
              let first_orig = List.hd (Hashtbl.find tbl first_off) in
              let lidx = List.nth srcs first_orig in
              let lidx =
                if grp_len > 1 then
                  let scalar = Dtype.Val.scalarize (Dtype.Ptr.base buf_pty) in
                  let wide_pty = Dtype.Ptr.with_base
                    (Dtype.Val.vec grp_len scalar) buf_pty in
                  K.cast ~src:lidx ~dtype:(Dtype.Ptr wide_pty)
                else lidx
              in
              List.iteri (fun lane_i g ->
                List.iter (fun oi ->
                  idxs.(oi) <- !global_offset + lane_i)
                  (Hashtbl.find tbl g)) grp;
              ret := lidx :: !ret;
              global_offset := !global_offset + grp_len)
              (group_consecutive sorted_args))
            (List.rev !key_order);
          if Array.exists (fun x -> x < 0) idxs then None
          else Some (List.rev !ret, Array.to_list idxs, !global_offset)
        in
        (* Assemble PTRCAT + GEP result. *)
        match group_and_widen () with
        | None | Some (_, _, 0) -> None
        | Some (ret, idxs_list, total) ->
            let scalar = Dtype.Val.scalarize (Dtype.Ptr.base buf_pty) in
            let cat_pty = Dtype.Ptr.with_base scalar buf_pty in
            let post_cat = K.ptrcat ~srcs:ret
              ~dtype:(Dtype.Ptr.vec total cat_pty) in
            Some (K.gep_multi ~src:post_cat ~idxs:idxs_list)
      end)
  | _ -> None

(* Push GEP through LOAD: Load(GEP(x, arg)) -> GEP(Load(x, wider_dtype), arg). *)
let gep_after_load (node : K.t) : K.t option =
  match K.view node with
  | Load { src; alt; dtype } -> (
      match K.view src with
      | Gep { src = inner; idxs; dtype = gep_dt } ->
          let wide_dt = Dtype.vec (Dtype.Val.count gep_dt) (Dtype.scalarize (Dtype.Val dtype)) in
          let wide_load = K.replace node
            ~children:(inner :: Option.to_list alt)
            ~dtype:wide_dt () in
          Some (K.gep_multi ~src:wide_load ~idxs)
      | _ -> None)
  | _ -> None

(* Push GEP through STORE: Store(GEP(x, perm), val) -> Store(x, GEP(val, inv_perm)).
   XXX does not handle expanding (duplicate) GEPs — same as tinygrad. *)
let gep_on_store (node : K.t) : K.t option =
  match K.view node with
  | Store { dst; value; ranges } -> (
      match K.view dst with
      | Gep { src = inner; idxs; _ } ->
          let n = List.length idxs in
          let inv = Array.make n 0 in
          List.iteri (fun i x -> if x >= 0 && x < n then inv.(x) <- i) idxs;
          Some (K.store ~dst:inner
                  ~value:(K.gep_multi ~src:value ~idxs:(Array.to_list inv))
                  ~ranges)
      | _ -> None)
  | _ -> None

(* Split Load(Ptrcat(p0,...,pN)) into Vcat(Load(p0),...,Load(pN)). *)
let ptrcat_after_load (node : K.t) : K.t option =
  match K.view node with
  | Load { src; alt; _ } -> (
      match K.view src with
      | Ptrcat { srcs; _ } ->
          Some (K.vcat ~srcs:(List.map (fun p -> K.load ~src:p ?alt ()) srcs))
      | _ -> None)
  | _ -> None

(* Split Store(Ptrcat(p0,...,pN), data) into Group(Store(p0, slice0), ...). *)
let ptrcat_after_store (node : K.t) : K.t option =
  match K.view node with
  | Store { dst; value; ranges } -> (
      match K.view dst with
      | Ptrcat { srcs; _ } ->
          let rec go acc offset = function
            | [] -> Some (K.group (List.rev acc))
            | p :: rest ->
                let n = Dtype.count (K.dtype p) in
                let chunk = K.gep_multi ~src:value
                  ~idxs:(List.init n (fun j -> offset + j)) in
                go (K.store ~dst:p ~value:chunk ~ranges :: acc) (offset + n) rest
          in
          go [] 0 srcs
      | _ -> None)
  | _ -> None

(* correct load/store *)

(* Extract (ptr, idxs, gate, buf_pty, sz) from a Cast(Index(...)) src. *)
let extract_cast_index src =
  match K.view src with
  | Cast { src = idx; dtype = Dtype.Ptr pty } -> (
      match K.view idx with
      | Index { ptr; idxs; gate; _ } ->
          let sz = Dtype.Val.count (Dtype.Ptr.base pty) in
          if not (K.is_ptr ptr) then None
          else
            let buf_pty = K.ptr_dtype ptr in
            if sz = 1 || Dtype.Ptr.addrspace buf_pty = Dtype.Reg then None
            else Some (ptr, idxs, gate, buf_pty, sz)
      | _ -> None)
  | _ -> None

(* Split wide Load/Store(Cast(Index)) into renderer-supported widths.
   Image and DSP paths omitted. *)
let split_load_store (ren : Renderer.t) (node : K.t) : K.t option =
  (* Determine fold widths, filter by divisibility, split into chunks. *)
  let split ptr idxs gate buf_pty sz mk_item =
    let base_scalar = Dtype.Val.scalarize (Dtype.Ptr.base buf_pty) in
    let widths = match Dtype.Val.scalar base_scalar with
      | Float32 | Float16 | Fp8e4m3 | Fp8e5m2
        when Renderer.supports_float4 ren ->
          if Dtype.Val.scalar base_scalar = Float16 && Helpers.allow_half8
          then [8; 4; 2]
          else if Helpers.amx then [16; 8; 4; 2]
          else [4; 2]
      | _ -> []
    in
    let offset = List.hd idxs in
    let lengths = List.filter (fun x ->
      K.divides offset x <> None) (widths @ [1]) in
    let rec go acc off =
      if off >= sz then List.rev acc
      else match List.find_opt (fun fl -> off + fl <= sz) lengths with
      | None -> List.rev acc
      | Some fl ->
          let new_idxs =
            if off = 0 then idxs
            else List.map (fun i ->
              K.binary ~op:`Add ~lhs:i ~rhs:(K.const_int off)) idxs in
          let base_idx = K.index ~ptr ~idxs:new_idxs ?gate () in
          let lidx =
            if fl > 1 then
              let wide_pty = Dtype.Ptr.with_base
                (Dtype.Val.vec fl base_scalar) buf_pty in
              K.cast ~src:base_idx ~dtype:(Dtype.Ptr wide_pty)
            else base_idx in
          go (mk_item lidx fl off :: acc) (off + fl)
    in
    match go [] 0 with [] | [_] -> None | ret -> Some ret
  in
  match K.view node with
  | Load { src; alt; dtype } -> (
      match extract_cast_index src with
      | None -> None
      | Some (ptr, idxs, gate, buf_pty, sz) ->
          Option.map (fun ret -> K.vcat ~srcs:ret)
            (split ptr idxs gate buf_pty sz (fun lidx fl _off ->
               K.replace node ~children:(lidx :: Option.to_list alt)
                 ~dtype:(Dtype.vec fl (Dtype.scalarize (Dtype.Val dtype))) ())))
  | Store { dst; value; ranges } -> (
      match extract_cast_index dst with
      | None -> None
      | Some (ptr, idxs, gate, buf_pty, sz) ->
          Option.map K.group
            (split ptr idxs gate buf_pty sz (fun lidx fl off ->
               K.store ~dst:lidx ~ranges
                 ~value:(K.gep_multi ~src:value
                           ~idxs:(List.init fl (fun j -> off + j))))))
  | _ -> None

let pm_correct_load_store_rule ren = split_load_store ren

(* devectorize *)

let prod lst = List.fold_left ( * ) 1 lst

(* Break a wide WMMA into multiple smaller WMMAs matching the upcast chunk size. *)
let no_vectorized_wmma (node : K.t) : K.t option =
  match K.view node with
  | Wmma { a; b; c; dtype; upcast_axes = ua, ub, uc; _ } ->
      let out_sz = prod (List.map snd uc) in
      if Dtype.Val.count dtype = out_sz then None
      else
        let chunk src axes =
          let ssz = prod (List.map snd axes) in
          let cnt = Dtype.count (K.dtype src) in
          List.init (cnt / ssz) (fun g ->
            K.gep_multi ~src ~idxs:(List.init ssz (fun j -> g * ssz + j)))
        in
        let wmma_dt = Dtype.vec out_sz (Dtype.scalarize (Dtype.Val dtype)) in
        let wmmas = List.map2 (fun (a, b) c ->
          K.replace node ~children:[a; b; c] ~dtype:wmma_dt ())
          (List.combine (chunk a ua) (chunk b ub)) (chunk c uc) in
        let srcs = List.concat_map (fun w ->
          List.init out_sz (fun i -> K.gep ~src:w ~idx:i)) wmmas in
        Some (K.vectorize ~srcs)
  | _ -> None

(* Scalarize vectorized ALU/Cast/Bitcast by extracting each lane. *)
let no_vectorized_alu (node : K.t) : K.t option =
  match K.view node with
  (* WHERE with Invalid 3rd arg: image index pattern, skip *)
  | Ternary { op = `Where; c; _ }
    when (match K.view c with Invalid_index _ -> true | _ -> false) -> None
  | Unary _ | Binary _ | Ternary _ | Cast _ | Bitcast _ ->
      let adt = K.dtype node in
      let vc = Dtype.vcount adt in
      if vc <= 1 then None
      else
        let children = K.children node in
        let scalar_dt = Dtype.scalarize adt in
        let srcs = List.init vc (fun i ->
          K.replace node
            ~children:(List.map (fun s -> K.gep ~src:s ~idx:i) children)
            ~dtype:scalar_dt ()) in
        Some (K.vectorize ~srcs)
  | _ -> None

(* Scalarize DEFINE_LOCAL/DEFINE_REG with vector base: widen size, scalarize
   base, cast back. *)
let no_vectorized_buf (node : K.t) : K.t option =
  let scalarize size dtype mk =
    let cnt = Dtype.Val.count (Dtype.Ptr.base dtype) in
    let scalar_pty = Dtype.Ptr.with_size (Dtype.Ptr.size dtype * cnt)
      (Dtype.Ptr.with_base (Dtype.Val.scalarize (Dtype.Ptr.base dtype)) dtype) in
    Some (K.cast ~src:(mk (size * cnt) scalar_pty)
      ~dtype:(Dtype.Ptr dtype))
  in
  match K.view node with
  | Define_local { size; dtype } when Dtype.Val.count (Dtype.Ptr.base dtype) > 1 ->
      scalarize size dtype (fun size dtype -> K.define_local ~size ~dtype)
  | Define_reg { size; dtype; slot } when Dtype.Val.count (Dtype.Ptr.base dtype) > 1 ->
      scalarize size dtype (fun size dtype -> K.define_reg ~size ~dtype ~slot)
  | _ -> None

(* Scalarize a vector Index on local/reg memory.
   Handles three ptr shapes matching tinygrad's devectorize_buf_and_index:
   1. Cast(buf).index(idx)              — plain scalar index
   2. Cast(buf).broadcast(b).index(idx) — broadcast index
   3. Cast(buf).gep(g).index(idx)       — GEP-selected lanes *)
let no_vectorized_index (node : K.t) : K.t option =
  let rec is_local_or_reg n = match K.view n with
    | After { src; _ } -> is_local_or_reg src
    | Define_local _ | Define_reg _ -> true
    | _ -> false
  in
  let check_cast n = match K.view n with
    | Cast { src = buf; dtype = Dtype.Ptr cp; _ }
      when is_local_or_reg buf -> Some (buf, cp)
    | _ -> None
  in
  match K.view node with
  | Index { ptr; idxs; dtype = Dtype.Ptr pty; _ }
    when Dtype.Val.count (Dtype.Ptr.base pty) > 1 ->
      (* Decompose ptr into (buf, cast_pty, bcast_kind) *)
      let found = match K.view ptr with
        | Cast _ ->
            Option.map (fun (buf, cp) -> (buf, cp, `Plain)) (check_cast ptr)
        | Vectorize { srcs = s :: _; _ } ->
            Option.map (fun (buf, cp) -> (buf, cp, `Broadcast ptr)) (check_cast s)
        | Gep { src = inner; idxs = gep_idxs; _ } ->
            Option.map (fun (buf, cp) -> (buf, cp, `Gep gep_idxs)) (check_cast inner)
        | _ -> None
      in
      Option.bind found (fun (buf, cast_pty, bcast) ->
        let cnt = Dtype.Val.count (Dtype.Ptr.base cast_pty) in
        let pairs = match bcast with
          | `Gep gep_idxs ->
              let vc = Dtype.Ptr.v cast_pty in
              let n_gep = List.length gep_idxs in
              List.init (vc * n_gep) (fun i ->
                (i mod n_gep, i / n_gep + List.nth gep_idxs (i mod n_gep)))
          | `Broadcast bnode ->
              let bvc = Dtype.vcount (K.dtype bnode) in
              List.init (cnt * bvc) (fun i -> (i mod bvc, i / bvc))
          | `Plain ->
              List.init cnt (fun c -> (0, c))
        in
        let n = List.length pairs in
        let open K.O in
        let idx = match idxs with
          | [] -> int_ 0
          | first :: rest -> List.fold_left ( + ) first rest in
        let lane_sel = K.gep_multi ~src:idx ~idxs:(List.map fst pairs) in
        let stride = K.broadcast (int_ cnt) n in
        let off = K.vectorize ~srcs:(List.map (fun (_, o) -> int_ o) pairs) in
        let wide_idx = lane_sel * stride + off in
        Some (K.index ~ptr:(K.broadcast buf n) ~idxs:[wide_idx] ()))
  | _ -> None

(* Move Cast out of After: After(Cast(x, dt), deps) -> Cast(After(x, deps), dt). *)
let cast_after_after (node : K.t) : K.t option =
  match K.view node with
  | After { src; deps } -> (
      match K.view src with
      | Cast { src = inner; dtype } ->
          Some (K.cast ~src:(K.after ~src:inner ~deps) ~dtype)
      | _ -> None)
  | _ -> None

(* pm_render *)

(* Expand vector Const into Vectorize of scalar copies. *)
let expand_vector_const (node : K.t) : K.t option =
  match K.view node with
  | Const { value; dtype } when Dtype.Val.count dtype > 1 ->
      let c = K.const value in
      Some (K.vectorize ~srcs:(List.init (Dtype.Val.count dtype) (fun _ -> c)))
  | _ -> None

(* Expand Vconst into Vectorize of per-lane scalar constants. *)
let expand_vconst (node : K.t) : K.t option =
  match K.view node with
  | Vconst { values; _ } ->
      Some (K.vectorize ~srcs:(List.map K.const values))
  | _ -> None

(* Expand multi-element GEP into Vectorize of single-element GEPs. *)
let expand_multi_gep (node : K.t) : K.t option =
  match K.view node with
  | Gep { src; idxs; _ } when List.length idxs > 1 ->
      Some (K.vectorize ~srcs:(List.map (fun x -> K.gep ~src ~idx:x) idxs))
  | _ -> None

(* Remove trivial GEP(x, 0) when x is scalar. *)
let trivial_gep (node : K.t) : K.t option =
  match K.view node with
  | Gep { src; idxs = [0]; _ } ->
      if Dtype.vcount (K.dtype src) = 1 then Some src
      else None
  | _ -> None

(* Remove single-element Vectorize. *)
let trivial_vectorize (node : K.t) : K.t option =
  match K.view node with
  | Vectorize { srcs = [src]; _ } -> Some src
  | _ -> None

(* Find the INDEX gate through Cast/Bitcast wrappers. *)
let rec find_gate n = match K.view n with
  | Index { gate = Some g; _ } -> Some g
  | Cast { src; _ } | Bitcast { src; _ } -> find_gate src
  | _ -> None

(* Give gated loads a zero alt value when they don't have one yet.
   Tinygrad also checks for CUSTOM/STORE/BARRIER in the alt position;
   the OCaml IR uses a typed [alt : t option] field so effect nodes
   cannot appear there — matching [alt = None] is sufficient. *)
let masked_load_alt (node : K.t) : K.t option =
  match K.view node with
  | Load { src; alt = None; _ } when find_gate src <> None ->
      Some (K.load ~src ~alt:(K.zero_like node) ())
  | _ -> None

(* Is [gate] the logical negation of [cond]?  i.e. gate = xor(cond, true). *)
let is_negated cond gate = match K.view gate with
  | Binary { op = `Xor; lhs; rhs; _ } ->
      (lhs == cond && K.const_arg rhs = Some (Bool true))
      || (rhs == cond && K.const_arg lhs = Some (Bool true))
  | _ -> false

(* Fold Where(cond, Load(gated), fallback) into Load(gated, alt=fallback)
   when the INDEX gate matches or negates the WHERE condition. *)
let where_after_gated_load (node : K.t) : K.t option =
  let try_fold cond load_side alt_side ~negated =
    let inner, wrap_dt = match K.view load_side with
      | Cast { src; dtype } -> src, Some dtype
      | _ -> load_side, None
    in
    match K.view inner with
    | Load { src; dtype = load_dt; _ } -> (
        match find_gate src with
        | Some gate
          when (if negated then is_negated cond gate else cond == gate) ->
            (* Unwrap Cast if inner already matches load dtype, avoiding
               a roundtrip cast (e.g. uint->float->uint). *)
            let alt = match K.view alt_side with
              | Cast { src = inner_alt; _ }
                when K.dtype_opt inner_alt = Some (Dtype.Val load_dt) -> inner_alt
              | _ -> K.cast ~src:alt_side ~dtype:(Dtype.Val load_dt)
            in
            let load = K.load ~src ~alt () in
            let result_dt = match wrap_dt with
              | Some dt -> dt | None -> Dtype.Val load_dt in
            Some (K.cast ~src:load ~dtype:result_dt)
        | _ -> None)
    | _ -> None
  in
  match K.view node with
  | Ternary { op = `Where; a = cond; b = true_side; c = false_side; _ } -> (
      match try_fold cond true_side false_side ~negated:false with
      | Some _ as r -> r
      | None -> try_fold cond false_side true_side ~negated:true)
  | _ -> None

(* Reduce lowering *)

let identity_element = Const.identity_element

(* Split horizontal reduction lanes when input is wider than output. *)
let horizontal_reduce (inp : K.t) (out_dtype : Dtype.t) : K.t list =
  let inp_dt = K.dtype inp in
  if Dtype.equal inp_dt out_dtype then [inp]
  else
    let amount = Dtype.count inp_dt / Dtype.count out_dtype in
    List.init amount (fun i ->
      K.gep_multi ~src:inp
        ~idxs:(List.init (Dtype.count out_dtype) (fun j -> i + j * amount)))

(* Reduce a list with a binary op: [a; b; c] -> op(op(a, b), c). *)
let reduce_fold op = function
  | [] -> failwith "reduce_fold: empty list"
  | first :: rest ->
      List.fold_left
        (fun a x -> K.binary ~op:(op :> Op.binary) ~lhs:a ~rhs:x)
        first rest

type reduce_ctx = { mutable acc_num : int }

(* Lower Reduce into an explicit register accumulator with END loop. *)
let reduce_to_acc (ctx : reduce_ctx) (node : K.t) : K.t option =
  match K.view node with
  | Reduce { op; src = inp; ranges = reduce_range; dtype } ->
      let lst = horizontal_reduce inp (Dtype.Val dtype) in
      if reduce_range = [] then Some (reduce_fold op lst)
      else begin
        let topo = K.toposort inp in
        let ended = K.Ref_tbl.create 16 in
        List.iter (fun n -> match K.view n with
          | End { ranges; _ } ->
              List.iter (fun r -> K.Ref_tbl.replace ended r ()) ranges
          | _ -> ()) topo;
        let reduce_set = K.Ref_tbl.create 8 in
        List.iter (fun r -> K.Ref_tbl.replace reduce_set r ()) reduce_range;
        let input_ranges = List.filter (fun n ->
          K.is_range n
          && not (K.Ref_tbl.mem reduce_set n)
          && not (K.Ref_tbl.mem ended n)) topo in
        let identity = K.broadcast
          (K.const (identity_element op (Dtype.Val.scalarize dtype)))
          (Dtype.Val.count dtype) in
        let acc_pty = Dtype.Ptr.create dtype ~addrspace:Dtype.Reg ~size:1 in
        let acc = K.define_reg ~size:1 ~dtype:acc_pty ~slot:ctx.acc_num in
        ctx.acc_num <- ctx.acc_num + 1;
        let zero = K.const_int 0 in
        let idx ptr = K.index ~ptr ~idxs:[zero] ~as_ptr:false () in
        let acc_after_input = match input_ranges with
          | [] -> acc | deps -> K.after ~src:acc ~deps in
        let acc_init =
          K.store ~dst:(idx acc_after_input) ~value:identity ~ranges:[] in
        let acc_in_loop =
          K.after ~src:acc ~deps:(acc_init :: reduce_range) in
        let ret = reduce_fold op (idx acc_in_loop :: lst) in
        let store_back = K.store ~dst:(idx acc) ~value:ret ~ranges:[] in
        let end_node =
          K.end_ ~value:store_back ~ranges:reduce_range ~tag:"mergeable" () in
        Some (idx (K.after ~src:acc ~deps:[end_node]))
      end
  | _ -> None

(* Merge END nodes that share the same ranges (created by reduce_to_acc). *)
let merge_reduce_ends (_ctx : reduce_ctx) (node : K.t) : K.t option =
  match K.view node with
  | Sink _ ->
      let by_ranges : (K.t list, K.t list) Hashtbl.t = Hashtbl.create 8 in
      List.iter (fun n ->
        match K.view n with
        | End { ranges; _ } when K.tag n = Some "mergeable" ->
            let prev = match Hashtbl.find_opt by_ranges ranges with
              | Some l -> l | None -> [] in
            Hashtbl.replace by_ranges ranges (n :: prev)
        | _ -> ()) (K.toposort node);
      let mappings = Hashtbl.fold (fun ranges ends acc ->
        if List.length ends <= 1 then acc
        else
          let stores = List.map (fun e ->
            match K.view e with
            | End { value; _ } -> value | _ -> assert false) ends in
          let merged = K.end_ ~value:(K.group stores) ~ranges () in
          List.fold_left (fun acc old -> (old, merged) :: acc) acc ends)
        by_ranges [] in
      (match mappings with
       | [] -> None
       | _ -> Some (K.substitute mappings node))
  | _ -> None

(* Fold ADD(WMMA, x) into WMMA's accumulator: WMMA(a, b, c+x). *)
let wmma_accumulate (node : K.t) : K.t option =
  match K.view node with
  | Binary { op = `Add; lhs; rhs; _ } ->
      let try_fold wmma other = match K.view wmma with
        | Wmma { a; b; c; _ } ->
            Some (K.replace wmma ~children:[a; b; K.O.( + ) c other] ())
        | _ -> None
      in
      (match try_fold lhs rhs with
       | Some _ as r -> r
       | None -> try_fold rhs lhs)
  | _ -> None

(* Insert Load for value-typed Index; collapse Store(Load(x), v) -> Store(x, v). *)
let add_loads_rule (node : K.t) : K.t option =
  match K.view node with
  | Index { dtype = Dtype.Val _; ptr; idxs; gate; _ } ->
      let ptr_pty = K.ptr_dtype ptr in
      let ptr_idx =
        K.index_raw ~ptr ~idxs ?gate ~dtype:(Dtype.Ptr ptr_pty) () in
      Some (K.load ~src:ptr_idx ())
  | Index { dtype = Dtype.Ptr _; _ } -> None
  | Store { dst; value; ranges } -> (
      match K.view dst with
      | Load { src; _ } -> Some (K.store ~dst:src ~value ~ranges)
      | _ -> None)
  | _ -> None

(* Passes *)

let pm_reduce (root : K.t) : K.t =
  let ctx = { acc_num = 0 } in
  K.graph_rewrite ~name:"remove_reduce"
    (K.first_match [
      reduce_to_acc ctx; wmma_accumulate;
      merge_reduce_ends ctx;
      Symbolic.gep_pushing;
    ]) root

let pm_add_loads (root : K.t) : K.t =
  K.graph_rewrite ~name:"** add loads (code)" add_loads_rule root

let pm_devectorize (ren : Renderer.t) (root : K.t) : K.t =
  K.graph_rewrite ~name:"devectorize"
    (K.first_match [
      Symbolic.sym;
      cast_after_after; no_vectorized_alu; no_vectorized_wmma;
      no_vectorized_buf; no_vectorized_index;
      expand_index; fold_expanded_index;
      gep_after_load; gep_on_store;
      ptrcat_after_load; ptrcat_after_store;
      split_load_store ren;
      load_store_indexing;
    ]) root

let pm_render_rule =
  K.first_match [
    expand_vector_const; expand_vconst; expand_multi_gep; trivial_gep;
    trivial_vectorize; masked_load_alt; where_after_gated_load;
  ]

let pm_render (root : K.t) : K.t =
  K.graph_rewrite pm_render_rule root
