(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Port of tinygrad/codegen/__init__.py::full_rewrite_to_sink to tolk_uop.
   Runs every pass from "postopt symbolic" through "add control flow". *)

open Tolk_uop
module U = Uop
module PM = Upat.Pattern_matcher

(* Helpers *)

let prod = List.fold_left ( * ) 1

let int_ n = U.const (Const.int Dtype.Val.weakint n)

(* [Invalid] test on a Const uop. *)
let is_invalid_const u =
  match U.op u, U.arg u with
  | Ops.Const, U.Arg.Value c -> Const.view c = Const.Invalid
  | _ -> false

let invalid_where_value idx =
  match U.op idx, U.src idx with
  | Ops.Where, [| valid; value; invalid |] when is_invalid_const invalid ->
      Some (valid, value, invalid)
  | _ -> None

let lane src i = U.index ~ptr:src ~idxs:[ int_ i ] ()

let lanes src idxs =
  match idxs with
  | [ i ] -> lane src i
  | _ -> U.stack (List.map (lane src) idxs)

let index_lanes u =
  match U.as_index u with
  | Some { ptr; idxs } ->
      let rec loop acc = function
      | [] -> Some (ptr, List.rev acc)
      | idx :: idxs -> (
          match U.const_int_value idx with
          | Some i -> loop (i :: acc) idxs
          | None -> None)
      in
      loop [] idxs
  | None -> None

(* Tail of [src u] as a list ([u]'s children past the first). *)
let src_tail u =
  let s = U.src u in
  Array.sub s 1 (Array.length s - 1) |> Array.to_list

let scalar_index_exn where = function
  | [ idx ] -> idx
  | _ -> invalid_arg (where ^ ": expected scalar INDEX")

let rec strip_index_casts idx =
  match U.op idx, U.src idx with
  | Ops.Cast, [| inner |]
    when Dtype.is_int (U.dtype inner) || Dtype.is_bool (U.dtype inner) ->
      Some inner
  | Ops.Where, [| gate; value; invalid |] when is_invalid_const invalid -> (
      match strip_index_casts value with
      | None -> None
      | Some value ->
          let invalid =
            match U.dtype value with
            | Dtype.Val dtype -> U.invalid ~dtype ()
            | Dtype.Ptr _ -> invalid
          in
          Some (U.alu_ternary ~op:Ops.Where ~a:gate ~b:value ~c:invalid))
  | Ops.Stack, srcs ->
      let changed = ref false in
      let srcs =
        Array.map
          (fun src ->
            match strip_index_casts src with
            | None -> src
            | Some src ->
                changed := true;
                src)
          srcs
      in
      if !changed then Some (U.stack (Array.to_list srcs)) else None
  | _ -> None

let load_store_indexing : Upat.Pattern_matcher.t =
  let open Upat in
  Pattern_matcher.make [
    op ~name:"idx" Ops.Index
    => fun bs ->
         let idx_node = bs $ "idx" in
         match U.as_index idx_node with
         | None -> None
         | Some { ptr; idxs } ->
             let changed = ref false in
             let idxs =
               List.map
                 (fun idx ->
                   match strip_index_casts idx with
                   | None -> idx
                   | Some idx ->
                       changed := true;
                       idx)
                 idxs
             in
             if !changed then
               Some
                 (U.index ~ptr ~idxs
                    ~as_ptr:(Dtype.is_ptr (U.dtype idx_node)) ())
             else None;
  ]

(* load/store grouping *)

(* Expand [Index(Stack(buf, ...), vec)] into a stack of per-lane indexes. *)
let expand_index =
  let open Upat in
  op ~name:"idx" Ops.Index
  => fun bs ->
       let idx = bs $ "idx" in
       match U.as_index idx with
       | Some { ptr; idxs = [ vec ] } when U.op ptr = Ops.Stack ->
           let ptrs = U.src ptr in
           let n = Dtype.count (U.dtype vec) in
           if Array.length ptrs <> n || n = 0 then None
           else
             Some
               (U.stack
                  (List.init n (fun i ->
                       U.index ~ptr:ptrs.(i)
                         ~idxs:[ lane vec i ]
                         ~as_ptr:true ())))
       | _ -> None

let ptr_load_dtype ptr =
  match U.dtype ptr with
  | Dtype.Ptr p -> Dtype.Val (Dtype.Ptr.value p)
  | Dtype.Val _ -> invalid_arg "ptr_load_dtype: expected pointer"

let ptr_load_width ptr =
  match U.dtype ptr with
  | Dtype.Ptr p -> Dtype.Val.count (Dtype.Ptr.value p)
  | Dtype.Val _ -> invalid_arg "ptr_load_width: expected pointer"

let chunk_tail offset width s =
  if Dtype.vcount (U.dtype s) > 1 then
    lanes s (List.init width (fun i -> offset + i))
  else s

let load_stack =
  let open Upat in
  op ~name:"ld" Ops.Load
  => fun bs ->
       let ld = bs $ "ld" in
       match U.as_load ld with
       | Some { src; _ } when U.op src = Ops.Stack ->
           let offset = ref 0 in
           let tail = src_tail ld in
           let loads =
             Array.to_list (U.src src)
             |> List.map (fun ptr ->
                    let width = ptr_load_width ptr in
                    let srcs =
                      ptr :: List.map (chunk_tail !offset width) tail
                    in
                    offset := !offset + width;
                    U.replace ld ~src:(Array.of_list srcs)
                      ~dtype:(ptr_load_dtype ptr) ())
           in
           Some (U.stack loads)
       | _ -> None

let load_store_folding : Upat.Pattern_matcher.t =
  Upat.Pattern_matcher.make
    [
      expand_index;
      load_stack;
    ]

(* expander: devectorize *)

(* Unpack WMMA: replace each non-Stack source with an explicit Stack of its
   scalar elements. *)
let do_stack_wmma =
  let open Upat in
  op ~name:"wmma" Ops.Wmma
  => fun bs ->
       let wmma = bs $ "wmma" in
       let srcs = U.src wmma in
       if
         Array.for_all (fun x -> U.op x = Ops.Stack || U.op x = Ops.Wmma) srcs
       then None
       else begin
         assert (List.length (U.shape wmma) = 1);
         let stacked b =
           if U.op b = Ops.Stack then b
           else U.stack (List.init (prod (U.max_shape b)) (lane b))
         in
         Some (U.replace wmma ~src:(Array.map stacked srcs) ())
       end

(* Scalarize vectorised ALU / Cast / Bitcast / Where by extracting each
   lane via value INDEX and re-vectorising. *)
let no_vectorized_alu_rule alu =
  let dt = U.dtype alu in
  let vc = Dtype.vcount dt in
  if vc <= 1 then None
  else
    let children = Array.to_list (U.src alu) in
    let scalar_dt = Dtype.scalarize dt in
    let lane i =
      let src_arr =
        List.map (fun s -> lane s i) children
        |> Array.of_list
      in
      U.replace alu ~src:src_arr ~dtype:scalar_dt ()
    in
    Some (U.stack (List.init vc lane))

let no_vectorized_alu =
  let open Upat in
  ops ~name:"alu" (Ops.Cast :: Ops.Bitcast :: Ops.Group.alu)
  => fun bs -> no_vectorized_alu_rule (bs $ "alu")

let all_same_shape src =
  match Array.to_list src with
  | [] -> true
  | first :: rest ->
      let shape = U.shape first in
      List.for_all
        (fun u ->
          let shape' = U.shape u in
          List.length shape = List.length shape'
          && List.for_all2 U.equal shape shape')
        rest

let compatible_devectorize_shape node shape =
  match U.op node with
  | Ops.Store ->
      Array.for_all
        (fun src ->
          match try Some (U.shape src) with Invalid_argument _ -> None with
          | None -> false
          | Some src_shape ->
              src_shape = []
              || (List.length shape = List.length src_shape
                  && List.for_all2 U.equal (List.map int_ shape) src_shape))
        (U.src node)
  | _ -> all_same_shape (U.src node)

let shape_ints u =
  let rec loop acc = function
  | [] -> Some (List.rev acc)
  | d :: ds -> (
      match U.const_int_value d with
      | Some d -> loop (d :: acc) ds
      | None -> None)
  in
  try loop [] (U.shape u) with Invalid_argument _ -> None

let shape_numel shape =
  List.fold_left (fun acc dim -> acc * U.vmax dim) 1 shape

let index_for_shape src idxs =
  match U.dtype src with
  | Dtype.Ptr _ ->
      Some
        (U.index ~ptr:src ~idxs:(List.map int_ idxs) ~as_ptr:true ())
  | Dtype.Val _ ->
      let shape = shape_ints src in
      if Option.is_none shape then None
      else
        let shape = Option.get shape in
        if List.length shape <> List.length idxs then None
        else Some (U.index ~ptr:src ~idxs:(List.map int_ idxs) ())

let is_global_shrink u =
  U.op u = Ops.Shrink
  &&
  match U.src u with
  | [| src; _; _ |] -> U.addrspace src = Some Dtype.Global
  | _ -> false

let do_devectorize node =
  match shape_ints node with
  | None | Some [] -> None
  | Some _ when not (all_same_shape (U.src node)) -> None
  | Some shape ->
      let rec product = function
      | [] -> [ [] ]
      | n :: ns ->
          List.concat_map
            (fun i -> List.map (fun tail -> i :: tail) (product ns))
            (List.init n Fun.id)
      in
      let lanes =
        product shape
        |> List.map (fun idxs ->
               let src =
                 Array.to_list (U.src node)
                 |> List.map (fun src -> index_for_shape src idxs)
               in
               if List.exists Option.is_none src then None
               else
                 Some
                   (U.replace node
                      ~src:
                        (Array.of_list
                           (List.map
                              (function Some u -> u | None -> assert false)
                              src))
                      ()))
      in
      if List.exists Option.is_none lanes then None
      else
        let lanes =
          List.map (function Some u -> u | None -> assert false) lanes
        in
        if U.op node = Ops.Store then Some (U.group lanes)
        else
          Some
            (U.reshape ~src:(U.stack lanes)
               ~shape:(U.stack (List.map int_ shape)))

let devectorizer2_rule node =
  let op = U.op node in
  let keep_vector_store =
    match U.as_store node with
    | Some { dst; _ } -> is_global_shrink dst
    | _ -> false
  in
  if keep_vector_store then None
  else
  if Ops.Group.is_elementwise op || op = Ops.Load || op = Ops.Store then
    do_devectorize node
  else None

let index_stack_reshape_rule node =
  match U.op node, U.src node with
  | Ops.Index, srcs when Array.length srcs >= 2 -> (
      match U.op srcs.(0) with
      | Ops.Reshape
        when Array.length srcs = 2
             && U.const_int_value srcs.(1) = Some 0 -> (
          match U.src srcs.(0) with
          | [| x; _ |] when shape_ints x = Some [] -> Some x
          | _ -> None)
      | Ops.Stack -> (
          match U.const_int_value srcs.(1) with
          | Some i when i >= 0 && i < Array.length (U.src srcs.(0)) ->
              let lane = (U.src srcs.(0)).(i) in
              let rest =
                Array.sub srcs 2 (Array.length srcs - 2) |> Array.to_list
              in
              Some (U.index ~ptr:lane ~idxs:rest ())
          | _ -> None)
      | Ops.Index ->
          let inner = srcs.(0) in
          let inner_srcs = Array.to_list (U.src inner) in
          let outer =
            Array.sub srcs 1 (Array.length srcs - 1) |> Array.to_list
          in
          let idxs =
            match List.tl inner_srcs, outer, U.dtype inner with
            | [ base ], [ idx ], Dtype.Ptr _ -> [ U.O.(base + idx) ]
            | inner_idxs, outer, _ -> inner_idxs @ outer
          in
          Some
            (U.index ~ptr:(List.hd inner_srcs) ~idxs
               ~as_ptr:(Dtype.is_ptr (U.dtype node)) ())
      | Ops.Shrink -> (
          match U.src srcs.(0), Array.sub srcs 1 (Array.length srcs - 1) |> Array.to_list with
          | [| ptr; offset; _size |], [ idx ] ->
              Some (U.index ~ptr ~idxs:[ U.O.(offset + idx) ] ())
          | _ -> None)
      | Ops.Param | Ops.Buffer -> (
          match U.op srcs.(1) with
          | Ops.Stack
            when Array.length srcs = 2
                 && (match U.dtype srcs.(0) with
                     | Dtype.Ptr p -> not (Dtype.Ptr.is_image p)
                     | Dtype.Val _ -> true) ->
              let lanes =
                U.src srcs.(1)
                |> Array.to_list
                |> List.map (fun idx -> U.index ~ptr:srcs.(0) ~idxs:[ idx ] ())
              in
              Some (U.stack lanes)
          | Ops.Reshape when Array.length srcs = 2 -> (
              match U.src srcs.(1) with
              | [| inner_idx; shape |] ->
                  let indexed =
                    U.index ~ptr:srcs.(0) ~idxs:[ inner_idx ]
                      ~as_ptr:(Dtype.is_ptr (U.dtype node)) ()
                  in
                  Some (U.reshape ~src:indexed ~shape)
              | _ -> None)
          | _ -> None)
      | _ -> None)
  | Ops.Index, [| x |] -> Some x
  | _ -> None

let reshape_expand_to_stack_rule node =
  match U.op node, U.src node with
  | Ops.Expand, [| reshape; _shape |] when U.op reshape = Ops.Reshape -> (
      match try Some (U.shape node) with Invalid_argument _ -> None with
      | None -> None
      | Some out_shape ->
          let max_numel =
            List.fold_left (fun acc dim -> acc * U.vmax dim) 1 out_shape
          in
          (match out_shape, U.src reshape with
           | [ dim ], [| x; _ |] when U.vmax dim = max_numel ->
               Some (U.stack (List.init max_numel (fun _ -> x)))
           | _ -> None))
  | _ -> None

let reshape_singleton_to_scalar_rule node =
  match U.op node, U.src node with
  | Ops.Reshape, [| src; _shape |]
    when shape_ints node = Some [] && shape_ints src = Some [ 1 ] ->
      Some (U.index ~ptr:src ~idxs:[ int_ 0 ] ())
  | _ -> None

let devectorizer2 =
  let open Upat in
  ops ~name:"node" (Ops.Group.elementwise @ [ Ops.Load; Ops.Store ])
  => fun bs -> devectorizer2_rule (bs $ "node")

let index_stack_reshape =
  let open Upat in
  op ~name:"node" Ops.Index
  => fun bs -> index_stack_reshape_rule (bs $ "node")

let reshape_expand_to_stack =
  let open Upat in
  op ~name:"node" Ops.Expand
  => fun bs -> reshape_expand_to_stack_rule (bs $ "node")

let reshape_singleton_to_scalar =
  let open Upat in
  op ~name:"node" Ops.Reshape
  => fun bs -> reshape_singleton_to_scalar_rule (bs $ "node")

let store_singleton_value_rule node =
  match U.as_store node with
  | Some { dst; value; gate } -> (
      match U.op value, U.src value with
      | Ops.Reshape, [| scalar; _shape |]
        when shape_ints value = Some [ 1 ] && shape_ints scalar = Some [] ->
          Some (U.store ~dst ~value:scalar ?gate ())
      | _ -> None)
  | None -> None

let store_singleton_value =
  let open Upat in
  op ~name:"node" Ops.Store
  => fun bs -> store_singleton_value_rule (bs $ "node")

(* Scalarise local/register [Buffer] nodes with a vector base: widen storage by
   [count], scalarise the base, cast back to the original pointer type. *)
let no_vectorized_buf_rule buf =
  match U.dtype buf with
  | Dtype.Ptr p
    when Dtype.Val.count (Dtype.Ptr.base p) > 1
         && (Dtype.Ptr.addrspace p = Dtype.Local
             || Dtype.Ptr.addrspace p = Dtype.Reg) ->
      let cnt = Dtype.Val.count (Dtype.Ptr.base p) in
      let scalar_base = Dtype.Val.scalarize (Dtype.Ptr.base p) in
      let new_pty =
        Dtype.Ptr.with_size
          (Dtype.Ptr.size p * cnt)
          (Dtype.Ptr.with_base scalar_base p)
      in
      let new_buf =
        match U.as_buffer buf with
        | Some { buffer; shape = _ } ->
            Some
              (U.buffer ~slot:buffer.slot ~dtype:(Dtype.Ptr new_pty)
                 ~shape:(U.const_int (Dtype.Ptr.size new_pty))
                 ?name:buffer.name ~addrspace:buffer.addrspace
                 ?axis:buffer.axis ?device:buffer.device ())
        | None -> None
      in
      Option.map (fun b -> U.cast ~src:b ~dtype:(Dtype.Ptr p)) new_buf
  | _ -> None

let no_vectorized_buf =
  let open Upat in
  op ~name:"buf" Ops.Buffer
  => fun bs -> no_vectorized_buf_rule (bs $ "buf")

let flat_index_for_shape idxs shape =
  let rec go stride acc idxs shape =
    match idxs, shape with
    | [], [] -> acc
    | idx :: idxs, dim :: shape ->
        let term =
          if U.equal stride (int_ 1) then idx else U.O.(idx * stride)
        in
        go U.O.(stride * dim) (term :: acc) idxs shape
    | _ -> invalid_arg "flat_index_for_shape: rank mismatch"
  in
  match go (int_ 1) [] (List.rev idxs) (List.rev shape) with
  | [] -> int_ 0
  | term :: terms ->
      List.fold_left (fun acc term -> U.O.(acc + term)) term terms

let const_int_node node =
  match U.op node, U.arg node with
  | Ops.Const, U.Arg.Value c -> (
      match Const.view c with
      | Const.Int i -> Some (Int64.to_int i)
      | Const.Bool _ | Const.Float _ | Const.Invalid -> None)
  | _ -> None

let rec split_add_const node =
  match U.op node, U.src node with
  | Ops.Add, [| a; b |] -> (
      match const_int_node b with
      | Some c ->
          let base, offset = split_add_const a in
          (base, offset + c)
      | None -> (
          match const_int_node a with
          | Some c ->
              let base, offset = split_add_const b in
              (base, offset + c)
          | None -> (node, 0)))
  | _ -> (node, 0)

let rec guarded_upper_bound base gate =
  match U.op gate, U.src gate with
  | Ops.And, [| a; b |] -> (
      match guarded_upper_bound base a, guarded_upper_bound base b with
      | Some a, Some b -> Some (max a b)
      | Some _ as bound, None | None, (Some _ as bound) -> bound
      | None, None -> None)
  | Ops.Cmplt, [| lhs; rhs |] when U.compare_structure lhs base = 0 ->
      const_int_node rhs
  | Ops.Cmplt, [| _; rhs |] -> const_int_node rhs
  | _ -> None

let index_lane_bounds idx =
  let idx =
    match U.op idx, U.src idx with
    | Ops.Where, [| gate; value; invalid |] when is_invalid_const invalid ->
        let base, offset = split_add_const value in
        (match guarded_upper_bound base gate with
         | Some upper -> U.const_int (upper + offset)
         | None -> value)
    | Ops.Where, [| gate; value; _ |] ->
        let base, offset = split_add_const value in
        (match guarded_upper_bound base gate with
         | Some upper -> U.const_int (upper + offset)
         | None -> value)
    | _ -> idx
  in
  let lo = U.vmin idx and hi = U.vmax idx in
  if lo = 0 && hi >= 0 then Some (int_ (hi + 1)) else None

let infer_flat_param_shape ptr idxs flat_shape =
  let infer_dims_from_flat flat_size =
    let bounds = List.map index_lane_bounds idxs in
    let known =
      List.fold_left
        (fun acc -> function
          | Some dim -> (
              match acc, U.const_int_value dim with
              | Some acc, Some dim -> Some (acc * dim)
              | _ -> None)
          | None -> acc)
        (Some 1) bounds
    in
    match known with
    | None -> None
    | Some known when known = 0 || flat_size mod known <> 0 -> None
    | Some known ->
        let missing = List.length (List.filter Option.is_none bounds) in
        let remaining = flat_size / known in
        let fill =
          match missing with
          | 0 when remaining = 1 -> Some []
          | 1 -> Some [ int_ remaining ]
          | 2 ->
              let root = int_of_float (sqrt (float_of_int remaining)) in
              if root * root = remaining then Some [ int_ root; int_ root ]
              else None
          | _ -> None
        in
        Option.map
          (fun fill ->
            let fill = ref fill in
            List.map
              (function
                | Some dim -> dim
                | None -> (
                    match !fill with
                    | dim :: rest ->
                        fill := rest;
                        dim
                    | [] -> assert false))
              bounds)
          fill
  in
  match flat_shape with
  | [ flat ] -> (
      match U.const_int_value flat with
      | None -> None
      | Some flat_size ->
          let dims = List.filter_map index_lane_bounds idxs in
          if List.length dims <> List.length idxs then
            infer_dims_from_flat flat_size
          else
            let size =
              List.fold_left
                (fun acc dim ->
                  match acc, U.const_int_value dim with
                  | Some acc, Some dim -> Some (acc * dim)
                  | _ -> None)
                (Some 1) dims
            in
            (match size with
             | Some size when size = flat_size -> Some dims
             | _ -> infer_dims_from_flat flat_size))
  | _ -> (
      match U.dtype ptr with
      | Dtype.Ptr p when Dtype.Ptr.size p >= 0 ->
          let flat_size = Dtype.Ptr.size p in
          let dims = List.filter_map index_lane_bounds idxs in
          if List.length dims <> List.length idxs then
            infer_dims_from_flat flat_size
          else
            let size =
              List.fold_left
                (fun acc dim ->
                  match acc, U.const_int_value dim with
                  | Some acc, Some dim -> Some (acc * dim)
                  | _ -> None)
                (Some 1) dims
            in
            (match size with
             | Some size when size = flat_size -> Some dims
             | _ -> infer_dims_from_flat flat_size)
      | _ -> None)

let repeated_flat_index_shape idxs flat_size =
  let dims = List.map (fun idx -> U.vmax idx + 1) idxs in
  let total = prod dims in
  let shape dims = List.map int_ dims in
  if total = flat_size then Some (idxs, shape dims)
  else
    match idxs with
    | first :: _ when U.vmax first + 1 = flat_size -> Some ([ first ], [ int_ flat_size ])
    | _ when total > flat_size && flat_size > 0 && total mod flat_size = 0 ->
        let ratio = ref (total / flat_size) in
        let pairs =
          List.rev (List.combine idxs dims)
          |> List.rev_map (fun (idx, dim) ->
                 if !ratio = 1 then (idx, dim)
                 else
                   let rec divisor d =
                     if d <= 1 then 1
                     else if !ratio mod d = 0 && dim mod d = 0 then d
                     else divisor (d - 1)
                   in
                   let d = divisor (min !ratio dim) in
                   ratio := !ratio / d;
                   let idx =
                     if d = 1 then idx else U.O.(idx // int_ d)
                   in
                   (idx, dim / d))
        in
        if !ratio = 1 then
          let idxs, dims = List.split pairs in
          Some (idxs, shape dims)
        else None
    | _ -> None

let sliding_window_index_shape idxs flat_size =
  match idxs with
  | [ n; c; ky; kx; oy; ox ] -> (
      match
        index_lane_bounds n, index_lane_bounds c, index_lane_bounds ky,
        index_lane_bounds kx, index_lane_bounds oy, index_lane_bounds ox
      with
      | ( Some n_dim,
          Some c_dim,
          Some ky_dim,
          Some kx_dim,
          Some oy_dim,
          Some ox_dim ) -> (
          match
            U.const_int_value n_dim, U.const_int_value c_dim,
            U.const_int_value ky_dim, U.const_int_value kx_dim,
            U.const_int_value oy_dim, U.const_int_value ox_dim
          with
          | ( Some n_sz,
              Some c_sz,
              Some ky_sz,
              Some kx_sz,
              Some oy_sz,
              Some ox_sz ) ->
              let iy_sz = oy_sz + ky_sz - 1 in
              let ix_sz = ox_sz + kx_sz - 1 in
              if n_sz * c_sz * iy_sz * ix_sz = flat_size then
                let open U.O in
                Some ([ n; c; oy + ky; ox + kx ], [ n_sz; c_sz; iy_sz; ix_sz ])
              else None
          | _ -> None)
      | _ -> None)
  | _ -> None

let flatten_index_for_flat_size idxs flat_size =
  match sliding_window_index_shape idxs flat_size with
  | Some (idxs, shape) -> Some (idxs, List.map int_ shape)
  | None -> repeated_flat_index_shape idxs flat_size

(* Scalarize a vector [Index] on local/reg memory.  Three ptr shapes:
   1. [Cast(buf).index(idx)]              — plain scalar index.
   2. [Cast(buf).broadcast(b).index(idx)] — broadcast index.
   3. [Cast(buf).index(g).index(idx)]     — value-indexed lanes. *)
let no_vectorized_index_rule node =
  let rec is_local_or_reg n =
    match U.op n, U.src n with
    | Ops.After, srcs when Array.length srcs > 0 -> is_local_or_reg srcs.(0)
    | Ops.Buffer, _ -> (
        match U.dtype n with
        | Dtype.Ptr p ->
            Dtype.Ptr.addrspace p = Dtype.Local
            || Dtype.Ptr.addrspace p = Dtype.Reg
        | Dtype.Val _ -> false)
    | _ -> false
  in
  let check_cast n =
    match U.op n, U.src n with
    | Ops.Cast, [| buf |] -> (
        match U.dtype n with
        | Dtype.Ptr cp when is_local_or_reg buf -> Some (buf, cp)
        | _ -> None)
    | _ -> None
  in
  match U.as_index node with
  | Some { ptr; idxs; _ }
    when (match U.dtype node with
          | Dtype.Ptr p -> Dtype.Val.count (Dtype.Ptr.base p) > 1
          | Dtype.Val _ -> false)
          && idxs <> [] ->
      let idx =
        match idxs with
        | [ idx ] -> idx
        | _ -> (
            try
              let shape = U.shape ptr in
              if List.length shape = List.length idxs then
                flat_index_for_shape idxs shape
              else
                let size =
                  match U.dtype ptr with
                  | Dtype.Ptr p -> Dtype.Ptr.size p
                  | Dtype.Val _ -> -1
                in
                let idxs, shape =
                  match infer_flat_param_shape ptr idxs [ int_ size ] with
                  | Some shape -> (idxs, shape)
                  | None -> (
                      match flatten_index_for_flat_size idxs size with
                      | Some pair -> pair
                      | None ->
                          invalid_arg
                            "no_vectorized_index: cannot flatten variadic index")
                in
                flat_index_for_shape idxs shape
            with Invalid_argument _ ->
              invalid_arg "no_vectorized_index: cannot flatten variadic index")
      in
      let found =
        match U.op ptr, U.src ptr with
        | Ops.Cast, _ ->
            Option.map
              (fun (buf, cp) -> (buf, cp, `Plain))
              (check_cast ptr)
        | Ops.Stack, srcs when Array.length srcs > 0 ->
            Option.map
              (fun (buf, cp) -> (buf, cp, `Broadcast ptr))
              (check_cast srcs.(0))
        | Ops.Index, _ -> (
            match index_lanes ptr with
            | Some (inner, idxs) ->
                Option.map
                  (fun (buf, cp) -> (buf, cp, `Index idxs))
                  (check_cast inner)
            | None -> None)
        | _ -> None
      in
      (match found with
       | None -> None
       | Some (buf, cast_pty, shape) ->
           let cnt = Dtype.Val.count (Dtype.Ptr.base cast_pty) in
           let pairs =
             match shape with
             | `Index index_idxs ->
                 let vc = Dtype.Ptr.v cast_pty in
                 let n_index = List.length index_idxs in
                 let arr = Array.of_list index_idxs in
                 List.init (vc * n_index) (fun i ->
                     (i mod n_index, (i / n_index) + arr.(i mod n_index)))
             | `Broadcast bnode ->
                 let bvc = Dtype.vcount (U.dtype bnode) in
                 List.init (cnt * bvc) (fun i -> (i mod bvc, i / bvc))
             | `Plain -> List.init cnt (fun c -> (0, c))
           in
           let n = List.length pairs in
           let open U.O in
           let lane_sel = lanes idx (List.map fst pairs) in
           let stride = U.broadcast (int_ cnt) n in
           let off =
             U.stack (List.map (fun (_, o) -> int_ o) pairs)
           in
           let wide_idx = (lane_sel * stride) + off in
           Some
             (U.index ~ptr:(U.broadcast buf n) ~idxs:[wide_idx] ~as_ptr:true ()))
  | _ -> None

let no_vectorized_index =
  let open Upat in
  op ~name:"root" Ops.Index
  => fun bs -> no_vectorized_index_rule (bs $ "root")

(* Move [Cast] out of [After]: [After(Cast(x, dt), deps)] ->
   [Cast(After(x, deps), dt)]. *)
let cast_after_after =
  let open Upat in
  op ~name:"a"
    ~src:[ cast ~name:"c" any ]
    ~allow_any_len:true Ops.After
  => fun bs ->
       let c = bs $ "c" and a = bs $ "a" in
       let inner = (U.src c).(0) in
       Some
         (U.cast
            ~src:(U.after ~src:inner ~deps:(src_tail a))
            ~dtype:(U.dtype c))

let devectorize_buf_and_index : Upat.Pattern_matcher.t =
  Upat.Pattern_matcher.make [ no_vectorized_buf; no_vectorized_index ]

let devectorize : Upat.Pattern_matcher.t =
  Upat.Pattern_matcher.(
    make
      [
        cast_after_after;
        devectorizer2;
        index_stack_reshape;
        reshape_expand_to_stack;
        reshape_singleton_to_scalar;
        store_singleton_value;
        no_vectorized_alu;
        do_stack_wmma;
      ]
    ++ devectorize_buf_and_index)

(* pm_render *)

(* Expand a vector [Const] into a [Stack] of scalar copies. *)
let expand_vector_const =
  let open Upat in
  op ~name:"c" Ops.Const
  => fun bs ->
       let c = bs $ "c" in
       match U.dtype c with
       | Dtype.Val v when Dtype.Val.count v > 1 -> (
           match U.arg c with
           | U.Arg.Value value ->
               let n = Dtype.Val.count v in
               let scalar_c =
                 U.const
                   (Const.of_view (Dtype.Val.scalarize v) (Const.view value))
               in
               Some (U.stack (List.init n (fun _ -> scalar_c)))
           | _ -> None)
       | _ -> None

(* [Stack] of a single source -> the source. *)
let trivial_stack =
  let open Upat in
  op ~name:"v" ~src:[ var "x" ] Ops.Stack
  => fun bs -> Some (bs $ "x")

let flatten_variadic_index =
  let open Upat in
  op ~name:"idx" Ops.Index
  => fun bs ->
       let node = bs $ "idx" in
       match U.as_index node with
	       | Some { ptr; idxs = _ :: _ :: _ as idxs } -> (
	           match U.dtype ptr with
	           | Dtype.Ptr p when Dtype.Ptr.is_image p -> None
	           | _ ->
	               let flattened =
                 match try Some (U.shape ptr) with Invalid_argument _ -> None with
	                 | None -> None
	                 | Some ptr_shape ->
	                     if List.length ptr_shape = List.length idxs then
	                       Some (idxs, ptr_shape)
	                     else
	                       match infer_flat_param_shape ptr idxs ptr_shape with
	                       | Some shape -> Some (idxs, shape)
	                       | None -> (
	                           match ptr_shape with
	                           | [ flat ] -> (
	                               match U.const_int_value flat with
	                               | Some flat_size ->
	                                   flatten_index_for_flat_size idxs flat_size
	                               | None -> None)
	                           | _ -> None)
	               in
               (match flattened with
	                | Some (idxs, shape) ->
	                    let flat = flat_index_for_shape idxs shape in
	                    if Dtype.is_ptr (U.dtype node) then
	                      Some (U.index ~ptr ~idxs:[ flat ] ~as_ptr:true ())
	                    else
	                      Some
	                        (U.replace node ~src:(Array.of_list [ ptr; flat ]) ())
                | None -> None))
       | _ -> None

let pm_flatten_indexes root =
  let matcher =
    Upat.Pattern_matcher.(make [ flatten_variadic_index ] ++ load_store_folding)
  in
  U.graph_rewrite ~name:"flatten variadic indexes"
    (fun node -> Upat.Pattern_matcher.rewrite matcher node)
    root

let pm_render : Upat.Pattern_matcher.t =
  Upat.Pattern_matcher.(
    make
      [
        expand_vector_const;
        trivial_stack;
      ]
    ++ load_store_folding
  )

(* Ops.Reduce -> register Buffer accumulator *)

(* Identity element for a reduction op at [dtype]. *)
let identity_element op dtype =
  match op with
  | Ops.Add -> Const.zero dtype
  | Ops.Mul -> Const.one dtype
  | Ops.Max -> Const.min_value dtype
  | _ -> invalid_arg "identity_element: unsupported reduce op"

(* Split horizontal reduction lanes when input is wider than output. *)
let horizontal_reduce (inp : U.t) (out_dtype : Dtype.t) : U.t list =
  let inp_dt = U.dtype inp in
  if Dtype.equal inp_dt out_dtype then [ inp ]
  else
    let amount = Dtype.count inp_dt / Dtype.count out_dtype in
    List.init amount (fun i ->
        lanes inp
          (List.init (Dtype.count out_dtype) (fun j ->
               i + (j * amount))))

(* Fold a non-empty list with a binary op. *)
let reduce_fold op = function
  | [] -> invalid_arg "reduce_fold: empty list"
  | first :: rest ->
      List.fold_left
        (fun a x -> U.alu_binary ~op ~lhs:a ~rhs:x)
        first rest

let expand_horizontal_reduce_rule node =
  match U.as_reduce node with
  | Some { src; ranges = []; op; axes = _ :: _ as axes } ->
      let src_shape = U.shape src in
      let rank = List.length src_shape in
      if List.exists (fun axis -> axis < 0 || axis >= rank) axes then None
      else
        let permute =
          axes
          @ (List.init rank Fun.id
             |> List.filter (fun axis -> not (List.mem axis axes)))
        in
        let inp =
          if List.mapi (fun i _ -> i) src_shape = permute then src
          else U.permute ~src ~order:permute
        in
        let rec product = function
        | [] -> [ [] ]
        | axis :: axes ->
            let tail = product axes in
            List.concat
              (List.init (U.vmax (List.nth (U.shape inp) axis)) (fun i ->
                   List.map (fun idx -> U.const_int i :: idx) tail))
        in
        Some
          (reduce_fold op
             (List.map
                (fun idxs -> U.index ~ptr:inp ~idxs ())
                (product (List.init (List.length axes) Fun.id))))
  | Some _ | None -> None

let expand_horizontal_reduce =
  let open Upat in
  op ~name:"red" Ops.Reduce
  => fun bs -> expand_horizontal_reduce_rule (bs $ "red")

type reduce_ctx = {
  mutable acc_num : int;
  acc_slots : int U.Ref_tbl.t;
}

let reduce_slots_in_tinygrad_order root =
  let replaced = U.Ref_tbl.create 128 in
  let on_stack = U.Ref_tbl.create 128 in
  let waitlist = U.Ref_tbl.create 32 in
  let order = ref [] in
  let stack = ref [ (root, 0) ] in
  U.Ref_tbl.replace on_stack root ();
  let push_wait dep item =
    let items =
      match U.Ref_tbl.find_opt waitlist dep with
      | Some items -> item :: items
      | None -> [ item ]
    in
    U.Ref_tbl.replace waitlist dep items
  in
  let release_waiters n =
    match U.Ref_tbl.find_opt waitlist n with
    | None -> ()
    | Some items ->
        U.Ref_tbl.remove waitlist n;
        stack := items @ !stack
  in
  let rec first_missing srcs i =
    if i = Array.length srcs then None
    else if U.Ref_tbl.mem replaced srcs.(i) then first_missing srcs (i + 1)
    else Some srcs.(i)
  in
  while !stack <> [] do
    match !stack with
    | [] -> ()
    | (node, stage) :: rest ->
        stack := rest;
        if not (U.Ref_tbl.mem replaced node) then
          if stage = 0 then begin
            stack := (node, 1) :: !stack;
            let srcs = U.src node in
            let enter =
              match U.op node with Ops.Call | Ops.Function -> false | _ -> true
            in
            for i = Array.length srcs - 1 downto 0 do
              let child = srcs.(i) in
              if (i <> 0 || enter) && not (U.Ref_tbl.mem on_stack child) then begin
                stack := (child, 0) :: !stack;
                U.Ref_tbl.replace on_stack child ()
              end
            done
          end
          else
            let srcs = U.src node in
            match first_missing srcs 0 with
            | Some dep -> push_wait dep (node, 1)
            | None ->
                (match U.as_reduce node with
                | Some { ranges = _ :: _; _ } -> order := node :: !order
                | Some { ranges = []; _ } | None -> ());
                U.Ref_tbl.replace replaced node ();
                release_waiters node
  done;
  let slots = U.Ref_tbl.create 16 in
  List.rev !order
  |> List.iteri (fun slot node -> U.Ref_tbl.replace slots node slot);
  slots

let reduce_input_ranges src reduce_range =
  let rec ended_ranges u =
    let children = U.src u in
    match U.op u with
    | Ops.After ->
        let ret = ref [] in
        for i = 1 to Array.length children - 1 do
          ret := ended_ranges children.(i) @ !ret
        done;
        !ret
    | _ ->
        let start =
          match U.op u with
          | Ops.Stage | Ops.Reduce | Ops.End | Ops.Call | Ops.Function
          | Ops.Copy ->
              Some 1
          | Ops.Slice -> Some 2
          | Ops.Wmma -> Some 3
          | Ops.Linear -> Some 0
          | _ -> None
        in
        (match start with
        | None -> []
        | Some k ->
            let ret = ref [] in
            for i = k to Array.length children - 1 do
              ret := children.(i) :: !ret
            done;
            !ret)
  in
  let topo = U.toposort src in
  let ended = U.Ref_tbl.create 16 in
  List.iter
    (fun n ->
      List.iter (fun r -> U.Ref_tbl.replace ended r ()) (ended_ranges n))
    topo;
  let reduce_set = U.Ref_tbl.create 8 in
  List.iter (fun r -> U.Ref_tbl.replace reduce_set r ()) reduce_range;
  List.filter
    (fun n ->
      U.op n = Ops.Range
      && (not (U.Ref_tbl.mem reduce_set n))
      && not (U.Ref_tbl.mem ended n))
    topo

let shape_arg_of_ints dims = U.stack (List.map U.const_int dims)

let reg_placeholder_like node ~slot =
  let dtype =
    match U.dtype node with
    | Dtype.Val dtype -> dtype
    | Dtype.Ptr _ -> invalid_arg "reg_placeholder_like: expected value dtype"
  in
  let shape = U.max_shard_shape node in
  let count = Dtype.Val.count dtype in
  let storage_shape =
    [ prod shape ] @ if count > 1 then [ count ] else []
  in
  let buf =
    U.buffer ~slot ~dtype:(Dtype.Val dtype)
      ~shape:(shape_arg_of_ints storage_shape) ~addrspace:Dtype.Reg ()
  in
  if List.length shape > 1 then
    let view_shape = shape @ if count > 1 then [ count ] else [] in
    U.reshape ~src:buf ~shape:(shape_arg_of_ints view_shape)
  else buf

(* A Stage reaching codegen becomes a local (or reg) placeholder written by
   an ended, barriered store; readers keep their multi-range INDEX into the
   shaped view of the flat buffer. Mirrors [add_local_buffer]. *)
let add_local_buffer_rule counter node =
  match U.as_stage node with
  | None -> None
  | Some { src; ranges; opts } ->
      let dtype =
        match U.dtype node with
        | Dtype.Val dtype -> dtype
        | Dtype.Ptr _ -> invalid_arg "add_local_buffer: expected value dtype"
      in
      let shape = U.max_shape node in
      let count = Dtype.Val.count dtype in
      let vec_tail = if count > 1 then [ count ] else [] in
      let slot = !counter in
      incr counter;
      let buf =
        U.buffer ~slot ~dtype:(Dtype.Val dtype)
          ~shape:(shape_arg_of_ints ([ prod shape ] @ vec_tail))
          ~addrspace:opts.addrspace ()
      in
      let buf =
        if List.length shape > 1 then
          U.reshape ~src:buf ~shape:(shape_arg_of_ints (shape @ vec_tail))
        else buf
      in
      let store =
        U.store ~dst:(U.index ~ptr:buf ~idxs:ranges ()) ~value:src ()
      in
      Some
        (U.after ~src:buf
           ~deps:[ U.barrier ~srcs:[ U.end_ ~value:store ~ranges ] () ])

let reduce_to_acc_rule (ctx : reduce_ctx) node =
  match U.as_reduce node with
  | None -> None
	  | Some v ->
	      let reduce_range = v.ranges in
	      let dtype_v = Dtype.val_of (U.dtype node) in
      let lst = horizontal_reduce v.src (U.dtype node) in
      if reduce_range = [] then Some (reduce_fold v.op lst)
      else
        let input_ranges = reduce_input_ranges v.src reduce_range in
        let identity =
          U.const (identity_element v.op (Dtype.Val.scalarize dtype_v))
        in
        let slot =
          match U.Ref_tbl.find_opt ctx.acc_slots node with
          | Some slot ->
              ctx.acc_num <- max ctx.acc_num (slot + 1);
              slot
          | None ->
              let slot = ctx.acc_num in
              ctx.acc_num <- ctx.acc_num + 1;
              slot
        in
        let acc = reg_placeholder_like node ~slot in
        let acc_after_input =
          match input_ranges with
          | [] -> acc
          | deps -> U.after ~src:acc ~deps
        in
        let acc_init = U.store ~dst:acc_after_input ~value:identity () in
        let acc_in_loop =
          U.after ~src:acc ~deps:(acc_init :: reduce_range)
        in
        let ret = reduce_fold v.op (acc_in_loop :: lst) in
        let store_back = U.store ~dst:acc_in_loop ~value:ret () in
        let end_node =
          U.with_tag "mergeable"
            (U.end_ ~value:store_back ~ranges:reduce_range)
        in
        Some (U.after ~src:acc ~deps:[ end_node ])

let reduce_to_acc ctx =
  let open Upat in
  op ~name:"red" ~allow_any_len:true Ops.Reduce
  => fun bs -> reduce_to_acc_rule ctx (bs $ "red")

let reduce_invalid_to_identity_rule node =
  match U.as_reduce node with
  | Some { src; ranges; op; _ } -> (
      match invalid_where_value src with
      | None -> None
      | Some (valid, value, _invalid) ->
          let value_dtype = Dtype.val_of (U.dtype value) in
          let scalar_identity =
            U.const (identity_element op (Dtype.Val.scalarize value_dtype))
          in
          let identity =
            if Dtype.Val.count value_dtype = 1 then scalar_identity
            else U.broadcast scalar_identity (Dtype.Val.count value_dtype)
          in
          let src = U.O.where valid value identity in
          Some
            (U.reduce ~op ~src ~ranges
               ~dtype:(Dtype.val_of (U.dtype node))))
  | None -> None

let reduce_invalid_to_identity =
  let open Upat in
  op ~name:"red" ~allow_any_len:true Ops.Reduce
  => fun bs -> reduce_invalid_to_identity_rule (bs $ "red")

(* Merge [End] nodes that share the same ranges (created by
   [reduce_to_acc]).  Keeps diagnostic tag [mergeable] as the sentinel. *)
let merge_reduce_ends_rule node =
  match U.op node with
  | Ops.Sink ->
      let topo = U.toposort node in
      let range_groups = ref [] in
      let same_ranges a b =
        List.length a = List.length b && List.for_all2 U.equal a b
      in
      let add_end ranges e =
        match List.find_opt (fun (ranges', _) -> same_ranges ranges ranges')
                !range_groups with
        | Some (_, ends) -> ends := e :: !ends
        | None -> range_groups := !range_groups @ [ (ranges, ref [ e ]) ]
      in
      List.iter
        (fun n ->
          match U.as_end n with
          | Some { ranges; _ } when U.node_tag n = Some "mergeable" ->
              add_end ranges n
          | _ -> ())
        (U.backward_slice node);
      let next_axis =
        List.fold_left
          (fun acc n ->
            match U.as_range n with
            | Some { axis; _ } -> max acc (axis + 1)
            | None -> acc)
          0 topo
        |> ref
      in
      let clone_ranges ranges =
        let base = !next_axis in
        next_axis := !next_axis + List.length ranges;
        List.mapi
          (fun j r ->
            match U.as_range r with
            | Some { size; parents; sub; kind; _ } ->
                let dtype = Dtype.val_of (U.dtype r) in
                U.range ~size ~axis:(base + j) ~kind ~sub ~dtype
                  ~parents ()
            | None -> r)
	          ranges
	      in
	      let mappings =
	        List.fold_left
	          (fun acc (ranges, ends_ref) ->
	            let ends = List.rev !ends_ref in
	            if List.length ends <= 1 then acc
	              else
              let scope_equal a b =
                let mem x xs = List.exists (U.equal x) xs in
                List.length a = List.length b && List.for_all (fun x -> mem x b) a
              in
              let groups = ref [] in
              List.iter
                (fun e ->
                  let scope = U.ranges e in
                  match
                    List.find_opt
                      (fun (scope', _) -> scope_equal scope scope')
                      !groups
                  with
                  | Some (_, group) -> group := e :: !group
                  | None -> groups := !groups @ [ (scope, ref [ e ]) ])
                ends;
              let groups = List.map (fun (_, group) -> List.rev !group) !groups in
              let _, acc =
                List.fold_left
                  (fun (group_idx, acc) group ->
                    let ranges', mapped_group =
                      if group_idx = 0 then (ranges, group)
                      else
                        let ranges' = clone_ranges ranges in
                        let subs = List.combine ranges ranges' in
                        (ranges', List.map (fun e -> U.substitute subs e) group)
                    in
                    let merged =
                      match mapped_group with
                      | [ e ] -> e
                      | _ ->
                          let stores =
                            List.map
                              (fun e ->
                                match U.as_end e with
                                | Some { value; _ } -> value
                                | None -> assert false)
                              mapped_group
                          in
                          U.end_ ~value:(U.group stores) ~ranges:ranges'
                    in
	                    let acc =
	                      List.fold_left
	                        (fun acc old ->
                            if old == merged
                               || List.exists (( == ) old)
                                    (U.backward_slice merged)
                            then acc
                            else (old, merged) :: acc)
	                        acc group
	                    in
                    (group_idx + 1, acc))
	                  (0, acc) groups
	              in
	              acc)
	          [] !range_groups
	      in
      (match mappings with
       | [] -> None
       | _ -> Some (U.substitute mappings node))
  | _ -> None

let merge_reduce_ends =
  let open Upat in
  op ~name:"sink" Ops.Sink
  => fun bs -> merge_reduce_ends_rule (bs $ "sink")

(* Fold [ADD(Wmma, x)] into the Wmma's accumulator: [Wmma(a, b, c + x)]. *)
let wmma_accumulate_rule node =
  match U.op node, U.src node with
  | Ops.Add, [| lhs; rhs |] ->
      let fold w add =
        match U.as_wmma w with
        | None -> None
        | Some v ->
            Some (U.replace w ~src:[| v.a; v.b; U.O.(v.c + add) |] ())
      in
      (match fold lhs rhs with Some folded -> Some folded | None -> fold rhs lhs)
  | _ -> None

let wmma_accumulate =
  let open Upat in
  op ~name:"add" Ops.Add
  => fun bs -> wmma_accumulate_rule (bs $ "add")

let range_kind_is kind r =
  match U.as_range r with
  | Some v -> Axis_type.equal v.kind kind
  | None -> false

let clone_group_reduce_range r =
  match U.as_range r with
  | Some v ->
      U.range ~size:v.size ~axis:(v.axis + 100) ~kind:Axis_type.Reduce
        ~sub:v.sub ~dtype:(Dtype.val_of (U.dtype r)) ~parents:v.parents ()
  | None -> invalid_arg "clone_group_reduce_range: expected RANGE"

let fix_group_for_reduce_rule node =
  match U.as_reduce node with
  | None -> None
  | Some v ->
      let group_reduce, reduce_ranges =
        List.partition (range_kind_is Axis_type.Group_reduce) v.ranges
      in
      if group_reduce = [] then None
      else
        let upstream_locals =
          U.toposort node
          |> List.filter (range_kind_is Axis_type.Local)
        in
        let partial =
          U.replace node
            ~src:(Array.of_list (v.src :: reduce_ranges))
            ()
        in
        let reduce_loop = List.map clone_group_reduce_range group_reduce in
        let opts : U.stage_opts =
          { device = None; addrspace = Dtype.Local; removable = false }
        in
        let local =
          U.stage ~src:partial ~ranges:(upstream_locals @ group_reduce)
            ~opts
        in
        let indexed =
          U.index ~ptr:local ~idxs:(upstream_locals @ reduce_loop) ()
        in
        Some (U.reduce ~src:indexed ~op:v.op ~ranges:reduce_loop
                ~dtype:(Dtype.val_of (U.dtype node)))

let fix_group_for_reduce =
  let open Upat in
  op ~name:"reduce" Ops.Reduce
  => fun bs -> fix_group_for_reduce_rule (bs $ "reduce")

let pm_group_for_reduce : Upat.Pattern_matcher.t =
  Upat.Pattern_matcher.make [ fix_group_for_reduce ]

let pm_reduce (root : U.t) : U.t =
  let ctx = { acc_num = 0; acc_slots = reduce_slots_in_tinygrad_order root } in
  let matcher =
    Upat.Pattern_matcher.make
      [
        reduce_invalid_to_identity;
        expand_horizontal_reduce;
        reduce_to_acc ctx;
        wmma_accumulate;
        merge_reduce_ends;
      ]
  in
  U.graph_rewrite ~name:"remove_reduce"
    (U.first_match [ Rangeify.mop_cleanup; Upat.Pattern_matcher.rewrite matcher ])
    root

(* pm_devectorize *)

let pm_devectorize (root : U.t) : U.t =
  let matcher =
    Upat.Pattern_matcher.(
      Symbolic.symbolic_simple ++ devectorize)
  in
  U.graph_rewrite ~name:"devectorize"
    (U.first_match [ Rangeify.movement_ops; Upat.Pattern_matcher.rewrite matcher ])
    root

let debug = Helpers.getenv "DEBUG" 0
let image_env = Helpers.getenv "IMAGE" 0
let disable_fast_idiv = Helpers.getenv "DISABLE_FAST_IDIV" 1 <> 0
let transcendental_env = Helpers.getenv "TRANSCENDENTAL" 1
let spec_enabled () = Helpers.getenv "SPEC" 0 <> 0

(* Stamp renderer capabilities with env-derived decomposition toggles. *)
let supported_ops_of (ren : Renderer.t) : Decomp_op.supported_ops =
  let ir = Renderer.supported_ops ren in
  {
    has_exp2 = ir.has_exp2; has_log2 = ir.has_log2;
    has_sin = ir.has_sin; has_sqrt = ir.has_sqrt;
    has_neg = ir.has_neg; has_sub = ir.has_sub; has_max = ir.has_max;
    has_shl = ir.has_shl; has_shr = ir.has_shr;
    has_and = ir.has_and; has_or = ir.has_or;
    has_cmplt = ir.has_cmplt; has_cmpeq = ir.has_cmpeq;
    has_fdiv = ir.has_fdiv; has_threefry = ir.has_threefry;
    has_mulacc = ir.has_mulacc;
    is_metal = Renderer.name ren = "metal";
    supports_dtype = Renderer.supports_dtype ren;
    disable_fast_idiv;
    force_transcendental = transcendental_env >= 2;
  }

let int_const dtype n = U.const (Const.int dtype n)

let shape_size_const size = int_const Dtype.Val.int32 size

let remove_vec_dtype node =
  match U.op node, U.dtype node with
  | Ops.Const, Dtype.Val v when Dtype.Val.count v > 1 ->
      let scalar = Dtype.Val.scalarize v in
      let value =
        match U.arg node with
        | U.Arg.Value c -> Const.view c
        | _ -> invalid_arg "vector CONST without value"
      in
      Some
        (U.stack
           (List.init (Dtype.Val.count v) (fun _ ->
                U.const (Const.of_view scalar value))))
  | (Ops.Param | Ops.Buffer), Dtype.Ptr p when not (Dtype.Ptr.is_image p) ->
      Some
        (U.replace node
           ~src:[| shape_size_const (Dtype.Ptr.size p) |]
           ~dtype:(Dtype.Val (Dtype.Ptr.base p)) ())
  | (Ops.Param | Ops.Buffer), _ -> None
  | _ ->
      let dtype = Dtype.Val (Dtype.Val.scalarize (Dtype.val_of (U.dtype node))) in
      if Dtype.equal dtype (U.dtype node) then None
      else Some (U.replace node ~dtype ())

let clean_up_group_sink node =
  match U.op node, Array.to_list (U.src node) with
  | Ops.Group, [ x ] -> Some x
  | (Ops.Sink | Ops.Group), srcs ->
      let remove_like u =
        match U.op u with
        | Ops.Noop | Ops.Stack | Ops.Sink | Ops.Group -> true
        | _ -> false
      in
      let srcs' =
        List.concat_map
          (fun u -> if remove_like u then Array.to_list (U.src u) else [ u ])
          srcs
      in
      let srcs' =
        List.fold_left
          (fun acc src ->
            if List.exists (U.equal src) acc then acc else src :: acc)
          [] srcs'
        |> List.rev
      in
      if List.length srcs = List.length srcs'
         && List.for_all2 U.equal srcs srcs'
      then None
      else Some (U.replace node ~src:(Array.of_list srcs') ())
  | _ -> None

let pm_remove_vec_dtypes sink =
  U.graph_rewrite ~name:"transform to new style"
    (U.first_match [ remove_vec_dtype; clean_up_group_sink ])
    sink

let symbolic = Symbolic.symbolic
let sym = Symbolic.sym

let pm_indexing_simplify sink =
  U.graph_rewrite ~name:"simplify load/store indexing"
    (PM.rewrite Coalese.indexing_simplify) sink

let int32_for_weakint dtype =
  match dtype with
  | Dtype.Val v when Dtype.Val.scalar v = Dtype.Weakint ->
      Some (Dtype.Val (Dtype.Val.with_scalar Dtype.Int32 v))
  | _ -> None

let cleanup_weakints node =
  match int32_for_weakint (U.dtype node) with
  | None -> None
  | Some dtype ->
      let arg =
        match U.op node, U.arg node, dtype with
        | Ops.Const, U.Arg.Value c, Dtype.Val v ->
            U.Arg.Value (Const.of_view v (Const.view c))
        | _ -> U.arg node
      in
      Some (U.replace node ~dtype ~arg ())

let load_value src =
  match U.dtype src with
  | Dtype.Ptr _ -> U.load ~src ()
  | Dtype.Val _ -> U.load ~src ~dtype:(U.dtype src) ()

let maybe_load src =
  match U.addrspace src with
  | Some (Dtype.Global | Dtype.Local | Dtype.Reg) -> load_value src
  | Some Dtype.Alu | None -> src

let pm_move_regs_rule node =
  let rewrite_all_srcs () =
    let src = U.src node in
    let src' = Array.map maybe_load src in
    if Array.for_all2 U.equal src src' then None
    else Some (U.replace node ~src:src' ())
  in
  match U.op node with
  | op when Ops.Group.is_elementwise op -> rewrite_all_srcs ()
  | Ops.Reduce | Ops.Wmma | Ops.Stack -> rewrite_all_srcs ()
  | Ops.Store -> (
      match U.as_store node with
      | Some { dst; value; gate } ->
          let value' = maybe_load value in
          if U.equal value value' then None
          else Some (U.store ~dst ~value:value' ?gate ())
      | None -> None)
  | _ -> None

let pm_move_regs sink =
  U.graph_rewrite ~name:"** add loads" pm_move_regs_rule sink

let shape_arg dims =
  match dims with
  | [ dim ] -> dim
  | dims -> U.stack dims

let shape_opt u = try Some (U.shape u) with Invalid_argument _ -> None

let dim_is_one dim =
  match U.const_int_value dim with Some 1 -> true | _ -> false

let broadcast_shape shapes =
  let rec last = function
  | [] -> None
  | [ x ] -> Some x
  | _ :: xs -> last xs
  in
  let rec drop_last = function
  | [] | [ _ ] -> []
  | x :: xs -> x :: drop_last xs
  in
  let rec loop shapes =
    match List.filter (( <> ) []) shapes with
    | [] -> []
    | shapes ->
        let tails = List.filter_map last shapes in
        let dim =
          List.fold_left
            (fun acc d ->
              match acc with
              | None -> Some d
              | Some a when U.equal a d -> Some a
              | Some a when dim_is_one a -> Some d
              | Some a when dim_is_one d -> Some a
              | Some a -> Some a)
            None tails
        in
        let prefixes = List.map drop_last shapes in
        let prefix =
          if List.for_all (( = ) []) prefixes then [] else loop prefixes
        in
        (match dim with None -> prefix | Some d -> prefix @ [ d ])
  in
  loop shapes

let align_left rank shape =
  List.init (rank - List.length shape) (fun _ -> U.const_int 1) @ shape

let normalize_broadcast_src target_shape src shape =
  let aligned_shape = align_left (List.length target_shape) shape in
  if List.equal U.equal shape target_shape then src
  else
    let src =
      if List.equal U.equal aligned_shape shape then src
      else U.reshape ~src ~shape:(shape_arg aligned_shape)
    in
    U.expand ~src ~shape:(shape_arg target_shape)

let index_singleton_dst dst shape =
  if shape = [] then dst
  else U.index ~ptr:dst ~idxs:(List.map (fun _ -> int_ 0) shape) ~as_ptr:true ()

let broadcast_binary node =
  match U.op node with
  | Ops.Store -> (
      match U.as_store node with
      | None -> None
      | Some ({ dst; value; gate } as store) -> (
          match shape_opt dst, shape_opt value with
          | Some [], _ -> None
          | Some dst_shape, Some value_shape ->
              let value = normalize_broadcast_src dst_shape value value_shape in
              let gate =
                match gate with
                | None -> None
                | Some gate ->
                    (match shape_opt gate with
                     | None -> Some gate
                     | Some gate_shape ->
                         Some (normalize_broadcast_src dst_shape gate gate_shape))
              in
              if U.equal value store.value
                 && Option.equal U.equal gate store.gate
              then None
              else Some (U.store ~dst ~value ?gate ())
          | _ -> None))
  | op when Ops.Group.is_broadcastable op -> (
      let srcs = U.src node in
      let shapes = Array.to_list srcs |> List.map shape_opt in
      if List.exists Option.is_none shapes then None
      else
        let shapes = List.map Option.get shapes in
        (match shapes with
        | [] -> None
        | first :: rest when List.for_all (List.equal U.equal first) rest ->
            None
        | _ ->
            let target_shape = broadcast_shape shapes in
            let src =
              Array.of_list
                (List.map2 (normalize_broadcast_src target_shape)
                   (Array.to_list srcs) shapes)
            in
            Some (U.replace node ~src ())))
  | _ -> None

let devec_wmma node =
  match U.as_wmma node with
  | None -> None
  | Some _ -> (
      let srcs = U.src node in
      let src_shapes = Array.to_list srcs |> List.map shape_opt in
      if List.exists Option.is_none src_shapes then None
      else
        let src_shapes = List.map Option.get src_shapes in
        let prefix_shapes =
          List.map
            (fun shape ->
              match List.rev shape with
              | [] -> []
              | _lane :: prefix -> List.rev prefix)
            src_shapes
        in
        (match prefix_shapes with
        | [] | _ :: [] -> None
        | first :: rest when List.for_all (List.equal U.equal first) rest ->
            None
        | _ ->
            let target_prefix = broadcast_shape prefix_shapes in
            let rank = List.length target_prefix in
            let normalized =
              List.map2
                (fun src shape ->
                  match List.rev shape with
                  | [] -> src
                  | lane :: rev_prefix ->
                      let prefix = List.rev rev_prefix in
                      let aligned = align_left rank prefix @ [ lane ] in
                      U.expand
                        ~src:(U.reshape ~src ~shape:(shape_arg aligned))
                        ~shape:(shape_arg (target_prefix @ [ lane ])))
                (Array.to_list srcs) src_shapes
            in
            let rec product = function
            | [] -> [ [] ]
            | dim :: dims ->
                let tail = product dims in
                List.concat
                  (List.init (U.vmax dim) (fun i ->
                       List.map (fun idx -> U.const_int i :: idx) tail))
            in
            let lanes =
              List.map
                (fun idxs ->
                  U.replace node
                    ~src:
                      (Array.of_list
                         (List.map
                            (fun src -> U.index ~ptr:src ~idxs ())
                            normalized))
                    ())
                (product target_prefix)
            in
            Some (U.reshape ~src:(U.stack lanes) ~shape:(shape_arg (U.shape node)))))

let unbroadcast node =
  match broadcast_binary node with
  | Some _ as r -> r
  | None -> devec_wmma node

let const_ints dtype values =
  U.stack ~dtype:dtype (List.map (fun i -> U.const (Const.int dtype i)) values)

let argsort order =
  order
  |> List.mapi (fun i v -> (v, i))
  |> List.sort compare
  |> List.map snd

let build_range_map sink =
  let ctx = Hashtbl.create 8 in
  U.toposort sink
  |> List.iter (fun node ->
       match U.as_range node with
       | Some { axis; kind = Axis_type.Unroll | Axis_type.Upcast; _ } ->
           if not (Hashtbl.mem ctx axis) then
             Hashtbl.add ctx axis (Hashtbl.length ctx)
       | Some _ | None -> ());
  ctx

let expand_reduce node =
  match U.as_reduce node with
  | None -> None
  | Some { src; ranges; op; axes = [] } ->
      let range_srcs = ref [] in
      let new_axes = ref [] in
      List.iter
        (fun u ->
          if U.op u = Ops.Range then range_srcs := u :: !range_srcs
          else
            U.shape u
            |> List.iteri (fun i s ->
                 if U.vmax s > 1 then new_axes := i :: !new_axes))
        ranges;
      let new_axes = List.rev !new_axes in
      if new_axes = [] then None
      else
        let src_shape = U.shape src in
        let out_shape =
          List.mapi
            (fun i s -> if List.mem i new_axes then U.const_int 1 else s)
            src_shape
        in
        Some
          (U.reshape
             ~src:
               (U.reduce ~op
                  ~src:(U.reduce_axis ~src ~op ~axes:new_axes)
                  ~ranges:(List.rev !range_srcs)
                  ~dtype:(Dtype.val_of (U.dtype node)))
             ~shape:(shape_arg out_shape))
  | Some _ -> None

let contract_axis range_map u axes =
  let permute_tail =
    List.map (fun (rn, _) -> Hashtbl.find range_map rn) axes
  in
  let shape = U.shape u in
  let permute_head =
    List.init (List.length shape) Fun.id
    |> List.filter (fun i -> not (List.mem i permute_tail))
  in
  let out = U.permute ~src:u ~order:(permute_head @ permute_tail) in
  let out_shape = U.max_shape out in
  let head_shape =
    List.filteri (fun i _ -> i < List.length permute_head) out_shape
  in
  (* Flatten the tail axes into one dimension (reshape [..., -1]). *)
  let tail = prod out_shape / prod head_shape in
  U.reshape ~src:out
    ~shape:(shape_arg (List.map U.const_int (head_shape @ [ tail ])))

let unroll_axis range_map u axes =
  let permute_tail =
    List.map (fun (rn, _) -> Hashtbl.find range_map rn) axes
  in
  let shape = U.shape u in
  let prefix =
    match List.rev shape with
    | [] -> []
    | _last :: rest -> List.rev rest
  in
  let out =
    U.reshape ~src:u
      ~shape:
        (shape_arg
           (prefix @ List.map (fun (_, size) -> U.const_int size) axes))
  in
  let out_shape = U.shape out in
  let permute_head =
    List.init (List.length out_shape) Fun.id
    |> List.filter (fun i -> not (List.mem i permute_tail))
  in
  U.permute ~src:out ~order:(argsort (permute_head @ permute_tail))

let expand_wmma range_map node =
  match U.as_wmma node with
  | Some { a; b; c; info } when U.node_tag node = Some "1" ->
      let in0, in1, out0 = info.upcast_axes in
      let wmma =
        U.replace node
          ~src:
            [| contract_axis range_map a in0;
               contract_axis range_map b in1;
               c |]
          ~node_tag:None ()
      in
      Some (unroll_axis range_map wmma out0)
  | Some _ | None -> None

let expander2 range_map =
  let open Upat in
  Pattern_matcher.make [
    op ~name:"r" Ops.Reduce => (fun bs -> expand_reduce (bs $ "r"));
    op ~name:"r" Ops.Range
    => (fun bs ->
         let r = bs $ "r" in
         match U.as_range r with
         | Some { axis; _ } when Hashtbl.mem range_map axis ->
             let idx = Hashtbl.find range_map axis in
             let n = U.vmax r + 1 in
             let dtype = Dtype.val_of (U.dtype r) in
             let dims =
               List.init (Hashtbl.length range_map) (fun i ->
                   U.const_int (if i = idx then n else 1))
             in
             Some
               (U.reshape ~src:(const_ints dtype (List.init n Fun.id))
                  ~shape:(shape_arg dims))
         | Some _ | None -> None);
    op ~name:"u" Ops.Wmma => (fun bs -> expand_wmma range_map (bs $ "u"));
  ]

let number_params sink =
  let next_slot =
    U.toposort sink
    |> List.fold_left
         (fun acc node ->
           match U.as_param node with
           | Some { param = { slot; _ }; _ } when slot >= 0 -> acc + 1
           | _ -> acc)
         0
    |> ref
  in
  let rewrite_param node =
    match U.as_param node with
    | Some { param; _ } when param.slot = -1 ->
        let slot = !next_slot in
        incr next_slot;
        Some
          (U.replace node
             ~arg:(U.Arg.Param_arg { param with slot })
             ())
    | _ -> None
  in
  U.graph_rewrite ~name:"number params with -1" ~walk:true rewrite_param sink

(* Lower an optimized kernel AST to a form ready for linearization.
   Mirrors [full_rewrite_to_sink] after the [apply_opts] call. *)
let lower (ren : Renderer.t) (sink : U.t) : U.t =
  let rewrite ?name rule = U.graph_rewrite ?name rule in
  let pm pm' = PM.rewrite pm' in
  let rule_new_style rule node =
    try rule node with
    | Invalid_argument msg
      when String.equal msg "Uop.index: expected pointer ptr" ->
        None
  in
  let pm_new_style pm' = rule_new_style (PM.rewrite pm') in

  (* early movement ops: [pm_mops + pm_syntactic_sugar]. *)
  let sink = Rangeify.rewrite_movement_ops sink in

  (* transform to new style: [pm_remove_vec_dtypes]. *)
  let sink = pm_remove_vec_dtypes sink in

  (* postopt symbolic: [sym + pm_move_where_on_load + pm_flatten_range]. *)
  let pm_postopt =
    U.first_match
      [
        pm
          Upat.Pattern_matcher.(
            sym ++ Symbolic.pm_move_where_on_load);
        Simplify.flatten_range;
      ]
  in
  let sink = rewrite ~name:"postopt symbolic" pm_postopt sink in

  (* expander: [expander2]. *)
  let range_map = build_range_map sink in
  let sink =
    rewrite ~name:"expander"
      (U.first_match
         [ pm (expander2 range_map); Simplify.flatten_range ])
      sink
  in

  (* group for reduce: [pm_group_for_reduce]. *)
  let sink = rewrite ~name:"group for reduce" (pm pm_group_for_reduce) sink in

  (* add locals: [pm_add_local_buffers = add_local_buffer + pm_mops]. *)
  let local_slots = ref 0 in
  let sink =
    rewrite ~name:"add local buffers"
      (U.first_match
         [ add_local_buffer_rule local_slots; Rangeify.movement_ops ])
      sink
  in

  (* remove reduce: [mop_cleanup + pm_reduce_local]. *)
  let sink = pm_reduce sink in

  (* add gpu dims: [pm_add_gpudims]. *)
  let sink = Gpudims.pm_add_gpudims ren sink in

  (* unbroadcast: [symbolic_simple + unbroadcast]. *)
  let sink =
    rewrite ~name:"unbroadcast"
      (U.first_match [ pm Symbolic.symbolic_simple; unbroadcast ])
      sink
  in

  (* add loads: [pm_move_regs]. *)
  let sink = pm_move_regs sink in

  (* devectorize: [symbolic_simple + devectorizer2]. *)
  let sink = pm_devectorize sink in

  (* simplify indexing: [indexing_simplify]. *)
  let sink = pm_indexing_simplify sink in

  (* render index/stack after indexing simplify. *)
  let sink = rewrite ~name:"render index/stack" (pm pm_render) sink in

  (* some coalesing misses without this: [sym]. *)
  let sink = rewrite ~name:"early symbolic" (pm sym) sink in

  (* do memory coalesing (late). *)
  let sink = Coalese.memory_coalesing ren sink in

  (* add images: [pm_simplify_add_image]. Also folds per-lane re-stacks of
     freshly coalesced vector loads back into the load. *)
  let sink =
    U.graph_rewrite ~name:"add images" ~bottom_up:true
      (U.first_match
         [
           pm (Coalese.pm_simplify_add_image ren);
           pm Symbolic.pm_fold_lane_stack;
         ])
      sink
  in

  (* extra symbolic before index dtype lowering. *)
  let sink =
    rewrite ~name:"extra symbolic before index dtype"
      (pm_new_style sym)
      sink
  in

  (* lower all index dtypes:
     [pm_lower_index_dtype + indexing_simplify]. *)
  let sink =
    rewrite ~name:"lower all index dtypes"
      (U.first_match
         [
           pm_new_style Symbolic.pm_lower_index_dtype;
           PM.rewrite Coalese.indexing_simplify;
         ])
      sink
  in
  let sink =
    rewrite ~name:"final symbolic" (pm_new_style symbolic) sink
  in

  let sink =
    match Renderer.pre_matcher ren with
    | None -> sink
    | Some pre -> rewrite ~name:"pre_matcher" pre sink
  in

  (* early decompositions: [symbolic_simple + get_simplifying_rewrite_patterns]. *)
  let ops = supported_ops_of ren in
  let pm_early_decomp =
    U.first_match
      [ pm_new_style Symbolic.symbolic_simple;
        rule_new_style (Decomp_op.get_simplifying_rewrite_patterns ops) ]
  in
  let sink = rewrite ~name:"early decompositions" pm_early_decomp sink in

  (* decomp dtypes: [pm_dtype_decomps]. *)
  let sink = Decomp_dtype.do_dtype_decomps ren sink in

  (* late decompositions: [early decompositions + get_late_rewrite_patterns]. *)
  let pm_decomp =
    U.first_match
      [ pm_early_decomp;
        rule_new_style (Decomp_op.get_late_rewrite_patterns ops);
        rule_new_style (Decomp_transcendental.get_transcendental_patterns ops) ]
  in
  let sink = rewrite ~name:"late decompositions" pm_decomp sink in

  (* gater: [pm_move_gates_from_index]. *)
  let sink = Gater.pm_move_gates_from_index sink in

  (* final rewrite:
     tinygrad has
     [pm_decomp + extra_matcher + pm_split_ends + pm_no_weakints
      + pm_remove_invalid]. *)
  let extra_matcher =
    match Renderer.extra_matcher ren with
    | None -> fun _ -> None
    | Some extra -> extra
  in
  let pm_final =
    U.first_match
      [ pm_decomp;
        extra_matcher;
        Linearizer.do_split_ends;
        cleanup_weakints;
        PM.rewrite Symbolic.pm_remove_invalid ]
  in
  let sink = rewrite ~name:"final rewrite" pm_final sink in

  (* add control flow: [pm_add_control_flow], bottom-up. *)
  let sink = Linearizer.pm_add_control_flow sink in

  (* put unnumbered variable PARAMs in slots. *)
  let sink = number_params sink in

  if spec_enabled () then Spec.type_verify Spec.program_spec sink;

  if debug >= 6 then print_string (Render.uops_to_string ~label:"lower" sink);
  sink
