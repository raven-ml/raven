(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

type device = Single of string | Multi of string list
type id = int
type metadata = { name : string; caller : string; backward : bool }

type emit = view -> id
and grad_fxn = emit:emit -> grad_output:id -> call:id -> id option list
and callee = Ref of id | Ast of Kernel.t

and call_info = {
  grad_fxn : grad_fxn option;
  metadata : metadata list;
  name : string option;
  precompile : bool;
}

and view =
  | Sink of { srcs : id list; kernel_info : Kernel.kernel_info option }
  | Group of { srcs : id list }
  | After of { src : id; deps : id list; dtype : Dtype.t }
  | Unique of { id : int }
  | Lunique of { id : int }
  | Device of { device : device }
  | Buffer of { unique : id; device : id; size : int; dtype : Dtype.t }
  | Buffer_view of { src : id; size : int; offset : int; dtype : Dtype.t }
  | Const of { value : Const.t; dtype : Dtype.t; srcs : id list }
  | Vconst of { values : Const.t list; dtype : Dtype.t; srcs : id list }
  | Define_var of { name : string; lo : int; hi : int; dtype : Dtype.t }
  | Bind of { var : id; value : id option; dtype : Dtype.t }
  | Param of {
      slot : int;
      dtype : Dtype.t;
      shape : id option;
      device : id option;
    }
  | Call of {
      callee : callee;
      args : id list;
      info : call_info;
      dtype : Dtype.t;
    }
  | Detach of { src : id; dtype : Dtype.t }
  | Contiguous of { src : id; ranges : id list; dtype : Dtype.t }
  | Contiguous_backward of { src : id; dtype : Dtype.t }
  | Copy of { src : id; device : id; dtype : Dtype.t }
  | Allreduce of { src : id; device : id; op : Op.reduce; dtype : Dtype.t }
  | Multi of { src : id; axis : int; dtype : Dtype.t }
  | Mstack of { srcs : id list; dtype : Dtype.t }
  | Mselect of { src : id; index : int; dtype : Dtype.t }
  | Reduce_axis of {
      src : id;
      op : Op.reduce;
      axes : int list;
      dtype : Dtype.t;
    }
  | Reduce of { src : id; ranges : id list; op : Op.reduce; dtype : Dtype.t }
  | Reshape of { src : id; shape : id; dtype : Dtype.t }
  | Expand of { src : id; shape : id; dtype : Dtype.t }
  | Pad of { src : id; before : id; after : id; dtype : Dtype.t }
  | Shrink of { src : id; before : id; after : id; dtype : Dtype.t }
  | Permute of { src : id; order : int list; dtype : Dtype.t }
  | Flip of { src : id; dims : bool list; dtype : Dtype.t }
  | Range of { size : id; dtype : Dtype.t; axis : int; sub : int list; kind : Axis_kind.t }
  | End of { value : id; ranges : id list }
  | Index of { ptr : id; idxs : id list; gate : id option; dtype : Dtype.t }
  | Store of { dst : id; value : id }
  | Vectorize of { srcs : id list; dtype : Dtype.t }
  | Cast of { src : id; dtype : Dtype.t }
  | Bitcast of { src : id; dtype : Dtype.t }
  | Unary of { op : Op.unary; src : id; dtype : Dtype.t }
  | Binary of { op : Op.binary; lhs : id; rhs : id; dtype : Dtype.t }
  | Ternary of { op : Op.ternary; a : id; b : id; c : id; dtype : Dtype.t }
  | Noop of { src : id option; dtype : Dtype.t }
  | Bufferize of {
      src : id;
      ranges : id list;
      dtype : Dtype.t;
      opts : Kernel.bufferize_opts;
    }
  | Invalid_index of { dtype : Dtype.t }
  | Define_local of { size : int; dtype : Dtype.ptr }
  | Barrier

type t = view array
type builder = { mutable data : view array; mutable len : int }

let pp_comma fmt () = Format.fprintf fmt ", "
let pp_ref fmt r = Format.fprintf fmt "%%%d" r
let pp_refs fmt refs = Format.pp_print_list ~pp_sep:pp_comma pp_ref fmt refs
let create () = { data = Array.make 16 (Group { srcs = [] }); len = 0 }

let ensure_capacity b =
  if b.len = Array.length b.data then begin
    let next = Array.make (2 * Array.length b.data) (Group { srcs = [] }) in
    Array.blit b.data 0 next 0 b.len;
    b.data <- next
  end

let emit builder (instr : view) =
  ensure_capacity builder;
  let id = builder.len in
  builder.data.(id) <- instr;
  builder.len <- id + 1;
  id

let finish builder = Array.sub builder.data 0 builder.len

let node_dtype = function
  | Buffer { dtype; _ }
  | Buffer_view { dtype; _ }
  | Const { dtype; _ }
  | Vconst { dtype; _ }
  | Define_var { dtype; _ }
  | Bind { dtype; _ }
  | Param { dtype; _ }
  | Call { dtype; _ }
  | After { dtype; _ }
  | Detach { dtype; _ }
  | Contiguous { dtype; _ }
  | Contiguous_backward { dtype; _ }
  | Copy { dtype; _ }
  | Allreduce { dtype; _ }
  | Multi { dtype; _ }
  | Mstack { dtype; _ }
  | Mselect { dtype; _ }
  | Reduce_axis { dtype; _ }
  | Reduce { dtype; _ }
  | Reshape { dtype; _ }
  | Expand { dtype; _ }
  | Pad { dtype; _ }
  | Shrink { dtype; _ }
  | Permute { dtype; _ }
  | Flip { dtype; _ }
  | Range { dtype; _ }
  | Index { dtype; _ }
  | Vectorize { dtype; _ }
  | Cast { dtype; _ }
  | Bitcast { dtype; _ }
  | Unary { dtype; _ }
  | Binary { dtype; _ }
  | Ternary { dtype; _ } ->
      Some dtype
  | Noop { dtype; _ }
  | Bufferize { dtype; _ }
  | Invalid_index { dtype; _ } ->
      Some dtype
  | Define_local { dtype; _ } -> Some (Dtype.base dtype)
  | Sink _ | Group _ | Unique _ | Lunique _ | Device _ | End _ | Store _
  | Barrier ->
      None

let dtype_of_id builder id =
  if id < 0 || id >= builder.len then None else node_dtype builder.data.(id)

let value_dtype_exn builder ctx id =
  match dtype_of_id builder id with
  | Some dtype -> dtype
  | None -> invalid_arg (Printf.sprintf "Tensor.%s expects a value node" ctx)

let children_of = function
  | Sink { srcs; _ } | Group { srcs } -> srcs
  | After { src; deps; _ } -> src :: deps
  | Unique _ | Lunique _ | Device _ | Define_var _ | Invalid_index _ | Barrier ->
      []
  | Buffer { unique; device; _ } -> [ unique; device ]
  | Buffer_view { src; _ } -> [ src ]
  | Const { srcs; _ } | Vconst { srcs; _ } -> srcs
  | Bind { var; value; _ } -> var :: Option.to_list value
  | Param { shape; device; _ } -> Option.to_list shape @ Option.to_list device
  | Call { callee; args; _ } -> (
      match callee with Ref fn -> fn :: args | Ast _ -> args)
  | Detach { src; _ }
  | Contiguous_backward { src; _ }
  | Multi { src; _ }
  | Mselect { src; _ }
  | Cast { src; _ }
  | Bitcast { src; _ }
  | Unary { src; _ } ->
      [ src ]
  | Contiguous { src; ranges; _ } | Reduce { src; ranges; _ } -> src :: ranges
  | Copy { src; device; _ }
  | Allreduce { src; device; _ } ->
      [ src; device ]
  | Mstack { srcs; _ } | Vectorize { srcs; _ } -> srcs
  | Reduce_axis { src; _ } -> [ src ]
  | Reshape { src; shape; _ } | Expand { src; shape; _ } -> [ src; shape ]
  | Pad { src; before; after; _ } | Shrink { src; before; after; _ } ->
      [ src; before; after ]
  | Permute { src; _ } | Flip { src; _ } -> [ src ]
  | Range { size; _ } -> [ size ]
  | End { value; ranges } -> value :: ranges
  | Index { ptr; idxs; gate; _ } -> (ptr :: idxs) @ Option.to_list gate
  | Store { dst; value } -> [ dst; value ]
  | Binary { lhs; rhs; _ } -> [ lhs; rhs ]
  | Ternary { a; b; c; _ } -> [ a; b; c ]
  | Noop { src; _ } -> Option.to_list src
  | Bufferize { src; ranges; _ } -> src :: ranges
  | Define_local _ -> []

let map_children map_ref instr =
  let map_ref_list = List.map map_ref in
  let map_ref_opt = Option.map map_ref in
  match instr with
  | Sink { srcs; kernel_info } -> Sink { srcs = map_ref_list srcs; kernel_info }
  | Group { srcs } -> Group { srcs = map_ref_list srcs }
  | After { src; deps; dtype } ->
      After { src = map_ref src; deps = map_ref_list deps; dtype }
  | Unique _ | Lunique _ | Device _ | Define_var _ | Invalid_index _ | Barrier ->
      instr
  | Buffer { unique; device; size; dtype } ->
      Buffer { unique = map_ref unique; device = map_ref device; size; dtype }
  | Buffer_view { src; size; offset; dtype } ->
      Buffer_view { src = map_ref src; size; offset; dtype }
  | Const { value; dtype; srcs } ->
      Const { value; dtype; srcs = map_ref_list srcs }
  | Vconst { values; dtype; srcs } ->
      Vconst { values; dtype; srcs = map_ref_list srcs }
  | Bind { var; value; dtype } ->
      Bind { var = map_ref var; value = map_ref_opt value; dtype }
  | Param { slot; dtype; shape; device } ->
      Param
        { slot; dtype; shape = map_ref_opt shape; device = map_ref_opt device }
  | Call { callee; args; info; dtype } ->
      let callee = match callee with Ref fn -> Ref (map_ref fn) | Ast _ -> callee in
      Call { callee; args = map_ref_list args; info; dtype }
  | Detach { src; dtype } -> Detach { src = map_ref src; dtype }
  | Contiguous { src; ranges; dtype } ->
      Contiguous { src = map_ref src; ranges = map_ref_list ranges; dtype }
  | Contiguous_backward { src; dtype } ->
      Contiguous_backward { src = map_ref src; dtype }
  | Copy { src; device; dtype } ->
      Copy { src = map_ref src; device = map_ref device; dtype }
  | Allreduce { src; device; op; dtype } ->
      Allreduce { src = map_ref src; device = map_ref device; op; dtype }
  | Multi { src; axis; dtype } -> Multi { src = map_ref src; axis; dtype }
  | Mstack { srcs; dtype } -> Mstack { srcs = map_ref_list srcs; dtype }
  | Mselect { src; index; dtype } -> Mselect { src = map_ref src; index; dtype }
  | Reduce_axis { src; op; axes; dtype } ->
      Reduce_axis { src = map_ref src; op; axes; dtype }
  | Reduce { src; ranges; op; dtype } ->
      Reduce { src = map_ref src; ranges = map_ref_list ranges; op; dtype }
  | Reshape { src; shape; dtype } ->
      Reshape { src = map_ref src; shape = map_ref shape; dtype }
  | Expand { src; shape; dtype } ->
      Expand { src = map_ref src; shape = map_ref shape; dtype }
  | Pad { src; before; after; dtype } ->
      Pad { src = map_ref src; before = map_ref before;
            after = map_ref after; dtype }
  | Shrink { src; before; after; dtype } ->
      Shrink { src = map_ref src; before = map_ref before;
               after = map_ref after; dtype }
  | Permute { src; order; dtype } -> Permute { src = map_ref src; order; dtype }
  | Flip { src; dims; dtype } -> Flip { src = map_ref src; dims; dtype }
  | Range { size; dtype; axis; sub; kind } ->
      Range { size = map_ref size; dtype; axis; sub; kind }
  | End { value; ranges } ->
      End { value = map_ref value; ranges = map_ref_list ranges }
  | Index { ptr; idxs; gate; dtype } ->
      Index { ptr = map_ref ptr; idxs = map_ref_list idxs;
              gate = map_ref_opt gate; dtype }
  | Store { dst; value } -> Store { dst = map_ref dst; value = map_ref value }
  | Vectorize { srcs; dtype } -> Vectorize { srcs = map_ref_list srcs; dtype }
  | Cast { src; dtype } -> Cast { src = map_ref src; dtype }
  | Bitcast { src; dtype } -> Bitcast { src = map_ref src; dtype }
  | Unary { op; src; dtype } -> Unary { op; src = map_ref src; dtype }
  | Binary { op; lhs; rhs; dtype } ->
      Binary { op; lhs = map_ref lhs; rhs = map_ref rhs; dtype }
  | Ternary { op; a; b; c; dtype } ->
      Ternary { op; a = map_ref a; b = map_ref b; c = map_ref c; dtype }
  | Noop { src; dtype } -> Noop { src = map_ref_opt src; dtype }
  | Bufferize { src; ranges; dtype; opts } ->
      Bufferize { src = map_ref src; ranges = map_ref_list ranges; dtype; opts }
  | Define_local _ -> instr

let intern (program : view array) =
  let len = Array.length program in
  let module Tbl = Hashtbl.Make (struct
    type t = view

    let equal = ( = )
    let hash x = Hashtbl.hash_param 100 100 x
  end) in
  let tbl = Tbl.create (len * 2) in
  let map = Array.make len (-1) in
  let next = ref 0 in
  let acc = ref [] in
  for i = 0 to len - 1 do
    let mapped = map_children (fun r -> map.(r)) program.(i) in
    match Tbl.find_opt tbl mapped with
    | Some idx -> map.(i) <- idx
    | None ->
        let idx = !next in
        incr next;
        map.(i) <- idx;
        Tbl.add tbl mapped idx;
        acc := mapped :: !acc
  done;
  Array.of_list (List.rev !acc)
let view program id = program.(id)
let length program = Array.length program

let dtype program id =
  if id < 0 || id >= Array.length program then None else node_dtype program.(id)

let children program id = children_of program.(id)

let const builder ?(srcs = []) value =
  emit builder (Const { value; dtype = Const.dtype value; srcs })

let vconst builder ~values ~dtype ?(srcs = []) () =
  emit builder (Vconst { values; dtype; srcs })

let unique builder ~id = emit builder (Unique { id })
let lunique builder ~id = emit builder (Lunique { id })
let device builder device = emit builder (Device { device })
let sink builder ?kernel_info srcs = emit builder (Sink { srcs; kernel_info })
let group builder srcs = emit builder (Group { srcs })

let after builder ~src ~deps =
  let dtype = Option.value ~default:Dtype.void (dtype_of_id builder src) in
  emit builder (After { src; deps; dtype })

let buffer builder ~unique ~device ~size ~dtype =
  emit builder (Buffer { unique; device; size; dtype })

let buffer_view builder ~src ~size ~offset ~dtype =
  emit builder (Buffer_view { src; size; offset; dtype })

let define_var builder ~name ~lo ~hi ?(dtype = Dtype.index) () =
  emit builder (Define_var { name; lo; hi; dtype })

let bind builder ~var ?value () =
  let dtype = value_dtype_exn builder "bind" var in
  emit builder (Bind { var; value; dtype })

let param builder ~slot ~dtype ?shape ?device () =
  emit builder (Param { slot; dtype; shape; device })

let call builder ~callee ~args ~info ~dtype =
  emit builder (Call { callee; args; info; dtype })

let detach builder ~src =
  emit builder (Detach { src; dtype = value_dtype_exn builder "detach" src })

let contiguous builder ~src ?(ranges = []) () =
  emit builder
    (Contiguous
       { src; ranges; dtype = value_dtype_exn builder "contiguous" src })

let contiguous_backward builder ~src =
  emit builder
    (Contiguous_backward
       { src; dtype = value_dtype_exn builder "contiguous_backward" src })

let copy builder ~src ~device () =
  emit builder
    (Copy { src; device; dtype = value_dtype_exn builder "copy" src })

let allreduce builder ~src ~device ~op =
  emit builder
    (Allreduce
       { src; device; op; dtype = value_dtype_exn builder "allreduce" src })

let multi builder ~src ~axis =
  emit builder
    (Multi { src; axis; dtype = value_dtype_exn builder "multi" src })

let mstack builder ~srcs =
  match srcs with
  | [] -> invalid_arg "Tensor.mstack expects at least one source"
  | src :: _ ->
      emit builder
        (Mstack { srcs; dtype = value_dtype_exn builder "mstack" src })

let mselect builder ~src ~index =
  emit builder
    (Mselect { src; index; dtype = value_dtype_exn builder "mselect" src })


let reduce_axis builder ~src ~op ~axes =
  let dtype = value_dtype_exn builder "reduce_axis" src in
  emit builder (Reduce_axis { src; op; axes; dtype })

let reduce builder ~src ~ranges ~op ~dtype =
  emit builder (Reduce { src; ranges; op; dtype })

let reshape builder ~src ~shape =
  emit builder
    (Reshape { src; shape; dtype = value_dtype_exn builder "reshape" src })

let expand builder ~src ~shape =
  emit builder
    (Expand { src; shape; dtype = value_dtype_exn builder "expand" src })

let pad builder ~src ~before ~after =
  emit builder
    (Pad { src; before; after; dtype = value_dtype_exn builder "pad" src })

let shrink builder ~src ~before ~after =
  emit builder
    (Shrink { src; before; after; dtype = value_dtype_exn builder "shrink" src })

let permute builder ~src ~order =
  emit builder
    (Permute { src; order; dtype = value_dtype_exn builder "permute" src })

let flip builder ~src ~dims =
  emit builder (Flip { src; dims; dtype = value_dtype_exn builder "flip" src })

let range builder ~size ~axis ?(sub = []) ~kind ?(dtype = Dtype.index) () =
  emit builder (Range { size; dtype; axis; sub; kind })

let end_ builder ~value ~ranges = emit builder (End { value; ranges })

let index builder ~ptr ~idxs ?gate ~dtype () =
  emit builder (Index { ptr; idxs; gate; dtype })

let store builder ~dst ~value = emit builder (Store { dst; value })

let assign builder ~target ~value ?(extras = []) () =
  let store_id = store builder ~dst:target ~value in
  after builder ~src:target ~deps:(store_id :: extras)

let vectorize builder ~srcs =
  match srcs with
  | [] -> invalid_arg "Tensor.vectorize expects at least one source"
  | src :: rest ->
      let scalar = Dtype.scalar_of (value_dtype_exn builder "vectorize" src) in
      emit builder
        (Vectorize { srcs; dtype = Dtype.vec scalar (1 + List.length rest) })

let cast builder ~src ~dtype = emit builder (Cast { src; dtype })
let bitcast builder ~src ~dtype = emit builder (Bitcast { src; dtype })

let unary builder ~op ~src =
  emit builder (Unary { op; src; dtype = value_dtype_exn builder "unary" src })

let binary builder ~op ~lhs ~rhs =
  let lhs_dtype = value_dtype_exn builder "binary" lhs in
  let dtype =
    match op with
    | `Cmplt | `Cmpeq | `Cmpne ->
        Dtype.vec Dtype.bool (Dtype.count lhs_dtype)
    | _ -> lhs_dtype
  in
  emit builder (Binary { op; lhs; rhs; dtype })

let ternary builder ~op ~a ~b ~c =
  let dtype =
    match op with
    | `Where -> value_dtype_exn builder "ternary" b
    | `Mulacc -> value_dtype_exn builder "ternary" a
  in
  emit builder (Ternary { op; a; b; c; dtype })

let noop builder ?src ~dtype () = emit builder (Noop { src; dtype })

let bufferize builder ~src ~ranges ~dtype ~opts =
  emit builder (Bufferize { src; ranges; dtype; opts })

let invalid_index builder ~dtype = emit builder (Invalid_index { dtype })

let define_local builder ~size ~dtype =
  emit builder (Define_local { size; dtype })

let barrier builder = emit builder Barrier

let shape builder shape =
  let emit_dim = function
    | Shape.Static dim -> const builder ~srcs:[] (Const.int Dtype.index dim)
    | Shape.Symbol { name; lo; hi } -> define_var builder ~name ~lo ~hi ()
  in
  match Shape.dims shape with
  | [] -> vconst builder ~values:[] ~dtype:(Dtype.vec Dtype.index 0) ()
  | [ dim ] -> emit_dim dim
  | dims ->
      let srcs = List.map emit_dim dims in
      emit builder
        (Vectorize { srcs; dtype = Dtype.vec Dtype.index (List.length srcs) })

let int_of_const value =
  match Const.view value with
  | Int n -> (try Some (Int64.to_int n) with Failure _ -> None)
  | _ -> None

let extract_int_shape (program : t) id =
  match program.(id) with
  | Vectorize { srcs; _ } ->
      let dims =
        List.filter_map
          (fun s ->
            match program.(s) with
            | Const { value; _ } -> int_of_const value
            | Define_var { lo; _ } -> Some lo
            | _ -> None)
          srcs
      in
      if List.length dims = List.length srcs then Some dims else None
  | Const { value; _ } -> Option.map (fun d -> [ d ]) (int_of_const value)
  | Vconst { values = []; _ } -> Some []
  | _ -> None

let extract_marg (program : t) = function
  | Reshape { shape; _ } | Expand { shape; _ } -> extract_int_shape program shape
  | _ -> None

let extract_marg_pairs (program : t) = function
  | Pad { before; after; _ } | Shrink { before; after; _ } -> (
      match extract_int_shape program before, extract_int_shape program after with
      | Some bs, Some es -> Some (List.combine bs es)
      | _ -> None)
  | _ -> None

(* Validation *)

let check_dtype_eq fail i ~ctx ~expected ~got =
  match (expected, got) with
  | Some expected, Some got when Dtype.equal expected got -> ()
  | Some expected, Some got ->
      fail i
        (Printf.sprintf "%s: expected %s, got %s" ctx (Dtype.to_string expected)
           (Dtype.to_string got))
  | None, _ -> fail i (Printf.sprintf "%s: expected dtype not available" ctx)
  | _, None -> fail i (Printf.sprintf "%s: operand dtype not available" ctx)

let check_dtype_match fail i ~ctx left right =
  match (left, right) with
  | Some left, Some right when Dtype.equal left right -> ()
  | Some _, Some _ ->
      fail i (Printf.sprintf "%s: operand dtypes don't match" ctx)
  | _ -> fail i (Printf.sprintf "%s: operand dtype not available" ctx)

let check_bool_scalar fail get_dtype i ~ctx r =
  match get_dtype r with
  | Some (dt : Dtype.t) when Dtype.scalar dt = Dtype.Bool && Dtype.count dt = 1 -> ()
  | Some _ -> fail i (Printf.sprintf "%s must be bool scalar" ctx)
  | None -> fail i (Printf.sprintf "%s dtype not available" ctx)

let check_shift_rhs fail get_dtype i rhs dtype =
  match get_dtype rhs with
  | Some rhs_dtype when Dtype.equal rhs_dtype dtype -> ()
  | Some rhs_dtype when Dtype.equal rhs_dtype Dtype.uint32 -> ()
  | Some _ -> fail i "shift rhs must match lhs dtype or be uint32"
  | None -> fail i "shift rhs dtype not available"

let check_index_like fail get_dtype i ~ctx r =
  match get_dtype r with
  | Some (dt : Dtype.t)
    when Dtype.count dt = 1 && (Dtype.scalar dt = Dtype.Index || Dtype.scalar dt = Dtype.Int32) ->
      ()
  | Some _ -> fail i (Printf.sprintf "%s must be index or int32 scalar" ctx)
  | None -> fail i (Printf.sprintf "%s dtype not available" ctx)

let validate (program : t) : unit =
  let fail i msg =
    failwith (Printf.sprintf "Tensor.validate: instruction %d: %s" i msg)
  in
  let get_dtype r =
    if r < 0 || r >= Array.length program then None else node_dtype program.(r)
  in
  let check_dtype_eq = check_dtype_eq fail in
  let check_dtype_match = check_dtype_match fail in
  let check_bool_scalar = check_bool_scalar fail get_dtype in
  let check_index_like = check_index_like fail get_dtype in
  let check_shift_rhs = check_shift_rhs fail get_dtype in
  let check_index_scalar i ~ctx r =
    match get_dtype r with
    | Some dt when Dtype.scalar dt = Dtype.Index && Dtype.count dt = 1 -> ()
    | Some _ -> fail i (Printf.sprintf "%s must be index scalar" ctx)
    | None -> fail i (Printf.sprintf "%s dtype not available" ctx)
  in
  let check_index_vector i ~ctx r =
    match get_dtype r with
    | Some dt when Dtype.scalar dt = Dtype.Index -> ()
    | Some _ -> fail i (Printf.sprintf "%s must be index vector" ctx)
    | None -> fail i (Printf.sprintf "%s dtype not available" ctx)
  in
  let check_src_dtype i ~ctx src dtype =
    check_dtype_eq i ~ctx ~expected:(Some dtype) ~got:(get_dtype src)
  in
  Array.iteri
    (fun i instr ->
      List.iter
        (fun r ->
          if r < 0 || r >= i then
            fail i
              (Printf.sprintf "references %%%d (out of bounds or forward)" r))
        (children_of instr);
      match instr with
      | Sink _ | Group _ | Unique _ | Lunique _ | Device _ | Invalid_index _
      | Barrier ->
          ()
      | Buffer { unique; device; size; _ } -> (
          if size < 0 then fail i "Buffer size must be non-negative";
          (match program.(unique) with
          | Unique _ | Lunique _ -> ()
          | _ -> fail i "Buffer unique must reference Unique/Lunique");
          match program.(device) with
          | Device _ -> ()
          | _ -> fail i "Buffer device must reference Device")
      | Buffer_view { src; size; offset; _ } -> (
          if size < 0 then fail i "Buffer_view size must be non-negative";
          if offset < 0 then fail i "Buffer_view offset must be non-negative";
          match program.(src) with
          | Buffer _ | Index _ -> ()
          | _ -> fail i "Buffer_view src must be Buffer or Index")
      | Const { value; dtype; _ } -> (
          match Const.view value with
          | Bool _ ->
              if Dtype.scalar dtype <> Dtype.Bool then
                fail i "Bool const must have bool dtype"
          | Int _ ->
              if not (Dtype.is_int dtype) then
                fail i "Int const must have int/index dtype"
          | Float _ ->
              if not (Dtype.is_float dtype) then
                fail i "Float const must have float dtype")
      | Vconst { values; dtype; _ } ->
          if Dtype.count dtype <> List.length values then
            fail i "Vconst values must match vector width";
          List.iter
            (fun value ->
              match Const.view value with
              | Bool _ ->
                  if Dtype.scalar dtype <> Dtype.Bool then
                    fail i "Vconst bool elements must have bool dtype"
              | Int _ ->
                  if not (Dtype.is_int dtype) then
                    fail i "Vconst int elements must have int/index dtype"
              | Float _ ->
                  if not (Dtype.is_float dtype) then
                    fail i "Vconst float elements must have float dtype")
            values
      | Define_var { lo; hi; dtype; _ } ->
          if Dtype.count dtype <> 1 then fail i "Define_var must be scalar";
          if not (Dtype.is_int dtype || Dtype.scalar dtype = Dtype.Index) then
            fail i "Define_var must be int/index";
          if lo > hi then fail i "Define_var bounds invalid (lo > hi)"
      | Bind { var; value; dtype } ->
          (match program.(var) with
          | Define_var { dtype = vdt; _ } ->
              check_dtype_eq i ~ctx:"Bind dtype" ~expected:(Some vdt)
                ~got:(Some dtype)
          | _ -> fail i "Bind var must reference Define_var");
          Option.iter (fun v -> check_src_dtype i ~ctx:"Bind value" v dtype) value
      | Param { shape; device; _ } ->
          Option.iter (check_index_vector i ~ctx:"Param shape") shape;
          Option.iter
            (fun d ->
              match program.(d) with
              | Device _ -> ()
              | _ -> fail i "Param device must reference Device")
            device
      | Call { callee; dtype; _ } -> (
          match callee with
          | Ref fn -> check_src_dtype i ~ctx:"Call dtype" fn dtype
          | Ast _ -> ())
      | After { src; dtype; _ } ->
          check_src_dtype i ~ctx:"After dtype" src dtype
      | Detach { src; dtype }
      | Contiguous_backward { src; dtype }
      | Multi { src; dtype; _ }
      | Mselect { src; dtype; _ } ->
          check_src_dtype i ~ctx:"dtype" src dtype
      | Reduce_axis { src; axes; dtype; _ } ->
          check_src_dtype i ~ctx:"Reduce_axis dtype" src dtype;
          if axes = [] then fail i "Reduce_axis must have at least one axis";
          if List.length axes <> List.length (List.sort_uniq compare axes) then
            fail i "Reduce_axis axes must be unique"
      | Permute { src; order; dtype } ->
          check_src_dtype i ~ctx:"Permute dtype" src dtype;
          let n = List.length order in
          if List.sort compare order <> List.init n Fun.id then
            fail i
              (Printf.sprintf
                 "Permute order must be a valid permutation of 0..%d" (n - 1))
      | Flip { src; dtype; _ } -> check_src_dtype i ~ctx:"Flip dtype" src dtype
      | Copy { src; device; dtype } -> (
          check_src_dtype i ~ctx:"Copy dtype" src dtype;
          match program.(device) with
          | Device _ -> ()
          | _ -> fail i "Copy device must reference a Device node")
      | Allreduce { src; device; dtype; _ } -> (
          check_src_dtype i ~ctx:"Allreduce dtype" src dtype;
          match program.(device) with
          | Device _ -> ()
          | _ -> fail i "Allreduce device must reference Device")
      | Contiguous { src; ranges; dtype } ->
          check_src_dtype i ~ctx:"Contiguous dtype" src dtype;
          List.iter (check_index_scalar i ~ctx:"Contiguous range") ranges
      | Mstack { srcs; dtype } ->
          if srcs = [] then fail i "Mstack must have srcs";
          List.iter (fun r -> check_src_dtype i ~ctx:"Mstack src" r dtype) srcs
      | Reduce { src; ranges; dtype; _ } ->
          check_src_dtype i ~ctx:"Reduce dtype" src dtype;
          List.iter (check_index_scalar i ~ctx:"Reduce range") ranges
      | Reshape { src; shape; dtype } -> (
          check_src_dtype i ~ctx:"Reshape dtype" src dtype;
          check_index_vector i ~ctx:"Shape" shape;
          match extract_int_shape program shape with
          | Some dims ->
              if List.exists (fun d -> d < 0) dims then
                fail i "Reshape shape must not contain negative numbers"
          | None -> ())
      | Expand { src; shape; dtype } ->
          check_src_dtype i ~ctx:"Expand dtype" src dtype;
          check_index_vector i ~ctx:"Shape" shape
      | Pad { src; before; after; dtype } | Shrink { src; before; after; dtype }
        -> (
          check_src_dtype i ~ctx:"Pad/Shrink dtype" src dtype;
          check_index_vector i ~ctx:"Pad/Shrink before" before;
          check_index_vector i ~ctx:"Pad/Shrink after" after;
          match (get_dtype before, get_dtype after) with
          | Some bdt, Some adt when Dtype.count bdt = Dtype.count adt -> ()
          | Some _, Some _ ->
              fail i "Pad/Shrink before/after vector width mismatch"
          | _ -> fail i "Pad/Shrink dtype not available")
      | Range { size; dtype; _ } ->
          if not (Dtype.is_int dtype) then fail i "Range must have int/index";
          if Dtype.count dtype <> 1 then fail i "Range must be scalar";
          check_src_dtype i ~ctx:"Range size" size dtype
      | End { ranges; _ } ->
          List.iter (check_index_like i ~ctx:"End range") ranges
      | Index { idxs; gate; _ } ->
          if idxs = [] then fail i "Index must have at least one index";
          List.iter (check_index_scalar i ~ctx:"Index operand") idxs;
          Option.iter (check_bool_scalar i ~ctx:"Index gate") gate
      | Store { dst; value } -> (
          match get_dtype dst with
          | Some dst_dtype ->
              check_dtype_eq i ~ctx:"Store value" ~expected:(Some dst_dtype)
                ~got:(get_dtype value)
          | None -> fail i "Store dst dtype not available")
      | Vectorize { srcs; dtype } ->
          if srcs = [] then fail i "Vectorize must have at least one operand";
          if Dtype.count dtype <> List.length srcs then
            fail i "Vectorize dtype count must match operand count";
          let scalar_ty = Dtype.scalar dtype in
          List.iter
            (fun r ->
              match get_dtype r with
              | Some dt when Dtype.count dt = 1 && Dtype.scalar dt = scalar_ty -> ()
              | Some _ -> fail i "Vectorize operands must be scalar and match"
              | None -> fail i "Vectorize operand dtype not available")
            srcs
      | Cast { src; dtype } -> (
          match get_dtype src with
          | Some sdt ->
              if Dtype.count sdt <> Dtype.count dtype then
                fail i "Cast must preserve vector width"
          | None -> fail i "Cast src dtype not available")
      | Bitcast { src; _ } -> (
          match get_dtype src with
          | Some _ -> ()
          | None -> fail i "Bitcast src dtype not available")
      | Unary { src; dtype; _ } ->
          check_src_dtype i ~ctx:"Unary operand" src dtype
      | Binary
          {
            op =
              ( `Add | `Sub | `Mul | `Fdiv | `Max | `Pow | `And | `Or | `Xor
              | `Threefry );
            lhs;
            rhs;
            dtype;
          } ->
          check_dtype_match i ~ctx:"Binary operands" (get_dtype lhs)
            (get_dtype rhs);
          check_dtype_eq i ~ctx:"Binary result" ~expected:(Some dtype)
            ~got:(get_dtype lhs)
      | Binary { op = `Idiv | `Mod; lhs; rhs; dtype } ->
          check_dtype_match i ~ctx:"Binary operands" (get_dtype lhs)
            (get_dtype rhs);
          check_dtype_eq i ~ctx:"Binary result" ~expected:(Some dtype)
            ~got:(get_dtype lhs);
          if not (Dtype.is_int dtype) then
            fail i "Idiv/Mod must have int/index dtype"
      | Binary { op = `Shl | `Shr; lhs; rhs; dtype } ->
          check_src_dtype i ~ctx:"Shift lhs" lhs dtype;
          check_shift_rhs i rhs dtype;
          if not (Dtype.is_int dtype) then
            fail i "Shift must have int/index dtype"
      | Binary { op = `Cmplt | `Cmpeq | `Cmpne; lhs; rhs; dtype } ->
          if Dtype.scalar dtype <> Dtype.Bool then
            fail i "Comparison must produce bool";
          check_dtype_match i ~ctx:"Comparison operands" (get_dtype lhs)
            (get_dtype rhs)
      | Ternary { op = `Where; a = cond; b = then_; c = else_; dtype } ->
          check_bool_scalar i ~ctx:"Where condition" cond;
          check_dtype_match i ~ctx:"Where arms" (get_dtype then_)
            (get_dtype else_);
          check_dtype_eq i ~ctx:"Where result" ~expected:(Some dtype)
            ~got:(get_dtype then_)
      | Ternary { op = `Mulacc; a; b; c; dtype } ->
          check_dtype_match i ~ctx:"Mulacc a/b" (get_dtype a) (get_dtype b);
          check_dtype_match i ~ctx:"Mulacc a/c" (get_dtype a) (get_dtype c);
          check_dtype_eq i ~ctx:"Mulacc result" ~expected:(Some dtype)
            ~got:(get_dtype a)
      | Noop { src; dtype } ->
          Option.iter (fun s -> check_src_dtype i ~ctx:"Noop src" s dtype) src
      | Bufferize { src; ranges; dtype; _ } ->
          check_src_dtype i ~ctx:"Bufferize src" src dtype;
          List.iter (check_index_scalar i ~ctx:"Bufferize range") ranges
      | Define_local { size; _ } ->
          if size <= 0 then fail i "Define_local size must be positive")
    program

let rebuild rewrite program =
  let b = create () in
  let remap = Array.make (Array.length program) (-1) in
  Array.iteri
    (fun id instr ->
      let mapped = map_children (fun r -> remap.(r)) instr in
      let v = match rewrite id mapped with Some v -> v | None -> mapped in
      remap.(id) <- emit b v)
    program;
  finish b |> intern

let rewrite_fixpoint ?(max_iters = 16) rewrite program =
  let rec loop n prog =
    if n >= max_iters then
      Printf.ksprintf failwith
        "Tensor.rewrite_fixpoint: fixpoint not reached after %d passes" max_iters;
    let prog' = rebuild rewrite prog in
    if prog' = prog then prog else loop (n + 1) prog'
  in
  loop 0 program

let rebuild_grow rewrite program =
  let b = create () in
  let remap = Array.make (Array.length program) (-1) in
  let lookup id =
    if id >= 0 && id < b.len then b.data.(id) else Group { srcs = [] }
  in
  Array.iteri
    (fun id instr ->
      let mapped = map_children (fun r -> remap.(r)) instr in
      let v =
        match rewrite ~lookup (emit b) id mapped with
        | Some v -> v
        | None -> mapped
      in
      remap.(id) <- emit b v)
    program;
  finish b |> intern

let rewrite_fixpoint_grow ?(max_iters = 16) rewrite program =
  let rec loop n prog =
    if n >= max_iters then
      Printf.ksprintf failwith
        "Tensor.rewrite_fixpoint_grow: fixpoint not reached after %d passes" max_iters;
    let prog' = rebuild_grow rewrite prog in
    if prog' = prog then prog else loop (n + 1) prog'
  in
  loop 0 program

let merge_builder (program : t) (extra : builder) =
  let base_len = Array.length program in
  let b = create () in
  Array.iter (fun v -> ignore (emit b v)) program;
  let shift id = id + base_len in
  Array.iter
    (fun v -> ignore (emit b (map_children shift v)))
    (finish extra);
  (finish b, shift)

let compute_shapes (program : t) =
  let n = Array.length program in
  let shapes = Array.make n None in
  let sh id = shapes.(id) in
  for i = 0 to n - 1 do
    shapes.(i) <-
      (match program.(i) with
      | Sink _ | Group _ | Unique _ | Lunique _ | Device _ | Range _ | Store _
      | End _ | Barrier | Define_local _ ->
          None
      | Const _ | Vconst _ | Define_var _ | Bind _ | Invalid_index _ -> Some []
      | Buffer { size; _ } | Buffer_view { size; _ } -> Some [ size ]
      | Bufferize { ranges; _ } ->
          let dims =
            List.filter_map
              (fun r ->
                match program.(r) with
                | Range { size = sz; _ } -> (
                    match program.(sz) with
                    | Const { value; _ } -> int_of_const value
                    | _ -> None)
                | Const _ -> Some 1
                | _ -> None)
              ranges
          in
          if List.length dims = List.length ranges then Some dims else None
      | Param { shape; _ } -> Option.bind shape (extract_int_shape program)
      | Reduce { src; _ }
      | Mstack { srcs = src :: _; _ }
      | Mselect { src; _ }
      | Detach { src; _ }
      | Contiguous { src; _ }
      | Contiguous_backward { src; _ }
      | After { src; _ }
      | Flip { src; _ }
      | Multi { src; _ }
      | Unary { src; _ }
      | Cast { src; _ }
      | Bitcast { src; _ }
      | Copy { src; _ }
      | Allreduce { src; _ } ->
          sh src
      | Binary { lhs; _ } -> sh lhs
      | Ternary { b; _ } -> sh b
      | Reshape { shape; _ } | Expand { shape; _ } ->
          extract_int_shape program shape
      | Permute { src; order; _ } ->
          Option.map (fun ps -> List.map (List.nth ps) order) (sh src)
      | Pad { src; _ } as v -> (
          match sh src, extract_marg_pairs program v with
          | Some ps, Some pairs ->
              Some (List.map2 (fun s (b, e) -> s + b + e) ps pairs)
          | _ -> None)
      | Shrink _ as v -> (
          match extract_marg_pairs program v with
          | Some pairs -> Some (List.map (fun (b, e) -> e - b) pairs)
          | _ -> None)
      | Reduce_axis { src; axes; _ } ->
          Option.map
            (List.mapi (fun j s -> if List.mem j axes then 1 else s))
            (sh src)
      | Noop { src; _ } -> Option.bind src sh
      | Vectorize _ | Index _ | Mstack { srcs = []; _ } -> None
      | Call { callee = Ref fn; _ } -> sh fn
      | Call { callee = Ast _; _ } -> None)
  done;
  shapes

let compute_devices (program : t) =
  let n = Array.length program in
  let devices = Array.make n None in
  let dev id = devices.(id) in
  for i = 0 to n - 1 do
    devices.(i) <-
      (match program.(i) with
      | Device { device } -> Some device
      | Bufferize { opts; _ } -> (
          match opts.device with
          | Some (Device_single s) -> Some (Single s)
          | Some (Device_multi ss) -> Some (Multi ss)
          | _ -> None)
      | After { src; _ } -> dev src
      | Mselect { src; index; _ } -> (
          match dev src with
          | Some (Multi devs) ->
              (try Some (Single (List.nth devs index)) with Failure _ -> None)
          | d -> d)
      | Mstack { srcs; _ } ->
          let ds =
            List.filter_map
              (fun s ->
                match dev s with Some (Single d) -> Some d | _ -> None)
              srcs
          in
          if List.length ds = List.length srcs then Some (Multi ds) else None
      | Copy { device = d; _ }
      | Buffer { device = d; _ }
      | Allreduce { device = d; _ } ->
          dev d
      | _ -> List.find_map dev (children_of program.(i)))
  done;
  devices

let rec base (program : t) id =
  match program.(id) with
  | Reshape { src; _ }
  | Expand { src; _ }
  | Pad { src; _ }
  | Shrink { src; _ }
  | Permute { src; _ }
  | Flip { src; _ }
  | Multi { src; _ }
  | Detach { src; _ } ->
      base program src
  | _ -> id

let consumer_map (program : t) =
  let n = Array.length program in
  let consumers = Array.make n [] in
  for i = 0 to n - 1 do
    List.iter
      (fun c -> consumers.(c) <- i :: consumers.(c))
      (children_of program.(i))
  done;
  Array.map List.rev consumers

let backward_slice (program : t) root =
  let n = Array.length program in
  let visited = Array.make n false in
  let rec visit id =
    if not visited.(id) then begin
      visited.(id) <- true;
      List.iter visit (children_of program.(id))
    end
  in
  visit root;
  let acc = ref [] in
  for i = n - 1 downto 0 do
    if visited.(i) then acc := i :: !acc
  done;
  !acc

let pp_device fmt = function
  | Single s -> Format.pp_print_string fmt s
  | Multi devs ->
      Format.fprintf fmt "[%a]"
        (Format.pp_print_list ~pp_sep:pp_comma Format.pp_print_string)
        devs

let pp_view fmt = function
  | Sink { srcs; kernel_info = _ } -> Format.fprintf fmt "sink %a" pp_refs srcs
  | Group { srcs } -> Format.fprintf fmt "group %a" pp_refs srcs
  | After { src; deps; dtype } ->
      Format.fprintf fmt "after %a, deps=[%a] : %a" pp_ref src pp_refs deps
        Dtype.pp dtype
  | Unique { id } -> Format.fprintf fmt "unique #%d" id
  | Lunique { id } -> Format.fprintf fmt "lunique #%d" id
  | Device { device } -> Format.fprintf fmt "device %a" pp_device device
  | Buffer { unique; device; size; dtype } ->
      Format.fprintf fmt "buffer %a, %a, size=%d : %a" pp_ref unique pp_ref
        device size Dtype.pp dtype
  | Buffer_view { src; size; offset; dtype } ->
      Format.fprintf fmt "buffer_view %a, size=%d, offset=%d : %a" pp_ref src
        size offset Dtype.pp dtype
  | Const { value; dtype; srcs } ->
      Format.fprintf fmt "const %a : %a" Const.pp value Dtype.pp dtype;
      if srcs <> [] then Format.fprintf fmt " [%a]" pp_refs srcs
  | Vconst { values; dtype; srcs } ->
      Format.fprintf fmt "vconst [%a] : %a"
        (Format.pp_print_list ~pp_sep:pp_comma Const.pp)
        values Dtype.pp dtype;
      if srcs <> [] then Format.fprintf fmt " [%a]" pp_refs srcs
  | Define_var { name; lo; hi; dtype } ->
      Format.fprintf fmt "define_var %s : %a [%d..%d]" name Dtype.pp dtype lo hi
  | Bind { var; value; dtype } ->
      Format.fprintf fmt "bind %a" pp_ref var;
      Option.iter (fun v -> Format.fprintf fmt " = %a" pp_ref v) value;
      Format.fprintf fmt " : %a" Dtype.pp dtype
  | Param { slot; dtype; shape; device } ->
      Format.fprintf fmt "param %d : %a" slot Dtype.pp dtype;
      Option.iter (fun s -> Format.fprintf fmt " shape=%a" pp_ref s) shape;
      Option.iter (fun d -> Format.fprintf fmt " device=%a" pp_ref d) device
  | Call { callee; args; info; dtype } ->
      let n ast = List.length (Kernel.toposort ast) in
      (match callee with
      | Ref fn -> Format.fprintf fmt "call %a(" pp_ref fn
      | Ast ast ->
          match info.name with
          | Some name -> Format.fprintf fmt "call <kernel:%s/%d>(" name (n ast)
          | None -> Format.fprintf fmt "call <kernel:%d>(" (n ast));
      Format.fprintf fmt "%a)" pp_refs args;
      if info.precompile then Format.fprintf fmt " [precompile]";
      if info.metadata <> [] then
        Format.fprintf fmt " [meta=%d]" (List.length info.metadata);
      Format.fprintf fmt " : %a" Dtype.pp dtype
  | Detach { src; dtype } ->
      Format.fprintf fmt "detach %a : %a" pp_ref src Dtype.pp dtype
  | Contiguous { src; ranges; dtype } ->
      Format.fprintf fmt "contiguous %a" pp_ref src;
      if ranges <> [] then Format.fprintf fmt ", ranges=[%a]" pp_refs ranges;
      Format.fprintf fmt " : %a" Dtype.pp dtype
  | Contiguous_backward { src; dtype } ->
      Format.fprintf fmt "contiguous_backward %a : %a" pp_ref src Dtype.pp dtype
  | Copy { src; device; dtype } ->
      Format.fprintf fmt "copy %a, %a : %a" pp_ref src pp_ref device Dtype.pp dtype
  | Allreduce { src; device; op; dtype } ->
      Format.fprintf fmt "allreduce.%a %a, %a : %a" Op.pp_reduce op pp_ref src
        pp_ref device Dtype.pp dtype
  | Multi { src; axis; dtype } ->
      Format.fprintf fmt "multi %a, axis=%d : %a" pp_ref src axis Dtype.pp dtype
  | Mstack { srcs; dtype } ->
      Format.fprintf fmt "mstack %a : %a" pp_refs srcs Dtype.pp dtype
  | Mselect { src; index; dtype } ->
      Format.fprintf fmt "mselect %a, index=%d : %a" pp_ref src index Dtype.pp
        dtype
  | Reduce_axis { src; op; axes; dtype } ->
      Format.fprintf fmt "reduce_axis.%a %a, axes=[%a] : %a" Op.pp_reduce op
        pp_ref src
        (Format.pp_print_list ~pp_sep:pp_comma Format.pp_print_int)
        axes Dtype.pp dtype
  | Reduce { src; ranges; op; dtype } ->
      Format.fprintf fmt "reduce.%a %a, ranges=[%a] : %a" Op.pp_reduce op pp_ref
        src pp_refs ranges Dtype.pp dtype
  | Reshape { src; shape; dtype } ->
      Format.fprintf fmt "reshape %a, %a : %a" pp_ref src pp_ref shape Dtype.pp
        dtype
  | Expand { src; shape; dtype } ->
      Format.fprintf fmt "expand %a, %a : %a" pp_ref src pp_ref shape Dtype.pp
        dtype
  | Pad { src; before; after; dtype } ->
      Format.fprintf fmt "pad %a, %a, %a : %a" pp_ref src pp_ref before pp_ref
        after Dtype.pp dtype
  | Shrink { src; before; after; dtype } ->
      Format.fprintf fmt "shrink %a, %a, %a : %a" pp_ref src pp_ref before
        pp_ref after Dtype.pp dtype
  | Permute { src; order; dtype } ->
      Format.fprintf fmt "permute %a, [%a] : %a" pp_ref src
        (Format.pp_print_list ~pp_sep:pp_comma Format.pp_print_int)
        order Dtype.pp dtype
  | Flip { src; dims; dtype } ->
      Format.fprintf fmt "flip %a, [%a] : %a" pp_ref src
        (Format.pp_print_list ~pp_sep:pp_comma (fun fmt b ->
             Format.fprintf fmt "%b" b))
        dims Dtype.pp dtype
  | Range { size; dtype; axis; sub; kind } ->
      Format.fprintf fmt "range %a : %a [axis=%d, %a%a]" pp_ref size Dtype.pp
        dtype axis Axis_kind.pp kind
        (fun fmt sub ->
          if sub <> [] then
            Format.fprintf fmt ", sub=[%a]"
              (Format.pp_print_list ~pp_sep:(fun fmt () -> Format.fprintf fmt ";") Format.pp_print_int)
              sub)
        sub
  | End { value; ranges } ->
      Format.fprintf fmt "end %a, ranges=[%a]" pp_ref value pp_refs ranges
  | Index { ptr; idxs; gate; dtype } ->
      Format.fprintf fmt "index %a, %a%a : %a" pp_ref ptr pp_refs idxs
        (fun fmt -> function
          | None -> () | Some g -> Format.fprintf fmt " gate=%%%d" g)
        gate Dtype.pp dtype
  | Store { dst; value } ->
      Format.fprintf fmt "store %a, %a" pp_ref dst pp_ref value
  | Vectorize { srcs; dtype } ->
      Format.fprintf fmt "vec %a : %a" pp_refs srcs Dtype.pp dtype
  | Cast { src; dtype } ->
      Format.fprintf fmt "cast %a : %a" pp_ref src Dtype.pp dtype
  | Bitcast { src; dtype } ->
      Format.fprintf fmt "bitcast %a : %a" pp_ref src Dtype.pp dtype
  | Unary { op; src; dtype } ->
      Format.fprintf fmt "%a %a : %a" Op.pp_unary op pp_ref src Dtype.pp dtype
  | Binary { op; lhs; rhs; dtype } ->
      Format.fprintf fmt "%a %a, %a : %a" Op.pp_binary op pp_ref lhs pp_ref rhs
        Dtype.pp dtype
  | Ternary { op; a; b; c; dtype } ->
      Format.fprintf fmt "%a %a, %a, %a : %a" Op.pp_ternary op pp_ref a pp_ref b
        pp_ref c Dtype.pp dtype
  | Noop { src; dtype } ->
      Format.fprintf fmt "noop";
      Option.iter (fun s -> Format.fprintf fmt " %a" pp_ref s) src;
      Format.fprintf fmt " : %a" Dtype.pp dtype
  | Bufferize { src; ranges; dtype; _ } ->
      Format.fprintf fmt "bufferize %a, ranges=[%a] : %a" pp_ref src pp_refs
        ranges Dtype.pp dtype
  | Invalid_index { dtype } ->
      Format.fprintf fmt "invalid_index : %a" Dtype.pp dtype
  | Define_local { size; dtype } ->
      Format.fprintf fmt "define_local size=%d : %a" size Dtype.pp_ptr dtype
  | Barrier -> Format.fprintf fmt "barrier"

let pp fmt t =
  Array.iteri (fun i v -> Format.fprintf fmt "%3d: %a@\n" i pp_view v) t
