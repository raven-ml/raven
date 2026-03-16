(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Types *)

type bufferize_device =
  | Device_single of string
  | Device_multi of string list
  | Device_index of int

type estimate = Int of int | Symbolic of string
type estimates = { ops : estimate; lds : estimate; mem : estimate }
type sort = Value | Pointer | Index | Effect

module Opt = struct
  type t =
    | Local of { axis : int; amount : int }
    | Upcast of { axis : int; amount : int }
    | Unroll of { axis : int; amount : int }
    | Group of { axis : int; amount : int }
    | Grouptop of { axis : int; amount : int }
    | Thread of { axis : int; amount : int }
    | Nolocals
    | Tc of { axis : int; tc_select : int; tc_opt : int; use_tc : int }
    | Padto of { axis : int; amount : int }
    | Swap of { axis : int; with_axis : int }

  let to_string = function
    | Local { axis; amount } -> Printf.sprintf "LOCAL:%d:%d" axis amount
    | Upcast { axis; amount } -> Printf.sprintf "UPCAST:%d:%d" axis amount
    | Unroll { axis; amount } -> Printf.sprintf "UNROLL:%d:%d" axis amount
    | Group { axis; amount } -> Printf.sprintf "GROUP:%d:%d" axis amount
    | Grouptop { axis; amount } -> Printf.sprintf "GROUPTOP:%d:%d" axis amount
    | Thread { axis; amount } -> Printf.sprintf "THREAD:%d:%d" axis amount
    | Nolocals -> "NOLOCALS"
    | Tc { axis; tc_select; tc_opt; use_tc } ->
        Printf.sprintf "TC:%d:%d:%d:%d" axis tc_select tc_opt use_tc
    | Padto { axis; amount } -> Printf.sprintf "PADTO:%d:%d" axis amount
    | Swap { axis; with_axis } -> Printf.sprintf "SWAP:%d:%d" axis with_axis

  let pp fmt t = Format.pp_print_string fmt (to_string t)
end

type bufferize_opts = {
  device : bufferize_device option;
  addrspace : Dtype.addr_space;
  removable : bool;
}

type kernel_info = {
  name : string;
  axis_kinds : Axis_kind.t list;
  dont_use_locals : bool;
  applied_opts : Opt.t list;
  opts_to_apply : Opt.t list option;
  estimates : estimates option;
  metadata_tags : string list;
}

type t = view

and view =
  | Sink of { srcs : t list; kernel_info : kernel_info option }
  | Group of { srcs : t list }
  | After of { src : t; deps : t list }
  | Param of { idx : int; dtype : Dtype.ptr }
  | Param_image of { idx : int; dtype : Dtype.ptr; width : int; height : int }
  | Define_local of { size : int; dtype : Dtype.ptr }
  | Define_reg of { size : int; dtype : Dtype.ptr }
  | Define_var of { name : string; lo : int; hi : int; dtype : Dtype.t }
  | Bufferize of {
      src : t;
      ranges : t list;
      dtype : Dtype.ptr;
      opts : bufferize_opts;
    }
  | Const of { value : Const.t; dtype : Dtype.t }
  | Invalid_index of { dtype : Dtype.t }
  | Index of { ptr : t; idxs : t list; gate : t option; dtype : Dtype.ptr }
  | Ptrcat of { srcs : t list; dtype : Dtype.ptr }
  | Load of { src : t; alt : t option; dtype : Dtype.t }
  | Store of { dst : t; value : t; ranges : t list }
  | Unary of { op : Op.unary; src : t; dtype : Dtype.t }
  | Binary of { op : Op.binary; lhs : t; rhs : t; dtype : Dtype.t }
  | Ternary of { op : Op.ternary; a : t; b : t; c : t; dtype : Dtype.t }
  | Cast of { src : t; dtype : Dtype.t }
  | Bitcast of { src : t; dtype : Dtype.t }
  | Vectorize of { srcs : t list; dtype : Dtype.t }
  | Cat of { srcs : t list; dtype : Dtype.t }
  | Gep of { src : t; idx : int; dtype : Dtype.t }
  | Range of { size : t; dtype : Dtype.t; axis : int; kind : Axis_kind.t }
  | End of { value : t; ranges : t list }
  | Barrier
  | Special of { dim : Special_dim.t; size : t; dtype : Dtype.t }
  | Reduce of { op : Op.reduce; src : t; ranges : t list; dtype : Dtype.t }
  | Unroll of { src : t; axes : (int * int) list; dtype : Dtype.t }
  | Contract of { src : t; axes : (int * int) list; dtype : Dtype.t }
  | Wmma of {
      name : string;
      a : t;
      b : t;
      c : t;
      dtype : Dtype.t;
      dims : int * int * int;
      dtype_in : Dtype.scalar;
      dtype_out : Dtype.scalar;
      device : string;
      threads : int;
      upcast_axes : (int * int) list * (int * int) list * (int * int) list;
      reduce_axes : int list;
    }
  | Custom of { fmt : string; args : t list }
  | Custom_inline of { fmt : string; args : t list; dtype : Dtype.t }

(* Building *)

let view node = node
let sink ?kernel_info srcs = Sink { srcs; kernel_info }
let group srcs = Group { srcs }
let after ~src ~deps = After { src; deps }
let param ~idx ~dtype = Param { idx; dtype }

let param_image ~idx ~dtype ~width ~height =
  Param_image { idx; dtype; width; height }

let define_local ~size ~dtype = Define_local { size; dtype }
let define_reg ~size ~dtype = Define_reg { size; dtype }

let define_var ~name ~lo ~hi ?(dtype = Dtype.index) () =
  Define_var { name; lo; hi; dtype }

let bufferize ~src ~ranges ~dtype ~opts = Bufferize { src; ranges; dtype; opts }
let const value = Const { value; dtype = Const.dtype value }

let invalid_index ?(lanes = 1) () =
  Invalid_index { dtype = Dtype.vec Dtype.index lanes }

let rec get_ptr_dtype = function
  | Param { dtype; _ }
  | Param_image { dtype; _ }
  | Define_local { dtype; _ }
  | Define_reg { dtype; _ }
  | Bufferize { dtype; _ }
  | Index { dtype; _ }
  | Ptrcat { dtype; _ } ->
      Some dtype
  | After { src; _ } | Cast { src; _ } | Bitcast { src; _ } -> get_ptr_dtype src
  | _ -> None

let pointer_dtype_exn ctx node =
  match get_ptr_dtype node with
  | Some dtype -> dtype
  | None -> invalid_arg (Printf.sprintf "Kernel.%s expects a pointer node" ctx)

let rec node_dtype = function
  | Param { dtype; _ }
  | Param_image { dtype; _ }
  | Define_local { dtype; _ }
  | Define_reg { dtype; _ }
  | Bufferize { dtype; _ }
  | Index { dtype; _ }
  | Ptrcat { dtype; _ } ->
      Some dtype.base
  | Define_var { dtype; _ }
  | Const { dtype; _ }
  | Invalid_index { dtype; _ }
  | Load { dtype; _ }
  | Unary { dtype; _ }
  | Binary { dtype; _ }
  | Ternary { dtype; _ }
  | Cast { dtype; _ }
  | Bitcast { dtype; _ }
  | Vectorize { dtype; _ }
  | Cat { dtype; _ }
  | Gep { dtype; _ }
  | Range { dtype; _ }
  | Special { dtype; _ }
  | Reduce { dtype; _ }
  | Unroll { dtype; _ }
  | Contract { dtype; _ }
  | Wmma { dtype; _ }
  | Custom_inline { dtype; _ } ->
      Some dtype
  | Sink _ | Group _ | After _ | Store _ | End _ | Barrier | Custom _ -> None

let value_dtype_exn ctx node =
  match node_dtype node with
  | Some dtype -> dtype
  | None -> invalid_arg (Printf.sprintf "Kernel.%s expects a value dtype" ctx)

let index ~ptr ~idxs ?gate () =
  Index { ptr; idxs; gate; dtype = pointer_dtype_exn "index" ptr }

let ptrcat ~srcs ~dtype = Ptrcat { srcs; dtype }

let load ~src ?alt () =
  let dtype = (pointer_dtype_exn "load" src).base in
  Load { src; alt; dtype }

let store ~dst ~value ~ranges = Store { dst; value; ranges }
let unary ~op ~src = Unary { op; src; dtype = value_dtype_exn "unary" src }
let comparison_dtype (dtype : Dtype.t) = { dtype with scalar = Dtype.Bool }

let binary ~op ~lhs ~rhs =
  let lhs_dtype = value_dtype_exn "binary" lhs in
  let dtype =
    match op with
    | `Cmplt | `Cmpeq | `Cmpne -> comparison_dtype lhs_dtype
    | _ -> lhs_dtype
  in
  Binary { op; lhs; rhs; dtype }

let ternary ~op ~a ~b ~c =
  let dtype =
    match op with
    | `Where -> value_dtype_exn "ternary" b
    | `Mulacc -> value_dtype_exn "ternary" a
  in
  Ternary { op; a; b; c; dtype }

let cast ~src ~dtype = Cast { src; dtype }
let bitcast ~src ~dtype = Bitcast { src; dtype }

let vectorize ~srcs =
  match srcs with
  | [] -> invalid_arg "Kernel.vectorize expects at least one source"
  | src :: rest ->
      let scalar = Dtype.scalar_of (value_dtype_exn "vectorize" src) in
      let dtype = Dtype.vec scalar (1 + List.length rest) in
      Vectorize { srcs; dtype }

let cat ~srcs =
  match srcs with
  | [] -> invalid_arg "Kernel.cat expects at least one source"
  | srcs -> (
      match List.map (value_dtype_exn "cat") srcs with
      | [] -> assert false
      | first :: rest ->
          let count =
            List.fold_left
              (fun acc (dtype : Dtype.t) ->
                if dtype.scalar <> first.scalar then
                  invalid_arg "Kernel.cat expects a common scalar dtype";
                acc + dtype.count)
              first.count rest
          in
          Cat { srcs; dtype = { first with count } })

(* Matches tinygrad's UOp.gep() eager folding.
   VECTORIZE → extract lane.  CONST → scalar const.
   Everything else → create GEP node (no source validation). *)
let gep ~src ~idx =
  match src with
  | Vectorize { srcs; _ } when idx >= 0 && idx < List.length srcs ->
      List.nth srcs idx
  | Const { value; _ } ->
      Const { value; dtype = Dtype.scalar_of (Const.dtype value) }
  | _ -> (
      match node_dtype src with
      | Some dt -> Gep { src; idx; dtype = Dtype.scalar_of dt }
      | None -> Gep { src; idx; dtype = Dtype.void })

let range ~size ~axis ~kind ?(dtype = Dtype.index) () =
  Range { size; dtype; axis; kind }

let end_ ~value ~ranges = End { value; ranges }
let barrier = Barrier
let special ~dim ~size ?(dtype = Dtype.int32) () = Special { dim; size; dtype }
let reduce ~op ~src ~ranges ~dtype = Reduce { op; src; ranges; dtype }
let unroll ~src ~axes ~dtype = Unroll { src; axes; dtype }
let contract ~src ~axes ~dtype = Contract { src; axes; dtype }

let wmma ~name ~a ~b ~c ~dtype ~dims ~dtype_in ~dtype_out ~device ~threads
    ~upcast_axes ~reduce_axes =
  Wmma
    {
      name;
      a;
      b;
      c;
      dtype;
      dims;
      dtype_in;
      dtype_out;
      device;
      threads;
      upcast_axes;
      reduce_axes;
    }

let custom ~fmt ~args = Custom { fmt; args }
let custom_inline ~fmt ~args ~dtype = Custom_inline { fmt; args; dtype }

let gep_multi ~src ~idxs =
  match idxs with
  | [] -> invalid_arg "Kernel.gep_multi expects at least one index"
  | [ idx ] -> (
      match node_dtype src with
      | Some dt when dt.count = 1 && idx = 0 -> src
      | _ -> gep ~src ~idx)
  | _ ->
      let srcs = List.map (fun idx -> gep ~src ~idx) idxs in
      let dt =
        match node_dtype (List.hd srcs) with
        | Some dt -> dt
        | None -> Dtype.void
      in
      if dt = Dtype.void then List.hd srcs
      else Vectorize { srcs; dtype = Dtype.vec dt (List.length idxs) }

let broadcast node n =
  if n <= 1 then node
  else
    match get_ptr_dtype node with
    | Some _ -> node
    | None -> (
        match node_dtype node with
        | None -> node
        | Some dt when dt.count = 1 ->
            Vectorize { srcs = List.init n (fun _ -> node); dtype = Dtype.vec dt n }
        | Some dt ->
            Cat
              {
                srcs = List.init n (fun _ -> node);
                dtype = Dtype.vec (Dtype.scalar_of dt) (dt.count * n);
              })

let const_int n = Const { value = Const.int Dtype.index n; dtype = Dtype.index }
let const_float x = Const { value = Const.float Dtype.float32 x; dtype = Dtype.float32 }
let const_bool b = Const { value = Const.bool b; dtype = Dtype.bool }

let zero_like node =
  match node_dtype node with
  | None -> invalid_arg "Kernel.zero_like: node has no dtype"
  | Some dt ->
      let value =
        if Dtype.is_float dt then Const.float (Dtype.scalar_of dt) 0.0
        else if dt.scalar = Dtype.Bool then Const.bool false
        else Const.int (Dtype.scalar_of dt) 0
      in
      Const { value; dtype = dt }

(* Inspecting *)

let dtype = node_dtype

let sort node =
  match node with
  | Param _ | Param_image _ | Define_local _ | Define_reg _ | Bufferize _
  | Index _ | Ptrcat _ ->
      Pointer
  | Sink _ | Group _ | After _ | Store _ | End _ | Barrier | Custom _ -> Effect
  | Define_var _ | Invalid_index _ | Range _ | Special _ -> Index
  | _ -> (
      match dtype node with
      | Some dtype when dtype.scalar = Dtype.Index -> Index
      | _ -> Value)

let children = function
  | Sink { srcs; _ } | Group { srcs } -> srcs
  | After { src; deps } -> src :: deps
  | Param _ | Param_image _ | Define_local _ | Define_reg _ | Define_var _
  | Const _ | Invalid_index _ | Barrier ->
      []
  | Bufferize { src; ranges; _ } -> src :: ranges
  | Index { ptr; idxs; gate; _ } -> (ptr :: idxs) @ Option.to_list gate
  | Ptrcat { srcs; _ } -> srcs
  | Load { src; alt; _ } -> src :: Option.to_list alt
  | Store { dst; value; ranges } -> dst :: value :: ranges
  | Unary { src; _ }
  | Cast { src; _ }
  | Bitcast { src; _ }
  | Gep { src; _ }
  | Unroll { src; _ }
  | Contract { src; _ } ->
      [ src ]
  | Range { size; _ } | Special { size; _ } -> [ size ]
  | End { value; ranges } -> value :: ranges
  | Binary { lhs; rhs; _ } -> [ lhs; rhs ]
  | Ternary { a; b; c; _ } -> [ a; b; c ]
  | Vectorize { srcs; _ } | Cat { srcs; _ } -> srcs
  | Reduce { src; ranges; _ } -> src :: ranges
  | Wmma { a; b; c; _ } -> [ a; b; c ]
  | Custom { args; _ } | Custom_inline { args; _ } -> args

let replace node ?children:childs ?(dtype : Dtype.t option) () =
  let src =
    Array.of_list (match childs with Some c -> c | None -> children node)
  in
  let pos = ref 0 in
  let take () =
    let v = src.(!pos) in
    incr pos;
    v
  in
  let take_n n = List.init n (fun _ -> take ()) in
  let take_opt present = if present then Some (take ()) else None in
  let take_rest () =
    let len = Array.length src in
    let r = List.init (len - !pos) (fun j -> src.(!pos + j)) in
    pos := len;
    r
  in
  let dt old = match dtype with Some d -> d | None -> old in
  match node with
  | Sink { kernel_info; _ } -> Sink { srcs = take_rest (); kernel_info }
  | Group _ -> Group { srcs = take_rest () }
  | After _ ->
      let s = take () in
      After { src = s; deps = take_rest () }
  | Param _ | Param_image _ | Define_local _ | Define_reg _ | Define_var _
  | Const _ | Invalid_index _ | Barrier ->
      node
  | Bufferize { dtype = ptr_dt; opts; _ } ->
      let s = take () in
      Bufferize { src = s; ranges = take_rest (); dtype = ptr_dt; opts }
  | Index { idxs; gate; dtype = ptr_dt; _ } ->
      let ptr = take () in
      let idxs = take_n (List.length idxs) in
      let gate = take_opt (Option.is_some gate) in
      Index { ptr; idxs; gate; dtype = ptr_dt }
  | Ptrcat { dtype = ptr_dt; _ } -> Ptrcat { srcs = take_rest (); dtype = ptr_dt }
  | Load { alt; dtype = old_dt; _ } ->
      let s = take () in
      let alt = take_opt (Option.is_some alt) in
      Load { src = s; alt; dtype = dt old_dt }
  | Store _ ->
      let dst = take () in
      let value = take () in
      Store { dst; value; ranges = take_rest () }
  | Unary { op; dtype = old_dt; _ } -> Unary { op; src = take (); dtype = dt old_dt }
  | Binary { op; dtype = old_dt; _ } ->
      let lhs = take () in
      let rhs = take () in
      Binary { op; lhs; rhs; dtype = dt old_dt }
  | Ternary { op; dtype = old_dt; _ } ->
      let a = take () in
      let b = take () in
      let c = take () in
      Ternary { op; a; b; c; dtype = dt old_dt }
  | Cast { dtype = old_dt; _ } -> Cast { src = take (); dtype = dt old_dt }
  | Bitcast { dtype = old_dt; _ } -> Bitcast { src = take (); dtype = dt old_dt }
  | Vectorize { dtype = old_dt; _ } ->
      Vectorize { srcs = take_rest (); dtype = dt old_dt }
  | Cat { dtype = old_dt; _ } -> Cat { srcs = take_rest (); dtype = dt old_dt }
  | Gep { idx; dtype = old_dt; _ } -> Gep { src = take (); idx; dtype = dt old_dt }
  | Range { axis; kind; dtype = old_dt; _ } ->
      Range { size = take (); dtype = dt old_dt; axis; kind }
  | End _ ->
      let value = take () in
      End { value; ranges = take_rest () }
  | Special { dim; dtype = old_dt; _ } ->
      Special { dim; size = take (); dtype = dt old_dt }
  | Reduce { op; ranges; dtype = old_dt; _ } ->
      let s = take () in
      let ranges = take_n (List.length ranges) in
      Reduce { op; src = s; ranges; dtype = dt old_dt }
  | Unroll { axes; dtype = old_dt; _ } ->
      Unroll { src = take (); axes; dtype = dt old_dt }
  | Contract { axes; dtype = old_dt; _ } ->
      Contract { src = take (); axes; dtype = dt old_dt }
  | Wmma ({ dtype = old_dt; _ } as w) ->
      let a = take () in
      let b = take () in
      let c = take () in
      Wmma { w with a; b; c; dtype = dt old_dt }
  | Custom { fmt; _ } -> Custom { fmt; args = take_rest () }
  | Custom_inline { fmt; dtype = old_dt; _ } ->
      Custom_inline { fmt; args = take_rest (); dtype = dt old_dt }

let map_children map_child (instr : view) : view =
  let map_ref_list = List.map map_child in
  let map_ref_opt = Option.map map_child in
  match instr with
  | Sink { srcs; kernel_info } -> Sink { srcs = map_ref_list srcs; kernel_info }
  | Group { srcs } -> Group { srcs = map_ref_list srcs }
  | After { src; deps } ->
      After { src = map_child src; deps = map_ref_list deps }
  | Param _ | Param_image _ | Define_local _ | Define_reg _ | Define_var _
  | Const _ | Invalid_index _ | Barrier ->
      instr
  | Bufferize { src; ranges; dtype; opts } ->
      Bufferize
        { src = map_child src; ranges = map_ref_list ranges; dtype; opts }
  | Index { ptr; idxs; gate; dtype } ->
      Index
        {
          ptr = map_child ptr;
          idxs = map_ref_list idxs;
          gate = map_ref_opt gate;
          dtype;
        }
  | Ptrcat { srcs; dtype } -> Ptrcat { srcs = map_ref_list srcs; dtype }
  | Load { src; alt; dtype } ->
      Load { src = map_child src; alt = map_ref_opt alt; dtype }
  | Store { dst; value; ranges } ->
      Store
        {
          dst = map_child dst;
          value = map_child value;
          ranges = map_ref_list ranges;
        }
  | Unary { op; src; dtype } -> Unary { op; src = map_child src; dtype }
  | Binary { op; lhs; rhs; dtype } ->
      Binary { op; lhs = map_child lhs; rhs = map_child rhs; dtype }
  | Ternary { op; a; b; c; dtype } ->
      Ternary { op; a = map_child a; b = map_child b; c = map_child c; dtype }
  | Cast { src; dtype } -> Cast { src = map_child src; dtype }
  | Bitcast { src; dtype } -> Bitcast { src = map_child src; dtype }
  | Vectorize { srcs; dtype } -> Vectorize { srcs = map_ref_list srcs; dtype }
  | Cat { srcs; dtype } -> Cat { srcs = map_ref_list srcs; dtype }
  | Gep { src; idx; dtype } -> Gep { src = map_child src; idx; dtype }
  | Range { size; dtype; axis; kind } ->
      Range { size = map_child size; dtype; axis; kind }
  | End { value; ranges } ->
      End { value = map_child value; ranges = map_ref_list ranges }
  | Special { dim; size; dtype } ->
      Special { dim; size = map_child size; dtype }
  | Reduce { op; src; ranges; dtype } ->
      Reduce { op; src = map_child src; ranges = map_ref_list ranges; dtype }
  | Unroll { src; axes; dtype } -> Unroll { src = map_child src; axes; dtype }
  | Contract { src; axes; dtype } ->
      Contract { src = map_child src; axes; dtype }
  | Wmma
      {
        name;
        a;
        b;
        c;
        dtype;
        dims;
        dtype_in;
        dtype_out;
        device;
        threads;
        upcast_axes;
        reduce_axes;
      } ->
      Wmma
        {
          name;
          a = map_child a;
          b = map_child b;
          c = map_child c;
          dtype;
          dims;
          dtype_in;
          dtype_out;
          device;
          threads;
          upcast_axes;
          reduce_axes;
        }
  | Custom { fmt; args } -> Custom { fmt; args = map_ref_list args }
  | Custom_inline { fmt; args; dtype } ->
      Custom_inline { fmt; args = map_ref_list args; dtype }

(* Hash tables *)

module Tbl = Hashtbl.Make (struct
  type t = view

  let equal = ( = )
  let hash = Hashtbl.hash
end)

module Ref_tbl = (Hashtbl.Make (struct
  type t = view

  let equal = ( == )
  let hash = Hashtbl.hash
end) : Hashtbl.S with type key = t)

let toposort (root : t) : t list =
  let state = Ref_tbl.create 256 in
  let order = ref [] in
  let rec visit node =
    match Ref_tbl.find_opt state node with
    | Some 2 -> ()
    | Some 1 -> failwith "Kernel.toposort: cyclic graph"
    | Some _ -> assert false
    | None ->
        Ref_tbl.add state node 1;
        List.iter visit (children node);
        Ref_tbl.replace state node 2;
        order := node :: !order
  in
  visit root;
  List.rev !order

let intern (root : t) : t =
  let nodes = toposort root in
  let canon = Ref_tbl.create (List.length nodes) in
  let uniq = Tbl.create (List.length nodes) in
  let lookup node =
    match Ref_tbl.find_opt canon node with Some node -> node | None -> node
  in
  List.iter
    (fun original ->
      let rewritten = map_children lookup original in
      let shared =
        match Tbl.find_opt uniq rewritten with
        | Some shared -> shared
        | None ->
            Tbl.add uniq rewritten rewritten;
            rewritten
      in
      Ref_tbl.replace canon original shared)
    nodes;
  lookup root

let is_alu = function
  | Unary _ | Binary _ | Ternary _ -> true
  | _ -> false

let is_ptr node = Option.is_some (get_ptr_dtype node)
let dtype_or default node = match node_dtype node with Some dt -> dt | None -> default

let first_match rules node =
  let rec go = function
    | [] -> None
    | rule :: rest -> (
        match rule node with Some _ as r -> r | None -> go rest)
  in
  go rules

(* Validation *)

let validate (root : t) : unit =
  let nodes = toposort root in
  let ids = Ref_tbl.create (List.length nodes) in
  List.iteri (fun i node -> Ref_tbl.add ids node i) nodes;
  let id node =
    match Ref_tbl.find_opt ids node with Some i -> i | None -> -1
  in
  let fail node msg =
    failwith
      (Printf.sprintf "Kernel.validate: instruction %d: %s" (id node) msg)
  in
  let rec get_dtype node =
    match node with
    | After { src; _ } -> get_dtype src
    | End { value; _ } -> get_dtype value
    | _ -> dtype node
  in
  let check_dtype_eq node ~ctx ~expected ~got =
    match (expected, got) with
    | Some e, Some g when Dtype.equal e g -> ()
    | Some e, Some g ->
        fail node
          (Printf.sprintf "%s: expected %s, got %s" ctx (Dtype.to_string e)
             (Dtype.to_string g))
    | None, _ ->
        fail node (Printf.sprintf "%s: expected dtype not available" ctx)
    | _, None ->
        fail node (Printf.sprintf "%s: operand dtype not available" ctx)
  in
  let check_dtype_match node ~ctx dt1 dt2 =
    match (dt1, dt2) with
    | Some d1, Some d2 when Dtype.equal d1 d2 -> ()
    | Some _, Some _ ->
        fail node (Printf.sprintf "%s: operand dtypes don't match" ctx)
    | _ -> fail node (Printf.sprintf "%s: operand dtype not available" ctx)
  in
  let check_bool_scalar node ~ctx value =
    match get_dtype value with
    | Some dt when dt.scalar = Dtype.Bool && dt.count = 1 -> ()
    | Some _ -> fail node (Printf.sprintf "%s must be bool scalar" ctx)
    | None -> fail node (Printf.sprintf "%s dtype not available" ctx)
  in
  let check_shift_rhs node rhs dtype =
    match get_dtype rhs with
    | Some dt when Dtype.equal dt dtype || Dtype.equal dt Dtype.uint32 -> ()
    | Some _ -> fail node "shift rhs must match lhs dtype or be uint32"
    | None -> fail node "shift rhs dtype not available"
  in
  let check_index_like node ~ctx value =
    match get_dtype value with
    | Some dt when dt.scalar = Dtype.Index || dt.scalar = Dtype.Int32 -> ()
    | Some _ -> fail node (Printf.sprintf "%s must be index-like" ctx)
    | None -> fail node (Printf.sprintf "%s dtype not available" ctx)
  in
  let check_horizontal_reduce_src node ~src ~dtype =
    match get_dtype src with
    | Some src_dtype when Dtype.equal src_dtype dtype -> ()
    | Some src_dtype
      when src_dtype.scalar = dtype.scalar
           && src_dtype.count >= dtype.count
           && src_dtype.count mod dtype.count = 0 ->
        ()
    | Some got ->
        fail node
          (Printf.sprintf "Reduce src: expected %s or horizontal vector, got %s"
             (Dtype.to_string dtype) (Dtype.to_string got))
    | None -> fail node "Reduce src dtype not available"
  in
  let rec index_base = function
    | After { src; _ } | Cast { src; _ } | Bitcast { src; _ } -> index_base src
    | Param _ | Param_image _ | Define_local _ | Define_reg _ | Bufferize _
    | Ptrcat _ ->
        true
    | _ -> false
  in
  let rec ptr_ref = function
    | Index { dtype; gate; _ } as node -> Some (node, dtype, gate)
    | ( Ptrcat { dtype; _ }
      | Param { dtype; _ }
      | Param_image { dtype; _ }
      | Define_local { dtype; _ }
      | Define_reg { dtype; _ }
      | Bufferize { dtype; _ } ) as node ->
        Some (node, dtype, None)
    | Gep { src; dtype; _ } as node -> (
        match ptr_ref src with
        | Some (_, pty, gate) ->
            let pty = { pty with base = dtype } in
            Some (node, pty, gate)
        | None -> None)
    | After { src; _ } | Cast { src; _ } | Bitcast { src; _ } -> ptr_ref src
    | _ -> None
  in
  let prod lst = List.fold_left ( * ) 1 lst in
  List.iter
    (fun instr ->
      match instr with
      | Sink _ | Group _ | After _ -> ()
      | Param { dtype; _ } | Param_image { dtype; _ } ->
          if dtype.addrspace <> Dtype.Global then
            fail instr "Param must have Global addrspace"
      | Define_local { dtype; _ } ->
          if dtype.addrspace <> Dtype.Local then
            fail instr "Define_local must have Local addrspace"
      | Define_reg { dtype; _ } ->
          if dtype.addrspace <> Dtype.Reg then
            fail instr "Define_reg must have Reg addrspace"
      | Define_var { lo; hi; dtype; _ } ->
          if dtype.count <> 1 then fail instr "Define_var must be scalar";
          if not (Dtype.is_int dtype || dtype.scalar = Dtype.Index) then
            fail instr "Define_var must be int/index";
          if lo > hi then fail instr "Define_var bounds invalid (lo > hi)"
      | Bufferize { ranges; dtype; opts; _ } ->
          if dtype.addrspace <> opts.addrspace then
            fail instr "Bufferize dtype addrspace mismatch";
          List.iter (check_index_like instr ~ctx:"Bufferize range") ranges
      | Const { value; dtype } -> (
          match Const.view value with
          | Bool _ ->
              if dtype.scalar <> Dtype.Bool then
                fail instr "Bool const must have bool dtype"
          | Int _ ->
              if not (Dtype.is_int dtype) then
                fail instr "Int const must have int/index dtype"
          | Float _ ->
              if not (Dtype.is_float dtype) then
                fail instr "Float const must have float dtype")
      | Invalid_index { dtype } ->
          if dtype.scalar <> Dtype.Index then
            fail instr "Invalid_index must have Index dtype"
      | Range { size; dtype; _ } ->
          if not (Dtype.is_int dtype) then
            fail instr "Range must have int/index";
          if dtype.count <> 1 then fail instr "Range must be scalar";
          check_dtype_eq instr ~ctx:"Range size" ~expected:(Some dtype)
            ~got:(get_dtype size)
      | End { ranges; _ } ->
          List.iter (check_index_like instr ~ctx:"End range") ranges
      | Barrier -> ()
      | Special { size; dtype; _ } ->
          if dtype.count <> 1 then fail instr "Special must be scalar";
          if not (dtype.scalar = Dtype.Index || dtype.scalar = Dtype.Int32) then
            fail instr "Special must be index or int32";
          check_dtype_eq instr ~ctx:"Special size" ~expected:(Some dtype)
            ~got:(get_dtype size)
      | Index { ptr; idxs; gate; dtype } -> (
          if idxs = [] then fail instr "Index must have at least one index";
          if not (index_base ptr) then
            fail instr "Index base must be a buffer define/bufferize";
          List.iter (check_index_like instr ~ctx:"Index operand") idxs;
          Option.iter (check_bool_scalar instr ~ctx:"Index gate") gate;
          match get_ptr_dtype ptr with
          | Some base
            when Dtype.equal base.base dtype.base
                 && base.addrspace = dtype.addrspace
                 && base.v = dtype.v ->
              ()
          | Some _ -> fail instr "Index dtype must match base pointer type"
          | None -> fail instr "Index base dtype not available")
      | Ptrcat { srcs; dtype } ->
          if srcs = [] then fail instr "Ptrcat must have at least one source";
          let total_vcount = ref 0 in
          List.iter
            (fun src ->
              match ptr_ref src with
              | Some (_, pty, _) ->
                  if pty.addrspace <> dtype.addrspace then
                    fail instr "Ptrcat addrspace mismatch";
                  if not (Dtype.equal pty.base dtype.base) then
                    fail instr "Ptrcat base dtype mismatch";
                  total_vcount := !total_vcount + pty.base.count
              | None -> fail instr "Ptrcat sources must be pointers")
            srcs;
          if !total_vcount <> dtype.v then fail instr "Ptrcat vcount mismatch"
      | Load { src; alt; dtype } -> (
          match ptr_ref src with
          | Some (_, pty, gate) -> (
              (* Allow widened Load dtype (e.g. f32×4 from f32 pointer).
                 Intermediate state after do_expand.  Check scalar match. *)
              if dtype.scalar <> pty.base.scalar then
                check_dtype_eq instr ~ctx:"Load dtype" ~expected:(Some pty.base)
                  ~got:(Some dtype);
              match alt with
              | None -> ()
              | Some alt_ref -> (
                  check_dtype_eq instr ~ctx:"Load alt" ~expected:(Some dtype)
                    ~got:(get_dtype alt_ref);
                  match gate with
                  | None -> fail instr "Load alt requires gated Index"
                  | Some _ -> ()))
          | None -> fail instr "Load src must reference a pointer")
      | Store { dst; value; ranges } -> (
          List.iter (check_index_like instr ~ctx:"Store range") ranges;
          match ptr_ref dst with
          | Some (_, pty, _) -> (
              (* Allow widened Store value (e.g. i32×4 to i32 pointer).
                 Intermediate state after do_expand.  Check scalar match. *)
              match get_dtype value with
              | Some vdt when vdt.scalar <> pty.base.scalar ->
                  check_dtype_eq instr ~ctx:"Store value"
                    ~expected:(Some pty.base) ~got:(Some vdt)
              | _ -> ())
          | None -> fail instr "Store dst must reference a pointer")
      | Unary { src; dtype; _ } ->
          check_dtype_eq instr ~ctx:"Unary operand" ~expected:(Some dtype)
            ~got:(get_dtype src)
      | Binary { op; lhs; rhs; dtype } -> (
          let ldt = get_dtype lhs and rdt = get_dtype rhs in
          match op with
          | `Shl | `Shr ->
              check_dtype_eq instr ~ctx:"Shift lhs" ~expected:(Some dtype)
                ~got:ldt;
              check_shift_rhs instr rhs dtype;
              if not (Dtype.is_int dtype) then
                fail instr "Shift must have int/index dtype"
          | `Cmplt | `Cmpeq | `Cmpne ->
              if dtype.scalar <> Dtype.Bool then
                fail instr "Comparison must produce bool";
              check_dtype_match instr ~ctx:"Comparison operands" ldt rdt
          | `Idiv | `Mod ->
              check_dtype_match instr ~ctx:"Binary operands" ldt rdt;
              check_dtype_eq instr ~ctx:"Binary result" ~expected:(Some dtype)
                ~got:ldt;
              if not (Dtype.is_int dtype) then
                fail instr "Idiv/Mod must have int/index dtype"
          | _ ->
              check_dtype_match instr ~ctx:"Binary operands" ldt rdt;
              check_dtype_eq instr ~ctx:"Binary result" ~expected:(Some dtype)
                ~got:ldt)
      | Ternary { op; a; b; c; dtype } -> (
          match op with
          | `Where ->
              check_bool_scalar instr ~ctx:"Where condition" a;
              check_dtype_match instr ~ctx:"Where arms" (get_dtype b)
                (get_dtype c);
              check_dtype_eq instr ~ctx:"Where result" ~expected:(Some dtype)
                ~got:(get_dtype b)
          | `Mulacc ->
              check_dtype_match instr ~ctx:"Mulacc a/b" (get_dtype a)
                (get_dtype b);
              check_dtype_match instr ~ctx:"Mulacc a/c" (get_dtype a)
                (get_dtype c);
              check_dtype_eq instr ~ctx:"Mulacc result" ~expected:(Some dtype)
                ~got:(get_dtype a))
      | Vectorize { srcs; dtype } ->
          if srcs = [] then
            fail instr "Vectorize must have at least one operand";
          if dtype.count <> List.length srcs then
            fail instr "Vectorize dtype count must match operand count";
          let scalar = dtype.scalar in
          List.iter
            (fun src ->
              match get_dtype src with
              | Some dt when dt.count = 1 && dt.scalar = scalar -> ()
              | Some _ ->
                  fail instr "Vectorize operands must be scalar and match"
              | None -> fail instr "Vectorize operand dtype not available")
            srcs
      | Cat { srcs; dtype } ->
          if srcs = [] then fail instr "Cat must have at least one operand";
          let total = ref 0 in
          List.iter
            (fun src ->
              match get_dtype src with
              | Some dt ->
                  if dt.scalar <> dtype.scalar then
                    fail instr "Cat operand scalar mismatch";
                  total := !total + dt.count
              | None -> fail instr "Cat operand dtype not available")
            srcs;
          if !total <> dtype.count then fail instr "Cat count mismatch"
      | Gep { src; idx; dtype } -> (
          match get_dtype src with
          | Some dt when dt.count > 1 ->
              if idx < 0 || idx >= dt.count then
                fail instr "Gep index out of bounds";
              if dtype.count <> 1 || dtype.scalar <> dt.scalar then
                fail instr "Gep dtype must be scalar of source"
          | Some _ ->
              (* Scalar source: GEP on a non-vector node is valid.
                 This arises from do_contract on non-vector sources
                 (e.g. WMMA with scalar result dtype). *)
              ()
          | None ->
              (* Void/effect source: GEP produces void.
                 Arises from do_contract on void sources (Store, End).
                 Cleaned up by gep_pushing's gep_void rule. *)
              ())
      | Reduce { src; ranges; dtype; _ } ->
          check_horizontal_reduce_src instr ~src ~dtype;
          List.iter (check_index_like instr ~ctx:"Reduce range") ranges
      | Unroll { src; axes; dtype } ->
          if dtype.scalar <> Dtype.Void then begin
            let expected = prod (List.map snd axes) * dtype.count in
            match get_dtype src with
            | Some dt when dt.count = expected -> ()
            | Some _ -> fail instr "Unroll source count mismatch"
            | None -> fail instr "Unroll source dtype not available"
          end
      | Contract { axes; dtype; _ } ->
          if dtype.scalar <> Dtype.Void then begin
            let expected = prod (List.map snd axes) in
            if dtype.count <> expected then
              fail instr "Contract dtype count mismatch"
          end
      | Wmma _ -> ()
      | Cast _ | Bitcast _ | Custom _ | Custom_inline _ -> ())
    nodes

(* Rewriting *)

let rebuild rewrite root =
  let nodes = toposort root in
  let rebuilt = Ref_tbl.create (List.length nodes) in
  let lookup node =
    match Ref_tbl.find_opt rebuilt node with Some node -> node | None -> node
  in
  List.iter
    (fun node ->
      let mapped = map_children lookup node in
      let next =
        match rewrite mapped with Some rewritten -> rewritten | None -> mapped
      in
      Ref_tbl.replace rebuilt node (intern next))
    nodes;
  lookup root

let rewrite_fixpoint ?(max_iters = 16) rewrite root =
  let rec loop iter node =
    if iter >= max_iters then
      Printf.ksprintf failwith
        "Kernel.rewrite_fixpoint: fixpoint not reached after %d passes"
        max_iters
    else
      let node' = rebuild rewrite node in
      if node' = node then node else loop (iter + 1) node'
  in
  loop 0 root

(* Formatting *)

let pp_comma fmt () = Format.fprintf fmt ", "

let pp_ptr fmt (dtype : Dtype.ptr) =
  Format.fprintf fmt "%a*%s [%a]" Dtype.pp dtype.base
    (if dtype.v = 1 then "" else Printf.sprintf ".vec(%d)" dtype.v)
    Dtype.pp_addr_space dtype.addrspace

let pp_axes fmt axes =
  Format.pp_print_list ~pp_sep:pp_comma
    (fun fmt (a, s) -> Format.fprintf fmt "(%d, %d)" a s)
    fmt axes

let pp_view_with ids fmt instr =
  let pp_ref fmt node = Format.fprintf fmt "%%%d" (Tbl.find ids node) in
  let pp_refs fmt refs =
    Format.pp_print_list ~pp_sep:pp_comma pp_ref fmt refs
  in
  match instr with
  | Sink { srcs; kernel_info = _ } -> Format.fprintf fmt "sink %a" pp_refs srcs
  | Group { srcs } -> Format.fprintf fmt "group %a" pp_refs srcs
  | After { src; deps } ->
      Format.fprintf fmt "after %a, deps=[%a]" pp_ref src pp_refs deps
  | Param { idx; dtype } -> Format.fprintf fmt "param %d : %a" idx pp_ptr dtype
  | Param_image { idx; dtype; width; height } ->
      Format.fprintf fmt "param_image %d : %a [%dx%d]" idx pp_ptr dtype width
        height
  | Define_local { size; dtype } ->
      Format.fprintf fmt "define_local %a, size=%d" pp_ptr dtype size
  | Define_reg { size; dtype } ->
      Format.fprintf fmt "define_reg %a, size=%d" pp_ptr dtype size
  | Define_var { name; lo; hi; dtype } ->
      Format.fprintf fmt "define_var %s : %a [%d..%d]" name Dtype.pp dtype lo hi
  | Bufferize { src; ranges; dtype; _ } ->
      Format.fprintf fmt "bufferize %a, ranges=[%a] : %a" pp_ref src pp_refs
        ranges pp_ptr dtype
  | Const { value; dtype } ->
      Format.fprintf fmt "const %a : %a" Const.pp value Dtype.pp dtype
  | Invalid_index { dtype } ->
      Format.fprintf fmt "invalid_index : %a" Dtype.pp dtype
  | Index { ptr; idxs; gate; dtype } ->
      Format.fprintf fmt "index %a, %a%a : %a" pp_ref ptr pp_refs idxs
        (fun fmt -> function
          | None -> () | Some gate -> Format.fprintf fmt " gate=%a" pp_ref gate)
        gate pp_ptr dtype
  | Ptrcat { srcs; dtype } ->
      Format.fprintf fmt "ptrcat %a : %a" pp_refs srcs pp_ptr dtype
  | Load { src; alt; dtype } ->
      Format.fprintf fmt "load %a%a : %a" pp_ref src
        (fun fmt -> function
          | None -> () | Some alt -> Format.fprintf fmt " alt=%a" pp_ref alt)
        alt Dtype.pp dtype
  | Store { dst; value; ranges } ->
      Format.fprintf fmt "store %a, %a, ranges=[%a]" pp_ref dst pp_ref value
        pp_refs ranges
  | Unary { op; src; dtype } ->
      Format.fprintf fmt "%a %a : %a" Op.pp_unary op pp_ref src Dtype.pp dtype
  | Cast { src; dtype } ->
      Format.fprintf fmt "cast %a : %a" pp_ref src Dtype.pp dtype
  | Bitcast { src; dtype } ->
      Format.fprintf fmt "bitcast %a : %a" pp_ref src Dtype.pp dtype
  | Binary { op; lhs; rhs; dtype } ->
      Format.fprintf fmt "%a %a, %a : %a" Op.pp_binary op pp_ref lhs pp_ref rhs
        Dtype.pp dtype
  | Ternary { op; a; b; c; dtype } ->
      Format.fprintf fmt "%a %a, %a, %a : %a" Op.pp_ternary op pp_ref a pp_ref b
        pp_ref c Dtype.pp dtype
  | Vectorize { srcs; dtype } ->
      Format.fprintf fmt "vec %a : %a" pp_refs srcs Dtype.pp dtype
  | Cat { srcs; dtype } ->
      Format.fprintf fmt "cat %a : %a" pp_refs srcs Dtype.pp dtype
  | Gep { src; idx; dtype } ->
      Format.fprintf fmt "gep %a, %d : %a" pp_ref src idx Dtype.pp dtype
  | Range { size; dtype; axis; kind } ->
      Format.fprintf fmt "range %a : %a [axis=%d, %a]" pp_ref size Dtype.pp
        dtype axis Axis_kind.pp kind
  | End { value; ranges } ->
      Format.fprintf fmt "end %a, ranges=[%a]" pp_ref value pp_refs ranges
  | Barrier -> Format.fprintf fmt "barrier"
  | Special { dim; size; dtype } ->
      Format.fprintf fmt "special %a, %a : %a" Special_dim.pp dim pp_ref size
        Dtype.pp dtype
  | Reduce { op; src; ranges; dtype } ->
      Format.fprintf fmt "reduce.%a %a, ranges=[%a] : %a" Op.pp_reduce op pp_ref
        src pp_refs ranges Dtype.pp dtype
  | Unroll { src; axes; dtype } ->
      Format.fprintf fmt "unroll %a, axes=[%a] : %a" pp_ref src pp_axes axes
        Dtype.pp dtype
  | Contract { src; axes; dtype } ->
      Format.fprintf fmt "contract %a, axes=[%a] : %a" pp_ref src pp_axes axes
        Dtype.pp dtype
  | Wmma
      {
        name;
        a;
        b;
        c;
        dtype;
        dims = n, m, k;
        dtype_in;
        dtype_out;
        device;
        threads;
        _;
      } ->
      Format.fprintf fmt
        "wmma.%s %a, %a, %a : %a [%dx%dx%d, %a -> %a, %s, threads=%d]" name
        pp_ref a pp_ref b pp_ref c Dtype.pp dtype n m k Dtype.pp_scalar dtype_in
        Dtype.pp_scalar dtype_out device threads
  | Custom { fmt = f; args } ->
      Format.fprintf fmt "custom \"%s\" %a" f pp_refs args
  | Custom_inline { fmt = f; args; dtype } ->
      Format.fprintf fmt "custom_inline \"%s\" %a : %a" f pp_refs args Dtype.pp
        dtype

let assign_ids root =
  let nodes = toposort root in
  let ids = Tbl.create (List.length nodes) in
  List.iteri (fun i node -> Tbl.add ids node i) nodes;
  (ids, nodes)

let pp_view fmt instr =
  let ids, _ = assign_ids instr in
  pp_view_with ids fmt instr

let pp fmt root =
  let ids, nodes = assign_ids root in
  List.iteri
    (fun i node -> Format.fprintf fmt "%3d: %a@\n" i (pp_view_with ids) node)
    nodes
