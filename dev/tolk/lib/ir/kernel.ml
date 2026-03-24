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
}

type tagged_view = { view : view; tag : string option }

and t = tagged_view Hashcons.hash_consed

and view =
  | Sink of { srcs : t list; kernel_info : kernel_info option }
  | Group of { srcs : t list }
  | After of { src : t; deps : t list }
  | Param of { idx : int; dtype : Dtype.ptr }
  | Param_image of { idx : int; dtype : Dtype.ptr; width : int; height : int }
  | Define_local of { size : int; dtype : Dtype.ptr }
  | Define_reg of { size : int; dtype : Dtype.ptr; slot : int }
  | Define_var of { name : string; lo : int; hi : int; dtype : Dtype.t }
  | Bufferize of {
      src : t;
      ranges : t list;
      dtype : Dtype.ptr;
      opts : bufferize_opts;
    }
  | Const of { value : Const.t; dtype : Dtype.t }
  | Invalid_index of { dtype : Dtype.t }
  | Index of { ptr : t; idxs : t list; gate : t option; dtype : Dtype.any }
  | Ptrcat of { srcs : t list; dtype : Dtype.ptr }
  | Load of { src : t; alt : t option; dtype : Dtype.t }
  | Store of { dst : t; value : t; ranges : t list }
  | Unary of { op : Op.unary; src : t; dtype : Dtype.t }
  | Binary of { op : Op.binary; lhs : t; rhs : t; dtype : Dtype.t }
  | Ternary of { op : Op.ternary; a : t; b : t; c : t; dtype : Dtype.t }
  | Cast of { src : t; dtype : Dtype.any }
  | Bitcast of { src : t; dtype : Dtype.t }
  | Vectorize of { srcs : t list; dtype : Dtype.any }
  | Cat of { srcs : t list; dtype : Dtype.t }
  | Gep of { src : t; idxs : int list; dtype : Dtype.t }
  | Range of { size : t; dtype : Dtype.t; axis : int; sub : int list; kind : Axis_kind.t }
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

let view node = node.Hashcons.node.view
let tag node = node.Hashcons.node.tag

(* Shallow hash of a view: hash the variant tag, children by their unique
   hashcons tag (physical identity), and non-child fields by generic hash.
   Children are already interned, so their tag is stable. *)
let shallow_hash_view v =
  let h = ref (Hashtbl.hash (Obj.tag (Obj.repr v))) in
  let add_id node = h := !h * 31 + node.Hashcons.tag in
  let add x = h := !h * 31 + Hashtbl.hash x in
  (match v with
   | Sink { srcs; kernel_info } -> List.iter add_id srcs; add kernel_info
   | Group { srcs } -> List.iter add_id srcs
   | After { src; deps } -> add_id src; List.iter add_id deps
   | Param { idx; dtype } -> add idx; add dtype
   | Param_image { idx; dtype; width; height } -> add idx; add dtype; add width; add height
   | Define_local { size; dtype } -> add size; add dtype
   | Define_reg { size; dtype; slot } -> add size; add dtype; add slot
   | Define_var { name; lo; hi; dtype } -> add name; add lo; add hi; add dtype
   | Bufferize { src; ranges; dtype; opts } -> add_id src; List.iter add_id ranges; add dtype; add opts
   | Const { value; dtype } ->
       (match Const.view value with
        | Bool b -> add b | Int i -> add i
        | Float f -> add (Int64.bits_of_float f));
       add dtype
   | Invalid_index { dtype } -> add dtype
   | Index { ptr; idxs; gate; dtype } -> add_id ptr; List.iter add_id idxs; (match gate with Some g -> add_id g | None -> ()); add dtype
   | Ptrcat { srcs; dtype } -> List.iter add_id srcs; add dtype
   | Load { src; alt; dtype } -> add_id src; (match alt with Some a -> add_id a | None -> ()); add dtype
   | Store { dst; value; ranges } -> add_id dst; add_id value; List.iter add_id ranges
   | Unary { op; src; dtype } -> add op; add_id src; add dtype
   | Binary { op; lhs; rhs; dtype } -> add op; add_id lhs; add_id rhs; add dtype
   | Ternary { op; a; b; c; dtype } -> add op; add_id a; add_id b; add_id c; add dtype
   | Cast { src; dtype } -> add_id src; add dtype
   | Bitcast { src; dtype } -> add_id src; add dtype
   | Vectorize { srcs; dtype } -> List.iter add_id srcs; add dtype
   | Cat { srcs; dtype } -> List.iter add_id srcs; add dtype
   | Gep { src; idxs; dtype } -> add_id src; add idxs; add dtype
   | Range { size; dtype; axis; sub; kind } -> add_id size; add dtype; add axis; add sub; add kind
   | End { value; ranges } -> add_id value; List.iter add_id ranges
   | Barrier -> ()
   | Special { dim; size; dtype } -> add dim; add_id size; add dtype
   | Reduce { op; src; ranges; dtype } -> add op; add_id src; List.iter add_id ranges; add dtype
   | Unroll { src; axes; dtype } -> add_id src; add axes; add dtype
   | Contract { src; axes; dtype } -> add_id src; add axes; add dtype
   | Wmma { name; a; b; c; dtype; dims; dtype_in; dtype_out; device; threads; upcast_axes; reduce_axes } ->
       add name; add_id a; add_id b; add_id c; add dtype; add dims; add dtype_in; add dtype_out; add device; add threads; add upcast_axes; add reduce_axes
   | Custom { fmt; args } -> add fmt; List.iter add_id args
   | Custom_inline { fmt; args; dtype } -> add fmt; List.iter add_id args; add dtype);
  !h

(* Shallow equality: same variant, children physically equal, non-child
   fields structurally equal. *)
let shallow_equal_view v1 v2 =
  match v1, v2 with
  | Sink { srcs = s1; kernel_info = k1 }, Sink { srcs = s2; kernel_info = k2 } ->
      List.length s1 = List.length s2 && List.for_all2 (==) s1 s2 && k1 = k2
  | Group { srcs = s1 }, Group { srcs = s2 } ->
      List.length s1 = List.length s2 && List.for_all2 (==) s1 s2
  | After { src = s1; deps = d1 }, After { src = s2; deps = d2 } ->
      s1 == s2 && List.length d1 = List.length d2 && List.for_all2 (==) d1 d2
  | Param r1, Param r2 -> r1.idx = r2.idx && r1.dtype = r2.dtype
  | Param_image r1, Param_image r2 ->
      r1.idx = r2.idx && r1.dtype = r2.dtype && r1.width = r2.width && r1.height = r2.height
  | Define_local r1, Define_local r2 -> r1.size = r2.size && r1.dtype = r2.dtype
  | Define_reg r1, Define_reg r2 -> r1.size = r2.size && r1.dtype = r2.dtype && r1.slot = r2.slot
  | Define_var r1, Define_var r2 ->
      r1.name = r2.name && r1.lo = r2.lo && r1.hi = r2.hi && r1.dtype = r2.dtype
  | Bufferize { src = s1; ranges = r1; dtype = d1; opts = o1 },
    Bufferize { src = s2; ranges = r2; dtype = d2; opts = o2 } ->
      s1 == s2 && List.length r1 = List.length r2 && List.for_all2 (==) r1 r2 && d1 = d2 && o1 = o2
  | Const r1, Const r2 -> Const.equal r1.value r2.value && r1.dtype = r2.dtype
  | Invalid_index r1, Invalid_index r2 -> r1.dtype = r2.dtype
  | Index { ptr = p1; idxs = i1; gate = g1; dtype = d1 },
    Index { ptr = p2; idxs = i2; gate = g2; dtype = d2 } ->
      p1 == p2 && List.length i1 = List.length i2 && List.for_all2 (==) i1 i2
      && (match g1, g2 with None, None -> true | Some a, Some b -> a == b | _ -> false) && d1 = d2
  | Ptrcat { srcs = s1; dtype = d1 }, Ptrcat { srcs = s2; dtype = d2 } ->
      List.length s1 = List.length s2 && List.for_all2 (==) s1 s2 && d1 = d2
  | Load { src = s1; alt = a1; dtype = d1 }, Load { src = s2; alt = a2; dtype = d2 } ->
      s1 == s2 && (match a1, a2 with None, None -> true | Some x, Some y -> x == y | _ -> false) && d1 = d2
  | Store { dst = d1; value = v1; ranges = r1 }, Store { dst = d2; value = v2; ranges = r2 } ->
      d1 == d2 && v1 == v2 && List.length r1 = List.length r2 && List.for_all2 (==) r1 r2
  | Unary { op = o1; src = s1; dtype = d1 }, Unary { op = o2; src = s2; dtype = d2 } ->
      o1 = o2 && s1 == s2 && d1 = d2
  | Binary { op = o1; lhs = l1; rhs = r1; dtype = d1 }, Binary { op = o2; lhs = l2; rhs = r2; dtype = d2 } ->
      o1 = o2 && l1 == l2 && r1 == r2 && d1 = d2
  | Ternary { op = o1; a = a1; b = b1; c = c1; dtype = d1 },
    Ternary { op = o2; a = a2; b = b2; c = c2; dtype = d2 } ->
      o1 = o2 && a1 == a2 && b1 == b2 && c1 == c2 && d1 = d2
  | Cast { src = s1; dtype = d1 }, Cast { src = s2; dtype = d2 } -> s1 == s2 && d1 = d2
  | Bitcast { src = s1; dtype = d1 }, Bitcast { src = s2; dtype = d2 } -> s1 == s2 && d1 = d2
  | Vectorize { srcs = s1; dtype = d1 }, Vectorize { srcs = s2; dtype = d2 } ->
      List.length s1 = List.length s2 && List.for_all2 (==) s1 s2 && d1 = d2
  | Cat { srcs = s1; dtype = d1 }, Cat { srcs = s2; dtype = d2 } ->
      List.length s1 = List.length s2 && List.for_all2 (==) s1 s2 && d1 = d2
  | Gep { src = s1; idxs = i1; dtype = d1 }, Gep { src = s2; idxs = i2; dtype = d2 } ->
      s1 == s2 && i1 = i2 && d1 = d2
  | Range r1, Range r2 ->
      r1.size == r2.size && r1.dtype = r2.dtype && r1.axis = r2.axis && r1.sub = r2.sub && r1.kind = r2.kind
  | End { value = v1; ranges = r1 }, End { value = v2; ranges = r2 } ->
      v1 == v2 && List.length r1 = List.length r2 && List.for_all2 (==) r1 r2
  | Barrier, Barrier -> true
  | Special { dim = d1; size = s1; dtype = t1 }, Special { dim = d2; size = s2; dtype = t2 } ->
      d1 = d2 && s1 == s2 && t1 = t2
  | Reduce { op = o1; src = s1; ranges = r1; dtype = d1 }, Reduce { op = o2; src = s2; ranges = r2; dtype = d2 } ->
      o1 = o2 && s1 == s2 && List.length r1 = List.length r2 && List.for_all2 (==) r1 r2 && d1 = d2
  | Unroll { src = s1; axes = a1; dtype = d1 }, Unroll { src = s2; axes = a2; dtype = d2 } ->
      s1 == s2 && a1 = a2 && d1 = d2
  | Contract { src = s1; axes = a1; dtype = d1 }, Contract { src = s2; axes = a2; dtype = d2 } ->
      s1 == s2 && a1 = a2 && d1 = d2
  | Wmma w1, Wmma w2 ->
      w1.name = w2.name && w1.a == w2.a && w1.b == w2.b && w1.c == w2.c
      && w1.dtype = w2.dtype && w1.dims = w2.dims && w1.dtype_in = w2.dtype_in
      && w1.dtype_out = w2.dtype_out && w1.device = w2.device && w1.threads = w2.threads
      && w1.upcast_axes = w2.upcast_axes && w1.reduce_axes = w2.reduce_axes
  | Custom { fmt = f1; args = a1 }, Custom { fmt = f2; args = a2 } ->
      f1 = f2 && List.length a1 = List.length a2 && List.for_all2 (==) a1 a2
  | Custom_inline { fmt = f1; args = a1; dtype = d1 }, Custom_inline { fmt = f2; args = a2; dtype = d2 } ->
      f1 = f2 && List.length a1 = List.length a2 && List.for_all2 (==) a1 a2 && d1 = d2
  | _ -> false

let shallow_hash_tagged tv =
  shallow_hash_view tv.view * 31 + Hashtbl.hash tv.tag

let shallow_equal_tagged tv1 tv2 =
  tv1.tag = tv2.tag && shallow_equal_view tv1.view tv2.view

module View_hc = Hashcons.Make (struct
  type t = tagged_view
  let equal = shallow_equal_tagged
  let hash = shallow_hash_tagged
end)

let hc_table = View_hc.create 4096
let mk ?tag v = View_hc.hashcons hc_table { view = v; tag }
let with_tag t node = mk ~tag:t (view node)
let sink ?kernel_info srcs = mk (Sink { srcs; kernel_info })
let group srcs = match srcs with [x] -> x | _ -> mk (Group { srcs })
let after ~src ~deps = match deps with [] -> src | _ -> mk (After { src; deps })
let param ~idx ~dtype = mk (Param { idx; dtype })

let param_image ~idx ~dtype ~width ~height =
  mk (Param_image { idx; dtype; width; height })

let define_local ~size ~dtype = mk (Define_local { size; dtype })
let define_reg ~size ~dtype ~slot = mk (Define_reg { size; dtype; slot })

let define_var ~name ~lo ~hi ?(dtype = Dtype.index) () =
  mk (Define_var { name; lo; hi; dtype })

let bufferize ~src ~ranges ~dtype ~opts = mk (Bufferize { src; ranges; dtype; opts })
let const value = mk (Const { value; dtype = Const.dtype value })

let invalid_index ?(lanes = 1) () =
  mk (Invalid_index { dtype = Dtype.vec Dtype.index lanes })

let rec get_ptr_dtype node = match node.Hashcons.node.view with
  | Param { dtype; _ }
  | Param_image { dtype; _ }
  | Define_local { dtype; _ }
  | Define_reg { dtype; _ }
  | Bufferize { dtype; _ }
  | Ptrcat { dtype; _ } ->
      Some dtype
  | Index { dtype = Dtype.P p; _ } -> Some p
  | Index { dtype = Dtype.T _; _ } -> None
  | Vectorize { dtype = Dtype.P p; _ } -> Some p
  | Vectorize { dtype = Dtype.T _; _ } -> None
  | Cast { dtype = Dtype.P p; _ } -> Some p
  | After { src; _ } | Cast { src; _ } | Bitcast { src; _ } -> get_ptr_dtype src
  | _ -> None

let pointer_dtype_exn ctx node =
  match get_ptr_dtype node with
  | Some dtype -> dtype
  | None -> Printf.ksprintf invalid_arg "Kernel.%s expects a pointer node" ctx

let rec node_dtype node = match node.Hashcons.node.view with
  | Param { dtype; _ } | Param_image { dtype; _ } | Define_local { dtype; _ }
  | Define_reg { dtype; _ } | Bufferize { dtype; _ }
  | Ptrcat { dtype; _ } ->
      Some (Dtype.base dtype)
  | Index { dtype; _ } | Cast { dtype; _ } | Vectorize { dtype; _ } ->
      Some (Dtype.any_to_val dtype)
  | Define_var { dtype; _ } | Const { dtype; _ } | Invalid_index { dtype; _ }
  | Load { dtype; _ } | Unary { dtype; _ } | Binary { dtype; _ }
  | Ternary { dtype; _ } | Bitcast { dtype; _ }
  | Cat { dtype; _ } | Gep { dtype; _ }
  | Range { dtype; _ } | Special { dtype; _ } | Reduce { dtype; _ }
  | Unroll { dtype; _ } | Contract { dtype; _ } | Wmma { dtype; _ }
  | Custom_inline { dtype; _ } ->
      Some dtype
  | Sink _ | Group _ | After _ | Store _ | End _ | Barrier | Custom _ -> None

let value_dtype_exn ctx node =
  match node_dtype node with
  | Some dtype -> dtype
  | None -> Printf.ksprintf invalid_arg "Kernel.%s expects a value dtype" ctx

let index ~ptr ~idxs ?gate ?(as_ptr = true) () =
  let pty = pointer_dtype_exn "index" ptr in
  let dtype =
    if as_ptr then Dtype.ptr_to_any pty else Dtype.to_any (Dtype.base pty)
  in
  mk (Index { ptr; idxs; gate; dtype })

let index_raw ~ptr ~idxs ?gate ~dtype () =
  mk (Index { ptr; idxs; gate; dtype })

let ptrcat ~srcs ~dtype = mk (Ptrcat { srcs; dtype })

let load ~src ?alt () =
  let dtype = Dtype.base (pointer_dtype_exn "load" src) in
  mk (Load { src; alt; dtype })

let store ~dst ~value ~ranges = mk (Store { dst; value; ranges })
let unary ~op ~src = mk (Unary { op; src; dtype = value_dtype_exn "unary" src })

let binary ~op ~lhs ~rhs =
  let lhs_dtype = value_dtype_exn "binary" lhs in
  let dtype = match op with
    | `Cmplt | `Cmpeq | `Cmpne ->
        Dtype.vec Dtype.bool (Dtype.count lhs_dtype)
    | _ -> lhs_dtype
  in
  mk (Binary { op; lhs; rhs; dtype })

let ternary ~op ~a ~b ~c =
  let dtype = match op with
    | `Where -> value_dtype_exn "ternary" b
    | `Mulacc -> value_dtype_exn "ternary" a
  in
  mk (Ternary { op; a; b; c; dtype })

let cast ~src ~dtype = mk (Cast { src; dtype })
let bitcast ~src ~dtype = mk (Bitcast { src; dtype })

let vectorize ~srcs =
  match srcs with
  | [] -> invalid_arg "Kernel.vectorize expects at least one source"
  | src :: rest ->
      let count = 1 + List.length rest in
      let dtype : Dtype.any =
        match get_ptr_dtype src with
        | Some pty -> Dtype.ptr_to_any (Dtype.ptr_with_v pty count)
        | None ->
            let scalar = Dtype.scalar_of (value_dtype_exn "vectorize" src) in
            Dtype.to_any (Dtype.vec scalar count)
      in
      mk (Vectorize { srcs; dtype })

let cat ~srcs =
  match srcs with
  | [] -> invalid_arg "Kernel.cat expects at least one source"
  | _ ->
      let dtypes = List.map (value_dtype_exn "cat") srcs in
      let first = List.hd dtypes in
      let total_count =
        List.fold_left
          (fun acc dtype ->
            if Dtype.scalar dtype <> Dtype.scalar first then
              invalid_arg "Kernel.cat expects a common scalar dtype";
            acc + Dtype.count dtype)
          0 dtypes
      in
      mk (Cat { srcs; dtype = Dtype.vec (Dtype.scalar_of first) total_count })

(* Eager GEP folding.
   VECTORIZE → extract lane.  CONST → scalar const.
   Everything else → create GEP node (no source validation). *)
let gep ~src ~idx =
  match src.Hashcons.node.view with
  | Vectorize { srcs; _ } when idx >= 0 && idx < List.length srcs ->
      List.nth srcs idx
  | Const { value; _ } ->
      mk (Const { value; dtype = Dtype.scalar_of (Const.dtype value) })
  | _ -> (
      match node_dtype src with
      | Some dt -> mk (Gep { src; idxs = [idx]; dtype = Dtype.scalar_of dt })
      | None -> mk (Gep { src; idxs = [idx]; dtype = Dtype.void }))

let range ~size ~axis ?(sub = []) ~kind ?(dtype = Dtype.index) () =
  mk (Range { size; dtype; axis; sub; kind })

let end_ ~value ~ranges ?tag () =
  mk ?tag (End { value; ranges })
let barrier = mk Barrier
let special ~dim ~size ?(dtype = Dtype.int32) () = mk (Special { dim; size; dtype })
let reduce ~op ~src ~ranges ~dtype = mk (Reduce { op; src; ranges; dtype })
let unroll ~src ~axes ~dtype = mk (Unroll { src; axes; dtype })
let contract ~src ~axes ~dtype = mk (Contract { src; axes; dtype })

let wmma ~name ~a ~b ~c ~dtype ~dims ~dtype_in ~dtype_out ~device ~threads
    ~upcast_axes ~reduce_axes =
  mk (Wmma {
      name; a; b; c; dtype; dims; dtype_in; dtype_out;
      device; threads; upcast_axes; reduce_axes })

let custom ~fmt ~args = mk (Custom { fmt; args })
let custom_inline ~fmt ~args ~dtype = mk (Custom_inline { fmt; args; dtype })

let gep_multi ~src ~idxs =
  match idxs with
  | [] -> invalid_arg "Kernel.gep_multi expects at least one index"
  | [ idx ] ->
      let is_scalar_zero =
        match node_dtype src with Some dt -> Dtype.count dt = 1 && idx = 0 | None -> false
      in
      if is_scalar_zero then src else gep ~src ~idx
  | _ ->
      let dt = match node_dtype src with Some dt -> dt | None -> Dtype.void in
      if dt = Dtype.void then src
      else
        let scalar = Dtype.scalar_of dt in
        mk (Gep { src; idxs; dtype = Dtype.vec scalar (List.length idxs) })

let broadcast node n =
  if n <= 1 then node
  else match get_ptr_dtype node with
    | Some pty ->
        let copies = List.init n (fun _ -> node) in
        mk (Vectorize { srcs = copies; dtype = Dtype.ptr_to_any (Dtype.ptr_with_v pty n) })
    | None ->
        match node_dtype node with
        | None -> node
        | Some dt ->
            let copies = List.init n (fun _ -> node) in
            if Dtype.count dt = 1 then mk (Vectorize { srcs = copies; dtype = Dtype.to_any (Dtype.vec dt n) })
            else mk (Cat { srcs = copies; dtype = Dtype.vec (Dtype.scalar_of dt) (Dtype.count dt * n) })

let const_int n = mk (Const { value = Const.int Dtype.index n; dtype = Dtype.index })
let const_float x = mk (Const { value = Const.float Dtype.float32 x; dtype = Dtype.float32 })
let const_bool b = mk (Const { value = Const.bool b; dtype = Dtype.bool })

let zero_like node =
  match node_dtype node with
  | None -> invalid_arg "Kernel.zero_like: node has no dtype"
  | Some dt ->
      let value =
        if Dtype.is_float dt then Const.float (Dtype.scalar_of dt) 0.0
        else if Dtype.scalar dt = Dtype.Bool then Const.bool false
        else Const.int (Dtype.scalar_of dt) 0
      in
      mk (Const { value; dtype = dt })

(* Inspecting *)

let dtype = node_dtype

let sort node =
  match node.Hashcons.node.view with
  | Param _ | Param_image _ | Define_local _ | Define_reg _ | Bufferize _
  | Index { dtype = Dtype.P _; _ } | Vectorize { dtype = Dtype.P _; _ }
  | Cast { dtype = Dtype.P _; _ } | Ptrcat _ ->
      Pointer
  | Sink _ | Group _ | After _ | Store _ | End _ | Barrier | Custom _ -> Effect
  | Define_var _ | Invalid_index _ | Range _ | Special _ -> Index
  | _ -> (
      match dtype node with
      | Some dtype when Dtype.scalar dtype = Dtype.Index -> Index
      | _ -> Value)

let children node = match node.Hashcons.node.view with
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
      [src]
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
  let new_view =
    match node.Hashcons.node.view with
    | Sink { kernel_info; _ } -> Sink { srcs = take_rest (); kernel_info }
    | Group _ -> Group { srcs = take_rest () }
    | After _ ->
        let s = take () in
        After { src = s; deps = take_rest () }
    | Param _ | Param_image _ | Define_local _ | Define_reg _ | Barrier ->
        node.Hashcons.node.view
    | Define_var { name; lo; hi; dtype = old_dt } ->
        Define_var { name; lo; hi; dtype = dt old_dt }
    | Const { value; dtype = old_dt } ->
        Const { value; dtype = dt old_dt }
    | Invalid_index { dtype = old_dt } ->
        Invalid_index { dtype = dt old_dt }
    | Bufferize { dtype = ptr_dt; opts; _ } ->
        let s = take () in
        Bufferize { src = s; ranges = take_rest (); dtype = ptr_dt; opts }
    | Index { idxs; gate; dtype = any_dt; _ } ->
        let ptr = take () in
        let idxs = take_n (List.length idxs) in
        let gate = take_opt (Option.is_some gate) in
        let any_dt = match dtype with
          | Some d ->
              (* Preserve the ptr/value distinction: if the Index was ptr-typed,
                 update the base of the ptr dtype; if value-typed, update the
                 value dtype. *)
              (match any_dt with
               | Dtype.P p -> Dtype.ptr_to_any (Dtype.ptr_with_base p d)
               | Dtype.T _ -> Dtype.to_any d)
          | None -> any_dt
        in
        Index { ptr; idxs; gate; dtype = any_dt }
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
    | Cast { dtype = any_dt; _ } ->
        let any_dt = match dtype with
          | Some d -> (match any_dt with Dtype.P _ -> any_dt | Dtype.T _ -> Dtype.to_any d)
          | None -> any_dt
        in
        Cast { src = take (); dtype = any_dt }
    | Bitcast { dtype = old_dt; _ } -> Bitcast { src = take (); dtype = dt old_dt }
    | Vectorize { dtype = any_dt; _ } ->
        let any_dt = match dtype with
          | Some d -> (match any_dt with Dtype.P _ -> any_dt | Dtype.T _ -> Dtype.to_any d)
          | None -> any_dt
        in
        Vectorize { srcs = take_rest (); dtype = any_dt }
    | Cat { dtype = old_dt; _ } -> Cat { srcs = take_rest (); dtype = dt old_dt }
    | Gep { idxs; dtype = old_dt; _ } -> Gep { src = take (); idxs; dtype = dt old_dt }
    | Range { axis; sub; kind; dtype = old_dt; _ } ->
        Range { size = take (); dtype = dt old_dt; axis; sub; kind }
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
  in
  mk ?tag:(tag node) new_view

let map_children f (instr : view) : view =
  let fl = List.map f and fo = Option.map f in
  match instr with
  | Sink { srcs; kernel_info } -> Sink { srcs = fl srcs; kernel_info }
  | Group { srcs } -> Group { srcs = fl srcs }
  | After { src; deps } -> After { src = f src; deps = fl deps }
  | Param _ | Param_image _ | Define_local _ | Define_reg _ | Define_var _
  | Const _ | Invalid_index _ | Barrier ->
      instr
  | Bufferize { src; ranges; dtype; opts } ->
      Bufferize { src = f src; ranges = fl ranges; dtype; opts }
  | Index { ptr; idxs; gate; dtype } ->
      Index { ptr = f ptr; idxs = fl idxs; gate = fo gate; dtype }
  | Ptrcat { srcs; dtype } -> Ptrcat { srcs = fl srcs; dtype }
  | Load { src; alt; dtype } -> Load { src = f src; alt = fo alt; dtype }
  | Store { dst; value; ranges } ->
      Store { dst = f dst; value = f value; ranges = fl ranges }
  | Unary { op; src; dtype } -> Unary { op; src = f src; dtype }
  | Binary { op; lhs; rhs; dtype } ->
      Binary { op; lhs = f lhs; rhs = f rhs; dtype }
  | Ternary { op; a; b; c; dtype } ->
      Ternary { op; a = f a; b = f b; c = f c; dtype }
  | Cast { src; dtype } -> Cast { src = f src; dtype }
  | Bitcast { src; dtype } -> Bitcast { src = f src; dtype }
  | Vectorize { srcs; dtype } -> Vectorize { srcs = fl srcs; dtype }
  | Cat { srcs; dtype } -> Cat { srcs = fl srcs; dtype }
  | Gep { src; idxs; dtype } -> Gep { src = f src; idxs; dtype }
  | Range { size; dtype; axis; sub; kind } ->
      Range { size = f size; dtype; axis; sub; kind }
  | End { value; ranges } -> End { value = f value; ranges = fl ranges }
  | Special { dim; size; dtype } -> Special { dim; size = f size; dtype }
  | Reduce { op; src; ranges; dtype } ->
      Reduce { op; src = f src; ranges = fl ranges; dtype }
  | Unroll { src; axes; dtype } -> Unroll { src = f src; axes; dtype }
  | Contract { src; axes; dtype } -> Contract { src = f src; axes; dtype }
  | Wmma w -> Wmma { w with a = f w.a; b = f w.b; c = f w.c }
  | Custom { fmt; args } -> Custom { fmt; args = fl args }
  | Custom_inline { fmt; args; dtype } ->
      Custom_inline { fmt; args = fl args; dtype }

(* Hash tables *)

module Tbl = Hashtbl.Make (struct
  type nonrec t = t
  let equal = ( = )
  let hash node = node.Hashcons.tag
end)

module Ref_tbl = (Hashtbl.Make (struct
  type nonrec t = t
  let equal = ( == )
  let hash node = node.Hashcons.tag
end) : Hashtbl.S with type key = t)

let toposort root =
  let state = Ref_tbl.create 256 in
  let order = ref [] in
  let rec visit node = match Ref_tbl.find_opt state node with
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
  let n = List.length nodes in
  let canon = Ref_tbl.create n in
  let lookup node = match Ref_tbl.find_opt canon node with Some n -> n | None -> node in
  List.iter
    (fun original ->
      let changed = List.exists (fun c -> lookup c != c) (children original) in
      let shared =
        if changed then
          mk ?tag:(tag original) (map_children lookup (view original))
        else original
      in
      Ref_tbl.replace canon original shared)
    nodes;
  lookup root

let is_alu node = match node.Hashcons.node.view with
  | Unary _ | Binary _ | Ternary _ -> true
  | _ -> false

let is_ptr node = Option.is_some (get_ptr_dtype node)
let dtype_or default node = match node_dtype node with Some dt -> dt | None -> default

let first_match rules node =
  List.find_map (fun rule -> rule node) rules

(* Validation *)

let validate root =
  let nodes = toposort root in
  let ids = Ref_tbl.create (List.length nodes) in
  List.iteri (fun i node -> Ref_tbl.add ids node i) nodes;
  let id node = match Ref_tbl.find_opt ids node with Some i -> i | None -> -1 in
  let fail node msg =
    Printf.ksprintf failwith "Kernel.validate: instruction %d: %s" (id node) msg
  in
  let rec get_dtype node =
    match node.Hashcons.node.view with
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
    | Some dt when Dtype.scalar dt = Dtype.Bool && Dtype.count dt = 1 -> ()
    | Some _ -> fail node (Printf.sprintf "%s must be bool scalar" ctx)
    | None -> fail node (Printf.sprintf "%s dtype not available" ctx)
  in
  let check_bool node ~ctx value =
    match get_dtype value with
    | Some dt when Dtype.scalar dt = Dtype.Bool -> ()
    | Some _ -> fail node (Printf.sprintf "%s must be bool" ctx)
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
    | Some dt when Dtype.scalar dt = Dtype.Index || Dtype.scalar dt = Dtype.Int32 -> ()
    | Some _ -> fail node (Printf.sprintf "%s must be index-like" ctx)
    | None -> fail node (Printf.sprintf "%s dtype not available" ctx)
  in
  let check_horizontal_reduce_src node ~src ~dtype =
    match get_dtype src with
    | Some src_dtype when Dtype.equal src_dtype dtype -> ()
    | Some src_dtype
      when Dtype.scalar src_dtype = Dtype.scalar dtype
           && Dtype.count src_dtype >= Dtype.count dtype
           && Dtype.count src_dtype mod Dtype.count dtype = 0 ->
        ()
    | Some got ->
        fail node
          (Printf.sprintf "Reduce src: expected %s or horizontal vector, got %s"
             (Dtype.to_string dtype) (Dtype.to_string got))
    | None -> fail node "Reduce src dtype not available"
  in
  let rec index_base node = match node.Hashcons.node.view with
    | After { src; _ } | Cast { src; _ } | Bitcast { src; _ } -> index_base src
    | Param _ | Param_image _ | Define_local _ | Define_reg _ | Bufferize _
    | Ptrcat _ ->
        true
    | Vectorize { srcs; _ } | Cat { srcs; _ } ->
        (* After do_expand, buffer ptrs may be wrapped in Vectorize/Cat *)
        srcs <> [] && List.for_all index_base srcs
    | _ -> false
  in
  let rec ptr_ref node = match node.Hashcons.node.view with
    | Index { dtype = Dtype.P pty; gate; _ } -> Some (node, pty, gate)
    | Index { dtype = Dtype.T _; _ } -> None
    | ( Ptrcat { dtype; _ }
      | Param { dtype; _ }
      | Param_image { dtype; _ }
      | Define_local { dtype; _ }
      | Define_reg { dtype; _ }
      | Bufferize { dtype; _ } ) ->
        Some (node, dtype, None)
    | Gep { src; dtype; _ } -> (
        match ptr_ref src with
        | Some (_, pty, gate) ->
            let pty = Dtype.ptr_with_base pty dtype in
            Some (node, pty, gate)
        | None -> None)
    | Cast { src; dtype = Dtype.P pty } ->
        let gate = match ptr_ref src with Some (_, _, g) -> g | None -> None in
        Some (node, pty, gate)
    | After { src; _ } | Cast { src; _ } | Bitcast { src; _ } -> ptr_ref src
    | _ -> None
  in
  let prod lst = List.fold_left ( * ) 1 lst in
  List.iter
    (fun instr ->
      match instr.Hashcons.node.view with
      | Sink _ | Group _ | After _ -> ()
      | Param { dtype; _ } | Param_image { dtype; _ } ->
          if Dtype.addrspace dtype <> Dtype.Global then
            fail instr "Param must have Global addrspace"
      | Define_local { dtype; _ } ->
          if Dtype.addrspace dtype <> Dtype.Local then
            fail instr "Define_local must have Local addrspace"
      | Define_reg { dtype; _ } ->
          if Dtype.addrspace dtype <> Dtype.Reg then
            fail instr "Define_reg must have Reg addrspace"
      | Define_var { lo; hi; dtype; _ } ->
          if Dtype.count dtype <> 1 then fail instr "Define_var must be scalar";
          if not (Dtype.is_int dtype || Dtype.scalar dtype = Dtype.Index) then
            fail instr "Define_var must be int/index";
          if lo > hi then fail instr "Define_var bounds invalid (lo > hi)"
      | Bufferize { ranges; dtype; opts; _ } ->
          if Dtype.addrspace dtype <> opts.addrspace then
            fail instr "Bufferize dtype addrspace mismatch";
          List.iter (check_index_like instr ~ctx:"Bufferize range") ranges
      | Const { value; dtype } -> (
          match Const.view value with
          | Bool _ ->
              if Dtype.scalar dtype <> Dtype.Bool then
                fail instr "Bool const must have bool dtype"
          | Int _ ->
              if not (Dtype.is_int dtype) then
                fail instr "Int const must have int/index dtype"
          | Float _ ->
              if not (Dtype.is_float dtype) then
                fail instr "Float const must have float dtype")
      | Invalid_index { dtype } ->
          if Dtype.scalar dtype <> Dtype.Index then
            fail instr "Invalid_index must have Index dtype"
      | Range { size; dtype; _ } ->
          if not (Dtype.is_int dtype) then
            fail instr "Range must have int/index";
          if Dtype.count dtype <> 1 then fail instr "Range must be scalar";
          check_dtype_eq instr ~ctx:"Range size" ~expected:(Some dtype)
            ~got:(get_dtype size)
      | End { ranges; _ } ->
          List.iter (check_index_like instr ~ctx:"End range") ranges
      | Barrier -> ()
      | Special { size; dtype; _ } ->
          if Dtype.count dtype <> 1 then fail instr "Special must be scalar";
          if not (Dtype.scalar dtype = Dtype.Index || Dtype.scalar dtype = Dtype.Int32) then
            fail instr "Special must be index or int32";
          check_dtype_eq instr ~ctx:"Special size" ~expected:(Some dtype)
            ~got:(get_dtype size)
      | Index { ptr; idxs; gate; dtype } -> (
          if idxs = [] then fail instr "Index must have at least one index";
          if not (index_base ptr) then
            fail instr "Index base must be a buffer define/bufferize";
          List.iter (check_index_like instr ~ctx:"Index operand") idxs;
          Option.iter (check_bool_scalar instr ~ctx:"Index gate") gate;
          (* Walk through Vectorize/Cast/After to find the underlying ptr dtype *)
          let rec find_ptr_dtype n = match n.Hashcons.node.view with
            | Vectorize { srcs = s :: _; _ } | Cat { srcs = s :: _; _ } -> find_ptr_dtype s
            | After { src; _ } | Cast { src; _ } | Bitcast { src; _ } -> find_ptr_dtype src
            | _ -> get_ptr_dtype n
          in
          match find_ptr_dtype ptr, dtype with
          | Some base_pty, Dtype.P pty
            when Dtype.equal (Dtype.base base_pty) (Dtype.base pty)
                 && Dtype.addrspace base_pty = Dtype.addrspace pty
                 && Dtype.ptr_v base_pty = Dtype.ptr_v pty ->
              ()
          | Some base_pty, Dtype.T dt
            when Dtype.scalar (Dtype.base base_pty) = Dtype.scalar dt ->
              (* Allow vectorized Index: base scalar matches but count may differ *)
              ()
          | Some base_pty, Dtype.P pty
            when Dtype.scalar (Dtype.base base_pty) = Dtype.scalar (Dtype.base pty) ->
              (* Allow vectorized Index: scalar matches, addrspace matches *)
              ()
          | Some _, _ -> fail instr "Index dtype must match base pointer type"
          | None, _ -> fail instr "Index base dtype not available")
      | Ptrcat { srcs; dtype } ->
          if srcs = [] then fail instr "Ptrcat must have at least one source";
          let total_vcount = ref 0 in
          List.iter
            (fun src ->
              match ptr_ref src with
              | Some (_, pty, _) ->
                  if Dtype.addrspace pty <> Dtype.addrspace dtype then
                    fail instr "Ptrcat addrspace mismatch";
                  if not (Dtype.equal (Dtype.base pty) (Dtype.base dtype)) then
                    fail instr "Ptrcat base dtype mismatch";
                  total_vcount := !total_vcount + Dtype.count (Dtype.base pty)
              | None -> fail instr "Ptrcat sources must be pointers")
            srcs;
          if !total_vcount <> Dtype.ptr_v dtype then fail instr "Ptrcat vcount mismatch"
      | Load { src; alt; dtype } -> (
          match ptr_ref src with
          | Some (_, pty, gate) -> (
              (* Allow widened Load dtype (e.g. f32×4 from f32 pointer).
                 Intermediate state after do_expand.  Check scalar match. *)
              if Dtype.scalar dtype <> Dtype.scalar (Dtype.base pty) then
                check_dtype_eq instr ~ctx:"Load dtype" ~expected:(Some (Dtype.base pty))
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
          let dst_ok = match ptr_ref dst with
            | Some (_, pty, _) -> (
                (* Allow widened Store value (e.g. i32×4 to i32 pointer).
                   Intermediate state after do_expand.  Check scalar match. *)
                match get_dtype value with
                | Some vdt when Dtype.scalar vdt <> Dtype.scalar (Dtype.base pty) ->
                    check_dtype_eq instr ~ctx:"Store value"
                      ~expected:(Some (Dtype.base pty)) ~got:(Some vdt)
                | _ -> ());
                true
            | None -> false
          in
          (* Also accept value-typed Index as dst (before pm_add_loads). *)
          if not dst_ok then
            let rec has_index n = match n.Hashcons.node.view with
              | Index { dtype = Dtype.T _; _ } -> true
              | After { src; _ } | Cast { src; _ } | Bitcast { src; _ } -> has_index src
              | _ -> false
            in
            if not (has_index dst) then
              fail instr "Store dst must reference a pointer or value-typed Index")
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
              if Dtype.scalar dtype <> Dtype.Bool then
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
              check_bool instr ~ctx:"Where condition" a;
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
          (* For ptr-typed Vectorize, vcount is the ptr v field;
             for value-typed, it's the dtype count. *)
          let vcount = match dtype with
            | Dtype.P p -> Dtype.ptr_v p
            | Dtype.T t -> Dtype.count t
          in
          if vcount <> List.length srcs then
            fail instr "Vectorize dtype count must match operand count";
          List.iter
            (fun src ->
              let ok = match dtype, get_dtype src with
                | Dtype.T dt, Some sdt ->
                    Dtype.count sdt = 1 && Dtype.scalar sdt = Dtype.scalar dt
                | Dtype.P p, _ -> (
                    match get_ptr_dtype src with
                    | Some sp ->
                        Dtype.equal (Dtype.base sp) (Dtype.base p)
                        && Dtype.addrspace sp = Dtype.addrspace p
                    | None -> false)
                | _, None -> false
              in
              if not ok then
                fail instr "Vectorize operands must be scalar and match")
            srcs
      | Cat { srcs; dtype } ->
          if srcs = [] then fail instr "Cat must have at least one operand";
          let total = ref 0 in
          List.iter
            (fun src ->
              match get_dtype src with
              | Some dt ->
                  if Dtype.scalar dt <> Dtype.scalar dtype then
                    fail instr "Cat operand scalar mismatch";
                  total := !total + Dtype.count dt
              | None -> fail instr "Cat operand dtype not available")
            srcs;
          if !total <> Dtype.count dtype then fail instr "Cat count mismatch"
      | Gep { src; idxs; dtype } -> (
          if idxs = [] then fail instr "Gep must have at least one index";
          match get_dtype src with
          | Some dt when Dtype.count dt > 1 ->
              List.iter (fun idx ->
                if idx < 0 || idx >= Dtype.count dt then
                  fail instr "Gep index out of bounds") idxs;
              let n = List.length idxs in
              if n = 1 then begin
                if Dtype.count dtype <> 1 || Dtype.scalar dtype <> Dtype.scalar dt then
                  fail instr "Gep dtype must be scalar of source"
              end else begin
                if Dtype.count dtype <> n || Dtype.scalar dtype <> Dtype.scalar dt then
                  fail instr "Gep dtype must be vec(scalar, len(idxs))"
              end
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
          if Dtype.scalar dtype <> Dtype.Void then begin
            let expected = prod (List.map snd axes) * Dtype.count dtype in
            match get_dtype src with
            | Some dt when Dtype.count dt = expected -> ()
            | Some _ -> fail instr "Unroll source count mismatch"
            | None -> fail instr "Unroll source dtype not available"
          end
      | Contract { axes; dtype; _ } ->
          if Dtype.scalar dtype <> Dtype.Void then begin
            let expected = prod (List.map snd axes) in
            if Dtype.count dtype <> expected then
              fail instr "Contract dtype count mismatch"
          end
      | Wmma _ -> ()
      | Cast _ | Bitcast _ | Custom _ | Custom_inline _ -> ())
    nodes

(* Rewriting *)

(* Debug: columnar print_uops for kernel DAG inspection.
   Defined before graph_rewrite so the debug hook can call it. *)

let debug_level =
  lazy (match Sys.getenv_opt "DEBUG" with
    | Some s -> (try int_of_string s with _ -> 0) | None -> 0)

let view_op_name = function
  | Sink _ -> "Ops.SINK"
  | Group _ -> "Ops.GROUP"
  | After _ -> "Ops.AFTER"
  | Param _ -> "Ops.PARAM"
  | Param_image _ -> "Ops.PARAM_IMAGE"
  | Define_local _ -> "Ops.DEFINE_LOCAL"
  | Define_reg _ -> "Ops.DEFINE_REG"
  | Define_var _ -> "Ops.DEFINE_VAR"
  | Bufferize _ -> "Ops.BUFFERIZE"
  | Const _ -> "Ops.CONST"
  | Invalid_index _ -> "Ops.INVALID_INDEX"
  | Index _ -> "Ops.INDEX"
  | Ptrcat _ -> "Ops.PTRCAT"
  | Load _ -> "Ops.LOAD"
  | Store _ -> "Ops.STORE"
  | Unary { op; _ } -> "Ops." ^ String.uppercase_ascii (Format.asprintf "%a" Op.pp_unary op)
  | Binary { op; _ } -> "Ops." ^ String.uppercase_ascii (Format.asprintf "%a" Op.pp_binary op)
  | Ternary { op; _ } -> "Ops." ^ String.uppercase_ascii (Format.asprintf "%a" Op.pp_ternary op)
  | Cast _ -> "Ops.CAST"
  | Bitcast _ -> "Ops.BITCAST"
  | Vectorize _ -> "Ops.VECTORIZE"
  | Cat _ -> "Ops.CAT"
  | Gep _ -> "Ops.GEP"
  | Range _ -> "Ops.RANGE"
  | End _ -> "Ops.END"
  | Barrier -> "Ops.BARRIER"
  | Special _ -> "Ops.SPECIAL"
  | Reduce { op; _ } -> "Ops.REDUCE"
  | Unroll _ -> "Ops.UNROLL"
  | Contract _ -> "Ops.CONTRACT"
  | Wmma _ -> "Ops.WMMA"
  | Custom _ -> "Ops.CUSTOM"
  | Custom_inline _ -> "Ops.CUSTOM_INLINE"

let scalar_name (s : Dtype.scalar) = match s with
  | Float32 -> "dtypes.float" | Float16 -> "dtypes.half"
  | Float64 -> "dtypes.double" | Int32 -> "dtypes.int"
  | Int64 -> "dtypes.long" | Int16 -> "dtypes.short"
  | Int8 -> "dtypes.char" | Uint8 -> "dtypes.uchar"
  | Uint16 -> "dtypes.ushort" | Uint32 -> "dtypes.uint"
  | Uint64 -> "dtypes.ulong" | Bool -> "dtypes.bool"
  | Index -> "dtypes.weakint"
  | _ -> Printf.sprintf "dtypes.%s" (Format.asprintf "%a" Dtype.pp_scalar s)

let debug_dtype_str dt =
  let count = Dtype.count dt in
  let base = scalar_name (Dtype.scalar dt) in
  if count > 1 then Printf.sprintf "%s.vec(%d)" base count
  else base

let debug_ptr_str (ptr : Dtype.ptr) =
  let base = debug_dtype_str (Dtype.base ptr) in
  let s = Printf.sprintf "%s.ptr(%d)" base (Dtype.ptr_size ptr) in
  if Dtype.ptr_v ptr > 1 then Printf.sprintf "%s.vec(%d)" s (Dtype.ptr_v ptr)
  else s

let dtype_str_full node = match view node with
  | Param { dtype; _ } | Param_image { dtype; _ } | Define_local { dtype; _ }
  | Define_reg { dtype; _ } | Ptrcat { dtype; _ } | Bufferize { dtype; _ } ->
      debug_ptr_str dtype
  | Index { dtype = Dtype.P p; _ } | Cast { dtype = Dtype.P p; _ }
  | Vectorize { dtype = Dtype.P p; _ } ->
      debug_ptr_str p
  | Index { dtype = Dtype.T t; _ } | Cast { dtype = Dtype.T t; _ }
  | Vectorize { dtype = Dtype.T t; _ } ->
      debug_dtype_str t
  | Sink _ | Group _ | After _ | Store _ | End _ | Barrier | Custom _ ->
      "dtypes.void"
  | _ ->
      match node_dtype node with Some dt -> debug_dtype_str dt | None -> "dtypes.void"

let axis_kind_str = function
  | Axis_kind.Global -> "AxisType.GLOBAL"
  | Axis_kind.Thread -> "AxisType.THREAD"
  | Axis_kind.Local -> "AxisType.LOCAL"
  | Axis_kind.Warp -> "AxisType.WARP"
  | Axis_kind.Loop -> "AxisType.LOOP"
  | Axis_kind.Upcast -> "AxisType.UPCAST"
  | Axis_kind.Group_reduce -> "AxisType.GROUP_REDUCE"
  | Axis_kind.Reduce -> "AxisType.REDUCE"
  | Axis_kind.Unroll -> "AxisType.UNROLL"
  | Axis_kind.Placeholder -> "AxisType.PLACEHOLDER"

(* Python tuple repr: () for empty, (x,) for single, (x, y) for multi *)
let py_tuple = function
  | [] -> "()"
  | [x] -> Printf.sprintf "(%s,)" x
  | items -> Printf.sprintf "(%s)" (String.concat ", " items)

let const_value_str value =
  match Const.view value with
  | Bool v -> string_of_bool v
  | Int v -> Int64.to_string v
  | Float v -> Printf.sprintf "%g" v

let view_arg = function
  | Const { value; _ } -> const_value_str value
  | Param { idx; _ } | Param_image { idx; _ } -> string_of_int idx
  | Define_var { name; lo; hi; _ } ->
      Printf.sprintf "('%s', %d, %d)" name lo hi
  | Define_local { size; _ } | Define_reg { size; _ } ->
      Printf.sprintf "size=%d" size
  | Range { axis; kind; _ } ->
      Printf.sprintf "(%d, %s)" axis (axis_kind_str kind)
  | Special { dim; _ } -> Format.asprintf "%a" Special_dim.pp dim
  | Reduce { op; _ } ->
      "Ops." ^ String.uppercase_ascii (Format.asprintf "%a" Op.pp_reduce op)
  | Wmma { name; dims = n, m, k; _ } ->
      Printf.sprintf "%s %dx%dx%d" name n m k
  | Gep { idxs; _ } ->
      Printf.sprintf "(%s,)" (String.concat ", " (List.map string_of_int idxs))
  | Custom { fmt; _ } | Custom_inline { fmt; _ } -> fmt
  | Sink { kernel_info = Some ki; _ } ->
      let axis_types = py_tuple (List.map axis_kind_str ki.axis_kinds) in
      let opt_repr opt =
        let op, axis, arg = match opt with
          | Opt.Local { axis; amount } -> "OptOps.LOCAL", string_of_int axis, string_of_int amount
          | Opt.Upcast { axis; amount } -> "OptOps.UPCAST", string_of_int axis, string_of_int amount
          | Opt.Unroll { axis; amount } -> "OptOps.UNROLL", string_of_int axis, string_of_int amount
          | Opt.Group { axis; amount } -> "OptOps.GROUP", string_of_int axis, string_of_int amount
          | Opt.Grouptop { axis; amount } -> "OptOps.GROUPTOP", string_of_int axis, string_of_int amount
          | Opt.Thread { axis; amount } -> "OptOps.THREAD", string_of_int axis, string_of_int amount
          | Opt.Nolocals -> "OptOps.NOLOCALS", "None", "None"
          | Opt.Tc { axis; tc_select; tc_opt; use_tc } ->
              "OptOps.TC", string_of_int axis,
              Printf.sprintf "(%d, %d, %d)" tc_select tc_opt use_tc
          | Opt.Padto { axis; amount } -> "OptOps.PADTO", string_of_int axis, string_of_int amount
          | Opt.Swap { axis; with_axis } -> "OptOps.SWAP", string_of_int axis, string_of_int with_axis
        in
        Printf.sprintf "Opt(op=%s, axis=%s, arg=%s)" op axis arg
      in
      let applied_opts = py_tuple (List.map opt_repr ki.applied_opts) in
      let opts_to_apply = match ki.opts_to_apply with
        | None -> "None"
        | Some opts -> py_tuple (List.map opt_repr opts)
      in
      let estimates = match ki.estimates with
        | None -> "None"
        | Some _ -> "..."
      in
      Printf.sprintf
        "KernelInfo(name='%s', axis_types=%s, dont_use_locals=%s, applied_opts=%s, opts_to_apply=%s, estimates=%s)"
        ki.name axis_types
        (if ki.dont_use_locals then "True" else "False")
        applied_opts opts_to_apply estimates
  | _ -> "None"

let print_uops ?label root =
  let nodes = toposort root in
  let ids = Tbl.create (List.length nodes) in
  List.iteri (fun i node -> Tbl.add ids node i) nodes;
  (* Compute ranges per node (which RANGE nodes each value lives within).
     For each node, its ranges are the union of its children's ranges
     minus any ended ranges. *)
  let range_map : t list Ref_tbl.t = Ref_tbl.create (List.length nodes) in
  List.iter (fun node ->
    let child_ranges =
      List.fold_left (fun acc c ->
        let c_rngs = match Ref_tbl.find_opt range_map c with
          | Some r -> r | None -> [] in
        List.fold_left (fun a r ->
          if List.exists (fun x -> x == r) a then a else r :: a) acc c_rngs)
        [] (children node)
    in
    let ended = match view node with
      | End { ranges; _ } -> ranges
      | Reduce { ranges; _ } -> ranges
      | Store { ranges; _ } -> ranges
      | Bufferize { ranges; _ } -> ranges
      | _ -> []
    in
    let rngs =
      List.filter (fun r ->
        not (List.exists (fun e -> e == r) ended)) child_ranges
    in
    let rngs = match view node with
      | Range _ -> if List.exists (fun r -> r == node) rngs then rngs else node :: rngs
      | _ -> rngs
    in
    Ref_tbl.replace range_map node rngs) nodes;
  (match label with
   | Some l -> Printf.eprintf "=== %s ===\n" l
   | None -> ());
  List.iteri
    (fun i node ->
      let v = view node in
      let src_strs =
        List.map
          (fun c -> match view c with
            | Const { value; _ } -> Printf.sprintf "'%s'" (const_value_str value)
            | _ ->
                (match Tbl.find_opt ids c with
                 | Some idx -> string_of_int idx | None -> "--"))
          (children node)
      in
      let srcs = Printf.sprintf "[%s]" (String.concat ", " src_strs) in
      let ranges =
        match Ref_tbl.find_opt range_map node with
        | None | Some [] -> ""
        | Some rngs ->
            let get_axis r = match view r with Range { axis; _ } -> axis | _ -> max_int in
            let sorted = List.sort (fun a b -> compare (get_axis a) (get_axis b)) rngs in
            String.concat "," (List.map (fun r -> string_of_int (get_axis r)) sorted)
      in
      Printf.eprintf "%4d %-20s: %-10s %-40s %-32s %s\n"
        i (view_op_name v) ranges (dtype_str_full node) srcs (view_arg v))
    nodes;
  Printf.eprintf "%!"

(* Stack-based graph rewrite (unified_rewrite).
   Stage 0: push children, advance to stage 1.
   Stage 1: rebuild with rewritten children, apply rewrite.
   Stage 2: link original node to the final result of the rewritten node.
   Uses a waitlist for children not yet ready. *)
let graph_rewrite ?(name="") rewrite root =
  let replace : t Ref_tbl.t = Ref_tbl.create 256 in
  let on_stack : unit Ref_tbl.t = Ref_tbl.create 256 in
  let waitlist : (t * int * t) list Ref_tbl.t = Ref_tbl.create 16 in
  let stack : (t * int * t) Stack.t = Stack.create () in
  let lookup c =
    match Ref_tbl.find_opt replace c with Some r -> r | None -> c
  in
  let set_replace n v =
    Ref_tbl.replace replace n v;
    match Ref_tbl.find_opt waitlist n with
    | Some waiting ->
        Ref_tbl.remove waitlist n;
        List.iter (fun entry -> Stack.push entry stack) waiting
    | None -> ()
  in
  Stack.push (root, 0, root) stack;
  Ref_tbl.replace on_stack root ();
  let counter = ref 0 in
  while not (Stack.is_empty stack) do
    let n, stage, new_n = Stack.pop stack in
    if Ref_tbl.mem replace n then ()
    else begin
      incr counter;
      if !counter > 250000 then
        failwith (Printf.sprintf "graph_rewrite(%s): %d nodes" name !counter);
      if !counter >= 249990 then
        Printf.eprintf "  [%s] %d: stage=%d %s tag=%d in_replace=%b\n%!" name !counter stage
          (view_op_name new_n.Hashcons.node.view) new_n.Hashcons.tag (Ref_tbl.mem replace new_n);
      if stage = 0 then begin
        (* Stage 0: push self at stage 1, then push children *)
        Stack.push (n, 1, new_n) stack;
        List.iter (fun x ->
          if not (Ref_tbl.mem on_stack x) then begin
            Stack.push (x, 0, x) stack;
            Ref_tbl.replace on_stack x ()
          end) (List.rev (children new_n))
      end
      else if stage = 1 then begin
        (* Stage 1: check all children are ready *)
        let all_ready = ref true in
        let new_src =
          List.map (fun x ->
            match Ref_tbl.find_opt replace x with
            | Some r -> r
            | None -> all_ready := false; x)
            (children new_n)
        in
        if not !all_ready then begin
          (* Some child not ready — register in waitlist *)
          let missing = List.find (fun x -> not (Ref_tbl.mem replace x)) (children new_n) in
          let prev = match Ref_tbl.find_opt waitlist missing with Some l -> l | None -> [] in
          Ref_tbl.replace waitlist missing ((n, 1, new_n) :: prev)
        end
        else begin
          let old_src = children new_n in
          let changed = not (List.for_all2 (fun a b -> a == b) old_src new_src) in
          if not changed then begin
            (* Children unchanged. Try rewrite. *)
            match rewrite new_n with
            | None ->
                set_replace n new_n
            | Some rewritten when rewritten == new_n ->
                (* Identity rewrite — treat as no match. *)
                set_replace n new_n
            | Some rewritten ->
                Stack.push (n, 2, rewritten) stack;
                Stack.push (rewritten, 0, rewritten) stack
          end
          else begin
            (* Children changed. Rebuild and push for full processing. *)
            let rebuilt = mk ?tag:(tag new_n) (map_children lookup (view new_n)) in
            Stack.push (n, 2, rebuilt) stack;
            Stack.push (rebuilt, 0, rebuilt) stack
          end
        end
      end
      else begin
        (* Stage 2: link n → result of new_n *)
        match Ref_tbl.find_opt replace new_n with
        | Some result ->
            set_replace n result
        | None ->
            (* new_n not ready — register in waitlist *)
            let prev = match Ref_tbl.find_opt waitlist new_n with Some l -> l | None -> [] in
            Ref_tbl.replace waitlist new_n ((n, 2, new_n) :: prev)
      end
    end
  done;
  let result = lookup root in
  if Lazy.force debug_level >= 6 && name <> "" then
    print_uops ~label:name result;
  result

let propagate_tag tags old_node new_node =
  Option.iter
    (fun t -> Option.iter (Ref_tbl.replace t new_node) (Ref_tbl.find_opt t old_node))
    tags

let substitute ?tags mappings root =
  let tbl = Ref_tbl.create (List.length mappings) in
  List.iter
    (fun (old_node, new_node) ->
      Ref_tbl.replace tbl old_node new_node;
      propagate_tag tags old_node new_node)
    mappings;
  let nodes = toposort root in
  let rebuilt = Ref_tbl.create (List.length nodes) in
  let lookup node = match Ref_tbl.find_opt rebuilt node with Some n -> n | None -> node in
  List.iter
    (fun node ->
      match Ref_tbl.find_opt tbl node with
      | Some replacement -> Ref_tbl.replace rebuilt node replacement
      | None ->
          if List.exists (fun c -> lookup c != c) (children node) then begin
            let new_node = mk ?tag:(tag node) (map_children lookup (view node)) in
            Ref_tbl.replace rebuilt node new_node;
            propagate_tag tags node new_node
          end)
    nodes;
  lookup root

(* Analysis *)

let backward_slice root = toposort root

let in_backward_slice needle haystack =
  let visited = Ref_tbl.create 64 in
  let rec search node =
    if node == needle then true
    else if Ref_tbl.mem visited node then false
    else begin
      Ref_tbl.add visited node ();
      List.exists search (children node)
    end
  in
  search haystack

let find_nodes pred root =
  List.filter pred (toposort root)

(* Node predicates *)

let is_range node = match node.Hashcons.node.view with Range _ -> true | _ -> false
let is_const node = match node.Hashcons.node.view with Const _ -> true | _ -> false

(* Range analysis *)

(* range_start: index at which range args begin for ops that carry ranges.
   Bufferize: 1, Reduce: 1, Store: 2, Wmma: 3, End: 1. *)
let range_start node = match node.Hashcons.node.view with
  | Bufferize _ -> Some 1
  | Reduce _ -> Some 1
  | Store _ -> Some 2
  | Wmma _ -> Some 3
  | End _ -> Some 1
  | _ -> None

let rec ended_ranges ?(live = fun _ -> []) (node : t) : t list =
  match range_start node with
  | Some off -> List.filteri (fun i _ -> i >= off) (children node)
  | None -> match node.Hashcons.node.view with
    | After { deps; _ } -> List.concat_map (ended_ranges ~live) deps
    | Contract { axes; _ } ->
        let axis_ids = List.map fst axes in
        let src = List.hd (children node) in
        List.filter
          (fun r -> match r.Hashcons.node.view with
            | Range { axis; _ } -> List.mem axis axis_ids
            | _ -> false)
          (live src)
    | _ -> []

let live_ranges_tbl root =
  let nodes = toposort root in
  let tbl = Ref_tbl.create (List.length nodes) in
  let get node = match Ref_tbl.find_opt tbl node with Some r -> r | None -> [] in
  List.iter
    (fun node ->
      let live = Ref_tbl.create 16 in
      List.iter (fun c -> List.iter (fun r -> Ref_tbl.replace live r ()) (get c)) (children node);
      List.iter
        (fun er ->
          if is_range er then Ref_tbl.remove live er
          else List.iter (fun r -> Ref_tbl.remove live r) (get er))
        (ended_ranges ~live:get node);
      if is_range node then Ref_tbl.replace live node ();
      Ref_tbl.replace tbl node (Ref_tbl.fold (fun k () acc -> k :: acc) live []))
    nodes;
  tbl

let live_ranges node =
  let tbl = live_ranges_tbl node in
  match Ref_tbl.find_opt tbl node with Some r -> r | None -> []

(* Accessors *)

let range_size node = match node.Hashcons.node.view with
  | Range { size; _ } -> size
  | _ -> invalid_arg "Kernel.range_size: not a Range node"

let range_axis node = match node.Hashcons.node.view with
  | Range { axis; _ } -> axis
  | _ -> invalid_arg "Kernel.range_axis: not a Range node"

let range_kind node = match node.Hashcons.node.view with
  | Range { kind; _ } -> kind
  | _ -> invalid_arg "Kernel.range_kind: not a Range node"

let range_sub node = match node.Hashcons.node.view with
  | Range { sub; _ } -> sub
  | _ -> invalid_arg "Kernel.range_sub: not a Range node"

let const_to_int node = match node.Hashcons.node.view with
  | Const { value; _ } -> (
      match Const.view value with
      | Int n -> Int64.to_int n
      | Bool b -> if b then 1 else 0
      | Float _ -> invalid_arg "Kernel.const_to_int: float constant")
  | _ -> invalid_arg "Kernel.const_to_int: not a Const node"

(* Operators *)

module O = struct
  let ( + ) a b = binary ~op:`Add ~lhs:a ~rhs:b
  let ( * ) a b = binary ~op:`Mul ~lhs:a ~rhs:b
  let ( / ) a b = binary ~op:`Idiv ~lhs:a ~rhs:b
  let ( mod ) a b = binary ~op:`Mod ~lhs:a ~rhs:b
  let ( < ) a b = binary ~op:`Cmplt ~lhs:a ~rhs:b
  let eq a b = binary ~op:`Cmpeq ~lhs:a ~rhs:b
  let ne a b = binary ~op:`Cmpne ~lhs:a ~rhs:b
  let where cond then_ else_ = ternary ~op:`Where ~a:cond ~b:then_ ~c:else_
  let neg x = unary ~op:`Neg ~src:x
  let not_ x = binary ~op:`Cmpeq ~lhs:x ~rhs:(zero_like x)
  let cast dtype node = mk (Cast { src = node; dtype = Dtype.to_any dtype })
  let int_ = const_int
  let float_ = const_float
  let bool_ = const_bool
end

(* Structural comparison — canonical ordering for commutative operands. *)

let view_ordinal = function
  | Define_var _ -> 1 | Special _ -> 3 | Define_local _ -> 4
  | Define_reg _ -> 5 | Param _ -> 8 | Param_image _ -> 9
  | Sink _ -> 15 | After _ -> 16 | Group _ -> 17 | Gep _ -> 18
  | Vectorize _ -> 19 | Index _ -> 20 | Load _ -> 21 | Store _ -> 22
  | Wmma _ -> 23 | Cast _ -> 24 | Bitcast _ -> 25 | Unary _ -> 26
  | Binary _ -> 27 | Ternary _ -> 28 | Barrier -> 29 | Range _ -> 30
  | End _ -> 31 | Const _ -> 33 | Custom _ -> 34 | Custom_inline _ -> 35
  | Reduce _ -> 100 | Cat _ | Ptrcat _ | Unroll _ | Contract _
  | Bufferize _ | Invalid_index _ -> 100

(* Compare the non-child, non-dtype payload fields of two views of the same
   variant.  Returns 0 when both views are the same constructor with identical
   payload, a negative or positive integer otherwise. Assumes [view_ordinal a =
   view_ordinal b]; cross-constructor comparison is handled by the caller. *)
let compare_view_args a b =
  match a, b with
  | Sink _, Sink _ | Group _, Group _ | After _, After _
  | Barrier, Barrier | End _, End _ ->
      0
  | Param { idx = i1; _ }, Param { idx = i2; _ } -> Int.compare i1 i2
  | Param_image { idx = i1; width = w1; height = h1; _ },
    Param_image { idx = i2; width = w2; height = h2; _ } ->
      let c = Int.compare i1 i2 in
      if c <> 0 then c
      else let c = Int.compare w1 w2 in
      if c <> 0 then c else Int.compare h1 h2
  | Define_local { size = s1; _ }, Define_local { size = s2; _ } ->
      Int.compare s1 s2
  | Define_reg { size = s1; slot = sl1; _ }, Define_reg { size = s2; slot = sl2; _ } ->
      let c = Int.compare s1 s2 in
      if c <> 0 then c else Int.compare sl1 sl2
  | Define_var { name = n1; lo = l1; hi = h1; _ },
    Define_var { name = n2; lo = l2; hi = h2; _ } ->
      let c = String.compare n1 n2 in
      if c <> 0 then c
      else let c = Int.compare l1 l2 in
      if c <> 0 then c else Int.compare h1 h2
  | Bufferize { opts = o1; _ }, Bufferize { opts = o2; _ } ->
      Stdlib.compare o1 o2
  | Const { value = v1; _ }, Const { value = v2; _ } -> Const.compare v1 v2
  | Invalid_index _, Invalid_index _ -> 0
  | Index _, Index _ -> 0
  | Ptrcat _, Ptrcat _ -> 0
  | Load _, Load _ -> 0
  | Store _, Store _ -> 0
  | Unary { op = o1; _ }, Unary { op = o2; _ } -> Op.compare_unary o1 o2
  | Binary { op = o1; _ }, Binary { op = o2; _ } -> Op.compare_binary o1 o2
  | Ternary { op = o1; _ }, Ternary { op = o2; _ } -> Op.compare_ternary o1 o2
  | Cast _, Cast _ -> 0
  | Bitcast _, Bitcast _ -> 0
  | Vectorize _, Vectorize _ -> 0
  | Cat _, Cat _ -> 0
  | Gep { idxs = i1; _ }, Gep { idxs = i2; _ } -> Stdlib.compare i1 i2
  | Range { axis = a1; sub = s1; kind = k1; _ },
    Range { axis = a2; sub = s2; kind = k2; _ } ->
      let c = Int.compare a1 a2 in
      if c <> 0 then c
      else let c = Stdlib.compare s1 s2 in
      if c <> 0 then c else Axis_kind.compare k1 k2
  | Special { dim = d1; _ }, Special { dim = d2; _ } -> Special_dim.compare d1 d2
  | Reduce { op = o1; _ }, Reduce { op = o2; _ } -> Op.compare_reduce o1 o2
  | Unroll { axes = a1; _ }, Unroll { axes = a2; _ } -> Stdlib.compare a1 a2
  | Contract { axes = a1; _ }, Contract { axes = a2; _ } -> Stdlib.compare a1 a2
  | Wmma { name = n1; dims = d1; dtype_in = di1; dtype_out = do1;
           device = dv1; threads = t1; upcast_axes = u1; reduce_axes = r1; _ },
    Wmma { name = n2; dims = d2; dtype_in = di2; dtype_out = do2;
           device = dv2; threads = t2; upcast_axes = u2; reduce_axes = r2; _ } ->
      let c = String.compare n1 n2 in
      if c <> 0 then c
      else let c = Stdlib.compare d1 d2 in
      if c <> 0 then c
      else let c = Stdlib.compare di1 di2 in
      if c <> 0 then c
      else let c = Stdlib.compare do1 do2 in
      if c <> 0 then c
      else let c = String.compare dv1 dv2 in
      if c <> 0 then c
      else let c = Int.compare t1 t2 in
      if c <> 0 then c
      else let c = Stdlib.compare u1 u2 in
      if c <> 0 then c else Stdlib.compare r1 r2
  | Custom { fmt = f1; _ }, Custom { fmt = f2; _ } -> String.compare f1 f2
  | Custom_inline { fmt = f1; _ }, Custom_inline { fmt = f2; _ } ->
      String.compare f1 f2
  | _ ->
      (* Different constructors — fall through to ordinal comparison. *)
      0

(* Full dtype of a node for structural comparison, using the precise any type. *)
let node_any_dtype node = match node.Hashcons.node.view with
  | Param { dtype; _ } | Param_image { dtype; _ } | Define_local { dtype; _ }
  | Define_reg { dtype; _ } | Bufferize { dtype; _ } | Ptrcat { dtype; _ } ->
      Some (Dtype.ptr_to_any dtype)
  | Index { dtype; _ } | Cast { dtype; _ } | Vectorize { dtype; _ } ->
      Some dtype
  | Define_var { dtype; _ } | Const { dtype; _ } | Invalid_index { dtype; _ }
  | Load { dtype; _ } | Unary { dtype; _ } | Binary { dtype; _ }
  | Ternary { dtype; _ } | Bitcast { dtype; _ }
  | Cat { dtype; _ } | Gep { dtype; _ }
  | Range { dtype; _ } | Special { dtype; _ } | Reduce { dtype; _ }
  | Unroll { dtype; _ } | Contract { dtype; _ } | Wmma { dtype; _ }
  | Custom_inline { dtype; _ } ->
      Some (Dtype.to_any dtype)
  | Sink _ | Group _ | After _ | Store _ | End _ | Barrier | Custom _ ->
      None

let compare_opt_any_dtype a b =
  match a, b with
  | None, None -> 0
  | None, Some _ -> -1
  | Some _, None -> 1
  | Some a, Some b -> Dtype.any_compare a b

let rec compare_structure a b =
  if a == b then 0
  else
    let va = view a and vb = view b in
    let c = Int.compare (view_ordinal va) (view_ordinal vb) in
    if c <> 0 then c
    else let c = compare_opt_any_dtype (node_any_dtype a) (node_any_dtype b) in
    if c <> 0 then c
    else let c = compare_view_args va vb in
    if c <> 0 then c
    else compare_children_struct (children a) (children b)
and compare_children_struct xs ys =
  match xs, ys with
  | [], [] -> 0 | [], _ -> -1 | _, [] -> 1
  | x :: xs', y :: ys' ->
      let c = compare_structure x y in
      if c <> 0 then c else compare_children_struct xs' ys'

(* Formatting *)

let pp_comma fmt () = Format.fprintf fmt ", "

let pp_ptr fmt (dtype : Dtype.ptr) =
  Format.fprintf fmt "%s" (Dtype.ptr_to_string dtype)

let pp_axes fmt axes =
  Format.pp_print_list ~pp_sep:pp_comma
    (fun fmt (a, s) -> Format.fprintf fmt "(%d, %d)" a s)
    fmt axes

let pp_view_with ids fmt instr =
  let pp_ref fmt node = Format.fprintf fmt "%%%d" (Tbl.find ids node) in
  let pp_refs fmt refs = Format.pp_print_list ~pp_sep:pp_comma pp_ref fmt refs in
  let pp_opt_ref label fmt = function
    | None -> () | Some n -> Format.fprintf fmt " %s=%a" label pp_ref n
  in
  (match tag instr with Some t -> Format.fprintf fmt "[%s] " t | None -> ());
  match instr.Hashcons.node.view with
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
  | Define_reg { size; dtype; slot } ->
      Format.fprintf fmt "define_reg %a, size=%d, slot=%d" pp_ptr dtype size slot
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
        (pp_opt_ref "gate") gate Dtype.pp_any dtype
  | Ptrcat { srcs; dtype } ->
      Format.fprintf fmt "ptrcat %a : %a" pp_refs srcs pp_ptr dtype
  | Load { src; alt; dtype } ->
      Format.fprintf fmt "load %a%a : %a" pp_ref src
        (pp_opt_ref "alt") alt Dtype.pp dtype
  | Store { dst; value; ranges } ->
      Format.fprintf fmt "store %a, %a, ranges=[%a]" pp_ref dst pp_ref value
        pp_refs ranges
  | Unary { op; src; dtype } ->
      Format.fprintf fmt "%a %a : %a" Op.pp_unary op pp_ref src Dtype.pp dtype
  | Cast { src; dtype } ->
      Format.fprintf fmt "cast %a : %a" pp_ref src Dtype.pp_any dtype
  | Bitcast { src; dtype } ->
      Format.fprintf fmt "bitcast %a : %a" pp_ref src Dtype.pp dtype
  | Binary { op; lhs; rhs; dtype } ->
      Format.fprintf fmt "%a %a, %a : %a" Op.pp_binary op pp_ref lhs pp_ref rhs
        Dtype.pp dtype
  | Ternary { op; a; b; c; dtype } ->
      Format.fprintf fmt "%a %a, %a, %a : %a" Op.pp_ternary op pp_ref a pp_ref b
        pp_ref c Dtype.pp dtype
  | Vectorize { srcs; dtype } ->
      Format.fprintf fmt "vec %a : %a" pp_refs srcs Dtype.pp_any dtype
  | Cat { srcs; dtype } ->
      Format.fprintf fmt "cat %a : %a" pp_refs srcs Dtype.pp dtype
  | Gep { src; idxs; dtype } ->
      Format.fprintf fmt "gep %a, [%a] : %a" pp_ref src
        (Format.pp_print_list ~pp_sep:(fun fmt () -> Format.fprintf fmt ";")
           Format.pp_print_int) idxs
        Dtype.pp dtype
  | Range { size; dtype; axis; sub; kind } ->
      Format.fprintf fmt "range %a : %a [axis=%d, %a%a]" pp_ref size Dtype.pp
        dtype axis Axis_kind.pp kind
        (fun fmt -> function
          | [] -> ()
          | sub ->
              let pp_semi fmt () = Format.fprintf fmt ";" in
              Format.fprintf fmt ", sub=[%a]"
                (Format.pp_print_list ~pp_sep:pp_semi Format.pp_print_int) sub)
        sub
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
  | Wmma { name; a; b; c; dtype; dims = n, m, k; dtype_in; dtype_out;
           device; threads; _ } ->
      Format.fprintf fmt
        "wmma.%s %a, %a, %a : %a [%dx%dx%d, %a -> %a, %s, threads=%d]"
        name pp_ref a pp_ref b pp_ref c Dtype.pp dtype n m k
        Dtype.pp_scalar dtype_in Dtype.pp_scalar dtype_out device threads
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

