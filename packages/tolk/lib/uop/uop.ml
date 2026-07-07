(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Types *)

type device = Single of string | Multi of string list | Index of int

module Opt = struct
  (* Variant order is load-bearing: total order over Opt.t uses
     Stdlib.compare on the constructor ordinal first. *)
  type t =
    | Tc of { axis : int; tc_select : int; tc_opt : int; use_tc : int }
    | Upcast of { axis : int; amount : int }
    | Unroll of { axis : int; amount : int }
    | Local of { axis : int; amount : int }
    | Thread of { axis : int; amount : int }
    | Group of { axis : int; amount : int }
    | Grouptop of { axis : int; amount : int }
    | Nolocals
    | Padto of { axis : int; amount : int }
    | Swap of { axis : int; with_axis : int }

  let to_string = function
    | Tc { axis; tc_select; tc_opt; use_tc } ->
        Printf.sprintf "TC:%d:%d:%d:%d" axis tc_select tc_opt use_tc
    | Upcast { axis; amount } -> Printf.sprintf "UPCAST:%d:%d" axis amount
    | Unroll { axis; amount } -> Printf.sprintf "UNROLL:%d:%d" axis amount
    | Local { axis; amount } -> Printf.sprintf "LOCAL:%d:%d" axis amount
    | Thread { axis; amount } -> Printf.sprintf "THREAD:%d:%d" axis amount
    | Group { axis; amount } -> Printf.sprintf "GROUP:%d:%d" axis amount
    | Grouptop { axis; amount } -> Printf.sprintf "GROUPTOP:%d:%d" axis amount
    | Nolocals -> "NOLOCALS"
    | Padto { axis; amount } -> Printf.sprintf "PADTO:%d:%d" axis amount
    | Swap { axis; with_axis } -> Printf.sprintf "SWAP:%d:%d" axis with_axis

  let pp fmt t = Format.pp_print_string fmt (to_string t)

  let axis = function
    | Tc { axis; _ } | Upcast { axis; _ } | Unroll { axis; _ }
    | Local { axis; _ } | Thread { axis; _ } | Group { axis; _ }
    | Grouptop { axis; _ } | Padto { axis; _ } | Swap { axis; _ } -> Some axis
    | Nolocals -> None

  let amount = function
    | Upcast { amount; _ } | Unroll { amount; _ } | Local { amount; _ }
    | Thread { amount; _ } | Group { amount; _ } | Grouptop { amount; _ }
    | Padto { amount; _ } -> Some amount
    | Tc _ | Swap _ | Nolocals -> None

  let with_amount t a = match t with
    | Upcast r -> Upcast { r with amount = a }
    | Unroll r -> Unroll { r with amount = a }
    | Local r -> Local { r with amount = a }
    | Thread r -> Thread { r with amount = a }
    | Group r -> Group { r with amount = a }
    | Grouptop r -> Grouptop { r with amount = a }
    | Padto r -> Padto { r with amount = a }
    | (Tc _ | Swap _ | Nolocals) as t -> t
end

type stage_opts = {
  device : device option;
  addrspace : Dtype.addr_space;
  removable : bool;
}

type metadata = { name : string; caller : string; backward : bool }

type param_arg = {
  slot : int;
  vmin_vmax : (int * int) option;
  name : string option;
  addrspace : Dtype.addr_space;
  axis : int option;
  device : device option;
}

type reduce_arg = { op : Ops.t; axes : int list }

type estimate = Int of int | Sym of t

and estimates = { ops : estimate; lds : estimate; mem : estimate }

and kernel_info = {
  name : string;
  axis_types : Axis_type.t list;
  dont_use_locals : bool;
  applied_opts : Opt.t list;
  opts_to_apply : Opt.t list option;
  estimates : estimates option;
  beam : int;
}

and grad_fxn = grad_output:t -> call:t -> t option list

and call_info = {
  grad_fxn : grad_fxn option;
  metadata : metadata list;
  name : string option;
  precompile : bool;
  precompile_backward : bool;
  aux : string option;
}

and launch_dim = Launch_int of int | Launch_float of float | Launch_sym of t

and launch_value = Launch_value_int of int | Launch_value_float of float

and program_info = {
  name : string;
  global_size : launch_dim list;
  local_size : int list option;
  vars : t list;
  globals : int list;
  outs : int list;
  ins : int list;
  aux : string list;
}

and wmma_info = {
  name : string;
  dims : int * int * int;
  dtype_in : Dtype.scalar;
  dtype_out : Dtype.scalar;
  device : string;
  threads : int;
  upcast_axes : (int * int) list * (int * int) list * (int * int) list;
  reduce_axes : int list;
}

and shaped_wmma_info = {
  dims : int * int * int;
  device : string;
  threads : int;
}

and arg =
  | Empty
  | Int of int
  | Ints of int list
  | Bools of bool list
  | String of string
  | Value of Const.t
  | Op of Ops.t
  | Range_info of { axis : int; sub : int list; kind : Axis_type.t }
  | Param_arg of param_arg
  | Reduce_arg of reduce_arg
  | Device of device
  | Op_device of Ops.t * device
  | Stage_info of stage_opts
  | Opts of Opt.t list
  | Kernel_info of kernel_info
  | Call_info of call_info
  | Program_info of program_info
  | Wmma_info of wmma_info
  | Shaped_wmma_info of shaped_wmma_info

and node = {
  op : Ops.t;
  dtype : Dtype.t;
  src : t array;
  arg : arg;
  node_tag : string option;
}

and t = node Hashcons.hash_consed

type realization_state =
  | Never_realized
  | Runtime_dependent of t list

type const_value =
  | Const_scalar of Dtype.storage_scalar
  | Const_invalid
  | Const_tuple of const_value list

exception Bottom_up_gate

module Ref_tbl = Hashtbl.Make (struct
  type nonrec t = t
  let equal = ( == )
  let hash u = u.Hashcons.tag
end)

(* Arg module: re-export [arg] with its constructors under the [Arg.t]
   name so callers can write [Uop.Arg.Int 5] or [match a with Uop.Arg.Empty
   -> ...]. *)
module Arg = struct
  type nonrec t = arg =
    | Empty
    | Int of int
    | Ints of int list
    | Bools of bool list
    | String of string
    | Value of Const.t
    | Op of Ops.t
    | Range_info of { axis : int; sub : int list; kind : Axis_type.t }
    | Param_arg of param_arg
    | Reduce_arg of reduce_arg
    | Device of device
    | Op_device of Ops.t * device
    | Stage_info of stage_opts
    | Opts of Opt.t list
    | Kernel_info of kernel_info
    | Call_info of call_info
    | Program_info of program_info
    | Wmma_info of wmma_info
    | Shaped_wmma_info of shaped_wmma_info

  let equal a b = match a, b with
    | Empty, Empty -> true
    | Int x, Int y -> x = y
    | Ints x, Ints y -> x = y
    | Bools x, Bools y -> x = y
    | String x, String y -> String.equal x y
    | Value x, Value y -> Const.equal x y
    | Op x, Op y -> Ops.equal x y
    | Range_info a, Range_info b ->
        a.axis = b.axis && a.sub = b.sub && Axis_type.equal a.kind b.kind
    | Param_arg x, Param_arg y -> x = y
    | Reduce_arg x, Reduce_arg y ->
        Ops.equal x.op y.op && x.axes = y.axes
    | Device x, Device y -> x = y
    | Op_device (opx, dx), Op_device (opy, dy) ->
        Ops.equal opx opy && dx = dy
    | Stage_info x, Stage_info y -> x = y
    | Opts x, Opts y -> x = y
    | Kernel_info x, Kernel_info y -> x = y
    | Call_info x, Call_info y ->
        (match x.grad_fxn, y.grad_fxn with
         | Option.None, Option.None -> true
         | Option.Some a, Option.Some b -> a == b
         | _ -> false)
        && x.metadata = y.metadata && x.name = y.name
        && x.precompile = y.precompile
        && x.precompile_backward = y.precompile_backward
        && x.aux = y.aux
    | Program_info x, Program_info y -> x = y
    | Wmma_info x, Wmma_info y -> x = y
    | Shaped_wmma_info x, Shaped_wmma_info y -> x = y
    | _ -> false

  let compare = Stdlib.compare

  let as_int = function Int n -> Option.Some n | _ -> Option.None
  let as_ints = function Ints l -> Option.Some l | _ -> Option.None
  let as_bools = function Bools l -> Option.Some l | _ -> Option.None
  let as_string = function String s -> Option.Some s | _ -> Option.None
  let as_value = function Value v -> Option.Some v | _ -> Option.None
  let as_op = function Op o -> Option.Some o | _ -> Option.None
  let as_param_arg = function Param_arg a -> Option.Some a | _ -> Option.None
  let as_reduce_arg = function
    | Reduce_arg a -> Option.Some a
    | _ -> Option.None
  let as_device = function Device d -> Option.Some d | _ -> Option.None
  let as_opts = function Opts o -> Option.Some o | _ -> Option.None
  let as_stage_info = function
    | Stage_info b -> Option.Some b
    | _ -> Option.None
  let as_program_info = function
    | Program_info p -> Option.Some p
    | _ -> Option.None
end

(* Hash-cons setup *)

module Node_hashed = struct
  type t = node

  let equal (a : node) (b : node) =
    Ops.equal a.op b.op
    && Dtype.equal a.dtype b.dtype
    && Array.length a.src = Array.length b.src
    && (let n = Array.length a.src in
        let rec check i =
          if i = n then true
          else if a.src.(i) == b.src.(i) then check (i + 1)
          else false
        in
        check 0)
    && Arg.equal a.arg b.arg
    && a.node_tag = b.node_tag

  let hash (n : node) =
    let h = ref (Hashtbl.hash n.op * 17 + Hashtbl.hash n.dtype) in
    Array.iter (fun s -> h := !h * 31 + s.Hashcons.tag) n.src;
    h := !h * 31 + Hashtbl.hash n.arg;
    h := !h * 31 + Hashtbl.hash n.node_tag;
    !h land max_int
end

module H = Hashcons.Make (Node_hashed)

let global_table = H.create 4096

let intern_node node = H.hashcons global_table node

module Side_metadata_tbl = Hashtbl.Make (struct
  type nonrec t = t
  let equal = ( == )
  let hash u = u.Hashcons.tag
end)

let side_metadata : metadata list Side_metadata_tbl.t =
  Side_metadata_tbl.create 64

let default_param_arg ?vmin_vmax ?name ?(addrspace = Dtype.Global) ?axis ?device
    slot =
  { slot; vmin_vmax; name; addrspace; axis; device }

let sanitize_function_name name =
  let len = String.length name in
  let buf = Buffer.create (max 1 len) in
  let is_name_char = function
    | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '_' -> true
    | _ -> false
  in
  let add_codepoint cp =
    if cp >= 0 && cp < 128 && is_name_char (Char.chr cp) then
      Buffer.add_char buf (Char.chr cp)
    else Buffer.add_string buf (Printf.sprintf "%02X" cp)
  in
  let byte i = Char.code name.[i] in
  let rec loop i =
    if i < len then
      match name.[i] with
      | '\027'
        when i + 1 < len && Char.equal name.[i + 1] '[' ->
          let rec skip j =
            if j >= len then len
            else
              let c = byte j in
              if c >= 0x40 && c <= 0x7E then j + 1 else skip (j + 1)
          in
          loop (skip (i + 2))
      | c ->
          let b0 = Char.code c in
          if b0 < 0x80 then (add_codepoint b0; loop (i + 1))
          else if b0 land 0xE0 = 0xC0 && i + 1 < len then (
            let cp = ((b0 land 0x1F) lsl 6) lor (byte (i + 1) land 0x3F) in
            add_codepoint cp;
            loop (i + 2))
          else if b0 land 0xF0 = 0xE0 && i + 2 < len then (
            let cp =
              ((b0 land 0x0F) lsl 12)
              lor ((byte (i + 1) land 0x3F) lsl 6)
              lor (byte (i + 2) land 0x3F)
            in
            add_codepoint cp;
            loop (i + 3))
          else if b0 land 0xF8 = 0xF0 && i + 3 < len then (
            let cp =
              ((b0 land 0x07) lsl 18)
              lor ((byte (i + 1) land 0x3F) lsl 12)
              lor ((byte (i + 2) land 0x3F) lsl 6)
              lor (byte (i + 3) land 0x3F)
            in
            add_codepoint cp;
            loop (i + 4))
          else (add_codepoint b0; loop (i + 1))
  in
  loop 0;
  Buffer.contents buf

let kernel_function_name (info : kernel_info) = sanitize_function_name info.name
let program_function_name (info : program_info) = sanitize_function_name info.name

(* Accessors *)

let op u = u.Hashcons.node.op
let dtype u = u.Hashcons.node.dtype
let src u = u.Hashcons.node.src
let arg u = u.Hashcons.node.arg
let node_tag u = u.Hashcons.node.node_tag
let tag u = u.Hashcons.tag
let metadata u = Option.value (Side_metadata_tbl.find_opt side_metadata u) ~default:[]

let with_metadata md u =
  Side_metadata_tbl.replace side_metadata u md;
  u

let children u = Array.to_list (src u)
let equal a b = a.Hashcons.tag = b.Hashcons.tag
let compare a b = Int.compare a.Hashcons.tag b.Hashcons.tag

let int64_as_native n =
  if Int64.compare n (Int64.of_int min_int) >= 0
     && Int64.compare n (Int64.of_int max_int) <= 0
  then Option.Some (Int64.to_int n)
  else Option.None

let saturate_int64 n =
  if Int64.compare n (Int64.of_int min_int) < 0 then min_int
  else if Int64.compare n (Int64.of_int max_int) > 0 then max_int
  else Int64.to_int n

let const_int_bounds dtype n =
  match dtype with
  | Dtype.Val v
    when Dtype.Val.is_unsigned v && Int64.compare n 0L < 0 ->
      max_int, max_int
  | _ ->
      let n = saturate_int64 n in
      n, n

let program_var_name u =
  match op u, arg u with
  | Ops.Param, Arg.Param_arg { name = Some name; _ } -> Some name
  | _ -> None

let program_runtimevars (info : program_info) =
  let rec loop i acc = function
    | [] -> List.rev acc
    | var :: vars ->
        let acc =
          match program_var_name var with
          | Some "core_id" -> ("core_id", i) :: acc
          | Some _ | None -> acc
        in
        loop (i + 1) acc vars
  in
  loop 0 [] info.vars

(* View accessors — structured views over per-op src/arg contracts. *)

type index_view = { ptr : t; idxs : t list }
type load_view = { src : t; alt : t option; gate : t option }
type store_view = { dst : t; value : t; gate : t option }
type range_view = {
  size : t;
  parents : t list;
  axis : int;
  sub : int list;
  kind : Axis_type.t;
}
type end_view = { value : t; ranges : t list }
type if_view = { cond : t; idx_for_dedup : t }
type reduce_view = { src : t; ranges : t list; op : Ops.t; axes : int list }
type allreduce_view = { src : t; device : device; op : Ops.t }
type stage_view = { src : t; ranges : t list; opts : stage_opts }
type slice_view = { src : t; offset : t; size : int }
type wait_view = { src : t; wait_for : t }
type param_view = { param : param_arg; shape : t }
type buffer_view = { buffer : param_arg; shape : t }
type wmma_view = { a : t; b : t; c : t; info : wmma_info }
type shaped_wmma_view = { a : t; b : t; acc : t; info : shaped_wmma_info }
type call_view = { body : t; args : t list; info : call_info }
type special_view = { name : string; size : t }
type bind_view = { var : t; value : t }
type marg =
  | Marg_shape of t list
  | Marg_bounds of (t * t) list
  | Marg_permute of int list
  | Marg_flip of bool list

let as_index u =
  match op u, Array.to_list (src u) with
  | Ops.Index, ptr :: idxs ->
      Option.Some { ptr; idxs }
  | _ -> Option.None

let as_load u =
  match op u, Array.to_list (src u) with
  | Ops.Load, [ src ] ->
      Option.Some { src; alt = Option.None; gate = Option.None }
  | Ops.Load, [ src; alt; gate ] ->
      Option.Some { src; alt = Option.Some alt; gate = Option.Some gate }
  | _ -> Option.None

let as_store u =
  match op u, Array.to_list (src u) with
  | Ops.Store, [ dst; value ] ->
      Option.Some { dst; value; gate = Option.None }
  | Ops.Store, [ dst; value; gate ] ->
      Option.Some { dst; value; gate = Option.Some gate }
  | _ -> Option.None

let as_range u =
  match op u, arg u, Array.to_list (src u) with
  | Ops.Range, Arg.Range_info { axis; sub; kind }, size :: parents ->
      Option.Some { size; parents; axis; sub; kind }
  | _ -> Option.None

let as_end u =
  match op u, Array.to_list (src u) with
  | Ops.End, value :: ranges -> Option.Some { value; ranges }
  | _ -> Option.None

let as_if u =
  match op u, Array.to_list (src u) with
  | Ops.If, [ cond; idx_for_dedup ] ->
      Option.Some { cond; idx_for_dedup }
  | _ -> Option.None

let as_reduce u =
  match op u, arg u, Array.to_list (src u) with
  | Ops.Reduce, Arg.Reduce_arg { op; axes }, src :: ranges ->
      Option.Some { src; ranges; op; axes }
  | _ -> Option.None

let as_allreduce u =
  match op u, arg u, Array.to_list (src u) with
  | Ops.Allreduce, Arg.Op_device (op, device), [ src ] ->
      Option.Some { src; device; op }
  | _ -> Option.None

let as_stage u =
  match op u, arg u, Array.to_list (src u) with
  | Ops.Stage, Arg.Stage_info opts, src :: ranges ->
      Option.Some { src; ranges; opts }
  | _ -> Option.None

let as_slice u =
  match op u, arg u, Array.to_list (src u) with
  | Ops.Slice, Arg.Int size, [ src; offset ] ->
      Option.Some { src; offset; size }
  | _ -> Option.None

let as_wait u =
  match op u, Array.to_list (src u) with
  | Ops.Wait, [ src; wait_for ] -> Option.Some { src; wait_for }
  | _ -> Option.None

let as_param u =
  match op u, arg u, Array.to_list (src u) with
  | Ops.Param, Arg.Param_arg arg, [ shape ] ->
      Option.Some { param = arg; shape }
  | _ -> Option.None

let as_buffer u =
  match op u, arg u, Array.to_list (src u) with
  | Ops.Buffer, Arg.Param_arg arg, [ shape ] ->
      Option.Some { buffer = arg; shape }
  | _ -> Option.None

let as_wmma u =
  match op u, arg u, Array.to_list (src u) with
  | Ops.Wmma, Arg.Wmma_info info, [ a; b; c ] ->
      Option.Some { a; b; c; info }
  | _ -> Option.None

let as_shaped_wmma u =
  match op u, arg u, Array.to_list (src u) with
  | Ops.Shaped_wmma, Arg.Shaped_wmma_info info, [ a; b; acc ] ->
      Option.Some { a; b; acc; info }
  | _ -> Option.None

let as_call u =
  match op u, arg u, Array.to_list (src u) with
  | (Ops.Call | Ops.Function), Arg.Call_info info, body :: args ->
      Option.Some { body; args; info }
  | _ -> Option.None

let as_special u =
  match op u, arg u, Array.to_list (src u) with
  | Ops.Special, Arg.String name, [ size ] -> Some { name; size }
  | _ -> Option.None

let as_bind u =
  match op u, Array.to_list (src u) with
  | Ops.Bind, [ var; value ] -> Option.Some { var; value }
  | _ -> Option.None

let device_cache : device option Ref_tbl.t = Ref_tbl.create 32

let rec device_of u =
  match Ref_tbl.find_opt device_cache u with
  | Some d -> d
  | None ->
      let d = compute_device u in
      Ref_tbl.add device_cache u d;
      d

and compute_device u =
  let children = src u in
  match op u with
  | Ops.Stage ->
      (match Arg.as_stage_info (arg u) with
       | Some opts -> opts.device
       | None -> None)
  | Ops.After when Array.length children >= 1 -> device_of children.(0)
  | Ops.Mselect when Array.length children >= 1 ->
      (match device_of children.(0), Arg.as_int (arg u) with
       | Some (Multi devs), Some i when i >= 0 && i < List.length devs ->
           Some (Single (List.nth devs i))
       | _ -> None)
  | Ops.Mstack ->
      let per = Array.map device_of children in
      let all_single =
        Array.for_all
          (function
            | Some (Single _) -> true
            | None | Some (Multi _) | Some (Index _) -> false)
          per
      in
      if (not all_single) || Array.length per = 0 then None
      else
        let names =
          Array.map
            (function
              | Some (Single s) -> s
              | None | Some (Multi _) | Some (Index _) -> assert false)
            per
        in
        Some (Multi (Array.to_list names))
  | Ops.Param | Ops.Buffer ->
      (match Arg.as_param_arg (arg u) with
       | Some param -> param.device
       | None -> None)
  | Ops.Copy ->
      (match Arg.as_device (arg u) with Some d -> Some d | None -> None)
  | Ops.Allreduce ->
      (match arg u with
       | Arg.Op_device (_, device) -> Some device
       | _ -> None)
  | _ ->
      let found = ref None in
      Array.iter
        (fun c ->
          if !found = None then
            match device_of c with
            | Some d -> found := Some d
            | None -> ())
        children;
      !found

let as_contiguous_opts u =
  match op u, arg u with
  | Ops.Contiguous, Arg.Opts o -> Option.Some o
  | _ -> Option.None

let as_kernel_info u =
  match op u, arg u with
  | Ops.Sink, Arg.Kernel_info ki -> Option.Some ki
  | _ -> Option.None

let as_call_info u =
  match op u, arg u with
  | (Ops.Call | Ops.Function), Arg.Call_info info -> Option.Some info
  | _ -> Option.None

let as_program_info u =
  match op u, arg u with
  | Ops.Program, Arg.Program_info info -> Option.Some info
  | _ -> Option.None

(* Raw constructor *)

let mk ~op ~dtype ~src ~arg =
  intern_node { op; dtype; src; arg; node_tag = Option.None }

let mk_tagged ~op ~dtype ~src ~arg ~node_tag =
  intern_node { op; dtype; src; arg; node_tag }

(* Smart constructors *)

let void_dtype = Dtype.Val Dtype.Val.void

let sink ?kernel_info srcs =
  let arg = match kernel_info with
    | Option.None -> Arg.Empty
    | Option.Some ki -> Arg.Kernel_info ki
  in
  mk ~op:Ops.Sink ~dtype:void_dtype ~src:(Array.of_list srcs) ~arg

let group = function
  | [ s ] -> s
  | srcs ->
      mk ~op:Ops.Group ~dtype:void_dtype
        ~src:(Array.of_list srcs) ~arg:Arg.Empty

let after ~src:s ~deps =
  if deps = [] then s
  else
    mk ~op:Ops.After ~dtype:(dtype s)
      ~src:(Array.of_list (s :: deps)) ~arg:Arg.Empty

let noop ?src ~dtype () =
  let srcs = match src with Option.None -> [||] | Option.Some s -> [| s |] in
  mk ~op:Ops.Noop ~dtype ~src:srcs ~arg:Arg.Empty

let shape_to_shape_arg = function
  | Option.None -> noop ~dtype:void_dtype ()
  | Option.Some shape -> shape

let linear srcs =
  mk ~op:Ops.Linear ~dtype:void_dtype
    ~src:(Array.of_list srcs) ~arg:Arg.Empty

let param ~slot ~dtype ?shape ?device ?vmin_vmax ?name ?addrspace ?axis () =
  let shape = shape_to_shape_arg shape in
  mk ~op:Ops.Param ~dtype ~src:[| shape |]
    ~arg:(Arg.Param_arg
            (default_param_arg ?vmin_vmax ?name ?addrspace ?axis ?device slot))

let buffer ~slot ~dtype ?shape ?name ?addrspace ?axis ?device () =
  let shape = shape_to_shape_arg shape in
  mk ~op:Ops.Buffer ~dtype ~src:[| shape |]
    ~arg:(Arg.Param_arg
            (default_param_arg ?name ?addrspace ?axis ?device slot))

let stage ~src ~ranges ~opts =
  mk ~op:Ops.Stage ~dtype:(dtype src)
    ~src:(Array.of_list (src :: ranges))
    ~arg:(Arg.Stage_info opts)

let variable ~name ~min_val ~max_val ?(dtype = Dtype.Val.weakint) () =
  let count = Dtype.Val.count dtype in
  let shape =
    if count > 1 then
      mk ~op:Ops.Const ~dtype:(Dtype.Val Dtype.Val.weakint) ~src:[||]
        ~arg:(Arg.Value (Const.int Dtype.Val.weakint count))
    else mk ~op:Ops.Stack ~dtype:void_dtype ~src:[||] ~arg:Arg.Empty
  in
  param ~slot:(-1) ~dtype:(Dtype.Val dtype) ~name
    ~shape ~vmin_vmax:(min_val, max_val)
    ~addrspace:Dtype.Alu ()

let bind ~var ~value =
  let in_bounds = match arg var, op value, arg value with
    | Arg.Param_arg { vmin_vmax = Some (lo, hi); _ }, Ops.Const, Arg.Value c ->
        (match Const.view c with
         | Const.Int n ->
             (match int64_as_native n with
              | Some n -> lo <= n && n <= hi
              | None -> false)
         | Const.Bool b ->
             let n = if b then 1 else 0 in
             lo <= n && n <= hi
         | Const.Float _ | Const.Invalid -> true)
    | _ -> true
  in
  if in_bounds then
    mk ~op:Ops.Bind ~dtype:(dtype var) ~src:[| var; value |] ~arg:Arg.Empty
  else invalid_arg "Uop.bind: value outside variable bounds"

let const ?(srcs = []) v =
  let dtype = Dtype.Val (Const.dtype v) in
  mk ~op:Ops.Const ~dtype ~src:(Array.of_list srcs) ~arg:(Arg.Value v)

let invalid ?(dtype = Dtype.Val.weakint) () =
  const (Const.invalid ~dtype ())

let const_int n = const (Const.int Dtype.Val.weakint n)
let const_float x = const (Const.float Dtype.Val.float32 x)
let const_bool b = const (Const.bool b)

let zero_like u =
  let dt = dtype u in
  match dt with
  | Dtype.Val v -> const (Const.of_scalar v (`Int 0L))
  | Dtype.Ptr _ -> invalid_arg "Uop.zero_like: pointer has no zero"

let const_like u n =
  match dtype u with
  | Dtype.Val v -> const (Const.of_scalar v (`Int (Int64.of_int n)))
  | Dtype.Ptr _ -> invalid_arg "Uop.const_like: pointer has no const"

let index ~ptr ~idxs ?(as_ptr = false) () =
  let const_int_value u =
    match op u, arg u with
    | Ops.Const, Arg.Value c -> (
        match Const.view c with
        | Const.Int n
          when Int64.compare n (Int64.of_int min_int) >= 0
               && Int64.compare n (Int64.of_int max_int) <= 0 ->
            Some (Int64.to_int n)
        | Const.Int _ | Const.Bool _ | Const.Float _ | Const.Invalid -> None)
    | _ -> None
  in
  match as_ptr, idxs, op ptr with
  | false, [ idx ], Ops.Stack -> (
      match const_int_value idx with
      | Some i ->
          let src = src ptr in
          if i >= 0 && i < Array.length src then src.(i)
          else
            mk ~op:Ops.Index
              ~dtype:(Dtype.Val (Dtype.Val.scalarize (Dtype.val_of (dtype ptr))))
              ~src:[| ptr; idx |] ~arg:Arg.Empty
      | None ->
          let dtype =
            Dtype.Val (Dtype.Val.scalarize (Dtype.val_of (dtype ptr)))
          in
          mk ~op:Ops.Index ~dtype ~src:[| ptr; idx |] ~arg:Arg.Empty)
  | _ ->
  let dtype =
    if as_ptr then dtype ptr
    else match dtype ptr with
      | Dtype.Ptr p -> Dtype.Val (Dtype.Ptr.value p)
      | Dtype.Val _ when op ptr = Ops.Stage -> dtype ptr
      | Dtype.Val v -> Dtype.Val (Dtype.Val.scalarize v)
  in
  mk ~op:Ops.Index ~dtype ~src:(Array.of_list (ptr :: idxs)) ~arg:Arg.Empty

let load ~src ?dtype:load_dtype ?alt ?gate () =
  let dtype =
    match load_dtype with
    | Some dtype -> dtype
    | None ->
        match dtype src with
        | Dtype.Ptr p -> Dtype.Val (Dtype.Ptr.value p)
        | Dtype.Val _ -> invalid_arg "Uop.load: expected pointer src"
  in
  let srcs = match alt, gate with
    | Option.None, Option.None -> [| src |]
    | Option.Some a, Option.Some g -> [| src; a; g |]
    | Option.None, Option.Some _ ->
        invalid_arg "Uop.load: gate requires alt"
    | Option.Some _, Option.None ->
        invalid_arg "Uop.load: alt requires gate"
  in
  mk ~op:Ops.Load ~dtype ~src:srcs ~arg:Arg.Empty

let store ~dst ~value ?gate () =
  let src =
    match gate with
    | Option.None -> [| dst; value |]
    | Option.Some g -> [| dst; value; g |]
  in
  mk ~op:Ops.Store ~dtype:void_dtype
    ~src ~arg:Arg.Empty

let alu_unary ~op ~src =
  if not (Ops.Group.is_unary op) then
    invalid_arg
      (Printf.sprintf "Uop.alu_unary: %s is not unary" (Ops.name op));
  mk ~op ~dtype:(dtype src) ~src:[| src |] ~arg:Arg.Empty

let alu_binary ~op ~lhs ~rhs =
  if not (Ops.Group.is_binary op) then
    invalid_arg
      (Printf.sprintf "Uop.alu_binary: %s is not binary" (Ops.name op));
  let dt =
    if Ops.Group.is_comparison op then
      match dtype lhs with
      | Dtype.Val v ->
          Dtype.Val (if Dtype.Val.count v = 1 then Dtype.Val.bool
                      else Dtype.Val.vec (Dtype.Val.count v) Dtype.Val.bool)
      | Dtype.Ptr _ -> invalid_arg "Uop.alu_binary: comparison on pointer"
    else dtype lhs
  in
  mk ~op ~dtype:dt ~src:[| lhs; rhs |] ~arg:Arg.Empty

let alu_ternary ~op ~a ~b ~c =
  if not (Ops.Group.is_ternary op) then
    invalid_arg
      (Printf.sprintf "Uop.alu_ternary: %s is not ternary" (Ops.name op));
  let dt = match op with
    | Ops.Where -> dtype b
    | Ops.Mulacc -> dtype a
    | _ -> dtype a
  in
  mk ~op ~dtype:dt ~src:[| a; b; c |] ~arg:Arg.Empty

let cast ~src ~dtype:target_dtype =
  let src_count = Dtype.count (dtype src) in
  let adjusted_dtype =
    if Dtype.count target_dtype = 1 && src_count <> 1 then
      Dtype.vec src_count target_dtype
    else target_dtype
  in
  if Dtype.equal (dtype src) adjusted_dtype then src
  else mk ~op:Ops.Cast ~dtype:adjusted_dtype ~src:[| src |] ~arg:Arg.Empty

let bitcast ~src ~dtype:target_dtype =
  if Dtype.equal (dtype src) target_dtype then src
  else mk ~op:Ops.Bitcast ~dtype:target_dtype ~src:[| src |] ~arg:Arg.Empty

let stack ?dtype:dtype_opt srcs =
  let dt = match dtype_opt, srcs with
    | Option.Some dt, _ -> Dtype.Val (Dtype.Val.scalarize dt)
    | Option.None, [] ->
        void_dtype
    | Option.None, first :: _ ->
        (match dtype first with
         | Dtype.Val _ as dt -> dt
         | Dtype.Ptr _ as dt -> dt)
  in
  mk ~op:Ops.Stack ~dtype:dt ~src:(Array.of_list srcs) ~arg:Arg.Empty

let slice ~src ~offset ~size ~dtype =
  mk ~op:Ops.Slice ~dtype ~src:[| src; offset |] ~arg:(Arg.Int size)

let getaddr ~src = mk ~op:Ops.Getaddr ~dtype:(dtype src) ~src:[| src |] ~arg:Arg.Empty

let broadcast u n =
  if n <= 1 then u
  else
    let srcs = List.init n (fun _ -> u) in
    stack srcs

let range ~size ~axis ~kind ?(sub = []) ?(dtype = Dtype.Val.weakint)
    ?(parents = []) () =
  mk ~op:Ops.Range ~dtype:(Dtype.Val dtype)
    ~src:(Array.of_list (size :: parents))
    ~arg:(Arg.Range_info { axis; sub; kind })

let end_ ~value ~ranges =
  if ranges = [] then value
  else
    mk ~op:Ops.End ~dtype:(dtype value)
      ~src:(Array.of_list (value :: ranges)) ~arg:Arg.Empty

let if_ ~cond ~idx_for_dedup =
  mk ~op:Ops.If ~dtype:void_dtype
    ~src:[| cond; idx_for_dedup |] ~arg:Arg.Empty

let endif ~if_ =
  mk ~op:Ops.Endif ~dtype:void_dtype ~src:[| if_ |] ~arg:Arg.Empty

let barrier ?(srcs = []) () =
  mk ~op:Ops.Barrier ~dtype:void_dtype
    ~src:(Array.of_list srcs) ~arg:Arg.Empty

let wait ~src ~wait_for =
  mk ~op:Ops.Wait ~dtype:void_dtype
    ~src:[| src; wait_for |] ~arg:Arg.Empty

let special ~name ~size ?(dtype = Dtype.Val.weakint) () =
  let node_dtype = Dtype.Val dtype in
  mk ~op:Ops.Special ~dtype:node_dtype
    ~src:[| cast ~src:size ~dtype:node_dtype |] ~arg:(Arg.String name)

let reduce ~src ~ranges ~op ~dtype =
  mk ~op:Ops.Reduce ~dtype:(Dtype.Val dtype)
    ~src:(Array.of_list (src :: ranges))
    ~arg:(Arg.Reduce_arg { op; axes = [] })

let reduce_axis ~src ~op ~axes =
  match axes with
  | [] -> src
  | _ ->
      mk ~op:Ops.Reduce ~dtype:(dtype src) ~src:[| src |]
        ~arg:(Arg.Reduce_arg { op; axes })

let allreduce ~src ~device ~op =
  mk ~op:Ops.Allreduce ~dtype:(dtype src)
    ~src:[| src |] ~arg:(Arg.Op_device (op, device))

let multi ~src ~axis =
  mk ~op:Ops.Multi ~dtype:(dtype src) ~src:[| src |] ~arg:(Arg.Int axis)

let mstack srcs =
  let dt = match srcs with
    | s :: _ -> dtype s
    | [] -> invalid_arg "Uop.mstack: empty"
  in
  mk ~op:Ops.Mstack ~dtype:dt ~src:(Array.of_list srcs) ~arg:Arg.Empty

let mselect ~src ~index =
  mk ~op:Ops.Mselect ~dtype:(dtype src) ~src:[| src |] ~arg:(Arg.Int index)

let copy ~src ~device () =
  mk ~op:Ops.Copy ~dtype:(dtype src) ~src:[| src |] ~arg:(Arg.Device device)

let rec base u =
  let srcs = src u in
  match op u with
  | op when Ops.Group.is_movement op ->
      if Array.length srcs = 0 then u else base srcs.(0)
  | Ops.Multi | Ops.Detach ->
      if Array.length srcs = 0 then u else base srcs.(0)
  | _ -> u

let rec addrspace u =
  let srcs = src u in
  match op u with
  | Ops.Param ->
      (match Arg.as_param_arg (arg u) with
       | Some param -> Some param.addrspace
       | None -> None)
  | Ops.Buffer ->
      (match Arg.as_param_arg (arg u) with
       | Some buffer -> Some buffer.addrspace
       | None -> None)
  | Ops.Special | Ops.Range -> Some Dtype.Alu
  | Ops.Load -> Some Dtype.Alu
  | Ops.Index | Ops.Cast | Ops.After | Ops.Reduce | Ops.Store | Ops.Mstack
  | Ops.Mselect ->
      if Array.length srcs = 0 then None else addrspace srcs.(0)
  | op when Ops.Group.is_movement op ->
      if Array.length srcs = 0 then None else addrspace srcs.(0)
  | Ops.Stack | Ops.Wmma ->
      let spaces =
        Array.fold_left
          (fun acc s ->
            match addrspace s with None -> acc | Some a -> a :: acc)
          [] srcs
      in
      (match spaces with
       | [] -> None
       | first :: rest ->
           if List.for_all (( = ) first) rest then Some first else None)
  | op when Ops.Group.is_elementwise op ->
      let spaces =
        Array.fold_left
          (fun acc s ->
            match addrspace s with None -> acc | Some a -> a :: acc)
          [] srcs
      in
      (match spaces with
       | [] -> None
       | first :: rest ->
           if List.for_all (( = ) first) rest then Some first else None)
  | _ -> None

let rec buf_uop u =
  let srcs = src u in
  match op u with
  | Ops.Buffer | Ops.Param -> u
  | Ops.Mselect -> (
      match Array.to_list srcs, Arg.as_int (arg u) with
      | [ src ], Some index -> mselect ~src:(buf_uop src) ~index
      | _ -> u)
  | Ops.Mstack -> mstack (List.map buf_uop (Array.to_list srcs))
  | _ ->
      let b = base u in
      if op b = Ops.After && Array.length (src b) > 0 then
        base (buf_uop (src b).(0))
      else
        let rec walk s =
          let srcs = src s in
          match op s with
          | Ops.Buffer | Ops.Param | Ops.Stage | Ops.Mstack -> s
          | _ -> if Array.length srcs = 0 then s else walk srcs.(0)
        in
        walk u

let rec has_buffer_identity u =
  let srcs = src u in
  match op u with
  | Ops.Reshape | Ops.Multi ->
      Array.length srcs > 0 && has_buffer_identity srcs.(0)
  | Ops.Gettuple -> (
      match Arg.as_int (arg u), Array.to_list srcs with
      | Some i, [ tuple ] when op tuple = Ops.Tuple ->
          let tuple_srcs = src tuple in
          i >= 0
          && i < Array.length tuple_srcs
          && has_buffer_identity tuple_srcs.(i)
      | _ -> false)
  | Ops.Buffer | Ops.Slice | Ops.Param -> true
  | _ -> false

let reshape ~src ~shape =
  mk ~op:Ops.Reshape ~dtype:(dtype src) ~src:[| src; shape |] ~arg:Arg.Empty

let expand ~src ~shape =
  mk ~op:Ops.Expand ~dtype:(dtype src) ~src:[| src; shape |] ~arg:Arg.Empty

let pad ~src ~offset ~size =
  mk ~op:Ops.Pad ~dtype:(dtype src)
    ~src:[| src; offset; size |] ~arg:Arg.Empty

let shrink ~src ~offset ~size =
  mk ~op:Ops.Shrink ~dtype:(dtype src)
    ~src:[| src; offset; size |] ~arg:Arg.Empty

let permute ~src ~order =
  mk ~op:Ops.Permute ~dtype:(dtype src) ~src:[| src |]
    ~arg:(Arg.Ints order)

let flip ~src ~dims =
  mk ~op:Ops.Flip ~dtype:(dtype src) ~src:[| src |]
    ~arg:(Arg.Bools dims)

let detach ~src =
  mk ~op:Ops.Detach ~dtype:(dtype src) ~src:[| src |] ~arg:Arg.Empty

let contiguous ~src ?(ranges = []) ?(force = false) () =
  match (force, ranges, op src, device_of src) with
  | false, _, Ops.Contiguous, _ -> src
  | false, [], _, None -> src
  | false, [], _, _ when has_buffer_identity src -> src
  | _ ->
      mk ~op:Ops.Contiguous ~dtype:(dtype src)
        ~src:(Array.of_list (src :: ranges)) ~arg:Arg.Empty

let contiguous_backward ~src =
  mk ~op:Ops.Contiguous_backward ~dtype:(dtype src) ~src:[| src |]
    ~arg:Arg.Empty

let tuple srcs =
  mk ~op:Ops.Tuple ~dtype:void_dtype
    ~src:(Array.of_list srcs) ~arg:Arg.Empty

let gettuple ~src ~index =
  let dt = match op src with
    | Ops.Tuple ->
        let arr = Array.of_list (children src) in
        if index < 0 || index >= Array.length arr then
          invalid_arg "Uop.gettuple: index out of range";
        dtype arr.(index)
    | Ops.Function ->
        let body = (Array.of_list (children src)).(0) in
        let arr = Array.of_list (children body) in
        if index < 0 || index >= Array.length arr then
          invalid_arg "Uop.gettuple: index out of range";
        dtype arr.(index)
    | _ -> invalid_arg "Uop.gettuple: src must be Tuple or Function"
  in
  mk ~op:Ops.Gettuple ~dtype:dt ~src:[| src |] ~arg:(Arg.Int index)

let program ~sink ?linear ?source ?binary ~info () =
  let srcs =
    match linear, source, binary with
    | None, None, None -> [ sink ]
    | Some linear, None, None -> [ sink; linear ]
    | Some linear, Some source, None -> [ sink; linear; source ]
    | Some linear, Some source, Some binary -> [ sink; linear; source; binary ]
    | None, Some _, _ | None, None, Some _ | Some _, None, Some _ ->
        invalid_arg "Uop.program: optional sources must form a prefix"
  in
  mk ~op:Ops.Program ~dtype:void_dtype
    ~src:(Array.of_list srcs) ~arg:(Arg.Program_info info)

let set ~target ~value ?(extras = []) () =
  let st = store ~dst:target ~value () in
  after ~src:target ~deps:(st :: extras)

let shaped_wmma ~a ~b ~acc ~dims ~device ~threads ~dtype =
  mk ~op:Ops.Shaped_wmma ~dtype ~src:[| a; b; acc |]
    ~arg:(Arg.Shaped_wmma_info { dims; device; threads })

let wmma ~a ~b ~c ~info ~dtype =
  mk ~op:Ops.Wmma ~dtype:(Dtype.Val dtype) ~src:[| a; b; c |]
    ~arg:(Arg.Wmma_info info)

let custom ~fmt ~args =
  mk ~op:Ops.Custom ~dtype:void_dtype
    ~src:(Array.of_list args) ~arg:(Arg.String fmt)

let custom_inline ~fmt ~args ~dtype =
  mk ~op:Ops.Customi ~dtype:(Dtype.Val dtype)
    ~src:(Array.of_list args) ~arg:(Arg.String fmt)

let source s =
  mk ~op:Ops.Source ~dtype:void_dtype ~src:[||] ~arg:(Arg.String s)

let binary s =
  mk ~op:Ops.Binary ~dtype:void_dtype ~src:[||] ~arg:(Arg.String s)

let rewrite_error ~src ~msg =
  mk ~op:Ops.Rewrite_error ~dtype:void_dtype ~src ~arg:(Arg.String msg)

let ins ~mnemonic ~operands ?(dtype = void_dtype) () =
  mk ~op:Ops.Ins ~dtype
    ~src:(Array.of_list operands) ~arg:(Arg.String mnemonic)

let custom_function ~name ~srcs =
  mk ~op:Ops.Custom_function ~dtype:void_dtype
    ~src:(Array.of_list srcs) ~arg:(Arg.String name)

(* Replace / with_tag *)

let replace u ?op:op_opt ?src:src_opt ?arg:arg_opt ?dtype:dtype_opt
    ?node_tag:node_tag_opt () =
  let n : node = u.Hashcons.node in
  let op = Option.value op_opt ~default:n.op in
  let src = Option.value src_opt ~default:n.src in
  let arg = Option.value arg_opt ~default:n.arg in
  let dtype = Option.value dtype_opt ~default:n.dtype in
  let node_tag = Option.value node_tag_opt ~default:n.node_tag in
  intern_node { op; src; arg; dtype; node_tag }

let with_tag s u =
  intern_node { u.Hashcons.node with node_tag = Option.Some s }

(* Traversal *)

module Tbl = Hashtbl.Make (struct
  type nonrec t = t
  let equal = equal
  let hash u = u.Hashcons.tag
end)

let toposort ?(gate = fun _ -> true) ?(enter_calls = true) root =
  let done_ = Ref_tbl.create 64 in
  let order = ref [] in
  let stack = Stack.create () in
  Stack.push (root, false) stack;
  while not (Stack.is_empty stack) do
    let node, visited = Stack.pop stack in
    if Ref_tbl.mem done_ node then ()
    else if not visited then begin
      if gate node then begin
        Stack.push (node, true) stack;
        let srcs = src node in
        let enter =
          enter_calls
          || (match op node with
              | Ops.Call | Ops.Function -> false
              | _ -> true)
        in
        if enter then
          for i = Array.length srcs - 1 downto 0 do
            Stack.push (srcs.(i), false) stack
          done
        else
          for i = Array.length srcs - 1 downto 1 do
            Stack.push (srcs.(i), false) stack
          done
      end
    end else begin
      Ref_tbl.replace done_ node ();
      order := node :: !order
    end
  done;
  List.rev !order

let backward_slice root =
  List.filter (fun u -> not (u == root)) (toposort root)

let find_nodes p root =
  List.filter p (toposort root)

let in_backward_slice needle haystack =
  List.exists (fun u -> u == needle) (backward_slice haystack)

let is_scratch_buffer u =
  match op u, Arg.as_param_arg (arg u) with
  | Ops.Buffer, Some { addrspace = Dtype.Local | Dtype.Reg; _ } -> true
  | _ -> false

let runtime_realization_state u =
  let root = base u in
  match op root with
  | Ops.Buffer ->
      if is_scratch_buffer root then Never_realized
      else Runtime_dependent [ root ]
  | Ops.Mstack ->
      let buffers =
        List.filter (fun node -> Ops.equal (op node) Ops.Buffer)
          (toposort root)
      in
      if buffers = [] || List.exists is_scratch_buffer buffers then
        Never_realized
      else Runtime_dependent buffers
  | _ -> Never_realized

(* Index into [src] where the ranges that this op "closes" begin.
   A child at or after this index is an ended range and does not
   propagate into this node's in-scope range set. *)
let range_start_idx = function
  | Ops.Stage | Ops.Reduce | Ops.End | Ops.Call | Ops.Function
  | Ops.Copy -> Option.Some 1
  | Ops.Slice -> Option.Some 2
  | Ops.Wmma -> Option.Some 3
  | _ -> Option.None

module Ref_set = Set.Make (struct
  type nonrec t = t
  let compare a b = Int.compare a.Hashcons.tag b.Hashcons.tag
end)

let ranges_cache : Ref_set.t Ref_tbl.t = Ref_tbl.create 64

let rec ranges_set u =
  match Ref_tbl.find_opt ranges_cache u with
  | Some set -> set
  | None ->
      let set = compute_ranges u in
      Ref_tbl.add ranges_cache u set;
      set

and compute_ranges u =
  let acc = ref Ref_set.empty in
  let children = src u in
  Array.iter (fun c -> acc := Ref_set.union !acc (ranges_set c)) children;
  List.iter (fun ended ->
    if op ended = Ops.Range then acc := Ref_set.remove ended !acc
    else
      Ref_set.iter (fun r -> acc := Ref_set.remove r !acc)
        (ranges_set ended))
    (ended_ranges u);
  (if op u = Ops.Range then acc := Ref_set.add u !acc);
  !acc

and ended_ranges u =
  let children = src u in
  match op u with
  | Ops.After ->
      let ret = ref [] in
      for i = 1 to Array.length children - 1 do
        ret := List.rev_append (ended_ranges children.(i)) !ret
      done;
      List.rev !ret
  | _ ->
      (match range_start_idx (op u) with
       | Option.None -> []
       | Option.Some k ->
           Array.to_list (Array.sub children k (Array.length children - k)))

let ranges u =
  let mem_ref x xs = List.exists (fun y -> y == x) xs in
  let add_unique_rev acc r = if mem_ref r acc then acc else r :: acc in
  let remove_many acc rs =
    List.filter (fun r -> not (mem_ref r rs)) acc
  in
  let rec ranges_list u =
    let children = src u in
    let acc = ref [] in
    Array.iter
      (fun c ->
         List.iter (fun r -> acc := add_unique_rev !acc r) (ranges_list c))
      children;
    List.iter
      (fun ended ->
         if op ended = Ops.Range then acc := remove_many !acc [ ended ]
         else acc := remove_many !acc (ranges_list ended))
      (ended_ranges u);
    let ordered = List.rev !acc in
    if op u = Ops.Range then u :: remove_many ordered [ u ] else ordered
  in
  ranges_list u

let ranges_subset sub sup =
  let sup_set = ranges_set sup in
  List.for_all (fun r -> Ref_set.mem r sup_set) (ranges sub)

let opaque_call_body = function
  | Ops.Sink | Ops.Program | Ops.Linear | Ops.Copy | Ops.Slice
  | Ops.Custom_function -> true
  | _ -> false

let call ~body ~args ~info =
  if ranges body <> [] then
    invalid_arg "Uop.call: ranges are leaking out of the call body";
  let op, body =
    if opaque_call_body (op body) then Ops.Call, body
    else
      let body = if op body = Ops.Tuple then body else tuple [ body ] in
      Ops.Function, body
  in
  mk ~op ~dtype:void_dtype
    ~src:(Array.of_list (body :: args))
    ~arg:(Arg.Call_info info)

(* Rewriting *)

let first_match rules n = List.find_map (fun r -> r n) rules

let graph_rewrite ?loc ?(name = "") ?(enter_calls = false) ?(bottom_up = false)
    ?bpm ?(walk = false) ?(on_rebuild = fun ~old_n:_ ~new_n:_ -> ()) f root =
  let cache = Ref_tbl.create 64 in
  let active = Ref_tbl.create 64 in
  let pre_cache = Ref_tbl.create 64 in
  let post, pre = if bottom_up then None, Some f else Some f, bpm in
  let loc_suffix =
    match loc with
    | None -> ""
    | Some (file, line, _, _) -> Printf.sprintf " at %s:%d" file line
  in
  let cycle_message =
    if name = "" then "Uop.graph_rewrite: rewrite cycle detected"
    else Printf.sprintf "Uop.graph_rewrite(%s): rewrite cycle detected" name
  in
  let cycle_message = cycle_message ^ loc_suffix in
  let maybe_rewrite fn u =
    match fn u with
    | Some u' when not (u == u') -> Some u'
    | _ -> None
  in
  let cached_pre_rewrite fn u =
    match Ref_tbl.find_opt pre_cache u with
    | Some r -> r
    | None ->
        let r = maybe_rewrite fn u in
        Ref_tbl.replace pre_cache u r;
        r
  in
  let apply_pre_once u =
    match pre with
    | None -> `Unchanged u
    | Some fn -> (
        try
          match cached_pre_rewrite fn u with
          | Some u' -> `Rewritten u'
          | None -> `Unchanged u
        with Bottom_up_gate -> `Gated u)
  in
  let apply_pre_fixed u =
    match pre with
    | None -> `Continue u
    | Some fn ->
        let seen = Ref_tbl.create 8 in
        let rec loop u =
          if Ref_tbl.mem seen u then invalid_arg cycle_message;
          Ref_tbl.replace seen u ();
          try
            match cached_pre_rewrite fn u with
            | Some u' -> loop u'
            | None -> `Continue u
          with Bottom_up_gate -> `Gated u
        in
        loop u
  in
  let rec rw u =
    match Ref_tbl.find_opt cache u with
    | Some r -> r
    | None ->
        if Ref_tbl.mem active u then invalid_arg cycle_message;
        Ref_tbl.replace active u ();
        try
          let result = rewrite_node u in
          Ref_tbl.replace cache u result;
          Ref_tbl.remove active u;
          if not (u == result) then on_rebuild ~old_n:u ~new_n:result;
          result
        with exn ->
          Ref_tbl.remove active u;
          raise exn
  and rewrite_node u =
    if walk then
      match apply_pre_once u with
      | `Rewritten u' | `Gated u' -> u'
      | `Unchanged u -> descend_and_post u
    else
      match apply_pre_fixed u with
      | `Gated u -> u
      | `Continue u -> descend_and_post u
  and descend_and_post u =
    let u' = rewrite_children u in
    match post with
    | None ->
        (* Bottom-up (fixed-point): once the children are rewritten and the
           node rebuilt, re-examine it so [pre] can match on the new sources.
           tinygrad re-pushes the rebuilt node through stage 0 for this. *)
        if (not walk) && not (u == u') then rw u' else u'
    | Some fn -> (
        match maybe_rewrite fn u' with
        | Some u'' -> if walk then u'' else rw u''
        | None -> u')
  and rewrite_children u =
    let srcs = src u in
    let n = Array.length srcs in
    if n = 0 then u
    else
      let changed = ref false in
      let new_srcs = Array.make n u in
      let enter =
        enter_calls
        || (match op u with Ops.Call | Ops.Function -> false | _ -> true)
      in
      for i = 0 to n - 1 do
        let child = srcs.(i) in
        let c' =
          if i = 0 && not enter
             && (match op u with Ops.Call | Ops.Function -> true | _ -> false)
          then child
          else rw child
        in
        if not (child == c') then changed := true;
        new_srcs.(i) <- c'
      done;
      if !changed then
        let u' = replace u ~src:new_srcs () in
        on_rebuild ~old_n:u ~new_n:u';
        u'
      else u
  in
  rw root

let remove_all_tags root =
  graph_rewrite ~enter_calls:true
    (fun u ->
      match node_tag u with
      | None -> None
      | Some _ ->
          let rebuilt = intern_node { u.Hashcons.node with node_tag = None } in
          let md = metadata u in
          if md = [] then Some rebuilt
          else Some (with_metadata md rebuilt))
    root

let substitute mappings root =
  let f u = List.assq_opt u mappings in
  graph_rewrite ~bottom_up:true f root

let intern root =
  graph_rewrite (fun _ -> Option.None) root

(* Analysis *)

let min_max_cache : (int * int) Ref_tbl.t = Ref_tbl.create 1024

let rec min_max u =
  match Ref_tbl.find_opt min_max_cache u with
  | Option.Some mm -> mm
  | Option.None ->
      let mm = compute_min_max u in
      Ref_tbl.replace min_max_cache u mm;
      mm

and compute_min_max u =
  (* Dtype-tight integer bounds. [min_int, max_int] for non-integer
     dtypes (bool/float are out of this analysis's domain) so callers
     treat those as "unknown". *)
  let dtype_bounds () =
    match dtype u with
    | Dtype.Val v when Dtype.Val.is_int v ->
        (match Dtype.min (Dtype.Val v), Dtype.max (Dtype.Val v) with
         | `SInt a, `SInt b -> saturate_int64 a, saturate_int64 b
         | `UInt a, `UInt b ->
             (* Unsigned dtypes store their max as a raw 64-bit bit
                pattern in int64; treating it as signed would give a
                spurious negative. Saturate to max_int for widths that
                overflow OCaml's native int. *)
             let lo = saturate_int64 a in
             let hi = if Int64.compare b 0L >= 0 then saturate_int64 b else max_int in
             lo, hi
         | _ -> min_int, max_int)
    | Dtype.Val v when Dtype.Val.is_bool v -> 0, 1
    | _ -> min_int, max_int
  in
  let sat_add a b =
    if b > 0 && a > max_int - b then max_int
    else if b < 0 && a < min_int - b then min_int
    else a + b
  in
  let sat_neg a = if a = min_int then max_int else -a in
  let sat_sub a b = sat_add a (sat_neg b) in
  let sat_mul a b =
    if a = 0 || b = 0 then 0
    else
      let p = Float.of_int a *. Float.of_int b in
      if p >= Float.of_int max_int then max_int
      else if p <= Float.of_int min_int then min_int
      else a * b
  in
  let sat_lsl a b =
    if b < 0 then 0
    else if b >= Sys.int_size - 1 then
      if a = 0 then 0 else if a > 0 then max_int else min_int
    else sat_mul a (1 lsl b)
  in
  let same_sign_nonzero lo hi = (lo > 0 && hi > 0) || (lo < 0 && hi < 0) in
  let floor_div a b =
    if b = 0 then 0
    else if a = min_int && b = -1 then max_int
    else
      let q = a / b in
      let r = a mod b in
      if r <> 0 && ((r > 0) <> (b > 0)) then q - 1 else q
  in
  let floor_mod a b =
    if b = 0 then 0 else sat_sub a (sat_mul (floor_div a b) b)
  in
  let binary_int o =
    let s0_lo, s0_hi = min_max (src u).(0) in
    let s1_lo, s1_hi = min_max (src u).(1) in
    let min4 w x y z = min (min w x) (min y z) in
    let max4 w x y z = max (max w x) (max y z) in
    match o with
    | Ops.Add -> Option.Some (sat_add s0_lo s1_lo, sat_add s0_hi s1_hi)
    | Ops.Sub -> Option.Some (sat_sub s0_lo s1_hi, sat_sub s0_hi s1_lo)
    | Ops.And
      when (match dtype u with Dtype.Val v -> Dtype.Val.is_int v | _ -> false)
           && s1_lo = s1_hi && s1_lo >= 0 ->
        let hi = if s0_lo < 0 then s1_hi else min s0_hi s1_hi in
        Option.Some (0, hi)
    | Ops.Mul ->
        let vals =
          ( sat_mul s0_lo s1_lo,
            sat_mul s0_lo s1_hi,
            sat_mul s0_hi s1_lo,
            sat_mul s0_hi s1_hi )
        in
        Option.Some
          (match vals with
           | w, x, y, z -> min4 w x y z, max4 w x y z)
    | Ops.Shl when s1_lo = s1_hi ->
        Option.Some (sat_lsl s0_lo s1_lo, sat_lsl s0_hi s1_lo)
    | Ops.Shr when s1_lo = s1_hi ->
        Option.Some (s0_lo asr s1_lo, s0_hi asr s1_lo)
    | Ops.Cmod ->
        let c = s1_lo in
        if c = s1_hi && c > 0 then
          let lo =
            if s0_lo > 0 then 0
            else if s0_lo > -c then s0_lo
            else -(s1_hi - 1)
          in
          let hi =
            if s0_hi < 0 then 0
            else if s0_hi < c then s0_hi
            else c - 1
          in
          Option.Some (lo, hi)
        else if s1_lo > 0 then
          if s0_lo >= 0 then Option.Some (0, s1_hi - 1)
          else if s0_hi <= 0 then Option.Some (-(s1_hi - 1), 0)
          else Option.Some (-(s1_hi - 1), s1_hi - 1)
        else if s1_hi < 0 then
          if s0_lo >= 0 then Option.Some (0, -s1_lo - 1)
          else if s0_hi <= 0 then Option.Some (-(-s1_lo - 1), 0)
          else Option.Some (-(-s1_lo - 1), -s1_lo - 1)
        else Option.None
    | Ops.Cdiv when same_sign_nonzero s1_lo s1_hi ->
        let cdiv a b =
          (* truncation toward zero: OCaml's integer division already
             truncates toward zero. *)
          a / b
        in
        Option.Some
          (min4 (cdiv s0_lo s1_lo) (cdiv s0_lo s1_hi)
              (cdiv s0_hi s1_lo) (cdiv s0_hi s1_hi),
           max4 (cdiv s0_lo s1_lo) (cdiv s0_lo s1_hi)
              (cdiv s0_hi s1_lo) (cdiv s0_hi s1_hi))
    | Ops.Floordiv when s0_lo > s0_hi ->
        Option.Some (0, 0)
    | Ops.Floordiv when same_sign_nonzero s1_lo s1_hi ->
        Option.Some
          (min4 (floor_div s0_lo s1_lo) (floor_div s0_lo s1_hi)
              (floor_div s0_hi s1_lo) (floor_div s0_hi s1_hi),
           max4 (floor_div s0_lo s1_lo) (floor_div s0_lo s1_hi)
              (floor_div s0_hi s1_lo) (floor_div s0_hi s1_hi))
    | Ops.Floormod when s0_lo > s0_hi ->
        Option.Some (0, 0)
    | Ops.Floormod ->
        if s1_lo = s1_hi && s1_lo > 0 then
          let c = s1_lo in
          if floor_div s0_lo c = floor_div s0_hi c then
            Option.Some (floor_mod s0_lo c, floor_mod s0_hi c)
          else Option.Some (0, c - 1)
        else if s1_lo = s1_hi && s1_lo < 0 then
          let c = s1_lo in
          if floor_div s0_lo c = floor_div s0_hi c then
            Option.Some (floor_mod s0_lo c, floor_mod s0_hi c)
          else Option.Some (c + 1, 0)
        else if s1_lo > 0 then Option.Some (0, s1_hi - 1)
        else if s1_hi < 0 then Option.Some (s1_lo + 1, 0)
        else Option.None
    | Ops.Xor when s1_lo = s1_hi && s1_lo = -1 ->
        Option.Some (lnot s0_hi, lnot s0_lo)
    | Ops.Max -> Option.Some (max s0_lo s1_lo, max s0_hi s1_hi)
    | Ops.Cmplt ->
        Option.Some
          ((if s0_hi < s1_lo then 1 else 0),
           (if s0_lo < s1_hi then 1 else 0))
    | Ops.Cmpne ->
        let lo = if s0_hi < s1_lo || s1_hi < s0_lo then 1 else 0 in
        let hi =
          if s0_lo = s0_hi && s0_lo = s1_lo && s1_lo = s1_hi then 0 else 1
        in
        Option.Some (lo, hi)
    | Ops.Or
      when (match dtype u with Dtype.Val v -> Dtype.Val.is_bool v | _ -> false) ->
        let b_or a b = if a <> 0 || b <> 0 then 1 else 0 in
        Option.Some (b_or s0_lo s1_lo, b_or s0_hi s1_hi)
    | Ops.And
      when (match dtype u with Dtype.Val v -> Dtype.Val.is_bool v | _ -> false) ->
        let b_and a b = if a <> 0 && b <> 0 then 1 else 0 in
        Option.Some (b_and s0_lo s1_lo, b_and s0_hi s1_hi)
    | _ -> Option.None
  in
  let is_int_dtype =
    match dtype u with Dtype.Val v -> Dtype.Val.is_int v | _ -> false
  in
  let is_float_dtype =
    match dtype u with Dtype.Val v -> Dtype.Val.is_float v | _ -> false
  in
  let binary_result =
    if is_float_dtype then Option.None
    else if Ops.Group.is_binary (op u) && Array.length (src u) >= 2 then
      binary_int (op u)
    else Option.None
  in
  match binary_result with
  | Option.Some r -> r
  | Option.None ->
  (match op u with
  | Ops.Where when is_int_dtype && Array.length (src u) >= 3 ->
      let t_lo, t_hi = min_max (src u).(1) in
      let f_lo, f_hi = min_max (src u).(2) in
      min t_lo f_lo, max t_hi f_hi
  | Ops.Const ->
      (match arg u with
       | Arg.Value c ->
           (match Const.view c with
            | Const.Int n -> const_int_bounds (dtype u) n
            | Const.Bool b -> (if b then 1 else 0), (if b then 1 else 0)
            | Const.Float _ | Const.Invalid -> dtype_bounds ())
       | _ -> dtype_bounds ())
  | Ops.Param ->
      (match arg u with
       | Arg.Param_arg { vmin_vmax = Some (lo, hi); _ } -> lo, hi
       | _ -> dtype_bounds ())
  | Ops.Range | Ops.Special ->
      let _, hi = min_max (src u).(0) in
      0, sat_sub hi 1
  | Ops.Bind when Array.length (src u) > 0 -> min_max (src u).(0)
  | Ops.Stack ->
      let srcs = src u in
      if Array.length srcs = 0 then dtype_bounds ()
      else Array.fold_left (fun (lo, hi) s ->
        let sl, sh = min_max s in min lo sl, max hi sh)
        (max_int, min_int) srcs
  | Ops.Index
    when Array.length (src u) > 0 && not (Dtype.is_ptr (dtype (src u).(0))) ->
      min_max (src u).(0)
  | Ops.Cast when Array.length (src u) > 0 ->
      (* CAST to bool/unsigned is not monotone; only tighten for floats
         and signed ints (including weakint). *)
      let monotone =
        match dtype u with
        | Dtype.Val v ->
            Dtype.Val.is_float v
            || (Dtype.Val.is_int v && not (Dtype.Val.is_unsigned v))
        | _ -> false
      in
      if monotone then
        let s_lo, s_hi = min_max (src u).(0) in
        let d_lo, d_hi = dtype_bounds () in
        max d_lo s_lo, min s_hi d_hi
      else dtype_bounds ()
  | _ -> dtype_bounds ())

let vmin u = fst (min_max u)
let vmax u = snd (min_max u)

let const_int_value u =
  match op u, arg u with
  | Ops.Const, Arg.Value c ->
      (match Const.view c with
       | Const.Int n
         when Int64.compare n (Int64.of_int min_int) >= 0
              && Int64.compare n (Int64.of_int max_int) <= 0 ->
           Option.Some (Int64.to_int n)
       | Const.Int _ -> Option.None
       | Const.Bool _ | Const.Float _ | Const.Invalid -> Option.None)
  | _ -> Option.None

let shape_arg dims = match dims with [ d ] -> d | ds -> stack ds

let dim_one = const_int 1

let dim_is_one d =
  match const_int_value d with Some 1 -> true | Some _ | None -> false

let dim_binary op a b =
  match const_int_value a, const_int_value b with
  | Some x, Some y ->
      let z = match op with
        | Ops.Add -> x + y
        | Ops.Sub -> x - y
        | Ops.Mul -> x * y
        | Ops.Floordiv -> x / y
        | _ -> invalid_arg "Uop.dim_binary: unsupported op"
      in
      const_int z
  | _ -> alu_binary ~op ~lhs:a ~rhs:b

let dim_add a b = dim_binary Ops.Add a b
let dim_sub a b = dim_binary Ops.Sub a b
let dim_mul a b = dim_binary Ops.Mul a b
let dim_div a b = dim_binary Ops.Floordiv a b

let dim_prod dims = List.fold_left dim_mul dim_one dims

let dims_equal a b =
  List.length a = List.length b && List.for_all2 equal a b

let dim_non_negative d = vmin d >= 0

let dim_leq a b =
  match const_int_value a, const_int_value b with
  | Some a, Some b -> a <= b
  | _ -> equal a b || vmax a <= vmin b

let invalid_shape op msg =
  invalid_arg
    (Printf.sprintf "Uop.shape: invalid %s: %s" (Ops.name op) msg)

let require_movement_len op src_shape marg =
  if List.length src_shape <> List.length marg then
    invalid_shape op
      (Printf.sprintf "rank %d does not match argument rank %d"
         (List.length src_shape) (List.length marg))

let require_non_negative_shape op dims =
  if not (List.for_all dim_non_negative dims) then
    invalid_shape op "shape contains a negative dimension"

let require_reshape op src_shape target =
  require_non_negative_shape op target;
  match const_int_value (dim_prod src_shape), const_int_value (dim_prod target) with
  | Some src_count, Some target_count when src_count <> target_count ->
      invalid_shape op
        (Printf.sprintf "element count changes from %d to %d" src_count
           target_count)
  | _ ->
      if not (equal (dim_prod src_shape) (dim_prod target)) then ()

let require_expand op src_shape target =
  require_movement_len op src_shape target;
  require_non_negative_shape op target;
  List.iter2
    (fun old_dim new_dim ->
      if not (equal old_dim new_dim || dim_is_one old_dim) then
        invalid_shape op "dimension is neither unchanged nor broadcast from one")
    src_shape target

let require_permute op rank order =
  let sorted = List.sort Int.compare order in
  let expected = List.init rank Fun.id in
  if sorted <> expected then invalid_shape op "invalid permutation"

let require_pad op src_shape offsets sizes =
  require_movement_len op src_shape offsets;
  require_movement_len op src_shape sizes;
  require_non_negative_shape op offsets;
  require_non_negative_shape op sizes;
  List.iter2
    (fun src_dim (offset, size) ->
      if not (dim_leq (dim_add offset src_dim) size) then
        invalid_shape op "padded size is smaller than offset plus input size")
    src_shape (List.combine offsets sizes)

let require_shrink op src_shape offsets sizes =
  require_movement_len op src_shape offsets;
  require_movement_len op src_shape sizes;
  require_non_negative_shape op offsets;
  require_non_negative_shape op sizes;
  List.iter2
    (fun src_dim (offset, size) ->
      if not (dim_leq (dim_add offset size) src_dim) then
        invalid_shape op "slice extends past the input shape")
    src_shape (List.combine offsets sizes)

let rec broadcast_shape shapes =
  let rec last = function
  | [] -> None
  | [ x ] -> Some x
  | _ :: xs -> last xs
  in
  let rec drop_last = function
  | [] | [ _ ] -> []
  | x :: xs -> x :: drop_last xs
  in
  match List.filter (( <> ) []) shapes with
  | [] -> []
  | shapes ->
    let tails = List.filter_map last shapes in
    let dim =
      List.fold_left
        (fun acc d ->
          match acc with
          | None -> Some d
          | Some a when equal a d -> Some a
          | Some a when dim_is_one a -> Some d
          | Some a when dim_is_one d -> Some a
          | Some a -> Some a)
        None tails
    in
    let prefixes = List.map drop_last shapes in
    let prefix =
      if List.for_all (( = ) []) prefixes then [] else broadcast_shape prefixes
    in
    match dim with None -> prefix | Some d -> prefix @ [ d ]

let rec as_shape u =
  match op u with
  | Ops.Stack -> Array.to_list (src u)
  | Ops.Const ->
      let count = Dtype.count (dtype u) in
      List.init count (fun _ -> u)
  | _ -> [ u ]

let marg u =
  let srcs = src u in
  match op u with
  | Ops.Reshape | Ops.Expand ->
      if Array.length srcs >= 2 then Marg_shape (as_shape srcs.(1))
      else invalid_arg "Uop.marg: movement op missing shape source"
  | Ops.Pad | Ops.Shrink ->
      if Array.length srcs >= 3 then (
        try Marg_bounds (List.combine (as_shape srcs.(1)) (as_shape srcs.(2)))
        with Invalid_argument _ ->
          invalid_arg "Uop.marg: movement bounds length mismatch")
      else invalid_arg "Uop.marg: movement op missing bound sources"
  | Ops.Permute ->
      (match Arg.as_ints (arg u) with
       | Some order -> Marg_permute order
       | None -> invalid_arg "Uop.marg: Permute arg is not an int list")
  | Ops.Flip ->
      (match Arg.as_bools (arg u) with
       | Some dims -> Marg_flip dims
       | None -> invalid_arg "Uop.marg: Flip arg is not a bool list")
  | _ -> invalid_arg "Uop.marg: op is not a movement op"

let substitute_function_shape_args fn dims =
  let fn_srcs = src fn in
  let n_args = max 0 (Array.length fn_srcs - 1) in
  let args = Array.init n_args (fun i -> fn_srcs.(i + 1)) in
  let resolve_param u =
    match op u, arg u with
    | Ops.Param, Arg.Param_arg { slot; _ } when slot >= 0 && slot < n_args ->
        Some args.(slot)
    | _ -> None
  in
  List.map (graph_rewrite ~walk:true resolve_param) dims

(* Per-node shape memo. Nodes are hash-consed and immutable, so a shape is a
   pure function of the node; caching it by node identity (as [device_of] and
   [min_max] already do) avoids recomputing shared subgraphs, which would
   otherwise re-walk a diamond once per parent and blow up exponentially. *)
let shape_cache : t list option Ref_tbl.t = Ref_tbl.create 1024

let rec shape u =
  match shape_opt u with
  | Some shape -> shape
  | None ->
      invalid_arg
        (Printf.sprintf "Uop.shape: %s does not have a shape"
           (Ops.name (op u)))

and shape_opt u =
  match Ref_tbl.find_opt shape_cache u with
  | Some cached -> cached
  | None ->
      let result = compute_shape_opt u in
      Ref_tbl.add shape_cache u result;
      result

and compute_shape_opt u =
  let srcs = src u in
  let first_shape () =
    if Array.length srcs = 0 then None else shape_opt srcs.(0)
  in
  match op u with
  | Ops.If | Ops.Barrier | Ops.Custom | Ops.Customi | Ops.Sink
  | Ops.Rewrite_error | Ops.Endif | Ops.Linear | Ops.Program | Ops.Source
  | Ops.Ins | Ops.Tuple | Ops.Call | Ops.Function | Ops.Custom_function ->
      None
  | Ops.Noop ->
      if Array.length srcs = 0 then None else shape_opt srcs.(0)
  | Ops.Gettuple ->
      (match Arg.as_int (arg u), Array.to_list srcs with
       | Some i, [ tuple ] when op tuple = Ops.Tuple ->
           let tuple_srcs = src tuple in
           if i >= 0 && i < Array.length tuple_srcs then
             shape_opt tuple_srcs.(i)
           else None
       | Some i, [ fn ] when op fn = Ops.Function ->
           let fn_srcs = src fn in
           if Array.length fn_srcs = 0 then None
           else
             let tuple = fn_srcs.(0) in
             let tuple_srcs = src tuple in
             if op tuple = Ops.Tuple && i >= 0 && i < Array.length tuple_srcs
             then
               Option.map (substitute_function_shape_args fn)
                 (shape_opt tuple_srcs.(i))
             else None
       | _ -> None)
  | Ops.Index ->
      if Array.length srcs = 0 then None
      else
        let ptr_shape = shape srcs.(0) in
        let index_shapes =
          Array.to_list srcs |> List.tl |> List.concat_map shape
        in
        let dropped =
          let rec drop n xs =
            if n = 0 then xs else match xs with [] -> [] | _ :: xs -> drop (n - 1) xs
          in
          drop (Array.length srcs - 1) ptr_shape
        in
        Some (index_shapes @ dropped)
  | Ops.Stack ->
      if Array.length srcs = 0 then Some []
      else
        (match dtype u with
         | Dtype.Ptr _ -> shape_opt srcs.(0)
         | Dtype.Val _ -> Some (const_int (Array.length srcs) :: shape srcs.(0)))
  | Ops.Const ->
      let count = Dtype.count (dtype u) in
      Some (if count > 1 then [ const_int count ] else [])
  | Ops.Getaddr | Ops.Bind | Ops.Range | Ops.Special | Ops.Binary -> Some []
  | Ops.Buffer ->
      (match Array.to_list srcs, dtype u with
       | shape :: _, _ when op shape <> Ops.Noop -> Some (as_shape shape)
       | _, Dtype.Ptr p when Dtype.Ptr.is_image p ->
           Some (List.map const_int (Dtype.Image.shape p))
       | _, Dtype.Ptr p -> Some [ const_int (Dtype.Ptr.size p) ]
       | _, Dtype.Val v ->
           let count = Dtype.Val.count v in
           Some (if count > 1 then [ const_int count ] else []))
  | Ops.Param ->
      (match Array.to_list srcs, dtype u with
       | shape :: _, _ when op shape <> Ops.Noop -> Some (as_shape shape)
       | _, Dtype.Ptr p when Dtype.Ptr.is_image p ->
           Some (List.map const_int (Dtype.Image.shape p))
       | _, Dtype.Ptr p -> Some [ const_int (Dtype.Ptr.size p) ]
       | _, Dtype.Val v ->
           let count = Dtype.Val.count v in
           Some (if count > 1 then [ const_int count ] else []))
  | Ops.Slice ->
      if Array.length srcs > 0 && op srcs.(0) = Ops.Index then Some []
      else
        Option.map
          (fun ({ size; _ } : slice_view) -> [ const_int size ])
          (as_slice u)
  | Ops.Stage ->
      let ranges = Array.to_list srcs |> List.tl in
      let range_shape =
        List.map
          (fun r ->
            let rs = src r in
            if op r = Ops.Range && Array.length rs > 0 then rs.(0)
            else const_int (vmax r + 1))
          ranges
      in
      if Array.length srcs = 0 then None else Some (range_shape @ shape srcs.(0))
  | Ops.Wmma ->
      (* output shape = broadcast of the srcs' shapes minus the packed tail,
         plus the upcast output tail *)
      if Array.length srcs >= 3 then
        match arg u with
        | Arg.Wmma_info info -> (
            let drop_last l =
              match List.rev l with [] -> [] | _ :: rest -> List.rev rest
            in
            match
              (shape_opt srcs.(0), shape_opt srcs.(1), shape_opt srcs.(2))
            with
            | Some s0, Some s1, Some s2 ->
                let _, _, out0 = info.upcast_axes in
                let out_sz =
                  List.fold_left (fun acc (_, sz) -> acc * sz) 1 out0
                in
                Some
                  (broadcast_shape [ drop_last s0; drop_last s1; drop_last s2 ]
                  @ [ const_int out_sz ])
            | _ -> None)
        | _ -> None
      else None
  | Ops.Shaped_wmma ->
      if Array.length srcs >= 3 then shape_opt srcs.(2) else None
  | Ops.Mstack | Ops.Mselect | Ops.Detach | Ops.Contiguous
  | Ops.Contiguous_backward | Ops.After | Ops.Load | Ops.Copy
  | Ops.Allreduce | Ops.Store | Ops.End ->
      first_shape ()
  | Ops.Reduce ->
      (match as_reduce u with
       | Some { src; axes = []; _ } -> Some (shape src)
       | Some { src; axes; _ } ->
           let src_shape = shape src in
           Some (List.filteri (fun i _ -> not (List.mem i axes)) src_shape)
       | None -> None)
  | Ops.Bitcast ->
      (match first_shape () with
       | None -> None
       | Some ps ->
           let input_size = Dtype.itemsize (dtype srcs.(0)) in
           let output_size = Dtype.itemsize (dtype u) in
           if input_size = output_size then Some ps
           else
             (match List.rev ps with
              | [] -> Some ps
              | last :: rev_prefix ->
                  Some
                    (List.rev rev_prefix
                     @ [ dim_div (dim_mul last (const_int input_size))
                           (const_int output_size) ])))
  | Ops.Reshape ->
      if Array.length srcs >= 2 then begin
        let target = as_shape srcs.(1) in
        (if op srcs.(0) = Ops.Noop then
          require_non_negative_shape Ops.Reshape target
        else
          let src_shape = shape srcs.(0) in
          require_reshape Ops.Reshape src_shape target);
        Some target
      end
      else None
  | Ops.Expand ->
      if Array.length srcs >= 2 then begin
        let target = as_shape srcs.(1) in
        let src_shape = shape srcs.(0) in
        (try require_expand Ops.Expand src_shape target
         with Invalid_argument msg ->
           invalid_arg
             (Printf.sprintf "%s src=%s target=%s child=%s child_srcs=%s node=%s"
                msg
                (String.concat ","
                   (List.map (fun s ->
                        Printf.sprintf "%d:%s" (tag s) (Ops.name (op s)))
                      src_shape))
                (String.concat ","
                   (List.map (fun s ->
                        Printf.sprintf "%d:%s" (tag s) (Ops.name (op s)))
                      target))
                (Printf.sprintf "%d:%s" (tag srcs.(0))
                   (Ops.name (op srcs.(0))))
                (Array.to_list (src srcs.(0))
                |> List.map (fun child ->
                       let grand =
                         Array.to_list (src child)
                         |> List.map (fun grand ->
                                Printf.sprintf "%d:%s/%d" (tag grand)
                                  (Ops.name (op grand))
                                  (try List.length (shape grand)
                                   with Invalid_argument _ -> -1))
                         |> String.concat ","
                       in
                       Printf.sprintf "%d:%s/%d" (tag child)
                         (Ops.name (op child))
                         (try List.length (shape child)
                          with Invalid_argument _ -> -1)
                         ^ "[" ^ grand ^ "]")
                |> String.concat ";")
                (Printf.sprintf "%d:%s" (tag u) (Ops.name (op u)))));
        Some target
      end
      else None
  | Ops.Permute ->
      (match Arg.as_ints (arg u), first_shape () with
       | Some order, Some ps ->
           (try require_permute Ops.Permute (List.length ps) order
            with Invalid_argument msg ->
              invalid_arg
                (Printf.sprintf "%s rank=%d order=[%s] child=%s node=%s"
                   msg (List.length ps)
                   (String.concat "," (List.map string_of_int order))
                   (if Array.length srcs > 0 then
                      Printf.sprintf "%d:%s" (tag srcs.(0))
                        (Ops.name (op srcs.(0)))
                    else "-")
                   (Printf.sprintf "%d:%s" (tag u) (Ops.name (op u)))));
           Some (List.map (List.nth ps) order)
       | _ -> None)
  | Ops.Pad ->
      if Array.length srcs >= 3 then
        let ps = shape srcs.(0) in
        let offsets = as_shape srcs.(1) in
        let sizes = as_shape srcs.(2) in
        require_pad Ops.Pad ps offsets sizes;
        Some sizes
      else None
  | Ops.Shrink ->
      if Array.length srcs >= 3 then
        let ps = shape srcs.(0) in
        let offsets = as_shape srcs.(1) in
        let sizes = as_shape srcs.(2) in
        require_shrink Ops.Shrink ps offsets sizes;
        Some sizes
      else None
  | Ops.Flip ->
      (match first_shape (), Arg.as_bools (arg u) with
       | Some ps, Some dims ->
           if List.length ps <> List.length dims then
             invalid_shape Ops.Flip "rank does not match argument rank";
           Some ps
       | ps, _ -> ps)
  | Ops.Multi ->
      (match first_shape (), Arg.as_int (arg u), device_of u with
       | Some ps, Some axis, Some (Multi devs) ->
           Some
             (List.mapi
                (fun i d ->
                  if i = axis then dim_mul d (const_int (List.length devs))
                  else d)
                ps)
       | ps, _, _ -> ps)
  | op when Ops.Group.is_unary op || op = Ops.Cast || op = Ops.Load ->
      first_shape ()
  | op when Ops.Group.is_broadcastable op ->
      let shapes = Array.to_list srcs |> List.filter_map shape_opt in
      if shapes = [] then None else Some (broadcast_shape shapes)
  | _ -> None

let max_shape u = List.map vmax (shape u)

let rec axis u =
  let srcs = src u in
  match op u with
  | Ops.Copy -> None
  | Ops.Multi -> Arg.as_int (arg u)
  | Ops.Gettuple ->
      (match Arg.as_int (arg u), Array.to_list srcs with
       | Some i, [ tuple ] when op tuple = Ops.Tuple ->
           let tuple_srcs = src tuple in
           if i >= 0 && i < Array.length tuple_srcs then axis tuple_srcs.(i)
           else None
       | Some i, [ fn ] when op fn = Ops.Function ->
           let fn_srcs = src fn in
           if Array.length fn_srcs = 0 then None
           else
             let tuple = fn_srcs.(0) in
             let tuple_srcs = src tuple in
             if op tuple = Ops.Tuple && i >= 0 && i < Array.length tuple_srcs
             then axis tuple_srcs.(i)
             else None
       | _ -> None)
  | Ops.Param ->
      (match Arg.as_param_arg (arg u) with
       | Some param -> param.axis
       | None -> None)
  | op when Ops.Group.is_alu op ->
      let axes =
        Array.fold_left
          (fun acc s ->
            match axis s with
            | None -> acc
            | Some a -> if List.mem a acc then acc else a :: acc)
          [] srcs
      in
      (match axes with [] -> None | a :: _ -> Some a)
  | _ when Array.length srcs = 0 -> None
  | Ops.Shrink -> (
      match axis srcs.(0) with
      | None -> None
      | Some ax ->
          let src_shape = shape srcs.(0) in
          let before = if Array.length srcs > 1 then as_shape srcs.(1) else [] in
          let after = if Array.length srcs > 2 then as_shape srcs.(2) else [] in
          if ax < List.length before && ax < List.length after
             && equal (List.nth before ax) (const_int 0)
             && equal (List.nth after ax) (List.nth src_shape ax)
          then Some ax
          else None)
  | Ops.Reduce -> (
      match axis srcs.(0), as_reduce u with
      | Some ax, Some { axes; _ } ->
          if List.mem ax axes then None else Some ax
      | ax, _ -> ax)
  | Ops.Reshape -> (
      match axis srcs.(0), device_of u with
      | None, _ -> None
      | Some src_axis, _ ->
          let src_shape = shape srcs.(0) in
          let target = shape u in
          let prefix = dim_prod (List.filteri (fun i _ -> i < src_axis) src_shape) in
          let rec scan last prod i = function
          | [] -> last
          | d :: ds ->
              let last = if equal prod prefix then Some i else last in
              scan last (dim_mul prod d) (i + 1) ds
          in
          (match scan None dim_one 0 target with
           | None -> None
           | Some new_axis ->
               (match device_of u, const_int_value (List.nth target new_axis) with
                | Some (Multi devs), Some n when n mod List.length devs <> 0 ->
                    invalid_arg
                      (Printf.sprintf
                         "Uop.axis: reshape moved items between shards")
                | _ -> ());
               Some new_axis))
  | Ops.Permute -> (
      match axis srcs.(0), Arg.as_ints (arg u) with
      | Some ax, Some order ->
          let rec find i = function
          | [] -> None
          | x :: xs -> if x = ax then Some i else find (i + 1) xs
          in
          find 0 order
      | ax, _ -> ax)
  | _ -> axis srcs.(0)

let shard_shape u =
  match device_of u, axis u with
  | Some (Multi devs), Some ax ->
      List.mapi
        (fun i d ->
          if i = ax then dim_div d (const_int (List.length devs)) else d)
        (shape u)
  | _ -> shape u

let max_shard_shape u = List.map vmax (shard_shape u)

let bounds u =
  match axis u, device_of u with
  | None, _ -> invalid_arg "Uop.bounds: axis is None"
  | _, None | _, Some (Single _) | _, Some (Index _) ->
      invalid_arg "Uop.bounds: device is not multi"
  | Some ax, Some (Multi devs) ->
      let source =
        let srcs = src u in
        if Array.length srcs = 0 then u else srcs.(0)
      in
      let shard = List.nth (shape source) ax in
      List.init (List.length devs) (fun i ->
          let lo = dim_mul shard (const_int i) in
          let hi = dim_mul shard (const_int (i + 1)) in
          lo, hi)

let contiguous_view_offset u =
  let exact_int t = const_int_value t in
  let dim_is_zero t = match exact_int t with Some 0 -> true | _ -> false in
  let dim_max_at_most n t = vmax t <= n in
  let same_dim a b =
    equal a b
    ||
    match exact_int a, exact_int b with
    | Some a, Some b -> a = b
    | _ -> false
  in
  let exact_stride dims =
    let rec loop acc = function
    | [] -> Some acc
    | d :: ds ->
        (match exact_int d with
         | Some n -> loop (acc * n) ds
         | None -> None)
    in
    loop 1 dims
  in
  let shape_arg srcs i =
    if Array.length srcs > i then Some (as_shape srcs.(i)) else None
  in
  let pairs_arg srcs =
    match shape_arg srcs 1, shape_arg srcs 2 with
    | Some before, Some after -> (
        try Some (List.combine before after) with
        | Invalid_argument _ -> None)
    | _ -> None
  in
  let valid_permutation order len =
    List.length order = len
    && List.sort Int.compare order = List.init len Fun.id
  in
  let permute_list order xs =
    try Some (List.map (List.nth xs) order) with
    | Failure _ | Invalid_argument _ -> None
  in
  let contiguous_permutation order dims =
    if not (valid_permutation order (List.length dims)) then false
    else
      let non_singleton =
        List.filter_map
          (fun (i, d) -> if dim_max_at_most 1 d then None else Some i)
          (List.mapi (fun i d -> i, d) dims)
      in
      List.filter (fun i -> List.mem i non_singleton) order = non_singleton
  in
  let shrink_shape_and_offset shape pairs =
    let axis_count = List.length shape in
    let dims = Array.of_list shape in
    let bounds = Array.of_list pairs in
    if Array.length bounds <> axis_count then None
    else
      let rec loop prefix_one offset out i =
        if i = axis_count then Some (offset, List.rev out)
        else
          let dim = dims.(i) in
          let offset_dim, size = bounds.(i) in
          if dim_is_zero offset_dim && same_dim size dim then
            loop (prefix_one && dim_max_at_most 1 dim) offset (dim :: out)
              (i + 1)
          else
            match exact_int offset_dim, exact_int size, exact_int dim with
            | Some offset_dim, Some size, Some dim
              when offset_dim >= 0 && size >= 0 && offset_dim + size <= dim
                   && prefix_one ->
                let trailing = List.filteri (fun j _ -> j > i) shape in
                (match exact_stride trailing with
                 | Some stride ->
                     loop (size <= 1) (offset + (offset_dim * stride))
                       (const_int size :: out) (i + 1)
                 | None when offset_dim = 0 ->
                     loop (size <= 1) offset (const_int size :: out) (i + 1)
                 | None -> None)
            | _ -> None
      in
      loop true 0 [] 0
  in
  let rec walk node =
    match op node with
    | Ops.Buffer | Ops.Param -> Some (0, shape node)
    | Ops.Slice ->
        (match as_slice node with
         | Some { src; offset; _ } ->
             (match exact_int offset, walk src with
              | Some off, Some (base_off, _) ->
                  Some (base_off + off, shape node)
              | _ -> None)
         | None -> None)
    | Ops.Detach | Ops.Contiguous | Ops.Contiguous_backward | Ops.After ->
        let srcs = src node in
        if Array.length srcs = 0 then None else walk srcs.(0)
    | Ops.Reshape ->
        let srcs = src node in
        if Array.length srcs = 0 then None
        else
          (match walk srcs.(0), shape_arg srcs 1 with
           | Some (base_off, _), Some target -> Some (base_off, target)
           | _ -> None)
    | Ops.Expand ->
        let srcs = src node in
        if Array.length srcs = 0 then None
        else
          (match walk srcs.(0), shape_arg srcs 1 with
           | Some (base_off, current), Some target
             when List.length current = List.length target
                  && List.for_all2 same_dim current target ->
               Some (base_off, target)
           | _ -> None)
    | Ops.Pad ->
        let srcs = src node in
        if Array.length srcs = 0 then None
        else
          (match walk srcs.(0), pairs_arg srcs with
           | Some ((_, current) as state), Some pairs
             when List.length current = List.length pairs
                  && List.for_all2
                       (fun dim (offset, size) ->
                         dim_is_zero offset && same_dim dim size)
                       current pairs ->
               Some state
           | _ -> None)
    | Ops.Shrink ->
        let srcs = src node in
        if Array.length srcs = 0 then None
        else
          (match walk srcs.(0), pairs_arg srcs with
           | Some (base_off, current), Some pairs ->
               (match shrink_shape_and_offset current pairs with
                | Some (offset, shape) -> Some (base_off + offset, shape)
                | None -> None)
           | _ -> None)
    | Ops.Permute ->
        let srcs = src node in
        if Array.length srcs = 0 then None
        else
          (match walk srcs.(0), Arg.as_ints (arg node) with
           | Some (base_off, current), Some order
             when contiguous_permutation order current ->
               (match permute_list order current with
                | Some shape -> Some (base_off, shape)
                | None -> None)
           | _ -> None)
    | Ops.Flip ->
        let srcs = src node in
        if Array.length srcs = 0 then None
        else
          (match walk srcs.(0), Arg.as_bools (arg node) with
           | Some state, Some dims
             when List.length dims = List.length (snd state)
                  && List.for_all2
                       (fun flipped dim ->
                         (not flipped) || dim_max_at_most 1 dim)
                       dims (snd state) ->
               Some state
           | _ -> None)
    | _ -> None
  in
  Option.map fst (walk u)

let rec const_of_dtype ?shape:target_shape dtype value =
  let ret =
    match value with
    | Const_scalar value -> const (Const.of_scalar dtype value)
    | Const_invalid -> const (Const.invalid ~dtype ())
    | Const_tuple values ->
        let count = Dtype.Val.count dtype in
        let len = List.length values in
        if len <> count then
          invalid_arg
            (Printf.sprintf "Uop.const_of_dtype: tuple length %d mismatches %s"
               len (Dtype.Val.to_string dtype));
        let scalar_dtype = Dtype.Val.scalarize dtype in
        let src =
          Array.of_list
            (List.map (const_of_dtype scalar_dtype) values)
        in
        mk ~op:Ops.Stack ~dtype:(Dtype.Val dtype) ~src ~arg:Arg.Empty
  in
  match target_shape with
  | None -> ret
  | Some target_arg ->
      let target = as_shape target_arg in
      if target = [] || dims_equal (shape ret) target then ret
      else
        let ones = List.map (fun _ -> dim_one) target in
        expand ~src:(reshape ~src:ret ~shape:(shape_arg ones))
          ~shape:target_arg

let rec const_factor u =
  match op u with
  | Ops.Const ->
      (match const_int_value u with Option.Some n -> n | Option.None -> 1)
  | Ops.Stack ->
      let srcs = src u in
      if Array.length srcs = 0 then 0
      else
        let rec gcd x y = if y = 0 then abs x else gcd y (x mod y) in
        Array.fold_left (fun acc s -> gcd acc (const_factor s))
          (const_factor srcs.(0)) srcs
  | Ops.Add ->
      let a = (src u).(0) and b = (src u).(1) in
      let rec gcd x y = if y = 0 then abs x else gcd y (x mod y) in
      gcd (const_factor a) (const_factor b)
  | Ops.Mul ->
      let a = (src u).(0) and b = (src u).(1) in
      (match const_int_value a, const_int_value b with
       | Option.Some n, _ | _, Option.Some n -> n
       | _ -> 1)
  | _ -> 1

let rec divides u n =
  if n = 1 then Option.Some u
  else match op u with
  | Ops.Const ->
      (match const_int_value u with
       | Option.Some m when m mod n = 0 -> Option.Some (const_like u (m / n))
       | _ -> Option.None)
  | Ops.Stack ->
      let divided =
        Array.map (fun s -> divides s n) (src u)
      in
      if Array.exists Option.is_none divided then Option.None
      else
        let srcs = Array.to_list (Array.map Option.get divided) in
        (match dtype u with
         | Dtype.Val dt -> Option.Some (stack ~dtype:dt srcs)
         | Dtype.Ptr _ -> Option.Some (stack srcs))
  | Ops.Add ->
      let a = (src u).(0) and b = (src u).(1) in
      (match divides a n, divides b n with
       | Option.Some qa, Option.Some qb ->
           Option.Some (alu_binary ~op:Ops.Add ~lhs:qa ~rhs:qb)
       | _ -> Option.None)
  | Ops.Mul ->
      let a = (src u).(0) and b = (src u).(1) in
      (match divides a n with
       | Option.Some qa -> Option.Some (alu_binary ~op:Ops.Mul ~lhs:qa ~rhs:b)
       | Option.None ->
           (match divides b n with
            | Option.Some qb -> Option.Some (alu_binary ~op:Ops.Mul ~lhs:a ~rhs:qb)
            | Option.None -> Option.None))
  | Ops.Shl -> (
      let a = (src u).(0) and b = (src u).(1) in
      match const_int_value b with
      | Some shift when shift >= 0 && shift < Sys.int_size - 2 ->
          let factor = 1 lsl shift in
          if factor mod n = 0 then
            let q = factor / n in
            if q = 1 then Some a
            else Some (alu_binary ~op:Ops.Mul ~lhs:a ~rhs:(const_like u q))
          else if n mod factor = 0 then divides a (n / factor)
          else None
      | _ -> None)
  | _ -> Option.None

let pop_const u =
  match op u with
  | Ops.Add ->
      let a = (src u).(0) and b = (src u).(1) in
      (match const_int_value b with
       | Option.Some n -> a, n
       | Option.None -> u, 0)
  | _ -> u, 0

let rec split_uop u target_op =
  if op u = target_op && Array.length (src u) = 2 then
    split_uop (src u).(0) target_op @ split_uop (src u).(1) target_op
  else [ u ]

let err_empty_list fn =
  invalid_arg (Printf.sprintf "Uop.%s: empty list" fn)

let usum = function
  | [] -> err_empty_list "usum"
  | x :: xs ->
      List.fold_left (fun acc y -> alu_binary ~op:Ops.Add ~lhs:acc ~rhs:y) x xs

let uprod = function
  | [] -> err_empty_list "uprod"
  | x :: xs ->
      List.fold_left (fun acc y -> alu_binary ~op:Ops.Mul ~lhs:acc ~rhs:y) x xs

let remove_one_factor needle factors =
  let rec loop prefix = function
    | [] -> Option.None
    | x :: xs ->
        if equal x needle then Option.Some (List.rev_append prefix xs)
        else loop (x :: prefix) xs
  in
  loop [] factors

let product_factors u =
  let factors = split_uop u Ops.Mul in
  let const_prod, rest =
    List.fold_left
      (fun (c, xs) f ->
        match const_int_value f with
        | Option.Some n -> (c * n, xs)
        | Option.None -> (c, f :: xs))
      (1, []) factors
  in
  (const_prod, List.rev rest)

let product_like exemplar const_part factors =
  let nodes =
    if const_part = 1 && factors <> [] then factors
    else const_like exemplar const_part :: factors
  in
  match nodes with
  | [] -> const_like exemplar 1
  | [ x ] -> x
  | xs -> uprod xs

let divide_product_exact u d =
  let u_const, u_factors = product_factors u in
  let d_const, d_factors = product_factors d in
  if d_const = 0 || u_const mod d_const <> 0 then Option.None
  else
    let rec remove_all remaining = function
      | [] -> Option.Some remaining
      | f :: fs ->
          (match remove_one_factor f remaining with
           | Option.None -> Option.None
           | Option.Some remaining -> remove_all remaining fs)
    in
    match remove_all u_factors d_factors with
    | Option.None -> Option.None
    | Option.Some factors -> Option.Some (product_like u (u_const / d_const) factors)

let common_product_factors = function
  | [] -> []
  | first :: rest ->
      let rec take common remaining_rest = function
        | [] -> List.rev common
        | f :: fs ->
            let rec remove_from_all acc = function
              | [] -> Option.Some (List.rev acc)
              | factors :: more ->
                  (match remove_one_factor f factors with
                   | Option.None -> Option.None
                   | Option.Some factors -> remove_from_all (factors :: acc) more)
            in
            (match remove_from_all [] remaining_rest with
             | Option.None -> take common remaining_rest fs
             | Option.Some remaining_rest -> take (f :: common) remaining_rest fs)
      in
      take [] rest first

(* Exact divisibility by a uop [d]. For a constant [d], defer to
   [divides]. Otherwise, [u] must be a sum whose every term divides by
   [d], or a product containing all multiplicative factors in [d]. *)
let rec divide_exact u d =
  if equal u d then Option.Some (const_like u 1)
  else match op d, const_int_value d with
  | Ops.Const, Option.Some n -> divides u n
  | _ ->
      match op u with
      | Ops.Add ->
          let a = (src u).(0) and b = (src u).(1) in
          (match divide_exact a d, divide_exact b d with
           | Option.Some qa, Option.Some qb ->
               Option.Some (alu_binary ~op:Ops.Add ~lhs:qa ~rhs:qb)
           | _ -> Option.None)
      | Ops.Mul ->
          divide_product_exact u d
      | _ -> Option.None

let gcd = function
  | [] -> err_empty_list "gcd"
  | xs ->
      let rec gcd_int a b = if b = 0 then abs a else gcd_int b (a mod b) in
      let decompose x =
        let factor = const_factor x in
        if factor = 0 then Option.None
        else
          match divides x factor with
          | Option.None -> Option.None
          | Option.Some term -> Option.Some (factor, split_uop term Ops.Mul)
      in
      let decomposed = List.filter_map decompose xs in
      if List.length decomposed <> List.length xs then const_like (List.hd xs) 1
      else
        let factors = List.map fst decomposed in
        let term_factors = List.map snd decomposed in
        let common = common_product_factors term_factors in
        let g = List.fold_left gcd_int 0 factors in
        product_like (List.hd xs) g common

let simplify_ref : (t -> t) ref = ref (fun u -> u)
let simplify u = !simplify_ref u

(* Structural compare *)

let rec compare_structure a b =
  if a == b then 0
  else
    let c = Ops.compare (op a) (op b) in
    if c <> 0 then c
    else
      let c = Arg.compare (arg a) (arg b) in
      if c <> 0 then c
      else
        let c = Dtype.compare (dtype a) (dtype b) in
        if c <> 0 then c
        else
          let sa = src a and sb = src b in
          let rec cmp i =
            if i = Array.length sa || i = Array.length sb then
              Int.compare (Array.length sa) (Array.length sb)
            else
              let c = compare_structure sa.(i) sb.(i) in
              if c <> 0 then c else cmp (i + 1)
          in
          cmp 0

let param_slot u =
  match op u, arg u with
  | Ops.Param, Arg.Param_arg p -> Some p.slot
  | _ -> None

let dedup_refs nodes =
  let seen = Ref_tbl.create (List.length nodes) in
  let rec loop acc = function
    | [] -> List.rev acc
    | u :: us ->
        if Ref_tbl.mem seen u then loop acc us
        else begin
          Ref_tbl.replace seen u ();
          loop (u :: acc) us
        end
  in
  loop [] nodes

let sort_program_vars vars =
  List.sort
    (fun a b ->
      match param_slot a, param_slot b with
      | Some sa, Some sb ->
          let c = Int.compare sa sb in
          if c <> 0 then c else compare_structure a b
      | Some _, None -> -1
      | None, Some _ -> 1
      | None, None -> compare_structure a b)
    (dedup_refs vars)

let sort_uniq_ints xs = List.sort_uniq Int.compare xs

let set_nth name xs i value =
  if i < 0 || i >= List.length xs then
    invalid_arg (Printf.sprintf "Uop.%s: launch axis out of range" name);
  List.mapi (fun j x -> if i = j then value else x) xs

let launch_dim_of_uop u =
  match const_int_value (simplify u) with
  | Some n -> Launch_int n
  | None -> Launch_sym u

let special_axis_with_prefix name prefix =
  let n = String.length prefix in
  if String.length name >= n && String.sub name 0 n = prefix then
    int_of_string_opt (String.sub name n (String.length name - n))
  else None

let exact_int_for_program name u =
  match const_int_value (simplify u) with
  | Some n -> n
  | None ->
      invalid_arg
        (Printf.sprintf "Uop.%s: expected concrete integer launch dimension" name)

let program_index_buffer u =
  let index =
    match op u, Array.to_list (src u) with
    | (Ops.Index | Ops.Shrink), _ -> Some u
    | Ops.Cast, [ inner ] when Ops.equal (op inner) Ops.Index -> Some inner
    | _ -> None
  in
  match index with
  | Some idx when Array.length (src idx) > 0 -> Some (buf_uop (src idx).(0))
  | _ -> None

let program_info_from_sink ?(aux = []) sink =
  let vars = ref [] in
  let globals = ref [] in
  let outs = ref [] in
  let ins = ref [] in
  let global_size = ref [ Launch_int 1; Launch_int 1; Launch_int 1 ] in
  let local_size = ref (Some [ 1; 1; 1 ]) in
  let collect_buffer_slot target u =
    match program_index_buffer u with
    | Some buf when Ops.equal (op buf) Ops.Param ->
        (match arg buf with
         | Arg.Param_arg p -> target := p.slot :: !target
         | _ -> ())
    | Some _ | None -> ()
  in
  let update_special name size =
    match special_axis_with_prefix name "idx" with
    | Some axis ->
        local_size := None;
        global_size :=
          set_nth "program_info_from_sink" !global_size axis
            (launch_dim_of_uop size)
    | None ->
        match special_axis_with_prefix name "gidx" with
        | Some axis ->
            global_size :=
              set_nth "program_info_from_sink" !global_size axis
                (launch_dim_of_uop size)
        | None ->
            match special_axis_with_prefix name "lidx" with
            | None -> ()
            | Some axis ->
                (match !local_size with
                 | None -> ()
                 | Some local ->
                     local_size :=
                       Some
                         (set_nth "program_info_from_sink" local axis
                            (exact_int_for_program "program_info_from_sink" size)))
  in
  List.iter
    (fun u ->
      (match op u, arg u with
       | Ops.Param, Arg.Param_arg p when p.addrspace = Dtype.Alu ->
           vars := u :: !vars
       | Ops.Param, Arg.Param_arg p ->
           globals := p.slot :: !globals
       | Ops.Store, _ ->
           let srcs = src u in
           if Array.length srcs > 0 then collect_buffer_slot outs srcs.(0)
       | Ops.Load, _ ->
           let srcs = src u in
           if Array.length srcs > 0 then collect_buffer_slot ins srcs.(0)
       | Ops.Special, _ -> (
           match as_special u with
           | Some { name; size } -> update_special name size
           | None -> ())
       | _ -> ());
      match op u, arg u with
      | Ops.Param, Arg.Param_arg { addrspace = Dtype.Alu; name = Some "core_id"; _ }
        ->
          global_size :=
            set_nth "program_info_from_sink" !global_size 0
              (Launch_int (vmax u + 1))
      | _ -> ())
    (toposort sink);
  let name =
    match as_kernel_info sink with
    | Some info -> info.name
    | None -> "test"
  in
  {
    name;
    global_size = !global_size;
    local_size = !local_size;
    vars = sort_program_vars !vars;
    globals = sort_uniq_ints !globals;
    outs = sort_uniq_ints !outs;
    ins = sort_uniq_ints !ins;
    aux;
  }

let int_floor_div a b =
  let q = a / b and r = a mod b in
  if r <> 0 && ((a < 0) <> (b < 0)) then q - 1 else q

let int_floor_mod a b = a - (int_floor_div a b * b)

let rec infer_int var_vals u =
  match const_int_value (simplify u) with
  | Some n -> n
  | None ->
      let srcs = src u in
      let binary f =
        if Array.length srcs < 2 then raise Not_found;
        f (infer_int var_vals srcs.(0)) (infer_int var_vals srcs.(1))
      in
      match op u with
      | Ops.Param -> (
          match program_var_name u with
          | Some name -> List.assoc name var_vals
          | None -> raise Not_found)
      | Ops.Bind when Array.length srcs >= 2 -> infer_int var_vals srcs.(1)
      | Ops.Cast when Array.length srcs >= 1 -> infer_int var_vals srcs.(0)
      | Ops.Add -> binary ( + )
      | Ops.Sub -> binary ( - )
      | Ops.Mul -> binary ( * )
      | Ops.Cdiv -> binary ( / )
      | Ops.Cmod -> binary ( mod )
      | Ops.Floordiv -> binary int_floor_div
      | Ops.Floormod -> binary int_floor_mod
      | Ops.Max -> binary max
      | Ops.Cmplt -> binary (fun a b -> if a < b then 1 else 0)
      | Ops.Cmpne -> binary (fun a b -> if a <> b then 1 else 0)
      | Ops.Cmpeq -> binary (fun a b -> if a = b then 1 else 0)
      | Ops.And -> binary ( land )
      | Ops.Or -> binary ( lor )
      | Ops.Xor -> binary ( lxor )
      | Ops.Shl -> binary ( lsl )
      | Ops.Shr -> binary ( asr )
      | Ops.Neg when Array.length srcs >= 1 -> -infer_int var_vals srcs.(0)
      | Ops.Where when Array.length srcs >= 3 ->
          if infer_int var_vals srcs.(0) <> 0 then infer_int var_vals srcs.(1)
          else infer_int var_vals srcs.(2)
      | _ -> raise Not_found

let program_launch_dim var_vals = function
  | Launch_int n -> Launch_value_int n
  | Launch_float f -> Launch_value_float f
  | Launch_sym u -> Launch_value_int (infer_int var_vals u)

let program_launch_dims info ~var_vals =
  ( List.map (program_launch_dim var_vals) info.global_size,
    info.local_size )

let program_vals info ~var_vals =
  let runtimevars = program_runtimevars info in
  List.map
    (fun var ->
      match program_var_name var with
      | Some name when List.mem_assoc name runtimevars -> None
      | Some name -> Some (List.assoc name var_vals)
      | None -> raise Not_found)
    info.vars

let semantic_key root =
  let semantic_arg = function
    | Arg.Call_info info -> Arg.Call_info { info with aux = None }
    | arg -> arg
  in
  let rec key u =
    let header =
      Hashtbl.hash (op u, dtype u, semantic_arg (arg u)) |> string_of_int
    in
    let children =
      Array.to_list (src u) |> List.map key |> String.concat ""
    in
    Digest.to_hex (Digest.string (header ^ children))
  in
  key root

(* Operators *)

module O = struct
  let ( + ) a b = alu_binary ~op:Ops.Add ~lhs:a ~rhs:b
  let ( * ) a b = alu_binary ~op:Ops.Mul ~lhs:a ~rhs:b
  let ( - ) a b = alu_binary ~op:Ops.Sub ~lhs:a ~rhs:b
  let ( / ) a b = alu_binary ~op:Ops.Fdiv ~lhs:a ~rhs:b
  let ( // ) a b = alu_binary ~op:Ops.Floordiv ~lhs:a ~rhs:b
  let ( mod ) a b = alu_binary ~op:Ops.Floormod ~lhs:a ~rhs:b
  let ( < ) a b = alu_binary ~op:Ops.Cmplt ~lhs:a ~rhs:b
  let cdiv a b = alu_binary ~op:Ops.Cdiv ~lhs:a ~rhs:b
  let cmod a b = alu_binary ~op:Ops.Cmod ~lhs:a ~rhs:b
  let floordiv a b = alu_binary ~op:Ops.Floordiv ~lhs:a ~rhs:b
  let floormod a b = alu_binary ~op:Ops.Floormod ~lhs:a ~rhs:b
  let ne a b = alu_binary ~op:Ops.Cmpne ~lhs:a ~rhs:b
  let where a b c = alu_ternary ~op:Ops.Where ~a ~b ~c
  let neg a = alu_unary ~op:Ops.Neg ~src:a
  let not_ a = ne a (const_bool true)
  let cast dt a = cast ~src:a ~dtype:dt
  let int_ n = const_int n
  let float_ x = const_float x
  let bool_ b = const_bool b
end

(* Formatting *)

let rec pp_uop fmt u =
  Format.fprintf fmt "%s:%s" (Ops.name (op u)) (Dtype.to_string (dtype u));
  let srcs = src u in
  if Array.length srcs > 0 then begin
    Format.fprintf fmt "(";
    Array.iteri (fun i s ->
      if i > 0 then Format.fprintf fmt ", ";
      pp_uop fmt s) srcs;
    Format.fprintf fmt ")"
  end

let pp = pp_uop
