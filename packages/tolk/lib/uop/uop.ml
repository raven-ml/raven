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
    | Split of { axis : int; amount : int; kind : Axis_type.t; top : bool }
    | Padto of { axis : int; amount : int }
    | Swap of { axis : int; with_axis : int }

  let to_string = function
    | Tc { axis; tc_select; tc_opt; use_tc } ->
        Printf.sprintf "TC:%d:%d:%d:%d" axis tc_select tc_opt use_tc
    | Split { axis; amount; kind; top } ->
        Printf.sprintf "SPLIT:%d:%d:%s:%b" axis amount (Axis_type.to_string kind) top
    | Padto { axis; amount } -> Printf.sprintf "PADTO:%d:%d" axis amount
    | Swap { axis; with_axis } -> Printf.sprintf "SWAP:%d:%d" axis with_axis

  let pp fmt t = Format.pp_print_string fmt (to_string t)

  let axis = function
    | Tc { axis; _ } | Split { axis; _ }
    | Padto { axis; _ } | Swap { axis; _ } -> axis

  let amount = function
    | Split { amount; _ } | Padto { amount; _ } -> Some amount
    | Tc _ | Swap _ -> None

  let with_amount t amount = match t with
    | Split r -> Split { r with amount }
    | Padto r -> Padto { r with amount }
    | (Tc _ | Swap _) as t -> t

end

type stage_opts = {
  device : device option;
  addrspace : Dtype.addr_space;
  removable : bool;
}

type metadata = { name : string; backward : bool }

type param_arg = {
  slot : int;
  dtype : Dtype.t;
  size : int option;
  image : (int * int) option;
  vmin_vmax : (Bound.t * Bound.t) option;
  multiple_of : int option;
  name : string option;
  addrspace : Dtype.addr_space;
  axis : int option;
  device : device option;
  volatile : bool;
  bind_on_realize : bool;
  buffer : Storage.t list option;
}

type reduce_arg = { op : Ops.t; num_axes : int }

type estimate = Int of int | Sym of t

and estimates = { ops : estimate; lds : estimate; mem : estimate }

and kernel_info = {
  name : string;
  applied_opts : Opt.t list;
  opts_to_apply : Opt.t list option;
  estimates : estimates option;
  beam : int;
}

and grad_fxn = grad_output:t -> call:t -> t option list

and call_info = {
  grad_fxn : grad_fxn option;
  name : string option;
  precompile : bool;
  precompile_backward : bool;
  aux : string option;
  dtype : Dtype.t;
}

and launch_dim = Launch_int of int | Launch_float of float | Launch_sym of t

and launch_value = Launch_value_int of int | Launch_value_float of float

and program_info = {
  target : Target.t;
  global_size : launch_dim list;
  local_size : launch_dim list;
  vars : t list;
  globals : int list;
  outs : int list;
  ins : int list;
}

and wmma_info = {
  dims : int * int * int;
  dtype_in : Dtype.t;
  device : string;
  threads : int;
  tc_upcast_axes :
    ((int list * int) list * (int list * int) list * (int list * int) list) option;
}

and arg =
  | Empty
  | Int of int
  | Ints of int list
  | Bools of bool list
  | Dtype of Dtype.t
  | Typed of string * Dtype.t
  | String of string
  | Value of Const.t
  | Op of Ops.t
  | Range_info of { axis : int; sub : int list; kind : Axis_type.t }
  | Param_arg of param_arg
  | Reduce_arg of reduce_arg
  | Device of device
  | Op_device of Ops.t * device
  | Stage_info of stage_opts
  | Kernel_info of kernel_info
  | Call_info of call_info
  | Program_info of program_info
  | Wmma_info of wmma_info

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

module Ref_key = struct
  type nonrec t = t
  let equal = ( == )
  let hash u = u.Hashcons.tag
end

module Ref_tbl = Hashtbl.Make (Ref_key)
module Weak_tbl = Ephemeron.K1.Make (Ref_key)

(* Arg module: re-export [arg] with its constructors under the [Arg.t]
   name so callers can write [Uop.Arg.Int 5] or [match a with Uop.Arg.Empty
   -> ...]. *)
module Arg = struct
  type nonrec t = arg =
    | Empty
    | Int of int
    | Ints of int list
    | Bools of bool list
    | Dtype of Dtype.t
    | Typed of string * Dtype.t
    | String of string
    | Value of Const.t
    | Op of Ops.t
    | Range_info of { axis : int; sub : int list; kind : Axis_type.t }
    | Param_arg of param_arg
    | Reduce_arg of reduce_arg
    | Device of device
    | Op_device of Ops.t * device
    | Stage_info of stage_opts
    | Kernel_info of kernel_info
    | Call_info of call_info
    | Program_info of program_info
    | Wmma_info of wmma_info

  let equal a b = match a, b with
    | Empty, Empty -> true
    | Int x, Int y -> x = y
    | Ints x, Ints y -> x = y
    | Bools x, Bools y -> x = y
    | Dtype x, Dtype y -> Dtype.equal x y
    | Typed (x, dx), Typed (y, dy) -> String.equal x y && Dtype.equal dx dy
    | String x, String y -> String.equal x y
    | Value x, Value y -> Const.equal x y
    | Op x, Op y -> Ops.equal x y
    | Range_info a, Range_info b ->
        a.axis = b.axis && a.sub = b.sub && Axis_type.equal a.kind b.kind
    | Param_arg x, Param_arg y ->
        { x with buffer = None } = { y with buffer = None }
        && Option.equal (List.equal (fun a b -> Storage.id a = Storage.id b))
             x.buffer y.buffer
    | Reduce_arg x, Reduce_arg y ->
        Ops.equal x.op y.op && x.num_axes = y.num_axes
    | Device x, Device y -> x = y
    | Op_device (opx, dx), Op_device (opy, dy) ->
        Ops.equal opx opy && dx = dy
    | Stage_info x, Stage_info y -> x = y
    | Kernel_info x, Kernel_info y -> x = y
    | Call_info x, Call_info y ->
        (match x.grad_fxn, y.grad_fxn with
         | Option.None, Option.None -> true
         | Option.Some a, Option.Some b -> a == b
         | _ -> false)
        && x.name = y.name
        && x.precompile = y.precompile
        && x.precompile_backward = y.precompile_backward
        && x.aux = y.aux
        && Dtype.equal x.dtype y.dtype
    | Program_info x, Program_info y -> x = y
    | Wmma_info x, Wmma_info y -> x = y
    | _ -> false

  let identity = function
    | Param_arg p ->
        (Param_arg { p with buffer = None },
         Option.map (List.map Storage.id) p.buffer)
    | arg -> (arg, None)

  let compare a b = Stdlib.compare (identity a) (identity b)
  let hash arg = Hashtbl.hash (identity arg)

  let as_int = function Int n -> Option.Some n | _ -> Option.None
  let as_ints = function Ints l -> Option.Some l | _ -> Option.None
  let as_bools = function Bools l -> Option.Some l | _ -> Option.None
  let as_string = function String s | Typed (s, _) -> Option.Some s | _ -> Option.None
  let as_value = function Value v -> Option.Some v | _ -> Option.None
  let as_op = function Op o -> Option.Some o | _ -> Option.None
  let as_param_arg = function Param_arg a -> Option.Some a | _ -> Option.None
  let as_reduce_arg = function
    | Reduce_arg a -> Option.Some a
    | _ -> Option.None
  let as_device = function Device d -> Option.Some d | _ -> Option.None
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
    h := !h * 31 + Arg.hash n.arg;
    h := !h * 31 + Hashtbl.hash n.node_tag;
    !h land max_int
end

module H = Hashcons.Make (Node_hashed)

let global_table = H.create 4096

(* Hash-consing and node metadata are global mutable state: beam search can
   compile candidates in parallel domains, so every access takes this lock.
   No other [intern_mutex] section nests inside one, and it is the only
   lock in this module, so there is no lock ordering to deadlock on. *)
let intern_mutex = Mutex.create ()

let derived_dtype (node : node) =
  let dt (u : t) = u.Hashcons.node.dtype in
  let source i =
    if i >= Array.length node.src then
      invalid_arg ("Uop: missing source for " ^ Ops.name node.op);
    node.src.(i)
  in
  let first () = dt (source 0) in
  let promote sources = match sources with
    | [] -> Dtype.void
    | first :: rest ->
        let dtype = dt first in
        if List.for_all (fun u -> Dtype.equal (dt u) dtype) rest then dtype
        else Dtype.least_upper_dtype (List.map dt sources)
  in
  let rec invalid u =
    let n = u.Hashcons.node in
    match n.op, n.arg with
    | Ops.Const, Arg.Value c -> Const.view c = Const.Invalid
    | op, _ when (Ops.Group.is_movement op || op = Ops.Detach)
                 && Array.length n.src > 0 -> invalid n.src.(0)
    | _ -> false
  in
  match node.op with
  | Ops.Store | Ops.Linear | Ops.Sink | Ops.Program | Ops.Source
  | Ops.Backedge | Ops.Barrier | Ops.Group | Ops.If | Ops.Endif | Ops.Noop
  | Ops.Custom_function | Ops.Rewrite_error | Ops.Pyliteral | Ops.Wait -> Dtype.void
  | Ops.Call ->
      (match node.arg with Arg.Call_info info -> info.dtype | _ -> Dtype.void)
  | Ops.Custom | Ops.Customi | Ops.Ins ->
      (match node.arg with
       | Arg.Typed (_, dtype) -> dtype
       | _ -> invalid_arg "Uop: custom instructions require a typed payload")
  | Ops.Cast | Ops.Bitcast ->
      (match node.arg with
       | Arg.Dtype dtype -> dtype
       | _ -> invalid_arg "Uop: casts require a dtype payload")
  | Ops.Buffer | Ops.Alloc | Ops.Param ->
      (match node.arg with
       | Arg.Param_arg p -> p.dtype
       | _ -> invalid_arg "Uop: storage requires ParamArg")
  | Ops.Const ->
      (match node.arg with
       | Arg.Value value -> (match Const.view value with
           | Const.Bool _ | Const.Invalid -> Dtype.bool
           | Const.Int _ -> Dtype.weakint
           | Const.Float _ -> Dtype.weakfloat)
       | _ -> invalid_arg "Uop: CONST requires a scalar value")
  | Ops.Binary -> Dtype.uint8
  | Ops.Cmplt | Ops.Cmpne | Ops.Cmpeq -> Dtype.bool
  | Ops.Getaddr | Ops.Threefry -> Dtype.uint64
  | Ops.Sin | Ops.Log2 | Ops.Exp2 | Ops.Sqrt | Ops.Reciprocal ->
      if invalid (source 0) then Dtype.bool else Dtype.least_upper_float (first ())
  | Ops.Fdiv -> Dtype.least_upper_float (promote (Array.to_list node.src))
  | Ops.Shl | Ops.Shr ->
      if not (Array.for_all (fun u -> Dtype.is_int (dt u) || invalid u) node.src) then
        invalid_arg "Uop: shift operands must be integers";
      first ()
  | Ops.Where ->
      if not (Dtype.equal (first ()) Dtype.bool) then
        invalid_arg "Uop: WHERE condition must be bool";
      promote (List.tl (Array.to_list node.src))
  | Ops.Stack -> promote (Array.to_list node.src)
  | Ops.Wmma -> dt (source 2)
  | Ops.Index ->
      (match (source 0).Hashcons.node.arg with
       | Arg.Param_arg { image = Some _; _ } -> Dtype.float32
       | _ -> first ())
  | Ops.Load | Ops.Unshard | Ops.Reduce | Ops.After | Ops.Range | Ops.Copy
  | Ops.Stage | Ops.Detach | Ops.Mstack | Ops.Mselect | Ops.Allreduce | Ops.Special
  | Ops.End | Ops.Contiguous_backward -> first ()
  | op when Ops.Group.is_unary op || Ops.Group.is_movement op -> first ()
  | op when Ops.Group.is_broadcastable op -> promote (Array.to_list node.src)
  | op -> invalid_arg ("Uop: no dtype rule for " ^ Ops.name op)

let intern_node (node : node) =
  let node = { node with dtype = derived_dtype node } in
  Mutex.protect intern_mutex (fun () -> H.hashcons global_table node)

let side_metadata : metadata list Weak_tbl.t = Weak_tbl.create 64

let default_param_arg ~dtype ?size ?image ?vmin_vmax ?multiple_of ?name
    ?(addrspace = Dtype.Global) ?axis ?device ?(volatile = false) slot =
  { slot; dtype; size; image; vmin_vmax; multiple_of; name; addrspace; axis;
    device; volatile; bind_on_realize = false; buffer = None }

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

(* Accessors *)

let op u = u.Hashcons.node.op
let dtype (u : t) = u.Hashcons.node.dtype
let src u = u.Hashcons.node.src
let arg u = u.Hashcons.node.arg
let node_tag u = u.Hashcons.node.node_tag
let tag u = u.Hashcons.tag
let metadata u =
  Mutex.protect intern_mutex (fun () ->
      Option.value (Weak_tbl.find_opt side_metadata u) ~default:[])

let with_metadata md u =
  Mutex.protect intern_mutex (fun () ->
      Weak_tbl.replace side_metadata u md;
      u)

let children u = Array.to_list (src u)
let equal a b = a.Hashcons.tag = b.Hashcons.tag
let compare a b = Int.compare a.Hashcons.tag b.Hashcons.tag

(* The set of ops appearing among a node's direct children, deduplicated.
   Memoised across rewrite passes: pattern-matcher early-reject consults it on
   every candidate, and nodes are immutable so the set is stable.

   Domain-local, like every per-node memo cache in this module and its
   consumers: each memoizes a pure function of an immutable hash-consed node,
   so beam-search workers compiling candidates in parallel domains fill their
   own tables instead of racing on one unsynchronized Hashtbl. *)
let child_ops_cache : Ops.t list Weak_tbl.t Domain.DLS.key =
  Domain.DLS.new_key (fun () -> Weak_tbl.create 1024)

let child_ops u =
  let child_ops_cache = Domain.DLS.get child_ops_cache in
  match Weak_tbl.find_opt child_ops_cache u with
  | Some ops -> ops
  | None ->
      let ops =
        Array.fold_left
          (fun acc s ->
            let o = op s in
            if List.exists (Ops.equal o) acc then acc else o :: acc)
          [] (src u)
      in
      Weak_tbl.add child_ops_cache u ops;
      ops

let integer_as_native n = if Z.fits_int n then Some (Z.to_int n) else None

let program_var_name u =
  match op u, arg u with
  | (Ops.Param | Ops.Buffer), Arg.Param_arg { name = Some name; _ } -> Some name
  | _ -> None

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
type reduce_view = { src : t; ranges : t list; op : Ops.t; num_axes : int }
type allreduce_view = { src : t; device : device; op : Ops.t }
type stage_view = { src : t; ranges : t list; opts : stage_opts }
type param_view = { param : param_arg; shape : t }
type buffer_view = { buffer : param_arg; shape : t }
type wmma_view = { a : t; b : t; c : t; info : wmma_info }
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

let axis_id u =
  match op u, arg u with
  | Ops.Range, Arg.Range_info { axis; sub; _ } -> axis :: sub
  | _ -> invalid_arg "Uop.axis_id: expected RANGE"

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
  | Ops.Reduce, Arg.Reduce_arg { op; num_axes }, src :: ranges ->
      Option.Some { src; ranges; op; num_axes }
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

let as_wmma u =
  match op u, arg u, Array.to_list (src u) with
  | Ops.Wmma, Arg.Wmma_info info, [ a; b; c ] ->
      Option.Some { a; b; c; info }
  | _ -> Option.None

let as_call u =
  match op u, arg u, Array.to_list (src u) with
  | Ops.Call, Arg.Call_info info, body :: args ->
      Option.Some { body; args; info }
  | _ -> Option.None

let as_special u =
  match op u, arg u, Array.to_list (src u) with
  | Ops.Special, Arg.String name, [ size ] -> Some { name; size }
  | _ -> Option.None

let is_variable u =
  match op u, arg u, src u with
  | Ops.Buffer, Arg.Param_arg { addrspace = Dtype.Alu; size = None;
      vmin_vmax = Some _; _ }, [||] -> true
  | _ -> false

let as_bind u =
  match op u, src u with
  | Ops.After, [| var; store |] when is_variable var ->
      (match op store, src store with
       | Ops.Store, [| dst; value |] when dst == var && op value = Ops.Const ->
           Some { var; value }
       | _ -> None)
  | _ -> None

let is_bound_var u = Option.is_some (as_bind u)

let device_cache : device option Weak_tbl.t Domain.DLS.key =
  Domain.DLS.new_key (fun () -> Weak_tbl.create 32)

let rec device_of u =
  let device_cache = Domain.DLS.get device_cache in
  match Weak_tbl.find_opt device_cache u with
  | Some d -> d
  | None ->
      let d = compute_device u in
      Weak_tbl.add device_cache u d;
      d

and compute_device u =
  let children = src u in
  match op u with
  | Ops.Stage ->
      (match Arg.as_stage_info (arg u) with
       | Some opts -> opts.device
       | None when Array.length children > 0 -> device_of children.(0)
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
  | Ops.Param | Ops.Buffer | Ops.Alloc ->
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

let on_disk u = match device_of u with
  | Some (Single device) -> String.starts_with ~prefix:"DISK" device
  | _ -> false

(* No device is nowhere to store; a weak dtype is no width to store. Either
   way the value cannot back a buffer as it stands. *)
let is_virtual u =
  Option.is_none (device_of u) || Dtype.is_weak (dtype u)

let as_kernel_info u =
  match op u, arg u with
  | Ops.Sink, Arg.Kernel_info ki -> Option.Some ki
  | _ -> Option.None

let as_call_info u =
  match op u, arg u with
  | Ops.Call, Arg.Call_info info -> Option.Some info
  | _ -> Option.None

let as_program_info u =
  match op u, arg u with
  | Ops.Program, Arg.Program_info info -> Option.Some info
  | _ -> Option.None

(* Raw constructor *)

let mk ~op ~dtype ~src ~arg =
  intern_node { op; dtype; src; arg; node_tag = Option.None }

(* Smart constructors *)

let void_dtype = Dtype.void

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

(* Buffer slots come from one process-wide counter: buffers hash-cons on
   (slot, dtype, shape, device), so reusing a slot would collapse two distinct
   allocations onto one node identity. *)
let next_buffer_slot = ref 0

let fresh_buffer_slot () =
  let s = !next_buffer_slot in
  incr next_buffer_slot;
  s

let reserve_buffer_slots n =
  if n > !next_buffer_slot then next_buffer_slot := n

let stage ~src ~ranges ~opts =
  mk ~op:Ops.Stage ~dtype:(dtype src)
    ~src:(Array.of_list (src :: ranges))
    ~arg:(Arg.Stage_info opts)

let variable ~name ~min_val ~max_val ?(dtype = Dtype.weakint)
    ?(multiple_of = 1) ?(param = false) () =
  if min_val > max_val || multiple_of <= 0 then
    invalid_arg "Uop.variable: invalid bounds or divisor";
  mk ~op:(if param then Ops.Param else Ops.Buffer) ~dtype ~src:[||]
    ~arg:(Arg.Param_arg (default_param_arg ~dtype ~name
      ~vmin_vmax:(Bound.int min_val, Bound.int max_val) ~multiple_of
      ~addrspace:Dtype.Alu (-1)))


let cast ~src ~dtype:target_dtype =
  if Dtype.equal (dtype src) target_dtype then src
  else mk ~op:Ops.Cast ~dtype:target_dtype ~src:[| src |] ~arg:(Arg.Dtype target_dtype)


let const v =
  let target = Const.dtype v in
  let weak = match Const.view v with
    | Const.Int _ -> Dtype.weakint
    | Const.Float _ -> Dtype.weakfloat
    | Const.Bool _ | Const.Invalid -> Dtype.bool in
  let value = Const.of_view weak (Const.view v) in
  let node = mk ~op:Ops.Const ~dtype:weak ~src:[||] ~arg:(Arg.Value value) in
  cast ~src:node ~dtype:target

let as_const u =
  match op u, arg u, src u with
  | Ops.Const, Arg.Value value, _ -> Some value
  | Ops.Cast, _, [| value |] when op value = Ops.Const ->
      (match arg value with
       | Arg.Value c -> Some (Const.of_view (dtype u) (Const.view c))
       | _ -> None)
  | _ -> None

let ccast ~src ~dtype =
  match op src, arg src with
  | Ops.Const, Arg.Value value -> const (Const.of_view dtype (Const.view value))
  | _ -> cast ~src ~dtype

let cconst value dtype =
  let value = const value in
  let value = if op value = Ops.Cast then (src value).(0) else value in
  mk ~op:Ops.Cast ~dtype ~src:[| value |] ~arg:(Arg.Dtype dtype)

let bind ~var ~value =
  if not (is_variable var) then invalid_arg "Uop.bind: expected a variable";
  let value = match as_const value with
    | Some c ->
        let weak = match Const.view c with
          | Const.Int _ -> Dtype.weakint | Const.Float _ -> Dtype.weakfloat
          | Const.Bool _ | Const.Invalid -> Dtype.bool in
        const (Const.of_view weak (Const.view c))
    | None -> invalid_arg "Uop.bind: expected a constant value" in
  let p = match arg var with Arg.Param_arg p -> p | _ -> assert false in
  let c = match as_const value with Some c -> c | None -> assert false in
  let bound = match Const.view c with
    | Const.Int n -> `Int n | Const.Bool b -> `Bool b | Const.Float f -> `Float f
    | Const.Invalid -> invalid_arg "Uop.bind: invalid value" in
  let lo, hi = Option.get p.vmin_vmax in
  if not (Bound.le lo bound && Bound.le bound hi) then
    invalid_arg "Uop.bind: value outside variable bounds";
  (match Const.view c with
   | Const.Int n when not (Z.equal Z.zero
       (Z.rem n (Z.of_int (Option.value p.multiple_of ~default:1)))) ->
       invalid_arg "Uop.bind: value violates variable divisor"
   | _ -> ());
  let store = mk ~op:Ops.Store ~dtype:void_dtype ~src:[| var; value |] ~arg:Arg.Empty in
  mk ~op:Ops.After ~dtype:(dtype var) ~src:[| var; store |] ~arg:Arg.Empty


let invalid () = const Const.invalid
let const_int n = const (Const.int Dtype.weakint n)
let const_float x = const (Const.float Dtype.weakfloat x)
let const_bool b = const (Const.bool b)
let zero_like u = const (Const.of_scalar (dtype u) (`Int 0L))
let const_like u n = const (Const.of_scalar (dtype u) (`Int (Int64.of_int n)))

let index ~ptr ~idxs () =
  let const_int_value u =
    match as_const u with
    | Some c -> (
        match Const.view c with
        | Const.Int n
          when Z.fits_int n -> Some (Z.to_int n)
        | Const.Int _ | Const.Bool _ | Const.Float _ | Const.Invalid -> None)
    | _ -> None
  in
  (* A buffer or vector indexes to a scalar element, whose dtype is the same
     scalar dtype the source already carries. A constant index into a stack
     selects the lane directly. *)
  match idxs, op ptr with
  | [ idx ], Ops.Stack -> (
      match const_int_value idx with
      | Some i when i >= 0 && i < Array.length (src ptr) -> (src ptr).(i)
      | Some _ | None ->
          mk ~op:Ops.Index ~dtype:(dtype ptr) ~src:[| ptr; idx |] ~arg:Arg.Empty)
  | _ ->
      mk ~op:Ops.Index ~dtype:(dtype ptr) ~src:(Array.of_list (ptr :: idxs))
        ~arg:Arg.Empty

let storage_shape (p : param_arg) =
  match p.image, p.size with
  | Some (h, w), _ -> [const_int h; const_int w; const_int 4]
  | None, Some size -> [const_int size]
  | None, None -> []

let storage_shape_arg p = match storage_shape p with
  | [dim] -> dim
  | dims -> mk ~op:Ops.Stack ~dtype:(if dims = [] then Dtype.void else Dtype.weakint)
      ~src:(Array.of_list dims) ~arg:Arg.Empty

let as_param u =
  match op u, arg u, Array.length (src u) with
  | Ops.Param, Arg.Param_arg p, 0 -> Some { param = p; shape = storage_shape_arg p }
  | _ -> None

let as_buffer u =
  match op u, arg u, Array.length (src u) with
  | Ops.Buffer, Arg.Param_arg p, 0 -> Some { buffer = p; shape = storage_shape_arg p }
  | _ -> None

let load ~src ?dtype:load_dtype ?alt ?gate () =
  (* The indexed source already carries the element dtype. *)
  let dtype = match load_dtype with Some dtype -> dtype | None -> dtype src in
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

let promo_dtype srcs =
  match srcs with
  | [] -> invalid_arg "Uop.promo_dtype: no operands"
  | first :: rest ->
      let dt = dtype first in
      (* Identical operands settle without consulting the lattice, so ops on
         dtypes that have no upper bound with anything -- pointers, void --
         still pass their own dtype through. *)
      if List.for_all (fun u -> Dtype.equal (dtype u) dt) rest then dt
      else Dtype.least_upper_dtype (List.map dtype srcs)

let alu_unary ~op ~src =
  if not (Ops.Group.is_unary op) then
    invalid_arg
      (Printf.sprintf "Uop.alu_unary: %s is not unary" (Ops.name op));
  mk ~op ~dtype:void_dtype ~src:[| src |] ~arg:Arg.Empty

let alu_binary ~op ~lhs ~rhs =
  if not (Ops.Group.is_binary op) then
    invalid_arg
      (Printf.sprintf "Uop.alu_binary: %s is not binary" (Ops.name op));
  mk ~op ~dtype:void_dtype ~src:[| lhs; rhs |] ~arg:Arg.Empty

let alu_ternary ~op ~a ~b ~c =
  if not (Ops.Group.is_ternary op) then
    invalid_arg
      (Printf.sprintf "Uop.alu_ternary: %s is not ternary" (Ops.name op));
  mk ~op ~dtype:void_dtype ~src:[| a; b; c |] ~arg:Arg.Empty

let valid ~src ~cond =
  let inv = const Const.invalid in
  alu_ternary ~op:Ops.Where ~a:cond ~b:src ~c:inv


let bitcast ~src ~dtype:target_dtype =
  if Dtype.equal (dtype src) target_dtype then src
  else mk ~op:Ops.Bitcast ~dtype:target_dtype ~src:[| src |] ~arg:(Arg.Dtype target_dtype)

let stack ?dtype:dtype_opt srcs =
  let dt = match srcs with
    | [] -> void_dtype
    | _ -> Option.value dtype_opt ~default:(promo_dtype srcs)
  in
  let src = List.map (fun u ->
      match op u, arg u with
      | Ops.Const, Arg.Value c when Const.view c = Const.Invalid -> u
      | _ -> cast ~src:u ~dtype:dt) srcs in
  mk ~op:Ops.Stack ~dtype:dt ~src:(Array.of_list src) ~arg:Arg.Empty

let getaddr ?device ~src () =
  let arg = match device with None -> Arg.Empty | Some d -> Arg.Device (Single d) in
  mk ~op:Ops.Getaddr ~dtype:Dtype.uint64 ~src:[| src |] ~arg

let broadcast u n =
  if n <= 1 then u
  else
    let srcs = List.init n (fun _ -> u) in
    stack srcs

let is_invalid_const u =
  match op u, arg u with
  | Ops.Const, Arg.Value c -> (
      match Const.view c with Const.Invalid -> true | _ -> false)
  | _ -> false

(* An index expression may be gated as [where cond idx invalid]; get_idx
   recovers the index and get_valid the guard. Both recurse through stacked
   lanes and require an integer index. *)
let rec get_idx u =
  if not (Dtype.is_int (dtype u)) then
    invalid_arg "Uop.get_idx: expected an integer index expression";
  match op u with
  | Ops.Stack -> stack (List.map get_idx (children u))
  | Ops.Where
    when Array.length (src u) = 3 && is_invalid_const (src u).(2) ->
      (src u).(1)
  | _ -> u

let rec get_valid u =
  if not (Dtype.is_int (dtype u)) then
    invalid_arg "Uop.get_valid: expected an integer index expression";
  match op u with
  | Ops.Stack -> stack (List.map get_valid (children u))
  | Ops.Where
    when Array.length (src u) = 3 && is_invalid_const (src u).(2) ->
      (src u).(0)
  | Ops.Const when is_invalid_const u -> const_bool false
  | _ -> const_bool true

let range ~size ~axis ~kind ?(sub = []) ?(dtype = Dtype.weakint)
    ?(parents = []) () =
  mk ~op:Ops.Range ~dtype
    ~src:(Array.of_list (size :: parents))
    ~arg:(Arg.Range_info { axis; sub; kind })

let loop ~axis =
  range ~size:(noop ~dtype:Dtype.void ()) ~axis ~kind:Axis_type.Weak
    ~dtype:Dtype.void ()

let backedge ~body ~loop ~cond =
  mk ~op:Ops.Backedge ~dtype:Dtype.void ~src:[| body; loop; cond |]
    ~arg:Arg.Empty

let end_ ~value ~ranges =
  if ranges = [] then value
  else
    mk ~op:Ops.End ~dtype:void_dtype
      ~src:(Array.of_list (value :: ranges)) ~arg:Arg.Empty

let if_ ~cond ~idx_for_dedup =
  mk ~op:Ops.If ~dtype:void_dtype
    ~src:[| cond; idx_for_dedup |] ~arg:Arg.Empty

let endif ~if_ =
  mk ~op:Ops.Endif ~dtype:void_dtype ~src:[| if_ |] ~arg:Arg.Empty

let barrier ?(srcs = []) () =
  mk ~op:Ops.Barrier ~dtype:void_dtype
    ~src:(Array.of_list srcs) ~arg:Arg.Empty

let special ~name ~size ?(dtype = Dtype.weakint) () =
  mk ~op:Ops.Special ~dtype
    ~src:[| cast ~src:size ~dtype |] ~arg:(Arg.String name)

let reduce ~src ~ranges ~op ~dtype =
  mk ~op:Ops.Reduce ~dtype
    ~src:(Array.of_list (src :: ranges))
    ~arg:(Arg.Reduce_arg { op; num_axes = 0 })

let allreduce ~src ~device ~op =
  mk ~op:Ops.Allreduce ~dtype:(dtype src)
    ~src:[| src |] ~arg:(Arg.Op_device (op, device))

let multi ~src ~axis =
  mk ~op:Ops.Unshard ~dtype:(dtype src) ~src:[| src |] ~arg:(Arg.Int axis)

let mstack srcs =
  let dt = match srcs with
    | s :: _ -> dtype s
    | [] -> invalid_arg "Uop.mstack: empty"
  in
  mk ~op:Ops.Mstack ~dtype:dt ~src:(Array.of_list srcs) ~arg:Arg.Empty

let mselect ~src ~index =
  mk ~op:Ops.Mselect ~dtype:(dtype src) ~src:[| src |] ~arg:(Arg.Int index)

let copy ~src ~device () =
  let disk = String.starts_with ~prefix:"DISK" in
  if (match device with Single d -> disk d | Multi ds -> List.exists disk ds | Index _ -> false) then
    invalid_arg "Uop.copy: disk destinations require an explicit store";
  if Dtype.is_weak (dtype src) then
    invalid_arg "Uop.copy: storage requires a concrete dtype";
  mk ~op:Ops.Copy ~dtype:(dtype src) ~src:[| src |] ~arg:(Arg.Device device)

let rec base u =
  let srcs = src u in
  match op u with
  | op when Ops.Group.is_movement op ->
      if Array.length srcs = 0 then u else base srcs.(0)
  | Ops.Detach ->
      (* DETACH can't change base. MULTI is its own base. *)
      if Array.length srcs = 0 then u else base srcs.(0)
  | _ -> u

let rec storage_base u =
  let b = base u in
  match op b with
  | Ops.Bitcast | Ops.After | Ops.Unshard -> storage_base (src b).(0)
  | _ -> b

(* Memoized: an unmemoized walk revisits shared subgraphs and goes
   exponential on wide unrolled ALU chains. *)
let addrspace_cache : Dtype.addr_space option Weak_tbl.t Domain.DLS.key =
  Domain.DLS.new_key (fun () -> Weak_tbl.create 256)

let rec addrspace u =
  let addrspace_cache = Domain.DLS.get addrspace_cache in
  match Weak_tbl.find_opt addrspace_cache u with
  | Some a -> a
  | None ->
      let a = compute_addrspace u in
      Weak_tbl.add addrspace_cache u a;
      a

and compute_addrspace u =
  let srcs = src u in
  match op u with
  | Ops.Param | Ops.Buffer | Ops.Alloc ->
      (match Arg.as_param_arg (arg u) with
       | Some param -> Some param.addrspace
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
  | Ops.Buffer | Ops.Alloc | Ops.Param -> u
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
          | Ops.Buffer | Ops.Alloc | Ops.Param | Ops.Stage | Ops.Mstack -> s
          | _ -> if Array.length srcs = 0 then s else walk srcs.(0)
        in
        walk u

let rec has_buffer_identity ?(after_ok = false) u =
  let srcs = src u in
  match op u with
  | Ops.Reshape | Ops.Unshard | Ops.Mselect ->
      Array.length srcs > 0 && has_buffer_identity ~after_ok srcs.(0)
  | Ops.After when after_ok ->
      Array.length srcs > 0 && has_buffer_identity ~after_ok srcs.(0)
  | Ops.Buffer | Ops.Alloc | Ops.Param -> true
  | _ -> false

let expand ~src ~dims =
  (* EXPAND prepends [dims] as new leading axes; expanding by an empty shape
     (an empty stack) is a no-op. *)
  if op dims = Ops.Stack && children dims = [] then src
  else mk ~op:Ops.Expand ~dtype:(dtype src) ~src:[| src; dims |] ~arg:Arg.Empty

let pad ~src ~offset ~size =
  mk ~op:Ops.Pad ~dtype:(dtype src)
    ~src:[| src; offset; size |] ~arg:Arg.Empty

let shrink ~src ~offset ~size =
  mk ~op:Ops.Shrink ~dtype:(dtype src)
    ~src:[| src; offset; size |] ~arg:Arg.Empty

let permute ~src ~order =
  if List.mapi (fun i o -> i = o) order |> List.for_all Fun.id then src
  else
    mk ~op:Ops.Permute ~dtype:(dtype src) ~src:[| src |]
      ~arg:(Arg.Ints order)

let flip ~src ~dims =
  mk ~op:Ops.Flip ~dtype:(dtype src) ~src:[| src |]
    ~arg:(Arg.Bools dims)

let detach ~src =
  mk ~op:Ops.Detach ~dtype:(dtype src) ~src:[| src |] ~arg:Arg.Empty

let contiguous ~src ?(force = false) () =
  if not force && (is_virtual src || has_buffer_identity src ||
      (op src = Ops.Stage && arg src = Arg.Empty)) then src
  else mk ~op:Ops.Stage ~dtype:(dtype src) ~src:[|src|] ~arg:Arg.Empty

let contiguous_backward ~src =
  mk ~op:Ops.Contiguous_backward ~dtype:(dtype src) ~src:[| src |]
    ~arg:Arg.Empty

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

let wmma ~a ~b ~c ~info ~dtype =
  mk ~op:Ops.Wmma ~dtype ~src:[| a; b; c |] ~arg:(Arg.Wmma_info info)

let custom ~fmt ~args =
  mk ~op:Ops.Custom ~dtype:void_dtype
    ~src:(Array.of_list args) ~arg:(Arg.Typed (fmt, void_dtype))

let custom_inline ~fmt ~args ~dtype =
  mk ~op:Ops.Customi ~dtype
    ~src:(Array.of_list args) ~arg:(Arg.Typed (fmt, dtype))

let source s =
  mk ~op:Ops.Source ~dtype:void_dtype ~src:[||] ~arg:(Arg.String s)

let binary s =
  mk ~op:Ops.Binary ~dtype:Dtype.uint8 ~src:[||] ~arg:(Arg.String s)

let rewrite_error ~src ~msg =
  mk ~op:Ops.Rewrite_error ~dtype:void_dtype ~src ~arg:(Arg.String msg)

let ins ~mnemonic ~operands ?(dtype = void_dtype) () =
  mk ~op:Ops.Ins ~dtype
    ~src:(Array.of_list operands) ~arg:(Arg.Typed (mnemonic, dtype))

let custom_function ~name ~srcs =
  mk ~op:Ops.Custom_function ~dtype:void_dtype
    ~src:(Array.of_list srcs) ~arg:(Arg.String name)

(* Replace / with_tag *)

let replace u ?op:op_opt ?src:src_opt ?arg:arg_opt
    ?node_tag:node_tag_opt () =
  let n : node = u.Hashcons.node in
  let op = Option.value op_opt ~default:n.op in
  let src = Option.value src_opt ~default:n.src in
  let arg = Option.value arg_opt ~default:n.arg in
  let node_tag = Option.value node_tag_opt ~default:n.node_tag in
  intern_node { op; src; arg; dtype = Dtype.void; node_tag }

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
              | Ops.Call -> false
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

let topovisit visitor cache root =
  let stack = Stack.create () in
  Stack.push (root, false) stack;
  while not (Stack.is_empty stack) do
    let node, visited = Stack.pop stack in
    if Hashtbl.mem cache (tag node) then ()
    else if not visited then begin
      Stack.push (node, true) stack;
      let srcs = src node in
      for i = Array.length srcs - 1 downto 0 do
        Stack.push (srcs.(i), false) stack
      done
    end
    else Hashtbl.replace cache (tag node) (visitor node)
  done;
  Hashtbl.find cache (tag root)

let backward_slice_cache : (t list * unit Ref_tbl.t) Weak_tbl.t Domain.DLS.key =
  Domain.DLS.new_key (fun () -> Weak_tbl.create 64)

let backward_slice_property root =
  let cache = Domain.DLS.get backward_slice_cache in
  match Weak_tbl.find_opt cache root with
  | Some result -> result
  | None ->
      let nodes = List.filter (fun u -> not (u == root)) (toposort root) in
      let members = Ref_tbl.create (List.length nodes) in
      List.iter (fun u -> Ref_tbl.add members u ()) nodes;
      let result = nodes, members in
      Weak_tbl.add cache root result;
      result

let backward_slice root = fst (backward_slice_property root)

let find_nodes p root =
  List.filter p (toposort root)

let in_backward_slice needle haystack =
  Ref_tbl.mem (snd (backward_slice_property haystack)) needle

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
  | Ops.Linear -> Option.Some 0
  | Ops.Stage | Ops.Reduce | Ops.End | Ops.Call
  | Ops.Copy -> Option.Some 1
  | Ops.Wmma -> Option.Some 3
  | _ -> Option.None

module Ref_set = Set.Make (struct
  type nonrec t = t
  let compare a b = Int.compare a.Hashcons.tag b.Hashcons.tag
end)

let ranges_cache : Ref_set.t Weak_tbl.t Domain.DLS.key =
  Domain.DLS.new_key (fun () -> Weak_tbl.create 64)

let rec ranges_set u =
  let ranges_cache = Domain.DLS.get ranges_cache in
  match Weak_tbl.find_opt ranges_cache u with
  | Some set -> set
  | None ->
      let set = compute_ranges u in
      Weak_tbl.add ranges_cache u set;
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
  | Ops.Call
    when Array.length children > 0
         && op children.(0) = Ops.Custom_function
         && Array.length (src children.(0)) = 1 -> []
  | Ops.Backedge -> [ children.(1) ]
  | Ops.End ->
      Array.to_list children |> List.tl
      |> List.filter (fun r -> op r = Ops.Range)
  | Ops.Barrier -> Array.to_list children |> List.concat_map ended_ranges
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

(* Ordered variant of [ranges_set]; memoized like it, since an unmemoized
   walk revisits shared subgraphs and goes exponential on unrolled kernels. *)
(* Bool-typed nodes reachable from a node, itself included: a backward slice
   pruned to the only dtype a condition can have. Memoized, since the
   where-closure fold queries it on every WHERE and an unmemoized walk
   revisits shared subgraphs. *)
let bool_slice_cache : Ref_set.t Weak_tbl.t Domain.DLS.key =
  Domain.DLS.new_key (fun () -> Weak_tbl.create 256)

let rec bool_slice u =
  let bool_slice_cache = Domain.DLS.get bool_slice_cache in
  match Weak_tbl.find_opt bool_slice_cache u with
  | Some s -> s
  | None ->
      let acc = ref Ref_set.empty in
      Array.iter (fun c -> acc := Ref_set.union !acc (bool_slice c)) (src u);
      if Dtype.is_bool (dtype u) then acc := Ref_set.add u !acc;
      Weak_tbl.add bool_slice_cache u !acc;
      !acc

let bool_slice_mem root u = Ref_set.mem u (bool_slice root)

let ranges_list_cache : t list Weak_tbl.t Domain.DLS.key =
  Domain.DLS.new_key (fun () -> Weak_tbl.create 64)

let ranges u =
  let ranges_list_cache = Domain.DLS.get ranges_list_cache in
  let mem_ref x xs = List.exists (fun y -> y == x) xs in
  let add_unique_rev acc r = if mem_ref r acc then acc else r :: acc in
  let remove_many acc rs =
    List.filter (fun r -> not (mem_ref r rs)) acc
  in
  let rec ranges_list u =
    match Weak_tbl.find_opt ranges_list_cache u with
    | Some l -> l
    | None ->
        let l = compute_ranges_list u in
        Weak_tbl.add ranges_list_cache u l;
        l
  and compute_ranges_list u =
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
  | Ops.Sink | Ops.Program | Ops.Linear | Ops.Store
  | Ops.Custom_function -> true
  | _ -> false

let call ~body ~args ~info =
  (* Calls are launched per device, so an open device range may cross the
     call boundary. *)
  let is_device_range r =
    match as_range r with
    | Some { kind = Axis_type.Device; _ } -> true
    | _ -> false
  in
  if List.exists (fun r -> not (is_device_range r)) (ranges body) then
    invalid_arg "Uop.call: ranges are leaking out of the call body";
  if not (opaque_call_body (op body)) then
    invalid_arg "Uop.call: value-producing bodies require call_with_outputs";
  mk ~op:Ops.Call ~dtype:void_dtype
    ~src:(Array.of_list (body :: args))
    ~arg:(Arg.Call_info info)

(* Rewriting *)

let first_match rules n = List.find_map (fun r -> r n) rules

(* A node's rewrite runs in three steps held on an explicit stack: descend
   into its sources, rebuild it once they all have results, then link it to
   the result of rewriting the rebuilt node. A node whose sources are not all
   resolved parks on the one it is missing and resumes when that source
   resolves, rather than recursing into it: a source reached this way is
   already scheduled further down the stack, and recursing would settle it,
   and everything stacked above it, in a different order. *)
type rewrite_step =
  | Descend of t
  | Rebuild of t * t
  | Link of t * t

let graph_rewrite ?loc ?(name = "") ?(enter_calls = false) ?(bottom_up = false)
    ?bpm ?(walk = false) ?(on_rebuild = fun ~old_n:_ ~new_n:_ -> ()) f root =
  let results = Ref_tbl.create 64 in
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
  (* Rewriting iterates: a node's replacement is itself rewritten, and so on
     until one settles. [chains] carries the nodes already passed through on
     the way to a result, so a rewrite that loops back is reported where it
     closes. This is the top-down twin of [apply_pre_fixed]'s [seen]. An
     entry dies as soon as its node resolves, so only live chains are held. *)
  let chains = Ref_tbl.create 16 in
  let extend_chain origin target =
    let seen =
      match Ref_tbl.find_opt chains origin with
      | Some seen -> seen
      | None -> [ origin ]
    in
    if List.memq target seen then invalid_arg cycle_message;
    Ref_tbl.replace chains target (target :: seen)
  in
  let resolve u r =
    Ref_tbl.replace results u r;
    Ref_tbl.remove chains u;
    if not (u == r) then on_rebuild ~old_n:u ~new_n:r
  in
  (* A call or function body is pinned to itself, so that no path reaching it
     is rewritten -- not merely the one through this node. *)
  let pin_body u =
    if
      (not enter_calls)
      && match op u with Ops.Call -> true | _ -> false
    then
      let body = (src u).(0) in
      Ref_tbl.replace results body body
  in
  let sources_changed srcs new_srcs =
    let changed = ref false in
    Array.iteri
      (fun i child -> if not (child == new_srcs.(i)) then changed := true)
      srcs;
    !changed
  in
  (* Single pass: a replacement is final and is not traversed again. *)
  let walk_rewrite root =
    let stack = ref [ (root, false) ] in
    let push step = stack := step :: !stack in
    let rec run () =
      match !stack with
      | [] -> ()
      | (u, descended) :: rest ->
          stack := rest;
          if not (Ref_tbl.mem results u) then
            if not descended then (
              match apply_pre_once u with
              | `Rewritten r | `Gated r -> resolve u r
              | `Unchanged _ ->
                  push (u, true);
                  pin_body u;
                  let srcs = src u in
                  for i = Array.length srcs - 1 downto 0 do
                    if not (Ref_tbl.mem results srcs.(i)) then
                      push (srcs.(i), false)
                  done)
            else begin
              let srcs = src u in
              let new_srcs =
                Array.map
                  (fun child ->
                    match Ref_tbl.find_opt results child with
                    | Some r -> r
                    | None -> child)
                  srcs
              in
              let u' =
                if sources_changed srcs new_srcs then begin
                  let rebuilt = replace u ~src:new_srcs () in
                  on_rebuild ~old_n:u ~new_n:rebuilt;
                  rebuilt
                end
                else u
              in
              let u' =
                match post with
                | None -> u'
                | Some fn -> (
                    match maybe_rewrite fn u' with Some r -> r | None -> u')
              in
              resolve u u'
            end;
          run ()
    in
    run ();
    match Ref_tbl.find_opt results root with Some r -> r | None -> root
  in
  (* Parking is the visit order, not a scheduling detail. When [rebuild] meets
     a source that has not resolved yet it suspends the whole step on that
     source and returns, rather than descending into it; the step resumes when
     the source settles. Rewriting this as a recursive descent -- which is what
     it reads like it wants to be -- produces the same results in a different
     order, and order is observable: every pass that threads a mutable context
     (accumulator numbering, slot counters) numbers off this traversal. No unit
     test covers it. The only thing that does is the parity goldens, so a
     change here that keeps the suites green has proved nothing. *)
  let unified_rewrite root =
    let stack = ref [ Descend root ] in
    let scheduled = Ref_tbl.create 64 in
    let parked = Ref_tbl.create 64 in
    Ref_tbl.replace scheduled root ();
    let push step = stack := step :: !stack in
    let park u step =
      let waiting =
        match Ref_tbl.find_opt parked u with Some l -> l | None -> []
      in
      Ref_tbl.replace parked u (step :: waiting)
    in
    let settle u r =
      resolve u r;
      match Ref_tbl.find_opt parked u with
      | None -> ()
      | Some waiting ->
          Ref_tbl.remove parked u;
          List.iter push (List.rev waiting)
    in
    let descend u =
      match apply_pre_fixed u with
      | `Gated r -> settle u r
      | `Continue u' ->
          push (Rebuild (u, u'));
          pin_body u';
          let srcs = src u' in
          for i = Array.length srcs - 1 downto 0 do
            let child = srcs.(i) in
            if not (Ref_tbl.mem scheduled child) then begin
              push (Descend child);
              Ref_tbl.replace scheduled child ()
            end
          done
    in
    let advance u target =
      extend_chain u target;
      push (Link (u, target));
      push (Descend target)
    in
    let rebuild u u' =
      let srcs = src u' in
      let len = Array.length srcs in
      let new_srcs = Array.make len u' in
      let missing = ref None in
      let i = ref 0 in
      while Option.is_none !missing && !i < len do
        (match Ref_tbl.find_opt results srcs.(!i) with
        | Some r -> new_srcs.(!i) <- r
        | None -> missing := Some srcs.(!i));
        incr i
      done;
      match !missing with
      | Some child -> park child (Rebuild (u, u'))
      | None ->
          if sources_changed srcs new_srcs then begin
            let rebuilt = replace u' ~src:new_srcs () in
            on_rebuild ~old_n:u' ~new_n:rebuilt;
            advance u rebuilt
          end
          else (
            match post with
            | None -> settle u u'
            | Some fn -> (
                match maybe_rewrite fn u' with
                | None -> settle u u'
                | Some r -> advance u r))
    in
    let link u target =
      match Ref_tbl.find_opt results target with
      | None -> park target (Link (u, target))
      | Some r -> settle u r
    in
    let rec run () =
      match !stack with
      | [] -> ()
      | step :: rest ->
          stack := rest;
          (match step with
          | Descend u -> if not (Ref_tbl.mem results u) then descend u
          | Rebuild (u, u') -> if not (Ref_tbl.mem results u) then rebuild u u'
          | Link (u, target) ->
              if not (Ref_tbl.mem results u) then link u target);
          run ()
    in
    run ();
    match Ref_tbl.find_opt results root with Some r -> r | None -> root
  in
  if walk then walk_rewrite root else unified_rewrite root

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

let substitute ?(walk = false) mappings root =
  let f u = List.assq_opt u mappings in
  graph_rewrite ~bottom_up:true ~walk f root

(* Analysis *)

let min_max_cache : (Bound.t * Bound.t) Weak_tbl.t Domain.DLS.key =
  Domain.DLS.new_key (fun () -> Weak_tbl.create 1024)

let rec min_max u =
  let cache = Domain.DLS.get min_max_cache in
  match Weak_tbl.find_opt cache u with
  | Some bounds -> bounds
  | None ->
      let bounds = compute_min_max u in
      Weak_tbl.replace cache u bounds;
      bounds

and compute_min_max u =
  let module B = Bound in
  let zero = B.zero in
  let dtype_bounds () = Dtype.min (dtype u), Dtype.max (dtype u) in
  let corners f a b c d =
    let w, x, y, z = f a c, f a d, f b c, f b d in
    B.min (B.min w x) (B.min y z), B.max (B.max w x) (B.max y z)
  in
  let binary () =
    let a, b = min_max (src u).(0) and c, d = min_max (src u).(1) in
    let positive x = B.lt zero x and negative x = B.lt x zero in
    let divisor_nonzero = (positive c && positive d) || (negative c && negative d) in
    match op u with
    | Ops.Add -> Some (B.add a c, B.add b d)
    | Ops.Sub -> Some (B.sub a d, B.sub b c)
    | Ops.And when Dtype.is_int (dtype u) && B.equal c d && B.le zero c ->
        Some (zero, if negative a then d else B.min b d)
    | Ops.Mul -> Some (corners B.mul a b c d)
    | Ops.Shl when B.equal c d -> Some (B.shift_left a c, B.shift_left b c)
    | Ops.Shr when B.equal c d -> Some (B.shift_right a c, B.shift_right b c)
    | Ops.Cmod ->
        if B.equal c d && positive c then
          Some ((if positive a then zero else if B.lt (B.neg c) a then a else B.neg (B.pred d)),
                (if negative b then zero else if B.lt b c then b else B.pred c))
        else if positive c then
          Some ((if B.le zero a then zero else B.neg (B.pred d)),
                (if B.le b zero then zero else B.pred d))
        else if negative d then
          let hi = B.pred (B.neg c) in
          Some ((if B.le zero a then zero else B.neg hi),
                (if B.le b zero then zero else hi))
        else None
    | Ops.Cdiv when divisor_nonzero -> Some (corners B.cdiv a b c d)
    | Ops.Floordiv | Ops.Floormod when B.lt b a -> Some (zero, zero)
    | Ops.Floordiv when divisor_nonzero -> Some (corners B.floordiv a b c d)
    | Ops.Floormod ->
        if B.equal c d && not (B.equal c zero) then
          if B.equal (B.floordiv a c) (B.floordiv b c) then
            Some (B.floormod a c, B.floormod b c)
          else if positive c then Some (zero, B.pred c)
          else Some (B.succ c, zero)
        else if positive c then Some (zero, B.pred d)
        else if negative d then Some (B.succ c, zero)
        else None
    | Ops.Xor when B.equal c d && B.equal c (B.int (-1)) ->
        Some (B.lognot b, B.lognot a)
    | Ops.Max -> Some (B.max a c, B.max b d)
    | Ops.Cmplt -> Some (`Bool (B.lt b c), `Bool (B.lt a d))
    | Ops.Cmpne ->
        Some (`Bool (B.lt b c || B.lt d a),
              `Bool (not (B.equal a b && B.equal a c && B.equal c d)))
    | Ops.Or when Dtype.is_bool (dtype u) -> Some (B.max a c, B.max b d)
    | Ops.And when Dtype.is_bool (dtype u) -> Some (B.min a c, B.min b d)
    | _ -> None
  in
  let binary_result =
    if not (Dtype.is_float (dtype u)) && Ops.Group.is_binary (op u)
       && Array.length (src u) >= 2 then binary () else None
  in
  match binary_result with
  | Some bounds -> bounds
  | None ->
      match op u, src u with
      | Ops.Where, [| _; t; f |] ->
          let a, b = min_max t and c, d = min_max f in
          B.min a c, B.max b d
      | Ops.Const, _ ->
          (match arg u with
           | Arg.Value c ->
               (match Const.view c with
                | Const.Int n -> `Int n, `Int n
                | Const.Bool b -> `Bool b, `Bool b
                | Const.Float f when not (Float.is_nan f) -> `Float f, `Float f
                | Const.Float _ | Const.Invalid -> dtype_bounds ())
           | _ -> dtype_bounds ())
      | (Ops.Param | Ops.Buffer | Ops.Alloc), _ ->
          (match arg u with
           | Arg.Param_arg { vmin_vmax = Some (lo, hi); _ } -> lo, hi
           | _ -> dtype_bounds ())
      | (Ops.Range | Ops.Special), srcs when Array.length srcs > 0 ->
          zero, B.pred (snd (min_max srcs.(0)))
      | Ops.Stack, srcs when Array.length srcs > 0 ->
          Array.fold_left (fun (lo, hi) s ->
              let a, b = min_max s in B.min lo a, B.max hi b)
            (min_max srcs.(0)) srcs
      | Ops.Pad, srcs when Array.length srcs > 0 ->
          let lo, hi = min_max srcs.(0) in B.min lo zero, B.max hi zero
      | (Ops.Index | Ops.Stage | Ops.After | Ops.Detach | Ops.Copy
        | Ops.Contiguous_backward), srcs when Array.length srcs > 0 -> min_max srcs.(0)
      | movement, srcs when Ops.Group.is_movement movement && Array.length srcs > 0 -> min_max srcs.(0)
      | Ops.Cast, [| s |] ->
          let dt = dtype u in
          let lo, hi = dtype_bounds () in
          let a, b = min_max s in
          let a, b = B.round dt a, B.round dt b in
          let is_nan = function `Float f -> Float.is_nan f | _ -> false in
          if is_nan a || is_nan b then lo, hi
          else if Dtype.is_unsigned dt && B.le zero a && B.le b hi then a, b
          else if (Dtype.is_float dt || (Dtype.is_int dt && not (Dtype.is_unsigned dt)))
                  && B.le a hi && B.le lo b then B.max lo a, B.min b hi
          else lo, hi
      | _ -> dtype_bounds ()

let vmin u = fst (min_max u)
let vmax u = snd (min_max u)

let commit_dtype ?(default_int = Dtype.default_int) u =
  if not (Dtype.equal (dtype u) Dtype.weakint) then Dtype.strong_dtype (dtype u)
  else
    let lo, hi = min_max u in
    if Bound.equal lo hi &&
       (Bound.lt lo (Dtype.min Dtype.int64) || Bound.lt (Dtype.max Dtype.uint64) hi)
    then invalid_arg "Uop.commit_dtype: integer does not fit any storage dtype";
    match List.find_opt (fun dt ->
        Bound.le (Dtype.min dt) lo && Bound.le hi (Dtype.max dt))
        [ default_int; Dtype.int32; Dtype.int64; Dtype.uint64 ] with
    | Some dt -> dt
    | None -> Dtype.int64

let const_int_value u =
  match as_const u with
  | Some c -> (match Const.view c with
      | Const.Int n when Z.fits_int n -> Some (Z.to_int n)
      | _ -> None)
  | None -> None

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
let dim_mul a b = dim_binary Ops.Mul a b
let dim_div a b = dim_binary Ops.Floordiv a b

let dim_prod dims = List.fold_left dim_mul dim_one dims

let dims_equal a b =
  List.length a = List.length b && List.for_all2 equal a b

let dim_non_negative d = Bound.le Bound.zero (vmin d)

let dim_leq a b =
  match const_int_value a, const_int_value b with
  | Some a, Some b -> a <= b
  | _ -> equal a b || Bound.le (vmax a) (vmin b)

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
  (* A shrink bound is rejected only when it is *provably* violated; a
     symbolically-undecidable bound is accepted, since its out-of-range
     accesses are masked by a surrounding gate (e.g. a per-shard shrink whose
     offset carries a symbolic device index). So a shrink offset or size is
     negative only when definitely negative ([vmax < 0]), and the slice
     overruns only when it definitely exceeds the input ([vmin] of
     [offset + size] past the input's [vmax]). *)
  let provably_negative d = Bound.lt (vmax d) Bound.zero in
  if List.exists provably_negative offsets
     || List.exists provably_negative sizes then
    invalid_shape op "shape contains a negative dimension";
  List.iter2
    (fun src_dim (offset, size) ->
      if Bound.lt (vmax src_dim) (vmin (dim_add offset size)) then
        invalid_shape op "slice extends past the input shape")
    src_shape (List.combine offsets sizes)

let broadcast_shape shapes =
  let rank = List.fold_left (fun rank shape -> max rank (List.length shape)) 0 shapes in
  let shapes = List.map (fun shape ->
      List.init (rank - List.length shape) (fun _ -> dim_one) @ shape) shapes in
  let same_dim a b =
    equal a b || match const_int_value a, const_int_value b with
    | Some x, Some y -> x = y
    | _ -> false
  in
  List.init rank (fun axis ->
      List.fold_left (fun chosen shape ->
          let dim = List.nth shape axis in
          if dim_is_one chosen then dim
          else if dim_is_one dim || same_dim chosen dim then chosen
          else invalid_arg "Uop.broadcast_shape: shapes cannot be broadcast")
        dim_one shapes)

let as_shape u =
  match op u with
  | Ops.Stack -> Array.to_list (src u)
  | Ops.Const -> [ u ]
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

(* Per-node shape memo. Nodes are hash-consed and immutable, so a shape is a
   pure function of the node; caching it by node identity (as [device_of] and
   [min_max] already do) avoids recomputing shared subgraphs, which would
   otherwise re-walk a diamond once per parent and blow up exponentially. *)
let shape_cache : t list option Weak_tbl.t Domain.DLS.key =
  Domain.DLS.new_key (fun () -> Weak_tbl.create 1024)

let rec shape u =
  match shape_opt u with
  | Some shape -> shape
  | None ->
      invalid_arg
        (Printf.sprintf "Uop.shape: %s does not have a shape"
           (Ops.name (op u)))

and shape_opt u =
  let shape_cache = Domain.DLS.get shape_cache in
  match Weak_tbl.find_opt shape_cache u with
  | Some cached -> cached
  | None ->
      let result = compute_shape_opt u in
      Weak_tbl.add shape_cache u result;
      result

and compute_shape_opt u =
  let srcs = src u in
  let first_shape () =
    if Array.length srcs = 0 then None else shape_opt srcs.(0)
  in
  match op u with
  | Ops.If | Ops.Barrier | Ops.Sink | Ops.Rewrite_error | Ops.Endif | Ops.Backedge
  | Ops.Group | Ops.Linear | Ops.Program | Ops.Source
  | Ops.Custom_function ->
      None
  | Ops.Call ->
      if Dtype.equal (dtype u) Dtype.void then None else Some []
  | Ops.Ins ->
      (* Scalar shape; the vector width is carried in the instruction
         encoding, not the shape. *)
      if Dtype.equal (dtype u) void_dtype then None else Some []
  | Ops.Custom | Ops.Customi ->
      if Dtype.equal (dtype u) void_dtype then None
      else
        let shapes = Array.to_list srcs |> List.filter_map shape_opt in
        if shapes = [] then None else Some (broadcast_shape shapes)
  | Ops.Noop ->
      if Array.length srcs = 0 then None else shape_opt srcs.(0)
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
      else Some (const_int (Array.length srcs) :: shape srcs.(0))
  | Ops.Const -> Some []
  | Ops.Getaddr | Ops.Range | Ops.Special -> Some []
  | Ops.Binary ->
      (* One dimension per compiled byte. *)
      (match arg u with
       | Arg.String s -> Some [ const_int (String.length s) ]
       | _ -> Some [])
  | Ops.Buffer | Ops.Alloc | Ops.Param ->
      (match Arg.as_param_arg (arg u) with
       | Some p -> Some (storage_shape p)
       | None -> None)
  | Ops.Stage ->
      let ranges = Array.to_list srcs |> List.tl in
      let range_shape =
        List.map
          (fun r ->
            let rs = src r in
            if op r = Ops.Range && Array.length rs > 0 then rs.(0)
            else const (Bound.const Dtype.weakint (Bound.succ (vmax r))))
          ranges
      in
      if Array.length srcs = 0 then None else Some (range_shape @ shape srcs.(0))
  | Ops.Wmma ->
      (* Broadcast of the operands' shapes without their packed tails, plus
         the accumulator's own tail: the lanes one thread holds are already
         the accumulator's last dimension, so the width is read off the node
         rather than recomputed from the tensor-core axes — which are dropped
         once the expander has consumed them. *)
      if Array.length srcs >= 3 then
        let drop_last l =
          match List.rev l with [] -> [] | _ :: rest -> List.rev rest
        in
        match (shape_opt srcs.(0), shape_opt srcs.(1), shape_opt srcs.(2)) with
        | Some s0, Some s1, Some s2 when s2 <> [] ->
            Some
              (broadcast_shape
                 [ drop_last s0; drop_last s1; drop_last s2 ]
              @ [ List.nth s2 (List.length s2 - 1) ])
        | _ -> None
      else None
  | Ops.Mstack | Ops.Mselect | Ops.Detach
  | Ops.Contiguous_backward | Ops.After | Ops.Load | Ops.Copy
  | Ops.Allreduce | Ops.Store | Ops.End ->
      first_shape ()
  | Ops.Reduce ->
      (* The reduced axes are the leading [num_axes] of the source shape. *)
      (match as_reduce u with
       | Some { src; num_axes; _ } ->
           Some (List.filteri (fun i _ -> i >= num_axes) (shape src))
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
                  (match const_int_value last with
                   | Some n when n * input_size mod output_size <> 0 ->
                       invalid_shape Ops.Bitcast "unsupported size in bitcast"
                   | _ -> ());
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
      (* EXPAND prepends its argument dims to the source shape. *)
      if Array.length srcs >= 2 then Some (as_shape srcs.(1) @ shape srcs.(0))
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
  | Ops.Unshard ->
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
      Some (broadcast_shape (Array.to_list srcs |> List.map shape))
  | _ -> None

(* A reshape to [src]'s own shape is [src]. Indexing through the no-op node
   would re-derive the index by mod and div, which folds a range of at most one
   iteration to 0. *)
let reshape ~src ~shape =
  match shape_opt src with
  | Some dims when List.equal equal dims (as_shape shape) -> src
  | _ ->
      mk ~op:Ops.Reshape ~dtype:(dtype src) ~src:[| src; shape |] ~arg:Arg.Empty

let max_shape u = List.map (fun d -> Bound.to_int (vmax d)) (shape u)
let max_shape_numel dims =
  List.fold_left (fun n dim -> Bound.mul n (vmax dim)) Bound.one dims
  |> Bound.to_int

let max_numel u = max_shape_numel (shape u)

let storage_size dims =
  match dims with
  | [] -> None
  | _ -> Some (max_shape_numel dims)

let view_as node dims =
  match dims with
  | [] -> node
  | _ ->
      let max_dims = List.map (fun dim -> const (Bound.const Dtype.weakint (vmax dim))) dims in
      let node = if List.length dims > 1 then reshape ~src:node ~shape:(shape_arg max_dims) else node in
      if List.for_all (fun dim -> Bound.equal (vmin dim) (vmax dim)) dims then node
      else shrink ~src:node ~offset:(shape_arg (List.map (fun _ -> const_int 0) dims))
          ~size:(shape_arg dims)

let param ~slot ~dtype ?shape:shape_arg ?image ?device ?vmin_vmax ?multiple_of ?name
    ?addrspace ?axis ?volatile () =
  let dims = match shape_arg with None -> [] | Some shape when op shape = Ops.Noop -> [] | Some shape -> as_shape shape in
  let size = match image with
    | None -> storage_size dims
    | Some (h, w) -> storage_size [const_int h; const_int w; const_int 4]
  in
  let p = default_param_arg ~dtype ?size ?image ?vmin_vmax ?multiple_of ?name
      ?addrspace ?axis ?device ?volatile slot in
  let node = mk ~op:Ops.Param ~dtype ~src:[||] ~arg:(Arg.Param_arg p) in
  if Option.is_some image then node else view_as node dims

let buffer ~slot ~dtype ?shape:shape_arg ?name ?addrspace ?axis ?device ?volatile () =
  let dims = match shape_arg with None -> [] | Some shape when op shape = Ops.Noop -> [] | Some shape -> as_shape shape in
  let size = storage_size dims in
  let p = default_param_arg ~dtype ?size ?name ?addrspace ?axis ?device ?volatile slot in
  let devices = match p.addrspace, device with
    | Dtype.Global, Some (Single device) -> Some [device]
    | Dtype.Global, Some (Multi devices) -> Some devices
    | _ -> None
  in
  let buffer = Option.map (fun devices ->
      if devices = [] then invalid_arg "Uop.buffer: empty device placement";
      List.map (fun device -> Storage.on_device ~device ~size:(Option.value size ~default:1) ~dtype ()) devices)
      devices in
  view_as (mk ~op:Ops.Buffer ~dtype ~src:[||] ~arg:(Arg.Param_arg { p with buffer })) dims

let alloc ~slot ~dtype ?shape:shape_arg ?device ?(bind_on_realize = false) () =
  if Dtype.is_weak dtype then invalid_arg "Uop.alloc: dtype must be concrete";
  let dims = match shape_arg with None -> [] | Some s -> as_shape s in
  let p = default_param_arg ~dtype ?size:(storage_size dims) ?device slot in
  view_as (mk ~op:Ops.Alloc ~dtype ~src:[||]
    ~arg:(Arg.Param_arg { p with bind_on_realize })) dims

let from_buffer buf =
  let dtype = Storage.dtype buf in
  let p = default_param_arg ~dtype ~size:(Storage.size buf)
      ~device:(Single (Storage.device buf)) (-1 - Storage.id buf) in
  mk ~op:Ops.Buffer ~dtype ~src:[||]
    ~arg:(Arg.Param_arg { p with buffer = Some [buf] })

(* Memoized like [shape]: sources are shared DAGs, and an unmemoized walk is
   exponential in residual depth. *)
let axis_cache : int option Weak_tbl.t Domain.DLS.key =
  Domain.DLS.new_key (fun () -> Weak_tbl.create 1024)

let rec axis u =
  let axis_cache = Domain.DLS.get axis_cache in
  match Weak_tbl.find_opt axis_cache u with
  | Some cached -> cached
  | None ->
      let result = compute_axis u in
      Weak_tbl.add axis_cache u result;
      result

and compute_axis u =
  let srcs = src u in
  match op u with
  | Ops.Copy -> None
  | Ops.Unshard -> Arg.as_int (arg u)
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
  | Ops.Stack ->
      (* Stack adds a leading axis, so a sharded source's axis shifts up. *)
      let axes =
        Array.fold_left
          (fun acc s ->
            match axis s with
            | None -> acc
            | Some a -> if List.mem a acc then acc else a :: acc)
          [] srcs
      in
      (match axes with [] -> None | a :: _ -> Some (a + 1))
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
      | Some src_axis, Some { num_axes; _ } ->
          if src_axis < num_axes then None else Some (src_axis - num_axes)
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
  | Ops.Expand -> (
      (* Prepended dims shift the sharded axis up. *)
      match axis srcs.(0) with
      | None -> None
      | Some src_axis ->
          let n_dims =
            if Array.length srcs >= 2 then List.length (as_shape srcs.(1)) else 0
          in
          Some (src_axis + n_dims))
  | _ -> axis srcs.(0)

let shard_shape u =
  match device_of u, axis u with
  | Some (Multi devs), Some ax ->
      List.mapi
        (fun i d ->
          if i = ax then dim_div d (const_int (List.length devs)) else d)
        (shape u)
  | _ -> shape u

let max_shard_shape u = List.map (fun d -> Bound.to_int (vmax d)) (shard_shape u)
let max_shard_numel u = max_shape_numel (shard_shape u)

(* Calls pass storage explicitly; outputs are allocations in the caller. *)

let param_like u ~slot =
  let variable = match as_bind u with
    | Some {var; _} -> Some var
    | None when is_variable u -> Some u
    | None -> None in
  match variable with
  | Some var ->
      let p = Option.get (Arg.as_param_arg (arg var)) in
      replace var ~op:Ops.Param
        ~arg:(Arg.Param_arg {p with slot; name = Some ("p" ^ string_of_int slot)}) ()
  | None ->
      let device = device_of u in
      let dims = shard_shape u in
      let volatile = match Arg.as_param_arg (arg (buf_uop u)) with
        | Some p -> p.volatile | None -> false in
      let p = param ~slot ~dtype:(dtype u) ~shape:(shape_arg dims) ?device ~volatile () in
      match device, axis u with
      | Some (Multi _), Some axis -> multi ~src:p ~axis
      | _ -> p

let store_call ~dst ~src =
  let body = store ~dst:(param_like dst ~slot:0) ~value:(param_like src ~slot:1) () in
  let info = { grad_fxn = None; name = None; precompile = false;
    precompile_backward = false; dtype = Dtype.void; aux = None } in
  call ~body ~args:[dst; src] ~info

let call_with_outputs ?output_pos ~values ~args ~info () =
  let count = List.length values + List.length args in
  let positions = match output_pos with
    | Some positions -> positions
    | None -> List.mapi (fun i _ -> List.length args + i) values in
  if List.length positions <> List.length values
     || positions <> List.sort_uniq Int.compare positions
     || List.exists (fun p -> p < 0 || p >= count) positions then
    invalid_arg "Uop.call_with_outputs: invalid output positions";
  let actuals = Array.make count None in
  let inputs = ref args in
  for slot = 0 to count - 1 do
    if not (List.mem slot positions) then
      match !inputs with
      | x :: xs -> actuals.(slot) <- Some x; inputs := xs
      | [] -> assert false
  done;
  let resolve_dim dim = graph_rewrite ~walk:true (fun n ->
      match as_param n with
      | Some {param = {slot; _}; _} when slot >= 0 && slot < count -> actuals.(slot)
      | _ -> None) dim in
  let default_device = List.find_map device_of (values @ args) in
  let outputs = List.map (fun value ->
      let device = match device_of value with Some _ as d -> d | None -> default_device in
      let dims = List.map resolve_dim (shard_shape value) in
      let size = if dims = [] then 1 else Option.get (storage_size dims) in
      let storage = alloc ~slot:(fresh_buffer_slot ()) ~dtype:(dtype value)
          ~shape:(const_int size) ?device () in
      let view = if dims = [] then reshape ~src:storage ~shape:(shape_arg []) else view_as storage dims in
      match device, axis value with
      | Some (Multi _), Some axis -> multi ~src:view ~axis
      | _ -> view) values in
  List.iter2 (fun slot output -> actuals.(slot) <- Some output) positions outputs;
  let body = sink (List.map2 (fun value slot ->
      store ~dst:(param_like value ~slot) ~value ()) values positions) in
  let args = Array.to_list (Array.mapi (fun slot arg ->
      let arg = Option.get arg in
      if info.precompile && not (List.mem slot positions) then contiguous ~src:arg () else arg) actuals) in
  let invoked = call ~body ~args ~info in
  List.map (fun output -> after ~src:output ~deps:[invoked]) outputs

(* Placeholders and custom kernels *)

let placeholder ~shape:dims ~dtype ~slot ?(addrspace = Dtype.Global) ?device
    ?volatile () =
  let dtype = Dtype.strong_dtype dtype in
  let flat = const_int (List.fold_left ( * ) 1 dims) in
  let base =
    match addrspace with
    | Dtype.Global -> param ~slot ~dtype ~shape:flat ~addrspace ?device ?volatile ()
    | Dtype.Local | Dtype.Reg ->
        if Option.is_some device then
          invalid_arg
            "Uop.placeholder: local and reg placeholders cannot have a device";
        buffer ~slot ~dtype ~shape:flat ~addrspace ()
    | Dtype.Alu -> invalid_arg "Uop.placeholder: alu address space"
  in
  if List.length dims > 1 then
    reshape ~src:base ~shape:(shape_arg (List.map const_int dims))
  else base

let placeholder_like u ~slot ?(addrspace = Dtype.Global) () =
  if List.exists (fun d -> Option.is_none (const_int_value d)) (shape u) then
    invalid_arg "Uop.placeholder_like: symbolic shape";
  placeholder ~shape:(max_shard_shape u) ~dtype:(dtype u) ~slot ~addrspace ()

let custom_kernel ?grad_fxn ~fxn srcs =
  let placeholders =
    List.mapi (fun slot s -> placeholder_like s ~slot ()) srcs
  in
  let info =
    { grad_fxn; name = None; precompile = false; precompile_backward = false;
      aux = None; dtype = Dtype.void }
  in
  let kernel = call ~body:(fxn placeholders) ~args:srcs ~info in
  List.map (fun s -> after ~src:s ~deps:[ kernel ]) srcs

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

let contiguous_view u =
  let exact_int t =
    match const_int_value t with
    | Some _ as value -> value
    | None ->
        let lo, hi = min_max t in
        if Bound.equal lo hi then
          (try Some (Bound.to_int lo) with Invalid_argument _ -> None)
        else None
  in
  let dim_is_zero t = match exact_int t with Some 0 -> true | _ -> false in
  let dim_max_at_most n t = Bound.le (vmax t) (Bound.int n) in
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
            match exact_int offset_dim, exact_int dim with
            | Some offset_dim, Some dim
              when offset_dim >= 0 && Bound.le Bound.zero (vmin size)
                   && Bound.le (vmax size) (Bound.int (dim - offset_dim))
                   && prefix_one ->
                let trailing = List.filteri (fun j _ -> j > i) shape in
                (match exact_stride trailing with
                 | Some stride ->
                     loop (dim_max_at_most 1 size) (offset + (offset_dim * stride))
                       (size :: out) (i + 1)
                 | None when offset_dim = 0 ->
                     loop (dim_max_at_most 1 size) offset (size :: out) (i + 1)
                 | None -> None)
            | _ -> None
      in
      loop true 0 [] 0
  in
  let rec walk node =
    match op node with
    | Ops.Buffer | Ops.Alloc | Ops.Param -> Some (node, 0, shape node)
    | Ops.Mselect | Ops.Mstack -> Some (node, 0, shape node)
    | Ops.Bitcast ->
        Option.map (fun (base, offset, _) -> base, offset, shape node)
          (walk (src node).(0))
    | Ops.Stage when arg node = Arg.Empty -> walk (src node).(0)
    | Ops.Detach | Ops.Contiguous_backward | Ops.After ->
        let srcs = src node in
        if Array.length srcs = 0 then None else walk srcs.(0)
    | Ops.Reshape ->
        let srcs = src node in
        if Array.length srcs = 0 then None
        else
          (match walk srcs.(0), shape_arg srcs 1 with
           | Some (base, base_off, _), Some target -> Some (base, base_off, target)
           | _ -> None)
    | Ops.Expand ->
        let srcs = src node in
        if Array.length srcs = 0 then None
        else
          (match walk srcs.(0), shape_arg srcs 1 with
           | Some (base, base_off, current), Some target
             when List.length current = List.length target
                  && List.for_all2 same_dim current target ->
               Some (base, base_off, target)
           | _ -> None)
    | Ops.Pad ->
        let srcs = src node in
        if Array.length srcs = 0 then None
        else
          (match walk srcs.(0), pairs_arg srcs with
           | Some ((_, _, current) as state), Some pairs
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
           | Some (base, base_off, current), Some pairs ->
               (match shrink_shape_and_offset current pairs with
                | Some (offset, shape) -> Some (base, base_off + offset * Dtype.itemsize (dtype node), shape)
                | None -> None)
           | _ -> None)
    | Ops.Permute ->
        let srcs = src node in
        if Array.length srcs = 0 then None
        else
          (match walk srcs.(0), Arg.as_ints (arg node) with
           | Some (base, base_off, current), Some order
             when contiguous_permutation order current ->
               (match permute_list order current with
                | Some shape -> Some (base, base_off, shape)
                | None -> None)
           | _ -> None)
    | Ops.Flip ->
        let srcs = src node in
        if Array.length srcs = 0 then None
        else
          (match walk srcs.(0), Arg.as_bools (arg node) with
           | Some ((_, _, current) as state), Some dims
             when List.length dims = List.length current
                  && List.for_all2
                       (fun flipped dim ->
                         (not flipped) || dim_max_at_most 1 dim)
                       dims current ->
               Some state
           | _ -> None)
    | _ -> None
  in
  Option.map (fun (base, offset, _) -> base, offset) (walk u)

let reduce_axis ~src ~op ~axes =
  let shp = shape src in
  let axes = List.sort_uniq Int.compare axes in
  match axes with
  | [] -> src
  | _ ->
      (* Reducing a size-1 axis just drops it, so only genuinely reduced axes
         drive the REDUCE: they are permuted to the front and their count is
         stored in the arg, then the result is reshaped back to the kept axes. *)
      let reduce_axes =
        List.filter (fun x -> not (dim_is_one (List.nth shp x))) axes
      in
      let out_shape = List.filteri (fun i _ -> not (List.mem i axes)) shp in
      (match reduce_axes with
       | [] -> reshape ~src ~shape:(shape_arg out_shape)
       | _ ->
           let rest =
             List.filter
               (fun i -> not (List.mem i reduce_axes))
               (List.init (List.length shp) Fun.id)
           in
           let permuted = permute ~src ~order:(reduce_axes @ rest) in
           let red =
             mk ~op:Ops.Reduce ~dtype:(dtype src) ~src:[| permuted |]
               ~arg:(Arg.Reduce_arg { op; num_axes = List.length reduce_axes })
           in
           if axes = reduce_axes then red
           else reshape ~src:red ~shape:(shape_arg out_shape))

let broadcast_to ~src ~shape:target_shape =
  let cur = shape src in
  let target = as_shape target_shape in
  if dims_equal cur target then src
  else begin
    let cur = Array.of_list cur and target = Array.of_list target in
    let cur_rank = Array.length cur and new_rank = Array.length target in
    if cur_rank > new_rank then
      invalid_arg "Uop.broadcast_to: cannot broadcast to fewer dimensions";
    (* Align right by prepending [n_left] new leading axes, then check each
       existing axis is either unchanged or broadcast from size one. *)
    let n_left = new_rank - cur_rank in
    for i = 0 to cur_rank - 1 do
      if not (equal cur.(i) target.(n_left + i) || dim_is_one cur.(i)) then
        invalid_arg "Uop.broadcast_to: shapes are not broadcast-compatible"
    done;
    (* EXPAND only adds leading axes, so squeeze the size-one axes that must
       grow, prepend them (and the new leading axes) via EXPAND, then permute
       everything back into the target order. *)
    let is_expand i =
      dim_is_one cur.(i) && not (dim_is_one target.(n_left + i))
    in
    let axes = List.init cur_rank Fun.id in
    let expand_at = List.filter is_expand axes in
    let kept = List.filter (fun i -> not (is_expand i)) axes in
    let squeezed =
      reshape ~src ~shape:(shape_arg (List.map (fun i -> cur.(i)) kept))
    in
    let prepend =
      Array.to_list (Array.sub target 0 n_left)
      @ List.map (fun i -> target.(n_left + i)) expand_at
    in
    let expanded = expand ~src:squeezed ~dims:(shape_arg prepend) in
    let n_expand = List.length expand_at in
    let position_in lst x =
      let rec go i = function
        | [] -> raise Not_found
        | y :: ys -> if y = x then i else go (i + 1) ys
      in
      go 0 lst
    in
    let order =
      List.init n_left Fun.id
      @ List.init cur_rank (fun i ->
            if is_expand i then n_left + position_in expand_at i
            else n_left + n_expand + position_in kept i)
    in
    permute ~src:expanded ~order
  end

let rec const_of_dtype ?shape:target_shape dtype value =
  let ret =
    match value with
    | Const_scalar value -> const (Const.of_scalar dtype value)
    | Const_invalid -> const Const.invalid
    | Const_tuple values ->
        let src = Array.of_list (List.map (const_of_dtype dtype) values) in
        mk ~op:Ops.Stack ~dtype ~src ~arg:Arg.Empty
  in
  match target_shape with
  | None -> ret
  | Some target_arg ->
      let target = as_shape target_arg in
      (* A scalar constant broadcasts to [target] by prepending it. *)
      if target = [] || dims_equal (shape ret) target then ret
      else expand ~src:ret ~dims:target_arg

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
  | Ops.Param | Ops.Buffer | Ops.Alloc ->
      (match Arg.as_param_arg (arg u) with
       | Option.Some { multiple_of = Option.Some m; _ } -> m
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
        Option.Some (stack ~dtype:(dtype u) srcs)
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
  | Ops.Param | Ops.Buffer | Ops.Alloc ->
      (match Arg.as_param_arg (arg u) with
       | Option.Some { multiple_of = Option.Some m; _ } when m mod n = 0 ->
           Option.Some (alu_binary ~op:Ops.Floordiv ~lhs:u ~rhs:(const_like u n))
       | _ -> Option.None)
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

(* On a boolean first operand the folds are logical or / and. *)
let fold_first name ~bool_op ~op = function
  | [] -> err_empty_list name
  | x :: xs ->
      let op = if Dtype.equal (dtype x) Dtype.bool then bool_op else op in
      List.fold_left (fun acc y -> alu_binary ~op ~lhs:acc ~rhs:y) x xs

let usum = fold_first "usum" ~bool_op:Ops.Or ~op:Ops.Add
let uprod = fold_first "uprod" ~bool_op:Ops.And ~op:Ops.Mul

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

(* Symbolic variables of [u], as (node, name, vmin, vmax). *)
let symbolic_vars u =
  find_nodes (fun n -> op n = Ops.Param || is_variable n) u
  |> List.filter_map (fun n ->
      match Arg.as_param_arg (arg n) with
      | Some { addrspace = Dtype.Alu; name = Some name;
          vmin_vmax = Some (lo, hi); _ } -> Some (n, name, lo, hi)
      | _ -> None)

let sym_infer u var_vals =
  let mappings =
    symbolic_vars u
    |> List.filter_map (fun (n, name, _, _) ->
        match List.assoc_opt name var_vals with
        | Some value -> Some (n, const_int value)
        | None -> None)
  in
  match const_int_value (simplify (substitute mappings u)) with
  | Some n -> n
  | None -> invalid_arg "sym_infer: expression did not reduce to a constant"

(* Symbolic integer ("sint") helpers. A dimension or size is a plain node: a
   concrete integer is a [Const] and a symbolic value is any other
   integer-valued expression. *)

let resolve ?(default = true) u =
  if not (Dtype.is_bool (dtype u)) then
    invalid_arg "Uop.resolve: expected a boolean expression";
  let s = simplify u in
  let lo = vmin s in
  if Bound.equal lo (vmax s) then not (Bound.equal lo Bound.zero) else default

let smax = function
  | [] -> invalid_arg "Uop.smax: empty list"
  | x :: xs ->
      simplify
        (List.fold_left (fun a b -> alu_binary ~op:Ops.Max ~lhs:a ~rhs:b) x xs)

let smin = function
  | [] -> invalid_arg "Uop.smin: empty list"
  | x :: xs ->
      (* Negation as multiplication by -1 keeps the terms visible to the
         value-bounds analysis, so the maximum of the negations folds. *)
      let neg u = alu_binary ~op:Ops.Mul ~lhs:u ~rhs:(const_like u (-1)) in
      simplify
        (neg
           (List.fold_left
              (fun a b -> alu_binary ~op:Ops.Max ~lhs:a ~rhs:(neg b))
              (neg x) xs))

let sprod dims = simplify (dim_prod dims)

let unbind u =
  match as_bind u with
  | Some { var; value } -> (
      match const_int_value value with
      | Some n -> (var, n)
      | None -> invalid_arg "Uop.unbind: bound value is not an integer")
  | _ -> invalid_arg "Uop.unbind: expected a bound variable"

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

let program_info_from_sink ?(target = Target.of_string "") sink =
  let vars = ref [] in
  let globals = ref [] in
  let outs = ref [] in
  let ins = ref [] in
  let global_size = ref [ Launch_int 1; Launch_int 1; Launch_int 1 ] in
  let local_size = ref [ Launch_int 1; Launch_int 1; Launch_int 1 ] in
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
                local_size :=
                  set_nth "program_info_from_sink" !local_size axis
                    (launch_dim_of_uop size)
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
       | _ -> ()))
    (toposort sink);
  let globals = sort_uniq_ints !globals in
  let outs, ins =
    if !outs = [] && !ins = [] then globals, globals
    else sort_uniq_ints !outs, sort_uniq_ints !ins
  in
  { target; global_size = !global_size; local_size = !local_size;
    vars = sort_program_vars !vars; globals; outs; ins }

let int_floor_div a b =
  let q = a / b and r = a mod b in
  if r <> 0 && ((a < 0) <> (b < 0)) then q - 1 else q

let int_floor_mod a b = a - (int_floor_div a b * b)

(* Scalar ALU execution uses mathematical integers until an explicit dtype
   truncation. Storage conversion and host-sized indexing happen elsewhere. *)
let const_as_float c =
  match Const.view c with
  | Const.Float f -> Some f
  | Const.Int n -> Some (Z.to_float n)
  | Const.Bool b -> Some (if b then 1.0 else 0.0)
  | Const.Invalid -> None

let const_as_integer c =
  match Const.view c with
  | Const.Int n -> Some n
  | Const.Bool b -> Some (if b then Z.one else Z.zero)
  | Const.Float _ | Const.Invalid -> None

let const_of_target ~truncate_output ~(target : Dtype.t) value =
  match value with
  | `Int n ->
      if Dtype.is_bool target then Some (Const.bool (Z.sign n <> 0))
      else if Dtype.is_float target then Some (Const.float target (Z.to_float n))
      else
        Some (Const.integer target
          (if truncate_output then Dtype.truncate_integer target n else n))
  | `Float f -> Some (Const.float target f)

let compare_integer_float n f =
  if Float.is_nan f then None
  else if f = Float.infinity then Some (-1)
  else if f = Float.neg_infinity then Some 1
  else
    let c = Z.compare n (Z.of_float f) in
    Some (if c <> 0 || f = Float.trunc f then c else if f > 0. then -1 else 1)

let compare_constants a b =
  match const_as_integer a, const_as_integer b, Const.view a, Const.view b with
  | Some x, Some y, _, _ -> Some (Z.compare x y)
  | Some n, None, _, Const.Float f -> compare_integer_float n f
  | None, Some n, Const.Float f, _ -> Option.map (fun c -> -c) (compare_integer_float n f)
  | None, None, Const.Float x, Const.Float y ->
      if Float.is_nan x || Float.is_nan y then None else Some (Float.compare x y)
  | _ -> None

let any_invalid args = List.exists (fun c -> Const.view c = Const.Invalid) args

let exec_unary ~truncate_output op (target : Dtype.t) c =
  if Dtype.is_float target then
    match const_as_float c with
    | None -> None
    | Some x ->
        let result = match op with
          | Ops.Neg -> Some (-.x)
          | Ops.Exp2 -> Some (2.0 ** x)
          | Ops.Log2 -> Some (if x > 0.0 then log x /. log 2.0
                             else if x = 0.0 then Float.neg_infinity else Float.nan)
          | Ops.Sqrt -> Some (if x >= 0.0 then sqrt x else Float.nan)
          | Ops.Reciprocal -> Some (1.0 /. x)
          | Ops.Sin -> Some (if Float.is_finite x then sin x else Float.nan)
          | Ops.Trunc -> Some (Float.trunc x)
          | _ -> None
        in
        Option.bind result (fun f -> const_of_target ~truncate_output ~target (`Float f))
  else
    let result = Option.bind (const_as_integer c) (fun x ->
      match op with Ops.Neg -> Some (Z.neg x) | Ops.Trunc -> Some x | _ -> None) in
    Option.bind result (fun n -> const_of_target ~truncate_output ~target (`Int n))

let exec_binary ~truncate_output op (target : Dtype.t) a b =
  if Ops.Group.is_comparison op then
    let comparison = compare_constants a b in
    let result = match op with
      | Ops.Cmpeq -> comparison = Some 0
      | Ops.Cmpne -> comparison <> Some 0
      | Ops.Cmplt -> Option.fold ~none:false ~some:(fun c -> c < 0) comparison
      | _ -> assert false
    in
    Some (Const.bool result)
  else if Dtype.is_float target then
    match const_as_float a, const_as_float b with
    | Some x, Some y ->
        let result = match op with
          | Ops.Add -> Some (x +. y)
          | Ops.Sub -> Some (x -. y)
          | Ops.Mul -> Some (x *. y)
          | Ops.Fdiv -> Some (x /. y)
          | Ops.Max -> Some (if x < y then y else x)
          | Ops.Pow -> Some (if x = 0. && y < 0. then Float.infinity else x ** y)
          | _ -> None
        in
        Option.bind result (fun f -> const_of_target ~truncate_output ~target (`Float f))
    | _ -> None
  else
    match const_as_integer a, const_as_integer b with
    | Some x, Some y ->
        let result = match op with
          | Ops.Add -> Some (Z.add x y)
          | Ops.Sub -> Some (Z.sub x y)
          | Ops.Mul -> Some (Z.mul x y)
          | Ops.Cdiv -> Some (if Z.equal y Z.zero then Z.zero else Z.div x y)
          | Ops.Cmod -> Some (if Z.equal y Z.zero then x else Z.rem x y)
          | Ops.Floordiv -> Some (if Z.equal y Z.zero then Z.zero else Z.fdiv x y)
          | Ops.Floormod -> Some (if Z.equal y Z.zero then x else Z.sub x (Z.mul (Z.fdiv x y) y))
          | Ops.Max -> Some (Z.max x y)
          | Ops.Xor -> Some (Z.logxor x y)
          | Ops.Or -> Some (Z.logor x y)
          | Ops.And -> Some (Z.logand x y)
          | Ops.Shl -> Some (Z.shift_left x (Z.to_int y))
          | Ops.Shr -> Some (Z.shift_right x (Z.to_int y))
          | _ -> None
        in
        Option.bind result (fun n -> const_of_target ~truncate_output ~target (`Int n))
    | _ -> None

let exec_ternary ~truncate_output op (target : Dtype.t) a b c =
  match op with
  | Ops.Where ->
      let condition = match Const.view a with
        | Const.Bool b -> Some b
        | Const.Int n -> Some (Z.sign n <> 0)
        | Const.Float f -> Some (f <> 0.)
        | Const.Invalid -> None
      in
      Option.map (fun condition -> if condition then b else c) condition
  | Ops.Mulacc ->
      if Dtype.is_float target then
        (match const_as_float a, const_as_float b, const_as_float c with
         | Some x, Some y, Some z ->
             const_of_target ~truncate_output ~target (`Float ((x *. y) +. z))
         | _ -> None)
      else
        (match const_as_integer a, const_as_integer b, const_as_integer c with
         | Some x, Some y, Some z ->
             const_of_target ~truncate_output ~target (`Int (Z.add (Z.mul x y) z))
         | _ -> None)
  | _ -> None

let exec_alu ?(truncate_output = true) op (target : Dtype.t) args =
  let is_binary = Ops.Group.is_binary op in
  if is_binary && any_invalid args then Some Const.invalid
  else
    match args with
    | [ a ] when Ops.Group.is_unary op -> exec_unary ~truncate_output op target a
    | [ a; b ] when is_binary -> exec_binary ~truncate_output op target a b
    | [ a; b; c ] when Ops.Group.is_ternary op ->
        exec_ternary ~truncate_output op target a b c
    | _ -> None

let rec infer_int var_vals u =
  match const_int_value (simplify u) with
  | Some n -> n
  | None ->
      let srcs = src u in
      let binary f =
        if Array.length srcs < 2 then raise Not_found;
        f (infer_int var_vals srcs.(0))
          (infer_int var_vals srcs.(1))
      in
      match op u with
      | Ops.Param | Ops.Buffer -> (
          match program_var_name u with
          | Some name ->
              (match List.assoc_opt name var_vals with
               | Some value -> value
               | None -> invalid_arg
                   (Printf.sprintf "program: missing launch variable %S" name))
          | None -> invalid_arg
              "program: unnamed launch variable")
      | Ops.After when is_bound_var u -> infer_int var_vals (Option.get (as_bind u)).value
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
          if infer_int var_vals srcs.(0) <> 0 then
            infer_int var_vals srcs.(1)
          else infer_int var_vals srcs.(2)
      | _ -> raise Not_found

let program_launch_dim var_vals = function
  | Launch_int n -> Launch_value_int n
  | Launch_float f -> Launch_value_float f
  | Launch_sym u -> Launch_value_int (infer_int var_vals u)

let program_launch_dims (info : program_info) ~var_vals =
  ( List.map (program_launch_dim var_vals) info.global_size,
    List.map (program_launch_dim var_vals) info.local_size )

let program_vals (info : program_info) ~var_vals =
  List.map
    (fun var ->
      match program_var_name var with
      | Some name ->
          (match List.assoc_opt name var_vals with
           | Some value -> value
           | None -> invalid_arg
               (Printf.sprintf "program: missing variable %S" name))
      | None -> invalid_arg
          "program: unnamed variable")
    info.vars

(* Serialization *)

(* Uops embedded in node arguments are graph edges just like [src] entries:
   serialization and re-interning must traverse them. The embedding points
   are exactly [Kernel_info] estimates ([Sym]), [Program_info] variables,
   and symbolic [Program_info] launch dimensions ([Launch_sym]); every
   other argument payload is pure data. *)
let arg_uops = function
  | Arg.Kernel_info { estimates = Option.Some { ops; lds; mem }; _ } ->
      let sym acc (e : estimate) =
        match e with Sym u -> u :: acc | Int _ -> acc
      in
      sym (sym (sym [] ops) lds) mem
  | Arg.Program_info { vars; global_size; local_size; _ } ->
      List.fold_left
        (fun acc (d : launch_dim) ->
          match d with
          | Launch_sym u -> u :: acc
          | Launch_int _ | Launch_float _ -> acc)
        vars (global_size @ local_size)
  | _ -> []

let map_arg_uops f = function
  | Arg.Kernel_info ({ estimates = Option.Some e; _ } as ki) ->
      let est (x : estimate) : estimate =
        match x with Sym u -> Sym (f u) | Int _ as x -> x
      in
      Arg.Kernel_info
        { ki with
          estimates =
            Option.Some { ops = est e.ops; lds = est e.lds; mem = est e.mem } }
  | Arg.Program_info pi ->
      let dim (d : launch_dim) =
        match d with
        | Launch_sym u -> Launch_sym (f u)
        | (Launch_int _ | Launch_float _) as d -> d
      in
      Arg.Program_info
        { pi with
          vars = List.map f pi.vars;
          global_size = List.map dim pi.global_size;
          local_size = List.map dim pi.local_size }
  | a -> a

let semantic_key root =
  (* Argument-embedded nodes are edges too. Keep their positions in the
     header and digest their semantics alongside the source edges. *)
  let semantic_arg = function
    | Arg.Kernel_info ({ estimates = Some e; _ } as ki) ->
        let scalar = function Int n -> Int n | Sym _ -> Int 0 in
        let symbolic = function Sym _ -> true | Int _ -> false in
        (Arg.Kernel_info { ki with estimates = Some
            { ops = scalar e.ops; lds = scalar e.lds; mem = scalar e.mem } },
         [ symbolic e.ops; symbolic e.lds; symbolic e.mem ], 0)
    | Arg.Program_info pi ->
        let scalar = function Launch_sym _ -> Launch_int 0 | x -> x in
        let symbolic = function Launch_sym _ -> true | _ -> false in
        (Arg.Program_info { pi with vars = [];
           global_size = List.map scalar pi.global_size;
           local_size = List.map scalar pi.local_size },
         List.map symbolic (pi.global_size @ pi.local_size), List.length pi.vars)
    | Arg.Call_info info -> (Arg.Call_info { info with aux = None }, [], 0)
    | Arg.Param_arg p -> (Arg.Param_arg { p with buffer = None }, [], 0)
    | arg -> (arg, [], 0)
  in
  (* Memoized per call: an unmemoized walk re-digests shared subgraphs and
     goes exponential on graphs with heavy sharing. *)
  let memo : string Ref_tbl.t = Ref_tbl.create 256 in
  let rec key u =
    match Ref_tbl.find_opt memo u with
    | Some k -> k
    | None ->
        let k = compute_key u in
        Ref_tbl.add memo u k;
        k
  and compute_key u =
    (* The header must separate any two nodes whose own payload differs. The
       polymorphic hash is unreliable here: [Hashtbl.hash] stops after 10
       meaningful words (too few to reach a payload buried behind the dtype),
       and the built-in [int64] hash folds the two halves with xor, colliding
       adjacent constants such as [0L] and [-1L]. Hash with a deep traversal
       and render constant payloads exactly. *)
    let header =
      let payload =
        (op u, dtype u, semantic_arg (arg u))
      in
      let value =
        match arg u with Arg.Value c -> Const.to_string c | _ -> ""
      in
      Printf.sprintf "%d.%d.%s"
        (Hashtbl.hash_param 1000 1000 payload)
        (Hashtbl.seeded_hash_param 1000 1000 17 payload)
        value
    in
    let children =
      (Array.to_list (src u) @ arg_uops (arg u))
      |> List.map key |> String.concat ""
    in
    Digest.to_hex (Digest.string (header ^ children))
  in
  key root

let program_signature (info : program_info) linear =
  let buffer_slots = List.mapi (fun i slot -> slot, i) info.globals in
  let argument slot u =
    match op u, arg u with
    | Ops.Param, Arg.Param_arg p ->
        let shape = List.map (fun dim -> Bound.to_int (vmax dim)) (shape u) in
        Tiny_elf.{ name = p.name; slot; dtype = dtype u; shape;
          addrspace = p.addrspace }
    | _ -> invalid_arg "Uop.program_signature: expected a parameter"
  in
  let buffers = List.filter_map (fun u ->
      match op u, arg u with
      | Ops.Param, Arg.Param_arg p when p.addrspace <> Dtype.Alu ->
          (match List.assoc_opt p.slot buffer_slots with
           | Some slot -> Some (argument slot u)
           | None -> invalid_arg "Uop.program_signature: buffer missing from globals")
      | _ -> None) linear in
  let slots = List.map (fun (a : Tiny_elf.argument) -> a.slot) buffers in
  if List.sort Int.compare slots <> List.init (List.length info.globals) Fun.id then
    invalid_arg "Uop.program_signature: globals and linear parameters disagree";
  let scalars = List.mapi (fun i u -> argument (List.length info.globals + i) u)
      info.vars in
  buffers @ scalars

let program_function_name u =
  match op u, children u with
  | Ops.Program, sink :: _ when op sink = Ops.Sink ->
      (match as_kernel_info sink with
       | Some kernel -> kernel_function_name kernel
       | None -> "test")
  | _ -> invalid_arg "Uop.program_function_name: expected a PROGRAM with a SINK"

let to_elf u =
  match op u, arg u, children u with
  | Ops.Program, Arg.Program_info info, [sink; linear; _source; binary]
    when op sink = Ops.Sink && op linear = Ops.Linear && op binary = Ops.Binary ->
      let name = program_function_name u in
      let lib = match Arg.as_string (arg binary) with
        | Some lib -> Bytes.of_string lib
        | None -> invalid_arg "Uop.to_elf: binary is not a byte string" in
      Tiny_elf.{ lib; name; target = info.target;
        signature = program_signature info (children linear);
        profile_key = Some (semantic_key u) }
  | _ -> invalid_arg "Uop.to_elf: expected a compiled PROGRAM"

let export_magic = "TOLKUOP\x00"
let export_version = 25

type serialized_node = {
  serialized_op : Ops.t;
  serialized_dtype : Dtype.t;
  serialized_src : int array;
  serialized_arg : arg;
  serialized_tag : string option;
  serialized_buffers : int list option;
}

let export root =
  let ids = Ref_tbl.create 512 in
  let nodes = ref [] and buffers = ref [] in
  let buffer_ids = Hashtbl.create 16 in
  let buffer_id buf =
    match Hashtbl.find_opt buffer_ids (Storage.id buf) with
    | Some id -> id
    | None ->
        let id = Hashtbl.length buffer_ids in
        Hashtbl.add buffer_ids (Storage.id buf) id;
        buffers := buf :: !buffers;
        id
  in
  let stack = Stack.create () in
  Stack.push (root, false) stack;
  while not (Stack.is_empty stack) do
    let u, expanded = Stack.pop stack in
    if not (Ref_tbl.mem ids u) then
      if expanded then begin
        let arg, owned = match arg u with
          | Arg.Call_info { grad_fxn = Some _; _ } ->
              invalid_arg "Uop.export: graph carries a gradient function"
          | Arg.Param_arg p ->
              Arg.Param_arg { p with buffer = None },
              Option.map (List.map buffer_id) p.buffer
          | arg -> arg, None
        in
        let node = {
          serialized_op = op u; serialized_dtype = dtype u;
          serialized_src = Array.map (Ref_tbl.find ids) (src u);
          serialized_arg = map_arg_uops (fun u -> const_int (Ref_tbl.find ids u)) arg;
          serialized_tag = node_tag u; serialized_buffers = owned;
        } in
        Ref_tbl.add ids u (Ref_tbl.length ids);
        nodes := node :: !nodes
      end else begin
        Stack.push (u, true) stack;
        Array.iter (fun c -> Stack.push (c, false) stack) (src u);
        List.iter (fun c -> Stack.push (c, false) stack) (arg_uops (arg u))
      end
  done;
  let data = Array.of_list (List.rev !nodes), Storage.snapshot (List.rev !buffers) in
  String.concat ""
    [export_magic; Marshal.to_string export_version []; Marshal.to_string data []]

let import s =
  let error () = failwith "Uop.import: malformed input" in
  let magic_len = String.length export_magic in
  let len = String.length s in
  if len < magic_len
     || not (String.equal (String.sub s 0 magic_len) export_magic)
  then error ();
  (* Validate each block's advertised size against the available bytes
     before unmarshalling, so truncated input fails cleanly. *)
  let bytes = Bytes.unsafe_of_string s in
  let block_size ofs =
    if len - ofs < Marshal.header_size then error ();
    match Marshal.total_size bytes ofs with
    | exception Failure _ -> error ()
    | size ->
        if size > len - ofs then error ();
        size
  in
  let version_ofs = magic_len in
  let version_size = block_size version_ofs in
  let version : int =
    try Marshal.from_string s version_ofs with Failure _ -> error ()
  in
  if version <> export_version then
    failwith (Printf.sprintf "Uop.import: unsupported format version %d" version);
  let graph_ofs = version_ofs + version_size in
  ignore (block_size graph_ofs : int);
  let (serialized, snapshots : serialized_node array * Storage.snapshot list) =
    try Marshal.from_string s graph_ofs with Failure _ -> error ()
  in
  if Array.length serialized = 0 then error ();
  let buffers = Array.of_list (Storage.of_snapshot snapshots) in
  let nodes = Array.make (Array.length serialized) (const_int 0) in
  Array.iteri (fun i n ->
      let node id = if id < 0 || id >= i then error (); nodes.(id) in
      let arg = map_arg_uops (fun u -> match const_int_value u with
          | Some id -> node id | None -> error ()) n.serialized_arg in
      let arg = match arg, n.serialized_buffers with
        | Arg.Param_arg p, Some ids ->
            let buffer = List.map (fun id ->
                if id < 0 || id >= Array.length buffers then error ();
                buffers.(id)) ids in
            Arg.Param_arg { p with buffer = Some buffer }
        | _, None -> arg
        | _ -> error ()
      in
      nodes.(i) <- intern_node {
        op = n.serialized_op; dtype = n.serialized_dtype;
        src = Array.map node n.serialized_src; arg;
        node_tag = n.serialized_tag;
      }) serialized;
  nodes.(Array.length nodes - 1)

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
