(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* Lists *)

(* The first [n] elements of a list and the rest, as Python's slices: short
   lists give what they have. *)
let rec take n l =
  if n <= 0 then [] else match l with [] -> [] | x :: r -> x :: take (n - 1) r

let rec drop n l =
  if n <= 0 then l else match l with [] -> [] | _ :: r -> drop (n - 1) r

(* Axis types *)

module Axis_type = struct
  type t =
    | Device
    | Global
    | Warp
    | Local
    | Weak
    | Reduce
    | Upcast
    | Unroll
    | Placeholder
    | Loop

  let all =
    [
      Device;
      Global;
      Warp;
      Local;
      Weak;
      Reduce;
      Upcast;
      Unroll;
      Placeholder;
      Loop;
    ]

  let to_int = function
    | Device -> 0
    | Global -> 1
    | Warp -> 2
    | Local -> 3
    | Weak -> 4
    | Reduce -> 5
    | Upcast -> 6
    | Unroll -> 7
    | Placeholder -> 8
    | Loop -> 9

  let equal (a0 : t) a1 = a0 = a1
  let compare a0 a1 = Int.compare (to_int a0) (to_int a1)

  let name = function
    | Device -> "DEVICE"
    | Global -> "GLOBAL"
    | Warp -> "WARP"
    | Local -> "LOCAL"
    | Weak -> "WEAK"
    | Reduce -> "REDUCE"
    | Upcast -> "UPCAST"
    | Unroll -> "UNROLL"
    | Placeholder -> "PLACEHOLDER"
    | Loop -> "LOOP"

  let of_string s =
    match List.find_opt (fun a -> String.equal (name a) s) all with
    | Some a -> Ok a
    | None -> Error (strf "unknown axis type %S" s)

  let no_placeholder () = invalid_arg "the placeholder axis type has no letter"

  let letter = function
    | Device -> "d"
    | Global -> "g"
    | Local -> "l"
    | Warp -> "w"
    | Weak | Loop -> "L"
    | Upcast -> "u"
    | Reduce -> "R"
    | Unroll -> "r"
    | Placeholder -> no_placeholder ()

  let color : t -> Helpers.color = function
    | Device -> Green
    | Global -> Blue
    | Local -> Cyan
    | Warp -> Bright_cyan
    | Weak | Loop -> Bright_white
    | Upcast -> Yellow
    | Reduce -> Red
    | Unroll -> Magenta
    | Placeholder -> no_placeholder ()

  let position = function
    | Device -> -2
    | Weak | Loop -> -1
    | Global -> 0
    | Warp -> 1
    | Local -> 2
    | Upcast -> 3
    | Reduce -> 4
    | Unroll -> 5
    | Placeholder -> no_placeholder ()

  let repr a = "AxisType." ^ name a
  let pp ppf a = Format.pp_print_string ppf (repr a)
end

(* Python literals *)

(* Arguments and tags print as the Python literals that denote them, so that
   listings match tinygrad's. *)

let repr_quoted ~bytes s =
  let quote =
    if String.contains s '\'' && not (String.contains s '"') then '"' else '\''
  in
  let b = Buffer.create (String.length s + 2) in
  if bytes then Buffer.add_char b 'b';
  Buffer.add_char b quote;
  String.iter
    (fun c ->
      match c with
      | '\\' -> Buffer.add_string b "\\\\"
      | '\n' -> Buffer.add_string b "\\n"
      | '\r' -> Buffer.add_string b "\\r"
      | '\t' -> Buffer.add_string b "\\t"
      | c when c = quote ->
          Buffer.add_char b '\\';
          Buffer.add_char b c
      | c
        when Char.code c < 0x20
             || Char.code c = 0x7f
             || (bytes && Char.code c > 0x7f) ->
          Buffer.add_string b (strf "\\x%02x" (Char.code c))
      | c -> Buffer.add_char b c)
    s;
  Buffer.add_char b quote;
  Buffer.contents b

let repr_string s = repr_quoted ~bytes:false s
let repr_bytes s = repr_quoted ~bytes:true s

let repr_tuple = function
  | [ x ] -> "(" ^ x ^ ",)"
  | xs -> "(" ^ String.concat ", " xs ^ ")"

let repr_option f = function None -> "None" | Some x -> f x
let repr_bool b = if b then "True" else "False"
let repr_const c = Format.asprintf "%a" Dtype.pp_const c
let repr_dtype dt = Format.asprintf "%a" Dtype.pp dt
let repr_addr_space a = Format.asprintf "%a" Dtype.pp_addr_space a

(* Devices *)

type device = Single of string | Multi of string list

let equal_device d0 d1 =
  match (d0, d1) with
  | Single s0, Single s1 -> String.equal s0 s1
  | Multi l0, Multi l1 -> List.equal String.equal l0 l1
  | _ -> false

let repr_device = function
  | Single s -> repr_string s
  | Multi l -> repr_tuple (List.map repr_string l)

let pp_device ppf d = Format.pp_print_string ppf (repr_device d)
let device_names = function Single s -> [ s ] | Multi l -> l

let is_disk_device d =
  List.exists
    (fun n ->
      let kind =
        match String.index_opt n ':' with
        | Some i -> String.sub n 0 i
        | None -> n
      in
      String.equal (String.uppercase_ascii kind) "DISK")
    (device_names d)

(* Tags *)

module Tag = struct
  type t =
    | Bool of bool
    | Int of int
    | String of string
    | Bytes of string
    | Dtype of Dtype.t
    | Tuple of t list

  let rec equal g0 g1 =
    match (g0, g1) with
    | Bool b0, Bool b1 -> Bool.equal b0 b1
    | Int n0, Int n1 -> Int.equal n0 n1
    | String s0, String s1 | Bytes s0, Bytes s1 -> String.equal s0 s1
    | Dtype d0, Dtype d1 -> Dtype.equal d0 d1
    | Tuple l0, Tuple l1 -> List.equal equal l0 l1
    | _ -> false

  let hash = Hashtbl.hash

  let rec repr = function
    | Bool b -> repr_bool b
    | Int n -> string_of_int n
    | String s -> repr_string s
    | Bytes s -> repr_bytes s
    | Dtype d -> repr_dtype d
    | Tuple l -> repr_tuple (List.map repr l)

  let pp ppf g = Format.pp_print_string ppf (repr g)
end

(* Arguments *)

type param_arg = {
  slot : int;
  dtype : Dtype.t;
  size : int option;
  vmin_vmax : (Dtype.value * Dtype.value) option;
  multiple_of : int option;
  name : string option;
  addrspace : Dtype.addr_space option;
  device : device option;
  volatile : bool;
  bind_on_realize : bool;
  bound : Dtype.value option;
  phase : int;
  align : int;
}

type keep = Removable | Broadcast | Whole

type bufferize_opts = {
  device : device option;
  addrspace : Dtype.addr_space;
  keep : keep;
}

type wmma = {
  dims : int * int * int;
  dtype_in : Dtype.t;
  threads : int;
  upcast_axes :
    ((int list * int) list * (int list * int) list * (int list * int) list)
    option;
}

(* The payloads of calls name nodes, and nodes carry them: the two groups of
   types are recursive modules, since one group of types cannot repeat a field
   name. *)
module rec Calls : sig
  type hcq_kernel = {
    devices : string list;
    name : string;
    estimates : Node.estimates;
    stamps : int list;
    profile_key : string option;
    input_slots : int list;
    outs : int list;
    ins : int list;
  }

  type hcq_info = {
    device : string list;
    kernels : hcq_kernel list;
    estimates : Node.estimates;
    nargs : int;
    table : int;
    inputs : (Node.t * int * string) list;
    slots : (string * int) list;
    written_bufs : Node.t list;
    writes : Node.t list;
    copies : (string * string * int) list;
  }

  type call_info = {
    name : string option;
    precompile : bool;
    aux : hcq_info option;
    dtype : Dtype.t;
  }
end =
  Calls

and Node : sig
  type t = {
    op : Op.t;
    src : t list;
    arg : arg;
    tag : Tag.t option;
    dtype : Dtype.t;
    id : int;
    (* Properties computed on first use. Each is a function of the node, so
       domains racing to fill one write the same value; the fields are atomic so
       that a reader that sees a value sees it whole. *)
    mutable shape_memo : sint list option option; [@atomic]
    mutable ranges_memo : nodes option; [@atomic]
    mutable ended_ranges_memo : t list option; [@atomic]
    mutable min_max_memo : (Dtype.value * Dtype.value) option; [@atomic]
    mutable device_memo : device option option; [@atomic]
    mutable addrspace_memo : Dtype.addr_space option option; [@atomic]
    mutable backward_slice_memo : nodes option; [@atomic]
    mutable axis_memo : int option option; [@atomic]
    mutable marg_memo : movement option; [@atomic]
    mutable key_memo : string option; [@atomic]
    mutable arg_repr_memo : string option; [@atomic]
    mutable src_ops_memo : Op.Set.t option; [@atomic]
  }

  (* A set in the order its nodes joined it. Small sets are searched in order;
     larger ones carry a table of their nodes' ids. *)
  and nodes = {
    order : t list;
    cardinal : int;
    ids : (int, unit) Hashtbl.t option;
  }

  and sint = Int of int | Sym of t
  and estimates = { ops : sint; lds : sint; mem : sint }
  and split = { iterations : sint; lo : int; hi : int }

  and kernel_info = {
    name : string;
    applied_opts : Opt.t list;
    opts_to_apply : Opt.t list option;
    estimates : estimates option;
    beam : int;
    split : split option;
  }

  and program_info = {
    global_size : sint list;
    local_size : sint list;
    vars : t list;
    globals : int list;
    outs : int list;
    ins : int list;
    target : Helpers.Target.t;
  }

  and arg =
    | No_arg
    | Const of Dtype.const
    | Dtype of Dtype.t
    | Param of param_arg
    | Range of { axis_id : int list; axis_type : Axis_type.t }
    | Reduce of { op : Op.t; num_axes : int }
    | Allreduce of { op : Op.t; device : device }
    | Device of device
    | Shard of int
    | Axes of int list
    | Flips of bool list
    | String of string
    | Bytes of string
    | Queue of { devices : string list; queue : string }
    | Region of { name : string; align : int }
    | Code of { code : string; dtype : Dtype.t }
    | Bufferize of bufferize_opts
    | Kernel of kernel_info
    | Program of program_info
    | Call of Calls.call_info
    | Wmma of wmma

  and movement =
    | Reshape of sint list
    | Expand of sint list
    | Pad of (sint * sint) list
    | Shrink of (sint * sint) list
    | Permute of int list
    | Flip of bool list
end =
  Node

include Calls
include Node

(* Equality and hashing of arguments. Nodes inside them compare by [==]:
   polymorphic equality would walk whole graphs and their memo fields. *)

let equal_sint s0 s1 =
  match (s0, s1) with
  | Int n0, Int n1 -> Int.equal n0 n1
  | Sym u0, Sym u1 -> u0 == u1
  | _ -> false

let hash_sint = function Int n -> Hashtbl.hash n | Sym u -> Hashtbl.hash u.id

let equal_value_pair (a0, b0) (a1, b1) =
  Dtype.equal_const a0 a1 && Dtype.equal_const b0 b1

let equal_param_arg (p0 : param_arg) (p1 : param_arg) =
  Int.equal p0.slot p1.slot
  && Dtype.equal p0.dtype p1.dtype
  && Option.equal Int.equal p0.size p1.size
  && Option.equal equal_value_pair p0.vmin_vmax p1.vmin_vmax
  && Option.equal Int.equal p0.multiple_of p1.multiple_of
  && Option.equal String.equal p0.name p1.name
  && Option.equal ( = ) p0.addrspace p1.addrspace
  && Option.equal equal_device p0.device p1.device
  && Bool.equal p0.volatile p1.volatile
  && Bool.equal p0.bind_on_realize p1.bind_on_realize
  && Option.equal Dtype.equal_const p0.bound p1.bound
  && Int.equal p0.phase p1.phase
  && Int.equal p0.align p1.align

let equal_estimates (e0 : estimates) (e1 : estimates) =
  equal_sint e0.ops e1.ops && equal_sint e0.lds e1.lds
  && equal_sint e0.mem e1.mem

let equal_split (s0 : split) (s1 : split) =
  equal_sint s0.iterations s1.iterations && s0.lo = s1.lo && s0.hi = s1.hi

let equal_kernel_info (k0 : kernel_info) (k1 : kernel_info) =
  String.equal k0.name k1.name
  && List.equal Opt.equal k0.applied_opts k1.applied_opts
  && Option.equal (List.equal Opt.equal) k0.opts_to_apply k1.opts_to_apply
  && Option.equal equal_estimates k0.estimates k1.estimates
  && Int.equal k0.beam k1.beam
  && Option.equal equal_split k0.split k1.split

let equal_program_info (p0 : program_info) (p1 : program_info) =
  List.equal equal_sint p0.global_size p1.global_size
  && List.equal equal_sint p0.local_size p1.local_size
  && List.equal ( == ) p0.vars p1.vars
  && p0.globals = p1.globals && p0.outs = p1.outs && p0.ins = p1.ins
  && p0.target = p1.target

let equal_bufferize_opts (b0 : bufferize_opts) (b1 : bufferize_opts) =
  Option.equal equal_device b0.device b1.device
  && b0.addrspace = b1.addrspace
  && b0.keep = b1.keep

let equal_hcq_kernel (k0 : hcq_kernel) (k1 : hcq_kernel) =
  k0.devices = k1.devices
  && String.equal k0.name k1.name
  && equal_estimates k0.estimates k1.estimates
  && k0.stamps = k1.stamps
  && Option.equal String.equal k0.profile_key k1.profile_key
  && k0.input_slots = k1.input_slots
  && k0.outs = k1.outs && k0.ins = k1.ins

let equal_hcq_info (h0 : hcq_info) (h1 : hcq_info) =
  h0.device = h1.device
  && List.equal equal_hcq_kernel h0.kernels h1.kernels
  && equal_estimates h0.estimates h1.estimates
  && Int.equal h0.nargs h1.nargs
  && Int.equal h0.table h1.table
  && List.equal
       (fun (u0, i0, a0) (u1, i1, a1) ->
         u0 == u1 && Int.equal i0 i1 && String.equal a0 a1)
       h0.inputs h1.inputs
  && h0.slots = h1.slots
  && List.equal ( == ) h0.written_bufs h1.written_bufs
  && List.equal ( == ) h0.writes h1.writes
  && h0.copies = h1.copies

let equal_call_info (c0 : call_info) (c1 : call_info) =
  Option.equal String.equal c0.name c1.name
  && Bool.equal c0.precompile c1.precompile
  && Option.equal equal_hcq_info c0.aux c1.aux
  && Dtype.equal c0.dtype c1.dtype

let equal_arg (a0 : arg) (a1 : arg) =
  match (a0, a1) with
  | No_arg, No_arg -> true
  | Const c0, Const c1 -> Dtype.equal_const c0 c1
  | Dtype d0, Dtype d1 -> Dtype.equal d0 d1
  | Param p0, Param p1 -> equal_param_arg p0 p1
  | Range r0, Range r1 ->
      r0.axis_id = r1.axis_id && Axis_type.equal r0.axis_type r1.axis_type
  | Reduce r0, Reduce r1 ->
      Op.equal r0.op r1.op && Int.equal r0.num_axes r1.num_axes
  | Allreduce r0, Allreduce r1 ->
      Op.equal r0.op r1.op && equal_device r0.device r1.device
  | Device d0, Device d1 -> equal_device d0 d1
  | Shard n0, Shard n1 -> Int.equal n0 n1
  | Axes l0, Axes l1 -> l0 = l1
  | Flips l0, Flips l1 -> l0 = l1
  | String s0, String s1 | Bytes s0, Bytes s1 -> String.equal s0 s1
  | Queue q0, Queue q1 ->
      List.equal String.equal q0.devices q1.devices
      && String.equal q0.queue q1.queue
  | Region r0, Region r1 -> String.equal r0.name r1.name && r0.align = r1.align
  | Code c0, Code c1 ->
      String.equal c0.code c1.code && Dtype.equal c0.dtype c1.dtype
  | Bufferize b0, Bufferize b1 -> equal_bufferize_opts b0 b1
  | Kernel k0, Kernel k1 -> equal_kernel_info k0 k1
  | Program p0, Program p1 -> equal_program_info p0 p1
  | Call c0, Call c1 -> equal_call_info c0 c1
  | Wmma w0, Wmma w1 ->
      w0.dims = w1.dims
      && Dtype.equal w0.dtype_in w1.dtype_in
      && Int.equal w0.threads w1.threads
      && w0.upcast_axes = w1.upcast_axes
  | _ -> false

let hash_arg (a : arg) =
  let h = Hashtbl.hash in
  match a with
  | No_arg -> 0
  | Const c -> h (1, Dtype.hash_const c)
  | Dtype d -> h (2, Dtype.hash d)
  | Param p -> h (3, p.slot, p.size, p.name, p.device)
  | Range r -> h (4, r.axis_id, r.axis_type)
  | Reduce r -> h (5, Op.to_int r.op, r.num_axes)
  | Allreduce r -> h (6, Op.to_int r.op, r.device)
  | Device d -> h (7, d)
  | Shard n -> h (8, n)
  | Axes l -> h (9, l)
  | Flips l -> h (10, l)
  | String s -> h (11, s)
  | Bytes s -> h (12, s)
  | Queue q -> h (13, q.devices, q.queue)
  | Region r -> h (20, r.name, r.align)
  | Code c -> h (14, c.code, Dtype.hash c.dtype)
  | Bufferize b -> h (15, b.device, b.keep)
  | Kernel k -> h (16, k.name, k.beam)
  | Program p -> h (17, p.globals, List.map hash_sint p.global_size)
  | Call c -> h (18, c.name, c.precompile)
  | Wmma w -> h (19, w.dims, w.threads)

(* Hash-consing *)

module Interned = struct
  type nonrec t = t

  let equal u0 u1 =
    Op.equal u0.op u1.op
    && List.equal ( == ) u0.src u1.src
    && equal_arg u0.arg u1.arg
    && Option.equal Tag.equal u0.tag u1.tag

  let hash u =
    List.fold_left
      (fun h s -> (h * 31) + s.id)
      (Hashtbl.hash (Op.to_int u.op, hash_arg u.arg, Option.map Tag.hash u.tag))
      u.src
end

module Table = Stdlib.Weak.Make (Interned)

(* The table is split in shards, each behind its own lock, so that domains
   building nodes rarely wait for each other. Each shard starts at the least
   size and grows with its nodes, so that a program that builds none keeps no
   table in the heap every major collection marks. *)
let shards = Array.init 64 (fun _ -> (Table.create 0, Mutex.create ()))
let next_id = Atomic.make 0

let node op src arg tag dtype id =
  {
    op;
    src;
    arg;
    tag;
    dtype;
    id;
    shape_memo = None;
    ranges_memo = None;
    ended_ranges_memo = None;
    min_max_memo = None;
    device_memo = None;
    addrspace_memo = None;
    backward_slice_memo = None;
    axis_memo = None;
    marg_memo = None;
    key_memo = None;
    arg_repr_memo = None;
    src_ops_memo = None;
  }

(* The whole-specification check that construction runs when the setting [SPEC]
   is 2 or more; [Spec] installs it. Nodes built while the library is
   initialised, before [Spec] installs it, are not checked. *)
let construction_check : (t -> unit) option Atomic.t = Atomic.make None

(* Data types *)

let first op = function
  | s :: _ -> s
  | [] -> invalid_argf "%s needs a source" (Op.name op)

let is_const_invalid u =
  u.op = Op.Const && match u.arg with Const `Invalid -> true | _ -> false

let rec base u =
  if Op.Set.mem u.op Op.Set.movement || u.op = Op.Detach then
    base (first u.op u.src)
  else u

let promo_dtype us =
  match us with
  | [] -> invalid_arg "no types to promote"
  | u :: rest ->
      if List.for_all (fun s -> Dtype.equal s.dtype u.dtype) rest then u.dtype
      else Dtype.least_upper (List.map (fun s -> s.dtype) us)

let dtype_of op src arg =
  let mismatch what =
    invalid_argf "%s needs %s as its argument" (Op.name op) what
  in
  match op with
  | Op.Store | Op.Linear | Op.Sink | Op.Program | Op.Source | Op.Backedge
  | Op.Barrier | Op.Group | Op.If | Op.Endif | Op.Noop | Op.Custom_function ->
      Dtype.Void
  | Op.Call -> ( match arg with Call c -> c.dtype | _ -> Dtype.Void)
  | Op.Custom | Op.Customi | Op.Ins -> (
      match arg with Code c -> c.dtype | _ -> mismatch "code and a type")
  | Op.Index -> (first op src).dtype
  | Op.Load | Op.Unshard | Op.Reduce | Op.After | Op.Range
  | Op.Contiguous_backward | Op.Copy | Op.Stage | Op.Detach | Op.Mstack
  | Op.Mselect | Op.Allreduce | Op.Special | Op.End ->
      (first op src).dtype
  | Op.Cmplt | Op.Cmpne | Op.Cmpeq -> Dtype.Bool
  | Op.Sin | Op.Log2 | Op.Exp2 | Op.Sqrt | Op.Reciprocal ->
      let x = first op src in
      if is_const_invalid (base x) then Dtype.Bool
      else Dtype.least_upper_float x.dtype
  | Op.Where -> (
      match src with
      | c :: branches ->
          if not (Dtype.equal c.dtype Dtype.Bool) then
            invalid_argf "the condition of a where is %s, not dtypes.bool"
              (repr_dtype c.dtype);
          promo_dtype branches
      | [] -> invalid_arg "where needs a condition")
  | Op.Stack -> if List.is_empty src then Dtype.Void else promo_dtype src
  | Op.Wmma -> (
      match src with
      | [ _; _; acc ] -> acc.dtype
      | _ -> invalid_arg "wmma needs three sources")
  | Op.Getaddr | Op.Threefry -> Dtype.Uint64
  | Op.Fdiv -> Dtype.least_upper_float (promo_dtype src)
  | Op.Shl | Op.Shr ->
      if
        not
          (List.for_all
             (fun x -> Dtype.is_int x.dtype || is_const_invalid (base x))
             src)
      then
        invalid_argf "shift operands must be integers, not %s"
          (String.concat ", " (List.map (fun x -> repr_dtype x.dtype) src));
      (first op src).dtype
  | Op.Buffer | Op.Alloc | Op.Param -> (
      match arg with Param p -> p.dtype | _ -> mismatch "a ParamArg")
  | Op.Binary -> Dtype.Uint8
  | Op.Cast | Op.Bitcast -> (
      match arg with Dtype dt -> dt | _ -> mismatch "a type")
  | Op.Const -> (
      match arg with Const c -> Dtype.of_const c | _ -> mismatch "a constant")
  | op when Op.Set.mem op Op.Set.unary -> (first op src).dtype
  | op when Op.Set.mem op Op.Set.broadcastable -> promo_dtype src
  | op when Op.Set.mem op Op.Set.movement -> (first op src).dtype
  | op -> invalid_argf "%s has no type" (Op.name op)

(* Nodes *)

let v ?(src = []) ?(arg = No_arg) ?tag op =
  let probe = node op src arg tag Dtype.Void (-1) in
  let created = ref false in
  let table, lock = shards.(Interned.hash probe land 63) in
  let u =
    Mutex.protect lock (fun () ->
        match Table.find_opt table probe with
        | Some u -> u
        | None ->
            let u =
              node op src arg tag (dtype_of op src arg)
                (Atomic.fetch_and_add next_id 1)
            in
            Table.add table u;
            created := true;
            u)
  in
  (if !created && Helpers.Context_var.value Helpers.spec > 1 then
     match Atomic.get construction_check with
     | Some check -> check u
     | None -> ());
  u

let op u = u.op
let dtype u = u.dtype
let src u = u.src
let arg u = u.arg
let tag u = u.tag

let nth u i =
  match List.nth_opt u.src i with
  | Some s -> s
  | None -> invalid_argf "%s has no source %d" (Op.name u.op) i

let replace ?op ?src ?arg ?tag u =
  let op = Option.value op ~default:u.op
  and src = Option.value src ~default:u.src
  and arg = Option.value arg ~default:u.arg
  and tag = Option.value tag ~default:u.tag in
  if
    Op.equal op u.op
    && List.equal ( == ) src u.src
    && equal_arg arg u.arg
    && Option.equal Tag.equal tag u.tag
  then u
  else v ~src ~arg ?tag op

let rtag ?(tag = Tag.Bool true) u = replace ~tag:(Some tag) u
let equal u0 u1 = u0 == u1
let compare u0 u1 = Int.compare u0.id u1.id
let hash u = u.id

module Key = struct
  type nonrec t = t

  let equal = ( == )
  let hash u = u.id
end

module Tbl = Hashtbl.Make (Key)

let dedup_nodes l = Helpers.dedup (module Key) l

module Nodes = struct
  type t = nodes

  let of_list order =
    let cardinal = List.length order in
    let ids =
      if cardinal <= 8 then None
      else
        let ids = Hashtbl.create cardinal in
        List.iter (fun u -> Hashtbl.replace ids u.id ()) order;
        Some ids
    in
    { order; cardinal; ids }

  let mem u s =
    match s.ids with
    | Some ids -> Hashtbl.mem ids u.id
    | None -> List.memq u s.order

  let to_list s = s.order
  let fold f s acc = List.fold_left (fun acc u -> f u acc) acc s.order
  let cardinal s = s.cardinal
end

(* Printing *)

let repr_sint_with repr_node = function
  | Int n -> string_of_int n
  | Sym u -> repr_node u

let repr_opt o = Format.asprintf "%a" Opt.pp o
let repr_value_pair (a, b) = repr_tuple [ repr_const a; repr_const b ]

let repr_param_arg (p : param_arg) =
  let fields =
    [
      ("vmin_vmax", Option.map repr_value_pair p.vmin_vmax);
      ("multiple_of", Option.map string_of_int p.multiple_of);
      ("name", Option.map repr_string p.name);
      ( "addrspace",
        match p.addrspace with
        | Some Dtype.Global -> None
        | a -> Some (repr_option repr_addr_space a) );
      ("device", Option.map repr_device p.device);
      ("volatile", if p.volatile then Some "True" else None);
      ("bind_on_realize", if p.bind_on_realize then Some "True" else None);
      ("val", Option.map repr_const p.bound);
      ("phase", if p.phase = 0 then None else Some (string_of_int p.phase));
      ("align", if p.align = 16 then None else Some (string_of_int p.align));
    ]
  in
  let args =
    [ string_of_int p.slot; repr_dtype p.dtype ]
    @ Option.to_list (Option.map string_of_int p.size)
    @ List.filter_map (fun (k, v) -> Option.map (fun v -> k ^ "=" ^ v) v) fields
  in
  strf "ParamArg(%s)" (String.concat ", " args)

let pp_param_arg ppf p = Format.pp_print_string ppf (repr_param_arg p)

let repr_bufferize_opts (b : bufferize_opts) =
  strf "BufferizeOpts(device=%s, addrspace=%s, removable=%s, broadcast=%s)"
    (repr_option repr_device b.device)
    (repr_addr_space b.addrspace)
    (repr_bool (b.keep <> Whole))
    (repr_bool (b.keep = Broadcast))

let pp_bufferize_opts ppf b = Format.pp_print_string ppf (repr_bufferize_opts b)

let repr_call_info (c : call_info) =
  strf "CallInfo(None, %s, %s, False%s)"
    (repr_option repr_string c.name)
    (repr_bool c.precompile)
    (if Dtype.equal c.dtype Dtype.Void then ""
     else ", dtype=" ^ repr_dtype c.dtype)

let pp_call_info ppf c = Format.pp_print_string ppf (repr_call_info c)

let repr_wmma w =
  let m, n, k = w.dims in
  let repr_axes axes =
    repr_tuple
      (List.map
         (fun (ids, size) ->
           repr_tuple
             [ repr_tuple (List.map string_of_int ids); string_of_int size ])
         axes)
  in
  repr_tuple
    [
      repr_tuple (List.map string_of_int [ m; n; k ]);
      repr_dtype w.dtype_in;
      string_of_int w.threads;
      repr_option
        (fun (a, b, c) -> repr_tuple (List.map repr_axes [ a; b; c ]))
        w.upcast_axes;
    ]

(* A node prints as the constructor call that rebuilds it. A node reached more
   than once is named [xN:=] where first printed and [xN] afterwards. *)
let rec repr u =
  let cache = Tbl.create 64 in
  let rec count u =
    List.iter
      (fun s ->
        match Tbl.find_opt cache s with
        | Some (i, n, p) -> Tbl.replace cache s (i, n + 1, p)
        | None ->
            Tbl.replace cache s (Tbl.length cache, 1, false);
            count s)
      u.src
  in
  count u;
  let b = Buffer.create 256 in
  let rec print d u =
    let i, n, printed =
      match Tbl.find_opt cache u with Some e -> e | None -> (0, 0, false)
    in
    Buffer.add_string b (String.make d ' ');
    if printed then Buffer.add_string b (strf "x%d" i)
    else begin
      Tbl.replace cache u (i, n, true);
      if n > 1 then Buffer.add_string b (strf "x%d:=" i);
      Buffer.add_string b
        (strf "UOp(%s, arg=%s%s, src=("
           (Format.asprintf "%a" Op.pp u.op)
           (arg_repr u) (tag_str u));
      List.iter
        (fun s ->
          Buffer.add_char b '\n';
          print (d + 2) s;
          Buffer.add_char b ',')
        u.src;
      Buffer.add_string b "))"
    end
  in
  print 0 u;
  Buffer.contents b

and tag_str u =
  match u.tag with
  | None -> ""
  | Some (Tag.String s) -> ", tag=" ^ s
  | Some g -> ", tag=" ^ Tag.repr g

and arg_repr u =
  match u.arg_repr_memo with
  | Some s -> s
  | None ->
      let s = repr_arg u.arg in
      u.arg_repr_memo <- Some s;
      s

and repr_sint s = repr_sint_with repr s

and repr_estimates (e : estimates) =
  strf "Estimates(ops=%s, lds=%s, mem=%s)" (repr_sint e.ops) (repr_sint e.lds)
    (repr_sint e.mem)

and repr_kernel_info (k : kernel_info) =
  strf
    "KernelInfo(name=%s, applied_opts=%s, opts_to_apply=%s, estimates=%s, \
     beam=%d, split=%s)"
    (repr_string k.name)
    (repr_tuple (List.map repr_opt k.applied_opts))
    (repr_option (fun l -> repr_tuple (List.map repr_opt l)) k.opts_to_apply)
    (repr_option repr_estimates k.estimates)
    k.beam
    (repr_option
       (fun s ->
         repr_tuple
           [ repr_sint s.iterations; string_of_int s.lo; string_of_int s.hi ])
       k.split)

and repr_program_info (p : program_info) =
  let ints l = repr_tuple (List.map string_of_int l) in
  strf
    "ProgramInfo(global_size=%s, local_size=%s, vars=%s, globals=%s, outs=%s, \
     ins=%s, target=%s)"
    (repr_tuple (List.map repr_sint p.global_size))
    (repr_tuple (List.map repr_sint p.local_size))
    (repr_tuple (List.map repr p.vars))
    (ints p.globals) (ints p.outs) (ints p.ins)
    (Format.asprintf "%a" Helpers.Target.pp p.target)

and repr_hcq_kernel (k : hcq_kernel) =
  let ints l = repr_tuple (List.map string_of_int l) in
  repr_tuple
    [
      repr_tuple (List.map repr_string k.devices);
      repr_string k.name;
      repr_estimates k.estimates;
      ints k.stamps;
      repr_option repr_bytes k.profile_key;
      ints k.input_slots;
      repr_tuple [ ints k.outs; ints k.ins ];
    ]

and repr_hcq_info (h : hcq_info) =
  let pair f g (a, b) = repr_tuple [ f a; g b ] in
  strf
    "HCQInfo(device=%s, kernels=%s, estimates=%s, nargs=%d, table=%d, \
     inputs=%s, slots=%s, written_bufs=%s)"
    (repr_tuple (List.map repr_string h.device))
    (repr_tuple (List.map repr_hcq_kernel h.kernels))
    (repr_estimates h.estimates)
    h.nargs h.table
    (repr_tuple
       (List.map
          (fun (u, slot, space) ->
            repr_tuple [ repr u; string_of_int slot; repr_string space ])
          h.inputs))
    (repr_tuple (List.map (pair repr_string string_of_int) h.slots))
    (repr_tuple (List.map repr h.written_bufs))

and repr_arg = function
  | No_arg -> "None"
  | Const (`Float x) -> strf "ConstFloat(%s)" (repr_const (`Float x))
  | Const c -> repr_const c
  | Dtype dt -> repr_dtype dt
  | Param p -> repr_param_arg p
  | Range r ->
      repr_tuple
        (List.map string_of_int r.axis_id @ [ Axis_type.repr r.axis_type ])
  | Reduce r ->
      repr_tuple [ Format.asprintf "%a" Op.pp r.op; string_of_int r.num_axes ]
  | Allreduce r ->
      repr_tuple [ Format.asprintf "%a" Op.pp r.op; repr_device r.device ]
  | Device d -> repr_device d
  | Shard i -> string_of_int i
  | Axes l -> repr_tuple (List.map string_of_int l)
  | Flips l -> repr_tuple (List.map repr_bool l)
  | String s -> repr_string s
  | Bytes s -> repr_bytes s
  | Queue q ->
      repr_tuple
        [ repr_tuple (List.map repr_string q.devices); repr_string q.queue ]
  | Region { name; align = 128 } -> repr_string name
  | Region r -> repr_tuple [ repr_string r.name; string_of_int r.align ]
  | Code c -> repr_tuple [ repr_string c.code; repr_dtype c.dtype ]
  | Bufferize b -> repr_bufferize_opts b
  | Kernel k -> repr_kernel_info k
  | Program p -> repr_program_info p
  | Call c -> repr_call_info c
  | Wmma w -> repr_wmma w

let pp ppf u = Format.pp_print_string ppf (repr u)
let pp_arg ppf a = Format.pp_print_string ppf (repr_arg a)
let pp_estimates ppf e = Format.pp_print_string ppf (repr_estimates e)
let pp_kernel_info ppf k = Format.pp_print_string ppf (repr_kernel_info k)
let pp_hcq_info ppf h = Format.pp_print_string ppf (repr_hcq_info h)
let pp_program_info ppf p = Format.pp_print_string ppf (repr_program_info p)

(* Graphs *)

(* Each node is pushed with a flag: unset, its sources are pushed after it; set,
   its sources are done and it is. A node pushed twice before it is done is
   finished at its first pop with the flag set. *)
let toposort ?gate ?(enter_calls = true) root =
  let cache = Tbl.create 64 and order = ref [] in
  let stack = Stack.create () in
  Stack.push (root, false) stack;
  while not (Stack.is_empty stack) do
    let node, visited = Stack.pop stack in
    if not (Tbl.mem cache node) then
      if not visited then
        begin if match gate with None -> true | Some g -> g node then begin
          Stack.push (node, true) stack;
          let srcs =
            if (not enter_calls) && node.op = Op.Call then drop 1 node.src
            else node.src
          in
          List.iter (fun s -> Stack.push (s, false) stack) (List.rev srcs)
        end
        end
      else begin
        Tbl.replace cache node ();
        order := node :: !order
      end
  done;
  List.rev !order

let topovisit root f cache =
  let stack = Stack.create () in
  Stack.push (root, false) stack;
  while not (Stack.is_empty stack) do
    let node, visited = Stack.pop stack in
    if not (Tbl.mem cache node) then
      if not visited then begin
        Stack.push (node, true) stack;
        List.iter (fun s -> Stack.push (s, false) stack) (List.rev node.src)
      end
      else Tbl.replace cache node (f node)
  done;
  Tbl.find cache root

(* A recursive property is filled bottom-up over the nodes that lack it, so a
   deep graph never recurses deeply. *)
let memoized ~get ~set ~compute u =
  match get u with
  | Some x -> x
  | None ->
      List.iter
        (fun n -> set n (compute n))
        (toposort ~gate:(fun n -> Option.is_none (get n)) u);
      Option.get (get u)

let backward_slice u =
  match u.backward_slice_memo with
  | Some s -> s
  | None ->
      let all = toposort ~enter_calls:false u in
      let s = Nodes.of_list (List.filter (fun n -> n != u) all) in
      u.backward_slice_memo <- Some s;
      s

let backward_slice_with_self u =
  Nodes.of_list (u :: Nodes.to_list (backward_slice u))

let op_in_backward_slice_with_self u ops =
  let has n = List.exists (Op.equal n.op) ops in
  has u || List.exists has (Nodes.to_list (backward_slice u))

let bool_slice u =
  Nodes.of_list
    (List.filter (fun n -> Dtype.equal n.dtype Dtype.Bool) (toposort u))

let rec split_uop u sep =
  if Op.equal u.op sep then List.concat_map (fun s -> split_uop s sep) u.src
  else [ u ]

let compare_structure u0 u1 =
  let rec cmp u0 u1 =
    if u0 == u1 then 0
    else
      match Op.compare u0.op u1.op with
      | 0 -> (
          match String.compare (arg_repr u0) (arg_repr u1) with
          | 0 -> (
              match Dtype.compare u0.dtype u1.dtype with
              | 0 -> srcs u0.src u1.src
              | c -> c)
          | c -> c)
      | c -> c
  and srcs l0 l1 =
    match (l0, l1) with
    | [], [] -> 0
    | [], _ -> -1
    | _, [] -> 1
    | s0 :: r0, s1 :: r1 -> ( match cmp s0 s1 with 0 -> srcs r0 r1 | c -> c)
  in
  cmp u0 u1

let key u =
  memoized
    ~get:(fun n -> n.key_memo)
    ~set:(fun n k -> n.key_memo <- Some k)
    ~compute:(fun n ->
      let b = Buffer.create 128 in
      Buffer.add_string b
        (repr_tuple
           [ Format.asprintf "%a" Op.pp n.op; repr_dtype n.dtype; arg_repr n ]);
      List.iter (fun s -> Buffer.add_string b (Option.get s.key_memo)) n.src;
      Digest.BLAKE256.string (Buffer.contents b))
    u

let identity_element op dt : Dtype.const =
  match op with
  | Op.Add -> Dtype.const dt (`Int Bigint.zero)
  | Op.Mul -> Dtype.const dt (`Int Bigint.one)
  | Op.Max -> Dtype.const dt (Dtype.min dt :> Dtype.const)
  | op -> invalid_argf "%s has no identity element" (Op.name op)

(* Numbers *)

module Value = Dtype.Value

(* Division rounding toward zero ([cdiv]) or down ([floordiv]), and its
   remainder: a zero divisor gives the quotient zero, a float one if an operand
   is a float. *)
let divide op ~toward_zero (x : Dtype.value) (y : Dtype.value) : Dtype.value =
  let zero = Value.of_int 0 in
  let abs v = if Value.(v < zero) then Value.(~-v) else v in
  let quotient =
    if Value.(y = zero) then
      match (x, y) with `Float _, _ | _, `Float _ -> `Float 0. | _ -> zero
    else if toward_zero then
      let q = Value.(abs x // abs y) in
      if Value.(x * y < zero) then Value.(~-q) else q
    else Value.(x // y)
  in
  match op with
  | `Quotient -> quotient
  | `Remainder -> Value.(x - (quotient * y))

(* Elementwise operations *)

module type Elementwise = sig
  type t

  val alu : t -> Op.t -> t list -> t
  val cast : t -> Dtype.t -> t
  val add : t -> t -> t
  val mul : t -> t -> t
  val lt : t -> t -> t
  val ne : t -> t -> t
  val maximum : t -> t -> t
  val bitwise_and : t -> t -> t
  val bitwise_or : t -> t -> t
  val bitwise_xor : t -> t -> t
  val shl : t -> t -> t
  val shr : t -> t -> t
  val reciprocal : t -> t
  val trunc : t -> t
  val sqrt : t -> t
  val exp2 : t -> t
  val log2 : t -> t
  val sub : t -> t -> t
  val neg : t -> t
  val div : ?rounding:[ `Trunc | `Floor ] -> t -> t -> t
  val mod_ : t -> t -> t
  val fmod : t -> t -> t
  val floor : t -> t
  val pow : t -> t -> t
  val gt : t -> t -> t
  val le : t -> t -> t
  val ge : t -> t -> t
  val eq : t -> t -> t
  val logical_not : t -> t
  val bitwise_not : t -> t
  val where : t -> t -> t -> t
  val minimum : t -> t -> t

  module O : sig
    val int : int -> t
    val float : float -> t
    val bool : bool -> t
    val ( + ) : t -> t -> t
    val ( - ) : t -> t -> t
    val ( * ) : t -> t -> t
    val ( / ) : t -> t -> t
    val ( // ) : t -> t -> t
    val ( % ) : t -> t -> t
    val ( ~- ) : t -> t
    val ( < ) : t -> t -> t
    val ( > ) : t -> t -> t
    val ( <= ) : t -> t -> t
    val ( >= ) : t -> t -> t
    val ( <> ) : t -> t -> t
    val ( land ) : t -> t -> t
    val ( lor ) : t -> t -> t
    val ( lxor ) : t -> t -> t
    val lnot : t -> t
    val ( lsl ) : t -> t -> t
    val ( lsr ) : t -> t -> t
  end
end

(* What nodes and patterns each provide for the operations they share. *)
module type Elementwise_base = sig
  type t

  val dtype : t -> Dtype.t
  val alu : t -> Op.t -> t list -> t
  val cast : t -> Dtype.t -> t
  val literal : Dtype.const -> t
  (* The literal a constant operand becomes. *)

  val literal_value : t -> Dtype.const option
  (* The value of a literal operand. *)

  val broadcasted : t -> t -> t * t
  (* The operands of a binary operation, promoted. *)

  val direct_floor : bool
  (* Whether [//] and [%] are FLOORDIV and FLOORMOD whatever the operands'
     types. *)
end

module Make_elementwise (B : Elementwise_base) = struct
  open B

  let alu = alu
  let cast = cast

  let binop op x y =
    let a, b = broadcasted x y in
    alu a op [ b ]

  let int n = literal (`Int (Bigint.of_int n))
  let bool b = literal (`Bool b)
  let add x y = binop Op.Add x y
  let mul x y = binop Op.Mul x y
  let ne x y = binop Op.Cmpne x y
  let logical_not x = ne (cast x Dtype.Bool) (bool true)

  let neg x =
    if Dtype.equal (dtype x) Dtype.Bool then logical_not x else mul x (int (-1))

  (* The promoted operands are combined with [alu]: promoting the negation again
     would cast it, since only a weak constant stays weak. *)
  let sub x y =
    let a, b = broadcasted x y in
    alu a Op.Add [ neg b ]

  let reciprocal x = alu x Op.Reciprocal []
  let trunc x = alu x Op.Trunc []
  let sqrt x = alu x Op.Sqrt []
  let exp2 x = alu x Op.Exp2 []
  let log2 x = alu x Op.Log2 []
  let lt x y = binop Op.Cmplt x y
  let gt x y = binop Op.Cmplt y x
  let le x y = logical_not (gt x y)
  let ge x y = logical_not (lt x y)
  let eq x y = logical_not (ne x y)

  let where c x y =
    let x, y = broadcasted x y in
    alu c Op.Where [ x; y ]

  let floor x =
    let b = trunc x in
    where (lt x b) (sub b (int 1)) b

  let div ?rounding x y =
    let a, b = broadcasted x y in
    let ints = Dtype.is_int (dtype a) && Dtype.is_int (dtype b) in
    match rounding with
    | Some `Trunc when ints -> alu a Op.Cdiv [ b ]
    | Some `Floor when ints -> alu a Op.Floordiv [ b ]
    | _ -> (
        let a =
          if Dtype.is_int (dtype a) || Dtype.equal (dtype a) Dtype.Bool then
            cast a (Dtype.default_float ())
          else a
        in
        let d = alu a Op.Mul [ reciprocal b ] in
        match rounding with
        | None -> d
        | Some `Trunc -> trunc d
        | Some `Floor -> floor d)

  let remainder op rounding x y =
    let a, b = broadcasted x y in
    if Dtype.is_int (dtype a) && Dtype.is_int (dtype b) then alu a op [ b ]
    else sub a (mul (div ~rounding a b) b)

  let mod_ x y =
    if direct_floor then binop Op.Floormod x y
    else remainder Op.Floormod `Floor x y

  let fmod x y = remainder Op.Cmod `Trunc x y

  let pow x y =
    let base, exponent = broadcasted x y in
    let non_negative_int =
      match literal_value y with
      | Some (`Int z) -> Bigint.geq z Bigint.zero
      | Some (`Bool _) -> true
      | Some _ -> false
      | None -> true
    in
    if
      (not (Dtype.is_float (Dtype.least_upper [ dtype base; dtype exponent ])))
      && not non_negative_int
    then
      invalid_arg
        "the base of a power by a negative or non-integer constant must be a \
         float";
    alu base Op.Pow [ exponent ]

  let bitwise_and x y = binop Op.And x y
  let bitwise_or x y = binop Op.Or x y
  let bitwise_xor x y = binop Op.Xor x y
  let shl x y = binop Op.Shl x y
  let shr x y = binop Op.Shr x y

  let bitwise_not x =
    let dt = dtype x in
    if Dtype.equal dt Dtype.Bool then logical_not x
    else if Dtype.is_unsigned dt then
      bitwise_xor x (literal (Dtype.max dt :> Dtype.const))
    else bitwise_xor x (int (-1))

  let maximum x y = binop Op.Max x y

  (* An integer minimum maps each operand through an order-reversing xor, takes
     the maximum, and maps back. *)
  let minimum x y =
    let t, x = broadcasted x y in
    let dt = Dtype.least_upper [ dtype t; dtype x ] in
    if Dtype.is_float dt then neg (alu (neg t) Op.Max [ neg x ])
    else
      let k =
        literal
          (Dtype.const dt
             (Value.( + ) (Dtype.min dt) (Dtype.max dt) :> Dtype.const))
      in
      bitwise_xor (alu (bitwise_xor t k) Op.Max [ bitwise_xor x k ]) k

  module O = struct
    let int = int
    let float x = literal (`Float x)
    let bool = bool
    let ( + ) = add
    let ( - ) = sub
    let ( * ) = mul
    let ( / ) x y = div x y

    let ( // ) x y =
      if direct_floor then binop Op.Floordiv x y else div ~rounding:`Floor x y

    let ( % ) = mod_
    let ( ~- ) = neg
    let ( < ) = lt
    let ( > ) = gt
    let ( <= ) = le
    let ( >= ) = ge
    let ( <> ) = ne
    let ( land ) = bitwise_and
    let ( lor ) = bitwise_or
    let ( lxor ) = bitwise_xor
    let lnot = bitwise_not
    let ( lsl ) = shl
    let ( lsr ) = shr
  end
end

(* Constants and casts *)

let cast x dt =
  if Dtype.equal x.dtype dt then x else v Op.Cast ~src:[ x ] ~arg:(Dtype dt)

let const ?dtype (c : Dtype.const) =
  let dt =
    match (dtype, c) with
    | None, _ | _, `Invalid -> Dtype.of_const c
    | Some dt, _ -> dt
  in
  (* The cast folds away at the types a literal has. *)
  cast (v Op.Const ~arg:(Const (Dtype.const dt c))) dt

let value u : Dtype.const =
  match (u.op, u.arg, u.src) with
  | Op.Const, Const c, _ -> c
  | Op.Cast, _, [ { op = Op.Const; arg = Const c; _ } ] -> c
  | op, _, _ -> invalid_argf "%s is not a constant" (Op.name op)

let is_invalid u = is_const_invalid u

let ccast u dt =
  if u.op = Op.Const then const ~dtype:dt (value u) else cast u dt

let rec remint u dt =
  if u.op = Op.Const then ccast u dt
  else
    match u.src with
    | s :: rest -> replace u ~src:(remint s dt :: rest)
    | [] -> ccast u dt

let broadcasted x y =
  let out = Dtype.least_upper [ x.dtype; y.dtype ] in
  let promote t =
    let b = base t in
    if is_invalid b then t
    else if List.mem t.dtype Dtype.weaks && b.op = Op.Const then
      let dt = Dtype.weak out in
      if Dtype.equal t.dtype dt then t else remint t dt
    else cast t out
  in
  (promote x, promote y)

include Make_elementwise (struct
  type nonrec t = t

  let dtype u = u.dtype
  let alu x op rest = v op ~src:(x :: rest)
  let cast = cast
  let literal c = const c

  let literal_value u =
    match (u.op, u.arg) with Op.Const, Const c -> Some c | _ -> None

  let broadcasted = broadcasted
  let direct_floor = false
end)

let int ?dtype n = const ?dtype (`Int (Bigint.of_int n))
let float ?dtype x = const ?dtype (`Float x)
let bool ?dtype b = const ?dtype (`Bool b)
let invalid = const `Invalid

(* Bounds *)

let mselect u i = v Op.Mselect ~src:[ u ] ~arg:(Shard i)

let rec buf_uop u =
  match u.op with
  | Op.Buffer | Op.Alloc | Op.Param -> u
  | Op.Mselect -> (
      match u.arg with
      | Shard i -> mselect (buf_uop (first u.op u.src)) i
      | _ -> invalid_arg "a shard selection needs a shard")
  | Op.Mstack -> v Op.Mstack ~src:(List.map buf_uop u.src)
  | _ ->
      let b = base u in
      if b.op = Op.After then base (buf_uop (first b.op b.src))
      else
        let rec down s =
          match s.src with
          | first :: _
            when not (List.mem s.op Op.[ Buffer; Alloc; Param; Stage; Mstack ])
            ->
              down first
          | _ -> s
        in
        down u

(* The elements of a constant table read as [dt]. *)
let table_values dt bytes : Dtype.value list =
  let fmt =
    match Dtype.fmt dt with
    | Some f -> f
    | None -> invalid_argf "%s values cannot be read from bytes" (repr_dtype dt)
  in
  let size = Dtype.itemsize dt in
  if String.length bytes mod size <> 0 then
    invalid_argf "%d bytes do not hold %s values" (String.length bytes)
      (repr_dtype dt);
  List.init
    (String.length bytes / size)
    (fun i ->
      let at = i * size in
      match fmt with
      | '?' -> `Bool (bytes.[at] <> '\000')
      | 'b' -> `Int (Bigint.of_int (String.get_int8 bytes at))
      | 'B' -> `Int (Bigint.of_int (String.get_uint8 bytes at))
      | 'h' -> `Int (Bigint.of_int (String.get_int16_le bytes at))
      | 'H' -> `Int (Bigint.of_int (String.get_uint16_le bytes at))
      | 'i' -> `Int (Bigint.of_int32 (String.get_int32_le bytes at))
      | 'I' -> `Int (Bigint.of_int32_unsigned (String.get_int32_le bytes at))
      | 'q' -> `Int (Bigint.of_int64 (String.get_int64_le bytes at))
      | 'Q' -> `Int (Bigint.of_int64_unsigned (String.get_int64_le bytes at))
      | 'e' ->
          Dtype.bitcast Dtype.Uint16 Dtype.Float16
            (`Int (Bigint.of_int (String.get_uint16_le bytes at)))
      | 'f' -> `Float (Int32.float_of_bits (String.get_int32_le bytes at))
      | 'd' -> `Float (Int64.float_of_bits (String.get_int64_le bytes at))
      | c -> invalid_argf "unknown format %C" c)

(* The least and greatest of values, as Python's [min] and [max] of them. *)
let extremes = function
  | x :: rest ->
      (List.fold_left Value.min x rest, List.fold_left Value.max x rest)
  | [] -> invalid_arg "no values to bound"

(* The types a cast's bounds are clamped to. *)
let clamped_types = Dtype.sints @ Dtype.weaks

(* The bounds of a value of [dt] that nothing proves more of. Finite float
   bounds state that the value is not NaN, and every float type holds NaN, so a
   float's are infinite even where its type has no infinities. *)
let unbounded dt : Dtype.value * Dtype.value =
  if Dtype.is_float dt then (`Float Float.neg_infinity, `Float Float.infinity)
  else (Dtype.min dt, Dtype.max dt)

(* A committed integer wraps at its width: bounds that leave its type are one
   value wrapped, or the type's. *)
let at_width dt ((lo, hi) as b : Dtype.value * Dtype.value) =
  match (lo, hi) with
  | `Int _, `Int _
    when List.mem dt Dtype.ints
         && Value.(lo < Dtype.min dt || Dtype.max dt < hi) ->
      if Value.(lo = hi) then
        let v = Dtype.truncate dt lo in
        (v, v)
      else (Dtype.min dt, Dtype.max dt)
  | _ -> b

(* Bounds read only the sources their rule needs, so they recurse rather than
   fill the whole graph below. A node's bounds, and those of an operand an
   operation commits to its type, are at their width. *)
let rec min_max u =
  match u.min_max_memo with
  | Some b -> b
  | None ->
      let b = at_width u.dtype (compute_min_max u) in
      u.min_max_memo <- Some b;
      b

and operand_bounds u s =
  let operands =
    if Op.Set.mem u.op Op.Set.comparison then promo_dtype u.src else u.dtype
  in
  at_width operands (min_max s)

and compute_min_max u : Dtype.value * Dtype.value =
  let dt = u.dtype in
  let bounds x = operand_bounds u x in
  let binary =
    if Op.Set.mem u.op Op.Set.binary then
      match u.src with
      | [ x; y ] when Dtype.is_float dt ->
          float_bounds u.op dt (bounds x) (bounds y)
      | [ x; y ] -> binary_bounds u (bounds x) (bounds y)
      | _ -> None
    else None
  in
  match binary with
  | Some b -> b
  | None -> (
      let src0 () = first u.op u.src in
      match (u.op, u.arg) with
      | Op.Where, _ -> (
          match u.src with
          | [ c; x; y ] ->
              let (x0, x1), (y0, y1) = (selected c (bounds x) x, bounds y) in
              (Value.min x0 y0, Value.max x1 y1)
          | _ -> invalid_arg "where needs three sources")
      | (Op.Param | Op.Buffer | Op.Alloc), Param { vmin_vmax = Some b; _ } -> b
      | (Op.Range | Op.Special), _ when not (Dtype.equal dt Dtype.Void) ->
          (`Int Bigint.zero, snd (min_max (sub (src0 ()) (int 1))))
      | Op.Stack, _ when not (List.is_empty u.src) ->
          let bs = List.map bounds u.src in
          (fst (extremes (List.map fst bs)), snd (extremes (List.map snd bs)))
      | Op.Load, _ when (buf_uop (src0 ())).op = Op.Binary -> (
          match (buf_uop (src0 ())).arg with
          | Bytes b ->
              let values = table_values dt b in
              let is_nan = function `Float x -> Float.is_nan x | _ -> false in
              if List.exists is_nan values then unbounded dt
              else extremes values
          | _ -> invalid_arg "a constant table needs bytes")
      | Op.Const, Const ((`Bool _ | `Int _) as c) -> (c, c)
      | Op.Const, Const (`Float x) when not (Float.is_nan x) ->
          (`Float x, `Float x)
      | Op.Pad, _ ->
          let lo, hi = bounds (src0 ()) in
          (Value.min lo (`Int Bigint.zero), Value.max hi (`Int Bigint.zero))
      | op, _
        when Op.Set.mem op Op.Set.movement
             || List.mem op
                  Op.[ Index; Stage; After; Detach; Copy; Contiguous_backward ]
        ->
          bounds (src0 ())
      | Op.Trunc, _ ->
          let trunc = function `Float x -> `Float (Float.trunc x) | v -> v in
          let lo, hi = bounds (src0 ()) in
          (trunc lo, trunc hi)
      | Op.Neg, _ when Dtype.is_float dt ->
          let lo, hi = bounds (src0 ()) in
          (Value.( ~- ) hi, Value.( ~- ) lo)
      | Op.Cast, _ -> (
          let x = src0 () in
          match cast_bounds x.dtype dt (min_max x) with
          | Some b -> b
          | None -> unbounded dt)
      | _ -> unbounded dt)

(* Where a float comparison [a < b] holds, neither operand is NaN, [a] is below
   [b]'s greatest value and [b] above [a]'s least: [selected c (lo, hi) t] is
   [t]'s bounds [(lo, hi)] narrowed so, where [c] selects [t]. *)
and selected c (lo, hi) t =
  match (c.op, c.src) with
  | Op.Cmplt, [ a; b ] when Dtype.is_float a.dtype ->
      if t == a then (lo, Value.min hi (snd (min_max b)))
      else if t == b then (Value.max lo (fst (min_max a)), hi)
      else (lo, hi)
  | _ -> (lo, hi)

(* A float sum, difference or product of operands with finite bounds has bounds:
   the corners, widened by more than the result's rounding, a relative 2^-m and
   the smallest normal of its type. A target that flushes subnormals to zero
   may flush an operand as well as the result: an operand's end within the
   subnormals counts as 0, and the widening covers the result. Finite operands
   make a finite result, never NaN, which no bounds hold; a result that may
   overflow its type has none. *)
and float_bounds op dt (s0_min, s0_max) (s1_min, s1_max) =
  let finite : Dtype.value -> float option = function
    | `Float x when Float.is_finite x -> Some x
    | `Int z -> Some (Bigint.to_float z)
    | _ -> None
  in
  match
    ( op,
      List.mem dt Dtype.weaks,
      List.map finite [ s0_min; s0_max; s1_min; s1_max ] )
  with
  | (Op.Add | Op.Sub | Op.Mul), false, [ Some a; Some b; Some c; Some d ] ->
      let e, m = Dtype.finfo dt in
      let rel = Float.ldexp 1. (-m)
      and tiny = Float.ldexp 1. (2 - (1 lsl (e - 1))) in
      let low x = if 0. < x && x < tiny then 0. else x
      and high x = if -.tiny < x && x < 0. then 0. else x in
      let a = low a and b = high b and c = low c and d = high d in
      let lo, hi =
        match op with
        | Op.Add -> (a +. c, b +. d)
        | Op.Sub -> (a -. d, b -. c)
        | _ ->
            let corners = [ a *. c; a *. d; b *. c; b *. d ] in
            ( List.fold_left Float.min Float.infinity corners,
              List.fold_left Float.max Float.neg_infinity corners )
      in
      let lo = lo -. (Float.abs lo *. rel) -. tiny
      and hi = hi +. (Float.abs hi *. rel) +. tiny in
      let fits x =
        match Dtype.truncate dt (`Float x) with
        | `Float y -> Float.is_finite y
        | _ -> false
      in
      if fits lo && fits hi then Some (`Float lo, `Float hi) else None
  | _ -> None

(* Rounding is monotone, so a cast maps bounds to bounds: toward zero into an
   integer, to nearest into a float. An integer keeps its value in an integer
   type, where [min_max] wraps it. A float target holds the rounded bounds only
   where they are its values: a value past them is NaN in a type without
   infinities, and a NaN bound is no value. A signed target holds the part of
   the source range that overlaps it; a float overflowing an integer is
   undefined. *)
and cast_bounds src dt (lo, hi) =
  let round (x : Dtype.value) : Dtype.value =
    match x with
    | `Float f when not (Float.is_finite f) -> x
    | _ ->
        if Dtype.is_float dt && not (List.mem dt Dtype.weaks) then
          Dtype.truncate dt x
        else if Dtype.is_int dt then `Int (Value.to_z x)
        else x
  in
  let lo, hi = (round lo, round hi) in
  if Dtype.is_int dt && not (Dtype.is_float src) then Some (lo, hi)
  else if
    Dtype.is_unsigned dt
    && Value.( <= ) (`Int Bigint.zero) lo
    && Value.( <= ) hi (Dtype.max dt)
  then Some (lo, hi)
  else if Dtype.is_float dt then
    if Value.(Dtype.min dt <= lo && hi <= Dtype.max dt) then Some (lo, hi)
    else None
  else if
    List.mem dt clamped_types
    && Value.( <= ) lo (Dtype.max dt)
    && Value.( <= ) (Dtype.min dt) hi
  then Some (Value.max (Dtype.min dt) lo, Value.min hi (Dtype.max dt))
  else None

and binary_bounds u (s0_min, s0_max) (s1_min, s1_max) =
  let add = Value.( + ) and sub = Value.( - ) and mul = Value.( * ) in
  let neg = Value.( ~- ) and lt = Value.( < ) and le = Value.( <= ) in
  let equal = Value.( = ) in
  let z = Value.to_z and zero = `Int Bigint.zero in
  let ints = List.for_all (function `Float _ -> false | _ -> true) in
  let c1 = equal s1_min s1_max in
  let corners f =
    Some
      (extremes
         [ f s0_min s1_min; f s0_min s1_max; f s0_max s1_min; f s0_max s1_max ])
  in
  match u.op with
  | Op.Add -> Some (add s0_min s1_min, add s0_max s1_max)
  | Op.Sub -> Some (sub s0_min s1_max, sub s0_max s1_min)
  | Op.And when Dtype.is_int u.dtype && c1 && le zero s1_max ->
      if lt s0_min zero then Some (zero, s1_max)
      else
        let mask = Bigint.pred (Bigint.shift_left Bigint.one (Bigint.numbits (z s0_max))) in
        Some (zero, Value.min s0_max (`Int (Bigint.logand (z s1_max) mask)))
  | Op.Mul -> corners mul
  (* A shift by a negative count is undefined: its bounds are the type's. *)
  | Op.Shl when c1 && ints [ s0_min; s0_max; s1_min ] && le zero s1_min ->
      let k = Bigint.to_int (z s1_min) in
      Some (`Int (Bigint.shift_left (z s0_min) k), `Int (Bigint.shift_left (z s0_max) k))
  | Op.Shr when c1 && ints [ s0_min; s0_max; s1_min ] && le zero s1_min ->
      let k = Bigint.to_int (z s1_min) in
      Some (`Int (Bigint.shift_right (z s0_min) k), `Int (Bigint.shift_right (z s0_max) k))
  | Op.Cmod when c1 && lt zero s1_max ->
      let c = s1_min in
      Some
        ( (if lt zero s0_min then zero
           else if lt (neg c) s0_min then s0_min
           else neg (sub s1_max (`Int Bigint.one))),
          if lt s0_max zero then zero
          else if lt s0_max c then s0_max
          else sub c (`Int Bigint.one) )
  | Op.Cmod when lt zero s1_min ->
      let m = sub s1_max (`Int Bigint.one) in
      Some
        (if le zero s0_min then (zero, m)
         else if le s0_max zero then (neg m, zero)
         else (neg m, m))
  | Op.Cmod when lt s1_max zero ->
      let m = sub (neg s1_min) (`Int Bigint.one) in
      Some
        (if le zero s0_min then (zero, m)
         else if le s0_max zero then (neg m, zero)
         else (neg m, m))
  | Op.Cdiv when Bigint.gt (Bigint.mul (z s1_min) (z s1_max)) Bigint.zero ->
      let d a b = divide `Quotient ~toward_zero:true a b in
      corners d
  | (Op.Floordiv | Op.Floormod) when lt s0_max s0_min -> Some (zero, zero)
  | Op.Floordiv when Bigint.gt (Bigint.mul (z s1_min) (z s1_max)) Bigint.zero ->
      let d a b = Value.(a // b) in
      corners d
  | Op.Floormod when c1 && not (Bigint.equal (z s1_min) Bigint.zero) ->
      let c = s1_min in
      if Value.(s0_min // c = s0_max // c) then
        Some (Value.(s0_min % c), Value.(s0_max % c))
      else if lt zero c then Some (zero, sub c (`Int Bigint.one))
      else Some (add c (`Int Bigint.one), zero)
  | Op.Floormod when lt zero s1_min -> Some (zero, sub s1_max (`Int Bigint.one))
  | Op.Floormod when lt s1_max zero -> Some (add s1_min (`Int Bigint.one), zero)
  | Op.Xor
    when equal s1_min (`Int Bigint.minus_one)
         && equal s1_max (`Int Bigint.minus_one)
         && ints [ s0_min; s0_max ] ->
      Some (`Int (Bigint.lognot (z s0_max)), `Int (Bigint.lognot (z s0_min)))
  | Op.Max -> Some (Value.max s0_min s1_min, Value.max s0_max s1_max)
  | Op.Cmplt -> Some (`Bool (lt s0_max s1_min), `Bool (lt s0_min s1_max))
  | Op.Cmpne ->
      Some
        ( `Bool (lt s0_max s1_min || lt s1_max s0_min),
          `Bool
            (not
               (equal s0_min s0_max && equal s0_max s1_min
              && equal s1_min s1_max)) )
  | Op.Or when Dtype.equal u.dtype Dtype.Bool ->
      Some
        ( (if Value.to_bool s0_min then s0_min else s1_min),
          if Value.to_bool s0_max then s0_max else s1_max )
  | Op.And when Dtype.equal u.dtype Dtype.Bool ->
      Some
        ( (if Value.to_bool s0_min then s1_min else s0_min),
          if Value.to_bool s0_max then s1_max else s0_max )
  | _ -> None

let vmin u = fst (min_max u)
let vmax u = snd (min_max u)

let overflows u dt =
  Value.( < ) (vmin u) (Dtype.min dt) || Value.( < ) (Dtype.max dt) (vmax u)

let exact dt vs =
  (not (List.mem dt Dtype.ints))
  || List.for_all (fun v -> Value.(Dtype.min dt <= v && v <= Dtype.max dt)) vs

(* Simplification *)

(* The symbolic rewrite, installed by [Symbolic] when the library is
   initialised. *)
let simplify_hook : (t -> t) option Atomic.t = Atomic.make None

(* The symbolic rules leave a graph of constants as it is: a sink of constants
   and of stacks of constants is itself, which lets shapes be built before the
   rules are installed. *)
let simplify u =
  let constant s =
    s.op = Op.Const
    || (s.op = Op.Stack && List.for_all (fun c -> c.op = Op.Const) s.src)
  in
  if u.op = Op.Const then u
  else if u.op = Op.Sink && List.for_all constant u.src then u
  else
    match Atomic.get simplify_hook with
    | Some rewrite -> rewrite u
    | None -> invalid_arg "the symbolic rules are not installed"

let resolve ?(default = true) u =
  if not (Dtype.equal u.dtype Dtype.Bool) then
    invalid_argf "only a boolean resolves, not a %s" (repr_dtype u.dtype);
  let lo, hi = min_max (simplify u) in
  if Value.( = ) lo hi then Value.to_bool lo else default

let sint_of_const (c : Dtype.const) =
  match c with
  | `Int z when Bigint.fits_int z -> Some (Int (Bigint.to_int z) : sint)
  | `Bool b -> Some (Int (Bool.to_int b))
  | _ -> None

let ssimplify u : sint =
  let r = simplify u in
  let known =
    match (r.op, r.src) with
    | Op.Cast, [ ({ op = Op.Const; _ } as c) ] ->
        sint_of_const (Dtype.const r.dtype (value c))
    | Op.Const, _ -> sint_of_const (value r)
    | _ -> None
  in
  match known with Some s -> s | None -> Sym r

let ssimplify_sint = function Int n -> Int n | Sym u -> ssimplify u

let eval u ~kinds ~what =
  if not (List.exists (Dtype.equal u.dtype) kinds) then
    invalid_argf "a %s is not %s" (repr_dtype u.dtype) what;
  let s = simplify u in
  let lo, hi = min_max s in
  if not (Value.( = ) lo hi) then
    invalid_argf "the value ranges from %s to %s" (repr_const lo)
      (repr_const hi);
  lo

let to_bool u =
  match eval u ~kinds:[ Dtype.Bool ] ~what:"a boolean" with
  | `Bool b -> b
  | v -> invalid_argf "%s is not a boolean" (repr_const v)

let to_z u =
  match eval u ~kinds:(Dtype.Weak_int :: Dtype.ints) ~what:"an integer" with
  | (`Int _ | `Bool _) as v -> Value.to_z v
  | v -> invalid_argf "%s is not an integer" (repr_const v)

let to_float u =
  match eval u ~kinds:(Dtype.Weak_float :: Dtype.floats) ~what:"a float" with
  | `Float x -> x
  | v -> invalid_argf "%s is not a float" (repr_const v)

(* Symbolic integers *)

module Sint = struct
  type node = t
  type t = sint

  let node : t -> node = function Int n -> int n | Sym u -> u

  (* Integers compute exactly, as Python's do, and a result past [int] raises
     rather than wraps. *)
  let of_z z =
    if Bigint.fits_int z then Int (Bigint.to_int z)
    else invalid_argf "%s is larger than an int" (Bigint.to_string z)

  let arith fz fu a b =
    match (a, b) with
    | Int x, Int y -> of_z (fz (Bigint.of_int x) (Bigint.of_int y))
    | _ -> Sym (fu (node a) (node b))

  let ( + ) = arith Bigint.add add
  let ( - ) = arith Bigint.sub sub
  let ( * ) = arith Bigint.mul mul

  let ( // ) =
    arith
      (fun x y -> Value.to_z Value.(`Int x // `Int y))
      (div ~rounding:`Floor)

  let ( % ) = arith (fun x y -> Value.to_z Value.(`Int x % `Int y)) mod_
  let neg = function Int n -> Int 0 - Int n | Sym u -> Sym (neg u)

  (* A product of integers is taken whole, so that a zero makes it zero past
     partial products that do not fit. *)
  let prod l =
    let ints = List.filter_map (function Int n -> Some n | Sym _ -> None) l in
    if List.length ints = List.length l then
      of_z (List.fold_left (fun z n -> Bigint.mul z (Bigint.of_int n)) Bigint.one ints)
    else List.fold_left ( * ) (Int 1) l

  type cond = Known of bool | Cond of node

  let compare fi fu a b =
    match (a, b) with
    | Int x, Int y -> Known (fi x y)
    | _ -> Cond (fu (node a) (node b))

  let ( < ) = compare Stdlib.( < ) lt
  let ( <= ) = compare Stdlib.( <= ) le
  let ( > ) = compare Stdlib.( > ) gt
  let ( >= ) = compare Stdlib.( >= ) ge
  let ( <> ) = compare Stdlib.( <> ) ne
  let resolve ?default = function Known b -> b | Cond u -> resolve ?default u

  (* The truth of a condition that must be decidable. *)
  let truth = function Known b -> b | Cond u -> to_bool u
  let equal = equal_sint
  let pp ppf s = Format.pp_print_string ppf (repr_sint s)
end

(* The symbolic operands, then the extreme of the integer ones. *)
let smax_smin ~name ~combine ~pick ss =
  let syms = List.filter_map (function Sym u -> Some u | Int _ -> None) ss
  and ints = List.filter_map (function Int n -> Some n | Sym _ -> None) ss in
  let extreme = function
    | [] -> []
    | n :: rest -> [ List.fold_left pick n rest ]
  in
  match (syms, extreme ints) with
  | [], [] -> invalid_argf "%s of nothing" name
  | [], n -> Int (List.hd n)
  | u :: rest, n -> ssimplify (List.fold_left combine u (rest @ List.map int n))

let smax ss = smax_smin ~name:"smax" ~combine:maximum ~pick:Stdlib.max ss
let smin ss = smax_smin ~name:"smin" ~combine:minimum ~pick:Stdlib.min ss

let sint_to_uop ?(dtype = Dtype.Weak_int) = function
  | Int n -> int ~dtype n
  | Sym u -> cast u dtype

(* The number of elements of a shape of known sizes. *)
let size_of shape =
  if List.mem 0 shape then 0
  else
    List.fold_left
      (fun acc n ->
        if n <> 0 && abs acc > max_int / abs n then
          invalid_argf "a shape of %s elements is larger than an int"
            (Bigint.to_string
               (List.fold_left (fun z n -> Bigint.mul z (Bigint.of_int n)) Bigint.one shape))
        else acc * n)
      1 shape

let to_max_shape shape =
  List.map (function Int n -> n | Sym u -> Value.to_int (vmax u)) shape

(* Shapes *)

let as_shape u : sint list =
  let known s =
    match sint_of_const (value s) with
    | Some n -> n
    | None -> invalid_argf "%s is not a size" (repr_const (value s))
  in
  match u.op with
  | Op.Const -> [ known u ]
  | Op.Stack ->
      List.map (fun s -> if s.op = Op.Const then known s else ssimplify s) u.src
  | _ -> [ ssimplify u ]

let marg u =
  match u.marg_memo with
  | Some m -> m
  | None ->
      let shape_src i = as_shape (nth u i) in
      let m =
        match (u.op, u.arg) with
        | Op.Reshape, _ -> Reshape (shape_src 1)
        | Op.Expand, _ -> Expand (shape_src 1)
        | Op.Pad, _ -> Pad (List.combine (shape_src 1) (shape_src 2))
        | Op.Shrink, _ -> Shrink (List.combine (shape_src 1) (shape_src 2))
        | Op.Permute, Axes l -> Permute l
        | Op.Flip, Flips l -> Flip l
        | op, _ -> invalid_argf "%s is not a movement" (Op.name op)
      in
      u.marg_memo <- Some m;
      m

let marg_shape u =
  match marg u with
  | Reshape s | Expand s -> s
  | _ -> invalid_argf "%s has no shape argument" (Op.name u.op)

let marg_bounds u =
  match marg u with
  | Pad b | Shrink b -> b
  | _ -> invalid_argf "%s has no bounds argument" (Op.name u.op)

let align_left shapes =
  let n = List.fold_left (fun m s -> max m (List.length s)) 0 shapes in
  List.map (fun s -> List.init (n - List.length s) (fun _ -> Int 1) @ s) shapes

let rec transpose = function
  | [] | [] :: _ -> []
  | rows -> List.map List.hd rows :: transpose (List.map List.tl rows)

let equal_shape = List.equal equal_sint
let repr_shape s = repr_tuple (List.map repr_sint s)

let broadcast_shape shapes =
  match shapes with
  | [] -> invalid_arg "no shapes to broadcast"
  | s :: rest when List.for_all (equal_shape s) rest -> s
  | _ ->
      List.map
        (fun sizes ->
          let rest =
            List.fold_left
              (fun acc s ->
                match s with
                | Int 1 -> acc
                | s when List.exists (equal_sint s) acc -> acc
                | s -> acc @ [ s ])
              [] sizes
          in
          match rest with
          | [] -> Int 1
          | [ s ] -> s
          | _ ->
              invalid_argf "shapes %s cannot be broadcast to one shape"
                (String.concat ", " (List.map repr_shape shapes)))
        (transpose (align_left shapes))

let broadcast_axes src out =
  let nleft = List.length out - List.length src in
  if nleft < 0 then
    invalid_argf "cannot broadcast %s into %s" (repr_shape src) (repr_shape out);
  List.init nleft Fun.id
  @ List.filter_map Fun.id
      (List.mapi
         (fun i s ->
           let one = match s with Int 1 -> true | _ -> false in
           let o = List.nth out (nleft + i) in
           if one && Sint.resolve Sint.(o <> Int 1) then Some (nleft + i)
           else None)
         src)

let rec shape_opt u =
  memoized
    ~get:(fun n -> n.shape_memo)
    ~set:(fun n s -> n.shape_memo <- Some s)
    ~compute:compute_shape u

and shape u =
  match shape_opt u with
  | Some s -> s
  | None -> invalid_argf "%s has no shape" (Op.name u.op)

and compute_shape u : sint list option =
  let src0 () = first u.op u.src in
  let void = Dtype.equal u.dtype Dtype.Void in
  match u.op with
  | Op.If | Op.Barrier | Op.Sink | Op.Endif | Op.Backedge | Op.Group | Op.Linear
  | Op.Program | Op.Source | Op.Custom_function ->
      None
  | Op.Call | Op.Ins -> if void then None else Some []
  | Op.Reshape when (src0 ()).op = Op.Noop -> Some (marg_shape u)
  | Op.Noop -> ( match u.src with s :: _ -> shape_opt s | [] -> None)
  | Op.Index ->
      let buf = src0 () and idxs = drop 1 u.src in
      Some (List.concat_map shape idxs @ drop (List.length idxs) (shape buf))
  | Op.Stack -> (
      match u.src with
      | [] -> Some []
      | s :: _ -> Some (Int (List.length u.src) :: shape s))
  | Op.Const | Op.Getaddr | Op.Range | Op.Special -> Some []
  | Op.Binary -> (
      match u.arg with
      | Bytes b -> Some [ Int (String.length b) ]
      | _ -> invalid_arg "binary needs bytes")
  | Op.Buffer | Op.Alloc | Op.Param -> (
      match u.arg with
      | Param { size = None; _ } -> Some []
      | Param { size = Some n; _ } -> Some [ Int n ]
      | _ -> invalid_arg "storage needs a ParamArg")
  | Op.Custom | Op.Customi -> (
      if void then None
      else
        match List.filter_map shape_opt u.src with
        | [] -> None
        | shapes -> Some (broadcast_shape shapes))
  | Op.Stage ->
      let rs = drop 1 u.src in
      Some
        (List.map
           (fun r -> Int (Value.to_int (Value.( + ) (vmax r) (`Int Bigint.one))))
           rs
        @ shape (src0 ()))
  | Op.Wmma -> (
      match u.src with
      | [ a; b; acc ] ->
          let init s = take (List.length s - 1) s in
          let last s = List.nth s (List.length s - 1) in
          Some
            (broadcast_shape
               [ init (shape a); init (shape b); init (shape acc) ]
            @ [ last (shape acc) ])
      | _ -> invalid_arg "wmma needs three sources")
  | Op.Mstack | Op.Mselect | Op.Detach | Op.Contiguous_backward | Op.After
  | Op.Load | Op.Copy | Op.Allreduce | Op.Store | Op.End ->
      shape_opt (src0 ())
  | Op.Bitcast -> (
      match shape_opt (src0 ()) with
      | None -> None
      | Some [] -> Some []
      | Some ps ->
          let out_sz = Dtype.itemsize u.dtype
          and in_sz = Dtype.itemsize (src0 ()).dtype in
          if out_sz = in_sz then Some ps
          else
            let n = List.length ps in
            let last = List.nth ps (n - 1) in
            (match last with
            | Int l when l * in_sz mod out_sz <> 0 ->
                invalid_argf "a bitcast cannot resize an axis of %d" l
            | _ -> ());
            Some
              (take (n - 1) ps
              @ [ ssimplify_sint Sint.(last * Int in_sz // Int out_sz) ]))
  | Op.Unshard when List.is_empty u.src -> None
  | op when Op.Set.mem op Op.Set.movement || op = Op.Unshard || op = Op.Reduce
    ->
      let ps =
        match shape_opt (src0 ()) with
        | Some ps -> ps
        | None ->
            invalid_argf "%s needs a shape, and %s has none" (Op.name op)
              (Op.name (src0 ()).op)
      in
      Some (movement_shape u ps)
  | op when Op.Set.mem op Op.Set.unary || op = Op.Cast ->
      (match u.src with
      | [ _ ] -> ()
      | _ -> invalid_argf "%s needs one source" (Op.name op));
      shape_opt (src0 ())
  | op when Op.Set.mem op Op.Set.broadcastable ->
      let shapes =
        List.map
          (fun s ->
            match shape_opt s with
            | Some sh -> sh
            | None -> invalid_argf "%s of a node without a shape" (Op.name op))
          u.src
      in
      if List.is_empty shapes then invalid_argf "%s needs sources" (Op.name op);
      if
        Helpers.Context_var.value Helpers.disallow_broadcast
        && not (Helpers.all_same equal_shape shapes)
      then
        invalid_argf "%s of shapes %s" (Op.name op)
          (String.concat ", " (List.map repr_shape shapes));
      Some (broadcast_shape shapes)
  | op -> invalid_argf "%s has no shape rule" (Op.name op)

and movement_shape u ps =
  let bad what =
    invalid_argf "invalid %s %s for %s" (Op.name u.op) what (repr_shape ps)
  in
  let ok = Sint.resolve ?default:None in
  (* A size is a number: one that holds Invalid, such as the size of a shrink
     whose bounds carry a validity, evaluates to none. *)
  let numbers s =
    List.iter
      (function
        | Sym d when List.exists is_const_invalid (toposort d) ->
            invalid_argf "%s of sizes %s, one holding Invalid, which is no number"
              (Op.name u.op) (repr_shape s)
        | _ -> ())
      s;
    s
  in
  match u.op with
  | Op.Unshard -> (
      match u.arg with
      | Axes axes ->
          let ranges = drop 1 u.src in
          List.mapi
            (fun a s ->
              match List.find_index (Int.equal a) axes with
              | Some i ->
                  Sint.(
                    s
                    * Int
                        (Value.to_int
                           (Value.( + ) (vmax (List.nth ranges i)) (`Int Bigint.one))))
              | None -> s)
            ps
      | _ -> invalid_arg "an unshard needs its axes")
  | Op.Reduce -> (
      match u.arg with
      | Reduce { num_axes; _ } when num_axes >= 0 && num_axes <= List.length ps
        ->
          drop num_axes ps
      | _ -> invalid_argf "invalid reduction axes for %s" (repr_shape ps))
  | _ -> (
      match marg u with
      | Reshape s ->
          if not (List.for_all (fun x -> Sint.truth Sint.(x >= Int 0)) s) then
            invalid_argf "a shape cannot hold negative sizes: %s" (repr_shape s);
          if Sint.resolve ~default:false Sint.(prod ps <> prod s) then
            invalid_argf "cannot reshape %s to %s" (repr_shape ps)
              (repr_shape s);
          numbers s
      | Expand s -> numbers s @ ps
      | Permute order ->
          if List.sort Int.compare order <> List.init (List.length ps) Fun.id
          then bad (repr_tuple (List.map string_of_int order));
          List.map (List.nth ps) order
      | Pad bounds ->
          if
            List.length ps <> List.length bounds
            || not
                 (List.for_all2
                    (fun s (o, sz) ->
                      ok Sint.(sz >= Int 0)
                      && ok Sint.(o >= Int 0)
                      && ok Sint.(o + s <= sz))
                    ps bounds)
          then bad "padding";
          numbers (List.map (fun (_, sz) -> ssimplify_sint sz) bounds)
      | Shrink bounds ->
          if
            List.length ps <> List.length bounds
            || not
                 (List.for_all2
                    (fun s (o, sz) ->
                      ok Sint.(o >= Int 0)
                      && ok Sint.(sz >= Int 0)
                      && ok Sint.(o + sz <= s))
                    ps bounds)
          then bad "bounds";
          numbers (List.map (fun (_, sz) -> ssimplify_sint sz) bounds)
      | Flip flips ->
          if List.length flips <> List.length ps then bad "axes";
          ps)

let ndim u = List.length (shape u)
let numel u = Sint.prod (shape u)
let max_shape u = to_max_shape (shape u)
let max_numel u = size_of (max_shape u)

(* Ranges *)

let range_start = function
  | Op.Stage | Op.Reduce | Op.End | Op.Call -> Some 1
  | Op.Linear -> Some 0
  | _ -> None

let body u =
  match (u.op, u.src) with
  | Op.Call, b :: _ -> b
  | op, _ -> invalid_argf "%s is not a call" (Op.name op)

let rec ended_ranges u =
  match u.ended_ranges_memo with
  | Some l -> l
  | None ->
      let l =
        match u.op with
        | Op.Call
          when (body u).op = Op.Custom_function
               && not (List.is_empty (body u).src) ->
            []
        | Op.End -> List.filter (fun r -> r.op = Op.Range) (drop 1 u.src)
        | Op.Backedge -> (
            match u.src with _ :: loop :: _ -> [ loop ] | _ -> [])
        | op when Option.is_some (range_start op) ->
            drop (Option.get (range_start op)) u.src
        | Op.After -> List.concat_map ended_ranges (drop 1 u.src)
        | Op.Barrier -> List.concat_map ended_ranges u.src
        | Op.Unshard -> drop 1 u.src
        | _ -> []
      in
      u.ended_ranges_memo <- Some l;
      l

let rec ranges u =
  memoized
    ~get:(fun n -> n.ranges_memo)
    ~set:(fun n r -> n.ranges_memo <- Some r)
    ~compute:compute_ranges u

(* The node itself if it is a range, then the ranges of its sources in order of
   first appearance, minus those it ends. *)
and compute_ranges u =
  let all =
    dedup_nodes (List.concat_map (fun s -> Nodes.to_list (ranges s)) u.src)
  in
  let inner =
    List.fold_left
      (fun acc er ->
        if er.op = Op.Range then List.filter (fun r -> r != er) acc
        else
          let gone = ranges er in
          List.filter (fun r -> not (Nodes.mem r gone)) acc)
      all (ended_ranges u)
  in
  Nodes.of_list
    (if u.op = Op.Range then u :: List.filter (fun r -> r != u) inner else inner)

let range_arg u =
  match (u.op, u.arg) with
  | Op.Range, Range { axis_id; axis_type } -> (axis_id, axis_type)
  | op, _ -> invalid_argf "%s is not a range" (Op.name op)

let axis_id u = fst (range_arg u)
let axis_type u = snd (range_arg u)

let range_str ?(color = false) u =
  let s =
    String.concat "_"
      (List.map
         (fun x -> if x >= 0 then string_of_int x else "m" ^ string_of_int (-x))
         (axis_id u))
  in
  if color then Helpers.colored (Axis_type.color (axis_type u)) s else s

let compare_range_arg r0 r1 =
  let id0, t0 = range_arg r0 and id1, t1 = range_arg r1 in
  match Stdlib.compare id0 id1 with 0 -> Axis_type.compare t0 t1 | c -> c

let multirange_str ?color ?pad rs =
  let s =
    String.concat ","
      (List.map (range_str ?color) (List.stable_sort compare_range_arg rs))
  in
  match pad with
  | Some w -> s ^ String.make (max 0 (w - Helpers.ansilen s)) ' '
  | None -> s

(* Construction *)

let as_index (c : Dtype.const) : Dtype.value =
  match c with
  | `Invalid -> invalid_arg "Invalid is not an index"
  | #Dtype.value as x -> x

let is_param_arg = function Param p -> Some p | _ -> None

let param_arg_of u =
  match is_param_arg u.arg with
  | Some p -> p
  | None -> invalid_argf "%s has no ParamArg" (Op.name u.op)

let sink ?kernel ?tag us =
  v Op.Sink ~src:us
    ~arg:(match kernel with Some k -> Kernel k | None -> No_arg)
    ?tag

let group = function [ u ] -> u | us -> v Op.Group ~src:us

let broadcast u n =
  if n = 1 then u else v Op.Stack ~src:(List.init n (fun _ -> u))

let index ?tag u idxs =
  match (u.op, idxs) with
  | Op.Stack, [ ({ op = Op.Const; _ } as c) ] ->
      let i = Value.to_int (as_index (value c)) in
      let n = List.length u.src in
      let i = if i < 0 then n + i else i in
      if i < 0 || i >= n then
        invalid_argf "index %s of a stack of %d" (repr_const (value c)) n;
      List.nth u.src i
  | _ -> v Op.Index ~src:(u :: idxs) ?tag

let load ?tag u rest = v Op.Load ~src:(u :: rest) ?tag
let store ?gate ?tag p x = v Op.Store ~src:([ p; x ] @ Option.to_list gate) ?tag
let end_ u = function [] -> u | rs -> v Op.End ~src:(u :: rs)
let backedge u ~loop ~cond = v Op.Backedge ~src:[ u; loop; cond ]
let after ?tag u = function [] -> u | deps -> v Op.After ~src:(u :: deps) ?tag

let rec without_after u =
  if u.op = Op.After then without_after (first u.op u.src) else u

let barrier u rest = v Op.Barrier ~src:(u :: rest)

let ins ?src ?dtype ?tag u i =
  let src = Option.value src ~default:u.src
  and dtype = Option.value dtype ~default:u.dtype
  and tag = Option.value tag ~default:u.tag in
  v Op.Ins ~src ~arg:(Code { code = i; dtype }) ?tag

let cconst dt c = v Op.Cast ~src:[ const c ] ~arg:(Dtype dt)

let range ?(axis_type = Axis_type.Weak) ?(dtype = Dtype.Weak_int) ?(src = [])
    end_ axis_id =
  v Op.Range
    ~src:(sint_to_uop ~dtype end_ :: src)
    ~arg:(Range { axis_id; axis_type })

let loop axis =
  v Op.Range
    ~src:[ v Op.Noop ]
    ~arg:(Range { axis_id = [ axis ]; axis_type = Weak })

let special end_ name =
  v Op.Special ~src:[ sint_to_uop end_ ] ~arg:(String name)

let wmma ?upcast_axes a b ~acc ~dims ~threads =
  v Op.Wmma ~src:[ a; b; acc ]
    ~arg:(Wmma { dims; dtype_in = a.dtype; threads; upcast_axes })

let reduce u op ranges =
  v Op.Reduce ~src:(u :: ranges) ~arg:(Reduce { op; num_axes = 0 })

let bufferize ?opts u ranges =
  v Op.Stage ~src:(u :: ranges)
    ~arg:(match opts with Some o -> Bufferize o | None -> No_arg)

let get_idx_scalar u =
  match (u.op, u.src) with
  | Op.Where, [ _; x; y ] when is_invalid y -> x
  | _ -> u

let rec get_idx u =
  if u.op = Op.Stack then v Op.Stack ~src:(List.map get_idx u.src)
  else get_idx_scalar u

let rec get_valid u =
  if u.op = Op.Stack then v Op.Stack ~src:(List.map get_valid u.src)
  else
    match (u.op, u.src) with
    | Op.Where, [ c; _; y ] when is_invalid y -> c
    | _ -> const (`Bool (not (is_invalid u)))

(* Devices *)

let rec device u =
  memoized
    ~get:(fun n -> n.device_memo)
    ~set:(fun n d -> n.device_memo <- Some d)
    ~compute:compute_device u

and compute_device u =
  let src0 () = first u.op u.src in
  match (u.op, u.arg) with
  | (Op.Param | Op.Buffer | Op.Alloc), Param p -> p.device
  | Op.Stage, Bufferize b -> b.device
  | Op.Stage, _ | Op.After, _ -> device (src0 ())
  | Op.Mselect, Shard i -> (
      match device (src0 ()) with
      | Some (Multi ds) -> Some (Single (List.nth ds i))
      | _ -> invalid_arg "a shard selection needs a value on several devices")
  | Op.Mstack, _ ->
      Some
        (Multi
           (List.map
              (fun s ->
                match device s with
                | Some (Single d) -> d
                | _ ->
                    invalid_arg
                      "a multi-device stack needs one device per source")
              u.src))
  | Op.Copy, Device d -> Some d
  | Op.Allreduce, Allreduce r -> Some r.device
  | _ -> List.find_map device u.src

let on_disk u =
  match device u with
  | Some (Single d) -> String.starts_with ~prefix:"DISK" d
  | _ -> false

let is_virtual u = Option.is_none (device u) || List.mem u.dtype Dtype.weaks

let rec addrspace u =
  memoized
    ~get:(fun n -> n.addrspace_memo)
    ~set:(fun n a -> n.addrspace_memo <- Some a)
    ~compute:compute_addrspace u

and compute_addrspace u =
  match (u.op, u.arg) with
  | (Op.Param | Op.Buffer | Op.Alloc), Param p -> p.addrspace
  | (Op.Special | Op.Range | Op.Const), _ | Op.Load, _ -> Some Dtype.Alu
  | Op.Binary, _ -> Some Dtype.Global
  | ( ( Op.Index | Op.Cast | Op.After | Op.Reduce | Op.Store | Op.Mstack
      | Op.Mselect | Op.End | Op.Unshard ),
      _ ) ->
      addrspace (first u.op u.src)
  | op, _ when Op.Set.mem op Op.Set.movement -> addrspace (first u.op u.src)
  | op, _
    when List.mem op Op.[ Stack; Wmma; Group ]
         || Op.Set.mem op Op.Set.elementwise -> (
      match List.filter_map addrspace u.src with
      | a :: rest when List.for_all (( = ) a) rest -> Some a
      | _ -> None)
  | _ -> None

let rec has_buffer_identity ?(after_ok = false) u =
  match u.op with
  | Op.Reshape | Op.Unshard | Op.Mselect ->
      has_buffer_identity ~after_ok (first u.op u.src)
  | Op.After when after_ok -> has_buffer_identity ~after_ok (first u.op u.src)
  | op -> List.mem op Op.[ Buffer; Alloc; Param ]

(* Movement *)

let unsharded_base u =
  if Op.Set.mem u.op Op.Set.movement || u.op = Op.Detach || u.op = Op.Unshard
  then base (first u.op u.src)
  else u

let storage_base u =
  let rec strip b =
    if List.mem b.op Op.[ Bitcast; After; Unshard ] then
      strip (unsharded_base (first b.op b.src))
    else b
  in
  strip (unsharded_base u)

let needs_storage u =
  (not (is_virtual u))
  && ((storage_base u).op = Op.Alloc || not (has_buffer_identity u))

let shape_to_shape_arg (arg : sint list) =
  let src = List.map (function Int n -> int n | Sym u -> u) arg in
  List.iter
    (fun x ->
      if not (Dtype.is_int x.dtype) then
        invalid_argf "a shape holds integers, not %s" (repr_dtype x.dtype))
    src;
  match src with [ x ] -> x | src -> v Op.Stack ~src

let mop u (m : movement) =
  let simplified args =
    (simplify (sink (List.map shape_to_shape_arg args))).src
  in
  let scalar_noop () =
    if not (List.is_empty (shape u)) then
      invalid_arg "an empty pad or shrink needs a scalar";
    u
  in
  match m with
  | Expand [] -> u
  | Pad [] | Shrink [] -> scalar_noop ()
  | Reshape s -> v Op.Reshape ~src:(u :: simplified [ s ])
  | Expand s -> v Op.Expand ~src:(u :: simplified [ s ])
  | Pad b -> v Op.Pad ~src:(u :: simplified [ List.map fst b; List.map snd b ])
  | Shrink b ->
      v Op.Shrink ~src:(u :: simplified [ List.map fst b; List.map snd b ])
  | Permute l -> v Op.Permute ~src:[ u ] ~arg:(Axes l)
  | Flip l -> v Op.Flip ~src:[ u ] ~arg:(Flips l)

let resolve_dim ?(extra = 0) u dim =
  let total = ndim u + extra in
  let bound = max 1 total in
  if dim < -bound || dim > bound - 1 then
    invalid_argf "axis %d is out of range [%d, %d]" dim (-bound) (bound - 1);
  if dim < 0 then dim + total else dim

(* A movement that leaves the shape unchanged is no movement. *)
let unless_same u ret = if equal_shape (shape ret) (shape u) then u else ret

let reshape u new_shape =
  let old = shape u in
  let inferred = List.length (List.filter (Sint.equal (Int (-1))) new_shape) in
  if inferred > 1 then
    invalid_argf "only one size can be inferred, in %s" (repr_shape new_shape);
  let new_shape =
    if inferred = 0 then new_shape
    else
      let known = Sint.prod new_shape in
      List.map
        (fun s ->
          if Sint.equal s (Int (-1)) then Sint.(neg (prod old) // known) else s)
        new_shape
  in
  if Sint.truth Sint.(prod old <> prod new_shape) then
    invalid_argf "cannot reshape %s to %s" (repr_shape old)
      (repr_shape new_shape);
  unless_same u (mop u (Reshape new_shape))

let permute u order =
  let order = List.map (resolve_dim u) order in
  let n = ndim u in
  if List.sort Int.compare order <> List.init n Fun.id then
    invalid_argf "%s is not a permutation"
      (repr_tuple (List.map string_of_int order));
  if order = List.init n Fun.id then u else mop u (Permute order)

let broadcast_to u new_shape =
  let old = shape u in
  if equal_shape old new_shape then u
  else begin
    if List.length old > List.length new_shape then
      invalid_argf "cannot broadcast %s to fewer axes, %s" (repr_shape old)
        (repr_shape new_shape);
    let aligned = List.hd (align_left [ old; new_shape ]) in
    if
      not
        (List.for_all2
           (fun s ns -> equal_sint s ns || equal_sint s (Int 1))
           aligned new_shape)
    then
      invalid_argf "cannot broadcast %s to %s" (repr_shape old)
        (repr_shape new_shape);
    let n_left = List.length new_shape - List.length old in
    let expand_at =
      List.filter_map
        (fun i -> if i >= n_left then Some (i - n_left) else None)
        (broadcast_axes old new_shape)
    in
    let kept =
      List.filter
        (fun i -> not (List.mem i expand_at))
        (List.init (List.length old) Fun.id)
    in
    let squeezed = reshape u (List.map (List.nth old) kept) in
    let expanded =
      mop squeezed
        (Expand
           (take n_left new_shape
           @ List.map (fun i -> List.nth new_shape (n_left + i)) expand_at))
    in
    let index_of x l = Option.get (List.find_index (Int.equal x) l) in
    permute expanded
      (List.init n_left Fun.id
      @ List.init (List.length old) (fun i ->
          n_left
          +
          if List.mem i expand_at then index_of i expand_at
          else List.length expand_at + index_of i kept))
  end

let expand u new_shape =
  let aligned = align_left [ shape u; new_shape ] in
  broadcast_to u
    (List.map2
       (fun from to_ -> if Sint.equal to_ (Int (-1)) then from else to_)
       (List.nth aligned 0) (List.nth aligned 1))

let flip u axes =
  let axes = List.map (resolve_dim u) axes in
  if List.length (List.sort_uniq Int.compare axes) <> List.length axes then
    invalid_argf "an axis appears twice in %s"
      (repr_tuple (List.map string_of_int axes));
  let flips = List.init (ndim u) (fun i -> List.mem i axes) in
  if List.exists Fun.id flips then mop u (Flip flips) else u

let check_rank u what l =
  if ndim u <> List.length l then
    invalid_argf "%s of %d axes for a node of %d" what (List.length l) (ndim u)

let shrink u bounds =
  check_rank u "bounds" bounds;
  unless_same u
    (mop u
       (Shrink
          (List.map2
             (fun b s ->
               match b with
               | Some (lo, hi) -> (lo, Sint.(hi - lo))
               | None -> (Int 0, s))
             bounds (shape u))))

let shrink_to u new_shape =
  shrink u (List.map (Option.map (fun ns -> (Int 0, ns))) new_shape)

let movement_pad u pads =
  check_rank u "padding" pads;
  unless_same u
    (mop u
       (Pad
          (List.map2
             (fun (before, after) s -> (before, Sint.(s + before + after)))
             pads (shape u))))

let is_zero (c : Dtype.const) =
  match c with
  | `Invalid -> false
  | #Dtype.value as x -> Value.( = ) x (`Int Bigint.zero)

let const_like ?dtype u c =
  let ret = const ?dtype:(Some (Option.value dtype ~default:u.dtype)) c in
  match shape_opt u with
  | Some (_ :: _ as s) when not (equal_shape (shape ret) s) ->
      mop ret (Expand s)
  | _ -> ret

let pad ?(value = `Int Bigint.zero) u padding =
  let pads = List.map (Option.value ~default:(Int 0, Int 0)) padding in
  check_rank u "padding" pads;
  let has_neg =
    not
      (List.for_all
         (fun p -> Sint.resolve Sint.(p >= Int 0))
         (List.concat_map (fun (b, a) -> [ b; a ]) pads))
  in
  let x, pads =
    if not has_neg then (u, pads)
    else
      ( shrink u
          (List.map2
             (fun (b, a) s ->
               Some (Sint.neg (smin [ b; Int 0 ]), smin [ Sint.(a + s); s ]))
             pads (shape u)),
        List.map (fun (b, a) -> (smax [ b; Int 0 ], smax [ a; Int 0 ])) pads )
  in
  let padded = movement_pad x pads in
  if is_zero value then padded
  else
    where
      (movement_pad (const_like ~dtype:Dtype.Bool x (`Bool true)) pads)
      padded (const value)

let pad_to ?(value = `Int Bigint.zero) u new_shape =
  let to_pad x =
    if List.length new_shape <> ndim x then
      invalid_argf "%d sizes for a node of %d axes" (List.length new_shape)
        (ndim x);
    unless_same x
      (mop x
         (Pad
            (List.map2
               (fun s ns -> (Int 0, Option.value ns ~default:s))
               (shape x) new_shape)))
  in
  let ret = to_pad u in
  if is_zero value || ret == u then ret
  else
    where
      (to_pad (const_like ~dtype:Dtype.Bool u (`Bool true)))
      ret (const value)

let flatten ?(start = 0) ?(stop = -1) u =
  let start = resolve_dim u start and stop = resolve_dim u stop in
  let s = shape u in
  reshape u
    (take start s
    @ [ Sint.prod (take (stop - start + 1) (drop start s)) ]
    @ drop (stop + 1) s)

let unflatten u axis sizes =
  let axis = resolve_dim u axis in
  let s = shape u in
  reshape u (take axis s @ sizes @ drop (axis + 1) s)

let squeeze ?axis u =
  match axis with
  | None ->
      reshape u (List.filter (fun s -> Sint.truth Sint.(s <> Int 1)) (shape u))
  | Some axis ->
      let axis = resolve_dim u axis in
      if ndim u = 0 || Sint.truth Sint.(List.nth (shape u) axis <> Int 1) then u
      else reshape u (List.filteri (fun i _ -> i <> axis) (shape u))

let unsqueeze u axis =
  let axis = resolve_dim ~extra:1 u axis in
  reshape u (take axis (shape u) @ (Int 1 :: drop axis (shape u)))

let transpose u a b =
  let a = resolve_dim u a and b = resolve_dim u b in
  permute u
    (List.init (ndim u) (fun i -> if i = a then b else if i = b then a else i))

let split ?(axis = 0) u sizes =
  let axis = resolve_dim u axis in
  let n =
    match List.nth (shape u) axis with
    | Int n -> n
    | Sym _ -> invalid_arg "a split along an axis of symbolic size"
  in
  let total = List.fold_left ( + ) 0 sizes in
  if total <> n then
    invalid_argf "sizes that sum to %d split an axis of %d elements" total n;
  let cut (lo, pieces) k =
    let bounds =
      List.init (ndim u) (fun i ->
          if i = axis then Some (Int lo, Int (lo + k)) else None)
    in
    (lo + k, shrink u bounds :: pieces)
  in
  List.rev (snd (List.fold_left cut (0, []) sizes))

let repeat u repeats =
  let base =
    List.hd (align_left [ shape u; List.map (fun _ -> Int 1) repeats ])
  in
  let pairs = List.combine repeats base in
  let unsqueezed =
    List.concat_map (fun (r, s) -> if r = 1 then [ s ] else [ Int 1; s ]) pairs
  in
  let expanded =
    List.concat_map (fun (r, s) -> if r = 1 then [ s ] else [ Int r; s ]) pairs
  in
  reshape
    (expand (reshape u unsqueezed) expanded)
    (List.map (fun (r, s) -> Sint.(Int r * s)) pairs)

let pool ?stride ?dilation u kernel =
  let n = List.length kernel in
  let given = function Some l -> l | None -> List.init n (fun _ -> 1) in
  let stride = given stride and dilation = given dilation in
  if ndim u < n then
    invalid_argf "cannot pool %s with %d kernel axes" (repr_shape (shape u)) n;
  if List.length stride <> n || List.length dilation <> n then
    invalid_arg "one stride and one dilation per kernel axis";
  let lead = ndim u - n in
  let noop = take lead (shape u) and keep = List.init lead (fun _ -> None) in
  let axes =
    List.map2
      (fun (k, s) (d, i) -> (k, s, d, i))
      (List.combine kernel stride)
      (List.combine dilation (drop lead (shape u)))
  in
  let reach (k, _, d, _) = d * (k - 1) in
  List.iter
    (fun ((_, _, _, i) as a) ->
      let need = reach a + 1 in
      if not (Sint.resolve Sint.(Int need <= i)) then
        invalid_arg "kernel size cannot be greater than actual input size")
    axes;
  let ceildiv a b = Sint.((a + b - Int 1) // b) in
  let o =
    List.map
      (fun ((_, s, _, i) as a) -> ceildiv Sint.(i - Int (reach a)) (Int s))
      axes
  in
  (* Scales the input so that a stride can be cut from it. *)
  let f =
    List.map2
      (fun (_, s, d, i) o ->
        smax [ Int 1; ceildiv Sint.((o * Int s) - Int d) i ])
      axes o
  in
  let each g = List.map2 (fun a (o, f) -> g a o f) axes (List.combine o f) in
  let span (k, _, d, i) f = Sint.(Int k * ((i * f) + Int d)) in
  let x =
    repeat u
      (List.map (fun _ -> 1) noop
      @ each (fun ((_, _, _, i) as a) _ f ->
          match ceildiv (span a f) i with
          | Int r -> r
          | Sym _ -> invalid_arg "a symbolic pool needs a concrete repeat"))
  in
  let x = shrink_to x (keep @ each (fun a _ f -> Some (span a f))) in
  let x =
    reshape x
      (noop
      @ List.concat
          (each (fun (k, _, d, i) _ f -> [ Int k; Sint.((i * f) + Int d) ])))
  in
  let x =
    shrink_to x
      (keep
      @ List.concat
          (each (fun (k, s, _, _) o _ ->
               [ Some (Int k); Some Sint.(o * Int s) ])))
  in
  let x =
    reshape x
      (noop @ List.concat (each (fun (k, s, _, _) o _ -> [ Int k; o; Int s ])))
  in
  let x =
    shrink_to x
      (keep
      @ List.concat
          (each (fun (k, _, _, _) o _ -> [ Some (Int k); Some o; Some (Int 1) ]))
      )
  in
  let x =
    reshape x (noop @ List.concat (each (fun (k, _, _, _) o _ -> [ Int k; o ])))
  in
  permute x
    (List.init lead Fun.id
    @ List.init n (fun a -> lead + (2 * a) + 1)
    @ List.init n (fun a -> lead + (2 * a)))

let stack ?(axis = 0) us =
  match us with
  | [] -> invalid_arg "stack needs a node"
  | first :: _ ->
      let axis = resolve_dim ~extra:1 first axis in
      let s = shape first in
      if not (List.for_all (fun u -> equal_shape (shape u) s) us) then
        invalid_argf "stacked shapes differ: %s"
          (String.concat ", " (List.map (fun u -> repr_shape (shape u)) us));
      let dt = dtype_of Op.Stack us No_arg in
      let ret =
        v Op.Stack
          ~src:
            (List.map
               (fun u -> if is_invalid (base u) then u else ccast u dt)
               us)
      in
      permute ret
        (List.init axis (fun i -> i + 1)
        @ [ 0 ]
        @ List.init (ndim ret - axis - 1) (fun i -> axis + 1 + i))

let consts ?dtype cs =
  let dtype = match dtype with Some dt -> dt | None -> Dtype.of_consts cs in
  stack (List.map (const ~dtype) cs)

let valid u cond = where cond u (const_like u `Invalid)
let vconst_like u c = broadcast (const ~dtype:u.dtype c) (max_numel u)

let rop u op axes =
  let axes = List.sort Int.compare axes in
  let s = shape u in
  let reduce_axes =
    List.filter (fun a -> Sint.resolve Sint.(List.nth s a <> Int 1)) axes
  in
  let kept = List.filteri (fun i _ -> not (List.mem i axes)) s in
  if List.is_empty reduce_axes then reshape u kept
  else
    let perm =
      reduce_axes
      @ List.filter
          (fun i -> not (List.mem i reduce_axes))
          (List.init (List.length s) Fun.id)
    in
    let ret =
      v Op.Reduce
        ~src:[ permute u perm ]
        ~arg:(Reduce { op; num_axes = List.length reduce_axes })
    in
    if axes <> reduce_axes then reshape ret kept else ret

(* Data types *)

let bitcast x dt =
  if List.mem x.dtype Dtype.weaks || List.mem dt Dtype.weaks then
    invalid_argf "a bitcast needs committed types, not %s to %s"
      (repr_dtype x.dtype) (repr_dtype dt);
  if Dtype.equal x.dtype dt then x else v Op.Bitcast ~src:[ x ] ~arg:(Dtype dt)

let commit_dtype ?default_int x =
  if Dtype.equal x.dtype Dtype.Weak_int then
    Dtype.commit_int ?default_int (Value.to_z (vmin x)) (Value.to_z (vmax x))
  else Dtype.strong x.dtype

let element_size x =
  if List.mem x.dtype Dtype.weaks then
    invalid_argf "a %s has no size" (repr_dtype x.dtype);
  Dtype.itemsize x.dtype

let nbytes u =
  match numel u with
  | Int n -> n * element_size u
  | Sym s -> Bigint.to_int (to_z s) * element_size u

let contiguous x =
  if List.mem x.dtype Dtype.weaks then x
  else if x.op = Op.Stage || Option.is_none (device x) || has_buffer_identity x
  then x
  else v Op.Stage ~src:[ x ]

let usum x ys =
  List.fold_left
    (if Dtype.equal x.dtype Dtype.Bool then bitwise_or else add)
    x ys

let uprod x ys =
  List.fold_left
    (if Dtype.equal x.dtype Dtype.Bool then bitwise_and else mul)
    x ys

let cat ?(axis = 0) u rest =
  let axis = resolve_dim u axis in
  let s = shape u in
  List.iter
    (fun x ->
      let sx = shape x in
      if
        List.length sx <> List.length s
        || not
             (List.for_all2
                (fun (i, a) b -> i = axis || equal_sint a b)
                (List.mapi (fun i a -> (i, a)) s)
                sx)
      then
        invalid_argf "cannot concatenate %s and %s" (repr_shape s)
          (repr_shape sx))
    rest;
  let dim x = List.nth (shape x) axis in
  if List.for_all (fun x -> equal_sint (dim x) (dim u)) rest then
    flatten ~start:axis ~stop:(axis + 1) (stack ~axis (u :: rest))
  else
    let all = u :: rest in
    let starts =
      List.rev
        (List.fold_left
           (fun acc x -> Sint.(List.hd acc + dim x) :: acc)
           [ Int 0 ] all)
    in
    let total = List.nth starts (List.length all) in
    let padded =
      List.mapi
        (fun i x ->
          pad x
            (List.init (ndim x) (fun j ->
                 if j = axis then
                   let next = List.nth starts (i + 1) in
                   Some (List.nth starts i, Sint.(total - next))
                 else None)))
        all
    in
    usum (List.hd padded) (List.tl padded)

(* Running operations *)

let split_cumalu = 256

(* [None] for each axis of [u] but [axis], which is [Some p]. *)
let at_axis u axis p =
  List.init (ndim u) (fun i -> if i = axis then Some p else None)

let running_size u axis =
  match List.nth (shape u) axis with
  | Int n -> n
  | Sym _ -> invalid_arg "a running operation along an axis of symbolic size"

(* The running [op] along the last axis of [u]: over each element's window of
   the elements up to it, the axis padded before with [op]'s identity. *)
let pooled_cumalu u op =
  let last = ndim u - 1 in
  let n = running_size u last in
  let value = identity_element op u.dtype in
  rop
    (pool (pad ~value u (at_axis u last (Int (n - 1), Int 0))) [ n ])
    op
    [ last + 1 ]

let cumalu u axis op =
  let axis = resolve_dim u axis and last = ndim u - 1 in
  let s = running_size u axis and t = transpose u axis last in
  if List.exists (fun d -> equal_sint d (Int 0)) (shape u) then u
  else if s <= 2 * split_cumalu then transpose (pooled_cumalu t op) axis last
  else
    let value = identity_element op u.dtype in
    let rounded = Helpers.round_up s split_cumalu in
    let t = pad ~value t (at_axis t last (Int (rounded - s), Int 0)) in
    let chunks =
      pooled_cumalu
        (unflatten t last [ Int (rounded / split_cumalu); Int split_cumalu ])
        op
    in
    let ends =
      squeeze ~axis:(last + 1)
        (shrink chunks
           (at_axis chunks (last + 1)
              (Int (split_cumalu - 1), Int split_cumalu)))
    in
    let base =
      pad ~value (pooled_cumalu ends op) (at_axis ends last (Int 1, Int (-1)))
    in
    let combine =
      if Op.equal op Op.Add then add
      else if Op.equal op Op.Mul then mul
      else maximum
    in
    let whole =
      flatten ~start:last
        (combine chunks (reshape base (shape base @ [ Int 1 ])))
    in
    transpose
      (shrink whole (at_axis whole last (Int (rounded - s), Int rounded)))
      axis last

let arange ?(start = 0) ?(step = 1) ?dtype stop =
  if step = 0 then invalid_arg "an arange of step 0";
  let lo, hi =
    if step > 0 then (start, stop - step) else (stop - step, start)
  in
  let dt =
    match dtype with
    | Some dt -> dt
    | None -> Dtype.commit_int (Bigint.of_int lo) (Bigint.of_int hi)
  in
  if
    Value.(`Int (Bigint.of_int lo) < Dtype.min dt)
    || Value.(Dtype.max dt < `Int (Bigint.of_int hi))
  then
    invalid_argf "arange [%d, %d) is not representable in %s" start stop
      (repr_dtype dt);
  let n = Helpers.ceildiv (stop - start) step in
  let full dt c k = expand (const ~dtype:dt (`Int (Bigint.of_int c))) [ Int k ] in
  if n <= 0 then full dt 0 0
  else
    let acc =
      if Dtype.is_float dt then Dtype.least_upper [ dt; Float32 ] else dt
    in
    cast (add (pooled_cumalu (full acc step n) Op.Add) (int (start - step))) dt

(* Several devices *)

let rec axis u =
  match u.axis_memo with
  | Some a -> a
  | None ->
      let a = compute_axis u in
      u.axis_memo <- Some a;
      a

and compute_axis u =
  let src0 () = first u.op u.src in
  match u.op with
  | Op.Copy | Op.Param -> None
  | Op.Unshard -> (
      match u.arg with
      | Axes [ a ] -> Some a
      | Axes l ->
          invalid_argf "the value is sharded on several axes, %s"
            (repr_tuple (List.map string_of_int l))
      | _ -> invalid_arg "an unshard needs its axes")
  | op when Op.Set.mem op Op.Set.alu || op = Op.Stack -> (
      let n = ndim u in
      let axes =
        List.filter_map
          (fun x -> Option.map (fun a -> a + n - ndim x) (axis x))
          u.src
      in
      match List.rev (Helpers.dedup (module Int) axes) with
      | [] -> None
      | last :: _ -> Some last)
  | _ when List.is_empty u.src -> None
  | op -> (
      let src_axis = axis (src0 ()) in
      match (op, src_axis) with
      | Op.Shrink, Some a ->
          let o, sz = List.nth (marg_bounds u) a in
          if
            equal_sint o (Int 0) && equal_sint sz (List.nth (shape (src0 ())) a)
          then Some a
          else None
      | Op.Reduce, a -> (
          match (a, u.arg) with
          | None, _ -> None
          | Some a, Reduce { num_axes; _ } ->
              if a < num_axes then None else Some (a - num_axes)
          | _ -> invalid_arg "a reduction needs its argument")
      | Op.Reshape, None -> None
      | Op.Reshape, Some a -> Some (reshape_axis u a)
      | Op.Permute, a -> (
          match (a, marg u) with
          | Some a, Permute order -> List.find_index (Int.equal a) order
          | _ -> None)
      | Op.Expand, a -> (
          match (a, marg u) with
          | Some a, Expand s -> Some (a + List.length s)
          | _ -> None)
      | _, a -> a)

(* The new axis is the last one before which the element count is the count
   before the source's axis, and it must not move elements between shards. *)
and reshape_axis u src_axis =
  let src = first u.op u.src in
  let new_shape = marg_shape u in
  let prefix =
    List.rev
      (List.fold_left
         (fun acc s -> Sint.(List.hd acc * s) :: acc)
         [ Int 1 ] new_shape)
  in
  let acc = List.map ssimplify_sint prefix in
  let target = ssimplify_sint (Sint.prod (take src_axis (shape src))) in
  let moved () =
    invalid_argf "a reshape of %s to %s moves elements between shards"
      (repr_shape (shape src))
      (repr_shape (shape u))
  in
  let rec last_index i best = function
    | [] -> best
    | x :: rest ->
        last_index (i + 1) (if equal_sint x target then Some i else best) rest
  in
  let new_axis =
    match last_index 0 None acc with Some i -> i | None -> moved ()
  in
  let dcount =
    match device u with
    | Some (Multi ds) -> List.length ds
    | _ -> (
        match List.find_opt (fun n -> n.op = Op.Unshard) (toposort src) with
        | Some un -> Value.to_int (vmax (nth un 1)) + 1
        | None -> moved ())
  in
  if Sint.truth Sint.(List.nth (shape u) new_axis % Int dcount <> Int 0) then
    moved ();
  new_axis

let sharding u =
  match (u.op, u.arg) with
  | Op.Unshard, Axes axes -> List.combine axes (drop 1 u.src)
  | _ -> []

let shard_count u =
  if u.op = Op.Unshard then Value.to_int (vmax (nth u 1)) + 1
  else
    match device u with
    | Some (Multi ds) -> List.length ds
    | _ -> invalid_arg "the value is not on several devices"

let bounds u =
  match axis u with
  | None -> invalid_arg "bounds need a sharded value"
  | Some a ->
      let size = List.nth (shape (first u.op u.src)) a in
      let starts =
        List.rev
          (List.fold_left
             (fun acc _ -> Sint.(List.hd acc + size) :: acc)
             [ Int 0 ]
             (List.init (shard_count u) Fun.id))
      in
      let rec pairs = function
        | x :: (y :: _ as rest) -> (x, y) :: pairs rest
        | _ -> []
      in
      pairs starts

let shard_shape u =
  match device u with
  | Some (Multi _) -> (
      match axis u with
      | Some a ->
          let n = shard_count u in
          List.mapi
            (fun i x -> if i = a then Sint.(x // Int n) else x)
            (shape u)
      | None -> shape u)
  | _ -> shape u

let max_shard_shape u = to_max_shape (shard_shape u)

let device_range_src = function
  | Some (Multi ds) ->
      [ range ~axis_type:Axis_type.Device (Int (List.length ds)) [ -1 ] ]
  | _ -> []

let unshard ?ranges u axes =
  let ranges =
    match ranges with
    | Some r -> r
    | None -> (
        match device u with
        | Some (Multi ds) ->
            [ range ~axis_type:Axis_type.Device (Int (List.length ds)) [ -1 ] ]
        | _ -> invalid_arg "an unshard needs a value on several devices")
  in
  if
    List.length axes <> List.length ranges
    || List.length (List.sort_uniq Int.compare axes) <> List.length axes
  then invalid_arg "an unshard needs one range per distinct axis";
  let pairs =
    List.stable_sort
      (fun (a, _) (b, _) -> Int.compare a b)
      (List.combine axes ranges)
  in
  v Op.Unshard ~src:(u :: List.map snd pairs) ~arg:(Axes (List.map fst pairs))

let copy_to_device ?shard u d =
  if is_disk_device d then
    invalid_arg "cannot copy to a disk; store into a disk buffer instead";
  let inp =
    match shard with
    | None -> u
    | Some i ->
        (match device u with
        | Some (Multi _) -> ()
        | _ -> invalid_arg "a shard copy needs a value on several devices");
        mselect u i
  in
  if List.mem inp.dtype Dtype.weaks then
    invalid_argf "cannot store a weak %s" (repr_dtype inp.dtype);
  v Op.Copy ~src:(inp :: device_range_src (Some d)) ~arg:(Device d)

let shard_slice u a rng =
  match shape u with
  | [] -> u
  | s ->
      let dcount = Value.to_int (vmax rng) + 1 in
      let size = List.nth s a in
      if Sint.truth Sint.(size % Int dcount <> Int 0) then
        invalid_argf "axis %d of size %s does not split over %d devices" a
          (repr_sint size) dcount;
      let sz = Sint.(size // Int dcount) in
      let r = Sym rng in
      shrink u
        (List.mapi
           (fun i x ->
             if i = a then Some (Sint.(r * sz), Sint.((r * sz) + sz))
             else Some (Int 0, x))
           s)

let shard ?axis u devices =
  let copied = copy_to_device u (Multi devices) in
  match axis with
  | None -> copied
  | Some a ->
      let rng =
        range ~axis_type:Axis_type.Device (Int (List.length devices)) [ -1 ]
      in
      unshard (shard_slice copied a rng) [ a ]

let mstack u = function [] -> u | rest -> v Op.Mstack ~src:(u :: rest)

let allreduce u op d =
  (match device u with
  | Some (Multi _) -> ()
  | _ -> invalid_arg "an allreduce needs a value on several devices");
  v Op.Allreduce ~src:[ u ] ~arg:(Allreduce { op; device = d })

(* Storage *)

let unique = Atomic.make 0
let unique_num () = Atomic.fetch_and_add unique 1

let getaddr ?device:dev u =
  if
    not
      (List.mem (without_after u).op
         Op.
           [
             Buffer;
             Alloc;
             Shrink;
             Bitcast;
             Binary;
             Mstack;
             Mselect;
             Param;
             Linear;
           ])
  then u
  else
    let d =
      match (dev, device u) with
      | Some d, _ -> d
      | None, Some d -> List.hd (device_names d)
      | None, None -> invalid_arg "an address needs a device"
    in
    v Op.Getaddr ~src:[ u ] ~arg:(Device (Single d))

let param_arg ?size ?vmin_vmax ?multiple_of ?name
    ?(addrspace = Some Dtype.Global) ?device ?(volatile = false)
    ?(bind_on_realize = false) ?bound ?(phase = 0) ?(align = 16) ~slot dtype =
  if not (List.mem align [ 1; 2; 4; 8; 16 ]) then
    invalid_argf "alignment %d is not a power of two up to 16" align;
  if
    phase < 0 || phase >= align
    || phase mod min (Dtype.itemsize dtype) align <> 0
  then
    invalid_argf "phase %d is not a multiple of a %s's size below %d" phase
      (repr_dtype dtype) align;
  {
    slot;
    dtype;
    size;
    vmin_vmax;
    multiple_of;
    name;
    addrspace;
    device;
    volatile;
    bind_on_realize;
    bound;
    phase;
    align;
  }

let weak_storage dt =
  if List.mem dt Dtype.weaks then
    invalid_argf "a %s cannot be stored" (repr_dtype dt)

let new_buffer ?slot ?phase d size dt =
  weak_storage dt;
  let slot = match slot with Some s -> s | None -> unique_num () in
  v Op.Buffer
    ~src:(device_range_src (Some d))
    ~arg:(Param (param_arg ~slot ~size ~device:d ?phase dt))

let empty ?device new_shape dt =
  weak_storage dt;
  let max_shape = to_max_shape new_shape in
  let u =
    v Op.Alloc ~src:(device_range_src device)
      ~arg:
        (Param
           (param_arg ~slot:(unique_num ()) ~size:(size_of max_shape) ?device
              ~bind_on_realize:true dt))
  in
  shrink_to
    (reshape u (List.map (fun n -> Int n) max_shape))
    (List.map Option.some new_shape)

let view_as ?axis u new_shape =
  let max_shape = List.map (fun n -> Int n) (to_max_shape new_shape) in
  let ret = if List.length new_shape > 1 then reshape u max_shape else u in
  let ret =
    if equal_shape max_shape new_shape then ret
    else shrink_to ret (List.map Option.some new_shape)
  in
  match axis with None -> ret | Some a -> unshard ret [ a ]

let empty_like ?dtype ?device:dev u =
  let dev = match dev with Some d -> Some d | None -> device u in
  let dtype = match dtype with Some dt -> dt | None -> commit_dtype u in
  match dev with
  | Some (Multi _) when Option.is_some (axis u) ->
      unshard (empty ?device:dev (shard_shape u) dtype) [ Option.get (axis u) ]
  | _ -> empty ?device:dev (shape u) dtype

let clone ?device:dev u =
  let dev = match dev with Some d -> Some d | None -> device u in
  (match dev with
  | Some d when is_disk_device d ->
      invalid_arg "cannot clone a disk; store into a disk buffer instead"
  | _ -> ());
  let ret = empty_like ?device:dev u in
  let src =
    match (device u, dev) with
    | None, _ -> u
    | Some d, Some d' when equal_device d d' -> u
    | _, Some d' -> copy_to_device u d'
    | _, None -> u
  in
  after ret [ store ret (cast src ret.dtype) ]

let alloc ?slot ?(addrspace = Dtype.Global) ?device ?axis new_shape dt =
  let slot = match slot with Some s -> s | None -> unique_num () in
  let ret =
    v Op.Alloc ~src:(device_range_src device)
      ~arg:
        (Param
           (param_arg ~slot
              ~size:(size_of (to_max_shape new_shape))
              ~addrspace:(Some addrspace) ?device (Dtype.strong dt)))
  in
  if List.is_empty new_shape then reshape ret []
  else view_as ?axis ret new_shape

let alloc_like ?slot ?addrspace u =
  alloc ?slot ?addrspace (List.map (fun n -> Int n) (max_shard_shape u)) u.dtype

let placeholder ?slot ?(addrspace = Dtype.Global) ?device ?(volatile = false)
    ?tag new_shape dt =
  let dt = Dtype.strong dt in
  let slot = match slot with Some s -> s | None -> unique_num () in
  let name = match tag with Some (Tag.String s) -> Some s | _ -> None in
  let size = size_of new_shape in
  let ret =
    match addrspace with
    | Dtype.Global ->
        v Op.Param
          ~arg:
            (Param
               (param_arg ~slot ~size ?name ~addrspace:(Some addrspace) ?device
                  ~volatile dt))
    | Dtype.Local | Dtype.Reg ->
        if Option.is_some device then
          invalid_arg "workgroup and register storage has no device";
        v Op.Buffer
          ~arg:
            (Param (param_arg ~slot ~size ?name ~addrspace:(Some addrspace) dt))
    | Dtype.Alu -> invalid_arg "a placeholder cannot be a scalar variable"
  in
  let ret = match tag with Some g -> rtag ~tag:g ret | None -> ret in
  if List.length new_shape > 1 then
    reshape ret (List.map (fun n -> Int n) new_shape)
  else ret

let placeholder_like ?addrspace u slot =
  if not (List.for_all (function Int _ -> true | Sym _ -> false) (shape u))
  then invalid_arg "a placeholder needs a shape of known sizes";
  placeholder ~slot ?addrspace (max_shard_shape u) u.dtype

let param ?shape:new_shape ?device ?vmin_vmax ?multiple_of ?name
    ?(addrspace = Some Dtype.Global) ?(volatile = false) ?phase ?align slot dt
    =
  weak_storage dt;
  let make size =
    v Op.Param
      ~arg:
        (Param
           (param_arg ?size ?vmin_vmax ?multiple_of ?name ~addrspace ?device
              ~volatile ?phase ?align ~slot dt))
  in
  match new_shape with
  | None | Some [] -> make None
  | Some s -> view_as (make (Some (size_of (to_max_shape s)))) s

let set ?(ends = []) p x = after (first p.op p.src) [ end_ (store p x) ends ]

(* Variables *)

let variable ?(dtype = Dtype.Weak_int) ?(multiple_of = 1) name lo hi =
  v Op.Param
    ~arg:
      (Param
         (param_arg ~vmin_vmax:(lo, hi) ~multiple_of ~name
            ~addrspace:(Some Dtype.Alu) ~slot:(-1) dtype))

let is_variable u =
  u.op = Op.Param
  && (match u.arg with
    | Param { vmin_vmax = Some _; addrspace = Some Dtype.Alu; _ } -> true
    | _ -> false)
  && match shape_opt u with Some [] -> true | _ -> false

let is_bound_var u =
  is_variable u
  && match u.arg with Param { bound = Some _; _ } -> true | _ -> false

let expr u =
  match (u.op, u.arg) with
  | (Op.Param | Op.Buffer), Param { name = Some n; _ } -> n
  | op, _ -> invalid_argf "%s has no name" (Op.name op)

(* Divisibility *)

let rec const_factor u : Bigint.t =
  match u.op with
  | Op.Const -> integer (value u)
  | Op.Stack ->
      List.fold_left (fun g s -> Bigint.gcd g (const_factor s)) Bigint.zero u.src
  | Op.Add -> Bigint.gcd (const_factor (nth u 0)) (const_factor (nth u 1))
  | Op.Mul ->
      if (nth u 0).op = Op.Const then integer (value (nth u 0))
      else if (nth u 1).op = Op.Const then integer (value (nth u 1))
      else Bigint.one
  | op when Op.Set.mem op Op.Set.defines -> (
      match u.arg with
      | Param { multiple_of = Some m; _ } -> Bigint.of_int m
      | _ -> Bigint.one)
  | _ -> Bigint.one

and as_value (c : Dtype.const) : Dtype.value =
  match c with
  | `Invalid -> invalid_arg "Invalid is not a number"
  | #Dtype.value as x -> x

(* A constant that takes part in a greatest common divisor: an integer. *)
and integer c =
  match as_value c with
  | `Float x when not (Float.is_integer x) ->
      invalid_argf "%s is not an integer" (repr_const c)
  | v -> Value.to_z v

let rec divides u (n : Bigint.t) =
  if Bigint.equal n Bigint.one then Some u
  else
    match u.op with
    | Op.Const ->
        let x = as_value (value u) and n = `Int n in
        if Value.(x % n = of_int 0) then
          Some (const_like u (Value.(x // n) :> Dtype.const))
        else None
    | Op.Stack ->
        let srcs = List.map (fun s -> divides s n) u.src in
        if List.exists Option.is_none srcs then None
        else Some (v Op.Stack ~src:(List.map Option.get srcs))
    | Op.Add -> (
        match (divides (nth u 0) n, divides (nth u 1) n) with
        | Some d0, Some d1 -> Some (add d0 d1)
        | _ -> None)
    | Op.Mul -> (
        match divides (nth u 0) n with
        | Some d0 -> Some (mul d0 (nth u 1))
        | None -> Option.map (fun d1 -> mul (nth u 0) d1) (divides (nth u 1) n))
    | op when Op.Set.mem op Op.Set.defines -> (
        match u.arg with
        | Param { multiple_of = Some m; _ } ->
            if Bigint.equal (Bigint.rem (Bigint.of_int m) n) Bigint.zero then
              Some (div ~rounding:`Floor u (const (`Int n)))
            else None
        | _ -> None)
    | _ -> None

let pop_const ?(op = Op.Add) u : t * Dtype.const =
  match u.src with
  | [ x; c ] when Op.equal u.op op && c.op = Op.Const -> (x, value c)
  | _ -> (u, identity_element op u.dtype)

(* The alignment and phase of the storage [u] views: its storage's, moved by
   the bytes a shrink of storage seen whole skips, known modulo fewer bytes when
   the shrink's start is symbolic. Any other view keeps its storage's, as views
   are taken to start aligned, unless a symbolic start moves it; storage the
   graph allocates, a stage's included, starts on a boundary. *)
let rec storage_phase u =
  let rec whole v =
    match (v.op, v.src) with
    | (Op.Buffer | Op.Param | Op.Alloc | Op.Stage), _ -> true
    | (Op.Bitcast | Op.Reshape | Op.After | Op.Mselect), v :: _ -> whole v
    | _ -> false
  in
  let ints l = List.for_all (function Int _ -> true | Sym _ -> false) l in
  match (u.op, u.src) with
  | _ when on_disk u -> (16, 0)
  | (Op.Buffer | Op.Param | Op.Alloc), _ ->
      let p = param_arg_of u in
      (p.align, p.phase)
  | Op.Shrink, x :: _ -> (
      let align, phase = storage_phase x in
      let bytes = element_size x in
      (* The start moved by [first] bytes and by multiples of [by] bytes: known
         modulo the largest power of two up to [align] that divides [by]. *)
      let moved by first =
        let rec known a =
          if a < align && Bigint.divisible by (Bigint.of_int (2 * a)) then known (2 * a)
          else a
        in
        let a = known 1 in
        (a, (((phase + first) mod a) + a) mod a)
      in
      match marg u with
      | Shrink bounds when whole x && ints (shape x) -> (
          (* The element the shrink starts at, by the row-major strides of [x]'s
             shape: a constant, and multiples of a constant that move. *)
          let offset =
            List.fold_left2
              (fun acc (start, _) size -> Sint.((acc * size) + start))
              (Int 0) bounds (shape x)
          in
          match offset with
          | Int n -> moved Bigint.zero (n * bytes)
          | Sym e ->
              let rest, c = pop_const (simplify e) in
              moved
                (Bigint.mul (const_factor rest) (Bigint.of_int bytes))
                (Bigint.to_int (integer c) * bytes))
      | Shrink bounds when not (ints (List.map fst bounds)) ->
          (* A symbolic start into a view that reorders or pads its storage
             moves by bytes unknown here: only the element's size holds. *)
          moved (Bigint.of_int bytes) 0
      | _ -> (align, phase))
  | (Op.Bitcast | Op.After | Op.Mselect), x :: _ -> storage_phase x
  | o, x :: _ when Op.Set.mem o Op.Set.movement -> storage_phase x
  | _ -> (16, 0)

let param_like u slot =
  match u.op with
  | Op.Param when addrspace u = Some Dtype.Alu ->
      let p = param_arg_of u in
      v Op.Param ~arg:(Param { p with slot; name = None; bound = None })
  | _ -> (
      let a = axis u and align, phase = storage_phase u in
      match (a, device u) with
      | Some a, Some (Multi _ as d) ->
          let ss = shard_shape u in
          view_as ~axis:a
            (v Op.Param
               ~arg:
                 (Param
                    (param_arg ~slot
                       ~size:(size_of (to_max_shape ss))
                       ~device:d ~phase ~align u.dtype)))
            ss
      | _ ->
          param ?shape:(shape_opt u) ?device:(device u) ~phase ~align slot
            u.dtype)

(* Multisets of nodes, in order of first insertion. *)
let add_count k counts t =
  match List.assq_opt t counts with
  | Some _ ->
      List.map (fun (u, m) -> if u == t then (u, m + k) else (u, m)) counts
  | None -> counts @ [ (t, k) ]

let count_terms terms = List.fold_left (add_count 1) [] terms
let subtract counts terms = List.fold_left (add_count (-1)) counts terms

let elements counts =
  List.concat_map (fun (u, n) -> List.init (max 0 n) (fun _ -> u)) counts

let product start terms = List.fold_left mul start terms

let gcd us =
  match us with
  | [] -> invalid_arg "gcd of nothing"
  | first_u :: _ ->
      let popped = List.map (pop_const ~op:Op.Mul) us in
      let common =
        List.fold_left
          (fun acc (term, _) ->
            let c = count_terms (split_uop term Op.Mul) in
            List.filter_map
              (fun (u, n) ->
                match List.assq_opt u c with
                | Some m when min n m > 0 -> Some (u, min n m)
                | _ -> None)
              acc)
          (count_terms (split_uop (fst (List.hd popped)) Op.Mul))
          (List.tl popped)
      in
      let factors =
        if List.is_empty common then List.map const_factor us
        else List.map (fun (_, c) -> integer c) popped
      in
      product
        (const_like first_u (`Int (List.fold_left Bigint.gcd Bigint.zero factors)))
        (elements common)

let rec divide_exact u d =
  if u == d then Some (const_like u (`Int Bigint.one))
  else if d.op = Op.Const then divides u (Value.to_z (as_value (value d)))
  else
    match u.op with
    | Op.Add -> (
        match (divide_exact (nth u 0) d, divide_exact (nth u 1) d) with
        | Some s0, Some s1 -> Some (add s0 s1)
        | _ -> None)
    | Op.Mul ->
        let fac, c = pop_const ~op:Op.Mul u
        and dfac, dc = pop_const ~op:Op.Mul d in
        let c = as_value c and dc = as_value dc in
        let counts =
          subtract (count_terms (split_uop fac Op.Mul)) (split_uop dfac Op.Mul)
        in
        if
          Value.(c % dc = of_int 0)
          && List.for_all (fun (_, n) -> n >= 0) counts
        then
          Some
            (product
               (const_like u (Value.(c // dc) :> Dtype.const))
               (elements counts))
        else None
    | _ -> None

(* Evaluation *)

let py_pow (x : Dtype.value) (y : Dtype.value) : Dtype.value =
  match (x, y) with
  | (`Bool _ | `Int _), (`Bool _ | `Int _) when Bigint.geq (Value.to_z y) Bigint.zero ->
      `Int (Bigint.pow (Value.to_z x) (Value.to_int y))
  | _ ->
      let fx = Value.to_float x and fy = Value.to_float y in
      (* C's pow is Python's, except that a zero to a negative power raises in
         Python, which the folding catches as an infinity. *)
      if Float.equal fx 0. && fy < 0. then `Float Float.infinity
      else `Float (Float.pow fx fy)

let py_exp2 (x : Dtype.value) : Dtype.value =
  match x with
  | (`Bool _ | `Int _) when Bigint.geq (Value.to_z x) Bigint.zero ->
      `Int (Bigint.shift_left Bigint.one (Value.to_int x))
  | _ -> `Float (Float.pow 2. (Value.to_float x))

let int_operands op (x : Dtype.value) (y : Dtype.value) =
  match (x, y) with
  | `Float _, _ | _, `Float _ -> invalid_argf "%s needs integers" (Op.name op)
  | _ -> (Value.to_z x, Value.to_z y)

(* A shift's operands, its count as an [int]. *)
let shift_operands op x y =
  let a, b = int_operands op x y in
  if Bigint.sign b < 0 then
    invalid_argf "a shift by a negative count, %s, has no value" (Bigint.to_string b);
  (a, Bigint.to_int b)

let bitwise op fb fz (x : Dtype.value) (y : Dtype.value) : Dtype.value =
  match (x, y) with
  | `Bool a, `Bool b -> `Bool (fb a b)
  | _ ->
      let a, b = int_operands op x y in
      `Int (fz a b)

let python_alu op (args : Dtype.value list) : Dtype.value =
  let unary f =
    match args with
    | [ x ] -> f x
    | _ -> invalid_argf "%s takes one operand" (Op.name op)
  in
  let binary f =
    match args with
    | [ x; y ] -> f x y
    | _ -> invalid_argf "%s takes two operands" (Op.name op)
  in
  match op with
  | Op.Log2 ->
      unary (fun x ->
          if Value.( < ) (`Int Bigint.zero) x then
            `Float (Float.log2 (Value.to_float x))
          else if Value.( = ) x (`Int Bigint.zero) then `Float Float.neg_infinity
          else `Float Dtype.nan)
  | Op.Exp2 -> unary py_exp2
  | Op.Sqrt ->
      unary (fun x ->
          if Value.( <= ) (`Int Bigint.zero) x then
            `Float (Float.sqrt (Value.to_float x))
          else `Float Dtype.nan)
  | Op.Reciprocal ->
      unary (fun x ->
          let f = Value.to_float x in
          if Float.equal f 0. then `Float (Float.copy_sign Float.infinity f)
          else `Float (1. /. f))
  | Op.Sin ->
      unary (fun x ->
          let f = Value.to_float x in
          `Float
            (if Float.equal (Float.abs f) Float.infinity then Dtype.nan
             else Float.sin f))
  | Op.Pow -> binary py_pow
  | Op.Trunc ->
      unary (fun x ->
          match x with
          | `Float f -> `Float (Float.trunc f)
          | x -> `Int (Value.to_z x))
  | Op.Neg -> unary Value.( ~- )
  | Op.Add -> binary Value.( + )
  | Op.Sub -> binary Value.( - )
  | Op.Mul -> binary Value.( * )
  | Op.Fdiv -> binary (fun x y -> `Float (Value.to_float x /. Value.to_float y))
  | Op.Cmpne -> binary (fun x y -> `Bool (not (Value.( = ) x y)))
  | Op.Cmplt -> binary (fun x y -> `Bool (Value.( < ) x y))
  | Op.Cmpeq -> binary (fun x y -> `Bool (Value.( = ) x y))
  | Op.Xor -> binary (bitwise op ( <> ) Bigint.logxor)
  | Op.Or -> binary (bitwise op ( || ) Bigint.logor)
  | Op.And -> binary (bitwise op ( && ) Bigint.logand)
  | Op.Shr ->
      binary (fun x y ->
          let a, k = shift_operands op x y in
          `Int (Bigint.shift_right a k))
  | Op.Shl ->
      binary (fun x y ->
          let a, k = shift_operands op x y in
          `Int (Bigint.shift_left a k))
  | Op.Max -> binary Value.max
  | Op.Cmod -> binary (divide `Remainder ~toward_zero:true)
  | Op.Cdiv -> binary (divide `Quotient ~toward_zero:true)
  | Op.Floordiv -> binary (divide `Quotient ~toward_zero:false)
  | Op.Floormod -> binary (divide `Remainder ~toward_zero:false)
  | Op.Mulacc -> (
      match args with
      | [ x; y; z ] -> Value.( + ) (Value.( * ) x y) z
      | _ -> invalid_arg "MULACC takes three operands")
  | op -> invalid_argf "%s is not computed on constants" (Op.name op)

(* A multiply-add of floats rounds once (D25). Truncated to a dtype below
   float64, its operands have at most 24 bits, so their product is exact in a
   double, and the sum is rounded to odd there, which the truncation then rounds
   as the exact value. Otherwise, and for a weak float, a double, it is the
   double nearest the exact value. *)
let mulacc ~truncated dt a b c =
  if
    (not truncated)
    || Dtype.equal dt Dtype.Float64
    || Dtype.equal dt Dtype.Weak_float
  then Float.fma a b c
  else
    let p = a *. b in
    let r = p +. c in
    let d = r -. p in
    let e = p -. (r -. d) +. (c -. d) in
    let odd = Int64.logand (Int64.bits_of_float r) 1L = 1L in
    if (not (Float.is_finite r)) || e = 0. || odd then r
    else if e > 0. then Float.succ r
    else Float.pred r

let exec_alu ?(truncate_output = true) op dt (args : Dtype.const list) :
    Dtype.const =
  let truncate (x : Dtype.value) : Dtype.const =
    if
      truncate_output
      && not (Dtype.equal dt Dtype.Void || List.mem dt Dtype.weaks)
    then (Dtype.truncate dt x :> Dtype.const)
    else (x :> Dtype.const)
  in
  if
    Op.Set.mem op Op.Set.binary
    && List.exists (function `Invalid -> true | _ -> false) args
  then `Invalid
  else
    match (op, args) with
    | Op.Where, [ c; x; y ] -> (
        match if Value.to_bool (as_value c) then x else y with
        | `Invalid -> `Invalid
        | #Dtype.value as v -> truncate v)
    | _ ->
        let args = List.map as_value args in
        let is_nan = function `Float f -> Float.is_nan f | _ -> false in
        (* The NaN of an invalid operation is the canonical one, whatever the
           host's FPU makes of it: x86 gives a negative NaN. *)
        let v =
          match (op, args) with
          | Op.Mulacc, [ a; b; c ] when Dtype.is_float dt ->
              let f = Value.to_float in
              `Float (mulacc ~truncated:truncate_output dt (f a) (f b) (f c))
          | _ -> python_alu op args
        in
        let v =
          match v with
          | v when is_nan v && not (List.exists is_nan args) -> `Float Dtype.nan
          | v -> v
        in
        truncate v

(* A cast in a symbolic integer converts without truncating. *)
let sym_cast dt x : Dtype.value =
  if Dtype.is_float dt then `Float (Value.to_float x)
  else if Dtype.equal dt Dtype.Bool then `Bool (Value.to_bool x)
  else `Int (Value.to_z x)

let sym_alu op dt xs =
  as_value
    (exec_alu ~truncate_output:false op dt
       (List.map (fun x -> (x :> Dtype.const)) xs))

let sym_infer (s : sint) vars =
  match s with
  | Int n -> n
  | Sym u ->
      let s = simplify u in
      let cache = Tbl.create 16 in
      let get n = Tbl.find cache n in
      let eval n : Dtype.value =
        match n.op with
        | Op.Const -> as_value (value n)
        | Op.Param when addrspace n = Some Dtype.Alu || is_variable n -> (
            let name = expr n in
            match List.assoc_opt name vars with
            | Some x -> `Int (Bigint.of_int x)
            | None -> invalid_argf "the variable %s has no value" name)
        | Op.Cast -> sym_cast n.dtype (get (first n.op n.src))
        | Op.Bitcast ->
            Dtype.bitcast (first n.op n.src).dtype n.dtype
              (get (first n.op n.src))
        | op when Op.Set.mem op Op.Set.alu ->
            sym_alu op n.dtype (List.map get n.src)
        | op -> invalid_argf "%s cannot be evaluated" (Op.name op)
      in
      Value.to_int (topovisit s eval cache)

(* The integer operations [sym_compile] computes on [int]s. Each raises
   [Inexact] where the result may not be an [int], and the operation is then
   computed exactly. *)

exception Inexact

(* Whether [x] times a value of the same bound fits an [int]. *)
let small x = x > -0x4000_0000 && x < 0x4000_0000

let int_add x y =
  let s = x + y in
  if x >= 0 = (y >= 0) && s >= 0 <> (x >= 0) then raise Inexact else s

let int_mul x y = if small x && small y then x * y else raise Inexact

(* A division by zero is zero, and its remainder the dividend. *)
let int_quotient ~toward_zero x y =
  if y = 0 then 0
  else if x = min_int then raise Inexact
  else
    let q = x / y in
    if toward_zero || x mod y = 0 || x < 0 = (y < 0) then q else q - 1

let int_remainder ~toward_zero x y =
  int_add x (-int_mul (int_quotient ~toward_zero x y) y)

let int_binary = function
  | Op.Add -> Some int_add
  | Op.Sub ->
      Some (fun x y -> int_add x (if y = min_int then raise Inexact else -y))
  | Op.Mul -> Some int_mul
  | Op.Max -> Some Int.max
  | Op.Cdiv -> Some (int_quotient ~toward_zero:true)
  | Op.Cmod -> Some (int_remainder ~toward_zero:true)
  | Op.Floordiv -> Some (int_quotient ~toward_zero:false)
  | Op.Floormod -> Some (int_remainder ~toward_zero:false)
  | _ -> None

let sym_compile (s : sint) var =
  match s with
  | Int n -> fun _ -> n
  | Sym u ->
      let s = simplify u in
      let memo tbl f n =
        match Tbl.find_opt tbl n with
        | Some g -> g
        | None ->
            let g = f n in
            Tbl.replace tbl n g;
            g
      in
      (* Each node's value, as [sym_infer] computes it. *)
      let values = Tbl.create 16 in
      let rec compute n = memo values exact n
      and exact n =
        match n.op with
        | Op.Const ->
            let v = as_value (value n) in
            fun _ -> v
        | Op.Param when addrspace n = Some Dtype.Alu || is_variable n ->
            let read = var n in
            fun env -> `Int (Bigint.of_int (read env))
        | Op.Cast ->
            let x = compute (first n.op n.src) in
            fun env -> sym_cast n.dtype (x env)
        | Op.Bitcast ->
            let src = first n.op n.src in
            let x = compute src in
            fun env -> Dtype.bitcast src.dtype n.dtype (x env)
        | op when Op.Set.mem op Op.Set.alu ->
            let xs = List.map compute n.src in
            fun env -> sym_alu op n.dtype (List.map (fun x -> x env) xs)
        | op -> fun _ -> invalid_argf "%s cannot be evaluated" (Op.name op)
      in
      (* An integer node of integer sources, computed on [int]s. *)
      let ints = Tbl.create 16 in
      let rec int n = memo ints native n
      and native n =
        if not (Dtype.is_int n.dtype) then None
        else
          match (n.op, n.src) with
          | Op.Const, _ -> (
              match as_value (value n) with
              | `Int z when Bigint.fits_int z ->
                  let c = Bigint.to_int z in
                  Some (fun _ -> c)
              | _ -> None)
          | Op.Param, _ when addrspace n = Some Dtype.Alu || is_variable n ->
              Some (var n)
          | Op.Neg, [ x ] ->
              Option.map
                (fun x env ->
                  let v = x env in
                  if v = min_int then raise Inexact else -v)
                (int x)
          | o, [ x; y ] -> (
              match (int_binary o, int x, int y) with
              | Some f, Some x, Some y -> Some (fun env -> f (x env) (y env))
              | _ -> None)
          | _ -> None
      in
      let exact = compute s in
      let to_int env = Value.to_int (exact env) in
      match int s with
      | None -> to_int
      | Some f -> fun env -> ( try f env with Inexact -> to_int env)

(* Patterns *)

module Upat = struct
  type node = t

  type t = {
    ops : Op.t list option;
    dtypes : Dtype.t list option;
    p_arg : arg option;
    p_name : string option;
    tags : Tag.t list option;
    in_src : in_src option;
    is_any : bool;
    p_src : srcs option;
    strict_length : bool;
    required_len : int;
    custom_early_reject : Op.t list option;
    early_reject : Op.t list;
    cached : bool;
        (* Variables are memoised by their arguments: equal variables are one
           pattern, so their operands in any order are one ordering. *)
  }

  and in_src = Fixed of t list | Perm of t list | Each of t
  and srcs = Tuples of t list list | Repeat of t

  let rec permutations = function
    | [] -> [ [] ]
    | l ->
        List.concat
          (List.mapi
             (fun i x ->
               List.map
                 (fun p -> x :: p)
                 (permutations (List.filteri (fun j _ -> j <> i) l)))
             l)

  let same p q =
    p == q
    || p.cached && q.cached
       && Option.equal (List.equal Op.equal) p.ops q.ops
       && Option.equal String.equal p.p_name q.p_name
       && Option.equal (List.equal Dtype.equal) p.dtypes q.dtypes
       && Option.equal equal_arg p.p_arg q.p_arg

  let make ?ops ?dtypes ?in_src ?(is_any = false) ?(allow_any_len = false) ?arg
      ?name ?tags ?early_reject ?(cached = false) () =
    let p_src =
      match in_src with
      | None -> None
      | Some (Fixed l) -> Some (Tuples [ l ])
      | Some (Perm l) ->
          Some
            (Tuples (if Helpers.all_same same l then [ l ] else permutations l))
      | Some (Each p) -> Some (Repeat p)
    in
    let strict_length, required_len =
      match in_src with
      | Some (Fixed l | Perm l) -> (not allow_any_len, List.length l)
      | Some (Each _) | None -> (false, 0)
    in
    let early =
      match early_reject with
      | Some l -> l
      | None ->
          let firsts =
            match in_src with
            | Some (Each p) -> [ p ]
            | None -> []
            | Some (Fixed l | Perm l) -> l
          in
          List.fold_left
            (fun acc p ->
              match p.ops with
              | Some [ o ] when not (List.mem o acc) -> acc @ [ o ]
              | _ -> acc)
            [] firsts
    in
    {
      ops;
      dtypes;
      p_arg = arg;
      p_name = name;
      tags;
      in_src;
      is_any;
      p_src;
      strict_length;
      required_len;
      custom_early_reject = early_reject;
      early_reject = early;
      cached;
    }

  let only ~src ~perm ~each =
    match (src, perm, each) with
    | None, None, None -> None
    | Some l, None, None -> Some (Fixed l)
    | None, Some l, None -> Some (Perm l)
    | None, None, Some p -> Some (Each p)
    | _ -> invalid_arg "a pattern takes one of src, perm and each"

  let v ?op ?dtype ?src ?perm ?each ?allow_any_len ?arg ?name ?tag ?early_reject
      () =
    make
      ?ops:(Option.map Op.Set.to_list op)
      ?dtypes:dtype ?in_src:(only ~src ~perm ~each) ?allow_any_len ?arg ?name
      ?tags:tag ?early_reject ()

  let op ?dtype ?src ?perm ?each ?allow_any_len ?arg ?name ?tag ?early_reject o
      =
    make ~ops:[ o ] ?dtypes:dtype ?in_src:(only ~src ~perm ~each) ?allow_any_len
      ?arg ?name ?tags:tag ?early_reject ()

  let wild = make ()
  let var ?dtype name = make ?dtypes:dtype ~name ~cached:true ()

  let cvar ?dtype ?arg name =
    make ~ops:[ Op.Const ] ?dtypes:dtype
      ?arg:(Option.map (fun c -> Const c) arg)
      ~name ~cached:true ()

  let const ?dtype c = make ~ops:[ Op.Const ] ?dtypes:dtype ~arg:(Const c) ()
  let any ps = make ~in_src:(Fixed ps) ~is_any:true ()

  let named name p =
    make ?ops:p.ops ?dtypes:p.dtypes ?in_src:p.in_src
      ~allow_any_len:(not p.strict_length) ?arg:p.p_arg ~name ?tags:p.tags
      ?early_reject:p.custom_early_reject ()

  let dtype p = match p.dtypes with Some (dt :: _) -> dt | _ -> Dtype.Void

  include Make_elementwise (struct
    type nonrec t = t

    let dtype = dtype

    let alu p o rest =
      let srcs = p :: rest in
      let dtypes =
        if o = Op.Cmplt || o = Op.Cmpne then Some [ Dtype.Bool ]
        else (List.nth srcs (List.length srcs - 1)).dtypes
      in
      make ~ops:[ o ] ?dtypes
        ~in_src:
          (if Op.Set.mem o Op.Set.commutative then Perm srcs else Fixed srcs)
        ()

    let cast p dt =
      match p.dtypes with
      | Some [ d ] when Dtype.equal d dt -> p
      | _ -> make ~ops:[ Op.Cast ] ~dtypes:[ dt ] ~in_src:(Fixed [ p ]) ()

    let literal c = make ~ops:[ Op.Const ] ~arg:(Const c) ~cached:true ()
    let literal_value _ = None
    let broadcasted x y = (x, y)
    let direct_floor = true
  end)

  let or_casted ?name p =
    any
      [
        (match name with None -> p | Some n -> named n p);
        make ~ops:[ Op.Cast ] ?name ~in_src:(Fixed [ p ]) ();
      ]

  let or_bitcasted ?name p =
    any
      [
        (match name with None -> p | Some n -> named n p);
        make ~ops:[ Op.Bitcast ] ?name ~in_src:(Fixed [ p ]) ();
      ]

  let or_after ?name p =
    any
      [
        (match name with None -> p | Some n -> named n p);
        make ~ops:[ Op.After ] ?name ~in_src:(Fixed [ p ]) ~allow_any_len:true
          ();
      ]

  let f ?dtype ?name ?arg ?allow_any_len p o =
    make ~ops:[ o ] ?dtypes:dtype ?name ?arg ?allow_any_len
      ~in_src:(Fixed [ p ]) ()

  let sink ?name ps = make ~ops:[ Op.Sink ] ?name ~in_src:(Fixed ps) ()

  let index ?name ?allow_any_len p rest =
    make ~ops:[ Op.Index ] ?name ?allow_any_len ~in_src:(Fixed (p :: rest)) ()

  let load ?name ?allow_any_len p rest =
    make ~ops:[ Op.Load ] ?name ?allow_any_len ~in_src:(Fixed (p :: rest)) ()

  let store ?name ?allow_any_len p rest =
    make ~ops:[ Op.Store ] ?name ?allow_any_len ~in_src:(Fixed (p :: rest)) ()

  let reduce ?name ?op ?allow_any_len p rest =
    make ~ops:[ Op.Reduce ] ?dtypes:p.dtypes ?name ?allow_any_len
      ?arg:(Option.map (fun op -> Reduce { op; num_axes = 0 }) op)
      ~in_src:(Fixed (p :: rest))
      ()

  let broadcast ?name p =
    make ~ops:[ Op.Stack ] ?dtypes:p.dtypes ?name ~in_src:(Each p) ()

  let after ?name ?allow_any_len p rest =
    make ~ops:[ Op.After ] ?dtypes:p.dtypes ?name ?allow_any_len
      ~in_src:(Fixed (p :: rest))
      ()

  let end_ ?name ?allow_any_len p rest =
    make ~ops:[ Op.End ] ?name ?allow_any_len ~in_src:(Fixed (p :: rest)) ()

  let backedge ?name p ~loop ~cond =
    make ~ops:[ Op.Backedge ] ?name ~in_src:(Fixed [ p; loop; cond ]) ()

  let bitcast ?dtype p =
    make ~ops:[ Op.Bitcast ]
      ?dtypes:(Option.map (fun d -> [ d ]) dtype)
      ~in_src:(Fixed [ p ]) ()

  (* Arguments compare as numbers do. *)
  let arg_matches (a : arg) (b : arg) =
    match (a, b) with
    | Const (`Float x), Const (`Float y) when Float.is_nan x && Float.is_nan y
      ->
        true
    | Const (#Dtype.value as x), Const (#Dtype.value as y) -> Value.( = ) x y
    | _ -> equal_arg a b

  let rec zip l0 l1 =
    match (l0, l1) with x :: r0, y :: r1 -> (x, y) :: zip r0 r1 | _ -> []

  let rec matches p (u : node) store =
    if p.is_any then
      match p.p_src with
      | Some (Tuples [ alternatives ]) ->
          List.concat_map (fun alt -> matches alt u store) alternatives
      | _ -> []
    else if
      match p.ops with Some ops -> not (List.mem u.op ops) | None -> false
    then []
    else
      let store =
        match p.p_name with
        | None -> Some store
        | Some n -> (
            match List.assoc_opt n store with
            | Some bound -> if bound == u then Some store else None
            | None -> Some ((n, u) :: store))
      in
      match store with
      | None -> []
      | Some store -> (
          let n_src = List.length u.src in
          if
            (match p.dtypes with
              | Some dts -> not (List.exists (Dtype.equal u.dtype) dts)
              | None -> false)
            || (match p.p_arg with
              | Some a -> not (arg_matches a u.arg)
              | None -> false)
            || (match p.tags with
              | Some tags -> (
                  match u.tag with
                  | Some g -> not (List.exists (Tag.equal g) tags)
                  | None -> true)
              | None -> false)
            || n_src < p.required_len
            || (p.strict_length && n_src <> p.required_len)
          then []
          else
            match p.p_src with
            | None -> [ store ]
            | Some srcs ->
                let tuples =
                  match srcs with
                  | Tuples ts -> ts
                  | Repeat q -> [ List.map (fun _ -> q) u.src ]
                in
                List.concat_map
                  (fun tuple ->
                    List.fold_left
                      (fun stores (child, pat) ->
                        List.concat_map (fun s -> matches pat child s) stores)
                      [ store ] (zip u.src tuple))
                  tuples)

  let match_ p u = List.map List.rev (matches p u [])

  include O
end

module Pattern_matcher = struct
  type ('ctx, 'r) rule = {
    pattern : Upat.t;
    fn : 'ctx -> (string -> t) -> 'r option;
  }

  let rule pattern f = { pattern; fn = (fun _ m -> f m) }
  let rule_ctx pattern fn = { pattern; fn }

  (* A matcher is a list of parts, each with its rules indexed by operation and
     its own way of declining, so that appending shares the parts. *)
  type ('ctx, 'r) part = {
    by_op : ('ctx, 'r) rule list array;
    declines : 'r -> t -> bool;
  }

  (* A matcher builds its parts on its first rewrite, so that a program that
     never rewrites keeps no rules in the heap, which every major collection
     marks. Domains that race to the first rewrite each build the parts, and
     all keep the first built. *)
  type ('ctx, 'r) t = {
    build : unit -> ('ctx, 'r) part list;
    parts : ('ctx, 'r) part list option Atomic.t;
  }

  let make build = { build; parts = Atomic.make None }

  let parts m =
    match Atomic.get m.parts with
    | Some p -> p
    | None ->
        let p = m.build () in
        if Atomic.compare_and_set m.parts None (Some p) then p
        else Option.get (Atomic.get m.parts)

  let op_count = List.length (Op.Set.to_list Op.Set.all)

  let part ~declines rules =
    let by_op = Array.make op_count [] in
    List.iter
      (fun r ->
        match r.pattern.ops with
        | None -> invalid_arg "a rule's pattern needs an operation"
        | Some ops ->
            List.iter
              (fun o -> by_op.(Op.to_int o) <- r :: by_op.(Op.to_int o))
              ops)
      rules;
    { by_op = Array.map List.rev by_op; declines }

  let v rules = make (fun () -> [ part ~declines:( == ) (rules ()) ])
  let fold rules = make (fun () -> [ part ~declines:(fun _ _ -> false) (rules ()) ])
  let append m0 m1 = make (fun () -> parts m0 @ parts m1)
  let concat ms = make (fun () -> List.concat_map parts ms)

  let with_ctx m =
    let ignore_ctx r = { r with fn = (fun _ m -> r.fn () m) } in
    make (fun () ->
        List.map
          (fun p -> { p with by_op = Array.map (List.map ignore_ctx) p.by_op })
          (parts m))

  let src_ops u =
    match u.src_ops_memo with
    | Some s -> s
    | None ->
        let s = Op.Set.of_list (List.map (fun s -> s.op) u.src) in
        u.src_ops_memo <- Some s;
        s

  let lookup store name =
    match List.assoc_opt name store with
    | Some u -> u
    | None -> invalid_argf "the pattern names no %s" name

  let rewrite m ctx u =
    let applies r =
      List.for_all (fun o -> Op.Set.mem o (src_ops u)) r.pattern.early_reject
    in
    let rec first_rule p = function
      | [] -> None
      | r :: rest -> (
          if not (applies r) then first_rule p rest
          else
            match
              List.find_map
                (fun store -> r.fn ctx (lookup store))
                (Upat.matches r.pattern u [])
            with
            | Some x when not (p.declines x u) -> Some x
            | _ -> first_rule p rest)
    in
    List.find_map (fun p -> first_rule p p.by_op.(Op.to_int u.op)) (parts m)
end

(* Rewriting *)

exception Bottom_up_gate

let rewrite_stack_limit =
  Helpers.Context_var.int ~reach:Process "REWRITE_STACK_LIMIT" 250000
let src_without_body u = if u.op = Op.Call then drop 1 u.src else u.src

let graph_rewrite ?(bottom_up = false) ?bpm ?(walk = false)
    ?(enter_calls = false) ~ctx root pm =
  let exception Gate of t in
  let pm, bpm =
    match (bottom_up, bpm) with
    | true, Some _ -> invalid_arg "a bottom-up rewrite takes no second matcher"
    | true, None -> (None, Some pm)
    | false, bpm -> (Some pm, bpm)
  in
  let replaced = Tbl.create 64 in
  let bpm_cache = Tbl.create (if Option.is_some bpm then 64 else 1) in
  let bpm_rewrite x =
    match Tbl.find_opt bpm_cache x with
    | Some r -> r
    | None ->
        let r = Pattern_matcher.rewrite (Option.get bpm) ctx x in
        Tbl.replace bpm_cache x r;
        r
  in
  let pm_rewrite x =
    match pm with None -> None | Some pm -> Pattern_matcher.rewrite pm ctx x
  in
  let body_skipped n = n.op = Op.Call && not enter_calls in
  let rest_of n = if body_skipped n then drop 1 n.src else n.src in
  let rebuild n =
    let src =
      if body_skipped n then
        List.hd n.src
        :: List.map
             (fun x -> Option.value (Tbl.find_opt replaced x) ~default:x)
             (drop 1 n.src)
      else
        List.map
          (fun x -> Option.value (Tbl.find_opt replaced x) ~default:x)
          n.src
    in
    if List.equal ( == ) src n.src then n else v n.op ~src ~arg:n.arg ?tag:n.tag
  in
  if walk then begin
    (* A single pass: a node rewritten on the way down is not entered, and one
       rewritten on the way up is not rewritten again. *)
    let stack = Stack.create () in
    Stack.push (root, false) stack;
    while not (Stack.is_empty stack) do
      let n, processed = Stack.pop stack in
      if not (Tbl.mem replaced n) then
        if not processed then
          begin match if Option.is_some bpm then bpm_rewrite n else None with
          | Some r -> Tbl.replace replaced n r
          | None ->
              Stack.push (n, true) stack;
              List.iter
                (fun x ->
                  if not (Tbl.mem replaced x) then Stack.push (x, false) stack)
                (List.rev (rest_of n))
          end
        else
          let new_n = rebuild n in
          let new_n =
            match pm_rewrite new_n with Some r -> r | None -> new_n
          in
          Tbl.replace replaced n new_n
    done;
    Option.value (Tbl.find_opt replaced root) ~default:root
  end
  else begin
    (* Each entry is a node, a stage and the node it currently stands for. Down:
       rewrite bottom-up to a fixed point and push the sources. Rebuild: once
       the sources are done, rebuild on them and rewrite top-down. Link: the
       node becomes whatever its rewrite became. An entry whose dependency is
       not done waits for it instead of spinning. *)
    let limit = Helpers.Context_var.value rewrite_stack_limit in
    let stack = Stack.create () in
    let on_stack = Tbl.create 64 and waitlist = Tbl.create 16 in
    let wait dep entry =
      Tbl.replace waitlist dep
        (entry :: Option.value (Tbl.find_opt waitlist dep) ~default:[])
    in
    let finish n r =
      Tbl.replace replaced n r;
      match Tbl.find_opt waitlist n with
      | Some waiting ->
          Tbl.remove waitlist n;
          List.iter (fun e -> Stack.push e stack) (List.rev waiting)
      | None -> ()
    in
    Stack.push (root, `Down, root) stack;
    Tbl.replace on_stack root ();
    while not (Stack.is_empty stack) do
      if Stack.length stack > limit then
        invalid_arg
          "graph_rewrite does not terminate: its work list is too long";
      let n, stage, new_n = Stack.pop stack in
      if not (Tbl.mem replaced n) then
        match stage with
        | `Down -> (
            let fixed_point =
              match bpm with
              | None -> Ok n
              | Some _ -> (
                  let seen = Tbl.create 4 in
                  let rec loop current =
                    if Tbl.mem seen current then
                      invalid_arg
                        "graph_rewrite does not terminate: a bottom-up rewrite \
                         cycles";
                    Tbl.replace seen current ();
                    match bpm_rewrite current with
                    | Some next -> loop next
                    | None -> current
                    | exception Bottom_up_gate -> raise_notrace (Gate current)
                  in
                  try Ok (loop n) with Gate current -> Error current)
            in
            match fixed_point with
            | Error gated -> finish n gated
            | Ok new_n ->
                Stack.push (n, `Rebuild, new_n) stack;
                List.iter
                  (fun x ->
                    if not (Tbl.mem on_stack x) then begin
                      Stack.push (x, `Down, x) stack;
                      Tbl.replace on_stack x ()
                    end)
                  (List.rev (rest_of new_n)))
        | `Rebuild -> (
            match
              List.find_opt (fun x -> not (Tbl.mem replaced x)) (rest_of new_n)
            with
            | Some pending -> wait pending (n, `Rebuild, new_n)
            | None -> (
                let rebuilt = rebuild new_n in
                let next =
                  if rebuilt == new_n then pm_rewrite new_n else Some rebuilt
                in
                match next with
                | None -> finish n new_n
                | Some next ->
                    Stack.push (n, `Link, next) stack;
                    Stack.push (next, `Down, next) stack))
        | `Link -> (
            match Tbl.find_opt replaced new_n with
            | Some r -> finish n r
            | None -> wait new_n (n, `Link, new_n))
    done;
    match Tbl.find_opt replaced root with
    | Some r -> r
    | None ->
        invalid_arg
          "graph_rewrite does not terminate: a node's rewrite depends on itself"
  end

let pm_substitute : (t Tbl.t, t) Pattern_matcher.t =
  Pattern_matcher.(
    v
      (fun () -> [
        rule_ctx (Upat.v ~op:Op.Set.all ~name:"x" ()) (fun subs m ->
            Tbl.find_opt subs (m "x"));
      ]))

let substitute ?extra_pm ?(walk = false) ?(enter_calls = false) u subs =
  let tbl = Tbl.create 16 in
  List.iter (fun (k, x) -> Tbl.replace tbl k x) subs;
  Tbl.filter_map_inplace (fun k x -> if k == x then None else Some x) tbl;
  if Tbl.length tbl = 0 then u
  else
    let pm =
      match extra_pm with
      | Some extra -> Pattern_matcher.append extra pm_substitute
      | None -> pm_substitute
    in
    graph_rewrite ~bottom_up:true ~walk ~enter_calls ~ctx:tbl u pm

let remove_all_tags =
  Pattern_matcher.(
    v
      (fun () -> [
        rule (Upat.v ~op:Op.Set.all ~name:"x" ()) (fun m ->
            let x = m "x" in
            if Option.is_none x.tag then None else Some (replace ~tag:None x));
      ]))

let pm_drop_after =
  Pattern_matcher.(
    v (fun () -> [ rule (Upat.op Op.After ~name:"a") (fun m -> Some (nth (m "a") 0)) ]))

let resolve_returned_after r effects =
  let target = unsharded_base r in
  match
    List.filter
      (fun st -> st.op = Op.Store && unsharded_base (nth st 0) == target)
      effects.src
  with
  | [ st ] -> Some (if target.op = Op.Param then after r [ st ] else nth st 1)
  | _ -> None

let gate_kernel_sink u =
  match (u.op, u.arg) with
  | Op.Linear, _ | Op.Sink, Kernel _ -> false
  | _ -> true

(* Graph construction that rewrites *)

let contract u rs =
  List.iter
    (fun r ->
      if not (Axis_type.equal (axis_type r) Axis_type.Upcast) then
        invalid_arg "contracted ranges must be upcast")
    rs;
  let rec product = function
    | [] -> [ [] ]
    | r :: rest ->
        let n = Value.to_int (vmax r) + 1 in
        List.concat_map
          (fun i -> List.map (fun t -> i :: t) (product rest))
          (List.init n Fun.id)
  in
  stack
    (List.map
       (fun idx ->
         substitute u
           (List.map2 (fun r i -> (r, const_like r (`Int (Bigint.of_int i)))) rs idx))
       (product rs))

let bind var x =
  if (not (is_variable var)) || is_bound_var var then
    invalid_argf "only an unbound variable binds, not %s" (Op.name var.op);
  let c = const (x :> Dtype.const) in
  if not (Value.( <= ) (vmin var) (vmin c) && Value.( <= ) (vmax c) (vmax var))
  then
    invalid_argf "%s is out of [%s, %s]" (repr_const x)
      (repr_const (vmin var))
      (repr_const (vmax var));
  let p = param_arg_of var in
  let multiple = Option.value p.multiple_of ~default:1 in
  if Option.is_none (divides c (Bigint.of_int multiple)) then
    invalid_argf "%s is not a multiple of %d" (repr_const x) multiple;
  replace var ~arg:(Param { p with bound = Some x })

let unbound var =
  if not (is_variable var) then
    invalid_argf "%s is not a variable" (Op.name var.op);
  replace var ~arg:(Param { (param_arg_of var) with bound = None }) ~tag:None

let unbind var =
  match (is_bound_var var, var.arg) with
  | true, Param { bound = Some x; _ } -> (unbound var, x)
  | _ -> invalid_arg "only a bound variable unbinds"

let unbind_all u =
  let bound =
    List.filter is_bound_var (Nodes.to_list (backward_slice_with_self u))
  in
  let pairs = List.map (fun x -> (x, unbound x)) bound in
  ( substitute ~walk:true u pairs,
    List.map (fun (x, var) -> (var, snd (unbind x))) pairs )

let variables u =
  let found =
    dedup_nodes
      (List.filter_map
         (fun x ->
           if x.op = Op.Param && addrspace x = Some Dtype.Alu then
             Some (if is_variable x then unbound x else x)
           else if
             x.op = Op.Range && Axis_type.equal (axis_type x) Axis_type.Device
           then
             Some (variable ~dtype:x.dtype "_device_num" (`Int Bigint.zero) (vmax x))
           else None)
         (Nodes.to_list (backward_slice_with_self u)))
  in
  let key x =
    let p = param_arg_of x in
    (Option.value p.name ~default:"", p.slot)
  in
  List.stable_sort (fun a b -> Stdlib.compare (key a) (key b)) found

(* Calls *)

let opaque_call_bodies =
  Op.Set.of_list Op.[ Sink; Program; Linear; Store; Custom_function ]

let custom_function name args =
  v Op.Custom_function ~src:args ~arg:(String name)

let call ?(ret_dtype = Dtype.Void) ?name ?(precompile = false) ?aux body args =
  if not (Op.Set.mem body.op opaque_call_bodies) then
    invalid_argf "a %s cannot be called" (Op.name body.op);
  if
    not
      (List.for_all
         (fun r -> Axis_type.equal (axis_type r) Axis_type.Device)
         (Nodes.to_list (ranges body)))
  then invalid_arg "ranges leak out of the called body";
  v Op.Call ~src:(body :: args)
    ~arg:(Call { name; precompile; aux; dtype = ret_dtype })

let store_call dst src =
  call (store (param_like dst 0) (param_like src 1)) [ dst; src ]

let is_inline_call u =
  u.op = Op.Call && (body u).op = Op.Sink
  && (match (body u).arg with No_arg -> true | _ -> false)
  && match u.arg with Call c -> not c.precompile | _ -> false

let unbound_output x =
  let b = unsharded_base x in
  b.op = Op.Alloc && not (param_arg_of b).bind_on_realize

let has_unbound_outputs c =
  c.op = Op.Call && List.exists unbound_output (drop 1 c.src)

let unbound_outputs c =
  List.filter_map
    (fun x -> if unbound_output x then Some (after x [ c ]) else None)
    (drop 1 c.src)

let pm_resolve_params : (t option array, t) Pattern_matcher.t =
  Pattern_matcher.(
    v
      (fun () -> [
        rule_ctx (Upat.op Op.Param ~name:"p") (fun params m ->
            let slot = (param_arg_of (m "p")).slot in
            if slot >= 0 then params.(slot) else None);
      ]))

let call_with_outputs ?name ?(precompile = false) ?aux ?output_pos values args =
  let n = List.length args + List.length values in
  let default_dev = List.find_map device (values @ args) in
  let pos =
    match output_pos with
    | None -> List.init (List.length values) (fun i -> List.length args + i)
    | Some p -> p
  in
  if
    List.length pos <> List.length values
    || List.length (List.sort_uniq Int.compare pos) <> List.length pos
  then invalid_arg "output_pos needs one distinct position per output";
  let rec ascending = function
    | a :: (b :: _ as r) -> a < b && ascending r
    | _ -> true
  in
  if not (ascending pos) then
    invalid_arg "output_pos must be strictly ascending";
  if not (List.for_all (fun p -> p >= 0 && p < n) pos) then
    invalid_arg "output_pos must be within the argument list";
  let params = Array.make n None in
  let remaining = ref args in
  for i = 0 to n - 1 do
    if not (List.mem i pos) then
      match !remaining with
      | a :: rest ->
          params.(i) <- Some a;
          remaining := rest
      | [] -> ()
  done;
  let mint o p =
    let dev = match device o with Some d -> Some d | None -> default_dev in
    let axis = match device o with Some (Multi _) -> axis o | _ -> None in
    let buf = alloc (shard_shape o) o.dtype ?device:dev ?axis in
    let resolved =
      List.map
        (function
          | Int k -> Int k
          | Sym s ->
              Sym (graph_rewrite ~walk:true ~ctx:params s pm_resolve_params))
        (shard_shape o)
    in
    ( alloc resolved o.dtype ~slot:(param_arg_of (buf_uop buf)).slot ?device:dev
        ?axis,
      param_like buf p )
  in
  let outputs = List.map2 mint values pos in
  let body = sink (List.map2 (fun x (_, p) -> store p x) values outputs) in
  let inputs =
    ref (List.map (fun x -> if precompile then contiguous x else x) args)
  in
  let call_args =
    List.init n (fun i ->
        match List.find_index (Int.equal i) pos with
        | Some k -> fst (List.nth outputs k)
        | None -> (
            match !inputs with
            | x :: rest ->
                inputs := rest;
                x
            | [] -> invalid_arg "too few arguments"))
  in
  let c = call body call_args ?name ~precompile ?aux in
  List.map (fun (r, _) -> after r [ c ]) outputs

let call_with_output ?name ?precompile value args =
  List.hd (call_with_outputs ?name ?precompile [ value ] args)

let custom_kernel args f =
  let placeholders = List.mapi (fun i s -> placeholder_like s i) args in
  let kernel = call (f placeholders) args in
  List.map (fun s -> after s [ kernel ]) args

(* Programs *)

let kernel_info ?(name = "test") ?(applied_opts = []) ?opts_to_apply ?estimates
    ?(beam = 0) ?split () =
  { name; applied_opts; opts_to_apply; estimates; beam; split }

let function_name (k : kernel_info) = Helpers.to_function_name k.name

let program_info_of_sink
    ?(target =
      Helpers.Target.
        { device = ""; renderer = ""; arch = ""; interface = ""; indices = "" })
    sink =
  let vars = ref [] and globals = ref [] and outs = ref [] and ins = ref [] in
  let global_size = Array.make 3 (Int 1)
  and local_size = Array.make 3 (Int 1) in
  (match sink.arg with
  | Kernel { split = Some s; _ } -> global_size.(0) <- s.iterations
  | _ -> ());
  List.iter
    (fun u ->
      if u.op = Op.Param then
        if addrspace u = Some Dtype.Alu then vars := u :: !vars
        else globals := (param_arg_of u).slot :: !globals;
      if u.op = Op.Store || u.op = Op.Load then begin
        let s0 = nth u 0 in
        let idx =
          if s0.op = Op.Index || s0.op = Op.Shrink then Some s0
          else if s0.op = Op.Cast && (nth s0 0).op = Op.Index then
            Some (nth s0 0)
          else None
        in
        match idx with
        | Some idx ->
            let buf = buf_uop (nth idx 0) in
            if buf.op = Op.Param then
              let slot = (param_arg_of buf).slot in
              if u.op = Op.Store then outs := slot :: !outs
              else ins := slot :: !ins
        | None -> ()
      end;
      if u.op = Op.Special then
        match u.arg with
        | String name ->
            let axis =
              Char.code name.[String.length name - 1] - Char.code '0'
            in
            let sizes = if name.[0] = 'l' then local_size else global_size in
            sizes.(axis) <- ssimplify (nth u 0)
        | _ -> invalid_arg "a hardware index needs a name")
    (toposort sink);
  let sorted l = List.sort_uniq Int.compare l in
  let outs, ins =
    if List.is_empty !outs && List.is_empty !ins then (!globals, !globals)
    else (!outs, !ins)
  in
  let vars =
    List.stable_sort
      (fun a b -> Int.compare (param_arg_of a).slot (param_arg_of b).slot)
      (dedup_nodes (List.rev !vars))
  in
  {
    global_size = Array.to_list global_size;
    local_size = Array.to_list local_size;
    vars;
    globals = sorted !globals;
    outs = sorted outs;
    ins = sorted ins;
    target;
  }

let launch_dims (p : program_info) vars =
  ( List.map (fun s -> sym_infer s vars) p.global_size,
    List.map (fun s -> sym_infer s vars) p.local_size )

let vals (p : program_info) vars =
  List.map
    (fun x ->
      let name = expr x in
      match List.assoc_opt name vars with
      | Some n -> n
      | None -> invalid_argf "the variable %s has no value" name)
    p.vars

(* Late bindings *)

module Private = struct
  let set_once slot what f =
    if not (Atomic.compare_and_set slot None (Some f)) then
      invalid_argf "the %s are set already" what

  let set_symbolic pm =
    set_once simplify_hook "symbolic rules" (fun u ->
        graph_rewrite ~ctx:() u pm)

  let set_spec pm =
    set_once construction_check "specification rules" (fun u ->
        if Helpers.Context_var.value Helpers.spec > 2 then ignore (shape_opt u);
        match
          Helpers.context
            [ B (Helpers.check_oob, false) ]
            (fun () -> Pattern_matcher.rewrite pm () u)
        with
        | Some true -> ()
        | verdict ->
            invalid_argf "the node breaks the specification (%s):\n%s"
              (match verdict with Some b -> repr_bool b | None -> "None")
              (repr u))
end
