(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* Lists *)

(* The list without its first [n] elements, as Python's slice: a short list
   gives what it has. *)
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
    memos : memos;
  }

  (* Properties computed on first use. Each is a function of the node, so
     domains racing to fill one write the same value; the fields are atomic so
     that a reader that sees a value sees it whole. They are a record of their
     own so that the probe a lookup builds ({!v}) shares one empty record. *)
  and memos = {
    mutable shape_memo : sint list option option; [@atomic]
    mutable ranges_memo : nodes option; [@atomic]
    mutable ended_ranges_memo : t list option; [@atomic]
    mutable min_max_memo : (Dtype.value * Dtype.value) option; [@atomic]
    mutable device_memo : device option option; [@atomic]
    mutable addrspace_memo : Dtype.addr_space option option; [@atomic]
    mutable backward_slice_memo : nodes option; [@atomic]
    mutable ops_reached_memo : Op.Set.t option; [@atomic]
    mutable axis_memo : int option option; [@atomic]
    mutable marg_memo : movement option; [@atomic]
    mutable key_memo : string option; [@atomic]
    mutable arg_repr_memo : string option; [@atomic]
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
    let tag = match u.tag with None -> 0 | Some g -> 1 + Tag.hash g in
    List.fold_left
      (fun h s -> (h * 31) + s.id)
      (Hashtbl.hash ((((Op.to_int u.op * 31) + hash_arg u.arg) * 31) + tag))
      u.src
end

module Table = Stdlib.Weak.Make (Interned)

(* The table is split in shards, each behind its own lock, so that domains
   building nodes rarely wait for each other. Each shard starts at the least
   size and grows with its nodes, so that a program that builds none keeps no
   table in the heap every major collection marks.

   A shard picks its buckets by the node's hash modulo their number, so the
   shard is picked by a mix of the hash. Picked by the hash's low bits, a
   shard's nodes would share those bits and fill only the buckets they select:
   at an even number of buckets that is at most half of them, and a weak table
   grows only when more than half of its buckets overflow, so the shard would
   stop growing and every lookup would scan a bucket of thousands of nodes. *)
let shards = Array.init 64 (fun _ -> (Table.create 0, Mutex.create ()))
let next_id = Atomic.make 0

let memos () =
  {
    shape_memo = None;
    ranges_memo = None;
    ended_ranges_memo = None;
    min_max_memo = None;
    device_memo = None;
    addrspace_memo = None;
    backward_slice_memo = None;
    ops_reached_memo = None;
    axis_memo = None;
    marg_memo = None;
    key_memo = None;
    arg_repr_memo = None;
  }

let node op src arg tag dtype id =
  { op; src; arg; tag; dtype; id; memos = memos () }

(* The memos of every probe, which no one reads or fills: a probe only looks its
   node up. *)
let probe_memos = memos ()

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
  let probe =
    { op; src; arg; tag; dtype = Dtype.Void; id = -1; memos = probe_memos }
  in
  let table, lock = shards.(Hashtbl.hash (Interned.hash probe) land 63) in
  Mutex.lock lock;
  match Table.find_opt table probe with
  | Some u ->
      Mutex.unlock lock;
      u
  | None -> (
      match dtype_of op src arg with
      | exception e ->
          let bt = Printexc.get_raw_backtrace () in
          Mutex.unlock lock;
          Printexc.raise_with_backtrace e bt
      | dtype ->
          let u = node op src arg tag dtype (Atomic.fetch_and_add next_id 1) in
          Table.add table u;
          Mutex.unlock lock;
          u)

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
  match u.memos.arg_repr_memo with
  | Some s -> s
  | None ->
      let s = repr_arg u.arg in
      u.memos.arg_repr_memo <- Some s;
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

type calls = Enter | Skip

(* A stack of entries, each a node, a stage and a second node, held in arrays
   that double as they fill: a walk pushes an entry per node it visits, and
   pushing allocates nothing. *)
module Work = struct
  type node = t

  type 's t = {
    mutable nodes : node array;
    mutable stages : 's array;
    mutable others : node array;
    mutable len : int;
  }

  let create u s =
    {
      nodes = Array.make 16 u;
      stages = Array.make 16 s;
      others = Array.make 16 u;
      len = 0;
    }

  let is_empty w = w.len = 0
  let length w = w.len

  let grow w a =
    let b = Array.make (2 * Array.length a) (Array.unsafe_get a 0) in
    Array.blit a 0 b 0 w.len;
    b

  let push w u s o =
    if w.len = Array.length w.nodes then begin
      w.nodes <- grow w w.nodes;
      w.stages <- grow w w.stages;
      w.others <- grow w w.others
    end;
    Array.unsafe_set w.nodes w.len u;
    Array.unsafe_set w.stages w.len s;
    Array.unsafe_set w.others w.len o;
    w.len <- w.len + 1

  (* The top entry's parts, and its removal. *)
  let node w = Array.unsafe_get w.nodes (w.len - 1)
  let stage w = Array.unsafe_get w.stages (w.len - 1)
  let other w = Array.unsafe_get w.others (w.len - 1)
  let drop w = w.len <- w.len - 1
end

(* Each node is pushed with a flag: unset, its sources are pushed after it; set,
   its sources are done and it is. A node pushed twice before it is done is
   finished at its first pop with the flag set. *)
let toposort ?gate ~calls root =
  let cache = Tbl.create 64 and order = ref [] in
  let work = Work.create root false in
  let rec push_srcs = function
    | [] -> ()
    | s :: rest ->
        push_srcs rest;
        Work.push work s false s
  in
  Work.push work root false root;
  while not (Work.is_empty work) do
    let node = Work.node work and visited = Work.stage work in
    Work.drop work;
    if not (Tbl.mem cache node) then
      if not visited then
        begin if match gate with None -> true | Some g -> g node then begin
          Work.push work node true node;
          push_srcs
            (if calls = Skip && node.op = Op.Call then drop 1 node.src
             else node.src)
        end
        end
      else begin
        Tbl.replace cache node ();
        order := node :: !order
      end
  done;
  List.rev !order

let topovisit root f cache =
  let work = Work.create root false in
  let rec push_srcs = function
    | [] -> ()
    | s :: rest ->
        push_srcs rest;
        Work.push work s false s
  in
  Work.push work root false root;
  while not (Work.is_empty work) do
    let node = Work.node work and visited = Work.stage work in
    Work.drop work;
    if not (Tbl.mem cache node) then
      if not visited then begin
        Work.push work node true node;
        push_srcs node.src
      end
      else Tbl.replace cache node (f node)
  done;
  Tbl.find cache root

(* A recursive property is filled bottom-up over the nodes that lack it, so a
   deep graph never recurses deeply. *)
let memoized ~calls ~get ~set ~compute u =
  match get u with
  | Some x -> x
  | None ->
      (* A new node is mostly built on nodes that have the property. *)
      let srcs =
        if calls = Skip && u.op = Op.Call then drop 1 u.src else u.src
      in
      if List.for_all (fun s -> Option.is_some (get s)) srcs then
        set u (compute u)
      else
        List.iter
          (fun n -> set n (compute n))
          (toposort ~calls ~gate:(fun n -> Option.is_none (get n)) u);
      Option.get (get u)

(* A slice outside call bodies is kept with its node; one that enters them is
   walked anew. *)
let backward_slice ~calls u =
  let walk () =
    Nodes.of_list (List.filter (fun n -> n != u) (toposort ~calls u))
  in
  match (calls, u.memos.backward_slice_memo) with
  | Enter, _ -> walk ()
  | Skip, Some s -> s
  | Skip, None ->
      let s = walk () in
      u.memos.backward_slice_memo <- Some s;
      s

let backward_slice_with_self ~calls u =
  Nodes.of_list (u :: Nodes.to_list (backward_slice ~calls u))

(* The operations of [u] and of the nodes it reaches outside call bodies, as a
   property of each node, so that asking costs no walk of the slice. A node
   whose set is one of its sources' shares that source's set. *)
let ops_reached u =
  memoized ~calls:Skip
    ~get:(fun n -> n.memos.ops_reached_memo)
    ~set:(fun n s -> n.memos.ops_reached_memo <- Some s)
    ~compute:(fun n ->
      let srcs = if n.op = Op.Call then drop 1 n.src else n.src in
      let sets = List.map (fun s -> Option.get s.memos.ops_reached_memo) srcs in
      let all = List.fold_left Op.Set.union (Op.Set.of_list [ n.op ]) sets in
      Option.value ~default:all (List.find_opt (Op.Set.equal all) sets))
    u

let op_in_backward_slice_with_self ~calls u ops =
  match calls with
  | Enter -> List.exists (fun n -> List.mem n.op ops) (toposort ~calls u)
  | Skip ->
      let reached = ops_reached u in
      List.exists (fun o -> Op.Set.mem o reached) ops

(* A node is built after its sources, so ids grow along every edge: the search
   for [x] never enters a node built before it, and stops once it meets [x]. *)
let reaches ~calls u x =
  let gate n =
    if n == x then raise_notrace Exit;
    n.id >= x.id
  in
  u == x
  || x.id < u.id
     && match toposort ~calls ~gate u with _ -> false | exception Exit -> true

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
  memoized ~calls:Enter
    ~get:(fun n -> n.memos.key_memo)
    ~set:(fun n k -> n.memos.key_memo <- Some k)
    ~compute:(fun n ->
      let b = Buffer.create 128 in
      Buffer.add_string b
        (repr_tuple
           [ Format.asprintf "%a" Op.pp n.op; repr_dtype n.dtype; arg_repr n ]);
      List.iter
        (fun s -> Buffer.add_string b (Option.get s.memos.key_memo))
        n.src;
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

(* A weak float that a committed float operation reads is rounded to that type,
   which keeps the order of values: its bounds rounded hold its values rounded,
   as a constant that rounds to 0 there. A bound that rounds to NaN, past a type
   without infinities, leaves the type's bounds. *)
let rounded dt ((lo, hi) as b : Dtype.value * Dtype.value) =
  match (lo, hi) with
  | `Float _, `Float _ -> (
      match (Dtype.truncate dt lo, Dtype.truncate dt hi) with
      | (`Float l as lo), (`Float h as hi)
        when not (Float.is_nan l || Float.is_nan h) ->
          (lo, hi)
      | _ -> unbounded dt)
  | _ -> b

(* Bounds read only the sources their rule needs, so they recurse rather than
   fill the whole graph below. A node's bounds, and those of an operand an
   operation commits to its type, are at their width. *)
let rec min_max u =
  match u.memos.min_max_memo with
  | Some b -> b
  | None ->
      let b = at_width u.dtype (compute_min_max u) in
      u.memos.min_max_memo <- Some b;
      b

and operand_bounds u s =
  let operands =
    if Op.Set.mem u.op Op.Set.comparison then promo_dtype u.src else u.dtype
  in
  if Dtype.equal s.dtype Dtype.Weak_float && List.mem operands Dtype.floats then
    rounded operands (min_max s)
  else at_width operands (min_max s)

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
      | Op.Bitcast, _ -> bitcast_bounds (src0 ()) dt
      | _ -> unbounded dt)

(* A bitcast between integer types of one width keeps the bits, so a value both
   types hold, from 0 to the lesser of their greatest values, is unchanged:
   reading an index of [0, n) as unsigned, as a bounds check does. *)
and bitcast_bounds x dt =
  let lo, hi = min_max x in
  let integer t = List.mem t Dtype.ints in
  if
    integer x.dtype && integer dt
    && Dtype.itemsize x.dtype = Dtype.itemsize dt
    && Value.(
         `Int Bigint.zero <= lo && hi <= min (Dtype.max x.dtype) (Dtype.max dt))
  then (lo, hi)
  else unbounded dt

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

(* Bounds of one value would make the storage's parameter that constant, in
   place of the storage. *)
let stored_bounds u dt =
  let lo = vmin u and hi = vmax u in
  if Value.(lo < hi && (Dtype.min dt < lo || hi < Dtype.max dt)) then
    Some (lo, hi)
  else None

let exact dt vs =
  (not (List.mem dt Dtype.ints))
  || List.for_all (fun v -> Value.(Dtype.min dt <= v && v <= Dtype.max dt)) vs

(* Shapes *)

let sint_to_uop ?(dtype = Dtype.Weak_int) = function
  | Int n -> int ~dtype n
  | Sym u -> cast u dtype

let to_max_shape shape =
  List.map (function Int n -> n | Sym u -> Value.to_int (vmax u)) shape

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
  match u.memos.ended_ranges_memo with
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
      u.memos.ended_ranges_memo <- Some l;
      l

let rec ranges u =
  memoized ~calls:Enter
    ~get:(fun n -> n.memos.ranges_memo)
    ~set:(fun n r -> n.memos.ranges_memo <- Some r)
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
  memoized ~calls:Enter
    ~get:(fun n -> n.memos.device_memo)
    ~set:(fun n d -> n.memos.device_memo <- Some d)
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
  memoized ~calls:Enter
    ~get:(fun n -> n.memos.addrspace_memo)
    ~set:(fun n a -> n.memos.addrspace_memo <- Some a)
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

(* Several devices *)

let sharding u =
  match (u.op, u.arg) with
  | Op.Unshard, Axes axes -> List.combine axes (drop 1 u.src)
  | _ -> []

let device_range_src = function
  | Some (Multi ds) ->
      [ range ~axis_type:Axis_type.Device (Int (List.length ds)) [ -1 ] ]
  | _ -> []

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

let set ?(ends = []) p x = after (first p.op p.src) [ end_ (store p x) ends ]

(* Variables *)

let variable ?(dtype = Dtype.Weak_int) ?(multiple_of = 1) name lo hi =
  v Op.Param
    ~arg:
      (Param
         (param_arg ~vmin_vmax:(lo, hi) ~multiple_of ~name
            ~addrspace:(Some Dtype.Alu) ~slot:(-1) dtype))

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

let pop_const ?(op = Op.Add) u : t * Dtype.const =
  match u.src with
  | [ x; c ] when Op.equal u.op op && c.op = Op.Const -> (x, value c)
  | _ -> (u, identity_element op u.dtype)

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
    unique : bool;  (* At most one naming of any node. *)
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
      unique =
        (not is_any)
        &&
        match p_src with
        | None -> true
        | Some (Tuples [ tuple ]) -> List.for_all (fun q -> q.unique) tuple
        | Some (Repeat q) -> q.unique
        | Some (Tuples _) -> false;
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

  (* The nodes a match names, the latest first. *)
  type store = Empty | Bound of string * node * store

  let rec mem_dtype dt = function
    | [] -> false
    | d :: rest -> Dtype.equal dt d || mem_dtype dt rest

  let rec mem_tag g = function
    | [] -> false
    | t :: rest -> Tag.equal g t || mem_tag g rest

  (* Whether [u] itself fits [p], sources aside. *)
  let fits p u =
    let n_src = List.length u.src in
    (match p.dtypes with Some dts -> mem_dtype u.dtype dts | None -> true)
    && (match p.p_arg with Some a -> arg_matches a u.arg | None -> true)
    && (match p.tags with
      | Some tags -> (
          match u.tag with Some g -> mem_tag g tags | None -> false)
      | None -> true)
    && n_src >= p.required_len
    && not (p.strict_length && n_src <> p.required_len)

  exception No_match

  (* [store] extended with what [p] names in [u]. Raises [No_match] if the name
     is bound to another node. *)
  let rec bound_in n u store = function
    | Empty -> Bound (n, u, store)
    | Bound (m, b, rest) ->
        if not (String.equal m n) then bound_in n u store rest
        else if b == u then store
        else raise_notrace No_match

  let named_in p u store =
    match p.p_name with None -> store | Some n -> bound_in n u store store

  (* A unique pattern names a node in at most one way, which [bind] finds
     without continuations: a match costs only the names it binds. *)
  let rec bind p (u : node) store =
    match p.ops with
    | Some ops when not (List.mem u.op ops) -> raise_notrace No_match
    | _ when not (fits p u) -> raise_notrace No_match
    | _ -> (
        let store = named_in p u store in
        match p.p_src with
        | None -> store
        | Some (Tuples [ tuple ]) -> bind_sources tuple u.src store
        | Some (Repeat q) -> bind_each q u.src store
        | Some (Tuples _) -> assert false)

  and bind_sources pats srcs store =
    match (pats, srcs) with
    | p :: pats, u :: srcs -> bind_sources pats srcs (bind p u store)
    | _ -> store

  and bind_each q srcs store =
    match srcs with
    | [] -> store
    | u :: srcs -> bind_each q srcs (bind q u store)

  (* Matching enumerates each naming of [u] by [p] in order, each extending
     [store], and is the first result [k] gives one: alternatives and orderings
     in turn, and each source's namings before the next source's. *)
  let rec first p (u : node) store k =
    if p.unique then
      match bind p u store with store -> k store | exception No_match -> None
    else if p.is_any then
      match p.p_src with
      | Some (Tuples [ alternatives ]) -> first_of alternatives u store k
      | _ -> None
    else
      match p.ops with
      | Some ops when not (List.mem u.op ops) -> None
      | _ when not (fits p u) -> None
      | _ -> (
          match named_in p u store with
          | exception No_match -> None
          | store -> (
              match p.p_src with
              | None -> k store
              | Some (Tuples tuples) -> first_tuple tuples u.src store k
              | Some (Repeat q) -> each q u.src store k))

  and first_of alternatives u store k =
    match alternatives with
    | [] -> None
    | p :: rest -> (
        match first p u store k with
        | Some _ as r -> r
        | None -> first_of rest u store k)

  and first_tuple tuples srcs store k =
    match tuples with
    | [] -> None
    | tuple :: rest -> (
        match sources tuple srcs store k with
        | Some _ as r -> r
        | None -> first_tuple rest srcs store k)

  and sources pats srcs store k =
    match (pats, srcs) with
    | [ p ], u :: _ -> first p u store k
    | p :: pats, u :: srcs when p.unique -> (
        match bind p u store with
        | store -> sources pats srcs store k
        | exception No_match -> None)
    | p :: pats, u :: srcs ->
        first p u store (fun store -> sources pats srcs store k)
    | _ -> k store

  and each q srcs store k =
    match srcs with
    | [] -> k store
    | [ u ] -> first q u store k
    | u :: srcs -> first q u store (fun store -> each q srcs store k)

  let rec naming acc = function
    | Empty -> acc
    | Bound (n, u, rest) -> naming ((n, u) :: acc) rest

  let match_ p u =
    let namings = ref [] in
    ignore
      (first p u Empty (fun store ->
           namings := naming [] store :: !namings;
           None));
    List.rev !namings

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

  let rec lookup store name =
    match store with
    | Upat.Empty -> invalid_argf "the pattern names no %s" name
    | Bound (n, u, rest) -> if String.equal n name then u else lookup rest name

  let rec has_op o = function
    | [] -> false
    | s :: srcs -> Op.equal s.op o || has_op o srcs

  (* Whether [u] has a source of each operation in [ops]. *)
  let rec has_sources ops u =
    match ops with
    | [] -> true
    | o :: ops -> has_op o u.src && has_sources ops u

  let applied r ctx u =
    if r.pattern.unique then
      match Upat.bind r.pattern u Empty with
      | store -> r.fn ctx (lookup store)
      | exception Upat.No_match -> None
    else Upat.first r.pattern u Empty (fun store -> r.fn ctx (lookup store))

  let rec first_rule p ctx u = function
    | [] -> None
    | r :: rest -> (
        if not (has_sources r.pattern.early_reject u) then
          first_rule p ctx u rest
        else
          match applied r ctx u with
          | Some x when not (p.declines x u) -> Some x
          | _ -> first_rule p ctx u rest)

  let rec first_part ctx u = function
    | [] -> None
    | p :: parts -> (
        match first_rule p ctx u p.by_op.(Op.to_int u.op) with
        | Some _ as r -> r
        | None -> first_part ctx u parts)

  let rewrite m ctx u = first_part ctx u (parts m)
end

(* Rewriting *)

exception Bottom_up_gate

let src_without_body u = if u.op = Op.Call then drop 1 u.src else u.src

type pass = Fixed_point | Once

type 'ctx rules =
  | After_sources of ('ctx, t) Pattern_matcher.t
  | Before_sources of ('ctx, t) Pattern_matcher.t
  | Around_sources of {
      before : ('ctx, t) Pattern_matcher.t;
      after : ('ctx, t) Pattern_matcher.t;
    }

let graph_rewrite ~calls ~pass ~ctx root rules =
  let exception Gate of t in
  let pm, bpm =
    match rules with
    | After_sources m -> (Some m, None)
    | Before_sources m -> (None, Some m)
    | Around_sources { before; after } -> (Some after, Some before)
  in
  let walk = pass = Once in
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
  let body_skipped n = n.op = Op.Call && calls = Skip in
  let rest_of n = if body_skipped n then drop 1 n.src else n.src in
  let now x = Option.value (Tbl.find_opt replaced x) ~default:x in
  let rec unchanged = function
    | [] -> true
    | x :: rest -> now x == x && unchanged rest
  in
  let rebuild n =
    if unchanged (rest_of n) then n
    else
      let src =
        if body_skipped n then List.hd n.src :: List.map now (drop 1 n.src)
        else List.map now n.src
      in
      v n.op ~src ~arg:n.arg ?tag:n.tag
  in
  if walk then begin
    (* A single pass: a node rewritten on the way down is not entered, and one
       rewritten on the way up is not rewritten again. *)
    let work = Work.create root false in
    let rec push_srcs = function
      | [] -> ()
      | x :: rest ->
          push_srcs rest;
          if not (Tbl.mem replaced x) then Work.push work x false x
    in
    Work.push work root false root;
    while not (Work.is_empty work) do
      let n = Work.node work and processed = Work.stage work in
      Work.drop work;
      if not (Tbl.mem replaced n) then
        if not processed then
          begin match if Option.is_some bpm then bpm_rewrite n else None with
          | Some r -> Tbl.replace replaced n r
          | None ->
              Work.push work n true n;
              push_srcs (rest_of n)
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
    let limit = Setting.value Setting.rewrite_stack_limit in
    let work = Work.create root `Down in
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
          List.iter
            (fun (n, stage, new_n) -> Work.push work n stage new_n)
            (List.rev waiting)
      | None -> ()
    in
    (* A node rewritten bottom-up to a fixed point; the set of the nodes it went
       through is made only when it moves. *)
    let fixed_point n =
      match bpm with
      | None -> n
      | Some _ -> (
          match bpm_rewrite n with
          | None -> n
          | exception Bottom_up_gate -> raise_notrace (Gate n)
          | Some next ->
              let seen = Tbl.create 4 in
              Tbl.replace seen n ();
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
              loop next)
    in
    (* The sources not yet on the stack, the first on top. *)
    let rec push_down = function
      | [] -> ()
      | x :: rest ->
          push_down rest;
          if not (Tbl.mem on_stack x) then begin
            Work.push work x `Down x;
            Tbl.replace on_stack x ()
          end
    in
    let rec pending = function
      | [] -> None
      | x :: rest -> if Tbl.mem replaced x then pending rest else Some x
    in
    Work.push work root `Down root;
    Tbl.replace on_stack root ();
    while not (Work.is_empty work) do
      if Work.length work > limit then
        invalid_arg
          "graph_rewrite does not terminate: its work list is too long";
      let n = Work.node work
      and stage = Work.stage work
      and new_n = Work.other work in
      Work.drop work;
      if not (Tbl.mem replaced n) then
        match stage with
        | `Down -> (
            match fixed_point n with
            | exception Gate gated -> finish n gated
            | new_n ->
                Work.push work n `Rebuild new_n;
                push_down (rest_of new_n))
        | `Rebuild -> (
            match pending (rest_of new_n) with
            | Some dep -> wait dep (n, `Rebuild, new_n)
            | None -> (
                let rebuilt = rebuild new_n in
                let next =
                  if rebuilt == new_n then pm_rewrite new_n else Some rebuilt
                in
                match next with
                | None -> finish n new_n
                | Some next ->
                    Work.push work n `Link next;
                    Work.push work next `Down next))
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

let substitute ?extra_pm ~calls ~pass u subs =
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
    graph_rewrite ~calls ~pass ~ctx:tbl u (Before_sources pm)

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

(* Programs *)

let kernel_info ?(name = "test") ?(applied_opts = []) ?opts_to_apply ?estimates
    ?(beam = 0) ?split () =
  { name; applied_opts; opts_to_apply; estimates; beam; split }

let function_name (k : kernel_info) = Helpers.to_function_name k.name

let vals (p : program_info) vars =
  List.map
    (fun x ->
      let name = expr x in
      match List.assoc_opt name vars with
      | Some n -> n
      | None -> invalid_argf "the variable %s has no value" name)
    p.vars

(* Memos *)

let shape_memo u = u.memos.shape_memo
let set_shape_memo u s = u.memos.shape_memo <- Some s
let movement_memo u = u.memos.marg_memo
let set_movement_memo u m = u.memos.marg_memo <- Some m
let axis_memo u = u.memos.axis_memo
let set_axis_memo u a = u.memos.axis_memo <- Some a
