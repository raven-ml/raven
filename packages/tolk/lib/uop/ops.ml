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
  type t = Global | Local | Warp | Weak | Loop | Upcast | Placeholder

  let all = [ Global; Local; Warp; Weak; Loop; Upcast; Placeholder ]

  let to_int = function
    | Global -> 1
    | Local -> 2
    | Warp -> 3
    | Weak -> 4
    | Loop -> 5
    | Upcast -> 6
    | Placeholder -> 7

  let equal (a0 : t) a1 = a0 = a1
  let compare a0 a1 = Int.compare (to_int a0) (to_int a1)

  let name = function
    | Global -> "GLOBAL"
    | Local -> "LOCAL"
    | Warp -> "WARP"
    | Weak -> "WEAK"
    | Loop -> "LOOP"
    | Upcast -> "UPCAST"
    | Placeholder -> "PLACEHOLDER"

  let of_string s =
    match List.find_opt (fun a -> String.equal (name a) s) all with
    | Some a -> Ok a
    | None -> Error (strf "unknown axis type %S" s)

  let no_placeholder () = invalid_arg "the placeholder axis type has no letter"

  let letter = function
    | Global -> "g"
    | Local -> "l"
    | Warp -> "w"
    | Weak | Loop -> "L"
    | Upcast -> "u"
    | Placeholder -> no_placeholder ()

  let color : t -> Helpers.color = function
    | Global -> Blue
    | Local -> Cyan
    | Warp -> Bright_cyan
    | Weak | Loop -> Red
    | Upcast -> Yellow
    | Placeholder -> no_placeholder ()

  let repr a = "AxisType." ^ name a
  let pp ppf a = Format.pp_print_string ppf (repr a)
end

(* Python literals *)

(* Arguments print as the Python literals that denote them, so that
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
(* As [Dtype.pp] prints it, without a formatter: keys write one per node. *)
let repr_dtype dt = "dtypes." ^ Dtype.name dt
let repr_addr_space a = Format.asprintf "%a" Dtype.pp_addr_space a

(* Devices *)

type device = string

let repr_device = repr_string
let pp_device ppf d = Format.pp_print_string ppf (repr_device d)

let is_disk_device d =
  let kind =
    match String.index_opt d ':' with Some i -> String.sub d 0 i | None -> d
  in
  String.equal (String.uppercase_ascii kind) "DISK"

(* Tags *)

(* Arguments *)

type param_arg = {
  slot : int;
  dtype : Dtype.t;
  vmin_vmax : (Dtype.value * Dtype.value) option;
  name : string option;
  addrspace : Dtype.addr_space;
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

(* Sets and tables of nodes keyed by their ids with open addressing in an int
   array of [2^bits] slots: a lookup reads its node once and probes ints, and
   growing reads no node, where a hash table of nodes reads every node it
   rehashes. Multiplying an id by an odd constant and keeping the top bits
   spreads ids that are close, as a graph's nodes are, over the whole array. *)
module Ids = struct
  let free = -1
  let mix = 0x3C6EF372FE94F82B
  let slot bits id = (id * mix) lsr (Sys.int_size - bits)

  (* The slot holding [id], or the free slot where it goes. *)
  let rec probe keys mask id i =
    let k = Array.unsafe_get keys i in
    if k = id || k = free then i else probe keys mask id ((i + 1) land mask)

  let index keys bits id =
    if bits = 0 then -1
    else probe keys (Array.length keys - 1) id (slot bits id)

  let held keys i = i >= 0 && Array.unsafe_get keys i <> free

  (* The bits of the least array that holds [n] ids at most three quarters
     full. *)
  let bits_for n =
    let rec go b = if 3 lsl b >= 4 * n then b else go (b + 1) in
    go 3

  (* Puts [id], absent, in [keys]; there is a free slot. *)
  let add keys bits id = Array.unsafe_set keys (index keys bits id) id
end

type wmma_axis = (Axis_type.t * int list) * int

type wmma = {
  dims : int * int * int;
  dtype_in : Dtype.t;
  threads : int;
  upcast_axes : (wmma_axis list * wmma_axis list * wmma_axis list) option;
}

type custom_function = { name : string; dtype : Dtype.t; fmt : bool }
type call_info = { name : string option; precompile : bool; inline : bool }

type t = {
  op : Op.t;
  src : t list;
  arg : arg;
  dtype : Dtype.t;
  id : int;
  mutable memos : memos; [@atomic]
}

(* Properties computed on first use. Each is a function of the node, so
   domains racing to fill one write the same value; the fields are atomic so
   that a reader that sees a value sees it whole. A field not yet set holds its
   type's sentinel, a value made once that no computation returns, compared by
   [==]: a set field holds its value unboxed. The storage base, set once for a
   whole path of nodes, is an option. *)
and memos = {
  mutable shape_memo : sint list option; [@atomic]
  mutable ranges_memo : nodes; [@atomic]
  mutable ended_ranges_memo : t list; [@atomic]
  mutable min_max_memo : Dtype.value * Dtype.value; [@atomic]
  mutable device_memo : device option; [@atomic]
  mutable addrspace_memo : Dtype.addr_space option; [@atomic]
  mutable backward_slice_memo : nodes; [@atomic]
  mutable ops_reached_memo : Op.Set.t; [@atomic]
  mutable marg_memo : movement; [@atomic]
  mutable key_memo : string; [@atomic]
  mutable arg_repr_memo : string; [@atomic]
  mutable storage_base_memo : t option; [@atomic]
}

(* A set in the order its nodes joined it. Small sets are searched in order;
   larger ones carry a table of their nodes' ids. *)
and nodes = {
  order : t list;
  cardinal : int;
  ids : int array;
  bits : int;
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
  optimized : bool;
  kernel : bool;
  split : split option;
}

and program_info = {
  global_size : sint list;
  local_size : sint list;
  params : t list;
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
  | Range of { axis_type : Axis_type.t; axis_id : int list }
  | Reduce of { op : Op.t; num_axes : int }
  | Device of device
  | String of string
  | Bytes of string
  | Code of { code : string; dtype : Dtype.t }
  | Bufferize of bufferize_opts
  | Kernel of kernel_info
  | Program of program_info
  | Custom_function of custom_function
  | Call of call_info
  | Wmma of wmma

and movement =
  | Reshape of sint list
  | Expand of sint list
  | Pad of (sint * sint) list
  | Shrink of (sint * sint) list
  | Permute of int list

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
  && Option.equal equal_value_pair p0.vmin_vmax p1.vmin_vmax
  && Option.equal String.equal p0.name p1.name
  && p0.addrspace = p1.addrspace
  && Option.equal String.equal p0.device p1.device
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
  && Bool.equal k0.optimized k1.optimized
  && Bool.equal k0.kernel k1.kernel
  && Option.equal equal_split k0.split k1.split

let equal_program_info (p0 : program_info) (p1 : program_info) =
  List.equal equal_sint p0.global_size p1.global_size
  && List.equal equal_sint p0.local_size p1.local_size
  && List.equal ( == ) p0.params p1.params
  && p0.globals = p1.globals && p0.outs = p1.outs && p0.ins = p1.ins
  && p0.target = p1.target

let equal_bufferize_opts (b0 : bufferize_opts) (b1 : bufferize_opts) =
  Option.equal String.equal b0.device b1.device
  && b0.addrspace = b1.addrspace
  && b0.keep = b1.keep

let equal_custom_function (f0 : custom_function) (f1 : custom_function) =
  String.equal f0.name f1.name && Dtype.equal f0.dtype f1.dtype
  && Bool.equal f0.fmt f1.fmt

let equal_call_info (c0 : call_info) (c1 : call_info) =
  Option.equal String.equal c0.name c1.name
  && Bool.equal c0.precompile c1.precompile
  && Bool.equal c0.inline c1.inline

let equal_arg (a0 : arg) (a1 : arg) =
  match (a0, a1) with
  | No_arg, No_arg -> true
  | Const c0, Const c1 -> Dtype.equal_const c0 c1
  | Dtype d0, Dtype d1 -> Dtype.equal d0 d1
  | Param p0, Param p1 -> equal_param_arg p0 p1
  | Range r0, Range r1 ->
      List.equal Int.equal r0.axis_id r1.axis_id
      && Axis_type.equal r0.axis_type r1.axis_type
  | Reduce r0, Reduce r1 ->
      Op.equal r0.op r1.op && Int.equal r0.num_axes r1.num_axes
  | Device d0, Device d1 -> String.equal d0 d1
  | String s0, String s1 | Bytes s0, Bytes s1 -> String.equal s0 s1
  | Code c0, Code c1 ->
      String.equal c0.code c1.code && Dtype.equal c0.dtype c1.dtype
  | Bufferize b0, Bufferize b1 -> equal_bufferize_opts b0 b1
  | Kernel k0, Kernel k1 -> equal_kernel_info k0 k1
  | Program p0, Program p1 -> equal_program_info p0 p1
  | Custom_function f0, Custom_function f1 -> equal_custom_function f0 f1
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
  | Param p -> h (3, p.slot, p.name, p.device)
  | Range r -> h (4, r.axis_id, r.axis_type)
  | Reduce r -> h (5, Op.to_int r.op, r.num_axes)
  | Device d -> h (7, d)
  | String s -> h (11, s)
  | Bytes s -> h (12, s)
  | Code c -> h (14, c.code, Dtype.hash c.dtype)
  | Bufferize b -> h (15, b.device, b.keep)
  | Kernel k -> h (16, k.name, k.beam)
  | Program p -> h (17, p.globals, List.map hash_sint p.global_size)
  | Custom_function f -> h (20, f.name, Dtype.hash f.dtype, f.fmt)
  | Call c -> h (18, c.name, c.precompile, c.inline)
  | Wmma w -> h (19, w.dims, w.threads)

(* Hash-consing *)

(* A node's hash from its parts, computed once per lookup: the low bits pick its
   slot and the high bits its shard, so both are mixed. *)
let node_hash op src arg =
  let rec fold h = function
    | [] -> h
    | s :: rest -> fold ((h * 31) + s.id) rest
  in
  let h = fold ((Op.to_int op * 31) + hash_arg arg) src * 0x4F1BBCDCBFA53E0B in
  (h lxor (h lsr 29)) land max_int

(* The table is split in shards, each behind its own lock, so that domains
   building nodes rarely wait for each other. A shard is open addressing over
   [2^bits] slots, each the hash of a node, [-1] if never used, and the node,
   held weakly so that the table keeps no node alive. A lookup compares hashes
   and reads only the nodes whose hash is its own. A slot whose node was
   collected stays used until the shard is rebuilt, keeping its live nodes,
   once half its slots are used. A shard starts at the least size, so that a
   program that builds no node keeps no table in the heap every major
   collection marks. *)
type shard = {
  mutable hashes : int array;
  mutable nodes : t Weak.t;
  mutable log_size : int;
  mutable used : int;
  lock : Mutex.t;
}

let shard_bits = 6

let shards =
  Array.init (1 lsl shard_bits) (fun _ ->
      {
        hashes = Array.make 8 (-1);
        nodes = Weak.create 8;
        log_size = 3;
        used = 0;
        lock = Mutex.create ();
      })

let next_id = Atomic.make 0

(* The sentinels of the memo fields. Each is allocated when the program
   starts, from an argument the compiler cannot see, so that no constant of
   the program is one. *)
let unset_shape : sint list option = Some (Sys.opaque_identity [])

let unset_nodes =
  { order = []; cardinal = Sys.opaque_identity (-1); ids = [||]; bits = 0 }

let unset_bounds : Dtype.value * Dtype.value =
  (Sys.opaque_identity (`Bool false), `Bool false)

let unset_device : device option = Some (Sys.opaque_identity "")

let unset_addrspace : Dtype.addr_space option =
  Some (Sys.opaque_identity Dtype.Global)

let unset_ops = Op.Set.of_list (Sys.opaque_identity [ Op.Noop ])
let unset_movement = Permute (Sys.opaque_identity [])
let unset_string = String.make 1 (Sys.opaque_identity '\000')

let memos_with ended_ranges =
  {
    shape_memo = unset_shape;
    ranges_memo = unset_nodes;
    ended_ranges_memo = ended_ranges;
    min_max_memo = unset_bounds;
    device_memo = unset_device;
    addrspace_memo = unset_addrspace;
    backward_slice_memo = unset_nodes;
    ops_reached_memo = unset_ops;
    marg_memo = unset_movement;
    key_memo = unset_string;
    arg_repr_memo = unset_string;
    storage_base_memo = None;
  }

(* The sentinel of ended ranges is a list of a node no program holds. *)
let unset_ranges =
  [
    {
      op = Op.Noop;
      src = [];
      arg = No_arg;
      dtype = Dtype.Void;
      id = Sys.opaque_identity (-1);
      memos = memos_with [];
    };
  ]

let memos () = memos_with unset_ranges

(* A node's memos until one is set: most nodes a rewrite makes are dropped
   before any property of theirs is asked. Never written. *)
let no_memos = memos ()

let node op src arg dtype id = { op; src; arg; dtype; id; memos = no_memos }

(* The memos of [u], made once one is set. Domains that race to make them
   each make them, and all keep the first made. *)
let own_memos u =
  let m = u.memos in
  if m != no_memos then m
  else
    let m = memos () in
    if Atomic.Loc.compare_and_set [%atomic.loc u.memos] no_memos m then m
    else u.memos

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

let rec all_of dt = function
  | [] -> true
  | s :: rest -> Dtype.equal s.dtype dt && all_of dt rest

let promo_dtype us =
  match us with
  | [] -> invalid_arg "no types to promote"
  | u :: rest ->
      if all_of u.dtype rest then u.dtype
      else Dtype.least_upper (List.map (fun s -> s.dtype) us)

let dtype_of op src arg =
  let mismatch what =
    invalid_argf "%s needs %s as its argument" (Op.name op) what
  in
  match op with
  | Op.Store | Op.Linear | Op.Sink | Op.Program | Op.Source | Op.Backedge
  | Op.Barrier | Op.If | Op.Endif | Op.Noop ->
      Dtype.Void
  (* a call has the type of its body, void for opaque bodies *)
  | Op.Call -> (first op src).dtype
  (* an external function states the type it returns *)
  | Op.Custom_function -> (
      match arg with
      | Custom_function f -> f.dtype
      | _ -> mismatch "a CustomFunction")
  | Op.Ins -> (
      match arg with Code c -> c.dtype | _ -> mismatch "code and a type")
  | Op.Index -> (first op src).dtype
  | Op.Load | Op.Reduce | Op.After | Op.Range | Op.Contiguous_backward
  | Op.Copy | Op.Stage | Op.Detach | Op.Special | Op.End ->
      (first op src).dtype
  | Op.Cmplt | Op.Cmpne | Op.Cmpeq -> Dtype.Bool
  | Op.Sin | Op.Log2 | Op.Exp2 | Op.Sqrt | Op.Recip ->
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
  | Op.Buffer | Op.Alloc | Op.Param | Op.Refparam -> (
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

external reraise : exn -> 'a = "%reraise"

(* Rebuilds [sh] with its live nodes, at most a quarter full. A node moves
   with [Weak.get] and [Weak.set]: [Weak.blit] scans both arrays whole during a
   phase of the major collection, which would make a rebuild quadratic. *)
let rebuild sh =
  let size = 1 lsl sh.log_size in
  let live = ref 0 in
  for i = 0 to size - 1 do
    if Weak.check sh.nodes i then incr live
  done;
  let bits = ref 3 in
  while 1 lsl !bits < 4 * !live do
    incr bits
  done;
  let bits = !bits in
  let hashes = Array.make (1 lsl bits) (-1)
  and nodes = Weak.create (1 lsl bits) in
  let mask = (1 lsl bits) - 1 and moved = ref 0 in
  for i = 0 to size - 1 do
    match Weak.get sh.nodes i with
    | None -> ()
    | Some _ as node ->
        incr moved;
        let h = Array.unsafe_get sh.hashes i in
        let j = ref (h land mask) in
        while Array.unsafe_get hashes !j <> -1 do
          j := (!j + 1) land mask
        done;
        Array.unsafe_set hashes !j h;
        Weak.set nodes !j node
  done;
  sh.hashes <- hashes;
  sh.nodes <- nodes;
  sh.log_size <- bits;
  sh.used <- !moved

(* The node of [op], [src] and [arg], found in [sh] from slot [i] on or added to
   it at the first unused slot, its hash and its node set with nothing
   allocated between. A slot whose node was collected is passed over, and
   freed once the shard is rebuilt. *)
let rec intern sh hash op src arg i =
  let h = Array.unsafe_get sh.hashes i in
  let next = (i + 1) land ((1 lsl sh.log_size) - 1) in
  if h = -1 then begin
    let dtype = dtype_of op src arg in
    let u = node op src arg dtype (Atomic.fetch_and_add next_id 1) in
    let some = Some u in
    Array.unsafe_set sh.hashes i hash;
    Weak.set sh.nodes i some;
    sh.used <- sh.used + 1;
    if 2 * sh.used > 1 lsl sh.log_size then rebuild sh;
    u
  end
  else if h <> hash then intern sh hash op src arg next
  else
    match Weak.get sh.nodes i with
    | Some u
      when u.op = op && List.equal ( == ) u.src src && equal_arg u.arg arg ->
        u
    | _ -> intern sh hash op src arg next

let v ?(src = []) ?(arg = No_arg) op =
  let hash = node_hash op src arg in
  let sh = shards.(hash lsr (Sys.int_size - 1 - shard_bits)) in
  (* Written as [Mutex.protect] is, without its closure: nothing runs between
     the lock and the handler, so an exception raised in the section, as a
     timeout's or Ctrl-C's is at a poll, releases the lock. The shard stays
     whole: a slot takes its hash and its node with no poll between, and a
     rebuilt shard replaces the old one only once whole. *)
  Mutex.lock sh.lock;
  match intern sh hash op src arg (hash land ((1 lsl sh.log_size) - 1)) with
  | u ->
      Mutex.unlock sh.lock;
      u
  | exception e ->
      Mutex.unlock sh.lock;
      reraise e

let op u = u.op
let dtype u = u.dtype
let src u = u.src
let arg u = u.arg

let nth u i =
  match List.nth_opt u.src i with
  | Some s -> s
  | None -> invalid_argf "%s has no source %d" (Op.name u.op) i

let replace ?op ?src ?arg u =
  let op = Option.value op ~default:u.op
  and src = Option.value src ~default:u.src
  and arg = Option.value arg ~default:u.arg in
  if Op.equal op u.op && List.equal ( == ) src u.src && equal_arg arg u.arg then u
  else v ~src ~arg op

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

  (* A set of more than 8 nodes also holds their ids, in [Ids]' slots. *)
  let of_list order =
    let cardinal = List.length order in
    if cardinal <= 8 then { order; cardinal; ids = [||]; bits = 0 }
    else
      let bits = Ids.bits_for cardinal in
      let ids = Array.make (1 lsl bits) Ids.free in
      List.iter
        (fun u ->
          if not (Ids.held ids (Ids.index ids bits u.id)) then
            Ids.add ids bits u.id)
        order;
      { order; cardinal; ids; bits }

  let mem u s =
    if s.bits = 0 then List.memq u s.order
    else Ids.held s.ids (Ids.index s.ids s.bits u.id)

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
      ("name", Option.map repr_string p.name);
      ( "addrspace",
        match p.addrspace with
        | Dtype.Global -> None
        | a -> Some (repr_addr_space a) );
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
    @ List.filter_map (fun (k, v) -> Option.map (fun v -> k ^ "=" ^ v) v) fields
  in
  strf "ParamArg(%s)" (String.concat ", " args)

let repr_bufferize_opts (b : bufferize_opts) =
  strf "BufferizeOpts(device=%s, addrspace=%s, removable=%s, broadcast=%s)"
    (repr_option repr_device b.device)
    (repr_addr_space b.addrspace)
    (repr_bool (b.keep <> Whole))
    (repr_bool (b.keep = Broadcast))

let repr_custom_function (f : custom_function) =
  strf "CustomFunction(name=%s, dtype=%s, fmt=%s)" (repr_string f.name)
    (repr_dtype f.dtype) (repr_bool f.fmt)

let repr_call_info (c : call_info) =
  strf "CallInfo(None, %s, %s, False, %s, None)"
    (repr_option repr_string c.name)
    (repr_bool c.precompile) (repr_bool c.inline)

let repr_wmma w =
  let m, n, k = w.dims in
  let repr_axes axes =
    repr_tuple
      (List.map
         (fun ((t, ids), size) ->
           repr_tuple
             [
               repr_tuple (Axis_type.repr t :: List.map string_of_int ids);
               string_of_int size;
             ])
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
        (strf "UOp(%s, arg=%s, src=("
           (Format.asprintf "%a" Op.pp u.op)
           (arg_repr u));
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

and arg_repr u =
  let s = u.memos.arg_repr_memo in
  if s != unset_string then s
  else
    let s = repr_arg u.arg in
    (own_memos u).arg_repr_memo <- s;
    s

and repr_sint s = repr_sint_with repr s

and repr_estimates (e : estimates) =
  strf "Estimates(ops=%s, lds=%s, mem=%s)" (repr_sint e.ops) (repr_sint e.lds)
    (repr_sint e.mem)

and repr_kernel_info (k : kernel_info) =
  strf
    "KernelInfo(name=%s, applied_opts=%s, opts_to_apply=%s, estimates=%s, \
     beam=%d, optimized=%s, kernel=%s, split=%s)"
    (repr_string k.name)
    (repr_tuple (List.map repr_opt k.applied_opts))
    (repr_option (fun l -> repr_tuple (List.map repr_opt l)) k.opts_to_apply)
    (repr_option repr_estimates k.estimates)
    k.beam (repr_bool k.optimized) (repr_bool k.kernel)
    (repr_option
       (fun s ->
         repr_tuple
           [ repr_sint s.iterations; string_of_int s.lo; string_of_int s.hi ])
       k.split)

and repr_program_info (p : program_info) =
  let ints l = repr_tuple (List.map string_of_int l) in
  strf
    "ProgramInfo(global_size=%s, local_size=%s, params=%s, globals=%s, \
     outs=%s, ins=%s, target=%s)"
    (repr_tuple (List.map repr_sint p.global_size))
    (repr_tuple (List.map repr_sint p.local_size))
    (repr_tuple (List.map repr p.params))
    (ints p.globals) (ints p.outs) (ints p.ins)
    (Format.asprintf "%a" Helpers.Target.pp p.target)

and repr_arg = function
  | No_arg -> "None"
  | Const (`Float x) -> strf "ConstFloat(%s)" (repr_const (`Float x))
  | Const c -> repr_const c
  | Dtype dt -> repr_dtype dt
  | Param p -> repr_param_arg p
  | Range r ->
      repr_tuple
        (Axis_type.repr r.axis_type :: List.map string_of_int r.axis_id)
  | Reduce r ->
      repr_tuple [ Format.asprintf "%a" Op.pp r.op; string_of_int r.num_axes ]
  | Device d -> repr_device d
  | String s -> repr_string s
  | Bytes s -> repr_bytes s
  | Code c -> repr_tuple [ repr_string c.code; repr_dtype c.dtype ]
  | Bufferize b -> repr_bufferize_opts b
  | Kernel k -> repr_kernel_info k
  | Program p -> repr_program_info p
  | Custom_function f -> repr_custom_function f
  | Call c -> repr_call_info c
  | Wmma w -> repr_wmma w

let pp ppf u = Format.pp_print_string ppf (repr u)
let pp_arg ppf a = Format.pp_print_string ppf (repr_arg a)
let pp_estimates ppf e = Format.pp_print_string ppf (repr_estimates e)

(* Graphs *)

type calls = Enter | Skip

(* Nodes no graph holds. Making an array of more than 256 words filled with a
   young value forces a minor collection, so the arrays below are filled with
   immediate values or with these nodes, made once at initialisation and old by
   the time a walk needs such an array. [unset] stands for an entry not yet
   set. *)
let placeholder = node Op.Noop [] No_arg Dtype.Void (-2)
let unset = node Op.Noop [] No_arg Dtype.Void (-3)

(* The arrays of walks, which each domain keeps for its later walks, by size: a
   pass over a program of 50,000 nodes fills arrays of 2^16 words, which the
   major heap would otherwise allocate, mark and sweep anew for each of the
   dozens of passes over the program. Arrays have a power of two of entries.
   The domain holds its pool weakly: the passes of a compile follow each other
   within a few major collections and reuse the arrays, and once tolk is idle
   the next collection frees them unmarked. Arrays of nodes come back filled
   with [unset], which a walk's new rows start as, and arrays of pairs with
   each pair free; other arrays of ints come back as they were left. A walk
   that an exception ends gives its arrays back too. *)
module Pool = struct
  (* A stack of arrays. A cell is made before it is linked, so that nothing
     allocates, and no other thread of the domain runs, between reading a
     stack and writing it. *)
  type 'a stack = Nil | Cell of { a : 'a; mutable next : 'a stack }

  (* The stacks by size, past [Max_young_wosize] entries so that the major heap
     holds them from the start: a weak pointer to a young block is cleared at
     the next minor collection, and a pool that lasted until then would not
     follow the passes. *)
  let length = 257

  (* [top] is at least the class of each array given back since the stacks
     were made, so that finding the largest scans no class above it. *)
  type 'a pool = { weak : 'a stack array Weak.t; mutable top : int }
  type held = {
    ints : int array pool;
    pairs : int array pool;
    nodes : t array pool;
  }

  let held =
    Domain.DLS.new_key (fun () ->
        {
          ints = { weak = Weak.create 1; top = 0 };
          pairs = { weak = Weak.create 1; top = 0 };
          nodes = { weak = Weak.create 1; top = 0 };
        })

  (* The domain's stacks of [pool], made anew once collected. Threads of the
     domain that both find them collected each make them, and one is kept. *)
  let stacks pool =
    match Weak.get pool.weak 0 with
    | Some p -> p
    | None ->
        let p = Array.make length Nil in
        Weak.set pool.weak 0 (Some p);
        pool.top <- 0;
        p

  (* The bits of the least power of two at least [n]. *)
  let bits n =
    let b = ref 0 in
    while 1 lsl !b < n do
      incr b
    done;
    !b

  let give pool a =
    let stacks = stacks pool and b = bits (Array.length a) in
    let cell = Cell { a; next = Nil } in
    match cell with
    | Cell c ->
        c.next <- stacks.(b);
        stacks.(b) <- cell;
        if b > pool.top then pool.top <- b
    | Nil -> ()

  (* An array of class [b] from [stacks], or [make (1 lsl b)]. *)
  let take stacks b make =
    match stacks.(b) with
    | Cell c ->
        stacks.(b) <- c.next;
        c.a
    | Nil -> make (1 lsl b)

  (* The class of the largest array in [stacks] of class [b] to [c], [b] if
     none. *)
  let rec largest stacks b c =
    if c <= b then b
    else
      match stacks.(c) with Cell _ -> c | Nil -> largest stacks b (c - 1)

  let make_ints n = Array.make n Ids.free
  let make_nodes n = Array.make n unset

  (* An array of at least [n] entries. *)
  let ints n = take (stacks (Domain.DLS.get held).ints) (bits n) make_ints

  (* The largest array of at least [n] entries: a walk that outgrows its
     arrays takes the largest, which an earlier pass over the program left,
     rather than doubling them, each doubling a copy of what it holds. *)
  let largest_ints n =
    let pool = (Domain.DLS.get held).ints in
    let stacks = stacks pool and b = bits n in
    take stacks (largest stacks b pool.top) make_ints

  (* Pairs of ints, each free, of at least [n] ints. *)
  let pairs n = take (stacks (Domain.DLS.get held).pairs) (bits n) make_ints

  let largest_pairs n =
    let pool = (Domain.DLS.get held).pairs in
    let stacks = stacks pool and b = bits n in
    take stacks (largest stacks b pool.top) make_ints

  let nodes n = take (stacks (Domain.DLS.get held).nodes) (bits n) make_nodes

  let largest_nodes n =
    let pool = (Domain.DLS.get held).nodes in
    let stacks = stacks pool and b = bits n in
    take stacks (largest stacks b pool.top) make_nodes

  let give_ints a = give (Domain.DLS.get held).ints a

  (* Gives [a] back, each of its pairs free. *)
  let give_pairs a = give (Domain.DLS.get held).pairs a

  (* Gives [a] back, of which the first [used] entries may be set. *)
  let give_nodes a used =
    Array.fill a 0 used unset;
    give (Domain.DLS.get held).nodes a

  (* Copies the first [n] entries of [src] to [dst]. [Array.blit] writes each
     int into an array of the major heap as it would a pointer. *)
  let copy_ints (src : int array) (dst : int array) n =
    for i = 0 to n - 1 do
      Array.unsafe_set dst i (Array.unsafe_get src i)
    done
end

(* The nodes a walk meets, numbered in the order it meets them, so that what it
   keeps for a node lies in a row of plain arrays and its stack holds numbers.
   A row is the node and [width - 1] more nodes in [nodes], and in [ints] the
   index of the node's pair and [iwidth] more ints. A node's number is found by
   its id in [Ids]' slots, each a pair of ints, the id and the number, so that
   a lookup reads one line. Releasing frees each row's pair, so the pairs go
   back to the pool free whatever their length. *)
module Local = struct
  type node = t

  type t = {
    mutable slots : int array;
    mutable bits : int;
    mutable count : int;
    mutable rows : int;
    mutable nodes : node array;
    mutable ints : int array;
    width : int;
    stride : int;
    mutable moving : bool;
  }

  let create ~width ~iwidth =
    let nodes = Pool.nodes (8 * width) in
    let rows = Array.length nodes / width in
    {
      slots = Pool.pairs 16;
      bits = 3;
      count = 0;
      rows;
      nodes;
      ints = Pool.ints (rows * (iwidth + 1));
      width;
      stride = iwidth + 1;
      moving = false;
    }

  (* The pair holding [id], or the free pair where it goes. *)
  let rec probe slots mask id i =
    let k = Array.unsafe_get slots (2 * i) in
    if k = id || k = Ids.free then i
    else probe slots mask id ((i + 1) land mask)

  let pair t id = probe t.slots ((1 lsl t.bits) - 1) id (Ids.slot t.bits id)

  (* The number of [u], [Ids.free] if it has none. *)
  let find t (u : node) =
    let i = pair t u.id in
    if Array.unsafe_get t.slots (2 * i) = Ids.free then Ids.free
    else Array.unsafe_get t.slots ((2 * i) + 1)

  (* Moves the pairs to the largest the pool holds, at least twice as many,
     leaving the old ones free. While they move, [moving] keeps [release] from
     trusting the rows' pairs. *)
  let grow_slots t =
    let slots = Pool.largest_pairs (4 lsl t.bits) in
    let bits = Pool.bits (Array.length slots) - 1 in
    let mask = (1 lsl bits) - 1 and old = t.slots in
    t.moving <- true;
    for i = 0 to (1 lsl t.bits) - 1 do
      let id = Array.unsafe_get old (2 * i) in
      if id <> Ids.free then begin
        let l = Array.unsafe_get old ((2 * i) + 1) in
        let j = probe slots mask id (Ids.slot bits id) in
        Array.unsafe_set slots (2 * j) id;
        Array.unsafe_set slots ((2 * j) + 1) l;
        Array.unsafe_set t.ints (l * t.stride) j;
        Array.unsafe_set old (2 * i) Ids.free
      end
    done;
    t.slots <- slots;
    t.bits <- bits;
    t.moving <- false;
    Pool.give_pairs old

  (* Moves the rows to the largest arrays the pool holds, each long enough for
     at least twice the rows. An array given back stops being the walk's before
     anything else can raise. *)
  let grow_rows t =
    let rows = 2 * t.rows in
    let nodes = Pool.largest_nodes (rows * t.width)
    and ints = Pool.largest_ints (rows * t.stride) in
    Array.blit t.nodes 0 nodes 0 (t.count * t.width);
    Pool.copy_ints t.ints ints (t.count * t.stride);
    Pool.give_nodes t.nodes (t.count * t.width);
    t.nodes <- nodes;
    Pool.give_ints t.ints;
    t.ints <- ints;
    t.rows <- min (Array.length nodes / t.width) (Array.length ints / t.stride)

  (* The number of [u], which it is given if it has none. A new row holds [u],
     [unset] nodes and [Ids.free] ints. Its pair is set last, after the row
     counts, with nothing between that could raise: every pair set belongs to
     a row that [release] frees, even if an exception from a signal ends the
     walk. *)
  let add t (u : node) =
    let i = pair t u.id in
    if Array.unsafe_get t.slots (2 * i) <> Ids.free then
      Array.unsafe_get t.slots ((2 * i) + 1)
    else begin
      let l = t.count in
      if l = t.rows then grow_rows t;
      (* At most three quarters full. *)
      let i =
        if 4 * (l + 1) <= 3 lsl t.bits then i
        else (
          grow_slots t;
          pair t u.id)
      in
      Array.unsafe_set t.nodes (l * t.width) u;
      Array.unsafe_set t.ints (l * t.stride) i;
      for k = (l * t.stride) + 1 to ((l + 1) * t.stride) - 1 do
        Array.unsafe_set t.ints k Ids.free
      done;
      t.count <- l + 1;
      Array.unsafe_set t.slots (2 * i) u.id;
      Array.unsafe_set t.slots ((2 * i) + 1) l;
      l
    end

  let node t l = Array.unsafe_get t.nodes (l * t.width)
  let get t l k = Array.unsafe_get t.nodes ((l * t.width) + k)
  let set t l k x = Array.unsafe_set t.nodes ((l * t.width) + k) x
  let iget t l k = Array.unsafe_get t.ints ((l * t.stride) + k + 1)
  let iset t l k x = Array.unsafe_set t.ints ((l * t.stride) + k + 1) x

  (* Gives the arrays back, the pairs freed row by row. Pairs that were moving
     when an exception ended the walk are dropped. *)
  let release t =
    if not t.moving then begin
      for l = 0 to t.count - 1 do
        Array.unsafe_set t.slots (2 * Array.unsafe_get t.ints (l * t.stride))
          Ids.free
      done;
      Pool.give_pairs t.slots
    end;
    Pool.give_nodes t.nodes (t.count * t.width);
    Pool.give_ints t.ints
end

(* A stack of entries of four ints, which pushing allocates nothing for. *)
module Work = struct
  type t = { mutable a : int array; mutable len : int }

  let create () = { a = Pool.ints 64; len = 0 }
  let is_empty s = s.len = 0
  let length s = s.len / 4

  let push s w x y z =
    if s.len = Array.length s.a then begin
      let a = Pool.largest_ints (2 * s.len) in
      Pool.copy_ints s.a a s.len;
      Pool.give_ints s.a;
      s.a <- a
    end;
    let a = s.a and i = s.len in
    Array.unsafe_set a i w;
    Array.unsafe_set a (i + 1) x;
    Array.unsafe_set a (i + 2) y;
    Array.unsafe_set a (i + 3) z;
    s.len <- i + 4

  (* Part [k] of entry [e], counted from the bottom, and of the top entry. *)
  let get s e k = Array.unsafe_get s.a ((4 * e) + k)
  let set s e k x = Array.unsafe_set s.a ((4 * e) + k) x
  let top s k = Array.unsafe_get s.a (s.len - 4 + k)
  let drop s = s.len <- s.len - 4
  let release s = Pool.give_ints s.a
end

(* Each node is pushed with a flag: unset, its sources are pushed after it; set,
   its sources are done and it is. A node pushed twice before it is done is
   finished at its first pop with the flag set. A row's int is set once its node
   is done, and its number pushed on [order], from which the list is made last
   first, so that it is made once. *)
let toposort ?gate ~calls root =
  let seen = Local.create ~width:1 ~iwidth:1
  and work = Work.create ()
  and order = Work.create () in
  let rec push_srcs = function
    | [] -> ()
    | s :: rest ->
        push_srcs rest;
        Work.push work (Local.add seen s) 0 0 0
  in
  let walk () =
    Work.push work (Local.add seen root) 0 0 0;
    while not (Work.is_empty work) do
      let l = Work.top work 0 and visited = Work.top work 1 in
      Work.drop work;
      if Local.iget seen l 0 = Ids.free then begin
        let node = Local.node seen l in
        if visited = 0 then begin
          if match gate with None -> true | Some g -> g node then begin
            Work.push work l 1 0 0;
            push_srcs
              (if calls = Skip && node.op = Op.Call then drop 1 node.src
               else node.src)
          end
        end
        else begin
          Local.iset seen l 0 1;
          Work.push order l 0 0 0
        end
      end
    done;
    let nodes = ref [] in
    for e = Work.length order - 1 downto 0 do
      nodes := Local.node seen (Work.get order e 0) :: !nodes
    done;
    !nodes
  in
  let release () =
    Local.release seen;
    Work.release work;
    Work.release order
  in
  match walk () with
  | nodes ->
      release ();
      nodes
  | exception e ->
      release ();
      reraise e

let topovisit root f cache =
  let seen = Local.create ~width:1 ~iwidth:0 and work = Work.create () in
  let rec push_srcs = function
    | [] -> ()
    | s :: rest ->
        push_srcs rest;
        Work.push work (Local.add seen s) 0 0 0
  in
  let walk () =
    Work.push work (Local.add seen root) 0 0 0;
    while not (Work.is_empty work) do
      let l = Work.top work 0 and visited = Work.top work 1 in
      Work.drop work;
      let node = Local.node seen l in
      if not (Tbl.mem cache node) then
        if visited = 0 then begin
          Work.push work l 1 0 0;
          push_srcs node.src
        end
        else Tbl.replace cache node (f node)
    done
  in
  match walk () with
  | () ->
      Local.release seen;
      Work.release work;
      Tbl.find cache root
  | exception e ->
      Local.release seen;
      Work.release work;
      reraise e

let rec all_known unset get = function
  | [] -> true
  | s :: srcs -> get s != unset && all_known unset get srcs

(* A recursive property is filled bottom-up over the nodes that lack it, so a
   deep graph never recurses deeply. *)
let memoized ~calls ~unset ~get ~set ~compute u =
  let x = get u in
  if x != unset then x
  else begin
    (* A new node is mostly built on nodes that have the property. *)
    let srcs =
      if calls = Skip && u.op = Op.Call then drop 1 u.src else u.src
    in
    if all_known unset get srcs then set u (compute u)
    else
      List.iter
        (fun n -> set n (compute n))
        (toposort ~calls ~gate:(fun n -> get n == unset) u);
    get u
  end

(* A slice outside call bodies is kept with its node; one that enters them is
   walked anew. *)
let backward_slice ~calls u =
  let walk () =
    Nodes.of_list (List.filter (fun n -> n != u) (toposort ~calls u))
  in
  match calls with
  | Enter -> walk ()
  | Skip ->
      let s = u.memos.backward_slice_memo in
      if s != unset_nodes then s
      else
        let s = walk () in
        (own_memos u).backward_slice_memo <- s;
        s

let backward_slice_with_self ~calls u =
  Nodes.of_list (u :: Nodes.to_list (backward_slice ~calls u))

(* The operations of [u] and of the nodes it reaches outside call bodies, as a
   property of each node, so that asking costs no walk of the slice. A node
   whose set is one of its sources' shares that source's set. *)
let ops_reached u =
  memoized ~calls:Skip ~unset:unset_ops
    ~get:(fun n -> n.memos.ops_reached_memo)
    ~set:(fun n s -> (own_memos n).ops_reached_memo <- s)
    ~compute:(fun n ->
      let srcs = if n.op = Op.Call then drop 1 n.src else n.src in
      let sets = List.map (fun s -> s.memos.ops_reached_memo) srcs in
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
let reaches ?(gate = fun _ -> true) ~calls u x =
  let gate n =
    if n == x then raise_notrace Exit;
    n.id >= x.id && gate n
  in
  u == x
  || x.id < u.id
     && match toposort ~calls ~gate u with _ -> false | exception Exit -> true

let rec split_uop u sep =
  if Op.equal u.op sep then List.concat_map (fun s -> split_uop s sep) u.src
  else [ u ]

(* Equal arguments or types, which most ties have, compare without the
   arguments' reprs or the types' order. *)
let compare_structure u0 u1 =
  let args u0 u1 =
    if u0.arg == u1.arg then 0 else String.compare (arg_repr u0) (arg_repr u1)
  in
  let dtypes (d0 : Dtype.t) d1 = if d0 = d1 then 0 else Dtype.compare d0 d1 in
  let rec cmp u0 u1 =
    if u0 == u1 then 0
    else
      match Int.compare (Op.to_int u0.op) (Op.to_int u1.op) with
      | 0 -> (
          match args u0 u1 with
          | 0 -> (
              match dtypes u0.dtype u1.dtype with
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

(* An argument as [key] reads it. A NaN constant enters with its bits, since
   NaNs of different bits are different nodes and a repr writes every NaN [nan].
   Any other enters as its repr, which [repr] gives. *)
let key_of_arg repr = function
  | Const (`Float x) when Float.is_nan x ->
      strf "ConstFloat(nan 0x%Lx)" (Int64.bits_of_float x)
  | _ -> repr ()

let arg_key n = key_of_arg (fun () -> arg_repr n) n.arg

(* A node's digest from its argument's key and its sources' digests. *)
let rec add_keys b src_key = function
  | [] -> ()
  | s :: rest ->
      Buffer.add_string b (src_key s);
      add_keys b src_key rest

let digest n arg_key src_key =
  let b = Buffer.create 128 in
  Buffer.add_string b
    (repr_tuple [ "Ops." ^ Op.name n.op; repr_dtype n.dtype; arg_key ]);
  add_keys b src_key n.src;
  Digest.BLAKE256.string (Buffer.contents b)

let key u =
  memoized ~calls:Enter ~unset:unset_string
    ~get:(fun n -> n.memos.key_memo)
    ~set:(fun n k -> (own_memos n).key_memo <- k)
    ~compute:(fun n -> digest n (arg_key n) (fun s -> s.memos.key_memo))
    u

(* The nodes [arg] gives an argument and those above them differ from their
   [key]: their digests are held here, the others' are their keys, filled in
   order. *)
let key_with ~arg u =
  let moved = Tbl.create 16 in
  let src_key s =
    match Tbl.find moved s with k -> k | exception Not_found -> key s
  in
  let is_moved s = Tbl.mem moved s in
  List.iter
    (fun n ->
      match arg n with
      | Some a ->
          let text = key_of_arg (fun () -> repr_arg a) a in
          Tbl.replace moved n (digest n text src_key)
      | None ->
          if List.exists is_moved n.src then
            Tbl.replace moved n (digest n (arg_key n) src_key)
          else ignore (key n))
    (toposort ~calls:Enter u);
  src_key u

let identity_element op dt : Dtype.const =
  match op with
  | Op.Add -> Dtype.const dt (`Int Bigint.zero)
  | Op.Mul -> Dtype.const dt (`Int Bigint.one)
  (* A weak type has no least value: a weak integer's minimum is that of its
     800-bit place in the lattice, which no integer holds. *)
  | Op.Max -> Dtype.const dt (Dtype.min (Dtype.strong dt) :> Dtype.const)
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
  val eq : t -> t -> t
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
  val ne : t -> t -> t
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
  (* Whether [//] and [%] are DIV and MOD whatever the operands'
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
  let eq x y = binop Op.Cmpeq x y
  let logical_not x = eq (cast x Dtype.Bool) (bool false)
  let ne x y = logical_not (eq x y)

  let neg x =
    if Dtype.equal (dtype x) Dtype.Bool then logical_not x else mul x (int (-1))

  (* The promoted operands are combined with [alu]: promoting the negation again
     would cast it, since only a weak constant stays weak. *)
  let sub x y =
    let a, b = broadcasted x y in
    alu a Op.Add [ neg b ]

  let reciprocal x = alu x Op.Recip []
  let trunc x = alu x Op.Trunc []
  let sqrt x = alu x Op.Sqrt []
  let exp2 x = alu x Op.Exp2 []
  let log2 x = alu x Op.Log2 []
  let lt x y = binop Op.Cmplt x y
  let gt x y = binop Op.Cmplt y x
  let le x y = logical_not (gt x y)
  let ge x y = logical_not (lt x y)

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
    | Some `Trunc when ints -> alu a Op.Truncdiv [ b ]
    | Some `Floor when ints -> alu a Op.Div [ b ]
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
    if direct_floor then binop Op.Mod x y
    else remainder Op.Mod `Floor x y

  let fmod x y = remainder Op.Truncmod `Trunc x y

  let pow x y =
    let base, exponent = broadcasted x y in
    (* An exponent is a number, whatever its base's type holds. *)
    let exponent =
      if Dtype.equal (dtype y) Dtype.Weak_int && Dtype.is_int (dtype base) then y
      else exponent
    in
    let negative_int =
      match literal_value y with
      | Some (`Int z) -> Bigint.lt z Bigint.zero
      | _ -> false
    in
    if
      (not (Dtype.is_float (Dtype.least_upper [ dtype base; dtype exponent ])))
      && negative_int
    then invalid_arg "integers to negative integer powers are not allowed";
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
      if direct_floor then binop Op.Div x y else div ~rounding:`Floor x y

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

let is_literal u =
  u.op = Op.Const || (u.op = Op.Cast && (List.hd u.src).op = Op.Const)

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
    else if Dtype.mem t.dtype Dtype.weaks && b.op = Op.Const then
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

let rec buf_uop u =
  match u.op with
  | op when Op.Set.mem op Op.Set.defines -> u
  | _ ->
      let b = base u in
      if b.op = Op.After then base (buf_uop (first b.op b.src))
      else
        let rec down s =
          match s.src with
          | first :: _
            when not (s.op = Op.Stage || Op.Set.mem s.op Op.Set.defines)
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
    when Dtype.mem dt Dtype.ints
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
(* A count no [int] holds is past the bits of every value, which
   [Bigint.shift_right] reads from any [int]. *)
let shift_count z = if Bigint.fits_int z then Bigint.to_int z else Int.max_int

(* The most bits a left shift of no width computes exactly: past them it has no
   value, which keeps a count from allocating more than memory holds. *)
let max_exact_shift = 0x100_0000

let rec min_max u =
  let b = u.memos.min_max_memo in
  if b != unset_bounds then b
  else
    let b = at_width u.dtype (compute_min_max u) in
    (own_memos u).min_max_memo <- b;
    b

and operand_bounds u s =
  let operands =
    if Op.Set.mem u.op Op.Set.comparison then promo_dtype u.src else u.dtype
  in
  match (u.op, u.src) with
  (* a count is a number of bits, read at no width but its own *)
  | (Op.Shl | Op.Shr), [ _; c ] when c == s -> min_max s
  | _ ->
      if Dtype.equal s.dtype Dtype.Weak_float && Dtype.mem operands Dtype.floats
      then rounded operands (min_max s)
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
          (* an integer [where (a < b) a b] is the minimum of [a] and [b] *)
          | [ { op = Op.Cmplt; src = [ a; b ]; _ }; x; y ]
            when Dtype.is_int dt && a == x && b == y ->
              let (x0, x1), (y0, y1) = (bounds x, bounds y) in
              (Value.min x0 y0, Value.min x1 y1)
          | [ c; x; y ] ->
              let (x0, x1), (y0, y1) = (selected c (bounds x) x, bounds y) in
              (Value.min x0 y0, Value.max x1 y1)
          | _ -> invalid_arg "where needs three sources")
      | ( (Op.Param | Op.Refparam | Op.Buffer | Op.Alloc),
          Param { vmin_vmax = Some b; _ } ) ->
          b
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
              (* a gated load reads its alternative where its gate fails *)
              let alt =
                match u.src with
                | [ _; alt; _ ] ->
                    let lo, hi = bounds alt in
                    [ lo; hi ]
                | _ -> []
              in
              if List.exists is_nan values then unbounded dt
              else extremes (values @ alt)
          | _ -> invalid_arg "a constant table needs bytes")
      | Op.Const, Const ((`Bool _ | `Int _) as c) -> (c, c)
      | Op.Const, Const (`Float x) when not (Float.is_nan x) ->
          (`Float x, `Float x)
      | Op.Pad, _ ->
          let lo, hi = bounds (src0 ()) in
          (Value.min lo (`Int Bigint.zero), Value.max hi (`Int Bigint.zero))
      | ( Op.Index | Op.Stage | Op.After | Op.Detach | Op.Copy
        | Op.Contiguous_backward ),
        _ ->
          bounds (src0 ())
      | op, _ when Op.Set.mem op Op.Set.movement -> bounds (src0 ())
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
  let integer t = Dtype.mem t Dtype.ints in
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
      Dtype.mem dt Dtype.weaks,
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
   integer, to nearest into a float, which for a weak float is a double, as its
   value is ({!Dtype.const}). An integer keeps its value in an integer type,
   where [min_max] wraps it. A float target holds the rounded bounds only where
   they are its values: a value past them is NaN in a type without infinities,
   and a NaN bound is no value. A signed target holds the part of the source
   range that overlaps it; a float overflowing an integer is undefined. *)
and cast_bounds src dt (lo, hi) =
  let round (x : Dtype.value) : Dtype.value =
    match x with
    | `Float f when not (Float.is_finite f) -> x
    | _ ->
        if Dtype.equal dt Weak_float then `Float (Value.to_float x)
        else if Dtype.is_float dt then Dtype.truncate dt x
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
    Dtype.mem dt clamped_types
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
  let float_operands =
    match u.src with s :: _ -> Dtype.is_float s.dtype | [] -> false
  in
  let corners f =
    Some
      (extremes
         [ f s0_min s1_min; f s0_min s1_max; f s0_max s1_min; f s0_max s1_max ])
  in
  match u.op with
  | Op.Add -> Some (add s0_min s1_min, add s0_max s1_max)
  | Op.Sub -> Some (sub s0_min s1_max, sub s0_max s1_min)
  | Op.And
    when Dtype.is_int u.dtype && c1 && (le zero s0_min || le zero s1_max) ->
      if lt s0_min zero then Some (zero, s1_max)
      else
        let mask = Bigint.pred (Bigint.shift_left Bigint.one (Bigint.numbits (z s0_max))) in
        Some (zero, Value.min s0_max (`Int (Bigint.logand (z s1_max) mask)))
  | Op.Mul -> corners mul
  (* A shift by a negative count is undefined: its bounds are the type's. *)
  | Op.Shl when c1 && ints [ s0_min; s0_max; s1_min ] && le zero s1_min ->
      let k = shift_count (z s1_min) in
      if Dtype.mem u.dtype Dtype.ints && k >= Dtype.bitsize u.dtype then
        Some (zero, zero)
      else if k > max_exact_shift then None
      else
        Some (`Int (Bigint.shift_left (z s0_min) k), `Int (Bigint.shift_left (z s0_max) k))
  | Op.Shr when c1 && ints [ s0_min; s0_max; s1_min ] && le zero s1_min ->
      let k = shift_count (z s1_min) in
      Some (`Int (Bigint.shift_right (z s0_min) k), `Int (Bigint.shift_right (z s0_max) k))
  | Op.Truncmod when c1 && lt zero s1_max ->
      let c = s1_min in
      Some
        ( (if lt zero s0_min then zero
           else if lt (neg c) s0_min then s0_min
           else neg (sub s1_max (`Int Bigint.one))),
          if lt s0_max zero then zero
          else if lt s0_max c then s0_max
          else sub c (`Int Bigint.one) )
  | Op.Truncmod when lt zero s1_min ->
      let m = sub s1_max (`Int Bigint.one) in
      Some
        (if le zero s0_min then (zero, m)
         else if le s0_max zero then (neg m, zero)
         else (neg m, m))
  | Op.Truncmod when lt s1_max zero ->
      let m = sub (neg s1_min) (`Int Bigint.one) in
      Some
        (if le zero s0_min then (zero, m)
         else if le s0_max zero then (neg m, zero)
         else (neg m, m))
  | Op.Truncdiv when Bigint.gt (Bigint.mul (z s1_min) (z s1_max)) Bigint.zero ->
      let d a b = divide `Quotient ~toward_zero:true a b in
      corners d
  | (Op.Div | Op.Mod) when lt s0_max s0_min -> Some (zero, zero)
  | Op.Div when Bigint.gt (Bigint.mul (z s1_min) (z s1_max)) Bigint.zero ->
      let d a b = Value.(a // b) in
      corners d
  | Op.Mod when c1 && not (Bigint.equal (z s1_min) Bigint.zero) ->
      let c = s1_min in
      if Value.(s0_min // c = s0_max // c) then
        Some (Value.(s0_min % c), Value.(s0_max % c))
      else if lt zero c then Some (zero, sub c (`Int Bigint.one))
      else Some (add c (`Int Bigint.one), zero)
  | Op.Mod when lt zero s1_min -> Some (zero, sub s1_max (`Int Bigint.one))
  | Op.Mod when lt s1_max zero -> Some (add s1_min (`Int Bigint.one), zero)
  | Op.Xor
    when equal s1_min (`Int Bigint.minus_one)
         && equal s1_max (`Int Bigint.minus_one)
         && ints [ s0_min; s0_max ] ->
      Some (`Int (Bigint.lognot (z s0_max)), `Int (Bigint.lognot (z s0_min)))
  | Op.Xor
    when Dtype.is_int u.dtype && ints [ s0_max; s1_max ] && le zero s0_min
         && le zero s1_min ->
      let n = Bigint.numbits (z (Value.max s0_max s1_max)) in
      Some (zero, `Int (Bigint.pred (Bigint.shift_left Bigint.one n)))
  | Op.Max -> Some (Value.max s0_min s1_min, Value.max s0_max s1_max)
  (* bounds never prove a float comparison true: a NaN compares false *)
  | Op.Cmplt ->
      Some
        ( `Bool (lt s0_max s1_min && not float_operands),
          `Bool (lt s0_min s1_max) )
  | Op.Cmpeq ->
      Some
        ( `Bool
            (equal s0_min s0_max && equal s0_max s1_min && equal s1_min s1_max
           && not float_operands),
          `Bool (not (lt s0_max s1_min || lt s1_max s0_min)) )
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
  (not (Dtype.mem dt Dtype.ints))
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
  | Op.Stage | Op.Reduce | Op.End -> Some 1
  | Op.Linear -> Some 0
  | _ -> None

let body u =
  match (u.op, u.src) with
  | Op.Call, b :: _ -> b
  | op, _ -> invalid_argf "%s is not a call" (Op.name op)

let rec ended_ranges u =
  let l = u.memos.ended_ranges_memo in
  if l != unset_ranges then l
  else
    let l =
      match u.op with
      | Op.End -> List.filter (fun r -> r.op = Op.Range) (drop 1 u.src)
      | Op.Backedge -> ( match u.src with _ :: loop :: _ -> [ loop ] | _ -> [])
      | op when Option.is_some (range_start op) ->
          drop (Option.get (range_start op)) u.src
      | Op.After -> List.concat_map ended_ranges (drop 1 u.src)
      | Op.Barrier -> List.concat_map ended_ranges u.src
      | _ -> []
    in
    (own_memos u).ended_ranges_memo <- l;
    l

let rec ranges u =
  memoized ~calls:Enter ~unset:unset_nodes
    ~get:(fun n -> n.memos.ranges_memo)
    ~set:(fun n r -> (own_memos n).ranges_memo <- r)
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
  match Axis_type.compare t0 t1 with
  | 0 -> List.compare Int.compare id0 id1
  | c -> c

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

let param_arg_of u =
  match u.arg with
  | Param p -> p
  | _ -> invalid_argf "%s has no ParamArg" (Op.name u.op)

let sink ?kernel us =
  v Op.Sink ~src:us
    ~arg:(match kernel with Some k -> Kernel k | None -> No_arg)


let broadcast u n =
  if n = 1 then u else v Op.Stack ~src:(List.init n (fun _ -> u))

let index u idxs =
  match (u.op, idxs) with
  | Op.Stack, [ ({ op = Op.Const; _ } as c) ] ->
      let i = Value.to_int (as_index (value c)) in
      let n = List.length u.src in
      let i = if i < 0 then n + i else i in
      if i < 0 || i >= n then
        invalid_argf "index %s of a stack of %d" (repr_const (value c)) n;
      List.nth u.src i
  | _ -> v Op.Index ~src:(u :: idxs)

let load u rest = v Op.Load ~src:(u :: rest)
let store ?gate p x = v Op.Store ~src:([ p; x ] @ Option.to_list gate)
let end_ u = function [] -> u | rs -> v Op.End ~src:(u :: rs)
let backedge u ~loop ~cond = v Op.Backedge ~src:[ u; loop; cond ]
let after u = function [] -> u | deps -> v Op.After ~src:(u :: deps)
let group = function [ u ] -> u | us -> after (v Op.Stack) us

let is_group u =
  match (u.op, u.src) with
  | Op.After, s :: _ -> s.op = Op.Stack && s.src = []
  | _ -> false

let is_join u =
  match (u.op, u.src) with
  | Op.After, _ :: (_ :: _ as deps) ->
      List.for_all (fun d -> d.op = Op.After) deps
  | _ -> false

let rec without_after u =
  if u.op = Op.After then without_after (first u.op u.src) else u

let bound_loads r =
  ignore (range_arg r);
  match r.src with
  | [ bound ] ->
      op_in_backward_slice_with_self ~calls:Skip (without_after bound)
        [ Op.Load ]
  | _ -> false

let known_at_launch r =
  ignore (range_arg r);
  match r.src with
  | [ bound ] ->
      Nodes.cardinal (ranges (without_after bound)) = 0 && not (bound_loads r)
  | _ -> false

let barrier u rest = v Op.Barrier ~src:(u :: rest)

let ins ?src ?dtype u i =
  let src = Option.value src ~default:u.src
  and dtype = Option.value dtype ~default:u.dtype in
  v Op.Ins ~src ~arg:(Code { code = i; dtype })

let cconst dt c = v Op.Cast ~src:[ const c ] ~arg:(Dtype dt)

let range ~axis_type ?(dtype = Dtype.Weak_int) end_ axis_id =
  v Op.Range ~src:[ sint_to_uop ~dtype end_ ] ~arg:(Range { axis_id; axis_type })

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
  memoized ~calls:Enter ~unset:unset_device
    ~get:(fun n -> n.memos.device_memo)
    ~set:(fun n d -> (own_memos n).device_memo <- d)
    ~compute:compute_device u

and compute_device u =
  let src0 () = first u.op u.src in
  match (u.op, u.arg) with
  | (Op.Param | Op.Buffer | Op.Alloc), Param p -> p.device
  | Op.Stage, Bufferize b -> b.device
  | Op.Stage, _ | Op.After, _ -> device (src0 ())
  | Op.Copy, Device d -> Some d
  | _ -> List.find_map device u.src

let on_disk u =
  match device u with
  | Some d -> String.starts_with ~prefix:"DISK" d
  | None -> false

let is_virtual u = Option.is_none (device u) || Dtype.mem u.dtype Dtype.weaks

let rec addrspace u =
  memoized ~calls:Enter ~unset:unset_addrspace
    ~get:(fun n -> n.memos.addrspace_memo)
    ~set:(fun n a -> (own_memos n).addrspace_memo <- a)
    ~compute:compute_addrspace u

(* A node of no value has no memory: it lives in the ALU. *)
and compute_addrspace u =
  match (u.op, u.arg) with
  | _ when Dtype.equal u.dtype Dtype.Void -> Some Dtype.Alu
  | (Op.Param | Op.Refparam | Op.Buffer | Op.Alloc), Param p -> Some p.addrspace
  | (Op.Special | Op.Range | Op.Const), _ | Op.Load, _ -> Some Dtype.Alu
  | Op.Binary, _ -> Some Dtype.Global
  | (Op.Index | Op.Cast | Op.After | Op.Reduce | Op.Store | Op.End), _ ->
      addrspace (first u.op u.src)
  | op, _ when Op.Set.mem op Op.Set.movement -> addrspace (first u.op u.src)
  | op, _
    when op = Op.Stack || op = Op.Wmma || Op.Set.mem op Op.Set.elementwise -> (
      match List.filter_map addrspace u.src with
      | a :: rest when List.for_all (( = ) a) rest -> Some a
      | _ -> None)
  | _ -> None

let rec has_buffer_identity ?(after_ok = false) u =
  match u.op with
  | Op.Reshape -> has_buffer_identity ~after_ok (first u.op u.src)
  | Op.After when after_ok -> has_buffer_identity ~after_ok (first u.op u.src)
  | op -> Op.Set.mem op Op.Set.defines

(* Movement *)

(* Each node walked keeps the storage found: the n stores into one buffer
   target a chain of n afters, which their storage bases then walk once in all,
   not once each. *)
let storage_base u =
  let strips n =
    match n.op with
    | Op.Detach | Op.Bitcast | Op.After -> true
    | op -> Op.Set.mem op Op.Set.movement
  in
  let rec walk path n =
    match n.memos.storage_base_memo with
    | Some b -> (b, path)
    | None -> if strips n then walk (n :: path) (first n.op n.src) else (n, path)
  in
  let b, path = walk [] u in
  let memo = Some b in
  List.iter (fun n -> (own_memos n).storage_base_memo <- memo) path;
  b

let needs_storage u =
  (not (is_virtual u))
  && ((storage_base u).op = Op.Alloc || not (has_buffer_identity u))

(* Data types *)

let bitcast x dt =
  if Dtype.mem x.dtype Dtype.weaks || Dtype.mem dt Dtype.weaks then
    invalid_argf "a bitcast needs committed types, not %s to %s"
      (repr_dtype x.dtype) (repr_dtype dt);
  if Dtype.equal x.dtype dt then x else v Op.Bitcast ~src:[ x ] ~arg:(Dtype dt)

let commit_dtype ?default_int x =
  if Dtype.equal x.dtype Dtype.Weak_int then
    Dtype.commit_int ?default_int (Value.to_z (vmin x)) (Value.to_z (vmax x))
  else Dtype.strong x.dtype

let element_size x =
  if Dtype.mem x.dtype Dtype.weaks then
    invalid_argf "a %s has no size" (repr_dtype x.dtype);
  Dtype.itemsize x.dtype

let contiguous x =
  if Dtype.mem x.dtype Dtype.weaks then x
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

(* Devices *)

let size_src = function None -> v Op.Stack | Some n -> int n

let copy_to_device u d =
  if is_disk_device d then
    invalid_arg "cannot copy to a disk; store into a disk buffer instead";
  if Dtype.mem u.dtype Dtype.weaks then
    invalid_argf "cannot store a weak %s" (repr_dtype u.dtype);
  v Op.Copy ~src:[ u ] ~arg:(Device d)

(* Storage *)

let unique = Atomic.make 0
let unique_num () = Atomic.fetch_and_add unique 1

let rec take_unique n =
  if n = max_int then invalid_arg "Ops.take_unique: no number is greater";
  let next = Atomic.get unique in
  if next <= n && not (Atomic.compare_and_set unique next (n + 1)) then
    take_unique n

let param_arg ?vmin_vmax ?name
    ?(addrspace = Dtype.Global) ?device ?(volatile = false)
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
    vmin_vmax;
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
  if Dtype.mem dt Dtype.weaks then
    invalid_argf "a %s cannot be stored" (repr_dtype dt)

let new_buffer ?slot ?phase d size dt =
  weak_storage dt;
  let slot = match slot with Some s -> s | None -> unique_num () in
  v Op.Buffer
    ~src:[ size_src (Some size) ]
    ~arg:(Param (param_arg ~slot ~device:d ?phase dt))

let set ?(ends = []) p x = after (first p.op p.src) [ end_ (store p x) ends ]

(* Variables *)

let variable ?(dtype = Dtype.Weak_int) name lo hi =
  v Op.Param ~src:[ size_src None ]
    ~arg:
      (Param
         (param_arg ~vmin_vmax:(lo, hi) ~name
            ~addrspace:Dtype.Alu ~slot:(-1) dtype))

let expr u =
  match (u.op, u.arg) with
  | (Op.Param | Op.Buffer), Param { name = Some n; _ } -> n
  | op, _ -> invalid_argf "%s has no name" (Op.name op)

(* Divisibility *)

let rec const_factor u : Bigint.t =
  match u.op with
  | Op.Const -> integer (value u)
  | Op.Add -> Bigint.gcd (const_factor (nth u 0)) (const_factor (nth u 1))
  | Op.Mul ->
      if (nth u 0).op = Op.Const then integer (value (nth u 0))
      else if (nth u 1).op = Op.Const then integer (value (nth u 1))
      else Bigint.one
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

(* Correctly rounded folds

   [sin], [log2], [exp2] and a float [pow] are computed on integers and rounded
   once, so that a folded constant is the same on every host: C libraries
   differ in the last place, and glibc's [log2], [exp2] and [pow] round some
   operands one unit off. *)

(* [n * 2^e] rounded to the nearest float, ties to even, for [n >= 0]: a
   subnormal below the least normal, an infinity past the greatest float. *)
let round_scaled n e =
  let top = Bigint.numbits n - 1 + e in
  if Bigint.sign n = 0 || top < -1076 then 0.
  else if top > 1024 then Float.infinity
  else
    (* [q] is the exponent of the float's last place, [drop] the bits of [n]
       below it. *)
    let q = Int.max (top - 52) (-1074) in
    let drop = q - e in
    let m =
      if drop <= 0 then Bigint.shift_left n (-drop)
      else
        let m = Bigint.shift_right n drop in
        let rest = Bigint.(n - shift_left m drop) in
        let half = Bigint.shift_left Bigint.one (drop - 1) in
        let c = Bigint.compare rest half in
        if c > 0 || (c = 0 && Bigint.(logand m one = one)) then Bigint.succ m
        else m
    in
    Float.ldexp (Bigint.to_float m) q

let round_signed n e =
  if Bigint.sign n < 0 then -.round_scaled (Bigint.neg n) e
  else round_scaled n e

(* [ziv f] is the float that the bounds of [f w] round to, for the first [w] of
   64, 128, 256, ... at which both round to one float. [f w] is [(n, err, e)],
   a value within [err * 2^e] of [n * 2^e], where [err] shrinks against [n] as
   [w] grows. It ends where the value is neither a float nor halfway between
   two, true of every value it is given here: a logarithm, a power of two and a
   power that are not exact are irrational. The bounds compare by their bits,
   so that [-0.] and [0.] differ. *)
let rec ziv ?(w = 64) f =
  let n, err, e = f w in
  let lo = round_signed Bigint.(n - err) e
  and hi = round_signed Bigint.(n + err) e in
  if Int64.equal (Int64.bits_of_float lo) (Int64.bits_of_float hi) then lo
  else ziv ~w:(2 * w) f

(* [atanh_fixed num den w] is [atanh (num / den) * 2^w] for [0 <= num / den <=
   1/3], truncated, and a bound on its error in units: the sum of [s^(2k+1) /
   (2k+1)], each power truncated from the last. A power's error stays below
   [1 / (1 - s^2)], under 2 units, a term's below 3, and the terms dropped
   once a power is 0 sum below 3. *)
let atanh_fixed num den w =
  let num2 = Bigint.(num * num) and den2 = Bigint.(den * den) in
  let rec sum acc t k =
    if Bigint.sign t = 0 then (acc, (3 * k) + 3)
    else
      let odd = (2 * k) + 1 in
      let acc = Bigint.(acc + div t (of_int odd)) in
      sum acc Bigint.(div (t * num2) den2) (k + 1)
  in
  sum Bigint.zero Bigint.(div (shift_left num w) den) 0

(* atanh (1/3) = ln 2 / 2, and its error *)
let half_ln2 w = atanh_fixed Bigint.one (Bigint.of_int 3) w

(* [log2_fixed m e w] is [log2 (m * 2^e)], [m > 0], as [ziv] takes it, at [w]
   bits after the point: [m * 2^e] is [r * 2^k] with [r] in [[1/sqrt 2, sqrt
   2)], and [log2 r] is [atanh s / atanh (1/3)] for [s = (r - 1) / (r + 1)],
   [|s| < 0.172]. The quotient's error is below 3 units of each operand's error
   and 1 of its truncation. *)
let log2_fixed m e w =
  let nb = Bigint.numbits m in
  (* [m / 2^(nb - 1)], in [[1, 2)], is past sqrt 2 where [m^2 >= 2^(2 nb - 1)]:
     then [r] is half of it. *)
  let up = Bigint.(geq (m * m) (shift_left one (Int.pred (Int.add nb nb)))) in
  let k = if up then nb else nb - 1 in
  let base = Bigint.shift_left Bigint.one k in
  let d = Bigint.(m - base) and s = Bigint.(m + base) in
  let a, a_err = atanh_fixed (Bigint.abs d) s w in
  let b, b_err = half_ln2 w in
  let q = Bigint.(div (shift_left a w) b) in
  let q = if Bigint.sign d < 0 then Bigint.neg q else q in
  ( Bigint.(shift_left (of_int (Int.add e k)) w + q),
    Bigint.of_int ((3 * (a_err + b_err)) + 2),
    -w )

(* [exp2_fixed z z_err w] is [2^(z * 2^-w)] as [ziv] takes it, where [z] is
   within [z_err] units of the exponent: [2^k * e^t] for [k] the floor of the
   exponent and [t] its fraction times ln 2, [0 <= t < 0.7], by its series,
   each term truncated from the last. With [t]'s error [t_err], a term's error
   stays below [t_err + 3] and the terms dropped sum below [2 (t_err + 3)]; an
   exponent's error of one unit moves the result, below [2^(k+1)], by under 2
   units. [k] saturates at [+-2000], where the power rounds to an infinity or
   0 as it does past them. *)
let exp2_fixed z z_err w =
  let k = Bigint.shift_right z w in
  let frac = Bigint.(z - shift_left k w) in
  let h, h_err = half_ln2 w in
  let t = Bigint.(shift_right (frac * shift_left h 1) w) in
  let t_err = (2 * h_err) + 1 in
  let one = Bigint.shift_left Bigint.one w in
  let rec series acc term n =
    let next = n + 1 in
    let term = Bigint.(div (shift_right (term * t) w) (of_int next)) in
    if Bigint.sign term = 0 then (acc, n)
    else series Bigint.(acc + term) term (n + 1)
  in
  let sum, terms = series one one 0 in
  let k = Bigint.(to_int (max (of_int (-2000)) (min (of_int 2000) k))) in
  let series_err = ((terms + 3) * (t_err + 3)) + 1 in
  let err = Bigint.(of_int series_err + shift_left z_err 1) in
  (sum, err, k - w)

(* [x], a finite nonzero float, as [(m, e)] with [x = m * 2^e] *)
let mantissa x =
  let f, e = Float.frexp x in
  (Bigint.of_float (Float.ldexp f 53), e - 53)

(* [log2 x] for [x > 0], of an integer past the greatest float from its value,
   as for any other operand from the float nearest it. *)
let py_log2 (x : Dtype.value) =
  match x with
  | `Float f when not (Float.is_finite f) -> f
  | _ ->
      let f = Value.to_float x in
      let m, e = if Float.is_finite f then mantissa f else (Value.to_z x, 0) in
      if Bigint.popcount m = 1 then Float.of_int (e + Bigint.numbits m - 1)
      else ziv (log2_fixed m e)

let py_exp2 (x : Dtype.value) : Dtype.value =
  match x with
  | (`Bool _ | `Int _) when Bigint.geq (Value.to_z x) Bigint.zero ->
      `Int (Bigint.shift_left Bigint.one (Value.to_int x))
  | _ ->
      let f = Value.to_float x in
      (* past these, 2^x rounds to an infinity or to 0 *)
      if not (Float.is_finite f) then `Float (Float.pow 2. f)
      else if f > 1024. then `Float Float.infinity
      else if f < -1076. then `Float 0.
      else if Float.is_integer f then `Float (Float.ldexp 1. (Float.to_int f))
      else
        let m, e = mantissa f in
        (* [x * 2^w], exact where [x] has no bits below [2^-w], else floored *)
        let fixed w =
          let s = e + w in
          if s >= 0 then exp2_fixed (Bigint.shift_left m s) Bigint.zero w
          else exp2_fixed (Bigint.shift_right m (-s)) Bigint.one w
        in
        `Float (ziv fixed)

(* The integer whose [q]th power is [n >= 0], if there is one. *)
let iroot n q =
  let rec search lo hi =
    if Bigint.geq lo hi then lo
    else
      let mid = Bigint.(shift_right (lo + hi) 1) in
      if Bigint.lt (Bigint.pow mid q) n then search (Bigint.succ mid) hi
      else search lo mid
  in
  let top = Bigint.shift_left Bigint.one ((Bigint.numbits n / q) + 1) in
  let r = search Bigint.zero top in
  if Bigint.equal (Bigint.pow r q) n then Some r else None

(* [x] as [(m, e)], [x = m * 2^e] with [m] odd, for a finite [x <> 0]. *)
let odd_mantissa x =
  let m, e = mantissa x in
  let z = Bigint.trailing_zeros m in
  (Bigint.shift_right m z, e + z)

(* [|x|^y] for finite [x <> 0] and [y <> 0], where [x < 0] only with an integer
   [y]. With [|x| = mx * 2^ex] and [y = a / 2^tb], [mx] and [a] odd or [tb] 0:

   - a power of two, [mx = 1], is [2^(ex y)], exact where [ex y] is an
     integer;
   - [ziv] ends unless the power is halfway between two floats, an odd integer
     of 54 bits times a power of two: so a power [r^a * 2^(ex y)] for
     [mx = r^(2^tb)], [a > 0] and [r^a] below [2^54] is computed exactly. Any
     other power is a float, or no float and no midpoint, or irrational;
   - [ziv] computes it as [2^(y log2 |x|)], the logarithm to [log2 |y|] more
     bits. Where the exponent is past 1100 in magnitude, the power is an
     infinity or 0. *)
let float_pow fx fy =
  let mx, ex = odd_mantissa (Float.abs fx) in
  let my, ey = odd_mantissa (Float.abs fy) in
  let a = if fy < 0. then Bigint.neg my else my in
  let a = Bigint.shift_left a (Int.max 0 ey) in
  let tb = Int.max 0 (-ey) in
  let ex_a = Bigint.(of_int ex * a) in
  let exact_root () =
    let nb = Bigint.numbits mx and cap = 54 lsl tb in
    if
      tb > 6 || Bigint.sign a <= 0 || ex mod (1 lsl tb) <> 0
      || Bigint.(geq (of_int (Int.pred nb) * a) (of_int cap))
    then None
    else Option.map (fun r -> (r, Bigint.to_int a)) (iroot mx (1 lsl tb))
  in
  if Bigint.equal mx Bigint.one && Bigint.trailing_zeros ex_a >= tb then
    let k = Bigint.shift_right ex_a tb in
    if Bigint.gt k (Bigint.of_int 1023) then Float.infinity
    else if Bigint.lt k (Bigint.of_int (-1076)) then 0.
    else Float.ldexp 1. (Bigint.to_int k)
  else
    match exact_root () with
    | Some (r, a) -> round_scaled (Bigint.pow r a) (a * (ex asr tb))
    | None ->
        let w0 = 64 in
        let l, l_err, _ = log2_fixed mx ex w0 in
        (* past 1100 in magnitude by a lower bound, the exponent saturates *)
        let low = Bigint.(abs l - l_err) in
        let bound = Bigint.shift_left (Bigint.of_int 1100) (w0 + tb) in
        if Bigint.gt Bigint.(low * abs a) bound then
          if Bigint.sign l = Bigint.sign a then Float.infinity else 0.
        else
          let grow = Int.max 0 (Bigint.numbits a - tb) in
          ziv (fun w ->
              let wl = w + grow in
              let l, l_err, _ = log2_fixed mx ex wl in
              let z = Bigint.(shift_right (l * a) tb) in
              let z_err = Bigint.(shift_right (abs a * l_err) tb + of_int 2) in
              exp2_fixed z z_err wl)

let is_number u i =
  i = 1
  &&
  match (u.op, u.src) with
  | (Op.Shl | Op.Shr), _ -> true
  | Op.Pow, [ b; e ] ->
      Dtype.is_int b.dtype && Dtype.equal e.dtype Dtype.Weak_int
  | _ -> false

(* Whether [b ** e] can be computed exactly: a base of 0, 1 or -1 stays small at
   any exponent, and otherwise each step of a power adds its base's bits. *)
let exact_power b e =
  let b = Bigint.abs (Value.to_z b) and e = Value.to_z e in
  Bigint.leq b Bigint.one || Bigint.sign e < 0
  || Bigint.leq
       (Bigint.mul e (Bigint.of_int (Bigint.numbits b)))
       (Bigint.of_int max_exact_shift)

let py_pow (x : Dtype.value) (y : Dtype.value) : Dtype.value =
  match (x, y) with
  | (`Bool _ | `Int _), (`Bool _ | `Int _) ->
      let b = Value.to_z x and e = Value.to_z y in
      if Bigint.geq e Bigint.zero && not (Bigint.leq (Bigint.abs b) Bigint.one) then (
        if not (exact_power x y) then
          invalid_argf "a power %s ** %s, of no width, has no value"
            (Bigint.to_string b) (Bigint.to_string e);
        `Int (Bigint.pow b (Bigint.to_int e)))
      else if Bigint.geq e Bigint.zero then
        (* 0, 1 and -1 stay small at any exponent *)
        `Int
          (if Bigint.sign b = 0 then
             if Bigint.sign e = 0 then Bigint.one else Bigint.zero
           else if Bigint.divisible e (Bigint.of_int 2) then Bigint.one
           else b)
      else if Bigint.equal (Bigint.abs b) Bigint.one then
        (* a negative power truncates toward 0, which only 1 and -1 escape *)
        `Int (if Bigint.divisible e (Bigint.of_int 2) then Bigint.one else b)
      else `Int Bigint.zero
  | _ ->
      let fx = Value.to_float x and fy = Value.to_float y in
      (* C's pow is Python's, except that a zero to a negative power raises in
         Python, which the folding catches as an infinity. Where an operand is
         a zero, an infinity or a NaN, and for a negative [x] to a power that is
         not an integer, C's pow gives the value IEEE states. *)
      if Float.equal fx 0. && fy < 0. then `Float Float.infinity
      else if
        Float.equal fx 0. || Float.equal fy 0.
        || (not (Float.is_finite fx && Float.is_finite fy))
        || (fx < 0. && not (Float.is_integer fy))
      then `Float (Float.pow fx fy)
      else
        let p = float_pow fx fy in
        let odd = fx < 0. && not (Float.is_integer (fy /. 2.)) in
        `Float (if odd then -.p else p)

(* pi/2 rounded to [half_pi_bits] bits after the point. Its error times the
   greatest quotient of a float by pi/2, about 2^1024, is far below the smallest
   remainder of a float, about 2^-62. *)
let half_pi_bits = 1200
let half_pi =
  Bigint.of_string
    "0x1921fb54442d18469898cc51701b839a252049c1114cf98e804177d4c76273644a\
     29410f31c6809bbdf2a33679a748636605614dbe4be286e9fc26adadaa3848bc90\
     b6aecc4bcfd8de89885d34c6fdad617feb96de80d6fdbdc70d7f6b5133f4b5d3e4\
     822f8963fcc9250cca3d9c8b67b8400f97142c77e0b31b4906c38aba734d22c7f5\
     1fa499ebf06caba47b9475b2c38c5e6ac410b"

(* The bits [sine] keeps of a reduced argument. *)
let sine_precision = 192

(* [sine x] is [sin x] for a finite [x], correctly rounded but in cases far past
   [sine_precision]. It is computed on integers, since C libraries differ:
   mingw-w64's sine reduces by x87's 66-bit pi, and gave [sin 1e300] the wrong
   sign. [|x|] is reduced exactly to [r = |x| - k pi/2], [|r| <= pi/4], whose
   sine or cosine, by [k]'s quadrant, is the sum of its series in fixed point.
   Below 2^-26, [sin x] rounds to [x]. *)
let sine x =
  let a = Float.abs x in
  if a < 0x1p-26 then x
  else
    let f, e = Float.frexp a in
    let m = Bigint.of_float (Float.ldexp f 53) in
    let scaled = Bigint.shift_left m (e - 53 + half_pi_bits) in
    let k = Bigint.(div (scaled + shift_right half_pi 1) half_pi) in
    let rem = Bigint.(scaled - (k * half_pi)) in
    let drop = Int.max 0 (Bigint.numbits (Bigint.abs rem) - sine_precision) in
    let p = half_pi_bits - drop in
    let r = Bigint.shift_right (Bigint.abs rem) drop in
    let r2 = Bigint.(shift_right (r * r) p) in
    (* [t - t1 + t2 - ...], where [t(i+1)] is [ti r^2 / ((j+1)(j+2))] and [j]
       counts up by 2 from [j]. *)
    let rec series sum t j =
      let divisor = Bigint.of_int ((j + 1) * (j + 2)) in
      let t = Bigint.(div (shift_right (t * r2) p) divisor) in
      if Bigint.equal t Bigint.zero then sum
      else
        let sum = if j mod 4 < 2 then Bigint.(sum - t) else Bigint.(sum + t) in
        series sum t (j + 2)
    in
    let one = Bigint.shift_left Bigint.one p in
    let sin_r () = series r r 1 and cos_r () = series one one 0 in
    (* sin |x| is sin r, cos r, -sin r, -cos r by quadrant, and sin r has the
       sign of [rem]. *)
    let quadrant = Bigint.to_int (Bigint.logand k (Bigint.of_int 3)) in
    let y =
      round_scaled (if quadrant land 1 = 0 then sin_r () else cos_r ()) (-p)
    in
    let negative =
      quadrant >= 2 <> (quadrant land 1 = 0 && Bigint.sign rem < 0)
    in
    if negative <> Float.sign_bit x then -.y else y

let int_operands op (x : Dtype.value) (y : Dtype.value) =
  match (x, y) with
  | `Float _, _ | _, `Float _ -> invalid_argf "%s needs integers" (Op.name op)
  | _ -> (Value.to_z x, Value.to_z y)

(* A shift's operands, its count as an [int]. *)
let shift_operands op x y =
  let a, b = int_operands op x y in
  if Bigint.sign b < 0 then
    invalid_argf "a shift by a negative count, %s, has no value" (Bigint.to_string b);
  (a, shift_count b)

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
          if Value.( < ) (`Int Bigint.zero) x then `Float (py_log2 x)
          else if Value.( = ) x (`Int Bigint.zero) then `Float Float.neg_infinity
          else `Float Dtype.nan)
  | Op.Exp2 -> unary py_exp2
  | Op.Sqrt ->
      unary (fun x ->
          if Value.( <= ) (`Int Bigint.zero) x then
            `Float (Float.sqrt (Value.to_float x))
          else `Float Dtype.nan)
  | Op.Recip ->
      unary (fun x ->
          let f = Value.to_float x in
          if Float.equal f 0. then `Float (Float.copy_sign Float.infinity f)
          else `Float (1. /. f))
  | Op.Sin ->
      unary (fun x ->
          let f = Value.to_float x in
          `Float (if Float.is_finite f then sine f else Dtype.nan))
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
          if Bigint.sign a <> 0 && k > max_exact_shift then
            invalid_argf "a left shift by %s, of no width, has no value"
              (Bigint.to_string (Value.to_z y));
          `Int (Bigint.shift_left a k))
  | Op.Max -> binary Value.max
  | Op.Truncmod -> binary (divide `Remainder ~toward_zero:true)
  | Op.Truncdiv -> binary (divide `Quotient ~toward_zero:true)
  | Op.Div -> binary (divide `Quotient ~toward_zero:false)
  | Op.Mod -> binary (divide `Remainder ~toward_zero:false)
  | Op.Mulacc -> (
      match args with
      | [ x; y; z ] -> Value.( + ) (Value.( * ) x y) z
      | _ -> invalid_arg "MULACC takes three operands")
  | op -> invalid_argf "%s is not computed on constants" (Op.name op)

(* A multiply-add of floats rounds once. Truncated to a dtype below float64, its
   operands have at most 24 bits, so their product is exact in a double, and the
   sum is rounded to odd there, which the truncation then rounds as the exact
   value. Otherwise, and for a weak float, a double, it is the double nearest
   the exact value. *)
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

(* [b] to the power [e], [e >= 0], modulo [2^bits]: the low bits repeated
   products keep, without the power's full width. *)
let pow_wrapped b e bits =
  let m = Bigint.shift_left Bigint.one bits in
  let rec go e =
    if Bigint.equal e Bigint.zero then Bigint.one
    else
      let h = go (Bigint.shift_right e 1) in
      let h = Bigint.rem (Bigint.mul h h) m in
      if Bigint.divisible e (Bigint.of_int 2) then h
      else Bigint.rem (Bigint.mul h b) m
  in
  go e

let exec_alu ?(truncate_output = true) op dt (args : Dtype.const list) :
    Dtype.const =
  let truncate (x : Dtype.value) : Dtype.const =
    if
      truncate_output
      && not (Dtype.equal dt Dtype.Void || Dtype.mem dt Dtype.weaks)
    then (Dtype.truncate dt x :> Dtype.const)
    else (x :> Dtype.const)
  in
  (* A selection and a negation convert nothing, so a NaN keeps its bits, a
     signalling one's included, where a truncation would quiet it. *)
  let move (x : Dtype.value) : Dtype.const =
    match x with
    | `Float v when truncate_output && Float.is_nan v && Dtype.is_float dt ->
        Dtype.const dt x
    | x -> truncate x
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
        | #Dtype.value as v -> move v)
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
          (* a power truncated to a committed integer keeps its low bits alone,
             so that a large exponent folds in its bits' count of steps *)
          | Op.Pow, [ ((`Int _ | `Bool _) as b); ((`Int _ | `Bool _) as e) ]
            when truncate_output
                 && Dtype.mem dt Dtype.ints
                 && Bigint.geq (Value.to_z e) Bigint.zero ->
              `Int
                (pow_wrapped (Value.to_z b) (Value.to_z e)
                   (8 * Dtype.itemsize dt))
          (* a left shift's value is 0 once its count passes the width, and
             a huge count leaves the exact shift out of memory *)
          | Op.Shl, [ (`Int _ | `Bool _); ((`Int _ | `Bool _) as k) ]
            when truncate_output
                 && Dtype.mem dt Dtype.ints
                 && Bigint.geq (Value.to_z k) (Bigint.of_int (Dtype.bitsize dt)) ->
              `Int Bigint.zero
          | _ -> python_alu op args
        in
        let v =
          match v with
          | v when is_nan v && not (List.exists is_nan args) -> `Float Dtype.nan
          | v -> v
        in
        if Op.equal op Op.Neg then move v else truncate v

(* Patterns *)

module Upat = struct
  type node = t

  type t = {
    ops : Op.Set.t option;
    dtypes : Dtype.t list option;
    p_arg : arg option;
    p_name : string option;
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
    nameless : bool;  (* Names no node: a match is a test. *)
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
       && Option.equal Op.Set.equal p.ops q.ops
       && Option.equal String.equal p.p_name q.p_name
       && Option.equal (List.equal Dtype.equal) p.dtypes q.dtypes
       && Option.equal equal_arg p.p_arg q.p_arg

  let make ?ops ?dtypes ?in_src ?(is_any = false) ?(allow_any_len = false) ?arg
      ?name ?early_reject ?(cached = false) () =
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
              match Option.map Op.Set.to_list p.ops with
              | Some [ o ] when not (List.mem o acc) -> acc @ [ o ]
              | _ -> acc)
            [] firsts
    in
    {
      ops = Option.map Op.Set.of_list ops;
      dtypes;
      p_arg = arg;
      p_name = name;
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
        && (match p_src with
           | None -> true
           | Some (Tuples [ tuple ]) -> List.for_all (fun q -> q.unique) tuple
           | Some (Repeat q) -> q.unique
           | Some (Tuples _) -> false);
      nameless =
        Option.is_none name
        &&
        match in_src with
        | None -> true
        | Some (Fixed l | Perm l) -> List.for_all (fun q -> q.nameless) l
        | Some (Each q) -> q.nameless;
    }

  let only ~src ~perm ~each =
    match (src, perm, each) with
    | None, None, None -> None
    | Some l, None, None -> Some (Fixed l)
    | None, Some l, None -> Some (Perm l)
    | None, None, Some p -> Some (Each p)
    | _ -> invalid_arg "a pattern takes one of src, perm and each"

  let v ?op ?dtype ?src ?perm ?each ?allow_any_len ?arg ?name ?early_reject () =
    make
      ?ops:(Option.map Op.Set.to_list op)
      ?dtypes:dtype ?in_src:(only ~src ~perm ~each) ?allow_any_len ?arg ?name
      ?early_reject ()

  let op ?dtype ?src ?perm ?each ?allow_any_len ?arg ?name ?early_reject o =
    make ~ops:[ o ] ?dtypes:dtype ?in_src:(only ~src ~perm ~each) ?allow_any_len
      ?arg ?name ?early_reject ()

  let wild = make ()
  let var ?dtype name = make ?dtypes:dtype ~name ~cached:true ()

  let cvar ?dtype ?arg name =
    make ~ops:[ Op.Const ] ?dtypes:dtype
      ?arg:(Option.map (fun c -> Const c) arg)
      ~name ~cached:true ()

  let const ?dtype c = make ~ops:[ Op.Const ] ?dtypes:dtype ~arg:(Const c) ()
  let any ps = make ~in_src:(Fixed ps) ~is_any:true ()

  let named name p =
    make
      ?ops:(Option.map Op.Set.to_list p.ops)
      ?dtypes:p.dtypes ?in_src:p.in_src
      ~allow_any_len:(not p.strict_length) ?arg:p.p_arg ~name
      ?early_reject:p.custom_early_reject ()

  let dtype p = match p.dtypes with Some (dt :: _) -> dt | _ -> Dtype.Void

  include Make_elementwise (struct
    type nonrec t = t

    let dtype = dtype

    let alu p o rest =
      let srcs = p :: rest in
      let dtypes =
        if Op.Set.mem o Op.Set.comparison then Some [ Dtype.Bool ]
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

  exception No_match

  (* [store] extended with what [p] names in [u]. Raises [No_match] if the name
     is bound to another node. *)
  let rec bound_in n u store = function
    | Empty -> Bound (n, u, store)
    | Bound (m, b, rest) ->
        if not (String.equal m n) then bound_in n u store rest
        else if b == u then store
        else raise_notrace No_match

  (* Patterns that each name a node in at most one way, whose namings, in order,
     are [p]'s namings in the order {!match_} enumerates them, at most [limit]
     patterns; [None] past it, or when a pattern that repeats over every source
     has several namings, whose number is the number of sources. A nameless
     pattern is one test. *)
  let alternatives ~limit p =
    let exception Too_many in
    let check l = if List.length l > limit then raise_notrace Too_many else l in
    let rec alts p =
      if p.unique || p.nameless then [ p ]
      else if p.is_any then
        match p.p_src with
        | Some (Tuples [ choices ]) -> check (List.concat_map alts choices)
        | _ -> []
      else
        match p.p_src with
        | Some (Tuples tuples) ->
            check
              (List.concat_map
                 (fun tuple ->
                   List.map
                     (fun tuple ->
                       {
                         p with
                         p_src = Some (Tuples [ tuple ]);
                         unique = true;
                       })
                     (product tuple))
                 tuples)
        | Some (Repeat q) when q.nameless -> [ { p with unique = true } ]
        | Some (Repeat _) | None -> raise_notrace Too_many
    (* Each choice of an alternative per pattern, the first pattern's choices
       outermost. *)
    and product = function
      | [] -> [ [] ]
      | p :: rest ->
          let rests = product rest in
          check
            (List.concat_map
               (fun a -> List.map (fun r -> a :: r) rests)
               (alts p))
    in
    match alts p with l -> Some l | exception Too_many -> None

  (* Compiling

     A pattern is compiled on its rule's first try into closures that test only
     the fields it has: an operation, a type, an argument, a fixed tuple of
     sources or their orderings, a pattern every source matches. Matching a
     node runs them and never inspects the pattern again. A test is [None] when
     every node passes it, so that a source matched by any node is not visited.
     A rule's own operation is not tested: the matcher only tries the rules of
     the node's operation. *)

  type test = (node -> bool) option

  let holds (t : test) u = match t with None -> true | Some t -> t u

  (* An operation among several is a bit of two words, by its position. *)
  type op_test = Any_op | One_op of Op.t | Among of int * int

  let op_width = Sys.int_size - 1

  let among low high (o : Op.t) =
    let i = Op.to_int o in
    if i < op_width then (low lsr i) land 1 = 1
    else (high lsr (i - op_width)) land 1 = 1

  let op_test ~root p =
    match p.ops with
    | _ when root -> Any_op
    | None -> Any_op
    | Some ops -> (
        match Op.Set.to_list ops with
        | [ o ] -> One_op o
        | os ->
            let bit (low, high) o =
              let i = Op.to_int o in
              if i < op_width then (low lor (1 lsl i), high)
              else (low, high lor (1 lsl (i - op_width)))
            in
            let low, high = List.fold_left bit (0, 0) os in
            Among (low, high))

  let rec dtype_mem (dt : Dtype.t) = function
    | [] -> false
    | d :: ds -> dt = d || dtype_mem dt ds

  (* The test of a node's type and argument. *)
  let fields_test p : test =
    let dtype =
      match p.dtypes with
      | None -> None
      | Some [ (d : Dtype.t) ] -> Some (fun u -> u.dtype = d)
      | Some ds -> Some (fun u -> dtype_mem u.dtype ds)
    in
    match (dtype, p.p_arg) with
    | dtype, None -> dtype
    | None, Some a -> Some (fun u -> arg_matches a u.arg)
    | Some d, Some a -> Some (fun u -> d u && arg_matches a u.arg)

  (* The test of a node, from the tests of its operation, of its type and
     argument, and of its sources. *)
  let node_test op (fields : test) (srcs : (node list -> bool) option) : test =
    match (op, fields, srcs) with
    | Any_op, f, None -> f
    | Any_op, None, Some s -> Some (fun u -> s u.src)
    | Any_op, Some f, Some s -> Some (fun u -> f u && s u.src)
    | One_op o, None, None -> Some (fun u -> u.op = o)
    | One_op o, Some f, None -> Some (fun u -> u.op = o && f u)
    | One_op o, None, Some s -> Some (fun u -> u.op = o && s u.src)
    | One_op o, Some f, Some s -> Some (fun u -> u.op = o && f u && s u.src)
    | Among (l, h), None, None -> Some (fun u -> among l h u.op)
    | Among (l, h), Some f, None -> Some (fun u -> among l h u.op && f u)
    | Among (l, h), None, Some s -> Some (fun u -> among l h u.op && s u.src)
    | Among (l, h), Some f, Some s ->
        Some (fun u -> among l h u.op && f u && s u.src)

  let rec exactly ts us =
    match (ts, us) with
    | [], [] -> true
    | t :: ts, u :: us -> holds t u && exactly ts us
    | _ -> false

  let rec at_least ts us =
    match (ts, us) with
    | [], _ -> true
    | t :: ts, u :: us -> holds t u && at_least ts us
    | _ :: _, [] -> false

  (* The test of sources matched in order by [ts], exactly as many if
     [strict]. *)
  let tuple_test ~strict ts =
    match (strict, ts) with
    | false, [] -> None
    | true, [] -> Some (function [] -> true | _ :: _ -> false)
    | true, [ None ] -> Some (function [ _ ] -> true | _ -> false)
    | true, [ Some t ] -> Some (function [ a ] -> t a | _ -> false)
    | false, [ None ] -> Some (function _ :: _ -> true | [] -> false)
    | false, [ Some t ] -> Some (function a :: _ -> t a | [] -> false)
    | true, [ None; None ] -> Some (function [ _; _ ] -> true | _ -> false)
    | true, [ Some t; None ] -> Some (function [ a; _ ] -> t a | _ -> false)
    | true, [ None; Some t ] -> Some (function [ _; b ] -> t b | _ -> false)
    | true, [ Some t; Some t' ] ->
        Some (function [ a; b ] -> t a && t' b | _ -> false)
    | true, ts -> Some (exactly ts)
    | false, ts -> Some (at_least ts)

  let rec any_holds ts us =
    match ts with [] -> false | t :: ts -> t us || any_holds ts us

  let any_test = function
    | [] -> Some (fun _ -> false)
    | ts when List.exists Option.is_none ts -> None
    | [ Some t; Some t' ] -> Some (fun u -> t u || t' u)
    | ts -> Some (any_holds (List.filter_map Fun.id ts))

  (* The test of the sources of [p]'s nodes: each tuple in turn, or each source
     by one pattern. *)
  let rec sources_test p =
    let strict = p.strict_length in
    match p.p_src with
    | None -> None
    | Some (Tuples [ tuple ]) ->
        tuple_test ~strict (List.map (test ~root:false) tuple)
    | Some (Tuples tuples) ->
        let tuple t = tuple_test ~strict (List.map (test ~root:false) t) in
        any_test (List.map tuple tuples)
    | Some (Repeat q) -> Option.map List.for_all (test ~root:false q)

  (* Whether a node matches [p], names aside. *)
  and test ~root p : test =
    if p.is_any then
      match p.p_src with
      | Some (Tuples [ choices ]) ->
          any_test (List.map (test ~root:false) choices)
      | _ -> Some (fun _ -> false)
    else node_test (op_test ~root p) (fields_test p) (sources_test p)

  (* Names

     A pattern that names a node in at most one way names it at a path of
     sources from the node it matches. A path through a pattern repeated over
     every source goes through the first source that has the rest of the path: a
     node whose sources a repeat matches has its names only if it has one. *)

  type step = Src of int | First_with

  let rec nth_src i = function
    | [] -> raise_notrace No_match
    | u :: us -> if i = 0 then u else nth_src (i - 1) us

  let rec first_with at = function
    | [] -> raise_notrace No_match
    | u :: us -> (
        match at u with v -> v | exception No_match -> first_with at us)

  (* The node at [path] from a node. Raises [No_match] if it has none. *)
  let rec at = function
    | [] -> Fun.id
    | [ Src i ] -> fun u -> nth_src i u.src
    | Src i :: path ->
        let at = at path in
        fun u -> at (nth_src i u.src)
    | First_with :: path ->
        let at = at path in
        fun u -> first_with at u.src

  let fallible path = List.mem First_with path

  let rec first_at ats u =
    match ats with
    | [] -> raise_notrace No_match
    | at :: ats -> (
        match at u with v -> v | exception No_match -> first_at ats u)

  (* The node at the first of [paths] a node has. *)
  let at_any = function
    | [ path ] -> at path
    | paths ->
        let ats = List.map at paths in
        fun u -> first_at ats u

  (* Whether the nodes at [ats] from [u] that exist are one node. *)
  let rec agree_from v u = function
    | [] -> true
    | at :: ats -> (
        match at u with
        | w -> w == v && agree_from v u ats
        | exception No_match -> agree_from v u ats)

  let rec agree u = function
    | [] -> true
    | at :: ats -> (
        match at u with
        | v -> agree_from v u ats
        | exception No_match -> agree u ats)

  (* Whether the nodes at [at] from the sources [us] that exist are one node. *)
  let rec agree_over_from v at = function
    | [] -> true
    | u :: us -> (
        match at u with
        | w -> w == v && agree_over_from v at us
        | exception No_match -> agree_over_from v at us)

  let rec agree_over at = function
    | [] -> true
    | u :: us -> (
        match at u with
        | v -> agree_over_from v at us
        | exception No_match -> agree_over at us)

  (* [occurrences]' names, each with its paths in order, in the order of their
     first occurrences. *)
  let by_name occurrences =
    List.fold_left
      (fun groups (n, path) ->
        if List.mem_assoc n groups then
          List.map
            (fun (m, paths) ->
              if String.equal m n then (m, paths @ [ path ]) else (m, paths))
            groups
        else groups @ [ (n, [ path ]) ])
      [] occurrences

  (* [t] and the test that a name used more than once in [occurrences] names one
     node. *)
  let with_agreement (t : test) occurrences : test =
    let agreement (_, paths) =
      match paths with
      | [ _ ] -> None
      | [ p; p' ] when not (fallible p || fallible p') ->
          let a = at p and a' = at p' in
          Some (fun u -> a u == a' u)
      | paths ->
          let ats = List.map at paths in
          Some (fun u -> agree u ats)
    in
    match (t, List.filter_map agreement (by_name occurrences)) with
    | t, [] -> t
    | None, [ c ] -> Some c
    | Some t, [ c ] -> Some (fun u -> t u && c u)
    | t, cs -> Some (fun u -> holds t u && List.for_all (fun c -> c u) cs)

  (* The test of [p], which names nodes in at most one way, and the paths of its
     names in the order a match binds them, a node's before its sources'. The
     test checks that a repeated pattern names one node on every source; that a
     name used twice elsewhere names one node is left to the caller
     ({!with_agreement}). *)
  let rec unique_parts ~root p =
    if p.nameless then (test ~root p, [])
    else
      let own = match p.p_name with None -> [] | Some n -> [ (n, []) ] in
      let node srcs = node_test (op_test ~root p) (fields_test p) srcs in
      match p.p_src with
      | None -> (node None, own)
      | Some (Tuples [ tuple ]) ->
          let parts = List.map (unique_parts ~root:false) tuple in
          let srcs = tuple_test ~strict:p.strict_length (List.map fst parts) in
          let under i (_, occurrences) =
            List.map (fun (n, path) -> (n, Src i :: path)) occurrences
          in
          (node srcs, own @ List.concat (List.mapi under parts))
      | Some (Repeat q) ->
          let t, occurrences = unique_parts ~root:false q in
          let each = with_agreement t occurrences in
          let ats =
            List.map (fun (_, paths) -> at_any paths) (by_name occurrences)
          in
          let srcs us =
            (match each with None -> true | Some t -> List.for_all t us)
            && List.for_all (fun at -> agree_over at us) ats
          in
          ( node (Some srcs),
            own
            @ List.map (fun (n, path) -> (n, First_with :: path)) occurrences )
      | Some (Tuples _) -> assert false

  (* The node [occurrences] names [name] in [u]. *)
  let rec find_name occurrences u name =
    match occurrences with
    | [] -> invalid_argf "the pattern names no %s" name
    | (n, at) :: rest -> (
        if not (String.equal n name) then find_name rest u name
        else
          match at u with v -> v | exception No_match -> find_name rest u name)

  let no_names name = invalid_argf "the pattern names no %s" name

  (* [p], which names nodes in at most one way, as a test of a node and the
     function of a node that passes it to the nodes it names. *)
  let compile ~root p =
    let t, occurrences = unique_parts ~root p in
    let names =
      match occurrences with
      | [] -> fun _ -> no_names
      | occurrences ->
          let occurrences =
            List.map (fun (n, path) -> (n, at path)) occurrences
          in
          fun u name -> find_name occurrences u name
    in
    (with_agreement t occurrences, names)

  (* [store] extended with what [occurrences] names in [u]. Raises [No_match] if
     a name is bound to another node. *)
  let rec bind_all occurrences u store =
    match occurrences with
    | [] -> store
    | (n, at) :: rest -> (
        match at u with
        | v -> bind_all rest u (bound_in n v store store)
        | exception No_match -> bind_all rest u store)

  (* Matching in general

     A pattern that names nodes in several ways enumerates its namings, each
     extending a store, and is the first result a continuation gives one:
     alternatives and orderings in turn, and each source's namings before the
     next source's. *)

  type first = {
    first : 'a. node -> store -> (store -> 'a option) -> 'a option;
  }

  let rec first_of :
      'a. first list -> node -> store -> (store -> 'a option) -> 'a option =
   fun alternatives u store k ->
    match alternatives with
    | [] -> None
    | p :: rest -> (
        match p.first u store k with
        | Some _ as r -> r
        | None -> first_of rest u store k)

  let rec sources :
      'a. first list -> node list -> store -> (store -> 'a option) -> 'a option
      =
   fun pats srcs store k ->
    match (pats, srcs) with
    | [ p ], u :: _ -> p.first u store k
    | p :: pats, u :: srcs ->
        p.first u store (fun store -> sources pats srcs store k)
    | _ -> k store

  let rec first_tuple :
      'a.
      first list list -> node list -> store -> (store -> 'a option) -> 'a option
      =
   fun tuples srcs store k ->
    match tuples with
    | [] -> None
    | tuple :: rest -> (
        match sources tuple srcs store k with
        | Some _ as r -> r
        | None -> first_tuple rest srcs store k)

  let rec each :
      'a. first -> node list -> store -> (store -> 'a option) -> 'a option =
   fun q srcs store k ->
    match srcs with
    | [] -> k store
    | [ u ] -> q.first u store k
    | u :: srcs -> q.first u store (fun store -> each q srcs store k)

  let length_test p =
    let n = p.required_len in
    if p.strict_length then Some (fun us -> List.compare_length_with us n = 0)
    else if n > 0 then Some (fun us -> List.compare_length_with us n >= 0)
    else None

  let rec general ~root p =
    if p.unique || p.nameless then
      let t, occurrences = unique_parts ~root p in
      let t = with_agreement t occurrences in
      let occurrences = List.map (fun (n, path) -> (n, at path)) occurrences in
      let bind u store k =
        match bind_all occurrences u store with
        | store -> k store
        | exception No_match -> None
      in
      match t with
      | None -> { first = (fun u store k -> bind u store k) }
      | Some t ->
          { first = (fun u store k -> if t u then bind u store k else None) }
    else if p.is_any then
      match p.p_src with
      | Some (Tuples [ choices ]) ->
          let choices = List.map (general ~root:false) choices in
          { first = (fun u store k -> first_of choices u store k) }
      | _ -> { first = (fun _ _ _ -> None) }
    else
      let t = node_test (op_test ~root p) (fields_test p) (length_test p) in
      let srcs =
        match p.p_src with
        | Some (Tuples tuples) ->
            let tuples = List.map (List.map (general ~root:false)) tuples in
            { first = (fun u store k -> first_tuple tuples u.src store k) }
        | Some (Repeat q) ->
            let q = general ~root:false q in
            { first = (fun u store k -> each q u.src store k) }
        | None -> assert false
      in
      let name =
        match p.p_name with
        | None -> fun store _ -> store
        | Some n -> fun store u -> bound_in n u store store
      in
      {
        first =
          (fun u store k ->
            if not (holds t u) then None
            else
              match name store u with
              | exception No_match -> None
              | store -> srcs.first u store k);
      }

  let rec naming acc = function
    | Empty -> acc
    | Bound (n, u, rest) -> naming ((n, u) :: acc) rest

  let match_ p u =
    let namings = ref [] in
    ignore
      ((general ~root:false p).first u Empty (fun store ->
           namings := naming [] store :: !namings;
           None));
    List.rev !namings

  include O
end

module Pattern_matcher = struct
  type node = t

  (* A rule's operations, and its pattern compiled with its function: the result
     of the function on the first naming of a node it gives one for. *)
  type ('ctx, 'r) rule = {
    ops : Op.Set.t option;
    apply : ('ctx -> t -> 'r option) Atomic.t;
  }

  let rec has_op o = function
    | [] -> false
    | s :: srcs -> s.op = o || has_op o srcs

  (* Whether [u] has a source of each operation in [ops]. *)
  let rec has_sources ops u =
    match ops with
    | [] -> true
    | o :: ops -> has_op o u.src && has_sources ops u

  (* The test that a node has a source of each operation in [ops], which a node
     must pass for a rule to be tried. *)
  let early_test = function
    | [] -> None
    | [ o ] -> Some (fun u -> has_op o u.src)
    | [ o; o' ] -> Some (fun u -> has_op o u.src && has_op o' u.src)
    | ops -> Some (has_sources ops)

  let rec lookup store name =
    match store with
    | Upat.Empty -> invalid_argf "the pattern names no %s" name
    | Bound (n, u, rest) -> if String.equal n name then u else lookup rest name

  (* The first of [fn]'s results on the namings of [u] by [alternatives], in
     order. *)
  let rec first_alternative fn ctx u = function
    | [] -> None
    | (t, names) :: rest -> (
        if not (t u) then first_alternative fn ctx u rest
        else
          match fn ctx (names u) with
          | Some _ as x -> x
          | None -> first_alternative fn ctx u rest)

  let always _ = true

  (* A pattern that names nodes in a few ways is compiled as the patterns that
     each name them in one, which match without continuations and allocate
     nothing unless they match. *)
  let compiled pattern fn =
    let early = early_test pattern.Upat.early_reject in
    match (early, Upat.alternatives ~limit:16 pattern) with
    | None, Some [ p ] -> (
        match Upat.compile ~root:true p with
        | None, names -> fun ctx u -> fn ctx (names u)
        | Some t, names -> fun ctx u -> if t u then fn ctx (names u) else None)
    | Some e, Some [ p ] -> (
        match Upat.compile ~root:true p with
        | None, names -> fun ctx u -> if e u then fn ctx (names u) else None
        | Some t, names ->
            fun ctx u -> if e u && t u then fn ctx (names u) else None)
    | early, Some alternatives ->
        let compiled p =
          let t, names = Upat.compile ~root:true p in
          (Option.value t ~default:always, names)
        in
        let alternatives = List.map compiled alternatives in
        let e = Option.value early ~default:always in
        fun ctx u ->
          if e u then first_alternative fn ctx u alternatives else None
    | early, None ->
        let p = Upat.general ~root:true pattern in
        let e = Option.value early ~default:always in
        fun ctx u ->
          if e u then p.first u Empty (fun store -> fn ctx (lookup store))
          else None

  (* A rule compiles its pattern on its first try, so that a matcher built for
     one rewrite compiles only the rules that rewrite tries. Domains that race
     to the first try each compile it. *)
  let rule_ctx pattern fn =
    let apply = Atomic.make (fun _ _ -> None) in
    let first ctx u =
      let compiled = compiled pattern fn in
      Atomic.set apply compiled;
      compiled ctx u
    in
    Atomic.set apply first;
    { ops = pattern.ops; apply }

  let rule pattern f = rule_ctx pattern (fun _ m -> f m)

  (* The context a part's rules read: the matcher's, or none. *)
  type ('ctx, 'c) reads = Ctx : ('ctx, 'ctx) reads | No_ctx : ('ctx, unit) reads

  (* A matcher is a list of parts, each with its rules indexed by operation, its
     own way of declining and the context its rules read, so that appending and
     {!with_ctx} share the parts: a matcher composed for one rewrite copies no
     rules. *)
  type ('ctx, 'r) part =
    | Part : {
        by_op : ('c, 'r) rule list array;
        declines : 'r -> t -> bool;
        reads : ('ctx, 'c) reads;
      }
        -> ('ctx, 'r) part

  (* A matcher builds its parts on its first rewrite, so that a program that
     never rewrites keeps no rules in the heap, which every major collection
     marks. Domains that race to the first rewrite each build the parts, and all
     keep the first built. *)
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
        match r.ops with
        | None -> invalid_arg "a rule's pattern needs an operation"
        | Some ops ->
            List.iter
              (fun o -> by_op.(Op.to_int o) <- r :: by_op.(Op.to_int o))
              (Op.Set.to_list ops))
      rules;
    Part { by_op = Array.map List.rev by_op; declines; reads = Ctx }

  let v rules = make (fun () -> [ part ~declines:( == ) (rules ()) ])

  let fold rules =
    make (fun () -> [ part ~declines:(fun _ _ -> false) (rules ()) ])

  let append m0 m1 = make (fun () -> parts m0 @ parts m1)
  let concat ms = make (fun () -> List.concat_map parts ms)

  (* A part of a matcher without context reads none, whatever the context. *)
  let ignore_ctx : type ctx r. (unit, r) part -> (ctx, r) part =
   fun (Part p) ->
    match p.reads with
    | Ctx -> Part { by_op = p.by_op; declines = p.declines; reads = No_ctx }
    | No_ctx -> Part { by_op = p.by_op; declines = p.declines; reads = No_ctx }

  let with_ctx m = make (fun () -> List.map ignore_ctx (parts m))

  let rec first_rule declines ctx u = function
    | [] -> None
    | r :: rest -> (
        match (Atomic.get r.apply) ctx u with
        | Some x when not (declines x u) -> Some x
        | _ -> first_rule declines ctx u rest)

  (* The first result of the parts' rules for [u], whose operation is [o]. *)
  let rec first_part : type ctx r.
      ctx -> node -> int -> (ctx, r) part list -> r option =
   fun ctx u o -> function
    | [] -> None
    | Part p :: parts -> (
        match p.by_op.(o) with
        | [] -> first_part ctx u o parts
        | rules -> (
            let r =
              match p.reads with
              | Ctx -> first_rule p.declines ctx u rules
              | No_ctx -> first_rule p.declines () u rules
            in
            match r with Some _ -> r | None -> first_part ctx u o parts))

  let rewrite m ctx u = first_part ctx u (Op.to_int u.op) (parts m)
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
  let exception Blocked in
  let pm, bpm =
    match rules with
    | After_sources m -> (Some m, None)
    | Before_sources m -> (None, Some m)
    | Around_sources { before; after } -> (Some after, Some before)
  in
  let walk = pass = Once in
  (* A row per node met: the node; what it stands for once rewritten, [unset]
     before it is pushed and, in the fixed-point rewrite, [placeholder] while it
     is on the stack; its rewrite before the sources, [unset] before one is
     tried and [placeholder] when none applies; and, in the fixed-point
     rewrite, the first and last entries waiting for it. *)
  let rows = Local.create ~width:3 ~iwidth:2 in
  let replaced l = Local.get rows l 1 in
  let is_replaced l =
    let r = replaced l in
    r != unset && r != placeholder
  in
  let bpm_rewrite l =
    let r = Local.get rows l 2 in
    if r != unset then r
    else
      let x = Local.node rows l in
      let r =
        match Pattern_matcher.rewrite (Option.get bpm) ctx x with
        | Some r -> r
        | None -> placeholder
      in
      Local.set rows l 2 r;
      r
  in
  let pm_rewrite x =
    match pm with None -> None | Some pm -> Pattern_matcher.rewrite pm ctx x
  in
  let body_skipped n = n.op = Op.Call && calls = Skip in
  let rest_of n = if body_skipped n then drop 1 n.src else n.src in
  let rebuilt n srcs =
    let src = if body_skipped n then List.hd n.src :: srcs else srcs in
    v n.op ~src ~arg:n.arg
  in
  let work = Work.create () in
  let waits = Work.create () in
  let release () =
    Local.release rows;
    Work.release work;
    Work.release waits
  in
  let once () =
    (* A single pass: a node rewritten on the way down is not entered, and one
       rewritten on the way up is not rewritten again. *)
    let now x =
      let l = Local.find rows x in
      if l = Ids.free || not (is_replaced l) then x else replaced l
    in
    (* [srcs] with each source replaced by what it stands for, [srcs] itself
       when each stands for itself. *)
    let rec sources srcs =
      match srcs with
      | [] -> srcs
      | x :: rest ->
          let r = now x and rest' = sources rest in
          if r == x && rest' == rest then srcs else r :: rest'
    in
    let rec push_srcs = function
      | [] -> ()
      | x :: rest ->
          push_srcs rest;
          let l = Local.add rows x in
          if replaced l == unset then Work.push work l 0 0 0
    in
    Work.push work (Local.add rows root) 0 0 0;
    while not (Work.is_empty work) do
      let l = Work.top work 0 and processed = Work.top work 1 in
      Work.drop work;
      if replaced l == unset then begin
        let n = Local.node rows l in
        if processed = 0 then begin
          let r = if Option.is_some bpm then bpm_rewrite l else placeholder in
          if r != placeholder then Local.set rows l 1 r
          else begin
            Work.push work l 1 0 0;
            push_srcs (rest_of n)
          end
        end
        else
          let rest = rest_of n in
          let srcs = sources rest in
          let new_n = if srcs == rest then n else rebuilt n srcs in
          let new_n =
            match pm_rewrite new_n with Some r -> r | None -> new_n
          in
          Local.set rows l 1 new_n
      end
    done;
    now root
  in
  let fixed_point () =
    (* Each entry is a node's number, a stage and the number of the node it
       currently stands for. Down: rewrite bottom-up to a fixed point and push
       the sources. Rebuild: once the sources are done, rebuild on them and
       rewrite top-down. Link: the node becomes whatever its rewrite became. An
       entry whose dependency is not done waits for it instead of spinning. *)
    let down = 0 and rebuild = 1 and link = 2 in
    let limit = Setting.value Setting.rewrite_stack_limit in
    (* An entry waits in [waits], chained from its dependency's row in the
       order the entries began to wait, by the entries' last ints. *)
    let wait d l stage other =
      let e = Work.length waits in
      Work.push waits l stage other Ids.free;
      let last = Local.iget rows d 1 in
      if last = Ids.free then Local.iset rows d 0 e
      else Work.set waits last 3 e;
      Local.iset rows d 1 e
    in
    let finish l r =
      Local.set rows l 1 r;
      let e = ref (Local.iget rows l 0) in
      if !e <> Ids.free then begin
        Local.iset rows l 0 Ids.free;
        Local.iset rows l 1 Ids.free;
        while !e <> Ids.free do
          Work.push work (Work.get waits !e 0) (Work.get waits !e 1)
            (Work.get waits !e 2) 0;
          e := Work.get waits !e 3
        done
      end
    in
    (* A node rewritten bottom-up to a fixed point; the set of the nodes it went
       through is made only when it moves. *)
    let fixed_point l n =
      match bpm with
      | None -> n
      | Some _ -> (
          match bpm_rewrite l with
          | exception Bottom_up_gate -> raise_notrace (Gate n)
          | next when next == placeholder -> n
          | next ->
              let seen = Tbl.create 4 in
              Tbl.replace seen n ();
              let rec loop current =
                if Tbl.mem seen current then
                  invalid_arg
                    "graph_rewrite does not terminate: a bottom-up rewrite \
                     cycles";
                Tbl.replace seen current ();
                match bpm_rewrite (Local.add rows current) with
                | next when next == placeholder -> current
                | next -> loop next
                | exception Bottom_up_gate -> raise_notrace (Gate current)
              in
              loop next)
    in
    (* The sources not yet on the stack, the first on top. *)
    let rec push_down = function
      | [] -> ()
      | x :: rest ->
          push_down rest;
          let d = Local.add rows x in
          if replaced d == unset then begin
            Work.push work d down d 0;
            Local.set rows d 1 placeholder
          end
    in
    (* [srcs] with each source replaced by its rewrite, [srcs] itself when each
       is its own. Raises [Blocked], with the first source not yet rewritten in
       [blocker]. *)
    let blocker = ref Ids.free in
    let rec sources srcs =
      match srcs with
      | [] -> srcs
      | x :: rest ->
          let d = Local.add rows x in
          if not (is_replaced d) then begin
            blocker := d;
            raise_notrace Blocked
          end;
          let r = replaced d and rest' = sources rest in
          if r == x && rest' == rest then srcs else r :: rest'
    in
    let root_l = Local.add rows root in
    Work.push work root_l down root_l 0;
    Local.set rows root_l 1 placeholder;
    while not (Work.is_empty work) do
      if Work.length work > limit then
        invalid_arg
          "graph_rewrite does not terminate: its work list is too long";
      let l = Work.top work 0
      and stage = Work.top work 1
      and other = Work.top work 2 in
      Work.drop work;
      if not (is_replaced l) then
        if stage = down then begin
          let n = Local.node rows l in
          match fixed_point l n with
          | exception Gate gated -> finish l gated
          | new_n ->
              let nl = if new_n == n then l else Local.add rows new_n in
              Work.push work l rebuild nl 0;
              push_down (rest_of new_n)
        end
        else if stage = rebuild then begin
          let new_n = Local.node rows other in
          let rest = rest_of new_n in
          match sources rest with
          | exception Blocked -> wait !blocker l rebuild other
          | srcs -> (
              let next =
                if srcs == rest then pm_rewrite new_n
                else Some (rebuilt new_n srcs)
              in
              match next with
              | None -> finish l new_n
              | Some next ->
                  let nl = Local.add rows next in
                  Work.push work l link nl 0;
                  Work.push work nl down nl 0)
        end
        else if is_replaced other then finish l (replaced other)
        else wait other l link other
    done;
    if is_replaced root_l then replaced root_l
    else
      invalid_arg
        "graph_rewrite does not terminate: a node's rewrite depends on itself"
  in
  match if walk then once () else fixed_point () with
  | r ->
      release ();
      r
  | exception e ->
      release ();
      reraise e

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

(* A range is a loop, whatever the values it computes with: assuming a fact
   inside one's bound would make another loop of the same identity, which runs
   apart. *)
let pm_keep_ranges =
  Pattern_matcher.(
    v
      (fun () -> [
        rule_ctx (Upat.op Op.Range ~name:"r") (fun subs m ->
            if Tbl.mem subs (m "r") then None else raise Bottom_up_gate);
      ]))

let assuming u subs =
  substitute ~extra_pm:pm_keep_ranges ~calls:Skip ~pass:Fixed_point u subs

let pm_drop_after =
  Pattern_matcher.(
    v (fun () -> [ rule (Upat.op Op.After ~name:"a") (fun m -> Some (nth (m "a") 0)) ]))

let resolve_returned_after r effects =
  let target = base r in
  match
    List.filter
      (fun st -> st.op = Op.Store && base (nth st 0) == target)
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

let custom_function ?(dtype = Dtype.Void) ?(shape = []) ?(fmt = false) name
    args =
  let shape_src =
    if Dtype.equal dtype Dtype.Void then []
    else
      match List.map (function Int n -> int n | Sym u -> u) shape with
      | [ n ] -> [ n ]
      | ns -> [ v Op.Stack ~src:ns ]
  in
  v Op.Custom_function ~src:(shape_src @ args)
    ~arg:(Custom_function { name; dtype; fmt })

let call ?name ?(precompile = false) ?(inline = false) body args =
  if not (Op.Set.mem body.op opaque_call_bodies) then
    invalid_argf "a %s cannot be called" (Op.name body.op);
  if Nodes.cardinal (ranges body) > 0 then
    invalid_arg "ranges leak out of the called body";
  v Op.Call ~src:(body :: args)
    ~arg:(Call { name; precompile; inline })

let is_inline_call u =
  u.op = Op.Call && (body u).op = Op.Sink
  && (match (body u).arg with No_arg -> true | _ -> false)
  && match u.arg with Call c -> not c.precompile | _ -> false

let unbound_output x =
  let b = base x in
  b.op = Op.Alloc && not (param_arg_of b).bind_on_realize

let has_unbound_outputs c =
  c.op = Op.Call && List.exists unbound_output (drop 1 c.src)

let unbound_outputs c =
  List.filter_map
    (fun x -> if unbound_output x then Some (after x [ c ]) else None)
    (drop 1 c.src)

(* Programs *)

let kernel_info ?(name = "test") ?(applied_opts = []) ?opts_to_apply ?estimates
    ?(beam = 0) ?(optimized = false) ?(kernel = true) ?split () =
  {
    name;
    applied_opts;
    opts_to_apply;
    estimates;
    beam;
    optimized;
    kernel;
    split;
  }

let function_name (k : kernel_info) = Helpers.to_function_name k.name

(* Memos *)

let memoized_shape ~compute u =
  memoized ~calls:Enter ~unset:unset_shape
    ~get:(fun n -> n.memos.shape_memo)
    ~set:(fun n s -> (own_memos n).shape_memo <- s)
    ~compute u

let memoized_movement ~compute u =
  let m = u.memos.marg_memo in
  if m != unset_movement then m
  else
    let m = compute u in
    (own_memos u).marg_memo <- m;
    m
