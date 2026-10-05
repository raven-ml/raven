(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk

(* Devices *)

(* The kind of device a vendor library claims, if any. *)
let vendor d =
  if Metal.claims d then Some "METAL"
  else if Option.is_some (Nx_cuda_device.of_device d) then Some "CUDA"
  else if Option.is_some (Nx_amd_device.of_device d) then Some "AMD"
  else if Option.is_some (Nx_nv_device.of_device d) then Some "NV"
  else None

(* The target of a device kind, which names no interface and no indices. *)
let of_kind ?(renderer = "") ?(arch = "") device =
  { Helpers.Target.device; renderer; arch; interface = ""; indices = "" }

let target d =
  match vendor d with
  | Some kind -> of_kind ~arch:(Nx_device.arch d) kind
  | None
    when d != Nx_device.disk
         && (Nx_device.host_of d == d || Nx_device.shares_host_memory d) ->
      let host = Nx_device.host_of d in
      let cpu = if host == Nx_device.host then "native" else "generic" in
      of_kind ~renderer:"CLANG" ~arch:(Nx_device.arch host ^ "," ^ cpu) "CPU"
  | None ->
      invalid_arg
        (Printf.sprintf "Tolk_engine.target: %s runs no program"
           (Nx_device.name d))

let renderer d =
  match Device.renderer (target d) with Ok r -> r | Error why -> failwith why

let find fn devices name =
  match List.assoc_opt name devices with
  | Some d -> d
  | None ->
      invalid_arg
        (Printf.sprintf "Tolk_engine.%s: no device is named %s" fn name)

(* Engine devices *)

type device = {
  device : Nx_device.t;
  compiler : Hcq2.device;
  placeholder : Ops.t -> Nx_device.Buffer.t option;
  submitting : unit -> unit;
}

(* The vendors whose encoders the compiler has, in the order they are tried: a
   vendor that claims a device gives its queues, the storage of the placeholders
   its commands name and its refresh inside each submission. *)
let vendors = [ Metal.queues; Cuda.queues; Amd.queues; Nv.queues ]

(* [devices] with the host of each of its devices that none of its names names,
   under the host's own name. *)
let with_hosts devices =
  List.fold_left
    (fun devices (_, d) ->
      let h = Nx_device.host_of d in
      let n = Nx_device.name h in
      if
        List.exists (fun (_, d') -> d' == h) devices || List.mem_assoc n devices
      then devices
      else devices @ [ (n, h) ])
    devices devices

(* Refuses [devices] if it gives one name twice. *)
let rec distinct = function
  | [] -> ()
  | (n, _) :: devices ->
      if List.mem_assoc n devices then
        invalid_arg
          (Printf.sprintf "Tolk_engine.device: %s names two devices" n);
      distinct devices

let device devices name =
  distinct devices;
  let devices = with_hosts devices in
  let d = find "device" devices name in
  (* The disk runs no program: its copies are the runtime's. *)
  let target = if d == Nx_device.disk then of_kind "DISK" else target d in
  (* Any name of the host names it: the first one serves. The host is missing
     only when another device has its name. *)
  let host =
    lazy
      (let h = Nx_device.host_of d in
       match List.find_opt (fun (_, h') -> h' == h) devices with
       | Some (host, _) -> host
       | None ->
           invalid_arg
             (Printf.sprintf
                "Tolk_engine.device: %s names another device than %s's host"
                (Nx_device.name h) name))
  in
  match List.find_map (fun queues -> queues ~host devices name d) vendors with
  | Some (queues, placeholder, submitting) ->
      {
        device = d;
        compiler = { Hcq2.target; queues = Some queues };
        placeholder;
        submitting;
      }
  | None ->
      {
        device = d;
        compiler = { Hcq2.target; queues = None };
        placeholder = (fun _ -> None);
        submitting = ignore;
      }

(* Host programs *)

module Program = struct
  (* A program whose launch splits a loop into blocks: the loop's iterations,
     the program's operations, and the positions among its variables of those
     that hold a block's bounds. *)
  type split = { extent : Ops.sint; ops : Ops.sint; lo : int; hi : int }

  type t = {
    program : Nx_device.Program.t;
    buffers : Device.Tiny_elf.param list; (* in the order of the globals *)
    globals : int list; (* the argument slot of each *)
    vars : Ops.t list;
    split : split option;
  }

  let split_of prg (info : Ops.program_info) =
    let at slot =
      List.find_index
        (fun v ->
          match Ops.arg v with Ops.Param p -> p.slot = slot | _ -> false)
        info.vars
    in
    match Ops.arg (Ops.nth prg 0) with
    | Ops.Kernel { split = Some s; estimates; _ } -> (
        let ops =
          match estimates with Some e -> e.ops | None -> s.iterations
        in
        match (at s.lo, at s.hi) with
        | Some lo, Some hi -> Some { extent = s.iterations; ops; lo; hi }
        | _ ->
            invalid_arg
              "Tolk_engine.Program.load: a split's bounds are no variables")
    | _ -> None

  let load d prg =
    if (target d).device <> "CPU" then
      invalid_arg
        (Printf.sprintf "Tolk_engine.Program.load: %s runs no host program"
           (Nx_device.name d));
    let elf = Device.Tiny_elf.of_program prg in
    let info =
      match Ops.arg prg with Ops.Program info -> info | _ -> assert false
    in
    let nbuffers = List.length info.globals in
    let buffers = List.filteri (fun i _ -> i < nbuffers) elf.signature in
    match
      Nx_device.Program.load (Nx_device.host_of d) ~binary:elf.lib
        ~name:elf.name
    with
    | Ok program ->
        {
          program;
          buffers;
          globals = info.globals;
          vars = info.vars;
          split = split_of prg info;
        }
    | Error why -> failwith why

  let value vars v =
    match Ops.arg v with
    | Ops.Param p -> (
        match
          (Option.bind p.name (fun n -> List.assoc_opt n vars), p.bound)
        with
        | Some x, _ -> x
        | None, Some b -> Dtype.Value.to_int b
        | None, None ->
            invalid_arg
              (Printf.sprintf "Tolk_engine: variable %s is unbound"
                 (Option.value p.name ~default:(string_of_int p.slot))))
    | _ -> assert false

  (* A block repays waking the pool's threads and joining them once it does 2^18
     operations. A wake and a join cost a few microseconds, and a block should
     cost ten times that, about 50 to 100 us. A core runs 1 to 2 G scalar
     operations a second and about 12 G on vectorised loops, so 100 us is 2^17
     to 2^20 operations; 2^18 sits between. A launch runs at most four blocks
     per thread, so that the threads that claim blocks as they finish balance
     fast and slow cores. *)
  let block_ops = 1 lsl 18
  let blocks_per_worker = 4

  (* Whether the variable of slot [i] is a block's bound, which a launch sets
     itself. *)
  let bounds_block p i =
    match p.split with Some s -> i = s.lo || i = s.hi | None -> false

  (* The values of [vals], [p]'s variables or their bindings, each read by
     [value], save the blocks' bounds. *)
  let values p vals value =
    Array.of_list
      (List.mapi (fun i v -> if bounds_block p i then 0 else value v) vals)

  (* The blocks of a launch of [extent] iterations and [ops] operations that
     splits [s]. *)
  let blocks_of s ~extent ~ops =
    let most = blocks_per_worker * Nx_device.Program.workers () in
    let blocks = max 1 (min (min extent most) (ops / block_ops)) in
    { Nx_device.Program.extent; blocks; lo = s.lo; hi = s.hi }

  (* [s] with each of its variables read by [value]. The loop a launch splits
     may end at a variable its program no longer reads, as it reads the block's
     bounds instead. *)
  let eval value (s : Ops.sint) =
    let variables u =
      List.filter_map
        (fun v ->
          if Ops.is_variable v then Some (Ops.expr v, value v) else None)
        (Ops.toposort u)
    in
    match s with Int n -> n | Sym u -> Ops.sym_infer s (variables u)

  (* The split of a launch of [p] whose variables [value] reads, if [p]
     splits. *)
  let split p value =
    match p.split with
    | None -> None
    | Some s ->
        Some
          (blocks_of s ~extent:(eval value s.extent) ~ops:(eval value s.ops))

  let call p value buffers values =
    Nx_device.Program.call ?split:(split p value) p.program buffers values

  let blocks ?(vars = []) p =
    match split p (value vars) with Some s -> s.blocks | None -> 1

  let run ?(vars = []) p buffers =
    let fn = "Tolk_engine.Program.run" in
    let n = List.length p.buffers in
    if List.length buffers <> n then
      invalid_arg
        (Printf.sprintf "%s: %d buffers for %d parameters" fn
           (List.length buffers) n);
    List.iter2
      (fun (param : Device.Tiny_elf.param) b ->
        let need =
          List.fold_left ( * ) (Dtype.itemsize param.dtype) param.shape
        in
        if Nx_device.Buffer.nbytes b < need then
          invalid_arg
            (Printf.sprintf "%s: slot %d holds %d bytes, not %d" fn param.slot
               (Nx_device.Buffer.nbytes b)
               need))
      p.buffers buffers;
    call p (value vars) (Array.of_list buffers) (values p p.vars (value vars))
end

(* Linked schedules *)

let strf = Printf.sprintf
let fail fn fmt = Printf.ksprintf (fun m -> invalid_arg (fn ^ ": " ^ m)) fmt

module B = Nx_device.Buffer

(* Variables *)

(* A run's variables, a cell each: those the run binds, those of the ranges
   running a trip, and the others unset. Linking gives each name a cell, so
   that a call reads a value without looking up its name. *)
type env = { values : int array; set : bool array }

let cell cells name =
  match Hashtbl.find_opt cells name with
  | Some i -> i
  | None ->
      let i = Hashtbl.length cells in
      Hashtbl.add cells name i;
      i

(* How a value of a view's offset or a range's trips reads the variable [u], as
   [Ops.sym_infer] does: a variable the run leaves unset has no value. *)
let strict cells u =
  let name = Ops.expr u in
  let i = cell cells name in
  fun env ->
    if env.set.(i) then env.values.(i)
    else
      invalid_arg (strf "Tolk_engine.run: the variable %s has no value" name)

(* How a call reads the variable [u] it passes a program: a variable the run
   leaves unset takes its bound value. *)
let binding cells u =
  let name = Ops.expr u in
  let i = cell cells name in
  let unset =
    match Ops.arg u with
    | Ops.Param { bound = Some b; _ } ->
        let b = Dtype.Value.to_int b in
        fun () -> b
    | _ ->
        fun () ->
          invalid_arg (strf "Tolk_engine.run: variable %s is unbound" name)
  in
  fun env -> if env.set.(i) then env.values.(i) else unset ()

let offset cells (s : Ops.sint) = Ops.sym_compile s (strict cells)

(* An input of a batch's address table, resolved at link: the parameter's slot
   whose buffers it is, its shard, its byte offset, the device whose address the
   table holds, and whether the batch may write it. Only a parameter's address
   changes between runs, so the table's other addresses are written at link. *)
type input = {
  slot : int;
  shard : int option;
  offset : int;
  on : Nx_device.t;
  written : bool;
}

(* The value of a host program's variable on each run: a constant, a device's
   last submitted value or the value its work signals, or the schedule's. *)
type binder =
  | Fixed of int
  | Submitted of Nx_device.t
  | Signals of Nx_device.t
  | Var of (env -> int)

(* A batch, as linked: its host program, the devices whose queues it submits,
   its arguments, and the values its previous run signals on each device. What
   depends on the linked schedule alone is resolved once, so that a run
   allocates little. *)
type batch = {
  info : Ops.hcq_info;
  named : string -> Nx_device.t; (* the devices of the batch, by name *)
  host_program : Program.t;
  devices : Nx_device.t list; (* whose queues it submits *)
  queues : Nx_device.t array; (* the same, by index *)
  submitting : (unit -> unit) list; (* each device's, before the host program *)
  copies : (Nx_device.t * Nx_device.t * int) list; (* from, into, bytes *)
  arguments : B.t list; (* by argument slot *)
  buffers : B.t array; (* the host program's, in the order of its globals *)
  binders : binder array; (* the host program's variables, in order *)
  values : int array; (* the variables' values of the run being made *)
  inputs : input array; (* the address table's entries, in order *)
  run_inputs : B.t array; (* the inputs' buffers of the run being made *)
  run_reached : B.t array; (* how its devices reach them, by input *)
  addresses : nativeint array; (* the inputs' addresses of the run being made *)
  touched : B.t list;
      (* the arguments, the storage its words address, and how link reached that
         storage *)
  table : (int64, Bigarray.int64_elt, Bigarray.c_layout) Bigarray.Array1.t;
  last : int array;
}

(* The buffers a run gives a view of storage, one per device, from the
   buffers bound to the parameters and the run's variables. *)
type operand = B.t list array -> env -> B.t list

(* A launch of a host program on the device of a lane of its call: how a run
   finds the buffers and values it passes the program, and the arrays it
   passes them in, which each run fills anew. *)
type launch = {
  program : Program.t;
  args : (B.t list array -> env -> B.t) array; (* in the order of the globals *)
  vals : (env -> int) array; (* by variable, a block's bounds 0 *)
  split : (Program.split * (env -> int) * (env -> int)) option;
      (* the split, its iterations and its operations *)
}

(* A call of a schedule, as linked: a host program launched on each device of
   its first argument, a copy, a batch, or calls run once for each value of
   ranges, each range a cell of its variable and its number of trips. *)
type call =
  | Kernel of { call : Ops.t; launches : launch array }
  | Copy of { call : Ops.t; dst : operand; src : operand }
  | Batch of batch
  | Range of { ranges : (int * (env -> int)) list; body : call list }

type t = {
  lock : Mutex.t; (* runs share the storage: one at a time *)
  calls : call list;
  params : (int * Nx_device.t list * int) list;
      (* the parameters the runs bind: slot, devices and bytes *)
  cells : (string, int) Hashtbl.t; (* the variables' cells, by name *)
  env : env; (* the variables of the run being made *)
  mutable synced : Nx_device.t list;
      (* the devices the run's host programs synchronized since it last queued
         work *)
}

let names u =
  match Ops.device u with
  | Some (Single d) -> [ d ]
  | Some (Multi ds) -> ds
  | None -> invalid_arg (Format.asprintf "%a has no device" Ops.pp u)

let is_slot u =
  Ops.op u = Op.Param
  && Option.is_none (Ops.tag u)
  && Ops.addrspace u <> Some Dtype.Alu

let bytes u = Ops.max_numel u * Dtype.itemsize (Ops.dtype u)

let at b off n =
  if
    (off = 0
    && B.nbytes b = n)
    [@mutate off "a view of a whole buffer is the buffer"]
  then b
  else B.view b ~offset:off Nx_dtype.Scalar.UInt8 n

let sint u =
  match Ops.arg u with
  | Ops.Const (`Int z) -> Ops.Int (Bigint.to_int z)
  | _ -> Sym u

let lane buffers i = match buffers with [ b ] -> b | bs -> List.nth bs i

let refused d b why =
  invalid_arg
    (strf "Tolk_engine: %s cannot address memory of %s: %s" (Nx_device.name d)
       (Nx_device.name (B.device b))
       why)

(* How [d]'s work reaches [b]: [b] on [d], and otherwise as nx.device's
   [Buffer.reach] gives it for [access], staged where [d] maps none of it. *)
let reach d b access =
  if B.device b == d then b
  else match B.reach d b access with Ok r -> r | Error why -> refused d b why

(* A placeholder's storage is its device's to address, and is borrowed alone: a
   signal word, which the host waits on while the work runs, cannot be
   staged. *)
let borrowed d b =
  if B.device b == d then b
  else match B.borrow d b with Ok r -> r | Error why -> refused d b why

(* Whether [b] and [b'] are the same bytes. *)
let same_memory b b' =
  b == b'
  || (B.nbytes b = B.nbytes b' && Nativeint.equal (B.address b) (B.address b'))

let int_of_const u =
  match Ops.arg u with
  | Ops.Const (`Int z) -> Bigint.to_int z
  | Ops.Const (`Bool b) -> Bool.to_int b
  | _ -> invalid_arg (Format.asprintf "%a is no integer" Ops.pp u)

(* How a run reads the value [u] that a call passes a program on the device of
   lane [lane]: a constant, the lane's number, or a variable. *)
let reader cells lane u =
  if Ops.op u = Op.Const then
    let c = int_of_const u in
    fun _ -> c
  else
    match (Ops.expr u, Ops.arg u) with
    | "_device_num", _ -> fun _ -> lane
    | _, Ops.Param _ -> binding cells u
    | _ -> assert false

let not_storage base =
  invalid_arg (Format.asprintf "%a is not storage" Ops.pp base)

(* The buffers of the view [u], one per device, or one shared by them all: of a
   parameter the run binds, of linked storage or placeholders, or of a stack of
   one view per device. A view of linked storage at a constant offset is made
   once. *)
let rec operand storage cells u : operand =
  let base, shard, off = Hcq2.lane_offset u in
  let n = bytes u in
  let select bs = match shard with Some i -> [ List.nth bs i ] | None -> bs in
  let views bs o = List.map (fun b -> at b o n) (select bs) in
  let moving = offset cells off in
  match (Ops.op base, Ops.arg base, off) with
  | Op.Param, Ops.Param p, _ when Option.is_none (Ops.tag base) ->
      fun slots env -> views slots.(p.slot) (moving env)
  | (Op.Param | Op.Buffer), _, Int o -> (
      match Ops.Tbl.find_opt storage base with
      | Some bs ->
          let vs = views bs o in
          fun _ _ -> vs
      | None -> fun _ _ -> views (Ops.Tbl.find storage base) o)
  | (Op.Param | Op.Buffer), _, Sym _ ->
      fun _ env -> views (Ops.Tbl.find storage base) (moving env)
  | Op.Mstack, _, _ ->
      let stack = List.map (operand storage cells) (Ops.src base) in
      fun slots env ->
        views (List.concat_map (fun v -> v slots env) stack) (moving env)
  | _ -> not_storage base

(* The buffer of the view [u] that the launch of lane [i] passes: a lane or
   shard of a parameter or of linked storage, at an offset that may move with a
   range. A view of linked storage at a constant offset is made once. *)
let lane_operand storage cells u i =
  let base, shard, off = Hcq2.lane_offset u in
  let n = bytes u in
  let pick bs = match shard with Some j -> List.nth bs j | None -> lane bs i in
  let moving = offset cells off in
  let linked = Ops.Tbl.find_opt storage base in
  match (Ops.op base, Ops.arg base, off, linked) with
  | Op.Param, Ops.Param p, _, _ when Option.is_none (Ops.tag base) ->
      fun slots env -> at (pick slots.(p.slot)) (moving env) n
  | (Op.Param | Op.Buffer), _, Int o, Some bs ->
      let b = at (pick bs) o n in
      fun _ _ -> b
  | (Op.Param | Op.Buffer), _, Sym _, Some bs ->
      let b = pick bs in
      fun _ env -> at b (moving env) n
  | _ ->
      let v = operand storage cells u in
      fun slots env -> lane (v slots env) i

(* The launch of the host program [prg] that [call] makes on [d], the device of
   its lane [i]. *)
let launch storage cells call prg d i =
  let p = Program.load d prg in
  let info = match Ops.arg prg with Ops.Program i -> i | _ -> assert false in
  let args = Realize.get_call_arg_uops call in
  let value k v =
    if Program.bounds_block p k then fun _ -> 0 else reader cells i v
  in
  let count (s : Ops.sint) = Ops.sym_compile s (reader cells i) in
  {
    program = p;
    args =
      Array.of_list
        (List.map
           (fun g -> lane_operand storage cells (List.nth args g) i)
           info.globals);
    vals = Array.of_list (List.mapi value (Realize.get_call_var_uops call prg));
    split =
      Option.map
        (fun (s : Program.split) -> (s, count s.extent, count s.ops))
        p.split;
  }

(* The storage each device holds of [base], linked: storage, a placeholder, or
   the stack of one view per device. *)
let holds storage base =
  match Ops.op base with
  | Op.Param | Op.Buffer -> Ops.Tbl.find storage base
  | Op.Mstack ->
      let cells = Hashtbl.create 1 in
      let stack = List.map (operand storage cells) (Ops.src base) in
      let unset = Hashtbl.length cells in
      let env = { values = Array.make unset 0; set = Array.make unset false } in
      List.concat_map (fun v -> v [||] env) stack
  | _ -> not_storage base

(* Link patches *)

(* The value of a word known at link, with each address resolved. *)
let rec word addr u : Dtype.value =
  let v x : Dtype.value = match word addr x with #Dtype.value as x -> x in
  match (Ops.op u, Ops.arg u) with
  | Op.Const, Ops.Const (#Dtype.value as c) -> c
  | Op.Getaddr, Ops.Device d ->
      let dn = match d with Single n | Multi (n :: _) -> n | Multi [] -> "" in
      `Int (Bigint.of_nativeint (addr dn (Ops.nth u 0)))
  | Op.Cast, _ -> Dtype.truncate (Ops.dtype u) (v (Ops.nth u 0))
  | Op.Bitcast, _ ->
      let x = Ops.nth u 0 in
      Dtype.bitcast (Ops.dtype x) (Ops.dtype u) (v x)
  | o, _ -> (
      match
        Ops.exec_alu o (Ops.dtype u)
          (List.map (fun x -> (v x :> Dtype.const)) (Ops.src u))
      with
      | #Dtype.value as x -> x
      | `Invalid ->
          invalid_arg (Format.asprintf "Tolk_engine.link: %a" Ops.pp u))

(* The little-endian bytes of [v], a value of [dt]. *)
let le dt (v : Dtype.value) =
  let n = Dtype.itemsize dt in
  let unsigned =
    match n with
    | 1 -> Dtype.Uint8
    | 2 -> Dtype.Uint16
    | 4 -> Dtype.Uint32
    | _ -> Dtype.Uint64
  in
  let z =
    match (dt, v) with
    | Dtype.Bool, `Bool b -> Bigint.of_int (Bool.to_int b)
    | _ -> (
        match Dtype.bitcast dt unsigned v with `Int z -> z | _ -> assert false)
  in
  String.init n (fun k -> Char.chr (Bigint.to_int (Bigint.extract z (8 * k) 8)))

let write b off s =
  let n = String.length s in
  if (n > 0) [@mutate off "a copy of no bytes writes nothing"] then
    let src = Bigarray.(Array1.init char c_layout n (String.get s)) in
    B.copy ~src:(B.of_bigarray src) ~dst:(at b off n)

(* The linked buffer the view [u] is of, the shard it selects, and the view's
   byte offset in it. *)
let linked_at storage u =
  let base, shard, off = Hcq2.unwrap_lane u in
  (List.nth (holds storage base) (Option.value shard ~default:0), off)

(* Writes the link patch [p], a store of bytes or of words at constant element
   indices, into the linked storage its view is of. *)
let apply storage addr p =
  let dst = Ops.nth p 0 and v = Ops.nth p 1 in
  let base_at = linked_at storage in
  let bytes_of u =
    match (Ops.op u, Ops.arg u) with
    | Op.Binary, Ops.Bytes s -> Some s
    | Op.Bitcast, _ -> (
        match Ops.arg (Ops.nth u 0) with Ops.Bytes s -> Some s | _ -> None)
    | _ -> None
  in
  match (bytes_of v, Ops.op dst) with
  | Some s, _ ->
      let b, off = base_at dst in
      write b off s
  | None, Op.Index -> (
      let viewed = Ops.nth dst 0 in
      let b, off = base_at viewed in
      let dt = Ops.dtype v in
      let n = Dtype.itemsize dt in
      (* A group of one row is its index and its word, unstacked. *)
      let lanes u = if Ops.op u = Op.Stack then Ops.src u else [ u ] in
      match (lanes (Ops.nth dst 1), lanes v) with
      | offs, words when List.length offs = List.length words ->
          List.iter2
            (fun o w ->
              write b (off + (int_of_const o * n)) (le dt (word addr w)))
            offs words
      | _ -> invalid_arg "Tolk_engine.link: a patch of mismatched stacks")
  | None, _ ->
      invalid_arg
        (Format.asprintf "Tolk_engine.link: %a is no link patch" Ops.pp p)

(* Batches *)

let signal_word_tag = Ops.Tag.String "timeline"
let staging_tag = Ops.Tag.String "staging"

(* The host's staging memory, which nx.device's copies and the staged copies of
   every linked schedule share: each run that stages through it touches it, and
   so holds the host taken, as nx.device's copies do. *)
let staging h n =
  let b = Nx_device.staging h in
  if B.nbytes b < n then
    invalid_arg
      (strf "Tolk_engine.link: %s's staging memory holds %d bytes, not %d"
         (Nx_device.name h) (B.nbytes b) n);
  b

(* The storage of a batch's placeholder [u]: the signal word of its device for
   ["timeline"], the host's staging memory for ["staging"], the address of a C
   function for a [("cfunc", lib, f)] tuple, and for any other, which the host
   writes: pinned memory for a volatile placeholder, which the device writes
   too, and for a command buffer, which the device fetches; mapped memory, which
   the device reads as its own, for the rest, such as kernel arguments. *)
(* The vendor whose commands name a placeholder gives its storage: its own
   device's, or a device of the batch's, such as the vendor that owns a C
   function its host program calls. *)
let placeholder device queues u =
  let dev = device (List.hd (names u)) in
  let d = dev.device in
  let named =
    List.find_map
      (fun n -> (device n).placeholder u)
      (List.hd (names u) :: queues)
  in
  match named with
  | Some b when B.nbytes b < bytes u ->
      invalid_arg
        (Format.asprintf "Tolk_engine.link: %a's storage holds %d bytes, not %d"
           Ops.pp u (B.nbytes b) (bytes u))
  | Some b -> b
  | None -> (
      match Ops.tag u with
      | Some t when Ops.Tag.equal t signal_word_tag -> Nx_device.signal_word d
      | Some t when Ops.Tag.equal t staging_tag -> staging d (bytes u)
      | Some (Ops.Tag.Tuple [ String "cfunc"; String lib; String _ ]) ->
          invalid_arg (strf "Tolk_engine.link: no C library %s" lib)
      | _ ->
          let volatile =
            match Ops.arg u with Ops.Param p -> p.volatile | _ -> false
          in
          let memory : B.memory =
            if volatile || Hcq2.is_cmdbuf u then Pinned else Mapped
          in
          B.create ~memory d Nx_dtype.Scalar.UInt8 (max 1 (bytes u)))

let hcq_info call =
  match Ops.arg call with Ops.Call { aux; _ } -> aux | _ -> None

let link_batch ~device ~storage ~cells call patches =
  let info = Option.get (hcq_info call) in
  let args = Realize.get_call_arg_uops call in
  let is_placeholder u = Ops.op u = Op.Param && Option.is_some (Ops.tag u) in
  (* The placeholders the patches address are the batch's too, though its host
     program may not take them, such as a signal word only its queues read. *)
  let patched = List.concat_map Ops.toposort patches in
  List.iter
    (fun u ->
      if is_placeholder u && not (Ops.Tbl.mem storage u) then
        Ops.Tbl.replace storage u [ placeholder device info.device u ])
    (args @ patched);
  let reached =
    List.filter_map
      (fun u ->
        if Ops.op u = Op.Buffer || is_placeholder u then
          Some (Ops.Tbl.find storage u)
        else None)
      patched
    |> List.concat
  in
  (* How its devices reach the storage its words address, which its runs touch:
     their work's completion, not the link, ends them. A memory is reached once
     per device, for reading and writing when the batch writes it through any of
     its words. *)
  let reaches = ref [] in
  let getaddr g =
    match Ops.arg g with
    | Ops.Device (Single dn | Multi (dn :: _)) ->
        let base, _, _ = Hcq2.unwrap_lane (Ops.nth g 0) in
        Some ((device dn).device, base, fst (linked_at storage (Ops.nth g 0)))
    | _ -> None
  in
  let written =
    List.filter_map
      (fun g ->
        if Ops.op g <> Op.Getaddr then None
        else
          match getaddr g with
          | Some (d, base, b) when List.memq base info.writes -> Some (d, b)
          | _ -> None)
      patched
  in
  let reach_word d base b =
    match
      List.find_opt (fun (d', b', _) -> d' == d && same_memory b b') !reaches
    with
    | Some (_, _, r) -> r
    | None ->
        let r =
          if is_placeholder base then borrowed d b
          else
            let writes (d', b') = d' == d && same_memory b b' in
            reach d b
              (if List.exists writes written then B.Read_write else B.Read)
        in
        reaches := (d, b, r) :: !reaches;
        r
  in
  let addr dn u =
    let base, _, _ = Hcq2.unwrap_lane u in
    let b, off = linked_at storage u in
    let d = (device dn).device in
    Nativeint.add (B.address (reach_word d base b)) (Nativeint.of_int off)
  in
  List.iter (apply storage addr) patches;
  let arguments = List.map (fun u -> List.hd (Ops.Tbl.find storage u)) args in
  let queues = List.map (fun n -> (device n).device) info.device in
  (* The host programs of a device's queues run on the host they name. *)
  let host =
    match (device (List.hd info.device)).compiler.queues with
    | Some q -> (device q.host).device
    | None -> invalid_arg "Tolk_engine.link: a batch of a device without queues"
  in
  let table =
    if info.table < 0 then Bigarray.(Array1.create int64 c_layout 0)
    else
      let b = List.nth arguments info.table in
      let hb =
        match B.borrow host b with Ok m -> m | Error why -> invalid_arg why
      in
      reaches := (host, b, hb) :: !reaches;
      B.bigarray Bigarray.int64 hb
  in
  let named = List.map (fun n -> (n, (device n).device)) info.device in
  let named n = List.assoc n named in
  let host_program = Program.load host (Ops.body call) in
  let binder u =
    if Ops.op u = Op.Const then Fixed (int_of_const u)
    else
      let name = Ops.expr u in
      let is var =
        List.find_opt (fun n -> Ops.expr (var n) = name) info.device
      in
      match (is Hcq2.submitted, is Hcq2.value) with
      | Some n, _ -> Submitted (named n)
      | None, Some n -> Signals (named n)
      | None, None -> Var (reader cells 0 u)
  in
  let input (base, off, dev) =
    let base, shard, inner = Hcq2.unwrap_lane base in
    match (Ops.op base, Ops.arg base) with
    | Op.Param, Ops.Param p when Option.is_none (Ops.tag base) ->
        {
          slot = p.slot;
          shard;
          offset = off + inner;
          on = named dev;
          written = List.memq base info.writes;
        }
    | _ ->
        invalid_arg
          (Format.asprintf "Tolk_engine.link: an input of %a" Ops.pp base)
  in
  let inputs = Array.of_list (List.map input info.inputs) in
  let binders = Array.of_list (List.map binder host_program.vars) in
  {
    info;
    named;
    host_program;
    submitting = List.map (fun n -> (device n).submitting) info.device;
    copies =
      List.map
        (fun (s, d, n) -> ((device s).device, (device d).device, n))
        info.copies;
    devices = queues;
    queues = Array.of_list queues;
    arguments;
    buffers = Array.of_list (List.map (List.nth arguments) host_program.globals);
    binders;
    values = Array.make (Array.length binders) 0;
    inputs;
    run_inputs = Array.make (Array.length inputs) (List.hd arguments);
    run_reached = Array.make (Array.length inputs) (List.hd arguments);
    addresses = Array.make (Array.length inputs) 0n;
    touched = arguments @ reached @ List.map (fun (_, _, r) -> r) !reaches;
    table;
    last = Array.make (List.length queues) 0;
  }

(* A loop: a closure over the submission would allocate on every run. *)
let rec report_copies s = function
  | [] -> ()
  | (src, dst, n) :: rest ->
      Nx_device.Submission.copied s ~src ~dst n;
      report_copies s rest

(* Whether the inputs [j] and [k] of a run are the same memory on one device,
   the first input of a run over input [k]'s, and whether any input over it is
   written. *)
let same_input b j k =
  b.inputs.(j).on == b.inputs.(k).on
  && same_memory b.run_inputs.(j) b.run_inputs.(k)

let rec first_same b k j = if same_input b j k then j else first_same b k (j + 1)

let rec any_written b k j =
  j < Array.length b.inputs
  && ((b.inputs.(j).written && same_input b j k) || any_written b k (j + 1))

let run_batch ~env slots b =
  let n = Array.length b.inputs in
  for k = 0 to n - 1 do
    let i = b.inputs.(k) in
    let bs = slots.(i.slot) in
    b.run_inputs.(k) <-
      (match i.shard with Some j -> List.nth bs j | None -> List.hd bs)
  done;
  (* The run's work reads and writes through how its devices reach its inputs,
     which are among its touches until it completes. An input's memory is
     reached once per device, for reading and writing when any input over it is
     written. *)
  let touches = ref b.touched in
  for k = 0 to n - 1 do
    let i = b.inputs.(k) in
    let first = first_same b k 0 in
    if first < k then b.run_reached.(k) <- b.run_reached.(first)
    else begin
      let access = if any_written b k k then B.Read_write else B.Read in
      let r = reach i.on b.run_inputs.(k) access in
      b.run_reached.(k) <- r;
      touches := r :: !touches
    end;
    b.addresses.(k) <-
      Nativeint.add (B.address b.run_reached.(k)) (Nativeint.of_int i.offset)
  done;
  Nx_device.submit b.devices ~touches:!touches (fun s ->
      for i = 0 to Array.length b.queues - 1 do
        if (b.last.(i) > 0) [@mutate off "a wait for 0 returns at once"] then
          Nx_device.Submission.wait s b.queues.(i) b.last.(i)
      done;
      List.iter
        (fun (d', v) ->
          if not (List.memq d' b.devices) then Nx_device.Submission.wait s d' v)
        (Nx_device.Submission.waits s);
      for k = 0 to Array.length b.inputs - 1 do
        b.table.{k} <- Int64.of_nativeint b.addresses.(k)
      done;
      for i = 0 to Array.length b.binders - 1 do
        b.values.(i) <-
          (match b.binders.(i) with
          | Fixed v -> v
          | Submitted d -> Nx_device.submitted d
          | Signals d -> Nx_device.Submission.value s d
          | Var v -> v env)
      done;
      List.iter (fun f -> f ()) b.submitting;
      Nx_device.Program.call b.host_program.program b.buffers b.values;
      report_copies s b.copies;
      if Nx_device.Profile.enabled () then
        List.iter
          (fun (k : Ops.hcq_kernel) ->
            match k.stamps with
            | first :: _ ->
                List.iter
                  (fun dn ->
                    let d = b.named dn in
                    let slots =
                      List.nth b.arguments (List.assoc dn b.info.slots)
                    in
                    Nx_device.Submission.record s d
                      ~lane:
                        (if Option.is_some k.profile_key then "compute"
                         else "copy")
                      ~name:k.name
                      (B.view slots
                         ~offset:(8 * (first - 1))
                         Nx_dtype.Scalar.UInt64 4))
                  k.devices
            | [] -> ())
          b.info.kernels;
      for i = 0 to Array.length b.queues - 1 do
        b.last.(i) <- Nx_device.Submission.value s b.queues.(i)
      done)

(* The calls of a schedule's entry: itself, or those a range is around. *)
let rec calls_of entry =
  match Ops.op entry with
  | Op.End -> calls_of (Ops.nth entry 0)
  | Op.Linear -> List.concat_map calls_of (Ops.src entry)
  | _ -> [ entry ]

let link ~devices ?(bound = []) linear =
  let fn = "Tolk_engine.link" in
  if Ops.op linear <> Op.Linear then fail fn "not a compiled schedule";
  let device = devices in
  let nx name = (devices name).device in
  let storage = Ops.Tbl.create 16 in
  List.iter
    (fun (u, bs) ->
      let ns = names u in
      if Ops.op u <> Op.Buffer || List.length bs <> List.length ns then
        fail fn "%s is not bound to a buffer on each device"
          (Format.asprintf "%a" Ops.pp u);
      List.iter2
        (fun n b ->
          if B.device b != nx n || B.nbytes b < bytes u then
            fail fn "a storage node is not bound to %d bytes of %s" (bytes u) n)
        ns bs;
      Ops.Tbl.replace storage u bs)
    bound;
  let entries = Ops.src linear in
  let nodes =
    List.concat_map
      (fun e ->
        let c = Ops.without_after e in
        let patches = if Ops.op e = Op.After then List.tl (Ops.src e) else [] in
        (* A batch's inputs are in its table, not among its arguments. *)
        let inputs =
          match hcq_info c with
          | Some i -> List.map (fun (base, _, _) -> base) i.inputs
          | None -> []
        in
        List.concat_map Ops.toposort
          (Realize.get_call_arg_uops c @ patches @ inputs))
      (List.concat_map calls_of entries)
  in
  List.iter
    (fun u ->
      if Ops.op u = Op.Buffer && not (Ops.Tbl.mem storage u) then
        Ops.Tbl.replace storage u
          (List.map
             (fun n -> B.create (nx n) Nx_dtype.Scalar.UInt8 (max (bytes u) 1))
             (names u)))
    nodes;
  let params =
    List.sort_uniq compare
      (List.filter_map
         (fun u ->
           match Ops.arg u with
           | Ops.Param p when is_slot u -> Some (p.slot, u)
           | _ -> None)
         nodes)
    |> List.map (fun (slot, u) -> (slot, List.map nx (names u), bytes u))
  in
  let cells = Hashtbl.create 16 in
  let trips r =
    ( cell cells (Ops.expr (Hcq2.range_value r)),
      offset cells (sint (Ops.nth r 0)) )
  in
  let rec linked entry =
    match Ops.op entry with
    | Op.End ->
        let body = Ops.nth entry 0 in
        Range
          {
            ranges = List.map trips (List.tl (Ops.src entry));
            body =
              List.map linked
                (if Ops.op body = Op.Linear then Ops.src body else [ body ]);
          }
    | _ -> linked_call entry
  and linked_call entry =
    let call = Ops.without_after entry in
    let body = Ops.body call in
    match (Ops.op body, hcq_info call) with
    | Op.Store, _ -> (
        match Realize.get_call_arg_uops call with
        | [ dst; src ] ->
            Copy
              {
                call;
                dst = operand storage cells dst;
                src = operand storage cells src;
              }
        | _ -> fail fn "a copy of other than two buffers")
    | Op.Program, Some _ ->
        let patches =
          if Ops.op entry = Op.After then List.tl (Ops.src entry) else []
        in
        Batch (link_batch ~device ~storage ~cells call patches)
    | Op.Program, None ->
        let first = List.hd (Realize.get_call_arg_uops call) in
        Kernel
          {
            call;
            launches =
              Array.of_list
                (List.mapi
                   (fun i n -> launch storage cells call body (nx n) i)
                   (names first));
          }
    | o, _ -> fail fn "%s is no call to run" (Format.asprintf "%a" Op.pp o)
  in
  let calls = List.map linked entries in
  let n = Hashtbl.length cells in
  {
    lock = Mutex.create ();
    calls;
    params;
    cells;
    env = { values = Array.make n 0; set = Array.make n false };
    synced = [];
  }

let check_slots t slots =
  let fn = "Tolk_engine.run" in
  List.iter
    (fun (slot, ds, n) ->
      if slot >= Array.length slots then fail fn "no buffers for slot %d" slot;
      let bs = slots.(slot) in
      if List.length bs <> List.length ds then
        fail fn "slot %d takes %d buffers, not %d" slot (List.length ds)
          (List.length bs);
      List.iter2
        (fun d b ->
          if B.device b != d then
            fail fn "slot %d takes a buffer of %s, not of %s" slot
              (Nx_device.name d)
              (Nx_device.name (B.device b));
          if B.nbytes b < n then
            fail fn "slot %d takes %d bytes, not %d" slot n (B.nbytes b))
        ds bs)
    t.params

(* Reports *)

let reporting () = Setting.value Setting.debug >= 2

(* A run prints its reports as [DEBUG] asks, or nothing: a timing run is
   silent. *)
type reports = Reported | Silent

let reported = function Reported -> reporting () | Silent -> false
let kernels_run = Atomic.make 0

(* One line per kernel, as [DEBUG=2] prints it: its device, how many kernels ran
   before it, its name, its number of arguments and, when known, its time and
   throughput. *)
let report ~device ~name ~args ~vars (e : Ops.estimates) time =
  let count = Atomic.fetch_and_add kernels_run 1 + 1 in
  let timing =
    match time with
    | None -> ""
    | Some s ->
        let per x = Float.of_int (Ops.sym_infer x vars) /. Float.max s 1e-20 in
        Printf.sprintf " tm %s (%7.0f GFLOPS %4.0f GB/s)"
          (Helpers.time_to_str ~w:9 s)
          (per e.ops *. 1e-9)
          (per e.mem *. 1e-9)
  in
  let device = String.sub device 0 (Int.min 7 (String.length device)) in
  Printf.printf "*** %-7s %4d %s arg %2d%s\n%!" device count
    (Helpers.ansipad name 46) args timing

(* The seconds [f] takes on the host clock. *)
let seconds f =
  let t0 = Nx_device.Profile.now () in
  f ();
  Float.of_int (Nx_device.Profile.now () - t0) *. 1e-9

(* A batch's kernels run on its devices, which stamp them: under a profile of
   its own, the batch reports each kernel's span once its devices synchronized.
   While a profile is taken elsewhere, the spans are that profile's, and the
   batch reports no time. *)
let run_reported ~env ~vars slots b =
  let own =
    if Nx_device.Profile.enabled () then None
    else try Some (Nx_device.Profile.start ()) with Invalid_argument _ -> None
  in
  let events =
    match run_batch ~env slots b with
    | () -> (
        match own with
        | Some p -> Nx_device.Profile.stop p
        | None ->
            Array.iter Nx_device.synchronize b.queues;
            [])
    | exception e ->
        Option.iter (fun p -> ignore (Nx_device.Profile.stop p)) own;
        raise e
  in
  let spans = ref events in
  let span d name =
    let rec take = function
      | [] -> (None, [])
      | Nx_device.Profile.Span sp :: rest when sp.device == d && sp.name = name
        ->
          (Some (Float.of_int (sp.stop - sp.start) *. 1e-9), rest)
      | e :: rest ->
          let found, rest = take rest in
          (found, e :: rest)
    in
    let found, rest = take !spans in
    spans := rest;
    found
  in
  List.iter
    (fun (k : Ops.hcq_kernel) ->
      List.iter
        (fun dn ->
          report ~device:dn ~name:k.name
            ~args:(List.length k.input_slots)
            ~vars k.estimates
            (span (b.named dn) k.name))
        k.devices)
    b.info.kernels

(* The variables the run binds and the trips its ranges run, by name, for the
   reports. *)
let bindings t =
  Hashtbl.fold
    (fun name i vars ->
      if t.env.set.(i) then (name, t.env.values.(i)) :: vars else vars)
    t.cells []

(* A host program is outside the devices' ordering: the work that touched its
   buffers, such as a batch's before it, completes first. A device is
   synchronized once until the run queues work again, which only a batch does:
   a copy returns once its bytes have landed. *)
let settle t buffers =
  for k = 0 to Array.length buffers - 1 do
    let d = B.device buffers.(k) in
    if not (List.memq d t.synced) then begin
      Nx_device.synchronize d;
      t.synced <- d :: t.synced
    end
  done

let run_launch t reports call slots l =
  let env = t.env in
  let buffers = Array.map (fun a -> a slots env) l.args in
  settle t buffers;
  let values = Array.map (fun v -> v env) l.vals in
  let split =
    match l.split with
    | None -> None
    | Some (s, extent, ops) ->
        Some (Program.blocks_of s ~extent:(extent env) ~ops:(ops env))
  in
  let launch () =
    Nx_device.Program.call ?split l.program.program buffers values
  in
  if reported reports then
    let vars = bindings t in
    report
      ~device:(Nx_device.name (B.device buffers.(0)))
      ~name:
        (Realize.get_call_name ~var_vals:vars call
           (Realize.get_call_arg_uops call))
      ~args:(Array.length buffers) ~vars
      (Realize.estimate_uop call)
      (Some (seconds launch))
  else launch ()

let rec run_call t reports slots = function
  | Copy { call; dst; src } ->
      let dsts = dst slots t.env and srcs = src slots t.env in
      List.iteri
        (fun i dst ->
          let copy () = B.copy ~src:(lane srcs i) ~dst in
          if reported reports then
            let vars = bindings t in
            report
              ~device:(Nx_device.name (B.device dst))
              ~name:
                (Realize.get_call_name ~var_vals:vars call
                   (Realize.get_call_arg_uops call))
              ~args:2 ~vars
              (Realize.estimate_uop call)
              (Some (seconds copy))
          else copy ())
        dsts
  | Kernel { call; launches } ->
      for i = 0 to Array.length launches - 1 do
        run_launch t reports call slots launches.(i)
      done
  | Batch b ->
      if reported reports then
        run_reported ~env:t.env ~vars:(bindings t) slots b
      else run_batch ~env:t.env slots b;
      t.synced <- []
  | Range { ranges; body } ->
      let env = t.env in
      let rec trips = function
        | [] -> run_calls t reports slots body
        | (c, count) :: rest ->
            let n = count env in
            let value = env.values.(c) and set = env.set.(c) in
            env.set.(c) <- true;
            for i = 0 to n - 1 do
              env.values.(c) <- i;
              trips rest
            done;
            env.values.(c) <- value;
            env.set.(c) <- set
      in
      trips ranges

and run_calls t reports slots = function
  | [] -> ()
  | call :: rest ->
      run_call t reports slots call;
      run_calls t reports slots rest

let run_with reports ~vars t slots =
  check_slots t slots;
  let n = List.length t.calls in
  if
    reports = Reported
    && Setting.value Setting.debug >= 1
    && n >= 10
  then Printf.printf "jit execs %d calls\n%!" n;
  Mutex.protect t.lock (fun () ->
      let env = t.env in
      Array.fill env.set 0 (Array.length env.set) false;
      (* The first binding of a name binds it. *)
      List.iter
        (fun (name, v) ->
          match Hashtbl.find_opt t.cells name with
          | Some i when not env.set.(i) ->
              env.values.(i) <- v;
              env.set.(i) <- true
          | Some _ | None -> ())
        vars;
      t.synced <- [];
      run_calls t reports slots t.calls)

let run ?(vars = []) t slots = run_with Reported ~vars t slots

(* Timing *)

(* Only NV invalidates its caches on demand. *)
let invalidate_caches d =
  Option.iter Nx_nv_device.invalidate_caches (Nx_nv_device.of_device d)

(* The span of [f]'s work named [name] on [d], as [d] stamps it, or, while a
   profile is taken elsewhere, [f] and [d]'s synchronization on the host
   clock. *)
let timed d name f =
  let now = Nx_device.Profile.now in
  if Nx_device.Profile.enabled () then (
    let t0 = now () in
    f ();
    Nx_device.synchronize d;
    now () - t0)
  else
    let p = Nx_device.Profile.start () in
    let events =
      match f () with
      | () -> Nx_device.Profile.stop p
      | exception e ->
          ignore (Nx_device.Profile.stop p);
          raise e
    in
    let span = function
      | Nx_device.Profile.Span sp when sp.device == d && sp.name = name ->
          Some (sp.stop - sp.start)
      | _ -> None
    in
    match List.find_map span events with
    | Some ns -> ns
    | None -> invalid_arg ("Tolk_engine.timer: no span of " ^ name)

(* The storage parameters of [kernel], by slot. *)
let storage_params kernel =
  List.filter_map
    (fun u ->
      match Ops.arg u with
      | Ops.Param { slot; _ } when is_slot u && slot >= 0 -> Some (slot, u)
      | _ -> None)
    (Ops.toposort kernel)
  |> List.sort_uniq (fun (s, _) (s', _) -> Int.compare s s')

let timer ~devices name ~vars kernel =
  let dev = devices name in
  let params = storage_params kernel in
  if List.map fst params <> List.init (List.length params) Fun.id then
    fail "Tolk_engine.timer" "the kernel's parameters skip a slot";
  let args =
    List.map
      (fun (slot, u) ->
        Ops.param
          ~shape:[ Ops.Int (Ops.max_numel u) ]
          ~device:(Single name) slot (Ops.dtype u))
      params
  in
  let slots =
    Array.of_list
      (List.map
         (fun u ->
           [ B.create dev.device Nx_dtype.Scalar.UInt8 (max 1 (bytes u)) ])
         args)
  in
  let on =
    match dev.compiler.queues with
    | None -> Nx_device.host_of dev.device
    | Some _ -> dev.device
  in
  fun prg ->
    let s =
      link ~devices
        (Hcq2.compile_linear ~profile:true
           ~devices:(fun n -> (devices n).compiler)
           (Ops.v Op.Linear ~src:[ Ops.call prg args ]))
    in
    let kernel_name = (Device.Tiny_elf.of_program prg).name in
    fun () ->
      invalidate_caches dev.device;
      Float.of_int
        (timed on kernel_name (fun () -> run_with Silent ~vars s slots))
      *. 1e-9
