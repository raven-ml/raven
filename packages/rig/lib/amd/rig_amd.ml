(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let ( let* ) = Result.bind

(* Sources: the Linux kernel's amdgpu driver for the PM4 and SDMA packet
   headers (nvd.h, soc15d.h, sdma_v*_0_pkt_open.h), its fences (gfx_v12_0.c,
   sdma_v7_0.c) and its flush of the host data path
   (amdgpu_hdp_generic_flush); ROCm's runtime for the AQL queue's descriptor
   (amd_queue.h) and its creation (amd_aql_queue.cpp). *)

module Abi = Rig_amd_abi
module Packet = Abi.Packet
module Aql = Abi.Aql
module Pm4 = Abi.Pm4
module Sdma = Abi.Sdma

exception Fault of string

type 'm memory = { address : int; host : int option; data : 'm }

type 'm path = {
  key : 'm Type.Id.t;
  index : int;
  gpu : Abi.Gpu.t;
  waves : int;
  lds : int;
  clock_hz : int;
  mec : int;
  wgps : int array array;
  budget : int;
  alloc : Rig_edge.memory -> int -> 'm memory option;
  map_host : (int -> int -> 'm memory option) option;
  reaches : int -> bool;
  map_peer : 'm memory -> 'm memory option;
  free : 'm memory -> unit;
  queue :
    [ `Pm4 | `Aql | `Sdma ] ->
    ring:int ->
    bytes:int ->
    read:int ->
    write:int ->
    (int, string) result;
  hdp : int option;
  interrupt : int;
  hang_ms : int option;
  sleep : ms:int -> unit;
  stable_power : unit -> (unit, string) result;
  stop : fault:string option -> [ `Stopped | `Unknown ];
}

(* Whose a path's memory is, by the path's key, and the path's [free]. Memory
   holds its owner, so that a free after stop needs nothing of the device. *)
type 'm owner = { key : 'm Type.Id.t; free : 'm memory -> unit }

(* A path's memory, with its owner. *)
type mem = Mem : 'm owner * 'm memory -> mem

let mem_address (Mem (_, m)) = m.address
let mem_host (Mem (_, m)) = m.host
let free_mem (Mem (o, m)) = o.free m

(* The owner of memory given back, of which a region keeps only the
   addresses. *)
let given : unit owner = { key = Type.Id.make (); free = ignore }

(* The path's functions over memory whose type the owner hides. *)
type ops = {
  alloc : Rig_edge.memory -> int -> mem option;
  map_host : (int -> int -> mem option) option;
  map_peer : mem -> mem option;
  sleep : ms:int -> unit;
  stable_power : unit -> (unit, string) result;
  stop : fault:string option -> [ `Stopped | `Unknown ];
}

let ops (type m) (p : m path) =
  let owner = { key = p.key; free = p.free } in
  let pack m = Mem (owner, m) in
  let own (Mem (o, m)) : m memory option =
    match Type.Id.provably_equal o.key p.key with
    | Some Type.Equal -> Some m
    | None -> None
  in
  {
    alloc = (fun k n -> Option.map pack (p.alloc k n));
    map_host = Option.map (fun f a n -> Option.map pack (f a n)) p.map_host;
    map_peer =
      (fun m -> Option.bind (own m) (fun m -> Option.map pack (p.map_peer m)));
    sleep = p.sleep;
    stable_power = p.stable_power;
    stop = p.stop;
  }

(* Memory *)

type region = {
  bytes : int;
  mem : mem;
  flush : int option; (* the HDP register this region keeps flushed *)
}

let region ?flush n m = { bytes = n; mem = m; flush }

(* C state *)

external create : unit -> int = "caml_rig_amd_create"
external destroy : int -> unit = "caml_rig_amd_destroy" [@@noalloc]

external set_memory : int -> int -> int -> int -> int -> unit
  = "caml_rig_amd_memory"

external set_segment : int -> int -> int -> int -> unit = "caml_rig_amd_segment"

external set_ring : int -> int -> int -> int -> int -> int -> int -> unit
  = "caml_rig_amd_ring_byte" "caml_rig_amd_ring"

external set_template : int -> int -> string -> string -> unit
  = "caml_rig_amd_template"

external set_max_copy : int -> int -> unit = "caml_rig_amd_max_copy"
external hdp_count : int -> int -> int -> bool = "caml_rig_amd_hdp"
external zero : int -> int -> unit = "caml_rig_amd_zero"
external poke32 : int -> int -> int -> unit = "caml_rig_amd_poke32"

external publish_scratch : int -> int -> int array -> int array -> int
  = "caml_rig_amd_scratch"

external scratch_taken : int -> int = "caml_rig_amd_scratch_taken" [@@noalloc]
external signaled_word : int -> int = "caml_rig_amd_signaled" [@@noalloc]

external c_launch_pm4 :
  string -> string -> int array -> int array -> int array -> int
  = "caml_rig_amd_launch_pm4"

external c_launch_aql :
  string -> string -> int array -> int array -> int array -> int
  = "caml_rig_amd_launch_aql"

external free_launch : int -> unit = "caml_rig_amd_launch_free"

external set_copy_launch : int -> int -> unit = "caml_rig_amd_copy_launch"
[@@noalloc]

external write_code : int -> string -> unit = "caml_rig_amd_write_code"
[@@noalloc]

(* Device state *)

(* What growing the scratch answered: [Short bytes] if the GPU has no memory for
   the [bytes] of scratch it needs. *)
type grown = Served | Stopped | Short of int

(* The device's scratch, which every kernel the compute queue runs takes, one at
   a time: installed, published for the next submission to take, and replaced,
   each freed once the word reaches the value that installed its successor. An
   AQL queue's descriptor names it; a launch's dispatch does on a PM4 queue. *)
type scratch = {
  mutable installed : (mem * int) option;
      (* the buffer and its bytes per lane *)
  mutable pending : (mem * int) option;
  mutable retired : (mem * int) list; (* freed once the word reaches the int *)
}

(* What only a running device uses, under the device's guard. The handle
   outlives the device in its properties, so stop drops it. *)
type live = {
  ops : ops;
  own : mem list; (* rings, pointers, segment, slots, private code *)
  copy_launch : int; (* private launch until confirmed stop; 0 when absent *)
  desc : int; (* the compute queue's descriptor *)
  scratch : scratch;
  mutable traces : (Abi.Capability.trace * mem list) option;
      (* the trace buffers, made at the first trace: the capability's record and
         the memory under it *)
}

(* A device: its facts, what [peer] uses for the life of the process, and its
   live state. *)
type t = {
  self : int;
  path : int; (* the path's key, as an integer *)
  index : int; (* the GPU's number in bus order *)
  gpu : Abi.Gpu.t;
  aql : bool; (* the compute queue reads AQL packets *)
  lds : int;
  hdp : int option;
  hdps : Mutex.t; (* the HDP registers' counts and [word] *)
  budget : int;
  hang_ms : int option;
  waits64 : bool; (* its queues run 64-bit waits *)
  maps_host : bool;
  mutable word : region option;
      (* [None] once rig freed it, and the C state with it *)
  (* Where the word is, which the facts report also once it is freed. *)
  word_address : int;
  word_host : int;
  reaches : int -> bool; (* the path's *)
  capability : Abi.Capability.t;
  guard : Mutex.t;
      (* held by [grow_scratch], [make_trace], and stop over [live] *)
  mutable live : live option; (* [None] once stop began *)
}

(* The minimum version of a GC's compute firmware whose queues run 64-bit waits
   (Pm4.wait_64), by GC version: a version enters once a queue was seen
   comparing all 64 bits of a word the host moved across 2^32. GC 12.0.1: the
   R9700, MEC firmware 3010. *)
let wait64_from = [ ((12, 0, 1), 3010) ]
let ring_bytes = 16 lsl 20
let segment_bytes = 1 lsl 20
let slots = 513
let pointers_bytes = 4096

(* RIG_AMD_WAITS, the waits a submission's reserved room holds *)
let max_waits = 255

(* The queues, as rig_amd_ring.c's [runs] accepts their parts. *)
let device_queues =
  Rig_edge.
    [
      { name = "COMPUTE:0"; runs = [ Words; Launch ] };
      { name = "COPY:0"; runs = [ Words; Copy ] };
    ]

(* Whether [g]'s compute queue reads AQL packets, which every die runs: on a GPU
   of several dies, else PM4 packets, which one die runs. *)
let reads_aql (g : Abi.Gpu.t) = g.xccs > 1

(* The queues' positions in the pointers: the compute queue's at the start,
   where an AQL queue's descriptor (amd_hsa_queue.h, amd_queue_t) holds them,
   the copy queue's after the descriptor. *)
let descriptor_bytes = 256
let read_at ~aql = function 0 when aql -> 128 | 0 -> 0 | _ -> descriptor_bytes
let write_at ~aql = function 0 when aql -> 56 | q -> read_at ~aql q + 8

(* The descriptor's fields an AQL queue's creator writes (amd_hsa_queue.h). *)
let max_cu_id = 72
let max_wave_id = 76
let read_dispatch_id_field_base_byte_offset = 136
let queue_properties = 180
let is_ptr64 = 1 lsl 1
let enable_profiling = 1 lsl 3

(* The descriptor's scratch fields. *)
let compute_tmpring_size = 140
let scratch_resource_descriptor = 144
let scratch_backing_memory_location = 160
let scratch_wave64_lane_byte_size = 176

(* Templates *)

(* Each packet the writer places, its values the arguments 0, 1 and 2 of a use,
   in rig_amd_stubs.h's order. On an AQL queue, whose PM4 words every die runs,
   a release writes once: on die 0, after the barrier of its packet, every die's
   work is done. *)
let templates (g : Abi.Gpu.t) ~interrupt ~waits64 ~aql =
  let once p = if aql then Pm4.pred_exec ~xcc_mask:1 p else p in
  [
    Pm4.wait g (Memory 0) Equal 1 ();
    Pm4.event_write Cs_partial_flush;
    Pm4.acquire_mem g System;
    (if waits64 then Pm4.wait_64 g 0 Greater_equal 1 () else []);
    once (Pm4.release_mem g System 0 (Low_32 1));
    once (Pm4.release_mem g System ~interrupt 0 (Data_64 1));
    Pm4.write_data (Memory 0) 1;
    Sdma.poll 0 Equal 1 ();
    Sdma.fence g 0 1;
    Sdma.trap;
    Sdma.copy_linear ~dst:0 ~src:1 ~bytes:2;
    Abi.Aql.indirect_buffer 0 ~dwords:1;
    Pm4.acquire_mem g Agent;
  ]

let set_templates self g ~interrupt ~waits64 ~aql =
  let set i p =
    let words, holes = Template.flatten Packet (fun _ -> None) Fun.id p in
    set_template self i words holes
  in
  List.iteri set (templates g ~interrupt ~waits64 ~aql);
  set_max_copy self (Sdma.max_copy g)

let supported (g : Abi.Gpu.t) =
  let major, _, _ = g.target in
  if not (major = 11 || major = 12 || List.mem g.target [ (9, 4, 2); (9, 5, 0) ])
  then Error (strf "the device drives no %s GPU" (Abi.Gpu.processor g))
  else if Abi.Register.registers g = [] then
    Error (strf "no registers are known for the GC of %s" (Abi.Gpu.processor g))
  else Ok ()

let waits64 (p : _ path) =
  match List.assoc_opt p.gpu.gc wait64_from with
  | Some from -> p.mec >= from
  | None -> false

(* AQL scratch *)

(* The writes that point an AQL queue's descriptor at [desc] to scratch [m] for
   kernels of [n] bytes per lane: GPU addresses and 32-bit values. *)
let scratch_writes (g : Abi.Gpu.t) ~desc m n bytes =
  let base = mem_address m in
  let d = Abi.Scratch.descriptor g ~base bytes in
  let word i = Int32.to_int (String.get_int32_le d (4 * i)) land 0xffff_ffff in
  [
    (desc + scratch_backing_memory_location, base land 0xffff_ffff);
    (desc + scratch_backing_memory_location + 4, base lsr 32);
    (desc + scratch_wave64_lane_byte_size, n);
    (desc + compute_tmpring_size, Abi.Scratch.tmpring g n);
  ]
  @ List.init 4 (fun i ->
      (desc + scratch_resource_descriptor + (4 * i), word i))

(* [p], placed by the submission of [v], replaces the installed scratch, which
   is freed once the word reaches [v]. *)
let install st p v =
  Option.iter (fun (m, _) -> st.retired <- (m, v) :: st.retired) st.installed;
  st.installed <- Some p;
  st.pending <- None

(* Installs a published scratch the queue took, and frees the retired ones the
   word passed. *)
let settle_scratch self st =
  let taken = scratch_taken self in
  (match st.pending with
  | Some p when taken > 0 -> install st p taken
  | _ -> ());
  let word = signaled_word self in
  let reached, kept = List.partition (fun (_, v) -> word >= v) st.retired in
  List.iter (fun (m, _) -> free_mem m) reached;
  st.retired <- kept

(* [grow_scratch] under the device's guard, on a device not stopped. *)
let grow_held self (ops : ops) (g : Abi.Gpu.t) ~aql ~desc st n =
  settle_scratch self st;
  let have =
    match (st.pending, st.installed) with
    | Some (_, h), _ | None, Some (_, h) -> h
    | None, None -> 0
  in
  if have >= n then Served
  else
    let bytes = Abi.Scratch.size g n in
    match ops.alloc Device bytes with
    | None -> Short bytes
    | Some m ->
        (* [m] is this call's until it is published: it is freed if making its
           descriptor raises, as for more scratch per die than a descriptor
           holds. *)
        let ats, values =
          match if aql then scratch_writes g ~desc m n bytes else [] with
          | writes ->
              let ats, values = List.split writes in
              (Array.of_list ats, Array.of_list values)
          | exception e ->
              free_mem m;
              raise e
        in
        let took = publish_scratch self (mem_address m) ats values in
        (* The publication this one replaces: placed by the submission of [took]
           since the settle above, or never placed and never in the queue's use,
           which is freed once [m] is recorded. *)
        (match st.pending with
        | Some p when took > 0 -> install st p took
        | Some _ | None -> ());
        let old = st.pending in
        st.pending <- Some (m, n);
        Option.iter (fun (old, _) -> free_mem old) old;
        Served

let grow_scratch g n =
  Mutex.protect g.guard @@ fun () ->
  match g.live with
  | None -> Stopped
  | Some l -> grow_held g.self l.ops g.gpu ~aql:g.aql ~desc:l.desc l.scratch n

(* Traces *)

(* The runs a device's trace buffers hold, and the bytes each shader engine
   traces into over all of them. *)
let trace_slots = 32
let trace_bytes = 256 lsl 20

(* The trace buffers, in GPU memory the host reads through the BAR where the
   path has some, else in host memory, and the end words in host memory. They
   are made under the device's guard, as the scratch grows, so a stop, which
   frees them, waits for them or refuses them. *)
let make_trace g =
  Mutex.protect g.guard @@ fun () ->
  match g.live with
  | None -> Error "the device is stopped"
  | Some { traces = Some (t, _); _ } -> Ok t
  | Some ({ ops; traces = None; _ } as l) -> (
      let* () = ops.stable_power () in
      let engines = g.gpu.shader_engines * g.gpu.xccs in
      let window = trace_bytes / trace_slots in
      let n = window * trace_slots * engines in
      let buffers =
        match ops.alloc Mapped n with
        | Some m -> Some m
        | None -> ops.alloc Pinned n
      in
      match (buffers, ops.alloc Pinned (4 * trace_slots * engines)) with
      | Some b, Some e when mem_host b <> None && mem_host e <> None ->
          let t =
            {
              Abi.Capability.buffers = mem_address b;
              buffers_host = Option.get (mem_host b);
              window;
              slots = trace_slots;
              engines;
              ends = mem_address e;
              ends_host = Option.get (mem_host e);
            }
          in
          l.traces <- Some (t, [ b; e ]);
          Ok t
      | b, e ->
          Option.iter free_mem b;
          Option.iter free_mem e;
          Error (strf "no memory for %d bytes of trace buffers" n))

(* The capability's [scratch]: [g]'s scratch grown for [n] bytes per lane. *)
let scratch g n =
  match grow_scratch g n with
  | Served -> Ok ()
  | Stopped -> Error "the device is stopped"
  | Short bytes -> Error (strf "no GPU memory for %d bytes of scratch" bytes)

let host_of what m =
  match mem_host m with
  | Some h -> Ok h
  | None -> Error (strf "the host does not address the %s" what)

let is_gpu ~vendor ~class_ =
  let base = class_ lsr 16 in
  vendor = 0x1002 && (base = 0x03 || base = 0x12)

(* Facts *)

let key = Type.Id.make ()

(* [g]'s word once rig freed it: where it was. *)
let freed g =
  let m = { address = g.word_address; host = Some g.word_host; data = () } in
  { bytes = 8; mem = Mem (given, m); flush = None }

let facts g =
  {
    Rig_edge.arch = Abi.Gpu.processor g.gpu;
    budget = g.budget;
    queues = device_queues;
    copy_engine = Some 1;
    completion = Store;
    waits =
      {
        stores = g.waits64;
        hosts = g.waits64;
        objects = false;
        most = max_waits;
      };
    may_block = false;
    hang_ms = g.hang_ms;
    maps_host = g.maps_host;
    host_addresses = false;
    capability = Abi.Capability.declare g.capability;
    word = (match g.word with Some w -> w | None -> freed g);
    edge = Nativeint.of_int g.self;
  }

let capability g = g.capability

(* [g]'s live state. rig calls nothing that uses it after stop. *)
let live g =
  match g.live with
  | Some l -> l
  | None -> invalid_arg "Rig_amd: the device was stopped"

(* Memory *)

(* Counts a region that needs [reg] flushed before a doorbell: [false] if the
   device keeps no more registers. Once [g]'s word is freed no doorbell rings,
   and the count is left. *)
let count_hdp g reg delta =
  Mutex.protect g.hdps (fun () ->
      Option.is_some g.word && hdp_count g.self reg delta)

let alloc g (kind : Rig_edge.memory) n =
  if n < 1 then invalid_argf "Rig_amd.alloc: %d bytes, expected at least 1" n;
  let ops = (live g).ops in
  match kind with
  | Device | Pinned -> Option.map (region n) (ops.alloc kind n)
  | Mapped -> (
      match g.hdp with
      | None -> None
      | Some reg -> (
          match ops.alloc Mapped n with
          | None -> None
          | Some m ->
              ignore (count_hdp g reg 1);
              Some (region ~flush:reg n m)))

(* Gives back [m]: a free the path refuses loses the memory, which no caller can
   act on. *)
let give_back m = try free_mem m with Fault _ -> ()

(* Frees [g]'s C state with its word. rig frees the word after the stop, but
   may free other regions and unload images after it. *)
let free_state g =
  Mutex.protect g.hdps (fun () ->
      g.word <- None;
      destroy g.self)

let free g r =
  Option.iter (fun reg -> ignore (count_hdp g reg (-1))) r.flush;
  give_back r.mem;
  match g.word with Some w when r == w -> free_state g | _ -> ()

(* The GPU names memory by address: the handle is the address. *)
let locate r =
  let a = mem_address r.mem in
  {
    Rig_edge.address = Some a;
    host = mem_host r.mem;
    handle = Nativeint.of_int a;
  }

let peer g g' = g.path = g'.path && g.reaches g'.index

let map_peer g _ r =
  match (live g).ops.map_peer r.mem with
  | None -> None
  | Some m -> (
      let view flush = Some (region ?flush r.bytes m) in
      match r.flush with
      | Some reg when count_hdp g reg 1 -> view (Some reg)
      | Some _ ->
          free_mem m;
          None
      | None -> view None)

let map_host g a n =
  if n < 1 then invalid_argf "Rig_amd.map_host: %d bytes, expected at least 1" n;
  match (live g).ops.map_host with
  | None -> None
  | Some map -> Option.map (region n) (map a n)

(* Images *)

module Code_object = Abi.Code_object

type image = {
  co : Code_object.t;
  base : int; (* the address of the region it was laid over *)
  dev : t;
  lock : Mutex.t;
  launches : (string, int) Hashtbl.t; (* by function: its C launch *)
}

let too_large g co =
  let large name =
    match Code_object.kernel co name with
    | Some k when k.group_segment > g.lds -> Some (name, k.group_segment)
    | _ -> None
  in
  List.find_map large (Code_object.kernels co)

let image g bin =
  let* co = Code_object.of_string bin in
  if not (Code_object.runs_on co g.gpu) then
    Error
      (strf "a code object for %s; the GPU is %s" (Code_object.target co)
         (Abi.Gpu.processor g.gpu))
  else
    match too_large g co with
    | Some (name, n) ->
        Error
          (strf "kernel %s takes %d bytes of local data share; the GPU has %d"
             name n g.lds)
    | None ->
        let lay r =
          let base = mem_address r.mem in
          let launches = Hashtbl.create 8 in
          ( { co; base; dev = g; lock = Mutex.create (); launches },
            Code_object.image co )
        in
        Ok (Rig_edge.Place (Code_object.size co, lay))

(* A dispatch's values: known to its launch, or the hand-over's argument [i], in
   the order of rig_amd_stubs.h's L_ARGS, L_SCRATCH, L_THREADS, L_GROUPS and
   L_GRID. *)
type value = Known of int | Arg of int

let word s i = Int32.to_int (String.get_int32_le s (4 * i)) land 0xffff_ffff

(* The offsets of the implicit arguments [k] reads, in the order of
   rig_amd_stubs.h's H_BLOCKS, H_GROUP, H_REMAINDER, H_OFFSET, H_DIMS and H_LDS,
   or -1. *)
let hidden f (k : Code_object.kernel) =
  let at = Array.make 14 (-1) in
  let axis : Code_object.axis -> int = function X -> 0 | Y -> 1 | Z -> 2 in
  let set (h : Code_object.hidden) off =
    match h with
    | Block_count a -> at.(axis a) <- off
    | Group_size a -> at.(3 + axis a) <- off
    | Remainder a -> at.(6 + axis a) <- off
    | Global_offset a -> at.(9 + axis a) <- off
    | Grid_dims -> at.(12) <- off
    | Dynamic_lds_size -> at.(13) <- off
    | Other kind ->
        invalid_argf
          "Rig_amd.entry: kernel %s reads the implicit argument %s, which a \
           launch does not write"
          f kind
  in
  List.iter (fun (h, off) -> set h off) k.hidden;
  at

(* The values of [k]'s launch the C launch keeps beside its words: the LDS [k]
   takes itself, its most work-items and shared memory on a GPU whose groups
   take at most [lds] bytes, its arguments' bytes, and its explicit ones'. *)
let limits (k : Code_object.kernel) ~lds =
  [|
    k.group_segment;
    k.max_threads;
    lds - k.group_segment;
    k.kernarg_size;
    k.explicit_size;
  |]

let template p =
  Template.flatten Launch
    (function Known n -> Some (Int64.of_int n) | Arg _ -> None)
    (function Arg i -> i | Known _ -> assert false (* known: no hole *))
    p

(* The C launch of kernel [k], named [f], of an image at [base], on [g], whose
   groups take at most [lds] bytes of LDS.

   On a PM4 queue, its dispatch, whose word that holds the LDS its groups take,
   COMPUTE_PGM_RSRC2, the hand-over sets from the launch's shared memory. It is
   the one word that a group segment one granule larger changes.

   On an AQL queue, its kernel dispatch packet, whose workgroup sizes and group
   segment size the hand-over sets at their offsets, and which names the
   kernel's descriptor: the queue hands it its scratch and the packet's
   address. *)
let c_launch_of g (k : Code_object.kernel) ~base ~lds f =
  if reads_aql g then
    let words, holes =
      template
        (Aql.dispatch
           { k with group_segment = 0 }
           ~descriptor:(Known (base + k.descriptor))
           ~args:(Arg 0) ~threads:(1, 1, 1) ~grid:(Arg 8, Arg 9, Arg 10))
    in
    c_launch_aql words holes
      Aql.
        [|
          offset (Workgroup_size X);
          offset (Workgroup_size Y);
          offset (Workgroup_size Z);
          offset Group_segment_size;
        |]
      (limits k ~lds) (hidden f k)
  else
    let dispatch group =
      template
        (Pm4.dispatch g
           { k with group_segment = group }
           ~program:(Known (base + k.entry))
           ~scratch:(Arg 1) ~args:(Arg 0) ~packet:(Known 0)
           ~threads:(Arg 2, Arg 3, Arg 4) ~groups:(Arg 5, Arg 6, Arg 7) ())
    in
    let granule = Pm4.lds_granule g in
    let words, holes = dispatch 0 and more, _ = dispatch granule in
    let at =
      match
        List.filter
          (fun i -> word words i <> word more i)
          (List.init (String.length words / 4) Fun.id)
      with
      | [ at ] -> at
      | _ -> invalid_argf "Rig_amd.entry: kernel %s's LDS is not one word" f
    in
    c_launch_pm4 words holes
      [| at; word words at; word more at - word words at; granule |]
      (limits k ~lds) (hidden f k)

(* [m]'s launch of its kernel [k], named [f], and the scratch of its private
   segment, which grows only for a kernel the launch takes: [Error bytes] if the
   GPU has no memory for the [bytes] of scratch, having made nothing. A PM4
   dispatch hands a kernel no dispatch packet. *)
let launch m f (k : Code_object.kernel) =
  let g = m.dev in
  if k.dispatch_ptr && not g.aql then
    invalid_argf
      "Rig_amd.entry: kernel %s reads its dispatch packet, which a launch on a \
       GPU of one die does not write"
      f;
  let l = c_launch_of g.gpu k ~base:m.base ~lds:g.lds f in
  let grown =
    if k.private_segment > 0 then grow_scratch g k.private_segment else Served
  in
  match grown with
  | Served -> Ok l
  | Short bytes ->
      free_launch l;
      Error bytes
  | Stopped ->
      free_launch l;
      raise (Fault "the device is stopped")

let entry m f =
  match Code_object.kernel m.co f with
  | None -> Rig_edge.Absent
  | Some k -> (
      let made =
        Mutex.protect m.lock @@ fun () ->
        match Hashtbl.find_opt m.launches f with
        | Some l -> Ok l
        | None ->
            let made = launch m f k in
            Result.iter (Hashtbl.add m.launches f) made;
            made
      in
      match made with
      | Ok l ->
          Entry
            {
              code = m.base + k.descriptor;
              launch = Nativeint.of_int l;
              params = Some k.explicit_size;
              threads = Some k.max_threads;
            }
      | Error bytes -> No_memory bytes)

(* The image's code is in a region rig frees; the launches [entry] made go
   here. *)
let unload _ m =
  Mutex.protect m.lock @@ fun () ->
  Hashtbl.iter (fun _ l -> free_launch l) m.launches

(* Opening *)

(* Makes the device's memory and queues, giving back what it took if one of them
   is refused. *)
let make (type m) (p : m path) =
  if p.interrupt = 0 then
    invalid_arg "Rig_amd.make: the release's interrupt context is 0";
  let count field n =
    if n < 1 then
      invalid_argf "Rig_amd.make: the GPU's %s is %d, expected at least 1"
        field n
  in
  count "xccs" p.gpu.xccs;
  count "shader_engines" p.gpu.shader_engines;
  count "compute_units" p.gpu.compute_units;
  count "scratch_slots" p.gpu.scratch_slots;
  let* () = supported p.gpu in
  let ops = ops p in
  let taken = ref [] and queues = ref false and copy_launch = ref 0 in
  (* Memory a queue the path could not stop may still read stays. *)
  let undo () =
    let stopped =
      (not !queues)
      ||
      match ops.stop ~fault:None with
      | `Stopped -> true
      | `Unknown -> false
      | exception Fault _ -> false
    in
    if stopped then begin
      if !copy_launch <> 0 then free_launch !copy_launch;
      List.iter give_back !taken
    end
  in
  let alloc what n =
    match ops.alloc Pinned n with
    | Some m ->
        taken := m :: !taken;
        Ok m
    | None -> Error (strf "no memory for its %s" what)
  in
  let waits64 = waits64 p and aql = reads_aql p.gpu in
  let open_device () =
    let* word = alloc "timeline word" 8 in
    let* slot_words = alloc "slot words" (8 * slots) in
    let* segment = alloc "argument segment" segment_bytes in
    let* compute = alloc "compute ring" ring_bytes in
    let* copy = alloc "copy ring" ring_bytes in
    let* pointers = alloc "queue positions" pointers_bytes in
    let* word_host = host_of "timeline word" word in
    let* slots_host = host_of "slot words" slot_words in
    let* segment_host = host_of "argument segment" segment in
    let* pointers_host = host_of "queue positions" pointers in
    let* compute_host = host_of "compute ring" compute in
    let* copy_host = host_of "copy ring" copy in
    zero pointers_host pointers_bytes;
    if aql then begin
      let cus = p.gpu.compute_units * p.gpu.xccs in
      poke32 pointers_host queue_properties (is_ptr64 lor enable_profiling);
      poke32 pointers_host read_dispatch_id_field_base_byte_offset
        (read_at ~aql 0);
      poke32 pointers_host max_cu_id (cus - 1);
      poke32 pointers_host max_wave_id (p.waves - 1)
    end;
    (* The kernel the device copies with, where Copy_kernel holds one for the
       GPU's processor. rig_amd_ring.c runs a copy of 2 MiB or more within this
       device's Device memory with it, on the compute engine, and every other
       copy on SDMA, whose dispatch costs less. *)
    let* copy_code =
      let processor = Abi.Gpu.processor p.gpu in
      match Copy_kernel.find processor with
      | None -> Ok []
      | Some bin ->
          let name = "copy_bytes" in
          let refused why =
            strf "the copy kernel %s for %s does not load: %s" name processor
              why
          in
          let* co = Result.map_error refused (Code_object.of_string bin) in
          let k = Option.get (Code_object.kernel co name) in
          let* code = alloc "private copy code" (Code_object.size co) in
          let* host = host_of "private copy code" code in
          write_code host (Code_object.image co);
          copy_launch :=
            c_launch_of p.gpu k ~base:(mem_address code) ~lds:p.lds name;
          Ok [ code ]
    in
    let queue q kind ring =
      let at = mem_address pointers in
      let* doorbell =
        p.queue kind ~ring:(mem_address ring) ~bytes:ring_bytes
          ~read:(at + read_at ~aql q)
          ~write:(at + write_at ~aql q)
      in
      queues := true;
      Ok doorbell
    in
    let compute_kind = if aql then `Aql else `Pm4 in
    let* compute_bell = queue 0 compute_kind compute in
    let* copy_bell = queue 1 `Sdma copy in
    (* The queues read none of the C state before a doorbell. *)
    let self = create () in
    set_memory self word_host (mem_address word) slots_host
      (mem_address slot_words);
    set_segment self segment_host (mem_address segment) segment_bytes;
    (* The GPU's own register takes the first of the empty table's slots, with
       no region counted, so that its own [`Mapped] memory always finds one and
       views of other GPUs' take at most the rest. *)
    Option.iter (fun reg -> ignore (hdp_count self reg 0)) p.hdp;
    set_templates self p.gpu ~interrupt:p.interrupt ~waits64 ~aql;
    set_copy_launch self !copy_launch;
    let ring q kind host bell =
      let kind = match kind with `Pm4 -> 0 | `Aql -> 1 | `Sdma -> 2 in
      set_ring self q host ring_bytes
        (pointers_host + write_at ~aql q)
        bell kind
    in
    ring 0 compute_kind compute_host compute_bell;
    ring 1 `Sdma copy_host copy_bell;
    let rec g =
      {
        self;
        path = Type.Id.uid p.key;
        index = p.index;
        gpu = p.gpu;
        aql;
        lds = p.lds;
        hdp = p.hdp;
        hdps = Mutex.create ();
        budget = p.budget;
        hang_ms = p.hang_ms;
        waits64;
        maps_host = Option.is_some ops.map_host;
        word = Some (region 8 word);
        word_address = mem_address word;
        word_host;
        reaches = p.reaches;
        capability =
          {
            gpu = p.gpu;
            clock_hz = p.clock_hz;
            compute =
              (if aql then Aql { scratch = (fun n -> scratch g n) } else Pm4);
            wgps = p.wgps;
            trace = (fun () -> make_trace g);
          };
        guard = Mutex.create ();
        live =
          Some
            {
              ops;
              own = copy_code @ [ slot_words; segment; compute; copy; pointers ];
              copy_launch = !copy_launch;
              desc = mem_address pointers;
              scratch = { installed = None; pending = None; retired = [] };
              traces = None;
            };
      }
    in
    Ok g
  in
  match open_device () with
  | Ok _ as g -> g
  | Error _ as e ->
      undo ();
      e
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      undo ();
      Printexc.raise_with_backtrace e bt

(* Work *)

external last : int -> int = "caml_rig_amd_last" [@@noalloc]

(* Timeline *)

external settle : int -> unit = "caml_rig_amd_settle"

let signaled g = signaled_word g.self

let sleep g ~seen ~still_ms =
  if signaled g = seen then (live g).ops.sleep ~ms:still_ms

(* Loss *)

(* A queue the path could not destroy may still run, so its memory stays and its
   own releases raise the word. *)
let halt g l ~fault =
  match l.ops.stop ~fault with
  | exception Fault _ -> ()
  | `Unknown -> ()
  | `Stopped ->
      settle g.self;
      if l.copy_launch <> 0 then free_launch l.copy_launch;
      let st = l.scratch in
      let buffers = Option.to_list st.installed @ Option.to_list st.pending in
      let traces = match l.traces with Some (_, ms) -> ms | None -> [] in
      List.iter give_back
        (List.map fst (buffers @ st.retired) @ traces @ l.own)

let stop g ~fault =
  let l =
    Mutex.protect g.guard @@ fun () ->
    let l = g.live in
    g.live <- None;
    l
  in
  Option.iter (fun l -> halt g l ~fault) l

(* Tests *)

external renumber_device : int -> int -> int -> unit = "caml_rig_amd_renumber"

let renumber ?(age = 0) g v =
  let last = last g.self in
  if signaled g <> last then
    invalid_arg "Rig_amd.renumber: the device's work runs";
  if v - 1 < last then
    invalid_argf "Rig_amd.renumber: value %d, expected at least %d" v (last + 1);
  if age < 0 || age > v - 1 then
    invalid_argf "Rig_amd.renumber: age %d, expected 0 to %d" age (v - 1);
  renumber_device g.self v age

external placed : int -> int -> int * int = "caml_rig_amd_placed"

let ring g q =
  let named (d : Rig_edge.queue) = d.name = q in
  match List.find_index named device_queues with
  | Some i -> placed g.self i
  | None -> invalid_argf "Rig_amd.ring: no queue %S" q
