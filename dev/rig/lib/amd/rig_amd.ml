(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let ( let* ) = Result.bind

module Abi = Rig_amd_abi
module Packet = Abi.Packet
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
  alloc : [ `Gpu | `Bar | `System ] -> int -> 'm memory option;
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
  stop : unit -> [ `Stopped | `Unknown ];
}

(* A path's memory, with the key that tells whose it is. *)
type mem = Mem : 'm Type.Id.t * 'm memory -> mem

let mem_address (Mem (_, m)) = m.address
let mem_host (Mem (_, m)) = m.host

(* The path's functions over memory whose type the key hides. *)
type ops = {
  alloc : [ `Gpu | `Bar | `System ] -> int -> mem option;
  map_host : (int -> int -> mem option) option;
  reaches : int -> bool;
  map_peer : mem -> mem option;
  free : mem -> unit;
  sleep : ms:int -> unit;
  stop : unit -> [ `Stopped | `Unknown ];
}

let ops (type m) (p : m path) =
  let pack m = Mem (p.key, m) in
  let own (Mem (id, m)) : m memory option =
    match Type.Id.provably_equal id p.key with
    | Some Type.Equal -> Some m
    | None -> None
  in
  {
    alloc = (fun k n -> Option.map pack (p.alloc k n));
    map_host = Option.map (fun f a n -> Option.map pack (f a n)) p.map_host;
    reaches = p.reaches;
    map_peer =
      (fun m -> Option.bind (own m) (fun m -> Option.map pack (p.map_peer m)));
    free = (fun m -> Option.iter p.free (own m));
    sleep = p.sleep;
    stop = p.stop;
  }

(* Memory *)

type region = {
  owner : int; (* the device's C state *)
  bytes : int;
  mem : mem;
  flush : int option; (* the HDP register this region keeps flushed *)
  live : bool Atomic.t; (* taken once, by the free that ends it *)
}

let region owner ?flush n m =
  { owner; bytes = n; mem = m; flush; live = Atomic.make true }

(* C state *)

external create : unit -> int = "caml_rig_amd_create"

external set_memory : int -> int -> int -> int -> int -> unit
  = "caml_rig_amd_memory"

external set_segment : int -> int -> int -> int -> unit = "caml_rig_amd_segment"

external set_ring : int -> int -> int -> int -> int -> int -> int -> unit
  = "caml_rig_amd_ring_byte" "caml_rig_amd_ring"

external set_template : int -> int -> string -> int array -> unit
  = "caml_rig_amd_template"

external set_max_copy : int -> int -> unit = "caml_rig_amd_max_copy"
external hdp_count : int -> int -> int -> bool = "caml_rig_amd_hdp"
external zero : int -> int -> unit = "caml_rig_amd_zero"
external poke32 : int -> int -> int -> unit = "caml_rig_amd_poke32"

external publish_scratch : int -> int array -> int array -> int
  = "caml_rig_amd_scratch"

external scratch_taken : int -> int = "caml_rig_amd_scratch_taken" [@@noalloc]
external signaled_word : int -> int = "caml_rig_amd_signaled" [@@noalloc]
external now_ms : unit -> int = "caml_rig_amd_now_ms" [@@noalloc]

(* Opening *)

type t = {
  self : int;
  path : int; (* the path's key, as an integer *)
  index : int; (* the GPU's number in bus order *)
  gpu : Abi.Gpu.t;
  budget : int;
  lds : int;
  waits64 : bool;
  hdp : int option;
  hdps : Mutex.t; (* the HDP registers' counts *)
  ops : ops;
  capability : Abi.Capability.t;
  word : region;
  own : mem list; (* rings, pointers, segment, slots *)
  scratch : scratch;
  traces : traces;
  hang_ms : int option;
  progress : progress Atomic.t;
  fault : string option Atomic.t; (* the first fault sleep raised *)
}

(* The word as [sleep] last saw it, whether the device was idle then, and since
   when, in milliseconds of the monotonic clock. *)
and progress = { seen : int; idle : bool; since : int }

(* The device's trace buffers, made at the first trace: the capability's record
   and the memory under it. *)
and traces = {
  traces_lock : Mutex.t;
  mutable made : (Abi.Capability.trace * mem list) option;
}

(* An AQL queue's scratch: in its descriptor, published for the next submission
   to write there, and replaced, each freed once the word reaches the value that
   installed its successor. *)
and scratch = {
  lock : Mutex.t;
  mutable installed : (mem * int) option;
      (* the buffer and its bytes per lane *)
  mutable pending : (mem * int) option;
  mutable retired : (mem * int) list; (* freed once the word reaches the int *)
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

(* Templates: each packet the writer places, its values the arguments 0, 1 and 2
   of a use. Their order is rig_amd_stubs.h's. *)

let op_add = 0
let op_shift = 1
let op_or = 2

let hole (at, w) =
  let rec flatten ops : int Packet.term -> _ = function
    | Value i -> (i, ops)
    | Add (t, k) -> flatten ((op_add, Int64.to_int k) :: ops) t
    | Shift (t, n) -> flatten ((op_shift, n) :: ops) t
    | Or (t, k) -> flatten ((op_or, Int64.to_int k) :: ops) t
  in
  let wide, t =
    match (w : int Packet.word) with
    | W32 t -> (1, t)
    | W64 t -> (2, t)
    | Dword _ -> assert false
  in
  let arg, ops = flatten [] t in
  [ at; wide; arg; List.length ops ]
  @ List.concat_map (fun (op, k) -> [ op; k ]) ops

(* On an AQL queue, whose PM4 words every die runs, a release writes once: on
   die 0, after the barrier of its packet, every die's work is done. *)
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
  ]

let set_templates self g ~interrupt ~waits64 ~aql =
  let set i p =
    let words, holes = Packet.template (fun _ -> None) p in
    set_template self i words (Array.of_list (List.concat_map hole holes))
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

external place_entry : unit -> int = "caml_rig_amd_place_entry"
external segment_entry : unit -> int = "caml_rig_amd_segment_entry"

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
let settle_scratch self ops st =
  let taken = scratch_taken self in
  (match st.pending with
  | Some p when taken > 0 -> install st p taken
  | _ -> ());
  let word = signaled_word self in
  let reached, kept = List.partition (fun (_, v) -> word >= v) st.retired in
  List.iter (fun (m, _) -> ops.free m) reached;
  st.retired <- kept

let grow_scratch self ops (g : Abi.Gpu.t) ~desc st n =
  Mutex.protect st.lock @@ fun () ->
  settle_scratch self ops st;
  let have =
    match (st.pending, st.installed) with
    | Some (_, h), _ | None, Some (_, h) -> h
    | None, None -> 0
  in
  if have >= n then Ok ()
  else
    let bytes = Abi.Scratch.size g n in
    match ops.alloc `Gpu bytes with
    | None -> Error (strf "no GPU memory for %d bytes of scratch" bytes)
    | Some m ->
        let ats, values = List.split (scratch_writes g ~desc m n bytes) in
        let took =
          publish_scratch self (Array.of_list ats) (Array.of_list values)
        in
        (* The publication this one replaces: placed by the submission of [took]
           since the settle above, or never placed and never in the queue's
           use. *)
        (match st.pending with
        | Some p when took > 0 -> install st p took
        | Some (old, _) -> ops.free old
        | None -> ());
        st.pending <- Some (m, n);
        Ok ()

(* Traces *)

(* The runs a device's trace buffers hold, and the bytes each shader engine
   traces into over all of them. *)
let trace_slots = 32
let trace_bytes = 256 lsl 20

(* The trace buffers, in GPU memory the host reads through the BAR where the
   path has some, else in host memory, and the end words in host memory. *)
let make_trace (p : _ path) ops st () =
  Mutex.protect st.traces_lock @@ fun () ->
  match st.made with
  | Some (t, _) -> Ok t
  | None -> (
      let* () = p.stable_power () in
      let engines = p.gpu.shader_engines * p.gpu.xccs in
      let window = trace_bytes / trace_slots in
      let n = window * trace_slots * engines in
      let buffers =
        match ops.alloc `Bar n with
        | Some m -> Some m
        | None -> ops.alloc `System n
      in
      match (buffers, ops.alloc `System (4 * trace_slots * engines)) with
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
          st.made <- Some (t, [ b; e ]);
          Ok t
      | b, e ->
          Option.iter ops.free b;
          Option.iter ops.free e;
          Error (strf "no memory for %d bytes of trace buffers" n))

let capability_of (p : _ path) ~aql ~grow ~trace =
  {
    Abi.Capability.gpu = p.gpu;
    clock_hz = p.clock_hz;
    compute = (if aql then Aql { scratch = grow } else Pm4);
    place = Nativeint.of_int (place_entry ());
    segment = Nativeint.of_int (segment_entry ());
    wgps = p.wgps;
    trace;
  }

let host_of what m =
  match mem_host m with
  | Some h -> Ok h
  | None -> Error (strf "the host does not address the %s" what)

(* Makes the device's memory and queues, giving back what it took if one of them
   is refused. *)
let make (type m) (p : m path) =
  if p.interrupt = 0 then
    invalid_arg "Rig_amd.make: the release's interrupt context is 0";
  Option.iter
    (fun n ->
      if n < 1 then
        invalid_argf "Rig_amd.make: a hang bound of %d ms, expected at least 1"
          n)
    p.hang_ms;
  let* () = supported p.gpu in
  let ops = ops p in
  let taken = ref [] and queues = ref false in
  (* Memory a queue the path could not stop may still read stays. *)
  let give_back () =
    let stopped =
      (not !queues)
      ||
      match ops.stop () with
      | `Stopped -> true
      | `Unknown -> false
      | exception Fault _ -> false
    in
    if stopped then
      List.iter (fun m -> try ops.free m with Fault _ -> ()) !taken
  in
  let alloc what n =
    match ops.alloc `System n with
    | Some m ->
        taken := m :: !taken;
        Ok m
    | None -> Error (strf "no memory for its %s" what)
  in
  let waits64 = waits64 p and aql = p.gpu.xccs > 1 in
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
    (* The C state, made once nothing can fail: the queues read none of it
       before a doorbell. *)
    let self = create () in
    set_memory self word_host (mem_address word) slots_host
      (mem_address slot_words);
    set_segment self segment_host (mem_address segment) segment_bytes;
    (* The GPU's own register takes the first of the empty table's slots, with
       no region counted, so that its own [`Mapped] memory always finds one and
       views of other GPUs' take at most the rest. *)
    Option.iter (fun reg -> ignore (hdp_count self reg 0)) p.hdp;
    set_templates self p.gpu ~interrupt:p.interrupt ~waits64 ~aql;
    let ring q kind host bell =
      let kind = match kind with `Pm4 -> 0 | `Aql -> 1 | `Sdma -> 2 in
      set_ring self q host ring_bytes
        (pointers_host + write_at ~aql q)
        bell kind
    in
    ring 0 compute_kind compute_host compute_bell;
    ring 1 `Sdma copy_host copy_bell;
    let scratch =
      { lock = Mutex.create (); installed = None; pending = None; retired = [] }
    in
    let traces = { traces_lock = Mutex.create (); made = None } in
    let trace = make_trace p ops traces in
    let grow =
      grow_scratch self ops p.gpu ~desc:(mem_address pointers) scratch
    in
    let word = region self 8 word in
    Ok
      {
        self;
        path = Type.Id.uid p.key;
        index = p.index;
        gpu = p.gpu;
        budget = p.budget;
        lds = p.lds;
        waits64;
        hdp = p.hdp;
        hdps = Mutex.create ();
        ops;
        capability = capability_of p ~aql ~grow ~trace;
        word;
        own = [ slot_words; segment; compute; copy; pointers ];
        scratch;
        traces;
        hang_ms = p.hang_ms;
        progress = Atomic.make { seen = 0; idle = true; since = 0 };
        fault = Atomic.make None;
      }
  in
  match open_device () with
  | Ok _ as g -> g
  | Error _ as e ->
      give_back ();
      e
  | exception (Fault _ as e) ->
      give_back ();
      raise e

let is_gpu ~vendor ~class_ =
  let base = class_ lsr 16 in
  vendor = 0x1002 && (base = 0x03 || base = 0x12)

(* Facts *)

let key = Type.Id.make ()
let arch g = Abi.Gpu.processor g.gpu
let budget g = g.budget
let queues _ = [ "COMPUTE:0"; "COPY:0" ]
let completion _ = `Store
let waits_on g = function `Store | `Host -> g.waits64 | `Object -> false

(* RIG_AMD_WAITS, the waits a submission's reserved room holds *)
let max_waits _ = 255
let blocks _ = `Returns
let maps_host g = Option.is_some g.ops.map_host

type capability = Abi.Capability.t

let capability g = g.capability
let capability_key = Abi.Capability.key
let self g = Nativeint.of_int g.self

(* Memory *)

(* Counts a region that needs [reg] flushed before a doorbell: [false] if the
   device keeps no more registers. *)
let count_hdp g reg delta =
  Mutex.protect g.hdps (fun () -> hdp_count g.self reg delta)

let alloc g kind n =
  if n < 1 then invalid_argf "Rig_amd.alloc: %d bytes, expected at least 1" n;
  let system () = Option.map (region g.self n) (g.ops.alloc `System n) in
  match kind with
  | `Device -> Option.map (region g.self n) (g.ops.alloc `Gpu n)
  | `Pinned -> system ()
  | `Mapped -> (
      match g.hdp with
      | None -> system ()
      | Some reg -> (
          match g.ops.alloc `Bar n with
          | None -> system ()
          | Some m ->
              ignore (count_hdp g reg 1);
              Some (region g.self ~flush:reg n m)))

(* Gives back [m]: a free the path refuses loses the memory, which no caller can
   act on. *)
let give_back g m = try g.ops.free m with Fault _ -> ()

(* Gives back [r], whose [live] the caller took. *)
let release g r =
  Option.iter (fun reg -> ignore (count_hdp g reg (-1))) r.flush;
  give_back g r.mem

let free g r =
  if r.owner <> g.self || r == g.word then
    invalid_arg
      "Rig_amd.free: the region is no allocation or mapping of the device";
  if not (Atomic.compare_and_set r.live true false) then
    invalid_arg "Rig_amd.free: the region was freed";
  release g r

let address r = Some (mem_address r.mem)
let handle r = Nativeint.of_int (mem_address r.mem)
let host r = mem_host r.mem
let peer g g' = g.self <> g'.self && g.path = g'.path && g.ops.reaches g'.index

let map_peer g g' r =
  if g.self = g'.self then invalid_arg "Rig_amd.map_peer: the devices are one";
  if r.owner <> g'.self || not (Atomic.get r.live) then
    invalid_arg "Rig_amd.map_peer: the region is no live region of the peer";
  match g.ops.map_peer r.mem with
  | None -> None
  | Some m -> (
      let view flush = Some (region g.self ?flush r.bytes m) in
      match r.flush with
      | Some reg when count_hdp g reg 1 -> view (Some reg)
      | Some _ ->
          g.ops.free m;
          None
      | None -> view None)

let map_host g a n =
  if n < 1 then invalid_argf "Rig_amd.map_host: %d bytes, expected at least 1" n;
  match g.ops.map_host with
  | None -> None
  | Some map -> Option.map (region g.self n) (map a n)

(* Images *)

module Code_object = Abi.Code_object

type image = {
  holder : int;
  co : Code_object.t;
  base : int; (* the address of the region it was laid over *)
  loaded : bool Atomic.t; (* taken once, by the unload *)
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
         (arch g))
  else
    match too_large g co with
    | Some (name, n) ->
        Error
          (strf "kernel %s takes %d bytes of local data share; the GPU has %d"
             name n g.lds)
    | None ->
        let lay r =
          let base = mem_address r.mem in
          ( { holder = g.self; co; base; loaded = Atomic.make true },
            Code_object.image co )
        in
        Ok (`Place (Code_object.size co, lay))

let entry m f =
  if not (Atomic.get m.loaded) then
    invalid_arg "Rig_amd.entry: the image was unloaded";
  Option.map
    (fun (k : Code_object.kernel) -> m.base + k.descriptor)
    (Code_object.kernel m.co f)

let unload g m =
  if m.holder <> g.self then
    invalid_arg "Rig_amd.unload: the image is another device's";
  if not (Atomic.compare_and_set m.loaded true false) then
    invalid_arg "Rig_amd.unload: the image was unloaded"

(* Work *)

external last : int -> int = "caml_rig_amd_last" [@@noalloc]
external room_entry : unit -> int = "caml_rig_amd_room_entry"
external submit_entry : unit -> int = "caml_rig_amd_submit_entry"

let room_entry = Nativeint.of_int (room_entry ())
let submit_entry = Nativeint.of_int (submit_entry ())

(* Timeline *)

external settle : int -> unit = "caml_rig_amd_settle"

let word g = g.word
let signaled g = signaled_word g.self

(* A fault is the device's for good: the first one raised is raised again by
   every later call. *)
let faulted g why =
  ignore (Atomic.compare_and_set g.fault None (Some why));
  raise (Fault why)

let path_sleep g ms = try g.ops.sleep ~ms with Fault why -> faulted g why

(* The clock restarts when the word moved or the device was idle at the last
   look, and runs on while the same value stays outstanding. *)
let sleep g ~seen ~still_ms =
  Option.iter (fun why -> raise (Fault why)) (Atomic.get g.fault);
  let w = signaled g in
  if w = seen then
    match g.hang_ms with
    | None -> path_sleep g still_ms
    | Some hang ->
        let now = now_ms () and p = Atomic.get g.progress in
        let idle = last g.self <= w in
        if idle || p.idle || p.seen <> w then begin
          Atomic.set g.progress { seen = w; idle; since = now };
          path_sleep g (if idle then still_ms else Int.min still_ms hang)
        end
        else
          let left = p.since + hang - now in
          if left <= 0 then faulted g (strf "no progress for %d ms" hang);
          path_sleep g (Int.min still_ms left)

(* Loss *)

(* A queue the path could not destroy may still run, so its memory stays and its
   own releases raise the word. *)
let stop g =
  match g.ops.stop () with
  | exception Fault _ -> ()
  | `Unknown -> ()
  | `Stopped ->
      settle g.self;
      let st = g.scratch in
      let buffers = Option.to_list st.installed @ Option.to_list st.pending in
      let traces = match g.traces.made with Some (_, ms) -> ms | None -> [] in
      List.iter (give_back g)
        (List.map fst (buffers @ st.retired) @ traces @ g.own)

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
