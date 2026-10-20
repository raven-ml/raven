(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Any domain may call any function, as the interface says. The C state is
   written by [make], then by [room] and [submit], which rig calls one at a
   time for each device, and by [grow]; [local] holds the device's lock while
   it grows the local memory, [grow] while it makes rings and channels, and
   [stop] while it drops the device's live state. The C state's own locks,
   [flush_mu] and [channels_mu] (rig_nv_stubs.h), are leaves: no OCaml code
   runs while one is held, and no other lock is taken inside it.

   Every lock of the driver's and of its paths is a [Mutex] taken through
   [take_lock], which counts it among the locks its thread holds (rig_hold in
   rig_edge.h). Locks have ranks, in the order they nest: an image's, a
   device's local memory lock, a path's. A take of a lock whose rank is no
   higher than one its thread holds raises [Invalid_argument], having taken
   nothing, so no path lock is taken inside another, and no two threads
   wait for each other. Over PCI, an open holds rig.pci's lock of its GPUs
   around the path's locks, and the path holds a path lock around rig.pci's
   locks of memory. The capability's [local] and the paths' openers check
   the count as they start, as rig's calls do, so a call of theirs from a
   signal handler, finaliser or Gc.Memprof callback inside one of these
   locks raises before it changes anything.

   A path holds its lock across waits of the hardware's: the GSP's RPCs,
   the RM's opens of clients and GPUs, its range counts. A call of rig's from
   a callback during one raises, since those waits must see their state
   whole and cannot end early.

   References:
   - NVIDIA's open-gpu-kernel-modules, releases 570.144, 580, 610 and 615:
     the RM's classes and controls in src/common/sdk/nvidia/inc (cl0080.h,
     cl2080.h, cla06c.h, clc36f.h, cl83de.h, ctrl2080gr.h, ctrl2080perf.h,
     ctrl83debug.h, ctrlc36f.h, nvos.h), and nvstatuscodes.h.
   - NVIDIA's open-gpu-doc: manuals/ampere/ga100/dev_pbdma.ref.txt (GPFIFO
     entries, semaphores, RELEASE_WFI) and dev_ram.ref.txt (USERD, the
     doorbell). *)

module D = Defs
module Abi = Rig_nv_abi
module Template = Abi.Template
module Method = Abi.Method
module Cubin = Abi.Cubin
module Launch = Abi.Launch

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let ( let* ) = Result.bind

exception Fault of string

(* Locks *)

(* The locks the calling thread holds, and a count of one more and one fewer
   ([rig_holds], [rig_hold] and [rig_unhold] in [rig_edge.h]). *)
external holds : unit -> (int[@untagged])
  = "caml_rig_edge_holds_byte" "caml_rig_edge_holds"
[@@noalloc]

external hold : unit -> unit = "caml_rig_edge_hold" [@@noalloc]
external unhold : unit -> unit = "caml_rig_edge_unhold" [@@noalloc]
external reraise : exn -> 'a = "%reraise"

(* The rank of the last lock of the driver's the calling thread took, [-1]
   for none, and its change. *)
external held_rank : unit -> (int[@untagged])
  = "caml_rig_nv_rank_byte" "caml_rig_nv_rank"
[@@noalloc]

external set_rank : (int[@untagged]) -> unit
  = "caml_rig_nv_set_rank_byte" "caml_rig_nv_set_rank"
[@@noalloc]

(* A lock, its rank, and the rank its holder held before it took it. *)
type lock = { mutex : Mutex.t; rank : int; mutable below : int }

let image_rank = 0
let local_rank = 1
let path_rank = 2
let ranked rank = { mutex = Mutex.create (); rank; below = -1 }

let refused () =
  invalid_arg
    "Rig_nv: a lock taken inside one of its rank or a later one: a call \
     from a signal handler, finaliser or Gc.Memprof callback skipped its \
     check, or a path took a lock inside one of its own"

(* [take_lock l] takes [l], counted from before the take, so that a signal
   handler that runs inside a contended take sees it held. A thread that
   holds a lock of [l]'s rank or a later one, [l] included, raises having
   taken nothing. An exception that ends the take, such as one a handler
   raised, gives the count back first. *)
let take_lock l =
  if held_rank () >= l.rank then refused ();
  hold ();
  match Mutex.lock l.mutex with
  | () ->
      l.below <- held_rank ();
      set_rank l.rank
  | exception e ->
      unhold ();
      reraise e

(* Gives [l] back, then its count, with nothing between that allocates or
   polls. *)
let give_lock l =
  set_rank l.below;
  Mutex.unlock l.mutex;
  unhold ()

(* [l] is taken as the last step before the handler and given back as the
   first step of both exits, where nothing allocates or polls, so an
   exception anywhere in [f], an asynchronous one too, leaves it free and
   the count right. *)
let protect l f =
  take_lock l;
  match f () with
  | x ->
      give_lock l;
      x
  | exception e ->
      give_lock l;
      reraise e

(* [try_protect l f] takes [l] as [take_lock] does, but answers [false],
   having taken nothing, where the take would wait or raise. *)
let try_protect l f =
  if held_rank () >= l.rank then false
  else begin
    hold ();
    match Mutex.try_lock l.mutex with
    | false ->
        unhold ();
        false
    | exception e ->
        unhold ();
        reraise e
    | true -> (
        l.below <- held_rank ();
        set_rank l.rank;
        match f () with
        | () ->
            give_lock l;
            true
        | exception e ->
            give_lock l;
            reraise e)
  end

(* A function a caller outside rig makes, and that may take a lock, starts
   with [check_held fn]. *)
let check_held fn =
  if holds () <> 0 then
    invalid_arg
      (fn
     ^ ": called from a signal handler, finaliser or Gc.Memprof callback \
        inside a lock of the calling thread's")

module Lock = struct
  type t = lock

  let create () = ranked path_rank
  let protect = protect
  let try_protect = try_protect
  let check = check_held
end

(* Paths *)

type params =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

type refusal = Lacking of string | Refused of string

type rm = {
  release : int;
  client : int;
  alloc : parent:int -> int -> params option -> (int, refusal) result;
  control : int -> int -> params option -> (unit, refusal) result;
  free : (parent:int -> int -> (unit, refusal) result) option;
}

type gpu = {
  channel_class : int;
  compute_class : int;
  copy_class : int;
  sm_version : int;
  gpcs : int;
  tpcs_per_gpc : int;
  sms_per_tpc : int;
  warps_per_sm : int;
}

type 'm memory = { address : int; host : int option; handle : int; data : 'm }
type clock = Source of int | Registers of int

type 'm path = {
  key : 'm Type.Id.t;
  index : int;
  rm : rm;
  device : int;
  subdevice : int;
  vaspace : int;
  gpu : gpu;
  budget : int;
  doorbell : int;
  flush : int option;
  alloc : Rig_edge.memory -> int -> 'm memory option;
  map_host : (int -> int -> 'm memory option) option;
  reaches : int -> bool;
  map_peer : 'm memory -> 'm memory option;
  free : 'm memory -> unit;
  register : int -> (unit, string) result;
  unregister : int -> (unit, string) result;
  check : unit -> unit;
  hang_ms : int option;
  clock : clock;
  stop : unit -> [ `Stopped | `Unknown ];
}

let is_gpu ~vendor ~class_ = vendor = 0x10de && class_ lsr 16 = 0x03

(* GPU addresses. Ring segments, launch descriptors and semaphores lie below
   2^40: a GPFIFO entry holds 40 bits of a segment's address (clc56f.h,
   GP_ENTRY1_GET_HI 7:0), and a semaphore method and a descriptor's release
   40 bits of theirs (SEM_ADDR_HI_OFFSET 7:0; clc9c0qmd.h,
   RELEASE0_ADDRESS_UPPER 807:800). Kernels see their local and shared memory
   through windows in two slots of [window_slot] bytes below 2^40, the local one
   first. Each starts on 4 GiB, as Hopper's shared window must, and spans less:
   Mesa's NVK places the two 16 MiB apart (nvk_cmd_dispatch.c). *)
let semaphore_limit = 1 lsl 40
let window_slot = 4 lsl 30
let windows = semaphore_limit - (2 * window_slot)
let local_window = windows
let shared_window = windows + window_slot

(* Where a path's memory must lie: memory the host addresses, where the device
   places its own words, rings and descriptors, below the windows; Device
   memory anywhere outside them. *)
type reach = Below_windows | Outside_windows

let reach_of : Rig_edge.memory -> reach = function
  | Device -> Outside_windows
  | Pinned | Mapped -> Below_windows

(* The memory [m] of [n] bytes a path answered, where every path's interface
   promises it lies. *)
let placed reach n = function
  | Some (m : _ memory) as r ->
      let last = m.address + n in
      (match reach with
      | Below_windows -> assert (last <= windows)
      | Outside_windows ->
          assert (last <= windows || m.address >= semaphore_limit));
      r
  | None -> None

let path_alloc (p : _ path) kind n = placed (reach_of kind) n (p.alloc kind n)

(* Constants *)

let page = 4096

(* Both paths place each region on a page of at least 4 KiB: the RM path at
   addresses aligned to its page ([Rig_nv_nvidia]), the PCI path in its GPU
   space ([Rig_pci.Memory.alloc]). *)
let region_alignment = page

(* The device's own words: a timeline word, used where the path maps no host
   memory, then the channels' join words, as rig_nv_stubs.h's JOIN_GPU lays
   them out. *)
let words_bytes = 8 * 3

(* The bytes, from a 64 KiB boundary of rig's array of words, that the device
   maps to reach its word there: whole pages on every host. *)
let chunk = 1 lsl 16

(* Each channel's ring holds [entries] entries at open, and its segment ring
   [segment_bytes] bytes, both powers of two. A submission's launches and
   copies take ring entries only to close the driver's segments: at most a
   segment ring's bytes, which lie below the windows, so at most 2^39, the
   largest power of two there; and an entry for each [SEGMENT_BYTES_MAX / 2]
   of them, 2^22 - 2 bytes (rig_nv_ring.c's [segment_closes]), 2^17 entries,
   with 3 more each channel takes. [entries] holds them, so only Words make
   a device's channels anew. *)
let entries = 1 lsl 18
let segment_bytes = 1 lsl 20

(* The most values in flight, a power of two: each holds a slot of the stamp
   ring, one per value modulo their number (rig_nv_stubs.h's Times), and a
   mark of each channel it uses. A slot is three records of 16 bytes. *)
let slots = 1 lsl 15
let stamps_bytes = slots * 48

(* The compute channel's launch ring holds [launch_bytes] bytes of launches'
   descriptors and banks, a power of two: a launch takes about 1 KiB. *)
let launch_bytes = 1 lsl 20

(* rig_nv_stubs.h's templates, by index. *)
let t_acquire = 0
let t_release = 1
let t_copy_release = 2
let t_copy = 3
let t_local = 4
let t_setup = 5
let t_setup_copy = 6
let t_invalidate = 7
let t_idle = 8
let t_schedule = 9
let t_stamp = 10
let t_copy_stamp = 11
let t_fill = 12
let t_fill_bytes = 13
let t_box = 14

(* rig_nv_stubs.h's launch values, by slot: the grid's sizes along X, Y and Z,
   the group's, bank 0's address, the next descriptor's, the dynamic shared
   memory. *)
let v_grid_x = 0
let v_block_x = 3
let v_bank0 = 6
let v_next = 7
let v_shared = 8

(* The sizes in the order of rig_nv_stubs.h's launch limits. *)
let dims = Abi.Qmd.[ Grid X; Grid Y; Grid Z; Block X; Block Y; Block Z ]

(* A pending local memory is one word: bit 0 clear, the number of its address's
   4 KiB page in the [local_page_bits] bits above, which hold the 49 bits of an
   address the local memory methods take (clc7c0.h,
   SET_SHADER_LOCAL_MEMORY_A_ADDRESS_UPPER 16:0), and its bytes per cluster in
   units of 32 KiB above them. The submission of value [v] that takes it leaves
   [v lsl 1 lor 1] in its place. *)
let local_page_shift = 12
let local_page_bits = 37
let local_unit_shift = 15

let local_word address ~per_tpc =
  assert (address land ((1 lsl local_page_shift) - 1) = 0);
  assert (address lsr local_page_shift < 1 lsl local_page_bits);
  assert (
    per_tpc lsr local_unit_shift < 1 lsl (Sys.int_size - 2 - local_page_bits));
  ((per_tpc lsr local_unit_shift) lsl (1 + local_page_bits))
  lor ((address lsr local_page_shift) lsl 1)

(* The multiprocessors whose errors the RM reports, at most. *)
let sm_errors =
  let _, _, n = D.Sm_error_states.sm_error_state_array in
  n

(* The C state *)

external create : int -> int -> int -> int -> int -> int = "caml_rig_nv_create"
[@@noalloc]

external destroy : int -> unit = "caml_rig_nv_destroy" [@@noalloc]

external set_channels : int -> int -> int array -> unit
  = "caml_rig_nv_channels"
[@@noalloc]

external channels_record : int -> int array -> int
  = "caml_rig_nv_channels_record"
[@@noalloc]

external hand_channels : int -> int -> unit = "caml_rig_nv_hand_channels"
[@@noalloc]

external set_joins : int -> int -> unit = "caml_rig_nv_joins" [@@noalloc]

external channels_state : int -> int * int = "caml_rig_nv_channels_state"

external free_channels : int -> unit = "caml_rig_nv_channels_free"
[@@noalloc]

external set_segments : int -> int -> int -> int -> int -> unit
  = "caml_rig_nv_segments"
[@@noalloc]

external set_doorbell : int -> int -> unit = "caml_rig_nv_doorbell" [@@noalloc]
external set_stamps : int -> int -> int -> int -> int -> bool
  = "caml_rig_nv_stamps"

external set_clock : int -> int -> int -> bool = "caml_rig_nv_clock"

external set_launches : int -> int -> int -> int -> bool -> unit
  = "caml_rig_nv_launches"
[@@noalloc]

external set_template : int -> int -> string -> string -> unit
  = "caml_rig_nv_template"

external set_entry : int -> int -> int -> unit = "caml_rig_nv_entry" [@@noalloc]
external set_flush : int -> int -> unit = "caml_rig_nv_bar" [@@noalloc]
external count_flush : int -> int -> int -> bool = "caml_rig_nv_flush"
[@@noalloc]

external drop_flush : int -> unit = "caml_rig_nv_drop" [@@noalloc]
external zero : int -> int -> unit = "caml_rig_nv_zero" [@@noalloc]
external copy_word : int -> int -> unit = "caml_rig_nv_copy_word" [@@noalloc]

(* rig's array of timeline words: its address and its bytes. *)
external array_of_words : unit -> int * int = "caml_rig_edge_words"

external exchange_local : int -> int -> int = "caml_rig_nv_exchange_local"
[@@noalloc]

external pending_local : int -> int = "caml_rig_nv_pending_local" [@@noalloc]
external owe_invalidate : int -> unit = "caml_rig_nv_owe_invalidate" [@@noalloc]
external read_word : int -> int = "caml_rig_nv_signaled" [@@noalloc]

external notification : int -> int -> int = "caml_rig_nv_notification"
[@@noalloc]

external new_launch : string array -> int array -> nativeint
  = "caml_rig_nv_launch"

external free_launch : nativeint -> unit = "caml_rig_nv_launch_free" [@@noalloc]

external free_launches : nativeint list -> unit = "caml_rig_nv_launches_free"
[@@noalloc]

external needed : int -> int array = "caml_rig_nv_needed"

external hand : int -> int -> int -> int -> int -> bool -> int
  = "caml_rig_nv_hand_bytecode" "caml_rig_nv_hand"
[@@noalloc]

external retired : int -> int -> int = "caml_rig_nv_retired" [@@noalloc]
external rings_retired : int -> bool = "caml_rig_nv_retiring" [@@noalloc]

external install_kernels : int -> nativeint -> nativeint -> int -> string -> unit
  = "caml_rig_nv_kernels"
[@@noalloc]

external raise_word : int -> unit = "caml_rig_nv_raise" [@@noalloc]
external end_state : int -> bool -> unit = "caml_rig_nv_end" [@@noalloc]

(* RM parameters *)

external get16 : params -> int -> int = "%caml_bigstring_get16"
external get32 : params -> int -> int32 = "%caml_bigstring_get32"
external get64 : params -> int -> int64 = "%caml_bigstring_get64"
external set16 : params -> int -> int -> unit = "%caml_bigstring_set16"
external set32 : params -> int -> int32 -> unit = "%caml_bigstring_set32"
external set64 : params -> int -> int64 -> unit = "%caml_bigstring_set64"

let params n =
  let p = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n in
  Bigarray.Array1.fill p '\000';
  p

let get p (at, n) =
  match n with
  | 1 -> Char.code (Bigarray.Array1.get p at)
  | 2 -> get16 p at
  | 4 -> Int32.to_int (get32 p at) land 0xffff_ffff
  | 8 -> Int64.to_int (get64 p at)
  | n -> invalid_argf "Rig_nv.get: a field of %d bytes" n

let set p (at, n) v =
  match n with
  | 1 -> Bigarray.Array1.set p at (Char.unsafe_chr (v land 0xff))
  | 2 -> set16 p at (v land 0xffff)
  | 4 -> set32 p at (Int32.of_int v)
  | 8 -> set64 p at (Int64.of_int v)
  | n -> invalid_argf "Rig_nv.set: a field of %d bytes" n

(* Element [i] of the array field (offset, bytes of an element, elements), and
   the field [f] of that element. *)
let elt (at, n, _) i = (at + (i * n), n)
let elt_field (at, n, _) i (f, fn) = (at + (i * n) + f, fn)
let bits (lo, n) v = (v land ((1 lsl n) - 1)) lsl lo

(* Devices *)

(* A local memory the GPU's work no longer uses once the device reached
   [until], the value of the submission that took its successor. *)
type 'm retired = { mem : 'm memory; mutable until : int }

(* A ring a growth makes, listed before the path's call so that its memory is
   reachable from what stop frees as the path returns it: [None] until
   then. *)
type 'm ring = { mutable ring : 'm memory option }

(* A set of the device's two channels: the GPU address of their rings, by
   which rig_nv_stubs.h's struct channels names them; each ring's entries;
   the Pinned memory of their rings and USERDs, and of their error
   notifiers; their group, whose context they share; the debugger, which
   reads the compute engine's errors; the channels, compute's first; and the
   C record of a set a growth handed, 0 until then and for the set made at
   open. The RM fixes a ring's entries when it allocates its channel, so a
   ring grows as a new set. *)
type 'm channels = {
  id : int;
  entries : int array;
  block : 'm memory;
  notifiers : 'm memory array;
  group : int;
  debugger : int;
  handles : int array;
  record : int;
}

(* What only a running device uses. The handle outlives the device in its
   properties, so stop drops it. *)
type 'm live = {
  path : 'm path;
  error_names : (int * string) list;
  owned : 'm memory list;
  mutable rings : 'm ring list;
      (* the rings growths made, under [local_lock], until the device retired
         them and its word passed their work (rig_nv_stubs.h) *)
  release : (module D.RELEASE);
  mutable sets : 'm channels list;
      (* the sets of channels, the newest first, under [local_lock]: the one
         installed, the one handed, and those not yet given back *)
  mutable per_thread : int; (* under [local_lock], as the fields below *)
  mutable local_current : 'm memory option;
  mutable local_pending : 'm memory option;
  mutable local_retired : 'm retired list;
}

(* A device: its facts, what [peer] uses for the life of the process, and its
   live state. *)
type 'm dev = {
  self : int;
  arch : string;
  budget : int;
  hang_ms : int option;
  maps_host : bool;
  key : 'm Type.Id.t;
  index : int;
  reaches : int -> bool; (* the path's *)
  capability : Abi.Gpu.t;
  flush : int option; (* the path's flush word *)
  word : 'm reg; (* kept, with the C state, for the process *)
  local_lock : lock;
      (* held by [grow_local], [grow] and the retirement of rings, and by stop
         over [live] *)
  mutable live : 'm live option; (* [None] once stop began *)
}

(* A region holds what frees it, the path's [free], so that a free after stop
   needs nothing of the device. *)
and 'm reg = {
  dev : 'm dev;
  mem : 'm memory;
  free : 'm memory -> unit;
  bytes : int;
  flush : int option;
      (* the device whose BAR the host writes the memory through, by its
         [self], for Mapped memory and views of it *)
}

type t = T : 'm dev -> t

type region = R : 'm reg -> region

let key : t Type.Id.t = Type.Id.make ()

(* Facts *)

(* The architecture of SM version [v]: its major and minor numbers, but for
   Blackwell's 0xa04, which is sm_120. *)
let arch_of v =
  if v = 0xa04 then "sm_120"
  else
    let minor = v land 0xff in
    strf "sm_%d%d"
      ((v lsr 8) land 0xff)
      (if minor > 0xf then minor lsr 4 else minor)

(* The SASS version of SM version [v], as cubins state it: major and minor in
   one byte each nibble. *)
let sass_of v = ((v land 0xf00) lsr 4) lor (v land 0xf)

(* The logical queues: the writer preserves each queue's order across its
   choices of hardware channel. *)
let queues =
  Rig_edge.
    [
      { name = "COMPUTE:0"; runs = [ Words; Launch ] };
      { name = "COPY:0"; runs = [ Words; Copy ] };
    ]

(* Its channels wait on any 64-bit word the device maps, whoever writes it. *)
let waits = { Rig_edge.stores = true; hosts = true }

let capability (T d) = d.capability

let facts (T d) =
  {
    Rig_edge.arch = d.arch;
    budget = d.budget;
    queues;
    copy_engine = Some 1;
    completion = Store;
    waits;
    may_block = false;
    hang_ms = d.hang_ms;
    maps_host = d.maps_host;
    host_addresses = false;
    alignment = region_alignment;
    capability = Abi.declare d.capability;
    word = R d.word;
    edge = Nativeint.of_int d.self;
  }

(* Gives [m] back through [free], its path's. The device holds [m] no more
   whatever the path answers: a path that fails to take it back keeps it. *)
let give free m = try free m with Fault _ -> ()

(* [d]'s live state. rig calls nothing that uses it after stop. *)
let live d =
  match d.live with
  | Some l -> l
  | None -> invalid_arg "Rig_nv: the device was stopped"

(* Local memory *)

(* The pending local memory's word [x] says a submission took it iff its bit 0
   is set, and then holds the submission's value above it. *)
let taken x = x land 1 = 1
let taker x = x lsr 1

(* What making the pending memory current stores, allocated ahead, so that a
   change allocates nothing between its first store and its last: the current
   memory retired, its value to be set, and the retired list it heads. *)
let retiring l =
  match l.local_current with
  | Some c ->
      let r = { mem = c; until = 0 } in
      (Some r, r :: l.local_retired)
  | None -> (None, l.local_retired)

(* Makes the pending memory, [current], current from the value [v], which
   the current one served up to. *)
let succeed l (r, retired) current v =
  (match r with Some r -> r.until <- v | None -> ());
  l.local_retired <- retired;
  l.local_current <- current

(* Moves the pending local memory to current once a submission took it. *)
let settle d l =
  match l.local_pending with
  | None -> ()
  | Some _ as current ->
      let x = pending_local d.self in
      if taken x then begin
        let ahead = retiring l in
        succeed l ahead current (taker x);
        l.local_pending <- None
      end

(* Frees the retired local memories no work uses, those whose successor's
   value is reached, one at a time: each leaves the list before its free, so
   an exception leaves the rest for the next call. *)
let rec retire d l =
  let seen = read_word d.self in
  match List.find_opt (fun r -> r.until <= seen) l.local_retired with
  | None -> ()
  | Some r ->
      l.local_retired <- List.filter (fun r' -> r' != r) l.local_retired;
      give l.path.free r.mem;
      retire d l

(* Makes [m], whose word is [packed], the pending local memory, freeing a
   pending one that no submission took. The exchange hands [m] to the
   submissions, so what it decides is allocated before it. *)
let publish d l m packed =
  let pending = Some m and current = l.local_pending in
  let ahead = retiring l in
  let x = exchange_local d.self packed in
  match current with
  | Some _ when taken x ->
      succeed l ahead current (taker x);
      l.local_pending <- pending
  | Some p ->
      l.local_pending <- pending;
      give l.path.free p
  | None -> l.local_pending <- pending

type served = Served | Stopped | Short of int

(* Makes the local memory serve kernels whose threads need [n] bytes each:
   [Short bytes] if the GPU has no memory for the [bytes] that takes. Raises
   [Fault] if the path does. *)
let grow_local d n =
  protect d.local_lock @@ fun () ->
  match d.live with
  | None -> Stopped
  | Some l -> (
      settle d l;
      retire d l;
      let need = Abi.Local_memory.make d.capability n in
      if need.per_thread <= l.per_thread then Served
      else
        match path_alloc l.path Device need.bytes with
        | None -> Short need.bytes
        | Some m ->
            publish d l m (local_word m.address ~per_tpc:need.per_tpc);
            l.per_thread <- need.per_thread;
            Served)

(* Retired rings *)

(* Gives back the ring at the GPU address [a], which no work uses. It leaves
   [rings] just before its free, with nothing allocated between. *)
let drop_ring l a =
  let at r = match r.ring with Some m -> m.address = a | None -> false in
  match List.find_opt at l.rings with
  | Some ({ ring = Some m; _ } as r) ->
      l.rings <- List.filter (fun r' -> r' != r) l.rings;
      give l.path.free m
  | Some { ring = None; _ } | None -> ()

(* Gives back the rings the device retired whose work its word passed. *)
let rec retire_rings d l =
  match retired d.self (read_word d.self) with
  | 0 -> ()
  | a ->
      drop_ring l a;
      retire_rings d l

(* Retired channels *)

let why = function Lacking why | Refused why -> why

(* Whether the RM freed its object [h] under [parent]: [false] if it refused,
   or frees no object. *)
let freed (p : _ path) ~parent h =
  match p.rm.free with
  | None -> false
  | Some free -> (
      match free ~parent h with
      | Ok () -> true
      | Error _ | (exception Fault _) -> false)

(* Gives back the set [s]: its channels, then their memory, which the GPU no
   longer reads once the RM freed them; a set the RM did not free keeps its
   memory. *)
let drop_channels l s =
  let unregistered =
    Array.for_all Fun.id
      (Array.map
         (fun ch ->
           match l.path.unregister ch with
           | Ok () -> true
           | Error _ | (exception Fault _) -> false)
         s.handles)
  in
  let device = l.path.device in
  let freed =
    freed l.path ~parent:device s.debugger
    && freed l.path ~parent:device s.group
  in
  if unregistered && freed then begin
    give l.path.free s.block;
    Array.iter (give l.path.free) s.notifiers
  end;
  if s.record <> 0 then free_channels s.record

(* Gives back the sets of channels neither installed nor handed. No work
   uses them: a room installs a set once the device reached every value
   submitted, and a set replaced before its install ran nothing. Where the
   path's RM frees no object, they stay until the device stops. *)
let retire_channels d l =
  match l.sets with
  | [] | [ _ ] -> ()
  | _ when Option.is_none l.path.rm.free -> ()
  | sets ->
      let installed, handed = channels_state d.self in
      let kept s = s.id = installed || s.id = handed in
      l.sets <- List.filter kept sets;
      List.iter (drop_channels l) (List.filter (fun s -> not (kept s)) sets)

let local d n =
  check_held "Rig_nv_abi.Gpu.local";
  match grow_local d n with
  | Served -> Ok ()
  | Stopped -> Error "the device is stopped"
  | Short bytes ->
      Error (strf "no GPU memory for %d bytes of local memory" bytes)
  | exception Fault why -> Error why

(* Templates *)

(* [t]'s patches as rig_nv_stubs.c reads them: five little-endian 64-bit words
   each, its byte, bits, slot, addition and shift. *)
let patches (t : Template.t) =
  let b = Buffer.create 256 in
  let add n = Buffer.add_int64_le b (Int64.of_int n) in
  List.iter
    (fun (p : Template.patch) ->
      add p.at;
      add p.bits;
      add p.slot;
      Buffer.add_int64_le b p.add;
      add p.shift)
    t.patches;
  Buffer.contents b

let template self k (t : Template.t) = set_template self k t.bytes (patches t)

let templates self (g : gpu) =
  (* The writer's values, by slot: two addresses or an address and a value, then
     a size, then a box's rows and pitches. *)
  let a = Template.Slot 0 and b = Template.Slot 1 and n = Template.Slot 2 in
  let rows = Template.Slot 3
  and dst_pitch = Template.Slot 4
  and src_pitch = Template.Slot 5 in
  template self t_acquire (Method.acquire a b);
  template self t_release (Method.release a b);
  template self t_copy_release (Method.copy_release a b);
  template self t_copy (Method.copy ~dst:a ~src:b n);
  template self t_box
    (Method.box ~dst:a ~src:b ~dst_pitch ~src_pitch ~rows n);
  template self t_fill (Method.fill Four ~dst:a b n);
  template self t_fill_bytes (Method.fill One ~dst:a b n);
  template self t_local (Method.local_memory a ~per_tpc:b);
  template self t_setup
    (Template.concat
       [
         Method.set_object Method.Compute g.compute_class;
         Method.local_memory_window (Int local_window);
         Method.shared_memory_window (Int shared_window);
       ]);
  template self t_setup_copy (Method.set_object Method.Copy g.copy_class);
  template self t_invalidate Method.invalidate_caches;
  template self t_idle Method.wait_for_idle;
  template self t_schedule (Method.schedule a);
  template self t_stamp (Method.release_stamp a b);
  template self t_copy_stamp (Method.copy_release_stamp a b);
  (* An entry is its segment's address plus a constant plus its words times
     another: the address is its own field, and the words another. *)
  let entry words =
    let e = Abi.Gpfifo.entry (Int 0) ~offset:0 ~words in
    Int64.to_int (String.get_int64_le e.bytes 0)
  in
  set_entry self (entry 0) (entry 1 - entry 0)

(* The image of [c] for an upload at [base]: its object's image, zeros up to its
   size, and its relocations' patches. *)
let image_bytes c ~base =
  let o = Cubin.elf c in
  let b = Bytes.make (Cubin.size c) '\000' in
  let put (s : Rig_elf.section) =
    match s.offset with
    | Some off -> Bytes.blit_string o.file s.at b off s.length
    | None -> ()
  in
  Iarray.iter put o.sections;
  let patch (at, p) = Bytes.blit_string p 0 b at (String.length p) in
  List.iter patch (Cubin.patches c ~base);
  Bytes.unsafe_to_string b

(* [t] without its zero bytes past its last nonzero byte and its patches'
   words. Bank 0's template runs up to the kernel's parameters, which ptxas
   places at byte 0x1a80 of a kernel of more than 4 KiB of them on sm_89,
   past a structure's bytes. Below them bank 0 holds the driver's words, at
   the offsets of any kernel's, then zeros: the cubin's own .nv.constant0
   section is zeros there, such a kernel reads only the driver's words below
   its parameters (test/nv/fixtures/launch.cu's wide reads c[0x0][0x28] and
   c[0x0][0x118]), and tinygrad's NV runtime (NVProgramData in
   tinygrad/runtime/ops_nv.py) builds bank 0 as zeros up to the offset
   EIATTR_PARAM_CBANK gives, with the same words set. The driver's words end
   within a structure, and dispatch writes the zeros after them. *)
let trimmed (t : Template.t) =
  let width bits =
    if bits <= 8 then 1 else if bits <= 16 then 2 else if bits <= 32 then 4 else 8
  in
  let last =
    List.fold_left
      (fun n (p : Template.patch) -> Int.max n (p.at + width p.bits))
      0 t.patches
  in
  let rec length n =
    if n > last && t.bytes.[n - 1] = '\000' then length (n - 1) else n
  in
  { t with bytes = String.sub t.bytes 0 (length (String.length t.bytes)) }

(* Descriptors and bank 0 for a launch whose local memory is already backed. *)
let descriptors l base code per_thread (k : Cubin.kernel) =
  let open Template in
  let bank (b : Cubin.bank) =
    (b.index, if b.index = 0 then Slot v_bank0 else Int (base + b.offset))
  in
  let d =
    {
      Abi.Qmd.program = Int code;
      local = Int per_thread;
      banks = List.map bank (Launch.banks l);
      grid = (Slot v_grid_x, Slot (v_grid_x + 1), Slot (v_grid_x + 2));
      block = (Slot v_block_x, Slot (v_block_x + 1), Slot (v_block_x + 2));
      shared = None;
      chain = None;
      releases = [];
    }
  in
  let shared = { d with Abi.Qmd.shared = Some (Slot v_shared) } in
  let chain (d : Abi.Qmd.t) = { d with chain = Some (Slot v_next) } in
  let qmds = List.map (Abi.Qmd.make l) [ d; chain d; shared; chain shared ] in
  let bank0_bytes =
    List.fold_left
      (fun n (b : Cubin.bank) -> if b.index = 0 then b.bytes else n)
      0 (Launch.banks l)
  in
  let strings =
    List.concat_map
      (fun (t : Template.t) -> [ t.bytes; patches t ])
      (qmds @ [ trimmed (Abi.Qmd.parameters l d) ])
  in
  let ints =
    [ k.params_offset; k.params_bytes; bank0_bytes ]
    @ List.map Abi.Qmd.max_size dims
    @ [ Launch.max_threads l; Launch.dynamic_shared l ]
  in
  new_launch (Array.of_list strings) (Array.of_list ints)

(* Opening *)

let in_rm what = Result.map_error (fun e -> strf "%s: %s" what (why e))

(* [r], a path's result, in the RM's terms. *)
let of_path r = Result.map_error (fun why -> Refused why) r

let host what (m : _ memory) =
  match m.host with
  | Some h -> Ok h
  | None -> Error (strf "the %s is not mapped for the host" what)

(* The RM's object of class [cls] under [parent], its [size] bytes of
   parameters set by [fill], given back by [taken] if a later step fails. *)
let new_object (p : _ path) ~taken ~parent cls what fill size =
  let q = params size in
  fill q;
  let* h = in_rm what (p.rm.alloc ~parent cls (Some q)) in
  taken (fun () -> ignore (freed p ~parent h : bool));
  Ok h

(* Channels *)

(* A channel's place in the memory of its set: the two rings, of [entries]
   entries each, then the two USERDs, a page each. *)
let ring_at entries q = if q = 0 then 0 else 8 * entries.(0)
let userd_at entries q = (8 * (entries.(0) + entries.(1))) + (q * page)
let block_bytes entries = userd_at entries 2

(* The memory of a set of channels whose rings hold [entries], from [alloc]:
   their rings and USERDs, and a page for each one's error notifier,
   zeroed. *)
let channel_memory alloc entries =
  let* block = alloc (block_bytes entries) "channels' rings" in
  let notifier q =
    let* m = alloc page (strf "error notifier of channel %d" q) in
    let* h = host "error notifier" m in
    zero h page;
    Ok m
  in
  let* compute = notifier 0 in
  let* copy = notifier 1 in
  Ok (block, [| compute; copy |])

(* Makes the set of channels whose rings hold [entries] over [block] and
   [notifiers], each RM object given back by [taken] if a later step fails:
   the set, and the ints rig_nv_stubs.c's [channels_of] reads. *)
let make_channels (type m) (p : m path) (module R : D.RELEASE) ~taken entries
    ((block : m memory), notifiers) =
  let new_object = new_object p ~taken in
  let* block_host = host "channels' rings" block in
  let* group =
    new_object ~parent:p.device D.kepler_channel_group_a "the channel group"
      (fun q ->
        set q R.Channel_group_alloc.engine_type D.nv2080_engine_type_graphics)
      R.Channel_group_alloc.sizeof
  in
  let* ctxshare =
    let module C = D.Ctxshare_alloc in
    new_object ~parent:group D.fermi_context_share_a "the context share"
      (fun q ->
        set q C.h_va_space p.vaspace;
        set q C.flags
          (bits D.nv_ctxshare_allocation_flags_subcontext
             D.nv_ctxshare_allocation_flags_subcontext_async))
      C.sizeof
  in
  let channel q engine =
    let module G = R.Gpfifo_alloc in
    let notifier : m memory = notifiers.(q) in
    let* notifier_host = host "error notifier" notifier in
    let* ch =
      new_object ~parent:group p.gpu.channel_class (strf "channel %d" q)
        (fun f ->
          set f G.gp_fifo_offset (block.address + ring_at entries q);
          set f G.gp_fifo_entries entries.(q);
          set f G.h_object_error notifier.handle;
          set f G.h_object_buffer block.handle;
          set f (elt G.h_userd_memory 0) block.handle;
          set f (elt G.userd_offset 0) (userd_at entries q);
          set f G.h_context_share ctxshare)
        G.sizeof
    in
    let* engine_object =
      in_rm "an engine" (p.rm.alloc ~parent:ch engine None)
    in
    let token = params D.Work_submit_token.sizeof in
    set token D.Work_submit_token.work_submit_token 0xffff_ffff;
    let* () =
      in_rm "the work submit token"
        (p.rm.control ch D.nvc36f_ctrl_cmd_gpfifo_get_work_submit_token
           (Some token))
    in
    let* () =
      in_rm (strf "registering channel %d" q) (of_path (p.register ch))
    in
    taken (fun () -> ignore (p.unregister ch));
    let ints =
      [
        block_host + ring_at entries q;
        entries.(q);
        block_host + userd_at entries q + fst D.Userd.gp_put;
        get token D.Work_submit_token.work_submit_token;
        notifier_host;
      ]
    in
    Ok (ch, engine_object, ints)
  in
  let* compute, compute_engine, compute_ints =
    channel 0 p.gpu.compute_class
  in
  let* copy, _, copy_ints = channel 1 p.gpu.copy_class in
  let* debugger =
    let module A = D.Nv83de_alloc in
    new_object ~parent:p.device D.gt200_debugger "the debugger"
      (fun q ->
        set q A.h_app_client p.rm.client;
        set q A.h_class3d_object compute_engine)
      A.sizeof
  in
  let schedule = params R.Group_schedule.sizeof in
  set schedule R.Group_schedule.b_enable 1;
  let* () =
    in_rm "scheduling the channels"
      (p.rm.control group D.nva06c_ctrl_cmd_gpfifo_schedule (Some schedule))
  in
  let set =
    {
      id = block.address;
      entries;
      block;
      notifiers;
      group;
      debugger;
      handles = [| compute; copy |];
      record = 0;
    }
  in
  Ok (set, Array.of_list (compute_ints @ copy_ints))

(* The device's objects and memory, each given back by [taken] if a later step
   fails. *)
let start (type m) ~word (p : m path) (module R : D.RELEASE) ~taken =
  let rm = p.rm in
  let alloc kind bytes what =
    match path_alloc p kind bytes with
    | None -> Error (strf "no memory for the %s" what)
    | Some m ->
        taken (fun () -> p.free m);
        Ok m
  in
  let* words = alloc Pinned page "join words" in
  let* words_host = host "join words" words in
  zero words_host words_bytes;
  (* The word rig gave, mapped from its chunk of the array, or, where the
     path maps no host memory or the word lies outside the array, a word of
     the device's own, started at the given word's value. *)
  let word = Nativeint.to_int word in
  let at = word land lnot (chunk - 1) in
  let start, bytes = array_of_words () in
  let mapped =
    match p.map_host with
    | None -> None
    | Some _ when word < start || word + 8 > start + bytes -> None
    | Some map -> (
        match placed Below_windows chunk (map at chunk) with
        | None -> None
        | Some m ->
            taken (fun () -> p.free m);
            Some m)
  in
  let word_host, word_mem =
    match mapped with
    | Some m ->
        ( word,
          { m with address = m.address + (word - at); host = Some word } )
    | None ->
        copy_word words_host word;
        (words_host, words)
  in
  let opened = [| entries; entries |] in
  let* memory = channel_memory (alloc Pinned) opened in
  let* segments = alloc Pinned (2 * segment_bytes) "segment rings" in
  let* segments_host = host "segment rings" segments in
  let* stamps = alloc Pinned stamps_bytes "stamp ring" in
  let* stamps_host = host "stamp ring" stamps in
  zero stamps_host stamps_bytes;
  (* The GPU reads a launch's descriptor and bank as the launch before it
     completes: from its own memory, through the BAR, where the BAR has room
     and a flush word orders the host's writes through it. *)
  let in_bar =
    match p.flush with
    | None -> None
    | Some _ -> path_alloc p Mapped launch_bytes
  in
  let* launches, launches_bar =
    match in_bar with
    | Some m ->
        taken (fun () -> p.free m);
        Ok (m, true)
    | None ->
        let* m = alloc Pinned launch_bytes "launch ring" in
        Ok (m, false)
  in
  let* launches_host = host "launch ring" launches in
  let self =
    create word_host word_mem.address words.address
      (fst D.Notification.info32)
      (fst D.Notification.status)
  in
  if self = 0 then Error "no host memory for the device's state"
  else
    let () = taken (fun () -> destroy self) in
    set_joins self (words_host + 8);
    let* () =
      if set_stamps self stamps_host stamps.address stamps_bytes slots then
        Ok ()
      else Error "no host memory for the stamps' and the marks' state"
    in
    let* channels, ints =
      make_channels p (module R) ~taken opened memory
    in
    set_channels self channels.id ints;
    for q = 0 to 1 do
      let at = q * segment_bytes in
      set_segments self q (segments_host + at) (segments.address + at)
        segment_bytes
    done;
    let boost = params D.Perf_boost.sizeof in
    set boost D.Perf_boost.duration D.nv2080_ctrl_perf_boost_duration_infinite;
    set boost D.Perf_boost.flags
      (bits D.nv2080_ctrl_perf_boost_flags_cuda
         D.nv2080_ctrl_perf_boost_flags_cuda_yes
      lor bits D.nv2080_ctrl_perf_boost_flags_cuda_priority
            D.nv2080_ctrl_perf_boost_flags_cuda_priority_high
      lor bits D.nv2080_ctrl_perf_boost_flags_cmd
            D.nv2080_ctrl_perf_boost_flags_cmd_boost_to_max);
    let* () =
      in_rm "raising the GPU's clocks"
        (rm.control p.subdevice D.nv2080_ctrl_cmd_perf_boost (Some boost))
    in
    set_doorbell self p.doorbell;
    set_launches self launches_host launches.address launch_bytes launches_bar;
    templates self p.gpu;
    Option.iter (set_flush self) p.flush;
    let source, bar =
      match p.clock with Source c -> (c, 0) | Registers a -> (0, a)
    in
    let* () =
      if set_clock self source bar then Ok ()
      else Error "the GPU's timer could not be read"
    in
    (* The private copy and fill kernels. On sm_80, sm_86, sm_87 and sm_89,
       a copy of one row of at least 256 KiB between two of the device's own
       [Device] allocations, and a fill of as many bytes of one, run as these
       kernels on the compute channel (rig_nv_ring.c's [kernel_copies] and
       [route]). Smaller copies and fills, boxes of more rows, copies and
       fills that touch host, [Pinned], [Mapped] or a peer's memory, and
       every copy and fill on other architectures, run on the copy channel.
       The kernels' image takes [Cubin.size] bytes of [Pinned] memory, and
       their descriptors lie in host memory until the channels end. *)
    let* copy_kernel =
      match Copy_kernel.find (arch_of p.gpu.sm_version) with
      | None -> Ok None
      | Some bin ->
          let* c = Cubin.of_string bin in
          let k = Option.get (Cubin.kernel c "copy_bytes") in
          let f = Option.get (Cubin.kernel c "fill_bytes") in
          let* code = alloc Pinned (Cubin.size c) "copy kernel" in
          let* code_host = host "copy kernel" code in
          Ok (Some (c, k, f, code, code_host))
    in
    (* rig's word goes back with the device's mapping of it; rig frees a
       word of the device's own, with its page, once no queue waits on it. *)
    let owned =
      [ segments; launches; stamps ]
      @ match mapped with Some m -> [ m; words ] | None -> []
    in
    let owned =
      match copy_kernel with
      | Some (_, _, _, code, _) -> code :: owned
      | None -> owned
    in
    let g = p.gpu in
    let rec d =
      {
        self;
        arch = arch_of g.sm_version;
        budget = p.budget;
        hang_ms = p.hang_ms;
        maps_host = Option.is_some p.map_host;
        key = p.key;
        index = p.index;
        reaches = p.reaches;
        capability =
          {
            Abi.Gpu.compute_class = g.compute_class;
            sass_version = sass_of g.sm_version;
            gpcs = g.gpcs;
            tpcs_per_gpc = g.tpcs_per_gpc;
            sms_per_tpc = g.sms_per_tpc;
            warps_per_sm = g.warps_per_sm;
            shared_window;
            local_window;
            local = (fun n -> local d n);
          };
        word = { dev = d; mem = word_mem; free = p.free; bytes = 8; flush = None };
        flush = p.flush;
        local_lock = ranked local_rank;
        live =
          Some
            {
              path = p;
              error_names = R.robust_channel_errors;
              owned;
              rings = [];
              release = (module R);
              sets = [ channels ];
              per_thread = 0;
              local_current = None;
              local_pending = None;
              local_retired = [];
            };
      }
    in
    let* () =
      match copy_kernel with
      | None -> Ok ()
      | Some (c, k, f, code, code_host) ->
          let* lk = Launch.make d.capability k in
          let* lf = Launch.make d.capability f in
          let bytes = image_bytes c ~base:code.address in
          (* These kernels have no stack, calls or local instructions. Their
             descriptors need none of the public launch's local reserve. *)
          let launch l (k : Cubin.kernel) =
            descriptors l code.address (code.address + k.code) 0 k
          in
          install_kernels self (launch lk k) (launch lf f) code_host bytes;
          Ok ()
    in
    Ok (T d)

let make ~word p =
  match D.release p.rm.release with
  | None ->
      Error
        (strf "the RM's release %d is none of %s" p.rm.release
           (String.concat ", " (List.map string_of_int D.releases)))
  | Some release -> (
      let undo = ref [] in
      let taken f = undo := f :: !undo in
      (* A failure giving something back does not hide the open's. *)
      let give_back () =
        List.iter (fun f -> try f () with Fault _ -> ()) !undo
      in
      match start ~word p release ~taken with
      | exception Fault why ->
          give_back ();
          Error why
      | Error _ as e ->
          give_back ();
          e
      | exception exn ->
          let bt = Printexc.get_raw_backtrace () in
          give_back ();
          Printexc.raise_with_backtrace exn bt
      | Ok _ as d -> d)

(* Memory *)

let region d (p : _ path) flush bytes m =
  R { dev = d; mem = m; free = p.free; bytes; flush }

(* Counts [delta] regions written through the BAR of the device [src]'s flush
   word in [d]'s flushes: [false] if [d] has no free entry for it, or [src]
   has stopped. *)
let count d src delta = count_flush d.self src delta

let alloc (T d) (memory : Rig_edge.memory) n =
  if n < 1 then invalid_argf "Rig_nv.alloc: %d bytes, expected at least 1" n;
  let l = live d in
  let p = l.path in
  (* Rings and channels the device retired go back first where no growth
     holds them. *)
  let channels_retire =
    Option.is_some p.rm.free && List.compare_length_with l.sets 1 > 0
  in
  if rings_retired d.self || channels_retire then
    ignore
      (try_protect d.local_lock (fun () ->
           retire_rings d l;
           retire_channels d l)
        : bool);
  match memory with
  | Device | Pinned ->
      Option.map (region d p None n) (path_alloc p memory n)
  | Mapped -> (
      match d.flush with
      | None -> None
      | Some _ -> (
          match path_alloc p Mapped n with
          | None -> None
          | Some m ->
              (* The device's own word has its entry from [make]. *)
              ignore (count d d.self 1 : bool);
              Some (region d p (Some d.self) n m)))

(* A region holds its device: rig frees it on that device. *)
let free (T _) (R r) =
  give r.free r.mem;
  Option.iter (fun src -> ignore (count r.dev src (-1) : bool)) r.flush

let address (R r) = r.mem.address

let locate (R r as region) =
  {
    Rig_edge.address = Some (address region);
    host = r.mem.host;
    handle = Nativeint.of_int (address region);
  }

let peer (T d) (T d') =
  match Type.Id.provably_equal d.key d'.key with
  | Some Type.Equal -> d.reaches d'.index
  | None -> false

(* [r] holds its device, the other one. A view of Mapped memory makes [d]'s
   submissions flush the host's writes through the BAR behind it, which [d]'s
   own doorbell does not order. *)
let map_peer (T d) (T _) (R r) =
  match Type.Id.provably_equal d.key r.dev.key with
  | None -> None
  | Some Type.Equal -> (
      let l = live d in
      let m = l.path.map_peer r.mem in
      match placed Outside_windows r.bytes m with
      | None -> None
      | Some m -> (
          match r.flush with
          | None -> Some (region d l.path None r.bytes m)
          | Some src when count d src 1 ->
              Some (region d l.path (Some src) r.bytes m)
          | Some _ ->
              give l.path.free m;
              None))

let map_host (T d) a n =
  if n < 1 then invalid_argf "Rig_nv.map_host: %d bytes, expected at least 1" n;
  let p = (live d).path in
  match p.map_host with
  | None -> None
  | Some map ->
      let m = placed Below_windows n (map a n) in
      Option.map (region d p None n) m

(* Rings *)

(* The rings [needed] numbers: each channel's segment ring, then the launch
   ring. *)
let launch_ring = 2

(* The bytes of a ring a growth makes for [want] bytes: the first power of two
   from a page that holds twice them, so that a submission of that size takes
   at most half of it, and growths come at doublings. *)
let grown_bytes want =
  let rec up n = if n >= 2 * want then n else up (2 * n) in
  up page

(* Makes [r] ring [i] of [n] bytes, and answers whether the host writes it
   through the BAR: as at open, the launch ring in the GPU's memory where the
   BAR has room and a flush word orders the host's writes, every other in
   [Pinned] memory. [r] stays [None] without memory. *)
let make_ring (p : _ path) i n r =
  if i = launch_ring && Option.is_some p.flush then
    r.ring <- path_alloc p Mapped n;
  Option.is_some r.ring
  || begin
    r.ring <- path_alloc p Pinned n;
    false
  end

(* Hands [m], ring [i] of [n] bytes, to the next room that needs it, and gives
   back the ring the hand did not keep. *)
let hand_ring d l i n bar (m : _ memory) =
  match hand d.self i (Option.get m.host) m.address n bar with
  | -1 -> raise (Fault "no host memory for a ring's record")
  | 0 -> ()
  | a -> drop_ring l a

(* Hands the next room that needs it a set of channels whose rings have
   [want] entries, 0 for a channel whose ring holds what the room asked, and
   no fewer than the newest set's, so that it holds what every earlier
   growth made a set for. Answers [0], or the bytes of memory the set takes
   if the path lacked them, or the RM lacked memory or another resource for
   it; the device keeps its channels then. The set enters [sets] before its
   hand, and an exception before it enters gives back what the growth
   made. *)
let grow_channels d l want =
  if Array.for_all (( = ) 0) want then 0
  else
    let p = l.path in
    let entries = Array.map2 max want (List.hd l.sets).entries in
    let undo = ref [] and lacked = ref false and recorded = ref false in
    let taken f = undo := f :: !undo in
    let give_back () =
      List.iter (fun f -> try f () with Fault _ -> ()) !undo
    in
    let alloc n what =
      match path_alloc p Pinned n with
      | Some m ->
          taken (fun () -> p.free m);
          Ok m
      | None ->
          lacked := true;
          Error (strf "no memory for the %s" what)
    in
    let lacking r =
      (match r with Error (Lacking _) -> lacked := true | Ok _ | Error _ -> ());
      r
    in
    let rm =
      {
        p.rm with
        alloc = (fun ~parent cls q -> lacking (p.rm.alloc ~parent cls q));
        control = (fun obj cmd q -> lacking (p.rm.control obj cmd q));
      }
    in
    let record = ref 0 in
    taken (fun () -> if !record <> 0 then free_channels !record);
    let made =
      try
        let* memory = channel_memory alloc entries in
        let* set, ints =
          make_channels { p with rm } l.release ~taken entries memory
        in
        record := channels_record set.id ints;
        if !record = 0 then Error "no host memory for the channels' record"
        else
          let set = { set with record = !record } in
          let sets = set :: l.sets in
          l.sets <- sets;
          recorded := true;
          Ok set
      with exn ->
        let bt = Printexc.get_raw_backtrace () in
        if not !recorded then give_back ();
        Printexc.raise_with_backtrace exn bt
    in
    match made with
    | Error _ when !lacked ->
        give_back ();
        block_bytes entries + (2 * page)
    | Error why ->
        give_back ();
        raise (Fault why)
    | Ok set ->
        hand_channels d.self set.record;
        0

(* The rings and channels the submission whose room answered RIG_GROW on
   this thread needs, each listed before its allocation. A growth short of
   memory for one still hands those it made, and the submission's room asks
   for the rest anew after its round of reclamation. *)
let grow (T d) =
  protect d.local_lock @@ fun () ->
  let l = live d in
  retire_rings d l;
  retire_channels d l;
  let needs = needed d.self in
  let w = Array.sub needs 0 (launch_ring + 1) in
  let sizes = Array.map grown_bytes w in
  let rings = Array.map (fun _ -> { ring = None }) w in
  let bars = Array.make (Array.length w) false in
  l.rings <- Array.fold_right List.cons rings l.rings;
  let short = ref 0 in
  for i = 0 to Array.length w - 1 do
    if w.(i) > 0 then begin
      bars.(i) <- make_ring l.path i sizes.(i) rings.(i);
      if Option.is_none rings.(i).ring && !short = 0 then short := sizes.(i)
    end
  done;
  l.rings <- List.filter (fun r -> Option.is_some r.ring) l.rings;
  Array.iteri
    (fun i r -> Option.iter (hand_ring d l i sizes.(i) bars.(i)) r.ring)
    rings;
  let lacked = grow_channels d l (Array.sub needs (launch_ring + 1) 2) in
  if !short = 0 then short := lacked;
  if !short = 0 then Rig_edge.Grown else Short !short

(* Images *)

module Names = Map.Make (String)

(* An image is a cubin laid over a region of its device: the address of the
   region's first byte. It keeps the entries made of its functions, and their
   C launches, until it is unloaded. Both change by stores of values made
   first, with nothing between the stores that allocates or polls, so an
   exception leaves them whole and in step for the unload. *)
type image = {
  base : int;
  cubin : Cubin.t;
  gpu : Abi.Gpu.t;
  grow_local : int -> served; (* its device's [grow_local] *)
  lock : lock;
  mutable entries : Rig_edge.entry Names.t;
  mutable launches : nativeint list; (* the entries' *)
}

(* The cubin [c] laid over the region [r] of [d]. A cubin may load where another
   one's code was, so the compute channel owes an instruction cache
   invalidation. *)
let lay d c r =
  let base = address r in
  owe_invalidate d.self;
  let i =
    {
      base;
      cubin = c;
      gpu = d.capability;
      grow_local = grow_local d;
      lock = ranked image_rank;
      entries = Names.empty;
      launches = [];
    }
  in
  (i, image_bytes c ~base)

let image (T d) bin =
  let* c = Cubin.of_string bin in
  Ok (Rig_edge.Place (Cubin.size c, lay d c))

(* The entry of [c]'s function [k], named [name], once the local memory serves
   it; a function the GPU has no memory for makes nothing. The launch is
   [c]'s from [new_launch] on: an exception before [c] holds it, such as an
   asynchronous one where the entry allocates, frees it. *)
let set_up c name (k : Cubin.kernel) =
  let l =
    match Launch.make c.gpu k with
    | Ok l -> l
    | Error why -> invalid_arg ("Rig_nv.entry: " ^ why)
  in
  let bytes = Launch.local_bytes l in
  match c.grow_local bytes with
  | Short n -> Rig_edge.No_memory n
  | Stopped -> raise (Fault "the device is stopped")
  | Served -> (
      let code = c.base + k.code in
      let local = Abi.Local_memory.make c.gpu bytes in
      let launch = descriptors l c.base code local.per_thread k in
      match
        let e =
          {
            Rig_edge.code;
            launch;
            params = Some k.params_bytes;
            threads = Some (Launch.max_threads l);
          }
        in
        let entries = Names.add name e c.entries
        and launches = launch :: c.launches in
        c.entries <- entries;
        c.launches <- launches;
        e
      with
      | e -> Entry e
      | exception x ->
          free_launch launch;
          Printexc.raise_with_backtrace x (Printexc.get_raw_backtrace ()))

let entry c name =
  protect c.lock @@ fun () ->
  match Names.find_opt name c.entries with
  | Some e -> Rig_edge.Entry e
  | None -> (
      match Cubin.kernel c.cubin name with
      | None -> Absent
      | Some k -> set_up c name k)

(* An image holds of its device its code region, which rig frees, and its
   functions' launches. rig unloads an image once, so past its take the
   unload allocates nothing and polls nowhere: an exception cannot end it
   with launches held. One that ends the take ends it before it changed
   anything. *)
let unload (T _) c =
  take_lock c.lock;
  let launches = c.launches in
  c.entries <- Names.empty;
  c.launches <- [];
  free_launches launches;
  give_lock c.lock

(* Timeline *)

let signaled (T d) = read_word d.self

let fault_name table v =
  match List.assoc_opt v table with Some n -> n | None -> strf "0x%x" v

(* The faults of [d]'s work, one per line: each different error the RM wrote
   into the channels' notifiers when it stopped them (it writes an error of
   the channel group into each), then the faults the multiprocessors or the
   MMU reported to it. *)
let report d l =
  let module S = D.Sm_error_states in
  let module E = D.Sm_error_state in
  let module M = D.Mmu_fault_info in
  let module F = D.Mmu_fault_entry in
  let error q =
    let x = notification d.self q in
    if x = 0 then None
    else
      let code = x lsr 16 and status = x land 0xffff in
      Some
        (strf "channel error %d (%s), status 0x%x" code
           (fault_name l.error_names code)
           status)
  in
  let channels =
    match List.filter_map error [ 0; 1 ] with
    | [ a; b ] when a = b -> [ a ]
    | errors -> errors
  in
  let control (c : _ channels) cmd p =
    Result.map_error why (l.path.rm.control c.debugger cmd (Some p))
  in
  let sm = params S.sizeof and mmu = params M.sizeof in
  set sm S.num_s_ms_to_read sm_errors;
  let sm_fault i =
    let f x = elt_field S.sm_error_state_array i x in
    let global = get sm (f E.hww_global_esr)
    and warp = get sm (f E.hww_warp_esr) in
    if global = 0 && warp = 0 then None
    else
      Some
        (strf "SM %d fault: esr=0x%x warp_esr=0x%x warp_pc=0x%x" i global warp
           (get sm (f E.hww_warp_esr_pc64)))
  in
  let mmu_fault i =
    let f x = elt_field M.mmu_fault_info_list i x in
    strf "MMU fault: 0x%X | %s | %s"
      (get mmu (f F.fault_address))
      (fault_name D.fault_types (get mmu (f F.fault_type)))
      (fault_name D.access_types (get mmu (f F.access_type)))
  in
  (* The installed set's debugger, under the lock a growth gives sets back
     under, so that the set stays while its debugger reads. *)
  let faults =
    protect d.local_lock @@ fun () ->
    let installed, _ = channels_state d.self in
    let c = List.find (fun s -> s.id = installed) l.sets in
    set sm S.h_target_channel c.handles.(0);
    match control c D.nv83de_ctrl_cmd_debug_read_all_sm_error_states sm with
    | Error e -> [ strf "reading the multiprocessors' errors: %s" e ]
    | Ok () when get sm S.mmu_fault_valid = 0 ->
        List.filter_map sm_fault (List.init sm_errors Fun.id)
    | Ok () -> (
        match control c D.nv83de_ctrl_cmd_debug_read_mmu_fault_info mmu with
        | Error e -> [ strf "reading the MMU's faults: %s" e ]
        | Ok () -> List.init (get mmu M.count) mmu_fault)
  in
  channels @ faults

(* Whether a channel's notifier holds an error. *)
let notified d = notification d.self 0 <> 0 || notification d.self 1 <> 0

let check_faults d =
  let l = live d in
  l.path.check ();
  match report d l with
  | [] -> ()
  | report -> raise (Fault (String.concat "\n" report))

(* The host's clock, [rig_now_ns] of [rig_edge.h]. *)
external now : unit -> (int[@untagged])
  = "caml_rig_edge_now_byte" "caml_rig_edge_now"
[@@noalloc]

(* A sleep reads the word and the notifiers each [nap_s]: a fault is found
   within a nap of the RM's report, and the wake latency falls on waits rig has
   already spun on, which are long. *)
let nap_s = 0.001

let sleep (T d) ~seen ~still_ms =
  if read_word d.self = seen then begin
    check_faults d;
    let until = now () + (still_ms * 1_000_000) in
    let more = ref true in
    while !more && read_word d.self = seen && not (notified d) do
      if now () >= until then more := false
      else Unix.sleepf nap_s
    done;
    if notified d then check_faults d
  end

(* Loss *)

(* Stops [d], whose live state was [l]. *)
let halt d l =
  let ok f =
    match f () with Ok () -> true | Error _ | (exception Fault _) -> false
  in
  let unregister ch = ok (fun () -> l.path.unregister ch) in
  let unregistered =
    List.for_all Fun.id
      (List.concat_map
         (fun s -> Array.to_list (Array.map unregister s.handles))
         l.sets)
  in
  let device = l.path.device in
  (* The first set the RM does not free ends the frees: a GPU that cannot
     free one frees none of the others either. *)
  let freed =
    List.for_all
      (fun s ->
        freed l.path ~parent:device s.debugger
        && freed l.path ~parent:device s.group)
      l.sets
  in
  (* No work on another device uses this GPU's memory once it is lost
     ([Rig]'s loss section), so their submissions stop reading its flush word
     before the path may unmap it. *)
  drop_flush d.self;
  let path = try l.path.stop () with Fault _ -> `Unknown in
  let stopped = (unregistered && freed) || path = `Stopped in
  if stopped then begin
    raise_word d.self;
    (* The work no longer runs: what it used goes back, and a failure to give
       some back leaves the device stopped. *)
    List.iter (give l.path.free) l.owned;
    List.iter
      (fun s ->
        give l.path.free s.block;
        Array.iter (give l.path.free) s.notifiers)
      l.sets;
    List.iter (fun r -> Option.iter (give l.path.free) r.ring) l.rings;
    Option.iter (give l.path.free) l.local_current;
    Option.iter (give l.path.free) l.local_pending;
    List.iter (fun (r : _ retired) -> give l.path.free r.mem) l.local_retired
  end
    (* The RM stopped both channels on a fault: nothing runs, though their
       objects stay. Otherwise their own releases bring the word up as their
       work ends. A word raised here could be lowered by a late release: a
       channel the RM did not stop would find its waits on the word met and
       release its own values. *)
  else if notification d.self 0 <> 0 && notification d.self 1 <> 0 then
    raise_word d.self;
  end_state d.self stopped;
  List.iter (fun s -> if s.record <> 0 then free_channels s.record) l.sets

let stop (T d) ~fault:_ =
  let l =
    protect d.local_lock @@ fun () ->
    let l = d.live in
    d.live <- None;
    l
  in
  Option.iter (halt d) l
