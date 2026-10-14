(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Any domain may call any function, as the interface says. The C state is
   written by [make], then by [room] and [submit], which rig calls one at a
   time for each device; [local] holds the device's lock while it grows the
   local memory, and [stop] while it drops the device's live state.

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
exception Refused of string

(* Paths *)

type params =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

type rm = {
  release : int;
  client : int;
  alloc : parent:int -> int -> params option -> (int, string) result;
  control : int -> int -> params option -> (unit, string) result;
  free : parent:int -> int -> (unit, string) result;
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

(* The memory [m] of [n] bytes a path answered, below [address_limit] as
   every path's interface promises. *)
let address_limit = 1 lsl 40

let below n = function
  | Some (m : _ memory) as r ->
      assert (m.address + n <= address_limit);
      r
  | None -> None

let path_alloc (p : _ path) kind n = below n (p.alloc kind n)

(* Constants *)

let page = 4096

(* The device's own words: a timeline word, used where the path maps no host
   memory, then the channels' join words, as rig_nv_stubs.h's JOIN_GPU lays
   them out. *)
let words_bytes = 8 * 3

(* The bytes, from a 64 KiB boundary of rig's array of words, that the device
   maps to reach its word there: whole pages on every host. *)
let chunk = 1 lsl 16

(* Each channel's ring holds [entries] entries, and its segment ring
   [segment_bytes] bytes, both powers of two: a submission takes at least one
   entry of each channel it uses, so thousands may be in flight. *)
let entries = 16384
let segment_bytes = 1 lsl 20

(* The stamp ring's slots, one per value modulo their number, rig_nv_stubs.h's
   Times: more than the values the two channels' rings may hold unreached. A
   slot is three records of 16 bytes. *)
let slots = 2 * entries
let stamps_bytes = slots * 48

(* The compute channel's launch ring holds [launch_bytes] bytes of launches'
   descriptors and banks, a power of two: a launch takes about 1 KiB. *)
let launch_bytes = 1 lsl 20

(* The addresses at which kernels see their shared and local memory, above 2^40,
   where no memory the device maps lies. *)
let shared_window = 0x7294_0000_0000
let local_window = 0x7293_0000_0000

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

(* rig_nv_stubs.h's most bytes of a launch's structure. *)
let structure_bytes = 1024

(* The most bytes of a launch's constant bank 0: rig_nv_stubs.h's
   BANK_MAX, which the device's most parts assumes. *)
external bank_max : unit -> int = "caml_rig_nv_bank_max"

let bank_max = bank_max ()

(* A pending local memory is one word: its address, below 2^40 and 4 KiB
   aligned, and its bytes per cluster in units of 32 KiB above them. The
   submission of value [v] that takes it leaves [v lsl 1 lor 1] in its
   place. *)
let local_address_bits = 40
let local_unit_shift = 15

(* The multiprocessors whose errors the RM reports, at most. *)
let sm_errors =
  let _, _, n = D.Sm_error_states.sm_error_state_array in
  n

(* The C state *)

external create : int -> int -> int -> int -> int -> int = "caml_rig_nv_create"
[@@noalloc]

external destroy : int -> unit = "caml_rig_nv_destroy" [@@noalloc]
external most_parts : int -> int = "caml_rig_nv_most_parts"

external set_channel : int -> int -> int array -> bool = "caml_rig_nv_channel"
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

external install_copy : int -> nativeint -> int -> string -> unit
  = "caml_rig_nv_copy_kernel"
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

(* A field is (byte offset, bytes); the generated tables hold widths of 1, 2, 4
   and 8 bytes only. *)
let get p (at, n) =
  match n with
  | 1 -> Char.code (Bigarray.Array1.get p at)
  | 2 -> get16 p at
  | 4 -> Int32.to_int (get32 p at) land 0xffff_ffff
  | _ -> Int64.to_int (get64 p at)

let set p (at, n) v =
  match n with
  | 1 -> Bigarray.Array1.set p at (Char.unsafe_chr (v land 0xff))
  | 2 -> set16 p at (v land 0xffff)
  | 4 -> set32 p at (Int32.of_int v)
  | _ -> set64 p at (Int64.of_int v)

(* Element [i] of the array field (offset, bytes of an element, elements), and
   the field [f] of that element. *)
let elt (at, n, _) i = (at + (i * n), n)
let elt_field (at, n, _) i (f, fn) = (at + (i * n) + f, fn)
let bits (lo, n) v = (v land ((1 lsl n) - 1)) lsl lo

(* Devices *)

(* A local memory the GPU's work no longer uses once the device reached
   [until], the value of the submission that took its successor. *)
type 'm retired = { mem : 'm memory; mutable until : int }

(* What only a running device uses. The handle outlives the device in its
   properties, so stop drops it. *)
type 'm live = {
  path : 'm path;
  error_names : (int * string) list;
  owned : 'm memory list;
  group : int;
  debugger : int;
  channels : int list;
  compute_channel : int;
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
  flush_lock : Mutex.t; (* serialises count_flush *)
  word : 'm reg; (* kept, with the C state, for the process *)
  local_lock : Mutex.t; (* held by [grow], and by stop over [live] *)
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
      (* the flush word of the GPU whose BAR the host writes the memory
         through, for Mapped memory and views of it *)
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

(* Its channels wait on any 64-bit word the device maps, whoever writes it, up
   to rig_nv_ring.c's MAX_WAITS. *)
let waits =
  { Rig_edge.stores = true; hosts = true; objects = false; most = 256 }

let capability (T d) = d.capability

let facts (T d) =
  {
    Rig_edge.arch = d.arch;
    budget = d.budget;
    queues;
    copy_engine = Some 1;
    completion = Store;
    waits;
    most_parts = most_parts d.self;
    may_block = false;
    hang_ms = d.hang_ms;
    maps_host = d.maps_host;
    host_addresses = false;
    capability = Abi.Gpu.declare d.capability;
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

type grown = Served | Stopped | Short of int

(* Makes the local memory serve kernels whose threads need [n] bytes each:
   [Short bytes] if the GPU has no memory for the [bytes] that takes. Raises
   [Fault] if the path does. *)
let grow d n =
  Mutex.protect d.local_lock @@ fun () ->
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
            publish d l m
              (m.address
              lor ((need.per_tpc lsr local_unit_shift) lsl local_address_bits));
            l.per_thread <- need.per_thread;
            Served)

let local d n =
  match grow d n with
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
     a size. *)
  let a = Template.Slot 0 and b = Template.Slot 1 and n = Template.Slot 2 in
  template self t_acquire (Method.acquire a b);
  template self t_release (Method.release a b);
  template self t_copy_release (Method.copy_release a b);
  template self t_copy (Method.copy ~dst:a ~src:b n);
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
      (qmds @ [ Abi.Qmd.parameters l d ])
  in
  if k.params_offset > structure_bytes then
    invalid_arg "Rig_nv.entry: a launch places parameters past 1 KiB";
  if bank0_bytes > bank_max then
    invalid_argf "Rig_nv.entry: a kernel's constant bank 0 takes %d bytes, \
                  more than %d" bank0_bytes bank_max;
  let ints =
    [ k.params_offset; bank0_bytes ]
    @ List.map Abi.Qmd.max_size dims
    @ [ Launch.max_threads l; Launch.dynamic_shared l ]
  in
  new_launch (Array.of_list strings) (Array.of_list ints)

(* Opening *)

(* A channel's place in the block of channel memory: the two rings, the two
   USERDs a page each, then the two segment rings. *)
let ring_bytes = 8 * entries
let ring_at q = q * ring_bytes
let userd_at q = (2 * ring_bytes) + (q * page)
let segments_at q = (2 * ring_bytes) + (2 * page) + (q * segment_bytes)
let block_bytes = segments_at 2
let in_rm what = Result.map_error (fun e -> strf "%s: %s" what e)

let host what (m : _ memory) =
  match m.host with
  | Some h -> Ok h
  | None -> Error (strf "the %s is not mapped for the host" what)

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
  let new_object ~parent cls what fill size =
    let q = params size in
    fill q;
    let* h = in_rm what (rm.alloc ~parent cls (Some q)) in
    taken (fun () -> ignore (rm.free ~parent h));
    Ok h
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
        match below chunk (map at chunk) with
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
  let* block = alloc Pinned block_bytes "channels' rings" in
  let* block_host = host "channels' rings" block in
  let notifier q =
    let* m = alloc Pinned page (strf "error notifier of channel %d" q) in
    let* h = host "error notifier" m in
    zero h page;
    Ok (m, h)
  in
  let* stamps = alloc Pinned stamps_bytes "stamp ring" in
  let* stamps_host = host "stamp ring" stamps in
  zero stamps_host stamps_bytes;
  let* compute_notifier = notifier 0 in
  let* copy_notifier = notifier 1 in
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
  let self =
    create word_host word_mem.address words.address
      (fst D.Notification.info32)
      (fst D.Notification.status)
  in
  if self = 0 then Error "no host memory for the device's state"
  else
    let () = taken (fun () -> destroy self) in
    let channel q engine (notifier, notifier_host) =
      let module G = R.Gpfifo_alloc in
      let* ch =
        new_object ~parent:group p.gpu.channel_class (strf "channel %d" q)
          (fun f ->
            set f G.gp_fifo_offset (block.address + ring_at q);
            set f G.gp_fifo_entries entries;
            set f G.h_object_error notifier.handle;
            set f G.h_object_buffer block.handle;
            set f (elt G.h_userd_memory 0) block.handle;
            set f (elt G.userd_offset 0) (userd_at q);
            set f G.h_context_share ctxshare)
          G.sizeof
      in
      let* engine_object =
        in_rm "an engine" (rm.alloc ~parent:ch engine None)
      in
      let token = params D.Work_submit_token.sizeof in
      set token D.Work_submit_token.work_submit_token 0xffff_ffff;
      let* () =
        in_rm "the work submit token"
          (rm.control ch D.nvc36f_ctrl_cmd_gpfifo_get_work_submit_token
             (Some token))
      in
      let* () = in_rm (strf "registering channel %d" q) (p.register ch) in
      taken (fun () -> ignore (p.unregister ch));
      let ints =
        [|
          block_host + ring_at q;
          entries;
          block_host + userd_at q + fst D.Userd.gp_put;
          get token D.Work_submit_token.work_submit_token;
          block_host + segments_at q;
          block.address + segments_at q;
          segment_bytes;
          notifier_host;
        |]
      in
      if not (set_channel self q ints) then
        Error "no host memory for the channels' state"
      else Ok (ch, engine_object)
    in
    let* compute, compute_engine =
      channel 0 p.gpu.compute_class compute_notifier
    in
    let* copy, _ = channel 1 p.gpu.copy_class copy_notifier in
    let* debugger =
      let module A = D.Nv83de_alloc in
      new_object ~parent:p.device D.gt200_debugger "the debugger"
        (fun q ->
          set q A.h_app_client rm.client;
          set q A.h_class3d_object compute_engine)
        A.sizeof
    in
    let schedule = params R.Group_schedule.sizeof in
    set schedule R.Group_schedule.b_enable 1;
    let* () =
      in_rm "scheduling the channels"
        (rm.control group D.nva06c_ctrl_cmd_gpfifo_schedule (Some schedule))
    in
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
      if not (set_stamps self stamps_host stamps.address stamps_bytes slots)
      then Error "no host memory for the stamp ring's state"
      else if not (set_clock self source bar) then
        Error "the GPU's timer could not be read"
      else Ok ()
    in
    (* The private copy kernel. On sm_80, sm_86, sm_87 and sm_89, a copy of
       at least 256 KiB between two of the device's own [Device] allocations
       runs as this kernel on the compute channel (rig_nv_ring.c's
       [kernel_copies] and [route]). Smaller copies, copies that touch host,
       [Pinned], [Mapped] or a peer's memory, and every copy on other
       architectures, run on the copy channel. The kernel's image takes
       [Cubin.size] bytes of [Pinned] memory, and its descriptors lie in host
       memory until the channels end. *)
    let* copy_kernel =
      match Copy_kernel.find (arch_of p.gpu.sm_version) with
      | None -> Ok None
      | Some bin ->
          let* c = Cubin.of_string bin in
          let k = Option.get (Cubin.kernel c "copy_bytes") in
          let* code = alloc Pinned (Cubin.size c) "copy kernel" in
          let* code_host = host "copy kernel" code in
          Ok (Some (c, k, code, code_host))
    in
    (* rig's word goes back with the device's mapping of it; rig frees a
       word of the device's own, with its page, once no queue waits on it. *)
    let owned =
      [ block; launches; stamps; fst compute_notifier; fst copy_notifier ]
      @ match mapped with Some m -> [ m; words ] | None -> []
    in
    let owned =
      match copy_kernel with
      | Some (_, _, code, _) -> code :: owned
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
        flush_lock = Mutex.create ();
        local_lock = Mutex.create ();
        live =
          Some
            {
              path = p;
              error_names = R.robust_channel_errors;
              owned;
              group;
              debugger;
              channels = [ compute; copy ];
              compute_channel = compute;
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
      | Some (c, k, code, code_host) ->
          let* l = Launch.make d.capability k in
          let bytes = image_bytes c ~base:code.address in
          (* This kernel has no stack, calls or local instructions. Its
             descriptors need none of the public launch's local reserve. *)
          let launch = descriptors l code.address (code.address + k.code) 0 k in
          install_copy self launch code_host bytes;
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

(* Counts [delta] regions written through the BAR of the flush word [w] in
   [d]'s flushes: [false] if [d] has no free entry for [w]. *)
let count d w delta =
  Mutex.protect d.flush_lock (fun () -> count_flush d.self w delta)

let alloc (T d) (memory : Rig_edge.memory) n =
  if n < 1 then invalid_argf "Rig_nv.alloc: %d bytes, expected at least 1" n;
  let l = live d in
  let p = l.path in
  match memory with
  | Device | Pinned ->
      Option.map (region d p None n) (path_alloc p memory n)
  | Mapped -> (
      match d.flush with
      | None -> None
      | Some w -> (
          match path_alloc p Mapped n with
          | None -> None
          | Some m ->
              (* The device's own word has its entry from [make]. *)
              ignore (count d w 1 : bool);
              Some (region d p (Some w) n m)))

(* A region holds its device: rig frees it on that device. *)
let free (T _) (R r) =
  give r.free r.mem;
  Option.iter (fun w -> ignore (count r.dev w (-1) : bool)) r.flush

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
      match below r.bytes m with
      | None -> None
      | Some m -> (
          match r.flush with
          | None -> Some (region d l.path None r.bytes m)
          | Some w when count d w 1 -> Some (region d l.path (Some w) r.bytes m)
          | Some _ ->
              give l.path.free m;
              None))

let map_host (T d) a n =
  if n < 1 then invalid_argf "Rig_nv.map_host: %d bytes, expected at least 1" n;
  let p = (live d).path in
  match p.map_host with
  | None -> None
  | Some map ->
      let m = below n (map a n) in
      Option.map (region d p None n) m

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
  grow : int -> grown; (* its device's [grow] *)
  lock : Mutex.t;
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
      grow = grow d;
      lock = Mutex.create ();
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
  match c.grow bytes with
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
  Mutex.protect c.lock @@ fun () ->
  match Names.find_opt name c.entries with
  | Some e -> Rig_edge.Entry e
  | None -> (
      match Cubin.kernel c.cubin name with
      | None -> Absent
      | Some k -> set_up c name k)

(* An image holds of its device its code region, which rig frees, and its
   functions' launches. rig unloads an image once, so the unload allocates
   nothing and polls nowhere: an exception cannot end it with launches
   held. *)
let unload (T _) c =
  Mutex.lock c.lock;
  let launches = c.launches in
  c.entries <- Names.empty;
  c.launches <- [];
  free_launches launches;
  Mutex.unlock c.lock

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
  let control cmd p = l.path.rm.control l.debugger cmd (Some p) in
  let sm = params S.sizeof and mmu = params M.sizeof in
  set sm S.h_target_channel l.compute_channel;
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
  let faults =
    match control D.nv83de_ctrl_cmd_debug_read_all_sm_error_states sm with
    | Error e -> [ strf "reading the multiprocessors' errors: %s" e ]
    | Ok () when get sm S.mmu_fault_valid = 0 ->
        List.filter_map sm_fault (List.init sm_errors Fun.id)
    | Ok () -> (
        match control D.nv83de_ctrl_cmd_debug_read_mmu_fault_info mmu with
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
  let rm = l.path.rm in
  let unregistered =
    List.for_all Fun.id
      (List.map (fun ch -> ok (fun () -> l.path.unregister ch)) l.channels)
  in
  let freed =
    ok (fun () -> rm.free ~parent:l.path.device l.debugger)
    && ok (fun () -> rm.free ~parent:l.path.device l.group)
  in
  let path = try l.path.stop () with Fault _ -> `Unknown in
  let stopped = (unregistered && freed) || path = `Stopped in
  if stopped then begin
    raise_word d.self;
    (* The work no longer runs: what it used goes back, and a failure to give
       some back leaves the device stopped. *)
    List.iter (give l.path.free) l.owned;
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
  Mutex.protect d.flush_lock (fun () -> end_state d.self stopped)

let stop (T d) ~fault:_ =
  let l =
    Mutex.protect d.local_lock @@ fun () ->
    let l = d.live in
    d.live <- None;
    l
  in
  Option.iter (halt d) l
