(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The field order of [region], [base], [claim] and [Buffer.t] up to the fields
   nx_device.h reads is its C ABI. *)
type region = {
  host : nativeint option;
  address : nativeint;
  handle : nativeint;
  nbytes : int;
}

type signal = { signaled : unit -> int; wait : int -> ms:int -> bool }
type allocator = { alloc : int -> region option; free : region -> unit }

type mapping =
  | Identity
  | Pages of {
      map : nativeint -> int -> (region, string) result;
      unmap : region -> unit;
    }

type copy = dst:nativeint -> src:nativeint -> int -> signal:int -> unit

type io = {
  read : src:nativeint -> dst:nativeint -> int -> unit;
  write : dst:nativeint -> src:nativeint -> int -> unit;
  copy : dst:nativeint -> src:nativeint -> int -> unit;
}

type dma = { bus : string; pages : (int * int) list }
type clock = Host_clock | Device_clock of { hz : int }
type sleep = timeline:region -> still:int -> int -> unit

type completion =
  | Poll
  | Sleep of sleep
  | Signal of (timeline:region -> signal)

type host_programs = {
  load :
    binary:string -> entry:string -> (nativeint * (unit -> unit), string) result;
  call : nativeint -> (nativeint * int) array -> int array -> unit;
}

(* A binary as its driver loaded it. *)
type image = {
  code : region option;
  entry : string -> (nativeint, string) result;
  unload : unit -> unit;
}

(* Which of a device's memories [Buffer.create] allocates. *)
type memory = Device | Pinned | Mapped

(* What a device's work does with a buffer it reaches. *)
type access = Read | Read_write

(* The bytes a device allocated: an atomic count, or the host's, which the
   finalisers of its buffers' tokens return. *)
type allocated = Count of int Atomic.t | Heap_bytes

type t = {
  id : int;
  name : string;
  arch : string;
  machine : t option; (* the host of the device's machine, [None] for a host *)
  kind : kind;
  lock : Mutex.t;
  peer : (t -> region -> (region * (unit -> unit), string) result) option;
  reaches_peer : t -> bool; (* whether its driver maps a device's memory *)
  load : (binary:string -> (image, string) result) option;
  call : (nativeint -> (nativeint * int) array -> int array -> unit) option;
      (* how a host calls its programs *)
  link : (src:buffer -> dst:buffer -> link option) option;
  dma : (region -> (dma, string) result) option;
  signal : signal option;
  sleep : (still:int -> int -> unit) option;
  synchronized : unit -> unit;
  report : (unit -> event list) option;
      (* the counters of its work done since the last report *)
  room : unit -> bool;
      (* whether each of its queues has room for a submission *)
  finalize : failed:bool -> unit;
  clock : clock;
  resolve : nativeint -> unit;
  timeline : region; (* the signal word, then two 16-byte slots of timestamps *)
  timeline_keep : keep;
  last : int Atomic.t; (* the submitted value, which only this module writes *)
  settled : int Atomic.t; (* the latest value a wait saw signaled *)
  mutable slots : buffer option; (* a host's staging memory, once made *)
  mutable staging : buffer option;
      (* the device's borrow of its host's staging memory, once made, kept for
         the device's life *)
  released : nativeint; (* its release list (see the stubs) *)
  failed : string option Atomic.t;
      (* why the device was lost, which every operation raises *)
  cache : (int * memory, reusable list) Hashtbl.t;
      (* by size, and the memory it is *)
  mutable retiring : retiring list;
      (* released memory that waits for other devices' work *)
  pending : (int, t * int) Hashtbl.t;
      (* the devices whose work touched this one's memory, and the value that
         work signals *)
  images : (int, loaded Weak.t) Hashtbl.t;
      (* the loaded binaries, by length, while they are reachable *)
  mutable indexed : int; (* how many [images] held when last swept *)
  spans : pending list Atomic.t;
      (* the spans recorded on the device whose stamps are still to read, latest
         first *)
  mutable held : keep list; (* retained memory, and what it keeps *)
  mutable bytes_in : int;
  mutable bytes_out : int;
}

(* Memory in a device's cache: its region, the latest value of the device's work
   that touched it, which a reuse of it waits for, and the device's last value
   when it was released, which its free waits for (see [release]). *)
and reusable = { region : region; touched : int; released : int }

(* What a device is, with the memories it allocates. *)
and kind =
  | Machine of { remote : (string * io) option; pool : pool }
    (* a machine's host: this machine's, whose memory is the heap, or another
       machine's, at an address and reached through [io] *)
  | Disk
  | Shared of { pool : pool; mapping : mapping option }
  | Local of {
      own : pool;
      mapped : pool option; (* a window onto [own] that the host addresses *)
      pinned : pool; (* the host's memory, which counts in no budget *)
      mapping : mapping;
      queue : queue;
    }

(* One memory of a device: its allocator, [None] for the heap, the most bytes it
   may hold, and the bytes it holds in live buffers and loaded code, in its
   cache, and retained. *)
and pool = {
  allocator : allocator option;
  mutable ceiling : int;
  in_use : allocated;
  mutable cached : int;
  mutable retained : int;
}

and queue = {
  copy : copy;
  transfer : t -> copy option;
  stamp : slot:nativeint -> signal:int -> unit;
  clock : clock;
}

and link = { through : t list; move : src:buffer -> dst:buffer -> unit }

and buffer = {
  base : base;
  offset : int; (* bytes into [base.memory] *)
  dtype : Nx_dtype.Scalar.t;
  length : int;
  generation : generation; (* its memory's when the buffer was made *)
}

(* A base has no mutable field: the state that changes is in records it shares
   with its copies, which release the memory (see [owned]). *)
and base = {
  owner : t;
  memory : region;
  bytes : int; (* of owned memory, 0 when borrowed *)
  kind : memory; (* the owner's memory it is, [Device] when borrowed *)
  borrowed : bool;
  keep : keep;
  source : (base * mapped) option;
      (* for a borrow, the memory it maps, and the mapping *)
  links : links Atomic.t;
  file : file option; (* on the disk, the file *)
  claim : claim;
      (* shared by every base over the memory, its borrows' and copies'
         included *)
}

(* The host memory a staged buffer stands in for on its device, and what the
   device's work does with it: a submission copies [original] in before the
   work, and back once the work is done when it writes it. [busy] is held by the
   submission that uses it from before its copy in until after its copy back, so
   that another domain's run of the same work waits; submissions take the stages
   they touch in the order of their [id]. *)
and stage = { original : buffer; access : access; id : int; busy : Mutex.t }

(* What must stay reachable for as long as a base does. Host memory is the
   bigarray that holds it from its first byte: views of it join that bigarray's
   storage, which outlives the base. *)
and keep =
  | Keep : 'a -> keep
  | Host : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> keep
  | Heap : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t * heap_token -> keep
  | Addressed : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t * 'c -> keep
    (* another device's memory the host addresses, as a bigarray that owns
       nothing, and what keeps the memory *)
  | With : keep * 'a -> keep
    (* a memory's keep, and one more holder, such as a release token *)
  | Staged : keep * stage -> keep
(* a staged buffer's memory, and the memory it stands in for (see [stage]) *)

(* The host memory of [create] is kept with its token: a custom block whose
   finaliser returns the reserved bytes to the host's count. *)
and heap_token

(* The claims on a memory and its consumptions: one record per memory, which
   every base over it holds, its borrows, a file's host pages and the copies of
   a base included. *)
and claim = {
  mutable claims : int; [@atomic]
      (* the read claims, or -1 while one holder is exclusive; a holder outside
         the claims, such as a bigarray's, holds one it never releases *)
  mutable generation : generation; [@atomic]
      (* the memory's: a buffer made at another is dead *)
}

(* A generation of a memory: the first, or the one a consumption began, for the
   reason [why]. Generations compare physically, and only those of one memory
   are compared. *)
and generation = { why : string }

(* A resource's token: a custom block whose collection puts the resource's
   record on its device's release list (see the stubs). The bases over the
   resource keep it. *)
and token

(* The other devices that reach a base's memory, changed together. *)
and links = {
  maps : mapped list; (* the mappings of this memory, one per device *)
  stamps : stamp list;
      (* each device whose work touched this memory, and the latest value that
         work signals: [max_int] where a lost device's work may still write
         it *)
  depends : (unit -> unit) list;
      (* what runs once the memory retires, after all work on it *)
}

(* A device's latest work on a memory. It is raised in place, with the devices
   of the memory taken, so that a submission allocates nothing for the memory
   its devices' work touched before. *)
and stamp = { by : t; mutable upto : int }

(* Released memory that waits for the work that may still use it: [retire] runs
   once each stamp [until] gives is signaled, and the memory is retained, with
   [kept], once one of them is a lost device's unfinished work. [bytes] are
   owned bytes, and [key] the cache the memory goes to. *)
and retiring = {
  until : unit -> (t * int) list;
  bytes : int;
  space : memory; (* the memory [bytes] are *)
  key : (int * memory) option;
  kept : keep;
  retire : unit -> unit;
}

(* A mapping of a base's memory on a device, shared by the device's borrows and
   copies of it, and how the device releases it. [borrows] changes only with the
   device taken. *)
and mapped = {
  on : t;
  mapped : region;
  skip : int; (* bytes of [mapped] before the memory's first byte *)
  unmap : region -> unit;
  ends : ends;
  mutable borrows : int;
  mutable work : (t * int) list; (* the stamps of its released borrows *)
  mutable unmapping : bool;
      (* its last borrow was released, and its unmap waits for [work] *)
}

(* When a mapping is released: once its borrows are unreachable, as a mapping of
   host memory is, or with the memory, as another device's mapping of a device's
   memory is, which its copies into that memory use too. *)
and ends = With_borrows | With_memory

and program = {
  p_device : t;
  p_name : string;
  p_handle : nativeint;
  p_loaded : loaded;
}

(* A file a disk buffer is over, which it names by its path and identity: its
   device, its number there and when it last changed, which the buffer's own
   writes advance. A descriptor of it is open while it is in the disk's
   descriptor cache. Its pages are the host memory of its mapping, made at its
   first borrow. *)
and file = {
  path : string;
  writable : bool;
  size : int;
  mutable identity : identity;
  mutable fd : nativeint option;
  mutable used : int; (* when its descriptor was last used *)
  mutable pages : base option;
}

and identity = { dev : int; ino : int; changed : int }

(* A binary loaded on a device, with the functions found in it. Its programs and
   the buffers of its code keep it, and its token releases it once none does. *)
and loaded = {
  binary : string;
  image : image;
  entries : (string, nativeint) Hashtbl.t; (* with the device taken *)
  kept : keep; (* what its programs and its code keep: its token and itself *)
}

(* What a token puts on its device's release list once it is collected: owned
   memory, or the image of a loaded binary. *)
and release = Memory of base | Code of image

(* A span of [Device_clock] device holds its ticks until its profile is taken,
   which calibrates them. *)
and event =
  | Span of {
      device : t;
      lane : string;
      name : string;
      start : int;
      stop : int;
    }
  | Allocation of { device : t; time : int; allocated : int }
  | Load of { program : program; binary : string; time : int }
  | Counters of {
      device : t;
      name : string;
      start : int;
      stop : int;
      counters : (string * int array) list;
    }
  | Trace of {
      device : t;
      name : string;
      start : int;
      stop : int;
      part : int;
      data : string;
    }
  | Overwritten of { device : t; time : int; runs : int }

and collector = {
  counters : string list;
  trace : bool;
  mutable epochs : epoch list;
      (* the epochs it was taken in, latest first, which only [change] writes *)
}

(* The events recorded while one set of profiles is taken, which each of them
   reads when it stops: every event is one push, which a profile reads with all
   the events recorded before it. The events of devices' reports are apart, and
   each profile keeps those it asks for. *)
and epoch = {
  taken : collector list; (* latest first *)
  counted : string list; (* the counters they ask for *)
  traced : bool; (* whether one asks for traces *)
  events : event list Atomic.t;
  reports : event list Atomic.t;
}

(* A span whose stamps are in the two 16-byte slots at [address], which [stamps]
   keeps, for the profiles taken when it was recorded. *)
and pending = {
  address : nativeint;
  stamps : keep;
  lane : string;
  name : string;
  into : epoch;
}

exception Lost of t * string
exception Out_of_memory of t * int

let () =
  Printexc.register_printer (function
    | Lost (d, why) -> Some (d.name ^ " lost: " ^ why)
    | Out_of_memory (d, n) ->
        Some (Printf.sprintf "Nx_device.Out_of_memory(%s, %d bytes)" d.name n)
    | _ -> None)

(* Host memory *)

external bigarray_address :
  ('a, 'b, 'c) Bigarray.Array1.t -> (nativeint[@unboxed])
  = "caml_nx_device_bigarray_address_byte" "caml_nx_device_bigarray_address"
[@@noalloc]

external memmove :
  (nativeint[@unboxed]) -> (nativeint[@unboxed]) -> (int[@untagged]) -> unit
  = "caml_nx_device_memmove_byte" "caml_nx_device_memmove"

external load_u64 : (nativeint[@unboxed]) -> (int64[@unboxed])
  = "caml_nx_device_load_u64_byte" "caml_nx_device_load_u64"
[@@noalloc]

external store_u64 : (nativeint[@unboxed]) -> (int64[@unboxed]) -> unit
  = "caml_nx_device_store_u64_byte" "caml_nx_device_store_u64"
[@@noalloc]

external external_bytes :
  nativeint ->
  int ->
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
  = "caml_nx_device_external_bytes"

external bigarray_view :
  ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t ->
  ('c, 'd) Bigarray.kind ->
  int ->
  int ->
  ('c, 'd, Bigarray.c_layout) Bigarray.Array1.t = "caml_nx_device_bigarray_view"

external wait_u64 :
  (nativeint[@unboxed]) ->
  (int64[@unboxed]) ->
  (int[@untagged]) ->
  (int[@untagged]) = "caml_nx_device_wait_u64_byte" "caml_nx_device_wait_u64"

external page_size : unit -> int = "caml_nx_device_page_size" [@@noalloc]
external release_list : unit -> nativeint = "caml_nx_device_release_list"

external make_token : nativeint -> release -> int -> int -> int -> token
  = "caml_nx_device_token"

external released : nativeint -> release list = "caml_nx_device_released"

external heap_bytes : unit -> (int[@untagged])
  = "caml_nx_device_heap_bytes_byte" "caml_nx_device_heap_bytes"
[@@noalloc]

external heap_reserve : (int[@untagged]) -> (int[@untagged]) -> bool
  = "caml_nx_device_heap_reserve_byte" "caml_nx_device_heap_reserve"
[@@noalloc]

external heap_return : (int[@untagged]) -> unit
  = "caml_nx_device_heap_return_byte" "caml_nx_device_heap_return"
[@@noalloc]

external heap_token : int -> heap_token = "caml_nx_device_heap_token"

external heap_alloc :
  int -> (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
  = "caml_nx_device_heap_alloc"

external heap_aligned :
  int ->
  int ->
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t option
  = "caml_nx_device_heap_aligned"

external heap_cached : unit -> (int[@untagged])
  = "caml_nx_device_heap_cached_byte" "caml_nx_device_heap_cached"
[@@noalloc]

external heap_drop : unit -> unit = "caml_nx_device_heap_drop" [@@noalloc]
external heap_init : unit -> unit = "caml_nx_device_heap_init" [@@noalloc]

let () = heap_init ()

external now_ns : unit -> (int[@untagged])
  = "caml_nx_device_now_ns_byte" "caml_nx_device_now_ns"
[@@noalloc]

external now_ms : unit -> (int[@untagged])
  = "caml_nx_device_now_ms_byte" "caml_nx_device_now_ms"
[@@noalloc]

(* Files *)

external file_open : string -> int -> int -> int * nativeint * int
  = "caml_nx_device_file_open"

external file_identity : nativeint -> int * int * int * int
  = "caml_nx_device_file_identity"

external file_close : nativeint -> unit = "caml_nx_device_file_close"

external file_read :
  (nativeint[@unboxed]) ->
  (int[@untagged]) ->
  (nativeint[@unboxed]) ->
  (int[@untagged]) ->
  (int[@untagged]) = "caml_nx_device_file_read_byte" "caml_nx_device_file_read"

external file_write :
  (nativeint[@unboxed]) ->
  (int[@untagged]) ->
  (nativeint[@unboxed]) ->
  (int[@untagged]) ->
  (int[@untagged])
  = "caml_nx_device_file_write_byte" "caml_nx_device_file_write"

external error_message : int -> string = "caml_nx_device_error_message"

external file_map :
  nativeint ->
  int ->
  int * (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
  = "caml_nx_device_file_map"

external file_advise : nativeint -> int -> int -> unit
  = "caml_nx_device_file_advise"

(* The codes [file_open] gives, besides the system's, and its modes. *)
let not_regular = -1
let too_many = -2
let read_mode = 0
let write_mode = 1
let create_mode = 2
let page = page_size ()

(* [shared ba] is [ba] with the proxy of its storage made. The runtime makes a
   proxy on a bigarray's first sub without synchronization, which
   [bigarray_view] makes safe for the bigarrays this module alone views: one
   that other code may view, such as one of [of_bigarray], is kept through a sub
   of it. *)
let shared ba = Bigarray.Array1.sub ba 0 (Bigarray.Array1.dim ba)

let heap_memory ba =
  let a = bigarray_address ba in
  {
    host = Some a;
    address = a;
    handle = 0n;
    nbytes = Bigarray.Array1.size_in_bytes ba;
  }

(* Host buffers of at least this many bytes start on a page, so that devices can
   map them: a mapping locks whole pages, which memory of another buffer must
   not share. Aligning costs up to a page of slack, at most a quarter of the
   buffer; smaller buffers are copied through staging instead. *)
let aligned_from = Int.max (64 * 1024) (4 * page)

(* [n] bytes of the heap, on a page from [aligned_from]: allocated there, or cut
   from a page of more bytes where the C library aligns nothing it frees. Those
   bytes pace the collector by the program's whole memory (see the stubs).
   Smaller buffers are the runtime's own bigarrays, which the minor heap holds
   and where most die: the runtime paces minor collections by the memory they
   hold against the minor heap's size, which [caml_alloc_custom] would replace
   with the bound it paces major cycles by. *)
let heap n =
  if n < aligned_from then
    Bigarray.Array1.create Bigarray.char Bigarray.c_layout n
  else
    match heap_aligned page n with
    | Some ba -> ba
    | None ->
        if n > max_int - page then raise Stdlib.Out_of_memory;
        let ba = heap_alloc (n + page - 1) in
        let a = Nativeint.to_int (bigarray_address ba) in
        let skip = (page - (a mod page)) mod page in
        Bigarray.Array1.sub ba skip n

(* Profiling *)

(* The epoch of the profiles being taken, if any. Every recording site reads it
   once, and allocates nothing when it is [None]. *)
let profiles : epoch option Atomic.t = Atomic.make None

let rec push r x =
  let l = Atomic.get r in
  if not (Atomic.compare_and_set r l (x :: l)) then push r x

(* A host program's calls run with the runtime released, through Program.entry,
   which records their spans in C until the profile stops: each a program's
   name, its domain's number, its start and its stop. The names are those of the
   host's programs, registered as they are loaded. *)
external record_spans : bool -> unit = "caml_nx_device_record_spans"

external host_spans : unit -> (string * int * int * int) array
  = "caml_nx_device_host_spans"

external name_program : nativeint -> string -> unit
  = "caml_nx_device_name_program"

(* Pools *)

let used p =
  match p.in_use with Count c -> Atomic.get c | Heap_bytes -> heap_bytes ()

(* The bytes [p] keeps for reuse: the heap's are the collected buffers its
   allocator keeps (see the stubs). *)
let pool_cached p =
  match p.in_use with Count _ -> p.cached | Heap_bytes -> heap_cached ()

(* Loaded code is counted after its driver allocated it, so [room] can be
   negative: allocations of that memory are then refused, and the last resort
   collects until unloaded code makes room again. *)
let room p = p.ceiling - used p - pool_cached p - p.retained

(* Applies [f] to each pool an allocation of [d]'s memory [kind] counts in:
   mapped memory is the device's own, through a window with a ceiling of its
   own. *)
let on_pools d kind f =
  match (d.kind, kind) with
  | (Machine { pool; _ } | Shared { pool; _ }), _ -> f pool
  | Local { own; mapped = Some window; _ }, Mapped ->
      f window;
      f own
  | Local { pinned; _ }, (Pinned | Mapped) -> f pinned
  | Local { own; _ }, Device -> f own
  | Disk, _ -> ()

(* [f] summed over each pool of [d], once. Nothing is allocated: devices count
   on every operation. *)
let sum f d =
  match d.kind with
  | Machine { pool; _ } | Shared { pool; _ } -> f pool
  | Local { own; pinned; _ } -> f own + f pinned
  | Disk -> 0

let allocated d = sum used d
let cached d = sum pool_cached d
let retained d = sum (fun p -> p.retained) d

let allocate_bytes d kind n =
  on_pools d kind (fun p ->
      match p.in_use with
      | Count c -> ignore (Atomic.fetch_and_add c n)
      | Heap_bytes -> heap_return (-n))

(* Bytes of [d]'s memory [kind] counted as retained, or no longer. *)
let retain_bytes d kind n =
  on_pools d kind (fun p -> p.retained <- p.retained + n)

let cache_bytes d kind n = on_pools d kind (fun p -> p.cached <- p.cached + n)

let memory_changed d =
  match Atomic.get profiles with
  | None -> ()
  | Some ep ->
      push ep.events
        (Allocation { device = d; time = now_ns (); allocated = allocated d })

(* The lane of the calling domain on the host. *)
let lane_of domain = Printf.sprintf "domain %d" domain
let domain_lane () = lane_of (Domain.self () :> int)

(* Devices *)

let ids = Atomic.make 0
let opened = Atomic.make []

let rec remember d =
  let l = Atomic.get opened in
  if not (Atomic.compare_and_set opened l (d :: l)) then remember d

(* Words of a machine's memory: this process's, or another machine's through its
   host's [io]. *)
let read_word io a =
  match io with
  | None -> load_u64 a
  | Some io ->
      let b = Bigarray.Array1.create Bigarray.int64 Bigarray.c_layout 1 in
      io.read ~src:a ~dst:(bigarray_address b) 8;
      Bigarray.Array1.unsafe_get b 0

let write_word io a v =
  match io with
  | None -> store_u64 a v
  | Some io ->
      let b = Bigarray.Array1.create Bigarray.int64 Bigarray.c_layout 1 in
      Bigarray.Array1.unsafe_set b 0 v;
      io.write ~dst:a ~src:(bigarray_address b) 8

(* The timeline is memory that the device's machine's host and the device's work
   address: the device's host memory when it allocates some, the heap on this
   machine, the host's memory on another. It lives as long as the device. Its
   first word is the signal word, and after a word that aligns them come two
   slots of 16 bytes, whose second words take the timestamps of the device's
   copy queue. On the heap, it starts on a page for a device that maps [pages]
   of host memory, which maps no other. *)
let timeline_bytes = 48

let timeline_of ~pages ~host_alloc (host_memory : allocator option) =
  let check = function
    | Some ({ host = Some _; _ } as m) -> m
    | Some { host = None; _ } ->
        invalid_arg
          "Nx_device.Driver.device: host memory the host does not address"
    | None -> failwith "Nx_device.Driver.device: no memory for the timeline"
  in
  match (host_memory, host_alloc) with
  | Some (a : allocator), _ | None, Some a ->
      (check (a.alloc timeline_bytes), Keep ())
  | None, None when pages ->
      let ba = Bigarray.Array1.sub (heap aligned_from) 0 timeline_bytes in
      (heap_memory ba, Host ba)
  | None, None ->
      let ba =
        shared
          (Bigarray.Array1.create Bigarray.int64 Bigarray.c_layout
             (timeline_bytes / 8))
      in
      (heap_memory ba, Host ba)

(* A device as [create] takes it, but for its memory. *)
module Description = struct
  type nonrec t = {
    name : string;
    arch : string;
    machine : t option;
    budget : int;
    peer : (t -> region -> (region * (unit -> unit), string) result) option;
    reaches_peer : t -> bool;
    load : (binary:string -> (image, string) result) option;
    call : (nativeint -> (nativeint * int) array -> int array -> unit) option;
    link : (src:buffer -> dst:buffer -> link option) option;
    dma : (region -> (dma, string) result) option;
    completion : completion;
    synchronized : unit -> unit;
    report : (unit -> event list) option;
    room : unit -> bool;
    finalize : failed:bool -> unit;
    resolve : nativeint -> unit;
  }

  let default =
    {
      name = "";
      arch = "";
      machine = None;
      budget = max_int;
      peer = None;
      reaches_peer = (fun _ -> false);
      load = None;
      call = None;
      link = None;
      dma = None;
      completion = Poll;
      synchronized = ignore;
      report = None;
      room = (fun () -> true);
      finalize = (fun ~failed:_ -> ());
      resolve = ignore;
    }
end

(* The memory a device is made of: a driver's description, or the host's and the
   disk's. *)
type made_of =
  | Process_heap (* this machine's host *)
  | Remote of { address : string; io : io; memory : allocator }
  | Files (* the disk *)
  | Driver_memory of driver_memory

(* How a driver describes a device's memory (see [Driver.memory]). *)
and driver_memory =
  | Host_visible of { memory : allocator; mapping : mapping option }
  | Device_local of {
      memory : allocator;
      host_memory : allocator;
      mapped : (allocator * int) option;
      mapping : mapping;
      queue : timeline:region -> queue;
    }

let create (desc : Description.t) made_of =
  (* A host of another machine keeps its timeline in its own memory. *)
  let host_alloc =
    match (desc.machine, made_of) with
    | Some { kind = Machine { pool = { allocator = Some a; _ }; _ }; _ }, _
    | None, Remote { memory = a; _ } ->
        Some a
    | _ -> None
  in
  let pages, host_memory =
    match made_of with
    | Driver_memory (Host_visible { mapping = Some (Pages _); _ }) ->
        (true, None)
    | Driver_memory (Device_local { host_memory; mapping; _ }) ->
        ( (match mapping with Pages _ -> true | Identity -> false),
          Some host_memory )
    | Process_heap | Remote _ | Files | Driver_memory (Host_visible _) ->
        (false, None)
  in
  let timeline, timeline_keep = timeline_of ~pages ~host_alloc host_memory in
  let machine_io =
    match (desc.machine, made_of) with
    | Some { kind = Machine { remote = Some (_, io); _ }; _ }, _
    | None, Remote { io; _ } ->
        Some io
    | _ -> None
  in
  let words = Option.get timeline.host in
  write_word machine_io words 0L;
  let pool ?(in_use = Count (Atomic.make 0)) ceiling allocator =
    { allocator; ceiling; in_use; cached = 0; retained = 0 }
  in
  let kind =
    match made_of with
    | Process_heap ->
        Machine
          { remote = None; pool = pool ~in_use:Heap_bytes desc.budget None }
    | Remote { address; io; memory } ->
        Machine
          { remote = Some (address, io); pool = pool desc.budget (Some memory) }
    | Files -> Disk
    | Driver_memory (Host_visible { memory; mapping }) ->
        Shared { pool = pool desc.budget (Some memory); mapping }
    | Driver_memory
        (Device_local { memory; host_memory; mapped; mapping; queue }) ->
        Local
          {
            own = pool desc.budget (Some memory);
            mapped = Option.map (fun (a, window) -> pool window (Some a)) mapped;
            pinned = pool max_int (Some host_memory);
            mapping;
            queue = queue ~timeline;
          }
  in
  let signal, sleep =
    match desc.completion with
    | Poll -> (None, None)
    | Sleep sleep -> (None, Some (sleep ~timeline))
    | Signal signal -> (Some (signal ~timeline), None)
  in
  let d =
    {
      id = Atomic.fetch_and_add ids 1;
      name = desc.name;
      arch = desc.arch;
      machine = desc.machine;
      kind;
      lock = Mutex.create ();
      peer = desc.peer;
      reaches_peer = desc.reaches_peer;
      load = desc.load;
      call = desc.call;
      link = desc.link;
      dma = desc.dma;
      signal;
      sleep;
      synchronized = desc.synchronized;
      report = desc.report;
      room = desc.room;
      finalize = desc.finalize;
      clock = (match kind with Local l -> l.queue.clock | _ -> Host_clock);
      resolve = desc.resolve;
      timeline;
      timeline_keep;
      last = Atomic.make 0;
      settled = Atomic.make 0;
      slots = None;
      staging = None;
      released = release_list ();
      failed = Atomic.make None;
      cache = Hashtbl.create 16;
      retiring = [];
      pending = Hashtbl.create 4;
      images = Hashtbl.create 16;
      indexed = 8;
      spans = Atomic.make [];
      held = [];
      bytes_in = 0;
      bytes_out = 0;
    }
  in
  remember d;
  d

let host_arch = match Host_arch.architecture with "amd64" -> "x86_64" | a -> a

(* How the host loads and calls programs: ELF objects linked into its memory. *)
external call_addresses :
  nativeint -> (nativeint * int) array -> int array -> unit
  = "caml_nx_device_call_addresses"

let host_programs =
  let load =
    match Host_program.load with
    | Some load -> load
    | None -> fun ~binary:_ ~entry:_ -> Error "the host loads no programs"
  in
  { load; call = call_addresses }

(* A host's image of [binary]: each function is linked at its first load, and
   unlinked with the image. *)
let host_image load ~binary =
  let unloads = ref [] in
  let entry entry =
    Result.map
      (fun (h, unload) ->
        unloads := unload :: !unloads;
        h)
      (load ~binary ~entry)
  in
  Ok
    {
      code = None;
      entry;
      unload = (fun () -> List.iter (fun f -> f ()) !unloads);
    }

let host =
  let load = Option.map host_image Host_program.load in
  create
    { Description.default with name = "CPU"; arch = host_arch; load }
    Process_heap

(* The disk's buffers are files, which it opens and never allocates. It has no
   processor. *)
let disk =
  create { Description.default with name = "DISK"; machine = Some host } Files

let name d = d.name
let arch d = d.arch
let equal = ( == )
let compare a b = Int.compare a.id b.id
let pp ppf d = Format.pp_print_string ppf d.name

(* The pool whose ceiling is [d]'s budget, if any. *)
let budgeted d =
  match d.kind with
  | Machine { pool; _ } | Shared { pool; _ } | Local { own = pool; _ } ->
      Some pool
  | Disk -> None

(* Allocates nothing: the host checks it for every buffer. *)
let budget d =
  match d.kind with
  | Machine { pool; _ } | Shared { pool; _ } | Local { own = pool; _ } ->
      pool.ceiling
  | Disk -> max_int

let host_of d = match d.machine with Some h -> h | None -> d

(* How [d] addresses host memory, if it does. *)
let mapping_of d =
  match d.kind with
  | Machine { remote = None; _ } -> Some Identity
  | Shared { mapping; _ } -> mapping
  | Local { mapping; _ } -> Some mapping
  | Machine { remote = Some _; _ } | Disk -> None

let queue_of d =
  match d.kind with Local { queue; _ } -> Some queue | _ -> None

(* The allocator of [d]'s memory [kind], if [d] allocates it: its first
   pool's. *)
let allocator_of d kind =
  let first = ref None in
  on_pools d kind (fun p -> if Option.is_none !first then first := p.allocator);
  !first

(* Memory the host addresses, as the host's own and a device's over it are. *)
let host_addressed d =
  match d.kind with
  | Machine { remote = None; _ } | Shared { mapping = Some _; _ } -> true
  | Machine _ | Shared _ | Local _ | Disk -> false

let shares_host_memory d = host_of d == host && host_addressed d

let runs_on_host d =
  d == host
  ||
  match d.kind with
  | Shared _ -> host_of d == host && Option.is_none d.load
  | Machine _ | Local _ | Disk -> false

let reaches d d' =
  d == d'
  || (d != disk && d' != disk && host_of d == host_of d')
     &&
     let maps = Option.is_some (mapping_of d) in
     if d' == host_of d then maps
     else if d == host_of d then host_addressed d'
     else (host_addressed d' && maps) || d.reaches_peer d'

(* How the process reaches the memory of [d]'s machine: [None] on this one. *)
let io_of d =
  match (host_of d).kind with
  | Machine { remote = Some (_, io); _ } -> Some io
  | _ -> None

(* Timeline *)

let timeline_address d = Option.get d.timeline.host
let submitted d = Atomic.get d.last

(* The last value [d] signaled. A driver that signals in its own way raises
   [Failure] if [d] faulted. *)
let read_signaled d =
  match d.signal with
  | Some s -> s.signaled ()
  | None -> Int64.to_int (read_word (io_of d) (timeline_address d))

let commit d v = Atomic.set d.last v

(* The devices lost so far in the process. While it is [0], checking whether a
   lost device reaches some memory is this one read. *)
let losses = Atomic.make 0

(* A device that hung or faulted is in an unknown state: its first error loses
   it for good, and every later operation raises that error at once. *)
let lose d why =
  if Atomic.compare_and_set d.failed None (Some why) then Atomic.incr losses

let check d =
  match Atomic.get d.failed with
  | None -> ()
  | Some why -> raise (Lost (d, why))

let fail d why =
  lose d why;
  raise (Lost (d, Option.get (Atomic.get d.failed)))

(* Runs [f], a callback of [d]'s driver: a [Failure] is a fault, which loses
   [d]. *)
let driver d f = try f () with Failure why -> fail d why

(* How long a wait sees no progress before it lets the device sleep on its
   interrupts, and the longest any call a wait blocks in lasts: each returns to
   OCaml code within it, where a pending [Sys.Break] from Ctrl-C raises. *)
let slice_ms = 200

(* Waits until [ready ()]. Once the word at [word], in the memory [io] reaches,
   has stayed still for [slice_ms], [sleep ~still ms] runs between checks,
   [still] being how long it has: the driver blocks on the device's interrupts
   and raises its faults, and it alone decides whether a still word is a hang.
   Before that, and without [sleep], [spin ~still] runs between checks. A
   condition that holds at once reads no word. *)
let await ~io ~sleep ~spin word ready =
  let rec go seen since =
    if not (ready ()) then begin
      let w = read_word io word and now = now_ms () in
      let since = if w <> seen then now else since in
      let still = now - since in
      (match sleep with
      | Some sleep when still >= slice_ms -> sleep ~still slice_ms
      | _ -> spin ~still);
      go w since
    end
  in
  if not (ready ()) then go (read_word io word) (now_ms ())

(* Waits until the word at [word] reaches [v]. On this machine a wait blocks in
   the word's own wait between checks; on another, each read of the word is a
   round trip. *)
let poll_word ~io ~sleep word v =
  let target = Int64.of_int v in
  let reached () = Int64.unsigned_compare (read_word io word) target >= 0 in
  let spin ~still =
    if Option.is_none io then
      let ms = if still < slice_ms then slice_ms - still else slice_ms in
      ignore (wait_u64 word target ms)
  in
  await ~io ~sleep ~spin word reached

let relax ~still:_ = Domain.cpu_relax ()

(* Values complete in order: a wait for a value at or below one a wait saw
   signaled is over, with no read of the device's machine. *)
let rec settle d v =
  let s = Atomic.get d.settled in
  if v > s && not (Atomic.compare_and_set d.settled s v) then settle d v

(* Waits until [d] signals [v], however long that takes: only a fault its driver
   reports, or the failed connection to its machine, loses [d]. *)
let wait_signal d v =
  check d;
  if v > Atomic.get d.settled then begin
    (try
       match d.signal with
       | Some s ->
           while not (s.wait v ~ms:slice_ms) do
             ()
           done
       | None -> poll_word ~io:(io_of d) ~sleep:d.sleep (timeline_address d) v
     with Failure why -> fail d why);
    settle d v
  end

let failed d = Atomic.get d.failed
let lost = failed

(* A fault the driver reports while [d]'s value is read loses [d], whose value
   stays the last one a wait saw. *)
let signaled d =
  match read_signaled d with
  | v -> v
  | exception Failure why ->
      lose d why;
      Atomic.get d.settled

(* Waits until each of [d]'s queues has room for a submission, as a wait for its
   work does: work completing frees room. *)
let wait_room d =
  driver d (fun () ->
      await ~io:(io_of d) ~sleep:d.sleep ~spin:relax (timeline_address d) d.room)

(* A driver error while enqueueing leaves [d]'s queue in an unknown state: like
   a fault, it fails [d]. *)
let enqueue d (f : signal:int -> unit) =
  let v = submitted d + 1 in
  driver d (fun () -> f ~signal:v);
  commit d v;
  v

(* Timestamp slot [i] of [d]'s timeline memory, as [d]'s work addresses it, and
   the timestamp it holds. *)
let stamp_slot d i =
  Nativeint.add d.timeline.address (Nativeint.of_int (16 + (16 * i)))

let stamp d i =
  Int64.to_int
    (read_word (io_of d)
       (Nativeint.add (timeline_address d) (Nativeint.of_int (24 + (16 * i)))))

(* Reads the stamps of the spans recorded on [d], whose work is done: the second
   words of their two slots. A later record of the same stamps replaced the
   earlier ones. *)
let read_spans d =
  match Atomic.get d.spans with
  | [] -> ()
  | _ :: _ ->
      let seen = Hashtbl.create 8 in
      List.iter
        (fun p ->
          if not (Hashtbl.mem seen p.address) then begin
            Hashtbl.add seen p.address ();
            driver d (fun () -> d.resolve p.address);
            let io = io_of d in
            let start = Int64.to_int (read_word io (Nativeint.add p.address 8n))
            and stop =
              Int64.to_int (read_word io (Nativeint.add p.address 24n))
            in
            (* [resolve] may release the runtime: the stamps must outlive their
               reads. *)
            ignore (Sys.opaque_identity p.stamps);
            push p.into.events
              (Span { device = d; lane = p.lane; name = p.name; start; stop })
          end)
        (Atomic.exchange d.spans [])

let counts c = c.counters <> [] || c.trace

(* [e], a report's event, as the profile [c] asks for it: the counters it asks
   for, in its order, traces and the spans decoded from them if it asks for
   traces, and the runs lost if it asks for either. *)
let reported c e =
  match e with
  | Counters r -> (
      let asked n =
        Option.map (fun v -> (n, v)) (List.assoc_opt n r.counters)
      in
      match List.filter_map asked c.counters with
      | [] -> None
      | counters -> Some (Counters { r with counters }))
  | Trace _ | Span _ -> if c.trace then Some e else None
  | Overwritten _ -> if counts c then Some e else None
  | Allocation _ | Load _ -> Some e

(* The counters and traces of [d]'s work done since its last report, into the
   profiles being taken if one asks for some. [d] is taken. *)
let read_reports d =
  match (d.report, Atomic.get profiles) with
  | Some report, Some ep when ep.counted <> [] || ep.traced ->
      List.iter (push ep.reports) (driver d report)
  | Some _, _ | None, _ -> ()

(* Waits for [d]'s work and for the work that touched [d]'s memory, then reads
   the stamps of the spans recorded on [d] and the counters of its work. [d] is
   taken. A failed device will never signal, so its work is not waited for: the
   memory it can reach raises its error instead. *)
let sync d =
  wait_signal d (submitted d);
  Hashtbl.iter
    (fun _ (d', v) ->
      if failed d' = None then try wait_signal d' v with Lost _ -> ())
    d.pending;
  read_spans d;
  read_reports d;
  try d.synchronized () with Failure why -> fail d why

(* Every memory's generation until its first consumption: buffers made then
   compare it physically. *)
let first_generation = { why = "" }

(* Raises [Lost] for a lost device that can reach [base]'s memory: its own
   device, a device it is mapped on or whose transfer into it could not be
   waited for, or those of the memory it maps. *)
let rec reach_lost base =
  let { maps; stamps; _ } = Atomic.get base.links in
  List.iter check (base.owner :: List.map (fun m -> m.on) maps);
  List.iter
    (fun s -> if s.upto > Atomic.get s.by.settled then check s.by)
    stamps;
  Option.iter (fun (src, _) -> reach_lost src) base.source

let check_reach base = if Atomic.get losses > 0 then reach_lost base

let rec update_links base f =
  let l = Atomic.get base.links in
  if not (Atomic.compare_and_set base.links l (f l)) then update_links base f

let no_links = { maps = []; stamps = []; depends = [] }

(* [stamps] with [d]'s work signaling [v]. *)
let rec stamped stamps d v =
  match stamps with
  | [] -> [ (d, v) ]
  | (d', v') :: rest when d' == d -> (d, Int.max v v') :: rest
  | s :: rest -> s :: stamped rest d v

let rec raise_stamp d v = function
  | [] -> false
  | s :: rest ->
      if s.by == d then begin
        if v > s.upto then s.upto <- v;
        true
      end
      else raise_stamp d v rest

let stamp_use base d v =
  if not (raise_stamp d v (Atomic.get base.links).stamps) then
    update_links base (fun l ->
        if raise_stamp d v l.stamps then l
        else { l with stamps = { by = d; upto = v } :: l.stamps })

(* [base]'s memory stamped by the work of [d] that may still use it, after an
   error: [d]'s last value, or [max_int] once [d] is lost. *)
let poison base d =
  stamp_use base d (if failed d = None then submitted d else max_int)

(* Whether the work of [stamps] is done, still to be waited for, or a lost
   device's unfinished work. Nothing blocks. *)
let progress stamps =
  List.fold_left
    (fun acc (d, v) ->
      if acc = `Lost || v <= Atomic.get d.settled then acc
      else if failed d <> None then `Lost
      else
        match read_signaled d with
        | s when s >= v ->
            settle d v;
            acc
        | _ -> `Waiting
        | exception Failure why ->
            lose d why;
            `Lost)
    `Done stamps

let update_maps base f = update_links base (fun l -> { l with maps = f l.maps })

let mapping_on d base =
  List.find_opt (fun m -> m.on == d) (Atomic.get base.links).maps

(* [d]'s mapping of [src]'s memory, which [map] makes if there is none yet, with
   no borrow of it. [d] is taken. *)
let map_on d src ~ends ~map =
  match mapping_on d src with
  | Some m -> Ok m
  | None ->
      Result.map
        (fun (mapped, skip, unmap) ->
          let m =
            {
              on = d;
              mapped;
              skip;
              unmap;
              ends;
              borrows = 0;
              work = [];
              unmapping = false;
            }
          in
          update_maps src (List.cons m);
          m)
        (driver d map)

(* How a driver's [peer] maps the whole memory of [src], and unmaps it. *)
let peer_map peer src () =
  Result.map
    (fun ((mapped : region), unmap) ->
      (mapped, 0, fun (_ : region) -> unmap ()))
    (peer src.owner src.memory)

(* The device's address of the host address [a] in the mapping [m]. *)
let mapped_address (m : region) a =
  Nativeint.add m.address (Nativeint.sub a (Option.get m.host))

(* The disk's descriptors. A file's descriptor is open while the file is in this
   cache, which holds at most [max_descriptors] of them and closes the least
   recently used to open another: a disk buffer holds no descriptor while it
   waits to be collected. A file reopened by its path must still be the one the
   buffer opened. Everything here runs with the disk taken. *)

let max_descriptors = 64
let descriptors : file list ref = ref []
let descriptor_uses = ref 0

let close_descriptor f =
  match f.fd with
  | None -> ()
  | Some fd ->
      f.fd <- None;
      descriptors := List.filter (fun f' -> f' != f) !descriptors;
      file_close fd

let close_descriptors () = List.iter close_descriptor !descriptors

let identify path fd =
  match file_identity fd with
  | 0, dev, ino, changed -> { dev; ino; changed }
  | code, _, _, _ ->
      file_close fd;
      raise (Sys_error (path ^ ": " ^ error_message code))

(* Opens [path] with [mode], closing the cached descriptors once if the process
   has too many open. *)
let open_path path mode n =
  match file_open path mode n with
  | code, _, _ when code = too_many && !descriptors <> [] ->
      close_descriptors ();
      file_open path mode n
  | r -> r

let cache f fd =
  if List.length !descriptors >= max_descriptors then begin
    let oldest =
      List.fold_left
        (fun o f -> if f.used < o.used then f else o)
        (List.hd !descriptors) !descriptors
    in
    close_descriptor oldest
  end;
  incr descriptor_uses;
  f.used <- !descriptor_uses;
  f.fd <- Some fd;
  descriptors := f :: !descriptors

(* [f]'s descriptor, opened again if it was closed. Raises [Sys_error] naming
   the file if it cannot be opened, or if [f.path] now names another file or one
   that changed since [f]'s buffer last saw it. *)
let descriptor f =
  match f.fd with
  | Some fd ->
      incr descriptor_uses;
      f.used <- !descriptor_uses;
      fd
  | None -> (
      let mode = if f.writable then write_mode else read_mode in
      match open_path f.path mode 0 with
      | 0, fd, _ ->
          let i = identify f.path fd and i' = f.identity in
          if i.dev <> i'.dev || i.ino <> i'.ino || i.changed <> i'.changed then begin
            file_close fd;
            raise
              (Sys_error
                 (f.path ^ ": the file changed since its buffers opened it"))
          end;
          cache f fd;
          fd
      | code, _, _ when code = too_many ->
          raise (Sys_error (f.path ^ ": too many open files"))
      | code, _, _ -> raise (Sys_error (f.path ^ ": " ^ error_message code)))

(* Memory reclamation. Everything below runs with the device taken. *)

(* Frees cached [memories] of [d], each of its memory, once [d]'s work that
   touched them, which signals [v] at the latest, is done: at once with
   [~wait:false], which is given only memories whose work is done. If that work
   cannot be waited for, or a free faults, which loses [d], the memory not yet
   freed is retained, never freed or reused, since its state is unknown. *)
let free_all d ~wait v memories =
  let retain rest =
    List.iter (fun (_, (m : region), kind) -> retain_bytes d kind m.nbytes) rest;
    d.held <- Keep rest :: d.held
  in
  let rec go = function
    | [] -> ()
    | (free, m, _) :: tail as rest -> (
        match free m with
        | () -> go tail
        | exception Failure why ->
            retain rest;
            fail d why)
  in
  if memories <> [] then
    match if wait then wait_signal d v with
    | () -> go memories
    | exception (Lost _ as e) ->
        retain memories;
        raise e

let free_of d kind = (Option.get (allocator_of d kind)).free

(* The memory the code of [i] lies in on [d]. On a device with memory of its
   own, it is that memory where the host does not address the code, and
   otherwise memory the host writes and [d]'s work reads: its mapped memory
   where it has a window, and pinned memory where it has none. *)
let code_kind d (i : image) =
  match (d.kind, i.code) with
  | Local _, Some { host = None; _ } -> Device
  | Local { mapped = Some _; _ }, _ -> Mapped
  | Local _, _ -> Pinned
  | (Machine _ | Shared _ | Disk), _ -> Device

(* The bytes of [i]'s code in its device's memory, [0] for a driver that keeps
   it elsewhere. *)
let code_bytes (i : image) = match i.code with Some r -> r.nbytes | None -> 0

(* The room left in [d]'s pools for its memory [kind]. Nothing is allocated:
   devices check it on every operation. *)
let room_for d kind =
  match (d.kind, kind) with
  | (Machine { pool; _ } | Shared { pool; _ }), _ -> room pool
  | Local { own; mapped = Some window; _ }, Mapped ->
      Int.min (room window) (room own)
  | Local { pinned; _ }, (Pinned | Mapped) -> room pinned
  | Local { own; _ }, Device -> room own
  | Disk, _ -> max_int

let fits d kind n = n <= room_for d kind

(* Whether [d]'s window and its own memory could hold [n] more bytes of mapped
   memory once its cache is released. *)
let mappable d n =
  match d.kind with
  | Local { own; mapped = Some window; _ } ->
      let free p = p.ceiling - used p - p.retained in
      n <= free window && n <= free own
  | Local { mapped = None; _ } | Machine _ | Shared _ | Disk -> false

(* A token that puts [r] on [d]'s release list once it is collected. Its [bytes]
   of [d]'s memory [kind] pace the collector by the room left in its pools, and
   for memory that is the host's, by the program's memory too (see the
   stubs). *)
let release_token d kind r bytes =
  let room = room_for d kind in
  let live =
    match (d.kind, kind) with
    | Shared { pool; _ }, _ when shares_host_memory d -> used pool
    | Local { pinned; _ }, Pinned -> used pinned
    | _ -> -1
  in
  make_token d.released r bytes (Int.max 0 room) live

(* [base], whose memory [d] releases once the base returned and every base made
   from it are unreachable: they keep a token that puts [base], which keeps
   none, on [d]'s release list once it is collected. A base has no mutable
   field, so [base] sees the links and claims that its copies change. *)
(* The memory a staged buffer's base stands in for, if it is one. *)
let rec staged_keep = function
  | Staged (_, st) -> Some st
  | With (keep, _) -> staged_keep keep
  | Keep _ | Host _ | Heap _ | Addressed _ -> None

let stage_of base = staged_keep base.keep

let owned d (base : base) =
  {
    base with
    keep = With (base.keep, release_token d base.kind (Memory base) base.bytes);
  }

(* How much of a device's cache to free: all of it, or until its memory [kind]
   has room for [n] more bytes. *)
type want = All | Room of memory * int

(* Frees cached memory, of the memory [only] if given, to the system until [d]
   has what [want] asks, or its cache is empty. With [~wait:false], only memory
   whose work is done is freed, and nothing blocks. *)
let enough d = function All -> false | Room (kind, n) -> fits d kind n

let release_cache ?only ~wait d want =
  match d.kind with
  | Machine { pool = { in_use = Heap_bytes; _ }; _ } ->
      if not (enough d want) then heap_drop ()
  | _ ->
      if cached d > 0 && not (enough d want) then begin
        let first = match want with All -> Device | Room (kind, _) -> kind in
        let freed = ref [] and last = ref 0 in
        (* Memory of the kind asked for goes first: other memory may not free
           the pool that refuses it, such as mapped memory's window. *)
        let keys =
          Hashtbl.fold
            (fun ((_, k) as key) _ acc ->
              if Option.fold ~none:true ~some:(( = ) k) only then key :: acc
              else acc)
            d.cache []
          |> List.stable_sort (fun (_, a) (_, b) ->
              Bool.compare (a <> first) (b <> first))
        in
        List.iter
          (fun ((size, cached_kind) as key) ->
            let free = free_of d cached_kind in
            let rec drop = function
              | e :: es when enough d want -> e :: es
              | (e : reusable) :: es ->
                  if wait || progress [ (d, e.released) ] = `Done then begin
                    cache_bytes d cached_kind (-size);
                    freed := (free, e.region, cached_kind) :: !freed;
                    last := Int.max !last e.released;
                    drop es
                  end
                  else e :: drop es
              | [] -> []
            in
            match drop (Hashtbl.find d.cache key) with
            | [] -> Hashtbl.remove d.cache key
            | ms -> Hashtbl.replace d.cache key ms)
          keys;
        free_all d ~wait !last !freed
      end

(* The latest value of [d]'s own work in [stamps], and the others. *)
let own_stamp d stamps =
  List.fold_left (fun v (d', v') -> if d' == d then v' else v) 0 stamps

let foreign d stamps = List.filter (fun (d', _) -> d' != d) stamps

(* Released owned memory of [b] enters [d]'s cache, with [d]'s own work on it:
   [d]'s later work on it is ordered after that work by [d]'s queue. *)
let cache_memory d (b : base) ~touched ~released =
  allocate_bytes d b.kind (-b.bytes);
  let key = (b.bytes, b.kind) in
  let ms = Option.value ~default:[] (Hashtbl.find_opt d.cache key) in
  Hashtbl.replace d.cache key ({ region = b.memory; touched; released } :: ms);
  cache_bytes d b.kind b.bytes

(* [m], [d]'s mapping of [src]'s memory, is unmapped: its borrows are
   unreachable and their work is done. Only then does the memory leave [d]'s
   reach. *)
let unmap d src m =
  driver d (fun () -> m.unmap m.mapped);
  update_maps src (List.filter (fun m' -> m' != m))

(* Retires each released memory of [d] whose work is done, retains those a lost
   device's unfinished work touched, and keeps the others waiting. *)
let retire d =
  let before = allocated d in
  let rec go = function
    | [] -> ()
    | r :: rest -> (
        match progress (r.until ()) with
        | `Waiting ->
            d.retiring <- r :: d.retiring;
            go rest
        | `Lost ->
            if r.bytes > 0 then begin
              allocate_bytes d r.space (-r.bytes);
              retain_bytes d r.space r.bytes
            end;
            d.held <- r.kept :: d.held;
            go rest
        | `Done -> (
            match r.retire () with
            | () -> go rest
            | exception e ->
                (* A release that faulted lost [d]: what it released stays. *)
                d.held <- r.kept :: d.held;
                d.retiring <- rest @ d.retiring;
                raise e))
  in
  match d.retiring with
  | [] -> ()
  | pending -> (
      d.retiring <- [];
      match go pending with
      | () -> if allocated d <> before then memory_changed d
      | exception e ->
          if allocated d <> before then memory_changed d;
          raise e)

(* Unreachable memory is released, and retires once the work that may still use
   it is done (see [retire]). Owned memory waits for other devices' work alone:
   [d]'s own work on it orders its reuse. Owned memory that something depends on
   waits for all work, [d]'s too, since what depends on it may hold objects that
   work reads. So does a staged buffer: the host writes it, which no queue
   orders after [d]'s reads of it. A mapping waits for every device's work on
   its borrows, [d]'s too, once the last of them is unreachable: a mapping of
   host memory is unmapped, and one of another device's memory stays its
   driver's until that memory is freed. Host memory never comes here: it is the
   heap's, returned when the collector finds its base unreachable. *)
let release d (b : base) =
  let stamps = List.map (fun s -> (s.by, s.upto)) (Atomic.get b.links).stamps in
  match b.source with
  | Some (_, m) when m.ends = With_memory -> m.borrows <- m.borrows - 1
  | Some (src, m) ->
      m.work <- List.fold_left (fun l (e, v) -> stamped l e v) m.work stamps;
      m.borrows <- m.borrows - 1;
      (* The unmap also waits for every piece of [d]'s work submitted before the
         last borrow went, which may reach the mapping without having listed it.
         A borrow made before the unmap takes the mapping again: the unmap then
         does nothing, and the next last release queues another. *)
      if m.borrows = 0 then m.work <- stamped m.work d (submitted d);
      if m.borrows = 0 && not m.unmapping then begin
        m.unmapping <- true;
        d.retiring <-
          {
            until = (fun () -> m.work);
            bytes = 0;
            space = Device;
            key = None;
            kept = Keep b;
            retire =
              (fun () ->
                m.unmapping <- false;
                if m.borrows = 0 then unmap d src m);
          }
          :: d.retiring
      end
  | None when Option.is_some b.file ->
      (* No read or write of a file outlives the copy that made it. *)
      close_descriptor (Option.get b.file)
  | None ->
      (* A free to the driver waits for every piece of [d]'s work submitted
         before the release, listed or not, as tinygrad's free synchronizes the
         device; [d]'s own reuse waits for none of it. *)
      let touched = own_stamp d stamps in
      let own = Int.max touched (submitted d)
      and until = foreign d stamps
      and { maps; depends; _ } = Atomic.get b.links in
      (* Another device's mapping of the memory is unmapped first, once every
         piece of that device's work submitted before the release is done. *)
      let peers = List.filter (fun m -> m.ends = With_memory) maps in
      let until =
        List.fold_left (fun u m -> stamped u m.on (submitted m.on)) until peers
      in
      let until =
        if depends = [] && Option.is_none (stage_of b) then until
        else stamped until d own
      in
      let retire () =
        List.iter (fun m -> unmap m.on b m) peers;
        match List.iter (fun f -> f ()) (List.rev depends) with
        | () -> cache_memory d b ~touched ~released:own
        | exception Failure _ ->
            allocate_bytes d b.kind (-b.bytes);
            retain_bytes d b.kind b.bytes;
            d.held <- Keep b :: d.held
      in
      d.retiring <-
        {
          until = (fun () -> until);
          bytes = b.bytes;
          space = b.kind;
          key = Some (b.bytes, b.kind);
          kept = Keep b;
          retire;
        }
        :: d.retiring

(* An unreachable image of [d] is unloaded once all work [d] submitted is done:
   only [d]'s queues run its code, and a launch may run it without listing its
   memory. The host's programs return before the image can be unreachable. *)
let unload d (i : image) =
  let v = submitted d and bytes = code_bytes i and space = code_kind d i in
  d.retiring <-
    {
      until = (fun () -> [ (d, v) ]);
      bytes;
      space;
      key = None;
      kept = Keep i;
      retire =
        (fun () ->
          driver d i.unload;
          allocate_bytes d space (-bytes));
    }
    :: d.retiring

let rec release_all d = function
  | [] -> ()
  | Memory b :: rs ->
      release d b;
      release_all d rs
  | Code i :: rs ->
      unload d i;
      release_all d rs

let reclaim d =
  release_all d (released d.released);
  retire d;
  release_cache ~wait:false d (Room (Device, 0))

(* Waits for the work of released memory of [key], or of all of it if none is of
   [key], then retires what it can. A lost device's work is not waited for: its
   memory is retained. *)
let wait_retiring d key =
  let some = List.filter (fun r -> r.key = Some key) d.retiring in
  List.iter
    (fun r ->
      List.iter
        (fun (e, v) ->
          if failed e = None then try wait_signal e v with Lost _ -> ())
        (r.until ()))
    (if some = [] then d.retiring else some);
  retire d

let cached_of d kind =
  Hashtbl.fold (fun (_, k) _ c -> c || k = kind) d.cache false

let take_cached d key =
  match Hashtbl.find_opt d.cache key with
  | Some ((e : reusable) :: es) ->
      if es = [] then Hashtbl.remove d.cache key
      else Hashtbl.replace d.cache key es;
      cache_bytes d (snd key) (-fst key);
      Some (e.region, Keep (), e.touched)
  | Some [] | None -> None

(* [n] bytes of [d]'s memory [kind], with the memory they are and the last value
   of [d]'s work that used them, [0] for memory new from the driver. Mapped
   memory that the window and the device's own memory could not hold even with
   the cache released is pinned memory, and keeps the cache, which pinned memory
   does not count in. Mapped memory the driver refuses releases the cached
   mapped memory, which holds the window, and tries again, then is pinned
   memory. Any other allocation its pools or the driver refuse releases the
   cache and tries again, then waits for the work of released memory and tries
   again. One still refused raises [Exhausted] until [last]: the unreachable
   buffers, whose memory the collector cannot see, may hold what it needs (see
   [last_resort_rounds] and [exhausted]). *)
exception Exhausted

let rec allocate d n ~kind ~last =
  if kind <> Mapped then
    on_pools d kind (fun p ->
        if n > p.ceiling then raise (Out_of_memory (d, n)));
  match take_cached d (n, kind) with
  | Some (m, keep, used) -> (m, keep, kind, used)
  | None when kind = Mapped && not (mappable d n) ->
      allocate d n ~kind:Pinned ~last
  | None -> (
      release_cache ~wait:true d (Room (kind, n));
      let alloc n =
        Option.map
          (fun m -> (m, Keep ()))
          ((Option.get (allocator_of d kind)).alloc n)
      in
      match if fits d kind n then driver d (fun () -> alloc n) else None with
      | Some (({ host = None; _ } as m), _)
        when kind <> Device || Option.is_none (queue_of d) ->
          (* The host addresses a [Host_visible] device's memory and pinned and
             mapped memory: a region without a host address is a driver's
             bug. *)
          free_of d kind m;
          invalid_arg
            (Printf.sprintf
               "Nx_device.Driver.device: %s's allocator gave memory the host \
                does not address"
               d.name)
      | Some (m, keep) -> (m, keep, kind, 0)
      | None when kind = Mapped && cached_of d Mapped ->
          release_cache ~only:Mapped ~wait:true d All;
          allocate d n ~kind ~last
      | None when kind = Mapped -> allocate d n ~kind:Pinned ~last
      | None when cached d > 0 ->
          release_cache ~wait:true d All;
          allocate d n ~kind ~last
      | None when d.retiring <> [] ->
          wait_retiring d (n, kind);
          allocate d n ~kind ~last
      | None when not last -> raise Exhausted
      | None -> raise (Out_of_memory (d, n)))

(* Taking devices *)

(* A round of the last resort of an allocation of [d]'s memory (see
   [last_resort_rounds]), run with no device taken. Unreachable buffers may hold
   the memory, which only a complete collection finds. A borrow of it on another
   device holds it until that device's next operation releases the borrow, so
   each device that may map [d]'s memory and is not busy is drained too: one
   whose lock is held is busy, and drains when its operation ends. The next
   round's collection frees what the drained borrows held. *)
let exhausted d =
  Gc.full_major ();
  List.iter
    (fun e ->
      if e != d && reaches e d && Mutex.try_lock e.lock then
        Fun.protect
          ~finally:(fun () -> Mutex.unlock e.lock)
          (fun () -> if failed e = None then try reclaim e with Lost _ -> ()))
    (Atomic.get opened)

let with_devices ds f =
  let ds = List.sort_uniq (fun a b -> Int.compare a.id b.id) ds in
  List.iter (fun d -> Mutex.lock d.lock) ds;
  let rec unlock = function
    | [] -> ()
    | d :: ds ->
        unlock ds;
        Mutex.unlock d.lock
  in
  match
    List.iter check ds;
    List.iter reclaim ds;
    f ()
  with
  | r ->
      unlock ds;
      r
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      unlock ds;
      Printexc.raise_with_backtrace e bt

let synchronize d = with_devices [ d ] (fun () -> sync d)

(* How many collections the last resort of a refused allocation runs, with no
   device taken, trying the allocation again after each. Unreachable buffers may
   hold the memory, which only a complete collection finds, and a buffer that a
   finaliser closure keeps returns only a cycle after that closure runs: a chain
   of such holders takes a cycle per link, and nothing short of the allocation
   succeeding tells that its memory came back. *)
let last_resort_rounds = 4

let set_budget d n =
  if n < 0 then invalid_arg (Printf.sprintf "Nx_device.set_budget: %d < 0" n);
  with_devices [ d ] (fun () ->
      Option.iter (fun p -> p.ceiling <- n) (budgeted d);
      release_cache ~wait:true d (Room (Device, 0)))

let free_cache d = with_devices [ d ] (fun () -> release_cache ~wait:true d All)

(* At exit every device is finalized, a failed one too: its hardware may still
   reach the memory the process is about to release. *)
let () =
  at_exit (fun () ->
      List.iter
        (fun d ->
          let report what e =
            Printf.eprintf "%s %s failed before exiting: %s\n%!" d.name what
              (Printexc.to_string e)
          in
          Mutex.protect d.lock (fun () ->
              (if failed d = None then
                 try sync d with e -> report "synchronization" e);
              try d.finalize ~failed:(failed d <> None)
              with e -> report "finalization" e))
        (Atomic.get opened))

(* Buffers *)

module Buffer = struct
  type device = t
  type nonrec memory = memory = Device | Pinned | Mapped
  type nonrec access = access = Read | Read_write

  type t = buffer = {
    base : base;
    offset : int; (* bytes into [base.memory] *)
    dtype : Nx_dtype.Scalar.t;
    length : int;
    generation : generation; (* its memory's when the buffer was made *)
  }

  (* [n * bitsize s / 8] rounded up, without the product overflowing. *)
  let nbytes_of s n =
    let bits = Nx_dtype.Scalar.bitsize s in
    (n / 8 * bits) + (((n mod 8 * bits) + 7) / 8)

  (* The bytes of [n] elements of [s], for a new buffer or view. *)
  let checked_nbytes fn s n =
    if n < 0 then invalid_arg (Printf.sprintf "Nx_device.%s: %d elements" fn n);
    let size = Nx_dtype.Scalar.bitsize s / 8 in
    if size > 0 && n > max_int / size then
      invalid_arg
        (Printf.sprintf "Nx_device.%s: %d elements of %s overflow" fn n
           (Nx_dtype.Scalar.to_string s));
    nbytes_of s n

  let nbytes b = nbytes_of b.dtype b.length
  let device b = b.base.owner
  let dtype b = b.dtype
  let length b = b.length
  let is_borrowed b = b.base.borrowed
  let ( +! ) a n = Nativeint.add a (Nativeint.of_int n)

  (* Raises unless [b]'s memory was not consumed since [b] was made. *)
  let live b =
    let g = b.base.claim.generation in
    if b.generation != g then invalid_arg g.why

  let address b =
    live b;
    b.base.memory.address +! b.offset

  (* [b]'s address in the address space of its machine's host, if that host
     addresses it. *)
  let hosted b = Option.map (fun a -> a +! b.offset) b.base.memory.host
  let local b = host_of b.base.owner == host

  (* Raises [Lost] for a lost device that can reach [b]'s memory. *)
  let reachable b =
    live b;
    check_reach b.base

  (* No byte of it is ever read or written, so the host addresses it. *)
  let no_memory = { host = Some 0n; address = 0n; handle = 0n; nbytes = 0 }

  (* A base over [memory], whose claims are its source's for a borrow, [claim]
     when given, and new otherwise, held by whoever holds the bigarray or the
     file when [exported]. *)
  let base ?(bytes = 0) ?(kind = Device) ?source ?file ?claim
      ?(exported = false) ~borrowed ~keep d memory =
    let claim =
      match (source, claim) with
      | Some (src, _), _ -> src.claim
      | None, Some c -> c
      | None, None ->
          let claims = if exported then 1 else 0 in
          { claims; generation = first_generation }
    in
    {
      owner = d;
      memory;
      bytes;
      kind;
      borrowed;
      keep;
      source;
      links = Atomic.make no_links;
      file;
      claim;
    }

  (* The buffer of [n] elements of [s] at the start of [base]'s memory. *)
  let first base s n =
    let generation = base.claim.generation in
    { base; offset = 0; dtype = s; length = n; generation }

  let empty ~borrowed d s n =
    first (base ~borrowed ~keep:(Keep ()) d no_memory) s n

  (* Host memory takes neither the host nor its release list: its bytes are
     reserved atomically against the budget and returned by the finaliser of a
     token the base keeps, which the collector runs where it frees the token, in
     the collection that finds the base unreachable. No wait is needed: a
     device's work reaches host memory only through a borrow, which keeps it
     alive. *)

  (* [n] reserved bytes of the heap. A refused reservation or allocation
     collects garbage once and tries again. *)
  let rec host_heap n ~round =
    if heap_reserve n (budget host) then (
      match heap n with
      | ba -> ba
      | exception Stdlib.Out_of_memory ->
          heap_return n;
          host_refused n ~round)
    else host_refused n ~round

  and host_refused n ~round =
    if round = last_resort_rounds then raise (Out_of_memory (host, n));
    exhausted host;
    host_heap n ~round:(round + 1)

  let not_files fn =
    invalid_arg
      (Printf.sprintf
         "Nx_device.%s: DISK buffers are files: open one with Buffer.of_file \
          or Buffer.create_file"
         fn)

  let allocated ?stage ~memory d s n =
    if d == disk then not_files "Buffer.create";
    match checked_nbytes "Buffer.create" s n with
    | 0 ->
        check d;
        empty ~borrowed:false d s n
    | bytes when d == host ->
        check host;
        if bytes > budget host then raise (Out_of_memory (host, bytes));
        let ba = host_heap bytes ~round:0 in
        let base =
          base ~bytes ~borrowed:false
            ~keep:(Heap (ba, heap_token bytes))
            host (heap_memory ba)
        in
        (* While a profile is taken, the return of the bytes is recorded too. *)
        if Option.is_some (Atomic.get profiles) then begin
          memory_changed host;
          Gc.finalise_last (fun () -> memory_changed host) base
        end;
        first base s n
    | bytes ->
        let kind =
          match (d.kind, memory) with
          | Local { mapped = None; _ }, Mapped -> Pinned
          | Local _, kind -> kind
          | (Machine _ | Shared _ | Disk), _ -> Device
        in
        let rec take round =
          match
            with_devices [ d ] (fun () ->
                let last = round = last_resort_rounds in
                let ((_, _, given, _) as m) = allocate d bytes ~kind ~last in
                allocate_bytes d given bytes;
                memory_changed d;
                m)
          with
          | m -> m
          | exception Exhausted ->
              exhausted d;
              take (round + 1)
        in
        let memory, keep, kind, used = take 0 in
        let keep =
          match stage with Some st -> Staged (keep, st) | None -> keep
        in
        let base = base ~bytes ~kind ~borrowed:false ~keep d memory in
        let base = owned d base in
        (* Cached memory may still be read by [d]'s work, which a write of the
           host to it, such as a staged buffer's copy, waits for. *)
        if used > 0 then stamp_use base d used;
        first base s n

  let create ?(memory = Device) d s n = allocated ~memory d s n

  (* The bytes an element of [k] is aligned to: one component's for the complex
     kinds. *)
  let component_size (type a b) (k : (a, b) Bigarray.kind) =
    match k with
    | Bigarray.Complex32 -> 4
    | Bigarray.Complex64 -> 8
    | k -> Bigarray.kind_size_in_bytes k

  let of_bigarray ba =
    let dtype =
      match Nx_dtype.Scalar.of_bigarray_kind (Bigarray.Array1.kind ba) with
      | Some s -> s
      | None ->
          invalid_arg
            "Nx_device.Buffer.of_bigarray: the kind is no storage format"
    in
    (* A host buffer's elements lie at multiples of their size, as every typed
       read of it expects. *)
    let align = component_size (Bigarray.Array1.kind ba) in
    let address = bigarray_address ba in
    if
      Bigarray.Array1.size_in_bytes ba > 0
      && Nativeint.rem address (Nativeint.of_int align) <> 0n
    then
      invalid_arg
        (Printf.sprintf
           "Nx_device.Buffer.of_bigarray: the elements at 0x%nx are not \
            aligned to %d bytes"
           address align);
    let ba = shared ba in
    (* Whoever holds the bigarray reaches the memory outside the claims. *)
    let base =
      base ~exported:true ~borrowed:true ~keep:(Host ba) host (heap_memory ba)
    in
    first base dtype (Bigarray.Array1.dim ba)

  let refuse fmt = Printf.ksprintf (fun why -> Error why) fmt

  (* A file is opened with the disk taken, into its descriptor cache. *)
  let open_file path ~create n =
    let mode = if create then create_mode else read_mode in
    let opened () =
      match open_path path mode n with
      | 0, fd, size -> (
          match identify path fd with
          | identity ->
              let f =
                {
                  path;
                  writable = create;
                  size;
                  identity;
                  fd = None;
                  used = 0;
                  pages = None;
                }
              in
              cache f fd;
              Ok f
          | exception Sys_error why -> Error why)
      | code, _, _ when code = not_regular ->
          refuse "%s: not a regular file" path
      | code, _, _ when code = too_many -> refuse "%s: too many open files" path
      | code, _, _ -> refuse "%s: %s" path (error_message code)
    in
    Result.map
      (fun f ->
        let memory =
          { host = None; address = 0n; handle = 0n; nbytes = f.size }
        in
        (* The file is a holder outside the claims. *)
        let base =
          base ~exported:true ~borrowed:true ~keep:(Keep ()) ~file:f disk memory
        in
        let base = owned disk base in
        first base Nx_dtype.Scalar.UInt8 f.size)
      (with_devices [ disk ] opened)

  let of_file path = open_file path ~create:false 0

  let create_file path n =
    if n < 0 then
      invalid_arg (Printf.sprintf "Nx_device.Buffer.create_file: %d bytes" n);
    open_file path ~create:true n

  let file_of b = Option.get b.base.file

  (* A file's pages are its mapping on the host, made once and kept with the
     file. *)
  let pages b =
    let f = file_of b in
    with_devices [ disk ] (fun () ->
        match f.pages with
        | Some base -> Ok base
        | None -> (
            match file_map (descriptor f) f.size with
            | exception Sys_error why -> Error why
            | 0, ba ->
                let base =
                  base ~claim:b.base.claim ~borrowed:true ~keep:(Host ba) host
                    (heap_memory ba)
                in
                f.pages <- Some base;
                Ok base
            | code, _ -> refuse "%s: %s" f.path (error_message code)))

  (* [d]'s mapping of the memory of [src], made by its first borrow with [map]
     and shared by the later ones, and [skip] bytes into it. *)
  let share_taken d src ~ends ~map =
    Result.map
      (fun m ->
        m.borrows <- m.borrows + 1;
        m)
      (map_on d src ~ends ~map)

  let share d src ~ends ~map =
    with_devices [ d ] (fun () -> share_taken d src ~ends ~map)

  (* A buffer over the memory of [b] that [m] maps on [d]. *)
  let borrowed d b m =
    let base =
      base ~source:(b.base, m) ~borrowed:true ~keep:(Keep b) d m.mapped
    in
    let base = owned d base in
    { b with base; offset = m.skip + b.offset }

  (* A device whose mapping is the identity addresses host memory at its host
     addresses: its borrow is over the memory itself, with no page to lock and
     no driver to call. It keeps [b], whose memory is its source, so that the
     memory's device reaches it. On a host, a bigarray that owns nothing makes
     the memory the host's buffer. *)
  let identity d b =
    let src = b.base in
    let a = Option.get src.memory.host in
    let region = { src.memory with address = a } in
    let keep =
      if d == host then Addressed (external_bytes a region.nbytes, b)
      else Keep b
    in
    let mapped =
      {
        on = d;
        mapped = region;
        skip = 0;
        unmap = ignore;
        ends = With_borrows;
        borrows = 1;
        work = [];
        unmapping = false;
      }
    in
    let base = base ~source:(src, mapped) ~borrowed:true ~keep d region in
    (* A device other than a host may run work on the memory after [submit]
       returns, as test devices whose queues run behind the host do: its release
       waits for that work, and keeps [b] until then. A host's work is the calls
       it makes, which return once done. *)
    let base = if d == host then base else owned d base in
    { b with base }

  (* A device maps the whole host memory under [b], once, and its borrows share
     the mapping. A mapping locks whole pages, so it starts on one: host memory
     of another buffer then never shares its pages. *)
  let borrow_host ?(share = share) d b =
    let h = host_of d and src = b.base in
    let first = Option.get src.memory.host in
    match mapping_of d with
    | _ when d == h && src.owner == h -> Ok b
    | None -> refuse "%s cannot address host memory" d.name
    | Some Identity -> Ok (identity d b)
    (* A host buffer nx made starts on a page from [aligned_from] bytes; a
       smaller one is refused wherever it happens to start, so that whether it
       borrows does not depend on the allocator. *)
    | Some (Pages _)
      when src.owner == host && (not src.borrowed)
           && src.memory.nbytes < aligned_from ->
        refuse
          "a host buffer of %d bytes does not start on a page, and %s maps \
           whole pages; host buffers start on one from %d bytes"
          src.memory.nbytes d.name aligned_from
    (* Another machine's pages are its own; its devices check them. *)
    | Some (Pages _)
      when h == host && Nativeint.rem first (Nativeint.of_int page) <> 0n ->
        refuse
          "the host memory at 0x%nx does not start on a page, and %s maps \
           whole pages; host buffers start on one from %d bytes"
          first d.name aligned_from
    | Some (Pages { map; unmap }) -> (
        let map () =
          Result.map
            (fun (mapped : region) ->
              ( mapped,
                Nativeint.to_int (Nativeint.sub first (Option.get mapped.host)),
                unmap ))
            (map first src.memory.nbytes)
        in
        match share d src ~ends:With_borrows ~map with
        | Error why ->
            refuse "%s cannot map the host memory at 0x%nx: %s" d.name first why
        | Ok m -> Ok (borrowed d b m))

  (* Another device's memory is mapped by [d]'s driver until the memory is
     released (see [peer_map]). *)
  let borrow_peer d b =
    let src = b.base in
    match d.peer with
    | None -> refuse "%s cannot address %s memory" d.name src.owner.name
    | Some peer -> (
        match share d src ~ends:With_memory ~map:(peer_map peer src) with
        | Error why ->
            refuse "%s cannot map %s memory: %s" d.name src.owner.name why
        | Ok m -> Ok (borrowed d b m))

  (* A file's bytes are borrowed from its pages, by devices that share the
     host's memory. *)
  let borrow_file d b =
    let f = file_of b in
    let size = Int.max 1 (Nx_dtype.Scalar.bitsize b.dtype / 8) in
    if not (shares_host_memory d) then
      refuse "%s does not share the host's memory: copy %s's bytes to it" d.name
        f.path
    else if b.offset mod size <> 0 then
      refuse "byte %d of %s is not aligned to %s's %d bytes" b.offset f.path
        (Nx_dtype.Scalar.to_string b.dtype)
        size
    else
      Result.bind (pages b) (fun pages ->
          (* A device faulting a file's pages in reads a few times slower than
             the disk; the host reads them as it needs them. *)
          if d != host then
            with_devices [ disk ] (fun () ->
                (* Advice is a hint: a file that cannot be reopened gets none,
                   and its pages, mapped already, stay valid. *)
                match descriptor f with
                | fd -> file_advise fd b.offset (nbytes b)
                | exception Sys_error _ -> ());
          borrow_host d { b with base = pages })

  (* [b] over the memory its borrows map, down to memory no borrow holds. *)
  let rec root b =
    match b.base.source with
    | Some (src, m) -> root { b with base = src; offset = b.offset - m.skip }
    | None -> b

  let spans b =
    let r = root b in
    r.offset = 0 && nbytes r = r.base.memory.nbytes

  (* Memory is borrowed by where it lives: the disk's through the file's pages,
     system memory (the host's, a [Host_visible] device's, and any device's
     pinned memory) through [d]'s mapping of host memory, and the memory of a
     [Device_local] device, its mapped memory included, through [d]'s peer
     mapping, even where a BAR gives it a host address. *)
  let borrow d b =
    live b;
    check d;
    if b.base.owner == d then Ok b
    else
      let r = root b in
      let o = r.base.owner in
      if o != disk && host_of o != host_of d then
        invalid_arg
          (Printf.sprintf
             "Nx_device.Buffer.borrow: the buffer is on %s, not on %s's machine"
             o.name d.name)
      else if o == d then Ok r
      else if nbytes b = 0 then Ok (empty ~borrowed:true d b.dtype b.length)
      else if o == disk then borrow_file d r
      else if o == host_of d then borrow_host d r
      else if
        r.base.kind = Pinned
        || (r.base.kind = Mapped && d == host_of o)
        || Option.is_none (queue_of o)
      then
        if Option.is_none d.machine && d != host then
          refuse "%s is reached over the network, and maps no memory" d.name
        else borrow_host d r
      else borrow_peer d r

  let is_staged b = Option.is_some (stage_of b.base)
  let stage_ids = Atomic.make 0

  (* Memory of this machine's host that [d] does not map is staged: under
     [aligned_from] bytes, it is the memory nx puts off a page, which no device
     that maps whole pages maps. More would be copied on every submission and
     held twice, so it is refused. *)
  let reach d b access =
    match borrow d b with
    | Ok r -> Ok r
    | Error why ->
        let o = (root b).base.owner in
        if host_of d != host || not (shares_host_memory o) then Error why
        else if nbytes b >= aligned_from then
          Error
            (Printf.sprintf "%d bytes of host memory %s does not map: %s"
               (nbytes b) d.name why)
        else
          Ok
            (allocated
               ~stage:
                 {
                   original = b;
                   access;
                   id = Atomic.fetch_and_add stage_ids 1;
                   busy = Mutex.create ();
                 }
               ~memory:Pinned d b.dtype b.length)

  let view b ~offset s n =
    let fail fmt =
      Printf.ksprintf (fun m -> invalid_arg ("Nx_device.Buffer.view: " ^ m)) fmt
    in
    if offset < 0 then fail "negative offset %d" offset;
    let bytes = checked_nbytes "Buffer.view" s n in
    if offset > nbytes b - bytes then
      fail "%d bytes at offset %d do not fit in %d bytes" bytes offset
        (nbytes b);
    (* A file's bytes are read and written at any offset. *)
    let size = Int.max 1 (Nx_dtype.Scalar.bitsize s / 8) in
    if
      b.base.owner != disk
      && Nativeint.rem (address b +! offset) (Nativeint.of_int size) <> 0n
    then
      fail "offset %d is not aligned to %s's %d bytes" offset
        (Nx_dtype.Scalar.to_string s)
        size;
    { b with offset = b.offset + offset; dtype = s; length = n }

  (* The address space memory is addressed in: its machine's host's when the
     host addresses it, its device's otherwise, and its file's on the disk. *)
  type space = In_host of int | In_device of int | In_file of int * int

  (* Where [b]'s bytes lie, below its borrows: a space and the first byte. *)
  let place b =
    let r = root b in
    let m = r.base.memory and o = r.base.owner in
    if o == disk then
      let i = (file_of r).identity in
      (In_file (i.dev, i.ino), Nativeint.of_int r.offset)
    else
      match m.host with
      | Some a -> (In_host (host_of o).id, a +! r.offset)
      | None -> (In_device o.id, m.address +! r.offset)

  let overlaps b b' =
    let n = nbytes b and n' = nbytes b' in
    n > 0 && n' > 0
    &&
    let s, a = place b and s', a' = place b' in
    s = s'
    && Nativeint.compare a (a' +! n') < 0
    && Nativeint.compare a' (a +! n) < 0

  module Claim = struct
    let busy () =
      invalid_arg
        "Nx_device.Buffer.Claim: the memory is in use by a consuming call"

    let unbalanced () =
      invalid_arg "Nx_device.Buffer.Claim.release: unbalanced claim"

    (* Claims count on the memory's record, so a borrow and the memory it maps
       share them. Each loop retries only a CAS that another domain's claim
       beat. *)

    let rec read_claim m =
      let n = m.claims in
      if n < 0 then busy ()
      else if not (Atomic.Loc.compare_and_set [%atomic.loc m.claims] n (n + 1))
      then read_claim m

    let rec release_claim m =
      let n = m.claims in
      if n <= 0 then unbalanced ()
      else if not (Atomic.Loc.compare_and_set [%atomic.loc m.claims] n (n - 1))
      then release_claim m

    let exclusive_claim m =
      Atomic.Loc.compare_and_set [%atomic.loc m.claims] 1 (-1)

    let finish_claim m =
      if not (Atomic.Loc.compare_and_set [%atomic.loc m.claims] (-1) 1) then
        invalid_arg "Nx_device.Buffer.Claim.finish: the memory is not exclusive"

    let read b =
      reachable b;
      read_claim b.base.claim

    let release b = release_claim b.base.claim
    let try_exclusive b = exclusive_claim b.base.claim
    let finish b = finish_claim b.base.claim
    let export = read

    (* The memories a bracket reads, with repeats, and those it holds
       exclusive. *)
    type t = { reads : claim list; exclusive : claim list }

    let exclusive c b = List.memq b.base.claim c.exclusive

    let consume c ~why b =
      let m = b.base.claim in
      if not (List.memq m c.reads) then
        invalid_arg "Nx_device.Buffer.Claim.consume: the buffer is not claimed";
      live b;
      if not (spans b) then
        invalid_arg
          "Nx_device.Buffer.Claim.consume: the buffer is a window of its memory";
      let generation = { why } in
      m.generation <- generation;
      { b with generation }

    (* Raises if a buffer of [donated] shares a byte with another of [rs] or
       [donated]: the bytes sorted by where they lie, each is checked against
       the furthest end reached before it, and against the furthest a donated
       one reached. *)
    let refuse_overlaps rs donated =
      let bytes donated b =
        let s, a = place b in
        (s, a, a +! nbytes b, donated)
      in
      let some = List.filter (fun b -> nbytes b > 0) in
      let all =
        List.map (bytes false) (some rs) @ List.map (bytes true) (some donated)
      in
      let all =
        List.sort
          (fun (s, a, _, _) (s', a', _, _) ->
            match Stdlib.compare s s' with
            | 0 -> Nativeint.compare a a'
            | c -> c)
          all
      in
      let refuse () =
        invalid_arg
          "Nx_device.Buffer.Claim.with_: a donated buffer overlaps another \
           buffer of the call"
      in
      let further a b = if Nativeint.compare a b < 0 then b else a in
      let rec sweep space reached by_donated = function
        | [] -> ()
        | (s, a, e, donated) :: rest ->
            let reached, by_donated =
              if s = space then (reached, by_donated) else (a, a)
            in
            if Nativeint.compare a (if donated then reached else by_donated) < 0
            then refuse ();
            sweep s (further reached e)
              (if donated then further by_donated e else by_donated)
              rest
      in
      match all with [] -> () | (s, a, _, _) :: _ -> sweep s a a all

    let with_ ~read:rs ~donate f =
      let donated = List.concat donate in
      refuse_overlaps rs donated;
      (* Reads in order, releasing those taken if one is refused. *)
      let reads =
        List.fold_left
          (fun taken b ->
            match read b with
            | () -> b.base.claim :: taken
            | exception e ->
                List.iter release_claim taken;
                raise e)
          [] (rs @ donated)
      in
      (* A value is exclusive if each of its shards is; a window is never. *)
      let exclusive =
        List.concat_map
          (fun shards ->
            let claims = List.map (fun b -> b.base.claim) shards in
            if not (List.for_all spans shards) then []
            else
              let rec upgrade taken = function
                | [] -> claims
                | m :: rest ->
                    if exclusive_claim m then upgrade (m :: taken) rest
                    else begin
                      List.iter finish_claim taken;
                      []
                    end
              in
              upgrade [] claims)
          donate
      in
      let release () =
        List.iter finish_claim exclusive;
        List.iter release_claim reads
      in
      match f { reads; exclusive } with
      | v ->
          release ();
          v
      | exception e ->
          let bt = Printexc.get_raw_backtrace () in
          release ();
          Printexc.raise_with_backtrace e bt
  end

  let bigarray (type a b) (k : (a, b) Bigarray.kind) buf :
      (a, b, Bigarray.c_layout) Bigarray.Array1.t =
    let fail fmt =
      Printf.ksprintf
        (fun m -> invalid_arg ("Nx_device.Buffer.bigarray: " ^ m))
        fmt
    in
    if Nx_dtype.Scalar.of_bigarray_kind k = None then
      fail "the kind is no storage format";
    if not (buf.base.owner == host) then
      fail "the buffer is on %s, not CPU" buf.base.owner.name;
    reachable buf;
    let size = Bigarray.kind_size_in_bytes k and bytes = nbytes buf in
    let align = component_size k in
    if bytes mod size <> 0 then
      fail "%d bytes are not a whole number of %d-byte elements" bytes size;
    if Nativeint.rem (address buf) (Nativeint.of_int align) <> 0n then
      fail "the buffer is not aligned to %d bytes" align;
    if bytes = 0 then Bigarray.Array1.create k Bigarray.c_layout 0
    else
      let rec view = function
        | Host ba -> bigarray_view ba k buf.offset (bytes / size)
        | Heap (ba, _) -> bigarray_view ba k buf.offset (bytes / size)
        | Addressed (ba, _) -> bigarray_view ba k buf.offset (bytes / size)
        | With (keep, _) | Staged (keep, _) -> view keep
        | Keep _ -> assert false (* host memory is always a bigarray's *)
      in
      view buf.base.keep

  (* Copies. The devices involved are taken and synchronized. A device's copy is
     work on its timeline, waited for at once. *)

  (* Bytes per slot of a host's staging memory, which has two. *)
  let chunk = 64 lsl 20

  (* The host [h]'s staging memory, made at its first use and kept for the life
     of the process: this machine's on the heap, another machine's in that
     machine's memory. It is used with [h] taken, as are its slots: a host fill
     of one first waits for the work of every device that used the memory (see
     [copy] and [staging_on]). *)
  let rec staging h =
    match h.slots with
    | None -> fresh_staging h
    | Some b -> (
        match check_reach b.base with
        | () -> b
        | exception Lost _ ->
            (* A lost device may still write it: it is retained, and the host
               stages through new memory. *)
            h.held <- Keep b :: h.held;
            fresh_staging h)

  and fresh_staging h =
    let base =
      if h == host then
        match heap (2 * chunk) with
        | ba -> base ~borrowed:false ~keep:(Host ba) host (heap_memory ba)
        | exception Stdlib.Out_of_memory ->
            raise (Out_of_memory (host, 2 * chunk))
      else
        match
          driver h (fun () ->
              (Option.get (allocator_of h Device)).alloc (2 * chunk))
        with
        | Some m -> base ~borrowed:false ~keep:(Keep ()) h m
        | None -> raise (Out_of_memory (h, 2 * chunk))
    in
    let b = first base Nx_dtype.Scalar.UInt8 (2 * chunk) in
    h.slots <- Some b;
    b

  (* Waits for the work of every device that used the host [h]'s staging memory,
     such as a compiled batch's staged copies, before a copy uses its slots. A
     device lost meanwhile is not the copy's to raise: the copy's next use of
     the memory retains it and stages through new memory (see [staging]). *)
  let staging_done h =
    match h.slots with
    | None -> ()
    | Some _ ->
        List.iter
          (fun s -> try wait_signal s.by s.upto with Lost _ -> ())
          (Atomic.get (staging h).base.links).stamps

  (* The address of the host [h]'s staging memory, in [h]'s address space. *)
  let slots h = Option.get (hosted (staging h))

  (* [e]'s address of its host's staging memory, through its borrow of it, which
     [e] makes at its first staged copy, with [e] taken, and keeps. Work that
     touches the borrow, such as a compiled batch's staged copies, stamps the
     staging memory. *)
  let staging_on e =
    let h = host_of e in
    let current = staging h in
    match e.staging with
    | Some ({ base = { source = Some (src, _); _ }; _ } as b)
      when src == current.base ->
        address b
    | Some _ | None -> (
        match borrow_host ~share:share_taken e current with
        | Ok b ->
            e.staging <- Some b;
            address b
        | Error why ->
            failwith
              (Printf.sprintf "%s cannot map %s's staging memory: %s" e.name
                 h.name why))

  let queue e =
    match queue_of e with
    | Some q -> q
    | None ->
        invalid_arg
          (Printf.sprintf "Nx_device.Buffer.copy: %s has no copy queue" e.name)

  (* The memory a host addresses. Another machine's is reached through the
     host's [io], whose error fails that host. *)
  let io_call h f = try f () with Failure msg -> fail h msg

  (* [n] bytes within the memory the host [h] addresses. *)
  let host_move h ~dst ~src n =
    match io_of h with
    | None -> memmove dst src n
    | Some io -> io_call h (fun () -> io.copy ~dst ~src n)

  (* Enqueues [e]'s copy [f] of one chunk of a copy, and is the value to wait
     for. A [timed] copy stamps the start of its first chunk and the stop of its
     last into [e]'s timestamp slots. *)
  let enqueue_copy ~timed ~first ~last e q f =
    if timed && first then ignore (enqueue e (q.stamp ~slot:(stamp_slot e 0)));
    let v = enqueue e f in
    if timed && last then enqueue e (q.stamp ~slot:(stamp_slot e 1)) else v

  let run ~timed e q f =
    wait_signal e (enqueue_copy ~timed ~first:true ~last:true e q f)

  let chunks n = (n + chunk - 1) / chunk
  let length_of n i = Int.min chunk (n - (i * chunk))
  let slot i = i land 1 * chunk

  (* [e]'s copy on its queue [q] of chunk [i] of a copy of [n] bytes, from its
     [src] to its [dst], and the value to wait for. *)
  let copy_chunk ~timed e q n i ~dst ~src =
    enqueue_copy ~timed ~first:(i = 0)
      ~last:(i = chunks n - 1)
      e q
      (q.copy ~dst ~src (length_of n i))

  (* Runs [f], the host's side of a copy of [e]: if it raises, [e]'s copies are
     waited for first, so that none outlives the call. *)
  let settled e f =
    match f () with
    | () -> ()
    | exception ex ->
        wait_signal e (submitted e);
        raise ex

  (* The host fills one slot while [e] copies the other: [fill a pos len] puts
     the [len] bytes of the source from its byte [pos] at [a], in the memory of
     [e]'s host. *)
  let stage_in ~timed e q ~fill ~dst n =
    let at = slots (host_of e) and on_e = staging_on e in
    let last = [| 0; 0 |] in
    for i = 0 to chunks n - 1 do
      wait_signal e last.(i land 1);
      settled e (fun () -> fill (at +! slot i) (i * chunk) (length_of n i));
      last.(i land 1) <-
        copy_chunk ~timed e q n i ~dst:(dst +! (i * chunk)) ~src:(on_e +! slot i)
    done;
    wait_signal e (submitted e)

  (* [e] fills one slot while the host drains the other: [drain a pos len] puts
     the [len] bytes at [a] into the destination from its byte [pos]. *)
  let stage_out ~timed e q ~src ~drain n =
    let at = slots (host_of e) and on_e = staging_on e in
    let last = [| 0; 0 |] in
    let fill i =
      if i < chunks n then
        last.(i land 1) <-
          copy_chunk ~timed e q n i
            ~dst:(on_e +! slot i)
            ~src:(src +! (i * chunk))
    in
    fill 0;
    fill 1;
    for i = 0 to chunks n - 1 do
      wait_signal e last.(i land 1);
      settled e (fun () -> drain (at +! slot i) (i * chunk) (length_of n i));
      fill (i + 2)
    done

  (* Between devices that cannot reach each other, the bytes go through their
     host's staging memory: [s] fills one slot while [d] drains the other. *)
  let bounce ~timed s ~src d ~dst n =
    let qs = queue s and qd = queue d in
    let on_s = staging_on s and on_d = staging_on d in
    let last = [| 0; 0 |] in
    for i = 0 to chunks n - 1 do
      wait_signal d last.(i land 1);
      wait_signal s
        (copy_chunk ~timed s qs n i
           ~dst:(on_s +! slot i)
           ~src:(address src +! (i * chunk)));
      last.(i land 1) <-
        copy_chunk ~timed d qd n i
          ~dst:(address dst +! (i * chunk))
          ~src:(on_d +! slot i)
    done;
    wait_signal d (submitted d)

  (* Runs [f] with [b]'s address for [e]'s work, if [e] addresses [b]'s memory:
     its own, memory it maps, or pinned memory of another device, which [e] maps
     for the copy alone. If [f] raises, [e] may still use that memory: it stays
     mapped and in [e]'s reach. *)
  let with_address e b f =
    if b.base.owner == e then f (Some (address b))
    else
      match mapping_on e b.base with
      | Some m -> f (Some (m.mapped.address +! (m.skip + b.offset)))
      | None when b.base.kind = Pinned -> (
          let first = Option.get b.base.memory.host in
          match Option.get (mapping_of e) with
          | Identity -> f (hosted b)
          | Pages { map; unmap } -> (
              match driver e (fun () -> map first b.base.memory.nbytes) with
              | Error _ -> f None
              | Ok m -> (
                  match f (Some (mapped_address m (Option.get (hosted b)))) with
                  | r -> (
                      match unmap m with
                      | () -> r
                      | exception Failure why ->
                          lose e why;
                          poison b.base e;
                          fail e why)
                  | exception e' ->
                      poison b.base e;
                      raise e')))
      | None -> f None

  (* [e] copies [src], which its host addresses, into [dst], its memory. *)
  let into ~timed e ~src ~dst n =
    let q = queue e and h = host_of e in
    with_address e src (function
      | Some a -> run ~timed e q (q.copy ~dst:(address dst) ~src:a n)
      | None ->
          let src = Option.get (hosted src) in
          stage_in ~timed e q
            ~fill:(fun a pos len -> host_move h ~dst:a ~src:(src +! pos) len)
            ~dst:(address dst) n)

  (* [e] copies [src], its memory, into [dst], which its host addresses or [e]
     does. *)
  let out_of ~timed e ~src ~dst n =
    let q = queue e and h = host_of e in
    with_address e dst (function
      | Some a -> run ~timed e q (q.copy ~dst:a ~src:(address src) n)
      | None ->
          let dst = Option.get (hosted dst) in
          stage_out ~timed e q ~src:(address src)
            ~drain:(fun a pos len -> host_move h ~dst:(dst +! pos) ~src:a len)
            n)

  (* Between machines, chunk by chunk: the chunk reaches memory the source's
     host addresses ([b] itself, or a staging slot its device fills), crosses
     into this process and out to memory the destination's host addresses, which
     the destination's device copies from when it is not [dst] itself. *)
  let on_host ~timed b n i =
    match hosted b with
    | Some a -> a +! (i * chunk)
    | None ->
        let e = device b in
        wait_signal e
          (copy_chunk ~timed e (queue e) n i
             ~dst:(staging_on e +! slot i)
             ~src:(address b +! (i * chunk)));
        slots (host_of e) +! slot i

  let landing b i =
    match hosted b with
    | Some a -> a +! (i * chunk)
    | None -> slots (host_of (device b)) +! slot i

  let from_host ~timed b n i =
    if hosted b = None then
      let e = device b in
      wait_signal e
        (copy_chunk ~timed e (queue e) n i
           ~dst:(address b +! (i * chunk))
           ~src:(staging_on e +! slot i))

  let across ~timed ~src ~dst n =
    let hs = host_of (device src) and hd = host_of (device dst) in
    for i = 0 to chunks n - 1 do
      let len = length_of n i in
      let a = on_host ~timed src n i and b = landing dst i in
      (match (io_of hs, io_of hd) with
      | None, None -> memmove b a len
      | None, Some io -> io_call hd (fun () -> io.write ~dst:b ~src:a len)
      | Some io, None -> io_call hs (fun () -> io.read ~src:a ~dst:b len)
      | Some io, Some io' ->
          let relay = slots host +! slot i in
          io_call hs (fun () -> io.read ~src:a ~dst:relay len);
          io_call hd (fun () -> io'.write ~dst:b ~src:relay len));
      from_host ~timed dst n i
    done

  (* [read b ~pos a n] reads the [n] bytes of the disk buffer [b] from its byte
     [pos] into host memory at [a]. *)
  let read b ~pos a n =
    let f = file_of b and at = b.offset + pos in
    match file_read (descriptor f) at a n with
    | k when k = n -> ()
    | k when k >= 0 ->
        raise
          (Sys_error
             (Printf.sprintf "%s: the file ends at byte %d, before byte %d"
                f.path (at + k) (at + n)))
    | code -> raise (Sys_error (f.path ^ ": " ^ error_message (-code)))

  (* [write b ~pos a n] writes the [n] bytes of host memory at [a] into the disk
     buffer [b] from its byte [pos]. *)
  let write b ~pos a n =
    let f = file_of b in
    let fd = descriptor f in
    let code = file_write fd (b.offset + pos) a n in
    if code < 0 then raise (Sys_error (f.path ^ ": " ^ error_message (-code)));
    (* The write changed the file: its later reopens must not take it for
       another one. *)
    match file_identity fd with
    | 0, dev, ino, changed -> f.identity <- { dev; ino; changed }
    | _ -> ()

  (* A copy from or to the disk reads or writes its file: straight from or into
     memory the host addresses, and through the host's staging memory otherwise,
     which the device of the other buffer copies to or from. *)
  let file_copy ~timed ~src ~dst n =
    match (device src == disk, device dst == disk) with
    | true, true ->
        let a = slots host in
        for i = 0 to chunks n - 1 do
          read src ~pos:(i * chunk) a (length_of n i);
          write dst ~pos:(i * chunk) a (length_of n i)
        done
    | true, false -> (
        match hosted dst with
        | Some a -> read src ~pos:0 a n
        | None ->
            let e = device dst in
            stage_in ~timed e (queue e)
              ~fill:(fun a pos len -> read src ~pos a len)
              ~dst:(address dst) n)
    | false, _ -> (
        match hosted src with
        | Some a -> write dst ~pos:0 a n
        | None ->
            let e = device src in
            stage_out ~timed e (queue e) ~src:(address src)
              ~drain:(fun a pos len -> write dst ~pos a len)
              n)

  (* The first device with a link that carries a copy of [src] into [dst]. *)
  let link_for ~src ~dst =
    List.find_map
      (fun d ->
        match d.link with
        | Some link when failed d = None -> link ~src ~dst
        | _ -> None)
      (Atomic.get opened)

  type route =
    | Host_copy
    | Into
    | Out_of
    | Transfer of copy * nativeint (* and the destination's address *)
    | Bounce
    | Across
    | Link of link
    | File of device option
  (* from or to the disk, staged by the device whose copy queue it names *)

  (* Where [e]'s work addresses [b], another device's memory: through [e]'s
     mapping of the memory under [b], which lasts until that memory is released,
     or at [b]'s own address on a device that maps no other device's memory.
     [None] if [e]'s driver refuses the mapping. *)
  let address_on e b =
    match e.peer with
    | None -> Some (address b)
    | Some peer -> (
        let r = root b in
        let map = peer_map peer r.base in
        match
          with_devices [ e ] (fun () -> map_on e r.base ~ends:With_memory ~map)
        with
        | Ok m ->
            Some
              (Nativeint.add m.mapped.address
                 (Nativeint.of_int (m.skip + r.offset)))
        | Error _ -> None)

  (* The route of a copy of [src] into [dst], and the devices it takes besides
     the two: the hosts whose staging memory or [io] it uses, a link's devices.
     A transfer the source cannot map the destination for goes through the
     host. *)
  let route ~src ~dst =
    let s = device src and d = device dst in
    let h = host_of s in
    let remote = if Option.is_some (io_of h) then [ h ] else [] in
    if s == disk || d == disk then
      let other = if s == disk then dst else src in
      let e = device other in
      if e == disk then (File None, [ host ])
      else if hosted other = None then (File (Some e), [ host ])
      else (File None, [])
    else if h != host_of d then
      match link_for ~src ~dst with
      | Some l -> (Link l, l.through)
      | None -> (Across, [ h; host_of d; host ])
    else
      match (hosted src, hosted dst) with
      | Some _, Some _ -> (Host_copy, remote)
      | Some _, None -> (Into, if src.base.owner != d then [ h ] else [])
      | None, Some _ -> (Out_of, if dst.base.owner != s then [ h ] else [])
      | None, None when s == d -> (Out_of, [])
      | None, None -> (
          let transfer =
            Option.bind ((queue s).transfer d) (fun copy ->
                Option.map (fun at -> (copy, at)) (address_on s dst))
          in
          match transfer with
          | Some (copy, at) -> (Transfer (copy, at), [])
          | None -> (Bounce, [ h ]))

  (* The devices whose copy queues a copy of [route] runs on. *)
  let queues route ~src ~dst =
    let s = device src and d = device dst in
    match route with
    | Host_copy | Link _ -> []
    | Into -> [ d ]
    | Out_of | Transfer _ -> [ s ]
    | Bounce -> [ s; d ]
    | Across ->
        List.filter_map
          (fun b -> if hosted b = None then Some (device b) else None)
          [ src; dst ]
    | File e -> Option.to_list e

  let move ~timed route ~src ~dst n =
    let s = device src and d = device dst in
    match route with
    | Host_copy ->
        host_move (host_of s)
          ~dst:(Option.get (hosted dst))
          ~src:(Option.get (hosted src))
          n
    | Into -> into ~timed d ~src ~dst n
    | Out_of -> out_of ~timed s ~src ~dst n
    | Transfer (transfer, at) -> (
        match
          run ~timed s (queue s) (transfer ~dst:at ~src:(address src) n)
        with
        | () -> ()
        | exception (Lost _ as e) ->
            (* [s]'s copy engine may still write [dst]. *)
            poison dst.base s;
            raise e)
    | Bounce -> bounce ~timed s ~src d ~dst n
    | Across -> across ~timed ~src ~dst n
    | Link l -> (
        match l.move ~src ~dst with
        | () -> ()
        | exception Failure why ->
            (* The link's state is unknown at both ends, and its devices may
               still write [dst]. *)
            List.iter
              (fun t ->
                lose t why;
                poison dst.base t)
              l.through;
            List.iter check l.through;
            failwith why)
    | File _ -> file_copy ~timed ~src ~dst n

  (* A profiled copy is a span of the host on the calling domain's lane, and one
     on the copy lane of each device whose copy queue ran it, from its timestamp
     slots. *)
  let profiled ep route ~src ~dst n =
    let s = device src and d = device dst in
    let start = now_ns () in
    move ~timed:true route ~src ~dst n;
    let stop = now_ns () in
    let name = s.name ^ " -> " ^ d.name in
    push ep.events
      (Span { device = host; lane = domain_lane (); name; start; stop });
    List.iter
      (fun e ->
        push ep.events
          (Span
             {
               device = e;
               lane = "copy";
               name;
               start = stamp e 0;
               stop = stamp e 1;
             }))
      (queues route ~src ~dst)

  let copy ~src ~dst =
    let fail fmt =
      Printf.ksprintf (fun m -> invalid_arg ("Nx_device.Buffer.copy: " ^ m)) fmt
    in
    let n = nbytes src in
    if n <> nbytes dst then fail "%d bytes into %d bytes" n (nbytes dst);
    if overlaps src dst then fail "the source and destination overlap";
    let s = device src and d = device dst in
    if (s == disk || d == disk) && not (local src && local dst) then
      fail "DISK copies to and from the devices of this machine";
    if d == disk && not (file_of dst).writable then
      fail "%s is open for reading only" (file_of dst).path;
    let route, also = route ~src ~dst in
    (* The devices whose memory a borrow maps are waited for too. *)
    let rs = (root src).base.owner and rd = (root dst).base.owner in
    with_devices (s :: d :: rs :: rd :: also) (fun () ->
        List.iter sync
          (List.sort_uniq (fun a b -> Int.compare a.id b.id) [ s; d; rs; rd ]);
        List.iter staging_done also;
        reachable src;
        reachable dst;
        (if n > 0 then
           match Atomic.get profiles with
           | None -> move ~timed:false route ~src ~dst n
           | Some ep -> profiled ep route ~src ~dst n);
        (* A borrow's memory is its host's, which the copy counts in. *)
        let holder b =
          match b.base.source with Some (m, _) -> m.owner | None -> device b
        in
        let s = holder src and d = holder dst in
        if d != s then begin
          s.bytes_out <- s.bytes_out + n;
          d.bytes_in <- d.bytes_in + n
        end);
    (* The copy runs on addresses with the runtime released: the buffers, and
       the memory they keep, must stay reachable until it returns. *)
    ignore (Sys.opaque_identity src);
    ignore (Sys.opaque_identity dst)

  let offset b = b.offset
end

let staging h =
  if Option.is_some h.machine || h == disk then
    invalid_arg (Printf.sprintf "Nx_device.staging: %s is no host" h.name);
  with_devices [ h ] (fun () -> Buffer.staging h)

(* Programs *)

module Program = struct
  type t = program

  (* [d]'s reachable image of [binary], among the images of its length. The
     binary of a load is found again as the same string, which [String.equal]
     tells before reading a byte, so that finding a function of a loaded binary
     costs nothing like reading the binary. *)
  let indexed d binary =
    List.find_map
      (fun cell ->
        match Weak.get cell 0 with
        | Some l when String.equal l.binary binary -> Some l
        | Some _ | None -> None)
      (Hashtbl.find_all d.images (String.length binary))

  (* Indexes [l]. The entries of collected images go once they are as many as
     those that were live at the last sweep. *)
  let index d l =
    if Hashtbl.length d.images >= 2 * d.indexed then begin
      Hashtbl.filter_map_inplace
        (fun _ cell -> if Weak.check cell 0 then Some cell else None)
        d.images;
      d.indexed <- Int.max 8 (Hashtbl.length d.images)
    end;
    let cell = Weak.create 1 in
    Weak.set cell 0 (Some l);
    Hashtbl.add d.images (String.length l.binary) cell

  (* [d]'s image of [binary], loaded unless one is reachable. A loader that
     raises [Failure] loses its device. *)
  let loaded d load binary =
    match indexed d binary with
    | Some l -> Ok l
    | None -> (
        match load ~binary with
        | Ok image ->
            (* Code in the device's memory counts there; a driver's own objects
               pace the collector by the binary's size. *)
            let code = code_bytes image in
            if code > 0 then begin
              allocate_bytes d (code_kind d image) code;
              memory_changed d
            end;
            let bytes = if code > 0 then code else String.length binary in
            let token = release_token d (code_kind d image) (Code image) bytes
            and entries = Hashtbl.create 4 in
            let rec l = { binary; image; entries; kept = Keep (token, l) } in
            index d l;
            Ok l
        | Error why -> Error why
        | exception Failure why -> fail d why)

  (* The function [name] of [l], found once per image. *)
  let entry d l name =
    match Hashtbl.find_opt l.entries name with
    | Some h -> Ok (h, false)
    | None -> (
        match l.image.entry name with
        | Ok h ->
            Hashtbl.replace l.entries name h;
            Ok (h, true)
        | Error why -> Error why
        | exception Failure why -> fail d why)

  let find d load ~binary ~name =
    match
      Result.bind (loaded d load binary) (fun l ->
          Result.map (fun e -> (l, e)) (entry d l name))
    with
    | Error why -> Error (d.name ^ ": " ^ why)
    | Ok (l, (h, found)) ->
        let p = { p_device = d; p_name = name; p_handle = h; p_loaded = l } in
        if found && d == host then name_program h name;
        (match Atomic.get profiles with
        | Some ep when found ->
            push ep.events (Load { program = p; binary; time = now_ns () })
        | Some _ | None -> ());
        Ok p

  (* A driver with no memory for the code raises [Out_of_memory] having changed
     nothing, and the load runs the last resort of an allocation (see
     [last_resort_rounds]). *)
  let load d ~binary ~name =
    match d.load with
    | None -> Error (d.name ^ ": the device loads no programs")
    | Some load ->
        let rec take round =
          match with_devices [ d ] (fun () -> find d load ~binary ~name) with
          | r -> r
          | exception Out_of_memory (d', _)
            when d' == d && round < last_resort_rounds ->
              exhausted d;
              take (round + 1)
        in
        take 0

  let keep p (b : Buffer.t) =
    { b with base = { b.base with keep = With (b.base.keep, p.p_loaded.kept) } }

  let code p =
    Option.map
      (fun (r : region) ->
        Buffer.first
          (Buffer.base ~borrowed:true ~keep:p.p_loaded.kept p.p_device r)
          Nx_dtype.Scalar.UInt8 r.nbytes)
      p.p_loaded.image.code

  let device p = p.p_device
  let name p = p.p_name
  let handle p = p.p_handle

  type split = { extent : int; blocks : int; lo : int; hi : int }

  external workers : unit -> int = "caml_nx_device_workers"
  external entry_address : unit -> nativeint = "caml_nx_device_entry"

  let entry = entry_address ()

  external call_host : nativeint -> Buffer.t array -> int array -> unit
    = "caml_nx_device_call"

  external call_split :
    nativeint -> Buffer.t array -> int array -> split -> unit
    = "caml_nx_device_call_split"

  let call ?split p buffers values =
    let refuse fmt =
      Printf.ksprintf
        (fun m -> invalid_arg ("Nx_device.Program.call: " ^ m))
        fmt
    in
    let d = p.p_device in
    let slot s = s >= 0 && s < Array.length values in
    Option.iter
      (fun s ->
        if s.extent < 0 then refuse "a split of %d iterations" s.extent;
        if s.blocks < 1 then refuse "a split into %d blocks" s.blocks;
        if s.extent > max_int / s.blocks then
          refuse "a split of %d iterations into %d blocks overflows" s.extent
            s.blocks;
        if not (slot s.lo && slot s.hi) then
          refuse "a split's slots %d and %d among %d values" s.lo s.hi
            (Array.length values);
        if s.lo = s.hi then refuse "a split's bounds share the slot %d" s.lo)
      split;
    let run () =
      match split with
      | None -> call_host p.p_handle buffers values
      | Some s -> call_split p.p_handle buffers values s
    in
    if d != host && Option.is_none d.call then
      refuse "the program is on %s, which runs no programs" d.name;
    Array.iter
      (fun (b : Buffer.t) ->
        if Option.is_none b.base.memory.host || host_of b.base.owner != d then
          refuse "%s does not address %s memory" d.name b.base.owner.name;
        Buffer.reachable b)
      buffers;
    if d == host then begin
      (match Atomic.get profiles with
      | None -> run ()
      | Some ep ->
          let start = now_ns () in
          run ();
          let stop = now_ns () in
          push ep.events
            (Span
               {
                 device = host;
                 lane = domain_lane ();
                 name = p.p_name;
                 start;
                 stop;
               }));
      (* The program runs with the runtime released: it must stay reachable, and
         its code mapped, until it returns. *)
      ignore (Sys.opaque_identity p)
    end
    else
      let values =
        match split with
        | None -> values
        | Some s ->
            let v = Array.copy values in
            v.(s.lo) <- 0;
            v.(s.hi) <- s.extent;
            v
      in
      let at b = (Option.get (Buffer.hosted b), Buffer.nbytes b) in
      try Option.get d.call p.p_handle (Array.map at buffers) values
      with Failure msg -> fail d msg
end

(* Statistics *)

module Stats = struct
  type t = {
    allocated : int;
    cached : int;
    retained : int;
    bytes_in : int;
    bytes_out : int;
  }

  let allocated s = s.allocated
  let cached s = s.cached
  let retained s = s.retained
  let bytes_in s = s.bytes_in
  let bytes_out s = s.bytes_out

  let diff s s' =
    {
      allocated = s'.allocated - s.allocated;
      cached = s'.cached - s.cached;
      retained = s'.retained - s.retained;
      bytes_in = s'.bytes_in - s.bytes_in;
      bytes_out = s'.bytes_out - s.bytes_out;
    }
end

(* Statistics read counters and reach no memory, so they answer on a failed
   device, which reclaims nothing. *)
let stats d =
  Mutex.protect d.lock (fun () ->
      (if failed d = None then try reclaim d with Lost _ -> ());
      {
        Stats.allocated = allocated d;
        cached = cached d;
        retained = retained d;
        bytes_in = d.bytes_in;
        bytes_out = d.bytes_out;
      })

(* Submitting work *)

module Submission = struct
  type device = t

  type t = {
    devices : device list; (* the devices whose work it is *)
    values : int array; (* the value of each, in order *)
    taken : device list;
    waits : (device * int) list;
    mutable spans : (device * pending) list;
  }

  let invalid fmt = Printf.ksprintf invalid_arg ("Nx_device.Submission." ^^ fmt)

  let value s d =
    let rec find i = function
      | [] -> invalid "value: %s is not a device of the submission" d.name
      | d' :: ds -> if d' == d then s.values.(i) else find (i + 1) ds
    in
    find 0 s.devices

  let waits s = s.waits

  let wait s d v =
    if not (List.memq d s.taken) then
      invalid "wait: %s is not taken by the submission" d.name;
    if v > submitted d then
      invalid "wait: %s has submitted %d, not %d" d.name (submitted d) v;
    wait_signal d v

  let record s d ~lane ~name (stamps : buffer) =
    if not (List.memq d s.devices) then
      invalid "record: %s is not a device of the submission" d.name;
    if stamps.dtype <> Nx_dtype.Scalar.UInt64 || stamps.length <> 4 then
      invalid "record: the stamps are not four UInt64";
    match (stamps.base.memory.host, Atomic.get profiles) with
    | None, _ ->
        invalid "record: the host does not address the stamps on %s"
          stamps.base.owner.name
    | Some _, _ when host_of stamps.base.owner != host_of d ->
        invalid "record: the stamps on %s are not of %s's machine"
          stamps.base.owner.name d.name
    | Some _, None -> ()
    | Some a, Some into ->
        let address = Nativeint.add a (Nativeint.of_int stamps.offset) in
        s.spans <-
          (d, { address; stamps = Keep stamps; lane; name; into }) :: s.spans

  let copied s ~src ~dst n =
    if not (List.memq src s.taken && List.memq dst s.taken) then
      invalid "copied: the submission does not take %s and %s" src.name dst.name;
    if n < 0 then invalid "copied: %d bytes" n;
    if src != dst then begin
      src.bytes_out <- src.bytes_out + n;
      dst.bytes_in <- dst.bytes_in + n
    end
end

let by_id a b = Int.compare a.id b.id
let runs_work d = Option.is_some d.machine && d != disk

(* Whether the work of every device of [ds] can wait on [d']'s signal word: [d']
   stores its values into it, and each device of [ds] addresses it, as memory of
   its own machine that it maps. *)
let encodable ds d' =
  runs_work d' && Option.is_none d'.signal
  && List.for_all
       (fun d ->
         d == d' || (host_of d == host_of d' && Option.is_some (mapping_of d)))
       ds

(* [d] added to the devices [l], once. *)
let add d l = if List.memq d l then l else d :: l

(* [l] with the devices whose memory [b] reaches: its own, and those of the
   memory its borrows map. *)
let rec reach_into l b =
  let l = add b.owner l in
  match b.source with Some (src, _) -> reach_into l src | None -> l

(* Waits on the host for the work that touched [b]'s memory. *)
let wait_work (b : buffer) =
  List.iter (fun s -> wait_signal s.by s.upto) (Atomic.get b.base.links).stamps

(* The staged buffers of [touches], with what they stand in for. A submission
   touching none allocates nothing here. *)
let rec staged_of = function
  | [] -> []
  | (b : buffer) :: rest -> (
      match stage_of b.base with
      | Some st -> (b, st) :: staged_of rest
      | None -> staged_of rest)

(* A staged buffer's copies, on the host: its original into it once the work
   that wrote the original and the work that used the staged memory are done,
   and back into its original once its own work is. The device's work never
   touches the original, which the submission neither takes nor stamps. *)
let stage_in ((b : buffer), st) =
  Buffer.reachable st.original;
  wait_work st.original;
  wait_work b;
  memmove
    (Option.get (Buffer.hosted b))
    (Option.get (Buffer.hosted st.original))
    (Buffer.nbytes b)

let stage_out ((b : buffer), st) =
  wait_work b;
  memmove
    (Option.get (Buffer.hosted st.original))
    (Option.get (Buffer.hosted b))
    (Buffer.nbytes b)

(* Stamps each memory that [touches] reach with the values [values] of the
   devices [ds], which signal once their work is done: the release of that
   memory waits for them. *)
let rec stamp_devices base values i = function
  | [] -> ()
  | d :: rest ->
      stamp_use base d values.(i);
      stamp_devices base values (i + 1) rest

let rec stamp_reach ds values base =
  stamp_devices base values 0 ds;
  match base.source with
  | Some (src, _) -> stamp_reach ds values src
  | None -> ()

let rec stamp_touches ds values = function
  | [] -> ()
  | (b : buffer) :: rest ->
      stamp_reach ds values b.base;
      stamp_touches ds values rest

let submit_taken ds ~touches f =
  let invalid fmt = Printf.ksprintf invalid_arg ("Nx_device.submit: " ^^ fmt) in
  let ds = List.sort_uniq by_id ds in
  if ds = [] then invalid "no device";
  List.iter
    (fun d -> if not (runs_work d) then invalid "%s runs no work" d.name)
    ds;
  List.iter
    (fun (b : buffer) ->
      if b.base.owner == disk then invalid "a buffer on the disk")
    touches;
  let reached =
    List.fold_left (fun l (b : buffer) -> reach_into l b.base) [] touches
  in
  let on =
    List.fold_left (fun l (b : buffer) -> add b.base.owner l) ds touches
  in
  let taken = List.fold_left (fun l d -> add d l) on reached in
  with_devices taken @@ fun () ->
  List.iter Buffer.reachable touches;
  (* The latest value each device's work touched the reached memory with: the
     stamps of each buffer's memory and of the memory its borrows map. *)
  let latest = ref [] in
  let note (s : stamp) =
    (* A lost device's work never completes; the memory it can reach raises its
       loss instead. *)
    if failed s.by = None then
      match List.assq_opt s.by !latest with
      | Some v when !v >= s.upto -> ()
      | Some v -> v := s.upto
      | None -> latest := (s.by, ref s.upto) :: !latest
  in
  let rec note_reach base =
    List.iter note (Atomic.get base.links).stamps;
    match base.source with Some (src, _) -> note_reach src | None -> ()
  in
  List.iter (fun (b : buffer) -> note_reach b.base) touches;
  let d_set = List.sort by_id (List.filter (encodable ds) on) in
  (* A device's own earlier work is ordered by its vendor's rule. *)
  let alone d' = match ds with [ d ] -> d == d' | _ -> false in
  List.iter
    (fun (d', v) ->
      if not (List.memq d' d_set || alone d') then wait_signal d' !v)
    !latest;
  let waits =
    List.map
      (fun d' ->
        match List.assq_opt d' !latest with
        | Some v -> (d', !v)
        | None -> (d', Atomic.get d'.settled))
      d_set
  in
  List.iter wait_room ds;
  List.iter stage_in (staged_of touches);
  let s =
    {
      Submission.devices = ds;
      values = Array.of_list (List.map (fun d -> submitted d + 1) ds);
      taken;
      waits;
      spans = [];
    }
  in
  let r = f s in
  List.iteri (fun i d -> commit d s.values.(i)) ds;
  stamp_touches ds s.values touches;
  List.iter
    (fun t ->
      List.iteri
        (fun i d ->
          if t != d then Hashtbl.replace t.pending d.id (d, s.values.(i)))
        ds)
    reached;
  List.iter (fun (d, p) -> push d.spans p) (List.rev s.spans);
  r

(* A submission that touches staged buffers holds them from before its copies in
   until after its copies back, which wait for its work with no device taken, so
   that they block no other domain's work on the devices. *)
let submit ds ~touches f =
  match staged_of touches with
  | [] -> submit_taken ds ~touches f
  | staged ->
      let staged =
        List.sort_uniq
          (fun (_, (s : stage)) (_, (s' : stage)) -> Int.compare s.id s'.id)
          staged
      in
      let rec hold = function
        | [] -> ()
        | (_, st) :: rest -> (
            Mutex.lock st.busy;
            match hold rest with
            | () -> ()
            | exception e ->
                Mutex.unlock st.busy;
                raise e)
      in
      hold staged;
      Fun.protect
        ~finally:(fun () ->
          List.iter (fun (_, st) -> Mutex.unlock st.busy) staged)
        (fun () ->
          let r = submit_taken ds ~touches f in
          List.iter
            (fun ((_, st) as b) -> if st.access = Read_write then stage_out b)
            staged;
          r)

let signal_word d =
  (* A [Device_local] device's timeline is its pinned memory, which the devices
     of its machine borrow through their mappings of host memory. *)
  let base =
    match d.kind with
    | Local _ ->
        Buffer.base ~kind:Pinned ~borrowed:true ~keep:d.timeline_keep d
          d.timeline
    | _ ->
        Buffer.base ~borrowed:true ~keep:d.timeline_keep (host_of d) d.timeline
  in
  Buffer.first base Nx_dtype.Scalar.UInt64 1

(* Profiles *)

module Profile = struct
  type nonrec event = event =
    | Span of {
        device : t;
        lane : string;
        name : string;
        start : int;
        stop : int;
      }
    | Allocation of { device : t; time : int; allocated : int }
    | Load of { program : program; binary : string; time : int }
    | Counters of {
        device : t;
        name : string;
        start : int;
        stop : int;
        counters : (string * int array) list;
      }
    | Trace of {
        device : t;
        name : string;
        start : int;
        stop : int;
        part : int;
        data : string;
      }
    | Overwritten of { device : t; time : int; runs : int }

  type t = collector

  let now = now_ns
  let enabled () = Option.is_some (Atomic.get profiles)

  let taken () =
    match Atomic.get profiles with Some ep -> ep.taken | None -> []

  (* The epoch of the profiles [cs]: the counters of each, each once, earliest
     profile first. *)
  let epoch cs =
    let counted =
      List.fold_left
        (fun acc c ->
          acc @ List.filter (fun n -> not (List.mem n acc)) c.counters)
        [] (List.rev cs)
    in
    let traced = List.exists (fun c -> c.trace) cs in
    {
      taken = cs;
      counted;
      traced;
      events = Atomic.make [];
      reports = Atomic.make [];
    }

  (* Starts and stops change the profiles taken one at a time, each beginning an
     epoch. The spans of host programs C recorded since the last change go to
     the epoch that ends; C records while a profile is taken. *)
  let changing = Mutex.create ()

  let change f =
    Mutex.protect changing (fun () ->
        let ended = Atomic.get profiles in
        let next = f (taken ()) in
        Option.iter
          (fun ep ->
            Array.iter
              (fun (name, lane, start, stop) ->
                push ep.events
                  (Span
                     { device = host; lane = lane_of lane; name; start; stop }))
              (host_spans ()))
          ended;
        (match (ended, next) with
        | None, _ :: _ -> record_spans true
        | Some _, [] -> record_spans false
        | _ -> ());
        match next with
        | [] -> Atomic.set profiles None
        | cs ->
            let ep = epoch cs in
            List.iter (fun c -> c.epochs <- ep :: c.epochs) cs;
            Atomic.set profiles (Some ep))

  let not_taken () =
    invalid_arg "Nx_device.Profile.stop: the profile is not being taken"

  (* Stops taking [p] without reading its events. *)
  let close p =
    change (fun cs ->
        if List.memq p cs then List.filter (fun c -> c != p) cs
        else not_taken ())

  (* The events of [p]'s epochs, in the order they were recorded. The latest
     epoch is read first, so that the events read include every event recorded
     before any of them. *)
  let collect p =
    let read acc ep =
      let events = List.rev (Atomic.get ep.events)
      and reports = List.rev (Atomic.get ep.reports) in
      (events @ List.filter_map (reported p) reports) :: acc
    in
    List.concat (List.fold_left read [] p.epochs)

  let start ?(counters = []) ?(trace = false) () =
    let rec once = function
      | [] -> ()
      | n :: rest when List.mem n rest ->
          invalid_arg
            (Printf.sprintf
               "Nx_device.Profile.start: the counter %s is asked twice" n)
      | _ :: rest -> once rest
    in
    once counters;
    let c = { counters; trace; epochs = [] } in
    change (fun cs -> c :: cs);
    c

  let counters () =
    match Atomic.get profiles with Some ep -> ep.counted | None -> []

  let traced () =
    match Atomic.get profiles with Some ep -> ep.traced | None -> false

  let span name f =
    match Atomic.get profiles with
    | None -> f ()
    | Some ep -> (
        let start = now_ns () in
        let record () =
          let stop = now_ns () in
          push ep.events
            (Span { device = host; lane = domain_lane (); name; start; stop })
        in
        match f () with
        | r ->
            record ();
            r
        | exception e ->
            let bt = Printexc.get_raw_backtrace () in
            record ();
            Printexc.raise_with_backtrace e bt)

  (* The host time of a tick of [d]'s clock of [hz] ticks per second. The sample
     whose wait brackets its stamp most narrowly bounds the error best. *)
  let calibrate d hz =
    with_devices [ d ] (fun () ->
        let q = Option.get (queue_of d) in
        let rec sample k (width, mid, tick) =
          if k = 0 then (mid, tick)
          else
            let h0 = now_ns () in
            wait_signal d (enqueue d (q.stamp ~slot:(stamp_slot d 0)));
            let h1 = now_ns () in
            sample (k - 1)
              (if h1 - h0 < width then (h1 - h0, h0 + ((h1 - h0) / 2), stamp d 0)
               else (width, mid, tick))
        in
        let mid, tick = sample 5 (max_int, 0, 0) in
        let ns = 1e9 /. Float.of_int hz in
        fun t ->
          mid + Float.to_int (Float.round (Float.of_int (t - tick) *. ns)))

  let time = function
    | Span s -> s.start
    | Allocation m -> m.time
    | Load p -> p.time
    | Counters c -> c.start
    | Trace t -> t.start
    | Overwritten o -> o.time

  let length = function
    | Span s -> s.stop - s.start
    | Counters c -> c.stop - c.start
    | Trace t -> t.stop - t.start
    | Allocation _ | Load _ | Overwritten _ -> 0

  (* By time, and at equal times longest first, so that nested spans follow the
     spans they are in. *)
  let order a b =
    match Int.compare (time a) (time b) with
    | 0 -> Int.compare (length b) (length a)
    | c -> c

  (* The devices' spans and reports are read while [p] is still taken, so that
     they go to it. *)
  let stop p =
    if not (List.memq p (taken ())) then not_taken ();
    List.iter
      (fun d ->
        let counted = counts p && Option.is_some d.report in
        if (Atomic.get d.spans <> [] || counted) && failed d = None then
          try with_devices [ d ] (fun () -> sync d) with Lost _ -> ())
      (Atomic.get opened);
    close p;
    let clocks = Hashtbl.create 4 in
    let calibrated d hz =
      match Hashtbl.find_opt clocks d.id with
      | Some f -> f
      | None ->
          let f = try Some (calibrate d hz) with Lost _ -> None in
          Hashtbl.add clocks d.id f;
          f
    in
    collect p
    |> List.filter_map (function
      | Span ({ device = { clock = Device_clock { hz }; _ } as d; _ } as s) ->
          Option.map
            (fun f -> Span { s with start = f s.start; stop = f s.stop })
            (calibrated d hz)
      | Counters ({ device = { clock = Device_clock { hz }; _ } as d; _ } as c)
        ->
          Option.map
            (fun f -> Counters { c with start = f c.start; stop = f c.stop })
            (calibrated d hz)
      | Trace ({ device = { clock = Device_clock { hz }; _ } as d; _ } as t) ->
          Option.map
            (fun f -> Trace { t with start = f t.start; stop = f t.stop })
            (calibrated d hz)
      | e -> Some e)
    |> List.stable_sort order

  let take ?counters ?trace f =
    let p = start ?counters ?trace () in
    match f () with
    | r -> (r, stop p)
    | exception e ->
        let bt = Printexc.get_raw_backtrace () in
        close p;
        Printexc.raise_with_backtrace e bt

  (* Chrome's trace event format *)

  let device_of = function
    | Span s -> s.device
    | Allocation m -> m.device
    | Load p -> p.program.p_device
    | Counters c -> c.device
    | Trace t -> t.device
    | Overwritten o -> o.device

  (* [s] as a JSON string. Malformed UTF-8 becomes U+FFFD. *)
  let string oc s =
    output_char oc '"';
    let rec go i =
      if i < String.length s then begin
        let d = String.get_utf_8_uchar s i in
        let n = Uchar.utf_decode_length d in
        (if not (Uchar.utf_decode_is_valid d) then output_string oc "\u{FFFD}"
         else
           match s.[i] with
           | '"' -> output_string oc "\\\""
           | '\\' -> output_string oc "\\\\"
           | '\n' -> output_string oc "\\n"
           | '\r' -> output_string oc "\\r"
           | '\t' -> output_string oc "\\t"
           | c when Char.code c < 0x20 ->
               Printf.fprintf oc "\\u%04x" (Char.code c)
           | _ -> output_substring oc s i n);
        go (i + n)
      end
    in
    go 0;
    output_char oc '"'

  (* [ns] nanoseconds in microseconds. *)
  let micros oc ns =
    if ns < 0 then output_char oc '-';
    Printf.fprintf oc "%d.%03d" (abs ns / 1000) (abs ns mod 1000)

  let counter_args oc counters =
    output_string oc ",\"args\":{";
    List.iteri
      (fun i (name, values) ->
        if i > 0 then output_char oc ',';
        string oc name;
        Printf.fprintf oc ":%d" (Array.fold_left ( + ) 0 values))
      counters;
    output_char oc '}'

  (* The lane of a device on which its runs' counters show. *)
  let counters_lane = "counters"

  let output_chrome_trace oc events =
    let events = List.stable_sort order events in
    let origin = match events with [] -> 0 | e :: _ -> time e in
    let pids = Hashtbl.create 8 and tids = Hashtbl.create 8 in
    let first = ref true in
    let next () = if !first then first := false else output_string oc ",\n" in
    let meta ~pid ~tid what name =
      next ();
      Printf.fprintf oc
        "{\"ph\":\"M\",\"pid\":%d,\"tid\":%d,\"name\":\"%s\",\"args\":{\"name\":"
        pid tid what;
      string oc name;
      output_string oc "}}"
    in
    let pid d =
      match Hashtbl.find_opt pids d.id with
      | Some pid -> pid
      | None ->
          let pid = Hashtbl.length pids + 1 in
          Hashtbl.add pids d.id pid;
          meta ~pid ~tid:0 "process_name" d.name;
          pid
    in
    let tid d pid lane =
      match Hashtbl.find_opt tids (d.id, lane) with
      | Some tid -> tid
      | None ->
          let tid = Hashtbl.length tids + 1 in
          Hashtbl.add tids (d.id, lane) tid;
          meta ~pid ~tid "thread_name" lane;
          tid
    in
    output_string oc "{\"traceEvents\":[\n";
    List.iter
      (fun e ->
        let pid = pid (device_of e) in
        let tid =
          match e with
          | Span s -> tid s.device pid s.lane
          | Counters c -> tid c.device pid counters_lane
          | Allocation _ | Load _ | Trace _ | Overwritten _ -> 0
        in
        next ();
        let ph =
          match e with
          | Span _ | Counters _ -> "X"
          | Allocation _ -> "C"
          | Load _ | Trace _ | Overwritten _ -> "i"
        in
        Printf.fprintf oc "{\"ph\":\"%s\",\"pid\":%d,\"tid\":%d,\"ts\":" ph pid
          tid;
        micros oc (time e - origin);
        (match e with
        | Span s ->
            output_string oc ",\"dur\":";
            micros oc (s.stop - s.start);
            output_string oc ",\"name\":";
            string oc s.name
        | Counters c ->
            output_string oc ",\"dur\":";
            micros oc (c.stop - c.start);
            output_string oc ",\"name\":";
            string oc c.name;
            counter_args oc c.counters
        | Trace t ->
            output_string oc ",\"s\":\"p\",\"name\":";
            string oc t.name;
            Printf.fprintf oc ",\"args\":{\"part\":%d,\"bytes\":%d}" t.part
              (String.length t.data)
        | Overwritten o ->
            Printf.fprintf oc
              ",\"s\":\"p\",\"name\":\"overwritten\",\"args\":{\"runs\":%d}"
              o.runs
        | Allocation m ->
            Printf.fprintf oc ",\"name\":\"memory\",\"args\":{\"allocated\":%d}"
              m.allocated
        | Load p ->
            output_string oc ",\"s\":\"p\",\"name\":";
            string oc p.program.p_name;
            Printf.fprintf oc ",\"args\":{\"handle\":\"0x%nx\"}"
              p.program.p_handle);
        output_char oc '}')
      events;
    output_string oc "\n]}\n"
end

(* Drivers *)

module Driver = struct
  module Region = struct
    type t = region

    let v ?host ?(handle = 0n) address nbytes =
      if nbytes < 0 then
        invalid_arg
          (Printf.sprintf "Nx_device.Driver.Region.v: %d bytes" nbytes);
      { host; address; handle; nbytes }

    let address (r : t) = r.address
    let host_address (r : t) = r.host
    let handle (r : t) = r.handle
    let nbytes (r : t) = r.nbytes
    let of_buffer (b : Buffer.t) = b.base.memory
  end

  type nonrec allocator = allocator = {
    alloc : int -> region option;
    free : region -> unit;
  }

  type nonrec mapping = mapping =
    | Identity
    | Pages of {
        map : nativeint -> int -> (region, string) result;
        unmap : region -> unit;
      }

  type nonrec copy = copy
  type nonrec clock = clock = Host_clock | Device_clock of { hz : int }

  type nonrec queue = queue = {
    copy : copy;
    transfer : t -> copy option;
    stamp : slot:nativeint -> signal:int -> unit;
    clock : clock;
  }

  type memory = driver_memory =
    | Host_visible of { memory : allocator; mapping : mapping option }
    | Device_local of {
        memory : allocator;
        host_memory : allocator;
        mapped : (allocator * int) option;
        mapping : mapping;
        queue : timeline:region -> queue;
      }

  type nonrec signal = signal = {
    signaled : unit -> int;
    wait : int -> ms:int -> bool;
  }

  type nonrec sleep = sleep

  type nonrec completion = completion =
    | Poll
    | Sleep of sleep
    | Signal of (timeline:region -> signal)

  type nonrec dma = dma = { bus : string; pages : (int * int) list }

  type nonrec link = link = {
    through : t list;
    move : src:buffer -> dst:buffer -> unit;
  }

  type nonrec io = io = {
    read : src:nativeint -> dst:nativeint -> int -> unit;
    write : dst:nativeint -> src:nativeint -> int -> unit;
    copy : dst:nativeint -> src:nativeint -> int -> unit;
  }

  type nonrec host_programs = host_programs = {
    load :
      binary:string ->
      entry:string ->
      (nativeint * (unit -> unit), string) result;
    call : nativeint -> (nativeint * int) array -> int array -> unit;
  }

  type nonrec image = image = {
    code : region option;
    entry : string -> (nativeint, string) result;
    unload : unit -> unit;
  }

  (* The host's heap: its regions keep their bigarrays until they are freed. *)
  let host_memory =
    let held = Hashtbl.create 16 and lock = Mutex.create () in
    let alloc n =
      match heap n with
      | ba ->
          let r = heap_memory ba in
          Mutex.protect lock (fun () -> Hashtbl.replace held r.address ba);
          Some r
      | exception Stdlib.Out_of_memory -> None
    in
    let free (r : region) =
      Mutex.protect lock (fun () -> Hashtbl.remove held r.address)
    in
    { alloc; free }

  let host_programs = host_programs

  let refuse fn fmt =
    Printf.ksprintf
      (fun m -> invalid_arg (Printf.sprintf "Nx_device.Driver.%s: %s" fn m))
      fmt

  let wait ?(host = host) ~sleep ~(timeline : region) ready =
    if Option.is_some host.machine then
      refuse "wait" "%s is not a host" host.name;
    if timeline.nbytes < 8 then
      refuse "wait" "a timeline of %d bytes" timeline.nbytes;
    match timeline.host with
    | None -> refuse "wait" "the host does not address the timeline"
    | Some a ->
        await ~io:(io_of host)
          ~sleep:(Some (sleep ~timeline))
          ~spin:relax a ready

  let compose ?(host = host) local =
    match host.kind with
    | Machine { remote = Some (a, _); _ } -> local ^ "@" ^ a
    | _ -> local

  let name = compose

  (* The device of each name minted on each machine, by its host's id, so that
     no two live devices of one machine share a name: a name identifies a device
     of its machine. A name is reserved, as [None], before its device is made,
     and given back if making it fails. A lost device's name is minted again,
     for a fresh device of the same hardware. *)
  let minted = Hashtbl.create 16
  let minted_lock = Mutex.create ()

  let () =
    Hashtbl.replace minted (host.id, host.name) (Some host);
    Hashtbl.replace minted (host.id, disk.name) (Some disk)

  let reserve key name =
    Mutex.protect minted_lock (fun () ->
        match Hashtbl.find_opt minted key with
        | Some (Some d) when failed d <> None -> Hashtbl.replace minted key None
        | Some _ ->
            refuse "device" "a device named %s exists on its machine" name
        | None -> Hashtbl.replace minted key None)

  let minted_as key d =
    Mutex.protect minted_lock (fun () -> Hashtbl.replace minted key (Some d))

  let give_back key =
    Mutex.protect minted_lock (fun () -> Hashtbl.remove minted key)

  let device ~name ~arch ~budget ?(host = host) ?(completion = Poll) ?load ?peer
      ?(reaches = fun _ -> false) ?link ?dma ?(resolve = ignore)
      ?(synchronized = ignore) ?report ?(room = fun () -> true)
      ?(finalize = fun ~failed:_ -> ()) memory =
    if budget < 0 then refuse "device" "budget %d < 0" budget;
    if Option.is_some host.machine then
      refuse "device" "%s is not a host" host.name;
    let key = (host.id, name) and name = compose ~host name in
    if name = host.name then
      refuse "device" "a device named %s exists on its machine" name;
    let memory =
      match memory with
      | Host_visible _ -> memory
      | Device_local l ->
          let queue ~timeline =
            let q = l.queue ~timeline in
            (match q.clock with
            | Device_clock { hz } when hz <= 0 ->
                refuse "device" "a clock of %d Hz" hz
            | Host_clock | Device_clock _ -> ());
            q
          in
          Device_local { l with queue }
    in
    reserve key name;
    let description =
      {
        Description.default with
        name;
        arch;
        machine = Some host;
        budget;
        peer;
        reaches_peer = reaches;
        load;
        link;
        dma;
        completion;
        synchronized;
        report;
        room;
        finalize;
        resolve;
      }
    in
    match create description (Driver_memory memory) with
    | d ->
        minted_as key d;
        d
    | exception e ->
        let bt = Printexc.get_raw_backtrace () in
        give_back key;
        Printexc.raise_with_backtrace e bt

  let buffer d (r : region) s n =
    if d == disk then Buffer.not_files "Driver.buffer";
    if d == host then
      refuse "buffer" "CPU memory is borrowed with Buffer.of_bigarray";
    let bytes = Buffer.checked_nbytes "Driver.buffer" s n in
    if bytes > r.nbytes then
      refuse "buffer" "%d bytes do not fit in a region of %d" bytes r.nbytes;
    Buffer.first (Buffer.base ~borrowed:true ~keep:(Keep ()) d r) s n

  let dma (b : Buffer.t) =
    let d = b.base.owner in
    match d.dma with
    | None -> Error (d.name ^ " does not describe its memory to others")
    | Some f -> driver d (fun () -> f b.base.memory)

  let depends (b : Buffer.t) f =
    if b.base.borrowed || b.base.owner == host || b.base.bytes = 0 then
      refuse "depends" "the memory is not one a device allocated";
    update_links b.base (fun l -> { l with depends = f :: l.depends })

  (* Last: it shadows the host. *)
  let host ~address ~arch ?programs ?(synchronized = ignore)
      ?(finalize = fun ~failed:_ -> ()) ~memory io =
    let load = Option.map (fun p -> host_image p.load) programs in
    create
      {
        Description.default with
        name = "CPU@" ^ address;
        arch;
        load;
        call = Option.map (fun p -> p.call) programs;
        synchronized;
        finalize;
      }
      (Remote { address; io; memory })
end
