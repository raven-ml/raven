(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The field order of [region], [base], [life] and [Buffer.t] up to the fields
   nx_device.h reads is its C ABI. *)
type region = {
  host : nativeint option;
  address : nativeint;
  handle : nativeint;
  nbytes : int;
}

type signal = { signaled : unit -> int; wait : int -> timeout_ms:int -> bool }
type allocator = { alloc : int -> region option; free : region -> unit }

type mapping = {
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

type completion =
  | Poll
  | Sleep of (timeline:region -> int -> unit)
  | Signal of (timeline:region -> signal)

type host_programs = {
  load :
    binary:string -> entry:string -> (nativeint * (unit -> unit), string) result;
  call : nativeint -> (nativeint * int) array -> int array -> unit;
}

(* What must stay reachable for as long as a base does. Host memory is the
   bigarray that holds it from its first byte: views of it join that bigarray's
   storage, which outlives the base. *)
type keep =
  | Keep : 'a -> keep
  | Host : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> keep
  | Heap : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t * heap_token -> keep
  | Device : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t * 'c -> keep
(* another device's memory the host addresses, as a bigarray that owns nothing,
   and what keeps the memory *)

(* The host memory of [create] is kept with its token: a custom block whose
   finaliser returns the reserved bytes to the host's count. *)
and heap_token

(* The bytes a device allocated: an atomic count, or the host's, which the
   finalisers of its buffers' tokens return. *)
type allocated = Count of int Atomic.t | Heap_bytes

type t = {
  id : int;
  name : string;
  arch : string;
  machine : t option; (* the host of the device's machine, [None] for a host *)
  remote : string option; (* for the host of another machine, its address *)
  io : io option; (* for the host of another machine, how it is reached *)
  lock : Mutex.t;
  alloc : int -> (region * keep) option;
  free : region -> unit;
  host_memory : allocator option;
  mapping : mapping option;
  copy_queue : queue option;
  peer : (t -> region -> (region, string) result) option;
  load :
    (binary:string ->
    entry:string ->
    (nativeint * (unit -> unit) option, string) result)
    option;
      (* a program's handle, and how it is released if it can be *)
  call : (nativeint -> (nativeint * int) array -> int array -> unit) option;
      (* how a host calls its programs *)
  link : (src:buffer -> dst:buffer -> link option) option;
  dma : (region -> (dma, string) result) option;
  signal : signal option;
  sleep : (int -> unit) option;
  timeout_ms : int Atomic.t;
  synchronized : unit -> unit;
  finalize : failed:bool -> unit;
  clock : clock;
  resolve : nativeint -> unit;
  timeline : region; (* the signal word, then two 16-byte slots of timestamps *)
  timeline_keep : keep;
  last : int Atomic.t; (* the submitted value, which only this module writes *)
  settled : int Atomic.t; (* the latest value a wait saw signaled *)
  mutable slots : region option; (* a host's staging memory, once made *)
  mutable staging : nativeint option;
      (* the device's address of its host's staging memory, once mapped *)
  peers : (nativeint, (unit -> unit) list) Hashtbl.t;
      (* by the address of owned memory, what runs before it is freed *)
  peers_lock : Mutex.t;
  released : base list Atomic.t;
  failed : string option Atomic.t;
      (* why the device was lost, which every operation raises *)
  cache : (int * bool, region list) Hashtbl.t;
      (* by size, and whether it is pinned memory *)
  pending : (int, t * int) Hashtbl.t;
      (* the devices whose work touched this one's memory, and the value that
         work signals *)
  programs : (string * string, cached) Hashtbl.t;
  dropped : dropped list Atomic.t;
  spans : pending list Atomic.t;
      (* the spans recorded on the device whose stamps are still to read, latest
         first *)
  mutable held : keep list; (* retained memory, and what it keeps *)
  mutable budget : int;
  allocated : allocated;
  mutable cached : int;
  mutable retained : int;
  mutable bytes_in : int;
  mutable bytes_out : int;
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
}

and base = {
  owner : t;
  memory : region;
  bytes : int; (* of owned memory, 0 when borrowed *)
  pinned : bool; (* allocated by the owner's [host_memory] *)
  borrowed : bool;
  keep : keep;
  source : (base * mapped) option;
      (* for a borrow, the memory it maps, and the mapping *)
  links : links Atomic.t;
  file : file option; (* on the disk, the file *)
  mutable life : life;
}

(* Whether a base's buffers may reach its memory. A consumed base is [Dead], and
   the base its consumption made in its place, over the same memory, is its
   [Heir]: it keeps the dead one, whose finaliser releases the memory,
   reachable. *)
and life = Live | Heir of base | Dead of string

(* The other devices that reach a base's memory, changed together. *)
and links = {
  maps : mapped list; (* the mappings of this memory, one per device *)
  reached : t list;
      (* other devices whose work may still write this memory: the source of a
         transfer into it that could not be waited for *)
}

(* A mapping of a base's memory on a device, shared by the device's borrows of
   it, and how the device releases it once they are unreachable. [borrows]
   changes only with the device taken. *)
and mapped = {
  on : t;
  mapped : region;
  skip : int; (* bytes of [mapped] before the memory's first byte *)
  unmap : region -> unit;
  mutable borrows : int;
}

and program = { p_device : t; p_name : string; p_handle : nativeint }

(* A file a disk buffer is over: its memory's handle is the descriptor. Its
   pages are the host memory of its mapping, made at its first borrow. *)
and file = { path : string; writable : bool; mutable pages : base option }

(* A program that its device can release is cached weakly, and released once
   unreachable; the others are kept for the device's life. *)
and cached = Kept of program | Collectable of program Weak.t

(* An unreachable program, to release. *)
and dropped = {
  key : string * string;
  cell : program Weak.t;
  unload : unit -> unit;
}

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

and collector = { events : event list Atomic.t }

(* A span whose stamps are in the two 16-byte slots at [address], which [stamps]
   keeps. *)
and pending = {
  address : nativeint;
  stamps : keep;
  lane : string;
  name : string;
  into : collector;
}

exception Lost of t * string
exception Out_of_memory of t * int

let () =
  Printexc.register_printer (function
    | Lost (d, why) -> Some (d.name ^ ": " ^ why)
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

external heap_aligned :
  int ->
  int ->
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t option
  = "caml_nx_device_heap_aligned"

external now_ns : unit -> (int[@untagged])
  = "caml_nx_device_now_ns_byte" "caml_nx_device_now_ns"
[@@noalloc]

external now_ms : unit -> (int[@untagged])
  = "caml_nx_device_now_ms_byte" "caml_nx_device_now_ms"
[@@noalloc]

(* Files *)

external file_open : string -> bool -> int -> int * nativeint * int
  = "caml_nx_device_file_open"

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

(* The codes [file_open] gives, besides the system's. *)
let not_regular = -1
let too_many = -2
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
   from a page of more bytes where the C library aligns nothing it frees. *)
let heap n =
  if n < aligned_from then
    Bigarray.Array1.create Bigarray.char Bigarray.c_layout n
  else
    match heap_aligned page n with
    | Some ba -> ba
    | None ->
        if n > max_int - page then raise Stdlib.Out_of_memory;
        let ba =
          Bigarray.Array1.create Bigarray.char Bigarray.c_layout (n + page - 1)
        in
        let a = Nativeint.to_int (bigarray_address ba) in
        let skip = (page - (a mod page)) mod page in
        Bigarray.Array1.sub ba skip n

(* Profiling *)

(* The collector of the profile being taken, if any. Every recording site reads
   it once, and allocates nothing when it is [None]. *)
let profile : collector option Atomic.t = Atomic.make None

let rec push r x =
  let l = Atomic.get r in
  if not (Atomic.compare_and_set r l (x :: l)) then push r x

let allocated d =
  match d.allocated with Count c -> Atomic.get c | Heap_bytes -> heap_bytes ()

let allocate_bytes d n =
  match d.allocated with
  | Count c -> ignore (Atomic.fetch_and_add c n)
  | Heap_bytes -> heap_return (-n)

let memory_changed d =
  match Atomic.get profile with
  | None -> ()
  | Some c ->
      push c.events
        (Allocation { device = d; time = now_ns (); allocated = allocated d })

(* The lane of the calling domain on the host. *)
let domain_lane () = Printf.sprintf "domain %d" (Domain.self () :> int)

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
   copy queue. *)
let timeline_bytes = 48

let timeline_of ~host_alloc (host_memory : allocator option) =
  let check = function
    | Some ({ host = Some _; _ } as m) -> m
    | Some { host = None; _ } ->
        invalid_arg
          "Nx_device.Driver.device: host memory the host does not address"
    | None -> failwith "Nx_device.Driver.device: no memory for the timeline"
  in
  match (host_memory, host_alloc) with
  | Some (a : allocator), _ -> (check (a.alloc timeline_bytes), Keep ())
  | None, Some alloc -> (check (Option.map fst (alloc timeline_bytes)), Keep ())
  | None, None ->
      let ba =
        shared
          (Bigarray.Array1.create Bigarray.int64 Bigarray.c_layout
             (timeline_bytes / 8))
      in
      (heap_memory ba, Host ba)

let default_timeout = 30_000

let create ~name ~arch ~machine ~remote ~io ~budget ~alloc ~free ~host_memory
    ~mapping ~queue ~peer ~load ~call ~link ~dma ~completion ~synchronized
    ~finalize ~resolve =
  (* A host of another machine keeps its timeline in its own memory. *)
  let host_alloc =
    match (machine, io) with
    | Some h, _ when Option.is_some h.io -> Some h.alloc
    | None, Some _ -> Some alloc
    | _ -> None
  in
  let timeline, timeline_keep = timeline_of ~host_alloc host_memory in
  let machine_io = match machine with Some h -> h.io | None -> io in
  let words = Option.get timeline.host in
  write_word machine_io words 0L;
  let copy_queue = Option.map (fun queue -> queue ~timeline) queue in
  let signal, sleep =
    match completion with
    | Poll -> (None, None)
    | Sleep sleep -> (None, Some (sleep ~timeline))
    | Signal signal -> (Some (signal ~timeline), None)
  in
  let d =
    {
      id = Atomic.fetch_and_add ids 1;
      name;
      arch;
      machine;
      remote;
      io;
      lock = Mutex.create ();
      alloc;
      free;
      host_memory;
      mapping;
      copy_queue;
      peer;
      load;
      call;
      link;
      dma;
      signal;
      sleep;
      timeout_ms = Atomic.make default_timeout;
      synchronized;
      finalize;
      clock = (match copy_queue with Some q -> q.clock | None -> Host_clock);
      resolve;
      timeline;
      timeline_keep;
      last = Atomic.make 0;
      settled = Atomic.make 0;
      slots = None;
      staging = None;
      peers = Hashtbl.create 4;
      peers_lock = Mutex.create ();
      released = Atomic.make [];
      failed = Atomic.make None;
      cache = Hashtbl.create 16;
      pending = Hashtbl.create 4;
      programs = Hashtbl.create 16;
      dropped = Atomic.make [];
      spans = Atomic.make [];
      held = [];
      budget;
      allocated =
        (match (machine, io) with
        | None, None -> Heap_bytes
        | Some _, _ | _, Some _ -> Count (Atomic.make 0));
      cached = 0;
      retained = 0;
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

let host =
  (* Buffer.create takes host memory from the heap, never from [alloc]. *)
  let alloc _ = assert false in
  let load =
    Option.map
      (fun load ~binary ~entry ->
        Result.map (fun (h, free) -> (h, Some free)) (load ~binary ~entry))
      Host_program.load
  in
  create ~name:"CPU" ~arch:host_arch ~machine:None ~remote:None ~io:None
    ~budget:max_int ~alloc ~free:ignore ~host_memory:None ~mapping:None
    ~queue:None ~peer:None ~load ~call:None ~link:None ~dma:None
    ~completion:Poll ~synchronized:ignore
    ~finalize:(fun ~failed:_ -> ())
    ~resolve:ignore

(* The disk's buffers are files, which it opens and never allocates. It has no
   processor. *)
let disk =
  let alloc _ = assert false in
  create ~name:"DISK" ~arch:"" ~machine:(Some host) ~remote:None ~io:None
    ~budget:max_int ~alloc ~free:ignore ~host_memory:None ~mapping:None
    ~queue:None ~peer:None ~load:None ~call:None ~link:None ~dma:None
    ~completion:Poll ~synchronized:ignore
    ~finalize:(fun ~failed:_ -> ())
    ~resolve:ignore

let name d = d.name
let arch d = d.arch
let equal = ( == )
let budget d = d.budget
let host_of d = match d.machine with Some h -> h | None -> d

let shares_host_memory d =
  d == host
  || host_of d == host
     && Option.is_none d.copy_queue
     && Option.is_some d.mapping

(* How the process reaches the memory of [d]'s machine: [None] on this one. *)
let io_of d = (host_of d).io

(* Timeline *)

let timeline_address d = Option.get d.timeline.host
let submitted d = Atomic.get d.last

let signaled d =
  match d.signal with
  | Some s -> s.signaled ()
  | None -> Int64.to_int (read_word (io_of d) (timeline_address d))

let commit d v = Atomic.set d.last v

(* A device that hung or faulted is in an unknown state: its first error loses
   it for good, and every later operation raises that error at once. *)
let lose d why = ignore (Atomic.compare_and_set d.failed None (Some why))

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

(* How long a wait sees the signal word still before it lets the device sleep on
   its interrupts. *)
let sleep_after_ms = 200

(* Polls the signal word for [v]. Once the word has stayed still for
   [sleep_after_ms], the device sleeps between polls, and once more before a
   hang is declared, so a fault it reports names the cause. The timeout counts
   from the word's last move. On this machine a wait blocks until the word
   moves; on another, each read of the word is a round trip. *)
let poll d v sleep =
  let io = io_of d in
  let word = timeline_address d and target = Int64.of_int v in
  let reached w = Int64.unsigned_compare w target >= 0 in
  let rec go seen still_since =
    let w = read_word io word and now = now_ms () in
    let still_since = if w <> seen then now else still_since in
    let still = now - still_since in
    let left = Atomic.get d.timeout_ms - still in
    if reached w then true
    else if left <= 0 then begin
      sleep 1;
      reached (read_word io word)
    end
    else if still < sleep_after_ms then
      Option.is_none io
      && wait_u64 word target (Int.min (sleep_after_ms - still) left) <> 0
      || go w still_since
    else begin
      sleep (Int.min sleep_after_ms left);
      go w still_since
    end
  in
  let w = read_word io word in
  go w (now_ms ())

(* Values complete in order: a wait for a value at or below one a wait saw
   signaled is over, with no read of the device's machine. *)
let rec settle d v =
  let s = Atomic.get d.settled in
  if v > s && not (Atomic.compare_and_set d.settled s v) then settle d v

let wait_signal d v =
  check d;
  if v > Atomic.get d.settled then
    match
      match (d.signal, d.sleep) with
      | Some s, _ -> s.wait v ~timeout_ms:(Atomic.get d.timeout_ms)
      | None, Some sleep -> poll d v sleep
      | None, None when Option.is_some (io_of d) -> poll d v ignore
      | None, None ->
          wait_u64 (timeline_address d) (Int64.of_int v)
            (Atomic.get d.timeout_ms)
          <> 0
    with
    | true -> settle d v
    | false -> fail d "hang detected"
    | exception Failure why -> fail d why

let failed d = Atomic.get d.failed

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

(* Waits for [d]'s work and for the work that touched [d]'s memory, then reads
   the stamps of the spans recorded on [d]. [d] is taken. A failed device will
   never signal, so its work is not waited for: the memory it can reach raises
   its error instead. *)
let sync d =
  wait_signal d (submitted d);
  Hashtbl.iter
    (fun _ (d', v) ->
      if failed d' = None then try wait_signal d' v with Lost _ -> ())
    d.pending;
  read_spans d;
  try d.synchronized () with Failure why -> fail d why

(* Raises [Lost] for a lost device that can reach [base]'s memory: its own
   device, a device it is mapped on or whose transfer into it could not be
   waited for, or those of the memory it maps. *)
let rec check_reach base =
  let { maps; reached } = Atomic.get base.links in
  List.iter check (base.owner :: (List.map (fun m -> m.on) maps @ reached));
  Option.iter (fun (src, _) -> check_reach src) base.source

let rec update_links base f =
  let l = Atomic.get base.links in
  if not (Atomic.compare_and_set base.links l (f l)) then update_links base f

let no_links = { maps = []; reached = [] }
let update_maps base f = update_links base (fun l -> { l with maps = f l.maps })

let mapping_on d base =
  List.find_opt (fun m -> m.on == d) (Atomic.get base.links).maps

(* The device's address of the host address [a] in the mapping [m]. *)
let mapped_address (m : region) a =
  Nativeint.add m.address (Nativeint.sub a (Option.get m.host))

(* Memory reclamation. Everything below runs with the device taken. *)

let release d b = push d.released b

(* Frees [memories], each with its function, once no work of [d] can use them.
   If that work cannot be waited for, or a free faults, which loses [d], the
   memory not yet freed is retained: kept with [keep], and never freed or
   reused, since its state is unknown. [owned] memory came from [d]'s
   allocators, and its bytes count as retained. *)
let free_all d ~owned ~keep memories =
  let retain rest =
    if owned then
      d.retained <-
        List.fold_left (fun n (_, (m : region)) -> n + m.nbytes) d.retained rest;
    d.held <- Keep (rest, keep) :: d.held
  in
  let rec go = function
    | [] -> ()
    | (free, m) :: tail as rest -> (
        match free m with
        | () -> go tail
        | exception Failure why ->
            retain rest;
            fail d why)
  in
  if memories <> [] then
    match sync d with
    | () -> go memories
    | exception (Lost _ as e) ->
        retain memories;
        raise e

let free_of d ~pinned =
  match d.host_memory with
  | Some (a : allocator) when pinned -> a.free
  | _ -> d.free

let fits d n = n <= d.budget - allocated d - d.cached - d.retained

(* Runs what other devices that map [m] registered, before [d] frees it: they
   unmap it. Memory one of them could not unmap is retained. [d] is
   synchronized. *)
let free_mapped d free size (m : region) =
  let peers =
    Mutex.protect d.peers_lock (fun () ->
        let l = Option.value ~default:[] (Hashtbl.find_opt d.peers m.address) in
        Hashtbl.remove d.peers m.address;
        l)
  in
  match List.iter (fun f -> f ()) peers with
  | () -> free m
  | exception Failure _ ->
      d.retained <- d.retained + size;
      d.held <- Keep m :: d.held

(* Frees cached memory to the system until [d] fits [n] more bytes, or its cache
   is empty. *)
let release_cache d n =
  if d.cached > 0 && not (fits d n) then begin
    let freed = ref [] in
    let keys = Hashtbl.fold (fun key _ acc -> key :: acc) d.cache [] in
    List.iter
      (fun ((size, pinned) as key) ->
        let free = free_mapped d (free_of d ~pinned) size in
        let rec drop = function
          | m :: ms when not (fits d n) ->
              d.cached <- d.cached - size;
              freed := (free, m) :: !freed;
              drop ms
          | ms -> ms
        in
        match drop (Hashtbl.find d.cache key) with
        | [] -> Hashtbl.remove d.cache key
        | ms -> Hashtbl.replace d.cache key ms)
      keys;
    free_all d ~owned:true ~keep:() !freed
  end

(* An unreachable program leaves the cache, unless a load replaced it there, and
   is released at once: only the host releases programs, and it runs them in the
   domains that call them, which keep them reachable. *)
let unload d =
  List.iter
    (fun { key; cell; unload } ->
      (match Hashtbl.find_opt d.programs key with
      | Some (Collectable c) when c == cell -> Hashtbl.remove d.programs key
      | _ -> ());
      unload ())
    (Atomic.exchange d.dropped [])

(* Unreachable owned memory returns to the cache without a wait: work is ordered
   after earlier work on the queue. A mapping is released once the last borrow
   of it is unreachable and the borrowing device's work is done: a mapping of
   host memory is unmapped, and one of another device's memory stays its
   driver's until that memory is freed. Host memory never comes here: it is the
   heap's, returned when the collector finds its base unreachable. *)
let reclaim d =
  unload d;
  match Atomic.exchange d.released [] with
  | [] -> ()
  | bases ->
      let emptied = ref [] and before = allocated d in
      List.iter
        (fun b ->
          match b.source with
          | Some (src, m) ->
              m.borrows <- m.borrows - 1;
              if m.borrows = 0 then emptied := (src, m) :: !emptied
          | None when Option.is_some b.file ->
              (* No read or write of a file outlives the copy that made it. *)
              file_close b.memory.handle
          | None when (Atomic.get b.links).reached <> [] ->
              (* Another device's work may still write it. *)
              allocate_bytes d (-b.bytes);
              d.retained <- d.retained + b.bytes;
              d.held <- Keep b :: d.held
          | None ->
              allocate_bytes d (-b.bytes);
              let key = (b.bytes, b.pinned) in
              let ms =
                Option.value ~default:[] (Hashtbl.find_opt d.cache key)
              in
              Hashtbl.replace d.cache key (b.memory :: ms);
              d.cached <- d.cached + b.bytes)
        bases;
      if allocated d <> before then memory_changed d;
      (match !emptied with
      | [] -> ()
      | emptied ->
          free_all d ~owned:false ~keep:bases
            (List.map (fun (_, m) -> (m.unmap, m.mapped)) emptied);
          (* Only once the device's work is done does the memory leave its
             reach: a failed wait above keeps the mappings in [maps]. *)
          List.iter
            (fun (src, m) -> update_maps src (List.filter (fun m' -> m' != m)))
            emptied);
      (* The memory under the borrows must outlive the wait in [free_all], which
         releases the runtime. *)
      ignore (Sys.opaque_identity bases);
      release_cache d 0

let take_cached d key =
  match Hashtbl.find_opt d.cache key with
  | Some (m :: ms) ->
      if ms = [] then Hashtbl.remove d.cache key
      else Hashtbl.replace d.cache key ms;
      d.cached <- d.cached - fst key;
      Some (m, Keep ())
  | Some [] | None -> None

(* An allocation the budget or the driver refuses releases the cache and tries
   again; one that is still refused collects the unreachable buffers, whose
   memory the collector cannot see, and tries once more. *)
let rec allocate d n ~pinned ~collected =
  if n > d.budget then raise (Out_of_memory (d, n));
  match take_cached d (n, pinned) with
  | Some m -> m
  | None -> (
      release_cache d n;
      let alloc n =
        match d.host_memory with
        | Some (a : allocator) when pinned ->
            Option.map (fun m -> (m, Keep ())) (a.alloc n)
        | _ -> d.alloc n
      in
      match if fits d n then driver d (fun () -> alloc n) else None with
      | Some (({ host = None; _ } as m), _)
        when pinned || Option.is_none d.copy_queue ->
          (* The host addresses a [Host_visible] device's memory and pinned
             memory: a region without a host address is a driver's bug. *)
          (free_of d ~pinned) m;
          invalid_arg
            (Printf.sprintf
               "Nx_device.Driver.device: %s's allocator gave memory the host \
                does not address"
               d.name)
      | Some m -> m
      | None when d.cached > 0 ->
          release_cache d max_int;
          allocate d n ~pinned ~collected
      | None when not collected ->
          Gc.full_major ();
          reclaim d;
          allocate d n ~pinned ~collected:true
      | None -> raise (Out_of_memory (d, n)))

(* Taking devices *)

let with_devices ds f =
  let ds = List.sort_uniq (fun a b -> Int.compare a.id b.id) ds in
  List.iter (fun d -> Mutex.lock d.lock) ds;
  Fun.protect
    ~finally:(fun () -> List.iter (fun d -> Mutex.unlock d.lock) (List.rev ds))
    (fun () ->
      List.iter check ds;
      List.iter reclaim ds;
      f ())

let synchronize d = with_devices [ d ] (fun () -> sync d)

let set_budget d n =
  if n < 0 then invalid_arg (Printf.sprintf "Nx_device.set_budget: %d < 0" n);
  with_devices [ d ] (fun () ->
      d.budget <- n;
      release_cache d 0)

let set_timeout d ms =
  if ms <= 0 then invalid_arg (Printf.sprintf "Nx_device.set_timeout: %d ms" ms);
  Atomic.set d.timeout_ms ms

let timeout d = Atomic.get d.timeout_ms

(* [fits d max_int] fails whenever [d] caches anything. *)
let free_cache d = with_devices [ d ] (fun () -> release_cache d max_int)

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

  type t = buffer = {
    base : base;
    offset : int; (* bytes into [base.memory] *)
    dtype : Nx_dtype.Scalar.t;
    length : int;
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
    match b.base.life with Dead why -> invalid_arg why | Live | Heir _ -> ()

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

  let base ?(bytes = 0) ?(pinned = false) ?source ?file ~borrowed ~keep d memory
      =
    {
      owner = d;
      memory;
      bytes;
      pinned;
      borrowed;
      keep;
      source;
      links = Atomic.make no_links;
      file;
      life = Live;
    }

  let empty ~borrowed d s n =
    let base = base ~borrowed ~keep:(Keep ()) d no_memory in
    { base; offset = 0; dtype = s; length = n }

  (* Host memory takes neither the host nor its release list: its bytes are
     reserved atomically against the budget and returned by the finaliser of a
     token the base keeps, which the collector runs where it frees the token, in
     the collection that finds the base unreachable. No wait is needed: a
     device's work reaches host memory only through a borrow, which keeps it
     alive. *)

  (* [n] reserved bytes of the heap. A refused reservation or allocation
     collects garbage once and tries again. *)
  let rec host_heap n ~collected =
    if heap_reserve n host.budget then (
      match heap n with
      | ba -> ba
      | exception Stdlib.Out_of_memory ->
          heap_return n;
          host_refused n ~collected)
    else host_refused n ~collected

  and host_refused n ~collected =
    if collected then raise (Out_of_memory (host, n));
    Gc.full_major ();
    host_heap n ~collected:true

  let not_files fn =
    invalid_arg
      (Printf.sprintf
         "Nx_device.%s: DISK buffers are files: open one with Buffer.of_file \
          or Buffer.create_file"
         fn)

  let create ?(pinned = false) d s n =
    if d == disk then not_files "Buffer.create";
    match checked_nbytes "Buffer.create" s n with
    | 0 -> empty ~borrowed:false d s n
    | bytes when d == host ->
        check host;
        if bytes > host.budget then raise (Out_of_memory (host, bytes));
        let ba = host_heap bytes ~collected:false in
        let base =
          {
            owner = host;
            memory = heap_memory ba;
            bytes;
            pinned = false;
            borrowed = false;
            keep = Heap (ba, heap_token bytes);
            source = None;
            links = Atomic.make no_links;
            file = None;
            life = Live;
          }
        in
        (* While a profile is taken, the return of the bytes is recorded too. *)
        if Atomic.get profile <> None then begin
          memory_changed host;
          Gc.finalise_last (fun () -> memory_changed host) base
        end;
        { base; offset = 0; dtype = s; length = n }
    | bytes ->
        let pinned = pinned && Option.is_some d.host_memory in
        let memory, keep =
          with_devices [ d ] (fun () ->
              let m = allocate d bytes ~pinned ~collected:false in
              allocate_bytes d bytes;
              memory_changed d;
              m)
        in
        let base = base ~bytes ~pinned ~borrowed:false ~keep d memory in
        Gc.finalise (release d) base;
        { base; offset = 0; dtype = s; length = n }

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
    let base = base ~borrowed:true ~keep:(Host ba) host (heap_memory ba) in
    { base; offset = 0; dtype; length = Bigarray.Array1.dim ba }

  let refuse fmt = Printf.ksprintf (fun why -> Error why) fmt

  (* A file is opened with the disk taken, so that the descriptors of the files
     already collected are closed first. An open refused for too many open files
     collects the unreachable buffers, whose descriptors the collector cannot
     see, and tries once more. *)
  let open_file path ~create n =
    let rec go ~collected =
      match file_open path create n with
      | 0, fd, size -> Ok (fd, size)
      | code, _, _ when code = too_many && not collected ->
          Gc.full_major ();
          reclaim disk;
          go ~collected:true
      | code, _, _ when code = not_regular ->
          refuse "%s: not a regular file" path
      | code, _, _ when code = too_many -> refuse "%s: too many open files" path
      | code, _, _ -> refuse "%s: %s" path (error_message code)
    in
    Result.map
      (fun (fd, size) ->
        let memory =
          { host = None; address = 0n; handle = fd; nbytes = size }
        in
        let base =
          base ~borrowed:true ~keep:(Keep ())
            ~file:{ path; writable = create; pages = None }
            disk memory
        in
        Gc.finalise (release disk) base;
        { base; offset = 0; dtype = Nx_dtype.Scalar.UInt8; length = size })
      (with_devices [ disk ] (fun () -> go ~collected:false))

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
            match file_map b.base.memory.handle b.base.memory.nbytes with
            | 0, ba ->
                let base =
                  base ~borrowed:true ~keep:(Host ba) host (heap_memory ba)
                in
                f.pages <- Some base;
                Ok base
            | code, _ -> refuse "%s: %s" f.path (error_message code)))

  (* [d]'s mapping of the memory of [src], made by its first borrow with [map]
     and shared by the later ones, and [skip] bytes into it. *)
  let share d src ~map ~unmap =
    with_devices [ d ] (fun () ->
        match mapping_on d src with
        | Some m ->
            m.borrows <- m.borrows + 1;
            Ok m
        | None ->
            Result.map
              (fun (mapped, skip) ->
                let m = { on = d; mapped; skip; unmap; borrows = 1 } in
                update_maps src (List.cons m);
                m)
              (driver d map))

  (* A buffer over the memory of [b] that [m] maps on [d]. *)
  let borrowed d b m =
    let base =
      base ~source:(b.base, m) ~borrowed:true ~keep:(Keep b) d m.mapped
    in
    Gc.finalise (release d) base;
    { base; offset = m.skip + b.offset; dtype = b.dtype; length = b.length }

  (* A device maps the whole host memory under [b], once, and its borrows share
     the mapping. A mapping locks whole pages, so it starts on one: host memory
     of another buffer then never shares its pages. *)
  let borrow_host d b =
    let h = host_of d and src = b.base in
    let first = Option.get src.memory.host in
    match d.mapping with
    | _ when d == h -> Ok b
    | None -> refuse "%s cannot address host memory" d.name
    (* A host buffer nx made starts on a page from [aligned_from] bytes; a
       smaller one is refused wherever it happens to start, so that whether it
       borrows does not depend on the allocator. *)
    | Some _
      when src.owner == host && (not src.borrowed)
           && src.memory.nbytes < aligned_from ->
        refuse
          "a host buffer of %d bytes does not start on a page, and %s maps \
           whole pages; host buffers start on one from %d bytes"
          src.memory.nbytes d.name aligned_from
    (* Another machine's pages are its own; its devices check them. *)
    | Some _ when h == host && Nativeint.rem first (Nativeint.of_int page) <> 0n
      ->
        refuse
          "the host memory at 0x%nx does not start on a page, and %s maps \
           whole pages; host buffers start on one from %d bytes"
          first d.name aligned_from
    | Some mapping -> (
        let map () =
          Result.map
            (fun (mapped : region) ->
              ( mapped,
                Nativeint.to_int (Nativeint.sub first (Option.get mapped.host))
              ))
            (mapping.map first src.memory.nbytes)
        in
        match share d src ~map ~unmap:mapping.unmap with
        | Error why ->
            refuse "%s cannot map the host memory at 0x%nx: %s" d.name first why
        | Ok m -> Ok (borrowed d b m))

  (* Another device's memory is mapped by [d]'s driver, which keeps the mapping
     until that memory is freed. *)
  let borrow_peer d b =
    let src = b.base in
    match d.peer with
    | None -> refuse "%s cannot address %s memory" d.name src.owner.name
    | Some peer -> (
        let map () =
          Result.map (fun mapped -> (mapped, 0)) (peer src.owner src.memory)
        in
        match share d src ~map ~unmap:ignore with
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
          if d != host then file_advise b.base.memory.handle b.offset (nbytes b);
          borrow_host d { b with base = pages })

  (* The host maps system memory of another device as the identity: the host's
     buffer is over the memory's host addresses, through a bigarray that owns
     nothing and a base that keeps [b], whose source it is so that the device
     can reach it. *)
  let borrow_system b =
    let src = b.base in
    let a = Option.get src.memory.host in
    let region = { src.memory with address = a } in
    let ba = external_bytes a region.nbytes in
    let mapped =
      { on = host; mapped = region; skip = 0; unmap = ignore; borrows = 1 }
    in
    let base =
      base ~source:(src, mapped) ~borrowed:true
        ~keep:(Device (ba, b))
        host region
    in
    { b with base }

  (* [b] over the memory its borrows map, down to memory no borrow holds. *)
  let rec root b =
    match b.base.source with
    | Some (src, m) -> root { b with base = src; offset = b.offset - m.skip }
    | None -> b

  (* Memory is borrowed by where it lives: the disk's through the file's pages,
     system memory (the host's, a [Host_visible] device's, and any device's
     pinned memory) through [d]'s mapping of host memory, and the memory of a
     [Device_local] device through [d]'s peer mapping, even where a BAR gives it
     a host address. *)
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
      else if r.base.pinned || Option.is_none o.copy_queue then
        if d == host then Ok (borrow_system r)
        else if Option.is_none d.machine then
          refuse "%s is reached over the network, and maps no memory" d.name
        else borrow_host d r
      else borrow_peer d r

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

  let spans b = b.offset = 0 && nbytes b = b.base.memory.nbytes

  (* The address space memory is addressed in: its machine's host's when the
     host addresses it, its device's otherwise, and its file's on the disk. *)
  type space = In_host of int | In_device of int | In_file of nativeint

  (* Where [b]'s bytes lie, below its borrows: a space and the first byte. *)
  let place b =
    let r = root b in
    let m = r.base.memory and o = r.base.owner in
    if o == disk then (In_file m.handle, Nativeint.of_int r.offset)
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

  let consume ~why b =
    live b;
    if not (spans b) then
      invalid_arg
        "Nx_device.Buffer.consume: the buffer is a window of its memory";
    let base = b.base in
    let heir = { base with life = Heir base } in
    base.life <- Dead why;
    { b with base = heir }

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
      match buf.base.keep with
      | Host ba -> bigarray_view ba k buf.offset (bytes / size)
      | Heap (ba, _) -> bigarray_view ba k buf.offset (bytes / size)
      | Device (ba, _) -> bigarray_view ba k buf.offset (bytes / size)
      | Keep _ -> assert false (* host memory is always a bigarray's *)

  (* Copies. The devices involved are taken and synchronized. A device's copy is
     work on its timeline, waited for at once. *)

  (* Bytes per slot of a host's staging memory, which has two. *)
  let chunk = 64 lsl 20

  (* This machine's staging memory, made at the first staged copy and kept for
     the life of the process. It is used with the host taken. *)
  let staging = ref None

  let staging_memory () =
    match !staging with
    | Some ba -> ba
    | None -> (
        match heap (2 * chunk) with
        | ba ->
            staging := Some ba;
            ba
        | exception Stdlib.Out_of_memory ->
            raise (Out_of_memory (host, 2 * chunk)))

  (* The address of the host [h]'s staging memory: this machine's, or memory of
     another machine's host, allocated there at its first staged copy and kept
     for the life of the process. It is used with [h] taken. *)
  let slots h =
    if h == host then bigarray_address (staging_memory ())
    else
      match h.slots with
      | Some m -> Option.get m.host
      | None -> (
          match driver h (fun () -> h.alloc (2 * chunk)) with
          | Some (m, _) ->
              h.slots <- Some m;
              Option.get m.host
          | None -> raise (Out_of_memory (h, 2 * chunk)))

  (* [e]'s address of its host's staging memory, which [e] maps at its first
     staged copy and keeps mapped. *)
  let staging_on e =
    match e.staging with
    | Some a -> a
    | None -> (
        let h = host_of e in
        let first = slots h in
        match (Option.get e.mapping).map first (2 * chunk) with
        | Ok m ->
            let a = mapped_address m first in
            e.staging <- Some a;
            a
        | Error why ->
            failwith
              (Printf.sprintf "%s cannot map %s's staging memory: %s" e.name
                 h.name why))

  let queue e =
    match e.copy_queue with
    | Some q -> q
    | None ->
        invalid_arg
          (Printf.sprintf "Nx_device.Buffer.copy: %s has no copy queue" e.name)

  (* The memory a host addresses. Another machine's is reached through the
     host's [io], whose error fails that host. *)
  let io_call h f = try f () with Failure msg -> fail h msg

  (* [n] bytes within the memory the host [h] addresses. *)
  let host_move h ~dst ~src n =
    match h.io with
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
      if i >= 2 then wait_signal e last.(i land 1);
      settled e (fun () -> fill (at +! slot i) (i * chunk) (length_of n i));
      last.(i land 1) <-
        enqueue_copy ~timed ~first:(i = 0)
          ~last:(i = chunks n - 1)
          e q
          (q.copy
             ~dst:(dst +! (i * chunk))
             ~src:(on_e +! slot i)
             (length_of n i))
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
          enqueue_copy ~timed ~first:(i = 0)
            ~last:(i = chunks n - 1)
            e q
            (q.copy
               ~dst:(on_e +! slot i)
               ~src:(src +! (i * chunk))
               (length_of n i))
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
      let first = i = 0 and final = i = chunks n - 1 in
      if i >= 2 then wait_signal d last.(i land 1);
      wait_signal s
        (enqueue_copy ~timed ~first ~last:final s qs
           (qs.copy
              ~dst:(on_s +! slot i)
              ~src:(address src +! (i * chunk))
              (length_of n i)));
      last.(i land 1) <-
        enqueue_copy ~timed ~first ~last:final d qd
          (qd.copy
             ~dst:(address dst +! (i * chunk))
             ~src:(on_d +! slot i)
             (length_of n i))
    done;
    wait_signal d (submitted d)

  let add_reached base e =
    update_links base (fun l -> { l with reached = e :: l.reached })

  (* Runs [f] with [b]'s address for [e]'s work, if [e] addresses [b]'s memory:
     its own, memory it maps, or pinned memory of another device, which [e] maps
     for the copy alone. If [f] raises, [e] may still use that memory: it stays
     mapped and in [e]'s reach. *)
  let with_address e b f =
    if b.base.owner == e then f (Some (address b))
    else
      match mapping_on e b.base with
      | Some m -> f (Some (m.mapped.address +! (m.skip + b.offset)))
      | None when b.base.pinned -> (
          let mapping = Option.get e.mapping in
          let first = Option.get b.base.memory.host in
          match driver e (fun () -> mapping.map first b.base.memory.nbytes) with
          | Error _ -> f None
          | Ok m -> (
              match f (Some (mapped_address m (Option.get (hosted b)))) with
              | r -> (
                  match mapping.unmap m with
                  | () -> r
                  | exception Failure why ->
                      add_reached b.base e;
                      fail e why)
              | exception e' ->
                  add_reached b.base e;
                  raise e'))
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
  let on_host ~timed ~first ~last b ~pos len k =
    match hosted b with
    | Some a -> a +! pos
    | None ->
        let e = device b in
        let q = queue e in
        wait_signal e
          (enqueue_copy ~timed ~first ~last e q
             (q.copy ~dst:(staging_on e +! slot k) ~src:(address b +! pos) len));
        slots (host_of e) +! slot k

  let landing b ~pos k =
    match hosted b with
    | Some a -> a +! pos
    | None -> slots (host_of (device b)) +! slot k

  let from_host ~timed ~first ~last b ~pos len k =
    if hosted b = None then
      let e = device b in
      let q = queue e in
      wait_signal e
        (enqueue_copy ~timed ~first ~last e q
           (q.copy ~dst:(address b +! pos) ~src:(staging_on e +! slot k) len))

  let across ~timed ~src ~dst n =
    let hs = host_of (device src) and hd = host_of (device dst) in
    for i = 0 to chunks n - 1 do
      let pos = i * chunk and len = length_of n i and k = i land 1 in
      let first = i = 0 and last = i = chunks n - 1 in
      let a = on_host ~timed ~first ~last src ~pos len k in
      let b = landing dst ~pos k in
      (match (hs.io, hd.io) with
      | None, None -> memmove b a len
      | None, Some io -> io_call hd (fun () -> io.write ~dst:b ~src:a len)
      | Some io, None -> io_call hs (fun () -> io.read ~src:a ~dst:b len)
      | Some io, Some io' ->
          let relay = bigarray_address (staging_memory ()) +! slot k in
          io_call hs (fun () -> io.read ~src:a ~dst:relay len);
          io_call hd (fun () -> io'.write ~dst:b ~src:relay len));
      from_host ~timed ~first ~last dst ~pos len k
    done

  (* [read b ~pos a n] reads the [n] bytes of the disk buffer [b] from its byte
     [pos] into host memory at [a]. *)
  let read b ~pos a n =
    let f = file_of b and at = b.offset + pos in
    match file_read b.base.memory.handle at a n with
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
    let code = file_write b.base.memory.handle (b.offset + pos) a n in
    if code < 0 then raise (Sys_error (f.path ^ ": " ^ error_message (-code)))

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
    | Transfer of copy
    | Bounce
    | Across
    | Link of link
    | File of device option
  (* from or to the disk, staged by the device whose copy queue it names *)

  (* The route of a copy of [src] into [dst], and the devices it takes besides
     the two: the hosts whose staging memory or [io] it uses, a link's
     devices. *)
  let route ~src ~dst =
    let s = device src and d = device dst in
    let h = host_of s in
    let remote = if Option.is_some h.io then [ h ] else [] in
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
          match (queue s).transfer d with
          | Some transfer -> (Transfer transfer, [])
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
    | Transfer transfer -> (
        match
          run ~timed s (queue s)
            (transfer ~dst:(address dst) ~src:(address src) n)
        with
        | () -> ()
        | exception (Lost _ as e) ->
            (* [s]'s copy engine may still write [dst]. *)
            add_reached dst.base s;
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
                add_reached dst.base t)
              l.through;
            List.iter check l.through;
            failwith why)
    | File _ -> file_copy ~timed ~src ~dst n

  (* A profiled copy is a span of the host on the calling domain's lane, and one
     on the copy lane of each device whose copy queue ran it, from its timestamp
     slots. *)
  let profiled c route ~src ~dst n =
    let s = device src and d = device dst in
    let start = now_ns () in
    move ~timed:true route ~src ~dst n;
    let stop = now_ns () in
    let name = s.name ^ " -> " ^ d.name in
    push c.events
      (Span { device = host; lane = domain_lane (); name; start; stop });
    List.iter
      (fun e ->
        push c.events
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
        reachable src;
        reachable dst;
        (if n > 0 then
           match Atomic.get profile with
           | None -> move ~timed:false route ~src ~dst n
           | Some c -> profiled c route ~src ~dst n);
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

(* Programs *)

module Program = struct
  type t = program

  let cached d key =
    match Hashtbl.find_opt d.programs key with
    | Some (Kept p) -> Some p
    | Some (Collectable cell) -> Weak.get cell 0
    | None -> None

  (* Caches [d]'s new program. The finaliser runs once the weak pointer is
     erased: no lookup can find a program whose release is queued. *)
  let add d ~binary ~name (handle, unload) =
    let key = (binary, name) in
    let p = { p_device = d; p_name = name; p_handle = handle } in
    (match unload with
    | None -> Hashtbl.replace d.programs key (Kept p)
    | Some unload ->
        let cell = Weak.create 1 in
        Weak.set cell 0 (Some p);
        Hashtbl.replace d.programs key (Collectable cell);
        Gc.finalise_last (fun () -> push d.dropped { key; cell; unload }) p);
    (match Atomic.get profile with
    | None -> ()
    | Some c -> push c.events (Load { program = p; binary; time = now_ns () }));
    p

  (* A loader that raises [Failure] loses its device. *)
  let load d ~binary ~name =
    match d.load with
    | None -> Error (d.name ^ ": the device loads no programs")
    | Some load ->
        with_devices [ d ] (fun () ->
            match cached d (binary, name) with
            | Some p -> Ok p
            | None -> (
                match load ~binary ~entry:name with
                | Ok loaded -> Ok (add d ~binary ~name loaded)
                | Error why -> Error (d.name ^ ": " ^ why)
                | exception Failure why -> fail d why))

  let device p = p.p_device
  let name p = p.p_name
  let handle p = p.p_handle

  external call_host : nativeint -> Buffer.t array -> int array -> unit
    = "caml_nx_device_call"

  let call p buffers values =
    let refuse fmt =
      Printf.ksprintf
        (fun m -> invalid_arg ("Nx_device.Program.call: " ^ m))
        fmt
    in
    let d = p.p_device in
    let call =
      match d.call with
      | Some call -> call
      | None when d == host -> fun _ _ _ -> ()
      | None -> refuse "the program is on %s, which runs no programs" d.name
    in
    Array.iter
      (fun (b : Buffer.t) ->
        if Option.is_none b.base.memory.host || host_of b.base.owner != d then
          refuse "%s does not address %s memory" d.name b.base.owner.name;
        Buffer.reachable b)
      buffers;
    if d == host then begin
      (match Atomic.get profile with
      | None -> call_host p.p_handle buffers values
      | Some c ->
          let start = now_ns () in
          call_host p.p_handle buffers values;
          let stop = now_ns () in
          push c.events
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
      let at b = (Option.get (Buffer.hosted b), Buffer.nbytes b) in
      try call p.p_handle (Array.map at buffers) values
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
        cached = d.cached;
        retained = d.retained;
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
    match (stamps.base.memory.host, Atomic.get profile) with
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
end

let by_id a b = Int.compare a.id b.id

(* The devices whose memory [b] reaches: its own, and those of the memory its
   borrows map. *)
let rec reach b =
  b.owner :: (match b.source with Some (src, _) -> reach src | None -> [])

let runs_work d = Option.is_some d.machine && d != disk

(* Whether the work of every device of [ds] can wait on [d']'s signal word: [d']
   stores its values into it, and each device of [ds] addresses it, as memory of
   its own machine that it maps. *)
let encodable ds d' =
  runs_work d' && Option.is_none d'.signal
  && List.for_all
       (fun d ->
         d == d' || (host_of d == host_of d' && Option.is_some d.mapping))
       ds

let submit ds ~touches f =
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
    List.sort_uniq by_id (List.concat_map (fun b -> reach b.base) touches)
  in
  let on =
    List.sort_uniq by_id
      (ds @ List.map (fun (b : buffer) -> b.base.owner) touches)
  in
  let taken = List.sort_uniq by_id (on @ reached) in
  with_devices taken @@ fun () ->
  List.iter Buffer.reachable touches;
  (* The latest value each device's work touched the reached memory with. *)
  let latest = Hashtbl.create 8 in
  let note (d', v) =
    (* A lost device's work never completes; the memory it can reach raises its
       loss instead. *)
    if failed d' = None then
      match Hashtbl.find_opt latest d'.id with
      | Some (_, v') when v' >= v -> ()
      | _ -> Hashtbl.replace latest d'.id (d', v)
  in
  List.iter
    (fun t ->
      if runs_work t then note (t, submitted t);
      Hashtbl.iter (fun _ p -> note p) t.pending)
    reached;
  let d_set = List.filter (encodable ds) on in
  (* A device's own earlier work is ordered by its vendor's rule. *)
  let alone d' = match ds with [ d ] -> d == d' | _ -> false in
  Hashtbl.iter
    (fun _ (d', v) ->
      if not (List.memq d' d_set || alone d') then wait_signal d' v)
    latest;
  let waits =
    List.map
      (fun d' ->
        match Hashtbl.find_opt latest d'.id with
        | Some (_, v) -> (d', v)
        | None -> (d', Atomic.get d'.settled))
      d_set
  in
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
  List.iter
    (fun t ->
      List.iteri
        (fun i d ->
          if t != d then Hashtbl.replace t.pending d.id (d, s.values.(i)))
        ds)
    reached;
  List.iter (fun (d, p) -> push d.spans p) (List.rev s.spans);
  r

let signal_word d =
  let owner = if Option.is_some d.host_memory then d else host_of d in
  let base =
    Buffer.base ~borrowed:true ~keep:d.timeline_keep owner d.timeline
  in
  { Buffer.base; offset = 0; dtype = Nx_dtype.Scalar.UInt64; length = 1 }

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

  type t = collector

  let now = now_ns
  let enabled () = Option.is_some (Atomic.get profile)

  let start () =
    let c = { events = Atomic.make [] } in
    if not (Atomic.compare_and_set profile None (Some c)) then
      invalid_arg "Nx_device.Profile.start: already profiling";
    c

  let span name f =
    match Atomic.get profile with
    | None -> f ()
    | Some c -> (
        let start = now_ns () in
        let record () =
          let stop = now_ns () in
          push c.events
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
        let q = Option.get d.copy_queue in
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

  let length = function
    | Span s -> s.stop - s.start
    | Allocation _ | Load _ -> 0

  (* By time, and at equal times longest first, so that nested spans follow the
     spans they are in. *)
  let order a b =
    match Int.compare (time a) (time b) with
    | 0 -> Int.compare (length b) (length a)
    | c -> c

  let stop p =
    let taken = Atomic.get profile in
    match taken with
    | Some c when c == p && Atomic.compare_and_set profile taken None ->
        List.iter
          (fun d ->
            if Atomic.get d.spans <> [] && failed d = None then
              try synchronize d with Lost _ -> ())
          (Atomic.get opened);
        let clocks = Hashtbl.create 4 in
        let calibrated d hz =
          match Hashtbl.find_opt clocks d.id with
          | Some f -> f
          | None ->
              let f = try Some (calibrate d hz) with Lost _ -> None in
              Hashtbl.add clocks d.id f;
              f
        in
        Atomic.get c.events
        |> List.filter_map (function
          | Span ({ device = { clock = Device_clock { hz }; _ } as d; _ } as s)
            ->
              Option.map
                (fun f -> Span { s with start = f s.start; stop = f s.stop })
                (calibrated d hz)
          | e -> Some e)
        |> List.stable_sort order
    | _ -> invalid_arg "Nx_device.Profile.stop: the profile is not being taken"

  (* Chrome's trace event format *)

  let device_of = function
    | Span s -> s.device
    | Allocation m -> m.device
    | Load p -> p.program.p_device

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
        let tid = match e with Span s -> tid s.device pid s.lane | _ -> 0 in
        next ();
        let ph =
          match e with Span _ -> "X" | Allocation _ -> "C" | Load _ -> "i"
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

  type nonrec mapping = mapping = {
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

  type memory =
    | Host_visible of { memory : allocator; mapping : mapping option }
    | Device_local of {
        memory : allocator;
        host_memory : allocator;
        mapping : mapping;
        queue : timeline:region -> queue;
      }

  type nonrec signal = signal = {
    signaled : unit -> int;
    wait : int -> timeout_ms:int -> bool;
  }

  type nonrec completion = completion =
    | Poll
    | Sleep of (timeline:region -> int -> unit)
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

  let default_timeout = default_timeout

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

  let owned (a : allocator) n = Option.map (fun m -> (m, Keep ())) (a.alloc n)

  let compose ?(host = host) local =
    match host.remote with Some a -> local ^ "@" ^ a | None -> local

  let name = compose

  let device ~name ~arch ~budget ?(host = host) ?(completion = Poll) ?load ?peer
      ?link ?dma ?(resolve = ignore) ?(synchronized = ignore)
      ?(finalize = fun ~failed:_ -> ()) memory =
    if budget < 0 then refuse "device" "budget %d < 0" budget;
    if Option.is_some host.machine then
      refuse "device" "%s is not a host" host.name;
    let memory, host_memory, mapping, queue =
      match memory with
      | Host_visible { memory; mapping } -> (memory, None, mapping, None)
      | Device_local { memory; host_memory; mapping; queue } ->
          (memory, Some host_memory, Some mapping, Some queue)
    in
    let queue =
      Option.map
        (fun queue ~timeline ->
          let q = queue ~timeline in
          (match q.clock with
          | Device_clock { hz } when hz <= 0 ->
              refuse "device" "a clock of %d Hz" hz
          | Host_clock | Device_clock _ -> ());
          q)
        queue
    in
    let load =
      Option.map
        (fun load ~binary ~entry ->
          Result.map (fun handle -> (handle, None)) (load ~binary ~entry))
        load
    in
    create ~name:(compose ~host name) ~arch ~machine:(Some host) ~remote:None
      ~io:None ~budget ~alloc:(owned memory) ~free:memory.free ~host_memory
      ~mapping ~queue ~peer ~load ~call:None ~link ~dma ~completion
      ~synchronized ~finalize ~resolve

  let buffer d (r : region) s n =
    if d == disk then Buffer.not_files "Driver.buffer";
    if d == host then
      refuse "buffer" "CPU memory is borrowed with Buffer.of_bigarray";
    let bytes = Buffer.checked_nbytes "Driver.buffer" s n in
    if bytes > r.nbytes then
      refuse "buffer" "%d bytes do not fit in a region of %d" bytes r.nbytes;
    let base = Buffer.base ~borrowed:true ~keep:(Keep ()) d r in
    { Buffer.base; offset = 0; dtype = s; length = n }

  let dma (b : Buffer.t) =
    let d = b.base.owner in
    match d.dma with
    | None -> Error (d.name ^ " does not describe its memory to others")
    | Some f -> driver d (fun () -> f b.base.memory)

  let on_free (b : Buffer.t) f =
    if b.base.borrowed || b.base.owner == host || b.base.bytes = 0 then
      refuse "on_free" "the memory is not one a device allocated";
    let d = b.base.owner and key = b.base.memory.address in
    Mutex.protect d.peers_lock (fun () ->
        let l = Option.value ~default:[] (Hashtbl.find_opt d.peers key) in
        Hashtbl.replace d.peers key (f :: l))

  (* Last: it shadows the host. *)
  let host ~address ~arch ?programs ?(synchronized = ignore)
      ?(finalize = fun ~failed:_ -> ()) ~memory io =
    let load =
      Option.map
        (fun p ~binary ~entry ->
          Result.map
            (fun (handle, unload) -> (handle, Some unload))
            (p.load ~binary ~entry))
        programs
    in
    create ~name:("CPU@" ^ address) ~arch ~machine:None ~remote:(Some address)
      ~io:(Some io) ~budget:max_int ~alloc:(owned memory) ~free:memory.free
      ~host_memory:None ~mapping:None ~queue:None ~peer:None ~load
      ~call:(Option.map (fun p -> p.call) programs)
      ~link:None ~dma:None ~completion:Poll ~synchronized ~finalize
      ~resolve:ignore
end
