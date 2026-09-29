(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The field order of [memory], [base] and [Buffer.t] up to the fields
   nx_device.h reads is its C ABI. *)
type memory = {
  host : nativeint option;
  device : nativeint;
  handle : nativeint;
}

type signal = { signaled : unit -> int; wait : int -> timeout_ms:int -> bool }
type allocator = { alloc : int -> memory option; free : memory -> unit }

type mapping = {
  map : nativeint -> int -> (memory, string) result;
  unmap : memory -> unit;
}

type copy = dst:nativeint -> src:nativeint -> int -> int -> unit
type file = { path : string; size : int; mtime : float; inode : int }

(* What must stay reachable for as long as a base does. Host memory is the
   bigarray that holds it from its first byte: views of it join that bigarray's
   storage, which outlives the base. *)
type keep =
  | Keep : 'a -> keep
  | Host : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> keep

type t = {
  id : int;
  name : string;
  arch : string;
  lock : Mutex.t;
  alloc : int -> (memory * keep) option;
  free : memory -> unit;
  host_memory : allocator option;
  mapping : mapping option;
  copy_queue : copy_queue option;
  load : (binary:string -> name:string -> nativeint) option;
  signal : signal option;
  timeout_ms : int Atomic.t;
  synchronized : unit -> unit;
  timeline : memory; (* [signaled; submitted] *)
  timeline_keep : keep;
  mutable staging : nativeint option;
      (* the device's address of the host's staging memory, once mapped *)
  released : base list Atomic.t;
  failed : string option Atomic.t;
      (* the error that failed the device, which every operation raises *)
  cache : (int * bool, memory list) Hashtbl.t;
      (* by size, and whether it is host memory *)
  pending : (int, t * int) Hashtbl.t;
      (* the devices whose work touched this one's memory, and the value that
         work signals *)
  programs : (string * string, program) Hashtbl.t;
  mutable held : keep list; (* retained memory, and what it keeps *)
  mutable budget : int;
  allocated : int Atomic.t;
  mutable cached : int;
  mutable retained : int;
  mutable bytes_in : int;
  mutable bytes_out : int;
}

and copy_queue = { copy : copy; transfer : t -> copy option }

and base = {
  owner : t;
  memory : memory;
  extent : int; (* bytes of [memory] from its first byte *)
  bytes : int; (* of owned memory, 0 when borrowed *)
  pinned : bool; (* allocated by the owner's [host_memory] *)
  borrowed : bool;
  keep : keep;
  source : (base * mapped) option;
      (* for a borrow, the host memory it maps, and the mapping *)
  links : links Atomic.t;
  file : file option; (* the file this memory maps, from its first byte *)
}

(* The other devices that reach a base's memory, changed together. *)
and links = {
  maps : mapped list; (* the mappings of this memory, one per device *)
  reached : t list;
      (* other devices whose work may still write this memory: the source of a
         transfer into it that could not be waited for *)
}

(* A mapping of a host base on a device, shared by the device's borrows of it.
   [borrows] changes only with the device taken. *)
and mapped = { on : t; mapped : memory; mutable borrows : int }
and program = { p_device : t; p_name : string; p_handle : nativeint }

exception Out_of_memory of t * int

let () =
  Printexc.register_printer (function
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

let page = page_size ()

(* [shared ba] is [ba] with the proxy of its storage made. The runtime makes a
   proxy on a bigarray's first sub without synchronization, so a keep must have
   one before views of it can be taken from several domains. *)
let shared ba = Bigarray.Array1.sub ba 0 (Bigarray.Array1.dim ba)

let heap_memory ba =
  let a = bigarray_address ba in
  { host = Some a; device = a; handle = 0n }

(* Host buffers of at least this many bytes start on a page, so that devices can
   map them: a mapping locks whole pages, which memory of another buffer must
   not share. Aligning costs up to a page of slack, at most a quarter of the
   buffer; smaller buffers are copied through staging instead. *)
let aligned_from = Int.max (64 * 1024) (4 * page)

(* [n] bytes of the heap. The sub that aligns them also makes their proxy. *)
let heap n =
  if n < aligned_from then
    shared (Bigarray.Array1.create Bigarray.char Bigarray.c_layout n)
  else if n > max_int - page then raise Stdlib.Out_of_memory
  else
    let ba =
      Bigarray.Array1.create Bigarray.char Bigarray.c_layout (n + page - 1)
    in
    let a = Nativeint.to_int (bigarray_address ba) in
    let skip = (page - (a mod page)) mod page in
    Bigarray.Array1.sub ba skip n

(* Devices *)

let ids = Atomic.make 0
let opened = Atomic.make []

let rec remember d =
  let l = Atomic.get opened in
  if not (Atomic.compare_and_set opened l (d :: l)) then remember d

(* The timeline is memory that the host and the device's work address: the
   device's host memory when it allocates some, the heap otherwise. It lives as
   long as the device. *)
let timeline_of (host_memory : allocator option) =
  match host_memory with
  | None ->
      let ba =
        shared (Bigarray.Array1.create Bigarray.int64 Bigarray.c_layout 2)
      in
      (heap_memory ba, Host ba)
  | Some a -> (
      match a.alloc 16 with
      | Some ({ host = Some _; _ } as m) -> (m, Keep ())
      | Some { host = None; _ } ->
          invalid_arg "Nx_device.make: host memory the host does not address"
      | None -> failwith "Nx_device.make: no memory for the timeline")

let create ~name ~arch ~budget ~alloc ~free ~host_memory ~mapping ~copy_queue
    ~load ~signal ~timeout_ms ~synchronized =
  let timeline, timeline_keep = timeline_of host_memory in
  let words = Option.get timeline.host in
  store_u64 words 0L;
  store_u64 (Nativeint.add words 8n) 0L;
  let d =
    {
      id = Atomic.fetch_and_add ids 1;
      name;
      arch;
      lock = Mutex.create ();
      alloc;
      free;
      host_memory;
      mapping;
      load;
      copy_queue = Option.map (fun copy_queue -> copy_queue timeline) copy_queue;
      signal = Option.map (fun signal -> signal timeline) signal;
      timeout_ms = Atomic.make timeout_ms;
      synchronized;
      timeline;
      timeline_keep;
      staging = None;
      released = Atomic.make [];
      failed = Atomic.make None;
      cache = Hashtbl.create 16;
      pending = Hashtbl.create 4;
      programs = Hashtbl.create 16;
      held = [];
      budget;
      allocated = Atomic.make 0;
      cached = 0;
      retained = 0;
      bytes_in = 0;
      bytes_out = 0;
    }
  in
  remember d;
  d

let host_arch = match Host_arch.architecture with "amd64" -> "x86_64" | a -> a
let default_timeout_ms = 30_000

let host =
  let alloc n =
    match heap n with
    | ba -> Some (heap_memory ba, Host ba)
    | exception Stdlib.Out_of_memory -> None
  in
  create ~name:"CPU" ~arch:host_arch ~budget:max_int ~alloc ~free:ignore
    ~host_memory:None ~mapping:None ~copy_queue:None ~load:None ~signal:None
    ~timeout_ms:default_timeout_ms ~synchronized:ignore

let name d = d.name
let arch d = d.arch
let equal = ( == )
let budget d = d.budget

(* Timeline *)

let timeline_address d = Option.get d.timeline.host

let submitted d =
  Int64.to_int (load_u64 (Nativeint.add (timeline_address d) 8n))

let signaled d =
  match d.signal with
  | Some s -> s.signaled ()
  | None -> Int64.to_int (load_u64 (timeline_address d))

(* A device that hung or faulted is in an unknown state: its first error fails
   it for good, and every later operation raises that error at once. *)
let fail d msg =
  ignore (Atomic.compare_and_set d.failed None (Some msg));
  failwith (Option.get (Atomic.get d.failed))

let check d = Option.iter failwith (Atomic.get d.failed)

let wait_signal d v =
  check d;
  match
    match d.signal with
    | Some s -> s.wait v ~timeout_ms:(Atomic.get d.timeout_ms)
    | None ->
        wait_u64 (timeline_address d) (Int64.of_int v) (Atomic.get d.timeout_ms)
        <> 0
  with
  | true -> ()
  | false -> fail d (d.name ^ " hang detected")
  | exception Failure msg -> fail d (d.name ^ ": " ^ msg)

let failed d = Atomic.get d.failed

(* Waits for [d]'s work and for the work that touched [d]'s memory. [d] is
   taken. A failed device will never signal, so its work is not waited for: the
   memory it can reach raises its error instead. *)
let sync d =
  wait_signal d (submitted d);
  Hashtbl.iter
    (fun _ (d', v) ->
      if failed d' = None then try wait_signal d' v with Failure _ -> ())
    d.pending;
  d.synchronized ()

(* The error of a failed device that can reach [base]'s memory: its own device,
   a device it is mapped on or whose transfer into it could not be waited for,
   or those of the memory it maps. *)
let rec failure_of base =
  let { maps; reached } = Atomic.get base.links in
  let reaching = base.owner :: (List.map (fun m -> m.on) maps @ reached) in
  match List.find_map failed reaching with
  | Some _ as e -> e
  | None -> Option.bind base.source (fun (src, _) -> failure_of src)

let rec update_links base f =
  let l = Atomic.get base.links in
  if not (Atomic.compare_and_set base.links l (f l)) then update_links base f

let update_maps base f = update_links base (fun l -> { l with maps = f l.maps })

let mapping_on d base =
  List.find_opt (fun m -> m.on == d) (Atomic.get base.links).maps

(* The device's address of the host address [a] in the mapping [m]. *)
let mapped_address (m : memory) a =
  Nativeint.add m.device (Nativeint.sub a (Option.get m.host))

(* Memory reclamation. Everything below runs with the device taken. *)

let rec release d b =
  let l = Atomic.get d.released in
  if not (Atomic.compare_and_set d.released l (b :: l)) then release d b

(* Frees [memories], each with its function, once no work of [d] can use them.
   If that work cannot be waited for, the memory is retained: kept with [keep],
   and never freed or reused, since its state is unknown. [owned] of its bytes
   came from [d]'s allocators. *)
let free_all d ~owned ~keep memories =
  if memories <> [] then
    match sync d with
    | () -> List.iter (fun (free, m) -> free m) memories
    | exception (Failure _ as e) ->
        d.retained <- d.retained + owned;
        d.held <- Keep (memories, keep) :: d.held;
        raise e

let free_of d ~pinned =
  match d.host_memory with
  | Some (a : allocator) when pinned -> a.free
  | _ -> d.free

let fits d n = n <= d.budget - Atomic.get d.allocated - d.cached - d.retained

(* Frees cached memory to the system until [d] fits [n] more bytes, or its cache
   is empty. *)
let release_cache d n =
  if d.cached > 0 && not (fits d n) then begin
    let freed = ref [] and bytes = ref 0 in
    let keys = Hashtbl.fold (fun key _ acc -> key :: acc) d.cache [] in
    List.iter
      (fun ((size, pinned) as key) ->
        let free = free_of d ~pinned in
        let rec drop = function
          | m :: ms when not (fits d n) ->
              d.cached <- d.cached - size;
              bytes := !bytes + size;
              freed := (free, m) :: !freed;
              drop ms
          | ms -> ms
        in
        match drop (Hashtbl.find d.cache key) with
        | [] -> Hashtbl.remove d.cache key
        | ms -> Hashtbl.replace d.cache key ms)
      keys;
    free_all d ~owned:!bytes ~keep:() !freed
  end

(* Unreachable owned memory returns to the cache without a wait: work is ordered
   after earlier work on the queue. A mapping is unmapped once the last borrow
   of it is unreachable and the borrowing device's work is done. Host memory
   never comes here: it is the heap's, returned when the collector finds its
   base unreachable. *)
let reclaim d =
  match Atomic.exchange d.released [] with
  | [] -> ()
  | bases ->
      let emptied = ref [] in
      List.iter
        (fun b ->
          match b.source with
          | Some (src, m) ->
              m.borrows <- m.borrows - 1;
              if m.borrows = 0 then emptied := (src, m) :: !emptied
          | None when (Atomic.get b.links).reached <> [] ->
              (* Another device's work may still write it. *)
              ignore (Atomic.fetch_and_add d.allocated (-b.bytes));
              d.retained <- d.retained + b.bytes;
              d.held <- Keep b :: d.held
          | None ->
              ignore (Atomic.fetch_and_add d.allocated (-b.bytes));
              let key = (b.bytes, b.pinned) in
              let ms =
                Option.value ~default:[] (Hashtbl.find_opt d.cache key)
              in
              Hashtbl.replace d.cache key (b.memory :: ms);
              d.cached <- d.cached + b.bytes)
        bases;
      (match !emptied with
      | [] -> ()
      | emptied ->
          let unmap = (Option.get d.mapping).unmap in
          free_all d ~owned:0 ~keep:bases
            (List.map (fun (_, m) -> (unmap, m.mapped)) emptied);
          (* Only once the device's work is done does the host memory leave its
             reach: a failed wait above keeps the mappings in [maps]. *)
          List.iter
            (fun (src, m) -> update_maps src (List.filter (fun m' -> m' != m)))
            emptied);
      (* The host memory under the borrows must outlive the wait in [free_all],
         which releases the runtime. *)
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
      match if fits d n then alloc n else None with
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

let () =
  at_exit (fun () ->
      List.iter
        (fun d ->
          if Atomic.get d.failed = None then
            try synchronize d
            with e ->
              Printf.eprintf "%s synchronization failed before exiting: %s\n%!"
                d.name (Printexc.to_string e))
        (Atomic.get opened))

(* Buffers *)

module Buffer = struct
  type nonrec file = file = {
    path : string;
    size : int;
    mtime : float;
    inode : int;
  }

  type t = {
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
    if n < 0 then
      invalid_arg (Printf.sprintf "Nx_device.Buffer.%s: %d elements" fn n);
    let size = Nx_dtype.Scalar.bitsize s / 8 in
    if size > 0 && n > max_int / size then
      invalid_arg
        (Printf.sprintf "Nx_device.Buffer.%s: %d elements of %s overflow" fn n
           (Nx_dtype.Scalar.to_string s));
    nbytes_of s n

  let nbytes b = nbytes_of b.dtype b.length
  let device b = b.base.owner
  let dtype b = b.dtype
  let length b = b.length
  let is_borrowed b = b.base.borrowed
  let ( +! ) a n = Nativeint.add a (Nativeint.of_int n)
  let address b = b.base.memory.device +! b.offset
  let host_of b = Option.map (fun a -> a +! b.offset) b.base.memory.host

  let host_address b =
    match host_of b with
    | Some a -> a
    | None ->
        invalid_arg
          (Printf.sprintf
             "Nx_device.Buffer.host_address: the host does not address %s \
              memory"
             b.base.owner.name)

  let handle b = b.base.memory.handle

  (* Raises the error of a failed device that can reach [b]'s memory. *)
  let reachable b = Option.iter failwith (failure_of b.base)
  let offset b = b.offset

  (* No byte of it is ever read or written, so the host addresses it. *)
  let no_memory = { host = Some 0n; device = 0n; handle = 0n }

  let base ?(bytes = 0) ?(pinned = false) ?source ?file ~borrowed ~keep ~extent
      d memory =
    {
      owner = d;
      memory;
      extent;
      bytes;
      pinned;
      borrowed;
      keep;
      source;
      links = Atomic.make { maps = []; reached = [] };
      file;
    }

  let empty ~borrowed d s n =
    let base = base ~borrowed ~keep:(Keep ()) ~extent:0 d no_memory in
    { base; offset = 0; dtype = s; length = n }

  (* Host memory takes neither the host nor its release list: its bytes are
     reserved atomically against the budget and returned by a finaliser that
     does not resurrect the base, so the memory is freed in the collection that
     finds it unreachable. No wait is needed: a device's work reaches host
     memory only through a borrow, which keeps it alive. *)
  let rec reserve n =
    let a = Atomic.get host.allocated in
    n <= host.budget - a
    && (Atomic.compare_and_set host.allocated a (a + n) || reserve n)

  (* [n] reserved bytes of the heap. A refused reservation or allocation
     collects garbage once and tries again. *)
  let rec host_heap n ~collected =
    if reserve n then (
      match heap n with
      | ba -> ba
      | exception Stdlib.Out_of_memory ->
          ignore (Atomic.fetch_and_add host.allocated (-n));
          host_refused n ~collected)
    else host_refused n ~collected

  and host_refused n ~collected =
    if collected then raise (Out_of_memory (host, n));
    Gc.full_major ();
    host_heap n ~collected:true

  let create ?host:(pinned = false) d s n =
    match checked_nbytes "create" s n with
    | 0 -> empty ~borrowed:false d s n
    | bytes when d == host ->
        check host;
        if bytes > host.budget then raise (Out_of_memory (host, bytes));
        let ba = host_heap bytes ~collected:false in
        let base =
          base ~bytes ~borrowed:false ~keep:(Host ba) ~extent:bytes d
            (heap_memory ba)
        in
        Gc.finalise_last
          (fun () -> ignore (Atomic.fetch_and_add host.allocated (-bytes)))
          base;
        { base; offset = 0; dtype = s; length = n }
    | bytes ->
        let pinned = pinned && Option.is_some d.host_memory in
        let memory, keep =
          with_devices [ d ] (fun () ->
              let m = allocate d bytes ~pinned ~collected:false in
              ignore (Atomic.fetch_and_add d.allocated bytes);
              m)
        in
        let base =
          base ~bytes ~pinned ~borrowed:false ~keep ~extent:bytes d memory
        in
        Gc.finalise (release d) base;
        { base; offset = 0; dtype = s; length = n }

  (* The bytes an element of [k] is aligned to: one component's for the complex
     kinds. *)
  let component_size (type a b) (k : (a, b) Bigarray.kind) =
    match k with
    | Bigarray.Complex32 -> 4
    | Bigarray.Complex64 -> 8
    | k -> Bigarray.kind_size_in_bytes k

  (* The format of the elements of [k]. *)
  let format_of_kind (type a b) (k : (a, b) Bigarray.kind) =
    match k with
    | Bigarray.Float16 -> Nx_dtype.Scalar.Float16
    | Bigarray.Float32 -> Float32
    | Bigarray.Float64 -> Float64
    | Bigarray.Int8_signed -> Int8
    | Bigarray.Int8_unsigned | Bigarray.Char -> UInt8
    | Bigarray.Int16_signed -> Int16
    | Bigarray.Int16_unsigned -> UInt16
    | Bigarray.Int32 -> Int32
    | Bigarray.Int64 -> Int64
    | Bigarray.Complex32 -> Complex64
    | Bigarray.Complex64 -> Complex128
    | Bigarray.Int | Bigarray.Nativeint ->
        invalid_arg
          "Nx_device.Buffer.of_bigarray: the kind is no storage format"

  let of_bigarray ?file ba =
    let dtype = format_of_kind (Bigarray.Array1.kind ba) in
    let extent = Bigarray.Array1.size_in_bytes ba in
    Option.iter
      (fun f ->
        if f.size <> extent then
          invalid_arg
            (Printf.sprintf
               "Nx_device.Buffer.of_bigarray: %s has %d bytes, the mapping %d"
               f.path f.size extent))
      file;
    (* A host buffer's elements lie at multiples of their size, as every typed
       read of it expects. *)
    let align = component_size (Bigarray.Array1.kind ba) in
    let address = bigarray_address ba in
    if extent > 0 && Nativeint.rem address (Nativeint.of_int align) <> 0n then
      invalid_arg
        (Printf.sprintf
           "Nx_device.Buffer.of_bigarray: the elements at 0x%nx are not \
            aligned to %d bytes"
           address align);
    let ba = shared ba in
    let base =
      base ?file ~borrowed:true ~keep:(Host ba) ~extent host (heap_memory ba)
    in
    { base; offset = 0; dtype; length = Bigarray.Array1.dim ba }

  let rec origin base =
    match base.file with
    | Some f -> Some (f, base)
    | None -> Option.bind base.source (fun (src, _) -> origin src)

  let file b =
    match (origin b.base, host_of b) with
    | Some (f, base), Some a ->
        Some
          (f, Nativeint.to_int (Nativeint.sub a (Option.get base.memory.host)))
    | _ -> None

  (* A device maps the whole host memory under [b], once, and its borrows share
     the mapping. A mapping locks whole pages, so it starts on one: host memory
     of another buffer then never shares its pages. *)
  let borrow d b =
    let fail fmt =
      Printf.ksprintf
        (fun m -> invalid_arg ("Nx_device.Buffer.borrow: " ^ m))
        fmt
    in
    if not (b.base.owner == host) then
      fail "the buffer is on %s, not CPU" b.base.owner.name;
    if d == host then b
    else
      match d.mapping with
      | None -> fail "%s cannot address host memory" d.name
      | Some _ when nbytes b = 0 -> empty ~borrowed:true d b.dtype b.length
      | Some mapping ->
          let src = b.base in
          let first = Option.get src.memory.host in
          if Nativeint.rem first (Nativeint.of_int page) <> 0n then
            fail
              "the host memory at 0x%nx does not start on a page, and %s maps \
               whole pages; host buffers start on one from %d bytes"
              first d.name aligned_from;
          let m =
            with_devices [ d ] (fun () ->
                match mapping_on d src with
                | Some m ->
                    m.borrows <- m.borrows + 1;
                    m
                | None -> (
                    match mapping.map first src.extent with
                    | Error why ->
                        fail "%s cannot map the host memory at 0x%nx: %s" d.name
                          first why
                    | Ok mapped ->
                        let m = { on = d; mapped; borrows = 1 } in
                        update_maps src (List.cons m);
                        m))
          in
          let skip =
            Nativeint.to_int (Nativeint.sub first (Option.get m.mapped.host))
          in
          let base =
            base ~source:(src, m) ~borrowed:true ~keep:(Keep b)
              ~extent:(skip + src.extent) d m.mapped
          in
          Gc.finalise (release d) base;
          { base; offset = skip + b.offset; dtype = b.dtype; length = b.length }

  let view b ~offset s n =
    let fail fmt =
      Printf.ksprintf (fun m -> invalid_arg ("Nx_device.Buffer.view: " ^ m)) fmt
    in
    if offset < 0 then fail "negative offset %d" offset;
    let bytes = checked_nbytes "view" s n in
    if offset > nbytes b - bytes then
      fail "%d bytes at offset %d do not fit in %d bytes" bytes offset
        (nbytes b);
    let size = Int.max 1 (Nx_dtype.Scalar.bitsize s / 8) in
    if Nativeint.rem (address b +! offset) (Nativeint.of_int size) <> 0n then
      fail "offset %d is not aligned to %s's %d bytes" offset
        (Nx_dtype.Scalar.to_string s)
        size;
    { b with offset = b.offset + offset; dtype = s; length = n }

  let bigarray (type a b) (k : (a, b) Bigarray.kind) buf :
      (a, b, Bigarray.c_layout) Bigarray.Array1.t =
    let fail fmt =
      Printf.ksprintf
        (fun m -> invalid_arg ("Nx_device.Buffer.bigarray: " ^ m))
        fmt
    in
    (match k with
    | Bigarray.Int | Bigarray.Nativeint -> fail "the kind is no storage format"
    | _ -> ());
    if not (buf.base.owner == host) then
      fail "the buffer is on %s, not CPU" buf.base.owner.name;
    reachable buf;
    let size = Bigarray.kind_size_in_bytes k and bytes = nbytes buf in
    let align = component_size k in
    if bytes mod size <> 0 then
      fail "%d bytes are not a whole number of %d-byte elements" bytes size;
    if Nativeint.rem (host_address buf) (Nativeint.of_int align) <> 0n then
      fail "the buffer is not aligned to %d bytes" align;
    if bytes = 0 then Bigarray.Array1.create k Bigarray.c_layout 0
    else
      match buf.base.keep with
      | Host ba -> bigarray_view ba k buf.offset (bytes / size)
      | Keep _ -> assert false (* host memory is always a bigarray's *)

  (* Copies. The devices involved are taken and synchronized. A device's copy is
     work on its timeline, waited for at once. *)

  (* Bytes per slot of the host's staging memory, which has two. *)
  let chunk = 64 lsl 20

  (* The host's staging memory, made at the first staged copy and kept for the
     life of the process. It is used with the host taken. *)
  let staging = ref None

  let staging_memory () =
    match !staging with
    | Some ba -> ba
    | None ->
        let ba = heap (2 * chunk) in
        staging := Some ba;
        ba

  (* [e]'s address of the host's staging memory, which [e] maps at its first
     staged copy and keeps mapped. *)
  let staging_on e =
    match e.staging with
    | Some a -> a
    | None -> (
        let first = bigarray_address (staging_memory ()) in
        match (Option.get e.mapping).map first (2 * chunk) with
        | Ok m ->
            let a = mapped_address m first in
            e.staging <- Some a;
            a
        | Error why ->
            failwith
              (Printf.sprintf "%s cannot map the host's staging memory: %s"
                 e.name why))

  let queue e =
    match e.copy_queue with
    | Some q -> q
    | None ->
        invalid_arg
          (Printf.sprintf "Nx_device.Buffer.copy: %s has no copy queue" e.name)

  (* A driver error while enqueueing leaves [e]'s queue in an unknown state:
     like a fault, it fails [e]. *)
  let enqueue e f =
    let v = submitted e + 1 in
    (try f v with Failure msg -> fail e (e.name ^ ": " ^ msg));
    store_u64 (timeline_address e +! 8) (Int64.of_int v);
    v

  let run e f = wait_signal e (enqueue e f)
  let chunks n = (n + chunk - 1) / chunk
  let length_of n i = Int.min chunk (n - (i * chunk))
  let slot i = i land 1 * chunk

  (* The host fills one slot while [e] copies the other. *)
  let stage_in e q ~src ~dst n =
    let host = bigarray_address (staging_memory ()) and on_e = staging_on e in
    let last = [| 0; 0 |] in
    for i = 0 to chunks n - 1 do
      if i >= 2 then wait_signal e last.(i land 1);
      memmove (host +! slot i) (src +! (i * chunk)) (length_of n i);
      last.(i land 1) <-
        enqueue e
          (q.copy
             ~dst:(dst +! (i * chunk))
             ~src:(on_e +! slot i)
             (length_of n i))
    done;
    wait_signal e (submitted e)

  (* [e] fills one slot while the host drains the other. *)
  let stage_out e q ~src ~dst n =
    let host = bigarray_address (staging_memory ()) and on_e = staging_on e in
    let last = [| 0; 0 |] in
    let fill i =
      if i < chunks n then
        last.(i land 1) <-
          enqueue e
            (q.copy
               ~dst:(on_e +! slot i)
               ~src:(src +! (i * chunk))
               (length_of n i))
    in
    fill 0;
    fill 1;
    for i = 0 to chunks n - 1 do
      wait_signal e last.(i land 1);
      memmove (dst +! (i * chunk)) (host +! slot i) (length_of n i);
      fill (i + 2)
    done

  (* Between devices that cannot reach each other, the bytes go through the
     host's staging memory: [s] fills one slot while [d] drains the other. *)
  let bounce s ~src d ~dst n =
    let qs = queue s and qd = queue d in
    let on_s = staging_on s and on_d = staging_on d in
    let last = [| 0; 0 |] in
    for i = 0 to chunks n - 1 do
      if i >= 2 then wait_signal d last.(i land 1);
      run s
        (qs.copy
           ~dst:(on_s +! slot i)
           ~src:(address src +! (i * chunk))
           (length_of n i));
      last.(i land 1) <-
        enqueue d
          (qd.copy
             ~dst:(address dst +! (i * chunk))
             ~src:(on_d +! slot i)
             (length_of n i))
    done;
    wait_signal d (submitted d)

  let add_reached base e =
    update_links base (fun l -> { l with reached = e :: l.reached })

  (* Runs [f] with [b]'s address for [e]'s work, if [e] addresses [b]'s memory:
     its own, host memory it maps, or host memory of another device, which [e]
     maps for the copy alone. If [f] raises, [e] may still use that memory: it
     stays mapped and in [e]'s reach. *)
  let with_address e b f =
    if b.base.owner == e then f (Some (address b))
    else
      match mapping_on e b.base with
      | Some m -> f (Some (mapped_address m.mapped (host_address b)))
      | None when b.base.pinned -> (
          let mapping = Option.get e.mapping in
          let first = Option.get b.base.memory.host in
          match mapping.map first b.base.extent with
          | Error _ -> f None
          | Ok m -> (
              match f (Some (mapped_address m (host_address b))) with
              | r ->
                  mapping.unmap m;
                  r
              | exception e' ->
                  add_reached b.base e;
                  raise e'))
      | None -> f None

  (* [e] copies [src], which the host addresses, into [dst], its memory. *)
  let into e ~src ~dst n =
    let q = queue e in
    with_address e src (function
      | Some a -> run e (q.copy ~dst:(address dst) ~src:a n)
      | None -> stage_in e q ~src:(host_address src) ~dst:(address dst) n)

  (* [e] copies [src], its memory, into [dst], which the host addresses or [e]
     does. *)
  let out_of e ~src ~dst n =
    let q = queue e in
    with_address e dst (function
      | Some a -> run e (q.copy ~dst:a ~src:(address src) n)
      | None -> stage_out e q ~src:(address src) ~dst:(host_address dst) n)

  type route = Host_copy | Into | Out_of | Transfer of copy | Bounce

  (* The route of a copy of [src] into [dst], and whether it may use the host's
     staging memory. *)
  let route ~src ~dst =
    let s = device src and d = device dst in
    match (host_of src, host_of dst) with
    | Some _, Some _ -> (Host_copy, false)
    | Some _, None -> (Into, src.base.owner != d)
    | None, Some _ -> (Out_of, dst.base.owner != s)
    | None, None when s == d -> (Out_of, false)
    | None, None -> (
        match (queue s).transfer d with
        | Some transfer -> (Transfer transfer, false)
        | None -> (Bounce, true))

  let move route ~src ~dst n =
    let s = device src and d = device dst in
    match route with
    | Host_copy -> memmove (host_address dst) (host_address src) n
    | Into -> into d ~src ~dst n
    | Out_of -> out_of s ~src ~dst n
    | Transfer transfer -> (
        match run s (transfer ~dst:(address dst) ~src:(address src) n) with
        | () -> ()
        | exception (Failure _ as e) ->
            (* [s]'s copy engine may still write [dst]. *)
            add_reached dst.base s;
            raise e)
    | Bounce -> bounce s ~src d ~dst n

  let copy ~src ~dst =
    let fail fmt =
      Printf.ksprintf (fun m -> invalid_arg ("Nx_device.Buffer.copy: " ^ m)) fmt
    in
    let n = nbytes src in
    if n <> nbytes dst then fail "%d bytes into %d bytes" n (nbytes dst);
    if
      n > 0 && src.base == dst.base
      && src.offset < dst.offset + n
      && dst.offset < src.offset + n
    then fail "the source and destination overlap";
    let s = device src and d = device dst in
    let route, staged = route ~src ~dst in
    with_devices
      (if staged then [ s; d; host ] else [ s; d ])
      (fun () ->
        sync s;
        if d != s then sync d;
        reachable src;
        reachable dst;
        if n > 0 then move route ~src ~dst n;
        if d != s then begin
          s.bytes_out <- s.bytes_out + n;
          d.bytes_in <- d.bytes_in + n
        end);
    (* The copy runs on addresses with the runtime released: the buffers, and
       the memory they keep, must stay reachable until it returns. *)
    ignore (Sys.opaque_identity src);
    ignore (Sys.opaque_identity dst)
end

(* Programs *)

module Program = struct
  type t = program

  let load d ~binary ~name =
    match d.load with
    | None ->
        invalid_arg
          (Printf.sprintf "Nx_device.Program.load: %s loads no programs" d.name)
    | Some load ->
        with_devices [ d ] (fun () ->
            match Hashtbl.find_opt d.programs (binary, name) with
            | Some p -> p
            | None ->
                let p =
                  { p_device = d; p_name = name; p_handle = load ~binary ~name }
                in
                Hashtbl.add d.programs (binary, name) p;
                p)

  let device p = p.p_device
  let name p = p.p_name
  let handle p = p.p_handle
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
      (if failed d = None then try reclaim d with Failure _ -> ());
      {
        Stats.allocated = Atomic.get d.allocated;
        cached = d.cached;
        retained = d.retained;
        bytes_in = d.bytes_in;
        bytes_out = d.bytes_out;
      })

(* Submitting work *)

let submit d ~touches f =
  with_devices (d :: touches) (fun () ->
      let v = submitted d + 1 in
      let r = f v in
      store_u64 (Nativeint.add (timeline_address d) 8n) (Int64.of_int v);
      List.iter
        (fun t -> if t != d then Hashtbl.replace t.pending d.id (d, v))
        touches;
      r)

let timeline d =
  let owner = if Option.is_some d.host_memory then d else host in
  let base =
    Buffer.base ~borrowed:true ~keep:d.timeline_keep ~extent:16 owner d.timeline
  in
  { Buffer.base; offset = 0; dtype = Nx_dtype.Scalar.UInt64; length = 2 }

(* Vendor runtimes *)

let make ~name ~arch ~budget ~(memory : allocator) ?host_memory ?mapping
    ?copy_queue ?load ?signal ?(timeout_ms = default_timeout_ms)
    ?(synchronized = ignore) () =
  let fail fmt =
    Printf.ksprintf (fun m -> invalid_arg ("Nx_device.make: " ^ m)) fmt
  in
  if budget < 0 then fail "budget %d < 0" budget;
  if timeout_ms <= 0 then fail "timeout %d ms" timeout_ms;
  if Option.is_some copy_queue && Option.is_none mapping then
    fail "%s has a copy queue but maps no host memory" name;
  let alloc n = Option.map (fun m -> (m, Keep ())) (memory.alloc n) in
  create ~name ~arch ~budget ~alloc ~free:memory.free ~host_memory ~mapping
    ~copy_queue ~load ~signal ~timeout_ms ~synchronized

let external_buffer d m s n =
  if d == host then
    invalid_arg
      "Nx_device.external_buffer: CPU memory is borrowed with \
       Buffer.of_bigarray";
  let bytes = Buffer.checked_nbytes "external_buffer" s n in
  let base = Buffer.base ~borrowed:true ~keep:(Keep ()) ~extent:bytes d m in
  { Buffer.base; offset = 0; dtype = s; length = n }
