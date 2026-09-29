(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type memory = { host : nativeint; device : nativeint; handle : nativeint }
type signal = { signaled : unit -> int; wait : int -> timeout_ms:int -> bool }

(* A value that must stay reachable for as long as a base does. *)
type keep = Keep : 'a -> keep

type t = {
  id : int;
  name : string;
  arch : string;
  lock : Mutex.t;
  caches : bool;
  alloc : int -> (memory * keep) option;
  free : memory -> unit;
  map : (nativeint -> int -> memory option) option;
  load : (binary:string -> name:string -> nativeint) option;
  signal : signal option;
  timeout_ms : int;
  synchronized : unit -> unit;
  timeline : (int64, Bigarray.int64_elt, Bigarray.c_layout) Bigarray.Array1.t;
      (* [signaled; submitted] *)
  released : base list Atomic.t;
  cache : (int, memory list) Hashtbl.t;
  pending : (int, t * int) Hashtbl.t;
      (* the devices whose work touched this one's memory, and the value that
         work signals *)
  programs : (string * string, program) Hashtbl.t;
  mutable held : keep list; (* retained memory, and what it keeps *)
  mutable budget : int;
  mutable allocated : int;
  mutable cached : int;
  mutable retained : int;
  mutable bytes_in : int;
  mutable bytes_out : int;
}

and base = {
  owner : t;
  memory : memory;
  bytes : int; (* of owned memory, 0 when borrowed *)
  borrowed : bool;
  keep : keep;
}

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

external wait_u64 :
  (nativeint[@unboxed]) ->
  (int64[@unboxed]) ->
  (int[@untagged]) ->
  (int[@untagged]) = "caml_nx_device_wait_u64_byte" "caml_nx_device_wait_u64"

let host_memory ba =
  let a = bigarray_address ba in
  { host = a; device = a; handle = 0n }

(* Devices *)

let ids = Atomic.make 0
let opened = Atomic.make []

let rec remember d =
  let l = Atomic.get opened in
  if not (Atomic.compare_and_set opened l (d :: l)) then remember d

let create ~name ~arch ~budget ~caches ~alloc ~free ~map ~load ~signal
    ~timeout_ms ~synchronized =
  let timeline = Bigarray.Array1.create Bigarray.int64 Bigarray.c_layout 2 in
  Bigarray.Array1.fill timeline 0L;
  let d =
    {
      id = Atomic.fetch_and_add ids 1;
      name;
      arch;
      lock = Mutex.create ();
      caches;
      alloc;
      free;
      map;
      load;
      signal;
      timeout_ms;
      synchronized;
      timeline;
      released = Atomic.make [];
      cache = Hashtbl.create 16;
      pending = Hashtbl.create 4;
      programs = Hashtbl.create 16;
      held = [];
      budget;
      allocated = 0;
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
    match Bigarray.Array1.create Bigarray.char Bigarray.c_layout n with
    | ba -> Some (host_memory ba, Keep ba)
    | exception Stdlib.Out_of_memory -> None
  in
  create ~name:"CPU" ~arch:host_arch ~budget:max_int ~caches:false ~alloc
    ~free:ignore ~map:None ~load:None ~signal:None
    ~timeout_ms:default_timeout_ms ~synchronized:ignore

let name d = d.name
let arch d = d.arch
let equal = ( == )
let budget d = d.budget

(* Timeline *)

let timeline_address d = bigarray_address d.timeline

let submitted d =
  Int64.to_int (load_u64 (Nativeint.add (timeline_address d) 8n))

let signaled d =
  match d.signal with
  | Some s -> s.signaled ()
  | None -> Int64.to_int (load_u64 (timeline_address d))

let wait_signal d v =
  let signaled =
    match d.signal with
    | Some s -> s.wait v ~timeout_ms:d.timeout_ms
    | None -> wait_u64 (timeline_address d) (Int64.of_int v) d.timeout_ms <> 0
  in
  if not signaled then failwith (d.name ^ " hang detected")

(* Waits for [d]'s work and for the work that touched [d]'s memory. [d] is
   taken. *)
let sync d =
  wait_signal d (submitted d);
  Hashtbl.iter (fun _ (d', v) -> wait_signal d' v) d.pending;
  d.synchronized ()

(* Memory reclamation. Everything below runs with the device taken. *)

let rec release d b =
  let l = Atomic.get d.released in
  if not (Atomic.compare_and_set d.released l (b :: l)) then release d b

(* Frees [memories] to the driver once no work of [d] can use them. If that work
   cannot be waited for, the memory is retained: kept with [keep], and never
   freed or reused, since its state is unknown. [owned] of its bytes came from
   [d]'s allocator. *)
let free_all d ~owned ~keep memories =
  if memories <> [] then
    match sync d with
    | () -> List.iter d.free memories
    | exception Failure _ ->
        d.retained <- d.retained + owned;
        d.held <- Keep (memories, keep) :: d.held

let fits d n = n <= d.budget - d.allocated - d.cached - d.retained

(* Frees cached memory to the system until [d] fits [n] more bytes, or its cache
   is empty. *)
let release_cache d n =
  if d.cached > 0 && not (fits d n) then begin
    let freed = ref [] and bytes = ref 0 in
    let sizes = Hashtbl.fold (fun size _ acc -> size :: acc) d.cache [] in
    List.iter
      (fun size ->
        let rec drop = function
          | m :: ms when not (fits d n) ->
              d.cached <- d.cached - size;
              bytes := !bytes + size;
              freed := m :: !freed;
              drop ms
          | ms -> ms
        in
        match drop (Hashtbl.find d.cache size) with
        | [] -> Hashtbl.remove d.cache size
        | ms -> Hashtbl.replace d.cache size ms)
      sizes;
    free_all d ~owned:!bytes ~keep:() !freed
  end

(* Unreachable owned memory returns to the cache, or to the heap on the host,
   without a wait: work is ordered after earlier work on the queue, and host
   memory that a device's work uses is borrowed by that device. A borrow is
   unmapped once the borrowing device's work is done. *)
let reclaim d =
  match Atomic.exchange d.released [] with
  | [] -> ()
  | bases ->
      let borrowed = List.filter (fun b -> b.borrowed) bases in
      List.iter
        (fun b ->
          if not b.borrowed then begin
            d.allocated <- d.allocated - b.bytes;
            if d.caches then begin
              let ms =
                Option.value ~default:[] (Hashtbl.find_opt d.cache b.bytes)
              in
              Hashtbl.replace d.cache b.bytes (b.memory :: ms);
              d.cached <- d.cached + b.bytes
            end
          end)
        bases;
      free_all d ~owned:0 ~keep:borrowed (List.map (fun b -> b.memory) borrowed);
      (* The host memory under the borrows must outlive the wait in [free_all],
         which releases the runtime. *)
      List.iter (fun b -> ignore (Sys.opaque_identity b.keep)) borrowed;
      release_cache d 0

let take_cached d n =
  match Hashtbl.find_opt d.cache n with
  | Some (m :: ms) ->
      if ms = [] then Hashtbl.remove d.cache n else Hashtbl.replace d.cache n ms;
      d.cached <- d.cached - n;
      Some (m, Keep ())
  | Some [] | None -> None

(* An allocation the budget or the driver refuses releases the cache and tries
   again; one that is still refused collects the unreachable buffers, whose
   memory the collector cannot see, and tries once more. *)
let rec allocate d n ~collected =
  if n > d.budget then raise (Out_of_memory (d, n));
  match take_cached d n with
  | Some m -> m
  | None -> (
      release_cache d n;
      match if fits d n then d.alloc n else None with
      | Some m -> m
      | None when d.cached > 0 ->
          release_cache d max_int;
          allocate d n ~collected
      | None when not collected ->
          Gc.full_major ();
          reclaim d;
          allocate d n ~collected:true
      | None -> raise (Out_of_memory (d, n)))

(* Taking devices *)

let with_devices ds f =
  let ds = List.sort_uniq (fun a b -> Int.compare a.id b.id) ds in
  List.iter (fun d -> Mutex.lock d.lock) ds;
  Fun.protect
    ~finally:(fun () -> List.iter (fun d -> Mutex.unlock d.lock) (List.rev ds))
    (fun () ->
      List.iter reclaim ds;
      f ())

let synchronize d = with_devices [ d ] (fun () -> sync d)

let set_budget d n =
  if n < 0 then invalid_arg (Printf.sprintf "Nx_device.set_budget: %d < 0" n);
  with_devices [ d ] (fun () ->
      d.budget <- n;
      release_cache d 0)

(* [fits d max_int] fails whenever [d] caches anything. *)
let free_cache d = with_devices [ d ] (fun () -> release_cache d max_int)

let () =
  at_exit (fun () ->
      List.iter
        (fun d ->
          try synchronize d
          with e ->
            Printf.eprintf "%s synchronization failed before exiting: %s\n%!"
              d.name (Printexc.to_string e))
        (Atomic.get opened))

(* Buffers *)

module Buffer = struct
  type t = {
    base : base;
    offset : int; (* bytes into [base.memory] *)
    dtype : Nx_dtype.Scalar.t;
    length : int;
  }

  let nbytes_of s n = ((n * Nx_dtype.Scalar.bitsize s) + 7) / 8

  (* The bytes of [n] elements of [s], for a new buffer or view. *)
  let checked_nbytes fn s n =
    if n < 0 then
      invalid_arg (Printf.sprintf "Nx_device.Buffer.%s: %d elements" fn n);
    if n > (max_int - 7) / Nx_dtype.Scalar.bitsize s then
      invalid_arg
        (Printf.sprintf "Nx_device.Buffer.%s: %d elements of %s overflow" fn n
           (Nx_dtype.Scalar.to_string s));
    nbytes_of s n

  let nbytes b = nbytes_of b.dtype b.length
  let device b = b.base.owner
  let dtype b = b.dtype
  let length b = b.length
  let is_borrowed b = b.base.borrowed
  let address b = Nativeint.add b.base.memory.device (Nativeint.of_int b.offset)

  let host_address b =
    Nativeint.add b.base.memory.host (Nativeint.of_int b.offset)

  let handle b = b.base.memory.handle
  let offset b = b.offset
  let no_memory = { host = 0n; device = 0n; handle = 0n }

  let empty ~borrowed d s n =
    let base =
      { owner = d; memory = no_memory; bytes = 0; borrowed; keep = Keep () }
    in
    { base; offset = 0; dtype = s; length = n }

  let create d s n =
    match checked_nbytes "create" s n with
    | 0 -> empty ~borrowed:false d s n
    | bytes ->
        let memory, keep =
          with_devices [ d ] (fun () ->
              let m = allocate d bytes ~collected:false in
              d.allocated <- d.allocated + bytes;
              m)
        in
        let base = { owner = d; memory; bytes; borrowed = false; keep } in
        Gc.finalise (release d) base;
        { base; offset = 0; dtype = s; length = n }

  let of_bigarray ba =
    let base =
      {
        owner = host;
        memory = host_memory ba;
        bytes = 0;
        borrowed = true;
        keep = Keep ba;
      }
    in
    {
      base;
      offset = 0;
      dtype = Nx_dtype.Scalar.UInt8;
      length = Bigarray.Array1.dim ba;
    }

  let borrow d b =
    if not (b.base.owner == host) then
      invalid_arg
        (Printf.sprintf "Nx_device.Buffer.borrow: the buffer is on %s, not CPU"
           b.base.owner.name);
    let cannot () =
      invalid_arg
        (Printf.sprintf "Nx_device.Buffer.borrow: %s cannot address host memory"
           d.name)
    in
    if d == host then b
    else
      match d.map with
      | None -> cannot ()
      | Some _ when nbytes b = 0 -> empty ~borrowed:true d b.dtype b.length
      | Some map -> (
          let a = host_address b and bytes = nbytes b in
          match with_devices [ d ] (fun () -> map a bytes) with
          | None -> cannot ()
          | Some memory ->
              let offset = Nativeint.to_int (Nativeint.sub a memory.host) in
              let base =
                { owner = d; memory; bytes = 0; borrowed = true; keep = Keep b }
              in
              Gc.finalise (release d) base;
              { base; offset; dtype = b.dtype; length = b.length })

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
    let first = Nativeint.add (host_address b) (Nativeint.of_int offset) in
    if Nativeint.rem first (Nativeint.of_int size) <> 0n then
      fail "offset %d is not aligned to %s's %d bytes" offset
        (Nx_dtype.Scalar.to_string s)
        size;
    { b with offset = b.offset + offset; dtype = s; length = n }

  let copy ~src ~dst =
    let n = nbytes src in
    if n <> nbytes dst then
      invalid_arg
        (Printf.sprintf "Nx_device.Buffer.copy: %d bytes into %d bytes" n
           (nbytes dst));
    let s = device src and d = device dst in
    with_devices [ s; d ] (fun () ->
        sync s;
        if d != s then sync d;
        if n > 0 then memmove (host_address dst) (host_address src) n;
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

let stats d =
  with_devices [ d ] (fun () ->
      {
        Stats.allocated = d.allocated;
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
  let base =
    {
      owner = host;
      memory = host_memory d.timeline;
      bytes = 0;
      borrowed = true;
      keep = Keep d.timeline;
    }
  in
  { Buffer.base; offset = 0; dtype = Nx_dtype.Scalar.UInt64; length = 2 }

(* Vendor runtimes *)

let make ~name ~arch ~budget ~alloc ~free ?borrow ?load ?signal
    ?(timeout_ms = default_timeout_ms) ?(synchronized = ignore) () =
  if budget < 0 then
    invalid_arg (Printf.sprintf "Nx_device.make: budget %d < 0" budget);
  if timeout_ms <= 0 then
    invalid_arg (Printf.sprintf "Nx_device.make: timeout %d ms" timeout_ms);
  let alloc n = Option.map (fun m -> (m, Keep ())) (alloc n) in
  create ~name ~arch ~budget ~caches:true ~alloc ~free ~map:borrow ~load ~signal
    ~timeout_ms ~synchronized
