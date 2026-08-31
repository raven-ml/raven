(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

module Ffi = struct
  type constants = {
    o_rdonly : int;
    o_rdwr : int;
    prot_none : int;
    prot_read : int;
    prot_write : int;
    map_shared : int;
    map_private : int;
    map_anonymous : int;
    map_fixed : int;
    map_noreserve : int;
  }

  external constants : unit -> constants = "caml_tolk_hcq_constants"
  external openfile : string -> int -> int = "caml_tolk_hcq_open"
  external close : int -> unit = "caml_tolk_hcq_close"

  external mmap : nativeint -> int -> int -> int -> int -> int64 -> nativeint
    = "caml_tolk_hcq_mmap_bc" "caml_tolk_hcq_mmap"

  external munmap : nativeint -> int -> unit = "caml_tolk_hcq_munmap"
  external read32 : nativeint -> int32 = "caml_tolk_hcq_read32"

  external write32 : nativeint -> int32 -> unit = "caml_tolk_hcq_write32"
  [@@noalloc]

  external read64 : nativeint -> int64 = "caml_tolk_hcq_read64"

  external write64 : nativeint -> int64 -> unit = "caml_tolk_hcq_write64"
  [@@noalloc]

  external fence : unit -> unit = "caml_tolk_hcq_fence" [@@noalloc]

  external read64_int : nativeint -> int = "caml_tolk_hcq_read64_int"
  [@@noalloc]

  external monotonic_ms : unit -> int = "caml_tolk_hcq_monotonic_ms"
  [@@noalloc]

  external memcpy_to_ptr : nativeint -> bytes -> int -> int -> unit
    = "caml_tolk_hcq_memcpy_to_ptr"
  [@@noalloc]

  external memcpy_from_ptr : bytes -> int -> nativeint -> int -> unit
    = "caml_tolk_hcq_memcpy_from_ptr"
  [@@noalloc]
end

module File_io = struct
  let {
        Ffi.o_rdonly;
        o_rdwr;
        prot_none;
        prot_read;
        prot_write;
        map_shared;
        map_private;
        map_anonymous;
        map_fixed;
        map_noreserve;
      } =
    Ffi.constants ()

  let openfile path ~flags = Ffi.openfile path flags
  let close fd = Ffi.close fd

  let mmap ~addr ~size ~prot ~flags ~fd ~offset =
    Ffi.mmap addr size prot flags fd offset

  let munmap addr ~size = Ffi.munmap addr size
end

module Mmio = struct
  type t = { addr : nativeint; size : int }

  let make ~addr ~size =
    if size < 0 then invalid_arg "Mmio.make: negative size";
    { addr; size }

  let addr t = t.addr
  let size t = t.size

  let check t off len =
    if off < 0 || len < 0 || off + len > t.size then
      invalid_arg
        (Printf.sprintf "Mmio: range %d+%d exceeds size %d" off len t.size)

  let ptr t off = Nativeint.add t.addr (Nativeint.of_int off)

  let view t ~off ?size () =
    let size = match size with Some s -> s | None -> t.size - off in
    check t off size;
    { addr = ptr t off; size }

  let read32 t off =
    check t off 4;
    Ffi.read32 (ptr t off)

  let write32 t off v =
    check t off 4;
    Ffi.write32 (ptr t off) v

  let read64 t off =
    check t off 8;
    Ffi.read64 (ptr t off)

  let write64 t off v =
    check t off 8;
    Ffi.write64 (ptr t off) v

  let blit_bytes t ~off src =
    let len = Bytes.length src in
    check t off len;
    Ffi.memcpy_to_ptr (ptr t off) src 0 len

  let read_bytes t ~off ~len =
    check t off len;
    let dst = Bytes.create len in
    Ffi.memcpy_from_ptr dst 0 (ptr t off) len;
    dst

  let fence = Ffi.fence
end

module Buffer = struct
  type 'meta t = {
    va : nativeint;
    size : int;
    view : Mmio.t option;
    meta : 'meta;
    base : 'meta t option;
  }

  let make ~va ~size ?view ~meta () =
    if size < 0 then invalid_arg "Buffer.make: negative size";
    { va; size; view; meta; base = None }

  let va t = t.va
  let size t = t.size
  let view t = t.view
  let meta t = t.meta
  let base t = match t.base with Some b -> b | None -> t

  let cpu_view t =
    match t.view with
    | Some v -> v
    | None -> invalid_arg "Buffer.cpu_view: buffer has no view"

  let offset t ~off ?size () =
    let size = match size with Some s -> s | None -> t.size - off in
    if off < 0 || size < 0 || off + size > t.size then
      invalid_arg
        (Printf.sprintf "Buffer.offset: range %d+%d exceeds size %d" off size
           t.size);
    {
      va = Nativeint.add t.va (Nativeint.of_int off);
      size;
      view = Option.map (fun v -> Mmio.view v ~off ~size ()) t.view;
      meta = t.meta;
      base = Some (base t);
    }
end

module Q = struct
  type t = { mutable buf : int array; mutable len : int }

  let create () = { buf = Array.make 64 0; len = 0 }
  let length t = t.len

  let grow t =
    let buf = Array.make (2 * Array.length t.buf) 0 in
    Array.blit t.buf 0 buf 0 t.len;
    t.buf <- buf

  let push t v =
    if v lsr 32 <> 0 then invalid_arg "Q.push: not a 32-bit value";
    if t.len = Array.length t.buf then grow t;
    Array.unsafe_set t.buf t.len v;
    t.len <- t.len + 1

  let push64 t v =
    push t (Int64.to_int (Int64.logand v 0xFFFFFFFFL));
    push t (Int64.to_int (Int64.shift_right_logical v 32))

  let get t i =
    if i < 0 || i >= t.len then invalid_arg "Q.get: index out of bounds";
    Array.unsafe_get t.buf i

  let set t i v =
    if i < 0 || i >= t.len then invalid_arg "Q.set: index out of bounds";
    if v lsr 32 <> 0 then invalid_arg "Q.set: not a 32-bit value";
    Array.unsafe_set t.buf i v

  let dwords t = Array.sub t.buf 0 t.len
  let clear t = t.len <- 0
end

module Signal = struct
  type ('meta, 'dev) t = {
    buf : 'meta Buffer.t;
    view : Mmio.t;
    value_ptr : nativeint;
    owner : 'dev option;
    is_timeline : bool;
    timestamp_divider : float;
    sleep : int -> unit;
  }

  exception Timeout of { timeout_ms : int; goal : int; value : int }

  let default_timeout_ms = Tolk.Helpers.getenv "HCQDEV_WAIT_TIMEOUT_MS" 30000
  let value t = Ffi.read64_int t.value_ptr
  let set_value t v = Mmio.write64 t.view 0 (Int64.of_int v)

  let make ?(value = 0) ?(is_timeline = false) ?(timestamp_divider = 1000.)
      ?(sleep = fun (_ : int) -> ()) ?owner buf =
    if Buffer.size buf < 16 then
      invalid_arg "Signal.make: slot smaller than 16 bytes";
    let view = Buffer.cpu_view buf in
    let t =
      {
        buf;
        view;
        value_ptr = Mmio.addr view;
        owner;
        is_timeline;
        timestamp_divider;
        sleep;
      }
    in
    set_value t value;
    t

  let buf t = t.buf
  let owner t = t.owner
  let is_timeline t = t.is_timeline
  let value_addr t = Buffer.va t.buf
  let timestamp_addr t = Nativeint.add (Buffer.va t.buf) 8n
  let timestamp t = Int64.to_float (Mmio.read64 t.view 8) /. t.timestamp_divider

  let wait t ?timeout_ms goal =
    let timeout_ms =
      match timeout_ms with Some ms -> ms | None -> default_timeout_ms
    in
    let start = ref (Ffi.monotonic_ms ()) in
    let passed = ref false in
    let timed_out = ref false in
    while (not !passed) && not !timed_out do
      let prev = value t in
      if prev >= goal then passed := true
      else
        let cur = Ffi.monotonic_ms () in
        if cur - !start >= timeout_ms then timed_out := true
        else begin
          t.sleep (cur - !start);
          (* Progress resets the deadline: only a stalled signal times out. *)
          if value t <> prev then start := Ffi.monotonic_ms ()
        end
    done;
    if not !passed then begin
      let last = value t in
      if last < goal then raise (Timeout { timeout_ms; goal; value = last })
    end

  module Pool = struct
    let slot_size = 16

    type 'meta t = {
      alloc_page : unit -> 'meta Buffer.t;
      mutable pages : 'meta Buffer.t list;
      mutable free : 'meta Buffer.t list;
    }

    let create ~alloc_page = { alloc_page; pages = []; free = [] }

    let get t =
      (match t.free with
      | [] ->
          let page = t.alloc_page () in
          if Buffer.size page < slot_size then
            invalid_arg "Signal.Pool.get: page smaller than one slot";
          t.pages <- page :: t.pages;
          for i = 0 to (Buffer.size page / slot_size) - 1 do
            t.free <-
              Buffer.offset page ~off:(i * slot_size) ~size:slot_size ()
              :: t.free
          done
      | _ :: _ -> ());
      match t.free with
      | slot :: rest ->
          t.free <- rest;
          slot
      | [] -> assert false

    let put t slot = t.free <- slot :: t.free
    let pages t = List.rev t.pages
  end
end

(* Timeline lifecycle and host-transfer staging shared by hardware-queue
   device runtimes (hcq.py:384-517 HCQCompiled, :576-645 HCQAllocator). *)
module Timeline = struct
  type ('meta, 'dev) t = {
    mutable timeline : ('meta, 'dev) Signal.t;
    mutable shadow_timeline : ('meta, 'dev) Signal.t;
    mutable timeline_value : int;
    mutable error_state : exn option;
    (* Rotating pinned staging buffers for host transfers; each slot records
       the timeline value of its last use so reuse waits only for that
       submission. *)
    bounce : 'meta Buffer.t array;
    bounce_timeline : int array;
    mutable bounce_next : int;
    on_hang : unit -> unit;
  }

  let next_timeline t =
    t.timeline_value <- t.timeline_value + 1;
    t.timeline_value - 1

  (* The timeline counter must stay a signal dword: past 2^31 the counter
     restarts at 1 on the shadow signal, whose stale value cannot be mistaken
     for a future one, and the staging slots forget their old values. *)
  let wrap_timeline_signal t =
    let tl = t.timeline in
    t.timeline <- t.shadow_timeline;
    t.shadow_timeline <- tl;
    t.timeline_value <- 1;
    Signal.set_value t.timeline 0;
    Array.fill t.bounce_timeline 0 (Array.length t.bounce_timeline) 0

  (* A stalled or faulted wait latches the device error so every later
     synchronize fails loudly with the fault report; a passed wait rolls the
     timeline over before its counter outgrows the signal dword. The wait
     failure and the hang report are folded into one exception: each may be
     all the information there is. *)
  let guarded_wait t f =
    match f () with
    | r ->
        if t.timeline_value > 1 lsl 31 then wrap_timeline_signal t;
        r
    | exception ((Signal.Timeout _ | Failure _) as e) ->
        let base =
          match e with
          | Signal.Timeout { timeout_ms; goal; value } ->
              Printf.sprintf
                "Wait timeout: %d ms! (the signal is not set to %d, but %d)"
                timeout_ms goal value
          | Failure msg -> msg
          | e -> Printexc.to_string e
        in
        let report =
          match t.on_hang () with
          | () -> None
          | exception Failure report -> Some report
          | exception e -> Some (Printexc.to_string e)
        in
        let combined =
          Failure
            (match report with
            | None | Some "" -> base
            | Some r when String.equal r base -> base
            | Some r -> base ^ "\n" ^ r)
        in
        t.error_state <- Some combined;
        raise combined

  let synchronize t =
    (match t.error_state with Some e -> raise e | None -> ());
    guarded_wait t (fun () -> Signal.wait t.timeline (t.timeline_value - 1))

  let copyin t ~submit_chunk buf bytes =
    let total = Bytes.length bytes in
    let step = Buffer.size t.bounce.(0) in
    let off = ref 0 in
    while !off < total do
      t.bounce_next <- (t.bounce_next + 1) mod Array.length t.bounce;
      let slot = t.bounce_next in
      Signal.wait t.timeline t.bounce_timeline.(slot);
      let len = min step (total - !off) in
      Mmio.blit_bytes
        (Buffer.cpu_view t.bounce.(slot))
        ~off:0
        (Bytes.sub bytes !off len);
      submit_chunk ~dest:(Buffer.offset buf ~off:!off ()) ~src:t.bounce.(slot)
        len;
      t.bounce_timeline.(slot) <- t.timeline_value - 1;
      off := !off + len
    done

  let copyout t ~submit_chunk bytes buf =
    let total = Bytes.length bytes in
    let staging = t.bounce.(0) in
    let step = Buffer.size staging in
    let off = ref 0 in
    while !off < total do
      let len = min step (total - !off) in
      submit_chunk ~dest:staging ~src:(Buffer.offset buf ~off:!off ()) len;
      Signal.wait t.timeline (t.timeline_value - 1);
      Bytes.blit
        (Mmio.read_bytes (Buffer.cpu_view staging) ~off:0 ~len)
        0 bytes !off len;
      off := !off + len
    done
end

module Kernargs = struct
  type 'meta t = { buf : 'meta Buffer.t; bump : Tolk.Bump.t }

  let create buf = { buf; bump = Tolk.Bump.create ~size:(Buffer.size buf) () }

  let alloc t size =
    Buffer.offset t.buf ~off:(Tolk.Bump.alloc t.bump size ~align:8 ()) ~size ()

  let write_args ?(prefix = [||]) slot ~bufs ~vals =
    let view = Buffer.cpu_view slot in
    Array.iteri
      (fun i w ->
        if w < 0 || w > 0xFFFF_FFFF then
          invalid_arg "Kernargs.write_args: not a 32-bit value";
        Mmio.write32 view (4 * i) (Int32.of_int w))
      prefix;
    let base = 4 * Array.length prefix in
    Array.iteri
      (fun i va -> Mmio.write64 view (base + (8 * i)) (Int64.of_nativeint va))
      bufs;
    let base = base + (8 * Array.length bufs) in
    Array.iteri
      (fun i v ->
        if v < -0x8000_0000 || v > 0xFFFF_FFFF then
          invalid_arg "Kernargs.write_args: not a 32-bit value";
        Mmio.write32 view (base + (4 * i)) (Int32.of_int v))
      vals
end
