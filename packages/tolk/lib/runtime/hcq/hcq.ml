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

  external wait_progress : nativeint -> nativeint -> int64 -> unit
    = "caml_tolk_hcq_wait_progress"

  external submission_symbol : string -> nativeint = "caml_tolk_hcq_submission_symbol"

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
    if off < 0 || len < 0 || off > t.size || len > t.size - off then
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

module Submission = struct
  type t = { buffer : Tolk_uop.Storage.t; view : Mmio.t }

  let create () =
    let open Tolk_uop in
    let allocator = Storage.Host_allocator.make ~synchronize:(fun () -> ()) in
    let buffer = Storage.create ~device:"CPU" ~size:2 ~dtype:Dtype.uint64
        (Storage.Allocator.Pack allocator) in
    Storage.ensure_allocated buffer;
    let view = Mmio.make ~addr:(Option.get (Storage.host_addr buffer)) ~size:16 in
    {buffer; view}

  let buffer t = t.buffer
  let check t =
    match Mmio.read64 t.view 8 with
    | 0L -> ()
    | 2L -> failwith "HCQ command stream exceeds ring capacity"
    | _ -> failwith "HCQ submission timed out"

  let clear_error t = Mmio.write64 t.view 8 0L

  let prepare ?(timeout_ms = 30000) t =
    check t;
    if timeout_ms < 0 then invalid_arg "Submission.prepare: negative timeout";
    Mmio.write64 t.view 0 Int64.(add (of_int (Ffi.monotonic_ms ())) (of_int timeout_ms))

  let wait_progress t progress ~target =
    Ffi.wait_progress (Mmio.addr t.view) (Mmio.addr progress) target;
    check t

  let lower name u =
    let open Tolk_uop in
    let module U = Uop in
    let context () = U.placeholder ~shape:[2] ~dtype:Dtype.uint64 ~slot:0
        ~device:(U.Single name) ~volatile:true ~allocation:("hcq_submission", "") () in
    let index ptr i = U.index ~ptr ~idxs:[U.const_int i] () in
    match U.as_load u, U.as_store u with
    | Some {src; _}, _ ->
        (match U.as_index src with
         | Some {ptr; idxs = [i]} when U.const_int_value i = Some 0
             && U.node_tag (U.buf_uop ptr) = Some "timeline" && U.op ptr = Ops.After ->
             let deps = List.tl (U.children ptr) in
             (match deps with
              | target :: _ -> Some (Tolk.Hcq2.ccall ~host:name ~after:deps
                  ~name:"tolk_hcq_poll" ~dtype:Dtype.uint64
                  [index (context ()) 0; index (U.without_after ptr) 0; target])
              | [] -> None)
         | _ -> None)
    | _, Some {dst; value; gate = None} ->
        (match U.as_index dst with
         | Some {ptr; idxs} when List.for_all (fun i -> U.equal (U.get_idx i) i) idxs ->
             let waits = U.toposort ~enter_calls:true dst |> List.exists (fun n ->
                 U.op n = Ops.Custom_function && List.mem (U.Arg.as_string (U.arg n))
                   [Some "tolk_hcq_poll"; Some "tolk_hcq_wait_progress"]) in
             if not waits then None else
               let state = U.after ~src:(context ()) ~deps:[dst] in
               let error = U.load ~src:(index state 1) () in
               let cond = U.alu_binary ~op:Ops.Cmpeq ~lhs:error
                   ~rhs:(U.const (Const.int Dtype.uint64 0)) in
               let dst = U.index ~ptr ~idxs:(List.map (fun src -> U.valid ~src ~cond) idxs) () in
               Some (U.store ~dst ~value ())
         | _ -> None)
    | _ -> None

  let symbol = Ffi.submission_symbol
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
    if off < 0 || size < 0 || off > t.size || size > t.size - off then
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
    if is_timeline then Mmio.write64 view 8 0L;
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
    timeline : ('meta, 'dev) Signal.t;
    mutable error_state : exn option;
    (* Rotating pinned staging buffers for host transfers; each slot records
       the timeline value of its last use so reuse waits only for that
       submission. *)
    bounce : 'meta Buffer.t array;
    bounce_timeline : int array;
    mutable bounce_next : int;
    on_hang : unit -> unit;
  }

  let submitted t = Int64.to_int (Mmio.read64 (Buffer.cpu_view (Signal.buf t.timeline)) 8)

  (* Failures latch until device recovery explicitly clears them. *)
  let guarded_wait t f =
    (match t.error_state with Some e -> raise e | None -> ());
    match f () with
    | r -> r
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
        (* Recovery runs inside the fault reporter. Latch before entering it
           so a successful reset can explicitly clear the failed epoch. *)
        t.error_state <- Some (Failure base);
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
        if Option.is_some t.error_state then t.error_state <- Some combined;
        raise combined

  let synchronize ?timeout_ms t =
    (match t.error_state with Some e -> raise e | None -> ());
    guarded_wait t (fun () -> Signal.wait ?timeout_ms t.timeline (submitted t))

  let prepare t =
    (match t.error_state with Some e -> raise e | None -> ());
    let value = submitted t in
    if value land 0xffffffff >= 1 lsl 31 then begin
      (* GPU waits compare the low dword. Drain before restarting that dword,
         but retain a monotonically increasing epoch for host replay fences.
         Neither the signal address nor any retained fence needs rebinding. *)
      synchronize t;
      if value > max_int - (1 lsl 32) then failwith "HCQ timeline exhausted";
      let epoch = ((value lsr 32) + 1) lsl 32 in
      Signal.set_value t.timeline epoch;
      Mmio.write64 (Buffer.cpu_view (Signal.buf t.timeline)) 8 (Int64.of_int epoch)
    end

  let submit t f =
    Tolk_uop.Storage.with_operation (fun () ->
        prepare t;
        let value = submitted t + 1 in
        let result = guarded_wait t (fun () ->
            match f value with
            | result -> result
            | exception ((Signal.Timeout _ | Failure _) as error) -> raise error
            | exception error ->
                (* Submission may have touched hardware before raising. Keep
                   storage until the failed device has been retired. *)
                let backtrace = Printexc.get_raw_backtrace () in
                t.error_state <- Some error;
                Printexc.raise_with_backtrace error backtrace) in
        Mmio.write64 (Buffer.cpu_view (Signal.buf t.timeline)) 8 (Int64.of_int value);
        result)

  let copyin t ~submit_chunk buf bytes =
    let total = Bytes.length bytes in
    let step = Buffer.size t.bounce.(0) in
    let off = ref 0 in
    while !off < total do
      t.bounce_next <- (t.bounce_next + 1) mod Array.length t.bounce;
      let slot = t.bounce_next in
      guarded_wait t (fun () -> Signal.wait t.timeline t.bounce_timeline.(slot));
      let len = min step (total - !off) in
      Mmio.blit_bytes
        (Buffer.cpu_view t.bounce.(slot))
        ~off:0
        (Bytes.sub bytes !off len);
      submit_chunk ~dest:(Buffer.offset buf ~off:!off ()) ~src:t.bounce.(slot)
        len;
      t.bounce_timeline.(slot) <- submitted t;
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
      guarded_wait t (fun () -> Signal.wait t.timeline (submitted t));
      Bytes.blit
        (Mmio.read_bytes (Buffer.cpu_view staging) ~off:0 ~len)
        0 bytes !off len;
      off := !off + len
    done
end

let profile_offset name =
  let open Tolk in
  let open Tolk_uop in
  let module U = Uop in
  let calibration = lazy (
    let device = Device.get name in
    let stamp = Device.create_buffer ~size:2 ~dtype:Dtype.uint64
        ~spec:{Device.Buffer_spec.default with host = true; uncached = true; nolru = true} device in
    let timeline = Hcq2.timeline name in
    let at ptr i = U.index ~ptr ~idxs:[U.const_int i] () in
    let value = U.load ~src:(at timeline 1) () in
    let next = U.alu_binary ~op:Ops.Add ~lhs:value ~rhs:(U.const (Const.int Dtype.uint64 1)) in
    let instructions = U.linear [
        U.ins ~mnemonic:"wait" ~operands:[timeline; value] ();
        U.ins ~mnemonic:"timestamp" ~operands:[U.from_buffer stamp] ();
        U.ins ~mnemonic:"store" ~operands:[timeline; next] ()]
      |> fun linear -> U.replace linear ~arg:(U.Arg.Device (U.Single name)) () in
    let backend = String.lowercase_ascii (List.hd (String.split_on_char ':' name)) in
    let submit = U.custom_function ~name:("submit_" ^ backend ^ "_compute_0")
        ~srcs:[instructions; U.group []] in
    let bump = U.store ~dst:(at (U.after ~src:timeline ~deps:[submit]) 1) ~value:next () in
    let kernel_info = U.{name = "clock_calibration"; applied_opts = []; opts_to_apply = None;
      estimates = None; beam = 0} in
    let call = Hcq2.lower_call ~devices:[name] (U.sink ~kernel_info [bump]) in
    device, stamp, Realize.link_linear (U.linear [call])) in
  fun () -> Helpers.Context_var.with_context [Helpers.Context_var.B (Helpers.debug, 0)] (fun () ->
    let device, stamp, linked = Lazy.force calibration in
    let queue = Option.get (Device.queue device) in
    let to_program device = Codegen.to_program ~optimize:false device (Device.renderer device) in
    Profile.calibrate (fun () ->
        Realize.run_linear ~device ~to_program ~jit:true ~wait:false ~update_stats:false linked;
        fun () ->
          Device.synchronize device;
          let view = Option.get (Device.Buffer.as_buffer stamp) in
          let ticks = ref 0L in
          for i = 0 to 7 do
            ticks := Int64.logor !ticks
              (Int64.shift_left (Int64.of_int (Bigarray.Array1.unsafe_get view (8 + i))) (8 * i))
          done;
          Int64.to_float !ticks /. queue.timestamp_divider))
