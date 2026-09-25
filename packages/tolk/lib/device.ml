(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Tolk_uop

(* Buffer + Allocators *)

module Buffer_spec = Storage.Buffer_spec
module Allocator = Storage.Allocator

module Lru_allocator = struct
  (* Buffers are freed by GC finalisers, which can run inside any allocation,
     including one that [alloc] makes while it searches the cache, so the cache
     only changes by compare-and-set: an update that raced with a free is
     retried rather than overwriting it. *)
  let wrap (inner : 'buf Allocator.t) : 'buf Allocator.t =
    let cache : (int * Buffer_spec.t * 'buf) list Atomic.t = Atomic.make [] in
    let rec take size spec =
      let entries = Atomic.get cache in
      let rec find acc = function
        | (s, sp, buf) :: rest when s = size && sp = spec ->
            Some (buf, List.rev_append acc rest)
        | entry :: rest -> find (entry :: acc) rest
        | [] -> None
      in
      match find [] entries with
      | None -> None
      | Some (buf, rest) ->
          if Atomic.compare_and_set cache entries rest then Some buf
          else take size spec
    in
    let rec cache_buf entry =
      let entries = Atomic.get cache in
      if not (Atomic.compare_and_set cache entries (entry :: entries)) then
        cache_buf entry
    in
    let free_cache () =
      let rec restore entries =
        let current = Atomic.get cache in
        if not (Atomic.compare_and_set cache current (current @ entries)) then
          restore entries
      in
      let rec free = function
        | [] -> ()
        | (size, spec, buf) :: rest ->
            (match Storage.release (fun () -> inner.free buf size spec) with
             | () -> free rest
             | exception exn ->
                 let backtrace = Printexc.get_raw_backtrace () in
                 (* Only untouched entries may be reused. Concurrent additions
                    stay in the cache, and the uncertain owner stays retained. *)
                 restore rest;
                 Printexc.raise_with_backtrace exn backtrace)
      in
      free (Atomic.exchange cache [])
    in
    {
      inner with
      alloc =
        (fun size spec ->
          match take size spec with
          | Some buf -> buf
          | None -> (
              try inner.alloc size spec
              with exn -> (
                free_cache ();
                try inner.alloc size spec with _ -> raise exn)));
      free =
        (fun buf size spec ->
          if Helpers.Context_var.get Helpers.lru <> 0
             && (not spec.Buffer_spec.nolru)
             && Option.is_none spec.external_ptr
          then cache_buf (size, spec, buf)
          else inner.free buf size spec);
    }
end

module Buffer = Storage

(* Compiled devices *)

type prog = {
  call :
    Buffer.t array -> global:int array -> local:int array option ->
    vals:int64 array -> wait:bool -> timeout:int option -> float option;
  free : unit -> unit;
  handle : nativeint;
}

type runtime = Tolk_uop.Tiny_elf.t -> prog

type queue = {
  timestamp_divider : float;
  profile_offset : unit -> float;
  completion : unit -> (int option -> unit);
  prepare : unit -> unit;
  host : string;
  max_kernel_bindings : int option;
  copy : Tolk_uop.Uop.t -> string option;
  encode : Uop.t -> Uop.t option;
  lower : Uop.t -> Uop.t option;
  compile : Uop.t -> Uop.t;
  config : unit -> string;
}

module Renderer_set = struct
  type t = {
    device : string;
    arch : string;
    entries : (string * (Target.t -> Renderer.t)) list;
    cache : (Target.t, Renderer.t) Hashtbl.t;
    mutex : Mutex.t;
  }

  let make ?(arch = "") ~device entries =
    { device; arch; entries; cache = Hashtbl.create 4; mutex = Mutex.create () }

  let target set = Helpers.target ~arch:set.arch set.device

  let select set = Mutex.protect set.mutex (fun () ->
    let target = target set in
    List.iter (fun (name, _) ->
        let key = set.device ^ "_" ^ name in
        if Helpers.getenv key 0 <> 0 then
          invalid_arg (Printf.sprintf "%s is deprecated, use DEV=%s instead"
            key (Target.to_string { target with renderer = name }))) set.entries;
    match Hashtbl.find_opt set.cache target with
    | Some renderer -> renderer
    | None ->
        let entries = List.filter (fun (name, _) ->
            target.renderer = "" || target.renderer = name) set.entries in
        if entries = [] then
          invalid_arg (Printf.sprintf "%s has no renderer %S" set.device target.renderer);
        let renderer = Helpers.select_first_inited
            ~message:(Printf.sprintf "No renderer for %s is available" set.device)
            (List.map (fun (name, create) () ->
                 let target = { target with renderer = name } in
                 Renderer.with_target target (create target)) entries) in
        Hashtbl.add set.cache target renderer;
        renderer)
end

type pending_timing = { buffer : Buffer.t; first : int; last : int; label : string; queue_name : string }

type t = {
  name : string;
  peer_group : string;
  allocator : Allocator.packed;
  renderer_set : Renderer_set.t;
  runtime : runtime option;
  synchronize : int option -> unit;
  invalidate_caches_fn : (unit -> unit) option;
  queue : queue option;
  bufferize : Uop.t -> Buffer.t option;
  program_buffers : Buffer.t Uop.Tbl.t;
  program_lock : Mutex.t;
  synchronize_lock : Mutex.t;
  pending_lock : Mutex.t;
  pending_accesses : (string, int option -> unit) Hashtbl.t;
  pending_timings : (int * int, pending_timing) Hashtbl.t;
  mutable profile_events : Profile.event list;
}

type device = t

let canonicalize device =
  let device =
    match String.index_opt device ':' with
    | Some i ->
        String.uppercase_ascii (String.sub device 0 i)
        ^ String.sub device i (String.length device - i)
    | None -> String.uppercase_ascii device
  in
  let len = String.length device in
  if len >= 2 && String.equal (String.sub device (len - 2) 2) ":0" then
    String.sub device 0 (len - 2)
  else device

let openers : (string, string -> t) Hashtbl.t = Hashtbl.create 8
let opened : (string, t) Hashtbl.t = Hashtbl.create 8

let make ~name ~allocator ~renderer_set ?runtime ~synchronize
    ?invalidate_caches ?peer_group ?queue ?(bufferize = fun _ -> None)
    ?(initialize = fun _ -> ()) () =
  let queue = Option.map (fun (q : queue) -> {q with
      prepare = (fun () -> Storage.with_operation q.prepare);
      profile_offset = (fun () -> Storage.with_operation q.profile_offset);
      completion = (fun () ->
        let wait = Storage.with_operation q.completion in
        fun timeout -> Storage.with_operation (fun () -> wait timeout));
    }) queue in
  let peer_group = Option.value peer_group ~default:(List.hd (String.split_on_char ':' (canonicalize name))) in
  let device = { name; peer_group; allocator; renderer_set; runtime; synchronize;
    invalidate_caches_fn = invalidate_caches; queue; bufferize;
    program_buffers = Uop.Tbl.create 16; program_lock = Mutex.create ();
    synchronize_lock = Mutex.create (); pending_lock = Mutex.create ();
    pending_accesses = Hashtbl.create 0; pending_timings = Hashtbl.create 0;
    profile_events = [] } in
  let key = canonicalize name in
  let previous = Hashtbl.find_opt opened key in
  Hashtbl.replace opened key device;
  match initialize device with
  | () -> device
  | exception exn ->
      let backtrace = Printexc.get_raw_backtrace () in
      (match Hashtbl.find_opt opened key with
       | Some current when current == device ->
           (match previous with
            | Some previous -> Hashtbl.replace opened key previous
            | None -> Hashtbl.remove opened key)
       | _ -> ());
      Printexc.raise_with_backtrace exn backtrace

let name d = d.name
let peer_group d = d.peer_group
let renderer d = Renderer_set.select d.renderer_set
let load_runtime ~ordered d (obj : Tolk_uop.Tiny_elf.t) =
  Storage.with_operation (fun () ->
    let nbufs = List.fold_left (fun n (a : Tolk_uop.Tiny_elf.argument) ->
        if a.addrspace = Tolk_uop.Dtype.Alu then n else n + 1) 0 obj.signature in
    let nvals = List.length obj.signature - nbufs in
    let seen = Array.make (nbufs + nvals) false in
    List.iter (fun (a : Tolk_uop.Tiny_elf.argument) ->
        if a.slot < 0 || a.slot >= Array.length seen || seen.(a.slot)
           || ((a.addrspace = Tolk_uop.Dtype.Alu) <> (a.slot >= nbufs)) then
          invalid_arg (Printf.sprintf "program %S: invalid argument slot %d" obj.name a.slot);
        seen.(a.slot) <- true) obj.signature;
    let runtime = match d.runtime with
      | Some runtime -> runtime
      | None -> invalid_arg (Printf.sprintf "%s requires compiled queue submission" d.name) in
    let prg = runtime obj in
    let name = obj.name in
    let call bufs ~global ~local ~vals ~wait ~timeout =
      Storage.with_operation (fun () ->
        if Array.length bufs <> nbufs || Array.length vals <> nvals then
          invalid_arg (Printf.sprintf
              "program %S: expected %d buffers and %d scalars, received %d and %d"
              name nbufs nvals (Array.length bufs) (Array.length vals));
        if not ordered then Array.iter (Buffer.synchronize ~device:d.name) bufs;
        prg.call bufs ~global ~local ~vals ~wait ~timeout)
    in
    { prg with call; free = (fun () -> Storage.with_operation prg.free) })

let runtime d = load_runtime ~ordered:false d
let queue_runtime d = load_runtime ~ordered:true d

let with_pending_lock d f =
  Storage.with_operation (fun () ->
    Mutex.lock d.pending_lock;
    Fun.protect ~finally:(fun () -> Mutex.unlock d.pending_lock) f)

let depend_on d source =
  if d != source then begin
    let queue = match source.queue with
      | Some queue -> queue
      | None -> invalid_arg "Device.depend_on: source has no queue" in
    with_pending_lock d (fun () ->
        Hashtbl.replace d.pending_accesses source.name (queue.completion ()))
  end

let record_timing d ~name ~queue ~buffer ~first ~last =
  if first < 0 || last < 0 || max first last >= Buffer.nbytes buffer / 8 then
    invalid_arg "Device.record_timing: timestamp outside storage";
  if Option.is_none d.queue then invalid_arg "Device.record_timing: device has no queue";
  with_pending_lock d (fun () ->
      Hashtbl.replace d.pending_timings (Buffer.id buffer, first)
        {buffer; first; last; label = name; queue_name = queue})

let with_synchronize_lock d f =
  Storage.with_operation (fun () -> Mutex.protect d.synchronize_lock f)

let wait_dependencies d ~ordered = with_synchronize_lock d (fun () ->
  let accesses = with_pending_lock d (fun () ->
      let accesses = Hashtbl.to_seq d.pending_accesses |> List.of_seq
        |> List.filter (fun (source, _) -> not (List.mem source ordered)) in
      List.iter (fun (source, _) -> Hashtbl.remove d.pending_accesses source) accesses;
      accesses) in
  try List.iter (fun (_, wait) -> wait None) accesses
  with exn ->
    let backtrace = Printexc.get_raw_backtrace () in
    with_pending_lock d (fun () ->
        List.iter (fun (source, wait) ->
            if not (Hashtbl.mem d.pending_accesses source) then
              Hashtbl.add d.pending_accesses source wait) accesses);
    Printexc.raise_with_backtrace exn backtrace)

let synchronize ?timeout d = with_synchronize_lock d (fun () ->
  let pending, accesses = with_pending_lock d (fun () ->
      let pending = Hashtbl.to_seq_values d.pending_timings |> List.of_seq in
      Hashtbl.clear d.pending_timings;
      let accesses = Hashtbl.to_seq d.pending_accesses |> List.of_seq in
      Hashtbl.clear d.pending_accesses;
      pending, accesses) in
  let events = try
    d.synchronize timeout;
    List.iter (fun (_, wait) -> wait timeout) accesses;
    match pending with
    | [] -> []
    | _ ->
        let snapshots = Hashtbl.create 4 in
        let divider = (Option.get d.queue).timestamp_divider in
        List.map (fun entry ->
            let id = Buffer.id entry.buffer in
            let bytes = match Hashtbl.find_opt snapshots id with
              | Some bytes -> bytes
              | None -> let bytes = Buffer.as_bytes entry.buffer in
                  Hashtbl.add snapshots id bytes; bytes in
            let start = Bytes.get_int64_le bytes (8 * entry.first)
            and finish = Bytes.get_int64_le bytes (8 * entry.last) in
            Profile.{device = d.name; queue = entry.queue_name; name = entry.label;
              start_us = Int64.to_float start /. divider;
              duration_us = Int64.to_float (Int64.sub finish start) /. divider}) pending
  with exn ->
    let backtrace = Printexc.get_raw_backtrace () in
    with_pending_lock d (fun () ->
      List.iter (fun (source, wait) ->
          if not (Hashtbl.mem d.pending_accesses source) then
            Hashtbl.add d.pending_accesses source wait) accesses;
      List.iter (fun entry ->
        let key = Buffer.id entry.buffer, entry.first in
        if not (Hashtbl.mem d.pending_timings key) then
          Hashtbl.add d.pending_timings key entry) pending);
    Printexc.raise_with_backtrace exn backtrace in
  if events <> [] then
    with_pending_lock d (fun () -> d.profile_events <- List.rev_append events d.profile_events))

let profile d =
  synchronize d;
  if with_pending_lock d (fun () -> d.profile_events = []) then [] else
    let offset = (Option.get d.queue).profile_offset () in
    if not (Float.is_finite offset) then invalid_arg "Device.profile: invalid clock offset";
    with_pending_lock d (fun () ->
        let events = List.rev_map (fun event ->
            {event with Profile.start_us = event.Profile.start_us +. offset}) d.profile_events in
        d.profile_events <- [];
        events)

let queue d = d.queue
let bufferize d u = Storage.with_operation (fun () ->
    if Uop.node_tag u <> Some "program" then d.bufferize u
    else Mutex.protect d.program_lock (fun () ->
        match Uop.Tbl.find_opt d.program_buffers u with
        | Some buffer -> Some buffer
        | None ->
            let buffer = d.bufferize u in
            Option.iter (Uop.Tbl.add d.program_buffers u) buffer;
            buffer))

let compile_program d ?name ?(applied_opts = []) ?(estimates = Program_spec.Estimates.zero) program =
  let module U = Tolk_uop.Uop in
  (* TinyELF and dispatch share a buffer-first signature. Hand-built linear
     programs need the same formal order as the codegen linearizer. *)
  let params, body = List.partition (fun u -> U.op u = Tolk_uop.Ops.Param) program in
  let buffers, scalars = List.partition
      (fun u -> U.addrspace u <> Some Tolk_uop.Dtype.Alu) params in
  let program = buffers @ scalars @ body in
  let ren = Renderer_set.select d.renderer_set in
  let comp = match Renderer.compiler ren with
    | Some c -> c
    | None -> invalid_arg "device has no compiler"
  in
  let name = Option.value name ~default:"kern" in
  let src = Renderer.render ren ~name program in
  let lib = Compiler.compile_cached comp src in
  Program_spec.of_program ~name ~src ~device:d.name ~target:(Renderer.target ren)
    ~lib ~applied_opts ~estimates program

let create_buffer ~size ~dtype ?spec d =
  Buffer.create ~device:d.name ~size ~dtype ?spec d.allocator

let invalidate_caches d = Storage.with_operation (fun () ->
    match d.invalidate_caches_fn with
    | Some invalidate -> invalidate (); true
    | None -> false)

(* Device registry

   Canonical-name lookup opening and caching device runtimes, with backend
   openers registered by prefix. The engine resolves the device names carried
   by a scheduled graph through [get], so multi-device schedules can span
   device instances the caller never opened itself. *)

let register prefix opener =
  Hashtbl.replace openers (String.uppercase_ascii prefix) opener

let device_prefix device =
  match String.index_opt device ':' with
  | Some i -> String.sub device 0 i
  | None -> device

let get device =
  Storage.with_operation (fun () ->
    let device = canonicalize device in
    match Hashtbl.find_opt opened device with
    | Some d -> d
    | None ->
        let d =
          match Hashtbl.find_opt openers (device_prefix device) with
          | Some create -> create device
          | None -> failwith (Printf.sprintf "unknown device %S" device)
        in
        Hashtbl.replace opened device d;
        d)

let () = Storage.install_allocator_resolver (fun name -> (get name).allocator)

module Multi_buffer = struct
  type t = { bufs : Buffer.t list }

  let create ~devices ~size ~dtype ?spec () =
    if devices = [] then invalid_arg "multi buffer requires at least one device";
    let bufs =
      List.map
        (fun device -> create_buffer ~size ~dtype ?spec (get device))
        devices
    in
    { bufs }

  let of_bufs bufs =
    match bufs with
    | [] -> invalid_arg "multi buffer requires at least one buffer"
    | first :: rest ->
        if
          not
            (List.for_all
               (fun b ->
                 Buffer.size b = Buffer.size first
                 && Dtype.equal (Buffer.dtype b) (Buffer.dtype first))
               rest)
        then invalid_arg "multi buffer requires matching sizes and dtypes";
        { bufs }

  let bufs t = t.bufs
  let size t = Buffer.size (List.hd t.bufs)
  let dtype t = Buffer.dtype (List.hd t.bufs)

  let add_ref t cnt =
    List.iter (fun buf -> ignore (Buffer.add_ref buf cnt)) t.bufs;
    t

  let is_allocated t = List.for_all Buffer.is_allocated t.bufs

  let view t ~size ~dtype ~offset =
    { bufs = List.map (fun b -> Buffer.view b ~size ~dtype ~offset) t.bufs }
end
