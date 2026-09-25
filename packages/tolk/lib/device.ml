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
    let free_cache () =
      List.iter
        (fun (size, spec, buf) -> inner.free buf size spec)
        (Atomic.exchange cache [])
    in
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
  host : string;
  copy : Tolk_uop.Uop.t -> bool;
  encode : Uop.t -> Uop.t option;
  lower : Uop.t -> Uop.t option;
  compile : Uop.t -> Uop.t;
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

type t = {
  name : string;
  allocator : Allocator.packed;
  renderer_set : Renderer_set.t;
  runtime : runtime;
  synchronize : unit -> unit;
  invalidate_caches_fn : (unit -> unit) option;
  queue : queue option;
  bufferize : Uop.t -> Buffer.t option;
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

let make ~name ~allocator ~renderer_set ~runtime ~synchronize
    ?invalidate_caches ?queue ?(bufferize = fun _ -> None) () =
  let device = { name; allocator; renderer_set; runtime; synchronize;
    invalidate_caches_fn = invalidate_caches; queue; bufferize } in
  Hashtbl.replace opened (canonicalize name) device;
  device

let name d = d.name
let renderer d = Renderer_set.select d.renderer_set
let runtime d (obj : Tolk_uop.Tiny_elf.t) =
  let nbufs = List.fold_left (fun n (a : Tolk_uop.Tiny_elf.argument) ->
      if a.addrspace = Tolk_uop.Dtype.Alu then n else n + 1) 0 obj.signature in
  let nvals = List.length obj.signature - nbufs in
  let seen = Array.make (nbufs + nvals) false in
  List.iter (fun (a : Tolk_uop.Tiny_elf.argument) ->
      if a.slot < 0 || a.slot >= Array.length seen || seen.(a.slot)
         || ((a.addrspace = Tolk_uop.Dtype.Alu) <> (a.slot >= nbufs)) then
        invalid_arg (Printf.sprintf "program %S: invalid argument slot %d" obj.name a.slot);
      seen.(a.slot) <- true) obj.signature;
  let prg = d.runtime obj in
  let name = obj.name in
  let call bufs ~global ~local ~vals ~wait ~timeout =
    if Array.length bufs <> nbufs || Array.length vals <> nvals then
      invalid_arg (Printf.sprintf
          "program %S: expected %d buffers and %d scalars, received %d and %d"
          name nbufs nvals (Array.length bufs) (Array.length vals));
    prg.call bufs ~global ~local ~vals ~wait ~timeout
  in
  { prg with call }
let synchronize d = d.synchronize ()
let queue d = d.queue
let bufferize d = d.bufferize

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

let invalidate_caches d = Option.iter (fun f -> f ()) d.invalidate_caches_fn

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
      d

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
