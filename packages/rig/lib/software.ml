(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external state : nativeint -> nativeint = "caml_rig_software_new"
external drop : nativeint -> unit = "caml_rig_software_drop"
external malloc : int -> int = "caml_rig_software_alloc"
external free : int -> unit = "caml_rig_software_free"
external load : int -> int = "caml_rig_load64"

module D = struct
  type t = { self : nativeint; word : int }

  (* A region is memory at [at] that [free] gives back from [base]: memory the
     device allocated; [base] is 0 for its word and a mapping of host memory,
     which [free] leaves alone. *)
  type region = { at : int; base : int }
  type image = Def.linked

  exception Fault of string

  let key : t Type.Id.t = Type.Id.make ()
  let capability_key : unit Type.Id.t = Type.Id.make ()

  (* The alignment [caml_rig_software_alloc] gives, a page from 64 KiB on
     ([rig_host_code.c]). *)
  let alignment = 64

  (* It runs the host's code, so its arch is the host's. Its hand-over runs
     the submission's work, which may take long. *)
  let facts d =
    {
      Rig_edge.arch = Cpu.host_arch;
      budget = max_int;
      queues =
        [
          { name = "COMPUTE:0"; runs = [ Launch; Fill; Copy ] };
          { name = "COPY:0"; runs = [ Fill; Copy ] };
        ];
      copy_engine = Some 1;
      completion = Host;
      waits = { stores = false; hosts = false };
      may_block = true;
      hang_ms = None;
      maps_host = true;
      host_addresses = true;
      mapped = false;
      alignment;
      (* Its memory is the host's, behind the host's caches. *)
      cache_bytes = Dev.host.Def.cache_bytes;
      counters = [];
      capability = Capability (capability_key, ());
      word = { at = d.word; base = 0 };
      bus = None;
      edge = d.self;
    }

  let alloc _ _ n =
    let at = malloc n in
    if at = 0 then None else Some { at; base = at }

  let free _ r = if r.base <> 0 then free r.base

  let locate r =
    {
      Rig_edge.address = Some r.at;
      host = Some r.at;
      handle = Nativeint.of_int r.at;
    }

  let peer _ _ = true
  let map_peer _ _ r = Rig_edge.Maps { r with base = 0 }
  let map_host _ p _ = Rig_edge.Maps { at = p; base = 0 }

  (* rig asks no software device: its memory is the host's. *)
  let dma _ _ ~fn:_ =
    Error
      "a software device's memory is the process's own, which no driver gives"

  let flush _ = ()

  (* Its code is the host's: linked as the host links it, and unmapped once
     rig's records of the images that share the link are collected. *)
  let image _ binary =
    Result.map (fun c -> Rig_edge.Loaded c) (Host_code.link binary)

  (* The host knows neither a function's parameters nor a limit on its
     threads. *)
  let entry c f =
    match Host_code.entry c f with
    | Some code ->
        Rig_edge.Entry { code; launch = 0n; params = None; threads = None }
    | None -> Absent

  let prepare _ _ = ()
  let unload _ _ = ()
  let signaled d = load d.word
  let sleep _ ~seen:_ ~still_ms:_ = ()
  let stop d ~fault:_ = drop d.self

  (* It has no queues. *)
  let grow _ = Rig_edge.Grown
end

let open_ name =
  Dev.open_driver ~software:true
    (module D)
    ~machine:Def.Here ~name
    (fun ~word ->
      Ok { D.self = state word; word = Nativeint.to_int word })
