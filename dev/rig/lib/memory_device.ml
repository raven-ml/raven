(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external state : unit -> nativeint = "caml_rig_memory_new"
external word_of : nativeint -> int = "caml_rig_memory_word"
external malloc : int -> int = "caml_rig_memory_alloc"
external free : int -> unit = "caml_rig_memory_free"
external load : int -> int = "caml_rig_load64"

module D = struct
  type t = { self : nativeint; word : int }

  (* A region is memory at [at] that [free] gives back from [base]: memory the
     device allocated, or its word, whose free ends the device's state; [base]
     is 0 for a mapping of host memory, which [free] leaves alone. *)
  type region = { at : int; base : int }
  type image = unit

  exception Fault of string

  let key : t Type.Id.t = Type.Id.make ()
  let capability_key : unit Type.Id.t = Type.Id.make ()
  let all = Rig_edge.[ Fill; Copy ]

  (* Its hand-over runs the submission's work, which may take long. *)
  let facts d =
    {
      Rig_edge.arch = "memory";
      budget = max_int;
      queues =
        [ { name = "COMPUTE:0"; runs = all }; { name = "COPY:0"; runs = all } ];
      completion = Host;
      waits = { stores = false; hosts = false; objects = false; most = 0 };
      may_block = true;
      hang_ms = None;
      maps_host = true;
      capability = Capability (capability_key, ());
      word = { at = d.word; base = Nativeint.to_int d.self };
      edge = d.self;
    }

  let alloc _ _ n =
    let at = malloc n in
    if at = 0 then None else Some { at; base = at }

  let free _ r = if r.base <> 0 then free r.base

  let locate r =
    { Rig_edge.address = Some r.at; host = Some r.at; handle = Nativeint.of_int r.at }

  let peer _ _ = true
  let map_peer _ _ r = Some { r with base = 0 }
  let map_host _ p _ = Some { at = p; base = 0 }
  let image _ _ = Error "a memory device loads no code"
  let entry () _ = None
  let unload _ () = ()
  let signaled d = load d.word
  let sleep _ ~seen:_ ~still_ms:_ = ()
  let stop _ ~fault:_ = ()
end

let open_ name =
  Dev.open_driver ~memory_device:true
    (module D)
    ~name
    (fun () ->
      let self = state () in
      Ok { D.self; word = word_of self })
