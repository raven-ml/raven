(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external state : unit -> nativeint = "caml_rig_memory_new"
external malloc : int -> int = "caml_rig_memory_alloc"
external free : int -> unit = "caml_rig_memory_free"
external room_entry : unit -> nativeint = "caml_rig_memory_room"
external submit_entry : unit -> nativeint = "caml_rig_memory_submit"
external load : int -> int = "caml_rig_load64"

module D = struct
  type t = { self : nativeint }

  (* A region is memory the device allocated, which [free] gives back, or a
     mapping of host memory, which it leaves alone. The word is the device's
     state, which its free ends. *)
  type region = { at : int; owned : bool }
  type image = unit
  type capability = unit

  exception Fault of string

  let key : t Type.Id.t = Type.Id.make ()
  let arch _ = "memory"
  let budget _ = max_int
  let queues _ = [ "COMPUTE:0"; "COPY:0" ]

  let alloc _ _ n =
    let at = malloc n in
    if at = 0 then None else Some { at; owned = true }

  let free _ r = if r.owned then free r.at
  let address r = Some r.at
  let handle r = Nativeint.of_int r.at
  let host r = Some r.at
  let peer _ _ = true
  let map_peer _ _ r = Some { r with owned = false }
  let map_host _ p _ = Some { at = p; owned = false }
  let image _ _ = Error "a memory device loads no code"
  let entry () _ = None
  let unload _ () = ()
  let word d = { at = Nativeint.to_int d.self; owned = true }
  let signaled d = load (Nativeint.to_int d.self)
  let sleep _ ~seen:_ ~still_ms:_ = ()
  let completion _ = `Host
  let waits_on _ _ = false
  let max_waits _ = 0
  let maps_host _ = true

  (* Its hand-over runs the submission's work, which may take long. *)
  let blocks _ = `May_block
  let room_entry = room_entry ()
  let submit_entry = submit_entry ()
  let self d = d.self
  let capability _ = ()
  let capability_key : capability Type.Id.t = Type.Id.make ()
  let stop _ = ()
end

let open_ name =
  Dev.open_driver ~memory_device:true
    (module D)
    ~name
    (fun () -> Ok { D.self = state () })
