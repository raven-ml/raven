(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A driver whose memory is the host's and whose work runs in the submitting
   thread: the copies and fills of a submission, then the word. *)

external state : unit -> nativeint = "caml_device_core_memory_new"
external malloc : int -> int = "caml_device_core_memory_alloc"
external free : int -> unit = "caml_device_core_memory_free"
external room_entry : unit -> nativeint = "caml_device_core_memory_room"
external submit_entry : unit -> nativeint = "caml_device_core_memory_submit"

external call_fill : nativeint -> nativeint -> int -> int
  = "caml_device_core_memory_fill"

external load : int -> int = "caml_device_core_load64"
external store : int -> int -> unit = "caml_device_core_store64"
external memmove : int -> int -> int -> unit = "caml_device_core_memmove"

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

module D = struct
  type t = { self : nativeint; mutable last : int }
  type region = { at : int; owned : bool }
  type image = unit

  type part =
    | Fill of nativeint * nativeint
    | Copy of region * int * region * int * int

  type capability = unit

  exception Fault of string

  let key : t Type.Id.t = Type.Id.make ()
  let arch _ = "memory"
  let budget _ = max_int
  let queues _ = [ "COMPUTE:0"; "COPY:0" ]

  let alloc _ _ n =
    let at = malloc n in
    if at = 0 then None else Some { at; owned = true }

  let free _ r =
    if not r.owned then
      invalid_arg "Device_core.memory_device: free of a mapping";
    free r.at

  let address r = Some r.at
  let handle r = Nativeint.of_int r.at
  let host r = Some (Nativeint.of_int r.at)
  let peer _ _ = true
  let map_peer _ _ r = Some { r with owned = false }
  let map_host _ p _ = Some { at = Nativeint.to_int p; owned = false }
  let unmap _ _ = ()
  let image _ _ = Error (`Refused "a memory device loads no code")
  let entry () _ = None
  let unload _ () = ()
  let word d = { at = Nativeint.to_int d.self; owned = false }
  let signaled d = load (Nativeint.to_int d.self)
  let sleep _ ~seen:_ ~still_ms:_ = ()
  let completion _ = `Host
  let waits_on _ _ = false
  let blocks _ = `Returns

  let part _ ~queue ?after:_ = function
    | `Words _ ->
        invalid_argf "Device_core.memory_device: %s runs no words" queue
    | `Fill (f, arg, 0, 0) -> Fill (f, arg)
    | `Fill _ ->
        invalid_arg "Device_core.memory_device: a fill declares no room"
    | `Copy ((dst, o), (src, o'), n) -> Copy (dst, o, src, o', n)

  let room _ _ = `Fits

  let submit d ~v ~waits:_ ~handles:_ parts =
    if v <> d.last + 1 then
      invalid_argf "Device_core.memory_device: value %d after %d" v d.last;
    d.last <- v;
    let rec run i =
      if i = Array.length parts then begin
        store (Nativeint.to_int d.self) v;
        `Ok
      end
      else
        match parts.(i) with
        | Fill (f, arg) ->
            if call_fill f arg v <> 0 then `Failed "a fill failed"
            else run (i + 1)
        | Copy (dst, o, src, o', n) ->
            memmove (dst.at + o) (src.at + o') n;
            run (i + 1)
    in
    run 0

  let room_entry = room_entry ()
  let submit_entry = submit_entry ()
  let self d = d.self
  let capability _ = ()
  let capability_key : capability Type.Id.t = Type.Id.make ()
  let stop _ = `Stopped
end

let open_ name =
  Dev.open_driver ~memory_device:true
    (module D)
    ~name
    (fun () -> Ok { D.self = state (); last = 0 })
