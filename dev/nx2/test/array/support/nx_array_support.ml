(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external row : int -> (string * int * int) option = "nx_array_support_row"
external decode : int -> int -> int = "nx_array_support_decode"
external of_int64 : int -> int64 -> int = "nx_array_support_of_i64"
external of_uint64 : int -> int64 -> int = "nx_array_support_of_u64"
external layout : Nx_array.Layout.t -> int array = "nx_array_support_layout"
external add : 'z -> 'x -> 'y -> int = "nx_array_support_add"

external copy_into : ('v, 's) Nx_array.t -> ('v, 's) Nx_array.t -> int
  = "nx_array_copy"

external of_array_into : ('v, 's) Nx_array.t -> 'v array -> int
  = "nx_array_of_array"

external int16_at :
  int ->
  int ->
  (int, Bigarray.int16_signed_elt, Bigarray.c_layout) Bigarray.Genarray.t
  = "nx_array_support_int16_at"

external collect : ('v, 's) Nx_array.t -> int = "nx_array_support_collect"

(* An io device over bigarrays *)

type bytes =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

external blit_in : bytes -> int -> int -> int -> unit
  = "nx_array_support_blit_in"
[@@noalloc]

external blit_out : bytes -> int -> int -> int -> unit
  = "nx_array_support_blit_out"
[@@noalloc]

module Io = struct
  type t = unit
  type region = bytes

  exception Fault of string

  let region_key : region Type.Id.t = Type.Id.make ()
  let budget () = max_int
  let allocations = Atomic.make 0

  let alloc () n =
    Atomic.incr allocations;
    Some (Bigarray.Array1.create Bigarray.char Bigarray.c_layout n)

  let free () (_ : region) = ()
  let read () r ~at ~dst ~len = blit_out r at dst len
  let write () r ~at ~src ~len = blit_in r at src len
  let pages () (_ : region) = None
  let prefetch () (_ : region) ~at:(_ : int) ~len:(_ : int) = ()
  let stop () = ()
end

let io =
  lazy
    (match Rig.open_io (module Io) ~name:"nx2-io" (fun () -> Ok ()) with
    | Ok d -> d
    | Error e -> failwith e)

let io_device () = Lazy.force io
let io_allocations () = Atomic.get Io.allocations

(* Late, a device over host memory whose work runs only when a wait sleeps on
   it: its submit queues the work, its sleep and its stop run it. *)

external late_new : unit -> nativeint = "nx_array_support_late_new"
external late_publish : nativeint -> unit = "nx_array_support_late_publish"
external late_signaled : nativeint -> int = "nx_array_support_late_signaled"
external late_room : unit -> nativeint = "nx_array_support_late_room"
external late_submit : unit -> nativeint = "nx_array_support_late_submit"
external malloc : int -> int = "nx_array_support_malloc"
external free : int -> unit = "nx_array_support_free"
external store_fill : unit -> nativeint = "nx_array_support_store"

module Driver = struct
  type t = { self : nativeint; fault : string option Atomic.t }

  (* [raw] is what [malloc] gave, [0] for the word. *)
  type region = { at : int; raw : int }
  type image = unit
  type capability = unit

  exception Fault of string

  let key : t Type.Id.t = Type.Id.make ()
  let arch _ = "late"
  let budget _ = max_int
  let queues _ = [ "COMPUTE:0" ]
  let completion _ = `Host
  let waits_on _ _ = false
  let max_waits _ = 0
  let blocks _ = `Returns
  let maps_host _ = false
  let capability _ = ()
  let capability_key : capability Type.Id.t = Type.Id.make ()

  let alloc _ _ n =
    let raw = malloc n in
    if raw = 0 then None else Some { at = (raw + 63) land lnot 63; raw }

  let free _ r = if r.raw <> 0 then free r.raw
  let address r = Some r.at
  let handle r = Nativeint.of_int r.at
  let host r = Some r.at
  let peer _ _ = false
  let map_peer _ _ _ = None
  let map_host _ _ _ = None
  let image _ _ = Error "late loads no code"
  let entry () _ = None
  let unload _ () = ()
  let word d = { at = Nativeint.to_int d.self; raw = 0 }
  let signaled d = late_signaled d.self

  let sleep d ~seen:_ ~still_ms:_ =
    match Atomic.get d.fault with
    | Some why -> raise (Fault why)
    | None -> late_publish d.self

  let room_entry = late_room ()
  let submit_entry = late_submit ()
  let self d = d.self
  let stop d = late_publish d.self
end

module Late = struct
  type t = Driver.t

  let open_ name =
    let t = { Driver.self = late_new (); fault = Atomic.make None } in
    match Rig.open_ (module Driver) ~name (fun () -> Ok t) with
    | Ok d -> (d, t)
    | Error e -> failwith e

  let fault (t : t) why = Atomic.set t.fault (Some why)
end

let store = store_fill ()

let write b s =
  let n = String.length s in
  let arg = Bytes.create (16 + n) in
  Bytes.set_int64_le arg 0 (Int64.of_int (Rig.Buffer.address b));
  Bytes.set_int64_le arg 8 (Int64.of_int n);
  Bytes.blit_string s 0 arg 16 n;
  let arg = Rig.Buffer.of_string (Bytes.unsafe_to_string arg) in
  let work =
    Rig.Submission.Fill { fill = store; arg; ring_units = 0; segment_bytes = 0 }
  in
  let s =
    Rig.Submission.make ~reads:0 ~writes:1 (Rig.Buffer.device b)
      [| { queue = "COMPUTE:0"; after = [||]; work } |]
  in
  ignore (Rig.submit s ~reads:[||] ~writes:[| b |] ~waits:[||])
