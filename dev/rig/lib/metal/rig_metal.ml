(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* The C side. A device is the address of its C state, as an int; a Metal object
   is a retained pointer, a pipeline's as an int; a buffer is the triple of its
   object, GPU address and host address. *)

type buffer = nativeint * int * int

external count : unit -> int = "caml_rig_metal_count"
external open_device : unit -> int = "caml_rig_metal_open"
external facts : int -> int * int * buffer = "caml_rig_metal_facts"
external alloc_buffer : int -> int -> buffer option = "caml_rig_metal_alloc"

external map_buffer : int -> int -> int -> buffer option
  = "caml_rig_metal_map_host"

external free_buffer : int -> nativeint -> unit = "caml_rig_metal_free"
external release : int -> unit = "caml_rig_metal_release"

external load : int -> string -> string * string array * int array
  = "caml_rig_metal_image"

external make_icb :
  int -> nativeint -> int array -> int array -> string * nativeint array
  = "caml_rig_metal_icb"

external release_icb : nativeint -> unit = "caml_rig_metal_icb_release"

external signaled_word : (int[@untagged]) -> (int[@untagged])
  = "caml_rig_metal_signaled_byte" "caml_rig_metal_signaled"
[@@noalloc]

external sleep_word : int -> int -> int -> int = "caml_rig_metal_sleep"
external failure : int -> string = "caml_rig_metal_failure"
external stop_ring : int -> unit = "caml_rig_metal_stop"

external entries : unit -> nativeint * nativeint * nativeint
  = "caml_rig_metal_entries"

let room_entry, submit_entry, split = entries ()

exception Fault of string

(* Memory *)

type region = {
  owner : int;
  handle : nativeint;
  address : int;
  host : int;
  live : bool Atomic.t;
}

let region owner (handle, address, host) =
  { owner; handle; address; host; live = Atomic.make true }

let address r = Some r.address
let handle r = r.handle
let host r = Some r.host

(* Opening *)

type capability = Rig_metal_abi.t

type t = {
  self : int;
  arch : string;
  budget : int;
  word : region;
  cap : capability;
}

let device_name i =
  if i < 0 then invalid_argf "Rig_metal.device_name: GPU %d is negative" i;
  if i = 0 then "METAL" else strf "METAL:%d" i

let icb self align buffer (ds : Rig_metal_abi.dispatch array) =
  let sizes = Array.make (7 * Array.length ds) 0 in
  let record i (d : Rig_metal_abi.dispatch) =
    let gx, gy, gz = d.groups and tx, ty, tz = d.threads in
    if d.offset < 0 || d.offset mod align <> 0 then
      invalid_argf
        "Rig_metal_abi.icb: dispatch %d's offset %d, expected a \
         non-negative multiple of %d"
        i d.offset align;
    if gx < 1 || gy < 1 || gz < 1 || tx < 1 || ty < 1 || tz < 1 then
      invalid_argf
        "Rig_metal_abi.icb: dispatch %d has groups %dx%dx%d and threads \
         %dx%dx%d, expected each at least 1"
        i gx gy gz tx ty tz;
    Array.blit [| d.offset; gx; gy; gz; tx; ty; tz |] 0 sizes (7 * i) 7
  in
  Array.iteri record ds;
  let pipelines =
    Array.map (fun (d : Rig_metal_abi.dispatch) -> d.pipeline) ds
  in
  match make_icb self buffer pipelines sizes with
  | "", objects ->
      let released = Atomic.make false in
      let release () =
        if not (Atomic.compare_and_set released false true) then
          invalid_arg "Rig_metal_abi.icb: release called twice";
        release_icb objects.(0)
      in
      let commands = Array.sub objects 2 (Array.length ds) in
      Ok { Rig_metal_abi.handle = objects.(1); commands; release }
  | why, _ -> Error why

(* The minimum constant buffer offset alignment of Apple GPU families, from the
   Metal feature set tables (May 21, 2026, page 7). The tables list none for Mac
   families; 256 meets every smaller power of two and costs only padding. *)
let apple_align = 4
let mac_align = 256

(* The causes of open's failures, by the stubs' codes; the last is the host's
   lack of memory. *)
let open_failures =
  [|
    "Metal exists on macOS only";
    "no GPU of this Mac supports Metal";
    "a device needs macOS 15 or later, for residency sets";
    "the GPU belongs to no Apple or Mac GPU family";
    "Metal made no command queue";
    "Metal made no fence";
    "Metal made no residency set";
    "Metal made no buffer for the word";
  |]

let no_memory = Array.length open_failures + 1

let open_ i =
  if i < 0 then invalid_argf "Rig_metal.open_: GPU %d is negative" i;
  if i > 0 then Error (strf "no such GPU; Metal sees %d" (count ()))
  else
    let self = open_device () in
    if self = -no_memory then raise Out_of_memory
    else if self < 0 then Error open_failures.(-self - 1)
    else
      let family, budget, word = facts self in
      let arch = if family > 0 then strf "Apple%d" family else "Mac2" in
      let align = if family > 0 then apple_align else mac_align in
      let icb = icb self align in
      let word = region self word in
      Ok { self; arch; budget; word; cap = { align; icb; split } }

(* Facts *)

let key = Type.Id.make ()
let arch d = d.arch
let budget d = d.budget
let queues _ = [ "COMPUTE:0" ]
let completion _ = `Host
let waits_on _ _ = false
let blocks _ = `May_block
let capability d = d.cap
let capability_key = Rig_metal_abi.key
let self d = Nativeint.of_int d.self

(* Memory *)

(* Apple documents no alignment for a shared buffer's first byte; Metal places
   one below a page at a multiple of 256 bytes, larger ones at a page. The
   driver promises 256 and checks it. *)
let region_align = 256

let alloc d _ n =
  if n < 1 then
    invalid_argf "Rig_metal.alloc: %d bytes, expected at least 1" n;
  match alloc_buffer d.self n with
  | None -> None
  | Some ((handle, address, host) as b) ->
      if address mod region_align = 0 && host mod region_align = 0 then
        Some (region d.self b)
      else begin
        free_buffer d.self handle;
        raise
          (Fault
             (strf
                "Metal placed %d bytes at GPU address 0x%x and host address \
                 0x%x, expected multiples of %d"
                n address host region_align))
      end

let map_host d p n =
  if n < 1 then
    invalid_argf "Rig_metal.map_host: %d bytes, expected at least 1" n;
  Option.map (region d.self) (map_buffer d.self p n)

let peer _ _ = false

let map_peer d d' r =
  if d.self = d'.self then
    invalid_arg "Rig_metal.map_peer: the two devices are one";
  if r.owner <> d'.self || not (Atomic.get r.live) then
    invalid_arg
      "Rig_metal.map_peer: the region is no live region of the second device";
  None

let free d r =
  if r.owner <> d.self || r == d.word then
    invalid_arg
      "Rig_metal.free: the region is no allocation or mapping of the device";
  if not (Atomic.compare_and_set r.live true false) then
    invalid_arg "Rig_metal.free: the region was freed";
  free_buffer d.self r.handle

(* Images *)

type image = {
  owner : int;
  names : string array;
  pipelines : int array;
  loaded : bool Atomic.t;
}

let image d b =
  match load d.self b with
  | "", names, pipelines ->
      Ok
        (`Loaded { owner = d.self; names; pipelines; loaded = Atomic.make true })
  | why, _, _ -> Error why

let entry i f =
  if not (Atomic.get i.loaded) then
    invalid_arg "Rig_metal.entry: the image was unloaded";
  Array.find_index (String.equal f) i.names
  |> Option.map (Array.get i.pipelines)

let unload d i =
  if i.owner <> d.self then
    invalid_arg "Rig_metal.unload: the image is another device's";
  if not (Atomic.compare_and_set i.loaded true false) then
    invalid_arg "Rig_metal.unload: the image was unloaded";
  Array.iter release i.pipelines

(* Timeline and loss *)

let word d = d.word
let signaled d = signaled_word d.self

let sleep d ~seen ~still_ms =
  if sleep_word d.self seen still_ms <> 0 then raise (Fault (failure d.self))

let stop d = stop_ring d.self
