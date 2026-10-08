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

external count : unit -> int = "caml_device_metal_count"
external open_device : unit -> int = "caml_device_metal_open"
external facts : int -> int * int * buffer = "caml_device_metal_facts"
external alloc_buffer : int -> int -> buffer option = "caml_device_metal_alloc"

external map_buffer : int -> int -> int -> buffer option
  = "caml_device_metal_map_host"

external free_buffer : int -> nativeint -> unit = "caml_device_metal_free"
external release : int -> unit = "caml_device_metal_release"

external load : int -> string -> string * string array * int array
  = "caml_device_metal_image"

external make_icb :
  int -> nativeint -> int array -> int array -> string * nativeint array
  = "caml_device_metal_icb"

external release_icb : nativeint -> unit = "caml_device_metal_icb_release"

external signaled_word : (int[@untagged]) -> (int[@untagged])
  = "caml_device_metal_signaled_byte" "caml_device_metal_signaled"
[@@noalloc]

external last : (int[@untagged]) -> (int[@untagged])
  = "caml_device_metal_last_byte" "caml_device_metal_last"
[@@noalloc]

external sleep_word : int -> int -> int -> int = "caml_device_metal_sleep"
external failure : int -> string = "caml_device_metal_failure"
external stop_ring : int -> unit = "caml_device_metal_stop"

external entries : unit -> nativeint * nativeint * nativeint
  = "caml_device_metal_entries"

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

type capability = Device_metal_abi.t

type t = {
  self : int;
  arch : string;
  budget : int;
  word : region;
  cap : capability;
}

let device_name i =
  if i < 0 then invalid_argf "Device_metal.device_name: GPU %d is negative" i;
  if i = 0 then "METAL" else strf "METAL:%d" i

let icb self align buffer (ds : Device_metal_abi.dispatch array) =
  let sizes = Array.make (7 * Array.length ds) 0 in
  let record i (d : Device_metal_abi.dispatch) =
    let gx, gy, gz = d.groups and tx, ty, tz = d.threads in
    if d.offset < 0 || d.offset mod align <> 0 then
      invalid_argf
        "Device_metal_abi.icb: dispatch %d's offset %d, expected a \
         non-negative multiple of %d"
        i d.offset align;
    if gx < 1 || gy < 1 || gz < 1 || tx < 1 || ty < 1 || tz < 1 then
      invalid_argf
        "Device_metal_abi.icb: dispatch %d has groups %dx%dx%d and threads \
         %dx%dx%d, expected each at least 1"
        i gx gy gz tx ty tz;
    Array.blit [| d.offset; gx; gy; gz; tx; ty; tz |] 0 sizes (7 * i) 7
  in
  Array.iteri record ds;
  let pipelines =
    Array.map (fun (d : Device_metal_abi.dispatch) -> d.pipeline) ds
  in
  match make_icb self buffer pipelines sizes with
  | "", objects ->
      let released = Atomic.make false in
      let release () =
        if not (Atomic.compare_and_set released false true) then
          invalid_arg "Device_metal_abi.icb: release called twice";
        release_icb objects.(0)
      in
      let commands = Array.sub objects 2 (Array.length ds) in
      Ok { Device_metal_abi.handle = objects.(1); commands; release }
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
  if i < 0 then invalid_argf "Device_metal.open_: GPU %d is negative" i;
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
let capability_key = Device_metal_abi.key
let self d = Nativeint.of_int d.self

(* Memory *)

(* Apple documents no alignment for a shared buffer's first byte; Metal places
   one below a page at a multiple of 256 bytes, larger ones at a page. The
   driver promises 256 and checks it. *)
let region_align = 256

let alloc d _ n =
  if n < 1 then
    invalid_argf "Device_metal.alloc: %d bytes, expected at least 1" n;
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
    invalid_argf "Device_metal.map_host: %d bytes, expected at least 1" n;
  Option.map (region d.self) (map_buffer d.self p n)

let peer _ _ = false

let map_peer d d' r =
  if d.self = d'.self then
    invalid_arg "Device_metal.map_peer: the two devices are one";
  if r.owner <> d'.self || not (Atomic.get r.live) then
    invalid_arg
      "Device_metal.map_peer: the region is no live region of the second device";
  None

let free d r =
  if r.owner <> d.self || r == d.word then
    invalid_arg
      "Device_metal.free: the region is no allocation or mapping of the device";
  if not (Atomic.compare_and_set r.live true false) then
    invalid_arg "Device_metal.free: the region was freed";
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
    invalid_arg "Device_metal.entry: the image was unloaded";
  Array.find_index (String.equal f) i.names
  |> Option.map (Array.get i.pipelines)

let unload d i =
  if i.owner <> d.self then
    invalid_arg "Device_metal.unload: the image is another device's";
  if not (Atomic.compare_and_set i.loaded true false) then
    invalid_arg "Device_metal.unload: the image was unloaded";
  Array.iter release i.pipelines

(* Work *)

(* A part is its device and the ints the C submit reads: nx_part's int fields in
   its order (queue, fill, arg, ring_units, segment_bytes, copy_dst,
   copy_dst_offset, copy_src, copy_src_offset, copy_bytes), then the [after]
   indices. *)
type part = { owner : int; ints : int array }

let after_at = 10

(* nx_edge.h's answer for a submission handed over. *)
let nx_ok = 0

let part d ~queue ?(after = [||]) w =
  if queue <> "COMPUTE:0" then
    invalid_argf "Device_metal.part: queue %S, expected COMPUTE:0" queue;
  let negative i =
    if i < 0 then invalid_argf "Device_metal.part: after index %d is negative" i
  in
  Array.iter negative after;
  match w with
  | `Fill (fill, arg, 0, 0) ->
      let fill = Nativeint.to_int fill and arg = Nativeint.to_int arg in
      let ints = [| 0; fill; arg; 0; 0; 0; 0; 0; 0; 0 |] in
      { owner = d.self; ints = Array.append ints after }
  | `Fill (_, _, units, bytes) ->
      invalid_argf
        "Device_metal.part: a fill declares %d ring units and %d segment \
         bytes, expected 0 of each"
        units bytes
  | `Words _ -> invalid_arg "Device_metal.part: the device runs no words"
  | `Copy _ -> invalid_arg "Device_metal.part: the device runs no copies"

let room _ _ = `Fits

external submit_parts : int -> int -> part array -> int
  = "caml_device_metal_submit"

let check_part self i p =
  if p.owner <> self then
    invalid_argf "Device_metal.submit: part %d is another device's" i;
  for k = after_at to Array.length p.ints - 1 do
    if p.ints.(k) >= i then
      invalid_argf
        "Device_metal.submit: part %d waits for part %d, expected an earlier \
         part"
        i p.ints.(k)
  done

let submit d ~v ~waits ~handles:_ ps =
  let next = last d.self + 1 in
  if v <> next then
    invalid_argf "Device_metal.submit: value %d, expected %d" v next;
  if Array.length waits > 0 then
    invalid_arg "Device_metal.submit: the device waits on no word";
  for i = 0 to Array.length ps - 1 do
    check_part d.self i ps.(i)
  done;
  if submit_parts d.self v ps = nx_ok then `Ok else `Failed (failure d.self)

(* Timeline and loss *)

let word d = d.word
let signaled d = signaled_word d.self

let sleep d ~seen ~still_ms =
  if sleep_word d.self seen still_ms <> 0 then raise (Fault (failure d.self))

let stop d = stop_ring d.self
