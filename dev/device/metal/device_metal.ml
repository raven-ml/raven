(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* The C side. A device is the address of its C state, as an int; a Metal object
   is a retained pointer; a buffer is the triple of its object, GPU address and
   host address. *)

type buffer = nativeint * int * nativeint

external count : unit -> int = "caml_device_metal_count"
external open_device : unit -> int = "caml_device_metal_open"
external facts : int -> int * int * buffer = "caml_device_metal_facts"
external alloc_buffer : int -> int -> buffer option = "caml_device_metal_alloc"

external map_buffer : int -> nativeint -> int -> buffer option
  = "caml_device_metal_map_host"

external free_buffer : int -> nativeint -> unit = "caml_device_metal_free"
external release : nativeint -> unit = "caml_device_metal_release"

external load : int -> string -> string * string array * nativeint array
  = "caml_device_metal_image"

external make_icb :
  int -> nativeint -> int array -> int array -> string * nativeint array
  = "caml_device_metal_icb"

external release_icb : int -> nativeint -> unit
  = "caml_device_metal_icb_release"

external signaled_word : (int[@untagged]) -> (int[@untagged])
  = "caml_device_metal_signaled_byte" "caml_device_metal_signaled"
[@@noalloc]

external last : (int[@untagged]) -> (int[@untagged])
  = "caml_device_metal_last_byte" "caml_device_metal_last"
[@@noalloc]

external sleep_word : int -> int -> int -> string option
  = "caml_device_metal_sleep"

external stop_ring : int -> bool = "caml_device_metal_stop"

external entries : unit -> nativeint * nativeint * nativeint
  = "caml_device_metal_entries"

let room_entry, submit_entry, split = entries ()

exception Fault of string

(* Memory *)

type kind = Alloc | Mapped | Word

type region = {
  owner : int;
  kind : kind;
  handle : nativeint;
  address : int;
  host : nativeint;
  live : bool Atomic.t;
}

let region owner kind (handle, address, host) =
  { owner; kind; handle; address; host; live = Atomic.make true }

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
  if i < 0 then invalid_argf "Device_metal.device_name: device %d is negative" i;
  if i = 0 then "METAL" else strf "METAL:%d" i

let icb self align buffer (ds : Device_metal_abi.dispatch array) =
  let check i (d : Device_metal_abi.dispatch) =
    let gx, gy, gz = d.groups and tx, ty, tz = d.threads in
    if d.offset < 0 || d.offset mod align <> 0 then
      invalid_argf
        "Device_metal_abi.icb: dispatch %d's offset %d is no multiple of %d in \
         the buffer"
        i d.offset align;
    if List.exists (fun x -> x < 1) [ gx; gy; gz; tx; ty; tz ] then
      invalid_argf "Device_metal_abi.icb: dispatch %d has a size below 1" i
  in
  Array.iteri check ds;
  let sizes (d : Device_metal_abi.dispatch) =
    let gx, gy, gz = d.groups and tx, ty, tz = d.threads in
    [| d.offset; gx; gy; gz; tx; ty; tz |]
  in
  let pipelines =
    Array.map (fun (d : Device_metal_abi.dispatch) -> d.pipeline) ds
  in
  let sizes = Array.concat (Array.to_list (Array.map sizes ds)) in
  match make_icb self buffer pipelines sizes with
  | "", objects ->
      let released = Atomic.make false in
      let release () =
        if not (Atomic.compare_and_set released false true) then
          invalid_arg "Device_metal_abi.icb: release called twice";
        release_icb self objects.(0)
      in
      let commands = Array.sub objects 2 (Array.length ds) in
      Ok { Device_metal_abi.handle = objects.(1); commands; release }
  | why, _ -> Error why

(* The minimum constant buffer offset alignment of Apple GPU families, from the
   Metal feature set tables (May 21, 2026, page 7). The tables list none for Mac
   families; 256 meets every smaller power of two and costs only padding. *)
let apple_align = 4
let mac_align = 256

(* The causes of open's failures, by the stubs' codes, and the code of the
   host's lack of memory. *)
let no_memory = 6

let open_failure = function
  | 1 -> "Metal exists on macOS only"
  | 2 -> "no GPU of this Mac supports Metal"
  | 3 -> "a device needs macOS 15 or later, for residency sets"
  | 4 -> "the GPU belongs to no Apple or Mac GPU family"
  | _ -> "Metal made no queue, fence, residency set or word for the GPU"

let open_ i =
  if i < 0 then invalid_argf "Device_metal.open_: device %d is negative" i;
  if i > 0 then Error (strf "no device %d: a Mac has one GPU, device 0" i)
  else
    let self = open_device () in
    if self = -no_memory then raise Out_of_memory
    else if self < 0 then Error (open_failure (-self))
    else
      let family, budget, word = facts self in
      let arch = if family > 0 then strf "Apple%d" family else "Mac2" in
      let align = if family > 0 then apple_align else mac_align in
      let icb = icb self align in
      let word = region self Word word in
      Ok { self; arch; budget; word; cap = { align; icb; split } }

(* Facts *)

let key = Type.Id.make ()
let arch d = d.arch
let machine _ = None
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
  if n < 1 then invalid_argf "Device_metal.alloc: %d bytes, below 1" n;
  match alloc_buffer d.self n with
  | None -> None
  | Some ((handle, address, host) as b) ->
      let host = Nativeint.to_int host in
      if address mod region_align = 0 && host mod region_align = 0 then
        Some (region d.self Alloc b)
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
  if n < 1 then invalid_argf "Device_metal.map_host: %d bytes, below 1" n;
  Option.map (region d.self Mapped) (map_buffer d.self p n)

let map_peer _ _ _ = None

let give_back fn kind d r =
  if r.owner <> d.self || r.kind <> kind then
    invalid_argf "Device_metal.%s: the region is none of the device's %s" fn
      (if kind = Alloc then "allocations" else "mapped ranges");
  if not (Atomic.compare_and_set r.live true false) then
    invalid_argf "Device_metal.%s: the region was given back" fn;
  free_buffer d.self r.handle

let free d r = give_back "free" Alloc d r
let unmap d r = give_back "unmap" Mapped d r

(* Images *)

type image = { names : string array; pipelines : nativeint array }

let image d b =
  match load d.self b with
  | "", names, pipelines -> Ok ({ names; pipelines }, None)
  | why, _, _ -> Error why

let entry i f =
  Array.find_index (String.equal f) i.names
  |> Option.map (fun k -> Nativeint.to_int i.pipelines.(k))

let unload _ i = Array.iter release i.pipelines

(* Work *)

(* A part is the ints the C submit reads: the device's C state, then nx_part's
   queue, fill, arg, copy_dst, copy_dst_offset, copy_src, copy_src_offset and
   copy_bytes, then the [after] indices. *)
type part = int array

let after_at = 9

let part d ~queue ?(after = [||]) w =
  if queue <> "COMPUTE:0" then
    invalid_argf "Device_metal.part: queue %S is not COMPUTE:0" queue;
  let negative i =
    if i < 0 then invalid_argf "Device_metal.part: part index %d is negative" i
  in
  Array.iter negative after;
  match w with
  | `Fill (fill, arg, 0, 0) ->
      let fill = Nativeint.to_int fill and arg = Nativeint.to_int arg in
      Array.append [| d.self; 0; fill; arg; 0; 0; 0; 0; 0 |] after
  | `Fill (_, _, units, bytes) ->
      invalid_argf
        "Device_metal.part: a fill declares %d ring units and %d segment \
         bytes, not none"
        units bytes
  | `Words _ -> invalid_arg "Device_metal.part: the device runs no words"
  | `Copy _ ->
      invalid_arg "Device_metal.part: the device runs no copies; the host does"

let room _ _ = `Fits

external submit_parts : int -> int -> part array -> string option
  = "caml_device_metal_submit"

let check_part self i (p : part) =
  if p.(0) <> self then
    invalid_argf "Device_metal.submit: part %d is another device's" i;
  for k = after_at to Array.length p - 1 do
    if p.(k) >= i then
      invalid_argf "Device_metal.submit: part %d waits for part %d, not earlier"
        i p.(k)
  done

let submit d ~v ~waits ~handles:_ ps =
  let next = last d.self + 1 in
  if v <> next then invalid_argf "Device_metal.submit: value %d, not %d" v next;
  if Array.length waits > 0 then
    invalid_arg "Device_metal.submit: the device waits on no word";
  Array.iteri (check_part d.self) ps;
  match submit_parts d.self v ps with None -> `Ok | Some why -> `Failed why

(* Timeline and loss *)

let word d = d.word
let signaled d = signaled_word d.self

let sleep d ~seen ~still_ms =
  if still_ms < 0 then
    invalid_argf "Device_metal.sleep: still_ms %d is negative" still_ms;
  match sleep_word d.self seen still_ms with
  | None -> ()
  | Some why -> raise (Fault why)

let stop d = if stop_ring d.self then `Stopped else `Unknown
