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
external free_word : int -> unit = "caml_rig_metal_free_word"
external release : int -> unit = "caml_rig_metal_release"

external load : int -> string -> string * nativeint * string array
  = "caml_rig_metal_image"

external pipeline : nativeint -> string -> string * int * int
  = "caml_rig_metal_pipeline"

external release_library : nativeint -> unit = "caml_rig_metal_release_library"
external buffer_bytes : int -> nativeint -> int = "caml_rig_metal_buffer_bytes"

external make_icb :
  int -> nativeint -> int array -> int array -> nativeint array
  = "caml_rig_metal_icb"

external release_icb : nativeint -> unit = "caml_rig_metal_icb_release"

external signaled_word : (int[@untagged]) -> (int[@untagged])
  = "caml_rig_metal_signaled_byte" "caml_rig_metal_signaled"
[@@noalloc]

external sleep_word : int -> int -> int -> int = "caml_rig_metal_sleep"
external failure : int -> string = "caml_rig_metal_failure"
external stop_ring : int -> unit = "caml_rig_metal_stop"

external split : unit -> nativeint = "caml_rig_metal_split"

let split = split ()

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

(* An image's pipelines, 0 until its first [entry], and whether it is loaded,
   both under [guard]: an [entry] compiles holding it, so calls for one function
   from several domains make one pipeline, and an [unload] either waits for a
   compile or makes the [entry] after it raise. [limits] holds each pipeline's
   most threads per threadgroup. *)
type image = {
  owner : int;
  library : nativeint;
  names : string array;
  pipelines : int array;
  limits : int array;
  guard : Mutex.t;
  mutable loaded : bool;
}

type t = {
  self : int;
  arch : string;
  budget : int;
  word : region;
  cap : capability;
  guard : Mutex.t; (* held by an icb call, by stop and over [images] *)
  stopped : bool Atomic.t; (* stop began *)
  images : image list ref; (* loaded, whose pipelines an icb may record *)
}

let device_name i =
  if i < 0 then invalid_argf "Rig_metal.device_name: GPU %d is negative" i;
  if i = 0 then "METAL" else strf "METAL:%d" i

(* The most threads per threadgroup of [p], a pipeline that [entry] gave for one
   of [images], else [0]. An icb asks it of every dispatch, so it allocates
   nothing. *)
let rec limit images p =
  match images with
  | [] -> 0
  | (i : image) :: rest ->
      let n = Array.length i.pipelines in
      let k = ref 0 in
      while !k < n && i.pipelines.(!k) <> p do
        incr k
      done;
      if !k < n && p <> 0 then i.limits.(!k) else limit rest p

(* Why dispatch [i] cannot be recorded with a [bytes]-byte argument buffer:
   [Invalid_argument] for a pipeline [entry] never gave or an offset outside the
   buffer, [Some why] for more threads than its pipeline allows. *)
let refusal images bytes i (d : Rig_metal_abi.dispatch) =
  let max = limit images d.pipeline in
  if max = 0 then
    invalid_argf
      "Rig_metal_abi.icb: dispatch %d's pipeline is of no image the device \
       loaded"
      i;
  if d.offset >= bytes then
    invalid_argf
      "Rig_metal_abi.icb: dispatch %d's offset %d lies outside the buffer's %d \
       bytes"
      i d.offset bytes;
  (* The product of three sizes can overflow, so each is compared with the bound
     divided by the ones before it; every size is at least 1. *)
  let tx, ty, tz = d.threads in
  if tx > max || ty > max / tx || tz > max / (tx * ty) then
    Some
      (strf
         "dispatch %d asks for %dx%dx%d threads per threadgroup, expected at \
          most %d in all"
         i tx ty tz max)
  else None

(* Compiled code calls a device's [icb] beside every other call, so [icb],
   [stop] and an [unload] exclude each other under the device's [guard]: once
   the stop began or the unload took the image, the pipelines [icb] would retain
   may be released. *)
let icb self guard stopped images align buffer
    (ds : Rig_metal_abi.dispatch array) =
  let sizes = Array.make (7 * Array.length ds) 0 in
  let record i (d : Rig_metal_abi.dispatch) =
    let gx, gy, gz = d.groups and tx, ty, tz = d.threads in
    if d.offset < 0 || d.offset mod align <> 0 then
      invalid_argf
        "Rig_metal_abi.icb: dispatch %d's offset %d, expected a non-negative \
         multiple of %d"
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
  Mutex.protect guard @@ fun () ->
  if Atomic.get stopped then Error "the device was stopped"
  else
    let bytes = buffer_bytes self buffer in
    if bytes < 0 then
      invalid_arg "Rig_metal_abi.icb: the argument buffer is another GPU's";
    let why = ref None in
    for i = 0 to Array.length ds - 1 do
      let r = refusal !images bytes i ds.(i) in
      if Option.is_none !why then why := r
    done;
    match !why with
    | Some why -> Error why
    | None -> (
        match make_icb self buffer pipelines sizes with
        | [||] -> Error "Metal made no indirect command buffer"
        | objects ->
            let released = Atomic.make false in
            let release () =
              if not (Atomic.compare_and_set released false true) then
                invalid_arg "Rig_metal_abi.icb: release called twice";
              release_icb objects.(0)
            in
            let commands = Array.sub objects 2 (Array.length ds) in
            Ok { Rig_metal_abi.handle = objects.(1); commands; release })

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
      let guard = Mutex.create () and stopped = Atomic.make false in
      let images = ref [] in
      let icb = icb self guard stopped images align in
      let word = region self word in
      let cap = { Rig_metal_abi.align; icb; split } in
      Ok { self; arch; budget; word; cap; guard; stopped; images }

(* Facts *)

let key = Type.Id.make ()
let arch d = d.arch
let budget d = d.budget
let queues _ = [ "COMPUTE:0" ]
let completion _ = `Host
let waits_on _ _ = false
let max_waits _ = 0
let blocks _ = `May_block
let maps_host _ = true
let capability d = d.cap
let capability_key = Rig_metal_abi.key
let edge d = Nativeint.of_int d.self

(* Memory *)

(* Apple documents no alignment for a shared buffer's first byte; Metal places
   one below a page at a multiple of 256 bytes, larger ones at a page. The
   driver promises 256 and checks it. *)
let region_align = 256

let alloc d _ n =
  if n < 1 then invalid_argf "Rig_metal.alloc: %d bytes, expected at least 1" n;
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

let map_peer d d' (r : region) =
  if d.self = d'.self then
    invalid_arg "Rig_metal.map_peer: the two devices are one";
  if r.owner <> d'.self || not (Atomic.get r.live) then
    invalid_arg
      "Rig_metal.map_peer: the region is no live region of the second device";
  None

let free d (r : region) =
  if r.owner <> d.self then
    invalid_arg
      "Rig_metal.free: the region is no allocation or mapping of the device";
  if not (Atomic.compare_and_set r.live true false) then
    invalid_arg "Rig_metal.free: the region was freed";
  if r == d.word then free_word d.self else free_buffer d.self r.handle

(* Images *)

let image d b =
  match load d.self b with
  | "", library, names ->
      let n = Array.length names in
      let pipelines = Array.make n 0 and limits = Array.make n 0 in
      let i =
        {
          owner = d.self;
          library;
          names;
          pipelines;
          limits;
          guard = Mutex.create ();
          loaded = true;
        }
      in
      Mutex.protect d.guard (fun () -> d.images := i :: !(d.images));
      Ok (`Loaded i)
  | why, _, _ -> Error why

(* A refusal is not kept: a later call compiles again. *)
let entry (i : image) f =
  Mutex.protect i.guard @@ fun () ->
  if not i.loaded then invalid_arg "Rig_metal.entry: the image was unloaded";
  match Array.find_index (String.equal f) i.names with
  | None -> None
  | Some k when i.pipelines.(k) <> 0 -> Some i.pipelines.(k)
  | Some k -> (
      match pipeline i.library f with
      | "", p, limit ->
          i.limits.(k) <- limit;
          i.pipelines.(k) <- p;
          Some p
      | why, _, _ ->
          invalid_argf "Rig_metal.entry: Metal makes no pipeline of %S: %s" f
            why)

let unload d (i : image) =
  if i.owner <> d.self then
    invalid_arg "Rig_metal.unload: the image is another device's";
  Mutex.protect i.guard @@ fun () ->
  if not i.loaded then invalid_arg "Rig_metal.unload: the image was unloaded";
  i.loaded <- false;
  Mutex.protect d.guard (fun () ->
      d.images := List.filter (fun j -> j != i) !(d.images));
  Array.iter (fun p -> if p <> 0 then release p) i.pipelines;
  release_library i.library

(* Timeline and loss *)

let word d = d.word
let signaled d = signaled_word d.self

let sleep d ~seen ~still_ms =
  if sleep_word d.self seen still_ms <> 0 then raise (Fault (failure d.self))

let stop d =
  Mutex.protect d.guard @@ fun () ->
  Atomic.set d.stopped true;
  stop_ring d.self
