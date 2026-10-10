(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* The C side. A device is the address of its C state, as an int; a Metal object
   is a retained pointer, a pipeline's as an int; a buffer is its location, its
   object as handle; an entry's launch is the address of its C state, struct
   rig_metal_entry, which holds its pipeline. *)

type buffer = Rig_edge.location

external count : unit -> int = "caml_rig_metal_count"
external open_device : unit -> int = "caml_rig_metal_open"
external device_facts : int -> int * int * buffer = "caml_rig_metal_facts"
external alloc_buffer : int -> int -> buffer option = "caml_rig_metal_alloc"

external map_buffer : int -> int -> int -> buffer option
  = "caml_rig_metal_map_host"

external free_buffer : int -> nativeint -> unit = "caml_rig_metal_free"
external free_word : int -> unit = "caml_rig_metal_free_word"
external release : nativeint -> unit = "caml_rig_metal_release"

external load : int -> string -> string * nativeint * string array
  = "caml_rig_metal_image"

external pipeline : nativeint -> string -> nativeint = "caml_rig_metal_pipeline"
external entry_code : nativeint -> int = "caml_rig_metal_entry_code" [@@noalloc]

external entry_threads : nativeint -> int = "caml_rig_metal_entry_threads"
[@@noalloc]

external release_library : nativeint -> unit = "caml_rig_metal_release_library"

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

(* A region is its location, made once by the C side, so [locate] allocates
   nothing. It is live while its device's [regions] holds its handle. *)
type region = Rig_edge.location

let locate (r : region) = r

(* Opening *)

(* An image's entries, [none] until its first [entry], under [guard]: an [entry]
   compiles holding it, so calls for one function from several domains make one
   pipeline. [pipelines] and [device_guard] are its device's. *)
type image = {
  library : nativeint;
  names : string array;
  entries : Rig_edge.entry array;
  guard : Mutex.t;
  pipelines : (int, int) Hashtbl.t;
  device_guard : Mutex.t;
}

let none = { Rig_edge.code = 0; launch = 0n }

type t = {
  self : int;
  facts : region Rig_edge.facts;
  cap : Rig_metal_abi.t;
  guard : Mutex.t;
      (* held by an icb call, by stop, over [pipelines] and [regions] *)
  stopped : bool Atomic.t; (* stop began *)
  pipelines : (int, int) Hashtbl.t;
      (* the pipelines an icb may record, those entries of loaded images gave,
         with each one's most threads per threadgroup *)
  regions : (nativeint, int) Hashtbl.t;
      (* the live regions' handles, its allocations, mappings and word, with
         the bytes an icb may address in each: [0] for the word, which no icb
         takes *)
}

let device_name i =
  if i < 0 then invalid_argf "Rig_metal.device_name: GPU %d is negative" i;
  if i = 0 then "METAL" else strf "METAL:%d" i

(* The most threads per threadgroup of [p], a pipeline in [pipelines], else
   [0]. An icb asks it of every dispatch, so it allocates nothing. *)
let limit pipelines p =
  match Hashtbl.find pipelines p with l -> l | exception Not_found -> 0

(* Why dispatch [i] cannot be recorded with a [bytes]-byte argument buffer:
   [Invalid_argument] for a pipeline [entry] never gave or an offset outside the
   buffer, [Some why] for more threads than its pipeline allows. *)
let refusal pipelines bytes i (d : Rig_metal_abi.dispatch) =
  let max = limit pipelines d.pipeline in
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
   [stop], an [unload] and a [free] exclude each other under the device's
   [guard]: once the stop began, the unload took the image or the free took the
   region, the pipelines [icb] would retain or its buffer may be released. The
   caller may write [ds] meanwhile, so [icb] reads each dispatch once, into its
   own copy, and checks and records that copy: a dispatch read again could hold
   a pipeline the checks never saw. *)
let icb self guard stopped live regions align buffer
    (ds : Rig_metal_abi.dispatch array) =
  let ds = Array.copy ds in
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
    let bytes =
      match Hashtbl.find_opt regions buffer with
      | Some bytes -> bytes
      | None ->
          invalid_arg
            "Rig_metal_abi.icb: the argument buffer is no live region of the \
             device"
    in
    let why = ref None in
    for i = 0 to Array.length ds - 1 do
      let r = refusal live bytes i ds.(i) in
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
      let family, budget, word = device_facts self in
      let arch = if family > 0 then strf "Apple%d" family else "Mac2" in
      let align = if family > 0 then apple_align else mac_align in
      let guard = Mutex.create () and stopped = Atomic.make false in
      let pipelines = Hashtbl.create 16 and regions = Hashtbl.create 64 in
      Hashtbl.replace regions word.handle 0;
      let icb = icb self guard stopped pipelines regions align in
      let cap = { Rig_metal_abi.align; icb; split } in
      let facts =
        {
          Rig_edge.arch;
          budget;
          queues = [ { name = "COMPUTE:0"; runs = [ Fill; Launch ] } ];
          completion = Host;
          waits = { stores = false; hosts = false; objects = false; most = 0 };
          may_block = true;
          hang_ms = None;
          maps_host = true;
          host_addresses = true;
          capability = Capability (Rig_metal_abi.key, cap);
          word;
          edge = Nativeint.of_int self;
        }
      in
      Ok { self; facts; cap; guard; stopped; pipelines; regions }

(* Facts *)

let key = Type.Id.make ()
let facts d = d.facts
let capability d = d.cap

(* Memory *)

(* Apple documents no alignment for a shared buffer's first byte; Metal places
   one below a page at a multiple of 256 bytes, larger ones at a page. The
   driver promises 256 and checks it. *)
let region_align = 256

(* [locked d g r] is [g d r] under [d.guard], which an exception also releases.
   [g] is a toplevel function, so a call makes no closure: every alloc and free
   takes the guard. *)
let locked d g r =
  Mutex.lock d.guard;
  match g d r with
  | x ->
      Mutex.unlock d.guard;
      x
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      Mutex.unlock d.guard;
      Printexc.raise_with_backtrace e bt

let forget d (r : region) = Hashtbl.remove d.regions r.handle

(* Makes [d]'s [n]-byte region [r] live. *)
let enter d (r : region) n =
  Mutex.lock d.guard;
  Hashtbl.replace d.regions r.handle n;
  Mutex.unlock d.guard

let alloc d _ n =
  if n < 1 then invalid_argf "Rig_metal.alloc: %d bytes, expected at least 1" n;
  match alloc_buffer d.self n with
  | None -> None
  | Some r as some ->
      let address = Option.get r.address and host = Option.get r.host in
      if address mod region_align = 0 && host mod region_align = 0 then begin
        enter d r n;
        some
      end
      else begin
        free_buffer d.self r.handle;
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
  match map_buffer d.self p n with
  | None -> None
  | Some r as some ->
      enter d r n;
      some

let peer _ _ = false

let map_peer _ _ _ = None

let free d (r : region) =
  locked d forget r;
  if r == d.facts.word then free_word d.self else free_buffer d.self r.handle

(* Images *)

let image d b =
  match load d.self b with
  | "", library, names ->
      let entries = Array.make (Array.length names) none in
      let guard = Mutex.create () in
      let pipelines = d.pipelines and device_guard = d.guard in
      Ok
        (Rig_edge.Loaded
           { library; names; entries; guard; pipelines; device_guard })
  | why, _, _ -> Error why

(* The index of [f] in [names] from [k], or [-1]. *)
let rec index names f k =
  if k = Array.length names then -1
  else if String.equal (Array.unsafe_get names k) f then k
  else index names f (k + 1)

(* [entry i f] under [i.guard]. A refusal is not kept: a later call compiles
   again. *)
let entry_held (i : image) f =
  let k = index i.names f 0 in
  if k < 0 then None
  else if i.entries.(k).code <> 0 then Some i.entries.(k)
  else
    match pipeline i.library f with
    | launch ->
        let e = { Rig_edge.code = entry_code launch; launch } in
        Mutex.lock i.device_guard;
        Hashtbl.replace i.pipelines e.code (entry_threads launch);
        Mutex.unlock i.device_guard;
        i.entries.(k) <- e;
        Some e
    | exception Failure why ->
        invalid_argf "Rig_metal.entry: Metal makes no pipeline of %S: %s" f why

(* Holds [i.guard], taken without a closure. *)
let entry (i : image) f =
  Mutex.lock i.guard;
  match entry_held i f with
  | e ->
      Mutex.unlock i.guard;
      e
  | exception x ->
      let bt = Printexc.get_raw_backtrace () in
      Mutex.unlock i.guard;
      Printexc.raise_with_backtrace x bt

(* Takes [i]'s pipelines out of the ones an icb may record. *)
let forget_pipelines d (i : image) =
  for k = 0 to Array.length i.entries - 1 do
    Hashtbl.remove d.pipelines i.entries.(k).code
  done

let unload d (i : image) =
  locked d forget_pipelines i;
  Array.iter
    (fun (e : Rig_edge.entry) -> if e.code <> 0 then release e.launch)
    i.entries;
  release_library i.library

(* Timeline and loss *)

let signaled d = signaled_word d.self

let sleep d ~seen ~still_ms =
  if sleep_word d.self seen still_ms <> 0 then raise (Fault (failure d.self))

let stop d ~fault:_ =
  Mutex.protect d.guard @@ fun () ->
  Atomic.set d.stopped true;
  stop_ring d.self
