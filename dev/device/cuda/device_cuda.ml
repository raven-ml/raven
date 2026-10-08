(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Any domain may call any function, as the interface states. The loader, the
   table of GPUs and the page-lock registry are the process's, each behind a
   mutex of its own; a device's C state is written only by [submit] and [stop],
   which their caller serialises. *)

let strf = Printf.sprintf
let invalid_argf fmt = Printf.ksprintf invalid_arg fmt
let ( let* ) = Result.bind

exception Fault of string

(* The CUDA library. A stub with a numeric result returns it as a non-negative
   int, or CUDA's status negated. *)

external load : unit -> int * string = "caml_device_cuda_load"
external error : int -> string = "caml_device_cuda_error"
external symbol : string -> int = "caml_device_cuda_symbol"
external page_size : unit -> int = "caml_device_cuda_page_size"
external driver_version : unit -> int = "caml_device_cuda_driver_version"
external device_count : unit -> int = "caml_device_cuda_count"
external device : int -> int = "caml_device_cuda_device"
external attribute : int -> int -> int = "caml_device_cuda_attribute"
external total_memory : int -> int = "caml_device_cuda_total_memory"

(* From cuda.h *)

let attribute_unified_addressing = 41
let attribute_compute_capability_major = 75
let attribute_compute_capability_minor = 76
let attribute_pci_bus_id = 33
let attribute_pci_device_id = 34
let attribute_pci_domain_id = 50
let attribute_can_use_64_bit_stream_mem_ops = 122
let fault s = raise (Fault (error s))
let get r = if r < 0 then fault (-r) else r

external failed : int -> int = "caml_device_cuda_failed"

(* [refused self x] is [x], the answer to CUDA's refusal of a call's arguments,
   unless the context of the device [self] failed: CUDA answers its error to
   every call, and [refused] raises it. *)
let refused self x = match failed self with 0 -> x | e -> fault e

(* Loads the library and finds its GPUs once, at the first call that needs
   them. *)

type gpus = { devices : int array; busy : bool Atomic.t array }

let lock = Mutex.create ()
let gpus = ref None

let discover () =
  let* () =
    match load () with
    | 0, _ -> Ok ()
    | -1, name ->
        Error
          (strf
             "no CUDA library (%s) on the library search path; install the \
              NVIDIA driver"
             name)
    | -2, name ->
        Error
          (strf "the CUDA library lacks %s; install a newer NVIDIA driver" name)
    | s, _ -> Error ("initialising CUDA: " ^ error s)
  in
  (* GPU [i] is the [i]th by PCI address, whatever order CUDA_DEVICE_ORDER gives
     CUDA's ordinals. *)
  let n = max 0 (device_count ()) in
  let address d =
    List.map
      (fun a -> get (attribute d a))
      [ attribute_pci_domain_id; attribute_pci_bus_id; attribute_pci_device_id ]
  in
  match
    List.init n (fun k ->
        let d = get (device k) in
        (address d, d))
  with
  | exception Fault why -> Error why
  | ds ->
      let devices = Array.of_list (List.map snd (List.sort compare ds)) in
      Ok { devices; busy = Array.map (fun _ -> Atomic.make false) devices }

let find_gpus () =
  Mutex.protect lock @@ fun () ->
  match !gpus with
  | Some r -> r
  | None ->
      let r = discover () in
      gpus := Some r;
      r

(* Memory. Page-locking is the process's: a registration serves every device,
   and ends at the last unmap of a region over it, from any device. *)

type registration = {
  start : int;
  bytes : int;
  address : int; (* where CUDA's work reaches [start] *)
  mutable maps : int;
  mutable stuck : bool; (* CUDA refused its unregistration *)
}

type kind =
  | Device
  | Host
  | Word
  | Locked of registration option (* [None]: another owner's *)
  | Peer of kind

type region = {
  owner : int; (* the device's C state *)
  kind : kind;
  address : int;
  handle : int;
  bytes : int;
  home : int; (* the device whose GPU holds GPU memory *)
  mutable live : bool;
}

let region owner kind ?(home = owner) ~address ~handle bytes =
  { owner; kind; address; handle; bytes; home; live = true }

let rec on_host = function
  | Device -> false
  | Host | Word | Locked _ -> true
  | Peer k -> on_host k

(* Opening *)

type t = {
  self : int;
  arch : string;
  budget : int;
  word : region;
  busy : bool Atomic.t;
}

external open_device : int -> int = "caml_device_cuda_open"
external word_address : int -> int = "caml_device_cuda_word" [@@noalloc]

let count () =
  match find_gpus () with Ok g -> Array.length g.devices | Error _ -> 0

let device_name i =
  if i < 0 then invalid_argf "Device_cuda.device_name: GPU %d is negative" i;
  if i = 0 then "CUDA" else strf "CUDA:%d" i

let version () =
  match driver_version () with
  | v when v < 0 -> "of unknown version"
  | v -> strf "%d.%d" (v / 1000) (v mod 1000 / 10)

let open_ i =
  if i < 0 then invalid_argf "Device_cuda.open_: GPU %d is negative" i;
  let* g = find_gpus () in
  let n = Array.length g.devices in
  if i >= n then
    Error
      (strf "GPU %d does not exist: CUDA sees %d GPU%s" i n
         (if n = 1 then "" else "s"))
  else
    let d = g.devices.(i) in
    match
      let a = attribute d in
      ( get (a attribute_can_use_64_bit_stream_mem_ops),
        get (a attribute_unified_addressing),
        get (a attribute_compute_capability_major),
        get (a attribute_compute_capability_minor),
        get (total_memory d) )
    with
    | exception Fault why -> Error why
    | 0, _, _, _, _ ->
        Error
          (strf "GPU %d lacks 64-bit stream memory operations (CUDA %s)" i
             (version ()))
    | _, 0, _, _, _ ->
        Error (strf "GPU %d lacks unified addressing (CUDA %s)" i (version ()))
    | _, _, major, minor, budget ->
        if not (Atomic.compare_and_set g.busy.(i) false true) then
          Error (strf "GPU %d has a device open; stop it first" i)
        else
          let self = open_device d in
          if self < 0 then begin
            Atomic.set g.busy.(i) false;
            Error (error (-self))
          end
          else
            let w = word_address self in
            let word = region self Word ~address:w ~handle:w 8 in
            Ok
              {
                self;
                arch = strf "sm_%d%d" major minor;
                budget;
                word;
                busy = g.busy.(i);
              }

(* Facts *)

let key = Type.Id.make ()
let arch g = g.arch
let machine _ = None
let budget g = g.budget
let queues _ = [ "COMPUTE:0"; "COPY:0" ]
let completion _ = `Store
let waits_on _ = function `Store | `Host -> true | `Object -> false
let blocks _ = `May_block

type capability = Device_cuda_abi.t

let functions =
  let symbol name =
    if String.contains name '\000' then None
    else match symbol name with 0 -> None | a -> Some (Nativeint.of_int a)
  in
  { Device_cuda_abi.symbol }

let capability _ = functions
let capability_key = Device_cuda_abi.key
let self g = Nativeint.of_int g.self

(* Memory *)

external alloc_memory : int -> bool -> int -> int = "caml_device_cuda_alloc"
external free_memory : int -> bool -> int -> unit = "caml_device_cuda_free"
external mapped : int -> int -> int = "caml_device_cuda_mapped"
external register : int -> int -> int -> int = "caml_device_cuda_register"
external unregister : int -> int -> int = "caml_device_cuda_unregister"
external peer : int -> int -> int = "caml_device_cuda_peer"

let alloc g kind n =
  if n < 1 then
    invalid_argf "Device_cuda.alloc: %d bytes, expected at least 1" n;
  let host = match kind with `Device -> false | `Pinned | `Mapped -> true in
  match alloc_memory g.self host n with
  | a when a >= 0 ->
      Some
        (region g.self (if host then Host else Device) ~address:a ~handle:a n)
  | _ -> refused g.self None

let free g r =
  match r.kind with
  | (Device | Host) when r.owner = g.self ->
      if not r.live then invalid_arg "Device_cuda.free: the region was freed";
      r.live <- false;
      free_memory g.self (r.kind = Host) r.address
  | _ ->
      invalid_arg "Device_cuda.free: the region is no allocation of the device"

let address r = Some r.address
let handle r = Nativeint.of_int r.handle
let host r = if on_host r.kind then Some (Nativeint.of_int r.handle) else None

let map_peer g g' r =
  if g.self = g'.self then
    invalid_arg "Device_cuda.map_peer: the two devices are one";
  if r.owner <> g'.self || not r.live then
    invalid_arg
      "Device_cuda.map_peer: the region is no live region of the second device";
  let kind = match r.kind with Peer k -> k | k -> k in
  let reach =
    on_host kind
    ||
    match peer g.self r.home with
    | 1 -> true
    | 0 -> false
    | _ -> refused g.self false
  in
  if not reach then None
  else Some { r with owner = g.self; kind = Peer kind; live = true }

let registry : (int, registration) Hashtbl.t = Hashtbl.create 16
let registry_lock = Mutex.create ()
let page = page_size ()
let pages a n = (a / page * page, (a + n + page - 1) / page * page)
let locked g e a n address = region g.self (Locked e) ~address ~handle:a n

(* A range CUDA did not page-lock is registered. One it did, for another owner,
   is mapped as it is if both its ends are locked. *)
let page_lock g a n =
  let first = mapped g.self a and last = mapped g.self (a + n - 1) in
  if first >= 0 && last >= 0 then Some (locked g None a n first)
  else if first >= 0 || last >= 0 then None
  else if register g.self a n <> 0 then refused g.self None
  else
    let address = get (mapped g.self a) in
    let e = { start = a; bytes = n; address; maps = 1; stuck = false } in
    Hashtbl.replace registry a e;
    Some (locked g (Some e) a n address)

let map_host g a n =
  if n < 1 then
    invalid_argf "Device_cuda.map_host: %d bytes, expected at least 1" n;
  let a = Nativeint.to_int a in
  let lo, hi = pages a n in
  let inside e = e.start <= a && a + n <= e.start + e.bytes in
  let shares e =
    let lo', hi' = pages e.start e.bytes in
    lo < hi' && lo' < hi
  in
  Mutex.protect registry_lock @@ fun () ->
  let entries = Hashtbl.to_seq_values registry in
  match Seq.find inside entries with
  | Some e when e.stuck -> None
  | Some e ->
      e.maps <- e.maps + 1;
      Some (locked g (Some e) a n (e.address + (a - e.start)))
  | None -> if Seq.exists shares entries then None else page_lock g a n

let unmap g r =
  (match r.kind with
  | (Locked _ | Peer _) when r.owner = g.self -> ()
  | _ -> invalid_arg "Device_cuda.unmap: the region is no mapping of the device");
  if not r.live then invalid_arg "Device_cuda.unmap: the region was unmapped";
  r.live <- false;
  match r.kind with
  | Locked (Some e) ->
      Mutex.protect registry_lock @@ fun () ->
      e.maps <- e.maps - 1;
      if e.maps = 0 && not e.stuck then
        if unregister g.self e.start = 0 then Hashtbl.remove registry e.start
        else e.stuck <- true
  | _ -> ()

(* Images *)

type image = { owner : int; m : int; mutable loaded : bool }

external load_module : int -> string -> int = "caml_device_cuda_load_module"

external get_function : int -> int -> string -> int
  = "caml_device_cuda_function"

external unload_module : int -> int -> int = "caml_device_cuda_unload"

let image g bin =
  match load_module g.self bin with
  | m when m >= 0 -> Ok ({ owner = g.self; m; loaded = true }, None)
  | s -> refused g.self (Error (error (-s)))

let entry (m : image) f =
  if not m.loaded then invalid_arg "Device_cuda.entry: the image was unloaded";
  if String.contains f '\000' then None
  else
    match get_function m.owner m.m f with
    | h when h >= 0 -> Some h
    | _ -> refused m.owner None

let unload g (m : image) =
  if m.owner <> g.self then
    invalid_arg "Device_cuda.unload: the image is another device's";
  if not m.loaded then invalid_arg "Device_cuda.unload: the image was unloaded";
  m.loaded <- false;
  match unload_module g.self m.m with 0 -> () | s -> fault s

(* Work. A part is the ints the C submit reads: the device's C state, then
   nx_part's queue, fill, arg, copy_dst, copy_dst_offset, copy_src,
   copy_src_offset and copy_bytes, then the [after] indices. *)

type part = int array

let after_at = 9

(* nx_edge.h's codes *)

let nx_word = 0
let nx_equal = 1
let nx_ok = 0

external last : int -> int = "caml_device_cuda_last" [@@noalloc]

external submit_parts : int -> int -> int array -> part array -> int
  = "caml_device_cuda_submit"

external failure : int -> string = "caml_device_cuda_failure"
external room_entry : unit -> int = "caml_device_cuda_room_entry"
external submit_entry : unit -> int = "caml_device_cuda_submit_entry"

let part g ~queue ?(after = [||]) w =
  let queue =
    match queue with
    | "COMPUTE:0" -> 0
    | "COPY:0" -> 1
    | q -> invalid_argf "Device_cuda.part: %S is no queue of the device" q
  in
  Array.iter
    (fun j ->
      if j < 0 then
        invalid_argf "Device_cuda.part: after index %d is negative" j)
    after;
  let part work = Array.concat [ [| g.self; queue |]; work; after ] in
  match w with
  | `Words _ -> invalid_arg "Device_cuda.part: a CUDA device has no ring words"
  | `Fill (f, arg, units, bytes) ->
      if units <> 0 || bytes <> 0 then
        invalid_argf
          "Device_cuda.part: the fill declares %d ring units and %d segment \
           bytes, expected 0"
          units bytes;
      part [| Nativeint.to_int f; Nativeint.to_int arg; 0; 0; 0; 0; 0 |]
  | `Copy ((dst, o), (src, o'), n) ->
      let check what (r : region) o =
        if r.owner <> g.self || not r.live then
          invalid_argf
            "Device_cuda.part: the copy's %s is no live region of the device"
            what;
        if o < 0 || n < 0 || o + n > r.bytes then
          invalid_argf
            "Device_cuda.part: the copy's %s range [%d, %d) lies outside its \
             %d bytes"
            what o (o + n) r.bytes
      in
      check "destination" dst o;
      check "source" src o';
      part [| 0; 0; dst.handle; o; src.handle; o'; n |]

let room _ _ = `Fits

let check_part self i (p : part) =
  if p.(0) <> self then
    invalid_argf "Device_cuda.submit: part %d is another device's" i;
  for k = after_at to Array.length p - 1 do
    if p.(k) >= i then
      invalid_argf
        "Device_cuda.submit: part %d runs after part %d, expected an earlier \
         one"
        i p.(k)
  done

let wait_kind = function
  | `Word -> nx_word
  | `Equal -> nx_equal
  | `Object ->
      invalid_arg "Device_cuda.submit: a CUDA device waits on no driver object"

(* Allocates nothing for a submission without waits. *)
let submit g ~v ~waits ~handles:_ parts =
  let next = last g.self + 1 in
  if v <> next then
    invalid_argf "Device_cuda.submit: value %d, expected %d" v next;
  for i = 0 to Array.length parts - 1 do
    check_part g.self i parts.(i)
  done;
  let words = Array.make (3 * Array.length waits) 0 in
  for k = 0 to Array.length waits - 1 do
    let kind, at, w = waits.(k) in
    words.(3 * k) <- wait_kind kind;
    words.((3 * k) + 1) <- at;
    words.((3 * k) + 2) <- w
  done;
  if submit_parts g.self v words parts = nx_ok then `Ok
  else `Failed (failure g.self)

let room_entry = Nativeint.of_int (room_entry ())
let submit_entry = Nativeint.of_int (submit_entry ())

(* Timeline *)

external signaled : int -> int = "caml_device_cuda_signaled" [@@noalloc]
external sleep : int -> int -> int -> int = "caml_device_cuda_sleep"
external stop : int -> bool = "caml_device_cuda_stop"

let word g = g.word
let signaled g = signaled g.self

let sleep g ~seen ~still_ms =
  match sleep g.self seen still_ms with 0 -> () | s -> fault s

(* Loss *)

let stop g =
  let stopped = stop g.self in
  Atomic.set g.busy false;
  if stopped then `Stopped else `Unknown
