(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external load : unit -> string option = "caml_nx_cuda_load"

external driver_function : string -> nativeint option
  = "caml_nx_cuda_driver_function"

external host_stamp : unit -> nativeint = "caml_nx_cuda_host_stamp"
external device_count : unit -> int = "caml_nx_cuda_count"
external driver_version : unit -> int = "caml_nx_cuda_driver_version"
external describe : int -> string = "caml_nx_cuda_describe"

external device : int -> int * int * int * int * bool * bool
  = "caml_nx_cuda_device"

external retain : int -> nativeint = "caml_nx_cuda_retain"
external release : int -> unit = "caml_nx_cuda_release"
external stream : nativeint -> nativeint = "caml_nx_cuda_stream"

external stream_destroy : nativeint -> nativeint -> unit
  = "caml_nx_cuda_stream_destroy"

module Driver = Nx_device.Driver
module Region = Driver.Region

external alloc : nativeint -> int -> nativeint option = "caml_nx_cuda_alloc"
external free : nativeint -> nativeint -> unit = "caml_nx_cuda_free"

(* Page-locked host memory's host and device addresses. *)
external host_alloc : nativeint -> int -> (nativeint * nativeint) option
  = "caml_nx_cuda_host_alloc"

external host_free : nativeint -> nativeint -> unit = "caml_nx_cuda_host_free"

external device_pointer : nativeint -> nativeint -> nativeint
  = "caml_nx_cuda_device_pointer"

external register : nativeint -> nativeint -> int -> int
  = "caml_nx_cuda_register"

external unregister : nativeint -> nativeint -> unit = "caml_nx_cuda_unregister"

external pointer :
  nativeint -> nativeint -> (nativeint * nativeint * int) option
  = "caml_nx_cuda_pointer"

external host_pointer : nativeint -> nativeint -> nativeint
  = "caml_nx_cuda_host_pointer"

external copy :
  nativeint ->
  nativeint ->
  nativeint ->
  nativeint ->
  nativeint ->
  int ->
  int ->
  unit = "caml_nx_cuda_copy_byte" "caml_nx_cuda_copy"

external peer :
  nativeint ->
  nativeint ->
  nativeint ->
  nativeint ->
  nativeint ->
  nativeint ->
  int ->
  int ->
  unit = "caml_nx_cuda_peer_byte" "caml_nx_cuda_peer"

external stamp : nativeint -> nativeint -> nativeint -> nativeint -> int -> unit
  = "caml_nx_cuda_stamp"

external signaled : nativeint -> int = "caml_nx_cuda_signaled" [@@noalloc]

external wait :
  nativeint -> nativeint -> nativeint -> nativeint -> int -> int -> bool
  = "caml_nx_cuda_wait_byte" "caml_nx_cuda_wait"

external load_module : nativeint -> string -> nativeint = "caml_nx_cuda_module"

external get_function : nativeint -> nativeint -> string -> nativeint
  = "caml_nx_cuda_function"

type t = {
  dev : Nx_device.t;
  context : nativeint;
  compute : nativeint;
  copy : nativeint;
}

(* Page-locked host memory *)

(* Page-locking is the process's, not a context's: the driver refuses to
   register a range twice. Every device maps a registered range with its own
   device pointer, and the range is unregistered when the last device unmaps it;
   each device unmaps only after its work is done. Memory that [cuMemHostAlloc]
   made is page-locked already and is never registered. *)
type locked = Allocated | Registered of { bytes : int; mutable maps : int }

let locked : (nativeint, locked) Hashtbl.t = Hashtbl.create 16
let locked_lock = Mutex.create ()

let refusal status =
  match status with
  | 1 ->
      describe status
      ^ " (the driver refuses memory mapped read-only, which cannot be \
         borrowed)"
  | 712 -> describe status ^ " (another buffer's memory overlaps it)"
  | _ -> describe status

let map ctx a n =
  Mutex.protect locked_lock @@ fun () ->
  let mapped () = Ok (Region.v ~host:a ~handle:a (device_pointer ctx a) n) in
  match Hashtbl.find_opt locked a with
  | Some Allocated -> mapped ()
  | Some (Registered r) when r.bytes = n ->
      r.maps <- r.maps + 1;
      mapped ()
  | Some (Registered _) -> Error (refusal 712)
  | None -> (
      match register ctx a n with
      | 0 -> (
          match mapped () with
          | m ->
              Hashtbl.replace locked a (Registered { bytes = n; maps = 1 });
              m
          | exception e ->
              unregister ctx a;
              raise e)
      | status -> Error (refusal status))

let unmap ctx r =
  let a = Region.handle r in
  Mutex.protect locked_lock @@ fun () ->
  match Hashtbl.find_opt locked a with
  | Some (Registered r) ->
      r.maps <- r.maps - 1;
      if r.maps = 0 then begin
        unregister ctx a;
        Hashtbl.remove locked a
      end
  | Some Allocated | None -> ()

(* Page-locked memory that is not write-combined: coherent for the host and the
   GPU. *)
let host_memory ctx =
  {
    Driver.alloc =
      (fun n ->
        Option.map
          (fun (host, device) ->
            Mutex.protect locked_lock (fun () ->
                Hashtbl.replace locked host Allocated);
            Region.v ~host ~handle:host device n)
          (host_alloc ctx n));
    free =
      (fun r ->
        let a = Region.handle r in
        Mutex.protect locked_lock (fun () -> Hashtbl.remove locked a);
        host_free ctx a);
  }

(* Opening *)

let lock = Mutex.create ()

(* The driver is loaded and initialized once, by the first call. *)
let driver = ref None

let loaded () =
  match !driver with
  | Some r -> r
  | None ->
      let r = match load () with None -> Ok () | Some msg -> Error msg in
      driver := Some r;
      r

let opened : (int * t) list Atomic.t = Atomic.make []

let find d =
  List.find_map
    (fun (_, c) -> if Nx_device.equal c.dev d then Some c else None)
    (Atomic.get opened)

let count () =
  Mutex.protect lock @@ fun () ->
  match loaded () with
  | Error _ -> 0
  | Ok () -> ( try device_count () with Failure _ -> 0)

let name i = if i = 0 then "CUDA" else Printf.sprintf "CUDA:%d" i

(* Programs are functions of modules, each image loaded once. The driver's
   refusal of an image or a name leaves the context usable. *)
let loader ctx =
  let modules = Hashtbl.create 8 in
  fun ~binary ~entry ->
    match
      let m =
        match Hashtbl.find_opt modules binary with
        | Some m -> m
        | None ->
            let m = load_module ctx binary in
            Hashtbl.add modules binary m;
            m
      in
      get_function ctx m entry
    with
    | f -> Ok f
    | exception Failure why -> Error why

let open_cuda i ~arch ~budget ctx =
  let compute = stream ctx in
  let copy_stream =
    try stream ctx
    with e ->
      stream_destroy ctx compute;
      raise e
  in
  (* Work signals the timeline's first word through its device address. A
     timestamp is the host clock, which a host function stores through the host
     address of the slot's second word. *)
  let queue ~timeline =
    let signal = Region.address timeline in
    let host_word slot =
      Nativeint.(
        add
          (Option.get (Region.host_address timeline))
          (add (sub slot signal) 8n))
    in
    {
      Driver.copy =
        (fun ~dst ~src n ~signal:v -> copy ctx copy_stream signal dst src n v);
      transfer =
        (fun d ->
          Option.map
            (fun c ~dst ~src n ~signal:v ->
              peer ctx copy_stream signal dst c.context src n v)
            (find d));
      stamp =
        (fun ~slot ~signal:v -> stamp ctx copy_stream signal (host_word slot) v);
      clock = Host_clock;
    }
  in
  (* A sleep waits for the signal word to move, and queries the streams each
     millisecond for a fault. *)
  let sleep ~timeline ms =
    let word = Option.get (Region.host_address timeline) in
    ignore (wait ctx compute copy_stream word (signaled word + 1) ms)
  in
  let memory =
    {
      Driver.alloc =
        (fun n -> Option.map (fun d -> Region.v ~handle:d d n) (alloc ctx n));
      free = (fun r -> free ctx (Region.address r));
    }
  in
  match
    Driver.device ~name:(name i) ~arch ~budget ~completion:(Sleep sleep)
      ~load:(loader ctx)
      (Device_local
         {
           memory;
           host_memory = host_memory ctx;
           mapped = None;
           mapping = Pages { map = map ctx; unmap = unmap ctx };
           queue;
         })
  with
  | dev -> { dev; context = ctx; compute; copy = copy_stream }
  | exception e ->
      stream_destroy ctx compute;
      stream_destroy ctx copy_stream;
      raise e

let version () =
  let v = driver_version () in
  Printf.sprintf "%d.%d" (v / 1000) (v mod 1000 / 10)

let get i =
  if i < 0 then invalid_arg (Printf.sprintf "Nx_cuda_device.get: %d < 0" i);
  let refuse fmt =
    Printf.ksprintf (fun why -> Error (name i ^ ": " ^ why)) fmt
  in
  Mutex.protect lock @@ fun () ->
  match List.assoc_opt i (Atomic.get opened) with
  | Some c -> Ok c.dev
  | None -> (
      match loaded () with
      | Error why -> refuse "%s" why
      | Ok () -> (
          try
            let n = device_count () in
            if i >= n then refuse "no such device; there are %d CUDA devices" n
            else
              let ordinal, major, minor, budget, memory_ops, unified =
                device i
              in
              if not memory_ops then
                refuse
                  "the GPU cannot write 64-bit values from its streams (stream \
                   memory operations), which the runtime needs; the driver is \
                   CUDA %s"
                  (version ())
              else if not unified then
                refuse
                  "the GPU has no unified addressing; the driver is CUDA %s"
                  (version ())
              else
                let ctx = retain ordinal in
                let arch = Printf.sprintf "sm_%d%d" major minor in
                match open_cuda i ~arch ~budget ctx with
                | c ->
                    Atomic.set opened ((i, c) :: Atomic.get opened);
                    Ok c.dev
                | exception e ->
                    (* The first error says why; a failing release does not hide
                       it. *)
                    (try release ordinal with Failure _ -> ());
                    raise e
          with Failure why -> refuse "%s" why))

let v i = match get i with Ok d -> d | Error msg -> failwith msg

let cuda fn d =
  match find d with
  | Some c -> c
  | None ->
      invalid_arg
        (Printf.sprintf "Nx_cuda_device.%s: %s is not a CUDA device" fn
           (Nx_device.name d))

let of_device = find
let context c = c.context
let compute c = c.compute
let copy c = c.copy

(* The whole allocation under [a], viewed from [a]. *)
let of_address d a s n =
  let c = cuda "of_address" d in
  let fail fmt =
    Printf.ksprintf
      (fun m -> invalid_arg ("Nx_cuda_device.of_address: " ^ m))
      fmt
  in
  match pointer c.context a with
  | None -> fail "the driver knows no memory at 0x%nx" a
  | Some (owner, _, _) when owner <> c.context ->
      fail "0x%nx is not memory of %s's context" a (Nx_device.name d)
  | Some (_, start, size) ->
      let host =
        match host_pointer c.context start with 0n -> None | host -> Some host
      in
      let region = Region.v ?host ~handle:start start size in
      let whole = Driver.buffer d region Nx_dtype.Scalar.UInt8 size in
      Nx_device.Buffer.view whole
        ~offset:(Nativeint.to_int (Nativeint.sub a start))
        s n

(* Last: it shadows the runtime's own stamp. *)
let stamp = host_stamp ()
