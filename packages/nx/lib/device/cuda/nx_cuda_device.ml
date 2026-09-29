(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external load : unit -> string option = "caml_nx_cuda_load"
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

external alloc : nativeint -> int -> Nx_device.memory option
  = "caml_nx_cuda_alloc"

external free : nativeint -> nativeint -> unit = "caml_nx_cuda_free"

external host_alloc : nativeint -> int -> Nx_device.memory option
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

external signaled : nativeint -> int = "caml_nx_cuda_signaled" [@@noalloc]

external wait :
  nativeint -> nativeint -> nativeint -> nativeint -> int -> int -> bool
  = "caml_nx_cuda_wait_byte" "caml_nx_cuda_wait"

external load_module : nativeint -> string -> nativeint = "caml_nx_cuda_module"

external get_function : nativeint -> nativeint -> string -> nativeint
  = "caml_nx_cuda_function"

type handles = {
  context : nativeint;
  compute : nativeint;
  copy : nativeint;
  signal : nativeint;
}

type cuda = { dev : Nx_device.t; handles : handles }

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
  let mapped () =
    Ok { Nx_device.host = Some a; device = device_pointer ctx a; handle = a }
  in
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

let unmap ctx (m : Nx_device.memory) =
  Mutex.protect locked_lock @@ fun () ->
  match Hashtbl.find_opt locked m.handle with
  | Some (Registered r) ->
      r.maps <- r.maps - 1;
      if r.maps = 0 then begin
        unregister ctx m.handle;
        Hashtbl.remove locked m.handle
      end
  | Some Allocated | None -> ()

let host_memory ctx =
  {
    Nx_device.alloc =
      (fun n ->
        let m = host_alloc ctx n in
        Option.iter
          (fun (m : Nx_device.memory) ->
            Mutex.protect locked_lock (fun () ->
                Hashtbl.replace locked m.handle Allocated))
          m;
        m);
    free =
      (fun m ->
        Mutex.protect locked_lock (fun () -> Hashtbl.remove locked m.handle);
        host_free ctx m.handle);
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

let opened : (int * cuda) list Atomic.t = Atomic.make []

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

(* Programs are functions of modules, each image loaded once. *)
let loader ctx =
  let modules = Hashtbl.create 8 in
  fun ~binary ~name ->
    let m =
      match Hashtbl.find_opt modules binary with
      | Some m -> m
      | None ->
          let m = load_module ctx binary in
          Hashtbl.add modules binary m;
          m
    in
    get_function ctx m name

let open_cuda i ~arch ~budget ctx =
  let compute = stream ctx in
  let copy_stream =
    try stream ctx
    with e ->
      stream_destroy ctx compute;
      raise e
  in
  (* Work signals the timeline's first word through its device address. *)
  let copy_queue (timeline : Nx_device.memory) =
    let signal = timeline.device in
    {
      Nx_device.copy =
        (fun ~dst ~src n v -> copy ctx copy_stream signal dst src n v);
      transfer =
        (fun d ->
          Option.map
            (fun c ~dst ~src n v ->
              peer ctx copy_stream signal dst c.handles.context src n v)
            (find d));
    }
  in
  let signal (timeline : Nx_device.memory) =
    let word = Option.get timeline.host in
    {
      Nx_device.signaled = (fun () -> signaled word);
      wait =
        (fun v ~timeout_ms -> wait ctx compute copy_stream word v timeout_ms);
    }
  in
  match
    Nx_device.make ~name:(name i) ~arch ~budget
      ~memory:
        {
          alloc = alloc ctx;
          free = (fun (m : Nx_device.memory) -> free ctx m.device);
        }
      ~host_memory:(host_memory ctx)
      ~mapping:{ map = map ctx; unmap = unmap ctx }
      ~copy_queue ~load:(loader ctx) ~signal ()
  with
  | dev ->
      let signal = Nx_device.Buffer.address (Nx_device.timeline dev) in
      { dev; handles = { context = ctx; compute; copy = copy_stream; signal } }
  | exception e ->
      stream_destroy ctx compute;
      stream_destroy ctx copy_stream;
      raise e

let version () =
  let v = driver_version () in
  Printf.sprintf "%d.%d" (v / 1000) (v mod 1000 / 10)

let get i =
  if i < 0 then invalid_arg (Printf.sprintf "Nx_cuda_device.get: %d < 0" i);
  Mutex.protect lock @@ fun () ->
  match List.assoc_opt i (Atomic.get opened) with
  | Some c -> Ok c.dev
  | None -> (
      match loaded () with
      | Error _ as e -> e
      | Ok () -> (
          try
            let n = device_count () in
            if i >= n then
              Error
                (Printf.sprintf "CUDA: no device %d; there are %d CUDA devices"
                   i n)
            else
              let ordinal, major, minor, budget, memory_ops, unified =
                device i
              in
              if not memory_ops then
                Error
                  (Printf.sprintf
                     "CUDA: %s cannot write 64-bit values from its streams \
                      (stream memory operations), which the runtime needs; the \
                      driver is CUDA %s"
                     (name i) (version ()))
              else if not unified then
                Error
                  (Printf.sprintf
                     "CUDA: %s has no unified addressing; the driver is CUDA %s"
                     (name i) (version ()))
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
          with Failure msg -> Error ("CUDA: " ^ msg)))

let v i = match get i with Ok d -> d | Error msg -> invalid_arg msg

let cuda fn d =
  match find d with
  | Some c -> c
  | None ->
      invalid_arg
        (Printf.sprintf "Nx_cuda_device.%s: %s is not a CUDA device" fn
           (Nx_device.name d))

let handles d = (cuda "handles" d).handles

(* The whole allocation under [a], viewed from [a]. *)
let of_address d a s n =
  let c = cuda "of_address" d in
  let fail fmt =
    Printf.ksprintf
      (fun m -> invalid_arg ("Nx_cuda_device.of_address: " ^ m))
      fmt
  in
  match pointer c.handles.context a with
  | None -> fail "the driver knows no memory at 0x%nx" a
  | Some (owner, _, _) when owner <> c.handles.context ->
      fail "0x%nx is not memory of %s's context" a (Nx_device.name d)
  | Some (_, start, size) ->
      let host = host_pointer c.handles.context start in
      let memory =
        {
          Nx_device.host = (if host = 0n then None else Some host);
          device = start;
          handle = start;
        }
      in
      let whole =
        Nx_device.external_buffer d memory Nx_dtype.Scalar.UInt8 size
      in
      Nx_device.Buffer.view whole
        ~offset:(Nativeint.to_int (Nativeint.sub a start))
        s n
