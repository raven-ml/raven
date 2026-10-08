(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* GPU 0 through the driver, its work submitted through nx.device, each row
   beside the CUDA calls that bound it, made from C on streams of their own: a
   release by a stream write and a spin on the word; a queue switch by an event
   wait, the write and an event record; foreign waits, which nx.device makes
   only across devices, through the driver's C submit, by one batch of memory
   operations; launches, copies, allocations and page-locking by the same calls.
   A row waits by spinning on the word. Each case opens its device in its own
   worker, so that no process forks after CUDA started. Without an NVIDIA GPU
   the suite has no rows. *)

module C = Device_cuda
module S = Device_cuda_support

external names : unit -> string array = "device_cuda_bench_names"
external bind : nativeint array -> unit = "device_cuda_bench_bind"
external start : unit -> unit = "device_cuda_bench_start"
external floor_release : int -> unit = "device_cuda_bench_release"
external floor_switch : unit -> unit = "device_cuda_bench_switch"
external floor_waits : int -> int -> unit = "device_cuda_bench_waits"
external floor_launch : nativeint -> int -> unit = "device_cuda_bench_launch"
external buffer : bool -> int -> nativeint = "device_cuda_bench_buffer"

external floor_copy : nativeint -> nativeint -> int -> unit
  = "device_cuda_bench_copy"

external floor_alloc : int -> unit = "device_cuda_bench_alloc"
external floor_map_host : int -> int -> unit = "device_cuda_bench_map_host"

external entry_waits : nativeint -> int -> int -> int -> unit
  = "device_cuda_bench_entry_waits"

let fixtures = "../test/fixtures"
let kib = 1024
let mib = 1024 * kib

type dev = { c : Device_core.t; g : C.t; mutable v : int }

let strf = Printf.sprintf
let get = function Ok x -> x | Error why -> failwith why
let host r = Option.get (C.host r)
let opens = ref 0

(* GPU 0, opened through nx.device under a name of its own. *)
let dev () =
  incr opens;
  let g = ref None in
  let make () =
    Result.map
      (fun x ->
        g := Some x;
        x)
      (C.open_ 0)
  in
  let c =
    get (Device_core.open_ (module C) ~name:(strf "CUDA:bench-%d" !opens) make)
  in
  let g = Option.get !g in
  S.bind g;
  { c; g; v = 0 }

(* The prepared submission of [parts] on [t]. *)
let prepare t parts =
  Device_core.Submission.make ~reads:0 ~writes:0 ~waits:0 t.c parts

let submit t s = t.v <- Device_core.Point.value (Device_core.submit s)

let wait t =
  while C.signaled t.g < t.v do
    Domain.cpu_relax ()
  done

let run t s =
  submit t s;
  wait t

(* The floor's calls, found by a device's capability. *)
let floor () =
  let g = get (C.open_ 0) in
  let { Device_cuda_abi.symbol } = C.capability g in
  bind (Array.map (fun n -> Option.get (symbol n)) (names ()));
  start ();
  g

let row name setup f = Thumper.bench_with_setup ~setup name f
let empty g = snd (S.kernels ~dir:fixtures g) "empty"

let release_rows =
  let empty () =
    let t = dev () in
    (t, prepare t [||])
  in
  let switching () =
    let t, empty = empty () in
    (t, prepare t [| S.part ~queue:"COPY:0" (S.failing 0) |], empty)
  in
  Thumper.group "release"
    [
      row "driver" empty (fun (t, s) -> run t s);
      row "floor" floor (fun _ -> floor_release 1);
      row "switch" switching (fun (t, copy, empty) ->
          run t (if t.v land 1 = 0 then copy else empty));
      row "floor-switch" floor (fun _ -> floor_switch ());
      row "no-wait-100" empty (fun (t, s) ->
          for _ = 1 to 100 do
            submit t s
          done;
          wait t);
      row "floor-no-wait-100" floor (fun _ -> floor_release 100);
    ]

let wait_rows =
  let word g =
    let w = Option.get (C.alloc g `Pinned 8) in
    S.set64 (host w) 1;
    w
  in
  let waiting () =
    let t = dev () in
    (t, C.self t.g, Option.get (C.address (word t.g)))
  in
  let floor_waiting () = host (word (floor ())) in
  Thumper.group "waits"
    [
      row "4" waiting (fun (t, self, at) ->
          t.v <- t.v + 1;
          entry_waits self t.v at 4;
          wait t);
      row "floor-4" floor_waiting (fun at -> floor_waits at 4);
    ]

let launch_rows =
  let launching count () =
    let t = dev () in
    let f = S.launch ~count (empty t.g) ~grid:1 ~block:1 0 0 in
    (t, prepare t [| S.part ~queue:"COMPUTE:0" f |])
  in
  let floor_launching () = Nativeint.of_int (empty (floor ())) in
  Thumper.group "launch"
    [
      row "1" (launching 1) (fun (t, s) -> run t s);
      row "64" (launching 64) (fun (t, s) -> run t s);
      row "floor-1" floor_launching (fun f -> floor_launch f 1);
      row "floor-64" floor_launching (fun f -> floor_launch f 64);
    ]

let copy_rows =
  let n = 256 * mib in
  let copying (dst, src) () =
    let t = dev () in
    let create m = Device_core.Buffer.create ~memory:m t.c n in
    let dst = create dst and src = create src in
    (t, prepare t [| S.copy ~queue:"COPY:0" ~dst src |])
  in
  let floor_copying (dst, src) () =
    ignore (floor ());
    (buffer dst n, buffer src n)
  in
  let copy name kinds host =
    [
      row name (copying kinds) (fun (t, s) -> run t s);
      row ("floor-" ^ name) (floor_copying host) (fun (dst, src) ->
          floor_copy dst src n);
    ]
  in
  Thumper.group "copy"
    (copy "h2d-256MiB" (Device, Pinned) (false, true)
    @ copy "d2h-256MiB" (Pinned, Device) (true, false)
    @ copy "d2d-256MiB" (Device, Device) (false, false))

let alloc_rows =
  let alloc name n =
    [
      row name dev (fun t -> C.free t.g (Option.get (C.alloc t.g `Device n)));
      row ("floor-" ^ name) floor (fun _ -> floor_alloc n);
    ]
  in
  Thumper.group "alloc" (alloc "64KiB" (64 * kib) @ alloc "64MiB" (64 * mib))

let map_host_rows =
  let n = 256 * mib in
  Thumper.group "map-host"
    [
      row "256MiB"
        (fun () -> (dev (), S.pages n))
        (fun (t, p) -> C.free t.g (Option.get (C.map_host t.g p n)));
      row "floor-256MiB"
        (fun () -> (floor (), S.pages n))
        (fun (_, p) -> floor_map_host p n);
    ]

let () =
  if Sys.file_exists "/dev/nvidiactl" then
    exit
    @@ Thumper.run "device_cuda"
         [
           release_rows;
           wait_rows;
           launch_rows;
           copy_rows;
           alloc_rows;
           map_host_rows;
         ]
