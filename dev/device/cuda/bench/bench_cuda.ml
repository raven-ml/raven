(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* GPU 0 through the driver, each row beside the CUDA calls that bound it, made
   from C on streams of their own: a release by a stream write and a spin on the
   word; a queue switch by an event wait, the write and an event record; foreign
   waits by one batch of memory operations; launches, copies, allocations and
   page-locking by the same calls. A row waits by spinning on the word. Each
   case opens its device in its own worker, so that no process forks after CUDA
   started. Without an NVIDIA GPU the suite has no rows. *)

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

let fixtures = "../test/fixtures"
let kib = 1024
let mib = 1024 * kib

type dev = { g : C.t; mutable v : int }

let get = function Ok x -> x | Error why -> failwith why
let host r = Option.get (C.host r)

let dev () =
  let g = get (C.open_ 0) in
  S.bind g;
  { g; v = 0 }

let submit ?(waits = [||]) t ps =
  t.v <- t.v + 1;
  match C.submit t.g ~v:t.v ~waits ~handles:[||] ps with
  | `Ok -> ()
  | `Failed why -> failwith why

let wait t =
  while C.signaled t.g < t.v do
    Domain.cpu_relax ()
  done

let run ?waits t ps =
  submit ?waits t ps;
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
  let switching () =
    let t = dev () in
    let nothing = S.failing 0 in
    (t, nothing, [| S.part t.g ~queue:"COPY:0" nothing |])
  in
  Thumper.group "release"
    [
      row "driver" dev (fun t -> run t [||]);
      row "floor" floor (fun _ -> floor_release 1);
      row "switch" switching (fun (t, _, copy) ->
          run t (if t.v land 1 = 0 then copy else [||]));
      row "floor-switch" floor (fun _ -> floor_switch ());
      row "no-wait-100" dev (fun t ->
          for _ = 1 to 100 do
            submit t [||]
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
    let at = Option.get (C.address (word t.g)) in
    (t, Array.make 4 (`Word, at, 1))
  in
  let floor_waiting () = host (word (floor ())) in
  Thumper.group "waits"
    [
      row "4" waiting (fun (t, waits) -> run ~waits t [||]);
      row "floor-4" floor_waiting (fun at -> floor_waits at 4);
    ]

let launch_rows =
  let launching count () =
    let t = dev () in
    let f = S.launch ~count (empty t.g) ~grid:1 ~block:1 0 0 in
    (t, f, [| S.part t.g ~queue:"COMPUTE:0" f |])
  in
  let floor_launching () = Nativeint.of_int (empty (floor ())) in
  Thumper.group "launch"
    [
      row "1" (launching 1) (fun (t, _, ps) -> run t ps);
      row "64" (launching 64) (fun (t, _, ps) -> run t ps);
      row "floor-1" floor_launching (fun f -> floor_launch f 1);
      row "floor-64" floor_launching (fun f -> floor_launch f 64);
    ]

let copy_rows =
  let n = 256 * mib in
  let copying (dst, src) () =
    let t = dev () in
    let alloc k = Option.get (C.alloc t.g k n) in
    let dst = alloc dst and src = alloc src in
    (t, [| C.part t.g ~queue:"COPY:0" (`Copy ((dst, 0), (src, 0), n)) |])
  in
  let floor_copying (dst, src) () =
    ignore (floor ());
    (buffer dst n, buffer src n)
  in
  let copy name kinds host =
    [
      row name (copying kinds) (fun (t, ps) -> run t ps);
      row ("floor-" ^ name) (floor_copying host) (fun (dst, src) ->
          floor_copy dst src n);
    ]
  in
  Thumper.group "copy"
    (copy "h2d-256MiB" (`Device, `Pinned) (false, true)
    @ copy "d2h-256MiB" (`Pinned, `Device) (true, false)
    @ copy "d2d-256MiB" (`Device, `Device) (false, false))

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
