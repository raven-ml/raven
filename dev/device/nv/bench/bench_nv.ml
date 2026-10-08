(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* GPU 0 through the driver, opened through NVIDIA's kernel driver, each row
   beside its floor: the same submissions through device_nv_room and
   device_nv_submit, called from C in a loop on the device the row opened. A row
   waits by spinning on the timeline word. Each case opens its device in its own
   worker. Without an NVIDIA GPU the suite has no rows. *)

module N = Device_nv
module Abi = Device_nv_abi

external start : nativeint -> int -> int -> unit = "device_nv_bench_start"
external floor_release : int -> unit = "device_nv_bench_release"
external floor_switch : unit -> unit = "device_nv_bench_switch"
external floor_waits : int -> int -> unit = "device_nv_bench_waits"
external floor_entry : int -> int -> unit = "device_nv_bench_entry"
external floor_copy : int -> int -> int -> unit = "device_nv_bench_copy"
external pages : int -> int = "device_nv_bench_pages"
external write : int -> string -> unit = "device_nv_bench_write"
external set64 : int -> int -> unit = "device_nv_bench_set64"

let kib = 1024
let mib = 1024 * kib
let get = function Ok x -> x | Error why -> failwith why
let host r = Option.get (N.host r)
let address r = Option.get (N.address r)
let at r off = host r + off
let row name setup f = Thumper.bench_with_setup ~setup name f

(* The driver *)

type dev = { g : N.t; mutable v : int }

let dev () = { g = get (Device_nv_nvidia.open_ 0); v = 0 }
let alloc t kind n = Option.get (N.alloc t.g kind n)

let submit ?(waits = [||]) t ps =
  t.v <- t.v + 1;
  (match N.room t.g ps with
  | `Fits -> ()
  | `Later | `Never -> failwith "the parts do not fit");
  match N.submit t.g ~v:t.v ~waits ~handles:[||] ps with
  | `Ok -> ()
  | `Failed why -> failwith why

let wait t =
  while N.signaled t.g < t.v do
    Domain.cpu_relax ()
  done

let run ?waits t ps =
  submit ?waits t ps;
  wait t

(* The floor of [t]: its later values are given from C. *)
let floor t =
  start (N.self t.g) (host (N.word t.g)) t.v;
  t

(* A device with a live allocation of [kind], which keeps its page tables, as in
   a program's steady state. *)
let live kind () =
  let t = dev () in
  ignore (alloc t kind 4096);
  t

let copy_part t dst src n = N.part t.g ~queue:"COPY:0" (`Copy (dst, src, n))

(* Launches *)

let cubin =
  lazy
    (In_channel.with_open_bin "../test/fixtures/kernels_sm89.cubin"
       In_channel.input_all)

let words s =
  Array.init
    (String.length s / 4)
    (fun i -> Int32.to_int (String.get_int32_le s (4 * i)) land 0xffff_ffff)

let encode = Abi.Structure.encode Int64.of_int

(* The cubin loaded on [t]'s GPU, its image copied by a copy from [staging] into
   a new code region; the image and the region. *)
let upload t staging =
  match N.image t.g (Lazy.force cubin) with
  | Ok (`Place (n, lay)) ->
      let code = alloc t `Device n in
      let m, bytes = lay code in
      write (host staging) bytes;
      run t [| copy_part t (code, 0) (staging, 0) n |];
      (m, code)
  | Ok (`Loaded _) -> failwith "an image without code"
  | Error why -> failwith why

(* [count] launches of the kernel [empty] over one thread: descriptors chained
   in [`Mapped] memory, constant bank 0 there too, and one segment that
   schedules the first, as compiled code places them. The result is the
   segment's ring entry, as two words. *)
let launches t count =
  let c = get (Abi.Cubin.of_string (Lazy.force cubin)) in
  let k = Option.get (Abi.Cubin.kernel c "empty") in
  let image, _ = upload t (alloc t `Pinned (Abi.Cubin.size c)) in
  let entry = Option.get (N.entry image "empty") in
  let cap = N.capability t.g in
  let l = get (Abi.Launch.make cap k) in
  get (cap.local (Abi.Launch.local_bytes l));
  let local = Abi.Local_memory.make cap (Abi.Launch.local_bytes l) in
  let bank0 = alloc t `Mapped 4096 in
  let base = entry - k.code in
  let bank (q : int Abi.Qmd.t) (b : Abi.Cubin.bank) =
    let at = if b.index = 0 then address bank0 else base + b.offset in
    Abi.Qmd.set_bank b.index at q
  in
  let one = [ Abi.Qmd.Grid X; Grid Y; Grid Z; Block X; Block Y; Block Z ] in
  let q =
    List.fold_left (fun q d -> Abi.Qmd.set_dim d 1 q) (Abi.Qmd.make l) one
    |> Abi.Qmd.set_program entry
    |> Abi.Qmd.set_local_memory local.per_thread
  in
  let q = List.fold_left bank q (Abi.Launch.banks l) in
  write (host bank0) (encode (Abi.Qmd.parameters q));
  let stride = 512 in
  let qmds = alloc t `Mapped (count * stride) in
  for i = 0 to count - 1 do
    let next = address qmds + ((i + 1) * stride) in
    let q = if i < count - 1 then Abi.Qmd.chain next q else q in
    write (at qmds (i * stride)) (encode (Abi.Qmd.structure q))
  done;
  let segment = alloc t `Mapped 4096 in
  let ws =
    Abi.Packet.encode Int64.of_int (Abi.Method.schedule (address qmds))
  in
  write (host segment) ws;
  let e =
    Abi.Gpfifo.entry (address segment) ~offset:0 ~words:(String.length ws / 4)
  in
  words (Abi.Packet.encode Int64.of_int e)

(* Rows *)

let release_rows =
  let switching () =
    let t = dev () in
    (t, [| N.part t.g ~queue:"COPY:0" (`Words [||]) |])
  in
  let mapped () =
    let t = dev () in
    ignore (alloc t `Mapped 4096);
    t
  in
  Thumper.group "release"
    [
      row "driver" dev (fun t -> run t [||]);
      row "floor" (fun () -> floor (dev ())) (fun _ -> floor_release 1);
      row "mapped" mapped (fun t -> run t [||]);
      row "floor-mapped"
        (fun () -> floor (mapped ()))
        (fun _ -> floor_release 1);
      row "switch" switching (fun (t, copy) ->
          run t (if t.v land 1 = 0 then copy else [||]));
      row "floor-switch" (fun () -> floor (dev ())) (fun _ -> floor_switch ());
      row "no-wait-100" dev (fun t ->
          for _ = 1 to 100 do
            submit t [||]
          done;
          wait t);
      row "floor-no-wait-100"
        (fun () -> floor (dev ()))
        (fun _ -> floor_release 100);
    ]

let wait_rows =
  let waiting () =
    let t = dev () in
    let w = alloc t `Pinned 8 in
    set64 (host w) 1;
    (t, w)
  in
  Thumper.group "waits"
    [
      row "4"
        (fun () ->
          let t, w = waiting () in
          (t, Array.make 4 (`Word, address w, 1)))
        (fun (t, waits) -> run ~waits t [||]);
      row "floor-4"
        (fun () ->
          let t, w = waiting () in
          ignore (floor t);
          address w)
        (fun at -> floor_waits at 4);
    ]

(* A run of 64 launches follows the GPU's SM clock, which moves by up to a tenth
   between and within runs and which no unprivileged process pins. *)
let launch_rows =
  let launching count () =
    let t = dev () in
    let e = launches t count in
    (t, e, [| N.part t.g ~queue:"COMPUTE:0" (`Words e) |])
  in
  let floor_launching count () =
    let t, e, _ = launching count () in
    ignore (floor t);
    e
  in
  Thumper.group "launch"
    [
      row "1" (launching 1) (fun (t, _, ps) -> run t ps);
      row "floor-1" (floor_launching 1) (fun e -> floor_entry e.(0) e.(1));
      row "64" (launching 64) (fun (t, _, ps) -> run t ps);
      row "floor-64" (floor_launching 64) (fun e -> floor_entry e.(0) e.(1));
    ]

let copy_rows =
  let copying n (dst, src) () =
    let t = dev () in
    let dst = alloc t dst n and src = alloc t src n in
    (t, dst, src, [| copy_part t (dst, 0) (src, 0) n |])
  in
  let copy name n kinds =
    [
      row name (copying n kinds) (fun (t, _, _, ps) -> run t ps);
      row ("floor-" ^ name)
        (fun () ->
          let t, dst, src, _ = copying n kinds () in
          ignore (floor t);
          (address dst, address src))
        (fun (dst, src) -> floor_copy dst src n);
    ]
  in
  let big = 256 * mib in
  Thumper.group "copy"
    (copy "h2d-256MiB" big (`Device, `Pinned)
    @ copy "d2h-256MiB" big (`Pinned, `Device))

let alloc_rows =
  let alloc name kind n =
    row name (live kind) (fun t -> N.free t.g (alloc t kind n))
  in
  Thumper.group "alloc"
    [
      alloc "64KiB" `Device (64 * kib);
      alloc "64MiB" `Device (64 * mib);
      alloc "pinned-64KiB" `Pinned (64 * kib);
      alloc "pinned-64MiB" `Pinned (64 * mib);
    ]

let map_host_rows =
  let n = 256 * mib in
  Thumper.group "map-host"
    [
      row "256MiB"
        (fun () -> (dev (), pages n))
        (fun (t, p) -> N.free t.g (Option.get (N.map_host t.g p n)));
    ]

(* A cubin loaded, its code region allocated and its image copied there, then
   unloaded and the region freed. *)
let image_rows =
  Thumper.group "image"
    [
      row "kernels"
        (fun () ->
          let t = live `Device () in
          let c = get (Abi.Cubin.of_string (Lazy.force cubin)) in
          (t, alloc t `Pinned (Abi.Cubin.size c)))
        (fun (t, staging) ->
          let m, code = upload t staging in
          N.unload t.g m;
          N.free t.g code);
    ]

let () =
  if Device_nv_nvidia.count () > 0 then
    exit
    @@ Thumper.run "device_nv"
         [
           release_rows;
           wait_rows;
           launch_rows;
           copy_rows;
           alloc_rows;
           map_host_rows;
           image_rows;
         ]
