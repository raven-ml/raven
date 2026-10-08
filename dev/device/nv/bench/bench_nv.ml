(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* GPU 0, opened through NVIDIA's kernel driver and Device_core, each row beside
   its floor: the same submissions through device_nv_room and device_nv_submit,
   called from C in a loop on the device the row opened, then a spin on the
   timeline word. A row submits through Device_core.submit and waits with
   Device_core.wait, as a program does, so its distance to the floor is the
   core's share and the OCaml side of the driver. Memory rows call the driver.
   Each case opens its device in its own worker. Without an NVIDIA GPU the suite
   has no rows. *)

module N = Device_nv
module C = Device_core
module B = Device_core.Buffer
module Sub = Device_core.Submission
module Abi = Device_nv_abi
module S = Device_nv_support

external start : nativeint -> int -> int -> unit = "device_nv_bench_start"
external floor_release : int -> unit = "device_nv_bench_release"
external floor_switch : unit -> unit = "device_nv_bench_switch"
external floor_waits : int -> int -> unit = "device_nv_bench_waits"
external floor_entry : int -> int -> unit = "device_nv_bench_entry"
external floor_copy : int -> int -> int -> unit = "device_nv_bench_copy"

let kib = 1024
let mib = 1024 * kib
let get = function Ok x -> x | Error why -> failwith why
let host r = Option.get (N.host r)
let address r = Option.get (N.address r)
let at r off = host r + off
let row name setup f = Thumper.bench_with_setup ~setup name f

(* The device: [d] the core's, [g] its driver's. *)

type dev = { d : C.t; g : N.t }

let dev () =
  let g = ref None in
  let make () =
    let r = Device_nv_nvidia.open_ 0 in
    Result.iter (fun x -> g := Some x) r;
    r
  in
  let d = get (C.open_ (module N) ~name:"NV" make) in
  { d; g = Option.get !g }

let alloc t kind n = Option.get (N.alloc t.g kind n)
let submission t ps = Sub.make ~reads:0 ~writes:0 ~waits:0 t.d ps

let run t s =
  let p = C.submit s in
  C.wait t.d (C.Point.value p)

(* The floor of [t]: its later values are given from C. *)
let floor t =
  start (N.self t.g) (host (N.word t.g)) (C.submitted t.d);
  t

let part queue work = { Sub.queue; after = [||]; work }

(* A host buffer of the ring entry [e], two words. *)
let entry_words e =
  let b = B.create C.host 8 in
  let ba = B.bigarray Bigarray.int32 b in
  Array.iteri (fun i w -> Bigarray.Array1.set ba i (Int32.of_int w)) e;
  b

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

(* [count] launches of the kernel [empty] over one thread: descriptors chained
   in [`Mapped] memory, constant bank 0 there too, and one segment that
   schedules the first, as compiled code places them. The result is the loaded
   program, which stays loaded while reachable, and the segment's ring entry, as
   two words. *)
let launches t count =
  let c = get (Abi.Cubin.of_string (Lazy.force cubin)) in
  let k = Option.get (Abi.Cubin.kernel c "empty") in
  let p = get (C.Program.load t.d (Lazy.force cubin)) in
  let entry = Option.get (C.Program.entry p "empty") in
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
  S.write (host bank0) (encode (Abi.Qmd.parameters q));
  let stride = 512 in
  let qmds = alloc t `Mapped (count * stride) in
  for i = 0 to count - 1 do
    let next = address qmds + ((i + 1) * stride) in
    let q = if i < count - 1 then Abi.Qmd.chain next q else q in
    S.write (at qmds (i * stride)) (encode (Abi.Qmd.structure q))
  done;
  let segment = alloc t `Mapped 4096 in
  let ws =
    Abi.Packet.encode Int64.of_int (Abi.Method.schedule (address qmds))
  in
  S.write (host segment) ws;
  let e =
    Abi.Gpfifo.entry (address segment) ~offset:0 ~words:(String.length ws / 4)
  in
  (p, words (Abi.Packet.encode Int64.of_int e))

(* Rows *)

let release_rows =
  let empty () =
    let t = dev () in
    (t, submission t [||])
  in
  let mapped () =
    let t, s = empty () in
    (t, s, B.create ~memory:Mapped t.d 4096)
  in
  let switching () =
    let t, none = empty () in
    let copy = submission t [| part "COPY:0" (Words (B.create C.host 0)) |] in
    (t, copy, none)
  in
  let floor_of setup () = floor (fst (setup ())) in
  Thumper.group "release"
    [
      row "driver" empty (fun (t, s) -> run t s);
      row "floor" (floor_of empty) (fun _ -> floor_release 1);
      row "mapped" mapped (fun (t, s, _) -> run t s);
      row "floor-mapped"
        (fun () ->
          let t, _, m = mapped () in
          (floor t, m))
        (fun _ -> floor_release 1);
      row "switch" switching (fun (t, copy, none) ->
          run t (if C.submitted t.d land 1 = 0 then copy else none));
      row "floor-switch" (floor_of empty) (fun _ -> floor_switch ());
      row "no-wait-100" empty (fun (t, s) ->
          for _ = 1 to 99 do
            ignore (C.submit s)
          done;
          run t s);
      row "floor-no-wait-100" (floor_of empty) (fun _ -> floor_release 100);
    ]

(* The core passes a driver only the waits not yet reached, so four satisfied
   waits are timed from C alone: the waits the driver encodes and the
   release. *)
let wait_rows =
  Thumper.group "waits"
    [
      row "floor-4"
        (fun () ->
          let t = dev () in
          let w = alloc t `Pinned 8 in
          S.set64 (host w) 1;
          ignore (floor t);
          (t, address w))
        (fun (_, at) -> floor_waits at 4);
    ]

(* A run of 64 launches follows the GPU's SM clock, which moves by up to a tenth
   between and within runs and which no unprivileged process pins. *)
let launch_rows =
  let launching count () =
    let t = dev () in
    let p, e = launches t count in
    (t, p, e, submission t [| part "COMPUTE:0" (Words (entry_words e)) |])
  in
  let floor_launching count () =
    let t, p, e, _ = launching count () in
    (floor t, p, e)
  in
  let floor_run (_, _, e) = floor_entry e.(0) e.(1) in
  Thumper.group "launch"
    [
      row "1" (launching 1) (fun (t, _, _, s) -> run t s);
      row "floor-1" (floor_launching 1) floor_run;
      row "64" (launching 64) (fun (t, _, _, s) -> run t s);
      row "floor-64" (floor_launching 64) floor_run;
    ]

let copy_rows =
  let copying n (dst, src) () =
    let t = dev () in
    let dst = B.create ~memory:dst t.d n and src = B.create ~memory:src t.d n in
    (t, dst, src, submission t [| part "COPY:0" (Copy { src; dst }) |])
  in
  let copy name n kinds =
    [
      row name (copying n kinds) (fun (t, _, _, s) -> run t s);
      row ("floor-" ^ name)
        (fun () ->
          let t, dst, src, _ = copying n kinds () in
          (floor t, dst, src))
        (fun (_, dst, src) -> floor_copy (B.address dst) (B.address src) n);
    ]
  in
  let big = 256 * mib in
  Thumper.group "copy"
    (copy "h2d-256MiB" big (B.Device, B.Pinned)
    @ copy "d2h-256MiB" big (B.Pinned, B.Device))

(* A device with a live allocation of [kind], which keeps its page tables, as in
   a program's steady state. *)
let live kind () =
  let t = dev () in
  ignore (alloc t kind 4096);
  t

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
        (fun () -> (dev (), S.pages n))
        (fun (t, p) -> N.free t.g (Option.get (N.map_host t.g p n)));
    ]

(* A cubin loaded by the driver over a new code region, then unloaded and the
   region freed. *)
let image_rows =
  Thumper.group "image"
    [
      row "kernels" (live `Device) (fun t ->
          match N.image t.g (Lazy.force cubin) with
          | Ok (`Place (n, lay)) ->
              let code = alloc t `Device n in
              let m, _ = lay code in
              N.unload t.g m;
              N.free t.g code
          | Ok (`Loaded _) -> failwith "an image without code"
          | Error why -> failwith why);
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
