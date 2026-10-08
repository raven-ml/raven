(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The Mac's GPU through the driver, each row beside the raw Metal calls that
   bound it, made on a queue of their own: a release by a commit and a wait; a
   launch from an indirect command buffer by the same indirect command buffer
   and by the same dispatches encoded directly; memory and images by the Metal
   objects they make. A row waits by spinning on the word, except the rows of
   [sleep], which block in the driver's [sleep] while spinning threads hold
   every core. Each case opens its device in its own worker, so that no process
   forks after Metal started. *)

module M = Device_metal
module S = Device_metal_support

external floor : string -> nativeint = "device_metal_bench_floor"
external floor_release : nativeint -> int -> unit = "device_metal_bench_release"
external floor_launch : nativeint -> int -> unit = "device_metal_bench_launch"
external floor_icb : nativeint -> int -> nativeint = "device_metal_bench_icb"

external floor_execute : nativeint -> nativeint -> int -> unit
  = "device_metal_bench_execute"

external floor_buffers : nativeint -> int -> unit = "device_metal_bench_buffers"
external floor_alloc : nativeint -> int -> unit = "device_metal_bench_alloc"

external floor_map_host : nativeint -> int -> int -> unit
  = "device_metal_bench_map_host"

external floor_image : nativeint -> unit = "device_metal_bench_image"
external default_class : unit -> unit = "device_metal_bench_default_class"
external load_start : int -> unit = "device_metal_bench_load_start"
external load_stop : unit -> unit = "device_metal_bench_load_stop"

let metallib = S.fixture ~dir:"../test/fixtures" "fill"
let kib = 1024
let mib = 1024 * kib

type dev = { d : M.t; mutable v : int; step : int; args : M.region }

let get = function Ok x -> x | Error why -> failwith why
let alloc t n = Option.get (M.alloc t.d `Device n)
let host r = Option.get (M.host r)

let load d =
  match get (M.image d metallib) with
  | `Loaded i -> i
  | `Place _ -> failwith "Metal asked to place its code"

(* A device whose argument buffer points [step] at a word of its own. *)
let dev () =
  let d = get (M.open_ 0) in
  let image = load d in
  let step = Option.get (M.entry image "step") in
  let args = Option.get (M.alloc d `Device 16) in
  let t = { d; v = 0; step; args } in
  S.set64 (host args) 0 (Int64.of_int (Option.get (M.address (alloc t 16))));
  t

let submit t ps =
  t.v <- t.v + 1;
  match M.submit t.d ~v:t.v ~waits:[||] ~handles:[||] ps with
  | `Ok -> ()
  | `Failed why -> failwith why

let wait t =
  while M.signaled t.d < t.v do
    Domain.cpu_relax ()
  done

let run t ps =
  submit t ps;
  wait t

(* A part running an indirect command buffer of [n] dispatches of [step]. *)
let launch t n =
  let dispatch =
    {
      Device_metal_abi.pipeline = t.step;
      offset = 0;
      groups = (1, 1, 1);
      threads = (1, 1, 1);
    }
  in
  let b =
    get ((M.capability t.d).icb (M.handle t.args) (Array.make n dispatch))
  in
  let f = S.execute b ~pipelines:[| t.step |] in
  (f, [| S.part t.d f |])

(* A part dispatching [step] once, directly. *)
let step t =
  let f = S.dispatch ~pipeline:t.step t.args ~groups:1 ~threads:1 in
  (f, [| S.part t.d f |])

let stepping () =
  let t = dev () in
  (t, step t)

let row name setup f = Thumper.bench_with_setup ~setup name f
let floor () = floor metallib

let release_rows =
  Thumper.group "release"
    [
      row "driver" dev (fun t -> run t [||]);
      row "floor" floor (fun f -> floor_release f 1);
      row "pipelined-256" dev (fun t ->
          for _ = 1 to 256 do
            submit t [||]
          done;
          wait t);
      row "floor-pipelined-256" floor (fun f -> floor_release f 256);
      row "after-idle-20ms" dev (fun t ->
          M.sleep t.d ~seen:t.v ~still_ms:20;
          run t [||]);
    ]

let launch_rows =
  let launched n () =
    let t = dev () in
    (t, launch t n)
  in
  let live () =
    let t, l = launched 1 () in
    let regions = List.init 4096 (fun _ -> alloc t (64 * kib)) in
    (t, l, regions)
  in
  let indirect n () =
    let f = floor () in
    (f, floor_icb f n)
  in
  Thumper.group "launch"
    [
      row "1" (launched 1) (fun (t, (_, ps)) -> run t ps);
      row "64" (launched 64) (fun (t, (_, ps)) -> run t ps);
      row "floor-icb-1" (indirect 1) (fun (f, b) -> floor_execute f b 1);
      row "floor-icb-64" (indirect 64) (fun (f, b) -> floor_execute f b 64);
      row "floor-1" floor (fun f -> floor_launch f 1);
      row "floor-64" floor (fun f -> floor_launch f 64);
      row "1-live-4096" live (fun (t, (_, ps), _) -> run t ps);
    ]

let split_rows =
  let splitting () =
    let t = dev () in
    let f = S.dispatch ~pipeline:t.step t.args ~groups:1 ~threads:1 in
    S.split f t.d 64 ~times:(host (alloc t (16 * 64)));
    (t, f, [| S.part t.d f |])
  in
  Thumper.group "split"
    [
      row "64" splitting (fun (t, _, ps) -> run t ps);
      row "floor-64" floor (fun f -> floor_buffers f 65);
    ]

let alloc_rows =
  let n_row name n = row name dev (fun t -> M.free t.d (alloc t n))
  and floor_row name n = row name floor (fun f -> floor_alloc f n) in
  Thumper.group "alloc"
    [
      n_row "64KiB" (64 * kib);
      n_row "64MiB" (64 * mib);
      floor_row "floor-64KiB" (64 * kib);
      floor_row "floor-64MiB" (64 * mib);
      row "first-use-64MiB" stepping (fun (t, (_, ps)) ->
          let r = alloc t (64 * mib) in
          S.set64 (host t.args) 0 (Int64.of_int (Option.get (M.address r)));
          run t ps;
          M.free t.d r);
    ]

let map_host_rows =
  let n = 64 * mib in
  let pages () = (dev (), S.pages n)
  and floor_pages () = (floor (), S.pages n) in
  Thumper.group "map-host"
    [
      row "64MiB" pages (fun (t, p) ->
          M.free t.d (Option.get (M.map_host t.d p n)));
      row "floor-64MiB" floor_pages (fun (f, p) -> floor_map_host f p n);
    ]

let image_rows =
  Thumper.group "image"
    [
      row "fill" dev (fun t -> M.unload t.d (load t.d));
      row "floor" floor floor_image;
    ]

let icb_rows =
  let icb t =
    let dispatch =
      {
        Device_metal_abi.pipeline = t.step;
        offset = 0;
        groups = (1, 1, 1);
        threads = (1, 1, 1);
      }
    in
    (get ((M.capability t.d).icb (M.handle t.args) (Array.make 64 dispatch)))
      .release ()
  in
  Thumper.group "icb" [ row "64" dev icb ]

(* A launch waited by blocking in [sleep] from a thread of the default class,
   while one spinning thread per core competes for the processor. On the M1 Max
   (macOS 26) a process's waits under this load return either about 10 us or
   about 1.4 ms after the release, the same for the whole process, whatever the
   waiting thread's class; raw Metal's wait does the same. *)
let sleep_rows =
  let loaded setup () =
    default_class ();
    let x = setup () in
    load_start (Domain.recommended_domain_count ());
    x
  in
  let rec sleep t =
    let seen = M.signaled t.d in
    if seen < t.v then begin
      M.sleep t.d ~seen ~still_ms:1000;
      sleep t
    end
  in
  let teardown _ = load_stop () in
  Thumper.group "sleep"
    [
      Thumper.bench_with_setup ~setup:(loaded stepping) ~teardown "under-load"
        (fun (t, (_, ps)) ->
          submit t ps;
          sleep t);
      Thumper.bench_with_setup ~setup:(loaded floor) ~teardown
        "floor-under-load" (fun f -> floor_launch f 1);
    ]

(* A launch whose argument points into 1 GiB of memory, back to back and after 3
   s idle. On the M1 Max (macOS 26) the first submission after an idle of 1.2 to
   3 s or more waits 20 to 80 ms while the memory is made resident again; asking
   for residency at that submission does not shorten it. *)
let residency_rows =
  let gib () =
    let t = dev () in
    let r = alloc t (1024 * mib) in
    S.set64 (host t.args) 0 (Int64.of_int (Option.get (M.address r)));
    (t, step t)
  in
  Thumper.group "residency"
    [
      row "warm-1GiB" gib (fun (t, (_, ps)) -> run t ps);
      row "after-idle-3s" gib (fun (t, (_, ps)) ->
          M.sleep t.d ~seen:t.v ~still_ms:3000;
          run t ps);
    ]

(* The idle row takes about 3 s a call. *)
let config = Thumper.Config.(default |> deadline 120.)

let () =
  exit
  @@ Thumper.run ~config "device_metal"
       [
         release_rows;
         launch_rows;
         split_rows;
         alloc_rows;
         map_host_rows;
         image_rows;
         icb_rows;
         sleep_rows;
         residency_rows;
       ]
