(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The Mac's GPU through the driver, its work submitted through rig, each row
   beside the raw Metal calls that bound it, made on a queue of their own: a
   release by a commit and a wait; launches by the same dispatches encoded
   directly; a launch from an indirect command buffer by the same indirect
   command buffer; memory and images by the Metal objects they make. A row waits
   as a caller does, through rig's wait; the rows of [sleep] wait while spinning
   threads hold every core. A floor waits as Metal's own API does, blocking in
   waitUntilCompleted. Each case opens its device in its own worker, so that no
   process forks after Metal started. *)

module M = Rig_metal
module S = Rig_metal_support
module H = Rig_gpu_support.Host

(* The function an entry answered, which the bench needs. *)
let found : Rig_edge.entry_answer -> Rig_edge.entry = function
  | Entry e -> e
  | Absent -> failwith "no such function"
  | No_memory n -> failwith (Printf.sprintf "no memory for %d bytes" n)

external floor : string -> nativeint = "rig_metal_bench_floor"
external floor_release : nativeint -> int -> unit = "rig_metal_bench_release"
external floor_launch : nativeint -> int -> unit = "rig_metal_bench_launch"
external floor_icb : nativeint -> int -> nativeint = "rig_metal_bench_icb"

external floor_execute : nativeint -> nativeint -> int -> unit
  = "rig_metal_bench_execute"

external floor_buffers : nativeint -> int -> unit = "rig_metal_bench_buffers"

external floor_pipelined : nativeint -> int -> int -> unit
  = "rig_metal_bench_pipelined"
external floor_alloc : nativeint -> int -> unit = "rig_metal_bench_alloc"

external floor_map_host : nativeint -> int -> int -> unit
  = "rig_metal_bench_map_host"

external floor_image : nativeint -> string -> unit = "rig_metal_bench_image"
external default_class : unit -> unit = "rig_metal_bench_default_class"
external load_start : int -> unit = "rig_metal_bench_load_start"
external load_stop : unit -> unit = "rig_metal_bench_load_stop"

let metallib = S.fixture ~dir:"../../test/metal/fixtures" "fill"

(* The conformance suite's kernels, [wide] among them. *)
let launch_lib = S.fixture ~dir:"../../test/metal/fixtures" "launch"
let kib = 1024
let mib = 1024 * kib

(* A device and [step] as the driver names it and as rig loaded it, which a
   launch writes into the one buffer of [buffers]. *)
type dev = {
  d : Rig.t;
  g : M.t;
  mutable v : int;
  mutable last : Rig.Point.t;
  step : int;
  args : M.region;
  image : Rig.Image.t;
  buffers : Rig.Buffer.t array;
}

let get = function Ok x -> x | Error why -> failwith why
let alloc t n = Option.get (M.alloc t.g Rig_edge.Device n)
let host r = Option.get (M.locate r).host
let address r = Option.get (M.locate r).address
let handle r = (M.locate r).handle

let load d =
  match get (M.image d metallib) with
  | Rig_edge.Loaded i -> i
  | Place _ -> failwith "Metal asked to place its code"

(* The point of an empty submission on [d]: the last point a row waits for
   until it submits. *)
let first_point d =
  Rig.submit (Rig.Submission.make d [||]) ~run:(Rig.Submission.Run.make ())
    ~buffers:[||]

(* A device opened through rig, whose argument buffer points [step] at a word of
   its own. *)
let dev () =
  let { S.d; g } = S.open_ () in
  let step = (found (M.entry (load g) "step")).code in
  let args = Option.get (M.alloc g Rig_edge.Device 16) in
  let image = get (Rig.Image.load d metallib) in
  let out = Rig.Buffer.create d 16 in
  let t =
    {
      d;
      g;
      v = 0;
      last = first_point d;
      step;
      args;
      image;
      buffers = [| out |];
    }
  in
  H.set64 (host args) (address (alloc t 16));
  t

(* The prepared submission of [parts] on [t]. *)
let prepare t parts =
  (Rig.Submission.make t.d parts, Rig.Submission.Run.make (), [||])

let prepare_fill t f = Rig_gpu_support.prepare t.d [ S.part f ]

let submit t (s, run, buffers) =
  let p = Rig.submit s ~run ~buffers in
  t.last <- p;
  t.v <- Rig.Point.value p

(* [step] launched [n] times in one submission, each over one thread and writing
   [out], with its run's blocks initialized. *)
let launches t n =
  let module Sub = Rig.Submission in
  let fn = Option.get (Rig.Image.resolve t.image "step") in
  let step = Sub.launch ~refs:[| Sub.write 0 ~at:0 |] fn ~params:16 in
  let s = Sub.make t.d (Array.make n step) in
  let run = Sub.Run.make () in
  for i = 0 to n - 1 do
    let b = Sub.block s i in
    Sub.Run.groups run b 1 1 1;
    Sub.Run.threads run b 1 1 1
  done;
  (s, run, t.buffers)

(* [wide] launched once over one thread with [n] bytes of parameters, its
   ref writing [out]. *)
let wide t n =
  let module Sub = Rig.Submission in
  let image = get (Rig.Image.load t.d launch_lib) in
  let fn = Option.get (Rig.Image.resolve image "wide") in
  let w = Sub.launch ~refs:[| Sub.write 0 ~at:0 |] fn ~params:n in
  let s = Sub.make t.d [| w |] in
  let run = Sub.Run.make () in
  let b = Sub.block s 0 in
  Sub.Run.groups run b 1 1 1;
  Sub.Run.threads run b 1 1 1;
  (s, run, t.buffers)

let wait t = Rig.Point.wait t.last

let run t s =
  submit t s;
  wait t

(* A part running an indirect command buffer of [n] dispatches of [step]. *)
let indirect t n =
  let dispatch =
    {
      Rig_metal_abi.pipeline = t.step;
      offset = 0;
      groups = (1, 1, 1);
      threads = (1, 1, 1);
    }
  in
  let b =
    get ((M.capability t.g).icb (handle t.args) (Array.make n dispatch))
  in
  prepare_fill t (S.execute b)

(* A part dispatching [step] once, directly. *)
let step t =
  let f = S.dispatch ~pipeline:t.step t.args ~groups:1 ~threads:1 in
  prepare_fill t f

let stepping () =
  let t = dev () in
  (t, step t)

let row name setup f = Thumper.bench_with_setup ~setup name f
let floor () = floor metallib

let release_rows =
  let empty () =
    let t = dev () in
    (t, prepare t [||])
  in
  Thumper.group "release"
    [
      row "driver" empty (fun (t, s) -> run t s);
      row "floor" floor (fun f -> floor_release f 1);
      row "pipelined-256" empty (fun (t, s) ->
          for _ = 1 to 256 do
            submit t s
          done;
          wait t);
      row "floor-pipelined-256" floor (fun f -> floor_release f 256);
      row "after-idle-20ms" empty (fun (t, s) ->
          M.sleep t.g ~seen:t.v ~still_ms:20;
          run t s);
    ]

let launch_rows =
  (* A submission of [n] launches of [step]: [floor-n] runs the same dispatches
     in one encoder. *)
  let launched n () =
    let t = dev () in
    (t, launches t n)
  in
  let run_launches (t, s) =
    submit t s;
    wait t
  in
  (* [n] submissions of one launch of [step] each, then a wait for the last:
     [64]'s dispatches, one value each. *)
  let submits n (t, s) =
    for _ = 1 to n do
      submit t s
    done;
    wait t
  in
  (* [runs] submissions of one launch of [step] each, each after a wait for the
     one [depth] back, as a program keeps [depth] steps in flight. *)
  let depth = 3 and runs = 100 in
  let pipelined (t, s) =
    let points = Array.make depth t.last in
    for k = 0 to runs - 1 do
      if k >= depth then Rig.Point.wait points.(k mod depth);
      submit t s;
      points.(k mod depth) <- t.last
    done;
    wait t
  in
  let live () =
    let t, s = launched 1 () in
    let regions = List.init 4096 (fun _ -> alloc t (64 * kib)) in
    (t, s, regions)
  in
  let recorded n () =
    let t = dev () in
    (t, indirect t n)
  in
  let floor_recorded n () =
    let f = floor () in
    (f, floor_icb f n)
  in
  Thumper.group "launch"
    [
      row "1" (launched 1) run_launches;
      (* Past the 4 KB setBytes takes: the parameters go in the arena. *)
      row "params-8192"
        (fun () ->
          let t = dev () in
          (t, wide t 8192))
        run_launches;
      row "64" (launched 64) run_launches;
      row "submits-64" (launched 1) (submits 64);
      row "submits-1024" (launched 1) (submits 1024);
      row "pipelined-100" (launched 1) pipelined;
      row "floor-1" floor (fun f -> floor_launch f 1);
      row "floor-64" floor (fun f -> floor_launch f 64);
      row "floor-pipelined-100" floor (fun f -> floor_pipelined f depth runs);
      row "1-live-4096" live (fun (t, s, _) -> run_launches (t, s));
      row "icb-1" (recorded 1) (fun (t, s) -> run t s);
      row "icb-64" (recorded 64) (fun (t, s) -> run t s);
      row "floor-icb-1" (floor_recorded 1) (fun (f, b) -> floor_execute f b 1);
      row "floor-icb-64" (floor_recorded 64) (fun (f, b) ->
          floor_execute f b 64);
    ]

let split_rows =
  let splitting () =
    let t = dev () in
    let f = S.dispatch ~pipeline:t.step t.args ~groups:1 ~threads:1 in
    S.split f t.g 64 ~times:(host (alloc t (16 * 64)));
    (t, prepare_fill t f)
  in
  Thumper.group "split"
    [
      row "64" splitting (fun (t, s) -> run t s);
      row "floor-64" floor (fun f -> floor_buffers f 65);
    ]

let alloc_rows =
  let n_row name n = row name dev (fun t -> M.free t.g (alloc t n))
  and floor_row name n = row name floor (fun f -> floor_alloc f n) in
  Thumper.group "alloc"
    [
      n_row "64KiB" (64 * kib);
      n_row "64MiB" (64 * mib);
      floor_row "floor-64KiB" (64 * kib);
      floor_row "floor-64MiB" (64 * mib);
      row "first-use-64MiB" stepping (fun (t, s) ->
          let r = alloc t (64 * mib) in
          H.set64 (host t.args) (address r);
          run t s;
          M.free t.g r);
    ]

let map_host_rows =
  let n = 64 * mib in
  (* Pages written once, so that no map faults them in. *)
  let written () =
    let p = H.pages n in
    for i = 0 to (n / H.page) - 1 do
      H.set8 (p + (i * H.page)) 0
    done;
    p
  in
  let pages () = (dev (), written ())
  and floor_pages () = (floor (), written ()) in
  Thumper.group "map-host"
    [
      row "64MiB" pages (fun (t, p) ->
          M.free t.g (Option.get (M.map_host t.g p n)));
      row "floor-64MiB" floor_pages (fun (f, p) -> floor_map_host f p n);
    ]

(* An image's load makes no pipeline; its first entry makes one, which Metal's
   shader cache holds after the first run on the machine. *)
let image_rows =
  let first t =
    let i = load t.g in
    ignore (M.entry i "fill");
    M.unload t.g i
  in
  Thumper.group "image"
    [
      row "fill" dev (fun t -> M.unload t.g (load t.g));
      row "floor" floor (fun f -> floor_image f "");
      row "first-entry" dev first;
      row "floor-first-entry" floor (fun f -> floor_image f "fill");
    ]

let icb_rows =
  let icb t =
    let dispatch =
      {
        Rig_metal_abi.pipeline = t.step;
        offset = 0;
        groups = (1, 1, 1);
        threads = (1, 1, 1);
      }
    in
    ignore
      (get ((M.capability t.g).icb (handle t.args) (Array.make 64 dispatch))
        : Rig_metal_abi.icb)
  in
  Thumper.group "icb" [ row "64" dev icb ]

(* A launch waited by rig, which blocks in the driver's [sleep], from a thread
   of the default class, while one spinning thread per core competes for the
   processor. On the M1 Max (macOS 26) a process's waits under this load return
   either about 10 us or about 1.4 ms after the release, the same for the whole
   process, whatever the waiting thread's class; raw Metal's wait does the
   same. *)
let sleep_rows =
  let loaded setup () =
    default_class ();
    let x = setup () in
    load_start (Domain.recommended_domain_count ());
    x
  in
  let teardown _ = load_stop () in
  Thumper.group "sleep"
    [
      Thumper.bench_with_setup ~setup:(loaded stepping) ~teardown "under-load"
        (fun (t, s) ->
          submit t s;
          wait t);
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
    H.set64 (host t.args) (address r);
    (t, step t)
  in
  Thumper.group "residency"
    [
      row "warm-1GiB" gib (fun (t, s) -> run t s);
      row "after-idle-3s" gib (fun (t, s) ->
          M.sleep t.g ~seen:t.v ~still_ms:3000;
          run t s);
    ]

(* The idle row takes about 3 s a call. *)
let config = Thumper.Config.(default |> deadline 120.)

(* Without Metal the suite skips, as a GPU suite does without its GPU. It looks
   for the framework: counting GPUs would start Metal in this process, and a
   worker forked after that crashes. *)
let () =
  if not (Sys.file_exists "/System/Library/Frameworks/Metal.framework") then
    print_endline "rig_metal: skipped, no Metal"
  else
    exit
    @@ Thumper.run ~config "rig_metal"
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
