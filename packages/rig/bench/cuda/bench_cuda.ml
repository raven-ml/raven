(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* GPU 0 through the driver, its work submitted through rig, as a program
   submits it. Memory and image rows call the driver. Each case opens its
   device in its own worker, so that no process forks after CUDA started.
   Without an NVIDIA GPU the suite has no rows. *)

module C = Rig_cuda
module S = Rig_cuda_support
module H = Rig_gpu_support.Host

let fixtures = "../../test/cuda/fixtures"
let kib = 1024
let mib = 1024 * kib

(* A device and the run its row submits with. *)
type dev = { d : Rig.t; g : C.t; mutable v : int; mutable last : Rig.Point.t }

let get = function Ok x -> x | Error why -> failwith why

(* The point of an empty submission on [d]: the last point a row waits for
   until it submits. *)
let first_point d =
  Rig.submit (Rig.Submission.Run.make (Rig.Submission.make d [||]))
    ~buffers:[||]

(* GPU 0, opened through rig. *)
let dev () =
  let { S.d; g } = S.open_ () in
  { d; g; v = 0; last = first_point d }

(* A run of the prepared submission of [parts] on [t]. *)
let prepare t parts = Rig.Submission.(Run.make (make t.d parts))

let prepare_fill t f =
  let run, buffers = Rig_gpu_support.prepare t.d [ S.part t.d f ] in
  assert (Array.length buffers = 0);
  run

let submit t run =
  let p = Rig.submit run ~buffers:[||] in
  t.last <- p;
  t.v <- Rig.Point.value p

let wait t = Rig.Point.wait t.last

let run t s =
  submit t s;
  wait t

let row name setup f = Thumper.bench_with_setup ~setup name f
let empty g = snd (S.kernels ~dir:fixtures g) "empty"

(* [n] launches of [empty] over one thread, as parts of COMPUTE:0 loaded by rig,
   their blocks stored in [t]'s run. *)
let launches t n =
  let module Sub = Rig.Submission in
  let bin = S.fixture ~dir:fixtures "kernels.ptx" in
  let image = get (Rig.Image.load t.d bin) in
  let launch =
    Sub.launch (Option.get (Rig.Image.resolve image "empty")) ~params:16
  in
  let run = prepare t (Array.make n launch) in
  for i = 0 to n - 1 do
    let b = Sub.block (Sub.Run.submission run) i in
    Sub.Run.groups run b 1 1 1;
    Sub.Run.threads run b 1 1 1
  done;
  run

let release_rows =
  let empty () =
    let t = dev () in
    (t, prepare t [||])
  in
  let switching () =
    let t, empty = empty () in
    (* A copy of no bytes runs on the copy stream: it follows nothing. *)
    let copy = Rig.Submission.(copy ~src:(Buffer (Rig.Buffer.create t.d 0))
        ~dst:(Buffer (Rig.Buffer.create t.d 0))) in
    (t, prepare t [| copy |], empty)
  in
  Thumper.group "release"
    [
      row "driver" empty (fun (t, s) -> run t s);
      row "switch" switching (fun (t, copy, empty) ->
          run t (if t.v land 1 = 0 then copy else empty));
      row "no-wait-100" empty (fun (t, s) ->
          for _ = 1 to 100 do
            submit t s
          done;
          wait t);
    ]

let kernels f n = Array.make n (S.kernel f 0 0)

(* A graph of [n] empty kernels, made through the capability, launched by a fill
   that updates no node or, [updated], every node's first argument to the run's
   parity: two prepared submissions, one per parity, alternate. *)
let graphing ?(updated = false) n () =
  let t = dev () in
  let f = empty t.g in
  let gr = get ((C.capability t.g).graph (kernels f n)) in
  let updates a =
    if updated then Array.init n (fun j -> (j, S.kernel f a 0)) else [||]
  in
  let prepared a = prepare_fill t (S.graph_launch gr (updates a)) in
  (t, prepared 0, prepared 1)

let launch_rows =
  (* [n] launches as parts of one submission: [4096-parts] shows what the
     driver's ordering costs per part of a large submission. *)
  let launching n () =
    let t = dev () in
    (t, launches t n)
  in
  let graph_launching (t, even, odd) =
    run t (if t.v land 1 = 0 then even else odd)
  in
  Thumper.group "launch"
    [
      row "1" (launching 1) (fun (t, s) -> run t s);
      row "64" (launching 64) (fun (t, s) -> run t s);
      row "4096-parts" (launching 4096) (fun (t, s) -> run t s);
      row "submits-64" (launching 1) (fun (t, s) ->
          for _ = 1 to 64 do
            submit t s
          done;
          wait t);
      row "graph-1" (graphing 1) graph_launching;
      row "graph-64" (graphing 64) graph_launching;
      row "graph-64-updated" (graphing ~updated:true 64) graph_launching;
    ]

(* The maker of a graph of 64 empty kernels and its release: link-time work. *)
let graph_rows =
  let making () =
    let t = dev () in
    ((C.capability t.g).graph, kernels (empty t.g) 64)
  in
  Thumper.group "graph"
    [ row "make-64" making (fun (graph, ks) -> ignore (get (graph ks))) ]

let copy_rows =
  let n = 256 * mib in
  let copying (dst, src) () =
    let t = dev () in
    let create m = Rig.Buffer.create ~memory:m t.d n in
    let dst = create dst and src = create src in
    (t, prepare t Rig.Submission.[| copy ~src:(Buffer src) ~dst:(Buffer dst) |])
  in
  let copy name kinds = row name (copying kinds) (fun (t, s) -> run t s) in
  Thumper.group "copy"
    [
      copy "h2d-256MiB" (Device, Pinned);
      copy "d2h-256MiB" (Pinned, Device);
      copy "d2d-256MiB" (Device, Device);
    ]

let alloc_rows =
  let alloc name n =
    row name dev (fun t -> C.free t.g (Option.get (C.alloc t.g Device n)))
  in
  Thumper.group "alloc" [ alloc "64KiB" (64 * kib); alloc "64MiB" (64 * mib) ]

let map_host_rows =
  let n = 256 * mib in
  Thumper.group "map-host"
    [
      row "256MiB"
        (fun () -> (dev (), H.pages n))
        (fun (t, p) -> C.free t.g (Option.get (C.map_host t.g p n)));
    ]

(* Loading an image of 140 kernels, the size of a kernel library's, every
   function's code placed (Rig_cuda.image), then unloading it. *)
let image_rows =
  let loading () =
    ( dev (),
      In_channel.with_open_bin
        (Filename.concat fixtures "many.cubin")
        In_channel.input_all )
  in
  Thumper.group "image"
    [
      row "load-140" loading (fun (t, bin) ->
          match C.image t.g bin with
          | Ok (Loaded m) -> C.unload t.g m
          | Ok (Place _) -> failwith "a CUDA image is placed by CUDA"
          | Error why -> failwith why);
    ]

(* An open of the GPU's device through rig and its close: the device's
   streams, its events and its first measure of the GPU's clock against the
   host's. *)
let open_rows =
  Thumper.group "open"
    [ row "close" (fun () -> ()) (fun () -> S.close (S.open_ ())) ]

let () =
  if Sys.file_exists "/dev/nvidiactl" then
    exit
    @@ Thumper.run "rig_cuda"
         [
           release_rows;
           launch_rows;
           graph_rows;
           copy_rows;
           alloc_rows;
           map_host_rows;
           image_rows;
           open_rows;
         ]
