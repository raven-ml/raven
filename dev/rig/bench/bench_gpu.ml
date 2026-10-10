(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Submits on this machine's GPUs through rig, beside the same submits
   through each driver's C entries alone. An executable of its own: a process
   that links the GPU drivers makes every full collection longer, and rig's
   host rows run collections. *)

module B = Rig.Buffer
module Sub = Rig.Submission

let strf = Printf.sprintf

external floor_new : nativeint -> nativeint -> nativeint = "rig_bench_floor_new"

external floor_submit : nativeint -> int -> unit = "rig_bench_floor_submit"
[@@noalloc]

external floor_encode : nativeint -> int -> unit = "rig_bench_floor_encode"
[@@noalloc]

external floor_commit : nativeint -> unit = "rig_bench_floor_commit" [@@noalloc]

external floor_handles : nativeint -> nativeint array -> unit
  = "rig_bench_floor_handles"

external floor_fill : nativeint -> nativeint -> int -> int -> int -> unit
  = "rig_bench_floor_fill"

external floor_words : nativeint -> int -> int -> unit = "rig_bench_floor_words"
external floor_at : nativeint -> int -> unit = "rig_bench_floor_at"

external floor_copy : nativeint -> int -> nativeint -> nativeint -> int -> unit
  = "rig_bench_floor_copy"

external floor_launch : nativeint -> int -> nativeint -> int -> unit
  = "rig_bench_floor_launch"

external evict : string -> int = "rig_bench_evict"

let drain = 64


(* The runs a replay keeps in flight: each waits for the run [depth] back. At
   two, an R9700 flips between a fast and a slow mode within one process: a few
   hundred nanoseconds more host time a run let its queue drain, and each run
   then pays the GPU's wake. Three keeps the next run queued. *)
let depth = 3
let slots = 24
let runs = 100

(* Rig's still interval: a wait returns to OCaml at least this often. *)
let still_ms = 200
let row name setup f = Thumper.bench_with_setup ~setup name f

(* Files: a file's bytes copied into a GPU's memory and back, through
   Buffer.copy, as a program loads weights. Thumper measures each row in a child
   that leaves without running [at_exit], so only the bench's own process
   removes the files. *)

let file_bytes = 256 lsl 20
let dir = "files"
let path name = Filename.concat dir name

let clear () =
  if Sys.file_exists dir then
    Array.iter (fun f -> Sys.remove (Filename.concat dir f)) (Sys.readdir dir)

let () =
  clear ();
  if not (Sys.file_exists dir) then Sys.mkdir dir 0o755;
  at_exit (fun () ->
      clear ();
      Sys.rmdir dir)

(* The file [name] of [n] bytes, none zero, written once. *)
let data name n =
  let p = path name in
  if not (Sys.file_exists p) then begin
    let mib = 1 lsl 20 in
    let chunk = String.init mib (fun i -> Char.chr (1 + (i mod 255))) in
    Out_channel.with_open_bin (p ^ ".part") (fun oc ->
        let left = ref n in
        while !left > 0 do
          let k = Int.min !left mib in
          Out_channel.output_substring oc chunk 0 k;
          left := !left - k
        done);
    Sys.rename (p ^ ".part") p
  end;
  p

let ok = function Ok v -> v | Error why -> failwith why

(* Drops the file [p]'s pages from the page cache: its next read comes from the
   disk. *)
let evicted p =
  let e = evict p in
  if e <> 0 then failwith (strf "evicting %s: errno %d" p e)

(* Opens the file [p] and copies it into [dst], as a program loads its weights:
   each call opens the file anew, so a device maps its pages anew. It first
   collects the last call's opening, drains the disk and [dst]'s device, which
   releases the device's mapping and closes the file, and collects again, which
   unmaps the file's pages from the process; with [cold], it then drops them
   from the page cache, which keeps pages a process maps. *)
let load ~cold (p, dst) =
  Gc.full_major ();
  ignore (B.create Rig_disk.device 0);
  ignore (B.create (B.device dst) 0);
  Gc.full_major ();
  if cold then evicted p;
  B.copy ~src:(ok (Rig_disk.of_file p)) ~dst

(* Copies: 256 MiB between a GPU's memory and the host's, both ways. Host memory
   that starts on a page is memory the GPU maps; memory 16 bytes past a page,
   such as a C library's large allocation, the GPU maps none of, and the bytes
   go through the host's staging memory, one host copy per byte beside the
   GPU's. Host memory is written first: pages never written read as one page of
   zeros, which no copy of real data reads as fast. *)

let copy_bytes = 256 lsl 20

(* [n] bytes of host memory written, from 16 bytes past a page if [off]. *)
let written ?(off = false) n =
  let at = if off then 16 else 0 in
  let b = B.create Rig.host (n + at) in
  let ba = Bigarray.Array1.sub (B.bigarray Bigarray.char b) at n in
  Bigarray.Array1.fill ba 'w';
  if off then B.of_bigarray ba else b

(* A copy between host memory and a borrow of host memory: a size whose two
   buffers fit the last-level cache. A copy of a size that does not is bound
   by the machine's memory bandwidth, which every process on the host shares,
   and lands on levels up to 20% apart from one run to the next; in the cache
   a second copy still doubles the time. *)
let borrow_bytes = 4 lsl 20

(* The host's copy of as many bytes, which bounds a staged copy, and of the
   bytes a borrow copies. *)
let host_rows =
  let chars n () =
    let b () = B.bigarray Bigarray.char (written n) in
    (b (), b ())
  in
  let blit (a, b) = Bigarray.Array1.blit a b in
  let file () = (data "file" file_bytes, written file_bytes) in
  Thumper.group "floor/host"
    [
      row "memcpy-256M" (chars copy_bytes) blit;
      row "memcpy-4M" (chars borrow_bytes) blit;
      row "load-256M" file (load ~cold:false);
      row "load-cold-256M" file (load ~cold:true);
    ]

(* A copy of the step: its submission and the run it submits with, its
   argument, and the buffers each of its runs reads (the parameters, then the
   argument) and writes. *)
type gpu_copy = {
  gs : Sub.t;
  grun : Sub.Run.t;
  gargs : B.t;
  greads : B.t array;
  gwrites : B.t array;
}

type gpu_replay = {
  g : Rig.t;
  gcopies : gpu_copy array;
  keep : unit -> unit;  (** Holds what the copies' part runs. *)
}

(* A driver alone: its C entries through [entries], which submit [parts] parts,
   [sent] the last value they handed over. *)
type 'd alone = {
  drv : 'd;
  entries : nativeint;
  sent : int ref;
  parts : int;
  hold : unit -> unit;
}

(* Makes [p], a part of the first compute queue, the floor's part. *)
let floor_part f (p : Sub.part) =
  if p.queue <> "COMPUTE:0" then invalid_arg "floor_part: not COMPUTE:0";
  match p.work with
  | Sub.Fill { fill; arg; ring_units; segment_bytes } ->
      floor_fill f fill (B.address arg) ring_units segment_bytes
  | Sub.Words b -> floor_words f (B.address b) (B.length b / 4)
  | Sub.Copy _ -> invalid_arg "floor_part: a copy"
  | Sub.Launch _ -> invalid_arg "floor_part: a launch"

(* A kernel's work: its part; how a floor makes it the floor's part, which a
   launch's entry, held by no part, needs; and what must stay reachable while
   it runs. *)
type work = { part : Sub.part; floor : nativeint -> unit; keep : unit -> unit }

let work ?floor part keep =
  let floor = Option.value floor ~default:(fun f -> floor_part f part) in
  { part; floor; keep }

(* Stores one group of one thread into [run] for each launch of [s]. *)
let one_thread s run (parts : Sub.part array) =
  Array.iteri
    (fun i (p : Sub.part) ->
      match p.work with
      | Launch _ ->
          let b = Sub.block s i in
          Sub.Run.groups run b 1 1 1;
          Sub.Run.threads run b 1 1 1
      | Words _ | Fill _ | Copy _ -> ())
    parts

(* A GPU's submits through rig, beside the same submits through its driver's C
   entries alone. [empty] and [cost] submit no work and wait for each submit or
   every [drain]; [wait/reached] waits for a value reached; [kernel] submits the
   part [kernel] makes over three buffers and waits every [drain]. A floor
   submits and commits each value; [encode] commits every [drain], as [cost]'s
   waits do. The replay rows run [depth] copies of a step over [slots]
   parameters, each run waiting for its copy's last run: with no part, and with
   the part [kernel] makes, a launch of the vendor's smallest kernel. A floor
   spins on the word; [release-sleep], for a driver whose host writes the word
   ([sleeps]), and the replay floors wait as rig waits for that driver: in its
   [sleep] from the first read, or spinning. The kernel floor drives the device
   rig opened, through its driver's entries alone, after rig loaded the kernel.
   With [graph], the graph rows do the same with the part [graph] makes, the
   launch of a recorded step of 64 such kernels. Each case opens its GPU in its
   own worker, so that no process forks after a vendor library started: through
   rig with [opened], or the driver alone with [open_]. *)
let gpu_rows (type a) (module D : Rig.Driver with type t = a) ?(sleeps = false)
    ?(copies = true) ?graph v ~opened open_ ~kernel =
  let get = function Ok x -> x | Error why -> failwith why in
  let rig () =
    let g, _ = opened () in
    (g, Sub.make ~reads:0 ~writes:0 g [||], Sub.Run.make (), ref 0)
  in
  (* A device whose last value is reached. *)
  let reached () =
    let g, s, run, _ = rig () in
    let p = Rig.submit s ~run ~reads:[||] ~writes:[||] ~waits:[||] in
    let v = Rig.Point.value p in
    Rig.wait g v;
    (g, v)
  in
  (* A launch of [kernel] reading two buffers and writing a third, as an
     operation of two operands. *)
  let kernel_rig () =
    let g, d = opened () in
    let k = kernel d g in
    let buffers n = Array.init n (fun _ -> B.create g 8) in
    let s = Sub.make ~reads:2 ~writes:1 g [| k.part |] in
    let run = Sub.Run.make () in
    one_thread s run [| k.part |];
    (g, s, run, buffers 2, buffers 1, k.keep, ref 0)
  in
  let replay g parts keep =
    let gparams = Array.init slots (fun _ -> B.create g 8) in
    let copy () =
      let gargs = B.create g 8 in
      let gs = Sub.make ~reads:(slots + 1) ~writes:1 g parts in
      let grun = Sub.Run.make () in
      one_thread gs grun parts;
      {
        gs;
        grun;
        gargs;
        greads = Array.append gparams [| gargs |];
        gwrites = [| B.create g 8 |];
      }
    in
    ({ g; gcopies = Array.init depth (fun _ -> copy ()); keep }, ref 0)
  in
  let replaying () = replay (fst (opened ())) [||] ignore in
  let part_replaying make () =
    let g, d = opened () in
    let k = make d g in
    replay g [| k.part |] k.keep
  in
  let run (r, n) =
    let c = r.gcopies.(!n mod depth) in
    incr n;
    B.wait c.gargs B.Read_write;
    ignore
      (Rig.submit c.gs ~run:c.grun ~reads:c.greads ~writes:c.gwrites
         ~waits:[||])
  in
  let pipelined ((r, _) as x) =
    for _ = 1 to runs do
      run x
    done;
    Rig.wait r.g (Rig.submitted r.g);
    r.keep ()
  in
  let entries d = floor_new (D.facts d).edge 0n in
  let alone () =
    let drv = get (open_ ()) in
    { drv; entries = entries drv; sent = ref 0; parts = 0; hold = ignore }
  in
  (* The driver naming as many regions as a replay run names. *)
  let named a =
    let region () = Option.get (D.alloc a.drv Device 8) in
    floor_handles a.entries
      (Array.init (slots + 2) (fun _ -> (D.locate (region ())).handle));
    a
  in
  let part_alone make () =
    let g, drv = opened () in
    let k = make drv g in
    let a =
      {
        drv;
        entries = entries drv;
        sent = ref (Rig.submitted g);
        parts = 1;
        hold =
          (fun () ->
            ignore (Sys.opaque_identity (g, k.part));
            k.keep ());
      }
    in
    k.floor a.entries;
    floor_at a.entries !(a.sent);
    named a
  in
  let release a =
    incr a.sent;
    floor_submit a.entries a.parts
  in
  let spin a v =
    while D.signaled a.drv < v do
      Domain.cpu_relax ()
    done
  in
  let rec sleep a v =
    let seen = D.signaled a.drv in
    if seen < v then begin
      D.sleep a.drv ~seen ~still_ms;
      sleep a v
    end
  in
  let wait = if sleeps then sleep else spin in
  let floor_run a =
    wait a (!(a.sent) + 1 - depth);
    release a
  in
  let floor_pipelined a =
    for _ = 1 to runs do
      floor_run a
    done;
    wait a !(a.sent);
    a.hold ()
  in
  let to_device () =
    let g, _ = opened () in
    (ok (Rig_disk.of_file (data "file" file_bytes)), B.create g file_bytes)
  in
  let loading () =
    let g, _ = opened () in
    (data "file" file_bytes, B.create g file_bytes)
  in
  let from_device () =
    let g, _ = opened () in
    let p = path (strf "out-%s" v) in
    if Sys.file_exists p then Sys.remove p;
    (B.create g file_bytes, ok (Rig_disk.create_file p file_bytes))
  in
  (* Host memory and a borrow on the GPU of other host memory: the host copies
     between them, as between any memory it addresses. *)
  let borrowed () =
    let g, _ = opened () in
    (written borrow_bytes, Option.get (B.borrow g (written borrow_bytes)))
  in
  let copying host ~to_device () =
    let g, _ = opened () in
    let h = host () and d = B.create g copy_bytes in
    if to_device then (h, d) else (d, h)
  in
  let paged () = written copy_bytes
  and off_page () = written ~off:true copy_bytes in
  let copy (src, dst) = B.copy ~src ~dst in
  (* The driver alone copying between its pinned memory and its own on its copy
     queue (Rig.queues), into its own memory [to_device] or out of it. *)
  let copy_alone ~to_device () =
    let a = alone () in
    let queues = (D.facts a.drv).queues in
    let copies (q : Rig.queue) = List.mem Rig.Copy q.runs in
    let q =
      match List.find_index copies (List.tl queues) with
      | Some i -> Some (i + 1)
      | None -> if copies (List.hd queues) then Some 0 else None
    in
    let region k = (D.locate (Option.get (D.alloc a.drv k copy_bytes))).handle in
    let pinned = region Pinned and device = region Device in
    let dst, src = if to_device then (device, pinned) else (pinned, device) in
    floor_copy a.entries (Option.get q) dst src copy_bytes;
    floor_handles a.entries [| dst; src |];
    { a with parts = 1 }
  in
  let copy_once a =
    release a;
    spin a !(a.sent)
  in
  let copy_floors =
    if not copies then []
    else
      [
        row "copy-to-device-256M" (copy_alone ~to_device:true) copy_once;
        row "copy-from-device-256M" (copy_alone ~to_device:false) copy_once;
      ]
  in
  [
    Thumper.group (strf "file/%s" v)
      [
        row "to-device-256M" to_device (fun (src, dst) -> B.copy ~src ~dst);
        row "from-device-256M" from_device (fun (src, dst) -> B.copy ~src ~dst);
        row "load-256M" loading (load ~cold:false);
        row "load-cold-256M" loading (load ~cold:true);
      ];
    Thumper.group (strf "copy/%s" v)
      [
        row "host-to-borrow-4M" borrowed (fun (h, b) -> B.copy ~src:h ~dst:b);
        row "borrow-to-host-4M" borrowed (fun (h, b) -> B.copy ~src:b ~dst:h);
        row "to-device-256M" (copying paged ~to_device:true) copy;
        row "from-device-256M" (copying paged ~to_device:false) copy;
        row "to-device-off-page-256M" (copying off_page ~to_device:true) copy;
        row "from-device-off-page-256M" (copying off_page ~to_device:false) copy;
      ];
    Thumper.group (strf "wait/%s" v)
      [ row "reached" reached (fun (g, v) -> Rig.wait g v) ];
    Thumper.group (strf "submit/%s" v)
      [
        row "empty" rig (fun (_, s, run, _) ->
            Rig.Point.wait
              (Rig.submit s ~run ~reads:[||] ~writes:[||] ~waits:[||]));
        row "cost" rig (fun (_, s, run, n) ->
            let p = Rig.submit s ~run ~reads:[||] ~writes:[||] ~waits:[||] in
            incr n;
            if !n mod drain = 0 then Rig.Point.wait p);
        row "kernel" kernel_rig (fun (_, s, run, reads, writes, keep, n) ->
            let p = Rig.submit s ~run ~reads ~writes ~waits:[||] in
            incr n;
            if !n mod drain = 0 then begin
              Rig.Point.wait p;
              keep ()
            end);
      ];
    Thumper.group (strf "replay/%s" v)
      ([
         row "params-24" replaying run;
         row "pipelined-100" replaying pipelined;
         row "kernel-pipelined-100" (part_replaying kernel) pipelined;
       ]
      @ Option.fold graph ~none:[] ~some:(fun g ->
          [ row "graph-pipelined-100" (part_replaying g) pipelined ]));
    Thumper.group (strf "floor/%s" v)
      ([
         row "release" alone (fun a ->
             release a;
             spin a !(a.sent));
         row "cost" alone (fun a ->
             release a;
             if !(a.sent) mod drain = 0 then spin a !(a.sent));
         row "encode" alone (fun a ->
             incr a.sent;
             floor_encode a.entries a.parts;
             if !(a.sent) mod drain = 0 then begin
               floor_commit a.entries;
               spin a !(a.sent)
             end);
         row "run" (fun () -> named (alone ())) floor_run;
         row "pipelined-100" (fun () -> named (alone ())) floor_pipelined;
         row "kernel-pipelined-100" (part_alone kernel) floor_pipelined;
       ]
      @ Option.fold graph ~none:[] ~some:(fun g ->
          [ row "graph-pipelined-100" (part_alone g) floor_pipelined ])
      @ copy_floors
      @
      if sleeps then
        [
          row "release-sleep" alone (fun a ->
              release a;
              sleep a !(a.sent));
        ]
      else []);
  ]

(* Kernels: each vendor's smallest, as the work of one part of the first
   compute queue. *)

(* The fixtures of [vendor]'s suite. *)
let fixtures vendor = "../test/" ^ vendor ^ "/fixtures"
let host_of r = Option.get (Rig_metal.locate r).host

(* [step] over one thread, its argument pointing at a word of its own. *)
let metal_kernel d _ =
  let module S = Rig_metal_support in
  let image =
    match Rig_metal.image d (S.fixture ~dir:(fixtures "metal") "fill") with
    | Ok (Rig_edge.Loaded i) -> i
    | Ok (Place _) -> failwith "Metal asked to place its code"
    | Error why -> failwith why
  in
  let step = (Option.get (Rig_metal.entry image "step")).code in
  let region n = Option.get (Rig_metal.alloc d Device n) in
  let args = region 16 in
  let word = Option.get (Rig_metal.locate (region 16)).address in
  Rig_gpu_support.Host.set64 (host_of args) word;
  let f = S.dispatch ~pipeline:step args ~groups:1 ~threads:1 in
  work (S.part f) (fun () -> ignore (Sys.opaque_identity (image, f)))

(* [empty] over one thread as a launch, loaded by rig; the floor's launch is of
   the image the driver loads. *)
let cuda_kernel g d =
  let module S = Rig_cuda_support in
  let bin = S.fixture ~dir:(fixtures "cuda") "kernels.ptx" in
  let image =
    match Rig.Image.load d bin with Ok i -> i | Error why -> failwith why
  in
  let m, _ = S.kernels ~dir:(fixtures "cuda") g in
  let e = Option.get (Rig_cuda.entry m "empty") in
  let launch =
    Sub.Launch { image; kernel = "empty"; params = 16; refs = [||] }
  in
  work
    ~floor:(fun f -> floor_launch f e.code e.launch 16)
    { Sub.queue = "COMPUTE:0"; after = [||]; work = launch }
    (fun () -> ignore (Sys.opaque_identity m))

(* A graph of 64 [empty] kernels over one thread each, made through the device's
   capability, launched by one part. *)
let cuda_graph g _ =
  let module S = Rig_cuda_support in
  S.bind g;
  let image, kernels = S.kernels ~dir:(fixtures "cuda") g in
  let k = S.kernel (kernels "empty") 0 0 in
  let gr =
    match (Rig_cuda.capability g).graph (Array.make 64 k) with
    | Ok gr -> gr
    | Error why -> failwith why
  in
  let f = S.graph_launch gr [||] in
  work (S.part ~queue:"COMPUTE:0" f) (fun () ->
      ignore (Sys.opaque_identity (image, gr, f)))

(* [empty] over one block, loaded by rig. *)
let nv_kernel g d =
  let module S = Rig_nv_support in
  let k = S.kernels ~dir:(fixtures "nv") { S.d; g } in
  let l = S.launches g in
  work
    (S.words (S.launch l k "empty" ~blocks:1 []))
    (fun () -> ignore (Sys.opaque_identity (k, l)))

(* [empty] over one work-item as a launch, loaded by rig; the floor's launch is
   of the image the driver lays over pinned memory, which the host writes. *)
let amd_kernel g d =
  let binary =
    In_channel.with_open_bin
      (Filename.concat (fixtures "amd") "kernels_gfx1201.hsaco")
      In_channel.input_all
  in
  let image =
    match Rig.Image.load d binary with Ok i -> i | Error why -> failwith why
  in
  let m, r =
    match Rig_amd.image g binary with
    | Ok (Rig_edge.Place (n, lay)) ->
        let r = Option.get (Rig_amd.alloc g Pinned n) in
        let m, code = lay r in
        Rig_gpu_support.Host.write (Option.get (Rig_amd.locate r).host) code;
        (m, r)
    | Ok (Loaded _) -> failwith "AMD loaded its code itself"
    | Error why -> failwith why
  in
  let e = Option.get (Rig_amd.entry m "empty") in
  let launch = Sub.Launch { image; kernel = "empty"; params = 0; refs = [||] } in
  work
    ~floor:(fun f -> floor_launch f e.code e.launch 0)
    { Sub.queue = "COMPUTE:0"; after = [||]; work = launch }
    (fun () -> ignore (Sys.opaque_identity (m, r)))

(* A GPU opened through its suite's fixture, as rig's device and the
   driver's. *)
let fixture (type a) (module S : Rig_gpu_support.S with type gpu = a) () =
  let { S.d; g } = S.open_ () in
  (d, g)

let gpus =
  List.concat
    [
      (if Rig_metal_support.present () then
         gpu_rows
           (module Rig_metal)
           ~sleeps:true ~copies:false "metal"
           ~opened:(fixture (module Rig_metal_support))
           (fun () -> Rig_metal.open_ 0)
           ~kernel:metal_kernel
       else []);
      (if Rig_cuda_support.present () then
         gpu_rows
           (module Rig_cuda)
           "cuda"
           ~opened:(fixture (module Rig_cuda_support))
           (fun () -> Rig_cuda.open_ 0)
           ~kernel:cuda_kernel ~graph:cuda_graph
       else []);
      (if Rig_nv_support.present () then
         gpu_rows
           (module Rig_nv)
           "nv"
           ~opened:(fixture (module Rig_nv_support))
           (fun () -> Rig_nv_nvidia.open_ 0)
           ~kernel:nv_kernel
       else []);
      (if Rig_amd_support.present () then
         gpu_rows
           (module Rig_amd)
           "amd"
           ~opened:(fixture (module Rig_amd_support))
           Rig_amd_support.open_gpu
           ~kernel:amd_kernel
       else []);
    ]

let () =
  if gpus <> [] then Rig_gpu_lock.hold ();
  exit @@ Thumper.run "rig-gpu" (if gpus = [] then [] else host_rows :: gpus)
