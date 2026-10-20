(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Submits, copies and loads on this machine's GPUs through rig. An executable
   of its own: a process that links the GPU drivers makes every full collection
   longer, and rig's host rows run collections. *)

module B = Rig.Buffer
module Sub = Rig.Submission

let strf = Printf.sprintf

external evict : string -> int = "rig_bench_evict"

let drain = 64

(* The runs a replay keeps in flight: each waits for the run [depth] back. At
   two, an R9700 flips between a fast and a slow mode within one process: a few
   hundred nanoseconds more host time a run let its queue drain, and each run
   then pays the GPU's wake. Three keeps the next run queued. *)
let depth = 3
let slots = 24
let runs = 100

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

(* The copies a queued row submits before its one wait. *)
let queued = 8

(* [n] bytes of host memory written, from 16 bytes past a page if [off]. *)
let written ?(off = false) n =
  let at = if off then 16 else 0 in
  let b = B.create Rig.host (n + at) in
  let ba = Bigarray.Array1.sub (B.export b Bigarray.char) at n in
  Bigarray.Array1.fill ba 'w';
  if off then B.of_bigarray ba else b

(* A copy between host memory and a borrow of host memory: a size whose two
   buffers fit the last-level cache. A copy of a size that does not is bound by
   the machine's memory bandwidth, which every process on the host shares, and
   lands on levels up to 20% apart from one run to the next; in the cache a
   second copy still doubles the time. *)
let borrow_bytes = 4 lsl 20

(* The host's copy of as many bytes, which bounds a staged copy, and of the
   bytes a borrow copies. *)
let host_rows =
  let chars n () =
    let b () = B.export (written n) Bigarray.char in
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

(* A copy of the step: the run of its submission it submits with, its
   argument, and the buffers each of its runs uses: the parameters and the
   argument, read, then its output, written. *)
type gpu_copy = {
  grun : Sub.Run.t;
  gargs : B.t;
  gbuffers : B.t array;
}

type gpu_replay = {
  gcopies : gpu_copy array;
  keep : unit -> unit;  (** Holds what the copies' part runs. *)
}

(* A kernel's work, and the objects it keeps. *)
type work = { work : Rig_gpu_support.work; keep : unit -> unit }

(* A launch's block runs one group of one thread; a fill reads no geometry. *)
let work part keep =
  let init s run i =
    match Sub.block s i with
    | b ->
        Sub.Run.groups run b 1 1 1;
        Sub.Run.threads run b 1 1 1
    | exception Invalid_argument _ -> ()
  in
  { work = { (Rig_gpu_support.work part) with init }; keep }

(* A GPU's submits through rig. [empty] and [cost] submit no work and wait for
   each submit or every [drain]; [wait/reached] waits for a value reached;
   [kernel] submits the part [kernel] makes over three buffers and waits every
   [drain]. The replay rows run [depth] copies of a step over [slots]
   parameters, each run waiting for its copy's last run: with no part, and with
   the part [kernel] makes, a launch of the vendor's smallest kernel. With
   [graph], the graph row does the same with the part [graph] makes, the launch
   of a recorded step of 64 such kernels. Each case opens its GPU in its own
   worker with [opened], so that no process forks after a vendor library
   started. *)
let gpu_rows ?graph v ~opened ~kernel =
  let rig () =
    let g, _ = opened () in
    (g, Sub.Run.make (Sub.make g [||]), ref 0)
  in
  (* A device whose last value is reached. *)
  let reached () =
    let _, run, _ = rig () in
    let p = Rig.submit run ~buffers:[||] in
    Rig.Point.wait p;
    p
  in
  (* A launch of [kernel] reading two buffers and writing a third, as an
     operation of two operands. *)
  let kernel_rig () =
    let g, d = opened () in
    let k = kernel d g in
    let access = [| Rig.Read; Rig.Read; Rig.Read_write |] in
    let uses = Array.to_list (Array.map (fun a -> (B.create g 8, a)) access) in
    let run, buffers = Rig_gpu_support.prepare ~fixed:uses g [ k.work ] in
    (g, run, buffers, k.keep, ref 0)
  in
  let replay g parts keep =
    let gparams = Array.init slots (fun _ -> B.create g 8) in
    let access =
      Array.append (Array.make (slots + 1) Rig.Read) [| Rig.Read_write |]
    in
    let copy () =
      let gargs = B.create g 8 in
      let gbuffers = Array.concat [ gparams; [| gargs; B.create g 8 |] ] in
      let uses = Array.to_list (Array.combine gbuffers access) in
      let grun, gbuffers = Rig_gpu_support.prepare ~fixed:uses g parts in
      { grun; gargs; gbuffers }
    in
    ({ gcopies = Array.init depth (fun _ -> copy ()); keep }, ref 0)
  in
  let replaying () = replay (fst (opened ())) [] ignore in
  let part_replaying make () =
    let g, d = opened () in
    let k = make d g in
    replay g [ k.work ] k.keep
  in
  let run (r, n) =
    let c = r.gcopies.(!n mod depth) in
    incr n;
    B.wait c.gargs Rig.Read_write;
    Rig.submit c.grun ~buffers:c.gbuffers
  in
  let pipelined ((r, _) as x) =
    for _ = 1 to runs - 1 do
      ignore (run x)
    done;
    Rig.Point.wait (run x);
    r.keep ()
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
  let within_device bytes () =
    let g, _ = opened () in
    let src = B.create g bytes and dst = B.create g bytes in
    B.copy ~src:(written bytes) ~dst:src;
    (src, dst)
  in
  let copy (src, dst) = B.copy ~src ~dst in
  (* Eight small copies from host memory off a page, which no GPU borrows, so
     each goes through the host's staging memory, queued one after the other
     and waited for once. *)
  let uploads () =
    let g, _ = opened () in
    Array.init queued (fun _ ->
        (written ~off:true (4 lsl 10), B.create g (4 lsl 10)))
  in
  let upload_all ends =
    let last = ref None in
    Array.iter
      (fun (src, dst) ->
        last := Some (B.submit_copy ~src ~dst [| B.row (4 lsl 10) |]))
      ends;
    Option.iter Rig.Point.wait !last
  in
  let filling bytes () =
    let g, _ = opened () in
    B.create g bytes
  in
  let blit_into bytes () =
    let g, _ = opened () in
    (String.make bytes 'b', B.create g bytes)
  in
  (* Thumper drops each returned buffer. Its warm-up fills the device's cache;
     measured batches include the collections and drains that make reuse
     possible. *)
  let allocating () = fst (opened ()) in
  let allocation name bytes =
    Thumper.bench_with_setup ~setup:allocating
      ~metrics:
        Thumper.Metric.
          [ wall_time; alloc_words; minor_collections; major_collections ]
      name
      (fun d -> B.create d bytes)
  in
  [
    Thumper.group (strf "buffer/%s" v)
      [
        allocation "create-drop-64KiB" (64 lsl 10);
        allocation "create-drop-64MiB" (64 lsl 20);
      ];
    Thumper.group (strf "file/%s" v)
      [
        row "to-device-256M" to_device (fun (src, dst) -> B.copy ~src ~dst);
        row "from-device-256M" from_device (fun (src, dst) -> B.copy ~src ~dst);
        row "load-256M" loading (load ~cold:false);
        row "load-cold-256M" loading (load ~cold:true);
      ];
    Thumper.group (strf "copy/%s" v)
      [
        row "within-device-4K" (within_device (4 lsl 10)) copy;
        row "within-device-256M" (within_device copy_bytes) copy;
        row "blit-from-string-4K" (blit_into (4 lsl 10)) (fun (s, b) ->
            B.blit_from_string s 0 b 0 (String.length s));
        row "host-to-borrow-4M" borrowed (fun (h, b) -> B.copy ~src:h ~dst:b);
        row "borrow-to-host-4M" borrowed (fun (h, b) -> B.copy ~src:b ~dst:h);
        row "to-device-256M" (copying paged ~to_device:true) copy;
        row "from-device-256M" (copying paged ~to_device:false) copy;
        row "to-device-off-page-256M" (copying off_page ~to_device:true) copy;
        row "from-device-off-page-256M" (copying off_page ~to_device:false) copy;
        row
          (strf "queued/host-to-device-%dx4K" queued)
          uploads upload_all;
      ];
    (* A fill of the device's own memory, which its copy engine sets where
       the host does not address it; one byte is what nx sets at the end
       of each sub-byte result. *)
    Thumper.group (strf "fill/%s" v)
      [
        row "device-64M" (filling (64 lsl 20)) (fun b -> B.fill b 'f');
        row "device-1B" (filling 1) (fun b -> B.fill b 'f');
      ];
    Thumper.group (strf "wait/%s" v)
      [ row "reached" reached Rig.Point.wait ];
    Thumper.group (strf "submit/%s" v)
      [
        row "empty" rig (fun (_, run, _) ->
            Rig.Point.wait (Rig.submit run ~buffers:[||]));
        row "cost" rig (fun (_, run, n) ->
            let p = Rig.submit run ~buffers:[||] in
            incr n;
            if !n mod drain = 0 then Rig.Point.wait p);
        row "kernel" kernel_rig (fun (_, run, buffers, keep, n) ->
            let p = Rig.submit run ~buffers in
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
  ]

(* Kernels: each vendor's smallest, as the work of one part of the first compute
   queue. *)

(* The fixtures of [vendor]'s suite. *)
let fixtures vendor = "../test/" ^ vendor ^ "/fixtures"

(* [bump] over one thread as a launch, loaded by rig, its parameters left 0: a
   count of 0 bytes, so it writes nothing. A case runs in a forked child, which
   reaches no Metal compiler: the pipelines of [fill]'s kernels come from
   Metal's shader cache. *)
let metal_kernel _ d =
  let module S = Rig_metal_support in
  let bin = S.fixture ~dir:(fixtures "metal") "fill" in
  let image =
    match Rig.Image.load d bin with Ok i -> i | Error why -> failwith why
  in
  let launch =
    Sub.launch (Option.get (Rig.Image.resolve image "bump")) ~params:16
  in
  work launch ignore

(* [empty] over one thread as a launch, loaded by rig. *)
let cuda_kernel _ d =
  let module S = Rig_cuda_support in
  let bin = S.fixture ~dir:(fixtures "cuda") "kernels.ptx" in
  let image =
    match Rig.Image.load d bin with Ok i -> i | Error why -> failwith why
  in
  let launch =
    Sub.launch (Option.get (Rig.Image.resolve image "empty")) ~params:16
  in
  work launch ignore

(* A graph of 64 [empty] kernels over one thread each, made through the device's
   capability, launched by one part. *)
let cuda_graph g d =
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
  {
    work = S.part d f;
    keep = (fun () -> ignore (Sys.opaque_identity (image, gr, f)));
  }

(* [empty] over one block, loaded by rig. *)
let nv_kernel g d =
  let module S = Rig_nv_support in
  let k = S.kernels ~dir:(fixtures "nv") { S.d; g } in
  let l = S.launches g in
  let ws = S.launch l k "empty" ~blocks:1 [] in
  let words = B.create Rig.host (4 * Array.length ws) in
  B.write words Bigarray.int32 (fun a ->
      Array.iteri (fun i w -> a.{i} <- Int32.of_int w) ws);
  work (Sub.words words) (fun () -> ignore (Sys.opaque_identity (k, l)))

(* [empty] over one work-item as a launch, loaded by rig. *)
let amd_kernel _ d =
  let binary =
    In_channel.with_open_bin
      (Filename.concat (fixtures "amd") "kernels_gfx1201.hsaco")
      In_channel.input_all
  in
  let image =
    match Rig.Image.load d binary with Ok i -> i | Error why -> failwith why
  in
  let launch =
    Sub.launch (Option.get (Rig.Image.resolve image "empty")) ~params:0
  in
  work launch ignore

(* A GPU opened through its suite's fixture, as rig's device and the
   driver's. *)
let fixture (type a) (module S : Rig_gpu_support.S with type gpu = a) () =
  let { S.d; g } = S.open_ () in
  (d, g)

let gpus =
  List.concat
    [
      (if Rig_metal_support.present () then
         gpu_rows "metal"
           ~opened:(fixture (module Rig_metal_support))
           ~kernel:metal_kernel
       else []);
      (if Rig_cuda_support.present () then
         gpu_rows "cuda"
           ~opened:(fixture (module Rig_cuda_support))
           ~kernel:cuda_kernel ~graph:cuda_graph
       else []);
      (if Rig_nv_support.present () then
         gpu_rows "nv"
           ~opened:(fixture (module Rig_nv_support))
           ~kernel:nv_kernel
       else []);
      (if Rig_amd_support.present () then
         gpu_rows "amd"
           ~opened:(fixture (module Rig_amd_support))
           ~kernel:amd_kernel
       else []);
    ]

let () =
  exit @@ Thumper.run "rig-gpu" (if gpus = [] then [] else host_rows :: gpus)
