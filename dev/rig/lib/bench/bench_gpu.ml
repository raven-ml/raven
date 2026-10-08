(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Submits on this machine's GPUs through the core, beside the same submits
   through each driver's C entries alone. An executable of its own: a process
   that links the GPU drivers makes every full collection longer, and the core's
   host rows run collections. *)

module C = Rig
module B = Rig.Buffer
module Sub = Rig.Submission

let strf = Printf.sprintf

external floor_new :
  nativeint -> nativeint -> nativeint -> nativeint -> nativeint
  = "rig_bench_floor_new"

external floor_submit : nativeint -> int -> unit = "rig_bench_floor_submit"
[@@noalloc]

external floor_handles : nativeint -> nativeint array -> unit
  = "rig_bench_floor_handles"

external floor_fill : nativeint -> nativeint -> int -> int -> int -> unit
  = "rig_bench_floor_fill"

external floor_words : nativeint -> int -> int -> unit = "rig_bench_floor_words"
external floor_at : nativeint -> int -> unit = "rig_bench_floor_at"

let drain = 64
let slots = 24
let runs = 100

(* The core's still interval: a wait returns to OCaml at least this often. *)
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

type gpu_copy = { gs : Sub.t; gargs : B.t; gout : B.t }

type gpu_replay = {
  g : C.t;
  gparams : B.t array;
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

(* A GPU's submits through the core, beside the same submits through its
   driver's C entries alone. [empty] and [cost] submit no work and wait for each
   submit or every [drain]. The replay rows run two copies of a step over
   [slots] parameters, each run waiting for its copy's run before last, as the
   Polled rows do: with no part, and with the part [kernel] makes, a launch of
   the vendor's smallest kernel. A floor spins on the word; [release-sleep], for
   a driver whose host writes the word ([sleeps]), and the replay floors wait as
   the core waits for that driver: in its [sleep] from the first read, or
   spinning. The kernel floor drives the device the core opened, through its
   driver's entries alone, after the core loaded the kernel. Each case opens its
   GPU in its own worker, so that no process forks after a vendor library
   started. *)
let gpu_rows (type a) (module D : C.Driver with type t = a) ?(sleeps = false) v
    ~name open_ ~kernel =
  let get = function Ok x -> x | Error why -> failwith why in
  let opened () =
    let d = ref None in
    let make () =
      Result.map
        (fun x ->
          d := Some x;
          x)
        (open_ ())
    in
    let g = get (C.open_ (module D) ~name make) in
    (g, Option.get !d)
  in
  let core () =
    let g, _ = opened () in
    (g, Sub.make ~reads:0 ~writes:0 ~waits:0 g [||], ref 0)
  in
  let replay g parts keep =
    let copy () =
      {
        gs = Sub.make ~reads:(slots + 1) ~writes:1 ~waits:0 g parts;
        gargs = B.create g 8;
        gout = B.create g 8;
      }
    in
    ( {
        g;
        gparams = Array.init slots (fun _ -> B.create g 8);
        gcopies = [| copy (); copy () |];
        keep;
      },
      ref 0 )
  in
  let replaying () = replay (fst (opened ())) [||] ignore in
  let kernel_replaying () =
    let g, d = opened () in
    let p, keep = kernel d g in
    replay g [| p |] keep
  in
  let run (r, n) =
    let c = r.gcopies.(!n land 1) in
    incr n;
    B.wait c.gargs B.Read_write;
    for i = 0 to slots - 1 do
      Sub.read c.gs i (Array.unsafe_get r.gparams i)
    done;
    Sub.read c.gs slots c.gargs;
    Sub.write c.gs 0 c.gout;
    ignore (C.submit c.gs)
  in
  let pipelined ((r, _) as x) =
    for _ = 1 to runs do
      run x
    done;
    C.wait r.g (C.submitted r.g);
    r.keep ()
  in
  let entries d = floor_new (D.self d) D.room_entry D.submit_entry 0n in
  let alone () =
    let drv = get (open_ ()) in
    { drv; entries = entries drv; sent = ref 0; parts = 0; hold = ignore }
  in
  (* The driver naming as many regions as a replay run names. *)
  let named a =
    let region () = Option.get (D.alloc a.drv `Device 8) in
    floor_handles a.entries
      (Array.init (slots + 2) (fun _ -> D.handle (region ())));
    a
  in
  let kernel_alone () =
    let g, drv = opened () in
    let p, hold = kernel drv g in
    let a =
      {
        drv;
        entries = entries drv;
        sent = ref (C.submitted g);
        parts = 1;
        hold =
          (fun () ->
            ignore (Sys.opaque_identity (g, p));
            hold ());
      }
    in
    floor_part a.entries p;
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
    wait a (!(a.sent) - 1);
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
    ( B.create C.host file_bytes,
      Option.get (B.borrow g (B.create C.host file_bytes)) )
  in
  [
    Thumper.group (strf "file/%s" v)
      [
        row "to-device-256M" to_device (fun (src, dst) -> B.copy ~src ~dst);
        row "from-device-256M" from_device (fun (src, dst) -> B.copy ~src ~dst);
      ];
    Thumper.group (strf "copy/%s" v)
      [
        row "host-to-borrow-256M" borrowed (fun (h, b) -> B.copy ~src:h ~dst:b);
        row "borrow-to-host-256M" borrowed (fun (h, b) -> B.copy ~src:b ~dst:h);
      ];
    Thumper.group (strf "submit/%s" v)
      [
        row "empty" core (fun (g, s, _) ->
            C.wait g (C.Point.value (C.submit s)));
        row "cost" core (fun (g, s, n) ->
            let p = C.submit s in
            incr n;
            if !n mod drain = 0 then C.wait g (C.Point.value p));
      ];
    Thumper.group (strf "replay/%s" v)
      [
        row "params-24" replaying run;
        row "pipelined-100" replaying pipelined;
        row "kernel-pipelined-100" kernel_replaying pipelined;
      ];
    Thumper.group (strf "floor/%s" v)
      ([
         row "release" alone (fun a ->
             release a;
             spin a !(a.sent));
         row "cost" alone (fun a ->
             release a;
             if !(a.sent) mod drain = 0 then spin a !(a.sent));
         row "run" (fun () -> named (alone ())) floor_run;
         row "pipelined-100" (fun () -> named (alone ())) floor_pipelined;
         row "kernel-pipelined-100" kernel_alone floor_pipelined;
       ]
      @
      if sleeps then
        [
          row "release-sleep" alone (fun a ->
              release a;
              sleep a !(a.sent));
        ]
      else []);
  ]

(* Kernels: each vendor's smallest, as one part of the first compute queue, and
   what must stay reachable while it runs. *)

let fixtures = "fixtures"
let host_of r = Option.get (Rig_metal.host r)

(* [step] over one thread, its argument pointing at a word of its own. *)
let metal_kernel d _ =
  let module S = Rig_metal_support in
  let image =
    match Rig_metal.image d (S.fixture ~dir:fixtures "fill") with
    | Ok (`Loaded i) -> i
    | Ok (`Place _) -> failwith "Metal asked to place its code"
    | Error why -> failwith why
  in
  let step = Option.get (Rig_metal.entry image "step") in
  let region n = Option.get (Rig_metal.alloc d `Device n) in
  let args = region 16 in
  let word = Option.get (Rig_metal.address (region 16)) in
  S.set64 (host_of args) 0 (Int64.of_int word);
  let f = S.dispatch ~pipeline:step args ~groups:1 ~threads:1 in
  (S.part f, fun () -> ignore (Sys.opaque_identity (image, f)))

(* [empty] over one thread. *)
let cuda_kernel g _ =
  let module S = Rig_cuda_support in
  S.bind g;
  let image, kernels = S.kernels ~dir:fixtures g in
  let f = S.launch ~count:1 (kernels "empty") ~grid:1 ~block:1 0 0 in
  ( S.part ~queue:"COMPUTE:0" f,
    fun () -> ignore (Sys.opaque_identity (image, f)) )

(* [empty] over one block, loaded by the core. *)
let nv_kernel g c =
  let module S = Rig_nv_support in
  let k = S.kernels ~file:"kernels_sm89.cubin" { S.d = c; g } in
  let l = S.launches g in
  ( S.words (S.launch l k "empty" ~blocks:1 []),
    fun () -> ignore (Sys.opaque_identity (k, l)) )

(* [empty] over one work-item, loaded by the core, as the packets that dispatch
   it. *)
let amd_kernel g c =
  let module Abi = Rig_amd_abi in
  let module Pm4 = Abi.Pm4 in
  let get = function Ok x -> x | Error why -> failwith why in
  let binary =
    In_channel.with_open_bin
      (Filename.concat fixtures "kernels_gfx1201.hsaco")
      In_channel.input_all
  in
  let k =
    Option.get
      (Abi.Code_object.kernel (get (Abi.Code_object.of_string binary)) "empty")
  in
  let p = get (C.Program.load c binary) in
  let base = Option.get (C.Program.entry p "empty") - k.descriptor in
  let gpu = (Rig_amd.capability g).gpu in
  let packets =
    Abi.Packet.encode Int64.of_int
      (Pm4.run gpu
         (Pm4.dispatch gpu k ~program:(base + k.entry) ~scratch:0 ~args:0
            ~packet:0 ~threads:(1, 1, 1) ~groups:(1, 1, 1) ()))
  in
  let words =
    Array.init
      (String.length packets / 4)
      (fun i ->
        Int32.to_int (String.get_int32_le packets (4 * i)) land 0xffff_ffff)
  in
  ( Rig_amd_support.words_part ~queue:"COMPUTE:0" words,
    fun () -> ignore (Sys.opaque_identity p) )

let gpus =
  List.concat
    [
      (if Sys.file_exists "/System/Library/Frameworks/Metal.framework" then
         gpu_rows
           (module Rig_metal)
           ~sleeps:true "metal" ~name:(Rig_metal.device_name 0)
           (fun () -> Rig_metal.open_ 0)
           ~kernel:metal_kernel
       else []);
      (if Sys.file_exists "/dev/nvidiactl" then
         gpu_rows
           (module Rig_cuda)
           "cuda" ~name:(Rig_cuda.device_name 0)
           (fun () -> Rig_cuda.open_ 0)
           ~kernel:cuda_kernel
         @ gpu_rows
             (module Rig_nv)
             "nv"
             ~name:(Rig_nv_nvidia.device_name 0)
             (fun () -> Rig_nv_nvidia.open_ 0)
             ~kernel:nv_kernel
       else []);
      (if Rig_amd_amdgpu.count () > 0 then
         gpu_rows
           (module Rig_amd)
           "amd"
           ~name:(Rig_amd_amdgpu.device_name 0)
           (fun () -> Rig_amd_amdgpu.open_ 0)
           ~kernel:amd_kernel
       else []);
    ]

(* The machine's GPU lock, which every suite and bench that acts on a GPU of the
   machine takes before it runs, so that no GPU row runs beside a GPU test. The
   process holds it until it exits, its forked workers with it. *)

external lock : string -> string -> int = "rig_bench_lock"

let gpu_lock = "/tmp/raven-rig-gpu.lock"

(* The longest wait for the lock, in seconds: the machine's suites, from every
   checkout and user, take it in turn. *)
let gpu_wait = 300

let holder () =
  match In_channel.with_open_bin gpu_lock In_channel.input_all with
  | note -> String.trim note
  | exception Sys_error _ -> "a process that left no note"

(* [lock] naps 100 ms each time it is refused. *)
let rec take refused =
  match lock gpu_lock Sys.executable_name with
  | 0 -> ()
  | -1 when refused < gpu_wait * 10 -> take (refused + 1)
  | -1 ->
      failwith
        (strf "%s: still held after %d s, by %s" gpu_lock gpu_wait (holder ()))
  | errno -> failwith (strf "%s: errno %d" gpu_lock errno)

let () =
  if gpus <> [] then take 0;
  exit @@ Thumper.run "rig-gpu" gpus
