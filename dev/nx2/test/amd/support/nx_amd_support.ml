(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Abi = Rig_amd_abi

let strf = Printf.sprintf

external code_object : unit -> string = "nx_amd_support_code_object"
external kernels : unit -> string array = "nx_amd_support_kernels"
external sizes : unit -> int * int * int * int * int = "nx_amd_support_sizes"

external entries :
  nativeint -> nativeint -> (string * int array) array -> nativeint
  = "nx_amd_support_entries"

external record_c : (int array * string) array -> string
  = "nx_amd_support_record"

type bytes =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

external run_arg : nativeint -> string -> bytes = "nx_amd_support_run"

external seq_arg : bytes array -> int array -> bytes * int * int
  = "nx_amd_support_seq"

external seq_fill : unit -> nativeint = "nx_amd_support_seq_fill"
external lock : string -> string -> int = "nx_amd_support_lock"

let threads, copy_threads, read_threads, read_vecs, hog_threads = sizes ()

(* The GPU lock *)

let gpu_lock = "/tmp/raven-rig-gpu.lock"

(* The longest wait for the lock, in seconds: the machine's suites, from every
   checkout and user, take it in turn. [lock] naps 100 ms each time it is
   refused. *)
let gpu_wait = 300

let holder () =
  match In_channel.with_open_bin gpu_lock In_channel.input_all with
  | note -> String.trim note
  | exception Sys_error _ -> "a process that left no note"

let rec take refused =
  match lock gpu_lock Sys.executable_name with
  | 0 -> ()
  | -1 when refused < gpu_wait * 10 -> take (refused + 1)
  | -1 ->
      failwith
        (strf "%s: still held after %d s, by %s" gpu_lock gpu_wait (holder ()))
  | errno -> failwith (strf "%s: errno %d" gpu_lock errno)

let hold_gpu () =
  if Sys.getenv_opt "RIG_GPU_LOCK_HELD" = None && Rig_amd_amdgpu.count () > 0
  then take 0

(* Devices and images *)

type image = { table : nativeint; names : string array; loaded : Rig.Image.t }

(* A device of GPU 0, its capability, and the harness loaded on it. *)
type device = { rig : Rig.t; cap : Abi.Capability.t; harness : image }
type gpu = { work : device; mutable beside : device option }

(* The values of a dispatch's words: known now, or left for the fill. *)
type hole = Known of int | Args | Threads of int | Groups of int

(* The dispatch on [gpu] of [k], the kernel [name], as [nx_amd_dispatch] holds
   it: its words with the fill's holes zero, and their indices, args then
   threads and groups. *)
let dispatch gpu name (k : Abi.Code_object.kernel) ~program =
  (* The dispatch names no scratch and no dispatch packet. *)
  if k.private_segment > 0 then
    failwith (strf "kernel %s takes scratch memory" name);
  if k.dispatch_ptr then
    failwith (strf "kernel %s reads its dispatch packet" name);
  let p =
    Abi.Pm4.run gpu
      (Abi.Pm4.dispatch gpu k ~program:(Known program) ~scratch:(Known 0)
         ~args:Args ~packet:(Known 0)
         ~threads:(Threads 0, Threads 1, Threads 2)
         ~groups:(Groups 0, Groups 1, Groups 2)
         ())
  in
  let known = function Known n -> Some (Int64.of_int n) | _ -> None in
  let words, holes = Abi.Packet.template known p in
  let at = Array.make 8 (-1) in
  at.(7) <- k.kernarg_size;
  List.iter
    (fun (i, w) ->
      match (w : hole Abi.Packet.word) with
      | W64 (Value Args) -> at.(0) <- i
      | W32 (Value (Threads d)) -> at.(1 + d) <- i
      | W32 (Value (Groups d)) -> at.(4 + d) <- i
      | _ ->
          failwith
            (strf "kernel %s: a dispatch word the fill cannot place" name))
    holes;
  if Array.mem (-1) at then
    failwith (strf "kernel %s: a dispatch lacks a word the fill places" name);
  (words, at)

let image_on rig (cap : Abi.Capability.t) bin names =
  let co =
    match Abi.Code_object.of_string bin with
    | Ok co -> co
    | Error why -> failwith why
  in
  match Rig.Image.load rig bin with
  | Error why -> failwith why
  | Ok loaded ->
      let entry n =
        match (Rig.Image.entry loaded n, Abi.Code_object.kernel co n) with
        | Some descriptor, Some k ->
            dispatch cap.gpu n k ~program:(descriptor - k.descriptor + k.entry)
        | _ -> failwith (strf "the code object has no kernel %s" n)
      in
      let table = entries cap.place cap.segment (Array.map entry names) in
      { table; names; loaded }

let open_device name =
  let rig =
    match
      Rig.open_ (module Rig_amd) ~name (fun () -> Rig_amd_amdgpu.open_ 0)
    with
    | Ok d -> d
    | Error why -> failwith why
  in
  let cap = Option.get (Rig.capability rig Abi.Capability.key) in
  (match cap.compute with
  | Pm4 -> ()
  | Aql _ -> failwith "nx.amd's fill places PM4: the GPU's queue reads AQL");
  { rig; cap; harness = image_on rig cap (code_object ()) (kernels ()) }

let opened = ref None

let gpu () =
  match !opened with
  | Some g -> g
  | None ->
      if Rig_amd_amdgpu.count () = 0 then
        Windtrap.skip ~reason:"the machine has no AMD GPU" ();
      hold_gpu ();
      let g = { work = open_device "AMD:nx2"; beside = None } in
      opened := Some g;
      g

let arch g = Abi.Gpu.processor g.work.cap.gpu

(* Each work-group processor has two compute units. *)
let cus g =
  let rec bits n = if n = 0 then 0 else (n land 1) + bits (n lsr 1) in
  Array.fold_left
    (Array.fold_left (fun n wgps -> n + (2 * bits wgps)))
    0 g.work.cap.wgps

(* [ns] nanoseconds as ticks of [d]'s GPU clock, and back. *)
let ticks d ns = ns * (d.cap.clock_hz / 1_000_000) / 1_000
let ns_of d t = float t *. 1e9 /. float d.cap.clock_hz
let image g bin names = image_on g.work.rig g.work.cap bin names
let harness g = g.work.harness

let kernel i name =
  match Array.find_index (String.equal name) i.names with
  | Some k -> k
  | None -> invalid_arg ("Nx_amd_support.record: no kernel " ^ name)

(* Records *)

type param = A of Rig.Buffer.t | W of int | D of int * int

(* A run of records holds the nx_amd_run a fill reads, [launches] records long,
   and keeps its image and the buffers it addresses alive while it is. *)
type records = {
  image : image;
  arg : bytes;
  launches : int;
  held : Rig.Buffer.t list;
}

type run =
  | Records of records
  | Copy of { src : Rig.Buffer.t; dst : Rig.Buffer.t }

let words ps =
  let b = Buffer.create 64 in
  let addrs = ref 0 and words = ref false in
  List.iter
    (function
      | A x ->
          if !words then invalid_arg "Nx_amd_support.record: address last";
          incr addrs;
          Buffer.add_int64_le b (Int64.of_int (Rig.Buffer.address x))
      | W x ->
          words := true;
          Buffer.add_int64_le b (Int64.of_int x)
      | D (x, y) ->
          words := true;
          Buffer.add_int32_le b (Int32.of_int x);
          Buffer.add_int32_le b (Int32.of_int y))
    ps;
  (!addrs, Buffer.contents b)

(* A launch is the record of its kernel in an image, and the buffers its
   parameters address. *)
type launch = image -> (int array * string) * Rig.Buffer.t list

let launch name ~groups:(gx, gy, gz) ~threads ps image =
  let addrs, words = words ps in
  ( ([| kernel image name; gx; gy; gz; threads; addrs; 0 |], words),
    List.filter_map (function A b -> Some b | _ -> None) ps )

let record image ls =
  let rs, held = List.split (List.map (fun l -> l image) ls) in
  let arg = run_arg image.table (record_c (Array.of_list rs)) in
  Records { image; arg; launches = List.length ls; held = List.concat held }

let driver_copy ~src ~dst = Copy { src; dst }

(* Runs *)

let part queue work = { Rig.Submission.queue; after = [||]; work }

let records = function
  | Records r -> r
  | Copy _ -> invalid_arg "Nx_amd_support: a driver copy has no records"

(* A part on ["COMPUTE:0"] that fills the record runs [rs], each repeated as
   [ns] says, declaring the ring words and segment bytes they take. *)
let fill rs ns =
  let runs = Array.of_list (List.map (fun r -> (records r).arg) rs) in
  let arg, ring_units, segment_bytes = seq_arg runs (Array.of_list ns) in
  part "COMPUTE:0"
    (Fill
       {
         fill = seq_fill ();
         arg = Rig.Buffer.of_bigarray arg;
         ring_units;
         segment_bytes;
       })

(* The parts that run [r] [n] times. *)
let body r n =
  match r with
  | Records _ -> [ fill [ r ] [ n ] ]
  | Copy { src; dst } ->
      List.init n (fun _ -> part "COPY:0" (Copy { src; dst }))

let submit d parts =
  let s = Rig.Submission.make ~reads:0 ~writes:0 d.rig (Array.of_list parts) in
  Rig.Point.value (Rig.submit s ~reads:[||] ~writes:[||] ~waits:[||])

(* The images and buffers a run's work uses stay alive until it is done
   (Rig.Image.entry). *)
let keep = function
  | Records r -> ignore (Sys.opaque_identity (r.image.loaded, r.arg, r.held))
  | Copy { src; dst } -> ignore (Sys.opaque_identity (src, dst))

let buffer g n = Rig.Buffer.create g.work.rig n

let read b =
  let h =
    Bigarray.Array1.create Bigarray.char Bigarray.c_layout (Rig.Buffer.length b)
  in
  Rig.Buffer.copy ~src:b ~dst:(Rig.Buffer.of_bigarray h);
  String.init (Bigarray.Array1.dim h) (Bigarray.Array1.get h)

let write b s = Rig.Buffer.copy ~src:(Rig.Buffer.of_string s) ~dst:b
let zero b = write b (String.make (Rig.Buffer.length b) '\000')

(* A hog holds the compute units it fills from a second device of the GPU, whose
   compute queue runs beside the work's. Its counter and compute units are the
   work device's memory, which the GPU's devices share. *)
type hog = {
  blocks : int;
  started : Rig.Buffer.t;
  cu : Rig.Buffer.t;
  device : device;
  run : run;
}

(* The hog's device, opened at the first hog and kept for the process. *)
let beside g =
  match g.beside with
  | Some d -> d
  | None ->
      let d = open_device "AMD:nx2-hog" in
      g.beside <- Some d;
      d

let hog g ~ns =
  let d = beside g in
  let blocks = Int.max 1 (cus g / 2) in
  let started = buffer g 4 and cu = buffer g (4 * blocks) in
  let on b = Option.get (Rig.Buffer.borrow d.rig b) in
  let run =
    record d.harness
      [
        launch "hog" ~groups:(blocks, 1, 1) ~threads:hog_threads
          [ A (on started); A (on cu); W (ticks d ns) ];
      ]
  in
  { blocks; started; cu; device = d; run }

let held_cus h =
  let s = read h.cu in
  List.init h.blocks (fun i -> Int32.to_int (String.get_int32_le s (4 * i)))

(* A delay gives up after 2 s: a hog's workgroups start well within it. *)
let hold_ns = 2_000_000_000

(* Beside a hog, the work waits on its queue until every hog workgroup holds its
   compute unit: rig orders nothing between two devices' queues. *)
let run ?beside g r =
  match beside with
  | None ->
      Rig.wait g.work.rig (submit g.work (body r 1));
      keep r
  | Some h ->
      zero h.started;
      let late = buffer g 4 in
      zero late;
      let wait =
        record (harness g)
          [
            launch "delay" ~groups:(1, 1, 1) ~threads:1
              [ A h.started; A late; D (h.blocks, 0); W (ticks g.work hold_ns) ];
          ]
      in
      let held = submit h.device [ fill [ h.run ] [ 1 ] ] in
      let v = submit g.work [ fill [ wait; r ] [ 1; 1 ] ] in
      Rig.wait g.work.rig v;
      Rig.wait h.device.rig held;
      List.iter keep [ r; wait; h.run ];
      if read late <> "\000\000\000\000" then
        failwith "the hog's workgroups did not all start within the hold"

(* rig.amd refuses a submission that takes more than half its argument segment.
   A harness launch takes 64 bytes of it, so a round of 1,024 takes 64 KiB. A
   submission holds at most 512 parts: 256 driver copies. *)
let round = 1024
let copies = 256

(* The words [p] as a part on [queue]. *)
let words_part queue p =
  part queue (Words (Rig.Buffer.of_string (Abi.Packet.encode Int64.of_int p)))

(* rig.amd's queues read none of a submission before all of it is placed
   (Rig_amd), so the host never starves the GPU: no hold is needed. A stamp is
   the GPU's clock, which [r]'s queue writes once the work before it is done. *)
let device_time g r ~count =
  let at = buffer g 16 in
  let slot i = Rig.Buffer.view at ~first:(8 * i) ~length:8 in
  let queue, per_round, clock =
    match r with
    | Records x ->
        ( "COMPUTE:0",
          Int.max 1 (round / x.launches),
          Abi.Pm4.copy_data Posted Clock )
    | Copy _ -> ("COPY:0", copies, Abi.Sdma.timestamp)
  in
  let stamp i = words_part queue (clock (Rig.Buffer.address (slot i))) in
  let rec go left span =
    if left = 0 then span
    else begin
      let n = Int.min left per_round in
      let ps = (stamp 0 :: body r n) @ [ stamp 1 ] in
      Rig.wait g.work.rig (submit g.work ps);
      keep r;
      ignore (Sys.opaque_identity ps);
      let s = read at in
      let t i = Int64.to_int (String.get_int64_le s (8 * i)) in
      go (left - n) (span + (t 1 - t 0))
    end
  in
  ns_of g.work (go count 0) *. 1e-9 /. float count

(* Ten million cycles: some 3.4 ms at 2.9 GHz, in which the launch's few
   microseconds weigh under 0.2%. *)
let cu_clock g =
  let spun = buffer g 8 in
  let r =
    record (harness g)
      [
        launch "cu_clock" ~groups:(1, 1, 1) ~threads:1 [ A spun; W 10_000_000 ];
      ]
  in
  let t = device_time g r ~count:1 in
  Int64.to_float (String.get_int64_le (read spun) 0) /. t *. 1e-6

(* Floors *)

let vectors what b =
  let n = Rig.Buffer.length b in
  if n mod 16 <> 0 then
    invalid_arg (strf "Nx_amd_support.%s: %d bytes are no whole vectors" what n);
  n / 16

let blocks n per = Int.max 1 ((n + per - 1) / per)

let floor_copy ~ins ~out =
  if ins = [] || List.length ins > 3 then
    invalid_arg "Nx_amd_support.floor_copy: one to three inputs";
  let vecs = vectors "floor_copy" out in
  let pad l d = l @ List.init (3 - List.length l) (fun _ -> d) in
  let addrs = pad (List.map (fun b -> A b) ins) (A out) in
  let lens = pad (List.map (fun b -> W (vectors "floor_copy" b)) ins) (W 0) in
  launch "floor_copy"
    ~groups:(blocks vecs copy_threads, 1, 1)
    ~threads:copy_threads
    (addrs @ [ A out ] @ lens @ [ W vecs ])

let floor_read g b =
  let vecs = vectors "floor_read" b in
  let per = read_threads * read_vecs in
  let out = buffer g (4 * blocks vecs per) in
  launch "floor_read"
    ~groups:(blocks vecs per, 1, 1)
    ~threads:read_threads [ A b; A out; W vecs ]

(* Memory *)

type draw = Uniform | Wide of int | Small

let generate (type v s) g b (dt : (v, s) Nx_array.Dtype.t) draw ~seed =
  (match dt with
  | Float4_e2m1fn | Int4 | Uint4 | Complex128 | Complex64 | Bit ->
      invalid_arg
        (strf "Nx_amd_support.generate: %s is not drawn"
           (Nx_array.Dtype.name dt))
  | _ -> ());
  let n = Rig.Buffer.length b * 8 / Nx_array.Dtype.bits dt in
  let draw, spread =
    match draw with Uniform -> (0, 0) | Wide e -> (1, e) | Small -> (2, 0)
  in
  run g
    (record (harness g)
       [
         launch "generate"
           ~groups:(blocks n threads, 1, 1)
           ~threads
           [ A b; W n; W seed; D (Nx_array.Dtype.code dt, draw); D (spread, 0) ];
       ])
