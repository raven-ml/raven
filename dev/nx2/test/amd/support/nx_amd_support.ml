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

external run_arg : nativeint -> string -> bytes * int * int
  = "nx_amd_support_run"

external fill_fn : unit -> nativeint = "nx_amd_support_fill"
external library_c : int -> string option = "nx_amd_support_library"

external library_kernels : unit -> string array
  = "nx_amd_support_library_kernels"

(* An operand as the plan's stub reads it: address, dtype, shape, strides. *)
type op_c = int * int * int array * int array

external call_c : op_c array -> int array -> int array -> int -> bool -> bytes
  = "nx_amd_support_call"

external plan_c : bytes -> (string * int * int) option = "nx_amd_support_plan"
external plan_only : bytes -> int = "nx_amd_support_plan_only" [@@noalloc]
external rebase : string -> int -> string = "nx_amd_support_rebase"

let threads, copy_threads, read_threads, read_vecs, hog_threads = sizes ()

(* The GPU lock *)

let hold_gpu () = if Rig_amd_amdgpu.count () > 0 then Rig_gpu_lock.hold ()

(* Devices and images *)

let ok = function Ok x -> x | Error why -> failwith why

(* A loaded code object, the table of its kernels' dispatches, and the scratch
   buffer they run with, if one takes scratch memory. *)
type image = {
  table : nativeint;
  names : string array;
  loaded : Rig.Image.t;
  scratch : Rig.Buffer.t option;
}

(* A device of GPU 0, its capability, the harness loaded on it, nx.amd's
   code object, loaded at its first use, and the two 64-bit stamps
   device_time reads, in memory the host reads without a copy. *)
type device = {
  rig : Rig.t;
  cap : Abi.Capability.t;
  harness : image;
  library : image Lazy.t;
  stamps : Rig.Buffer.t;
}

type gpu = { work : device; mutable beside : device option }

(* The values of a dispatch's words: known now, or left for the fill. *)
type hole = Known of int | Args | Threads of int | Groups of int

(* The dispatch on [gpu] of [k], the kernel [name], with the scratch buffer at
   [scratch], as [nx_amd_dispatch] holds it: its words with the fill's holes
   zero, and their indices, args then threads and groups, and the bytes of
   arguments [k] reads. The dispatch names no dispatch packet. *)
let dispatch gpu name (k : Abi.Code_object.kernel) ~program ~scratch =
  if k.dispatch_ptr then
    failwith (strf "kernel %s reads its dispatch packet" name);
  let p =
    Abi.Pm4.run gpu
      (Abi.Pm4.dispatch gpu k ~program:(Known program) ~scratch:(Known scratch)
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
  let co = ok (Abi.Code_object.of_string bin)
  and loaded = ok (Rig.Image.load rig bin) in
  let kernel n =
    match (Rig.Image.entry loaded n, Abi.Code_object.kernel co n) with
    | Some descriptor, Some k -> (descriptor, k)
    | _ -> failwith (strf "the code object has no kernel %s" n)
  in
  let ks = Array.map kernel names in
  (* One buffer serves every kernel: a queue runs one at a time. *)
  let lane =
    Array.fold_left
      (fun n (_, k) -> Int.max n k.Abi.Code_object.private_segment)
      0 ks
  in
  let scratch =
    if lane = 0 then None
    else Some (Rig.Buffer.create rig (Abi.Scratch.size cap.gpu lane))
  in
  let base = Option.fold ~none:0 ~some:Rig.Buffer.address scratch in
  let entry n (descriptor, (k : Abi.Code_object.kernel)) =
    dispatch cap.gpu n k ~scratch:base
      ~program:(descriptor - k.descriptor + k.entry)
  in
  let table = entries cap.place cap.segment (Array.map2 entry names ks) in
  { table; names; loaded; scratch }

(* nx.amd's code object for the GPU's processor, gfx1201's under 1201. *)
let library_object (cap : Abi.Capability.t) =
  let name = Abi.Gpu.processor cap.gpu in
  match
    library_c (int_of_string (String.sub name 3 (String.length name - 3)))
  with
  | Some co -> co
  | None | (exception Failure _) ->
      failwith (strf "nx.amd has no code object for %s" name)

let open_device name =
  let rig =
    ok (Rig.open_ (module Rig_amd) ~name (fun () -> Rig_amd_amdgpu.open_ 0))
  in
  let cap = Option.get (Rig.capability rig Abi.Capability.key) in
  (match cap.compute with
  | Pm4 -> ()
  | Aql _ -> failwith "nx.amd's fill places PM4: the GPU's queue reads AQL");
  let harness = image_on rig cap (code_object ()) (kernels ()) in
  let library =
    lazy (image_on rig cap (library_object cap) (library_kernels ()))
  in
  let stamps = Rig.Buffer.create ~memory:Rig.Buffer.Pinned rig 16 in
  { rig; cap; harness; library; stamps }

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

let wgps g =
  let rec bits n = if n = 0 then 0 else (n land 1) + bits (n lsr 1) in
  Array.fold_left (Array.fold_left (fun n w -> n + bits w)) 0 g.work.cap.wgps

(* Each work-group processor has two compute units. *)
let cus g = 2 * wgps g

(* [ns] nanoseconds as ticks of [d]'s GPU clock, and back. *)
let ticks d ns = ns * (d.cap.clock_hz / 1_000_000) / 1_000
let ns_of d t = float t *. 1e9 /. float d.cap.clock_hz
let harness g = g.work.harness
let library g = Lazy.force g.work.library

let library_size g =
  (Array.length (library g).names, String.length (library_object g.work.cap))

let kernel i name =
  match Array.find_index (String.equal name) i.names with
  | Some k -> k
  | None -> invalid_arg ("Nx_amd_support.record: no kernel " ^ name)

(* Records *)

type param = A of Rig.Buffer.t | W of int | D of int * int

(* A run of [launches] records keeps its image and the buffers it addresses
   alive while it is. *)
type run =
  | Records of {
      image : image;
      records : string;
      launches : int;
      held : Rig.Buffer.t list;
    }
  | Copy of { src : Rig.Buffer.t; dst : Rig.Buffer.t }

(* The count of [ps]'s addresses, and their bytes. *)
let words ps =
  let b = Buffer.create 64 and i64 x = Int64.of_int x in
  let address = function A _ -> true | W _ | D _ -> false in
  let addrs = List.length (List.filter address ps) in
  if List.exists address (List.filteri (fun i _ -> i >= addrs) ps) then
    invalid_arg "Nx_amd_support.record: address last";
  List.iter
    (function
      | A x -> Buffer.add_int64_le b (i64 (Rig.Buffer.address x))
      | W x -> Buffer.add_int64_le b (i64 x)
      | D (x, y) ->
          Buffer.add_int32_le b (Int32.of_int x);
          Buffer.add_int32_le b (Int32.of_int y))
    ps;
  (addrs, Buffer.contents b)

(* A launch is the record of its kernel in an image, and the buffers its
   parameters address. *)
type launch = image -> (int array * string) * Rig.Buffer.t list

let launch name ~groups:(gx, gy, gz) ~threads ps image =
  let addrs, words = words ps in
  ( ([| kernel image name; gx; gy; gz; threads; addrs; 0 |], words),
    List.filter_map (function A b -> Some b | _ -> None) ps )

let record image ls =
  let rs, held = List.split (List.map (fun l -> l image) ls) in
  let records = record_c (Array.of_list rs) in
  Records { image; records; launches = List.length ls; held = List.concat held }

let driver_copy ~src ~dst = Copy { src; dst }

(* Runs *)

let part queue work = { Rig.Submission.queue; after = [||]; work }

(* The parts that run [r] [n] times: nx_amd_fill over its records, [n] times
   over, declaring the ring words and segment bytes they take. *)
let body r n =
  match r with
  | Copy { src; dst } ->
      List.init n (fun _ -> part "COPY:0" (Copy { src; dst }))
  | Records x ->
      let records = String.concat "" (List.init n (fun _ -> x.records)) in
      let arg, ring_units, segment_bytes = run_arg x.image.table records in
      let arg = Rig.Buffer.of_bigarray arg in
      [
        part "COMPUTE:0"
          (Fill { fill = fill_fn (); arg; ring_units; segment_bytes });
      ]

let submit d parts =
  let s = Rig.Submission.make ~reads:0 ~writes:0 d.rig (Array.of_list parts) in
  Rig.Point.value (Rig.submit s ~reads:[||] ~writes:[||] ~waits:[||])

(* The images and buffers a run's work uses stay alive until it is done
   (Rig.Image.entry). *)
let keep = function
  | Records r ->
      ignore (Sys.opaque_identity (r.image.loaded, r.image.scratch, r.held))
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

(* A hog holds the work-group processors it fills from a second device of the
   GPU, whose compute queue runs beside the work's, until the work's queue
   releases it. Its words are the work device's memory, which the GPU's
   devices share: [started] counts the hog's workgroups that hold their
   processors, [release] lets them go, and [late] and [let_go] say that the
   work's queue or the hog gave up waiting for the other. *)
type hog = {
  blocks : int;
  started : Rig.Buffer.t;
  release : Rig.Buffer.t;
  late : Rig.Buffer.t;
  let_go : Rig.Buffer.t;
  wgp : Rig.Buffer.t;
  device : device;
  hold : run;
  wait : run;
  free : run;
}

(* The hog's device, opened at the first hog and kept for the process. *)
let beside g =
  match g.beside with
  | Some d -> d
  | None ->
      let d = open_device "AMD:nx2-hog" in
      g.beside <- Some d;
      d

(* Each side gives up on the other after 2 s: a hog's workgroups start, and
   the work beside it ends, well within it. *)
let hold_ns = 2_000_000_000

let hog g =
  let d = beside g in
  let blocks = Int.max 1 (wgps g / 2) in
  let word () = buffer g 4 in
  let started = word () and release = word () and late = word () in
  let let_go = word () and wgp = buffer g (4 * blocks) in
  let on b = Option.get (Rig.Buffer.borrow d.rig b) in
  let hold =
    record d.harness
      [
        launch "hog" ~groups:(blocks, 1, 1) ~threads:hog_threads
          [
            A (on started);
            A (on wgp);
            A (on release);
            A (on let_go);
            W (ticks d hold_ns);
          ];
      ]
  in
  let wait =
    record (harness g)
      [
        launch "delay" ~groups:(1, 1, 1) ~threads:1
          [ A started; A late; D (blocks, 0); W (ticks g.work hold_ns) ];
      ]
  in
  let free =
    record (harness g)
      [ launch "release" ~groups:(1, 1, 1) ~threads:1 [ A release ] ]
  in
  { blocks; started; release; late; let_go; wgp; device = d; hold; wait; free }

let held_wgps h =
  let s = read h.wgp in
  List.init h.blocks (fun i -> Int32.to_int (String.get_int32_le s (4 * i)))

(* Beside a hog, the work waits on its queue until every hog workgroup holds its
   processor, and releases them once done: rig orders nothing between two
   devices' queues. *)
let run ?beside g r =
  match beside with
  | None ->
      Rig.wait g.work.rig (submit g.work (body r 1));
      keep r
  | Some h ->
      List.iter zero [ h.started; h.release; h.late; h.let_go ];
      let held = submit h.device (body h.hold 1) in
      let v = submit g.work (body h.wait 1 @ body r 1 @ body h.free 1) in
      Rig.wait g.work.rig v;
      Rig.wait h.device.rig held;
      List.iter keep [ r; h.wait; h.free; h.hold ];
      let set b = read b <> "\000\000\000\000" in
      if set h.late then
        failwith "the hog's workgroups did not all start within the hold";
      if set h.let_go then failwith "the hog let go before the work was done"

(* rig.amd refuses a submission that takes more than half its argument segment.
   A launch's parameters take at most 256 bytes of it, so a round of 1,024
   launches at most 256 KiB. A submission holds at most 512 parts: 256 driver
   copies. *)
let round = 1024
let copies = 256

(* The runs of [r] a submission holds. *)
let per_round = function
  | Records x -> Int.max 1 (round / x.launches)
  | Copy _ -> copies

(* The words [p] as a part on [queue]. *)
let words_part queue p =
  part queue (Words (Rig.Buffer.of_string (Abi.Packet.encode Int64.of_int p)))

let enqueue g ~count r =
  let rec go left =
    if left > 0 then begin
      let n = Int.min left (per_round r) in
      ignore (submit g.work (body r n));
      go (left - n)
    end
  in
  go count

(* rig.amd's queues read none of a submission before all of it is placed
   (Rig_amd), so the host never starves the GPU: no hold is needed. A stamp is
   the GPU's clock, which [r]'s queue writes once the work before it is done,
   into the device's stamps: made once, so that a call allocates no GPU
   memory, and read by the host without a copy. *)
let device_time g r ~count =
  let at = g.work.stamps in
  let slot i = Rig.Buffer.view at ~first:(8 * i) ~length:8 in
  let queue, clock =
    match r with
    | Records _ -> ("COMPUTE:0", Abi.Pm4.copy_data Posted Clock)
    | Copy _ -> ("COPY:0", Abi.Sdma.timestamp)
  in
  let stamp i = words_part queue (clock (Rig.Buffer.address (slot i))) in
  let rec go left span =
    if left = 0 then span
    else begin
      let n = Int.min left (per_round r) in
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

(* Contractions *)

type operand = {
  buffer : Rig.Buffer.t;
  dtype : int;
  shape : int array;
  strides : int array;
  first : int;
}

let ops a b init y =
  match init with None -> [ a; b; y ] | Some i -> [ a; b; i; y ]

let call ~a ~b ?init ~y ~batch ~contracting ~acc () =
  let op (o : operand) : op_c =
    (Rig.Buffer.address o.buffer + o.first, o.dtype, o.shape, o.strides)
  in
  let pairs l = Array.of_list (List.concat_map (fun (x, y) -> [ x; y ]) l) in
  call_c
    (Array.of_list (List.map op (ops a b init y)))
    (pairs batch) (pairs contracting) acc (Option.is_some init)

let planner ~a ~b ?init ~y ~batch ~contracting ~acc () =
  let c = call ~a ~b ?init ~y ~batch ~contracting ~acc () in
  fun () -> plan_only c

let contract g ~a ~b ?init ~y ~batch ~contracting ~acc () =
  let plan = plan_c (call ~a ~b ?init ~y ~batch ~contracting ~acc ()) in
  let held = List.map (fun (o : operand) -> o.buffer) (ops a b init y) in
  let image = library g in
  match plan with
  | None -> None
  | Some (records, 0, launches) ->
      Some (Records { image; records; launches; held })
  | Some (records, bytes, launches) ->
      let s = buffer g bytes in
      let records = rebase records (Rig.Buffer.address s) in
      Some (Records { image; records; launches; held = s :: held })

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
