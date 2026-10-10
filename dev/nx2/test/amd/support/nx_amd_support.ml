(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Abi = Rig_amd_abi
module A = Nx_array
module Spec = Nx_kernel.Spec

let strf = Printf.sprintf

external code_object : unit -> string = "nx_amd_support_code_object"
external kernels : unit -> string array = "nx_amd_support_kernels"
external sizes : unit -> int * int * int * int * int = "nx_amd_support_sizes"

let threads, copy_threads, read_threads, read_vecs, hog_threads = sizes ()

(* The GPU lock *)

let hold_gpu () = if Rig_amd_amdgpu.count () > 0 then Rig_gpu_lock.hold ()

(* Devices and images *)

let ok = function Ok x -> x | Error why -> failwith why

type image = { names : string array; loaded : Rig.Image.t }

(* A device of GPU 0, its capability, the harness loaded on it, and the two
   64-bit stamps device_time reads, in memory the host reads without a copy. *)
type device = {
  rig : Rig.t;
  cap : Abi.Capability.t;
  harness : image;
  stamps : Rig.Buffer.t;
}

type gpu = { work : device; mutable beside : device option }

let open_device name =
  let rig =
    ok (Rig.open_ (module Rig_amd) ~name (fun () -> Rig_amd_amdgpu.open_ 0))
  in
  let cap = Option.get (Rig.capability rig Abi.Capability.key) in
  let harness =
    { names = kernels (); loaded = ok (Rig.Image.load rig (code_object ())) }
  in
  let stamps = Rig.Buffer.create ~memory:Rig.Buffer.Pinned rig 16 in
  { rig; cap; harness; stamps }

let opened = ref None

let gpu () =
  match !opened with
  | Some g -> g
  | None ->
      if Rig_amd_amdgpu.count () = 0 then
        Windtrap.skip ~reason:"the machine has no AMD GPU" ();
      hold_gpu ();
      let work = open_device "AMD:nx2" in
      if not (Nx_amd.computes_on work.rig) then
        failwith (strf "nx.amd does not compute on %s" (Rig.arch work.rig));
      let g = { work; beside = None } in
      opened := Some g;
      g

let device g = g.work.rig
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

(* Launches *)

type param = A of Rig.Buffer.t | W of int | D of int * int

type launch = {
  kernel : string;
  groups : int * int * int;
  threads : int;
  params : param list;
}

let launch kernel ~groups ~threads params =
  let rec addresses_first = function
    | A _ :: ps -> addresses_first ps
    | ps -> List.for_all (function A _ -> false | W _ | D _ -> true) ps
  in
  if not (addresses_first params) then
    invalid_arg "Nx_amd_support.launch: an address follows another parameter";
  { kernel; groups; threads; params }

type contract = { spec : Spec.contract Spec.t; dst : A.any; ops : A.any array }

(* Submissions *)

(* Work on a queue: a launch of an image's kernel, a driver copy, or PM4 or SDMA
   words. *)
type work =
  | Launch of image * launch
  | Copied of Rig.Buffer.t * Rig.Buffer.t
  | Words of string

(* A submission made once, its run's blocks written, submitted any number of
   times. *)
type prepared = {
  sub : Rig.Submission.t;
  blocks : Rig.Submission.Run.t;
  buffers : Rig.Buffer.t array;
}

type body =
  | Launches of { image : image; launches : launch list }
  | Copy of { src : Rig.Buffer.t; dst : Rig.Buffer.t }
  | Contract of contract

(* A run and the submissions of its repetitions, made on first use: a key names
   what surrounds the run and its count. *)
type run = { body : body; mutable prepared : (string * prepared) list }

let record image launches =
  List.iter
    (fun l ->
      if not (Array.mem l.kernel image.names) then
        invalid_arg ("Nx_amd_support.record: no kernel " ^ l.kernel))
    launches;
  { body = Launches { image; launches }; prepared = [] }

let driver_copy ~src ~dst = { body = Copy { src; dst }; prepared = [] }

(* Makes the submission of [works] in order on [d], each on its queue, every
   buffer a launch addresses written: one slot an address. *)
let prepare d works =
  let slots = ref [] and count = ref 0 in
  let launch_work image l =
    let ref i = function
      | A b ->
          slots := b :: !slots;
          incr count;
          [ { Rig.Submission.at = 8 * i; slot = !count - 1 } ]
      | W _ | D _ -> []
    in
    Rig.Submission.Launch
      {
        image = image.loaded;
        kernel = l.kernel;
        params = 8 * List.length l.params;
        refs = Array.of_list (List.concat (List.mapi ref l.params));
      }
  in
  let part (q, w) =
    let work =
      match w with
      | Copied (src, dst) -> Rig.Submission.Copy { src; dst }
      | Words s -> Rig.Submission.Words (Rig.Buffer.of_string s)
      | Launch (image, l) -> launch_work image l
    in
    { Rig.Submission.queue = q; after = [||]; work }
  in
  let parts = Array.of_list (List.map part works) in
  let buffers = Array.of_list (List.rev !slots) in
  let access = Array.make (Array.length buffers) Rig.Buffer.Read_write in
  let sub = Rig.Submission.make ~access d.rig parts in
  let blocks = Rig.Submission.Run.make () in
  List.iteri
    (fun i (_, w) ->
      match w with
      | Copied _ | Words _ -> ()
      | Launch (_, l) ->
          let b = Rig.Submission.block sub i in
          let x, y, z = l.groups in
          Rig.Submission.Run.groups blocks b x y z;
          Rig.Submission.Run.threads blocks b l.threads 1 1;
          Rig.Submission.Run.shared blocks b 0;
          List.iteri
            (fun k -> function
              | A _ -> Rig.Submission.Run.int64 blocks b (8 * k) 0
              | W v -> Rig.Submission.Run.int64 blocks b (8 * k) v
              | D (u, v) ->
                  Rig.Submission.Run.int32 blocks b (8 * k) u;
                  Rig.Submission.Run.int32 blocks b ((8 * k) + 4) v)
            l.params)
    works;
  { sub; blocks; buffers }

let submit p =
  Rig.Point.value
    (Rig.submit p.sub ~run:p.blocks ~buffers:p.buffers ~waits:[||])

let compute = "COMPUTE:0"

let works r =
  match r.body with
  | Launches x -> List.map (fun l -> (compute, Launch (x.image, l))) x.launches
  | Copy { src; dst } -> [ ("COPY:0", Copied (src, dst)) ]
  | Contract _ -> invalid_arg "Nx_amd_support: a contraction among launches"

(* [r]'s submission on [d] under [key], made by [works] on first use. *)
let prepared d r key works =
  match List.assoc_opt key r.prepared with
  | Some p -> p
  | None ->
      let p = prepare d (works ()) in
      r.prepared <- (key, p) :: r.prepared;
      p

let repeat n xs = List.concat (List.init n (fun _ -> xs))

(* Calls nx.amd's contraction. *)
let call_contract c =
  match Nx_amd.contract c.spec ~dst:c.dst c.ops with
  | A.Done -> ()
  | A.Declined ->
      failwith "Nx_amd_support: nx.amd declines a contraction it computed"
  | r -> A.refused "Nx_amd.contract" r (c.dst :: Array.to_list c.ops)

let call r =
  match r.body with
  | Contract c -> call_contract c
  | Launches _ | Copy _ -> invalid_arg "Nx_amd_support.call: no contraction"

(* Submits [r] once on the work device, a contraction as a call of nx.amd's, and
   is the value to wait for. *)
let once g r =
  match r.body with
  | Contract c ->
      call_contract c;
      Rig.submitted g.work.rig
  | Launches _ | Copy _ -> submit (prepared g.work r "run" (fun () -> works r))

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
   releases it. rig orders one device's submissions but never two devices', and
   two devices of one GPU run on its processors at once: a result computed
   beside a hog is a schedule that happens, as beside another process. Its words
   are the work device's memory, which the GPU's devices share: [started] counts
   the hog's workgroups that hold their processors, [release] lets them go, and
   [late] and [let_go] say that the work's queue or the hog gave up waiting for
   the other. *)
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

(* Each side gives up on the other after 2 s: a hog's workgroups start, and the
   work beside it ends, well within it. *)
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
   devices' queues, and orders the work device's submissions as made. *)
let run ?beside g r =
  match beside with
  | None -> Rig.wait g.work.rig (once g r)
  | Some h ->
      (match r.body with
      | Copy _ -> invalid_arg "Nx_amd_support.run: a driver copy beside a hog"
      | Launches _ | Contract _ -> ());
      List.iter zero [ h.started; h.release; h.late; h.let_go ];
      let held =
        submit (prepared h.device h.hold "run" (fun () -> works h.hold))
      in
      ignore (once g h.wait);
      ignore (once g r);
      let v = once g h.free in
      Rig.wait g.work.rig v;
      Rig.wait h.device.rig held;
      let set b = read b <> "\000\000\000\000" in
      if set h.late then
        failwith "the hog's workgroups did not all start within the hold";
      if set h.let_go then failwith "the hog let go before the work was done"

(* rig.amd refuses a submission of more than 512 parts, or one that takes more
   than half its argument segment, which 512 launches of at most 256 bytes of
   parameters each stay under. A timed round's two stamps are parts too, so a
   round of launches holds 510; one of driver copies holds 256. A round of
   contractions is 256 calls, each a submission of its own. *)
let parts = 510
let copies = 256
let calls = 256

(* The runs of [r] a submission holds. *)
let per_round r =
  match r.body with
  | Launches x -> Int.max 1 (parts / List.length x.launches)
  | Copy _ -> copies
  | Contract _ -> calls

let enqueue g ~count r =
  let rec go left =
    if left > 0 then begin
      let n = Int.min left (per_round r) in
      (match r.body with
      | Contract c ->
          for _ = 1 to n do
            call_contract c
          done
      | Launches _ | Copy _ ->
          let key = strf "enqueue %d" n in
          ignore (submit (prepared g.work r key (fun () -> repeat n (works r)))));
      go (left - n)
    end
  in
  go count

let issue g ~count r =
  match r.body with
  | Launches _ | Copy _ -> invalid_arg "Nx_amd_support.issue: no contraction"
  | Contract c ->
      for _ = 1 to count do
        call_contract c
      done;
      Rig.wait g.work.rig (Rig.submitted g.work.rig)

(* A stamp is the GPU's clock, which [queue] writes once the work before it is
   done, into the device's stamps: made once, so that a call allocates no GPU
   memory, and read by the host without a copy. A run of launches and its stamps
   are one submission, which rig.amd's queues read only once placed whole, so
   the host never starves the GPU; a contraction's calls are submissions of
   their own between two stamps. *)
let device_time g r ~count =
  let at = g.work.stamps in
  let slot i = Rig.Buffer.view at ~first:(8 * i) ~length:8 in
  let queue, clock =
    match r.body with
    | Launches _ | Contract _ -> (compute, Abi.Pm4.copy_data Posted Clock)
    | Copy _ -> ("COPY:0", Abi.Sdma.timestamp)
  in
  let stamp i =
    ( queue,
      Words
        (Abi.Packet.encode Int64.of_int (clock (Rig.Buffer.address (slot i))))
    )
  in
  let rec go left span =
    if left = 0 then span
    else begin
      let n = Int.min left (per_round r) in
      let v =
        match r.body with
        | Contract c ->
            ignore
              (submit (prepared g.work r "stamp 0" (fun () -> [ stamp 0 ])));
            for _ = 1 to n do
              call_contract c
            done;
            submit (prepared g.work r "stamp 1" (fun () -> [ stamp 1 ]))
        | Launches _ | Copy _ ->
            let works () = (stamp 0 :: repeat n (works r)) @ [ stamp 1 ] in
            submit (prepared g.work r (strf "timed %d" n) works)
      in
      Rig.wait g.work.rig v;
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

let dtype code =
  List.find (fun (A.Dtype.Any d) -> A.Dtype.code d = code) A.Dtype.all

(* [o] as an array over its buffer, as nx hands the kernels their operands. *)
let array (o : operand) =
  let (A.Dtype.Any d) = dtype o.dtype in
  let bytes = Int.max 1 (A.Dtype.bits d / 8) in
  let layout =
    A.Layout.v ~offset:(o.first / bytes) ~strides:o.strides o.shape
  in
  A.Any (A.v d layout o.buffer)

let contract (_ : gpu) ~a ~b ?init ~y ~batch ~contracting ~acc () =
  let spec =
    Spec.contract ~batch:(Array.of_list batch)
      ~contracting:(Array.of_list contracting)
      ~acc:(dtype acc) ~out:(dtype y.dtype) ~init:(Option.is_some init)
  in
  let c =
    {
      spec;
      dst = array y;
      ops = Array.of_list (List.map array (a :: b :: Option.to_list init));
    }
  in
  match Nx_amd.contract c.spec ~dst:c.dst c.ops with
  | A.Done -> Some { body = Contract c; prepared = [] }
  | A.Declined -> None
  | r -> A.refused "Nx_amd.contract" r (c.dst :: Array.to_list c.ops)

(* Memory *)

type draw = Uniform | Wide of int | Small

let generate (type v s) g b (dt : (v, s) A.Dtype.t) draw ~seed =
  (match dt with
  | Float4_e2m1fn | Int4 | Uint4 | Complex128 | Complex64 | Bit ->
      invalid_arg
        (strf "Nx_amd_support.generate: %s is not drawn" (A.Dtype.name dt))
  | _ -> ());
  let n = Rig.Buffer.length b * 8 / A.Dtype.bits dt in
  let draw, spread =
    match draw with Uniform -> (0, 0) | Wide e -> (1, e) | Small -> (2, 0)
  in
  run g
    (record (harness g)
       [
         launch "generate"
           ~groups:(blocks n threads, 1, 1)
           ~threads
           [ A b; W n; W seed; D (A.Dtype.code dt, draw); D (spread, 0) ];
       ])
