(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

external cubin : unit -> string = "nx_cuda_support_cubin"
external kernels : unit -> string array = "nx_cuda_support_kernels"
external bind_symbols : nativeint array -> unit = "nx_cuda_support_bind"
external attribute : int -> int = "nx_cuda_support_attribute"
external entries : int array -> nativeint = "nx_cuda_support_entries"

external record_c : (int array * string) array -> string
  = "nx_cuda_support_record"

external floors : unit -> int * int * int = "nx_cuda_support_floors"

type bytes =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

external run_arg : nativeint -> string -> bytes = "nx_cuda_support_run"
external seq_arg : bytes array -> int array -> bytes = "nx_cuda_support_seq"
external seq_fill : unit -> nativeint = "nx_cuda_support_seq_fill"
external lock : string -> string -> int = "nx_cuda_support_lock"

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
  if
    Sys.getenv_opt "RIG_GPU_LOCK_HELD" = None
    && Sys.file_exists "/dev/nvidiactl"
  then take 0

(* The GPU *)

type image = { table : nativeint; names : string array; loaded : Rig.Image.t }

type gpu = {
  device : Rig.t;
  cuda : Rig_cuda.t;
  mutable harness : image option;
  (* Host memory the device maps, for the hold's flag and late words and two
     stamps: host words 0 to 3. *)
  page : (int64, Bigarray.int64_elt, Bigarray.c_layout) Bigarray.Array1.t;
  mapped : Rig.Buffer.t;
}

let arch g = Rig_cuda.arch g.cuda

(* cuda.h's CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT and
   CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN. *)
let sms _ = attribute 16
let shared_max () = attribute 97

let bind g =
  let symbol = (Rig_cuda.capability g).symbol in
  bind_symbols
    (Array.map
       (fun n -> Option.get (symbol n))
       [| "cuLaunchKernel"; "cuDeviceGetAttribute" |])

let opened = ref None

let gpu () =
  match !opened with
  | Some g -> g
  | None ->
      if Rig_cuda.count () = 0 then Windtrap.skip ~reason:"CUDA sees no GPU" ();
      hold_gpu ();
      let cuda = ref None in
      let make () =
        Result.map
          (fun g ->
            cuda := Some g;
            g)
          (Rig_cuda.open_ 0)
      in
      let device =
        Result.get_ok (Rig.open_ (module Rig_cuda) ~name:"CUDA:nx2" make)
      in
      let cuda = Option.get !cuda in
      bind cuda;
      let host = Rig.Buffer.create Rig.host 65536 in
      let mapped = Option.get (Rig.Buffer.borrow device host) in
      let page = Rig.Buffer.bigarray Bigarray.int64 host in
      let g = { device; cuda; harness = None; page; mapped } in
      opened := Some g;
      g

(* Images *)

let image g bin names =
  match Rig.Image.load g.device bin with
  | Error why -> failwith why
  | Ok loaded ->
      let func n =
        match Rig.Image.entry loaded n with
        | Some f -> f
        | None -> failwith (strf "the cubin has no kernel %s" n)
      in
      { table = entries (Array.map func names); names; loaded }

let harness g =
  match g.harness with
  | Some h -> h
  | None ->
      let h = image g (cubin ()) (kernels ()) in
      g.harness <- Some h;
      h

let kernel i name =
  match Array.find_index (String.equal name) i.names with
  | Some k -> k
  | None -> invalid_arg ("Nx_cuda_support.record: no kernel " ^ name)

(* Records *)

type param = A of Rig.Buffer.t | W of int | D of int * int

(* A run of records holds the nx_cuda_run a fill reads, [launches] records long,
   and keeps its image and the buffers it addresses alive while it is. *)
type run =
  | Records of {
      image : image;
      arg : bytes;
      launches : int;
      held : Rig.Buffer.t list;
    }
  | Copy of { src : Rig.Buffer.t; dst : Rig.Buffer.t }

let words ps =
  let b = Buffer.create 64 in
  let addrs = ref 0 and words = ref false in
  List.iter
    (function
      | A x ->
          if !words then invalid_arg "Nx_cuda_support.record: address last";
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

let launch name ~grid:(gx, gy, gz) ~block ?(shared = 0) ps image =
  let addrs, words = words ps in
  ( ([| kernel image name; gx; gy; gz; block; shared; addrs; 0 |], words),
    List.filter_map (function A b -> Some b | _ -> None) ps )

let record image ls =
  let rs, held = List.split (List.map (fun l -> l image) ls) in
  let records = record_c (Array.of_list rs) in
  let arg = run_arg image.table records in
  Records { image; arg; launches = List.length ls; held = List.concat held }

let driver_copy ~src ~dst = Copy { src; dst }

(* Runs *)

let part queue work = { Rig.Submission.queue; after = [||]; work }

(* A part on [queue] that fills the record runs [rs], each repeated as [ns]
   says. *)
let fill queue rs ns =
  part queue
    (Fill
       {
         fill = seq_fill ();
         arg = Rig.Buffer.of_bigarray (seq_arg (Array.of_list rs) ns);
         ring_units = 0;
         segment_bytes = 0;
       })

let records = function
  | Records r -> r.arg
  | Copy _ -> invalid_arg "Nx_cuda_support: a driver copy has no records"

(* [r]'s queue, and the parts that run it [n] times there. *)
let queue = function Records _ -> "COMPUTE:0" | Copy _ -> "COPY:0"

let body r n =
  match r with
  | Records _ -> [ fill (queue r) [ records r ] [| n |] ]
  | Copy { src; dst } ->
      List.init n (fun _ -> part (queue r) (Copy { src; dst }))

let submit g parts =
  let s =
    Rig.Submission.make ~reads:0 ~writes:0 g.device (Array.of_list parts)
  in
  Rig.Point.value (Rig.submit s ~reads:[||] ~writes:[||] ~waits:[||])

(* The images and buffers a run's work uses stay alive until it is done
   (Rig.Image.entry). *)
let keep = function
  | Records r -> ignore (Sys.opaque_identity (r.image.loaded, r.arg, r.held))
  | Copy { src; dst } -> ignore (Sys.opaque_identity (src, dst))

(* Host words the device maps: the hold's flag (0) and late word (1), and two
   stamps (2, 3). *)
let word g i = Rig.Buffer.view g.mapped ~first:(8 * i) ~length:8

(* A hold gives up after 2 s: a round whose launches outgrow the stream blocks
   its submit until then, and a hog's blocks start well within it. *)
let hold_ns = 2_000_000_000

let delay g flag ~want =
  record (harness g)
    [
      launch "delay" ~grid:(1, 1, 1) ~block:1
        [ A flag; A (word g 1); D (want, 0); W hold_ns ];
    ]

type hog = {
  blocks : int;
  started : Rig.Buffer.t;
  sm : Rig.Buffer.t;
  run : run;
}

let buffer g n = Rig.Buffer.create g.device n

let hog g ~ns =
  let blocks = Int.max 1 (sms g / 2) in
  let started = buffer g 4 and sm = buffer g (4 * blocks) in
  let run =
    record (harness g)
      [
        launch "hog" ~grid:(blocks, 1, 1) ~block:1024 ~shared:(shared_max ())
          [ A started; A sm; W ns ];
      ]
  in
  { blocks; started; sm; run }

let read b =
  let h =
    Bigarray.Array1.create Bigarray.char Bigarray.c_layout (Rig.Buffer.length b)
  in
  Rig.Buffer.copy ~src:b ~dst:(Rig.Buffer.of_bigarray h);
  String.init (Bigarray.Array1.dim h) (Bigarray.Array1.get h)

let write b s = Rig.Buffer.copy ~src:(Rig.Buffer.of_string s) ~dst:b

let held_sms h =
  let s = read h.sm in
  List.init h.blocks (fun i -> Int32.to_int (String.get_int32_le s (4 * i)))

(* Beside a hog, the work waits on its queue until every hog block holds its SM:
   rig orders nothing between the two queues. *)
let run ?beside g r =
  let parts, kept =
    match beside with
    | None -> (body r 1, [])
    | Some _ when queue r = "COPY:0" ->
        invalid_arg "Nx_cuda_support.run: a driver copy beside a hog"
    | Some h ->
        write h.started "\000\000\000\000";
        g.page.{1} <- 0L;
        let wait = delay g h.started ~want:h.blocks in
        ( [
            fill "COMPUTE:0" [ records wait; records r ] [| 1; 1 |];
            fill "COPY:0" [ records h.run ] [| 1 |];
          ],
          [ wait; h.run ] )
  in
  Rig.wait g.device (submit g parts);
  List.iter keep (r :: kept);
  ignore (Sys.opaque_identity parts);
  if Option.is_some beside && g.page.{1} <> 0L then
    failwith "the hog's blocks did not all start within the hold"

(* kimchi's driver (615) holds 1,023 launches queued behind a kernel that runs;
   a round takes half. *)
let round = 512

let device_time g r ~count =
  let stamp i =
    record (harness g)
      [ launch "stamp" ~grid:(1, 1, 1) ~block:1 [ A (word g i) ] ]
  in
  let hold = delay g (word g 0) ~want:1 in
  let t0 = stamp 2 and t1 = stamp 3 and q = queue r in
  let launches = match r with Records x -> x.launches | Copy _ -> 1 in
  let per_round = Int.max 1 (round / Int.max 1 launches) in
  let rec go left span =
    if left = 0 then span
    else begin
      let n = Int.min left per_round in
      g.page.{0} <- 0L;
      g.page.{1} <- 0L;
      let parts =
        (fill q [ records hold; records t0 ] [| 1; 1 |] :: body r n)
        @ [ fill q [ records t1 ] [| 1 |] ]
      in
      let v = submit g parts in
      g.page.{0} <- 1L;
      Rig.wait g.device v;
      if g.page.{1} <> 0L then
        failwith (strf "%d runs of %d launches outgrew the stream" n launches);
      List.iter keep [ r; hold; t0; t1 ];
      ignore (Sys.opaque_identity parts);
      go (left - n) (span + Int64.to_int (Int64.sub g.page.{3} g.page.{2}))
    end
  in
  float (go count 0) *. 1e-9 /. float count

(* A million cycles: some 360 us at 2.8 GHz. *)
let sm_clock g =
  let ns = buffer g 8 and cycles = 1_000_000 in
  run g
    (record (harness g)
       [ launch "sm_clock" ~grid:(1, 1, 1) ~block:1 [ A ns; W cycles ] ]);
  float cycles *. 1e3 /. Int64.to_float (String.get_int64_le (read ns) 0)

(* Floors *)

let copy_threads, read_threads, read_vecs = floors ()

let vectors what b =
  let n = Rig.Buffer.length b in
  if n mod 16 <> 0 then
    invalid_arg
      (strf "Nx_cuda_support.%s: %d bytes are no whole vectors" what n);
  n / 16

let blocks n per = Int.max 1 ((n + per - 1) / per)

let floor_copy ~ins ~out =
  if ins = [] || List.length ins > 3 then
    invalid_arg "Nx_cuda_support.floor_copy: one to three inputs";
  let vecs = vectors "floor_copy" out in
  let pad l d = l @ List.init (3 - List.length l) (fun _ -> d) in
  let addrs = pad (List.map (fun b -> A b) ins) (A out) in
  let lens = pad (List.map (fun b -> W (vectors "floor_copy" b)) ins) (W 0) in
  launch "floor_copy"
    ~grid:(blocks vecs copy_threads, 1, 1)
    ~block:copy_threads
    (addrs @ [ A out ] @ lens @ [ W vecs ])

let floor_read g b =
  let vecs = vectors "floor_read" b in
  let per = read_threads * read_vecs in
  let out = buffer g (4 * blocks vecs per) in
  launch "floor_read"
    ~grid:(blocks vecs per, 1, 1)
    ~block:read_threads [ A b; A out; W vecs ]

(* Memory *)

type draw = Uniform | Wide of int | Small

let generate (type v s) g b (dt : (v, s) Nx_array.Dtype.t) draw ~seed =
  (match dt with
  | Float4_e2m1fn | Int4 | Uint4 | Complex128 | Complex64 | Bit ->
      invalid_arg
        (strf "Nx_cuda_support.generate: %s is not drawn"
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
           ~grid:(Int.min 4096 (blocks n 256), 1, 1)
           ~block:256
           [ A b; W n; W seed; D (Nx_array.Dtype.code dt, draw); D (spread, 0) ];
       ])
