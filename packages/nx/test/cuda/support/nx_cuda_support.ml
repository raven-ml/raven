(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

external cubin : unit -> string = "nx_cuda_support_cubin"
external kernels : unit -> string array = "nx_cuda_support_kernels"
external bind_symbol : nativeint -> unit = "nx_cuda_support_bind"
external attribute : int -> int = "nx_cuda_support_attribute"
external floors : unit -> int * int * int = "nx_cuda_support_floors"

module A = Nx_array

(* The GPU *)

type image = { names : string array; loaded : Rig.Image.t }

type gpu = {
  device : Rig.t;
  arch : string;
  sms : int;
  harness : image Lazy.t;
  (* Host memory the device maps, for the hold's flag and late words and two
     stamps: host words 0 to 3. *)
  page : (int64, Bigarray.int64_elt, Bigarray.c_layout) Bigarray.Array1.t;
  mapped : Rig.Buffer.t;
  empty : Rig.Submission.Run.t;
      (* A run of an empty submission, which drains its work. *)
}

let sms g = g.sms
let arch g = g.arch

let image device bin names =
  match Rig.Image.load device bin with
  | Error why -> failwith why
  | Ok loaded -> { names; loaded }

let opened = ref None

(* GPU 0 through CUDA's library, and its count of multiprocessors:
   cuda.h's CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT. *)
let cuda () =
  (match Rig_cuda.count () with
  | Ok n when n > 0 -> ()
  | Ok _ -> Windtrap.skip ~reason:"CUDA sees no GPU" ()
  | Error why -> Windtrap.skip ~reason:("CUDA: " ^ why) ());
  let device =
    Result.get_ok (Rig.open_ { (Rig_cuda.gpu 0) with name = "CUDA:nx" })
  in
  let capability = Option.get (Rig.capability device Rig_cuda_abi.key) in
  bind_symbol (Option.get (capability.symbol "cuDeviceGetAttribute"));
  (device, attribute 16)

(* GPU 0 through NVIDIA's kernel driver and rig.nv, and its count of
   multiprocessors as the GPU's description states them. *)
let nv () =
  (match Rig_nv_nvidia.count () with
  | Ok n when n > 0 -> ()
  | Ok _ | Error _ ->
      Windtrap.skip ~reason:"NVIDIA's kernel driver sees no GPU" ());
  let device =
    Result.get_ok (Rig.open_ { (Rig_nv_nvidia.gpu 0) with name = "NV:nx" })
  in
  let g = Option.get (Rig.capability device Rig_nv_abi.key) in
  (device, g.gpcs * g.tpcs_per_gpc * g.sms_per_tpc)

let through_nv () = Sys.getenv_opt "NX_CUDA_RIG" = Some "nv"

let gpu () =
  match !opened with
  | Some g -> g
  | None ->
      let device, sms =
        match Sys.getenv_opt "NX_CUDA_RIG" with
        | Some "nv" -> nv ()
        | None | Some "cuda" -> cuda ()
        | Some other -> invalid_arg ("NX_CUDA_RIG: " ^ other)
      in
      let arch = Rig.arch device in
      if not (Nx_cuda.computes_on device) then
        failwith (strf "nx.cuda does not compute on %s" arch);
      let host = Rig.Buffer.create Rig.host 65536 in
      let mapped = Option.get (Rig.Buffer.borrow device host) in
      let page = Rig.Buffer.export host Bigarray.int64 in
      let harness = lazy (image device (cubin ()) (kernels ())) in
      let empty = Rig.Submission.(take (make device [||])) in
      let g = { device; arch; sms; harness; page; mapped; empty } in
      opened := Some g;
      g

let device g = g.device
let harness g = Lazy.force g.harness

(* Launches *)

type param = A of Rig.Buffer.t | W of int | D of int * int

type launch = {
  kernel : string;
  grid : int * int * int;
  block : int;
  shared : int;
  params : param list;
}

let launch kernel ~grid ~block ?(shared = 0) params =
  let rec addresses_first = function
    | A _ :: ps -> addresses_first ps
    | ps -> List.for_all (function A _ -> false | W _ | D _ -> true) ps
  in
  if not (addresses_first params) then
    invalid_arg "Nx_cuda_support.launch: an address follows another parameter";
  { kernel; grid; block; shared; params }

(* A call of one of nx.cuda's kernels, what it is called as, the arrays it
   names, and the most launches it submits. *)
type call = {
  call : unit -> A.answer;
  what : string;
  arrays : A.any list;
  launches : int;
}

(* Submissions *)

(* Work on a queue: a hold until the next {!release}, a launch of an image's
   kernel, or a driver copy. *)
type work =
  | Hold
  | Launch of image * launch
  | Copied of Rig.Buffer.t * Rig.Buffer.t

(* A submission made once, its run's blocks written, submitted any number of
   times; [held] if its first part is a hold, whose [want] each submit
   stores. The test's one thread is its only taker, so each take returns
   the run its blocks were written into. *)
type prepared = {
  sub : Rig.Submission.t;
  buffers : Rig.Buffer.t array;
  held : bool;
}

type body =
  | Launches of { image : image; launches : launch list }
  | Copy of { src : Rig.Buffer.t; dst : Rig.Buffer.t }
  | Call of call

(* A run and the submissions of its repetitions, made on first use: a key names
   what surrounds the run and its count. *)
type run = { body : body; mutable prepared : (string * prepared) list }

let record image launches =
  List.iter
    (fun l ->
      if not (Array.mem l.kernel image.names) then
        invalid_arg ("Nx_cuda_support.record: no kernel " ^ l.kernel))
    launches;
  { body = Launches { image; launches }; prepared = [] }

let driver_copy ~src ~dst = { body = Copy { src; dst }; prepared = [] }

(* Host words the device maps: the count of releases (0), the late word (1), and
   two stamps (2, 3). *)
let word g i = Rig.Buffer.view g.mapped ~first:(8 * i) ~length:8

(* A hold gives up after 2 s: a round whose launches outgrow the stream blocks
   its submit until then. *)
let hold_ns = 2_000_000_000

(* The hold's [want], at byte 16 of the delay's parameters. *)
let want_at = 16

let delay g =
  launch "delay" ~grid:(1, 1, 1) ~block:1
    [ A (word g 0); A (word g 1); D (0, 0); W hold_ns ]

(* Makes the submission of [works] in order, each on its queue, every buffer a
   launch addresses written: one slot an address. *)
let prepare g works =
  let slots = ref [] and count = ref 0 in
  let launch_work image l =
    let ref i = function
      | A b ->
          slots := b :: !slots;
          incr count;
          [ Rig.Submission.write (!count - 1) ~at:(8 * i) ]
      | W _ | D _ -> []
    in
    let refs = Array.of_list (List.concat (List.mapi ref l.params)) in
    let f =
      match Rig.Image.resolve image.loaded l.kernel with
      | Some f -> f
      | None -> invalid_arg ("Nx_cuda_support: no kernel " ^ l.kernel)
    in
    Rig.Submission.launch ~refs f ~params:(8 * List.length l.params)
  in
  let part (_, w) =
    let work =
      match w with
      | Copied (src, dst) ->
          Rig.Submission.copy ~src:(Buffer src) ~dst:(Buffer dst)
      | Hold -> launch_work (harness g) (delay g)
      | Launch (image, l) -> launch_work image l
    in
    work
  in
  let parts = Array.of_list (List.map part works) in
  let buffers = Array.of_list (List.rev !slots) in
  let sub = Rig.Submission.make g.device parts in
  let blocks = Rig.Submission.take sub in
  let block i l =
    let b = Rig.Submission.block sub i in
    let x, y, z = l.grid in
    Rig.Submission.Run.groups blocks b x y z;
    Rig.Submission.Run.threads blocks b l.block 1 1;
    Rig.Submission.Run.shared blocks b l.shared;
    List.iteri
      (fun k -> function
        | A _ -> Rig.Submission.Run.int64 blocks b (8 * k) 0L
        | W v -> Rig.Submission.Run.int64 blocks b (8 * k) (Int64.of_int v)
        | D (u, v) ->
            Rig.Submission.Run.int32 blocks b (8 * k) u;
            Rig.Submission.Run.int32 blocks b ((8 * k) + 4) v)
      l.params
  in
  List.iteri
    (fun i (_, w) ->
      match w with
      | Copied _ -> ()
      | Hold -> block i (delay g)
      | Launch (_, l) -> block i l)
    works;
  Rig.Submission.Run.release blocks;
  let held = match works with (_, Hold) :: _ -> true | _ -> false in
  { sub; buffers; held }

(* The point after [g]'s work: an empty submission follows every earlier
   value. Its run holds no bytes, so submits share it. *)
let drain g = Rig.submit g.empty ~buffers:[||]

(* Submits [p], its hold waiting for the next {!release}, and is its point. The
   count only grows, so a hold an earlier submission queued, which the GPU has
   not run yet, stays released whatever later submissions do. *)
let submit g p =
  let blocks = Rig.Submission.take p.sub in
  if p.held then
    Rig.Submission.Run.int32 blocks
      (Rig.Submission.block p.sub 0)
      want_at
      (Int64.to_int g.page.{0} + 1);
  Rig.submit blocks ~buffers:p.buffers

let release g = g.page.{0} <- Int64.succ g.page.{0}

(* Waits for [v] and fails with [late] if a hold timed out. *)
let finish g v ~late =
  Rig.Point.wait v;
  if g.page.{1} <> 0L then failwith late

let compute = "COMPUTE:0"

let works r =
  match r.body with
  | Launches x -> List.map (fun l -> (compute, Launch (x.image, l))) x.launches
  | Copy { src; dst } -> [ ("COPY:0", Copied (src, dst)) ]
  | Call _ -> invalid_arg "Nx_cuda_support: a kernel's call among launches"

(* [r]'s submission under [key], made by [works] on first use. *)
let prepared g r key works =
  match List.assoc_opt key r.prepared with
  | Some p -> p
  | None ->
      let p = prepare g (works ()) in
      r.prepared <- (key, p) :: r.prepared;
      p

(* The submissions of no run: holds and stamps, made once. *)
let fixed = Hashtbl.create 4

let fixed_prepared g key works =
  match Hashtbl.find_opt fixed key with
  | Some p -> p
  | None ->
      let p = prepare g works in
      Hashtbl.replace fixed key p;
      p

let stamp g i =
  ( compute,
    Launch (harness g, launch "stamp" ~grid:(1, 1, 1) ~block:1 [ A (word g i) ])
  )

(* Calls [c]'s kernel. *)
let call_kernel c =
  match c.call () with
  | A.Done -> ()
  | A.Declined ->
      failwith
        ("Nx_cuda_support: nx.cuda declines a call it computed: " ^ c.what)
  | r -> A.refused c.what r c.arrays

let repeat n xs = List.concat (List.init n (fun _ -> xs))

let run g r =
  g.page.{1} <- 0L;
  match r.body with
  | Call c ->
      call_kernel c;
      finish g (drain g) ~late:""
  | Launches _ | Copy _ ->
      finish g (submit g (prepared g r "run" (fun () -> works r))) ~late:""

let issue g ~count r =
  match r.body with
  | Launches _ | Copy _ -> invalid_arg "Nx_cuda_support.issue: no call"
  | Call c ->
      g.page.{1} <- 0L;
      ignore (submit g (fixed_prepared g "hold" [ (compute, Hold) ]));
      for _ = 1 to count do
        call_kernel c
      done;
      release g;
      finish g (drain g) ~late:"the hold ran out"

let call r =
  match r.body with
  | Call c -> call_kernel c
  | Launches _ | Copy _ -> invalid_arg "Nx_cuda_support.call: no call"

(* NVIDIA's driver 615 holds 1,023 launches queued behind a kernel that
   runs; a round takes half. *)
let round = 512

let launches r =
  match r.body with
  | Launches x -> List.length x.launches
  | Copy _ -> 1
  | Call c -> c.launches

let busy g r ~count =
  let per_round = Int.max 1 (round / launches r) in
  let rec go left last =
    if left = 0 then Option.iter (fun v -> finish g v ~late:"") last
    else
      let n = Int.min left per_round in
      g.page.{1} <- 0L;
      let v =
        match r.body with
        | Call c ->
            for _ = 1 to n do
              call_kernel c
            done;
            drain g
        | Launches _ | Copy _ ->
            let works () = repeat n (works r) in
            submit g (prepared g r (strf "busy %d" n) works)
      in
      go (left - n) (Some v)
  in
  go count None

let device_time g r ~count =
  let launches = launches r in
  let per_round = Int.max 1 (round / launches) in
  let late n =
    strf "%d runs of %d launches outgrew the stream or outlasted the hold" n
      launches
  in
  let rec go left span =
    if left = 0 then span
    else begin
      let n = Int.min left per_round in
      g.page.{1} <- 0L;
      let v =
        match r.body with
        | Call c ->
            let head = [ (compute, Hold); stamp g 2 ] in
            ignore (submit g (fixed_prepared g "timed" head));
            for _ = 1 to n do
              call_kernel c
            done;
            submit g (fixed_prepared g "stamp" [ stamp g 3 ])
        | Launches _ | Copy _ ->
            let works () =
              ((compute, Hold) :: stamp g 2 :: repeat n (works r))
              @ [ stamp g 3 ]
            in
            submit g (prepared g r (strf "timed %d" n) works)
      in
      release g;
      finish g v ~late:(late n);
      go (left - n) (span + Int64.to_int (Int64.sub g.page.{3} g.page.{2}))
    end
  in
  float (go count 0) *. 1e-9 /. float count

let buffer g n = Rig.Buffer.create g.device n

(* One copy into the string's own bytes: a byte at a time costs seconds on
   a buffer of a GiB. *)
let read b =
  let n = Rig.Buffer.length b in
  let s = Bytes.create n in
  Rig.Buffer.blit_to_bytes b 0 s 0 n;
  Bytes.unsafe_to_string s

let write b s = Rig.Buffer.copy ~src:(Rig.Buffer.of_string s) ~dst:b

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
  if n mod 16 = 0 then n / 16
  else invalid_arg (strf "Nx_cuda_support.%s: %d bytes, not vectors" what n)

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

(* Operands *)

type operand = {
  buffer : Rig.Buffer.t;
  dtype : int;
  shape : int array;
  strides : int array;
  first : int;
}

let dtype code = Option.get (A.Dtype.of_code code)

(* [o] as an array over its buffer, as nx hands the kernels their operands. *)
let array (o : operand) =
  let (A.Dtype.Any d) = dtype o.dtype in
  let bytes = Int.max 1 (A.Dtype.bits d / 8) in
  let layout =
    A.Layout.v ~offset:(o.first / bytes)
      ~strides:(Iarray.of_array o.strides)
      (Iarray.of_array o.shape)
  in
  A.Any (A.v d layout o.buffer)

(* [i] as an array of [y]'s dtype. *)
let same_dtype : type v s. (v, s) A.t -> A.any -> (v, s) A.t =
 fun y (A.Any i) ->
  match A.Dtype.equal_witness (A.dtype y) (A.dtype i) with
  | Some Equal -> i
  | None -> invalid_arg "Nx_cuda_support.contract: init is not y's dtype"

(* Calls *)

(* Makes [c] once, and is its run; [None] if nx.cuda declines it. *)
let first_call c =
  match c.call () with
  | A.Done -> Some { body = Call c; prepared = [] }
  | A.Declined -> None
  | r -> A.refused c.what r c.arrays

(* Contractions *)

(* The most launches a contraction's call submits within one launch's
   bounds. *)
let contract_launches = 3

let contract (_ : gpu) ~a ~b ?init ~y ~batch ~contracting ~acc () =
  let batch = Iarray.of_list batch
  and contracting = Iarray.of_list contracting in
  let acc = dtype acc in
  let (A.Any dst) = array y in
  let (A.Any x) = array a in
  let (A.Any w) = array b in
  let i = Option.map (fun i -> same_dtype dst (array i)) init in
  first_call
    {
      call =
        (fun () ->
          Nx_cuda.contract ~batch ~contracting ~segments:None ~acc ~dst ~init:i
            { array = A.Any x; scale = None }
            { array = A.Any w; scale = None });
      what = "Nx_cuda.contract";
      arrays =
        A.Any dst :: A.Any x :: A.Any w
        :: Option.to_list (Option.map (fun i -> A.Any i) i);
      launches = contract_launches;
    }

(* Reductions and scans *)

(* The program that gives its one operand, of [x]'s dtype, back. *)
let identity (A.Any x) =
  let dt = A.Dtype.Any (A.dtype x) in
  Nx_kernel.Prog.v ~ins:(Iarray.of_list [ dt ])
    (Iarray.of_list [ Nx_kernel.Prog.In 0 ])
    ~outs:(Iarray.of_list [ 0 ])

(* A loop's call in launches of a round's budget. It submits up to two (a
   first pass, then a tree or a rescan), but each carries 32 KiB of
   parameters: on fudge 233 calls a round outgrew the stream, the host
   blocked behind the hold, and the hold ran out. 16 calls a round fit. *)
let loop_launches = 32

let reduce (_ : gpu) m ~axes ~x ~y =
  let xs = array x and ys = array y in
  let (A.Any xa) = xs in
  let p = identity xs in
  let shape = A.Layout.shape (A.layout xa) in
  let axes = Iarray.of_list axes in
  first_call
    {
      call =
        (fun () ->
          Nx_cuda.reduce p ~shape ~axes
            [| (Nx_kernel.Monoid m, 0) |]
            ~dsts:[| ys |] [| Nx_kernel.Plain xa |]);
      what = "Nx_cuda.reduce";
      arrays = [ ys; xs ];
      launches = loop_launches;
    }

let scan (_ : gpu) m ~axis ~x ~y =
  let xs = array x and ys = array y in
  let (A.Any xa) = xs in
  let p = identity xs in
  first_call
    {
      call =
        (fun () ->
          Nx_cuda.scan p ~axis (Nx_kernel.Monoid m, 0) ~dsts:[| ys |]
            [| Nx_kernel.Plain xa |]);
      what = "Nx_cuda.scan";
      arrays = [ ys; xs ];
      launches = loop_launches;
    }

let kernel_call ~what ~launches arrays call =
  first_call { call; what; arrays; launches }

(* Memory *)

type draw = Uniform | Wide of int | Small

let generate (type v s) g b (dt : (v, s) A.Dtype.t) draw ~seed =
  let module D = A.Dtype in
  (match dt with
  | Float4_e2m1fn | Int4 | Uint4 | Complex128 | Complex64 | Bit ->
      invalid_arg (strf "Nx_cuda_support.generate: %s is not drawn" (D.name dt))
  | _ -> ());
  let bits = D.bits dt and code = D.code dt in
  let n = Rig.Buffer.length b * 8 / bits in
  let draw, spread =
    match draw with Uniform -> (0, 0) | Wide e -> (1, e) | Small -> (2, 0)
  in
  let lo = if D.is Signed dt then -8 else 0 in
  let narrow = Bool.to_int (D.is Float dt && bits < 32) in
  let ps =
    [ A b; W n; W seed; D (code, draw); D (spread, bits / 8); D (lo, narrow) ]
  in
  let grid = (Int.min 4096 (blocks n 256), 1, 1) in
  run g (record (harness g) [ launch "generate" ~grid ~block:256 ps ])
