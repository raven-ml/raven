(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Sub = Rig.Submission
module A = Nx_array

external harness_object : unit -> string = "nx_amd_support_harness"

let strf = Printf.sprintf

(* The GPU *)

let device () =
  match Nx_kernels_support.amd () with
  | Some b -> b.device
  | None -> Windtrap.skip ~reason:"no AMD GPU nx.amd computes on" ()

let harness =
  lazy
    (match Rig.Image.load (device ()) (harness_object ()) with
    | Ok image -> image
    | Error why -> failwith ("the harness does not load: " ^ why))

let func kernel =
  match Rig.Image.resolve (Lazy.force harness) kernel with
  | Some f -> f
  | None -> invalid_arg ("Nx_amd_support: no kernel " ^ kernel)

type gpu = {
  device : Rig.t;
  arch : string;
  units : int;
  clock_hz : int; (* The rate of the clock the harness stamps. *)
  (* Host memory the device maps, for the hold's count and late word, two stamps
     and two words that order stamps: host words 0 to 5. *)
  page : (int64, Bigarray.int64_elt, Bigarray.c_layout) Bigarray.Array1.t;
  mapped : Rig.Buffer.t;
  empty : Sub.Run.t; (* A run of an empty submission, which drains. *)
}

let opened = ref None

let gpu () =
  match !opened with
  | Some g -> g
  | None ->
      let device = device () in
      let c = Option.get (Rig.capability device Rig_amd_abi.key) in
      let host = Rig.Buffer.create Rig.host 65536 in
      let mapped =
        match Rig.Buffer.borrow device host with
        | Some m -> m
        | None -> failwith "Nx_amd_support.gpu: the GPU maps no host memory"
      in
      let g =
        {
          device;
          arch = Rig.arch device;
          units = c.gpu.xccs * c.gpu.compute_units;
          clock_hz = c.clock_hz;
          page = Rig.Buffer.export host Bigarray.int64;
          mapped;
          empty = Sub.(take (make device [||]));
        }
      in
      Bigarray.Array1.fill g.page 0L;
      opened := Some g;
      g

let arch g = g.arch
let compute_units g = g.units

(* Launches *)

type word = A of Rig.Buffer.t | W of int

type dispatch = {
  func : Rig.Image.func;
  groups : int * int * int;
  threads : int;
  words : word list;
}

let dispatch kernel ~groups ~threads words =
  { func = func kernel; groups; threads; words }

(* Work on the GPU: a dispatch, a driver's copy, or a barrier: a part that
   follows every earlier part, and that every later part on its queue follows
   (rule 2 of rig's order). *)
type work =
  | Launch of dispatch
  | Copied of Rig.Buffer.t * Rig.Buffer.t
  | Barrier

(* A barrier's words: none, which place nothing on the queue. *)
let no_words = lazy (Rig.Buffer.create Rig.host 0)

(* A submission made once, its run's blocks written, submitted any number of
   times; [held] if its first part is a hold, whose [want] each submit stores.
   The suite's one thread is its only taker, so each take returns the run its
   blocks were written into. *)
type prepared = { sub : Sub.t; buffers : Rig.Buffer.t array; held : bool }

(* Makes the submission of [works] in order, every buffer a dispatch addresses
   written: a slot an address. *)
let prepare ~held works =
  let slots = ref [] and count = ref 0 in
  let part = function
    | Copied (src, dst) -> Sub.copy ~src:(Buffer src) ~dst:(Buffer dst)
    | Barrier -> Sub.words (Lazy.force no_words)
    | Launch d ->
        let ref i = function
          | A b ->
              slots := b :: !slots;
              incr count;
              [ Sub.write (!count - 1) ~at:(8 * i) ]
          | W _ -> []
        in
        let refs = Array.of_list (List.concat (List.mapi ref d.words)) in
        Sub.launch ~refs d.func ~params:(8 * List.length d.words)
  in
  let parts = Array.of_list (List.map part works) in
  let buffers = Array.of_list (List.rev !slots) in
  let sub = Sub.make (device ()) parts in
  let r = Sub.take sub in
  let block i d =
    let at = Sub.block sub i in
    let x, y, z = d.groups in
    Sub.Run.groups r at x y z;
    Sub.Run.threads r at d.threads 1 1;
    Sub.Run.shared r at 0;
    List.iteri
      (fun k w ->
        let v = match w with A _ -> 0 | W v -> v in
        Sub.Run.int64 r at (8 * k) (Int64.of_int v))
      d.words
  in
  List.iteri
    (fun i -> function Copied _ | Barrier -> () | Launch d -> block i d)
    works;
  Sub.Run.release r;
  { sub; buffers; held }

(* The hold's [want], at byte 16 of the delay's words. *)
let want_at = 16

(* Submits [p], its hold, if it has one, waiting for [want]. *)
let start ?want p =
  let r = Sub.take p.sub in
  Option.iter
    (fun w -> Sub.Run.int64 r (Sub.block p.sub 0) want_at (Int64.of_int w))
    want;
  Rig.submit r ~buffers:p.buffers

let launch kernel ~groups ~threads words =
  let p =
    prepare ~held:false [ Launch (dispatch kernel ~groups ~threads words) ]
  in
  Rig.Point.wait (start p)

(* Holds and stamps *)

(* Host words the device maps: the count of releases (0), the late word (1), two
   stamps (2, 3) and two words that order stamps (4, 5). *)
let word g i = Rig.Buffer.view g.mapped ~first:(8 * i) ~length:8

(* A hold gives up after 2 s of the GPU's clock: a round whose calls block the
   host blocks until then. *)
let delay g =
  dispatch "delay" ~groups:(1, 1, 1) ~threads:32
    [ A (word g 0); A (word g 1); W 0; W (2 * g.clock_hz) ]

(* Stamp [i], ordered against the parts that write [order] (rule 2 of rig's
   order): it writes nothing there, but addresses it as written. *)
let stamp g i ~order =
  dispatch "stamp" ~groups:(1, 1, 1) ~threads:32 [ A (word g i); A order ]

(* The parts of a timed submission around its runs [ws]: the hold, then stamp 2
   once the hold returns, then the runs, then stamp 3 once they are done.
   Barriers order the stamps against every launch; a driver's copy runs on
   another queue, so stamp 2 addresses its destination, which the copy then
   follows. *)
let timed g ws =
  let first = match ws with Copied (_, dst) :: _ -> dst | _ -> word g 4 in
  [ Launch (delay g); Barrier; Launch (stamp g 2 ~order:first); Barrier ]
  @ ws
  @ [ Barrier; Launch (stamp g 3 ~order:(word g 5)) ]

(* Submits [p], its hold waiting for the next {!release}, and is its point. The
   count only grows, so a hold an earlier submission queued, which the GPU has
   not run yet, stays released whatever later submissions do. *)
let submit g p =
  if p.held then start p ~want:(Int64.to_int g.page.{0} + 1) else start p

let release g = g.page.{0} <- Int64.succ g.page.{0}

(* The point after [g]'s work: an empty submission follows every earlier value.
   Its run holds no bytes, so submits share it. *)
let drain g = Rig.submit g.empty ~buffers:[||]

(* Waits for [v] and fails with [late] if a hold ran out. *)
let finish g v ~late =
  Rig.Point.wait v;
  if g.page.{1} <> 0L then failwith late

(* Runs *)

(* A call of one of nx.amd's kernels, what it is called as, the arrays it names,
   and the most launches it submits. *)
type call = {
  call : unit -> A.answer;
  what : string;
  arrays : A.any list;
  launches : int;
}

type body =
  | Dispatches of dispatch list
  | Copy of { src : Rig.Buffer.t; dst : Rig.Buffer.t }
  | Call of call

(* A run and the submissions of its repetitions, made on first use: a key names
   what surrounds the run and its count. *)
type run = { body : body; mutable prepared : (string * prepared) list }

let record ds = { body = Dispatches ds; prepared = [] }
let driver_copy ~src ~dst = { body = Copy { src; dst }; prepared = [] }

let works r =
  match r.body with
  | Dispatches ds -> List.map (fun d -> Launch d) ds
  | Copy { src; dst } -> [ Copied (src, dst) ]
  | Call _ -> invalid_arg "Nx_amd_support: a kernel's call among launches"

(* [r]'s submission under [key], made of [works] on first use. *)
let prepared r key ~held works =
  match List.assoc_opt key r.prepared with
  | Some p -> p
  | None ->
      let p = prepare ~held (works ()) in
      r.prepared <- (key, p) :: r.prepared;
      p

(* The submissions of no run: holds and stamps, made once. *)
let fixed = Hashtbl.create 4

let fixed_prepared key ~held works =
  match Hashtbl.find_opt fixed key with
  | Some p -> p
  | None ->
      let p = prepare ~held works in
      Hashtbl.replace fixed key p;
      p

let call_kernel c =
  match c.call () with
  | A.Done -> ()
  | A.Declined ->
      failwith ("Nx_amd_support: nx.amd declines a call it computed: " ^ c.what)
  | r -> A.refused c.what r c.arrays

let repeat n xs = List.concat (List.init n (fun _ -> xs))

let run g r =
  g.page.{1} <- 0L;
  match r.body with
  | Call c ->
      call_kernel c;
      finish g (drain g) ~late:""
  | Dispatches _ | Copy _ ->
      let p = prepared r "run" ~held:false (fun () -> works r) in
      finish g (submit g p) ~late:""

let call r =
  match r.body with
  | Call c -> call_kernel c
  | Dispatches _ | Copy _ -> invalid_arg "Nx_amd_support.call: no call"

let issue g ~count r =
  match r.body with
  | Dispatches _ | Copy _ -> invalid_arg "Nx_amd_support.issue: no call"
  | Call c ->
      g.page.{1} <- 0L;
      ignore (submit g (fixed_prepared "hold" ~held:true [ Launch (delay g) ]));
      for _ = 1 to count do
        call_kernel c
      done;
      release g;
      finish g (drain g) ~late:"the hold ran out"

(* A round's launches wait behind its hold, their arguments in the device's
   argument segment, 2 MiB unless a submission grew it. A round takes at most
   512 launches and a quarter of the segment: a harness launch its words, in
   steps of 64 bytes, and a launch of nx.amd 4 KiB, the 512 words its loops and
   attention carry at most. *)
let round_launches = 512
let round_bytes = 512 * 1024
let nx_launch_bytes = 4096
let launch_bytes d = 64 * (((8 * List.length d.words) + 63) / 64)

let launches r =
  match r.body with
  | Dispatches ds -> List.length ds
  | Copy _ -> 1
  | Call c -> c.launches

let bytes r =
  match r.body with
  | Dispatches ds -> List.fold_left (fun n d -> n + launch_bytes d) 0 ds
  | Copy _ -> 0
  | Call c -> c.launches * nx_launch_bytes

(* The runs of [r] in a round. *)
let per_round r =
  let by_bytes = round_bytes / Int.max 1 (bytes r) in
  Int.max 1 (Int.min (round_launches / launches r) by_bytes)

let busy g r ~count =
  let per_round = per_round r in
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
        | Dispatches _ | Copy _ ->
            let works () = repeat n (works r) in
            submit g (prepared r (strf "busy %d" n) ~held:false works)
      in
      go (left - n) (Some v)
  in
  go count None

let device_time g r ~count =
  let per_round = per_round r in
  let late n =
    strf "%d runs of %d launches and %d argument bytes outlasted the hold" n
      (launches r) (bytes r)
  in
  let rec go left span =
    if left = 0 then span
    else begin
      let n = Int.min left per_round in
      g.page.{1} <- 0L;
      let v =
        match r.body with
        | Call c ->
            (* Calls are submissions of their own, which follow the head and
               precede the tail by device order (rule 1). *)
            let head =
              [
                Launch (delay g); Barrier; Launch (stamp g 2 ~order:(word g 4));
              ]
            in
            ignore (submit g (fixed_prepared "timed" ~held:true head));
            for _ = 1 to n do
              call_kernel c
            done;
            submit g
              (fixed_prepared "stamp" ~held:false
                 [ Launch (stamp g 3 ~order:(word g 5)) ])
        | Dispatches _ | Copy _ ->
            let works () = timed g (repeat n (works r)) in
            submit g (prepared r (strf "timed %d" n) ~held:true works)
      in
      release g;
      finish g v ~late:(late n);
      go (left - n) (span + Int64.to_int (Int64.sub g.page.{3} g.page.{2}))
    end
  in
  float (go count 0) /. float g.clock_hz /. float count

let buffer g n = Rig.Buffer.create g.device n

(* One copy into the string's own bytes: a byte at a time costs seconds on a
   buffer of a GiB. *)
let read b =
  let n = Rig.Buffer.length b in
  let s = Bytes.create n in
  Rig.Buffer.blit_to_bytes b 0 s 0 n;
  Bytes.unsafe_to_string s

let write b s = Rig.Buffer.copy ~src:(Rig.Buffer.of_string s) ~dst:b

(* A million cycles: some 350 us at 2.9 GHz. *)
let clock g =
  let out = buffer g 16 and cycles = 1_000_000 in
  run g
    (record
       [
         dispatch "shader_clock" ~groups:(1, 1, 1) ~threads:32
           [ A out; W cycles ];
       ]);
  let s = read out in
  let spun = Int64.to_float (String.get_int64_le s 0) in
  let ticks = Int64.to_float (String.get_int64_le s 8) in
  spun /. (ticks /. float g.clock_hz) *. 1e-6

(* Floors *)

(* The floors' and generate's groups: 256 work-items, as the harness sizes them,
   along x up to the 65,535 groups nx.amd's kernels stride over. *)
let threads = 256
let most_groups = 65_535
let groups n per = Int.min most_groups (Int.max 1 ((n + per - 1) / per))

let vectors what b =
  let n = Rig.Buffer.length b in
  if n mod 16 = 0 then n / 16
  else invalid_arg (strf "Nx_amd_support.%s: %d bytes, not vectors" what n)

let floor_copy ~ins ~out =
  if ins = [] || List.length ins > 3 then
    invalid_arg "Nx_amd_support.floor_copy: one to three inputs";
  let vecs = vectors "floor_copy" out in
  let pad l d = l @ List.init (3 - List.length l) (fun _ -> d) in
  let addrs = pad (List.map (fun b -> A b) ins) (A out) in
  let lens = pad (List.map (fun b -> W (vectors "floor_copy" b)) ins) (W 0) in
  let n = groups vecs threads in
  dispatch "floor_copy" ~groups:(n, 1, 1) ~threads
    (addrs @ [ A out ] @ lens @ [ W vecs; W n ])

(* floor_read reads four vectors a work-item a step. *)
let floor_read ?out g b =
  let vecs = vectors "floor_read" b in
  let n = groups vecs (4 * threads) in
  let out =
    match out with
    | None -> buffer g (4 * n)
    | Some o when Rig.Buffer.length o >= 4 * n -> o
    | Some o ->
        invalid_arg
          (strf "Nx_amd_support.floor_read: %d bytes of out, not %d"
             (Rig.Buffer.length o) (4 * n))
  in
  dispatch "floor_read" ~groups:(n, 1, 1) ~threads [ A b; A out; W vecs; W n ]

(* 4 groups of 8 waves a compute unit; each work-item stores 32 bytes. *)
let peak g kernel ~rounds =
  let n = 4 * g.units in
  let o = buffer g (32 * n * threads) in
  let d = dispatch kernel ~groups:(n, 1, 1) ~threads [ A o; W rounds ] in
  (d, n * 8 * rounds * 16)

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
  | None -> invalid_arg "Nx_amd_support.contract: init is not y's dtype"

(* Calls *)

(* Makes [c] once, and is its run; [None] if nx.amd declines it. *)
let first_call c =
  match c.call () with
  | A.Done -> Some { body = Call c; prepared = [] }
  | A.Declined -> None
  | r -> A.refused c.what r c.arrays

(* The most launches a contraction's call submits. *)
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
          Nx_amd.contract ~batch ~contracting ~segments:None ~acc ~dst ~init:i
            { array = A.Any x; scale = None }
            { array = A.Any w; scale = None });
      what = "Nx_amd.contract";
      arrays =
        A.Any dst :: A.Any x :: A.Any w
        :: Option.to_list (Option.map (fun i -> A.Any i) i);
      launches = contract_launches;
    }

(* The program that gives its one operand, of [x]'s dtype, back. *)
let identity (A.Any x) =
  let dt = A.Dtype.Any (A.dtype x) in
  Nx_kernel.Prog.v ~ins:(Iarray.of_list [ dt ])
    (Iarray.of_list [ Nx_kernel.Prog.In 0 ])
    ~outs:(Iarray.of_list [ 0 ])

(* A loop's call: up to three launches. *)
let loop_launches = 3

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
          Nx_amd.reduce p ~shape ~axes
            [| (Nx_kernel.Monoid m, 0) |]
            ~dsts:[| ys |] [| Nx_kernel.Plain xa |]);
      what = "Nx_amd.reduce";
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
          Nx_amd.scan p ~axis (Nx_kernel.Monoid m, 0) ~dsts:[| ys |]
            [| Nx_kernel.Plain xa |]);
      what = "Nx_amd.scan";
      arrays = [ ys; xs ];
      launches = loop_launches;
    }

let kernel_call ~what ~launches arrays call =
  first_call { call; what; arrays; launches }

let empty_call (_ : gpu) =
  let p =
    prepare ~held:false
      [ Launch (dispatch "empty" ~groups:(1, 1, 1) ~threads:32 []) ]
  in
  let call () =
    ignore (start p);
    A.Done
  in
  Option.get (first_call { call; what = "empty"; arrays = []; launches = 1 })

(* Drawn operands *)

type draw = Uniform | Wide of int | Small

let generate (type v s) g b (dt : (v, s) A.Dtype.t) draw ~seed =
  let module D = A.Dtype in
  (match dt with
  | Float4_e2m1fn | Int4 | Uint4 | Complex128 | Complex64 | Bit ->
      invalid_arg (strf "Nx_amd_support.generate: %s is not drawn" (D.name dt))
  | _ -> ());
  let bits = D.bits dt in
  let n = Rig.Buffer.length b * 8 / bits in
  (* harness.hip's DRAW_UNIFORM, DRAW_WIDE and DRAW_SMALL *)
  let draw, spread =
    match draw with Uniform -> (0, 0) | Wide e -> (1, e) | Small -> (2, 0)
  in
  let lo = if D.is Signed dt then -8 else 0 in
  let narrow = Bool.to_int (D.is Float dt && bits < 32) in
  let n_groups = Int.min 4096 (groups n threads) in
  let ws =
    [
      A b;
      W n;
      W seed;
      W (D.code dt);
      W draw;
      W spread;
      W (bits / 8);
      W lo;
      W narrow;
      W n_groups;
    ]
  in
  run g (record [ dispatch "generate" ~groups:(n_groups, 1, 1) ~threads ws ])
