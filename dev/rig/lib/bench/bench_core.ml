(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Submits on Polled, a driver over host memory whose queue runs when the bench
   runs it, each row beside the floor that bounds it: the driver's own room and
   submit entries called directly, with the same parts and the same runs of the
   queue. A row's distance to its floor is the core's share of a submit.

   A row that does not wait runs the queue every [drain] submits, as its floor
   does, so the queue stays short. A replay row runs the queue once per run,
   before its submit: the device completes run N-1 while the host prepares run
   N, and the wait for run N-2 finds it reached. *)

module C = Rig
module B = Rig.Buffer
module Sub = Rig.Submission
module P = Rig_support.Polled
module Support = Rig_support
module Claim = Rig.Claim

let strf = Printf.sprintf

external floor_new :
  nativeint -> nativeint -> nativeint -> nativeint -> nativeint
  = "rig_bench_floor_new"

external floor_submit : nativeint -> int -> unit = "rig_bench_floor_submit"
[@@noalloc]

external floor_turn_submit : nativeint -> int -> unit
  = "rig_bench_floor_turn_submit"

external floor_share : nativeint -> nativeint -> unit = "rig_bench_floor_share"

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

type dev = { d : C.t; p : P.t; mutable n : int }

let opened = ref 0

let dev () =
  incr opened;
  let d, p = P.open_ (strf "bench:%d" !opened) in
  { d; p; n = 0 }

let drained t =
  t.n <- t.n + 1;
  if t.n mod drain = 0 then ignore (P.run t.p)

let words d n = Array.init n (fun _ -> B.create d 8)

(* A part that adds 1 to the word [arg] holds. *)
let bump arg =
  {
    Sub.queue = "COMPUTE:0";
    after = [||];
    work =
      Sub.Fill { fill = Support.bump; arg; ring_units = 0; segment_bytes = 0 };
  }

let bumping ?(reads = 0) ?(writes = 0) d =
  Sub.make ~reads ~writes ~waits:0 d [| bump (B.create C.host 8) |]

let submit_read s bs =
  for i = 0 to Array.length bs - 1 do
    Sub.read s i (Array.unsafe_get bs i)
  done;
  ignore (C.submit s)

(* Submits *)

let empty () =
  let t = dev () in
  (t, Sub.make ~reads:0 ~writes:0 ~waits:0 t.d [||])

let costing () =
  let t = dev () in
  (t, bumping t.d)

let reading () =
  let t = dev () in
  (t, bumping ~reads:slots t.d, words t.d slots)

(* Buffers of another device, its last write reached, borrowed on [t]'s. *)
let foreign () =
  let t = dev () and o = dev () in
  let bs = words o.d slots in
  let w = Sub.make ~reads:0 ~writes:slots ~waits:0 o.d [||] in
  Array.iteri (fun i b -> Sub.write w i b) bs;
  let v = C.Point.value (C.submit w) in
  ignore (P.run o.p);
  C.wait o.d v;
  ( t,
    bumping ~reads:slots t.d,
    Array.map (fun b -> Option.get (B.borrow t.d b)) bs )

(* Another domain submitting to the same device until the row ends. *)
let contended () =
  let t, s = costing () in
  let stop = Atomic.make false in
  let other = { t with n = 0 } and s' = bumping t.d in
  let rival =
    Domain.spawn (fun () ->
        while not (Atomic.get stop) do
          ignore (C.submit s');
          drained other
        done)
  in
  (t, s, stop, rival)

let submit_rows =
  let row name setup f = Thumper.bench_with_setup ~setup name f in
  Thumper.group "submit/polled"
    [
      row "empty" empty (fun (t, s) ->
          let v = C.Point.value (C.submit s) in
          ignore (P.run t.p);
          C.wait t.d v);
      row "cost" costing (fun (t, s) ->
          ignore (C.submit s);
          drained t);
      row "slots-24" reading (fun (t, s, bs) ->
          submit_read s bs;
          drained t);
      row "foreign-24" foreign (fun (t, s, bs) ->
          submit_read s bs;
          drained t);
      Thumper.bench_with_setup ~setup:contended
        ~teardown:(fun (_, _, stop, rival) ->
          Atomic.set stop true;
          Domain.join rival)
        "two-domains"
        (fun (t, s, _, _) ->
          ignore (C.submit s);
          drained t);
    ]

(* Replays *)

type copy = { s : Sub.t; args : B.t; at : int; out : B.t }
type replay = { t : dev; params : B.t array; copies : copy array }

(* Two copies of a step over [slots] parameters: each reads them, writes its
   output, and fills its own arguments, which the host rewrites before each
   run. *)
let replaying () =
  let t = dev () in
  let copy () =
    let args = B.create C.host 8 in
    let s = Sub.make ~reads:slots ~writes:1 ~waits:0 t.d [| bump args |] in
    { s; args; at = B.address args; out = B.create t.d 8 }
  in
  { t; params = words t.d slots; copies = [| copy (); copy () |] }

let run r =
  let c = r.copies.(r.t.n land 1) in
  r.t.n <- r.t.n + 1;
  B.wait c.args B.Read_write;
  ignore (P.run r.t.p);
  Support.store c.at r.t.n;
  for i = 0 to slots - 1 do
    Sub.read c.s i (Array.unsafe_get r.params i)
  done;
  Sub.write c.s 0 c.out;
  ignore (C.submit c.s)

let replay_rows =
  Thumper.group "replay/polled"
    [
      Thumper.bench_with_setup ~setup:replaying "params-24" run;
      Thumper.bench_with_setup ~setup:replaying "pipelined-100" (fun r ->
          for _ = 1 to runs do
            run r
          done;
          ignore (P.run r.t.p);
          C.wait r.t.d (C.submitted r.t.d));
    ]

(* Memory *)

let kib = 1024
let mib = 1024 * kib

let memory () =
  incr opened;
  match C.memory_device (strf "memory:%d" !opened) with
  | Ok d -> d
  | Error why -> failwith why

let host n = B.create C.host n
let chars n = Bigarray.Array1.create Bigarray.char Bigarray.c_layout n

(* Buffers of [d] whose last write, a submission of [d], is reached. *)
let written d n =
  let bs = Array.init n (fun _ -> B.create d 8) in
  let w = Sub.make ~reads:0 ~writes:n ~waits:0 d [||] in
  Array.iteri (fun i b -> Sub.write w i b) bs;
  C.wait d (C.Point.value (C.submit w));
  bs

let row name setup f = Thumper.bench_with_setup ~setup name f
let create d n () = ignore (B.create d n)

let buffer_rows =
  Thumper.group "buffer"
    [
      Thumper.bench "host-create-16" (create C.host 16);
      Thumper.bench "host-create-1M" (create C.host mib);
      row "memory-create-cached-4K" memory (fun d -> create d (4 * kib) ());
      row "wait-reached-24"
        (fun () -> written (memory ()) slots)
        (fun bs ->
          for i = 0 to slots - 1 do
            B.wait (Array.unsafe_get bs i) B.Read_write
          done);
    ]

let claim_rows =
  Thumper.group "claim"
    [
      row "read-release"
        (fun () -> host 16)
        (fun b ->
          Claim.read b;
          Claim.release b);
      row "with-24"
        (fun () -> (List.init slots (fun _ -> host 16), [ [ host 16 ] ]))
        (fun (read, donate) -> Claim.with_ ~read ~donate ignore);
    ]

let copy_rows =
  let pair d n () = (B.create d n, B.create d n) in
  let copy (src, dst) = B.copy ~src ~dst in
  Thumper.group "copy"
    [
      row "host-4K" (pair C.host (4 * kib)) copy;
      row "host-64M" (pair C.host (64 * mib)) copy;
      row "memory-4K" (fun () -> pair (memory ()) (4 * kib) ()) copy;
    ]

(* A collection hands the memory of 1,000 dropped buffers back, and the next
   create drains it into the device's cache. *)
let dropped = 1000

let drain_rows =
  Thumper.group "drain"
    [
      row "dropped-1000" memory (fun d ->
          for _ = 1 to dropped do
            create d (4 * kib) ()
          done;
          Gc.full_major ();
          create d (4 * kib) ());
    ]

let wait_rows =
  Thumper.group "wait/memory"
    [
      row "reached"
        (fun () ->
          let d = memory () in
          ( d,
            C.Point.value
              (C.submit (Sub.make ~reads:0 ~writes:0 ~waits:0 d [||])) ))
        (fun (d, v) -> C.wait d v);
    ]

(* A host buffer of 64 MiB collected: the end of the cycle returns it, being
   more than the cache keeps. *)
let heap_rows =
  Thumper.group "heap"
    [
      Thumper.bench "trim-64M" (fun () ->
          create C.host (64 * mib) ();
          Gc.full_major ());
    ]

let memory_floor_rows =
  let word () = B.address (host 8) in
  let blit (src, dst) = Bigarray.Array1.blit src dst in
  let chars2 n () = (chars n, chars n) in
  [
    Thumper.bench "bigarray-create-16" (fun () -> chars 16);
    Thumper.bench "bigarray-create-1M" (fun () -> chars mib);
    row "word-load" word Support.load;
    row "word-load-24" word (fun a ->
        for _ = 1 to slots do
          ignore (Support.load a)
        done);
    row "atomic-cas-2"
      (fun () -> Atomic.make 0)
      (fun a ->
        ignore (Atomic.compare_and_set a 0 1);
        ignore (Atomic.compare_and_set a 1 0));
    row "memcpy-4K" (chars2 (4 * kib)) blit;
    row "memcpy-64M" (chars2 (64 * mib)) blit;
    Thumper.bench "bigarray-dropped-1000" (fun () ->
        for _ = 1 to dropped do
          ignore (chars (4 * kib))
        done;
        Gc.full_major ();
        chars (4 * kib));
    Thumper.bench "bigarray-collected-64M" (fun () ->
        ignore (chars (64 * mib));
        Gc.full_major ());
  ]

(* Floors *)

type floor = { f : nativeint; fp : P.t; word : int; mutable k : int }

let floor () =
  let fp = P.make () in
  let f = floor_new (P.self fp) P.room_entry P.submit_entry Support.bump in
  { f; fp; word = B.address (B.create C.host 8); k = 0 }

let floor_drained t =
  t.k <- t.k + 1;
  if t.k mod drain = 0 then ignore (P.run t.fp)

let floor_run t =
  t.k <- t.k + 1;
  ignore (P.run t.fp);
  Support.store t.word t.k;
  floor_submit t.f 1

(* Another domain calling the same device's entries until the row ends, with a
   floor of its own: the driver alone serializes the two. With [share], each
   submit also takes one turn the two share, as the core takes a device's: by
   try-lock, and otherwise with the runtime released. *)
let floor_contended ~share () =
  let t = floor () in
  let stop = Atomic.make false in
  let other =
    {
      t with
      f = floor_new (P.self t.fp) P.room_entry P.submit_entry Support.bump;
      k = 0;
    }
  in
  if share then floor_share t.f other.f;
  let submit = if share then floor_turn_submit else floor_submit in
  let rival =
    Domain.spawn (fun () ->
        while not (Atomic.get stop) do
          submit other.f 1;
          floor_drained other
        done)
  in
  (t, stop, rival)

let floor_rows =
  let row name f = Thumper.bench_with_setup ~setup:floor name f in
  Thumper.group "floor"
    ([
       row "polled/release" (fun t ->
           floor_submit t.f 0;
           ignore (P.run t.fp);
           ignore (P.signaled t.fp));
       row "polled/cost" (fun t ->
           floor_submit t.f 1;
           floor_drained t);
       Thumper.bench_with_setup
         ~setup:(floor_contended ~share:false)
         ~teardown:(fun (_, stop, rival) ->
           Atomic.set stop true;
           Domain.join rival)
         "polled/two-domains"
         (fun (t, _, _) ->
           floor_submit t.f 1;
           floor_drained t);
       Thumper.bench_with_setup
         ~setup:(floor_contended ~share:true)
         ~teardown:(fun (_, stop, rival) ->
           Atomic.set stop true;
           Domain.join rival)
         "polled/two-domains-turn"
         (fun (t, _, _) ->
           floor_turn_submit t.f 1;
           floor_drained t);
       row "polled/run" floor_run;
       row "polled/pipelined-100" (fun t ->
           for _ = 1 to runs do
             floor_run t
           done;
           ignore (P.run t.fp);
           ignore (P.signaled t.fp));
       Thumper.bench_with_setup ~setup:Mutex.create "mutex-section" (fun m ->
           Mutex.lock m;
           Mutex.unlock m);
     ]
    @ memory_floor_rows)

(* GPUs *)

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
  [
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
  exit
  @@ Thumper.run "rig"
       ([
          submit_rows;
          replay_rows;
          buffer_rows;
          claim_rows;
          copy_rows;
          drain_rows;
          wait_rows;
          heap_rows;
          floor_rows;
        ]
       @ gpus)
