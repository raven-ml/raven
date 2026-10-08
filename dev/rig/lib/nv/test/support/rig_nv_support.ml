(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module N = Rig_nv
module A = Rig_nv_abi
module C = Rig
module B = Rig.Buffer
module Sub = Rig.Submission

let strf = Printf.sprintf

(* The machine's GPU lock *)

external lock : string -> string -> int = "rig_nv_test_lock"

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

let hold_gpu () = if Rig_nv_nvidia.count () > 0 then take 0

(* The GPU *)

type dev = { d : C.t; g : N.t }

(* The driver device a test opened, with its core device if it has one, until a
   test stops it: one a failed test left open is stopped by the next. The core
   stops a device it lost. *)
let opened = ref None

let stop g =
  (match !opened with Some (o, _) when o == g -> opened := None | _ -> ());
  N.stop g

(* Before a core device's driver stops, the core gives back what the device
   mapped of collected host memory: the host's next allocation hands each
   mapping to its device, whose next allocation gives it back. A stopped device
   allocates no more, and the path would keep mappings of memory the process may
   map anew at the same addresses. *)
let drain d =
  Gc.full_major ();
  Gc.full_major ();
  ignore (Sys.opaque_identity (B.create C.host 1));
  C.wait d (C.submitted d);
  ignore (Sys.opaque_identity (B.create d 1))

let stop_opened () =
  match !opened with
  | Some (_, Some d) when Option.is_some (C.lost d) -> opened := None
  | Some (g, Some d) ->
      drain d;
      stop g
  | Some (g, None) -> stop g
  | None -> ()

let open_driver () =
  if Rig_nv_nvidia.count () = 0 then
    skip ~reason:"the machine has no NVIDIA GPU" ();
  hold_gpu ();
  stop_opened ();
  match Rig_nv_nvidia.open_ 0 with
  | Ok g -> g
  | Error why -> failf "opening GPU 0: %s" why

let driver () =
  let g = open_driver () in
  opened := Some (g, None);
  g

(* Each core device takes a name of its own: the core keeps a name's device open
   until it is lost, and a test stops the driver device under it. *)
let names = ref 0

let gpu () =
  let g = open_driver () in
  incr names;
  match C.open_ (module N) ~name:(strf "NV:test%d" !names) (fun () -> Ok g) with
  | Ok d ->
      opened := Some (g, Some d);
      { d; g }
  | Error why ->
      opened := Some (g, None);
      failf "opening GPU 0 in the core: %s" why

let close t =
  if Option.is_none (C.lost t.d) then begin
    drain t.d;
    stop t.g
  end
  else
    match !opened with Some (o, _) when o == t.g -> opened := None | _ -> ()

let stop_left g =
  match !opened with Some (o, _) when o == g -> stop_opened () | _ -> ()

let with_driver f =
  let g = driver () in
  Fun.protect ~finally:(fun () -> stop_left g) (fun () -> f g)

let with_gpu f =
  let t = gpu () in
  Fun.protect ~finally:(fun () -> stop_left t.g) (fun () -> f t)

(* Host memory *)

external page_size : unit -> int = "rig_nv_test_page_size"
external pages : int -> int = "rig_nv_test_pages"
external free_pages : int -> int -> unit = "rig_nv_test_free_pages"
external get64 : int -> int = "rig_nv_test_get64"
external set64 : int -> int -> unit = "rig_nv_test_set64"
external read : int -> int -> string = "rig_nv_test_read"
external write : int -> string -> unit = "rig_nv_test_write"
external pattern : int -> int -> int -> unit = "rig_nv_test_pattern"
external mismatch : int -> int -> int -> int = "rig_nv_test_mismatch"

let page = page_size ()

let host r =
  match N.host r with Some a -> a | None -> fail "the host does not address r"

let address r = Option.get (N.address r)

let get32 a i =
  Int32.to_int (String.get_int32_le (read (a + (4 * i)) 4) 0) land 0xffff_ffff

(* Work through the core *)

let submit t ps =
  C.Point.value (C.submit (Sub.make ~reads:0 ~writes:0 ~waits:0 t.d ps))

let run t ps = C.wait t.d (submit t ps)

let words ?(after = [||]) ws =
  let b = B.create C.host (4 * Array.length ws) in
  let ba = B.bigarray Bigarray.int32 b in
  Array.iteri (fun i w -> Bigarray.Array1.set ba i (Int32.of_int w)) ws;
  { Sub.queue = "COMPUTE:0"; after; work = Words b }

let copy ?(after = [||]) ~dst src =
  { Sub.queue = "COPY:0"; after; work = Copy { src; dst } }

(* Host buffers of 64 KiB or more start on a page, which a device borrows. *)
let shared t n =
  let h = B.view (B.create C.host (max n 65536)) ~first:0 ~length:n in
  match B.borrow t.d h with
  | Some b -> (b, B.address h)
  | None -> fail "the device does not borrow host memory"

(* Work at the C edge *)

type part = {
  queue : int;
  after : int array;
  work : [ `Words of int array | `Copy of int * int * int | `Fill of int * int ];
}

let flat p =
  let kind, a, b, c, ws =
    match p.work with
    | `Words ws -> (0, Array.length ws, 0, 0, ws)
    | `Copy (dst, src, n) -> (1, dst, src, n, [||])
    | `Fill (units, bytes) -> (2, units, bytes, 0, [||])
  in
  Array.concat
    [ [| p.queue; kind; a; b; c; Array.length p.after |]; p.after; ws ]

external room : nativeint -> int array array -> int = "rig_nv_test_room"

external edge_submit_ : nativeint -> int -> int array -> int array array -> int
  = "rig_nv_test_submit"

let edge_room g ps = room (N.self g) (Array.map flat ps)

let edge_submit g ~v ~waits ps =
  let w =
    Array.concat (List.map (fun (a, x) -> [| a; x |]) (Array.to_list waits))
  in
  let r = edge_submit_ (N.self g) v w (Array.map flat ps) in
  if r <> 0 then failf "rig_nv_submit of %d answered %d" v r

let wait g v =
  let word = host (N.word g) in
  let t0 = Unix.gettimeofday () in
  while get64 word < v do
    if Unix.gettimeofday () -. t0 < 0.2 then Domain.cpu_relax ()
    else N.sleep g ~seen:(get64 word) ~still_ms:100
  done

let still ?msg w x f ~ms =
  let t0 = Sys.time () in
  while Sys.time () -. t0 < Float.of_int ms /. 1000. do
    equal ?msg w x (f ())
  done

let watchdog what f =
  let finished = Atomic.make false in
  let d =
    Domain.spawn (fun () ->
        let until = Unix.gettimeofday () +. 10. in
        while (not (Atomic.get finished)) && Unix.gettimeofday () < until do
          Unix.sleepf 0.01
        done;
        if not (Atomic.get finished) then begin
          prerr_endline ("watchdog: " ^ what ^ " did not return in 10 s");
          Unix._exit 2
        end)
  in
  Fun.protect
    ~finally:(fun () ->
      Atomic.set finished true;
      Domain.join d)
    f

(* Kernels *)

let fixture f =
  In_channel.with_open_bin (Filename.concat "fixtures" f) In_channel.input_all

let cubin_of file bin =
  match A.Cubin.of_string bin with
  | Ok c -> c
  | Error e -> failf "%s: %s" file e

type kernels = { cubin : A.Cubin.t; program : C.Program.t }

let kernels ?(file = "kernels_sm89.cubin") t =
  let bin = fixture file in
  match C.Program.load t.d bin with
  | Ok program -> { cubin = cubin_of file bin; program }
  | Error e -> failf "loading %s: %s" file e

let image g bin =
  match N.image g bin with
  | Error e -> failf "loading: %s" e
  | Ok (`Loaded _) -> fail "an image with nothing to place"
  | Ok (`Place (n, lay)) ->
      let r = require_some (N.alloc g `Device n) in
      let i, bytes = lay r in
      (i, r, bytes)

(* Launches *)

(* Each launch takes a slot of [slot] bytes: its descriptor at 0, its constant
   bank 0 at [bank_at], its segment at [segment_at]. *)
let slot = 4096
let bank_at = 512
let segment_at = 3584
let slots = 256

type launches = { g : N.t; memory : N.region; mutable next : int }

let launches g =
  { g; memory = require_some (N.alloc g `Mapped (slots * slot)); next = 0 }

let reset l = l.next <- 0
let free_launches l = N.free l.g l.memory

let take l =
  if l.next = slots then fail "no launch slot left";
  let at = l.next * slot in
  l.next <- l.next + 1;
  at

let at_host l at = host l.memory + at
let at_gpu l at = address l.memory + at

let entry_of l at words =
  let e =
    A.Packet.encode Int64.of_int
      (A.Gpfifo.entry (at_gpu l at) ~offset:0 ~words:(String.length words / 4))
  in
  Array.init 2 (fun i ->
      Int32.to_int (String.get_int32_le e (4 * i)) land 0xffff_ffff)

let segment l p =
  let at = take l + segment_at in
  let words = A.Packet.encode Int64.of_int p in
  write (at_host l at) words;
  entry_of l at words

(* A launch of [kernel] of [cubin], whose first instruction is at [entry]. *)
let launch_kernel l cubin name entry ~blocks args =
  let g = l.g in
  let kernel =
    match A.Cubin.kernel cubin name with
    | Some k -> k
    | None -> failf "no kernel %s" name
  in
  let cap = N.capability g in
  let launch =
    match A.Launch.make cap kernel with
    | Ok l -> l
    | Error e -> failf "%s: %s" name e
  in
  let bytes = A.Launch.local_bytes launch in
  (match cap.local bytes with Ok () -> () | Error e -> failf "local: %s" e);
  let local = A.Local_memory.make cap bytes in
  let at = take l in
  let base = entry - kernel.code in
  let bank q (b : A.Cubin.bank) =
    let a = if b.index = 0 then at_gpu l (at + bank_at) else base + b.offset in
    A.Qmd.set_bank b.index a q
  in
  let q =
    A.Qmd.make launch
    |> A.Qmd.set_dim (Grid X) blocks
    |> A.Qmd.set_dim (Grid Y) 1 |> A.Qmd.set_dim (Grid Z) 1
    |> A.Qmd.set_dim (Block X) 256
    |> A.Qmd.set_dim (Block Y) 1 |> A.Qmd.set_dim (Block Z) 1
    |> A.Qmd.set_program entry
    |> A.Qmd.set_local_memory local.per_thread
  in
  let q = List.fold_left bank q (A.Launch.banks launch) in
  write
    (at_host l (at + bank_at))
    (A.Structure.encode Int64.of_int (A.Qmd.parameters q));
  List.iteri
    (fun i x ->
      let b = Bytes.create 8 in
      Bytes.set_int64_le b 0 (Int64.of_int x);
      write
        (at_host l (at + bank_at + kernel.params_offset + (8 * i)))
        (Bytes.to_string b))
    args;
  write (at_host l at) (A.Structure.encode Int64.of_int (A.Qmd.structure q));
  let words = A.Packet.encode Int64.of_int (A.Method.schedule (at_gpu l at)) in
  write (at_host l (at + segment_at)) words;
  entry_of l (at + segment_at) words

let launch l k f ~blocks args =
  let entry =
    match C.Program.entry k.program f with
    | Some e -> e
    | None -> failf "no kernel %s" f
  in
  launch_kernel l k.cubin f entry ~blocks args

let launch_at l ~code bin f ~blocks args =
  let cubin = cubin_of f bin in
  let kernel = Option.get (A.Cubin.kernel cubin f) in
  launch_kernel l cubin f (code + kernel.code) ~blocks args
