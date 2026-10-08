(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module N = Device_nv
module A = Device_nv_abi

external lock : string -> bool = "device_nv_test_lock"

(* The GPU *)

let gpu_lock = "DEVICE_NV_TEST_GPU_LOCK"

(* The lock is taken once and kept: [Some true] once taken. *)
let held = ref None

let take_lock () =
  match !held with
  | Some taken -> taken
  | None ->
      let taken =
        match Sys.getenv_opt gpu_lock with
        | None | Some "" -> skip ~reason:(gpu_lock ^ " names no lock file") ()
        | Some file -> lock file
      in
      held := Some taken;
      taken

(* The device gpu opened, until a test stops it: one a failed test left open is
   stopped by the next gpu. *)
let opened = ref None

let stop g =
  (match !opened with Some o when o == g -> opened := None | _ -> ());
  N.stop g

let gpu () =
  if Device_nv_nvidia.count () = 0 then
    skip ~reason:"the machine has no NVIDIA GPU" ();
  if not (take_lock ()) then
    skip ~reason:"another process holds the GPU lock" ();
  Option.iter (fun g -> ignore (stop g)) !opened;
  let g =
    match Device_nv_nvidia.open_ 0 with
    | Ok g -> g
    | Error why -> failf "opening GPU 0: %s" why
  in
  opened := Some g;
  g

let with_gpu f =
  let g = gpu () in
  let stop_left () =
    match !opened with Some o when o == g -> ignore (stop g) | _ -> ()
  in
  Fun.protect ~finally:stop_left (fun () -> f g)

(* Host memory *)

external page_size : unit -> int = "device_nv_test_page_size"
external pages : int -> int = "device_nv_test_pages"
external free_pages : int -> int -> unit = "device_nv_test_free_pages"
external get64 : int -> int = "device_nv_test_get64"
external set64 : int -> int -> unit = "device_nv_test_set64"
external read : int -> int -> string = "device_nv_test_read"
external write : int -> string -> unit = "device_nv_test_write"
external pattern : int -> int -> int -> unit = "device_nv_test_pattern"
external mismatch : int -> int -> int -> int = "device_nv_test_mismatch"

let page = page_size ()

let host r =
  match N.host r with Some a -> a | None -> fail "the host does not address r"

let address r = Option.get (N.address r)

let get32 a i =
  Int32.to_int (String.get_int32_le (read (a + (4 * i)) 4) 0) land 0xffff_ffff

(* Values *)

let values : (N.t * int ref) list ref = ref []

let value g =
  match List.assq_opt g !values with
  | Some v -> v
  | None ->
      let v = ref 0 in
      values := (g, v) :: !values;
      v

let last g = !(value g)
let given g v = value g := v

let wait g v =
  let word = host (N.word g) in
  let t0 = Sys.time () in
  while get64 word < v do
    if Sys.time () -. t0 > 10. then
      failf "the word stayed at %d below %d for 10 s" (get64 word) v;
    Domain.cpu_relax ()
  done

let answer =
  Testable.make
    ~pp:(fun ppf -> function
      | `Ok -> Format.pp_print_string ppf "`Ok"
      | `Failed why -> Format.fprintf ppf "`Failed %S" why)
    ~equal:( = )

let submit ?(waits = [||]) g ps =
  let rec fits () =
    match N.room g ps with
    | `Fits -> ()
    | `Later ->
        wait g (last g);
        fits ()
    | `Never -> fail "room is Never"
  in
  fits ();
  let v = last g + 1 in
  (match N.submit g ~v ~waits ~handles:[||] ps with
  | `Ok -> given g v
  | `Failed why -> failf "submit of %d failed: %s" v why);
  v

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

external room :
  nativeint -> int -> int -> bool -> int -> int -> bool -> int array -> int
  = "device_nv_test_room_byte" "device_nv_test_room"

let room g ~queue ~words ?(fill = false) ?(units = 0) ?(bytes = 0)
    ?(copy = false) after =
  room (N.self g) queue words fill units bytes copy after

(* Kernels *)

let fixture f =
  In_channel.with_open_bin (Filename.concat "fixtures" f) In_channel.input_all

type kernels = { cubin : A.Cubin.t; image : N.image; code : N.region }

let image k = k.image
let code k = k.code

let kernels ?(file = "kernels_sm89.cubin") g =
  let bin = fixture file in
  let cubin =
    match A.Cubin.of_string bin with
    | Ok c -> c
    | Error e -> failf "%s: %s" file e
  in
  match N.image g bin with
  | Error e -> failf "loading %s: %s" file e
  | Ok (`Loaded _) -> failf "%s has no code to place" file
  | Ok (`Place (n, lay)) ->
      let code = require_some (N.alloc g `Device n) in
      let image, bytes = lay code in
      let staging = require_some (N.alloc g `Pinned n) in
      write (host staging) bytes;
      let copy = `Copy ((code, 0), (staging, 0), n) in
      wait g (submit g [| N.part g ~queue:"COPY:0" copy |]);
      N.free g staging;
      { cubin; image; code }

let unload g k =
  N.unload g k.image;
  N.free g k.code

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
let free_launches g l = N.free g l.memory

let take l =
  if l.next = slots then fail "no launch slot left";
  let at = l.next * slot in
  l.next <- l.next + 1;
  at

let at_host l at = host l.memory + at
let at_gpu l at = address l.memory + at

(* The entry of the segment of the words [p] in a new slot. *)
let segment l at p =
  let words = A.Packet.encode Int64.of_int p in
  write (at_host l (at + segment_at)) words;
  let e =
    A.Packet.encode Int64.of_int
      (A.Gpfifo.entry
         (at_gpu l (at + segment_at))
         ~offset:0
         ~words:(String.length words / 4))
  in
  Array.init 2 (fun i ->
      Int32.to_int (String.get_int32_le e (4 * i)) land 0xffff_ffff)

let release l a x = segment l (take l) (A.Method.release System a x)

let launch l k f ~blocks args =
  let g = l.g in
  let kernel =
    match A.Cubin.kernel k.cubin f with
    | Some k -> k
    | None -> failf "no kernel %s" f
  in
  let entry = Option.get (N.entry k.image f) in
  let cap = N.capability g in
  let launch =
    match A.Launch.make cap kernel with
    | Ok l -> l
    | Error e -> failf "%s: %s" f e
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
  segment l at (A.Method.schedule (at_gpu l at))
