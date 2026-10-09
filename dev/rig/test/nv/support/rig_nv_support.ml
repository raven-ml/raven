(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module N = Rig_nv
module A = Rig_nv_abi
module B = Rig.Buffer
module Sub = Rig.Submission
module H = Rig_gpu_support.Host

(* The GPU *)

include Rig_gpu_support.Make (struct
  module D = N

  let class_ = "NV"
  let present () = Rig_nv_nvidia.count () > 0
  let open_ () = Rig_nv_nvidia.open_ 0
end)

(* Host memory *)

external pattern : int -> int -> int -> unit = "rig_nv_test_pattern"
external mismatch : int -> int -> int -> int = "rig_nv_test_mismatch"

let host r =
  match (N.locate r).host with
  | Some a -> a
  | None -> fail "the host does not address r"

let address r = Option.get (N.locate r).address
let word g = (N.facts g).word

let gpu g : A.Gpu.t =
  let (Rig_edge.Capability (k, c)) = (N.facts g).capability in
  match Type.Id.provably_equal k A.Gpu.key with
  | Some Equal -> c
  | None -> fail "the capability is no Rig_nv_abi.Gpu.t"

let mapped g n =
  match N.alloc g Mapped n with
  | Some r -> r
  | None -> require_some (N.alloc g Pinned n)

(* Work through rig *)

let run t ps = wait t (submit t ps)

let words ?(after = [||]) ws =
  let b = B.create Rig.host (4 * Array.length ws) in
  let ba = B.bigarray Bigarray.int32 b in
  Array.iteri (fun i w -> Bigarray.Array1.set ba i (Int32.of_int w)) ws;
  { Sub.queue = "COMPUTE:0"; after; work = Words b }

let copy ?(after = [||]) ~dst src =
  { Sub.queue = "COPY:0"; after; work = Copy { src; dst } }

(* Host buffers of 64 KiB or more start on a page, which a device borrows. *)
let shared t n =
  let h = B.view (B.create Rig.host (max n 65536)) ~first:0 ~length:n in
  match B.borrow t.d h with
  | Some b -> (b, B.address h)
  | None -> fail "the device does not borrow host memory"

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

let fixture ?(dir = "fixtures") f =
  In_channel.with_open_bin (Filename.concat dir f) In_channel.input_all

let cubin_of file bin =
  match A.Cubin.of_string bin with
  | Ok c -> c
  | Error e -> failf "%s: %s" file e

type kernels = { cubin : A.Cubin.t; image : Rig.Image.t }

let kernels ?dir ?(file = "kernels_sm89.cubin") t =
  let bin = fixture ?dir file in
  match Rig.Image.load t.d bin with
  | Ok image -> { cubin = cubin_of file bin; image }
  | Error e -> failf "loading %s: %s" file e

let image g bin =
  match N.image g bin with
  | Error e -> failf "loading: %s" e
  | Ok (Loaded _) -> fail "an image with nothing to place"
  | Ok (Place (n, lay)) ->
      let r = require_some (N.alloc g Device n) in
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
  { g; memory = mapped g (slots * slot); next = 0 }

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
  H.write (at_host l at) words;
  entry_of l at words

(* A launch of [kernel] of [cubin], whose first instruction is at [entry]. *)
let launch_kernel l cubin name entry ~blocks args =
  let g = l.g in
  let kernel =
    match A.Cubin.kernel cubin name with
    | Some k -> k
    | None -> failf "no kernel %s" name
  in
  let cap = gpu g in
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
  H.write
    (at_host l (at + bank_at))
    (A.Structure.encode Int64.of_int (A.Qmd.parameters q));
  List.iteri
    (fun i x ->
      let b = Bytes.create 8 in
      Bytes.set_int64_le b 0 (Int64.of_int x);
      H.write
        (at_host l (at + bank_at + kernel.params_offset + (8 * i)))
        (Bytes.to_string b))
    args;
  H.write (at_host l at) (A.Structure.encode Int64.of_int (A.Qmd.structure q));
  let words = A.Packet.encode Int64.of_int (A.Method.schedule (at_gpu l at)) in
  H.write (at_host l (at + segment_at)) words;
  entry_of l (at + segment_at) words

let launch l k f ~blocks args =
  let entry =
    match Rig.Image.entry k.image f with
    | Some e -> e
    | None -> failf "no kernel %s" f
  in
  launch_kernel l k.cubin f entry ~blocks args

let launch_at l ~code bin f ~blocks args =
  let cubin = cubin_of f bin in
  let kernel = Option.get (A.Cubin.kernel cubin f) in
  launch_kernel l cubin f (code + kernel.code) ~blocks args
