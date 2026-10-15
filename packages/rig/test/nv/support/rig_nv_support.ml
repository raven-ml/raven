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

module D = N

include Rig_gpu_support.Make (struct
  module D = N

  let class_ = "NV"
  let present () = Rig_nv_nvidia.count () > 0
  let open_ ~word = (Rig_nv_nvidia.gpu 0).make ~word
end)

(* The C entries *)

module Edge = struct
  external submit_c : nativeint -> int -> (int * int array) array -> unit
    = "rig_nv_test_submit"

  let submit g ~v ps = submit_c (N.facts g).edge v ps
end

(* Host memory *)

external pattern : int -> int -> int -> unit = "rig_nv_test_pattern"
external mismatch : int -> int -> int -> int = "rig_nv_test_mismatch"

let host r =
  match (N.locate r).host with
  | Some a -> a
  | None -> fail "the host does not address r"

let address r = Option.get (N.locate r).address
let word g = (N.facts g).word

let mapped g n =
  match N.alloc g Mapped n with
  | Some r -> r
  | None -> require_some (N.alloc g Pinned n)

(* Work through rig *)

let run t ps = Rig.Point.wait (submit t ps)

let words ws =
  let b = B.create Rig.host (4 * Array.length ws) in
  B.write b Bigarray.int32 (fun ba ->
      Array.iteri (fun i w -> Bigarray.Array1.set ba i (Int32.of_int w)) ws);
  Sub.words b

let copy ~dst src = Sub.(copy ~src:(Buffer src) ~dst:(Buffer dst))

(* Host buffers of 64 KiB or more start on a page, which a device borrows. *)
let shared t n =
  let h = B.view (B.create Rig.host (max n 65536)) ~first:0 ~length:n in
  match B.borrow t.d h with
  | Some b -> (b, H.at h)
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

let launches g = { g; memory = mapped g (slots * slot); next = 0 }
let reset l = l.next <- 0
let free_launches l = N.free l.g l.memory

let take l =
  if l.next = slots then fail "no launch slot left";
  let at = l.next * slot in
  l.next <- l.next + 1;
  at

let at_host l at = host l.memory + at
let at_gpu l at = address l.memory + at

(* The ring entry, as two words, of the segment [words] at the GPU address
   [a]. *)
let ring_entry a words =
  let e =
    (A.Gpfifo.entry (Int a) ~offset:0 ~words:(String.length words / 4)).bytes
  in
  Array.init 2 (fun i ->
      Int32.to_int (String.get_int32_le e (4 * i)) land 0xffff_ffff)

let segment l (p : A.Template.t) =
  let at = take l + segment_at in
  let words = p.bytes in
  H.write (at_host l at) words;
  ring_entry (at_gpu l at) words

(* A launch of [kernel] of [cubin], whose first instruction is at [entry], in
   the slot at the GPU address [addr] whose bytes [put] stores, at an offset of
   the slot: the ring entry of its segment. *)
let schedule g ~addr ~put cubin name entry ~blocks args =
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
  let base = entry - kernel.code in
  let bank (b : A.Cubin.bank) =
    let a = if b.index = 0 then addr + bank_at else base + b.offset in
    (b.index, A.Template.Int a)
  in
  let one = A.Template.Int 1 in
  let q =
    {
      A.Qmd.program = Int entry;
      local = Int local.per_thread;
      banks = List.map bank (A.Launch.banks launch);
      grid = (Int blocks, one, one);
      block = (Int 256, one, one);
      shared = None;
      chain = None;
      releases = [];
    }
  in
  put bank_at (A.Qmd.parameters launch q).bytes;
  List.iteri
    (fun i x ->
      let b = Bytes.create 8 in
      Bytes.set_int64_le b 0 (Int64.of_int x);
      put (bank_at + kernel.params_offset + (8 * i)) (Bytes.to_string b))
    args;
  put 0 (A.Qmd.make launch q).bytes;
  let words = (A.Method.schedule (Int addr)).bytes in
  put segment_at words;
  ring_entry (addr + segment_at) words

let launch_kernel l cubin name entry ~blocks args =
  let at = take l in
  schedule l.g ~addr:(at_gpu l at)
    ~put:(fun o s -> H.write (at_host l (at + o)) s)
    cubin name entry ~blocks args

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

(* Conformance *)

let fixtures = "../nv/fixtures"
let cubin = "kernels_sm89.cubin"

let binary () =
  ( fixture ~dir:fixtures cubin,
    [ "empty"; "double_index"; "copy_after"; "spin"; "stack"; "stack_held" ] )

let second ~word =
  if Rig_nv_nvidia.count () < 2 then None
  else Some ((Rig_nv_nvidia.gpu 1).make ~word)

let images = Rig_gpu_support.loader (fun () -> fst (binary ()))

let launching t kernel ~params ~blocks uses store =
  Rig_gpu_support.launching (images t.d) kernel ~params ~groups:blocks
    ~threads:256 uses store

let copy_words t ~dst ~src =
  launching t "copy_after" ~params:32 ~blocks:1
    [| (dst, Rig.Read_write, 8); (src, Rig.Read, 16) |]
    (fun run b -> Sub.Run.int64 run b 24 (B.length src))

(* rig_nv.mli: "5 us once a measure within the bound is taken, plus the
   line's rate error over the time since the latest measure, at most
   62.5 ms. That rate error is at most 10 us over the measures' distance,
   10 ppm, and the timer's own until they are a second apart." The timer's
   own is taken within the same 10 ppm. *)
let clock_error ~since:_ = 5_000 + 625

let spin t ~ns =
  let flag = Rig_gpu_support.arguments t.d (String.make 8 '\000') in
  launching t "spin" ~params:16 ~blocks:1
    [| (flag, Rig.Read, 0) |]
    (fun run b -> Sub.Run.int64 run b 8 ns)

let launch_binary () = Some (fixture ~dir:fixtures "launch_sm89.cubin")
let wide_binary _ = launch_binary ()
