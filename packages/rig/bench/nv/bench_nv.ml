(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* GPU 0, opened through NVIDIA's kernel driver and Rig, each row beside its
   floor: the same submissions through rig_nv_room and rig_nv_submit, called
   from C in a loop on the device the row opened, then a spin on the timeline
   word. A row submits through Rig.submit and waits with Rig.Point.wait, as a program
   does, so its distance to the floor is rig's share and the OCaml side of the
   driver. Memory rows call the driver. Each case opens its device in its own
   worker. Without an NVIDIA GPU the suite has no rows. *)

module N = Rig_nv
module B = Rig.Buffer
module Sub = Rig.Submission
module Abi = Rig_nv_abi
module S = Rig_nv_support
module H = Rig_gpu_support.Host

external start : nativeint -> int -> int -> unit = "rig_nv_bench_start"
external floor_release : int -> unit = "rig_nv_bench_release"
external floor_switch : unit -> unit = "rig_nv_bench_switch"
external floor_waits : int -> int -> unit = "rig_nv_bench_waits"
external floor_entry : int -> int -> unit = "rig_nv_bench_entry"
external floor_entries : int array -> unit = "rig_nv_bench_entries"
external floor_rung : int array -> bool -> unit = "rig_nv_bench_rung"
external floor_copy : int -> int -> int -> unit = "rig_nv_bench_copy"

let kib = 1024
let mib = 1024 * kib
let get = function Ok x -> x | Error why -> failwith why
let host = S.host
let address = S.address
let at r off = host r + off
let row name setup f = Thumper.bench_with_setup ~setup name f

(* The device: [d] rig's, [g] its driver's. *)

type dev = S.t = { d : Rig.t; g : N.t }

let dev = S.open_
let alloc t kind n = Option.get (N.alloc t.g kind n)

(* A submission of parts on [t] and the run it submits with. *)
type submission = { s : Sub.t; run : Sub.Run.t }

let submission t ps = { s = Sub.make t.d ps; run = Sub.Run.make () }
let submit { s; run } = Rig.submit s ~run ~buffers:[||]
let run s = Rig.Point.wait (submit s)

(* The floor of [t]: its later values are given from C. *)
let floor t =
  let f = N.facts t.g in
  start f.edge (host f.word) (Rig.submitted t.d);
  t

(* A host buffer of the ring entry [e], two words. *)
let entry_words e =
  let b = B.create Rig.host 8 in
  B.write b Bigarray.int32 (fun ba ->
      Array.iteri (fun i w -> Bigarray.Array1.set ba i (Int32.of_int w)) e);
  b

(* Launches *)

let cubin =
  lazy
    (In_channel.with_open_bin "../../test/nv/fixtures/kernels_sm89.cubin"
       In_channel.input_all)

let words s =
  Array.init
    (String.length s / 4)
    (fun i -> Int32.to_int (String.get_int32_le s (4 * i)) land 0xffff_ffff)

(* [count] launches of the kernel [empty] over one thread, their descriptors and
   constant bank 0 in [Mapped] memory, as compiled code places them: [`Chained],
   descriptors chained from one segment that schedules the first; [`Apart], a
   segment scheduling each. The result is the loaded image, which stays loaded
   while reachable, and the segments' ring entries, two words each. *)
let launches t count how =
  let c = get (Abi.Cubin.of_string (Lazy.force cubin)) in
  let k = Option.get (Abi.Cubin.kernel c "empty") in
  let p = get (Rig.Image.load t.d (Lazy.force cubin)) in
  let entry = Rig.Image.entry (Option.get (Rig.Image.resolve p "empty")) in
  let cap = N.capability t.g in
  let l = get (Abi.Launch.make cap k) in
  get (cap.local (Abi.Launch.local_bytes l));
  let local = Abi.Local_memory.make cap (Abi.Launch.local_bytes l) in
  let bank0 = S.mapped t.g 4096 in
  let base = entry - k.code in
  let bank (b : Abi.Cubin.bank) =
    let at = if b.index = 0 then address bank0 else base + b.offset in
    (b.index, Abi.Template.Int at)
  in
  let one = Abi.Template.Int 1 in
  let q =
    {
      Abi.Qmd.program = Int entry;
      local = Int local.per_thread;
      banks = List.map bank (Abi.Launch.banks l);
      grid = (one, one, one);
      block = (one, one, one);
      shared = None;
      chain = None;
      releases = [];
    }
  in
  H.write (host bank0) (Abi.Qmd.parameters l q).bytes;
  let stride = 512 in
  let qmds = S.mapped t.g (count * stride) in
  for i = 0 to count - 1 do
    let next = address qmds + ((i + 1) * stride) in
    let q =
      if how = `Chained && i < count - 1 then
        { q with chain = Some (Abi.Template.Int next) }
      else q
    in
    H.write (at qmds (i * stride)) (Abi.Qmd.make l q).bytes
  done;
  let segments = match how with `Chained -> 1 | `Apart -> count in
  let segment_bytes = 256 in
  let segment = S.mapped t.g (segments * segment_bytes) in
  let entry i =
    let qmd = address qmds + (i * stride) in
    let ws = (Abi.Method.schedule (Int qmd)).bytes in
    H.write (at segment (i * segment_bytes)) ws;
    let start = address segment + (i * segment_bytes) in
    let e =
      Abi.Gpfifo.entry (Int start) ~offset:0 ~words:(String.length ws / 4)
    in
    words e.bytes
  in
  (p, Array.init segments entry)

(* Rows *)

let release_rows =
  let empty () =
    let t = dev () in
    (t, submission t [||])
  in
  let mapped () =
    let t, s = empty () in
    (t, s, B.create ~memory:Mapped t.d 4096)
  in
  let switching () =
    let t, none = empty () in
    (* A copy of no bytes runs on the copy channel: it follows nothing. *)
    let copy =
      let src = B.create t.d 0 and dst = B.create t.d 0 in
      submission t Sub.[| copy ~src:(Buffer src) ~dst:(Buffer dst) |]
    in
    (t, copy, none)
  in
  let floor_of setup () = floor (fst (setup ())) in
  Thumper.group "release"
    [
      row "driver" empty (fun (_, s) -> run s);
      row "floor" (floor_of empty) (fun _ -> floor_release 1);
      row "mapped" mapped (fun (_, s, _) -> run s);
      row "floor-mapped"
        (fun () ->
          let t, _, m = mapped () in
          (floor t, m))
        (fun _ -> floor_release 1);
      row "switch" switching (fun (t, copy, none) ->
          run (if Rig.submitted t.d land 1 = 0 then copy else none));
      row "floor-switch" (floor_of empty) (fun _ -> floor_switch ());
      row "no-wait-100" empty (fun (_, s) ->
          for _ = 1 to 99 do
            ignore (submit s)
          done;
          run s);
      row "floor-no-wait-100" (floor_of empty) (fun _ -> floor_release 100);
    ]

(* Rig passes a driver only the waits not yet reached, so four satisfied waits
   are timed from C alone: the waits the driver encodes and the release. *)
let wait_rows =
  Thumper.group "waits"
    [
      row "floor-4"
        (fun () ->
          let t = dev () in
          let w = alloc t Pinned 8 in
          H.set64 (host w) 1;
          ignore (floor t);
          (t, address w))
        (fun (_, at) -> floor_waits at 4);
    ]

(* [count] launches of the kernel [empty] over one thread, as Launch parts of
   one submission on [t], each block stored. *)
let launching t count =
  let image = get (Rig.Image.load t.d (Lazy.force cubin)) in
  let fn = Option.get (Rig.Image.resolve image "empty") in
  let work = Sub.launch fn ~params:0 in
  let s = submission t (Array.make count work) in
  for i = 0 to count - 1 do
    let b = Sub.block s.s i in
    Sub.Run.groups s.run b 1 1 1;
    Sub.Run.threads s.run b 1 1 1
  done;
  s

(* A run of 64 launches follows the GPU's SM clock, which moves by up to a tenth
   between and within runs and which no unprivileged process pins. Launches are
   Launch parts, which the driver chains on one channel. Chained launches in one
   part of ring words are their floor; [64-parts] orders launches as ring words
   of parts of one queue, each scheduled after a wait for idle, and [4096-parts]
   shows what the writer costs per part of a large submission. [4096] is 4,096
   Launch parts in one submission, more than the launch ring holds when the
   device opens. *)
let launch_rows =
  let launching_as how count () =
    let t = dev () in
    let p, es = launches t count how in
    let part e = Rig.Submission.words (entry_words e) in
    (t, p, es.(0), submission t (Array.map part es))
  in
  let floor_launching count () =
    let t, p, e, _ = launching_as `Chained count () in
    (floor t, p, e)
  in
  let launch_parts count () =
    let t = dev () in
    (t, launching t count)
  in
  let floor_run (_, _, e) = floor_entry e.(0) e.(1) in
  (* 64 launches as 64 submissions, and from C as a submission and a release
     each through the driver, then as the driver's words written from C, 64
     values each rung: with a release each, or each ending in the wait for idle
     that orders the next, with one release for them all. The last two rows' gap
     is what the releases cost the GPU. *)
  let submitting () =
    let t = dev () in
    (t, (), Array.init 64 (fun _ -> launching t 1))
  in
  let submit_all (_, _, ss) =
    for i = 0 to Array.length ss - 2 do
      ignore (submit ss.(i))
    done;
    Rig.Point.wait (submit ss.(Array.length ss - 1))
  in
  let floor_apart () =
    let t = dev () in
    let p, es = launches t 64 `Apart in
    (floor t, p, Array.concat (Array.to_list es))
  in
  Thumper.group "launch"
    [
      row "1" (launch_parts 1) (fun (_, s) -> run s);
      row "floor-1" (floor_launching 1) floor_run;
      row "64" (launch_parts 64) (fun (_, s) -> run s);
      row "floor-64" (floor_launching 64) floor_run;
      row "64-parts" (launching_as `Apart 64) (fun (_, _, _, s) -> run s);
      row "submits-64" submitting submit_all;
      row "floor-submits-64" floor_apart (fun (_, _, ws) -> floor_entries ws);
      row "floor-submits-64-rung" floor_apart (fun (_, _, ws) ->
          floor_rung ws true);
      row "floor-submits-64-unreleased" floor_apart (fun (_, _, ws) ->
          floor_rung ws false);
      row "4096-parts" (launching_as `Apart 4096) (fun (_, _, _, s) -> run s);
      row "4096" (launch_parts 4096) (fun (_, s) -> run s);
    ]

let copy_rows =
  let copying n (dst, src) () =
    let t = dev () in
    let dst = B.create ~memory:dst t.d n and src = B.create ~memory:src t.d n in
    let s = submission t Sub.[| copy ~src:(Buffer src) ~dst:(Buffer dst) |] in
    (t, dst, src, s)
  in
  let copy name n kinds =
    [
      row name (copying n kinds) (fun (_, _, _, s) -> run s);
      row ("floor-" ^ name)
        (fun () ->
          let t, dst, src, _ = copying n kinds () in
          let s = Sub.make ~fixed:[ (dst, Read_write); (src, Read) ] t.d [||] in
          (floor t, s, Sub.address s dst, Sub.address s src))
        (fun (_, s, dst, src) ->
          floor_copy dst src n;
          ignore (Sys.opaque_identity s));
    ]
  in
  let big = 256 * mib in
  Thumper.group "copy"
    (copy "16B" 16 (B.Device, B.Device)
    @ copy "h2d-256MiB" big (B.Device, B.Pinned)
    @ copy "d2h-256MiB" big (B.Pinned, B.Device)
    @ copy "d2d-256MiB" big (B.Device, B.Device))

(* A device with a live allocation of [kind], which keeps its page tables, as in
   a program's steady state. *)
let live kind () =
  let t = dev () in
  ignore (alloc t kind 4096);
  t

let alloc_rows =
  let alloc name kind n =
    row name (live kind) (fun t -> N.free t.g (alloc t kind n))
  in
  Thumper.group "alloc"
    [
      alloc "64KiB" Device (64 * kib);
      alloc "64MiB" Device (64 * mib);
      alloc "pinned-64KiB" Pinned (64 * kib);
      alloc "pinned-64MiB" Pinned (64 * mib);
    ]

let map_host_rows =
  let n = 256 * mib in
  Thumper.group "map-host"
    [
      row "256MiB"
        (fun () -> (dev (), H.pages n))
        (fun (t, p) -> N.free t.g (Option.get (N.map_host t.g p n)));
    ]

(* A cubin loaded by the driver over a new code region, then unloaded and the
   region freed. *)
let image_rows =
  Thumper.group "image"
    [
      row "kernels" (live Device) (fun t ->
          match N.image t.g (Lazy.force cubin) with
          | Ok (Place (n, lay)) ->
              let code = alloc t Device n in
              let m, _ = lay code in
              N.unload t.g m;
              N.free t.g code
          | Ok (Loaded _) -> failwith "an image without code"
          | Error why -> failwith why);
    ]

let () =
  if Rig_nv_nvidia.count () > 0 then
    exit
    @@ Thumper.run "rig_nv"
         [
           release_rows;
           wait_rows;
           launch_rows;
           copy_rows;
           alloc_rows;
           map_host_rows;
           image_rows;
         ]
