(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* NV devices on a real GPU: opening, memory and its budget, every copy route
   with the timeline steps that identify it, borrows, one host buffer borrowed
   by two GPUs, peer copies, every route around the staging slot and the copy
   engine's longest line, copies that wrap the copy channel, coherence through
   each kind of memory, programs and local memory, the timeline across 2^32,
   nx's runtime laws over the devices, and last, work that never signals, which
   fails the device. Every test skips without a GPU, and the peer tests without
   two. GPUs are reached through the kernel driver when it is loaded; over PCI
   the runtime takes a GPU from its kernel driver, so the suite does only when
   NX_NV_PCI_TEST names the index of a GPU it may take. *)

open Windtrap
module B = Nx_device.Buffer
module S = Nx_dtype.Scalar

let mib = 1 lsl 20

(* The index of the first GPU the suite may take over PCI, if any. *)
let pci_first () = Option.map int_of_string (Sys.getenv_opt "NX_NV_PCI_TEST")

(* The GPU [i] under test, or a skip. *)
let device ?(i = 0) () =
  match pci_first () with
  | Some first -> (
      match Nx_nv_device.get ~interface:Pci (first + i) with
      | Ok d -> d
      | Error msg -> skip ~reason:msg ())
  | None ->
      if Nx_nv_device.count ~interface:Kernel () <= i then
        skip
          ~reason:
            "no NVIDIA GPU through the kernel driver; set NX_NV_PCI_TEST=i to \
             take GPU i over PCI"
          ()
      else Nx_nv_device.v ~interface:Kernel i

(* The GPUs nx's runtime laws run on. *)
let gpus =
  match pci_first () with
  | Some first -> (
      match Nx_nv_device.get ~interface:Pci first with
      | Ok d -> [ d ]
      | Error _ -> [])
  | None ->
      List.init
        (Nx_nv_device.count ~interface:Kernel ())
        (Nx_nv_device.v ~interface:Kernel)

let fill_host n f =
  let b = B.create Nx_device.host S.UInt8 n in
  let ba = B.bigarray Bigarray.int8_unsigned b in
  for i = 0 to n - 1 do
    ba.{i} <- f i land 0xff
  done;
  b

let to_host b =
  let h = B.create Nx_device.host S.UInt8 (B.nbytes b) in
  B.copy ~src:b ~dst:h;
  h

let same_bytes a b =
  let x = B.bigarray Bigarray.int8_unsigned a
  and y = B.bigarray Bigarray.int8_unsigned b in
  Bigarray.Array1.dim x = Bigarray.Array1.dim y
  &&
  let rec go i = i = Bigarray.Array1.dim x || (x.{i} = y.{i} && go (i + 1)) in
  go 0

(* The timeline steps a copy takes: one per direct copy, one per chunk of a
   staged one. *)
let steps d f =
  let before = Nx_device.submitted d in
  f ();
  Nx_device.submitted d - before

let test_open () =
  let d = device () in
  is_true ~msg:"memoized" (Nx_device.equal d (device ()));
  starts_with ~msg:"name" ~affix:"NV" (Nx_device.name d);
  starts_with ~msg:"arch" ~affix:"sm_" (Nx_device.arch d);
  let h = Nx_nv_device.handles d in
  is_true ~msg:"multiprocessors"
    (h.props.gpcs * h.props.tpcs_per_gpc * h.props.sms_per_tpc > 0);
  is_true ~msg:"warps" (h.props.warps_per_sm > 0);
  equal ~msg:"the signal is the timeline's" nativeint
    (B.address (Nx_device.timeline d))
    h.signal;
  is_true ~msg:"channel rings below 2^40"
    (Nativeint.to_int h.compute.ring < 1 lsl 40
    && Nativeint.to_int h.copy.ring < 1 lsl 40);
  is_true ~msg:"a budget" (Nx_device.budget d > 0);
  Nx_nv_device.invalidate_caches d

let test_memory () =
  let d = device () in
  let n = (3 * mib) + 17 in
  let src = fill_host n (fun i -> i * 7) in
  let v = B.create d S.UInt8 n in
  raises_match (Exn.invalid_arg ~substring:"host does not address") (fun () ->
      B.host_address v);
  equal ~msg:"into VRAM directly from aligned host memory" int 1
    (steps d (fun () -> B.copy ~src ~dst:v));
  let back = B.create Nx_device.host S.UInt8 n in
  equal ~msg:"out" int 1 (steps d (fun () -> B.copy ~src:v ~dst:back));
  is_true ~msg:"round trip" (same_bytes src back);
  let pinned = B.create ~host:true d S.UInt8 n in
  is_true ~msg:"host memory the host addresses" (B.host_address pinned <> 0n);
  equal ~msg:"VRAM to host memory of the GPU" int 1
    (steps d (fun () -> B.copy ~src:v ~dst:pinned));
  is_true ~msg:"its bytes" (same_bytes src (to_host pinned));
  let w = B.create d S.UInt8 n in
  equal ~msg:"VRAM to VRAM" int 1 (steps d (fun () -> B.copy ~src:v ~dst:w));
  let a = B.view w ~offset:4 S.UInt8 100
  and b = B.view v ~offset:1000 S.UInt8 100 in
  B.copy ~src:b ~dst:a;
  is_true ~msg:"views at offsets"
    (same_bytes (to_host a) (to_host (B.view src ~offset:1000 S.UInt8 100)));
  raises_match
    (function Nx_device.Out_of_memory _ -> true | _ -> false)
    (fun () -> B.create d S.UInt8 (Nx_device.budget d + 1))

let test_staged () =
  let d = device () in
  let n = (130 * mib) + 5 in
  let src = fill_host n (fun i -> (i * 13) + (i lsr 16)) in
  let v = B.create d S.UInt8 (n + 64) in
  let dst = B.view v ~offset:64 S.UInt8 n in
  (* Small host buffers do not start on a page, so they are staged. *)
  let small = fill_host 1000 (fun i -> i) and sv = B.create d S.UInt8 1000 in
  equal ~msg:"a small buffer through one chunk" int 1
    (steps d (fun () -> B.copy ~src:small ~dst:sv));
  let unaligned = B.view src ~offset:1 S.UInt8 (n - 1) in
  let dst' = B.view v ~offset:64 S.UInt8 (n - 1) in
  equal ~msg:"three chunks into an offset view" int 3
    (steps d (fun () -> B.copy ~src:unaligned ~dst:dst'));
  let back = B.create Nx_device.host S.UInt8 (n - 1) in
  B.copy ~src:dst' ~dst:(B.view back ~offset:0 S.UInt8 (n - 1));
  is_true ~msg:"the bytes" (same_bytes unaligned back);
  ignore (Sys.opaque_identity dst)

let test_borrow () =
  let d = device () in
  let n = 4 * mib in
  let host = fill_host n (fun i -> i) in
  let borrowed = B.borrow d host in
  is_true ~msg:"borrowed" (B.is_borrowed borrowed);
  let v = B.create d S.UInt8 n in
  let src = fill_host n (fun i -> 255 - i) in
  B.copy ~src ~dst:v;
  equal ~msg:"into borrowed memory directly" int 1
    (steps d (fun () -> B.copy ~src:v ~dst:borrowed));
  Nx_device.synchronize Nx_device.host;
  is_true ~msg:"seen through the host buffer" (same_bytes src host);
  let also = B.borrow d (B.view host ~offset:(64 * 1024) S.UInt8 1024) in
  B.copy ~src:(B.view v ~offset:0 S.UInt8 1024) ~dst:also;
  raises_match (Exn.invalid_arg ~substring:"does not start on a page")
    (fun () ->
      B.borrow d
        (B.of_bigarray
           (Bigarray.Array1.sub
              (B.bigarray Bigarray.int8_unsigned
                 (B.create Nx_device.host S.UInt8 (2 * mib)))
              1 mib)))

let test_peer () =
  let d0 = device () and d1 = device ~i:1 () in
  let n = 2 * mib in
  let src = fill_host n (fun i -> i * 3) in
  let a = B.create d0 S.UInt8 n and b = B.create d1 S.UInt8 n in
  B.copy ~src ~dst:a;
  let on_d0 = ref 0 in
  let on_d1 =
    steps d1 (fun () -> on_d0 := steps d0 (fun () -> B.copy ~src:a ~dst:b))
  in
  equal ~msg:"one step on the source" int 1 !on_d0;
  is_true ~msg:"none on the destination for a transfer, one for a bounce"
    (on_d1 = 0 || on_d1 = 1);
  let back = B.create Nx_device.host S.UInt8 n in
  B.copy ~src:b ~dst:back;
  is_true ~msg:"the bytes" (same_bytes src back);
  let h1 = B.create ~host:true d1 S.UInt8 n in
  B.copy ~src:a ~dst:h1;
  let back = B.create Nx_device.host S.UInt8 n in
  B.copy ~src:h1 ~dst:back;
  is_true ~msg:"into the other GPU's host memory" (same_bytes src back);
  let t0 =
    Domain.spawn (fun () ->
        for _ = 1 to 20 do
          B.copy ~src ~dst:a
        done)
  in
  for _ = 1 to 20 do
    B.copy ~src ~dst:b
  done;
  Domain.join t0

(* One host buffer borrowed by two GPUs, and let go by the first: the second
   still reads it, and the first borrows it again. *)
let test_two_borrows () =
  let d0 = device () and d1 = device ~i:1 () in
  let n = 2 * mib in
  let host = fill_host n (fun i -> i * 11) in
  let read d =
    let v = B.create d S.UInt8 n in
    B.copy ~src:(B.borrow d host) ~dst:v;
    to_host v
  in
  is_true ~msg:"the first GPU" (same_bytes host (read d0));
  is_true ~msg:"the second GPU" (same_bytes host (read d1));
  Gc.full_major ();
  is_true ~msg:"the second once the first let go" (same_bytes host (read d1));
  is_true ~msg:"the first again" (same_bytes host (read d0))

(* A kernel compiled at test time, when NVIDIA's compiler is at hand: it writes
   through a local array, so that it uses local memory. *)
let compile arch =
  match
    List.find_opt Sys.file_exists
      [ "/usr/local/cuda/bin/nvcc"; "/usr/bin/nvcc" ]
  with
  | None -> None
  | Some nvcc ->
      let dir = Filename.temp_dir "nx-nv" "" in
      let src = Filename.concat dir "k.cu"
      and out = Filename.concat dir "k.cubin" in
      Out_channel.with_open_text src (fun oc ->
          output_string oc
            "extern \"C\" __global__ void fill(int *p) { volatile int s[64]; \
             s[threadIdx.x % 64] = 42; p[threadIdx.x] = s[threadIdx.x % 64]; }\n");
      let cmd =
        Printf.sprintf "%s -cubin -arch=%s -O2 %s -o %s 2>/dev/null" nvcc arch
          src out
      in
      if Sys.command cmd = 0 then
        Some (In_channel.with_open_bin out In_channel.input_all)
      else None

let test_programs () =
  let d = device () in
  match compile (Nx_device.arch d) with
  | None -> skip ~reason:"no compiler for the GPU's architecture" ()
  | Some binary ->
      let p = Nx_device.Program.load d ~binary ~name:"fill" in
      is_true ~msg:"cached" (p == Nx_device.Program.load d ~binary ~name:"fill");
      let k = Nx_nv_device.kernel p in
      equal ~msg:"the handle is the entry" nativeint k.entry
        (Nx_device.Program.handle p);
      is_true ~msg:"inside the image" (k.entry >= k.image);
      is_true ~msg:"registers" (k.registers > 0);
      is_true ~msg:"threads" (k.max_threads >= 32);
      raises_match (Exn.failure ~substring:"no function") (fun () ->
          Nx_device.Program.load d ~binary ~name:"absent")

let test_local_memory () =
  let d = device () in
  let l = Nx_nv_device.local_memory d 256 in
  is_true ~msg:"memory" (l.address <> 0n && l.bytes > 0);
  is_true ~msg:"per thread" (l.per_thread >= 256 && l.per_thread mod 32 = 0);
  let again = Nx_nv_device.local_memory d 128 in
  equal ~msg:"kept for smaller kernels" nativeint l.address again.address;
  let more = Nx_nv_device.local_memory d 4096 in
  is_true ~msg:"grown for larger ones" (more.bytes > l.bytes);
  Nx_device.synchronize d

(* The device's timeline across 2^32: from 2^32 - 2, three copies take it
   through the carry, which the copy engine writes a word at a time. *)
let test_carry () =
  let d = device () in
  Nx_device.synchronize d;
  let start = (1 lsl 32) - 2 in
  let words = B.create Nx_device.host S.UInt64 2 in
  Bigarray.Array1.fill (B.bigarray Bigarray.int64 words) (Int64.of_int start);
  B.copy ~src:words ~dst:(Nx_device.timeline d);
  let src = fill_host mib (fun i -> i * 5) and v = B.create d S.UInt8 mib in
  for _ = 1 to 3 do
    B.copy ~src ~dst:v
  done;
  Nx_device.synchronize d;
  equal ~msg:"submitted" int (start + 3) (Nx_device.submitted d);
  equal ~msg:"signaled" int (start + 3) (Nx_device.signaled d);
  is_true ~msg:"the bytes" (same_bytes src (to_host v))

(* Boundaries, wraps and coherence: copies whose bytes a wrong chunk, a ring
   overwritten or a wrong page-table bit would change without hanging. *)

type bytes_ba =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

let bytes_of b : bytes_ba = B.bigarray Bigarray.int8_unsigned b

(* Byte [i] of the pattern of [seed]: a hash of the index, so that a byte
   shifted, duplicated or left from an earlier pattern is not the one
   expected. *)
let byte seed i =
  let h = (i + (seed lsl 40)) * 0x9E3779B97F4A7C1 in
  ((h lxor (h lsr 29)) lsr 17) land 0xff

(* Writes into the host buffer [b] the pattern of [seed] from its byte
   [from]. *)
let write_pattern ?(from = 0) seed b =
  let ba = bytes_of b in
  for i = 0 to Bigarray.Array1.dim ba - 1 do
    Bigarray.Array1.unsafe_set ba i (byte seed (from + i))
  done

(* Checks every byte of [b], on any device, against the pattern of [seed] from
   its byte [from]. *)
let check ~msg ?(from = 0) seed b =
  let host =
    if Nx_device.equal (B.device b) Nx_device.host then b else to_host b
  in
  let ba = bytes_of host in
  let n = Bigarray.Array1.dim ba in
  let rec go i =
    if i < n then
      let want = byte seed (from + i) in
      if Bigarray.Array1.unsafe_get ba i <> want then
        fail
          (Printf.sprintf "%s: byte %d of %d is 0x%02x, not 0x%02x" msg i n
             ba.{i} want)
      else go (i + 1)
  in
  go 0

let view b off n = B.view b ~offset:off S.UInt8 n

(* The sizes around [max], a staging slot of 64 MiB or the copy engine's longest
   line of 2 GiB: one under, at, one over, and two and a tail. *)
let around max = [ max - 1; max; max + 1; (2 * max) + 4097 ]

(* The largest of [around max], with room for the offsets, if two buffers of it
   fit in half of [d]'s budget; a skip otherwise. *)
let room d max =
  let top = List.fold_left Int.max 0 (around max) + 8 in
  if 4 * top > Nx_device.budget d then
    skip
      ~reason:(Printf.sprintf "copies of %d bytes need more GPU memory" top)
      ();
  top

(* Every route at every size around [max]: direct from and to aligned host
   memory, staged from and to host memory off a page, VRAM to VRAM at offsets,
   and into and out of borrowed memory. *)
let boundaries max () =
  let d = device () in
  let top = room d max in
  let src = B.create Nx_device.host S.UInt8 top in
  write_pattern 1 src;
  let out = B.create Nx_device.host S.UInt8 top in
  let lent = B.create Nx_device.host S.UInt8 top in
  let borrowed = B.borrow d lent in
  let v = B.create d S.UInt8 top and w = B.create d S.UInt8 top in
  List.iter
    (fun n ->
      let at what = Printf.sprintf "%s, %d bytes" what n in
      equal ~msg:(at "direct in") int 1
        (steps d (fun () -> B.copy ~src:(view src 0 n) ~dst:(view v 0 n)));
      B.copy ~src:(view v 0 n) ~dst:(view out 0 n);
      check ~msg:(at "direct in and out") 1 (view out 0 n);
      B.copy ~src:(view src 1 n) ~dst:(view v 0 n);
      B.copy ~src:(view v 0 n) ~dst:(view out 3 n);
      check ~msg:(at "staged in and out") ~from:1 1 (view out 3 n);
      B.copy ~src:(view v 0 n) ~dst:(view w 5 n);
      check ~msg:(at "VRAM to VRAM") ~from:1 1 (view w 5 n);
      B.copy ~src:(view w 5 n) ~dst:(view borrowed 2 n);
      Nx_device.synchronize Nx_device.host;
      check ~msg:(at "into borrowed memory") ~from:1 1 (view lent 2 n);
      B.copy ~src:(view borrowed 2 n) ~dst:(view v 7 n);
      check ~msg:(at "out of borrowed memory") ~from:1 1 (view v 7 n))
    (around max)

(* Between two GPUs, a transfer or a bounce, at every size around [max]. *)
let peer_boundaries max () =
  let d0 = device () and d1 = device ~i:1 () in
  let top = Int.max (room d0 max) (room d1 max) in
  let src = B.create Nx_device.host S.UInt8 top in
  write_pattern 3 src;
  let a = B.create d0 S.UInt8 top and b = B.create d1 S.UInt8 top in
  B.copy ~src ~dst:a;
  List.iter
    (fun n ->
      B.copy ~src:(view a 1 n) ~dst:(view b 6 n);
      check
        ~msg:(Printf.sprintf "to the other GPU, %d bytes" n)
        ~from:1 3 (view b 6 n))
    (around max)

(* Copies of 16 bytes one after the other, three times as many as the copy
   channel has entries, which wraps its ring three times and the runtime's
   command segments many more: every copy lands where it should. *)
let test_ring_wraps () =
  let d = device () in
  let copies = (3 * (Nx_nv_device.handles d).copy.entries) + 17 in
  let n = 16 * copies in
  let src = B.create Nx_device.host S.UInt8 n in
  write_pattern 2 src;
  let v = B.create d S.UInt8 n and w = B.create d S.UInt8 n in
  B.copy ~src ~dst:v;
  let before = Nx_device.submitted d in
  for k = 0 to copies - 1 do
    B.copy ~src:(view v (16 * k) 16) ~dst:(view w (16 * k) 16)
  done;
  equal ~msg:"one step a copy" int copies (Nx_device.submitted d - before);
  check ~msg:"every copy" 2 w

(* The host writes, the GPU copies, the host reads, round after round of new
   bytes over the same memory, through the GPU's host memory, borrowed memory
   and VRAM: a byte the GPU or the host read from a stale cache shows. *)
let test_coherence () =
  let d = device () in
  let n = (2 * mib) + 4099 in
  let written = B.create Nx_device.host S.UInt8 n in
  let h1 = B.create ~host:true d S.UInt8 n
  and h2 = B.create ~host:true d S.UInt8 n in
  let lent1 = B.create Nx_device.host S.UInt8 n
  and lent2 = B.create Nx_device.host S.UInt8 n in
  let b1 = B.borrow d lent1 and b2 = B.borrow d lent2 in
  let v = B.create d S.UInt8 n in
  for round = 1 to 8 do
    let at what = Printf.sprintf "%s, round %d" what round in
    write_pattern (10 + round) written;
    B.copy ~src:written ~dst:h1;
    B.copy ~src:h1 ~dst:v;
    B.copy ~src:v ~dst:h2;
    check ~msg:(at "the GPU's host memory and VRAM") (10 + round) h2;
    write_pattern (20 + round) lent1;
    B.copy ~src:b1 ~dst:v;
    B.copy ~src:v ~dst:b2;
    Nx_device.synchronize Nx_device.host;
    check ~msg:(at "borrowed memory and VRAM") (20 + round) lent2
  done

(* Last: work that never signals hangs the device, which fails after its
   timeout. *)
let test_hang () =
  let d = device () in
  Nx_device.set_timeout d 500;
  ignore (Nx_device.submit d ~touches:[] Fun.id);
  raises_match (Exn.failure ~substring:"hang detected") (fun () ->
      Nx_device.synchronize d);
  raises_match (Exn.failure ~substring:"hang detected") (fun () ->
      B.create d S.UInt8 1)

let () =
  exit
    (run "nx.nv.device (hardware)"
       [
         group "devices" [ test "open" test_open ];
         group "memory"
           [
             test "memory and copies" test_memory;
             test "staged copies" test_staged;
             test "borrows" test_borrow;
             test "one buffer borrowed by two GPUs" test_two_borrows;
             test "two GPUs" test_peer;
           ];
         group "boundaries"
           [
             test "every route around 64 MiB copies" (boundaries (64 * mib));
             test "every route around 2 GiB copies" (boundaries (2048 * mib));
             test "two GPUs around 64 MiB copies" (peer_boundaries (64 * mib));
             test "two GPUs around 2 GiB copies" (peer_boundaries (2048 * mib));
             test "copies that wrap the copy channel" test_ring_wraps;
             test "host writes, GPU copies, host reads" test_coherence;
           ];
         group "programs"
           [
             test "cubins" test_programs; test "local memory" test_local_memory;
           ];
         group "timeline" [ test "across 2^32" test_carry ];
         group "nx" (Nx_test.Runtimes.laws gpus);
         group "failures"
           [ test "work that never signals fails the device" test_hang ];
       ])
