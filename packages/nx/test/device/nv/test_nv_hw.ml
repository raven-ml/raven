(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* NV devices on a real GPU: opening, memory and its budget, every copy route
   with the timeline steps that identify it, borrows, one host buffer borrowed
   by two GPUs, peer copies, every route around the staging slot and the copy
   engine's longest line, copies that wrap the copy channel, coherence through
   each kind of memory, programs and local memory, nx's runtime laws over the
   devices, and last, work that never signals, which loses the device. Every
   test skips without a GPU, and the peer tests without two. GPUs are reached
   through the kernel driver when it is loaded; over PCI the runtime takes a GPU
   from its kernel driver, so the suite does only when NX_NV_PCI_TEST names the
   index of a GPU it may take. *)

open Windtrap
module B = Nx_device.Buffer
module S = Nx_dtype.Scalar

let mib = 1 lsl 20

(* The host address of [b]'s first byte, if the host addresses [b]'s memory. *)
let hosted b =
  let r = Nx_device.Driver.Region.of_buffer b in
  Option.map
    (fun a -> Nativeint.add a (Nativeint.of_int (B.offset b)))
    (Nx_device.Driver.Region.host_address r)

(* [d]'s borrow of [b], which it maps. *)
let borrow d b = match B.borrow d b with Ok b -> b | Error why -> failwith why

(* The function [name] of [binary], which [d] loads. *)
let program d ~binary ~name =
  match Nx_device.Program.load d ~binary ~name with
  | Ok p -> p
  | Error why -> failwith why

let hung d = function
  | Nx_device.Lost (d', why) -> d' == d && why = "hang detected"
  | _ -> false

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

(* The GPUs nx's runtime laws run on: over PCI, the one the suite may take and
   the next, if there is one. *)
let gpus =
  match pci_first () with
  | Some first ->
      List.filter_map
        (fun i -> Result.to_option (Nx_nv_device.get ~interface:Pci i))
        [ first; first + 1 ]
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

let low d = Option.get (Nx_nv_device.of_device d)

let test_open () =
  let d = device () in
  is_true ~msg:"memoized" (Nx_device.equal d (device ()));
  starts_with ~msg:"name" ~affix:"NV" (Nx_device.name d);
  starts_with ~msg:"arch" ~affix:"sm_" (Nx_device.arch d);
  let n = low d in
  let props = Nx_nv_device.props n in
  is_true ~msg:"multiprocessors"
    (props.gpcs * props.tpcs_per_gpc * props.sms_per_tpc > 0);
  is_true ~msg:"warps" (props.warps_per_sm > 0);
  is_true ~msg:"the signal word, the device's pinned memory"
    (Nx_device.equal d (B.device (Nx_device.signal_word d)));
  let below_2_40 (c : Nx_nv_device.channel) =
    Nativeint.to_int (B.address c.ring) < 1 lsl 40
  in
  is_true ~msg:"channel rings below 2^40"
    (below_2_40 (Nx_nv_device.compute n) && below_2_40 (Nx_nv_device.copy n));
  is_true ~msg:"a budget" (Nx_device.budget d > 0);
  Nx_nv_device.invalidate_caches n

let test_memory () =
  let d = device () in
  let n = (3 * mib) + 17 in
  let src = fill_host n (fun i -> i * 7) in
  let v = B.create d S.UInt8 n in
  equal ~msg:"VRAM the host does not address" (option nativeint) None (hosted v);
  equal ~msg:"into VRAM directly from aligned host memory" int 1
    (steps d (fun () -> B.copy ~src ~dst:v));
  let back = B.create Nx_device.host S.UInt8 n in
  equal ~msg:"out" int 1 (steps d (fun () -> B.copy ~src:v ~dst:back));
  is_true ~msg:"round trip" (same_bytes src back);
  let pinned = B.create ~pinned:true d S.UInt8 n in
  is_true ~msg:"pinned memory the host addresses" (hosted pinned <> None);
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
  let borrowed = borrow d host in
  is_true ~msg:"borrowed" (B.is_borrowed borrowed);
  let v = B.create d S.UInt8 n in
  let src = fill_host n (fun i -> 255 - i) in
  B.copy ~src ~dst:v;
  equal ~msg:"into borrowed memory directly" int 1
    (steps d (fun () -> B.copy ~src:v ~dst:borrowed));
  Nx_device.synchronize Nx_device.host;
  is_true ~msg:"seen through the host buffer" (same_bytes src host);
  let also = borrow d (B.view host ~offset:(64 * 1024) S.UInt8 1024) in
  B.copy ~src:(B.view v ~offset:0 S.UInt8 1024) ~dst:also;
  match
    B.borrow d
      (B.of_bigarray
         (Bigarray.Array1.sub
            (B.bigarray Bigarray.int8_unsigned
               (B.create Nx_device.host S.UInt8 (2 * mib)))
            1 mib))
  with
  | Ok _ -> fail "borrowed memory off a page"
  | Error why -> contains ~msg:"off a page" ~sub:"does not start on a page" why

(* The kilobytes of the process's memory Linux keeps locked. *)
let locked_kib () =
  In_channel.with_open_text "/proc/self/status" In_channel.input_lines
  |> List.find_map (fun l ->
      match String.split_on_char ':' l with
      | [ "VmLck"; v ] ->
          int_of_string_opt
            (String.trim (Filename.chop_suffix (String.trim v) "kB"))
      | _ -> None)
  |> Option.get

(* Copies between two GPUs, in both directions, whether they reach each other or
   bounce through host memory, and through each one's host memory: the other
   GPU's pages stay locked. *)
let test_peer () =
  let d0 = device () and d1 = device ~i:1 () in
  let n = 2 * mib in
  let src = fill_host n (fun i -> i * 3) in
  let a = B.create d0 S.UInt8 n and b = B.create d1 S.UInt8 n in
  B.copy ~src ~dst:a;
  equal ~msg:"one step on the source" int 1
    (steps d0 (fun () -> B.copy ~src:a ~dst:b));
  is_true ~msg:"the first GPU into the second" (same_bytes src (to_host b));
  let src' = fill_host n (fun i -> i * 7) in
  B.copy ~src:src' ~dst:b;
  equal ~msg:"one step on the source" int 1
    (steps d1 (fun () -> B.copy ~src:b ~dst:a));
  is_true ~msg:"the second GPU into the first" (same_bytes src' (to_host a));
  B.copy ~src ~dst:a;
  let h1 = B.create ~pinned:true d1 S.UInt8 n in
  let locked = locked_kib () in
  B.copy ~src:a ~dst:h1;
  equal ~msg:"the other GPU's host memory stays locked" int locked
    (locked_kib ());
  let back = B.create Nx_device.host S.UInt8 n in
  B.copy ~src:h1 ~dst:back;
  is_true ~msg:"into the other GPU's host memory" (same_bytes src back);
  let h0 = B.create ~pinned:true d0 S.UInt8 n in
  B.copy ~src:src' ~dst:h0;
  B.copy ~src:h0 ~dst:b;
  is_true ~msg:"out of the other GPU's host memory"
    (same_bytes src' (to_host b));
  (* a failed device raises its error here *)
  Nx_device.synchronize d0;
  Nx_device.synchronize d1;
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
    B.copy ~src:(borrow d host) ~dst:v;
    to_host v
  in
  is_true ~msg:"the first GPU" (same_bytes host (read d0));
  is_true ~msg:"the second GPU" (same_bytes host (read d1));
  Gc.full_major ();
  is_true ~msg:"the second once the first let go" (same_bytes host (read d1));
  is_true ~msg:"the first again" (same_bytes host (read d0))

(* Kernels compiled at test time, when NVIDIA's compiler is at hand: [fill]
   writes through a local array, so that it uses local memory; [small] needs few
   registers and one parameter, [big] many registers, a stack and three
   parameters. *)
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
             s[threadIdx.x % 64] = 42; p[threadIdx.x] = s[threadIdx.x % 64]; }\n\
             extern \"C\" __global__ void small(int *p) { p[threadIdx.x] = 1; }\n\
             extern \"C\" __global__ void big(float *p, float *q, int n) { \
             float a[32]; for (int i = 0; i < 32; i++) a[i] = p[i * n + \
             threadIdx.x]; float s = 0; for (int i = 0; i < 32; i++) for (int \
             j = 0; j < 32; j++) s += a[i] * a[j]; volatile float t[64]; \
             t[threadIdx.x % 64] = s; q[threadIdx.x] = t[(threadIdx.x + 1) % \
             64]; }\n");
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
      let p = program d ~binary ~name:"fill" in
      is_true ~msg:"cached" (p == program d ~binary ~name:"fill");
      let k = Option.get (Nx_nv_device.kernel p) in
      equal ~msg:"the handle is the entry" nativeint k.entry
        (Nx_device.Program.handle p);
      is_true ~msg:"inside the image" (k.entry >= k.image);
      is_true ~msg:"registers" (k.registers > 0);
      is_true ~msg:"threads" (k.max_threads >= 32);
      (match Nx_device.Program.load d ~binary ~name:"absent" with
      | Ok _ -> fail "loaded an absent function"
      | Error why -> contains ~msg:"refused" ~sub:"no function" why);
      (* Each function of a cubin of several has its own registers, stack and
         bank 0, whichever comes last in the cubin. *)
      let kernel name =
        Option.get (Nx_nv_device.kernel (program d ~binary ~name))
      in
      let small = kernel "small" and big = kernel "big" in
      let bank0 (k : Nx_nv_device.kernel) =
        List.find_map (fun (i, _, n) -> if i = 0 then Some n else None) k.banks
      in
      is_true ~msg:"registers of their own" (small.registers < big.registers);
      is_true ~msg:"stacks of their own" (small.local_bytes < big.local_bytes);
      is_true ~msg:"banks 0 of their own" (bank0 small < bank0 big)

let test_local_memory () =
  let d = device () in
  equal ~msg:"none needed, no work" int 0
    (steps d (fun () -> ignore (Nx_nv_device.local_memory (low d) 0)));
  let l = Nx_nv_device.local_memory (low d) 256 in
  is_true ~msg:"memory" (l.address <> 0n && l.bytes > 0);
  is_true ~msg:"per thread" (l.per_thread >= 256 && l.per_thread mod 32 = 0);
  let again = Nx_nv_device.local_memory (low d) 128 in
  equal ~msg:"kept for smaller kernels" nativeint l.address again.address;
  let more = Nx_nv_device.local_memory (low d) 4096 in
  is_true ~msg:"grown for larger ones" (more.bytes > l.bytes);
  Nx_device.synchronize d

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
  let borrowed = borrow d lent in
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
  let entries = B.length (Nx_nv_device.copy (low d)).ring in
  let copies = (3 * entries) + 17 in
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
  let h1 = B.create ~pinned:true d S.UInt8 n
  and h2 = B.create ~pinned:true d S.UInt8 n in
  let lent1 = B.create Nx_device.host S.UInt8 n
  and lent2 = B.create Nx_device.host S.UInt8 n in
  let b1 = borrow d lent1 and b2 = borrow d lent2 in
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

(* Allocations up to the budget's end and back: an allocation the driver refuses
   partway leaves its addresses free, so the memory freed is allocated again and
   holds its bytes. *)
let test_exhaustion () =
  let d = device () in
  let rec fill size held =
    if size < 64 * 1024 then held
    else
      match B.create d S.UInt8 size with
      | b -> fill size (b :: held)
      | exception Nx_device.Out_of_memory _ -> fill (size / 2) held
  in
  let held = fill (256 * mib) [] in
  is_true ~msg:"memory was allocated" (held <> []);
  ignore (Sys.opaque_identity held);
  Gc.full_major ();
  let n = 64 * mib in
  let src = fill_host n (fun i -> i * 9) and v = B.create d S.UInt8 n in
  B.copy ~src ~dst:v;
  is_true ~msg:"the bytes" (same_bytes src (to_host v))

(* Last: work that never signals hangs the device, which is lost after its
   timeout, having slept: the wait's checks for faults do not keep a core
   busy. *)
let test_hang () =
  let d = device () in
  Nx_device.set_timeout d 2000;
  Nx_device.submit [ d ] ~touches:[] ignore;
  let cpu = Sys.time () in
  raises_match (hung d) (fun () -> Nx_device.synchronize d);
  is_true ~msg:"most of the 2 s asleep" (Sys.time () -. cpu < 1.);
  raises_match (hung d) (fun () -> B.create d S.UInt8 1)

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
         group "profiles" (Nx_test.Profiles.copies ~slack:1_000_000 gpus);
         group "nx" (Nx_test.Runtimes.laws gpus);
         group "failures"
           [
             test "allocations up to the budget's end" test_exhaustion;
             test "work that never signals loses the device" test_hang;
           ];
       ])
