(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* AMD devices on a real GPU: opening, memory and its budget, every copy route
   with the timeline steps that identify it, borrows, peer copies, every route
   around the copy engine's largest copy, copies that wrap its ring, coherence
   through each kind of memory, programs, scratch, nx's runtime laws over the
   devices, and last, work that never signals, which loses the device. Every
   test skips without a GPU, and the peer tests without two. GPUs are reached
   through the kernel driver when it is loaded; over PCI the suite detaches a
   GPU from its kernel driver, unless it is detached already, and resets it, so
   it does only when NX_AMD_PCI_TEST names the index of a GPU it may take. *)

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
let pci_first () = Option.map int_of_string (Sys.getenv_opt "NX_AMD_PCI_TEST")

(* The number of GPUs the kernel driver may hold: none without it. *)
let kernel_gpus () =
  if Sys.file_exists "/dev/kfd" then Nx_amd_device.count () else 0

(* GPU [i] over PCI, prepared as an open needs it: reset, if it is detached
   already, as when an administrator bound it to vfio-pci, whose firmware was
   fetched before; otherwise given back to its kernel driver, which names its
   firmware, the firmware fetched, then detached and reset. Once per process,
   since an open GPU refuses them. *)
let pci_opens = Hashtbl.create 2

let open_pci i =
  match Hashtbl.find_opt pci_opens i with
  | Some r -> r
  | None ->
      let ( let* ) = Result.bind in
      let prepare () =
        let* () = Nx_amd_device.attach i in
        let* () = Nx_amd_device.fetch_firmware i in
        let* () = Nx_amd_device.detach i in
        Nx_amd_device.reset i
      in
      let r =
        let* () =
          match Nx_amd_device.reset i with
          | Ok () -> Ok ()
          | Error _ -> prepare ()
        in
        Nx_amd_device.get ~interface:Pci i
      in
      Hashtbl.replace pci_opens i r;
      r

(* The GPU [i] under test, or a skip. *)
let device ?(i = 0) () =
  match pci_first () with
  | Some first -> (
      match open_pci (first + i) with
      | Ok d -> d
      | Error msg -> skip ~reason:msg ())
  | None -> (
      if kernel_gpus () <= i then
        skip
          ~reason:
            "no AMD GPU through the kernel driver; set NX_AMD_PCI_TEST=i to \
             take GPU i over PCI"
          ()
      else
        match Nx_amd_device.get ~interface:Kernel i with
        | Ok d -> d
        | Error e -> failwith e)

(* The GPUs nx's runtime laws run on. *)
let gpus =
  match pci_first () with
  | Some first -> ( match open_pci first with Ok d -> [ d ] | Error _ -> [])
  | None ->
      List.init (kernel_gpus ()) (fun i ->
          match Nx_amd_device.get ~interface:Kernel i with
          | Ok d -> d
          | Error e -> failwith e)

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

let low d = Option.get (Nx_amd_device.of_device d)

let test_open () =
  let d = device () in
  is_true ~msg:"memoized" (Nx_device.equal d (device ()));
  let name =
    match pci_first () with
    | Some 0 -> "AMD-PCI"
    | Some i -> Printf.sprintf "AMD-PCI:%d" i
    | None -> "AMD"
  in
  equal ~msg:"the name of GPU 0 or of the GPU the suite may take" string name
    (Nx_device.name d);
  starts_with ~msg:"arch" ~affix:"gfx" (Nx_device.arch d);
  let a = low d in
  let props = Nx_amd_device.props a in
  is_true ~msg:"compute units" (props.compute_units > 0);
  is_true ~msg:"XCCs" (props.xccs >= 1);
  is_true ~msg:"LDS" (props.lds_bytes >= 32 * 1024);
  is_true ~msg:"an SDMA queue" (Nx_amd_device.sdma a <> []);
  equal ~msg:"AQL on several XCCs" bool (props.xccs > 1) (Nx_amd_device.aql a);
  is_true ~msg:"the signal word, the device's pinned memory"
    (Nx_device.equal d (B.device (Nx_device.signal_word d)));
  is_true ~msg:"a budget" (Nx_device.budget d > 0)

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
  let pinned = B.create ~memory:Pinned d S.UInt8 n in
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
  let h1 = B.create ~memory:Pinned d1 S.UInt8 n in
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

(* The OpenCL source [src], compiled at test time for the processor [arch] in a
   code object of version 6, when a compiler for it is at hand: an object the
   load relocates, so no linker is needed. *)
let compile_source src arch =
  let clang =
    List.find_opt Sys.file_exists
      [ "/opt/rocm/llvm/bin/clang"; "/usr/bin/clang"; "/usr/local/bin/clang" ]
  in
  match clang with
  | None -> None
  | Some clang ->
      let dir = Filename.temp_dir "nx-amd" "" in
      let out = Filename.concat dir "k.co" in
      Out_channel.with_open_text (Filename.concat dir "k.cl") (fun oc ->
          output_string oc src);
      let cmd =
        Printf.sprintf
          "%s -c -x cl -cl-std=CL2.0 -target amdgcn-amd-amdhsa -mcpu=%s \
           -mcode-object-version=6 -nogpulib -O2 %s -o %s 2>/dev/null"
          clang arch
          (Filename.concat dir "k.cl")
          out
      in
      if Sys.command cmd = 0 then
        Some (In_channel.with_open_bin out In_channel.input_all)
      else None

(* A kernel that stores [value] at each work-item's index. *)
let compile ?(value = 42) arch =
  compile_source
    (Printf.sprintf
       "kernel void fill(global int *p) { p[__builtin_amdgcn_workitem_id_x()] \
        = %d; }\n"
       value)
    arch

let test_programs () =
  let d = device () in
  match compile (Nx_device.arch d) with
  | None -> skip ~reason:"no compiler for the GPU's target" ()
  | Some binary -> (
      let p = program d ~binary ~name:"fill" in
      equal ~msg:"loaded once" nativeint
        (Nx_device.Program.handle p)
        (Nx_device.Program.handle (program d ~binary ~name:"fill"));
      let k = Option.get (Nx_amd_device.kernel p) in
      equal ~msg:"the handle is the descriptor" nativeint k.descriptor
        (Nx_device.Program.handle p);
      let code = B.address k.code
      and end_ =
        Nativeint.add (B.address k.code) (Nativeint.of_int (B.nbytes k.code))
      in
      is_true ~msg:"inside the code"
        (k.descriptor >= code && k.descriptor < end_);
      equal ~msg:"on the device" bool true (Nx_device.equal (B.device k.code) d);
      equal ~msg:"in memory the host does not address" (option nativeint) None
        (hosted k.code);
      match Nx_device.Program.load d ~binary ~name:"absent" with
      | Ok _ -> fail "loaded an absent function"
      | Error why -> contains ~msg:"refused" ~sub:"no kernel" why)

(* A code object for another GPU is refused, naming both, and one for the
   generic processor of the GPU's generation, as LLVM's AMDGPUUsage lists them,
   loads. *)
let test_targets () =
  let d = device () in
  let gpu = Nx_device.arch d in
  let other = if gpu = "gfx1100" then "gfx1201" else "gfx1100" in
  let generic =
    if String.starts_with ~prefix:"gfx12" gpu then "gfx12-generic"
    else if String.starts_with ~prefix:"gfx11" gpu then "gfx11-generic"
    else "gfx9-4-generic"
  in
  match (compile other, compile generic) with
  | None, _ | _, None -> skip ~reason:"no compiler for the GPU's targets" ()
  | Some foreign, Some binary ->
      (match Nx_device.Program.load d ~binary:foreign ~name:"fill" with
      | Ok _ -> fail "loaded a code object for another GPU"
      | Error why ->
          equal string
            (Printf.sprintf "%s: a code object for %s; the GPU is %s"
               (Nx_device.name d) other gpu)
            why);
      ignore (program d ~binary ~name:"fill")

(* A launch of the kernel that stores [value] at each work-item's index: the
   whole workgroup's elements written once the launch's work is done, and the
   launches past the device's limits refused before any work. *)
let test_launch () =
  let d = device () in
  match compile ~value:7 (Nx_device.arch d) with
  | None -> skip ~reason:"no compiler for the GPU's target" ()
  | Some binary ->
      let program = program d ~binary ~name:"fill" in
      let b = B.create d S.Int32 64 in
      let args = Bytes.create 8 in
      Bytes.set_int64_le args 0 (Int64.of_nativeint (B.address b));
      let dispatch ?(threads = (64, 1, 1)) ?(args = Bytes.to_string args) () =
        { Nx_amd_device.program; groups = (1, 1, 1); threads; args }
      in
      let before = Nx_device.submitted d in
      Nx_amd_device.launch ~touches:[ b ] [ dispatch () ];
      equal ~msg:"one timeline value" int (before + 1) (Nx_device.submitted d);
      let h = B.create Nx_device.host S.Int32 64 in
      B.copy ~src:b ~dst:h;
      let h = B.bigarray Bigarray.int32 h in
      equal (array int32) (Array.make 64 7l) (Array.init 64 (fun i -> h.{i}));
      let read = Nx_device.submitted d in
      let refused f = raises_match Exn.invalid_arg (fun () -> f ()) in
      refused (fun () -> Nx_amd_device.launch ~touches:[ b ] []);
      refused (fun () ->
          Nx_amd_device.launch ~touches:[ b ]
            [ dispatch ~threads:(1025, 1, 1) () ]);
      refused (fun () ->
          Nx_amd_device.launch ~touches:[ b ]
            [ dispatch ~args:(String.make 65536 '\000') () ]);
      equal ~msg:"no work for a refusal" int read (Nx_device.submitted d)

(* What a launch allocates once its arguments' ring has wrapped many times:
   bounded, whatever the launches before it. A launch's own bookkeeping takes
   about 2,500 words; one that scanned every segment the ring holds took
   14,000. *)
let launch_words = 5000

let test_launch_allocation () =
  let d = device () in
  match compile (Nx_device.arch d) with
  | None -> skip ~reason:"no compiler for the GPU's target" ()
  | Some binary ->
      let program = program d ~binary ~name:"fill" in
      let b = B.create d S.Int32 64 in
      let args = Bytes.create 8 in
      Bytes.set_int64_le args 0 (Int64.of_nativeint (B.address b));
      let ds =
        [
          {
            Nx_amd_device.program;
            groups = (1, 1, 1);
            threads = (64, 1, 1);
            args = Bytes.to_string args;
          };
        ]
      in
      (* Far more launches than the ring of arguments holds segments. *)
      for _ = 1 to 3 * 4096 do
        Nx_amd_device.launch ~touches:[ b ] ds
      done;
      let launches = 100 in
      let before = Gc.minor_words () in
      for _ = 1 to launches do
        Nx_amd_device.launch ~touches:[ b ] ds
      done;
      let per_launch = (Gc.minor_words () -. before) /. float_of_int launches in
      less ~msg:"words a launch allocates" float_exact
        ~than:(float_of_int launch_words)
        per_launch

(* A load waits for the copy of its code, which takes microseconds: the median
   of fresh loads stays under [load_ms], which a wait that sleeps on the
   kernel's timer, for a tick or more, exceeds. *)
let load_ms = 2.

let test_load_time () =
  let d = device () in
  let binaries =
    List.filter_map
      (fun value -> compile ~value (Nx_device.arch d))
      [ 1; 2; 3; 4; 5 ]
  in
  if binaries = [] then skip ~reason:"no compiler for the GPU's target" ();
  let ms binary =
    let t0 = Nx_device.Profile.now () in
    ignore (program d ~binary ~name:"fill");
    Float.of_int (Nx_device.Profile.now () - t0) /. 1e6
  in
  let times = List.sort Float.compare (List.map ms binaries) in
  less ~msg:"median load, ms" float_exact ~than:load_ms
    (List.nth times (List.length times / 2))

let test_scratch () =
  let d = device () in
  let s = Nx_amd_device.scratch (low d) 256 in
  is_true ~msg:"memory of the device"
    (B.nbytes s > 0 && Nx_device.equal (B.device s) d);
  is_true ~msg:"kept for smaller kernels"
    (Nx_amd_device.scratch (low d) 128 == s);
  let more = Nx_amd_device.scratch (low d) 4096 in
  is_true ~msg:"grown for larger ones" (B.nbytes more > B.nbytes s)

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

(* The sizes around a largest linear copy of [max] bytes, which is 4 MiB or 1
   GiB by the copy engine's version: one under, at, one over, and two copies and
   a tail. *)
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

(* Every route at every size around [max]: direct from host memory the device
   borrows, staged from and to host memory it does not, VRAM to VRAM at offsets,
   and into and out of borrowed memory. *)
let boundaries max () =
  let d = device () in
  let top = room d max in
  let src = B.create Nx_device.host S.UInt8 top in
  write_pattern 1 src;
  let mapped = borrow d src in
  let staged = B.create Nx_device.host S.UInt8 top in
  write_pattern 1 staged;
  let out = B.create Nx_device.host S.UInt8 top in
  let lent = B.create Nx_device.host S.UInt8 top in
  let borrowed = borrow d lent in
  let v = B.create d S.UInt8 top and w = B.create d S.UInt8 top in
  List.iter
    (fun n ->
      let at what = Printf.sprintf "%s, %d bytes" what n in
      equal ~msg:(at "direct in") int 1
        (steps d (fun () -> B.copy ~src:(view mapped 0 n) ~dst:(view v 0 n)));
      B.copy ~src:(view v 0 n) ~dst:(view out 0 n);
      check ~msg:(at "direct in, staged out") 1 (view out 0 n);
      B.copy ~src:(view staged 1 n) ~dst:(view v 0 n);
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

(* Copies of 16 bytes one after the other, enough to wrap the SDMA ring three
   times, as each puts at least a wait, a copy, a fence and a trap on it, over
   64 bytes: every copy lands where it should. *)
let test_ring_wraps () =
  let d = device () in
  let ring = B.nbytes (List.hd (Nx_amd_device.sdma (low d))).ring in
  let copies = (3 * ring / 64) + 1 in
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
  let h1 = B.create ~memory:Pinned d S.UInt8 n
  and h2 = B.create ~memory:Pinned d S.UInt8 n in
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

(* Mapped memory: the host writes it through the BAR, and the copy engine reads
   what it wrote, a write-combined window flushed through the host data path
   before each copy. *)
let test_mapped () =
  let d = device () in
  let n = (2 * mib) + 4099 in
  let m = B.create ~memory:Mapped d S.UInt8 n in
  let host_view =
    match B.borrow Nx_device.host m with
    | Ok h -> h
    | Error why -> fail ("the host addresses mapped memory: " ^ why)
  in
  let v = B.create d S.UInt8 n
  and written = B.create Nx_device.host S.UInt8 n in
  for round = 1 to 8 do
    let at what = Printf.sprintf "%s, round %d" what round in
    write_pattern (30 + round) host_view;
    B.copy ~src:m ~dst:v;
    check ~msg:(at "the host's writes through the BAR") (30 + round) v;
    write_pattern (40 + round) written;
    B.copy ~src:written ~dst:v;
    B.copy ~src:v ~dst:m;
    Nx_device.synchronize d;
    check ~msg:(at "the GPU's writes, read by the host") (40 + round) host_view
  done

(* The power levels of the GPUs the amdgpu driver holds. *)
let power_levels () =
  Sys.readdir "/sys/class/drm"
  |> Array.to_list |> List.sort compare
  |> List.filter_map (fun n ->
      let f =
        Printf.sprintf
          "/sys/class/drm/%s/device/power_dpm_force_performance_level" n
      in
      if String.starts_with ~prefix:"renderD" n && Sys.file_exists f then
        Some (String.trim (In_channel.with_open_text f In_channel.input_all))
      else None)

(* A profile that counts or traces gives the device its profiling, the same for
   the same request in any later profile, laid out as the counters' blocks count
   them; a GPU without a counter refuses it by name. Under the kernel driver, a
   GPU past GFX9 is then in its stable power state. *)
let test_profiling () =
  let a = low (device ()) in
  let props = Nx_amd_device.props a in
  is_true ~msg:"no profile" (Option.is_none (Nx_amd_device.profiling a));
  let profiling ?counters ?trace () =
    let p = Nx_device.Profile.start ?counters ?trace () in
    Fun.protect
      ~finally:(fun () -> ignore (Nx_device.Profile.stop p))
      (fun () ->
        match Nx_amd_device.profiling a with
        | exception Failure why
          when String.ends_with ~suffix:"which another process holds" why ->
            skip ~reason:why ()
        | p -> Option.get p)
  in
  let p = profiling ~counters:[ "GRBM_GUI_ACTIVE"; "SQ_BUSY_CYCLES" ] () in
  (match (pci_first (), props.target, power_levels ()) with
  | None, (major, _, _), [ level ] when major > 9 ->
      equal string ~msg:"the stable power state" "profile_standard" level
  | _ -> ());
  let c = Option.get p.counting in
  equal int ~msg:"the log" (8 * (1 + (3 * p.slots))) (B.nbytes p.log);
  equal int ~msg:"the samples" (p.slots * c.size) (B.nbytes c.samples);
  is_true ~msg:"no tracing" (Option.is_none p.tracing);
  (match c.counters with
  | [ grbm; sq ] ->
      equal int ~msg:"one GRBM value per die" 0 grbm.offset;
      equal int ~msg:"the SQ's after" (8 * props.xccs) sq.offset;
      equal int ~msg:"an SQ value per engine" props.shader_engines sq.engines;
      equal int ~msg:"the samples' bytes"
        (8 * props.xccs * (1 + (sq.engines * sq.arrays * sq.wgps)))
        c.size
  | _ -> fail "two counters");
  let traced = profiling ~trace:true () in
  let t = Option.get traced.tracing in
  is_true ~msg:"no counting" (Option.is_none traced.counting);
  equal int ~msg:"every engine" (props.shader_engines * props.xccs) t.engines;
  equal int ~msg:"the traces"
    (traced.slots * t.engines * t.window)
    (B.nbytes t.traces);
  equal int ~msg:"the ends" (4 * traced.slots * t.engines) (B.nbytes t.ends);
  let before = Nx_device.stats (device ()) in
  let both = profiling ~counters:[ "GRBM_GUI_ACTIVE" ] ~trace:true () in
  let grown =
    Nx_device.Stats.(allocated (diff before (Nx_device.stats (device ()))))
  in
  is_true ~msg:"one trace set for the device"
    ((Option.get both.tracing).traces == t.traces);
  is_true ~msg:"a second traced request allocates no traces"
    (grown < B.nbytes t.traces);
  is_true ~msg:"kept for later profiles"
    (profiling ~counters:[ "GRBM_GUI_ACTIVE"; "SQ_BUSY_CYCLES" ] () == p);
  let p = Nx_device.Profile.start ~counters:[ "NO_SUCH_COUNTER" ] () in
  Fun.protect
    ~finally:(fun () -> ignore (Nx_device.Profile.stop p))
    (fun () ->
      raises_match (Exn.invalid_arg ~substring:"counts no NO_SUCH_COUNTER")
        (fun () -> Nx_amd_device.profiling a))

(* Failures *)

let long_seconds = 35

(* Spins one work-item until [long_seconds] have passed on the GPU's 100 MHz
   steady clock, then stores the seconds it spun. *)
let spin =
  Printf.sprintf
    "kernel void spin(global uint *p) { ulong t0 = \
     __builtin_readsteadycounter(); ulong t; do { t = \
     __builtin_readsteadycounter(); } while (t - t0 < %d00000000ul); p[0] = \
     (uint)((t - t0) / 100000000ul); }\n"
    long_seconds

(* Stores past the end of the GPU's address space. *)
let wild = "kernel void wild(global int *p) { p[1ul << 46] = 1; }\n"

(* Whether the suite takes its GPUs over PCI, where the process drives them. *)
let over_pci () = Option.is_some (pci_first ())

let run_one d ~binary ~name =
  let program = program d ~binary ~name in
  let b = B.create d S.UInt32 1 in
  let args = Bytes.create 8 in
  Bytes.set_int64_le args 0 (Int64.of_nativeint (B.address b));
  Nx_amd_device.launch ~touches:[ b ]
    [
      {
        Nx_amd_device.program;
        groups = (1, 1, 1);
        threads = (1, 1, 1);
        args = Bytes.to_string args;
      };
    ];
  b

(* Under the kernel driver, a kernel that runs for longer than any limit a wait
   had completes: the driver alone declares faults. *)
let test_long () =
  if over_pci () then skip ~reason:"the process drives the GPU" ();
  let d = device () in
  match compile_source spin (Nx_device.arch d) with
  | None -> skip ~reason:"no compiler for the GPU's target" ()
  | Some binary ->
      let t0 = Unix.gettimeofday () in
      let b = run_one d ~binary ~name:"spin" in
      Nx_device.synchronize d;
      at_least ~msg:"seconds waited" (float 0.1)
        ~than:(float_of_int long_seconds)
        (Unix.gettimeofday () -. t0);
      equal ~msg:"not lost" (option string) None (Nx_device.lost d);
      let h = to_host b in
      equal ~msg:"the seconds it spun" int32
        (Int32.of_int long_seconds)
        (B.bigarray Bigarray.int32 h).{0}

(* Last: under the kernel driver, a store the GPU's page tables refuse loses the
   device with the driver's report. *)
let test_fault () =
  if over_pci () then skip ~reason:"the process drives the GPU" ();
  let d = device () in
  match compile_source wild (Nx_device.arch d) with
  | None -> skip ~reason:"no compiler for the GPU's target" ()
  | Some binary ->
      ignore (run_one d ~binary ~name:"wild");
      let faulted = function
        | Nx_device.Lost (d', why) ->
            d' == d && String.starts_with ~prefix:"memory fault at" why
        | _ -> false
      in
      raises_match faulted (fun () -> Nx_device.synchronize d);
      raises_match faulted (fun () -> B.create d S.UInt8 1)

(* Last: over PCI, the process drives the GPU, and work that makes no progress
   for 30 seconds loses it. *)
let test_hang () =
  if not (over_pci ()) then
    skip ~reason:"the kernel driver alone reports faults" ();
  let d = device () in
  Nx_device.submit [ d ] ~touches:[] ignore;
  let t0 = Unix.gettimeofday () in
  raises_match (hung d) (fun () -> Nx_device.synchronize d);
  at_least ~msg:"seconds waited" (float 0.1) ~than:30.
    (Unix.gettimeofday () -. t0);
  raises_match (hung d) (fun () -> B.create d S.UInt8 1)

let () =
  exit
    (run "nx.amd.device (hardware)"
       [
         group "devices" [ test "open" test_open ];
         group "memory"
           [
             test "memory and copies" test_memory;
             test "staged copies" test_staged;
             test "borrows" test_borrow;
             test "two GPUs" test_peer;
           ];
         group "boundaries"
           [
             test "every route around 4 MiB copies" (boundaries (4 * mib));
             test "every route around 1 GiB copies" (boundaries (1024 * mib));
             test "two GPUs around 4 MiB copies" (peer_boundaries (4 * mib));
             test "two GPUs around 1 GiB copies" (peer_boundaries (1024 * mib));
             test "copies that wrap the SDMA ring" test_ring_wraps;
             test "host writes, GPU copies, host reads" test_coherence;
             test "mapped memory, written by the host and by the GPU"
               test_mapped;
           ];
         group "programs"
           [
             test "code objects" test_programs;
             test "code objects for another GPU" test_targets;
             test "launches" test_launch;
             test "what a launch allocates stays bounded" test_launch_allocation;
             test "loads wait for their copy alone" test_load_time;
             test "scratch" test_scratch;
           ];
         group "profiles"
           (test
              "a profile that counts or traces gives the device its profiling"
              test_profiling
           :: Nx_test.Profiles.copies ~slack:1_000_000 gpus);
         group "nx" (Nx_test.Runtimes.laws gpus);
         group "failures"
           [
             slow "a kernel that runs 35 s completes, and loses no device"
               test_long;
             test "a fault the driver reports loses the device" test_fault;
             slow "work that makes no progress for 30 s over PCI is a hang"
               test_hang;
           ];
       ])
