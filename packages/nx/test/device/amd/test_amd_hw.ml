(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* AMD devices on a real GPU: opening, memory and its budget, every copy route
   with the timeline steps that identify it, borrows, peer copies, programs,
   scratch, the timeline across 2^32, nx's runtime laws over the devices, and
   last, work that never signals, which fails the device. Every test skips
   without a GPU, and the peer tests without two. GPUs are reached through the
   kernel driver when it is loaded; over PCI the runtime takes a GPU from its
   kernel driver, so the suite does only when NX_AMD_PCI_TEST names the index of
   a GPU it may take. *)

open Windtrap
module B = Nx_device.Buffer
module S = Nx_dtype.Scalar

let mib = 1 lsl 20

(* The index of the first GPU the suite may take over PCI, if any. *)
let pci_first () = Option.map int_of_string (Sys.getenv_opt "NX_AMD_PCI_TEST")

(* The GPU [i] under test, or a skip. *)
let device ?(i = 0) () =
  match pci_first () with
  | Some first -> (
      match Nx_amd_device.get ~interface:Pci (first + i) with
      | Ok d -> d
      | Error msg -> skip ~reason:msg ())
  | None ->
      if Nx_amd_device.count ~interface:Kernel () <= i then
        skip
          ~reason:
            "no AMD GPU through the kernel driver; set NX_AMD_PCI_TEST=i to \
             take GPU i over PCI"
          ()
      else Nx_amd_device.v ~interface:Kernel i

(* The GPUs nx's runtime laws run on. *)
let gpus =
  match pci_first () with
  | Some first -> (
      match Nx_amd_device.get ~interface:Pci first with
      | Ok d -> [ d ]
      | Error _ -> [])
  | None ->
      List.init
        (Nx_amd_device.count ~interface:Kernel ())
        (Nx_amd_device.v ~interface:Kernel)

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
  starts_with ~msg:"name" ~affix:"AMD" (Nx_device.name d);
  starts_with ~msg:"arch" ~affix:"gfx" (Nx_device.arch d);
  let h = Nx_amd_device.handles d in
  is_true ~msg:"compute units" (h.props.compute_units > 0);
  is_true ~msg:"XCCs" (h.props.xccs >= 1);
  is_true ~msg:"LDS" (h.props.lds_bytes >= 32 * 1024);
  is_true ~msg:"an SDMA queue" (h.sdma <> []);
  equal ~msg:"AQL on several XCCs" bool (h.props.xccs > 1) h.aql;
  equal ~msg:"the signal is the timeline's" nativeint
    (B.address (Nx_device.timeline d))
    h.signal;
  is_true ~msg:"a budget" (Nx_device.budget d > 0)

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

(* A kernel compiled at test time, when a compiler for the GPU's target is at
   hand. *)
let compile arch =
  let clang =
    List.find_opt Sys.file_exists
      [ "/opt/rocm/llvm/bin/clang"; "/usr/bin/clang"; "/usr/local/bin/clang" ]
  in
  match clang with
  | None -> None
  | Some clang ->
      let dir = Filename.temp_dir "nx-amd" "" in
      let src = Filename.concat dir "k.cl"
      and out = Filename.concat dir "k.co" in
      Out_channel.with_open_text src (fun oc ->
          output_string oc
            "kernel void fill(global int *p) { \
             p[__builtin_amdgcn_workitem_id_x()] = 42; }\n");
      let cmd =
        Printf.sprintf
          "%s -x cl -cl-std=CL2.0 -target amdgcn-amd-amdhsa -mcpu=%s -nogpulib \
           -O2 %s -o %s 2>/dev/null"
          clang arch src out
      in
      if Sys.command cmd = 0 then
        Some (In_channel.with_open_bin out In_channel.input_all)
      else None

let test_programs () =
  let d = device () in
  match compile (Nx_device.arch d) with
  | None -> skip ~reason:"no compiler for the GPU's target" ()
  | Some binary ->
      let p = Nx_device.Program.load d ~binary ~name:"fill" in
      is_true ~msg:"cached" (p == Nx_device.Program.load d ~binary ~name:"fill");
      let k = Nx_amd_device.kernel p in
      equal ~msg:"the handle is the descriptor" nativeint k.descriptor
        (Nx_device.Program.handle p);
      is_true ~msg:"inside the code"
        (k.descriptor >= k.code && k.entry >= k.code);
      is_true ~msg:"room for its pointer" (k.kernarg_segment >= 8);
      raises_match (Exn.failure ~substring:"no kernel") (fun () ->
          Nx_device.Program.load d ~binary ~name:"absent")

(* The device's timeline across 2^32: from 2^32 - 2, three copies take it
   through the carry, and the copy after it waits for a low word of 0. *)
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

let test_scratch () =
  let d = device () in
  let s = Nx_amd_device.scratch d 256 in
  is_true ~msg:"memory" (s.address <> 0n && s.bytes > 0);
  is_true ~msg:"a ring size" (s.tmpring_size <> 0);
  let again = Nx_amd_device.scratch d 128 in
  equal ~msg:"kept for smaller kernels" nativeint s.address again.address;
  let more = Nx_amd_device.scratch d 4096 in
  is_true ~msg:"grown for larger ones" (more.bytes > s.bytes)

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
         group "programs"
           [ test "code objects" test_programs; test "scratch" test_scratch ];
         group "timeline" [ test "across 2^32" test_carry ];
         group "nx" (Nx_test.Runtimes.laws gpus);
         group "failures"
           [ test "work that never signals fails the device" test_hang ];
       ])
