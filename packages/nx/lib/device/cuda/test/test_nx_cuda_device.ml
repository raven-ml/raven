(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* CUDA devices on a real GPU: opening, memory and its budget, every copy route,
   borrowing, memory of another library, programs, a kernel launched as timeline
   work, nx over the device, and a fault. Every test skips on a machine without
   a CUDA device, and the peer copy on one without two. *)

open Windtrap
module B = Nx_device.Buffer
module S = Nx_dtype.Scalar

external launch :
  nativeint -> nativeint -> nativeint -> nativeint -> nativeint -> int -> unit
  = "test_launch_byte" "test_launch"

external foreign_alloc : nativeint -> int -> nativeint = "test_alloc"

external foreign_host_alloc : nativeint -> int -> nativeint * nativeint
  = "test_host_alloc"

external other_context_alloc : int -> int -> nativeint
  = "test_other_context_alloc"

let cuda () =
  if Nx_cuda_device.count () = 0 then skip ~reason:"no CUDA device" ()
  else Nx_cuda_device.v 0

let mib = 1 lsl 20

(* More bytes than a page on every platform. *)
let pages = 1 lsl 16

type bytes_ba =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

let bytes_of b : bytes_ba = B.bigarray Bigarray.int8_unsigned b

(* A host buffer of [n] bytes holding [f i] at [i]. *)
let host_of n f =
  let b = B.create Nx_device.host S.UInt8 n in
  let ba = bytes_of b in
  for i = 0 to n - 1 do
    ba.{i} <- f i land 0xff
  done;
  b

(* Whether [b] holds [f i] at [i]. *)
let holds b f =
  let host = B.create Nx_device.host S.UInt8 (B.nbytes b) in
  B.copy ~src:b ~dst:host;
  let ba = bytes_of host in
  let rec go i = i = B.nbytes b || (ba.{i} = f i land 0xff && go (i + 1)) in
  go 0

let two () =
  if Nx_cuda_device.count () < 2 then skip ~reason:"one CUDA device" ()
  else (Nx_cuda_device.v 0, Nx_cuda_device.v 1)

(* The timeline values [f] used on [d]. *)
let values d f =
  let v = Nx_device.submitted d in
  f ();
  Nx_device.submitted d - v

let collect d =
  Gc.full_major ();
  ignore (Nx_device.stats d)

(* Opening *)

let test_open () =
  let d = cuda () in
  equal ~msg:"name" string "CUDA" (Nx_device.name d);
  is_true ~msg:"arch" (String.starts_with ~prefix:"sm_" (Nx_device.arch d));
  is_true ~msg:"budget" (Nx_device.budget d > 0);
  is_true ~msg:"memoised" (Nx_cuda_device.v 0 == d);
  let n = Nx_cuda_device.count () in
  (match Nx_cuda_device.get n with
  | Error msg -> is_true ~msg (String.starts_with ~prefix:"CUDA: no device" msg)
  | Ok _ -> fail "a device past the count");
  raises_match (Exn.invalid_arg ~substring:"not a CUDA device") (fun () ->
      Nx_cuda_device.handles Nx_device.host)

(* Memory *)

let test_memory () =
  let d = cuda () in
  let s0 = Nx_device.stats d in
  let b = B.create d S.Float32 250 in
  raises_match (Exn.invalid_arg ~substring:"does not address CUDA memory")
    (fun () -> B.host_address b);
  let p = B.create ~host:true d S.UInt8 100 in
  is_true ~msg:"the host addresses host memory" (B.host_address p <> 0n);
  equal ~msg:"both counted" int 1100
    (Nx_device.Stats.allocated (Nx_device.Stats.diff s0 (Nx_device.stats d)));
  ignore (Sys.opaque_identity (b, p))

let test_out_of_memory () =
  let d = cuda () in
  let budget = Nx_device.budget d in
  Fun.protect ~finally:(fun () -> Nx_device.set_budget d budget) @@ fun () ->
  raises_match
    (function Nx_device.Out_of_memory (d', _) -> d' == d | _ -> false)
    (fun () -> B.create d S.UInt8 (budget + 1));
  Nx_device.set_budget d max_int;
  raises_match
    (function Nx_device.Out_of_memory (d', _) -> d' == d | _ -> false)
    (fun () -> B.create d S.UInt8 (budget + 1))

(* Copies *)

let test_small_copies () =
  let d = cuda () in
  let s0 = Nx_device.stats d in
  let b = B.create d S.UInt8 7 in
  B.copy ~src:(host_of 7 (fun i -> i + 1)) ~dst:b;
  is_true ~msg:"round trip" (holds b (fun i -> i + 1));
  let v = B.view b ~offset:3 S.UInt8 3 in
  B.copy ~src:(host_of 3 (fun _ -> 9)) ~dst:v;
  is_true ~msg:"into a view"
    (holds b (fun i -> if i >= 3 && i < 6 then 9 else i + 1));
  B.copy ~src:(B.create d S.UInt8 0) ~dst:(B.create Nx_device.host S.UInt8 0);
  let s = Nx_device.Stats.diff s0 (Nx_device.stats d) in
  equal ~msg:"in" int 10 (Nx_device.Stats.bytes_in s);
  equal ~msg:"out" int 14 (Nx_device.Stats.bytes_out s)

(* Three chunks through the staging slots, each way, into a view at an
   offset. *)
let test_staged () =
  let d = cuda () in
  let n = (130 * mib) + 7 in
  let pattern i = (i * 7) + (i lsr 20) in
  let whole = B.create d S.UInt8 (n + 1000) in
  let b = B.view whole ~offset:1000 S.UInt8 n in
  equal ~msg:"three chunks in" int 3
    (values d (fun () -> B.copy ~src:(host_of n pattern) ~dst:b));
  let back = B.create Nx_device.host S.UInt8 n in
  equal ~msg:"three chunks out" int 3
    (values d (fun () -> B.copy ~src:b ~dst:back));
  let ba = bytes_of back in
  let rec go i = i = n || (ba.{i} = pattern i land 0xff && go (i + 1)) in
  is_true ~msg:"bytes" (go 0)

(* Page-locked memory, device memory, and mapped host memory, in one copy
   each. *)
let test_direct () =
  let d = cuda () in
  let n = 100 * mib in
  let pinned = B.create ~host:true d S.UInt8 n in
  B.copy ~src:(host_of n (fun i -> i * 3)) ~dst:pinned;
  let b = B.create d S.UInt8 n in
  equal ~msg:"from page-locked memory" int 1
    (values d (fun () -> B.copy ~src:pinned ~dst:b));
  let b' = B.create d S.UInt8 n in
  equal ~msg:"device to device" int 1
    (values d (fun () -> B.copy ~src:b ~dst:b'));
  let mapped = B.create Nx_device.host S.UInt8 n in
  let bm = B.borrow d mapped in
  equal ~msg:"into mapped host memory" int 1
    (values d (fun () -> B.copy ~src:b' ~dst:mapped));
  is_true ~msg:"bytes" (holds mapped (fun i -> i * 3));
  ignore (Sys.opaque_identity bm)

let test_peer () =
  let d, d1 = two () in
  equal ~msg:"name" string "CUDA:1" (Nx_device.name d1);
  let src = B.create d S.UInt8 1000 and dst = B.create d1 S.UInt8 1000 in
  B.copy ~src:(host_of 1000 (fun i -> i)) ~dst:src;
  equal ~msg:"one transfer by the source" int 1
    (values d (fun () -> B.copy ~src ~dst));
  is_true ~msg:"bytes" (holds dst (fun i -> i));
  let pinned = B.create ~host:true d1 S.UInt8 1000 in
  B.copy ~src:dst ~dst:pinned;
  let on_d = B.create d S.UInt8 1000 in
  equal ~msg:"from another GPU's page-locked memory, directly" int 1
    (values d (fun () -> B.copy ~src:pinned ~dst:on_d));
  is_true ~msg:"its bytes" (holds on_d (fun i -> i))

(* One host buffer borrowed on two GPUs shares one registration, released when
   both devices' borrows are gone. *)
let test_borrow_two () =
  let d, d1 = two () in
  let hb = host_of (2 * pages) (fun i -> i) in
  let off_page = bytes_of hb in
  let overlap () = B.of_bigarray (Bigarray.Array1.sub off_page pages pages) in
  (fun () ->
    let on_d = B.borrow d hb and on_d1 = B.borrow d1 hb in
    let a = B.create d S.UInt8 16 and b = B.create d1 S.UInt8 16 in
    B.copy ~src:(B.view on_d ~offset:0 S.UInt8 16) ~dst:a;
    B.copy ~src:(B.view on_d1 ~offset:pages S.UInt8 16) ~dst:b;
    is_true ~msg:"on the first" (holds a (fun i -> i));
    is_true ~msg:"on the second" (holds b (fun i -> pages + i)))
    ();
  collect d;
  raises_match (Exn.invalid_arg ~substring:"cannot map") (fun () ->
      B.borrow d (overlap ()));
  collect d1;
  is_true ~msg:"unregistered with the last device's borrows"
    (B.is_borrowed (B.borrow d (overlap ())))

(* Two domains copy to two GPUs at once. *)
let test_two_domains () =
  let d, d1 = two () in
  let copy_on dev k =
    Domain.spawn (fun () ->
        let b = B.create dev S.UInt8 (8 * mib) in
        B.copy ~src:(host_of (8 * mib) (fun i -> i + k)) ~dst:b;
        holds b (fun i -> i + k))
  in
  let a = copy_on d 1 and b = copy_on d1 2 in
  is_true ~msg:"first" (Domain.join a);
  is_true ~msg:"second" (Domain.join b)

(* Another domain runs on a thread of its own, where the device's context is
   made current. *)
let test_domains () =
  let d = cuda () in
  let b = B.create d S.UInt8 64 in
  let ok =
    Domain.join
      (Domain.spawn (fun () ->
           B.copy ~src:(host_of 64 (fun i -> 64 - i)) ~dst:b;
           holds b (fun i -> 64 - i)))
  in
  is_true ~msg:"copied from another domain" ok

(* Borrowing *)

let test_borrow () =
  let d = cuda () in
  let hb = host_of (2 * pages) (fun i -> i) in
  let off_page = bytes_of hb in
  let overlap () = B.of_bigarray (Bigarray.Array1.sub off_page pages pages) in
  (fun () ->
    let whole = B.borrow d hb in
    let tail = B.borrow d (B.view hb ~offset:pages S.UInt8 16) in
    equal ~msg:"the host's bytes" nativeint (B.host_address hb)
      (B.host_address whole);
    let b = B.create d S.UInt8 16 in
    B.copy ~src:tail ~dst:b;
    is_true ~msg:"read through the mapping" (holds b (fun i -> pages + i));
    raises_match (Exn.invalid_arg ~substring:"does not start on a page")
      (fun () -> B.borrow d (B.of_bigarray (Bigarray.Array1.sub off_page 1 16)));
    raises_match (Exn.invalid_arg ~substring:"cannot map") (fun () ->
        B.borrow d (overlap ())))
    ();
  collect d;
  let again = B.borrow d (overlap ()) in
  is_true ~msg:"unmapped with its last borrow" (B.is_borrowed again)

let test_of_address () =
  let d = cuda () in
  let h = Nx_cuda_device.handles d in
  let a = foreign_alloc h.context 64 in
  let b = Nx_cuda_device.of_address d a S.Int32 16 in
  is_true ~msg:"borrowed" (B.is_borrowed b);
  B.copy ~src:(host_of 64 (fun i -> i)) ~dst:b;
  is_true ~msg:"bytes" (holds b (fun i -> i));
  let inner = Nx_cuda_device.of_address d (Nativeint.add a 16n) S.Int32 4 in
  equal ~msg:"the allocation is the handle" nativeint a (B.handle inner);
  equal ~msg:"at an offset" int 16 (B.offset inner);
  is_true ~msg:"inner bytes" (holds inner (fun i -> 16 + i));
  raises_match (Exn.invalid_arg ~substring:"do not fit") (fun () ->
      Nx_cuda_device.of_address d a S.Int32 17);
  raises_match (Exn.invalid_arg ~substring:"knows no memory") (fun () ->
      Nx_cuda_device.of_address d 0x1000n S.UInt8 1);
  let host, device = foreign_host_alloc h.context 64 in
  let p = Nx_cuda_device.of_address d device S.UInt8 64 in
  equal ~msg:"page-locked memory, which the host addresses" nativeint host
    (B.host_address p);
  let other = other_context_alloc 0 64 in
  raises_match (Exn.invalid_arg ~substring:"not memory of") (fun () ->
      Nx_cuda_device.of_address d other S.UInt8 64)

(* Programs *)

(* Writes [2 * i] into element [i] of its argument, for 256 threads. *)
let ptx =
  {|.version 7.0
.target sm_50
.address_size 64
.visible .entry double_index(.param .u64 out) {
  .reg .u32 %r<3>;
  .reg .u64 %rd<4>;
  ld.param.u64 %rd1, [out];
  cvta.to.global.u64 %rd2, %rd1;
  mov.u32 %r1, %tid.x;
  shl.b32 %r2, %r1, 1;
  mul.wide.u32 %rd3, %r1, 4;
  add.s64 %rd2, %rd2, %rd3;
  st.global.u32 [%rd2], %r2;
  ret;
}
|}

let test_programs () =
  let d = cuda () in
  let f = Nx_device.Program.load d ~binary:ptx ~name:"double_index" in
  is_true ~msg:"cached"
    (Nx_device.Program.load d ~binary:ptx ~name:"double_index" == f);
  raises_match (Exn.failure ~substring:"CUDA_ERROR_") (fun () ->
      Nx_device.Program.load d ~binary:"not a module" ~name:"f");
  raises_match (Exn.failure ~substring:"CUDA_ERROR_NOT_FOUND") (fun () ->
      Nx_device.Program.load d ~binary:ptx ~name:"missing")

(* Launches [double_index] over [out] as timeline work. *)
let double_index d out =
  let h = Nx_cuda_device.handles d in
  let f = Nx_device.Program.load d ~binary:ptx ~name:"double_index" in
  Nx_device.submit d ~touches:[] (fun v ->
      launch h.context h.compute h.signal (Nx_device.Program.handle f) out v)

let test_launch () =
  let d = cuda () in
  let out = B.create d S.Int32 256 in
  double_index d (B.address out);
  Nx_device.synchronize d;
  equal ~msg:"signaled" int (Nx_device.submitted d) (Nx_device.signaled d);
  let host = B.create Nx_device.host S.Int32 256 in
  B.copy ~src:out ~dst:host;
  let ba = B.bigarray Bigarray.int32 host in
  is_true ~msg:"computed"
    (List.for_all
       (fun i -> ba.{i} = Int32.of_int (2 * i))
       (List.init 256 Fun.id))

let test_timeline () =
  let d = cuda () in
  let t = Nx_device.timeline d in
  is_true ~msg:"on the device" (Nx_device.equal (B.device t) d);
  equal ~msg:"the signal" nativeint (Nx_cuda_device.handles d).signal
    (B.address t);
  Nx_device.synchronize d;
  let words = B.create Nx_device.host S.UInt64 2 in
  B.copy ~src:t ~dst:words;
  equal ~msg:"in the signal word" int (Nx_device.signaled d)
    (Int64.to_int (B.bigarray Bigarray.int64 words).{0})

(* nx over the device *)

let test_nx () =
  let d = cuda () in
  let on_gpu = Nx.Placement.device (Nx.Device.of_runtime d) in
  let x = Nx.arange Nx.int32 0 12 1 |> Nx.reshape [| 3; 4 |] in
  let y = Nx.place on_gpu x in
  equal ~msg:"round trip" (array int32) (Nx.to_array x) (Nx.to_array y);
  equal ~msg:"transpose" (array int32)
    (Nx.to_array (Nx.transpose x))
    (Nx.to_array (Nx.transpose y));
  equal ~msg:"compute" (array int32)
    (Nx.to_array (Nx.mul x x))
    (Nx.to_array (Nx.mul y y))

(* A kernel that writes to address 0 faults the device, which fails for good
   with the driver's error. It runs last: the fault ends the context. *)
let test_fault () =
  let d = cuda () in
  let b = B.create d S.UInt8 8 in
  double_index d 0n;
  let illegal = Exn.failure ~substring:"CUDA: CUDA_ERROR_" in
  raises_match illegal (fun () -> Nx_device.synchronize d);
  raises_match illegal (fun () -> B.create d S.UInt8 1);
  raises_match illegal (fun () -> holds b (fun _ -> 0))

let () =
  exit
    (run "nx.cuda.device"
       [
         test "open" test_open;
         group "memory"
           [
             test "device and host memory" test_memory;
             test "out of memory" test_out_of_memory;
           ];
         group "copies"
           [
             test "small copies and views" test_small_copies;
             test "staged in chunks" test_staged;
             test "direct" test_direct;
             test "peer" test_peer;
             test "another domain" test_domains;
             test "two domains on two GPUs" test_two_domains;
           ];
         group "borrowing"
           [
             test "borrow" test_borrow;
             test "borrow on two GPUs" test_borrow_two;
             test "memory of another library" test_of_address;
           ];
         group "programs"
           [
             test "load" test_programs;
             test "launch through submit" test_launch;
             test "the timeline" test_timeline;
           ];
         test "nx" test_nx;
         test "a fault fails the device" test_fault;
       ])
