(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module B = Nx_device.Buffer
module S = Nx_dtype.Scalar

external compile : string -> string = "test_metal_compile"
external set_signaled : nativeint -> int -> unit = "test_metal_set_signaled"
external contains : nativeint -> nativeint -> bool = "test_metal_contains"
external allocation_count : nativeint -> int = "test_metal_allocation_count"

external dispatch :
  nativeint ->
  nativeint ->
  nativeint ->
  nativeint array ->
  nativeint ->
  nativeint ->
  int ->
  int ->
  unit = "test_metal_dispatch_byte" "test_metal_dispatch"

let metal = Nx_metal_device.v 0

let kernel =
  {|#include <metal_stdlib>
using namespace metal;
struct args { device uint *out; };
kernel void fill(constant args &a [[buffer(0)]],
                 uint i [[threadgroup_position_in_grid]]) {
  a.out[i] = i * 3u + 1u;
}|}

let library = lazy (compile kernel)

let host_bytes n =
  Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout n

let bytes_of_list l =
  let ba = host_bytes (List.length l) in
  List.iteri (fun i x -> ba.{i} <- x) l;
  ba

let read b =
  let ba = host_bytes (B.nbytes b) in
  B.copy ~src:b ~dst:(B.of_bigarray ba);
  List.init (Bigarray.Array1.dim ba) (fun i -> ba.{i})

let write b l = B.copy ~src:(B.of_bigarray (bytes_of_list l)) ~dst:b
let bytes = list int

(* [b]'s bytes as little-endian 32-bit words. *)
let read_words b =
  let l = Array.of_list (read b) in
  List.init
    (Array.length l / 4)
    (fun i ->
      l.(4 * i)
      lor (l.((4 * i) + 1) lsl 8)
      lor (l.((4 * i) + 2) lsl 16)
      lor (l.((4 * i) + 3) lsl 24))

(* Opening *)

let test_open () =
  equal ~msg:"count" int 1 (Nx_metal_device.count ());
  is_true ~msg:"memoised" (Nx_metal_device.v 0 == metal);
  is_true ~msg:"get"
    (match Nx_metal_device.get 0 with Ok d -> d == metal | Error _ -> false);
  is_error ~msg:"one device" (Nx_metal_device.get 1);
  raises_match (Exn.invalid_arg ~substring:"no device 1") (fun () ->
      Nx_metal_device.v 1);
  raises_match (Exn.invalid_arg ~substring:"< 0") (fun () ->
      Nx_metal_device.get (-1));
  equal ~msg:"name" string "METAL" (Nx_device.name metal);
  starts_with ~msg:"arch" ~affix:"Apple" (Nx_device.arch metal);
  is_true ~msg:"budget" (Nx_device.budget metal > 0)

let test_handles () =
  let h = Nx_metal_device.handles metal in
  is_true ~msg:"objects"
    (h.device <> 0n && h.queue <> 0n && h.event <> 0n && h.fence <> 0n);
  raises_match (Exn.invalid_arg ~substring:"not a Metal device") (fun () ->
      Nx_metal_device.handles Nx_device.host)

(* Buffers *)

let test_copies () =
  let s0 = Nx_device.stats metal in
  let b = B.create metal S.UInt8 6 in
  is_true ~msg:"a Metal buffer" (B.handle b <> 0n && B.address b <> 0n);
  write b [ 1; 2; 3; 4; 5; 6 ];
  equal ~msg:"round trip" bytes [ 1; 2; 3; 4; 5; 6 ] (read b);
  let c = B.create metal S.UInt8 3 in
  B.copy ~src:(B.view b ~offset:3 S.UInt8 3) ~dst:c;
  equal ~msg:"on the device" bytes [ 4; 5; 6 ] (read c);
  let d = Nx_device.Stats.diff s0 (Nx_device.stats metal) in
  equal ~msg:"in" int 6 (Nx_device.Stats.bytes_in d);
  equal ~msg:"out" int 9 (Nx_device.Stats.bytes_out d);
  equal ~msg:"allocated" int 9 (Nx_device.Stats.allocated d)

(* More bytes than a page on every platform. *)
let pages = 1 lsl 16

let test_borrow () =
  let whole = B.create Nx_device.host S.UInt8 pages in
  let ba = B.bigarray Bigarray.int8_unsigned whole in
  for i = 0 to 99 do
    ba.{i} <- i
  done;
  let host = B.view whole ~offset:37 S.UInt8 5 in
  let b = B.borrow metal host in
  is_true ~msg:"borrowed" (B.is_borrowed b);
  is_true ~msg:"on Metal" (Nx_device.equal (B.device b) metal);
  equal ~msg:"same bytes" nativeint (B.host_address host) (B.host_address b);
  equal ~msg:"shares memory" bytes [ 37; 38; 39; 40; 41 ] (read b);
  ba.{38} <- 0;
  equal ~msg:"sees host writes" bytes [ 37; 0; 39; 40; 41 ] (read b);
  raises_match (Exn.invalid_arg ~substring:"does not start on a page")
    (fun () -> B.borrow metal (B.of_bigarray (Bigarray.Array1.sub ba 1 3)));
  let e = B.borrow metal (B.view host ~offset:0 S.UInt8 0) in
  is_true ~msg:"a zero-byte borrow is borrowed" (B.is_borrowed e);
  let s0 = Nx_device.stats metal in
  ignore (Sys.opaque_identity (B.borrow metal host));
  equal ~msg:"not counted" int 0
    (Nx_device.Stats.allocated
       (Nx_device.Stats.diff s0 (Nx_device.stats metal)))

let test_budget () =
  let budget = Nx_device.budget metal in
  Fun.protect ~finally:(fun () -> Nx_device.set_budget metal budget)
  @@ fun () ->
  Gc.full_major ();
  let held = Nx_device.Stats.allocated (Nx_device.stats metal) in
  Nx_device.set_budget metal (held + 1024);
  let b = B.create metal S.UInt8 1024 in
  raises_match
    (function Nx_device.Out_of_memory (d, 1) -> d == metal | _ -> false)
    (fun () -> B.create metal S.UInt8 1);
  ignore (Sys.opaque_identity b)

(* Programs and submission *)

let test_programs () =
  let binary = Lazy.force library in
  let p = Nx_device.Program.load metal ~binary ~name:"fill" in
  is_true ~msg:"pipeline" (Nx_device.Program.handle p <> 0n);
  is_true ~msg:"cached" (Nx_device.Program.load metal ~binary ~name:"fill" == p);
  raises_match (Exn.failure ~substring:"no function") (fun () ->
      Nx_device.Program.load metal ~binary ~name:"missing");
  raises_match (Exn.failure ~substring:"") (fun () ->
      Nx_device.Program.load metal ~binary:"not a metallib" ~name:"fill")

(* Every allocation and borrow is resident while it lives: in the residency set
   where Metal has one, in [resources] otherwise. A collected borrow leaves. *)
let test_residency () =
  let set = (Nx_metal_device.handles metal).residency_set in
  let resident b =
    match set with
    | Some set -> contains set (B.handle b)
    | None -> Array.mem (B.handle b) (Nx_metal_device.resources metal)
  in
  let held () =
    match set with
    | Some set -> allocation_count set
    | None -> Array.length (Nx_metal_device.resources metal)
  in
  if set <> None then
    equal ~msg:"no table beside the set" int 0
      (Array.length (Nx_metal_device.resources metal));
  let b = B.create metal S.UInt8 16 in
  is_true ~msg:"an allocation" (resident b);
  Gc.full_major ();
  ignore (Nx_device.stats metal);
  let before = held () in
  (fun () ->
    let bm = B.borrow metal (B.create Nx_device.host S.UInt8 pages) in
    is_true ~msg:"a borrow" (resident bm);
    equal ~msg:"one more" int (before + 1) (held ()))
    ();
  Gc.full_major ();
  ignore (Nx_device.stats metal);
  equal ~msg:"a collected borrow leaves" int before (held ());
  ignore (Sys.opaque_identity b)

let test_dispatch () =
  let p =
    Nx_device.Program.load metal ~binary:(Lazy.force library) ~name:"fill"
  in
  let out = B.create metal S.UInt32 8 in
  write out (List.init 32 (fun _ -> 0));
  let h = Nx_metal_device.handles metal in
  let v =
    Nx_device.submit metal ~touches:[] (fun v ->
        dispatch h.queue h.event h.fence
          (Nx_metal_device.resources metal)
          (Nx_device.Program.handle p)
          (B.address out) 8 v;
        v)
  in
  equal ~msg:"submitted" int v (Nx_device.submitted metal);
  equal ~msg:"the GPU wrote through the address" (list int)
    (List.init 8 (fun i -> (i * 3) + 1))
    (read_words out);
  is_true ~msg:"signaled" (Nx_device.signaled metal >= v)

let test_signal_wait () =
  let h = Nx_metal_device.handles metal in
  let signaled = Atomic.make false in
  let domain =
    Nx_device.submit metal ~touches:[ Nx_device.host ] (fun v ->
        Domain.spawn (fun () ->
            Unix.sleepf 0.05;
            Atomic.set signaled true;
            set_signaled h.event v))
  in
  Nx_device.synchronize Nx_device.host;
  is_true ~msg:"the host waited for Metal's work" (Atomic.get signaled);
  Domain.join domain

let () =
  exit
    (run "nx.metal.device"
       [
         group "opening" [ test "open" test_open; test "handles" test_handles ];
         group "buffers"
           [
             test "copies" test_copies;
             test "borrow host memory" test_borrow;
             test "budget" test_budget;
           ];
         group "submission"
           [
             test "programs" test_programs;
             test "residency" test_residency;
             test "dispatch and signal" test_dispatch;
             test "a signal from another thread" test_signal_wait;
           ];
       ])
