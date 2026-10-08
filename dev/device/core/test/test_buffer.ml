(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module C = Device_core
module B = Device_core.Buffer
module P = Device_core_support.Polled
module S = Device_dtype.Scalar

let timeout = 60.
let memory name = require_ok ~pp:Format.pp_print_string (C.memory_device name)
let lost = function C.Lost _ -> true | _ -> false

let bytes b =
  let ba = B.bigarray Bigarray.char b in
  String.init (Bigarray.Array1.dim ba) (Bigarray.Array1.get ba)

let filled n c =
  let b = B.create C.host S.UInt8 n in
  Bigarray.Array1.fill (B.bigarray Bigarray.char b) c;
  b

(* Host buffers *)

let test_create () =
  let b = B.create C.host S.Float32 10 in
  equal int 40 (B.nbytes b);
  equal int 10 (B.length b);
  equal bool false (B.is_borrowed b);
  equal bool true (B.spans b);
  let big = B.create C.host S.UInt8 (1 lsl 20) in
  equal int 0 (B.address big mod 4096)

let test_bit_sizes () =
  equal int 2 (B.nbytes (B.create C.host S.Int4 3));
  equal int 2 (B.nbytes (B.create C.host S.Bit 9));
  equal int 0 (B.nbytes (B.create C.host S.Bit 0))

let test_refusals () =
  raises_match Exn.invalid_arg (fun () -> B.create C.host S.UInt8 (-1));
  raises_match Exn.invalid_arg (fun () -> B.create C.host S.Float64 max_int);
  let b = B.create C.host S.UInt8 16 in
  raises_match Exn.invalid_arg (fun () -> B.view b ~offset:12 S.Float32 2);
  raises_match Exn.invalid_arg (fun () -> B.view b ~offset:2 S.Float32 1);
  raises_match Exn.invalid_arg (fun () -> B.view b ~offset:(-1) S.UInt8 1)

let test_views () =
  let b = filled 8 'a' in
  let v = B.view b ~offset:4 S.UInt8 4 in
  Bigarray.Array1.fill (B.bigarray Bigarray.char v) 'b';
  equal string "aaaabbbb" (bytes b);
  equal bool false (B.spans v);
  equal bool true (B.overlaps b v);
  equal bool false (B.overlaps (B.view b ~offset:0 S.UInt8 4) v)

let test_of_bigarray () =
  let ba = Bigarray.Array1.create Bigarray.float32 Bigarray.c_layout 4 in
  Bigarray.Array1.fill ba 1.5;
  let b = B.of_bigarray ba in
  equal bool true (B.is_borrowed b);
  equal int 16 (B.nbytes b);
  equal float_exact 1.5 (B.bigarray Bigarray.float32 b).{3};
  raises_match Exn.invalid_arg (fun () ->
      B.of_bigarray (Bigarray.Array1.create Bigarray.int Bigarray.c_layout 1))

(* Copies *)

let test_copy_host () =
  let src = filled 100 'x' and dst = filled 100 'y' in
  B.copy ~src ~dst;
  equal string (String.make 100 'x') (bytes dst);
  raises_match Exn.invalid_arg (fun () -> B.copy ~src ~dst:(filled 99 'y'));
  raises_match Exn.invalid_arg (fun () ->
      B.copy
        ~src:(B.view src ~offset:0 S.UInt8 50)
        ~dst:(B.view src ~offset:10 S.UInt8 50))

let test_copy_device () =
  let d = memory "buffer:copy" in
  let on = B.create d S.UInt8 64 in
  B.copy ~src:(filled 64 'q') ~dst:on;
  let back = filled 64 'z' in
  B.copy ~src:on ~dst:back;
  equal string (String.make 64 'q') (bytes back)

(* A copy on a Polled device runs only when a wait reaches its sleep: the copy's
   own wait for its point. *)
let test_copy_waits () =
  let d, p = P.open_ "buffer:copy-polled" in
  let on = B.create d S.UInt8 4096 in
  B.copy ~src:(filled 4096 'w') ~dst:on;
  let back = filled 4096 '0' in
  B.copy ~src:on ~dst:back;
  equal string (String.make 4096 'w') (bytes back);
  equal int 0 (P.queued p)

(* Borrows *)

let test_borrow () =
  let d = memory "buffer:borrow" in
  let h = filled 32 'h' in
  let b = require_some (B.borrow d h) in
  equal bool true (B.is_borrowed b);
  equal bool true (B.overlaps b h);
  equal bool true (C.equal d (B.device b));
  equal int (B.address h) (B.address b)

(* A host buffer a device maps is mapped once, whatever its borrows. *)
let test_borrow_maps_once () =
  let d, p = P.open_ "buffer:map-once" in
  let h = B.create C.host S.UInt8 (1 lsl 16) in
  for _ = 1 to 3 do
    ignore (Sys.opaque_identity (B.borrow d h))
  done;
  equal int 1 (List.length (List.filter (( = ) "map_host") (P.log p)))

let test_borrow_small () =
  let d, _ = P.open_ "buffer:small" in
  let paged = B.create C.host S.UInt8 (1 lsl 16) in
  let off_page = Bigarray.Array1.sub (B.bigarray Bigarray.char paged) 8 16 in
  is_none (B.borrow d (B.of_bigarray off_page))

(* Waits *)

let test_wait () =
  let d, p = P.open_ "buffer:wait" in
  let n = 1 lsl 16 in
  let h = filled n 'a' and on = B.create d S.UInt8 n in
  let part =
    {
      C.Submission.queue = "COPY:0";
      after = [||];
      work = C.Submission.Copy { src = require_some (B.borrow d h); dst = on };
    }
  in
  ignore (C.submit (C.Submission.make ~reads:0 ~writes:0 ~waits:0 d [| part |]));
  equal int 1 (P.queued p);
  B.wait on B.Read;
  equal int 0 (P.queued p)

let test_dead () =
  let b = B.create C.host S.UInt8 8 in
  C.Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
      ignore (C.Claim.consume c ~why:"donated" b));
  raises_match (Exn.invalid_arg ~substring:"donated") (fun () ->
      B.wait b B.Read)

(* Claims *)

let test_claims () =
  let b = B.create C.host S.UInt8 8 in
  C.Claim.read b;
  C.Claim.release b;
  raises_match Exn.invalid_arg (fun () -> C.Claim.release b);
  C.Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
      equal bool true (C.Claim.exclusive c b);
      raises_match Exn.invalid_arg (fun () -> C.Claim.read b));
  C.Claim.read b;
  C.Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
      equal bool false (C.Claim.exclusive c b));
  C.Claim.release b

let test_claim_overlaps () =
  let b = B.create C.host S.UInt8 8 in
  raises_match Exn.invalid_arg (fun () ->
      C.Claim.with_
        ~read:[ B.view b ~offset:4 S.UInt8 4 ]
        ~donate:[ [ b ] ] ignore);
  C.Claim.with_ ~read:[ b; b ] ~donate:[] ignore

let test_claim_of_bigarray () =
  let b =
    B.of_bigarray (Bigarray.Array1.create Bigarray.char Bigarray.c_layout 8)
  in
  C.Claim.with_ ~read:[] ~donate:[ [ b ] ] (fun c ->
      equal bool false (C.Claim.exclusive c b))

let test_lost_memory () =
  let d, p = P.open_ "buffer:lost" in
  let on = B.create d S.UInt8 8 in
  let s = C.Submission.make ~reads:0 ~writes:1 ~waits:0 d [||] in
  C.Submission.write s 0 on;
  P.fail p;
  raises_match lost (fun () -> C.submit s);
  raises_match lost (fun () -> C.Claim.read on);
  raises_match lost (fun () -> B.wait on B.Read)

let tests =
  [
    group ~timeout "host buffers"
      [
        test "a buffer holds its elements' bytes" test_create;
        test "sub-byte formats pack their elements" test_bit_sizes;
        test "a buffer refuses bounds it cannot hold" test_refusals;
        test "a view shares its buffer's memory" test_views;
        test "a bigarray's buffer is its elements" test_of_bigarray;
      ];
    group ~timeout "copies"
      [
        test "a copy between host buffers moves their bytes" test_copy_host;
        test "a copy to and from a memory device moves the bytes"
          test_copy_device;
        test "a copy returns once the device's work ran" test_copy_waits;
      ];
    group ~timeout "borrows"
      [
        test "a borrow is the memory it maps" test_borrow;
        test "a memory a device borrows is mapped once" test_borrow_maps_once;
        test "host memory off a page does not borrow on a driver's device"
          test_borrow_small;
      ];
    group ~timeout "waits"
      [
        test "a wait returns once the work that wrote the memory ran" test_wait;
        test "a dead buffer's wait raises its reason" test_dead;
        test "memory a lost device wrote raises Lost" test_lost_memory;
      ];
    group ~timeout "claims"
      [
        test "readers share a memory, an exclusive claim excludes them"
          test_claims;
        test "a donated buffer overlapping another is refused"
          test_claim_overlaps;
        test "a bigarray's memory is never exclusive" test_claim_of_bigarray;
      ];
  ]

let () = exit (run "device_core.buffer" tests)
