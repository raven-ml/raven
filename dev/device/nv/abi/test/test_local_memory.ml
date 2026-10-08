(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Local memory sizing: a thread's need rounded to 32 bytes, for every thread of
   every multiprocessor, in TPCs of 32 KiB multiples and an allocation of 128
   KiB multiples. *)

open Windtrap
open Device_nv_abi
module S = Device_nv_abi_support

let geometry =
  let open Gen in
  let+ gpcs = int_range 1 16
  and+ tpcs_per_gpc = int_range 1 10
  and+ sms_per_tpc = int_range 1 4
  and+ warps_per_sm = int_range 1 64 in
  { (S.gpu ()) with gpcs; tpcs_per_gpc; sms_per_tpc; warps_per_sm }

let pp_gpu ppf (g : Gpu.t) =
  Format.fprintf ppf
    "{ gpcs = %d; tpcs_per_gpc = %d; sms_per_tpc = %d; warps_per_sm = %d }"
    g.gpcs g.tpcs_per_gpc g.sms_per_tpc g.warps_per_sm

let need =
  Gen.frequency
    [
      (3, Gen.int_range 0 0x10_0000);
      (1, Gen.of_list ~pp:Format.pp_print_int [ 0; 1; 31; 32; 33; 576 ]);
    ]

let local =
  Testable.make
    ~pp:(fun ppf (l : Local_memory.t) ->
      Format.fprintf ppf "{ per_thread = %d; per_tpc = %d; bytes = %d }"
        l.per_thread l.per_tpc l.bytes)
    ~equal:( = )

let tests =
  group ~timeout:10. "make"
    [
      test "no local memory is none" (fun () ->
          equal local
            { per_thread = 0; per_tpc = 0; bytes = 0 }
            (Local_memory.make (S.gpu ()) 0));
      test "596 bytes a thread on an RTX 5000 Ada" (fun () ->
          (* 608 bytes a thread, 19456 a warp, 96 warps a TPC: 57 * 32 KiB; 66
             TPCs: 123273216 bytes, rounded up to 941 * 128 KiB. *)
          equal local
            { per_thread = 608; per_tpc = 57 * 0x8000; bytes = 941 * 0x20000 }
            (Local_memory.make (S.gpu ()) 596));
      prop "local memory is the least multiples that hold every thread"
        (Gen.pair (Gen.with_pp pp_gpu geometry) need)
        (fun (g, n) ->
          let per_thread = S.round_up n 32 in
          let per_tpc =
            S.round_up (per_thread * 32 * g.warps_per_sm * g.sms_per_tpc) 0x8000
          in
          let bytes = S.round_up (per_tpc * g.tpcs_per_gpc * g.gpcs) 0x20000 in
          equal local { per_thread; per_tpc; bytes } (Local_memory.make g n));
      test "2^40 bytes a thread on an RTX 5000 Ada" (fun () ->
          (* 2^45 bytes a warp, 96 warps a TPC, 66 TPCs: no step wraps. *)
          let per_thread = 1 lsl 40 in
          let per_tpc = per_thread * 32 * 96 in
          equal local
            { per_thread; per_tpc; bytes = per_tpc * 66 }
            (Local_memory.make (S.gpu ()) per_thread));
      cases ~name:string_of_int
        "a need whose allocation would pass max_int is refused"
        (* Each step wraps first for one of them: the thread's rounding, the
           warp's share, the TPC's, the allocation. *)
        [ max_int; max_int - 30; (max_int / 32) + 1; 1 lsl 52; 1 lsl 45 ]
        (fun n ->
          raises_match (Exn.invalid_arg ~substring:"Local_memory.make")
            (fun () -> Local_memory.make (S.gpu ()) n));
      cases ~name:string_of_int "a negative need is refused" [ min_int; -1 ]
        (fun n ->
          raises_match (Exn.invalid_arg ~substring:"Local_memory.make")
            (fun () -> Local_memory.make (S.gpu ()) n));
    ]

let () = exit (run "device_nv_abi.local_memory" [ tests ])
