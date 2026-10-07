(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Launch rules: the classes a launch takes, its banks, the driver's parameters,
   and its shared memory, local memory and threads. *)

open Windtrap
open Device_nv_abi
module S = Device_nv_abi_support

let strf = Printf.sprintf
let pp_error = Format.pp_print_string

let classes =
  group ~timeout:10. "classes"
    [
      cases ~name:S.class_name "a class the GPU names takes a launch" S.classes
        (fun cls ->
          is_ok ~pp:pp_error
            (Launch.make (S.gpu ~compute_class:cls ()) (S.kernel ())));
      cases ~name:S.class_name "another class is refused"
        [ 0; -1; 0xc6c0; 0xc8c0; 0xcbc0; 0xcdc0; 0xcfc0; 0xc7b5 ] (fun cls ->
          raises_match (Exn.invalid_arg ~substring:"Launch.make") (fun () ->
              Launch.make (S.gpu ~compute_class:cls ()) (S.kernel ())));
    ]

(* Shared memory up to 100 KiB, the driver's 1 KiB included *)

let limit = (100 * 1024) - 1024

let shared =
  Gen.frequency
    [
      (3, Gen.int_range 0 (2 * limit));
      ( 1,
        Gen.of_list ~pp:Format.pp_print_int
          [ 0; limit - 1; limit; limit + 1; 1 lsl 40 ] );
    ]

let memory =
  group ~timeout:10. "memory"
    [
      prop
        "a launch takes up to 100 KiB of shared memory, the driver's 1 KiB \
         included" (Gen.pair S.compute_class shared) (fun (cls, shared_bytes) ->
          cover "the most" (shared_bytes = limit);
          cover "one byte more" (shared_bytes = limit + 1);
          equal ~msg:"Ok" bool (shared_bytes <= limit)
            (Result.is_ok
               (Launch.make
                  (S.gpu ~compute_class:cls ())
                  (S.kernel ~shared_bytes ()))));
      test "a kernel of max_int bytes of shared memory is refused" (fun () ->
          is_error (Launch.make (S.gpu ()) (S.kernel ~shared_bytes:max_int ())));
      prop "a thread's local memory is its stack and 576 bytes"
        (Gen.pair S.compute_class (Gen.int_range 0 0x10_0000))
        (fun (cls, stack_bytes) ->
          equal int (stack_bytes + 576)
            (Launch.local_bytes
               (S.launch
                  (S.gpu ~compute_class:cls ())
                  (S.kernel ~stack_bytes ()))));
    ]

(* Banks *)

let banks_gen =
  let open Gen in
  let* indices = subsequence [ 0; 1; 2; 3; 4; 5; 6; 7 ] in
  (* Bank 0 first when the kernel has one: test "banks" holds the other case. *)
  let* indices = permutation indices in
  let indices =
    if List.mem 0 indices then 0 :: List.filter (( <> ) 0) indices else indices
  in
  let+ sizes =
    list ~size:(constant (List.length indices)) (int_range 0 0xffff)
  in
  List.mapi
    (fun j (index, bytes) -> { Cubin.index; offset = 0x400 * j; bytes })
    (List.combine indices sizes)

let banks =
  group ~timeout:10. "banks"
    [
      prop
        "a launch addresses the kernel's banks, after a bank 0 of 352 bytes if \
         it has none" (Gen.pair S.compute_class banks_gen) (fun (cls, banks) ->
          let has_0 = List.exists (fun (b : Cubin.bank) -> b.index = 0) banks in
          cover "a bank 0" has_0;
          cover "no bank 0" (not has_0);
          let expected =
            if has_0 then banks
            else { Cubin.index = 0; offset = 0; bytes = 352 } :: banks
          in
          equal (list S.bank) expected
            (Launch.banks
               (S.launch (S.gpu ~compute_class:cls ()) (S.kernel ~banks ()))));
      test "a kernel's bank 0 keeps its place" (fun () ->
          let banks =
            [
              { Cubin.index = 1; offset = 0; bytes = 16 };
              { index = 0; offset = 0x100; bytes = 16 };
            ]
          in
          equal (list S.bank) banks
            (Launch.banks (S.launch (S.gpu ()) (S.kernel ~banks ()))));
    ]

(* The driver's parameters *)

let find s sub =
  let n = String.length sub in
  let rec go i =
    if i + n > String.length s then None
    else if String.sub s i n = sub then Some i
    else go (i + 4)
  in
  go 0

let parameters cls ?(params_offset = 0) ?shared_window ?local_window () =
  Launch.driver_parameters
    (S.launch
       (S.gpu ~compute_class:cls ?shared_window ?local_window ())
       (S.kernel ~params_offset ()))

(* A window at or above 2^40 and below 2^49 whose six low bytes are nonzero, so
   that its bytes are found where it is and nowhere else. *)
let window =
  let open Gen in
  let+ bytes = list ~size:(constant 5) (int_range 1 255)
  and+ top = int_range 1 255 in
  List.fold_left (fun w b -> (w lsl 8) lor b) top bytes

let driver =
  group ~timeout:10. "driver parameters"
    [
      prop "the parameters reach the kernel's, padded with zeros"
        (Gen.pair S.compute_class (Gen.int_range 0 0x1000))
        (fun (cls, params_offset) ->
          let own = parameters cls ()
          and p = parameters cls ~params_offset () in
          let n = String.length own in
          cover "past the class's" (params_offset > n);
          equal ~msg:"length" int (Int.max n params_offset) (String.length p);
          equal ~msg:"the class's" string own (String.sub p 0 n);
          equal ~msg:"zeros" string
            (String.make (String.length p - n) '\000')
            (String.sub p n (String.length p - n)));
      prop "the parameters hold the two windows in place of zeros"
        Gen.(triple S.compute_class window window)
        (fun (cls, s, l) ->
          assume (s <> l);
          let p = parameters cls ~shared_window:s ~local_window:l ()
          and zero = parameters cls ~shared_window:0 ~local_window:0 () in
          let at w =
            match find p (S.le64 (Int64.of_int w)) with
            | Some i -> i
            | None -> failf "no window 0x%x in the parameters" w
          in
          let b = Bytes.of_string zero in
          List.iter
            (fun w ->
              let i = at w in
              equal ~msg:"under a window" string (String.make 8 '\000')
                (String.sub zero i 8);
              Bytes.blit_string (S.le64 (Int64.of_int w)) 0 b i 8)
            [ s; l ];
          equal string (Bytes.to_string b) p);
      test "each class's parameters" (fun () ->
          let show cls =
            let p = parameters cls () in
            strf "%s: %d bytes\n%s" (S.class_name cls) (String.length p)
              (String.concat "\n"
                 (List.filter_map
                    (fun (i, w) ->
                      if w = 0 then None
                      else Some (strf "  word %d: 0x%08x" i w))
                    (List.mapi (fun i w -> (i, w)) (S.words p))))
          in
          expect (String.concat "\n" (List.map show S.classes))
          @@ __POS_OF__
               {|
            ampere: 48 bytes
              word 7: 0x00007294
              word 9: 0x00007293
              word 10: 0x00fffdc0
            ada: 48 bytes
              word 7: 0x00007294
              word 9: 0x00007293
              word 10: 0x00fffdc0
            blackwell: 896 bytes
              word 189: 0x00007294
              word 191: 0x00007293
              word 223: 0x00fffdc0
            |});
    ]

(* Threads: the register file of 65536 registers, allocated per warp in units of
   256, and warps in units of 4 (CUDA C++ Programming Guide, "Technical
   Specifications per Compute Capability", and its Occupancy Calculator's
   allocation granularities for compute capability 8.x). *)

let threads =
  group ~timeout:10. "threads"
    [
      cases
        ~name:(fun (r, _) -> strf "%d registers" r)
        "a block takes 1024 threads, fewer for many registers"
        [
          (0, 1024);
          (16, 1024);
          (64, 1024);
          (72, 896);
          (128, 512);
          (168, 384);
          (255, 256);
        ]
        (fun (registers, n) ->
          equal int n
            (Launch.max_threads (S.launch (S.gpu ()) (S.kernel ~registers ()))));
      prop "more registers never take more threads"
        (Gen.pair (Gen.int_range 0 255) (Gen.int_range 0 255))
        (fun (a, b) ->
          let threads registers =
            -Launch.max_threads (S.launch (S.gpu ()) (S.kernel ~registers ()))
          in
          assume (a <> b);
          Law.monotone int int threads (a, b));
    ]

let () =
  exit (run "device_nv_abi.launch" [ classes; memory; banks; driver; threads ])
