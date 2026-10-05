(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* NVIDIA packets of integers: words, method headers, copies, ring entries,
   launch programs and descriptors. *)

open Windtrap
module P = Nx_nv_packet
module M = P.Methods
module Gpfifo = P.Gpfifo
module Qmd = P.Qmd

(* The spec: NVIDIA's class headers clc56f.h (host), clc6b5.h (copy engine),
   clcdc0.h and clc6c0qmd.h (launch descriptors, version 3). Fields are (lowest
   bit, bits). *)
module D = struct
  let nvc56f_sem_addr_lo = 0x5c
  let nvc56f_non_stall_interrupt = 0x20
  let nvc56f_sem_execute_operation = (0, 3)
  let nvc56f_sem_execute_operation_acq_circ_geq = 3
  let nvc56f_sem_execute_operation_release = 1
  let nvc56f_sem_execute_payload_size = (24, 1)
  let nvc56f_sem_execute_payload_size_64bit = 1
  let nvc56f_sem_execute_release_wfi = (20, 1)
  let nvc56f_sem_execute_release_timestamp = (25, 1)
  let nvc56f_gp_entry1_level = (9, 1)
  let nvc56f_gp_entry1_level_subroutine = 1
  let nvc56f_gp_entry1_length = (10, 21)
  let nvc6b5_set_semaphore_a = 0x240
  let nvc6b5_launch_dma = 0x300
  let nvc6b5_offset_in_upper = 0x400
  let nvc6b5_line_length_in = 0x418
  let nvc6b5_launch_dma_semaphore_type = (3, 2)
  let nvc6b5_launch_dma_semaphore_type_release_one_word_semaphore = 1
  let blackwell_compute_a = 0xcdc0

  (* The lowest bit of fields of version 3's descriptors. *)
  let qmd_v3 =
    [
      ("cta_raster_width", 384);
      ("program_address_lower", 1536);
      ("program_address_upper", 1568);
      ("program_prefetch_addr_lower_shifted", 256);
      ("program_prefetch_addr_upper_shifted", 1632);
    ]
end

let mask32 = 0xffff_ffff

(* Words *)

let values =
  Gen.frequency
    [
      (3, Gen.int_range 0 ((1 lsl 62) - 1));
      ( 1,
        Gen.of_list ~pp:Format.pp_print_int
          [ 0; mask32; 1 lsl 32; (1 lsl 32) + 1; (1 lsl 62) - 1 ] );
    ]

(* A term computes on 64-bit unsigned integers; the values drawn keep the sum
   within OCaml's integers. *)
let term_law ((v, n), k) =
  equal int ((v + n) lsr k) (P.eval (P.Shift (P.Add (P.Value v, n), k)));
  equal int ((v lsr k) + n) (P.eval (P.Add (P.Shift (P.Value v, k), n)))

let terms =
  Gen.(
    pair
      (pair (int_range 0 (1 lsl 61)) (int_range 0 (1 lsl 32)))
      (int_range 0 62))

let words_law v =
  equal (list int)
    [ v land mask32; v land mask32; v land mask32; (v lsr 32) land mask32 ]
    (P.dwords [ P.Dword v; P.W32 (P.Value v); P.W64 (P.Value v) ])

(* Method headers *)

(* The methods of [words] as (subchannel, method, words), checking that each
   header counts the words that follow it. *)
let rec decode = function
  | [] -> []
  | h :: rest ->
      equal ~msg:"an incrementing method" int 2 (h lsr 28);
      let n = (h lsr 16) land 0x1fff
      and s = (h lsr 13) land 7
      and m = (h land 0x1fff) lsl 2 in
      less ~msg:"the words the header counts" int ~than:(List.length rest + 1) n;
      (s, m, List.filteri (fun i _ -> i < n) rest)
      :: decode (List.filteri (fun i _ -> i >= n) rest)

let field (lo, bits) v = (v lsr lo) land ((1 lsl bits) - 1)

let test_semaphores () =
  let a = 0x12_3456_7890 and v = (1 lsl 33) + 5 in
  (match decode (P.dwords (M.acquire a v)) with
  | [ (0, m, [ alo; ahi; vlo; vhi; exe ]) ] ->
      equal ~msg:"the semaphore's methods" int D.nvc56f_sem_addr_lo m;
      equal ~msg:"address" int a (alo lor (ahi lsl 32));
      equal ~msg:"value" int v (vlo lor (vhi lsl 32));
      equal ~msg:"an acquire" int D.nvc56f_sem_execute_operation_acq_circ_geq
        (field D.nvc56f_sem_execute_operation exe);
      equal ~msg:"of 64 bits" int D.nvc56f_sem_execute_payload_size_64bit
        (field D.nvc56f_sem_execute_payload_size exe)
  | _ -> fail "an acquire is one run of five methods");
  (match decode (P.dwords (M.release a v)) with
  | [ (0, _, [ _; _; _; _; exe ]); (0, intr, [ 0 ]) ] ->
      equal ~msg:"a release" int D.nvc56f_sem_execute_operation_release
        (field D.nvc56f_sem_execute_operation exe);
      equal ~msg:"after the channel idles" int 1
        (field D.nvc56f_sem_execute_release_wfi exe);
      equal ~msg:"no timestamp" int 0
        (field D.nvc56f_sem_execute_release_timestamp exe);
      equal ~msg:"then an interrupt" int D.nvc56f_non_stall_interrupt intr
  | _ -> fail "a release is a semaphore and an interrupt");
  match decode (P.dwords (M.release_stamp a v)) with
  | [ (0, _, [ _; _; _; _; exe ]) ] ->
      equal ~msg:"a timestamp" int 1
        (field D.nvc56f_sem_execute_release_timestamp exe)
  | _ -> fail "a stamp is a semaphore alone"

(* Copies *)

let line = 1 lsl 31

(* The copies [words] make, as (dst, src, bytes). *)
let copies words =
  let rec go = function
    | (4, m, [ shi; slo; dhi; dlo ])
      :: (4, l, [ n ])
      :: (4, launch, [ _ ])
      :: rest
      when m = D.nvc6b5_offset_in_upper
           && l = D.nvc6b5_line_length_in
           && launch = D.nvc6b5_launch_dma ->
        ((dhi lsl 32) lor dlo, (shi lsl 32) lor slo, n) :: go rest
    | [] -> []
    | _ -> fail "a copy is runs of offsets, a length and a launch"
  in
  go (decode words)

let copy_law ((dst, src), n) =
  let lines = (n + line - 1) / line in
  cover "several lines" (lines > 1);
  equal
    (list (triple int int int))
    (List.init lines (fun i ->
         (dst + (i * line), src + (i * line), Int.min line (n - (i * line)))))
    (copies (P.dwords (M.copy ~dst ~src n)))

let sizes =
  Gen.frequency
    [
      (2, Gen.int_range 0 (1 lsl 20));
      ( 1,
        Gen.of_list ~pp:Format.pp_print_int
          [ 0; 1; line - 1; line; line + 1; 1 lsl 32; 5 * line ] );
    ]

let addresses = Gen.int_range 0 (1 lsl 48)

let test_copy_release () =
  match decode (P.dwords (M.copy_release 0x1_0000_0010 ((1 lsl 32) + 7))) with
  | [ (4, m, [ hi; lo; w ]); (4, l, [ launch ]) ] ->
      equal ~msg:"the semaphore" int D.nvc6b5_set_semaphore_a m;
      equal ~msg:"address, high word first" (pair int int) (1, 0x10) (hi, lo);
      equal ~msg:"the value's low word" int 7 w;
      equal ~msg:"launched" int D.nvc6b5_launch_dma l;
      equal ~msg:"one word" int
        D.nvc6b5_launch_dma_semaphore_type_release_one_word_semaphore
        (field D.nvc6b5_launch_dma_semaphore_type launch)
  | _ -> fail "a copy release is a semaphore and a launch"

(* Ring entries *)

let entry_law ((a, offset), words) =
  let a = a * 4 and offset = offset * 4 in
  let e = P.eval (Gpfifo.entry a ~offset ~words) in
  equal ~msg:"the segment's address" int (a + offset) (e land ((1 lsl 40) - 1));
  equal ~msg:"a subroutine" int D.nvc56f_gp_entry1_level_subroutine
    (field D.nvc56f_gp_entry1_level (e lsr 32));
  equal ~msg:"its words" int words (field D.nvc56f_gp_entry1_length (e lsr 32))

(* An entry's length field holds 21 bits of words. *)
let test_entry_words () =
  let max = (1 lsl 21) - 1 in
  equal ~msg:"max_words" int max Gpfifo.max_words;
  ignore (Gpfifo.entry 0x1000 ~offset:0 ~words:max);
  raises_match (Exn.invalid_arg ~substring:"words") (fun () ->
      Gpfifo.entry 0x1000 ~offset:0 ~words:(max + 1));
  raises_match (Exn.invalid_arg ~substring:"words") (fun () ->
      Gpfifo.entry 0x1000 ~offset:0 ~words:(-1))

let entries =
  Gen.pair
    (Gen.pair (Gen.int_range 0 ((1 lsl 37) - 1)) (Gen.int_range 0 0xffff))
    (Gen.frequency
       [
         (3, Gen.int_range 0 ((1 lsl 20) - 1));
         (1, Gen.of_list ~pp:Format.pp_print_int [ 0; 1; (1 lsl 20) - 1 ]);
       ])

(* Programs *)

let kernel ?(registers = 32) ?(shared_bytes = 0) ?(params_offset = 0)
    ?(banks = []) () =
  {
    Nx_nv_cubin.code = 0x80;
    code_bytes = 0x100;
    registers;
    shared_bytes;
    stack_bytes = 0x20;
    params_offset;
    banks;
  }

let make ?(blackwell = false) k =
  P.Program.make
    ~compute_class:(if blackwell then D.blackwell_compute_a else 0xc9c0)
    ~sass_version:0x89 ~shared_window:0x7294_0000_0000
    ~local_window:0x7293_0000_0000 k

let program ?blackwell k =
  require_ok ~pp:Format.pp_print_string (make ?blackwell k)

let bank =
  Testable.make
    ~pp:(fun ppf (b : Nx_nv_cubin.bank) ->
      Format.fprintf ppf "{ %d; 0x%x; 0x%x }" b.index b.offset b.bytes)
    ~equal:( = )

let test_banks () =
  equal (list bank)
    [ { index = 0; offset = 0; bytes = 352 } ]
    (P.Program.banks (program (kernel ())));
  let own =
    [
      { Nx_nv_cubin.index = 3; offset = 0x400; bytes = 8 };
      { index = 0; offset = 0x200; bytes = 0x180 };
    ]
  in
  equal (list bank)
    [ { index = 0; offset = 0x200; bytes = 0x180 }; List.hd own ]
    (P.Program.banks (program (kernel ~banks:own ())))

let test_driver_parameters () =
  let words s =
    List.init
      (String.length s / 4)
      (fun i -> Int32.to_int (String.get_int32_le s (4 * i)) land mask32)
  in
  let ada = words (P.Program.driver_parameters (program (kernel ()))) in
  equal ~msg:"twelve words before Blackwell" int 12 (List.length ada);
  equal ~msg:"the windows and the stack limit" (list int)
    [ 0; 0x7294; 0; 0x7293; 0xfffdc0; 0 ]
    (List.filteri (fun i _ -> i >= 6) ada);
  let blackwell =
    words (P.Program.driver_parameters (program ~blackwell:true (kernel ())))
  in
  equal ~msg:"224 words from Blackwell" int 224 (List.length blackwell);
  equal ~msg:"the windows" (list int) [ 0; 0x7294; 0; 0x7293 ]
    (List.filteri (fun i _ -> i >= 188 && i < 192) blackwell);
  equal ~msg:"the stack limit" int 0xfffdc0 (List.nth blackwell 223);
  equal ~msg:"up to the kernel's parameters" int 0x200
    (String.length
       (P.Program.driver_parameters (program (kernel ~params_offset:0x200 ()))))

let test_limits () =
  equal ~msg:"128 registers leave 512 threads" int 512
    (P.Program.max_threads (program (kernel ~registers:128 ())));
  (* One register a thread takes 256 per warp: 256 warps of the register
     file. *)
  equal ~msg:"no registers count as one" int 8192
    (P.Program.max_threads (program (kernel ~registers:0 ())));
  equal ~msg:"local memory: the stack and 576 bytes" int (0x20 + 576)
    (P.Program.local_bytes (program (kernel ())));
  let limit = (100 * 1024) - 1024 in
  ignore (program (kernel ~shared_bytes:limit ()));
  contains ~sub:"shared memory"
    (require_error (make (kernel ~shared_bytes:(limit + 1) ())))

(* Launch descriptors *)

let test_holes () =
  let q = Qmd.make (program (kernel ())) in
  Qmd.patch_dim q (Grid X) 7;
  Qmd.set_program q 0x12_3456_7800;
  let s = Qmd.structure q in
  (* The widest of 8, 4, 2 and 1 bytes within each field, by offset. *)
  let hole f bytes = (List.assoc f D.qmd_v3 / 8, bytes) in
  equal
    (list (pair int int))
    (List.sort compare
       [
         hole "cta_raster_width" 4;
         hole "program_address_lower" 4;
         hole "program_address_upper" 2;
         hole "program_prefetch_addr_lower_shifted" 4;
         hole "program_prefetch_addr_upper_shifted" 1;
       ])
    (List.map (fun (h : int P.hole) -> (h.at, h.bytes)) s.holes);
  equal ~msg:"fill writes the grid's width" int 7
    (Int32.to_int
       (String.get_int32_le (P.fill s)
          (List.assoc "cta_raster_width" D.qmd_v3 / 8)))

let test_dims () =
  let q = Qmd.make (program ~blackwell:true (kernel ())) in
  Qmd.set_dim q (Block Z) 0xff;
  raises_match (Exn.invalid_arg ~substring:"does not fit") (fun () ->
      Qmd.set_dim q (Block Z) 0x100)

let test_releases () =
  let q = Qmd.make (program (kernel ())) in
  equal ~msg:"a first release" bool true (Qmd.release q 0x1000 1);
  equal ~msg:"a second" bool true (Qmd.release_stamp q 0x2000 0);
  equal ~msg:"no third" bool false (Qmd.release q 0x3000 2)

let () =
  exit
    (run "nx.nv.packet"
       [
         group "terms and words"
           [
             prop "a term adds and shifts in its order" terms term_law;
             prop "a value's words are its low 32 bits, then its high" values
               words_law;
           ];
         group "methods"
           [
             test "semaphores acquire, release and stamp" test_semaphores;
             prop "a copy goes in lines of at most 2 GiB"
               (Gen.pair (Gen.pair addresses addresses) sizes)
               copy_law;
             test "the copy engine releases a value's low word"
               test_copy_release;
           ];
         group "ring entries"
           [
             prop "an entry holds the segment's address and words" entries
               entry_law;
             test
               "an entry of more words than its length field holds is refused"
               test_entry_words;
           ];
         group "programs"
           [
             test "bank 0 is the cubin's, or 352 bytes" test_banks;
             test "the driver's parameters start bank 0" test_driver_parameters;
             test "threads, local and shared memory" test_limits;
           ];
         group "launch descriptors"
           [
             test "holes are the fields' widest words, in order" test_holes;
             test "a size that does not fit its field is refused" test_dims;
             test "two releases, then none" test_releases;
           ];
       ])
