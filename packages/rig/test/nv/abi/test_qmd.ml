(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Launch descriptors against NVIDIA's QMD headers (open-gpu-doc): version 3 in
   clc7c0qmd.h (Ampere and Ada), version 5 in clcec0qmd.h (Blackwell). A field
   is read back at the MW(hi:lo) position the headers give; a descriptor with
   every value named is listed with its patches. *)

open Windtrap
open Rig_nv_abi
module S = Rig_nv_abi_support

let strf = Printf.sprintf
let zero = Template.Int 0

(* A descriptor's values, all [0], without banks, dynamic shared memory, a chain
   or releases. *)
let base =
  {
    Qmd.program = zero;
    local = zero;
    banks = [];
    grid = (zero, zero, zero);
    block = (zero, zero, zero);
    shared = None;
    chain = None;
    releases = [];
  }

let launch ?(k = S.kernel ()) cls = S.launch (S.gpu ~compute_class:cls ()) k
let qmd ?k cls d = (Qmd.make (launch ?k cls) d).bytes

(* Fields *)

(* The field MW(hi:lo) of [b], the notation of NVIDIA's QMD headers: bits [lo]
   to [hi] read as one little-endian integer. *)
let field b (hi, lo) =
  let n = ref 0 in
  for bit = hi downto lo do
    n := (!n lsl 1) lor ((Char.code b.[bit / 8] lsr (bit mod 8)) land 1)
  done;
  !n

(* An address in a lower and an upper field, shifted. *)
let address b (lower, upper, shift) =
  Int64.(
    shift_left
      (logor (of_int (field b lower)) (shift_left (of_int (field b upper)) 32))
      shift)

(* PROGRAM_ADDRESS_LOWER and _UPPER: version 3 in bytes, version 5 in 16-byte
   units. *)
let program_v3 = ((1567, 1536), (1584, 1568), 0)
let program_v5 = ((1055, 1024), (1076, 1056), 4)

let descriptors =
  group ~timeout:10. "descriptors"
    [
      test "64 KiB of local memory a thread fills its field" (fun () ->
          let local = Template.Int 0x10000 in
          equal ~msg:"version 3" int 0x10000
            (field (qmd S.ada { base with local }) (1623, 1600));
          equal ~msg:"version 5" int (0x10000 lsr 4)
            (field (qmd S.blackwell { base with local }) (1215, 1200)));
      test "a program address keeps bit 48" (fun () ->
          let a = (1 lsl 48) lor 0x12_3456_7800 in
          let program = Template.Int a in
          equal ~msg:"version 3" int64 (Int64.of_int a)
            (address (qmd S.ada { base with program }) program_v3);
          equal ~msg:"version 5" int64 (Int64.of_int a)
            (address (qmd S.blackwell { base with program }) program_v5));
      test "a patch keeps the fields it shares bytes with" (fun () ->
          (* PROGRAM_PREFETCH_SIZE MW(1649:1641) shares its bytes with the patch
             of PROGRAM_PREFETCH_ADDR_UPPER_SHIFTED MW(1640:1632). *)
          let t =
            Qmd.make (launch S.ada) { base with program = Template.Slot 0 }
          in
          let known = field t.bytes (1649, 1641) in
          greater ~msg:"PROGRAM_PREFETCH_SIZE of 0x100 bytes of code" int
            ~than:0 known;
          equal int known
            (field
               (Template.fill (fun _ -> 0x1_ffff_ffff_ff00L) t)
               (1649, 1641)));
      cases ~name:S.class_name
        "dynamic shared memory adds to the block's own, in the largest \
         configuration"
        S.classes (fun cls ->
          let l = launch cls in
          let own = 0x19000 - Launch.dynamic_shared l in
          let size, shift, configs =
            (* SHARED_MEMORY_SIZE, then MIN, TARGET and MAX_SM_CONFIG: 100 KiB
               is 26, its 4 KiB pages plus one. *)
            if cls = S.blackwell then
              ((1162, 1152), 7, [ (1168, 1163); (1180, 1175); (1174, 1169) ])
            else ((561, 544), 0, [ (567, 562); (662, 657); (574, 569) ])
          in
          List.iter
            (fun v ->
              let b =
                (Qmd.make l { base with shared = Some (Template.Int v) }).bytes
              in
              equal ~msg:(strf "size beside %d" v) int
                ((own + v) lsr shift)
                (field b size);
              List.iter
                (fun c -> equal ~msg:"config" int 26 (field b c))
                configs)
            [ 0; 128; Launch.dynamic_shared l ]);
    ]

(* Limits *)

let limits =
  group ~timeout:10. "limits"
    [
      test "a grid takes 2^31-1 blocks along X, 65535 along Y and Z" (fun () ->
          equal (list int)
            [ (1 lsl 31) - 1; 65535; 65535 ]
            (List.map Qmd.max_size [ Grid X; Grid Y; Grid Z ]));
      test "a block takes 1024 threads along X and Y, 64 along Z" (fun () ->
          equal (list int) [ 1024; 1024; 64 ]
            (List.map Qmd.max_size [ Block X; Block Y; Block Z ]));
      cases ~name:S.class_name "more than two releases are refused" S.classes
        (fun cls ->
          let r = Qmd.Release (Template.Int 0x1000, Int 1) in
          ignore (qmd cls { base with releases = [ r; r ] } : string);
          raises_match (Exn.invalid_arg ~substring:"Qmd.make") (fun () ->
              qmd cls { base with releases = [ r; r; r ] }));
      cases ~name:fst "a negative known value is refused"
        [
          ("a program address", { base with program = Int (-1) });
          ("a size", { base with grid = (Int (-1), zero, zero) });
        ]
        (fun (_, d) ->
          raises_match (Exn.invalid_arg ~substring:"unsigned") (fun () ->
              qmd S.ada d));
      cases ~name:string_of_int "a bank the launch lacks is refused"
        [ -1; 1; 7; 8 ] (fun i ->
          raises_match (Exn.invalid_arg ~substring:"Qmd.make") (fun () ->
              qmd S.ada { base with banks = [ (i, Template.Int 0x1000) ] }));
      cases ~name:fst "a kernel at Launch.make's bounds has a descriptor"
        [
          ("255 registers", S.kernel ~registers:255 ());
          ( "a bank of 64 KiB",
            S.kernel ~banks:[ { index = 0; offset = 0; bytes = 0x10000 } ] () );
          ( "bank 7",
            S.kernel ~banks:[ { index = 7; offset = 0; bytes = 16 } ] () );
        ]
        (fun (_, k) ->
          List.iter (fun cls -> ignore (qmd ~k cls base : string)) S.classes);
    ]

(* A descriptor with every value named *)

let values =
  [|
    ("program", 0x12_3456_7800);
    ("grid_x", 100);
    ("grid_y", 2);
    ("grid_z", 3);
    ("block_x", 128);
    ("block_y", 2);
    ("block_z", 1);
    ("bank0", 0x40_0000_1000);
    ("bank3", 0x40_0000_2000);
    ("local", 0x800);
    ("next", 0x9_8765_4300);
    ("release0", 0x1_0000_0010);
    ("payload0", 7);
    ("release1", 0x1_0000_0020);
    ("payload1", 0x1_0000_0001);
  |]

(* [cls]'s descriptor of a kernel with two banks, a chain and a release of each
   kind, value [i] given by [v i]. *)
let named cls v =
  let k =
    S.kernel ~registers:40 ~shared_bytes:0x800 ~stack_bytes:0x40
      ~banks:
        [
          { index = 0; offset = 0; bytes = 0x17c };
          { index = 3; offset = 0x200; bytes = 0x40 };
        ]
      ()
  in
  let v name =
    let rec find i = if fst values.(i) = name then v i else find (i + 1) in
    find 0
  in
  Qmd.make (launch ~k cls)
    {
      base with
      program = v "program";
      grid = (v "grid_x", v "grid_y", v "grid_z");
      block = (v "block_x", v "block_y", v "block_z");
      banks = [ (0, v "bank0"); (3, v "bank3") ];
      local = v "local";
      chain = Some (v "next");
      releases =
        [
          Release (v "release0", v "payload0");
          Stamped (v "release1", v "payload1");
        ];
    }

let known i = Template.Int (snd values.(i))
let slot i = Template.Slot i

let dump cls =
  let s = named cls slot in
  let row i =
    strf "%04x %s" i
      (String.concat " "
         (List.init 16 (fun j -> strf "%02x" (Char.code s.bytes.[i + j]))))
  in
  String.concat "\n"
    (List.init (String.length s.bytes / 16) (fun i -> row (16 * i))
    @ List.map
        (fun (p : Template.patch) ->
          strf "patch at %d, %d bits: (%s + 0x%Lx) >> %d" p.at p.bits
            (fst values.(p.slot))
            p.add p.shift)
        s.patches)

let slots =
  group ~timeout:10. "slots"
    [
      cases ~name:S.class_name
        "a value left to its slot, filled, is the value known" S.classes
        (fun cls ->
          equal string (named cls known).bytes
            (Template.fill
               (fun i -> Int64.of_int (snd values.(i)))
               (named cls slot)));
    ]

let words =
  group ~timeout:10. "words"
    [
      test "an Ada descriptor" (fun () ->
          expect (dump S.ada)
          @@ __POS_OF__
               {|
            0000 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            0010 7f 00 00 00 00 00 00 3c 00 00 00 00 00 00 00 00
            0020 00 00 00 00 00 00 00 00 00 00 00 00 00 00 01 44
            0030 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            0040 13 00 00 00 00 0c 24 34 30 00 00 00 00 00 00 00
            0050 09 28 12 00 00 00 00 00 00 00 00 00 00 00 00 08
            0060 00 00 00 00 00 00 80 a0 00 00 00 00 00 00 00 00
            0070 00 00 00 00 00 00 80 20 00 00 00 00 00 00 00 00
            0080 00 00 00 00 00 00 c4 00 00 00 00 00 00 00 00 00
            0090 00 00 00 00 00 00 00 00 00 00 00 00 00 00 20 00
            00a0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            00b0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            00c0 00 00 00 00 00 00 00 00 00 00 00 00 00 02 00 89
            00d0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            00e0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            00f0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            patch at 32, 32 bits: (program + 0x0) >> 8
            patch at 48, 32 bits: (grid_x + 0x0) >> 0
            patch at 52, 16 bits: (grid_y + 0x0) >> 0
            patch at 56, 16 bits: (grid_z + 0x0) >> 0
            patch at 60, 32 bits: (next + 0x0) >> 8
            patch at 74, 16 bits: (block_x + 0x0) >> 0
            patch at 76, 16 bits: (block_y + 0x0) >> 0
            patch at 78, 16 bits: (block_z + 0x0) >> 0
            patch at 96, 32 bits: (release0 + 0x0) >> 0
            patch at 100, 8 bits: (release0 + 0x0) >> 32
            patch at 104, 32 bits: (payload0 + 0x0) >> 0
            patch at 108, 32 bits: (payload0 + 0x0) >> 32
            patch at 112, 32 bits: (release1 + 0x0) >> 0
            patch at 116, 8 bits: (release1 + 0x0) >> 32
            patch at 120, 32 bits: (payload1 + 0x0) >> 0
            patch at 124, 32 bits: (payload1 + 0x0) >> 32
            patch at 128, 32 bits: (bank0 + 0x0) >> 0
            patch at 132, 17 bits: (bank0 + 0x0) >> 32
            patch at 152, 32 bits: (bank3 + 0x0) >> 0
            patch at 156, 17 bits: (bank3 + 0x0) >> 32
            patch at 192, 32 bits: (program + 0x0) >> 0
            patch at 196, 17 bits: (program + 0x0) >> 32
            patch at 200, 24 bits: (local + 0x0) >> 0
            patch at 204, 9 bits: (program + 0x0) >> 40
            |});
      test "a Blackwell descriptor" (fun () ->
          expect (dump S.blackwell)
          @@ __POS_OF__
               {|
            0000 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            0010 00 00 3f 01 00 00 00 00 00 00 00 00 00 00 00 00
            0020 00 00 00 00 05 10 01 10 00 00 13 00 00 00 00 00
            0030 00 00 00 00 00 00 00 00 89 03 50 0f 00 00 00 00
            0040 00 00 00 00 00 00 00 00 00 00 00 00 00 00 01 00
            0050 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            0060 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            0070 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            0080 00 00 00 00 00 00 20 00 00 00 00 00 00 28 02 00
            0090 18 48 b4 04 00 00 00 00 00 00 00 00 00 00 00 00
            00a0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 c0 00
            00b0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            00c0 00 00 00 00 00 00 20 00 00 00 00 00 00 00 00 00
            00d0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            00e0 00 00 00 00 00 00 00 00 09 10 00 00 00 00 00 00
            00f0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            0100 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            0110 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            0120 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            0130 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            0140 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            0150 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            0160 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            0170 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            patch at 48, 32 bits: (next + 0x0) >> 8
            patch at 60, 32 bits: (release0 + 0x0) >> 0
            patch at 64, 25 bits: (release0 + 0x0) >> 32
            patch at 68, 32 bits: (payload0 + 0x0) >> 0
            patch at 72, 32 bits: (payload0 + 0x0) >> 32
            patch at 96, 32 bits: (release1 + 0x0) >> 0
            patch at 100, 25 bits: (release1 + 0x0) >> 32
            patch at 104, 32 bits: (payload1 + 0x0) >> 0
            patch at 108, 32 bits: (payload1 + 0x0) >> 32
            patch at 128, 32 bits: (program + 0x0) >> 4
            patch at 132, 21 bits: (program + 0x0) >> 36
            patch at 136, 16 bits: (block_x + 0x0) >> 0
            patch at 138, 16 bits: (block_y + 0x0) >> 0
            patch at 140, 8 bits: (block_z + 0x0) >> 0
            patch at 150, 16 bits: (local + 0x0) >> 4
            patch at 156, 32 bits: (grid_x + 0x0) >> 0
            patch at 160, 16 bits: (grid_y + 0x0) >> 0
            patch at 164, 16 bits: (grid_z + 0x0) >> 0
            patch at 168, 32 bits: (bank0 + 0x0) >> 6
            patch at 172, 19 bits: (bank0 + 0x0) >> 38
            patch at 192, 32 bits: (bank3 + 0x0) >> 6
            patch at 196, 19 bits: (bank3 + 0x0) >> 38
            patch at 236, 32 bits: (program + 0x0) >> 8
            patch at 240, 17 bits: (program + 0x0) >> 40
            |});
    ]

let () = exit (run "rig_nv_abi.qmd" [ descriptors; limits; slots; words ])
