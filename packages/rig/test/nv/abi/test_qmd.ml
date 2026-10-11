(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Launch descriptors against NVIDIA's QMD headers (open-gpu-doc): version 3 in
   clc7c0qmd.h (Ampere and Ada), version 5 in clcec0qmd.h (Blackwell). A
   descriptor is read back field by field, at the MW(hi:lo) positions the
   headers give, from its structure filled with its values. *)

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

type release = {
  enable : int * int;
  size : int * int; (* STRUCTURE_SIZE: FOUR_WORDS 0, TWO_WORDS 2 *)
  payload64b : int * int;
  addr : (int * int) * (int * int);
  payload : (int * int) * (int * int);
}

type layout = {
  major : int;
  major_version : int * int;
  register_count : int * int;
  sass_version : int * int;
  grid : (int * int) list;
  block : (int * int) list;
  program : (int * int) * (int * int) * int; (* lower, upper, shift *)
  bank_addr : int -> (int * int) * (int * int) * int;
  bank_valid : int -> int * int;
  bank_size : int -> int * int; (* SIZE_SHIFTED4 *)
  local : (int * int) * int;
  chain_enable : int * int;
  chain_action : int * int; (* QMD_SCHEDULE 1 *)
  chain_pointer : int * int;
  releases : release list;
}

let bit n = (n, n)

let v3 =
  let release j =
    let o = 128 * j in
    {
      enable = bit (823 + o);
      size = (831 + o, 830 + o);
      payload64b = bit (829 + o);
      addr = ((799 + o, 768 + o), (807 + o, 800 + o));
      payload = ((863 + o, 832 + o), (895 + o, 864 + o));
    }
  in
  {
    major = 3;
    major_version = (583, 580);
    register_count = (656, 648);
    sass_version = (1663, 1656);
    grid = [ (415, 384); (431, 416); (463, 448) ];
    block = [ (607, 592); (623, 608); (639, 624) ];
    program = ((1567, 1536), (1584, 1568), 0);
    bank_addr =
      (fun i ->
        let o = 64 * i in
        ((1055 + o, 1024 + o), (1072 + o, 1056 + o), 0));
    bank_valid = (fun i -> bit (640 + i));
    bank_size = (fun i -> (1087 + (64 * i), 1075 + (64 * i)));
    local = ((1623, 1600), 0);
    chain_enable = bit 512;
    chain_action = (515, 513);
    chain_pointer = (511, 480);
    releases = [ release 0; release 1 ];
  }

let v5 =
  let release j addr payload =
    let o = 16 * j in
    {
      enable = bit (288 + o);
      size = (290 + o, 289 + o);
      payload64b = bit (300 + o);
      addr;
      payload;
    }
  in
  {
    major = 5;
    major_version = (471, 468);
    register_count = (1136, 1128);
    sass_version = (455, 448);
    grid = [ (1279, 1248); (1295, 1280); (1327, 1312) ];
    block = [ (1103, 1088); (1119, 1104); (1127, 1120) ];
    program = ((1055, 1024), (1076, 1056), 4);
    bank_addr =
      (fun i ->
        let o = 64 * i in
        ((1375 + o, 1344 + o), (1394 + o, 1376 + o), 6));
    bank_valid = (fun i -> bit (1856 + (4 * i)));
    bank_size = (fun i -> (1407 + (64 * i), 1395 + (64 * i)));
    local = ((1215, 1200), 4);
    chain_enable = bit 336;
    chain_action = (339, 337);
    chain_pointer = (415, 384);
    releases =
      [
        release 0 ((511, 480), (536, 512)) ((575, 544), (607, 576));
        release 1 ((799, 768), (824, 800)) ((863, 832), (895, 864));
      ];
  }

let layout cls = if cls = S.blackwell then v5 else v3
let field = S.field

(* An address in a lower and an upper field, shifted. *)
let address b (lower, upper, shift) =
  Int64.(
    shift_left
      (logor (of_int (field b lower)) (shift_left (of_int (field b upper)) 32))
      shift)

let dim_field l : Qmd.dim -> int * int = function
  | Grid X -> List.nth l.grid 0
  | Grid Y -> List.nth l.grid 1
  | Grid Z -> List.nth l.grid 2
  | Block X -> List.nth l.block 0
  | Block Y -> List.nth l.block 1
  | Block Z -> List.nth l.block 2

(* Reading the descriptor's fields against its values *)

let check (d : S.drawn) =
  let b = S.filled d in
  let l = layout d.gpu.compute_class in
  let f = field b and v = S.value d in
  equal ~msg:"QMD_MAJOR_VERSION" int l.major (f l.major_version);
  equal ~msg:"REGISTER_COUNT" int d.kernel.registers (f l.register_count);
  equal ~msg:"SASS_VERSION" int d.gpu.sass_version (f l.sass_version);
  List.iter
    (fun (bk : Cubin.bank) ->
      let msg = strf "bank %d" bk.index in
      equal
        ~msg:(msg ^ " CONSTANT_BUFFER_VALID")
        int 1
        (f (l.bank_valid bk.index));
      (* SIZE_SHIFTED4: the bank in 16-byte units, all of it. *)
      equal
        ~msg:(msg ^ " CONSTANT_BUFFER_SIZE_SHIFTED4")
        int
        ((bk.bytes + 15) / 16)
        (f (l.bank_size bk.index)))
    (Launch.banks (S.launch d.gpu d.kernel));
  List.iteri
    (fun i dim ->
      let slot = if i < 3 then S.grid_slot + i else S.block_slot + i - 3 in
      equal ~msg:(S.dim_name dim) int64 (v slot)
        (Int64.of_int (f (dim_field l dim))))
    S.dims;
  equal ~msg:"PROGRAM_ADDRESS" int64 (v S.program_slot) (address b l.program);
  List.iter
    (fun (i, _) ->
      equal
        ~msg:(strf "CONSTANT_BUFFER_ADDR %d" i)
        int64
        (v (S.bank_slot i))
        (address b (l.bank_addr i)))
    d.desc.banks;
  (let fl, shift = l.local in
   equal ~msg:"SHADER_LOCAL_MEMORY_HIGH_SIZE" int64 (v S.local_slot)
     (Int64.shift_left (Int64.of_int (f fl)) shift));
  List.iteri
    (fun j (r : release) ->
      let msg what = strf "release %d %s" j what in
      match List.nth_opt d.desc.releases j with
      | None -> equal ~msg:(msg "RELEASE_ENABLE") int 0 (f r.enable)
      | Some release ->
          let size = match release with Qmd.Release _ -> 2 | Stamped _ -> 0 in
          let slot = S.release_slot j in
          equal ~msg:(msg "RELEASE_ENABLE") int 1 (f r.enable);
          equal ~msg:(msg "RELEASE_PAYLOAD64B") int 1 (f r.payload64b);
          equal ~msg:(msg "RELEASE_STRUCTURE_SIZE") int size (f r.size);
          equal ~msg:(msg "address") int64 (v slot)
            (address b (fst r.addr, snd r.addr, 0));
          equal ~msg:(msg "payload") int64
            (v (slot + 1))
            (address b (fst r.payload, snd r.payload, 0)))
    l.releases;
  match d.desc.chain with
  | None ->
      equal ~msg:"DEPENDENT_QMD0_ENABLE unchained" int 0 (f l.chain_enable)
  | Some _ ->
      equal ~msg:"DEPENDENT_QMD0_ENABLE" int 1 (f l.chain_enable);
      equal ~msg:"DEPENDENT_QMD0_ACTION" int 1 (f l.chain_action);
      equal ~msg:"DEPENDENT_QMD0_POINTER" int64 (v S.chain_slot)
        (Int64.shift_left (Int64.of_int (f l.chain_pointer)) 8)

(* [d] with every value known. *)
let known (d : S.drawn) =
  let k = function
    | Template.Slot i -> Template.Int d.values.(i)
    | Int _ as v -> v
  in
  let three (x, y, z) = (k x, k y, k z) in
  let q = d.desc in
  {
    Qmd.program = k q.program;
    local = k q.local;
    banks = List.map (fun (i, v) -> (i, k v)) q.banks;
    grid = three q.grid;
    block = three q.block;
    shared = Option.map k q.shared;
    chain = Option.map k q.chain;
    releases =
      List.map
        (function
          | Qmd.Release (a, v) -> Qmd.Release (k a, k v)
          | Stamped (a, v) -> Stamped (k a, k v))
        q.releases;
  }

let slots (d : S.drawn) = List.length (S.descriptor d).patches

let descriptors =
  group ~timeout:10. "descriptors"
    [
      prop ~count:300 "a descriptor holds its launch and its values" S.drawn
        (fun d ->
          List.iter
            (fun c -> cover (S.class_name c) (d.gpu.compute_class = c))
            S.classes;
          cover "dynamic shared memory" (Option.is_some d.desc.shared);
          cover "a chain" (Option.is_some d.desc.chain);
          cover "two releases" (List.length d.desc.releases = 2);
          check d);
      prop "a value left to its slot, filled, is the value known" S.drawn
        (fun d ->
          cover "a slot" (slots d > 0);
          equal string (Qmd.make (S.launch d.gpu d.kernel) (known d)).bytes
            (S.filled d));
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
            (address (qmd S.ada { base with program }) v3.program);
          equal ~msg:"version 5" int64 (Int64.of_int a)
            (address (qmd S.blackwell { base with program }) v5.program));
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

(* The words of a descriptor, over named values *)

let dump cls =
  let k =
    S.kernel ~registers:40 ~shared_bytes:0x800 ~stack_bytes:0x40
      ~banks:
        [
          { index = 0; offset = 0; bytes = 0x17c };
          { index = 3; offset = 0x200; bytes = 0x40 };
        ]
      ()
  in
  let names = [| "program"; "grid_y"; "bank0"; "bank3"; "local"; "next" |] in
  let slot name =
    let rec find i = if names.(i) = name then i else find (i + 1) in
    Template.Slot (find 0)
  in
  let s =
    Qmd.make
      (S.launch (S.gpu ~compute_class:cls ()) k)
      {
        base with
        program = slot "program";
        grid = (zero, slot "grid_y", zero);
        banks = [ (0, slot "bank0"); (3, slot "bank3") ];
        local = slot "local";
        chain = Some (slot "next");
      }
  in
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
            names.(p.slot) p.add p.shift)
        s.patches)

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
            0060 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            0070 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            0080 00 00 00 00 00 00 c4 00 00 00 00 00 00 00 00 00
            0090 00 00 00 00 00 00 00 00 00 00 00 00 00 00 20 00
            00a0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            00b0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            00c0 00 00 00 00 00 00 00 00 00 00 00 00 00 02 00 89
            00d0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            00e0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            00f0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            patch at 32, 32 bits: (program + 0x0) >> 8
            patch at 52, 16 bits: (grid_y + 0x0) >> 0
            patch at 60, 32 bits: (next + 0x0) >> 8
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
            0020 00 00 00 00 00 00 00 00 00 00 13 00 00 00 00 00
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
            patch at 128, 32 bits: (program + 0x0) >> 4
            patch at 132, 21 bits: (program + 0x0) >> 36
            patch at 150, 16 bits: (local + 0x0) >> 4
            patch at 160, 16 bits: (grid_y + 0x0) >> 0
            patch at 168, 32 bits: (bank0 + 0x0) >> 6
            patch at 172, 19 bits: (bank0 + 0x0) >> 38
            patch at 192, 32 bits: (bank3 + 0x0) >> 6
            patch at 196, 19 bits: (bank3 + 0x0) >> 38
            patch at 236, 32 bits: (program + 0x0) >> 8
            patch at 240, 17 bits: (program + 0x0) >> 40
            |});
    ]

let () = exit (run "rig_nv_abi.qmd" [ descriptors; limits; words ])
