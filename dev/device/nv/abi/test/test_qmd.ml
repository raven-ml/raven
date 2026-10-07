(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Launch descriptors against NVIDIA's QMD headers (open-gpu-doc): version 3 in
   clc7c0qmd.h (Ampere and Ada), version 5 in clcec0qmd.h (Blackwell). A
   descriptor is read back field by field, at the MW(hi:lo) positions the
   headers give, from its structure filled with its values. *)

open Windtrap
open Device_nv_abi
module S = Device_nv_abi_support

let strf = Printf.sprintf
let encode q = Structure.encode Fun.id (Qmd.structure q)

let qmd ?(k = S.kernel ()) cls =
  Qmd.make (S.launch (S.gpu ~compute_class:cls ()) k)

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
  cwd_membar_type : int * int; (* L1_SYSMEMBAR 1 *)
  grid : (int * int) list;
  block : (int * int) list;
  program : (int * int) * (int * int) * int; (* lower, upper, shift *)
  bank_addr : int -> (int * int) * (int * int) * int;
  bank_valid : int -> int * int;
  bank_size : int -> int * int; (* SIZE_SHIFTED4 *)
  local : (int * int) * int;
  releases : release list;
  chain_enable : int * int;
  chain_action : int * int; (* QMD_SCHEDULE 1 *)
  chain_pointer : int * int;
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
    cwd_membar_type = (369, 368);
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
    releases = [ release 0; release 1 ];
    chain_enable = bit 512;
    chain_action = (515, 513);
    chain_pointer = (511, 480);
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
    cwd_membar_type = (625, 624);
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
    releases =
      [
        release 0 ((511, 480), (536, 512)) ((575, 544), (607, 576));
        release 1 ((799, 768), (824, 800)) ((863, 832), (895, 864));
      ];
    chain_enable = bit 336;
    chain_action = (339, 337);
    chain_pointer = (415, 384);
  }

let layout cls = if cls = S.blackwell then v5 else v3
let field = S.field

(* An address in a lower and an upper field, shifted. *)
let address b (lower, upper, shift) =
  Int64.(
    shift_left
      (logor (of_int (field b lower)) (shift_left (of_int (field b upper)) 32))
      shift)

let wide b (lower, upper) = address b (lower, upper, 0)

let dim_field l : Qmd.dim -> int * int = function
  | Grid X -> List.nth l.grid 0
  | Grid Y -> List.nth l.grid 1
  | Grid Z -> List.nth l.grid 2
  | Block X -> List.nth l.block 0
  | Block Y -> List.nth l.block 1
  | Block Z -> List.nth l.block 2

(* What a descriptor's setters set *)

type model = {
  dims : (Qmd.dim * int64) list;
  program : int64 option;
  banks : (int * int64) list;
  local : int64 option;
  releases : (bool * int64 * int64) list; (* stamped, address, value *)
  system : bool; (* a release of scope System *)
  chain : int64 option;
}

let empty =
  {
    dims = [];
    program = None;
    banks = [];
    local = None;
    releases = [];
    system = false;
    chain = None;
  }

let set k v l = (k, v) :: List.remove_assoc k l

let release m stamp s a v =
  if List.length m.releases = 2 then m
  else
    {
      m with
      releases = m.releases @ [ (stamp, a, v) ];
      system = m.system || s = Packet.System;
    }

let step m : S.op -> model = function
  | Set_dim (d, n) -> { m with dims = set d (Int64.of_int n) m.dims }
  | Patch_dim (d, v) -> { m with dims = set d v m.dims }
  | Set_program a -> { m with program = Some a }
  | Set_bank (i, a) -> { m with banks = set i a m.banks }
  | Set_local_memory n -> { m with local = Some n }
  | Release (s, a, v) -> release m false s a v
  | Release_stamp (s, a, v) -> release m true s a v
  | Chain a -> { m with chain = Some a }

let triple_release =
  Testable.make
    ~pp:(fun ppf (st, a, v) ->
      Format.fprintf ppf "(%s, 0x%Lx, 0x%Lx)"
        (if st then "stamp" else "release")
        a v)
    ~equal:( = )

(* Reading the descriptor's fields against the model *)

let check (d : S.drawn) =
  let q = S.descriptor d in
  let b = encode q in
  let l = layout d.gpu.compute_class in
  let m = List.fold_left step empty d.ops in
  let f = field b in
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
  List.iter
    (fun (dim, n) ->
      equal ~msg:(S.dim_name dim) int64 n (Int64.of_int (f (dim_field l dim))))
    m.dims;
  Option.iter
    (fun a -> equal ~msg:"PROGRAM_ADDRESS" int64 a (address b l.program))
    m.program;
  List.iter
    (fun (i, a) ->
      equal
        ~msg:(strf "CONSTANT_BUFFER_ADDR %d" i)
        int64 a
        (address b (l.bank_addr i)))
    m.banks;
  Option.iter
    (fun n ->
      let fl, shift = l.local in
      equal ~msg:"SHADER_LOCAL_MEMORY_HIGH_SIZE" int64 n
        (Int64.shift_left (Int64.of_int (f fl)) shift))
    m.local;
  let enabled = List.filter (fun r -> f r.enable = 1) l.releases in
  List.iter
    (fun r -> equal ~msg:"RELEASE_PAYLOAD64B" int 1 (f r.payload64b))
    enabled;
  equal ~msg:"releases"
    (slist triple_release compare)
    m.releases
    (List.map
       (fun r -> (f r.size = 0, wide b r.addr, wide b r.payload))
       enabled);
  if m.system then
    equal ~msg:"CWD_MEMBAR_TYPE of a System release" int 1 (f l.cwd_membar_type);
  match m.chain with
  | None ->
      equal ~msg:"DEPENDENT_QMD0_ENABLE unchained" int 0 (f l.chain_enable)
  | Some a ->
      equal ~msg:"DEPENDENT_QMD0_ENABLE" int 1 (f l.chain_enable);
      equal ~msg:"DEPENDENT_QMD0_ACTION" int 1 (f l.chain_action);
      equal ~msg:"DEPENDENT_QMD0_POINTER" int64 a
        (Int64.shift_left (Int64.of_int (f l.chain_pointer)) 8)

let structure_t =
  Testable.make
    ~pp:(fun ppf (s : int64 Structure.t) ->
      Format.fprintf ppf "<%d bytes, %d holes>" (String.length s.bytes)
        (List.length s.holes))
    ~equal:( = )

let descriptors =
  group ~timeout:10. "descriptors"
    [
      prop ~count:300 "a descriptor holds what its launch and setters set"
        S.drawn (fun d ->
          List.iter
            (fun c -> cover (S.class_name c) (d.gpu.compute_class = c))
            S.classes;
          cover "a size set and patched"
            (List.exists
               (fun d ->
                 List.exists
                   (function S.Set_dim (d', _) -> d' = d | _ -> false)
                   d.ops
                 && List.exists
                      (function S.Patch_dim (d', _) -> d' = d | _ -> false)
                      d.ops)
               S.dims);
          cover "both releases"
            (List.length
               (List.filter
                  (function
                    | S.Release _ | Release_stamp _ -> true | _ -> false)
                  d.ops)
            >= 2);
          check d);
      prop "setters leave their descriptor as it was" S.drawn (fun d ->
          let q = S.descriptor d in
          let before = Qmd.structure q in
          List.iter
            (fun op ->
              ignore (S.apply op q);
              equal
                ~msg:(Format.asprintf "%a" S.pp_op op)
                structure_t before (Qmd.structure q))
            d.ops);
      prop "a size filled later is the size set now"
        Gen.(
          pair S.drawn
            (let* d = S.dim in
             map (fun n -> (d, n)) (int_range 0 (Qmd.max_size d))))
        (fun (d, (dim, n)) ->
          let q = S.descriptor d in
          equal string
            (encode (Qmd.set_dim dim n q))
            (encode (Qmd.patch_dim dim (Int64.of_int n) q)));
      test "64 KiB of local memory a thread fills its field" (fun () ->
          equal ~msg:"version 3" int 0x10000
            (field
               (encode (Qmd.set_local_memory 0x10000L (qmd S.ada)))
               (1623, 1600));
          equal ~msg:"version 5" int (0x10000 lsr 4)
            (field
               (encode (Qmd.set_local_memory 0x10000L (qmd S.blackwell)))
               (1215, 1200)));
      test "a program address keeps bit 48" (fun () ->
          let a = Int64.of_int ((1 lsl 48) lor 0x12_3456_7800) in
          equal ~msg:"version 3" int64 a
            (address (encode (Qmd.set_program a (qmd S.ada))) v3.program);
          equal ~msg:"version 5" int64 a
            (address (encode (Qmd.set_program a (qmd S.blackwell))) v5.program));
      test "a hole keeps the fields it shares bytes with" (fun () ->
          (* PROGRAM_PREFETCH_SIZE MW(1649:1641) shares its bytes with the hole
             of PROGRAM_PREFETCH_ADDR_UPPER_SHIFTED MW(1640:1632). *)
          let q = Qmd.set_program 0x1_ffff_ffff_ff00L (qmd S.ada) in
          let known = field (Qmd.structure q).bytes (1649, 1641) in
          greater ~msg:"PROGRAM_PREFETCH_SIZE of 0x100 bytes of code" int
            ~than:0 known;
          equal int known (field (encode q) (1649, 1641)));
      test "a size set after it was patched is the size set" (fun () ->
          let q =
            qmd S.ada |> Qmd.patch_dim (Grid Y) 5L |> Qmd.set_dim (Grid Y) 7
          in
          equal int 7 (field (encode q) (dim_field v3 (Grid Y))));
      cases ~name:S.class_name "two releases, then none" S.classes (fun cls ->
          let q = qmd cls in
          let q = require_some (Qmd.release System 0x1000L 1L q) in
          let q = require_some (Qmd.release_stamp Agent 0x2000L 2L q) in
          is_none (Qmd.release System 0x3000L 3L q);
          is_none (Qmd.release_stamp System 0x3000L 3L q));
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
      cases
        ~name:(fun (c, d) -> strf "%s %s" (S.class_name c) (S.dim_name d))
        "a size outside [0;max_size] is refused, its bounds are not"
        (List.concat_map (fun c -> List.map (fun d -> (c, d)) S.dims) S.classes)
        (fun (cls, d) ->
          let q = qmd cls and max = Qmd.max_size d in
          ignore (Qmd.set_dim d 0 q);
          ignore (Qmd.set_dim d max q);
          List.iter
            (fun n ->
              raises_match ~msg:(string_of_int n)
                (Exn.invalid_arg ~substring:"Qmd.set_dim") (fun () ->
                  Qmd.set_dim d n q))
            [ min_int; -1; max + 1; max_int ]);
      cases ~name:string_of_int "a bank the launch lacks is refused"
        [ -1; 1; 7; 8 ] (fun i ->
          raises_match (Exn.invalid_arg ~substring:"Qmd.set_bank") (fun () ->
              Qmd.set_bank i 0x1000L (qmd S.ada)));
      cases ~name:fst "a kernel at Launch.make's bounds has a descriptor"
        [
          ("255 registers", S.kernel ~registers:255 ());
          ( "a bank of 64 KiB",
            S.kernel ~banks:[ { index = 0; offset = 0; bytes = 0x10000 } ] () );
          ( "bank 7",
            S.kernel ~banks:[ { index = 7; offset = 0; bytes = 16 } ] () );
        ]
        (fun (_, k) -> List.iter (fun cls -> ignore (qmd ~k cls)) S.classes);
      cases ~name:fst "a kernel no descriptor holds is refused"
        [
          ("256 registers", S.kernel ~registers:256 ());
          ( "a bank of 64 KiB and a byte",
            S.kernel ~banks:[ { index = 0; offset = 0; bytes = 0x10001 } ] () );
          ( "bank 8",
            S.kernel ~banks:[ { index = 8; offset = 0; bytes = 16 } ] () );
          ( "bank -1",
            S.kernel ~banks:[ { index = -1; offset = 0; bytes = 16 } ] () );
        ]
        (fun (_, k) ->
          List.iter
            (fun cls -> is_error (Launch.make (S.gpu ~compute_class:cls ()) k))
            S.classes);
    ]

(* The words of a descriptor, over named values *)

let rec pp_term ppf : string Packet.term -> unit = function
  | Value v -> Format.pp_print_string ppf v
  | Add (t, n) -> Format.fprintf ppf "Add (%a, 0x%Lx)" pp_term t n
  | Shift (t, n) -> Format.fprintf ppf "Shift (%a, %d)" pp_term t n

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
  let q =
    Qmd.make (S.launch (S.gpu ~compute_class:cls ()) k)
    |> Qmd.set_dim (Grid X) 128
    |> Qmd.patch_dim (Grid Y) "grid_y"
    |> Qmd.set_dim (Block X) 256 |> Qmd.set_program "program"
    |> Qmd.set_bank 0 "bank0" |> Qmd.set_bank 3 "bank3"
    |> Qmd.set_local_memory "local"
    |> Qmd.chain "next"
  in
  let q = Option.get (Qmd.release System "signal" "value" q) in
  let s = Qmd.structure q in
  let row i =
    strf "%04x %s" i
      (String.concat " "
         (List.init 16 (fun j -> strf "%02x" (Char.code s.bytes.[i + j]))))
  in
  String.concat "\n"
    (List.init (String.length s.bytes / 16) (fun i -> row (16 * i))
    @ List.map
        (fun (h : string Structure.hole) ->
          Format.asprintf "hole at %d, %d bits: %a" h.at h.bits pp_term h.value)
        s.holes)

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
        0030 80 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
        0040 13 00 00 00 00 0c 24 34 30 00 00 01 00 00 00 00
        0050 09 28 12 00 00 00 00 00 00 00 00 00 00 00 00 08
        0060 00 00 00 00 00 00 80 a0 00 00 00 00 00 00 00 00
        0070 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
        0080 00 00 00 00 00 00 c4 00 00 00 00 00 00 00 00 00
        0090 00 00 00 00 00 00 00 00 00 00 00 00 00 00 20 00
        00a0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
        00b0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
        00c0 00 00 00 00 00 00 00 00 00 00 00 00 00 02 00 89
        00d0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
        00e0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
        00f0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
        hole at 32, 32 bits: Shift (program, 8)
        hole at 52, 16 bits: grid_y
        hole at 60, 32 bits: Shift (next, 8)
        hole at 96, 32 bits: signal
        hole at 100, 8 bits: Shift (signal, 32)
        hole at 104, 32 bits: value
        hole at 108, 32 bits: Shift (value, 32)
        hole at 128, 32 bits: Shift (bank0, 0)
        hole at 132, 17 bits: Shift (Shift (bank0, 0), 32)
        hole at 152, 32 bits: Shift (bank3, 0)
        hole at 156, 17 bits: Shift (Shift (bank3, 0), 32)
        hole at 192, 32 bits: Shift (program, 0)
        hole at 196, 17 bits: Shift (Shift (program, 0), 32)
        hole at 200, 24 bits: local
        hole at 204, 9 bits: Shift (Shift (program, 8), 32)
        |});
      test "a Blackwell descriptor" (fun () ->
          expect (dump S.blackwell)
          @@ __POS_OF__
               {|
            0000 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            0010 00 00 3f 01 00 00 00 00 00 00 00 00 00 00 00 00
            0020 00 00 00 00 05 10 00 00 00 00 13 00 00 00 00 00
            0030 00 00 00 00 00 00 00 00 89 03 50 0f 00 00 00 00
            0040 00 00 00 00 00 00 00 00 00 00 00 00 00 00 01 00
            0050 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            0060 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            0070 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
            0080 00 00 00 00 00 00 20 00 00 01 00 00 00 28 02 00
            0090 18 48 b4 04 00 00 00 00 00 00 00 00 80 00 00 00
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
            hole at 48, 32 bits: Shift (next, 8)
            hole at 60, 32 bits: signal
            hole at 64, 25 bits: Shift (signal, 32)
            hole at 68, 32 bits: value
            hole at 72, 32 bits: Shift (value, 32)
            hole at 128, 32 bits: Shift (program, 4)
            hole at 132, 21 bits: Shift (Shift (program, 4), 32)
            hole at 150, 16 bits: Shift (local, 4)
            hole at 160, 16 bits: grid_y
            hole at 168, 32 bits: Shift (bank0, 6)
            hole at 172, 19 bits: Shift (Shift (bank0, 6), 32)
            hole at 192, 32 bits: Shift (bank3, 6)
            hole at 196, 19 bits: Shift (Shift (bank3, 6), 32)
            hole at 236, 32 bits: Shift (program, 8)
            hole at 240, 17 bits: Shift (Shift (program, 8), 32)
            |});
    ]

let () = exit (run "device_nv_abi.qmd" [ descriptors; limits; words ])
