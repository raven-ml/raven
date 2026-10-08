(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* PM4 packets over integers, against the words of soc15d.h (GFX9), nvd.h (GFX10
   on) and PAL's WAIT_REG_MEM64. *)

open Windtrap
open Rig_amd_abi
module S = Rig_amd_abi_support

let strf = Printf.sprintf
let timeout = S.timeout
let gpu gc = S.gpu ~shader_engines:6 ~compute_units:48 gc
let gfx11 = gpu (11, 0, 0)
let gfx9 = gpu (9, 4, 3)
let words = S.encode
let generations = [ gfx9; gfx11; gpu (12, 0, 0) ]
let name (g : Gpu.t) = S.version g.gc

(* PACKET3(op, n): type 3, n + 1 words after the header. *)
let packet3 op n = (3 lsl 30) lor (n lsl 16) lor (op lsl 8)

(* SET_SH_REG's registers and SET_UCONFIG_REG's, as the .mli states them. *)
let ranges = [ (0x76, (0x2c00, 0x3000)); (0x79, (0xc000, 0x10000)) ]

(* A first register around a range's ends, and up to 4 words; one run in four
   ends exactly at a range's end. *)
let runs_of_regs =
  let open Gen in
  let edges = [ 0x2c00; 0x3000; 0xc000; 0x10000; 0xc000 + 0xffff ] in
  let word =
    frequency
      [
        (2, map (fun n -> Packet.Dword n) (int_range 0 0xffff_ffff));
        (1, map (fun n -> Packet.W64 (Value n)) (int_range 0 0xffff_ffff));
      ]
  in
  let words = list ~size:(int_range 0 4) word in
  let around =
    let+ edge = of_list edges and+ d = int_range (-4) 3 and+ ws = words in
    (edge + d, ws)
  in
  let up_to_end =
    let+ _, (_, stop) = of_list ranges
    and+ ws = list ~size:(int_range 1 4) word in
    (stop - Rig_packet.size ws, ws)
  in
  with_pp
    (fun ppf (a, ws) ->
      Format.fprintf ppf "0x%x, %d words" a (Rig_packet.size ws))
    (frequency [ (3, around); (1, up_to_end) ])

let memory =
  group ~timeout "memory and registers"
    [
      test "a write to a register, at one address" (fun () ->
          equal (list int)
            [ packet3 0x37 3; 1 lsl 16; 0x1234; 0; 0xabcd ]
            (words (Pm4.write_data (Register 0x1234) 0xabcd)));
      test "a confirmed write to memory" (fun () ->
          equal (list int)
            [ packet3 0x37 3; (1 lsl 20) lor (5 lsl 8); 0x40; 0x1; 9 ]
            (words (Pm4.write_data (Memory 0x1_0000_0040) 9)));
      test "an SH register is set from SH's start" (fun () ->
          equal (list int)
            [ packet3 0x76 1; 0x20c; 7 ]
            (words (Pm4.set_reg 0x2e0c [ W32 (Value 7) ])));
      test "a UCONFIG register is set from UCONFIG's start" (fun () ->
          equal (list int)
            [ packet3 0x79 1; 0x200; 7 ]
            (words (Pm4.set_reg 0xc200 [ W32 (Value 7) ])));
      prop "a run of registers is set iff it lies in one range" runs_of_regs
        (fun (a, ws) ->
          let fits (start, stop) =
            start <= a && a < stop && a + Rig_packet.size ws <= stop
          in
          let range = List.find_opt (fun (_, r) -> fits r) ranges in
          cover "past a range's end" (range = None);
          cover "up to a range's end"
            (List.exists
               (fun (_, (_, stop)) -> a + Rig_packet.size ws = stop)
               ranges);
          match (range, Pm4.set_reg a ws) with
          | Some (op, (start, _)), p ->
              equal (list int)
                ([ packet3 op (Rig_packet.size ws); a - start ] @ words ws)
                (words p)
          | None, _ -> fail "set"
          | exception Invalid_argument m ->
              equal (option int) None (Option.map fst range);
              contains ~sub:"Pm4.set_reg" m);
      test "a counter copied to memory through the L2" (fun () ->
          equal (list int)
            [ packet3 0x40 4; (2 lsl 8) lor 4; 0x99; 0; 0x8; 0 ]
            (words (Pm4.copy_data Posted (Counter 0x99) 8)));
      test "the clock copied to memory, 64 bits, confirmed" (fun () ->
          equal (list int)
            [
              packet3 0x40 4;
              9 lor (2 lsl 8) lor (1 lsl 16) lor (1 lsl 20);
              0;
              0;
              0x8;
              0;
            ]
            (words (Pm4.copy_data Confirmed Clock 8)));
    ]

let waits =
  group ~timeout "waits"
    [
      test "a wait on a register" (fun () ->
          equal (list int)
            [ packet3 0x3c 5; 3; 0x99; 0; 4; 4; 0x20 ]
            (words
               (Pm4.wait gfx11 (Register 0x99) Equal 4 ~mask:4 ~interval:0x20 ())));
      test "a wait on memory, every bit, every 4 clocks" (fun () ->
          equal (list int)
            [ packet3 0x3c 5; (1 lsl 4) lor 5; 0x8; 0; 1; 0xffff_ffff; 4 ]
            (words (Pm4.wait gfx11 (Memory 8) Greater_equal 1 ())));
      cases
        ~name:(fun (g, _) -> name g)
        "a wait on a UCONFIG register, from UCONFIG's start on GFX9"
        [ (gfx9, 0x8e8); (gfx11, 0xc8e8); (gpu (12, 0, 0), 0xc8e8) ]
        (fun (g, reg) ->
          equal int reg
            (List.nth
               (words (Pm4.wait g (Register 0xc8e8) Equal 0 ~mask:1 ()))
               2));
      test "a wait compares the reference's low 32 bits" (fun () ->
          equal int 4
            (List.nth
               (words (Pm4.wait gfx11 (Memory 8) Equal 0x7_0000_0004 ()))
               4));
      test "a 64-bit wait compares every bit of both words" (fun () ->
          equal (list int)
            [
              packet3 0x93 7;
              (1 lsl 4) lor 5;
              0x10;
              0x2;
              7;
              1;
              0xffff_ffff;
              0xffff_ffff;
              4;
            ]
            (words
               (Pm4.wait_64 gfx11 0x2_0000_0010 Greater_equal 0x1_0000_0007 ())));
      test "GFX9 has no 64-bit wait" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"Pm4.wait_64") (fun () ->
              Pm4.wait_64 gfx9 0 Equal 0 ()));
    ]

(* nvd.h's RELEASE_MEM: CACHE_FLUSH_AND_INV_TS_EVENT at the end of the pipe, and
   the GCR bits of amdgpu's GFX11 fence (gfx_v11_0.c): GLM_WB, GLM_INV, GL2_WB,
   SEQ. *)
let release_event = 0x14 lor (5 lsl 8)
let release_gcr = 0x1000 lor 0x2000 lor 0x20_0000 lor 0x40_0000

(* The cache bits of a release's control word, by generation and scope. A
   release writes back what readers of its scope would miss, and invalidates
   nothing a reader's acquire does not. - GFX9 (soc15d.h EOP_TC_WB_ACTION_EN,
   EOP_TC_NC_ACTION_EN), at both scopes: an agent of GFX942 may have an L2 per
   die, which LLVM's memory model writes back with buffer_wbl2 at agent scope
   (AMDGPUUsage, "Memory Model GFX942"). - GFX11 at System: amdgpu's fence
   (gfx_v11_0.c), above. - GFX12 at System: amdgpu's fence (gfx_v12_0.c), GL2_WB
   and SEQ. - GFX11 and GFX12 at Agent: none. The GPU's work shares one L2, and
   LLVM's memory model releases at agent scope with no write-back (AMDGPUUsage,
   "Memory Model GFX10-GFX11"; "Memory Model GFX12": global_wb is omitted for
   scopes below SCOPE_SYS). *)
let release_caches =
  let tc_wb = 0x8000 lor 0x8_0000 in
  [
    ((gfx9, Packet.Agent), tc_wb);
    ((gfx9, System), tc_wb);
    ((gfx11, Agent), 0);
    ((gfx11, System), release_gcr);
    ((gpu (12, 0, 0), Agent), 0);
    ((gpu (12, 0, 0), System), 0x20_0000 lor 0x40_0000);
  ]

(* The caches an acquire's control word names: scalar, vector, L1, instruction,
   L2 invalidated, L2 written back. GFX9's CP_COHER_CNTL (soc15d.h):
   SH_KCACHE_ACTION_ENA 27, TCL1_ACTION_ENA 22 (its vector cache is its L1),
   SH_ICACHE_ACTION_ENA 29, TC_ACTION_ENA 23, TC_WB_ACTION_ENA 18. GFX10 on,
   GCR_CNTL (nvd.h): GLK_INV 7, GLV_INV 8, GL1_INV 9, GLI_INV 0, GL2_INV 14,
   GL2_WB 15. *)
let acquired (g : Gpu.t) ws =
  let bit w n = w land (1 lsl n) <> 0 in
  match (g.gc, ws) with
  | (9, _, _), [ _; c; _; _; _; _; _ ] ->
      (bit c 27, bit c 22, bit c 22, bit c 29, bit c 23, bit c 18)
  | _, [ _; _; _; _; _; _; _; c ] ->
      (bit c 7, bit c 8, bit c 9, bit c 0, bit c 14, bit c 15)
  | _ -> failf "an acquire of %d words" (List.length ws)

let caches_w =
  Testable.make
    ~pp:(fun ppf (k, v, l1, i, inv, wb) ->
      Format.fprintf ppf
        "{ scalar = %b; vector = %b; l1 = %b; instruction = %b; l2_inv = %b; \
         l2_wb = %b }"
        k v l1 i inv wb)
    ~equal:( = )

let caches =
  group ~timeout "caches and signals"
    [
      test "a release with an interrupt" (fun () ->
          equal (list int)
            [
              packet3 0x49 6;
              release_event lor release_gcr;
              (2 lsl 29) lor (2 lsl 24);
              0x40;
              1;
              9;
              0;
              0x77;
            ]
            (words
               (Pm4.release_mem gfx11 System ~interrupt:0x77 0x1_0000_0040
                  (Data_64 9))));
      test "a release without an interrupt" (fun () ->
          equal (list int)
            [ packet3 0x49 6; release_event; 1 lsl 29; 0x40; 0; 9; 0; 0 ]
            (words (Pm4.release_mem gfx11 Agent 0x40 (Low_32 9))));
      cases
        ~name:(function Packet.Agent, _ -> "agent" | System, _ -> "system")
        "an acquire on GFX11 invalidates the L2 only for the system"
        [ (Packet.Agent, 0x3f0); (System, 0xc3f1) ]
        (fun (scope, cntl) ->
          equal (list int)
            [ packet3 0x58 6; 0; 0xffff_ffff; 0xffff_ffff; 0; 0; 0; cntl ]
            (words (Pm4.acquire_mem gfx11 scope)));
      cases
        ~name:(fun ((g : Gpu.t), s) ->
          strf "%s, %s" (name g)
            (match s with Packet.Agent -> "agent" | System -> "system"))
        "an acquire invalidates the caches its scope names"
        (List.concat_map
           (fun g -> [ (g, Packet.Agent); (g, System) ])
           generations)
        (fun (g, scope) ->
          let all = scope = Packet.System in
          equal caches_w
            (true, true, true, all, all, all)
            (acquired g (words (Pm4.acquire_mem g scope))));
      cases
        ~name:(fun (((g : Gpu.t), s), _) ->
          strf "%s, %s" (name g)
            (match s with Packet.Agent -> "agent" | System -> "system"))
        "a release places its scope's cache operations" release_caches
        (fun ((g, scope), bits) ->
          equal int (release_event lor bits)
            (List.nth (words (Pm4.release_mem g scope 0 (Low_32 0))) 1));
      cases ~name "an interrupt carries its id's low 32 bits" generations
        (fun g ->
          let ws =
            words
              (Pm4.release_mem g Agent ~interrupt:0x1_0000_0077 0 (Low_32 0))
          in
          equal (pair int int) (2, 0x77)
            ((List.nth ws 2 lsr 24) land 7, List.nth ws 7));
      cases ~name "a release without an interrupt raises none" generations
        (fun g ->
          let ws = words (Pm4.release_mem g System 0 (Data_64 0)) in
          equal int 0 ((List.nth ws 2 lsr 24) land 7));
      test "a partial flush" (fun () ->
          equal (list int)
            [ packet3 0x46 0; 7 lor (4 lsl 8) ]
            (words (Pm4.event_write Cs_partial_flush)));
      test "a thread trace's marker and finish" (fun () ->
          equal
            (pair (list int) (list int))
            ([ packet3 0x46 0; 0x35 ], [ packet3 0x46 0; 0x37 ])
            ( words (Pm4.event_write Thread_trace_marker),
              words (Pm4.event_write Thread_trace_finish) ));
    ]

let pp_word ppf : int Packet.word -> unit = function
  | Dword n -> Format.fprintf ppf "Dword 0x%x" n
  | W32 _ -> Format.fprintf ppf "W32 _"
  | W64 _ -> Format.fprintf ppf "W64 _"

(* Packets of any words over integers. *)
let packets =
  let open Gen in
  let addr = int_range 0 ((1 lsl 48) - 1) in
  with_pp
    (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_word)
    (list ~size:(int_range 0 40)
       (frequency
          [
            (2, map (fun n -> Packet.Dword n) (int_range 0 0xffff_ffff));
            (1, map (fun a -> Packet.W32 (Value a)) addr);
            (1, map (fun a -> Packet.W64 (Value a)) addr);
          ]))

let control =
  group ~timeout "control"
    [
      prop "a predicated block is its header, its mask and count, then itself"
        (Gen.pair (Gen.int_range 0 255) packets)
        (fun (xcc_mask, p) ->
          equal (list int)
            ([ packet3 0x23 0; (xcc_mask lsl 24) lor Rig_packet.size p ]
            @ words p)
            (words (Pm4.pred_exec ~xcc_mask p)));
      test "a predicated block of 16383 words" (fun () ->
          equal int 16385
            (Rig_packet.size
               (Pm4.pred_exec ~xcc_mask:0xff
                  (List.init 16383 (fun _ -> Packet.Dword 0)))));
      cases ~name:string_of_int "a die mask past 8 bits is refused"
        [ -1; 0x100 ] (fun xcc_mask ->
          raises_match (Exn.invalid_arg ~substring:"Pm4.pred_exec") (fun () ->
              Pm4.pred_exec ~xcc_mask []));
      test "a predicated block past 16383 words is refused" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"Pm4.pred_exec") (fun () ->
              Pm4.pred_exec ~xcc_mask:1
                (List.init 16384 (fun _ -> Packet.Dword 0))));
      test "an indirect buffer" (fun () ->
          equal (list int)
            [ packet3 0x3f 2; 0x100; 0x1; 16 lor (1 lsl 23) ]
            (words (Pm4.indirect_buffer 0x1_0000_0100 ~dwords:16)));
      test "an indirect buffer of 2^20 - 1 words leaves CHAIN clear" (fun () ->
          equal int
            (((1 lsl 20) - 1) lor (1 lsl 23))
            (List.nth
               (words (Pm4.indirect_buffer 0 ~dwords:((1 lsl 20) - 1)))
               3));
    ]

let kernel : Code_object.kernel =
  {
    descriptor = 0x1000;
    entry = 0x1100;
    group_segment = 1024;
    private_segment = 0;
    kernarg_size = 24;
    rsrc1 = 0x60af0000;
    rsrc2 = 0x1384;
    rsrc3 = 0;
    wave32 = true;
    dispatch_ptr = false;
    private_segment_buffer = false;
  }

let dispatch g =
  Pm4.dispatch g kernel ~program:0x1_0000_1100 ~scratch:0x2_0000_0000
    ~args:0x3_0000_0000 ~packet:0 ~threads:(64, 1, 1) ~groups:(2, 3, 4) ()

(* Kernels as compilers describe them: no privilege in COMPUTE_PGM_RSRC1 (bit
   20) and no LDS in COMPUTE_PGM_RSRC2 (bits 15 to 23), which AMDGPUUsage says
   the descriptor leaves 0. *)
let rsrc1_priv = 1 lsl 20
let rsrc2_lds = 0x1ff lsl 15

let pp_kernel ppf (k : Code_object.kernel) =
  Format.fprintf ppf
    "{ group = %d; private = %d; rsrc1 = 0x%x; rsrc2 = 0x%x; rsrc3 = 0x%x; \
     wave32 = %b; dispatch_ptr = %b; private_segment_buffer = %b }"
    k.group_segment k.private_segment k.rsrc1 k.rsrc2 k.rsrc3 k.wave32
    k.dispatch_ptr k.private_segment_buffer

let kernels =
  let open Gen in
  let u32 = int_range 0 0xffff_ffff in
  with_pp pp_kernel
    (let+ group_segment = int_range 0 65536
     and+ private_segment = int_range 0 (1 lsl 16)
     and+ rsrc = triple u32 u32 u32
     and+ flags = triple bool bool bool in
     let rsrc1, rsrc2, rsrc3 = rsrc and wave32, dispatch_ptr, psb = flags in
     {
       Code_object.descriptor = 0;
       entry = 0;
       group_segment;
       private_segment;
       kernarg_size = 0;
       rsrc1 = rsrc1 land lnot rsrc1_priv;
       rsrc2 = rsrc2 land lnot rsrc2_lds;
       rsrc3;
       wave32;
       dispatch_ptr;
       private_segment_buffer = psb;
     })

type launch = {
  g : Gpu.t;
  k : Code_object.kernel;
  program : int;
  scratch : int;
  args : int;
  packet : int;
  threads : int * int * int;
  groups : int * int * int;
  waves : int option;
}

let launches =
  let open Gen in
  let aligned n = map (fun a -> a * n) (int_range 0 (((1 lsl 48) - 1) / n)) in
  let side = int_range 1 1024 in
  let count = int_range 1 0x7fff_ffff in
  with_pp
    (fun ppf l ->
      let x, y, z = l.threads and gx, gy, gz = l.groups in
      Format.fprintf ppf
        "GC %s, %a, program 0x%x, scratch 0x%x, args 0x%x, packet 0x%x, \
         threads (%d, %d, %d), groups (%d, %d, %d), waves %s"
        (S.version l.g.gc) pp_kernel l.k l.program l.scratch l.args l.packet x y
        z gx gy gz
        (Option.fold ~none:"none" ~some:string_of_int l.waves))
    (let+ g = of_list generations
     and+ k = kernels
     and+ program, scratch = pair (aligned 256) (aligned 256)
     and+ args, packet = pair (aligned 16) (aligned 64)
     and+ threads = triple side side side
     and+ groups = triple count count count
     and+ waves = option (int_range 1 1023) in
     { g; k; program; scratch; args; packet; threads; groups; waves })

let launch l =
  Pm4.dispatch l.g l.k ~program:l.program ~scratch:l.scratch ~args:l.args
    ~packet:l.packet ~threads:l.threads ~groups:l.groups
    ?waves_per_array:l.waves ()

let address g name = Register.address g (require_some (Register.find g name))
let lo n = n land 0xffff_ffff
let hi n = n lsr 32

(* The registers a launch sets, as the .mli states them, at the addresses of the
   GC's headers: each _HI register follows its _LO, RSRC2 follows RSRC1, and
   NUM_THREAD_X to _Z follow START_X to _Z. *)
let expected l =
  let g = l.g and k = l.k in
  let gfx11 = match g.gc with 11, _, _ -> true | _ -> false in
  let x, y, z = l.threads in
  let pgm = address g "regCOMPUTE_PGM_LO"
  and rsrc1 = address g "regCOMPUTE_PGM_RSRC1"
  and threads = address g "regCOMPUTE_START_X" + 3
  and user = address g "regCOMPUTE_USER_DATA_0"
  and scratch = address g "regCOMPUTE_DISPATCH_SCRATCH_BASE_LO" in
  (* A scratch descriptor is 4 words: the address's low 32 bits, then its bits
     32 to 47 under the descriptor's fields, which a law cannot state. *)
  let user_data =
    (if k.private_segment_buffer then
       [ Some (lo l.scratch); Some (hi l.scratch); None; None ]
     else [])
    @ (if k.dispatch_ptr then [ Some (lo l.packet); Some (hi l.packet) ] else [])
    @ [ Some (lo l.args); Some (hi l.args) ]
  in
  [
    (pgm, lo (l.program lsr 8));
    (pgm + 1, hi (l.program lsr 8));
    (rsrc1, if gfx11 then k.rsrc1 lor rsrc1_priv else k.rsrc1);
    (address g "regCOMPUTE_PGM_RSRC3", k.rsrc3);
    (address g "regCOMPUTE_TMPRING_SIZE", Scratch.tmpring g k.private_segment);
    (scratch, lo (l.scratch lsr 8));
    (scratch + 1, hi (l.scratch lsr 8));
    (threads, x);
    (threads + 1, y);
    (threads + 2, z);
    ( address g "regCOMPUTE_RESOURCE_LIMITS",
      Register.encode
        (require_some (Register.find g "regCOMPUTE_RESOURCE_LIMITS"))
        [ ("waves_per_sh", Option.value ~default:0 l.waves) ] );
  ]
  @ List.concat
      (List.mapi
         (fun i v -> Option.fold ~none:[] ~some:(fun v -> [ (user + i, v) ]) v)
         user_data)

let initiator (g : Gpu.t) (k : Code_object.kernel) =
  let r = require_some (Register.find g "regCOMPUTE_DISPATCH_INITIATOR") in
  let w32 = k.wave32 && List.mem_assoc "cs_w32_en" r.fields in
  Register.encode r
    ([ ("compute_shader_en", 1); ("force_start_at_000", 1) ]
    @ if w32 then [ ("cs_w32_en", 1) ] else [])

let writes l = S.writes (words (launch l))
let fst3 (a, _, _) = a

(* The LDS_SIZE field of the COMPUTE_PGM_RSRC2 a dispatch sets. *)
let lds g group_segment =
  let k = { kernel with group_segment; rsrc2 = 0 } in
  let ws =
    S.writes
      (words
         (Pm4.dispatch g k ~program:0 ~scratch:0 ~args:0 ~packet:0
            ~threads:(1, 1, 1) ~groups:(1, 1, 1) ()))
  in
  (List.assoc (address g "regCOMPUTE_PGM_RSRC1" + 1) ws lsr 15) land 0x1ff

let runs =
  group ~timeout "runs"
    [
      prop "a dispatch sets the registers of its kernel and arguments" launches
        (fun l ->
          let ws = writes l in
          cover "a scratch descriptor" l.k.private_segment_buffer;
          cover "a dispatch packet" l.k.dispatch_ptr;
          List.iter
            (fun (a, v) ->
              let v' =
                require_some ~msg:(strf "0x%x is set" a) (List.assoc_opt a ws)
              in
              (* A scratch descriptor's second word holds the address's bits 32
                 to 47, and fields above them. *)
              let v' =
                if
                  l.k.private_segment_buffer
                  && a = address l.g "regCOMPUTE_USER_DATA_0" + 1
                then v' land 0xffff
                else v'
              in
              equal ~msg:(strf "0x%x" a) int v v')
            (expected l));
      prop "a dispatch's other words leave its kernel's resources" launches
        (fun l ->
          let rsrc2 =
            List.assoc (address l.g "regCOMPUTE_PGM_RSRC1" + 1) (writes l)
          in
          equal int l.k.rsrc2 (rsrc2 land lnot rsrc2_lds));
      prop "a dispatch ends in DISPATCH_DIRECT of its groups" launches (fun l ->
          let gx, gy, gz = l.groups in
          match List.rev (S.packets (words (launch l))) with
          | (op, body) :: rest ->
              equal
                (pair int (list int))
                (0x15, [ gx; gy; gz; initiator l.g l.k ])
                (op, body);
              List.iter
                (fun (op, _) -> equal ~msg:"SET_SH_REG before it" int 0x76 op)
                rest
          | [] -> fail "no packet");
      cases
        ~name:(fun (t, g) -> strf "%s, %d bytes" (S.version t) g)
        "a GFX9 or GFX11 workgroup's LDS, in 512-byte units"
        [
          ((9, 4, 2), 0);
          ((9, 4, 2), 1);
          ((9, 4, 2), 512);
          ((9, 4, 2), 513);
          ((11, 0, 0), 65536);
          ((11, 0, 0), 65535);
        ]
        (fun (target, group_segment) ->
          let g =
            S.gpu ~target (if fst3 target = 9 then (9, 4, 3) else target)
          in
          equal int ((group_segment + 511) / 512) (lds g group_segment));
      test "a GFX950 workgroup's LDS, in 1280-byte units" (fun () ->
          equal int 1 (lds (S.gpu ~target:(9, 5, 0) (9, 5, 0)) 1280));
      (* LDS_SIZE holds 511 units: of 1280 bytes on GFX950, 512 on the
         others. *)
      cases
        ~name:(fun (g, n, _) -> strf "GC %s, %d bytes" (S.version g.Gpu.gc) n)
        "a workgroup's LDS is set up to 511 units and refused past them"
        [
          (gfx11, 511 * 512, true);
          (gfx11, (511 * 512) + 1, false);
          (gfx9, (511 * 512) + 1, false);
          (S.gpu ~target:(9, 5, 0) (9, 5, 0), 511 * 1280, true);
          (S.gpu ~target:(9, 5, 0) (9, 5, 0), (511 * 1280) + 1, false);
        ]
        (fun (g, n, taken) ->
          if taken then equal int 511 (lds g n)
          else
            raises_match (Exn.invalid_arg ~substring:"Pm4.dispatch") (fun () ->
                lds g n));
      test "a kernel's scratch past what WAVESIZE holds is refused" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"Scratch.tmpring") (fun () ->
              Pm4.dispatch gfx11
                { kernel with private_segment = (32767 * 256 / 64) + 1 }
                ~program:0 ~scratch:0 ~args:0 ~packet:0 ~threads:(1, 1, 1)
                ~groups:(1, 1, 1) ()));
      cases ~name:string_of_int "a wave limit outside 10 bits is refused"
        [ 0; 1024 ] (fun n ->
          raises_match (Exn.invalid_arg ~substring:"waves_per_array") (fun () ->
              Pm4.dispatch gfx11 kernel ~program:0 ~scratch:0 ~args:0 ~packet:0
                ~threads:(1, 1, 1) ~groups:(1, 1, 1) ~waves_per_array:n ()));
      test "a GC with no dispatch registers is refused" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"Pm4.dispatch") (fun () ->
              dispatch (gpu (10, 3, 0))));
      prop "a run is an agent acquire, its words, then a partial flush"
        (Gen.pair (Gen.of_list generations) packets)
        (fun (g, p) ->
          equal (list int)
            (words (Pm4.acquire_mem g Agent)
            @ words p
            @ words (Pm4.event_write Cs_partial_flush))
            (words (Pm4.run g p)));
    ]

let () =
  exit (run "rig_amd_abi.pm4" [ memory; waits; caches; control; runs ])
