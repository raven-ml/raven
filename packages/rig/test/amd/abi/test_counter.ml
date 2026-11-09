(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Counters of rocprofiler's counter_defs.yaml
   (lib/amd/gen/headers/rocm/counter_defs.yaml) and the layout of a run's
   samples, as counter.mli states it. *)

open Windtrap
open Rig_amd_abi
module S = Rig_amd_abi_support

let timeout = S.timeout

(* An R9700's GC: one die of 4 engines and 64 compute units, 8 to an array. *)
let gfx12 =
  S.gpu ~target:(12, 0, 1) ~shader_engines:4 ~compute_units:64 ~array_units:8
    (12, 0, 1)

(* An RX 9070's: the same arrays, with 8 compute units that run no work. *)
let rx9070 = { gfx12 with compute_units = 56 }

let gfx11 = S.gpu ~shader_engines:6 ~compute_units:48 (11, 0, 0)

let gfx942 =
  S.gpu ~target:(9, 4, 2) ~sdma:(4, 4, 2) ~xccs:8 ~compute_units:38 (9, 4, 3)

let gfx90a = S.gpu ~target:(9, 0, 10) (9, 4, 3)

let counter =
  Testable.make
    ~pp:(fun ppf (c : Counter.t) ->
      Format.fprintf ppf "{%s %s/%d register %d; %d x %d x %d x %d at %d}"
        c.name c.block c.event c.register c.instances c.engines c.arrays c.wgps
        c.offset)
    ~equal:( = )

let layout =
  Testable.make
    ~pp:(fun ppf -> function
      | Ok (l : Counter.layout) ->
          Format.fprintf ppf "Ok {%a; %d bytes}"
            (Format.pp_print_list (Testable.pp counter))
            l.counters l.bytes
      | Error e -> Format.fprintf ppf "Error %S" e)
    ~equal:( = )

let c name block event register (instances, engines, arrays, wgps) offset =
  {
    Counter.name;
    block;
    event;
    register;
    instances;
    engines;
    arrays;
    wgps;
    offset;
  }

(* [l] with each element's repeats dropped. *)
let distinct l =
  List.rev
    (List.fold_left
       (fun acc x -> if List.mem x acc then acc else x :: acc)
       [] l)

(* A GPU of each processor the table knows, and a profile of its counters, each
   listed once. *)
let profiles =
  Gen.with_pp
    (fun ppf (g, names) ->
      Format.fprintf ppf "%s: %s" (Gpu.processor g) (String.concat ", " names))
    (let open Gen in
     let* g = of_list [ gfx12; gfx11; gfx942 ] in
     let names = Array.of_list (Counter.names g) in
     let+ picks =
       list ~size:(int_range 0 16) (int_range 0 (Array.length names - 1))
     in
     (g, distinct (List.map (fun i -> names.(i)) picks)))

(* The counter registers of [block] on [g]: its [_LO] registers the GC's facts
   list, from [0]. *)
let slots g block =
  let rec go n =
    match Register.find g (Printf.sprintf "reg%s_PERFCOUNTER%d_LO" block n) with
    | Some _ -> go (n + 1)
    | None -> n
  in
  go 0

(* Whether [names] asks more counters of a block than its registers. *)
let overflows g names =
  let block name =
    match Counter.layout g [ name ] with
    | Ok { counters = [ c ]; _ } -> c.block
    | _ -> fail ("no block for " ^ name)
  in
  let blocks = List.map block names in
  List.exists
    (fun b -> List.length (List.filter (( = ) b) blocks) > slots g b)
    blocks

let laws =
  group ~timeout "layout"
    [
      prop "a profile is refused exactly when a block lacks registers" profiles
        (fun (g, names) ->
          let over = overflows g names in
          cover "fits" (not over);
          cover "overflows" over;
          match Counter.layout g names with
          | Ok l ->
              equal bool ~msg:"overflows" false over;
              List.iter
                (fun (c : Counter.t) ->
                  less int ~msg:c.name ~than:(slots g c.block) c.register)
                l.counters
          | Error e -> equal bool ~msg:e true over);
      prop "a profile listing a counter twice is refused, naming it"
        Gen.(pair profiles nat)
        (fun ((g, names), i) ->
          assume (names <> []);
          let twice = List.nth names (i mod List.length names) in
          equal layout
            (Error (twice ^ " is listed twice"))
            (Counter.layout g (names @ [ twice ])));
      prop "a run's samples are each counter's values, one after another"
        profiles (fun (g, names) ->
          match Counter.layout g names with
          | Error _ -> reject ()
          | Ok l ->
              let size (c : Counter.t) =
                8 * g.xccs * c.instances * c.engines * c.arrays * c.wgps
              in
              let rec check offset seen = function
                | [] -> equal int ~msg:"bytes" offset l.bytes
                | (c : Counter.t) :: rest ->
                    equal int ~msg:"offset" offset c.offset;
                    equal int ~msg:"register"
                      (List.length (List.filter (( = ) c.block) seen))
                      c.register;
                    check (offset + size c) (c.block :: seen) rest
              in
              equal (list string) ~msg:"names" names
                (List.map (fun (c : Counter.t) -> c.name) l.counters);
              check 0 [] l.counters);
    ]

let cases =
  group ~timeout "cases"
    [
      test "an R9700 counts its engines' work-group processors" (fun () ->
          equal layout
            (Ok
               {
                 counters =
                   [
                     c "GRBM_GUI_ACTIVE" "GRBM" 2 0 (1, 1, 1, 1) 0;
                     c "SQ_WAVES" "SQ" 4 0 (1, 4, 2, 4) 8;
                     c "GL2C_HIT" "GL2C" 41 0 (32, 1, 1, 1) 264;
                     c "SQ_BUSY_CYCLES" "SQ" 3 1 (1, 4, 2, 4) 520;
                   ];
                 bytes = 776;
               })
            (Counter.layout gfx12
               [ "GRBM_GUI_ACTIVE"; "SQ_WAVES"; "GL2C_HIT"; "SQ_BUSY_CYCLES" ]));
      test
        "an RX 9070 counts every work-group processor of its arrays, those \
         that run no work included" (fun () ->
          equal layout
            (Ok
               {
                 counters = [ c "SQ_WAVES" "SQ" 4 0 (1, 4, 2, 4) 0 ];
                 bytes = 256;
               })
            (Counter.layout rx9070 [ "SQ_WAVES" ]));
      test "a gfx942 counts its SQ per engine, on each of its dies" (fun () ->
          equal layout
            (Ok
               {
                 counters =
                   [
                     c "SQ_WAVES" "SQ" 4 0 (1, 4, 1, 1) 0;
                     c "TCC_HIT" "TCC" 17 0 (16, 1, 1, 1) 256;
                   ];
                 bytes = 256 + (8 * 8 * 16);
               })
            (Counter.layout gfx942 [ "SQ_WAVES"; "TCC_HIT" ]));
      test "a counter the GPU lacks is refused, naming it" (fun () ->
          equal layout (Error "gfx1201 counts no SQ_FOO")
            (Counter.layout gfx12 [ "SQ_WAVES"; "SQ_FOO" ]));
      test "a fifth GL2C counter on an R9700 is refused, naming the block"
        (fun () ->
          equal layout (Error "gfx1201 counts at most 4 GL2C counters at once")
            (Counter.layout gfx12
               [
                 "GL2C_HIT";
                 "GL2C_MISS";
                 "GL2C_EA_RDREQ";
                 "GL2C_EA_WRREQ";
                 "GL2C_EA_WRREQ_STALL";
               ]));
      test "a processor the table lacks counts nothing" (fun () ->
          equal (list string) [] (Counter.names gfx90a);
          equal layout (Error "gfx90a counts no SQ_WAVES")
            (Counter.layout gfx90a [ "SQ_WAVES" ]));
      test "a GPU's names are in increasing order" (fun () ->
          let names = Counter.names gfx12 in
          equal (list string) (List.sort String.compare names) names);
    ]

(* Counting *)

let strf = Printf.sprintf

(* PACKET3's opcodes (soc15d.h, nvd.h): COPY_DATA, whose body is its control,
   its source's two words and its destination's; WRITE_DATA, whose body is its
   control, its destination's two words and the word; EVENT_WRITE, whose body is
   the event. COPY_DATA's control selects its source in bits 0 to 3, a counter
   register (4) or memory through the L2 (2), and its destination in bits 8 to
   11, memory through the L2 (2) or a register (0). *)
let copy_data = 0x40
let write_data = 0x37
let event_write = 0x46
let from_counter = 4
let from_memory = 2
let reg g name = require_some ~msg:name (Register.find g name)
let at g name = Register.address g (reg g name)

(* Field [f] of register [r] in [v]. *)
let field (r : Register.t) f v =
  let lo, hi = List.assoc f r.fields in
  (v lsr lo) land ((1 lsl (hi - lo + 1)) - 1)

(* What a program does, packet by packet: a register write; a copy of a counter
   register to memory, under the GRBM_GFX_INDEX last written; a copy of a word
   of memory to a register; a word written to memory; an event. *)
type step =
  | Set of int * int
  | Copy of { src : int; dst : int; grbm : int }
  | Load of { src : int; dst : int }
  | Data of { dst : int; word : int }
  | Event of int

let steps (g : Gpu.t) s =
  let grbm_at = at g "regGRBM_GFX_INDEX" and grbm = ref (-1) in
  let sets start = function
    | [] -> []
    | off :: vs ->
        List.mapi
          (fun i v ->
            let a = start + off + i in
            if a = grbm_at then grbm := v;
            Set (a, v))
          vs
  in
  let step (op, body) =
    match body with
    | [ control; src; _; lo; hi ]
      when op = copy_data && control land 0xf = from_counter ->
        [ Copy { src; dst = lo lor (hi lsl 32); grbm = !grbm } ]
    | [ control; lo; hi; dst; _ ]
      when op = copy_data && control land 0xf = from_memory ->
        [ Load { src = lo lor (hi lsl 32); dst } ]
    | [ _; lo; hi; word ] when op = write_data ->
        [ Data { dst = lo lor (hi lsl 32); word } ]
    | e :: _ when op = event_write -> [ Event e ]
    | _ when op = S.set_uconfig_reg -> sets S.uconfig_start body
    | _ when op = S.set_sh_reg -> sets S.sh_start body
    | _ -> failf "a packet of opcode 0x%x" op
  in
  List.concat_map step (S.packets (S.words s))

let cs_partial_flush = List.nth (S.encode (Pm4.event_write Cs_partial_flush)) 1

(* CP_PERFMON_CNTL, named CP_PERFMON_CNTL_1 on GFX12. *)
let perfmon g = reg g "regCP_PERFMON_CNTL_1"

(* GFX12 GPUs of one die, of 1 to 4 engines whose arrays hold 1 to 4 work-group
   processors. *)
let counting_gpus =
  let open Gen in
  let+ shader_engines = int_range 1 4 and+ wgps = int_range 1 4 in
  S.gpu ~shader_engines ~array_units:(2 * wgps) (12, 0, 0)

let lay g names =
  match Counter.layout g names with Ok l -> l | Error e -> fail e

let block g name = (List.hd (lay g [ name ]).counters).block

(* A profile of [g]'s SQ counters, each listed once, that [g] counts at once. *)
let sq_profile g =
  let names =
    Array.of_list (List.filter (fun n -> block g n = "SQ") (Counter.names g))
  in
  Gen.map
    (fun picks ->
      let names = distinct (List.map (fun i -> names.(i)) picks) in
      if overflows g names then [] else names)
    (Gen.list ~size:(Gen.int_range 0 8)
       (Gen.int_range 0 (Array.length names - 1)))

(* The work-group processors that run work, [wgps.(e).(a)] as a capability gives
   them: every one, or some. *)
let harvests (g : Gpu.t) =
  let full = (1 lsl (g.array_units / 2)) - 1 in
  let mask =
    Gen.frequency [ (1, Gen.constant full); (2, Gen.int_range 0 full) ]
  in
  Gen.map
    (fun ms -> Array.of_list (List.map Array.of_list ms))
    (Gen.list
       ~size:(Gen.constant g.shader_engines)
       (Gen.list ~size:(Gen.constant 2) mask))

let pp_gpu ppf (g : Gpu.t) =
  Format.fprintf ppf "GC %s, %d engines of %d processors an array"
    (S.version g.gc) g.shader_engines (g.array_units / 2)

let pp_harvest ppf wgps =
  Format.pp_print_string ppf
    (String.concat " "
       (Array.to_list
          (Array.map
             (fun a ->
               String.concat "," (Array.to_list (Array.map string_of_int a)))
             wgps)))

(* A store of a stop: a copy of a register under a GRBM_GFX_INDEX, or a word. *)
type store = Copied of int * int | Written of int

let store =
  Testable.make
    ~pp:(fun ppf (dst, s) ->
      match s with
      | Copied (src, grbm) ->
          Format.fprintf ppf "0x%x: 0x%x under GRBM 0x%x" dst src grbm
      | Written w -> Format.fprintf ppf "0x%x: word 0x%x" dst w)
    ~equal:( = )

(* What a stop must store at each word of [l]'s samples from [address]: each
   value's low and high halves, copied from its counter's registers with
   GRBM_GFX_INDEX selecting its engine, array and processor, the processor the
   instance from bit 2 (aqlprofile's grbm_se_sh_wgp_index_value), or 0 for a
   processor that runs no work, and 0 for every high half. SQ counter [n] of a
   processor is SQ_PERFCOUNTERn_LO, 32 bits. *)
let expected_stores g ~wgps (l : Counter.layout) address =
  let grbm = reg g "regGRBM_GFX_INDEX" in
  List.concat_map
    (fun (c : Counter.t) ->
      let lo = at g (strf "regSQ_PERFCOUNTER%d_LO" c.register) in
      List.concat
        (List.init (c.engines * c.arrays * c.wgps) (fun i ->
             let wgp = i mod c.wgps
             and sa = i / c.wgps mod c.arrays
             and se = i / (c.wgps * c.arrays) in
             let dst = address + c.offset + (8 * i) in
             let low =
               if wgps.(se).(sa) land (1 lsl wgp) = 0 then Written 0
               else
                 Copied
                   ( lo,
                     Register.encode grbm
                       [
                         ("se_index", se);
                         ("sa_index", sa);
                         ("instance_index", wgp lsl 2);
                       ] )
             in
             [ (dst, low); (dst + 4, Written 0) ])))
    l.counters

let stores steps =
  List.filter_map
    (function
      | Copy { src; dst; grbm } -> Some (dst, Copied (src, grbm))
      | Data { dst; word } -> Some (dst, Written word)
      | Set _ | Load _ | Event _ -> None)
    steps

(* The index in [l] of the first element that [p] holds of, and of the
   last. *)
let first p l =
  let rec go i = function
    | [] -> max_int
    | x :: rest -> if p x then i else go (i + 1) rest
  in
  go 0 l

let last p l =
  let n = List.length l in
  match first p (List.rev l) with i when i = max_int -> -1 | i -> n - 1 - i

let stop_law =
  let gen =
    Gen.with_pp
      (fun ppf (g, names, wgps, address) ->
        Format.fprintf ppf "%a; %s; processors %a; at 0x%x" pp_gpu g
          (String.concat ", " names) pp_harvest wgps address)
      (let open Gen in
       let* g = counting_gpus in
       let+ names = sq_profile g
       and+ wgps = harvests g
       and+ address = map (fun a -> 8 * a) (int_range 0 (1 lsl 40)) in
       (g, names, wgps, address))
  in
  prop
    "a stop stores each value at its place, once, and 0 where no work runs, \
     then clears the filter"
    gen (fun (g, names, wgps, address) ->
      let l = lay g names in
      let idle =
        Array.exists
          (Array.exists (fun m -> m <> (1 lsl (g.array_units / 2)) - 1))
          wgps
      in
      cover "two counters or more" (List.length l.counters >= 2);
      cover "a processor that runs no work" idle;
      let ss = steps g (Counter.stop g ~wgps l ~address) in
      let sort = List.sort compare in
      equal (list store)
        (sort (expected_stores g ~wgps l address))
        (sort (stores ss));
      (* The copies follow the stop, which follows the dispatches before. *)
      let pm = perfmon g in
      let stop = function
        | Set (a, v) ->
            a = Register.address g pm
            && field pm "perfmon_state" v = 2
            && field pm "perfmon_sample_enable" v = 1
        | _ -> false
      in
      let copy = function Copy _ | Data _ -> true | _ -> false in
      let flush = function Event e -> e = cs_partial_flush | _ -> false in
      less int ~msg:"a stop" ~than:max_int (first stop ss);
      less int ~msg:"the flush before the stop" ~than:(first stop ss)
        (first flush ss);
      if List.exists copy ss then
        less int ~msg:"the stop before the copies" ~than:(first copy ss)
          (first stop ss);
      (* The filter is cleared after the last store. *)
      let ctrl2 = reg g "regSQ_PERFCOUNTER_CTRL2" in
      let clear = function
        | Set (a, v) -> a = Register.address g ctrl2 && v = 0
        | _ -> false
      in
      equal int ~msg:"the filter cleared last" (List.length ss - 1)
        (last clear ss);
      less int ~msg:"the stores before the clear" ~than:(last clear ss)
        (last copy ss))

(* A start selects SQ counter [n]'s event in SQ_PERFCOUNTER(2n)_SELECT, the
   select of the low of the two 32-bit counters it takes; enables compute waves;
   copies the filter word into SQ_PERFCOUNTER_CTRL2; and starts once the
   dispatches before it complete and the counters are reset. *)
let start_law =
  let gen =
    Gen.with_pp
      (fun ppf (g, names, filter_address) ->
        Format.fprintf ppf "%a; %s; filter at 0x%x" pp_gpu g
          (String.concat ", " names) filter_address)
      (let open Gen in
       let* g = counting_gpus in
       let+ names = sq_profile g
       and+ filter_address = map (fun a -> 4 * a) (int_range 0 (1 lsl 40)) in
       (g, names, filter_address))
  in
  prop "a start selects each counter and counts the waves its filter selects"
    gen (fun (g, names, filter_address) ->
      let l = lay g names in
      cover "two counters or more" (List.length l.counters >= 2);
      let ss = steps g (Counter.start g l ~filter_address) in
      let sets =
        List.filter_map (function Set (a, v) -> Some (a, v) | _ -> None) ss
      in
      let last_set name =
        let r = reg g name in
        match
          List.rev (List.filter (fun (a, _) -> a = Register.address g r) sets)
        with
        | (_, v) :: _ -> (r, v)
        | [] -> failf "no %s" name
      in
      let ctrl, v = last_set "regSQ_PERFCOUNTER_CTRL" in
      equal int ~msg:"CS_EN" 1 (field ctrl "cs_en" v);
      List.iter
        (fun (c : Counter.t) ->
          let r, v = last_set (strf "regSQ_PERFCOUNTER%d_SELECT" (2 * c.register)) in
          equal int ~msg:(c.name ^ "'s select") c.event (field r "perf_sel" v))
        l.counters;
      let ctrl2 = at g "regSQ_PERFCOUNTER_CTRL2" in
      let load = function
        | Load { src; dst } -> src = filter_address && dst = ctrl2
        | _ -> false
      in
      equal int ~msg:"one load of the filter" 1
        (List.length (List.filter load ss));
      equal bool ~msg:"no other write of SQ_PERFCOUNTER_CTRL2" false
        (List.exists (fun (a, _) -> a = ctrl2) sets);
      let pm = perfmon g in
      let state n = function
        | Set (a, v) ->
            a = Register.address g pm && field pm "perfmon_state" v = n
        | _ -> false
      in
      let flush = function Event e -> e = cs_partial_flush | _ -> false in
      equal int ~msg:"the flush first" 0 (first flush ss);
      equal int ~msg:"the start last" (List.length ss - 1) (first (state 1) ss);
      less int ~msg:"the reset before the filter" ~than:(first load ss)
        (first (state 0) ss);
      less int ~msg:"the filter before the start" ~than:(first (state 1) ss)
        (first load ss))

(* The programs for one engine of one processor an array, the second running no
   work, counting SQ_WAVES and SQ_INSTS_VALU with the filter word at 0x800 into
   the samples at 0x1000. A baseline, learned from the code: the laws above
   state what it must hold. *)
let programs () =
  let gc = (12, 0, 0) in
  let g = S.gpu ~shader_engines:1 ~array_units:2 gc in
  let l = lay g [ "SQ_WAVES"; "SQ_INSTS_VALU" ] in
  let named = Hashtbl.create 1024 in
  List.iter
    (fun (r : Register.t) ->
      Hashtbl.replace named (Register.address g r)
        (String.sub r.name 3 (String.length r.name - 3)))
    (Register.registers g);
  let name a =
    match Hashtbl.find_opt named a with
    | Some n -> n
    | None -> (
        match Hashtbl.find_opt named (a - 1) with
        | Some n -> n ^ "+1"
        | None -> strf "0x%x" a)
  in
  let line = function
    | Set (a, v) -> strf "  %s 0x%x" (name a) v
    | Copy { src; dst; grbm } ->
        strf "  copy %s to +0x%x, GRBM 0x%x" (name src) (dst - 0x1000) grbm
    | Load { src; dst } -> strf "  load 0x%x to %s" src (name dst)
    | Data { dst; word } -> strf "  write 0x%x to +0x%x" word (dst - 0x1000)
    | Event e -> strf "  event 0x%x" e
  in
  let program what s =
    strf "GC %s %s:" (S.version gc) what :: List.map line (steps g s)
  in
  String.concat "\n"
    (program "start" (Counter.start g l ~filter_address:0x800)
    @ program "stop" (Counter.stop g ~wgps:[| [| 1; 0 |] |] l ~address:0x1000))

let no_gfx12 g = strf "%s is no GFX12 GPU of one die" (Gpu.processor g)

let counting =
  group ~timeout "counting"
    [
      start_law;
      stop_law;
      test "a filter counts one VMID's waves, forced" (fun () ->
          let ctrl2 = reg gfx12 "regSQ_PERFCOUNTER_CTRL2" in
          for vmid = 0 to 15 do
            let v = Counter.filter gfx12 ~vmid in
            let msg = strf "VMID %d" vmid in
            equal int ~msg (1 lor (1 lsl (vmid + 1))) v;
            equal int ~msg 1 (field ctrl2 "force_en" v);
            equal int ~msg (1 lsl vmid) (field ctrl2 "vmid_en" v)
          done);
      test "the programs, a baseline" (fun () ->
          expect (programs ()) @@ __POS_OF__ {|
            GC 12.0.0 start:
              event 0x407
              GRBM_GFX_INDEX 0xe0000000
              CP_PERFMON_CNTL_1 0x0
              SQ_PERFCOUNTER_CTRL 0x40
              load 0x800 to SQ_PERFCOUNTER_CTRL2
              SQ_PERFCOUNTER0_SELECT 0x4
              SQ_PERFCOUNTER2_SELECT 0x32
              COMPUTE_PERFCOUNT_ENABLE 0x1
              CP_PERFMON_CNTL_1 0x1
            GC 12.0.0 stop:
              event 0x407
              GRBM_GFX_INDEX 0xe0000000
              CP_PERFMON_CNTL_1 0x402
              event 0x407
              GRBM_GFX_INDEX 0x0
              copy SQ_PERFCOUNTER0_LO to +0x0, GRBM 0x0
              write 0x0 to +0x4
              write 0x0 to +0x8
              write 0x0 to +0xc
              GRBM_GFX_INDEX 0x0
              copy SQ_PERFCOUNTER1_LO to +0x10, GRBM 0x0
              write 0x0 to +0x14
              write 0x0 to +0x18
              write 0x0 to +0x1c
              GRBM_GFX_INDEX 0xe0000000
              SQ_PERFCOUNTER_CTRL2 0x0
            |});
      test "a VMID outside 0 to 15 is refused" (fun () ->
          List.iter
            (fun vmid ->
              raises
                (Invalid_argument
                   (strf "Counter.filter: VMID %d, expected 0 to 15" vmid))
                (fun () -> Counter.filter gfx12 ~vmid))
            [ -1; 16 ]);
      test "a GPU of GFX9 or GFX11 is refused" (fun () ->
          List.iter
            (fun g ->
              let l = lay g [ "SQ_WAVES" ] in
              raises
                (Invalid_argument ("Counter.filter: " ^ no_gfx12 g))
                (fun () -> Counter.filter g ~vmid:0);
              raises
                (Invalid_argument ("Counter.start: " ^ no_gfx12 g))
                (fun () -> Counter.start g l ~filter_address:0);
              raises
                (Invalid_argument ("Counter.stop: " ^ no_gfx12 g))
                (fun () ->
                  Counter.stop g ~wgps:(S.harvest_none g) l ~address:0))
            [ gfx942; gfx11 ]);
      test "a counter of a block other than SQ is refused" (fun () ->
          let others =
            List.filter (fun n -> block gfx12 n <> "SQ") (Counter.names gfx12)
          in
          greater int ~msg:"counters of other blocks" ~than:0
            (List.length others);
          List.iter
            (fun n ->
              let l = lay gfx12 [ "SQ_WAVES"; n ] in
              let no = strf "%s is no SQ counter" n in
              raises
                (Invalid_argument ("Counter.start: " ^ no))
                (fun () -> Counter.start gfx12 l ~filter_address:0);
              raises
                (Invalid_argument ("Counter.stop: " ^ no))
                (fun () ->
                  Counter.stop gfx12
                    ~wgps:(Array.make 4 [| 0xf; 0xf |])
                    l ~address:0))
            others);
      test "a misplaced filter or address, or wrong processors, is refused"
        (fun () ->
          let l = lay gfx12 [ "SQ_WAVES" ] in
          let wgps = Array.make 4 [| 0xf; 0xf |] in
          raises
            (Invalid_argument
               "Counter.start: filter address 0x2, expected a non-negative \
                multiple of 4") (fun () ->
              Counter.start gfx12 l ~filter_address:2);
          raises
            (Invalid_argument
               "Counter.stop: address 0x4, expected a non-negative multiple of \
                8") (fun () -> Counter.stop gfx12 ~wgps l ~address:4);
          raises
            (Invalid_argument "Counter.stop: wgps covers 3 engines, expected 4")
            (fun () ->
              Counter.stop gfx12
                ~wgps:(Array.make 3 [| 0xf; 0xf |])
                l ~address:0));
    ]

let () = exit (run "rig_amd_abi.counter" [ laws; cases; counting ])
