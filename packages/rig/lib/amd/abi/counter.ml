(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

type t = {
  name : string;
  block : string;
  event : int;
  register : int;
  instances : int;
  engines : int;
  arrays : int;
  wgps : int;
  offset : int;
}

type layout = { counters : t list; bytes : int }

(* The counters of the GPU's processor: the table rocprofiler defines for it,
   where GFX11 GPUs share one and GFX12 GPUs another, and for GFX12 GPUs
   SQ_WAVES_SAVED, which that table lacks, from SQ_PERF_SEL_WAVES_SAVED
   (0x14) of the SOC enumeration soc24_enum.h (gen.py's SOC_COUNTERS). *)
let table (g : Gpu.t) =
  match g.target with
  | 9, 4, 2 -> Defs.counters "gfx942"
  | 9, 5, 0 -> Defs.counters "gfx950"
  | 11, _, _ -> Defs.counters "gfx11"
  | 12, _, _ -> Defs.counters "gfx12"
  | _ -> [||]

let names g = Array.to_list (Array.map (fun (n, _, _) -> n) (table g))

(* GFX10 on: two shader arrays per engine, two compute units per work-group
   processor. *)
let arrays_per_engine = 2
let units_per_wgp = 2

(* Where a block's counter runs: (instances, engines, arrays, work-group
   processors). The SQ counts per engine on GFX9, and per work-group processor
   of each array after, those that run no work included. *)
let places (g : Gpu.t) = function
  | "GRBM" -> (1, 1, 1, 1)
  | "GL2C" -> (32, 1, 1, 1)
  | "TCC" -> (16, 1, 1, 1)
  | _ -> (
      match g.target with
      | 9, _, _ -> (1, g.shader_engines, 1, 1)
      | _ ->
          let wgps = g.array_units / units_per_wgp in
          (1, g.shader_engines, arrays_per_engine, wgps))

(* Whether [block] has counter register [n]. A block's counter registers are
   numbered from 0, so the first it lacks is how many it has. *)
let has g block n =
  Option.is_some (Register.find g (strf "reg%s_PERFCOUNTER%d_LO" block n))

(* The first name [names] lists twice, if any. *)
let rec twice = function
  | [] -> None
  | n :: rest -> if List.mem n rest then Some n else twice rest

let layout g names =
  let table = table g in
  let registers = Hashtbl.create 4 in
  let taken block =
    Option.value ~default:0 (Hashtbl.find_opt registers block)
  in
  let rec go offset acc = function
    | [] -> Ok { counters = List.rev acc; bytes = offset }
    | name :: rest -> (
        match Array.find_opt (fun (n, _, _) -> n = name) table with
        | None -> Error (strf "%s counts no %s" (Gpu.processor g) name)
        | Some (_, block, _) when not (has g block (taken block)) ->
            Error
              (strf "%s counts at most %d %s counters at once" (Gpu.processor g)
                 (taken block) block)
        | Some (_, block, event) ->
            let register = taken block in
            Hashtbl.replace registers block (register + 1);
            let instances, engines, arrays, wgps = places g block in
            let c =
              {
                name;
                block;
                event;
                register;
                instances;
                engines;
                arrays;
                wgps;
                offset;
              }
            in
            let n = g.xccs * instances * engines * arrays * wgps in
            go (offset + (8 * n)) (c :: acc) rest)
  in
  match twice names with
  | Some n -> Error (strf "%s is listed twice" n)
  | None -> go 0 [] names

(* Counting

   The programs are aqlprofile's for GFX12 (src/pm4/pmc_builder.h and
   gfxip/gfx12/gfx12_primitives.h at aqlprofile 8f7db79: Start,
   ReadXccPackets, Stop), register by register. They depart where the interface
   decides: they count the compute waves of the VMID a word in memory selects;
   a stop waits for the dispatches before it, stops counting and samples (Mesa's
   order), stores 0 for a processor that runs no work, and clears the
   filter. *)

exception Missing of string

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

let register g name =
  match Register.find g ("reg" ^ name) with
  | Some r -> r
  | None -> raise (Missing name)

let at g name = Register.address g (register g name)

let set g name fields =
  let r = register g name in
  Pm4.set_reg (Register.address g r) [ Packet.Dword (Register.encode r fields) ]

(* CP_PERFMON_CNTL, which GFX12's headers name CP_PERFMON_CNTL_1. *)
let perfmon g fields = set g "CP_PERFMON_CNTL_1" fields

(* Its PERFMON_STATE, CP_PERFMON_STATE of soc24_enum.h. *)
let disable_and_reset = 0
let start_counting = 1
let stop_counting = 2

(* GRBM_GFX_INDEX: the register writes and reads after it reach every instance
   of every engine and array. *)
let broadcast g =
  set g "GRBM_GFX_INDEX"
    [
      ("instance_broadcast_writes", 1);
      ("se_broadcast_writes", 1);
      ("sa_broadcast_writes", 1);
    ]

(* GRBM_GFX_INDEX for the SQ counters of work-group processor [wgp] of array
   [sa] of engine [se]: the processor is the instance index from bit 2
   (aqlprofile's grbm_se_sh_wgp_index_value). *)
let processor g ~se ~sa ~wgp =
  set g "GRBM_GFX_INDEX"
    [ ("se_index", se); ("sa_index", sa); ("instance_index", wgp lsl 2) ]

(* The select register of SQ counter [c]. An SQ counter is 64 bits, two of the
   32-bit counters the selects number, so counter [n] takes select [2n]. *)
let select g (c : t) =
  set g
    (strf "SQ_PERFCOUNTER%d_SELECT" (2 * c.register))
    [ ("perf_sel", c.event) ]

(* A KFD user queue's COMPUTE_PERFCOUNT_ENABLE does nothing under MES; the
   program writes it for queues without MES. *)
let start_program g ~filter (l : layout) =
  Pm4.event_write Cs_partial_flush
  @ broadcast g
  @ perfmon g [ ("perfmon_state", disable_and_reset) ]
  @ set g "SQ_PERFCOUNTER_CTRL" [ ("cs_en", 1) ]
  @ Pm4.load_reg filter (at g "SQ_PERFCOUNTER_CTRL2")
  @ List.concat_map (select g) l.counters
  @ set g "COMPUTE_PERFCOUNT_ENABLE" [ ("perfcount_enable", 1) ]
  @ perfmon g [ ("perfmon_state", start_counting) ]

(* Whether processor [wgp] of array [sa] of engine [se] runs work. *)
let runs wgps ~se ~sa ~wgp =
  let arrays = wgps.(se) in
  sa < Array.length arrays && arrays.(sa) land (1 lsl wgp) <> 0

(* The words that store SQ counter [c]'s values at the samples' byte offsets
   [address o]: one processor after another, [(engine, array, processor)], the
   last varying fastest. A processor's SQ counter is the 32 bits of
   SQ_PERFCOUNTERn_LO, which saturate, so each value's high half is [zero]. *)
let store g ~wgps ~address ~zero (c : t) =
  let lo = at g (strf "SQ_PERFCOUNTER%d_LO" c.register) in
  let place i =
    let wgp = i mod c.wgps
    and sa = i / c.wgps mod c.arrays
    and se = i / (c.wgps * c.arrays) in
    let word = c.offset + (8 * i) in
    let low =
      if runs wgps ~se ~sa ~wgp then
        processor g ~se ~sa ~wgp
        @ Pm4.copy_data Confirmed (Counter lo) (address word)
      else Pm4.write_data (Memory (address word)) zero
    in
    low @ Pm4.write_data (Memory (address (word + 4))) zero
  in
  List.concat (List.init (c.engines * c.arrays * c.wgps) place)

(* The second flush waits for the GPU to idle after the stop and sample, so
   that the copies read the sampled counters, as aqlprofile waits between them
   (ReadXccPackets). *)
let stop_program g ~wgps (l : layout) ~address ~zero =
  Pm4.event_write Cs_partial_flush
  @ broadcast g
  @ perfmon g [ ("perfmon_state", stop_counting); ("perfmon_sample_enable", 1) ]
  @ Pm4.event_write Cs_partial_flush
  @ List.concat_map (store g ~wgps ~address ~zero) l.counters
  @ broadcast g
  @ set g "SQ_PERFCOUNTER_CTRL2" [ ("force_en", 0); ("vmid_en", 0) ]

(* The programs count on GFX12 GPUs of one die, the only ones hardware has run
   them on, and SQ counters alone, the only ones it has read. *)
let check_gpu fn (g : Gpu.t) =
  match g.gc with
  | 12, _, _ when g.xccs = 1 -> ()
  | _ ->
      invalid_argf "Counter.%s: %s is no GFX12 GPU of one die" fn
        (Gpu.processor g)

let check_sq fn (l : layout) =
  List.iter
    (fun (c : t) ->
      if c.block <> "SQ" then
        invalid_argf "Counter.%s: %s is no SQ counter" fn c.name)
    l.counters

let missing fn (g : Gpu.t) r =
  let a, b, c = g.gc in
  invalid_argf "Counter.%s: GC %d.%d.%d has no reg%s" fn a b c r

let packets fn g program = try program () with Missing r -> missing fn g r
let words fn g program = Packet.encode Int64.of_int (packets fn g program)

let filter g ~vmid =
  check_gpu "filter" g;
  if vmid < 0 || vmid > 15 then
    invalid_argf "Counter.filter: VMID %d, expected 0 to 15" vmid;
  (* SQ_PERFCOUNTER_CTRL2: FORCE_EN counts whether or not the queue enabled
     counting, and VMID_EN has bit [v] for VMID [v]. *)
  try
    Register.encode
      (register g "SQ_PERFCOUNTER_CTRL2")
      [ ("force_en", 1); ("vmid_en", 1 lsl vmid) ]
  with Missing r -> missing "filter" g r

let check_start g l =
  check_gpu "start" g;
  check_sq "start" l

let start g l ~filter_address =
  check_start g l;
  if filter_address < 0 || filter_address mod 4 <> 0 then
    invalid_argf
      "Counter.start: filter address 0x%x, expected a non-negative multiple \
       of 4"
      filter_address;
  words "start" g (fun () -> start_program g ~filter:filter_address l)

let start_packets g l ~filter =
  check_start g l;
  packets "start" g (fun () -> start_program g ~filter l)

let check_stop (g : Gpu.t) ~wgps l =
  check_gpu "stop" g;
  check_sq "stop" l;
  if Array.length wgps <> g.shader_engines then
    invalid_argf "Counter.stop: wgps covers %d engines, expected %d"
      (Array.length wgps) g.shader_engines

let stop g ~wgps l ~address =
  check_stop g ~wgps l;
  if address < 0 || address mod 8 <> 0 then
    invalid_argf
      "Counter.stop: address 0x%x, expected a non-negative multiple of 8"
      address;
  words "stop" g (fun () ->
      stop_program g ~wgps l ~address:(fun o -> address + o) ~zero:0)

let stop_packets g ~wgps l ~at ~zero =
  check_stop g ~wgps l;
  packets "stop" g (fun () -> stop_program g ~wgps l ~address:at ~zero)
