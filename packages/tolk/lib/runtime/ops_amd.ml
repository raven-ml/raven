(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops

type counter = {
  block : string;
  event : int;
  register : int;
  instances : int;
  engines : int;
  arrays : int;
  wgps : int;
  offset : int;
}

type counting = {
  counters : counter list;
  size : int;
  wgp_active : engine:int -> array:int -> wgp:int -> bool;
}

type tracing = { window : int; engines : int }

type profiling = {
  slots : int;
  counting : counting option;
  tracing : tracing option;
}

type gpu = {
  target : int * int * int;
  gc : int * int * int;
  sdma : int * int * int;
  xccs : int;
  shader_engines : int;
  compute_units : int;
  scratch_slots_per_cu : int;
  aql : bool;
  compute_ring : int;
  copy_rings : int list;
  profiling : profiling option;
}

let major gpu =
  let m, _, _ = gpu.target in
  m

module P = Nx_amd_packet

(* A field of a word at its bits [(lo, hi)]. *)
let bits (lo, hi) v =
  (v
  land ((1 lsl ((hi - lo) [@mutate off "every value fits its field"] + 1)) - 1)
  )
  lsl lo

let u32 = Hcq2.Queue.dword
let u64 n = int ~dtype:Dtype.Uint64 n
let binary s = v Op.Binary ~arg:(Bytes s)
let q_of = Hcq2.Queue.q

(* Packets *)

(* A term as tinygrad's call sites compute it: an offset added as a [uint64]
   constant, a right shift by a weak literal. *)
let rec term = function
  | P.Value v -> v
  | Add (t, n) ->
      O.(term t + const ~dtype:Dtype.Uint64 (`Int (Bigint.of_int64_unsigned n)))
  | Shift (t, n) -> O.(term t lsr int n)

(* A packet's words as nodes: a constant word as a [uint32] constant, and a term
   as its node, which the caller made of the width its word takes. *)
let lower ws =
  List.map (function P.Dword n -> u32 n | W32 t | W64 t -> term t) ws

(* Nodes as the words of a packet, each of the width of its type. *)
let words vs =
  List.map
    (fun v ->
      let t = P.Value v in
      if Dtype.itemsize (dtype v) = 8 then P.W64 t else P.W32 t)
    vs

(* A packet's words as nodes, each run of constant words one {!Op.Binary}, as a
   structure in memory lays them out. *)
let blob ws =
  let run cs =
    let b = Bytes.create (4 * List.length cs) in
    List.iteri (fun i n -> Bytes.set_int32_le b (4 * i) (Int32.of_int n)) cs;
    binary (Bytes.to_string b)
  in
  let rec go cs = function
    | P.Dword n :: ws -> go (n :: cs) ws
    | (P.W32 t | W64 t) :: ws ->
        (if cs = [] then [] else [ run (List.rev cs) ]) @ (term t :: go [] ws)
    | [] -> if cs = [] then [] else [ run (List.rev cs) ]
  in
  go [] ws

(* PM4 *)

(* The poll interval of a wait, as tinygrad sets it. *)
let wait_interval = 4

(* The ring and its pointers, tagged [name_queue] like the placeholders the
   engine binds to the device's queue. *)
let queue_args devs queue ring =
  let arg name shape dt =
    placeholder ~slot:0 ~device:(Multi devs) ~volatile:true
      ~tag:(Tag.String (Hcq2.to_name [ name; queue ]))
      shape dt
  in
  ( arg "ring" [ ring / 4 ] Dtype.Uint32,
    arg "write_ptr" [ 1 ] Dtype.Uint64,
    arg "doorbell" [ 1 ] Dtype.Uint64,
    arg "put_value" [ 1 ] Dtype.Uint64 )

(* A device's signal word, which only the batch's own work moves past the value
   its queues wait for. *)
let is_signal_word s =
  match tag (fst (Hcq2.unwrap_view s)) with
  | Some (Tag.String "timeline") -> true
  | _ -> false

(* How a queue waits for a signal: for a device's signal word, its exact value;
   for a queue's signal, a value at least the one waited for. *)
let comparison s : P.comparison =
  if is_signal_word s then Equal else Greater_equal

(* RGP's marker of a pipeline's binding, in the user data of a thread trace
   (RGP's sqtt.h, RGP_SQTT_MARKER_IDENTIFIER_BIND_PIPELINE). *)
let rgp_sqtt_marker_identifier_bind_pipeline = 12

(* Program data *)

type program = {
  desc_offset : int;
  entry_point_offset : int;
  rsrc1 : int;
  rsrc2 : int;
  rsrc3 : int;
  wave : P.Pm4.wave;
  private_segment_size : int;
  group_segment_size : int;
  kernargs_segment_size : int;
  enable_dispatch_ptr : bool;
  enable_private_segment_sgpr : bool;
  image_size : int;
  libhash : int64; (* the first 8 bytes of the code object's MD5 *)
}

(* The kernel of the code object [lib], which holds one: its descriptor, as the
   device's loader relocates it, and the resource words a dispatch writes. *)
let program_data gpu lib =
  let module C = Nx_amd_code_object in
  let kernel co =
    match C.kernels co with
    | [ name ] -> C.kernel co name
    | names ->
        Error
          (Printf.sprintf "an AMD code object of %d kernels" (List.length names))
  in
  let co, k =
    match
      Result.bind (C.of_string lib) (fun co ->
          Result.map (fun k -> (co, k)) (kernel co))
    with
    | Ok r -> r
    | Error e -> invalid_arg e
  in
  let lds = (k.group_segment + 511) / 512 land 0x1ff in
  {
    desc_offset = k.descriptor;
    entry_point_offset = k.entry;
    (* gfx11 runs kernels privileged, for their context save and restore. *)
    rsrc1 = (k.rsrc1 lor if major gpu = 11 then 1 lsl 20 else 0);
    rsrc2 = k.rsrc2 lor (lds lsl 15);
    rsrc3 = k.rsrc3;
    wave = (if k.wave32 then Wave32 else Wave64);
    private_segment_size = k.private_segment;
    group_segment_size = k.group_segment;
    kernargs_segment_size = k.kernarg_size;
    enable_dispatch_ptr = k.dispatch_ptr;
    enable_private_segment_sgpr = k.private_segment_buffer;
    image_size = String.length (C.image co);
    libhash = String.get_int64_le (Digest.string lib) 0;
  }

(* The program's code object, which the engine loads on the devices. *)
let amd_build_program gpu prg devs =
  let obj = Device.Tiny_elf.of_program prg in
  let data = program_data gpu obj.lib in
  ( data,
    placeholder ~slot:0 ~device:(Multi devs)
      ~tag:(Tag.Tuple [ String "program"; Bytes obj.lib; String obj.name ])
      [ data.image_size ] Dtype.Uint8 )

(* As words: the grid may be symbolic. *)
let dispatch_packet data (info : program_info) ?(kernel_object = u64 0)
    ?(kernarg_address = u64 0) () =
  let local =
    List.map
      (function Int l -> l | Sym _ -> invalid_arg "a symbolic local size")
      info.local_size
  in
  let grid =
    List.map2
      (fun g l ->
        match g with
        | Int g -> u32 (g * l)
        | Sym g -> cast O.(g * int l) Dtype.Uint32)
      info.global_size local
  in
  match (local, grid) with
  | [ lx; ly; lz ], [ gx; gy; gz ] ->
      blob
        (P.Aql.dispatch ~threads:(lx, ly, lz) ~grid:(gx, gy, gz)
           ~private_segment:data.private_segment_size
           ~group_segment:data.group_segment_size ~descriptor:kernel_object
           ~args:kernarg_address)
  | _ -> invalid_arg "an AMD dispatch of other than three dimensions"

(* The GC register [name] of the GPU's graphics family: the latest family of its
   major at or before its version, as tinygrad picks a register module. *)
let register gpu name =
  match P.Gc.find gpu.gc ("reg" ^ name) with
  | Some r -> Some (P.Gc.address gpu.gc r, r)
  | None -> None

let register_exn gpu name =
  match register gpu name with
  | Some r -> r
  | None -> invalid_arg (Printf.sprintf "the GPU has no register %s" name)

(* The scratch memory's COMPUTE_TMPRING_SIZE for kernels of [n] bytes per lane:
   the waves it serves and the size of one. *)
let tmpring_size gpu n =
  P.Gc.tmpring_size ~gc:gpu.gc ~compute_units:gpu.compute_units
    ~slots:gpu.scratch_slots_per_cu ~shader_engines:gpu.shader_engines
    ~xccs:gpu.xccs n

(* An AQL queue's packets: dispatches and runs of PM4 packets, and loops around
   them, each with its range and the bytes of its trip in the command buffer. *)
type aql = Packets of Ops.t list | Trips of Ops.t * int * aql list

(* The compute queue, of PM4 packets or of AQL packets around them. *)
let compute_queue ~host gpu q : Hcq2.commands =
  let devs = Hcq2.Queue.devices q in
  let dev = List.hd devs in
  let queue = Hcq2.Queue.name q in
  let gfx9 = major gpu = 9 in
  let getaddr = getaddr ~device:dev in
  let emit ws = ignore (q_of q (lower ws)) in
  let wreg reg vals = emit (P.Pm4.set_reg reg (words vals)) in
  (* The count fills in when the block closes: its packet is encoded again with
     it. *)
  let pred_exec xcc_mask f =
    if gpu.xccs > 1 then emit (P.Pm4.pred_exec ~xcc_mask ~dwords:0);
    let start = Hcq2.Queue.size q in
    f ();
    if gpu.xccs > 1 then
      let dwords = (Hcq2.Queue.size q - start) / 4 in
      List.iteri
        (fun i w -> Hcq2.Queue.set_dword q (start - 8 + (4 * i)) w)
        (P.dwords (P.Pm4.pred_exec ~xcc_mask ~dwords))
  in
  let wait_reg_mem ?(mask = 0xffff_ffff) loc cmp value =
    emit (P.Pm4.wait ~gc:gpu.gc loc cmp value ~mask ~interval:wait_interval)
  in
  let acquire_mem caches = emit (P.Pm4.acquire_mem ~gc:gpu.gc caches) in
  let release_mem address data =
    emit (P.Pm4.release_mem ~gc:gpu.gc address data)
  in
  (* The host flushes the host data path before each submission, after its
     writes through the BAR, so a barrier only invalidates the GPU's caches. A
     flush from the queue, which requests every client's flush and waits until
     all of them are done, hangs a GFX12 compute queue within a few hundred
     batches: its MEC waits on the read of the done register. *)
  let memory_barrier () = acquire_mem All_caches in
  (* Profiling: a run's slot holds its counters and its traces until a
     synchronization reads them back. *)
  let register = register gpu in
  let address name = fst (register_exn gpu name) in
  (* Each value is cut to its field's width, so that none sets the next. *)
  let encode name values = P.Gc.encode (snd (register_exn gpu name)) values in
  let mask name field =
    bits (List.assoc field (snd (register_exn gpu name)).fields) (-1)
  in
  let set name values = wreg (address name) [ u32 (encode name values) ] in
  let set_grbm ?instance ?se ?sa ?wgp () =
    let instance =
      match wgp with
      | Some w -> Some ((w lsl 2) lor Option.value instance ~default:0)
      | None -> instance
    in
    let field key = function
      | None -> (key ^ "_broadcast_writes", 1)
      | Some v -> (key ^ "_index", v)
    in
    set "GRBM_GFX_INDEX"
      [
        field "instance" instance;
        field "se" se;
        field (if gfx9 then "sh" else "sa") sa;
      ]
  in
  let perfmon =
    if major gpu <= 11 then "CP_PERFMON_CNTL" else "CP_PERFMON_CNTL_1"
  in
  let reset_counters ~enable =
    set_grbm ();
    set perfmon [ ("perfmon_state", 0) ];
    if enable then set perfmon [ ("perfmon_state", 1) ]
  in
  let profiled name shape dtype =
    placeholder ~slot:0 ~device:(Multi devs) ~tag:(Tag.String name) [ shape ]
      dtype
  in
  let slots = Option.fold ~none:0 ~some:(fun p -> p.slots) gpu.profiling in
  (* After the count of runs, a slot of the log per run: its kernel's descriptor
     address, then when it started and when it stopped. *)
  let entry = 3 in
  let log = profiled "prof_log" (1 + (entry * slots)) Dtype.Uint64 in
  let samples c = profiled "pmc_buf" (slots * c.size) Dtype.Uint8 in
  let sample_register c =
    Printf.sprintf "%s_PERFCOUNTER%d" c.block c.register
  in
  let start_counting c =
    reset_counters ~enable:false;
    set "SQ_PERFCOUNTER_CTRL"
      ([ ("cs_en", 1); ("ps_en", 1); ("gs_en", 1); ("hs_en", 1) ]
      @ if gfx9 then [ ("vmid_mask", 0xffff) ] else []);
    if not gfx9 then
      set "SQ_PERFCOUNTER_CTRL2" [ ("force_en", 1); ("vmid_en", 0xffff) ];
    List.iter
      (fun ct ->
        (* GFX11 on selects SQ counters with even registers. *)
        let index =
          if (not gfx9) && ct.block = "SQ" then 2 * ct.register else ct.register
        in
        let select = Printf.sprintf "%s_PERFCOUNTER%d_SELECT" ct.block index in
        if register select = None then
          invalid_arg
            (Printf.sprintf "%s is out of counter registers: (%s is not found)"
               ct.block (sample_register ct));
        set select
          (("perf_sel", ct.event)
          ::
          (if gfx9 && ct.block = "SQ" then
             [
               ("simd_mask", 0xf);
               ("sqc_bank_mask", 0xf);
               ("sqc_client_mask", 0xf);
             ]
           else [])))
      c.counters;
    if gfx9 then
      set "SQ_PERFCOUNTER_MASK" [ ("sh0_mask", 0xffff); ("sh1_mask", 0xffff) ];
    set "COMPUTE_PERFCOUNT_ENABLE" [ ("perfcount_enable", 1) ];
    reset_counters ~enable:true
  in
  let read_counters c slot =
    let buf = O.(getaddr (samples c) + (slot * u64 c.size)) in
    set_grbm ();
    set perfmon [ ("perfmon_state", 1); ("perfmon_sample_enable", 1) ];
    List.iter
      (fun ct ->
        let offset = ref ct.offset in
        for xcc = 0 to gpu.xccs - 1 do
          pred_exec (1 lsl xcc) (fun () ->
              for inst = 0 to ct.instances - 1 do
                for se = 0 to ct.engines - 1 do
                  for sa = 0 to ct.arrays - 1 do
                    for wgp = 0 to ct.wgps - 1 do
                      let at = !offset in
                      offset := at + 8;
                      if ct.wgps = 1 || c.wgp_active ~engine:se ~array:sa ~wgp
                      then begin
                        if ct.instances > 1 then set_grbm ~instance:inst ()
                        else if gfx9 then set_grbm ~se ()
                        else set_grbm ~se ~sa ~wgp ();
                        let copy reg at =
                          (* From a performance counter to memory through the
                             L2. *)
                          emit
                            (P.Pm4.copy_data Posted (Counter reg)
                               O.(buf + u64 at))
                        in
                        let name = sample_register ct in
                        Option.iter
                          (fun (lo, _) -> copy lo at)
                          (register (name ^ "_LO"));
                        Option.iter
                          (fun (hi, _) -> copy hi (at + 4))
                          (register (name ^ "_HI"))
                      end
                    done
                  done
                done
              done)
        done)
      c.counters;
    reset_counters ~enable:true
  in
  (* Thread traces, as Mesa's ac_sqtt.c starts, stops and waits for them. Every
     shader engine traces its waves, and engines 0 and 1 their instructions, on
     the first SIMD of their first work-group processor. *)
  let gfx12 = major gpu >= 12 in
  let itraced se = se < 2 in
  let event_write e = emit (P.Pm4.event_write e) in
  let spi_config ~tracing =
    let t = Bool.to_int tracing in
    set "SPI_CONFIG_CNTL"
      [
        ("ps_pkr_priority_cntl", 3);
        ("exp_priority_order", 3);
        ("gpr_write_priority", 0x2c688);
        ("enable_sqg_bop_events", t);
        ("enable_sqg_top_events", t);
      ]
  in
  let trace_config ~tracing =
    set "SQ_THREAD_TRACE_CTRL"
      ([
         ("draw_event_en", 1);
         ("spi_stall_en", 1);
         ("sq_stall_en", 1);
         ("reg_at_hwm", 2);
         ("hiwater", 1);
         ("util_timer", 1);
         ("mode", Bool.to_int tracing);
       ]
      @
      if gfx12 then [] else [ ("rt_freq", P.Gc.Thread_trace.rt_freq_4096_clk) ]
      )
  in
  (* Words for the trace, in pairs of user data registers. *)
  let userdata words =
    let rec go = function
      | [] -> ()
      | [ w ] -> wreg (address "SQ_THREAD_TRACE_USERDATA_2") [ w ]
      | a :: b :: rest ->
          wreg (address "SQ_THREAD_TRACE_USERDATA_2") [ a; b ];
          go rest
    in
    go words
  in
  let commands = ref 0 in
  (* The RGP markers of a dispatch: the pipeline its program binds, and the
     dispatch with its grid. *)
  let trace_markers (data : program) (info : program_info) =
    let hash = data.libhash in
    userdata
      [
        u32 (rgp_sqtt_marker_identifier_bind_pipeline lor (1 lsl 7));
        u32 (Int64.to_int hash);
        u32 (Int64.to_int (Int64.shift_right_logical hash 32));
      ];
    userdata
      ([ u32 (1 lsl 31); u32 0; u32 !commands ]
      @ List.map (function Int g -> u32 g | Sym g -> g) info.global_size);
    incr commands
  in
  let traces t =
    profiled "sqtt_buf" (t.window * slots * t.engines) Dtype.Uint8
  in
  let start_trace t slot =
    memory_barrier ();
    let base = O.(getaddr (traces t) + (slot * u64 t.window)) in
    let buffer se shift =
      let window = u64 (se * slots * t.window) in
      cast O.((base + window) lsr u64 shift) Dtype.Uint32
    in
    let engines = gpu.shader_engines in
    if gfx9 then begin
      set_grbm ();
      set "SQ_THREAD_TRACE_MASK"
        [
          ("simd_en", 0xf);
          ("cu_sel", 0);
          ("sq_stall_en", 1);
          ("spi_stall_en", 1);
          ("reg_stall_en", 1);
          ("vm_id_mask", 0);
        ];
      for se = 0 to t.engines - 1 do
        (* Misc, time, registers, wave starts and ends, user data and compute
           registers; and the instructions of the engines that trace them. *)
        let tokens =
          List.fold_left
            (fun m b -> m lor (1 lsl b))
            0
            [ 0; 1; 2; 3; 6; 12; 5; 15 ]
          lor if itraced se then (1 lsl 10) lor (1 lsl 11) lor (1 lsl 13) else 0
        in
        pred_exec
          (1 lsl (se / engines))
          (fun () ->
            set_grbm ~se:(se mod engines) ~sa:0 ();
            set "SQ_THREAD_TRACE_TOKEN_MASK"
              [ ("reg_mask", 0xf); ("token_mask", tokens) ];
            set "SQ_THREAD_TRACE_TOKEN_MASK2" [ ("inst_mask", 0xffff_ffff) ];
            wreg (address "SQ_THREAD_TRACE_BASE") [ buffer se 12 ];
            wreg (address "SQ_THREAD_TRACE_BASE2") [ buffer se 44 ];
            set "SQ_THREAD_TRACE_SIZE" [ ("size", t.window lsr 12) ];
            set "SQ_THREAD_TRACE_CTRL" [ ("reset_buffer", 1) ];
            set "SQ_THREAD_TRACE_MODE"
              [ ("mask_cs", 1); ("autoflush_en", 1); ("mode", 1) ])
      done
    end
    else begin
      spi_config ~tracing:true;
      for se = 0 to t.engines - 1 do
        set_grbm ~se ~sa:0 ();
        if gfx12 then begin
          set "SQ_THREAD_TRACE_BUF0_SIZE" [ ("size", t.window lsr 12) ];
          wreg (address "SQ_THREAD_TRACE_BUF0_BASE_LO") [ buffer se 12 ];
          wreg (address "SQ_THREAD_TRACE_BUF0_BASE_HI") [ buffer se 44 ]
        end
        else begin
          let size =
            u32
              (encode "SQ_THREAD_TRACE_BUF0_SIZE" [ ("size", t.window lsr 12) ])
          in
          wreg
            (address "SQ_THREAD_TRACE_BUF0_SIZE")
            [ O.(size lor buffer se 44) ];
          wreg (address "SQ_THREAD_TRACE_BUF0_BASE") [ buffer se 12 ]
        end;
        set "SQ_THREAD_TRACE_MASK"
          [
            ( "wtype_include",
              if gfx12 then 1 lsl 6 else P.Gc.Thread_trace.wtype_include_cs_bit
            );
            ("simd_sel", 0);
            ("wgp_sel", 0);
            ("sa_sel", 0);
          ];
        let registers =
          P.Gc.Thread_trace.(
            token_mask_sqdec_bit lor token_mask_shdec_bit
            lor token_mask_gfxudec_bit lor token_mask_comp_bit
            lor token_mask_context_bit)
        in
        let excluded =
          if itraced se then 0
          else if gfx12 then 0x927
          else
            P.Gc.Thread_trace.(
              (1 lsl token_exclude_vmemexec_shift)
              lor (1 lsl token_exclude_aluexec_shift)
              lor (1 lsl token_exclude_valuinst_shift)
              lor (1 lsl token_exclude_immediate_shift)
              lor (1 lsl token_exclude_inst_shift))
        in
        (* A GFX11 trace includes its exec tokens (TTRACE_EXEC), the bit just
           past its 11 token exclusions. *)
        set "SQ_THREAD_TRACE_TOKEN_MASK"
          ([
             ("reg_include", registers);
             ("token_exclude", excluded);
             ("bop_events_token_include", 1);
           ]
          @
          if gfx12 then [ ("exclude_barrier_wait", 1) ]
          else [ ("ttrace_exec", 1) ]);
        trace_config ~tracing:true
      done
    end;
    set_grbm ();
    if not gfx9 then wreg (address "COMPUTE_THREAD_TRACE_ENABLE") [ u32 1 ];
    memory_barrier ()
  in
  let stop_trace t slot =
    memory_barrier ();
    set_grbm ();
    let ends = profiled "sqtt_wptrs" (slots * t.engines) Dtype.Uint32 in
    let run_bytes = u64 (t.engines * 4) in
    let run_ends = O.(getaddr ends + (slot * run_bytes)) in
    if gfx9 then
      set "SQ_THREAD_TRACE_MODE"
        [ ("mask_cs", 1); ("autoflush_en", 1); ("mode", 0) ]
    else begin
      wreg (address "COMPUTE_THREAD_TRACE_ENABLE") [ u32 0 ];
      event_write Thread_trace_finish
    end;
    let status = address "SQ_THREAD_TRACE_STATUS"
    and engines = gpu.shader_engines in
    for se = 0 to t.engines - 1 do
      pred_exec
        (1 lsl (se / engines))
        (fun () ->
          set_grbm ~se:(se mod engines) ~sa:0 ();
          let idle field =
            wait_reg_mem
              ~mask:(mask "SQ_THREAD_TRACE_STATUS" field)
              (Register status) Equal (u32 0)
          in
          if not gfx9 then begin
            idle "finish_pending";
            trace_config ~tracing:false
          end;
          idle "busy";
          event_write Cs_partial_flush;
          let engine_end = u64 (se * 4) in
          (* Where the engine's trace ends, to memory with its write
             confirmed. *)
          emit
            (P.Pm4.copy_data Confirmed
               (Counter (address "SQ_THREAD_TRACE_WPTR"))
               O.(run_ends + engine_end)))
    done;
    set_grbm ();
    if not gfx9 then spi_config ~tracing:false;
    memory_barrier ()
  in
  Option.iter (fun p -> Option.iter start_counting p.counting) gpu.profiling;
  let runs = ref [] in
  (* The [i]th word of [slot]'s entry. *)
  let word slot i =
    let base = O.(u64 1 + (u64 entry * slot) + u64 i) in
    O.(getaddr log + (base * u64 8))
  in
  (* The GPU's clock, written to [address] as the queue reaches the packet: the
     work before it is complete, as each dispatch waits for its waves, and the
     work after it has not started. An end-of-pipe write would wait for the pipe
     to drain, which on GFX12 can take in the dispatch queued behind it and
     stamp the start of a long kernel at its end. *)
  let clock_into address =
    pred_exec 1 (fun () -> emit (P.Pm4.copy_data Confirmed Clock address))
  in
  (* A profiled run takes the next slot of the log, which the host program
     writes. *)
  let start_run lib (data : program) info =
    Option.map
      (fun p ->
        let slot =
          O.(
            (load (index log [ int 0 ]) [] + u64 (List.length !runs))
            % u64 p.slots)
        in
        let at = O.(int 1 + (int entry * cast slot Dtype.Int32)) in
        runs :=
          !runs
          @ [ store (index log [ at ]) O.(getaddr lib + u64 data.desc_offset) ];
        clock_into (word slot 1);
        Option.iter
          (fun t ->
            start_trace t slot;
            trace_markers data info)
          p.tracing;
        (p, slot))
      gpu.profiling
  in
  let traced =
    Option.is_some (Option.bind gpu.profiling (fun p -> p.tracing))
  in
  let stop_run =
    Option.iter (fun (p, slot) ->
        clock_into (word slot 2);
        Option.iter (fun c -> read_counters c slot) p.counting;
        Option.iter (fun t -> stop_trace t slot) p.tracing)
  in
  (* Once its command buffer is written, the host program adds the submission's
     runs to the log's count. *)
  let count_runs cmdbuf =
    match !runs with
    | [] -> cmdbuf
    | rs ->
        after cmdbuf
          [
            store
              (index (after log (cmdbuf :: rs)) [ int 0 ])
              O.(load (index log [ int 0 ]) [] + u64 (List.length rs));
          ]
  in
  let kernargs call prg data =
    let info =
      match arg prg with
      | Program p -> p
      | _ -> invalid_arg "an AMD command runs a compiled program"
    in
    let bufs = Realize.get_call_arg_uops call in
    (* A bound value is a bare constant: the variable has the width. *)
    let args =
      List.map (fun g -> getaddr (List.nth bufs g)) info.globals
      @ List.map2
          (fun v b -> ccast b (dtype v))
          info.vars
          (Realize.get_call_var_uops call prg)
    in
    let words =
      Hcq2.pack_args (Hcq2.layout_args args) data.kernargs_segment_size
    in
    let packet =
      if data.enable_dispatch_ptr then dispatch_packet data info () else []
    in
    (* The arguments start on 128 bytes, as every buffer a queue's commands
       address does; a kernel's argument segment needs 16. *)
    ( info,
      v Op.Linear ~src:(words @ packet)
        ~arg:(Region { name = "kernargs"; align = 128 }) )
  in
  let wait signal value =
    wait_reg_mem
      (Memory (getaddr signal))
      (comparison signal) (cast value Dtype.Uint32)
  in
  let timestamp signal = clock_into O.(getaddr signal + u64 8) in
  (* A device's value is written whole, in one 64-bit write. *)
  let signal_mem signal value =
    let value = cast value Dtype.Uint64 in
    let data : _ P.Pm4.data =
      if is_signal_word signal then Data_64 value else Low_32 value
    in
    pred_exec 1 (fun () -> release_mem (getaddr signal) data)
  in
  let ring_queue = (queue, gpu.compute_ring) in
  let push cmdbuf words ?(unit = 4) ?(doorbell_lag = 0) () =
    let ring, wptr, doorbell, put =
      queue_args devs (fst ring_queue) (snd ring_queue)
    in
    (* put counts units, the ring dwords. *)
    let rs = snd ring_queue / 4 and n = max_numel words / 4 in
    let p = load (index put [ int 0 ]) [] in
    let per_unit = unit / 4 and units = max_numel words / unit in
    let tail = cast O.(p * int per_unit % int rs) Dtype.Int32 in
    let first = minimum O.(int rs - tail) (int n) in
    let copy cmdbuf rid dst src count =
      let i = range ~dtype:Dtype.Int32 ~src:[ cmdbuf ] (Sym count) [ rid ] in
      end_
        (store
           (index (after ring [ cmdbuf ]) [ O.(dst + i) ])
           (load (index (bitcast words Dtype.Uint32) [ O.(src + i) ]) []))
        [ i ]
    in
    let cmdbuf = copy cmdbuf 10 tail (int 0) first in
    let cmdbuf = copy cmdbuf 11 (int 0) first O.(int n - first) in
    let nxt = O.(p + int units) in
    let w = store (index (after wptr [ cmdbuf ]) [ int 0 ]) nxt in
    store
      (index
         (after doorbell [ store (index (after put [ w ]) [ int 0 ]) nxt ])
         [ int 0 ])
      O.(nxt - int doorbell_lag)
  in
  let program call prg =
    let data, lib = amd_build_program gpu prg devs in
    let info, ka = kernargs call prg data in
    (data, lib, info, ka)
  in
  (* Its address is patched in at submit. *)
  let ib_blob cmdbuf =
    let ws =
      P.dwords (P.Pm4.indirect_buffer 0 ~dwords:(max_numel cmdbuf / 4))
    in
    let b = Bytes.create (4 * List.length ws) in
    List.iteri (fun i w -> Bytes.set_int32_le b (4 * i) (Int32.of_int w)) ws;
    Bytes.to_string b
  in
  if not gpu.aql then
    let exec call prg =
      let data, lib, info, ka = program call prg in
      let prog_addr = O.(getaddr lib + int data.entry_point_offset) in
      let scratch_addr =
        getaddr
          (rtag ~tag:(Tag.String "scratch")
             (placeholder ~slot:0 ~device:(Multi devs)
                [ data.private_segment_size ]
                Dtype.Uint8))
      in
      let args_addr = getaddr ka in
      let user_regs =
        (if data.enable_private_segment_sgpr then
           [
             O.(scratch_addr lor const (`Int (Bigint.shift_left Bigint.one 63)));
             u32 0xffff_ffff;
             u32 0x20c14000;
           ]
         else [])
        @ (if data.enable_dispatch_ptr then
             [ O.(args_addr + int data.kernargs_segment_size) ]
           else [])
        @ [ args_addr ]
      in
      let groups =
        match
          List.map (function Int g -> u32 g | Sym g -> g) info.global_size
        with
        | [ x; y; z ] -> (x, y, z)
        | _ -> invalid_arg "an AMD dispatch of other than three dimensions"
      in
      let set name = wreg (address ("COMPUTE_" ^ name)) in
      acquire_mem Data_caches;
      let run = start_run lib data info in
      emit (P.Pm4.set_program ~gc:gpu.gc prog_addr);
      set "PGM_RSRC1" [ u32 data.rsrc1; u32 data.rsrc2 ];
      set "PGM_RSRC3" [ u32 data.rsrc3 ];
      set "TMPRING_SIZE" [ u32 (tmpring_size gpu data.private_segment_size) ];
      (* Architected flat scratch: each die gets its part. *)
      for xcc = 0 to gpu.xccs - 1 do
        let part = data.private_segment_size / gpu.xccs * xcc in
        pred_exec (1 lsl xcc) (fun () ->
            emit (P.Pm4.set_scratch ~gc:gpu.gc O.(scratch_addr + int part)))
      done;
      set "RESTART_X" [ u32 0; u32 0; u32 0 ];
      set "USER_DATA_0" user_regs;
      set "RESOURCE_LIMITS" [ u32 (Setting.value Setting.waves_per_sh) ];
      set "START_X"
        ([ u32 0; u32 0; u32 0 ]
        @ List.map (function Int l -> u32 l | Sym l -> l) info.local_size
        @ [ u32 0; u32 0 ]);
      emit (P.Pm4.dispatch_direct ~gc:gpu.gc data.wave groups);
      if traced then event_write Thread_trace_marker;
      event_write Cs_partial_flush;
      stop_run run
    in
    (* The ring gets an indirect buffer packet: 4 dwords, and put stays aligned
       so that it never wraps mid packet. *)
    let submit () =
      let cmdbuf = Hcq2.bufferize_cmdbuf q "cmdbuf" in
      let base, off = Hcq2.unwrap_view cmdbuf in
      let ib =
        placeholder ~device:(Single host)
          ~tag:(Tag.String (Hcq2.to_name [ "ib"; queue ]))
          [ 16 ] Dtype.Uint8
      in
      push (count_runs cmdbuf)
        (Hcq2.patch ib
           [ (int 4, O.(getaddr base + int off)) ]
           ~blob:(ib_blob cmdbuf))
        ()
    in
    {
      Hcq2.exec;
      copy = (fun _ _ _ -> invalid_arg "an AMD compute queue does not copy");
      wait;
      signal = signal_mem;
      timestamp;
      memory_barrier;
      loop = Hcq2.Queue.loop q;
      submit;
    }
  else
    (* The ring holds 64-byte AQL packets: a dispatch for each kernel, and the
       PM4 packets between them wrapped as indirect buffers. The packets point
       into the command buffer, whose address binds at submit. *)
    let cmd_addr =
      variable ~dtype:Dtype.Uint64 "cmdbuf" (`Int Bigint.zero)
        (`Int (Bigint.shift_left Bigint.one 48))
    in
    let items = ref [] and run_start = ref 0 in
    let add ws = items := !items @ [ Packets ws ] in
    let close_run end_ =
      if end_ > !run_start then
        add
          (lower
             (P.Aql.indirect_buffer
                O.(cmd_addr + int !run_start)
                ~dwords:((end_ - !run_start) / 4)));
      run_start := end_
    in
    (* Close the PM4 run, so that the signal waits for the dispatches before it:
       the dies could otherwise race. *)
    let signal s value =
      close_run (Hcq2.Queue.size q);
      signal_mem s value
    in
    let exec call prg =
      let data, lib, info, ka = program call prg in
      let run = start_run lib data info in
      close_run (Hcq2.Queue.size q);
      add
        (dispatch_packet data info
           ~kernel_object:O.(getaddr lib + int data.desc_offset)
           ~kernarg_address:(getaddr ka) ());
      run_start := Hcq2.Queue.size q;
      stop_run run
    in
    (* A loop's packets are once per trip, each pointing into its trip's bytes
       of the command buffer, which the queue repeats. *)
    let loop r body =
      close_run (Hcq2.Queue.size q);
      let outer = !items and start = Hcq2.Queue.size q and trip = ref 0 in
      items := [];
      Hcq2.Queue.loop q r (fun () ->
          body ();
          close_run (Hcq2.Queue.size q);
          trip := Hcq2.Queue.size q - start);
      items := outer @ [ Trips (r, !trip, !items) ];
      run_start := Hcq2.Queue.size q
    in
    (* The doorbell is the last packet's index. *)
    let submit () =
      let cmdbuf = Hcq2.bufferize_cmdbuf q "cmdbuf" in
      close_run (max_numel cmdbuf);
      let base, off = Hcq2.unwrap_view cmdbuf in
      Hcq2.Queue.reset q;
      let rec emit addr = function
        | Packets ws ->
            ignore (q_of q (src (substitute (sink ws) [ (cmd_addr, addr) ])))
        | Trips (r, trip, items) ->
            Hcq2.Queue.loop q r (fun () ->
                List.iter
                  (emit O.(addr + (cast r Dtype.Uint64 * int trip)))
                  items)
      in
      List.iter (emit O.(getaddr base + int off)) !items;
      let size = Hcq2.Queue.size q in
      (* A submission writes at most half the ring, which the device leaves room
         for. *)
      if 2 * size > gpu.compute_ring then
        raise
          (Hcq2.Over_capacity
             (Printf.sprintf
                "AQL packets of %d bytes exceed half their ring of %d bytes"
                size gpu.compute_ring));
      push (count_runs cmdbuf)
        (Hcq2.bufferize_cmdbuf ~device:(Single host) q "aql")
        ~unit:64 ~doorbell_lag:1 ()
    in
    {
      Hcq2.exec;
      copy = (fun _ _ _ -> invalid_arg "an AMD compute queue does not copy");
      wait;
      signal;
      timestamp;
      memory_barrier;
      loop;
      submit;
    }

(* SDMA *)

let copy_queue ~host gpu q : Hcq2.commands =
  let devs = Hcq2.Queue.devices q in
  let getaddr = getaddr ~device:(List.hd devs) in
  let queue = Hcq2.Queue.name q in
  let ring =
    match String.split_on_char ':' queue with
    | [ _; i ] -> (
        match List.nth_opt gpu.copy_rings (int_of_string i) with
        | Some ring -> ring
        | None ->
            invalid_arg
              (Printf.sprintf "the AMD GPU has no copy queue %s" queue))
    | _ -> invalid_arg (Printf.sprintf "%s is no AMD copy queue" queue)
  in
  let loop = Hcq2.Queue.loop q in
  let cmdbuf () = Hcq2.bufferize_cmdbuf q "cmdbuf" in
  let emit ws = ignore (q_of q (lower ws)) in
  let copy dst src sz =
    emit (P.Sdma.copy ~sdma:gpu.sdma ~dst:(getaddr dst) ~src:(getaddr src) sz)
  in
  let wait signal value =
    emit
      (P.Sdma.poll (getaddr signal) (comparison signal)
         (cast value Dtype.Uint32)
         ~mask:((1 lsl (8 * min (Dtype.itemsize (dtype value)) 4)) - 1))
  in
  let timestamp signal = emit (P.Sdma.timestamp O.(getaddr signal + u64 8)) in
  (* A device's value is written 32 bits at a time: its high half only when its
     low half is 0, the high half every later value shares; four NOPs otherwise
     . *)
  let signal signal value =
    let fence addr v = lower (P.Sdma.fence ~sdma:gpu.sdma addr v) in
    let high =
      if not (is_signal_word signal) then []
      else
        let carry = eq (cast value Dtype.Uint32) (u32 0) in
        List.map
          (fun w -> where carry w (const_like w (`Int Bigint.zero)))
          (fence
             O.(getaddr signal + u64 4)
             (cast O.(value lsr u64 32) Dtype.Uint32))
    in
    ignore
      (q_of q
         (fence (getaddr signal) (cast value Dtype.Uint32)
         @ high @ lower P.Sdma.trap))
  in
  (* SDMA needs the command buffer whole in the ring: if it does not fit before
     the ring's end, it restarts at 0 and zeroes the tail. *)
  let submit () =
    let cmdbuf = cmdbuf () in
    let ring_p, wptr, doorbell, put = queue_args devs queue ring in
    (* In host memory: streamed into the ring, the device never reads it. *)
    let base = fst (Hcq2.unwrap_view cmdbuf) in
    let on_host =
      match arg base with
      | Param p ->
          replace ~arg:(Param { p with device = Some (Single host) }) base
      | _ -> invalid_arg "an AMD command buffer is a placeholder"
    in
    let cmdbuf = substitute cmdbuf [ (base, on_host) ] in
    let rs = ring / 4 and size_dw = max_numel cmdbuf / 4 in
    (* Zeroing the tail can double what a submission writes, which is at most
       half the ring. *)
    if 4 * (size_dw * 4) > ring then
      raise
        (Hcq2.Over_capacity
           (Printf.sprintf
              "an SDMA command buffer of %d bytes exceeds a quarter of its \
               ring of %d bytes"
              (size_dw * 4) ring));
    let put_b = load (index put [ int 0 ]) [] in
    let ring_bytes = rs * 4 in
    let tail = cast O.(put_b % int ring_bytes // int 4) Dtype.Int32 in
    let fits = cast O.(int rs - tail >= int size_dw) Dtype.Int32 in
    let start_dw = O.(fits * tail)
    and zero_amt = O.((int 1 - fits) * (int rs - tail)) in
    let zi = range ~dtype:Dtype.Int32 ~src:[ cmdbuf ] (Sym zero_amt) [ 10 ] in
    let zero_tail =
      end_ (store (index ring_p [ O.(tail + zi) ]) (u32 0)) [ zi ]
    in
    let i = range ~dtype:Dtype.Int32 ~src:[ cmdbuf ] (Int size_dw) [ 11 ] in
    let copy =
      end_
        (store
           (index (after ring_p [ zero_tail ]) [ O.(start_dw + i) ])
           (load (index (bitcast cmdbuf Dtype.Uint32) [ i ]) []))
        [ i ]
    in
    let next_put =
      O.(put_b + cast ((zero_amt + int size_dw) * int 4) (dtype put_b))
    in
    let w = store (index (after wptr [ copy ]) [ int 0 ]) next_put in
    store
      (index
         (after doorbell [ store (index (after put [ w ]) [ int 0 ]) next_put ])
         [ int 0 ])
      next_put
  in
  {
    Hcq2.exec = (fun _ _ -> invalid_arg "an AMD copy queue runs no program");
    copy;
    wait;
    signal;
    timestamp;
    (* A copy queue has nothing to flush. *)
    memory_barrier = (fun () -> ());
    loop;
    submit;
  }

let queues ~host ~reaches gpu =
  (match gpu.target with
  | 9, 4, 2 | 9, 5, 0 -> ()
  | m, _, _ when m = 11 || m = 12 -> ()
  | m, n, s ->
      invalid_arg (Printf.sprintf "gfx%d%x%x is not a supported AMD GPU" m n s));
  if gpu.xccs > 1 && not gpu.aql then
    invalid_arg "an AMD GPU of several dies takes AQL packets";
  let commands q =
    if String.starts_with ~prefix:"COMPUTE" (Hcq2.Queue.name q) then
      compute_queue ~host gpu q
    else copy_queue ~host gpu q
  in
  {
    Hcq2.commands;
    copy_queue = gpu.copy_rings <> [];
    submission = Buffered;
    host;
    reaches;
  }

(* What the engine links *)

type storage =
  | Ring of string
  | Write_ptr of string
  | Put of string
  | Doorbell of string
  | Program of { binary : string; name : string }
  | Scratch of int
  | Log
  | Samples
  | Traces
  | Trace_ends

let storage u =
  match tag u with
  | Some (Tag.Tuple [ String "program"; Bytes binary; String name ]) ->
      Some (Program { binary; name })
  | Some (Tag.String "scratch") -> Some (Scratch (max_numel u))
  | Some (Tag.String "prof_log") -> Some Log
  | Some (Tag.String "pmc_buf") -> Some Samples
  | Some (Tag.String "sqtt_buf") -> Some Traces
  | Some (Tag.String "sqtt_wptrs") -> Some Trace_ends
  | Some (Tag.String t) -> (
      (* [name_queue_index], as queue_args tags them. *)
      match List.rev (String.split_on_char '_' t) with
      | i :: queue :: name -> (
          let queue = String.uppercase_ascii queue ^ ":" ^ i in
          match String.concat "_" (List.rev name) with
          | "ring" -> Some (Ring queue)
          | "write_ptr" -> Some (Write_ptr queue)
          | "put_value" -> Some (Put queue)
          | "doorbell" -> Some (Doorbell queue)
          | _ -> None)
      | _ -> None)
  | _ -> None
