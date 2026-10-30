(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Thread traces: the program each generation writes, the write pointer's units,
   and the decoder against a captured gfx1201 trace and GFX9, GFX11 and GFX12
   traces written by hand. *)

open Windtrap
open Rig_amd_abi
module S = Rig_amd_abi_support

let strf = Printf.sprintf
let timeout = S.timeout
let gpu = S.gpu

(* Recording *)

(* Where a program uses each engine's address: the dies its predication runs it
   on, if any, and the engine GRBM_GFX_INDEX selects. *)
type use = { engine : int; dies : int option; selected : int option }

let pp_use ppf u =
  let opt = Option.fold ~none:"-" ~some:string_of_int in
  Format.fprintf ppf "{ engine %d; dies %s; selected %s }" u.engine (opt u.dies)
    (opt u.selected)

(* The programs' addresses: engine [e]'s buffer of [size] bytes lies at
   [buffers + e * size], and its end is the word at [ends + 4 * e]. *)
let buffers = 0x10_0000
let ends = 0x20_0000
let page = 4096

(* PACKET3_COPY_DATA's opcode, whose body is its control, its source's two words
   and its destination's (soc15d.h, nvd.h). *)
let copy_data = 0x40

(* The register a start writes each engine's page to: the buffer's address
   from bit 12, its low 32 bits. *)
let page_register (g : Gpu.t) =
  let name =
    match g.gc with
    | 9, _, _ -> "regSQ_THREAD_TRACE_BASE"
    | 11, _, _ -> "regSQ_THREAD_TRACE_BUF0_BASE"
    | _ -> "regSQ_THREAD_TRACE_BUF0_BASE_LO"
  in
  Register.address g (require_some (Register.find g name))

(* Walks [p], the words of a program, and gives each use of an engine's
   address: each write of its buffer's page, and each copy to its end. *)
let uses g ~size p =
  let grbm = require_some (Register.find g "regGRBM_GFX_INDEX") in
  let grbm_at = Register.address g grbm and page_at = page_register g in
  let field w f =
    let lo, hi = List.assoc f grbm.fields in
    (w lsr lo) land ((1 lsl (hi - lo + 1)) - 1)
  in
  let selected = ref None and dies = ref None and predicated = ref 0 in
  let out = ref [] in
  let on (op, body) =
    let dies_now = if !predicated > 0 then !dies else None in
    if !predicated > 0 then
      predicated := !predicated - (List.length body + 1);
    let use engine =
      out := { engine; dies = dies_now; selected = !selected } :: !out
    in
    let write at v =
      if at = grbm_at then
        selected :=
          if field v "se_broadcast_writes" = 1 then None
          else Some (field v "se_index")
      else if at = page_at then
        let off = (v * page) - buffers in
        if off < 0 || off mod size <> 0 then
          failf "page 0x%x starts no engine's buffer" v
        else use (off / size)
    in
    match body with
    | [ w ] when op = S.pred_exec ->
        dies := Some (w lsr 24);
        predicated := w land 0x3fff
    | off :: vs when op = S.set_uconfig_reg ->
        List.iteri (fun i v -> write (S.uconfig_start + off + i) v) vs
    | off :: vs when op = S.set_sh_reg ->
        List.iteri (fun i v -> write (S.sh_start + off + i) v) vs
    | [ _; _; _; lo; hi ] when op = copy_data ->
        use (((lo lor (hi lsl 32)) - ends) / 4)
    | _ -> ()
  in
  List.iter on (S.packets (S.words p));
  List.rev !out

let engines (g : Gpu.t) = List.init (g.shader_engines * g.xccs) Fun.id
let start g ~size =
  Thread_trace.start g ~wgps:(S.harvest_none g) ~size ~address:buffers

let stop g = Thread_trace.stop g ~wgps:(S.harvest_none g) ~address:ends

(* The uses an engine's number must have: its die's predication on a GPU of
   several dies, and its engine selected. *)
let expected (g : Gpu.t) e =
  {
    engine = e;
    dies = (if g.xccs > 1 then Some (1 lsl (e / g.shader_engines)) else None);
    selected = Some (e mod g.shader_engines);
  }

let traced =
  Gen.with_pp
    (fun ppf (g : Gpu.t) ->
      Format.fprintf ppf "GC %s, %d dies of %d engines" (S.version g.gc) g.xccs
        g.shader_engines)
    (let open Gen in
     let+ gc, xccs =
       of_list
         [
           ((9, 4, 3), 1);
           ((9, 4, 3), 2);
           ((9, 4, 3), 8);
           ((11, 0, 0), 1);
           ((11, 5, 0), 1);
           ((12, 0, 0), 1);
         ]
     and+ shader_engines = int_range 1 8 in
     gpu ~xccs ~shader_engines gc)

(* Buffer sizes that tell an engine's offset [e * size] from [e * 4096]. *)
let sizes = [ page; 3 * page; 1 lsl 20 ]

(* Each engine that runs work uses its own address, as [expected] states. An
   engine whose arrays all run no work, harvested, is neither programmed nor
   awaited. An engine of one array off and one on runs work. *)
let engine_law name program =
  let gen =
    Gen.with_pp
      (fun ppf ((g : Gpu.t), size, off) ->
        Format.fprintf ppf "GC %s, %d dies of %d engines, %d bytes each, %s off"
          (S.version g.gc) g.xccs g.shader_engines size
          (String.concat " " (List.map string_of_int off)))
      (let open Gen in
       let* g = traced in
       let n = g.shader_engines * g.xccs in
       let some = list ~size:(int_range 1 n) (int_range 0 (n - 1)) in
       let+ size = of_list sizes
       and+ off =
         frequency
           [
             (1, constant []);
             (1, constant (List.init n Fun.id));
             (2, map (List.sort_uniq compare) some);
           ]
       in
       (g, size, off))
  in
  prop ~timeout name gen (fun (g, size, off) ->
      let n = g.shader_engines * g.xccs in
      cover "a buffer of more than a page" (size > page);
      cover "no engine off" (off = []);
      cover "every engine off" (List.length off = n);
      let wgps =
        Array.init n (fun e ->
            if List.mem e off then [| 0; 0 |] else [| 0; 1 |])
      in
      let us = uses g ~size (program g ~wgps ~size) in
      let check e =
        let mine = List.filter (fun u -> u.engine = e) us in
        if List.mem e off then
          equal int ~msg:(strf "uses of harvested engine %d" e) 0
            (List.length mine)
        else if mine = [] then failf "engine %d is unused" e
        else
          List.iter
            (fun u ->
              equal ~msg:(strf "engine %d" e)
                (Testable.make ~pp:pp_use ~equal:( = ))
                (expected g e) u)
            mine
      in
      List.iter check (engines g);
      List.iter (fun u -> less int ~than:n u.engine) us)

(* The GC versions whose trace programs differ in their size register: GFX9's
   SQ_THREAD_TRACE_SIZE, GFX11's BUF0_SIZE beside the address's top bits,
   GFX12's BUF0_SIZE alone. Each field is 22 bits of 4096-byte pages. *)
let trace_families = [ (9, 4, 3); (11, 0, 0); (12, 0, 0) ]

(* The largest multiple of 4096 an int holds. *)
let max_pages = max_int land lnot 4095

(* The pages the size register of [g]'s start program holds, for [size]. *)
let size_field (g : Gpu.t) size =
  let name =
    match g.gc with
    | 9, _, _ -> "regSQ_THREAD_TRACE_SIZE"
    | _ -> "regSQ_THREAD_TRACE_BUF0_SIZE"
  in
  let r = Option.get (Register.find g name) in
  let lo, hi = List.assoc "size" r.fields in
  let ws = S.words (start g ~size) in
  let v = List.assoc (Register.address g r) (S.writes ws) in
  (v lsr lo) land ((1 lsl (hi - lo + 1)) - 1)

(* What each engine traces, in the registers that choose it: the unit
   (SQ_THREAD_TRACE_MASK) and the tokens (SQ_THREAD_TRACE_TOKEN_MASK), the third
   engine tracing no instructions. A baseline. *)
let selection () =
  let lines gc =
    let g = gpu ~shader_engines:3 gc in
    let at name = Register.address g (Option.get (Register.find g name)) in
    let mask = at "regSQ_THREAD_TRACE_MASK"
    and tokens = at "regSQ_THREAD_TRACE_TOKEN_MASK" in
    let rec engines e = function
      | (a, m) :: (b, t) :: rest when a = mask && b = tokens ->
          strf "GFX %s engine %d: MASK 0x%x TOKEN_MASK 0x%x" (S.version gc) e m
            t
          :: engines (e + 1) rest
      | _ :: rest -> engines e rest
      | [] -> []
    in
    engines 0 (S.writes (S.words (start g ~size:4096)))
  in
  String.concat "\n" (List.concat_map lines trace_families)

let recording =
  group ~timeout "recording"
    [
      test "each engine traces one unit, engines 0 and 1 its instructions"
        (fun () -> expect (selection ()) @@ __POS_OF__ {|
          GFX 9.4.3 engine 0: MASK 0xcf80 TOKEN_MASK 0xffbfff
          GFX 9.4.3 engine 1: MASK 0xcf80 TOKEN_MASK 0xffbfff
          GFX 9.4.3 engine 2: MASK 0xcf80 TOKEN_MASK 0xff93ff
          GFX 11.0.0 engine 0: MASK 0x15400 TOKEN_MASK 0x3f1000
          GFX 11.0.0 engine 1: MASK 0x15400 TOKEN_MASK 0x3f1000
          GFX 11.0.0 engine 2: MASK 0x15400 TOKEN_MASK 0x3f1127
          GFX 12.0.0 engine 0: MASK 0x15400 TOKEN_MASK 0x83f6000
          GFX 12.0.0 engine 1: MASK 0x15400 TOKEN_MASK 0x83f6000
          GFX 12.0.0 engine 2: MASK 0x15400 TOKEN_MASK 0x83f6127
          |});
      engine_law
        "a start programs each engine's buffer on it, and no harvested one"
        (fun g ~wgps ~size ->
          Thread_trace.start g ~wgps ~size ~address:buffers);
      engine_law
        "a stop stores each engine's end from it, and no harvested one's"
        (fun g ~wgps ~size:_ -> Thread_trace.stop g ~wgps ~address:ends);
      prop "a start and a stop acquire every cache before and after" traced
        (fun g ->
          let acquire = S.encode (Pm4.acquire_mem g System) in
          let n = List.length acquire in
          List.iter
            (fun p ->
              let ws = S.words (p g) in
              let k = List.length ws in
              equal
                (pair (list int) (list int))
                (acquire, acquire)
                ( List.filteri (fun i _ -> i < n) ws,
                  List.filteri (fun i _ -> i >= k - n) ws ))
            [ start ~size:page; stop ]);
      cases "a size of no whole pages is refused" ~id:string_of_int
        [ min_int; -4096; 0; 1; 4095; 4097 ] (fun size ->
          raises_match (Exn.invalid_arg ~sub:"Thread_trace.start") (fun () ->
              start (gpu (11, 0, 0)) ~size));
      cases "a buffer address of no whole pages is refused" ~id:string_of_int
        [ min_int; -4096; -1; 1; 4095; 4097 ] (fun address ->
          raises_match (Exn.invalid_arg ~sub:"Thread_trace.start") (fun () ->
              let g = gpu (11, 0, 0) in
              Thread_trace.start g ~wgps:(S.harvest_none g) ~size:page ~address));
      cases "an end address of no whole words is refused" ~id:string_of_int
        [ min_int; -4; -1; 1; 2; 3; 4097 ] (fun address ->
          raises_match (Exn.invalid_arg ~sub:"Thread_trace.stop") (fun () ->
              let g = gpu (11, 0, 0) in
              Thread_trace.stop g ~wgps:(S.harvest_none g) ~address));
      cases "a harvest of another number of engines than the GPU's is refused"
        ~id:string_of_int [ 3; 5 ] (fun n ->
          let g = gpu (11, 0, 0) and wgps = Array.make n [| 1 |] in
          raises_match (Exn.invalid_arg ~sub:"Thread_trace.start") (fun () ->
              Thread_trace.start g ~wgps ~size:page ~address:buffers);
          raises_match (Exn.invalid_arg ~sub:"Thread_trace.stop") (fun () ->
              Thread_trace.stop g ~wgps ~address:ends));
      test "a GC without the programs' registers is refused" (fun () ->
          let g = gpu (10, 3, 0) in
          raises_match (Exn.invalid_arg ~sub:"Thread_trace.start") (fun () ->
              start g ~size:page);
          raises_match (Exn.invalid_arg ~sub:"Thread_trace.stop") (fun () ->
              stop g));
      cases "a size past 2^22 - 1 pages is refused"
        ~id:(fun (g, size) -> strf "GC %s, %d" (S.version g.Gpu.gc) size)
        (List.concat_map
           (fun gc -> List.map (fun s -> (gpu gc, s)) [ 1 lsl 34; max_pages ])
           trace_families)
        (fun (g, size) ->
          raises_match (Exn.invalid_arg ~sub:"Thread_trace.start") (fun () ->
              start g ~size));
      cases "a size up to 2^22 - 1 pages is the size field's pages"
        ~id:(fun (g, size) -> strf "GC %s, %d" (S.version g.Gpu.gc) size)
        (List.concat_map
           (fun gc ->
             List.map (fun s -> (gpu gc, s)) [ 4096; (1 lsl 34) - 4096 ])
           trace_families)
        (fun (g, size) -> equal int (size / 4096) (size_field g size));
      test "a GFX 11.0 write pointer counts from address 0" (fun () ->
          equal int 0x100
            (Thread_trace.length
               (gpu (11, 0, 0))
               ~buffer:0x10_0000
               ((0x10_0000 + 0x100) / 32)));
      cases "a write pointer counts from the buffer" ~id:S.version
        [ (9, 4, 3); (11, 5, 0); (12, 0, 1) ]
        (fun v ->
          equal int 0x100
            (Thread_trace.length (gpu v) ~buffer:0x10_0000 (0x100 / 32)));
    ]

(* Decoding *)

(* GFX12 traces written by hand, as GFX12 lays out its packets: AMD's
   MIT-licensed rocprof-trace-decoder, source/gfx12. *)

let gfx12 = gpu (12, 0, 1)

(* A stream of nibbles, least significant first. *)
type stream = { mutable units : (int * int) list }

let stream () = { units = [] }
let put s ~units v = s.units <- (v, units) :: s.units

let bytes s =
  let b = Buffer.create 64 and acc = ref 0 and bits = ref 0 in
  List.iter
    (fun (v, units) ->
      for i = 0 to units - 1 do
        acc := !acc lor (((v lsr (i * 4)) land 0xf) lsl !bits);
        bits := !bits + 4;
        while !bits >= 8 do
          Buffer.add_char b (Char.chr (!acc land 0xff));
          acc := !acc lsr 8;
          bits := !bits - 8
        done
      done)
    (List.rev s.units);
  if !bits > 0 then Buffer.add_char b (Char.chr !acc);
  Buffer.contents b

(* The trace's header, of version 4. *)
let header s = put s ~units:16 (0x11 lor (4 lsl 7))

(* A wave's slot: its shader array, work-group processor, SIMD and slot. *)
type key = { sa : int; wgp : int; simd : int; slot : int }

(* A wave's compute unit: its work-group processor, its array above it. *)
let cu k = k.wgp lor (k.sa lsl 4)

(* A wave event's packet, with [delta] in its delta field. *)
let wave ~start ~delta k =
  if start then
    ( 8,
      0b01100 lor (delta lsl 5) lor (k.sa lsl 7) lor (k.simd lsl 8)
      lor (k.wgp lsl 10) lor (k.slot lsl 15) )
  else
    ( 5,
      0b10101 lor (delta lsl 5) lor (k.sa lsl 8) lor (k.simd lsl 9)
      lor (k.wgp lsl 11) lor (k.slot lsl 15) )

(* A trace's events. A wave's start or end holds the cycles since the packet
   before it in its own delta field: at most 3 for a start, 7 for an end. *)
type event =
  | Gap of int  (** Cycles pass: 4 to 19 in a short packet, others long. *)
  | Start of key * int
  | End of key * int
  | Mark of int  (** A realtime marker of this realtime. *)

let trace evs =
  let s = stream () in
  header s;
  let on = function
    | Gap n when n >= 4 && n <= 19 ->
        put s ~units:2 (0b1000 lor ((n - 4) lsl 4))
    | Gap n -> put s ~units:16 (0x01 lor (n lsl 12))
    | Start (k, delta) ->
        let units, v = wave ~start:true ~delta k in
        put s ~units v
    | End (k, delta) ->
        let units, v = wave ~start:false ~delta k in
        put s ~units v
    | Mark rt -> put s ~units:16 (0x01 lor (1 lsl 7) lor (rt lsl 12))
  in
  List.iter on evs;
  bytes s

let decoded =
  Testable.make
    ~pp:(fun ppf (w : Thread_trace.wave) ->
      Format.fprintf ppf "{cu %d; simd %d; slot %d; %d-%d}" w.cu w.simd w.slot
        w.start w.stop)
    ~equal:( = )

let wave_of k start stop : Thread_trace.wave =
  { cu = cu k; simd = k.simd; slot = k.slot; start; stop }

(* A slot at each field's zero, one at each field's edge, and slots that
   differ from the first in one field. *)
let a = { sa = 0; wgp = 0; simd = 0; slot = 0 }
let b = { sa = 1; wgp = 15; simd = 3; slot = 31 }

let one_off =
  [
    { a with sa = 1 };
    { a with wgp = 15 };
    { a with simd = 3 };
    { a with slot = 31 };
  ]

(* Traces and the waves the .mli states for them. *)
let pairings =
  [
    ( "an end pairs with its slot's start, in the order waves end",
      [
        Start (a, 0);
        Gap 5;
        Start (b, 0);
        Gap 10;
        End (b, 0);
        Gap 1000;
        End (a, 7);
      ],
      [ wave_of b 5 15; wave_of a 0 1022 ] );
    ( "a start on an open slot drops the wave before it",
      [ Start (a, 3); Gap 4; Start (a, 0); Gap 19; End (a, 0) ],
      [ wave_of a 7 26 ] );
    ( "an end with no start since the slot's last end is dropped",
      [ End (a, 0); Start (a, 1); Gap 20; End (a, 0); End (a, 2) ],
      [ wave_of a 1 21 ] );
    ( "slots that differ in one field pair apart",
      List.map (fun k -> Start (k, 1)) (a :: one_off)
      @ List.rev_map (fun k -> End (k, 1)) (a :: one_off),
      List.rev
        (List.mapi (fun i k -> wave_of k (i + 1) (10 - i)) (a :: one_off)) );
  ]

(* Zero bytes after a trace: no-ops, which the hardware writes up to a whole
   32-byte unit. *)
let padding = String.make 32 '\000'

(* A trace an R9700 wrote (fixtures/README.md): its waves, each with its compute
   unit, SIMD, slot and times, and how many there are. Its 0x06 and 0x46 packets
   lie among their starts and ends. *)
let captured () =
  let trace =
    In_channel.with_open_bin "fixtures/ttracedata_gfx1201.sqtt"
      In_channel.input_all
  in
  let waves = Thread_trace.waves gfx12 trace in
  let line (w : Thread_trace.wave) =
    strf "cu %d simd %d slot %d: %d-%d" w.cu w.simd w.slot w.start w.stop
  in
  expect
    (String.concat "\n"
       (strf "%d waves" (List.length waves) :: List.map line waves))
  @@ __POS_OF__
       {|
    32 waves
    cu 1 simd 0 slot 0: 2562-5240
    cu 1 simd 2 slot 0: 2563-5247
    cu 1 simd 1 slot 0: 2579-5248
    cu 1 simd 3 slot 0: 2580-5251
    cu 0 simd 0 slot 0: 2558-5351
    cu 0 simd 2 slot 0: 2559-5371
    cu 0 simd 1 slot 0: 2577-5387
    cu 16 simd 0 slot 0: 2569-5392
    cu 16 simd 2 slot 0: 2570-5394
    cu 17 simd 0 slot 0: 2571-5397
    cu 17 simd 1 slot 0: 2587-5400
    cu 0 simd 3 slot 0: 2578-5403
    cu 16 simd 1 slot 0: 2585-5404
    cu 17 simd 2 slot 0: 2572-5405
    cu 17 simd 3 slot 0: 2588-5408
    cu 16 simd 3 slot 0: 2586-5420
    cu 3 simd 0 slot 0: 2567-5528
    cu 3 simd 1 slot 0: 2583-5533
    cu 3 simd 2 slot 0: 2568-5535
    cu 3 simd 3 slot 0: 2584-5537
    cu 2 simd 0 slot 0: 2565-5631
    cu 2 simd 2 slot 0: 2566-5639
    cu 2 simd 1 slot 0: 2581-5640
    cu 2 simd 3 slot 0: 2582-5642
    cu 19 simd 0 slot 0: 2575-5712
    cu 18 simd 0 slot 0: 2573-5713
    cu 19 simd 1 slot 0: 2594-5715
    cu 18 simd 2 slot 0: 2574-5717
    cu 19 simd 2 slot 0: 2576-5718
    cu 18 simd 1 slot 0: 2590-5720
    cu 19 simd 3 slot 0: 2595-5721
    cu 18 simd 3 slot 0: 2591-5723
  |}

(* AMD's rocprof-trace-decoder at 1753e7a4, source/gfx12/gfx12token.cpp: packet
   lengths and delta fields. Payloads can contain a WAVE_END prefix; consuming a
   short packet must not turn that payload into a wave event. *)
let packet_cases =
  [
    ("NEW_PC", 0x21, 18, 8, [ 0; 1; 7 ]);
    ("EXEC_POPCOUNT1", 0x66, 6, 7, [ 0; 1; 7 ]);
    ("EXEC_POPCOUNT3", 0x16, 12, 6, [ 0; 1; 7 ]);
    ("SHADER_DATA", 0x06, 14, 7, [ 0; 1; 7 ]);
    ("SHADER_DATA_SHORT", 0x46, 8, 7, [ 0; 1; 7 ]);
    (* The inherited prefixes survive in the byte keys no GFX12 row replaces.
       Their bit 5 is set and belongs to the delta. *)
    ("inherited SHADER_DATA", 0x26, 13, 5, [ 1; 5 ]);
    ("inherited SHADER_DATA_SHORT", 0x36, 7, 5, [ 1; 3; 5; 7 ]);
  ]

(* The trace of [s] decodes to [expected] whole, and to no wave cut short. *)
let only_whole ~msg s expected =
  let trace = bytes s in
  for n = 0 to String.length trace do
    equal
      ~msg:(strf "%s, %d bytes" msg n)
      (list decoded)
      (if n = String.length trace then [ expected ] else [])
      (Thread_trace.waves gfx12 (String.sub trace 0 n))
  done

let packet_framing (name, prefix, nibbles, delta_bit, deltas) =
  List.iter
    (fun (offset, delta) ->
      let s = stream () in
      header s;
      put s ~units:offset 0;
      let k = { sa = 1; wgp = 7; simd = 3; slot = 31 } in
      let emit ~start ~delta =
        let units, word = wave ~start ~delta k in
        put s ~units word
      in
      emit ~start:true ~delta:1;
      put s ~units:4 (prefix lor (delta lsl delta_bit));
      for i = 4 to nibbles - 1 do
        put s ~units:1 (if i = 8 then 5 else if i = 9 then 15 else 0)
      done;
      emit ~start:false ~delta:2;
      only_whole
        ~msg:(strf "%s, nibble %d, delta %d" name offset delta)
        s
        { cu = cu k; simd = k.simd; slot = k.slot; start = 1; stop = delta + 3 })
    (List.concat_map
       (fun offset -> List.map (fun delta -> (offset, delta)) deltas)
       [ 0; 1 ])

(* An extended start is followed by zero or more bytes whose low bit is set. The
   next non-NOP packet with an even prefix ends this continuation. See
   gfx12token.cpp and gfx12/quick_scan.cpp in the same AMD source. *)
let extended_start ~offset continuations =
  let s = stream () in
  header s;
  put s ~units:offset 0;
  let k = { sa = 1; wgp = 15; simd = 3; slot = 31 } in
  let units, word = wave ~start:true ~delta:3 k in
  put s ~units (word lor 0x10);
  put s ~units:2 0;
  List.iter (fun byte -> put s ~units:2 byte) continuations;
  (* An instruction with a delta of 2 ends the continuation. *)
  put s ~units:5 (0x2 lor (2 lsl 3));
  let units, word = wave ~start:false ~delta:4 k in
  put s ~units word;
  only_whole ~msg:(strf "nibble %d" offset) s
    { cu = cu k; simd = k.simd; slot = k.slot; start = 3; stop = 9 }

(* Other generations' traces, by hand *)

(* GFX9's 16-bit tokens (vega10_enum.h, gc_9_4_3_sh_mask.h): a WAVE_START (3) of
   two words, TIME_DELTA from bit 4, CU_ID from 6, WAVE_ID from 10 and SIMD_ID
   from 14; a MISC (0) of 25 deltas of 4 cycles at bit 4; a WAVE_END (6) of one
   word, laid out as the start. *)
let gfx9_trace = "\x43\x8c\x00\x00\x90\x01\x46\x8c"

(* GFX11's nibbles (rocprof-trace-decoder, source/gfx11): a header of version 3;
   a realtime marker of 1000 (0x01, bit 9 set, the time from bit 12); a wave
   start (0b01100) of array 1, work-group processor 2, SIMD 3 and slot 4, its
   array at bit 7, SIMD at 8, processor at 10 and slot at 13; a delta of 10
   cycles (0b1000, less 4 from bit 4); a marker of 1100; and the wave's end
   (0b10101), array at bit 8, SIMD at 9, processor at 11 and slot at 15. *)
let gfx11_trace () =
  let s = stream () in
  put s ~units:16 (0x11 lor (3 lsl 7));
  put s ~units:12 (0x01 lor (1 lsl 9) lor (1000 lsl 12));
  put s ~units:8
    (0b01100 lor (1 lsl 7) lor (3 lsl 8) lor (2 lsl 10) lor (4 lsl 13));
  put s ~units:2 (0b1000 lor (6 lsl 4));
  put s ~units:12 (0x01 lor (1 lsl 9) lor (1100 lsl 12));
  put s ~units:5
    (0b10101 lor (1 lsl 8) lor (3 lsl 9) lor (2 lsl 11) lor (4 lsl 15));
  bytes s ^ padding

let decoding =
  group ~timeout "decoding"
    [
      cases "an instruction packet preserves the following wave end"
        ~id:(fun (name, _, _, _, _) -> name)
        packet_cases packet_framing;
      cases "an extended start records its wave" ~id:fst
        [
          ("workgroup", []);
          ("VGPR/LDS", [ 0xf5; 0x3f ]);
          ("VGPR/LDS with NOPs", [ 0xf5; 0; 0x3f ]);
        ]
        (fun (_, continuations) ->
          List.iter
            (fun offset -> extended_start ~offset continuations)
            [ 0; 1 ]);
      cases "a trace's waves"
        ~id:(fun (n, _, _) -> n)
        pairings
        (fun (_, evs, expected) ->
          equal (list decoded) expected (Thread_trace.waves gfx12 (trace evs)));
      cases "a trace cut short yields a prefix of its waves"
        ~id:(fun (n, _, _) -> n)
        pairings
        (fun (_, evs, expected) ->
          let t = trace evs in
          for n = 0 to String.length t - 1 do
            let got = Thread_trace.waves gfx12 (String.sub t 0 n) in
            let k = List.length got in
            equal ~msg:(strf "%d bytes" n) (list decoded)
              (List.filteri (fun i _ -> i < k) expected)
              got
          done);
      test "a trace's clock runs on the line through the markers around a time"
        (fun () ->
          let t = trace [ Mark 1000; Gap 10; Mark 1100; Gap 20; Mark 1140 ] in
          let f = require_some (Thread_trace.clock gfx12 (t ^ padding)) in
          equal (list int)
            [ 900; 1000; 1010; 1050; 1100; 1102; 1120; 1140; 1142; 1160 ]
            (List.map f [ -10; 0; 1; 5; 10; 11; 20; 30; 31; 40 ]));
      test "a trace of one marker has no clock" (fun () ->
          let t = trace [ Gap 5; Mark 1000 ] in
          is_none (Thread_trace.clock gfx12 (t ^ padding)));
      cases "an empty trace has no waves and no clock" ~id:S.version
        [ (9, 4, 3); (11, 0, 0); (12, 0, 1) ]
        (fun v ->
          equal (pair int bool) (0, true)
            ( List.length (Thread_trace.waves (gpu v) ""),
              Option.is_none (Thread_trace.clock (gpu v) "") ));
      test "a gfx1201 trace of s_ttracedata decodes to its waves" captured;
      test "a GFX9 trace's wave counts 4 cycles a delta" (fun () ->
          let g = gpu (9, 4, 3) in
          equal (list decoded)
            [ { cu = 1; simd = 2; slot = 3; start = 0; stop = 100 } ]
            (Thread_trace.waves g gfx9_trace);
          is_none (Thread_trace.clock g gfx9_trace));
      test "a GFX11 trace's wave and clock" (fun () ->
          let g = gpu (11, 0, 0) and trace = gfx11_trace () in
          (* Compute unit 2 of array 1: 2 + (1 lsl 3). *)
          equal (list decoded)
            [ { cu = 10; simd = 3; slot = 4; start = 0; stop = 10 } ]
            (Thread_trace.waves g trace);
          let f = require_some (Thread_trace.clock g trace) in
          equal (list int) [ 1000; 1050; 1100 ] (List.map f [ 0; 5; 10 ]));
      test "two markers at one shader time give no clock" (fun () ->
          let s = stream () in
          put s ~units:16 (0x11 lor (3 lsl 7));
          put s ~units:12 (0x01 lor (1 lsl 9) lor (100 lsl 12));
          put s ~units:12 (0x01 lor (1 lsl 9) lor (200 lsl 12));
          is_none (Thread_trace.clock (gpu (11, 0, 0)) (bytes s ^ padding)));
    ]

let () = exit (run "rig_amd_abi.thread_trace" [ recording; decoding ])
