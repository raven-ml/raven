(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Thread traces: the program each generation writes, the write pointer's units,
   and the decoder against a captured gfx1201 trace, GFX12 traces this suite
   writes from a list of events, and GFX9 and GFX11 traces written by hand. *)

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

(* The words of [p] that make [n] 32-bit words, and the words after them. *)
let rec split n (p : int Packet.t) =
  if n <= 0 then ([], p)
  else
    match p with
    | [] -> ([], [])
    | w :: rest ->
        let body, rest = split (n - Packet.size [ w ]) rest in
        (w :: body, rest)

(* Walks [p], the words of a program over engine numbers, and gives each use of
   an engine's number. Each packet starts with a constant header. *)
let uses g (p : int Packet.t) =
  let grbm = require_some (Register.find g "regGRBM_GFX_INDEX") in
  let grbm_at = Register.address g grbm in
  let field w f =
    let lo, hi = List.assoc f grbm.fields in
    (w lsr lo) land ((1 lsl (hi - lo + 1)) - 1)
  in
  let selected = ref None and dies = ref None and predicated = ref 0 in
  let out = ref [] in
  let rec engines : int Packet.term -> int list = function
    | Value e -> [ e ]
    | Add (t, _) | Shift (t, _) | Or (t, _) -> engines t
  in
  let use dies t =
    List.iter
      (fun engine -> out := { engine; dies; selected = !selected } :: !out)
      (engines t)
  in
  let rec go = function
    | [] -> ()
    | Packet.Dword h :: rest ->
        let op = (h lsr 8) land 0xff and n = ((h lsr 16) land 0x3fff) + 1 in
        let body, rest = split n rest in
        let dies_now = if !predicated > 0 then !dies else None in
        if !predicated > 0 then
          predicated := !predicated - (Packet.size body + 1);
        (match (op, body) with
        | op, [ Dword w ] when op = S.pred_exec ->
            dies := Some (w lsr 24);
            predicated := w land 0x3fff
        | op, Dword off :: [ Dword w ]
          when op = S.set_uconfig_reg && S.uconfig_start + off = grbm_at ->
            selected :=
              if field w "se_broadcast_writes" = 1 then None
              else Some (field w "se_index")
        | _ -> ());
        List.iter
          (function Packet.W32 t | W64 t -> use dies_now t | Dword _ -> ())
          body;
        go rest
    | (W32 _ | W64 _) :: _ -> fail "a packet starts with a term"
  in
  go p;
  List.rev !out

let engines (g : Gpu.t) = List.init (g.shader_engines * g.xccs) Fun.id
let start g = Thread_trace.start g ~wgps:(S.harvest_none g)
let stop g = Thread_trace.stop g ~wgps:(S.harvest_none g)

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

let engine_law name program =
  prop ~timeout name traced (fun g ->
      let us = uses g (program g) in
      List.iter
        (fun e ->
          let mine = List.filter (fun u -> u.engine = e) us in
          if mine = [] then failf "engine %d is unused" e;
          List.iter
            (fun u ->
              equal ~msg:(strf "engine %d" e)
                (Testable.make ~pp:pp_use ~equal:( = ))
                (expected g e) u)
            mine)
        (engines g);
      List.iter (fun u -> less int ~than:(List.length (engines g)) u.engine) us)

(* An engine whose arrays all run no work, harvested, is neither programmed nor
   awaited: its die's other engines are, each as [engine_law] states. *)
let harvested name program =
  let gen =
    Gen.with_pp
      (fun ppf ((g : Gpu.t), off) ->
        Format.fprintf ppf "GC %s, %d dies of %d engines, %s off"
          (S.version g.gc) g.xccs g.shader_engines
          (String.concat " " (List.map string_of_int off)))
      (let open Gen in
       let* g = traced in
       let n = g.shader_engines * g.xccs in
       let+ off = list ~size:(int_range 1 n) (int_range 0 (n - 1)) in
       (g, List.sort_uniq compare off))
  in
  prop ~timeout name gen (fun (g, off) ->
      let wgps =
        Array.init (g.shader_engines * g.xccs) (fun e ->
            if List.mem e off then [| 0; 0 |] else [| 0; 1 |])
      in
      cover "every engine off" (List.length off = g.shader_engines * g.xccs);
      let us = uses g (program g wgps) in
      List.iter
        (fun e ->
          let mine = List.filter (fun u -> u.engine = e) us in
          if List.mem e off then
            equal int
              ~msg:(strf "uses of harvested engine %d" e)
              0 (List.length mine)
          else if mine = [] then failf "engine %d is unused" e)
        (engines g))

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
  let ws = S.encode (start g ~size (fun _ -> 0)) in
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
    engines 0 (S.writes (S.encode (start g ~size:4096 (fun _ -> 0))))
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
      engine_law "a start programs each engine's buffer on that engine"
        (fun g -> start g ~size:4096 Fun.id);
      engine_law "a stop stores each engine's end from that engine" (fun g ->
          stop g Fun.id);
      harvested "a start programs no engine whose arrays run no work"
        (fun g wgps -> Thread_trace.start g ~wgps ~size:4096 Fun.id);
      harvested "a stop awaits no engine whose arrays run no work"
        (fun g wgps -> Thread_trace.stop g ~wgps Fun.id);
      prop "a start and a stop acquire every cache before and after" traced
        (fun g ->
          let acquire = S.encode (Pm4.acquire_mem g System) in
          let n = List.length acquire in
          List.iter
            (fun p ->
              let ws = S.encode (p g) in
              let k = List.length ws in
              equal
                (pair (list int) (list int))
                (acquire, acquire)
                ( List.filteri (fun i _ -> i < n) ws,
                  List.filteri (fun i _ -> i >= k - n) ws ))
            [
              (fun g -> start g ~size:4096 (fun _ -> 0));
              (fun g -> stop g (fun _ -> 0));
            ]);
      cases ~name:string_of_int "a size of no whole pages is refused"
        [ min_int; -4096; 0; 1; 4095; 4097 ] (fun size ->
          raises_match (Exn.invalid_arg ~substring:"Thread_trace.start")
            (fun () -> start (gpu (11, 0, 0)) ~size (fun _ -> 0)));
      cases ~name:string_of_int
        "a harvest of another number of engines than the GPU's is refused"
        [ 3; 5 ] (fun n ->
          let g = gpu (11, 0, 0) and wgps = Array.make n [| 1 |] in
          raises_match (Exn.invalid_arg ~substring:"Thread_trace.start")
            (fun () -> Thread_trace.start g ~wgps ~size:4096 (fun _ -> 0));
          raises_match (Exn.invalid_arg ~substring:"Thread_trace.stop")
            (fun () -> Thread_trace.stop g ~wgps (fun _ -> 0)));
      test "a GC without the programs' registers is refused" (fun () ->
          let g = gpu (10, 3, 0) in
          raises_match (Exn.invalid_arg ~substring:"Thread_trace.start")
            (fun () -> start g ~size:4096 (fun _ -> 0));
          raises_match (Exn.invalid_arg ~substring:"Thread_trace.stop")
            (fun () -> stop g (fun _ -> 0)));
      cases ~name:string_of_int "a size of whole pages is taken"
        [ 4096; 1 lsl 30 ]
        (fun size ->
          let p = start (gpu (11, 0, 0)) ~size (fun _ -> 0) in
          greater int ~than:0 (Packet.size p));
      cases
        ~name:(fun (g, size) -> strf "GC %s, %d" (S.version g.Gpu.gc) size)
        "a size past 2^22 - 1 pages is refused"
        (List.concat_map
           (fun gc -> List.map (fun s -> (gpu gc, s)) [ 1 lsl 34; max_pages ])
           trace_families)
        (fun (g, size) ->
          raises_match (Exn.invalid_arg ~substring:"Thread_trace.start")
            (fun () -> start g ~size (fun _ -> 0)));
      cases
        ~name:(fun (g, size) -> strf "GC %s, %d" (S.version g.Gpu.gc) size)
        "a size up to 2^22 - 1 pages is the size field's pages"
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
      cases ~name:S.version "a write pointer counts from the buffer"
        [ (9, 4, 3); (11, 5, 0); (12, 0, 1) ]
        (fun v ->
          equal int 0x100
            (Thread_trace.length (gpu v) ~buffer:0x10_0000 (0x100 / 32)));
    ]

(* Decoding *)

(* Traces this suite writes from a list of events, as GFX12 lays out its
   packets: AMD's MIT-licensed rocprof-trace-decoder, source/gfx12. *)

type key = { sa : int; wgp : int; simd : int; slot : int }

type event =
  | Gap of int  (** Cycles pass. *)
  | Start of key * bool  (** A wave starts; [true] holds the gap in it. *)
  | End of key * bool  (** A wave ends. *)
  | Mark of int  (** A clock marker, at a slope of the realtime clock. *)
  | Data of int  (** An [s_ttracedata_imm]'s byte: a 0x46 packet. *)
  | Nop

let pp_key ppf k =
  Format.fprintf ppf "{sa %d; wgp %d; simd %d; slot %d}" k.sa k.wgp k.simd
    k.slot

let pp_event ppf = function
  | Gap n -> Format.fprintf ppf "Gap %d" n
  | Start (k, i) -> Format.fprintf ppf "Start (%a, %b)" pp_key k i
  | End (k, i) -> Format.fprintf ppf "End (%a, %b)" pp_key k i
  | Mark s -> Format.fprintf ppf "Mark %d" s
  | Data d -> Format.fprintf ppf "Data %d" d
  | Nop -> Format.fprintf ppf "Nop"

(* Keys of few values, so that waves share them, at the fields' edges. *)
let key =
  let open Gen in
  let+ sa = int_range 0 1
  and+ wgp = of_list [ 0; 1; 7 ]
  and+ simd = int_range 0 3
  and+ slot = of_list [ 0; 1; 15 ] in
  { sa; wgp; simd; slot }

let event key =
  let open Gen in
  frequency
    [
      (3, map (fun n -> Gap n) (int_range 0 20));
      (1, map (fun n -> Gap n) (int_range 0 (1 lsl 24)));
      (3, map (fun (k, i) -> Start (k, i)) (pair key bool));
      (3, map (fun (k, i) -> End (k, i)) (pair key bool));
      (1, map (fun s -> Mark s) (int_range 1 8));
      (1, map (fun d -> Data d) (int_range 0 255));
      (1, constant Nop);
    ]

(* Events over a few keys, so that waves end and keys come back. *)
let traces =
  Gen.with_pp
    (fun ppf evs ->
      Format.fprintf ppf "[%a]"
        (Format.pp_print_list
           ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
           pp_event)
        evs)
    (let open Gen in
     let* keys = list ~size:(int_range 1 3) key in
     list ~size:(int_range 0 60) (event (of_list keys)))

let gfx12 = gpu (12, 0, 1)

(* A stream of nibbles, least significant first. *)
type stream = { mutable units : (int * int) list; mutable length : int }

let stream () = { units = []; length = 0 }

let put s ~units v =
  s.units <- (v, units) :: s.units;
  s.length <- s.length + units

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

(* The largest delta a wave event's own field holds. *)
let inline_max ~start = if start then 3 else 7

type wave_t = { key : key; start : int; stop : int }

(* What a trace of [events] holds, as the .mli states it: waves paired by key,
   an end with its key's latest start, a start on an open key leaving the one
   before it no wave, in the order they end, each with the nibble where its
   end's packet ends; and markers, as (shader time, realtime). *)
type model = {
  trace : string;
  waves : (wave_t * int) list;
  markers : (int * int) list;
}

let write evs =
  let s = stream () in
  header s;
  let time = ref 0 and pending = ref 0 in
  let flush () =
    let d = !pending in
    pending := 0;
    if d >= 4 && d <= 19 then put s ~units:2 (0b1000 lor ((d - 4) lsl 4))
    else if d > 0 then put s ~units:16 (0x01 lor (d lsl 12))
  in
  let open_ = Hashtbl.create 8 in
  let waves = ref [] and markers = ref [] and last_rt = ref None in
  let on_wave ~start k inline =
    let delta =
      if inline && !pending <= inline_max ~start then begin
        let d = !pending in
        pending := 0;
        d
      end
      else begin
        flush ();
        0
      end
    in
    let units, v = wave ~start ~delta k in
    put s ~units v
  in
  List.iter
    (function
      | Gap n ->
          time := !time + n;
          pending := !pending + n
      | Start (k, inline) ->
          on_wave ~start:true k inline;
          Hashtbl.replace open_ k !time
      | End (k, inline) ->
          on_wave ~start:false k inline;
          Option.iter
            (fun start ->
              Hashtbl.remove open_ k;
              waves := ({ key = k; start; stop = !time }, s.length) :: !waves)
            (Hashtbl.find_opt open_ k)
      | Mark slope -> (
          match !last_rt with
          | Some (t, _) when t = !time -> ()
          | last ->
              flush ();
              let rt =
                match last with
                | None -> 1_000_000 + !time
                | Some (t, r) -> r + (slope * (!time - t))
              in
              put s ~units:16 (0x01 lor (1 lsl 7) lor (rt lsl 12));
              last_rt := Some (!time, rt);
              markers := (!time, rt) :: !markers)
      | Data d ->
          (* Its byte at bits 22 to 29, its delta field 0. *)
          flush ();
          put s ~units:8 (0x46 lor (d lsl 22))
      | Nop -> put s ~units:1 0)
    evs;
  flush ();
  { trace = bytes s; waves = List.rev !waves; markers = List.rev !markers }

(* Whether [evs] start a key that is open: a trace that lost a wave's end. *)
let restarts evs =
  let open_ = Hashtbl.create 8 in
  List.exists
    (function
      | Start (k, _) ->
          let restart = Hashtbl.mem open_ k in
          Hashtbl.replace open_ k ();
          restart
      | End (k, _) ->
          Hashtbl.remove open_ k;
          false
      | Gap _ | Mark _ | Data _ | Nop -> false)
    evs

let decoded =
  Testable.make
    ~pp:(fun ppf (w : Thread_trace.wave) ->
      Format.fprintf ppf "{cu %d; simd %d; slot %d; %d-%d}" w.cu w.simd w.slot
        w.start w.stop)
    ~equal:( = )

let as_decoded ({ key = k; start; stop }, _) : Thread_trace.wave =
  { cu = cu k; simd = k.simd; slot = k.slot; start; stop }

(* Zero bytes after a trace: no-ops, which the hardware writes up to a whole
   32-byte unit. *)
let padding = String.make 32 '\000'

(* The realtime of shader time [t] on the line of [markers] around it. *)
let realtime markers t =
  let rec go = function
    | (s0, r0) :: ((s1, r1) :: rest' as rest) ->
        if t < s1 || rest' = [] then r0 + ((t - s0) * (r1 - r0) / (s1 - s0))
        else go rest
    | _ -> invalid_arg "realtime"
  in
  go markers

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
      cases
        ~name:(fun (name, _, _, _, _) -> name)
        "an instruction packet preserves the following wave end" packet_cases
        packet_framing;
      cases ~name:fst "an extended start records its wave"
        [
          ("workgroup", []);
          ("VGPR/LDS", [ 0xf5; 0x3f ]);
          ("VGPR/LDS with NOPs", [ 0xf5; 0; 0x3f ]);
        ]
        (fun (_, continuations) ->
          List.iter
            (fun offset -> extended_start ~offset continuations)
            [ 0; 1 ]);
      prop "a trace's waves pair each end with its key's latest open start"
        traces (fun evs ->
          let m = write evs in
          cover "a wave" (m.waves <> []);
          cover "a 0x46 packet"
            (List.exists (function Data _ -> true | _ -> false) evs);
          cover "a wave whose key ends twice"
            (List.length
               (List.sort_uniq compare (List.map (fun (w, _) -> w.key) m.waves))
            < List.length m.waves);
          cover "a start on an open key" (restarts evs);
          equal (list decoded)
            (List.map as_decoded m.waves)
            (Thread_trace.waves gfx12 (m.trace ^ padding)));
      prop "a trace's clock passes through its markers, and their lines" traces
        (fun evs ->
          let m = write evs in
          match (m.markers, Thread_trace.clock gfx12 (m.trace ^ padding)) with
          | ([] | [ _ ]), c -> is_none ~msg:"fewer than two markers" c
          | (s0, _) :: _, None -> failf "no clock from the markers at %d" s0
          | markers, Some f ->
              cover "three markers" (List.length markers > 2);
              let first = fst (List.hd markers) in
              let last = fst (List.hd (List.rev markers)) in
              let ts =
                [ first - 1000; first; last; last + 1000 ]
                @ List.concat_map (fun (s, _) -> [ s; s + 1 ]) markers
              in
              List.iter
                (fun t ->
                  equal ~msg:(strf "time %d" t) int (realtime markers t) (f t))
                ts);
      prop "a trace cut short yields the waves of its whole packets" traces
        (fun evs ->
          let m = write evs in
          for n = 0 to String.length m.trace do
            let whole = List.filter (fun (_, e) -> e * 4 <= 8 * n) m.waves in
            equal ~msg:(strf "%d bytes" n) (list decoded)
              (List.map as_decoded whole)
              (Thread_trace.waves gfx12 (String.sub m.trace 0 n))
          done);
      cases ~name:S.version "an empty trace has no waves and no clock"
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
