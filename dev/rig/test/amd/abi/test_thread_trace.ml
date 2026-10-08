(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Thread traces: the program each generation writes, the write pointer's units,
   and the decoder against traces this suite writes from a list of events. *)

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
            equal int ~msg:(strf "uses of harvested engine %d" e) 0
              (List.length mine)
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

let recording =
  group ~timeout "recording"
    [
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

(* A trace this suite writes: a list of events, as the generations lay out their
   packets. Packet layouts are tinygrad's (tinygrad/renderer/amd/sqtt.py), the
   only reading of RDNA's format; GFX9's tokens are vega10_enum.h's
   SQ_THREAD_TRACE_TOKEN_* and gc_9_4_3_sh_mask.h's SQ_THREAD_TRACE_WORD_*
   fields, with deltas of 4 cycles. *)

type key = { sa : int; wgp : int; simd : int; slot : int }

type event =
  | Gap of int  (** Cycles pass. *)
  | Start of key * bool  (** A wave starts; [true] holds the gap in it. *)
  | End of key * bool  (** A wave ends. *)
  | Mark of int  (** A clock marker, at a slope of the realtime clock. *)
  | Nop

type format = Rdna3 | Rdna4 | Gfx9

let format_of (g : Gpu.t) =
  match g.gc with 9, _, _ -> Gfx9 | 11, _, _ -> Rdna3 | _ -> Rdna4

let pp_key ppf k =
  Format.fprintf ppf "{sa %d; wgp %d; simd %d; slot %d}" k.sa k.wgp k.simd
    k.slot

let pp_event ppf = function
  | Gap n -> Format.fprintf ppf "Gap %d" n
  | Start (k, i) -> Format.fprintf ppf "Start (%a, %b)" pp_key k i
  | End (k, i) -> Format.fprintf ppf "End (%a, %b)" pp_key k i
  | Mark s -> Format.fprintf ppf "Mark %d" s
  | Nop -> Format.fprintf ppf "Nop"

(* Keys of few values, so that waves share them, at the fields' edges. *)
let key =
  let open Gen in
  let+ sa = int_range 0 1
  and+ wgp = of_list [ 0; 1; 7 ]
  and+ simd = int_range 0 3
  and+ slot = of_list [ 0; 1; 15 ] in
  { sa; wgp; simd; slot }

let event ~far key =
  let open Gen in
  frequency
    [
      (3, map (fun n -> Gap n) (int_range 0 20));
      (1, map (fun n -> Gap n) (int_range 0 far));
      (3, map (fun (k, i) -> Start (k, i)) (pair key bool));
      (3, map (fun (k, i) -> End (k, i)) (pair key bool));
      (1, map (fun s -> Mark s) (int_range 1 8));
      (1, constant Nop);
    ]

let pp_gpu ppf (g : Gpu.t) = Format.fprintf ppf "GC %s" (S.version g.gc)

(* A GPU and events over a few keys, so that waves end and keys come back.
   GFX9's gaps stay short: its deltas take a token for each 1020 cycles. *)
let traces gpus =
  Gen.with_pp
    (fun ppf (g, evs) ->
      Format.fprintf ppf "%a: [%a]" pp_gpu g
        (Format.pp_print_list
           ~pp_sep:(fun ppf () -> Format.fprintf ppf "; ")
           pp_event)
        evs)
    (let open Gen in
     let* g = of_list gpus in
     let far = match g.Gpu.gc with 9, _, _ -> 1 lsl 10 | _ -> 1 lsl 24 in
     let* keys = list ~size:(int_range 1 3) key in
     let+ evs = list ~size:(int_range 0 60) (event ~far (of_list keys)) in
     (g, evs))

let gfx9 = gpu (9, 4, 3)
let rdna = [ gpu (11, 0, 0); gpu (12, 0, 1) ]

(* A stream of nibbles, or of GFX9's 16-bit words, least significant first. *)
type stream = { mutable units : (int * int) list; mutable length : int }

let put s ~units v =
  s.units <- (v, units) :: s.units;
  s.length <- s.length + units

let bytes fmt s =
  let unit_bits = match fmt with Gfx9 -> 16 | Rdna3 | Rdna4 -> 4 in
  let b = Buffer.create 64 and acc = ref 0 and bits = ref 0 in
  List.iter
    (fun (v, units) ->
      for i = 0 to units - 1 do
        acc :=
          !acc
          lor (((v lsr (i * unit_bits)) land ((1 lsl unit_bits) - 1)) lsl !bits);
        bits := !bits + unit_bits;
        while !bits >= 8 do
          Buffer.add_char b (Char.chr (!acc land 0xff));
          acc := !acc lsr 8;
          bits := !bits - 8
        done
      done)
    (List.rev s.units);
  if !bits > 0 then Buffer.add_char b (Char.chr !acc);
  Buffer.contents b

(* The cycles a delta field counts, and a wave's compute unit. *)
let scale = function Gfx9 -> 4 | Rdna3 | Rdna4 -> 1

let cu fmt k =
  match fmt with
  | Gfx9 -> k.wgp
  | Rdna3 -> k.wgp lor (k.sa lsl 3)
  | Rdna4 -> k.wgp lor (k.sa lsl 4)

(* A wave event's packet, with [delta] in its delta field. *)
let wave fmt ~start ~delta k =
  match (fmt, start) with
  | (Rdna3 | Rdna4), true ->
      let wgp, slot = if fmt = Rdna3 then (10, 13) else (10, 15) in
      ( 8,
        0b01100 lor (delta lsl 5) lor (k.sa lsl 7) lor (k.simd lsl 8)
        lor (k.wgp lsl wgp) lor (k.slot lsl slot) )
  | (Rdna3 | Rdna4), false ->
      ( 5,
        0b10101 lor (delta lsl 5) lor (k.sa lsl 8) lor (k.simd lsl 9)
        lor (k.wgp lsl 11) lor (k.slot lsl 15) )
  | Gfx9, _ ->
      let t = if start then 3 else 6 in
      ( (if start then 2 else 1),
        t lor (delta lsl 4) lor (k.wgp lsl 6) lor (k.slot lsl 10)
        lor (k.simd lsl 14) )

(* The largest delta a wave event's own field holds. *)
let inline_max fmt ~start =
  match fmt with Gfx9 -> 1 | Rdna3 | Rdna4 -> if start then 3 else 7

type wave_t = { key : key; start : int; stop : int }

(* What a trace of [events] holds, as the .mli states it: waves paired by key,
   in the order they end, each with the unit where its end's packet ends; and
   markers, as (shader time, realtime). *)
type model = {
  trace : string;
  waves : (wave_t * int) list;
  markers : (int * int) list;
}

let write fmt evs =
  let s = { units = []; length = 0 } in
  (match fmt with
  | Rdna3 -> put s ~units:16 (0x11 lor (3 lsl 7))
  | Rdna4 -> put s ~units:16 (0x11 lor (4 lsl 7))
  | Gfx9 -> ());
  let time = ref 0 and pending = ref 0 in
  let flush () =
    let d = !pending in
    pending := 0;
    match fmt with
    | Gfx9 ->
        let rec misc d =
          if d > 0 then begin
            let k = Int.min d 255 in
            put s ~units:1 (k lsl 4);
            misc (d - k)
          end
        in
        misc (d / 4)
    | Rdna3 | Rdna4 ->
        if d >= 4 && d <= 19 then put s ~units:2 (0b1000 lor ((d - 4) lsl 4))
        else if d > 0 then
          put s ~units:(if fmt = Rdna3 then 12 else 16) (0x01 lor (d lsl 12))
  in
  let open_ = Hashtbl.create 8 in
  let waves = ref [] and markers = ref [] and last_rt = ref None in
  let on_wave ~start k inline =
    let room = inline_max fmt ~start * scale fmt in
    let delta =
      if inline && !pending <= room then begin
        let d = !pending / scale fmt in
        pending := 0;
        d
      end
      else begin
        flush ();
        0
      end
    in
    let units, v = wave fmt ~start ~delta k in
    put s ~units v
  in
  List.iter
    (function
      | Gap n ->
          let n = n * scale fmt in
          time := !time + n;
          pending := !pending + n
      | Start (k, inline) ->
          let k = if fmt = Gfx9 then { k with sa = 0 } else k in
          if not (Hashtbl.mem open_ k) then begin
            on_wave ~start:true k inline;
            Hashtbl.replace open_ k !time
          end
      | End (k, inline) ->
          let k = if fmt = Gfx9 then { k with sa = 0 } else k in
          on_wave ~start:false k inline;
          Option.iter
            (fun start ->
              Hashtbl.remove open_ k;
              waves := ({ key = k; start; stop = !time }, s.length) :: !waves)
            (Hashtbl.find_opt open_ k)
      | Mark slope -> (
          match (fmt, !last_rt) with
          | Gfx9, _ -> ()
          | _, Some (t, _) when t = !time -> ()
          | _, last ->
              flush ();
              let rt =
                match last with
                | None -> 1_000_000 + !time
                | Some (t, r) -> r + (slope * (!time - t))
              in
              let v =
                if fmt = Rdna3 then 0x01 lor (1 lsl 9) lor (rt lsl 12)
                else 0x01 lor (1 lsl 7) lor (rt lsl 12)
              in
              put s ~units:(if fmt = Rdna3 then 12 else 16) v;
              last_rt := Some (!time, rt);
              markers := (!time, rt) :: !markers)
      | Nop -> if fmt <> Gfx9 then put s ~units:1 0)
    evs;
  flush ();
  { trace = bytes fmt s; waves = List.rev !waves; markers = List.rev !markers }

let decoded =
  Testable.make
    ~pp:(fun ppf (w : Thread_trace.wave) ->
      Format.fprintf ppf "{cu %d; simd %d; slot %d; %d-%d}" w.cu w.simd w.slot
        w.start w.stop)
    ~equal:( = )

let as_decoded fmt ({ key = k; start; stop }, _) : Thread_trace.wave =
  { cu = cu fmt k; simd = k.simd; slot = k.slot; start; stop }

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

(* Every prefix of a trace decodes to the waves whose end's packet it holds. *)
let cut_short name gpus =
  prop ~timeout name (traces gpus) (fun (g, evs) ->
      let fmt = format_of g in
      let m = write fmt evs in
      let bits = match fmt with Gfx9 -> 16 | Rdna3 | Rdna4 -> 4 in
      for n = 0 to String.length m.trace do
        let whole = List.filter (fun (_, e) -> e * bits <= 8 * n) m.waves in
        equal ~msg:(strf "%d bytes" n) (list decoded)
          (List.map (as_decoded fmt) whole)
          (Thread_trace.waves g (String.sub m.trace 0 n))
      done)

let decoding =
  group ~timeout "decoding"
    [
      prop "a trace's waves pair each start with its key's next end"
        (traces (gfx9 :: rdna))
        (fun (g, evs) ->
          let fmt = format_of g in
          let m = write fmt evs in
          cover "a wave" (m.waves <> []);
          cover "a wave whose key ends twice"
            (List.length
               (List.sort_uniq compare (List.map (fun (w, _) -> w.key) m.waves))
            < List.length m.waves);
          equal (list decoded)
            (List.map (as_decoded fmt) m.waves)
            (Thread_trace.waves g (m.trace ^ padding)));
      prop "a trace's clock passes through its markers, and their lines"
        (traces rdna) (fun (g, evs) ->
          let m = write (format_of g) evs in
          match (m.markers, Thread_trace.clock g (m.trace ^ padding)) with
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
      cut_short "a GFX9 trace cut short yields the waves of its whole packets"
        [ gfx9 ];
      cut_short "an RDNA trace cut short yields the waves of its whole packets"
        rdna;
      cases ~name:S.version "an empty trace has no waves and no clock"
        [ (9, 4, 3); (11, 0, 0); (12, 0, 1) ]
        (fun v ->
          equal (pair int bool) (0, true)
            ( List.length (Thread_trace.waves (gpu v) ""),
              Option.is_none (Thread_trace.clock (gpu v) "") ));
      test "two markers at one shader time give no clock" (fun () ->
          let s = { units = []; length = 0 } in
          put s ~units:16 (0x11 lor (3 lsl 7));
          put s ~units:12 (0x01 lor (1 lsl 9) lor (100 lsl 12));
          put s ~units:12 (0x01 lor (1 lsl 9) lor (200 lsl 12));
          is_none
            (Thread_trace.clock (gpu (11, 0, 0)) (bytes Rdna3 s ^ padding)));
    ]

let () = exit (run "rig_amd_abi.thread_trace" [ recording; decoding ])
