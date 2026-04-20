(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Memory planning.

   Reduces peak memory by reusing buffers whose lifetimes don't overlap.
   Each schedule step lists the buffers it touches; we compute live ranges per
   base buffer, then either suballocate from a per-lane TLSF arena (when the
   device supports offset views) or recycle freed buffers from a pool keyed by
   (device, dtype, spec, nbytes).

   Copy and compute buffers live in separate lanes so freeing a copy buffer
   never forces a dependency between the copy and compute queues. *)

module B = Device.Buffer

let debug = Helpers.getenv "DEBUG" 0
let no_memory_planner = Helpers.getenv "NO_MEMORY_PLANNER" 0 <> 0

let round_up n align = (n + align - 1) / align * align
let blk = 0x1000

(* Lane key: (device, is_copy_lane). *)
type lane_key = string * int

(* [buffers] is a list of per-step buffer lists (one per schedule item).
   [copies] is the (dst, src) pairs from copy operations. Returns a hashtable
   mapping buffer ids to replacement buffers. Buffers absent from the table
   keep their original allocation. *)
let internal_memory_planner ?(copies = []) ?(ignore_checks = false)
    ?(debug_prefix = "") buffers =
  let assigned = Hashtbl.create 64 in
  if no_memory_planner then assigned
  else begin

    (* Live ranges *)

    let first = Hashtbl.create 64 in
    let last = Hashtbl.create 64 in
    let bases = Hashtbl.create 64 in
    let to_opt = Hashtbl.create 64 in
    List.iteri (fun step bufs ->
      List.iter (fun buf ->
        let base = B.base buf in
        let bid = B.base_id buf in
        if ignore_checks
           || not (B.is_allocated buf || B.is_allocated base
                  || B.uop_refcount buf > 0)
        then begin
          if not (Hashtbl.mem first bid) then begin
            Hashtbl.replace first bid step;
            Hashtbl.replace bases bid base
          end;
          Hashtbl.replace last bid step;
          Hashtbl.replace to_opt (B.id buf) buf
        end) bufs) buffers;

    (* Lane separation *)

    (* Copy buffers are held for an extra lifetime so their free is deferred
       past any compute work sitting between two copy steps. *)
    let copy_set = Hashtbl.create 16 in
    List.iter (fun (dst, src) ->
      Hashtbl.replace copy_set (B.base_id dst) ();
      Hashtbl.replace copy_set (B.base_id src) ()) copies;
    let is_copy bid = Hashtbl.mem copy_set bid in
    let lane_key bid =
      (B.device (Hashtbl.find bases bid), if is_copy bid then 1 else 0) in
    let hold bid =
      if is_copy bid then Hashtbl.find last bid - Hashtbl.find first bid + 1
      else 0 in

    (* Sorted alloc/free timeline *)

    (* Encoding: 0 = free, 1 = alloc. Sorting places frees before allocs at
       the same step so recycled buffers are immediately available. *)
    let events =
      Hashtbl.fold (fun bid _ acc ->
        ((Hashtbl.find first bid, 1), bid)
        :: ((Hashtbl.find last bid + 1 + hold bid, 0), bid)
        :: acc)
        bases []
      |> List.sort (fun (k1, _) (k2, _) -> compare k1 k2)
    in

    (* Allocate or reuse *)

    let total_memory =
      Hashtbl.fold (fun bid _ acc ->
        acc + round_up (B.nbytes (Hashtbl.find bases bid)) blk)
        bases 0
      * 2 (* 2x headroom for fragmentation *)
    in
    (* Per-lane TLSF arena for devices that support suballocation;
       maps lane_key to (high_water_mark, allocator). *)
    let global_planner : (lane_key, int * Tlsf.t) Hashtbl.t =
      Hashtbl.create 8 in
    let get_planner lk =
      match Hashtbl.find_opt global_planner lk with
      | Some p -> p
      | None ->
          let p =
            (0, Tlsf.create ~size:total_memory ~block_size:blk
                  ~lv2_cnt:32 ()) in
          Hashtbl.replace global_planner lk p; p in
    (* One template buffer per lane to extract the allocator from. *)
    let lane_template : (lane_key, B.t) Hashtbl.t = Hashtbl.create 8 in
    let replace : (int, B.t option * int option) Hashtbl.t =
      Hashtbl.create 64 in
    let pool = Hashtbl.create 64 in
    List.iter (fun ((_, is_alloc), bid) ->
      let base = Hashtbl.find bases bid in
      if B.supports_offset base then begin
        let lk = lane_key bid in
        if not (Hashtbl.mem lane_template lk) then
          Hashtbl.replace lane_template lk base;
        let max_sz, tlsf = get_planner lk in
        let off =
          if is_alloc = 1 then begin
            let off =
              Tlsf.alloc tlsf (round_up (B.nbytes base) blk) () in
            Hashtbl.replace replace bid (None, Some off);
            off
          end else begin
            let off = match snd (Hashtbl.find replace bid) with
              | Some o -> o | None -> assert false in
            Tlsf.free tlsf off;
            off
          end
        in
        Hashtbl.replace global_planner lk
          (max max_sz (off + B.nbytes base), tlsf)
      end else begin
        let key =
          (lane_key bid, B.dtype base, B.spec base, B.nbytes base) in
        if is_alloc = 1 then begin
          let repl = match Hashtbl.find_opt pool key with
            | Some (b :: rest) -> Hashtbl.replace pool key rest; b
            | _ -> base in
          Hashtbl.replace replace bid (Some repl, None)
        end else begin
          let repl =
            match fst (Hashtbl.find replace bid) with
            | Some b -> b | None -> assert false in
          let freed =
            match Hashtbl.find_opt pool key with
            | Some l -> l | None -> [] in
          Hashtbl.replace pool key (repl :: freed)
        end
      end) events;

    (* Global arena buffers *)

    (* One shared int8 buffer per lane for all suballocated regions. *)
    let global_bufs : (lane_key, B.t) Hashtbl.t = Hashtbl.create 8 in
    Hashtbl.iter (fun lk (sz, _) ->
      if sz > 0 then begin
        let template = Hashtbl.find lane_template lk in
        let gb =
          B.create ~device:(fst lk) ~size:(round_up sz blk)
            ~dtype:Tolk_ir.Dtype.int8 (B.allocator template) in
        Hashtbl.replace global_bufs lk gb
      end) global_planner;

    (* Resolve suballocated entries: None base → global arena buffer. *)
    let resolved = Hashtbl.create 64 in
    Hashtbl.iter (fun bid (repl_opt, off) ->
      let base_buf = match repl_opt with
        | Some b -> b
        | None -> Hashtbl.find global_bufs (lane_key bid) in
      Hashtbl.replace resolved bid (base_buf, off)) replace;

    (* Build replacement map *)

    (* Base buffers that got a different physical buffer. *)
    Hashtbl.iter (fun bid (repl, off) ->
      let base = Hashtbl.find bases bid in
      if B.id base <> B.id repl then
        Hashtbl.replace assigned (B.id base)
          (match off with
           | None -> repl
           | Some off ->
               B.view repl ~size:(B.size base)
                 ~dtype:(B.dtype base) ~offset:off))
      resolved;

    (* Sub-buffers: rebase onto the (possibly replaced) parent. *)
    Hashtbl.iter (fun _ buf ->
      if B.id buf <> B.base_id buf then begin
        let base = B.base buf in
        let pbuf =
          match Hashtbl.find_opt assigned (B.id base) with
          | Some b -> b | None -> base in
        Hashtbl.replace assigned (B.id buf)
          (B.view (B.base pbuf)
             ~size:(B.size buf) ~dtype:(B.dtype buf)
             ~offset:(B.offset pbuf + B.offset buf))
      end) to_opt;

    (* Debug *)

    if debug >= 1 then begin
      let seen_k = Hashtbl.create 16 in
      let seen_v = Hashtbl.create 16 in
      let omem = ref 0 and nmem = ref 0 in
      let nk = ref 0 and nv = ref 0 in
      Hashtbl.iter (fun buf_id new_buf ->
        (match Hashtbl.find_opt bases buf_id with
         | Some orig when not (Hashtbl.mem seen_k buf_id) ->
             Hashtbl.replace seen_k buf_id ();
             omem := !omem + B.nbytes orig;
             incr nk
         | _ -> ());
        let vid = B.base_id new_buf in
        if B.id new_buf = vid && not (Hashtbl.mem seen_v vid) then begin
          Hashtbl.replace seen_v vid ();
          nmem := !nmem + B.nbytes new_buf;
          incr nv
        end) assigned;
      Hashtbl.iter (fun _ gb ->
        let vid = B.id gb in
        if not (Hashtbl.mem seen_v vid) then begin
          Hashtbl.replace seen_v vid ();
          nmem := !nmem + B.nbytes gb;
          incr nv
        end) global_bufs;
      if !omem <> !nmem then
        Printf.printf
          "%smemory reduced from %.2f MB -> %.2f MB, %d -> %d bufs\n"
          debug_prefix
          (Float.of_int !omem /. 1e6) (Float.of_int !nmem /. 1e6) !nk !nv
    end;

    assigned
  end

let memory_planner schedule =
  let buffers =
    List.map (fun si ->
      List.filter_map Fun.id (Realize.Exec_item.bufs si))
      schedule
  in
  let copies =
    List.filter_map (fun si ->
      let is_copy = match Tolk_ir.Tensor.view (Realize.Exec_item.ast si) with
        | Tolk_ir.Tensor.Call { callee = Ref r; _ } ->
            (match Tolk_ir.Tensor.view r with
             | Tolk_ir.Tensor.Copy _ -> true | _ -> false)
        | _ -> false in
      if is_copy then
        match Realize.Exec_item.bufs si with
        | Some dst :: Some src :: _ -> Some (dst, src)
        | _ -> None
      else None)
      schedule
  in
  let assigned = internal_memory_planner ~copies buffers in
  List.map (fun si ->
    let new_bufs =
      List.map (function
        | None -> None
        | Some buf ->
            Some (match Hashtbl.find_opt assigned (B.id buf) with
                  | Some repl -> repl | None -> buf))
        (Realize.Exec_item.bufs si)
    in
    Realize.Exec_item.make
      ~ast:(Realize.Exec_item.ast si)
      ~bufs:new_bufs
      ~var_vals:(Realize.Exec_item.var_vals si) ())
    schedule
