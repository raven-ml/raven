(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
module Tlsf = Support_memory.Tlsf_allocator

let rec collect_bufs u =
  match op u with
  | Op.Buffer -> [ u ]
  | Op.Mselect | Op.Mstack -> List.concat_map collect_bufs (src u)
  | _ -> []

(* The buffers an argument reaches through a view. *)
let rec viewed u =
  match op u with
  | Op.Shrink | Op.Bitcast | Op.After ->
      let v = nth u 0 in
      collect_bufs v @ viewed v
  | Op.Mselect | Op.Mstack -> List.concat_map viewed (src u)
  | _ -> []

(* What an entry of a schedule takes: the arguments of each call it runs, a
   copy's apart, and the flag of each loop that stops on one, which the loop
   reads between its calls. *)
let rec uses si =
  match op si with
  | Op.End | Op.Backedge ->
      let body = nth si 0 in
      let calls = if op body = Op.Linear then src body else [ body ] in
      let flag =
        if op si = Op.Backedge then [ (`Kernel, [ nth si 2 ]) ] else []
      in
      flag @ List.concat_map uses calls
  | _ ->
      let kind = if op (nth si 0) = Op.Store then `Copy else `Kernel in
      [ (kind, List.tl (src si)) ]

let can_plan held b =
  (not (Tbl.mem held b))
  && match device b with Some d -> not (is_disk_device d) | None -> true

type lane = { mutable peak : int; tlsf : Tlsf.t }

let memory_plan_rewrite ?(held_bufs = []) linear =
  if Setting.value Setting.no_memory_planner then linear
  else begin
    let held = Tbl.create 16 in
    List.iter (fun b -> Tbl.replace held b ()) held_bufs;
    (* The lifetimes of the buffers that can be planned, in order of first
       appearance. *)
    let first = Tbl.create 64 and last = Tbl.create 64 in
    let order = ref [] and copy_bufs = Tbl.create 16 in
    (* A buffer a call reaches through a view stays where it is: the plan does
       not see the calls that reach it that way. *)
    let through_views = Tbl.create 16 in
    List.iteri
      (fun i si ->
        List.iter
          (fun (kind, args) ->
            List.iter
              (fun b -> Tbl.replace through_views b ())
              (List.concat_map viewed args);
            let call_bufs =
              List.filter (can_plan held) (List.concat_map collect_bufs args)
            in
            List.iter
              (fun b ->
                if not (Tbl.mem first b) then begin
                  Tbl.replace first b i;
                  order := b :: !order
                end;
                Tbl.replace last b i)
              call_bufs;
            if kind = `Copy then
              List.iter (fun b -> Tbl.replace copy_bufs b ()) call_bufs)
          (uses si))
      (src linear);
    let bufs =
      List.filter (fun b -> not (Tbl.mem through_views b)) (List.rev !order)
    in
    if List.is_empty bufs then linear
    else begin
      (* Copies and kernels plan in separate lanes, lest reuse add a dependency
         from a copy to a kernel to a copy. *)
      let key b = (Option.get (device b), Tbl.mem copy_bufs b) in
      let buf_hold b =
        if Tbl.mem copy_bufs b then Tbl.find last b - Tbl.find first b + 1
        else 0
      in
      let block_size = 256 in
      let size b = max_numel b * element_size b in
      let rounded b = Helpers.round_up (size b) block_size in
      let events =
        List.stable_sort
          (fun (t0, o0, _) (t1, o1, _) ->
            match Int.compare t0 t1 with 0 -> Bool.compare o0 o1 | c -> c)
          (List.map (fun b -> (Tbl.find first b, true, b)) bufs
          @ List.map
              (fun b -> (Tbl.find last b + 1 + buf_hold b, false, b))
              bufs)
      in
      let total_memory = 2 * List.fold_left (fun n b -> n + rounded b) 0 bufs in
      let lanes = ref [] in
      let lane k =
        let same (d, c) = equal_device d (fst k) && c = snd k in
        match List.find_opt (fun (k', _) -> same k') !lanes with
        | Some (_, l) -> l
        | None ->
            let l =
              {
                peak = 0;
                tlsf = Tlsf.create ~block_size ~lv2_cnt:32 total_memory;
              }
            in
            lanes := !lanes @ [ (k, l) ];
            l
      in
      let offsets = Tbl.create 64 in
      List.iter
        (fun (_, is_open, b) ->
          let l = lane (key b) in
          if is_open then
            match Tlsf.alloc l.tlsf (rounded b) with
            | Some off -> Tbl.replace offsets b off
            | None -> invalid_arg "the memory plan outgrew its arena"
          else Tlsf.free l.tlsf (Tbl.find offsets b);
          l.peak <- max l.peak (Tbl.find offsets b + size b))
        events;
      (* Each buffer becomes a view of its lane's arena, at its offset. *)
      let arenas =
        List.rev
          (List.fold_left
             (fun acc ((d, _), l) ->
               (l, new_buffer d (Helpers.round_up l.peak block_size) Dtype.Int8)
               :: acc)
             [] !lanes)
      in
      let replace_map =
        List.map
          (fun b ->
            let arena = List.assq (lane (key b)) arenas in
            let off = Tbl.find offsets b in
            ( b,
              bitcast
                (shrink arena [ Some (Int off, Int (off + nbytes b)) ])
                (dtype b) ))
          bufs
      in
      if Setting.value Setting.debug >= 1 then begin
        let mb n = float_of_int n /. 1e6 in
        let omem = mb (List.fold_left (fun n b -> n + rounded b) 0 bufs)
        and nmem =
          mb
            (List.fold_left
               (fun n (_, l) -> n + Helpers.round_up l.peak block_size)
               0 !lanes)
        in
        if omem <> nmem then
          Printf.printf
            "memory reduced from %.2f MB -> %.2f MB, %d -> %d bufs\n%!" omem
            nmem (List.length bufs) (List.length arenas)
      end;
      substitute ~calls:Skip ~pass:Once linear replace_map
    end
  end
