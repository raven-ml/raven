(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Shape
module Tlsf = Support_memory.Tlsf_allocator

let collect_bufs u = if op u = Op.Buffer then [ u ] else []

(* The buffers an argument reaches through a view. *)
let rec viewed u =
  match op u with
  | Op.Shrink | Op.Bitcast | Op.After ->
      let v = nth u 0 in
      collect_bufs v @ viewed v
  | _ -> []

(* What an entry of a schedule takes: the arguments of each call it runs, and
   the flag of each loop that stops on one, which the loop reads between its
   calls. *)
let rec uses si =
  match op si with
  | Op.End | Op.Backedge ->
      let body = nth si 0 in
      let calls = if op body = Op.Linear then src body else [ body ] in
      let flag = if op si = Op.Backedge then [ [ nth si 2 ] ] else [] in
      flag @ List.concat_map uses calls
  | _ -> [ List.tl (src si) ]

let can_plan held b =
  (not (Tbl.mem held b))
  && match device b with Some d -> not (is_disk_device d) | None -> true

let storage_ops = Op.Set.of_list Op.[ Buffer; Alloc; Param ]

(* The storage an entry takes, each once, in the order its calls take it. *)
let taken si =
  let seen = Tbl.create 8 and acc = ref [] in
  List.iter
    (fun args ->
      List.iter
        (fun a ->
          let b = storage_base a in
          if
            Op.Set.mem (op b) storage_ops
            && not (Tbl.mem seen b)
          then begin
            Tbl.replace seen b ();
            acc := b :: !acc
          end)
        args)
    (uses si);
  List.rev !acc

type lane = { mutable peak : int; tlsf : Tlsf.t }

(* Each call of the schedule [linear] at its place in the order calls run, a
   loop's once, at the steps [first] to [last]: the arguments it takes, those
   steps, and the loops around it, innermost first, by their number. A back
   edge's flag is taken at every step of its loop, by the loop. Also the steps
   of each loop. *)
let steps linear =
  let step = ref 0 and loops = ref 0 and taken = ref [] in
  let spans = Hashtbl.create 8 in
  let rec walk around si =
    match op si with
    | Op.End | Op.Backedge ->
        let id = !loops and first = !step in
        incr loops;
        let body = nth si 0 in
        let calls = if op body = Op.Linear then src body else [ body ] in
        List.iter (walk (id :: around)) calls;
        let last = !step - 1 in
        Hashtbl.replace spans id (first, last);
        if op si = Op.Backedge then
          taken := ([ nth si 2 ], first, last, around) :: !taken
    | _ ->
        taken := (List.tl (src si), !step, !step, around) :: !taken;
        incr step
  in
  List.iter (walk []) (src linear);
  (List.rev !taken, Hashtbl.find spans)

(* The steps a buffer taken at [uses] lives, each a use's steps and the loops
   around it: within one trip where every use lies in one loop, from the first
   use to the last, and otherwise across each loop a use lies in that not all
   do, from its first step to its last, as if that loop were one call. *)
let life spans uses =
  let paths = List.map (fun (_, _, around) -> List.rev around) uses in
  let rec common = function
    | (l :: _) :: _ as ps
      when List.for_all (function l' :: _ -> l' = l | [] -> false) ps ->
        1 + common (List.map List.tl ps)
    | _ -> 0
  in
  let depth = common paths in
  let span (first, last, _) path =
    match List.nth_opt path depth with
    | Some loop -> spans loop
    | None -> (first, last)
  in
  List.fold_left2
    (fun (lo, hi) u path ->
      let first, last = span u path in
      (min lo first, max hi last))
    (max_int, min_int) uses paths

let memory_plan_rewrite ?(held_bufs = []) linear =
  if Setting.value Setting.no_memory_planner then linear
  else begin
    let held = Tbl.create 16 in
    List.iter (fun b -> Tbl.replace held b ()) held_bufs;
    (* The uses of the buffers that can be planned, in order of first
       appearance. A call takes the buffers it passes and those it views: a
       store into a run of a buffer passes a view of it. *)
    let taken, spans = steps linear in
    let used = Tbl.create 64 and order = ref [] in
    List.iter
      (fun ((args, _, _, _) as use) ->
        List.iter
          (fun b ->
            match Tbl.find_opt used b with
            | Some us -> Tbl.replace used b (use :: us)
            | None ->
                Tbl.replace used b [ use ];
                order := b :: !order)
          (List.filter (can_plan held)
             (List.concat_map (fun a -> collect_bufs a @ viewed a) args)))
      taken;
    let bufs = List.rev !order in
    let first = Tbl.create 64 and last = Tbl.create 64 in
    List.iter
      (fun b ->
        let uses = List.map (fun (_, f, l, a) -> (f, l, a)) (Tbl.find used b) in
        let f, l = life spans uses in
        Tbl.replace first b f;
        Tbl.replace last b l)
      bufs;
    if List.is_empty bufs then linear
    else begin
      (* A device's copies and kernels plan in one lane: a lane of copies would
         keep reuse from ordering a copy against a kernel, which the device's
         order of its work makes moot on one device, and would add its peak to
         the kernels'. Across devices it would avoid a wait only in part. A
         buffer's bytes are free once its last call is: an arena is one
         allocation, and work that writes an allocation follows every earlier
         use of it. *)
      let key b = Option.get (device b) in
      let block_size = 256 in
      (* A buffer whose first byte lies [phase] bytes past a 16-byte boundary,
         which its kernels read it at, lies as far past its place. *)
      let phase b = match arg b with Param p -> p.phase | _ -> 0 in
      let size b = phase b + (max_numel b * element_size b) in
      let rounded b = Helpers.round_up (size b) block_size in
      let events =
        List.stable_sort
          (fun (t0, o0, _) (t1, o1, _) ->
            match Int.compare t0 t1 with 0 -> Bool.compare o0 o1 | c -> c)
          (List.map (fun b -> (Tbl.find first b, true, b)) bufs
          @ List.map
              (fun b -> (Tbl.find last b + 1, false, b))
              bufs)
      in
      let total_memory = 2 * List.fold_left (fun n b -> n + rounded b) 0 bufs in
      let lanes = ref [] in
      let lane k =
        match List.find_opt (fun (k', _) -> String.equal k k') !lanes with
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
             (fun acc (d, l) ->
               (l, new_buffer d (Helpers.round_up l.peak block_size) Dtype.Int8)
               :: acc)
             [] !lanes)
      in
      let replace_map =
        List.map
          (fun b ->
            let arena = List.assq (lane (key b)) arenas in
            let off = Tbl.find offsets b + phase b in
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
