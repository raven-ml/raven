(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Shape

(* Load valid simplification *)

(* A load in the index runs whatever [valid], so its own index cannot assume it:
   a gather through a pad would read its indices outside the pad. Each load
   stands as a variable of its bounds while [valid] simplifies the arithmetic
   around it. *)
let simplify_valid_load buf start_idx valid =
  let loads =
    List.filter
      (fun u -> op u = Op.Load)
      (Nodes.to_list (backward_slice ~calls:Skip start_idx))
  in
  let held =
    List.mapi
      (fun i l ->
        let name = "load" ^ string_of_int i in
        (l, variable ~dtype:(dtype l) name (vmin l) (vmax l)))
      loads
  in
  let hold u = substitute ~calls:Skip ~pass:Fixed_point u held in
  let idx = Symbolic.uop_given_valid (hold valid) (hold start_idx) in
  let idx =
    substitute ~calls:Skip ~pass:Fixed_point idx
      (List.map (fun (l, v) -> (v, l)) held)
  in
  if idx == start_idx || idx == simplify start_idx then None
  else Some (index buf [ Shape.valid idx valid ])

let indexing_simplify =
  Pattern_matcher.v
    (fun () -> [
      Pattern_matcher.rule
        (Upat.op Op.Index ~src:[ Upat.var "buf"; Shape.invalid_gate ])
        (fun m -> simplify_valid_load (m "buf") (m "x") (m "cond"));
    ])

(* Memory coalescing *)

(* The accesses that may merge: of one operation, buffer, base index, gate,
   argument and depth. The base is [None] for constant indices. *)
type key = {
  op : Op.t;
  buf : t;
  base : t option;
  valid : t;
  arg : arg;
  depth : int * int;
}

let same k0 k1 =
  k0.op = k1.op && k0.buf == k1.buf
  && Option.equal ( == ) k0.base k1.base
  && k0.valid == k1.valid && equal_arg k0.arg k1.arg && k0.depth = k1.depth

let int_value u =
  match value u with
  | #Dtype.value as v -> Dtype.Value.to_z v
  | `Invalid -> invalid_arg "an invalid index"

(* The runs of consecutive integers of the sorted [l]. *)
let runs l =
  let rec go acc run = function
    | [] -> List.rev (List.rev run :: acc)
    | x :: rest -> (
        match run with
        | prev :: _ when Bigint.equal x (Bigint.succ prev) -> go acc (x :: run) rest
        | [] -> go acc [ x ] rest
        | _ -> go (List.rev run :: acc) [ x ] rest)
  in
  match l with [] -> [] | _ -> go [] [] l

let memory_coalescing sink (r : Renderer.t) =
  if Setting.value Setting.dmc then sink
  else begin
    (* Collect, each group's offsets in the order first seen. *)
    let groups = ref [] in
    let add key offset u =
      match List.find_opt (fun (k, _) -> same k key) !groups with
      | Some (_, offsets) -> (
          match List.find_opt (fun (o, _) -> Bigint.equal o offset) !offsets with
          | Some (_, us) -> us := u :: !us
          | None -> offsets := (offset, ref [ u ]) :: !offsets)
      | None -> groups := (key, ref [ (offset, ref [ u ]) ]) :: !groups
    in
    (* The depth of a node is the greatest pair, in lexicographic order, of
       counts of global and of local stores on a chain it waits on, itself
       included. Accesses merge only at one depth, a global access counting
       global stores alone, so that no merge closes a cycle: a merged store
       would otherwise wait on a store that waits on one of its parts. *)
    (* Most nodes wait on no store: the table holds the others. *)
    let depth = Tbl.create 16 in
    let depth_of u =
      match Tbl.find depth u with d -> d | exception Not_found -> (0, 0)
    in
    let later ((g, l) as d) ((g', l') as d') =
      if g > g' || (g = g' && l >= l') then d else d'
    in
    (* The loads of 16-bit floats read other than as the value a store puts
       in local memory: those copied there may merge further. *)
    let sixteen_bits dt = Dtype.mem dt Dtype.[ Float16; Bfloat16 ] in
    let read = Tbl.create 16 in
    let note u =
      List.iteri
        (fun i s ->
          let copy =
            op u = Op.Store && i = 1 && addrspace (nth u 0) = Some Dtype.Local
          in
          if op s = Op.Load && sixteen_bits (dtype s) && not copy then
            Tbl.replace read s ())
        (src u)
    in
    (* The loads of bytes a computation reads one lane at a time. Clang packs
       bytes that feed one lane each itself. Merged, such lanes reach LLVM's
       load combine, which folds an or of bytes in descending addresses, read
       from vector loads, into one byte-swapped load at the address of the
       first byte's vector: where that byte is not its vector's first, as in a
       value straddling two vector loads, it loads from the wrong address.
       Only bytes reverse so: a wider element keeps its own bytes ascending,
       and the combine folds no such order. *)
    let apart = Tbl.create 16 in
    let read_apart u =
      if Op.Set.mem (op u) Op.Set.elementwise then
        List.iter
          (fun s ->
            if op s = Op.Load && Dtype.mem (dtype s) Dtype.[ Int8; Uint8 ] then
              Tbl.replace apart s ())
          (src u)
    in
    List.iter
      (fun u ->
        if r.half8 then note u;
        read_apart u;
        let waits =
          List.fold_left
            (fun d s -> later d (depth_of s))
            (0, 0) (src_without_body u)
        in
        let d =
          if op u <> Op.Store then waits
          else
            let space = addrspace (nth u 0) in
            let count a = if space = Some a then 1 else 0 in
            (fst waits + count Dtype.Global, snd waits + count Dtype.Local)
        in
        if fst d > 0 || snd d > 0 then Tbl.replace depth u d;
        if op u = Op.Load || op u = Op.Store then begin
          if List.length (src u) <> if op u = Op.Store then 2 else 1 then
            invalid_arg "memory coalescing does not support gated loads/stores";
          let target = nth u 0 in
          if op target <> Op.Index then
            invalid_arg
              (Format.asprintf "memory coalescing should be on INDEX, not %a"
                 Op.pp (op target));
          match src target with
          | [ buf; idx_u ] ->
              let volatile =
                let b = buf_uop buf in
                op b = Op.Param
                && match arg b with Param p -> p.volatile | _ -> false
              in
              let idx = get_idx idx_u and valid = get_valid idx_u in
              (* Volatile accesses never merge. *)
              if
                addrspace buf <> Some Dtype.Reg
                && (not volatile)
                && not (is_invalid idx)
              then begin
                let base, offset =
                  match (op idx, src idx) with
                  | Op.Add, [ b; c ] when is_literal c -> (Some b, int_value c)
                  | Op.Add, [ c; b ] when is_literal c -> (Some b, int_value c)
                  | _ when is_literal idx -> (None, int_value idx)
                  | _ -> (Some idx, Bigint.zero)
                in
                let depth =
                  if addrspace buf = Some Dtype.Global then (fst waits, 0)
                  else waits
                in
                add { op = op u; buf; base; valid; arg = arg u; depth } offset u
              end
          | _ -> invalid_arg "an index of memory needs a buffer and one index"
        end)
      (toposort ~calls:Skip sink);
    (* A load of global memory read only by a store into local memory, and
       that store, copy an element into shared memory. *)
    let copied u =
      op u = Op.Load
      && addrspace (nth u 0) = Some Dtype.Global
      && not (Tbl.mem read u)
    in
    let copies u =
      if op u = Op.Load then copied u
      else addrspace (nth u 0) = Some Dtype.Local && copied (nth u 1)
    in
    (* Build the replacements. *)
    let replacements = ref [] in
    List.iter
      (fun (key, offsets) ->
        let offsets =
          List.rev_map (fun (o, us) -> (o, List.rev !us)) !offsets
        in
        let accesses o = List.assoc o offsets in
        let dt = dtype key.buf and size = Dtype.itemsize (dtype key.buf) in
        let merged = Dtype.([ Float32; Float16; Int32; Uint32 ] @ fp8s) in
        (* Where the target's vector registers are known, an access of these
           types, of doubles or of bytes fills up to one register: Clang packs
           a kernel's values into registers as wide as the stores it reads them
           from, so a store of four floats leaves half of a 32-byte register
           empty, and loads as wide feed it whole. Bytes loaded one at a time,
           as a quantised matrix's codes, are inserted into a register one at a
           time. *)
        (* Doubles computed lane by lane are stored as tinygrad stores them,
           one by one: merged, Clang packs the computations behind them into
           vectors, and rune's float64 normal sampler, whose lanes hash and
           convert 32- and 64-bit integers, ran twice as slow. A load, as an
           accumulator's read, is already a value in memory. Bytes are stored
           so too: only their loads were measured. *)
        let stores_packed () =
          List.for_all
            (fun (_, us) -> List.for_all (fun u -> op (nth u 1) = Op.Load) us)
            offsets
        in
        let read_alone () =
          List.exists (fun (_, us) -> List.exists (Tbl.mem apart) us) offsets
        in
        let width =
          match r.registers () with
          | Some { bytes; _ }
            when r.supports_float4
                 && (key.op <> Op.Store
                    || (not (Dtype.mem dt Dtype.[ Float64; Int8; Uint8 ]))
                    || stores_packed ()) ->
              Some bytes
          | _ -> None
        in
        let lengths =
          match width with
          | Some bytes
            when Dtype.mem dt Dtype.(Float64 :: Int8 :: Uint8 :: merged)
                 && not (read_alone ()) ->
              let rec halves l = if l < 2 then [ 1 ] else l :: halves (l / 2) in
              halves (bytes / size)
          | _ ->
              (* A GPU merges bfloat16 accesses as float16's: a tensor core's
                 operands, loaded two bytes at a time, made an RTX 5000 Ada's
                 bfloat16 products 1.5 times slower than its float16 ones. *)
              let gpu_bfloat16 =
                Dtype.equal dt Dtype.Bfloat16 && r.target.device <> "CPU"
              in
              (* 16-bit floats copied into shared memory merge to 16 bytes
                 where the target writes vectors of eight: a copy of 16 bytes
                 is one instruction, as a tensor core's operands staged there
                 take. Elsewhere 16 bytes gained as often as they lost: an
                 attention's product of probabilities by bfloat16 values ran
                 1.4 times slower, ptxas issuing fewer of its loads ahead. *)
              let copy () =
                sixteen_bits dt && r.half8
                && List.for_all (fun (_, us) -> List.for_all copies us) offsets
              in
              (if (Dtype.mem dt merged || gpu_bfloat16) && r.supports_float4
               then if copy () then [ 8; 4; 2 ] else [ 4; 2 ]
               else [])
              @ [ 1 ]
        in
        (* Elements before the buffer's first from the boundary behind it, a
           multiple of its alignment: an access of [l] elements is aligned where
           the element count from that boundary is a multiple of [l], and is no
           wider than the alignment. A vector on the CPU device is aligned to
           its element, so it loads from any element: storage there is taken to
           start on a boundary of its widest access, as storage of no known
           start is. *)
        let phase, align =
          match (arg (buf_uop key.buf), width) with
          | Param p, _ when r.target.device <> "CPU" -> (p.phase, p.align)
          | _, Some bytes when r.target.device = "CPU" -> (0, bytes)
          | _ -> (0, 16)
        in
        let lead = Bigint.of_int (phase / size) in
        (* An access of another width than the storage's elements, through a
           bitcast, may start between its own elements: it merges nothing. *)
        let lengths =
          if phase mod size = 0 then
            List.filter (fun l -> l = 1 || l * size <= align) lengths
          else [ 1 ]
        in
        let at first =
          match key.base with
          | Some b -> O.(b + const (`Int first))
          | None -> const (`Int first)
        in
        let sorted = List.sort Bigint.compare (List.map fst offsets) in
        List.iter
          (fun run ->
            let rec take grp =
              match grp with
              | [] -> ()
              | first :: _ ->
                  let length =
                    List.find
                      (fun l ->
                        l <= List.length grp
                        && Option.is_some
                             (divides (at (Bigint.add first lead)) (Bigint.of_int l)))
                      lengths
                  in
                  let now = List.filteri (fun i _ -> i < length) grp
                  and rest = List.filteri (fun i _ -> i >= length) grp in
                  if
                    key.op = Op.Store
                    && List.exists
                         (fun o -> List.compare_length_with (accesses o) 1 <> 0)
                         now
                  then invalid_arg "attempting multiple stores";
                  (* A run of one is left as it is. A run starts at its first
                     access's index, its gate included. *)
                  (if length > 1 then
                     let offset = nth (nth (List.hd (accesses first)) 0) 1 in
                     let idx =
                       v Op.Shrink ~src:[ key.buf; offset; int length ]
                     in
                     if key.op = Op.Store then begin
                       let stores = List.map accesses now in
                       let datas =
                         List.map (fun us -> nth (List.hd us) 1) stores
                       in
                       let st = store idx (stack datas) in
                       List.iter
                         (fun us ->
                           replacements := (List.hd us, st) :: !replacements)
                         stores
                     end
                     else
                       let ld = v Op.Load ~src:[ idx ] ~arg:key.arg in
                       List.iteri
                         (fun i g ->
                           List.iter
                             (fun oo ->
                               let value = index ld [ int i ] in
                               replacements := (oo, value) :: !replacements)
                             (accesses g))
                         now);
                  take rest
            in
            take run)
          (runs sorted))
      (List.rev !groups);
    substitute ~calls:Skip ~pass:Fixed_point sink !replacements
  end
