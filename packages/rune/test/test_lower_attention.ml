(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Attention lowered for each GPU target and rendered on the host, without a
   device or its toolchain: every dtype a target has takes a kernel of a
   product unit, an FFMA micro-tile's products sum in K's order
   (lower_attention.mli), and the forward's kernels and the derivative's on
   a tensor core use workgroup memory soundly. *)

open Windtrap
module O = Tolk.Ops

let gpu_targets =
  List.map
    (fun t -> Result.get_ok (Tolk.Helpers.Target.of_string t))
    [
      "CUDA:CUDA:sm_89"; "METAL:METAL:Apple9"; "AMD:HIP:gfx1201";
      "AMD:HIP:gfx942"; "CUDA:CUDA:sm_75";
    ]

let target_id (t : Tolk.Helpers.Target.t) = t.device ^ ":" ^ t.arch
let cuda = List.hd gpu_targets

(* A query's dtype and its accumulator's. *)
type dtypes = D : (float, 'a) Nx.dtype * (float, 'b) Nx.dtype -> dtypes

let dtypes_id (D (dt, acc)) = Nx.Dtype.name dt ^ " in " ^ Nx.Dtype.name acc

let supported target (D (dt, acc)) =
  let ds = Tolk.Renderer.supported_dtypes (Tolk.Cstyle.without_compiler target) in
  let has dt = List.mem (Rune.Lower_arith.dtype (Nx_array.Dtype.Any dt)) ds in
  has dt && has acc

(* The kernel named [attention] of the forward of [batches] (default 2)
   batches of 2 heads of 128 query rows and keys of [d] dimensions, unmasked
   or [causal], and its sink. Few batches split each row's keys, a tile a
   run. *)
let forward ?(batches = 2) ?(causal = false) target (D (dt, acc)) d =
  let shape = [| batches; 2; 128; d |] in
  let f () =
    let z () = Nx.full dt shape 0.5 in
    let q = z () and k = z () and v = z () in
    let mask = if causal then Some Nx.causal else None in
    Nx.cast Nx.float32 (fst (Nx.attention ?mask ~acc ~kv:(Dense { k; v }) q))
  in
  match List.filter (fun (n, _, _) -> n = "attention") (Lowered.sources ~target f) with
  | [ (_, ast, _) ] -> ast
  | ks -> failf "%d kernels named attention" (List.length ks)

let params ast =
  List.length
    (List.filter (fun u -> O.op u = Tolk.Op.Param) (O.toposort ~calls:Skip ast))

(* The linearized kernel [ast] for [target]'s renderer. *)
let linearized target ast =
  O.src (O.nth (Tolk.Codegen.linearize ast (Tolk.Cstyle.without_compiler target)) 1)

(* Every pair of a query's dtype and an accumulator's that a GPU target has
   takes a kernel of a product unit, at every head dimension: its products
   run on a tensor core or as fused multiply-adds ({!Tolk.Op.Wmma},
   {!Tolk.Op.Mulacc}), which the SIMT kernel's do not, and it reads [q], [k]
   and [v] and writes its two results, with none of the SIMT kernel's
   scratch. *)
let totality () =
  List.iter
    (fun target ->
      List.iter
        (fun ds ->
          if supported target ds then
            List.iter
              (fun d ->
                let msg = Printf.sprintf "%s %s d%d" (target_id target) (dtypes_id ds) d in
                let ast = forward target ds d in
                let unit u = List.mem (O.op u) Tolk.Op.[ Wmma; Mulacc ] in
                is_true ~msg (List.exists unit (linearized target ast));
                equal ~msg Testable.int 5 (params ast))
              [ 1; 36; 110; 160; 1024; 2048 ])
        [
          D (Nx.float16, Nx.float32); D (Nx.bfloat16, Nx.float32);
          D (Nx.float16, Nx.float64); D (Nx.bfloat16, Nx.float64);
          D (Nx.float32, Nx.float32); D (Nx.float32, Nx.float64);
          D (Nx.float64, Nx.float64);
        ])
    gpu_targets

(* An FFMA micro-tile's step adds its products in K's order: each chain of
   multiply-adds of the forward's linearized kernel takes its [A] operand,
   four rows' dimensions or a row's keys, from the first, innermost first,
   onto the accumulator the steps' loop carries, a step a trip
   ({!Tolk.Op.Mulacc}). A term is a lane of one vector, as float32 loads
   them, or one scalar at an offset from one address, as float64 does; its
   place is the lane or the offset from the first term's. *)
let key_order (target, ds, causal, d) =
  let uops = linearized target (forward ~causal target ds d) in
  let mulacc u = O.op u = Tolk.Op.Mulacc in
  let inner = List.filter_map (fun u -> if mulacc u then Some (O.nth u 2) else None) uops in
  let heads = List.filter (fun u -> mulacc u && not (List.memq u inner)) uops in
  let rec terms u acc = if mulacc u then terms (O.nth u 2) (O.nth u 0 :: acc) else (u, acc) in
  let const u =
    match (O.vmin u, O.vmax u) with
    | `Int lo, `Int hi when Tolk.Bigint.equal lo hi -> Some (Tolk.Bigint.to_int lo)
    | _ -> None
  in
  let rec place a =
    match O.op a with
    | Tolk.Op.Cast -> place (O.nth a 0)
    | Tolk.Op.Index when O.op (O.nth a 0) = Tolk.Op.Load ->
        Option.map (fun j -> ([ O.nth a 0 ], j)) (const (O.nth a 1))
    | Tolk.Op.Load when O.op (O.nth a 0) = Tolk.Op.Index ->
        let ix = O.nth a 0 in
        let i = O.nth ix 1 in
        let at, j =
          match O.op i with
          | Tolk.Op.Add -> (
              match const (O.nth i 1) with Some j -> (O.nth i 0, j) | None -> (i, 0))
          | _ -> (i, 0)
        in
        Some ([ O.nth ix 0; at ], j)
    | _ -> None
  in
  is_true ~msg:"chains of multiply-adds" (heads <> []);
  List.iter
    (fun h ->
      let base, terms = terms h [] in
      equal ~msg:"onto a carried accumulator" Testable.bool true
        (O.op base = Tolk.Op.Load);
      match List.map place terms with
      | Some (at0, j0) :: _ as places ->
          List.iteri
            (fun i p ->
              match p with
              | Some (at, j) ->
                  equal ~msg:"place" Testable.int i (j - j0);
                  is_true ~msg:"of one operand" (List.for_all2 ( == ) at at0)
              | None -> failf "a term not a vector's lane or an offset's scalar")
            places;
          if List.length at0 = 1 then equal ~msg:"from the first lane" Testable.int 0 j0
      | _ -> failf "a term not a vector's lane or an offset's scalar")
    heads

(* The batches of an FFMA forward the workgroup-memory laws render: 128,
   which run each row's tiles of keys in one workgroup, a loop of tiles. *)
let batches = 128

(* The workgroup memory of a kernel of attention's own

   Workgroup 0's threads run its linearized program with every loop at its
   first trip: [stores target ast] is each store into workgroup memory, by
   the barriers before it, its buffer and, at a thread [t], the addresses it
   writes: none where it does not run, one for an index, and a vector's
   for a shrink [(buffer, start, length)]. *)

type store = { span : int; buf : O.t; at : (string * int) list -> int list }

let value u =
  match (O.vmin u, O.vmax u) with
  | `Int lo, `Int hi when Tolk.Bigint.equal lo hi -> Some (Tolk.Bigint.to_int lo)
  | `Bool lo, `Bool hi when lo = hi -> Some (Bool.to_int lo)
  | _ -> None

let stores target ast =
  let uops = linearized target ast in
  let ranges = List.filter (fun u -> O.op u = Tolk.Op.Range) uops in
  let specials = List.filter (fun u -> O.op u = Tolk.Op.Special) uops in
  let at_thread t u =
    let zero r = O.const ~dtype:(O.dtype r) (`Int Tolk.Bigint.zero) in
    let lane s =
      match O.arg s with
      | String n -> O.const ~dtype:(O.dtype s) (`Int (Tolk.Bigint.of_int (Option.value (List.assoc_opt n t) ~default:0)))
      | _ -> zero s
    in
    let subs = List.map (fun r -> (r, zero r)) ranges @ List.map (fun s -> (s, lane s)) specials in
    value (Tolk.Shape.simplify (O.substitute ~calls:Skip ~pass:Fixed_point u subs))
  in
  let local u = O.addrspace u = Some Tolk.Dtype.Local in
  let rec walk span gates acc = function
    | [] -> List.rev acc
    | u :: rest -> (
        match O.op u with
        | Tolk.Op.Barrier -> walk (span + 1) gates acc rest
        | Tolk.Op.If -> walk span (O.nth u 0 :: gates) acc rest
        | Tolk.Op.Endif -> walk span (List.tl gates) acc rest
        | Tolk.Op.Store when local (O.nth u 0) ->
            let ix = O.nth u 0 in
            let n, gated =
              if O.op ix = Tolk.Op.Shrink then (Option.get (value (O.nth ix 2)), 3)
              else (1, 2)
            in
            let all = gates @ List.filteri (fun i _ -> i >= gated) (O.src ix) in
            let at t =
              match at_thread t (O.nth ix 1) with
              | Some a when List.for_all (fun g -> at_thread t g = Some 1) all ->
                  List.init n (( + ) a)
              | _ -> []
            in
            walk span gates ({ span; buf = O.buf_uop ix; at } :: acc) rest
        | _ -> walk span gates acc rest)
  in
  (walk 0 [] [] uops, specials)

(* The threads of workgroup 0: each of its lanes' and warps' values. *)
let threads specials =
  let locals =
    List.filter_map
      (fun s ->
        match (O.arg s, O.vmax s) with
        | String n, `Int z when String.starts_with ~prefix:"lidx" n ->
            Some (n, Tolk.Bigint.to_int z + 1)
        | _ -> None)
      specials
  in
  List.fold_left
    (fun ts (n, k) -> List.concat_map (fun t -> List.init k (fun i -> (n, i) :: t)) ts)
    [ [] ] locals

(* A kernel's threads store into distinct addresses of workgroup memory
   between two barriers: a copy of a row into workgroup memory stays within
   its row and the row's padding, whatever the padding the unit takes, and
   no two threads copy one element. *)
let no_races target ast =
  let ss, specials = stores target ast in
  is_true ~msg:"stores into workgroup memory" (ss <> []);
  let seen = Hashtbl.create 4096 in
  List.iter
    (fun t ->
      List.iter
        (fun s ->
          List.iter
            (fun a ->
              let key = (s.span, O.hash s.buf, a) in
              match Hashtbl.find_opt seen key with
              | Some t' when t' <> t ->
                  failf "two threads store into address %d of a buffer between barriers %d" a s.span
              | _ -> Hashtbl.replace seen key t)
            (s.at t))
        ss)
    (threads specials)

(* A kernel's workgroup buffer that the tiles' loop reads is written in that
   loop, at each tile: every such buffer holds a tile's rows or keys, or
   shares the memory of one that does, as the FFMA forward's chunk of [Q]
   shares the values', which each tile writes, so that a copy made once
   before the loop would be read as values from the second tile on. *)
let in_the_tiles_loop target ast =
  let uops = linearized target ast in
  let local u = O.addrspace u = Some Tolk.Dtype.Local in
  let tile = List.find (fun u -> O.op u = Tolk.Op.Range) uops in
  is_true ~msg:"a loop of tiles"
    (match O.vmax (O.nth tile 0) with `Int z -> Tolk.Bigint.to_int z > 1 | _ -> false);
  let inside = ref false and read = Hashtbl.create 8 and written = ref [] in
  List.iter
    (fun u ->
      if u == tile then inside := true
      else if O.op u = Tolk.Op.End && List.memq tile (O.src u) then inside := false
      else
        match O.op u with
        | Tolk.Op.Load when local (O.nth u 0) && !inside ->
            Hashtbl.replace read (O.hash (O.buf_uop (O.nth u 0))) ()
        | Tolk.Op.Store when local (O.nth u 0) ->
            written := (O.hash (O.buf_uop (O.nth u 0)), !inside) :: !written
        | _ -> ())
    uops;
  is_true ~msg:"buffers the tiles' loop reads" (Hashtbl.length read > 0);
  List.iter
    (fun (b, inside) ->
      if Hashtbl.mem read b then
        is_true ~msg:"a store of a buffer the tiles' loop reads is in that loop" inside)
    !written

(* FFMA's pairs and shapes: one chunk and several, chunks of 1, 20 and 28
   float64 columns, under no mask and a causal one. *)
let ffma_cases =
  List.concat_map
    (fun ds ->
      List.concat_map
        (fun d -> [ (cuda, ds, false, d); (cuda, ds, true, d) ])
        [ 1; 36; 64; 110 ])
    [
      D (Nx.float32, Nx.float32); D (Nx.float64, Nx.float64);
      D (Nx.float32, Nx.float64); D (Nx.bfloat16, Nx.float64);
    ]

(* The derivative's kernels of attention's own, [q̄]'s and [k̄]'s and [v̄]'s,
   of a bfloat16 call accumulated in float32 of 2 batches of 2 heads of 128
   rows and keys and values of [d] dimensions, unmasked or [causal], on
   [target]'s tensor core, rendered as a compiled gradient takes them
   ({!Lowered.derivative_sources}); [[]] where its derivative takes its
   default. *)
let derivative target ~causal d =
  let shape = [| 2; 2; 128; d |] in
  let f () =
    let z () = Nx.full Nx.bfloat16 shape 0.5 in
    let q = z () and k = z () and v = z () in
    let mask = if causal then Some Nx.causal else None in
    Nx.cast Nx.float32 (fst (Nx.attention ?mask ~acc:Nx.float32 ~kv:(Dense { k; v }) q))
  in
  match Lowered.derivative_sources ~target f with
  | [ Some ks ] ->
      List.filter_map
        (fun (n, ast, _) ->
          if String.starts_with ~prefix:"attention_d" n && n <> "attention_delta"
          then Some (n, ast)
          else None)
        ks
  | [ None ] -> []
  | l -> failf "%d calls of attention" (List.length l)

(* The derivative's rows: bfloat16 at d36 to d160, plain and causal, on each
   GPU target whose tensor core takes them; CUDA's must. *)
let derivative_cases =
  List.concat_map
    (fun t -> List.concat_map (fun d -> [ (t, false, d); (t, true, d) ]) [ 36; 64; 110; 160 ])
    (List.filter (fun (t : Tolk.Helpers.Target.t) -> t.arch <> "sm_75") gpu_targets)

let derivative_id (t, causal, d) =
  Printf.sprintf "%s bfloat16 d%d%s" (target_id t) d (if causal then " causal" else "")

(* [on_derivative law] holds [law] on each of a case's derivative kernels,
   [q̄]'s and [k̄]'s and [v̄]'s. *)
let on_derivative law (t, causal, d) =
  match derivative t ~causal d with
  | [] -> is_true ~msg:"CUDA's tensor core takes the derivative" (t != cuda)
  | ks ->
      is_true ~msg:"q̄'s kernel" (List.exists (fun (n, _) -> n = "attention_dq") ks);
      is_true ~msg:"k̄'s and v̄'s kernels"
        (List.exists (fun (n, _) -> String.starts_with ~prefix:"attention_dkdv" n) ks);
      List.iter (fun (_, ast) -> law t ast) ks

let ffma_id (t, ds, causal, d) =
  Printf.sprintf "%s %s d%d%s" (target_id t) (dtypes_id ds) d
    (if causal then " causal" else "")

let () =
  exit
    (run "Lower_attention"
       [
         group "a GPU's dtypes"
           [ test "each take a kernel of a product unit" totality ];
         group "an FFMA micro-tile"
           [
             cases "sums its products in K's order"
               ~id:(fun (t, ds, causal, d) ->
                 Printf.sprintf "%s %s d%d%s" (target_id t) (dtypes_id ds) d
                   (if causal then " causal" else ""))
               (List.concat_map
                  (fun t ->
                    List.concat_map
                      (fun ds ->
                        if supported t ds then
                          [ (t, ds, false, 36); (t, ds, true, 110); (t, ds, false, 160) ]
                        else [])
                      [
                        D (Nx.float32, Nx.float32); D (Nx.float32, Nx.float64);
                        D (Nx.float64, Nx.float64);
                      ])
                  gpu_targets)
               key_order;
             cases "stores each address of workgroup memory once between barriers"
               ~id:ffma_id ffma_cases
               (fun (t, ds, causal, d) -> no_races t (forward ~batches ~causal t ds d));
             cases "writes each buffer its tiles' loop reads in that loop"
               ~id:ffma_id ffma_cases
               (fun (t, ds, causal, d) ->
                 in_the_tiles_loop t (forward ~batches ~causal t ds d));
           ];
         group "the derivative's kernels on a tensor core"
           [
             cases "store each address of workgroup memory once between barriers"
               ~id:derivative_id derivative_cases (on_derivative no_races);
             cases "write each buffer their tiles' loop reads in that loop"
               ~id:derivative_id derivative_cases (on_derivative in_the_tiles_loop);
           ];
       ])
