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

(* [memo key f] is [f], computed once for each [key x]: the laws of a row
   share its kernels' rendering. *)
let memo key f =
  let seen = Hashtbl.create 64 in
  fun x ->
    let k = key x in
    match Hashtbl.find_opt seen k with
    | Some y -> y
    | None ->
        let y = f x in
        Hashtbl.add seen k y;
        y

let forward =
  let once =
    memo
      (fun (t, ds, batches, causal, d) -> (target_id t, dtypes_id ds, batches, causal, d))
      (fun (t, ds, batches, causal, d) -> forward ~batches ~causal t ds d)
  in
  fun ?(batches = 2) ?(causal = false) target ds d -> once (target, ds, batches, causal, d)

(* A kernel and what the laws compute of it, once each, by its target. *)
module Kernels = Hashtbl.Make (struct
  type t = string * O.t

  let equal (a, x) (b, y) = String.equal a b && O.equal x y
  let hash (a, x) = Hashtbl.hash (a, O.hash x)
end)

let by_kernel f =
  let seen = Kernels.create 64 in
  fun target ast ->
    let k = (target_id target, ast) in
    match Kernels.find_opt seen k with
    | Some y -> y
    | None ->
        let y = f target ast in
        Kernels.add seen k y;
        y

let params ast =
  List.length
    (List.filter (fun u -> O.op u = Tolk.Op.Param) (O.toposort ~calls:Skip ast))

(* The linearized kernel [ast] for [target]'s renderer. *)
let linearized =
  by_kernel (fun target ast ->
      let p =
        match Lowered.program ~target ast with
        | Some p -> p
        | None -> Tolk.Codegen.linearize ast (Tolk.Cstyle.without_compiler target)
      in
      O.src (O.nth p 1))

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

(* An access of workgroup memory: by the barriers before it, a store or a
   load, its buffer and, at every loop's first trip or every loop of a
   constant bound at its last ([~last]) and at a thread [t], the addresses
   it touches: none where it does not run, one for an index, and a
   vector's for a shrink [(buffer, start, length)]. *)
type access = {
  span : int;
  store : bool;
  buf : O.t;
  at : last:bool -> (string * int) list -> int list;
}

let value u =
  match (O.vmin u, O.vmax u) with
  | `Int lo, `Int hi when Tolk.Bigint.equal lo hi -> Some (Tolk.Bigint.to_int lo)
  | `Bool lo, `Bool hi when lo = hi -> Some (Bool.to_int lo)
  | _ -> None

let accesses =
  by_kernel @@ fun target ast ->
  let uops = linearized target ast in
  let ranges = List.filter (fun u -> O.op u = Tolk.Op.Range) uops in
  let specials = List.filter (fun u -> O.op u = Tolk.Op.Special) uops in
  let at_thread ~last t u =
    let int r n = O.const ~dtype:(O.dtype r) (`Int (Tolk.Bigint.of_int n)) in
    let trip r =
      match (last, value (O.nth r 0)) with
      | true, Some n when n > 0 -> int r (n - 1)
      | _ -> int r 0
    in
    let lane s =
      match O.arg s with
      | String n -> int s (Option.value (List.assoc_opt n t) ~default:0)
      | _ -> int s 0
    in
    (* Index arithmetic by tolk's own ({!O.exec_alu}), each node once; what
       is not index arithmetic substitutes and simplifies. *)
    let seen = Kernels.create 64 in
    let rec eval u =
      match Kernels.find_opt seen ("", u) with
      | Some v -> v
      | None ->
          let v =
            match O.op u with
            | Tolk.Op.Const -> Some (O.value u)
            | Tolk.Op.Range -> Some (O.value (trip u))
            | Tolk.Op.Special -> Some (O.value (lane u))
            | Tolk.Op.Cast when not (Tolk.Dtype.is_float (O.dtype u)) -> (
                match eval (O.nth u 0) with
                | Some (`Bool b) -> Some (Tolk.Dtype.truncate (O.dtype u) (`Int (Tolk.Bigint.of_int (Bool.to_int b))) :> Tolk.Dtype.const)
                | Some (`Int _ as v) -> Some (Tolk.Dtype.truncate (O.dtype u) v :> Tolk.Dtype.const)
                | _ -> None)
            | op when Tolk.Op.Set.mem op Tolk.Op.Set.alu ->
                let rec all acc = function
                  | [] -> Some (List.rev acc)
                  | x :: xs -> ( match eval x with Some v -> all (v :: acc) xs | None -> None)
                in
                Option.map (O.exec_alu op (O.dtype u)) (all [] (O.src u))
            | _ -> None
          in
          Kernels.add seen ("", u) v;
          v
    in
    match eval u with
    | Some (`Int z) -> Some (Tolk.Bigint.to_int z)
    | Some (`Bool b) -> Some (Bool.to_int b)
    | _ ->
        let subs = List.map (fun r -> (r, trip r)) ranges @ List.map (fun s -> (s, lane s)) specials in
        value (Tolk.Shape.simplify (O.substitute ~calls:Skip ~pass:Fixed_point u subs))
  in
  let local u = O.addrspace u = Some Tolk.Dtype.Local in
  let access span gates ~store ix =
    let n, gated =
      if O.op ix = Tolk.Op.Shrink then (Option.get (value (O.nth ix 2)), 3) else (1, 2)
    in
    let all = gates @ List.filteri (fun i _ -> i >= gated) (O.src ix) in
    let at =
      memo Fun.id (fun (last, t) ->
          match at_thread ~last t (O.nth ix 1) with
          | Some a when List.for_all (fun g -> at_thread ~last t g = Some 1) all ->
              List.init n (( + ) a)
          | _ -> [])
    in
    let at ~last t = at (last, t) in
    { span; store; buf = O.buf_uop ix; at }
  in
  let rec walk span gates acc = function
    | [] -> List.rev acc
    | u :: rest -> (
        match O.op u with
        | Tolk.Op.Barrier -> walk (span + 1) gates acc rest
        | Tolk.Op.If -> walk span (O.nth u 0 :: gates) acc rest
        | Tolk.Op.Endif -> walk span (List.tl gates) acc rest
        | Tolk.Op.Store when local (O.nth u 0) ->
            walk span gates (access span gates ~store:true (O.nth u 0) :: acc) rest
        | Tolk.Op.Load when local (O.nth u 0) ->
            walk span gates (access span gates ~store:false (O.nth u 0) :: acc) rest
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
  let ss, specials = accesses target ast in
  let ss = List.filter (fun a -> a.store) ss in
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
            (s.at ~last:false t))
        ss)
    (threads specials)

(* A kernel's threads load from workgroup memory only what its threads
   stored there before a barrier before the load, within the buffer: every
   loop at its first trip, and at its last where its bound is constant, as
   a chunk's last copy and product run. A product that read a staged tile
   past the columns its copy wrote would read a value no store wrote, a NaN
   where those bits are one, times its operand's zeros. *)
let reads_written target ast =
  let accs, specials = accesses target ast in
  is_true ~msg:"loads of workgroup memory" (List.exists (fun a -> not a.store) accs);
  let size buf = Option.get (value (O.nth buf 0)) in
  List.iter
    (fun last ->
      let written = Hashtbl.create 4096 in
      List.iter
        (fun a ->
          let n = size a.buf and key x = (O.hash a.buf, x) in
          List.iter
            (fun t ->
              List.iter
                (fun x ->
                  if x < 0 || x >= n then
                    failf "an access of element %d of a buffer of %d" x n;
                  if a.store then begin
                    if not (Hashtbl.mem written (key x)) then Hashtbl.replace written (key x) a.span
                  end
                  else
                    match Hashtbl.find_opt written (key x) with
                    | Some s when s < a.span -> ()
                    | _ ->
                        failf "a load of element %d of a buffer no store wrote before a barrier%s"
                          x (if last then ", at the loops' last trips" else ""))
                (a.at ~last t))
            (threads specials))
        accs)
    [ false; true ]

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
   of a call of the pair [ds] of 2 batches of 2 heads of 128 rows and keys
   and values of [d] dimensions, unmasked or [causal], rendered for [target]
   as a compiled gradient takes them ({!Lowered.derivative_sources}); [[]]
   where its derivative takes its default. *)
let derivative target (D (dt, acc)) ~causal d =
  let shape = [| 2; 2; 128; d |] in
  let f () =
    let z () = Nx.full dt shape 0.5 in
    let q = z () and k = z () and v = z () in
    let mask = if causal then Some Nx.causal else None in
    Nx.cast Nx.float32 (fst (Nx.attention ?mask ~acc ~kv:(Dense { k; v }) q))
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

let derivative =
  let once =
    memo
      (fun (t, ds, causal, d) -> (target_id t, dtypes_id ds, causal, d))
      (fun (t, ds, causal, d) -> derivative t ds ~causal d)
  in
  fun target ds ~causal d -> once (target, ds, causal, d)

let ffma_id (t, ds, causal, d) =
  Printf.sprintf "%s %s d%d%s" (target_id t) (dtypes_id ds) d
    (if causal then " causal" else "")

(* The derivative's rows on a tensor core: bfloat16 at d36 to d160, plain
   and causal, on each GPU target whose core takes them; CUDA's must. On
   FFMA: float32, float64 and bfloat16 in float64 at d1 to d160, plain and
   causal, on CUDA, on HIP gfx942, whose waves have 64 lanes, and in
   float32 on Metal, which has no float64; float64 at d24 on CUDA and
   gfx942, where a copy's thread of eight columns passes its row's padding;
   and bfloat16 in float32 on CUDA sm_75, which has no bfloat16 core; each
   must take them. *)
let both f ds = List.concat_map (fun d -> [ f false d; f true d ]) ds

let derivative_cases =
  List.concat_map
    (fun t -> both (fun causal d -> (t, D (Nx.bfloat16, Nx.float32), causal, d)) [ 36; 64; 110; 160 ])
    (List.filter (fun (t : Tolk.Helpers.Target.t) -> t.arch <> "sm_75") gpu_targets)

let ffma_derivative_cases =
  let target t = Result.get_ok (Tolk.Helpers.Target.of_string t) in
  List.concat_map
    (fun (t, pairs) ->
      List.concat_map
        (fun ds -> both (fun causal d -> (target t, ds, causal, d)) [ 1; 36; 64; 110; 160 ])
        pairs)
    [
      ( "CUDA:CUDA:sm_89",
        [ D (Nx.float32, Nx.float32); D (Nx.float64, Nx.float64); D (Nx.bfloat16, Nx.float64) ] );
      ( "AMD:HIP:gfx942",
        [ D (Nx.float32, Nx.float32); D (Nx.float64, Nx.float64); D (Nx.bfloat16, Nx.float64) ] );
      ("METAL:METAL:Apple9", [ D (Nx.float32, Nx.float32) ]);
      ("CUDA:CUDA:sm_75", [ D (Nx.bfloat16, Nx.float32) ]);
    ]
  @ List.concat_map
      (fun t -> both (fun causal d -> (target t, D (Nx.float64, Nx.float64), causal, d)) [ 24 ])
      [ "CUDA:CUDA:sm_89"; "AMD:HIP:gfx942" ]

let derivative_id = ffma_id

(* [on_derivative ~must law] holds [law] on each of a case's derivative
   kernels, [q̄]'s and [k̄]'s and [v̄]'s, which a target [must t] takes. *)
let on_derivative ~must law (t, ds, causal, d) =
  match derivative t ds ~causal d with
  | [] -> is_true ~msg:"the target takes the derivative's kernels" (not (must t))
  | ks ->
      is_true ~msg:"q̄'s kernel" (List.exists (fun (n, _) -> n = "attention_dq") ks);
      is_true ~msg:"k̄'s and v̄'s kernels"
        (List.exists (fun (n, _) -> String.starts_with ~prefix:"attention_dkdv" n) ks);
      List.iter (fun (_, ast) -> law t ast) ks

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
             cases "loads only what its threads stored in workgroup memory"
               ~id:ffma_id ffma_cases
               (fun (t, ds, causal, d) ->
                 reads_written t (forward ~batches ~causal t ds d));
           ];
         group "the derivative's kernels on a tensor core"
           [
             cases "store each address of workgroup memory once between barriers"
               ~id:derivative_id derivative_cases (on_derivative ~must:(fun t -> t == cuda) no_races);
             cases "write each buffer their tiles' loop reads in that loop"
               ~id:derivative_id derivative_cases (on_derivative ~must:(fun t -> t == cuda) in_the_tiles_loop);
             cases "load only what their threads stored in workgroup memory"
               ~id:derivative_id derivative_cases (on_derivative ~must:(fun t -> t == cuda) reads_written);
           ];
         group "the derivative's kernels on an FFMA micro-tile"
           [
             cases "store each address of workgroup memory once between barriers"
               ~id:derivative_id ffma_derivative_cases (on_derivative ~must:(fun _ -> true) no_races);
             cases "write each buffer their tiles' loop reads in that loop"
               ~id:derivative_id ffma_derivative_cases
               (on_derivative ~must:(fun _ -> true) in_the_tiles_loop);
             cases "load only what their threads stored in workgroup memory"
               ~id:derivative_id ffma_derivative_cases (on_derivative ~must:(fun _ -> true) reads_written);
           ];
       ])
