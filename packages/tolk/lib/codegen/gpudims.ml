(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* Port of tinygrad/codegen/gpudims.py to the tolk_uop IR. *)

open Tolk_uop
module U = Uop

let strf = Printf.sprintf

(* Helpers *)

let pp_ints a =
  String.concat "; " (Array.to_list (Array.map string_of_int a))

let err_limit dims max_sizes =
  strf "cannot limit dim [%s], max_sizes=[%s]"
    (pp_ints dims) (pp_ints max_sizes)

let dim_max (d : U.t) : int = U.vmax d

type dim_kind =
  | Group_id
  | Local_id
  | Global_idx

let special_name_of_kind kind i =
  let dim =
    match kind with
    | Group_id -> Gpu_dim.Group_id i
    | Local_id -> Gpu_dim.Local_id i
    | Global_idx -> Gpu_dim.Global_idx i
  in
  Gpu_dim.to_special_name dim

let smallest_factor n =
  let limit = int_of_float (ceil (sqrt (float_of_int n))) in
  let rec loop f =
    if f > limit then 1
    else if n mod f = 0 then f
    else loop (f + 1)
  in
  loop 2

let array_rev a =
  let n = Array.length a in
  Array.init n (fun i -> a.(n - 1 - i))

let product_uops_from a start =
  let open U.O in
  let acc = ref (int_ 1) in
  for i = start to Stdlib.(Array.length a - 1) do
    acc := !acc * a.(i)
  done;
  !acc

let group_dim_values dims max_sizes =
  let dims = ref (Array.copy dims) in
  let rec loop () =
    let d = !dims in
    let n = Array.length d in
    let nm = Array.length max_sizes in
    if n <= nm
       && not
            (Array.exists2
               (fun d m -> dim_max d > m)
               d (Array.sub max_sizes 0 (min n nm)))
    then Some d
    else
      let rec try_merge i =
        if i >= nm || i >= n - 1 then None
        else if dim_max d.(i) * dim_max d.(i + 1) <= max_sizes.(i) then begin
          dims := Array.init (n - 1) (fun j ->
            if j < i then d.(j)
            else if j = i then U.simplify U.O.(d.(i) * d.(Stdlib.(i + 1)))
            else d.(j + 1));
          loop ()
        end else try_merge (i + 1)
      in
      try_merge 0
  in
  loop ()

(* Split dims that exceed max_sizes by factoring into adjacent slots. *)
let split_dims dims max_sizes =
  if Array.for_all2 (fun d m -> d <= m)
       dims (Array.sub max_sizes 0 (Array.length dims))
  then dims
  else begin
    let d = Array.make 3 1 in
    for i = 0 to min (Array.length dims) 3 - 1 do d.(i) <- dims.(i) done;
    for i = 0 to 2 do
      while d.(i) > max_sizes.(i) do
        let div = smallest_factor d.(i) in
        if div = 1 then failwith (err_limit dims max_sizes);
        let next = (i + 1) mod 3 in
        d.(next) <- d.(next) * div;
        d.(i) <- d.(i) / div
      done
    done;
    if d.(2) = 1 then Array.sub d 0 2 else d
  end

let flat_index raw limited =
  if Array.length raw = 1 then raw.(0)
  else
  let open U.O in
  let acc = ref (int_ 0) in
  for i = 0 to Stdlib.(Array.length raw - 1) do
    acc := !acc + (raw.(i) * product_uops_from limited Stdlib.(i + 1))
  done;
  U.simplify !acc

let decompose_flat flat dims =
  let open U.O in
  Array.to_list
    (Array.mapi
       (fun i dim ->
         let tail = product_uops_from dims Stdlib.(i + 1) in
         let idx =
           if U.const_int_value tail = Some 1 then U.simplify flat
           else U.simplify (floordiv flat tail)
         in
         if i = 0 then idx else U.simplify (idx mod dim))
       dims)

let same_uop_array a b =
  Array.length a = Array.length b && Array.for_all2 U.equal a b

(* Map logical range sizes to physical GPU dimensions (SPECIAL nodes). *)
let rec get_grouped_dims kind dims max_sizes ~reverse =
  if reverse then
    List.rev (get_grouped_dims kind (array_rev dims) max_sizes ~reverse:false)
  else
    let idims = Array.map dim_max dims in
    let limited_dims =
      match max_sizes with
      | None -> dims
      | Some max_sizes ->
          let max_sizes = Array.of_list max_sizes in
          let limited = group_dim_values dims max_sizes in
          if Option.is_none limited
             && Array.length idims > Array.length max_sizes
          then
            failwith (err_limit idims max_sizes);
          (match limited with
           | Some limited when not (same_uop_array limited dims) ->
               limited
           | Some _ | None ->
               (* [split_dims] returns its argument physically unchanged when
                  every dim fits; keep the original (possibly symbolic) dims
                  in that case. *)
               let split = split_dims idims max_sizes in
               if split == idims then dims
               else Array.map U.O.int_ split)
    in
    let raw =
      Array.mapi (fun i s ->
        U.special ~name:(special_name_of_kind kind i) ~size:s ())
        limited_dims
    in
    decompose_flat (flat_index raw limited_dims) dims

(* Range key: (axis, sub) — everything except the kind. *)
module Range_key = struct
  type t = int * int list

  let compare = Stdlib.compare

  let of_range r =
    match U.as_range r with
    | Some v -> (v.axis, v.sub)
    | None -> invalid_arg "Range_key.of_range: not a range"
end

module Rkmap = Map.Make (Range_key)

(* Build a gated index when local ranges are missing from a global store. *)
let gate_missing_locals (idx : U.t) (idx_view : U.index_view)
    (missing : U.t list) : U.t =
  if List.length idx_view.idxs <> 1 then
    invalid_arg "index has 2 sources";
  let open U.O in
  let eq lhs rhs = U.alu_binary ~op:Ops.Cmpeq ~lhs ~rhs in
  let mask =
    List.fold_left
      (fun acc x -> U.alu_binary ~op:Ops.And ~lhs:acc ~rhs:(eq x (int_ 0)))
      (eq (List.hd missing) (int_ 0))
      (List.tl missing)
  in
  U.replace idx
    ~src:
      (Array.of_list
         (idx_view.ptr
         :: List.map (fun v -> U.valid ~src:v ~cond:mask) idx_view.idxs))
    ()

(* Per-device compute grid for a kernel. *)
let compute_idxs (ctx : Renderer.t) ki ~global_shape ~local_shape ~local_dims =
  if Renderer.has_threads ctx then begin
    if Array.length global_shape <> 1 || local_dims <> [] then
      invalid_arg
        "threaded renderer expects exactly one global range and no local ranges";
    let hi = dim_max global_shape.(0) - 1 in
    let core =
      U.variable ~name:"core_id" ~min_val:0 ~max_val:hi ~dtype:Dtype.int32 ()
    in
    [ U.cast ~src:core ~dtype:Dtype.index ]
  end
  else if ki.U.dont_use_locals then begin
    assert (local_dims = []);
    get_grouped_dims Global_idx global_shape (Renderer.global_max ctx)
      ~reverse:true
  end
  else begin
    let local_idxs =
      get_grouped_dims Local_id local_shape (Renderer.local_max ctx)
        ~reverse:false
    in
    let hw_local =
      List.filter_map
        (fun u -> Option.map (fun v -> dim_max v.U.size) (U.as_special u))
        local_idxs
    in
    let global_max =
      match Renderer.global_prod_max ctx with
      | None -> Renderer.global_max ctx
      | Some pm ->
          let gm = Option.value (Renderer.global_max ctx) ~default:pm in
          let rec zip3 gs ps ls = match gs, ps, ls with
            | g :: gs, p :: ps, l :: ls -> min g (p / l) :: zip3 gs ps ls
            | g :: gs, p :: ps, [] -> min g p :: zip3 gs ps []
            | _ -> []
          in
          Some (zip3 gm pm (hw_local @ [ 1; 1; 1 ]))
    in
    get_grouped_dims Group_id global_shape global_max ~reverse:true
    @ local_idxs
  end

(* Substitute ranges with SPECIAL-based GPU dimension indices. *)
let add_gpudims (ctx : Renderer.t) (s : U.t) : U.t option =
  match U.op s, U.as_kernel_info s with
  | Ops.Sink, Some ki ->
      let s_topo = U.toposort s in
      if List.exists (fun x -> U.op x = Ops.Special) s_topo then None
      else
        let all_ranges =
          List.fold_left
            (fun acc x ->
              if U.op x = Ops.Range
              then Rkmap.add (Range_key.of_range x) x acc
              else acc)
            Rkmap.empty s_topo
        in
        let range_kind r = (Option.get (U.as_range r)).U.kind in
        let extract_keys pred =
          Rkmap.fold
            (fun key x acc -> if pred (range_kind x) then key :: acc else acc)
            all_ranges []
          |> List.sort Range_key.compare
        in
        let global_dims =
          extract_keys (function
            | Axis_type.Global | Thread -> true
            | _ -> false)
        in
        let local_dims =
          extract_keys (function
            | Axis_type.Warp | Local | Group_reduce -> true
            | _ -> false)
        in
        if global_dims = [] && local_dims = [] then None
        else
          let shape_of keys =
            Array.of_list (List.map (fun k ->
              U.simplify (Option.get (U.as_range (Rkmap.find k all_ranges))).size)
              keys)
          in
          let global_shape = shape_of global_dims in
          let local_shape = shape_of local_dims in
          let idxs =
            compute_idxs ctx ki ~global_shape ~local_shape ~local_dims
          in
          let all_dim_keys = global_dims @ local_dims in
          let dim_idx, _ =
            List.fold_left
              (fun (acc, i) k -> (Rkmap.add k i acc, i + 1))
              (Rkmap.empty, 0) all_dim_keys
          in
          (* Two substitution passes. The gated index built below references
             the original local/global ranges; a single combined substitution
             would see its children rewritten to SPECIAL before the
             (old_idx -> gated_idx) mapping gets a chance to apply, silently
             dropping the Invalid sentinel. Pass 1 installs the gated
             indices; pass 2 maps non-reduce ranges to their SPECIAL idxs. *)
          let gate_subs = ref [] in
          let range_subs = ref [] in
          let gate_store r =
            match U.as_store r with
            | None -> ()
            | Some { dst = idx; _ } ->
                match U.as_index idx with
                | Some ({ ptr; _ } as idx_view)
                  when U.addrspace ptr = Some Dtype.Global ->
                    let idx_ranges = U.ranges idx in
                    let missing =
                      List.filter_map
                        (fun rk ->
                          let rng = Rkmap.find rk all_ranges in
                          if List.exists (U.equal rng) idx_ranges
                          then None else Some rng)
                        local_dims
                    in
                    if missing <> [] then
                      gate_subs :=
                        (idx, gate_missing_locals idx idx_view missing)
                        :: !gate_subs
                | _ -> ()
          in
          let sub_range r =
            if U.op r <> Ops.Range then ()
            else match Rkmap.find_opt (Range_key.of_range r) dim_idx with
              | Some ii when range_kind r <> Axis_type.Reduce ->
                  range_subs := (r, List.nth idxs ii) :: !range_subs
              | _ -> ()
          in
          List.iter (fun r -> gate_store r; sub_range r) s_topo;
          if !gate_subs = [] && !range_subs = [] then None
          else
            let s =
              if !gate_subs = [] then s else U.substitute !gate_subs s
            in
            Some (if !range_subs = [] then s else U.substitute !range_subs s)
  | _ -> None

let pm_add_gpudims (ctx : Renderer.t) (root : U.t) : U.t =
  U.graph_rewrite ~name:"add gpudims" (fun node -> add_gpudims ctx node) root
