(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(* GPU dimension mapping.

   Maps logical kernel ranges to physical GPU grid dimensions (SPECIAL nodes)
   via grouping, splitting, and contraction. *)

open Tolk_ir
module K = Kernel

let pp_ints a =
  String.concat "; " (Array.to_list (Array.map string_of_int a))

let err_limit idims max_sizes =
  Printf.sprintf "cannot limit dim [%s], max_sizes=[%s]"
    (pp_ints idims) (pp_ints max_sizes)

let dim_max (d : K.t) : int =
  match K.const_arg d with
  | Some (Int n) -> Int64.to_int n
  | _ -> failwith "dim_max: non-constant dimension not yet supported"

let dim_of_prefix prefix i = match prefix with
  | "gidx" -> Special_dim.Group_id i
  | "lidx" -> Special_dim.Local_id i
  | "idx" -> Special_dim.Global_idx i
  | s -> failwith (Printf.sprintf "unknown dim prefix: %s" s)

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

(* Merge adjacent dims until they fit within max_sizes. *)
let group_dims dims max_sizes =
  let dims = ref (Array.copy dims) in
  let rec loop () =
    let d = !dims in
    let n = Array.length d in
    let nm = Array.length max_sizes in
    if n <= nm
       && not (Array.exists2 (fun d m -> d > m)
                 d (Array.sub max_sizes 0 (min n nm)))
    then Some d
    else
      let rec try_merge i =
        if i >= nm || i >= n - 1 then None
        else if d.(i) * d.(i + 1) <= max_sizes.(i) then begin
          dims := Array.init (n - 1) (fun j ->
            if j < i then d.(j)
            else if j = i then d.(i) * d.(i + 1)
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
    if d.(2) = 1 then Array.sub d 0 2
    else if d.(1) = 1 && d.(2) = 1 then Array.sub d 0 1
    else d
  end

(* Flatten SPECIAL raw indices to 1D, then decompose back to original dims. *)
let flatten_and_decompose raw limited dims =
  let open K.O in
  let flat = match Array.length limited with
    | 2 -> raw.(0) * int_ limited.(1) + raw.(1)
    | _ -> raw.(0) * int_ Stdlib.(limited.(1) * limited.(2))
            + raw.(1) * int_ limited.(2) + raw.(2)
  in
  match Array.length dims with
  | 1 -> [flat]
  | 2 -> [flat / int_ dims.(1); flat mod int_ dims.(1)]
  | _ -> [flat / int_ Stdlib.(dims.(2) * dims.(1));
          flat / int_ dims.(2) mod int_ dims.(1);
          flat mod int_ dims.(2)]

(* Map logical range sizes to physical GPU dimensions (SPECIAL nodes). *)
let rec get_grouped_dims prefix dims max_sizes ~reverse =
  if reverse then
    List.rev (get_grouped_dims prefix (array_rev dims) max_sizes ~reverse:false)
  else
    let idims = Array.map dim_max dims in
    let limited = match max_sizes with
      | None -> idims
      | Some max_sizes ->
          let max_sizes = Array.of_list max_sizes in
          let limited = match group_dims idims max_sizes with
            | Some g -> g | None -> idims in
          if Array.length limited > Array.length max_sizes then
            failwith (err_limit idims max_sizes);
          if limited = idims then split_dims idims max_sizes else limited
    in
    let raw = Array.mapi (fun i s ->
      K.special ~dim:(dim_of_prefix prefix i) ~size:(K.O.int_ s) ()) limited in
    let nl = Array.length limited and nd = Array.length idims in
    if nl < nd then
      match Helpers.get_contraction idims limited with
      | None ->
          failwith (Printf.sprintf
            "get_contraction should not be None dims=[%s] limited=[%s]"
            (pp_ints idims) (pp_ints limited))
      | Some contraction ->
          let open K.O in
          let ret = ref [] in
          List.iteri (fun i group ->
            let cur = ref raw.(i) in
            let group = Array.of_list group in
            for j = 0 to Array.length group - 2 do
              let c = dims.(group.(j)) in
              ret := (!cur mod c) :: !ret;
              cur := !cur / c
            done;
            ret := !cur :: !ret) contraction;
          List.rev !ret
    else if nl > nd then
      let open K.O in
      if nl = 2 && nd = 1 then
        [raw.(0) * int_ limited.(1) + raw.(1)]
      else if nl = 3 && nd = 1 then
        [(raw.(0) * int_ limited.(1) + raw.(1)) * int_ limited.(2) + raw.(2)]
      else if limited <> idims then
        flatten_and_decompose raw limited idims
      else Array.to_list raw
    else if limited <> idims then
      flatten_and_decompose raw limited idims
    else Array.to_list raw

(* Range key: (axis, sub) — everything except the kind. *)
module Range_key = struct
  type t = int * int list
  let compare = Stdlib.compare
  let of_range r = (K.range_axis r, K.range_sub r)
end

module Rkmap = Map.Make (Range_key)

(* Substitute ranges with SPECIAL-based GPU dimension indices. *)
let add_gpudims (ctx : Renderer.t) (s : K.t) : K.t option =
  match K.view s with
  | Sink { kernel_info = None; _ } -> None
  | Sink { kernel_info = Some ki; _ } ->
      let s_topo = K.toposort s in
      if List.exists (fun x ->
        match K.view x with Special _ -> true | _ -> false) s_topo
      then None
      else
        (* Collect all ranges keyed by (axis, sub). *)
        let all_ranges = List.fold_left (fun acc x ->
          if K.is_range x then Rkmap.add (Range_key.of_range x) x acc
          else acc) Rkmap.empty s_topo in
        let extract_keys pred =
          Rkmap.fold (fun key x acc ->
            if pred (K.range_kind x) then key :: acc else acc)
            all_ranges []
          |> List.sort Range_key.compare in
        let global_dims = extract_keys (function
          | Axis_kind.Global | Thread -> true | _ -> false) in
        let local_dims = extract_keys (function
          | Axis_kind.Warp | Local | Group_reduce -> true | _ -> false) in
        if global_dims = [] && local_dims = [] then None
        else
          let shape_of keys = Array.of_list (List.map (fun k ->
            K.range_size (Rkmap.find k all_ranges)) keys) in
          let global_shape = shape_of global_dims in
          let local_shape = shape_of local_dims in
          (* Compute per-range index expressions. *)
          let idxs =
            if Renderer.has_threads ctx then begin
              assert (Array.length global_shape > 0);
              let hi = dim_max global_shape.(0) - 1 in
              let core = K.define_var ~name:"core_id" ~lo:0 ~hi
                ~dtype:Dtype.Val.int32 () in
              [K.cast ~src:core ~dtype:Dtype.index]
            end else if ki.dont_use_locals then begin
              assert (local_dims = []);
              get_grouped_dims "idx" global_shape
                (Renderer.global_max ctx) ~reverse:true
            end else begin
              let local_idxs = get_grouped_dims "lidx" local_shape
                (Renderer.local_max ctx) ~reverse:false in
              let hw_local = List.filter_map (fun u ->
                match K.view u with
                | Special { size; _ } -> Some (dim_max size)
                | _ -> None) local_idxs in
              let global_max = match Renderer.global_prod_max ctx with
                | None -> Renderer.global_max ctx
                | Some pm ->
                    let gm = match Renderer.global_max ctx with
                      | Some g -> g | None -> pm in
                    let rec zip3 gs ps ls = match gs, ps, ls with
                      | g :: gs, p :: ps, l :: ls ->
                          min g (p / l) :: zip3 gs ps ls
                      | g :: gs, p :: ps, [] ->
                          min g p :: zip3 gs ps []
                      | _ -> [] in
                    Some (zip3 gm pm (hw_local @ [1; 1; 1])) in
              get_grouped_dims "gidx" global_shape global_max ~reverse:true
              @ local_idxs
            end
          in
          (* Build substitution map. *)
          let lr_tbl = K.live_ranges_tbl s in
          let all_dim_keys = global_dims @ local_dims in
          let dim_idx = List.fold_left (fun (acc, i) k ->
            (Rkmap.add k i acc, i + 1)) (Rkmap.empty, 0) all_dim_keys
            |> fst in
          let subs = ref [] in
          List.iter (fun r ->
            (* Guard global stores against missing local ranges. *)
            (match K.view r with
             | Store { dst = idx; _ } ->
                 (match K.view idx with
                  | Index { ptr; idxs = idx_srcs; gate; dtype = Dtype.Ptr idx_pty }
                    when Dtype.Ptr.addrspace idx_pty = Dtype.Global ->
                      let idx_ranges = Option.value ~default:[]
                        (K.Ref_tbl.find_opt lr_tbl idx) in
                      let missing = List.filter_map (fun rk ->
                        let rng = Rkmap.find rk all_ranges in
                        if List.exists (fun x -> x == rng) idx_ranges
                        then None else Some rng) local_dims in
                      if missing <> [] then begin
                        assert (gate = None);
                        let open K.O in
                        let mask = List.fold_left (fun acc x ->
                          K.binary ~op:`And ~lhs:acc ~rhs:(eq x (int_ 0)))
                          (eq (List.hd missing) (int_ 0))
                          (List.tl missing) in
                        let value = match idx_srcs with
                          | [] -> K.const_int 0 | [v] -> v
                          | first :: rest ->
                              List.fold_left (fun a x ->
                                K.binary ~op:`Add ~lhs:a ~rhs:x) first rest in
                        let dt = K.dtype value in
                        let gated = K.O.where
                          (K.broadcast mask (Dtype.count dt)) value
                          (K.invalid_index ~lanes:(Dtype.count dt) ()) in
                        subs := (idx, K.index ~ptr ~idxs:[gated] ()) :: !subs
                      end
                  | _ -> ())
             | _ -> ());
            (* Substitute non-reduce ranges with their idx expression. *)
            if K.is_range r then begin
              let key = Range_key.of_range r in
              match Rkmap.find_opt key dim_idx with
              | Some ii when K.range_kind r <> Axis_kind.Reduce ->
                  subs := (r, List.nth idxs ii) :: !subs
              | _ -> ()
            end)
            s_topo;
          if !subs = [] then None
          else Some (K.substitute !subs s)
  | _ -> None

let pm_add_gpudims (ctx : Renderer.t) (root : K.t) : K.t =
  K.graph_rewrite ~name:"add gpudims" (fun node -> add_gpudims ctx node) root
