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

let strf = Printf.sprintf

let pp_ints a =
  String.concat "; " (Array.to_list (Array.map string_of_int a))

let dim_max (d : K.t) : int =
  match K.view d with
  | K.Const { value; _ } -> (
      match Const.view value with
      | Int n -> Int64.to_int n
      | _ -> failwith "dim_max: not an integer constant")
  | _ ->
      (* XXX use Divandmod.vmax when available *)
      failwith "dim_max: non-constant dimension not yet supported"

(* Grouping and splitting *)

let group_dims dims max_sizes =
  let dims = ref (Array.copy dims) in
  let n_max = Array.length max_sizes in
  let changed = ref true in
  while !changed do
    changed := false;
    let d = !dims in
    let n = Array.length d in
    if n > n_max
       || Array.exists2 (fun d m -> d > m) d
            (Array.sub max_sizes 0 (min n n_max))
    then begin
      let merged = ref false in
      for i = 0 to min n_max (n - 1) - 1 do
        if not !merged && i < n - 1
           && d.(i) * d.(i + 1) <= max_sizes.(i)
        then begin
          let new_dims = Array.make (n - 1) 0 in
          Array.blit d 0 new_dims 0 i;
          new_dims.(i) <- d.(i) * d.(i + 1);
          Array.blit d (i + 2) new_dims (i + 1) (n - i - 2);
          dims := new_dims;
          changed := true;
          merged := true
        end
      done;
      if not !merged then begin
        dims := [||];
        changed := false
      end
    end
  done;
  if Array.length !dims = 0 then None else Some !dims

let split_dims dims max_sizes =
  if Array.for_all2 (fun d m -> d <= m) dims
       (Array.sub max_sizes 0 (Array.length dims))
  then dims
  else begin
    let d = Array.make 3 1 in
    let n = Array.length dims in
    for i = 0 to min n 3 - 1 do d.(i) <- dims.(i) done;
    for i = 0 to 2 do
      while d.(i) > max_sizes.(i) do
        let ceil_sqrt = int_of_float (ceil (sqrt (float_of_int d.(i)))) in
        let div = ref 1 in
        for f = 2 to ceil_sqrt do
          if !div = 1 && d.(i) mod f = 0 then div := f
        done;
        if !div = 1 then
          failwith (strf "cannot limit dim [%s], max_sizes=[%s]"
                      (pp_ints dims) (pp_ints max_sizes));
        let next = (i + 1) mod 3 in
        d.(next) <- d.(next) * !div;
        d.(i) <- d.(i) / !div
      done
    done;
    if d.(2) = 1 then Array.sub d 0 2
    else if d.(1) = 1 && d.(2) = 1 then Array.sub d 0 1
    else d
  end

(* Flatten SPECIAL raw indices to 1D, then decompose back to original dims. *)
let flatten_and_decompose raw limited dims =
  let ( + ) = K.O.( + ) and ( * ) = K.O.( * ) in
  let ( / ) = K.O.( / ) and ( mod ) = K.O.( mod ) in
  let flat =
    if Array.length limited = 2 then
      raw.(0) * limited.(1) + raw.(1)
    else
      raw.(0) * (limited.(1) * limited.(2))
      + raw.(1) * limited.(2) + raw.(2)
  in
  match Array.length dims with
  | 1 -> [ flat ]
  | 2 -> [ flat / dims.(1); flat mod dims.(1) ]
  | _ ->
      [ flat / (dims.(2) * dims.(1))
      ; flat / dims.(2) mod dims.(1)
      ; flat mod dims.(2) ]

(* Map logical kernel range sizes to physical GPU grid dimensions (SPECIAL
   nodes).  Dimensions are optionally reversed, then grouped or split to fit
   within max_sizes limits; when the physical dim count differs from the
   logical count, a flatten-and-decompose step recovers per-range indices
   via divmod arithmetic on the combined SPECIAL values. *)
let rec get_grouped_dims prefix dims max_sizes ~reverse =
  if reverse then
    let rev = Array.copy dims in
    let n = Array.length rev in
    for i = 0 to n / 2 - 1 do
      let t = rev.(i) in
      rev.(i) <- rev.(n - 1 - i);
      rev.(n - 1 - i) <- t
    done;
    List.rev (get_grouped_dims prefix rev max_sizes ~reverse:false)
  else
    let idims = Array.map dim_max dims in
    let limited_ints =
      match max_sizes with
      | None -> idims
      | Some max_sizes ->
          let max_sizes = Array.of_list max_sizes in
          let limited =
            match group_dims idims max_sizes with
            | Some g -> g
            | None -> idims
          in
          if Array.length limited > Array.length max_sizes then
            failwith (strf "cannot limit dim [%s], max_sizes=[%s]"
                        (pp_ints idims) (pp_ints max_sizes));
          if limited = idims then split_dims idims max_sizes else limited
    in
    let dim_of_prefix i = match prefix with
      | "gidx" -> Special_dim.Group_id i
      | "lidx" -> Special_dim.Local_id i
      | "idx" -> Special_dim.Global_idx i
      | s -> failwith (strf "unknown dim prefix: %s" s)
    in
    let ( + ) = K.O.( + ) and ( * ) = K.O.( * ) in
    let ( / ) = K.O.( / ) and ( mod ) = K.O.( mod ) in
    let limited = Array.map K.O.int_ limited_ints in
    let raw =
      Array.mapi
        (fun i s -> K.special ~dim:(dim_of_prefix i) ~size:s ())
        limited
    in
    let nl = Array.length limited and nd = Array.length dims in
    if nl < nd then begin
      match Helpers.get_contraction idims limited_ints with
      | None ->
          failwith (strf
                      "get_contraction should not be None dims=[%s] limited=[%s]"
                      (pp_ints idims) (pp_ints limited_ints))
      | Some contraction ->
          let ret = ref [] in
          List.iteri
            (fun i contraction_group ->
              let cur = ref raw.(i) in
              let group = Array.of_list contraction_group in
              for j = 0 to Array.length group - 2 do
                let c = dims.(group.(j)) in
                ret := (!cur mod c) :: !ret;
                cur := !cur / c
              done;
              ret := !cur :: !ret)
            contraction;
          List.rev !ret
    end else if nl > nd then begin
      if nl = 2 && nd = 1 then
        [ raw.(0) * limited.(1) + raw.(1) ]
      else if nl = 3 && nd = 1 then
        [ (raw.(0) * limited.(1) + raw.(1)) * limited.(2) + raw.(2) ]
      else if limited_ints <> idims then
        flatten_and_decompose raw limited dims
      else Array.to_list raw
    end else if limited_ints <> idims then
      flatten_and_decompose raw limited dims
    else Array.to_list raw

(* Range key: (axis, sub) -- everything except the kind. *)

module Range_key = struct
  type t = int * int list
  let compare = Stdlib.compare
  let of_range r = (K.range_axis r, K.range_sub r)
end

module Rkmap = Map.Make (Range_key)

(* Pass *)

let add_gpudims (ctx : Renderer.t) (s : K.t) : K.t option =
  match K.view s with
  | Sink { kernel_info = None; _ } -> None
  | Sink { kernel_info = Some ki; _ } ->
      let s_topo = K.toposort s in
      if List.exists
           (fun x -> match K.view x with K.Special _ -> true | _ -> false)
           s_topo
      then None
      else begin
        let all_ranges =
          List.fold_left
            (fun acc x ->
              if K.is_range x then Rkmap.add (Range_key.of_range x) x acc
              else acc)
            Rkmap.empty s_topo
        in
        let extract_keys pred =
          Rkmap.fold
            (fun key x acc -> if pred (K.range_kind x) then key :: acc else acc)
            all_ranges []
          |> List.sort Range_key.compare
        in
        let global_dims =
          extract_keys (function Axis_kind.Global | Thread -> true | _ -> false)
        in
        let local_dims =
          extract_keys (function
            | Axis_kind.Warp | Local | Group_reduce -> true | _ -> false)
        in
        if global_dims = [] && local_dims = [] then None
        else begin
          let shape_of keys =
            Array.of_list
              (List.map
                 (fun k -> K.range_size (Rkmap.find k all_ranges))
                 keys)
          in
          let global_shape = shape_of global_dims in
          let local_shape = shape_of local_dims in
          let idxs =
            if Renderer.has_threads ctx then begin
              assert (Array.length global_shape > 0);
              let hi = dim_max global_shape.(0) - 1 in
              let core =
                K.define_var ~name:"core_id" ~lo:0 ~hi ~dtype:Dtype.int32 ()
              in
              [ K.cast ~src:core ~dtype:(Dtype.to_any Dtype.index) ]
            end else if ki.dont_use_locals then begin
              assert (local_dims = []);
              get_grouped_dims "idx" global_shape
                (Renderer.global_max ctx) ~reverse:true
            end else
              get_grouped_dims "gidx" global_shape
                (Renderer.global_max ctx) ~reverse:true
              @ get_grouped_dims "lidx" local_shape
                  (Renderer.local_max ctx) ~reverse:false
          in
          let lr_tbl = lazy (K.live_ranges_tbl s) in
          let all_dim_keys = global_dims @ local_dims in
          let dim_idx =
            List.fold_left
              (fun (acc, i) k -> (Rkmap.add k i acc, i + 1))
              (Rkmap.empty, 0) all_dim_keys
            |> fst
          in
          let subs = ref [] in
          List.iter
            (fun r ->
              match K.view r with
              | Store { dst = idx; _ } -> begin
                  match K.view idx with
                  | Index { ptr; idxs = idx_idxs; gate; dtype = Dtype.P idx_pty }
                    when Dtype.addrspace idx_pty = Dtype.Global ->
                      let idx_ranges =
                        Option.value ~default:[]
                          (K.Ref_tbl.find_opt (Lazy.force lr_tbl) idx)
                      in
                      let missing_locals =
                        List.filter_map
                          (fun rng_key ->
                            let rng = Rkmap.find rng_key all_ranges in
                            if List.exists (fun x -> x == rng) idx_ranges then
                              None
                            else Some rng)
                          local_dims
                      in
                      if missing_locals <> [] then begin
                        assert (gate = None);
                        let open K.O in
                        let mask =
                          List.fold_left
                            (fun acc x ->
                              K.binary ~op:`And ~lhs:acc ~rhs:(eq x (int_ 0)))
                            (eq (List.hd missing_locals) (int_ 0))
                            (List.tl missing_locals)
                        in
                        let value =
                          match idx_idxs with
                          | [] -> K.const_int 0
                          | [v] -> v
                          | first :: rest ->
                              List.fold_left
                                (fun acc x -> K.binary ~op:`Add ~lhs:acc ~rhs:x)
                                first rest
                        in
                        let dt = Option.value ~default:Dtype.index (K.dtype value) in
                        let gated =
                          K.O.where (K.broadcast mask (Dtype.count dt)) value
                            (K.invalid_index ~lanes:(Dtype.count dt) ())
                        in
                        subs := (idx, K.index ~ptr ~idxs:[ gated ] ()) :: !subs
                      end
                  | _ -> ()
                end
              | _ ->
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
        end
      end
  | _ -> None

let pm_add_gpudims (ctx : Renderer.t) (root : K.t) : K.t =
  K.graph_rewrite ~name:"add gpudims" (fun node -> add_gpudims ctx node) root
