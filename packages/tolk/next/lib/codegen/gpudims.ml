(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops

let dim_max = function
  | Int d -> Z.of_int d
  | Sym u -> Dtype.Value.to_z (vmax u)

let cannot_limit dims max_sizes =
  invalid_arg
    (Format.asprintf "cannot limit dim dims=(%a), max_sizes=(%a)"
       (Format.pp_print_list
          ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ", ")
          Sint.pp)
       dims
       (Format.pp_print_list
          ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ", ")
          Format.pp_print_int)
       max_sizes)

(* Merge the leftmost adjacent pair whose product fits its axis, until the dims
   fit. *)
let rec group_dims dims max_sizes =
  let rec fit ds ms =
    match (ds, ms) with
    | d :: ds, m :: ms -> Z.leq (dim_max d) (Z.of_int m) && fit ds ms
    | [], _ -> true
    | _ :: _, [] -> false
  in
  let rec merge ds ms =
    match (ds, ms) with
    | d0 :: (d1 :: rest as ds), m :: ms ->
        if Z.leq Z.(dim_max d0 * dim_max d1) (Z.of_int m) then
          Some (Sint.(d0 * d1) :: rest)
        else Option.map (List.cons d0) (merge ds ms)
    | _ -> None
  in
  if fit dims max_sizes then Some dims
  else
    Option.bind (merge dims max_sizes) (fun dims -> group_dims dims max_sizes)

(* Split each dim that exceeds its axis by its least divisor, moving the divisor
   to the next axis. *)
let split_dims dims max_sizes =
  let fits d m =
    match d with
    | Int d -> d <= m
    | Sym u -> Z.leq (Dtype.Value.to_z (vmax (simplify u))) (Z.of_int m)
  in
  let rec fit ds ms =
    match (ds, ms) with d :: ds, m :: ms -> fits d m && fit ds ms | _ -> true
  in
  if fit dims max_sizes then dims
  else
    let a =
      Array.of_list (dims @ List.init (3 - List.length dims) (fun _ -> Int 1))
    in
    let n = Array.length a in
    for i = 0 to n - 1 do
      let m =
        match List.nth_opt max_sizes i with
        | Some m -> m
        | None -> cannot_limit dims max_sizes
      in
      let rec limit () =
        match a.(i) with
        | Int d when d > m ->
            let last =
              int_of_float (Float.ceil (Float.sqrt (float_of_int d)))
            in
            let rec least k =
              if k > last then 1 else if d mod k = 0 then k else least (k + 1)
            in
            let div = least 2 in
            if div = 1 then cannot_limit dims max_sizes;
            a.(i) <- Int (d / div);
            let next = (i + 1) mod n in
            a.(next) <- Sint.(a.(next) * Int div);
            limit ()
        | Int _ -> ()
        (* A symbolic size that may exceed its bound cannot be split. *)
        | Sym _ as d -> if not (fits d m) then cannot_limit dims max_sizes
      in
      limit ()
    done;
    let dims = Array.to_list a in
    match a.(2) with Int 1 -> List.filteri (fun i _ -> i < 2) dims | _ -> dims

(* The product of the dims after each dim. *)
let rec suffix_prods = function
  | [] -> []
  | _ :: rest -> Sint.prod rest :: suffix_prods rest

let rec grouped_dims ?(reverse = false) prefix dims max_sizes =
  if reverse then List.rev (grouped_dims prefix (List.rev dims) max_sizes)
  else
    let limited =
      match max_sizes with
      | None -> dims
      | Some max_sizes ->
          let limited =
            match group_dims dims max_sizes with
            | Some (_ :: _ as grouped) -> grouped
            | _ -> dims
          in
          if List.compare_lengths limited max_sizes > 0 then
            cannot_limit dims max_sizes;
          if List.equal Sint.equal limited dims then split_dims dims max_sizes
          else limited
    in
    let raw_idxs =
      List.mapi (fun i s -> special s (prefix ^ string_of_int i)) limited
    in
    let flat =
      List.fold_left2
        (fun acc idx p -> Sint.(acc + (Sym idx * p)))
        (Int 0) raw_idxs (suffix_prods limited)
    in
    List.mapi
      (fun i (d, p) ->
        let q = Sint.(flat // p) in
        simplify (sint_to_uop (if i = 0 then q else Sint.(q % d))))
      (List.combine dims (suffix_prods dims))

let add_gpudims (r : Renderer.t) s =
  let s_topo = toposort s in
  match arg s with
  | Kernel _ when not (List.exists (fun x -> op x = Op.Special) s_topo) -> (
      let all_ranges = Hashtbl.create 8 in
      List.iter
        (fun x ->
          if op x = Op.Range then Hashtbl.replace all_ranges (axis_id x) x)
        s_topo;
      let range id = Hashtbl.find all_ranges id in
      let dims_of types =
        Hashtbl.fold
          (fun id x acc ->
            if List.mem (axis_type x) types then id :: acc else acc)
          all_ranges []
        |> List.sort (List.compare Int.compare)
      in
      let global_dims = dims_of [ Axis_type.Global ] in
      let local_dims = dims_of [ Axis_type.Warp; Axis_type.Local ] in
      match (global_dims, local_dims) with
      | [], [] -> None
      | _ ->
          let shape dims =
            List.map (fun id -> ssimplify (nth (range id) 0)) dims
          in
          let global_shape = shape global_dims
          and local_shape = shape local_dims in
          (* A warp keeps its own axis, so no other dim folds into it. *)
          let local_max =
            match (local_dims, local_shape, r.local_max) with
            | l0 :: _, w :: _, _ :: rest
              when axis_type (range l0) = Axis_type.Warp ->
                Z.to_int (dim_max w) :: rest
            | _ -> r.local_max
          in
          let local_idxs = grouped_dims "lidx" local_shape (Some local_max) in
          let hw_local =
            List.filter_map
              (fun u ->
                if op u = Op.Special then
                  Some (Z.to_int (dim_max (Sym (nth u 0))))
                else None)
              local_idxs
          in
          let global_max =
            match r.global_prod_max with
            | None -> r.global_max
            | Some prod_max ->
                let rec mins gs ps ls =
                  match (gs, ps, ls) with
                  | g :: gs, p :: ps, l :: ls -> min g (p / l) :: mins gs ps ls
                  | _ -> []
                in
                mins
                  (if r.global_max = [] then prod_max else r.global_max)
                  prod_max
                  (hw_local @ [ 1; 1; 1 ])
          in
          let idxs =
            grouped_dims ~reverse:true "gidx" global_shape (Some global_max)
            @ local_idxs
          in
          let subs = Tbl.create 16 in
          let axes = global_dims @ local_dims in
          List.iter
            (fun x ->
              (* A global store that does not use every thread index is masked
                 to the threads whose unused indices are 0. *)
              (if op x = Op.Store then
                 let idx = nth x 0 in
                 match src idx with
                 | buf :: _ when addrspace buf = Some Dtype.Global -> (
                     let missing =
                       List.filter_map
                         (fun id ->
                           let rng = range id in
                           if Nodes.mem rng (ranges idx) then None else Some rng)
                         local_dims
                     in
                     match missing with
                     | [] -> ()
                     | m :: ms ->
                         if List.length (src idx) <> 2 then
                           invalid_arg
                             "a global store's index misses a thread index but \
                              has more than one index";
                         let mask =
                           uprod
                             (eq m (int 0))
                             (List.map (fun x -> eq x (int 0)) ms)
                         in
                         Tbl.replace subs idx
                           (replace idx ~src:[ buf; valid (nth idx 1) mask ]))
                 | _ -> ());
              if op x = Op.Range then
                match
                  List.find_index (List.equal Int.equal (axis_id x)) axes
                with
                | Some i -> Tbl.replace subs x (List.nth idxs i)
                | None -> ())
            s_topo;
          Some (substitute s (Tbl.fold (fun k v acc -> (k, v) :: acc) subs [])))
  | _ -> None

let pm_device_to_var =
  Pattern_matcher.v
    [
      Pattern_matcher.rule (Upat.op Op.Range ~name:"r") (fun m ->
          let r = m "r" in
          if axis_type r = Axis_type.Device then
            Some
              (variable ~dtype:(dtype r) "_device_num" (Dtype.Value.of_int 0)
                 (vmax r))
          else None);
      Pattern_matcher.rule (Upat.op Op.End ~name:"e") (fun m ->
          let e = m "e" in
          let device_num s =
            op s = Op.Param
            &&
            match arg s with
            | Param p -> p.name = Some "_device_num"
            | _ -> false
          in
          match src e with
          | body :: ends when List.exists device_num ends ->
              Some
                (replace e
                   ~src:(body :: List.filter (fun s -> op s <> Op.Param) ends))
          | _ -> None);
    ]

let pm_add_gpudims =
  Pattern_matcher.append
    (Pattern_matcher.v
       [
         Pattern_matcher.rule_ctx (Upat.op Op.Sink ~name:"s") (fun r m ->
             add_gpudims r (m "s"));
       ])
    (Pattern_matcher.with_ctx pm_device_to_var)
