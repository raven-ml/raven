(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Shape

let dim_max = function
  | Int d -> Bigint.of_int d
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
    | d :: ds, m :: ms -> Bigint.leq (dim_max d) (Bigint.of_int m) && fit ds ms
    | [], _ -> true
    | _ :: _, [] -> false
  in
  let rec merge ds ms =
    match (ds, ms) with
    | d0 :: (d1 :: rest as ds), m :: ms ->
        if Bigint.leq Bigint.(dim_max d0 * dim_max d1) (Bigint.of_int m) then
          Some (Sint.(d0 * d1) :: rest)
        else Option.map (List.cons d0) (merge ds ms)
    | _ -> None
  in
  if fit dims max_sizes then Some dims
  else
    Option.bind (merge dims max_sizes) (fun dims -> group_dims dims max_sizes)

(* Sizes as the grouping computes them: exact integers, whose intermediate
   products may pass [int], or nodes. *)
type size = N of Bigint.t | U of t

let size = function Int n -> N (Bigint.of_int n) | Sym u -> U u
let node = function N z -> const (`Int z) | U u -> u

let ( *! ) a b =
  match (a, b) with N x, N y -> N Bigint.(x * y) | _ -> U O.(node a * node b)

let prod = List.fold_left ( *! ) (N Bigint.one)

(* Split each dim that exceeds its axis by its least divisor, moving the divisor
   to the next axis. *)
let split_dims dims max_sizes =
  let fits d m =
    match d with
    | N d -> Bigint.leq d (Bigint.of_int m)
    | U u -> Bigint.leq (Dtype.Value.to_z (vmax (simplify u))) (Bigint.of_int m)
  in
  let rec fit ds ms =
    match (ds, ms) with d :: ds, m :: ms -> fits d m && fit ds ms | _ -> true
  in
  let sizes = List.map size dims in
  if fit sizes max_sizes then sizes
  else
    let a =
      Array.of_list (sizes @ List.init (3 - List.length dims) (fun _ -> N Bigint.one))
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
        | N d when Bigint.gt d (Bigint.of_int m) ->
            let last = Bigint.of_float (Float.ceil (Float.sqrt (Bigint.to_float d))) in
            let rec least k =
              if Bigint.gt k last then Bigint.one
              else if Bigint.(equal (rem d k) zero) then k
              else least (Bigint.succ k)
            in
            let div = least (Bigint.of_int 2) in
            if Bigint.equal div Bigint.one then cannot_limit dims max_sizes;
            a.(i) <- N (Bigint.div d div);
            let next = (i + 1) mod n in
            a.(next) <- a.(next) *! N div;
            limit ()
        | N _ -> ()
        (* A symbolic size that may exceed its bound cannot be split. *)
        | U _ as d -> if not (fits d m) then cannot_limit dims max_sizes
      in
      limit ()
    done;
    let sizes = Array.to_list a in
    match a.(2) with
    | N z when Bigint.equal z Bigint.one -> List.filteri (fun i _ -> i < 2) sizes
    | _ -> sizes

(* The product of the sizes after each size. *)
let rec suffix_prods = function
  | [] -> []
  | _ :: rest -> prod rest :: suffix_prods rest

let rec grouped_dims ?(reverse = false) prefix dims max_sizes =
  if reverse then List.rev (grouped_dims prefix (List.rev dims) max_sizes)
  else
    let limited =
      match max_sizes with
      | None -> List.map size dims
      | Some max_sizes ->
          let limited =
            match group_dims dims max_sizes with
            | Some (_ :: _ as grouped) -> grouped
            | _ -> dims
          in
          if List.compare_lengths limited max_sizes > 0 then
            cannot_limit dims max_sizes;
          if List.equal Sint.equal limited dims then split_dims dims max_sizes
          else List.map size limited
    in
    (* Keep hardware axes as ranges through index lowering. The axis id is the
       hardware dimension. *)
    let axis_type =
      if prefix = "gidx" then Axis_type.Global else Axis_type.Local
    in
    let raw_idxs =
      List.mapi (fun i s -> range ~axis_type (Sym (node s)) [ i ]) limited
    in
    let flat =
      List.fold_left2
        (fun acc idx p -> O.(acc + (idx * node p)))
        (int 0) raw_idxs (suffix_prods limited)
    in
    let sizes = List.map size dims in
    List.mapi
      (fun i (d, p) ->
        let q = O.(flat // node p) in
        simplify (if i = 0 then q else O.(q % node d)))
      (List.combine sizes (suffix_prods sizes))

let group_gpudims (r : Renderer.t) s =
  let s_topo = toposort ~calls:Enter s in
  match arg s with
  | Kernel _ when not (List.exists (fun x -> op x = Op.Special) s_topo) -> (
      let all_ranges = Hashtbl.create 8 in
      List.iter
        (fun x ->
          if op x = Op.Range then
            Hashtbl.replace all_ranges (axis_type x, axis_id x) x)
        s_topo;
      let range key = Hashtbl.find all_ranges key in
      let ranges_of types =
        Hashtbl.fold
          (fun _ x acc ->
            if List.mem (axis_type x) types then x :: acc else acc)
          all_ranges []
      in
      let key x = (axis_type x, axis_id x) in
      let global_dims =
        List.map key
          (List.sort compare_range_arg (ranges_of [ Axis_type.Global ]))
      in
      (* WARP maps to hardware dimension zero, independently of nesting
         order. *)
      let local_dims =
        let order x = (axis_type x <> Axis_type.Warp, axis_id x) in
        List.map key
          (List.sort
             (fun a b -> Stdlib.compare (order a) (order b))
             (ranges_of [ Axis_type.Warp; Axis_type.Local ]))
      in
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
                Bigint.to_int (dim_max w) :: rest
            | _ -> r.local_max
          in
          let local_idxs = grouped_dims "lidx" local_shape (Some local_max) in
          let hw_local =
            List.filter_map
              (fun u ->
                if op u = Op.Range then
                  Some (Bigint.to_int (dim_max (Sym (nth u 0))))
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
          let subs = Tbl.create 16 and masks = Tbl.create 4 in
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
                         Tbl.replace masks idx
                           (replace idx ~src:[ buf; valid (nth idx 1) mask ]))
                 | _ -> ());
              if op x = Op.Range then
                match List.find_index (( = ) (key x)) axes with
                | Some i -> Tbl.replace subs x (List.nth idxs i)
                | None -> ())
            s_topo;
          let pairs t = Tbl.fold (fun k v acc -> (k, v) :: acc) t [] in
          (* Hardware ids may coincide with logical ids: replace ranges
             simultaneously, including those in the new masks. *)
          let masked =
            substitute ~calls:Skip ~pass:Fixed_point s (pairs masks)
          in
          Some (substitute ~calls:Skip ~pass:Once masked (pairs subs)))
  | _ -> None

(* Run once: grouping creates new GLOBAL/LOCAL ranges, which must not be grouped
   again. *)
let pm_group_gpudims =
  Pattern_matcher.v (fun () -> [
      Pattern_matcher.rule_ctx (Upat.op Op.Sink ~name:"s") (fun r m ->
          group_gpudims r (m "s"));
    ])

let pm_range_to_special =
  Pattern_matcher.append
    (Pattern_matcher.v
       (fun () -> [
         Pattern_matcher.rule (Upat.op Op.Range ~name:"r") (fun m ->
             let r = m "r" in
             let prefix =
               match axis_type r with
               | Axis_type.Global -> Some "gidx"
               | Local -> Some "lidx"
               | _ -> None
             in
             Option.map
               (fun prefix ->
                 let id = axis_id r in
                 let last = List.nth id (List.length id - 1) in
                 replace r ~op:Op.Special
                   ~arg:(String (prefix ^ string_of_int last)))
               prefix);
       ]))
    Linearizer.pm_split_ends
