(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Shape
module V = Dtype.Value

let rule_ctx = Pattern_matcher.rule_ctx

let flatten_range r =
  let off = Option.get (range_start (op r)) in
  let flat =
    List.concat_map
      (fun s -> if op s = Op.Range then [ s ] else Nodes.to_list (ranges s))
      (List.drop off (src r))
  in
  Some (replace r ~src:(List.take off (src r) @ dedup_nodes flat))

let pm_flatten_range =
  Pattern_matcher.v
    (fun () -> [
      rule
        (Upat.v ~op:(Op.Set.of_list [ Op.Reduce; Op.End ]) ~name:"r" ())
        (fun m -> flatten_range (m "r"));
    ])

let merge_rewrite =
  Pattern_matcher.concat
    [
      pm_substitute;
      Pattern_matcher.with_ctx Shape.symbolic;
      Pattern_matcher.with_ctx pm_flatten_range;
    ]

(* The operations that compute a value and nothing else: merging two ranges
   around them changes nothing but their indices. *)
let pure_ops =
  Op.Set.union Op.Set.elementwise
    (Op.Set.union Op.Set.irreducible (Op.Set.of_list Op.[ Index; Load; Stack ]))

let rec pairwise = function
  | r0 :: (r1 :: _ as rest) -> (r0, r1) :: pairwise rest
  | _ -> []

let permutations rs =
  List.concat_map
    (fun r0 ->
      List.filter_map (fun r1 -> if r0 == r1 then None else Some (r0, r1)) rs)
    rs

(* The pairs of ranges the node [x] closes that may merge. A reduction whose
   value has no effect merges any two; an end, or a reduction with effects, runs
   its ranges as loops nested by axis type, then identity, whatever order it
   lists them in, so only neighbours in that order merge. *)
let mergeable_pairs x =
  let pure () =
    Nodes.fold
      (fun y ok -> ok && (Op.Set.mem (op y) pure_ops || op y = Op.Reduce))
      (backward_slice ~calls:Skip x)
      true
  in
  match (op x, ended_ranges x) with
  | (Op.End | Op.Reduce), ([] | [ _ ]) -> []
  | Op.Reduce, ended when pure () -> permutations ended
  | (Op.End | Op.Reduce), ended ->
      pairwise (List.stable_sort compare_range_arg ended)
  | _ -> []

(* [merge_ranges u] merges, in the sink [u], two ranges [r0] and [r1] into one
   [n] of their sizes' product, [r0] becoming [n / size r1] and [r1] [n mod size
   r1], when each node that closes either may merge them, each node that is not
   pure runs inside both or neither, so that it runs as often as before, and the
   quotient and remainder fold away. Each rewrite makes one merge, so that the
   next pairs the new ranges, not stale ones. *)
let merge_ranges u =
  let slice = Nodes.to_list (backward_slice ~calls:Skip u) in
  let enders =
    List.filter_map
      (fun x ->
        if Option.is_some (range_start (op x)) then
          Some (ended_ranges x, mergeable_pairs x)
        else None)
      slice
  in
  let impure =
    lazy (List.filter (fun x -> not (Op.Set.mem (op x) pure_ops)) slice)
  in
  let same (a0, a1) (b0, b1) = a0 == b0 && a1 == b1 in
  let candidates =
    List.fold_left
      (fun acc p -> if List.exists (same p) acc then acc else p :: acc)
      []
      (List.concat_map snd enders)
  in
  let merges (r0, r1) =
    Axis_type.equal (axis_type r0) (axis_type r1)
    && (not (Nodes.mem r0 (ranges r1)))
    && (not (Nodes.mem r1 (ranges r0)))
    && List.for_all
         (fun (ended, pairs) ->
           (not (List.memq r0 ended || List.memq r1 ended))
           || List.exists (same (r0, r1)) pairs)
         enders
    && List.for_all
         (fun x -> Nodes.mem r0 (ranges x) = Nodes.mem r1 (ranges x))
         (Lazy.force impure)
  in
  let merge (r0, r1) =
    let s1 = nth r1 0 in
    let n = replace r0 ~src:[ simplify O.(nth r0 0 * s1) ] in
    let q = O.(n // s1) and m = O.(n % s1) in
    let subs = Tbl.create 2 in
    Tbl.replace subs r0 q;
    Tbl.replace subs r1 m;
    let ret =
      graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:subs u
        (After_sources merge_rewrite)
    in
    let left = backward_slice ~calls:Skip ret in
    if Nodes.mem q left || Nodes.mem m left then None else Some ret
  in
  List.find_map
    (fun p -> if merges p then merge p else None)
    (List.rev candidates)

(* [mark_gated ctx u] records in [ctx] the size each range may shrink to, for an
   index or an effect [u]. *)
let mark_gated ctx u =
  (* A store is the only effect with an index of its own. *)
  let idx = if op u = Op.Store then nth u 0 else u in
  let guards = Tbl.create 4 in
  let x =
    match src idx with
    | _ :: v :: _ when op idx = Op.Index && op v = Op.Where ->
        List.iter
          (fun g ->
            match src g with
            | [ r; c ]
              when op g = Op.Cmplt && op r = Op.Range && op c = Op.Const ->
                Tbl.replace guards r c
            | _ -> ())
          (split_uop (get_valid v) Op.And);
        get_idx v
    | _ -> idx
  in
  (* The greatest bound [c] over the guards [r < c], within the range's size,
     which may be a variable... *)
  Tbl.iter
    (fun r c ->
      match Tbl.find_opt ctx r with
      | Some b when not V.(vmax b < vmax c) -> ()
      | _ -> Tbl.replace ctx r (minimum c (nth r 0)))
    guards;
  (* ...but a range that is ever unguarded cannot shrink, and an effect runs at
     every step of a range its index does not guard. *)
  let unguarded = if op u = Op.Index then x else u in
  List.iter
    (fun r -> if not (Tbl.mem guards r) then Tbl.replace ctx r (nth r 0))
    (Nodes.to_list (ranges unguarded))

let do_substitute ctx x sub =
  match arg x with
  (* Only the kernel's root: rewriting a nested sink would leave the binders of
     its enclosing end unchanged. *)
  | No_arg -> None
  | _ ->
      (* The context keeps what earlier rounds recorded, since the nodes a round
         leaves are not visited again; a range it no longer holds is left. *)
      let within = backward_slice ~calls:Skip x in
      let ret =
        substitute ~calls:Skip ~pass:Fixed_point x
          (Tbl.fold
             (fun k v acc ->
               if Nodes.mem k within then (k, sub k v) :: acc else acc)
             ctx [])
      in
      if ret == x then None else Some (simplify ret)

let pm_simplify_ranges =
  let marked = Op.Set.union (Op.Set.of_list [ Op.Index ]) Op.Set.effectful in
  Pattern_matcher.v (fun () ->
      [
        rule (Upat.op Op.Sink ~name:"u") (fun m -> merge_ranges (m "u"));
        rule_ctx (Upat.v ~op:marked ~name:"u" ()) (fun ctx m ->
            mark_gated ctx (m "u");
            None);
        (* Reduction ranges cannot shrink. *)
        rule_ctx (Upat.op Op.Reduce ~name:"red") (fun ctx m ->
            List.iter
              (fun r -> Tbl.replace ctx r (nth r 0))
              (List.tl (src (m "red")));
            None);
        rule_ctx (Upat.op Op.Sink ~name:"x") (fun ctx m ->
            do_substitute ctx (m "x") (fun r c -> replace r ~src:[ c ]));
      ])

let mark_range_mod ctx r c =
  (* A range that is not looped over cannot be split. *)
  match value c with
  | `Int n
    when (not (Tbl.mem ctx r))
         && axis_type r <> Axis_type.Warp
         && is_literal (nth r 0)
         && Option.is_some (divides (nth r 0) n) ->
      Tbl.replace ctx r c
  | _ -> ()

let split k v =
  let axis_id = axis_id k and axis_type = axis_type k in
  let part i size =
    replace k ~src:[ size ]
      ~arg:(Range { axis_id = axis_id @ [ i ]; axis_type })
  in
  O.((part 0 (nth k 0 // v) * v) + part 1 v)

let pm_split_ranges =
  Pattern_matcher.v
    (fun () -> [
      rule_ctx
        Upat.O.(Upat.op Op.Range ~name:"r" % Upat.cvar "c")
        (fun ctx m ->
          mark_range_mod ctx (m "r") (m "c");
          None);
      rule_ctx (Upat.op Op.Sink ~name:"x") (fun ctx m ->
          do_substitute ctx (m "x") split);
    ])

(* Reductions *)

let no_range u = not (op_in_backward_slice_with_self ~calls:Skip u [ Op.Range ])

(* A reduction over ranges its value ignores folds the value once per run of
   them, [n] times, [n] the product of their sizes: a sum multiplies it by [n],
   a product raises it to [n], and a maximum, or a bool sum or product (an or,
   an and), which are idempotent, is the value. Ranges that run no time, as a
   variable's of size 0, make the reduction its identity, where the sum of an
   infinity would be NaN and the maximum the value. *)
let reduce_unparented red =
  match arg red with
  | Reduce { op = (Op.Add | Op.Max | Op.Mul) as rop; _ } -> (
      let value, rngs = (nth red 0, List.tl (src red)) in
      if not (List.for_all (fun x -> op x = Op.Range) rngs) then
        invalid_arg "some reduce srcs aren't ranges";
      let within = ranges value in
      match List.partition (fun x -> Nodes.mem x within) rngs with
      | _, [] -> None
      | parented, r :: rs ->
          let ret =
            if List.is_empty parented then value
            else replace red ~src:(value :: parented)
          in
          let n = uprod (nth r 0) (List.map (fun r -> nth r 0) rs) in
          let ret =
            match rop with
            | _ when Dtype.equal (dtype red) Dtype.Bool -> ret
            | Op.Add -> O.(ret * n)
            | Op.Mul -> pow ret n
            | _ -> ret
          in
          let identity = identity_element rop (dtype red) in
          Some (where O.(int 0 < n) ret (const_like ret identity)))
  | _ -> None

let pm_reduce_unparented =
  Pattern_matcher.v
    (fun () -> [
      rule (Upat.op Op.Reduce ~name:"red") (fun m ->
          reduce_unparented (m "red"));
    ])

(* The sum of [value] over the part of [r] within [lower, upper). A float sum of
   no terms is 0 whatever the value, where 0 times an infinity is NaN. *)
let sum_between ?lower ?upper r value =
  if not (no_range value) then None
  else
    let size = nth r 0 in
    let hi = match upper with Some u -> minimum u size | None -> size in
    let lo =
      match lower with
      | Some l -> maximum l (int 0)
      | None -> const_like r (`Int Bigint.zero)
    in
    let count = maximum O.(hi - lo) (int 0) in
    if Dtype.is_float (dtype value) then
      Some
        (where
           O.(int 0 < count)
           O.(count * value)
           (const_like value (`Float 0.)))
    else Some O.(count * value)

(* Solving [x + y] against [c] for [x] computes [x + y] and [c - y], which is
   exact for integers where neither wraps. *)
let solves_sum x y c =
  Dtype.is_int (dtype y)
  && exact (dtype y)
       V.[ vmin x + vmin y; vmax x + vmax y; vmin c - vmax y; vmax c - vmin y ]

let pm_reduce_collapse =
  let var = Upat.var and zero = Upat.O.int 0 in
  let range = Upat.op Op.Range ~name:"r" in
  let sum p = Upat.reduce ~op:Op.Add p [ var "r" ] in
  let sum_any p = Upat.reduce ~name:"r" ~op:Op.Add ~allow_any_len:true p [] in
  let over r x = reduce x Op.Add (List.tl (src r)) in
  Pattern_matcher.concat
    [
      pm_reduce_unparented;
      Pattern_matcher.v
        (fun () -> [
          (* Lift x + y out of a reduction on a comparison, where x + y does not
             wrap: an unsigned sum wraps, so the rule is for signed ones. *)
          rule
            Upat.O.(
              Upat.op Op.Add
                ~dtype:(Dtype.Weak_int :: Dtype.sints)
                ~perm:[ var "x"; var "y" ]
              < var "c")
            (fun m ->
              let x = m "x" and y = m "y" and c = m "c" in
              if no_range y && no_range c && solves_sum x y c then
                Some O.(x < c - y)
              else None);
          (* Lift x * y out of a reduction, where nothing wraps, as for x + y. *)
          rule
            Upat.O.(
              Upat.op Op.Mul
                ~dtype:(Dtype.Weak_int :: Dtype.sints)
                ~perm:[ var "x"; var "y" ]
              < var "c")
            (fun m ->
              let x = m "x" and y = m "y" and c = m "c" in
              let products =
                List.concat_map
                  (fun a -> V.[ a * vmin y; a * vmax y ])
                  [ vmin x; vmax x ]
              and ceilings =
                V.[ vmin c + vmin y - of_int 1; vmax c + vmax y - of_int 1 ]
              in
              if
                no_range y && no_range c
                && V.(vmin y > of_int 0)
                && exact (dtype y) (products @ ceilings)
              then Some O.(x < (c + y - int 1) // y)
              else None);
          (* The sum over r in [0, n) of [lower <= r < upper] * value is max
             (min upper n - max lower 0) 0 * value. *)
          rule
            (sum (Upat.where Upat.O.(range < var "upper") (var "val") zero))
            (fun m -> sum_between ~upper:(m "upper") (m "r") (m "val"));
          rule
            (sum (Upat.where Upat.O.(range < var "lower") zero (var "val")))
            (fun m -> sum_between ~lower:(m "lower") (m "r") (m "val"));
          rule
            (sum
               (Upat.where
                  Upat.O.(
                    Upat.logical_not (var "r" < var "lower")
                    land (range < var "upper"))
                  (var "val") zero))
            (fun m ->
              sum_between ~lower:(m "lower") ~upper:(m "upper") (m "r")
                (m "val"));
          rule
            (sum_any Upat.O.(var "x" + var "y"))
            (fun m ->
              let r = m "r" in
              Some O.(over r (m "x") + over r (m "y")));
          (* A conjunct without the range moves out of a sum on a selection
             as a selection: the product by its cast is -0. at a negative sum
             and NaN at an infinite one. *)
          rule
            (sum_any
               (Upat.where
                  Upat.O.(Upat.op Op.Param ~name:"x" land var "y")
                  (var "c") zero))
            (fun m ->
              let inner = over (m "r") (where (m "y") (m "c") (int 0)) in
              Some (where (m "x") inner (int 0)));
          (* A multiplication by a boolean cast, for integers: a float product
             by 0 is NaN at an infinity and -0. at a negative x. *)
          rule
            Upat.O.(
              var ~dtype:(Dtype.Bool :: Dtype.Weak_int :: Dtype.ints) "x"
              * Upat.f (var ~dtype:[ Dtype.Bool ] "gate") Op.Cast)
            (fun m -> Some (where (m "gate") (m "x") (int 0)));
        ]);
      Shape.symbolic;
    ]

let pm_reduce_load_collapse =
  let var = Upat.var in
  Pattern_matcher.concat
    [
      pm_reduce_collapse;
      Pattern_matcher.v
        (fun () -> [
          (* Lift x + y out of a reduction on an equality, where no cast
             narrows. *)
          rule
            (Upat.eq
               Upat.O.(Upat.or_casted ~name:"s" (var "x" + var "y"))
               (var "c"))
            (fun m ->
              let x = m "x" and y = m "y" and c = m "c" in
              let dt = dtype y in
              if
                no_range y && no_range c
                && (not (no_range x))
                && solves_sum x y c
                && Dtype.can_lossless_cast dt (dtype (m "s"))
                && exact dt [ vmin c; vmax c ]
              then Some (eq x O.(cast c dt - y))
              else None);
          (* A sum of a load gated on its integer index equal to the range is
             the load at that index, where the index does not depend on the
             range and a cast of the range keeps its values. An index read at
             the range, as [b.(r) = r], selects any number of trips. *)
          rule
            (Upat.reduce ~op:Op.Add
               (Upat.where
                  (Upat.eq
                     (var ~dtype:(Dtype.Weak_int :: Dtype.ints) "idx")
                     (Upat.or_casted ~name:"rc" (Upat.op Op.Range ~name:"r")))
                  (var "expr") (Upat.O.int 0))
               [ var "r" ])
            (fun m ->
              let r = m "r" and expr = m "expr" in
              if
                overflows r (dtype (m "rc"))
                || reaches ~calls:Skip (m "idx") r
              then None
              else
                let idx = cast (m "idx") (dtype r) in
                let v = O.((idx >= int 0) land (idx < nth r 0)) in
                Some
                  (where v
                     (substitute ~calls:Skip ~pass:Fixed_point expr
                        [ (r, valid idx v) ])
                     (int 0)));
        ]);
    ]

let reduce_collapse ?(pm = pm_reduce_collapse) red u =
  let rec collapse u = function
    | [] -> Some u
    | r :: rest ->
        let included =
          toposort ~calls:Skip ~gate:(fun x -> Nodes.mem r (ranges x)) u
        in
        if
          List.exists
            (fun x -> Op.Set.mem (op x) Op.Set.effectful || op x = Op.Reduce)
            included
        then None
        else
          let inside = Tbl.create 64 in
          List.iter (fun x -> Tbl.replace inside x ()) included;
          let replaces = Tbl.create 16 and order = ref [] in
          List.iter
            (fun x ->
              List.iter
                (fun s ->
                  if
                    not
                      (Tbl.mem inside s
                      || (Tbl.mem replaces s
                         || (op s = Op.Const || Op.Set.mem (op s) Op.Set.defines)
                         )
                         [@mutate
                           off
                             "a constant, parameter or replaced node folds \
                              back unchanged"])
                  then begin
                    let name = Printf.sprintf "in%d" (Tbl.length replaces) in
                    let v = variable ~dtype:(dtype s) name (vmin s) (vmax s) in
                    Tbl.replace replaces s v;
                    order := (s, v) :: !order
                  end)
                (src x))
            included;
          let sink =
            graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:()
              (reduce
                 (substitute ~calls:Skip ~pass:Fixed_point u !order)
                 Op.Add [ r ])
              (After_sources pm)
          in
          if not (no_range sink) then None
          else
            collapse
              (substitute ~calls:Skip ~pass:Fixed_point sink
                 (List.map (fun (k, v) -> (v, k)) !order))
              rest
  in
  collapse u (List.tl (src red))

(* Remove a reduction without loads: arange and indexing. *)
let pm_reduce_simplify =
  Pattern_matcher.concat
    [
      pm_reduce_unparented;
      Pattern_matcher.v
        (fun () -> [
          rule
            (Upat.op Op.Reduce ~name:"red" ~allow_any_len:true
               ~arg:(Reduce { op = Op.Add; num_axes = 0 })
               ~src:[ Upat.var "u" ])
            (fun m -> reduce_collapse (m "red") (m "u"));
        ]);
    ]

let no_load u = not (op_in_backward_slice_with_self ~calls:Skip u [ Op.Index ])

(* Remove a reduction on a load, from indexing a tensor with another. *)
let load_collapse red =
  match (arg red, src red) with
  | Reduce { op = Op.Add; num_axes = 0 }, [ u; _ ] ->
      reduce_collapse ~pm:pm_reduce_load_collapse red u
  | _ -> None

let pm_load_collapse =
  Pattern_matcher.v
    (fun () -> [
      rule (Upat.op Op.Reduce ~name:"red") (fun m -> load_collapse (m "red"));
      (* No arithmetic on a loaded index, since it can overflow: this undoes the
         lifting of pm_reduce_load_collapse. *)
      rule
        Upat.O.(
          Upat.var ~dtype:[ Dtype.Weak_int ] "x" + Upat.var "y" < Upat.var "c")
        (fun m ->
          let x = m "x" and y = m "y" and c = m "c" in
          if no_load y && no_load c && not (no_load x) then Some O.(x < c - y)
          else None);
    ])
