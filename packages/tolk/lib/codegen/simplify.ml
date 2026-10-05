(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Divandmod
module V = Dtype.Value

let rule_ctx = Pattern_matcher.rule_ctx

let dedup l =
  Helpers.dedup
    (module struct
      type t = Ops.t

      let equal = ( == )
      let hash = hash
    end)
    l

let flatten_range r =
  let off = Option.get (range_start (op r)) in
  match List.drop off (src r) with
  | [] -> None
  | rngs ->
      let flat =
        List.concat_map
          (fun s -> if op s = Op.Range then [ s ] else Nodes.to_list (ranges s))
          rngs
      in
      Some (replace r ~src:(List.take off (src r) @ dedup flat))

let pm_flatten_range =
  Pattern_matcher.v
    (fun () -> [
      rule
        (Upat.v ~op:(Op.Set.of_list [ Op.Reduce; Op.End ]) ~name:"r" ())
        (fun m -> flatten_range (m "r"));
    ])

(* Index and range arithmetic uses floor division and remainder until the late
   rewrites. *)
let count_divmod x =
  Nodes.fold
    (fun u n -> if op u = Op.Floordiv || op u = Op.Floormod then n + 1 else n)
    (backward_slice ~calls:Skip x)
    0

let merge_rewrite =
  Pattern_matcher.concat
    [
      pm_substitute;
      Pattern_matcher.with_ctx Symbolic.symbolic;
      Pattern_matcher.with_ctx pm_flatten_range;
    ]

let simplify_merge_adjacent u =
  let reduce_ranges =
    Nodes.fold
      (fun x acc -> if op x = Op.Reduce then ranges x :: acc else acc)
      (backward_slice_with_self ~calls:Skip u)
      []
  in
  let ended = ended_ranges u in
  (* An end merges only adjacent ranges; a reduction tries every pair. *)
  let pairs =
    if op u = Op.End then
      let rec adjacent = function
        | r0 :: (r1 :: _ as rest) -> (r0, r1) :: adjacent rest
        | _ -> []
      in
      adjacent ended
    else
      List.concat
        (List.mapi
           (fun i r0 ->
             List.filteri (fun j _ -> j <> i) ended
             |> List.map (fun r1 -> (r0, r1)))
           ended)
  in
  List.find_map
    (fun (r0, r1) ->
      if
        Axis_type.equal (axis_type r0) (axis_type r1)
        && List.for_all
             (fun rngs -> Nodes.mem r0 rngs = Nodes.mem r1 rngs)
             reduce_ranges
      then begin
        let s0 = nth r0 0 and s1 = nth r1 0 in
        let new_range = replace r0 ~src:[ O.(s0 * s1) ] in
        let subs = Tbl.create 2 in
        Tbl.replace subs r0 O.(new_range // s1);
        Tbl.replace subs r1 O.(new_range % s1);
        let nidx =
          graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:subs u
            (After_sources merge_rewrite)
        in
        (* Return after one merge, so that the next rewrite merges the new
           ranges, not stale pairs of the old ones. *)
        if count_divmod nidx <= count_divmod u then Some nidx else None
      end
      else None)
    pairs

let mark_gated ctx idx =
  let guards = Tbl.create 4 in
  let x =
    match src idx with
    | _ :: v :: _ when op v = Op.Where ->
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
  (* The greatest bound [c] over the guards [r < c]... *)
  Tbl.iter
    (fun r c ->
      match Tbl.find_opt ctx r with
      | Some b when not V.(vmax b < vmax c) -> ()
      | _ -> Tbl.replace ctx r c)
    guards;
  (* ...but a range that is ever unguarded cannot shrink. *)
  List.iter
    (fun r -> if not (Tbl.mem guards r) then Tbl.replace ctx r (nth r 0))
    (Nodes.to_list (ranges x))

let do_substitute ctx x sub =
  match arg x with
  (* Only the kernel's root: rewriting a nested sink would leave the binders of
     its enclosing end unchanged. *)
  | No_arg -> None
  | _ ->
      let ret =
        substitute ~calls:Skip ~pass:Fixed_point x
          (Tbl.fold (fun k v acc -> (k, sub k v) :: acc) ctx [])
      in
      Tbl.reset ctx;
      if ret == x then None else Some (simplify ret)

let pm_simplify_ranges =
  Pattern_matcher.v
    (fun () -> [
      rule
        (Upat.v ~op:(Op.Set.of_list [ Op.End; Op.Reduce ]) ~name:"u" ())
        (fun m -> simplify_merge_adjacent (m "u"));
      rule_ctx (Upat.op Op.Index ~name:"idx") (fun ctx m ->
          mark_gated ctx (m "idx");
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
         && (not (List.mem (axis_type r) Axis_type.[ Warp; Device ]))
         && op (nth r 0) = Op.Const
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

let reduce_unparented red =
  match arg red with
  | Reduce { op = (Op.Add | Op.Max | Op.Mul) as rop; _ } -> (
      let value, rngs = (nth red 0, List.tl (src red)) in
      if not (List.for_all (fun x -> op x = Op.Range) rngs) then
        invalid_arg "some reduce srcs aren't ranges";
      let within = ranges value in
      match List.partition (fun x -> Nodes.mem x within) rngs with
      | _, [] -> None
      | parented, unparented ->
          let ret =
            if List.is_empty parented then value
            else replace red ~src:(value :: parented)
          in
          Some
            (List.fold_left
               (fun ret r ->
                 match rop with
                 | Op.Add -> O.(ret * nth r 0)
                 | Op.Mul -> pow ret (nth r 0)
                 | _ -> ret)
               ret unparented))
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
          (* Lift x + y out of a reduction on a comparison. *)
          rule
            Upat.O.(var "x" + var "y" < var "c")
            (fun m ->
              let x = m "x" and y = m "y" and c = m "c" in
              if no_range y && no_range c && solves_sum x y c then
                Some O.(x < c - y)
              else None);
          (* Lift x * y out of a reduction, where nothing wraps. *)
          rule
            Upat.O.(var "x" * var "y" < var "c")
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
                && Dtype.is_int (dtype y)
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
          (* AND on a selection. *)
          rule
            (sum_any
               (Upat.where
                  Upat.O.(Upat.op Op.Param ~name:"x" land var "y")
                  (var "c") zero))
            (fun m ->
              Some O.(over (m "r") (where (m "y") (m "c") (int 0)) * m "x"));
          (* A multiplication by a boolean cast, for integers: a float product
             by 0 is NaN at an infinity and -0. at a negative x. *)
          rule
            Upat.O.(
              var ~dtype:(Dtype.Bool :: Dtype.Weak_int :: Dtype.ints) "x"
              * Upat.f (var ~dtype:[ Dtype.Bool ] "gate") Op.Cast)
            (fun m -> Some (where (m "gate") (m "x") (int 0)));
        ]);
      Symbolic.symbolic;
    ]

let pm_reduce_load_collapse =
  let var = Upat.var in
  Pattern_matcher.concat
    [
      pm_reduce_collapse;
      Pattern_matcher.v
        (fun () -> [
          (* Lift x + y out of a reduction on an inequality, where no cast
             narrows. *)
          rule
            Upat.O.(Upat.or_casted ~name:"s" (var "x" + var "y") <> var "c")
            (fun m ->
              let x = m "x" and y = m "y" and c = m "c" in
              let dt = dtype y in
              if
                no_range y && no_range c && solves_sum x y c
                && Dtype.can_lossless_cast dt (dtype (m "s"))
                && exact dt [ vmin c; vmax c ]
              then Some O.(x <> cast c dt - y)
              else None);
          (* A sum of a load gated on its index equal to the range is the load
             at that index. *)
          rule
            (Upat.reduce ~op:Op.Add
               (Upat.where
                  Upat.O.(
                    var "idx" <> Upat.or_casted (Upat.op Op.Range ~name:"r"))
                  (Upat.O.int 0) (var "expr"))
               [ var "r" ])
            (fun m ->
              let r = m "r" and expr = m "expr" in
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
          toposort ~calls:Enter ~gate:(fun x -> Nodes.mem r (ranges x)) u
        in
        if List.exists (fun x -> op x = Op.Store || op x = Op.Reduce) included
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
                         || List.mem (op s) Op.[ Const; Param; Buffer; Alloc ]
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
let pm_load_collapse =
  Pattern_matcher.v
    (fun () -> [
      rule
        (Upat.op Op.Reduce ~name:"red"
           ~arg:(Reduce { op = Op.Add; num_axes = 0 })
           ~src:[ Upat.var "u"; Upat.wild ])
        (fun m -> reduce_collapse ~pm:pm_reduce_load_collapse (m "red") (m "u"));
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
