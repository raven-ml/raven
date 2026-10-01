(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Forward mode, one row of nx's operations at a time: the primal, the tangent
   against a central difference and the closed form, a linear row's tangent
   against the row itself, the second order, and each row's edges. *)

open Windtrap

let operands () = Nx.Ptree.(list tensor)

(* A point and the seed of its directions. *)
let at gen =
  Gen.with_pp
    (fun ppf (i, seed) ->
      Format.fprintf ppf "%a@ seed %d" Case.pp_instance i seed)
    (Gen.pair gen (Gen.int_range 0 1_000_000))

let direction seed x =
  Reference.direction (Random.State.make [| seed |]) (operands ()) x

let shape_of t = (Nx_dtype.to_string (Nx.dtype t), Nx.shape t)
let meta = pair string (array int)

(* The laws *)

let primal (Case.Instance i, seed) =
  let expected = i.f i.x in
  let y, dy =
    Rune.jvp (operands ()) (operands ()) i.f i.x (direction seed i.x)
  in
  List.iter2 (equal ~msg:"jvp's primal" (Reference.exact ())) expected y;
  List.iter2
    (fun y dy ->
      equal ~msg:"the tangent's dtype and shape" meta (shape_of y) (shape_of dy))
    y dy;
  let y, _ = Rune.vjp (operands ()) (operands ()) i.f i.x in
  List.iter2 (equal ~msg:"vjp's primal" (Reference.exact ())) expected y

let no_tangent (Case.Instance i, seed) =
  let v = direction seed i.x in
  let _, dy = Rune.jvp (operands ()) (operands ()) i.f i.x v in
  List.iter
    (fun dy ->
      equal ~msg:"the tangent" (Reference.exact ()) (Nx.zeros_like dy) dy)
    dy;
  let y, pullback = Rune.vjp (operands ()) (operands ()) i.f i.x in
  let w =
    Reference.direction (Random.State.make [| seed + 1 |]) (operands ()) y
  in
  List.iter2
    (equal ~msg:"the gradient" (Reference.exact ()))
    (List.map Nx.zeros_like i.x)
    (pullback w)

let rows = list (Testable.make ~pp:Row.pp ~equal:Row.equal)

(* A case issues its row once, and besides it only its declared rows, or any
   when it goes on to compute forms of the result. *)
let issues_its_row (c : Case.t) (Case.Instance i, _) =
  let issued = List.map fst (Row.issued (fun () -> i.f i.x)) in
  match i.extra with
  | Some extra ->
      equal rows
        (List.sort Row.compare (c.row :: extra))
        (List.sort Row.compare issued)
  | None -> equal rows [ c.row ] (List.filter (Row.equal c.row) issued)

let central (eps, rel) (Case.Instance i, seed) =
  let v = direction seed i.x in
  let y, dy = Rune.jvp (operands ()) (operands ()) i.f i.x v in
  let scale =
    List.fold_left
      (fun m a -> Float.max m (Reference.norm a))
      1.
      (Reference.leaves (operands ()) y)
  in
  equal
    (Reference.close ~rel ~floor:(rel *. scale) ())
    (Reference.central (operands ()) (operands ()) ~eps i.f i.x v)
    (Reference.leaves (operands ()) dy)

let closed_form derivative (Case.Instance i, seed) =
  let v = direction seed i.x in
  let _, dy = Rune.jvp (operands ()) (operands ()) i.f i.x v in
  let expected =
    List.map2
      (fun x v ->
        Array.map2
          (fun x v -> Complex.mul { re = derivative x.Complex.re; im = 0. } v)
          (Reference.complexes x) (Reference.complexes v))
      i.x v
  in
  equal
    (Reference.close ~rel:1e-12 ())
    expected
    (Reference.leaves (operands ()) dy)

let linear_tangent (Case.Instance i, seed) =
  match i.linear with
  | None -> ()
  | Some l ->
      let v = direction seed i.x in
      let _, dy = Rune.jvp (operands ()) (operands ()) i.f i.x v in
      List.iter2 (equal ~msg:"the tangent" (Reference.exact ())) (l v) dy

(* The second order is a difference of tangents, at a step ten times the first
   order's and a tolerance a hundred times its. *)
let second_order (eps, rel) (Case.Instance i, seed) =
  let eps = 10. *. eps and rel = Float.min 0.1 (100. *. rel) in
  let v = direction seed i.x and u = direction (seed + 1) i.x in
  let along v x = snd (Rune.jvp (operands ()) (operands ()) i.f x v) in
  let _, uv = Rune.jvp (operands ()) (operands ()) (along v) i.x u in
  let _, vu = Rune.jvp (operands ()) (operands ()) (along u) i.x v in
  let scale =
    List.fold_left
      (fun m a -> Float.max m (Reference.norm a))
      1.
      (Reference.leaves (operands ()) (along v i.x))
  in
  equal
    (Reference.close ~rel ~floor:(rel *. scale) ())
    (Reference.central (operands ()) (operands ()) ~eps (along v) i.x u)
    (Reference.leaves (operands ()) uv);
  equal ~msg:"symmetric in its two directions"
    (Reference.close ~rel:(rel *. 1e-4) ~floor:1e-12 ())
    (Reference.leaves (operands ()) vu)
    (Reference.leaves (operands ()) uv)

(* The rows *)

let laws ~count (c : Case.t) =
  let prop ?(gen = c.smooth Case.float64) name law =
    prop ~count name (at gen) law
  in
  let dtype (Case.D d as t) =
    prop ~gen:(c.smooth t) (Format.asprintf "at %a" Nx.pp_dtype d) primal
  in
  let common =
    [
      prop "the case issues its own operation" (issues_its_row c);
      group
        "the primal is the operation's value, and the tangent has its dtype \
         and shape"
        (List.map dtype c.dtypes);
    ]
  in
  let complex =
    match c.complex with
    | Some gen ->
        [
          prop ~gen
            "on complex values the tangent agrees with a central difference"
            (central c.difference);
        ]
    | None -> []
  in
  let specific =
    match c.kind with
    | Case.Tangent ->
        List.concat
          [
            [
              prop "the tangent agrees with a central difference"
                (central c.difference);
            ];
            (match c.derivative with
            | Some d ->
                [
                  prop "the tangent is the derivative's closed form"
                    (closed_form d);
                ]
            | None -> []);
            [
              prop
                "a linear operation's tangent is the operation on the tangent"
                linear_tangent;
              prop
                "the second order agrees with a central difference of the \
                 tangent"
                (second_order c.difference);
            ];
          ]
    | Case.Plain | Case.Integer ->
        [ prop "it has no tangent and passes no cotangent" no_tangent ]
  in
  common @ specific @ complex

let rows ~count =
  List.map (fun (c : Case.t) -> group (Row.name c.row) (laws ~count c)) Case.all

let edges =
  List.filter_map
    (fun r ->
      match Jvp_edges.of_row r with
      | [] -> None
      | ts -> Some (group (Row.name r) ts))
    Row.all

let () =
  exit
    (run "rune jvp rules"
       (rows ~count:10
       @ [
           group "edges" edges;
           group "cumulative" Jvp_cumulative.tests;
           group "factorisations" Jvp_factorisations.tests;
           group ~tags:[ "slow" ] "swept" (rows ~count:500);
         ]))
