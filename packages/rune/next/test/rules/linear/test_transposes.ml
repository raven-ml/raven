(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Reverse mode, one row of nx's operations at a time: the pullback against the
   tangent by the adjoint identity, a linear row's pullback against the row
   itself, the maps rune states whole, and the recorder's refusal of a tangent
   map that is not linear. *)

open Windtrap
module Rune = Rune_next.Rune
module Op = Nx.Op

let operands () = Nx.Ptree.(list tensor)

let direction seed x =
  Reference.direction (Random.State.make [| seed |]) (operands ()) x

let at gen =
  Gen.with_pp
    (fun ppf (i, seed) ->
      Format.fprintf ppf "%a@ seed %d" Case.pp_instance i seed)
    (Gen.pair gen (Gen.int_range 0 1_000_000))

(* [pairing a b] is [Re ⟨a, b⟩] and [Σ |a| |b|], the size against which its
   rounding is measured. *)
let pairing a b =
  (Reference.dot (operands ()) a b, Reference.magnitude (operands ()) a b)

(* [adjoint ~tol l r] checks that the pairings [l] and [r] agree to rounding. *)
let adjoint ~tol (l, lm) (r, rm) =
  let bound = tol *. (lm +. rm) in
  equal ~msg:"Re <w, J v> = Re <J* w, v>" (float (Float.max 1e-300 bound)) l r

let meta t = (Nx_dtype.to_string (Nx.dtype t), Nx.shape t)

(* [cotangents g v] checks that the pullback's result [g] has the operands'
   dtypes and shapes. *)
let cotangents g v =
  equal ~msg:"the pullback's dtypes and shapes"
    (list (pair string (array int)))
    (List.map meta v) (List.map meta g)

let tolerance (c : Case.t) =
  match c.row with Cholesky | Qr | Lu | Solve_triangular -> 1e-10 | _ -> 1e-12

(* The laws *)

let pullback_is_adjoint ~tol (Case.Instance i, seed) =
  let v = direction seed i.x in
  let y, jv = Rune.jvp (operands ()) (operands ()) i.f i.x v in
  let _, pullback = Rune.vjp (operands ()) (operands ()) i.f i.x in
  let w = direction (seed + 1) y in
  let g = pullback w in
  cotangents g v;
  adjoint ~tol (pairing w jv) (pairing g v)

let transpose_is_adjoint ~tol (Case.Instance i, seed) =
  match i.linear with
  | None -> ()
  | Some l ->
      let v = direction seed i.x in
      let lv = l v in
      let _, pullback = Rune.vjp (operands ()) (operands ()) i.f i.x in
      let w = direction (seed + 1) lv in
      let g = pullback w in
      cotangents g v;
      adjoint ~tol (pairing w lv) (pairing g v)

let close = Reference.close ~rel:1e-12 ~floor:1e-300 ()
let leaves x = Reference.leaves (operands ()) x

(* Maps rune states whole: a remat of the row, and a custom_vjp whose pullback
   is the row's. *)

let remat_is_the_row ~tol (Case.Instance i, seed) =
  let g = Rune.remat Nx.Ptree.(list tensor @-> returns (list tensor)) i.f in
  let v = direction seed i.x in
  let y, jv = Rune.jvp (operands ()) (operands ()) g i.x v in
  let _, pullback = Rune.vjp (operands ()) (operands ()) g i.x in
  let w = direction (seed + 1) y in
  adjoint ~tol (pairing w jv) (pairing (pullback w) v);
  let _, row = Rune.vjp (operands ()) (operands ()) i.f i.x in
  equal ~msg:"the row's pullback" close (leaves (row w)) (leaves (pullback w))

let custom_vjp_is_the_row (Case.Instance i, seed) =
  let rule a = Rune.vjp (operands ()) (operands ()) i.f a in
  let g = Rune.custom_vjp (operands ()) (operands ()) rule in
  let y, pullback = Rune.vjp (operands ()) (operands ()) g i.x in
  let _, row = Rune.vjp (operands ()) (operands ()) i.f i.x in
  let w = direction seed y in
  equal ~msg:"the row's pullback" close (leaves (row w)) (leaves (pullback w))

(* The recorder: a custom_jvp whose tangent map applies the row. *)

let linear_map_is_transposed (Case.Instance i, seed) =
  match i.linear with
  | None -> ()
  | Some l ->
      let g =
        Rune.custom_jvp (operands ()) (operands ()) (fun a -> (i.f a, l))
      in
      let _, pullback = Rune.vjp (operands ()) (operands ()) g i.x in
      let y, row = Rune.vjp (operands ()) (operands ()) i.f i.x in
      let w = direction seed y in
      equal ~msg:"the row's pullback" close
        (leaves (row w))
        (leaves (pullback w))

let nonlinear_map_raises row (Case.Instance i, _) =
  match i.linear with
  | Some _ -> ()
  | None ->
      let _, name =
        List.find
          (fun (r, _) -> Row.equal r row)
          (Row.issued (fun () -> i.f i.x))
      in
      raises
        (Invalid_argument
           (Printf.sprintf
              "Rune.vjp: a custom_jvp tangent map applies %s to a tangent; a \
               tangent map must be linear in its tangents"
              name))
        (fun () ->
          let g =
            Rune.custom_jvp (operands ()) (operands ()) (fun a -> (i.f a, i.f))
          in
          Rune.vjp (operands ()) (operands ()) g i.x)

(* The rows *)

let laws ~count (c : Case.t) =
  let prop ?(gen = c.finite) name law = prop ~count name (at gen) law in
  let tol = tolerance c in
  match c.kind with
  | Case.Tangent ->
      List.concat
        [
          [
            prop "the pullback is the adjoint of the tangent map"
              (pullback_is_adjoint ~tol);
            prop "a linear operation's pullback is its adjoint"
              (transpose_is_adjoint ~tol);
            prop "a remat of it has its derivatives" (remat_is_the_row ~tol);
            prop "a custom_vjp whose pullback is its own has its pullback"
              custom_vjp_is_the_row;
            prop "a tangent map that applies it linearly is transposed"
              linear_map_is_transposed;
            prop
              "a tangent map that applies it to tangents nonlinearly raises \
               naming it"
              (nonlinear_map_raises c.row);
          ];
          (match c.complex with
          | Some gen ->
              [
                prop ~gen
                  "on complex values the pullback is the adjoint of the \
                   tangent map"
                  (pullback_is_adjoint ~tol);
                prop ~gen
                  "on complex values a linear operation's pullback is its \
                   adjoint"
                  (transpose_is_adjoint ~tol);
                prop ~gen
                  "on complex values a custom_vjp whose pullback is its own \
                   has its pullback"
                  custom_vjp_is_the_row;
              ]
          | None -> []);
        ]
  | Case.Plain -> (
      match c.complex with
      | Some gen ->
          [
            prop ~gen
              "on complex values the pullback is the adjoint of the tangent map"
              (pullback_is_adjoint ~tol);
          ]
      | None -> [])
  | Case.Integer -> []

let rows ~count =
  List.filter_map
    (fun (c : Case.t) ->
      match laws ~count c with
      | [] -> None
      | ts -> Some (group (Row.name c.row) ts))
    Case.all

let () =
  exit
    (run "rune.next transposes"
       (rows ~count:10
       @ [
           group "edges" Edges.tests;
           group "compositions" Compositions.tests;
           group ~tags:[ "slow" ] "swept" (rows ~count:300);
         ]))
