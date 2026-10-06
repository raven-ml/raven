(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Shape

let zero = `Int Bigint.zero

(* [holds dt w c] is [true] iff the constant [c] of type [w] converts to [dt]
   and back unchanged, its sign and bits included. *)
let holds dt w c =
  let there =
    match Dtype.const dt c with
    | #Dtype.value as v -> (Dtype.truncate dt v :> Dtype.const)
    | `Invalid -> `Invalid
  in
  Dtype.equal_const (Dtype.const w there) (Dtype.const w c)

(* The load reads the selection's other value where its gate fails, as a value
   of its own type, which the conversion after it then converts: only a value
   that conversion gives back unchanged moves. *)
let move_where_load m =
  let l = m "l" and a = m "a" and w = dtype (m "w") in
  let same = Dtype.equal (dtype l) w in
  let alt =
    if is_invalid a then Some (vconst_like l zero)
    else if op a = Op.Const then
      if same || holds (dtype l) w (value a) then Some (const_like l (value a))
      else None
    else if op a = Op.Cast && Dtype.equal (dtype (nth a 0)) (dtype l) then
      Some (nth a 0)
    else if same then Some a
    else if
      op a = Op.Cast && op (nth a 0) = Op.Const && holds (dtype l) w (value a)
    then Some (cast a (dtype l))
    else None
  in
  Option.map (fun alt -> cast (replace ~src:[ nth l 0; alt; nth l 2 ] l) w) alt

let gated idx = Upat.(where (var "gate") (var idx) (const `Invalid))

let mop =
  Upat.v
    ~op:(Op.Set.of_list [ Op.Index; Op.Shrink ])
    ~src:[ Upat.wild; gated "idx" ]
    ~allow_any_len:true ~name:"mop" ()

let ungated m =
  let mop = m "mop" in
  replace ~src:(nth mop 0 :: m "idx" :: List.drop 2 (src mop)) mop

let gated_load l = Upat.load Upat.wild [ Upat.wild; l ] ~name:"l"

let pm_move_gates_from_index =
  Pattern_matcher.(
    v
      (fun () -> [
        (* Two indices under one gate, as images index: this rule must come
           first, since the next ones would ungate only the first index. *)
        rule
          (Upat.load
             (Upat.index (Upat.var "buf") [ gated "idx_y"; gated "idx_x" ])
             [] ~name:"l")
          (fun m ->
            let idx = index (m "buf") [ m "idx_y"; m "idx_x" ] in
            Some (load idx [ vconst_like (m "l") zero; m "gate" ]));
        rule
          (Upat.store
             (Upat.index (Upat.var "buf") [ gated "idx_y"; gated "idx_x" ])
             [ Upat.var "data" ])
          (fun m ->
            let idx = index (m "buf") [ m "idx_y"; m "idx_x" ] in
            Some (store ~gate:(m "gate") idx (m "data")));
        (* A gated load reads 0 where its gate fails. *)
        rule (Upat.load mop [] ~name:"l") (fun m ->
            Some (load (ungated m) [ vconst_like (m "l") zero; m "gate" ]));
        rule
          (Upat.store mop [ Upat.var "data" ])
          (fun m -> Some (store ~gate:(m "gate") (ungated m) (m "data")));
        (* A where around a gated load becomes the load's alternate value. *)
        rule
          (Upat.named "w"
             Upat.(
               where (var "gate")
                 (or_casted (gated_load (var "gate" ~dtype:[ Dtype.Bool ])))
                 (var "a")))
          move_where_load;
        rule
          (Upat.named "w"
             Upat.(
               where (var "gate") (var "a")
                 (or_casted
                    (gated_load
                       (bitwise_not (var "gate" ~dtype:[ Dtype.Bool ]))))))
          move_where_load;
      ]))
