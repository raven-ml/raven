(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Shape

let zero = `Int Bigint.zero

(* The load reads the selection's other value where its gate fails. *)
let move_where_load m =
  let l = m "l" in
  Some (replace ~src:[ nth l 0; cast (m "a") (dtype l); nth l 2 ] l)

let gated = Upat.(where (var "gate") wild (const `Invalid))

let mop =
  Upat.v
    ~op:(Op.Set.of_list [ Op.Index; Op.Shrink ])
    ~src:[ Upat.wild; gated ]
    ~allow_any_len:true ~name:"mop" ()

(* [mop] with each of its sources selected by the gate, or Invalid, replaced by
   the value it selects. *)
let ungate m =
  let gate = m "gate" in
  let ungated s =
    if op s = Op.Where && nth s 0 == gate && is_invalid (nth s 2) then nth s 1
    else s
  in
  replace ~src:(List.map ungated (src (m "mop"))) (m "mop")

let gated_load l = Upat.load Upat.wild [ Upat.wild; l ] ~name:"l"

let pm_move_gates_from_index =
  Pattern_matcher.(
    v
      (fun () -> [
        (* A gated load reads 0 where its gate fails. *)
        rule (Upat.load mop [] ~name:"l") (fun m ->
            Some (load (ungate m) [ vconst_like (m "l") zero; m "gate" ]));
        rule
          (Upat.store mop [ Upat.var "data" ])
          (fun m -> Some (store ~gate:(m "gate") (ungate m) (m "data")));
        (* A where around a gated load becomes the load's alternate value. *)
        rule
          Upat.(
            where (var "gate")
              (gated_load (var "gate" ~dtype:[ Dtype.Bool ]))
              (var "a"))
          move_where_load;
        rule
          Upat.(
            where (var "gate") (var "a")
              (gated_load (bitwise_not (var "gate" ~dtype:[ Dtype.Bool ]))))
          move_where_load;
      ]))
