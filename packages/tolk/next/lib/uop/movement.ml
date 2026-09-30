(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops

let is_int_const i c =
  match value c with
  | #Dtype.value as v -> Dtype.Value.(v = of_int i)
  | `Invalid -> false

let shrink_arg u =
  match marg u with Shrink arg -> arg | _ -> invalid_arg "not a shrink"

let permute_arg u =
  match marg u with Permute arg -> arg | _ -> invalid_arg "not a permute"

let mop_cleanup =
  Pattern_matcher.(
    v
      [
        (* Merge adjacent shrinks. *)
        rule
          (Upat.f
             (Upat.op Op.Shrink ~name:"x")
             Op.Shrink ~allow_any_len:true ~name:"s")
          (fun m ->
            let x = m "x" and s = m "s" in
            let merge (o, _) (p, n) = (Sint.(o + p), n) in
            Some
              (mop (nth x 0)
                 (Shrink (List.map2 merge (shrink_arg x) (shrink_arg s)))));
        (* Merge adjacent reshapes. *)
        rule
          (Upat.op Op.Reshape ~name:"x"
             ~src:[ Upat.op Op.Reshape ~name:"x2"; Upat.wild ])
          (fun m ->
            let x = m "x" in
            Some (replace ~src:[ nth (m "x2") 0; nth x 1 ] x));
        (* Remove no-op reshapes. *)
        rule
          (Upat.op Op.Reshape ~name:"x" ~src:[ Upat.var "x2"; Upat.wild ])
          (fun m ->
            let x2 = m "x2" in
            match shape_opt x2 with
            | Some s when List.equal Sint.equal s (shape (m "x")) -> Some x2
            | _ -> None);
        (* Merge permutes. *)
        rule
          (Upat.op Op.Permute ~name:"x" ~src:[ Upat.op Op.Permute ~name:"x2" ])
          (fun m ->
            let x2 = m "x2" in
            let order =
              List.map (List.nth (permute_arg x2)) (permute_arg (m "x"))
            in
            Some (replace ~arg:(Axes order) x2));
        (* Remove no-op permutes. *)
        rule (Upat.op Op.Permute ~name:"x") (fun m ->
            let x = m "x" in
            let order = permute_arg x in
            let identity = List.init (List.length order) Fun.id in
            if List.equal Int.equal order identity then Some (nth x 0) else None);
        (* A stack of indexes by constants. *)
        rule
          (Upat.op Op.Stack ~name:"stk"
             ~each:(Upat.op Op.Index ~src:[ Upat.var "src"; Upat.op Op.Const ]))
          (fun m ->
            let stk = m "stk" and src = m "src" in
            let in_order i x = is_int_const i (nth x 1) in
            if
              List.equal Sint.equal (shape stk) (shape src)
              && List.for_all Fun.id (List.mapi in_order (Ops.src stk))
            then Some src
            else None);
        (* A constant index into a stack is that stack's source. *)
        rule
          (Upat.op Op.Index ~name:"idx" ~allow_any_len:true
             ~src:[ Upat.op Op.Stack ~name:"a"; Upat.cvar "i" ])
          (fun m ->
            let x = index (m "a") [ m "i" ] in
            match List.drop 2 (src (m "idx")) with
            | [] -> Some x
            | rest -> Some (index x rest));
        (* An index of an index is one index. *)
        rule
          (Upat.op Op.Index ~name:"idx2" ~allow_any_len:true
             ~src:[ Upat.op Op.Index ~name:"idx1" ~allow_any_len:true ])
          (fun m ->
            let idx1 = m "idx1" in
            let idxs = List.tl (src idx1) @ List.tl (src (m "idx2")) in
            if List.for_all (fun x -> List.is_empty (shape x)) idxs then
              Some (index (nth idx1 0) idxs)
            else None);
        (* An index of a shaped index. *)
        rule
          (Upat.op Op.Index ~name:"idx2" ~allow_any_len:true
             ~src:
               [ Upat.op Op.Index ~src:[ Upat.var "buf"; Upat.var "idx1_arg" ] ])
          (fun m ->
            let idx1_arg = m "idx1_arg"
            and idxs = List.drop 1 (src (m "idx2")) in
            if List.compare_length_with idxs (ndim idx1_arg) = 0 then
              Some (index (m "buf") [ index idx1_arg idxs ])
            else None);
      ])
