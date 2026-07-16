(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

let mop_cleanup : Upat.Pattern_matcher.t =
  let open Upat in
  Pattern_matcher.make
    [
      (* Merge adjacent reshapes: the outer reshape keeps its own target
         shape but reads through the inner reshape's input. *)
      (op ~name:"x" ~src:[ op ~name:"x2" Ops.Reshape; any ] Ops.Reshape
      => fun bs ->
        let x = bs $ "x" and x2 = bs $ "x2" in
        Some (Uop.replace x ~src:[| (Uop.src x2).(0); (Uop.src x).(1) |] ()));
      (* Drop a reshape that does not change the shape. *)
      (op ~name:"x" ~src:[ var "x2"; any ] Ops.Reshape
      => fun bs ->
        let x = bs $ "x" and x2 = bs $ "x2" in
        match Uop.shape_opt x2 with
        | Some s2 when List.equal Uop.equal s2 (Uop.shape x) -> Some x2
        | _ -> None);
      (* Merge permutes: compose the two orderings. *)
      (op ~name:"x" ~src:[ op ~name:"x2" Ops.Permute ] Ops.Permute
      => fun bs ->
        let x = bs $ "x" and x2 = bs $ "x2" in
        match Uop.Arg.as_ints (Uop.arg x2), Uop.Arg.as_ints (Uop.arg x) with
        | Some a2, Some a ->
            Some
              (Uop.replace x2 ~arg:(Uop.Arg.Ints (List.map (List.nth a2) a)) ())
        | _ -> None);
      (* Drop an identity permute. *)
      (op ~name:"x" Ops.Permute
      => fun bs ->
        let x = bs $ "x" in
        match Uop.Arg.as_ints (Uop.arg x) with
        | Some order when order = List.init (List.length order) Fun.id ->
            Some (Uop.src x).(0)
        | _ -> None);
      (* A stack whose lanes read the sequential elements of one source is
         that source. *)
      (op_src ~name:"stk"
         ~src:(repeat (op ~src:[ var "src"; op Ops.Const ] Ops.Index))
         Ops.Stack
      => fun bs ->
        let stk = bs $ "stk" and src = bs $ "src" in
        let lanes = Uop.src stk in
        let sequential =
          Array.to_list lanes
          |> List.mapi (fun i lane ->
                 Uop.const_int_value (Uop.src lane).(1) = Some i)
          |> List.for_all Fun.id
        in
        if sequential && List.equal Uop.equal (Uop.shape stk) (Uop.shape src)
        then Some src
        else None);
      (* A constant index into a stack selects that lane, carrying any
         remaining indices into the selected lane. *)
      (op ~name:"idx" ~allow_any_len:true
         ~src:[ op ~name:"a" Ops.Stack; cvar ~name:"i" () ]
         Ops.Index
      => fun bs ->
        let a = bs $ "a" and i = bs $ "i" and idx = bs $ "idx" in
        match Uop.const_int_value i with
        | Some iv when iv >= 0 && iv < Array.length (Uop.src a) ->
            let lane = (Uop.src a).(iv) in
            let idx_srcs = Uop.src idx in
            if Array.length idx_srcs <= 2 then Some lane
            else
              let extra =
                Array.to_list (Array.sub idx_srcs 2 (Array.length idx_srcs - 2))
              in
              Some (Uop.index ~ptr:lane ~idxs:extra ())
        | _ -> None);
    ]
