(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What only the engine's operations reach, through its private modules, copied
   here: maps of several nodes, programs of coordinates on split placements,
   checks, and the laws of constants. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module P = Nx_kernel.Prog
module C = Nx_support.Counting

type b

let m = Nx_support.memory
let s1 : b Devices.t = Devices.mint ~by:"t" ~kernels:(module C) [ m 0 ]
let s2 : b Devices.t = Devices.mint ~by:"t" ~kernels:(module C) [ m 0; m 1 ]
let at1 = Devices.one s1 0
let split = Devices.split ~by:"t" ~axis:0 s2

let bits =
  Testable.make ~pp:Format.pp_print_float ~equal:(fun a b ->
      Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b))

let f32 data : (float, D.float32_elt, b) Value.t =
  Value.Array
    {
      at = at1;
      a = A.to_device (m 0) (A.of_array D.Float32 [| Array.length data |] data);
    }

let elements (type v s) (x : (v, s, b) Value.t) : v array =
  match
    Repr.array
      (Place.value ~by:"t" (Devices.rebrand (Devices.one Devices.host 0)) x)
  with
  | Some a -> A.to_array a
  | None -> fail "no array"

let first (x, ()) = x

(* A one-node map built and run: no interpretation reaches the engine here. *)
let slow2 ~by k dt x y = first (Exec.run ~by (Prim.op2 k dt x y))

let maps =
  group "maps"
    [
      test "a map the kernels decline computes node by node" (fun () ->
          (* x0 * x1 + x0 *)
          let prog =
            P.v
              ~ins:[| D.Any D.Float32; D.Any D.Float32 |]
              [| In 0; In 1; Op2 (Binary Mul, 0, 1); Op2 (Binary Add, 2, 0) |]
              ~outs:[| 3 |]
          in
          let x = f32 [| 1.; 2.; 3. |] and y = f32 [| 4.; 5.; 6. |] in
          let z =
            first
              (Exec.run ~by:"t"
                 (Value.Map
                    {
                      layout = Nx_array.Layout.contiguous [| 3 |];
                      prog;
                      outs = Value.[ D.Float32 ];
                      loads = [| Plain x; Plain y |];
                    }))
          in
          equal (array bits) [| 5.; 12.; 21. |] (elements z));
      test "a creation of coordinates computes each device's window" (fun () ->
          let prog =
            P.v ~ins:[||]
              [| Coord 0; Op1 (Cast, D.Any D.Float32, 0) |]
              ~outs:[| 1 |]
          in
          let c =
            first
              (Exec.run ~by:"t"
                 (Value.Map
                    {
                      layout = Nx_array.Layout.contiguous [| 4 |];
                      prog;
                      outs = Value.[ D.Float32 ];
                      loads = [||];
                    }))
          in
          let placed = Exec.at split c in
          let shards = Option.get (Repr.shards placed) in
          equal
            (array (array bits))
            [| [| 0.; 1. |]; [| 2.; 3. |] |]
            (Array.map A.to_array shards));
    ]

exception Failed of int array * float

let checks =
  group "checks"
    [
      test "a check that holds raises nothing" (fun () ->
          let ok =
            Value.Array
              {
                at = at1;
                a =
                  A.to_device (m 0) (A.of_array D.Bool [| 2 |] [| true; true |]);
              }
          in
          Exec.run ~by:"t"
            (Value.Check { ok; data = []; fail = (fun _ _ -> Exit) }));
      test
        "a check raises its exception at the first failing index, with the \
         data there" (fun () ->
          let ok =
            Value.Array
              {
                at = at1;
                a =
                  A.to_device (m 0)
                    (A.of_array D.Bool [| 3 |] [| true; false; false |]);
              }
          in
          let fail i data =
            match data with
            | [ Value.Any x ] -> (
                match Repr.array x with
                | Some a -> Failed (i, A.get (A.expect D.Float32 (A.Any a)) [||])
                | None -> Exit)
            | _ -> Exit
          in
          raises
            (Failed ([| 1 |], 20.))
            (fun () ->
              Exec.run ~by:"t"
                (Value.Check
                   { ok; data = [ Any (f32 [| 10.; 20.; 30. |]) ]; fail })));
    ]

(* A fresh constant [1.]: a node of its own. *)
let fresh_one () =
  first
    (Exec.run ~by:"t"
       (Value.Map
          {
            layout = Nx_array.Layout.contiguous [||];
            prog =
              Prim.program (Const (D.Any D.Float32, P.bits D.Float32 1.)) [||];
            outs = Value.[ D.Float32 ];
            loads = [||];
          }))

let constants =
  group "constants"
    [
      test
        "a constant read computes once per placement, its intermediates once \
         per read" (fun () ->
          let one = fresh_one () in
          let c =
            Exec.apply2 ~slow:slow2 ~by:"t" (Binary Add) D.Float32 one one
          in
          let d = Exec.apply2 ~slow:slow2 ~by:"t" (Binary Mul) D.Float32 c c in
          let e = Exec.apply2 ~slow:slow2 ~by:"t" (Binary Add) D.Float32 c d in
          C.reset ();
          ignore (Exec.at at1 e);
          (* one, c, d, e: four nodes. *)
          equal ~msg:"reading e" int 4 (C.calls ());
          C.reset ();
          ignore (Exec.at at1 e);
          equal ~msg:"reading e again" int 0 (C.calls ());
          ignore (Exec.at at1 d);
          (* e's intermediates were its read's alone: one, c and d again. *)
          equal ~msg:"reading d" int 3 (C.calls ());
          C.reset ();
          ignore (Exec.at at1 d);
          equal ~msg:"reading d again" int 0 (C.calls ()));
      test "a read keeps the read constant's results, and no intermediate's"
        (fun () ->
          let one = fresh_one () in
          let c =
            Exec.apply2 ~slow:slow2 ~by:"t" (Binary Add) D.Float32 one one
          in
          let d = Exec.apply2 ~slow:slow2 ~by:"t" (Binary Mul) D.Float32 c c in
          let kept (type v s) (x : (v, s, b) Value.t) =
            match x with
            | Value.Deferred { node = Value.Node n; _ } ->
                List.length (Atomic.get n.memo)
            | Value.Array _ | Value.Shards _ | Value.Traced _ -> -1
          in
          ignore (Exec.at at1 d);
          equal ~msg:"d" int 1 (kept d);
          equal ~msg:"c" int 0 (kept c);
          equal ~msg:"one" int 0 (kept one));
      test "a chain of constants computes in one pass per node, at any length"
        (fun () ->
          let n = 100_000 in
          let one = fresh_one () in
          let rec chain k x =
            if k = 0 then x
            else
              chain (k - 1)
                (Exec.apply2 ~slow:slow2 ~by:"t" (Binary Add) D.Float32 x one)
          in
          let x =
            chain n
              (Exec.apply2 ~slow:slow2 ~by:"t" (Binary Mul) D.Float32 one one)
          in
          C.reset ();
          let v = Exec.at at1 x in
          (* The fill of [one], the product and the n sums. *)
          equal int (n + 2) (C.calls ());
          equal (array bits) [| Float.of_int (n + 1) |] (elements v));
      test "domains racing to compute a constant agree bit for bit" (fun () ->
          let one = fresh_one () in
          let c =
            Exec.apply2 ~slow:slow2 ~by:"t" (Binary Mul) D.Float32
              (Exec.apply2 ~slow:slow2 ~by:"t" (Binary Add) D.Float32 one one)
              one
          in
          let work () = elements (Exec.at at1 c) in
          let d = Domain.spawn work in
          let here = work () in
          equal (array bits) here (Domain.join d));
    ]

(* A movement or bitcast its rule refuses raises naming its function before any
   kernel runs, even where its arrays could only be moved after a copy. *)
let refused name op =
  C.reset ();
  raises_match (Exn.invalid_arg ~substring:"t: ") (fun () ->
      Exec.run ~by:"t" op);
  equal ~msg:(name ^ ": kernel calls") int 0 (C.calls ())

let rules =
  group "rules"
    [
      test "an ill-formed bitcast or movement calls no kernel" (fun () ->
          let x = f32 [| 1.; 2.; 3. |] in
          refused "a wider bitcast of an odd axis"
            (Value.Bitcast (D.Float64, x));
          refused "a permutation of another rank"
            (Value.Move (Permute [| 1; 0 |], x));
          refused "a reshape of another size" (Value.Move (Reshape [| 4 |], x)));
    ]

(* Programs of distinct literals are not kept: a loop of scalars leaves the
   domain's table as it was. *)
let test_literals_not_kept () =
  let one v = Prim.program (Const (D.Any D.Float32, P.bits D.Float32 v)) [||] in
  ignore (one 0.);
  let before = Prim.programs_kept () in
  for i = 1 to 10_000 do
    ignore (one (Float.of_int i))
  done;
  ignore (one 0.);
  equal ~msg:"programs kept" int before (Prim.programs_kept ())

let programs =
  group "programs"
    [ test "programs of distinct literals are not kept" test_literals_not_kept ]

let () = exit (run "nx maps" [ maps; checks; constants; rules; programs ])
