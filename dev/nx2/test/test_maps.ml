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
      dead = Prim.live;
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
let slow1 ~by k dt x = first (Exec.run ~by (Prim.op1 ~by k dt x))
let slow2 ~by k dt x y = first (Exec.run ~by (Prim.op2 ~by k dt x y))
let slow3 ~by k c x y = first (Exec.run ~by (Prim.op3 ~by k c x y))

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
                dead = Prim.live;
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
                dead = Prim.live;
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
            | Value.Array _ | Value.Shards _ | Value.Donated _ | Value.Traced _
              ->
                -1
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

(* Donation's reuse: an elementwise operation writes into a donated operand's
   memory where rig holds it exclusive and it fits the result. *)
let array_of (type v s) (x : (v, s, b) Value.t) : (v, s) A.t =
  match x with
  | Value.Array { a; _ } -> a
  | Value.Shards _ | Value.Donated _ | Value.Deferred _ | Value.Traced _ ->
      fail "not on one device"

let donated_add x =
  let d = Exec.donate ~by:"t" x in
  let y =
    first
      (Exec.run ~by:"t"
         (Prim.op2 ~by:"t" (Binary Add) D.Float32 d (f32 [| 1.; 1. |])))
  in
  (y, Rig.Buffer.overlaps (A.buffer (array_of x)) (A.buffer (array_of y)))

let test_reuse () =
  let x = f32 [| 1.; 2. |] in
  let y, reused = donated_add x in
  equal ~msg:"written in place" bool true reused;
  equal ~msg:"the result" (array bits) [| 2.; 3. |] (elements y)

(* The same through the fast path of an elementwise operation. *)
let test_reuse_fast () =
  let x = f32 [| 1.; 2. |] in
  let d = Exec.donate ~by:"t" x in
  let y =
    Exec.apply2 ~slow:slow2 ~by:"t" (Binary Add) D.Float32 (f32 [| 1.; 1. |]) d
  in
  equal ~msg:"written in place" bool true
    (Rig.Buffer.overlaps (A.buffer (array_of x)) (A.buffer (array_of y)));
  equal ~msg:"the result" (array bits) [| 2.; 3. |] (elements y)

(* The fast paths of one and three operands, each operand donated in turn. *)
let test_reuse_fast1 () =
  let x = f32 [| 1.; 2. |] in
  let y =
    Exec.apply1 ~slow:slow1 ~by:"t" (Unary Neg) D.Float32
      (Exec.donate ~by:"t" x)
  in
  equal ~msg:"written in place" bool true
    (Rig.Buffer.overlaps (A.buffer (array_of x)) (A.buffer (array_of y)));
  equal ~msg:"the result" (array bits) [| -1.; -2. |] (elements y)

let test_reuse_fast3 () =
  let c () =
    Value.Array
      {
        at = at1;
        a = A.to_device (m 0) (A.of_array D.Bool [| 2 |] [| true; false |]);
        dead = Prim.live;
      }
  in
  let where i =
    let ops = [| f32 [| 1.; 2. |]; f32 [| 3.; 4. |] |] in
    let x = ops.(i) in
    ops.(i) <- Exec.donate ~by:"t" x;
    let y = Exec.apply3 ~slow:slow3 ~by:"t" Where (c ()) ops.(0) ops.(1) in
    equal ~msg:"the result" (array bits) [| 1.; 4. |] (elements y);
    Rig.Buffer.overlaps (A.buffer (array_of x)) (A.buffer (array_of y))
  in
  equal ~msg:"into the second, in place" bool true (where 0);
  equal ~msg:"into the third, in place" bool true (where 1);
  let d = c () in
  let y =
    Exec.apply3 ~slow:slow3 ~by:"t" Where (Exec.donate ~by:"t" d)
      (f32 [| 1.; 2. |])
      (f32 [| 3.; 4. |])
  in
  equal ~msg:"a donated condition" (array bits) [| 1.; 4. |] (elements y);
  raises_match (Exn.invalid_arg ~substring:"t: operand 1 was donated to t")
    (fun () -> Exec.run ~by:"t" (Value.Copy d))

let test_no_reuse_shared () =
  let x = f32 [| 1.; 2. |] in
  Rig.Claim.share (A.buffer (array_of x));
  let y, reused = donated_add x in
  equal ~msg:"fresh memory" bool false reused;
  equal ~msg:"the result" (array bits) [| 2.; 3. |] (elements y)

let test_no_reuse_strided () =
  let base = f32 [| 1.; 0.; 2.; 0. |] in
  let x =
    Value.Array
      {
        at = at1;
        a =
          Option.get
            (A.move
               (Slice [| { start = 0; count = 2; step = 2 } |])
               (array_of base));
        dead = Prim.live;
      }
  in
  let y, reused = donated_add x in
  equal ~msg:"fresh memory" bool false reused;
  equal ~msg:"the result" (array bits) [| 2.; 3. |] (elements y)

(* A handle a view passes on keeps its memory unshared: the view is within the
   chain, so the final consumer writes into it. *)
let test_reuse_after_view () =
  let x = f32 [| 1.; 2. |] in
  let d =
    Exec.run ~by:"t" (Value.Move (Reshape [| 1; 2 |], Exec.donate ~by:"t" x))
  in
  let one =
    Exec.run ~by:"t" (Value.Move (Reshape [| 1; 2 |], f32 [| 1.; 1. |]))
  in
  let y =
    first (Exec.run ~by:"t" (Prim.op2 ~by:"t" (Binary Add) D.Float32 d one))
  in
  equal ~msg:"written in place" bool true
    (Rig.Buffer.overlaps (A.buffer (array_of x)) (A.buffer (array_of y)));
  equal ~msg:"the result" (array bits) [| 2.; 3. |] (elements y)

(* A contraction consumes its donated [init] and computes into fresh memory:
   here [y = init + a · b] elementwise, as a contraction with one batch pair and
   no sum. *)
let test_contract_consumes () =
  let init = f32 [| 10.; 20. |] in
  let d = Exec.donate ~by:"t" init in
  let spec =
    Nx_kernel.Spec.contract
      ~batch:[| (0, 0) |]
      ~contracting:[||] ~acc:(D.Any D.Float32) ~out:(D.Any D.Float32) ~init:true
  in
  let y =
    Exec.run ~by:"t"
      (Value.Contract
         {
           out = D.Float32;
           spec;
           a = f32 [| 1.; 2. |];
           b = f32 [| 3.; 4. |];
           init = Some d;
         })
  in
  equal ~msg:"the result" (array bits) [| 13.; 28. |] (elements y);
  equal ~msg:"fresh memory" bool false
    (Rig.Buffer.overlaps (A.buffer (array_of init)) (A.buffer (array_of y)));
  raises_match (Exn.invalid_arg ~substring:"t: operand 1 was donated to t")
    (fun () -> Exec.run ~by:"t" (Value.Copy init))

let donation =
  group "donation"
    [
      test "a handle passed on by a view is written in place"
        test_reuse_after_view;
      test "a contraction consumes its donated init" test_contract_consumes;
      test "an elementwise operation writes into a donated operand" test_reuse;
      test "the fast path writes into a donated operand too" test_reuse_fast;
      test "the fast path of one operand writes into it" test_reuse_fast1;
      test "the fast path of three operands writes into a donated one"
        test_reuse_fast3;
      test "shared memory is not written" test_no_reuse_shared;
      test "a strided operand is not written" test_no_reuse_strided;
    ]

let () =
  exit (run "nx maps" [ maps; checks; constants; rules; programs; donation ])
