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
      dead = Value.Live;
    }

let elements (type v s) (x : (v, s, b) Value.t) : v array =
  match
    Repr.array
      (Place.value ~by:"t" (Devices.one Devices.host 0) x)
  with
  | Some a -> A.to_array a
  | None -> fail "no array"

let first (x, ()) = x

(* The map [prog] of one float32 output over [loads], of shape [n]. *)
let map1 n prog loads =
  first
    (Eval.run ~by:"t"
       (Value.Map
          {
            layout = Nx_array.Layout.contiguous [| n |];
            prog;
            outs = Value.[ D.Float32 ];
            loads;
          }))

(* A map of [w] arrays, an output and [w - 1] loads, gives the bits of the
   same sum computed in pieces of two loads: Σ x_k · c_k in order of k, each
   product rounded, then each sum. *)
let test_wide_maps () =
  let n = 5 and f = D.Any D.Float32 in
  let c k = P.Const (f, P.bits D.Float32 (1. /. Float.of_int (k + 3))) in
  let x k = f32 (Array.init n (fun i -> Float.of_int ((7 * k) + i) /. 10.)) in
  List.iter
    (fun w ->
      let nl = w - 1 in
      let xs = Array.init nl x in
      (* Loads 0 to nl - 1, then per load its constant and product, then
         the sums. *)
      let nodes =
        List.init nl (fun k -> P.In k)
        @ List.concat
            (List.init nl (fun k ->
                 [ c k; P.Op2 (Binary Mul, k, nl + (2 * k)) ]))
        @ List.init (nl - 1) (fun i ->
              let prod k = nl + (2 * k) + 1 in
              let acc = if i = 0 then prod 0 else (3 * nl) + i - 1 in
              P.Op2 (Binary Add, acc, prod (i + 1)))
      in
      let prog =
        P.v ~ins:(Iarray.init nl (fun _ -> f)) (Iarray.of_list nodes)
          ~outs:[| List.length nodes - 1 |]
      in
      let whole =
        map1 n prog (Iarray.map (fun x -> Value.Plain x) (Iarray.of_array xs))
      in
      (* The pieces: x_0 · c_0, then acc + x_k · c_k for each next load. *)
      let first_piece =
        map1 n
          (P.v ~ins:[| f |]
             [| In 0; c 0; Op2 (Binary Mul, 0, 1) |]
             ~outs:[| 2 |])
          [| Plain xs.(0) |]
      in
      let step acc k =
        map1 n
          (P.v ~ins:[| f; f |]
             [|
               In 0; In 1; c k; Op2 (Binary Mul, 1, 2); Op2 (Binary Add, 0, 3);
             |]
             ~outs:[| 4 |])
          [| Plain acc; Plain xs.(k) |]
      in
      let pieces = ref first_piece in
      for k = 1 to nl - 1 do
        pieces := step !pieces k
      done;
      equal ~msg:(Printf.sprintf "%d arrays" w) (array bits)
        (elements !pieces) (elements whole))
    [ 17; 18; 19; 32; 100 ]

let maps =
  group "maps"
    [
      test "a map of several nodes computes each node" (fun () ->
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
              (Eval.run ~by:"t"
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
              (Eval.run ~by:"t"
                 (Value.Map
                    {
                      layout = Nx_array.Layout.contiguous [| 4 |];
                      prog;
                      outs = Value.[ D.Float32 ];
                      loads = [||];
                    }))
          in
          let placed = Constant.at split c in
          let shards = Option.get (Repr.shards placed) in
          equal
            (array (array bits))
            [| [| 0.; 1. |]; [| 2.; 3. |] |]
            (Iarray.to_array (Iarray.map A.to_array shards)));
      test "a map of any number of arrays computes as its pieces do"
        test_wide_maps;
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
                dead = Value.Live;
              }
          in
          Eval.run ~by:"t"
            (Value.Check { ok; data = []; fail = (fun _ _ -> Exit) });
          Check.wait ~by:"t" ok);
      test
        "a check off the host raises its exception at the next wait, at the \
         first failing index, with the data there" (fun () ->
          let ok =
            Value.Array
              {
                at = at1;
                a =
                  A.to_device (m 0)
                    (A.of_array D.Bool [| 3 |] [| true; false; false |]);
                dead = Value.Live;
              }
          in
          let fail i (x, ()) = Failed (i, x) in
          Eval.run ~by:"t"
            (Value.Check { ok; data = [ f32 [| 10.; 20.; 30. |] ]; fail });
          raises (Failed ([| 1 |], 20.)) (fun () -> Check.wait ~by:"t" ok));
    ]

(* A fresh constant [1.]: a node of its own. *)
let fresh_one () =
  first
    (Eval.run ~by:"t"
       (Value.Map
          {
            layout = Nx_array.Layout.contiguous [||];
            prog =
              P.of_node ~ins:[||]
                (Const (D.Any D.Float32, P.bits D.Float32 1.));
            outs = Value.[ D.Float32 ];
            loads = [||];
          }))

(* The coordinate along [axis], as a float32: a program of a coordinate
   alone. *)
let coords axis =
  P.v ~ins:[||] [| Coord axis; Op1 (Cast, D.Any D.Float32, 0) |] ~outs:[| 1 |]

let constants =
  group "constants"
    [
      test "a constant read computes each node it reads once, at each read"
        (fun () ->
          let one = fresh_one () in
          let c = Eval.apply2 ~by:"t" (Binary Add) D.Float32 one one in
          let d = Eval.apply2 ~by:"t" (Binary Mul) D.Float32 c c in
          let e = Eval.apply2 ~by:"t" (Binary Add) D.Float32 c d in
          C.reset ();
          ignore (Constant.at at1 e);
          (* one, c, d, e: four nodes. *)
          equal ~msg:"reading e" int 4 (C.calls ());
          C.reset ();
          ignore (Constant.at at1 e);
          equal ~msg:"reading e again" int 4 (C.calls ());
          C.reset ();
          ignore (Constant.at at1 d);
          equal ~msg:"reading d" int 3 (C.calls ()));
      test "a constant read at two placements in one read computes at each"
        (fun () ->
          let c =
            first
              (Eval.run ~by:"t"
                 (Value.Map
                    {
                      layout = Nx_array.Layout.contiguous [| 4 |];
                      prog = coords 0;
                      outs = Value.[ D.Float32 ];
                      loads = [||];
                    }))
          in
          let sorted, _ =
            Eval.run ~by:"t"
              (Value.Sort { axis = 0; descending = true; k = None; x = c })
          in
          let e = Eval.apply2 ~by:"t" (Binary Add) D.Float32 sorted c in
          C.reset ();
          let y = Constant.at split e in
          (* The add reads [c] split, a window on each device; the sort reads
             it whole on each device and cuts its results to the windows: [c]
             twice on each device, then the sort and the add once on each. *)
          equal ~msg:"split" int 8 (C.calls ());
          equal (array bits) [| 3.; 3.; 3.; 3. |] (elements y);
          C.reset ();
          ignore (Constant.at at1 e);
          equal ~msg:"on one device" int 3 (C.calls ()));
      test "a chain of constants computes in one pass per node, at any length"
        (fun () ->
          let n = 100_000 in
          let one = fresh_one () in
          let rec chain k x =
            if k = 0 then x
            else
              chain (k - 1) (Eval.apply2 ~by:"t" (Binary Add) D.Float32 x one)
          in
          let x =
            chain n (Eval.apply2 ~by:"t" (Binary Mul) D.Float32 one one)
          in
          C.reset ();
          let v = Constant.at at1 x in
          (* The fill of [one], the product and the n sums. *)
          equal int (n + 2) (C.calls ());
          equal (array bits) [| Float.of_int (n + 1) |] (elements v));
      cases "a chain of views of a constant reads at any length" ~name:fst
        [
          ("read by an operation", fun p x -> Constant.at p x);
          ( "placed",
            fun p x -> Eval.run ~by:"t" (Value.Place (p, x)) );
        ]
        (fun (_, read) ->
          let flip x =
            Eval.run ~by:"t"
              (Value.Move
                 (Slice [| { Nx_array.Move.start = 1; count = 2; step = -1 } |],
                  x))
          in
          let two =
            first
              (Eval.run ~by:"t"
                 (Value.Map
                    {
                      layout = Nx_array.Layout.contiguous [| 2 |];
                      prog = coords 0;
                      outs = Value.[ D.Float32 ];
                      loads = [||];
                    }))
          in
          let rec chain k x = if k = 0 then x else chain (k - 1) (flip x) in
          let x = chain 100_001 two in
          equal (array bits) [| 1.; 0. |] (elements (read at1 x)));
      test "domains racing to compute a constant agree bit for bit" (fun () ->
          let one = fresh_one () in
          let c =
            Eval.apply2 ~by:"t" (Binary Mul) D.Float32
              (Eval.apply2 ~by:"t" (Binary Add) D.Float32 one one)
              one
          in
          let work () = elements (Constant.at at1 c) in
          let d = Domain.spawn work in
          let here = work () in
          equal (array bits) here (Domain.join d));
    ]

(* Constants read at a placement their rule does not give: a creation, an
   assembly of pieces read on one device, and loops of no load, each against
   the elements the definition gives. *)

let on2 = Devices.on s2
let d0 = Devices.one s2 0
let split2 = Devices.split ~by:"t" ~axis:0 s2

let assembly pieces : (float, D.float32_elt, b) Value.t =
  Eval.run ~by:"t"
    (Value.Assemble { dtype = D.Float32; shape = [| 4 |]; fill = 7.; pieces })

let ones n : (float, D.float32_elt, b) Value.t =
  first
    (Eval.run ~by:"t"
       (Value.Map
          {
            layout = Nx_array.Layout.contiguous [| n |];
            prog =
              P.of_node ~ins:[||]
                (Const (D.Any D.Float32, P.bits D.Float32 1.));
            outs = Value.[ D.Float32 ];
            loads = [||];
          }))

(* The sum and the running sum, along the last axis, of [[0; 1; 2]] repeated
   over four rows. *)
let row_sums () : (float, D.float32_elt, b) Value.t =
  let s, () =
    Eval.run ~by:"t"
      (Value.Reduce
         {
           layout = Nx_array.Layout.contiguous [| 4; 3 |];
           axes = [| 1 |];
           prog = coords 1;
           reductions = Value.[ Monoid (Sum, 0, D.Float32) ];
           loads = [||];
         })
  in
  s

(* The sum along [axis] of the coordinate along axis 0 over [shape]. *)
let coordinate_sums shape axis : (float, D.float32_elt, b) Value.t =
  let s, () =
    Eval.run ~by:"t"
      (Value.Reduce
         {
           layout = Nx_array.Layout.contiguous shape;
           axes = [| axis |];
           prog = coords 0;
           reductions = Value.[ Monoid (Sum, 0, D.Float32) ];
           loads = [||];
         })
  in
  s

(* Its running sum along the last axis. *)
let coordinate_scans () : (float, D.float32_elt, b) Value.t =
  Eval.run ~by:"t"
    (Value.Scan
       {
         layout = Nx_array.Layout.contiguous [| 4; 3 |];
         axis = 1;
         prog = coords 0;
         reduction = Monoid (Sum, 0, D.Float32);
         loads = [||];
         carry = None;
       })

(* The shapes of [x]'s arrays, one per device. *)
let shards x =
  Iarray.to_list
    (Iarray.map
       (fun a -> Iarray.to_array (Nx_array.Layout.shape (A.layout a)))
       (Prim.arrays x))

let row_scans () : (float, D.float32_elt, b) Value.t =
  Eval.run ~by:"t"
    (Value.Scan
       {
         layout = Nx_array.Layout.contiguous [| 4; 3 |];
         axis = 1;
         prog = coords 1;
         reduction = Monoid (Sum, 0, D.Float32);
         loads = [||];
         carry = None;
       })

let forced =
  let at p x = elements (Constant.at p x) in
  group "forced constants"
    [
      cases "an empty assembly fills its placement" ~name:fst
        [ ("split", split2); ("on one device", d0); ("on the set", on2) ]
        (fun (_, p) ->
          equal (array bits) [| 7.; 7.; 7.; 7. |] (at p (assembly [])));
      cases "pieces read on one device assemble there" ~name:fst
        [ ("on one device", d0); ("split", split2) ]
        (fun (_, p) ->
          let r : _ iarray =
            [| { Nx_array.Move.start = 1; count = 2; step = 1 } |]
          in
          equal (array bits) [| 7.; 1.; 1.; 7. |]
            (at p (assembly [ (r, ones 2) ])));
      (* The kernels compute a loop of no load over its shape: one call on
         each device of its placement. *)
      cases "a sum of coordinates reduces at its placement, a call a device"
        ~name:(fun (n, _, _) -> n)
        [ ("split", split2, 2); ("on one device", d0, 1) ]
        (fun (_, p, calls) ->
          C.reset ();
          let y = Constant.at p (row_sums ()) in
          equal ~msg:"kernel calls" int calls (C.calls ());
          equal (array bits) [| 3.; 3.; 3.; 3. |] (elements y));
      cases "a running sum of coordinates scans at its placement, a call a device"
        ~name:(fun (n, _, _) -> n)
        [ ("split", split2, 2); ("on one device", d0, 1) ]
        (fun (_, p, calls) ->
          C.reset ();
          let y = Constant.at p (row_scans ()) in
          equal ~msg:"kernel calls" int calls (C.calls ());
          equal (array bits)
            [| 0.; 1.; 3.; 0.; 1.; 3.; 0.; 1.; 3.; 0.; 1.; 3. |]
            (elements y));
      (* On a split placement each device computes its window of a loop of no
         load, its coordinates offset to the window's. A reduction along axis 0
         reads it whole on each device: windowed, the column sums would be 1
         and 5. *)
      test "a loop of no load computes each device's window" (fun () ->
          let shapes = list (array int) in
          C.reset ();
          let sums = Constant.at split2 (coordinate_sums [| 4; 3 |] 1) in
          equal ~msg:"row sums" (array bits) [| 0.; 3.; 6.; 9. |]
            (elements sums);
          equal ~msg:"row sums' positions" int 12 (C.positions ());
          equal ~msg:"row sums' shards" shapes [ [| 2 |]; [| 2 |] ] (shards sums);
          C.reset ();
          let scans = Constant.at split2 (coordinate_scans ()) in
          equal ~msg:"running sums" (array bits)
            [| 0.; 0.; 0.; 1.; 2.; 3.; 2.; 4.; 6.; 3.; 6.; 9. |]
            (elements scans);
          equal ~msg:"running sums' positions" int 12 (C.positions ());
          equal ~msg:"running sums' shards" shapes
            [ [| 2; 3 |]; [| 2; 3 |] ]
            (shards scans);
          C.reset ();
          let columns = Constant.at split2 (coordinate_sums [| 4; 2 |] 0) in
          equal ~msg:"column sums" (array bits) [| 6.; 6. |] (elements columns);
          equal ~msg:"column sums' positions" int 8 (C.positions ());
          equal ~msg:"column sums' shards" shapes [ [| 1 |]; [| 1 |] ]
            (shards columns));
    ]

(* An operation its rule refuses raises naming its function before any kernel
   runs, even where its arrays could only be moved after a copy. *)
let refused name op =
  C.reset ();
  raises_match (Exn.invalid_arg ~substring:"t: ") (fun () ->
      Eval.run ~by:"t" op);
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
      test "a coordinate past a loop's rank calls no kernel" (fun () ->
          let layout = Nx_array.Layout.contiguous [| 3 |] in
          refused "a map"
            (Value.Map
               {
                 layout;
                 prog = coords 1;
                 outs = Value.[ D.Float32 ];
                 loads = [||];
               });
          refused "a reduction"
            (Value.Reduce
               {
                 layout;
                 axes = [| 0 |];
                 prog = coords 1;
                 reductions = Value.[ Monoid (Sum, 0, D.Float32) ];
                 loads = [||];
               });
          refused "a scan"
            (Value.Scan
               {
                 layout;
                 axis = 0;
                 prog = coords 1;
                 reduction = Monoid (Sum, 0, D.Float32);
                 loads = [||];
                 carry = None;
               }));
    ]

(* Donation's reuse: an elementwise operation writes into a donated operand's
   memory where rig holds it exclusive and it fits the result. *)
let array_of (type v s) (x : (v, s, b) Value.t) : (v, s) A.t =
  match x with
  | Value.Array { a; _ } -> a
  | Value.Shards _ | Value.Handle _ | Value.Deferred _ | Value.Traced _ ->
      fail "not on one device"

let donated_add x =
  let d = Donate.donate ~by:"t" x in
  let y =
    first
      (Eval.run ~by:"t"
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
  let d = Donate.donate ~by:"t" x in
  let y =
    Eval.apply2 ~by:"t" (Binary Add) D.Float32 (f32 [| 1.; 1. |]) d
  in
  equal ~msg:"written in place" bool true
    (Rig.Buffer.overlaps (A.buffer (array_of x)) (A.buffer (array_of y)));
  equal ~msg:"the result" (array bits) [| 2.; 3. |] (elements y)

(* The fast paths of one and three operands, each operand donated in turn. *)
let test_reuse_fast1 () =
  let x = f32 [| 1.; 2. |] in
  let y = Eval.apply1 ~by:"t" (Unary Neg) D.Float32 (Donate.donate ~by:"t" x) in
  equal ~msg:"written in place" bool true
    (Rig.Buffer.overlaps (A.buffer (array_of x)) (A.buffer (array_of y)));
  equal ~msg:"the result" (array bits) [| -1.; -2. |] (elements y)

let test_reuse_fast3 () =
  let c () =
    Value.Array
      {
        at = at1;
        a = A.to_device (m 0) (A.of_array D.Bool [| 2 |] [| true; false |]);
        dead = Value.Live;
      }
  in
  let where i =
    let ops = [| f32 [| 1.; 2. |]; f32 [| 3.; 4. |] |] in
    let x = ops.(i) in
    ops.(i) <- Donate.donate ~by:"t" x;
    let y = Eval.apply3 ~by:"t" Where (c ()) ops.(0) ops.(1) in
    equal ~msg:"the result" (array bits) [| 1.; 4. |] (elements y);
    Rig.Buffer.overlaps (A.buffer (array_of x)) (A.buffer (array_of y))
  in
  equal ~msg:"into the second, in place" bool true (where 0);
  equal ~msg:"into the third, in place" bool true (where 1);
  let d = c () in
  let y =
    Eval.apply3 ~by:"t" Where (Donate.donate ~by:"t" d)
      (f32 [| 1.; 2. |])
      (f32 [| 3.; 4. |])
  in
  equal ~msg:"a donated condition" (array bits) [| 1.; 4. |] (elements y);
  raises_match (Exn.invalid_arg ~substring:"t: bool [2] was donated to t")
    (fun () -> Eval.run ~by:"t" (Value.Copy d))

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
        dead = Value.Live;
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
    Eval.run ~by:"t" (Value.Move (Reshape [| 1; 2 |], Donate.donate ~by:"t" x))
  in
  let one =
    Eval.run ~by:"t" (Value.Move (Reshape [| 1; 2 |], f32 [| 1.; 1. |]))
  in
  let y =
    first (Eval.run ~by:"t" (Prim.op2 ~by:"t" (Binary Add) D.Float32 d one))
  in
  equal ~msg:"written in place" bool true
    (Rig.Buffer.overlaps (A.buffer (array_of x)) (A.buffer (array_of y)));
  equal ~msg:"the result" (array bits) [| 2.; 3. |] (elements y)

(* A contraction consumes its donated [init] and writes into its memory: here
   [y = init + a · b] elementwise, as a contraction with one batch pair and no
   sum. *)
let test_contract_consumes () =
  let init = f32 [| 10.; 20. |] in
  let d = Donate.donate ~by:"t" init in
  let y =
    Eval.run ~by:"t"
      (Value.Contract
         {
           batch = [| (0, 0) |];
           contracting = [||];
           acc = D.Any D.Float32;
           out = D.Float32;
           a = f32 [| 1.; 2. |];
           b = f32 [| 3.; 4. |];
           init = Some d;
           a_scale = None;
           b_scale = None;
           segments = None;
         })
  in
  equal ~msg:"the result" (array bits) [| 13.; 28. |] (elements y);
  equal ~msg:"written in place" bool true
    (Rig.Buffer.overlaps (A.buffer (array_of init)) (A.buffer (array_of y)));
  raises_match (Exn.invalid_arg ~substring:"t: float32 [2] was donated to t")
    (fun () -> Eval.run ~by:"t" (Value.Copy init))

(* Two handles of one donor, one a copy a movement passed on, are one handle
   to their consumer: both hold the donor, and [handles] keeps the first. *)
let test_one_donor_twice () =
  let x =
    Value.Array
      {
        at = at1;
        a =
          A.to_device (m 0)
            (A.of_array D.Float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |]);
        dead = Value.Live;
      }
  in
  let moved mv h = Eval.run ~by:"t" (Value.Move (mv, h)) in
  let h1 = Donate.donate ~by:"t" x in
  (* A reshape of a transpose copies. *)
  let h2 =
    moved (Reshape [| 4 |])
      (moved (Permute [| 1; 0 |]) (Donate.donate ~by:"t" x))
  in
  let positions = List.map (fun (Donate.Handle (i, _)) -> i) in
  equal (list int) [ 1 ]
    (positions (Donate.handles [ Value.Any x; Value.Any h1; Value.Any h2 ]))

let donation =
  group "donation"
    [
      test "a handle passed on by a view is written in place"
        test_reuse_after_view;
      test "a contraction writes into its donated init" test_contract_consumes;
      test "an elementwise operation writes into a donated operand" test_reuse;
      test "the fast path writes into a donated operand too" test_reuse_fast;
      test "the fast path of one operand writes into it" test_reuse_fast1;
      test "the fast path of three operands writes into a donated one"
        test_reuse_fast3;
      test "shared memory is not written" test_no_reuse_shared;
      test "a strided operand is not written" test_no_reuse_strided;
      test "two handles of one donor are one to their consumer"
        test_one_donor_twice;
    ]

(* A Logsumexp's terms: shifts from the maximum about the least integers whose
   exp is a normal float32 (-87) and float64 (-708), infinities, signed zeros
   and NaNs with payloads. *)
type lse =
  | Lse :
      (float, 's) D.t * int array * int array * float array * int list
      -> lse

let pp_lse ppf (Lse (dt, shape, axes, xs, _)) =
  Format.fprintf ppf "%s %a along %a: %a" (D.name dt)
    Format.(pp_print_list ~pp_sep:pp_print_space pp_print_int)
    (Array.to_list shape)
    Format.(pp_print_list ~pp_sep:pp_print_space pp_print_int)
    (Array.to_list axes)
    Format.(pp_print_list ~pp_sep:pp_print_space pp_print_float)
    (Array.to_list xs)

let lse_terms =
  [
    0.; -0.; 1.; 50.; -37.; -86.5; -87.; -87.25; -88.; -100.; -660.; -707.5;
    -708.; -708.25; -750.; Float.infinity; Float.neg_infinity; Float.nan;
    Int64.float_of_bits 0x7ff8_0000_2000_0000L;
    Int64.float_of_bits 0xfff8_0000_0000_0001L;
  ]

let lse =
  let open Gen in
  with_pp pp_lse
  @@ let* rank = int_range 1 2 in
     let* shape = array ~size:(constant rank) (int_range 0 9) in
     let* axes =
       such_that
         (fun a -> a <> [||])
         (map
            (fun keep ->
              Array.of_list
                (List.filteri (fun a _ -> keep.(a)) (List.init rank Fun.id)))
            (array ~size:(constant rank) bool))
     in
     let n = Array.fold_left ( * ) 1 shape in
     let* xs = array ~size:(constant n) (of_list lse_terms) in
     let* signalling =
       array ~size:(constant n)
         (frequency [ (11, constant false); (1, constant true) ])
     in
     let sn = List.filter (fun i -> signalling.(i)) (List.init n Fun.id) in
     let+ dt = int_range 0 3 in
     match dt with
     | 0 -> Lse (D.Float32, shape, axes, xs, sn)
     | 1 -> Lse (D.Float64, shape, axes, xs, sn)
     | 2 -> Lse (D.Float16, shape, axes, xs, sn)
     | _ -> Lse (D.Bfloat16, shape, axes, xs, sn)

(* [xs] of [shape] at [dt], a signalling NaN with a payload at each flat
   position of [sn]. *)
let signalling (type s) (dt : (float, s) D.t) shape xs sn :
    (float, s, b) Value.t =
  let a = A.of_array dt (Iarray.of_array shape) xs in
  let index i =
    let idx = Array.make (Array.length shape) 0 and r = ref i in
    for k = Array.length shape - 1 downto 0 do
      idx.(k) <- !r mod shape.(k);
      r := !r / shape.(k)
    done;
    Iarray.of_array idx
  in
  let put (type u r) (u : (u, r) D.t) (code : u) =
    let raw = Option.get (A.bitcast u a) in
    List.iter (fun i -> A.set raw (index i) code) sn
  in
  (match dt with
  | D.Float16 -> put D.Uint16 0x7d01
  | D.Bfloat16 -> put D.Uint16 0x7f81
  | D.Float32 -> put D.Uint32 0x7f800101l
  | D.Float64 -> put D.Uint64 0x7ff0_0000_0000_0101L
  | _ -> ());
  Value.Array { at = at1; a = A.to_device (m 0) a; dead = Value.Live }

(* [L + D + 5 + 6 ln n] of nx.cpu's Logsumexp bound for [n] terms. *)
let lse_rounds n =
  let n' = Float.of_int n in
  let l = ((min n 1024 + 15) / 16) + 3 in
  let blocks = (n + 1023) / 1024 in
  let rec depth b d = if b <= 1 then d else depth ((b + 1) / 2) (d + 1) in
  Float.of_int (l + depth blocks 0 + 5) +. (6. *. Float.log n')

(* [m + log Σ exp (t - m)] of [ts] in float64, [m] their maximum: a NaN
   where a term is, [m] where it is infinite. *)
let lse_ref ts =
  if List.exists Float.is_nan ts then Float.nan
  else
    let m = List.fold_left Float.max Float.neg_infinity ts in
    if not (Float.is_finite m) then m
    else
      let s = List.fold_left (fun s t -> s +. Float.exp (t -. m)) 0. ts in
      m +. Float.log s

(* nx.cpu's Logsumexp is [m + log Σ exp (x - m)] within the bound {!Nx_cpu}
   states, in the float32 carrier for a narrow float, which adds its own
   rounding: a NaN where a term is, [-∞] for no term or terms all [-∞], [+∞]
   for a [+∞] term. *)
let law_logsumexp (Lse (dt, shape, axes, xs, sn)) =
  let cut = match dt with D.Float64 -> -708. | _ -> -87. in
  let x = signalling dt shape xs sn in
  cover "no term" (Array.exists (fun a -> shape.(a) = 0) axes);
  cover "a NaN term" (Array.exists Float.is_nan xs);
  cover "a signalling NaN term" (sn <> []);
  cover "a signalling NaN of a narrow float" (sn <> [] && D.bits dt = 16);
  (* Each output's terms, as [x] holds them, in C order of the outputs. *)
  let kept =
    List.filter
      (fun a -> not (Array.mem a axes))
      (List.init (Array.length shape) Fun.id)
  in
  let outs = List.fold_left (fun n a -> n * shape.(a)) 1 kept in
  let rows = Array.make outs [] in
  Array.iteri
    (fun i t ->
      let idx = Array.make (Array.length shape) 0 and r = ref i in
      for a = Array.length shape - 1 downto 0 do
        idx.(a) <- !r mod shape.(a);
        r := !r / shape.(a)
      done;
      let o = List.fold_left (fun o a -> (o * shape.(a)) + idx.(a)) 0 kept in
      rows.(o) <- t :: rows.(o))
    (elements x);
  let below =
    Array.exists
      (fun row ->
        let m = List.fold_left Float.max Float.neg_infinity row in
        Float.is_finite m
        && List.exists (fun t -> t -. m < cut && t -. m > -1000.) row)
      rows
  in
  cover "a term past the cutoff below a finite maximum" below;
  let op =
    Value.Reduce
      {
        layout = A.Layout.contiguous (Iarray.of_array shape);
        axes = Iarray.of_array axes;
        prog = P.of_node ~ins:[| D.Any dt |] (In 0);
        reductions = Value.[ Logsumexp (0, dt) ];
        loads = [| Plain x |];
      }
  in
  let got = elements (first (Eval.run ~by:"t" op)) in
  let u = match dt with D.Float64 -> Float.epsilon /. 2. | _ -> 0x1p-24 in
  let narrow =
    match dt with D.Float16 -> 0x1p-11 | D.Bfloat16 -> 0x1p-8 | _ -> 0.
  in
  equal ~msg:"one result per output" int outs (Array.length got);
  Array.iteri
    (fun o row ->
      let want = lse_ref row and y = got.(o) in
      let msg = Printf.sprintf "output %d" o in
      if not (Float.is_finite want) then
        equal ~msg (option float_exact)
          (if Float.is_nan want then None else Some want)
          (if Float.is_nan y then None else Some y)
      else
        let n = List.length row in
        (* The float64 reference errs by about [n + 4] roundings. *)
        let slack =
          match dt with
          | D.Float64 -> Float.of_int (n + 4) *. u *. (1. +. Float.abs want)
          | _ -> 0.
        in
        let bound =
          (lse_rounds n *. u) +. ((u +. narrow) *. Float.abs want) +. slack
        in
        at_most ~msg float_exact ~than:bound (Float.abs (y -. want)))
    rows

let arithmetic =
  group "arithmetic"
    [
      prop ~count:500
        "a Logsumexp is within its bound, NaNs and infinities as stated" lse
        law_logsumexp;
    ]

let () =
  exit
    (run "nx maps"
       [
         maps; checks; constants; forced; rules; donation; arithmetic;
       ])
