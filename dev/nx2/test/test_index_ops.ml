(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Gather, Scatter and Assemble through the engine's private modules, copied
   here: their rules raise before any kernel, a split placement computes what
   one device does, a donated target takes the result, and an assembly the
   kernels decline expands. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module M = Nx_array.Move
module C = Nx_support.Counting

type b

let m = Nx_support.memory
let s1 : b Devices.t = Devices.mint ~by:"t" ~kernels:(module C) [ m 0 ]
let s2 : b Devices.t = Devices.mint ~by:"t" ~kernels:(module C) [ m 0; m 1 ]
let at1 = Devices.one s1 0
let split = Devices.split ~by:"t" ~axis:0 s2

let on (type v s) (dt : (v, s) D.t) shape (data : v array) : (v, s, b) Value.t =
  Value.Array
    {
      at = at1;
      a = A.to_device (m 0) (A.of_array dt shape data);
      dead = Prim.live;
    }

let positions shape ps = on D.Int64 shape (Array.map Int64.of_int ps)

let elements (type v s) (x : (v, s, b) Value.t) : v array =
  match
    Repr.array
      (Place.value ~by:"t" (Devices.rebrand (Devices.one Devices.host 0)) x)
  with
  | Some a -> A.to_array a
  | None -> fail "no array"

let array_of (type v s) (x : (v, s, b) Value.t) : (v, s) A.t =
  match x with
  | Value.Array { a; _ } -> a
  | Value.Shards _ | Value.Donated _ | Value.Deferred _ | Value.Traced _ ->
      fail "not on one device"

let ints = array int32
let x23 () = on D.Int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |]
let gather axis idx x = Exec.run ~by:"t" (Value.Gather { axis; idx; x })

let scatter ?(unique = false) combine axis idx updates into =
  Exec.run ~by:"t" (Value.Scatter { combine; unique; axis; idx; updates; into })

let row i : M.range array =
  [| { start = i; count = 1; step = 1 }; { start = 0; count = 3; step = 1 } |]

let rules =
  group "rules"
    [
      test "a gather of the wrong rank raises before any kernel" (fun () ->
          let x = x23 () in
          C.reset ();
          raises
            (Invalid_argument
               "t: positions [2] do not fit an operand [2; 3] along axis 0")
            (fun () -> gather 0 (positions [| 2 |] [| 0; 1 |]) x);
          raises (Invalid_argument "t: axis 2 of an operand of rank 2")
            (fun () -> gather 2 (positions [| 2; 3 |] (Array.make 6 0)) x);
          equal ~msg:"kernel calls" int 0 (C.calls ()));
      test "a scatter's operands raise before any kernel" (fun () ->
          let x = x23 () in
          C.reset ();
          raises
            (Invalid_argument "t: positions [2; 1] and updates [2; 2] differ")
            (fun () ->
              scatter Set 1
                (positions [| 2; 1 |] [| 0; 0 |])
                (on D.Int32 [| 2; 2 |] [| 0l; 0l; 0l; 0l |])
                x);
          raises (Invalid_argument "t: Add does not take bool") (fun () ->
              let b = on D.Bool [| 2 |] [| true; false |] in
              scatter Add 0 (positions [| 2 |] [| 0; 1 |]) b b);
          equal ~msg:"kernel calls" int 0 (C.calls ()));
      test "an assembly's pieces raise before any kernel" (fun () ->
          C.reset ();
          raises
            (Invalid_argument "t: piece 0 has shape [2; 3], its region [1; 3]")
            (fun () ->
              Exec.run ~by:"t"
                (Value.Assemble
                   {
                     dtype = D.Int32;
                     shape = [| 2; 3 |];
                     fill = 0l;
                     pieces = [ (row 0, x23 ()) ];
                   }));
          equal ~msg:"kernel calls" int 0 (C.calls ()));
    ]

let placements =
  group "placements"
    [
      test "a split gather, scatter and assembly compute what one device does"
        (fun () ->
          let x = x23 () and idx = positions [| 2; 2 |] [| 2; 0; 1; 1 |] in
          let one = gather 1 idx x in
          let two =
            gather 1
              (Place.value ~by:"t" split idx)
              (Place.value ~by:"t" split x)
          in
          equal ~msg:"gather" ints (elements one) (elements two);
          equal ints [| 3l; 1l; 5l; 5l |] (elements one);
          let u = on D.Int32 [| 2; 2 |] [| 10l; 20l; 30l; 40l |] in
          let one = scatter Add 1 idx u x in
          let two =
            scatter Add 1
              (Place.value ~by:"t" split idx)
              (Place.value ~by:"t" split u)
              (Place.value ~by:"t" split x)
          in
          equal ~msg:"scatter" ints (elements one) (elements two);
          equal ints [| 21l; 2l; 13l; 4l; 75l; 6l |] (elements one);
          let a =
            Value.Assemble
              {
                dtype = D.Int32;
                shape = [| 3; 3 |];
                fill = -1l;
                pieces =
                  [
                    ( row 0,
                      Place.value ~by:"t" (Devices.one s2 1)
                        (on D.Int32 [| 1; 3 |] [| 7l; 8l; 9l |]) );
                    ( row 2,
                      Place.value ~by:"t" (Devices.on s2)
                        (on D.Int32 [| 1; 3 |] [| 1l; 2l; 3l |]) );
                  ];
              }
          in
          equal ~msg:"assembly" ints
            [| 7l; 8l; 9l; -1l; -1l; -1l; 1l; 2l; 3l |]
            (elements (Exec.run ~by:"t" a)));
    ]

let donation =
  group "donation"
    [
      test "a target whose memory a view shares is copied" (fun () ->
          let x = x23 () in
          let v = Exec.run ~by:"t" (Value.Move (Reshape [| 6 |], x)) in
          let y =
            scatter Set 1
              (positions [| 2; 1 |] [| 2; 0 |])
              (on D.Int32 [| 2; 1 |] [| 9l; 8l |])
              (Exec.donate ~by:"t" x)
          in
          equal ~msg:"a copy" bool false
            (Rig.Buffer.overlaps
               (A.buffer (array_of v))
               (A.buffer (array_of y)));
          equal ~msg:"the view keeps its elements" ints
            [| 1l; 2l; 3l; 4l; 5l; 6l |]
            (elements v));
    ]

let () = exit (run "nx index ops" [ rules; placements; donation ])
