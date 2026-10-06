open Windtrap
open Tolk

let z n = `Int (Bigint.of_int n)
let ints = List.map z
let shape = List.map (fun n -> Ops.Int n)
let cpu = Ops.Single "CPU"
let two = Ops.Multi [ "CPU:0"; "CPU:1" ]

(* [p slot size] is storage of [size] 32-bit integers, viewed as [dims]. *)
let p ?(device = cpu) ?(dt = Dtype.Int32) ?dims slot size =
  let flat = Call.param ~device ~shape:[ Int size ] slot dt in
  match dims with None -> flat | Some d -> Shape.reshape flat (shape d)

let iota n = Array.init n (fun j -> z j)
let values = list (array Dtypes.const)
let one = function [ v ] -> Array.to_list v | _ -> fail "one device"
let eval ?(buffers = [ (1, iota 6) ]) u = one (Tensors.eval ~buffers u)
let rejects f = raises_match (Exn.invalid_arg ?substring:None) f
let consts = list Dtypes.const

let storage =
  group "storage"
    [
      test "storage holds its memory's elements" (fun () ->
          equal consts (ints [ 0; 1; 2; 3; 4; 5 ]) (eval (p 1 6)));
      test "memory not given holds Invalid" (fun () ->
          equal consts [ `Invalid; `Invalid ] (eval ~buffers:[] (p 1 2)));
      test "memory read past its array is refused" (fun () ->
          rejects (fun () -> eval ~buffers:[ (1, iota 2) ] (p 1 3)));
      test "scalar storage holds one element" (fun () ->
          equal consts
            [ z 0 ]
            (eval (Ops.v Param ~arg:(Param (Ops.param_arg ~slot:1 Int32)))));
      test "a variable is refused" (fun () ->
          rejects (fun () -> eval (Ops.variable "n" (z 1) (z 3))));
      test "device k of storage holds the k-th run of its memory" (fun () ->
          equal values
            [ [| z 0; z 1; z 2 |]; [| z 3; z 4; z 5 |] ]
            (Tensors.eval ~buffers:[ (1, iota 6) ] (p ~device:two 1 3)));
    ]

let movements =
  let m = p ~dims:[ 2; 3 ] 1 6 in
  group "movements"
    [
      test "a reshape keeps the order" (fun () ->
          equal consts
            (ints [ 0; 1; 2; 3; 4; 5 ])
            (eval (Shape.reshape m (shape [ 3; 2 ]))));
      test "a permute transposes" (fun () ->
          equal consts
            (ints [ 0; 3; 1; 4; 2; 5 ])
            (eval (Shape.permute m [ 1; 0 ])));
      test "a flip reverses an axis" (fun () ->
          equal consts (ints [ 2; 1; 0; 5; 4; 3 ]) (eval (Shape.flip m [ 1 ])));
      test "a shrink keeps a box" (fun () ->
          equal consts
            (ints [ 4; 5 ])
            (eval (Shape.shrink m [ Some (Int 1, Int 2); Some (Int 1, Int 3) ])));
      test "a pad adds zeros" (fun () ->
          equal consts
            (ints [ 0; 0; 1; 2; 0 ])
            (eval (Shape.pad (p 1 3) [ Some (Int 1, Int 1) ])));
      test "an expand broadcasts" (fun () ->
          equal consts
            (ints [ 0; 1; 0; 1 ])
            (eval (Shape.expand (p 1 2) (shape [ 2; 2 ]))));
    ]

let arithmetic =
  group "arithmetic"
    [
      test "an operation broadcasts its sources" (fun () ->
          equal consts
            (ints [ 0; 2; 4; 3; 5; 7 ])
            (eval
               Ops.O.(
                 p ~dims:[ 2; 3 ] 1 6 + Shape.reshape (p 1 3) (shape [ 1; 3 ]))));
      test "an operation rounds to its type" (fun () ->
          equal consts (ints [ -128 ])
            (eval
               ~buffers:[ (1, [| z 127 |]) ]
               (Ops.cast
                  Ops.O.(Ops.cast (p 1 1) Int8 + Ops.int ~dtype:Int8 1)
                  Int32)));
      test "Invalid poisons what reads it, and a selection picks its branch"
        (fun () ->
          let x = p 1 2 in
          equal consts [ `Invalid; `Invalid ] (eval ~buffers:[] Ops.O.(x + x));
          equal consts
            (ints [ 7; 7 ])
            (eval ~buffers:[]
               (Ops.where (Ops.bool true) (Ops.int ~dtype:Int32 7) x)));
      test "a bitcast to a narrower type splits each element, low bytes first"
        (fun () ->
          equal consts
            (ints [ 0x02; 0x01; 0x04; 0x03 ])
            (eval
               ~buffers:[ (1, [| z 0x0102; z 0x0304 |]) ]
               (Ops.bitcast (p ~dt:Uint16 1 2) Uint8)));
      test "a bitcast to a wider type joins each row's elements" (fun () ->
          equal consts
            (ints [ 0x0102; 0x0304 ])
            (eval
               ~buffers:[ (1, [| z 2; z 1; z 4; z 3 |]) ]
               (Ops.bitcast (p ~dt:Uint8 ~dims:[ 2; 2 ] 1 4) Uint16)));
      test "a reduction folds its leading axes, the last fastest" (fun () ->
          equal consts
            (ints [ 3; 5; 7 ])
            (eval (Shape.rop (p ~dims:[ 2; 3 ] 1 6) Add [ 0 ])));
      test "a reduction of nothing is its identity" (fun () ->
          equal consts
            [ `Float Float.neg_infinity ]
            (eval ~buffers:[] (Shape.rop (p ~dt:Float32 1 0) Max [ 0 ])));
      test "a stack stacks along a new leading axis" (fun () ->
          equal consts
            (ints [ 0; 1; 1; 0 ])
            (eval (Shape.stack [ p 1 2; Shape.flip (p 1 2) [ 0 ] ])));
    ]

let devices =
  let x = p ~device:two 1 3 in
  group "devices"
    [
      test "a shard selection is one device's value" (fun () ->
          equal consts (ints [ 3; 4; 5 ]) (eval (Ops.mselect x 1)));
      test "a gather stacks each source's device" (fun () ->
          equal values
            [ [| z 3; z 4; z 5 |]; [| z 0; z 1; z 2 |] ]
            (Tensors.eval
               ~buffers:[ (1, iota 6) ]
               (Ops.mstack (Ops.mselect x 1) [ Ops.mselect x 0 ])));
      test "a copy places its source on each target device" (fun () ->
          equal values
            [ [| z 3; z 4; z 5 |]; [| z 3; z 4; z 5 |] ]
            (Tensors.eval
               ~buffers:[ (1, iota 6) ]
               (Ops.copy_to_device (Ops.mselect x 1) two)));
      test "an allreduce folds the devices, element by element" (fun () ->
          equal values
            [ [| z 3; z 5; z 7 |] ]
            (Tensors.eval ~buffers:[ (1, iota 6) ] (Ops.allreduce x Add cpu)));
      test "an operation combines device by device, one device for all"
        (fun () ->
          equal values
            [ [| z 1; z 2; z 3 |]; [| z 4; z 5; z 6 |] ]
            (Tensors.eval
               ~buffers:[ (1, iota 6) ]
               Ops.O.(x + Ops.int ~dtype:Int32 1)));
    ]

let effects =
  let out = p 0 3 in
  let into value = Ops.sink [ Ops.after out [ Ops.store out value ] ] in
  let writes u = Tensors.writes ~buffers:[ (0, iota 3); (1, iota 6) ] u in
  let write = list (triple int int Dtypes.value) in
  group "effects"
    [
      test "a store writes its value, broadcast, through its destination"
        (fun () ->
          equal write
            [ (0, 0, z 7); (0, 1, z 7); (0, 2, z 7) ]
            (writes (into (Ops.int ~dtype:Int32 7))));
      test "a store through a movement writes where it views" (fun () ->
          equal write
            [ (0, 0, z 2); (0, 1, z 1); (0, 2, z 0) ]
            (writes (Ops.sink [ Ops.store (Shape.flip out [ 0 ]) (p 1 3) ])));
      test "storage reads memory before any store, and an after once they ran"
        (fun () ->
          let reversed = Shape.flip out [ 0 ] in
          equal write
            [
              (0, 0, z 2);
              (0, 1, z 1);
              (0, 2, z 0);
              (1, 0, z 2);
              (1, 1, z 1);
              (1, 2, z 0);
            ]
            (writes
               (Ops.sink
                  [
                    Ops.store (p 1 3) (Ops.after out [ Ops.store out reversed ]);
                  ])));
      test "call-local storage is memory of its own, and scratch" (fun () ->
          let scratch = Call.alloc ~slot:1 ~device:cpu [ Int 3 ] Int32 in
          let staged = Ops.after scratch [ Ops.store scratch (p 1 3) ] in
          equal write
            [ (0, 0, z 0); (0, 1, z 1); (0, 2, z 2) ]
            (writes (into staged)));
      test "a call runs its body on its arguments" (fun () ->
          let body =
            Ops.sink
              [
                Ops.store
                  (Call.param ~shape:[ Int 3 ] 0 Int32)
                  (Call.param ~shape:[ Int 3 ] 1 Int32);
              ]
          in
          equal write
            [ (0, 0, z 0); (0, 1, z 1); (0, 2, z 2) ]
            (writes (Ops.sink [ Ops.call body [ out; p 1 3 ] ])));
      test "a loop calls its body once a trip, each reading the last's stores"
        (fun () ->
          let x = Call.param ~shape:[ Int 3 ] 0 Int32 in
          let body = Ops.sink [ Ops.store x Ops.O.(x + x) ] in
          let trips = Ops.range ~axis_type:Loop (Int 3) [ 0 ] in
          equal write
            [ (0, 0, z 0); (0, 1, z 8); (0, 2, z 16) ]
            (writes (Ops.sink [ Ops.end_ (Ops.call body [ out ]) [ trips ] ])));
      test "a symbolic shape is refused" (fun () ->
          let n = Ops.variable "n" (z 1) (z 3) in
          rejects (fun () -> eval (Shape.shrink (p 1 6) [ Some (Int 0, Sym n) ])));
    ]

(* Sharded values *)

let four = Ops.Multi [ "CPU:0"; "CPU:1"; "CPU:2"; "CPU:3" ]
let device_range n = Ops.range ~axis_type:Device (Int n) [ -1 ]

let shards =
  let x = p ~device:two 1 3 in
  group "sharded values"
    [
      test "a movement moves device k's value with its device range at k"
        (fun () ->
          let d = device_range 2 in
          let start = Ops.O.(d * Ops.int 2) in
          equal values
            [ [| z 0; z 1 |]; [| z 2; z 3 |] ]
            (Tensors.eval
               ~buffers:[ (1, iota 6) ]
               (Shape.mop
                  (Ops.copy_to_device (p 1 4) two)
                  (Shrink [ (Sym start, Int 2) ]))));
      test "an unshard reassembles its devices' parts" (fun () ->
          equal consts (ints [ 0; 1; 2; 3; 4; 5 ]) (eval (Call.unshard x [ 0 ])));
      test "an unshard of two axes places each part at its ranges' values"
        (fun () ->
          let d = device_range 4 in
          let rows = Ops.O.(d // int 2) and cols = Ops.O.(d % int 2) in
          let x = p ~device:four ~dims:[ 1; 1 ] 1 1 in
          equal consts
            (ints [ 0; 1; 2; 3 ])
            (eval
               ~buffers:[ (1, iota 4) ]
               (Call.unshard ~ranges:[ rows; cols ] x [ 0; 1 ])));
      test "a value sharded across a kernel's threads is refused" (fun () ->
          let threads = Ops.range ~axis_type:Local (Int 2) [ 0 ] in
          rejects (fun () -> eval (Call.unshard ~ranges:[ threads ] x [ 0 ])));
      test "a copy of a value on several devices is its first device's"
        (fun () ->
          equal consts (ints [ 0; 1; 2 ]) (eval (Ops.copy_to_device x cpu)));
    ]

let sharded_effects =
  let out = Call.unshard (p ~device:two 0 2) [ 0 ] in
  let write = list (triple int int Dtypes.value) in
  let writes value =
    Tensors.writes
      ~buffers:[ (1, iota 4); (2, iota 8) ]
      (Ops.sink [ Ops.store out value ])
  in
  group "stores of sharded values"
    [
      test "a store into a sharded destination writes each device's part"
        (fun () ->
          equal write
            [ (0, 0, z 0); (0, 1, z 1); (0, 2, z 2); (0, 3, z 3) ]
            (writes (p 1 4)));
      test "an element takes the value of the device whose memory it views"
        (fun () ->
          equal write
            [ (0, 0, z 0); (0, 1, z 1); (0, 2, z 6); (0, 3, z 7) ]
            (writes (p ~device:two 2 4)));
    ]

let sharded_calls =
  let out = Call.unshard (p ~device:two 0 2) [ 0 ] in
  let param slot = Call.param ~device:two ~shape:[ Int 2 ] slot Int32 in
  let body =
    Ops.sink
      [ Ops.store (Call.unshard (param 0) [ 0 ]) (Call.unshard (param 1) [ 0 ]) ]
  in
  group "calls on sharded values"
    [
      test "a parameter of a sharded argument holds its parts" (fun () ->
          equal
            (list (triple int int Dtypes.value))
            [ (0, 0, z 0); (0, 1, z 1); (0, 2, z 2); (0, 3, z 3) ]
            (Tensors.writes
               ~buffers:[ (1, iota 4) ]
               (Ops.sink
                  [
                    Ops.call body [ out; Call.unshard (p ~device:two 1 2) [ 0 ] ];
                  ])));
    ]

let () =
  exit
    (run "Tensors"
       [
         storage;
         movements;
         arithmetic;
         devices;
         shards;
         effects;
         sharded_effects;
         sharded_calls;
       ])
