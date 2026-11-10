open Windtrap
open Tolk

let z n = `Int (Bigint.of_int n)
let ints = List.map z
let shape = List.map (fun n -> Ops.Int n)
let cpu = "CPU"

(* [p slot size] is storage of [size] 32-bit integers, viewed as [dims]. *)
let p ?(device = cpu) ?(dt = Dtype.Int32) ?dims slot size =
  let flat = Shape.param ~device ~shape:[ Int size ] slot dt in
  match dims with None -> flat | Some d -> Shape.reshape flat (shape d)

let iota n = Array.init n (fun j -> z j)
let eval ?(buffers = [ (1, iota 6) ]) u = Array.to_list (Tensors.eval ~buffers u)
let rejects f = raises_match (Exn.invalid_arg ?sub:None) f
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
          equal consts [ z 0 ] (eval (Shape.param 1 Int32)));
      test "a variable is refused" (fun () ->
          rejects (fun () -> eval (Ops.variable "n" (z 1) (z 3))));
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

let copies =
  group "copies"
    [
      test "a copy holds its source's elements" (fun () ->
          equal consts
            (ints [ 0; 1; 2 ])
            (eval (Ops.copy_to_device (p 1 3) "CPU:1")));
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
      test "an effect lands after those listed before it, whichever the sink \
            reaches first" (fun () ->
          let e = Ops.store out (p 1 3) in
          let a =
            Ops.after out [ Ops.store out (Shape.flip (p 1 3) [ 0 ]); e ]
          in
          equal write
            [ (0, 0, z 0); (0, 1, z 1); (0, 2, z 2) ]
            (writes (Ops.sink [ a ]));
          equal write (writes (Ops.sink [ a ])) (writes (Ops.sink [ e; a ]));
          equal write (writes (Ops.sink [ a ]))
            (writes (Ops.sink [ Ops.group [ e ]; a ])));
      test "call-local storage is memory of its own, and scratch" (fun () ->
          let scratch = Shape.alloc ~slot:1 ~device:cpu [ Int 3 ] Int32 in
          let staged = Ops.after scratch [ Ops.store scratch (p 1 3) ] in
          equal write
            [ (0, 0, z 0); (0, 1, z 1); (0, 2, z 2) ]
            (writes (into staged)));
      test "a call runs its body on its arguments" (fun () ->
          let body =
            Ops.sink
              [
                Ops.store
                  (Shape.param ~shape:[ Int 3 ] 0 Int32)
                  (Shape.param ~shape:[ Int 3 ] 1 Int32);
              ]
          in
          equal write
            [ (0, 0, z 0); (0, 1, z 1); (0, 2, z 2) ]
            (writes (Ops.sink [ Ops.call body [ out; p 1 3 ] ])));
      test "an argument a call writes is its storage as the call runs"
        (fun () ->
          let x = Shape.param ~shape:[ Int 3 ] 0 Int32
          and y = Shape.param ~shape:[ Int 3 ] 1 Int32 in
          let body = Ops.sink [ Ops.store x Ops.O.(y + x) ] in
          let twos = Ops.after out [ Ops.store out (Ops.int ~dtype:Int32 2) ] in
          let three = Ops.int ~dtype:Int32 3 in
          (* [out] names the state before the store, the call runs after it. *)
          let c = Ops.call body [ out; Ops.O.(three + twos) ] in
          equal write
            [ (0, 0, z 7); (0, 1, z 7); (0, 2, z 7) ]
            (writes (Ops.sink [ Ops.after out [ c ] ])));
      test
        "an after holds the latest of the state under it, the states its \
         writes of its storage leave and those its effects read, in either \
         order of a sink"
        (fun () ->
          let y = p 1 3 and o1 = p 2 3 and o2 = p 3 3 and o3 = p 4 3 in
          let w = p 5 3 in
          let int n = Ops.int ~dtype:Int32 n in
          let first = Ops.store out (int 1) in
          let ones = Ops.after out [ first ] in
          let twos = Ops.after ones [ Ops.store ones Ops.O.(ones + int 1) ] in
          (* A write of other storage leaves [out] in the state under it. *)
          let other = Ops.after ones [ Ops.store y (int 5) ] in
          (* A write listed again names the state it left. *)
          let again = Ops.after out [ first; Ops.store y (int 5) ] in
          (* A write of other storage that reads a state of [out] leaves
             [out] in that state. *)
          let read = Ops.after out [ Ops.store w Ops.O.(ones + int 4) ] in
          let outputs =
            [
              twos;
              Ops.after o1 [ Ops.store o1 other ];
              Ops.after o2 [ Ops.store o2 again ];
              Ops.after o3 [ Ops.store o3 read ];
            ]
          in
          let into_outputs u =
            List.filter
              (fun (s, _, _) -> List.mem s [ 2; 3; 4 ])
              (writes (Ops.sink u))
          in
          let expected =
            List.concat_map
              (fun s -> List.init 3 (fun i -> (s, i, z 1)))
              [ 2; 3; 4 ]
          in
          equal write expected (into_outputs outputs);
          equal write expected (into_outputs (List.rev outputs)));
      test
        "a loop calls its body once a trip, each reading the last's stores \
         into the storage its after names"
        (fun () ->
          let x = Shape.param ~shape:[ Int 3 ] 0 Int32 in
          let body = Ops.sink [ Ops.store x Ops.O.(x + x) ] in
          let trips = Ops.range ~axis_type:Loop (Int 3) [ 0 ] in
          equal write
            [ (0, 0, z 0); (0, 1, z 8); (0, 2, z 16) ]
            (writes
               (Ops.sink
                  [ Ops.after out [ Ops.end_ (Ops.call body [ out ]) [ trips ] ] ])));
      test
        "a loop no after lists reads the storage its call writes as the last \
         trip left it"
        (fun () ->
          let row = Shape.param ~shape:[ Int 2 ] 0 Int32
          and first = Shape.param ~shape:[ Int 2 ] 1 Int32 in
          let body = Ops.sink [ Ops.store row Ops.O.(first + int 1) ] in
          let r = Ops.range ~axis_type:Loop (Int 3) [ 0 ] in
          let y = p 1 6 in
          let rows = Shape.shrink y [ Some (Sym Ops.O.(r * int 2), Sym Ops.O.((r * int 2) + int 2)) ] in
          let call = Ops.call body [ rows; Shape.shrink y [ Some (Int 0, Int 2) ] ] in
          equal write
            [ (1, 0, z 1); (1, 1, z 2); (1, 2, z 2); (1, 3, z 3); (1, 4, z 2); (1, 5, z 3) ]
            (writes (Ops.sink [ Ops.end_ call [ r ] ])));
      test
        "nested loops run as one, the first range listed and the outer end's \
         varying slowest"
        (fun () ->
          let row = Shape.param ~shape:[ Int 2 ] 0 Int32
          and first = Shape.param ~shape:[ Int 2 ] 1 Int32 in
          let body = Ops.sink [ Ops.store row Ops.O.(first + int 1) ] in
          let o = Ops.range ~axis_type:Loop (Int 2) [ 0 ]
          and i = Ops.range ~axis_type:Loop (Int 2) [ 1 ] in
          let y = p 1 8 in
          (* Trip [o, i] stores into row [o + 2 i] row 1 plus one. *)
          let at = Ops.O.(o + (i * int 2)) in
          let rows = Shape.shrink y [ Some (Sym Ops.O.(at * int 2), Sym Ops.O.((at * int 2) + int 2)) ] in
          let call = Ops.call body [ rows; Shape.shrink y [ Some (Int 2, Int 4) ] ] in
          let writes e = Tensors.writes ~buffers:[ (1, iota 8) ] (Ops.sink [ e ]) in
          let o_slowest = ints [ 3; 4; 3; 4; 3; 4; 4; 5 ]
          and i_slowest = ints [ 3; 4; 3; 4; 4; 5; 4; 5 ] in
          let in_y xs = List.mapi (fun j x -> (1, j, x)) xs in
          equal write (in_y o_slowest) (writes (Ops.end_ call [ o; i ]));
          equal write (in_y i_slowest) (writes (Ops.end_ call [ i; o ]));
          equal write (in_y o_slowest)
            (writes (Ops.end_ (Ops.end_ call [ i ]) [ o ])));
      test "a symbolic shape is refused" (fun () ->
          let n = Ops.variable "n" (z 1) (z 3) in
          rejects (fun () -> eval (Shape.shrink (p 1 6) [ Some (Int 0, Sym n) ])));
    ]

let () =
  exit
    (run "Tensors"
       [
         storage;
         movements;
         arithmetic;
         copies;
         effects;
       ])
