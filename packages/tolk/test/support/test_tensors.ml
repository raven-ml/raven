open Windtrap
open Tolk

let z n = `Int (Bigint.of_int n)
let ints = List.map z
let shape = List.map (fun n -> Ops.Int n)
let cpu = "CPU"
let int32 n = Ops.int ~dtype:Int32 n

(* [p slot size] is storage of [size] 32-bit integers, viewed as [dims]. *)
let p ?(device = cpu) ?(dt = Dtype.Int32) ?dims slot size =
  let flat = Shape.param ~device ~shape:[ Int size ] slot dt in
  match dims with None -> flat | Some d -> Shape.reshape flat (shape d)

let iota n = Array.init n (fun j -> z j)
let eval ~buffers u = Array.to_list (Tensors.eval ~buffers u)
let consts = list Dtypes.const
let rejects f = raises_match (Exn.invalid_arg ?sub:None) f

(* Effects write into [out], memory 0 holding 0, 1, 2, and memory 1 holds 0 to
   5. *)
let out = p 0 3
let write = list (triple int int Dtypes.value)
let writes u = Tensors.writes ~buffers:[ (0, iota 3); (1, iota 6) ] u
let into value = Ops.sink [ Ops.after out [ Ops.store out value ] ]

(* [element r] is the box of the [r]th element of a vector. *)
let element r = [ Some (Ops.Sym r, Ops.Sym Ops.O.(r + int32 1)) ]

let storage =
  group "storage"
    [
      test "a parameter of slot s holds the elements of memory s" (fun () ->
          equal consts
            (ints [ 0; 1; 2; 3; 4; 5 ])
            (eval ~buffers:[ (0, iota 2); (1, iota 6) ] (p 1 6)));
      test "a buffer of slot s holds the elements of memory s" (fun () ->
          equal consts
            (ints [ 0; 1; 2 ])
            (eval ~buffers:[ (1, iota 3) ] (Ops.new_buffer ~slot:1 cpu 3 Int32)));
      test "scalar storage holds one element" (fun () ->
          equal consts
            [ z 7 ]
            (eval ~buffers:[ (1, [| z 7 |]) ] (Shape.param 1 Int32)));
      test
        "call-local storage is memory apart from slot s's, Invalid until \
         written" (fun () ->
          equal consts [ `Invalid; `Invalid ]
            (eval
               ~buffers:[ (1, iota 2) ]
               (Shape.alloc ~slot:1 ~device:cpu [ Int 2 ] Int32)));
    ]

let variables =
  group "variables"
    [
      test "a bound variable is the value it is bound to" (fun () ->
          equal consts
            [ z 5 ]
            (eval ~buffers:[]
               (Shape.bind (Ops.variable ~dtype:Int32 "n" (z 0) (z 9)) (z 5))));
    ]

let movements =
  let m = p ~dims:[ 2; 3 ] 1 6 in
  let eval = eval ~buffers:[ (1, iota 6) ] in
  group "movements"
    [
      test "a reshape keeps the order" (fun () ->
          equal consts
            (ints [ 0; 1; 2; 3; 4; 5 ])
            (eval (Shape.reshape m (shape [ 3; 2 ]))));
      test "a permute reorders axes" (fun () ->
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

let elementwise =
  let x = p 1 2 in
  group "elementwise operations"
    [
      test "an operation broadcasts its sources as numpy does" (fun () ->
          equal consts
            (ints [ 0; 2; 4; 3; 5; 7 ])
            (eval
               ~buffers:[ (1, iota 6) ]
               Ops.O.(
                 p ~dims:[ 2; 3 ] 1 6 + Shape.reshape (p 1 3) (shape [ 1; 3 ]))));
      test "an operation rounds each element to its type" (fun () ->
          equal consts
            [ z (-128) ]
            (eval
               ~buffers:[ (1, [| z 127 |]) ]
               (Ops.cast
                  Ops.O.(Ops.cast (p 1 1) Int8 + Ops.int ~dtype:Int8 1)
                  Int32)));
      (* Memory 1 is not given, so [x] is Invalid. *)
      cases "Invalid poisons what reads it" ~id:fst
        [
          ("an operand", Ops.O.(x + x));
          ( "a selection's condition",
            Ops.where Ops.O.(x < int32 1) (int32 7) (int32 8) );
        ]
        (fun (_, u) -> equal consts [ `Invalid; `Invalid ] (eval ~buffers:[] u));
      cases "a selection picks its branch" ~id:fst
        [
          ("a branch beside Invalid", (true, ints [ 7; 7 ]));
          ("an Invalid branch", (false, [ `Invalid; `Invalid ]));
        ]
        (fun (_, (c, expected)) ->
          equal consts expected
            (eval ~buffers:[] (Ops.where (Ops.bool c) (int32 7) x)));
      cases
        "a bit reinterpretation into another size reads each row's bytes, \
         little-endian"
        ~id:(fun (name, _, _) -> name)
        [
          ( "into a narrower type",
            ([| z 0x0102; z 0x0304 |], Ops.bitcast (p ~dt:Uint16 1 2) Uint8),
            ints [ 0x02; 0x01; 0x04; 0x03 ] );
          ( "into a wider type",
            ( [| z 2; z 1; z 4; z 3 |],
              Ops.bitcast (p ~dt:Uint8 ~dims:[ 2; 2 ] 1 4) Uint16 ),
            ints [ 0x0102; 0x0304 ] );
        ]
        (fun (_, (memory, u), expected) ->
          equal consts expected (eval ~buffers:[ (1, memory) ] u));
    ]

let reductions =
  group "reductions"
    [
      (* Float32 rounds 2^24 + 1 to 2^24 (ties to even), so the order of a sum
         shows: column 0 holds 2^24, 1, -2^24, 1, which adds to 1 in that order
         and to 2 with the first axis varying fastest. *)
      test "a reduction folds its leading axes, the last of them fastest"
        (fun () ->
          let f x = `Float x in
          let column0 = [ 16777216.; 1.; -16777216.; 1. ] in
          let column1 = [ 1.; 2.; 3.; 4. ] in
          let memory =
            Array.of_list
              (List.concat
                 (List.map2 (fun a b -> [ f a; f b ]) column0 column1))
          in
          equal consts
            [ f 1.; f 10. ]
            (eval
               ~buffers:[ (1, memory) ]
               (Shape.rop (p ~dt:Float32 ~dims:[ 2; 2; 2 ] 1 8) Add [ 0; 1 ])));
      test "a reduction of nothing is its identity" (fun () ->
          equal consts
            [ `Float Float.neg_infinity ]
            (eval ~buffers:[] (Shape.rop (p ~dt:Float32 1 0) Max [ 0 ])));
    ]

let stacks_and_copies =
  let eval = eval ~buffers:[ (1, iota 3) ] in
  group "stacks and copies"
    [
      test "a stack stacks its sources along a new leading axis" (fun () ->
          equal consts
            (ints [ 0; 1; 1; 0 ])
            (eval (Shape.stack [ p 1 2; Shape.flip (p 1 2) [ 0 ] ])));
      test "a stage holds its source's elements" (fun () ->
          equal consts
            (ints [ 1; 2; 3 ])
            (eval (Ops.v Stage ~src:[ Ops.O.(p 1 3 + int32 1) ])));
      test "a copy holds its source's elements" (fun () ->
          equal consts
            (ints [ 0; 1; 2 ])
            (eval (Ops.copy_to_device (p 1 3) "CPU:1")));
    ]

(* A source of three rows of two, and indices into its rows (memory 2) and
   columns (memory 3). *)
let indexes =
  let source = p ~dims:[ 3; 2 ] 1 6 in
  let buffers = [ (1, iota 6); (2, [| z 2; z 0 |]); (3, [| z 1 |]) ] in
  group "indexes"
    [
      cases "an index gathers the source's elements at its indices' positions"
        ~id:fst
        [
          ( "by one index, rest of the rows kept",
            ([ p 2 2 ], ints [ 4; 5; 0; 1 ]) );
          ("by two indices", ([ p 2 2; p 3 1 ], ints [ 5; 1 ]));
        ]
        (fun (_, (indices, expected)) ->
          equal consts expected (eval ~buffers (Ops.index source indices)));
      test "an element at an Invalid index is Invalid" (fun () ->
          equal consts [ `Invalid; `Invalid ]
            (eval ~buffers:[ (1, iota 6) ] (Ops.index (p 1 6) [ p 2 2 ])));
      test "a store through an Invalid index writes nothing" (fun () ->
          equal write []
            (Tensors.writes ~buffers:[]
               (Ops.sink [ Ops.store (Ops.index (p 0 3) [ p 2 2 ]) (int32 7) ])));
    ]

(* The output of each row is 1s: an after holds the latest of the state under it
   (here [ones], since [twos] is no source), the states its sources' writes of
   its storage leave, and the states of its storage they read. *)
let latest_state =
  let ones_store = Ops.store out (int32 1) in
  let ones = Ops.after out [ ones_store ] in
  let twos = Ops.after ones [ Ops.store ones Ops.O.(ones + int32 1) ] in
  let o = p 2 3 in
  cases "an after holds the latest of its states, in either order of a sink"
    ~id:fst
    [
      (* A write of other storage leaves [out] in the state under it. *)
      ( "over a write of other storage",
        Ops.after ones [ Ops.store (p 1 3) (int32 5) ] );
      (* A write listed again names the state it left. *)
      ( "listing a write again",
        Ops.after out [ ones_store; Ops.store (p 1 3) (int32 5) ] );
      (* A write of other storage that reads a state of [out] leaves [out] in
         that state. *)
      ( "over a write that reads a state",
        Ops.after out [ Ops.store (p 5 3) Ops.O.(ones + int32 4) ] );
    ]
    (fun (_, state) ->
      let output = Ops.after o [ Ops.store o state ] in
      let into_o u =
        List.filter (fun (s, _, _) -> s = 2) (writes (Ops.sink u))
      in
      let ones = List.init 3 (fun i -> (2, i, z 1)) in
      equal write ones (into_o [ twos; output ]);
      equal write ones (into_o [ output; twos ]))

let effects =
  group "effects"
    [
      test "a store writes its value, broadcast, into the memory it views"
        (fun () ->
          equal write
            [ (0, 0, z 7); (0, 1, z 7); (0, 2, z 7) ]
            (writes (into (int32 7))));
      test "a store through a movement writes where it views" (fun () ->
          equal write
            [ (0, 0, z 2); (0, 1, z 1); (0, 2, z 0) ]
            (writes (Ops.sink [ Ops.store (Shape.flip out [ 0 ]) (p 1 3) ])));
      cases
        "a store through a bit reinterpretation writes its bytes, each element \
         keeping its other bytes"
        ~id:fst
        [
          ( "bytes 2 to 5",
            ( Ops.store
                (Shape.shrink (Ops.bitcast out Uint8) [ Some (Int 2, Int 6) ])
                (Ops.const ~dtype:Uint8 (z 0xAA)),
              [ (0, 0, z (Int32.to_int 0xAAAA0000l)); (0, 1, z 0xAAAA) ] ) );
          ( "a float of one size",
            ( Ops.store
                (Shape.shrink (Ops.bitcast out Float32) [ Some (Int 1, Int 2) ])
                (Ops.float ~dtype:Float32 1.),
              [ (0, 1, z 0x3F800000) ] ) );
          ( "two elements an int64",
            ( Ops.store
                (Ops.bitcast
                   (Shape.shrink (p 1 6) [ Some (Int 2, Int 6) ])
                   Int64)
                (Ops.const ~dtype:Int64 (z ((1 lsl 32) + 5))),
              [ (1, 2, z 5); (1, 3, z 1); (1, 4, z 5); (1, 5, z 1) ] ) );
        ]
        (fun (_, (store, expected)) ->
          equal write expected (writes (Ops.sink [ store ])));
      test "storage is read as the memory held before any store" (fun () ->
          equal write
            [
              (0, 0, z 7);
              (0, 1, z 7);
              (0, 2, z 7);
              (1, 0, z 0);
              (1, 1, z 1);
              (1, 2, z 2);
            ]
            (writes
               (Ops.sink [ Ops.store out (int32 7); Ops.store (p 1 3) out ])));
      test "an after holds its storage as the stores it lists left it"
        (fun () ->
          let reversed =
            Ops.after out [ Ops.store out (Shape.flip out [ 0 ]) ]
          in
          equal write
            [
              (0, 0, z 2);
              (0, 1, z 1);
              (0, 2, z 0);
              (1, 0, z 2);
              (1, 1, z 1);
              (1, 2, z 0);
            ]
            (writes (Ops.sink [ Ops.store (p 1 3) reversed ])));
      latest_state;
      test "a join holds the value its first source has" (fun () ->
          let nines = Ops.after (p 1 3) [ Ops.store (p 1 3) (int32 9) ] in
          equal consts
            (ints [ 0; 1; 2 ])
            (eval ~buffers:[ (1, iota 3) ] (Ops.after (p 1 3) [ nines ])));
      test "an effect runs after those an after lists before it" (fun () ->
          let e = Ops.store out (p 1 3) in
          let a =
            Ops.after out [ Ops.store out (Shape.flip (p 1 3) [ 0 ]); e ]
          in
          equal write
            [ (0, 0, z 0); (0, 1, z 1); (0, 2, z 2) ]
            (writes (Ops.sink [ a ])));
      (* Run again, the flip would leave 2, 1, 0. *)
      cases "each node runs once, wherever the sink reaches it" ~id:fst
        [ ("first", `First); ("in a group, first", `Group); ("last", `Last) ]
        (fun (_, at) ->
          let e = Ops.store out (Shape.flip (p 1 3) [ 0 ]) in
          let a = Ops.after out [ e; Ops.store out (p 1 3) ] in
          let sink =
            match at with
            | `First -> [ e; a ]
            | `Group -> [ Ops.group [ e; Ops.store (p 2 3) (int32 9) ]; a ]
            | `Last -> [ a; e ]
          in
          let nines =
            if at = `Group then List.init 3 (fun i -> (2, i, z 9)) else []
          in
          equal write
            ([ (0, 0, z 0); (0, 1, z 1); (0, 2, z 2) ] @ nines)
            (writes (Ops.sink sink)));
      test "a call runs its body, parameter k standing for argument k"
        (fun () ->
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
      test "a call reshapes each argument to its parameter's shape" (fun () ->
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
            (writes (Ops.sink [ Ops.call body [ out; p ~dims:[ 1; 3 ] 1 3 ] ])));
      test "an argument a call writes is memory once every argument has run"
        (fun () ->
          let x = Shape.param ~shape:[ Int 3 ] 0 Int32
          and y = Shape.param ~shape:[ Int 3 ] 1 Int32 in
          let body = Ops.sink [ Ops.store x Ops.O.(y + x) ] in
          let twos = Ops.after out [ Ops.store out (int32 2) ] in
          (* [out] names the state before the store, the call runs after it. *)
          let c = Ops.call body [ out; Ops.O.(int32 3 + twos) ] in
          equal write
            [ (0, 0, z 7); (0, 1, z 7); (0, 2, z 7) ]
            (writes (Ops.sink [ Ops.after out [ c ] ])));
      test "a loop calls its body once a trip, each reading the trips before"
        (fun () ->
          let x = Shape.param ~shape:[ Int 3 ] 0 Int32 in
          let body = Ops.sink [ Ops.store x Ops.O.(x + x) ] in
          let trips = Ops.range ~axis_type:Loop (Int 3) [ 0 ] in
          equal write
            [ (0, 0, z 0); (0, 1, z 8); (0, 2, z 16) ]
            (writes
               (Ops.sink
                  [
                    Ops.after out [ Ops.end_ (Ops.call body [ out ]) [ trips ] ];
                  ])));
      test
        "a loop no after lists reads the storage its call writes as the trips \
         before left it" (fun () ->
          let row = Shape.param ~shape:[ Int 2 ] 0 Int32
          and first = Shape.param ~shape:[ Int 2 ] 1 Int32 in
          let body = Ops.sink [ Ops.store row Ops.O.(first + int32 1) ] in
          let r = Ops.range ~axis_type:Loop (Int 3) [ 0 ] in
          let y = p 1 6 in
          let rows =
            Shape.shrink y
              [
                Some
                  (Sym Ops.O.(r * int32 2), Sym Ops.O.((r * int32 2) + int32 2));
              ]
          in
          let call =
            Ops.call body [ rows; Shape.shrink y [ Some (Int 0, Int 2) ] ]
          in
          equal write
            [
              (1, 0, z 1);
              (1, 1, z 2);
              (1, 2, z 2);
              (1, 3, z 3);
              (1, 4, z 2);
              (1, 5, z 3);
            ]
            (writes (Ops.sink [ Ops.end_ call [ r ] ])));
      test "a looped call given an older state reads it as it was" (fun () ->
          let x = Shape.param ~shape:[ Int 3 ] 0 Int32
          and y = Shape.param ~shape:[ Int 3 ] 1 Int32 in
          let body = Ops.sink [ Ops.store x Ops.O.(x + y) ] in
          let ones = Ops.after out [ Ops.store out (int32 1) ] in
          let trips = Ops.range ~axis_type:Loop (Int 3) [ 0 ] in
          (* Each trip adds the elements [out] held before the store of ones. *)
          equal write
            [ (0, 0, z 1); (0, 1, z 4); (0, 2, z 7) ]
            (writes
               (Ops.sink
                  [
                    Ops.after ones
                      [ Ops.end_ (Ops.call body [ ones; out ]) [ trips ] ];
                  ])));
      test
        "nested loops run as one, the first range and the outer end's slowest"
        (fun () ->
          let row = Shape.param ~shape:[ Int 2 ] 0 Int32
          and first = Shape.param ~shape:[ Int 2 ] 1 Int32 in
          let body = Ops.sink [ Ops.store row Ops.O.(first + int32 1) ] in
          let o = Ops.range ~axis_type:Loop (Int 2) [ 0 ]
          and i = Ops.range ~axis_type:Loop (Int 2) [ 1 ] in
          let y = p 1 8 in
          (* Trip [o, i] stores into row [o + 2 i] row 1 plus one. *)
          let at = Ops.O.(o + (i * int32 2)) in
          let rows =
            Shape.shrink y
              [
                Some
                  ( Sym Ops.O.(at * int32 2),
                    Sym Ops.O.((at * int32 2) + int32 2) );
              ]
          in
          let call =
            Ops.call body [ rows; Shape.shrink y [ Some (Int 2, Int 4) ] ]
          in
          let writes e =
            Tensors.writes ~buffers:[ (1, iota 8) ] (Ops.sink [ e ])
          in
          let o_slowest = ints [ 3; 4; 3; 4; 3; 4; 4; 5 ]
          and i_slowest = ints [ 3; 4; 3; 4; 4; 5; 4; 5 ] in
          let in_y xs = List.mapi (fun j x -> (1, j, x)) xs in
          equal write (in_y o_slowest) (writes (Ops.end_ call [ o; i ]));
          equal write (in_y i_slowest) (writes (Ops.end_ call [ i; o ]));
          equal write (in_y o_slowest)
            (writes (Ops.end_ (Ops.end_ call [ i ]) [ o ])));
      cases "an end of stores runs them once for each value of its ranges"
        ~id:fst
        (let flipped r =
           Ops.store
             (Shape.shrink out (element r))
             (Shape.shrink (p 1 3) (element Ops.O.(int32 2 - r)))
         in
         let nines r = Ops.store (Shape.shrink (p 2 3) (element r)) (int32 9) in
         let reversed = [ (0, 0, z 2); (0, 1, z 1); (0, 2, z 0) ] in
         [
           ("of a store", (flipped, reversed));
           ( "of a group of stores",
             ( (fun r -> Ops.group [ flipped r; nines r ]),
               reversed @ List.init 3 (fun i -> (2, i, z 9)) ) );
         ])
        (fun (_, (stores, expected)) ->
          let r = Ops.range ~axis_type:Loop (Int 3) [ 0 ] in
          equal write expected (writes (Ops.sink [ Ops.end_ (stores r) [ r ] ])));
    ]

let eval_tests =
  let n = Ops.variable ~dtype:Int32 "n" (z 1) (z 3) in
  group "eval"
    [
      test "memory not given holds Invalid" (fun () ->
          equal consts [ `Invalid; `Invalid ] (eval ~buffers:[] (p 1 2)));
      cases
        "passes the first source of an operation that only orders or names it"
        ~id:(fun op -> Format.asprintf "%a" Op.pp op)
        [ Op.Detach; Contiguous_backward ]
        (fun op ->
          equal consts
            (ints [ 0; 1 ])
            (eval ~buffers:[ (1, iota 2) ] (Ops.v op ~src:[ p 1 2 ])));
      cases "raises Invalid_argument" ~id:fst
        [
          ("reading memory past its array", fun () -> p 1 3);
          ( "reading a source outside its shape through an index",
            fun () -> Ops.index (p 1 2) [ p 2 1 ] );
          ( "of a shape that is not concrete",
            fun () -> Shape.shrink (p 1 2) [ Some (Int 0, Sym n) ] );
          ("reading an unbound variable", fun () -> n);
          ( "ordering an effect after a state that needs it",
            fun () ->
              let e = Ops.store (p 0 2) (int32 1) in
              let reads_e = Ops.store (p 5 2) (Ops.after (p 0 2) [ e ]) in
              Ops.sink [ Ops.after (Ops.after (p 0 2) [ reads_e ]) [ e ] ] );
          ( "of an operation outside the list",
            fun () -> Ops.range ~axis_type:Loop (Int 2) [ 0 ] );
        ]
        (fun (_, u) ->
          rejects (fun () ->
              eval ~buffers:[ (0, iota 2); (1, iota 2); (2, [| z 7 |]) ] (u ())));
    ]

let writes_tests =
  group "writes"
    [
      (* Memory 1 is not given, so element 0's value is Invalid. *)
      test "leaves out an element whose last value is Invalid" (fun () ->
          let first = Ops.O.(p 0 2 < int32 1) in
          equal write
            [ (0, 1, z 3) ]
            (Tensors.writes
               ~buffers:[ (0, iota 2) ]
               (Ops.sink
                  [ Ops.store (p 0 2) (Ops.where first (p 1 2) (int32 3)) ])));
      test "leaves out call-local storage, which is scratch" (fun () ->
          let scratch = Shape.alloc ~slot:1 ~device:cpu [ Int 3 ] Int32 in
          let staged = Ops.after scratch [ Ops.store scratch (p 1 3) ] in
          equal write
            [ (0, 0, z 0); (0, 1, z 1); (0, 2, z 2) ]
            (Tensors.writes
               ~buffers:[ (1, iota 3) ]
               (Ops.sink [ Ops.after (p 0 3) [ Ops.store (p 0 3) staged ] ])));
    ]

let () =
  exit
    (run "Tensors"
       [
         storage;
         variables;
         movements;
         elementwise;
         reductions;
         stacks_and_copies;
         indexes;
         effects;
         eval_tests;
         writes_tests;
       ])
