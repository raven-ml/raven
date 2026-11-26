open Windtrap
open Tolk

let z n = `Int (Bigint.of_int n)
let cpu = "CPU"
let write = list (triple int int Dtypes.value)
let iota = Array.init 4 z
let tens = Array.make 4 (z 10)

(* [param ~dt slot] is storage of four elements of [dt] (default int32), as a
   kernel graph names it. *)
let param ?(dt = Dtype.Int32) ?(size = 4) slot =
  Shape.param ~device:cpu ~shape:[ Int size ] slot dt

(* [kernel f] is a kernel of storage parameters [0] and [1] that stores [f] of
   parameter [1]'s element into parameter [0]'s, over [count] elements (default
   four) from [first] (default [0]). *)
let kernel ?(dt = Dtype.Int32) ?(first = 0) ?(count = 4) f =
  let r = Ops.range ~axis_type:Loop (Int count) [ 0 ] in
  let at slot =
    Ops.index
      (Shape.param ~shape:[ Int 4 ] slot dt)
      [ Ops.O.(r + Ops.int first) ]
  in
  Ops.sink
    ~kernel:(Ops.kernel_info ~name:"k" ())
    [ Ops.end_ (Ops.store (at 0) (f (at 1))) [ r ] ]

let int32 n = Ops.int ~dtype:Int32 n
let incr = kernel (fun x -> Ops.O.(x + int32 1))

(* A kernel that adds the scalar parameter of slot [2]. *)
let n = Shape.param ~addrspace:Alu 2 Int32
let shifted = kernel (fun x -> Ops.O.(x + n))
let called k dst src = Ops.after dst [ Ops.call k [ dst; src ] ]

(* [into s values] is the writes of [values] into memory [s]. *)
let into s values = List.mapi (fun i v -> (s, i, v)) values
let ints = List.map z

let writes =
  group "writes"
    [
      test "a kernel writes into its call's argument" (fun () ->
          equal write
            (into 0 (ints [ 1; 2; 3; 4 ]))
            (Kernel_graphs.writes
               ~buffers:[ (1, iota) ]
               (Ops.sink [ called incr (param 0) (param 1) ])));
      test "a kernel reads storage as given where its argument names no after"
        (fun () ->
          let first = called incr (param 1) (param 2) in
          equal write
            (into 0 (ints [ 1; 2; 3; 4 ]) @ into 1 (ints [ 11; 11; 11; 11 ]))
            (Kernel_graphs.writes
               ~buffers:[ (1, iota); (2, tens) ]
               (Ops.sink [ called incr (param 0) (param 1); first ])));
      test "a kernel reads an after as the calls it lists leave it" (fun () ->
          let first = called incr (param 1) (param 2) in
          equal write
            (into 0 (ints [ 12; 12; 12; 12 ]) @ into 1 (ints [ 11; 11; 11; 11 ]))
            (Kernel_graphs.writes
               ~buffers:[ (2, tens) ]
               (Ops.sink [ called incr (param 0) first; first ])));
      test "an after of a call the state under it already follows is that state"
        (fun () ->
          let c = Ops.call incr [ param 0; param 1 ] in
          let zero = kernel (fun _ -> int32 0) in
          let zeros =
            Ops.after
              (Ops.after (param 0) [ c ])
              [ Ops.call zero [ param 0; param 1 ] ]
          in
          equal write
            (into 0 (ints [ 0; 0; 0; 0 ]))
            (Kernel_graphs.writes
               ~buffers:[ (1, iota) ]
               (Ops.sink [ Ops.after zeros [ c ] ])));
      test "a join names the state of its first source and writes nothing"
        (fun () ->
          let ones = called incr (param 0) (param 1) in
          let twos = called incr (param 2) ones in
          let join = Ops.after (param 0) [ twos ] in
          let copy = called (kernel Fun.id) (param 3) join in
          equal write
            (into 0 (ints [ 1; 2; 3; 4 ])
            @ into 2 (ints [ 2; 3; 4; 5 ])
            @ into 3 (ints [ 10; 10; 10; 10 ]))
            (Kernel_graphs.writes
               ~buffers:[ (0, tens); (1, iota) ]
               (Ops.sink [ ones; twos; join; copy ])));
      test "a scalar parameter is the value a variable passed bound has"
        (fun () ->
          let five =
            Shape.bind (Ops.variable ~dtype:Int32 "n" (z 0) (z 9)) (z 5)
          in
          equal write
            (into 0 (ints [ 5; 6; 7; 8 ]))
            (Kernel_graphs.writes
               ~buffers:[ (1, iota) ]
               (Ops.sink
                  [
                    Ops.after (param 0)
                      [ Ops.call shifted [ param 0; param 1; five ] ];
                  ])));
      test "a variable passed unbound takes the value vars gives its name"
        (fun () ->
          let v = Ops.variable ~dtype:Int32 "n" (z 0) (z 9) in
          equal write
            (into 0 (ints [ 5; 6; 7; 8 ]))
            (Kernel_graphs.writes
               ~vars:[ ("n", z 5) ]
               ~buffers:[ (1, iota) ]
               (Ops.sink
                  [
                    Ops.after (param 0)
                      [ Ops.call shifted [ param 0; param 1; v ] ];
                  ])));
      test "a scalar parameter no call passes takes the value params gives"
        (fun () ->
          equal write
            (into 0 (ints [ 5; 6; 7; 8 ]))
            (Kernel_graphs.writes
               ~params:[ (2, z 5) ]
               ~buffers:[ (1, iota) ]
               (Ops.sink [ called shifted (param 0) (param 1) ])));
      test "call-local storage is scratch, apart from the others' numbering"
        (fun () ->
          let scratch = Shape.alloc ~slot:0 ~device:cpu [ Int 4 ] Int32 in
          equal write
            (into 0 (ints [ 2; 3; 4; 5 ]))
            (Kernel_graphs.writes
               ~buffers:[ (1, iota) ]
               (Ops.sink
                  [ called incr (param 0) (called incr scratch (param 1)) ])));
      cases "memory not given holds NaN, the type's least integer, or false"
        ~id:(fun (dt, _) -> Format.asprintf "%a" Dtype.pp dt)
        [
          (Dtype.Float32, `Float Float.nan);
          (Int32, z (Int32.to_int Int32.min_int));
          (Uint8, z 0);
          (Bool, `Bool false);
        ]
        (fun (dt, least) ->
          equal write
            (into 0 [ least; least; least; least ])
            (Kernel_graphs.writes
               (Ops.sink
                  [ called (kernel ~dt Fun.id) (param ~dt 0) (param ~dt 1) ])));
      test "an element holds the last value written, whatever the sink's order"
        (fun () ->
          let once = called incr (param 0) (param 0) in
          let twice = called incr once once in
          let last = into 0 (ints [ 2; 3; 4; 5 ]) in
          let writes sink =
            Kernel_graphs.writes ~buffers:[ (0, iota) ] (Ops.sink sink)
          in
          equal write last (writes [ once; twice ]);
          equal write last (writes [ twice; once ]));
      cases "an argument that is neither storage nor a scalar of a value raises"
        ~id:fst
        [
          ( "an unbound variable",
            fun () ->
              let unbound = Ops.variable ~dtype:Int32 "n" (z 0) (z 9) in
              Ops.call shifted [ param 0; param 1; unbound ] );
          ( "a run of storage's elements",
            fun () ->
              let run =
                Shape.shrink (param ~size:8 0) [ Some (Int 2, Int 6) ]
              in
              Ops.call incr [ run; param 1 ] );
        ]
        (fun (_, call) ->
          raises_match (Exn.invalid_arg ?sub:None) (fun () ->
              Kernel_graphs.writes
                (Ops.sink [ Ops.after (param ~size:8 0) [ call () ] ])));
      test
        "a state listing what is neither a call of a kernel nor a state raises"
        (fun () ->
          let r = Ops.range ~axis_type:Loop (Int 2) [ 1 ] in
          let loop = Ops.end_ (Ops.call incr [ param 0; param 1 ]) [ r ] in
          raises_match (Exn.invalid_arg ?sub:None) (fun () ->
              Kernel_graphs.writes
                ~buffers:[ (1, iota) ]
                (Ops.sink [ Ops.after (param 0) [ loop ] ])));
      test
        "two states of one storage, neither after the other, join their writes"
        (fun () ->
          let plus_one first =
            kernel ~first ~count:2 (fun x -> Ops.O.(x + int32 1))
          in
          equal write
            (into 0 (ints [ 1; 2; 3; 4 ]))
            (Kernel_graphs.writes
               ~buffers:[ (1, iota) ]
               (Ops.sink
                  [
                    called (plus_one 0) (param 0) (param 1);
                    called (plus_one 2) (param 0) (param 1);
                  ])));
      test
        "two states of one storage, neither after the other, that differ at an \
         element raise" (fun () ->
          let zero = kernel (fun _ -> int32 0) in
          raises_match (Exn.invalid_arg ?sub:None) (fun () ->
              Kernel_graphs.writes
                ~buffers:[ (1, iota) ]
                (Ops.sink
                   [
                     called incr (param 0) (param 1);
                     called zero (param 0) (param 1);
                   ])));
      test "a kernel reading outside its storage raises as Interpreter.writes"
        (fun () ->
          raises_match (Exn.invalid_arg ?sub:None) (fun () ->
              Kernel_graphs.writes
                ~buffers:[ (1, Array.init 2 z) ]
                (Ops.sink [ called incr (param 0) (param ~size:2 1) ])));
    ]

let linear calls = Ops.v ~src:calls Linear

let linear_writes =
  group "linear_writes"
    [
      test "a call reads what the calls before it wrote" (fun () ->
          equal write
            (into 0 (ints [ 12; 12; 12; 12 ]) @ into 1 (ints [ 11; 11; 11; 11 ]))
            (Kernel_graphs.linear_writes
               ~buffers:[ (2, tens) ]
               (linear
                  [
                    Ops.call incr [ param 1; param 2 ];
                    Ops.call incr [ param 0; param 1 ];
                  ])));
      test "a call reads nothing the calls after it write" (fun () ->
          equal write
            (into 0 (ints [ 1; 2; 3; 4 ]) @ into 1 (ints [ 11; 11; 11; 11 ]))
            (Kernel_graphs.linear_writes
               ~buffers:[ (1, iota); (2, tens) ]
               (linear
                  [
                    Ops.call incr [ param 0; param 1 ];
                    Ops.call incr [ param 1; param 2 ];
                  ])));
      test "a run of storage's elements stands for those elements" (fun () ->
          let run = Shape.shrink (param ~size:8 0) [ Some (Int 2, Int 6) ] in
          equal write
            (List.init 4 (fun i -> (0, i + 2, z (i + 1))))
            (Kernel_graphs.linear_writes
               ~buffers:[ (1, iota) ]
               (linear [ Ops.call incr [ run; param 1 ] ])));
      test "the members of a launch read memory as the launch found it"
        (fun () ->
          let launch =
            Shape.launch
              [
                Ops.call incr [ param 0; param 2 ];
                Ops.call incr [ param 1; param 0 ];
              ]
          in
          equal write
            (into 0 (ints [ 11; 11; 11; 11 ]) @ into 1 (ints [ 1; 2; 3; 4 ]))
            (Kernel_graphs.linear_writes
               ~buffers:[ (0, iota); (2, tens) ]
               (linear [ launch ])));
      (* Trips 0, 1 and 2 add 3; trips from 1 would add 6. *)
      test "an end of calls runs them for each value of its ranges, from 0"
        (fun () ->
          let r = Ops.range ~axis_type:Loop (Int 3) [ 1 ] in
          equal write
            (into 0 (ints [ 3; 4; 5; 6 ]))
            (Kernel_graphs.linear_writes
               ~buffers:[ (0, iota) ]
               (linear
                  [ Ops.end_ (Ops.call shifted [ param 0; param 0; r ]) [ r ] ])));
      test "a scalar parameter no call passes takes the value params gives"
        (fun () ->
          equal write
            (into 0 (ints [ 5; 6; 7; 8 ]))
            (Kernel_graphs.linear_writes
               ~params:[ (2, z 5) ]
               ~buffers:[ (1, iota) ]
               (linear [ Ops.call shifted [ param 0; param 1 ] ])));
      test "a call of what is not a kernel raises" (fun () ->
          raises_match (Exn.invalid_arg ?sub:None) (fun () ->
              Kernel_graphs.linear_writes
                (linear
                   [
                     Ops.v
                       ~src:[ Ops.store (param 0) (param 1); param 0; param 1 ]
                       Call;
                   ])));
    ]

let () = exit (run "Kernel_graphs" [ writes; linear_writes ])
