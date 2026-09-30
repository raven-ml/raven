open Windtrap
open Tolk_next
open Dtypes

let x ?(slot = 0) dt = Ops.param slot dt
let int n = `Int (Z.of_int n)
let eval ?vars ?params u = Interpreter.eval ?vars ?params u
let var name = Ops.variable ~dtype:Dtype.Int32 name (int 0) (int 10)
let rejects f = raises_match (Exn.invalid_arg ?substring:None) f

let interpreter =
  group "eval"
    [
      test "a constant is its value" (fun () ->
          equal value (`Float 1.5) (eval (Ops.float ~dtype:Dtype.Float32 1.5)));
      test "a parameter is the value given to its slot" (fun () ->
          equal value (int 7)
            (eval ~params:[ (0, int 3); (1, int 7) ] (x ~slot:1 Dtype.Int32)));
      test "a parameter without a value is refused" (fun () ->
          rejects (fun () -> eval (x Dtype.Int32)));
      test "a variable is the value bound to its name" (fun () ->
          let u = Ops.sub (var "i") (var "j") in
          equal value (int (-4)) (eval ~vars:[ ("j", int 7); ("i", int 3) ] u));
      test "a variable without a value is refused, even with its slot given"
        (fun () -> rejects (fun () -> eval ~params:[ (-1, int 3) ] (var "i")));
      test "an operation rounds its result to its type" (fun () ->
          let sum = Ops.add (x Dtype.Float16) (x ~slot:1 Dtype.Float16) in
          equal value (`Float 2048.)
            (eval ~params:[ (0, `Float 2048.); (1, `Float 1.) ] sum));
      test "a cast to an integer rounds towards zero, then wraps" (fun () ->
          equal value (int 44)
            (eval
               ~params:[ (0, `Float 300.7) ]
               (Ops.cast (x Dtype.Float32) Dtype.Uint8));
          equal value (int (-3))
            (eval
               ~params:[ (0, `Float (-3.9)) ]
               (Ops.cast (x Dtype.Float32) Dtype.Int32)));
      test "a bit reinterpretation keeps the bits" (fun () ->
          equal value (int 0x3f800000)
            (eval
               ~params:[ (0, `Float 1.) ]
               (Ops.bitcast (x Dtype.Float32) Dtype.Int32)));
      test "a selection evaluates its chosen arm" (fun () ->
          let c = x Dtype.Bool in
          let u =
            Ops.where c
              (Ops.int ~dtype:Dtype.Int32 1)
              (Ops.int ~dtype:Dtype.Int32 2)
          in
          equal value (int 2) (eval ~params:[ (0, `Bool false) ] u));
      test "a node that is not arithmetic is refused" (fun () ->
          rejects (fun () -> eval (Ops.sink [ Ops.int ~dtype:Dtype.Int32 1 ])));
    ]

let () = exit (run "Interpreter" [ interpreter ])
