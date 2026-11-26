(* Shapes, resolving, symbolic integers and evaluation. *)

open Windtrap
open Tolk
open Common

let var = Simplification.var
let str pp x = Format.asprintf "%a" pp x

(* An outcome: a value, or the exception that computing it raised. *)
let outcome f = match f () with v -> Ok v | exception e -> Error e

(* Exceptions compared by constructor: their messages are not promised. *)
let exn =
  let kind = function
    | Invalid_argument _ -> "Invalid_argument"
    | e -> Printexc.to_string e
  in
  Testable.make
    ~pp:(fun ppf e -> Format.pp_print_string ppf (Printexc.to_string e))
    ~equal:(fun e e' -> kind e = kind e')

(* Shapes *)

let t_var () = var "t" 1 10

let none_of u =
  equal ~msg:(str Op.pp (Ops.op u)) (option shape) None (Shape.shape_opt u)

let shapes =
  group "shapes"
    [
      test "shape_opt is None for effects and program structure" (fun () ->
          List.iter none_of
            [
              Ops.sink [];
              Ops.call (Ops.custom_function ~fmt:true "nop" []) [];
              Ops.v ~arg:(Code { code = "nop"; dtype = Void }) Op.Ins;
              Ops.call (Ops.sink []) [];
              Ops.custom_function "f" [];
              Ops.v Op.Noop;
              Ops.v ~src:[ Ops.sink [] ] ~arg:(Dtype Int32) Op.Bitcast;
            ]);
      test "shape rejects a node that has none" (fun () ->
          rejects (fun () -> Shape.shape (Ops.sink [])));
      test "an elementwise operation broadcasts its sources' shapes" (fun () ->
          let row = Shape.expand (Ops.float 1.) (ints [ 4; 8 ]) in
          let at shape = Shape.expand (Ops.float 2.) (ints shape) in
          List.iter
            (fun u -> equal shape (ints [ 4; 8 ]) (Shape.shape u))
            Ops.O.
              [
                row + at [ 4; 1 ];
                row + at [ 1; 8 ];
                row * at [ 8 ];
                row * float 2.;
              ]);
      test "an elementwise operation keeps a symbolic shape" (fun () ->
          let t = t_var () in
          let sym = Shape.expand (Ops.float 1.) [ Int 1; Int 1; Sym t ] in
          equal shape [ Int 1; Int 1; Sym t ] (Shape.shape Ops.O.(sym + sym)));
      test "an elementwise operation rejects sources that do not broadcast"
        (fun () ->
          let a = Shape.param ~shape:(ints [ 2 ]) 812 Float32
          and b = Shape.param ~shape:(ints [ 3 ]) 813 Float32 in
          rejects (fun () -> Shape.shape Ops.O.(a + b)));
      test
        "an elementwise operation rejects sources of other shapes under \
         disallow_broadcast" (fun () ->
          let a = Shape.param ~shape:(ints [ 2; 3 ]) 0 Float32
          and b = Shape.param ~shape:(ints [ 3 ]) 1 Float32 in
          Setting.context
            [ B (Setting.disallow_broadcast, true) ]
            (fun () -> rejects (fun () -> Shape.shape Ops.O.(a + b)));
          equal shape (ints [ 2; 3 ]) (Shape.shape Ops.O.(a + b)));
      test "a call has its body's shape" (fun () ->
          let f = Ops.custom_function ~dtype:Float64 "sqrt" [] in
          equal shape [] (Shape.shape f);
          equal shape [] (Shape.shape (Ops.call f [ Ops.float 16. ]));
          let a = Shape.param ~shape:(ints [ 4 ]) 0 Float32
          and b = Shape.param ~shape:(ints [ 4 ]) 1 Float32 in
          equal shape (ints [ 4 ])
            (Shape.shape (Ops.call (Ops.store a b) [ a; b ])));
      test "an instruction is a scalar" (fun () ->
          equal shape []
            (Shape.shape
               (Ops.v
                  ~arg:(Code { code = "mov"; dtype = Int32 })
                  ~src:[ Ops.int 1 ]
                  Op.Ins)));
      test "code has the shape its function states" (fun () ->
          let pair = Shape.consts [ i 1; i 2 ] in
          equal shape (ints [ 2 ])
            (Shape.shape
               (Ops.call ~inline:true
                  (Ops.custom_function ~dtype:Int32 ~shape:(ints [ 2 ])
                     ~fmt:true "x" [])
                  [ pair ])));
      test "a stack prepends its number of sources" (fun () ->
          let vec = Shape.consts [ i 5; i 6; i 7 ] in
          equal shape (ints [ 3 ]) (Shape.shape vec);
          equal shape
            (ints [ 2; 3 ])
            (Shape.shape (Shape.stack [ vec; Shape.consts [ i 8; i 9; i 10 ] ]));
          equal shape [] (Shape.shape (Ops.v Op.Stack));
          equal dtype Void (Ops.dtype (Ops.v Op.Stack)));
      test "an expand prepends its sizes" (fun () ->
          let base = Shape.param ~shape:(ints [ 4; 5 ]) 0 Int32 in
          equal shape
            (ints [ 3; 4; 5 ])
            (Shape.shape (Shape.mop base (Expand (ints [ 3 ])))));
      test "a bitcast scales its last axis by the ratio of item sizes"
        (fun () ->
          let at n dt = Shape.param ~shape:(ints [ n ]) 1 dt in
          equal shape (ints [ 1 ]) (Shape.shape (Ops.bitcast (at 4 Int8) Int32));
          equal shape (ints [ 8 ]) (Shape.shape (Ops.bitcast (at 2 Int32) Int8));
          equal shape (ints [ 4 ])
            (Shape.shape (Ops.bitcast (at 4 Int32) Float32));
          equal shape []
            (Shape.shape (Ops.bitcast (Shape.param 0 Int32) Float32)));
      test "a bitcast rejects a last axis the ratio does not divide" (fun () ->
          rejects (fun () ->
              Shape.shape
                (Ops.bitcast (Shape.param ~shape:(ints [ 3 ]) 0 Int8) Int32)));
      test "binary code is a vector of its bytes" (fun () ->
          let bin = Ops.v ~arg:(Bytes "code") Op.Binary in
          equal dtype Uint8 (Ops.dtype bin);
          equal shape (ints [ 4 ]) (Shape.shape bin));
      test
        "an index has its indices' shape, then its source's axes past the \
         indices" (fun () ->
          let p = Shape.param ~shape:(ints [ 4; 5; 6 ]) 0 Float32 in
          equal shape (ints [ 5; 6 ]) (Shape.shape (Ops.index p [ Ops.int 1 ]));
          equal shape
            (ints [ 2; 6 ])
            (Shape.shape (Ops.index p [ Ops.int 1; Shape.consts [ i 0; i 1 ] ]));
          equal shape []
            (Shape.shape (Ops.index (Shape.param 0 Float32) [ Ops.int 0 ]));
          equal shape []
            (Shape.shape
               (Ops.index
                  (Shape.param ~shape:(ints [ 4 ]) 0 Float32)
                  [ Ops.int 0; Ops.int 1 ])));
      test "a stage puts its ranges' sizes in front" (fun () ->
          let r = Ops.range ~axis_type:Weak (Int 3) [ 0 ] in
          equal shape
            (ints [ 3; 2 ])
            (Shape.shape (Ops.bufferize (Shape.consts [ f 1.; f 2. ]) [ r ])));
      test "a matrix multiply-accumulate has its accumulator's shape" (fun () ->
          let a = Shape.param ~shape:(ints [ 8 ]) 0 Float16
          and b = Shape.param ~shape:(ints [ 8 ]) 1 Float16 in
          equal shape (ints [ 4 ])
            (Shape.shape
               (Ops.wmma a b
                  ~acc:(Shape.param ~shape:(ints [ 4 ]) 2 Float32)
                  ~dims:(8, 16, 16) ~threads:32)));
      test "a reduction drops its first num_axes axes" (fun () ->
          let p = Shape.param ~shape:(ints [ 2; 3; 4 ]) 0 Float32 in
          let reduce n =
            Ops.v ~src:[ p ]
              ~arg:(Reduce { op = Op.Add; num_axes = n })
              Op.Reduce
          in
          equal shape (ints [ 3; 4 ]) (Shape.shape (reduce 1));
          rejects (fun () -> Shape.shape (reduce 4)));
      test "a node passing its source through keeps its shape" (fun () ->
          let p = Shape.param ~device:"CPU" ~shape:(ints [ 2; 3 ]) 0 Float32 in
          List.iter
            (fun u ->
              equal
                ~msg:(str Op.pp (Ops.op u))
                shape
                (ints [ 2; 3 ])
                (Shape.shape u))
            [
              Ops.after p [ Ops.sink [] ];
              Ops.copy_to_device p "CUDA";
              Ops.v ~src:[ p ] Op.Detach;
              Ops.v ~src:[ p ] Op.Contiguous_backward;
              Ops.v ~src:[ p ] Op.Noop;
            ]);
      test "a reshape of a no-op has its argument's shape" (fun () ->
          equal shape
            (ints [ 2; 3 ])
            (Shape.shape
               (Ops.v
                  ~src:
                    [
                      Ops.v Op.Noop;
                      Ops.v ~src:[ Ops.int 2; Ops.int 3 ] Op.Stack;
                    ]
                  Op.Reshape)));
      test "a movement of a node without a shape is rejected" (fun () ->
          rejects (fun () ->
              Shape.shape (Ops.v ~src:[ Ops.sink []; Ops.int 1 ] Op.Expand)));
      test "a movement checks its argument against its source's shape"
        (fun () ->
          let p = Shape.param ~shape:(ints [ 2; 3 ]) 0 Float32 in
          List.iter
            (fun m -> rejects (fun () -> Shape.shape (Shape.mop p m)))
            [
              Pad [ (Int 0, Int 2) ];
              Shrink [ (Int 0, Int 2) ];
              Pad [ (Int (-1), Int 3); (Int 0, Int 3) ];
              Shrink [ (Int 0, Int (-1)); (Int 0, Int 3) ];
              Permute [ 0; 0 ];
              Permute [ 0 ];
              Reshape (ints [ 5 ]);
              Reshape (ints [ -6; -1 ]);
              Shrink [ (Int 1, Int 2); (Int 0, Int 3) ];
              Pad [ (Int 3, Int 2); (Int 0, Int 3) ];
            ]);
      test "a shrink past the largest int is rejected, not wrapped" (fun () ->
          let p = Shape.param ~shape:(ints [ 2; 3 ]) 0 Float32 in
          rejects (fun () ->
              Shape.shape
                (Shape.mop p (Shrink [ (Int max_int, Int 2); (Int 0, Int 3) ]))));
      test "a symbolic reshape is accepted unless its sizes provably differ"
        (fun () ->
          let n = var "n" 1 8 in
          let p = Shape.param ~shape:[ Sym n ] 0 Float32 in
          equal shape [ Sym n; Int 1 ]
            (Shape.shape (Shape.mop p (Reshape [ Sym n; Int 1 ])));
          equal shape (ints [ 4 ])
            (Shape.shape (Shape.mop p (Reshape (ints [ 4 ]))));
          let m = Shape.param ~shape:[ Sym (var "m" 10 20) ] 1 Float32 in
          rejects (fun () -> Shape.shape (Shape.mop m (Reshape (ints [ 4 ])))));
      test "the shape of a deep graph needs no deep recursion" (fun () ->
          let rec deepen n u =
            if n = 0 then u else deepen (n - 1) Ops.O.(u + u)
          in
          let p = Shape.param ~shape:(ints [ 2; 3 ]) 0 Float32 in
          equal shape (ints [ 2; 3 ]) (Shape.shape (deepen 10_000 p)));
      test "ndim is the length of the shape" (fun () ->
          equal (list int) [ 2; 0 ]
            (List.map Shape.ndim
               [
                 Shape.param ~shape:(ints [ 3; 8 ]) 0 Float32;
                 Shape.param 0 Float32;
               ]));
      test "numel is the product of the shape" (fun () ->
          let n = var "n" 1 8 in
          equal sint (Int 12)
            (Shape.numel (Shape.param ~shape:(ints [ 3; 4 ]) 1 Float32));
          equal sint (Int 1) (Shape.numel (Shape.param 2 Float32));
          equal sint
            (Sym Ops.O.(int 3 * n))
            (Shape.numel (Shape.param ~shape:[ Int 3; Sym n ] 1 Float32)));
      test "max_shape takes a symbolic size's greatest value" (fun () ->
          let n = var "n" 1 8 in
          let p = Shape.param ~shape:[ Int 3; Sym n ] 0 Float32 in
          equal (list int) [ 3; 8 ] (Shape.max_shape p));
      test "max_numel is the product of max_shape" (fun () ->
          let n = var "n" 1 8 in
          equal int 24
            (Shape.max_numel (Shape.param ~shape:[ Int 3; Sym n ] 0 Float32));
          equal int 0
            (Shape.max_numel
               (Shape.placeholder ~slot:0 [ 1 lsl 32; 1 lsl 32; 0 ] Float32)));
      test "broadcast_axes is the axes broadcasting adds or expands" (fun () ->
          let axes src out = Shape.broadcast_axes (ints src) (ints out) in
          equal (list int) [] (axes [ 4; 8 ] [ 4; 8 ]);
          equal (list int) [ 0 ] (axes [ 8 ] [ 4; 8 ]);
          equal (list int) [ 0; 1 ] (axes [] [ 4; 8 ]);
          equal (list int) [ 0; 2 ] (axes [ 3; 1 ] [ 4; 3; 8 ]);
          equal (list int) [] (axes [ 1; 8 ] [ 1; 8 ]));
      test "broadcast_axes compares symbolic sizes" (fun () ->
          let t = t_var () in
          equal (list int) []
            (Shape.broadcast_axes [ Sym t; Int 8 ] [ Sym t; Int 8 ]);
          equal (list int) [ 0 ]
            (Shape.broadcast_axes (ints [ 1; 8 ]) [ Sym t; Int 8 ]));
      test "broadcast_axes rejects a source with more axes than the output"
        (fun () ->
          rejects (fun () -> Shape.broadcast_axes (ints [ 4; 8 ]) (ints [ 8 ])));
    ]

(* Resolving *)

let a = var "a" 0 8
let b = var "b" 0 8

let resolving =
  let u = var "i" 1 10 in
  group "resolve"
    [
      test "resolve is the value of a condition with one possible value"
        (fun () ->
          let decided =
            Ops.O.[ u < int 11; u < int (-1); a < int 9; a * int 2 < int 0 ]
          in
          List.iter
            (fun default ->
              equal (list bool)
                [ true; false; true; false ]
                (List.map (Shape.resolve ~default) decided))
            [ true; false ]);
      test "resolve takes the default, true unless given, when undecided"
        (fun () ->
          equal (list bool) [ true; false; true ]
            [
              Shape.resolve ~default:true Ops.O.(u < int 4);
              Shape.resolve ~default:false Ops.O.(u < int 4);
              Shape.resolve Ops.O.(u < int 4);
            ]);
      test "resolve rejects a node that is not boolean" (fun () ->
          rejects (fun () -> Shape.resolve (Ops.int 3)));
      prop "simplify is symbolic's rewrite"
        ~examples:[ Ops.O.((a * int 4) + b < int 16 lor (a + a < int 3)) ]
        (Gen.with_pp (Testable.pp uop)
           (Gen.map Expressions.node Expressions.scenario))
        (fun e -> equal uop (Simplification.symbolic e) (Shape.simplify e));
      test
        "simplify leaves a constant, and a sink of constants and stacks of \
         constants, as they are" (fun () ->
          equal uop (Ops.int 3) (Shape.simplify (Ops.int 3));
          let s =
            Ops.sink
              [
                Ops.int 1;
                Ops.float 2.;
                Ops.v Op.Stack;
                Ops.v ~src:[ Ops.int 1; Ops.int 2 ] Op.Stack;
              ]
          in
          equal uop s (Shape.simplify s));
      test
        "simplify folds log2 of -1 to NaN and the reciprocal of 0 to infinity"
        (fun () ->
          equal Simplification.same_float (`Float Float.nan)
            (Ops.value (Shape.simplify (Ops.log2 (Ops.float (-1.)))));
          equal uop (Ops.float Float.infinity)
            (Shape.simplify (Ops.reciprocal (Ops.float 0.))));
      test "ssimplify is an integer constant as an Int" (fun () ->
          equal shape [ Int 3; Int 3; Int 6 ]
            (List.map Shape.ssimplify
               Ops.O.[ int 3; Ops.int ~dtype:Int32 3; int 2 * int 3 ]));
      test "ssimplify is anything else as a Sym of its simplification"
        (fun () ->
          equal sint
            (Sym Ops.O.(a + int 7))
            (Shape.ssimplify Ops.O.(a + int 3 + int 4)));
      test "smax and smin of integers are integers" (fun () ->
          equal shape [ Int 5; Int 2; Int 7 ]
            [
              Shape.smax [ Int 2; Int 5 ];
              Shape.smin [ Int 5; Int 2 ];
              Shape.smax [ Int 7 ];
            ]);
      test "smax and smin of a symbolic size bound it as max and min do"
        (fun () ->
          let v = var "v" 3 10 in
          let bounds_of = function
            | Ops.Sym u -> bounds u
            | Int n -> int_bounds n n
          in
          equal (pair value value) (int_bounds 3 10)
            (bounds_of (Shape.smax [ Int 2; Sym v ]));
          equal (pair value value) (int_bounds 5 10)
            (bounds_of (Shape.smax [ Sym v; Int 5 ]));
          equal (pair value value) (int_bounds 2 2)
            (bounds_of (Shape.smin [ Int 2; Sym v ]));
          equal (pair value value) (int_bounds 3 10)
            (bounds_of (Shape.smin [ Sym v; Int 20 ])));
      test "smax and smin reject the empty list" (fun () ->
          rejects (fun () -> Shape.smax []);
          rejects (fun () -> Shape.smin []));
      test "to_z, to_float and to_bool read a literal" (fun () ->
          equal z (Bigint.of_int 5) (Shape.to_z (Ops.int 5));
          equal float_exact 1.5 (Shape.to_float (Ops.float 1.5));
          equal bool true (Shape.to_bool (Ops.bool true)));
      test "to_z reads a typed constant and an integer sum of constants"
        (fun () ->
          equal (list z)
            (List.map Bigint.of_int [ 4; 11; 2 ])
            (List.map Shape.to_z
               Ops.O.
                 [
                   Ops.int ~dtype:Int32 4;
                   Ops.int ~dtype:Int32 4 + int 7;
                   int 8 // Ops.int ~dtype:Int32 4;
                 ]));
      test "to_bool decides a comparison of constants" (fun () ->
          equal (list bool)
            [ true; true; true; false; false ]
            (List.map Shape.to_bool
               Ops.O.
                 [
                   int 4 < int 7;
                   int 4 <= int 4;
                   int 4 <> int 7;
                   int 4 <> int 4;
                   int 4 > int 7;
                 ]));
      test "to_bool decides a comparison the bounds decide" (fun () ->
          let x = var "x" 1 10 and y = var "y" 5 10 in
          equal (list bool)
            [ true; true; false; false; true; false ]
            (List.map Shape.to_bool
               Ops.O.
                 [
                   u < int 20;
                   u // int 2 < int 20;
                   u < int 1;
                   u > int 11;
                   Ops.maximum x y < int 20;
                   Ops.maximum x y < int 3;
                 ]));
      test
        "to_bool decides a disjunction with true and a conjunction with false"
        (fun () ->
          equal (list bool) [ true; false ]
            (List.map Shape.to_bool
               Ops.O.[ flag "b" lor bool true; flag "b" land bool false ]));
      test "to_bool rejects a condition with two possible values" (fun () ->
          List.iter
            (fun c -> rejects (fun () -> Shape.to_bool c))
            Ops.O.
              [
                flag "b" lor bool false;
                flag "b" land bool true;
                u < u + int 1;
                (u > int 4) lor (u < int 6);
                u < int 5;
              ]);
      test "to_bool, to_z and to_float reject another type" (fun () ->
          rejects (fun () -> Shape.to_bool (Ops.int 1));
          rejects (fun () -> Shape.to_z (Ops.bool true));
          rejects (fun () -> Shape.to_z (Ops.float 1.));
          rejects (fun () -> Shape.to_float (Ops.int 1)));
    ]

(* Sint *)

(* Integers at the edges of int arithmetic: products of two of them leave ints,
   sums of two cross max_int, min_int's negation does not fit. *)
let edges =
  [
    0;
    1;
    -1;
    2;
    -2;
    3;
    -7;
    1 lsl 31;
    -(1 lsl 31);
    max_int;
    max_int - 1;
    min_int;
    min_int + 1;
    max_int / 2;
    min_int / 2;
  ]

(* Each operation, and its exact value on Bigint: the quotient rounded down, and
   the remainder with the divisor's sign. *)
let operations =
  [
    ("+", Shape.Sint.( + ), Bigint.add);
    ("-", Shape.Sint.( - ), Bigint.sub);
    ("*", Shape.Sint.( * ), Bigint.mul);
    ("//", Shape.Sint.( // ), Bigint.fdiv);
    ("%", Shape.Sint.( % ), fun x y -> Bigint.(sub x (mul (fdiv x y) y)));
  ]

let pp_operation ppf (name, _, _) = Format.pp_print_string ppf name
let past_an_int = Invalid_argument "past an int"

(* The exact value of [x op y] as an Int, or what it raises: Division_by_zero
   for a divisor 0, Invalid_argument past an int. *)
let exactly exact x y =
  match exact (Bigint.of_int x) (Bigint.of_int y) with
  | z when Bigint.fits_int z -> Ok (Ops.Int (Bigint.to_int z))
  | _ -> Error past_an_int
  | exception Division_by_zero -> Error Division_by_zero

let gen_arithmetic =
  let some = Gen.of_list ~pp:Format.pp_print_int edges in
  Gen.triple (Gen.of_list ~pp:pp_operation operations) some some

let arithmetic_examples =
  let op name = List.find (fun (n, _, _) -> n = name) operations in
  [
    (op "//", 7, 0);
    (op "%", 7, 0);
    (op "*", 1 lsl 31, 1 lsl 31);
    (op "+", max_int, 1);
    (op "-", min_int, 1);
    (op "//", min_int, -1);
    (op "//", -7, 2);
    (op "%", 7, -2);
  ]

let cond =
  Testable.make
    ~pp:(fun ppf -> function
      | Shape.Sint.Known b -> Format.fprintf ppf "Known %b" b
      | Cond u -> Format.fprintf ppf "Cond %a" Ops.pp u)
    ~equal:(fun c c' ->
      match (c, c') with
      | Shape.Sint.Known b, Shape.Sint.Known b' -> b = b'
      | Cond u, Cond u' -> Ops.equal u u'
      | _ -> false)

let sint_module =
  let n = var "n" 1 8 in
  group "Sint"
    [
      prop
        "arithmetic on integers is exact: a quotient rounds down, a remainder \
         has the divisor's sign, a result past an int and a divisor 0 raise"
        ~examples:arithmetic_examples gen_arithmetic
        (fun ((_, op, exact), x, y) ->
          let expected = exactly exact x y in
          cover "a result past an int" (expected = Error past_an_int);
          equal (result sint exn) expected
            (outcome (fun () -> op (Int x) (Int y))));
      test "arithmetic with a node builds the node's operation" (fun () ->
          equal shape
            [
              Sym Ops.O.(n + int 1);
              Sym Ops.O.(int 1 + n);
              Sym Ops.O.(n - int 1);
              Sym Ops.O.(int 2 * n);
              Sym Ops.O.(n // int 2);
              Sym Ops.O.(n % int 2);
            ]
            Shape.Sint.
              [
                Sym n + Int 1;
                Int 1 + Sym n;
                Sym n - Int 1;
                Int 2 * Sym n;
                Sym n // Int 2;
                Sym n % Int 2;
              ]);
      test "a node divided by 0 is the division node" (fun () ->
          equal shape
            [ Sym Ops.O.(n // int 0); Sym Ops.O.(n % int 0) ]
            Shape.Sint.[ Sym n // Int 0; Sym n % Int 0 ]);
      prop
        "prod of integers is their exact product, 1 if empty, and raises past \
         an int"
        ~examples:[ [ max_int; 2; 0 ]; [ max_int; 2 ]; [] ]
        Gen.(list ~size:(int_range 0 4) (of_list ~pp:Format.pp_print_int edges))
        (fun l ->
          let z =
            List.fold_left (fun z n -> Bigint.(mul z (of_int n))) Bigint.one l
          in
          let expected =
            if Bigint.fits_int z then Ok (Ops.Int (Bigint.to_int z))
            else Error past_an_int
          in
          equal (result sint exn) expected
            (outcome (fun () ->
                 Shape.Sint.prod (List.map (fun n -> Ops.Int n) l))));
      test "prod of a node and integers is the node times their product"
        (fun () ->
          let a = var "a" 1 5 in
          equal sint (Sym Ops.O.(int 2 * n)) (Shape.Sint.prod [ Int 2; Sym n ]);
          (* in either order, a node that simplifies to a * 12, which renders as
             (a*12) *)
          List.iter
            (fun l ->
              match Shape.Sint.prod l with
              | Sym u -> equal uop Ops.O.(a * int 12) (Shape.simplify u)
              | Int k -> failf "prod is the integer %d" k)
            [ [ Sym a; Int 3; Int 4 ]; [ Int 3; Int 4; Sym a ] ]);
      prop "a comparison of integers is known"
        ~examples:[ (3, 3) ]
        Gen.(pair (int_range (-3) 3) (int_range (-3) 3))
        (fun (x, y) ->
          equal (list cond)
            (List.map
               (fun b -> Shape.Sint.Known b)
               [ x < y; x <= y; x > y; x >= y; x <> y ])
            Shape.Sint.
              [
                Int x < Int y;
                Int x <= Int y;
                Int x > Int y;
                Int x >= Int y;
                Int x <> Int y;
              ]);
      test "a comparison with a node is the condition node" (fun () ->
          equal (list cond)
            Shape.Sint.
              [
                Cond Ops.O.(n < int 3);
                Cond Ops.O.(n <= int 3);
                Cond Ops.O.(n > int 3);
                Cond Ops.O.(n >= int 3);
                Cond Ops.O.(n <> int 3);
                Cond Ops.O.(int 3 > n);
              ]
            Shape.Sint.
              [
                Sym n < Int 3;
                Sym n <= Int 3;
                Sym n > Int 3;
                Sym n >= Int 3;
                Sym n <> Int 3;
                Int 3 > Sym n;
              ]);
      test "resolve is a known condition's value, whatever the default"
        (fun () ->
          equal (list bool)
            [ true; false; true; false ]
            [
              Shape.Sint.resolve ~default:false (Known true);
              Shape.Sint.resolve ~default:true (Known false);
              Shape.Sint.resolve (Known true);
              Shape.Sint.resolve (Known false);
            ]);
      test "resolve is Shape.resolve's for a node" (fun () ->
          equal (list bool)
            [ true; false; true; true ]
            [
              Shape.Sint.(resolve ~default:false (Sym n < Int 9));
              Shape.Sint.(resolve ~default:false (Sym n < Int 5));
              Shape.Sint.(resolve ~default:false (Sym n <> Int 0));
              Shape.Sint.(resolve (Sym n < Int 5));
            ]);
      test "truth is a known condition's value, and Shape.to_bool's for a node"
        (fun () ->
          equal (list bool) [ true; false; true ]
            Shape.Sint.
              [ truth (Known true); truth (Known false); truth (Sym n < Int 9) ];
          rejects (fun () -> Shape.Sint.(truth (Sym n < Int 5))));
      test "equal compares integers by value and nodes by identity" (fun () ->
          equal (list bool) [ true; false; true ]
            Shape.Sint.
              [
                equal (Int 3) (Int 3);
                equal (Int 3) (Sym (Ops.int 3));
                equal (Sym n) (Sym (var "n" 1 8));
              ]);
      test "pp formats an integer in decimal and a node as Ops.pp does"
        (fun () ->
          equal string "3" (str Shape.Sint.pp (Int 3));
          equal string (str Ops.pp n) (str Shape.Sint.pp (Sym n)));
    ]

(* Evaluating *)

let inference =
  group "sym_infer"
    [
      test "an integer is itself" (fun () ->
          equal int 5 (Shape.sym_infer (Int 5) []));
      test "a node takes its variables' values" (fun () ->
          let n = var "n" 0 100 in
          equal int 7
            (Shape.sym_infer (Sym Ops.O.((n * int 2) + int 1)) [ ("n", 3) ]);
          equal int 9
            (Shape.sym_infer
               (Sym Ops.O.(Shape.bind n (i 7) + int 1))
               [ ("n", 8) ]));
      test "an ordered value is its first source" (fun () ->
          let n = var "n" 0 100 in
          let loop =
            Ops.end_ (Ops.v Op.Noop) [ Ops.range ~axis_type:Weak (Int 4) [ 0 ] ]
          in
          equal int 7
            (Shape.sym_infer
               (Sym (Ops.after Ops.O.((n * int 2) + int 1) [ loop ]))
               [ ("n", 3) ]));
      (* -7 // 3 = -3 and -7 % 3 = 2 round down; truncated, -2 and -1 *)
      test "divisions round as their operations say" (fun () ->
          let n = var "n" (-10) 10 and d = var "d" (-10) 10 in
          let at nv dv e = Shape.sym_infer (Sym e) [ ("n", nv); ("d", dv) ] in
          equal (list int)
            [ -3; 2; -2; -1; -2; 1; -1 ]
            [
              at (-7) 0 Ops.O.(n // int 3);
              at (-7) 0 Ops.O.(n % int 3);
              at (-7) 0 (Ops.alu n Op.Truncdiv [ Ops.int 3 ]);
              at (-7) 0 (Ops.alu n Op.Truncmod [ Ops.int 3 ]);
              at 7 (-3) (Ops.alu n Op.Truncdiv [ d ]);
              at 7 (-3) (Ops.alu n Op.Truncmod [ d ]);
              at (-7) 3 (Ops.alu n Op.Truncmod [ d ]);
            ]);
      test "a cast converts without truncating to a width" (fun () ->
          let n = var "n" (-1000) 1000 in
          let half = Ops.O.(Ops.cast n Float32 / float 2.) in
          equal int 7
            (Shape.sym_infer
               (Sym Ops.O.(Ops.cast half Weak_int + int 10))
               [ ("n", -7) ]);
          equal int 300
            (Shape.sym_infer
               (Sym (Ops.cast (Ops.cast n Int8) Weak_int))
               [ ("n", 300) ]));
      (* 1069547520 is 0x3FC00000, the bits of the float32 1.5 *)
      test "a bit reinterpretation reads the bits" (fun () ->
          let a = var ~dtype:Int32 "a" 1 10
          and b = var ~dtype:Int32 "b" (-5) 5 in
          let c =
            Ops.variable ~dtype:Uint32 "c" (i 0)
              (`Int (Bigint.of_string "4294967295"))
          in
          let shifted =
            Ops.O.(Ops.bitcast (Ops.bitcast a Uint32 lsl int 1) Int32 + int 2)
          in
          equal int 6 (Shape.sym_infer (Sym shifted) [ ("a", 2) ]);
          equal int 0xFFFF_FFFF
            (Shape.sym_infer (Sym (Ops.bitcast b Uint32)) [ ("b", -1) ]);
          equal int (-1)
            (Shape.sym_infer (Sym (Ops.bitcast c Int32)) [ ("c", 0xFFFF_FFFF) ]);
          equal int 1069547520
            (Shape.sym_infer
               (Sym (Ops.bitcast (Ops.cast (Ops.float 1.5) Float32) Uint32))
               []));
      (* A boolean is true where its byte is not zero, and a bitcast back reads
         the byte, which no boolean value holds. *)
      test "a bitcast to a boolean of a byte neither 0 nor 1 is rejected"
        (fun () ->
          let byte name = Ops.bitcast (var ~dtype:Uint8 name 0 255) Bool in
          let picked =
            Ops.bitcast
              (Ops.where
                 (Ops.cast (var ~dtype:Int32 "c" 0 1) Bool)
                 (byte "x") (byte "y"))
              Uint8
          in
          let at x =
            Shape.sym_infer (Sym picked) [ ("c", 1); ("x", x); ("y", 0) ]
          in
          equal int 1 (at 1);
          rejects (fun () -> at 2));
      test "an expression nested 200 deep is evaluated" (fun () ->
          let a = var "a" 1 8192 and b = var "b" 0 8191 in
          let step e =
            Ops.O.((Ops.maximum (e * (b + a)) (int (-33554432)) * int (-1)) + a)
          in
          let rec nest n e = if n = 0 then e else nest (n - 1) (step e) in
          equal int 1
            (Shape.sym_infer (Sym (nest 200 a)) [ ("a", 1); ("b", 0) ]));
      test "a variable vars lacks is rejected" (fun () ->
          rejects (fun () -> Shape.sym_infer (Sym (var "n" 0 4)) []));
      test "a value past an int is rejected" (fun () ->
          let a = var "a" min_int max_int and b = var "b" min_int max_int in
          let p31 = 1 lsl 31 in
          rejects (fun () ->
              Shape.sym_infer (Sym Ops.O.(a * b)) [ ("a", p31); ("b", p31) ]));
    ]

(* sym_compile computes what exact arithmetic does, on expressions whose values
   fit an [int] and on those whose products do not fit one. *)

let small_constants = Gen.such_that (( <> ) 0) (Gen.int_range (-20) 20)

(* A symbolic integer over the variables [a] and [b] and [constants], of depth
   at most [depth]. *)
let rec expression ?(constants = small_constants) a b depth =
  let leaf =
    Gen.one_of [ Gen.constant a; Gen.constant b; Gen.map Ops.O.int constants ]
  in
  if depth = 0 then leaf
  else
    let sub = expression ~constants a b (depth - 1) in
    let node =
      let open Gen in
      let+ op = int_range 0 8 and+ x = sub and+ y = sub in
      match op with
      | 0 -> Ops.O.(x + y)
      | 1 -> Ops.O.(x - y)
      | 2 -> Ops.O.(x * y)
      | 3 -> Ops.O.(x // y)
      | 4 -> Ops.O.(x % y)
      | 5 -> Ops.alu x Op.Truncdiv [ y ]
      | 6 -> Ops.alu x Op.Truncmod [ y ]
      | 7 -> Ops.alu x Op.Max [ y ]
      | _ -> Ops.O.(-x)
    in
    Gen.frequency [ (1, leaf); (3, node) ]

(* A divisor that is 0 at the binding, where sym_infer promises no value. *)
exception Zero_divisor

let divisions = Op.[ Div; Mod; Truncdiv; Truncmod ]

(* The reference: each node of [e] computed on unbounded integers, as its
   operation says, each variable read from [env]. It raises [Zero_divisor] at a
   divisor that is 0. *)
let exact_value e env =
  let values = Ops.Tbl.create 16 in
  let eval n : Dtype.const =
    match Ops.op n with
    | Const -> Ops.value n
    | Param -> `Int (Bigint.of_int (List.assoc (Ops.expr n) env))
    (* a variable's size source, which no value reads *)
    | Stack when Ops.src n = [] -> `Bool false
    | op ->
        let operands = List.map (Ops.Tbl.find values) (Ops.src n) in
        (match (op, operands) with
        | _, [ _; `Int d ] when List.mem op divisions && Bigint.(equal d zero)
          ->
            raise Zero_divisor
        | _ -> ());
        Ops.exec_alu ~truncate_output:false op (Ops.dtype n) operands
  in
  match Ops.topovisit e eval values with
  | #Dtype.value as v -> v
  | `Invalid -> invalid_arg "Invalid is not a number"

let exact e env = Dtype.Value.to_int (exact_value e env)
let exact_z e env = Dtype.Value.to_z (exact_value e env)

(* [e] at [a] = [va] and [b] = [vb], exactly and by sym_compile. *)
let both e va vb =
  let env = [ ("a", va); ("b", vb) ] in
  let var u = List.assoc (Ops.expr u) in
  ( outcome (fun () -> exact e env),
    outcome (fun () -> Shape.sym_compile (Sym e) var env) )

let agrees ?constants ~name a b value =
  let draw =
    Gen.triple (Gen.with_pp Ops.pp (expression ?constants a b 3)) value value
  in
  prop name draw (fun (e, va, vb) ->
      let exact, compiled = both e va vb in
      assume (exact <> Error Zero_divisor);
      equal (result int exn) exact compiled)

let compiles_exactly ~bound =
  agrees
    ~name:
      (Printf.sprintf "computes what exact arithmetic does, variables within %d"
         bound)
    (var "a" (-bound) bound) (var "b" (-bound) bound)
    (Gen.int_range (-bound) bound)

(* The edges of [int] arithmetic: 2^30 and 2^31, around which the product of two
   values leaves [int]s; 2^61, whose sums cross [max_int]; and [min_int], whose
   negation and quotient by -1 do not fit. *)
let extremes =
  let edges =
    List.concat_map
      (fun x -> [ x - 1; x; x + 1; -x + 1; -x; -x - 1 ])
      [ 1 lsl 30; 1 lsl 31; 1 lsl 61 ]
  in
  [ 0; 1; -1; 2; -2; 3; max_int; max_int - 1; min_int; min_int + 1 ] @ edges

let compiles_at_the_extremes =
  let some = Gen.of_list ~pp:Format.pp_print_int extremes in
  agrees ~name:"computes what exact arithmetic does, at the int extremes"
    ~constants:(Gen.such_that (( <> ) 0) some)
    (var "a" min_int max_int) (var "b" min_int max_int) some

(* Values whose [int] arithmetic overflows on the way, and their exact values,
   computed by hand. *)
let overflowing =
  let a = var "a" min_int max_int and b = var "b" min_int max_int in
  (* ((2^30 - 1)^2 + 2^62 - 2) // 4 = 2^60 + 2^58 - 2^29 - 1: the sum crosses
     max_int, the quotient does not. *)
  let below = (1 lsl 30) - 1 and below_max = max_int - 1 in
  let quarter = (1 lsl 60) + (1 lsl 58) - (1 lsl 29) - 1 in
  let floor_third_of_min_int = (min_int / 3) - 1 in
  let p31 = 1 lsl 31 and p60 = 1 lsl 60 and p61 = 1 lsl 61 in
  let open Ops.O in
  [
    ( "a sum of a product past max_int",
      ((a * a) + b) // int 4,
      below,
      below_max,
      quarter );
    ("a product past max_int", a * b // int 4, p31, p31, p60);
    ("the negation of min_int", -a // int 2, min_int, 0, p61);
    ("min_int subtracted", b - a - int 1, min_int, 0, max_int);
    ("min_int divided, floored", a // b, min_int, 3, floor_third_of_min_int);
    ( "min_int divided toward zero",
      Ops.alu a Op.Truncdiv [ b ],
      min_int,
      3,
      Int.div min_int 3 );
    ("min_int's remainder by -1", a % b, min_int, -1, 0);
  ]

let computes_exactly (_, e, va, vb, expected) =
  let exact, compiled = both e va vb in
  equal ~msg:"exact" (result int exn) (Ok expected) exact;
  equal ~msg:"sym_compile" (result int exn) (Ok expected) compiled

(* Values that fit no [int], 2^62 each: sym_compile raises as exact arithmetic
   does, where [int] arithmetic gives min_int. *)
let unfit =
  let a = var "a" min_int max_int and b = var "b" min_int max_int in
  let p31 = 1 lsl 31 in
  let open Ops.O in
  [
    ("min_int // -1", a // b, min_int, -1);
    ("2^31 * 2^31", a * b, p31, p31);
    ("Neg min_int", Ops.alu a Op.Neg [], min_int, 0);
    ("0 Sub min_int", Ops.alu b Op.Sub [ a ], min_int, 0);
  ]

let raises_as_exact (_, e, va, vb) =
  let exact, compiled = both e va vb in
  ignore (require_error ~msg:"exact raises" ~pp:Format.pp_print_int exact);
  equal (result int exn) exact compiled

let compilation =
  group "sym_compile"
    [
      compiles_exactly ~bound:50;
      compiles_exactly ~bound:(1 lsl 31);
      compiles_at_the_extremes;
      cases "computes exactly where int arithmetic overflows"
        ~id:(fun (name, _, _, _, _) -> name)
        overflowing computes_exactly;
      cases "raises as exact arithmetic does where a value fits no int"
        ~id:(fun (name, _, _, _) -> name)
        unfit raises_as_exact;
      test "an integer is itself" (fun () ->
          equal int 5 (Shape.sym_compile (Int 5) (fun _ () -> 0) ()));
      test "a variable is what var reads" (fun () ->
          let n = var "n" 0 100 in
          equal int 7
            (Shape.sym_compile
               (Sym Ops.O.((n * int 2) + int 1))
               (fun _ x -> x)
               3));
      test "a variable var refuses raises" (fun () ->
          rejects (fun () ->
              Shape.sym_compile
                (Sym Ops.O.(var "n" 0 4 + int 1))
                (fun _ () -> invalid_arg "no n")
                ()));
    ]

(* sym_compile_z computes what exact arithmetic does, of any size. *)

let both_z e va vb =
  let env = [ ("a", va); ("b", vb) ] in
  let var u = List.assoc (Ops.expr u) in
  ( outcome (fun () -> exact_z e env),
    outcome (fun () -> Shape.sym_compile_z (Sym e) var env) )

let exact_compilation =
  let some = Gen.of_list ~pp:Format.pp_print_int extremes in
  let a = var "a" min_int max_int and b = var "b" min_int max_int in
  let draw =
    Gen.triple
      (Gen.with_pp Ops.pp
         (expression ~constants:(Gen.such_that (( <> ) 0) some) a b 3))
      some some
  in
  group "sym_compile_z"
    [
      prop "computes what exact arithmetic does, at the int extremes" draw
        (fun (e, va, vb) ->
          let exact, compiled = both_z e va vb in
          assume (exact <> Error Zero_divisor);
          equal (result z exn) exact compiled);
      cases "computes values that fit no int"
        ~id:(fun (name, _, _, _) -> name)
        unfit
        (fun (_, e, va, vb) ->
          let exact, compiled = both_z e va vb in
          is_ok ~msg:"exact" ~pp:(Testable.pp exn) exact;
          equal (result z exn) exact compiled);
    ]
