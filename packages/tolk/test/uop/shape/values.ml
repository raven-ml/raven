(* Values: shapes, resolving, symbolic integers and evaluation. *)

open Windtrap
open Tolk
open Common

let str pp x = Format.asprintf "%a" pp x

(* Shapes *)

let t_var () = weak_var "t" 1 10

let shapes =
  group "shapes"
    [
      test "broadcast_axes is the axes broadcasting adds or expands" (fun () ->
          equal (list int) []
            (Shape.broadcast_axes (ints [ 4; 8 ]) (ints [ 4; 8 ]));
          equal (list int) [ 0 ]
            (Shape.broadcast_axes (ints [ 8 ]) (ints [ 4; 8 ]));
          equal (list int) [ 0; 1 ] (Shape.broadcast_axes [] (ints [ 4; 8 ]));
          equal (list int) [ 0; 2 ]
            (Shape.broadcast_axes (ints [ 3; 1 ]) (ints [ 4; 3; 8 ]));
          equal (list int) []
            (Shape.broadcast_axes (ints [ 1; 8 ]) (ints [ 1; 8 ]));
          rejects (fun () -> Shape.broadcast_axes (ints [ 4; 8 ]) (ints [ 8 ])));
      test "broadcast_axes compares symbolic sizes" (fun () ->
          let t = t_var () in
          equal (list int) []
            (Shape.broadcast_axes [ Sym t; Int 8 ] [ Sym t; Int 8 ]);
          equal (list int) [ 0 ]
            (Shape.broadcast_axes (ints [ 1; 8 ]) [ Sym t; Int 8 ]));
      test "an elementwise operation broadcasts its sources' shapes" (fun () ->
          let row = Shape.expand (Ops.float 1.) (ints [ 4; 8 ]) in
          equal shape
            (ints [ 4; 8 ])
            (Shape.shape Ops.O.(row + Shape.expand (Ops.float 2.) (ints [ 4; 1 ])));
          equal shape
            (ints [ 4; 8 ])
            (Shape.shape Ops.O.(row + Shape.expand (Ops.float 2.) (ints [ 1; 8 ])));
          equal shape
            (ints [ 4; 8 ])
            (Shape.shape Ops.O.(row * Shape.expand (Ops.float 2.) (ints [ 8 ])));
          equal shape (ints [ 4; 8 ]) (Shape.shape Ops.O.(row * float 2.)));
      test "an elementwise operation keeps a symbolic shape" (fun () ->
          let t = t_var () in
          let sym = Shape.expand (Ops.float 1.) [ Int 1; Int 1; Sym t ] in
          equal shape [ Int 1; Int 1; Sym t ] (Shape.shape Ops.O.(sym + sym)));
      test "an elementwise operation rejects sources that do not broadcast"
        (fun () ->
          let a = Shape.param ~shape:(ints [ 2 ]) 812 Float32
          and b = Shape.param ~shape:(ints [ 3 ]) 813 Float32 in
          rejects (fun () -> Shape.shape Ops.O.(a + b)));
      test "shape_opt is None for effects and program structure" (fun () ->
          List.iter
            (fun u -> is_none ~msg:(str Op.pp (Ops.op u)) (Shape.shape_opt u))
            [
              Ops.sink [];
              Ops.call (Ops.custom_function ~fmt:true "nop" []) [];
              Ops.v ~arg:(Code { code = "nop"; dtype = Void }) Op.Ins;
              Ops.call (Ops.sink []) [];
              Ops.custom_function "f" [];
            ];
          rejects (fun () -> Shape.shape (Ops.sink [])));
      test "a call has its body's shape" (fun () ->
          let f = Ops.custom_function ~dtype:Float64 "sqrt" [] in
          equal shape [] (Shape.shape f);
          equal shape [] (Shape.shape (Ops.call f [ Ops.float 16. ]));
          let a = Shape.param ~shape:(ints [ 4 ]) 0 Float32
          and b = Shape.param ~shape:(ints [ 4 ]) 1 Float32 in
          equal shape (ints [ 4 ])
            (Shape.shape (Ops.call (Ops.store a b) [ a; b ])));
      test
        "a typed instruction is a scalar, and code has the shape its function \
         states" (fun () ->
          equal shape []
            (Shape.shape
               (Ops.v
                  ~arg:(Code { code = "mov"; dtype = Int32 })
                  ~src:[ Ops.int 1 ]
                  Op.Ins));
          let pair = Shape.consts [ i 1; i 2 ] in
          equal shape (ints [ 2 ])
            (Shape.shape
               (Ops.call ~inline:true
                  (Ops.custom_function ~dtype:Int32 ~shape:(ints [ 2 ])
                     ~fmt:true "x" [])
                  [ pair ])));
      test "a stack prepends its length" (fun () ->
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
            (Shape.shape (Shape.mop base (Expand (ints [ 3 ]))));
          is_true (Shape.mop base (Expand []) == base));
      test
        "a bitcast rescales the last axis, and rejects a size that does not \
         divide" (fun () ->
          equal shape (ints [ 1 ])
            (Shape.shape
               (Ops.bitcast (Shape.param ~shape:(ints [ 4 ]) 1 Int8) Int32));
          equal shape (ints [ 8 ])
            (Shape.shape
               (Ops.bitcast (Shape.param ~shape:(ints [ 2 ]) 1 Int32) Int8));
          rejects (fun () ->
              Shape.shape
                (Ops.bitcast (Shape.param ~shape:(ints [ 3 ]) 0 Int8) Int32)));
      test "binary code is a vector of its bytes" (fun () ->
          let bin = Ops.v ~arg:(Bytes "code") Op.Binary in
          equal dtype Uint8 (Ops.dtype bin);
          equal shape (ints [ 4 ]) (Shape.shape bin));
      test
        "an index takes its indices' shapes, then the source's remaining axes"
        (fun () ->
          let p = Shape.param ~shape:(ints [ 4; 5; 6 ]) 0 Float32 in
          equal shape (ints [ 5; 6 ]) (Shape.shape (Ops.index p [ Ops.int 1 ]));
          equal shape
            (ints [ 2; 6 ])
            (Shape.shape (Ops.index p [ Ops.int 1; Shape.consts [ i 0; i 1 ] ])));
      test "an index past the source's axes takes only its indices' shapes"
        (fun () ->
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
      test "a matrix multiply-accumulate has the accumulator's shape" (fun () ->
          let a = Shape.param ~shape:(ints [ 8 ]) 0 Float16
          and b = Shape.param ~shape:(ints [ 8 ]) 1 Float16 in
          equal shape (ints [ 4 ])
            (Shape.shape
               (Ops.wmma a b
                  ~acc:(Shape.param ~shape:(ints [ 4 ]) 2 Float32)
                  ~dims:(8, 16, 16) ~threads:32)));
      test
        "a reduction drops its leading axes, and rejects more axes than its \
         source has" (fun () ->
          let p = Shape.param ~shape:(ints [ 2; 3; 4 ]) 0 Float32 in
          equal shape
            (ints [ 3; 4 ])
            (Shape.shape
               (Ops.v ~src:[ p ]
                  ~arg:(Reduce { op = Op.Add; num_axes = 1 })
                  Op.Reduce));
          rejects (fun () ->
              Shape.shape
                (Ops.v ~src:[ p ]
                   ~arg:(Reduce { op = Op.Add; num_axes = 4 })
                   Op.Reduce)));
      test "ndim, numel, max_shape and max_numel read the shape" (fun () ->
          let n = weak_var "n" 1 8 in
          let p = Shape.param ~shape:(ints [ 3; 8 ]) 0 Float32 in
          equal int 2 (Shape.ndim p);
          equal (list int) [ 3; 8 ] (Shape.max_shape p);
          equal int 24 (Shape.max_numel p);
          equal (list int) [ 3; 8 ] (Ops.to_max_shape [ Int 3; Sym n ]);
          equal sint (Int 12)
            (Shape.numel (Shape.param ~shape:(ints [ 3; 4 ]) 1 Float32));
          equal sint (Int 1) (Shape.numel (Shape.param 2 Float32)));
      test "max_shape takes a symbolic size's greatest value" (fun () ->
          let n = weak_var "n" 1 8 in
          let p = Shape.param ~shape:[ Int 3; Sym n ] 0 Float32 in
          equal (list int) [ 3; 8 ] (Shape.max_shape p);
          equal int 24 (Shape.max_numel p));
      test "placeholder rejects a size past the largest int" (fun () ->
          rejects (fun () ->
              Shape.placeholder ~slot:0 [ 1 lsl 32; 1 lsl 32 ] Float32));
      test "the shape of a deep graph needs no deep recursion" (fun () ->
          let rec deepen n u =
            if n = 0 then u else deepen (n - 1) Ops.O.(u + u)
          in
          let p = Shape.param ~shape:(ints [ 2; 3 ]) 0 Float32 in
          equal shape (ints [ 2; 3 ]) (Shape.shape (deepen 10_000 p)));
      test "max_numel is 0 when an axis is empty" (fun () ->
          equal int 0
            (Shape.max_numel
               (Shape.placeholder ~slot:0 [ 1 lsl 32; 1 lsl 32; 0 ] Float32)));
      test "disallow_broadcast rejects sources of different shapes" (fun () ->
          let a = Shape.param ~shape:(ints [ 2; 3 ]) 0 Float32
          and b = Shape.param ~shape:(ints [ 3 ]) 1 Float32 in
          Setting.context
            [ B (Setting.disallow_broadcast, true) ]
            (fun () -> rejects (fun () -> Shape.shape Ops.O.(a + b)));
          equal shape (ints [ 2; 3 ]) (Shape.shape Ops.O.(a + b)));
      test "a node passing its source through keeps its shape" (fun () ->
          let p =
            Shape.param ~device:"CPU" ~shape:(ints [ 2; 3 ]) 0 Float32
          in
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
            ];
          is_none (Shape.shape_opt (Ops.v Op.Noop)));
      test "a bitcast of a scalar or to an equal size keeps the shape"
        (fun () ->
          equal shape [] (Shape.shape (Ops.bitcast (Shape.param 0 Int32) Float32));
          equal shape (ints [ 4 ])
            (Shape.shape
               (Ops.bitcast (Shape.param ~shape:(ints [ 4 ]) 0 Int32) Float32));
          is_none
            (Shape.shape_opt
               (Ops.v ~src:[ Ops.sink [] ] ~arg:(Dtype Int32) Op.Bitcast)));
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
      test "an axis out of range is rejected" (fun () ->
          let p = Shape.param ~shape:(ints [ 2; 1 ]) 0 Float32 in
          rejects (fun () -> Shape.squeeze ~axis:2 p);
          rejects (fun () -> Shape.squeeze ~axis:(-3) p);
          rejects (fun () -> Shape.flatten ~start:5 p);
          rejects (fun () -> Shape.unsqueeze p 3);
          rejects (fun () -> Shape.transpose p 0 2));
      test "a split whose sizes miss the axis's is rejected" (fun () ->
          let p = Shape.param ~shape:(ints [ 2; 3 ]) 0 Float32 in
          rejects (fun () -> Shape.split ~axis:1 p [ 1; 1 ]);
          rejects (fun () -> Shape.split p [ 1; 2 ]));
      test "a movement of a node without a shape is rejected" (fun () ->
          rejects (fun () ->
              Shape.shape (Ops.v ~src:[ Ops.sink []; Ops.int 1 ] Op.Expand)));
      test "a shrink past the largest int is rejected, not wrapped" (fun () ->
          let p = Shape.param ~shape:(ints [ 2; 3 ]) 0 Float32 in
          rejects (fun () ->
              Shape.shape
                (Shape.mop p (Shrink [ (Int max_int, Int 2); (Int 0, Int 3) ]))));
      test "a movement checks its argument against its source's shape"
        (fun () ->
          let p = Shape.param ~shape:(ints [ 2; 3 ]) 0 Float32 in
          rejects (fun () -> Shape.shape (Shape.mop p (Pad [ (Int 0, Int 2) ])));
          rejects (fun () -> Shape.shape (Shape.mop p (Shrink [ (Int 0, Int 2) ])));
          rejects (fun () ->
              Shape.shape (Shape.mop p (Pad [ (Int (-1), Int 3); (Int 0, Int 3) ])));
          rejects (fun () ->
              Shape.shape
                (Shape.mop p (Shrink [ (Int 0, Int (-1)); (Int 0, Int 3) ])));
          rejects (fun () -> Shape.shape (Shape.mop p (Permute [ 0; 0 ])));
          rejects (fun () -> Shape.shape (Shape.mop p (Permute [ 0 ])));
          rejects (fun () -> Shape.shape (Shape.mop p (Reshape (ints [ 5 ]))));
          rejects (fun () -> Shape.shape (Shape.mop p (Reshape (ints [ -6; -1 ]))));
          rejects (fun () ->
              Shape.shape (Shape.mop p (Shrink [ (Int 1, Int 2); (Int 0, Int 3) ])));
          rejects (fun () ->
              Shape.shape
                (Shape.mop p (Pad [ (Int 3, Int 2); (Int 0, Int 3) ]))));
      test "a symbolic reshape is accepted unless its sizes provably differ"
        (fun () ->
          let n = weak_var "n" 1 8 in
          let p = Shape.param ~shape:[ Sym n ] 0 Float32 in
          equal shape [ Sym n; Int 1 ]
            (Shape.shape (Shape.mop p (Reshape [ Sym n; Int 1 ]))));
    ]

(* Resolving *)

let resolving =
  group "resolve"
    [
      test "to_z, to_float and to_bool read a literal" (fun () ->
          equal z (Bigint.of_int 5) (Shape.to_z (Ops.int 5));
          equal float_exact 1.5 (Shape.to_float (Ops.float 1.5));
          is_true (Shape.to_bool (Ops.bool true)));
      test "to_z reads a typed constant and an integer sum of constants"
        (fun () ->
          equal z (Bigint.of_int 4) (Shape.to_z (Ops.int ~dtype:Int32 4));
          equal z (Bigint.of_int 11)
            (Shape.to_z Ops.O.(Ops.int ~dtype:Int32 4 + int 7));
          equal z (Bigint.of_int 2)
            (Shape.to_z Ops.O.(int 8 // Ops.int ~dtype:Int32 4)));
      test "to_bool decides comparisons of constants" (fun () ->
          is_true (Shape.to_bool Ops.O.(int 4 < int 7));
          is_true (Shape.to_bool Ops.O.(int 4 <= int 4));
          is_true (Shape.to_bool Ops.O.(int 4 <> int 7));
          is_false (Shape.to_bool Ops.O.(int 4 <> int 4));
          is_false (Shape.to_bool Ops.O.(int 4 > int 7)));
      test "to_bool decides a comparison the bounds decide" (fun () ->
          let v = weak_var "i" 1 10 in
          is_true (Shape.to_bool Ops.O.(v < int 20));
          is_true (Shape.to_bool Ops.O.(v // int 2 < int 20));
          is_false (Shape.to_bool Ops.O.(v < int 1));
          is_false (Shape.to_bool Ops.O.(v > int 11));
          let x = weak_var "x" 1 10 and y = weak_var "y" 5 10 in
          is_true (Shape.to_bool Ops.O.(Ops.maximum x y < int 20));
          is_false (Shape.to_bool Ops.O.(Ops.maximum x y < int 3)));
      test
        "to_bool decides a disjunction with true and a conjunction with false"
        (fun () ->
          is_true (Shape.to_bool Ops.O.(flag "b" lor bool true));
          is_false (Shape.to_bool Ops.O.(flag "b" land bool false)));
      test "to_bool rejects a condition with two possible values" (fun () ->
          let v = weak_var "i" 1 10 in
          rejects (fun () -> Shape.to_bool Ops.O.(flag "b" lor bool false));
          rejects (fun () -> Shape.to_bool Ops.O.(flag "b" land bool true));
          rejects (fun () -> Shape.to_bool Ops.O.(v < v + int 1));
          rejects (fun () -> Shape.to_bool Ops.O.((v > int 4) lor (v < int 6)));
          rejects (fun () -> Shape.to_bool Ops.O.(v < int 5)));
      test "to_bool, to_z and to_float reject another type" (fun () ->
          rejects (fun () -> Shape.to_bool (Ops.int 1));
          rejects (fun () -> Shape.to_z (Ops.bool true));
          rejects (fun () -> Shape.to_z (Ops.float 1.));
          rejects (fun () -> Shape.to_float (Ops.int 1)));
      test "resolve takes the default when the comparison is undecided"
        (fun () ->
          let u = weak_var "i" 1 10 in
          is_true (Shape.resolve Ops.O.(u < int 4));
          is_false (Shape.resolve ~default:false Ops.O.(u < int 4));
          is_true (Shape.resolve ~default:false Ops.O.(u < int 11));
          is_false (Shape.resolve ~default:false Ops.O.(u < int (-1)));
          is_false (Shape.resolve ~default:true Ops.O.(u < int (-1))));
      test "resolve rejects a node that is not boolean" (fun () ->
          rejects (fun () -> Shape.resolve (Ops.int 3)));
      test
        "simplify leaves a constant, and a sink of constants and stacks of \
         constants, alone" (fun () ->
          is_true (Shape.simplify (Ops.int 3) == Ops.int 3);
          let s =
            Ops.sink
              [
                Ops.int 1;
                Ops.float 2.;
                Ops.v Op.Stack;
                Ops.v ~src:[ Ops.int 1; Ops.int 2 ] Op.Stack;
              ]
          in
          is_true (Shape.simplify s == s));
      test "ssimplify is an integer constant as an integer" (fun () ->
          equal sint (Int 3) (Shape.ssimplify (Ops.int 3)));
      test "ssimplify reads a typed integer constant" (fun () ->
          equal sint (Int 3) (Shape.ssimplify (Ops.int ~dtype:Int32 3)));
      test "smax and smin of integers are integers" (fun () ->
          equal sint (Int 5) (Shape.smax [ Int 2; Int 5 ]);
          equal sint (Int 2) (Shape.smin [ Int 5; Int 2 ]);
          equal sint (Int 7) (Shape.smax [ Int 7 ]);
          rejects (fun () -> Shape.smax []);
          rejects (fun () -> Shape.smin []));
      test "smax and smin of a symbolic size bound it as max and min do"
        (fun () ->
          let v = weak_var "v" 3 10 in
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
    ]

let sint_module =
  let n = weak_var "n" 1 8 in
  group "Sint"
    [
      test "arithmetic on integers stays on integers" (fun () ->
          equal sint (Int 5) Shape.Sint.(Int 2 + Int 3);
          equal sint (Int (-1)) Shape.Sint.(Int 2 - Int 3);
          equal sint (Int 6) Shape.Sint.(Int 2 * Int 3);
          equal sint (Int (-4)) Shape.Sint.(Int (-7) // Int 2);
          equal sint (Int 1) Shape.Sint.(Int (-7) % Int 2);
          equal sint (Int (-1)) Shape.Sint.(Int 7 % Int (-2)));
      test "arithmetic with a node builds the node's operation" (fun () ->
          equal sint (Sym Ops.O.(n + int 1)) Shape.Sint.(Sym n + Int 1);
          equal sint (Sym Ops.O.(int 1 + n)) Shape.Sint.(Int 1 + Sym n);
          equal sint (Sym Ops.O.(n - int 1)) Shape.Sint.(Sym n - Int 1);
          equal sint (Sym Ops.O.(int 2 * n)) Shape.Sint.(Int 2 * Sym n);
          equal sint (Sym Ops.O.(n // int 2)) Shape.Sint.(Sym n // Int 2);
          equal sint (Sym Ops.O.(n % int 2)) Shape.Sint.(Sym n % Int 2));
      test "prod multiplies from 1" (fun () ->
          equal sint (Int 1) (Shape.Sint.prod []);
          equal sint (Int 6) (Shape.Sint.prod [ Int 2; Int 3 ]);
          equal sint (Sym Ops.O.(int 2 * n)) (Shape.Sint.prod [ Int 2; Sym n ]));
      test "comparisons of integers are known, of nodes conditions" (fun () ->
          is_true (Shape.Sint.(Int 2 < Int 3) = Known true);
          is_true (Shape.Sint.(Int 3 <= Int 2) = Known false);
          is_true (Shape.Sint.(Int 3 <> Int 3) = Known false);
          (match Shape.Sint.(Sym n < Int 3) with
          | Cond c -> equal uop Ops.O.(n < int 3) c
          | Known _ -> fail "a comparison with a node is a condition");
          match Shape.Sint.(Int 3 > Sym n) with
          | Cond c -> equal uop Ops.O.(n < int 3) c
          | Known _ -> fail "a comparison with a node is a condition");
      test "resolve is a known condition's value" (fun () ->
          is_true (Shape.Sint.resolve (Known true));
          is_false (Shape.Sint.resolve ~default:true (Known false)));
      test "resolve decides a node by its bounds" (fun () ->
          is_true (Shape.Sint.resolve Shape.Sint.(Sym n < Int 9));
          is_false (Shape.Sint.resolve ~default:false Shape.Sint.(Sym n < Int 5));
          is_true (Shape.Sint.resolve Shape.Sint.(Sym n <> Int 0)));
      test "equal compares integers by value and nodes by identity" (fun () ->
          is_true (Shape.Sint.equal (Int 3) (Int 3));
          is_false (Shape.Sint.equal (Int 3) (Sym (Ops.int 3)));
          is_true (Shape.Sint.equal (Sym n) (Sym (weak_var "n" 1 8))));
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
          let n = weak_var "n" 0 100 in
          equal int 7
            (Shape.sym_infer (Sym Ops.O.((n * int 2) + int 1)) [ ("n", 3) ]);
          equal int 9
            (Shape.sym_infer (Sym Ops.O.(Shape.bind n (i 7) + int 1)) [ ("n", 8) ]));
      test "an ordered value is its first source" (fun () ->
          let n = weak_var "n" 0 100 in
          let loop =
            Ops.end_ (Ops.v Op.Noop) [ Ops.range ~axis_type:Weak (Int 4) [ 0 ] ]
          in
          equal int 7
            (Shape.sym_infer
               (Sym (Ops.after Ops.O.((n * int 2) + int 1) [ loop ]))
               [ ("n", 3) ]));
      test "divisions round as their operations say" (fun () ->
          let n = weak_var "n" (-10) 10 in
          equal int (-3) (Shape.sym_infer (Sym Ops.O.(n // int 3)) [ ("n", -7) ]);
          equal int 2 (Shape.sym_infer (Sym Ops.O.(n % int 3)) [ ("n", -7) ]);
          equal int (-2)
            (Shape.sym_infer
               (Sym (Ops.alu n Op.Truncdiv [ Ops.int 3 ]))
               [ ("n", -7) ]);
          equal int (-1)
            (Shape.sym_infer
               (Sym (Ops.alu n Op.Truncmod [ Ops.int 3 ]))
               [ ("n", -7) ]);
          let d = weak_var "d" (-10) 10 in
          let at nv dv e = Shape.sym_infer (Sym e) [ ("n", nv); ("d", dv) ] in
          equal int (-2) (at 7 (-3) (Ops.alu n Op.Truncdiv [ d ]));
          equal int 1 (at 7 (-3) (Ops.alu n Op.Truncmod [ d ]));
          equal int (-1) (at (-7) 3 (Ops.alu n Op.Truncmod [ d ]));
          equal int 0 (at 7 0 (Ops.alu n Op.Truncdiv [ d ]));
          equal int 7 (at 7 0 (Ops.alu n Op.Truncmod [ d ])));
      test "a cast converts without truncating to a width" (fun () ->
          let n = weak_var "n" (-1000) 1000 in
          let half = Ops.O.(Ops.cast n Float32 / float 2.) in
          equal int 7
            (Shape.sym_infer
               (Sym Ops.O.(Ops.cast half Weak_int + int 10))
               [ ("n", -7) ]);
          equal int 300
            (Shape.sym_infer
               (Sym (Ops.cast (Ops.cast n Int8) Weak_int))
               [ ("n", 300) ]));
      test "a missing variable is rejected" (fun () ->
          rejects (fun () -> Shape.sym_infer (Sym (weak_var "n" 0 4)) []));
      test "sym_infer reads bits through bit reinterpretations" (fun () ->
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
      test "sym_infer rejects a bitcast to a boolean of a byte neither 0 nor 1"
        (fun () ->
          let byte name = Ops.bitcast (var ~dtype:Uint8 name 0 255) Bool in
          let picked =
            Ops.bitcast
              (Ops.where (Ops.cast (var "c" 0 1) Bool) (byte "x") (byte "y"))
              Uint8
          in
          let at x =
            Shape.sym_infer (Sym picked) [ ("c", 1); ("x", x); ("y", 0) ]
          in
          equal int 1 (at 1);
          rejects (fun () -> at 2));
      test "sym_infer evaluates an expression nested 200 deep" (fun () ->
          let a = weak_var "a" 1 8192 and b = weak_var "b" 0 8191 in
          let step e =
            Ops.O.((Ops.maximum (e * (b + a)) (int (-33554432)) * int (-1)) + a)
          in
          let rec nest n e = if n = 0 then e else nest (n - 1) (step e) in
          equal int 1 (Shape.sym_infer (Sym (nest 200 a)) [ ("a", 1); ("b", 0) ]));
    ]

(* sym_compile computes what exact arithmetic does, on expressions whose
   values fit an [int] and on those whose products do not fit one. *)

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

let outcome f = match f () with v -> Ok v | exception e -> Error e

let exn =
  Testable.make
    ~pp:(fun ppf e -> Format.pp_print_string ppf (Printexc.to_string e))
    ~equal:(fun e e' -> Printexc.to_string e = Printexc.to_string e')

(* The reference: [e] simplified, then each node computed on unbounded
   integers, each variable read from [env]. *)
let exact_value e env =
  let values = Ops.Tbl.create 16 in
  let eval n : Dtype.const =
    match Ops.op n with
    | Const -> Ops.value n
    | Param -> `Int (Bigint.of_int (List.assoc (Ops.expr n) env))
    (* a variable's size source, which no value reads *)
    | Stack when Ops.src n = [] -> `Bool false
    | op ->
        Ops.exec_alu ~truncate_output:false op (Ops.dtype n)
          (List.map (Ops.Tbl.find values) (Ops.src n))
  in
  match Ops.topovisit (Shape.simplify e) eval values with
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
      equal (result int exn) exact compiled)

let compiles_exactly ~bound =
  agrees
    ~name:
      (Printf.sprintf "computes what exact arithmetic does, variables within %d"
         bound)
    (weak_var "a" (-bound) bound)
    (weak_var "b" (-bound) bound)
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
    (weak_var "a" min_int max_int)
    (weak_var "b" min_int max_int)
    some

(* Values whose [int] arithmetic overflows on the way, and their exact values,
   computed by hand. *)
let exactly =
  let a = weak_var "a" min_int max_int and b = weak_var "b" min_int max_int in
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
    ("a product past max_int", (a * b) // int 4, p31, p31, p60);
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
  let a = weak_var "a" min_int max_int and b = weak_var "b" min_int max_int in
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

(* sym_compile_z computes what exact arithmetic does, of any size. *)

let z = Testable.make ~pp:Bigint.pp_print ~equal:Bigint.equal

let both_z e va vb =
  let env = [ ("a", va); ("b", vb) ] in
  let var u = List.assoc (Ops.expr u) in
  ( outcome (fun () -> exact_z e env),
    outcome (fun () -> Shape.sym_compile_z (Sym e) var env) )

let exact_compilation =
  let some = Gen.of_list ~pp:Format.pp_print_int extremes in
  let a = weak_var "a" min_int max_int and b = weak_var "b" min_int max_int in
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
          equal (result z exn) exact compiled);
      cases "computes values that fit no int"
        ~id:(fun (name, _, _, _) -> name)
        unfit
        (fun (_, e, va, vb) ->
          let exact, compiled = both_z e va vb in
          is_ok ~msg:"exact" ~pp:(Testable.pp exn) exact;
          equal (result z exn) exact compiled);
    ]

let compilation =
  group "sym_compile"
    [
      compiles_exactly ~bound:50;
      compiles_exactly ~bound:(1 lsl 31);
      compiles_at_the_extremes;
      cases "computes exactly where int arithmetic overflows"
        ~id:(fun (name, _, _, _, _) -> name)
        exactly computes_exactly;
      cases "raises as exact arithmetic does where a value fits no int"
        ~id:(fun (name, _, _, _) -> name)
        unfit raises_as_exact;
      test "an integer is itself" (fun () ->
          equal int 5 (Shape.sym_compile (Int 5) (fun _ () -> 0) ()));
      test "a variable is what var reads" (fun () ->
          let n = weak_var "n" 0 100 in
          equal int 7
            (Shape.sym_compile
               (Sym Ops.O.((n * int 2) + int 1))
               (fun _ x -> x)
               3));
      test "a variable var refuses raises" (fun () ->
          rejects (fun () ->
              Shape.sym_compile
                (Sym Ops.O.(weak_var "n" 0 4 + int 1))
                (fun _ () -> invalid_arg "no n")
                ()));
    ]
