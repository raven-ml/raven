(* Values: arithmetic on constants, bounds, resolving, divisibility and symbolic
   inference. *)

open Windtrap
open Tolk
open Common

let str pp x = Format.asprintf "%a" pp x

let bool_cell = function
  | "True" -> true
  | "False" -> false
  | s -> invalid_arg s

(* exec_alu *)

(* The host's libm computes these operations, and libms legitimately differ in
   the last place: macOS's sin 2.5 is one unit in the last place from glibc's,
   which is correctly rounded. Their results agree within one. *)
let host_libm = Op.[ Sin; Log2; Exp2; Pow ]

let within_ulp =
  let near a b =
    Float.equal a b
    || Float.equal (Float.succ a) b
    || Float.equal (Float.pred a) b
  in
  Testable.make ~pp:(Testable.pp const) ~equal:(fun c0 c1 ->
      match (c0, c1) with
      | `Float a, `Float b -> near a b
      | _ -> Testable.equal const c0 c1)

let alu_row cell =
  let o = op_of_cell (cell "op") and dt = dtype_of_cell (cell "dtype") in
  let args = consts_of_cell (cell "operands")
  and truncate_output = bool_cell (cell "truncate") in
  let w = if List.exists (Op.equal o) host_libm then within_ulp else const in
  expect w const_of_cell (cell "result") (fun () ->
      Ops.exec_alu ~truncate_output o dt args)

let alu dt o args = Ops.exec_alu o dt args
let exact dt o args = Ops.exec_alu ~truncate_output:false o dt args

let int_dtypes =
  Dtype.[ Int8; Uint8; Int16; Uint16; Int32; Uint32; Int64; Uint64; Weak_int ]

let gen_int_dtype = Gen.of_list ~pp:(Testable.pp dtype) int_dtypes
let small = Gen.map (fun n -> `Int (Bigint.of_int n)) (Gen.int_range (-300) 300)
let gen_int = Gen.map (fun n -> `Int n) Dtypes.integer

let binary_ops =
  Op.
    [
      Add;
      Sub;
      Mul;
      Cdiv;
      Cmod;
      Floordiv;
      Floormod;
      Max;
      Xor;
      Or;
      And;
      Cmplt;
      Cmpne;
      Cmpeq;
      Shl;
      Shr;
    ]

let exec_alu =
  group "exec_alu"
    [
      Golden.cases "exec_alu_values.golden"
        ~key:[ "op"; "dtype"; "operands"; "truncate" ]
        alu_row;
      test
        "max is its second operand only if that is greater, so a NaN wins first"
        (fun () ->
          equal const (f Float.nan) (alu Float32 Op.Max [ f Float.nan; f 1. ]);
          equal const (f 1.) (alu Float32 Op.Max [ f 1.; f Float.nan ]));
      test
        "an invalid operation's NaN is the canonical positive quiet NaN, \
         whatever the host gives" (fun () ->
          let inf = Float.infinity in
          (* The bits, since NaNs compare equal as values. x86 gives
             0xfff8000000000000. *)
          let bits = function
            | `Float x -> Printf.sprintf "%Lx" (Int64.bits_of_float x)
            | c -> str Dtype.pp_const c
          in
          List.iter
            (fun dt ->
              let nan = bits (Dtype.truncate dt (f Dtype.nan) :> Dtype.const) in
              List.iter
                (fun (o, args) ->
                  equal ~msg:(str Op.pp o) string nan (bits (alu dt o args)))
                Op.
                  [
                    (Add, [ f inf; f (-.inf) ]);
                    (Sub, [ f inf; f inf ]);
                    (Mul, [ f 0.; f inf ]);
                    (Fdiv, [ f 0.; f 0. ]);
                    (Fdiv, [ f inf; f (-.inf) ]);
                    (Sqrt, [ f (-1.) ]);
                    (Log2, [ f (-1.) ]);
                    (Sin, [ f inf ]);
                    (Pow, [ f (-2.); f 0.5 ]);
                  ])
            Dtype.[ Float32; Float64 ];
          equal string "7ff8000000000000" (bits (f Dtype.nan)));
      test "a float multiply-add folds rounded once (D25)" (fun () ->
          let e = Float.ldexp 1. in
          equal ~msg:"float32" const
            (f (e (-24)))
            (alu Float32 Op.Mulacc
               [ f (1. +. e (-12)); f (1. +. e (-12)); f (-1. -. e (-11)) ]);
          equal ~msg:"float64" const
            (f (e (-54)))
            (alu Float64 Op.Mulacc
               [ f (1. +. e (-27)); f (1. +. e (-27)); f (-1. -. e (-26)) ]);
          (* [1 + 3 2^-24 - 2^-70] is a double's midpoint of two float32s
             before it is a float32: rounded twice it ties to the even one. *)
          equal ~msg:"float32, past a double's rounding" const
            (f (1. +. e (-23)))
            (alu Float32 Op.Mulacc
               [ f (1. +. e (-23)); f (e (-24) -. e (-47)); f (1. +. e (-23)) ]));
      test "a shift by a negative count has no value" (fun () ->
          List.iter
            (fun o ->
              raises_match
                (Exn.invalid_arg ~substring:"a shift by a negative count, -1")
                (fun () -> ignore (alu Int32 o [ i 8; i (-1) ])))
            Op.[ Shl; Shr ]);
      test "an overflowing power is an infinity of the base's sign" (fun () ->
          equal const (f Float.infinity) (alu Float64 Op.Pow [ f 2.; f 10000. ]);
          equal const (f Float.neg_infinity)
            (alu Float64 Op.Pow [ f (-2.); f 10001. ]);
          equal const (f Float.infinity) (alu Float64 Op.Pow [ f 1e300; f 2. ]));
      test "sqrt of zero is zero" (fun () ->
          equal const (f 0.) (alu Float32 Op.Sqrt [ f 0. ]));
      prop "truncating is Dtype.truncate of the exact result"
        (Gen.triple gen_int_dtype
           (Gen.of_list ~pp:Op.pp
              Op.[ Add; Sub; Mul; Cdiv; Floordiv; Floormod; Xor; And; Or; Max ])
           (Gen.pair gen_int gen_int))
        (fun (dt, o, (a, b)) ->
          match exact dt o [ a; b ] with
          | #Dtype.value as v ->
              equal const
                (Dtype.truncate dt v :> Dtype.const)
                (alu dt o [ a; b ])
          | `Invalid -> fail "no Invalid operand");
      prop "Invalid poisons every binary operation"
        (Gen.pair (Gen.of_list ~pp:Op.pp binary_ops) small)
        (fun (o, x) ->
          let dt =
            if Op.Set.mem o Op.Set.comparison then Dtype.Bool else Int32
          in
          equal const `Invalid (alu dt o [ `Invalid; x ]);
          equal const `Invalid (alu dt o [ x; `Invalid ]));
      prop "addition, multiplication and max commute"
        (Gen.triple
           (Gen.of_list ~pp:Op.pp Op.[ Add; Mul; Max; And; Or; Xor ])
           gen_int gen_int)
        (fun (o, a, b) ->
          Law.commutative const (fun x y -> exact Weak_int o [ x; y ]) (a, b));
      prop "truncated division and remainder recompose the dividend"
        (Gen.pair gen_int gen_int) (fun (a, b) ->
          assume (b <> `Int Bigint.zero);
          let q = exact Weak_int Op.Cdiv [ a; b ]
          and r = exact Weak_int Op.Cmod [ a; b ] in
          equal const a
            (exact Weak_int Op.Add [ exact Weak_int Op.Mul [ q; b ]; r ]));
      prop
        "floor division and remainder recompose the dividend, the remainder \
         signed as the divisor"
        (Gen.pair gen_int gen_int) (fun (a, b) ->
          assume (b <> `Int Bigint.zero);
          let q = exact Weak_int Op.Floordiv [ a; b ]
          and r = exact Weak_int Op.Floormod [ a; b ] in
          equal const a
            (exact Weak_int Op.Add [ exact Weak_int Op.Mul [ q; b ]; r ]);
          match (r, b) with
          | `Int r, `Int b ->
              is_true (Bigint.sign r = 0 || Bigint.sign r = Bigint.sign b)
          | _ -> fail "integer remainder");
      test "a float division by zero is a float zero, weak or not" (fun () ->
          List.iter
            (fun o ->
              equal ~msg:(str Op.pp o) const (f 0.)
                (alu Weak_float o [ f 1.; f 0. ]);
              equal ~msg:(str Op.pp o) const (f 0.)
                (alu Float32 o [ f 1.; f 0. ]))
            Op.[ Cdiv; Floordiv ]);
      test "a void result is not truncated" (fun () ->
          equal const (i 3) (alu Void Op.Add [ i 1; i 2 ]));
      test "division by zero is zero" (fun () ->
          List.iter
            (fun o ->
              equal ~msg:(str Op.pp o) const (i 0)
                (exact Weak_int o [ i 7; i 0 ]))
            Op.[ Cdiv; Floordiv ]);
    ]

(* Bounds *)

let variable dt lo hi name = Ops.variable ~dtype:dt name lo hi

let binary_bounds cell =
  let o = op_of_cell (cell "op") and dt = dtype_of_cell (cell "dtype") in
  let x =
    variable dt (value_of_cell (cell "a_lo")) (value_of_cell (cell "a_hi")) "a"
  in
  let y =
    variable dt (value_of_cell (cell "b_lo")) (value_of_cell (cell "b_hi")) "b"
  in
  check_bounds (Ops.alu x o [ y ])
    (value_of_cell (cell "vmin"), value_of_cell (cell "vmax"))

let cast_bounds cell =
  let x =
    variable
      (dtype_of_cell (cell "from"))
      (value_of_cell (cell "lo"))
      (value_of_cell (cell "hi"))
      "x"
  in
  check_bounds
    (Ops.cast x (dtype_of_cell (cell "to")))
    (value_of_cell (cell "vmin"), value_of_cell (cell "vmax"))

(* The reference interpreter: the value of an integer expression over variables,
   each operation computed exactly. It discards a case where an operation leaves
   its type, where the result is undefined. *)
let rec eval env u =
  let fits dt v =
    match v with
    | `Int n when Dtype.is_int dt && not (Dtype.equal dt Weak_int) -> (
        match (Dtype.min dt, Dtype.max dt) with
        | `Int lo, `Int hi -> Bigint.leq lo n && Bigint.leq n hi
        | _ -> true)
    | _ -> true
  in
  let v =
    match (Ops.op u, Ops.arg u) with
    | Op.Param, Param { name = Some name; _ } -> List.assoc name env
    | Op.Const, Const (#Dtype.value as c) -> c
    | Op.Cast, Dtype dt -> (
        match eval env (Ops.nth u 0) with
        | `Bool b ->
            if Dtype.is_bool dt then `Bool b
            else `Int (if b then Bigint.one else Bigint.zero)
        | v -> v)
    | o, _ -> (
        match
          exact (Ops.dtype u) o
            (List.map (fun s -> (eval env s :> Dtype.const)) (Ops.src u))
        with
        | #Dtype.value as v -> v
        | `Invalid -> fail "Invalid in an expression without one")
  in
  assume (fits (Ops.dtype u) v);
  v

let compare_value (v0 : Dtype.value) (v1 : Dtype.value) =
  match (v0, v1) with
  | `Int a, `Int b -> Bigint.compare a b
  | `Bool a, `Bool b -> Bool.compare a b
  | `Float a, `Float b -> Float.compare a b
  | _ -> invalid_arg "values of different kinds"

let ordered_value = Testable.with_compare compare_value value
let gen_point = Gen.pair (Gen.int_range (-8) 8) (Gen.int_range 1 5)

(* An operation on a committed integer, as compiled code computes it, wraps
   at the type's width. [x o y], then [o'] with [x] again, over narrow variables
   of 61 values starting anywhere in their type, at a point of each. *)
let gen_wrapping =
  let narrow = Dtype.[ Int8; Uint8; Int16; Uint16 ] in
  let ops = Op.[ Add; Sub; Mul; Xor; Shl ] in
  Gen.triple
    (Gen.of_list ~pp:(Testable.pp dtype) narrow)
    (Gen.pair (Gen.of_list ~pp:Op.pp ops) (Gen.of_list ~pp:Op.pp ops))
    (Gen.triple (Gen.int_range 0 65535) (Gen.int_range 0 65535)
       (Gen.int_range 0 60))

let wrapping_case (dt, (o0, o1), (a, b, k)) =
  let lo = Bigint.to_int (Dtype.Value.to_z (Dtype.min dt))
  and hi = Bigint.to_int (Dtype.Value.to_z (Dtype.max dt)) in
  let start n = min hi (lo + (n mod (hi - lo + 1))) in
  let x0 = start a and y0 = start b in
  let x1 = min hi (x0 + 60) and y1 = min hi (y0 + 60) in
  let x = variable dt (i x0) (i x1) "x" and y = variable dt (i y0) (i y1) "y" in
  let apply u o v =
    if o = Op.Shl then Ops.O.(u lsl Ops.int ~dtype:dt 3) else Ops.alu u o [ v ]
  in
  let u = apply (apply x o0 y) o1 x in
  let vars = [ ("x", i (min x1 (x0 + k))); ("y", i (max y0 (y1 - k))) ] in
  (u, vars)

(* A float sum, difference or product of variables over intervals of a
   float type's values, at a point of each, or a selection of one by comparing
   them ([Op.Where]: [x] where [x < y], else [x] where [y < x], else [y]).
   Magnitudes span the type's exponents, so that sums overflow and products fall
   to the subnormals. *)
let gen_float_bounds =
  let open Gen in
  let* dt =
    of_list ~pp:(Testable.pp dtype)
      Dtype.[ Float16; Bfloat16; Float32; Float64 ]
  in
  let e, m = Dtype.finfo dt in
  let emax = 1 lsl (e - 1) in
  let magnitude =
    let+ x = float_range (-2.) 2. and+ k = int_range (-emax - m) emax in
    Float.ldexp x k
  in
  let+ o = of_list ~pp:Op.pp Op.[ Add; Sub; Mul; Where ]
  and+ ends = quad magnitude magnitude magnitude magnitude
  and+ at = pair (float_range 0. 1.) (float_range 0. 1.) in
  (dt, o, ends, at)

let float_bounds_case (dt, o, (a0, a1, b0, b1), (s, t)) =
  let value x =
    match Dtype.truncate dt (`Float x) with `Float y -> y | _ -> Float.nan
  in
  let interval x0 x1 = (value (Float.min x0 x1), value (Float.max x0 x1)) in
  let (a0, a1), (b0, b1) = (interval a0 a1, interval b0 b1) in
  assume (List.for_all Float.is_finite [ a0; a1; b0; b1 ]);
  let point lo hi s =
    value (Float.min hi (Float.max lo ((lo *. (1. -. s)) +. (hi *. s))))
  in
  let x = variable dt (f a0) (f a1) "x" and y = variable dt (f b0) (f b1) "y" in
  let u =
    if Op.equal o Where then
      Ops.where (Ops.lt x y) x (Ops.where (Ops.lt y x) x y)
    else Ops.alu x o [ y ]
  in
  match
    Interpreter.eval
      ~vars:[ ("x", f (point a0 a1 s)); ("y", f (point b0 b1 t)) ]
      u
  with
  | #Dtype.value as v ->
      at_most ordered_value ~than:v (Ops.vmin u);
      at_least ordered_value ~than:v (Ops.vmax u)
  | `Invalid -> fail "Invalid in an expression without one"

let bounds_group =
  group "bounds"
    [
      Golden.cases "binary_bounds.golden"
        ~key:[ "op"; "dtype"; "a_lo"; "a_hi"; "b_lo"; "b_hi" ]
        binary_bounds;
      Golden.cases "load_bounds.golden" (fun cell ->
          let dt = dtype_of_cell (cell "dtype") in
          let hex = cell "bytes" in
          let data =
            String.init
              (String.length hex / 2)
              (fun k ->
                Char.chr (int_of_string ("0x" ^ String.sub hex (2 * k) 2)))
          in
          let table = Ops.v ~arg:(Bytes data) Op.Binary in
          let table =
            if Dtype.equal dt Uint8 then table else Ops.bitcast table dt
          in
          let r =
            Ops.range (Int (String.length data / Dtype.itemsize dt)) [ 0 ]
          in
          check_bounds
            (Ops.load (Ops.index table [ r ]) [])
            (value_of_cell (cell "vmin"), value_of_cell (cell "vmax")));
      Golden.cases "cast_bounds.golden"
        ~key:[ "from"; "lo"; "hi"; "to" ]
        cast_bounds;
      prop "bounds hold every value the expression takes"
        (Gen.pair Nodes.gen_recipe gen_point) (fun (r, (x0, x1)) ->
          let u = Nodes.build (Nodes.leaves ()) r in
          let v = eval [ ("v0", i x0); ("v1", i x1) ] u in
          at_most ordered_value ~than:v (Ops.vmin u);
          at_least ordered_value ~than:v (Ops.vmax u));
      prop "bounds hold the value a committed integer wraps to"
        gen_wrapping (fun case ->
          let u, vars = wrapping_case case in
          match Interpreter.eval ~vars u with
          | #Dtype.value as v ->
              at_most ordered_value ~than:v (Ops.vmin u);
              at_least ordered_value ~than:v (Ops.vmax u)
          | `Invalid -> fail "Invalid in an expression without one");
      prop "bounds hold every value a float operation rounds to"
        gen_float_bounds float_bounds_case;
      test "a float operation of bounded operands is bounded" (fun () ->
          let x = variable Float32 (f 0.) (f 1.) "x" in
          let y = variable Float32 (f (-2.)) (f 3.) "y" in
          let widened lo hi =
            let tiny = Float.ldexp 1. (-126) and rel = Float.ldexp 1. (-23) in
            ( f (lo -. (Float.abs lo *. rel) -. tiny),
              f (hi +. (Float.abs hi *. rel) +. tiny) )
          in
          check_bounds (Ops.alu x Op.Add [ y ]) (widened (-2.) 4.);
          check_bounds (Ops.alu x Op.Sub [ y ]) (widened (-3.) 3.);
          check_bounds (Ops.alu x Op.Mul [ y ]) (widened (-2.) 3.));
      test
        "a float operation of an operand within the subnormals bounds a flushed \
         operand too" (fun () ->
          let x = variable Float32 (f 1e5) (f 2e5) "x" in
          let u = Ops.alu x Op.Mul [ Ops.float ~dtype:Float32 1e-40 ] in
          at_most ordered_value ~than:(f 0.) (Ops.vmin u);
          at_least ordered_value ~than:(f 2e-35) (Ops.vmax u));
      test
        "a float operation of an unbounded operand, or that can overflow, has \
         its type's bounds" (fun () ->
          let full dt = (Dtype.min dt, Dtype.max dt) in
          let one dt = Ops.float ~dtype:dt 1. in
          let h = variable Float16 (f 0.) (f 65504.) "h" in
          let w = variable Weak_float (f 0.) (f 1.) "w" in
          check_bounds
            (Ops.alu (Ops.param 0 Float32) Op.Add [ one Float32 ])
            (full Float32);
          check_bounds (Ops.alu h Op.Add [ h ]) (full Float16);
          check_bounds (Ops.alu h Op.Mul [ h ]) (full Float16);
          check_bounds (Ops.alu w Op.Add [ one Weak_float ]) (full Weak_float));
      test "a float selection by a comparison narrows what it selects"
        (fun () ->
          let x = variable Float32 (f (-10.)) (f 10.) "x" in
          let p = Ops.param 0 Float32 in
          let c = Ops.float ~dtype:Float32 2. in
          let minus_c = Ops.float ~dtype:Float32 (-2.) in
          check_bounds (Ops.where (Ops.lt x c) x minus_c) (f (-10.), f 2.);
          check_bounds (Ops.where (Ops.lt minus_c x) x c) (f (-2.), f 10.);
          check_bounds (Ops.where (Ops.lt x c) c x) (f (-10.), f 10.);
          let above = Ops.where (Ops.lt minus_c p) p minus_c in
          check_bounds above (f (-2.), f Float.infinity);
          check_bounds (Ops.where (Ops.lt above c) above c) (f (-2.), f 2.));
      test "a float truncation and negation map their operand's bounds"
        (fun () ->
          let x = variable Float32 (f (-2.5)) (f 3.75) "x" in
          check_bounds (Ops.alu x Op.Trunc []) (f (-2.), f 3.);
          check_bounds (Ops.alu x Op.Neg []) (f (-3.75), f 2.5));
      prop "vmin is at most vmax" Nodes.gen_recipe (fun r ->
          let u = Nodes.build (Nodes.leaves ()) r in
          at_most ordered_value ~than:(Ops.vmax u) (Ops.vmin u));
      test "a constant is its own bounds" (fun () ->
          check_bounds (Ops.int 42) (int_bounds 42 42));
      test "bounds are exact integers, whatever their size" (fun () ->
          let huge = Bigint.shift_left Bigint.one 200 in
          let c n = Ops.const (`Int n) in
          check_bounds Ops.O.(c huge - c (Bigint.pred huge)) (i 1, i 1);
          check_bounds
            Ops.O.(c huge * c huge)
            (`Int (Bigint.mul huge huge), `Int (Bigint.mul huge huge));
          check_bounds
            Ops.O.(c huge lsl int 100)
            ( `Int (Bigint.shift_left huge 100),
              `Int (Bigint.shift_left huge 100) ));
      test "a constant at its type's edge has exact bounds" (fun () ->
          List.iter
            (fun (dt, n) ->
              check_bounds ~msg:(str Dtype.pp dt)
                (Ops.const ~dtype:dt (`Int n))
                (`Int n, `Int n))
            Dtype.
              [
                (Int64, Bigint.of_int64 Int64.max_int);
                (Int64, Bigint.of_int64 Int64.min_int);
                (Uint64, Bigint.pred (Bigint.shift_left Bigint.one 64));
              ];
          check_bounds (Ops.param 7 Uint64) (Dtype.min Uint64, Dtype.max Uint64));
      test "a NaN constant has its type's bounds" (fun () ->
          check_bounds
            (Ops.float ~dtype:Float32 Float.nan)
            (f Float.neg_infinity, f Float.infinity));
      test "Invalid has no single value" (fun () ->
          not_equal value (Ops.vmin Ops.invalid) (Ops.vmax Ops.invalid));
      test "a stack's bounds span its values, Invalid left out" (fun () ->
          check_bounds
            (Ops.consts [ i 0; i 4; `Invalid; `Invalid ])
            (int_bounds 0 4);
          check_bounds (Ops.consts [ i 42 ]) (int_bounds 42 42);
          check_bounds
            (Ops.consts [ i 10; i 20; i (-5); i 7 ])
            (int_bounds (-5) 20);
          check_bounds
            (Ops.consts [ `Bool true; `Bool false; `Bool false ])
            (`Bool false, `Bool true);
          check_bounds
            (Ops.consts [ f 1.5; f (-3.2); f 0. ])
            (Dtype.truncate Float32 (f (-3.2)), f 1.5));
      test "a comparison of constants is decided" (fun () ->
          let c = Ops.int 42 in
          check_bounds (Ops.ne c (Ops.int 42)) (`Bool false, `Bool false);
          check_bounds (Ops.ne c (Ops.int 43)) (`Bool true, `Bool true);
          check_bounds (Ops.ne c (Ops.int 41)) (`Bool true, `Bool true));
      test "a variable offset or scaled moves its bounds" (fun () ->
          let x = weak_var "x" 10 20 in
          check_bounds Ops.O.(x + int 5) (int_bounds 15 25);
          check_bounds Ops.O.(x - int 5) (int_bounds 5 15);
          check_bounds Ops.O.(int 5 - x) (int_bounds (-15) (-5));
          check_bounds Ops.O.(weak_var "y" (-3) 4 * int 2) (int_bounds (-6) 8);
          check_bounds
            Ops.O.(weak_var "y" 2 5 * int (-3))
            (int_bounds (-15) (-6));
          check_bounds
            Ops.O.(weak_var "y" (-2) 5 * int (-3))
            (int_bounds (-15) 6));
      test "a mask bounds a variable by the mask" (fun () ->
          let x = weak_var "x" 10 20 in
          check_bounds Ops.O.(x land int 5) (int_bounds 0 5);
          check_bounds Ops.O.(x land int 15) (int_bounds 0 15);
          check_bounds Ops.O.(x land int 32) (int_bounds 0 0);
          check_bounds Ops.O.(x land int 48) (int_bounds 0 16);
          let s = var "x" (-100) 100 in
          check_bounds Ops.O.(s land int 511) (int_bounds 0 511);
          check_bounds Ops.O.(s land int 0x7FFFFFFF) (int_bounds 0 0x7FFFFFFF);
          check_bounds Ops.O.(s land int (-1)) (Dtype.min Int32, Dtype.max Int32));
      test "maximum then minimum clamps" (fun () ->
          check_bounds
            (Ops.minimum
               (Ops.maximum (weak_var "x" 0 10) (Ops.int 5))
               (Ops.int 8))
            (int_bounds 5 8));
      test "a selection spans both branches" (fun () ->
          let x = weak_var "x" 0 10 in
          check_bounds
            (Ops.where
               Ops.O.(x < int 5)
               (weak_var "y" 1 11) (weak_var "z" 2 12))
            (int_bounds 1 12));
      test "a shift by a constant shifts the bounds" (fun () ->
          check_bounds
            Ops.O.(weak_var "x" 0 10 lsl int 5)
            (int_bounds 0 (10 lsl 5));
          check_bounds
            Ops.O.(weak_var "x" 0 10 lsr int 2)
            (int_bounds 0 (10 lsr 2)));
      test "a shift by a negative constant has its type's bounds" (fun () ->
          let x = var "x" 0 10 in
          let count = Ops.cast (Ops.int (-1)) Int32 in
          check_bounds Ops.O.(x lsl count) (Dtype.min Int32, Dtype.max Int32);
          check_bounds Ops.O.(x lsr count) (Dtype.min Int32, Dtype.max Int32));
      test "a hardware index counts from 0 to below its end" (fun () ->
          check_bounds
            (Ops.special (Sym (var "i" 1 10)) "gidx0")
            (int_bounds 0 9);
          check_bounds (Ops.special (Int 0) "lidx0") (int_bounds 0 (-1)));
      test "an empty range divides to 0" (fun () ->
          let r = Ops.range (Int 0) [ 0 ] in
          check_bounds r (int_bounds 0 (-1));
          check_bounds Ops.O.(r // int 4) (int_bounds 0 0);
          check_bounds Ops.O.(r % int 4) (int_bounds 0 0));
      test "a product with an unbounded float is unbounded, never NaN"
        (fun () ->
          let y =
            Ops.load
              (Ops.index
                 (Ops.param ~shape:(ints [ 1 ]) 0 Float32)
                 [ Ops.int 0 ])
              []
          in
          check_bounds
            Ops.O.(float 0. * y)
            (f Float.neg_infinity, f Float.infinity));
      test "a load of an integer buffer has its type's bounds" (fun () ->
          let v =
            Ops.load
              (Ops.index (Ops.param ~shape:(ints [ 1 ]) 1 Int32) [ Ops.int 0 ])
              []
          in
          check_bounds Ops.O.(v // int 32) (int_bounds (-67108864) 67108863));
      test "a load from constant bytes spans the bytes" (fun () ->
          let table = Ops.v ~arg:(Bytes "\x03\x09\x01") Op.Binary in
          check_bounds
            (Ops.load (Ops.index table [ Ops.range (Int 3) [ 0 ] ]) [])
            (int_bounds 1 9));
      test "a pad adds zeros to the bounds" (fun () ->
          let x = Ops.expand (Ops.int ~dtype:Int32 5) (ints [ 2 ]) in
          check_bounds (Ops.pad x [ Some (Int 1, Int 1) ]) (int_bounds 0 5));
      test "a copy and a contiguous keep their source's bounds" (fun () ->
          let src = Ops.O.(Ops.placeholder ~slot:0 [ 4 ] Int32 land int 3) in
          check_bounds (Ops.contiguous src) (int_bounds 0 3);
          check_bounds (Ops.copy_to_device src (Single "NULL")) (int_bounds 0 3));
      test "a variable cast to float, bool or unsigned" (fun () ->
          let x = var "x" (-10) 10 in
          check_bounds (Ops.cast x Float32) (f (-10.), f 10.);
          check_bounds (Ops.cast x Bool) (`Bool false, `Bool true);
          check_bounds (Ops.cast x Uint32) (Dtype.min Uint32, Dtype.max Uint32));
      test
        "a typed integer constant outside its type is bounded by its wrapped \
         value, a non-finite one by the type" (fun () ->
          check_bounds (Ops.int ~dtype:Int8 300) (int_bounds 44 44);
          check_bounds (Ops.int ~dtype:Uint8 (-1)) (int_bounds 255 255);
          check_bounds
            (Ops.cast (Ops.float Float.infinity) Int32)
            (Dtype.min Int32, Dtype.max Int32);
          check_bounds
            (Ops.cast (Ops.float ~dtype:Float32 4.5) Int32)
            (int_bounds 4 4));
      test
        "a value of a float type without infinities, which may be NaN, has no \
         finite bounds" (fun () ->
          let unknown = (f Float.neg_infinity, f Float.infinity) in
          List.iter
            (fun dt ->
              check_bounds (Ops.param 0 dt) unknown;
              check_bounds (Ops.cast (Ops.param 0 dt) Float32) unknown;
              check_bounds (Ops.cast (Ops.param 0 Float32) dt) unknown;
              let x = variable Float32 (f (-1.)) (f 2.) "x" in
              check_bounds (Ops.cast x dt) (f (-1.), f 2.);
              let greatest = Dtype.max dt in
              let x = variable Float32 (f (-1e6)) (f 1e6) "x" in
              check_bounds (Ops.cast x dt)
                (Dtype.Value.( ~- ) greatest, greatest))
            Dtype.[ Fp8e4m3; Fp8e4m3fnuz; Fp8e5m2fnuz ]);
      test "a committed integer that can leave its type has its bounds"
        (fun () ->
          let full dt = (Dtype.min dt, Dtype.max dt) in
          let u = var ~dtype:Uint8 "u" 0 255 and y = var ~dtype:Int8 "y" 0 50 in
          check_bounds Ops.O.(u + int 1) (full Uint8);
          check_bounds Ops.O.(u - int 1) (full Uint8);
          check_bounds Ops.O.(y + int 100) (full Int8);
          check_bounds Ops.O.(y * int 4) (full Int8);
          check_bounds Ops.O.(y lsl int 2) (full Int8);
          check_bounds Ops.O.(u lxor int (-1)) (full Uint8);
          check_bounds Ops.O.(y + int 50) (int_bounds 50 100);
          check_bounds Ops.O.(weak_var "w" 0 50 * int 4) (int_bounds 0 200));
      test "an integer cast to a signed type it leaves wraps" (fun () ->
          let w = var "w" 0 255 in
          check_bounds (Ops.cast w Int8) (Dtype.min Int8, Dtype.max Int8);
          check_bounds (Ops.cast (var "v" 0 100) Int8) (int_bounds 0 100);
          check_bounds (Ops.cast (Ops.cast w Uint8) Int32) (int_bounds 0 255));
      test "a constant table holding a NaN has its type's bounds"
        (fun () ->
          let bits x =
            let b = Bytes.create 4 in
            Bytes.set_int32_le b 0 (Int32.bits_of_float x);
            Bytes.to_string b
          in
          let table =
            Ops.bitcast
              (Ops.v
                 ~arg:
                   (Bytes
                      (String.concat "" (List.map bits [ 1.; Float.nan; 2. ])))
                 Op.Binary)
              Float32
          in
          check_bounds
            (Ops.load (Ops.index table [ Ops.range (Int 3) [ 0 ] ]) [])
            (Dtype.min Float32, Dtype.max Float32));
      test "exact is whether committed integer values fit their type" (fun () ->
          is_true (Ops.exact Int8 [ i (-128); i 127 ]);
          is_false (Ops.exact Int8 [ i 0; i 128 ]);
          is_false (Ops.exact Uint8 [ i (-1) ]);
          is_true
            (Ops.exact Weak_int
               [ i (-1); `Int (Bigint.shift_left Bigint.one 100) ]));
      test "overflows is whether the bounds leave the type" (fun () ->
          is_true (Ops.overflows (weak_var "x" 0 200) Int8);
          is_false (Ops.overflows (weak_var "x" (-128) 127) Int8);
          is_true (Ops.overflows (weak_var "x" (-1) 3) Uint8));
    ]

(* Resolving *)

let resolving =
  group "resolve"
    [
      test "to_z, to_float and to_bool read a literal" (fun () ->
          equal z (Bigint.of_int 5) (Ops.to_z (Ops.int 5));
          equal float_exact 1.5 (Ops.to_float (Ops.float 1.5));
          is_true (Ops.to_bool (Ops.bool true)));
      test "to_z reads a typed constant and an integer sum of constants"
        (fun () ->
          equal z (Bigint.of_int 4) (Ops.to_z (Ops.int ~dtype:Int32 4));
          equal z (Bigint.of_int 11)
            (Ops.to_z Ops.O.(Ops.int ~dtype:Int32 4 + int 7));
          equal z (Bigint.of_int 2)
            (Ops.to_z Ops.O.(int 8 // Ops.int ~dtype:Int32 4)));
      test "to_bool decides comparisons of constants" (fun () ->
          is_true (Ops.to_bool Ops.O.(int 4 < int 7));
          is_true (Ops.to_bool Ops.O.(int 4 <= int 4));
          is_true (Ops.to_bool Ops.O.(int 4 <> int 7));
          is_false (Ops.to_bool Ops.O.(int 4 <> int 4));
          is_false (Ops.to_bool Ops.O.(int 4 > int 7)));
      test "to_bool decides a comparison the bounds decide" (fun () ->
          let v = weak_var "i" 1 10 in
          is_true (Ops.to_bool Ops.O.(v < int 20));
          is_true (Ops.to_bool Ops.O.(v // int 2 < int 20));
          is_false (Ops.to_bool Ops.O.(v < int 1));
          is_false (Ops.to_bool Ops.O.(v > int 11));
          let x = weak_var "x" 1 10 and y = weak_var "y" 5 10 in
          is_true (Ops.to_bool Ops.O.(Ops.maximum x y < int 20));
          is_false (Ops.to_bool Ops.O.(Ops.maximum x y < int 3)));
      test
        "to_bool decides a disjunction with true and a conjunction with false"
        (fun () ->
          is_true (Ops.to_bool Ops.O.(flag "b" lor bool true));
          is_false (Ops.to_bool Ops.O.(flag "b" land bool false)));
      test "to_bool rejects a condition with two possible values" (fun () ->
          let v = weak_var "i" 1 10 in
          rejects (fun () -> Ops.to_bool Ops.O.(flag "b" lor bool false));
          rejects (fun () -> Ops.to_bool Ops.O.(flag "b" land bool true));
          rejects (fun () -> Ops.to_bool Ops.O.(v < v + int 1));
          rejects (fun () -> Ops.to_bool Ops.O.((v > int 4) lor (v < int 6)));
          rejects (fun () -> Ops.to_bool Ops.O.(v < int 5)));
      test "to_bool, to_z and to_float reject another type" (fun () ->
          rejects (fun () -> Ops.to_bool (Ops.int 1));
          rejects (fun () -> Ops.to_z (Ops.bool true));
          rejects (fun () -> Ops.to_z (Ops.float 1.));
          rejects (fun () -> Ops.to_float (Ops.int 1)));
      test "resolve takes the default when the comparison is undecided"
        (fun () ->
          let u = weak_var "i" 1 10 in
          is_true (Ops.resolve Ops.O.(u < int 4));
          is_false (Ops.resolve ~default:false Ops.O.(u < int 4));
          is_true (Ops.resolve ~default:false Ops.O.(u < int 11));
          is_false (Ops.resolve ~default:false Ops.O.(u < int (-1)));
          is_false (Ops.resolve ~default:true Ops.O.(u < int (-1))));
      test "resolve rejects a node that is not boolean" (fun () ->
          rejects (fun () -> Ops.resolve (Ops.int 3)));
      test
        "simplify leaves a constant, and a sink of constants and stacks of \
         constants, alone" (fun () ->
          is_true (Ops.simplify (Ops.int 3) == Ops.int 3);
          let s =
            Ops.sink
              [
                Ops.int 1;
                Ops.float 2.;
                Ops.v Op.Stack;
                Ops.v ~src:[ Ops.int 1; Ops.int 2 ] Op.Stack;
              ]
          in
          is_true (Ops.simplify s == s));
      test "ssimplify is an integer constant as an integer" (fun () ->
          equal sint (Int 3) (Ops.ssimplify (Ops.int 3)));
      test "ssimplify reads a typed integer constant" (fun () ->
          equal sint (Int 3) (Ops.ssimplify (Ops.int ~dtype:Int32 3)));
      test "smax and smin of integers are integers" (fun () ->
          equal sint (Int 5) (Ops.smax [ Int 2; Int 5 ]);
          equal sint (Int 2) (Ops.smin [ Int 5; Int 2 ]);
          equal sint (Int 7) (Ops.smax [ Int 7 ]);
          rejects (fun () -> Ops.smax []);
          rejects (fun () -> Ops.smin []));
      test "smax and smin of a symbolic size bound it as max and min do"
        (fun () ->
          let v = weak_var "v" 3 10 in
          let bounds_of = function
            | Ops.Sym u -> bounds u
            | Int n -> int_bounds n n
          in
          equal (pair value value) (int_bounds 3 10)
            (bounds_of (Ops.smax [ Int 2; Sym v ]));
          equal (pair value value) (int_bounds 5 10)
            (bounds_of (Ops.smax [ Sym v; Int 5 ]));
          equal (pair value value) (int_bounds 2 2)
            (bounds_of (Ops.smin [ Int 2; Sym v ]));
          equal (pair value value) (int_bounds 3 10)
            (bounds_of (Ops.smin [ Sym v; Int 20 ])));
    ]

let sint_module =
  let n = weak_var "n" 1 8 in
  group "Sint"
    [
      test "arithmetic on integers stays on integers" (fun () ->
          equal sint (Int 5) Ops.Sint.(Int 2 + Int 3);
          equal sint (Int (-1)) Ops.Sint.(Int 2 - Int 3);
          equal sint (Int 6) Ops.Sint.(Int 2 * Int 3);
          equal sint (Int (-4)) Ops.Sint.(Int (-7) // Int 2);
          equal sint (Int 1) Ops.Sint.(Int (-7) % Int 2);
          equal sint (Int (-1)) Ops.Sint.(Int 7 % Int (-2)));
      test "arithmetic with a node builds the node's operation" (fun () ->
          equal sint (Sym Ops.O.(n + int 1)) Ops.Sint.(Sym n + Int 1);
          equal sint (Sym Ops.O.(int 1 + n)) Ops.Sint.(Int 1 + Sym n);
          equal sint (Sym Ops.O.(n - int 1)) Ops.Sint.(Sym n - Int 1);
          equal sint (Sym Ops.O.(int 2 * n)) Ops.Sint.(Int 2 * Sym n);
          equal sint (Sym Ops.O.(n // int 2)) Ops.Sint.(Sym n // Int 2);
          equal sint (Sym Ops.O.(n % int 2)) Ops.Sint.(Sym n % Int 2));
      test "prod multiplies from 1" (fun () ->
          equal sint (Int 1) (Ops.Sint.prod []);
          equal sint (Int 6) (Ops.Sint.prod [ Int 2; Int 3 ]);
          equal sint (Sym Ops.O.(int 2 * n)) (Ops.Sint.prod [ Int 2; Sym n ]));
      test "comparisons of integers are known, of nodes conditions" (fun () ->
          is_true (Ops.Sint.(Int 2 < Int 3) = Known true);
          is_true (Ops.Sint.(Int 3 <= Int 2) = Known false);
          is_true (Ops.Sint.(Int 3 <> Int 3) = Known false);
          (match Ops.Sint.(Sym n < Int 3) with
          | Cond c -> equal uop Ops.O.(n < int 3) c
          | Known _ -> fail "a comparison with a node is a condition");
          match Ops.Sint.(Int 3 > Sym n) with
          | Cond c -> equal uop Ops.O.(n < int 3) c
          | Known _ -> fail "a comparison with a node is a condition");
      test "resolve is a known condition's value" (fun () ->
          is_true (Ops.Sint.resolve (Known true));
          is_false (Ops.Sint.resolve ~default:true (Known false)));
      test "resolve decides a node by its bounds" (fun () ->
          is_true (Ops.Sint.resolve Ops.Sint.(Sym n < Int 9));
          is_false (Ops.Sint.resolve ~default:false Ops.Sint.(Sym n < Int 5));
          is_true (Ops.Sint.resolve Ops.Sint.(Sym n <> Int 0)));
      test "equal compares integers by value and nodes by identity" (fun () ->
          is_true (Ops.Sint.equal (Int 3) (Int 3));
          is_false (Ops.Sint.equal (Int 3) (Sym (Ops.int 3)));
          is_true (Ops.Sint.equal (Sym n) (Sym (weak_var "n" 1 8))));
      test "pp formats an integer in decimal and a node as Ops.pp does"
        (fun () ->
          equal string "3" (str Ops.Sint.pp (Int 3));
          equal string (str Ops.pp n) (str Ops.Sint.pp (Sym n)));
    ]

(* Divisibility *)

let divisibility =
  let x = weak_var "x" 10 20 in
  group "divisibility"
    [
      test "const_factor is a known divisor" (fun () ->
          equal z (Bigint.of_int 42) (Ops.const_factor (Ops.int 42));
          equal z (Bigint.of_int 6) (Ops.const_factor Ops.O.(int 30 + int 12));
          equal z (Bigint.of_int 5) (Ops.const_factor Ops.O.(int 5 * int 7));
          equal z (Bigint.of_int 3) (Ops.const_factor Ops.O.(x * int 3));
          equal z Bigint.one (Ops.const_factor Ops.O.(x // int 4));
          equal z (Bigint.of_int 4)
            (Ops.const_factor (weak_var ~multiple_of:4 "x" 16 32));
          let g = Ops.special (Int 8) "gidx0" in
          equal z Bigint.one (Ops.const_factor g);
          equal z (Bigint.of_int 3) (Ops.const_factor Ops.O.(g * int 3));
          equal z (Bigint.of_int 3)
            (Ops.const_factor Ops.O.((g * int 3) + int 6));
          equal z Bigint.one (Ops.const_factor Ops.O.((g * int 3) + int 1)));
      test "divides divides a stack lane by lane" (fun () ->
          equal (option uop)
            (Some (Ops.v ~src:[ Ops.O.(x * int 1); Ops.O.(x * int 2) ] Op.Stack))
            (Ops.divides
               (Ops.stack Ops.O.[ x * int 2; x * int 4 ])
               (Bigint.of_int 2));
          is_none
            (Ops.divides
               (Ops.stack Ops.O.[ x * int 2; x * int 3 ])
               (Bigint.of_int 2)));
      test "divides divides a product through its left factor first" (fun () ->
          let m = weak_var ~multiple_of:4 "m" 0 16 in
          equal (option uop)
            (Some Ops.O.(m // int 2 * x))
            (Ops.divides Ops.O.(m * x) (Bigint.of_int 2));
          is_none (Ops.divides Ops.O.(x + int 3) (Bigint.of_int 2)));
      test "const_factor of storage without a multiple is 1" (fun () ->
          equal z Bigint.one
            (Ops.const_factor (Ops.param ~shape:(ints [ 4 ]) 0 Int32)));
      test "gcd rejects nothing" (fun () -> rejects (fun () -> Ops.gcd []));
      test "gcd of constants is their gcd" (fun () ->
          equal uop (Ops.int 2) (Ops.gcd [ Ops.int 6; Ops.int 4 ]));
      test "const_factor of a stack is the gcd of its lanes" (fun () ->
          equal z (Bigint.of_int 2)
            (Ops.const_factor (Ops.stack Ops.O.[ x * int 2; x * int 4 ])));
      test "divides divides a known multiple" (fun () ->
          equal (option z)
            (Some (Bigint.of_int 6))
            (Option.map Ops.const_factor
               (Ops.divides (Ops.int 42) (Bigint.of_int 7)));
          is_none (Ops.divides (Ops.int 42) (Bigint.of_int 5));
          equal (option z) (Some Bigint.one)
            (Option.map Ops.const_factor
               (Ops.divides Ops.O.((x * int 6) + int 18) (Bigint.of_int 6)));
          is_none
            (Ops.divides Ops.O.(weak_var "x" 15 45 * int 4) (Bigint.of_int 3));
          is_some
            (Ops.divides (weak_var ~multiple_of:4 "x" 16 32) (Bigint.of_int 4));
          is_some
            (Ops.divides (weak_var ~multiple_of:4 "x" 16 32) (Bigint.of_int 2));
          equal (option uop) (Some x) (Ops.divides x Bigint.one));
      test "divides divides a float constant only into an integer" (fun () ->
          is_none (Ops.divides (Ops.float 2.5) (Bigint.of_int 2));
          equal (option uop)
            (Some (Ops.float 2.))
            (Ops.divides (Ops.float 4.) (Bigint.of_int 2)));
      test "a typed constant is a cast, whose divisors are not known" (fun () ->
          is_none (Ops.divides (Ops.int ~dtype:Int32 8) (Bigint.of_int 2));
          equal z Bigint.one (Ops.const_factor (Ops.int ~dtype:Int32 8)));
      test "pop_const splits off a constant operand" (fun () ->
          let e = Ops.O.(x + int 3) in
          equal (pair uop const) (x, i 3) (Ops.pop_const e);
          equal (pair uop const)
            (x, i 3)
            (Ops.pop_const ~op:Op.Mul Ops.O.(x * int 3));
          equal (pair uop const) (e, i 1) (Ops.pop_const ~op:Op.Mul e);
          equal (pair uop const) (x, i 0) (Ops.pop_const x));
      test "gcd keeps the common factors and the gcd of the coefficients"
        (fun () ->
          let y = weak_var "y" 1 4 and z' = weak_var "z" 1 4 in
          equal uop Ops.O.(int 1 * x) (Ops.gcd Ops.O.[ x * y; x * z' ]);
          equal uop (Ops.int 2) (Ops.gcd Ops.O.[ x * int 6; y * int 4 ]));
      test "divide_exact divides each term, or gives up" (fun () ->
          let y = weak_var "y" 1 4 and z' = weak_var "z" 1 4 in
          equal (option uop) (Some (Ops.int 1)) (Ops.divide_exact x x);
          equal (option uop)
            (Some Ops.O.((int 1 * y) + (int 1 * z')))
            (Ops.divide_exact Ops.O.((x * y) + (x * z')) x);
          equal (option uop)
            (Some (Ops.int 3))
            (Ops.divide_exact (Ops.int 6) (Ops.int 2));
          is_none (Ops.divide_exact Ops.O.(x + int 1) x);
          is_none
            (Ops.divide_exact
               Ops.O.(x * int 6)
               Ops.O.(weak_var "y" 1 4 * int 2));
          is_none
            (Ops.divide_exact
               (Ops.stack Ops.O.[ x * int 2; x * int 4 ])
               (Ops.consts [ i 2; i 4 ])));
    ]

(* Symbolic inference and programs *)

let alu_param ?(lo = 0) ?(hi = 8) name slot =
  Ops.param ~vmin_vmax:(i lo, i hi) ~name ~addrspace:(Some Alu) slot Int32

let inference =
  group "sym_infer"
    [
      test "an integer is itself" (fun () ->
          equal int 5 (Ops.sym_infer (Int 5) []));
      test "a node takes its variables' values" (fun () ->
          let n = weak_var "n" 0 100 in
          equal int 7
            (Ops.sym_infer (Sym Ops.O.((n * int 2) + int 1)) [ ("n", 3) ]);
          equal int 9
            (Ops.sym_infer (Sym Ops.O.(Ops.bind n (i 7) + int 1)) [ ("n", 8) ]));
      test "divisions round as their operations say" (fun () ->
          let n = weak_var "n" (-10) 10 in
          equal int (-3) (Ops.sym_infer (Sym Ops.O.(n // int 3)) [ ("n", -7) ]);
          equal int 2 (Ops.sym_infer (Sym Ops.O.(n % int 3)) [ ("n", -7) ]);
          equal int (-2)
            (Ops.sym_infer
               (Sym (Ops.alu n Op.Cdiv [ Ops.int 3 ]))
               [ ("n", -7) ]);
          equal int (-1)
            (Ops.sym_infer
               (Sym (Ops.alu n Op.Cmod [ Ops.int 3 ]))
               [ ("n", -7) ]));
      test "a cast converts without truncating to a width" (fun () ->
          let n = weak_var "n" (-1000) 1000 in
          let half = Ops.O.(Ops.cast n Float32 / float 2.) in
          equal int 7
            (Ops.sym_infer
               (Sym Ops.O.(Ops.cast half Weak_int + int 10))
               [ ("n", -7) ]);
          equal int 300
            (Ops.sym_infer
               (Sym (Ops.cast (Ops.cast n Int8) Weak_int))
               [ ("n", 300) ]));
      test "a missing variable is rejected" (fun () ->
          rejects (fun () -> Ops.sym_infer (Sym (weak_var "n" 0 4)) []));
    ]

(* sym_compile computes what sym_infer does, on expressions whose values fit
   an [int] and on those whose products do not fit one, which it computes
   exactly. *)

(* A symbolic integer over the variables [a] and [b], of depth at most
   [depth]. *)
let rec expression a b depth =
  let leaf =
    Gen.one_of
      [
        Gen.constant a;
        Gen.constant b;
        Gen.map Ops.O.int (Gen.such_that (( <> ) 0) (Gen.int_range (-20) 20));
      ]
  in
  if depth = 0 then leaf
  else
    let sub = expression a b (depth - 1) in
    let node =
      let open Gen in
      let+ op = int_range 0 8 and+ x = sub and+ y = sub in
      match op with
      | 0 -> Ops.O.(x + y)
      | 1 -> Ops.O.(x - y)
      | 2 -> Ops.O.(x * y)
      | 3 -> Ops.O.(x // y)
      | 4 -> Ops.O.(x % y)
      | 5 -> Ops.alu x Op.Cdiv [ y ]
      | 6 -> Ops.alu x Op.Cmod [ y ]
      | 7 -> Ops.alu x Op.Max [ y ]
      | _ -> Ops.O.(-x)
    in
    Gen.frequency [ (1, leaf); (3, node) ]

let outcome f = match f () with v -> Ok v | exception e -> Error e

let compiles_as_inferred ~bound =
  let a = weak_var "a" (-bound) bound and b = weak_var "b" (-bound) bound in
  let value = Gen.int_range (-bound) bound in
  let draw =
    Gen.triple
      (Gen.with_pp Ops.pp (expression a b 3))
      value value
  in
  prop
    (Printf.sprintf "computes what sym_infer does, variables within %d" bound)
    draw
    (fun (e, va, vb) ->
      let env = [ ("a", va); ("b", vb) ] in
      let var u = List.assoc (Ops.expr u) in
      let inferred = outcome (fun () -> Ops.sym_infer (Sym e) env)
      and compiled = outcome (fun () -> Ops.sym_compile (Sym e) var env) in
      let exn =
        Testable.make
          ~pp:(fun ppf e -> Format.pp_print_string ppf (Printexc.to_string e))
          ~equal:(fun e e' -> Printexc.to_string e = Printexc.to_string e')
      in
      equal (result int exn) inferred compiled)

let compilation =
  group "sym_compile"
    [
      compiles_as_inferred ~bound:50;
      compiles_as_inferred ~bound:(1 lsl 31);
      test "an integer is itself" (fun () ->
          equal int 5 (Ops.sym_compile (Int 5) (fun _ () -> 0) ()));
      test "a variable is what var reads" (fun () ->
          let n = weak_var "n" 0 100 in
          equal int 7
            (Ops.sym_compile
               (Sym Ops.O.((n * int 2) + int 1))
               (fun _ x -> x)
               3));
      test "a variable var refuses raises" (fun () ->
          rejects (fun () ->
              Ops.sym_compile
                (Sym Ops.O.(weak_var "n" 0 4 + int 1))
                (fun _ () -> invalid_arg "no n")
                ()));
    ]

let programs =
  group "programs"
    [
      test "program_info_of_sink reads launch sizes, variables and buffers"
        (fun () ->
          let core_id = alu_param ~hi:3 "core_id" 0
          and n = alu_param ~lo:2 "n" 1 in
          let input = Ops.param ~shape:(ints [ 16 ]) 2 Float32
          and output = Ops.param ~shape:(ints [ 16 ]) 3 Float32 in
          let stored =
            Ops.store (Ops.index output [ n ])
              (Ops.load (Ops.index input [ n ]) [])
          in
          let sink =
            Ops.sink
              [
                stored;
                Ops.special (Int 4) "gidx2";
                Ops.special (Int 8) "lidx1";
                core_id;
              ]
          in
          let info = Ops.program_info_of_sink sink in
          equal (list int) [ 2; 3 ] info.globals;
          equal (list int) [ 3 ] info.outs;
          equal (list int) [ 2 ] info.ins;
          equal uops [ core_id; n ] info.vars;
          equal (list int) [ 2; 6 ] (Ops.vals info [ ("n", 6); ("core_id", 2) ]);
          equal
            (pair (list int) (list int))
            ([ 1; 1; 4 ], [ 1; 8; 1 ])
            (Ops.launch_dims info []);
          is_true
            (info.target
            = Helpers.Target.
                {
                  device = "";
                  renderer = "";
                  arch = "";
                  interface = "";
                  indices = "";
                }));
      test "a load through a cast and a store into a shrink are accesses"
        (fun () ->
          let p0 = Ops.param ~shape:(ints [ 4 ]) 0 Uint8
          and p1 = Ops.param ~shape:(ints [ 4 ]) 1 Uint8 in
          let cast_read =
            Ops.load
              (Ops.v
                 ~src:[ Ops.index p0 [ Ops.int 0 ] ]
                 ~arg:(Dtype Int8) Op.Cast)
              []
          in
          let write =
            Ops.store
              (Ops.shrink p1 [ Some (Int 0, Int 2) ])
              (Ops.expand (Ops.int ~dtype:Uint8 1) (ints [ 2 ]))
          in
          let info = Ops.program_info_of_sink (Ops.sink [ write; cast_read ]) in
          equal (list int) [ 0 ] info.ins;
          equal (list int) [ 1 ] info.outs);
      test "a load through a cast of something other than an index is no access"
        (fun () ->
          let p0 = Ops.param ~shape:(ints [ 4 ]) 0 Uint8
          and p1 = Ops.param ~shape:(ints [ 4 ]) 1 Uint8 in
          let write =
            Ops.store (Ops.index p1 [ Ops.int 0 ]) (Ops.int ~dtype:Uint8 1)
          in
          let info =
            Ops.program_info_of_sink
              (Ops.sink [ write; Ops.load (Ops.cast p0 Int8) [] ])
          in
          equal (list int) [ 1 ] info.outs;
          equal (list int) [] info.ins);
      test "vals rejects a missing variable, naming it" (fun () ->
          let info =
            Ops.program_info_of_sink (Ops.sink [ alu_param "extent" 0 ])
          in
          raises_match (Exn.invalid_arg ~substring:"extent") (fun () ->
              Ops.vals info []));
      test "program_info_of_sink reads symbolic launch sizes" (fun () ->
          let core_id = alu_param ~hi:3 "core_id" 0
          and n = alu_param ~lo:2 "n" 1 in
          let input = Ops.param ~shape:(ints [ 16 ]) 2 Float32
          and output = Ops.param ~shape:(ints [ 16 ]) 3 Float32 in
          let stored =
            Ops.store (Ops.index output [ n ])
              (Ops.load (Ops.index input [ n ]) [])
          in
          let sink =
            Ops.sink
              ~kernel:(Ops.kernel_info ~name:"kernel name" ())
              [
                stored;
                Ops.special (Sym Ops.O.(n + int 1)) "gidx2";
                Ops.special (Int 8) "lidx1";
                core_id;
              ]
          in
          let info = Ops.program_info_of_sink sink in
          equal (list int) [ 2; 3 ] info.globals;
          equal (list int) [ 3 ] info.outs;
          equal (list int) [ 2 ] info.ins;
          equal uops [ core_id; n ] info.vars;
          equal (list int) [ 2; 6 ] (Ops.vals info [ ("n", 6); ("core_id", 2) ]);
          equal
            (pair (list int) (list int))
            ([ 1; 1; 7 ], [ 1; 8; 1 ])
            (Ops.launch_dims info [ ("n", 6); ("core_id", 2) ]);
          is_true
            (info.target
            = Helpers.Target.
                {
                  device = "";
                  renderer = "";
                  arch = "";
                  interface = "";
                  indices = "";
                }));
      test "every buffer reads and writes when no access says otherwise"
        (fun () ->
          let info =
            Ops.program_info_of_sink
              (Ops.sink
                 [
                   Ops.param ~shape:(ints [ 4 ]) 0 Float32;
                   Ops.param ~shape:(ints [ 4 ]) 5 Float32;
                 ])
          in
          equal (list int) [ 0; 5 ] info.globals;
          equal (list int) [ 0; 5 ] info.outs;
          equal (list int) [ 0; 5 ] info.ins;
          equal (list int) [ 1; 1; 1 ] (fst (Ops.launch_dims info [])));
      test "launch sizes divide as their operations say" (fun () ->
          let n = alu_param ~lo:(-10) ~hi:10 "n" 0 in
          let info =
            Ops.program_info_of_sink
              (Ops.sink
                 [
                   Ops.special (Sym Ops.O.(n // int 3)) "gidx0";
                   Ops.special (Sym Ops.O.(n % int 3)) "gidx1";
                 ])
          in
          equal (list int) [ -3; 2; 1 ]
            (fst (Ops.launch_dims info [ ("n", -7) ])));
      test "launch_dims rejects a missing variable" (fun () ->
          let extent = alu_param "extent" 0 in
          let info =
            Ops.program_info_of_sink
              (Ops.sink [ Ops.special (Sym extent) "gidx0" ])
          in
          raises_match (Exn.invalid_arg ~substring:"extent") (fun () ->
              Ops.vals info []);
          rejects (fun () -> Ops.launch_dims info []));
      test "program_info_of_sink records its target" (fun () ->
          let target = Result.get_ok (Helpers.Target.of_string "CPU:CLANG") in
          is_true
            ((Ops.program_info_of_sink ~target (Ops.sink [])).target = target));
    ]

let groups =
  [
    exec_alu;
    bounds_group;
    resolving;
    sint_module;
    divisibility;
    inference;
    compilation;
    programs;
  ]
