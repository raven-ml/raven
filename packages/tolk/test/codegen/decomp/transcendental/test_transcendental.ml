open Windtrap
open Tolk
open Dtypes
module T = Transcendental

let floats =
  [
    ("half", Dtype.Float16); ("float", Dtype.Float32); ("double", Dtype.Float64);
  ]

let dtype_named name = List.assoc name floats
let rejects f = raises_match (Exn.invalid_arg ?substring:None) f

(* [x ~slot dt] is the scalar parameter [slot], an input nothing folds. *)
let x ?(slot = 0) dt = Ops.param slot dt

(* [at dt f v] is the value of [f] applied to a [dt] input holding [v]. *)
let at dt f v = Interpreter.eval ~params:[ (0, `Float v) ] (f (x dt))

let as_float v =
  match (v :> Dtype.const) with
  | `Float f -> f
  | v -> failf "%a is not a float" (Testable.pp const) v

let as_int v =
  match (v :> Dtype.const) with
  | `Int n -> Bigint.to_int n
  | v -> failf "%a is not an integer" (Testable.pp const) v

let round dt f = as_float (Dtype.truncate dt (`Float f))

(* [close ~atol ~rtol] is numpy's [assert_allclose]: [|a - e| <= atol + rtol *
   |e|], with NaN equal to NaN and each infinity only to itself. *)
let close ~atol ~rtol =
  Testable.make ~pp:Format.pp_print_float ~equal:(fun expected actual ->
      if Float.is_nan expected || Float.is_nan actual then
        Float.is_nan expected && Float.is_nan actual
      else if Float.is_finite expected && Float.is_finite actual then
        Float.abs (actual -. expected) <= atol +. (rtol *. Float.abs expected)
      else expected = actual)

(* Graphs *)

let graph_of name f = Golden.graph (name ^ ".golden") f
let sink_of (a, b) = Ops.sink [ a; b ]

(* The functions are defined in float32 and float64 only. *)
let wide = List.filter (fun (_, dt) -> dt <> Dtype.Float16) floats

let functions_of (name, dt) =
  [
    graph_of ("xsin_" ^ name) (fun () -> Ops.sink [ T.xsin (x dt) ]);
    graph_of ("xsin_fast_" ^ name) (fun () ->
        Ops.sink [ T.xsin ~fast:true (x dt) ]);
    graph_of ("xsin_switch_over_" ^ name) (fun () ->
        Ops.sink [ T.xsin ~switch_over:100. (x dt) ]);
    graph_of ("xexp2_" ^ name) (fun () -> Ops.sink [ T.xexp2 (x dt) ]);
    graph_of ("xlog2_" ^ name) (fun () -> Ops.sink [ T.xlog2 (x dt) ]);
  ]

let graphs_of (name, dt) =
  [
    graph_of ("xpow_" ^ name) (fun () ->
        Ops.sink [ T.xpow (x dt) (x ~slot:1 dt) ]);
    graph_of ("frexp_" ^ name) (fun () -> sink_of (T.frexp (x dt)));
    graph_of ("payne_hanek_" ^ name) (fun () ->
        sink_of (T.payne_hanek_reduction (x dt)));
    graph_of ("cody_waite_" ^ name) (fun () ->
        sink_of (T.cody_waite_reduction (x dt)));
    graph_of ("rintk_" ^ name) (fun () -> Ops.sink [ T.rintk (x dt) ]);
  ]

let graphs =
  group "graphs"
    (List.concat_map functions_of wide
    @ List.concat_map graphs_of floats
    @ [
        graph_of "pow2if_short" (fun () ->
            Ops.sink [ T.pow2if (x Dtype.Int16) Dtype.Float16 ]);
        graph_of "pow2if_int" (fun () ->
            Ops.sink [ T.pow2if (x Dtype.Int32) Dtype.Float32 ]);
        graph_of "pow2if_long" (fun () ->
            Ops.sink [ T.pow2if (x Dtype.Int64) Dtype.Float64 ]);
        graph_of "shifts" (fun () ->
            let i = x Dtype.Int32 and f = x ~slot:1 Dtype.Float32 in
            Ops.sink
              [
                T.shl i 3; T.shr i 3; T.shl i 0; T.shr i 0; T.shl f 2; T.shr f 2;
              ]);
        cases ~name:fst
          "a sine of an angle bounded below the switch-over is its fast form"
          wide (fun (_, dt) ->
            let angle lo hi =
              Ops.variable ~dtype:dt "a" (`Float lo) (`Float hi)
            in
            let bounded = angle (-29.) 29. and wide = angle (-31.) 31. in
            is_true (Ops.equal (T.xsin bounded) (T.xsin ~fast:true bounded));
            is_false (Ops.equal (T.xsin wide) (T.xsin ~fast:true wide)));
      ])

(* Patterns *)

let transcendentals dt =
  let d = x dt in
  Ops.sink [ Ops.exp2 d; Ops.log2 d; Ops.alu d Sin []; Ops.sqrt d ]

let rewrite ?(force = false) ops dt =
  Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() (transcendentals dt)
    (After_sources (T.patterns ~force (Op.Set.of_list ops)))

let all_ops = Op.[ Exp2; Log2; Sin; Sqrt ]

let every_float =
  floats
  @ [
      ("bfloat16", Dtype.Bfloat16);
      ("fp8e4m3", Dtype.Fp8e4m3);
      ("fp8e5m2fnuz", Dtype.Fp8e5m2fnuz);
    ]

(* The tolerance of a rewritten operation: one unit in the last place of 1.0 for
   the narrow floats, which are computed in Float32 and rounded once, and
   tinygrad's relative tolerance for the others. *)
let precision = function
  | Dtype.Bfloat16 -> Float.ldexp 1. (-7)
  | Dtype.Fp8e4m3 -> Float.ldexp 1. (-3)
  | Dtype.Fp8e5m2fnuz -> Float.ldexp 1. (-2)
  | Dtype.Float16 -> 5e-3
  | _ -> 1e-5

let rewritten_values =
  let references = [ Float.exp2; Float.log2; Float.sin; Float.sqrt ] in
  cases ~name:fst "a rewritten operation computes it, rounded to its type"
    every_float (fun (_, dt) ->
      let results = Ops.src (rewrite [] dt) in
      List.iter
        (fun v ->
          let v = round dt v in
          List.iter2
            (fun reference u ->
              let expected = round dt (reference v) in
              let actual =
                as_float (Interpreter.eval ~params:[ (0, `Float v) ] u)
              in
              equal ~msg:(Printf.sprintf "at %h" v)
                (close ~atol:(precision dt) ~rtol:(precision dt))
                expected actual)
            references results)
        [ 0.5; 1.; 1.5; 3.; 10.; 0.1 ])

let patterns =
  group "patterns"
    (List.map
       (fun (name, dt) ->
         graph_of ("patterns_none_" ^ name) (fun () -> rewrite [] dt))
       every_float
    @ [
        graph_of "patterns_all_float" (fun () -> rewrite all_ops Dtype.Float32);
        graph_of "patterns_all_forced_float" (fun () ->
            rewrite ~force:true all_ops Dtype.Float32);
        graph_of "patterns_exp2_log2_float" (fun () ->
            rewrite Op.[ Exp2; Log2 ] Dtype.Float32);
        graph_of "patterns_sqrt_bfloat16" (fun () ->
            rewrite Op.[ Exp2; Log2; Sin ] Dtype.Bfloat16);
        test "a target with every operation keeps the graph" (fun () ->
            let sink = transcendentals Dtype.Float32 in
            is_true (Ops.equal sink (rewrite all_ops Dtype.Float32)));
        cases ~name:fst "a target without the operations is left none of them"
          every_float (fun (_, dt) ->
            let left =
              List.filter
                (fun u -> List.mem (Ops.op u) all_ops)
                (Ops.toposort ~calls:Enter (rewrite [] dt))
            in
            equal int 0 (List.length left));
        rewritten_values;
      ])

(* Values recorded from tinygrad's graphs *)

let unary = function
  | "xsin" -> T.xsin ?fast:None ?switch_over:None
  | "xsin_fast" -> T.xsin ~fast:true ?switch_over:None
  | "xexp2" -> T.xexp2
  | "xlog2" -> T.xlog2
  | f -> invalid_arg f

let recorded =
  group "values"
    [
      Golden.cases ~key:[ "function"; "dtype"; "x" ] "values.golden"
        (fun cell ->
          let dt = dtype_named (cell "dtype") in
          equal const
            (const_of_cell (cell "result"))
            (at dt
               (unary (cell "function"))
               (as_float (value_of_cell (cell "x")))));
      Golden.cases ~key:[ "dtype"; "base"; "exponent" ] "pow_values.golden"
        (fun cell ->
          let dt = dtype_named (cell "dtype") in
          let params =
            [
              (0, value_of_cell (cell "base"));
              (1, value_of_cell (cell "exponent"));
            ]
          in
          equal const
            (const_of_cell (cell "result"))
            (Interpreter.eval ~params (T.xpow (x dt) (x ~slot:1 dt))));
    ]

(* Bits *)

let bits =
  group "bits"
    [
      Golden.cases "exponent_biases.golden" (fun cell ->
          let dt = dtype_of_cell (cell "dtype") in
          match cell "exponent_bias" with
          | s when String.starts_with ~prefix:"raises" s ->
              rejects (fun () -> T.exponent_bias dt)
          | s -> equal int (int_of_string s) (T.exponent_bias dt));
      cases
        ~name:(fun (v, n, _, _) -> Printf.sprintf "%d by %d" v n)
        "shl multiplies and shr floors a division by a power of two"
        [
          (5, 0, 5, 5);
          (5, 1, 10, 2);
          (-5, 1, -10, -3);
          (-1, 3, -8, -1);
          (7, 3, 56, 0);
        ]
        (fun (v, n, left, right) ->
          let eval f =
            as_int
              (Interpreter.eval
                 ~params:[ (0, `Int (Bigint.of_int v)) ]
                 (f (x Dtype.Int32) n))
          in
          equal int left (eval T.shl);
          equal int right (eval T.shr));
      test "shl and shr refuse a negative count" (fun () ->
          rejects (fun () -> T.shl (x Dtype.Int32) (-1));
          rejects (fun () -> T.shr (x Dtype.Int32) (-1)));
      cases ~name:string_of_float "rintk rounds halves away from zero"
        [ 0.; 5.; 5.5; 5.999; -5.; -5.5; -5.999; 0.49; -0.5 ] (fun v ->
          let expected = Float.to_int (Float.round v) in
          equal int expected (as_int (at Dtype.Float32 T.rintk v)));
      test "rintk gives the signed integer of its float's width" (fun () ->
          List.iter
            (fun (f, i) -> equal dtype i (Ops.dtype (T.rintk (x f))))
            Dtype.[ (Float16, Int16); (Float32, Int32); (Float64, Int64) ]);
      cases ~name:string_of_int "pow2if is two to an integer"
        [ 0; 1; 2; 10; 63; -1; -2; -10; -63 ] (fun q ->
          let v =
            Interpreter.eval
              ~params:[ (0, `Int (Bigint.of_int q)) ]
              (T.pow2if (x Dtype.Int32) Dtype.Float32)
          in
          equal float_exact (Float.ldexp 1. q) (as_float v));
      test "pow2if gives the float of its integer's width" (fun () ->
          List.iter
            (fun (i, f) ->
              equal dtype f (Ops.dtype (T.pow2if (x i) Dtype.Float16)))
            Dtype.[ (Int16, Float16); (Int32, Float32); (Int64, Float64) ]);
    ]

(* frexp's cases are tinygrad's, on doubles: the sign of the mantissa is dropped
   there, since the mask of a double's fraction leaves out its sign. *)
let frexp_cases =
  cases
    ~name:(fun (v, _, _) -> string_of_float v)
    "frexp splits a double into a mantissa in [0.5, 1) and an exponent"
    [
      (1., 0.5, 1);
      (-1., 0.5, 1);
      (2., 0.5, 2);
      (-2., 0.5, 2);
      (5., 0.625, 3);
      (1000., 0.9765625, 10);
    ]
    (fun (v, m, e) ->
      let mantissa, exponent = T.frexp (x Dtype.Float64) in
      let eval u = Interpreter.eval ~params:[ (0, `Float v) ] u in
      equal float_exact m (as_float (eval mantissa));
      equal int e (as_int (eval exponent)))

let reductions =
  group "reductions"
    [
      frexp_cases;
      test "frexp keeps the sign of a float's mantissa" (fun () ->
          let mantissa, _ = T.frexp (x Dtype.Float32) in
          equal float_exact (-0.625)
            (as_float (Interpreter.eval ~params:[ (0, `Float (-5.)) ] mantissa)));
      cases
        ~name:(fun (v, _, _) -> Printf.sprintf "%.17g" v)
        "payne_hanek_reduction removes quarter turns"
        [
          ((12. *. Float.pi) +. 0.1, 0.1, 0);
          (12. *. Float.pi, 0., 4);
          ((12. *. Float.pi) -. 0.1, -0.1, 4);
        ]
        (fun (v, r, q) ->
          let rem, quadrant = T.payne_hanek_reduction (x Dtype.Float64) in
          let eval u = Interpreter.eval ~params:[ (0, `Float v) ] u in
          equal (close ~atol:1e-8 ~rtol:1e-7) r (as_float (eval rem));
          equal int q (as_int (eval quadrant)));
      (* The remainder at an angle near a multiple of pi/2, against pi to 1400
         bits: within two ulps of the reference, with its quadrant, for a
         float64's every exponent. *)
      Golden.cases ~key:[ "dtype"; "x" ] "payne_hanek_near_multiples.golden"
        (fun cell ->
          let dt = dtype_named (cell "dtype") in
          let rem, quadrant = T.payne_hanek_reduction (x dt) in
          let eval u =
            Interpreter.eval ~params:[ (0, value_of_cell (cell "x")) ] u
          in
          let rtol = if dt = Float64 then 0x1p-51 else 0x1p-22 in
          equal int
            (int_of_string (cell "quadrant"))
            (as_int (eval quadrant) land 3);
          equal (close ~atol:0. ~rtol)
            (as_float (value_of_cell (cell "r")))
            (as_float (eval rem)));
      test "cody_waite_reduction removes quarter turns" (fun () ->
          let v = (12. *. Float.pi) +. 0.1 in
          let rem, quadrant = T.cody_waite_reduction (x Dtype.Float64) in
          let eval u = Interpreter.eval ~params:[ (0, `Float v) ] u in
          equal (close ~atol:0. ~rtol:1e-7) 0.1 (as_float (eval rem));
          equal int 24 (as_int (eval quadrant)));
      test "the reductions give an Int32 quadrant" (fun () ->
          List.iter
            (fun (_, dt) ->
              equal dtype Dtype.Int32
                (Ops.dtype (snd (T.payne_hanek_reduction (x dt))));
              equal dtype Dtype.Int32
                (Ops.dtype (snd (T.cody_waite_reduction (x dt)))))
            floats);
    ]

(* Special values

   Exact, through the rewrite of each operation, in every float the patterns
   rewrite: IEEE's values at infinities, NaN and zeros, a zero's sign kept by
   the sine, and the sine of a subnormal, which rounds to the subnormal. *)

let sixteen_up =
  [
    ("half", Dtype.Float16);
    ("bfloat16", Dtype.Bfloat16);
    ("float", Dtype.Float32);
    ("double", Dtype.Float64);
  ]

(* [rewritten op dt] is [op] of a [dt] input, rewritten by the patterns. *)
let rewritten op dt =
  let none = Op.Set.of_list [] in
  match
    Ops.src
      (Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:()
         (Ops.sink [ op (x dt) ])
         (After_sources (T.patterns ~force:false none)))
  with
  | [ u ] -> u
  | _ -> invalid_arg "a sink of one node"

let exp2 = Ops.exp2
let log2 = Ops.log2
let sin d = Ops.alu d Sin []
let tiny dt = Float.ldexp 1. (1 - T.exponent_bias dt - snd (Dtype.finfo dt))

let greatest dt =
  Float.ldexp
    (2. -. Float.ldexp 1. (-snd (Dtype.finfo dt)))
    (T.exponent_bias dt)

let specials =
  let nan = Float.nan and inf = Float.infinity in
  let special name op cases =
    List.concat_map
      (fun (dname, dt) ->
        List.map
          (fun (v, expected) ->
            let v = v dt and expected = expected dt in
            test (Printf.sprintf "%s %s %h is %h" name dname v expected)
              (fun () ->
                equal const (`Float expected)
                  (at dt (fun _ -> rewritten op dt) v)))
          cases)
      sixteen_up
  in
  let c v _ = v in
  group "special values"
    (special "sin" sin
       [
         (c inf, c nan);
         (c (-.inf), c nan);
         (c nan, c nan);
         (c 0., c 0.);
         (c (-0.), c (-0.));
         (tiny, tiny);
         ((fun dt -> -.tiny dt), fun dt -> -.tiny dt);
       ]
    @ special "exp2" exp2
        [
          (c inf, c inf);
          (c (-.inf), c 0.);
          (c nan, c nan);
          (c 0., c 1.);
          (c (-0.), c 1.);
          (c 1., c 2.);
          (c (-1.), c 0.5);
          (greatest, c inf);
          ((fun dt -> -.greatest dt), c 0.);
        ]
    @ special "log2" log2
        [
          (c inf, c inf);
          (c 0., c (-.inf));
          (c (-0.), c (-.inf));
          (c (-1.), c nan);
          (c (-.inf), c nan);
          (c nan, c nan);
          (c 1., c 0.);
          (c 2., c 1.);
          (c 0.5, c (-1.));
          ((fun dt -> -.tiny dt), c nan);
        ]
    @ [
        test "xexp2 of float overflows at 128 and underflows below -149"
          (fun () ->
            equal const (`Float inf) (at Dtype.Float32 T.xexp2 128.);
            equal const
              (`Float (Float.ldexp 1. (-149)))
              (at Dtype.Float32 T.xexp2 (-149.));
            equal const (`Float 0.) (at Dtype.Float32 T.xexp2 (-151.)));
        test "xlog2 of the least subnormal float is -149" (fun () ->
            equal const (`Float (-149.))
              (at Dtype.Float32 T.xlog2 (Float.ldexp 1. (-149))));
        test
          "xlog2 of a negative number whose reciprocal overflows is NaN, and \
           of -0. is -inf" (fun () ->
            equal const (`Float nan)
              (at Dtype.Float32 T.xlog2 (-.Float.ldexp 1. (-130)));
            equal const (`Float nan)
              (at Dtype.Float64 T.xlog2 (-.Float.ldexp 1. (-1030)));
            equal const (`Float (-.inf)) (at Dtype.Float32 T.xlog2 (-0.));
            equal const (`Float (-.inf)) (at Dtype.Float64 T.xlog2 (-0.)));
      ])

let pow_specials =
  let pow b e =
    as_float
      (Interpreter.eval
         ~params:[ (0, `Float b); (1, `Float e) ]
         (T.xpow (x Dtype.Float32) (x ~slot:1 Dtype.Float32)))
  in
  group "xpow"
    [
      cases ~name:string_of_float "anything to the power 0 is 1"
        [ 0.; -0.; 1.; -2.; 0.5; Float.infinity; Float.neg_infinity; Float.nan ]
        (fun b -> equal float_exact 1. (pow b 0.));
      test "a negative base to a non-integer power is NaN" (fun () ->
          equal float_exact Float.nan (pow (-2.) 0.5);
          equal float_exact Float.nan (pow (-8.) (-1.5)));
      test "negative infinity to a non-integer power is not NaN" (fun () ->
          equal float_exact Float.infinity (pow Float.neg_infinity 0.5));
      test "a negative base's sign follows the exponent's parity" (fun () ->
          equal float_exact (-8.) (pow (-2.) 3.);
          equal float_exact 4. (pow (-2.) 2.);
          equal float_exact (-0.5) (pow (-2.) (-1.)));
    ]

(* Types *)

let refused_types =
  let refuse ?(also = []) name f =
    cases
      ~name:(Format.asprintf "%a" Dtype.pp)
      (name ^ " refuses a node of another type")
      (also @ Dtype.[ Int32; Bfloat16; Fp8e4m3; Bool ])
      (fun dt -> rejects (fun () -> f (x dt)))
  in
  let float16 = [ Dtype.Float16 ] in
  group "types"
    [
      refuse ~also:float16 "xsin" (T.xsin ?fast:None ?switch_over:None);
      refuse ~also:float16 "xsin ~fast" (T.xsin ~fast:true ?switch_over:None);
      refuse ~also:float16 "xexp2" T.xexp2;
      refuse ~also:float16 "xlog2" T.xlog2;
      refuse "frexp" T.frexp;
      refuse "rintk" T.rintk;
      refuse "payne_hanek_reduction" T.payne_hanek_reduction;
      refuse "cody_waite_reduction" T.cody_waite_reduction;
      cases
        ~name:(Format.asprintf "%a" Dtype.pp)
        "pow2if refuses an integer of another width"
        Dtype.[ Int8; Uint32; Float32; Bool ]
        (fun dt -> rejects (fun () -> T.pow2if (x dt) Dtype.Float32));
      test "xpow takes the other floats" (fun () ->
          List.iter
            (fun dt ->
              equal dtype dt (Ops.dtype (T.xpow (x dt) (x ~slot:1 dt))))
            Dtype.[ Bfloat16; Fp8e4m3; Float16 ]);
    ]

(* Accuracy

   Each function is within one unit in the last place of the correctly rounded
   result ({!Exact}), counted on the ordered line of its type, over values drawn
   from the bits of the type (every magnitude, the subnormals, the zeros, the
   infinities and NaN), from the range where the result is neither 0, 1 nor
   infinite, and for the sine near multiples of a quarter turn, where it
   cancels. *)

let pp_hex ppf v = Format.fprintf ppf "%h" v

let of_bits dt =
  let gen =
    match (dt : Dtype.t) with
    | Float16 | Bfloat16 ->
        Gen.map
          (fun b -> as_float (Dtype.bitcast Uint16 dt (`Int (Bigint.of_int b))))
          (Gen.int_range 0 0xffff)
    | Float32 -> Gen.map Int32.float_of_bits Gen.int32
    | _ -> Gen.map Int64.float_of_bits Gen.int64
  in
  Gen.with_pp pp_hex gen

let between dt lo hi =
  Gen.with_pp pp_hex (Gen.map (round dt) (Gen.float_range lo hi))

(* [k pi/2] rounded to [dt], for [k] from 1 to [n]. *)
let quarter_turns dt n =
  Gen.with_pp pp_hex
    (Gen.map
       (fun k -> round dt (Float.of_int k *. Float.pi /. 2.))
       (Gen.int_range 1 n))

let draws gens = Gen.frequency (List.map (fun g -> (1, g)) gens)

(* The exponents past which [2^x] overflows and underflows [dt]. *)
let exp2_range dt =
  let bias = T.exponent_bias dt and _, mbits = Dtype.finfo dt in
  (Float.of_int (-bias - mbits - 2), Float.of_int (bias + 2))

let last_turn dt = if dt = Dtype.Float16 then 40000 else 0x3fffffff

(* tinygrad's fuzzer found these inputs, run before any drawn one. *)
let least_normal dt = Float.ldexp 1. (1 - T.exponent_bias dt)

let fuzzed_sines =
  [ -35.; -25.; 25.; 30.; 35.; 0.; Float.pi /. 2.; 2. *. Float.pi ]

let fuzzed_logarithms dt =
  List.concat_map
    (fun scale ->
      let v = least_normal dt *. scale in
      [ v; -.v ])
    [ 1.; 1e10; 1e20; 1e30 ]
  @ [ 0.; 9e-7 ]

let accuracy_of =
  [
    ( "exp2",
      exp2,
      Exact.exp2,
      (fun _ -> []),
      fun dt ->
        let lo, hi = exp2_range dt in
        [ of_bits dt; between dt lo hi; between dt (-1.) 1. ] );
    ( "log2",
      log2,
      Exact.log2,
      fuzzed_logarithms,
      fun dt -> [ of_bits dt; between dt 0.5 2.; between dt 0. 1e-30 ] );
    ( "sin",
      sin,
      Exact.sin,
      (fun _ -> fuzzed_sines),
      fun dt ->
        [
          of_bits dt;
          between dt (-30.) 30.;
          quarter_turns dt 20;
          quarter_turns dt (last_turn dt);
        ] );
  ]

let within_ulp dt f exact v =
  let expected = exact dt v and actual = as_float (at dt f v) in
  at_most int ~than:1
    ~msg:(Printf.sprintf "at %h: %h, correctly rounded %h" v actual expected)
    (Exact.ulps dt expected actual)

let accuracy =
  group "accuracy"
    (List.concat_map
       (fun (name, op, exact, examples, gens) ->
         List.map
           (fun (dname, dt) ->
             let examples = List.map (round dt) (examples dt) in
             prop ~count:500 ~examples
               (Printf.sprintf "%s of %s is within an ulp" name dname)
               (draws (gens dt))
               (within_ulp dt (fun _ -> rewritten op dt) exact))
           sixteen_up)
       accuracy_of
    @ List.map
        (fun (dname, dt) ->
          prop ~count:500
            (Printf.sprintf "xsin ~fast of %s is within an ulp below 30" dname)
            (draws [ between dt (-30.) 30.; quarter_turns dt 19 ])
            (within_ulp dt (T.xsin ~fast:true) Exact.sin))
        [ ("float", Dtype.Float32); ("double", Dtype.Float64) ])

let () =
  exit
    (run "Tolk.Transcendental"
       [
         graphs;
         patterns;
         recorded;
         bits;
         reductions;
         specials;
         pow_specials;
         refused_types;
         accuracy;
       ])
