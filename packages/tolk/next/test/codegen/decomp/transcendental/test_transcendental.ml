open Windtrap
open Tolk_next
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

let as_float = function
  | `Float f -> f
  | v -> failf "%a is not a float" (Testable.pp value) v

let as_int = function
  | `Int n -> Z.to_int n
  | v -> failf "%a is not an integer" (Testable.pp value) v

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

let graphs_of (name, dt) =
  [
    graph_of ("xsin_" ^ name) (fun () -> Ops.sink [ T.xsin (x dt) ]);
    graph_of ("xsin_fast_" ^ name) (fun () ->
        Ops.sink [ T.xsin ~fast:true (x dt) ]);
    graph_of ("xsin_switch_over_" ^ name) (fun () ->
        Ops.sink [ T.xsin ~switch_over:100. (x dt) ]);
    graph_of ("xexp2_" ^ name) (fun () -> Ops.sink [ T.xexp2 (x dt) ]);
    graph_of ("xlog2_" ^ name) (fun () -> Ops.sink [ T.xlog2 (x dt) ]);
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
    (List.concat_map graphs_of floats
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
      ])

(* Patterns *)

let transcendentals dt =
  let d = x dt in
  Ops.sink [ Ops.exp2 d; Ops.log2 d; Ops.alu d Sin []; Ops.sqrt d ]

let rewrite ?(force = false) ops dt =
  Ops.graph_rewrite ~ctx:() (transcendentals dt)
    (T.patterns ~force (Op.Set.of_list ops))

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
                (Ops.toposort (rewrite [] dt))
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
          equal value
            (value_of_cell (cell "result"))
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
          equal value
            (value_of_cell (cell "result"))
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
                 ~params:[ (0, `Int (Z.of_int v)) ]
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
              ~params:[ (0, `Int (Z.of_int q)) ]
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
          ((12. *. Float.pi) +. 0.1, 0.1 -. (Float.pi /. 2.), 1);
          (12. *. Float.pi, 0., 4);
          ((12. *. Float.pi) -. 0.1, -0.1, 4);
        ]
        (fun (v, r, q) ->
          let rem, quadrant = T.payne_hanek_reduction (x Dtype.Float64) in
          let eval u = Interpreter.eval ~params:[ (0, `Float v) ] u in
          equal (close ~atol:1e-8 ~rtol:1e-7) r (as_float (eval rem));
          equal int q (as_int (eval quadrant)));
      test "cody_waite_reduction removes half turns" (fun () ->
          let v = (12. *. Float.pi) +. 0.1 in
          let rem, quadrant = T.cody_waite_reduction (x Dtype.Float64) in
          let eval u = Interpreter.eval ~params:[ (0, `Float v) ] u in
          equal (close ~atol:0. ~rtol:1e-7) 0.1 (as_float (eval rem));
          equal int 12 (as_int (eval quadrant)));
      test "the reductions give an Int32 quadrant" (fun () ->
          List.iter
            (fun (_, dt) ->
              equal dtype Dtype.Int32
                (Ops.dtype (snd (T.payne_hanek_reduction (x dt))));
              equal dtype Dtype.Int32
                (Ops.dtype (snd (T.cody_waite_reduction (x dt)))))
            floats);
    ]

(* Special values *)

let specials =
  let nan = Float.nan and inf = Float.infinity in
  let special name f cases =
    List.concat_map
      (fun (dname, dt) ->
        List.map
          (fun (v, expected) ->
            test (Printf.sprintf "%s %s %h is %h" name dname v expected)
              (fun () -> equal value (`Float expected) (at dt f v)))
          cases)
      floats
  in
  group "special values"
    (special "xsin" T.xsin [ (inf, nan); (-.inf, nan); (nan, nan); (0., 0.) ]
    @ special "xexp2" T.xexp2
        [
          (inf, inf);
          (-.inf, 0.);
          (nan, nan);
          (0., 1.);
          (1., 2.);
          (-1., 0.5);
          (2000., inf);
          (-2000., 0.);
        ]
    @ special "xlog2" T.xlog2
        [
          (inf, inf);
          (0., -.inf);
          (-0., -.inf);
          (-1., nan);
          (-.inf, nan);
          (nan, nan);
          (1., 0.);
          (2., 1.);
          (0.5, -1.);
        ]
    @ [
        test "xexp2 of float overflows at 128 and underflows below -149"
          (fun () ->
            equal value (`Float inf) (at Dtype.Float32 T.xexp2 128.);
            equal value
              (`Float (Float.ldexp 1. (-149)))
              (at Dtype.Float32 T.xexp2 (-149.));
            equal value (`Float 0.) (at Dtype.Float32 T.xexp2 (-151.)));
        test "xlog2 of the least subnormal float is -149" (fun () ->
            equal value (`Float (-149.))
              (at Dtype.Float32 T.xlog2 (Float.ldexp 1. (-149))));
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
  let refuse name f =
    cases
      ~name:(Format.asprintf "%a" Dtype.pp)
      (name ^ " refuses a node of another type")
      Dtype.[ Int32; Bfloat16; Fp8e4m3; Bool ]
      (fun dt -> rejects (fun () -> f (x dt)))
  in
  group "types"
    [
      refuse "xsin" (T.xsin ?fast:None ?switch_over:None);
      refuse "xsin ~fast" (T.xsin ~fast:true ?switch_over:None);
      refuse "xexp2" T.xexp2;
      refuse "xlog2" T.xlog2;
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

(* Accuracy: tinygrad's tolerances against libm, per type *)

let tolerance = function
  | Dtype.Float16 -> (1e-2, 5e-3)
  | Dtype.Float32 -> (2e-5, 1e-5)
  | _ -> (3e-2, 1e-5)

let inputs dt ~lo ~hi =
  Gen.with_pp Format.pp_print_float (Gen.map (round dt) (Gen.float_range lo hi))

(* tinygrad checks sine below 1e8. A double's sine loses about 4e-10 of absolute
   precision per unit of its argument, and passes 3e-2 near 7.7e7 (values.golden
   pins it there), so a double is drawn below 1e7. *)
let sine_bound = function Dtype.Float64 -> (-1e7, 1e7) | _ -> (-1e8, 1e8)

let accurate name f reference ~bound =
  List.map
    (fun (dname, dt) ->
      let atol, rtol = tolerance dt and lo, hi = bound dt in
      prop
        (Printf.sprintf "%s of %s is libm's within tinygrad's tolerance" name
           dname) (inputs dt ~lo ~hi) (fun v ->
          equal (close ~atol ~rtol)
            (round dt (reference v))
            (as_float (at dt f v))))
    floats

let accuracy =
  group "accuracy"
    (accurate "xexp2" T.xexp2 Float.exp2 ~bound:(fun _ -> (-160., 130.))
    @ accurate "xlog2" T.xlog2 Float.log2 ~bound:(fun _ -> (0., 1e30))
    @ accurate "xsin" T.xsin Float.sin ~bound:sine_bound
    @ accurate "xsin ~fast" (T.xsin ~fast:true) Float.sin ~bound:(fun _ ->
        (-30., 30.)))

(* The worst cases tinygrad's fuzzer found, each within [unit] units in the last
   place of 1.0. *)

let ulp_of_one = function
  | Dtype.Float16 -> Float.ldexp 1. (-10)
  | Dtype.Float32 -> Float.ldexp 1. (-23)
  | _ -> Float.epsilon

let least_normal = function
  | Dtype.Float16 -> Float.ldexp 1. (-14)
  | Dtype.Float32 -> Float.ldexp 1. (-126)
  | _ -> Float.ldexp 1. (-1022)

let within name f reference cases =
  List.concat_map
    (fun (dname, dt) ->
      List.map
        (fun (v, unit) ->
          test (Printf.sprintf "%s of %s %g" name dname v) (fun () ->
              let v = round dt v in
              equal
                (close ~atol:(unit *. ulp_of_one dt) ~rtol:1e-5)
                (round dt (reference v))
                (as_float (at dt f v))))
        (cases dt))
    floats

let fuzzed =
  group "fuzzer cases"
    (within "xsin" T.xsin Float.sin (fun _ ->
         [
           (-35., 1.);
           (-25., 1.);
           (25., 1.);
           (30., 1.);
           (35., 1.);
           (0., 1.);
           (Float.pi /. 2., 1.);
           (Float.pi *. 2., 1.5);
         ])
    @ within "xlog2" T.xlog2 Float.log2 (fun dt ->
        List.concat_map
          (fun scale ->
            let v = least_normal dt *. scale in
            [ (v, 1.); (-.v, 1.) ])
          [ 1.; 1e10; 1e20; 1e30 ]
        @ [ (0., 1.); (9e-7, 1.) ]))

let () =
  exit
    (run "Tolk_next.Transcendental"
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
         fuzzed;
       ])
