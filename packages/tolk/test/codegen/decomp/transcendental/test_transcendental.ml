open Windtrap
open Tolk
open Dtypes
module T = Transcendental

let floats =
  [
    ("half", Dtype.Float16); ("float", Dtype.Float32); ("double", Dtype.Float64);
  ]

let dtype_named name = List.assoc name floats
let rejects f = raises_match (Exn.invalid_arg ?sub:None) f

(* [x ~slot dt] is the scalar parameter [slot], an input nothing folds. *)
let x ?(slot = 0) dt = Shape.param slot dt

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
        cases
          "a sine of an angle bounded below the switch-over is its fast form"
          ~id:fst wide (fun (_, dt) ->
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
  cases "a rewritten operation computes it, rounded to its type" ~id:fst
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
        cases "a target without the operations is left none of them" ~id:fst
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
      cases "shl multiplies and shr floors a division by a power of two"
        ~id:(fun (v, n, _, _) -> Printf.sprintf "%d by %d" v n)
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
      cases "rintk rounds halves away from zero" ~id:string_of_float
        [ 0.; 5.; 5.5; 5.999; -5.; -5.5; -5.999; 0.49; -0.5 ] (fun v ->
          let expected = Float.to_int (Float.round v) in
          equal int expected (as_int (at Dtype.Float32 T.rintk v)));
      test "rintk gives the signed integer of its float's width" (fun () ->
          List.iter
            (fun (f, i) -> equal dtype i (Ops.dtype (T.rintk (x f))))
            Dtype.[ (Float16, Int16); (Float32, Int32); (Float64, Int64) ]);
      cases "pow2if is two to an integer" ~id:string_of_int
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
  cases "frexp splits a double into a mantissa in [0.5, 1) and an exponent"
    ~id:(fun (v, _, _) -> string_of_float v)
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
      cases "payne_hanek_reduction removes quarter turns"
        ~id:(fun (v, _, _) -> Printf.sprintf "%.17g" v)
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
      cases "anything to the power 0 is 1" ~id:string_of_float
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

(* Integer powers *)

(* [int_power dt b e] is [b] to the power [e] in [dt]: [e] factors of [b],
   wrapped at [dt]'s width, for [e >= 0]; [0] for a negative [e], but [1] for a
   base of [1] and [-1] or [1] by the parity of [e] for [-1]. *)
let int_power dt b e =
  let modulus = Bigint.shift_left Bigint.one (8 * Dtype.itemsize dt) in
  let rec power e =
    if e = 0 then Bigint.one
    else
      let h = power (e / 2) in
      let h = Bigint.rem (Bigint.mul h h) modulus in
      if e mod 2 = 0 then h else Bigint.rem (Bigint.mul h b) modulus
  in
  let z =
    if e >= 0 then power e
    else if Bigint.equal b Bigint.one then Bigint.one
    else if Bigint.equal b Bigint.minus_one then
      if e mod 2 = 0 then Bigint.one else Bigint.minus_one
    else Bigint.zero
  in
  (Dtype.truncate dt (`Int z) :> Dtype.const)

let int_powers =
  let integers =
    Dtype.[ Int8; Uint8; Int16; Uint16; Int32; Uint32; Int64; Uint64 ]
  in
  let law dt (b, e) =
    let width = 8 * Dtype.itemsize dt in
    if not (Dtype.is_unsigned dt) then begin
      cover "a negative exponent" (e < 0);
      cover "1 or -1 to a negative power" (e < 0 && abs b = 1)
    end;
    cover "a power that wraps" (abs b >= 2 && e >= width);
    let int n = `Int (Bigint.of_int n) in
    let actual =
      match T.xpow_int (x dt) (x ~slot:1 dt) dt with
      | Some p -> Interpreter.eval ~params:[ (0, int b); (1, int e) ] p
      | None -> failf "xpow_int refuses %a" Dtype.pp dt
    in
    equal const (int_power dt (Bigint.of_int b) e) actual
  in
  (* Bases of either sign, 1 and -1 among them, and exponents that are
     negative, small, or past the width, within the range of OCaml's int. *)
  let draws dt =
    let width = 8 * Dtype.itemsize dt in
    let unsigned = Dtype.is_unsigned dt in
    let lo, hi =
      if unsigned then (0, (1 lsl min 62 width) - 1)
      else
        let half = 1 lsl min 62 (width - 1) in
        (-half, half - 1)
    in
    let small = if unsigned then [ 0; 1; 2 ] else [ -1; 0; 1 ] in
    let negative = if unsigned then 0 else 2 in
    Gen.pair
      (Gen.frequency [ (1, Gen.of_list small); (3, Gen.int_range lo hi) ])
      (Gen.frequency
         [
           (negative, Gen.int_range (-9) (-1));
           (1, Gen.int_range 0 3);
           (2, Gen.int_range width (width + 9));
           (2, Gen.int_range 0 hi);
         ])
  in
  group "xpow_int"
    (List.map
       (fun dt ->
         prop ~count:300
           (Format.asprintf "a power of %a wraps as repeated products do"
              Dtype.pp dt)
           (draws dt) (law dt))
       integers
    @ [
        cases "refuses a type that is no committed integer"
          ~id:(Format.asprintf "%a" Dtype.pp)
          Dtype.[ Float32; Weak_int; Bool ]
          (fun dt ->
            equal bool true
              (Option.is_none (T.xpow_int (Ops.int 2) (Ops.int 3) dt)));
        (let wide = Bigint.shift_left Bigint.one 40 in
         let base =
           Ops.variable ~dtype:Dtype.Uint8 "b" (`Int Bigint.zero)
             (`Int (Bigint.of_int 255))
         in
         let w = Ops.variable "w" (`Int Bigint.zero) (`Int wide) in
         group "an exponent's bounds"
           [
             test "a weak expression no 64-bit integer holds is refused"
               (fun () ->
                 raises_match
                   (Exn.invalid_arg ~sub:"exponent past 64 bits")
                   (fun () ->
                     T.xpow_int base Ops.O.(w * w) Dtype.Uint8));
             test "a weak expression a 64-bit integer holds is computed"
               (fun () ->
                 equal bool false
                   (Option.is_none
                      (T.xpow_int base Ops.O.(w + int 3) Dtype.Uint8)));
             test "a literal exponent is computed at any size" (fun () ->
                 equal bool false
                   (Option.is_none
                      (T.xpow_int base
                         (Ops.const (`Int (Bigint.shift_left Bigint.one 70)))
                         Dtype.Uint8)));
           ]);
      ])

(* Types *)

let refused_types =
  let refuse ?(also = []) name f =
    cases
      (name ^ " refuses a node of another type")
      ~id:(Format.asprintf "%a" Dtype.pp)
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
      cases "pow2if refuses an integer of another width"
        ~id:(Format.asprintf "%a" Dtype.pp)
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
   result, counted on the ordered line of its type. [accuracy.golden] holds the
   results, mpmath's rounded once to the type, at values drawn from the bits of
   the type (every magnitude, the subnormals, the zeros, the infinities and
   NaN), from the range where the result is neither 0, 1 nor infinite, and for
   the sine near multiples of a quarter turn, where it cancels. *)

(* [rank dt v] is [v]'s place on the ordered line of [dt]'s values: [-0.] lies
   one below [0.], an infinity one past the greatest value. *)
let rank dt v =
  let w = 8 * Dtype.itemsize dt in
  let unsigned : Dtype.t =
    match w with 16 -> Uint16 | 32 -> Uint32 | _ -> Uint64
  in
  match Dtype.bitcast dt unsigned (`Float v) with
  | `Int bits ->
      let magnitude = Bigint.extract bits 0 (w - 1) in
      if Float.sign_bit v then Bigint.sub Bigint.minus_one magnitude
      else magnitude
  | _ -> invalid_arg "a float's bits"

(* [ulps dt expected actual] is the distance between two values of [dt] on its
   ordered line. A NaN is at no distance from a NaN, and at [max_int] from any
   number. *)
let ulps dt expected actual =
  if Float.is_nan expected || Float.is_nan actual then
    if Float.is_nan expected && Float.is_nan actual then 0 else max_int
  else
    let d = Bigint.abs (Bigint.sub (rank dt expected) (rank dt actual)) in
    if Bigint.fits_int d then Bigint.to_int d else max_int

let within_ulp dt f v expected =
  let actual = as_float (at dt f v) in
  at_most int ~than:1
    ~msg:(Printf.sprintf "at %h: %h, correctly rounded %h" v actual expected)
    (ulps dt expected actual)

let accuracy_of = [ ("exp2", exp2); ("log2", log2); ("sin", sin) ]

let accuracy =
  (* each function's rewrite, once for each type *)
  let rewrites =
    List.concat_map
      (fun (fname, op) ->
        List.map
          (fun (dname, dt) -> ((fname, dname), rewritten op dt))
          sixteen_up)
      accuracy_of
  in
  let row cell =
    let fname = cell "function" and dname = cell "dtype" in
    let dt = List.assoc dname sixteen_up in
    let v = as_float (value_of_cell (cell "x"))
    and expected = as_float (value_of_cell (cell "result")) in
    let rewrite = List.assoc (fname, dname) rewrites in
    within_ulp dt (fun _ -> rewrite) v expected;
    (* below 30, the fast sine too, defined in float32 and float64 *)
    if
      fname = "sin"
      && Float.abs v < 30.
      && List.mem dt Dtype.[ Float32; Float64 ]
    then within_ulp dt (T.xsin ~fast:true) v expected
  in
  group "accuracy"
    [ Golden.cases ~key:[ "function"; "dtype"; "x" ] "accuracy.golden" row ]

(* A matcher is made once for each argument: a call whose arguments equal an
   earlier one's gets its rules, and one whose arguments differ gets its own,
   whichever was made first. *)
let by_argument =
  let kept ~force ops =
    List.exists
      (fun n -> Ops.op n = Exp2)
      (Ops.toposort ~calls:Enter (rewrite ~force ops Dtype.Float32))
  in
  group "patterns by argument"
    [
      test "equal arguments give one matcher" (fun () ->
          equal bool true
            (T.patterns ~force:false (Op.Set.of_list Op.[ Exp2; Sqrt ])
            == T.patterns ~force:false (Op.Set.of_list Op.[ Sqrt; Exp2 ])));
      test "a set gets its own rules after another's" (fun () ->
          List.iter
            (fun (ops, keeps) -> equal bool keeps (kept ~force:false ops))
            Op.[ ([ Exp2 ], true); ([], false); ([ Exp2 ], true) ]);
      test "force gets its own rules after the other's" (fun () ->
          List.iter
            (fun force -> equal bool (not force) (kept ~force Op.[ Exp2 ]))
            [ false; true; false ]);
    ]

let () =
  exit
    (run "Tolk.Transcendental"
       [
         graphs;
         patterns;
         by_argument;
         recorded;
         bits;
         reductions;
         specials;
         pow_specials;
         int_powers;
         refused_types;
         accuracy;
       ])
