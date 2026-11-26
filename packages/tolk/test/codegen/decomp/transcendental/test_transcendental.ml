(* Tests of Tolk.Transcendental: each value's arithmetic and the graph it
   builds, the special values and accuracy of exp2, log2 and sin, and the
   patterns that rewrite the operations into them. *)

open Windtrap
open Tolk
open Dtypes
module T = Transcendental

let rejects f = raises_match (Exn.invalid_arg ?sub:None) f

(* [x ~slot dt] is the scalar parameter [slot], an input nothing folds. *)
let x ?(slot = 0) dt = Shape.param slot dt

(* [at dt f v] is the value of [f] applied to a [dt] input holding [v]. *)
let at dt f v = Interpreter.eval ~params:[ (0, `Float v) ] (f (x dt))

let as_float v =
  match (v :> Dtype.const) with
  | `Float f -> f
  | v -> failf "%a is not a float" (Testable.pp const) v

let as_z v =
  match (v :> Dtype.const) with
  | `Int n -> n
  | v -> failf "%a is not an integer" (Testable.pp const) v

let as_int v = Bigint.to_int (as_z v)
let round dt f = as_float (Dtype.truncate dt (`Float f))

(* Floats and their bits *)

let floats = Dtype.[ Float16; Float32; Float64 ]
let wide = Dtype.[ Float32; Float64 ]

(* The names the goldens give the floats. *)
let name = function
  | Dtype.Float16 -> "half"
  | Float32 -> "float"
  | Float64 -> "double"
  | Bfloat16 -> "bfloat16"
  | dt -> alias dt

(* [named s] is the float a golden's cell [s] names. *)
let named s =
  match List.find_opt (fun dt -> name dt = s) Dtype.(Bfloat16 :: floats) with
  | Some dt -> dt
  | None -> invalid_arg (Printf.sprintf "%S names no float" s)

let width dt = Dtype.bitsize dt
let mantissa dt = snd (Dtype.finfo dt)
let greatest_exponent dt = (1 lsl fst (Dtype.finfo dt)) - 2

let uint_of dt : Dtype.t =
  match width dt with 8 -> Uint8 | 16 -> Uint16 | 32 -> Uint32 | _ -> Uint64

(* [of_bits dt z] is the float of [dt] whose bits are [z], and [bits_of dt v]
   the bits of [v]; {!Dtype.bitcast} moves a NaN's bits by hand. *)
let of_bits dt z = as_float (Dtype.bitcast (uint_of dt) dt (`Int z))
let bits_of dt v = as_z (Dtype.bitcast dt (uint_of dt) (`Float v))

(* [up dt v] and [down dt v] are the floats of [dt] after and before the
   positive [v]; [ceil_in dt v] is the least float of [dt] at or above it. *)
let up dt v = of_bits dt (Bigint.succ (bits_of dt v))
let down dt v = of_bits dt (Bigint.pred (bits_of dt v))

let ceil_in dt v =
  let r = round dt v in
  if r >= v then r else up dt r

(* [drawn dt ~exponents:(lo, hi)] draws a float of [dt], of either sign, whose
   exponent field is in [lo, hi] and whose mantissa bits are drawn alike. *)
let drawn dt ~exponents:(lo, hi) =
  let open Gen in
  let+ negative = bool and+ e = int_range lo hi and+ m = int64 in
  let m = Bigint.extract (Bigint.of_int64_unsigned m) 0 (mantissa dt) in
  let top = if negative then Bigint.one else Bigint.zero in
  let w = width dt - 1 and p = mantissa dt in
  of_bits dt Bigint.(shift_left top w + shift_left (of_int e) p + m)

let pp_float ppf (dt, v) = Format.fprintf ppf "%s %h" (name dt) v

(* [of_type dts gen] draws one of [dts] and a float [gen] draws of it. *)
let of_type dts gen =
  Gen.(
    with_pp pp_float
      (let* dt = of_list dts in
       map (fun v -> (dt, v)) (gen dt)))

(* [rank dt v] is [v]'s place on the ordered line of [dt]'s values: [-0.] lies
   one below [0.], an infinity one past the greatest value. *)
let rank dt v =
  let magnitude = Bigint.extract (bits_of dt v) 0 (width dt - 1) in
  if Float.sign_bit v then Bigint.sub Bigint.minus_one magnitude else magnitude

(* [ulps dt expected actual] is the distance between two values of [dt] on its
   ordered line. A NaN is at no distance from a NaN, and at [max_int] from any
   number. *)
let ulps dt expected actual =
  if Float.is_nan expected || Float.is_nan actual then
    if Float.is_nan expected && Float.is_nan actual then 0 else max_int
  else
    let d = Bigint.abs (Bigint.sub (rank dt expected) (rank dt actual)) in
    if Bigint.fits_int d then Bigint.to_int d else max_int

(* [within_ulp dt u v expected] checks that [u], at the input [v], is within an
   ulp of the correctly rounded [expected]. *)
let within_ulp dt u v expected =
  let actual = as_float (Interpreter.eval ~params:[ (0, `Float v) ] u) in
  at_most int ~than:1
    ~msg:(Printf.sprintf "at %h: %h, correctly rounded %h" v actual expected)
    (ulps dt expected actual)

(* Bit manipulation *)

let exponent_bias =
  group "exponent_bias"
    [
      group "is the bias its golden states"
        [
          Golden.cases "exponent_biases.golden" (fun cell ->
              let dt = dtype_of_cell (cell "dtype") in
              match cell "exponent_bias" with
              | s when String.starts_with ~prefix:"raises" s ->
                  rejects (fun () -> T.exponent_bias dt)
              | s -> equal int (int_of_string s) (T.exponent_bias dt));
        ];
    ]

(* A shift of an integer of a type by a count below its width, or of a finite
   float32 by a count below 31. *)
let shift_case =
  let open Gen in
  let* dt = of_list ~pp:Dtype.pp Dtype.[ Int32; Uint32; Int64; Float32 ] in
  let value =
    if Dtype.is_float dt then
      map (fun f -> `Float f) (drawn dt ~exponents:(0, greatest_exponent dt))
    else value_of dt
  in
  let limit = if Dtype.is_float dt then 30 else width dt - 1 in
  let+ v = value and+ n = int_range 0 limit in
  (dt, v, n)

let pp_shift_case ppf (dt, v, n) =
  Format.fprintf ppf "%a %a by %d" Dtype.pp dt Dtype.pp_const v n

let shifted f (dt, v, n) = Interpreter.eval ~params:[ (0, v) ] (f (x dt) n)
let int32s = List.map (fun (v, n) -> (Dtype.Int32, `Int (Bigint.of_int v), n))

let shl =
  group "shl"
    [
      prop "shl x n is x * 2^n, an integer's wrapped at its width"
        ~examples:
          (int32s
             [ (5, 0); (5, 1); (-5, 1); (-1, 3); (7, 3); (1, 31); (-1, 31) ])
        (Gen.with_pp pp_shift_case shift_case)
        (fun ((dt, v, n) as c) ->
          cover (alias dt) true;
          let expected =
            match v with
            | `Float f -> `Float (round dt (Float.ldexp f n))
            | v -> Dtype.truncate dt (`Int Bigint.(shift_left (as_z v) n))
          in
          equal const (expected :> Dtype.const) (shifted T.shl c));
      test "shl raises on a negative count" (fun () ->
          rejects (fun () -> T.shl (x Dtype.Int32) (-1)));
      group "builds the graphs its golden states, with shr's"
        [
          Golden.graph "shifts.golden" (fun () ->
              let i = x Dtype.Int32 and f = x ~slot:1 Dtype.Float32 in
              Ops.sink
                [
                  T.shl i 3;
                  T.shr i 3;
                  T.shl i 0;
                  T.shr i 0;
                  T.shl f 2;
                  T.shr f 2;
                ]);
        ];
    ]

let shr =
  group "shr"
    [
      (* At int32's width less one the divisor is -2^31, so its least value
         gives 1; a float32 quotient of -2^-150 rounds to -0., which floors to
         -0. *)
      prop
        "shr x n is x // 2^n, the divisor read in x's type and a float's \
         quotient rounded before the floor"
        ~examples:
          ((Dtype.Float32, `Float (-.Float.ldexp 1. (-149)), 1)
          :: int32s
               [
                 (5, 0);
                 (5, 1);
                 (-5, 1);
                 (-1, 3);
                 (7, 3);
                 (-7, 3);
                 (-2147483648, 31);
                 (5, 31);
               ])
        (Gen.with_pp pp_shift_case shift_case)
        (fun ((dt, v, n) as c) ->
          cover (alias dt) true;
          let expected =
            match v with
            | `Float f -> `Float (Float.floor (round dt (Float.ldexp f (-n))))
            | v ->
                let d =
                  Dtype.truncate dt (`Int (Bigint.shift_left Bigint.one n))
                in
                `Int (Bigint.fdiv (as_z v) (as_z d))
          in
          equal const expected (shifted T.shr c));
      test "shr raises on a negative count" (fun () ->
          rejects (fun () -> T.shr (x Dtype.Int32) (-1)));
    ]

let rintk =
  (* from 0.25 to below the greatest value of the signed integer of the float's
     width *)
  let roundable dt =
    let bias = T.exponent_bias dt in
    drawn dt ~exponents:(bias - 2, bias + width dt - 2)
  in
  (* tinygrad's values, halves, and where the sum rounds: the float below a
     half, which gives 1, and odd integers from 2^(p-1), which give the even
     integer away from zero. *)
  let examples =
    List.concat_map
      (fun dt ->
        let p = mantissa dt + 1 in
        let odd = Float.ldexp 1. (p - 1) +. 1. in
        List.map
          (fun v -> (dt, v))
          [
            0.;
            -0.;
            5.;
            -5.;
            5.5;
            -5.5;
            5.999;
            -5.999;
            0.49;
            -0.5;
            1.5;
            2.5;
            down dt 0.5;
            -.down dt 0.5;
            odd;
            -.odd;
            odd +. 2.;
          ])
      floats
  in
  group "rintk"
    [
      prop
        "rintk d is d plus a half of its sign, rounded in d's type, truncated"
        ~examples (of_type floats roundable) (fun (dt, v) ->
          let v = round dt v in
          let sum = round dt (v +. Float.copy_sign 0.5 v) in
          cover (name dt) true;
          cover "the sum rounds" (sum -. v <> Float.copy_sign 0.5 v);
          cover "the nearest integer, halves away from zero"
            (Float.trunc sum = Float.round v);
          equal const
            (`Int (Bigint.of_float (Float.trunc sum)))
            (at dt T.rintk v));
      test "rintk gives the signed integer of its float's width" (fun () ->
          equal (list dtype)
            Dtype.[ Int16; Int32; Int64 ]
            (List.map (fun f -> Ops.dtype (T.rintk (x f))) floats));
      group "builds the graph its golden states"
        (List.map
           (fun dt ->
             Golden.graph
               ("rintk_" ^ name dt ^ ".golden")
               (fun () -> Ops.sink [ T.rintk (x dt) ]))
           floats);
    ]

let integer_of = function
  | Dtype.Float16 -> Dtype.Int16
  | Float32 -> Int32
  | _ -> Int64

(* Every normal exponent of each float. *)
let normal_exponents =
  List.concat_map
    (fun f ->
      let bias = T.exponent_bias f in
      List.init (greatest_exponent f) (fun k -> (f, k + 1 - bias)))
    floats

let pow2if =
  group "pow2if"
    [
      cases "pow2if q is 2^q at every normal exponent"
        ~id:(fun (f, q) -> Printf.sprintf "%s %d" (name f) q)
        normal_exponents
        (fun (f, q) ->
          equal const
            (`Float (Float.ldexp 1. q))
            (Interpreter.eval
               ~params:[ (0, `Int (Bigint.of_int q)) ]
               (T.pow2if (x (integer_of f)) f)));
      test "pow2if gives the float of its integer's width" (fun () ->
          equal (list dtype) floats
            (List.map
               (fun f -> Ops.dtype (T.pow2if (x (integer_of f)) Dtype.Float16))
               floats));
      group "builds the graph its golden states"
        (List.map
           (fun (i, f, n) ->
             Golden.graph
               ("pow2if_" ^ n ^ ".golden")
               (fun () -> Ops.sink [ T.pow2if (x i) f ]))
           Dtype.
             [
               (Int16, Float16, "short");
               (Int32, Float32, "int");
               (Int64, Float64, "long");
             ]);
    ]

(* [signed dt e] is [e], an integer of [dt]'s unsigned width, read as the signed
   value it wraps. *)
let signed dt e =
  let w = width dt in
  if Bigint.geq e (Bigint.shift_left Bigint.one (w - 1)) then
    Bigint.to_int (Bigint.sub e (Bigint.shift_left Bigint.one w))
  else Bigint.to_int e

let frexp =
  let normal dt = drawn dt ~exponents:(1, greatest_exponent dt) in
  let least_normal dt = Float.ldexp 1. (1 - T.exponent_bias dt) in
  group "frexp"
    [
      (* tinygrad's cases are the doubles 1, -1, 2, -2, 5 and 1000 *)
      prop
        "frexp v is (m, e), |v| = |m| * 2^e and 0.5 <= |m| < 1, m signed as v \
         but in Float64"
        ~examples:
          (List.map
             (fun v -> (Dtype.Float64, v))
             [ 1.; -1.; 2.; -2.; 5.; 1000. ]
          @ List.concat_map
              (fun dt ->
                List.map
                  (fun v -> (dt, v))
                  [
                    -5.;
                    0.3;
                    least_normal dt;
                    -.least_normal dt;
                    -.down dt Float.infinity;
                  ])
              floats)
        (of_type floats normal)
        (fun (dt, v) ->
          let v = round dt v in
          let m, e = T.frexp (x dt) in
          let eval u = Interpreter.eval ~params:[ (0, `Float v) ] u in
          let m = as_float (eval m) and e = signed dt (as_z (eval e)) in
          cover (name dt) true;
          cover "|v| < 0.5, whose exponent wraps" (Float.abs v < 0.5);
          let msg = Printf.sprintf "%h is %h * 2^%d" v m e in
          equal ~msg float_exact (Float.abs v) (Float.ldexp (Float.abs m) e);
          at_least ~msg float_exact ~than:0.5 (Float.abs m);
          less ~msg float_exact ~than:1. (Float.abs m);
          equal ~msg:("the sign of " ^ msg) float_exact
            (if dt <> Float64 && Float.sign_bit v then -1. else 1.)
            (Float.copy_sign 1. m));
      test "frexp's mantissa has v's type, its exponent v's unsigned width"
        (fun () ->
          equal
            (list (pair dtype dtype))
            Dtype.[ (Float16, Uint16); (Float32, Uint32); (Float64, Uint64) ]
            (List.map
               (fun dt ->
                 let m, e = T.frexp (x dt) in
                 (Ops.dtype m, Ops.dtype e))
               floats));
      group "builds the graph its golden states"
        (List.map
           (fun dt ->
             Golden.graph
               ("frexp_" ^ name dt ^ ".golden")
               (fun () ->
                 let m, e = T.frexp (x dt) in
                 Ops.sink [ m; e ]))
           floats);
    ]

(* Argument reductions *)

(* [rounds_once dt r] is [true] iff the float64 [r], an exact value rounded,
   lies more than a float64 ulp from every tie of [dt], so that [r] rounds to
   [dt] as the exact value does. *)
let rounds_once dt r =
  dt = Dtype.Float64
  ||
  let a = Float.abs r in
  let lo = round dt a in
  let lo = if lo > a then down dt lo else lo in
  let tie = (lo +. up dt lo) /. 2. in
  Float.abs (a -. tie) > Float.succ a -. a

let quarter_pi = Float.pi /. 4.

(* [around dt v] is [v] in [dt] and the floats each side of it. *)
let around dt v =
  let v = round dt v in
  [ down dt v; v; up dt v ]

let payne_hanek =
  (* angles about odd multiples of pi/4, where the remainder is about pi/4 *)
  let odd_eighths dt =
    List.concat_map
      (fun k -> around dt (float_of_int ((2 * k) + 1) *. quarter_pi))
      [ 1; 2; 3; 7; 100; 1001; 30001 ]
    |> List.filter (fun d -> d >= 1. && Float.is_finite d)
    |> List.map (fun d -> (dt, d))
  in
  let remainder dt d =
    as_float (at dt (fun d -> fst (T.payne_hanek_reduction d)) d)
  in
  group "payne_hanek_reduction"
    [
      (* The angle is 12 pi + r rounded, which moves the remainder from r by
         half an ulp of 37.8 and 12 times pi's own rounding: below 1e-14. *)
      cases "removes quarter turns from 12 pi and its neighbours"
        ~id:(fun (v, _, _) -> Printf.sprintf "%.17g" v)
        [
          ((12. *. Float.pi) +. 0.1, 0.1, 0);
          (12. *. Float.pi, 0., 4);
          ((12. *. Float.pi) -. 0.1, -0.1, 4);
        ]
        (fun (v, r, q) ->
          let rem, quadrant = T.payne_hanek_reduction (x Dtype.Float64) in
          let eval u = Interpreter.eval ~params:[ (0, `Float v) ] u in
          equal ~msg:"quadrant" int q (as_int (eval quadrant));
          equal ~msg:"remainder" (float 1e-14) r (as_float (eval rem)));
      (* The remainder at an angle near a multiple of pi/2, against the exact
         one from pi to 1400 bits, rounded to float64 and then to the row's
         type: the same as once where the float64 lies off the type's ties. *)
      group "is within an ulp of the remainder its golden states"
        [
          Golden.cases ~key:[ "dtype"; "x" ] "payne_hanek_near_multiples.golden"
            (fun cell ->
              let dt = named (cell "dtype") in
              let d = as_float (value_of_cell (cell "x")) in
              let rem, quadrant = T.payne_hanek_reduction (x dt) in
              let q = Interpreter.eval ~params:[ (0, `Float d) ] quadrant in
              equal ~msg:"quadrant" int
                (int_of_string (cell "quadrant"))
                (as_int q land 3);
              let r = as_float (value_of_cell (cell "r")) in
              satisfies ~msg:"the reference"
                ~claim:"more than a float64 ulp from a tie, so rounded once"
                float_exact (rounds_once dt) r;
              within_ulp dt rem d (round dt r));
        ];
      prop "the remainder is at most pi/4 in magnitude"
        ~examples:(List.concat_map odd_eighths floats)
        (of_type floats (fun dt ->
             let bias = T.exponent_bias dt in
             Gen.map Float.abs
               (drawn dt ~exponents:(bias, greatest_exponent dt))))
        (fun (dt, d) ->
          cover (name dt) true;
          at_most ~msg:(Printf.sprintf "at %h" d) float_exact
            ~than:(ceil_in dt quarter_pi)
            (Float.abs (remainder dt d)));
      test "the quadrant is an Int32" (fun () ->
          equal (list dtype)
            Dtype.[ Int32; Int32; Int32 ]
            (List.map
               (fun dt -> Ops.dtype (snd (T.payne_hanek_reduction (x dt))))
               floats));
      group "builds the graph its golden states"
        (List.map
           (fun dt ->
             Golden.graph
               ("payne_hanek_" ^ name dt ^ ".golden")
               (fun () ->
                 let r, q = T.payne_hanek_reduction (x dt) in
                 Ops.sink [ r; q ]))
           floats);
    ]

(* The bound of Cody-Waite's remainder at [d]: [q] is [d * 2/pi] rounded by
   rintk, the product computed in [d]'s type (float32 for a float16), so the
   exact [|d - q pi/2|] exceeds pi/4 by pi/2 times [u]: the product's error,
   half its ulp and [d] times the error of 2/pi in that type, and at the float
   below a half, which rintk rounds to 1, its distance from a half. The [r]
   compared is that remainder from four exact parts, rounded once in [d]'s type:
   their error, about [q 2^-50 pi/2], is far below half an ulp of [r], so the
   bound rounded up in [d]'s type holds it. *)
let cody_waite_bound dt d =
  let computed = if dt = Dtype.Float64 then dt else Dtype.Float32 in
  (* 2/pi rounded to float64 is 3.9357e-17 above it, and to float32 2.5683e-8
     below it, from its digits *)
  let two_over_pi = 0.636619772367581343075535053490057448 in
  let error = if dt = Dtype.Float64 then 3.9358e-17 else 2.5683e-8 in
  let product = Float.abs (round computed (d *. round computed two_over_pi)) in
  let half_ulp = (up computed product -. product) /. 2. in
  let below_half = if product = down computed 0.5 then 0.5 -. product else 0. in
  let u = half_ulp +. (Float.abs d *. error) +. below_half in
  let bound = ceil_in dt (quarter_pi +. (Float.pi /. 2. *. u)) in
  (* in float64 the bound itself is computed in float64: pi/4 is rounded down
     and the sum rounded, so it takes the float above *)
  if dt = Dtype.Float64 then up dt bound else bound

let cody_waite =
  let limit = 39800. in
  (* angles whose d * 2/pi is about a half-integer *)
  let halfway dt =
    List.concat_map
      (fun k ->
        List.concat_map
          (fun e -> around dt ((float_of_int k +. 0.5 +. e) *. Float.pi /. 2.))
          [ 0.; 1e-3; -1e-3; 2e-3; -2e-3 ])
      [ 0; 1; 2; 100; 10000; 25336 ]
    |> List.concat_map (fun d -> [ (dt, d); (dt, -.d) ])
  in
  group "cody_waite_reduction"
    [
      test "removes quarter turns from 12 pi + 0.1" (fun () ->
          let v = (12. *. Float.pi) +. 0.1 in
          let rem, quadrant = T.cody_waite_reduction (x Dtype.Float64) in
          let eval u = Interpreter.eval ~params:[ (0, `Float v) ] u in
          (* as for Payne-Hanek's 12 pi + 0.1 *)
          equal (float 1e-14) 0.1 (as_float (eval rem));
          equal int 24 (as_int (eval quadrant)));
      prop "the remainder is at most pi/4 + (pi/2) u in magnitude"
        ~examples:
          (List.concat_map halfway floats
          @ List.concat_map
              (fun dt -> [ (dt, limit); (dt, -.limit); (dt, 0.) ])
              floats)
        (of_type floats (fun dt ->
             Gen.map (round dt) (Gen.float_range (-.limit) limit)))
        (fun (dt, d) ->
          cover (name dt) true;
          let r = at dt (fun d -> fst (T.cody_waite_reduction d)) d in
          at_most ~msg:(Printf.sprintf "at %h" d) float_exact
            ~than:(cody_waite_bound dt d)
            (Float.abs (as_float r)));
      test "the quadrant is an Int32" (fun () ->
          equal (list dtype)
            Dtype.[ Int32; Int32; Int32 ]
            (List.map
               (fun dt -> Ops.dtype (snd (T.cody_waite_reduction (x dt))))
               floats));
      group "builds the graph its golden states"
        (List.map
           (fun dt ->
             Golden.graph
               ("cody_waite_" ^ name dt ^ ".golden")
               (fun () ->
                 let r, q = T.cody_waite_reduction (x dt) in
                 Ops.sink [ r; q ]))
           floats);
    ]

(* Functions *)

let graphs prefix f =
  group "builds the graph its golden states"
    (List.map
       (fun dt ->
         Golden.graph
           (prefix ^ "_" ^ name dt ^ ".golden")
           (fun () -> Ops.sink [ f (x dt) ]))
       wide)

let xsin =
  group "xsin"
    [
      cases
        "an angle whose bounds lie below the switch-over takes the fast form"
        ~id:name wide (fun dt ->
          let angle lo hi =
            Ops.variable ~dtype:dt "a" (`Float lo) (`Float hi)
          in
          let bounded = angle (-29.) 29. and unbounded = angle (-31.) 31. in
          equal ~msg:"below" Uops.uop
            (T.xsin ~fast:true bounded)
            (T.xsin bounded);
          not_equal ~msg:"across" Uops.uop
            (T.xsin ~fast:true unbounded)
            (T.xsin unbounded));
      graphs "xsin" T.xsin;
      group "builds the fast graph its golden states"
        [ graphs "xsin_fast" (T.xsin ~fast:true) ];
      group "builds the graph its golden states over 100"
        [ graphs "xsin_switch_over" (T.xsin ~switch_over:100.) ];
    ]

let xexp2 =
  group "xexp2"
    [
      cases "xexp2 of a float32 is inf from 128 and 0 below -149"
        ~id:(fun (v, _) -> Printf.sprintf "%g" v)
        [ (128., Float.infinity); (-149., Float.ldexp 1. (-149)); (-151., 0.) ]
        (fun (v, expected) ->
          equal const (`Float expected) (at Dtype.Float32 T.xexp2 v));
      graphs "xexp2" T.xexp2;
    ]

let xlog2 =
  group "xlog2"
    [
      test "xlog2 of the least subnormal float32 is -149" (fun () ->
          equal const (`Float (-149.))
            (at Dtype.Float32 T.xlog2 (Float.ldexp 1. (-149))));
      cases "xlog2 of a negative number whose reciprocal overflows is NaN"
        ~id:(fun (dt, _) -> name dt)
        Dtype.[ (Float32, -130); (Float64, -1030) ]
        (fun (dt, e) ->
          equal const (`Float Float.nan) (at dt T.xlog2 (-.Float.ldexp 1. e)));
      cases "xlog2 of -0. is -inf" ~id:name wide (fun dt ->
          equal const (`Float Float.neg_infinity) (at dt T.xlog2 (-0.)));
      graphs "xlog2" T.xlog2;
    ]

let unary = function
  | "xsin" -> T.xsin ?fast:None ?switch_over:None
  | "xsin_fast" -> T.xsin ~fast:true ?switch_over:None
  | "xexp2" -> T.xexp2
  | "xlog2" -> T.xlog2
  | f -> invalid_arg f

let functions =
  group "functions"
    [
      group "compute each value their golden states"
        [
          Golden.cases ~key:[ "function"; "dtype"; "x" ] "values.golden"
            (fun cell ->
              let dt = named (cell "dtype") in
              equal const
                (const_of_cell (cell "result"))
                (at dt
                   (unary (cell "function"))
                   (as_float (value_of_cell (cell "x")))));
        ];
    ]

let xpow =
  let pow b e =
    as_float
      (Interpreter.eval
         ~params:[ (0, `Float b); (1, `Float e) ]
         (T.xpow (x Dtype.Float32) (x ~slot:1 Dtype.Float32)))
  in
  let negative =
    Gen.map (fun b -> -.Float.abs b) (drawn Dtype.Float32 ~exponents:(117, 137))
  in
  group "xpow"
    [
      cases "anything to the power 0 is 1" ~id:string_of_float
        [ 0.; -0.; 1.; -2.; 0.5; Float.infinity; Float.neg_infinity; Float.nan ]
        (fun b -> equal float_exact 1. (pow b 0.));
      cases "a negative base to a non-integer power is NaN"
        ~id:(fun (b, e) -> Printf.sprintf "%g ^ %g" b e)
        [ (-2., 0.5); (-8., -1.5) ]
        (fun (b, e) -> equal float_exact Float.nan (pow b e));
      test "negative infinity to a non-integer power is not NaN" (fun () ->
          equal float_exact Float.infinity (pow Float.neg_infinity 0.5));
      prop
        "a negative base to an integer power is the power of its magnitude, \
         negated for an odd exponent"
        ~examples:[ (-2., 3); (-2., 2); (-2., -1); (-0.5, 3) ]
        Gen.(
          with_pp
            (fun ppf (b, e) -> Format.fprintf ppf "%h ^ %d" b e)
            (pair negative (int_range (-20) 20)))
        (fun (b, e) ->
          let odd = e land 1 = 1 in
          cover "an odd exponent" odd;
          cover "an even exponent" (not odd);
          let magnitude = pow (-.b) (float_of_int e) in
          equal float_exact
            (if odd then -.magnitude else magnitude)
            (pow b (float_of_int e)));
      group "computes each power its golden states"
        [
          Golden.cases ~key:[ "dtype"; "base"; "exponent" ] "pow_values.golden"
            (fun cell ->
              let dt = named (cell "dtype") in
              let params =
                [
                  (0, value_of_cell (cell "base"));
                  (1, value_of_cell (cell "exponent"));
                ]
              in
              equal const
                (const_of_cell (cell "result"))
                (Interpreter.eval ~params (T.xpow (x dt) (x ~slot:1 dt))));
        ];
      test "xpow takes the other floats" (fun () ->
          let others = Dtype.[ Bfloat16; Fp8e4m3; Float16 ] in
          equal (list dtype) others
            (List.map
               (fun dt -> Ops.dtype (T.xpow (x dt) (x ~slot:1 dt)))
               others));
      group "builds the graph its golden states"
        (List.map
           (fun dt ->
             Golden.graph
               ("xpow_" ^ name dt ^ ".golden")
               (fun () -> Ops.sink [ T.xpow (x dt) (x ~slot:1 dt) ]))
           floats);
    ]

(* [int_power dt b e] is [b] to the power [e] in [dt]: [e] factors of [b],
   wrapped at [dt]'s width, for [e >= 0]; [0] for a negative [e], but [1] for a
   base of [1] and [-1] or [1] by the parity of [e] for [-1]. *)
let int_power dt b e =
  let modulus = Bigint.shift_left Bigint.one (width dt) in
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

let integers =
  Dtype.[ Int8; Uint8; Int16; Uint16; Int32; Uint32; Int64; Uint64 ]

(* A type, a base of either sign, 1 and -1 among them, and an exponent that is
   negative, small, or past the width, within the range of OCaml's int. *)
let power_case =
  let open Gen in
  let* dt = of_list ~pp:Dtype.pp integers in
  let w = width dt and unsigned = Dtype.is_unsigned dt in
  let lo, hi =
    if unsigned then (0, (1 lsl min 62 w) - 1)
    else
      let half = 1 lsl min 62 (w - 1) in
      (-half, half - 1)
  in
  let small = if unsigned then [ 0; 1; 2 ] else [ -1; 0; 1 ] in
  let negative = if unsigned then 0 else 2 in
  let+ b = frequency [ (1, of_list small); (3, int_range lo hi) ]
  and+ e =
    frequency
      [
        (negative, int_range (-9) (-1));
        (1, int_range 0 3);
        (2, int_range w (w + 9));
        (2, int_range 0 hi);
      ]
  in
  (dt, b, e)

let xpow_int =
  let signed = List.filter (fun dt -> not (Dtype.is_unsigned dt)) integers in
  let base =
    Ops.variable ~dtype:Dtype.Uint8 "b" (`Int Bigint.zero)
      (`Int (Bigint.of_int 255))
  in
  let w =
    Ops.variable "w" (`Int Bigint.zero) (`Int (Bigint.shift_left Bigint.one 40))
  in
  let power_at e =
    Interpreter.eval
      ~vars:[ ("b", `Int (Bigint.of_int 3)); ("w", `Int (Bigint.of_int 5)) ]
      (require_some (T.xpow_int base e Dtype.Uint8))
  in
  group "xpow_int"
    [
      prop "a power wraps as repeated products do"
        ~examples:
          (List.concat_map
             (fun dt -> [ (dt, 1, -3); (dt, -1, -3); (dt, -1, -4) ])
             signed
          @ List.map
              (fun dt -> (dt, 3, width dt + 1))
              (List.filter Dtype.is_unsigned integers))
        (Gen.with_pp
           (fun ppf (dt, b, e) ->
             Format.fprintf ppf "%a %d ^ %d" Dtype.pp dt b e)
           power_case)
        (fun (dt, b, e) ->
          cover (alias dt) true;
          cover "a negative exponent" (e < 0);
          cover "1 or -1 to a negative power" (e < 0 && abs b = 1);
          cover "a power that wraps" (abs b >= 2 && e >= width dt);
          let int n = `Int (Bigint.of_int n) in
          let p = require_some (T.xpow_int (x dt) (x ~slot:1 dt) dt) in
          equal const
            (int_power dt (Bigint.of_int b) e)
            (Interpreter.eval ~params:[ (0, int b); (1, int e) ] p));
      cases "refuses a type that is no committed integer" ~id:alias
        Dtype.[ Float32; Weak_int; Bool ]
        (fun dt -> is_none (T.xpow_int (Ops.int 2) (Ops.int 3) dt));
      group "an exponent's bounds"
        [
          test "a weak expression no 64-bit integer holds is refused" (fun () ->
              raises_match (Exn.invalid_arg ~sub:"exponent past 64 bits")
                (fun () -> T.xpow_int base Ops.O.(w * w) Dtype.Uint8));
          (* 3^(5 + 3) = 6561, which is 161 modulo 256 *)
          test "a weak expression a 64-bit integer holds is computed" (fun () ->
              equal const
                (`Int (Bigint.of_int 161))
                (power_at Ops.O.(w + int 3)));
          (* 3 has order 64 modulo 256, which divides 2^70 *)
          test "a literal exponent is computed at any size" (fun () ->
              equal const (`Int Bigint.one)
                (power_at (Ops.const (`Int (Bigint.shift_left Bigint.one 70)))));
        ];
    ]

(* Rewriting *)

let sin d = Ops.alu d Sin []
let all_ops = Op.[ Exp2; Log2; Sin; Sqrt ]

let transcendentals dt =
  let d = x dt in
  Ops.sink [ Ops.exp2 d; Ops.log2 d; sin d; Ops.sqrt d ]

let rewrite ?(force = false) ops sink =
  Ops.graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:() sink
    (After_sources (T.patterns ~force (Op.Set.of_list ops)))

(* [rewritten op dt] is [op] of a [dt] input, rewritten by the patterns of a
   target that has none of the operations. *)
let rewritten op dt =
  match Ops.src (rewrite [] (Ops.sink [ op (x dt) ])) with
  | [ u ] -> u
  | _ -> invalid_arg "a sink of one node"

let every_float =
  Dtype.[ Float16; Float32; Float64; Bfloat16; Fp8e4m3; Fp8e5m2fnuz ]

let narrow_floats =
  Dtype.[ Float16; Bfloat16; Fp8e4m3; Fp8e5m2; Fp8e4m3fnuz; Fp8e5m2fnuz ]

let fp8s = Dtype.[ Fp8e4m3; Fp8e5m2; Fp8e4m3fnuz; Fp8e5m2fnuz ]

let transcendental_functions =
  [
    ("exp2", Ops.exp2, T.xexp2);
    ("log2", Ops.log2, T.xlog2);
    ("sin", sin, T.xsin ?fast:None ?switch_over:None);
  ]

(* Each function's rewrite at each narrow float, and its float32 function. *)
let narrow_rewrites =
  lazy
    (List.concat_map
       (fun (f, op, _) ->
         List.map (fun dt -> ((f, dt), rewritten op dt)) narrow_floats)
       transcendental_functions)

let float32_of =
  lazy
    (List.map
       (fun (f, _, t) -> (f, t (x Dtype.Float32)))
       transcendental_functions)

(* [float32_value f v] is the float32 function [f] at [v], computed once for
   each pair: the 8-bit floats share most of their values. *)
let float32_value =
  let seen = Hashtbl.create 1024 in
  fun f v ->
    let key = (f, Int64.bits_of_float v) in
    match Hashtbl.find_opt seen key with
    | Some y -> y
    | None ->
        let y =
          as_float
            (Interpreter.eval
               ~params:[ (0, `Float v) ]
               (List.assoc f (Lazy.force float32_of)))
        in
        Hashtbl.add seen key y;
        y

(* [decode dt c] is the value of [dt]'s code [c]. *)
let decode dt c =
  as_float (Dtype.bitcast (uint_of dt) dt (`Int (Bigint.of_int c)))

(* The rewrite on a narrow float is the float32 function at its widening,
   rounded once to the narrow float: the law checks that composition, at every
   code of each 8-bit float and at drawn codes of the 16-bit ones. It checks no
   accuracy: accuracy.golden holds the 16-bit floats' and none at the values the
   8-bit codes widen to. *)
let narrow_law =
  let functions = List.map (fun (f, _, _) -> f) transcendental_functions in
  let every_8_bit_code =
    List.concat_map
      (fun f ->
        List.concat_map (fun dt -> List.init 256 (fun c -> (f, dt, c))) fp8s)
      functions
  in
  prop
    "on a narrow float, an operation is its float32 function at the widened \
     value, rounded once"
    ~examples:every_8_bit_code
    Gen.(
      with_pp
        (fun ppf (f, dt, c) ->
          Format.fprintf ppf "%s %s 0x%x (%h)" f (alias dt) c (decode dt c))
        (let+ f = of_list functions
         and+ dt = of_list Dtype.[ Float16; Bfloat16 ]
         and+ c = int_range 0 0xffff in
         (f, dt, c)))
    (fun (f, dt, c) ->
      cover f true;
      cover (alias dt) true;
      let v = decode dt c in
      let in_float32 = float32_value f v in
      equal ~msg:(Printf.sprintf "at %h" v) const
        (Dtype.truncate dt (`Float in_float32) :> Dtype.const)
        (Interpreter.eval
           ~params:[ (0, `Float v) ]
           (List.assoc (f, dt) (Lazy.force narrow_rewrites))))

let patterns =
  let same =
    Testable.make
      ~pp:(fun ppf _ -> Format.pp_print_string ppf "<matcher>")
      ~equal:( == )
  in
  let first ~force ops = rewrite ~force ops (transcendentals Dtype.Float32) in
  group "patterns"
    [
      group "rewrite each graph as its golden states"
        (List.map
           (fun dt ->
             Golden.graph
               ("patterns_none_" ^ name dt ^ ".golden")
               (fun () -> rewrite [] (transcendentals dt)))
           every_float
        @ [
            Golden.graph "patterns_all_float.golden" (fun () ->
                rewrite all_ops (transcendentals Float32));
            Golden.graph "patterns_all_forced_float.golden" (fun () ->
                rewrite ~force:true all_ops (transcendentals Float32));
            Golden.graph "patterns_exp2_log2_float.golden" (fun () ->
                rewrite Op.[ Exp2; Log2 ] (transcendentals Float32));
            Golden.graph "patterns_sqrt_bfloat16.golden" (fun () ->
                rewrite Op.[ Exp2; Log2; Sin ] (transcendentals Bfloat16));
          ]);
      test "a target with every operation keeps the graph" (fun () ->
          let sink = transcendentals Dtype.Float32 in
          equal Uops.uop sink (rewrite all_ops sink));
      cases "a target without the operations is left none of them" ~id:name
        every_float (fun dt ->
          equal (list Uops.uop) []
            (List.filter
               (fun u -> List.mem (Ops.op u) all_ops)
               (Ops.toposort ~calls:Enter (rewrite [] (transcendentals dt)))));
      narrow_law;
      test "equal arguments give one matcher" (fun () ->
          equal same
            (T.patterns ~force:false (Op.Set.of_list Op.[ Exp2; Sqrt ]))
            (T.patterns ~force:false (Op.Set.of_list Op.[ Sqrt; Exp2 ])));
      test "a set gets its own rules after another's" (fun () ->
          let exp2 = first ~force:false Op.[ Exp2 ] in
          let none = first ~force:false [] in
          equal Uops.uop exp2 (first ~force:false Op.[ Exp2 ]);
          equal Uops.uop none (first ~force:false []));
      test "force gets its own rules after the other's" (fun () ->
          let kept = first ~force:false Op.[ Exp2 ] in
          let forced = first ~force:true Op.[ Exp2 ] in
          equal Uops.uop kept (first ~force:false Op.[ Exp2 ]);
          equal Uops.uop forced (first ~force:true Op.[ Exp2 ]);
          not_equal Uops.uop kept forced);
    ]

(* Special values

   Exact, through the rewrite of each operation, in every float the patterns
   rewrite: IEEE's values at infinities, NaN and zeros, a zero's sign kept by
   the sine, and the sine of a subnormal, which rounds to the subnormal. *)

let sixteen_up = Dtype.[ Float16; Bfloat16; Float32; Float64 ]
let tiny dt = Float.ldexp 1. (1 - T.exponent_bias dt - mantissa dt)

let greatest dt =
  Float.ldexp (2. -. Float.ldexp 1. (-mantissa dt)) (T.exponent_bias dt)

let specials =
  let nan = Float.nan and inf = Float.infinity in
  let c v _ = v in
  let special claim op rows =
    cases claim
      ~id:(fun (dt, v, _) -> Printf.sprintf "%s %h" (alias dt) v)
      (List.concat_map
         (fun dt -> List.map (fun (v, e) -> (dt, v dt, e dt)) rows)
         sixteen_up)
      (fun (dt, v, expected) ->
        equal const (`Float expected)
          (Interpreter.eval ~params:[ (0, `Float v) ] (rewritten op dt)))
  in
  group "special values"
    [
      special
        "sin is NaN at the infinities and NaN, and a zero or a subnormal itself"
        sin
        [
          (c inf, c nan);
          (c (-.inf), c nan);
          (c nan, c nan);
          (c 0., c 0.);
          (c (-0.), c (-0.));
          (tiny, tiny);
          ((fun dt -> -.tiny dt), fun dt -> -.tiny dt);
        ];
      special
        "exp2 is IEEE's at the infinities, NaN, zeros and ±1, and past the \
         greatest value"
        Ops.exp2
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
        ];
      special
        "log2 is IEEE's at the infinities, NaN, zeros, 1, 2, 0.5 and below 0"
        Ops.log2
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
        ];
    ]

(* Accuracy

   Each function is within one unit in the last place of the correctly rounded
   result, counted on the ordered line of its type. [accuracy.golden] holds the
   results, mpmath's rounded once to the type, at values drawn from the bits of
   the type (every magnitude, the subnormals, the zeros, the infinities and
   NaN), from the range where the result is neither 0, 1 nor infinite, and for
   the sine near multiples of a quarter turn, where it cancels. *)

let accuracy_of = [ ("exp2", Ops.exp2); ("log2", Ops.log2); ("sin", sin) ]

let accuracy =
  let rewrites =
    lazy
      (List.concat_map
         (fun (f, op) ->
           List.map (fun dt -> ((f, alias dt), rewritten op dt)) sixteen_up)
         accuracy_of)
  in
  let fast = Dtype.[ Float32; Float64 ] in
  let row cell =
    let f = cell "function" and dname = cell "dtype" in
    let dt = named dname in
    let v = as_float (value_of_cell (cell "x"))
    and expected = as_float (value_of_cell (cell "result")) in
    within_ulp dt (List.assoc (f, alias dt) (Lazy.force rewrites)) v expected;
    (* below 30, the fast sine too, defined in float32 and float64 *)
    if f = "sin" && Float.abs v < 30. && List.mem dt fast then
      within_ulp dt (T.xsin ~fast:true (x dt)) v expected
  in
  group "accuracy"
    [
      group
        "exp2, log2 and sin are within an ulp of the values their golden states"
        [ Golden.cases ~key:[ "function"; "dtype"; "x" ] "accuracy.golden" row ];
    ]

(* Errors *)

let errors =
  let refuse ?(also = []) name f =
    cases
      (name ^ " refuses a node of another type")
      ~id:alias
      (also @ Dtype.[ Int32; Bfloat16; Fp8e4m3; Bool ])
      (fun dt -> rejects (fun () -> f (x dt)))
  in
  let float16 = [ Dtype.Float16 ] in
  group "errors"
    [
      refuse ~also:float16 "xsin" (T.xsin ?fast:None ?switch_over:None);
      refuse ~also:float16 "xsin ~fast" (T.xsin ~fast:true ?switch_over:None);
      refuse ~also:float16 "xexp2" T.xexp2;
      refuse ~also:float16 "xlog2" T.xlog2;
      refuse "frexp" T.frexp;
      refuse "rintk" T.rintk;
      refuse "payne_hanek_reduction" T.payne_hanek_reduction;
      refuse "cody_waite_reduction" T.cody_waite_reduction;
      cases "pow2if refuses an integer of another width" ~id:alias
        Dtype.[ Int8; Uint32; Float32; Bool ]
        (fun dt -> rejects (fun () -> T.pow2if (x dt) Dtype.Float32));
    ]

let () =
  exit
    (run "Tolk.Transcendental"
       [
         exponent_bias;
         shl;
         shr;
         rintk;
         pow2if;
         frexp;
         payne_hanek;
         cody_waite;
         xsin;
         xexp2;
         xlog2;
         functions;
         xpow;
         xpow_int;
         patterns;
         specials;
         accuracy;
         errors;
       ])
