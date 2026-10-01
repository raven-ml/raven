(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Elementwise operations, lowered: each is checked against nx's eager result by
   its class. An exact operation gives eager's bits over the values that break
   arithmetic; a transcendental one is within its budget of units in the last
   place of the correctly rounded result. The graph is evaluated by tolk's
   reference interpreter, whose [exp2], [log2], [sin] and [sqrt] are correctly
   rounded: these tests measure the compositions, and the targets' primitives
   are measured apart. *)

open Windtrap
open Nx_test
open Traces

let pp_float ppf x = Format.fprintf ppf "%h" x

(* [traced f] is the value of [f ()] traced, its operands captured. *)
let traced f =
  let s, y = trace f in
  value s y

let agrees f = exact (f ()) (traced f)

(* Floats *)

type float_dtype = F : string * (float, 'b) Nx.dtype -> float_dtype

let float_dtypes =
  [
    F ("float32", Nx.float32);
    F ("float64", Nx.float64);
    F ("float16", Nx.float16);
    F ("bfloat16", Nx.bfloat16);
  ]

let floats dtype = viewed ~pp:pp_float dtype Gen.any_float

(* Two values of one shape. *)
let float_pairs dtype =
  let open Gen in
  let* shape = array ~size:(int_range 0 3) (int_range 1 4) in
  let n = Array.fold_left ( * ) 1 shape in
  let+ a = array ~size:(constant n) any_float
  and+ b = array ~size:(constant n) any_float in
  (Nx.create dtype shape a, Nx.create dtype shape b)

let float_rows name (f : 'b. (float, 'b) Nx.t -> (float, 'b) Nx.t) =
  group name
    (List.map
       (fun (F (dname, dt)) ->
         prop dname (floats dt) (fun x -> agrees (fun () -> f x)))
       float_dtypes)

type unary = { f : 'b. (float, 'b) Nx.t -> (float, 'b) Nx.t }

let exact_float_unary =
  group "exact unary floats"
    (List.map
       (fun (name, { f }) -> float_rows name f)
       [
         ("neg", { f = Nx.neg });
         ("recip", { f = Nx.recip });
         ("abs", { f = Nx.abs });
         ("sqrt", { f = Nx.sqrt });
         ("sign", { f = Nx.sign });
         ("trunc", { f = Nx.trunc });
         ("ceil", { f = Nx.ceil });
         ("floor", { f = Nx.floor });
         ("round", { f = Nx.round });
       ])

type binary = {
  g : 'b. (float, 'b) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t;
}

let exact_float_binary =
  group "exact binary floats"
    (List.map
       (fun (name, { g }) ->
         group name
           (List.map
              (fun (F (dname, dt)) ->
                (* A float64 remainder unrolls 187 reduction steps. *)
                let count =
                  if name = "mod" && dname = "float64" then 20 else 100
                in
                prop ~count dname (float_pairs dt) (fun (a, b) ->
                    agrees (fun () -> g a b)))
              float_dtypes))
       [
         ("add", { g = Nx.add });
         ("sub", { g = Nx.sub });
         ("mul", { g = Nx.mul });
         ("div", { g = Nx.div });
         ("mod", { g = Nx.mod_ });
       ])

let extremes =
  group "extremes"
    [
      test "the maximum and the minimum of zeros are a zero" (fun () ->
          let a = Nx.create Nx.float32 [| 2 |] [| -0.; 0. |] in
          let b = Nx.create Nx.float32 [| 2 |] [| 0.; -0. |] in
          let agrees f = exact_up_to_zero (f ()) (traced f) in
          agrees (fun () -> Nx.maximum a b);
          agrees (fun () -> Nx.minimum a b));
      test "relu, the maximum and the minimum read no sign bit" (fun () ->
          let x = Nx.zeros Nx.float32 [| 4 |]
          and y = Nx.ones Nx.float32 [| 4 |] in
          let reads_sign f =
            List.exists
              (fun u -> Tolk.Ops.op u = Tolk.Op.Bitcast)
              (Tolk.Ops.toposort (Programs.kernels (snd (trace f))))
          in
          is_false ~msg:"relu" (reads_sign (fun () -> Nx.relu x));
          is_false ~msg:"maximum" (reads_sign (fun () -> Nx.maximum x y));
          is_false ~msg:"minimum" (reads_sign (fun () -> Nx.minimum x y)));
    ]

let nan_extremes =
  group "extremes of NaN"
    [
      test "maximum and minimum are NaN when an operand is" (fun () ->
          let a = Nx.create Nx.float32 [| 3 |] [| Float.nan; 1.; Float.nan |] in
          let b = Nx.create Nx.float32 [| 3 |] [| 1.; Float.nan; Float.nan |] in
          agrees (fun () -> Nx.maximum a b);
          agrees (fun () -> Nx.minimum a b));
    ]

(* Integers *)

type law = { law : 'a 'b. ('a, 'b) Nx.t -> unit }
type law2 = { law2 : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> unit }

(* A property of each integer dtype, over the values that break arithmetic. *)
let int_prop (Int_dtype { name; dtype; bits; signed; of_i64; _ }) { law } =
  let value = Gen.map of_i64 (int_value ~bits ~signed) in
  prop name
    (viewed ~pp:(fun ppf _ -> Format.pp_print_string ppf "_") dtype value)
    law

let int_prop2 (Int_dtype { name; dtype; bits; signed; of_i64; _ }) { law2 } =
  let value = Gen.map of_i64 (int_value ~bits ~signed) in
  let pairs =
    let open Gen in
    let* shape = array ~size:(int_range 0 3) (int_range 1 4) in
    let n = Array.fold_left ( * ) 1 shape in
    let+ a = array ~size:(constant n) value
    and+ b = array ~size:(constant n) value in
    (Nx.create dtype shape a, Nx.create dtype shape b)
  in
  prop name pairs (fun (a, b) -> law2 a b)

type int_unary = { h : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t }

let exact_int_unary =
  group "exact unary integers"
    (List.map
       (fun (op, { h }) ->
         group op
           (List.map
              (fun d -> int_prop d { law = (fun x -> agrees (fun () -> h x)) })
              int_dtypes))
       [
         ("neg", { h = Nx.neg });
         ("recip", { h = Nx.recip });
         ("abs", { h = Nx.abs });
         ("sign", { h = Nx.sign });
         ("round", { h = Nx.round });
       ])

type int_binary = { k : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t }

let exact_int_binary =
  group "exact binary integers"
    (List.map
       (fun (op, { k }) ->
         group op
           (List.map
              (fun d ->
                int_prop2 d { law2 = (fun a b -> agrees (fun () -> k a b)) })
              int_dtypes))
       [
         ("add", { k = Nx.add });
         ("sub", { k = Nx.sub });
         ("mul", { k = Nx.mul });
         ("div", { k = Nx.div });
         ("mod", { k = Nx.mod_ });
         ("pow", { k = Nx.pow });
         ("maximum", { k = Nx.maximum });
         ("minimum", { k = Nx.minimum });
         ("and", { k = Nx.bitwise_and });
         ("or", { k = Nx.bitwise_or });
         ("xor", { k = Nx.bitwise_xor });
       ])

(* Comparisons and selections *)

let comparisons =
  group "comparisons"
    (List.map
       (fun (op, cmp) ->
         prop op (float_pairs Nx.float32) (fun (a, b) ->
             agrees (fun () -> cmp a b)))
       [
         ("equal", Nx.equal);
         ("not_equal", Nx.not_equal);
         ("less", Nx.less);
         ("less_equal", Nx.less_equal);
       ])

let selections =
  group "selections"
    [
      prop "where picks its branch" (float_pairs Nx.float32) (fun (a, b) ->
          let c = Nx.less a b in
          agrees (fun () -> Nx.where c a b));
    ]

(* Conversions *)

(* A bitcast between widths: operands, under a layout that keeps a widening's
   last axis whole, and the dtype they are read as. *)
type between =
  | Between : string * ('a, 'b) Nx.t Gen.t * ('c, 'd) Nx.dtype -> between

let betweens =
  let rows ~pp dtype k value =
    let shape =
      Gen.map
        (fun s -> Array.append s [| k |])
        (Gen.array ~size:(Gen.int_range 1 2) (Gen.int_range 0 3))
    in
    viewed ~shape ~layout:row_layout ~pp dtype value
  in
  let ints dtype k lo hi =
    rows ~pp:Format.pp_print_int dtype k (Gen.int_range lo hi)
  in
  let pp_i32 ppf v = Format.fprintf ppf "%ld" v in
  let pp_i64 ppf v = Format.fprintf ppf "%Ld" v in
  [
    Between ("uint8 to uint64", ints Nx.uint8 8 0 255, Nx.uint64);
    Between ("uint8 to float32", ints Nx.uint8 4 0 255, Nx.float32);
    Between ("int16 to int64", ints Nx.int16 4 (-32768) 32767, Nx.int64);
    Between
      ("uint32 to float64", rows ~pp:pp_i32 Nx.uint32 2 Gen.int32, Nx.float64);
    Between ("float32 to uint16", floats Nx.float32, Nx.uint16);
    Between ("float64 to uint8", floats Nx.float64, Nx.uint8);
    Between ("uint64 to int32", viewed ~pp:pp_i64 Nx.uint64 Gen.int64, Nx.int32);
  ]

let conversions =
  group "conversions"
    [
      group "a float saturates at an integer's range, NaN at 0"
        (List.map
           (fun (Int_dtype { name; dtype; _ }) ->
             prop name (floats Nx.float32) (fun x ->
                 agrees (fun () -> Nx.cast dtype x)))
           int_dtypes);
      group "an integer rounds to a float once"
        (List.map
           (fun d ->
             int_prop d
               {
                 law =
                   (fun x ->
                     agrees (fun () -> Nx.cast Nx.float32 x);
                     agrees (fun () -> Nx.cast Nx.bfloat16 x));
               })
           int_dtypes);
      prop "a double rounds to bfloat16 once" (floats Nx.float64) (fun x ->
          agrees (fun () -> Nx.cast Nx.bfloat16 x));
      prop "a float's bits read as an integer" (floats Nx.float32) (fun x ->
          agrees (fun () -> Nx.bitcast Nx.int32 x));
      group "a bitcast between widths reads the bytes eager reads"
        (List.map
           (fun (Between (name, operands, dt)) ->
             prop name operands (fun x -> agrees (fun () -> Nx.bitcast dt x)))
           betweens);
    ]

(* Transcendental functions

   Each function is measured over the inputs of its golden table: a sweep of its
   domain, both signs, and the special values. In [float64] the table holds the
   correctly rounded results (mpmath at 200 bits); in [float32] they are the
   double-precision libm's rounded once, which is correctly rounded but within a
   few units of the 29th bit of a tie. *)

type nary = { n : 'b. (float, 'b) Nx.t array -> (float, 'b) Nx.t }

(* The columns of a golden table, as floats. *)
let columns file =
  let rows = Golden.rows ("golden/lower_arith/" ^ file) in
  List.map
    (fun name ->
      Array.of_list (List.map (fun cell -> float_of_string (cell name)) rows))
    (Golden.columns ("golden/lower_arith/" ^ file))

(* How a traced value is computed: by the reference interpreter, or compiled for
   the host. *)
type evaluation = {
  eval : 'a 'b. Rune_internals.Lower.scope -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t;
}

let measure ~by ~budget ~libm { n } file (F (_, dt)) =
  let columns = columns file in
  let expected = List.nth columns (List.length columns - 1) in
  let inputs = List.filteri (fun i _ -> i < List.length columns - 1) columns in
  let length = Array.length expected in
  let args =
    Array.of_list (List.map (fun c -> Nx.create dt [| length |] c) inputs)
  in
  let expected =
    if Nx_dtype.equal dt Nx_dtype.float64 then
      Nx.create dt [| length |] expected
    else
      let xs = Array.map Nx.to_array args in
      Nx.create dt [| length |]
        (Array.init length (fun i -> libm (Array.map (fun x -> x.(i)) xs)))
  in
  let s, y = trace (fun () -> n args) in
  ulps ~budget ~expected args (by.eval s y)

(* The budget holds in [float32] and [float64]; the narrow floats compute at
   [float32] and round once, within one unit. *)
let transcendental ~by name ~budget ~libm op file =
  group name
    (List.map
       (fun (F (dname, dt) as d) ->
         let budget = if Nx_dtype.itemsize dt < 4 then 1 else budget in
         test dname (fun () -> measure ~by ~budget ~libm op file d))
       float_dtypes)

let unary ~by ?(file = "") name ~budget libm { f } =
  transcendental ~by name ~budget
    ~libm:(fun a -> libm a.(0))
    { n = (fun a -> f a.(0)) }
    (if file = "" then name ^ "64.golden" else file)

let binary ~by name ~budget libm { g } file =
  transcendental ~by name ~budget
    ~libm:(fun a -> libm a.(0) a.(1))
    { n = (fun a -> g a.(0) a.(1)) }
    file

(* [tanh] where its lowering changes form: at [|x| = 1/4], where [e^(-2|x|) - 1]
   stops being a series; around [1/2]; from 8 to 22, where [e^(-2|x|) - 1]
   rounds to [-1] in [float32] and then in [float64]; and at the tiny and
   subnormal arguments where [tanh x] rounds to [x]. Against libm, rounded once
   to the dtype. *)
let tanh_edges ~by =
  let around c =
    List.concat_map
      (fun k ->
        let d = Float.of_int k in
        [ c +. (d *. Float.ldexp c (-52)); c +. (d *. Float.ldexp c (-23)) ])
      [ -3; -2; -1; 0; 1; 2; 3 ]
  in
  let magnitudes =
    List.concat_map around [ 0.25; 0.5; 1.; 8.; 9.; 18.; 19.; 22. ]
    @ [ 0x1p-28; 0x1.8p-28; 0x1p-12; 0x1p-126; 0x1p-1074 ]
  in
  let points =
    Array.of_list (List.concat_map (fun m -> [ m; -.m ]) magnitudes)
  in
  group "tanh where its form changes"
    (List.map
       (fun (F (dname, dt)) ->
         test dname (fun () ->
             let x = Nx.create dt [| Array.length points |] points in
             let expected = Nx.map_item Float.tanh x in
             let s, y = trace (fun () -> Nx.tanh x) in
             ulps
               ~budget:(if Nx_dtype.itemsize dt < 4 then 1 else 8)
               ~expected [| x |] (by.eval s y)))
       float_dtypes)

let transcendentals ?tags name by =
  group ?tags name
    [
      unary ~by "exp" ~budget:4 Float.exp { f = Nx.exp };
      unary ~by "log" ~budget:4 Float.log { f = Nx.log };
      unary ~by "sin" ~budget:4 Float.sin { f = Nx.sin };
      unary ~by "cos" ~budget:4 Float.cos { f = Nx.cos };
      unary ~by "tan" ~budget:8 Float.tan { f = Nx.tan };
      unary ~by ~file:"sin_far64.golden" "sin of large arguments" ~budget:4
        Float.sin { f = Nx.sin };
      unary ~by ~file:"cos_far64.golden" "cos of large arguments" ~budget:4
        Float.cos { f = Nx.cos };
      unary ~by ~file:"tan_far64.golden" "tan of large arguments" ~budget:8
        Float.tan { f = Nx.tan };
      unary ~by "asin" ~budget:8 Float.asin { f = Nx.asin };
      unary ~by "acos" ~budget:8 Float.acos { f = Nx.acos };
      unary ~by "atan" ~budget:8 Float.atan { f = Nx.atan };
      unary ~by "sinh" ~budget:8 Float.sinh { f = Nx.sinh };
      unary ~by "cosh" ~budget:8 Float.cosh { f = Nx.cosh };
      unary ~by "tanh" ~budget:8 Float.tanh { f = Nx.tanh };
      tanh_edges ~by;
      unary ~by "erf" ~budget:8 Float.erf { f = Nx.erf };
      binary ~by "atan2" ~budget:8 Float.atan2 { g = Nx.atan2 }
        "atan2_64.golden";
      binary ~by "pow" ~budget:16 Float.pow { g = Nx.pow } "pow64.golden";
      binary ~by "pow of negative bases and integral exponents" ~budget:16
        Float.pow { g = Nx.pow } "pow_integral64.golden";
    ]

(* Logarithms

   The logarithm of a number below zero is NaN, the subnormals included, and
   that of either zero is [-inf]; the functions nx composes of it follow. *)

type logarithm = {
  name : string;
  libm : float -> float;
  log : 'b. (float, 'b) Nx.t -> (float, 'b) Nx.t;
}

let logarithms =
  [
    { name = "log"; libm = Float.log; log = Nx.log };
    { name = "log2"; libm = Float.log2; log = Nx.log2 };
    (* nx's [log1p], through the functions that use it. *)
    { name = "asinh"; libm = Float.asinh; log = Nx.asinh };
    { name = "atanh"; libm = Float.atanh; log = Nx.atanh };
  ]

(* The negative subnormals, from the least to the greatest in magnitude, the
   smallest negative normal, [-0.] and NaN, in [dt]. *)
let below_zero (F (_, dt)) =
  let tiny, top, normal =
    if Nx_dtype.equal dt Nx.float64 then
      (-0x1p-1074, -0x0.fffffffffffffp-1022, -0x1p-1022)
    else (-0x1p-149, -0x1.fffffcp-127, -0x1p-126)
  in
  [| tiny; top; normal; -0.; Float.nan |]

let wide = [ F ("float32", Nx.float32); F ("float64", Nx.float64) ]

(* The classes of results: NaN, [inf], [-inf] or finite. *)
let classes xs =
  Array.map
    (fun x ->
      if Float.is_nan x then "nan"
      else if x = Float.infinity then "inf"
      else if x = Float.neg_infinity then "-inf"
      else "finite")
    xs

let edge_float =
  Gen.frequency
    [
      (3, Gen.any_float);
      ( 2,
        Gen.of_list ~pp:pp_float
          [ -0x1p-1074; -0x1p-1030; -0x1p-149; -0x1p-130; -0x1p-126; -0. ] );
    ]

let logarithm_tests { eval } =
  List.map
    (fun { name; libm; log } ->
      group name
        (List.concat_map
           (fun (F (dname, dt) as d) ->
             [
               test (dname ^ " below zero") (fun () ->
                   let x = Nx.create dt [| 5 |] (below_zero d) in
                   let s, y = trace (fun () -> log x) in
                   exact (log x) (eval s y));
               prop ~count:40
                 (dname ^ " has libm's classes")
                 (Gen.with_pp
                    (Format.pp_print_list pp_float)
                    (Gen.list ~size:(Gen.constant 16) edge_float))
                 (fun xs ->
                   let x = Nx.create dt [| 16 |] (Array.of_list xs) in
                   let s, y = trace (fun () -> log x) in
                   equal (array string)
                     (classes (Array.map libm (Nx.to_array x)))
                     (classes (Nx.to_array (eval s y))));
             ])
           wide))
    logarithms

let logarithms_group ?tags name ({ eval } as by) =
  group ?tags name
    (test "log and log2 are NaN below zero and -inf at -0." (fun () ->
         List.iter
           (fun (F (_, dt) as d) ->
             let x = Nx.create dt [| 5 |] (below_zero d) in
             let expected =
               Nx.create dt [| 5 |]
                 [|
                   Float.nan;
                   Float.nan;
                   Float.nan;
                   Float.neg_infinity;
                   Float.nan;
                 |]
             in
             List.iter
               (fun log ->
                 let s, y = trace (fun () -> log x) in
                 exact expected (eval s y))
               [ Nx.log; Nx.log2 ])
           wide)
    :: logarithm_tests by)

(* Random bits *)

let random_bits =
  group "random bits"
    [
      test "a traced key draws eager's words, compiled for the host" (fun () ->
          let words =
            [| 0l; 0l; 1l; 0l; -1l; 0x7fffffffl; 0x12345678l; -0x789abcdel |]
          in
          let key = Nx.create Nx.int32 [| 1; 2 |] [| 0x1bd11bdal; -1l |] in
          let counter = Nx.create Nx.int32 [| 4; 2 |] words in
          let s = scope () in
          let y =
            within s (fun () ->
                let k = argument s key in
                Nx.Op.eval (Threefry (Nx.broadcast_to [| 4; 2 |] k, counter)))
          in
          let eager =
            Nx.Op.eval (Threefry (Nx.broadcast_to [| 4; 2 |] key, counter))
          in
          exact eager (Programs.compiled s y));
      test "a key that does not depend on the arguments is refused" (fun () ->
          let key = Nx.create Nx.int32 [| 2 |] [| 1l; 2l |] in
          let counter = Nx.create Nx.int32 [| 2 |] [| 3l; 4l |] in
          raises_match
            (function Rune_internals.Lower.Jit_error _ -> true | _ -> false)
            (fun () -> trace (fun () -> Nx.Op.eval (Threefry (key, counter)))));
    ]

(* Long reductions

   An angle with known bounds below the reductions' limits takes the short
   reduction alone, as Box-Muller's [2 pi u] does for a fused uniform [u] in
   [[0, 1)]. An angle read from a buffer has no bounds, and [sin] and [cos]
   reduce it themselves, the long way past their limit: the target's sine then
   sees a remainder bounded by a quarter turn, and takes the short reduction. *)

(* The first nonzero word of the bits of [1/(2 pi)], which only the long
   (Payne-Hanek) reduction reads. *)
let payne_hanek_word : Tolk.Dtype.const = `Int (Tolk.Bigint.of_int 0x28be60db)

(* The magnitude at which tolk's sine switches to the long reduction, which it
   compares with only where it cannot bound its angle. *)
let switch_over : Tolk.Dtype.const = `Float 30.

(* [kernels_reading c y] is the number of [y]'s kernels, lowered for the host,
   that read the constant [c]. *)
let kernels_reading c y =
  let reads k =
    List.exists
      (fun u ->
        match Tolk.Ops.arg u with Tolk.Ops.Const v -> v = c | _ -> false)
      (Tolk.Ops.toposort
         (Tolk.Codegen.full_rewrite_to_sink k (host Nx_device.host)))
  in
  List.length (List.filter reads (Tolk.Ops.src (Programs.kernels y)))

(* [drawn s f dt] is the draw [f] traced in [s] from a key argument. *)
let drawn s f dt =
  let key = (Nx.Rng.key 7 :> (int32, Nx.int32_elt) Nx.t) in
  within s (fun () -> f (Nx.Rng.of_tensor (argument s key)) dt [| 256 |])

let long_reductions_group =
  let read f dt =
    let s = scope () in
    let x = Nx.zeros dt [| 4 |] in
    within s (fun () -> f (argument s x))
  in
  group "long reductions"
    [
      cases
        ~name:(fun (F (name, _)) -> name)
        "a normal draw takes none" wide
        (fun (F (_, dt)) ->
          let s = scope () in
          equal int 0
            (kernels_reading payne_hanek_word (drawn s Nx.Rng.normal dt)));
      cases
        ~name:(fun (F (name, _)) -> name)
        "the sine of an angle read from a buffer takes its own only" wide
        (fun (F (_, dt)) ->
          let y = read Nx.sin dt in
          equal int 1 (kernels_reading payne_hanek_word y);
          equal int 0 (kernels_reading switch_over y));
      cases
        ~name:(fun (F (name, _)) -> name)
        "the cosine of an angle read from a buffer takes its own only" wide
        (fun (F (_, dt)) ->
          let y = read Nx.cos dt in
          equal int 1 (kernels_reading payne_hanek_word y);
          equal int 0 (kernels_reading switch_over y));
    ]

(* Random draws, compiled for the host: a uniform draw is eager's bits, and a
   normal one within the ulps that two implementations of [log], [sqrt] and
   [cos] or [sin] leave between them (3 measured over 4096 draws). *)

let random_draws =
  let compiled f dt =
    let key = Nx.Rng.key 7 in
    let draw k = f k dt [| 256 |] in
    let y =
      Rune_internals.Rune.jit
        Nx.Ptree.(Nx.Rng.ptree @-> returns tensor)
        draw key
    in
    (draw key, Nx.place Nx.Placement.host y)
  in
  group "random draws"
    [
      cases
        ~name:(fun (F (name, _)) -> name)
        "a uniform draw is eager's" wide
        (fun (F (_, dt)) ->
          let eager, y = compiled Nx.Rng.uniform dt in
          exact eager y);
      cases
        ~name:(fun (F (name, _)) -> name)
        "a normal draw is within 4 ulps of eager's" wide
        (fun (F (_, dt)) ->
          let eager, y = compiled Nx.Rng.normal dt in
          ulps ~budget:4 ~expected:eager [||] y);
    ]

(* Compiled for the host

   The exact operations over the values that break arithmetic, every pair of
   them for a binary operation, compiled with the host's compiler: what C leaves
   undefined (signed overflow, division by zero, conversions out of range) must
   not reach it. *)

let float_edges =
  [
    0.;
    -0.;
    1.;
    -1.;
    0.5;
    -0.5;
    1.5;
    -1.5;
    2.5;
    -2.5;
    0.49999997;
    -0.49999997;
    0x1p-149;
    -0x1p-149;
    0x1p-126;
    0x1.fffffep127;
    -0x1.fffffep127;
    3.25;
    -7.75;
    1e20;
    -1e20;
    4194304.5;
    Float.infinity;
    Float.neg_infinity;
    Float.nan;
  ]

let host_float name { f } =
  test name (fun () ->
      let x =
        Nx.create Nx.float32
          [| List.length float_edges |]
          (Array.of_list float_edges)
      in
      let s, y = trace (fun () -> f x) in
      exact (f x) (Programs.compiled s y))

let host_float2 name { g } =
  test name (fun () ->
      let n = List.length float_edges in
      let a = Nx.create Nx.float32 [| n; 1 |] (Array.of_list float_edges) in
      let b = Nx.create Nx.float32 [| 1; n |] (Array.of_list float_edges) in
      let s, y = trace (fun () -> g a b) in
      exact (g a b) (Programs.compiled s y))

let int_edges ~bits ~signed =
  let lo, hi = int_range ~bits ~signed in
  List.map (wrap ~bits ~signed)
    [
      lo;
      hi;
      Int64.succ lo;
      Int64.pred hi;
      0L;
      1L;
      -1L;
      2L;
      -2L;
      7L;
      100L;
      -100L;
    ]

let host_int { h } (Int_dtype { name; dtype; bits; signed; of_i64; _ }) =
  test name (fun () ->
      let values = Array.of_list (List.map of_i64 (int_edges ~bits ~signed)) in
      let x = Nx.create dtype [| Array.length values |] values in
      let s, y = trace (fun () -> h x) in
      exact (h x) (Programs.compiled s y))

let host_int2 { k } (Int_dtype { name; dtype; bits; signed; of_i64; _ }) =
  test name (fun () ->
      let values = Array.of_list (List.map of_i64 (int_edges ~bits ~signed)) in
      let n = Array.length values in
      let a = Nx.create dtype [| n; 1 |] values
      and b = Nx.create dtype [| 1; n |] values in
      let s, y = trace (fun () -> k a b) in
      exact (k a b) (Programs.compiled s y))

let on_the_host =
  group ~tags:[ "slow" ] "exact operations on the host"
    [
      group "floats"
        [
          host_float "neg" { f = Nx.neg };
          host_float "recip" { f = Nx.recip };
          host_float "abs" { f = Nx.abs };
          host_float "sqrt" { f = Nx.sqrt };
          host_float "sign" { f = Nx.sign };
          host_float "trunc" { f = Nx.trunc };
          host_float "ceil" { f = Nx.ceil };
          host_float "floor" { f = Nx.floor };
          host_float "round" { f = Nx.round };
          host_float2 "add" { g = Nx.add };
          host_float2 "mul" { g = Nx.mul };
          host_float2 "div" { g = Nx.div };
          host_float2 "mod" { g = Nx.mod_ };
          host_float2 "less_equal"
            { g = (fun a b -> Nx.cast (Nx.dtype a) (Nx.less_equal a b)) };
        ];
      test "a product and a sum round twice, fused in one kernel" (fun () ->
          (* (1 + 2^-12)^2 rounds to 1 + 2^-11, a tie to even, so the sum is 0;
             a fused multiply-add would keep 2^-24. *)
          let x = Nx.full Nx.float32 [| 16 |] (1. +. 0x1p-12) in
          let c = Nx.full Nx.float32 [| 16 |] (-.(1. +. 0x1p-11)) in
          let s, y = trace (fun () -> Nx.add (Nx.mul x x) c) in
          exact (Nx.zeros Nx.float32 [| 16 |]) (Programs.compiled s y));
      group "integers"
        (List.map
           (fun (op, h) -> group op (List.map (host_int h) int_dtypes))
           [ ("neg", { h = Nx.neg }); ("abs", { h = Nx.abs }) ]
        @ List.map
            (fun (op, k) -> group op (List.map (host_int2 k) int_dtypes))
            [
              ("mul", { k = Nx.mul });
              ("div", { k = Nx.div });
              ("mod", { k = Nx.mod_ });
              ("pow", { k = Nx.pow });
            ]);
      group "conversions"
        (List.map
           (fun (Int_dtype { name; dtype; _ }) ->
             test name (fun () ->
                 let x =
                   Nx.create Nx.float32
                     [| List.length float_edges |]
                     (Array.of_list float_edges)
                 in
                 let s, y = trace (fun () -> Nx.cast dtype x) in
                 exact (Nx.cast dtype x) (Programs.compiled s y)))
           int_dtypes
        @ [
            test "a double near a bfloat16 tie rounds once" (fun () ->
                let ties =
                  [|
                    1. +. 0x1p-8 +. 0x1p-30;
                    1. +. 0x1p-8 -. 0x1p-30;
                    1. +. 0x1p-8;
                    3. +. 0x1p-7 +. 0x1p-40;
                  |]
                in
                let x = Nx.create Nx.float64 [| 4 |] ties in
                let s, y = trace (fun () -> Nx.cast Nx.bfloat16 x) in
                exact (Nx.cast Nx.bfloat16 x) (Programs.compiled s y));
            test "a 64-bit integer near a float16 tie rounds once" (fun () ->
                let x =
                  Nx.create Nx.int64 [| 4 |]
                    [| 2049L; 2051L; 0x20000000000001L; -65519L |]
                in
                let s, y = trace (fun () -> Nx.cast Nx.float16 x) in
                exact (Nx.cast Nx.float16 x) (Programs.compiled s y));
            test "eight bytes read as one word, little-endian" (fun () ->
                let x =
                  Nx.create Nx.uint8 [| 2; 8 |]
                    (Array.init 16 (fun i -> ((i * 37) + 1) land 255))
                in
                let s, y = trace (fun () -> Nx.bitcast Nx.uint64 x) in
                exact (Nx.bitcast Nx.uint64 x) (Programs.compiled s y));
            test "a float read as its two halves" (fun () ->
                let x =
                  Nx.create Nx.float32
                    [| List.length float_edges |]
                    (Array.of_list float_edges)
                in
                let s, y = trace (fun () -> Nx.bitcast Nx.uint16 x) in
                exact (Nx.bitcast Nx.uint16 x) (Programs.compiled s y));
          ]);
    ]

(* Graph parity: the kernels tinygrad schedules for the same program. *)

let parity =
  let x () = Nx.zeros Nx.float32 [| 4; 4 |] in
  let i () = Nx.zeros Nx.int32 [| 4; 4 |] in
  let case file f =
    Golden.graph
      ("golden/lower_arith/" ^ file ^ ".golden")
      (fun () ->
        let args = f () in
        Programs.kernels (snd (trace (fun () -> args ()))))
  in
  group "graph parity"
    [
      case "neg" (fun () ->
          let a = x () in
          fun () -> Nx.neg a);
      case "recip" (fun () ->
          let a = x () in
          fun () -> Nx.recip a);
      case "sqrt" (fun () ->
          let a = x () in
          fun () -> Nx.sqrt a);
      case "trunc" (fun () ->
          let a = x () in
          fun () -> Nx.trunc a);
      case "ceil" (fun () ->
          let a = x () in
          fun () -> Nx.ceil a);
      case "floor" (fun () ->
          let a = x () in
          fun () -> Nx.floor a);
      case "add" (fun () ->
          let a = x () and b = x () in
          fun () -> Nx.add a b);
      case "sub" (fun () ->
          let a = x () and b = x () in
          fun () -> Nx.sub a b);
      case "mul" (fun () ->
          let a = x () and b = x () in
          fun () -> Nx.mul a b);
      case "equal" (fun () ->
          let a = x () and b = x () in
          fun () -> Nx.equal a b);
      case "not_equal" (fun () ->
          let a = x () and b = x () in
          fun () -> Nx.not_equal a b);
      case "less" (fun () ->
          let a = x () and b = x () in
          fun () -> Nx.less a b);
      case "where" (fun () ->
          let a = x () and b = x () and c = x () and d = x () in
          fun () -> Nx.where (Nx.less a b) c d);
      case "bitwise_and" (fun () ->
          let a = i () and b = i () in
          fun () -> Nx.bitwise_and a b);
      case "bitwise_or" (fun () ->
          let a = i () and b = i () in
          fun () -> Nx.bitwise_or a b);
      case "bitwise_xor" (fun () ->
          let a = i () and b = i () in
          fun () -> Nx.bitwise_xor a b);
      case "cast_to_half" (fun () ->
          let a = x () in
          fun () -> Nx.cast Nx.float16 a);
      case "cast_to_double" (fun () ->
          let a = x () in
          fun () -> Nx.cast Nx.float64 a);
      case "bitcast" (fun () ->
          let a = x () in
          fun () -> Nx.bitcast Nx.int32 a);
    ]

let () =
  exit
  @@ run "lower_arith"
       [
         exact_float_unary;
         exact_float_binary;
         extremes;
         nan_extremes;
         exact_int_unary;
         exact_int_binary;
         comparisons;
         selections;
         conversions;
         transcendentals "transcendental functions" { eval = value };
         transcendentals ~tags:[ "slow" ] "transcendental functions on the host"
           { eval = Programs.compiled };
         logarithms_group "logarithms" { eval = value };
         logarithms_group ~tags:[ "slow" ] "logarithms on the host"
           { eval = Programs.compiled };
         random_bits;
         long_reductions_group;
         random_draws;
         on_the_host;
         parity;
       ]
