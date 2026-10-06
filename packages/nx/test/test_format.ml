(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* How tensors print. [pp]'s layout is a baseline: its text is reviewed when
   accepted. *)

open Windtrap

let v dt xs = Nx.create dt [| Array.length xs |] xs

let printing =
  group "printing"
    [
      test "a scalar, a vector and a matrix" (fun () ->
          expect (Nx.to_string (Nx.scalar Nx.float32 1.5))
          @@ __POS_OF__ {| 1.5 |};
          expect (Nx.to_string (v Nx.int32 [| 1l; -2l; 3l |]))
          @@ __POS_OF__ {| [1, -2, 3] |};
          expect
            (Nx.to_string (Nx.reshape [| 2; 3 |] (Nx.arange Nx.int32 0 6 1)))
          @@ __POS_OF__
               {|
            int32 [2,3]
            [[0, 1, 2],
             [3, 4, 5]]
            |});
      test "an empty tensor and a tensor of three axes" (fun () ->
          expect (Nx.to_string (Nx.zeros Nx.float64 [| 0; 3 |]))
          @@ __POS_OF__
               {|
            float64 [0,3]
            []
            |};
          expect
            (Nx.to_string (Nx.reshape [| 2; 2; 2 |] (Nx.arange Nx.int32 0 8 1)))
          @@ __POS_OF__
               {|
            int32 [2,2,2]
            [[[0, 1],
              [2, 3]],
             [[4, 5],
              [6, 7]]]
            |});
      test "each kind of element" (fun () ->
          expect (Nx.to_string (v Nx.bool [| true; false |]))
          @@ __POS_OF__ {| [true, false] |};
          expect
            (Nx.to_string
               (v Nx.float64 [| 0.1; Float.nan; Float.infinity; -0. |]))
          @@ __POS_OF__ {| [0.1, nan, inf, -0] |};
          expect
            (Nx.to_string
               (v Nx.complex64
                  Complex.[| { re = 1.; im = -2. }; { re = 0.5; im = 3. } |]))
          @@ __POS_OF__ {| [(1-2i), (0.5+3i)] |};
          expect (Nx.to_string (v Nx.uint8 [| 0; 255 |]))
          @@ __POS_OF__ {| [0, 255] |});
      test "a long tensor is truncated" (fun () ->
          expect (Nx.to_string (Nx.arange Nx.int32 0 1000 1))
          @@ __POS_OF__
               {|
            int32 [1000]
            [0, 1, ..., 998, 999]
            |};
          expect
            (Nx.to_string
               (Nx.reshape [| 100; 100 |] (Nx.arange Nx.int32 0 10000 1)))
          @@ __POS_OF__
               {|
            int32 [100,100]
            [[0, 1, ..., 98, 99],
             [100, 101, ..., 198, 199],
             ...
             [9800, 9801, ..., 9898, 9899],
             [9900, 9901, ..., 9998, 9999]]
            |});
      test "a view prints its elements" (fun () ->
          let m = Nx.reshape [| 2; 3 |] (Nx.arange Nx.int32 0 6 1) in
          equal string
            (Nx.to_string (Nx.contiguous (Nx.transpose m)))
            (Nx.to_string (Nx.transpose m)));
      test "print writes to_string and a newline" (fun () ->
          let t = v Nx.int32 [| 1l; 2l |] in
          Nx.print t;
          equal string (Nx.to_string t ^ "\n") (output ()));
      test "pp_shape brackets the dimensions and pp_dtype names the dtype"
        (fun () ->
          equal string "[2,3,4]"
            (Format.asprintf "%a" Nx.pp_shape [| 2; 3; 4 |]);
          equal string "[]" (Format.asprintf "%a" Nx.pp_shape [||]);
          equal string "float32" (Format.asprintf "%a" Nx.pp_dtype Nx.float32);
          equal string "bfloat16" (Format.asprintf "%a" Nx.pp_dtype Nx.bfloat16));
    ]

(* Floats *)

type float_dtype = F : string * (float, 'b) Nx.dtype -> float_dtype

let float_dtypes =
  [
    F ("float16", Nx.float16);
    F ("float32", Nx.float32);
    F ("float64", Nx.float64);
    F ("bfloat16", Nx.bfloat16);
    F ("float8_e4m3", Nx.float8_e4m3);
    F ("float8_e5m2", Nx.float8_e5m2);
  ]

let pp_float_dtype ppf (F (name, _)) = Format.pp_print_string ppf name

(* Values across the exponents of every float dtype, NaN and the infinities. *)
let any_value =
  Gen.frequency
    [
      (2, Gen.any_float);
      (2, Gen.float_range (-1000.) 1000.);
      ( 2,
        Gen.map
          (fun (m, e) -> ldexp m e)
          (Gen.pair (Gen.float_range (-1.) 1.) (Gen.int_range (-80) 80)) );
    ]

(* The significant digits of a decimal text, as [1.50e+03] has two. *)
let significant text =
  let mantissa =
    match String.index_opt text 'e' with
    | Some i -> String.sub text 0 i
    | None -> text
  in
  let digits = String.concat "" (String.split_on_char '.' mantissa) in
  let digits = String.concat "" (String.split_on_char '-' digits) in
  let n = String.length digits in
  let first = ref 0 and last = ref (n - 1) in
  while !first < n && digits.[!first] = '0' do
    incr first
  done;
  while !last > !first && digits.[!last] = '0' do
    decr last
  done;
  !last - !first + 1

let rec pow10 p = if p = 0 then 1 else 10 * pow10 (p - 1)

(* The values of [dt] on either side of the positive finite [x]: an infinity
   past the largest finite value, and [0.] below the least. *)
let neighbours (type b) (dt : (float, b) Nx.dtype) x =
  match dt with
  | Nx_dtype.Float64 -> (Float.pred x, Float.succ x)
  | Nx_dtype.Float32 ->
      let bits = Int32.bits_of_float x in
      ( Int32.float_of_bits (Int32.pred bits),
        Int32.float_of_bits (Int32.succ bits) )
  | dt ->
      let s = Nx_dtype.Scalar.of_dtype dt in
      let code = Nx_dtype.Scalar.encode s x in
      let next = Nx_dtype.Scalar.decode s (code + 1) in
      ( Nx_dtype.Scalar.decode s (code - 1),
        if Float.is_nan next then infinity else next )

(* Whether [text] rounds to [x] in [dt], to nearest: a store of it gives [x],
   and for a finite [x] it is at most half a step past the largest finite value,
   beyond which the float8 dtypes saturate instead of rounding. *)
let reads_back dt x text =
  let v = float_of_string text in
  let top = Nx_dtype.max_finite dt in
  let below, _ = neighbours dt top in
  ((not (Float.is_finite x)) || Float.abs v <= top +. ((top -. below) /. 2.))
  && Int64.equal (Int64.bits_of_float x)
       (Int64.bits_of_float (Nx.item [] (Nx.scalar dt v)))

(* A decimal of fewer significant digits than [text] that reads back to the
   positive finite [x], found by trying every one between [x]'s neighbours. *)
let shorter_decimal dt x text =
  let prev, next = neighbours dt x in
  let k0 = Float.to_int (Float.floor (Float.log10 x)) in
  (* [v / 10^u] through logarithms, which neither overflow nor underflow, give
     or take a part in 1e13. *)
  let scaled v u = Float.pow 10. (Float.log10 v -. Float.of_int u) in
  let found = ref None in
  for q = 1 to significant text - 1 do
    for k = k0 - 1 to k0 + 1 do
      let u = k - q + 1 in
      let lo = scaled prev u in
      (* Past the largest finite value, the values that round to it span no more
         than those below it. *)
      let hi =
        if Float.is_finite next then scaled next u else (2. *. scaled x u) -. lo
      in
      let slack = Float.max 2. (hi *. 1e-13) in
      let lo = Float.to_int (Float.max 0. (lo -. slack))
      and hi = Float.to_int (Float.min 1e17 (hi +. slack)) in
      for m = Int.max lo (pow10 (q - 1)) to Int.min hi (pow10 q - 1) do
        let d = Printf.sprintf "%de%d" m u in
        if !found = None && reads_back dt x d then found := Some d
      done
    done
  done;
  !found

let fewest dt x =
  let text = Nx.to_string (Nx.scalar dt x) in
  equal ~msg:(text ^ " reads back") bool true (reads_back dt x text);
  if Float.is_finite x && x <> 0. then
    equal
      ~msg:("fewer digits than " ^ text)
      (option string) None
      (shorter_decimal dt (Float.abs x) text)

let shortest (F (_, dt), x) =
  let x = Nx.item [] (Nx.scalar dt x) in
  if Float.is_nan x then equal string "nan" (Nx.to_string (Nx.scalar dt x))
  else begin
    cover "a finite value" (Float.is_finite x && x <> 0.);
    fewest dt x
  end

(* Every positive finite value of a float8 dtype, and the powers of two each
   float dtype holds. *)
let float8_values (type b) (dt : (float, b) Nx.dtype) =
  let s = Nx_dtype.Scalar.of_dtype dt in
  List.filter Float.is_finite
    (List.init 0x7F (fun c -> Nx_dtype.Scalar.decode s (c + 1)))

let powers_of_two (type b) (dt : (float, b) Nx.dtype) =
  List.filter
    (fun x -> Float.is_finite x && Nx.item [] (Nx.scalar dt x) = x)
    (List.init 2200 (fun k -> Float.ldexp 1. (k - 1100)))

(* The golden of gen/float_text.py: the text of every positive finite value of
   the narrow float dtypes and of every positive power of two of float32 and
   float64, from numpy's shortest repr and an exact search. *)
let golden_texts () =
  let decode = function
    | "float16" -> (F ("float16", Nx.float16), Nx_dtype.Scalar.Float16)
    | "bfloat16" -> (F ("bfloat16", Nx.bfloat16), Nx_dtype.Scalar.BFloat16)
    | "float8_e4m3" ->
        (F ("float8_e4m3", Nx.float8_e4m3), Nx_dtype.Scalar.Float8_e4m3)
    | "float8_e5m2" ->
        (F ("float8_e5m2", Nx.float8_e5m2), Nx_dtype.Scalar.Float8_e5m2)
    | "float32" -> (F ("float32", Nx.float32), Nx_dtype.Scalar.Float32)
    | "float64" -> (F ("float64", Nx.float64), Nx_dtype.Scalar.Float64)
    | name -> failf "golden: unknown dtype %s" name
  in
  let value (s : Nx_dtype.Scalar.t) hex =
    match s with
    | Float32 -> Int32.float_of_bits (Int32.of_string ("0x" ^ hex))
    | Float64 -> Int64.float_of_bits (Int64.of_string ("0x" ^ hex))
    | s -> Nx_dtype.Scalar.decode s (int_of_string ("0x" ^ hex))
  in
  let ic = open_in "golden/float_text.txt" in
  let rec read current wrong =
    match In_channel.input_line ic with
    | None -> List.rev wrong
    | Some line -> (
        match String.split_on_char ' ' line with
        | [ name ] -> read (Some (decode name)) wrong
        | [ hex; text ] ->
            let F (name, dt), s = Option.get current in
            let printed = Nx.to_string (Nx.scalar dt (value s hex)) in
            if printed = text then read current wrong
            else
              read current
                (Printf.sprintf "%s %s: %s, not %s" name hex printed text
                :: wrong)
        | _ -> failf "golden: bad line %S" line)
  in
  let wrong =
    Fun.protect ~finally:(fun () -> close_in ic) (fun () -> read None [])
  in
  equal (list string) [] (List.filteri (fun i _ -> i < 20) wrong)

let floats =
  let prints dt x text () = equal string text (Nx.to_string (Nx.scalar dt x)) in
  group "floats"
    [
      prop
        "a float prints as the fewest digits that read back to it at its dtype"
        (Gen.pair
           (Gen.with_pp pp_float_dtype (Gen.of_list float_dtypes))
           any_value)
        shortest;
      test
        "every narrow float and every power of two prints as numpy's shortest \
         repr"
        golden_texts;
      test "every float8 value prints with the fewest digits" (fun () ->
          List.iter (fewest Nx.float8_e4m3) (float8_values Nx.float8_e4m3);
          List.iter (fewest Nx.float8_e5m2) (float8_values Nx.float8_e5m2));
      test "every power of two prints with the fewest digits" (fun () ->
          List.iter
            (fun (F (_, dt)) -> List.iter (fewest dt) (powers_of_two dt))
            float_dtypes);
      test "the largest float8 e4m3, 448, prints as a decimal that rounds to it"
        (prints Nx.float8_e4m3 448. "450");
      test "a float32 1.0000001 prints every digit it needs"
        (prints Nx.float32 1.0000001 "1.0000001");
      test "a float64 0.1 + 0.2 prints every digit it needs"
        (prints Nx.float64 (0.1 +. 0.2) "0.30000000000000004");
      test "a float32 third prints the digits of a float32"
        (prints Nx.float32 (1. /. 3.) "0.33333334");
      test "a NaN prints as nan, whatever its sign" (fun () ->
          let nan = Nx.scalar Nx.float32 Float.nan in
          equal string "nan" (Nx.to_string (Nx.neg nan));
          equal string "nan"
            (Nx.to_string (Nx.sqrt (Nx.scalar Nx.float64 (-1.)))));
      test "integers below 1e16 print in full"
        (prints Nx.float64 1e15 "1000000000000000");
      test "from 1e16 a float prints with an exponent"
        (prints Nx.float64 1e16 "1e+16");
      test "a float prints in full down to 1e-4"
        (prints Nx.float64 1e-4 "0.0001");
      test "below 1e-4 a float prints with an exponent"
        (prints Nx.float64 1.5e-5 "1.5e-05");
      test "an error message prints a float as pp does" (fun () ->
          raises_match (Exn.invalid_arg ~substring:"alpha 1.0000001,")
            (fun () -> Nx.ewma ~alpha:1.0000001 (Nx.zeros Nx.float32 [| 2 |])));
      test "a complex number prints each part as its float" (fun () ->
          equal string "(0.1+0.33333334i)"
            (Nx.to_string
               (Nx.scalar Nx.complex64 Complex.{ re = 0.1; im = 1. /. 3. })));
    ]

let () = exit (run "nx format" [ printing; floats ])
