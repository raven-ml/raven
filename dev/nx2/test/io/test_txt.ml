(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Text. A 0-d value, vector or matrix saved as text loads as itself, NaN aside,
   over every dtype but the complex ones, a 0-d value and a matrix of one row or
   column as a vector; floats are written in %.18e notation from their exact
   value and integers in decimal; a field is read by the grammar the interface
   states, and a file that breaks it raises [Failure] naming it. *)

open Windtrap
open Io_support

let txt_dtypes = List.filter (fun (D.Any dt) -> not (D.is D.Complex dt)) D.all

(* The elements of [dt], compared one by one, floats by their bits and any two
   NaNs alike, and printed one per line after its index, so that a failure names
   the elements that differ. *)
let elements (type v s) (dt : (v, s) D.t) : v array testable =
  let same : v -> v -> bool =
    match D.kind dt with
    | D.Float ->
        fun a b ->
          (Float.is_nan a && Float.is_nan b)
          || Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)
    | D.Complex | D.Signed | D.Unsigned | D.Boolean -> ( = )
  in
  let pp ppf a =
    Format.pp_open_vbox ppf 0;
    Array.iteri
      (fun i v ->
        if i > 0 then Format.pp_print_cut ppf ();
        Format.fprintf ppf "%d: %a" i (D.pp_value dt) v)
      a;
    Format.pp_close_box ppf ()
  in
  Testable.make ~pp ~equal:(fun a b ->
      Array.length a = Array.length b && Array.for_all2 same a b)

(* Text keeps a 0-d value and a matrix of one row or column as a vector. *)
let text_shape s =
  match s with [||] -> [| 1 |] | [| 1; n |] | [| n; 1 |] -> [| n |] | s -> s

(* Values of [dt] compare as text keeps them: by their shape as text holds it
   and their elements. *)
let as_text dt =
  Testable.contramap
    (fun x -> (text_shape (Nx.shape x), Nx.to_array x))
    (pair (array int) (elements dt))

let text_cases =
  case ~rank:2
    ~kinds:[ Scalar; Single; Nan; Negative_zero; Bounds; Many ]
    ~views:[ Plain; Transposed; Flipped ]
    txt_dtypes

let quoted = Format.(fun ppf s -> fprintf ppf "%S" s)
let separators = Gen.of_list ~pp:quoted [ " "; ","; "\t"; ";"; ", "; "||" ]
let prefixes = Gen.of_list ~pp:quoted [ "#"; "%%"; "//" ]

(* A comment's text, which may hold digits, separators and prefixes. *)
let comment = Gen.string_of ~size:(Gen.int_range 0 40) (Gen.char_range ' ' '~')

(* Floats read *)

(* [read_float s] is the float64 that [load_txt] reads from the field [s]. *)
let read_float s = (Nx.to_array (Nx_io.load_txt Nx.float64 (file s))).(0)

(* [exact m ~base k] is the decimal digits of [m base^k], computed a digit at a
   time, so that a literal states a value with no rounding. *)
let exact m ~base k =
  let ds = Array.make 800 0 in
  let mul n =
    let carry = ref 0 in
    for i = 0 to Array.length ds - 1 do
      let v = (ds.(i) * n) + !carry in
      ds.(i) <- v mod 10;
      carry := v / 10
    done
  in
  ds.(0) <- 1;
  mul m;
  for _ = 1 to k do
    mul base
  done;
  let top = ref (Array.length ds - 1) in
  while ds.(!top) = 0 do
    decr top
  done;
  String.init (!top + 1) (fun i -> Char.chr (48 + ds.(!top - i)))

(* Half the least subnormal, 2^-1075, and three times it, as decimals. *)
let half_least = exact 1 ~base:5 1075
let three_halves = exact 3 ~base:5 1075

(* The overflow threshold, (2^54 - 1) 2^970: the largest finite float64 and half
   its unit in the last place. *)
let threshold = exact ((1 lsl 54) - 1) ~base:2 970

(* [midpoint x] is the exact midpoint between the finite float64 [x >= 0] and
   the next one up, [(2n + 1) 2^(u - 1)] for [x = n 2^u], as decimal digits [d]
   and an exponent [e] for the value [d 10^e]. *)
let midpoint x =
  let bits = Int64.bits_of_float x in
  let biased = Int64.to_int (Int64.shift_right_logical bits 52) in
  let fraction = Int64.to_int (Int64.logand bits 0xF_FFFF_FFFF_FFFFL) in
  let n, u =
    if biased = 0 then (fraction, -1074)
    else (fraction + (1 lsl 52), biased - 1075)
  in
  let m = (2 * n) + 1 and k = u - 1 in
  if k >= 0 then (exact m ~base:2 k, 0) else (exact m ~base:5 (-k), k)

(* [below d] is the digits of [d - 1], for the digits [d] of a positive
   integer. *)
let below d =
  let b = Bytes.of_string d in
  let i = ref (Bytes.length b - 1) in
  while Bytes.get b !i = '0' do
    Bytes.set b !i '9';
    decr i
  done;
  Bytes.set b !i (Char.chr (Char.code (Bytes.get b !i) - 1));
  Bytes.to_string b

(* A finite float64 [x >= 0]: anywhere, a subnormal, zero, the top of a binade,
   whose next float up is the bottom of the next, or the largest. *)
let below_midpoints =
  let open Gen in
  let bits lo hi = map Int64.float_of_bits (int64_range lo hi) in
  with_pp
    (fun ppf x -> Format.fprintf ppf "%h" x)
    (one_of
       [
         bits 0L 0x7FEF_FFFF_FFFF_FFFFL;
         bits 1L 0xF_FFFF_FFFF_FFFFL;
         constant 0.;
         map (fun k -> Float.pred (Float.ldexp 1. k)) (int_range (-1021) 1023);
         constant Float.max_float;
       ])

(* The literals of floats.txt, each with the bits of the float64 nearest it. *)
let python_floats =
  In_channel.with_open_text (fixture "floats.txt") In_channel.input_lines
  |> List.map (fun line ->
      match String.split_on_char ' ' line with
      | [ s; bits ] -> (s, Int64.of_string ("0x" ^ bits))
      | _ -> invalid_arg line)

let floats_read =
  group "floats read"
    [
      test
        "a float field reads as Python's correctly rounded float() reads it, \
         to the bit" (fun () ->
          let literals = List.map fst python_floats in
          let path = file (String.concat "\n" literals ^ "\n") in
          let read = Nx.to_array (Nx_io.load_txt Nx.float64 path) in
          let wrong =
            List.concat
              (List.mapi
                 (fun i (s, bits) ->
                   let x = read.(i) in
                   if Int64.equal (Int64.bits_of_float x) bits then []
                   else
                     [
                       Printf.sprintf "%s: %h, not %h" s x
                         (Int64.float_of_bits bits);
                     ])
                 python_floats)
          in
          equal (list string) [] wrong);
      prop
        "a literal at the exact midpoint between two floats reads as the even \
         one, and one just above or below it as the float on its side"
        below_midpoints (fun x ->
          let up = Float.succ x in
          cover "zero" (x = 0.);
          cover "a subnormal" (x > 0. && x < Float.min_float);
          cover "the top of a binade" (x > 0. && fst (Float.frexp up) = 0.5);
          cover "the largest, whose midpoint rounds to infinity"
            (x = Float.max_float);
          let d, e = midpoint x in
          let path =
            file
              (Printf.sprintf "%se%d\n%s1e%d\n%s9e%d\n" d e d (e - 1) (below d)
                 (e - 1))
          in
          let even = Int64.(equal (logand (bits_of_float x) 1L) 0L) in
          equal (elements D.Float64)
            [| (if even then x else up); up; x |]
            (Nx.to_array (Nx_io.load_txt Nx.float64 path)));
      cases ~name:fst
        "a float field reads as the float64 nearest its value, a tie as the \
         even one"
        [
          ("2^53 + 1, a tie", ("9007199254740993", Float.ldexp 1. 53));
          ("2^53 + 3, a tie", ("9007199254740995", Float.ldexp 1. 53 +. 4.));
          ( "2^53 + 1 in 40 digits, a tie",
            ("9007199254740993000000000000000000000000e-24", Float.ldexp 1. 53)
          );
          ( "40 digits, the last past a tie",
            ( "9007199254740993000000000000000000000001e-24",
              Float.ldexp 1. 53 +. 2. ) );
          ( "just below 2^-792, a binade's bottom, nearer the float below",
            ("3.839223843572815e-239", Float.pred (Float.ldexp 1. (-792))) );
          ("half the least subnormal, a tie", (half_least ^ "e-1075", 0.));
          ( "just past half the least subnormal",
            (half_least ^ "1e-1076", Float.ldexp 1. (-1074)) );
          ( "three halves of the least subnormal, a tie",
            (three_halves ^ "e-1075", Float.ldexp 1. (-1073)) );
          ( "just below three halves of the least subnormal",
            (* The digits end in 5: one less, then a 9, is just below. *)
            ( String.sub three_halves 0 (String.length three_halves - 1)
              ^ "49e-1076",
              Float.ldexp 1. (-1074) ) );
          ( "1.7976931348623158e308, below the overflow threshold",
            ("1.7976931348623158e308", Float.max_float) );
          ( "1.797693134862315807e308, below the overflow threshold",
            ("1.797693134862315807e308", Float.max_float) );
          ( "1.797693134862315808e308, past the overflow threshold",
            ("1.797693134862315808e308", Float.infinity) );
          ("the overflow threshold, a tie", (threshold, Float.infinity));
          ("its negation", ("-" ^ threshold, Float.neg_infinity));
        ]
        (fun (_, (s, expected)) ->
          equal (elements D.Float64) [| expected |] [| read_float s |]);
    ]

let txt =
  group "text"
    [
      prop "a vector or a matrix saved as text loads as itself, NaN aside"
        text_cases (fun (Case (dt, _, _, _) as c) ->
          let x = Nx.unpack dt (value c) in
          covers ~empty:false ~broadcast:false c;
          cover "a sub-byte dtype" (D.bits dt < 8);
          Law.round_trip (as_text dt) string
            (saved "" (fun p x -> Nx_io.save_txt p x))
            (Nx_io.load_txt dt) x);
      prop
        "a value saved with a separator loads with it, lines under a comment \
         prefix skipped"
        (Gen.quad text_cases separators prefixes (Gen.pair comment comment))
        (fun ((Case (dt, _, _, _) as c), sep, prefix, (header, footer)) ->
          let commented x =
            let path = saved "" (fun p x -> Nx_io.save_txt ~sep p x) x in
            write path
              (prefix ^ header ^ "\n" ^ read path ^ prefix ^ footer ^ "\n");
            path
          in
          Law.round_trip (as_text dt) string commented
            (Nx_io.load_txt ~sep ~comments:prefix dt)
            (Nx.unpack dt (value c)));
      cases
        ~name:(fun s -> Format.asprintf "%a" Nx.pp_shape s)
        "a value with no element writes no row, and its file does not load"
        [ [| 0 |]; [| 0; 3 |]; [| 3; 0 |] ]
        (fun shape ->
          let path = temp_file () in
          Nx_io.save_txt path (Nx.zeros Nx.float32 shape);
          equal string "" (read path);
          fails ~path (fun () -> Nx_io.load_txt Nx.float32 path));
      test
        "a file numpy's savetxt wrote, with comment lines and CRLF line ends, \
         loads and saves back as its rows, its floats in %.18e notation from \
         their exact value" (fun () ->
          let numpy = fixture "savetxt.txt" and path = temp_file () in
          Nx_io.save_txt ~sep:"," path
            (Nx_io.load_txt ~sep:"," Nx.float64 numpy);
          let rows =
            String.split_on_char '\n' (read numpy)
            |> List.filter (fun l -> l <> "" && l.[0] <> '#')
            |> List.map (fun l -> String.sub l 0 (String.length l - 1) ^ "\n")
          in
          equal string (String.concat "" rows) (read path));
      cases ~name:fst
        "integers are written in decimal, unsigned ones as unsigned, and a \
         bool or bit as 1 or 0"
        [
          ( "bool",
            ( "1 0",
              fun p ->
                Nx_io.save_txt p (Nx.create Nx.bool [| 2 |] [| true; false |])
            ) );
          ( "bit",
            ( "1 0",
              fun p ->
                Nx_io.save_txt p (Nx.create Nx.bit [| 2 |] [| true; false |]) )
          );
          ( "int64",
            ( "-9223372036854775808 9223372036854775807",
              fun p ->
                Nx_io.save_txt p
                  (Nx.create Nx.int64 [| 2 |]
                     [| Int64.min_int; Int64.max_int |]) ) );
          ( "uint64",
            ( "18446744073709551615 0",
              fun p ->
                Nx_io.save_txt p (Nx.create Nx.uint64 [| 2 |] [| -1L; 0L |]) )
          );
          ( "uint32",
            ( "4294967295",
              fun p -> Nx_io.save_txt p (Nx.create Nx.uint32 [| 1 |] [| -1l |])
            ) );
          ( "int4",
            ( "-8 7",
              fun p -> Nx_io.save_txt p (Nx.create Nx.int4 [| 2 |] [| -8; 7 |])
            ) );
        ]
        (fun (_, (expected, save)) ->
          let path = temp_file () in
          save path;
          equal string (expected ^ "\n") (read path));
      test "rows are skipped and commented" (fun () ->
          let path = file "# a\n1 2\n\n3   4\n# b\n5 6\n" in
          equal (array int32)
            [| 1l; 2l; 3l; 4l; 5l; 6l |]
            (Nx.to_array (Nx_io.load_txt Nx.int32 path));
          equal (array int) [| 3; 2 |] (Nx.shape (Nx_io.load_txt Nx.int32 path));
          equal (array int32) [| 3l; 4l; 5l; 6l |]
            (Nx.to_array (Nx_io.load_txt ~skiprows:2 Nx.int32 path)));
      test "a float field takes a sign, a point, an exponent or a word"
        (fun () ->
          let path = file "+1e+3 .5 5. -2E-1 NaN -Infinity inf\n" in
          equal (elements D.Float64)
            [|
              1000.;
              0.5;
              5.;
              -0.2;
              Float.nan;
              Float.neg_infinity;
              Float.infinity;
            |]
            (Nx.to_array (Nx_io.load_txt Nx.float64 path));
          equal (array int32) [| 7l; -7l |]
            (Nx.to_array (Nx_io.load_txt Nx.int32 (file "+7 -7\n"))));
      test "a separator of several characters splits fields" (fun () ->
          let path = file "1::2::3\n" in
          equal (array int) [| 1; 2; 3 |]
            (Nx.to_array (Nx_io.load_txt ~sep:"::" Nx.int16 path)));
      cases ~name:fst
        "an empty field between separators other than the default fails, \
         naming its line"
        [
          ("between two", ("1,,2\n", ",", "line 1:"));
          ("at a line's start", ("# a header\n1,2\n,3\n", ",", "line 3:"));
          ("at a line's end", ("1;2;\n", ";", "line 1:"));
          ("of a separator of two characters", ("1::::2\n", "::", "line 1:"));
          ("at a line's start, by a tab", ("\t1\t2\n\t3\t4\n", "\t", "line 1:"));
          ("at a line's end, by a tab", ("1\t2\t\n3\t4\t\n", "\t", "line 1:"));
        ]
        (fun (_, (text, sep, line)) ->
          let path = file text in
          fails ~naming:line ~path (fun () ->
              Nx_io.load_txt ~sep Nx.float32 path));
      test "a field is trimmed of blanks, whatever the separator" (fun () ->
          equal (array int32) [| 1l; 2l; 3l; 4l |]
            (Nx.to_array
               (Nx_io.load_txt ~sep:"," Nx.int32 (file " 1, 2\n3 ,\t4 \r\n"))));
      cases ~name:fst
        "a row of another length or a field out of range names its line, \
         skipped and comment lines counted"
        [
          ("a short row", ("skipped\n# a comment\n1 2\n\n3\n", "line 5 "));
          ( "a field out of range",
            ("skipped\n# a comment\n1 2\n3 256\n", "line 4:") );
        ]
        (fun (_, (text, line)) ->
          let path = file text in
          fails ~naming:line ~path (fun () ->
              Nx_io.load_txt ~skiprows:1 Nx.uint8 path));
      test "a bool field is true, false or a number" (fun () ->
          let path = file "true FALSE 0 2.5 -0.0\n" in
          equal (array bool)
            [| true; false; false; true; false |]
            (Nx.to_array (Nx_io.load_txt Nx.bool path)));
      cases ~name:fst "a field outside its dtype or no literal of it fails"
        [
          ("256 in uint8", ("1 256\n", D.Any D.Uint8));
          ("-1 in uint32", ("-1\n", D.Any D.Uint32));
          ("2^64 in uint64", ("18446744073709551616\n", D.Any D.Uint64));
          ("2^63 in int64", ("9223372036854775808\n", D.Any D.Int64));
          ("8 in int4", ("8\n", D.Any D.Int4));
          ("a word in float32", ("1 x\n", D.Any D.Float32));
          ("hexadecimal in int32", ("0x10\n", D.Any D.Int32));
          ("an underscore in int32", ("1_0\n", D.Any D.Int32));
          ("a hexadecimal float", ("0x1p3\n", D.Any D.Float64));
          ("a NaN with a payload", ("nan(1)\n", D.Any D.Float64));
          ("an exponent without digits", ("1e\n", D.Any D.Float64));
          ("a lone point", (".\n", D.Any D.Float64));
          ("a float in int8", ("1.0\n", D.Any D.Int8));
          ("rows of two lengths", ("1 2\n3\n", D.Any D.Float32));
          ("no row", ("# only\n\n", D.Any D.Float32));
        ]
        (fun (_, (text, D.Any dt)) ->
          let path = file text in
          fails ~path (fun () -> Nx_io.load_txt dt path));
      test "a complex dtype, three axes or a negative skiprows are refused"
        (fun () ->
          let path = file "1\n" in
          invalid ~by:"Nx_io.load_txt" (fun () ->
              Nx_io.load_txt Nx.complex64 path);
          invalid ~by:"Nx_io.load_txt" (fun () ->
              Nx_io.load_txt ~skiprows:(-1) Nx.int8 path);

          invalid ~by:"Nx_io.save_txt" (fun () ->
              Nx_io.save_txt (temp_file ()) (Nx.zeros Nx.complex128 [| 2 |]));
          invalid ~by:"Nx_io.save_txt" (fun () ->
              Nx_io.save_txt (temp_file ()) (Nx.zeros Nx.int8 [| 1; 1; 1 |])));
    ]

let () = exit (run "nx.io text" [ txt; floats_read ])
