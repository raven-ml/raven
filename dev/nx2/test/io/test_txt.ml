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

let () = exit (run "nx.io text" [ txt ])
