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

let text_cases =
  case ~rank:2
    ~kinds:[ Scalar; Single; Nan; Negative_zero; Bounds; Many ]
    ~views:[ Plain; Transposed; Flipped ]
    txt_dtypes

let txt =
  group "text"
    [
      prop "a vector or a matrix saved as text loads as itself, NaN aside"
        text_cases (fun (Case (dt, _, _, _) as c) ->
          let x = Nx.unpack dt (value c) in
          covers ~empty:false ~broadcast:false c;
          cover "a sub-byte dtype" (D.bits dt < 8);
          let path = temp_file () in
          Nx_io.save_txt path x;
          let y = Nx_io.load_txt dt path in
          equal ~msg:"shape" (array int) (text_shape (Nx.shape x)) (Nx.shape y);
          equal ~msg:"elements" (elements dt) (Nx.to_array x) (Nx.to_array y));
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
        "integers are written in decimal, unsigned ones as unsigned"
        [
          ( "1 0",
            fun p ->
              Nx_io.save_txt p (Nx.create Nx.bool [| 2 |] [| true; false |]) );
          ( "-9223372036854775808 9223372036854775807",
            fun p ->
              Nx_io.save_txt p
                (Nx.create Nx.int64 [| 2 |] [| Int64.min_int; Int64.max_int |])
          );
          ( "18446744073709551615 0",
            fun p ->
              Nx_io.save_txt p (Nx.create Nx.uint64 [| 2 |] [| -1L; 0L |]) );
          ( "4294967295",
            fun p -> Nx_io.save_txt p (Nx.create Nx.uint32 [| 1 |] [| -1l |]) );
          ( "-8 7",
            fun p -> Nx_io.save_txt p (Nx.create Nx.int4 [| 2 |] [| -8; 7 |]) );
        ]
        (fun (expected, save) ->
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
