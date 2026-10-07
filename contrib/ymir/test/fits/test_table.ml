(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tables: every TFORM code, the unsigned offsets, TNULL, scaling, cell
   shapes, text grids, heap arrays and ASCII fields, read against astropy's
   reading of its own file (gen/fixtures.py); writing that reads back as
   its columns; and hostile bytes. *)

open Windtrap
open Ymir_fits
module H = Fits.Header
module V = Fits.Value
module T = Fits.Table

let path = "golden/tables.fits"
let hdus = lazy (require_ok (Fits.read path))
let hdu name = require_ok (Fits.get name (Lazy.force hdus))
let cat () = hdu "CAT"
let asc () = hdu "ASC"

let golden =
  lazy
    (In_channel.with_open_text "golden/tables.values" In_channel.input_lines
    |> List.map (fun l ->
        match String.split_on_char ' ' l with
        | name :: values -> (name, Array.of_list values)
        | [] -> assert false))

let expected name = List.assoc name (Lazy.force golden)

let unhex s =
  (* "s<hex>" *)
  let h = String.sub s 1 (String.length s - 1) in
  String.init
    (String.length h / 2)
    (fun i -> Char.chr (int_of_string ("0x" ^ String.sub h (2 * i) 2)))

let texts (type a b) (t : (a, b) Nx.t) : string array =
  let a = Nx.to_array (Nx.reshape [| Nx.numel t |] t) in
  match Nx.dtype t with
  | UInt8 -> Array.map string_of_int a
  | Int8 -> Array.map string_of_int a
  | Int16 -> Array.map string_of_int a
  | UInt16 -> Array.map string_of_int a
  | Int32 -> Array.map Int32.to_string a
  | UInt32 -> Array.map (Printf.sprintf "%lu") a
  | Int64 -> Array.map Int64.to_string a
  | Bool -> Array.map (fun b -> if b then "T" else "F") a
  | _ -> failwith "texts"

let floats name = Array.map float_of_string (expected name)

let strings_of (r : (int, Nx.uint8_elt) Nx_ragged.t) =
  let v = Nx.to_array (Nx_ragged.values r)
  and o = Nx.to_array (Nx_ragged.offsets r) in
  Array.init
    (Array.length o - 1)
    (fun i ->
      let a = Int64.to_int o.(i) and z = Int64.to_int o.(i + 1) in
      String.init (z - a) (fun k -> Char.chr v.(a + k)))

let ints (type a b) name (dtype : (a, b) Nx.dtype) h col =
  equal (array string) (expected name) (texts (require_ok (T.raw dtype col h)))

let reals (type b) name (dtype : (float, b) Nx.dtype) h col =
  let t = require_ok (T.raw dtype col h) in
  equal (array float_exact) (floats name)
    (Nx.to_array (Nx.reshape [| Nx.numel t |] t))

let texts_col name h col =
  equal (array string)
    (Array.map unhex (expected name))
    (strings_of (require_ok (T.ragged Nx.uint8 col h)))

let columns =
  cases ~name:fst "columns read as astropy reads them"
    [
      ("source_id", fun () -> ints "CAT.source_id" Nx.int64 (cat ()) "source_id");
      ("ra", fun () -> reals "CAT.ra" Nx.float64 (cat ()) "ra");
      ("mag", fun () -> reals "CAT.mag" Nx.float32 (cat ()) "mag");
      ("qual", fun () -> ints "CAT.qual" Nx.int16 (cat ()) "qual");
      ("u16", fun () -> ints "CAT.u16" Nx.uint16 (cat ()) "u16");
      ("i8", fun () -> ints "CAT.i8" Nx.int8 (cat ()) "i8");
      ("u32", fun () -> ints "CAT.u32" Nx.uint32 (cat ()) "u32");
      ("u8", fun () -> ints "CAT.u8" Nx.uint8 (cat ()) "u8");
      ("flag", fun () -> ints "CAT.flag" Nx.bool (cat ()) "flag");
      ( "bits",
        fun () ->
          let b = require_ok (T.raw Nx.bit "bits" (cat ())) in
          equal (array int) [| 6; 11 |] (Nx.shape b);
          equal (array string) (expected "CAT.bits") (texts (Nx.cast Nx.bool b))
      );
      ("name", fun () -> texts_col "CAT.name" (cat ()) "name");
      ( "cplx",
        fun () ->
          let c = require_ok (T.raw Nx.complex64 "cplx" (cat ())) in
          equal (array float_exact)
            (Array.concat
               (Array.to_list
                  (Array.map
                     (fun s ->
                       match String.split_on_char ',' s with
                       | [ a; b ] -> [| float_of_string a; float_of_string b |]
                       | _ -> [||])
                     (expected "CAT.cplx"))))
            (Array.concat
               (Array.to_list
                  (Array.map
                     (fun (z : Complex.t) -> [| z.re; z.im |])
                     (Nx.to_array c)))) );
      ( "vec",
        fun () ->
          let t = require_ok (T.raw Nx.float32 "vec" (cat ())) in
          equal (array int) [| 6; 3 |] (Nx.shape t);
          equal (array float_exact) (floats "CAT.vec")
            (Nx.to_array (Nx.reshape [| 18 |] t)) );
      ( "mat",
        fun () ->
          let t = require_ok (T.raw Nx.int32 "mat" (cat ())) in
          equal (array int) [| 6; 2; 3 |] (Nx.shape t);
          equal (array string) (expected "CAT.mat") (texts t) );
      ("grid", fun () -> texts_col "CAT.grid" (cat ()) "grid");
      ("scaled stored", fun () -> ints "CAT.scaled" Nx.int16 (cat ()) "scaled");
      ( "vla",
        fun () ->
          let r = require_ok (T.ragged Nx.int32 "vla" (cat ())) in
          equal (array string) (expected "CAT.vla") (texts (Nx_ragged.values r));
          equal (array int64)
            [| 0L; 3L; 1L; 5L; 2L; 0L |]
            (Nx.to_array (Nx_ragged.lengths r)) );
      ( "vld",
        fun () ->
          let r = require_ok (T.ragged Nx.float64 "vld" (cat ())) in
          equal (array float_exact) (floats "CAT.vld")
            (Nx.to_array (Nx_ragged.values r)) );
      ("vlt", fun () -> texts_col "CAT.vlt" (cat ()) "vlt");
      ("ascii id", fun () -> ints "ASC.id" Nx.int32 (asc ()) "id");
      ("ascii x", fun () -> reals "ASC.x" Nx.float64 (asc ()) "x");
      ("ascii y", fun () -> reals "ASC.y" Nx.float64 (asc ()) "y");
      ("ascii z", fun () -> reals "ASC.z" Nx.float64 (asc ()) "z");
      ("ascii s", fun () -> texts_col "ASC.s" (asc ()) "s");
    ]
    (fun (_, f) -> f ())

let physical () =
  let v = require_ok (T.values Nx.float64 "scaled" (cat ())) in
  equal (array float_exact)
    [| 100.; 100.5; 99.; 0.; 116.; 101.5 |]
    (Nx.to_array v);
  let q = Nx.to_array (require_ok (T.values Nx.float32 "qual" (cat ()))) in
  equal (array bool)
    [| false; true; false; false; true; false |]
    (Array.map Float.is_nan q);
  equal
    (result (option (array bool)) string)
    (Ok (Some [| true; false; true; true; false; true |]))
    (Result.map (Option.map Nx.to_array) (T.validity "qual" (cat ())));
  equal
    (result (option (array bool)) string)
    (Ok (Some [| true; true; false; true; true; true |]))
    (Result.map (Option.map Nx.to_array) (T.validity "mag" (cat ())));
  equal
    (result (option (array bool)) string)
    (Ok None)
    (Result.map (Option.map Nx.to_array) (T.validity "ra" (cat ())))

let exactness () =
  is_error (T.raw Nx.float32 "u32" (cat ()));
  is_error (T.raw Nx.int16 "u16" (cat ()));
  is_ok (T.raw Nx.int32 "u16" (cat ()));
  is_error (T.raw Nx.float64 "flag" (cat ()));
  is_error (T.values Nx.float64 "flag" (cat ()));
  is_error (T.raw Nx.int32 "vla" (cat ()));
  is_error (T.ragged Nx.int32 "ra" (cat ()));
  (match T.raw Nx.int64 "nope" (cat ()) with
  | Ok _ -> fail "a missing column read"
  | Error e -> contains ~sub:"source_id" e);
  is_error (T.raw ~rows:(0, 7) Nx.int64 "source_id" (cat ()));
  raises_match Exn.invalid_arg (fun () ->
      T.raw ~rows:(3, 2) Nx.int64 "source_id" (cat ()))

let rows_law =
  prop "rows read as the slice of all rows"
    Gen.(
      map
        (fun (a, b) -> (Int.min a b, Int.max a b))
        (pair (int_range 0 6) (int_range 0 6)))
    (fun (a, b) ->
      cover "empty" (a = b);
      let all = require_ok (T.raw Nx.int32 "mat" (cat ())) in
      let some = require_ok (T.raw ~rows:(a, b) Nx.int32 "mat" (cat ())) in
      equal (array int32)
        (Nx.to_array (Nx.slice [ Nx.R (a, b) ] all))
        (Nx.to_array some);
      let r = require_ok (T.ragged ~rows:(a, b) Nx.int32 "vla" (cat ())) in
      equal int (b - a) (Nx_ragged.length r))

let described () =
  let t = require_ok (T.of_hdu (cat ())) in
  let ra = List.find (fun (c : T.column) -> c.name = "ra") (T.columns t) in
  equal (list string) [ "TUNIT   = 'deg     '" ]
    (List.map String.trim (H.records ra.cards));
  equal (result bool string) (Ok true)
    (Result.map
       (fun u -> u = Some Ymir_units.Unit.degree)
       (T.unit "ra" (cat ())));
  expect (Format.asprintf "%a@.%a" T.pp t T.pp (require_ok (T.of_hdu (asc ()))))
  @@ __POS_OF__
       {|
    BINTABLE 6 rows, 20 columns, heap 153 bytes
       1 source_id            int64
       2 ra                   float64      deg
       3 mag                  float32      mag
       4 qual                 int16                TNULL -999
       5 u16                  uint16
       6 i8                   int8
       7 u32                  uint32
       8 u8                   uint8
       9 flag                 bool
      10 bits                 bit [11]
      11 name                 text [16]
      12 cplx                 complex64
      13 dcplx                complex128
      14 vec                  float32 [3]
      15 mat                  int32 [2; 3]
      16 grid                 text [4] [3]
      17 scaled               int16 scaled
      18 vla                  int32 list
      19 vld                  float64 list
      20 vlt                  text
    TABLE 6 rows, 5 columns, heap 0 bytes
       1 id                   int32                TNULL "-1"
       2 x                    float64
       3 y                    float64
       4 z                    float64
       5 s                    text [10]
    |}

(* An ASCII table built by hand: §7.2.5's implied decimals, exponents
   written as a bare sign, blank fields and TNULL. *)
let ascii_fields () =
  let row a b c = Printf.sprintf "%5s %6s %-4s" a b c in
  let rows =
    [ row "12345" "1.5-3" "NULL"; row "-7" "42" ""; row "" "2.5E2" "9" ]
  in
  let w = 17 in
  let data = String.concat "" rows in
  let h =
    H.(
      empty
      |> set V.string "XTENSION" "TABLE"
      |> set V.int "BITPIX" 8 |> set V.int "NAXIS" 2 |> set V.int "NAXIS1" w
      |> set V.int "NAXIS2" 3 |> set V.int "PCOUNT" 0 |> set V.int "GCOUNT" 1
      |> set V.int "TFIELDS" 3 |> set V.string "TTYPE1" "i"
      |> set V.string "TFORM1" "I5" |> set V.int "TBCOL1" 1
      |> set V.string "TTYPE2" "f"
      |> set V.string "TFORM2" "F6.2"
      |> set V.int "TBCOL2" 7 |> set V.string "TTYPE3" "n"
      |> set V.string "TFORM3" "I4" |> set V.int "TBCOL3" 14
      |> set V.string "TNULL3" "NULL")
  in
  let hdu =
    Fits.v h
      (Nx.init Nx.uint8
         [| String.length data |]
         (fun i -> Char.code data.[i.(0)]))
  in
  equal
    (result (array int32) string)
    (Ok [| 12345l; -7l; 0l |])
    (Result.map Nx.to_array (T.raw Nx.int32 "i" hdu));
  (* "1.5-3" is 1.5e-3; "42" has two implied decimals; "2.5E2" has a point *)
  equal
    (result (array float_exact) string)
    (Ok [| 1.5e-3; 0.42; 250. |])
    (Result.map Nx.to_array (T.raw Nx.float64 "f" hdu));
  equal
    (result (option (array bool)) string)
    (Ok (Some [| false; true; true |]))
    (Result.map (Option.map Nx.to_array) (T.validity "n" hdu));
  equal
    (result (array int) string)
    (Ok [| 0; 0; 9 |])
    (Result.map Nx.to_array (T.raw Nx.int16 "n" hdu))

(* Writing *)

let read_back hdus =
  let path = temp_file ~suffix:".fits" () in
  require_ok (Fits.write path hdus);
  let back = require_ok (Fits.read path) in
  List.iter (fun h -> equal (result unit string) (Ok ()) (Fits.verify h)) back;
  back

let bytes_of (Nx.P t) =
  Nx.to_array
    (Nx.bitcast Nx.uint8 (Nx.contiguous (Nx.reshape [| Nx.numel t |] t)))

let round_trip () =
  (* A table written reads back as its columns. *)
  (* Text grids are not written; the rest is. *)
  let cols =
    List.filter (fun (n, _, _) -> n <> "grid") (require_ok (T.read (cat ())))
  in
  let h = H.(empty |> set V.string "EXTNAME" "COPY") in
  let written = require_ok (T.hdu h cols) in
  let back = List.nth (read_back [ written ]) 1 in
  let cols' = require_ok (T.read back) in
  equal (list string)
    (List.map (fun (n, _, _) -> n) cols)
    (List.map (fun (n, _, _) -> n) cols');
  List.iter2
    (fun (n, c, d) (_, c', d') ->
      equal ~msg:n (list string) (H.records c) (H.records c');
      match (d, d') with
      | ( T.Array { values = Nx.P a; validity = va },
          T.Array { values = Nx.P b; validity = vb } ) ->
          equal ~msg:n (array int) (Nx.shape a) (Nx.shape b);
          (match (Nx.dtype a, Nx.dtype b) with
          | Bit, Bit ->
              equal ~msg:n (array bool)
                (Nx.to_array (Nx.cast Nx.bool a))
                (Nx.to_array (Nx.cast Nx.bool b))
          | Bool, Bool ->
              equal ~msg:n (array bool)
                (Nx.to_array (Nx.cast Nx.bool a))
                (Nx.to_array (Nx.cast Nx.bool b))
          | _ ->
              (* An undefined cell holds the writer's own marker. *)
              let defined x v =
                match v with
                | None -> x
                | Some v -> Nx.where (Nx.cast Nx.bool v) x (Nx.zeros_like x)
              in
              equal ~msg:n (array int)
                (bytes_of (Nx.P (defined a va)))
                (bytes_of (Nx.P (defined b vb))));
          equal ~msg:n
            (option (array bool))
            (Option.map Nx.to_array va)
            (Option.map Nx.to_array vb)
      | T.Lists { values = a; _ }, T.Lists { values = b; _ } ->
          equal ~msg:n (array int64)
            (Nx.to_array (Nx_ragged.lengths a))
            (Nx.to_array (Nx_ragged.lengths b));
          equal ~msg:n (array int)
            (bytes_of (Nx.P (Nx_ragged.values a)))
            (bytes_of (Nx.P (Nx_ragged.values b)))
      | T.Text a, T.Text b ->
          equal ~msg:n (array string) (strings_of a) (strings_of b)
      | _ -> fail (n ^ ": the column's kind changed"))
    cols cols';
  equal
    (result (option string) string)
    (Ok (Some "deg"))
    (H.find V.string "TUNIT2" (Fits.header back))

let nulls () =
  let v = Nx.create Nx.int16 [| 4 |] [| -32768; 1; 2; 3 |] in
  let valid =
    Nx.cast Nx.bit (Nx.create Nx.bool [| 4 |] [| true; false; true; true |])
  in
  let t =
    require_ok
      (T.hdu H.empty
         [ ("q", H.empty, T.Array { values = Nx.P v; validity = Some valid }) ])
  in
  (* -32768 is held by a valid cell, so TNULL is the next free value. *)
  equal (result string string) (Ok "-32767")
    (H.get V.text "TNULL1" (Fits.header t));
  equal
    (result (option (array bool)) string)
    (Ok (Some [| true; false; true; true |]))
    (Result.map (Option.map Nx.to_array) (T.validity "q" t));
  (* A validity without a false element reads back as none. *)
  let t =
    require_ok
      (T.hdu H.empty
         [
           ( "q",
             H.empty,
             T.Array
               { values = Nx.P v; validity = Some (Nx.ones Nx.bit [| 4 |]) } );
         ])
  in
  equal
    (result (option string) string)
    (Ok None)
    (H.find V.text "TNULL1" (Fits.header t));
  equal
    (result (option (array bool)) string)
    (Ok None)
    (Result.map (Option.map Nx.to_array) (T.validity "q" t));
  (* Every uint8 value held by a valid cell leaves no TNULL. *)
  let full = Nx.init Nx.uint8 [| 257 |] (fun i -> i.(0) mod 256) in
  let valid =
    Nx.cast Nx.bit (Nx.init Nx.bool [| 257 |] (fun i -> i.(0) < 256))
  in
  (match
     T.hdu H.empty
       [ ("b", H.empty, T.Array { values = Nx.P full; validity = Some valid }) ]
   with
  | Ok _ -> fail "a full uint8 column written"
  | Error e -> contains ~sub:"int16" e);
  (* NaN in a valid float cell reads back undefined. *)
  let f = Nx.create Nx.float32 [| 2 |] [| Float.nan; 1. |] in
  let t =
    require_ok
      (T.hdu H.empty
         [ ("f", H.empty, T.Array { values = Nx.P f; validity = None }) ])
  in
  equal
    (result (option (array bool)) string)
    (Ok (Some [| false; true |]))
    (Result.map (Option.map Nx.to_array) (T.validity "f" t))

let text_errors () =
  let r s =
    Nx_ragged.of_lengths
      (Nx.create Nx.int64 [| 1 |] [| Int64.of_int (String.length s) |])
      (Nx.init Nx.uint8 [| String.length s |] (fun i -> Char.code s.[i.(0)]))
  in
  is_error (T.hdu H.empty [ ("s", H.empty, T.Text (r "trailing ")) ]);
  is_error (T.hdu H.empty [ ("s", H.empty, T.Text (r "caf\xc3\xa9")) ]);
  is_ok (T.hdu H.empty [ ("s", H.empty, T.Text (r " lead")) ]);
  raises_match Exn.invalid_arg (fun () ->
      T.hdu H.empty
        [
          ( "a",
            H.empty,
            T.Array
              { values = Nx.P (Nx.zeros Nx.float32 [| 2 |]); validity = None }
          );
          ( "b",
            H.empty,
            T.Array
              { values = Nx.P (Nx.zeros Nx.float32 [| 3 |]); validity = None }
          );
        ]);
  raises_match Exn.invalid_arg (fun () ->
      T.hdu H.empty
        [
          ( "a",
            H.empty,
            T.Array
              { values = Nx.P (Nx.zeros Nx.float16 [| 2 |]); validity = None }
          );
        ])

(* Hostile bytes *)

let raw = lazy (In_channel.with_open_bin path In_channel.input_all)

let hostile =
  prop "a changed byte gives values or an Error"
    Gen.(
      pair (int_range 0 (String.length (Lazy.force raw) - 1)) (int_range 0 255))
    (fun (pos, byte) ->
      let raw = Lazy.force raw in
      let a =
        Bigarray.(Array1.create int8_unsigned c_layout (String.length raw))
      in
      String.iteri (fun i c -> Bigarray.Array1.unsafe_set a i (Char.code c)) raw;
      Bigarray.Array1.set a pos byte;
      match
        Fits.of_bytes ~name:"x" (Nx.of_bigarray (Bigarray.genarray_of_array1 a))
      with
      | Error _ -> collect "read fails"
      | Ok hdus ->
          List.iter
            (fun h ->
              match T.read h with
              | Ok _ -> collect "values"
              | Error _ -> collect "error")
            (List.tl hdus))

let () =
  exit
  @@ run "Fits.Table"
       [
         columns;
         test "physical values and validity" physical;
         test "exactness and lookup" exactness;
         rows_law;
         test "descriptions" described;
         test "ASCII fields" ascii_fields;
         group "writing"
           [
             test "round trip" round_trip;
             test "TNULL" nulls;
             test "text and caller errors" text_errors;
           ];
         hostile;
       ]
