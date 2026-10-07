(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Images: elements and the dtypes that hold them (the table of the
   interface, from FITS 4.0 Table 11), stored and physical values against
   astropy's reading of its own files (gen/fixtures.py), undefined pixels,
   windows, and the round trip through [Image.hdu] and [write]. *)

open Windtrap
open Ymir_fits
module H = Fits.Header
module V = Fits.Value
module I = Fits.Image
module S = Nx_dtype.Scalar

let images = "golden/images.fits"
let hdus () = require_ok (Fits.read images)
let hdu name = require_ok (Fits.get name (hdus ()))
let primary () = List.hd (hdus ())

let golden =
  lazy
    (In_channel.with_open_text "golden/images.values" In_channel.input_lines
    |> List.map (fun l ->
        match String.split_on_char ' ' l with
        | name :: values -> (name, Array.of_list values)
        | [] -> assert false))

let expected name = List.assoc name (Lazy.force golden)

(* A tensor's elements as the text astropy writes for integers, and as
   floats for floats. *)
let texts (type a b) (t : (a, b) Nx.t) : string array =
  let a = Nx.to_array t in
  match Nx.dtype t with
  | UInt8 -> Array.map string_of_int a
  | Int8 -> Array.map string_of_int a
  | Int16 -> Array.map string_of_int a
  | UInt16 -> Array.map string_of_int a
  | Int32 -> Array.map Int32.to_string a
  | UInt32 -> Array.map (Printf.sprintf "%lu") a
  | Int64 -> Array.map Int64.to_string a
  | UInt64 -> Array.map (Printf.sprintf "%Lu") a
  | _ -> failwith "texts: an integer dtype"

let floats name = Array.map float_of_string (expected name)

let check_raw (type a b) name (dtype : (a, b) Nx.dtype) =
  let t =
    require_ok (I.raw dtype (if name = "PRIMARY" then primary () else hdu name))
  in
  equal (array int) [| 5; 7 |] (Nx.shape t);
  match Nx_dtype.kind dtype with
  | Float ->
      equal (array float_exact) (floats name)
        (Nx.to_array (Nx.reshape [| 35 |] t))
  | _ -> equal (array string) (expected name) (texts (Nx.reshape [| 35 |] t))

(* Elements *)

let stored =
  cases ~name:fst "raw reads the stored numbers in their element"
    [
      ("PRIMARY", fun n -> check_raw n Nx.int16);
      ("U8", fun n -> check_raw n Nx.uint8);
      ("I8", fun n -> check_raw n Nx.int8);
      ("U16", fun n -> check_raw n Nx.uint16);
      ("U32", fun n -> check_raw n Nx.uint32);
      ("I32", fun n -> check_raw n Nx.int32);
      ("I64", fun n -> check_raw n Nx.int64);
      ("U64", fun n -> check_raw n Nx.uint64);
      ("F32", fun n -> check_raw n Nx.float32);
      ("F64", fun n -> check_raw n Nx.float64);
      ("NAN", fun n -> check_raw n Nx.float32);
    ]
    (fun (name, f) -> f name)

let elements () =
  let element name = Result.map I.element (I.of_hdu (hdu name)) in
  let w =
    result
      (Testable.make
         ~pp:(fun ppf s -> Format.pp_print_string ppf (S.to_string s))
         ~equal:S.equal)
      string
  in
  List.iter
    (fun (name, e) -> equal w (Ok e) (element name))
    [
      ("U8", S.UInt8);
      ("I8", Int8);
      ("U16", UInt16);
      ("U32", UInt32);
      ("I32", Int32);
      ("I64", Int64);
      ("U64", UInt64);
      ("F32", Float32);
      ("F64", Float64);
      ("SCALED", Int16);
    ];
  equal (result bool string) (Ok true)
    (Result.map I.scaled (I.of_hdu (hdu "SCALED")));
  equal (result bool string) (Ok false)
    (Result.map I.scaled (I.of_hdu (hdu "U64")));
  equal
    (result (array int) string)
    (Ok [| 2; 3; 4 |])
    (Result.map I.shape (I.of_hdu (hdu "CUBE")))

(* The interface's table: the real dtypes [raw] succeeds in, per element. *)
let table =
  [
    ( "U8",
      [
        "uint8";
        "int16";
        "uint16";
        "int32";
        "uint32";
        "int64";
        "uint64";
        "float16";
        "bfloat16";
        "float32";
        "float64";
      ] );
    ( "I8",
      [
        "int8";
        "int16";
        "int32";
        "int64";
        "float16";
        "bfloat16";
        "float32";
        "float64";
      ] );
    ("PRIMARY", [ "int16"; "int32"; "int64"; "float32"; "float64" ]);
    ( "U16",
      [ "uint16"; "int32"; "uint32"; "int64"; "uint64"; "float32"; "float64" ]
    );
    ("I32", [ "int32"; "int64"; "float64" ]);
    ("U32", [ "uint32"; "int64"; "uint64"; "float64" ]);
    ("I64", [ "int64" ]);
    ("U64", [ "uint64" ]);
    ("F32", [ "float32"; "float64" ]);
    ("F64", [ "float64" ]);
  ]

let real =
  [
    Nx.P (Nx.zeros Nx.int8 [| 0 |]);
    P (Nx.zeros Nx.uint8 [| 0 |]);
    P (Nx.zeros Nx.int16 [| 0 |]);
    P (Nx.zeros Nx.uint16 [| 0 |]);
    P (Nx.zeros Nx.int32 [| 0 |]);
    P (Nx.zeros Nx.uint32 [| 0 |]);
    P (Nx.zeros Nx.int64 [| 0 |]);
    P (Nx.zeros Nx.uint64 [| 0 |]);
    P (Nx.zeros Nx.float16 [| 0 |]);
    P (Nx.zeros Nx.bfloat16 [| 0 |]);
    P (Nx.zeros Nx.float32 [| 0 |]);
    P (Nx.zeros Nx.float64 [| 0 |]);
    P (Nx.zeros Nx.float8_e4m3 [| 0 |]);
    P (Nx.zeros Nx.bool [| 0 |]);
    P (Nx.zeros Nx.int4 [| 0 |]);
  ]

let exactness =
  cases ~name:fst "raw succeeds exactly in the dtypes that hold the element"
    table (fun (name, holds) ->
      let h = if name = "PRIMARY" then primary () else hdu name in
      List.iter
        (fun (Nx.P z) ->
          let d = Nx.dtype z in
          let ok = Result.is_ok (I.raw d h) in
          equal ~msg:(Nx_dtype.to_string d) bool
            (List.mem (Nx_dtype.to_string d) holds)
            ok)
        real)

let exactness_error () =
  match I.raw Nx.float32 (hdu "I32") with
  | Ok _ -> fail "int32 read as float32"
  | Error e ->
      equal string
        (images
       ^ ": HDU 5 (I32): int32 does not read exactly as float32; read it as \
          int32, int64 or float64")
        e

(* Values *)

let scaled () =
  let v = require_ok (I.values Nx.float64 (hdu "SCALED")) in
  equal (array float_exact) (floats "SCALED/values")
    (Nx.to_array (Nx.reshape [| 35 |] v));
  (* values d is cast d (values float64) on a scaled image. *)
  let v32 = require_ok (I.values Nx.float32 (hdu "SCALED")) in
  equal (array float_exact)
    (Nx.to_array (Nx.cast Nx.float32 v))
    (Nx.to_array v32);
  (* raw is the stored numbers, BLANK included. *)
  equal (array string) (expected "SCALED")
    (texts (Nx.reshape [| 35 |] (require_ok (I.raw Nx.int16 (hdu "SCALED")))))

let unscaled () =
  List.iter
    (fun name ->
      let r = require_ok (I.raw Nx.float64 (hdu name)) in
      let v = require_ok (I.values Nx.float64 (hdu name)) in
      equal (array float_exact) (Nx.to_array r) (Nx.to_array v))
    [ "U8"; "I8"; "U16"; "U32"; "I32"; "F32"; "F64"; "NAN" ];
  is_error (I.values Nx.float32 (hdu "I32"));
  is_error (I.values Nx.float64 (hdu "I64"))

let validity () =
  let mask name =
    Result.map
      (Option.map (fun m -> Nx.to_array (Nx.reshape [| 35 |] m)))
      (I.validity (hdu name))
  in
  let w = result (option (array bool)) string in
  equal w (Ok (Some (Array.init 35 (fun i -> i <> 3 && i <> 17)))) (mask "NAN");
  equal w
    (Ok (Some (Array.init 35 (fun i -> i <> 0 && i <> 9))))
    (mask "SCALED");
  equal w (Ok None) (mask "I32");
  equal w (Ok None) (mask "F64")

(* Windows *)

let window_gen shape =
  Gen.(
    let axis n =
      map
        (fun (a, b) -> (Int.min a b, Int.max a b))
        (pair (int_range 0 n) (int_range 0 n))
    in
    let some =
      match shape with
      | [| a; b |] -> map (fun (x, y) -> [| x; y |]) (pair (axis a) (axis b))
      | [| a; b; c |] ->
          map
            (fun (x, y, z) -> [| x; y; z |])
            (triple (axis a) (axis b) (axis c))
      | _ -> assert false
    in
    frequency [ (1, constant (Array.map (fun n -> (0, n)) shape)); (9, some) ])

let windows =
  let law name shape =
    prop (name ^ ": a window reads as the slice of the whole")
      (window_gen shape) (fun w ->
        let empty = Array.exists (fun (a, b) -> a = b) w in
        cover "empty" empty;
        cover "whole" (Array.for_all2 (fun (a, b) n -> a = 0 && b = n) w shape);
        cover "one pixel" (Array.for_all (fun (a, b) -> b = a + 1) w);
        let whole = require_ok (I.raw Nx.int64 (hdu name)) in
        let part = require_ok (I.raw ~window:w Nx.int64 (hdu name)) in
        let sliced =
          Nx.slice
            (Array.to_list (Array.map (fun (a, b) -> Nx.R (a, b)) w))
            whole
        in
        equal (array int) (Nx.shape sliced) (Nx.shape part);
        equal (array int64) (Nx.to_array sliced) (Nx.to_array part);
        if Array.length w = 2 then
          let v = require_ok (I.values ~window:w Nx.float64 (hdu "SCALED")) in
          equal (array float_exact)
            (Nx.to_array
               (Nx.slice
                  (Array.to_list (Array.map (fun (a, b) -> Nx.R (a, b)) w))
                  (require_ok (I.values Nx.float64 (hdu "SCALED")))))
            (Nx.to_array v))
  in
  [ law "U32" [| 5; 7 |]; law "CUBE" [| 2; 3; 4 |] ]

let window_errors () =
  let h = hdu "U16" in
  is_error (I.raw ~window:[| (0, 1) |] Nx.uint16 h);
  is_error (I.raw ~window:[| (0, 6); (0, 7) |] Nx.uint16 h);
  is_error (I.raw ~window:[| (0, 5); (0, 8) |] Nx.uint16 h);
  raises_match Exn.invalid_arg (fun () ->
      I.raw ~window:[| (3, 2); (0, 7) |] Nx.uint16 h);
  raises_match Exn.invalid_arg (fun () ->
      I.raw ~window:[| (-1, 2); (0, 7) |] Nx.uint16 h)

let not_images () =
  is_error
    (I.of_hdu
       (Fits.v
          H.(
            empty
            |> set V.string "XTENSION" "BINTABLE"
            |> set V.int "BITPIX" 8 |> set V.int "NAXIS" 2
            |> set V.int "NAXIS1" 0 |> set V.int "NAXIS2" 0
            |> set V.int "PCOUNT" 0 |> set V.int "GCOUNT" 1
            |> set V.int "TFIELDS" 0)
          (Nx.zeros Nx.uint8 [| 0 |])));
  let empty =
    Fits.v
      H.(
        empty |> set V.bool "SIMPLE" true |> set V.int "BITPIX" 8
        |> set V.int "NAXIS" 0)
      (Nx.zeros Nx.uint8 [| 0 |])
  in
  is_error (I.of_hdu empty)

(* Writing *)

let round_trip (type a b) ?(name = "") (dtype : (a, b) Nx.dtype)
    (t : (a, b) Nx.t) =
  let path = temp_file ~suffix:".fits" () in
  let h = H.(empty |> set V.string "EXTNAME" "X") in
  require_ok (Fits.write path [ I.hdu h t ]);
  let back = require_ok (Fits.read path) in
  List.iter
    (fun hdu -> equal ~msg:name (result unit string) (Ok ()) (Fits.verify hdu))
    back;
  let r = require_ok (I.raw dtype (List.hd back)) in
  equal (array int) (Nx.shape t) (Nx.shape r);
  (* Bit for bit: compare the bytes. *)
  let bytes x = Nx.to_array (Nx.bitcast Nx.uint8 (Nx.contiguous x)) in
  equal (array int) (bytes t) (bytes r)

let shape_gen = Gen.(array ~size:(int_range 1 3) (int_range 0 6))

let dtypes_round_trip =
  let law (type a b) (dtype : (a, b) Nx.dtype) =
    prop
      (Nx_dtype.to_string dtype ^ " reads back bit for bit")
      Gen.(pair shape_gen (list ~size:(int_range 0 300) int64))
      (fun (shape, bits) ->
        let n = Array.fold_left ( * ) 1 shape in
        let w = Nx_dtype.itemsize dtype in
        let bits = Array.of_list bits in
        (* Element bytes drawn from random 64-bit words, NaN payloads
           included. *)
        let byte i =
          if Array.length bits = 0 then 0
          else
            Int64.(
              to_int
                (logand
                   (shift_right_logical
                      bits.(i / 8 mod Array.length bits)
                      (8 * (i mod 8)))
                   0xFFL))
        in
        let u = Nx.init Nx.uint8 [| n * w |] (fun i -> byte i.(0)) in
        let t =
          Nx.reshape shape
            (if w = 1 then Nx.bitcast dtype u
             else Nx.bitcast dtype (Nx.reshape [| n; w |] u))
        in
        cover "empty" (n = 0);
        round_trip ~name:(Nx_dtype.to_string dtype) dtype t)
  in
  [
    law Nx.uint8;
    law Nx.int8;
    law Nx.int16;
    law Nx.uint16;
    law Nx.int32;
    law Nx.uint32;
    law Nx.int64;
    law Nx.uint64;
    law Nx.float32;
    law Nx.float64;
  ]

let unwritable () =
  raises_match (Exn.invalid_arg ~substring:"uint8") (fun () ->
      I.hdu H.empty (Nx.zeros Nx.bool [| 2 |]));
  raises_match (Exn.invalid_arg ~substring:"float32") (fun () ->
      I.hdu H.empty (Nx.zeros Nx.float16 [| 2 |]));
  raises_match (Exn.invalid_arg ~substring:"int8") (fun () ->
      I.hdu H.empty (Nx.zeros Nx.int4 [| 2 |]));
  raises_match Exn.invalid_arg (fun () ->
      I.hdu H.empty (Nx.zeros Nx.complex64 [| 2 |]));
  raises_match Exn.invalid_arg (fun () ->
      I.hdu H.empty (Nx.scalar Nx.float32 1.))

let structure () =
  (* The writer owns BITPIX, NAXISn, BZERO and BSCALE: a header read from a
     scaled int16 file heads a float32 image. *)
  let h = Fits.header (hdu "SCALED") in
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let x = I.hdu h t in
  let xh = Fits.header x in
  equal (result int string) (Ok (-32)) (H.get V.int "BITPIX" xh);
  equal (result (option string) string) (Ok None) (H.find V.text "BSCALE" xh);
  equal (result (option string) string) (Ok None) (H.find V.text "BLANK" xh);
  equal (result string string) (Ok "SCALED") (H.get V.string "EXTNAME" xh);
  let back = require_ok (I.values Nx.float32 x) in
  equal (array float_exact) (Nx.to_array t) (Nx.to_array back)

let write_file () =
  let path = temp_file ~suffix:".fits" () in
  let all = hdus () in
  let edit h =
    Fits.with_header (H.set V.string "OBJECT" "NGC 346" (Fits.header h)) h
  in
  require_ok (Fits.write path (List.map edit all));
  let back = require_ok (Fits.read path) in
  equal int (List.length all) (List.length back);
  List.iter2
    (fun a b ->
      equal (result unit string) (Ok ()) (Fits.verify b);
      equal (array int) (Nx.to_array (Fits.data a)) (Nx.to_array (Fits.data b));
      (* Every record the program did not touch is written as read, the
         checksums aside. *)
      let untouched h =
        List.filter
          (fun r ->
            not
              (List.mem (String.sub r 0 8)
                 [ "OBJECT  "; "CHECKSUM"; "DATASUM " ]))
          (H.records h)
      in
      equal (list string)
        (untouched (Fits.header a))
        (untouched (Fits.header b)))
    all back;
  (* Output is a function of input. *)
  let path' = temp_file ~suffix:".fits" () in
  require_ok (Fits.write path' (List.map edit all));
  let read p = In_channel.with_open_bin p In_channel.input_all in
  equal string
    (Digest.to_hex (Digest.string (read path)))
    (Digest.to_hex (Digest.string (read path')))

let verify_reference () =
  List.iter
    (fun h -> equal (result unit string) (Ok ()) (Fits.verify h))
    (hdus ())

let corrupt () =
  let raw = In_channel.with_open_bin images In_channel.input_all in
  let b = Bytes.of_string raw in
  (* A byte of HDU 1's data unit, which starts at 8640. *)
  Bytes.set b 8641 'X';
  let path = temp_file ~suffix:".fits" () in
  Out_channel.with_open_bin path (fun oc -> Out_channel.output_bytes oc b);
  let all = require_ok (Fits.read path) in
  let u8 = List.nth all 1 in
  is_error (Fits.verify u8);
  (match Fits.write (temp_file ~suffix:".fits" ()) all with
  | Ok () -> fail "a corrupt HDU written"
  | Error e -> contains ~sub:"Fits.v (Header.remove \"DATASUM\"" e);
  let fixed = Fits.v (H.remove "DATASUM" (Fits.header u8)) (Fits.data u8) in
  is_ok (Fits.write (temp_file ~suffix:".fits" ()) [ List.hd all; fixed ])

let empty_primary () =
  (* A file whose first HDU is not an image gets an empty primary. *)
  let path = temp_file ~suffix:".fits" () in
  let t =
    Fits.v
      H.(
        empty
        |> set V.string "XTENSION" "BINTABLE"
        |> set V.int "BITPIX" 8 |> set V.int "NAXIS" 2 |> set V.int "NAXIS1" 0
        |> set V.int "NAXIS2" 0 |> set V.int "PCOUNT" 0 |> set V.int "GCOUNT" 1
        |> set V.int "TFIELDS" 0)
      (Nx.zeros Nx.uint8 [| 0 |])
  in
  require_ok (Fits.write path [ t ]);
  let back = require_ok (Fits.read path) in
  equal int 2 (List.length back);
  equal (result int string) (Ok 0)
    (H.get V.int "NAXIS" (Fits.header (List.hd back)))

let () =
  exit
  @@ run "Fits.Image"
       [
         group "elements"
           [
             stored;
             test "elements" elements;
             exactness;
             test "the error names the dtypes" exactness_error;
           ];
         group "values"
           [
             test "scaled" scaled;
             test "unscaled" unscaled;
             test "validity" validity;
           ];
         group "windows" (windows @ [ test "errors" window_errors ]);
         test "other kinds" not_images;
         group "writing"
           (dtypes_round_trip
           @ [
               test "unwritable dtypes" unwritable;
               test "structure is the writer's" structure;
               test "write keeps what it did not change" write_file;
               test "astropy's checksums verify" verify_reference;
               test "a corrupt data unit" corrupt;
               test "empty primary" empty_primary;
             ]);
       ]
