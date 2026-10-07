(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tile-compressed images: Rice, gzip and uncompressed tiles of integers and
   floats, quantized floats under each dither, read against astropy's
   reading of its own files (gen/fixtures.py); windows that meet some tiles;
   and hostile bytes, which give values or an Error. *)

open Windtrap
open Ymir_fits
module H = Fits.Header
module V = Fits.Value
module I = Fits.Image
module S = Nx_dtype.Scalar

let path = "golden/tiles.fits"
let hdus = lazy (require_ok (Fits.read path))
let hdu name = require_ok (Fits.get name (Lazy.force hdus))

let golden =
  lazy
    (In_channel.with_open_text "golden/tiles.values" In_channel.input_lines
    |> List.map (fun l ->
        match String.split_on_char ' ' l with
        | name :: values -> (name, Array.of_list values)
        | [] -> assert false))

let expected name = List.assoc name (Lazy.force golden)

let texts (type a b) (t : (a, b) Nx.t) : string array =
  let a = Nx.to_array t in
  match Nx.dtype t with
  | UInt8 -> Array.map string_of_int a
  | Int16 -> Array.map string_of_int a
  | UInt16 -> Array.map string_of_int a
  | Int32 -> Array.map Int32.to_string a
  | UInt32 -> Array.map (Printf.sprintf "%lu") a
  | _ -> failwith "texts: an integer dtype"

let check (type a b) name (dtype : (a, b) Nx.dtype) =
  let t = require_ok (I.raw dtype (hdu name)) in
  let n = Nx.numel t in
  let flat = Nx.reshape [| n |] t in
  match Nx_dtype.kind dtype with
  | Float ->
      equal (array float_exact)
        (Array.map float_of_string (expected name))
        (Nx.to_array flat)
  | _ -> equal (array string) (expected name) (texts flat)

let decoded =
  cases ~name:fst "tiles decode as astropy decodes them"
    [
      ("RICE_I16", fun n -> check n Nx.int16);
      ("RICE_U16", fun n -> check n Nx.uint16);
      ("RICE_U8", fun n -> check n Nx.uint8);
      ("RICE_I32", fun n -> check n Nx.int32);
      ("GZIP1_I32", fun n -> check n Nx.int32);
      ("GZIP2_I32", fun n -> check n Nx.int32);
      ("GZIP2_U32", fun n -> check n Nx.uint32);
      ("NOCOMP_I16", fun n -> check n Nx.int16);
      ("GZIP2_F32", fun n -> check n Nx.float32);
      ("GZIP1_F64", fun n -> check n Nx.float64);
      ("Q_DITHER1", fun n -> check n Nx.float32);
      ("Q_DITHER2", fun n -> check n Nx.float32);
      ("Q_NODITHER", fun n -> check n Nx.float32);
      ("Q_GZIP", fun n -> check n Nx.float32);
      ("Q_F64", fun n -> check n Nx.float64);
      ("CUBE", fun n -> check n Nx.int16);
    ]
    (fun (name, f) -> f name)

let described () =
  let pp name = Format.asprintf "%a" Fits.pp (hdu name) in
  expect
    (String.concat "\n"
       (List.map pp
          [
            "RICE_I16";
            "RICE_U16";
            "GZIP2_F32";
            "Q_DITHER2";
            "Q_NODITHER";
            "CUBE";
          ]))
  @@ __POS_OF__
       {|
    BINTABLE RICE_I16 1: int16 [10; 13], RICE_1 tiles [3; 4]
    BINTABLE RICE_U16 1: uint16 [10; 13], RICE_1 tiles [3; 4]
    BINTABLE GZIP2_F32 1: float32 [10; 13], GZIP_2 tiles [3; 4]
    BINTABLE Q_DITHER2 1: float32 [10; 13], RICE_1 tiles [3; 4], quantized (SUBTRACTIVE_DITHER_2)
    BINTABLE Q_NODITHER 1: float32 [10; 13], RICE_1 tiles [3; 4], quantized (NO_DITHER)
    BINTABLE CUBE 1: int16 [2; 5; 7], RICE_1 tiles [1; 2; 3]
    |}

let validity () =
  let mask name =
    Result.map
      (Option.map (fun m -> Nx.to_array (Nx.reshape [| 130 |] m)))
      (I.validity (hdu name))
  in
  let w = result (option (array bool)) string in
  let nan = Array.init 130 (fun i -> i <> 4 && i <> 50) in
  equal w (Ok (Some nan)) (mask "Q_DITHER1");
  equal w (Ok None) (mask "RICE_I16")

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

let slice w t =
  Nx.slice (Array.to_list (Array.map (fun (a, b) -> Nx.R (a, b)) w)) t

let windows =
  let law name shape =
    prop (name ^ ": a window reads as the slice of the whole")
      (window_gen shape) (fun w ->
        cover "empty" (Array.exists (fun (a, b) -> a = b) w);
        cover "inside one tile"
          (Array.for_all (fun (a, b) -> b > a && a / 3 = (b - 1) / 3) w);
        let whole = require_ok (I.values Nx.float64 (hdu name)) in
        let part = require_ok (I.values ~window:w Nx.float64 (hdu name)) in
        equal (array int) (Nx.shape (slice w whole)) (Nx.shape part);
        equal (array float_exact)
          (Nx.to_array (slice w whole))
          (Nx.to_array part))
  in
  [
    law "RICE_I16" [| 10; 13 |];
    law "Q_DITHER2" [| 10; 13 |];
    law "GZIP2_U32" [| 10; 13 |];
    law "CUBE" [| 2; 5; 7 |];
  ]

(* Hostile bytes *)

let raw = lazy (In_channel.with_open_bin path In_channel.input_all)

let hostile =
  prop "a changed byte gives values or an Error"
    Gen.(pair (int_range 0 (106560 - 1)) (int_range 0 255))
    (fun (pos, byte) ->
      let raw = Lazy.force raw in
      let a =
        Bigarray.(Array1.create int8_unsigned c_layout (String.length raw))
      in
      String.iteri (fun i c -> Bigarray.Array1.unsafe_set a i (Char.code c)) raw;
      Bigarray.Array1.set a pos byte;
      let t = Nx.of_bigarray (Bigarray.genarray_of_array1 a) in
      match Fits.of_bytes ~name:"x" t with
      | Error _ -> collect "read fails"
      | Ok hdus ->
          List.iter
            (fun h ->
              match I.values Nx.float64 h with
              | Ok _ -> collect "values"
              | Error _ -> collect "error")
            hdus)

(* Writing *)

let quantizer () =
  (* ymir quantizes as cfitsio's C source states: each tile's ZSCALE is the
     reference noise (gen/fixtures.py, exact arithmetic), and where
     astropy's build computes the same noise, ymir's tiles decode to
     astropy's floats, so the zero point, dither and rounding agree. *)
  let src = require_ok (I.raw Nx.float32 (hdu "QSRC")) in
  let ours = I.quantized 16. H.empty src in
  equal (result int string)
    (H.get V.int "ZDITHER0" (Fits.header (hdu "QREF")))
    (H.get V.int "ZDITHER0" (Fits.header ours));
  let bytes = Nx.to_array (Fits.data ours) in
  let zscale r =
    let b = ref 0L in
    for j = 0 to 7 do
      b :=
        Int64.logor (Int64.shift_left !b 8)
          (Int64.of_int bytes.((r * 32) + 16 + j))
    done;
    Int64.float_of_bits !b
  in
  equal (array float_exact)
    (Array.map float_of_string (expected "QREF/zscale"))
    (Array.init 10 zscale);
  let back = Nx.to_array (require_ok (I.values Nx.float32 ours)) in
  let theirs = Array.map float_of_string (expected "QREF") in
  let same = expected "QREF/same" in
  Array.iteri
    (fun r s ->
      if s = "1" then
        equal (array float_exact)
          (Array.sub theirs (r * 13) 13)
          (Array.sub back (r * 13) 13))
    same

let quantized_nan () =
  let t =
    Nx.create Nx.float32 [| 2; 10 |]
      (Array.init 20 (fun i ->
           if i = 3 then Float.nan else Float.of_int (i * i)))
  in
  let h = I.quantized 4. H.empty t in
  equal
    (option (array bool))
    (Some (Array.init 20 (fun i -> i <> 3)))
    (Option.map
       (fun m -> Nx.to_array (Nx.reshape [| 20 |] m))
       (require_ok (I.validity h)));
  (* A tile without noise stays lossless. *)
  let flat = Nx.full Nx.float32 [| 3; 12 |] 2.5 in
  equal (array float_exact) (Nx.to_array flat)
    (Nx.to_array
       (require_ok (I.values Nx.float32 (I.quantized 4. H.empty flat))))

let lossless =
  let law (type a b) (dtype : (a, b) Nx.dtype) =
    prop
      (Nx_dtype.to_string dtype ^ " tiles read back bit for bit")
      Gen.(
        triple
          (array ~size:(int_range 1 3) (int_range 0 6))
          (array ~size:(int_range 3 3) (int_range 1 4))
          (list ~size:(int_range 1 40) int64))
      (fun (shape, tiles, bits) ->
        let n = Array.fold_left ( * ) 1 shape in
        let w = Nx_dtype.itemsize dtype in
        let bits = Array.of_list bits in
        let byte i =
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
        let tiles = Array.sub tiles 0 (Array.length shape) in
        cover "partial tiles"
          (Array.exists2 (fun t s -> s mod t <> 0) tiles shape);
        let r = require_ok (I.raw dtype (I.hdu ~tiles H.empty t)) in
        let bytes x = Nx.to_array (Nx.bitcast Nx.uint8 (Nx.contiguous x)) in
        equal (array int) (Nx.shape t) (Nx.shape r);
        equal (array int) (bytes t) (bytes r))
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

let written () =
  (* Through a file: checksums verify and every codec reads back. *)
  let t =
    Nx.init Nx.int16 [| 7; 11 |] (fun i -> (i.(0) * 1000) - (i.(1) * 37))
  in
  let f =
    Nx.init Nx.float32 [| 7; 11 |] (fun i ->
        Float.of_int ((i.(0) * 3) + i.(1)) *. 0.25)
  in
  let path = temp_file ~suffix:".fits" () in
  require_ok
    (Fits.write path
       [
         I.hdu ~tiles:[| 2; 3 |] H.(empty |> set V.string "EXTNAME" "INT") t;
         I.hdu ~tiles:[| 7; 4 |]
           H.(
             empty
             |> set V.string "EXTNAME" "FLT"
             |> set V.string "BUNIT" "MJy/sr")
           f;
         I.quantized 8. H.(empty |> set V.string "EXTNAME" "Q") f;
       ]);
  let back = require_ok (Fits.read path) in
  equal int 4 (List.length back);
  List.iter (fun h -> equal (result unit string) (Ok ()) (Fits.verify h)) back;
  let get n = require_ok (Fits.get n back) in
  equal (array int) (Nx.to_array t)
    (Nx.to_array (require_ok (I.raw Nx.int16 (get "INT"))));
  equal (array float_exact) (Nx.to_array f)
    (Nx.to_array (require_ok (I.raw Nx.float32 (get "FLT"))));
  equal (result string string) (Ok "MJy/sr")
    (H.get V.string "BUNIT" (Fits.header (get "FLT")));
  let q = require_ok (I.values Nx.float32 (get "Q")) in
  less float_exact ~than:0.25 (Nx.item [] (Nx.max (Nx.abs (Nx.sub q f))))

let () =
  exit
  @@ run "Fits tiles"
       [
         decoded;
         test "descriptions" described;
         test "validity" validity;
         group "windows" windows;
         hostile;
         group "writing"
           ([
              test "the quantizer is cfitsio's" quantizer;
              test "undefined and flat tiles" quantized_nan;
              test "through a file" written;
            ]
           @ lossless);
       ]
