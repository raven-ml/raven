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
      let b = Bytes.of_string (Lazy.force raw) in
      Bytes.set b pos (Char.chr byte);
      let t =
        Nx.create Nx.uint8
          [| Bytes.length b |]
          (Array.init (Bytes.length b) (fun i -> Char.code (Bytes.get b i)))
      in
      match Fits.of_bytes ~name:"x" t with
      | Error _ -> collect "read fails"
      | Ok hdus ->
          List.iter
            (fun h ->
              match I.values Nx.float64 h with
              | Ok _ -> collect "values"
              | Error _ -> collect "error")
            hdus)

let () =
  exit
  @@ run "Fits tiles"
       [
         decoded;
         test "descriptions" described;
         test "validity" validity;
         group "windows" windows;
         hostile;
       ]
