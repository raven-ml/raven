(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Writes a FITS file through every writer path: plain and tiled images of
   each dtype, a quantized image, and a binary table. Each coded image has
   an uncompressed copy named SRC_<name> that check_written.py compares it
   with.

     dune build ./contrib/ymir/test/fits/gen/sample.exe
     ./_build/default/contrib/ymir/test/fits/gen/sample.exe out.fits
     uv run contrib/ymir/test/fits/gen/check_written.py out.fits *)

open Ymir_fits
module H = Fits.Header
module V = Fits.Value

let named n = H.(empty |> set V.string "EXTNAME" n)
let shape = [| 37; 53 |]
let f i = Float.of_int ((i.(0) * 53) + i.(1))

let images =
  let pair (type a b) n (t : (a, b) Nx.t) =
    [
      Fits.Image.hdu ~tiles:[| 8; 8 |] (named n) t;
      Fits.Image.hdu (named ("SRC_" ^ n)) t;
    ]
  in
  List.concat
    [
      pair "I16"
        (Nx.init Nx.int16 shape (fun i -> (i.(0) * 911) - (i.(1) * 37)));
      pair "U16" (Nx.init Nx.uint16 shape (fun i -> (i.(0) * 1777) + i.(1)));
      pair "U8" (Nx.init Nx.uint8 shape (fun i -> (i.(0) * 7) + i.(1)));
      pair "I8"
        (Nx.init Nx.int8 shape (fun i -> (((i.(0) * 7) + i.(1)) mod 256) - 128));
      pair "I32"
        (Nx.init Nx.int32 shape (fun i ->
             Int32.of_int ((i.(0) * 100003) - (i.(1) * 77777))));
      pair "U32"
        (Nx.init Nx.uint32 shape (fun i ->
             Int32.of_int ((i.(0) * 100003) + i.(1))));
      pair "I64"
        (Nx.init Nx.int64 shape (fun i ->
             Int64.of_int ((i.(0) * 1000000007) - i.(1))));
      pair "F32" (Nx.init Nx.float32 shape (fun i -> sin (f i) *. 100.));
      pair "F64" (Nx.init Nx.float64 shape (fun i -> cos (f i) *. 1e-3));
    ]

let quantized =
  let t =
    Nx.init Nx.float32 shape (fun i ->
        100. +. (5. *. sin (f i *. 12.9898) *. cos (f i *. 78.233)))
  in
  [ Fits.Image.quantized 16. (named "Q") t; Fits.Image.hdu (named "SRC_Q") t ]

let table =
  let n = 4 in
  let lists lengths values =
    Nx_ragged.of_lengths (Nx.create Nx.int64 [| n |] lengths) values
  in
  let text l =
    let s = String.concat "" l in
    lists
      (Array.of_list (List.map (fun x -> Int64.of_int (String.length x)) l))
      (Nx.init Nx.uint8 [| String.length s |] (fun i -> Char.code s.[i.(0)]))
  in
  let tunit u = H.(empty |> set V.string "TUNIT" u) in
  Fits.Table.
    [
      ( "id",
        H.empty,
        Array
          {
            values = Nx.P (Nx.create Nx.int64 [| n |] [| 1L; 2L; 3L; 4L |]);
            validity = None;
          } );
      ( "ra",
        tunit "deg",
        Array
          {
            values =
              Nx.P (Nx.create Nx.float64 [| n |] [| 0.; 90.5; 180.; 359.75 |]);
            validity = None;
          } );
      ( "q",
        H.empty,
        Array
          {
            values = Nx.P (Nx.create Nx.int16 [| n |] [| 1; 2; 3; 4 |]);
            validity =
              Some
                (Nx.cast Nx.bit
                   (Nx.create Nx.bool [| n |] [| true; false; true; true |]));
          } );
      ( "u16",
        H.empty,
        Array
          {
            values = Nx.P (Nx.create Nx.uint16 [| n |] [| 0; 1; 65535; 32768 |]);
            validity = None;
          } );
      ( "flag",
        H.empty,
        Array
          {
            values =
              Nx.P (Nx.create Nx.bool [| n |] [| true; false; true; false |]);
            validity = None;
          } );
      ( "bits",
        H.empty,
        Array
          {
            values =
              Nx.P
                (Nx.cast Nx.bit
                   (Nx.init Nx.bool [| n; 11 |] (fun i ->
                        (i.(0) + i.(1)) mod 3 = 0)));
            validity = None;
          } );
      ( "vec",
        H.empty,
        Array
          {
            values = Nx.P (Nx.init Nx.float32 [| n; 2; 3 |] (fun i -> f i));
            validity = None;
          } );
      ( "z",
        H.empty,
        Array
          {
            values =
              Nx.P
                (Nx.init Nx.complex64 [| n |] (fun i ->
                     {
                       Complex.re = Float.of_int i.(0);
                       im = -.Float.of_int i.(0);
                     }));
            validity = None;
          } );
      ("name", H.empty, Text (text [ "alpha"; ""; "  lead"; "z" ]));
      ( "vla",
        H.empty,
        Lists
          {
            values =
              lists [| 0L; 3L; 1L; 2L |]
                (Nx.init Nx.int32 [| 6 |] (fun i -> Int32.of_int i.(0)));
            validity = None;
          } );
    ]

let () =
  let table =
    match Fits.Table.hdu (named "TAB") table with
    | Ok t -> t
    | Error e -> failwith e
  in
  match Fits.write Sys.argv.(1) (images @ quantized @ [ table ]) with
  | Ok () -> ()
  | Error e ->
      prerr_endline e;
      exit 1
