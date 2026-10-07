(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Files the archives wrote (survey/README.md) read as astropy reads them
   (gen/survey.py): each file holds the HDUs astropy finds, each HDU whose
   CHECKSUM astropy accepts verifies, and each image and numeric
   binary-table column astropy decodes has its shape and float64 values.
   Values compare as digests: MD5 of little-endian float64, every NaN as
   one NaN. *)

open Windtrap
open Ymir_fits
module I = Fits.Image
module T = Fits.Table

let expected =
  In_channel.with_open_text "survey/values" In_channel.input_lines
  |> List.map (String.split_on_char ' ')

let ok = function Ok x -> x | Error e -> fail e
let hdus file = ok (Fits.read ("survey/" ^ file ^ ".fits"))
let nth file i = List.nth (hdus file) (int_of_string i)

let digest t =
  let n = Nx.numel t in
  let a : float array = Nx.to_array (Nx.reshape [| n |] t) in
  let b = Bytes.create (8 * n) in
  Array.iteri
    (fun i x ->
      let bits =
        if Float.is_nan x then 0x7FF8000000000000L else Int64.bits_of_float x
      in
      Bytes.set_int64_le b (8 * i) bits)
    a;
  Digest.to_hex (Digest.bytes b)

let shape t =
  String.concat "x" (Array.to_list (Array.map string_of_int (Nx.shape t)))

let check_values (shape', digest') t =
  equal string shape' (shape t);
  equal string digest' (digest t)

let check = function
  | [ file; "hdus"; n ] -> equal int (int_of_string n) (List.length (hdus file))
  | [ file; i; "verify" ] -> ok (Fits.verify (nth file i))
  | [ file; i; "image"; s; d ] ->
      check_values (s, d) (ok (I.values Nx.float64 (nth file i)))
  | [ file; i; "column"; k; s; d ] ->
      let hdu = nth file i in
      let c = List.nth (T.columns (ok (T.of_hdu hdu))) (int_of_string k - 1) in
      check_values (s, d) (ok (T.values Nx.float64 c.name hdu))
  | l -> failf "survey/values: %s" (String.concat " " l)

let agree =
  cases
    ~name:(fun l -> String.concat " " (List.filteri (fun i _ -> i < 4) l))
    "real files read as astropy reads them" expected check

let () = exit @@ run "Fits survey files" [ agree ]
