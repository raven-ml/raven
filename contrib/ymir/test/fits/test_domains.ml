(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* HDUs shared between domains. A constructed HDU encodes itself when first
   asked, and a read HDU computes its file's digest when first asked; two
   domains asking at once both get the one value. The reference is the same
   HDU already computed. *)

open Windtrap
open Ymir_fits
module H = Fits.Header
module I = Fits.Image

let image () =
  Nx.init Nx.float32 [| 12; 20 |] (fun i ->
      100. +. sin (Float.of_int ((i.(0) * 20) + i.(1))))

let kinds = [| "plain"; "tiled"; "quantized" |]

let make k =
  match k with
  | 0 -> I.hdu H.empty (image ())
  | 1 -> I.hdu ~tiles:[| 4; 8 |] H.empty (image ())
  | _ -> I.quantized 4. H.empty (image ())

let read () = List.nth (require_ok (Fits.read "golden/tiles.fits")) 11
let hdu = abstract "h"

let kind =
  Gen.with_pp
    (fun ppf k -> Format.pp_print_string ppf kinds.(k))
    (Gen.int_range 0 2)

let bytes h =
  Digest.to_hex
    (Digest.string
       (String.concat ","
          (Array.to_list (Array.map string_of_int (Nx.to_array (Fits.data h))))))

let forced h =
  ignore (Fits.header h, Fits.data h, Fits.digest h);
  h

let commands =
  [
    command "construct" (kind @-> makes hdu) (fun k -> forced (make k)) make;
    command "read" (Gen.unit @-> makes hdu) (fun () -> forced (read ())) read;
    command "header"
      (hdu ^-> returns string)
      (fun h -> H.to_string (Fits.header h))
      (fun h -> H.to_string (Fits.header h));
    command "data" (hdu ^-> returns string) bytes bytes;
    command "digest" (hdu ^-> returns string) Fits.digest Fits.digest;
    command "values"
      (hdu ^-> returns (result (array float_exact) string))
      (fun h -> Result.map Nx.to_array (I.values Nx.float64 h))
      (fun h -> Result.map Nx.to_array (I.values Nx.float64 h));
  ]

let () =
  exit
  @@ run "Fits on domains"
       [
         stateful ~domains:2 ~count:10 ~steps:6
           "two domains force one HDU and get one value" commands;
       ]
