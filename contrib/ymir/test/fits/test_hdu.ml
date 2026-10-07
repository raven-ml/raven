(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* HDUs and files: walking a file reads every header and checks every
   extent, lookup by EXTNAME and EXTVER, the digest of the headers, and the
   structural failures [read] alone reports. Expected sizes follow FITS 4.0
   Eq. 1, 2 and 4; the reference file and its digest come from astropy
   (gen/fixtures.py). *)

open Windtrap
open Ymir_fits
module H = Fits.Header
module V = Fits.Value

let images = "golden/images.fits"

let names =
  [
    "PRIMARY";
    "U8";
    "I8";
    "U16";
    "U32";
    "I32";
    "I64";
    "U64";
    "F32";
    "F64";
    "NAN";
    "SCALED";
    "CUBE";
  ]

let file_bytes path = In_channel.with_open_bin path In_channel.input_all

let with_file contents f =
  let path = temp_file ~suffix:".fits" () in
  Out_channel.with_open_bin path (fun oc -> output_string oc contents);
  f path

let extname hdu =
  match H.find V.string "EXTNAME" (Fits.header hdu) with
  | Ok (Some n) -> n
  | _ -> "PRIMARY"

(* Reading *)

let read_all () =
  let hdus = require_ok (Fits.read images) in
  equal (list string) names (List.map extname hdus);
  List.iter (fun h -> equal string images (Fits.name h)) hdus

let data_sizes () =
  let hdus = require_ok (Fits.read images) in
  let size h = Nx.numel (Fits.data h) in
  (* |BITPIX|/8 × NAXIS1 × NAXIS2: 7 × 5 pixels *)
  equal (list int)
    [ 70; 35; 35; 70; 140; 140; 280; 280; 140; 280; 140; 70; 96 ]
    (List.map size hdus)

let data_is_the_file () =
  let hdus = require_ok (Fits.read images) in
  let raw = file_bytes images in
  let u8 = require_ok (Fits.get "U8" hdus) in
  let d = Nx.to_array (Fits.data u8) in
  (* HDU 1's data unit starts after the primary (header and padded data)
     and HDU 1's own header: one block each. *)
  let start = 2880 * 3 in
  equal (array int) (Array.init 35 (fun i -> Char.code raw.[start + i])) d

let of_bytes () =
  let raw = file_bytes images in
  let t =
    Nx.create Nx.uint8
      [| String.length raw |]
      (Array.init (String.length raw) (fun i -> Char.code raw.[i]))
  in
  let a = require_ok (Fits.read images)
  and b = require_ok (Fits.of_bytes ~name:"copy" t) in
  equal (list string)
    (List.map (fun h -> H.to_string (Fits.header h)) a)
    (List.map (fun h -> H.to_string (Fits.header h)) b);
  equal (list string) (List.map Fits.digest a) (List.map Fits.digest b);
  equal string "copy" (Fits.name (List.hd b))

(* Lookup *)

let lookup () =
  let hdus = require_ok (Fits.read images) in
  equal (result string string) (Ok "U16")
    (Result.map extname (Fits.get "U16" hdus));
  equal (result string string) (Ok "U16")
    (Result.map extname (Fits.get ~ver:1 "U16" hdus));
  is_error (Fits.get ~ver:2 "U16" hdus);
  match Fits.get "u16" hdus with
  | Ok _ -> fail "names compare exactly"
  | Error e -> contains ~sub:"U16" e

let several () =
  let h n v = H.(empty |> set V.string "EXTNAME" n |> set V.int "EXTVER" v) in
  let img n v = Fits.Image.hdu (h n v) (Nx.zeros Nx.uint8 [| 1 |]) in
  let hdus = [ img "SCI" 1; img "SCI" 2; img "ERR" 1 ] in
  is_error (Fits.get "SCI" hdus);
  equal (result int string) (Ok 2)
    (Result.bind (Fits.get ~ver:2 "SCI" hdus) (fun h ->
         H.get V.int "EXTVER" (Fits.header h)))

(* The digest *)

let digest () =
  let expected = String.trim (file_bytes "golden/images.digest") in
  let hdus = require_ok (Fits.read images) in
  List.iter (fun h -> equal string expected (Fits.digest h)) hdus

(* Structure fails in [read] *)

let cut n = String.sub (file_bytes images) 0 n

let truncated () =
  let raw = file_bytes images in
  (* HDU 1 (U8): header at 5760, data 8640-8674. *)
  with_file (cut 8650) (fun path ->
      match Fits.read path with
      | Ok _ -> fail "a cut data unit read"
      | Error e ->
          equal string
            (path
           ^ ": HDU 1 (U8), bytes 8640-8674: the data unit ends past the end \
              of the file, at byte 8650")
            e);
  (* Cut inside the next HDU's header, which starts XTENSION. *)
  with_file
    (cut ((2880 * 4) + 100))
    (fun path ->
      match Fits.read path with
      | Ok _ -> fail "a cut header read"
      | Error e -> contains ~sub:"HDU 2" e);
  (* Cut inside the last data unit's padding: the data is whole. *)
  let last = String.length raw - 2880 + 96 in
  with_file
    (cut (last + 10))
    (fun path -> equal int 13 (List.length (require_ok (Fits.read path))));
  (* Bytes after the last HDU that do not start XTENSION. *)
  with_file
    (raw ^ String.make 100 '\000')
    (fun path -> equal int 13 (List.length (require_ok (Fits.read path))))

let gzip () =
  with_file
    ("\x1f\x8b" ^ String.make 100 '\000')
    (fun path ->
      match Fits.read path with
      | Ok _ -> fail "a gzip stream read"
      | Error e -> contains ~sub:"Nx_io.gunzip" e)

let not_fits () =
  with_file (String.make 2880 ' ') (fun path -> is_error (Fits.read path));
  with_file "" (fun path -> is_error (Fits.read path));
  is_error (Fits.read "golden/does-not-exist.fits")

let bad_structure () =
  let pad r = r ^ String.make (80 - String.length r) ' ' in
  let file records data =
    let h = String.concat "" (List.map pad (records @ [ "END" ])) in
    let h = h ^ String.make (2880 - String.length h) ' ' in
    h ^ data
  in
  let check name records =
    with_file
      (file records (String.make 2880 '\000'))
      (fun path ->
        match Fits.read path with
        | Ok _ -> failf "%s read" name
        | Error e -> contains ~sub:"card" e)
  in
  check "BITPIX 12" [ "SIMPLE  = T"; "BITPIX  = 12"; "NAXIS   = 0" ];
  check "NAXIS -1" [ "SIMPLE  = T"; "BITPIX  = 8"; "NAXIS   = -1" ];
  check "NAXIS 1000" [ "SIMPLE  = T"; "BITPIX  = 8"; "NAXIS   = 1000" ];
  check "NAXIS1 -3"
    [ "SIMPLE  = T"; "BITPIX  = 8"; "NAXIS   = 1"; "NAXIS1  = -3" ];
  check "SIMPLE F" [ "SIMPLE  = F"; "BITPIX  = 8"; "NAXIS   = 0" ];
  (* A size past int's range. *)
  with_file
    (file
       [
         "SIMPLE  = T";
         "BITPIX  = -64";
         "NAXIS   = 3";
         "NAXIS1  = 4611686018427387903";
         "NAXIS2  = 4611686018427387903";
         "NAXIS3  = 7";
       ]
       "")
    (fun path ->
      match Fits.read path with
      | Ok _ -> fail "overflow read"
      | Error e -> contains ~sub:"overflow" e);
  (* Random groups: |BITPIX|/8 × GCOUNT × (PCOUNT + NAXIS2 × … × NAXISn). *)
  with_file
    (file
       [
         "SIMPLE  = T";
         "BITPIX  = 16";
         "NAXIS   = 2";
         "NAXIS1  = 0";
         "NAXIS2  = 3";
         "GROUPS  = T";
         "PCOUNT  = 2";
         "GCOUNT  = 4";
       ]
       (String.make 2880 '\000'))
    (fun path ->
      let hdus = require_ok (Fits.read path) in
      equal int 40 (Nx.numel (Fits.data (List.hd hdus)));
      is_error (Fits.Image.of_hdu (List.hd hdus)))

(* Constructed HDUs *)

let v () =
  let hdus = require_ok (Fits.read images) in
  let u16 = require_ok (Fits.get "U16" hdus) in
  let h = Fits.v (Fits.header u16) (Fits.data u16) in
  equal
    (result (option string) string)
    (Ok None)
    (H.find V.string "DATASUM" (Fits.header h));
  equal
    (result (option string) string)
    (Ok None)
    (H.find V.string "CHECKSUM" (Fits.header h));
  equal (array int) (Nx.to_array (Fits.data u16)) (Nx.to_array (Fits.data h));
  (* The primary becomes an IMAGE extension. *)
  let p = Fits.v (Fits.header (List.hd hdus)) (Fits.data (List.hd hdus)) in
  equal (result string string) (Ok "IMAGE")
    (H.get V.string "XTENSION" (Fits.header p));
  raises_match Exn.invalid_arg (fun () ->
      Fits.v (Fits.header u16) (Nx.zeros Nx.uint8 [| 3 |]))

let with_header () =
  let hdus = require_ok (Fits.read images) in
  let u16 = require_ok (Fits.get "U16" hdus) in
  let edited =
    Fits.header u16
    |> H.set V.string "OBJECT" "NGC 346"
    |> H.set V.int "NAXIS1" 3 |> H.remove "BZERO"
    |> H.add_commentary "HISTORY" "edited"
  in
  let h = Fits.header (Fits.with_header edited u16) in
  equal (result int string) (Ok 7) (H.get V.int "NAXIS1" h);
  equal (result string string) (Ok "32768") (H.get V.text "BZERO" h);
  equal (result string string) (Ok "NGC 346") (H.get V.string "OBJECT" h);
  equal (list string) [ "edited" ] (H.commentary "HISTORY" h);
  (* The records nobody edited keep their order and bytes. *)
  let kept =
    List.filter
      (fun r -> String.sub r 0 6 <> "OBJECT" && String.sub r 0 7 <> "HISTORY")
      (H.records h)
  in
  equal (list string) (H.records (Fits.header u16)) kept

let pp () =
  let hdus = require_ok (Fits.read images) in
  expect (String.concat "\n" (List.map (Format.asprintf "%a" Fits.pp) hdus))
  @@ __POS_OF__
       {|
    PRIMARY: int16 [5; 7]
    IMAGE U8 1: uint8 [5; 7]
    IMAGE I8 1: int8 [5; 7]
    IMAGE U16 1: uint16 [5; 7]
    IMAGE U32 1: uint32 [5; 7]
    IMAGE I32 1: int32 [5; 7]
    IMAGE I64 1: int64 [5; 7]
    IMAGE U64 1: uint64 [5; 7]
    IMAGE F32 1: float32 [5; 7]
    IMAGE F64 1: float64 [5; 7]
    IMAGE NAN 1: float32 [5; 7]
    IMAGE SCALED 1: int16 [5; 7], scaled (BSCALE 0.5, BZERO 10.0), BLANK -32768
    IMAGE CUBE 1: int32 [2; 3; 4]
    |}

let () =
  exit
  @@ run "Fits HDUs"
       [
         group "reading"
           [
             test "every HDU" read_all;
             test "data sizes" data_sizes;
             test "data is the file's bytes" data_is_the_file;
             test "of_bytes reads as read does" of_bytes;
           ];
         group "lookup"
           [
             test "by name and version" lookup; test "several of a name" several;
           ];
         test "digest of the headers" digest;
         group "structure"
           [
             test "truncated files" truncated;
             test "gzip streams" gzip;
             test "not FITS" not_fits;
             test "mandatory keywords" bad_structure;
           ];
         group "constructed"
           [ test "v" v; test "with_header keeps structure" with_header ];
         test "pp" pp;
       ]
