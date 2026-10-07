(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Hostile bytes give values or an Error (the RFC's second law). The corpus
   is the files astropy wrote (gen/fixtures.py): images of every format,
   tiles of every codec and dither, and binary and ASCII tables with every
   TFORM code; and cuts of files the archives wrote (survey/README.md):
   pipeline headers with CONTINUE and HIERARCH records, cfitsio's Rice, gzip
   and HCOMPRESS tiles, IDL's and STIL's tables, heap arrays, an ASCII table
   and random groups. A byte changed in each region where a decoder path
   reads, and a file cut around each of its boundaries, are read by every
   reader: none may raise. *)

open Windtrap
open Ymir_fits
module H = Fits.Header
module V = Fits.Value
module I = Fits.Image
module T = Fits.Table

let files =
  [ "golden/images.fits"; "golden/tiles.fits"; "golden/tables.fits" ]
  @ (Sys.readdir "survey" |> Array.to_list
    |> List.filter (fun f -> Filename.check_suffix f ".fits")
    |> List.sort compare
    |> List.map (Filename.concat "survey"))

let contents =
  List.map (fun p -> (p, In_channel.with_open_bin p In_channel.input_all)) files

let tensor s =
  let a = Bigarray.(Array1.create int8_unsigned c_layout (String.length s)) in
  String.iteri (fun i c -> Bigarray.Array1.unsafe_set a i (Char.code c)) s;
  Nx.of_bigarray (Bigarray.genarray_of_array1 a)

(* The keyword of a record that holds a value, by the interface's rules
   for names, HIERARCH ones included. *)
let keyword r =
  let name = String.trim (String.sub r 0 8) in
  let standard c =
    match c with 'A' .. 'Z' | '0' .. '9' | '-' | '_' -> true | _ -> false
  in
  let token t = String.for_all (fun c -> c > ' ' && c <= '~' && c <> '=') t in
  if
    String.sub r 8 2 = "= "
    && name <> ""
    && String.for_all standard name
    && String.starts_with ~prefix:name r
    && name <> "CONTINUE"
  then Some name
  else if String.starts_with ~prefix:"HIERARCH " r then
    match String.index_opt r '=' with
    | None -> None
    | Some i -> (
        let tokens =
          String.split_on_char ' ' (String.sub r 9 (i - 9))
          |> List.filter (( <> ) "")
        in
        match tokens with
        | [] -> None
        | [ t ] when String.length t <= 8 -> None
        | ts when List.for_all token ts -> Some (String.concat " " ts)
        | _ -> None)
  else None

(* Every reader on every HDU; an exception fails the law. *)
let read_all name s =
  let guard what f =
    match f () with
    | (_ : (unit, string) result) -> ()
    | exception e -> failf "%s raised %s" what (Printexc.to_string e)
  in
  match Fits.of_bytes ~name (tensor s) with
  | exception e -> failf "of_bytes raised %s" (Printexc.to_string e)
  | Error _ -> ()
  | Ok hdus ->
      List.iter
        (fun h ->
          guard "header values" (fun () ->
              let hh = Fits.header h in
              List.iter
                (fun r ->
                  Option.iter
                    (fun k ->
                      ignore (H.find V.text k hh);
                      ignore (H.find V.string k hh);
                      ignore (H.find V.float k hh))
                    (keyword r))
                (H.records hh);
              Ok ());
          guard "verify" (fun () -> Fits.verify h);
          guard "digest" (fun () -> Ok (ignore (Fits.digest h)));
          guard "unit" (fun () -> Result.map ignore (Fits.unit h));
          guard "pp" (fun () -> Ok (ignore (Format.asprintf "%a" Fits.pp h)));
          guard "image values" (fun () ->
              Result.map ignore (I.values Nx.float64 h));
          guard "image raw" (fun () -> Result.map ignore (I.raw Nx.float64 h));
          guard "image validity" (fun () -> Result.map ignore (I.validity h));
          guard "image window" (fun () ->
              Result.map ignore
                (I.values ~window:[| (0, 1); (0, 1) |] Nx.float32 h));
          guard "table" (fun () -> Result.map ignore (T.read h));
          guard "column units" (fun () ->
              Result.map
                (fun t ->
                  List.iter
                    (fun (c : T.column) ->
                      match H.find V.string "TUNIT" c.cards with
                      | Ok (Some u) -> ignore (Fits.Unit.parse ~scope:"t" u)
                      | Ok None | Error _ -> ())
                    (T.columns t))
                (T.of_hdu h)))
        hdus

(* The structure of the corpus: each HDU's header and data unit, in file
   bytes, and the regions where a decoder path reads. *)
type hdu_span = { header : int * int; data : int * int; h : H.t }

let spans s =
  let hdus = require_ok (Fits.of_bytes ~name:"corpus" (tensor s)) in
  let pos = ref 0 in
  List.map
    (fun hdu ->
      let h = Fits.header hdu in
      let hl = String.length (H.to_string h) in
      let dl = Nx.numel (Fits.data hdu) in
      let span =
        { header = (!pos, !pos + hl); data = (!pos + hl, !pos + hl + dl); h }
      in
      pos := !pos + hl + ((dl + 2879) / 2880 * 2880);
      span)
    hdus

let text k h = match H.find V.string k h with Ok (Some s) -> s | _ -> ""
let tiled h = H.find V.bool "ZIMAGE" h = Ok (Some true)

(* Bytes of a binary table field per TFORM, for the byte ranges of a
   column. *)
let field_width f =
  let n = String.length f in
  let i = ref 0 in
  while !i < n && f.[!i] >= '0' && f.[!i] <= '9' do
    incr i
  done;
  let r = if !i = 0 then 1 else int_of_string (String.sub f 0 !i) in
  match f.[!i] with
  | 'P' -> 8
  | 'Q' -> 16
  | 'X' -> (r + 7) / 8
  | 'L' | 'B' | 'A' -> r
  | 'I' -> 2 * r
  | 'J' | 'E' -> 4 * r
  | 'K' | 'D' | 'C' -> 8 * r
  | 'M' -> 16 * r
  | _ -> 0

(* The byte ranges, in a table's rows, of its columns whose TFORM [keep]
   accepts. *)
let column_ranges sp keep =
  let h = sp.h in
  let nf = match H.get V.int "TFIELDS" h with Ok n -> n | Error _ -> 0 in
  let rb = match H.get V.int "NAXIS1" h with Ok n -> n | Error _ -> 0 in
  let rows = match H.get V.int "NAXIS2" h with Ok n -> n | Error _ -> 0 in
  let start = ref 0 in
  List.concat
    (List.init nf (fun i ->
         let f = text (Printf.sprintf "TFORM%d" (i + 1)) h in
         let w = field_width f and s = !start in
         start := !start + w;
         if keep f then
           List.init rows (fun r ->
               (fst sp.data + (r * rb) + s, fst sp.data + (r * rb) + s + w))
         else []))

let regions =
  List.concat_map
    (fun (path, s) ->
      let sps = spans s in
      let file = Filename.basename path in
      let tag label ranges =
        List.map (fun r -> (file ^ ": " ^ label, path, r)) ranges
      in
      List.concat_map
        (fun sp ->
          let h = sp.h in
          let xt = text "XTENSION" h in
          tag "header" [ sp.header ]
          @
          if tiled h then
            let cmp = text "ZCMPTYPE" h in
            let label =
              if
                H.find V.string "ZQUANTIZ" h <> Ok None
                && text "ZQUANTIZ" h <> "NONE"
              then "dequantization"
              else if String.length cmp >= 4 && String.sub cmp 0 4 = "RICE" then
                "rice"
              else if String.length cmp >= 4 && String.sub cmp 0 4 = "GZIP" then
                "gzip"
              else if cmp = "NOCOMPRESS" then "uncompressed tiles"
              else "unknown codec"
            in
            tag label [ sp.data ]
          else if xt = "BINTABLE" then
            tag "descriptors"
              (column_ranges sp (fun f ->
                   String.contains f 'P' || String.contains f 'Q'))
            @ tag "bits" (column_ranges sp (fun f -> String.contains f 'X'))
            @ tag "heap"
                [
                  ( (fst sp.data
                    +
                    match (H.get V.int "NAXIS1" h, H.get V.int "NAXIS2" h) with
                    | Ok a, Ok b -> a * b
                    | _ -> 0),
                    snd sp.data );
                ]
            @ tag "binary fields" [ sp.data ]
          else if xt = "TABLE" then tag "ascii" [ sp.data ]
          else if H.find V.bool "GROUPS" h = Ok (Some true) then
            tag "random groups" [ sp.data ]
          else tag "plain image" [ sp.data ])
        sps)
    contents
  |> List.filter (fun (_, _, (a, b)) -> b > a)
  |> Array.of_list

let labels =
  List.sort_uniq compare
    (Array.to_list (Array.map (fun (l, _, _) -> l) regions))

(* The regions of each decoder path in each file, so that each pair is
   drawn as often. *)
let by_label =
  Array.of_list
    (List.map
       (fun l ->
         List.filter (fun (l', _, _) -> l' = l) (Array.to_list regions)
         |> Array.of_list)
       labels)

let mutated =
  prop "a changed byte in any decoder's bytes gives values or an Error"
    ~count:1000
    Gen.(
      quad
        (int_range 0 (Array.length by_label - 1))
        (int_range 0 1_000_000) (int_range 0 1_000_000) (int_range 0 255))
    (fun (l, k, off, byte) ->
      let rs = by_label.(l) in
      let label, path, (a, b) = rs.(k mod Array.length rs) in
      List.iter (fun l -> cover l (l = label)) labels;
      let s = Bytes.of_string (List.assoc path contents) in
      Bytes.set s (a + (off mod (b - a))) (Char.chr byte);
      read_all path (Bytes.to_string s))

(* Cuts: at every block boundary of each file and around each HDU's
   header and data unit ends, one byte either side. *)
let cuts =
  List.concat_map
    (fun (path, s) ->
      let n = String.length s in
      let blocks = List.init ((n / 2880) + 1) (fun i -> i * 2880) in
      let ends =
        List.concat_map (fun sp -> [ snd sp.header; snd sp.data ]) (spans s)
      in
      List.concat_map (fun p -> [ p - 1; p; p + 1 ]) (blocks @ ends)
      |> List.filter (fun p -> p >= 0 && p < n)
      |> List.sort_uniq compare
      |> List.map (fun p -> (path, p)))
    contents

let truncated =
  cases
    ~name:(fun (path, p) -> Printf.sprintf "%s cut at %d" path p)
    "a cut file gives values or an Error" cuts
    (fun (path, p) -> read_all path (String.sub (List.assoc path contents) 0 p))

let () = exit @@ run "Fits hostile bytes" [ mutated; truncated ]
