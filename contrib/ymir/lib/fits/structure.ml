(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The structural keywords the writer owns, and how they enter a header: a
   record holding the computed value stays at its place with its text; the
   others are printed at the standard's positions, the mandatory keywords
   first and in order (FITS 4.0 §4.4.1), the rest after them; the caller's
   differing copies go. *)

let strf = Printf.sprintf

type entry = {
  key : string;
  records : string list;  (** printed fresh *)
  same : string -> bool;  (** whether a record holds the computed value *)
}

let printed set = Header.records (set Header.empty)

let same_as v key r =
  match Header.keyword r with
  | Some (k, _) when k = key -> (
      let h = Header.of_records [| r |] in
      match Header.find_struct v key h with Ok (Some x) -> Some x | _ -> None)
  | _ -> None

let int key n =
  {
    key;
    records = printed (Header.set Value.int key n);
    same = (fun r -> same_as Value.int key r = Some n);
  }

let bool key b =
  {
    key;
    records = printed (Header.set Value.bool key b);
    same = (fun r -> same_as Value.bool key r = Some b);
  }

let string key s =
  {
    key;
    records = printed (Header.set Value.string key s);
    same = (fun r -> same_as Value.string key r = Some s);
  }

(* A number given as decimal text, as BZERO 9223372036854775808 is. *)
let decimal key t =
  {
    key;
    records = printed (Header.set Value.text key t);
    same =
      (fun r ->
        match same_as Value.text key r with
        | Some t' -> Decimal.equal t t'
        | None -> false);
  }

(* An entry whose value is records held elsewhere, compared as text. *)
let verbatim key records =
  {
    key;
    records;
    same = (fun r -> match records with r' :: _ -> r = r' | [] -> false);
  }

(* Keyword classes *)

let numbered root k =
  let n = String.length root in
  String.length k > n
  && String.sub k 0 n = root
  &&
  let d = String.sub k n (String.length k - n) in
  d.[0] <> '0' && String.for_all (fun c -> c >= '0' && c <= '9') d

let mandatory k =
  match k with
  | "SIMPLE" | "XTENSION" | "BITPIX" | "NAXIS" | "EXTEND" | "PCOUNT" | "GCOUNT"
  | "GROUPS" | "TFIELDS" ->
      true
  | _ -> numbered "NAXIS" k

let checksums k = k = "DATASUM" || k = "CHECKSUM"

(* [apply ~owned ~prefix ~others h] is [h] with the structure [prefix], the
   mandatory keywords in order, and [others], placed anywhere; every other
   record of a keyword [owned] holds is dropped. *)
let apply ~owned ~prefix ~others h =
  let rs = Array.of_list (Header.records h) in
  let n = Array.length rs in
  let used = Array.make n false in
  let head =
    List.mapi
      (fun p e ->
        if p < n && e.same rs.(p) then (
          used.(p) <- true;
          [ rs.(p) ])
        else e.records)
      prefix
    |> List.concat
  in
  let placed = Hashtbl.create 16 in
  let wanted = Hashtbl.create 16 in
  List.iter (fun e -> Hashtbl.replace wanted e.key e) others;
  let body = ref [] in
  let i = ref 0 in
  while !i < n do
    let r = rs.(!i) in
    let span = Header.span_of rs !i in
    (if used.(!i) then ()
     else
       match Header.keyword r with
       | Some (k, _) when owned k || mandatory k || checksums k -> (
           match Hashtbl.find_opt wanted k with
           | Some e when (not (Hashtbl.mem placed k)) && e.same r ->
               Hashtbl.replace placed k ();
               for j = !i to !i + span - 1 do
                 body := rs.(j) :: !body
               done
           | _ -> ())
       | _ ->
           for j = !i to !i + span - 1 do
             body := rs.(j) :: !body
           done);
    i := !i + span
  done;
  let fresh =
    List.concat_map
      (fun e -> if Hashtbl.mem placed e.key then [] else e.records)
      others
  in
  Header.of_records ~place:(Header.place h)
    (Array.of_list (head @ fresh @ List.rev !body))

(* Primary or extension *)

let image_prefix ~primary ~bitpix ~axes =
  let naxes =
    Array.to_list (Array.mapi (fun i n -> int (strf "NAXIS%d" (i + 1)) n) axes)
  in
  if primary then
    [ bool "SIMPLE" true; int "BITPIX" bitpix; int "NAXIS" (Array.length axes) ]
    @ naxes
    @ [ bool "EXTEND" true ]
  else
    [
      string "XTENSION" "IMAGE";
      int "BITPIX" bitpix;
      int "NAXIS" (Array.length axes);
    ]
    @ naxes
    @ [ int "PCOUNT" 0; int "GCOUNT" 1 ]

let image_owned k = k = "BZERO" || k = "BSCALE"

(* Column-numbered keywords: FITS 4.0 Table 8's, and the column forms of the
   WCS keywords (Table 26), with their alternate letter. *)
let column_roots =
  [
    "TTYPE";
    "TFORM";
    "TUNIT";
    "TSCAL";
    "TZERO";
    "TNULL";
    "TDISP";
    "TDIM";
    "TBCOL";
    "TLMIN";
    "TLMAX";
    "TDMIN";
    "TDMAX";
    "TUCD";
    "TUTYP";
    "TCTYP";
    "TCUNI";
    "TCRPX";
    "TCRVL";
    "TCDLT";
    "TCROT";
    "TCRDE";
    "TCSYE";
    "TCNAM";
    "TWCS";
  ]

let wcs_roots =
  [
    "CTYP";
    "CUNI";
    "CRPX";
    "CRVL";
    "CDLT";
    "CROT";
    "CRDE";
    "CSYE";
    "CNAM";
    "CTY";
    "CUN";
    "CRP";
    "CRV";
    "CDE";
    "CSY";
    "CNA";
    "CRD";
  ]

let is_digit c = c >= '0' && c <= '9'

let strip_alt k =
  let n = String.length k in
  if n >= 2 && k.[n - 1] >= 'A' && k.[n - 1] <= 'Z' && is_digit k.[n - 2] then
    String.sub k 0 (n - 1)
  else k

let column_numbered k =
  let k = strip_alt k in
  List.exists (fun r -> numbered r k) column_roots
  ||
  (* iCTYPn: an axis number, a root, a column number *)
  let n = String.length k in
  let i = ref 0 in
  while !i < n && is_digit k.[!i] do
    incr i
  done;
  !i > 0
  && k.[0] <> '0'
  && List.exists (fun r -> numbered r (String.sub k !i (n - !i))) wcs_roots

let table_owned k = k = "THEAP" || column_numbered k

let tile_keys =
  [
    "ZIMAGE";
    "ZSIMPLE";
    "ZTENSION";
    "ZBITPIX";
    "ZNAXIS";
    "ZCMPTYPE";
    "ZMASKCMP";
    "ZEXTEND";
    "ZBLOCKED";
    "ZPCOUNT";
    "ZGCOUNT";
    "ZHECKSUM";
    "ZDATASUM";
    "ZQUANTIZ";
    "ZDITHER0";
    "ZBLANK";
    "ZSCALE";
    "ZZERO";
  ]

let tile_owned k =
  List.mem k tile_keys || numbered "ZNAXIS" k || numbered "ZTILE" k
  || numbered "ZNAME" k || numbered "ZVAL" k
