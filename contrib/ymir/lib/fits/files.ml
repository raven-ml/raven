(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* HDUs from bytes, header edits, checksums and writing files. *)

open Err
module B = Nx_device.Buffer

let strf = Printf.sprintf

(* The keywords an HDU's kind owns beyond the mandatory ones and the
   checksums. *)
let owned_by h =
  match Header.find_struct Value.string "XTENSION" h with
  | Ok (Some ("BINTABLE" | "TABLE" | "A3DTABLE")) ->
      let tiled = Header.find_struct Value.bool "ZIMAGE" h = Ok (Some true) in
      fun k ->
        Structure.table_owned k
        || (tiled && (Structure.tile_owned k || Structure.image_owned k))
  | Ok (Some "IMAGE") | Ok None -> Structure.image_owned
  | _ -> fun _ -> false

let v header bytes =
  let header =
    Header.of_records
      (Array.of_list
         (List.filter
            (fun r ->
              match Header.keyword r with
              | Some (k, _) -> not (Structure.checksums k)
              | None -> true)
            (Header.records header)))
  in
  let primary = Header.find_struct Value.string "XTENSION" header = Ok None in
  let kind, size =
    match
      let kind = Hdu.kind_of primary header in
      (kind, Hdu.data_size kind header)
    with
    | r -> r
    | exception Fail e -> invalid_arg ("Fits.v: " ^ e)
  in
  if Nx.numel bytes <> size then
    invalid_arg
      (strf "Fits.v: the header describes %d bytes of data, the tensor holds %d"
         size (Nx.numel bytes));
  let header =
    match kind with
    | Primary { groups = false } ->
        let axes = Hdu.naxes header in
        Structure.apply
          ~owned:(fun _ -> false)
          ~prefix:
            (Structure.image_prefix ~primary:false ~bitpix:(Hdu.bitpix header)
               ~axes)
          ~others:[] header
    | _ -> header
  in
  let b = Hdu.buffer_of bytes in
  Hdu.constructed header (Lazy.from_val { Hdu.buffer = b; offset = 0; size })

(* [groups records structural] splits [records] into the leading run of
   mandatory records and the structural keywords after it, each with its
   records, in order. *)
let groups records structural =
  let prefix = ref [] and groups = ref [] in
  let i = ref 0 and in_prefix = ref true in
  while !i < Array.length records do
    let span = Header.span_of records !i in
    let recs = Array.to_list (Array.sub records !i span) in
    (match Header.keyword records.(!i) with
    | Some (k, _) when structural k ->
        if !in_prefix && Structure.mandatory k then prefix := !prefix @ recs
        else (
          in_prefix := false;
          match List.assoc_opt k !groups with
          | Some _ ->
              groups :=
                List.map
                  (fun (k', r) -> if k' = k then (k', r @ recs) else (k', r))
                  !groups
          | None -> groups := !groups @ [ (k, recs) ])
    | _ -> in_prefix := false);
    i := !i + span
  done;
  (!prefix, !groups)

(* [with_header h hdu] keeps [hdu]'s structural records: its mandatory ones
   first, each other one where [h] holds that keyword, and one [h] lacks
   after the structural record it followed in [hdu]. *)
let with_header h (hdu : Hdu.t) =
  let owned = owned_by (Hdu.header hdu) in
  let structural k =
    owned k || Structure.mandatory k || Structure.checksums k
  in
  let theirs = Array.of_list (Header.records h) in
  let present = Hashtbl.create 16 in
  Array.iter
    (fun r ->
      match Header.keyword r with
      | Some (k, _) when structural k -> Hashtbl.replace present k ()
      | _ -> ())
    theirs;
  let prefix, groups =
    groups (Array.of_list (Header.records (Hdu.header hdu))) structural
  in
  (* Each present group carries the absent ones that follow it; absent ones
     before every present one follow the prefix. *)
  let lead = ref [] and carried = Hashtbl.create 16 and last = ref None in
  List.iter
    (fun (k, recs) ->
      if Hashtbl.mem present k then (
        Hashtbl.replace carried k recs;
        last := Some k)
      else
        match !last with
        | None -> lead := !lead @ recs
        | Some p -> Hashtbl.replace carried p (Hashtbl.find carried p @ recs))
    groups;
  let body = ref [] and i = ref 0 in
  while !i < Array.length theirs do
    let span = Header.span_of theirs !i in
    (match Header.keyword theirs.(!i) with
    | Some (k, _) when structural k -> (
        match Hashtbl.find_opt carried k with
        | Some recs ->
            body := List.rev_append recs !body;
            Hashtbl.remove carried k
        | None -> ())
    | _ ->
        for j = !i to !i + span - 1 do
          body := theirs.(j) :: !body
        done);
    i := !i + span
  done;
  let records = Array.of_list (prefix @ !lead @ List.rev !body) in
  let header = Header.of_records records in
  let header =
    Header.with_place (Hdu.hdu_place hdu.name hdu.index header) header
  in
  { hdu with header = Lazy.from_val header; header_bytes = None }

(* Checksums *)

let chunk = 1 lsl 22

(* [each_chunk store f] calls [f] on host copies of [store]'s bytes, in order. *)
let each_chunk (s : Hdu.store) f =
  let host = Hdu.host_bytes (Int.min chunk (Int.max 1 s.size)) in
  let a = Hdu.bigbytes host in
  let pos = ref 0 in
  while !pos < s.size do
    let n = Int.min chunk (s.size - !pos) in
    Hdu.copy_in s.buffer ~offset:(s.offset + !pos) host ~at:0 n;
    f a n;
    pos := !pos + n
  done

let data_sum (hdu : Hdu.t) =
  let s = Checksum.summer () in
  each_chunk (Hdu.store hdu) (fun a n -> Checksum.feed s a 0 n);
  Checksum.total s

(* DATASUM reads as an unsigned decimal up to 2^32 - 1, blanks and leading
   zeros allowed; a blank one is absent. *)
let stated_datasum h =
  match Header.find_struct Value.string "DATASUM" h with
  | Error e -> fail "%s" e
  | Ok None -> None
  | Ok (Some s) ->
      let t = String.trim s in
      if t = "" then None
      else
        let at () =
          Header.card_place h (List.hd (Header.cards h "DATASUM")) "DATASUM"
        in
        if not (String.for_all (fun c -> c >= '0' && c <= '9') t) then
          fail_at (at ()) "%S is not an unsigned decimal" s;
        let rec strip i =
          if i < String.length t - 1 && t.[i] = '0' then strip (i + 1) else i
        in
        let t = String.sub t (strip 0) (String.length t - strip 0) in
        if String.length t > 10 || int_of_string t > Checksum.mask then
          fail_at (at ()) "%s is past 2^32 - 1" t;
        Some (int_of_string t)

let header_bytes (hdu : Hdu.t) =
  match hdu.header_bytes with
  | Some (offset, n) -> Hdu.read_string (Hdu.store hdu).buffer ~offset n
  | None -> Header.to_string (Hdu.header hdu)

let verify (hdu : Hdu.t) =
  catch (fun () ->
      let h = Hdu.header hdu in
      let place = Hdu.place hdu in
      let stated = stated_datasum h in
      let has_checksum =
        Header.find_struct Value.text "CHECKSUM" h <> Ok None
      in
      if stated = None then fail_at place "DATASUM is absent";
      if not has_checksum then fail_at place "CHECKSUM is absent";
      let d = data_sum hdu in
      let stated = Option.get stated in
      let hs = Checksum.string 0 (header_bytes hdu) in
      let all = Checksum.add hs d in
      if stated <> d then
        fail_at place "DATASUM states %d, the data unit sums to %d" stated d;
      if all <> Checksum.mask then
        fail_at place
          "CHECKSUM disagrees: the HDU sums to %d, where -0 is %d (the data \
           unit sums to %d)"
          all Checksum.mask d)

(* Writing *)

let empty_primary () =
  let h =
    Structure.apply
      ~owned:(fun _ -> false)
      ~prefix:(Structure.image_prefix ~primary:true ~bitpix:8 ~axes:[||])
      ~others:[] Header.empty
  in
  Hdu.constructed h (Lazy.from_val (Hdu.host_store (Hdu.host_bytes 0)))

let is_image h =
  match Header.find_struct Value.string "XTENSION" h with
  | Ok None -> true
  | Ok (Some "IMAGE") -> true
  | _ -> false

(* The header [hdu] is written with, first in the file or not, before its
   checksums: the first HDU is the primary, and an image HDU elsewhere is an
   IMAGE extension. *)
let positioned ~primary ~place h =
  let is_primary = Header.find_struct Value.string "XTENSION" h = Ok None in
  let kind = Hdu.kind_of is_primary h in
  let image prim =
    Structure.apply
      ~owned:(fun _ -> false)
      ~prefix:
        (Structure.image_prefix ~primary:prim ~bitpix:(Hdu.bitpix h)
           ~axes:(Hdu.naxes h))
      ~others:[] h
  in
  match kind with
  | Primary { groups = true }
    when match Hdu.naxes h with [||] -> false | a -> a.(0) = 0 ->
      if primary then h
      else fail_at place "random groups can only be the primary HDU"
  | Primary _ -> image primary
  | Extension "IMAGE" -> if primary then image true else h
  | Extension _ -> h

let with_sums h ~datasum ~checksum =
  h
  |> Header.set ~comment:"data unit checksum" Value.string "DATASUM"
       (string_of_int datasum)
  |> Header.set ~comment:"HDU checksum" Value.string "CHECKSUM" checksum

let zeros = "0000000000000000"
let prng = Domain.DLS.new_key Random.State.make_self_init

let write_all fd (a : Checksum.bigbytes) off len =
  let n = ref 0 in
  while !n < len do
    n := !n + Unix.write_bigarray fd a (off + !n) (len - !n)
  done

let write_string fd s =
  let n = ref 0 in
  while !n < String.length s do
    n := !n + Unix.write_substring fd s !n (String.length s - !n)
  done

(* Writes [hdu] at the descriptor's position. *)
let bigbytes_of_string str =
  Bigarray.Array1.init Bigarray.int8_unsigned Bigarray.c_layout
    (String.length str) (fun i -> Char.code str.[i])

(* Writes [hdu] at the descriptor's position: the data unit first, after
   room for a header of the final one's size, summed as it passes; then the
   header with its checksums. *)
let write_hdu fd ~primary (hdu : Hdu.t) =
  let stream =
    match hdu.stream with
    | Some st when not (Lazy.is_val hdu.data) -> Some (Lazy.force st)
    | _ -> None
  in
  let first =
    match stream with Some st -> st.provisional | None -> Hdu.header hdu
  in
  let place = Hdu.hdu_place hdu.name hdu.index first in
  let h0 = positioned ~primary ~place first in
  let start = Unix.lseek fd 0 Unix.SEEK_CUR in
  let hlen =
    String.length (Header.to_string (with_sums h0 ~datasum:0 ~checksum:zeros))
  in
  let data_start = start + hlen in
  ignore (Unix.lseek fd data_start Unix.SEEK_SET);
  let s = Checksum.summer () in
  let patched = ref 0 in
  let size = ref 0 in
  let emit a n =
    Checksum.feed s a 0 n;
    write_all fd a 0 n;
    size := !size + n
  in
  (* A patch overwrites bytes appended as zeros, which summed to nothing: its
     sum, at its offset, adds to the data unit's. *)
  let patch off str =
    let here = Unix.lseek fd 0 Unix.SEEK_CUR in
    ignore (Unix.lseek fd (data_start + off) Unix.SEEK_SET);
    write_string fd str;
    ignore (Unix.lseek fd here Unix.SEEK_SET);
    let p = Checksum.at off in
    Checksum.feed p (bigbytes_of_string str) 0 (String.length str);
    patched := Checksum.add !patched (Checksum.total p)
  in
  let final =
    match stream with
    | Some st ->
        st.write
          { Hdu.append = (fun b -> emit (Hdu.bigbytes b) (B.nbytes b)); patch }
    | None ->
        each_chunk (Hdu.store hdu) emit;
        first
  in
  let pad = Hdu.padded !size - !size in
  let fill =
    if Header.find_struct Value.string "XTENSION" h0 = Ok (Some "TABLE") then
      ' '
    else '\000'
  in
  write_string fd (String.make pad fill);
  if fill = ' ' then
    Checksum.feed s (bigbytes_of_string (String.make pad ' ')) 0 pad;
  let d = Checksum.add (Checksum.total s) !patched in
  if stream = None then
    begin match stated_datasum first with
    | Some stated when stated <> d ->
        fail_at place
          "DATASUM states %d, the data unit sums to %d; Fits.v (Header.remove \
           \"DATASUM\" (Fits.header hdu)) (Fits.data hdu) accepts the data as \
           it is"
          stated d
    | _ -> ()
    end;
  let h0 = if stream = None then h0 else positioned ~primary ~place final in
  let h1 = with_sums h0 ~datasum:d ~checksum:zeros in
  let c =
    Checksum.checksum (Checksum.add (Checksum.string 0 (Header.to_string h1)) d)
  in
  let h2 = Header.to_string (with_sums h0 ~datasum:d ~checksum:c) in
  (* the provisional header holds the final one's records *)
  assert (String.length h2 = hlen);
  let stop = Unix.lseek fd 0 Unix.SEEK_CUR in
  ignore (Unix.lseek fd start Unix.SEEK_SET);
  write_string fd h2;
  ignore (Unix.lseek fd stop Unix.SEEK_SET)

let remove_quietly p = try Sys.remove p with Sys_error _ -> ()

let rename temp path =
  try Unix.rename temp path
  with Unix.Unix_error _ when Sys.win32 ->
    (* Windows refuses to replace a file while a view of its pages is mapped:
       an HDU read from [path] may hold one until it is collected. *)
    Gc.full_major ();
    Nx_device.synchronize Nx_device.disk;
    Unix.rename temp path

let write path hdus =
  let hdus =
    match hdus with
    | h :: _ when not (is_image (Hdu.structure h)) -> empty_primary () :: hdus
    | [] -> [ empty_primary () ]
    | _ -> hdus
  in
  let dir = Filename.dirname path and base = Filename.basename path in
  let tag = Random.State.bits (Domain.DLS.get prng) land 0xffffff in
  let temp = Filename.concat dir (strf ".%s.%06x.tmp" base tag) in
  match Unix.openfile temp [ O_WRONLY; O_CREAT; O_EXCL; O_CLOEXEC ] 0o600 with
  | exception Unix.Unix_error (e, _, _) ->
      Error (strf "%s: %s" temp (Unix.error_message e))
  | fd -> (
      let result =
        match
          List.iteri (fun i h -> write_hdu fd ~primary:(i = 0) h) hdus;
          Unix.fchmod fd 0o640;
          Unix.fsync fd
        with
        | () -> Ok ()
        | exception Fail e -> Error e
        | exception Sys_error e -> Error e
        | exception Unix.Unix_error (e, _, _) ->
            Error (strf "%s: %s" path (Unix.error_message e))
      in
      Unix.close fd;
      match result with
      | Error e ->
          remove_quietly temp;
          Error e
      | Ok () -> (
          match rename temp path with
          | () -> Ok ()
          | exception Unix.Unix_error (e, _, _) ->
              remove_quietly temp;
              Error (strf "cannot replace %s: %s" path (Unix.error_message e))))
