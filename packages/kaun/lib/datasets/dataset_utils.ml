(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let src = Logs.Src.create "kaun.datasets" ~doc:"Kaun datasets"

module Log = (val Logs.src_log src : Logs.LOG)

let get_cache_dir ?(getenv = Sys.getenv_opt) dataset_name =
  let root =
    match getenv "RAVEN_CACHE_ROOT" with
    | Some dir when dir <> "" -> dir
    | _ ->
        let xdg =
          match getenv "XDG_CACHE_HOME" with
          | Some d when d <> "" -> d
          | _ ->
              (* The user's home: [HOME] on Unix, [USERPROFILE] on Windows. *)
              let home =
                match (getenv "HOME", getenv "USERPROFILE") with
                | Some d, _ when d <> "" -> d
                | _, Some d when d <> "" -> d
                | _ ->
                    failwith "no home directory: set HOME, or RAVEN_CACHE_ROOT"
              in
              Filename.concat home ".cache"
        in
        Filename.concat xdg "raven"
  in
  let path =
    List.fold_left Filename.concat root ("datasets" :: [ dataset_name ])
  in
  let sep = Filename.dir_sep.[0] in
  if path <> "" && path.[String.length path - 1] = sep then path
  else path ^ Filename.dir_sep

let download_file url dest_path =
  Log.info (fun m -> m "Downloading %s to %s" (Filename.basename url) dest_path);
  Kaun_download.file ~url dest_path;
  Log.info (fun m -> m "Downloaded %s" (Filename.basename dest_path))

let ensure_file url dest_path =
  if not (Sys.file_exists dest_path) then download_file url dest_path
  else Log.debug (fun m -> m "Found %s" dest_path)

(* Data is read and written [copy_block] bytes at a time. *)
let copy_block = 65536

(* [gz_chunks ~src fd] gives the data of the gzip file [src] read from [fd], a
   chunk a call, in order, then [""] once the stream ended. *)
let gz_chunks ~src fd =
  let d = Lz.Gzip.decoder () in
  let input =
    Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout copy_block
  in
  let pending = ref (Bigarray.Array1.sub input 0 0) in
  let last = ref false and ended = ref false in
  let rec next () =
    if !ended then ""
    else (
      if Bigarray.Array1.dim !pending = 0 && not !last then (
        let n =
          try Unix.read_bigarray fd input 0 (Bigarray.Array1.dim input)
          with Unix.Unix_error (e, _, _) ->
            failwith (Printf.sprintf "%s: %s" src (Unix.error_message e))
        in
        pending := Bigarray.Array1.sub input 0 n;
        last := n = 0);
      let out = ref "" in
      let copy run =
        let n = Bigarray.Array1.dim run in
        let b = Bytes.create n in
        for i = 0 to n - 1 do
          Bytes.unsafe_set b i
            (Char.unsafe_chr (Bigarray.Array1.unsafe_get run i))
        done;
        out := Bytes.unsafe_to_string b
      in
      match Lz.Decoder.step d ~last:!last !pending copy with
      | Error e -> failwith (Format.asprintf "%s: %a" src Lz.pp_error e)
      | Ok (Lz.Done _) ->
          ended := true;
          !out
      | Ok (Lz.More k) ->
          let n = Bigarray.Array1.dim !pending in
          pending := Bigarray.Array1.sub !pending k (n - k);
          if !out = "" then next () else !out)
  in
  next

let rec write_string fd s off n =
  if n > 0 then
    let k = Unix.write_substring fd s off n in
    write_string fd s (off + k) (n - k)

(* [gunzip ~src ~dst] writes the data of the gzip file [src] to [dst] as it
   decodes it, in bounded memory, through Kaun_download.write: [dst] holds the
   whole data or is as it was. *)
let gunzip ~src ~dst =
  Kaun_download.write dst @@ fun fd ->
  let fd_in = Unix.openfile src [ Unix.O_RDONLY; Unix.O_CLOEXEC ] 0 in
  Fun.protect
    ~finally:(fun () -> Unix.close fd_in)
    (fun () ->
      let next = gz_chunks ~src fd_in in
      let rec copy () =
        match next () with
        | "" -> ()
        | c ->
            write_string fd c 0 (String.length c);
            copy ()
      in
      copy ())

let ensure_decompressed_gz ~gz_path ~target_path =
  if Sys.file_exists target_path then (
    Log.debug (fun m -> m "Found %s" target_path);
    true)
  else if Sys.file_exists gz_path then (
    Log.info (fun m -> m "Decompressing %s..." gz_path);
    gunzip ~src:gz_path ~dst:target_path;
    Log.info (fun m -> m "Decompressed to %s" target_path);
    true)
  else (
    Log.warn (fun m -> m "Compressed file %s not found" gz_path);
    false)

(* ustar *)

(* An archive is a run of 512-byte records: each member a header, then its
   data padded to a whole record; a zero record ends it (POSIX pax, "ustar
   Interchange Format"). *)
let record = 512

(* [take next n] is the next [n] bytes of the chunks [next] gives, fewer at
   their end. *)
let take next =
  let chunk = ref "" and pos = ref 0 in
  let rec fill b n =
    let need = n - Buffer.length b in
    if need > 0 then
      if !pos < String.length !chunk then (
        let k = min need (String.length !chunk - !pos) in
        Buffer.add_substring b !chunk !pos k;
        pos := !pos + k;
        fill b n)
      else
        match next () with
        | "" -> ()
        | c ->
            chunk := c;
            pos := 0;
            fill b n
  in
  fun n ->
    (* Within the chunk, one copy; across chunks, through a buffer. *)
    if n <= String.length !chunk - !pos then (
      let s = String.sub !chunk !pos n in
      pos := !pos + n;
      s)
    else
      let b = Buffer.create n in
      fill b n;
      Buffer.contents b

let err_ustar src what = failwith (Printf.sprintf "%s: %s" src what)

(* The field at [off], [len] bytes, up to its first NUL. *)
let field h off len =
  let s = String.sub h off len in
  match String.index_opt s '\000' with Some i -> String.sub s 0 i | None -> s

(* A numeric field: leading zero-filled octal digits, ended by a space or a
   NUL. *)
let octal ~src h off len =
  let s = String.trim (field h off len) in
  let digit c = c >= '0' && c <= '7' in
  if s = "" || not (String.for_all digit s) then
    err_ustar src (Printf.sprintf "a header field at %d is not octal" off)
  else int_of_string ("0o" ^ s)

(* The checksum: the sum of the header's bytes, its own field counted as eight
   spaces. *)
let checksum h =
  let sum = ref (8 * Char.code ' ') in
  String.iteri
    (fun i c -> if i < 148 || i >= 156 then sum := !sum + Char.code c)
    h;
  !sum

(* A member's name: its name field, after its prefix in a POSIX header. A
   GNU header, whose magic is "ustar  ", holds other fields where POSIX's
   prefix is. *)
let member_name h =
  let name = field h 0 100 in
  let posix = String.sub h 257 6 = "ustar\000" in
  let prefix = if posix then field h 345 155 else "" in
  if prefix = "" then name else prefix ^ "/" ^ name

(* [drain take] reads to the end of what [take] reads. *)
let rec drain take = if take copy_block <> "" then drain take

let read_tar_gz path f =
  let truncated () = err_ustar path "the archive is truncated" in
  let read take =
    let skip n =
      let rec go n =
        if n > 0 then
          let k = min n copy_block in
          if String.length (take k) < k then truncated () else go (n - k)
      in
      go n
    in
    (* [f]'s view of a member of [size] bytes: at most what is left of it. *)
    let member size =
      let left = ref size in
      let take_member n =
        let n = min (max n 0) !left in
        let s = take n in
        if String.length s < n then truncated ();
        left := !left - n;
        s
      in
      (take_member, left)
    in
    let rec members () =
      let h = take record in
      if String.length h < record then truncated ()
      else if String.for_all (( = ) '\000') h then drain take
      else (
        if checksum h <> octal ~src:path h 148 8 then
          err_ustar path "a header's checksum does not match";
        let size = octal ~src:path h 124 12 in
        let padding = (record - (size mod record)) mod record in
        (match h.[156] with
        | '0' | '\000' | '7' ->
            let take_member, left = member size in
            f (member_name h) size take_member;
            skip (!left + padding)
        | '1' | '2' | '3' | '4' | '5' | '6' | 'g' -> skip (size + padding)
        | t -> err_ustar path (Printf.sprintf "member type %C is not read" t));
        members ())
    in
    members ()
  in
  match Unix.openfile path [ Unix.O_RDONLY; Unix.O_CLOEXEC ] 0 with
  | exception Unix.Unix_error (e, _, _) ->
      err_ustar path (Unix.error_message e)
  | fd ->
      Fun.protect
        ~finally:(fun () -> Unix.close fd)
        (fun () -> read (take (gz_chunks ~src:path fd)))
