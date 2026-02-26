(*---------------------------------------------------------------------------
  NumPy .npy and .npz file format reader/writer.

  Based on ocaml-npy by Laurent Mazare. Original:
  https://github.com/LaurentMazare/ocaml-npy SPDX-License-Identifier: Apache-2.0

  Copyright 2018 Laurent Mazare Copyright 2026 The Raven authors (modifications)
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

(* Errors *)

exception Read_error of string

let read_error fmt = Printf.ksprintf (fun s -> raise (Read_error s)) fmt

(* Constants *)

let magic = "\147NUMPY"
let magic_len = String.length magic

(* Byte-level genarray I/O for extended kinds (no C stubs needed) *)

let really_write_fd fd buf off len =
  let rec loop off remaining =
    if remaining > 0 then
      let w = Unix.write fd buf off remaining in
      loop (off + w) (remaining - w)
  in
  loop off len

let as_flat_c ga =
  let n = Array.fold_left ( * ) 1 (Nx_buffer.genarray_dims ga) in
  let ga = Nx_buffer.genarray_change_layout ga Bigarray.C_layout in
  (n, Nx_buffer.of_genarray (Bigarray.reshape ga [| n |]))

let write_genarray_to_fd fd ga =
  let n, buf = as_flat_c ga in
  let byte_size =
    n * Nx_buffer.kind_size_in_bytes (Nx_buffer.genarray_kind ga)
  in
  let bytes = Bytes.create byte_size in
  Nx_buffer.blit_to_bytes ~src_off:0 ~dst_off:0 ~len:n buf bytes;
  really_write_fd fd bytes 0 byte_size

let read_fd_to_genarray fd ga =
  let n, buf = as_flat_c ga in
  let byte_size =
    n * Nx_buffer.kind_size_in_bytes (Nx_buffer.genarray_kind ga)
  in
  let bytes = Bytes.create byte_size in
  let rec loop off =
    if off < byte_size then (
      let r = Unix.read fd bytes off (byte_size - off) in
      if r = 0 then read_error "unexpected eof reading tensor data";
      loop (off + r))
  in
  loop 0;
  Nx_buffer.blit_from_bytes ~src_off:0 ~dst_off:0 ~len:n bytes buf

(* Dtype string encoding *)

type packed_kind = K : (_, _) Nx_buffer.kind -> packed_kind

let dtype_string (K kind) =
  let endian =
    match kind with
    | Nx_buffer.Int8_signed | Int8_unsigned | Bool -> "|"
    | _ -> if Sys.big_endian then ">" else "<"
  in
  let descr =
    match kind with
    | Nx_buffer.Float16 -> "f2"
    | Float32 -> "f4"
    | Float64 -> "f8"
    | Bfloat16 -> "f2"
    | Float8_e4m3 -> "f1"
    | Float8_e5m2 -> "f1"
    | Int8_signed -> "i1"
    | Int8_unsigned -> "u1"
    | Int16_signed -> "i2"
    | Int16_unsigned -> "u2"
    | Int32 -> "i4"
    | Int64 -> "i8"
    | Uint32 -> "u4"
    | Uint64 -> "u8"
    | Int4_signed -> "i1"
    | Int4_unsigned -> "u1"
    | Complex32 -> "c8"
    | Complex64 -> "c16"
    | Bool -> "b1"
  in
  endian ^ descr

let kind_of_descr = function
  | "f4" -> K Float32
  | "f8" -> K Float64
  | "i4" -> K Int32
  | "i8" -> K Int64
  | "u4" -> K Uint32
  | "u8" -> K Uint64
  | "u1" -> K Int8_unsigned
  | "i1" -> K Int8_signed
  | "u2" -> K Int16_unsigned
  | "i2" -> K Int16_signed
  | "c8" -> K Complex32
  | "c16" -> K Complex64
  | "b1" -> K Bool
  | s -> read_error "unsupported dtype descriptor %s" s

(* Header parsing *)

(* Split a string on [on], respecting parentheses depth *)
let header_split str ~on =
  let parens = ref 0 in
  let cuts = ref [] in
  for i = 0 to String.length str - 1 do
    match str.[i] with
    | '(' -> incr parens
    | ')' -> decr parens
    | c when !parens = 0 && c = on -> cuts := i :: !cuts
    | _ -> ()
  done;
  List.fold_left
    (fun (prev, acc) i -> (i, String.sub str (i + 1) (prev - i - 1) :: acc))
    (String.length str, [])
    !cuts
  |> fun (first, acc) -> String.sub str 0 first :: acc

(* Trim characters from both ends *)
let header_trim str ~on =
  let len = String.length str in
  let rec scan_left i =
    if i >= len then i else if List.mem str.[i] on then scan_left (i + 1) else i
  in
  let rec scan_right j =
    if j <= 0 then j
    else if List.mem str.[j - 1] on then scan_right (j - 1)
    else j
  in
  let l = scan_left 0 in
  let r = scan_right len in
  if l >= r then "" else String.sub str l (r - l)

type header = { kind : packed_kind; fortran_order : bool; shape : int array }

let parse_header s =
  let s = header_trim s ~on:[ '{'; ' '; '}'; '\n' ] in
  let fields =
    header_split s ~on:',' |> List.map String.trim
    |> List.filter (fun s -> String.length s > 0)
    |> List.map (fun field ->
        match header_split field ~on:':' with
        | [ name; value ] ->
            ( header_trim name ~on:[ '\''; ' ' ],
              header_trim value ~on:[ '\''; ' '; '('; ')' ] )
        | _ -> read_error "unable to parse header field %s" field)
  in
  let find name =
    try List.assoc name fields
    with Not_found -> read_error "missing header field %s" name
  in
  let kind =
    let descr = find "descr" in
    (match descr.[0] with
    | '|' | '=' -> ()
    | '>' ->
        if not Sys.big_endian then
          read_error "big endian data on little endian arch"
    | '<' ->
        if Sys.big_endian then
          read_error "little endian data on big endian arch"
    | c -> read_error "unknown endianness marker %c" c);
    kind_of_descr (String.sub descr 1 (String.length descr - 1))
  in
  let fortran_order =
    match find "fortran_order" with
    | "False" -> false
    | "True" -> true
    | s -> read_error "invalid fortran_order %s" s
  in
  let shape =
    find "shape" |> header_split ~on:',' |> List.map String.trim
    |> List.filter (fun s -> String.length s > 0)
    |> List.map int_of_string |> Array.of_list
  in
  { kind; fortran_order; shape }

(* Header writing *)

let shape_string dims =
  match dims with
  | [| n |] -> strf "%d," n
  | _ -> Array.to_list dims |> List.map string_of_int |> String.concat ", "

let fortran_string (type a) (layout : a Bigarray.layout) =
  match layout with
  | Bigarray.C_layout -> "False"
  | Bigarray.Fortran_layout -> "True"

let encode_header ~layout ~packed_kind ~dims =
  let header =
    strf "{'descr': '%s', 'fortran_order': %s, 'shape': (%s), }"
      (dtype_string packed_kind) (fortran_string layout) (shape_string dims)
  in
  let total_len = String.length header + magic_len + 4 + 1 in
  let pad = if total_len mod 16 = 0 then 0 else 16 - (total_len mod 16) in
  let header_len = String.length header + pad + 1 in
  strf "%s\001\000%c%c%s%s\n" magic
    (header_len mod 256 |> Char.chr)
    (header_len / 256 |> Char.chr)
    header (String.make pad ' ')

(* Low-level I/O *)

let with_fd path flags perm f =
  let fd = Unix.openfile path flags perm in
  Fun.protect ~finally:(fun () -> Unix.close fd) (fun () -> f fd)

let really_read_fd fd n =
  let buf = Bytes.create n in
  let rec loop off =
    if off >= n then ()
    else
      let r = Unix.read fd buf off (n - off) in
      if r = 0 then read_error "unexpected eof";
      loop (off + r)
  in
  loop 0;
  Bytes.to_string buf

(* Create a genarray backed by the file, or allocate + read for extended
   kinds *)
let map_or_read fd ~pos kind layout shape =
  let is_scalar = Array.length shape = 0 in
  let actual = if is_scalar then [| 1 |] else shape in
  let ga =
    match Nx_buffer.to_stdlib_kind kind with
    | Some std_kind -> Unix.map_file fd ~pos std_kind layout false actual
    | None ->
        let ga = Nx_buffer.genarray_create kind layout actual in
        ignore (Unix.lseek fd (Int64.to_int pos) Unix.SEEK_SET);
        read_fd_to_genarray fd ga;
        ga
  in
  if is_scalar then Bigarray.reshape ga [||] else ga

(* Npy read/write *)

type packed = P : (_, _, _) Bigarray.Genarray.t -> packed

let read_copy path =
  with_fd path [ O_RDONLY ] 0 @@ fun fd ->
  let magic' = really_read_fd fd magic_len in
  if magic <> magic' then read_error "not a .npy file (bad magic)";
  let version = Char.code (really_read_fd fd 2).[0] in
  let hdr_len_bytes =
    match version with
    | 1 -> 2
    | 2 -> 4
    | v -> read_error "unsupported npy version %d" v
  in
  let hdr_len_str = really_read_fd fd hdr_len_bytes in
  let hdr_len = ref 0 in
  for i = String.length hdr_len_str - 1 downto 0 do
    hdr_len := (256 * !hdr_len) + Char.code hdr_len_str.[i]
  done;
  let hdr = parse_header (really_read_fd fd !hdr_len) in
  let pos = Int64.of_int (!hdr_len + hdr_len_bytes + magic_len + 2) in
  let (K kind) = hdr.kind in
  let build layout =
    let src = map_or_read fd ~pos kind layout hdr.shape in
    let dst =
      Nx_buffer.genarray_create kind layout (Nx_buffer.genarray_dims src)
    in
    Nx_buffer.genarray_blit src dst;
    P dst
  in
  if hdr.fortran_order then build Bigarray.Fortran_layout
  else build Bigarray.C_layout

let write ga path =
  with_fd path [ O_CREAT; O_TRUNC; O_RDWR ] 0o640 @@ fun fd ->
  let kind = Nx_buffer.genarray_kind ga in
  let dims = Nx_buffer.genarray_dims ga in
  let layout = Bigarray.Genarray.layout ga in
  let hdr = encode_header ~layout ~packed_kind:(K kind) ~dims in
  let hdr_len = String.length hdr in
  if Unix.write_substring fd hdr 0 hdr_len <> hdr_len then
    failwith "npy: incomplete header write";
  match Nx_buffer.to_stdlib_kind kind with
  | Some std_kind ->
      let dst =
        Unix.map_file fd ~pos:(Int64.of_int hdr_len) std_kind layout true dims
      in
      Bigarray.Genarray.blit ga dst
  | None ->
      ignore (Unix.lseek fd hdr_len Unix.SEEK_SET);
      write_genarray_to_fd fd ga

(* Npz read/write (via camlzip) *)

module Npz = struct
  let npy_suffix = ".npy"

  type in_file = Zip.in_file
  type out_file = Zip.out_file

  let open_in = Zip.open_in
  let close_in = Zip.close_in
  let open_out = Zip.open_out
  let close_out = Zip.close_out

  let entries t =
    List.map
      (fun (entry : Zip.entry) ->
        let name = entry.Zip.filename in
        let suf_len = String.length npy_suffix in
        if
          String.length name >= suf_len
          && String.sub name (String.length name - suf_len) suf_len = npy_suffix
        then String.sub name 0 (String.length name - suf_len)
        else name)
      (Zip.entries t)

  let read t name =
    let entry_name = name ^ npy_suffix in
    let entry =
      try Zip.find_entry t entry_name with Not_found -> raise Not_found
    in
    let tmp = Filename.temp_file "npy" ".tmp" in
    Fun.protect ~finally:(fun () -> Sys.remove tmp) @@ fun () ->
    Zip.copy_entry_to_file t entry tmp;
    read_copy tmp

  let write t name ga =
    let entry_name = name ^ npy_suffix in
    let tmp = Filename.temp_file "npy" ".tmp" in
    Fun.protect ~finally:(fun () -> Sys.remove tmp) @@ fun () ->
    write ga tmp;
    Zip.copy_file_to_entry tmp t entry_name
end
