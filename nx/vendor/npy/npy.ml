exception Cannot_write
exception Read_error of string

let read_error fmt = Printf.ksprintf (fun s -> raise (Read_error s)) fmt
let magic_string = "\147NUMPY"
let magic_string_len = String.length magic_string

type packed_kind = P : (_, _) Bigarray.kind -> packed_kind

let dtype ~packed_kind =
  let endianness =
    match packed_kind with
    | P Bigarray.Char -> "|"
    | P _ -> if Sys.big_endian then ">" else "<"
  in
  let kind =
    match packed_kind with
    | P Bigarray.Int32 -> "i4"
    | P Bigarray.Int64 -> "i8"
    | P Bigarray.Float16 -> "f2"
    | P Bigarray.Float32 -> "f4"
    | P Bigarray.Float64 -> "f8"
    | P Bigarray.Int8_unsigned -> "u1"
    | P Bigarray.Int8_signed -> "i1"
    | P Bigarray.Int16_unsigned -> "u2"
    | P Bigarray.Int16_signed -> "i2"
    | P Bigarray.Char -> "S1"
    | P Bigarray.Complex32 -> "c8" (* 2 32bits float. *)
    | P Bigarray.Complex64 -> "c16" (* 2 64bits float. *)
    | P Bigarray.Int -> failwith "Int is not supported"
    | P Bigarray.Nativeint -> failwith "Nativeint is not supported."
  in
  endianness ^ kind

let map_file file_descr ~pos kind layout shared shape =
  let is_scalar = Array.length shape = 0 in
  let array =
    Unix.map_file
      file_descr
      ~pos
      kind
      layout
      shared
      (if is_scalar then [| 1 |] else shape)
  in
  if is_scalar then Bigarray.reshape array [||] else array

let fortran_order (type a) ~(layout : a Bigarray.layout) =
  match layout with
  | Bigarray.C_layout -> "False"
  | Bigarray.Fortran_layout -> "True"

let shape ~dims =
  match dims with
  | [| dim1 |] -> Printf.sprintf "%d," dim1
  | dims -> Array.to_list dims |> List.map string_of_int |> String.concat ", "

let full_header ?header_len ~layout ~packed_kind ~dims () =
  let header =
    Printf.sprintf
      "{'descr': '%s', 'fortran_order': %s, 'shape': (%s), }"
      (dtype ~packed_kind)
      (fortran_order ~layout)
      (shape ~dims)
  in
  let padding_len =
    let total_len = String.length header + magic_string_len + 4 + 1 in
    match header_len with
    | None -> if total_len mod 16 = 0 then 0 else 16 - (total_len mod 16)
    | Some header_len ->
      if header_len mod 16 <> 0 then failwith "header_len has to be divisible by 16";
      if header_len < total_len then failwith "header_len is smaller than total_len";
      header_len - total_len
  in
  let total_header_len = String.length header + padding_len + 1 in
  Printf.sprintf
    "%s\001\000%c%c%s%s\n"
    magic_string
    (total_header_len mod 256 |> Char.chr)
    (total_header_len / 256 |> Char.chr)
    header
    (String.make padding_len ' ')

let with_file filename flags mask ~f =
  let file_descr = Unix.openfile filename flags mask in
  try
    let result = f file_descr in
    Unix.close file_descr;
    result
  with
  | exn ->
    Unix.close file_descr;
    raise exn

let write ?header_len bigarray filename =
  with_file filename [ O_CREAT; O_TRUNC; O_RDWR ] 0o640 ~f:(fun file_descr ->
    let full_header =
      full_header
        ()
        ?header_len
        ~layout:(Bigarray.Genarray.layout bigarray)
        ~packed_kind:(P (Bigarray.Genarray.kind bigarray))
        ~dims:(Bigarray.Genarray.dims bigarray)
    in
    let full_header_len = String.length full_header in
    if Unix.write_substring file_descr full_header 0 full_header_len <> full_header_len
    then raise Cannot_write;
    let file_array =
      map_file
        ~pos:(Int64.of_int full_header_len)
        file_descr
        (Bigarray.Genarray.kind bigarray)
        (Bigarray.Genarray.layout bigarray)
        true
        (Bigarray.Genarray.dims bigarray)
    in
    Bigarray.Genarray.blit bigarray file_array)

let write1 array1 filename = write (Bigarray.genarray_of_array1 array1) filename
let write2 array2 filename = write (Bigarray.genarray_of_array2 array2) filename
let write3 array3 filename = write (Bigarray.genarray_of_array3 array3) filename

module Batch_writer = struct
  let header_len = 128

  type t =
    { file_descr : Unix.file_descr
    ; mutable bytes_written_so_far : int
    ; mutable dims_and_packed_kind : (int array * packed_kind) option
    }

  let append t bigarray =
    let file_array =
      map_file
        ~pos:(Int64.of_int t.bytes_written_so_far)
        t.file_descr
        (Bigarray.Genarray.kind bigarray)
        (Bigarray.Genarray.layout bigarray)
        true
        (Bigarray.Genarray.dims bigarray)
    in
    Bigarray.Genarray.blit bigarray file_array;
    let size_in_bytes = Bigarray.Genarray.size_in_bytes bigarray in
    t.bytes_written_so_far <- t.bytes_written_so_far + size_in_bytes;
    match t.dims_and_packed_kind with
    | None ->
      let dims = Bigarray.Genarray.dims bigarray in
      let kind = Bigarray.Genarray.kind bigarray in
      t.dims_and_packed_kind <- Some (dims, P kind)
    | Some (dims, _kind) ->
      let dims' = Bigarray.Genarray.dims bigarray in
      let incorrect_dimensions =
        match Array.to_list dims, Array.to_list dims' with
        | [], _ | _, [] -> true
        | _ :: d, _ :: d' -> d <> d'
      in
      if incorrect_dimensions
      then
        Printf.sprintf "Incorrect dimensions %s vs %s." (shape ~dims) (shape ~dims:dims')
        |> failwith;
      dims.(0) <- dims.(0) + dims'.(0)

  let create filename =
    let file_descr = Unix.openfile filename [ O_CREAT; O_TRUNC; O_RDWR ] 0o640 in
    { file_descr; bytes_written_so_far = header_len; dims_and_packed_kind = None }

  let close t =
    assert (Unix.lseek t.file_descr 0 SEEK_SET = 0);
    let header =
      match t.dims_and_packed_kind with
      | None -> failwith "Nothing to write"
      | Some (dims, packed_kind) ->
        full_header ~header_len ~layout:C_layout ~dims ~packed_kind ()
    in
    if Unix.write_substring t.file_descr header 0 header_len <> header_len
    then raise Cannot_write;
    Unix.close t.file_descr
end

let really_read fd len =
  let buffer = Bytes.create len in
  let rec loop offset =
    let read = Unix.read fd buffer offset (len - offset) in
    if read + offset < len
    then loop (read + offset)
    else if read = 0
    then read_error "unexpected eof"
  in
  loop 0;
  Bytes.to_string buffer

module Header = struct
  type packed_kind = P : (_, _) Bigarray.kind -> packed_kind

  type t =
    { kind : packed_kind
    ; fortran_order : bool
    ; shape : int array
    }

  let split str ~on =
    let parens = ref 0 in
    let indexes = ref [] in
    for i = 0 to String.length str - 1 do
      match str.[i] with
      | '(' -> incr parens
      | ')' -> decr parens
      | c when !parens = 0 && c = on -> indexes := i :: !indexes
      | _ -> ()
    done;
    List.fold_left
      (fun (prev_p, acc) index ->
         index, String.sub str (index + 1) (prev_p - index - 1) :: acc)
      (String.length str, [])
      !indexes
    |> fun (first_pos, acc) -> String.sub str 0 first_pos :: acc

  let trim str ~on =
    let rec loopr start len =
      if len = 0
      then start, len
      else if List.mem str.[start + len - 1] on
      then loopr start (len - 1)
      else start, len
    in
    let rec loopl start len =
      if len = 0
      then start, len
      else if List.mem str.[start] on
      then loopl (start + 1) (len - 1)
      else loopr start len
    in
    let start, len = loopl 0 (String.length str) in
    String.sub str start len

  let parse header =
    let header_fields =
      trim header ~on:[ '{'; ' '; '}'; '\n' ]
      |> split ~on:','
      |> List.map String.trim
      |> List.filter (fun s -> String.length s > 0)
      |> List.map (fun header_field ->
        match split header_field ~on:':' with
        | [ name; value ] ->
          trim name ~on:[ '\''; ' ' ], trim value ~on:[ '\''; ' '; '('; ')' ]
        | _ -> read_error "unable to parse field %s" header_field)
    in
    let find_field field =
      try List.assoc field header_fields with
      | Not_found -> read_error "cannot find field %s" field
    in
    let kind =
      let kind = find_field "descr" in
      (match kind.[0] with
       | '|' | '=' -> ()
       | '>' ->
         if not Sys.big_endian then read_error "big endian data but arch is little endian"
       | '<' ->
         if Sys.big_endian then read_error "little endian data but arch is big endian"
       | otherwise -> read_error "incorrect endianness %c" otherwise);
      match String.sub kind 1 (String.length kind - 1) with
      | "f4" -> P Float32
      | "f8" -> P Float64
      | "i4" -> P Int32
      | "i8" -> P Int64
      | "u1" -> P Int8_unsigned
      | "i1" -> P Int8_signed
      | "u2" -> P Int16_unsigned
      | "i2" -> P Int16_signed
      | "S1" -> P Char
      | "c8" -> P Complex32
      | "c16" -> P Complex64
      | otherwise -> read_error "incorrect descr %s" otherwise
    in
    let fortran_order =
      match find_field "fortran_order" with
      | "False" -> false
      | "True" -> true
      | otherwise -> read_error "incorrect fortran_order %s" otherwise
    in
    let shape =
      find_field "shape"
      |> split ~on:','
      |> List.map String.trim
      |> List.filter (fun s -> String.length s > 0)
      |> List.map int_of_string
      |> Array.of_list
    in
    { kind; fortran_order; shape }
end

type packed_array = P : (_, _, _) Bigarray.Genarray.t -> packed_array
type packed_array1 = P1 : (_, _, _) Bigarray.Array1.t -> packed_array1
type packed_array2 = P2 : (_, _, _) Bigarray.Array2.t -> packed_array2
type packed_array3 = P3 : (_, _, _) Bigarray.Array3.t -> packed_array3

let read_mmap filename ~shared =
  let access = if shared then Unix.O_RDWR else O_RDONLY in
  let file_descr = Unix.openfile filename [ access ] 0 in
  let pos, header =
    try
      let magic_string' = really_read file_descr magic_string_len in
      if magic_string <> magic_string' then read_error "magic string mismatch";
      let version = really_read file_descr 2 |> fun v -> v.[0] |> Char.code in
      let header_len_len =
        match version with
        | 1 -> 2
        | 2 -> 4
        | _ -> read_error "unsupported version %d" version
      in
      let header, header_len =
        really_read file_descr header_len_len
        |> fun str ->
        let header_len = ref 0 in
        for i = String.length str - 1 downto 0 do
          header_len := (256 * !header_len) + Char.code str.[i]
        done;
        really_read file_descr !header_len, !header_len
      in
      let header = Header.parse header in
      Int64.of_int (header_len + header_len_len + magic_string_len + 2), header
    with
    | exn ->
      Unix.close file_descr;
      raise exn
  in
  let (Header.P kind) = header.kind in
  let build layout =
    let array = map_file file_descr ~pos kind layout shared header.shape in
    Gc.finalise (fun _ -> Unix.close file_descr) array;
    P array
  in
  if header.fortran_order then build Fortran_layout else build C_layout

let read_mmap1 filename ~shared =
  let (P array) = read_mmap filename ~shared in
  P1 (Bigarray.array1_of_genarray array)

let read_mmap2 filename ~shared =
  let (P array) = read_mmap filename ~shared in
  P2 (Bigarray.array2_of_genarray array)

let read_mmap3 filename ~shared =
  let (P array) = read_mmap filename ~shared in
  P3 (Bigarray.array3_of_genarray array)

let read_copy filename =
  let (P array) = read_mmap filename ~shared:false in
  let result =
    Bigarray.Genarray.create
      (Bigarray.Genarray.kind array)
      (Bigarray.Genarray.layout array)
      (Bigarray.Genarray.dims array)
  in
  Bigarray.Genarray.blit array result;
  P result

let read_copy1 filename =
  let (P array) = read_copy filename in
  P1 (Bigarray.array1_of_genarray array)

let read_copy2 filename =
  let (P array) = read_copy filename in
  P2 (Bigarray.array2_of_genarray array)

let read_copy3 filename =
  let (P array) = read_copy filename in
  P3 (Bigarray.array3_of_genarray array)

module Npz = struct
  let npy_suffix = ".npy"

  let maybe_add_suffix array_name ~suffix =
    let suffix =
      match suffix with
      | None -> npy_suffix
      | Some suffix -> suffix
    in
    array_name ^ suffix

  type in_file = Zip.in_file

  let open_in = Zip.open_in

  let entries t =
    Zip.entries t
    |> List.map (fun entry ->
      let filename = entry.Zip.filename in
      if String.length filename < String.length npy_suffix
      then filename
      else (
        let start_pos = String.length filename - String.length npy_suffix in
        if String.sub filename start_pos (String.length npy_suffix) = npy_suffix
        then String.sub filename 0 start_pos
        else filename))

  let close_in = Zip.close_in

  let read ?suffix t array_name =
    let array_name = maybe_add_suffix array_name ~suffix in
    let entry =
      try Zip.find_entry t array_name with
      | Not_found -> raise (Invalid_argument ("unable to find " ^ array_name))
    in
    let tmp_file = Filename.temp_file "ocaml-npz" ".tmp" in
    Zip.copy_entry_to_file t entry tmp_file;
    let data = read_copy tmp_file in
    Sys.remove tmp_file;
    data

  type out_file = Zip.out_file

  let open_out filename = Zip.open_out filename
  let close_out = Zip.close_out

  let write ?suffix t array_name array =
    let array_name = maybe_add_suffix array_name ~suffix in
    let tmp_file = Filename.temp_file "ocaml-npz" ".tmp" in
    write array tmp_file;
    Zip.copy_file_to_entry tmp_file t array_name;
    Sys.remove tmp_file
end

(** Type equalities module, used in conversion function *)
module Eq = struct
  (** An equality type to extract type equalities *)
  type ('a, 'b) t = W : ('a, 'a) t

  open Bigarray

  (** Type equalities for bigarray kinds *)
  module Kind = struct
    let ( === )
      : type a b c d. (a, b) kind -> (c, d) kind -> ((a, b) kind, (c, d) kind) t option
      =
      fun x y ->
      match x, y with
      | Float32, Float32 -> Some W
      | Float64, Float64 -> Some W
      | Int8_signed, Int8_signed -> Some W
      | Int8_unsigned, Int8_unsigned -> Some W
      | Int16_signed, Int16_signed -> Some W
      | Int16_unsigned, Int16_unsigned -> Some W
      | Int32, Int32 -> Some W
      | Int64, Int64 -> Some W
      | Int, Int -> Some W
      | Nativeint, Nativeint -> Some W
      | Complex32, Complex32 -> Some W
      | Complex64, Complex64 -> Some W
      | Char, Char -> Some W
      | _ -> None
  end

  (** Type equalities for layout *)
  module Layout = struct
    let ( === ) : type a b. a layout -> b layout -> (a layout, b layout) t option =
      fun x y ->
      match x, y with
      | Fortran_layout, Fortran_layout -> Some W
      | C_layout, C_layout -> Some W
      | _, _ -> None
  end
end

(** Conversion functions from packed arrays to bigarrays *)

let to_bigarray
      (type a b c)
      (layout : c Bigarray.layout)
      (kind : (a, b) Bigarray.kind)
      (P x)
  =
  match Eq.Layout.(Bigarray.Genarray.layout x === layout) with
  | None -> None
  | Some Eq.W ->
    (match Eq.Kind.(Bigarray.Genarray.kind x === kind) with
     | None -> None
     | Some Eq.W -> Some (x : (a, b, c) Bigarray.Genarray.t))

let to_bigarray1
      (type a b c)
      (layout : c Bigarray.layout)
      (kind : (a, b) Bigarray.kind)
      (P1 x)
  =
  match Eq.Layout.(Bigarray.Array1.layout x === layout) with
  | None -> None
  | Some Eq.W ->
    (match Eq.Kind.(Bigarray.Array1.kind x === kind) with
     | None -> None
     | Some Eq.W -> Some (x : (a, b, c) Bigarray.Array1.t))

let to_bigarray2
      (type a b c)
      (layout : c Bigarray.layout)
      (kind : (a, b) Bigarray.kind)
      (P2 x)
  =
  match Eq.Layout.(Bigarray.Array2.layout x === layout) with
  | None -> None
  | Some Eq.W ->
    (match Eq.Kind.(Bigarray.Array2.kind x === kind) with
     | None -> None
     | Some Eq.W -> Some (x : (a, b, c) Bigarray.Array2.t))

let to_bigarray3
      (type a b c)
      (layout : c Bigarray.layout)
      (kind : (a, b) Bigarray.kind)
      (P3 x)
  =
  match Eq.Layout.(Bigarray.Array3.layout x === layout) with
  | None -> None
  | Some Eq.W ->
    (match Eq.Kind.(Bigarray.Array3.kind x === kind) with
     | None -> None
     | Some Eq.W -> Some (x : (a, b, c) Bigarray.Array3.t))