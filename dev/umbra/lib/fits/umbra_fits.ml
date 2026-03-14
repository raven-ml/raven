(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let err_not_bintable = "Fits.read_table: HDU is not a BINTABLE"
let err_not_image = "Fits.read_image: HDU is not an image"
let err_unsupported_bitpix n = Printf.sprintf "Fits: unsupported BITPIX %d" n
let err_truncated_data = "Fits: unexpected end of file in data"

type header_card = { key : string; value : string; comment : string }
type hdu_type = Primary | Image | Bintable | Ascii_table

type hdu_info = {
  index : int;
  hdu_type : hdu_type;
  dimensions : int array;
  num_rows : int option;
  num_cols : int option;
}

let hdu_type_of_header i (hdr : Fits_parser.header) =
  match hdr.xtension with
  | "" -> if i = 0 then Primary else Image
  | "BINTABLE" -> Bintable
  | "TABLE" -> Ascii_table
  | "IMAGE" -> Image
  | _ -> Image

let read_input ic buf n =
  match In_channel.really_input ic buf 0 n with
  | None -> failwith err_truncated_data
  | Some () -> ()

let info path =
  let ic = In_channel.open_bin path in
  Fun.protect
    ~finally:(fun () -> In_channel.close ic)
    (fun () ->
      let headers = Fits_parser.read_headers ic in
      List.mapi
        (fun i (hdr : Fits_parser.header) ->
          let ht = hdu_type_of_header i hdr in
          let num_rows, num_cols =
            match ht with
            | Bintable | Ascii_table ->
                let nrows =
                  if Array.length hdr.naxis >= 2 then Some hdr.naxis.(1)
                  else None
                in
                let ncols =
                  Fits_parser.find_keyword_int hdr.keywords "TFIELDS"
                in
                (nrows, ncols)
            | _ -> (None, None)
          in
          {
            index = i;
            hdu_type = ht;
            dimensions = hdr.naxis;
            num_rows;
            num_cols;
          })
        headers)

let header ?(hdu = 0) path =
  let ic = In_channel.open_bin path in
  Fun.protect
    ~finally:(fun () -> In_channel.close ic)
    (fun () ->
      let headers = Fits_parser.read_headers ic in
      if hdu < 0 || hdu >= List.length headers then
        failwith (Printf.sprintf "Fits.header: HDU %d out of range" hdu);
      let h = List.nth headers hdu in
      List.map
        (fun (kw : Fits_parser.keyword) ->
          { key = kw.key; value = kw.value; comment = kw.comment })
        h.keywords)

let read_table ?(hdu = 1) path =
  let ic = In_channel.open_bin path in
  Fun.protect
    ~finally:(fun () -> In_channel.close ic)
    (fun () ->
      let headers = Fits_parser.read_headers ic in
      if hdu < 0 || hdu >= List.length headers then
        failwith (Printf.sprintf "Fits.read_table: HDU %d out of range" hdu);
      let h = List.nth headers hdu in
      (match hdu_type_of_header hdu h with
      | Bintable -> ()
      | _ -> failwith err_not_bintable);
      let cols = Fits_parser.parse_bintable_cols h in
      let nrows = if Array.length h.naxis >= 2 then h.naxis.(1) else 0 in
      let row_bytes = h.naxis.(0) in
      let (_ : int) = Fits_parser.seek_to_data ic headers hdu in
      let row_buf = Bytes.create row_bytes in
      let col_info =
        List.map
          (fun (cd : Fits_parser.col_desc) ->
            let elem_bytes = cd.repeat * cd.width in
            (cd, Bytes.create (nrows * elem_bytes), elem_bytes))
          cols
      in
      let col_offsets =
        let off = ref 0 in
        List.map
          (fun (cd : Fits_parser.col_desc) ->
            let o = !off in
            off := !off + (cd.repeat * cd.width);
            o)
          cols
      in
      for row = 0 to nrows - 1 do
        read_input ic row_buf row_bytes;
        List.iter2
          (fun offset (_cd, buf, elem_bytes) ->
            Bytes.blit row_buf offset buf (row * elem_bytes) elem_bytes)
          col_offsets col_info
      done;
      let err_vector name repeat =
        failwith
          (Printf.sprintf "Fits: vector column '%s' (repeat=%d) not supported"
             name repeat)
      in
      let talon_cols =
        List.map
          (fun (cd, buf, _) ->
            let col =
              match cd.Fits_parser.tform with
              | 'E' ->
                  if cd.repeat <> 1 then err_vector cd.name cd.repeat;
                  Talon.Col.float32
                    (Array.init nrows (fun i ->
                         let pos = i * 4 in
                         Fits_parser.swap32 buf pos;
                         let v =
                           Int32.float_of_bits (Bytes.get_int32_le buf pos)
                         in
                         if cd.tzero = 0.0 && cd.tscal = 1.0 then v
                         else (v *. cd.tscal) +. cd.tzero))
              | 'D' ->
                  if cd.repeat <> 1 then err_vector cd.name cd.repeat;
                  Talon.Col.float64
                    (Array.init nrows (fun i ->
                         let pos = i * 8 in
                         Fits_parser.swap64 buf pos;
                         let v =
                           Int64.float_of_bits (Bytes.get_int64_le buf pos)
                         in
                         if cd.tzero = 0.0 && cd.tscal = 1.0 then v
                         else (v *. cd.tscal) +. cd.tzero))
              | 'J' ->
                  if cd.repeat <> 1 then err_vector cd.name cd.repeat;
                  Talon.Col.int32
                    (Array.init nrows (fun i ->
                         let pos = i * 4 in
                         Fits_parser.swap32 buf pos;
                         let v = Bytes.get_int32_le buf pos in
                         if cd.tzero = 0.0 && cd.tscal = 1.0 then v
                         else
                           Int32.of_float
                             ((Int32.to_float v *. cd.tscal) +. cd.tzero)))
              | 'K' ->
                  if cd.repeat <> 1 then err_vector cd.name cd.repeat;
                  Talon.Col.int64
                    (Array.init nrows (fun i ->
                         let pos = i * 8 in
                         Fits_parser.swap64 buf pos;
                         let v = Bytes.get_int64_le buf pos in
                         if cd.tzero = 0.0 && cd.tscal = 1.0 then v
                         else
                           Int64.of_float
                             ((Int64.to_float v *. cd.tscal) +. cd.tzero)))
              | 'I' ->
                  if cd.repeat <> 1 then err_vector cd.name cd.repeat;
                  Talon.Col.int32
                    (Array.init nrows (fun i ->
                         let pos = i * 2 in
                         Fits_parser.swap16 buf pos;
                         let v = Bytes.get_int16_le buf pos in
                         if cd.tzero = 0.0 && cd.tscal = 1.0 then Int32.of_int v
                         else
                           Int32.of_float
                             ((Float.of_int v *. cd.tscal) +. cd.tzero)))
              | 'B' ->
                  if cd.repeat <> 1 then err_vector cd.name cd.repeat;
                  Talon.Col.int32
                    (Array.init nrows (fun i ->
                         let v = Bytes.get_uint8 buf i in
                         if cd.tzero = 0.0 && cd.tscal = 1.0 then Int32.of_int v
                         else
                           Int32.of_float
                             ((Float.of_int v *. cd.tscal) +. cd.tzero)))
              | 'L' ->
                  if cd.repeat <> 1 then err_vector cd.name cd.repeat;
                  Talon.Col.bool
                    (Array.init nrows (fun i ->
                         let c = Bytes.get buf i in
                         c = 'T' || c = '\x01'))
              | 'A' ->
                  Talon.Col.string
                    (Array.init nrows (fun i ->
                         Fits_parser.trim_right
                           (Bytes.sub_string buf (i * cd.repeat) cd.repeat)))
              | c -> failwith (Printf.sprintf "Fits: unsupported TFORM '%c'" c)
            in
            (cd.name, col))
          col_info
      in
      Talon.create talon_cols)

let find_keyword_float keywords key =
  match Fits_parser.find_keyword keywords key with
  | Some v -> Some (float_of_string (String.trim v))
  | None -> None

let read_image ?(hdu = 0) path =
  let ic = In_channel.open_bin path in
  Fun.protect
    ~finally:(fun () -> In_channel.close ic)
    (fun () ->
      let headers = Fits_parser.read_headers ic in
      if hdu < 0 || hdu >= List.length headers then
        failwith (Printf.sprintf "Fits.read_image: HDU %d out of range" hdu);
      let h = List.nth headers hdu in
      (match hdu_type_of_header hdu h with
      | Primary | Image -> ()
      | _ -> failwith err_not_image);
      let bscale =
        match find_keyword_float h.keywords "BSCALE" with
        | Some v -> v
        | None -> 1.0
      in
      let bzero =
        match find_keyword_float h.keywords "BZERO" with
        | Some v -> v
        | None -> 0.0
      in
      let has_scaling = bscale <> 1.0 || bzero <> 0.0 in
      let (_ : int) = Fits_parser.seek_to_data ic headers hdu in
      let shape = Array.to_list h.naxis |> List.rev |> Array.of_list in
      let total = Array.fold_left ( * ) 1 shape in
      let apply_scaling raw =
        Nx.add_s (Nx.mul_s (Nx.astype Nx.float64 raw) bscale) bzero
      in
      match h.bitpix with
      | 8 ->
          let buf = Bytes.create total in
          read_input ic buf total;
          let raw =
            Nx.create Nx.uint8 shape
              (Array.init total (fun i -> Bytes.get_uint8 buf i))
          in
          if has_scaling then Nx_io.P (apply_scaling raw) else Nx_io.P raw
      | 16 ->
          let buf = Bytes.create (total * 2) in
          read_input ic buf (total * 2);
          let raw =
            Nx.create Nx.int16 shape
              (Array.init total (fun i ->
                   let pos = i * 2 in
                   Fits_parser.swap16 buf pos;
                   Bytes.get_int16_le buf pos))
          in
          if has_scaling then Nx_io.P (apply_scaling raw) else Nx_io.P raw
      | 32 ->
          let buf = Bytes.create (total * 4) in
          read_input ic buf (total * 4);
          let raw =
            Nx.create Nx.int32 shape
              (Array.init total (fun i ->
                   let pos = i * 4 in
                   Fits_parser.swap32 buf pos;
                   Bytes.get_int32_le buf pos))
          in
          if has_scaling then Nx_io.P (apply_scaling raw) else Nx_io.P raw
      | 64 ->
          let buf = Bytes.create (total * 8) in
          read_input ic buf (total * 8);
          let raw =
            Nx.create Nx.int64 shape
              (Array.init total (fun i ->
                   let pos = i * 8 in
                   Fits_parser.swap64 buf pos;
                   Bytes.get_int64_le buf pos))
          in
          if has_scaling then Nx_io.P (apply_scaling raw) else Nx_io.P raw
      | -32 ->
          let buf = Bytes.create (total * 4) in
          read_input ic buf (total * 4);
          let raw =
            Nx.create Nx.float32 shape
              (Array.init total (fun i ->
                   let pos = i * 4 in
                   Fits_parser.swap32 buf pos;
                   Int32.float_of_bits (Bytes.get_int32_le buf pos)))
          in
          if has_scaling then Nx_io.P (apply_scaling raw) else Nx_io.P raw
      | -64 ->
          let buf = Bytes.create (total * 8) in
          read_input ic buf (total * 8);
          let raw =
            Nx.create Nx.float64 shape
              (Array.init total (fun i ->
                   let pos = i * 8 in
                   Fits_parser.swap64 buf pos;
                   Int64.float_of_bits (Bytes.get_int64_le buf pos)))
          in
          if has_scaling then Nx_io.P (apply_scaling raw) else Nx_io.P raw
      | n -> failwith (err_unsupported_bitpix n))

let pad_to_block oc written =
  let rem = written mod Fits_parser.block_size in
  if rem > 0 then
    output_string oc (String.make (Fits_parser.block_size - rem) '\x00')

let write_card oc key value =
  let card = Bytes.make 80 ' ' in
  Bytes.blit_string key 0 card 0 (Int.min 8 (String.length key));
  Bytes.set card 8 '=';
  Bytes.set card 9 ' ';
  let v = String.trim value in
  Bytes.blit_string v 0 card 10 (Int.min 70 (String.length v));
  output_bytes oc card

let write_card_str oc key value =
  write_card oc key (Printf.sprintf "'%-8s'" value)

let write_card_int oc key value =
  write_card oc key (Printf.sprintf "%20d" value)

let write_end oc cards_written =
  let card = Bytes.make 80 ' ' in
  Bytes.blit_string "END" 0 card 0 3;
  output_bytes oc card;
  let total_cards = cards_written + 1 in
  let rem = total_cards * 80 mod Fits_parser.block_size in
  if rem > 0 then
    output_string oc (String.make (Fits_parser.block_size - rem) ' ')

let write_empty_primary oc =
  write_card oc "SIMPLE" "                   T";
  write_card_int oc "BITPIX" 8;
  write_card_int oc "NAXIS" 0;
  write_end oc 3

let write_image_typed (type a b) ?(overwrite = true) path (tensor : (a, b) Nx.t)
    =
  if (not overwrite) && Sys.file_exists path then
    failwith ("Fits.write_image: file exists: " ^ path);
  let oc = Out_channel.open_bin path in
  Fun.protect
    ~finally:(fun () -> Out_channel.close oc)
    (fun () ->
      let shape = Nx.shape tensor in
      let ndim = Array.length shape in
      let fits_shape = Array.init ndim (fun i -> shape.(ndim - 1 - i)) in
      let total = Nx.numel tensor in
      let dt = Nx.dtype_to_string (Nx.dtype tensor) in
      let bitpix, elem_bytes =
        match dt with
        | "uint8" -> (8, 1)
        | "int16" -> (16, 2)
        | "int32" -> (32, 4)
        | "int64" -> (64, 8)
        | "float32" -> (-32, 4)
        | "float64" -> (-64, 8)
        | s -> failwith ("Fits.write_image: unsupported dtype " ^ s)
      in
      write_card oc "SIMPLE" "                   T";
      write_card_int oc "BITPIX" bitpix;
      write_card_int oc "NAXIS" ndim;
      for i = 0 to ndim - 1 do
        write_card_int oc (Printf.sprintf "NAXIS%d" (i + 1)) fits_shape.(i)
      done;
      write_end oc (3 + ndim);
      let flat = Nx.reshape [| total |] tensor in
      let arr = Nx.to_array flat in
      let data_bytes = total * elem_bytes in
      let buf = Bytes.create data_bytes in
      (match dt with
      | "uint8" ->
          Array.iteri
            (fun i (v : a) -> Bytes.set_uint8 buf i (Obj.magic v : int))
            arr
      | "int16" ->
          Array.iteri
            (fun i (v : a) ->
              let pos = i * 2 in
              Bytes.set_int16_le buf pos (Obj.magic v : int);
              Fits_parser.swap16 buf pos)
            arr
      | "int32" ->
          Array.iteri
            (fun i (v : a) ->
              let pos = i * 4 in
              Bytes.set_int32_le buf pos (Obj.magic v : int32);
              Fits_parser.swap32 buf pos)
            arr
      | "int64" ->
          Array.iteri
            (fun i (v : a) ->
              let pos = i * 8 in
              Bytes.set_int64_le buf pos (Obj.magic v : int64);
              Fits_parser.swap64 buf pos)
            arr
      | "float32" ->
          Array.iteri
            (fun i (v : a) ->
              let pos = i * 4 in
              Bytes.set_int32_le buf pos
                (Int32.bits_of_float (Obj.magic v : float));
              Fits_parser.swap32 buf pos)
            arr
      | "float64" ->
          Array.iteri
            (fun i (v : a) ->
              let pos = i * 8 in
              Bytes.set_int64_le buf pos
                (Int64.bits_of_float (Obj.magic v : float));
              Fits_parser.swap64 buf pos)
            arr
      | _ -> assert false);
      output_bytes oc buf;
      pad_to_block oc data_bytes)

let write_image ?overwrite path tensor =
  write_image_typed ?overwrite path tensor

let write_table ?(overwrite = true) path df =
  if (not overwrite) && Sys.file_exists path then
    failwith ("Fits.write_table: file exists: " ^ path);
  let oc = Out_channel.open_bin path in
  Fun.protect
    ~finally:(fun () -> Out_channel.close oc)
    (fun () ->
      write_empty_primary oc;
      let col_names = Talon.column_names df in
      let nrows = Talon.num_rows df in
      let ncols = List.length col_names in
      let col_info =
        List.map
          (fun name ->
            let col = Talon.get_column_exn df name in
            match Talon.Col.dtype col with
            | `Float32 -> (name, col, "1E", 4)
            | `Float64 -> (name, col, "1D", 8)
            | `Int32 -> (name, col, "1J", 4)
            | `Int64 -> (name, col, "1K", 8)
            | `String -> (
                match Talon.to_string_array df name with
                | Some arr ->
                    let maxlen =
                      Array.fold_left
                        (fun acc v ->
                          match v with
                          | Some s -> max acc (String.length s)
                          | None -> acc)
                        1 arr
                    in
                    (name, col, Printf.sprintf "%dA" maxlen, maxlen)
                | None -> failwith "Fits.write_table: string column missing")
            | `Bool -> (name, col, "1L", 1)
            | `Other -> failwith "Fits.write_table: unsupported dtype")
          col_names
      in
      let row_bytes =
        List.fold_left (fun acc (_, _, _, eb) -> acc + eb) 0 col_info
      in
      write_card_str oc "XTENSION" "BINTABLE";
      write_card_int oc "BITPIX" 8;
      write_card_int oc "NAXIS" 2;
      write_card_int oc "NAXIS1" row_bytes;
      write_card_int oc "NAXIS2" nrows;
      write_card_int oc "PCOUNT" 0;
      write_card_int oc "GCOUNT" 1;
      write_card_int oc "TFIELDS" ncols;
      let cards = ref 8 in
      List.iteri
        (fun i (name, _col, tform, _eb) ->
          let n = i + 1 in
          write_card_str oc (Printf.sprintf "TTYPE%d" n) name;
          write_card_str oc (Printf.sprintf "TFORM%d" n) tform;
          cards := !cards + 2)
        col_info;
      write_end oc !cards;
      let col_arrays =
        List.map
          (fun (name, col, _tform, _eb) ->
            match Talon.Col.dtype col with
            | `Float32 -> (
                match Talon.to_array Nx.float32 df name with
                | Some a -> `F32 a
                | None -> assert false)
            | `Float64 -> (
                match Talon.to_array Nx.float64 df name with
                | Some a -> `F64 a
                | None -> assert false)
            | `Int32 -> (
                match Talon.to_array Nx.int32 df name with
                | Some a -> `I32 a
                | None -> assert false)
            | `Int64 -> (
                match Talon.to_array Nx.int64 df name with
                | Some a -> `I64 a
                | None -> assert false)
            | `String -> (
                match Talon.to_string_array df name with
                | Some a -> `Str a
                | None -> assert false)
            | `Bool -> (
                match Talon.to_bool_array df name with
                | Some a -> `Bool a
                | None -> assert false)
            | `Other -> failwith "Fits.write_table: unsupported dtype")
          col_info
      in
      let row_buf = Bytes.create row_bytes in
      for row = 0 to nrows - 1 do
        let off = ref 0 in
        List.iter2
          (fun (_, _, _, eb) col_arr ->
            (match col_arr with
            | `F32 arr ->
                Bytes.set_int32_le row_buf !off (Int32.bits_of_float arr.(row));
                Fits_parser.swap32 row_buf !off
            | `F64 arr ->
                Bytes.set_int64_le row_buf !off (Int64.bits_of_float arr.(row));
                Fits_parser.swap64 row_buf !off
            | `I32 arr ->
                Bytes.set_int32_le row_buf !off arr.(row);
                Fits_parser.swap32 row_buf !off
            | `I64 arr ->
                Bytes.set_int64_le row_buf !off arr.(row);
                Fits_parser.swap64 row_buf !off
            | `Str arr -> (
                Bytes.fill row_buf !off eb ' ';
                match arr.(row) with
                | Some s ->
                    let len = Int.min eb (String.length s) in
                    Bytes.blit_string s 0 row_buf !off len
                | None -> ())
            | `Bool arr ->
                let v = match arr.(row) with Some true -> 'T' | _ -> 'F' in
                Bytes.set row_buf !off v);
            off := !off + eb)
          col_info col_arrays;
        output_bytes oc row_buf
      done;
      pad_to_block oc (nrows * row_bytes))
