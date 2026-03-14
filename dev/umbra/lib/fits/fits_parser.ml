(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let err_truncated = "Fits: unexpected end of file"
let err_no_simple = "Fits: missing SIMPLE keyword in primary HDU"
let err_bad_tform msg = "Fits: unsupported TFORM: " ^ msg
let block_size = 2880

type keyword = { key : string; value : string; comment : string }

type header = {
  keywords : keyword list;
  xtension : string;
  bitpix : int;
  naxis : int array;
  data_bytes : int;
}

type col_desc = {
  name : string;
  tform : char;
  repeat : int;
  width : int;
  tnull : int64 option;
  tscal : float;
  tzero : float;
}

let swap16 buf pos =
  let b0 = Bytes.get_uint8 buf pos in
  let b1 = Bytes.get_uint8 buf (pos + 1) in
  Bytes.set_uint8 buf pos b1;
  Bytes.set_uint8 buf (pos + 1) b0

let swap32 buf pos =
  let b0 = Bytes.get_uint8 buf pos in
  let b1 = Bytes.get_uint8 buf (pos + 1) in
  let b2 = Bytes.get_uint8 buf (pos + 2) in
  let b3 = Bytes.get_uint8 buf (pos + 3) in
  Bytes.set_uint8 buf pos b3;
  Bytes.set_uint8 buf (pos + 1) b2;
  Bytes.set_uint8 buf (pos + 2) b1;
  Bytes.set_uint8 buf (pos + 3) b0

let swap64 buf pos =
  let b0 = Bytes.get_uint8 buf pos in
  let b1 = Bytes.get_uint8 buf (pos + 1) in
  let b2 = Bytes.get_uint8 buf (pos + 2) in
  let b3 = Bytes.get_uint8 buf (pos + 3) in
  let b4 = Bytes.get_uint8 buf (pos + 4) in
  let b5 = Bytes.get_uint8 buf (pos + 5) in
  let b6 = Bytes.get_uint8 buf (pos + 6) in
  let b7 = Bytes.get_uint8 buf (pos + 7) in
  Bytes.set_uint8 buf pos b7;
  Bytes.set_uint8 buf (pos + 1) b6;
  Bytes.set_uint8 buf (pos + 2) b5;
  Bytes.set_uint8 buf (pos + 3) b4;
  Bytes.set_uint8 buf (pos + 4) b3;
  Bytes.set_uint8 buf (pos + 5) b2;
  Bytes.set_uint8 buf (pos + 6) b1;
  Bytes.set_uint8 buf (pos + 7) b0

let trim_right s =
  let len = String.length s in
  let i = ref (len - 1) in
  while !i >= 0 && s.[!i] = ' ' do
    decr i
  done;
  if !i = len - 1 then s else String.sub s 0 (!i + 1)

let parse_card card =
  let key = trim_right (String.sub card 0 8) in
  if key = "COMMENT" || key = "HISTORY" then
    let content =
      if String.length card > 8 then
        trim_right (String.sub card 8 (String.length card - 8))
      else ""
    in
    { key; value = content; comment = "" }
  else if String.length card < 10 || card.[8] <> '=' || card.[9] <> ' ' then
    { key; value = ""; comment = "" }
  else
    let rest = String.sub card 10 (String.length card - 10) in
    let rest = String.trim rest in
    if String.length rest > 0 && rest.[0] = '\'' then begin
      let len = String.length rest in
      let i = ref 1 in
      let buf = Buffer.create 68 in
      while !i < len do
        if rest.[!i] = '\'' then begin
          if !i + 1 < len && rest.[!i + 1] = '\'' then begin
            Buffer.add_char buf '\'';
            i := !i + 2
          end
          else i := len
        end
        else begin
          Buffer.add_char buf rest.[!i];
          i := !i + 1
        end
      done;
      { key; value = trim_right (Buffer.contents buf); comment = "" }
    end
    else begin
      match String.index_opt rest '/' with
      | Some i ->
          let value = trim_right (String.sub rest 0 i) in
          let comment =
            String.trim (String.sub rest (i + 1) (String.length rest - i - 1))
          in
          { key; value; comment }
      | None -> { key; value = trim_right rest; comment = "" }
    end

let read_one_header ic =
  let keywords = ref [] in
  let found_end = ref false in
  let card_buf = Bytes.create 80 in
  while not !found_end do
    let block = Bytes.create block_size in
    (match In_channel.really_input ic block 0 block_size with
    | None -> failwith err_truncated
    | Some () -> ());
    for card_i = 0 to 35 do
      if not !found_end then begin
        Bytes.blit block (card_i * 80) card_buf 0 80;
        let card = Bytes.to_string card_buf in
        let key = trim_right (String.sub card 0 8) in
        if key = "END" then found_end := true
        else if key <> "" then keywords := parse_card card :: !keywords
      end
    done
  done;
  List.rev !keywords

let find_keyword keywords key =
  match List.find_opt (fun kw -> kw.key = key) keywords with
  | Some kw -> Some kw.value
  | None -> None

let find_keyword_int keywords key =
  match find_keyword keywords key with
  | Some v -> Some (int_of_string (String.trim v))
  | None -> None

let find_keyword_exn keywords key =
  match find_keyword keywords key with
  | Some v -> v
  | None -> failwith ("Fits: missing required keyword " ^ key)

let find_keyword_int_exn keywords key =
  int_of_string (String.trim (find_keyword_exn keywords key))

let compute_data_bytes keywords =
  let bitpix = find_keyword_int_exn keywords "BITPIX" in
  let naxis_n = find_keyword_int_exn keywords "NAXIS" in
  if naxis_n = 0 then 0
  else begin
    let total = ref (abs bitpix / 8) in
    for i = 1 to naxis_n do
      let key = Printf.sprintf "NAXIS%d" i in
      total := !total * find_keyword_int_exn keywords key
    done;
    let pcount =
      match find_keyword_int keywords "PCOUNT" with Some v -> v | None -> 0
    in
    let gcount =
      match find_keyword_int keywords "GCOUNT" with Some v -> v | None -> 1
    in
    (!total + pcount) * gcount
  end

let build_header keywords =
  let bitpix = find_keyword_int_exn keywords "BITPIX" in
  let naxis_n = find_keyword_int_exn keywords "NAXIS" in
  let naxis =
    Array.init naxis_n (fun i ->
        find_keyword_int_exn keywords (Printf.sprintf "NAXIS%d" (i + 1)))
  in
  let xtension =
    match find_keyword keywords "XTENSION" with Some v -> v | None -> ""
  in
  let data_bytes = compute_data_bytes keywords in
  { keywords; xtension; bitpix; naxis; data_bytes }

let read_headers ic =
  In_channel.seek ic 0L;
  let headers = ref [] in
  let first = ref true in
  let continue = ref true in
  while !continue do
    let keywords = try Some (read_one_header ic) with Failure _ -> None in
    match keywords with
    | None -> continue := false
    | Some keywords ->
        if !first then begin
          first := false;
          match find_keyword keywords "SIMPLE" with
          | Some _ -> ()
          | None -> failwith err_no_simple
        end;
        let hdr = build_header keywords in
        headers := hdr :: !headers;
        let data_blocks =
          if hdr.data_bytes = 0 then 0
          else (hdr.data_bytes + block_size - 1) / block_size
        in
        In_channel.seek ic
          (Int64.add (In_channel.pos ic)
             (Int64.of_int (data_blocks * block_size)))
  done;
  List.rev !headers

let seek_to_data ic headers hdu =
  if hdu < 0 || hdu >= List.length headers then
    failwith
      (Printf.sprintf "Fits: HDU %d out of range (file has %d)" hdu
         (List.length headers));
  In_channel.seek ic 0L;
  for i = 0 to hdu do
    let h = List.nth headers i in
    let found_end = ref false in
    while not !found_end do
      let block = Bytes.create block_size in
      (match In_channel.really_input ic block 0 block_size with
      | None -> failwith err_truncated
      | Some () -> ());
      for card_i = 0 to 35 do
        if not !found_end then begin
          let key = trim_right (Bytes.sub_string block (card_i * 80) 8) in
          if key = "END" then found_end := true
        end
      done
    done;
    if i < hdu then begin
      let data_blocks =
        if h.data_bytes = 0 then 0
        else (h.data_bytes + block_size - 1) / block_size
      in
      In_channel.seek ic
        (Int64.add (In_channel.pos ic)
           (Int64.of_int (data_blocks * block_size)))
    end
  done;
  let h = List.nth headers hdu in
  h.data_bytes

let parse_tform s =
  let s = String.trim s in
  let len = String.length s in
  if len = 0 then failwith (err_bad_tform "empty");
  let i = ref 0 in
  while !i < len && s.[!i] >= '0' && s.[!i] <= '9' do
    incr i
  done;
  let repeat = if !i = 0 then 1 else int_of_string (String.sub s 0 !i) in
  if !i >= len then failwith (err_bad_tform s);
  let code = s.[!i] in
  let width =
    match code with
    | 'L' -> 1
    | 'B' -> 1
    | 'I' -> 2
    | 'J' -> 4
    | 'K' -> 8
    | 'E' -> 4
    | 'D' -> 8
    | 'A' -> 1
    | c -> failwith (err_bad_tform (String.make 1 c))
  in
  (code, repeat, width)

let parse_bintable_cols hdr =
  let keywords = hdr.keywords in
  let tfields = find_keyword_int_exn keywords "TFIELDS" in
  List.init tfields (fun i ->
      let col = i + 1 in
      let name =
        match find_keyword keywords (Printf.sprintf "TTYPE%d" col) with
        | Some v -> v
        | None -> Printf.sprintf "col%d" col
      in
      let tform_s = find_keyword_exn keywords (Printf.sprintf "TFORM%d" col) in
      let tform, repeat, width = parse_tform tform_s in
      let tnull =
        match find_keyword keywords (Printf.sprintf "TNULL%d" col) with
        | Some v -> Some (Int64.of_string (String.trim v))
        | None -> None
      in
      let tscal =
        match find_keyword keywords (Printf.sprintf "TSCAL%d" col) with
        | Some v -> float_of_string (String.trim v)
        | None -> 1.0
      in
      let tzero =
        match find_keyword keywords (Printf.sprintf "TZERO%d" col) with
        | Some v -> float_of_string (String.trim v)
        | None -> 0.0
      in
      { name; tform; repeat; width; tnull; tscal; tzero })
