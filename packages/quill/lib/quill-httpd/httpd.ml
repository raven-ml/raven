(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let err_malformed_req = "malformed request line"
let err_unsupported_meth = "unsupported method"
let err_missing_ws_key = "missing Sec-WebSocket-Key"
let err_ws_eof : _ format = "httpd: ws: end of file\n%!"
let err_ws_unix : _ format = "httpd: ws: %s in %s\n%!"
let err_handler : _ format = "httpd: handler: %s\n%!"
let err_parse : _ format = "httpd: parse: %s\n%!"
let err_connection : _ format = "httpd: connection: %s\n%!"
let err_accept : _ format = "httpd: accept: %s\n%!"

(*---------------------------------------------------------------------------
  String and URL utilities
  ---------------------------------------------------------------------------*)

let str_equal_case_insensitive a b =
  let len = String.length a in
  if len <> String.length b then false
  else
    let rec loop i =
      if i = len then true
      else
        let ca = Char.code a.[i] in
        let cb = Char.code b.[i] in
        let ca = if ca >= 65 && ca <= 90 then ca + 32 else ca in
        let cb = if cb >= 65 && cb <= 90 then cb + 32 else cb in
        if ca = cb then loop (i + 1) else false
    in
    loop 0

let sub_equal_case_insensitive s off len target =
  if len <> String.length target then false
  else
    let rec loop i =
      if i = len then true
      else
        let ca = Char.code s.[off + i] in
        let cb = Char.code target.[i] in
        let ca = if ca >= 65 && ca <= 90 then ca + 32 else ca in
        let cb = if cb >= 65 && cb <= 90 then cb + 32 else cb in
        if ca = cb then loop (i + 1) else false
    in
    loop 0

let starts_with prefix s =
  let len_p = String.length prefix in
  if String.length s < len_p then false
  else
    let rec loop i =
      if i = len_p then true
      else if s.[i] = prefix.[i] then loop (i + 1)
      else false
    in
    loop 0

let url_decode s =
  let hex c =
    if c >= '0' && c <= '9' then Char.code c - Char.code '0'
    else if c >= 'a' && c <= 'f' then Char.code c - Char.code 'a' + 10
    else if c >= 'A' && c <= 'F' then Char.code c - Char.code 'A' + 10
    else -1
  in
  let len = String.length s in
  let buf = Buffer.create len in
  let rec loop i =
    if i >= len then Buffer.contents buf
    else
      match String.unsafe_get s i with
      | '%' when i + 2 < len ->
          let h = hex (String.unsafe_get s (i + 1)) in
          let l = hex (String.unsafe_get s (i + 2)) in
          if h >= 0 && l >= 0 then begin
            Buffer.add_char buf (Char.chr ((h lsl 4) lor l));
            loop (i + 3)
          end
          else begin
            Buffer.add_char buf '%';
            loop (i + 1)
          end
      | '+' ->
          Buffer.add_char buf ' ';
          loop (i + 1)
      | c ->
          Buffer.add_char buf c;
          loop (i + 1)
  in
  loop 0

let parse_query_string s =
  if String.length s = 0 then []
  else
    let rec split acc start i =
      if i = String.length s then
        if start < i then String.sub s start (i - start) :: acc else acc
      else if String.unsafe_get s i = '&' then
        split (String.sub s start (i - start) :: acc) (i + 1) (i + 1)
      else split acc start (i + 1)
    in
    let pairs = split [] 0 0 in
    let rec decode_pairs acc = function
      | [] -> acc
      | pair :: rest -> (
          match String.index_opt pair '=' with
          | Some i ->
              let k = url_decode (String.sub pair 0 i) in
              let v =
                url_decode
                  (String.sub pair (i + 1) (String.length pair - i - 1))
              in
              decode_pairs ((k, v) :: acc) rest
          | None ->
              if String.length pair > 0 then
                decode_pairs ((url_decode pair, "") :: acc) rest
              else decode_pairs acc rest)
    in
    decode_pairs [] pairs

(*---------------------------------------------------------------------------
  MIME types and HTTP reasons
  ---------------------------------------------------------------------------*)

let mime_of_ext = function
  | ".html" | ".htm" -> "text/html; charset=utf-8"
  | ".css" -> "text/css; charset=utf-8"
  | ".js" | ".mjs" -> "application/javascript; charset=utf-8"
  | ".json" | ".map" -> "application/json; charset=utf-8"
  | ".png" -> "image/png"
  | ".jpg" | ".jpeg" -> "image/jpeg"
  | ".gif" -> "image/gif"
  | ".svg" -> "image/svg+xml"
  | ".ico" -> "image/x-icon"
  | ".woff" -> "font/woff"
  | ".woff2" -> "font/woff2"
  | ".ttf" -> "font/ttf"
  | ".otf" -> "font/otf"
  | ".wasm" -> "application/wasm"
  | ".txt" | ".md" -> "text/plain; charset=utf-8"
  | ".xml" -> "application/xml"
  | _ -> "application/octet-stream"

let mime_of_path path = mime_of_ext (Filename.extension path)

let reason_phrase = function
  | 100 -> "Continue"
  | 101 -> "Switching Protocols"
  | 200 -> "OK"
  | 201 -> "Created"
  | 204 -> "No Content"
  | 301 -> "Moved Permanently"
  | 302 -> "Found"
  | 304 -> "Not Modified"
  | 400 -> "Bad Request"
  | 401 -> "Unauthorized"
  | 403 -> "Forbidden"
  | 404 -> "Not Found"
  | 405 -> "Method Not Allowed"
  | 413 -> "Content Too Large"
  | 426 -> "Upgrade Required"
  | 500 -> "Internal Server Error"
  | code -> string_of_int code

(*---------------------------------------------------------------------------
  SHA-1 and Base64 (for WebSocket handshake)
  ---------------------------------------------------------------------------*)

module Ws_crypto = struct
  let base64_encode s =
    let alpha =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    in
    let rec loop len e ei s i =
      if i >= len then Bytes.unsafe_to_string e
      else
        let i0 = i and i1 = i + 1 and i2 = i + 2 in
        let b0 = Char.code s.[i0] in
        let b1 = if i1 >= len then 0 else Char.code s.[i1] in
        let b2 = if i2 >= len then 0 else Char.code s.[i2] in
        let u = (b0 lsl 16) lor (b1 lsl 8) lor b2 in
        Bytes.set e ei alpha.[u lsr 18];
        Bytes.set e (ei + 1) alpha.[(u lsr 12) land 63];
        Bytes.set e (ei + 2)
          (if i1 >= len then '=' else alpha.[(u lsr 6) land 63]);
        Bytes.set e (ei + 3) (if i2 >= len then '=' else alpha.[u land 63]);
        loop len e (ei + 4) s (i2 + 1)
    in
    match String.length s with
    | 0 -> ""
    | len -> loop len (Bytes.create ((len + 2) / 3 * 4)) 0 s 0

  let sha1 s =
    let sha_1_pad s =
      let len = String.length s in
      let blen = 8 * len in
      let rem = len mod 64 in
      let mlen = if rem > 55 then len + 128 - rem else len + 64 - rem in
      let m = Bytes.create mlen in
      Bytes.blit_string s 0 m 0 len;
      Bytes.fill m len (mlen - len) '\x00';
      Bytes.set m len '\x80';
      if Sys.word_size > 32 then begin
        Bytes.set m (mlen - 8) (Char.unsafe_chr ((blen lsr 56) land 0xFF));
        Bytes.set m (mlen - 7) (Char.unsafe_chr ((blen lsr 48) land 0xFF));
        Bytes.set m (mlen - 6) (Char.unsafe_chr ((blen lsr 40) land 0xFF));
        Bytes.set m (mlen - 5) (Char.unsafe_chr ((blen lsr 32) land 0xFF))
      end;
      Bytes.set m (mlen - 4) (Char.unsafe_chr ((blen lsr 24) land 0xFF));
      Bytes.set m (mlen - 3) (Char.unsafe_chr ((blen lsr 16) land 0xFF));
      Bytes.set m (mlen - 2) (Char.unsafe_chr ((blen lsr 8) land 0xFF));
      Bytes.set m (mlen - 1) (Char.unsafe_chr (blen land 0xFF));
      m
    in
    let ( &&& ) = ( land ) in
    let ( lor ) = Int32.logor in
    let ( lxor ) = Int32.logxor in
    let ( land ) = Int32.logand in
    let ( ++ ) = Int32.add in
    let lnot = Int32.lognot in
    let sr = Int32.shift_right in
    let sl = Int32.shift_left in
    let cls n x = sl x n lor Int32.shift_right_logical x (32 - n) in
    let m = sha_1_pad s in
    let w = Array.make 16 0l in
    let h0 = ref 0x67452301l
    and h1 = ref 0xEFCDAB89l
    and h2 = ref 0x98BADCFEl in
    let h3 = ref 0x10325476l and h4 = ref 0xC3D2E1F0l in
    let a = ref 0l
    and b = ref 0l
    and c = ref 0l
    and d = ref 0l
    and e = ref 0l in
    for i = 0 to (Bytes.length m / 64) - 1 do
      let base = i * 64 in
      for j = 0 to 15 do
        let k = base + (j * 4) in
        w.(j) <-
          sl (Int32.of_int (Char.code (Bytes.get m k))) 24
          lor sl (Int32.of_int (Char.code (Bytes.get m (k + 1)))) 16
          lor sl (Int32.of_int (Char.code (Bytes.get m (k + 2)))) 8
          lor Int32.of_int (Char.code (Bytes.get m (k + 3)))
      done;
      a := !h0;
      b := !h1;
      c := !h2;
      d := !h3;
      e := !h4;
      for t = 0 to 79 do
        let f, k =
          if t <= 19 then (!b land !c lor (lnot !b land !d), 0x5A827999l)
          else if t <= 39 then (!b lxor !c lxor !d, 0x6ED9EBA1l)
          else if t <= 59 then
            (!b land !c lor (!b land !d) lor (!c land !d), 0x8F1BBCDCl)
          else (!b lxor !c lxor !d, 0xCA62C1D6l)
        in
        let s = t &&& 0xF in
        if t >= 16 then
          w.(s) <-
            cls 1
              (w.(s + 13 &&& 0xF)
              lxor w.(s + 8 &&& 0xF)
              lxor w.(s + 2 &&& 0xF)
              lxor w.(s));
        let temp = cls 5 !a ++ f ++ !e ++ w.(s) ++ k in
        e := !d;
        d := !c;
        c := cls 30 !b;
        b := !a;
        a := temp
      done;
      h0 := !h0 ++ !a;
      h1 := !h1 ++ !b;
      h2 := !h2 ++ !c;
      h3 := !h3 ++ !d;
      h4 := !h4 ++ !e
    done;
    let h = Bytes.create 20 in
    let i2s h k i =
      Bytes.set h k (Char.unsafe_chr (Int32.to_int (sr i 24) &&& 0xFF));
      Bytes.set h (k + 1) (Char.unsafe_chr (Int32.to_int (sr i 16) &&& 0xFF));
      Bytes.set h (k + 2) (Char.unsafe_chr (Int32.to_int (sr i 8) &&& 0xFF));
      Bytes.set h (k + 3) (Char.unsafe_chr (Int32.to_int i &&& 0xFF))
    in
    i2s h 0 !h0;
    i2s h 4 !h1;
    i2s h 8 !h2;
    i2s h 12 !h3;
    i2s h 16 !h4;
    Bytes.unsafe_to_string h
end

(*---------------------------------------------------------------------------
  Buffered reader
  ---------------------------------------------------------------------------*)

type reader = {
  fd : Unix.file_descr;
  buf : bytes;
  mutable pos : int;
  mutable len : int;
}

let reader_create fd = { fd; buf = Bytes.create 4096; pos = 0; len = 0 }

let reader_fill r =
  if r.len = 0 then r.pos <- 0
  else if r.pos > 0 then begin
    Bytes.blit r.buf r.pos r.buf 0 r.len;
    r.pos <- 0
  end;
  let space = Bytes.length r.buf - r.len in
  if space > 0 then begin
    let n = Unix.read r.fd r.buf r.len space in
    if n = 0 then raise End_of_file;
    r.len <- r.len + n
  end

let reader_read_line r =
  let buf = Buffer.create 128 in
  let rec loop () =
    if r.len = 0 then reader_fill r;
    let limit = r.pos + r.len in
    let rec find_nl i =
      if i = limit then None
      else if Bytes.unsafe_get r.buf i = '\n' then Some i
      else find_nl (i + 1)
    in
    match find_nl r.pos with
    | Some i ->
        let line_len = i - r.pos in
        let end_len =
          if line_len > 0 && Bytes.unsafe_get r.buf (i - 1) = '\r' then
            line_len - 1
          else line_len
        in
        let s =
          if Buffer.length buf = 0 then Bytes.sub_string r.buf r.pos end_len
          else begin
            Buffer.add_subbytes buf r.buf r.pos end_len;
            Buffer.contents buf
          end
        in
        let consumed = line_len + 1 in
        r.pos <- r.pos + consumed;
        r.len <- r.len - consumed;
        s
    | None ->
        Buffer.add_subbytes buf r.buf r.pos r.len;
        r.pos <- 0;
        r.len <- 0;
        loop ()
  in
  loop ()

let reader_read_exact_bytes r n =
  if n <= r.len then begin
    let b = Bytes.sub r.buf r.pos n in
    r.pos <- r.pos + n;
    r.len <- r.len - n;
    b
  end
  else begin
    let res = Bytes.create n in
    let rec loop rem off =
      if rem = 0 then res
      else begin
        if r.len = 0 then reader_fill r;
        let take = min rem r.len in
        Bytes.blit r.buf r.pos res off take;
        r.pos <- r.pos + take;
        r.len <- r.len - take;
        loop (rem - take) (off + take)
      end
    in
    loop n 0
  end

let reader_read_exact r n = Bytes.unsafe_to_string (reader_read_exact_bytes r n)

(*---------------------------------------------------------------------------
  HTTP writing
  ---------------------------------------------------------------------------*)

let write_all fd s off len =
  let rec loop off len =
    if len > 0 then begin
      let n = Unix.write_substring fd s off len in
      loop (off + n) (len - n)
    end
  in
  loop off len

let write_string fd s = write_all fd s 0 (String.length s)

(*---------------------------------------------------------------------------
  Types
  ---------------------------------------------------------------------------*)

type meth = GET | HEAD | POST | PUT | DELETE

type request = {
  meth : meth;
  path : string;
  query : (string * string) list;
  headers : (string * string) list;
  body : string;
  client_addr : Unix.sockaddr;
}

type response = {
  status : int;
  headers : (string * string) list;
  body : string;
}

let response ?(status = 200) ?(headers = []) body = { status; headers; body }

let json ?(status = 200) body =
  { status; headers = [ ("Content-Type", "application/json") ]; body }

let header name (req : request) =
  let rec find = function
    | [] -> None
    | (k, v) :: rest ->
        if str_equal_case_insensitive k name then Some v else find rest
  in
  find req.headers

(*---------------------------------------------------------------------------
  HTTP Parsing
  ---------------------------------------------------------------------------*)

let meth_of_string = function
  | "GET" -> GET
  | "HEAD" -> HEAD
  | "POST" -> POST
  | "PUT" -> PUT
  | "DELETE" -> DELETE
  | _ -> failwith err_unsupported_meth

let trim_header_value s start =
  let len = String.length s in
  let rec trim_left j =
    if j < len && (s.[j] = ' ' || s.[j] = '\t') then trim_left (j + 1) else j
  in
  let rec trim_right j =
    if j >= start && (s.[j] = ' ' || s.[j] = '\t') then trim_right (j - 1)
    else j
  in
  let l = trim_left start in
  let r = trim_right (len - 1) in
  if l <= r then String.sub s l (r - l + 1) else ""

let parse_request reader client_addr =
  let line = reader_read_line reader in
  if String.length line = 0 then raise End_of_file;
  let i1 =
    match String.index_opt line ' ' with
    | Some i -> i
    | None -> failwith err_malformed_req
  in
  let i2 =
    match String.index_from_opt line (i1 + 1) ' ' with
    | Some i -> i
    | None -> failwith err_malformed_req
  in
  let meth_s = String.sub line 0 i1 in
  let raw_path = String.sub line (i1 + 1) (i2 - i1 - 1) in
  let meth = meth_of_string meth_s in
  let path, query_string =
    match String.index_opt raw_path '?' with
    | Some i ->
        ( url_decode (String.sub raw_path 0 i),
          String.sub raw_path (i + 1) (String.length raw_path - i - 1) )
    | None -> (url_decode raw_path, "")
  in
  let query = parse_query_string query_string in

  let rec loop_headers headers content_length keep_alive =
    let hline = reader_read_line reader in
    if String.length hline = 0 then (headers, content_length, keep_alive)
    else
      match String.index_opt hline ':' with
      | Some i ->
          let key = String.sub hline 0 i in
          let value = trim_header_value hline (i + 1) in

          let content_length =
            if str_equal_case_insensitive key "content-length" then
              try int_of_string value with _ -> content_length
            else content_length
          in
          let keep_alive =
            if
              str_equal_case_insensitive key "connection"
              && str_equal_case_insensitive value "close"
            then false
            else keep_alive
          in
          loop_headers ((key, value) :: headers) content_length keep_alive
      | None -> loop_headers headers content_length keep_alive
  in
  let headers, content_length, keep_alive = loop_headers [] 0 true in
  let body =
    if content_length > 0 then reader_read_exact reader content_length else ""
  in
  ( { meth; path; query; headers = List.rev headers; body; client_addr },
    keep_alive )

let write_response fd resp =
  let buf = Buffer.create 256 in
  Buffer.add_string buf "HTTP/1.1 ";
  Buffer.add_string buf (string_of_int resp.status);
  Buffer.add_char buf ' ';
  Buffer.add_string buf (reason_phrase resp.status);
  Buffer.add_string buf "\r\n";
  let rec add_headers = function
    | [] -> ()
    | (k, v) :: rest ->
        Buffer.add_string buf k;
        Buffer.add_string buf ": ";
        Buffer.add_string buf v;
        Buffer.add_string buf "\r\n";
        add_headers rest
  in
  add_headers resp.headers;
  Buffer.add_string buf "Content-Length: ";
  Buffer.add_string buf (string_of_int (String.length resp.body));
  Buffer.add_string buf "\r\n\r\n";
  write_string fd (Buffer.contents buf);
  if String.length resp.body > 0 then write_string fd resp.body

(*---------------------------------------------------------------------------
  WebSocket
  ---------------------------------------------------------------------------*)

type ws = {
  ws_fd : Unix.file_descr;
  ws_reader : reader;
  ws_mutex : Mutex.t;
  mutable ws_closed : bool;
}

let ws_magic = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

let ws_handshake req fd =
  let key =
    match header "Sec-WebSocket-Key" req with
    | Some k -> k
    | None -> failwith err_missing_ws_key
  in
  let accept = Ws_crypto.base64_encode (Ws_crypto.sha1 (key ^ ws_magic)) in
  let buf = Buffer.create 128 in
  Buffer.add_string buf "HTTP/1.1 101 Switching Protocols\r\n";
  Buffer.add_string buf "Upgrade: websocket\r\n";
  Buffer.add_string buf "Connection: Upgrade\r\n";
  Buffer.add_string buf "Sec-WebSocket-Accept: ";
  Buffer.add_string buf accept;
  Buffer.add_string buf "\r\n\r\n";
  write_string fd (Buffer.contents buf)

let ws_write_frame_unlocked ws opcode payload =
  if not ws.ws_closed then begin
    let len = String.length payload in
    let hlen = if len < 126 then 2 else if len < 65536 then 4 else 10 in
    let h = Bytes.create hlen in
    Bytes.unsafe_set h 0 (Char.unsafe_chr (0x80 lor opcode));
    if len < 126 then Bytes.unsafe_set h 1 (Char.unsafe_chr len)
    else if len < 65536 then begin
      Bytes.unsafe_set h 1 (Char.unsafe_chr 126);
      Bytes.unsafe_set h 2 (Char.unsafe_chr ((len lsr 8) land 0xFF));
      Bytes.unsafe_set h 3 (Char.unsafe_chr (len land 0xFF))
    end
    else begin
      Bytes.unsafe_set h 1 (Char.unsafe_chr 127);
      let len64 = Int64.of_int len in
      for i = 0 to 7 do
        let shift = (7 - i) * 8 in
        let b = Int64.logand (Int64.shift_right_logical len64 shift) 0xFFL in
        Bytes.unsafe_set h (2 + i) (Char.unsafe_chr (Int64.to_int b))
      done
    end;
    write_string ws.ws_fd (Bytes.unsafe_to_string h);
    if len > 0 then write_string ws.ws_fd payload
  end

let ws_write_frame ws opcode payload =
  Mutex.lock ws.ws_mutex;
  Fun.protect
    ~finally:(fun () -> Mutex.unlock ws.ws_mutex)
    (fun () -> ws_write_frame_unlocked ws opcode payload)

let ws_read_frame ws =
  let h = reader_read_exact_bytes ws.ws_reader 2 in
  let b0 = Char.code (Bytes.unsafe_get h 0) in
  let b1 = Char.code (Bytes.unsafe_get h 1) in
  let opcode = b0 land 0x0F in
  let masked = b1 land 0x80 <> 0 in
  let len_code = b1 land 0x7F in
  let payload_len =
    if len_code = 126 then
      let ext = reader_read_exact_bytes ws.ws_reader 2 in
      (Char.code (Bytes.unsafe_get ext 0) lsl 8)
      lor Char.code (Bytes.unsafe_get ext 1)
    else if len_code = 127 then
      let ext = reader_read_exact_bytes ws.ws_reader 8 in
      let rec loop i acc =
        if i = 8 then acc
        else loop (i + 1) ((acc lsl 8) lor Char.code (Bytes.unsafe_get ext i))
      in
      loop 0 0
    else len_code
  in
  let mask_key =
    if masked then Some (reader_read_exact_bytes ws.ws_reader 4) else None
  in
  let payload = reader_read_exact_bytes ws.ws_reader payload_len in
  match mask_key with
  | Some key ->
      for i = 0 to payload_len - 1 do
        let b = Char.code (Bytes.unsafe_get payload i) in
        let m = Char.code (Bytes.unsafe_get key (i land 3)) in
        Bytes.unsafe_set payload i (Char.unsafe_chr (b lxor m))
      done;
      (opcode, Bytes.unsafe_to_string payload)
  | None -> (opcode, Bytes.unsafe_to_string payload)

let ws_send ws msg = ws_write_frame ws 0x1 msg

let ws_recv ws =
  if ws.ws_closed then None
  else
    let rec loop () =
      match ws_read_frame ws with
      | (0x1 | 0x2), payload -> Some payload
      | 0x8, _ ->
          (try ws_write_frame ws 0x8 "" with _ -> ());
          ws.ws_closed <- true;
          None
      | 0x9, _ ->
          (try ws_write_frame ws 0xA "" with _ -> ());
          loop ()
      | 0xA, _ -> loop ()
      | _ -> loop () (* ignore unknown opcodes per RFC 6455 *)
    in
    try loop () with
    | End_of_file ->
        Printf.eprintf err_ws_eof;
        ws.ws_closed <- true;
        None
    | Unix.Unix_error (err, fn, _) ->
        Printf.eprintf err_ws_unix (Unix.error_message err) fn;
        ws.ws_closed <- true;
        None

let ws_close ws =
  Mutex.lock ws.ws_mutex;
  Fun.protect
    ~finally:(fun () -> Mutex.unlock ws.ws_mutex)
    (fun () ->
      if not ws.ws_closed then begin
        (try
           ws_write_frame_unlocked ws 0x8 "";
           Unix.shutdown ws.ws_fd Unix.SHUTDOWN_ALL
         with _ -> ());
        ws.ws_closed <- true
      end)

(*---------------------------------------------------------------------------
  Static file serving
  ---------------------------------------------------------------------------*)

let serve_static ~prefix ~loader req =
  let prefix_len = String.length prefix in
  let path_len = String.length req.path in
  let rel_path =
    let start =
      if path_len > prefix_len && req.path.[prefix_len] = '/' then
        prefix_len + 1
      else prefix_len
    in
    if start < path_len then String.sub req.path start (path_len - start)
    else ""
  in
  match loader rel_path with
  | Some data ->
      response ~headers:[ ("Content-Type", mime_of_path rel_path) ] data
  | None -> response ~status:404 "Not Found"

(*---------------------------------------------------------------------------
  Server evaluation
  ---------------------------------------------------------------------------*)

type route_entry =
  | Exact of meth * string * (request -> response)
  | Static of string * (string -> string option)
  | Websocket of string * (request -> ws -> unit)

type t = {
  addr : string;
  port : int;
  mutable routes : route_entry list;
  mutable running : bool;
  mutable listen_fd : Unix.file_descr option;
}

let create ?(addr = "127.0.0.1") ?(port = 8080) () =
  { addr; port; routes = []; running = false; listen_fd = None }

let route server meth path handler =
  server.routes <- Exact (meth, path, handler) :: server.routes

let static server ~prefix ~loader () =
  server.routes <- Static (prefix, loader) :: server.routes

let websocket server path handler =
  server.routes <- Websocket (path, handler) :: server.routes

let find_route routes req =
  let rec search = function
    | [] -> None
    | Exact (m, p, h) :: _ when m = req.meth && String.equal p req.path ->
        Some (`Handler h)
    | Static (prefix, loader) :: _
      when req.meth = GET && starts_with prefix req.path ->
        Some (`Static (prefix, loader))
    | Websocket (p, h) :: _ when req.meth = GET && String.equal p req.path ->
        Some (`Websocket h)
    | _ :: rest -> search rest
  in
  search routes

(* Check if a comma-separated header value contains [token]
   (case-insensitive) *)
let header_contains_token s token =
  let len = String.length s in
  let rec scan i =
    if i >= len then false
    else
      let rec skip_ws j =
        if j < len && (s.[j] = ' ' || s.[j] = '\t') then skip_ws (j + 1) else j
      in
      let start = skip_ws i in
      let rec find_sep j =
        if j < len && s.[j] <> ',' then find_sep (j + 1) else j
      in
      let stop = find_sep start in
      let rec rtrim j =
        if j > start && (s.[j - 1] = ' ' || s.[j - 1] = '\t') then rtrim (j - 1)
        else j
      in
      let right = rtrim stop in
      if sub_equal_case_insensitive s start (right - start) token then true
      else scan (stop + 1)
  in
  scan 0

let is_websocket_upgrade req =
  match (header "Connection" req, header "Upgrade" req) with
  | Some conn, Some upg ->
      header_contains_token conn "upgrade"
      && str_equal_case_insensitive upg "websocket"
  | _ -> false

let handle_ws_upgrade server req fd reader =
  match find_route server.routes req with
  | Some (`Websocket handler) -> (
      (* Use long timeouts for WebSocket (effectively infinite). Both recv and
         send must be increased — the initial HTTP SO_SNDTIMEO of 30s would
         otherwise kill the connection when the process is paused (e.g. inside a
         debugger). *)
      Unix.setsockopt_float fd Unix.SO_RCVTIMEO 86400.0;
      Unix.setsockopt_float fd Unix.SO_SNDTIMEO 86400.0;
      ws_handshake req fd;
      let ws =
        {
          ws_fd = fd;
          ws_reader = reader;
          ws_mutex = Mutex.create ();
          ws_closed = false;
        }
      in
      try handler req ws
      with exn ->
        Printf.eprintf "[ws] handler error: %s\n%!" (Printexc.to_string exn))
  | _ -> write_response fd (response ~status:404 "Not Found")

let dispatch_http server req =
  match find_route server.routes req with
  | Some (`Handler h) -> (
      try h req
      with exn ->
        Printf.eprintf err_handler (Printexc.to_string exn);
        response ~status:500 "Internal Server Error")
  | Some (`Static (prefix, loader)) -> serve_static ~prefix ~loader req
  | Some (`Websocket _) -> response ~status:426 "Upgrade Required"
  | None -> response ~status:404 "Not Found"

let handle_connection server fd client_addr =
  let reader = reader_create fd in
  Unix.setsockopt_float fd Unix.SO_RCVTIMEO 30.0;
  Unix.setsockopt_float fd Unix.SO_SNDTIMEO 30.0;
  Unix.setsockopt fd Unix.TCP_NODELAY true;
  let rec loop keep_alive =
    if keep_alive && server.running then begin
      match parse_request reader client_addr with
      | req, ka ->
          if is_websocket_upgrade req then
            handle_ws_upgrade server req fd reader
          else begin
            write_response fd (dispatch_http server req);
            loop ka
          end
      | exception End_of_file -> ()
      | exception Unix.Unix_error (Unix.ETIMEDOUT, _, _) -> ()
      | exception Unix.Unix_error (Unix.EAGAIN, _, _) -> ()
      | exception Unix.Unix_error (Unix.ECONNRESET, _, _) -> ()
      | exception exn -> (
          Printf.eprintf err_parse (Printexc.to_string exn);
          try write_response fd (response ~status:400 "Bad Request")
          with _ -> ())
    end
  in
  loop true

let shutdown_silent fd = try Unix.shutdown fd Unix.SHUTDOWN_ALL with _ -> ()
let close_silent fd = try Unix.close fd with _ -> ()

let run ?(after_start = ignore) server =
  if not Sys.win32 then
    ignore (Unix.sigprocmask Unix.SIG_BLOCK [ Sys.sigpipe ] : int list);
  let sock = Unix.socket Unix.PF_INET Unix.SOCK_STREAM 0 in
  Unix.setsockopt sock Unix.SO_REUSEADDR true;
  let inet_addr = Unix.inet_addr_of_string server.addr in
  Unix.bind sock (Unix.ADDR_INET (inet_addr, server.port));
  Unix.listen sock 128;
  Unix.set_nonblock sock;
  server.listen_fd <- Some sock;
  server.running <- true;
  server.routes <- List.rev server.routes;
  after_start ();
  let rec accept_loop () =
    if server.running then begin
      match Unix.accept sock with
      | client_fd, client_addr ->
          Unix.clear_nonblock client_fd;
          ignore
            (Thread.create
               (fun () ->
                 Fun.protect
                   ~finally:(fun () ->
                     shutdown_silent client_fd;
                     close_silent client_fd)
                   (fun () ->
                     try handle_connection server client_fd client_addr
                     with exn ->
                       Printf.eprintf err_connection (Printexc.to_string exn)))
               ()
              : Thread.t);
          accept_loop ()
      | exception Unix.Unix_error ((Unix.EAGAIN | Unix.EWOULDBLOCK), _, _) ->
          ignore (Unix.select [ sock ] [] [] 0.5 : _ * _ * _);
          accept_loop ()
      | exception Unix.Unix_error (Unix.EBADF, _, _) -> server.running <- false
      | exception exn ->
          Printf.eprintf err_accept (Printexc.to_string exn);
          Thread.delay 0.01;
          accept_loop ()
    end
  in
  accept_loop ();
  close_silent sock;
  server.listen_fd <- None

let stop server =
  server.running <- false;
  match server.listen_fd with
  | Some fd ->
      close_silent fd;
      server.listen_fd <- None
  | None -> ()
