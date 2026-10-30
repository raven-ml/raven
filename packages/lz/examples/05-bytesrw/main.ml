(* Compressing and decompressing through bytesrw.

   lz.bytesrw wraps lz's decoders and encoders as bytesrw filters, so a
   program that reads and writes through bytesrw readers and writers
   compresses with one more function. This program writes lines through a
   gzip writer filter into a buffer, reads them back through a gzip reader
   filter, and catches the error a damaged stream raises. *)

open Bytesrw

let text =
  String.concat "" (List.init 1000 (fun i -> Printf.sprintf "event %d: ok\n" i))

let () =
  (* A writer filter. Each write on [w] reaches [buf] compressed. The
     member ends at the [eod] written on [w], and [~eod:true] writes [eod]
     on the buffer's writer after it. *)
  let buf = Buffer.create 4096 in
  let gzip = Lz_bytesrw.Gzip.compress_writes () in
  let w = gzip ~eod:true (Bytes.Writer.of_buffer buf) in
  for i = 0 to 999 do
    Bytes.Writer.write_string w (Printf.sprintf "event %d: ok\n" i)
  done;
  Bytes.Writer.write_eod w;
  let gz = Buffer.contents buf in
  Printf.printf "%d bytes written, %d bytes of gzip\n" (String.length text)
    (String.length gz);

  (* A reader filter. Reads from [r] return the decoded bytes. *)
  let r = Lz_bytesrw.Gzip.decompress_reads () (Bytes.Reader.of_string gz) in
  let back = Bytes.Reader.to_string r in
  Printf.printf "%d bytes read back, same: %b\n" (String.length back)
    (back = text);

  (* The same member cut short. The read raises Bytes.Stream.Error with
     lz's error, at its position in the compressed stream. *)
  let cut = String.sub gz 0 (String.length gz / 2) in
  let r = Lz_bytesrw.Gzip.decompress_reads () (Bytes.Reader.of_string cut) in
  match Bytes.Reader.to_string r with
  | _ -> print_endline "no error"
  | exception Bytes.Stream.Error ((Lz_bytesrw.Error e, _) as err) ->
      Printf.printf "error at byte %d: %s\n" e.at
        (Bytes.Stream.error_message err)
