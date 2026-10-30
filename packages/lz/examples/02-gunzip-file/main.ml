(* Decoding a gzip file in steps.

   A decoder takes its input in pieces, so a program can decode a file of
   any size with a fixed buffer. This program writes [notes.txt.gz], a gzip
   file of two members, as `cat a.gz b.gz` makes one. It then reads the
   file 4 KiB at a time, steps a decoder over each piece, and counts the
   lines and the CRC-32 of the decoded bytes as each run arrives. The run
   is lent only until [f] returns, so [f] reads it there and keeps
   nothing. *)

open Bigarray

let file = "notes.txt.gz"

let notes first n =
  let note i = Printf.sprintf "note %d: nothing to report\n" (first + i) in
  String.concat "" (List.init n note)

(* [decode_channel d ic f] steps [d] over the bytes of [ic] until its
   stream ends, applying [f] to each run. [buf]'s [len] bytes from [start]
   are read and not yet taken. *)
let decode_channel d ic f =
  let buf = Array1.create int8_unsigned c_layout 4096 in
  let rec go start len last =
    match Lz.Decoder.step d ~last (Array1.sub buf start len) f with
    | Error e -> Error e
    | Ok (Done k) -> Ok k
    | Ok (More k) when k < len || last ->
        (* The run filled before the decoder took every byte, or the input
           has ended and output is still due. Step again on the rest. *)
        go (start + k) (len - k) last
    | Ok (More _) ->
        let n = In_channel.input_bigarray ic buf 0 (Array1.dim buf) in
        go 0 n (n = 0)
  in
  go 0 0 false

let () =
  (* Two members, each a gzip stream of its own. *)
  Out_channel.with_open_bin file (fun oc ->
      Out_channel.output_string oc (Lz.Gzip.encode_string (notes 0 3000));
      Out_channel.output_string oc (Lz.Gzip.encode_string (notes 3000 1000)));
  let length = In_channel.with_open_bin file In_channel.length in
  Printf.printf "%s: %Ld bytes\n" file length;

  let lines = ref 0 and size = ref 0 and crc = ref 0 in
  let count run =
    for i = 0 to Array1.dim run - 1 do
      if run.{i} = Char.code '\n' then incr lines
    done;
    size := !size + Array1.dim run;
    crc := Lz.Crc32.bigbytes ~crc:!crc run
  in
  let d = Lz.Gzip.decoder () in
  let decode ic = decode_channel d ic count in
  (match In_channel.with_open_bin file decode with
  | Ok _ -> print_endline "the stream ended"
  | Error e -> Format.printf "%a@." Lz.pp_error e);
  Printf.printf "%d lines, %d bytes, CRC-32 %08x\n" !lines !size !crc;

  (* The CRC-32 of the text the members hold. *)
  let text = notes 0 3000 ^ notes 3000 1000 in
  Printf.printf "expected: %d bytes, CRC-32 %08x\n" (String.length text)
    (Lz.Crc32.string text);
  Sys.remove file
