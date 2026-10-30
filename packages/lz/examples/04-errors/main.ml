(* Errors, and input the program does not trust.

   A decode that fails returns [Error { at; why }], where [why] says what is
   wrong and [at] is the offset of the input byte where it begins. This
   program damages a gzip member in the ways a program meets, a cut file, a
   flipped bit, bytes after the end and a destination too small, and
   prints each error. It then bounds the window a Zstandard decoder
   allocates, and shows the [Invalid_argument] that a dead or busy decoder
   raises. *)

open Bigarray

let of_string s =
  Array1.init int8_unsigned c_layout (String.length s) (fun i ->
      Char.code s.[i])

let text =
  let line i = Printf.sprintf "%d squared is %d\n" i (i * i) in
  String.concat "" (List.init 500 line)

let show what = function
  | Ok { Lz.consumed; produced } ->
      Printf.printf "%-18s ok, consumed %d, produced %d\n" what consumed
        produced
  | Error e -> Format.printf "%-18s %a@." what Lz.pp_error e

let gunzip what s ~size =
  let dst = Array1.create int8_unsigned c_layout size in
  show what (Lz.Gzip.decode (of_string s) dst)

let invalid f =
  match f () with
  | _ -> ()
  | exception Invalid_argument m -> Printf.printf "Invalid_argument: %s\n" m

let () =
  let gz = Lz.Gzip.encode_string text in
  let n = String.length gz and size = String.length text in
  Printf.printf "a gzip member of %d bytes, holding %d\n\n" n size;
  gunzip "whole" gz ~size;

  (* The input ends inside the member, or holds none: Truncated, at the
     input's length. A program that can read more bytes does. *)
  gunzip "cut at 100" (String.sub gz 0 100) ~size;
  gunzip "empty" "" ~size;

  (* A flipped bit in the member's CRC-32, which begins 8 bytes before its
     end: Malformed, at the field's first byte. *)
  let b = Bytes.of_string gz in
  Bytes.set_uint8 b (n - 8) (Bytes.get_uint8 b (n - 8) lxor 1);
  gunzip "CRC-32 flipped" (Bytes.to_string b) ~size;

  (* Gzip reads to the input's end, and a byte that begins no member is
     Malformed at its offset. *)
  gunzip "zeros after" (gz ^ "\000\000") ~size;

  (* A destination one byte short: Too_long. *)
  gunzip "destination short" gz ~size:(size - 1);

  (* A Zstandard decoder allocates the window a frame declares, up to its
     [window]. A frame written by steps declares a window larger than the
     1 KiB this decoder accepts. *)
  print_newline ();
  let zst = Buffer.create 4096 in
  let keep run =
    Buffer.add_string zst
      (String.init (Array1.dim run) (fun i -> Char.chr run.{i}))
  in
  let e = Lz.Zstd.encoder () in
  ignore (Lz.Encoder.step e ~last:true (of_string text) keep : Lz.step);
  let d = Lz.Zstd.decoder ~window:1024 () in
  let src = of_string (Buffer.contents zst) in
  (match Lz.Decoder.step d ~last:true src ignore with
  | Ok _ -> print_endline "1 KiB window: ok"
  | Error e -> Format.printf "1 KiB window: %a@." Lz.pp_error e);

  (* The decoder that returned the error is dead. *)
  invalid (fun () -> Lz.Decoder.step d ~last:true (of_string "") ignore);

  (* A decoder takes one step at a time, so a step from [f] finds it busy.
     The exception propagates out of the outer step, and leaves the decoder
     dead too. *)
  let d = Lz.Gzip.decoder () in
  let again _ = ignore (Lz.Decoder.step d ~last:true (of_string "") ignore) in
  invalid (fun () -> Lz.Decoder.step d ~last:true (of_string gz) again)
