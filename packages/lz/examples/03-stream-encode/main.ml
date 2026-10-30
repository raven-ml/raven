(* Encoding in steps, to a file.

   An encoder takes its input in pieces, so a program can compress data it
   produces as it goes without holding all of it. This program writes a log
   of 20,000 lines to [log.zst] through a Zstandard encoder, a batch of
   1,000 lines per step, and writes each run to the file inside [f], since
   the run is lent only until [f] returns. It then decodes the file whole
   and checks it, and shows that the stream's bytes do not depend on how
   the input was cut into steps. *)

open Bigarray

let file = "log.zst"
let lines = 20_000
let batch = 1_000

let line i =
  let status = if i mod 50 = 0 then 500 else 200 in
  Printf.sprintf "%06d request /item/%d status %d\n" i (i mod 97) status

let of_string s =
  Array1.init int8_unsigned c_layout (String.length s) (fun i ->
      Char.code s.[i])

let to_string b n = String.init n (fun i -> Char.chr b.{i})

(* [step_all e ~last src f] steps [e] until it has taken all of [src]. A
   step takes at most its format's quota, so a long [src] takes several. *)
let rec step_all e ~last src f =
  match Lz.Encoder.step e ~last src f with
  | Lz.Done _ -> ()
  | More k when k = Array1.dim src -> ()
  | More k -> step_all e ~last (Array1.sub src k (Array1.dim src - k)) f

(* [encode_in pieces] is the Zstandard stream of the concatenation of
   [pieces], one step for each piece. *)
let encode_in pieces =
  let e = Lz.Zstd.encoder () in
  let out = Buffer.create 4096 in
  let keep run = Buffer.add_string out (to_string run (Array1.dim run)) in
  let rec go = function
    | [] -> ()
    | [ p ] -> step_all e ~last:true (of_string p) keep
    | p :: rest ->
        step_all e ~last:false (of_string p) keep;
        go rest
  in
  go pieces;
  Buffer.contents out

(* [cut n s] is [s] in pieces of [n] bytes, the last one shorter. *)
let cut n s =
  let len = String.length s in
  List.init ((len + n - 1) / n) (fun i ->
      String.sub s (i * n) (min n (len - (i * n))))

let () =
  let all = String.concat "" (List.init lines line) in

  (* Write the log a batch at a time. Only the last step says [~last:true],
     so the stream ends after its bytes. *)
  Out_channel.with_open_bin file (fun oc ->
      let e = Lz.Zstd.encoder () in
      let write run = Out_channel.output_bigarray oc run 0 (Array1.dim run) in
      let batches = lines / batch in
      for b = 0 to batches - 1 do
        let text =
          String.concat "" (List.init batch (fun i -> line ((b * batch) + i)))
        in
        step_all e ~last:(b = batches - 1) (of_string text) write
      done);
  let stream = In_channel.with_open_bin file In_channel.input_all in
  Printf.printf "%d lines, %d bytes, into %s\n" lines (String.length all)
    file;

  (* The program knows the log's size, so it can size the destination and
     decode the file whole. *)
  let dst = Array1.create int8_unsigned c_layout (String.length all) in
  (match Lz.Zstd.decode (of_string stream) dst with
  | Ok { produced; _ } ->
      Printf.printf "decoded %d bytes, same as written: %b\n" produced
        (to_string dst produced = all)
  | Error e -> Format.printf "%a@." Lz.pp_error e);

  (* The same input cut differently gives the same stream. *)
  let same pieces = encode_in pieces = stream in
  Printf.printf "in 64 KiB steps, same bytes: %b\n" (same (cut 65536 all));
  Printf.printf "in 7-byte steps, same bytes: %b\n" (same (cut 7 all));
  Printf.printf "in one step, same bytes: %b\n" (same [ all ]);
  Sys.remove file
