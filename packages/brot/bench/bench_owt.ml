(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Throughput over a large corpus: [encode_batch_ids] on the documents of a file
   split on <|endoftext|>. Not on any alias; run it by hand:

   bench_owt.exe TOKENIZER.json CORPUS.txt [-domains N] [-repeat N] [-bytes N]

   [-bytes N] reads the documents that end within the corpus's first N bytes.
   With [-domains 1] the calling domain encodes, and a run reuses the pretoken
   caches the runs before it filled; with more, each run also spawns domains
   of its own, whose caches start cold.
   [-repeat 0] stops before encoding: the baseline that, subtracted from a
   one-run count of instructions, leaves the run's.

   Before timing, one pre-tokenization pass counts the pretokens — the
   denominator of ns/pretoken — and the ones over 15 bytes, which are the spans
   the C kernel hands back to OCaml on this path. *)

let read_file ~limit path =
  let ic = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () ->
      let len = in_channel_length ic in
      really_input_string ic (if limit > 0 then min limit len else len))

let marker = "<|endoftext|>"

(* The prefix before the last marker: a cut never splits a document. *)
let whole_documents ~limit s =
  if limit <= 0 then s
  else
    let rec last i =
      if i < 0 then String.length s
      else if String.sub s i (String.length marker) = marker then i
      else last (i - 1)
    in
    String.sub s 0 (last (String.length s - String.length marker))

let split_on_marker marker s =
  let n = String.length s and m = String.length marker in
  let rec find i =
    if i + m > n then None
    else
      match String.index_from_opt s i marker.[0] with
      | None -> None
      | Some j ->
          if j + m > n then None
          else if String.sub s j m = marker then Some j
          else find (j + 1)
  in
  let rec go acc start =
    match find start with
    | None -> List.rev (String.sub s start (n - start) :: acc)
    | Some j -> go (String.sub s start (j - start) :: acc) (j + m)
  in
  go [] 0

let () =
  let domains = ref 0 in
  let repeat = ref 3 in
  let limit = ref 0 in
  let anon = ref [] in
  Arg.parse
    [
      ("-domains", Arg.Set_int domains, "N domains to encode on (default: auto)");
      ("-repeat", Arg.Set_int repeat, "N timed runs (default: 3)");
      ("-bytes", Arg.Set_int limit, "N read the corpus's first N bytes");
    ]
    (fun a -> anon := a :: !anon)
    "bench_owt TOKENIZER.json CORPUS.txt [-domains N] [-repeat N] [-bytes N]";
  let tokenizer_path, corpus_path =
    match List.rev !anon with
    | [ t; c ] -> (t, c)
    | _ ->
        prerr_endline "usage: bench_owt TOKENIZER.json CORPUS.txt [options]";
        exit 2
  in
  let tok =
    match Brot.from_file tokenizer_path with
    | Ok tok -> tok
    | Error msg ->
        Printf.eprintf "cannot load %s: %s\n" tokenizer_path msg;
        exit 2
  in
  let corpus =
    read_file ~limit:!limit corpus_path |> whole_documents ~limit:!limit
  in
  let docs =
    split_on_marker marker corpus |> List.filter (fun d -> d <> "")
  in
  let bytes = List.fold_left (fun acc d -> acc + String.length d) 0 docs in
  let pretokens, long_pretokens =
    match Brot.pre_tokenizer tok with
    | None -> (0, 0)
    | Some pre ->
        List.fold_left
          (fun (n, long) d ->
            List.fold_left
              (fun (n, long) (_, (a, b)) ->
                (n + 1, if b - a > 15 then long + 1 else long))
              (n, long)
              (Brot.Pre_tokenizer.pre_tokenize pre d))
          (0, 0) docs
  in
  Printf.printf "corpus: %d docs, %d bytes" (List.length docs) bytes;
  if pretokens > 0 then
    Printf.printf ", %d pretokens (%.2f bytes each, %d over 15 bytes = %.3f%%)"
      pretokens
      (float_of_int bytes /. float_of_int pretokens)
      long_pretokens
      (100. *. float_of_int long_pretokens /. float_of_int pretokens);
  print_newline ();
  let domains = if !domains > 0 then Some !domains else None in
  (* Warm what a first call sets up, without seeding the pretoken caches with
     the corpus. *)
  ignore (Brot.encode_batch_ids tok ~add_special_tokens:false ?domains [ "a" ]);
  let best = ref infinity in
  for run = 1 to !repeat do
    let t0 = Unix.gettimeofday () in
    let ids, _ =
      Brot.encode_batch_ids tok ~add_special_tokens:false ?domains docs
    in
    let dt = Unix.gettimeofday () -. t0 in
    if dt < !best then best := dt;
    (* The sum of the ids tells two tokenizers' outputs apart. *)
    let sum = ref 0 in
    for i = 0 to Bigarray.Array1.dim ids - 1 do
      sum := !sum + Int32.to_int (Bigarray.Array1.unsafe_get ids i)
    done;
    Printf.printf "run %d: %.2f s, %.1f MB/s, %d ids, sum %d%s\n%!" run dt
      (float_of_int bytes /. 1e6 /. dt)
      (Bigarray.Array1.dim ids) !sum
      (if pretokens > 0 then
         Printf.sprintf ", %.1f ns/pretoken"
           (dt *. 1e9 /. float_of_int pretokens)
       else "")
  done;
  if !repeat > 0 then
    Printf.printf "best: %.1f MB/s%s\n"
      (float_of_int bytes /. 1e6 /. !best)
      (if pretokens > 0 then
         Printf.sprintf ", %.1f ns/pretoken"
           (!best *. 1e9 /. float_of_int pretokens)
       else "")
