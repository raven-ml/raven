(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Throughput over a large corpus: [encode_batch_ids] on the documents of a file
   split on <|endoftext|>. Not on any alias; run it by hand:

   bench_owt.exe TOKENIZER.json CORPUS.txt [-domains N] [-repeat N]

   Before timing, one pre-tokenization pass counts the pretokens — the
   denominator of ns/pretoken — and the ones over 15 bytes, which are the spans
   the C kernel hands back to OCaml on this path. *)

let read_file path =
  let ic = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () -> really_input_string ic (in_channel_length ic))

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
  let anon = ref [] in
  Arg.parse
    [
      ("-domains", Arg.Set_int domains, "N domains to encode on (default: auto)");
      ("-repeat", Arg.Set_int repeat, "N timed runs (default: 3)");
    ]
    (fun a -> anon := a :: !anon)
    "bench_owt TOKENIZER.json CORPUS.txt [-domains N] [-repeat N]";
  let tokenizer_path, corpus_path =
    match List.rev !anon with
    | [ t; c ] -> (t, c)
    | _ ->
        prerr_endline "usage: bench_owt TOKENIZER.json CORPUS.txt [-domains N]";
        exit 2
  in
  let tok =
    match Brot.from_file tokenizer_path with
    | Ok tok -> tok
    | Error msg ->
        Printf.eprintf "cannot load %s: %s\n" tokenizer_path msg;
        exit 2
  in
  let corpus = read_file corpus_path in
  let docs =
    split_on_marker "<|endoftext|>" corpus |> List.filter (fun d -> d <> "")
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
  let best = ref infinity in
  for run = 1 to max 1 !repeat do
    let t0 = Unix.gettimeofday () in
    let ids, _ =
      Brot.encode_batch_ids tok ~add_special_tokens:false ?domains docs
    in
    let dt = Unix.gettimeofday () -. t0 in
    if dt < !best then best := dt;
    Printf.printf "run %d: %.2f s, %.1f MB/s, %d ids%s\n%!" run dt
      (float_of_int bytes /. 1e6 /. dt)
      (Bigarray.Array1.dim ids)
      (if pretokens > 0 then
         Printf.sprintf ", %.1f ns/pretoken"
           (dt *. 1e9 /. float_of_int pretokens)
       else "")
  done;
  Printf.printf "best: %.1f MB/s%s\n"
    (float_of_int bytes /. 1e6 /. !best)
    (if pretokens > 0 then
       Printf.sprintf ", %.1f ns/pretoken"
         (!best *. 1e9 /. float_of_int pretokens)
     else "")
