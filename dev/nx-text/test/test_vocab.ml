(* Vocabulary tests for nx-text *)

open Alcotest
open Nx_text

(* Basic Vocabulary Tests *)

let test_vocab_create_empty () =
  let v = Vocab.create () in
  check int "empty vocab size" 4 (Vocab.size v);
  (* 4 special tokens *)
  check int "pad index" 0 (Vocab.pad_idx v);
  check int "unk index" 1 (Vocab.unk_idx v);
  check int "bos index" 2 (Vocab.bos_idx v);
  check int "eos index" 3 (Vocab.eos_idx v)

let test_vocab_add_single () =
  let v = Vocab.create () in
  Vocab.add v "hello";
  check int "size after add" 5 (Vocab.size v);
  check (option int) "get index" (Some 4) (Vocab.get_index v "hello");
  check (option string) "get token" (Some "hello") (Vocab.get_token v 4)

let test_vocab_add_duplicate () =
  let v = Vocab.create () in
  Vocab.add v "hello";
  Vocab.add v "hello";
  (* Should not increase size *)
  check int "size after duplicate" 5 (Vocab.size v)

let test_vocab_add_batch () =
  let v = Vocab.create () in
  Vocab.add_batch v [ "hello"; "world"; "test" ];
  check int "size after batch" 7 (Vocab.size v);
  check (option int) "hello index" (Some 4) (Vocab.get_index v "hello");
  check (option int) "world index" (Some 5) (Vocab.get_index v "world");
  check (option int) "test index" (Some 6) (Vocab.get_index v "test")

let test_vocab_unknown_token () =
  let v = Vocab.create () in
  check (option int) "unknown token" None (Vocab.get_index v "unknown");
  check (option string) "invalid index" None (Vocab.get_token v 100)

(* Building Vocabulary from Tokens *)

let test_vocab_from_tokens_simple () =
  let tokens = [ "hello"; "world"; "hello"; "there" ] in
  let v = vocab tokens in
  check int "vocab size" 7 (Vocab.size v)
(* 4 special + 3 unique tokens *)

let test_vocab_from_tokens_min_freq () =
  let tokens = [ "a"; "b"; "b"; "c"; "c"; "c" ] in
  let v = vocab ~min_freq:2 tokens in
  (* Should include special tokens + b(2) + c(3), not a(1) *)
  check int "vocab with min_freq" 6 (Vocab.size v);
  check (option int) "frequent token b" (Some 5) (Vocab.get_index v "b");
  check (option int) "frequent token c" (Some 4) (Vocab.get_index v "c");
  check (option int) "infrequent token a" None (Vocab.get_index v "a")

let test_vocab_from_tokens_max_size () =
  let tokens = List.init 100 (fun i -> Printf.sprintf "token%d" i) in
  let v = vocab ~max_size:10 tokens in
  check int "vocab with max_size" 10 (Vocab.size v)

let test_vocab_from_tokens_empty () =
  let v = vocab [] in
  check int "vocab from empty tokens" 4 (Vocab.size v)

(* Special Tokens Tests *)

let test_vocab_special_tokens_preserved () =
  let v = Vocab.create () in
  (* Try to add special tokens - should not change their indices *)
  Vocab.add v "<pad>";
  Vocab.add v "<unk>";
  check int "pad index unchanged" 0 (Vocab.pad_idx v);
  check int "unk index unchanged" 1 (Vocab.unk_idx v)

let test_vocab_get_special_tokens () =
  let v = Vocab.create () in
  check (option string) "get pad token" (Some "<pad>") (Vocab.get_token v 0);
  check (option string) "get unk token" (Some "<unk>") (Vocab.get_token v 1);
  check (option string) "get bos token" (Some "<bos>") (Vocab.get_token v 2);
  check (option string) "get eos token" (Some "<eos>") (Vocab.get_token v 3)

(* File I/O Tests *)

let test_vocab_save_load () =
  let v1 = Vocab.create () in
  Vocab.add_batch v1 [ "hello"; "world"; "test" ];

  let temp_file = Filename.temp_file "vocab_test" ".txt" in
  vocab_save v1 temp_file;

  let v2 = vocab_load temp_file in
  check int "loaded vocab size" (Vocab.size v1) (Vocab.size v2);

  (* Check all tokens are preserved *)
  check (option int) "hello preserved" (Some 4) (Vocab.get_index v2 "hello");
  check (option int) "world preserved" (Some 5) (Vocab.get_index v2 "world");
  check (option int) "test preserved" (Some 6) (Vocab.get_index v2 "test");

  Sys.remove temp_file

let test_vocab_load_nonexistent () =
  check_raises "load nonexistent file"
    (Sys_error "/nonexistent/file.txt: No such file or directory") (fun () ->
      ignore (vocab_load "/nonexistent/file.txt"))

(* Edge Cases *)

let test_vocab_large () =
  let tokens = List.init 10000 (fun i -> Printf.sprintf "token_%d" i) in
  let v = vocab tokens in
  check bool "large vocab created" true (Vocab.size v > 1000);

  (* Check some random tokens *)
  check (option int) "token_500"
    (Some (Vocab.get_index v "token_500" |> Option.get))
    (Vocab.get_index v "token_500");
  check (option string) "reverse lookup" (Some "token_500")
    (Vocab.get_token v (Vocab.get_index v "token_500" |> Option.get))

(* Test Suite *)

let vocab_tests =
  [
    (* Basic operations *)
    test_case "create empty" `Quick test_vocab_create_empty;
    test_case "add single" `Quick test_vocab_add_single;
    test_case "add duplicate" `Quick test_vocab_add_duplicate;
    test_case "add batch" `Quick test_vocab_add_batch;
    test_case "unknown token" `Quick test_vocab_unknown_token;
    (* Building from tokens *)
    test_case "from tokens simple" `Quick test_vocab_from_tokens_simple;
    test_case "from tokens min_freq" `Quick test_vocab_from_tokens_min_freq;
    test_case "from tokens max_size" `Quick test_vocab_from_tokens_max_size;
    test_case "from tokens empty" `Quick test_vocab_from_tokens_empty;
    (* Special tokens *)
    test_case "special tokens preserved" `Quick
      test_vocab_special_tokens_preserved;
    test_case "get special tokens" `Quick test_vocab_get_special_tokens;
    (* File I/O *)
    test_case "save and load" `Quick test_vocab_save_load;
    test_case "load nonexistent" `Quick test_vocab_load_nonexistent;
    (* Edge cases *)
    test_case "large vocab" `Slow test_vocab_large;
  ]

let () = Alcotest.run "nx-text vocabulary" [ ("vocabulary", vocab_tests) ]
