(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun.Dataset

(* Test helpers *)
let assert_equal_int_array expected actual =
  Alcotest.(check (array int)) "arrays equal" expected actual

let assert_equal_string_array expected actual =
  Alcotest.(check (array string)) "arrays equal" expected actual

let assert_dataset_length expected dataset =
  match cardinality dataset with
  | Finite n -> Alcotest.(check int) "dataset length" expected n
  | _ -> Alcotest.fail "Expected finite dataset"

let collect_dataset dataset = to_list dataset
let collect_n n dataset = dataset |> take n |> to_list

(* Create temporary test file *)
let with_temp_file content f =
  let temp_file = Filename.temp_file "test_dataset" ".txt" in
  let oc = open_out temp_file in
  output_string oc content;
  close_out oc;
  Fun.protect ~finally:(fun () -> Sys.remove temp_file) (fun () -> f temp_file)

let with_temp_jsonl content f =
  let temp_file = Filename.temp_file "test_dataset" ".jsonl" in
  let oc = open_out temp_file in
  output_string oc content;
  close_out oc;
  Fun.protect ~finally:(fun () -> Sys.remove temp_file) (fun () -> f temp_file)

let with_temp_csv content f =
  let temp_file = Filename.temp_file "test_dataset" ".csv" in
  let oc = open_out temp_file in
  output_string oc content;
  close_out oc;
  Fun.protect ~finally:(fun () -> Sys.remove temp_file) (fun () -> f temp_file)

(** Test dataset creation *)
let test_from_array () =
  let arr = [| 1; 2; 3; 4; 5 |] in
  let dataset = from_array arr in
  assert_dataset_length 5 dataset;
  let collected = collect_dataset dataset in
  Alcotest.(check (list int)) "collected values" [ 1; 2; 3; 4; 5 ] collected

let test_from_list () =
  let lst = [ "a"; "b"; "c" ] in
  let dataset = from_list lst in
  assert_dataset_length 3 dataset;
  let collected = collect_dataset dataset in
  Alcotest.(check (list string)) "collected values" [ "a"; "b"; "c" ] collected

let test_from_seq () =
  let seq = List.to_seq [ 10; 20; 30 ] in
  let dataset = from_seq seq in
  let collected = collect_dataset dataset in
  Alcotest.(check (list int)) "collected values" [ 10; 20; 30 ] collected

(** Test text file reading *)
let test_from_text_file () =
  let content = "line1\nline2\nline3\n" in
  with_temp_file content (fun path ->
      let dataset = from_text_file path in
      let collected = collect_dataset dataset in
      Alcotest.(check (list string))
        "lines read"
        [ "line1"; "line2"; "line3" ]
        collected)

(* Test for utf8 *)
let test_from_text_file_utf8 () =
  let content = "hello \xF0\x9F\x98\x8A\nsecond\n" in
  with_temp_file content (fun path ->
      let ds = from_text_file ~encoding:`UTF8 path in
      let lines = collect_dataset ds in
      Alcotest.(check (list string))
        "utf8 emoji preserved" [ "hello 😊"; "second" ] lines)

(* Test for Latin1 *)
let test_from_text_file_latin1 () =
  let content = "caf\xE9\nna\xEFve\n" in
  with_temp_file content (fun path ->
      let ds = from_text_file ~encoding:`LATIN1 path in
      let lines = collect_dataset ds in
      Alcotest.(check (list string)) "latin1 decoded" [ "café"; "naïve" ] lines)

let test_from_text_file_large_lines () =
  let line = String.make 1000 'x' in
  let content = line ^ "\n" ^ line ^ "\n" in
  with_temp_file content (fun path ->
      let dataset = from_text_file ~chunk_size:100 path in
      let collected = collect_dataset dataset in
      Alcotest.(check int) "number of lines" 2 (List.length collected);
      List.iter
        (fun l -> Alcotest.(check int) "line length" 1000 (String.length l))
        collected)

let test_from_text_file_reset () =
  let content = "line1\nline2\n" in
  with_temp_file content (fun path ->
      let dataset = from_text_file path in
      let expected = [ "line1"; "line2" ] in
      let first_pass = collect_dataset dataset in
      Alcotest.(check (list string)) "first pass" expected first_pass;
      reset dataset;
      let second_pass = collect_dataset dataset in
      Alcotest.(check (list string)) "after reset" expected second_pass)

let test_from_text_file_reset_mid_stream () =
  let content = "alpha\nbeta\ngamma\n" in
  with_temp_file content (fun path ->
      let dataset = from_text_file path in
      let first_chunk = collect_n 1 dataset in
      Alcotest.(check (list string))
        "consumed first element" [ "alpha" ] first_chunk;
      reset dataset;
      let refreshed = collect_n 2 dataset in
      Alcotest.(check (list string))
        "after reset first two elements" [ "alpha"; "beta" ] refreshed)

let test_from_text_files () =
  let content1 = "file1_line1\nfile1_line2\n" in
  let content2 = "file2_line1\nfile2_line2\n" in
  with_temp_file content1 (fun path1 ->
      with_temp_file content2 (fun path2 ->
          let dataset = from_text_files [ path1; path2 ] in
          let collected = collect_dataset dataset in
          Alcotest.(check (list string))
            "all lines"
            [ "file1_line1"; "file1_line2"; "file2_line1"; "file2_line2" ]
            collected))

let test_from_jsonl () =
  let content =
    "{\"text\": \"hello\", \"label\": 0}\n"
    ^ "{\"text\": \"world\", \"label\": 1}\n"
  in
  with_temp_jsonl content (fun path ->
      let dataset = from_jsonl path in
      let collected = collect_dataset dataset in
      Alcotest.(check (list string))
        "extracted text" [ "hello"; "world" ] collected)

let test_from_jsonl_custom_field () =
  let content =
    "{\"content\": \"foo\", \"text\": \"ignore\"}\n"
    ^ "{\"content\": \"bar\", \"text\": \"ignore\"}\n"
  in
  with_temp_jsonl content (fun path ->
      let dataset = from_jsonl ~field:"content" path in
      let collected = collect_dataset dataset in
      Alcotest.(check (list string))
        "extracted content" [ "foo"; "bar" ] collected)

let test_from_csv () =
  let content = "header1,header2,header3\nval1,val2,val3\nval4,val5,val6\n" in
  with_temp_csv content (fun path ->
      let dataset = from_csv path in
      let collected = collect_dataset dataset in
      Alcotest.(check (list string)) "first column" [ "val1"; "val4" ] collected)

let test_from_csv_custom_column () =
  let content = "h1,h2,h3\na,b,c\nd,e,f\n" in
  with_temp_csv content (fun path ->
      let dataset = from_csv ~text_column:2 path in
      let collected = collect_dataset dataset in
      Alcotest.(check (list string)) "third column" [ "c"; "f" ] collected)

let test_from_csv_no_header () =
  let content = "a,b,c\nd,e,f\n" in
  with_temp_csv content (fun path ->
      let dataset = from_csv ~has_header:false path in
      let collected = collect_dataset dataset in
      Alcotest.(check (list string)) "all rows" [ "a"; "d" ] collected)

let test_from_file () =
  let content = "100\n200\n300\n" in
  with_temp_file content (fun path ->
      let parser line = int_of_string line in
      let dataset = from_file parser path in
      let collected = collect_dataset dataset in
      Alcotest.(check (list int)) "parsed integers" [ 100; 200; 300 ] collected)

let test_from_csv_with_labels () =
  let content = "text,label\nhello,spam\nworld,ham\n" in
  with_temp_csv content (fun path ->
      let dataset = from_csv_with_labels ~label_column:1 path in
      let collected = collect_dataset dataset in
      Alcotest.(check (list (pair string string)))
        "text and labels with header"
        [ ("hello", "spam"); ("world", "ham") ]
        collected)

let test_from_csv_with_labels_no_header () =
  let content = "foo,bar\nbaz,qux\n" in
  with_temp_csv content (fun path ->
      let dataset =
        from_csv_with_labels ~label_column:1 ~has_header:false path
      in
      let collected = collect_dataset dataset in
      Alcotest.(check (list (pair string string)))
        "text and labels without header"
        [ ("foo", "bar"); ("baz", "qux") ]
        collected)

let test_from_csv_with_labels_custom_columns () =
  let content = "id,sentiment,text,score\n1,pos,great,5\n2,neg,bad,1\n" in
  with_temp_csv content (fun path ->
      let dataset = from_csv_with_labels ~text_column:2 ~label_column:1 path in
      let collected = collect_dataset dataset in
      Alcotest.(check (list (pair string string)))
        "custom columns"
        [ ("great", "pos"); ("bad", "neg") ]
        collected)

let test_from_csv_with_labels_custom_separator () =
  let content = "t1|l1\nt2|l2\n" in
  with_temp_csv content (fun path ->
      let dataset =
        from_csv_with_labels ~separator:'|' ~label_column:1 ~has_header:false
          path
      in
      let collected = collect_dataset dataset in
      Alcotest.(check (list (pair string string)))
        "text and labels with custom sep"
        [ ("t1", "l1"); ("t2", "l2") ]
        collected)

let test_from_csv_with_labels_malformed_rows () =
  let content = "text,label\nhello,positive\nincomplete\nworld,negative\n" in
  with_temp_csv content (fun path ->
      let dataset = from_csv_with_labels ~label_column:1 path in
      let collected = collect_dataset dataset in
      Alcotest.(check (list (pair string string)))
        "skip malformed rows"
        [ ("hello", "positive"); ("world", "negative") ]
        collected)

let test_map () =
  let ds = from_list [ 1; 2; 3 ] |> map (fun x -> x * 2) in
  let collected = collect_dataset ds in
  Alcotest.(check (list int)) "map doubled" [ 2; 4; 6 ] collected

let test_filter () =
  let ds = from_list [ 1; 2; 3; 4; 5 ] |> filter (fun x -> x mod 2 = 0) in
  let collected = collect_dataset ds in
  Alcotest.(check (list int)) "filter evens" [ 2; 4 ] collected

let test_flat_map () =
  let ds =
    from_list [ 1; 2; 3 ] |> flat_map (fun x -> from_list [ x; x + 1 ])
  in
  let collected = collect_dataset ds in
  Alcotest.(check (list int)) "flat_map expanded" [ 1; 2; 2; 3; 3; 4 ] collected

let test_zip () =
  let ds1 = from_list [ "a"; "b"; "c" ] in
  let ds2 = from_list [ 1; 2; 3; 4 ] in
  (* One extra element *)
  let dataset = zip ds1 ds2 in
  let collected = collect_dataset dataset in
  Alcotest.(check (list (pair string int)))
    "zipped pairs"
    [ ("a", 1); ("b", 2); ("c", 3) ]
    collected

let test_concatenate () =
  let ds1 = from_list [ 1; 2 ] in
  let ds2 = from_list [ 3; 4; 5 ] in
  let dataset = concatenate ds1 ds2 in
  let collected = collect_dataset dataset in
  Alcotest.(check (list int)) "concatenated" [ 1; 2; 3; 4; 5 ] collected

let test_interleave () =
  let ds1 = from_list [ 1; 2; 3 ] in
  let ds2 = from_list [ 10; 20 ] in
  let ds3 = from_list [ 100 ] in
  let dataset = interleave [ ds1; ds2; ds3 ] in
  let collected = collect_dataset dataset in
  (* Round-robin: 1, 10, 100, 2, 20, 3 *)
  Alcotest.(check (list int)) "interleaved" [ 1; 10; 100; 2; 20; 3 ] collected

let test_enumerate () =
  let dataset = from_list [ "a"; "b"; "c" ] |> enumerate in
  let collected = collect_dataset dataset in
  Alcotest.(check (list (pair int string)))
    "enumerated"
    [ (0, "a"); (1, "b"); (2, "c") ]
    collected

(** Test text processing *)
let test_normalize () =
  let dataset =
    from_list [ "Hello WORLD!"; "  multiple   spaces  " ]
    |> normalize ~lowercase:true ~remove_punctuation:true
         ~collapse_whitespace:true
  in
  let collected = collect_dataset dataset in
  Alcotest.(check (list string))
    "normalized"
    [ "hello world"; "multiple spaces" ]
    collected

let test_tokenize_whitespace () =
  let dataset =
    from_list [ "hello world"; "foo bar baz" ] |> tokenize whitespace_tokenizer
  in
  let collected = collect_dataset dataset in
  Alcotest.(check int) "number of samples" 2 (List.length collected);
  (* Check that tokens are integers *)
  List.iter
    (fun tokens -> Array.iter (fun tok -> assert (tok >= 0)) tokens)
    collected

let test_tokenize_with_special_tokens () =
  let dataset =
    from_list [ "hello" ]
    |> tokenize whitespace_tokenizer ~add_special_tokens:true
  in
  let collected = collect_dataset dataset in
  match collected with
  | [ tokens ] ->
      Alcotest.(check bool) "has BOS token" true (tokens.(0) = 0);
      Alcotest.(check bool)
        "has EOS token" true
        (tokens.(Array.length tokens - 1) = 1)
  | _ -> Alcotest.fail "Expected one tokenized sample"

let test_tokenize_truncation () =
  let dataset =
    from_list [ "one two three four five" ]
    |> tokenize whitespace_tokenizer ~max_length:3 ~truncation:true
  in
  let collected = collect_dataset dataset in
  match collected with
  | [ tokens ] ->
      Alcotest.(check int) "truncated length" 3 (Array.length tokens)
  | _ -> Alcotest.fail "Expected one tokenized sample"

let test_tokenize_padding () =
  let dataset =
    from_list [ "one two" ] |> tokenize whitespace_tokenizer ~padding:(`Max 5)
  in
  let collected = collect_dataset dataset in
  match collected with
  | [ tokens ] ->
      Alcotest.(check int) "padded length" 5 (Array.length tokens);
      (* Check last elements are padding (0) *)
      Alcotest.(check int) "padding value" 0 tokens.(4)
  | _ -> Alcotest.fail "Expected one tokenized sample"

(** Test batching *)
let test_batch_basic () =
  let dataset = from_list [ 1; 2; 3; 4; 5 ] |> batch_map 2 Fun.id in
  let collected = collect_dataset dataset in
  Alcotest.(check int) "number of batches" 3 (List.length collected);
  assert_equal_int_array [| 1; 2 |] (List.nth collected 0);
  assert_equal_int_array [| 3; 4 |] (List.nth collected 1);
  assert_equal_int_array [| 5 |] (List.nth collected 2)

let test_batch_drop_remainder () =
  let dataset =
    from_list [ 1; 2; 3; 4; 5 ] |> batch_map ~drop_remainder:true 2 Fun.id
  in
  let collected = collect_dataset dataset in
  Alcotest.(check int) "number of batches" 2 (List.length collected);
  assert_equal_int_array [| 1; 2 |] (List.nth collected 0);
  assert_equal_int_array [| 3; 4 |] (List.nth collected 1)

let test_batch_map_combiner () =
  let combiner arr = Array.fold_left ( + ) 0 arr in
  let dataset = from_list [ 1; 2; 3; 4 ] |> batch_map 2 combiner in
  let collected = collect_dataset dataset in
  Alcotest.(check (list int)) "batch sums" [ 3; 7 ] collected

let test_bucket_by_length () =
  let strings = [ "a"; "bb"; "ccc"; "dddd"; "e"; "ff"; "ggg" ] in
  let dataset =
    from_list strings
    |> bucket_by_length ~boundaries:[ 2; 3 ] ~batch_sizes:[ 2; 2; 1 ]
         String.length
  in
  let collected = collect_dataset dataset in
  (* Should create buckets: <2: ["a", "e"], 2-3: ["bb", "ff"], >3: ["ccc",
     "dddd", "ggg"] *)
  Alcotest.(check bool) "has batches" true (List.length collected > 0)

(** Test iteration control *)
let test_take () =
  let dataset = from_list [ 1; 2; 3; 4; 5 ] |> take 3 in
  let collected = collect_dataset dataset in
  Alcotest.(check (list int)) "took 3" [ 1; 2; 3 ] collected

let test_skip () =
  let dataset = from_list [ 1; 2; 3; 4; 5 ] |> skip 2 in
  let collected = collect_dataset dataset in
  Alcotest.(check (list int)) "skipped 2" [ 3; 4; 5 ] collected

let test_repeat_finite () =
  let dataset = from_list [ 1; 2 ] |> repeat ~count:3 in
  let collected = collect_dataset dataset in
  Alcotest.(check (list int)) "repeated 3 times" [ 1; 2; 1; 2; 1; 2 ] collected

let test_repeat_infinite () =
  let dataset = from_list [ 1; 2 ] |> repeat in
  let collected = collect_n 5 dataset in
  Alcotest.(check (list int)) "first 5 of infinite" [ 1; 2; 1; 2; 1 ] collected

let test_window () =
  let dataset = from_list [ 1; 2; 3; 4; 5 ] |> window 3 in
  let collected = collect_dataset dataset in
  Alcotest.(check int) "number of windows" 2 (List.length collected);
  assert_equal_int_array [| 1; 2; 3 |] (List.nth collected 0);
  assert_equal_int_array [| 4; 5 |] (List.nth collected 1)
(* Last window incomplete *)

let test_window_with_shift () =
  let dataset =
    from_list [ 1; 2; 3; 4; 5 ] |> window ~shift:1 ~drop_remainder:true 3
  in
  let collected = collect_dataset dataset in
  (* Overlapping windows: [1,2,3], [2,3,4], [3,4,5] *)
  Alcotest.(check int) "number of windows" 3 (List.length collected);
  assert_equal_int_array [| 1; 2; 3 |] (List.nth collected 0);
  assert_equal_int_array [| 2; 3; 4 |] (List.nth collected 1);
  assert_equal_int_array [| 3; 4; 5 |] (List.nth collected 2)

let test_window_drop_remainder () =
  let dataset = from_list [ 1; 2; 3; 4; 5 ] |> window ~drop_remainder:true 3 in
  let collected = collect_dataset dataset in
  Alcotest.(check int) "number of complete windows" 1 (List.length collected);
  assert_equal_int_array [| 1; 2; 3 |] (List.nth collected 0)

(** Test shuffling and sampling *)
let test_shuffle () =
  let dataset = from_list [ 1; 2; 3; 4; 5 ] |> shuffle ~buffer_size:3 in
  let collected = collect_dataset dataset in
  Alcotest.(check int) "same length after shuffle" 5 (List.length collected);
  (* Check all elements are present (order may differ) *)
  let sorted = List.sort compare collected in
  Alcotest.(check (list int)) "all elements present" [ 1; 2; 3; 4; 5 ] sorted

let test_shuffle_deterministic () =
  let rng = Rune.Rng.key 42 in
  let dataset1 = from_list [ 1; 2; 3; 4; 5 ] |> shuffle ~rng ~buffer_size:5 in
  let collected1 = collect_dataset dataset1 in

  let rng = Rune.Rng.key 42 in
  let dataset2 = from_list [ 1; 2; 3; 4; 5 ] |> shuffle ~rng ~buffer_size:5 in
  let collected2 = collect_dataset dataset2 in

  Alcotest.(check (list int))
    "same shuffle with same seed" collected1 collected2

let test_sample_without_replacement () =
  let dataset = from_list [ 1; 2; 3; 4; 5 ] |> sample 3 in
  let collected = collect_dataset dataset in
  Alcotest.(check int) "sampled 3" 3 (List.length collected);
  (* Check no duplicates *)
  let unique = List.sort_uniq compare collected in
  Alcotest.(check int) "no duplicates" 3 (List.length unique)

let test_sample_with_replacement () =
  let rng = Rune.Rng.key 42 in
  let dataset = from_list [ 1; 2; 3 ] |> sample ~rng ~replacement:true 5 in
  let collected = collect_dataset dataset in
  Alcotest.(check int) "sampled 5 with replacement" 5 (List.length collected);
  (* All samples should be from original set *)
  List.iter
    (fun x -> Alcotest.(check bool) "sample in range" true (x >= 1 && x <= 3))
    collected

let test_weighted_sample () =
  let weights = [| 0.1; 0.1; 0.8 |] in
  (* Heavy bias toward third element *)
  let rng = Rune.Rng.key 42 in
  let dataset =
    from_list [ "a"; "b"; "c" ] |> weighted_sample ~rng ~weights 10
  in
  let collected = collect_dataset dataset in
  Alcotest.(check int) "sampled 10" 10 (List.length collected);
  (* Most samples should be "c" due to weight *)
  let c_count = List.filter (( = ) "c") collected |> List.length in
  Alcotest.(check bool) "mostly c" true (c_count >= 5)

(** Test caching and prefetching *)
let test_cache_in_memory () =
  (* Create a dataset that tracks how many times it's been accessed *)
  let access_count = ref 0 in
  let original = [ 1; 2; 3 ] in

  (* Use from_seq to create a dataset that we can track *)
  let make_dataset () =
    incr access_count;
    from_seq (List.to_seq original)
  in

  let dataset = make_dataset () in
  let cached = cache dataset in

  (* First pass - builds cache *)
  let collected1 = collect_dataset cached in
  Alcotest.(check (list int)) "first pass" [ 1; 2; 3 ] collected1;

  (* Second pass - test that creating a new cache on same data works *)
  (* Since we can't reset from outside, test with a fresh dataset *)
  let dataset2 = make_dataset () in
  let cached2 = cache dataset2 in
  let collected2 = collect_dataset cached2 in
  Alcotest.(check (list int)) "second pass" [ 1; 2; 3 ] collected2

let test_prefetch () =
  let dataset = from_list [ 1; 2; 3; 4; 5 ] |> prefetch ~buffer_size:2 in
  let collected = collect_dataset dataset in
  Alcotest.(check (list int)) "prefetched" [ 1; 2; 3; 4; 5 ] collected

(** Test parallel processing *)
let test_parallel_map () =
  let dataset =
    from_list [ 1; 2; 3; 4; 5 ] |> parallel_map ~num_workers:2 (fun x -> x * x)
  in
  let collected = collect_dataset dataset in
  let sorted = List.sort compare collected in
  (* Order may vary *)
  Alcotest.(check (list int)) "squared values" [ 1; 4; 9; 16; 25 ] sorted

let test_parallel_interleave () =
  let dataset =
    from_list [ 1; 2; 3 ]
    |> parallel_interleave ~num_workers:2 ~block_length:1 (fun x ->
           from_list [ x; x * 10 ])
  in
  let collected = collect_dataset dataset in
  Alcotest.(check int) "correct number of elements" 6 (List.length collected);
  (* Check all expected values are present *)
  let sorted = List.sort compare collected in
  Alcotest.(check (list int))
    "all values present" [ 1; 2; 3; 10; 20; 30 ] sorted

(** Test iteration functions *)
let test_iter () =
  let sum = ref 0 in
  from_list [ 1; 2; 3; 4; 5 ] |> iter (fun x -> sum := !sum + x);
  Alcotest.(check int) "sum via iter" 15 !sum

let test_fold () =
  let result = from_list [ 1; 2; 3; 4 ] |> fold ( + ) 0 in
  Alcotest.(check int) "fold sum" 10 result

let test_to_seq () =
  let dataset = from_list [ 1; 2; 3 ] in
  let seq = to_seq dataset in
  let collected = List.of_seq seq in
  Alcotest.(check (list int)) "to_seq" [ 1; 2; 3 ] collected

let test_to_list () =
  let dataset = from_list [ 1; 2; 3 ] |> map (fun x -> x * 2) in
  let lst = to_list dataset in
  Alcotest.(check (list int)) "to_list" [ 2; 4; 6 ] lst

let test_to_array () =
  let dataset = from_list [ "a"; "b"; "c" ] in
  let arr = to_array dataset in
  assert_equal_string_array [| "a"; "b"; "c" |] arr

(** Test dataset information *)
let test_cardinality_finite () =
  let dataset = from_list [ 1; 2; 3 ] in
  match cardinality dataset with
  | Finite n -> Alcotest.(check int) "finite cardinality" 3 n
  | _ -> Alcotest.fail "Expected finite cardinality"

let test_cardinality_unknown () =
  let dataset = from_list [ 1; 2; 3 ] |> filter (fun x -> x > 1) in
  match cardinality dataset with
  | Unknown -> ()
  | _ -> Alcotest.fail "Expected unknown cardinality"

let test_cardinality_infinite () =
  let dataset = from_list [ 1; 2 ] |> repeat in
  match cardinality dataset with
  | Infinite -> () (* repeat without count returns Infinite *)
  | _ -> Alcotest.fail "Expected infinite cardinality for infinite repeat"

let test_element_spec () =
  let dataset = from_list [ "hello"; "world" ] in
  match element_spec dataset with
  | Unknown -> () (* Default spec for from_list *)
  | _ -> Alcotest.fail "Expected unknown element spec"

(** Test pipelines *)
let test_text_classification_pipeline () =
  let dataset =
    from_list [ "hello world"; "foo bar baz" ]
    |> text_classification_pipeline ~batch_size:2
  in
  let collected = collect_n 1 dataset in
  Alcotest.(check int) "got batch" 1 (List.length collected);
  match collected with
  | [ batch ] ->
      (* Check that we got a tensor *)
      Alcotest.(check bool)
        "got tensor" true
        (Rune.shape batch |> Array.length > 0)
  | _ -> Alcotest.fail "Expected one batch"

let test_language_model_pipeline () =
  let dataset =
    from_list [ "one two three four" ]
    |> language_model_pipeline ~sequence_length:4 ~batch_size:1
  in
  let collected = collect_n 1 dataset in
  match collected with
  | [ (input, target) ] ->
      Alcotest.(check bool)
        "input/target tensors" true
        (Rune.shape input |> Array.length > 0
        && Rune.shape target |> Array.length > 0)
  | _ -> Alcotest.fail "Expected batch of input/target pairs"

(** Test edge cases *)
let test_empty_dataset () =
  let dataset = from_list [] in
  let collected = collect_dataset dataset in
  Alcotest.(check (list int)) "empty dataset" [] collected

let test_single_element () =
  let dataset = from_list [ 42 ] in
  let collected = collect_dataset dataset in
  Alcotest.(check (list int)) "single element" [ 42 ] collected

let test_chain_many_operations () =
  let dataset =
    from_list [ 1; 2; 3; 4; 5; 6; 7; 8; 9; 10 ]
    |> filter (fun x -> x mod 2 = 0) (* [2; 4; 6; 8; 10] *)
    |> map (fun x -> x * 2) (* [4; 8; 12; 16; 20] *)
    |> take 3 (* [4; 8; 12] *)
    |> batch_map 2 Fun.id (* [[4; 8], [12]] *)
  in
  let collected = collect_dataset dataset in
  Alcotest.(check int) "number of batches" 2 (List.length collected);
  assert_equal_int_array [| 4; 8 |] (List.nth collected 0);
  assert_equal_int_array [| 12 |] (List.nth collected 1)

(** Test suite *)
let () =
  let open Alcotest in
  run "Dataset"
    [
      ( "creation",
        [
          test_case "from_array" `Quick test_from_array;
          test_case "from_list" `Quick test_from_list;
          test_case "from_seq" `Quick test_from_seq;
        ] );
      ( "text_files",
        [
          test_case "from_text_file" `Quick test_from_text_file;
          test_case "from_text_file_utf8" `Quick test_from_text_file_utf8;
          test_case "from_text_file_latin1" `Quick test_from_text_file_latin1;
          test_case "from_text_file_large_lines" `Quick
            test_from_text_file_large_lines;
          test_case "from_text_file_reset" `Quick test_from_text_file_reset;
          test_case "from_text_file_reset_mid_stream" `Quick
            test_from_text_file_reset_mid_stream;
          test_case "from_text_files" `Quick test_from_text_files;
          test_case "from_jsonl" `Quick test_from_jsonl;
          test_case "from_jsonl_custom_field" `Quick
            test_from_jsonl_custom_field;
          test_case "from_csv" `Quick test_from_csv;
          test_case "from_csv_custom_column" `Quick test_from_csv_custom_column;
          test_case "from_csv_no_header" `Quick test_from_csv_no_header;
          test_case "from_file" `Quick test_from_file;
          test_case "from_csv_with_labels" `Quick test_from_csv_with_labels;
          test_case "from_csv_with_labels_no_header" `Quick
            test_from_csv_with_labels_no_header;
          test_case "from_csv_with_labels_custom_columns" `Quick
            test_from_csv_with_labels_custom_columns;
          test_case "from_csv_with_labels_custom_separator" `Quick
            test_from_csv_with_labels_custom_separator;
          test_case "from_csv_with_labels_malformed_rows" `Quick
            test_from_csv_with_labels_malformed_rows;
        ] );
      ( "transformations",
        [
          test_case "map" `Quick test_map;
          test_case "filter" `Quick test_filter;
          test_case "flat_map" `Quick test_flat_map;
          test_case "zip" `Quick test_zip;
          test_case "concatenate" `Quick test_concatenate;
          test_case "interleave" `Quick test_interleave;
          test_case "enumerate" `Quick test_enumerate;
        ] );
      ( "text_processing",
        [
          test_case "normalize" `Quick test_normalize;
          test_case "tokenize_whitespace" `Quick test_tokenize_whitespace;
          test_case "tokenize_with_special_tokens" `Quick
            test_tokenize_with_special_tokens;
          test_case "tokenize_truncation" `Quick test_tokenize_truncation;
          test_case "tokenize_padding" `Quick test_tokenize_padding;
        ] );
      ( "batching",
        [
          test_case "batch_basic" `Quick test_batch_basic;
          test_case "batch_drop_remainder" `Quick test_batch_drop_remainder;
          test_case "batch_map_combiner" `Quick test_batch_map_combiner;
          test_case "bucket_by_length" `Quick test_bucket_by_length;
        ] );
      ( "iteration_control",
        [
          test_case "take" `Quick test_take;
          test_case "skip" `Quick test_skip;
          test_case "repeat_finite" `Quick test_repeat_finite;
          test_case "repeat_infinite" `Quick test_repeat_infinite;
          test_case "window" `Quick test_window;
          test_case "window_with_shift" `Quick test_window_with_shift;
          test_case "window_drop_remainder" `Quick test_window_drop_remainder;
        ] );
      ( "shuffling_sampling",
        [
          test_case "shuffle" `Quick test_shuffle;
          test_case "shuffle_deterministic" `Quick test_shuffle_deterministic;
          test_case "sample_without_replacement" `Quick
            test_sample_without_replacement;
          test_case "sample_with_replacement" `Quick
            test_sample_with_replacement;
          test_case "weighted_sample" `Quick test_weighted_sample;
        ] );
      ( "caching_prefetching",
        [
          test_case "cache_in_memory" `Quick test_cache_in_memory;
          test_case "prefetch" `Quick test_prefetch;
        ] );
      ( "parallel",
        [
          test_case "parallel_map" `Quick test_parallel_map;
          test_case "parallel_interleave" `Quick test_parallel_interleave;
        ] );
      ( "iteration",
        [
          test_case "iter" `Quick test_iter;
          test_case "fold" `Quick test_fold;
          test_case "to_seq" `Quick test_to_seq;
          test_case "to_list" `Quick test_to_list;
          test_case "to_array" `Quick test_to_array;
        ] );
      ( "dataset_info",
        [
          test_case "cardinality_finite" `Quick test_cardinality_finite;
          test_case "cardinality_unknown" `Quick test_cardinality_unknown;
          test_case "cardinality_infinite" `Quick test_cardinality_infinite;
          test_case "element_spec" `Quick test_element_spec;
        ] );
      ( "pipelines",
        [
          test_case "text_classification_pipeline" `Quick
            test_text_classification_pipeline;
          test_case "language_model_pipeline" `Quick
            test_language_model_pipeline;
        ] );
      ( "edge_cases",
        [
          test_case "empty_dataset" `Quick test_empty_dataset;
          test_case "single_element" `Quick test_single_element;
          test_case "chain_many_operations" `Quick test_chain_many_operations;
        ] );
    ]
