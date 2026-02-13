(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Kaun.Dataset

(* Test helpers *)
let assert_equal_int_array expected actual =
  equal ~msg:"arrays equal" (array int) expected actual

let assert_equal_string_array expected actual =
  equal ~msg:"arrays equal" (array string) expected actual

let assert_dataset_length expected dataset =
  match cardinality dataset with
  | Finite n -> equal ~msg:"dataset length" int expected n
  | _ -> fail "Expected finite dataset"

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

(* ───── Test Dataset Creation ───── *)

let test_from_array () =
  let arr = [| 1; 2; 3; 4; 5 |] in
  let dataset = from_array arr in
  assert_dataset_length 5 dataset;
  let collected = collect_dataset dataset in
  equal ~msg:"collected values" (list int) [ 1; 2; 3; 4; 5 ] collected

let test_from_list () =
  let lst = [ "a"; "b"; "c" ] in
  let dataset = from_list lst in
  assert_dataset_length 3 dataset;
  let collected = collect_dataset dataset in
  equal ~msg:"collected values" (list string) [ "a"; "b"; "c" ] collected

let test_from_seq () =
  let seq = List.to_seq [ 10; 20; 30 ] in
  let dataset = from_seq seq in
  let collected = collect_dataset dataset in
  equal ~msg:"collected values" (list int) [ 10; 20; 30 ] collected

(* ───── Test Text File Reading ───── *)

let test_from_text_file () =
  let content = "line1\nline2\nline3\n" in
  with_temp_file content (fun path ->
      let dataset = from_text_file path in
      let collected = collect_dataset dataset in
      equal ~msg:"lines read" (list string)
        [ "line1"; "line2"; "line3" ]
        collected)

(* Test for utf8 *)
let test_from_text_file_utf8 () =
  let content = "hello \xF0\x9F\x98\x8A\nsecond\n" in
  with_temp_file content (fun path ->
      let ds = from_text_file ~encoding:`UTF8 path in
      let lines = collect_dataset ds in
      equal ~msg:"utf8 emoji preserved" (list string) [ "hello 😊"; "second" ]
        lines)

(* Test for Latin1 *)
let test_from_text_file_latin1 () =
  let content = "caf\xE9\nna\xEFve\n" in
  with_temp_file content (fun path ->
      let ds = from_text_file ~encoding:`LATIN1 path in
      let lines = collect_dataset ds in
      equal ~msg:"latin1 decoded" (list string) [ "café"; "naïve" ] lines)

let test_from_text_file_large_lines () =
  let line = String.make 1000 'x' in
  let content = line ^ "\n" ^ line ^ "\n" in
  with_temp_file content (fun path ->
      let dataset = from_text_file ~chunk_size:100 path in
      let collected = collect_dataset dataset in
      equal ~msg:"number of lines" int 2 (List.length collected);
      List.iter
        (fun l -> equal ~msg:"line length" int 1000 (String.length l))
        collected)

let test_from_text_file_reset () =
  let content = "line1\nline2\n" in
  with_temp_file content (fun path ->
      let dataset = from_text_file path in
      let expected = [ "line1"; "line2" ] in
      let first_pass = collect_dataset dataset in
      equal ~msg:"first pass" (list string) expected first_pass;
      reset dataset;
      let second_pass = collect_dataset dataset in
      equal ~msg:"after reset" (list string) expected second_pass)

let test_from_text_file_reset_mid_stream () =
  let content = "alpha\nbeta\ngamma\n" in
  with_temp_file content (fun path ->
      let dataset = from_text_file path in
      let first_chunk = collect_n 1 dataset in
      equal ~msg:"consumed first element" (list string) [ "alpha" ] first_chunk;
      reset dataset;
      let refreshed = collect_n 2 dataset in
      equal ~msg:"after reset first two elements" (list string)
        [ "alpha"; "beta" ] refreshed)

let test_from_text_files () =
  let content1 = "file1_line1\nfile1_line2\n" in
  let content2 = "file2_line1\nfile2_line2\n" in
  with_temp_file content1 (fun path1 ->
      with_temp_file content2 (fun path2 ->
          let dataset = from_text_files [ path1; path2 ] in
          let collected = collect_dataset dataset in
          equal ~msg:"all lines" (list string)
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
      equal ~msg:"extracted text" (list string) [ "hello"; "world" ] collected)

let test_from_jsonl_custom_field () =
  let content =
    "{\"content\": \"foo\", \"text\": \"ignore\"}\n"
    ^ "{\"content\": \"bar\", \"text\": \"ignore\"}\n"
  in
  with_temp_jsonl content (fun path ->
      let dataset = from_jsonl ~field:"content" path in
      let collected = collect_dataset dataset in
      equal ~msg:"extracted content" (list string) [ "foo"; "bar" ] collected)

let test_from_csv () =
  let content = "header1,header2,header3\nval1,val2,val3\nval4,val5,val6\n" in
  with_temp_csv content (fun path ->
      let dataset = from_csv path in
      let collected = collect_dataset dataset in
      equal ~msg:"first column" (list string) [ "val1"; "val4" ] collected)

let test_from_csv_custom_column () =
  let content = "h1,h2,h3\na,b,c\nd,e,f\n" in
  with_temp_csv content (fun path ->
      let dataset = from_csv ~text_column:2 path in
      let collected = collect_dataset dataset in
      equal ~msg:"third column" (list string) [ "c"; "f" ] collected)

let test_from_csv_no_header () =
  let content = "a,b,c\nd,e,f\n" in
  with_temp_csv content (fun path ->
      let dataset = from_csv ~has_header:false path in
      let collected = collect_dataset dataset in
      equal ~msg:"all rows" (list string) [ "a"; "d" ] collected)

let test_from_file () =
  let content = "100\n200\n300\n" in
  with_temp_file content (fun path ->
      let parser line = int_of_string line in
      let dataset = from_file parser path in
      let collected = collect_dataset dataset in
      equal ~msg:"parsed integers" (list int) [ 100; 200; 300 ] collected)

let test_from_csv_with_labels () =
  let content = "text,label\nhello,spam\nworld,ham\n" in
  with_temp_csv content (fun path ->
      let dataset = from_csv_with_labels ~label_column:1 path in
      let collected = collect_dataset dataset in
      equal ~msg:"text and labels with header"
        (list (pair string string))
        [ ("hello", "spam"); ("world", "ham") ]
        collected)

let test_from_csv_with_labels_no_header () =
  let content = "foo,bar\nbaz,qux\n" in
  with_temp_csv content (fun path ->
      let dataset =
        from_csv_with_labels ~label_column:1 ~has_header:false path
      in
      let collected = collect_dataset dataset in
      equal ~msg:"text and labels without header"
        (list (pair string string))
        [ ("foo", "bar"); ("baz", "qux") ]
        collected)

let test_from_csv_with_labels_custom_columns () =
  let content = "id,sentiment,text,score\n1,pos,great,5\n2,neg,bad,1\n" in
  with_temp_csv content (fun path ->
      let dataset = from_csv_with_labels ~text_column:2 ~label_column:1 path in
      let collected = collect_dataset dataset in
      equal ~msg:"custom columns"
        (list (pair string string))
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
      equal ~msg:"text and labels with custom sep"
        (list (pair string string))
        [ ("t1", "l1"); ("t2", "l2") ]
        collected)

let test_from_csv_with_labels_malformed_rows () =
  let content = "text,label\nhello,positive\nincomplete\nworld,negative\n" in
  with_temp_csv content (fun path ->
      let dataset = from_csv_with_labels ~label_column:1 path in
      let collected = collect_dataset dataset in
      equal ~msg:"skip malformed rows"
        (list (pair string string))
        [ ("hello", "positive"); ("world", "negative") ]
        collected)

let test_map () =
  let ds = from_list [ 1; 2; 3 ] |> map (fun x -> x * 2) in
  let collected = collect_dataset ds in
  equal ~msg:"map doubled" (list int) [ 2; 4; 6 ] collected

let test_filter () =
  let ds = from_list [ 1; 2; 3; 4; 5 ] |> filter (fun x -> x mod 2 = 0) in
  let collected = collect_dataset ds in
  equal ~msg:"filter evens" (list int) [ 2; 4 ] collected

let test_flat_map () =
  let ds =
    from_list [ 1; 2; 3 ] |> flat_map (fun x -> from_list [ x; x + 1 ])
  in
  let collected = collect_dataset ds in
  equal ~msg:"flat_map expanded" (list int) [ 1; 2; 2; 3; 3; 4 ] collected

let test_zip () =
  let ds1 = from_list [ "a"; "b"; "c" ] in
  let ds2 = from_list [ 1; 2; 3; 4 ] in
  (* One extra element *)
  let dataset = zip ds1 ds2 in
  let collected = collect_dataset dataset in
  equal ~msg:"zipped pairs"
    (list (pair string int))
    [ ("a", 1); ("b", 2); ("c", 3) ]
    collected

let test_concatenate () =
  let ds1 = from_list [ 1; 2 ] in
  let ds2 = from_list [ 3; 4; 5 ] in
  let dataset = concatenate ds1 ds2 in
  let collected = collect_dataset dataset in
  equal ~msg:"concatenated" (list int) [ 1; 2; 3; 4; 5 ] collected

let test_interleave () =
  let ds1 = from_list [ 1; 2; 3 ] in
  let ds2 = from_list [ 10; 20 ] in
  let ds3 = from_list [ 100 ] in
  let dataset = interleave [ ds1; ds2; ds3 ] in
  let collected = collect_dataset dataset in
  (* Round-robin: 1, 10, 100, 2, 20, 3 *)
  equal ~msg:"interleaved" (list int) [ 1; 10; 100; 2; 20; 3 ] collected

let test_enumerate () =
  let dataset = from_list [ "a"; "b"; "c" ] |> enumerate in
  let collected = collect_dataset dataset in
  equal ~msg:"enumerated"
    (list (pair int string))
    [ (0, "a"); (1, "b"); (2, "c") ]
    collected

(* ───── Test Text Processing ───── *)

let test_normalize () =
  let dataset =
    from_list [ "Hello WORLD!"; "  multiple   spaces  " ]
    |> normalize ~lowercase:true ~remove_punctuation:true
         ~collapse_whitespace:true
  in
  let collected = collect_dataset dataset in
  equal ~msg:"normalized" (list string)
    [ "hello world"; "multiple spaces" ]
    collected

let test_tokenize_whitespace () =
  let dataset =
    from_list [ "hello world"; "foo bar baz" ] |> tokenize whitespace_tokenizer
  in
  let collected = collect_dataset dataset in
  equal ~msg:"number of samples" int 2 (List.length collected);
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
      equal ~msg:"has BOS token" bool true (tokens.(0) = 0);
      equal ~msg:"has EOS token" bool true (tokens.(Array.length tokens - 1) = 1)
  | _ -> fail "Expected one tokenized sample"

let test_tokenize_truncation () =
  let dataset =
    from_list [ "one two three four five" ]
    |> tokenize whitespace_tokenizer ~max_length:3 ~truncation:true
  in
  let collected = collect_dataset dataset in
  match collected with
  | [ tokens ] -> equal ~msg:"truncated length" int 3 (Array.length tokens)
  | _ -> fail "Expected one tokenized sample"

let test_tokenize_padding () =
  let dataset =
    from_list [ "one two" ] |> tokenize whitespace_tokenizer ~padding:(`Max 5)
  in
  let collected = collect_dataset dataset in
  match collected with
  | [ tokens ] ->
      equal ~msg:"padded length" int 5 (Array.length tokens);
      (* Check last elements are padding (0) *)
      equal ~msg:"padding value" int 0 tokens.(4)
  | _ -> fail "Expected one tokenized sample"

(* ───── Test Batching ───── *)

let test_batch_basic () =
  let dataset = from_list [ 1; 2; 3; 4; 5 ] |> batch_map 2 Fun.id in
  let collected = collect_dataset dataset in
  equal ~msg:"number of batches" int 3 (List.length collected);
  assert_equal_int_array [| 1; 2 |] (List.nth collected 0);
  assert_equal_int_array [| 3; 4 |] (List.nth collected 1);
  assert_equal_int_array [| 5 |] (List.nth collected 2)

let test_batch_drop_remainder () =
  let dataset =
    from_list [ 1; 2; 3; 4; 5 ] |> batch_map ~drop_remainder:true 2 Fun.id
  in
  let collected = collect_dataset dataset in
  equal ~msg:"number of batches" int 2 (List.length collected);
  assert_equal_int_array [| 1; 2 |] (List.nth collected 0);
  assert_equal_int_array [| 3; 4 |] (List.nth collected 1)

let test_batch_map_combiner () =
  let combiner arr = Array.fold_left ( + ) 0 arr in
  let dataset = from_list [ 1; 2; 3; 4 ] |> batch_map 2 combiner in
  let collected = collect_dataset dataset in
  equal ~msg:"batch sums" (list int) [ 3; 7 ] collected

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
  equal ~msg:"has batches" bool true (List.length collected > 0)

(* ───── Test Iteration Control ───── *)

let test_take () =
  let dataset = from_list [ 1; 2; 3; 4; 5 ] |> take 3 in
  let collected = collect_dataset dataset in
  equal ~msg:"took 3" (list int) [ 1; 2; 3 ] collected

let test_skip () =
  let dataset = from_list [ 1; 2; 3; 4; 5 ] |> skip 2 in
  let collected = collect_dataset dataset in
  equal ~msg:"skipped 2" (list int) [ 3; 4; 5 ] collected

let test_repeat_finite () =
  let dataset = from_list [ 1; 2 ] |> repeat ~count:3 in
  let collected = collect_dataset dataset in
  equal ~msg:"repeated 3 times" (list int) [ 1; 2; 1; 2; 1; 2 ] collected

let test_repeat_infinite () =
  let dataset = from_list [ 1; 2 ] |> repeat in
  let collected = collect_n 5 dataset in
  equal ~msg:"first 5 of infinite" (list int) [ 1; 2; 1; 2; 1 ] collected

let test_window () =
  let dataset = from_list [ 1; 2; 3; 4; 5 ] |> window 3 in
  let collected = collect_dataset dataset in
  equal ~msg:"number of windows" int 2 (List.length collected);
  assert_equal_int_array [| 1; 2; 3 |] (List.nth collected 0);
  assert_equal_int_array [| 4; 5 |] (List.nth collected 1)
(* Last window incomplete *)

let test_window_with_shift () =
  let dataset =
    from_list [ 1; 2; 3; 4; 5 ] |> window ~shift:1 ~drop_remainder:true 3
  in
  let collected = collect_dataset dataset in
  (* Overlapping windows: [1,2,3], [2,3,4], [3,4,5] *)
  equal ~msg:"number of windows" int 3 (List.length collected);
  assert_equal_int_array [| 1; 2; 3 |] (List.nth collected 0);
  assert_equal_int_array [| 2; 3; 4 |] (List.nth collected 1);
  assert_equal_int_array [| 3; 4; 5 |] (List.nth collected 2)

let test_window_drop_remainder () =
  let dataset = from_list [ 1; 2; 3; 4; 5 ] |> window ~drop_remainder:true 3 in
  let collected = collect_dataset dataset in
  equal ~msg:"number of complete windows" int 1 (List.length collected);
  assert_equal_int_array [| 1; 2; 3 |] (List.nth collected 0)

(* ───── Test Shuffling And Sampling ───── *)

let test_shuffle () =
  let dataset = from_list [ 1; 2; 3; 4; 5 ] |> shuffle ~buffer_size:3 in
  let collected = collect_dataset dataset in
  equal ~msg:"same length after shuffle" int 5 (List.length collected);
  (* Check all elements are present (order may differ) *)
  let sorted = List.sort compare collected in
  equal ~msg:"all elements present" (list int) [ 1; 2; 3; 4; 5 ] sorted

let test_shuffle_deterministic () =
  let rng = Rune.Rng.key 42 in
  let dataset1 = from_list [ 1; 2; 3; 4; 5 ] |> shuffle ~rng ~buffer_size:5 in
  let collected1 = collect_dataset dataset1 in

  let rng = Rune.Rng.key 42 in
  let dataset2 = from_list [ 1; 2; 3; 4; 5 ] |> shuffle ~rng ~buffer_size:5 in
  let collected2 = collect_dataset dataset2 in

  equal ~msg:"same shuffle with same seed" (list int) collected1 collected2

let test_sample_without_replacement () =
  let dataset = from_list [ 1; 2; 3; 4; 5 ] |> sample 3 in
  let collected = collect_dataset dataset in
  equal ~msg:"sampled 3" int 3 (List.length collected);
  (* Check no duplicates *)
  let unique = List.sort_uniq compare collected in
  equal ~msg:"no duplicates" int 3 (List.length unique)

let test_sample_with_replacement () =
  let rng = Rune.Rng.key 42 in
  let dataset = from_list [ 1; 2; 3 ] |> sample ~rng ~replacement:true 5 in
  let collected = collect_dataset dataset in
  equal ~msg:"sampled 5 with replacement" int 5 (List.length collected);
  (* All samples should be from original set *)
  List.iter
    (fun x -> equal ~msg:"sample in range" bool true (x >= 1 && x <= 3))
    collected

let test_weighted_sample () =
  let weights = [| 0.1; 0.1; 0.8 |] in
  (* Heavy bias toward third element *)
  let rng = Rune.Rng.key 42 in
  let dataset =
    from_list [ "a"; "b"; "c" ] |> weighted_sample ~rng ~weights 10
  in
  let collected = collect_dataset dataset in
  equal ~msg:"sampled 10" int 10 (List.length collected);
  (* Most samples should be "c" due to weight *)
  let c_count = List.filter (( = ) "c") collected |> List.length in
  equal ~msg:"mostly c" bool true (c_count >= 5)

(* ───── Test Caching And Prefetching ───── *)

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
  equal ~msg:"first pass" (list int) [ 1; 2; 3 ] collected1;

  (* Second pass - test that creating a new cache on same data works *)
  (* Since we can't reset from outside, test with a fresh dataset *)
  let dataset2 = make_dataset () in
  let cached2 = cache dataset2 in
  let collected2 = collect_dataset cached2 in
  equal ~msg:"second pass" (list int) [ 1; 2; 3 ] collected2

let test_prefetch () =
  let dataset = from_list [ 1; 2; 3; 4; 5 ] |> prefetch ~buffer_size:2 in
  let collected = collect_dataset dataset in
  equal ~msg:"prefetched" (list int) [ 1; 2; 3; 4; 5 ] collected

(* ───── Test Parallel Processing ───── *)

let test_parallel_map () =
  let dataset =
    from_list [ 1; 2; 3; 4; 5 ] |> parallel_map ~num_workers:2 (fun x -> x * x)
  in
  let collected = collect_dataset dataset in
  let sorted = List.sort compare collected in
  (* Order may vary *)
  equal ~msg:"squared values" (list int) [ 1; 4; 9; 16; 25 ] sorted

let test_parallel_interleave () =
  let dataset =
    from_list [ 1; 2; 3 ]
    |> parallel_interleave ~num_workers:2 ~block_length:1 (fun x ->
        from_list [ x; x * 10 ])
  in
  let collected = collect_dataset dataset in
  equal ~msg:"correct number of elements" int 6 (List.length collected);
  (* Check all expected values are present *)
  let sorted = List.sort compare collected in
  equal ~msg:"all values present" (list int) [ 1; 2; 3; 10; 20; 30 ] sorted

(* ───── Test Iteration Functions ───── *)

let test_iter () =
  let sum = ref 0 in
  from_list [ 1; 2; 3; 4; 5 ] |> iter (fun x -> sum := !sum + x);
  equal ~msg:"sum via iter" int 15 !sum

let test_fold () =
  let result = from_list [ 1; 2; 3; 4 ] |> fold ( + ) 0 in
  equal ~msg:"fold sum" int 10 result

let test_to_seq () =
  let dataset = from_list [ 1; 2; 3 ] in
  let seq = to_seq dataset in
  let collected = List.of_seq seq in
  equal ~msg:"to_seq" (list int) [ 1; 2; 3 ] collected

let test_to_list () =
  let dataset = from_list [ 1; 2; 3 ] |> map (fun x -> x * 2) in
  let lst = to_list dataset in
  equal ~msg:"to_list" (list int) [ 2; 4; 6 ] lst

let test_to_array () =
  let dataset = from_list [ "a"; "b"; "c" ] in
  let arr = to_array dataset in
  assert_equal_string_array [| "a"; "b"; "c" |] arr

(* ───── Test Dataset Information ───── *)

let test_cardinality_finite () =
  let dataset = from_list [ 1; 2; 3 ] in
  match cardinality dataset with
  | Finite n -> equal ~msg:"finite cardinality" int 3 n
  | _ -> fail "Expected finite cardinality"

let test_cardinality_unknown () =
  let dataset = from_list [ 1; 2; 3 ] |> filter (fun x -> x > 1) in
  match cardinality dataset with
  | Unknown -> ()
  | _ -> fail "Expected unknown cardinality"

let test_cardinality_infinite () =
  let dataset = from_list [ 1; 2 ] |> repeat in
  match cardinality dataset with
  | Infinite -> () (* repeat without count returns Infinite *)
  | _ -> fail "Expected infinite cardinality for infinite repeat"

let test_element_spec () =
  let dataset = from_list [ "hello"; "world" ] in
  match element_spec dataset with
  | Unknown -> () (* Default spec for from_list *)
  | _ -> fail "Expected unknown element spec"

(* ───── Test Pipelines ───── *)

let test_text_classification_pipeline () =
  let dataset =
    from_list [ "hello world"; "foo bar baz" ]
    |> text_classification_pipeline ~batch_size:2
  in
  let collected = collect_n 1 dataset in
  equal ~msg:"got batch" int 1 (List.length collected);
  match collected with
  | [ batch ] ->
      (* Check that we got a tensor *)
      equal ~msg:"got tensor" bool true (Rune.shape batch |> Array.length > 0)
  | _ -> fail "Expected one batch"

let test_language_model_pipeline () =
  let dataset =
    from_list [ "one two three four" ]
    |> language_model_pipeline ~sequence_length:4 ~batch_size:1
  in
  let collected = collect_n 1 dataset in
  match collected with
  | [ (input, target) ] ->
      equal ~msg:"input/target tensors" bool true
        (Rune.shape input |> Array.length > 0
        && Rune.shape target |> Array.length > 0)
  | _ -> fail "Expected batch of input/target pairs"

(* ───── Test Edge Cases ───── *)

let test_empty_dataset () =
  let dataset = from_list [] in
  let collected = collect_dataset dataset in
  equal ~msg:"empty dataset" (list int) [] collected

let test_single_element () =
  let dataset = from_list [ 42 ] in
  let collected = collect_dataset dataset in
  equal ~msg:"single element" (list int) [ 42 ] collected

let test_chain_many_operations () =
  let dataset =
    from_list [ 1; 2; 3; 4; 5; 6; 7; 8; 9; 10 ]
    |> filter (fun x -> x mod 2 = 0) (* [2; 4; 6; 8; 10] *)
    |> map (fun x -> x * 2) (* [4; 8; 12; 16; 20] *)
    |> take 3 (* [4; 8; 12] *)
    |> batch_map 2 Fun.id (* [[4; 8], [12]] *)
  in
  let collected = collect_dataset dataset in
  equal ~msg:"number of batches" int 2 (List.length collected);
  assert_equal_int_array [| 4; 8 |] (List.nth collected 0);
  assert_equal_int_array [| 12 |] (List.nth collected 1)

(* ───── Test Suite ───── *)

let () =
  run "Dataset"
    [
      group "creation"
        [
          test "from_array" test_from_array;
          test "from_list" test_from_list;
          test "from_seq" test_from_seq;
        ];
      group "text_files"
        [
          test "from_text_file" test_from_text_file;
          test "from_text_file_utf8" test_from_text_file_utf8;
          test "from_text_file_latin1" test_from_text_file_latin1;
          test "from_text_file_large_lines" test_from_text_file_large_lines;
          test "from_text_file_reset" test_from_text_file_reset;
          test "from_text_file_reset_mid_stream"
            test_from_text_file_reset_mid_stream;
          test "from_text_files" test_from_text_files;
          test "from_jsonl" test_from_jsonl;
          test "from_jsonl_custom_field" test_from_jsonl_custom_field;
          test "from_csv" test_from_csv;
          test "from_csv_custom_column" test_from_csv_custom_column;
          test "from_csv_no_header" test_from_csv_no_header;
          test "from_file" test_from_file;
          test "from_csv_with_labels" test_from_csv_with_labels;
          test "from_csv_with_labels_no_header"
            test_from_csv_with_labels_no_header;
          test "from_csv_with_labels_custom_columns"
            test_from_csv_with_labels_custom_columns;
          test "from_csv_with_labels_custom_separator"
            test_from_csv_with_labels_custom_separator;
          test "from_csv_with_labels_malformed_rows"
            test_from_csv_with_labels_malformed_rows;
        ];
      group "transformations"
        [
          test "map" test_map;
          test "filter" test_filter;
          test "flat_map" test_flat_map;
          test "zip" test_zip;
          test "concatenate" test_concatenate;
          test "interleave" test_interleave;
          test "enumerate" test_enumerate;
        ];
      group "text_processing"
        [
          test "normalize" test_normalize;
          test "tokenize_whitespace" test_tokenize_whitespace;
          test "tokenize_with_special_tokens" test_tokenize_with_special_tokens;
          test "tokenize_truncation" test_tokenize_truncation;
          test "tokenize_padding" test_tokenize_padding;
        ];
      group "batching"
        [
          test "batch_basic" test_batch_basic;
          test "batch_drop_remainder" test_batch_drop_remainder;
          test "batch_map_combiner" test_batch_map_combiner;
          test "bucket_by_length" test_bucket_by_length;
        ];
      group "iteration_control"
        [
          test "take" test_take;
          test "skip" test_skip;
          test "repeat_finite" test_repeat_finite;
          test "repeat_infinite" test_repeat_infinite;
          test "window" test_window;
          test "window_with_shift" test_window_with_shift;
          test "window_drop_remainder" test_window_drop_remainder;
        ];
      group "shuffling_sampling"
        [
          test "shuffle" test_shuffle;
          test "shuffle_deterministic" test_shuffle_deterministic;
          test "sample_without_replacement" test_sample_without_replacement;
          test "sample_with_replacement" test_sample_with_replacement;
          test "weighted_sample" test_weighted_sample;
        ];
      group "caching_prefetching"
        [
          test "cache_in_memory" test_cache_in_memory;
          test "prefetch" test_prefetch;
        ];
      group "parallel"
        [
          test "parallel_map" test_parallel_map;
          test "parallel_interleave" test_parallel_interleave;
        ];
      group "iteration"
        [
          test "iter" test_iter;
          test "fold" test_fold;
          test "to_seq" test_to_seq;
          test "to_list" test_to_list;
          test "to_array" test_to_array;
        ];
      group "dataset_info"
        [
          test "cardinality_finite" test_cardinality_finite;
          test "cardinality_unknown" test_cardinality_unknown;
          test "cardinality_infinite" test_cardinality_infinite;
          test "element_spec" test_element_spec;
        ];
      group "pipelines"
        [
          test "text_classification_pipeline" test_text_classification_pipeline;
          test "language_model_pipeline" test_language_model_pipeline;
        ];
      group "edge_cases"
        [
          test "empty_dataset" test_empty_dataset;
          test "single_element" test_single_element;
          test "chain_many_operations" test_chain_many_operations;
        ];
    ]
