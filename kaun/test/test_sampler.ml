open Alcotest

module S = Sampler

(* Test helpers *)
let _float_array_testable =
  testable
    (fun fmt arr ->
      Format.fprintf fmt "[%s]"
        (String.concat "; " (Array.to_list (Array.map string_of_float arr))))
    (fun a b ->
      Array.length a = Array.length b
      && Array.for_all2 (fun x y -> abs_float (x -. y) < 1e-6) a b)

let _int_array_testable =
  testable
    (fun fmt arr ->
      Format.fprintf fmt "[%s]"
        (String.concat "; " (Array.to_list (Array.map string_of_int arr))))
    ( = )

(* Mock tokenizer for testing *)
let mock_tokenizer text =
  (* Simple character-based tokenizer for testing *)
  let chars = String.to_seq text |> List.of_seq in
  Array.of_list (List.map Char.code chars)

(* Mock decoder for testing *)
let mock_decoder tokens =
  let chars = Array.map (fun i -> Char.chr (i mod 256)) tokens in
  String.init (Array.length chars) (fun i -> chars.(i))

(* Test greedy sampling *)
let test_greedy () =
  let logits = [| 1.0; 3.0; 2.0; 0.5 |] in
  let result = S.greedy logits in
  check int "greedy should select max logit index" 1 result;
  
  let logits2 = [| -1.0; -2.0; -0.5; -3.0 |] in
  let result2 = S.greedy logits2 in
  check int "greedy should select max logit index (negative)" 2 result2

(* Test softmax conversion *)
let test_softmax () =
  (* Internal test - we need to expose softmax or test indirectly *)
  let logits = [| 0.0; 0.0; 0.0 |] in
  (* With temperature=1.0 and uniform logits, should get ~uniform distribution *)
  let token = S.sample_token ~temperature:1.0 ~seed:42 logits in
  check bool "sample_token should return valid index" true (token >= 0 && token < 3)

(* Test temperature scaling *)
let test_temperature () =
  let logits = [| 1.0; 2.0; 3.0 |] in
  
  (* Very low temperature should make it more deterministic *)
  let token_low = S.sample_token ~temperature:0.01 ~seed:42 logits in
  check int "low temperature should be more deterministic" 2 token_low;
  
  (* High temperature makes distribution more uniform *)
  let results = ref [] in
  for seed = 0 to 100 do
    let token = S.sample_token ~temperature:10.0 ~seed logits in
    results := token :: !results
  done;
  (* With high temperature, we should see variety in selections *)
  let unique = List.sort_uniq compare !results in
  check bool "high temperature should produce variety" true (List.length unique > 1)

(* Test top-k filtering *)
let test_top_k () =
  let logits = [| 1.0; 2.0; 3.0; 4.0; 5.0 |] in
  
  (* With top_k=2, only indices 3 and 4 should be possible *)
  let results = ref [] in
  for seed = 0 to 50 do
    let token = S.sample_token ~top_k:2 ~seed logits in
    results := token :: !results
  done;
  
  let unique = List.sort_uniq compare !results in
  List.iter (fun idx ->
    check bool (Printf.sprintf "top-k=2 should only select from top 2 (got %d)" idx)
      true (idx = 3 || idx = 4)
  ) unique;
  
  (* Verify we get both values with enough samples *)
  check bool "should sample both top values" true 
    (List.mem 3 !results && List.mem 4 !results)

(* Test top-p (nucleus) filtering *)
let test_top_p () =
  (* Create logits where top 2 account for >90% probability *)
  let logits = [| 0.0; 0.0; 10.0; 9.0; 0.0 |] in
  
  let results = ref [] in
  for seed = 0 to 50 do
    let token = S.sample_token ~top_p:0.9 ~seed logits in
    results := token :: !results
  done;
  
  (* Should mostly/only see indices 2 and 3 *)
  let unique = List.sort_uniq compare !results in
  List.iter (fun idx ->
    check bool (Printf.sprintf "top-p=0.9 should select high prob tokens (got %d)" idx)
      true (idx = 2 || idx = 3)
  ) unique

(* Test combined sampling parameters *)
let test_combined_sampling () =
  let logits = [| 1.0; 2.0; 3.0; 4.0; 5.0 |] in
  
  (* Combine temperature with top-k *)
  let token = S.sample_token ~temperature:0.5 ~top_k:3 ~seed:42 logits in
  check bool "combined sampling should return valid index" true
    (token >= 2 && token <= 4);
  
  (* Combine all parameters *)
  let token2 = S.sample_token ~temperature:1.0 ~top_k:4 ~top_p:0.95 ~seed:42 logits in
  check bool "all parameters combined should work" true
    (token2 >= 0 && token2 < 5)

(* Test generation with mock logits function *)
let test_generate () =
  (* Simple logits function that always favors next sequential token *)
  let logits_fn prev_token =
    let logits = Array.make 256 0.0 in
    logits.((prev_token + 1) mod 256) <- 10.0;  (* Strongly favor next token *)
    logits
  in
  
  let tokens = S.generate ~max_tokens:5 ~temperature:0.01 ~seed:42 
    ~logits_fn ~tokenizer:mock_tokenizer () in
  
  check int "should generate requested number of tokens" 5 (Array.length tokens);
  
  (* With very low temperature, should be deterministic *)
  for i = 1 to Array.length tokens - 1 do
    let expected = (tokens.(i-1) + 1) mod 256 in
    check int (Printf.sprintf "token %d should be sequential" i) 
      expected tokens.(i)
  done

(* Test generation with starting text *)
let test_generate_with_start () =
  let logits_fn _prev_token =
    let logits = Array.make 256 (-10.0) in
    logits.(65) <- 10.0;  (* Strongly favor 'A' *)
    logits
  in
  
  let tokens = S.generate ~max_tokens:3 ~temperature:0.01 ~seed:42
    ~start:"Hi" ~logits_fn ~tokenizer:mock_tokenizer () in
  
  (* Should start with "Hi" tokens plus 3 generated *)
  check int "should include start tokens plus generated" 5 (Array.length tokens);
  check int "first token should be 'H'" 72 tokens.(0);
  check int "second token should be 'i'" 105 tokens.(1);
  (* Generated tokens should all be 'A' (65) *)
  for i = 2 to 4 do
    check int (Printf.sprintf "generated token %d should be 'A'" (i-2)) 65 tokens.(i)
  done

(* Test generate_text function *)
let test_generate_text () =
  let logits_fn _prev_token =
    let logits = Array.make 256 (-10.0) in
    (* Create a pattern: always generate 'A' after anything *)
    logits.(65) <- 10.0;
    logits
  in
  
  let text = S.generate_text ~max_tokens:3 ~temperature:0.01 ~seed:42
    ~start:"Test" ~logits_fn ~tokenizer:mock_tokenizer ~decoder:mock_decoder () in
  
  (* Should be "Test" + "AAA" *)
  check string "generated text should match expected" "TestAAA" text

(* Test edge cases *)
let test_edge_cases () =
  (* Empty logits array should not crash *)
  let empty_logits = [||] in
  check_raises "empty logits should raise exception"
    (Invalid_argument "index out of bounds")
    (fun () -> ignore (S.greedy empty_logits));
  
  (* Single logit *)
  let single = [| 1.0 |] in
  let token = S.sample_token ~seed:42 single in
  check int "single logit should return 0" 0 token;
  
  (* All equal logits *)
  let equal = [| 1.0; 1.0; 1.0; 1.0 |] in
  let results = ref [] in
  for seed = 0 to 100 do
    results := S.sample_token ~seed equal :: !results
  done;
  let unique = List.sort_uniq compare !results in
  (* With equal probabilities, should eventually sample all indices *)
  check bool "equal logits should allow all indices" true
    (List.length unique > 1);
  
  (* Very large logits differences *)
  let extreme = [| -1000.0; 1000.0; -1000.0 |] in
  let token = S.sample_token ~temperature:1.0 ~seed:42 extreme in
  check int "extreme logits should select max" 1 token

(* Test determinism with seeds *)
let test_determinism () =
  let logits = [| 1.0; 2.0; 3.0; 4.0 |] in
  
  (* Same seed should give same result *)
  let token1 = S.sample_token ~temperature:1.0 ~seed:123 logits in
  let token2 = S.sample_token ~temperature:1.0 ~seed:123 logits in
  check int "same seed should give same result" token1 token2;
  
  (* Different seeds should (usually) give different results *)
  let results = ref [] in
  for seed = 0 to 20 do
    results := S.sample_token ~temperature:1.0 ~seed logits :: !results
  done;
  let unique = List.sort_uniq compare !results in
  check bool "different seeds should give variety" true
    (List.length unique > 1)

(* Test top-k with k larger than array *)
let test_top_k_edge_cases () =
  let logits = [| 1.0; 2.0; 3.0 |] in
  
  (* top_k larger than array should work like no filtering *)
  let token = S.sample_token ~top_k:10 ~seed:42 logits in
  check bool "top_k > array length should work" true
    (token >= 0 && token < 3);
  
  (* top_k = 1 should be deterministic *)
  let results = ref [] in
  for seed = 0 to 10 do
    results := S.sample_token ~top_k:1 ~seed logits :: !results
  done;
  let unique = List.sort_uniq compare !results in
  check int "top_k=1 should be deterministic" 1 (List.length unique);
  check int "top_k=1 should select max" 2 (List.hd unique)

(* Test top-p edge cases *)
let test_top_p_edge_cases () =
  let logits = [| 1.0; 2.0; 3.0 |] in
  
  (* top_p = 1.0 should include everything *)
  let token = S.sample_token ~top_p:1.0 ~seed:42 logits in
  check bool "top_p=1.0 should work" true (token >= 0 && token < 3);
  
  (* Very small top_p should be restrictive *)
  let results = ref [] in
  for seed = 0 to 20 do
    results := S.sample_token ~top_p:0.1 ~seed logits :: !results
  done;
  (* Should mostly/only get the highest probability token *)
  let most_common = 
    List.fold_left (fun acc x ->
      let count_x = List.filter ((=) x) !results |> List.length in
      let count_acc = List.filter ((=) acc) !results |> List.length in
      if count_x > count_acc then x else acc
    ) (List.hd !results) !results
  in
  check int "small top_p should favor highest prob" 2 most_common

(* Test suite *)
let () =
  run "Sampler"
    [
      ( "basic",
        [
          test_case "greedy" `Quick test_greedy;
          test_case "softmax" `Quick test_softmax;
          test_case "temperature" `Quick test_temperature;
        ] );
      ( "filtering",
        [
          test_case "top_k" `Quick test_top_k;
          test_case "top_p" `Quick test_top_p;
          test_case "combined_sampling" `Quick test_combined_sampling;
        ] );
      ( "generation",
        [
          test_case "generate" `Quick test_generate;
          test_case "generate_with_start" `Quick test_generate_with_start;
          test_case "generate_text" `Quick test_generate_text;
        ] );
      ( "edge_cases",
        [
          test_case "edge_cases" `Quick test_edge_cases;
          test_case "top_k_edge_cases" `Quick test_top_k_edge_cases;
          test_case "top_p_edge_cases" `Quick test_top_p_edge_cases;
        ] );
      ( "determinism",
        [
          test_case "determinism" `Quick test_determinism;
        ] );
    ]