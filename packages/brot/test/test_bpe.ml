(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The BPE model: its caches, then the constructor. Expectations that name
   Hugging Face were read off tokenizers 0.23.1 with the same vocabulary. *)

open Windtrap
open Brot

let tokens t text = Array.to_list (Encoding.tokens (encode t text))
let numbered tokens = List.mapi (fun i token -> (token, i)) tokens

(* The byte tokens ["<0x20>"] to ["<0x7F>"], numbered from [first]. *)
let printable_bytes ~first =
  List.init 96 (fun i -> (Printf.sprintf "<0x%02X>" (0x20 + i), first + i))

(* The length of the character at [i]: a byte that opens no character is one. *)
let char_length text i =
  let d = String.get_utf_8_uchar text i in
  if Uchar.utf_decode_is_valid d then Uchar.utf_decode_length d else 1

let characters text =
  let rec go i acc =
    if i >= String.length text then List.rev acc
    else
      let n = char_length text i in
      go (i + n) (String.sub text i n :: acc)
  in
  go 0 []

let load path =
  match from_file path with Ok t -> t | Error msg -> failf "%s: %s" path msg

let quoted = Gen.with_pp (fun ppf s -> Format.fprintf ppf "%S" s)
let words list = Gen.of_list ~pp:(fun ppf s -> Format.fprintf ppf "%S" s) list
let capacities = Gen.of_list ~pp:Format.pp_print_int [ 1; 4; 16; 4096; 262144 ]

(* A tokenizer file's model: its vocabulary, its merges and their ranks. *)

type file_model = {
  ids : (string, int) Hashtbl.t;
  merges : (string * string) list;
  ranks : (string * string, int) Hashtbl.t;
}

let member name = function
  | Jsont.Object (members, _) -> (
      match Jsont.Json.find_mem name members with
      | Some (_, value) -> value
      | None -> failf "no member %s" name)
  | _ -> failf "%s: not an object" name

let file_model path =
  let json =
    match Jsont_bytesrw.decode_string Jsont.json (Fixture.read path) with
    | Ok json -> json
    | Error e -> failf "%s: %s" path e
  in
  let model = member "model" json in
  let ids = Hashtbl.create 65536 in
  (match member "vocab" model with
  | Jsont.Object (members, _) ->
      List.iter
        (fun ((token, _), id) ->
          match id with
          | Jsont.Number (id, _) -> Hashtbl.replace ids token (int_of_float id)
          | _ -> failf "vocab %s: not a number" token)
        members
  | _ -> failf "vocab: not an object");
  let merge = function
    | Jsont.String (m, _) -> (
        match String.index_opt m ' ' with
        | Some i ->
            (String.sub m 0 i, String.sub m (i + 1) (String.length m - i - 1))
        | None -> failf "merge %S" m)
    | Jsont.Array ([ Jsont.String (a, _); Jsont.String (b, _) ], _) -> (a, b)
    | _ -> failf "merge not a pair"
  in
  let merges =
    match member "merges" model with
    | Jsont.Array (merges, _) -> List.map merge merges
    | _ -> failf "merges: not an array"
  in
  let ranks = Hashtbl.create 65536 in
  List.iteri
    (fun rank pair ->
      if not (Hashtbl.mem ranks pair) then Hashtbl.add ranks pair rank)
    merges;
  { ids; merges; ranks }

(* [symbols] merged as [merges] says: the adjacent pair whose rule comes first
   is merged, the leftmost of equal pairs first, until no rule applies. [rank]
   gives a pair's place among the rules, [join] the symbol a merge makes. *)
let rec merge_by ?(join = ( ^ )) rank symbols =
  let rec best i acc = function
    | a :: (b :: _ as rest) ->
        let acc =
          match (rank (a, b), acc) with
          | Some r, Some (r', _) when r >= r' -> acc
          | Some r, _ -> Some (r, i)
          | None, _ -> acc
        in
        best (i + 1) acc rest
    | _ -> acc
  in
  match best 0 None symbols with
  | None -> symbols
  | Some (_, at) ->
      let rec merge i = function
        | a :: b :: rest when i = at -> join a b :: rest
        | a :: rest -> a :: merge (i + 1) rest
        | [] -> []
      in
      merge_by ~join rank (merge 0 symbols)

(* The GPT-2 byte-level alphabet: the character each byte is spelled as. *)
let byte_alphabet =
  let direct b = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174 in
  let next = ref 0 in
  Array.init 256 (fun b ->
      let code =
        if direct b then b
        else begin
          let c = 256 + !next in
          incr next;
          c
        end
      in
      let buf = Buffer.create 2 in
      Buffer.add_utf_8_uchar buf (Uchar.of_int code);
      Buffer.contents buf)

(* t *)

(* Eight letters, every two-letter merge, one four-letter merge per letter, and
   a three-letter entry per letter that no merge builds. *)
let collide_vocab, collide_merges =
  let letters = List.init 8 (fun i -> String.make 1 (Char.chr (97 + i))) in
  let pairs =
    List.concat_map (fun a -> List.map (fun b -> (a, b)) letters) letters
  in
  let quads = List.map (fun a -> (a ^ a, a ^ a)) letters in
  let joined = List.map (fun (a, b) -> a ^ b) in
  let vocab =
    letters @ joined pairs @ joined quads
    @ List.map (fun a -> a ^ a ^ a) letters
  in
  (numbered vocab, pairs @ quads)

let collide ?(ignore_merges = false) ?dropout cache_capacity =
  bpe ~vocab:collide_vocab ~merges:collide_merges ~cache_capacity ~ignore_merges
    ?dropout ()

(* Words over the eight letters of every length to 40, past the 15 bytes a cache
   key holds or of more than four tokens. *)
let collide_words =
  let state = Random.State.make [| 20260817 |] in
  List.init 400 (fun i ->
      String.init
        (1 + (i mod 40))
        (fun _ -> Char.chr (97 + Random.State.int state 8)))

let long_words =
  [ "abcdefgh"; "abcdefghabcdefg"; "abcdefghabcdefgh";
    "abcdefghabcdefghabcdefghabcdefgh"; "aaaabbbbccccddddeeeeffffgggghhhh";
    String.make 200 'a'; String.init 60 (fun i -> Char.chr (97 + (i mod 8))) ]
[@@ocamlformat "disable"]

let recurring = [ "abcd"; "bcda"; "cdab"; "dabc"; "aabb"; "bbaa"; "aaa" ]

(* A program makes a tokenizer whose caches hold one set (capacity 1), two sets
   each the other's neighbour (4), or more, then encodes words that share them:
   most often a few that recur, so that a set evicts and reads back. The
   reference is the same configuration without a cache, and the words it has
   met. *)
let bpe_cache =
  let tok = abstract "t" in
  let encode name pool =
    command name
      (tok ^-> words pool @-> returns (array int))
      (fun (uncached, met) word ->
        cover "a word encoded again after two others"
          (match !met with
          | _ :: _ :: older -> List.mem word older
          | _ -> false);
        met := word :: !met;
        let ids = encode_ids uncached word in
        cover "more than four tokens" (Array.length ids > 4);
        ids)
      (fun t word -> encode_ids t word)
  in
  stateful ~count:40 ~steps:30
    "gives the ids an uncached BPE tokenizer gives, whatever it encoded before"
    [
      command "bpe"
        (Gen.pair capacities Gen.bool @-> makes tok)
        (fun (capacity, ignore_merges) ->
          cover "a cache of one set" (capacity = 1);
          cover "a cache of two sets" (capacity = 4);
          (collide ~ignore_merges 0, ref []))
        (fun (capacity, ignore_merges) -> collide ~ignore_merges capacity);
      encode "encode_ids, a word that recurs" recurring;
      encode "encode_ids, another word that recurs" recurring;
      encode "encode_ids, a drawn word"
        (collide_words @ List.map fst collide_vocab);
      encode "encode_ids, a long word" long_words;
    ]

(* Two domains encoding at once, on four words that share the one set of the
   smallest cache, so that each domain's encodings evict and read back what the
   other's store. A cache slot written by two encodings at once (a06fd975f8) is
   ruled out today by each encoding claiming a state of its own; this guards
   that claim. The prefix's encodings leave a state in the pool for the branches
   to race for. *)
let shared_set_words =
  List.init 4 (fun k ->
      Printf.sprintf "a%ca%c" (Char.chr (97 + k)) (Char.chr (98 + k)))

let shared_set_bpe cache_capacity =
  let letter i = String.make 1 (Char.chr (97 + i)) in
  let vocab =
    List.init 26 (fun i -> (letter i, i))
    @ List.init 26 (fun i -> ("a" ^ letter i, 26 + i))
  in
  bpe ~vocab
    ~merges:(List.init 26 (fun i -> ("a", letter i)))
    ~cache_capacity ()

(* The reference is one uncached tokenizer, whose ids are a function of the text
   alone. *)
let uncached_shared_set = lazy (shared_set_bpe 0)

let bpe_domains =
  let tok = abstract "t" in
  (* [encode_ids] runs only in a case whose prefix drew [bpe]: at 4 cases all of
     them miss it about once in 300 runs, at 16 none did in 300. *)
  stateful ~domains:2 ~count:16 ~steps:6
    "encodes BPE text on two domains as on one"
    [
      command "bpe"
        (Gen.unit @-> makes tok)
        (fun () -> Lazy.force uncached_shared_set)
        (fun () -> shared_set_bpe 1);
      command "encode_ids"
        (tok ^-> words shared_set_words @-> returns (array int))
        (fun t word -> encode_ids t word)
        (fun t word -> encode_ids t word);
    ]

(* Every word of up to four of the eight letters, 4680 pretokens, through a
   full-size table in three orders: stored, read back from either way of its set
   or from the set beside it, moved there and traded back. *)
let test_full_table () =
  let uncached = collide 0 in
  let cached = collide 262144 in
  let words =
    let acc = ref [] in
    let rec go prefix len =
      if len > 0 then
        for c = 0 to 7 do
          let w = prefix ^ String.make 1 (Char.chr (97 + c)) in
          acc := w :: !acc;
          go w (len - 1)
        done
    in
    go "" 4;
    Array.of_list (List.rev !acc)
  in
  let n = Array.length words in
  let expected = Array.map (fun w -> encode_ids uncached w) words in
  let check pass i =
    equal
      ~msg:(Printf.sprintf "pass %d, %S" pass words.(i))
      (array int) expected.(i)
      (encode_ids cached words.(i))
  in
  for i = 0 to n - 1 do
    check 1 i
  done;
  for i = n - 1 downto 0 do
    check 2 i
  done;
  (* A coprime stride visits every word once more, mixing hot and cold. *)
  let j = ref 0 in
  for _ = 0 to n - 1 do
    check 3 !j;
    j := (!j + 2311) mod n
  done

(* gpt2's vocabulary and merges, under each capacity, on real text. *)
let test_real_text () =
  Fixture.with_tokenizer_file "gpt2" @@ fun path ->
  let model = file_model path in
  let vocab =
    Hashtbl.fold (fun token id acc -> (token, id) :: acc) model.ids []
  in
  let pre = pre_tokenizer (load path) in
  let make cache_capacity =
    bpe ?pre ~vocab ~merges:model.merges ~cache_capacity ()
  in
  let reference = make 0 in
  let corpora =
    List.map
      (fun name ->
        let text = Fixture.read (Printf.sprintf "support/parity/%s.txt" name) in
        (name, text, encode_ids reference text))
      [ "sample"; "edge_cases" ]
  in
  List.iter
    (fun capacity ->
      let cached = make capacity in
      for pass = 1 to 2 do
        List.iter
          (fun (name, text, expected) ->
            equal
              ~msg:(Printf.sprintf "%d entries, pass %d, %s" capacity pass name)
              (array int) expected (encode_ids cached text))
          corpora
      done)
    [ 1; 2; 4; 64; 262144 ]

(* bpe *)

(* Drawn merge rules over five characters, a stray byte among them, under a
   prefix and a suffix each absent, empty or set. Before any merge a word's
   characters take the prefix, all but the first, and the suffix, the last. A
   rule joins two symbols, each an affixed form of a character or what a rule
   before it makes, and the word is spelled from the same symbols, up to 40 of
   them, past the 32 a word merges in place. The vocabulary holds every symbol,
   so it holds entries no merge of the word builds. Hugging Face's tokens for
   the affixed examples were read off tokenizers 0.23.1. *)
let alphabet = [ "a"; "b"; "c"; "\u{e9}"; "\xff" ]

type merge_case = {
  prefix : string option;
  suffix : string option;
  merges : (string * string) list;
  text : string;
}

(* A merge drops the prefix of its right symbol. *)
let join prefix a b =
  let n = String.length prefix in
  if n > 0 && String.length b > n && String.starts_with ~prefix b then
    a ^ String.sub b n (String.length b - n)
  else a ^ b

let affix ~prefix ~suffix chars =
  let last = List.length chars - 1 in
  List.mapi
    (fun i c ->
      (if i > 0 then prefix else "") ^ c ^ if i = last then suffix else "")
    chars

let forms ~prefix ~suffix c =
  List.sort_uniq compare [ c; prefix ^ c; c ^ suffix; prefix ^ c ^ suffix ]

let merge_vocab { prefix; suffix; merges; _ } =
  let prefix = Option.value ~default:"" prefix in
  let suffix = Option.value ~default:"" suffix in
  List.concat_map (forms ~prefix ~suffix) alphabet
  @ List.map (fun (a, b) -> join prefix a b) merges
  |> List.sort_uniq compare |> numbered

let merge_case =
  let open Gen in
  let affix_of value =
    frequency
      [
        (1, constant None); (1, constant (Some "")); (2, constant (Some value));
      ]
  in
  let* prefix, suffix = pair (affix_of "##") (affix_of "</w>") in
  let p = Option.value ~default:"" prefix in
  let s = Option.value ~default:"" suffix in
  let base =
    List.concat_map
      (fun c -> List.map (fun f -> (f, c)) (forms ~prefix:p ~suffix:s c))
      alphabet
  in
  (* Each symbol with its spelling in the text. *)
  let rules picks =
    List.fold_left
      (fun (rules, made) (i, j) ->
        let symbols = Array.of_list (base @ made) in
        let pick k = symbols.(k mod Array.length symbols) in
        let (a, sa), (b, sb) = (pick i, pick j) in
        if List.mem (a, b) rules then (rules, made)
        else (rules @ [ (a, b) ], made @ [ (join p a b, sa ^ sb) ]))
      ([], []) picks
  in
  let* merges, made = map rules (list ~size:(int_range 0 16) (pair nat nat)) in
  let+ text =
    map (String.concat "")
      (list ~size:(int_range 0 40) (of_list (alphabet @ List.map snd made)))
  in
  { prefix; suffix; merges; text }

let pp_merge_case ppf { prefix; suffix; merges; text } =
  let affix = function None -> "none" | Some a -> Printf.sprintf "%S" a in
  let rule (a, b) = Printf.sprintf "(%S, %S)" a b in
  Format.fprintf ppf "prefix %s, suffix %s,@ merges [%s]@ text %S"
    (affix prefix) (affix suffix)
    (String.concat "; " (List.map rule merges))
    text

let merge_examples =
  let rows ?prefix ?suffix merges texts =
    List.map (fun text -> { prefix; suffix; merges; text }) texts
  in
  let a n = String.make n 'a' in
  let doubling = [ ("a", "a"); ("aa", "aa"); ("aaaa", "aaaa"); ("a", "b") ] in
  List.concat
    [
      rows doubling [ a 8; a 32; a 64; a 22 ^ "b" ];
      rows [ ("a", "a") ] [ "aaa" ];
      rows [ ("b", "c"); ("a", "b") ] [ "abc" ];
      (* HF: "a" -> a</w>, "ba" -> b, a</w>, "ab" -> ab</w>, "aab" -> a,
         ab</w>. *)
      rows ~suffix:"</w>" [ ("a", "b</w>") ] [ "a"; "ba"; "ab"; "aab"; "aba" ];
      rows ~suffix:"</w>" [] [ "\u{e9}"; "a\u{e9}"; "\u{e9}a" ];
      (* HF: "ab" -> a, b</w> and "aba" -> ab, a</w>. *)
      rows ~suffix:"</w>" [ ("a", "b") ] [ "ab"; "aba" ];
      rows ~prefix:"##" ~suffix:"</w>" [] [ "a"; "ab"; "abc" ];
      rows ~prefix:"##" [] [ "a"; "abc" ];
      rows ~prefix:"" ~suffix:"" [ ("a", "b") ] [ "ab"; "aab" ];
    ]

let merge_law =
  prop ~count:50
    "merges the pair of the first rule, the leftmost first, affixes in place"
    ~examples:merge_examples (Gen.with_pp pp_merge_case merge_case)
    (fun ({ prefix; suffix; merges; text } as case) ->
      let p = Option.value ~default:"" prefix in
      let s = Option.value ~default:"" suffix in
      let rank pair = List.find_index (( = ) pair) merges in
      let expected =
        merge_by ~join:(join p) rank
          (affix ~prefix:p ~suffix:s (characters text))
      in
      let made = List.map (fun (a, b) -> join p a b) merges in
      let deep =
        List.filter_map
          (fun (a, b) ->
            if List.mem a made || List.mem b made then Some (join p a b)
            else None)
          merges
      in
      cover "a prefix" (prefix = Some "##");
      cover "a suffix" (suffix = Some "</w>");
      cover "an empty affix" (prefix = Some "" || suffix = Some "");
      cover "a word past 32 characters" (List.length (characters text) > 32);
      cover "a token merged from a merged symbol"
        (List.exists (fun token -> List.mem token deep) expected);
      let t =
        bpe ~vocab:(merge_vocab case) ~merges ?continuing_subword_prefix:prefix
          ?end_of_word_suffix:suffix ()
      in
      equal ~msg:(Printf.sprintf "%S" text) (list string) expected
        (tokens t text))

(* phi4's file: the pieces of its cl100k pattern, each spelled in bytes and
   merged by rank. The pattern sits in a group, so that the regular expression
   engine cuts it rather than a walker of the tokenizer's own. *)
let cl100k =
  {re|(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+|re}

let phi4_reference model =
  let split =
    Pre_tokenizer.split_regex
      ~pattern:("(?:" ^ cl100k ^ ")")
      ~behavior:`Isolated ()
  in
  let id symbol =
    match Hashtbl.find_opt model.ids symbol with
    | Some id -> id
    | None -> failf "no id for %S" symbol
  in
  fun text ->
    Pre_tokenizer.pre_tokenize split text
    |> List.concat_map (fun (piece, _) ->
        List.init (String.length piece) (fun i ->
            byte_alphabet.(Char.code piece.[i]))
        |> List.filter (Hashtbl.mem model.ids)
        |> merge_by (Hashtbl.find_opt model.ranks)
        |> List.map id)
    |> Array.of_list

let sp = "\u{2581}"

(* Words to past the 15-byte key, contractions in any case, digit groups, every
   kind of white space and the cl100k rules around it, letters beyond ASCII,
   characters no piece holds, and ▁. *)
let cl100k_alphabet =
  [ "the"; "and"; "of"; "tokenizer"; "information"; "internationalization";
    "Hello"; "WORLD"; "aaaaaaaaaaaaaaaaaaaaaaaa"; "!!!!!!!!"; "don't";
    "it's"; "we'll"; "they're"; "I'm"; "'"; "1"; "2024"; "3.14159265358979";
    "."; ","; "\""; ")"; ";"; ":"; "!"; "?"; "("; "-"; "..."; " "; " ";
    " "; "  "; "\n"; "\n\n"; "\t"; " \r\n"; "caf\u{e9}"; "na\u{ef}ve";
    "\u{416}"; "\u{41f}\u{440}\u{438}\u{432}\u{435}\u{442}";
    "\u{65e5}\u{672c}\u{8a9e}"; "\u{1f600}"; "e\u{301}"; "\u{df}"; sp;
    sp ^ sp; "'S"; "'LL"; "'Re"; "'VE"; "'D"; "''"; "12"; "1234567"; "\r\n";
    " \n "; "  \n"; "\n\n  "; "\t?"; "?!"; ".\n\n"; "\x0B"; "\u{a0}";
    "\u{2019}"; "\u{663}"; "\u{3002}"; "\u{3000}"; "a\u{301}" ]
[@@ocamlformat "disable"]

let has_sub text sub =
  let n = String.length sub in
  let rec go i =
    i + n <= String.length text && (String.sub text i n = sub || go (i + 1))
  in
  go 0

let batch_rows t docs =
  let buf, lengths = encode_batch_ids t ~add_special_tokens:false docs in
  let pos = ref 0 in
  Array.to_list lengths
  |> List.map (fun n ->
      let row =
        Array.init n (fun k ->
            Int32.to_int (Bigarray.Array1.get buf (!pos + k)))
      in
      pos := !pos + n;
      row)

let phi4_law =
  let loaded =
    lazy
      (Fixture.with_tokenizer_file "phi4" (fun path ->
           (load path, phi4_reference (file_model path))))
  in
  prop ~count:60 "encodes phi4's text as its pattern's pieces merged by rank"
    (quoted
       Gen.(
         map (String.concat "")
           (list ~size:(int_range 0 80) (of_list cl100k_alphabet))))
    (fun text ->
      let t, reference = Lazy.force loaded in
      cover "a contraction in upper case"
        (has_sub text "'S" || has_sub text "'LL");
      cover "a group of three digits" (has_sub text "1234567");
      cover "a newline then spaces" (has_sub text "\n ");
      cover "punctuation before a newline" (has_sub text ".\n");
      cover "a character no piece holds" (has_sub text "\u{1f600}");
      let expected = reference text in
      equal ~msg:"alone" (array int) expected
        (encode_ids t ~add_special_tokens:false text);
      equal ~msg:"twice in a batch"
        (list (array int))
        [ expected; expected ]
        (batch_rows t [ text; text ]))

(* A pipeline with no splitting pre-tokenizer hands the model whole documents,
   which it merges as one word whatever ▁ and punctuation they hold. *)
let whole_document =
  let repeat n parts = List.concat (List.init n (fun _ -> parts)) in
  let doc n part = String.concat "" (List.init n (fun _ -> part)) in
  cases "merges a document as one word, across ▁ and punctuation"
    ~id:(fun (id, _, _, _) -> id)
    [
      ( "a run of ▁ merges with the word it opens",
        (fun () ->
          bpe
            ~vocab:[ ("a", 0); ("b", 1); (sp, 2); (sp ^ sp, 3) ]
            ~merges:[ (sp, sp) ]
            ()),
        doc 5 ("a" ^ sp ^ sp ^ "b"),
        repeat 5 [ "a"; sp ^ sp; "b" ] );
      ( "a piece crossing ▁ is built",
        (fun () ->
          bpe
            ~vocab:(numbered [ "z"; "a"; "b"; sp; "a" ^ sp; "a" ^ sp ^ "b" ])
            ~merges:[ ("a", sp); ("a" ^ sp, "b") ]
            ()),
        doc 4 ("za" ^ sp ^ "b"),
        repeat 4 [ "z"; "a" ^ sp ^ "b" ] );
      ( "fuse_unk fuses an unknown run across ▁",
        (fun () ->
          bpe
            ~vocab:[ ("a", 0); ("<u>", 1) ]
            ~merges:[] ~unk_token:"<u>" ~fuse_unk:true ()),
        "aaaaaxx" ^ sp ^ "xxaaaaa",
        repeat 5 [ "a" ] @ [ "<u>" ] @ repeat 5 [ "a" ] );
      ( "punctuation merges into the piece before it",
        (fun () ->
          bpe
            ~vocab:(numbered [ "b"; "a"; "."; "a." ])
            ~merges:[ ("a", ".") ]
            ()),
        doc 6 "ba.",
        repeat 6 [ "b"; "a." ] );
      ( "fuse_unk fuses an unknown run across punctuation",
        (fun () ->
          bpe
            ~vocab:[ ("a", 0); (sp, 1); ("<u>", 2) ]
            ~merges:[] ~unk_token:"<u>" ~fuse_unk:true ()),
        "aaaa...aaaa" ^ sp ^ "aaaa",
        repeat 4 [ "a" ] @ [ "<u>" ] @ repeat 4 [ "a" ] @ [ sp ]
        @ repeat 4 [ "a" ] );
      ( "a byte token merges with the ▁ byte after it",
        (fun () ->
          bpe
            ~vocab:(numbered [ "a"; "<0xE2>"; "<0x96>"; "<0x81>"; "a<0xE2>" ])
            ~merges:[ ("a", "<0xE2>") ]
            ~byte_fallback:true ()),
        String.make 16 'a' ^ sp ^ "aaaa",
        repeat 15 [ "a" ] @ [ "a<0xE2>"; "<0x96>"; "<0x81>" ] @ repeat 4 [ "a" ]
      );
      ( "byte tokens of punctuation merge",
        (fun () ->
          bpe
            ~vocab:(numbered [ "a"; "<0x2E>"; "<0x2E><0x2E>" ])
            ~merges:[ ("<0x2E>", "<0x2E>") ]
            ~byte_fallback:true ()),
        String.make 16 'a' ^ ".." ^ "aaaa",
        repeat 16 [ "a" ] @ [ "<0x2E><0x2E>" ] @ repeat 4 [ "a" ] );
      ( "a prefix goes on every character after the first",
        (fun () ->
          bpe
            ~vocab:(numbered [ "a"; "##a"; "."; "##." ])
            ~merges:[] ~continuing_subword_prefix:"##" ()),
        String.make 16 'a' ^ "." ^ "aaaa",
        [ "a" ] @ repeat 15 [ "##a" ] @ [ "##." ] @ repeat 4 [ "##a" ] );
      ( "ignore_merges takes the whole document, not a ▁ unit",
        (fun () ->
          bpe
            ~vocab:(numbered [ "a"; "b"; sp; sp ^ "ab" ])
            ~merges:[] ~ignore_merges:true ()),
        String.make 16 'a' ^ sp ^ "ab",
        repeat 16 [ "a" ] @ [ sp; "a"; "b" ] );
    ]
    (fun (_, make, doc, expected) ->
      equal (list string) expected (tokens (make ()) doc))

(* mistral's file, and the same with one more entry that crosses a ▁ and that no
   merge builds: it changes no result, so the two encode alike. *)
let with_crossing_entry root piece id =
  let in_object name f = function
    | Jsont.Object (members, meta) ->
        let member ((k, km), v) = ((k, km), if k = name then f v else v) in
        Jsont.Object (List.map member members, meta)
    | _ -> failf "expected a JSON object around %S" name
  in
  in_object "model"
    (in_object "vocab" (function
      | Jsont.Object (members, meta) ->
          let entry = (Jsont.Json.name piece, Jsont.Json.int id) in
          Jsont.Object (members @ [ entry ], meta)
      | _ -> failf "expected the vocabulary to be a JSON object"))
    root

let crossing_docs =
  [ ""; " "; "hello world"; "  leading and   trailing  "; "a" ^ sp ^ "b";
    sp ^ sp ^ sp; "tabs\tand\nnewlines\r\nCRLF"; "nbsp\u{a0}and\u{a0}runs";
    "caf\u{e9} na\u{ef}ve \u{65e5}\u{672c}\u{8a9e}";
    "emoji \u{1f600}\u{1f9e0} fallback"; "<s> specials </s> inline";
    "punct.,\");:!? runs ... !!! ??? end."; "digits 1234567890 mix3d w0rds";
    String.concat " " (List.init 40 (fun i -> Printf.sprintf "w%d" i));
    String.make 5000 'a'; String.concat "" (List.init 30 (fun _ -> "    indent"));
    String.concat "." (List.init 12 (fun _ -> "ab"));
    "so, it says: \"yes\" (twice); really?! yes... end.";
    ".,\");:!?.,\");:!?" ^ String.concat "," (List.init 8 (fun _ -> "1")) ]
[@@ocamlformat "disable"]

(* Documents drawn from a fixed seed, heavy in ▁, white space, digits and
   punctuation. *)
let random_docs n =
  let st = Random.State.make [| 0x7352 |] in
  let doc () =
    let b = Buffer.create 800 in
    for _ = 1 to 1 + Random.State.int st 400 do
      match Random.State.int st 100 with
      | 0 | 1 | 2 -> Buffer.add_string b sp
      | 3 | 4 -> Buffer.add_string b "  "
      | 5 -> Buffer.add_string b "    "
      | 6 -> Buffer.add_char b '\t'
      | 7 -> Buffer.add_char b '\n'
      | 8 -> Buffer.add_string b "\u{e9}"
      | 9 -> Buffer.add_string b "\u{65e5}"
      | 10 -> Buffer.add_string b "\u{1f600}"
      | 11 -> Buffer.add_string b "\u{436}"
      | 12 -> Buffer.add_char b (Char.chr (1 + Random.State.int st 31))
      | 13 -> Buffer.add_string b "<s>"
      | 14 -> Buffer.add_string b "</s>"
      | k when k < 40 -> Buffer.add_char b ' '
      | k when k < 55 ->
          Buffer.add_char b (Char.chr (48 + Random.State.int st 10))
      | k when k < 70 ->
          Buffer.add_char b ".,\"();:!?-'".[Random.State.int st 11]
      | _ -> Buffer.add_char b (Char.chr (97 + Random.State.int st 26))
    done;
    Buffer.contents b
  in
  List.init n (fun _ -> doc ())

let test_crossing_entry () =
  Fixture.with_tokenizer_file "mistral" @@ fun path ->
  let json =
    match Jsont_bytesrw.decode_string Jsont.json (Fixture.read path) with
    | Ok json -> json
    | Error msg -> failf "mistral.json: %s" msg
  in
  let read json =
    match of_json json with Ok t -> t | Error msg -> failf "mistral: %s" msg
  in
  let t = read json in
  let top = List.fold_left (fun m (_, id) -> max m id) (-1) (vocab t) in
  let crossing = read (with_crossing_entry json ("zz" ^ sp ^ "zz") (top + 1)) in
  List.iteri
    (fun i doc ->
      let a = encode t ~add_special_tokens:false doc in
      let b = encode crossing ~add_special_tokens:false doc in
      let msg what = Printf.sprintf "doc %d %s" i what in
      equal ~msg:(msg "ids") (array int) (Encoding.ids b) (Encoding.ids a);
      equal ~msg:(msg "offsets")
        (array (pair int int))
        (Encoding.offsets b) (Encoding.offsets a);
      equal ~msg:(msg "tokens") (array string) (Encoding.tokens b)
        (Encoding.tokens a))
    (crossing_docs @ random_docs 300)

let test_dropout_zero () =
  let t = collide ~dropout:0. 4096 and plain = collide 4096 in
  List.iter
    (fun word ->
      equal ~msg:word (array int) (encode_ids plain word) (encode_ids t word))
    (recurring @ long_words)

(* At probability one every merge is skipped, whether or not the vocabulary
   holds the pretoken, and whatever path encodes it. *)
let full_dropout =
  cases "skips every merge at dropout 1"
    ~id:(fun (id, _, _, _) -> id)
    [
      ( "a word vocab holds, under ignore_merges",
        (fun () ->
          bpe
            ~vocab:[ ("a", 0); ("b", 1); ("ab", 2) ]
            ~merges:[ ("a", "b") ]
            ~ignore_merges:true ~dropout:1.0 ()),
        "ab",
        [| 0; 1 |] );
      ( "byte-level pretokens the vocabulary holds",
        (fun () ->
          bpe
            ~vocab:
              (List.init 256 (fun b -> (byte_alphabet.(b), b))
              @ [ ("\u{120}a", 256) ])
            ~merges:[ ("\u{120}", "a") ]
            ~pre:(Pre_tokenizer.byte_level ~add_prefix_space:false ())
            ~dropout:1.0 ()),
        " a a",
        [| 32; 97; 32; 97 |] );
      ( "a document of ▁ units",
        (fun () ->
          bpe
            ~vocab:[ (sp, 0); ("a", 1); (sp ^ "a", 2) ]
            ~merges:[ (sp, "a") ]
            ~dropout:1.0 ()),
        String.concat "" (List.init 8 (fun _ -> sp ^ "a")),
        Array.concat (List.init 8 (fun _ -> [| 0; 1 |])) );
    ]
    (fun (_, make, text, ids) ->
      equal (array int) ids
        (encode_ids (make ()) ~add_special_tokens:false text))

(* ["ab"] is in the vocabulary but no merge builds it, so only [ignore_merges]
   makes the whole word win. *)
let test_ignore_merges () =
  let vocab = [ ("a", 0); ("b", 1); ("c", 2); ("bc", 3); ("ab", 4) ] in
  let merges = [ ("b", "c") ] in
  let merging = bpe ~vocab ~merges () in
  let ignoring = bpe ~vocab ~merges ~ignore_merges:true () in
  equal ~msg:"merges decide" (list string) [ "a"; "b" ] (tokens merging "ab");
  equal ~msg:"ignore_merges" (list string) [ "ab" ] (tokens ignoring "ab");
  equal ~msg:"a word vocab lacks" (list string) [ "a"; "bc" ]
    (tokens ignoring "abc")

(* Under [ignore_merges] a cache holds the entry itself, and a pretoken that
   shares its one set does not take its place. *)
let test_ignore_merges_cached () =
  let uncached = collide ~ignore_merges:true 0 in
  let cached = collide ~ignore_merges:true 1 in
  equal ~msg:"first" (list string) [ "aaa" ] (tokens cached "aaa");
  List.iter
    (fun word ->
      equal ~msg:word (array int) (encode_ids uncached word)
        (encode_ids cached word))
    collide_words;
  equal ~msg:"after the other words" (list string) [ "aaa" ]
    (tokens cached "aaa")

(* HF: with ["a</w>"] held and not ["a"], "ab" -> <unk>, b</w>. *)
let test_affixed_unknown () =
  let t =
    bpe
      ~vocab:(numbered [ "<unk>"; "a"; "b"; "b</w>" ])
      ~merges:[] ~end_of_word_suffix:"</w>" ~unk_token:"<unk>" ()
  in
  equal ~msg:"suffixed form missing" (list string) [ "<unk>" ] (tokens t "a");
  equal ~msg:"plain form found" (list string) [ "a"; "b</w>" ] (tokens t "ab");
  equal ~msg:"suffixed form missing at the end" (list string) [ "b"; "<unk>" ]
    (tokens t "ba");
  let ending =
    bpe
      ~vocab:(numbered [ "<unk>"; "b"; "a</w>"; "b</w>" ])
      ~merges:[] ~end_of_word_suffix:"</w>" ~unk_token:"<unk>" ()
  in
  equal ~msg:"plain form missing" (list string) [ "<unk>"; "b</w>" ]
    (tokens ending "ab");
  equal ~msg:"suffixed form found" (list string) [ "a</w>" ] (tokens ending "a")

(* A two-byte prefix turns one byte into three symbols, which the merge buffers
   must be sized for. HF: with only ["a"], "ba" -> <0x62>, <0x23>, <0x23>,
   <0x61>; seven characters give 19 tokens and forty give 118. *)
let test_affixed_byte_fallback () =
  let suffixed =
    bpe
      ~vocab:([ ("a", 0); ("b", 1) ] @ printable_bytes ~first:2)
      ~merges:[] ~end_of_word_suffix:"</w>" ~byte_fallback:true ()
  in
  let suffix = [ "<0x3C>"; "<0x2F>"; "<0x77>"; "<0x3E>" ] in
  equal ~msg:"suffix: one character" (list string) ("<0x61>" :: suffix)
    (tokens suffixed "a");
  equal ~msg:"suffix: the last character alone" (list string)
    ("a" :: "<0x62>" :: suffix)
    (tokens suffixed "ab");
  let prefixed =
    bpe
      ~vocab:(("a", 0) :: printable_bytes ~first:1)
      ~merges:[] ~continuing_subword_prefix:"##" ~byte_fallback:true ()
  in
  let spelled first rest n =
    first
    :: List.concat_map
         (fun i -> [ "<0x23>"; "<0x23>"; rest i ])
         (List.init (n - 1) Fun.id)
  in
  let b _ = "<0x62>" in
  equal ~msg:"prefix: the first character bare" (list string)
    (spelled "<0x62>" (fun _ -> "<0x61>") 2)
    (tokens prefixed "ba");
  equal ~msg:"prefix: a later character" (list string) (spelled "a" b 2)
    (tokens prefixed "ab");
  equal ~msg:"prefix: seven characters" (list string) (spelled "<0x62>" b 7)
    (tokens prefixed (String.make 7 'b'));
  equal ~msg:"prefix: forty characters" (list string) (spelled "<0x62>" b 40)
    (tokens prefixed (String.make 40 'b'));
  equal ~msg:"prefix: alternating characters" (list string)
    (spelled "a" (fun i -> if i mod 2 = 0 then "<0x62>" else "<0x61>") 7)
    (tokens prefixed "abababa")

(* Without unk_token a character vocab lacks gives no token. A word can reach
   the merges empty; the merge buffers are reused, and an empty word does not
   pick up the symbols of the one before it. *)
let test_unknown_character () =
  let t =
    bpe ~vocab:[ ("a", 0); ("b", 1); ("ab", 2) ] ~merges:[ ("a", "b") ] ()
  in
  equal ~msg:"alone" (list string) [] (tokens t "z");
  equal ~msg:"a merged word" (list string) [ "ab" ] (tokens t "ab");
  equal ~msg:"after a merged word" (list string) [] (tokens t "z");
  equal ~msg:"after a known character" (list string) [ "a" ] (tokens t "az")

(* HF: BPE(unk_token="<unk>") over a vocabulary without it encodes "a" and
   raises "Unk token `<unk>` not found in the vocabulary" at "az". Each walk a
   word starts with: characters, affixed characters and bytes. *)
let test_missing_unk () =
  let vocab = [ ("a", 0); ("b", 1) ] and unk_token = "<unk>" in
  let plain = bpe ~vocab ~unk_token () in
  let affixed = bpe ~vocab ~unk_token ~end_of_word_suffix:"</w>" () in
  let bytes =
    bpe
      ~pre:(Pre_tokenizer.byte_level ~add_prefix_space:false ())
      ~vocab ~unk_token ()
  in
  equal ~msg:"known" (list string) [ "a" ] (tokens plain "a");
  List.iter
    (fun (msg, t) -> raises_match ~msg Exn.failure (fun () -> encode t "az"))
    [ ("characters", plain); ("affixed", affixed); ("bytes", bytes) ]

(* HF: a byte fallback never lets a pending unknown token out, so the unknown
   token follows the bytes after it and comes out at the next vocabulary hit or
   at the end of the word; [fuse_unk] joins the unknown tokens of a run, across
   a fallback too. *)
let held_unknown =
  let fallback ~fuse_unk () =
    bpe
      ~vocab:
        (("<unk>", 100)
        :: List.mapi
             (fun i b -> (Printf.sprintf "<0x%02X>" b, i))
             [ 0x61; 0x3C; 0x2F; 0x77; 0x3E; 0x62 ])
      ~merges:[] ~unk_token:"<unk>" ~byte_fallback:true ~fuse_unk
      ~end_of_word_suffix:"</w>" ()
  in
  let plain = fallback ~fuse_unk:false and fused = fallback ~fuse_unk:true in
  let hit () =
    bpe
      ~vocab:[ ("<unk>", 100); ("<0x61>", 0); ("b", 1) ]
      ~merges:[] ~unk_token:"<unk>" ~byte_fallback:true ()
  in
  let a = [ "<0x61>"; "<0x3C>"; "<0x2F>"; "<0x77>"; "<0x3E>" ] in
  cases "holds an unknown token back to the next vocabulary hit"
    ~id:(fun (id, _, _, _) -> id)
    [
      ("a word spelled whole", plain, "a", a);
      ("after the bytes that follow it", plain, "za", a @ [ "<unk>" ]);
      ("before the bytes when last", plain, "az", [ "<0x61>"; "<unk>" ]);
      ("one per character", plain, "zaz", [ "<0x61>"; "<unk>"; "<unk>" ]);
      ("two in a row", plain, "zz", [ "<unk>"; "<unk>" ]);
      ("fused across the bytes", fused, "zaz", [ "<0x61>"; "<unk>" ]);
      ("fused in a row", fused, "zz", [ "<unk>" ]);
      ("fused before the bytes", fused, "zza", a @ [ "<unk>" ]);
      ("let out by a hit", hit, "zab", [ "<0x61>"; "<unk>"; "b" ]);
      ("nothing held before a hit", hit, "zba", [ "<unk>"; "b"; "<0x61>" ]);
      ( "several bytes before a hit",
        hit,
        "zaab",
        [ "<0x61>"; "<0x61>"; "<unk>"; "b" ] );
    ]
    (fun (_, make, text, expected) ->
      equal (list string) expected (tokens (make ()) text))

(* HF reports offsets in the order of the tokens. *)
let test_held_unknown_offsets () =
  let t =
    bpe
      ~vocab:[ ("<unk>", 100); ("<0x61>", 0); ("b", 1) ]
      ~merges:[] ~unk_token:"<unk>" ~byte_fallback:true ()
  in
  equal
    (array (pair int int))
    [| (0, 1); (1, 2); (2, 3) |]
    (Encoding.offsets (encode t "zab"))

(* A character cut short at the end of the input is its bytes, each a character
   of one byte, never the bytes its lead byte promises. *)
let test_cut_character () =
  let t = bpe ~vocab:(numbered [ "a"; "\u{e9}"; "\u{20ac}" ]) ~merges:[] () in
  equal ~msg:"whole characters" (list string)
    [ "a"; "\u{e9}"; "\u{20ac}" ]
    (tokens t "a\u{e9}\u{20ac}");
  equal ~msg:"two bytes cut to one" (list string) [ "a" ] (tokens t "a\xC3");
  equal ~msg:"three bytes cut to two" (list string) [ "a" ]
    (tokens t "a\xE2\x82");
  equal ~msg:"a four-byte lead alone" (list string) [ "a" ] (tokens t "a\xF0");
  let prefixed =
    bpe
      ~vocab:(("a", 0) :: printable_bytes ~first:1)
      ~merges:[] ~continuing_subword_prefix:"##" ~byte_fallback:true ()
  in
  equal ~msg:"spelled out under a prefix" (list string) [ "a" ]
    (tokens prefixed "a\xE2\x82");
  let suffixed =
    bpe
      ~vocab:[ ("<unk>", 0); ("a", 1) ]
      ~merges:[] ~unk_token:"<unk>" ~end_of_word_suffix:"</w>" ()
  in
  equal ~msg:"each byte unknown under a suffix" (list string)
    [ "a"; "<unk>"; "<unk>" ]
    (tokens suffixed "a\xE2\x82")

(* A model packs two ids, or a rank and an id, into one integer, and a merge
   queue a rank and a position: ids past 2^21, in a word merged in place, on the
   heap and through the byte-level walk. *)
let test_large_ids () =
  let a = (1 lsl 21) + 3 and b = (1 lsl 22) + 1 and ab = (1 lsl 23) + 7 in
  let vocab = [ ("a", a); ("b", b); ("ab", ab) ] and merges = [ ("a", "b") ] in
  let plain = bpe ~vocab ~merges () in
  let walked = bpe ~pre:(Pre_tokenizer.byte_level ()) ~vocab ~merges () in
  List.iter
    (fun (name, t) ->
      equal ~msg:(name ^ ", ab") (array int) [| ab |] (encode_ids t "ab");
      equal ~msg:(name ^ ", ba") (array int) [| b; a |] (encode_ids t "ba"))
    [ ("plain", plain); ("byte-level", walked) ];
  equal ~msg:"the heap" (array int) (Array.make 40 ab)
    (encode_ids plain (String.concat "" (List.init 40 (fun _ -> "ab"))))

let test_far_merge () =
  let n = 1 lsl 21 in
  let t =
    bpe ~vocab:[ ("a", 0); ("b", 1); ("ab", 2) ] ~merges:[ ("a", "b") ] ()
  in
  let ids = encode_ids t (String.make n 'a' ^ "ab") in
  equal ~msg:"ids" int (n + 1) (Array.length ids);
  equal ~msg:"the merge" int 2 ids.(n);
  equal ~msg:"the letters before it" (array int) (Array.make n 0)
    (Array.sub ids 0 n)

(* Tables sized by the largest id took 2^23 words or more for the sparse
   vocabulary below. *)
let sparse_ids make =
  let top = 1 lsl 23 in
  let before = Gc.allocated_bytes () in
  let t = make [ ("[UNK]", 0); ("a", 1); ("b", top) ] in
  let encoding = encode t "a b" in
  let allocated = Gc.allocated_bytes () -. before in
  less ~msg:"bytes allocated" float_exact ~than:16e6 allocated;
  equal ~msg:"ids" (array int) [| 1; top |] (Encoding.ids encoding);
  equal ~msg:"tokens" (array string) [| "a"; "b" |] (Encoding.tokens encoding);
  equal ~msg:"offsets"
    (array (pair int int))
    [| (0, 1); (2, 3) |]
    (Encoding.offsets encoding);
  equal ~msg:"id_to_token" (option string) (Some "b") (id_to_token t top);
  equal ~msg:"decode" string "a b" (decode t [| 1; top |]);
  let top = (1 lsl 31) - 1 in
  let t = make [ ("[UNK]", 0); ("b", top) ] in
  equal ~msg:"2^31 - 1" (option string) (Some "b") (id_to_token t top)

let refused_ids make =
  List.iter
    (fun id ->
      raises_match ~msg:(Printf.sprintf "id %d" id)
        (Exn.invalid_arg ~sub:"2^31") (fun () ->
          make [ ("[UNK]", 0); ("b", id) ]))
    [ -1; 1 lsl 31; max_int; min_int ]

(* A token given twice is its last pair on every path that reads the vocabulary,
   whichever walk the model encodes with. *)
let last_pair t ~pairs =
  equal ~msg:"token_to_id" (option int) (Some 2) (token_to_id t "a");
  equal ~msg:"the earlier id" (option string) None (id_to_token t 0);
  equal ~msg:"the last id" (option string) (Some "a") (id_to_token t 2);
  equal ~msg:"encode" (array int) [| 2 |] (encode_ids t "a");
  equal ~msg:"vocab_size" int (List.length pairs) (vocab_size t);
  equal ~msg:"vocab"
    (list (pair string int))
    pairs
    (List.sort compare (vocab t))

let bpe_given_twice =
  cases "keeps a token given twice as its last pair" ~id:fst
    [ ("plain", false); ("byte-level", true) ]
    (fun (_, byte_level) ->
      let pre =
        if byte_level then
          Some (Pre_tokenizer.byte_level ~add_prefix_space:false ())
        else None
      in
      let t =
        bpe ?pre
          ~vocab:[ ("a", 0); ("b", 1); ("a", 2); ("ab", 3) ]
          ~merges:[ ("a", "b") ]
          ()
      in
      last_pair t ~pairs:[ ("a", 2); ("ab", 3); ("b", 1) ];
      equal ~msg:"merged" (array int) [| 3 |] (encode_ids t "ab"))

let () =
  exit
    (run "bpe"
       [
         group "t"
           [
             bpe_cache;
             bpe_domains;
             test "reads back every pretoken of a full BPE table"
               test_full_table;
             test "encodes real text alike under every cache_capacity"
               test_real_text;
           ];
         group "bpe"
           [
             merge_law;
             phi4_law;
             whole_document;
             test "encodes mistral's documents as with an entry crossing ▁"
               test_crossing_entry;
             test "merges at dropout 0 as without dropout" test_dropout_zero;
             full_dropout;
             test "emits a word vocab holds as one token under ignore_merges"
               test_ignore_merges;
             test
               "keeps an entry no merge builds whole under ignore_merges, \
                cached"
               test_ignore_merges_cached;
             test
               "gives unk_token for a character whose affixed form vocab lacks"
               test_affixed_unknown;
             test "spells an affixed character out in bytes under byte_fallback"
               test_affixed_byte_fallback;
             test
               "gives no token for a character vocab lacks, without unk_token"
               test_unknown_character;
             test "raises Failure at a character when vocab lacks unk_token"
               test_missing_unk;
             held_unknown;
             test "reports offsets in the order of the tokens"
               test_held_unknown_offsets;
             test "reads a character cut short as its bytes" test_cut_character;
             test "gives ids past 2^21 through every merge" test_large_ids;
             test "merges at a position past 2^21" test_far_merge;
             test "allocates for its tokens, not its largest id" (fun () ->
                 sparse_ids (fun vocab ->
                     bpe ~pre:Pre_tokenizer.whitespace ~vocab ()));
             test "refuses an id negative or 2^31 and past" (fun () ->
                 refused_ids (fun vocab -> bpe ~vocab ()));
             bpe_given_twice;
           ];
       ])
