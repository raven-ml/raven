(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Encoding agrees with its references whichever kernels run. The native binary
   encodes through the C kernels and the bytecode one through their OCaml
   versions, so each claim here holds both to the same reference: the
   pre-tokenizer's or the regular expression's cut, and a BPE merge written the
   simplest way. The hand-written inputs are the kernels' edges: the seams of
   their buffers, pretokens past the cache's 15-byte key, characters classified
   as they are first met, bytes without an id. The values of brot.mli's hidden
   section that reach the kernels' caches and the batch's chunks follow. *)

open Windtrap

(* The GPT-2 byte-level alphabet, restated: brot does not expose the direction
   from bytes to symbols. *)
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

let base_vocab = List.init 256 (fun b -> (byte_alphabet.(b), b))
let pre () = Brot.Pre_tokenizer.byte_level ~add_prefix_space:false ()
let ids tok text = Brot.encode_ids tok ~add_special_tokens:false text
let encode tok text = Brot.encode tok ~add_special_tokens:false text
let sp_mark = "\u{2581}"
let repeat k s = String.concat "" (List.init k (fun _ -> s))
let pad n = String.init n (fun i -> "ab ".[i mod 3])

let has_sub text sub =
  let n = String.length sub in
  let rec go i =
    i + n <= String.length text && (String.sub text i n = sub || go (i + 1))
  in
  go 0

let has_any text subs = List.exists (has_sub text) subs
let quoted = Gen.with_pp (fun ppf s -> Format.fprintf ppf "%S" s)

let text_of ~size alphabet =
  quoted
    (Gen.map (String.concat "")
       (Gen.list ~size:(Gen.int_range 0 size) (Gen.of_list alphabet)))

(* Spans *)

(* The spans [tok] encodes [text] in: each word's first and last token. *)
let kernel_spans tok text =
  let e = encode tok text in
  let offsets = Brot.Encoding.offsets e in
  let words = Brot.Encoding.word_ids e in
  let n = Array.length offsets in
  let spans = ref [] in
  let i = ref 0 in
  while !i < n do
    let j = ref !i in
    while !j + 1 < n && words.(!j + 1) = words.(!i) do
      incr j
    done;
    spans := (fst offsets.(!i), snd offsets.(!j)) :: !spans;
    i := !j + 1
  done;
  Array.of_list (List.rev !spans)

let pre_spans pre text =
  Array.of_list (List.map snd (Brot.Pre_tokenizer.pre_tokenize pre text))

(* A pattern in a group is not one the walkers recognise, so the regular
   expression engine cuts it. *)
let regex_split pattern =
  Brot.Pre_tokenizer.split_regex
    ~pattern:("(?:" ^ pattern ^ ")")
    ~behavior:`Isolated ()

(* A recognised split, then the byte-level encoding of its pieces: the pipeline
   the cl100k and o200k walkers serve. *)
let pre_split pattern =
  Brot.Pre_tokenizer.sequence
    [
      Brot.Pre_tokenizer.split_regex ~pattern ~behavior:`Isolated ();
      Brot.Pre_tokenizer.byte_level ~add_prefix_space:false ~use_regex:false ();
    ]

(* Each family's shape at every [step]th offset from 0 to 80, encoded by [tok],
   cut by [reference]. The walkers classify 64-byte batches, so the shape meets
   every place of a batch's edge. *)
let sweep ?(step = 1) tok reference families =
  List.iter
    (fun (name, shape) ->
      for i = 0 to 80 / step do
        let text = shape (i * step) in
        equal
          ~msg:(Printf.sprintf "%s at offset %d" name (i * step))
          (array (pair int int))
          (pre_spans reference text) (kernel_spans tok text)
      done)
    families

(* The reference merge *)

type reference = {
  vocab : (string, int) Hashtbl.t;
  ranks : (string * string, int) Hashtbl.t;
}

let reference_of_lists vocab merges =
  let r = { vocab = Hashtbl.create 512; ranks = Hashtbl.create 512 } in
  List.iter (fun (s, id) -> Hashtbl.replace r.vocab s id) vocab;
  List.iteri
    (fun rank pair ->
      if not (Hashtbl.mem r.ranks pair) then Hashtbl.add r.ranks pair rank)
    merges;
  r

let json_member name = function
  | Jsont.Object (members, _) -> (
      match Jsont.Json.find_mem name members with
      | Some (_, v) -> v
      | None -> failf "no member %s" name)
  | _ -> failf "%s: not an object" name

let reference_of path =
  let json =
    match Jsont_bytesrw.decode_string Jsont.json (Fixture.read path) with
    | Ok json -> json
    | Error e -> failf "%s: %s" path e
  in
  let model = json_member "model" json in
  let vocab =
    match json_member "vocab" model with
    | Jsont.Object (members, _) ->
        List.map
          (fun ((name, _), v) ->
            match v with
            | Jsont.Number (id, _) -> (name, int_of_float id)
            | _ -> failf "vocab %s: not a number" name)
          members
    | _ -> failf "vocab: not an object"
  in
  let merge = function
    | Jsont.String (m, _) -> (
        match String.index_opt m ' ' with
        | Some i ->
            (String.sub m 0 i, String.sub m (i + 1) (String.length m - i - 1))
        | None -> failf "merge %S" m)
    | Jsont.Array ([ Jsont.String (a, _); Jsont.String (b, _) ], _) -> (a, b)
    | _ -> failf "merge: not a pair"
  in
  match json_member "merges" model with
  | Jsont.Array (merges, _) -> reference_of_lists vocab (List.map merge merges)
  | _ -> failf "merges: not an array"

(* Merges the leftmost adjacent pair of lowest rank until none has one. A symbol
   is its text and the bytes [start, stop) of the encoded text it stands for,
   which a merge joins. *)
let rec merge_all r symbols =
  let best = ref None in
  let rec scan i = function
    | (a, _, _) :: ((b, _, _) :: _ as rest) ->
        (match (Hashtbl.find_opt r.ranks (a, b), !best) with
        | Some rank, Some (best_rank, _) when rank >= best_rank -> ()
        | Some rank, _ -> best := Some (rank, i)
        | None, _ -> ());
        scan (i + 1) rest
    | _ -> ()
  in
  scan 0 symbols;
  match !best with
  | None -> symbols
  | Some (_, at) ->
      let rec join i = function
        | (a, start, _) :: (b, _, stop) :: rest when i = at ->
            (a ^ b, start, stop) :: rest
        | s :: rest -> s :: join (i + 1) rest
        | [] -> []
      in
      merge_all r (join 0 symbols)

let id_of r symbol =
  match Hashtbl.find_opt r.vocab symbol with
  | Some id -> id
  | None -> failf "no id for %S" symbol

(* A token of the reference: its id, its offsets and its word. *)
let tokens_of r ~word symbols =
  List.map (fun (s, start, stop) -> (id_of r s, (start, stop), word)) symbols

let ids_of tokens = Array.of_list (List.map (fun (id, _, _) -> id) tokens)
let offsets_of tokens = Array.of_list (List.map (fun (_, o, _) -> o) tokens)
let words_of tokens = Array.of_list (List.map (fun (_, _, w) -> Some w) tokens)

(* The characters of [s] and their bytes, as brot reads them: a byte that opens
   no whole character is a character of its own. *)
let chars s =
  let rec go i acc =
    if i >= String.length s then List.rev acc
    else
      let d = String.get_utf_8_uchar s i in
      let n =
        if Uchar.utf_decode_is_valid d then Uchar.utf_decode_length d else 1
      in
      go (i + n) ((String.sub s i n, i, i + n) :: acc)
  in
  go 0 []

(* A byte-level model merges the byte-level spelling of each piece of its
   pattern, cut by the regular expression engine, leaving out the bytes without
   an id; under [ignore_merges] a piece that is itself an entry is that entry. A
   token spans the whole characters its bytes belong to, so the tokens of one
   character's bytes each span it. *)
let byte_level_reference ?(normalize = Fun.id) ?(ignore_merges = false) ~pattern
    r =
  let split = regex_split pattern in
  fun text ->
    let text = normalize text in
    let first = Array.make (String.length text) 0 in
    let last = Array.make (String.length text) 0 in
    List.iter
      (fun (_, start, stop) ->
        for i = start to stop - 1 do
          first.(i) <- start;
          last.(i) <- stop
        done)
      (chars text);
    let token ~word (s, start, stop) =
      (id_of r s, (first.(start), last.(stop - 1)), word)
    in
    Brot.Pre_tokenizer.pre_tokenize split text
    |> List.mapi (fun word (piece, (start, stop)) ->
        let symbols =
          List.init (String.length piece) (fun i ->
              (byte_alphabet.(Char.code piece.[i]), start + i, start + i + 1))
        in
        let whole = String.concat "" (List.map (fun (s, _, _) -> s) symbols) in
        match Hashtbl.find_opt r.vocab whole with
        | Some _ when ignore_merges -> [ token ~word (whole, start, stop) ]
        | _ ->
            List.filter (fun (s, _, _) -> Hashtbl.mem r.vocab s) symbols
            |> merge_all r
            |> List.map (token ~word))
    |> List.concat

(* A SentencePiece model with no pre-tokenizer merges the whole text from its
   characters. A character without a piece is spelled by its bytes' fallback
   tokens when there is one for each, each standing for the whole character;
   otherwise it is the unknown token, which a run of such characters shares and
   which goes out at the next piece or at the end, a fallback spelling
   overtaking it; with no unknown token it is left out. *)
let piece_reference ?unk r text =
  let fallback c =
    List.init (String.length c) (fun i ->
        Printf.sprintf "<0x%02X>" (Char.code c.[i]))
  in
  let rec spell pending = function
    | [] -> Option.to_list pending
    | ((c, _, _) as s) :: rest when Hashtbl.mem r.vocab c ->
        Option.to_list pending @ (s :: spell None rest)
    | (c, start, stop) :: rest
      when List.for_all (Hashtbl.mem r.vocab) (fallback c) ->
        List.map (fun b -> (b, start, stop)) (fallback c) @ spell pending rest
    | (_, start, stop) :: rest -> (
        match (unk, pending) with
        | None, _ -> spell None rest
        | Some _, Some (u, first, _) -> spell (Some (u, first, stop)) rest
        | Some u, None -> spell (Some (u, start, stop)) rest)
  in
  spell None (chars text) |> merge_all r |> tokens_of r ~word:0

(* Pre_tokenizer.byte_level *)

let walker_alphabet =
  [
    "a"; "the"; "Ab"; " "; "  "; "\n"; "\t"; "\r\n"; "'s"; "'ll"; "'"; "1";
    "42"; "."; "!!"; "-"; "\195\169"; "Ж"; "ß"; "日本"; "\u{a0}"; "\u{85}";
    "\u{3000}"; "\u{2028}"; "\u{663}"; "\u{bd}"; "\u{201c}"; "\u{201d}";
    "\u{2014}"; "😀"; "e\u{301}"; "\x80"; "\xC3"; "\xE2\x80"; "\xF0\x9F";
    "\xC0\xAF"; "\xED\xA0\x80";
    (* Overlong forms of every ASCII class, of three bytes and of four: a
       letter, a digit, a space, a newline, punctuation, an apostrophe. *)
    "\xC1\x81"; "\xE0\x81\x81"; "\xF0\x80\x81\x81"; "\xE0\x80\xB1";
    "\xF0\x80\x80\xB1"; "\xE0\x80\xA0"; "\xF0\x80\x80\xA0"; "\xE0\x80\x8A";
    "\xF0\x80\x80\x8A"; "\xE0\x80\xAE"; "\xF0\x80\x80\xAE"; "\xE0\x80\xA7";
    (* Letters and numbers of four bytes, code points no table classifies
       before they are met, and bytes that never stand in UTF-8. *)
    "\u{1D400}"; "\u{10400}"; "\u{1D7CE}"; "\u{10107}"; "\u{378}"; "\u{E0001}";
    "\u{10FFFF}"; "\u{FFFE}"; "\xF5"; "\xF8"; "\xFE"; "\xFF";
  ]
[@@ocamlformat "disable"]

let walker_text =
  quoted
    (Gen.map (String.concat "")
       (Gen.list ~size:(Gen.int_range 40 160) (Gen.of_list walker_alphabet)))

let unmerged = lazy (Brot.bpe ~vocab:base_vocab ~merges:[] ~pre:(pre ()) ())

(* Texts long enough to cross several 64-byte batches. The examples hold
   overlong forms of white space inside a batch, and white space beyond ASCII
   before a letter, which one case in fifteen draws. *)
let walker_law =
  prop ~count:300 "a byte-level BPE cuts a text where the pre-tokenizer cuts it"
    walker_text
    ~examples:
      [
        "abcdefghij\xF0\x80\x80\xA0" ^ String.make 60 'x';
        "abcdefghij \xF0\x80\x80\xA0\xF0\x80\x80\x8A" ^ String.make 60 'x';
        "abcdefghij\xE0\x80\xA0\xF0\x80\x80\xA0y" ^ String.make 60 'x';
        "x\u{a0}a \u{3000}the\u{2028}Ab" ^ String.make 60 'x';
      ]
    (fun text ->
      cover "white space beyond ASCII before a letter"
        (has_any text [ "\u{a0}a"; "\u{3000}t"; "\u{2028}A" ]);
      cover "a letter beyond ASCII after a space"
        (has_any text [ " \195\169"; " Ж"; " 日" ]);
      cover "malformed UTF-8" (not (String.is_valid_utf_8 text));
      cover "an overlong form of white space"
        (has_any text [ "\xE0\x80\xA0"; "\xF0\x80\x80\xA0" ]);
      cover "a text over two batches" (String.length text > 128);
      equal ~msg:(Printf.sprintf "%S" text)
        (array (pair int int))
        (pre_spans (pre ()) text)
        (kernel_spans (Lazy.force unmerged) text))

(* Each shape that can cross a batch's edge, planted at every offset. The
   padding is short pretokens and the tails run at least 64 bytes past the
   shape, so that the shape crosses an edge inside the batches rather than at
   the start of a new walk; the 64-byte span keeps a long run before it, which
   starts one. *)
let test_batch_offsets () =
  sweep (Lazy.force unmerged) (pre ())
    [
      ("letters across the edge", fun off -> pad off ^ " " ^ String.make 70 'y');
      ( "a span of exactly 64 bytes",
        fun off -> String.make off 'z' ^ " " ^ String.make 63 'w' ^ " t" );
      ("non-ASCII", fun off -> pad off ^ "\195\169" ^ pad 80);
      ("contractions", fun off -> pad off ^ "'ll they've o'x " ^ pad 80);
      ("white space given back", fun off -> pad off ^ "  \t\r\n  b " ^ pad 80);
      ( "white space given back before non-ASCII white space",
        fun off -> String.make off 'x' ^ "  \194\160 " ^ String.make 80 'y' );
      ("digits", fun off -> pad off ^ " " ^ String.make 20 '8' ^ "a9 " ^ pad 64);
      ( "letters beyond ASCII across the edge",
        fun off -> pad off ^ " \195\169" ^ repeat 30 "Ж" ^ "a日本" ^ pad 80 );
      ( "white space beyond ASCII at the edge",
        fun off ->
          String.make off 'x'
          ^ " \u{a0}y \u{3000}\u{3000}z\u{a0}\u{a0} 1\u{2028}"
          ^ String.make 80 'y' );
      ( "numbers and punctuation beyond ASCII",
        fun off ->
          pad off ^ " \u{663}\u{664}\u{bd} \u{201c}x\u{201d} \u{2014}\u{2014}😀a"
          ^ pad 80 );
      ( "malformed bytes",
        fun off ->
          pad off ^ "a\x80b \xC3 c\xE2\x80d\xF0\x9F\x98 \xC0\xAF\xED\xA0\x80"
          ^ pad 80 );
    ]

(* A pretoken past the cache's key planted at every span from 600 to 1400,
   across the 1024 spans the encoder records at once and the 256 a kernel call
   stages, after three shapes of short pretokens. The OCaml walk cuts it as any
   other pretoken, so there every sixteenth place suffices. *)
let test_long_pretoken_seams () =
  let tok = Lazy.force unmerged in
  let step = if Brot.kernels then 1 else 16 in
  let long = " " ^ String.make 20 'q' in
  let shapes =
    [
      ("plain", fun k -> repeat k " a");
      ( "every third longer",
        fun k ->
          String.concat ""
            (List.init k (fun i -> if i mod 3 = 0 then " ab" else " a")) );
      ( "every fifth beyond ASCII",
        fun k ->
          String.concat ""
            (List.init k (fun i -> if i mod 5 = 0 then "\u{e9}" else " a")) );
    ]
  in
  List.iter
    (fun (name, shape) ->
      for i = 0 to 800 / step do
        let k = 600 + (i * step) in
        let text = shape k ^ long ^ shape 300 in
        equal
          ~msg:(Printf.sprintf "%s, long pretoken after %d" name k)
          (array (pair int int))
          (pre_spans (pre ()) text)
          (kernel_spans tok text)
      done)
    shapes

(* Pre_tokenizer.split_regex *)

(* The cl100k pattern as Llama 3's tokenizer file spells it, and its variants
   with numbers one at a time and with combining marks as letters. *)
let cl100k =
  {re|(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+|re}

let cl100k_single_digits =
  {re|(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+|re}

let cl100k_marks =
  {re|(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+|re}

(* The o200k pattern as gpt-oss's tokenizer file spells it, and Tekken's, which
   takes numbers one at a time and no contractions. *)
let o200k_with ~numbers ~contractions =
  String.concat "|"
    [
      {re|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|re}
      ^ contractions;
      {re|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|re}
      ^ contractions;
      numbers;
      {re| ?[^\s\p{L}\p{N}]+[\r\n/]*|re};
      {re|\s*[\r\n]+|re};
      {re|\s+(?!\S)|re};
      {re|\s+|re};
    ]

let o200k =
  o200k_with ~numbers:{re|\p{N}{1,3}|re}
    ~contractions:{re|(?i:'s|'t|'re|'ve|'m|'ll|'d)?|re}

let tekken = o200k_with ~numbers:{re|\p{N}|re} ~contractions:""

(* Every rule the cl100k walk evaluates on ASCII, each white space rule (a
   newline after punctuation, a newline then spaces, the run's last newline,
   white space given back), each prefix a word takes, digit groups, contractions
   in any case, white space crossing the edge, and characters beyond ASCII that
   extend a digit or white space run. *)
let split_families =
  [
    ("letters across the edge", fun off -> pad off ^ " " ^ String.make 70 'y');
    ("non-ASCII", fun off -> pad off ^ "\195\169" ^ pad 80);
    ( "contractions",
      fun off -> pad off ^ "'ll THEY'VE o'x 'S's ''d x 'Re" ^ pad 80 );
    ("newline runs", fun off -> pad off ^ " \n \n  x\n\n\ty\r\n  z " ^ pad 80);
    ( "newlines after punctuation",
      fun off -> pad off ^ "!!\n\n x?\n \n!\r\n\t." ^ pad 80 );
    ("prefixes", fun off -> pad off ^ "\ta ?b  !c\t!d .e ..f\x0Bg" ^ pad 80);
    ( "digits",
      fun off -> pad off ^ " " ^ String.make 20 '8' ^ "a9 1234 12" ^ pad 64 );
    ( "white space across the edge",
      fun off -> pad off ^ "      \n     \n  y \t \n\t z" ^ pad 80 );
    ( "non-ASCII digits and white space",
      fun off ->
        pad off ^ "\217\16312345 \227\128\130\n\n  x\194\160\n y \194\160 !"
        ^ pad 80 );
    ( "white space given back before non-ASCII white space",
      fun off -> String.make off 'x' ^ "  \194\160 " ^ String.make 80 'y' );
    ( "spaces after a newline, then non-ASCII white space and newlines",
      fun off -> pad off ^ "  \n  \194\160\n\n\n x" ^ pad 80 );
  ]

(* The o200k walk's own rules on top: runs of upper then lower case across the
   edge, letters of both its classes, contractions after a case run, and slashes
   after punctuation. *)
let o200k_families =
  split_families
  @ [
      ( "a case run across the edge",
        fun off ->
          pad off ^ " XMLHttp" ^ repeat 20 "Request" ^ "iPhone" ^ pad 80 );
      ( "letters of both classes",
        fun off ->
          pad off
          ^ " \xC3\x89COLE\xC3\xA9cole \xC7\x85ungla \xC7\x84A \
             \xE6\x97\xA5\xE6\x9C\xACA A\xCA\xB0B \xCC\x81Z !\xCC\x81Z \
             e\xCC\x81\xCC\x81E" ^ pad 80 );
      ( "contractions after a case run",
        fun off -> pad off ^ " eBay's McDONALD'S OK'd won't' x'LL " ^ pad 80 );
      ( "slashes after punctuation",
        fun off ->
          pad off ^ " a/b http://x.y/z \\/ //\n// !/\r\n/a !!/\n/x" ^ pad 80 );
    ]

(* The walk is the same at every offset, so the OCaml one takes every eighth. *)
let split_step = if Brot.kernels then 1 else 8

let test_split_offsets patterns families () =
  List.iter
    (fun pattern ->
      let tok =
        Brot.bpe ~vocab:base_vocab ~merges:[] ~pre:(pre_split pattern) ()
      in
      sweep ~step:split_step tok (regex_split pattern) families)
    patterns

(* bpe *)

(* A byte-level model with merges for " a", "ab" and " ab", whose ids follow the
   bytes', and the same without an id for "q". *)
let bl_merges = [ ("a", "b"); ("\196\160", "a"); ("\196\160", "ab") ]

let bl_vocab ~missing =
  let extra = [ ("ab", 256); ("\196\160a", 257); ("\196\160ab", 258) ] in
  List.filter (fun (s, _) -> not (missing && s = "q")) base_vocab @ extra

let gpt2_pattern =
  {|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+|}

(* Short pretokens that the cache holds, pretokens of 15 bytes, one of which no
   merge shortens, pretokens past the key, characters a kernel classifies as it
   first meets them, and "q", which one of the vocabularies has no id for. *)
let bl_alphabet =
  [
    " a"; " a"; " a"; " ab"; "ab"; " q"; "q"; " qa"; "aqb"; "1"; " 42"; "\n";
    "  "; "abcdefghijklmno"; "bcdefghijklmnop"; " bbbbbbbbbbbbbbbbbbbb"; "\u{2723}"; "\u{a0}";
    "\u{3b1}"; "\u{660}";
  ]
[@@ocamlformat "disable"]

let bl_case =
  let pp ppf (text, missing, cache) =
    Format.fprintf ppf "%S, %s, cache_capacity %s" text
      (if missing then "no id for q" else "every byte")
      (match cache with None -> "default" | Some n -> string_of_int n)
  in
  Gen.with_pp pp
    Gen.(
      let+ text =
        map (String.concat "")
          (list ~size:(int_range 0 600) (of_list bl_alphabet))
      and+ missing = bool
      and+ cache = of_list [ None; Some 0; Some 2 ] in
      (text, missing, cache))

(* The edges a kernel meets: a document crossing the spans a call stages and the
   1024 the encoder records at once; pretoken counts around the 256 a call
   stages; a pretoken past the key at the first and the last span of a call, and
   between two runs that cross a call; a character classified right after the
   walk resumes; pieces of exactly 15 bytes; one span far larger than the ids a
   call holds; caching off; a byte without an id; characters classified as they
   are met. *)
let bl_examples =
  let long = " " ^ String.make 20 'b' in
  List.map
    (fun text -> (text, false, None))
    ([ repeat 1500 " a"; repeat 600 " a" ^ long ^ repeat 600 " a" ]
    @ List.map
        (fun k -> repeat k " a")
        [ 1; 63; 64; 65; 127; 128; 129; 255; 256; 257; 258; 511; 512; 513 ]
    @ List.map
        (fun k -> repeat k " a" ^ long ^ repeat 300 " a")
        [ 0; 62; 63; 64; 65; 254; 255; 256; 257 ]
    @ [
        repeat 255 " a" ^ long ^ "\u{2723} a";
        String.concat "1" (List.init 800 (fun _ -> "bcdefghijklmnop"));
        String.make 8000 'c';
        " a\194\160\206\177\217\160 a";
      ])
  @ [ (repeat 600 " a", false, Some 0); ("aqb", true, None) ]

let bl_references =
  lazy
    (Array.map
       (fun missing ->
         byte_level_reference ~pattern:gpt2_pattern
           (reference_of_lists (bl_vocab ~missing) bl_merges))
       [| false; true |])

let bl_law =
  prop ~count:100
    "a byte-level model encodes each pretoken to its reference merge" bl_case
    ~examples:bl_examples (fun (text, missing, cache_capacity) ->
      let tok =
        Brot.bpe ~vocab:(bl_vocab ~missing) ~merges:bl_merges ~pre:(pre ())
          ?cache_capacity ()
      in
      let expect = (Lazy.force bl_references).(Bool.to_int missing) text in
      let dropped = missing && String.contains text 'q' in
      let pretokens =
        List.length (Brot.Pre_tokenizer.pre_tokenize (pre ()) text)
      in
      cover "more pretokens than a call stages" (pretokens > 256);
      cover "a pretoken past the key" (has_sub text "bbbbbbbbbbbbbbbb");
      cover "a character classified as it is met"
        (has_any text [ "\u{2723}"; "\u{3b1}"; "\u{660}" ]);
      cover "a byte without an id" dropped;
      cover "caching off" (cache_capacity = Some 0);
      cover "a cache of one set" (cache_capacity = Some 2);
      let e = encode tok text in
      equal ~msg:"ids" (array int) (ids_of expect) (ids tok text);
      equal ~msg:"ids of the encoding" (array int) (ids_of expect)
        (Brot.Encoding.ids e);
      equal ~msg:"word ids"
        (array (option int))
        (words_of expect) (Brot.Encoding.word_ids e);
      (* Which token's span takes in a byte without an id is not stated. *)
      if not dropped then
        equal ~msg:"offsets"
          (array (pair int int))
          (offsets_of expect) (Brot.Encoding.offsets e))

(* A SentencePiece model drawn from four choices: pieces for runs of ▁ and
   merges for them, a piece across a full stop, fallback tokens for every byte,
   and a cache. ▁ is 0, a to z 1 to 26, "." 27, "," 28, "é" 29, "▁a" 30. *)
let sp_vocab =
  (sp_mark, 0)
  :: List.init 26 (fun i -> (String.make 1 (Char.chr (0x61 + i)), 1 + i))
  @ [ (".", 27); (",", 28); ("\195\169", 29); (sp_mark ^ "a", 30) ]

type sp_model = { runs : bool; dot : bool; fallback : bool; cache : int option }

let simple = { runs = false; dot = false; fallback = false; cache = None }

let sp_lists m =
  let vocab =
    sp_vocab
    @ (if m.runs then [ (sp_mark ^ sp_mark, 31) ] else [])
    @ (if m.dot then [ ("a.", 32) ] else [])
    @
    if m.fallback then
      List.init 256 (fun b -> (Printf.sprintf "<0x%02X>" b, 40 + b))
    else []
  in
  let merges =
    (if m.runs then [ (sp_mark, sp_mark) ] else [])
    @ [ (sp_mark, "a") ]
    @ if m.dot then [ ("a", ".") ] else []
  in
  (vocab, merges)

let sp_tokenizer m =
  let vocab, merges = sp_lists m in
  Brot.bpe ~vocab ~merges ~byte_fallback:m.fallback ?cache_capacity:m.cache ()

(* Units of the ▁ walk and their neighbours: runs of ▁, units past the key,
   punctuation and a piece across a full stop; and, for a model with fallback
   tokens, characters without a piece. *)
let sp_alphabet =
  [
    sp_mark ^ "a"; sp_mark ^ "a"; sp_mark; sp_mark ^ sp_mark; "a"; "b"; "c";
    "."; ","; "a."; "\195\169"; sp_mark ^ "bbbbbbbbbbbbbbbbbbbb";
    "bbbbbbbbbbbbbbbbbbbb";
  ]
[@@ocamlformat "disable"]

let pieceless = [ "\195\188"; "😀" ]

let sp_case =
  let pp ppf (text, m) =
    Format.fprintf ppf "%S, runs %b, dot %b, fallback %b, cache_capacity %s"
      text m.runs m.dot m.fallback
      (match m.cache with None -> "default" | Some n -> string_of_int n)
  in
  Gen.with_pp pp
    Gen.(
      let* fallback = bool in
      let alphabet =
        if fallback then sp_alphabet @ pieceless else sp_alphabet
      in
      let+ text =
        map (String.concat "") (list ~size:(int_range 0 60) (of_list alphabet))
      and+ runs = bool
      and+ dot = bool
      and+ cache = of_list [ None; Some 0; Some 2 ] in
      (text, { runs; dot; fallback; cache }))

(* The edges of the ▁ walk: a stream of cached units crossing the spans a call
   stages; a unit past the key between short ones; a character spelled by its
   fallback tokens, met twice so that the second is served from the cache, and
   then followed by a run of ▁ where the walk resumes; a unit ending at a full
   stop after a letter, with and without a piece across it; caching off. *)
let sp_examples =
  [
    (repeat 1500 (sp_mark ^ "a"), simple);
    ( repeat 3 (sp_mark ^ "a")
      ^ sp_mark ^ String.make 20 'b'
      ^ repeat 3 (sp_mark ^ "a"),
      simple );
    ( repeat 2 (sp_mark ^ "a" ^ sp_mark ^ "\195\188"),
      { simple with fallback = true } );
    ( sp_mark ^ "\195\188" ^ repeat 3 sp_mark ^ "a" ^ sp_mark ^ "a",
      { simple with runs = true; fallback = true } );
    (sp_mark ^ "ba.b" ^ sp_mark ^ "c" ^ repeat 2 (sp_mark ^ "a"), simple);
    ( sp_mark ^ "ba.b" ^ sp_mark ^ "c" ^ repeat 2 (sp_mark ^ "a"),
      { simple with dot = true } );
    (repeat 600 (sp_mark ^ "a"), { simple with cache = Some 0 });
  ]

let sp_law =
  prop ~count:300
    "a SentencePiece model encodes its text to the reference merge" sp_case
    ~examples:sp_examples (fun (text, m) ->
      let vocab, merges = sp_lists m in
      let expect = piece_reference (reference_of_lists vocab merges) text in
      cover "a run of ▁" (has_sub text (sp_mark ^ sp_mark));
      cover "a unit past the key" (has_sub text "bbbbbbbbbbbbbbbb");
      (* After ▁, "a" merges with it first, so only "a." after a letter forms
         the piece. *)
      cover "a piece across a full stop"
        (m.dot && has_any text [ "aa."; "ba."; "ca." ]);
      cover "a character spelled by its fallback tokens"
        (m.fallback && has_any text pieceless);
      cover "caching off" (m.cache = Some 0);
      let tok = sp_tokenizer m in
      let e = encode tok text in
      equal ~msg:"ids" (array int) (ids_of expect) (ids tok text);
      equal ~msg:"ids of the encoding" (array int) (ids_of expect)
        (Brot.Encoding.ids e);
      equal ~msg:"offsets"
        (array (pair int int))
        (offsets_of expect) (Brot.Encoding.offsets e);
      equal ~msg:"word ids"
        (array (option int))
        (words_of expect) (Brot.Encoding.word_ids e))

(* A character with no piece, no fallback token and no unknown token gives no
   token: Hugging Face's tokenizers 0.23.1 gives these ids. *)
let test_sp_pieceless () =
  let text = sp_mark ^ "a" ^ sp_mark ^ "\195\188" ^ repeat 2 (sp_mark ^ "a") in
  equal (array int) [| 30; 0; 30; 30 |] (ids (sp_tokenizer simple) text)

(* 100000 units outgrow the ids one call holds, so the walk stops at a unit and
   the next call resumes it. "▁a" is a piece, and the merge of ▁ and a. *)
let test_sp_long_stream () =
  let text = repeat 100_000 (sp_mark ^ "a") in
  equal (array int) (Array.make 100_000 30) (ids (sp_tokenizer simple) text)

(* A merge whose entry stands for other bytes than the symbols it joins: the
   bytes C4 and 8A merge into "Ċ", the entry of the byte "\n", and the pieces
   "<", "0", "x", "4", "1", ">" merge into "<0x41>", the fallback token of "A".
   Every token spans the bytes it joined. *)
let disagreeing_byte_level =
  lazy
    (Brot.bpe
       ~vocab:[ ("\196", 0); ("\138", 1); ("\196\138", 2); ("a", 3) ]
       ~merges:[ ("\196", "\138") ]
       ~pre:(pre ()) ())

let disagreeing_sp =
  lazy
    (let spell =
       [ "<"; "0"; "4"; "1"; ">"; "<0"; "<0x"; "<0x4"; "1>"; "<0x41>" ]
     in
     Brot.bpe
       ~vocab:(sp_vocab @ List.mapi (fun i s -> (s, 31 + i)) spell)
       ~merges:
         [
           (sp_mark, "a");
           ("<", "0");
           ("<0", "x");
           ("<0x", "4");
           ("1", ">");
           ("<0x4", "1>");
         ]
       ~byte_fallback:true ())

let disagreeing =
  [
    ("byte-level, alone", disagreeing_byte_level, "\196\138", [ (2, (0, 2)) ]);
    ( "byte-level, a token after",
      disagreeing_byte_level,
      "\196\138a",
      [ (2, (0, 2)); (3, (2, 3)) ] );
    ( "byte-level, two inside a pretoken",
      disagreeing_byte_level,
      "a\196\138\196\138a",
      [ (3, (0, 1)); (2, (1, 3)); (2, (3, 5)); (3, (5, 6)) ] );
    ( "SentencePiece",
      disagreeing_sp,
      sp_mark ^ "a<0x41>" ^ sp_mark ^ "b" ^ sp_mark ^ "c",
      [
        (30, (0, 4));
        (40, (4, 10));
        (0, (10, 13));
        (2, (13, 14));
        (0, (14, 17));
        (3, (17, 18));
      ] );
  ]

let test_disagreeing =
  cases "the offsets of a merge whose entry stands for other bytes span them"
    ~id:(fun (id, _, _, _) -> id)
    disagreeing
    (fun (_, tok, text, expect) ->
      let e = encode (Lazy.force tok) text in
      equal
        (list (pair int (pair int int)))
        expect
        (List.combine
           (Array.to_list (Brot.Encoding.ids e))
           (Array.to_list (Brot.Encoding.offsets e))))

(* "abc" is an entry the merges never build, and "ab" one they do. The pretokens
   miss the cache, are looked up whole and stored, then hit; a cache of one set
   evicts them, so they miss again; with no cache each is looked up whole. *)
let test_ignore_merges =
  let vocab =
    base_vocab
    @ [ ("\196\160a", 256); ("ab", 257); ("abc", 258); ("\196\160abc", 259) ]
  in
  let text = "abc\nab" ^ repeat 50 " abc abcd ab" in
  let expect =
    Array.concat
      ([| 258; 10; 257 |]
      :: List.init 50 (fun _ -> [| 259; 256; 98; 99; 100; 256; 98 |]))
  in
  cases
    "under ignore_merges a pretoken in the vocabulary is that entry, cached, \
     evicted or uncached"
    ~id:fst
    [ ("cached", None); ("evicted", Some 2); ("uncached", Some 0) ]
    (fun (_, cache_capacity) ->
      let tok =
        Brot.bpe ~vocab
          ~merges:[ ("\196\160", "a"); ("a", "b") ]
          ~pre:(pre ()) ?cache_capacity ~ignore_merges:true ()
      in
      equal (array int) expect (ids tok text))

(* A SentencePiece pipeline writes ▁ for every space right before the model, in
   its normalizer or in a metaspace that does not split; a BPE model then reads
   the spaces themselves, so the text it is handed keeps them. The same pipeline
   with a stage that changes nothing after the marking (a strip of neither side,
   a sequence of the one metaspace) writes the same text and is read marked, and
   both must give the same tokens, offsets included. The vocabulary merges
   across ▁ runs, spells "ü" with fallback bytes and has no piece for "😀"; one
   added token is matched in the raw text and one in the normalized text. *)
let spaced_vocab =
  sp_vocab
  @ [
      (sp_mark ^ sp_mark, 31); (sp_mark ^ "t", 32); ("he", 33);
      (sp_mark ^ "the", 34); ("<0xC3>", 35); ("<0xBC>", 36); ("<unk>", 37);
      ("<s>", 38);
    ]
[@@ocamlformat "disable"]

let spaced_merges =
  [ (sp_mark, "a"); (sp_mark, sp_mark); (sp_mark, "t"); ("h", "e");
    (sp_mark ^ "t", "he") ]
[@@ocamlformat "disable"]

let spaced_alphabet =
  [ "a"; "the"; "q q"; "b"; " "; "  "; "   "; sp_mark; sp_mark ^ sp_mark;
    " " ^ sp_mark; "."; ","; "a.b"; "\195\169"; "\195\188"; "\240\159\152\128";
    "\n"; "<s>"; "aaaaaaaaaaaaaaaaaaaa"; " xxxxxxxxxxxxxxx";
    (* Bytes that open no whole character: leads without their continuation
       bytes, a lone continuation byte, and a lead cut short. *)
    "\233"; "\195"; "\226"; "\226\150"; "\240\159"; "\128"; "\150\129";
    "zz"; " zz " ]
[@@ocamlformat "disable"]

let spaced_case =
  let pp ppf (text, pipeline, byte_fallback) =
    let names =
      [|
        "Llama's normalizer";
        "metaspace, first";
        "metaspace, always";
        "metaspace, never";
        "prepend after the marking";
        "Llama's normalizer, stripping added token";
      |]
    in
    Format.fprintf ppf "%S, %s, byte_fallback %b" text names.(pipeline)
      byte_fallback
  in
  Gen.with_pp pp
    Gen.(
      let+ text =
        map (String.concat "")
          (list ~size:(int_range 0 60) (of_list spaced_alphabet))
      and+ pipeline = int_range 0 5
      and+ byte_fallback = bool in
      (text, pipeline, byte_fallback))

(* The spaced pipeline and its marked twin: the normalizer Llama spells, a
   metaspace prepending on [`First], on [`Always] and never, the marking
   followed by a prepend, and Llama's normalizer under a normalized added token
   that takes in the white space beside it, which keeps its pipeline marked:
   read spaced, it would take in the spaces too. *)
let spaced_pair pipeline ~byte_fallback =
  let stripping =
    if pipeline = 5 then
      [
        Brot.added_token ~special:false ~normalized:true ~lstrip:true
          ~rstrip:true "zz";
      ]
    else []
  in
  let make ?normalizer ?pre () =
    Brot.bpe ?normalizer ?pre ~vocab:spaced_vocab ~merges:spaced_merges
      ~unk_token:"<unk>" ~byte_fallback ~fuse_unk:true
      ~added_tokens:
        ([
           Brot.added_token ~special:true "<s>";
           Brot.added_token ~special:false ~normalized:true "q q";
         ]
        @ stripping)
      ()
  in
  let marking =
    [
      Brot.Normalizer.prepend sp_mark;
      Brot.Normalizer.replace ~pattern:" " ~replacement:sp_mark;
    ]
  in
  let noop = Brot.Normalizer.strip ~left:false ~right:false () in
  let metaspace prepend_scheme =
    Brot.Pre_tokenizer.metaspace ~prepend_scheme ~split:false ()
  in
  let twins marking =
    ( make ~normalizer:(Brot.Normalizer.sequence marking) (),
      make ~normalizer:(Brot.Normalizer.sequence (marking @ [ noop ])) () )
  in
  match pipeline with
  | 0 | 5 -> twins marking
  | 4 -> twins (List.rev marking)
  | n ->
      let scheme = List.nth [ `First; `Always; `Never ] (n - 1) in
      ( make ~pre:(metaspace scheme) (),
        make ~pre:(Brot.Pre_tokenizer.sequence [ metaspace scheme ]) () )

let batch_rows tok docs exec =
  let buf, lengths =
    Brot.encode_batch_ids tok ~add_special_tokens:false ?exec docs
  in
  let pos = ref 0 in
  Array.to_list lengths
  |> List.map (fun n ->
      let row =
        Array.init n (fun k ->
            Int32.to_int (Bigarray.Array1.get buf (!pos + k)))
      in
      pos := !pos + n;
      row)

(* The example holds the added token taking in white space, which one case in
   forty draws. *)
let spaced_law =
  prop ~count:300 "a spaced pipeline encodes as a marked one" spaced_case
    ~examples:[ ("a  zz b", 5, false) ]
    (fun (text, pipeline, byte_fallback) ->
      let spaced, marked = spaced_pair pipeline ~byte_fallback in
      cover "a space run" (has_sub text "  ");
      cover "a ▁ next to a space" (has_sub text (" " ^ sp_mark));
      cover "a unit past the key" (has_sub text "aaaaaaaaaaaaaaaaaaaa");
      cover "a character with fallback bytes" (has_sub text "\195\188");
      cover "a byte that opens no whole character"
        (not (String.is_valid_utf_8 text));
      cover "an added token" (has_sub text "<s>" || has_sub text "q q");
      cover "a prepend after the marking" (pipeline = 4);
      cover "an added token taking in white space"
        (pipeline = 5 && has_sub text "  zz");
      cover "a text that opens with a space"
        (String.length text > 0 && text.[0] = ' ');
      let a = encode spaced text and b = encode marked text in
      let msg what = Printf.sprintf "%S: %s" text what in
      equal ~msg:(msg "ids") (array int) (Brot.Encoding.ids b)
        (Brot.Encoding.ids a);
      equal ~msg:(msg "tokens") (array string) (Brot.Encoding.tokens b)
        (Brot.Encoding.tokens a);
      equal ~msg:(msg "offsets")
        (array (pair int int))
        (Brot.Encoding.offsets b) (Brot.Encoding.offsets a);
      equal ~msg:(msg "rows of a batch")
        (list (array int))
        [ Brot.Encoding.ids b; Brot.Encoding.ids b ]
        (batch_rows spaced [ text; text ] None))

(* A byte that opens no whole character is a character of its own, so the bytes
   after it, here the space Llama's normalizer marks, stay what they are:
   decoded, the stray byte is the replacement character and the rest of the text
   comes back. *)
let test_stray_lead_byte =
  cases "a stray lead byte takes in nothing after it" ~id:fst
    [
      ("caf\xE9 au lait", "caf\xEF\xBF\xBD au lait");
      ("a\xF0\x9F b", "a\xEF\xBF\xBD\xEF\xBF\xBD b");
      ("\xC3 x\x80 y", "\xEF\xBF\xBD x\xEF\xBF\xBD y");
    ]
    (fun (text, decoded) ->
      Fixture.with_tokenizer_file "llama" (fun path ->
          match Brot.from_file path with
          | Error e -> failf "%s: %s" path e
          | Ok tok -> equal string decoded (Brot.decode tok (ids tok text))))

(* A cache grows to hold the vocabulary's short entries, then doubles while the
   kernels fill it, from their stores and from OCaml's, under the tables the
   kernels read. Words of one to six letters, which the letter pairs merge to at
   most four ids, and some past the key, encoded three times over, give the ids
   an uncached tokenizer gives. *)
let grown_letters = List.init 6 (fun i -> String.make 1 (Char.chr (97 + i)))

let grown_pairs =
  List.concat_map
    (fun a -> List.map (fun b -> (a, b)) grown_letters)
    grown_letters

let grown_vocab first extra =
  first
  @ List.mapi
      (fun i (a, b) -> (a ^ b, List.length first + i))
      (extra @ grown_pairs)

let grown_byte_level cache_capacity =
  let extra = List.map (fun l -> ("\196\160", l)) grown_letters in
  Brot.bpe
    ~vocab:(grown_vocab base_vocab extra)
    ~merges:(grown_pairs @ extra) ~pre:(pre ()) ?cache_capacity ()

let grown_sp cache_capacity =
  Brot.bpe ~vocab:(grown_vocab sp_vocab []) ~merges:grown_pairs ?cache_capacity
    ()

let spaced words = String.concat " " words
let marked words = String.concat "" (List.map (( ^ ) sp_mark) words)

let grown_case =
  let open Gen in
  let word size = string_of ~size (char_range 'a' 'f') in
  with_pp
    (fun ppf words -> Format.fprintf ppf "%d words" (List.length words))
    (list ~size:(int_range 1000 2000)
       (frequency [ (20, word (int_range 1 6)); (1, word (constant 18)) ]))

let grown_law =
  prop ~count:10 "a cache grown by the kernels finds what it held" grown_case
    (fun words ->
      (* The seeded table doubles past 512 entries, as test_cache_doubles
         shows. *)
      cover "more distinct words than the seeded table holds before it doubles"
        (List.length (List.sort_uniq String.compare words) > 512);
      List.iter
        (fun (msg, make, text) ->
          let cached = make None and expect = ids (make (Some 0)) text in
          for round = 1 to 3 do
            equal
              ~msg:(Printf.sprintf "%s, round %d" msg round)
              (array int) expect (ids cached text)
          done)
        [
          ("byte-level", grown_byte_level, spaced words);
          ("SentencePiece", grown_sp, marked words);
        ])

let live_words () =
  Gc.full_major ();
  (Gc.stat ()).live_words

(* All 1296 words of four letters from a to f, which the kernels keep without
   handing them back to OCaml, are more than the seeded table holds, so it
   doubles twice, where as many copies of one word leave it seeded. Its first
   doubling adds 2048 live words, so more than 4096 shows the second. *)
let distinct_words =
  List.init 1296 (fun i ->
      String.init 4 (fun k ->
          Char.chr (97 + (i / int_of_float (6. ** float_of_int k) mod 6))))

let test_cache_doubles =
  cases "a cache doubles as the kernels fill half of it"
    ~id:(fun (id, _, _) -> id)
    [
      ("byte-level", grown_byte_level, spaced);
      ("SentencePiece", grown_sp, marked);
    ]
    (fun (_, make, join) ->
      let held words =
        let t = make None in
        ignore (ids t (join words) : int array);
        Brot.hold_cache t live_words
      in
      let one = held (List.map (fun _ -> "aaaa") distinct_words) in
      greater ~msg:"live words while the cache is held" int ~than:(one + 4096)
        (held distinct_words))

(* A cache that doubles after the last kernel call of an encoding, as one call
   over 900 distinct words of the seeded table's 1024 entries does, is released
   holding its new table alone. Encoding one word more, whose call takes the
   kernels' tables from the cache again, then frees nothing. *)
let test_released_state () =
  let words =
    List.init 900 (fun i ->
        String.init 5 (fun k ->
            Char.chr (97 + (i / int_of_float (6. ** float_of_int k) mod 6))))
  in
  let t = grown_byte_level None in
  ignore (ids t (spaced words) : int array);
  let released = Brot.hold_cache t live_words in
  ignore (ids t "a" : int array);
  let refreshed = Brot.hold_cache t live_words in
  at_most ~msg:"live words of the released state" int ~than:(refreshed + 512)
    released

(* A tokenizer file and the reference read from it, once, on the first case. *)
let law ~text ~cover name reference =
  let loaded =
    lazy
      (Fixture.with_tokenizer_file name (fun path ->
           match Brot.from_file path with
           | Ok tok -> (tok, reference (reference_of path))
           | Error e -> failf "%s: %s" path e))
  in
  prop ~count:300
    (Printf.sprintf
       "%s encodes a text as the reference merge, alone and in a batch" name)
    text (fun text ->
      let tok, reference = Lazy.force loaded in
      let expect = ids_of (reference text) in
      cover text;
      equal ~msg:"alone" (array int) expect (ids tok text);
      equal ~msg:"in a batch"
        (list (array int))
        [ expect; expect ]
        (batch_rows tok [ text; text ] None))

(* Words of every length up to past the key, repeated characters, contractions,
   digits, the punctuation SentencePiece cuts at, every kind of white space,
   letters beyond ASCII, characters no trimmed vocabulary holds, and ▁
   itself. *)
let law_alphabet =
  [
    "the"; "and"; "of"; "tokenizer"; "information"; "internationalization";
    "Hello"; "WORLD"; "aaaaaaaaaaaaaaaaaaaaaaaa"; "!!!!!!!!"; "don't";
    "it's"; "we'll"; "they're"; "I'm"; "'"; "1"; "2024"; "3.14159265358979";
    "."; ","; "\""; ")"; ";"; ":"; "!"; "?"; "("; "-"; "..."; " "; " ";
    " "; "  "; "\n"; "\n\n"; "\t"; " \r\n"; "café"; "naïve"; "Ж"; "Привет";
    "日本語"; "😀"; "e\u{0301}"; "ß"; "\u{2581}"; "\u{2581}\u{2581}";
  ]
[@@ocamlformat "disable"]

let law_text = text_of ~size:40 law_alphabet

let cover_law text =
  let words = String.split_on_char ' ' text in
  cover "a word past the key"
    (List.exists (fun w -> String.length w > 15) words);
  cover "a run of one character" (has_sub text "aaaaaaaa");
  cover "a contraction" (has_sub text "'s" || has_sub text "'ll");
  cover "a character beyond ASCII" (String.exists (fun c -> c >= '\x80') text);
  cover "a character no piece holds" (has_sub text "😀");
  cover "a ▁ in the text" (has_sub text sp_mark);
  cover "a split punctuation byte" (has_sub text "." || has_sub text "?");
  cover "white space other than a space"
    (String.exists (fun c -> c = '\n' || c = '\t') text)

(* The cl100k pattern's own cases on top: contractions in any case, digit
   groups, every white space rule, punctuation before newlines, the prefixes a
   word takes, and characters beyond ASCII that join digits, white space and
   punctuation; texts long enough to cross several 64-byte batches. *)
let cl100k_alphabet =
  law_alphabet
  @ [
      "'S"; "'LL"; "'Re"; "'VE"; "'D"; "''"; "12"; "1234567"; "\r\n"; " \n ";
      "  \n"; "\n\n  "; "\t?"; "?!"; ".\n\n"; "\x0B"; "\u{a0}"; "\u{2019}";
      "\u{663}"; "\u{3002}"; "\u{3000}"; "a\u{301}";
    ]
[@@ocamlformat "disable"]

let cl100k_text = text_of ~size:80 cl100k_alphabet

let cover_cl100k text =
  cover "a contraction in upper case" (has_sub text "'S" || has_sub text "'LL");
  cover "a group of three digits" (has_sub text "1234567");
  cover "a newline then spaces" (has_sub text "\n ");
  cover "punctuation before a newline" (has_sub text ".\n");
  cover "a text over two batches" (String.length text > 128)

(* Mistral marks spaces with ▁ and prepends one to a text that does not open
   with it, then merges the whole text. *)
let mistral_reference r text =
  if text = "" then []
  else
    let marked = String.concat sp_mark (String.split_on_char ' ' text) in
    let marked =
      if String.starts_with ~prefix:sp_mark marked then marked
      else sp_mark ^ marked
    in
    piece_reference ~unk:"<unk>" r marked

(* encode_batch_ids *)

(* A chunk writes the kernels' ids straight into its buffer, while added tokens
   and the pretokens handed back to OCaml go through a spill flushed before
   every kernel call and at the end of a document. Each document interleaves the
   three, so a flush out of order or a lost spill shows as a row that is not
   what encode_ids gives for that text alone. *)
let spill_docs =
  [
    ( "byte-level",
      lazy
        (Brot.bpe
           ~vocab:(base_vocab @ [ ("\196\160a", 256) ])
           ~merges:[ ("\196\160", "a") ]
           ~pre:(pre ())
           ~added_tokens:[ Brot.added_token "<sep>" ]
           ()),
      (* Enough ids that the buffer outgrows its first size, and a span of 70000
         letters whose room check grows it at a stuck resume before handing the
         span back. *)
      let long = " " ^ String.make 20 'b' in
      [
        repeat 3000 " a" ^ "<sep>" ^ long ^ "<sep>" ^ repeat 3000 " a";
        "<sep>";
        "";
        repeat 5 " a" ^ long;
        " " ^ String.make 70_000 'x';
      ] );
    ( "SentencePiece",
      lazy
        (Brot.bpe ~vocab:sp_vocab
           ~merges:[ (sp_mark, "a") ]
           ~added_tokens:[ Brot.added_token "<sep>" ]
           ()),
      (* One document ends on a hand-back, so its last flush is the one at the
         end of the document. *)
      let long = sp_mark ^ String.make 20 'b' in
      [
        repeat 3000 (sp_mark ^ "a")
        ^ "<sep>" ^ long
        ^ repeat 3000 (sp_mark ^ "a");
        "<sep>";
        repeat 2 (sp_mark ^ "a") ^ long;
      ] );
  ]

let test_spill =
  cases
    "a row holds what encode_ids gives around added tokens and long pretokens, \
     on one domain and on four"
    ~id:(fun (id, _, _) -> id)
    spill_docs
    (fun (_, tok, docs) ->
      let tok = Lazy.force tok in
      let expect = List.map (ids tok) docs in
      equal ~msg:"on the calling domain"
        (list (array int))
        expect (batch_rows tok docs None);
      equal ~msg:"on four domains"
        (list (array int))
        expect
        (batch_rows tok docs (Some (Fixture.spawn 4))))

(* hold_cache *)

(* Every word of one to three letters is an entry its letters spell, so the
   18278 entries are seeded into the cache, which doubles to 65536 entries, 2
   MB, to hold them. *)
let seeded_bpe () =
  let letters = List.init 26 (fun i -> String.make 1 (Char.chr (97 + i))) in
  let longer words =
    List.concat_map (fun w -> List.map (( ^ ) w) letters) words
  in
  let two = longer letters in
  let vocab = List.mapi (fun i w -> (w, i)) (letters @ two @ longer two) in
  Brot.bpe ~vocab ~merges:[] ()

(* Held three deeper than [Domain.recommended_domain_count ()] on one domain:
   the holds past that run on spares the model does not keep. *)
let test_states_bounded () =
  let most = Domain.recommended_domain_count () in
  let t = seeded_bpe () in
  let before = live_words () in
  let one = Brot.hold_cache t live_words - before in
  let rec nest depth =
    if depth = 0 then live_words ()
    else Brot.hold_cache t (fun () -> nest (depth - 1))
  in
  let peak = nest (most + 3) in
  let after = live_words () in
  ignore (Sys.opaque_identity t : Brot.t);
  greater ~msg:"live words of one state" int ~than:(1 lsl 17) one;
  at_most ~msg:"live words while every hold runs" int
    ~than:(before + (most * one) + 65536)
    peak;
  at_most ~msg:"live words once every hold returns" int
    ~than:(before + (most * one) + 65536)
    after

(* A kept state holds the 2 MB cache, so a dropped one shows as the live words
   that remain. *)
let test_raised_state_dropped () =
  let t = seeded_bpe () in
  let before = live_words () in
  raises ~msg:"the exception passes" Exit (fun () ->
      Brot.hold_cache t (fun () -> raise Exit));
  let after = live_words () in
  ignore (Sys.opaque_identity t : Brot.t);
  at_most ~msg:"live words after the raise" int ~than:(before + 65536) after

(* cache_set *)

let byte_string n =
  Gen.string_of ~size:(Gen.constant n) (Gen.map Char.chr (Gen.int_range 0 255))

let distinct_sets ~sets keys =
  List.length
    (List.sort_uniq Int.compare (List.map (Brot.cache_set ~sets) keys))

(* Mistral's ▁th words took one set before f7ef81edb6 and evicted each other. *)
let test_marked_words () =
  let words =
    List.map (( ^ ) sp_mark)
      [ "the"; "they"; "them"; "their"; "there"; "these"; "then" ]
  in
  equal ~msg:"sets of the ▁th words" int (List.length words)
    (distinct_sets ~sets:131072 words)

(* [n] keys after a stem of 5 to 8 bytes, told apart by their first byte past
   the stem and ending in 0 to 6 more bytes. *)
let family n =
  let open Gen in
  let* stem_len = int_range 5 8 in
  let+ stem = byte_string stem_len
  and+ tails =
    list ~size:(constant n) (bind (int_range 0 (14 - stem_len)) byte_string)
  in
  (stem, List.mapi (fun k t -> stem ^ String.make 1 (Char.chr k) ^ t) tails)

let stem_pp =
  Gen.with_pp (fun ppf (stem, _) -> Format.fprintf ppf "stem %S" stem)

(* Eight keys over the 131072 sets of a cache of the default size share one with
   odds of about 2 in 10^4 when the index draws on all their bytes; asking for
   six sets keeps a run from failing by chance. *)
let spread_large =
  prop ~count:300
    "keys sharing their first 5 to 8 bytes spread over 131072 sets"
    (stem_pp (family 8))
    (fun (_, keys) ->
      at_least ~msg:"distinct sets of eight keys" int ~than:6
        (distinct_sets ~sets:131072 keys))

(* A cache starts at 256 sets. 256 keys drawn uniformly fill about 162 of them,
   with a standard deviation of about 5; asking for 128 keeps a run from failing
   by chance. *)
let spread_small =
  prop ~count:50 "keys sharing their first 5 to 8 bytes spread over 256 sets"
    (stem_pp (family 256))
    (fun (_, keys) ->
      at_least ~msg:"distinct sets of 256 keys" int ~than:128
        (distinct_sets ~sets:256 keys))

(* cache_hashes *)

(* The key words of every pretoken a cache holds, and any two words, whatever
   their top bits and length tag. *)
let key_words =
  let open Gen in
  let pretoken lo hi =
    map Brot.cache_key (bind (int_range lo hi) byte_string)
  in
  let tagged =
    map (fun n -> Int64.shift_left (Int64.of_int n) 56) (int_range 0 255)
  in
  let labelled label = map (fun (k0, k1) -> (label, k0, k1)) in
  with_pp
    (fun ppf (label, k0, k1) -> Format.fprintf ppf "%s: %Lx %Lx" label k0 k1)
    (one_of
       [
         labelled "a key of one word" (pretoken 1 8);
         labelled "a key of two words" (pretoken 9 15);
         labelled "any two words" (pair int64 int64);
         labelled "a word and a length tag" (pair int64 tagged);
       ])

let hashes_law =
  prop ~count:2000 "the C kernels hash any key words as OCaml does" key_words
    (fun (label, k0, k1) ->
      List.iter
        (fun l -> cover l (l = label))
        [
          "a key of one word";
          "a key of two words";
          "any two words";
          "a word and a length tag";
        ];
      let ocaml, c = Brot.cache_hashes k0 k1 in
      equal int ocaml c)

(* run_chunks *)

(* An executor that runs its tasks one after the other and counts its calls. *)
let sequential calls n task =
  incr calls;
  for i = 0 to n - 1 do
    task i
  done

let chunk_sizes =
  Gen.with_pp
    (fun ppf sizes ->
      Format.fprintf ppf "[|%s|]"
        (String.concat "; " (Array.to_list (Array.map string_of_int sizes))))
    Gen.(array ~size:(int_range 0 40) (int_range 0 1000))

let handout_law =
  prop ~count:300
    "hands out each chunk once, largest first, through one call of exec"
    chunk_sizes ~examples:[ [||]; [| 7 |] ] (fun sizes ->
      let n = Array.length sizes in
      cover "one chunk" (n = 1);
      cover "chunks of different sizes"
        (Array.exists (fun s -> s <> sizes.(0)) sizes);
      let calls = ref 0 and worked = ref [] in
      Brot.run_chunks ~exec:(sequential calls) sizes (fun c ->
          worked := c :: !worked);
      let worked = List.rev !worked in
      equal ~msg:"chunks worked" (list int) (List.init n Fun.id)
        (List.sort Int.compare worked);
      let order = List.map (fun c -> sizes.(c)) worked in
      equal ~msg:"sizes in the order handed out" (list int)
        (List.sort (fun a b -> Int.compare b a) order)
        order;
      equal ~msg:"calls of exec" int (if n > 1 then 1 else 0) !calls)

let () =
  exit
    (run "kernels"
       [
         group "Pre_tokenizer.byte_level"
           [
             walker_law;
             test
               "a shape cuts where the pre-tokenizer cuts at every offset of a \
                64-byte batch"
               test_batch_offsets;
             test
               "a long pretoken cuts where the pre-tokenizer cuts from span \
                600 to 1400"
               test_long_pretoken_seams;
           ];
         group "Pre_tokenizer.split_regex"
           [
             test
               "a cl100k shape cuts as the regular expression at every offset \
                of a 64-byte batch"
               (test_split_offsets
                  [ cl100k; cl100k_single_digits; cl100k_marks ]
                  split_families);
             test
               "an o200k shape cuts as the regular expression at every offset \
                of a 64-byte batch"
               (test_split_offsets [ o200k; tekken ] o200k_families);
           ];
         group "bpe"
           [
             bl_law;
             sp_law;
             test "a stream of 100000 units encodes as its units"
               test_sp_long_stream;
             test
               "a character without a piece or an unknown token gives no token"
               test_sp_pieceless;
             test_disagreeing;
             test_ignore_merges;
             spaced_law;
             test_stray_lead_byte;
             grown_law;
             test_cache_doubles;
             test "a released state holds one table" test_released_state;
             law "gpt2"
               (byte_level_reference ~pattern:gpt2_pattern)
               ~text:law_text ~cover:cover_law;
             law "mistral" mistral_reference ~text:law_text ~cover:cover_law;
             law "llama3"
               (byte_level_reference ~pattern:cl100k ~ignore_merges:true)
               ~text:cl100k_text ~cover:cover_cl100k;
             law "qwen2_5"
               (byte_level_reference ~pattern:cl100k_single_digits
                  ~normalize:(Brot.Normalizer.apply Brot.Normalizer.nfc))
               ~text:cl100k_text ~cover:cover_cl100k;
           ];
         group "encode_batch_ids" [ test_spill ];
         group "hold_cache"
           [
             test "a tokenizer keeps at most a state per recommended domain"
               test_states_bounded;
             test "a state the function raises through is dropped"
               test_raised_state_dropped;
           ];
         group "cache_set"
           [
             test "the ▁th words take a set each" test_marked_words;
             spread_large;
             spread_small;
           ];
         group "cache_hashes" [ hashes_law ];
         group "run_chunks" [ handout_law ];
       ])
