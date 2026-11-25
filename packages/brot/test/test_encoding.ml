(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Brot

(* An encoding whose every column is a function of the token's ID, so that an
   operation that moves one column out of step with the others changes the
   encoding. [type_id], when given, is every token's type ID. *)
let numbered ?type_id ?overflowing ids =
  let ids = Array.of_list ids in
  let column f = Array.map f ids in
  let type_of id = Option.value type_id ~default:(id mod 3) in
  Encoding.create ~ids ~type_ids:(column type_of) ~tokens:(column string_of_int)
    ~words:(column (fun id -> if id mod 4 = 3 then None else Some (id / 2)))
    ~offsets:(column (fun id -> (id, id + 1)))
    ~special_tokens_mask:(column (fun id -> id land 1))
    ~attention_mask:(column (fun id -> (id lsr 1) land 1))
    ?overflowing ()

let positions n = List.init n Fun.id

let rec same a b =
  Encoding.ids a = Encoding.ids b
  && Encoding.type_ids a = Encoding.type_ids b
  && Encoding.tokens a = Encoding.tokens b
  && Encoding.word_ids a = Encoding.word_ids b
  && Encoding.offsets a = Encoding.offsets b
  && Encoding.special_tokens_mask a = Encoding.special_tokens_mask b
  && Encoding.attention_mask a = Encoding.attention_mask b
  && List.equal same (Encoding.overflowing a) (Encoding.overflowing b)

let rec pp_encoding ppf enc =
  Format.fprintf ppf "@[<v>%a" Encoding.pp enc;
  List.iteri
    (fun i window -> Format.fprintf ppf "@,window %d:@,%a" i pp_encoding window)
    (Encoding.overflowing enc);
  Format.fprintf ppf "@]"

(* Two encodings are equal when their seven columns and their windows are. *)
let encoding = Testable.make ~pp:pp_encoding ~equal:same

let direction =
  Gen.of_list
    ~pp:(fun ppf d ->
      Format.pp_print_string ppf
        (match d with `Left -> "`Left" | `Right -> "`Right"))
    [ `Right; `Left ]

(* Encoding.truncate, from the .mli: with [`Right] the first [max_length] IDs
   are kept and the rest is cut into windows of [max_length] that overlap the
   one before by [stride], walking right; the last window stops at the end.
   [`Left] is the same cut of the reversed IDs, read backwards. *)
let rec windows ~max_length ~stride ids =
  let window = List.filteri (fun i _ -> i < max_length) ids in
  if List.length ids <= max_length then [ window ]
  else
    let step = max_length - stride in
    window
    :: windows ~max_length ~stride (List.filteri (fun i _ -> i >= step) ids)

let cut ~direction ~max_length ~stride ids =
  match direction with
  | `Right -> windows ~max_length ~stride ids
  | `Left -> List.map List.rev (windows ~max_length ~stride (List.rev ids))

(* Encoding.pad, from the .mli: [target_length - length enc] padding tokens,
   each with the padding's ID, type ID and token, no word, the span (0, 0),
   attention 0 and special 1, after the tokens or, with [`Left], before them;
   each window padded the same way. *)
let pad_id = 100
let pad_type_id = 7
let pad_token = "<pad>"

let rec padded ~target_length ~direction enc =
  let n = max 0 (target_length - Encoding.length enc) in
  let place column fill =
    let padding = Array.make n fill in
    match direction with
    | `Right -> Array.append column padding
    | `Left -> Array.append padding column
  in
  Encoding.create
    ~ids:(place (Encoding.ids enc) pad_id)
    ~type_ids:(place (Encoding.type_ids enc) pad_type_id)
    ~tokens:(place (Encoding.tokens enc) pad_token)
    ~words:(place (Encoding.word_ids enc) None)
    ~offsets:(place (Encoding.offsets enc) (0, 0))
    ~special_tokens_mask:(place (Encoding.special_tokens_mask enc) 1)
    ~attention_mask:(place (Encoding.attention_mask enc) 0)
    ~overflowing:
      (List.map (padded ~target_length ~direction) (Encoding.overflowing enc))
    ()

(* empty *)

let test_empty () = equal encoding (numbered []) Encoding.empty

(* create *)

let accessors_return_arrays ids =
  let n = List.length ids in
  let ids = Array.of_list ids in
  let type_ids = Array.map (fun id -> id + 1) ids in
  let tokens = Array.map string_of_int ids in
  let words = Array.map (fun id -> Some (id + 2)) ids in
  let offsets = Array.map (fun id -> (id, id + 3)) ids in
  let special_tokens_mask = Array.make n 1 in
  let attention_mask = Array.make n 0 in
  let enc =
    Encoding.create ~ids ~type_ids ~tokens ~words ~offsets ~special_tokens_mask
      ~attention_mask ()
  in
  equal ~msg:"ids" (array int) ids (Encoding.ids enc);
  equal ~msg:"type_ids" (array int) type_ids (Encoding.type_ids enc);
  equal ~msg:"tokens" (array string) tokens (Encoding.tokens enc);
  equal ~msg:"word_ids" (array (option int)) words (Encoding.word_ids enc);
  equal ~msg:"offsets" (array (pair int int)) offsets (Encoding.offsets enc);
  equal ~msg:"special_tokens_mask" (array int) special_tokens_mask
    (Encoding.special_tokens_mask enc);
  equal ~msg:"attention_mask" (array int) attention_mask
    (Encoding.attention_mask enc);
  equal ~msg:"overflowing" (list encoding) [] (Encoding.overflowing enc)

let test_create_keeps_arrays () =
  let ids = [| 1; 2 |] in
  let enc =
    Encoding.create ~ids ~type_ids:[| 0; 0 |] ~tokens:[| "a"; "b" |]
      ~words:[| None; None |]
      ~offsets:[| (0, 1); (1, 2) |]
      ~special_tokens_mask:[| 0; 0 |] ~attention_mask:[| 1; 1 |] ()
  in
  ids.(0) <- 7;
  equal ~msg:"after a change to the array given" (array int) [| 7; 2 |]
    (Encoding.ids enc);
  (Encoding.ids enc).(1) <- 8;
  equal ~msg:"after a change to the array returned" (array int) [| 7; 8 |]
    (Encoding.ids enc)

let columns =
  [
    "ids";
    "type_ids";
    "tokens";
    "words";
    "offsets";
    "special_tokens_mask";
    "attention_mask";
  ]

let create_with_short short =
  let length column = if column = short then 1 else 2 in
  raises
    (Invalid_argument "Encoding.create: arrays must all have the same length")
    (fun () ->
      Encoding.create
        ~ids:(Array.make (length "ids") 0)
        ~type_ids:(Array.make (length "type_ids") 0)
        ~tokens:(Array.make (length "tokens") "")
        ~words:(Array.make (length "words") None)
        ~offsets:(Array.make (length "offsets") (0, 0))
        ~special_tokens_mask:(Array.make (length "special_tokens_mask") 0)
        ~attention_mask:(Array.make (length "attention_mask") 1)
        ())

(* concat *)

(* Part [i] carries the window [100 + i], so the result shows whose windows it
   kept. *)
let concatenates parts =
  cover "no parts" (parts = []);
  cover "an empty part" (List.mem [] parts);
  cover "three parts" (List.length parts = 3);
  let window i = [ numbered [ 100 + i ] ] in
  let expected =
    match parts with
    | [] -> numbered []
    | _ -> numbered ~overflowing:(window 0) (List.concat parts)
  in
  equal encoding expected
    (Encoding.concat
       (List.mapi (fun i ids -> numbered ~overflowing:(window i) ids) parts))

(* special_tokens_mask *)

(* Hugging Face's tokenizers 0.23.1 gives [0, 0, 0] for both. *)
let test_added_token_unmasked () =
  let mask special =
    let t =
      bpe
        ~vocab:[ ("a", 0); ("b", 1) ]
        ~merges:[]
        ~added_tokens:[ added_token ~special "<s>" ]
        ()
    in
    Encoding.special_tokens_mask (encode t "a<s>b")
  in
  equal ~msg:"special" (array int) [| 0; 0; 0 |] (mask true);
  equal ~msg:"not special" (array int) [| 0; 0; 0 |] (mask false)

(* with_type_id *)

let retypes (ids, type_id) =
  let window = [ numbered [ 99 ] ] in
  equal encoding
    (numbered ~type_id ~overflowing:window ids)
    (Encoding.with_type_id (numbered ~overflowing:window ids) type_id)

(* truncate *)

(* [enc] carries a window of its own, [99], which only the cut at [max_length =
   0] keeps. *)
let truncates (n, max_length, stride, direction) =
  let truncated = n > max_length in
  (* The values discarded here are the domain of the raise law below. *)
  assume (not (truncated && max_length > 0 && stride >= max_length));
  let enc = numbered ~overflowing:[ numbered [ 99 ] ] (positions n) in
  let expected =
    if not truncated then enc
    else if max_length = 0 then numbered ~overflowing:[ enc ] []
    else
      match cut ~direction ~max_length ~stride (positions n) with
      | kept :: windows ->
          numbered ~overflowing:(List.map (fun w -> numbered w) windows) kept
      | [] -> assert false
  in
  let windows = Encoding.overflowing expected in
  cover "unchanged" (not truncated);
  cover "max_length 0" (truncated && max_length = 0);
  cover "windows overlap"
    (truncated && max_length > 0 && stride > 0 && List.length windows >= 2);
  cover "last window shorter"
    (truncated && max_length > 0
    && List.exists (fun w -> Encoding.length w < max_length) windows);
  cover "`Left" (truncated && max_length > 0 && direction = `Left);
  equal encoding expected (Encoding.truncate ~stride ~direction enc ~max_length)

let raises_on_wide_stride (max_length, wider, longer, direction) =
  let enc = numbered (positions (max_length + 1 + longer)) in
  raises
    (Invalid_argument "Encoding.truncate: stride must be less than max_length")
    (fun () ->
      Encoding.truncate ~stride:(max_length + wider) ~direction enc ~max_length)

(* pad *)

let pads (n, window, target_length, direction) =
  let window = numbered (List.init window (fun i -> 50 + i)) in
  let enc = numbered ~overflowing:[ window ] (positions n) in
  cover "padded" (n < target_length);
  cover "kept, window padded"
    (n >= target_length && Encoding.length window < target_length);
  cover "`Left" (n < target_length && direction = `Left);
  cover "negative target_length" (target_length < 0);
  equal encoding
    (padded ~target_length ~direction enc)
    (Encoding.pad enc ~target_length ~pad_id ~pad_type_id ~pad_token ~direction)

(* pp *)

let test_pp () =
  let enc =
    Encoding.create ~ids:[| 101; 7592 |] ~type_ids:[| 0; 1 |]
      ~tokens:[| "[CLS]"; "hello" |] ~words:[| None; Some 0 |]
      ~offsets:[| (0, 0); (0, 5) |]
      ~special_tokens_mask:[| 1; 0 |] ~attention_mask:[| 1; 1 |] ()
  in
  expect (Format.asprintf "%a" Encoding.pp enc)
  @@ __POS_OF__
       {|
    Index Token     ID Offsets       Word  Type  Attn  Special
    0     [CLS]    101 (   0,   0)     -     0     1        1
    1     hello   7592 (   0,   5)     0     1     1        0
    |}

let test_pp_empty () =
  expect (Format.asprintf "%a" Encoding.pp Encoding.empty)
  @@ __POS_OF__
       {|
    Index Token     ID Offsets       Word  Type  Attn  Special
    |}

let ids = Gen.(list ~size:(int_range 0 6) (int_range 0 20))

let () =
  exit
    (run "Encoding"
       [
         group "empty" [ test "has no tokens" test_empty ];
         group "create"
           [
             prop "each accessor returns the array given" ids
               accessors_return_arrays;
             test "keeps the arrays it is given" test_create_keeps_arrays;
             cases "arrays of different lengths raise Invalid_argument"
               ~id:Fun.id columns create_with_short;
           ];
         group "concat"
           [
             prop ~examples:[ [] ]
               "is the parts in order, with the first part's overflowing"
               Gen.(
                 list ~size:(int_range 0 3)
                   (list ~size:(int_range 0 3) (int_range 0 20)))
               concatenates;
           ];
         group "special_tokens_mask"
           [
             test "is 0 for an added token found in the input"
               test_added_token_unmasked;
           ];
         group "is_empty"
           [
             prop ~examples:[ [] ] "holds iff there are no tokens" ids
               (fun ids ->
                 cover "empty" (ids = []);
                 satisfies ~claim:"is_empty iff there are no ids"
                   (pair (list int) bool)
                   (fun (ids, is_empty) -> is_empty = (ids = []))
                   (ids, Encoding.is_empty (numbered ids)));
           ];
         group "length"
           [
             prop "is the number of tokens" ids (fun ids ->
                 equal int (List.length ids) (Encoding.length (numbered ids)));
           ];
         group "with_type_id"
           [
             prop "sets every type ID and keeps the rest"
               Gen.(pair ids (int_range 0 3))
               retypes;
           ];
         group "truncate"
           [
             prop ~count:1100
               ~examples:[ (10, 4, 2, `Right) ]
               "keeps max_length tokens and windows the rest by stride"
               Gen.(
                 quad (int_range 0 8) (int_range 0 7) (int_range 0 6) direction)
               truncates;
             prop ~count:600
               "raises Invalid_argument when it truncates and stride is not \
                less than max_length"
               Gen.(
                 quad (int_range 1 6) (int_range 0 6) (int_range 0 6) direction)
               raises_on_wide_stride;
           ];
         group "pad"
           [
             prop ~count:400
               "extends to target_length with padding tokens, windows too"
               Gen.(
                 quad (int_range 0 4) (int_range 0 4) (int_range (-1) 6)
                   direction)
               pads;
           ];
         group "pp"
           [
             test "prints a row per token" test_pp;
             test "prints the header alone for no tokens" test_pp_empty;
           ];
       ])
