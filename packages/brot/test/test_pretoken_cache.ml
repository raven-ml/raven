(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Where the pretoken cache puts a key. A set holds two entries and each miss
   stored into it evicts the older, so pretokens that agree in their first bytes
   — words with a common stem, every ▁-marked word — must spread over the sets,
   or they evict each other and are encoded again on every occurrence. *)

open Windtrap

(* The back sets of a cache of the default 262144 entries. *)
let sets = 131072
let set key = Brot.cache_set key ~sets

let test_marked_words () =
  let words =
    List.map
      (fun w -> "\xE2\x96\x81" ^ w)
      [ "the"; "they"; "them"; "their"; "there"; "these"; "then" ]
  in
  let sets = List.sort_uniq Int.compare (List.map set words) in
  equal ~msg:"sets of the ▁th words" int (List.length words)
    (List.length sets)

let bytes n =
  Gen.string_of ~size:(Gen.constant n) (Gen.map Char.chr (Gen.int_range 0 255))

(* A stem of 5 to 8 bytes and eight keys after it, told apart by their first
   byte past the stem and ending in 0 to 6 more bytes. *)
let family =
  Gen.bind (Gen.int_range 5 8) (fun stem_len ->
      Gen.bind (bytes stem_len) (fun stem ->
          Gen.map
            (fun tails ->
              let key k t = stem ^ String.make 1 (Char.chr k) ^ t in
              (stem, List.mapi key tails))
            (Gen.list ~size:(Gen.constant 8)
               (Gen.bind (Gen.int_range 0 (14 - stem_len)) bytes))))

(* Eight keys over 131072 sets share one with odds of about 2 in 10^4 when the
   index draws on all their bytes; asking for six sets keeps a run from failing
   by chance. *)
let spread =
  prop ~count:300 "keys sharing their first 5 to 8 bytes spread over sets"
    (Gen.with_pp
       (fun ppf (stem, _) -> Format.fprintf ppf "stem %S" stem)
       family)
    (fun (_, keys) ->
      let distinct =
        List.length (List.sort_uniq Int.compare (List.map set keys))
      in
      at_least ~msg:"distinct sets of eight keys" int ~than:6 distinct)

let () =
  exit
    (run "Pretoken cache"
       [
         group "index"
           [ test "the ▁th words take a set each" test_marked_words; spread ];
       ])
