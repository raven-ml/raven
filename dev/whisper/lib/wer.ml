(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type counts = {
  hits : int;
  substitutions : int;
  deletions : int;
  insertions : int;
}

(* jiwer's wer_default: RemoveMultipleSpaces ([\s\s+] to a space), Strip and
   ReduceToListOfListOfWords ([s.split(" ")] without the empty words). *)
let words s =
  let b = Buffer.create (String.length s) in
  let n = String.length s in
  (* The code points of [s] from [i], with their ends, as a list. *)
  let rec chars i acc =
    if i >= n then List.rev acc
    else
      let d = String.get_utf_8_uchar s i in
      let j = i + Uchar.utf_decode_length d in
      chars j ((Uchar.utf_decode_uchar d, i, j) :: acc)
  in
  let a = Array.of_list (chars 0 []) in
  let m = Array.length a in
  let space k = k < m && (fun (u, _, _) -> Space.is_space u) a.(k) in
  let rec go k =
    if k < m then
      if space k && space (k + 1) then begin
        let rec stop j = if space j then stop (j + 1) else j in
        Buffer.add_char b ' ';
        go (stop k)
      end
      else begin
        let _, i, j = a.(k) in
        Buffer.add_string b (String.sub s i (j - i));
        go (k + 1)
      end
  in
  go 0;
  String.split_on_char ' ' (Space.strip (Buffer.contents b))
  |> List.filter (fun w -> w <> "")
  |> Array.of_list

(* rapidfuzz's Levenshtein.editops (Levenshtein_py.py): the common prefix and
   suffix are matched, then [d], the edit distances of the rest's prefixes, is
   walked back from its last cell: a deletion where [d] is one more than the
   cell before in the reference, else an insertion where the cell before in the
   hypothesis is one less than the cell before in both, else a pair. Hyyrö's bit
   vectors in rapidfuzz hold these differences of [d]. *)
let align ~reference ~hypothesis =
  let n1 = Array.length reference and n2 = Array.length hypothesis in
  let rec common_prefix k =
    if k < n1 && k < n2 && reference.(k) = hypothesis.(k) then
      common_prefix (k + 1)
    else k
  in
  let prefix = common_prefix 0 in
  let rec common_suffix k =
    if
      k < n1 - prefix
      && k < n2 - prefix
      && reference.(n1 - 1 - k) = hypothesis.(n2 - 1 - k)
    then common_suffix (k + 1)
    else k
  in
  let suffix = common_suffix 0 in
  let r = Array.sub reference prefix (n1 - prefix - suffix)
  and h = Array.sub hypothesis prefix (n2 - prefix - suffix) in
  let m1 = Array.length r and m2 = Array.length h in
  (* [d.(i).(j)] is the distance of [r]'s first [i] words to [h]'s first [j]. *)
  let d = Array.make_matrix (m1 + 1) (m2 + 1) 0 in
  for i = 0 to m1 do
    for j = 0 to m2 do
      d.(i).(j) <-
        (if i = 0 then j
         else if j = 0 then i
         else
           min
             (min (d.(i - 1).(j) + 1) (d.(i).(j - 1) + 1))
             (d.(i - 1).(j - 1) + if r.(i - 1) = h.(j - 1) then 0 else 1))
    done
  done;
  let rec back i j ((s, del, ins) as c) =
    if i = 0 then (s, del, ins + j)
    else if j = 0 then (s, del + i, ins)
    else if d.(i).(j) = d.(i - 1).(j) + 1 then back (i - 1) j (s, del + 1, ins)
    else if j > 1 && d.(i).(j - 1) = d.(i - 1).(j - 1) - 1 then
      back i (j - 1) (s, del, ins + 1)
    else if r.(i - 1) <> h.(j - 1) then back (i - 1) (j - 1) (s + 1, del, ins)
    else back (i - 1) (j - 1) c
  in
  let substitutions, deletions, insertions = back m1 m2 (0, 0, 0) in
  {
    hits = n1 - substitutions - deletions;
    substitutions;
    deletions;
    insertions;
  }

let corpus pairs =
  List.fold_left
    (fun acc (reference, hypothesis) ->
      let c =
        align ~reference:(words reference) ~hypothesis:(words hypothesis)
      in
      {
        hits = acc.hits + c.hits;
        substitutions = acc.substitutions + c.substitutions;
        deletions = acc.deletions + c.deletions;
        insertions = acc.insertions + c.insertions;
      })
    { hits = 0; substitutions = 0; deletions = 0; insertions = 0 }
    pairs

let rate c =
  let reference = c.hits + c.substitutions + c.deletions in
  if reference = 0 then float c.insertions
  else float (c.substitutions + c.deletions + c.insertions) /. float reference
