(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* [ids] are the held ids from [Array.length dense] on, ascending, and
   [values] theirs. *)
type 'a t = {
  dense : 'a array;
  ids : int array;
  values : 'a array;
  absent : 'a;
}

let of_list ~absent pairs =
  (* Sorted by id, a later pair of an id before an earlier one, so the first
     of each run of equal ids is the one kept. *)
  let pairs = Array.of_list (List.mapi (fun i (id, v) -> (id, -i, v)) pairs) in
  Array.sort
    (fun (a, i, _) (b, j, _) ->
      if a <> b then Int.compare a b else Int.compare i j)
    pairs;
  let distinct = ref [] in
  Array.iteri
    (fun i ((id, _, _) as p) ->
      let id', _, _ = pairs.(max 0 (i - 1)) in
      if i = 0 || id' <> id then distinct := p :: !distinct)
    pairs;
  let distinct = Array.of_list (List.rev !distinct) in
  (* The [k]th id, counted from 0, ends a prefix of [id + 1] ids that holds
     [k + 1] of them. *)
  let length = ref 0 in
  Array.iteri
    (fun k (id, _, _) -> if id + 1 <= 2 * (k + 1) then length := id + 1)
    distinct;
  let dense = Array.make !length absent in
  let rest = ref [] in
  Array.iter
    (fun (id, _, v) ->
      if id < !length then dense.(id) <- v else rest := (id, v) :: !rest)
    distinct;
  let rest = Array.of_list (List.rev !rest) in
  { dense; ids = Array.map fst rest; values = Array.map snd rest; absent }

(* Every non-negative int is an int32 where ints have 32 bits or fewer. *)
let max_id = if Sys.int_size > 32 then Int32.to_int Int32.max_int else max_int
let in_range id = id >= 0 && id <= max_id

let of_vocab ~who vocab =
  Hashtbl.iter
    (fun token id ->
      if not (in_range id) then
        invalid_arg
          (Printf.sprintf "%s: %S has id %d, outside [0, 2^31)" who token id))
    vocab;
  of_list ~absent:""
    (Hashtbl.fold (fun token id acc -> (id, token) :: acc) vocab [])

let of_array ~absent a = { dense = a; ids = [||]; values = [||]; absent }

let rec search ids id lo hi =
  if lo >= hi then -1
  else
    let mid = (lo + hi) lsr 1 in
    let m = Array.unsafe_get ids mid in
    if m = id then mid
    else if m < id then search ids id (mid + 1) hi
    else search ids id lo mid

let[@inline] get t id =
  if id < 0 then t.absent
  else if id < Array.length t.dense then Array.unsafe_get t.dense id
  else
    let i = search t.ids id 0 (Array.length t.ids) in
    if i < 0 then t.absent else Array.unsafe_get t.values i

let dense t = t.dense

let mapi ~absent f t =
  {
    dense = Array.mapi f t.dense;
    ids = t.ids;
    values = Array.mapi (fun i v -> f (Array.unsafe_get t.ids i) v) t.values;
    absent;
  }

let iteri f t =
  Array.iteri f t.dense;
  Array.iteri (fun i id -> f id (Array.unsafe_get t.values i)) t.ids

let to_list t =
  let acc = ref [] in
  iteri (fun id v -> acc := (id, v) :: !acc) t;
  List.rev !acc
