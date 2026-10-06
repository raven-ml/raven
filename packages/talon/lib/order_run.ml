(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* [column b n] is the one-batch table [b]'s column [n], as it is. *)
let column b n =
  let i = List.find_index (String.equal n) (Schema.names (Table.schema b)) in
  (Table.columns b).(Option.get i)

let words ks column =
  Key.order (List.map (fun (k : Order.t) -> (column k.name, k)) ks)

let rows idx b =
  Table.batch (Table.schema b) ~rows:(Nx.dim 0 idx)
    (Array.map (Column.gather idx) (Table.columns b))

let sub b ~offset ~length =
  Table.batch (Table.schema b) ~rows:length
    (Array.map (Column.sub ~offset ~length) (Table.columns b))

let sort ks b =
  if ks = [] then b
  else
    let p = Nx.lexsort (words ks (column b)) in
    Table.batch (Table.schema b) ~rows:(Table.rows b)
      (Array.map (Column.permute p) (Table.columns b))

(* Past [k] rows, the [k] first rows are among those whose first word is at most
   the [k]th smallest: [Nx.top_k] of the complemented words finds it, and only
   those rows are sorted, in row order so that the sort stays stable. *)
let top_k ~offset ~length ks b =
  let n = Table.rows b in
  let k = if length > max_int - offset then max_int else offset + length in
  if offset >= n || length = 0 then sub b ~offset:0 ~length:0
  else if k >= n || ks = [] then
    sub (sort ks b) ~offset ~length:(Int.min k n - offset)
  else
    let w = words ks (column b) in
    let first = Nx.slice [ A; I 0 ] w in
    let greatest, _ = Nx.top_k ~k (Nx.bitwise_not first) in
    let kth = Nx.bitwise_not (Nx.slice [ R (k - 1, k) ] greatest) in
    let candidates = Nx.positions (Nx.less_equal first kth) in
    let order = Nx.lexsort (Nx.take ~axis:0 ~indices:candidates w) in
    let idx = Nx.take ~indices:order candidates in
    rows (Nx.slice [ Nx.R (offset, k) ] idx) b

(* A row is out of order where, at the first word where it differs from the row
   before, its word is the smaller. *)
let unordered ks prev b =
  let read, shift =
    match prev with
    | None -> (column b, 1)
    | Some p ->
        let last = Table.rows p - 1 in
        let concat n =
          Column.concat
            [ Column.sub (column p n) ~offset:last ~length:1; column b n ]
        in
        (concat, 0)
  in
  let w = words ks read in
  let m = Nx.dim 0 w - 1 in
  if m < 1 then None
  else
    let before = Nx.slice [ Nx.R (0, m); Nx.R (0, Nx.dim 1 w) ] w
    and after = Nx.slice [ Nx.R (1, m + 1); Nx.R (0, Nx.dim 1 w) ] w in
    let tied = ref (Nx.ones Nx.bool [| m |])
    and out = ref (Nx.zeros Nx.bool [| m |]) in
    for j = 0 to Nx.dim 1 w - 1 do
      let x = Nx.slice [ A; I j ] before and y = Nx.slice [ A; I j ] after in
      out := Nx.logical_or !out (Nx.logical_and !tied (Nx.greater x y));
      tied := Nx.logical_and !tied (Nx.equal x y)
    done;
    let rs = Nx.positions !out in
    if Nx.dim 0 rs = 0 then None
    else Some (Int64.to_int (Nx.item [ 0 ] rs) + shift)
