(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t =
  | Blocking of (Table.t -> Table.t -> (Table.t, Eval.failure) result)
  | Streaming of {
      batch : Table.t -> Table.t -> (Table.t, Eval.failure) result;
      last : Table.t -> Table.t -> int -> (Table.t, Eval.failure) result;
    }

let ( let* ) = Result.bind
let arange n = Nx.arange Nx.int64 0 n 1
let none n = Nx.full Nx.int64 [| n |] (-1L)

(* [exclusive c] is the sum of the entries of [c] before each one. *)
let exclusive c = Nx.sub (Nx.cumsum c) c

(* The left rows a join keeps: [All] of them in order, whose columns it keeps
   as they are, or those [At] indices. *)
type rows = All | At of Nx.host Nx.int64_t

(* [assemble q kind l li r ri keys] is [q]'s rows: each column is the one [keys]
   give it, or else [l]'s taken at [li], or else [r]'s taken at [ri]. The side a
   [kind] of join pads, the left of a full join and the right of a left or full
   one, is null where an index is outside its rows, as [-1] is, and those rows
   are found once for all its columns. *)
let assemble q (kind : Join.kind) l li r ri keys =
  let found pads idx t =
    if not pads then None
    else
      Some
        (lazy
          (let inside =
             Nx.logical_and
               (Nx.greater_equal_s idx 0L)
               (Nx.less_s idx (Int64.of_int (Table.rows t)))
           in
           Column.mask (Nx.cast Nx.bit inside)))
  in
  let side found idx c =
    let g = Column.gather idx c in
    match found with
    | Some found when Option.is_none (Column.validity g) ->
        Column.restrict (Lazy.force found) g
    | _ -> g
  in
  let rows, left =
    match li with
    | All -> (Table.rows l, Fun.id)
    | At li -> (Nx.dim 0 li, side (found (kind = Full) li l) li)
  in
  let found_r = found (kind = Left || kind = Full) ri r in
  let column n =
    match List.assoc_opt n keys with
    | Some c -> c
    | None -> (
        match Schema.find (Table.schema l) n with
        | Some _ -> left (Table.column l n)
        | None -> side found_r ri (Table.column r n))
  in
  let s = Query.schema q in
  Table.batch s ~rows (Array.of_list (List.map column (Schema.names s)))

(* Assertions *)

let phrase : Join.count -> string = function
  | Any -> "any"
  | At_most_one -> "at most one"
  | One -> "one"
  | At_least_one -> "at least one"

(* [failure side count keys row c] is the failure of [side]'s row [row], whose
   [c] matches [count] does not allow, naming the row's [keys]. *)
let failure side count keys row c =
  let key ppf (n, k) =
    Format.fprintf ppf "%a is %t" Type.pp_quoted n (Form.pp_cell k row)
  in
  let whose ppf = function
    | [] -> ()
    | ks ->
        Format.fprintf ppf " whose %a"
          (Format.pp_print_list
             ~pp_sep:(fun ppf () -> Format.pp_print_string ppf " and ")
             key)
          ks
  in
  let why =
    Format.asprintf "the %s row%a matches %Ld rows, not %s" side whose keys c
      (phrase count)
  in
  Error { Eval.row; output = 0; cause = Data (Error.v (why ^ ".")) }

(* [check side count keys counts] fails at the first row of [side] whose number
   of matches [counts] [count] does not allow, naming the row's [keys]. *)
let check side (count : Join.count) keys counts =
  let bad =
    match count with
    | Any -> None
    | At_most_one -> Some (fun c -> Nx.greater_s c 1L)
    | One -> Some (fun c -> Nx.not_equal_s c 1L)
    | At_least_one -> Some (fun c -> Nx.less_s c 1L)
  in
  match bad with
  | None -> Ok ()
  | Some bad ->
      let c = Lazy.force counts in
      let at = Nx.positions (bad c) in
      if Nx.numel at = 0 then Ok ()
      else
        let row = Int64.to_int (Nx.item [ 0 ] at) in
        failure side count keys row (Nx.item [ row ] c)

(* [sized side count ~rows (k, a) b] is {!check} for matches that follow from
   the sides' sizes: [side]'s first [k] of [rows] rows match [a] rows each, and
   the others [b]. It reads no value, so it holds over traced columns too. *)
let sized side (count : Join.count) ~rows (k, a) b =
  let allows c =
    match count with
    | Any -> true
    | At_most_one -> c <= 1
    | One -> c = 1
    | At_least_one -> c >= 1
  in
  let fails row c = failure side count [] row (Int64.of_int c) in
  if Int.min k rows > 0 && not (allows a) then fails 0 a
  else if rows > k && not (allows b) then fails k b
  else Ok ()

let named t ns = List.map (fun n -> (n, Table.column t n)) ns

(* Equality joins

   The keys of both sides are coded at once: {!Key.groups} numbers the groups of
   equal keys, and a left row's matches are the right rows of its group. When
   no two right rows share a group, as in a lookup of a table by its key, a
   scatter gives each group its right row. Otherwise the right rows of each
   group, in order, are one row of [runs]. The numbering follows first
   appearance and the left rows come first, so the groups that hold a left row
   are those up to the greatest left id: only the right rows of those are
   counted and placed in runs, the others matching none. *)

(* The matches of the left rows: [One ri], each left row's one match or [-1],
   or [Runs matched], each left row's matches, one after the other. *)
type matches =
  | One of Nx.host Nx.int64_t
  | Runs of Nx.host Nx.int64_t

(* [left_pairs counts matched] pairs each left row with its matches, or with
   [-1] if it has none. A left row of one match at most keeps every left row in
   order: its match is a scatter of [matched] to the rows that have one. *)
let left_pairs counts matched =
  let n = Nx.dim 0 counts in
  if n = 0 || Nx.item [] (Nx.max counts) <= 1L then
    (All, Nx.scatter ~axis:0 (Nx.positions counts) matched (none n))
  else
    let one = Nx.maximum_s counts 1L in
    let li = Nx.positions one and inner = Nx.positions counts in
    let rank =
      Nx.sub (arange (Nx.dim 0 inner)) (Nx.take inner (exclusive counts))
    in
    let at = Nx.add (Nx.take inner (exclusive one)) rank in
    let ri = Nx.scatter ~axis:0 at matched (none (Nx.dim 0 li)) in
    (At li, ri)

(* [emit q kind counts l r (lnames, rnames) keys lcounts rcounts matches] is
   the [kind] join of [l] and [r] once its [counts] hold: [lcounts] and
   [rcounts] are the number of matches of each left and right row, [matches]
   the pairs, [lnames] and [rnames] the columns of the equality atoms, and
   [keys] their columns over both sides, which a [Full] join coalesces. *)
let emit q kind (each_left, each_right) l r (lnames, rnames) keys lcounts
    rcounts matches =
  let* () = check "left" each_left (named l lnames) lcounts in
  let* () = check "right" each_right (named r rnames) rcounts in
  let n = Table.rows l in
  let left_pairs () =
    match Lazy.force matches with
    | One ri -> (All, ri)
    | Runs matched -> left_pairs (Lazy.force lcounts) matched
  in
  let pairs (li, ri) = Ok (assemble q kind l li r ri []) in
  let only keep = pairs (At (Nx.positions keep), arange 0) in
  match (kind : Join.kind) with
  | Inner -> (
      let li = Nx.positions (Lazy.force lcounts) in
      match Lazy.force matches with
      | One ri -> pairs (At li, Nx.take li ri)
      | Runs matched -> pairs (At li, matched))
  | Semi -> only (Nx.greater_s (Lazy.force lcounts) 0L)
  | Anti -> only (Nx.equal_s (Lazy.force lcounts) 0L)
  | Left -> pairs (left_pairs ())
  | Full ->
      (* A right row without a match follows, its keys in the left's. *)
      let li, ri = left_pairs () in
      let li = match li with All -> arange n | At li -> li in
      let u = Nx.positions (Nx.equal_s (Lazy.force rcounts) 0L) in
      let at = Nx.concatenate ~axis:0 [ li; Nx.add_s u (Int64.of_int n) ] in
      let coalesced = List.map (fun (ln, k) -> (ln, Column.gather at k)) keys in
      let li = Nx.concatenate ~axis:0 [ li; none (Nx.dim 0 u) ]
      and ri = Nx.concatenate ~axis:0 [ ri; u ] in
      Ok (assemble q kind l (At li) r ri coalesced)

(* [keyed left right eqs] reads the columns of the equality atoms [eqs] of
   both sides, each pair at the type it meets at, in one column. *)
let keyed left right eqs =
  let widen (ln, rn) =
    let tl = Option.get (Schema.find (Query.schema left) ln)
    and tr = Option.get (Schema.find (Query.schema right) rn) in
    let t = Option.get (Join.common_type tl tr) in
    (ln, rn, Eval.widen tl t, Eval.widen tr t)
  in
  let atoms = List.map widen eqs in
  fun l r ->
    List.map
      (fun (ln, rn, wl, wr) ->
        (ln, Column.concat [ wl (Table.column l ln); wr (Table.column r rn) ]))
      atoms

let names eqs = (List.map fst eqs, List.map snd eqs)

let equality q kind counts left right eqs =
  let keyed = keyed left right eqs and names = names eqs in
  fun l r ->
    let n = Table.rows l and m = Table.rows r in
    let keys = keyed l r in
    (* Only the ids and their number: each group's first row and count would
       be a pass over every row that nothing reads. *)
    let g = Key.groups ~size:0 (List.map snd keys) in
    let groups = Int64.to_int (Nx.item [] g.count) in
    let lids = Nx.slice [ Nx.R (0, n) ] g.ids
    and rids = Nx.slice [ Nx.R (n, n + m) ] g.ids in
    (* The right rows of each of the [k] groups that hold a left row, and so
       each left row's matches: the id of a right row of a later group is past
       the segments, and counts nowhere. *)
    let k = if n = 0 then 0 else Int64.to_int (Nx.item [] (Nx.max lids)) + 1 in
    let ones = Nx.ones_like rids in
    let per = Nx.reduce_segments Nx.Add ~segments:k rids ones in
    let lcounts = Nx.take lids per in
    let matches =
      if k = 0 || Nx.item [] (Nx.max per) <= 1L then
        One (Nx.take lids (Nx.scatter ~axis:0 rids (arange m) (none k)))
      else
        (* Only those groups' right rows, [hits], take a place in the runs:
           every right row when every group holds a left row. *)
        let runs =
          if k = groups then Nx_ragged.of_ids ~segments:k rids (arange m)
          else
            let hits = Nx.positions (Nx.less_s rids (Int64.of_int k)) in
            Nx_ragged.of_ids ~segments:k (Nx.take hits rids) hits
        in
        Runs (Nx_ragged.values (Nx_ragged.take ~indices:lids runs))
    in
    (* A right row's matches are the left rows matched to it. *)
    let rcounts =
      lazy
        (let matched = match matches with One ri | Runs ri -> ri in
         Nx.reduce_segments Nx.Add ~segments:m matched (Nx.ones_like matched))
    in
    emit q kind counts l r names keys (Lazy.from_val lcounts) rcounts
      (Lazy.from_val matches)

(* Range joins

   A range join sorts the rows of one side, its band's, by their equality
   group and then by the order word ({!Key.value}) of one column, leaving out
   the rows where that column is null. Each row of the other side, a probe,
   matches the sorted rows at the positions [lo, hi) between two searches of
   them: the greatest of its lower bounds and the least of its upper ones, or
   its group's first and last rows. A band's pairs are its ranges expanded.
   Its counts need no pair: a probe's is [hi - lo], and a sorted row's the
   number of ranges holding it, the running sum of [+1] at each [lo] and [-1]
   at each [hi].

   [overlaps] runs as three disjoint bands over the periods' starts S and
   stops T, SQL-92's predicate being S1 = S2 with both stops known, or
   S1 < S2 < T1, or S2 < S1 < T2 (8.11, General Rule 6, with the unknown
   cases worked out):
   - the right rows of a known stop whose S2 is in [S1, T1), or is S1 when the
     left's period is a point;
   - the right rows of an unknown stop whose S2 is in (S1, T1);
   - the left rows whose S1 is in (S2, T2).
   A probe of an unknown stop matches none. *)

type known = Nx.host Nx.bool_t option

(* The order words of a column of one side, and where it holds a value. *)
type words = { w : Nx.host Nx.uint64_t; known : known }

(* A side's periods: their starts and stops, and where each is known. *)
type period = {
  s : Nx.host Nx.uint64_t;
  sk : known;
  t : Nx.host Nx.uint64_t;
  tk : known;
}

let both (a : known) b =
  match (a, b) with
  | None, k | k, None -> k
  | Some a, Some b -> Some (Nx.logical_and a b)

(* [where k a b] is [a] where [k] holds, else [b]. *)
let where k a b = match k with None -> a | Some k -> Nx.where k a b

(* [ordered cs] is the words of the columns [cs], of one type, which compare
   across the columns: a compound value's are codes over all of them. *)
let ordered cs =
  let w = Key.value Order (Column.concat cs) in
  let at = ref 0 in
  List.map
    (fun c ->
      let n = Column.length c in
      let w = Nx.slice [ Nx.R (!at, !at + n) ] w in
      at := !at + n;
      { w; known = Option.map (Nx.cast Nx.bool) (Column.validity c) })
    cs

(* SQL-92's general rules 3 and 5: the lesser value starts a period, and with
   one value null the other starts it and its stop is unknown. *)
let period d e =
  let lo = Nx.minimum d.w e.w in
  let s =
    match (d.known, e.known) with
    | None, None -> lo
    | dk, ek -> where (both dk ek) lo (where dk d.w e.w)
  in
  let sk =
    match (d.known, e.known) with
    | Some dk, Some ek -> Some (Nx.logical_or dk ek)
    | _ -> None
  in
  { s; sk; t = Nx.maximum d.w e.w; tk = both d.known e.known }

(* [ascending g w] is [true] iff the rows [(g, w)] are in order. *)
let ascending g w =
  let k = Nx.dim 0 w in
  k < 2
  ||
  let a x = Nx.slice [ Nx.R (0, k - 1) ] x
  and b x = Nx.slice [ Nx.R (1, k) ] x in
  let le = Nx.less_equal (a w) (b w) in
  let ok =
    match g with
    | None -> le
    | Some g ->
        Nx.logical_or
          (Nx.less (a g) (b g))
          (Nx.logical_and (Nx.equal (a g) (b g)) le)
  in
  Nx.item [] (Nx.all ok)

let stacked g w = match g with None -> w | Some g -> Nx.stack ~axis:1 [ g; w ]

(* The rows of a band's side in its order, their number and their groups. *)
type order = { rows : rows; size : int; group : Nx.host Nx.uint64_t option }

(* [taken rows x] is [x] at [rows]; [nth rows p] is the rows at the positions
   [p] of [rows]. *)
let taken rows x = match rows with All -> x | At at -> Nx.take at x
let nth rows p = match rows with All -> p | At at -> Nx.take p at

(* [order group known w] sorts the rows where [known] holds by their [group],
   then their words [w]; none when they are in that order. *)
let order group known w =
  let rows = match known with None -> All | Some k -> At (Nx.positions k) in
  let g = Option.map (taken rows) group and key = taken rows w in
  let size = Nx.dim 0 key in
  if ascending g key then { rows; size; group = g }
  else
    let p = Nx.lexsort (stacked g key) in
    { rows = At (nth rows p); size; group = Option.map (Nx.take p) g }

(* [search o pg side w v] is, for each probe of group [pg] and word [v], the
   positions in [o] before its first row of word [v] or above ([`Left]), or
   above [v] ([`Right]), [w] being the words of [o]'s side. *)
let search o pg side w v =
  Nx.searchsorted ~side (stacked o.group (taken o.rows w)) (stacked pg v)

(* A band: the rows of the side [sorted] in their order, and for each probe
   the positions [lo, hi) of its matches among them. *)
type band = {
  sorted : [ `Left | `Right ];
  order : order;
  lo : Nx.host Nx.int64_t;
  hi : Nx.host Nx.int64_t;
}

(* [band sorted o pg ~probes ?lo ?hi known] is the band of [o] whose probes,
   of groups [pg], match from [lo] to [hi], or from their group's first row
   to its last; a probe matches none where [known] fails. *)
let band sorted o pg ~probes ?lo ?hi known =
  let edge side =
    match (pg, o.group) with
    | Some pg, Some g -> Nx.searchsorted ~side g pg
    | _ -> (
        match side with
        | `Left -> Nx.zeros Nx.int64 [| probes |]
        | `Right -> Nx.full Nx.int64 [| probes |] (Int64.of_int o.size))
  in
  let lo = match lo with Some lo -> lo | None -> edge `Left in
  let hi = match hi with Some hi -> hi | None -> edge `Right in
  { sorted; order = o; lo; hi = where known hi lo }

let width b = Nx.maximum_s (Nx.sub b.hi b.lo) 0L

(* [band_counts b ~n ~m] is the number of matches [b] gives each left row and
   each right row. *)
let band_counts b ~n ~m =
  let width = width b in
  let held rows =
    let k = b.order.size in
    let one = Nx.minimum_s width 1L in
    let d = Nx.zeros Nx.int64 [| k + 1 |] in
    let d = Nx.scatter ~combine:Add ~axis:0 b.lo one d in
    let d = Nx.scatter ~combine:Add ~axis:0 b.hi (Nx.neg one) d in
    let held = Nx.slice [ Nx.R (0, k) ] (Nx.cumsum d) in
    match b.order.rows with
    | All -> held
    | At at -> Nx.scatter ~axis:0 at held (Nx.zeros Nx.int64 [| rows |])
  in
  match b.sorted with `Right -> (width, held m) | `Left -> (held n, width)

(* [pairs b] is [b]'s pairs of left and right rows, by probe. *)
let pairs b =
  let width = width b in
  let probe = Nx.positions width in
  let rank =
    Nx.sub (arange (Nx.dim 0 probe)) (Nx.take probe (exclusive width))
  in
  let other = nth b.order.rows (Nx.add (Nx.take probe b.lo) rank) in
  match b.sorted with `Right -> (probe, other) | `Left -> (other, probe)

(* The atoms between the sides, read: a comparison's words of each side, or
   each side's periods. *)
type pred =
  | Compared of Join.compare * words * words
  | Periods of period * period

(* [holds (li, ri) p] is where [p] holds on the pairs [(li, ri)]. *)
let holds (li, ri) p =
  let gl x = Nx.take li x and gr x = Nx.take ri x in
  let known ks r =
    match List.fold_left both None ks with
    | None -> r
    | Some k -> Nx.logical_and r k
  in
  match p with
  | Compared (op, wl, wr) ->
      let a = gl wl.w and b = gr wr.w in
      let r =
        match op with
        | `Lt -> Nx.less a b
        | `Le -> Nx.less_equal a b
        | `Gt -> Nx.greater a b
        | `Ge -> Nx.greater_equal a b
      in
      known [ Option.map gl wl.known; Option.map gr wr.known ] r
  | Periods (p1, p2) ->
      let s1 = gl p1.s and t1 = gl p1.t and s2 = gr p2.s and t2 = gr p2.t in
      let sk1 = Option.map gl p1.sk and tk1 = Option.map gl p1.tk in
      let sk2 = Option.map gr p2.sk and tk2 = Option.map gr p2.tk in
      let within a x b = Nx.logical_and (Nx.less a x) (Nx.less x b) in
      Nx.logical_or
        (known [ tk1; tk2 ] (Nx.equal s1 s2))
        (Nx.logical_or
           (known [ sk1; sk2; tk1 ] (within s1 s2 t1))
           (known [ sk1; sk2; tk2 ] (within s2 s1 t2)))

(* [band_atoms atoms] is the band, the first [overlaps] or the side and name
   of the column that the most comparisons name, a right one on a tie, else
   the first written, then the band's atoms and the others. *)
let band_atoms atoms =
  let is_overlaps = function Join.Overlaps _ -> true | _ -> false in
  match List.find_opt is_overlaps atoms with
  | Some o -> (`Periods, [ o ], List.filter (fun a -> a != o) atoms)
  | None ->
      let named =
        List.concat_map
          (function
            | Join.Cmp (_, l, r) -> [ (`Left, l); (`Right, r) ] | _ -> [])
          atoms
      in
      let count c = List.length (List.filter (( = ) c) named) in
      let better c b =
        count c > count b
        || (count c = count b && fst c = `Right && fst b = `Left)
      in
      let ((side, name) as best) =
        List.fold_left
          (fun b c -> if better c b then c else b)
          (List.hd named) named
      in
      let names = function
        | Join.Cmp (_, l, r) ->
            String.equal (if side = `Left then l else r) name
        | _ -> false
      in
      let banded, others = List.partition names atoms in
      (`Column best, banded, others)

let mirror : Join.compare -> Join.compare = function
  | `Lt -> `Gt
  | `Le -> `Ge
  | `Gt -> `Lt
  | `Ge -> `Le

(* [periods ~n ~m lg rg p1 p2] is the bands of [overlaps] over the left's
   periods [p1] and the right's [p2], of groups [lg] and [rg]. *)
let periods ~n ~m lg rg p1 p2 =
  let rights known =
    let o = order rg known p2.s in
    (o, search o lg)
  in
  let o, at = rights p2.tk in
  let point = Nx.equal p1.s p1.t in
  let hi = Nx.where point (at `Right p2.s p1.s) (at `Left p2.s p1.t) in
  let known_stops =
    band `Right o lg ~probes:n ~lo:(at `Left p2.s p1.s) ~hi p1.tk
  in
  let unknown_stops =
    match p2.tk with
    | None -> []
    | Some tk ->
        let o, at = rights (both p2.sk (Some (Nx.logical_not tk))) in
        [
          band `Right o lg ~probes:n ~lo:(at `Right p2.s p1.s)
            ~hi:(at `Left p2.s p1.t) p1.tk;
        ]
  in
  let o = order lg p1.sk p1.s in
  let at = search o rg in
  let lefts =
    band `Left o rg ~probes:m ~lo:(at `Right p1.s p2.s) ~hi:(at `Left p1.s p2.t)
      p2.tk
  in
  (known_stops :: unknown_stops) @ [ lefts ]

(* [column side ~n ~m lg rg bounds] is the band of the column of [side] whose
   [bounds] are [(op, x, v)]: the column's words [x] by [op] to the other
   side's [v]. *)
let column side ~n ~m lg rg bounds =
  let _, x, _ = List.hd bounds in
  let xg, pg, probes = if side = `Right then (rg, lg, n) else (lg, rg, m) in
  let o = order xg x.known x.w in
  let at = search o pg in
  let tighten f r v = r := Some (match !r with None -> v | Some u -> f u v) in
  let lo = ref None and hi = ref None in
  List.iter
    (fun (op, x, p) ->
      match op with
      | `Lt -> tighten Nx.minimum hi (at `Left x.w p.w)
      | `Le -> tighten Nx.minimum hi (at `Right x.w p.w)
      | `Gt -> tighten Nx.maximum lo (at `Right x.w p.w)
      | `Ge -> tighten Nx.maximum lo (at `Left x.w p.w))
    bounds;
  let known = List.fold_left (fun k (_, _, p) -> both k p.known) None bounds in
  band side o pg ~probes ?lo:!lo ?hi:!hi known

(* [in_left_order n counts (li, ri)] is the pairs in the join's order: by left
   row, then right row. A left row of one match at most is placed by a
   scatter. *)
let in_left_order n counts (li, ri) =
  if ascending (Some li) ri then (li, ri)
  else if Nx.item [] (Nx.max counts) <= 1L then
    let ri = Nx.scatter ~axis:0 li ri (none n) in
    let li = Nx.positions (Nx.greater_equal_s ri 0L) in
    (li, Nx.take li ri)
  else
    let p = Nx.lexsort (Nx.stack ~axis:1 [ li; ri ]) in
    (Nx.take p li, Nx.take p ri)

let range q kind counts left right atoms =
  let eqs =
    List.filter_map (function Join.Eq (l, r) -> Some (l, r) | _ -> None) atoms
  in
  let keyed = keyed left right eqs and names = names eqs in
  let typed side n =
    let q = if side = `Left then left else right in
    Option.get (Schema.find (Query.schema q) n)
  in
  (* [read cs] reads the columns [cs] of their sides at the type they meet. *)
  let read cs =
    let types = List.map (fun (s, n) -> typed s n) cs in
    let t = Option.get (Type.common_any types) in
    let cs = List.map2 (fun (s, n) ty -> (s, n, Eval.widen ty t)) cs types in
    fun l r ->
      ordered
        (List.map
           (fun (s, n, widen) ->
             widen (Table.column (if s = `Left then l else r) n))
           cs)
  in
  let pred = function
    | Join.Cmp (op, a, b) -> (
        let read = read [ (`Left, a); (`Right, b) ] in
        fun l r ->
          match read l r with
          | [ wl; wr ] -> Compared (op, wl, wr)
          | _ -> assert false)
    | Overlaps ((ls, le), (rs, re)) -> (
        let read =
          read [ (`Left, ls); (`Left, le); (`Right, rs); (`Right, re) ]
        in
        fun l r ->
          match read l r with
          | [ d1; e1; d2; e2 ] -> Periods (period d1 e1, period d2 e2)
          | _ -> assert false)
    | Eq _ | Position -> assert false
  in
  let shape, banded, others =
    band_atoms (List.filter (function Join.Eq _ -> false | _ -> true) atoms)
  in
  (* A bound reads the band's column at the type it meets it at, and those
     types may order the column apart, as a categorical orders by code and
     text by bytes: the band holds the bounds of the first one's type, and
     the others are tested on its pairs. *)
  let banded, others =
    match shape with
    | `Periods -> (banded, others)
    | `Column (side, name) ->
        let meet = function
          | Join.Cmp (_, l, r) ->
              let os, on = if side = `Left then (`Right, r) else (`Left, l) in
              Option.get (Type.common_any [ typed side name; typed os on ])
          | _ -> assert false
        in
        let (Type.Any first) = meet (List.hd banded) in
        let same a =
          let (Type.Any t) = meet a in
          Type.equal t first
        in
        let banded, apart = List.partition same banded in
        (banded, apart @ others)
  in
  let others = List.map pred others in
  let bands =
    match shape with
    | `Periods -> (
        let read = pred (List.hd banded) in
        fun ~n ~m lg rg l r ->
          match read l r with
          | Periods (p1, p2) -> periods ~n ~m lg rg p1 p2
          | Compared _ -> assert false)
    | `Column (side, name) ->
        let bound = function
          | Join.Cmp (op, l, r) ->
              if side = `Right then (mirror op, (`Left, l))
              else (op, (`Right, r))
          | _ -> assert false
        in
        let bounds = List.map bound banded in
        let columns = (side, name) :: List.map snd bounds in
        (* The band's column and its bounds meet at the bounds' one type. *)
        let read =
          let read = read columns in
          fun l r ->
            match read l r with
            | x :: vs -> List.map (fun v -> (x, v)) vs
            | [] -> assert false
        in
        fun ~n ~m lg rg l r ->
          let words = read l r in
          [
            column side ~n ~m lg rg
              (List.map2 (fun (op, _) (x, v) -> (op, x, v)) bounds words);
          ]
  in
  (* Groups are numbered in order of first appearance on the side that a band
     sorts, the right one's for [overlaps], so that a side whose rows come
     grouped by key is in the band's order when each group is. *)
  let first = match shape with `Column (side, _) -> side | `Periods -> `Right in
  fun l r ->
    let n = Table.rows l and m = Table.rows r in
    let keys = keyed l r in
    let lg, rg =
      match (keys, first) with
      | [], _ -> (None, None)
      | keys, `Left ->
          let g = (Key.groups ~size:0 (List.map snd keys)).ids in
          let g = Nx.cast Nx.uint64 g in
          ( Some (Nx.slice [ Nx.R (0, n) ] g),
            Some (Nx.slice [ Nx.R (n, n + m) ] g) )
      | keys, `Right ->
          let swap (_, c) =
            Column.concat
              [
                Column.sub c ~offset:n ~length:m;
                Column.sub c ~offset:0 ~length:n;
              ]
          in
          let g = (Key.groups ~size:0 (List.map swap keys)).ids in
          let g = Nx.cast Nx.uint64 g in
          ( Some (Nx.slice [ Nx.R (m, m + n) ] g),
            Some (Nx.slice [ Nx.R (0, m) ] g) )
    in
    let bands = bands ~n ~m lg rg l r in
    let others = List.map (fun p -> p l r) others in
    (* The bands' pairs, those on which every other atom holds. *)
    let found =
      lazy
        (let ps = List.map pairs bands in
         let li = Nx.concatenate ~axis:0 (List.map fst ps)
         and ri = Nx.concatenate ~axis:0 (List.map snd ps) in
         match others with
         | [] -> (li, ri)
         | p :: ps ->
             let keep =
               List.fold_left
                 (fun k p -> Nx.logical_and k (holds (li, ri) p))
                 (holds (li, ri) p)
                 ps
             in
             let at = Nx.positions keep in
             (Nx.take at li, Nx.take at ri))
    in
    let tally side rows =
      lazy
        (let i = side (Lazy.force found) in
         Nx.reduce_segments Nx.Add ~segments:rows i (Nx.ones_like i))
    in
    let lcounts, rcounts =
      match others with
      | [] ->
          let sum = function
            | [] -> assert false
            | c :: cs -> List.fold_left Nx.add c cs
          in
          let cs = lazy (List.map (band_counts ~n ~m) bands) in
          ( lazy (sum (List.map fst (Lazy.force cs))),
            lazy (sum (List.map snd (Lazy.force cs))) )
      | _ -> (tally fst n, tally snd m)
    in
    let matches =
      lazy
        (let _, ri = in_left_order n (Lazy.force lcounts) (Lazy.force found) in
         Runs ri)
    in
    emit q kind counts l r names keys lcounts rcounts matches

(* Position and all joins *)

let position q kind (each_left, each_right) l r =
  let n = Table.rows l and m = Table.rows r in
  let* () = sized "left" each_left ~rows:n (m, 1) 0 in
  let* () = sized "right" each_right ~rows:m (n, 1) 0 in
  let idx =
    match (kind : Join.kind) with
    | Inner | Semi -> arange (Int.min n m)
    | Left -> arange n
    | Full -> arange (Int.max n m)
    | Anti -> Nx.arange Nx.int64 (Int.min n m) n 1
  in
  let li = if kind = Left then All else At idx in
  Ok (assemble q kind l li r idx [])

let cross q kind (each_left, each_right) =
  let batch r b =
    let n = Table.rows b and m = Table.rows r in
    let* () = sized "left" each_left ~rows:n (0, 0) m in
    let kept keep = if keep then arange n else arange 0 in
    let li, ri =
      match (kind : Join.kind) with
      | (Inner | Left | Full) when m > 0 ->
          let p = arange (n * m) in
          (Nx.div_s p (Int64.of_int m), Nx.mod_s p (Int64.of_int m))
      | Inner -> (arange 0, arange 0)
      | Left | Full -> (arange n, none n)
      | Semi -> (kept (m > 0), arange 0)
      | Anti -> (kept (m = 0), arange 0)
    in
    Ok (assemble q kind b (At li) r ri [])
  in
  let last r e n =
    let m = Table.rows r in
    let* () = sized "right" each_right ~rows:m (0, 0) n in
    if kind = Full && n = 0 then
      Ok (assemble q kind e (At (none m)) r (arange m) [])
    else Ok (assemble q kind e (At (arange 0)) r (arange 0) [])
  in
  Streaming { batch; last }

let compile q =
  match Query.node q with
  | Join { kind; each_left; each_right; on; left; right } -> (
      let counts = (each_left, each_right) in
      let eq = function Join.Eq (l, r) -> Some (l, r) | _ -> None in
      match (on :> Join.atom list) with
      | [] -> cross q kind counts
      | [ Position ] -> Blocking (position q kind counts)
      | atoms
        when List.for_all (function Join.Eq _ -> true | _ -> false) atoms ->
          Blocking
            (equality q kind counts left right (List.filter_map eq atoms))
      | atoms -> Blocking (range q kind counts left right atoms))
  | _ -> invalid_arg "Join_run.compile: not a join"
