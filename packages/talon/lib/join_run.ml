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
  Error { Eval.row; cause = Data (Error.v (why ^ ".")) }

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
   group, in order, are one row of [runs]. *)

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

let equality q kind (each_left, each_right) left right eqs =
  let widen (ln, rn) =
    let tl = Option.get (Schema.find (Query.schema left) ln)
    and tr = Option.get (Schema.find (Query.schema right) rn) in
    let t = Option.get (Join.common_type tl tr) in
    (ln, rn, Eval.widen tl t, Eval.widen tr t)
  in
  let atoms = List.map widen eqs in
  fun l r ->
    let n = Table.rows l and m = Table.rows r in
    let keys =
      List.map
        (fun (ln, rn, wl, wr) ->
          (ln, Column.concat [ wl (Table.column l ln); wr (Table.column r rn) ]))
        atoms
    in
    let g = Key.groups (List.map snd keys) in
    let groups = Nx.dim 0 g.first in
    let lids = Nx.slice [ Nx.R (0, n) ] g.ids
    and rids = Nx.slice [ Nx.R (n, n + m) ] g.ids in
    (* Each group's right rows, and so each left row's matches. *)
    let ones = Nx.ones_like rids in
    let per = Nx.reduce_segments Nx.Add ~segments:groups rids ones in
    let counts = Nx.take lids per in
    let matches =
      if groups = 0 || Nx.item [] (Nx.max per) <= 1L then
        One (Nx.take lids (Nx.scatter ~axis:0 rids (arange m) (none groups)))
      else
        let runs = Nx_ragged.of_ids ~segments:groups rids (arange m) in
        Runs (Nx_ragged.values (Nx_ragged.take ~indices:lids runs))
    in
    (* A right row's matches are the left rows of its group: the others of its
       group when it is the group's one right row. *)
    let rcounts =
      lazy
        (match matches with
        | One _ -> Nx.sub_s (Nx.take rids g.counts) 1L
        | Runs matched ->
            Nx.reduce_segments Nx.Add ~segments:m matched
              (Nx.ones_like matched))
    in
    let* () =
      check "left" each_left (named l (List.map fst eqs)) (lazy counts)
    in
    let* () = check "right" each_right (named r (List.map snd eqs)) rcounts in
    let left_pairs () =
      match matches with
      | One ri -> (All, ri)
      | Runs matched -> left_pairs counts matched
    in
    let pairs (li, ri) = Ok (assemble q kind l li r ri []) in
    let only keep = pairs (At (Nx.positions keep), arange 0) in
    match (kind : Join.kind) with
    | Inner -> (
        let li = Nx.positions counts in
        match matches with
        | One ri -> pairs (At li, Nx.take li ri)
        | Runs matched -> pairs (At li, matched))
    | Semi -> only (Nx.greater_s counts 0L)
    | Anti -> only (Nx.equal_s counts 0L)
    | Left -> pairs (left_pairs ())
    | Full ->
        (* A right row without a match follows, its keys in the left's. *)
        let li, ri = left_pairs () in
        let li = match li with All -> arange n | At li -> li in
        let u = Nx.positions (Nx.equal_s (Lazy.force rcounts) 0L) in
        let at = Nx.concatenate ~axis:0 [ li; Nx.add_s u (Int64.of_int n) ] in
        let coalesced =
          List.map (fun (ln, k) -> (ln, Column.gather at k)) keys
        in
        let li = Nx.concatenate ~axis:0 [ li; none (Nx.dim 0 u) ]
        and ri = Nx.concatenate ~axis:0 [ ri; u ] in
        Ok (assemble q kind l (At li) r ri coalesced)

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
      let eq = function Join.Eq (l, r) -> Some (l, r) | Position -> None in
      match (on :> Join.atom list) with
      | [] -> cross q kind counts
      | [ Position ] -> Blocking (position q kind counts)
      | atoms ->
          let eqs = List.filter_map eq atoms in
          Blocking (equality q kind counts left right eqs))
  | _ -> invalid_arg "Join_run.compile: not a join"
