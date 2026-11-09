(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = Table.t

module Binary = Binary
module Time = Time
module Kind = Kind
module Record = Record
module Type = Type
module Schema = Schema

module Column = struct
  include Column

  let parse = Form.parse
  let print = Form.print
end

let v = Table.v
let of_batches = Table.of_batches
let batches = Table.batches
let schema = Table.schema
let rows = Table.rows
let column t name = Column.canonical (Table.column t name)
let take = Table.take
let to_tensor = Table.to_tensor
let to_dense = Table.to_dense
let of_dense = Table.of_dense
let equal = Table.equal

type limits = Display.limits = {
  head : int;
  tail : int;
  columns : int;
  width : int;
}

let limits = Display.limits
let pp_with = Display.pp
let pp = pp_with limits

module Error = Error
module Tz = Tz
module Sel = Sel
module Order = Order
module Expr = Expr

module Col = struct
  let v k name = Expr.make (Expr.Handle (k, name))
  let bool name = v Kind.bool name
  let int name = v Kind.int name
  let float name = v Kind.float name
  let string name = v Kind.string name
  let binary name = v Kind.binary name
  let date name = v Kind.date name
  let instant name = v Kind.instant name
  let wall name = v Kind.wall name
  let span name = v Kind.span name
end

module Ext = struct
  type ('e, 's) t = ('e, 's) Expr.ext

  let v ~name ?metadata ~ordered storage ~dec ~enc : _ t =
    { type_ = Type.ext ~name ?metadata storage; storage; ordered; dec; enc }

  let col e name = Expr.make (Expr.Ext_handle (e, name))
end

module Source = Source
module Join = Join

module Query = struct
  include Query

  let optimize = Optimize.query
  let fold = Run.fold
  let run = Run.run
  let values = Run.values
end

module Kit = struct
  (* Every composition builds its query with the verbs that [Talon] exports, so
     that its problems are its verbs'. *)

  let err fmt = Format.kasprintf invalid_arg fmt
  let column_names q = List.map fst (Schema.columns (Query.schema q))

  (* [repeated ns] is a name that [ns] holds twice, if any. *)
  let rec repeated = function
    | [] -> None
    | n :: ns -> if List.mem n ns then Some n else repeated ns

  (* Queries *)

  let head n q = Query.slice ~offset:0 ~length:n q
  let tail n q = Query.slice ~offset:(-n) ~length:n q
  let top_k k keys q = Query.slice ~offset:0 ~length:k (Query.sort keys q)

  let distinct q =
    match column_names q with
    | [] -> head 1 q
    | names -> Query.aggregate ~by:names [] q

  let count_by ks q = Query.aggregate ~by:ks Expr.[ "count" := rows ] q
  let value_counts c q = Query.sort [ Order.desc "count" ] (count_by [ c ] q)

  let describe q =
    let stats name x =
      Query.aggregate ~by:[]
        Expr.
          [
            "column" := string name;
            "count" := count x;
            "nulls" := rows - count x;
            "mean" := mean x;
            "std" := std x;
            "min" := cast Type.float64 (min x);
            "q25" := quantile 0.25 x;
            "median" := median x;
            "q75" := quantile 0.75 x;
            "max" := cast Type.float64 (max x);
          ]
        q
    in
    let numeric (n, Type.Any t) =
      let k = Type.kind t in
      match
        (Kind.provably_equal k Kind.int, Kind.provably_equal k Kind.float)
      with
      | Some Stdlib.Type.Equal, _ -> Some (stats n (Col.int n))
      | _, Some Stdlib.Type.Equal -> Some (stats n (Col.float n))
      | None, None -> None
    in
    match List.filter_map numeric (Schema.columns (Query.schema q)) with
    | [] -> head 0 (stats "" Expr.(store Type.float64 null))
    | s :: ss -> List.fold_left (fun acc s -> Query.append s acc) s ss

  let null_count q =
    Query.aggregate ~by:[]
      Expr.[ each Sel.all { column = (fun n x -> n := rows - count x) } ]
      q

  let drop sel q = Query.select Expr.[ keep Sel.(all - sel) ] q

  let rename pairs q =
    let olds = List.map fst pairs in
    Option.iter
      (err "Kit.rename: %a is renamed twice" Type.pp_quoted)
      (repeated olds);
    let name n = Option.value ~default:n (List.assoc_opt n pairs) in
    Query.select
      Expr.[ each Sel.(all + names olds) { column = (fun n x -> name n := x) } ]
      q

  let complete ks q =
    Option.iter
      (err "Kit.complete: %a is named twice" Type.pp_quoted)
      (repeated ks);
    match ks with
    | [] -> invalid_arg "Kit.complete: no column"
    | k :: ks' ->
        let values k =
          distinct (Query.select Expr.[ keep Sel.(names [ k ]) ] q)
        in
        let combos =
          List.fold_left
            (fun acc k -> Query.join ~on:Join.all (values k) acc)
            (values k) ks'
        in
        combos
        |> Query.join ~kind:Left ~on:(Join.keys ks) q
        |> Query.select Expr.[ keep (Sel.names (column_names q)) ]

  let one_hot c q =
    let s = Query.schema q in
    match Schema.find s c with
    | None ->
        (* [q] lacks [c], so the select raises its report. *)
        Query.select Expr.[ keep Sel.(names [ c ]) ] q
    | Some (Type.Any (Categorical categories)) ->
        let rec split before = function
          | (n, _) :: after when String.equal n c ->
              (List.rev before, List.map fst after)
          | (n, _) :: rest -> split (n :: before) rest
          | [] -> assert false (* [Schema.find] found [c]. *)
        in
        let before, after = split [] (Schema.columns s) in
        let indicator cat = Expr.(c ^ "_" ^ cat := Col.string c = string cat) in
        Query.select
          (Expr.keep (Sel.names before)
           :: List.map indicator (Iarray.to_list categories)
          @ [ Expr.keep (Sel.names after) ])
          q
    | Some (Type.Any t) ->
        err
          "Kit.one_hot: %a is %a, not categorical: cast it to a categorical \
           type first"
          Type.pp_quoted c Type.pp t

  let union rest q =
    let nulls from into =
      List.filter_map
        (fun (n, Type.Any t) ->
          match Schema.find into n with
          | Some _ -> None
          | None -> Some Expr.(n := store t null))
        (Schema.columns from)
    in
    let pad os q = if List.is_empty os then q else Query.derive os q in
    let s = Query.schema q and r = Query.schema rest in
    Query.append (pad (nulls s r) rest) (pad (nulls r s) q)

  (* [meet ts] is the type where the types [ts] of one column meet. Two kinds
     never meet: {!Type.common} says so. *)
  let meet : Type.any list -> Type.any option = function
    | [] -> None
    | Type.Any t0 :: ts ->
        let at : type a. a Type.t -> Type.any -> a Type.t option =
         fun t0 (Type.Any t) ->
          match Kind.provably_equal (Type.kind t) (Type.kind t0) with
          | Some Equal -> Some t
          | None -> None
        in
        let same = List.filter_map (at t0) ts in
        if List.compare_lengths same ts <> 0 then None
        else Option.map (fun t -> Type.Any t) (Type.common (t0 :: same))

  let union_all = function
    | [] -> Query.of_table (v ~rows:0 [])
    | qs ->
        let order = ref [] and types = Hashtbl.create 16 in
        let see (n, t) =
          match Hashtbl.find_opt types n with
          | Some ts -> ts := t :: !ts
          | None ->
              order := n :: !order;
              Hashtbl.add types n (ref [ t ])
        in
        List.iter (fun q -> List.iter see (Schema.columns (Query.schema q))) qs;
        (* Each column at its common type over every branch, which the order
           of the branches cannot change; a column that does not meet is left
           to [append] to report. *)
        let columns =
          List.rev_map (fun n -> (n, meet !(Hashtbl.find types n))) !order
        in
        let conform q =
          let s = Query.schema q in
          let out (n, common) =
            match (common, Schema.find s n) with
            | None, _ -> None
            | Some (Type.Any c), None -> Some Expr.(n := store c null)
            | Some (Type.Any c), Some (Type.Any t) when Type.equal c t -> None
            | Some (Type.Any c), Some (Type.Any t) ->
                Some Expr.(n := cast c (Col.v (Type.kind t) n))
          in
          match List.filter_map out columns with
          | [] -> q
          | os -> Query.derive os q
        in
        let branches = Array.of_list (List.map conform qs) in
        let rec join lo hi =
          if hi - lo = 1 then branches.(lo)
          else
            let mid = (lo + hi) / 2 in
            Query.append (join mid hi) (join lo mid)
        in
        join 0 (Array.length branches)

  (* Runs *)

  (* A dictionary's codes are int32. *)
  let max_categories = 0x7fff_ffff

  (* Windows *)

  let cumulative r = Expr.window r
  let fill_forward x = Expr.(window (last x))

  let fill_backward x =
    Expr.(window ~before:(int 1) ~after:(int max_int) (first x))

  let interpolate ?on x =
    let open Expr in
    let line ~at ~x0 ~k0 ~x1 ~k1 =
      coalesce
        [ x; if_ (at = k0) x0 (x0 +. ((x1 -. x0) *. (at -. k0) /. (k1 -. k0))) ]
    in
    match on with
    | Some on ->
        let k = cast Type.float64 on in
        let held v = if_ (is_null x) null v in
        let upto r = window ~on:k r in
        let past r =
          window ~on:k ~before:(float 0.) ~after:(float infinity) r
        in
        line ~at:k ~x0:(upto (last x)) ~k0:(upto (last (held k)))
          ~x1:(past (first x)) ~k1:(past (first (held k)))
    | None ->
        (* [c] values reach the row; the [c]th starts the run of rows of count
           [c], which ends before the next value. *)
        let place r = cast Type.float64 r in
        let c = window (count x) in
        let k0 = place (window ~on:c ~exclude_key:true rows + int 1) in
        let k1 = k0 +. place (window ~on:c ~before:(int 1) rows) in
        line ~at:(place (window rows)) ~x0:(window (last x)) ~k0
          ~x1:(window ~before:(int 0) ~after:(int max_int) (first x))
          ~k1

  (* As-of joins *)

  (* [type_of q c] is the type of [q]'s column [c]; a missing [c] is the
     select's problem. *)
  let type_of q c =
    let q = Query.select Expr.[ keep Sel.(names [ c ]) ] q in
    match Schema.columns (Query.schema q) with
    | [ (_, t) ] -> t
    | _ -> assert false

  (* [fresh taken base] is [base], primed until [taken] lacks it. *)
  let rec fresh taken base =
    if List.mem base taken then fresh taken (base ^ "'") else base

  (* [named n c] outputs the column [c] as [n], at any type. *)
  let named n c =
    Expr.each (Sel.names [ c ]) { column = (fun _ x -> Expr.(n := x)) }

  type direction = Backward | Forward | Nearest

  (* [nearer k] is, for a kind [k] with a distance, whether [a] is at most as
     far from [l] as [b] is, where [a <= l <= b]. *)
  let nearer (type a) (k : a Kind.t) :
      ((a, Expr.row) Expr.t ->
      (a, Expr.row) Expr.t ->
      (a, Expr.row) Expr.t ->
      (bool, Expr.row) Expr.t)
      option =
    let is k' = Option.is_some (Kind.provably_equal k k') in
    let ints = Kind.provably_equal k Kind.int in
    let floats = Kind.provably_equal k Kind.float in
    let open Expr in
    match (ints, floats) with
    | Some Stdlib.Type.Equal, _ ->
        (* The distances wrap at the type's width and compare unsigned. *)
        Some
          (fun l a b ->
            let da = l - a and db = b - l in
            if_ ((da < int 0) = (db < int 0)) (da <= db) (da >= int 0))
    | _, Some Stdlib.Type.Equal ->
        (* Rounding is monotone, so the rounded distances decide unless they
           are equal; TwoSum's residuals, exact while the sums are finite, then
           decide. A distance is infinite where a value is infinite or NaN. *)
        Some
          (fun l a b ->
            let l = cast Type.float64 l and a = cast Type.float64 a in
            let b = cast Type.float64 b in
            let finite x = x > float neg_infinity && x < float infinity in
            let two_sum x y =
              let s = x -. y in
              let r = s -. x in
              (s, x -. (s -. r) -. (y +. r))
            in
            let sa, ea = two_sum l a and sb, eb = two_sum b l in
            a = l
            || b <> l
               && ((not (finite b && finite l))
                  || (finite a && (sa < sb || (sa = sb && ea <= eb)))))
    | None, None when Stdlib.(is Kind.date || is Kind.instant || is Kind.wall || is Kind.span)
      ->
        Some (fun l a b -> Temporal.diff l a <= Temporal.diff b l)
    | None, None -> None

  let join_asof ?(by = []) ?(direction = Backward) ?(strict = false) ~on:(l, r)
      right left =
    Option.iter
      (err "Kit.join_asof: %a is named twice" Type.pp_quoted)
      (repeated by);
    let taken = by @ column_names left @ column_names right in
    let key = fresh taken "key" and side = fresh taken "side" in
    let j = fresh taken "j" and row = fresh taken "row" in
    let tagged q c s number =
      Query.select
        Expr.(keep (Sel.names by) :: named key c :: [ side := int s; number ])
        q
    in
    (* The right rows count from 1, and a null key numbers none. *)
    let number =
      let column _ x = Expr.(j := if_ (is_null x) null (window rows)) in
      Expr.each (Sel.names [ r ]) { column }
    in
    let frame =
      tagged left l 1 Expr.(j := store Type.int64 null)
      |> Query.append (tagged right r 0 number)
    in
    (* Along the key, a key's right rows come before its left rows where a
       backward pick takes them or a forward pick skips them. *)
    let pick ~forward x =
      let ahead = if forward = strict then Order.asc else Order.desc in
      let open Expr in
      let along = over ~by ~order:[ Order.asc key; ahead side ] in
      if forward then
        along (window ~before:(int 1) ~after:(int max_int) (first x))
      else along (window (last x))
    in
    let jx = Col.int j in
    let picked =
      match direction with
      | Backward -> pick ~forward:false jx
      | Forward -> pick ~forward:true jx
      | Nearest -> (
          let (Type.Any t) = type_of frame key in
          match nearer (Type.kind t) with
          | None ->
              err
                "Kit.join_asof: Nearest takes numbers, dates, datetimes, \
                 durations or clocks, not %a"
                Type.pp t
          | Some nearer ->
              let open Expr in
              let k = Col.v (Type.kind t) key in
              let held = if_ (is_null jx) null k in
              let ja = pick ~forward:false jx and jb = pick ~forward:true jx in
              (* A row with one pick compares no distance. *)
              let both x = if_ (is_null ja || is_null jb) null x in
              let ka = both (pick ~forward:false held) in
              let kb = both (pick ~forward:true held) in
              let nearest = if_ (nearer k ka kb) ja jb in
              if_ (is_null jb) ja (if_ (is_null ja) jb nearest))
    in
    (* A null left key matches nothing. *)
    let picked =
      let column _ x = Expr.(row := if_ (is_null x) null picked) in
      Expr.each (Sel.names [ key ]) { column }
    in
    let picks =
      frame
      |> Query.derive [ picked ]
      |> Query.filter Expr.(Col.int side = int 1)
      |> Query.select Expr.[ keep (Sel.names [ row ]) ]
    in
    let right = drop (Sel.names by) right in
    left
    |> Query.join ~on:Join.position picks
    |> Query.join ~kind:Left ~on:(Join.keys [ row ])
         (Query.derive Expr.[ row := window rows ] right)
    |> drop (Sel.names [ row ])

  (* Interval sets *)

  (* [sweep name ~by (s, e) holds q rest] is the normal form of the values at
     which [holds a b] is true, [a] and [b] counting the periods of [q] and of
     [rest] that hold them. *)
  let sweep name ~by (s, e) holds q rest =
    Option.iter
      (err "Kit.%s: %a is named twice" name Type.pp_quoted)
      (repeated by);
    let taken = by @ [ s; e ] in
    let at = fresh taken "at" and a = fresh taken "a" in
    let b = fresh taken "b" and held = fresh taken "held" in
    let events (q, (wa, wb)) =
      let (Type.Any t) = type_of q s in
      let s = Col.v (Type.kind t) s and e = Col.v (Type.kind t) e in
      let start = Expr.(if_ (s <= e) s e) in
      let stop = Expr.(if_ (s <= e) e s) in
      let q = Query.filter Expr.(start < stop) q in
      let edge x (wa, wb) =
        Query.select
          Expr.[ keep (Sel.names by); at := x; a := int wa; b := int wb ]
          q
      in
      edge start (wa, wb) |> Query.append (edge stop (-wa, -wb))
    in
    let keys = by @ [ at ] in
    let running c = Expr.(over ~by (window (sum (Col.int c)))) in
    let held' = Col.bool held in
    let stop =
      let column _ x = Expr.(e := over ~by (shift (-1) x)) in
      Expr.each (Sel.names [ at ]) { column }
    in
    let events =
      match rest with
      | None -> events (q, (1, 0))
      | Some rest -> Query.append (events (rest, (0, 1))) (events (q, (1, 0)))
    in
    events
    |> Query.sort (List.map Order.asc keys)
    |> Query.aggregate ~by:keys
         Expr.[ a := sum (Col.int a); b := sum (Col.int b) ]
    |> Query.derive Expr.[ held := holds (running a) (running b) ]
    |> Query.filter
         Expr.(held' <> coalesce [ over ~by (shift 1 held'); bool false ])
    |> Query.derive [ stop ]
    |> Query.filter held'
    |> Query.select
         Expr.(keep (Sel.names by) :: named s at :: [ keep (Sel.names [ e ]) ])

  let merge_intervals ?(by = []) period q =
    sweep "merge_intervals" ~by period Expr.(fun a _ -> a > int 0) q None

  let intersect_intervals ?(by = []) period rest q =
    let holds a b = Expr.(a > int 0 && b > int 0) in
    sweep "intersect_intervals" ~by period holds q (Some rest)

  let subtract_intervals ?(by = []) period rest q =
    let holds a b = Expr.(a > int 0 && b = int 0) in
    sweep "subtract_intervals" ~by period holds q (Some rest)

  let categorize cs q =
    Option.iter
      (err "Kit.categorize: %a is named twice" Type.pp_quoted)
      (repeated cs);
    (* Every plan is built, and so checked, before any runs. *)
    let words c =
      let x = Col.string c in
      let plan =
        Query.select Expr.[ c := cast Type.string x ] q
        |> Query.filter Expr.(not (is_null x))
        |> distinct
        |> Query.sort [ Order.asc c ]
      in
      fun () -> Query.values x plan
    in
    let queries = List.map (fun c -> (c, words c)) cs in
    let rec run outputs = function
      | [] -> Ok (if outputs = [] then q else Query.derive (List.rev outputs) q)
      | (c, words) :: rest -> (
          match words () with
          | Error e -> Error e
          | Ok ws when Array.length ws > max_categories ->
              Error
                (Error.v
                   (Format.asprintf
                      "Kit.categorize: %a holds more than 2^31 - 1 strings"
                      Type.pp_quoted c))
          | Ok ws ->
              let o = Expr.(c := cast (Type.categorical ws) (Col.string c)) in
              run (o :: outputs) rest)
    in
    run [] queries

  (* Reshaping *)

  let longer ~name ~value sel q =
    let ns = column_names (Query.select Expr.[ keep sel ] q) in
    if ns = [] then err "Kit.longer: the selector selects no column";
    let selected = Hashtbl.create (List.length ns) in
    List.iter (fun n -> Hashtbl.replace selected n ()) ns;
    let others =
      List.filter (fun n -> not (Hashtbl.mem selected n)) (column_names q)
    in
    let check c =
      if List.mem c others then
        err "Kit.longer: %a is a column the selector does not select"
          Type.pp_quoted c
    in
    check name;
    check value;
    let (Type.Any t) = type_of q (List.hd ns) in
    let k = Type.kind t in
    if Option.is_none (Kind.provably_equal k k) then
      err "Kit.longer: %a holds an extension, which no handle reads"
        Type.pp_quoted (List.hd ns);
    let cells = Array.of_list ns in
    q
    |> Query.unnest
         Expr.[
           name := store Type.(list (categorical cells)) (const cells);
           value := list (List.map (Col.v k) ns);
         ]
    |> Query.select Expr.[ keep (Sel.names (others @ [ name; value ])) ]

  let wider ~by ~key ~value q =
    let column n x = Expr.(n := pivot (Col.string key) x) in
    Query.aggregate ~by Expr.[ each (Sel.names [ value ]) { column } ] q
    (* A record is never null, so this keeps every row. It reads the pivot, so
       that the pivot's failures happen when no output reads a field. *)
    |> Query.filter Expr.(not (is_null (Col.v Record.kind value)))
    |> Query.select
         Expr.[
           keep (Sel.names by); within (Col.v Record.kind value) (keep Sel.all);
         ]

  (* Randomness *)

  (* [seed_of s] is [s], or a seed drawn when the plan is built, so that a plan
     always returns the same rows. *)
  let seed_of = function
    | Some s -> s
    | None ->
        let w = Nx.Rng.bits [| 2 |] in
        (Int32.to_int (Nx.item [ 0 ] w) lsl 31)
        lxor (Int32.to_int (Nx.item [ 1 ] w) land 0x7fff_ffff)

  let place_hash ~seed = Expr.(hash ~seed (window rows))

  let shuffle ?seed q =
    let seed = seed_of seed in
    let key = fresh (column_names q) "key" in
    Query.derive Expr.[ key := place_hash ~seed ] q
    |> Query.sort [ Order.asc key ]
    |> drop (Sel.names [ key ])

  (* The hash is below 2^62, so a share of that range keeps that share of
     rows. *)
  let hash_range = 4611686018427387904.

  let sample ?seed size q =
    let seed = seed_of seed in
    match size with
    | `Rows n ->
        if n < 0 then err "Kit.sample: %d rows, below 0" n;
        let key = fresh (column_names q) "key" in
        Query.derive Expr.[ key := place_hash ~seed ] q
        |> top_k n [ Order.asc key ]
        |> drop (Sel.names [ key ])
    | `Fraction f ->
        if not (f >= 0. && f <= 1.) then
          err "Kit.sample: the fraction %g is not in [0, 1]" f;
        if f >= 1. then q
        else
          let cutoff = Int.of_float (f *. hash_range) in
          Query.filter Expr.(place_hash ~seed < int cutoff) q

  let folds ?seed ~by k q =
    if by = [] then err "Kit.folds: no key";
    if k < 1 then err "Kit.folds: %d folds, below 1" k;
    Option.iter
      (err "Kit.folds: %a is named twice" Type.pp_quoted)
      (repeated by);
    if List.mem "fold" (column_names q) then
      err "Kit.folds: the query already has a column %a" Type.pp_quoted "fold";
    let seed = seed_of seed in
    let taken = by @ column_names q in
    let h = fresh taken "h" and g = fresh taken "g" in
    let groups = Query.select Expr.[ keep (Sel.names by) ] q |> distinct in
    let enough n =
      if n < k then
        err "Kit.folds: %d folds need at least as many groups, not %d" k n;
      n
    in
    let count =
      Query.aggregate ~by:[]
        Expr.[ g := store Type.int64 (const enough $ rows) ]
        groups
    in
    let key = Expr.(record [ keep (Sel.names by) ]) in
    let folded =
      groups
      |> Query.join ~on:Join.all count
      |> Query.derive Expr.[ h := hash ~seed key ]
      |> Query.sort [ Order.asc h ]
      |> Query.derive
           Expr.[ "fold" := (window rows - int 1) * int k / Col.int g ]
      |> drop (Sel.names [ h; g ])
    in
    q |> Query.join ~on:(Join.keys by) folded

  (* Time ranges *)

  let time_range (type a) (ty : a Type.t) ~start ~stop step name =
    (match step with
    | (Time.Months n | Time.Weeks n | Time.Days n) when n > 0 -> ()
    | Time.Exact d when Time.Span.to_ns d > 0L -> ()
    | step -> err "Kit.time_range: %a is not positive" Time.pp_step step);
    let c = Column.v ty [| start; stop |] in
    (* Binding a move by [step] reports a type or step it does not take. *)
    let x = Expr.make (Expr.Read (ty, name)) in
    ignore
      (Query.derive Expr.[ name := Temporal.offset step x ]
         (Query.of_table (v [ (name, c) ])));
    let one i = Column.sub c ~offset:i ~length:1 in
    v [ (name, Kernels.range (Any ty) ~start:(one 0) ~stop:(one 1) step) ]
end

let duration_sum xs ~copies =
  let x = Nx.create Nx.int64 [| Array.length xs |] xs in
  let sum y = Nx.mul_s (Nx.reshape [| 1 |] (Nx.sum y)) (Int64.of_int copies) in
  let c, _ = Reduce.exact_sum (Type.duration Ns) sum x in
  (Column.options Kind.span c).(0) |> Option.map Time.Span.to_ns
