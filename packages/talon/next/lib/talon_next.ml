(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = Table.t

module Binary = Binary
module Decimal = Decimal
module Time = Time
module Kind = Kind
module Record = Record
module Type = Type
module Schema = Schema

module Column = struct
  include Column

  let parse = Form.parse
end

let v = Table.v
let of_batches = Table.of_batches
let batches = Table.batches
let schema = Table.schema
let rows = Table.rows
let column = Table.column
let take = Table.take
let to_tensor = Table.to_tensor
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
module Window = Window
module Expr = Expr

module Col = struct
  let v k name = Expr.make (Expr.Handle (k, name))
  let bool name = v Kind.bool name
  let int name = v Kind.int name
  let float name = v Kind.float name
  let string name = v Kind.string name
  let binary name = v Kind.binary name
  let decimal name = v Kind.decimal name
  let date name = v Kind.date name
  let instant name = v Kind.instant name
  let span name = v Kind.span name
end

module Ext = struct
  type ('e, 's) t = ('e, 's) Expr.ext

  let v ~name ?metadata ~ordered storage ~dec ~enc : _ t =
    { type_ = Type.ext ~name ?metadata storage; storage; ordered; dec; enc }

  let col e name = Expr.make (Expr.Ext_handle (e, name))
  let storage e x = Expr.make (Expr.Storage (e, x))
  let wrap e x = Expr.make (Expr.Wrap (e, x))
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
  (* Every composition builds its query with the verbs that [Talon_next]
     exports, so that its problems are its verbs'. *)

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

  (* Expressions *)

  let cumulative r = Expr.rolling (Window.rows ~before:max_int ~after:0) r
  let index = Expr.(cumulative rows - int 1)
  let arg p v = Expr.(first (if_ (index = over p) v null))
  let fill_forward x = cumulative (Expr.last x)
end
