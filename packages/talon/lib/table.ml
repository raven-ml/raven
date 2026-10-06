(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A table holds at least one batch, which may have no rows. *)
type batch = { length : int; columns : Column.t array }
type t = { schema : Schema.t; rows : int; batches : batch list }

let err fmt = Format.kasprintf invalid_arg fmt
let schema t = t.schema
let rows t = t.rows

let batch schema ~rows columns =
  { schema; rows; batches = [ { length = rows; columns } ] }

let columns t =
  match t.batches with
  | [ b ] -> b.columns
  | bs -> err "Table.columns: a table of %d batches" (List.length bs)

let v ?rows cs =
  let schema =
    Schema.make ~by:"Talon.v" (List.map (fun (n, c) -> (n, Column.type_ c)) cs)
  in
  let rows =
    match (rows, cs) with
    | Some n, _ when n < 0 -> err "Talon.v: rows is %d, below 0" n
    | Some n, _ -> n
    | None, (_, c) :: _ -> Column.length c
    | None, [] -> invalid_arg "Talon.v: no column and no rows"
  in
  let check (n, c) =
    if Column.length c <> rows then
      err "Talon.v: column %a has %d rows, not %d" Type.pp_quoted n
        (Column.length c) rows
  in
  List.iter check cs;
  batch schema ~rows (Array.of_list (List.map snd cs))

let of_batches = function
  | [] -> invalid_arg "Talon.of_batches: no table"
  | t :: _ as ts ->
      let check t' =
        if not (Schema.equal t.schema t'.schema) then
          err "Talon.of_batches: schemas %a and %a differ" Schema.pp t.schema
            Schema.pp t'.schema
      in
      List.iter check ts;
      let rows = List.fold_left (fun n t -> n + t.rows) 0 ts in
      {
        schema = t.schema;
        rows;
        batches = List.concat_map (fun t -> t.batches) ts;
      }

let batches t =
  let one b = { schema = t.schema; rows = b.length; batches = [ b ] } in
  List.filter_map
    (fun b -> if b.length = 0 then None else Some (one b))
    t.batches

(* [parts t i] is column [i] of each of [t]'s batches. *)
let parts t i = List.map (fun b -> b.columns.(i)) t.batches

let index fn t name =
  let rec find i = function
    | [] -> err "Talon.%s: no column %a" fn Type.pp_quoted name
    | (n, _) :: cs -> if String.equal n name then i else find (i + 1) cs
  in
  find 0 (Schema.columns t.schema)

let concat t =
  match t.batches with
  | [ _ ] -> t
  | _ ->
      let k = List.length (Schema.columns t.schema) in
      batch t.schema ~rows:t.rows
        (Array.init k (fun i -> Column.concat (parts t i)))

let canonical t =
  let b = columns t in
  let cs = Array.map Column.canonical b in
  if Array.for_all2 ( == ) b cs then t else batch t.schema ~rows:t.rows cs

let column t name = Column.concat (parts t (index "column" t name))

let take indices t =
  if Nx.ndim indices <> 1 then
    err "Talon.take: indices of shape %a, not 1-D" Nx.pp_shape
      (Nx.shape indices);
  let outside =
    Nx.logical_or (Nx.less_s indices 0L)
      (Nx.greater_equal_s indices (Int64.of_int t.rows))
  in
  (* The first index outside is looked for only when there is one. *)
  if Nx.item [] (Nx.any outside) then begin
    let first = Nx.item [] (Nx.argmax outside) in
    let i = Nx.item [ Int64.to_int first ] indices in
    err "Talon.take: index %Ld of a table of %d rows" i t.rows
  end;
  let rows = Nx.dim 0 indices in
  batch t.schema ~rows (Array.map (Column.gather indices) (columns (concat t)))

let numeric : type a. a Type.t -> bool = function
  | Bool | Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32 | Uint64
  | Float16 | Float32 | Float64 ->
      true
  | _ -> false

let to_tensor (type a b) (dt : (a, b) Nx.dtype) names t : (a, b) Nx.t =
  let column name =
    let c = Column.concat (parts t (index "to_tensor" t name)) in
    let (Any ty) = Column.type_ c in
    if not (numeric ty) then
      err "Talon.to_tensor: %a is %a, neither numeric nor boolean"
        Type.pp_quoted name Type.pp ty;
    if Column.null_count c > 0 then
      err "Talon.to_tensor: %a has a null" Type.pp_quoted name;
    match Column.data c with Fixed (P x) -> Nx.cast dt x | _ -> assert false
  in
  match names with
  | [] -> Nx.zeros dt [| t.rows; 0 |]
  | names -> Nx.stack ~axis:1 (List.map column names)

(* Each column's identity words over both tables' rows, the first table's half
   against the second's: codes compare only within one computation. *)
let equal t0 t1 =
  let n = t0.rows in
  let same i =
    let w = Key.identity [ Column.concat (parts t0 i @ parts t1 i) ] in
    let half k =
      Nx.slice [ Nx.R (k * n, (k + 1) * n); Nx.R (0, Nx.dim 1 w) ] w
    in
    Nx.item [] (Nx.array_equal (half 0) (half 1))
  in
  Schema.equal t0.schema t1.schema
  && n = t1.rows
  && (n = 0
     || List.for_all same
          (List.init (List.length (Schema.columns t0.schema)) Fun.id))
