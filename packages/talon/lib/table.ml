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

let outside_rows i rows =
  Printf.sprintf "Talon.take: index %Ld of a table of %d rows" i rows

let take indices t =
  if Nx.ndim indices <> 1 then
    err "Talon.take: indices of shape %a, not 1-D" Nx.pp_shape
      (Nx.shape indices);
  let outside =
    Nx.logical_or (Nx.less_s indices 0L)
      (Nx.greater_equal_s indices (Int64.of_int t.rows))
  in
  (* Traced indices have no elements to read, so the program checks them
     ({!Nx.check}), and a compiled call raises as it returns. *)
  if Column.traced indices then
    Nx.check Nx.Ptree.tensor (Nx.logical_not outside) indices (fun _ i ->
        Invalid_argument (outside_rows (Nx.item [] i) t.rows))
  else if Nx.item [] (Nx.any outside) then begin
    (* The first index outside is looked for only when there is one. *)
    let first = Nx.item [] (Nx.argmax outside) in
    invalid_arg (outside_rows (Nx.item [ Int64.to_int first ] indices) t.rows)
  end;
  let rows = Nx.dim 0 indices in
  batch t.schema ~rows (Array.map (Column.gather indices) (columns (concat t)))

let numeric : type a. a Type.t -> bool = function
  | Bool | Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32 | Uint64
  | Float16 | Float32 | Float64 ->
      true
  | _ -> false

let to_tensor (type a b) (dt : (a, b) Nx.dtype) names t : (a, b, Nx.host) Nx.t =
  let column name =
    let c = Column.concat (parts t (index "to_tensor" t name)) in
    let (Any ty) = Column.type_ c in
    if not (numeric ty) then
      err "Talon.to_tensor: %a is %a, neither numeric nor boolean"
        Type.pp_quoted name Type.pp ty;
    let null = Format.asprintf "Talon.to_tensor: %a has a null" in
    Column.no_null c (fun _ -> Invalid_argument (null Type.pp_quoted name));
    match Column.data c with Fixed (P x) -> Nx.cast dt x | _ -> assert false
  in
  match names with
  | [] -> Nx.zeros dt [| t.rows; 0 |]
  | names -> Nx.stack ~axis:1 (List.map column names)

(* Dense tensors *)

let cells_key ~by t (name, extent) =
  if extent < 0 then
    err "%s: axis %a has extent %d, below 0" by Type.pp_quoted name extent;
  let c = column t name in
  let (Any ty) = Column.type_ c in
  (match ty with
  | Categorical d when Iarray.length d <> extent ->
      err "%s: axis %a has extent %d, but its dictionary holds %d" by
        Type.pp_quoted name extent (Iarray.length d)
  | _ -> ());
  let x =
    match (ty, Column.data c) with
    | ( ( Categorical _ | Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32
        | Uint64 ),
        Fixed (P x) ) ->
        Nx.cast Nx.int64 x
    | _ ->
        err "%s: axis %a is %a, neither categorical nor integer" by
          Type.pp_quoted name Type.pp ty
  in
  Column.no_null c (fun i ->
      Invalid_argument
        (Format.asprintf "%s: axis %a has a null at row %d" by Type.pp_quoted
           name i));
  let bad =
    Nx.logical_or (Nx.less_s x 0L) (Nx.greater_equal_s x (Int64.of_int extent))
  in
  if Nx.item [] (Nx.any bad) then begin
    let i = Int64.to_int (Nx.item [] (Nx.argmax bad)) in
    err "%s: axis %a holds %Ld at row %d, outside [0, %d)" by Type.pp_quoted
      name (Nx.item [ i ] x) i extent
  end;
  x

(* [cell_count ~by shape] is the elements of [shape]; past [int]'s range no
   tensor could hold them. *)
let cell_count ~by shape =
  Array.fold_left
    (fun acc d ->
      if d <> 0 && acc > max_int / d then err "%s: too many cells" by;
      acc * d)
    1 shape

let to_dense (type a b) (dt : (a, b) Nx.dtype) ~axes value t :
    (a, b, Nx.host) Nx.t * Nx.host Nx.bit_t option =
  let by = "Talon.to_dense" in
  if axes = [] then err "%s: no axis" by;
  Option.iter
    (fun n -> err "%s: axis %a is named twice" by Type.pp_quoted n)
    (match Problem.repeated (List.map fst axes) with
    | [] -> None
    | n :: _ -> Some n);
  let v = column t value in
  let (Any vt) = Column.type_ v in
  if not (numeric vt) then
    err "%s: %a is %a, neither numeric nor boolean" by Type.pp_quoted value
      Type.pp vt;
  let shape = Array.of_list (List.map snd axes) in
  let n = t.rows in
  let flat =
    List.fold_left2
      (fun acc k ext ->
        Nx.add (Nx.mul_s acc (Int64.of_int ext)) (cells_key ~by t k))
      (Nx.zeros Nx.int64 [| n |])
      axes (Array.to_list shape)
  in
  if n >= 2 then begin
    let order = Nx.argsort flat in
    let sorted = Nx.take order flat in
    let same =
      Nx.equal
        (Nx.slice [ Nx.R (0, n - 1) ] sorted)
        (Nx.slice [ Nx.R (1, n) ] sorted)
    in
    if Nx.item [] (Nx.any same) then begin
      let i = Int64.to_int (Nx.item [] (Nx.argmax same)) in
      let row j = Int64.to_int (Nx.item [ j ] order) in
      err "%s: rows %d and %d hold one cell" by (row i) (row (i + 1))
    end
  end;
  let cells = cell_count ~by shape in
  let x =
    match Column.data v with Fixed (P x) -> Nx.cast dt x | _ -> assert false
  in
  let present, x =
    match Column.validity v with
    | None -> (Nx.ones Nx.bool [| n |], x)
    | Some m ->
        let m = Nx.cast Nx.bool m in
        (m, Nx.where m x (Nx.zeros_like x))
  in
  let values = Nx.scatter ~axis:0 flat x (Nx.zeros dt [| cells |]) in
  let held = Nx.scatter ~axis:0 flat present (Nx.zeros Nx.bool [| cells |]) in
  let validity =
    if Nx.item [] (Nx.all held) then None
    else Some (Nx.cast Nx.bit (Nx.reshape shape held))
  in
  (Nx.reshape shape values, validity)

let of_dense ?validity ~axes value x =
  let by = "Talon.of_dense" in
  let shape = Nx.shape x in
  if axes = [] then err "%s: no axis" by;
  if List.length axes <> Array.length shape then
    err "%s: %d axes for a tensor of %d dimensions" by (List.length axes)
      (Array.length shape);
  Option.iter
    (fun m ->
      if Nx.shape m <> shape then
        err "%s: the validity's shape is not the tensor's" by)
    validity;
  let cells = cell_count ~by shape in
  let index = Nx.arange Nx.int64 0 cells 1 in
  (* Row-major, as [to_dense] lays cells: the last axis varies fastest. *)
  let keys, _ =
    List.fold_right2
      (fun name d (keys, stride) ->
        let k =
          if cells = 0 then Nx.zeros Nx.int64 [| 0 |]
          else
            Nx.mod_s (Nx.div_s index (Int64.of_int stride)) (Int64.of_int d)
        in
        ((name, Column.of_tensor k) :: keys, stride * d))
      axes (Array.to_list shape) ([], 1)
  in
  let validity = Option.map Nx.flatten validity in
  v (keys @ [ (value, Column.of_tensor ?validity (Nx.flatten x)) ])

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
