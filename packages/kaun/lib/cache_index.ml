(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type tokens =
  | Whole of Nx.int32_t
  | Tabled of {
      row : Nx.int32_t option;
      pos : Nx.int32_t;
      table : Nx.int32_t;
      blocks : (int * Nx.int32_t) list;
    }

(* [every] is how many positions a column holds: [1], or an [m] of the tokens'
   [blocks]. [columns] is the selection, the columns each token reads: [batch;
   seq; k]. *)
type t = {
  tokens : tokens;
  every : int;
  window : int option;
  columns : Nx.int32_t option;
}

let invalid fmt = Printf.ksprintf invalid_arg fmt

(* Lanes are padded on the left, so the last column is every lane's last
   token. *)
let left_padded ~seq lens =
  let batch = Array.length lens in
  Nx.create Nx.int32 [| batch; seq |]
    (Array.init (batch * seq) (fun t ->
         let b = t / seq and i = t mod seq in
         Int32.of_int (max (-1) (i - (seq - lens.(b))))))

let whole ?lens ~batch ~seq () =
  if batch <= 0 || seq <= 0 then
    invalid
      "Cache_index.whole: batch and seq must be positive, got batch=%d seq=%d"
      batch seq;
  let lens = Option.value lens ~default:(Array.make batch seq) in
  if Array.length lens <> batch then
    invalid "Cache_index.whole: %d lengths for %d lanes" (Array.length lens)
      batch;
  Array.iter
    (fun n ->
      if n < 0 || n > seq then
        invalid
          "Cache_index.whole: a lane of %d tokens does not fit %d positions" n
          seq)
    lens;
  {
    tokens = Whole (left_padded ~seq lens);
    every = 1;
    window = None;
    columns = None;
  }

let rec strides ~name = function
  | [] -> ()
  | m :: rest ->
      if m < 2 then
        invalid "Cache_index.%s: a block holds at least 2 positions, got %d"
          name m;
      if List.mem m rest then
        invalid "Cache_index.%s: two tables for blocks of %d positions" name m;
      strides ~name rest

let rows ?(every = []) ~context lens =
  let batch = Array.length lens in
  if batch = 0 then invalid_arg "Cache_index.rows: no lanes";
  if context <= 0 then
    invalid "Cache_index.rows: context must be positive, got %d" context;
  Array.iter
    (fun n ->
      if n < 0 || n > context then
        invalid
          "Cache_index.rows: a lane of %d tokens does not fit a context of %d" n
          context)
    lens;
  strides ~name:"rows" every;
  let seq = Array.fold_left max 1 lens in
  let runs columns =
    Nx.create Nx.int32 [| batch; columns |]
      (Array.init (batch * columns) Int32.of_int)
  in
  let blocks = List.map (fun m -> (m, runs ((context + m - 1) / m))) every in
  {
    tokens =
      Tabled
        {
          row = None;
          pos = left_padded ~seq lens;
          table = runs context;
          blocks;
        };
    every = 1;
    window = None;
    columns = None;
  }

let make ?row ?(every = []) ~pos ~table () =
  (match (Nx.shape pos, Nx.shape table) with
  | [| b; s |], [| r; c |] when b > 0 && s > 0 && r > 0 && c > 0 -> (
      match row with
      | None when r = b -> ()
      | Some row when Nx.shape row = [| b |] -> ()
      | None ->
          invalid "Cache_index.make: a table of %d rows for %d lanes needs ~row"
            r b
      | Some _ -> invalid "Cache_index.make: row must have shape [%d]" b)
  | _ ->
      invalid_arg
        "Cache_index.make: pos must have shape [batch; seq] and table [rows; \
         context], neither of them empty");
  strides ~name:"make" (List.map fst every);
  List.iter
    (fun (m, blocks) ->
      match Nx.shape blocks with
      | [| r; c |] when r = Nx.dim 0 table && c > 0 -> ()
      | _ ->
          invalid
            "Cache_index.make: the table of blocks of %d positions must have \
             shape [%d; context], context positive"
            m (Nx.dim 0 table))
    every;
  {
    tokens = Tabled { row; pos; table; blocks = every };
    every = 1;
    window = None;
    columns = None;
  }

let window w index =
  if w <= 0 then invalid "Cache_index.window: window must be positive, got %d" w;
  { index with window = Some w }

let raw index = match index.tokens with Whole pos | Tabled { pos; _ } -> pos
let batch index = Nx.dim 0 (raw index)
let seq index = Nx.dim 1 (raw index)

let select columns index =
  (match Nx.shape columns with
  | [| b; s; k |] when b = batch index && s = seq index && k > 0 -> ()
  | _ ->
      invalid "Cache_index.select: columns must have shape [%d; %d; k], k > 0"
        (batch index) (seq index));
  { index with columns = Some columns }

let every m index =
  if m <= 0 then invalid "Cache_index.every: m must be positive, got %d" m;
  if m = index.every then index
  else begin
    if index.every <> 1 then
      invalid
        "Cache_index.every: the index already reads blocks of %d positions"
        index.every;
    if Option.is_some index.columns then
      invalid_arg "Cache_index.every: the index selects columns";
    (match index.tokens with
    | Tabled { blocks; _ } when not (List.mem_assoc m blocks) ->
        invalid
          "Cache_index.every: the index has no table for blocks of %d positions"
          m
    | _ -> ());
    { index with every = m }
  end

(* The table the index reads: the positions', or its stride's. *)
let table_of index ~table ~blocks =
  if index.every = 1 then table else List.assoc index.every blocks

let context index =
  match index.tokens with
  | Whole pos -> (Nx.dim 1 pos + index.every - 1) / index.every
  | Tabled { table; blocks; _ } -> Nx.dim 1 (table_of index ~table ~blocks)

(* The position each column stands at, its block's last: a block is stored by
   its last token and seen from it on. At stride 1 a column is its position. *)
let stands ~every columns =
  if every = 1 then columns
  else
    Nx.add_s (Nx.mul_s columns (Int32.of_int every)) (Int32.of_int (every - 1))

let inside ~below t =
  Nx.logical_and (Nx.greater_equal_s t 0l) (Nx.less_s t (Int32.of_int below))

(* Positions, with every token of a lane whose row is outside the table as
   padding. *)
let pos index =
  match index.tokens with
  | Whole pos | Tabled { row = None; pos; _ } -> pos
  | Tabled { row = Some row; pos; table; _ } ->
      let named = inside ~below:(Nx.dim 0 table) row in
      Nx.where
        (Nx.reshape [| Nx.dim 0 pos; 1 |] named)
        pos (Nx.full_like pos (-1l))

(* The positions' table bounds the positions, whatever the stride. *)
let positions index =
  let last =
    match index.tokens with
    | Whole pos -> Nx.dim 1 pos - 1
    | Tabled { table; _ } -> Nx.dim 1 table - 1
  in
  Nx.clamp ~min:0l ~max:(Int32.of_int last) (raw index)

let advance index =
  match index.tokens with
  | Whole _ -> invalid_arg "Cache_index.advance: a whole index keeps nothing"
  | Tabled { row; table; blocks; _ } ->
      (* Any negative position is padding: a lane of it advances to 0. *)
      let last =
        Nx.maximum_s (Nx.max ~axes:[ 1 ] ~keepdims:true (pos index)) (-1l)
      in
      {
        index with
        tokens = Tabled { row; pos = Nx.add_s last 1l; table; blocks };
        columns = None;
      }

(* The table of each lane. *)
let lanes ~row table =
  match row with
  | None -> table
  | Some row ->
      let last = Int32.of_int (Nx.dim 0 table - 1) in
      Nx.take ~axis:0 ~indices:(Nx.clamp ~min:0l ~max:last row) table

(* Masks *)

(* [keys] broadcasts against [batch; seq; 1]: the position each key holds. *)
let sees ?window ~keys pos =
  let query = Nx.reshape [| Nx.dim 0 pos; Nx.dim 1 pos; 1 |] pos in
  let causal = Nx.less_equal keys query in
  match window with
  | None -> causal
  | Some w ->
      Nx.logical_and causal (Nx.greater keys (Nx.sub_s query (Int32.of_int w)))

let column ~context =
  Nx.reshape [| 1; context |] (Nx.arange Nx.int32 0 context 1)

(* The position each chosen column stands at, [-1] outside the context. On a
   whole index at stride 1 a column is a token of the lane, which stands at its
   token's position. *)
let chosen index columns =
  let context = context index in
  let at = Nx.clamp ~min:0l ~max:(Int32.of_int (context - 1)) columns in
  let at =
    match index.tokens with
    | Tabled _ -> stands ~every:index.every at
    | Whole _ when index.every > 1 -> stands ~every:index.every at
    | Whole pos ->
        let b = Nx.dim 0 at and s = Nx.dim 1 at and k = Nx.dim 2 at in
        Nx.reshape [| b; s; k |]
          (Nx.take_along_axis ~axis:1
             ~indices:(Nx.reshape [| b; s * k |] at)
             pos)
  in
  Nx.where (inside ~below:context columns) at (Nx.full_like at (-1l))

(* Which of its chosen columns each token sees, [batch; seq; k]. *)
let sees_chosen index columns =
  let keys = chosen index columns in
  Nx.logical_and
    (Nx.greater_equal_s keys 0l)
    (sees ?window:index.window ~keys (pos index))

let mask index =
  let pos = pos index and window = index.window in
  match (index.columns, index.tokens) with
  | Some columns, _ -> sees_chosen index columns
  | None, Whole _ when index.every = 1 ->
      let keys = Nx.reshape [| Nx.dim 0 pos; 1; Nx.dim 1 pos |] pos in
      Nx.logical_and (Nx.greater_equal_s keys 0l) (sees ?window ~keys pos)
  | None, _ ->
      let context = context index in
      let keys = stands ~every:index.every (column ~context) in
      sees ?window ~keys:(Nx.reshape [| 1; 1; context |] keys) pos

(* Pools *)

let pool ~slots dtype shape =
  if slots < 0 then
    invalid "Cache_index.pool: slots must not be negative, got %d" slots;
  Nx.zeros dtype (Array.append [| slots + 1 |] shape)

let tail pool =
  let shape = Nx.shape pool in
  if Array.length shape = 0 then
    invalid_arg "Cache_index.extend: a pool has a slot axis";
  Array.sub shape 1 (Array.length shape - 1)

let ones tail = Array.map (fun _ -> 1) tail

(* The column that stands at each token's position, or [-1]: at stride [m] only
   a block's last position has one. *)
let own ~every pos =
  if every = 1 then pos
  else
    let m = Int32.of_int every in
    let closes =
      Nx.logical_and
        (Nx.greater_equal_s pos 0l)
        (Nx.equal_s (Nx.mod_s (Nx.add_s pos 1l) m) 0l)
    in
    Nx.where closes (Nx.div_s pos m) (Nx.full_like pos (-1l))

(* [pool] with [values] at the slot the table names at each token's column [at].
   A token that has none writes the scratch row. *)
let write ~at ~table values pool =
  let batch = Nx.dim 0 at and seq = Nx.dim 1 at in
  let tail = tail pool in
  let slots = Nx.dim 0 pool - 1 and context = Nx.dim 1 table in
  let tokens = batch * seq in
  let slot =
    Nx.take_along_axis ~axis:1
      ~indices:(Nx.clamp ~min:0l ~max:(Int32.of_int (context - 1)) at)
      table
  in
  let addressed =
    Nx.logical_and (inside ~below:context at) (inside ~below:slots slot)
  in
  let target =
    Nx.where addressed slot (Nx.full_like slot (Int32.of_int slots))
  in
  let indices =
    Nx.broadcast_to
      (Array.append [| tokens |] tail)
      (Nx.reshape (Array.append [| tokens |] (ones tail)) target)
  in
  let values =
    Nx.reshape (Array.append [| tokens |] tail) (Nx.contiguous values)
  in
  Nx.scatter ~unique_indices:true ~axis:0 ~indices ~values pool

(* [pool] at each lane's columns, zero at the columns no token of the lane sees
   and at unallocated ones. *)
let read ?window ~every ~pos ~table pool =
  let tail = tail pool in
  let slots = Nx.dim 0 pool - 1 in
  let batch = Nx.dim 0 table and context = Nx.dim 1 table in
  let column = stands ~every (column ~context) in
  let allocated = inside ~below:slots table in
  let seen =
    let last = Nx.max ~axes:[ 1 ] ~keepdims:true pos in
    let upto = Nx.less_equal column last in
    match window with
    | None -> upto
    | Some w ->
        (* The least position of the lane's tokens, or none: padding is -1. *)
        let first =
          Nx.min ~axes:[ 1 ] ~keepdims:true
            (Nx.where
               (Nx.greater_equal_s pos 0l)
               pos
               (Nx.full_like pos Int32.max_int))
        in
        Nx.logical_and upto
          (Nx.greater column (Nx.sub_s first (Int32.of_int w)))
  in
  let slot =
    Nx.where allocated table (Nx.full_like table (Int32.of_int slots))
  in
  let live =
    Nx.reshape
      (Array.append [| batch * context |] (ones tail))
      (Nx.logical_and allocated seen)
  in
  let win =
    Nx.take ~axis:0 ~indices:(Nx.reshape [| batch * context |] slot) pool
  in
  Nx.reshape
    (Array.append [| batch; context |] tail)
    (Nx.where live win (Nx.zeros (Nx.dtype pool) [| 1 |]))

(* The rows a selection reads, [batch; seq; k] then [tail], zero where a token
   does not see its column. [fetch flat] is the rows at the columns [flat],
   [batch; seq * k] and clamped to the context, as [batch; seq * k] then [tail],
   with which of them are allocated when some may not be. *)
let read_chosen index columns ~tail fetch =
  let b = Nx.dim 0 columns and s = Nx.dim 1 columns and k = Nx.dim 2 columns in
  let last = Int32.of_int (context index - 1) in
  let flat = Nx.reshape [| b; s * k |] (Nx.clamp ~min:0l ~max:last columns) in
  let rows, allocated = fetch flat in
  let live = Nx.reshape [| b; s * k |] (sees_chosen index columns) in
  let live = Option.fold ~none:live ~some:(Nx.logical_and live) allocated in
  Nx.reshape
    (Array.append [| b; s; k |] tail)
    (Nx.where
       (Nx.reshape (Array.append [| b; s * k |] (ones tail)) live)
       rows
       (Nx.zeros (Nx.dtype rows) [| 1 |]))

(* The rows of [values], [batch; seq] then [tail], at the tokens [token],
   [batch; n]. *)
let rows_at ~tail values token =
  let indices =
    Nx.broadcast_to
      (Array.append (Nx.shape token) tail)
      (Nx.reshape (Array.append (Nx.shape token) (ones tail)) token)
  in
  Nx.take_along_axis ~axis:1 ~indices values

(* On a whole index, the token of each lane at the position the columns [flat],
   [batch; n], stand at, clamped to the lane, and whether the lane holds it.
   Lanes are padded on the left, so position [p] is token [p + pad]. *)
let closing ~every ~pos flat =
  let seq = Nx.dim 1 pos in
  let last = Nx.slice [ A; R (seq - 1, seq) ] pos in
  let token =
    Nx.add (stands ~every flat) (Nx.rsub_s (Int32.of_int (seq - 1)) last)
  in
  (Nx.clamp ~min:0l ~max:(Int32.of_int (seq - 1)) token, inside ~below:seq token)

(* The rows of [pool] at the slots [table] names at the columns [flat], [batch;
   n], as [batch; n] then [tail], and which of them are allocated. *)
let from_pool ~tail ~table pool flat =
  let slots = Nx.dim 0 pool - 1 in
  let slot = Nx.take_along_axis ~axis:1 ~indices:flat table in
  let allocated = inside ~below:slots slot in
  let slot = Nx.where allocated slot (Nx.full_like slot (Int32.of_int slots)) in
  let n = Nx.dim 0 flat * Nx.dim 1 flat in
  ( Nx.reshape
      (Array.append (Nx.shape flat) tail)
      (Nx.take ~axis:0 ~indices:(Nx.reshape [| n |] slot) pool),
    Some allocated )

let extend index values pool =
  let tail = tail pool in
  if Nx.shape values <> Array.append [| batch index; seq index |] tail then
    invalid
      "Cache_index.extend: values must have shape [%d; %d] then the pool's"
      (batch index) (seq index);
  let every = index.every in
  match (index.tokens, index.columns) with
  | Whole _, None when every = 1 -> (values, pool)
  | Whole pos, None ->
      let context = context index in
      let flat = Nx.broadcast_to [| batch index; context |] (column ~context) in
      let token, held = closing ~every ~pos flat in
      let held = Nx.reshape (Array.append (Nx.shape held) (ones tail)) held in
      ( Nx.where held
          (rows_at ~tail values token)
          (Nx.zeros (Nx.dtype values) [| 1 |]),
        pool )
  | Whole pos, Some columns ->
      (* A column a token sees is one of its lane's tokens, or a block one of
         them closes. *)
      let fetch flat =
        let token =
          if every = 1 then flat else fst (closing ~every ~pos flat)
        in
        (rows_at ~tail values token, None)
      in
      (read_chosen index columns ~tail fetch, pool)
  | Tabled { row; table; blocks; _ }, columns -> (
      let pos = pos index in
      let table = lanes ~row (table_of index ~table ~blocks) in
      let pool = write ~at:(own ~every pos) ~table values pool in
      match columns with
      | None -> (read ?window:index.window ~every ~pos ~table pool, pool)
      | Some columns ->
          (read_chosen index columns ~tail (from_pool ~tail ~table pool), pool))

(* Structure *)

module Walked = struct
  type nonrec _ t = t

  let block c (m, blocks) =
    let open Nx.Ptree.Walk in
    let m = int c m in
    (m, tensor c blocks)

  let tokens c tokens =
    let open Nx.Ptree.Walk in
    match tokens with
    | Whole pos ->
        case c "whole";
        Whole (field c "pos" tensor pos)
    | Tabled { row; pos; table; blocks } ->
        case c "tabled";
        let row = field c "row" (option tensor) row in
        let pos = field c "pos" tensor pos in
        let table = field c "table" tensor table in
        let blocks = field c "blocks" (list block) blocks in
        Tabled { row; pos; table; blocks }

  let walk c x =
    let open Nx.Ptree.Walk in
    let tokens = field c "tokens" tokens x.tokens in
    let every = field c "every" int x.every in
    let window = field c "window" (option int) x.window in
    let columns = field c "columns" (option tensor) x.columns in
    { tokens; every; window; columns }
end

let ptree : t Nx.Ptree.t = Nx.Ptree.instantiate (module Walked)
