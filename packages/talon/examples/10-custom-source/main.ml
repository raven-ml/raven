(* A source of your own, read in parts and batches, and a query folded over its
   batches without collecting them. *)

open Talon

(* [counter n] yields the integers [0] to [n - 1] in parts of 1,000 rows. It
   applies comparisons of [i] with an integer exactly, and claims its order. *)
(* [int64 v] is the integer that a predicate compares with, if it is one. *)
let int64 : Source.Pred.value -> int option = function
  | Value (Int64, v) -> Some v
  | Value _ -> None

(* [passes p i] is [true] iff the row [i] passes [p], a comparison of [i]. *)
let passes (p : Source.Pred.t) i =
  match p with
  | Cmp (_, op, v) -> (
      let c = compare i (Option.get (int64 v)) in
      match op with
      | `Eq -> c = 0
      | `Ne -> c <> 0
      | `Lt -> c < 0
      | `Le -> c <= 0
      | `Gt -> c > 0
      | `Ge -> c >= 0)
  | _ -> true

(* [counter n] yields the integers [0] to [n - 1] in parts of 1,000 rows. It
   applies comparisons of [i] with an integer exactly, and claims its order. *)
let counter n =
  let schema = Schema.v [ ("i", Type.Any Type.int64) ] in
  let pushdown : Source.Pred.t -> Source.answer = function
    | Cmp ("i", _, v) when Option.is_some (int64 v) -> Exact
    | _ -> Unsupported
  in
  let keep (req : Source.request) i =
    List.for_all (fun p -> passes p i) req.filters
  in
  let batch (req : Source.request) lo hi =
    let rows = List.filter (keep req) (List.init (hi - lo) (( + ) lo)) in
    if req.columns = [] then Talon.v ~rows:(List.length rows) []
    else Talon.v [ ("i", Column.v Type.int64 (Array.of_list rows)) ]
  in
  let part req lo hi =
    let open_ () =
      let read = ref false in
      let next () =
        if !read then Ok None
        else (
          read := true;
          Ok (Some (batch req lo hi)))
      in
      Ok { Source.next; close = ignore }
    in
    let rows = if req.filters = [] then Some (hi - lo) else None in
    { Source.rows; open_ }
  in
  let parts req =
    Ok
      (List.init
         ((n + 999) / 1000)
         (fun k -> part req (k * 1000) (min n ((k + 1) * 1000))))
  in
  Source.v ~name:"counter" ~schema ~rows:n
    ~sorted:[ Order.asc "i" ]
    ~pushdown parts

let () =
  let i = Col.int "i" in
  let q =
    Query.(
      of_source (counter 10_000)
      |> filter Expr.(i >= int 2500 && i mod int 7 = int 0)
      |> derive Expr.[ "square" := i * i ])
  in
  (* The comparison goes to the source; the modulo stays in the plan. *)
  Format.printf "%a@.@." Query.pp (Query.optimize q);

  (* Fold over the batches, keeping only a count and a sum. *)
  let rows, total =
    Error.get_ok
      (Query.fold q ~init:(0, 0) (fun (rows, total) b ->
           let squares = Column.values Kind.int (Talon.column b "square") in
           (rows + Talon.rows b, Array.fold_left ( + ) total squares)))
  in
  Format.printf "%d rows, squares sum to %d@." rows total
