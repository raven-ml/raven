(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type stream = { next : unit -> Table.t option; close : unit -> unit }

exception Failed of Error.t

(* [line pp v] is [v] formatted on one line, as errors name steps. *)
let line pp v =
  let b = Buffer.create 80 in
  let ppf = Format.formatter_of_buffer b in
  Format.pp_set_margin ppf 1_000_000;
  pp ppf v;
  Format.pp_print_flush ppf ();
  Buffer.contents b

(* [failed step f] raises the failure [f] that [step] found, [f.row] counted
   over the step's input. *)
let failed step (f : Eval.failure) =
  match f.cause with
  | Data e -> raise (Failed (Error.in_step step ~row:f.row e))
  | Raised (exn, bt) -> Printexc.raise_with_backtrace exn bt

let sub b ~offset ~length =
  if offset = 0 && length = Table.rows b then b
  else
    Table.batch (Table.schema b) ~rows:length
      (Array.map (Column.sub ~offset ~length) (Table.columns b))

(* [cut q b f cs] is the batch of [q]'s columns [cs], computed over [b]: its
   rows before the failure [f], if any. *)
let cut q b f cs =
  let rows, cs =
    match f with
    | None -> (Table.rows b, cs)
    | Some (f : Eval.failure) ->
        (f.row, List.map (Column.sub ~offset:0 ~length:f.row) cs)
  in
  Table.batch (Query.schema q) ~rows (Array.of_list cs)

(* Steps *)

let of_table t =
  let batches = ref (Table.batches t) in
  let next () =
    match !batches with
    | [] -> None
    | b :: bs ->
        batches := bs;
        Some b
  in
  { next; close = ignore }

(* [of_source q source request] is the rows of [source]'s parts for [request],
   [q]'s step: each part opened when the one before ends, its reader closed at
   its end, at an error, or when the stream closes. *)
let of_source q (source : Source.t) request =
  let ok = function Ok x -> x | Error e -> raise (Failed e) in
  let parts = ref None and reader = ref None and held = ref [] in
  let close () =
    Option.iter
      (fun (r : Source.reader) ->
        reader := None;
        r.close ())
      !reader
  in
  let check t =
    if not (Schema.equal (Table.schema t) (Query.schema q)) then
      Format.kasprintf invalid_arg
        "Query.run: %s yields a batch of the columns %a, not %a" source.name
        Schema.pp (Table.schema t) Schema.pp (Query.schema q);
    Table.batches t
  in
  let rec next () =
    match (!held, !reader, !parts) with
    | b :: bs, _, _ ->
        held := bs;
        Some b
    | [], Some r, _ -> (
        match r.next () with
        | Ok (Some t) ->
            held := check t;
            next ()
        | Ok None ->
            close ();
            next ()
        | Error e ->
            close ();
            raise (Failed e))
    | [], None, None ->
        parts := Some (ok (source.parts request));
        next ()
    | [], None, Some ((p : Source.part) :: ps) ->
        parts := Some ps;
        if p.rows <> Some 0 then reader := Some (ok (p.open_ ()));
        next ()
    | [], None, Some [] -> None
  in
  { next; close }

(* [streaming q s step] is [step] applied to each batch of [s], the input of
   [q]'s step. [step b] is [b]'s output and the failure, if any, at whose row
   the output stops; the next pull raises the failure. *)
let streaming q s step =
  let rows = ref 0 and failure = ref None in
  let next () =
    match !failure with
    | Some f -> failed (line Query.pp_step q) f
    | None -> (
        match s.next () with
        | None -> None
        | Some b ->
            let out, f = step b in
            Option.iter
              (fun (f : Eval.failure) ->
                failure := Some { f with row = !rows + f.row })
              f;
            rows := !rows + Table.rows b;
            Some out)
  in
  { next; close = s.close }

let empty s =
  let column (_, Type.Any ty) =
    Result.get_ok (Column.encode ty 0 (fun _ -> None))
  in
  Table.batch s ~rows:0 (Array.of_list (List.map column (Schema.columns s)))

(* [gather input s] is all of [s]'s rows, the step [input]'s, as one batch. *)
let gather input s =
  let rec pull bs =
    match s.next () with Some b -> pull (b :: bs) | None -> List.rev bs
  in
  match pull [] with
  | [] -> empty (Query.schema input)
  | bs -> Table.concat (Table.of_batches bs)

(* [once close f] is the stream of [f ()]'s batch, then its end. *)
let once close f =
  let ran = ref false in
  let next () =
    if !ran then None
    else begin
      ran := true;
      f ()
    end
  in
  { next; close }

(* [blocking q input s step] is [step] applied once, to all of [s]'s rows, the
   input of [q]'s step. It emits nothing when [step] fails. *)
let blocking q input s step =
  once s.close (fun () ->
      match step (gather input s) with
      | out, None -> Some out
      | _, Some f -> failed (line Query.pp_step q) f)

let select q outputs input =
  let eval = Eval.outputs (Query.schema input) outputs in
  fun b ->
    let cs, f = eval (Eval.frame b) in
    (cut q b f cs, f)

(* Where [derive] takes a column of its result from. *)
type source = Output of int | Input of int

let derive q outputs input =
  let eval = Eval.outputs (Query.schema input) outputs in
  let index n ns = List.find_index (String.equal n) ns in
  let outs = List.map fst outputs and ins = Schema.names (Query.schema input) in
  let source n =
    match index n outs with
    | Some i -> Output i
    | None -> Input (Option.get (index n ins))
  in
  let sources = List.map source (Schema.names (Query.schema q)) in
  fun b ->
    let cs, f = eval (Eval.frame b) in
    let cs = Array.of_list cs and ins = Table.columns b in
    let column = function Output i -> cs.(i) | Input j -> ins.(j) in
    (cut q b f (List.map column sources), f)

(* [unnest q outputs keep_empty input] repeats [q]'s input columns once per
   position of its outputs. *)
let unnest q outputs keep_empty input =
  let eval = Eval.unnest (Query.schema input) outputs ~keep_empty in
  let index n ns = List.find_index (String.equal n) ns in
  let outs = List.map fst outputs and ins = Schema.names (Query.schema input) in
  let source n =
    match index n outs with
    | Some i -> Output i
    | None -> Input (Option.get (index n ins))
  in
  let sources = List.map source (Schema.names (Query.schema q)) in
  fun b ->
    let outer, cs, f = eval (Eval.frame b) in
    let cs = Array.of_list cs and ins = Table.columns b in
    let column = function
      | Output i -> cs.(i)
      | Input j -> Column.gather outer ins.(j)
    in
    ( Table.batch (Query.schema q) ~rows:(Nx.dim 0 outer)
        (Array.of_list (List.map column sources)),
      f )

let filter q predicate input =
  let eval = Eval.predicate (Query.schema input) predicate in
  fun b ->
    let n = Table.rows b in
    let keep, f = eval (Eval.frame b) in
    let keep =
      match f with None -> keep | Some f -> Nx.slice [ Nx.R (0, f.row) ] keep
    in
    let idx = Nx.positions keep in
    let rows = Nx.dim 0 idx in
    let out =
      if rows = n then b
      else
        Table.batch (Query.schema q) ~rows
          (Array.map (Column.take idx) (Table.columns b))
    in
    (out, f)

(* [aggregate q by outputs input] computes [q]'s groups of the input's rows,
   keyed by the columns [by]. *)
let aggregate q by outputs input =
  let names = Schema.names (Query.schema input) in
  let eval = Eval.outputs (Query.schema input) outputs in
  fun b ->
    let cs = Table.columns b in
    let keys =
      List.map
        (fun k -> cs.(Option.get (List.find_index (String.equal k) names)))
        by
    in
    let s =
      match keys with [] -> Reduce.one (Table.rows b) | ks -> Reduce.group ks
    in
    let outs, f = eval (Eval.groups b s) in
    let keys = List.map (Column.gather (Reduce.first s)) keys in
    ( Table.batch (Query.schema q) ~rows:(Reduce.count s)
        (Array.of_list (keys @ outs)),
      f )

(* A streamed aggregate holds [O (groups + batch)] rows: each batch's operands
   as rows of the state's schema, and when the rows held reach the groups so
   far, they reduce together with the state rows, state first ({!Partial}), by
   the ordinary aggregate [merge]. A failure is held to the end, as a later
   batch can show a group's failure at an earlier row: the earliest row wins,
   then the earliest node in evaluation order, a reduction's operand before
   the reduction. A traced batch, whose rows a compaction could not take back,
   stays held, and its rows reduce once with the state at the end; [var] and
   [std] raise, as their fold reads its values. {!partial} blocks when one of
   the first two batches is traced, so that only a later one meets this. The
   state and the held rows are released before they reduce, so that one copy
   is live. *)
let rec traced c =
  Option.fold ~none:false ~some:Column.traced (Column.validity c)
  ||
  match Column.data c with
  | Fixed (P x) -> Column.traced x
  | Fields cs -> List.exists traced cs
  | _ -> false

let streamed q by (p : Partial.t) input s =
  let frames = Query.of_table (empty p.schema) in
  let state_q =
    Query.make (Aggregate { by; outputs = p.merge; input = frames })
  in
  let merge = aggregate state_q by p.merge frames in
  let finish = select q p.finish state_q in
  let project = Eval.outputs (Query.schema input) p.held in
  let keys = List.length by in
  let state = ref None and held = ref [] and held_rows = ref 0 in
  let rows = ref 0 and blocked = ref false and failure = ref None in
  let note row rank =
    Option.iter (fun (f : Eval.failure) ->
        let key = (row f.row, rank f.output) in
        match !failure with
        | Some (k, _) when k <= key -> ()
        | _ -> failure := Some (key, { f with row = fst key }))
  in
  let compact () =
    let tables = Option.to_list !state @ List.rev !held in
    let t = Table.concat (Table.of_batches tables) in
    state := None;
    held := [];
    held_rows := 0;
    let out, f = merge t in
    let first = Table.column t Partial.row_field in
    note
      (fun r -> Int64.to_int (Nx.item [ r ] (Column.to_tensor Nx.int64 first)))
      (fun o -> (2 * (o + keys)) + 1)
      f;
    state := Some out
  in
  let push b =
    let n = Table.rows b and base = !rows in
    let cs, f = project (Eval.frame b) in
    note (( + ) base) (fun o -> 2 * o) f;
    let row = Column.of_tensor (Nx.arange Nx.int64 base (base + n) 1) in
    let columns = Array.of_list (cs @ [ row ]) in
    held := Table.batch p.schema ~rows:n columns :: !held;
    held_rows := !held_rows + n;
    rows := base + n;
    if List.exists traced cs && p.folds then
      invalid_arg "Query.aggregate: var and std cannot fold a traced batch";
    blocked := !blocked || List.exists traced cs;
    let groups = match !state with Some s -> Table.rows s | None -> 0 in
    let due = !held_rows > 0 && !held_rows >= groups in
    if (not !blocked) && due then compact ()
  in
  let rec pull () =
    match s.next () with
    | Some b ->
        push b;
        pull ()
    | None ->
        if not (List.is_empty !held) then compact ();
        Option.iter (fun (_, f) -> failed (line Query.pp_step q) f) !failure;
        Some (fst (finish (Option.get !state)))
  in
  pull ()

(* [partial q by outputs input s] is the aggregate [q] of [s]. One batch, or
   outputs that {!Partial} does not stream, block as ever, and only a second
   batch pays for the lowering. The batches pulled to tell are replayed. *)
let partial q by outputs input s =
  once s.close (fun () ->
      let a = s.next () in
      let b = if Option.is_none a then None else s.next () in
      let peeked = ref (List.filter_map Fun.id [ a; b ]) in
      let next () =
        match !peeked with
        | x :: rest ->
            peeked := rest;
            Some x
        | [] -> s.next ()
      in
      let rest = { s with next } in
      (* What a batch's operands compute decides, as a table captured under
         [Rune.jit] computes traced values from untraced columns. *)
      let traced_held (p : Partial.t) =
        let project = Eval.outputs (Query.schema input) p.held in
        let held b = List.exists traced (fst (project (Eval.frame b))) in
        List.exists held !peeked
      in
      let lowered =
        if Option.is_none b then None
        else
          match Partial.v ~by outputs (Query.schema input) with
          | Some p when p.folds && traced_held p -> None
          | lowered -> lowered
      in
      match lowered with
      | Some p -> streamed q by p input rest
      | None -> (blocking q input rest (aggregate q by outputs input)).next ())

(* [head ~offset ~length s] is [s]'s rows [offset] to [offset + length - 1]; it
   closes [s] past them, and pulls nothing when it keeps no row. *)
let head ~offset ~length s =
  let stop =
    if length = 0 then 0
    else if length > max_int - offset then max_int
    else offset + length
  in
  let seen = ref 0 and closed = ref false in
  let close () =
    if not !closed then begin
      closed := true;
      s.close ()
    end
  in
  let rec next () =
    if !seen >= stop then begin
      close ();
      None
    end
    else
      match s.next () with
      | None -> None
      | Some b ->
          let first = !seen in
          seen := first + Table.rows b;
          let lo = Int.max offset first - first
          and hi = Int.min stop !seen - first in
          if lo >= hi then next ()
          else Some (sub b ~offset:lo ~length:(hi - lo))
  in
  { next; close }

(* [tail ~offset ~length s] is [s]'s rows from [-offset] before its end, at most
   [length]. It holds the batches of [s]'s last [-offset] rows. *)
let tail ~offset ~length s =
  let k = if offset = min_int then max_int else -offset in
  let held = Queue.create () in
  let rec pull h total =
    match s.next () with
    | None -> (h, total)
    | Some b ->
        let n = Table.rows b in
        Queue.push b held;
        let h = ref (h + n) in
        while !h - Table.rows (Queue.peek held) >= k do
          h := !h - Table.rows (Queue.pop held)
        done;
        pull !h (total + n)
  in
  (* The rows of the held batches, from [pos], and the kept ones, [first] to
     [stop - 1], counted over [s]. *)
  let window =
    lazy
      (let h, total = pull 0 0 in
       let start = total - k in
       let stop = if length >= total - start then total else start + length in
       (ref (total - h), Int.max start 0, stop))
  in
  let rec next () =
    let pos, first, stop = Lazy.force window in
    match Queue.take_opt held with
    | None -> None
    | Some b ->
        let p = !pos in
        pos := p + Table.rows b;
        let lo = Int.max first p - p and hi = Int.min stop !pos - p in
        if lo >= hi then next () else Some (sub b ~offset:lo ~length:(hi - lo))
  in
  { next; close = s.close }

(* [append q a r rest] is [a]'s batches, then [r]'s, whose columns, those of the
   schema [rest], are put in [q]'s order. *)
(* [conform q s] maps a batch of the columns [s] to one of [q]'s: its columns in
   [q]'s order, each widened to [q]'s type. *)
let conform q s =
  let schema = Query.schema q and names = Schema.names s in
  let column (n, t) =
    let i = Option.get (List.find_index (String.equal n) names) in
    (i, Eval.widen (Option.get (Schema.find s n)) t)
  in
  let columns = List.map column (Schema.columns schema) in
  fun b ->
    let cs = Table.columns b in
    Table.batch schema ~rows:(Table.rows b)
      (Array.of_list (List.map (fun (i, widen) -> widen cs.(i)) columns))

let append q a r ~input ~rest =
  let left = conform q input and right = conform q rest in
  let first = ref true in
  let rec next () =
    if not !first then Option.map right (r.next ())
    else
      match a.next () with
      | Some b -> Some (left b)
      | None ->
          first := false;
          next ()
  in
  {
    next;
    close =
      (fun () ->
        a.close ();
        r.close ());
  }

(* [join q left right l r] runs the join [q] of the streams [l] and [r], the
   steps [left] and [right]. A join that blocks on both pulls [l] to its end,
   then [r]; one that streams [l] pulls [r] to its end first. *)
let join q left right l r =
  let close () =
    l.close ();
    r.close ()
  in
  let emit = function
    | Ok b -> Some b
    | Error f -> failed (line Query.pp_step q) f
  in
  match Join_run.compile q with
  | Blocking f ->
      once close (fun () ->
          let l = gather left l in
          emit (f l (gather right r)))
  | Streaming { batch; last } ->
      let held = lazy (gather right r) in
      let rows = ref 0 and ended = ref false in
      let next () =
        let r = Lazy.force held in
        if !ended then None
        else
          match l.next () with
          | Some b ->
              let out =
                Result.map_error
                  (fun (f : Eval.failure) -> { f with row = !rows + f.row })
                  (batch r b)
              in
              rows := !rows + Table.rows b;
              emit out
          | None ->
              ended := true;
              emit (last r (empty (Query.schema left)) !rows)
      in
      { next; close }

(* [ordered q keys s] is [s], the rows of [q]'s source, failing at the first row
   out of the order of [keys]. *)
let ordered q keys s =
  let prev = ref None in
  streaming q s (fun b ->
      let r = Order_run.unordered keys !prev b in
      if Table.rows b > 0 then prev := Some b;
      match r with
      | None -> (b, None)
      | Some row ->
          let why =
            Format.asprintf "the row is out of the source's order %a."
              (Type.pp_list Order.pp) keys
          in
          ( sub b ~offset:0 ~length:row,
            Some { Eval.row; output = 0; cause = Data (Error.v why) } ))

(* [source q source r] reads [q]'s source with its request, and checks the
   source's order on the longest prefix of it that the request reads. *)
let source q (source : Source.t) (request : Source.request) =
  let rec read = function
    | (k : Order.t) :: ks when List.mem k.name request.columns -> k :: read ks
    | _ -> []
  in
  let s = of_source q source request in
  match read source.sorted with [] -> s | keys -> ordered q keys s

(* [readers q] is the number of reads of each step of [q] by the steps above it,
   [q] itself being read once. *)
let readers q =
  let counts = ref [] in
  let rec visit q =
    match List.assq_opt q !counts with
    | Some n -> incr n
    | None ->
        counts := (q, ref 1) :: !counts;
        List.iter visit (Query.inputs q)
  in
  visit q;
  fun q -> !(List.assq q !counts)

(* A step whose expressions read other rows than their own blocks. *)
let local outputs =
  List.for_all (fun (_, Expr.Packed e) -> Expr.row_local e) outputs

(* [compile q] is the stream of [q]'s rows. A step that several steps read is
   one stream, read through a tee. A slice from the start of a sort read once is
   a top-k. *)
let compile q =
  let readers = readers q and tees = ref [] in
  let rec stream q =
    if readers q = 1 then step q
    else
      let rest =
        match List.assq_opt q !tees with
        | Some rest -> rest
        | None ->
            let s = step q in
            let tee = Tee.v (readers q) s.next s.close in
            let rest =
              ref (List.map (fun (next, close) -> { next; close }) tee)
            in
            tees := (q, rest) :: !tees;
            rest
      in
      match !rest with
      | s :: ss ->
          rest := ss;
          s
      | [] -> assert false
  and step q =
    let step input local f =
      if local then streaming q (stream input) f
      else blocking q input (stream input) f
    in
    match Query.node q with
    | Of_table t -> of_table t
    | Of_source { source = src; columns; filters; limit } ->
        let filters =
          List.map (fun c -> Option.get (Optimize.pred c)) filters
        in
        source q src { columns; filters; limit }
    | Select { outputs; input } ->
        step input (local outputs) (select q outputs input)
    | Derive { outputs; input } ->
        step input (local outputs) (derive q outputs input)
    | Unnest { outputs; keep_empty; input; _ } ->
        step input (local outputs) (unnest q outputs keep_empty input)
    | Filter { predicate; input } ->
        step input (Expr.row_local predicate) (filter q predicate input)
    | Aggregate { by; outputs; input } ->
        partial q by outputs input (stream input)
    | Sort { keys; input } ->
        step input false (fun b -> (Order_run.sort keys b, None))
    | Slice { offset; length; input } when offset >= 0 -> (
        match Query.node input with
        | Sort { keys; input = sorted } when readers input = 1 && length > 0 ->
            step sorted false (fun b ->
                (Order_run.top_k ~offset ~length keys b, None))
        | _ -> head ~offset ~length (stream input))
    | Slice { offset; length; input } -> tail ~offset ~length (stream input)
    | Append { input; rest } ->
        append q (stream input) (stream rest) ~input:(Query.schema input)
          ~rest:(Query.schema rest)
    | Join { left; right; _ } -> join q left right (stream left) (stream right)
  in
  stream q

(* Running *)

let fold q ~init f =
  let s = compile (Optimize.query q) in
  let rec loop acc =
    match s.next () with
    | None -> acc
    | Some b -> loop (if Table.rows b = 0 then acc else f acc b)
  in
  match Fun.protect ~finally:s.close (fun () -> loop init) with
  | acc -> Ok acc
  | exception Failed e -> Error e

let run q =
  let batches = fold q ~init:[] (fun bs b -> b :: bs) in
  Result.map
    (fun bs ->
      match List.rev bs with
      | [] -> empty (Query.schema q)
      | bs -> Table.canonical (Table.concat (Table.of_batches bs)))
    batches

let values e q =
  let e = Query.check_values e q in
  let read n =
    let (Type.Any ty) = Option.get (Schema.find (Query.schema q) n) in
    (n, Expr.Packed (Expr.read ty n))
  in
  let input =
    Query.make (Select { outputs = List.map read (Expr.reads e); input = q })
  in
  let eval = Eval.values (Query.schema input) e in
  let step = "values " ^ line Expr.pp_arg e in
  let value (rows, parts) b =
    let vs, f = eval (Eval.frame b) in
    let stop = match f with Some f -> f.row | None -> Table.rows b in
    (match Array.find_index Option.is_none (Array.sub vs 0 stop) with
    | Some i ->
        let null = "the value is null; read it through Expr.option." in
        failed step { row = rows + i; output = 0; cause = Data (Error.v null) }
    | None ->
        Option.iter
          (fun (f : Eval.failure) -> failed step { f with row = rows + f.row })
          f);
    (rows + Table.rows b, Array.map Option.get vs :: parts)
  in
  Result.map
    (fun (_, parts) -> Array.concat (List.rev parts))
    (fold input ~init:(0, []) value)
