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

(* [blocking q input s step] is [step] applied once, to all of [s]'s rows, the
   input of [q]'s step. It emits nothing when [step] fails. *)
let blocking q input s step =
  let ran = ref false in
  let next () =
    if !ran then None
    else begin
      ran := true;
      match step (gather input s) with
      | out, None -> Some out
      | _, Some f -> failed (line Query.pp_step q) f
    end
  in
  { next; close = s.close }

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

let filter q predicate input =
  let eval = Eval.predicate (Query.schema input) predicate in
  fun b ->
    let n = Table.rows b in
    let keep, f = eval (Eval.frame b) in
    let keep =
      match f with None -> keep | Some f -> Nx.shrink [| (0, f.row) |] keep
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
    let keys = List.map (Column.take (Reduce.first s)) keys in
    ( Table.batch (Query.schema q) ~rows:(Reduce.count s)
        (Array.of_list (keys @ outs)),
      f )

(* [head ~offset ~length s] is [s]'s rows [offset] to [offset + length - 1]; it
   closes [s] past them. *)
let head ~offset ~length s =
  let stop = if length > max_int - offset then max_int else offset + length in
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
let append q a r rest =
  let names = Schema.names rest in
  let order =
    List.map
      (fun n -> Option.get (List.find_index (String.equal n) names))
      (Schema.names (Query.schema q))
  in
  let reorder b =
    let cs = Table.columns b in
    Table.batch (Query.schema q) ~rows:(Table.rows b)
      (Array.of_list (List.map (Array.get cs) order))
  in
  let first = ref true in
  let rec next () =
    if not !first then Option.map reorder (r.next ())
    else
      match a.next () with
      | Some _ as b -> b
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
  | None -> Eval.not_lowered (line Query.pp_step q)
  | Some (Blocking f) ->
      let ran = ref false in
      let next () =
        if !ran then None
        else begin
          ran := true;
          let l = gather left l in
          emit (f l (gather right r))
        end
      in
      { next; close }
  | Some (Streaming { batch; last }) ->
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

(* A step whose expressions read other rows than their own blocks. *)
let local outputs =
  List.for_all (fun (_, Expr.Packed e) -> Expr.row_local e) outputs

let rec stream q =
  let step input local f =
    if local then streaming q (stream input) f
    else blocking q input (stream input) f
  in
  match Query.node q with
  | Of_table t -> of_table t
  | Select { outputs; input } ->
      step input (local outputs) (select q outputs input)
  | Derive { outputs; input } ->
      step input (local outputs) (derive q outputs input)
  | Filter { predicate; input } ->
      step input (Expr.row_local predicate) (filter q predicate input)
  | Aggregate { by; outputs; input } ->
      step input false (aggregate q by outputs input)
  | Slice { offset; length; input } ->
      if offset >= 0 then head ~offset ~length (stream input)
      else tail ~offset ~length (stream input)
  | Append { input; rest } ->
      append q (stream input) (stream rest) (Query.schema rest)
  | Join { left; right; _ } -> join q left right (stream left) (stream right)
  | Of_source _ | Sort _ | Unnest _ -> Eval.not_lowered (line Query.pp_step q)

(* Running *)

let fold q ~init f =
  let s = stream (Optimize.query q) in
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
      | bs -> Table.concat (Table.of_batches bs))
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
        failed step { row = rows + i; cause = Data (Error.v null) }
    | None ->
        Option.iter
          (fun (f : Eval.failure) -> failed step { f with row = rows + f.row })
          f);
    (rows + Table.rows b, Array.map Option.get vs :: parts)
  in
  Result.map
    (fun (_, parts) -> Array.concat (List.rev parts))
    (fold input ~init:(0, []) value)
