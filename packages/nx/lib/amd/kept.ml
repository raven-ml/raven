(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* An entry's [used] is the clock's value at its last use, so the least
   recently used entry has the least. A keep replaces the cells and their
   bytes in one compare-and-set; a find reads them and raises [used] alone. *)

type ('k, 'v) entry = { key : 'k; value : 'v; size : int; used : int Atomic.t }

type ('k, 'v) cells = { cells : ('k, 'v) entry list array; bytes : int }

type ('k, 'v) t = {
  state : ('k, 'v) cells Atomic.t;
  clock : int Atomic.t;
  count : int;
}

let make ~cells ~count =
  if cells < 0 then invalid_arg "Kept.make: cells";
  if count <= 0 then invalid_arg "Kept.make: count";
  let state = Atomic.make { cells = Array.make cells []; bytes = 0 } in
  { state; clock = Atomic.make 0; count }

let cell fn s i =
  if i < 0 || i >= Array.length s.cells then
    invalid_arg ("Kept." ^ fn ^ ": cell");
  s.cells.(i)

(* Marks [e] used, unless no use came since its last one. [used] only
   rises: a find that took its stamp before another's must not set the
   older one over it. *)
let use t e =
  if Atomic.get e.used <> Atomic.get t.clock - 1 then begin
    let stamp = Atomic.fetch_and_add t.clock 1 in
    let rec raise_to () =
      let u = Atomic.get e.used in
      if u < stamp && not (Atomic.compare_and_set e.used u stamp) then
        raise_to ()
    in
    raise_to ()
  end

let rec first_by t f x = function
  | [] -> None
  | e :: rest ->
      if f e.key x then begin
        use t e;
        Some e.value
      end
      else first_by t f x rest

let find_by f t i x = first_by t f x (cell "find_by" (Atomic.get t.state) i)

(* The least recently used entry of [es], if any. *)
let oldest_of es =
  List.fold_left
    (fun o e ->
      match o with
      | Some b when Atomic.get b.used <= Atomic.get e.used -> o
      | _ -> Some e)
    None es

(* The cell and the least recently used entry of [cells], if any. *)
let oldest cells =
  let o = ref None in
  Array.iteri
    (fun i es ->
      match (oldest_of es, !o) with
      | Some e, Some (_, b) when Atomic.get b.used <= Atomic.get e.used -> ()
      | Some e, _ -> o := Some (i, e)
      | None, _ -> ())
    cells;
  !o

let drop cells i e = cells.(i) <- List.filter (fun x -> x != e) cells.(i)

(* [bytes] of [cells] once their least recently used entries are dropped
   while they and [size] exceed [most]. *)
let rec make_room cells ~most ~size bytes =
  if bytes + size <= most then bytes
  else
    match oldest cells with
    | None -> bytes
    | Some (i, e) ->
        drop cells i e;
        make_room cells ~most ~size (bytes - e.size)

let keep t i ~most ~bytes:size key value =
  if size < 0 then invalid_arg "Kept.keep: bytes";
  ignore (cell "keep" (Atomic.get t.state) i);
  let rec add () =
    let s = Atomic.get t.state in
    let cells = Array.copy s.cells in
    let bytes =
      if List.length cells.(i) < t.count then s.bytes
      else
        match oldest_of cells.(i) with
        | None -> s.bytes
        | Some e ->
            drop cells i e;
            s.bytes - e.size
    in
    let bytes = make_room cells ~most ~size bytes in
    let used = Atomic.make (Atomic.fetch_and_add t.clock 1) in
    cells.(i) <- { key; value; size; used } :: cells.(i);
    let kept = { cells; bytes = bytes + size } in
    if not (Atomic.compare_and_set t.state s kept) then add ()
  in
  if size <= most then add ()

let clear t =
  let s = Atomic.get t.state in
  Atomic.set t.state { cells = Array.make (Array.length s.cells) []; bytes = 0 }

let count t =
  Array.fold_left (fun n es -> n + List.length es) 0 (Atomic.get t.state).cells

let bytes t = (Atomic.get t.state).bytes
