(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Scalars
module A = Bigarray.Array1

type table = (int, Bigarray.int_elt, Bigarray.c_layout) A.t
type classes = (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) A.t

(* Over byte classes rather than bytes, states are numbered at multiples of
   [ncls], so a state number is its row's start in [trans]. *)

let unknown = -1
let accept = -2
let dead = -3

(* A kernel reads one column, so the cache lives for one kernel. Its table is
   bounded by words: a pattern with many byte classes gets fewer states. *)
let table_budget_words = 1 lsl 20
let default_limit ~ncls = Int.max 64 (table_budget_words / ncls)

type t = {
  prog : Regex.inst array;
  classes : classes;
  ncls : int;
  reps : int array;  (* a byte of each class *)
  mutable trans : table;
  mutable states : int array array;  (* NFA set of each state, by number *)
  mutable count : int;  (* states in use *)
  capacity : int;  (* its limit *)
  table : (string, int) Hashtbl.t;
  mutable start : int;  (* the start state's number, or [unknown] *)
  mutable start_set : int array;
  mutable empty_row : bool;  (* the answer for a row of no bytes *)
  mutable ends : Bytes.t;  (* a byte per state: 0 unknown, 1 and 2 its answer *)
  out : table;  (* one word, the C scan's resting state *)
  seen : int array;
  mutable gen : int;
}

(* A row's bytes are grouped into the classes of its Sparse boundaries: two
   bytes of a class behave alike in every state. *)
let byte_classes prog =
  let boundary = Array.make 257 false in
  boundary.(0) <- true;
  Array.iter
    (function
      | Regex.Sparse a ->
          let i = ref 0 in
          while !i < Array.length a do
            boundary.(a.(!i)) <- true;
            boundary.(a.(!i + 1) + 1) <- true;
            i := !i + 3
          done
      | _ -> ())
    prog;
  let classes = A.create Bigarray.int8_unsigned Bigarray.c_layout 256 in
  let reps = Array.make 256 0 in
  let c = ref (-1) in
  for b = 0 to 255 do
    if boundary.(b) then begin
      incr c;
      reps.(!c) <- b
    end;
    A.unsafe_set classes b !c
  done;
  (classes, !c + 1, Array.sub reps 0 (!c + 1))

(* [closure d pcs ~at_start ~at_end] is the sorted set reachable from the
   instructions [pcs] by empty moves: Save is skipped, Start_text is followed
   only at the row's start and End_text only at its end. An End_text kept
   unfollowed in a state is resolved when the row ends. *)
let closure d pcs ~at_start ~at_end =
  d.gen <- d.gen + 1;
  let gen = d.gen in
  let out = ref [] in
  let rec go pc =
    if d.seen.(pc) <> gen then begin
      d.seen.(pc) <- gen;
      match d.prog.(pc) with
      | Regex.Split (a, b) ->
          go a;
          go b
      | Save (_, nx) -> go nx
      | Assert (Start_text, nx) -> if at_start then go nx
      | Assert (End_text, nx) -> if at_end then go nx else out := pc :: !out
      | Assert ((Start_line | End_line | Word_boundary | Not_word_boundary), _) ->
          assert false
      | Sparse _ | Accept -> out := pc :: !out
    end
  in
  List.iter go pcs;
  let set = Array.of_list !out in
  Array.sort Int.compare set;
  set

let set_accepts d set = Array.exists (fun pc -> d.prog.(pc) = Regex.Accept) set

let intern d set =
  if Array.length set = 0 then dead
  else if set_accepts d set then accept
  else begin
    let key = Bytes.create (4 * Array.length set) in
    Array.iteri (fun i pc -> Bytes.set_int32_le key (4 * i) (Int32.of_int pc)) set;
    let key = Bytes.unsafe_to_string key in
    match Hashtbl.find_opt d.table key with
    | Some s -> s
    | None ->
        let s = d.count * d.ncls in
        Hashtbl.add d.table key s;
        d.states.(d.count) <- set;
        A.fill (A.sub d.trans s d.ncls) unknown;
        d.count <- d.count + 1;
        s
  end

(* The cache stays small on a small column: its table doubles up to [capacity]
   states, and past that it is cleared. A cleared state's number is gone, so a
   transition that clears is not written, and the next use of the start state
   re-adds it. *)
let make_room d =
  if d.count = Array.length d.states then begin
    if d.count < d.capacity then begin
      let size = Int.min d.capacity (d.count * 2) in
      let bigger = A.create Bigarray.int Bigarray.c_layout (size * d.ncls) in
      A.blit d.trans (A.sub bigger 0 (d.count * d.ncls));
      d.trans <- bigger;
      let states = Array.make size [||] in
      Array.blit d.states 0 states 0 d.count;
      d.states <- states;
      let ends = Bytes.make size '\000' in
      Bytes.blit d.ends 0 ends 0 d.count;
      d.ends <- ends;
      false
    end
    else begin
      Hashtbl.reset d.table;
      Bytes.fill d.ends 0 (Bytes.length d.ends) '\000';
      d.count <- 0;
      d.start <- unknown;
      true
    end
  end
  else false

let start_state d =
  if d.start = unknown then begin
    ignore (make_room d);
    d.start <- intern d d.start_set
  end;
  d.start

(* [resolve d s c] is the state after state [s] (its table row) on byte class
   [c], built and cached when it is unknown. *)
let resolve d s c =
  let set = d.states.(s / d.ncls) in
  let rep = d.reps.(c) in
  let targets = ref [] in
  Array.iter
    (fun pc ->
      match d.prog.(pc) with
      | Regex.Sparse a ->
          let nx = Regex.sparse_next a rep in
          if nx >= 0 then targets := nx :: !targets
      | Assert (End_text, _) | Accept -> ()
      | Split _ | Save _ | Assert _ -> assert false)
    set;
  let next = closure d !targets ~at_start:false ~at_end:false in
  let cleared = make_room d in
  let t = intern d next in
  if not cleared then A.unsafe_set d.trans (s + c) t;
  t

let create ?limit r =
  if not (Regex.dfa_handles r) then
    invalid_arg "Regex_dfa.create: the program uses \\b or a (?m) anchor";
  let prog = Regex.program r in
  let classes, ncls, reps = byte_classes prog in
  let limit =
    match limit with Some l -> l | None -> default_limit ~ncls
  in
  if limit < 2 then invalid_arg "Regex_dfa.create: limit below 2";
  let initial = Int.min limit 16 in
  let seen = Array.make (Array.length prog) 0 in
  let d =
    {
      prog;
      classes;
      ncls;
      reps;
      trans = A.create Bigarray.int Bigarray.c_layout (initial * ncls);
      states = Array.make initial [||];
      count = 0;
      capacity = limit;
      table = Hashtbl.create 64;
      start = unknown;
      start_set = [||];
      empty_row = false;
      ends = Bytes.make initial '\000';
      out = A.create Bigarray.int Bigarray.c_layout 1;
      seen;
      gen = 0;
    }
  in
  let entry = Regex.dfa_entry r in
  d.start_set <- closure d [ entry ] ~at_start:true ~at_end:false;
  d.empty_row <-
    set_accepts d (closure d [ entry ] ~at_start:true ~at_end:true);
  d

(* [ends_at_row_end d s] is [true] iff an End_text kept in [s] reaches Accept,
   the row having ended, computed once per state. *)
let ends_at_row_end d s =
  let i = s / d.ncls in
  match Bytes.get d.ends i with
  | '\002' -> false
  | '\001' -> true
  | _ ->
      let holds =
        set_accepts d
          (closure d (Array.to_list d.states.(i)) ~at_start:false ~at_end:true)
      in
      Bytes.set d.ends i (if holds then '\001' else '\002');
      holds

external c_scan :
  table ->
  classes ->
  buf ->
  (int[@untagged]) ->
  (int[@untagged]) ->
  (int[@untagged]) ->
  table ->
  (int[@untagged]) = "talon_dfa_scan_byte" "talon_dfa_scan"
[@@noalloc]

let matches d (v : buf) ~first ~stop =
  if first = stop then d.empty_row
  else begin
    let s = ref (start_state d) and at = ref first and answer = ref (-1) in
    (* The start state may itself be Accept or dead. *)
    if !s = accept then true
    else if !s = dead then false
    else begin
      while !answer < 0 do
        at := c_scan d.trans d.classes v !s !at stop d.out;
        s := A.unsafe_get d.out 0;
        if !at >= stop then
          answer := if ends_at_row_end d !s then 1 else 0
        else begin
          let c = A.unsafe_get d.classes (A.unsafe_get v !at) in
          let next =
            let t = A.unsafe_get d.trans (!s + c) in
            if t <> unknown then t else resolve d !s c
          in
          if next = accept then answer := 1
          else if next = dead then answer := 0
          else begin
            s := next;
            at := !at + 1
          end
        end
      done;
      !answer = 1
    end
  end
