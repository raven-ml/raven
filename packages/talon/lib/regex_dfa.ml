(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Scalars
module A = Bigarray.Array1

type table = (int, Bigarray.int_elt, Bigarray.c_layout) A.t
type classes = (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) A.t

(* A state is the set of NFA instructions alive at one position. States are
   numbered at multiples of [width], so a state's number is its row's start in
   [trans], which the C scan reads directly. A row has an entry per byte class
   and a last one, [row_end], for the end of the row. *)

let unknown = -1
let accept = -2
let dead = -3

(* What the cache may hold, in words: a kernel's memory for this DFA, whatever
   its rows. A state costs its table row, its set and its key; a set can reach
   the size of the program, so counting only states would hide that factor. *)
let budget_words = 1 lsl 20

(* Words a state costs beyond its row and set: the key's header, its Hashtbl
   bucket and the entry of [states]. *)
let state_overhead = 8

type t = {
  prog : Regex.inst array;
  classes : classes;
  width : int;
  row_end : int;  (* the class of the row's end *)
  reps : int array;  (* a byte of each class *)
  mutable trans : table;
  mutable states : int array array;  (* the set of each state, by number *)
  mutable count : int;
  mutable used : int;  (* words charged to the cache *)
  mutable epoch : int;  (* how many times the cache has been cleared *)
  table : (string, int) Hashtbl.t;
  mutable start : int;  (* the start state's number, or [unknown] *)
  mutable start_set : int array;
  mutable empty_row : bool;  (* the answer for a row of no bytes *)
  out : table;  (* one word, where the C scan leaves its state *)
  seen : int array;
  mutable gen : int;
}

(* Bytes that every Sparse treats alike share a class, so a table row is
   [width] entries rather than 256. *)
let byte_classes prog =
  let boundary = Array.make 257 false in
  boundary.(0) <- true;
  let add = function
    | Regex.Sparse a ->
        for i = 0 to (Array.length a / 3) - 1 do
          boundary.(a.(3 * i)) <- true;
          boundary.(a.((3 * i) + 1) + 1) <- true
        done
    | _ -> ()
  in
  Array.iter add prog;
  let classes = A.create Bigarray.int8_unsigned Bigarray.c_layout 256 in
  let reps = Array.make 256 0 and c = ref (-1) in
  for b = 0 to 255 do
    if boundary.(b) then begin
      incr c;
      reps.(!c) <- b
    end;
    A.unsafe_set classes b !c
  done;
  (classes, !c + 1, reps)

(* [closure d pcs ~at_start ~at_end] is the set of instructions reachable from
   [pcs] by empty moves, ascending. Save is passed through; Start_text is
   followed only at the row's start, End_text only at its end. An End_text not
   yet followed stays in the set and is resolved when the row ends. *)
let closure d pcs ~at_start ~at_end =
  d.gen <- d.gen + 1;
  let gen = d.gen in
  let rec go pc =
    if d.seen.(pc) <> gen then begin
      d.seen.(pc) <- gen;
      match d.prog.(pc) with
      | Regex.Split (a, b) ->
          go a;
          go b
      | Save (_, nx) -> go nx
      | Assert (Start_text, nx) -> if at_start then go nx
      | Assert (End_text, nx) -> if at_end then go nx
      | Assert _ -> assert false (* [create] refuses the others. *)
      | Sparse _ | Accept -> ()
    end
  in
  List.iter go pcs;
  (* Scanning in instruction order gives the set its canonical order in time
     proportional to the program, within the promised bound. *)
  let kept pc =
    d.seen.(pc) = gen
    &&
    match d.prog.(pc) with
    | Regex.Sparse _ | Accept | Assert (End_text, _) -> true
    | _ -> false
  in
  let size = ref 0 in
  for pc = 0 to Array.length d.prog - 1 do
    if kept pc then incr size
  done;
  let set = Array.make !size 0 and at = ref 0 in
  for pc = 0 to Array.length d.prog - 1 do
    if kept pc then begin
      set.(!at) <- pc;
      incr at
    end
  done;
  set

let accepts d set =
  Array.exists
    (fun pc -> match d.prog.(pc) with Regex.Accept -> true | _ -> false)
    set

let clear d =
  Hashtbl.reset d.table;
  d.count <- 0;
  d.used <- 0;
  d.epoch <- d.epoch + 1;
  d.start <- unknown

let grow d =
  let size = 2 * Array.length d.states in
  let trans = A.create Bigarray.int Bigarray.c_layout (size * d.width) in
  A.blit d.trans (A.sub trans 0 (d.count * d.width));
  d.trans <- trans;
  let states = Array.make size [||] in
  Array.blit d.states 0 states 0 d.count;
  d.states <- states

(* [intern d set] is the state of [set], or the sentinel it stands for. Making
   room may clear the cache, so a state over the budget by itself still fits
   alone: the search goes on, at a cost per byte, and memory stays bounded. *)
let intern d set =
  if Array.length set = 0 then dead
  else if accepts d set then accept
  else begin
    let n = Array.length set in
    let key = Bytes.create (4 * n) in
    Array.iteri (fun i pc -> Bytes.set_int32_le key (4 * i) (Int32.of_int pc)) set;
    let key = Bytes.unsafe_to_string key in
    match Hashtbl.find_opt d.table key with
    | Some s -> s
    | None ->
        let cost = d.width + (2 * n) + state_overhead in
        if d.used + cost > budget_words && d.count > 0 then clear d;
        if d.count = Array.length d.states then grow d;
        let s = d.count * d.width in
        Hashtbl.add d.table key s;
        d.states.(d.count) <- set;
        A.fill (A.sub d.trans s d.width) unknown;
        d.count <- d.count + 1;
        d.used <- d.used + cost;
        s
  end

let start_state d =
  if d.start = unknown then d.start <- intern d d.start_set;
  d.start

(* [resolve d s c] is the state after state [s] on class [c], or the row's
   answer for the class [row_end]. A transition found across a clear is not
   stored: [s] has lost its number. *)
let resolve d s c =
  let set = d.states.(s / d.width) in
  if c = d.row_end then begin
    let t =
      if accepts d (closure d (Array.to_list set) ~at_start:false ~at_end:true)
      then accept
      else dead
    in
    A.unsafe_set d.trans (s + c) t;
    t
  end
  else begin
    let rep = d.reps.(c) in
    let step pc =
      match d.prog.(pc) with
      | Regex.Sparse a -> Regex.sparse_next a rep
      | _ -> -1
    in
    let targets =
      Array.fold_left
        (fun acc pc ->
          let nx = step pc in
          if nx >= 0 then nx :: acc else acc)
        [] set
    in
    let next = closure d targets ~at_start:false ~at_end:false in
    let epoch = d.epoch in
    let t = intern d next in
    if d.epoch = epoch then A.unsafe_set d.trans (s + c) t;
    t
  end

let create r =
  if not (Regex.dfa_handles r) then
    invalid_arg "Regex_dfa.create: the program uses \\b or a (?m) anchor";
  let prog = Regex.program r in
  let classes, ncls, reps = byte_classes prog in
  let width = ncls + 1 and initial = 16 in
  let d =
    {
      prog;
      classes;
      width;
      row_end = ncls;
      reps;
      trans = A.create Bigarray.int Bigarray.c_layout (initial * width);
      states = Array.make initial [||];
      count = 0;
      used = 0;
      epoch = 0;
      table = Hashtbl.create 64;
      start = unknown;
      start_set = [||];
      empty_row = false;
      out = A.create Bigarray.int Bigarray.c_layout 1;
      seen = Array.make (Array.length prog) 0;
      gen = 0;
    }
  in
  let entry = Regex.start r in
  d.start_set <- closure d [ entry ] ~at_start:true ~at_end:false;
  d.empty_row <- accepts d (closure d [ entry ] ~at_start:true ~at_end:true);
  d

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

(* [next_state d s c] is the entry of state [s] on class [c], resolved if the
   cache does not know it yet. *)
let next_state d s c =
  let t = A.unsafe_get d.trans (s + c) in
  if t <> unknown then t else resolve d s c

let matches d (v : buf) ~first ~stop =
  if first = stop then d.empty_row
  else
    let start = start_state d in
    if start = accept then true
    else if start = dead then false
    else begin
      let s = ref start and at = ref first and answer = ref unknown in
      while !answer = unknown do
        at := c_scan d.trans d.classes v !s !at stop d.out;
        s := A.unsafe_get d.out 0;
        let c =
          if !at >= stop then d.row_end
          else A.unsafe_get d.classes (A.unsafe_get v !at)
        in
        let t = next_state d !s c in
        if t = accept || t = dead then answer := t
        else begin
          s := t;
          incr at
        end
      done;
      !answer = accept
    end
