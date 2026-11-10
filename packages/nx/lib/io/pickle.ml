(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  Python's unpickling machine, as CPython's Lib/pickletools.py documents its
  opcodes, run with each call kept as data. Framing follows PEP 3154: no opcode
  straddles a frame's end.
  ---------------------------------------------------------------------------*)

open Bigarray

type bytes = (int, int8_unsigned_elt, c_layout) Array1.t

type value =
  | Null
  | Bool of bool
  | Int of int
  | Long of string * int
  | Float of float
  | String of string * int
  | Bytes of string * int
  | Global of string * int
  | Node of node

and node = { id : int; at : int; kind : kind; mutable complete : bool }

and kind =
  | Tuple of value array
  | Frozenset of value array
  | Bytearray of string
  | List of value Dynarray.t
  | Dict of (value * value) Dynarray.t
  | Set of value Dynarray.t
  | Persistent of value
  | Call of call

and call = {
  fn : value;
  args : value;
  kwargs : value option;
  new_ : bool;
  mutable state : value option;
  items : value Dynarray.t;
  entries : (value * value) Dynarray.t;
}

type decoded = {
  value : value;
  stop : int;
  nodes : int;
  atoms : (string, int) Hashtbl.t;
}

(* The highest protocol, and the first that Python 3 reads without mapping
   Python 2's names (pickle.py, find_class). *)
let highest_protocol = 5
let python3_protocol = 3

(* Python 2's modules that Python 3 reads under other names
   (_compat_pickle.IMPORT_MAPPING), of those a load recognises. *)
let python2_modules = [ ("__builtin__", "builtins"); ("copy_reg", "copyreg") ]

(* Integers *)

(* CPython's bound on the decimal digits of an int it reads from text, its
   [sys.int_info.default_max_str_digits]: [int (s, 0)] refuses more. *)
let max_str_digits = 4300

(* [integer s] is the integer whose two's complement, little-endian, is [s], as
   LONG1 and LONG4 hold it: [Ok i] within [int]'s range, and [Error b]
   otherwise, [b] the fewest bytes that hold it. *)
let integer s =
  let byte i = Char.code s.[i] in
  let n = ref (String.length s) in
  (* A last byte that only extends the sign of the one before is redundant. *)
  while
    !n >= 2
    && ((byte (!n - 1) = 0 && byte (!n - 2) land 0x80 = 0)
       || (byte (!n - 1) = 0xff && byte (!n - 2) land 0x80 <> 0))
  do
    decr n
  done;
  let n = !n in
  if n > 8 then Error (String.sub s 0 n)
  else
    let v = ref (if n > 0 && byte (n - 1) land 0x80 <> 0 then -1L else 0L) in
    for i = n - 1 downto 0 do
      v := Int64.logor (Int64.shift_left !v 8) (Int64.of_int (byte i))
    done;
    if
      Int64.compare !v (Int64.of_int min_int) >= 0
      && Int64.compare !v (Int64.of_int max_int) <= 0
    then Ok (Int64.to_int !v)
    else Error (String.sub s 0 n)

(* [decimal at s] is the integer of the decimal [s], as {!integer} gives it: an
   optional [-], then at most [max_str_digits] digits with no leading zero but
   in zero itself, as Python's [int (s, 0)] reads INT's and LONG's text. The
   magnitude is built in 24-bit limbs, nine digits a step: each step multiplies
   the limbs reached so far by 10^9 and adds the digits. *)
let decimal at s =
  let n = String.length s in
  let first = if n > 0 && s.[0] = '-' then 1 else 0 in
  let digit c = c >= '0' && c <= '9' in
  let digits = String.sub s first (n - first) in
  let zero = String.for_all (Char.equal '0') digits in
  if
    digits = ""
    || (not (String.for_all digit digits))
    || (digits.[0] = '0' && not zero)
  then Fail.failf "byte %d: %S is not a decimal integer" at s;
  let count = String.length digits in
  if count > max_str_digits then
    Fail.failf "byte %d: an integer of more than %d decimal digits" at
      max_str_digits;
  match int_of_string_opt s with
  | Some i -> Ok i
  | None ->
      (* A digit holds under 3.33 bits: [count / 7 + 1] limbs of 24 bits hold
         the magnitude. A limb times 10^9 plus a carry stays under 2^54. *)
      let limbs = Array.make ((count / 7) + 2) 0 in
      let live = ref 0 in
      let pos = ref 0 in
      while !pos < count do
        let k = Int.min 9 (count - !pos) in
        let scale = ref 1 and chunk = ref 0 in
        for i = !pos to !pos + k - 1 do
          scale := !scale * 10;
          chunk := (!chunk * 10) + (Char.code digits.[i] - Char.code '0')
        done;
        pos := !pos + k;
        let carry = ref !chunk in
        for i = 0 to !live - 1 do
          let v = (limbs.(i) * !scale) + !carry in
          limbs.(i) <- v land 0xffffff;
          carry := v lsr 24
        done;
        while !carry > 0 do
          limbs.(!live) <- !carry land 0xffffff;
          carry := !carry lsr 24;
          incr live
        done
      done;
      (* The magnitude's bytes, little-endian, then a byte of room for the sign
         of its two's complement. *)
      let mag = Bytes.make ((3 * !live) + 1) '\000' in
      for i = 0 to (3 * !live) - 1 do
        Bytes.set mag i
          (Char.chr ((limbs.(i / 3) lsr (8 * (i mod 3))) land 0xff))
      done;
      if first = 1 then begin
        let carry = ref 1 in
        for i = 0 to Bytes.length mag - 1 do
          let v = (lnot (Char.code (Bytes.get mag i)) land 0xff) + !carry in
          Bytes.set mag i (Char.chr (v land 0xff));
          carry := v lsr 8
        done
      end;
      integer (Bytes.to_string mag)

(* Strings *)

let continuation c = c land 0xc0 = 0x80

(* [utf8_surrogates s] is whether [s] is UTF-8 in which the encodings of
   surrogate code points may occur, as Python's [surrogatepass] writes a lone
   one. *)
let utf8_surrogates s =
  let n = String.length s in
  let byte i = if i < n then Char.code s.[i] else 0 in
  let rec go i =
    if i >= n then true
    else
      let c = byte i in
      if c < 0x80 then go (i + 1)
      else if c < 0xc2 then false
      else if c < 0xe0 then continuation (byte (i + 1)) && go (i + 2)
      else if c < 0xf0 then
        let b1 = byte (i + 1) in
        continuation b1
        && (c <> 0xe0 || b1 >= 0xa0)
        && continuation (byte (i + 2))
        && go (i + 3)
      else if c < 0xf5 then
        let b1 = byte (i + 1) in
        continuation b1
        && (c <> 0xf0 || b1 >= 0x90)
        && (c <> 0xf4 || b1 < 0x90)
        && continuation (byte (i + 2))
        && continuation (byte (i + 3))
        && go (i + 4)
      else false
  in
  go 0

(* The machine *)

type machine = {
  data : bytes;
  start : int;
  mutable pos : int;
  mutable frame_end : int;  (** [-1] outside a frame. *)
  stack : value Dynarray.t;
  mutable marks : int list;  (** The stack's lengths at each [MARK]. *)
  memo : (int, value) Hashtbl.t;
  mutable proto : int;
  mutable nodes : int;
  atoms : (string, int) Hashtbl.t;
}

(* [atom atoms tag s] numbers the string [s] of kind [tag]: equal numbers are
   equal strings of one kind. *)
let atom atoms tag s =
  let k = String.make 1 tag ^ s in
  match Hashtbl.find_opt atoms k with
  | Some i -> i
  | None ->
      let i = Hashtbl.length atoms in
      Hashtbl.add atoms k i;
      i

let int_value m = function
  | Ok i -> Int i
  | Error b -> Long (b, atom m.atoms 'L' b)

let fail m at fmt = Fail.failf ("byte %d: " ^^ fmt) (at - m.start)

(* [take m at n] is the position of the next [n] bytes of the opcode at [at],
   which it consumes. *)
let take m at n =
  let limit = if m.frame_end >= 0 then m.frame_end else Array1.dim m.data in
  if n < 0 || n > limit - m.pos then
    if m.frame_end >= 0 then fail m at "an opcode across its frame's end"
    else fail m at "the pickle ends inside an opcode";
  let p = m.pos in
  m.pos <- p + n;
  p

let sub m p n = String.init n (fun i -> Char.unsafe_chr m.data.{p + i})
let u8 m at = m.data.{take m at 1}

(* [le m at n] is the [n]-byte little-endian unsigned integer of the opcode at
   [at], [n] at most 7. *)
let le m at n =
  let p = take m at n in
  let v = ref 0 in
  for i = n - 1 downto 0 do
    v := (!v lsl 8) lor m.data.{p + i}
  done;
  !v

let le32_signed m at =
  let v = le m at 4 in
  if v land 0x8000_0000 <> 0 then v - 0x1_0000_0000 else v

(* An 8-byte length, which a pickle states unsigned. *)
let le64 m at =
  let p = take m at 8 in
  if m.data.{p + 7} land 0x80 <> 0 then fail m at "a length beyond 2^63";
  let v = ref 0L in
  for i = 7 downto 0 do
    v := Int64.logor (Int64.shift_left !v 8) (Int64.of_int m.data.{p + i})
  done;
  if Int64.compare !v (Int64.of_int max_int) > 0 then
    fail m at "a length beyond OCaml's int";
  Int64.to_int !v

let line m at =
  let limit = if m.frame_end >= 0 then m.frame_end else Array1.dim m.data in
  let rec find i =
    if i >= limit then fail m at "a line without its newline"
    else if m.data.{i} = Char.code '\n' then i
    else find (i + 1)
  in
  let e = find m.pos in
  let s = sub m m.pos (e - m.pos) in
  m.pos <- e + 1;
  s

let node m at kind =
  let id = m.nodes in
  m.nodes <- id + 1;
  Node { id; at = at - m.start; kind; complete = false }

let push m v = Dynarray.add_last m.stack v

let pop m at =
  let floor = match m.marks with k :: _ -> k | [] -> 0 in
  if Dynarray.length m.stack <= floor then fail m at "an empty stack";
  Dynarray.pop_last m.stack

let top m at =
  let floor = match m.marks with k :: _ -> k | [] -> 0 in
  if Dynarray.length m.stack <= floor then fail m at "an empty stack";
  Dynarray.get_last m.stack

(* [pop_mark m at] is the values above the last [MARK], which it removes with
   the mark. *)
let pop_mark m at =
  match m.marks with
  | [] -> fail m at "no MARK"
  | k :: rest ->
      let n = Dynarray.length m.stack - k in
      let items = Array.init n (fun i -> Dynarray.get m.stack (k + i)) in
      Dynarray.truncate m.stack k;
      m.marks <- rest;
      items

(* [hold v] marks [v] complete, as a value another holds. *)
let hold = function Node n -> n.complete <- true | _ -> ()

let pairs m at items =
  let n = Array.length items in
  if n mod 2 <> 0 then fail m at "a key without its value";
  Array.init (n / 2) (fun i -> (items.(2 * i), items.((2 * i) + 1)))

(* [target m at ~what v] is the node an opcode at [at] adds [what] to, once the
   values it adds are held: refused if another value holds it, which a cycle
   needs. *)
let target m at ~what v =
  match v with
  | Node n when n.complete ->
      fail m at "%s added to a value another value holds" what
  | Node n -> n
  | _ -> fail m at "%s added to a value that takes none" what

let append m at items =
  Array.iter hold items;
  let n = target m at ~what:"items" (top m at) in
  match n.kind with
  | List l | Call { items = l; _ } -> Array.iter (Dynarray.add_last l) items
  | _ -> fail m at "items added to a value that takes none"

let set_items m at entries =
  Array.iter
    (fun (k, v) ->
      hold k;
      hold v)
    entries;
  let n = target m at ~what:"entries" (top m at) in
  match n.kind with
  | Dict d | Call { entries = d; _ } -> Array.iter (Dynarray.add_last d) entries
  | _ -> fail m at "entries added to a value that takes none"

let add_items m at items =
  Array.iter hold items;
  let n = target m at ~what:"items" (top m at) in
  match n.kind with
  | Set s | Call { items = s; _ } -> Array.iter (Dynarray.add_last s) items
  | _ -> fail m at "items added to a value that takes none"

let build m at =
  let state = pop m at in
  hold state;
  let n = target m at ~what:"a state" (top m at) in
  match n.kind with
  | Call c when c.state = None -> c.state <- Some state
  | Call _ -> fail m at "a second state"
  | _ -> fail m at "a state set on a value that takes none"

let call m at ?kwargs ~new_ fn args =
  hold fn;
  hold args;
  Option.iter hold kwargs;
  (match args with
  | Node { kind = Tuple _; _ } -> ()
  | _ -> fail m at "a call's arguments are not a tuple");
  node m at
    (Call
       {
         fn;
         args;
         kwargs;
         new_;
         state = None;
         items = Dynarray.create ();
         entries = Dynarray.create ();
       })

let global m module_ name =
  let module_ =
    match List.assoc_opt module_ python2_modules with
    | Some renamed when m.proto < python3_protocol -> renamed
    | Some _ | None -> module_
  in
  let name = module_ ^ "." ^ name in
  Global (name, atom m.atoms 'G' name)

let unicode m at s =
  if not (utf8_surrogates s) then fail m at "a string that is not UTF-8";
  String (s, atom m.atoms 'S' s)

(* A Python 2 string, read as UTF-8 as [torch.load] reads it. *)
let py2_string m at s =
  if not (String.is_valid_utf_8 s) then
    fail m at "a Python 2 string that is not UTF-8";
  String (s, atom m.atoms 'S' s)

let bytes m b = Bytes (b, atom m.atoms 'B' b)

let memo_get m at i =
  match Hashtbl.find_opt m.memo i with
  | Some v -> push m v
  | None -> fail m at "memo entry %d is unset" i

let memo_put m at i = Hashtbl.replace m.memo i (top m at)

(* [refused] names the opcodes a load refuses, with why. *)
let refused = function
  | '0' -> Some "POP discards a value"
  | '1' -> Some "POP_MARK discards values"
  | '2' -> Some "DUP copies a value"
  | 'F' | 'S' | 'V' | 'g' | 'p' | 'P' ->
      Some "a protocol-0 text opcode other than INT and LONG"
  | 'i' | 'o' -> Some "INST and OBJ build an object as its class decides"
  | '\x82' | '\x83' | '\x84' ->
      Some "EXT1, EXT2 and EXT4 index the writer's registry"
  | '\x97' -> Some "NEXT_BUFFER's buffer lies outside the file"
  | _ -> None

let decode data ~pos =
  let m =
    {
      data;
      start = pos;
      pos;
      frame_end = -1;
      stack = Dynarray.create ();
      marks = [];
      memo = Hashtbl.create ~random:true 64;
      proto = 0;
      nodes = 0;
      atoms = Hashtbl.create ~random:true 64;
    }
  in
  let result = ref None in
  while !result = None do
    if m.frame_end >= 0 && m.pos = m.frame_end then m.frame_end <- -1;
    let at = m.pos in
    if at >= Array1.dim data then fail m at "the pickle ends before its STOP";
    let op = Char.chr (u8 m at) in
    (match refused op with Some why -> fail m at "%s" why | None -> ());
    match op with
    | '.' ->
        if m.marks <> [] then fail m at "a MARK left at STOP";
        if Dynarray.length m.stack <> 1 then
          fail m at "%d values at STOP" (Dynarray.length m.stack);
        result := Some (Dynarray.get m.stack 0)
    | '\x80' ->
        let p = u8 m at in
        if p > highest_protocol then fail m at "protocol %d" p;
        m.proto <- p
    | '\x95' ->
        let n = le64 m at in
        if m.frame_end >= 0 then fail m at "a frame inside a frame";
        if n > Array1.dim data - m.pos then fail m at "a frame past the end";
        m.frame_end <- m.pos + n
    | '(' -> m.marks <- Dynarray.length m.stack :: m.marks
    | 'N' -> push m Null
    | '\x88' -> push m (Bool true)
    | '\x89' -> push m (Bool false)
    | 'I' -> (
        match line m at with
        | "00" -> push m (Bool false)
        | "01" -> push m (Bool true)
        | s -> push m (int_value m (decimal (at - m.start) s)))
    | 'L' ->
        let s = line m at in
        let s =
          if String.ends_with ~suffix:"L" s then
            String.sub s 0 (String.length s - 1)
          else s
        in
        push m (int_value m (decimal (at - m.start) s))
    | 'K' -> push m (Int (le m at 1))
    | 'M' -> push m (Int (le m at 2))
    | 'J' -> push m (Int (le32_signed m at))
    | '\x8a' ->
        let n = le m at 1 in
        push m (int_value m (integer (sub m (take m at n) n)))
    | '\x8b' ->
        let n = le32_signed m at in
        if n < 0 then fail m at "a negative length";
        push m (int_value m (integer (sub m (take m at n) n)))
    | 'G' ->
        let p = take m at 8 in
        let bits = ref 0L in
        for i = 0 to 7 do
          bits :=
            Int64.logor (Int64.shift_left !bits 8) (Int64.of_int data.{p + i})
        done;
        push m (Float (Int64.float_of_bits !bits))
    | 'X' ->
        let n = le m at 4 in
        push m (unicode m at (sub m (take m at n) n))
    | '\x8c' ->
        let n = le m at 1 in
        push m (unicode m at (sub m (take m at n) n))
    | '\x8d' ->
        let n = le64 m at in
        push m (unicode m at (sub m (take m at n) n))
    | 'T' ->
        let n = le32_signed m at in
        if n < 0 then fail m at "a negative length";
        push m (py2_string m at (sub m (take m at n) n))
    | 'U' ->
        let n = le m at 1 in
        push m (py2_string m at (sub m (take m at n) n))
    | 'B' ->
        let n = le m at 4 in
        push m (bytes m (sub m (take m at n) n))
    | 'C' ->
        let n = le m at 1 in
        push m (bytes m (sub m (take m at n) n))
    | '\x8e' ->
        let n = le64 m at in
        push m (bytes m (sub m (take m at n) n))
    | '\x96' ->
        let n = le64 m at in
        push m (node m at (Bytearray (sub m (take m at n) n)))
    | '\x98' -> (
        match top m at with
        | Bytes _ | Node { kind = Bytearray _; _ } -> ()
        | _ -> fail m at "READONLY_BUFFER on a value that is no buffer")
    | ')' -> push m (node m at (Tuple [||]))
    | '\x85' ->
        let a = pop m at in
        hold a;
        push m (node m at (Tuple [| a |]))
    | '\x86' ->
        let b = pop m at in
        let a = pop m at in
        hold a;
        hold b;
        push m (node m at (Tuple [| a; b |]))
    | '\x87' ->
        let c = pop m at in
        let b = pop m at in
        let a = pop m at in
        hold a;
        hold b;
        hold c;
        push m (node m at (Tuple [| a; b; c |]))
    | 't' ->
        let items = pop_mark m at in
        Array.iter hold items;
        push m (node m at (Tuple items))
    | ']' -> push m (node m at (List (Dynarray.create ())))
    | 'l' ->
        let items = pop_mark m at in
        Array.iter hold items;
        push m (node m at (List (Dynarray.of_array items)))
    | '}' -> push m (node m at (Dict (Dynarray.create ())))
    | 'd' ->
        let entries = pairs m at (pop_mark m at) in
        Array.iter
          (fun (k, v) ->
            hold k;
            hold v)
          entries;
        push m (node m at (Dict (Dynarray.of_array entries)))
    | '\x8f' -> push m (node m at (Set (Dynarray.create ())))
    | '\x91' ->
        let items = pop_mark m at in
        Array.iter hold items;
        push m (node m at (Frozenset items))
    | 'a' ->
        let v = pop m at in
        append m at [| v |]
    | 'e' -> append m at (pop_mark m at)
    | 's' ->
        let v = pop m at in
        let k = pop m at in
        set_items m at [| (k, v) |]
    | 'u' -> set_items m at (pairs m at (pop_mark m at))
    | '\x90' -> add_items m at (pop_mark m at)
    | 'c' ->
        let module_ = line m at in
        let name = line m at in
        if not (String.is_valid_utf_8 module_ && String.is_valid_utf_8 name)
        then fail m at "a global's name that is not UTF-8";
        push m (global m module_ name)
    | '\x93' -> (
        let name = pop m at in
        let module_ = pop m at in
        match (module_, name) with
        | String (module_, _), String (name, _) ->
            push m (global m module_ name)
        | _ -> fail m at "STACK_GLOBAL of values that are not strings")
    | 'R' ->
        let args = pop m at in
        let fn = pop m at in
        push m (call m at ~new_:false fn args)
    | '\x81' ->
        let args = pop m at in
        let cls = pop m at in
        push m (call m at ~new_:true cls args)
    | '\x92' ->
        let kwargs = pop m at in
        let args = pop m at in
        let cls = pop m at in
        (match kwargs with
        | Node { kind = Dict _; _ } -> ()
        | _ -> fail m at "NEWOBJ_EX's keywords are not a dict");
        push m (call m at ~kwargs ~new_:true cls args)
    | 'b' -> build m at
    | 'Q' ->
        let pid = pop m at in
        hold pid;
        push m (node m at (Persistent pid))
    | 'h' -> memo_get m at (le m at 1)
    | 'j' -> memo_get m at (le m at 4)
    | 'q' -> memo_put m at (le m at 1)
    | 'r' -> memo_put m at (le m at 4)
    | '\x94' -> memo_put m at (Hashtbl.length m.memo)
    | op -> fail m at "the opcode 0x%02x" (Char.code op)
  done;
  { value = Option.get !result; stop = m.pos; nodes = m.nodes; atoms = m.atoms }

let atom (d : decoded) tag s = atom d.atoms tag s
