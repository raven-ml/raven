(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type 'n axis = Name of 'n | Group of 'n list | Unit | Ellipsis

type t = {
  text : string;
  names : string array;
  operands : int axis list list;
  result : int axis list;
  plain : bool;
}

let pp ppf p = Format.fprintf ppf "%S" p.text

(* Raises naming [by] and the pattern's text [s]. *)
let fail ~by s fmt =
  Format.kasprintf
    (fun m -> invalid_arg (Printf.sprintf "%s: %S: %s" by s m))
    fmt

(* Text *)

let is_letter c = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
let is_name_char c = is_letter c || (c >= '0' && c <= '9') || c = '_'

let axis_text = function
  | Name n -> n
  | Group ns -> "(" ^ String.concat " " ns ^ ")"
  | Unit -> "1"
  | Ellipsis -> "..."

let layout_text l = String.concat " " (List.map axis_text l)

let text ~operands ~result ~summed =
  String.concat ", " (List.map layout_text operands)
  ^ " -> " ^ layout_text result
  ^ match summed with [] -> "" | ns -> " | " ^ String.concat " " ns

(* Parsing *)

type token =
  | Word of string
  | Unit1
  | Dots
  | Open
  | Close
  | Comma
  | Arrow
  | Bar

let tokens ~by s =
  let n = String.length s in
  let rec go i acc =
    if i >= n then List.rev acc
    else
      match s.[i] with
      | ' ' | '\t' | '\n' | '\r' -> go (i + 1) acc
      | '(' -> go (i + 1) (Open :: acc)
      | ')' -> go (i + 1) (Close :: acc)
      | ',' -> go (i + 1) (Comma :: acc)
      | '|' -> go (i + 1) (Bar :: acc)
      | '-' when i + 1 < n && s.[i + 1] = '>' -> go (i + 2) (Arrow :: acc)
      | '.' when i + 2 < n && s.[i + 1] = '.' && s.[i + 2] = '.' ->
          go (i + 3) (Dots :: acc)
      | '1' when i + 1 >= n || not (is_name_char s.[i + 1]) ->
          go (i + 1) (Unit1 :: acc)
      | c when is_letter c ->
          let j = ref (i + 1) in
          while !j < n && is_name_char s.[!j] do
            incr j
          done;
          go !j (Word (String.sub s i (!j - i)) :: acc)
      | c -> fail ~by s "unexpected %C at %d" c i
  in
  go 0 []

(* [ts] cut at each [sep]. *)
let cut sep ts =
  let rec go cur acc = function
    | [] -> List.rev (List.rev cur :: acc)
    | t :: ts when t = sep -> go [] (List.rev cur :: acc) ts
    | t :: ts -> go (t :: cur) acc ts
  in
  go [] [] ts

let layout ~by s ts =
  let rec group names = function
    | Word n :: ts -> group (n :: names) ts
    | Close :: ts ->
        if names = [] then fail ~by s "an empty group; write 1";
        (Group (List.rev names), ts)
    | [] -> fail ~by s "a group is not closed"
    | _ -> fail ~by s "a group holds names only"
  in
  let rec go acc = function
    | [] -> List.rev acc
    | Word n :: ts -> go (Name n :: acc) ts
    | Unit1 :: ts -> go (Unit :: acc) ts
    | Dots :: ts -> go (Ellipsis :: acc) ts
    | Open :: ts ->
        let g, ts = group [] ts in
        go (g :: acc) ts
    | Close :: _ -> fail ~by s "a group is closed but not opened"
    | (Comma | Arrow | Bar) :: _ -> fail ~by s "a layout holds axes only"
  in
  go [] ts

let names l =
  List.concat_map
    (function Name n -> [ n ] | Group ns -> ns | Unit | Ellipsis -> [])
    l

let has_ellipsis l = List.mem Ellipsis l

(* [ns] without repeats, each name where it first appears. *)
let uniq ns =
  List.rev
    (List.fold_left
       (fun acc n -> if List.mem n acc then acc else n :: acc)
       [] ns)

(* Raises if a name of [ns] repeats [where]. *)
let once ~by s where ns =
  let rec go seen = function
    | [] -> ()
    | n :: ns ->
        if List.mem n seen then fail ~by s "%s repeats %s" n where;
        go (n :: seen) ns
  in
  go [] ns

(* [...] at most once in [l], called [where]. *)
let dots ~by s where l =
  if List.length (List.filter (( = ) Ellipsis) l) > 1 then
    fail ~by s "... appears twice %s" where

(* No name twice and [...] at most once in [l], called [where]. *)
let distinct ~by s where l =
  once ~by s where (names l);
  dots ~by s where l

let repeats l =
  let ns = names l in
  List.length (uniq ns) <> List.length ns

let check_one ~by s operand result =
  distinct ~by s "on the left" operand;
  distinct ~by s "on the right" result;
  let left = names operand and right = names result in
  List.iter
    (fun n ->
      if not (List.mem n left) then fail ~by s "%s is not on the left" n)
    right;
  List.iter
    (fun n ->
      if not (List.mem n right) then fail ~by s "%s is not on the right" n)
    left;
  if has_ellipsis operand <> has_ellipsis result then
    fail ~by s "... is on one side only"

let check_many ~by s operands result summed =
  List.iteri
    (fun i l -> dots ~by s (Printf.sprintf "in operand %d" (i + 1)) l)
    operands;
  distinct ~by s "in the result" result;
  once ~by s "after |" summed;
  let ns = List.map names operands and nr = names result in
  let holders n = List.length (List.filter (List.mem n) ns) in
  List.iter
    (fun n ->
      if holders n = 0 then fail ~by s "%s is summed but in no operand" n;
      if List.mem n nr then fail ~by s "%s is summed and in the result" n)
    summed;
  List.iter
    (fun n -> if holders n = 0 then fail ~by s "%s is not on the left" n)
    nr;
  List.iter
    (fun n ->
      if not (List.mem n nr || List.mem n summed) then
        let k = holders n in
        fail ~by s "%s is in %s and not in the result; write it after |" n
          (if k = 1 then "one operand" else Printf.sprintf "%d operands" k))
    (uniq (List.concat ns));
  let leads l = match l with Ellipsis :: _ -> true | _ -> false in
  let all = result :: operands in
  if List.exists has_ellipsis all && not (List.for_all leads all) then
    fail ~by s "... leads every layout or none"

let index_of names n =
  let i = ref 0 in
  while names.(!i) <> n do
    incr i
  done;
  !i

(* [operands] and [result] with each name numbered by its first place among
   the operands. *)
let number operands result =
  let names = Array.of_list (uniq (List.concat_map names operands)) in
  let number = function
    | Name n -> Name (index_of names n)
    | Group g -> Group (List.map (index_of names) g)
    | Unit -> Unit
    | Ellipsis -> Ellipsis
  in
  let numbered l = List.map number l in
  (names, List.map numbered operands, numbered result)

let words ~by s =
  let ts = tokens ~by s in
  let left, right =
    match cut Arrow ts with
    | [ l; r ] -> (l, r)
    | _ -> fail ~by s "a pattern has one ->"
  in
  let operands = List.map (layout ~by s) (cut Comma left) in
  let result, summed =
    match cut Bar right with
    | [ r ] -> (layout ~by s r, [])
    | [ _; [] ] -> fail ~by s "| needs a name"
    | [ r; sum ] ->
        let name = function
          | Word n -> n
          | _ -> fail ~by s "names alone follow |"
        in
        (layout ~by s r, List.map name sum)
    | _ -> fail ~by s "| appears twice"
  in
  (match operands with
  | [ operand ] when summed = [] && not (repeats operand) ->
      check_one ~by s operand result
  | operands -> check_many ~by s operands result summed);
  (* No diagonal, and each summed name in two operands or more. *)
  let holders n =
    List.length (List.filter (fun l -> List.mem n (names l)) operands)
  in
  let plain =
    (not (List.exists repeats operands))
    && List.for_all (fun n -> holders n > 1) summed
  in
  let names, operands, result = number operands result in
  { text = s; names; operands; result; plain }

(* The pattern a string in NumPy's einsum notation means, each letter a name and
   the names not in the result summed; [None] for a string not in that
   notation. *)
let numpy s =
  let letters l =
    let l = String.trim l in
    let rec go i acc =
      if i >= String.length l then Some (List.rev acc)
      else if is_letter l.[i] then go (i + 1) (Name (String.make 1 l.[i]) :: acc)
      else if i + 2 < String.length l && String.sub l i 3 = "..." then
        go (i + 3) (Ellipsis :: acc)
      else None
    in
    go 0 []
  in
  let layouts ls =
    List.fold_right
      (fun l acc ->
        match (letters l, acc) with
        | Some l, Some acc -> Some (l :: acc)
        | _ -> None)
      ls (Some [])
  in
  match String.split_on_char '>' s with
  | [ l; r ] when String.ends_with ~suffix:"-" l -> (
      let l = String.sub l 0 (String.length l - 1) in
      match (layouts (String.split_on_char ',' l), letters r) with
      | Some operands, Some result
        when List.exists
               (fun l -> List.length (names l) > 1)
               (result :: operands) ->
          let summed =
            List.filter
              (fun n -> not (List.mem n (names result)))
              (uniq (List.concat_map names operands))
          in
          Some (text ~operands ~result ~summed)
      | _ -> None)
  | _ -> None

let parses s =
  match words ~by:"" s with _ -> true | exception Invalid_argument _ -> false

let v ~by s =
  match words ~by s with
  | p -> p
  | exception (Invalid_argument _ as e) -> (
      match numpy s with
      | Some s' when parses s' ->
          invalid_arg
            (Printf.sprintf "%s: %S is in NumPy's einsum notation; write %S" by
               s s')
      | _ -> raise e)

let inverse ~by p =
  match p.operands with
  | [ _ ] when not p.plain ->
      invalid_arg
        (Printf.sprintf "%s: %S takes a diagonal or sums within one operand"
           by p.text)
  | [ operand ] ->
      let named l =
        List.map
          (function
            | Name i -> Name p.names.(i)
            | Group is -> Group (List.map (Array.get p.names) is)
            | Unit -> Unit
            | Ellipsis -> Ellipsis)
          l
      in
      let left = named p.result and right = named operand in
      let names, operands, result = number [ left ] right in
      {
        text = text ~operands:[ left ] ~result:right ~summed:[];
        names;
        operands;
        result;
        plain = true;
      }
  | operands ->
      invalid_arg
        (Printf.sprintf "%s: %S has %d operands" by p.text
           (List.length operands))

(* Extents *)

(* A pattern met with operands: [by] and [label] for messages, the number of
   axes [...] covers, and each name's extent, [-1] where unknown, then those
   of the covered axes. [where] holds the operand each was learned from, or
   [-1] for [~sizes]. A call allocates this and walks the layouts with
   first-order functions, so that [rearrange] costs about its movements. *)
type state = {
  by : string;
  p : t;
  label : int -> string;
  covered : int;
  ext : int array;
  where : int array;
}

let shape_text s =
  "[" ^ String.concat "; " (Array.to_list (Array.map string_of_int s)) ^ "]"

let axes k = if k = 1 then "1 axis" else string_of_int k ^ " axes"

(* The number of axes [l] names, besides its ellipsis. *)
let written l =
  List.fold_left (fun w x -> match x with Ellipsis -> w | _ -> w + 1) 0 l

(* Name [i] of [p], or [...k] for the [k]th axis [...] covers. *)
let name p i =
  let n = Array.length p.names in
  if i < n then p.names.(i) else "..." ^ string_of_int (i - n)

let group_text p is = axis_text (Group (List.map (name p) is))
let source st k = if k < 0 then "~sizes" else st.label k

let rec has_zero ext = function
  | [] -> false
  | i :: is -> ext.(i) = 0 || has_zero ext is

(* The product of the known extents of [is], none [0], times [q]: [-1] past an
   int. *)
let rec known ext q = function
  | [] -> q
  | i :: is ->
      let d = ext.(i) in
      if d < 0 || q < 0 then known ext q is
      else if d > max_int / q then known ext (-1) is
      else known ext (q * d) is

(* The product of the known extents of [is]: [0] if one is, [-1] past an
   int. *)
let product ext is = if has_zero ext is then 0 else known ext 1 is

let rec unknowns ext = function
  | [] -> 0
  | i :: is -> (if ext.(i) < 0 then 1 else 0) + unknowns ext is

let rec unknown ext = function
  | [] -> assert false
  | i :: is -> if ext.(i) < 0 then i else unknown ext is

(* The place of the name [s] in [names], or [-1]. *)
let find names s =
  let rec go i =
    if i = Array.length names then -1
    else if String.equal names.(i) s then i
    else go (i + 1)
  in
  go 0

let learn st i d k =
  if st.ext.(i) < 0 then begin
    st.ext.(i) <- d;
    st.where.(i) <- k
  end
  else if st.ext.(i) <> d then
    fail ~by:st.by st.p.text "%s is %d in %s and %d in %s" (name st.p i)
      st.ext.(i) (source st st.where.(i)) d (source st k)

(* Raises unless each operand's rank fits its layout. *)
let rec ranks ~by ~label p k ls xs =
  match (ls, xs) with
  | l :: ls, (dt, s) :: xs ->
      let w = written l and r = Array.length s in
      let dots = has_ellipsis l in
      if (dots && r < w) || ((not dots) && r <> w) then
        invalid_arg
          (Printf.sprintf "%s: %S names %s%s for %s; %s is %s %s" by p.text
             (if dots then "at least " else "")
             (axes w) (label k) (label k) dt (shape_text s));
      ranks ~by ~label p (k + 1) ls xs
  | _ -> ()

(* The number of axes [...] covers in the [k]th operand on, the same as [c]
   in the first. *)
let rec covered ~by ~label p k c ls xs =
  match (ls, xs) with
  | l :: ls, (_, s) :: xs ->
      let c' = Array.length s - written l in
      if k > 0 && c' <> c then
        fail ~by p.text "... is %s in %s and %d in %s" (axes c) (label 0) c'
          (label k);
      covered ~by ~label p (k + 1) c' ls xs
  | _ -> c

let rec given st = function
  | [] -> ()
  | (s, d) :: sizes ->
      let i = find st.p.names s and text = st.p.text in
      if i < 0 then fail ~by:st.by text "~sizes names %s, which it lacks" s;
      if st.ext.(i) >= 0 then fail ~by:st.by text "%s is twice in ~sizes" s;
      if d < 0 then fail ~by:st.by text "%s = %d in ~sizes is negative" s d;
      learn st i d (-1);
      given st sizes

(* Learns the extents that the [k]th operand's layout [l] gives its names
   alone and its covered axes, from its axis [a] of shape [s] on. *)
let rec scan st k s a = function
  | [] -> ()
  | Name i :: l ->
      learn st i s.(a) k;
      scan st k s (a + 1) l
  | Unit :: l ->
      if s.(a) <> 1 then
        fail ~by:st.by st.p.text
          "axis %d of %s is 1 in the pattern and %d in %s" a (st.label k) s.(a)
          (st.label k);
      scan st k s (a + 1) l
  | Group _ :: l -> scan st k s (a + 1) l
  | Ellipsis :: l ->
      let n = Array.length st.p.names in
      for j = 0 to st.covered - 1 do
        learn st (n + j) s.(a + j) k
      done;
      scan st k s (a + st.covered) l

let rec scans st k ls xs =
  match (ls, xs) with
  | l :: ls, (_, s) :: xs ->
      scan st k s 0 l;
      scans st (k + 1) ls xs
  | _ -> ()

let multiplies st k is d q =
  fail ~by:st.by st.p.text "%s is %d in %s, and its names multiply to %s"
    (group_text st.p is) d (st.label k) q

(* Solves the group [is] on an axis of extent [d] of the [k]th operand: a
   group whose known extents multiply to 0 has no element and learns
   nothing; otherwise they divide [d] exactly and leave the one unknown name
   the quotient. [true] if it learned an extent. *)
let solve st k is d =
  let q = product st.ext is and u = unknowns st.ext is in
  if q = 0 then begin
    if d <> 0 then multiplies st k is d "0";
    false
  end
  else if u = 0 then begin
    if q < 0 then multiplies st k is d "more than an int counts";
    if q <> d then multiplies st k is d (string_of_int q);
    false
  end
  else if u > 1 then false
  else if q < 0 && d = 0 then begin
    learn st (unknown st.ext is) 0 k;
    true
  end
  else if q > 0 && d mod q = 0 then begin
    learn st (unknown st.ext is) (d / q) k;
    true
  end
  else
    fail ~by:st.by st.p.text "%s is %d in %s, which %s does not divide"
      (group_text st.p is) d (st.label k)
      (if q < 0 then "a product past an int" else string_of_int q)

(* One pass over the groups of the [k]th operand's layout [l] on, from its
   axis [a] of shape [s]; [true] if it or [learned] learned an extent. *)
let rec groups st k s a learned = function
  | [] -> learned
  | Group is :: l ->
      let learned = solve st k is s.(a) || learned in
      groups st k s (a + 1) learned l
  | (Name _ | Unit) :: l -> groups st k s (a + 1) learned l
  | Ellipsis :: l -> groups st k s (a + st.covered) learned l

let rec pass st k learned ls xs =
  match (ls, xs) with
  | l :: ls, (_, s) :: xs -> pass st (k + 1) (groups st k s 0 learned l) ls xs
  | _ -> learned

(* Raises if a group of the [k]th operand's layout on leaves a name unknown. *)
let rec solved st k = function
  | [] -> ()
  | l :: ls ->
      List.iter
        (function
          | Group is when unknowns st.ext is > 0 ->
              let us = List.filter (fun i -> st.ext.(i) < 0) is in
              fail ~by:st.by st.p.text
                "%s in %s leaves %s unknown; give one in ~sizes"
                (group_text st.p is) (st.label k)
                (String.concat " and " (List.map (name st.p) us))
          | Name _ | Group _ | Unit | Ellipsis -> ())
        l;
      solved st (k + 1) ls

let rec count c = function
  | [] -> 0
  | Name _ :: l -> 1 + count c l
  | Group is :: l -> List.length is + count c l
  | Unit :: l -> count c l
  | Ellipsis :: l -> c + count c l

let rec put a j = function
  | [] -> j
  | i :: is ->
      a.(j) <- i;
      put a (j + 1) is

(* Writes [l]'s names into [a] from [j] on, in order, those in groups
   included, its ellipsis written out as the covered axes. *)
let rec names_into st a j = function
  | [] -> a
  | Name i :: l ->
      a.(j) <- i;
      names_into st a (j + 1) l
  | Group is :: l -> names_into st a (put a j is) l
  | Unit :: l -> names_into st a j l
  | Ellipsis :: l ->
      let n = Array.length st.p.names in
      for e = 0 to st.covered - 1 do
        a.(j + e) <- n + e
      done;
      names_into st a (j + st.covered) l

let named st l = names_into st (Array.make (count st.covered l) 0) 0 l

(* Writes the extents of [l]'s axes into [s] from [j] on. *)
let rec extents_into st s j = function
  | [] -> s
  | Name i :: l ->
      s.(j) <- st.ext.(i);
      extents_into st s (j + 1) l
  | Group is :: l ->
      let q = product st.ext is in
      if q < 0 then
        fail ~by:st.by st.p.text "%s in the result multiplies past an int"
          (group_text st.p is);
      s.(j) <- q;
      extents_into st s (j + 1) l
  | Unit :: l ->
      s.(j) <- 1;
      extents_into st s (j + 1) l
  | Ellipsis :: l ->
      let n = Array.length st.p.names in
      for e = 0 to st.covered - 1 do
        s.(j + e) <- st.ext.(n + e)
      done;
      extents_into st s (j + st.covered) l

let rec all_named st = function
  | [] -> []
  | l :: ls -> named st l :: all_named st ls

let resolve ~by ~sizes ~label p xs =
  ranks ~by ~label p 0 p.operands xs;
  let covered = covered ~by ~label p 0 0 p.operands xs in
  let m = Array.length p.names + covered in
  let st =
    {
      by;
      p;
      label;
      covered;
      ext = Array.make m (-1);
      where = Array.make m (-1);
    }
  in
  given st sizes;
  scans st 0 p.operands xs;
  while pass st 0 false p.operands xs do
    ()
  done;
  solved st 0 p.operands;
  ( ~extents:st.ext,
    ~operands:(all_named st p.operands),
    ~result:(named st p.result),
    ~shape:
      (extents_into st
         (Array.make (written p.result + covered) 0)
         0 p.result) )

(* Movements *)

let is_identity perm =
  let rec go i = i = Array.length perm || (perm.(i) = i && go (i + 1)) in
  go 0

(* The place of [i] in [a]. *)
let position (a : int array) i =
  let rec go j = if a.(j) = i then j else go (j + 1) in
  go 0

(* Whether [s] is [split] permuted by [perm]. *)
let permutes (s : int array) split perm =
  let rec go j =
    j = Array.length s || (s.(j) = split.(perm.(j)) && go (j + 1))
  in
  Array.length s = Array.length perm && go 0

(* The operand split into its names, permuted into the result's, and merged
   into the result's shape. *)
let moves ~by ~sizes p x =
  match p.operands with
  | [ _ ] when not p.plain ->
      fail ~by p.text
        "takes a diagonal or sums within one operand; use Nx.einsum"
  | [ _ ] ->
      let label _ = "x" in
      let ~extents, ~operands, ~result, ~shape =
        resolve ~by ~sizes ~label p [ x ]
      in
      let left = List.hd operands in
      let split = Array.make (Array.length left) 0 in
      for j = 0 to Array.length left - 1 do
        split.(j) <- extents.(left.(j))
      done;
      let perm = Array.make (Array.length result) 0 in
      for j = 0 to Array.length result - 1 do
        perm.(j) <- position left result.(j)
      done;
      let moves =
        if permutes shape split perm then []
        else [ Nx_array.Move.Reshape (Iarray.of_array shape) ]
      in
      let moves =
        if is_identity perm then moves
        else Nx_array.Move.Permute (Iarray.of_array perm) :: moves
      in
      if split = snd x then moves
      else Nx_array.Move.Reshape (Iarray.of_array split) :: moves
  | operands ->
      fail ~by p.text "has %d operands; use Nx.einsum" (List.length operands)
