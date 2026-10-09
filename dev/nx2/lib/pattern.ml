(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type axis = Name of string | Group of string list | Unit | Ellipsis
type layout = axis list

(* An axis of a one-operand pattern, over its names numbered in their order on
   the left: a name or a group, [1], or [...]. *)
type slot = Of of int array | One | Rest

type plan = {
  names : string array;
  left : slot array;
  right : slot array;
  rest : int;  (** The number of names before [...] on the left; [-1]. *)
}

type t = {
  text : string;
  operands : layout list;
  result : layout;
  summed : string list;
  plan : plan option;
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

type token = Word of string | One | Dots | Open | Close | Comma | Arrow | Bar

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
          go (i + 1) (One :: acc)
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
    | One :: ts -> go (Unit :: acc) ts
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

(* No name twice and [...] at most once in [l], called [where]. *)
let distinct ~by s where l =
  let rec go seen = function
    | [] -> ()
    | n :: ns ->
        if List.mem n seen then fail ~by s "%s repeats %s" n where;
        go (n :: seen) ns
  in
  go [] (names l);
  if List.length (List.filter (( = ) Ellipsis) l) > 1 then
    fail ~by s "... appears twice %s" where

let check_one ~by s operand result summed =
  if summed <> [] then fail ~by s "| needs two operands";
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

let check_two ~by s a b result summed =
  let na = names a and nb = names b and nr = names result in
  let rec once seen = function
    | [] -> ()
    | n :: ns ->
        if List.mem n seen then fail ~by s "%s repeats after |" n;
        once (n :: seen) ns
  in
  once [] summed;
  List.iter
    (fun n ->
      if not (List.mem n na && List.mem n nb) then
        fail ~by s "%s is summed but not in both operands" n;
      if List.mem n nr then fail ~by s "%s is summed and in the result" n)
    summed;
  List.iter
    (fun n ->
      if not (List.mem n na || List.mem n nb) then
        fail ~by s "%s is not on the left" n)
    nr;
  List.iter
    (fun n ->
      if not (List.mem n nr || List.mem n summed) then
        if List.mem n na && List.mem n nb then
          fail ~by s
            "%s is in both operands and not in the result; write it after |" n
        else fail ~by s "%s is in one operand and not in the result" n)
    (na @ List.filter (fun n -> not (List.mem n na)) nb);
  let leads l = match l with Ellipsis :: _ -> true | _ -> false in
  let all = [ a; b; result ] in
  if List.exists has_ellipsis all && not (List.for_all leads all) then
    fail ~by s "... leads all three layouts or none"

let index_of names n =
  let i = ref 0 in
  while names.(!i) <> n do
    incr i
  done;
  !i

let plan operand result =
  let names = Array.of_list (names operand) in
  let slot = function
    | Name n -> Of [| index_of names n |]
    | Group ns -> Of (Array.of_list (List.map (index_of names) ns))
    | Unit -> One
    | Ellipsis -> Rest
  in
  let rec before k = function
    | [] -> -1
    | Ellipsis :: _ -> k
    | Name _ :: l -> before (k + 1) l
    | Group ns :: l -> before (k + List.length ns) l
    | Unit :: l -> before k l
  in
  {
    names;
    left = Array.of_list (List.map slot operand);
    right = Array.of_list (List.map slot result);
    rest = before 0 operand;
  }

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
    | [ r; sum ] ->
        let name = function
          | Word n -> n
          | _ -> fail ~by s "names alone follow |"
        in
        (layout ~by s r, List.map name sum)
    | _ -> fail ~by s "| appears twice"
  in
  (match operands with
  | [ a ] ->
      distinct ~by s "on the left" a;
      distinct ~by s "on the right" result;
      check_one ~by s a result summed
  | [ a; b ] ->
      distinct ~by s "in operand 1" a;
      distinct ~by s "in operand 2" b;
      distinct ~by s "in the result" result;
      check_two ~by s a b result summed
  | _ -> fail ~by s "a pattern has one or two operands");
  let plan = match operands with [ a ] -> Some (plan a result) | _ -> None in
  { text = s; operands; result; summed; plan }

(* The pattern a string in NumPy's einsum notation means, each letter a name and
   the names in both operands and not in the result summed; [None] for a string
   not in that notation. *)
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
            match operands with
            | [ a; b ] ->
                List.filter
                  (fun n ->
                    List.mem n (names b) && not (List.mem n (names result)))
                  (names a)
            | _ -> []
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
  | [ a ] ->
      {
        text = text ~operands:[ p.result ] ~result:a ~summed:[];
        operands = [ p.result ];
        result = a;
        summed = [];
        plan = Some (plan p.result a);
      }
  | _ -> invalid_arg (Printf.sprintf "%s: %S has two operands" by p.text)

(* Movements *)

let group_text names ids =
  axis_text (Group (Array.to_list (Array.map (fun i -> names.(i)) ids)))

let same a b = Array.length a = Array.length b && Array.for_all2 Int.equal a b

let moves ~by ~sizes p s what =
  let fail fmt = fail ~by p.text fmt in
  let pl =
    match p.plan with
    | Some pl -> pl
    | None -> fail "has two operands; use Nx.einsum or Nx.contract"
  in
  let names = pl.names in
  let k = Array.length names in
  let ext = Array.make k (-1) in
  List.iter
    (fun (n, e) ->
      if not (Array.mem n names) then fail "%s in ~sizes is not in it" n;
      let i = index_of names n in
      if ext.(i) >= 0 then fail "%s is twice in ~sizes" n;
      if e < 0 then fail "%s = %d in ~sizes is negative" n e;
      ext.(i) <- e)
    sizes;
  let r = Array.length s in
  let named = Array.length pl.left - if pl.rest >= 0 then 1 else 0 in
  let covered = r - named in
  if pl.rest >= 0 then
    begin if covered < 0 then
      fail "names at least %d axes; the operand is %t" named what
    end
  else if covered <> 0 then fail "names %d axes; the operand is %t" named what;
  (* Axes between the reshapes: the names in their order on the left, with the
     axes [...] covers after the [pl.rest] names before it. *)
  let atom i = if pl.rest >= 0 && i >= pl.rest then i + covered else i in
  let split = Array.make (k + max 0 covered) 0 in
  let a = ref 0 in
  Array.iter
    (fun slot ->
      match slot with
      | Rest ->
          Array.blit s !a split pl.rest covered;
          a := !a + covered
      | One ->
          if s.(!a) <> 1 then
            fail "axis %d has extent %d where it has 1" !a s.(!a);
          incr a
      | Of [| i |] ->
          let d = s.(!a) in
          if ext.(i) >= 0 && ext.(i) <> d then
            fail "%s = %d in ~sizes, axis %d has extent %d" names.(i) ext.(i) !a
              d;
          ext.(i) <- d;
          split.(atom i) <- d;
          incr a
      | Of ids ->
          let d = s.(!a) in
          let product = ref 1 and unknown = ref (-1) and unknowns = ref 0 in
          Array.iter
            (fun i ->
              if ext.(i) >= 0 then product := !product * ext.(i)
              else begin
                unknown := i;
                incr unknowns
              end)
            ids;
          if !unknowns = 0 && !product <> d then
            fail "%s multiplies to %d, axis %d has extent %d"
              (group_text names ids) !product !a d;
          if !unknowns = 1 then begin
            if !product = 0 || d mod !product <> 0 then
              fail "%s does not divide axis %d of extent %d"
                (group_text names ids) !a d;
            ext.(!unknown) <- d / !product
          end;
          if !unknowns > 1 then
            fail "%s has two unknown extents; give one in ~sizes"
              (group_text names ids);
          Array.iter (fun i -> split.(atom i) <- ext.(i)) ids;
          incr a)
    pl.left;
  (* The atoms in the result's order, and the result's extents. *)
  let perm = Array.make (Array.length split) 0 in
  let merged =
    Array.make
      (Array.length pl.right + if pl.rest >= 0 then covered - 1 else 0)
      0
  in
  let j = ref 0 and m = ref 0 in
  Array.iter
    (fun slot ->
      match slot with
      | Rest ->
          for c = 0 to covered - 1 do
            perm.(!j) <- pl.rest + c;
            merged.(!m) <- split.(pl.rest + c);
            incr j;
            incr m
          done
      | One ->
          merged.(!m) <- 1;
          incr m
      | Of ids ->
          let product = ref 1 in
          Array.iter
            (fun i ->
              perm.(!j) <- atom i;
              product := !product * ext.(i);
              incr j)
            ids;
          merged.(!m) <- !product;
          incr m)
    pl.right;
  let identity = ref true in
  Array.iteri (fun i p -> if p <> i then identity := false) perm;
  let permuted = Array.map (fun i -> split.(i)) perm in
  let moves =
    if same merged permuted then [] else [ Nx_array.Move.Reshape merged ]
  in
  let moves =
    if !identity then moves else Nx_array.Move.Permute perm :: moves
  in
  if same split s then moves else Nx_array.Move.Reshape split :: moves
