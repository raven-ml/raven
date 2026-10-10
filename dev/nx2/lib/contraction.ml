(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A two-operand pattern lowers to one kernel contraction between movements:

   a ──reshape──▶ a's names ─┐ ├─ Contract ─▶ batch, a's free, b's free b
   ──reshape──▶ b's names ─┘ │ permute, reshape ▼ the written result

   Each operand is reshaped so that every name is one axis, its groups split and
   its unit axes dropped. The batch and summed names pair in [a]'s order, the
   order in which the kernel sums. [init] takes the inverse movements into the
   kernel's order. *)

module D = Nx_array.Dtype
module M = Nx_array.Move
module S = Nx_kernel.Spec

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

let pp_shape ppf s =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))

(* Plans *)

type plan = {
  a : int array; (* [a] with each name an axis. *)
  b : int array;
  batch : (int * int) array;
  contracting : (int * int) array;
  order : int array; (* The kernel's axes in the result's names' order. *)
  names : int array; (* The result's names' extents, in written order. *)
  shape : int array; (* The result's shape. *)
}

(* [l] with its ellipsis replaced by [n] names that no pattern writes. *)
let expand l n =
  List.concat_map
    (function
      | Pattern.Ellipsis ->
          List.init n (fun i -> Pattern.Name ("..." ^ string_of_int i))
      | x -> [ x ])
    l

(* [l]'s names in order, those in groups included. *)
let names l =
  List.concat_map
    (function
      | Pattern.Name n -> [ n ] | Group ns -> ns | Unit | Ellipsis -> [])
    l

let written l = List.length (List.filter (( <> ) Pattern.Ellipsis) l)

let index_of n l =
  let rec go i = function
    | [] -> raise Not_found
    | x :: rest -> if String.equal x n then i else go (i + 1) rest
  in
  go 0 l

(* Raises naming [by] and the pattern [p]. *)
let failf ~by p fmt =
  Format.kasprintf (fun s -> invalid_argf "%s: %a: %s" by Pattern.pp p s) fmt

(* Each name's extent, from [sizes], the operands' axes and the division of
   their groups. *)
let extents ~by ~pat ~sizes layouts =
  let known = Hashtbl.create 16 in
  let learn n d where =
    match Hashtbl.find_opt known n with
    | Some (d', where') when d' <> d ->
        failf ~by pat "%s is %d in %s and %d in %s" n d' where' d where
    | Some _ -> ()
    | None -> Hashtbl.replace known n (d, where)
  in
  List.iter (fun (n, d) -> learn n d "sizes") sizes;
  List.iter
    (fun (label, l, s) ->
      List.iteri
        (fun i x ->
          match x with
          | Pattern.Name n -> learn n s.(i) label
          | Unit ->
              if s.(i) <> 1 then
                failf ~by pat "axis %d of %s is 1 in the pattern and %d in %s" i
                  label s.(i) label
          | Group _ | Ellipsis -> ())
        l)
    layouts;
  let groups =
    List.concat_map
      (fun (label, l, s) ->
        List.concat
          (List.mapi
             (fun i -> function
               | Pattern.Group ns -> [ (label, ns, s.(i)) ] | _ -> [])
             l))
      layouts
  in
  let ext n = Option.map fst (Hashtbl.find_opt known n) in
  let product ns = List.fold_left (fun p n -> p * Option.get (ext n)) 1 ns in
  let render ns = "(" ^ String.concat " " ns ^ ")" in
  (* One pass over the groups; [true] if it learned an extent. *)
  let pass () =
    List.fold_left
      (fun progress (label, ns, d) ->
        match List.filter (fun n -> ext n = None) ns with
        | [] ->
            let p = product ns in
            if p <> d then
              failf ~by pat "%s is %d in %s, and its names multiply to %d"
                (render ns) d label p;
            progress
        | [ u ] ->
            let p = product (List.filter (( <> ) u) ns) in
            if p = 0 then progress
            else if d mod p <> 0 then
              failf ~by pat "%s is %d in %s, which %d does not divide"
                (render ns) d label p
            else (
              learn u (d / p) label;
              true)
        | _ -> progress)
      false groups
  in
  while pass () do
    ()
  done;
  List.iter
    (fun (label, ns, _) ->
      match List.filter (fun n -> ext n = None) ns with
      | [] -> ()
      | us ->
          failf ~by pat "%s in %s leaves %s unknown; give one in ~sizes"
            (render ns) label (String.concat " and " us))
    groups;
  fun n -> Option.get (ext n)

(* The contraction of operands named [na] and [nb] into the names [nr], of the
   result [shape], summing [summed]: the batch and summed names pair in [a]'s
   order. *)
let lower ~ext ~summed na nb nr shape =
  let shape_of ns = Array.of_list (List.map ext ns) in
  let pairs ns =
    Array.of_list (List.map (fun n -> (index_of n na, index_of n nb)) ns)
  in
  let batch = List.filter (fun n -> List.mem n nb && List.mem n nr) na in
  let summed = List.filter (fun n -> List.mem n summed) na in
  let free_a = List.filter (fun n -> not (List.mem n nb)) na in
  let free_b = List.filter (fun n -> not (List.mem n na)) nb in
  let kernel = batch @ free_a @ free_b in
  {
    a = shape_of na;
    b = shape_of nb;
    batch = pairs batch;
    contracting = pairs summed;
    order = Array.of_list (List.map (fun n -> index_of n kernel) nr);
    names = shape_of nr;
    shape;
  }

let plan ~by ~sizes (p : Pattern.t) (da, sa) (db, sb) =
  let err fmt = failf ~by p fmt in
  let la, lb, summed =
    match p.kind with
    | Two { a; b; summed } -> (a, b, summed)
    | One _ -> invalid_argf "%s: %a has one operand" by Pattern.pp p
  in
  let ellipsis label dt l s =
    let w = written l and r = Array.length s in
    let dots = List.mem Pattern.Ellipsis l in
    if (dots && r < w) || ((not dots) && r <> w) then
      invalid_argf "%s: %a names %d axes for %s; %s is %s %a" by Pattern.pp p w
        label label dt pp_shape s
    else r - w
  in
  let na = ellipsis "a" da la sa and nb = ellipsis "b" db lb sb in
  if na <> nb then err "... is %d axes in a and %d in b" na nb;
  let la = expand la na and lb = expand lb na and lr = expand p.result na in
  let all = names la @ names lb in
  List.iter
    (fun (n, _) ->
      if not (List.mem n all) then err "~sizes names %s, which it lacks" n)
    sizes;
  let ext = extents ~by ~pat:p ~sizes [ ("a", la, sa); ("b", lb, sb) ] in
  let entry = function
    | Pattern.Name n -> ext n
    | Group ns -> List.fold_left (fun p n -> p * ext n) 1 ns
    | Unit | Ellipsis -> 1
  in
  lower ~ext ~summed (names la) (names lb) (names lr)
    (Array.of_list (List.map entry lr))

(* Accumulators *)

type kind = Floats | Complexes | Integers

let kind_of (type v s) (dt : (v, s) D.t) =
  match D.kind dt with
  | Float -> Floats
  | Complex -> Complexes
  | Signed | Unsigned | Boolean -> Integers

(* The width of [dt]'s real numbers: a complex dtype's parts. *)
let width (type v s) (dt : (v, s) D.t) =
  match D.kind dt with Complex -> D.bits dt / 2 | _ -> D.bits dt

let accumulator (type v s) ~by ?acc (dt : (v, s) D.t) (D.Any a) (D.Any b) =
  let ka = kind_of a and kb = kind_of b in
  let floats =
    List.filter (fun (k, _) -> k <> Integers) [ (ka, width a); (kb, width b) ]
  in
  let wide = List.fold_left (fun w (_, x) -> max w x) 0 floats in
  let complex = List.exists (fun (k, _) -> k = Complexes) floats in
  let (D.Any acc) =
    match acc with
    | Some acc -> acc
    | None -> (
        match (ka, kb) with
        | Integers, Integers -> D.Any dt
        | Integers, _ | _, Integers ->
            invalid_argf "%s: %s and %s operands need ~acc" by (D.name a)
              (D.name b)
        | _ when complex ->
            if wide <= 32 then D.Any D.Complex64 else D.Any D.Complex128
        | _ -> if wide <= 32 then D.Any D.Float32 else D.Any D.Float64)
  in
  let name = D.name acc in
  (match D.kind acc with
  | Boolean -> invalid_argf "%s: ~acc is %s" by name
  | Float | Complex ->
      if width acc < 32 then
        invalid_argf "%s: ~acc %s is narrower than float32" by name;
      if width acc < wide then
        invalid_argf "%s: ~acc %s is narrower than an operand" by name
  | Signed | Unsigned -> ());
  if kind_of acc <> kind_of dt then
    invalid_argf "%s: ~acc %s is of another kind than %s" by name (D.name dt);
  D.Any acc

(* Evaluation *)

let move ~by mv x = Eval.eval ~by (Value.Move (mv, x))
let reshape ~by s x = if Prim.has_shape x s then x else move ~by (M.Reshape s) x

let identity order =
  let ok = ref true in
  Array.iteri (fun i j -> if i <> j then ok := false) order;
  !ok

let inverse order =
  let inv = Array.make (Array.length order) 0 in
  Array.iteri (fun i j -> inv.(j) <- i) order;
  inv

let operand x = (D.name (Prim.dtype x), Prim.shape x)

(* [pl] applied to [a] and [b]: each reshaped to its names, contracted, and
   moved into the result's layout. *)
let run (type v s a b c e d) ~by ?acc ?init (dt : (v, s) D.t) pl
    (a : (a, b, d) Value.t) (b : (c, e, d) Value.t) : (v, s, d) Value.t =
  if D.is D.Boolean dt then invalid_argf "%s: the result's dtype is bool" by;
  let acc =
    accumulator ~by ?acc dt (D.Any (Prim.dtype a)) (D.Any (Prim.dtype b))
  in
  let init =
    Option.map
      (fun i ->
        if not (Prim.has_shape i pl.shape) then
          invalid_argf "%s: ~init is %a where the result is %a" by pp_shape
            (Prim.shape i) pp_shape pl.shape;
        let i = reshape ~by pl.names i in
        if identity pl.order then i
        else move ~by (M.Permute (inverse pl.order)) i)
      init
  in
  let spec =
    match
      S.contract ~batch:pl.batch ~contracting:pl.contracting ~acc
        ~out:(D.Any dt) ~init:(Option.is_some init)
    with
    | s -> s
    | exception Invalid_argument e -> invalid_argf "%s: %s" by e
  in
  let y =
    Eval.contract ~by spec dt (reshape ~by pl.a a) (reshape ~by pl.b b) init
  in
  let y = if identity pl.order then y else move ~by (M.Permute pl.order) y in
  reshape ~by pl.shape y

let contract ~by ?(sizes = []) ?acc ?init dt p a b =
  run ~by ?acc ?init dt (plan ~by ~sizes p (operand a) (operand b)) a b

(* The leading axes, aligned at their last, name batch axes where both operands
   have the extent, and free axes of the operand whose extent is not 1 where the
   other's is: an operand's unit axis drops, and no operand is broadcast. A 1-d
   [a] has no row and a 1-d [b] no column. *)
let matmul ~by a b =
  let sa = Prim.shape a and sb = Prim.shape b in
  let ra = Array.length sa and rb = Array.length sb in
  let both () =
    Format.asprintf "%s %a and %s %a"
      (D.name (Prim.dtype a))
      pp_shape sa
      (D.name (Prim.dtype b))
      pp_shape sb
  in
  if ra = 0 || rb = 0 then invalid_argf "%s: an operand is 0-d; %s" by (both ());
  let k = sa.(ra - 1) and k' = if rb = 1 then sb.(0) else sb.(rb - 2) in
  if k <> k' then
    invalid_argf "%s: inner extents %d and %d differ; %s" by k k' (both ());
  let lead_a = if ra = 1 then [||] else Array.sub sa 0 (ra - 2) in
  let lead_b = if rb = 1 then [||] else Array.sub sb 0 (rb - 2) in
  let l = max (Array.length lead_a) (Array.length lead_b) in
  let at s i =
    let j = i - (l - Array.length s) in
    if j < 0 then None else Some s.(j)
  in
  let extents = Hashtbl.create 8 in
  let name i e =
    let n = "l" ^ string_of_int i in
    Hashtbl.replace extents n e;
    [ n ]
  in
  let na = ref [] and nb = ref [] and nr = ref [] and shape = ref [] in
  for i = l - 1 downto 0 do
    let ea = at lead_a i and eb = at lead_b i in
    let ext = max (Option.value ~default:1 ea) (Option.value ~default:1 eb) in
    let named = name i ext in
    (match (ea, eb) with
    | Some x, Some y when x = y ->
        na := named @ !na;
        nb := named @ !nb
    | Some x, (Some 1 | None) when x = ext -> na := named @ !na
    | (Some 1 | None), Some y when y = ext -> nb := named @ !nb
    | _ -> invalid_argf "%s: the leading axes do not broadcast; %s" by (both ()));
    nr := named @ !nr;
    shape := ext :: !shape
  done;
  let rows = if ra = 1 then [] else [ ("i", sa.(ra - 2)) ] in
  let cols = if rb = 1 then [] else [ ("j", sb.(rb - 1)) ] in
  List.iter
    (fun (n, e) -> Hashtbl.replace extents n e)
    ((("k", k) :: rows) @ cols);
  let ns = List.map fst in
  let pl =
    lower ~ext:(Hashtbl.find extents) ~summed:[ "k" ]
      (!na @ ns rows @ [ "k" ])
      (!nb @ [ "k" ] @ ns cols)
      (!nr @ ns rows @ ns cols)
      (Array.of_list (!shape @ List.map snd rows @ List.map snd cols))
  in
  run ~by (Prim.dtype a) pl a b
