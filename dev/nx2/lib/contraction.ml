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

let invalid_argf fmt = Format.kasprintf invalid_arg fmt

let pp_shape ppf s =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))

(* Plans *)

(* Below a pattern, names are numbers: operands and results list them, and
   [ext.(n)] is name [n]'s extent. *)

type plan = {
  a : int array; (* [a] with each name an axis. *)
  b : int array;
  batch : (int * int) array;
  contracting : (int * int) array;
  order : int array; (* The kernel's axes in the result's names' order. *)
  names : int array; (* The result's names' extents, in written order. *)
  shape : int array; (* The result's shape. *)
}

let rec mem (n : int) = function [] -> false | x :: l -> x = n || mem n l

let index_of (n : int) l =
  let rec go i = function
    | [] -> raise Not_found
    | x :: rest -> if x = n then i else go (i + 1) rest
  in
  go 0 l

(* The contraction of operands named [na] and [nb] into the names [nr], of the
   result [shape]. A name both operands hold is a batch name where the result
   keeps it, else summed; the batch and summed names pair in [a]'s order, the
   order in which the kernel sums. *)
let lower ~ext na nb nr shape =
  let shape_of ns = Array.of_list (List.map (Array.get ext) ns) in
  let pairs ns =
    Array.of_list (List.map (fun n -> (index_of n na, index_of n nb)) ns)
  in
  let both = List.filter (fun n -> mem n nb) na in
  let batch = List.filter (fun n -> mem n nr) both in
  let summed = List.filter (fun n -> not (mem n nr)) both in
  let free_a = List.filter (fun n -> not (mem n nb)) na in
  let free_b = List.filter (fun n -> not (mem n na)) nb in
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

(* Patterns *)

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

(* Raises naming [by] and the pattern [p]. *)
let failf ~by p fmt =
  Format.kasprintf (fun s -> invalid_argf "%s: %a: %s" by Pattern.pp p s) fmt

let render ns = "(" ^ String.concat " " ns ^ ")"
let axes k = if k = 1 then "1 axis" else string_of_int k ^ " axes"

(* The product of the extents [ds], none negative: [0] if one is, else
   [None] past an int. *)
let product ds =
  if List.mem 0 ds then Some 0
  else
    List.fold_left
      (fun p d ->
        Option.bind p (fun p -> if d > max_int / p then None else Some (p * d)))
      (Some 1) ds

(* Each name's extent, from [sizes], the operands' axes and the division of
   their groups. A group with a known extent of 0 has no element, whatever
   its other names, which stay unknown; otherwise its known extents divide
   its axis exactly, their product checked, and leave the last unknown name
   the quotient. *)
let extents ~by ~pat ~sizes layouts =
  let known = Hashtbl.create 16 in
  let learn n d where =
    match Hashtbl.find_opt known n with
    | Some (d', where') when d' <> d ->
        failf ~by pat "%s is %d in %s and %d in %s" n d' where' d where
    | Some _ -> ()
    | None -> Hashtbl.replace known n (d, where)
  in
  List.iter
    (fun (n, d) ->
      if d < 0 then failf ~by pat "%s = %d in ~sizes is negative" n d;
      learn n d "sizes")
    sizes;
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
  (* One pass over the groups; [true] if it learned an extent. *)
  let pass () =
    List.fold_left
      (fun progress (label, ns, d) ->
        let found = List.filter_map ext ns in
        let multiply p =
          failf ~by pat "%s is %d in %s, and its names multiply to %s"
            (render ns) d label p
        in
        match (List.filter (fun n -> ext n = None) ns, product found) with
        | _, Some 0 ->
            if d <> 0 then multiply "0";
            progress
        | [], Some p ->
            if p <> d then multiply (string_of_int p);
            progress
        | [], None -> multiply "more than an int counts"
        | [ u ], None when d = 0 ->
            learn u 0 label;
            true
        | [ u ], Some p when d mod p = 0 ->
            learn u (d / p) label;
            true
        | [ _ ], p ->
            let p =
              Option.fold ~none:"a product past an int" ~some:string_of_int p
            in
            failf ~by pat "%s is %d in %s, which %s does not divide"
              (render ns) d label p
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

(* The operands' names, numbered, with ellipses written out, each name's
   extent, and the result's names and shape: [p]'s operand layouts [ls] met
   with the operands [xs], each a label for messages, a dtype's name and a
   shape. A name's number is its first place among the operands' names. *)
let prepare ~by ~sizes (p : Pattern.t) ls xs =
  let err fmt = failf ~by p fmt in
  let covered l (label, dt, s) =
    let w = written l and r = Array.length s in
    let dots = List.mem Pattern.Ellipsis l in
    if (dots && r < w) || ((not dots) && r <> w) then
      invalid_argf "%s: %a names %s for %s; %s is %s %a" by Pattern.pp p
        (axes w) label label dt pp_shape s
    else r - w
  in
  let cs = List.map2 covered ls xs in
  let c = List.hd cs and first, _, _ = List.hd xs in
  List.iter2
    (fun c' (label, _, _) ->
      if c' <> c then err "... is %s in %s and %d in %s" (axes c) first c' label)
    cs xs;
  let ls = List.map (fun l -> expand l c) ls and lr = expand p.result c in
  let all = List.concat_map names ls in
  List.iter
    (fun (n, _) ->
      if not (List.mem n all) then err "~sizes names %s, which it lacks" n)
    sizes;
  let ext =
    extents ~by ~pat:p ~sizes
      (List.map2 (fun l (label, _, s) -> (label, l, s)) ls xs)
  in
  let entry = function
    | Pattern.Name n -> ext n
    | Group ns -> (
        match product (List.map ext ns) with
        | Some p -> p
        | None -> err "%s in the result multiplies past an int" (render ns))
    | Unit | Ellipsis -> 1
  in
  let number n = Option.get (List.find_index (String.equal n) all) in
  let numbered l = List.map number (names l) in
  ( List.map numbered ls,
    Array.of_list (List.map ext all),
    numbered lr,
    Array.of_list (List.map entry lr) )

let plan ~by ~sizes (p : Pattern.t) (da, sa) (db, sb) =
  let la, lb =
    match p.kind with
    | Many { operands = [ a; b ]; _ } -> (a, b)
    | Many { operands; _ } ->
        invalid_argf "%s: %a has %d operands; use Nx.einsum" by Pattern.pp p
          (List.length operands)
    | One _ ->
        invalid_argf "%s: %a has one operand; use Nx.rearrange" by Pattern.pp p
  in
  let ns, ext, nr, shape =
    prepare ~by ~sizes p [ la; lb ] [ ("a", da, sa); ("b", db, sb) ]
  in
  lower ~ext (List.nth ns 0) (List.nth ns 1) nr shape

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
  (* The widest float or complex operand's width, and whether one is
     complex. *)
  let float_width k w = match k with Integers -> 0 | Floats | Complexes -> w in
  let wide = Int.max (float_width ka (width a)) (float_width kb (width b)) in
  let complex = ka = Complexes || kb = Complexes in
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
  if not (Nx_kernel.accumulates acc) then
    invalid_argf
      "%s: ~acc %s cannot accumulate (no bool, no float narrower than float32)"
      by name;
  if kind_of acc <> Integers && width acc < wide then
    invalid_argf "%s: ~acc %s is narrower than an operand" by name;
  if kind_of acc <> kind_of dt then
    invalid_argf "%s: ~acc %s is of another kind than %s" by name (D.name dt);
  D.Any acc

(* Evaluation *)

let move ~by mv x = Eval.eval ~by (Value.Move (mv, x))
let reshape ~by s x = if Prim.has_shape x s then x else move ~by (M.Reshape s) x

let identity order =
  let rec go i = i = Array.length order || (order.(i) = i && go (i + 1)) in
  go 0

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
  let y =
    Eval.contract ~by ~batch:pl.batch ~contracting:pl.contracting ~acc dt
      (reshape ~by pl.a a) (reshape ~by pl.b b) init
  in
  let y = if identity pl.order then y else move ~by (M.Permute pl.order) y in
  reshape ~by pl.shape y

let contract ~by ?(sizes = []) ?acc ?init dt p a b =
  run ~by ?acc ?init dt (plan ~by ~sizes p (operand a) (operand b)) a b

(* The leading axes, aligned at their last, are batch axes where both operands
   hold the result's extent, and free axes of the operand that holds it where
   the other's extent is 1 or the other lacks the axis: an operand's unit axis
   drops, and no operand is broadcast. A 1-d [a] has no row and a 1-d [b] no
   column. The plan is read off the shapes: the kernel's axes are the batch
   axes, a's free axes and row, then b's free axes and column. *)
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
  let lead =
    match Prim.merge lead_a lead_b with
    | Ok s -> s
    | Error _ ->
        invalid_argf "%s: the leading axes do not broadcast; %s" by (both ())
  in
  let l = Array.length lead in
  let holds s i =
    let j = i - (l - Array.length s) in
    j >= 0 && s.(j) = lead.(i)
  in
  let in_a i = holds lead_a i in
  let in_b i = holds lead_b i || not (in_a i) in
  let count p =
    let c = ref 0 in
    for i = 0 to l - 1 do
      if p i then incr c
    done;
    !c
  in
  let la = count in_a and lb = count in_b in
  let nbatch = la + lb - l in
  let rows = if ra = 1 then 0 else 1 and cols = if rb = 1 then 0 else 1 in
  let r = l + rows + cols in
  let ea = Array.make (la + rows + 1) k and eb = Array.make (lb + 1 + cols) k in
  let batch = Array.make nbatch (0, 0) in
  let shape = Array.make r 0 and order = Array.make r 0 in
  (* The next place of each kind: [next_a] and [next_b] among each operand's
     axes; [next_batch], [next_free_a] and [next_free_b] among the kernel's
     axes, for a batch axis, a free axis of a after the batch axes, and one of
     b after a's row. *)
  let next_a = ref 0 and next_b = ref 0 in
  let next_batch = ref 0 in
  let next_free_a = ref nbatch and next_free_b = ref (la + rows) in
  for i = 0 to l - 1 do
    let of_a = in_a i and of_b = in_b i in
    shape.(i) <- lead.(i);
    if of_a && of_b then begin
      batch.(!next_batch) <- (!next_a, !next_b);
      order.(i) <- !next_batch;
      incr next_batch
    end
    else if of_a then begin
      order.(i) <- !next_free_a;
      incr next_free_a
    end
    else begin
      order.(i) <- !next_free_b;
      incr next_free_b
    end;
    if of_a then begin
      ea.(!next_a) <- lead.(i);
      incr next_a
    end;
    if of_b then begin
      eb.(!next_b) <- lead.(i);
      incr next_b
    end
  done;
  if rows = 1 then begin
    ea.(la) <- sa.(ra - 2);
    shape.(l) <- sa.(ra - 2);
    order.(l) <- la
  end;
  if cols = 1 then begin
    eb.(lb + 1) <- sb.(rb - 1);
    shape.(r - 1) <- sb.(rb - 1);
    order.(r - 1) <- r - 1
  end;
  let pl =
    {
      a = ea;
      b = eb;
      batch;
      contracting = [| (la + rows, lb) |];
      order;
      names = shape;
      shape;
    }
  in
  run ~by (Prim.dtype a) pl a b

(* Several operands *)

(* The most operands [einsum] plans: the tree for [n] takes 3^n steps, about
   12 ms at this bound on the bench row contract/einsum-plan-11, and each
   operand more triples it. *)
let max_operands = 11

(* Counts of multiply-adds, held at [max_int] past it. *)
let mul a b = if a <> 0 && b > max_int / a then max_int else a * b
let add a b = if a > max_int - b then max_int else a + b
let rec popcount s = if s = 0 then 0 else 1 + popcount (s land (s - 1))

(* The names of a pattern of several operands, numbered: the bit set of the
   operands holding each, whether the result keeps it, and its extent. *)
type table = { held : int array; kept : bool array; extent : int array }

(* The work of contracting the parts [t] and [u] of the set [s], from the
   [j]th name on, times [w]: the product of the extents of the names it
   meets. A name meets it where [s] holds it, unless every operand holding it
   lies in one part and the result drops it: that part summed it. *)
let rec work tb s t u j w =
  if j = Array.length tb.held then w
  else
    let h = tb.held.(j) in
    let summed = (not tb.kept.(j)) && (h land t = h || h land u = h) in
    let w = if h land s <> 0 && not summed then mul w tb.extent.(j) else w in
    work tb s t u (j + 1) w

(* The number of names the value of the set [s] holds, from the [j]th name
   on, plus [r]: those [s] holds that the result or an operand outside [s]
   needs. *)
let rec rank tb full s j r =
  if j = Array.length tb.held then r
  else
    let h = tb.held.(j) in
    let live = h land s <> 0 && (tb.kept.(j) || h land (full lxor s) <> 0) in
    rank tb full s (j + 1) (if live then r + 1 else r)

(* Whether the part [t] wins a tie with [t']: it holds more operands, or as
   many and the earliest operand one of them lacks. *)
let better t t' =
  let c = popcount t and c' = popcount t' in
  let d = t lxor t' in
  c > c' || (c = c' && t land d land -d <> 0)

(* The order in which to contract operands of names [ops] into the names
   [result], as each set of operands' left part: [split.(s)] for a bit set [s]
   of two operands or more, [0] where no tree of [s] keeps every value within
   the layout's ranks. It is the tree with the least work in all among those,
   and of splits of equal work the left part holds the set's earliest
   operand, then [better] decides. Each set's best split is found from its
   subsets' best: 3^n steps for n operands. *)
let order ~ext ~result ops =
  let n = Array.length ops in
  let full = (1 lsl n) - 1 in
  let names = List.sort_uniq Int.compare (List.concat (Array.to_list ops)) in
  let holders nm =
    let h = ref 0 in
    Array.iteri (fun i ns -> if mem nm ns then h := !h lor (1 lsl i)) ops;
    !h
  in
  let tb =
    {
      held = Array.of_list (List.map holders names);
      kept = Array.of_list (List.map (fun nm -> mem nm result) names);
      extent = Array.of_list (List.map (Array.get ext) names);
    }
  in
  let total = Array.make (full + 1) 0 and split = Array.make (full + 1) 0 in
  let built s = s land (s - 1) = 0 || split.(s) <> 0 in
  for s = 1 to full do
    let low = s land -s in
    let rest = s lxor low in
    let fits = s = full || rank tb full s 0 0 <= Nx_array.Layout.max_rank in
    (* Each part [t] holding [low], over the subsets of [rest] but itself. *)
    let sub = ref 0 in
    while fits && rest <> 0 && !sub <> rest do
      let t = low lor !sub in
      let u = s lxor t in
      if built t && built u then begin
        let w = add (add total.(t) total.(u)) (work tb s t u 0 1) in
        if
          split.(s) = 0 || w < total.(s) || (w = total.(s) && better t split.(s))
        then begin
          total.(s) <- w;
          split.(s) <- t
        end
      end;
      sub := (!sub - rest) land rest
    done
  done;
  split

let einsum (type v s d) ~by ?(sizes = []) ?acc (p : Pattern.t)
    (xs : (v, s, d) Value.t list) : (v, s, d) Value.t =
  let layouts =
    match p.kind with
    | One _ ->
        invalid_argf "%s: %a has one operand; use Nx.rearrange" by Pattern.pp p
    | Many { operands; _ } -> operands
  in
  let k = List.length layouts and n = List.length xs in
  if k > max_operands then
    invalid_argf
      "%s: %a has %d operands; einsum plans at most %d: contract parts in \
       separate einsums"
      by Pattern.pp p k max_operands;
  if n <> k then
    invalid_argf "%s: %a has %d operands but the list holds %d" by Pattern.pp p
      k n;
  let dt = Prim.dtype (List.hd xs) in
  if D.is D.Boolean dt then invalid_argf "%s: the operands are booleans" by;
  let labels =
    List.mapi
      (fun i x -> ("x" ^ string_of_int (i + 1), D.name dt, Prim.shape x))
      xs
  in
  let ns, ext, nr, shape = prepare ~by ~sizes p layouts labels in
  let xs = Array.of_list xs and ns = Array.of_list ns in
  let (D.Any w) = accumulator ~by ?acc dt (D.Any dt) (D.Any dt) in
  (* Two parts' contraction into [into], of names [names] and shape [shape]:
     the names both hold and [names] lacks are summed. *)
  let pair (type x y) (into : (x, y) D.t) names shape (na, l) (nb, r) :
      (x, y, d) Value.t =
    match (l, r) with
    | Value.Any a, Value.Any b ->
        run ~by ~acc:(D.Any w) into (lower ~ext na nb names shape) a b
  in
  let leaf i = (ns.(i), Value.Any xs.(i)) in
  if n = 2 then pair dt nr shape (leaf 0) (leaf 1)
  else
    let split = order ~ext ~result:nr ns in
    let full = (1 lsl n) - 1 in
    if split.(full) = 0 then
      invalid_argf "%s: %a: every order holds a value of more than %d axes" by
        Pattern.pp p Nx_array.Layout.max_rank;
    (* The names of the set [s]'s value, in the kernel's order so that no
       intermediate moves, and the value, in [w]. *)
    let rec value s =
      if s land (s - 1) = 0 then leaf (popcount (s - 1))
      else
        let t = split.(s) in
        let ((na, _) as l) = value t and ((nb, _) as r) = value (s lxor t) in
        let rec outside nm i =
          i < n
          && (((1 lsl i) land s = 0 && mem nm ns.(i)) || outside nm (i + 1))
        in
        let needed nm = mem nm nr || outside nm 0 in
        let names =
          List.filter (fun nm -> mem nm nb && needed nm) na
          @ List.filter (fun nm -> not (mem nm nb)) na
          @ List.filter (fun nm -> not (mem nm na)) nb
        in
        let shape = Array.of_list (List.map (Array.get ext) names) in
        (names, Value.Any (pair w names shape l r))
    in
    let t = split.(full) in
    pair dt nr shape (value t) (value (full lxor t))
