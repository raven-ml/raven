(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Prog = Prog
module D = Nx_array.Dtype
module L = Nx_array.Layout

type ('v, 's) a = ('v, 's) Nx_array.t
type index = (int64, Nx_array.Dtype.int64_elt) Nx_array.t

type pad = {
  lo : int iarray;
  hi : int iarray;
  interior : int iarray;
  windows : Nx_array.Move.window iarray;
}

type load =
  | Plain : ('v, 's) a -> load
  | Padded : { x : ('v, 's) a; fill : 'v; pad : pad } -> load

type monoid = Sum | Prod | Max | Min
type extreme = Max | Min

(* The contract leaves a NaN result's sign and payload unspecified, but for
   a move's, so that a reduction may run by a tree and by the hardware's
   own instructions. IEEE 754 specifies neither for arithmetic, and where
   two NaNs meet, the one an instruction passes on depends on its
   operands' order, which a kernel's paths for different layouts may set
   differently. *)
type reduction = Monoid of monoid | Logsumexp | Moments | Arg of extreme

let accepts (type v s) r (dt : (v, s) D.t) =
  match (r, D.kind dt) with
  | Monoid (Sum | Prod), D.Boolean -> false
  | (Logsumexp | Moments), D.Float -> true
  | (Logsumexp | Moments), _ -> false
  | _ -> true

type any_float = Any_float : (float, 's) D.t -> any_float [@@unboxed]

let float_carrier (type s) (dt : (float, s) D.t) =
  match dt with
  | Float16 | Bfloat16 | Float8_e4m3fn | Float8_e5m2 | Float4_e2m1fn ->
      Any_float Float32
  | Float32 | Float64 -> Any_float dt

let carrier (D.Any dt as d) =
  match D.kind dt with
  | D.Float ->
      let (Any_float c) = float_carrier dt in
      D.Any c
  | D.Complex | D.Signed | D.Unsigned | D.Boolean -> (
      match dt with
      | Int4 -> D.Any Int8
      | Uint4 -> D.Any Uint8
      | Bit -> D.Any Bool
      | _ -> d)

type scale =
  | Exponent of (int, D.int8_unsigned_elt) Nx_array.t
  | Value : ('v, 's) Nx_array.t -> scale

type operand = { array : Nx_array.any; scale : scale option }

type 'i segments =
  | Rows of { ids : 'i; a_axis : int; b_axis : int }
  | Contracting of { ids : 'i; groups : int; pair : int }

type combine = Set | Add | Max | Min

let accumulates (type v s) (dt : (v, s) D.t) =
  match D.kind dt with
  | D.Boolean -> false
  | D.Float -> D.bits dt >= 32
  | D.Complex | D.Signed | D.Unsigned -> true

type direction = Forward | Inverse

let same_precision (type v s w t) (x : (v, s) D.t) (y : (w, t) D.t) =
  match (x, y) with
  | (Float32 | Complex64), (Float32 | Complex64)
  | (Float64 | Complex128), (Float64 | Complex128) ->
      true
  | _ -> false

type triangle = Lower | Upper
type factors = Reduced | Complete

let full_precision (type v s) (dt : (v, s) D.t) =
  match dt with
  | Float32 | Float64 | Complex64 | Complex128 -> true
  | _ -> false

type ('a, 'i) kv =
  | Dense of { k : 'a; v : 'a }
  | Paged of { k : 'a; v : 'a; table : 'i }

(* Refusals: each rule answers the refusal and why, formatting the reason
   only when it refuses. *)

let attended : D.any -> bool = function
  | Any (Float16 | Bfloat16 | Float32 | Float64) -> true
  | Any _ -> false

let accumulator : D.any -> bool = function
  | Any (Float32 | Float64) -> true
  | Any _ -> false

let is_bool : D.any -> bool = function Any Bool -> true | Any _ -> false
let is_of dt (D.Any x) = D.equal x dt
let output (p : Prog.t) k = Prog.dtype p (Iarray.get p.outs k)
let name (D.Any dt) = D.name dt
let refuse r fmt = Format.kasprintf (fun why -> Some (r, why)) fmt
let wrong fmt = refuse Nx_array.Wrong_dtype fmt
let misfits fmt = refuse Nx_array.Shape_mismatch fmt

type 'c dtypes = { arrays : 'c -> int; dtype : 'c -> int -> D.any }

type 'c reductions = {
  reductions : 'c -> int;
  kind : 'c -> int -> reduction;
  output : 'c -> int -> int;
}

(* The first load of [loads] from [i] on, read by [v], whose dtype is not
   that of [p]'s operand [first] places after it, or [-1]. *)
let rec misfit (p : Prog.t) ~first v loads i =
  if i = v.arrays loads then -1
  else
    let (D.Any want) = Iarray.get p.ins (first + i) in
    if is_of want (v.dtype loads i) then misfit p ~first v loads (i + 1)
    else i

(* The first axis at or past [rank] that a [Coord] node of [p] from [i] on
   reads, or [-1]. *)
let rec past (p : Prog.t) rank i =
  if i = Iarray.length p.nodes then -1
  else
    match Iarray.get p.nodes i with
    | Coord a when a >= rank -> a
    | _ -> past p rank (i + 1)

(* Whether node [o] of [p] reads the score: the nodes it reaches, marked
   from [o] down, since a node refers only to earlier ones. *)
let reads_score (p : Prog.t) o =
  let seen = Bytes.make (o + 1) '\000' and found = ref false in
  Bytes.set seen o '\001';
  for i = o downto 0 do
    if Bytes.get seen i = '\001' then
      match Iarray.get p.nodes i with
      | In k -> if k = 0 then found := true
      | Coord _ | Const _ -> ()
      | Op1 (_, _, a) -> Bytes.set seen a '\001'
      | Op2 (_, a, b) ->
          Bytes.set seen a '\001';
          Bytes.set seen b '\001'
      | Op3 (_, a, b, c) ->
          Bytes.set seen a '\001';
          Bytes.set seen b '\001';
          Bytes.set seen c '\001'
  done;
  !found

(* The program's refusals once its numbers of operands and outputs fit. *)
let program_refusal (type c) (p : Prog.t) ~rank ~(acc : (float, c) D.t) v
    loads =
  let bad = misfit p ~first:1 v loads 0 and axis = past p rank 0 in
  if not (is_of acc (Iarray.get p.ins 0)) then
    wrong "a pair's program reads its score as %s, not %s"
      (name (Iarray.get p.ins 0)) (D.name acc)
  else if bad >= 0 then
    wrong "load %d is %s where the program reads %s" bad
      (name (v.dtype loads bad))
      (name (Iarray.get p.ins (bad + 1)))
  else if not (is_bool (output p 0) && is_of acc (output p 1)) then
    wrong "a pair's program gives %s and %s, not bool and %s"
      (name (output p 0)) (name (output p 1)) (D.name acc)
  else if axis >= 0 then
    misfits "a coordinate along axis %d of pairs of rank %d" axis rank
  else if reads_score p (Iarray.get p.outs 0) then
    misfits "a pair's visibility reads its score"
  else None

let attention_refusal (type v s w r c) (p : Prog.t) ~(q : (v, s) D.t) ~rank
    ~(o : (w, r) D.t) ~(acc : (float, c) D.t) ~loads:v loads =
  let ins = Iarray.length p.ins and outs = Iarray.length p.outs in
  let n = v.arrays loads in
  if not (attended (Any q)) then
    wrong "attention of %s, not float16, bfloat16, float32 or float64"
      (D.name q)
  else if not (accumulator (Any acc)) || D.bits q > D.bits acc then
    wrong
      "attention's accumulator is float32 or float64, no narrower than the \
       query: %s for a query of %s"
      (D.name acc) (D.name q)
  else if not (D.equal o acc || D.equal o q) then
    wrong "attention's o of %s, not %s or %s" (D.name o)
      (D.name acc) (D.name q)
  else if ins <> n + 1 then
    misfits "a pair's program of %d operands over a score and %d loads" ins n
  else if outs <> 2 then
    misfits "a pair's program of %d outputs, not 2" outs
  else program_refusal p ~rank ~acc v loads

(* Loops *)

let reduction_name : reduction -> string = function
  | Monoid Sum -> "Sum"
  | Monoid Prod -> "Prod"
  | Monoid Max -> "Max"
  | Monoid Min -> "Min"
  | Logsumexp -> "Logsumexp"
  | Moments -> "Moments"
  | Arg Max -> "Arg Max"
  | Arg Min -> "Arg Min"

(* The destinations reduction [r] stores. *)
let width : reduction -> int = function
  | Arg _ | Moments -> 2
  | Monoid _ | Logsumexp -> 1

let takes r (D.Any dt) = accepts r dt
let is_int64 : D.any -> bool = function Any Int64 -> true | Any _ -> false

let operands_refusal (p : Prog.t) v loads =
  let ins = Iarray.length p.ins and n = v.arrays loads in
  if n <> ins then misfits "a program of %d operands over %d loads" ins n
  else None

(* Why a loop of [p] of rank [rank] refuses its loads' dtypes, read by
   [v], or its coordinates, once their numbers fit. *)
let loop_refusal (p : Prog.t) ~rank v loads =
  let bad = misfit p ~first:0 v loads 0 and axis = past p rank 0 in
  if bad >= 0 then
    wrong "load %d is %s where the program reads %s" bad
      (name (v.dtype loads bad))
      (name (Iarray.get p.ins bad))
  else if axis >= 0 then
    misfits "a coordinate along axis %d of a loop of rank %d" axis rank
  else None

(* The first destination of [dsts] from [i] on, read by [v], whose dtype is
   not its output's in [p], or [-1]. *)
let rec misgiven (p : Prog.t) v dsts i =
  if i = v.arrays dsts then -1
  else
    let (D.Any want) = output p i in
    if is_of want (v.dtype dsts i) then misgiven p v dsts (i + 1) else i

let map_refusal (p : Prog.t) ~rank ~dsts:dv dsts ~loads:lv loads =
  let outs = Iarray.length p.outs and nd = dv.arrays dsts in
  match operands_refusal p lv loads with
  | Some _ as e -> e
  | None ->
      if nd <> outs then
        misfits "%d destinations for a program of %d outputs" nd outs
      else
        let bad = misgiven p dv dsts 0 in
        if bad >= 0 then
          wrong "destination %d is %s where the program gives %s" bad
            (name (dv.dtype dsts bad))
            (name (output p bad))
        else loop_refusal p ~rank lv loads

(* Why reduction [r] of output [k] of [p], into the destinations of [dsts]
   from [at] on, read by [v], refuses its dtypes. *)
let reduction_refusal (p : Prog.t) r k v dsts at =
  let out = output p k in
  if not (takes r out) then
    wrong "%s does not take %s" (reduction_name r) (name out)
  else
    match r with
    | Arg _ when not (is_int64 (v.dtype dsts (at + 1))) ->
        wrong "an Arg's position of %s, not int64"
          (name (v.dtype dsts (at + 1)))
    | Arg _ | Monoid _ | Logsumexp | Moments -> None

(* The first of [rs] from [i] on, read by [v], of an output [p] does not
   have, or [-1]. *)
let rec unproduced (p : Prog.t) v rs i =
  if i = v.reductions rs then -1
  else
    let k = v.output rs i in
    if k < 0 || k >= Iarray.length p.outs then i
    else unproduced p v rs (i + 1)

let rec widths v rs i n =
  if i = v.reductions rs then n
  else widths v rs (i + 1) (n + width (v.kind rs i))

(* Why [rs] from [i] on, into the destinations from [at] on, refuse their
   dtypes. *)
let rec reductions_refusal p rv rs dv dsts i at =
  if i = rv.reductions rs then None
  else
    let r = rv.kind rs i in
    match reduction_refusal p r (rv.output rs i) dv dsts at with
    | Some _ as e -> e
    | None -> reductions_refusal p rv rs dv dsts (i + 1) (at + width r)

let reduce_refusal (p : Prog.t) ~rank ~reductions:rv rs ~dsts:dv dsts
    ~loads:lv loads =
  let bad = unproduced p rv rs 0 and nd = dv.arrays dsts in
  match operands_refusal p lv loads with
  | Some _ as e -> e
  | None -> (
      if rv.reductions rs = 0 then misfits "a reduction of no output"
      else if bad >= 0 then
        misfits "a reduction of output %d of a program of %d"
          (rv.output rs bad) (Iarray.length p.outs)
      else if nd <> widths rv rs 0 0 then
        misfits "%d destinations for reductions that store %d" nd
          (widths rv rs 0 0)
      else
        match reductions_refusal p rv rs dv dsts 0 0 with
        | Some _ as e -> e
        | None -> loop_refusal p ~rank lv loads)

let scan_refusal (p : Prog.t) ~rank r ~output:k ~dsts:dv dsts ?carry
    ~loads:lv loads =
  let nd = dv.arrays dsts in
  match operands_refusal p lv loads with
  | Some _ as e -> e
  | None -> (
      if k < 0 || k >= Iarray.length p.outs then
        misfits "a scan of output %d of a program of %d" k
          (Iarray.length p.outs)
      else if nd <> width r then
        misfits "%d destinations for a scan that stores %d" nd (width r)
      else
        match (carry, r) with
        | Some _, (Logsumexp | Moments | Arg _) ->
            misfits "a carry for %s" (reduction_name r)
        | _ -> (
            match reduction_refusal p r k dv dsts 0 with
            | Some _ as e -> e
            | None -> (
                let (D.Any want) = carrier (output p k) in
                match carry with
                | Some c when not (is_of want c) ->
                    wrong "a carry of %s where the terms carry in %s" (name c)
                      (D.name want)
                | Some _ | None -> loop_refusal p ~rank lv loads)))

module Shape = struct
  type t = int array

  let errorf fmt = Format.kasprintf Result.error fmt
  let ( let* ) = Result.bind

  let pp ppf s =
    Format.fprintf ppf "[%a]"
      (Format.pp_print_list
         ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
         Format.pp_print_int)
      (Array.to_list s)

  (* Whether [s]'s extents are non-negative and those other than 0 multiply to
     at most max_numel: the shapes a layout admits. *)
  let admitted s =
    let ok = ref true and n = ref 1 and i = ref 0 in
    while !ok && !i < Array.length s do
      let d = s.(!i) in
      if d < 0 || (d > 0 && d > L.max_numel / !n) then ok := false
      else if d > 0 then n := !n * d;
      incr i
    done;
    !ok

  let admit what s =
    if admitted s then Ok s
    else errorf "%s of shape %a has more than %d elements" what pp s L.max_numel

  (* [a + b], or [None] past an int. *)
  let add a b =
    if (b > 0 && a > max_int - b) || (b < 0 && a < min_int - b) then None
    else Some (a + b)

  (* The extent [lo + hi + d + interior·(d - 1)] of an axis of extent [d]
     padded by [lo], [hi] and [interior], or [None] past an int. *)
  let padded_extent lo hi interior d =
    let gaps =
      if d <= 1 then Some 0
      else if interior > max_int / (d - 1) then None
      else Some (interior * (d - 1))
    in
    Option.bind gaps (fun g ->
        Option.bind (add d g) (fun e -> Option.bind (add e lo) (fun e -> add e hi)))

  let padded (p : pad) x =
    let r = Array.length x in
    if Iarray.length p.lo <> r || Iarray.length p.hi <> r
       || Iarray.length p.interior <> r
    then
      errorf "a pad of rank %d for an operand of rank %d" (Iarray.length p.lo) r
    else if Iarray.exists (fun i -> i < 0) p.interior then
      errorf "a negative interior padding"
    else
      let padded =
        Array.mapi
          (fun i d ->
            padded_extent (Iarray.get p.lo i) (Iarray.get p.hi i)
              (Iarray.get p.interior i) d
            |> Option.value ~default:(-1))
          x
      in
      match Array.find_index (fun d -> d < 0) padded with
      | Some i -> errorf "padded axis %d is negative or overflows" i
      | None -> (
          let* padded = admit "a padded operand" padded in
          if Iarray.length p.windows = 0 then Ok padded
          else
            match
              Nx_array.Move.shape (Window p.windows) (Iarray.of_array padded)
            with
            | y -> Ok (Iarray.to_array y)
            | exception Invalid_argument e -> errorf "windows: %s" e)

  (* Whether [axes] from the [i]th on are not strictly increasing axes of a
     rank [r]. *)
  let rec bad_axes axes r i =
    i < Iarray.length axes
    && (let a = Iarray.get axes i in
        a < 0 || a >= r
        || (i > 0 && a <= Iarray.get axes (i - 1))
        || bad_axes axes r (i + 1))

  let check_axes axes r =
    if bad_axes axes r 0 then
      errorf "axes %a are not strictly increasing axes of a rank %d" pp
        (Iarray.to_array axes) r
    else Ok ()

  let extreme = function Monoid (Max | Min) | Arg _ -> true | _ -> false

  let reduce ~axes rs y =
    match check_axes axes (Array.length y) with
    | Error e -> Error e
    | Ok () ->
        (* One pass: the kept extents, their product, and the first reduced
           axis of no element. *)
        let n = Iarray.length axes in
        let kept = Array.make (Array.length y - n) 0 in
        let total = ref 1 and j = ref 0 and empty = ref (-1) in
        for i = 0 to Array.length y - 1 do
          if !j < n && Iarray.get axes !j = i then begin
            if y.(i) = 0 && !empty < 0 then empty := i;
            incr j
          end
          else begin
            kept.(i - !j) <- y.(i);
            total := !total * y.(i)
          end
        done;
        if !empty >= 0 && !total > 0 && Array.exists extreme rs then
          errorf "an extreme of no term: no element along axis %d of %a" !empty
            pp y
        else Ok kept

  let scan ~axis y =
    let r = Array.length y in
    if axis < 0 || axis >= r then
      errorf "axes %a are not strictly increasing axes of a rank %d" pp
        [| axis |] r
    else Ok y

  let carry ~axis y =
    let* y = scan ~axis y in
    let c = Array.copy y in
    c.(axis) <- 1;
    Ok c

  let rec same_off axis x y i =
    i = Array.length x
    || ((i = axis || x.(i) = y.(i)) && same_off axis x y (i + 1))

  (* Whether [x] and [y] have one rank and one extent along every axis but
     [axis]. *)
  let off_axis axis x y = Array.length x = Array.length y && same_off axis x y 0

  let gather ~axis idx x =
    if axis < 0 || axis >= Array.length x then
      errorf "axis %d of an operand of rank %d" axis (Array.length x)
    else if off_axis axis idx x then Ok idx
    else errorf "positions %a do not fit %a along axis %d" pp idx pp x axis

  let gather_ranges ~size starts ends x =
    let r = Array.length starts in
    let a = r - 1 in
    if r = 0 then errorf "0-d starts"
    else if starts <> ends then
      errorf "starts %a and ends %a differ" pp starts pp ends
    else if Array.length x <= a || Array.sub x 0 a <> Array.sub starts 0 a
    then errorf "runs %a do not fit %a along axis %d" pp starts pp x a
    else if size < 0 then errorf "a size of %d" size
    else
      let y = Array.copy x in
      y.(a) <- size;
      admit "a range gather" y

  let scatter ~axis into idx u =
    if axis < 0 || axis >= Array.length into then
      errorf "axis %d of a target of rank %d" axis (Array.length into)
    else if idx <> u then
      errorf "positions %a and updates %a differ" pp idx pp u
    else if off_axis axis u into then Ok into
    else errorf "updates %a do not fit %a along axis %d" pp u pp into axis

  let sort ~axis ~k x =
    if axis < 0 || axis >= Array.length x then
      errorf "axis %d of a rank %d" axis (Array.length x)
    else
      match k with
      | None -> Ok x
      | Some n when n < 0 || n > x.(axis) ->
          errorf "keeps %d of an axis of %d" n x.(axis)
      | Some n ->
          let y = Array.copy x in
          y.(axis) <- n;
          Ok y

  let unique ~size keys =
    if Array.length keys <> 2 then
      errorf "keys %a are not of rank 2" pp keys
    else if size < 0 then errorf "a size of %d" size
    else
      let* first = admit "a numbering's first" [| size |] in
      Ok ([| keys.(0) |], first, [||])

  let assemble y regions =
    let* y = admit "an assembly" y in
    let y = Iarray.of_array y in
    let piece j rs =
      match Nx_array.Move.shape (Slice rs) y with
      | s -> Ok (Iarray.to_array s)
      | exception Invalid_argument e -> errorf "region %d: %s" j e
    in
    let n = Array.length regions in
    let shapes = Array.make n [||] in
    let rec go j =
      if j = n then Ok shapes
      else
        let* s = piece j regions.(j) in
        shapes.(j) <- s;
        go (j + 1)
    in
    go 0

  (* Pair [k] of a contraction: its batch pairs, then its contracting ones. *)
  let nth_pair batch contracting k =
    let nb = Iarray.length batch in
    if k < nb then Iarray.get batch k else Iarray.get contracting (k - nb)

  (* Whether pair [k] is the first to name axis [i] of [side]. *)
  let first_on side batch contracting k i =
    let earlier = ref 0 in
    while !earlier < k && side (nth_pair batch contracting !earlier) <> i do
      incr earlier
    done;
    !earlier = k

  (* Whether a pair names axis [i] of [side]. *)
  let paired side batch contracting i =
    let n = Iarray.length batch + Iarray.length contracting in
    not (first_on side batch contracting n i)

  (* Whether a scale of shape [s] fits an operand of shape [x]: of its rank,
     each extent dividing the operand's, 0 only along an empty axis. *)
  let scale_fits x s =
    Array.length s = Array.length x
    && Array.for_all2
         (fun d e -> if e = 0 then d = 0 else e > 0 && d mod e = 0)
         x s

  let check_scale name x = function
    | Some s when not (scale_fits x s) ->
        errorf "a scale of shape %a for %s of shape %a" pp s name pp x
    | Some _ | None -> Ok ()

  (* Whether ids of shape [o] have [a]'s batch extents, then [n]. *)
  let fits batch a o n =
    let nb = Iarray.length batch in
    let fit = ref (Array.length o = nb + 1 && o.(nb) = n) in
    for k = 0 to nb - 1 do
      if !fit && o.(k) <> a.(fst (Iarray.get batch k)) then fit := false
    done;
    !fit

  let check_free name side r batch contracting ax =
    if ax < 0 || ax >= r then errorf "no axis %d of %s to segment" ax name
    else if paired side batch contracting ax then
      errorf "axis %d of %s is in a pair and segmented" ax name
    else Ok ()

  (* The groups' count of [segments], checked against the pairs. *)
  let check_segments ~batch ~contracting a b = function
    | None -> Ok 0
    | Some (Rows { ids; a_axis; b_axis }) ->
        let* () = check_free "a" fst (Array.length a) batch contracting a_axis in
        let* () = check_free "b" snd (Array.length b) batch contracting b_axis in
        if fits batch a ids a.(a_axis) then Ok b.(b_axis)
        else errorf "ids of shape %a" pp ids
    | Some (Contracting { ids; groups; pair }) ->
        if pair < 0 || pair >= Iarray.length contracting then
          errorf "no contracting pair %d to segment" pair
        else if groups < 0 then errorf "%d groups" groups
        else if fits batch a ids a.(fst (Iarray.get contracting pair)) then
          Ok groups
        else errorf "ids of shape %a" pp ids

  let check_pairs ~batch ~contracting a b =
    let ra = Array.length a and rb = Array.length b in
    let nb = Iarray.length batch in
    let n = nb + Iarray.length contracting in
    let error = ref None and k = ref 0 in
    while Option.is_none !error && !k < n do
      let i, j = nth_pair batch contracting !k in
      let what = if !k < nb then "batch" else "contracting" in
      let p = if !k < nb then !k else !k - nb in
      if i < 0 || i >= ra then
        error := Some (Printf.sprintf "%s pair %d: no axis %d of a" what p i)
      else if j < 0 || j >= rb then
        error := Some (Printf.sprintf "%s pair %d: no axis %d of b" what p j)
      else if not (first_on fst batch contracting !k i) then
        error := Some (Printf.sprintf "axis %d of a is in two pairs" i)
      else if not (first_on snd batch contracting !k j) then
        error := Some (Printf.sprintf "axis %d of b is in two pairs" j)
      else if a.(i) <> b.(j) then
        error :=
          Some
            (Printf.sprintf "%s pair %d: extents %d and %d" what p a.(i) b.(j));
      incr k
    done;
    match !error with Some e -> Error e | None -> Ok ()

  (* The groups' axis of [Contracting], the batch extents, then the free ones:
     [a]'s, then [b]'s but [Rows]' [b_axis]. *)
  let result ~batch ~contracting ~segments g a b =
    let ra = Array.length a and rb = Array.length b in
    let nb = Iarray.length batch in
    let n = nb + Iarray.length contracting in
    let lead, drop =
      match segments with
      | Some (Contracting _) -> (1, -1)
      | Some (Rows { b_axis; _ }) -> (0, b_axis)
      | None -> (0, -1)
    in
    let y = Array.make (lead + nb + ra + rb - (2 * n) - Bool.to_int (drop >= 0)) 0 in
    if lead = 1 then y.(0) <- g;
    for k = 0 to nb - 1 do
      y.(lead + k) <- a.(fst (Iarray.get batch k))
    done;
    let k = ref (lead + nb) in
    for ax = 0 to ra - 1 do
      if not (paired fst batch contracting ax) then begin
        y.(!k) <- a.(ax);
        incr k
      end
    done;
    for ax = 0 to rb - 1 do
      if ax <> drop && not (paired snd batch contracting ax) then begin
        y.(!k) <- b.(ax);
        incr k
      end
    done;
    if Array.length y > L.max_rank then
      errorf "a result of rank %d" (Array.length y)
    else admit "a contraction" y

  (* Plain matches and loops: the eager path allocates nothing here beyond the
     result. *)
  let contract ~batch ~contracting ~segments (a, sa) (b, sb) =
    match check_pairs ~batch ~contracting a b with
    | Error _ as e -> e
    | Ok () -> (
        match (check_scale "a" a sa, check_scale "b" b sb) with
        | (Error _ as e), _ | _, (Error _ as e) -> e
        | Ok (), Ok () -> (
            match check_segments ~batch ~contracting a b segments with
            | Error _ as e -> e
            | Ok g -> result ~batch ~contracting ~segments g a b))

  let c2c ~axes x =
    if Iarray.length axes = 0 then Error "a transform along no axis"
    else
      let* () = check_axes axes (Array.length x) in
      Ok x

  let last axes = Iarray.get axes (Iarray.length axes - 1)

  (* [x] with [n] along the last of [axes]. *)
  let along_last axes x n =
    let y = Array.copy x in
    y.(last axes) <- n;
    admit "a transform" y

  let r2c ~axes x =
    let* x = c2c ~axes x in
    along_last axes x ((x.(last axes) / 2) + 1)

  let c2r ~axes ~n x =
    let* x = c2c ~axes x in
    let last = last axes in
    if n < 0 then errorf "a transform of %d points" n
    else if x.(last) <> (n / 2) + 1 then
      errorf "%d bins along axis %d for %d points" x.(last) last n
    else along_last axes x n

  (* [a]'s batch, rows and columns. *)
  let matrix a =
    let r = Array.length a in
    if r < 2 then errorf "a matrix of rank %d" r
    else Ok (Array.sub a 0 (r - 2), a.(r - 2), a.(r - 1))

  let square a =
    let* ((_, m, n) as mat) = matrix a in
    if m <> n then errorf "a matrix of %d rows and %d columns" m n else Ok mat

  let cholesky a =
    let* _ = square a in
    Ok a

  let lu a =
    let* batch, m, n = matrix a in
    let y tail = Array.append batch tail in
    Ok (a, y [| min m n |], y [| m |])

  let qr f a =
    let* batch, m, n = matrix a in
    let y tail = Array.append batch tail in
    let q, r =
      match f with
      | Reduced -> (y [| m; min m n |], y [| min m n; n |])
      | Complete -> (y [| m; m |], a)
    in
    let* q = admit "q" q in
    let* r = admit "r" r in
    Ok (q, r)

  let svd f a =
    let* batch, m, n = matrix a in
    let y tail = Array.append batch tail in
    let k = min m n in
    let* s = admit "singular values" (y [| k |]) in
    let u, vh =
      match f with
      | Reduced -> (y [| m; k |], y [| k; n |])
      | Complete -> (y [| m; m |], y [| n; n |])
    in
    let* u = admit "u" u in
    let* vh = admit "vh" vh in
    Ok (s, u, vh)

  let eigh a =
    let* batch, _, n = square a in
    Ok (Array.append batch [| n |], a)

  let solve_triangular a b =
    let* batch, _, n = square a in
    if Array.length b <> Array.length a
       || Array.sub b 0 (Array.length b - 1) <> Array.append batch [| n |]
    then errorf "right-hand sides %a for a matrix %a" pp b pp a
    else Ok b

  (* [lead] then [tail]. *)
  let with_lead lead tail = Array.append lead tail

  let expect what x s =
    if x = s then Ok () else errorf "%s %a, not %a" what pp x pp s

  let attention q kv ~lengths loads =
    let r = Array.length q in
    if r < 3 then errorf "queries %a of fewer than three axes" pp q
    else
      let lead = Array.sub q 0 (r - 3) in
      let hq = q.(r - 3) and tq = q.(r - 2) and dk = q.(r - 1) in
      let rank4 what x =
        if Array.length x = 4 then Ok () else errorf "%s %a not of rank 4" what pp x
      in
      let* hk, tk, dv =
        match kv with
        | Dense { k; v } ->
            if Array.length k <> r || Array.length v <> r then
              errorf "keys %a and values %a not of the queries' rank %d" pp k
                pp v r
            else
              let hk = k.(r - 3) and tk = k.(r - 2) and dv = v.(r - 1) in
              let* () = expect "keys" k (with_lead lead [| hk; tk; dk |]) in
              let* () = expect "values" v (with_lead lead [| hk; tk; dv |]) in
              Ok (hk, tk, dv)
        | Paged { k; v; table } ->
            let* () = rank4 "a key pool" k in
            let* () = rank4 "a value pool" v in
            let pages = k.(0) and hk = k.(1) and page = k.(2) in
            let* () = expect "a key pool" k [| pages; hk; page; dk |] in
            let* () = expect "a value pool" v [| pages; hk; page; v.(3) |] in
            let rt = Array.length table in
            if rt <> r - 2 then
              errorf "a page table %a not of rank %d" pp table (r - 2)
            else
              let* () =
                expect "a page table" table
                  (with_lead lead [| table.(rt - 1) |])
              in
              let max_pages = table.(rt - 1) in
              if page > 0 && max_pages > L.max_numel / page then
                errorf "%d pages of %d keys" max_pages page
              else Ok (hk, max_pages * page, v.(3))
      in
      if (hk = 0 && hq <> 0) || (hk > 0 && hq mod hk <> 0) then
        errorf "%d key heads for %d query heads" hk hq
      else
        let* () =
          match lengths with
          | None -> Ok ()
          | Some l -> expect "lengths" l lead
        in
        let pairs = with_lead lead [| hq; tq; tk |] in
        let* () =
          Array.fold_left
            (fun acc x ->
              let* () = acc in
              expect "a load" x pairs)
            (Ok ()) loads
        in
        let* o = admit "attention's o" (with_lead lead [| hq; tq; dv |]) in
        let* lse = admit "attention's lse" (with_lead lead [| hq; tq |]) in
        Ok (o, lse)
end

module type S = sig
  type answer := Nx_array.answer
  type any := Nx_array.any

  val name : string
  val computes_on : Rig.t -> bool
  val map : Prog.t -> dsts:any array -> load array -> answer

  val reduce :
    Prog.t ->
    shape:int array ->
    axes:int iarray ->
    (reduction * int) array ->
    dsts:any array ->
    load array ->
    answer

  val scan :
    ?carry:any ->
    Prog.t ->
    axis:int ->
    reduction * int ->
    dsts:any array ->
    load array ->
    answer

  val gather : axis:int -> dst:('v, 's) a -> index -> ('v, 's) a -> answer

  val gather_ranges :
    dst:('v, 's) a -> starts:index -> ends:index -> ('v, 's) a -> answer

  val scatter :
    combine ->
    unique:bool ->
    axis:int ->
    dst:('v, 's) a ->
    into:('v, 's) a ->
    index ->
    ('v, 's) a ->
    answer

  val sort :
    axis:int ->
    descending:bool ->
    values:('v, 's) a ->
    positions:index ->
    ('v, 's) a ->
    answer

  val unique :
    ids:index -> first:index -> counts:index -> count:index -> index -> answer

  val assemble :
    fill:'v ->
    dst:('v, 's) a ->
    (Nx_array.Move.range iarray * ('v, 's) a) array ->
    answer

  val fold : pad -> dst:('v, 's) a -> ('v, 's) a -> answer
  val contract :
    batch:(int * int) iarray ->
    contracting:(int * int) iarray ->
    segments:index segments option ->
    acc:Nx_array.Dtype.any ->
    dst:('v, 's) a ->
    init:('v, 's) a option ->
    operand ->
    operand ->
    answer

  val c2c :
    direction -> axes:int iarray -> dst:(Complex.t, 's) a -> (Complex.t, 's) a -> answer

  val r2c : axes:int iarray -> dst:(Complex.t, 'c) a -> (float, 'r) a -> answer
  val c2r : axes:int iarray -> dst:(float, 'r) a -> (Complex.t, 'c) a -> answer
  val cholesky : triangle -> dst:('v, 's) a -> ('v, 's) a -> answer
  val lu : lu:('v, 's) a -> pivots:index -> perm:index -> ('v, 's) a -> answer
  val qr : q:('v, 's) a -> r:('v, 's) a -> ('v, 's) a -> answer

  val svd :
    s:(float, 'r) a ->
    ?vectors:(u:('v, 's) a * vh:('v, 's) a) ->
    ('v, 's) a ->
    answer

  val eigh : w:(float, 'r) a -> ?v:('v, 's) a -> ('v, 's) a -> answer

  val eig :
    w:(Complex.t, 'c) a -> ?v:(Complex.t, 'c) a -> ('v, 's) a -> answer

  val solve_triangular :
    triangle ->
    adjoint:bool ->
    unit_diagonal:bool ->
    dst:('v, 's) a ->
    ('v, 's) a ->
    ('v, 's) a ->
    answer

  val attention :
    Prog.t ->
    scale:float ->
    o:('w, 'r) a ->
    lse:(float, 'c) a ->
    ?lengths:index ->
    ('v, 's) a ->
    (('v, 's) a, index) kv ->
    any array ->
    answer
end
