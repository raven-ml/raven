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
  | Rows of { offsets : 'i; a_axis : int; b_axis : int }
  | Contracting of { offsets : 'i; pair : int }

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
            match Nx_array.Move.shape (Window p.windows) padded with
            | y -> Ok y
            | exception Invalid_argument e -> errorf "windows: %s" e)

  (* [axes] strictly increasing axes of a rank [r]. *)
  let check_axes axes r =
    let rec bad i =
      i < Iarray.length axes
      && (let a = Iarray.get axes i in
          a < 0 || a >= r
          || (i > 0 && a <= Iarray.get axes (i - 1))
          || bad (i + 1))
    in
    if bad 0 then
      errorf "axes %a are not strictly increasing axes of a rank %d" pp
        (Iarray.to_array axes) r
    else Ok ()

  let reduce ~axes rs y =
    let* () = check_axes axes (Array.length y) in
    let kept =
      Array.of_list
        (List.filteri (fun i _ -> not (Iarray.mem i axes)) (Array.to_list y))
    in
    let total = Array.fold_left ( * ) 1 kept in
    let extreme = function Monoid (Max | Min) | Arg _ -> true | _ -> false in
    match Iarray.find_opt (fun a -> y.(a) = 0) axes with
    | Some a when total > 0 && Array.exists extreme rs ->
        errorf "an extreme of no term: no element along axis %d of %a" a pp y
    | Some _ | None -> Ok kept

  let scan ~axis y =
    let* () = check_axes [| axis |] (Array.length y) in
    Ok y

  (* Whether [x] and [y] have one rank and one extent along every axis but
     [axis]. *)
  let off_axis axis x y =
    Array.length x = Array.length y
    && Array.for_all Fun.id (Array.mapi (fun i d -> i = axis || d = y.(i)) x)

  let gather ~axis idx x =
    if axis < 0 || axis >= Array.length x then
      errorf "axis %d of an operand of rank %d" axis (Array.length x)
    else if off_axis axis idx x then Ok idx
    else errorf "positions %a do not fit %a along axis %d" pp idx pp x axis

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

  let assemble y regions =
    let* y = admit "an assembly" y in
    let piece j rs =
      match Nx_array.Move.shape (Slice rs) y with
      | s -> Ok s
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

  (* [Some g] if offsets of shape [o] have [a]'s batch extents, then [g + 1]
     entries. *)
  let groups batch a o =
    let nb = Iarray.length batch in
    let fit = ref (Array.length o = nb + 1) in
    for k = 0 to nb - 1 do
      if !fit && o.(k) <> a.(fst (Iarray.get batch k)) then fit := false
    done;
    if !fit && o.(nb) >= 1 then Some (o.(nb) - 1) else None

  let check_free name side r batch contracting ax =
    if ax < 0 || ax >= r then errorf "no axis %d of %s to segment" ax name
    else if paired side batch contracting ax then
      errorf "axis %d of %s is in a pair and segmented" ax name
    else Ok ()

  (* The groups' count of [segments], checked against the pairs. *)
  let check_segments ~batch ~contracting a b = function
    | None -> Ok 0
    | Some (Rows { offsets; a_axis; b_axis }) -> (
        let* () = check_free "a" fst (Array.length a) batch contracting a_axis in
        let* () = check_free "b" snd (Array.length b) batch contracting b_axis in
        match groups batch a offsets with
        | Some g when g = b.(b_axis) -> Ok g
        | _ ->
            errorf "offsets of shape %a for %d groups" pp offsets b.(b_axis))
    | Some (Contracting { offsets; pair }) -> (
        if pair < 0 || pair >= Iarray.length contracting then
          errorf "no contracting pair %d to segment" pair
        else
          match groups batch a offsets with
          | Some g -> Ok g
          | None -> errorf "offsets of shape %a" pp offsets)

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

  let svd_values a =
    let* batch, m, n = matrix a in
    admit "singular values" (Array.append batch [| min m n |])

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

  let eigh_values a =
    let* batch, _, n = square a in
    Ok (Array.append batch [| n |])

  let eigh a =
    let* w = eigh_values a in
    Ok (w, a)

  let solve_triangular a b =
    let* batch, _, n = square a in
    if Array.length b <> Array.length a
       || Array.sub b 0 (Array.length b - 1) <> Array.append batch [| n |]
    then errorf "right-hand sides %a for a matrix %a" pp b pp a
    else Ok b
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
end
