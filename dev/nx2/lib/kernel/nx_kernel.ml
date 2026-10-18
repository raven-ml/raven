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

type monoid = Sum | Prod | Max | Min | Logsumexp
type extreme = Max | Min
type reduction = Monoid of monoid | Moments | Arg of extreme

let accepts (type v s) r (dt : (v, s) D.t) =
  match (r, D.kind dt) with
  | Monoid (Sum | Prod), D.Boolean -> false
  | (Monoid Logsumexp | Moments), D.Float -> true
  | (Monoid Logsumexp | Moments), _ -> false
  | _ -> true

type order = Chain | Lane
type combine = Set | Add | Max | Min

let accumulates (type v s) (dt : (v, s) D.t) =
  match D.kind dt with
  | D.Boolean -> false
  | D.Float -> D.bits dt >= 32
  | D.Complex | D.Signed | D.Unsigned -> true

type direction = Forward | Inverse

let same_precision (type v s c) (x : (v, s) D.t) (c : (Complex.t, c) D.t) =
  match (x, c) with
  | (Float32 | Complex64), Complex64 | (Float64 | Complex128), Complex128 ->
      true
  | _ -> false

type triangle = Lower | Upper
type factors = Reduced | Complete

type ('p, 'x, _) vectors =
  | No_vectors : ('p, 'x, unit) vectors
  | Vectors : 'p -> ('p, 'x, 'x) vectors

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
    let rec go i n =
      i = Array.length s
      ||
      let d = s.(i) in
      d >= 0
      && if d = 0 then go (i + 1) n else d <= L.max_numel / n && go (i + 1) (n * d)
    in
    go 0 1

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

  let load = function
    | Plain x -> Ok (L.shape (Nx_array.layout x))
    | Padded { x; pad; _ } -> padded pad (L.shape (Nx_array.layout x))

  let loaded ls =
    if ls = [||] then Error "a loop with no load has no shape of its own"
    else
      let* first = load ls.(0) in
      let rec go k =
        if k = Array.length ls then Ok first
        else
          let* y = load ls.(k) in
          if y <> first then
            errorf "load %d has shape %a, load 0 %a" k pp y pp first
          else go (k + 1)
      in
      go 1

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
    let terms = Iarray.fold_left (fun n a -> n * y.(a)) 1 axes in
    let total = Array.fold_left ( * ) 1 kept in
    let extreme = function Monoid (Max | Min) | Arg _ -> true | _ -> false in
    if terms = 0 && total > 0 && Array.exists extreme rs then
      Error "an extreme of no term"
    else Ok kept

  let scan ~axis y =
    let* () = check_axes [| axis |] (Array.length y) in
    Ok y

  (* Whether [x] and [y] have one rank, above [axis], and one extent along
     every other axis. *)
  let off_axis axis x y =
    Array.length x = Array.length y
    && axis >= 0
    && axis < Array.length x
    && Array.for_all Fun.id (Array.mapi (fun i d -> i = axis || d = y.(i)) x)

  let gather ~axis idx x =
    if off_axis axis idx x then Ok idx
    else errorf "positions %a do not fit %a along axis %d" pp idx pp x axis

  let scatter ~axis into idx u =
    if idx <> u then errorf "positions %a, updates %a" pp idx pp u
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

  let contract ~batch ~contracting a b =
    let ra = Array.length a and rb = Array.length b in
    (* Whether a pair names each axis: [a]'s, then [b]'s. *)
    let used = Array.make (ra + rb) false in
    let pair what k (i, j) =
      if i < 0 || i >= ra then errorf "%s pair %d: no axis %d of a" what k i
      else if j < 0 || j >= rb then errorf "%s pair %d: no axis %d of b" what k j
      else if used.(i) then errorf "axis %d of a is in two pairs" i
      else if used.(ra + j) then errorf "axis %d of b is in two pairs" j
      else if a.(i) <> b.(j) then
        errorf "%s pair %d: extents %d and %d" what k a.(i) b.(j)
      else begin
        used.(i) <- true;
        used.(ra + j) <- true;
        Ok ()
      end
    in
    let rec pairs what ps k =
      if k = Iarray.length ps then Ok ()
      else
        let* () = pair what k (Iarray.get ps k) in
        pairs what ps (k + 1)
    in
    let* () = pairs "batch" batch 0 in
    let* () = pairs "contracting" contracting 0 in
    (* The batch extents, then the free ones: [a]'s, then [b]'s. *)
    let nb = Iarray.length batch in
    let paired = nb + Iarray.length contracting in
    let y = Array.make (nb + ra + rb - (2 * paired)) 0 in
    for k = 0 to nb - 1 do
      y.(k) <- a.(fst (Iarray.get batch k))
    done;
    let k = ref nb in
    for ax = 0 to ra + rb - 1 do
      if not used.(ax) then begin
        y.(!k) <- (if ax < ra then a.(ax) else b.(ax - ra));
        incr k
      end
    done;
    if Array.length y > L.max_rank then
      errorf "a result of rank %d" (Array.length y)
    else admit "a contraction" y

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

  let svd : type r.
      (factors, t * t, r) vectors -> t -> (t * r, string) Stdlib.result =
   fun vs a ->
    let* batch, m, n = matrix a in
    let y tail = Array.append batch tail in
    let k = min m n in
    let* s = admit "singular values" (y [| k |]) in
    (* The annotation lets each case refine [r] inside [let*]'s function. *)
    (match vs with
     | No_vectors -> Ok (s, ())
     | Vectors f ->
         let u, vh =
           match f with
           | Reduced -> (y [| m; k |], y [| k; n |])
           | Complete -> (y [| m; m |], y [| n; n |])
         in
         let* u = admit "u" u in
         let* vh = admit "vh" vh in
         Ok (s, (u, vh))
      : (t * r, string) Stdlib.result)

  let eigh : type r.
      (unit, t, r) vectors -> t -> (t * r, string) Stdlib.result =
   fun vs a ->
    let* batch, _, n = square a in
    let w = Array.append batch [| n |] in
    (match vs with No_vectors -> Ok (w, ()) | Vectors () -> Ok (w, a)
      : (t * r, string) Stdlib.result)

  let eig = eigh

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
  val apply0 : Prog.op0 -> dst:('v, 's) a -> answer
  val apply1 : Prog.op1 -> dst:('v, 's) a -> ('a, 'b) a -> answer
  val apply2 : Prog.op2 -> dst:('v, 's) a -> ('a, 'b) a -> ('a, 'b) a -> answer

  val apply3 :
    Prog.op3 -> dst:('v, 's) a -> ('c, 'e) a -> ('v, 's) a -> ('v, 's) a -> answer

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
  val contract_order : rows:int -> columns:int -> order

  val contract :
    order:order ->
    batch:(int * int) iarray ->
    contracting:(int * int) iarray ->
    acc:Nx_array.Dtype.any ->
    dst:('v, 's) a ->
    init:('v, 's) a option ->
    ('a, 'b) a ->
    ('c, 'e) a ->
    answer

  val c2c :
    direction -> axes:int iarray -> dst:(Complex.t, 's) a -> (Complex.t, 's) a -> answer

  val r2c : axes:int iarray -> dst:(Complex.t, 'c) a -> (float, 'r) a -> answer
  val c2r : axes:int iarray -> dst:(float, 'r) a -> (Complex.t, 'c) a -> answer
  val cholesky : triangle -> dst:('v, 's) a -> ('v, 's) a -> answer
  val lu : lu:('v, 's) a -> pivots:index -> perm:index -> ('v, 's) a -> answer
  val qr : q:('v, 's) a -> r:('v, 's) a -> ('v, 's) a -> answer

  val svd :
    s:('v, 's) a ->
    ?vectors:(u:('v, 's) a * vh:('v, 's) a) ->
    ('v, 's) a ->
    answer

  val eigh : w:('v, 's) a -> ?v:('v, 's) a -> ('v, 's) a -> answer

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

module Contract_view = struct
  type operand = A | B | Init | Dst
  type axis = Batch | Row | Column | Contracted

  (* A view is an int array: the extents by axis, the offsets by operand and
     the strides by operand then axis, as nx_contract_view.h's
     nx_contract_view lays them out, then 1 iff it has an init. Operands and
     axes by index: A 0, B 1, Init 2, Dst 3; Batch 0, Row 1, Column 2,
     Contracted 3. C makes it (nx_kernel_view). *)
  type t = int array

  let at_offset = 4
  let at_stride = 8
  let at_has_init = 24
  let operand_index = function A -> 0 | B -> 1 | Init -> 2 | Dst -> 3

  let axis_index = function
    | Batch -> 0
    | Row -> 1
    | Column -> 2
    | Contracted -> 3

  let has_init v = v.(at_has_init) <> 0

  external v :
    batch:(int * int) iarray ->
    contracting:(int * int) iarray ->
    dst:('v, 's) Nx_array.t ->
    init:('v, 's) Nx_array.t option ->
    ('a, 'b) Nx_array.t ->
    ('c, 'e) Nx_array.t ->
    (t, Nx_array.answer) result = "nx_kernel_view_byte" "nx_kernel_view"

  let extent v x = v.(axis_index x)

  let check_init fn v o =
    if o = Init && not (has_init v) then
      invalid_arg ("Nx_kernel.Contract_view." ^ fn ^ ": the view has no init")

  let offset v o =
    check_init "offset" v o;
    v.(at_offset + operand_index o)

  let stride v o x =
    check_init "stride" v o;
    (match (o, x) with
    | A, Column | B, Row | (Init | Dst), Contracted ->
        invalid_arg
          "Nx_kernel.Contract_view.stride: an axis the operand does not have"
    | _ -> ());
    v.(at_stride + (4 * operand_index o) + axis_index x)
end
