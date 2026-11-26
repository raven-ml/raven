(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Linear algebra through Nx, in nx.mli's order: the dtypes and rounding of the
   section, then each function's defining relation within the bound nx_cpu.mli
   states for the host's kernels, its shapes, a failing matrix NaN in its batch
   alone, its edges and its refusals; then Prim.linalg's routines, through a
   kernel library that records them. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module P = Nx_kernel.Prog

(* Dtypes *)

let fulls = D.[ Any Float32; Any Float64; Any Complex64; Any Complex128 ]

(* The float of a dtype's precision, the routines' real results'. *)
let real_of (type v s) (dt : (v, s) D.t) : Nx_kernel.any_float =
  match dt with
  | D.Float32 | D.Complex64 -> Any_float D.Float32
  | _ -> Any_float D.Float64

(* The complex dtype of a dtype's precision, eig's results'. *)
type cplx = Cplx : (Complex.t, 'c) D.t -> cplx

let complex_of (type v s) (dt : (v, s) D.t) =
  match dt with
  | D.Float32 | D.Complex64 -> Cplx D.Complex64
  | _ -> Cplx D.Complex128

(* Every float format, each a dtype a real result may take. *)
let float_formats =
  D.
    [
      Nx_kernel.Any_float Float64;
      Any_float Float32;
      Any_float Float16;
      Any_float Bfloat16;
      Any_float Float8_e4m3fn;
      Any_float Float8_e5m2;
      Any_float Float4_e2m1fn;
    ]

(* Bounds

   nx_cpu.mli's: u the dtype's unit roundoff, η its least subnormal, γ(k) = k u
   / (1 - k u) and ε(k) = k η, k doubled on complex dtypes. *)

let unit_roundoff (type v s) (dt : (v, s) D.t) =
  match dt with D.Float32 | D.Complex64 -> 0x1p-24 | _ -> 0x1p-53

let least_subnormal (type v s) (dt : (v, s) D.t) =
  match dt with D.Float32 | D.Complex64 -> 0x1p-149 | _ -> 0x1p-1074

let gamma u k =
  let ku = Float.of_int k *. u in
  ku /. (1. -. ku)

let doubled dt k = if D.is D.Complex dt then 2 * k else k
let g dt k = gamma (unit_roundoff dt) (doubled dt k)
let e dt k = Float.of_int (doubled dt k) *. least_subnormal dt

(* The check's own rounding: a complex128 sum of [k] products errs by at most
   γ(2k + 4) at float64 of the sum of their magnitudes. *)
let here k = gamma 0x1p-53 ((2 * k) + 4)

(* Values *)

let c128 x = Nx.cast Nx.complex128 x
let mag x = Nx.magnitude Nx.float64 (c128 x)
let adj x = Nx.conjugate (Nx.matrix_transpose x)

(* Each matrix's Frobenius norm, and each column's 2-norm, at float64. *)
let fro x = Nx.sqrt (Nx.sum ~axes:[ -2; -1 ] (Nx.square (mag x)))
let column_norms x = Nx.sqrt (Nx.sum ~axes:[ -2 ] (Nx.square (mag x)))

(* [x]'s elements as floats: a complex one's real parts, then its imaginary
   ones. Float64 holds every float format's values. *)
let parts (type v s) (x : (v, s, Nx.host) Nx.t) =
  match D.kind (Nx.dtype x) with
  | D.Complex ->
      let z = c128 x in
      Array.append
        (Nx.to_array (Nx.real Nx.float64 z))
        (Nx.to_array (Nx.imag Nx.float64 z))
  | _ -> Nx.to_array (Nx.cast Nx.float64 x)

(* The greatest magnitude, NaN where one is. *)
let worst x =
  Array.fold_left
    (fun w v ->
      if Float.is_nan v || Float.is_nan w then Float.nan else Float.max w v)
    0.
    (Nx.to_array (mag x))

(* Each element of [err] at most its element of [bound], NaN in neither. *)
let within ~msg bound err =
  let b = Nx.to_array (Nx.broadcast_to (Nx.shape err) bound) in
  Array.iteri
    (fun i x ->
      let msg = Printf.sprintf "%s, element %d" msg i in
      if Float.is_nan x || Float.is_nan b.(i) then
        failf "%s: %h against a bound of %h" msg x b.(i);
      at_most ~msg float_exact ~than:b.(i) x)
    (Nx.to_array err)

let nans x = Array.map (fun _ -> Float.nan) x

(* [y] of a batch of two: its first matrix holds the bits of [x], the first
   computed alone, and its second only NaN. *)
let alone ~msg x y =
  equal
    ~msg:(msg ^ ": the finite matrix")
    (array float_exact) (parts x)
    (parts (Nx.get [ 0 ] y));
  let bad = parts (Nx.get [ 1 ] y) in
  equal ~msg:(msg ^ ": the failing matrix") (array float_exact) (nans bad) bad

let identity dt lead n =
  Nx.broadcast_to (Array.append lead [| n; n |]) (Nx.eye dt n)

(* [v] scaled along its last axis by [w]: [v · diag w]. *)
let columns_by v w = Nx.mul v (Nx.unsqueeze ~axes:[ -2 ] w)
let f64 s xs = Nx.create Nx.float64 s xs
let numel s = Array.fold_left ( * ) 1 s
let lead s = Array.sub s 0 (Array.length s - 2)
let rows s = s.(Array.length s - 2)
let cols s = s.(Array.length s - 1)
let invalid_by ~by f = raises_match (Exn.invalid_arg ~sub:(by ^ ": ")) f

(* A value of [dt] of shape [s], of real parts [re] and imaginary parts [im]
   where [dt] is complex, each rounded to [dt]. *)
let operand (type v s) (dt : (v, s) D.t) s re im : (v, s, Nx.host) Nx.t =
  match D.kind dt with
  | D.Float -> Nx.cast dt (f64 s re)
  | D.Complex -> Nx.complex dt ~re:(f64 s re) ~im:(f64 s im)
  | D.Signed | D.Unsigned | D.Boolean -> invalid_arg "operand"

(* How a drawn matrix is laid out: contiguous, as the transpose of a contiguous
   one, or one matrix broadcast along a batch of two. *)
type layout = Plain | Transposed | Broadcast

let laid (type v s) (dt : (v, s) D.t) layout s re im : (v, s, Nx.host) Nx.t =
  match layout with
  | Plain -> operand dt s re im
  | Transposed ->
      let t = Array.copy s and r = Array.length s in
      t.(r - 2) <- s.(r - 1);
      t.(r - 1) <- s.(r - 2);
      Nx.matrix_transpose (operand dt t re im)
  | Broadcast ->
      let one = Array.copy s in
      one.(0) <- 1;
      let k = numel one in
      Nx.broadcast_to s (operand dt one (Array.sub re 0 k) (Array.sub im 0 k))

type case =
  | Case : ('v, 's) D.t * int array * layout * float array * float array -> case

let pp_case ppf (Case (dt, s, l, _, _)) =
  let l =
    match l with
    | Plain -> ""
    | Transposed -> ", transposed"
    | Broadcast -> ", broadcast"
  in
  Format.fprintf ppf "%a [%s]%s" D.pp dt
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))
    l

(* Matrices of up to 5 rows, [columns m] columns for [m] rows, behind a batch of
   none, one or two, of the four dtypes, each element's parts in [-1, 1]. A
   broadcast layout has a batch of two. *)
let shaped columns =
  Gen.with_pp pp_case
    (let open Gen in
     let* (D.Any dt) = of_list fulls in
     let* layout = of_list [ Plain; Transposed; Broadcast ] in
     let* lead =
       if layout = Broadcast then constant [| 2 |]
       else of_list [ [||]; [| 0 |]; [| 1 |]; [| 2 |] ]
     in
     let* m = int_range 0 5 in
     let* n = columns m in
     let s = Array.append lead [| m; n |] in
     let k = numel s in
     let+ re = array ~size:(constant k) (float_range (-1.) 1.)
     and+ im = array ~size:(constant k) (float_range (-1.) 1.) in
     Case (dt, s, layout, re, im))

(* Matrices of up to 5 rows and columns, and square ones. *)
let matrices () = shaped (fun _ -> Gen.int_range 0 5)
let square_matrices () = shaped Gen.constant

(* Demands that a law of square matrices meet each shape the generator draws: a
   square matrix, an empty one, and a batch of two. *)
let cover_square_shapes s =
  cover "square" (rows s = cols s);
  cover "empty" (Array.exists (( = ) 0) s);
  cover "a batch of two" (Array.length s = 3 && s.(0) = 2)

(* The same, and wide and tall matrices. *)
let cover_shapes s =
  cover "wide" (rows s < cols s);
  cover "tall" (rows s > cols s);
  cover_square_shapes s

let cover_layout l =
  cover "a transposed view" (l = Transposed);
  cover "a broadcast view" (l = Broadcast)

(* [x] plus [c] on its main diagonal, in its dtype. *)
let shifted c x =
  let s = Nx.shape x in
  let eye = Nx.eye ~m:(cols s) Nx.float64 (rows s) in
  Nx.add x (Nx.cast (Nx.dtype x) (Nx.mul_s eye c))

(* [g gᴴ + (n + 1) I]: Hermitian, positive definite and well conditioned. *)
let definite g =
  shifted (Float.of_int (cols (Nx.shape g) + 1)) (Nx.matmul g (adj g))

(* [g] plus 6 on its diagonal: its least singular value is at least [6 - ‖g‖₂],
   far from zero for the drawn matrices. *)
let conditioned g = shifted 6. g

(* The parts of noise in the triangle a [triangle] read leaves unread, from [re]
   and [im], and of an imaginary noise [re] on the diagonal, whose imaginary
   parts no read takes. *)
let unread triangle s re im =
  let n = cols s in
  let nre = Array.make (Array.length re) 0. in
  let nim = Array.make (Array.length re) 0. in
  Array.iteri
    (fun k _ ->
      let i = k / n mod n and j = k mod n in
      let off = match triangle with Nx.Lower -> j > i | Nx.Upper -> i > j in
      if off then begin
        nre.(k) <- re.(k);
        nim.(k) <- im.(k)
      end
      else if i = j then nim.(k) <- re.(k))
    re;
  (nre, nim)

(* The Hermitian matrix that [x]'s lower triangle and the real parts of its
   diagonal name, at complex128. *)
let named x =
  let z = c128 x in
  let below = Nx.tril ~k:(-1) z in
  let d = c128 (Nx.real Nx.float64 (Nx.diagonal z)) in
  let n = cols (Nx.shape x) in
  Nx.add
    (Nx.add below (adj below))
    (Nx.mul (Nx.eye Nx.complex128 n) (Nx.unsqueeze ~axes:[ -1 ] d))

(* [x]'s drawn noise in its unread upper triangle and diagonal's imaginary
   parts, in [x]'s dtype. *)
let with_noise (type v s) (x : (v, s, Nx.host) Nx.t) re im =
  let s = Nx.shape x in
  let nre, nim = unread Nx.Lower s im re in
  Nx.add x (operand (Nx.dtype x) s nre nim)

(* Linear algebra *)

(* [x]'s values along their last axis of [k], each at least, or at most, the one
   before. *)
let ascending ~msg k x =
  let v = Nx.to_array (Nx.cast Nx.float64 x) in
  Array.iteri
    (fun i y -> if i mod k > 0 then at_least ~msg float_exact ~than:v.(i - 1) y)
    v

let descending ~msg k x =
  let v = Nx.to_array (Nx.cast Nx.float64 x) in
  Array.iteri
    (fun i y -> if i mod k > 0 then at_most ~msg float_exact ~than:v.(i - 1) y)
    v

let not_negative ~msg x =
  Array.iter
    (greater ~msg float_exact ~than:(-0.))
    (Nx.to_array (Nx.cast Nx.float64 x))

let test_rounded_once =
  prop "a result in dt is its operand's precision's, rounded once"
    (square_matrices ()) (fun (Case (dt, s, _, re, im)) ->
      cover_square_shapes s;
      let g = operand dt s re im in
      let a = Nx.add g (adj g) in
      let (Any_float r) = real_of dt in
      let rounded msg want x =
        equal ~msg (array float_exact) (parts want) (parts x)
      in
      let _, s_r, _ = Nx.svd r a in
      let values_r = Nx.svdvals r a in
      let w_r, _ = Nx.eigh r a in
      let wh_r = Nx.eigvalsh r a in
      let _, l_r = Nx.slogdet r a in
      let _, res_r, _, ls_r = Nx.lstsq r a a in
      List.iter
        (fun (Nx_kernel.Any_float f) ->
          cover "wider" (D.bits f > D.bits r);
          let msg what = what ^ " in " ^ D.name f in
          let _, s', _ = Nx.svd f a in
          rounded (msg "svd's s") (Nx.cast f s_r) s';
          rounded (msg "svdvals") (Nx.cast f values_r) (Nx.svdvals f a);
          rounded (msg "eigh's w") (Nx.cast f w_r) (fst (Nx.eigh f a));
          rounded (msg "eigvalsh") (Nx.cast f wh_r) (Nx.eigvalsh f a);
          rounded (msg "slogdet's logabs") (Nx.cast f l_r)
            (snd (Nx.slogdet f a));
          let _, res', _, ls' = Nx.lstsq f a a in
          rounded (msg "lstsq's residuals") (Nx.cast f res_r) res';
          rounded (msg "lstsq's s") (Nx.cast f ls_r) ls')
        float_formats;
      let (Cplx c) = complex_of dt in
      let w_c, v_c = Nx.eig c a in
      let values_c = Nx.eigvals c a in
      List.iter
        (fun (Cplx c') ->
          let msg what = what ^ " in " ^ D.name c' in
          let w', v' = Nx.eig c' a in
          rounded (msg "eig's w") (Nx.cast c' w_c) w';
          rounded (msg "eig's v") (Nx.cast c' v_c) v';
          rounded (msg "eigvals") (Nx.cast c' values_c) (Nx.eigvals c' a))
        [ Cplx D.Complex64; Cplx D.Complex128 ])

(* NaN and the infinities a function states for its matrix, in a dtype that
   lacks them, are what cast gives for them (nx.mli 3366-3368). *)
let test_unheld () =
  let nan = f64 [| 2; 2 |] [| 1.; Float.nan; 0.; 1. |] in
  let singular = f64 [| 2; 2 |] [| 1.; 2.; 2.; 4. |] in
  List.iter
    (fun (Nx_kernel.Any_float f) ->
      let msg what = what ^ " in " ^ D.name f in
      let cast v = parts (Nx.cast f (Nx.full Nx.float64 [| 2 |] v)) in
      equal ~msg:(msg "svdvals' NaN") (array float_exact) (cast Float.nan)
        (parts (Nx.svdvals f nan));
      equal ~msg:(msg "eigvalsh's NaN") (array float_exact) (cast Float.nan)
        (parts (Nx.eigvalsh f (Nx.matrix_transpose nan)));
      let _, _, _, s = Nx.lstsq f nan (Nx.ones Nx.float64 [| 2 |]) in
      equal ~msg:(msg "lstsq's s") (array float_exact) (cast Float.nan)
        (parts s);
      equal ~msg:(msg "slogdet's -inf") (array float_exact)
        [| (cast Float.neg_infinity).(0) |]
        (parts (snd (Nx.slogdet f singular))))
    D.[ Nx_kernel.Any_float Float8_e4m3fn; Any_float Float4_e2m1fn ]

(* The precision a norm computes at is the wider of its operand's and its
   dtype's, then it rounds once to its dtype: the same norm of the operand cast
   to that precision, cast to the dtype, has the same bits. *)

type precise =
  | Precise :
      ('v, 's) D.t * (float, 'r) D.t * int array * float array * float array
      -> precise

(* Every float format and both complex ones. *)
let operands =
  List.filter (fun (D.Any dt) -> D.is D.Float dt || D.is D.Complex dt) D.all

let precise_case =
  let open Gen in
  with_pp
    (fun ppf (Precise (dt, f, s, _, _)) ->
      Format.fprintf ppf "%a [%s] in %a" D.pp dt
        (String.concat "; " (Array.to_list (Array.map string_of_int s)))
        D.pp f)
    (let* (D.Any dt) = of_list operands in
     let* (Nx_kernel.Any_float f) = of_list float_formats in
     let* s = array ~size:(int_range 2 3) (int_range 0 3) in
     let part =
       array
         ~size:(constant (numel s))
         (frequency
            [
              (6, float_range (-8.) 8.);
              (1, of_list [ 0.; -0.; 1e4; -1e4; Float.infinity; Float.nan ]);
            ])
     in
     let+ re = part and+ im = part in
     Precise (dt, f, s, re, im))

type wide = Wide : ('v, 's, Nx.host) Nx.t -> wide

(* [x] cast to the dtype of its kind of [w]'s precision. *)
let widened (type v s r) (w : (float, r) D.t) (x : (v, s, Nx.host) Nx.t) =
  match (D.kind (Nx.dtype x), w) with
  | D.Complex, D.Float64 -> Wide (Nx.cast Nx.complex128 x)
  | D.Complex, _ -> Wide (Nx.cast Nx.complex64 x)
  | _, _ -> Wide (Nx.cast w x)

let law_precision (Precise (dt, f, s, re, im)) =
  let x = operand dt s re im in
  let operand64 = (D.is D.Float dt && D.bits dt = 64) || D.bits dt = 128 in
  let (Any_float w) =
    if operand64 || D.bits f = 64 then Nx_kernel.Any_float D.Float64
    else Any_float D.Float32
  in
  let (Wide wide) = widened w x in
  cover "complex" (D.is D.Complex dt);
  cover "a narrow operand" (D.is D.Float dt && D.bits dt < 32);
  cover "a dtype wider than the operand" (D.bits f = 64 && not operand64);
  cover "a dtype narrower than its computation" (D.bits f < D.bits w);
  let bits y =
    Array.map Int64.bits_of_float (Nx.to_array (Nx.cast Nx.float64 y))
  in
  let outcome g =
    match g () with y -> Some (bits y) | exception Invalid_argument _ -> None
  in
  let same msg
      (g :
        'r 'v 's.
        (float, 'r) D.t -> ('v, 's, Nx.host) Nx.t -> (float, 'r, Nx.host) Nx.t)
      =
    let direct = outcome (fun () -> g f x)
    and rounded = outcome (fun () -> Nx.cast f (g w wide)) in
    cover "computed" (Option.is_some direct);
    equal ~msg (option (array int64)) rounded direct
  in
  List.iter
    (fun ord ->
      same (Printf.sprintf "vector_norm ord %g" ord) (fun d x ->
          Nx.vector_norm d ~ord ~axes:[ -1 ] x))
    [ 2.; 1.; 3.; 0.5; 0.; -1.; -2.; Float.infinity; Float.neg_infinity ];
  List.iter
    (fun (name, ord) -> same name (fun d x -> Nx.matrix_norm d ~ord x))
    [
      ("fro", `Fro);
      ("nuc", `Nuc);
      ("one", `One);
      ("two", `Two);
      ("inf", `Inf);
      ("negone", `NegOne);
      ("negtwo", `NegTwo);
      ("neginf", `NegInf);
    ];
  if (D.float_format f).infinities then
    List.iter
      (fun (name, p) -> same name (fun d x -> Nx.cond d ~p x))
      [ ("cond one", `One); ("cond two", `Two); ("cond inf", `Inf) ]

(* A bfloat16 operand's norm in float64 is drawn about one case in ten; this
   example holds it on every run. *)
let wider_than_operand =
  Precise
    ( D.Bfloat16,
      D.Float64,
      [| 2; 3 |],
      [| 1.; -2.; 0.5; 3.; 0.25; -1. |],
      Array.make 6 0. )

let test_precision =
  prop ~examples:[ wider_than_operand ]
    "a norm or condition number is its value at the wider precision, rounded \
     once"
    precise_case law_precision

(* The narrow floats are taken by the norms, rank and condition, computed at
   float32: [1; -1; 1]'s norm √3 at float32 is not bfloat16's √3 widened. *)
let test_narrow () =
  let b = Nx.create Nx.bfloat16 [| 3 |] [| 1.; -1.; 1. |] in
  let root3 = Int32.float_of_bits (Int32.bits_of_float (Float.sqrt 3.)) in
  equal ~msg:"bfloat16 at float32" float_exact root3
    (Nx.item [] (Nx.vector_norm Nx.float32 b));
  equal ~msg:"bfloat16 at bfloat16" float_exact 1.734375
    (Nx.item [] (Nx.cast Nx.float32 (Nx.vector_norm Nx.bfloat16 b)));
  equal ~msg:"cond of bfloat16's identity" float_exact 1.
    (Nx.item [] (Nx.cond Nx.bfloat16 (Nx.eye Nx.bfloat16 3)));
  equal ~msg:"cond of float8_e4m3fn's identity at float32" float_exact 1.
    (Nx.item [] (Nx.cond Nx.float32 (Nx.eye Nx.float8_e4m3fn 3)))

(* cholesky *)

(* nx_cpu.mli: L Lᴴ = a + Δ with |Δ_ij| <= γ(n + 1) (|L| |Lᴴ|)_ij + ε(n + 1) (1
   + L_jj). *)
let test_cholesky =
  prop
    "L Lᴴ is the matrix its lower triangle and diagonal's real parts name, \
     within γ(n + 1)"
    (square_matrices ()) (fun (Case (dt, s, _, re, im)) ->
      cover_square_shapes s;
      let n = cols s in
      let x = with_noise (definite (operand dt s re im)) re im in
      let l = Nx.cholesky x in
      let h = named x in
      let ll = Nx.matmul (mag l) (Nx.matrix_transpose (mag l)) in
      let ljj =
        Nx.unsqueeze ~axes:[ -2 ] (Nx.real Nx.float64 (Nx.diagonal (c128 l)))
      in
      let bound =
        Nx.add
          (Nx.mul_s ll (g dt (n + 1) +. here n))
          (Nx.add
             (Nx.mul_s (Nx.add_s ljj 1.) (e dt (n + 1)))
             (Nx.mul_s (mag h) (here n)))
      in
      within ~msg:"|L Lᴴ - a|" bound
        (mag (Nx.sub (Nx.matmul (c128 l) (adj (c128 l))) h)))

let test_cholesky_form =
  prop
    "the other triangle is +0, the diagonal real and positive, and Upper's U \
     is Lᴴ"
    (square_matrices ()) (fun (Case (dt, s, _, re, im)) ->
      cover_square_shapes s;
      let x = with_noise (definite (operand dt s re im)) re im in
      let l = Nx.cholesky x in
      let above = parts (Nx.triu ~k:1 l) in
      equal ~msg:"above the diagonal" (array float_exact)
        (Array.map (fun _ -> 0.) above)
        above;
      let d = Nx.diagonal (c128 l) in
      Array.iter
        (greater ~msg:"a positive diagonal" float_exact ~than:0.)
        (Nx.to_array (Nx.real Nx.float64 d));
      let im = Nx.to_array (Nx.abs (Nx.imag Nx.float64 d)) in
      equal ~msg:"a real diagonal" (array float_exact)
        (Array.map (fun _ -> 0.) im)
        im;
      (* [aᴴ]'s upper triangle names the matrix [a]'s lower one does, and the
         noise lies in its lower one. *)
      (* As values: Lᴴ conjugates a +0 imaginary part into -0. *)
      let values x = Array.map (fun v -> v +. 0.) (parts x) in
      equal ~msg:"U is Lᴴ" (array float_exact)
        (values (adj l))
        (values (Nx.cholesky ~triangle:Upper (adj x))))

let test_cholesky_alone () =
  let definite = f64 [| 2; 2 |] [| 2.; 1.; 1.; 2. |] in
  let l = Nx.cholesky definite in
  alone ~msg:"a negative pivot" l
    (Nx.cholesky (Nx.stack [ definite; f64 [| 2; 2 |] [| 1.; 2.; 2.; 1. |] ]));
  alone ~msg:"a NaN pivot" l
    (Nx.cholesky
       (Nx.stack [ definite; f64 [| 2; 2 |] [| Float.nan; 0.; 0.; 1. |] ]));
  (* NaN in the unread triangle is never read. *)
  equal ~msg:"NaN above" (array float_exact) (parts l)
    (parts (Nx.cholesky (f64 [| 2; 2 |] [| 2.; Float.nan; 1.; 2. |])))

let test_cholesky_raises () =
  invalid_by ~by:"Nx.cholesky" (fun () ->
      Nx.cholesky (Nx.ones Nx.float64 [| 3 |]));
  invalid_by ~by:"Nx.cholesky" (fun () ->
      Nx.cholesky (Nx.ones Nx.float64 [| 2; 3 |]));
  invalid_by ~by:"Nx.cholesky" (fun () ->
      Nx.cholesky (Nx.ones Nx.float16 [| 2; 2 |]));
  invalid_by ~by:"Nx.cholesky" (fun () ->
      Nx.cholesky (Nx.ones Nx.int32 [| 2; 2 |]))

(* qr *)

(* nx_cpu.mli: with g = γ(k (2m + 10)) and h = ε(k (2m + 10)), ‖a_j - q r_j‖₂ <=
   g ‖a_j‖₂ + √m h, each element of qᴴq - I at most g. *)
let test_qr =
  prop
    "q r is a within γ(k (2m + 10)) per column, q's columns orthonormal within \
     it, r upper"
    (matrices ()) (fun (Case (dt, s, layout, re, im)) ->
      cover_shapes s;
      cover_layout layout;
      let a = laid dt layout s re im in
      let m = rows s and n = cols s in
      let k = Stdlib.min m n in
      let gq = g dt (k * ((2 * m) + 10)) and hq = e dt (k * ((2 * m) + 10)) in
      List.iter
        (fun (factors, c, name) ->
          let q, r = Nx.qr ~factors a in
          let msg what = name ^ ": " ^ what in
          equal ~msg:(msg "q's shape") (array int)
            (Array.append (lead s) [| m; c |])
            (Nx.shape q);
          equal ~msg:(msg "r's shape") (array int)
            (Array.append (lead s) [| c; n |])
            (Nx.shape r);
          let bound =
            Nx.add
              (Nx.add_s
                 (Nx.mul_s (column_norms a) gq)
                 (Float.sqrt (Float.of_int m) *. hq))
              (Nx.mul_s (column_norms (Nx.matmul (mag q) (mag r))) (here c))
          in
          within ~msg:(msg "‖a_j - q r_j‖") bound
            (column_norms (Nx.sub (c128 a) (Nx.matmul (c128 q) (c128 r))));
          let bound =
            Nx.add_s
              (Nx.mul_s
                 (Nx.matmul (Nx.matrix_transpose (mag q)) (mag q))
                 (here m))
              gq
          in
          within ~msg:(msg "qᴴq - I") bound
            (mag
               (Nx.sub
                  (Nx.matmul (adj (c128 q)) (c128 q))
                  (identity Nx.complex128 (lead s) c)));
          equal
            ~msg:(msg "r below its diagonal")
            float_exact 0.
            (worst (Nx.tril ~k:(-1) r)))
        [ (Nx.Reduced, k, "reduced"); (Nx.Complete, m, "complete") ])

let test_qr_raises () =
  invalid_by ~by:"Nx.qr" (fun () -> Nx.qr (Nx.ones Nx.float64 [| 3 |]));
  invalid_by ~by:"Nx.qr" (fun () -> Nx.qr (Nx.ones Nx.float16 [| 2; 2 |]))

(* lu *)

(* |U_pp| at each element (i, j) of [a]'s shape [s], p = min (i, j). *)
let pivot_magnitudes s u =
  let m = rows s and n = cols s in
  let k = Stdlib.min m n in
  let d = Nx.to_array (mag (Nx.diagonal u)) in
  f64 s
    (Array.init (numel s) (fun x ->
         let b = x / (m * n) and i = x / n mod m and j = x mod n in
         d.((b * k) + Stdlib.min i j)))

(* nx_cpu.mli: L U = P a + Δ with |Δ_ij| <= γ(2k) (|L| |U|)_ij + ε(2k) (1 +
   |U_pp|), p = min (i, j). *)
let test_lu =
  prop "row i of l u is row perm.{i} of a within γ(2k), l unit lower, u upper"
    (matrices ()) (fun (Case (dt, s, layout, re, im)) ->
      cover_shapes s;
      cover_layout layout;
      let a = laid dt layout s re im in
      let m = rows s and n = cols s in
      let k = Stdlib.min m n in
      let perm, l, u = Nx.lu a in
      equal ~msg:"perm's shape" (array int)
        (Array.append (lead s) [| m |])
        (Nx.shape perm);
      equal ~msg:"l's shape" (array int)
        (Array.append (lead s) [| m; k |])
        (Nx.shape l);
      equal ~msg:"u's shape" (array int)
        (Array.append (lead s) [| k; n |])
        (Nx.shape u);
      let rows_a =
        Nx.take_along_axis ~axis:(-2) (Nx.unsqueeze ~axes:[ -1 ] perm) a
      in
      let lu_abs = Nx.matmul (mag l) (mag u) in
      let bound =
        Nx.add
          (Nx.mul_s lu_abs (g dt (2 * k) +. here k))
          (Nx.add
             (Nx.mul_s (Nx.add_s (pivot_magnitudes s u) 1.) (e dt (2 * k)))
             (Nx.mul_s (mag rows_a) (here k)))
      in
      within ~msg:"|l u - rows|" bound
        (mag (Nx.sub (Nx.matmul (c128 l) (c128 u)) (c128 rows_a)));
      let d = Nx.diagonal l in
      equal ~msg:"l's diagonal" (array float_exact)
        (parts (Nx.ones_like d))
        (parts d);
      equal ~msg:"l above its diagonal" float_exact 0. (worst (Nx.triu ~k:1 l));
      equal ~msg:"u below its diagonal" float_exact 0.
        (worst (Nx.tril ~k:(-1) u));
      if not (D.is D.Complex dt) then
        Array.iter
          (at_most ~msg:"|l| of a real a" float_exact ~than:1.)
          (Nx.to_array (mag l)))

let test_lu_raises () =
  invalid_by ~by:"Nx.lu" (fun () -> Nx.lu (Nx.ones Nx.float64 [| 3 |]));
  invalid_by ~by:"Nx.lu" (fun () -> Nx.lu (Nx.ones Nx.int32 [| 2; 2 |]))

(* svd *)

(* nx_cpu.mli's bound of svd and of its values alone: g = γ(16 max (m, n) + 100)
   and h = ε(the same). *)
let svd_bound dt s =
  let k = (16 * Stdlib.max (rows s) (cols s)) + 100 in
  (g dt k, e dt k)

(* ‖a - u diag s vh‖_F <= g ‖a‖_F + h, ‖uᴴu - I‖_F <= g, ‖vh vhᴴ - I‖_F <= g. *)
let test_svd =
  prop
    "u diag s vh is a within its bound, u's columns and vh's rows orthonormal, \
     s descending and not negative"
    (matrices ()) (fun (Case (dt, s, layout, re, im)) ->
      cover_shapes s;
      cover_layout layout;
      let a = laid dt layout s re im in
      let m = rows s and n = cols s in
      let k = Stdlib.min m n in
      let gs, hs = svd_bound dt s in
      let (Any_float r) = real_of dt in
      let u, sv, vh = Nx.svd r a in
      equal ~msg:"s's shape" (array int)
        (Array.append (lead s) [| k |])
        (Nx.shape sv);
      let usv = Nx.matmul (columns_by (c128 u) (c128 sv)) (c128 vh) in
      let abs_usv =
        Nx.matmul (columns_by (mag u) (Nx.cast Nx.float64 sv)) (mag vh)
      in
      within ~msg:"‖a - u diag s vh‖_F"
        (Nx.add
           (Nx.add_s (Nx.mul_s (fro a) gs) hs)
           (Nx.mul_s (fro abs_usv) (here k)))
        (fro (Nx.sub (c128 a) usv));
      let orthonormal msg q c =
        let qq = Nx.matmul (adj (c128 q)) (c128 q) in
        let abs = Nx.matmul (Nx.matrix_transpose (mag q)) (mag q) in
        within ~msg
          (Nx.add_s (Nx.mul_s (fro abs) (here (rows (Nx.shape q)))) gs)
          (fro (Nx.sub qq (identity Nx.complex128 (lead s) c)))
      in
      orthonormal "uᴴu - I" u k;
      orthonormal "vh vhᴴ - I" (adj vh) k;
      descending ~msg:"descending" k sv;
      not_negative ~msg:"not negative, a zero +0" sv;
      let u, _, vh = Nx.svd ~factors:Complete r a in
      equal ~msg:"complete u's shape" (array int)
        (Array.append (lead s) [| m; m |])
        (Nx.shape u);
      orthonormal "complete uᴴu - I" u m;
      orthonormal "complete vh vhᴴ - I" (adj vh) n)

(* Of [[0; -0]; [-0; -2]] the values are 2 and +0, and of zeros of -0, +0. *)
let test_svd_zero () =
  let bits x = Array.map Int64.bits_of_float (Nx.to_array x) in
  let a = f64 [| 2; 2 |] [| 0.; -0.; -0.; -2. |] in
  let _, s, _ = Nx.svd Nx.float64 a in
  let want = Array.map Int64.bits_of_float [| 2.; 0. |] in
  equal ~msg:"svd" (array int64) want (bits s);
  equal ~msg:"svdvals" (array int64) want (bits (Nx.svdvals Nx.float64 a));
  let zeros = f64 [| 2; 3; 2 |] (Array.make 12 (-0.)) in
  let _, s, _ = Nx.svd Nx.float64 zeros in
  equal ~msg:"svd of -0" (array int64) (Array.make 4 0L) (bits s);
  equal ~msg:"svdvals of -0" (array int64) (Array.make 4 0L)
    (bits (Nx.svdvals Nx.float64 zeros))

let test_svd_alone () =
  let finite = f64 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let u, s, vh = Nx.svd Nx.float64 finite in
  List.iter
    (fun (name, v) ->
      let u', s', vh' =
        Nx.svd Nx.float64
          (Nx.stack [ finite; f64 [| 2; 2 |] [| 1.; v; 0.; 1. |] ])
      in
      alone ~msg:(name ^ ", u") u u';
      alone ~msg:(name ^ ", s") s s';
      alone ~msg:(name ^ ", vh") vh vh')
    [ ("NaN", Float.nan); ("an infinity", Float.infinity) ]

let test_svd_raises () =
  invalid_by ~by:"Nx.svd" (fun () ->
      Nx.svd Nx.float64 (Nx.ones Nx.float64 [| 3 |]));
  invalid_by ~by:"Nx.svd" (fun () ->
      Nx.svd Nx.float32 (Nx.ones Nx.float16 [| 2; 2 |]))

(* svdvals *)

(* Each value alone, and each of svd's s, lies within g ‖a‖₂ + h of the exact
   one (nx_cpu.mli): the two within twice that, ‖a‖_F >= ‖a‖₂. *)
let test_svdvals =
  prop "svdvals is svd's s within its bound, descending and not negative"
    (matrices ()) (fun (Case (dt, s, layout, re, im)) ->
      cover_shapes s;
      cover_layout layout;
      let a = laid dt layout s re im in
      let k = Stdlib.min (rows s) (cols s) in
      let gs, hs = svd_bound dt s in
      let (Any_float r) = real_of dt in
      let sv = Nx.svdvals r a in
      equal ~msg:"shape" (array int)
        (Array.append (lead s) [| k |])
        (Nx.shape sv);
      let _, s_full, _ = Nx.svd r a in
      within ~msg:"|svdvals - s|"
        (Nx.unsqueeze ~axes:[ -1 ]
           (Nx.add_s (Nx.mul_s (fro a) (2. *. gs)) (2. *. hs)))
        (mag (Nx.sub (c128 sv) (c128 s_full)));
      descending ~msg:"descending" k sv;
      not_negative ~msg:"not negative, a zero +0" sv)

let test_svdvals_edges () =
  let svdvals a = Nx.svdvals Nx.float64 a in
  equal ~msg:"1 × 1" (array float_exact) [| 3. |]
    (Nx.to_array (svdvals (f64 [| 1; 1 |] [| -3. |])));
  equal ~msg:"k = 0" (array int) [| 2; 0 |]
    (Nx.shape (svdvals (f64 [| 2; 0; 3 |] [||])));
  equal ~msg:"no matrix" (array int) [| 0; 2 |]
    (Nx.shape (svdvals (f64 [| 0; 2; 3 |] [||])))

let test_svdvals_alone () =
  let finite = f64 [| 2; 2 |] [| 2.; 1.; 1.; 3. |] in
  let s = Nx.svdvals Nx.float64 finite in
  alone ~msg:"NaN" s
    (Nx.svdvals Nx.float64
       (Nx.stack [ finite; f64 [| 2; 2 |] [| 1.; 0.; Float.nan; 1. |] ]));
  alone ~msg:"an infinity" s
    (Nx.svdvals Nx.float64
       (Nx.stack [ finite; f64 [| 2; 2 |] [| Float.infinity; 0.; 1.; 1. |] ]))

let test_svdvals_raises () =
  invalid_by ~by:"Nx.svdvals" (fun () ->
      Nx.svdvals Nx.float64 (Nx.ones Nx.float64 [| 3 |]));
  invalid_by ~by:"Nx.svdvals" (fun () ->
      Nx.svdvals Nx.float32 (Nx.ones Nx.float16 [| 2; 2 |]))

(* eig *)

(* [re] and [im] of square matrices of shape [s], each element below the
   diagonal given the magnitude of its mirror above and the signs of its own
   parts: a matrix with [|a_ij| = |a_ji|], which balancing leaves alone
   (nx_cpu.mli), so that its bound holds in [a]'s own coordinates. *)
let mirrored s re im =
  let n = cols s in
  let re = Array.copy re and im = Array.copy im in
  for b = 0 to (if n = 0 then 0 else Array.length re / (n * n)) - 1 do
    for i = 0 to n - 1 do
      for j = i + 1 to n - 1 do
        let up = (b * n * n) + (i * n) + j
        and down = (b * n * n) + (j * n) + i in
        re.(down) <- Float.copy_sign (Float.abs re.(up)) re.(down);
        im.(down) <- Float.copy_sign (Float.abs im.(up)) im.(down)
      done
    done
  done;
  (re, im)

(* nx_cpu.mli, for a matrix balancing leaves alone: with g = γ(16n) and h =
   ε(16n), each column has |‖v_j‖₂ - 1| <= g and ‖a v_j - w_j v_j‖₂ <= (g ‖a‖_F
   + h) ‖v_j‖₂. The norms here are computed at float64 from complex128 sums of
   up to n products. *)
let test_eig =
  prop
    "a v is v diag w within γ(16n), its columns of unit norm, of a matrix \
     balancing leaves alone"
    (square_matrices ()) (fun (Case (dt, s, layout, re, im)) ->
      cover_square_shapes s;
      cover_layout layout;
      let n = cols s in
      let re, im = mirrored s re im in
      let a = laid dt layout s re im in
      let (Cplx c) = complex_of dt in
      let w, v = Nx.eig c a in
      equal ~msg:"w's shape" (array int)
        (Array.append (lead s) [| n |])
        (Nx.shape w);
      let gs = g dt (16 * n) and hs = e dt (16 * n) in
      let vj = column_norms v in
      within ~msg:"|‖v_j‖₂ - 1|"
        (Nx.add_s (Nx.mul_s vj (here n)) gs)
        (Nx.abs (Nx.sub_s vj 1.));
      let residual =
        Nx.sub (Nx.matmul (c128 a) (c128 v)) (columns_by (c128 v) (c128 w))
      in
      let fa = Nx.unsqueeze ~axes:[ -1 ] (fro a) in
      let products =
        Nx.add (Nx.matmul (mag a) (mag v)) (columns_by (mag v) (mag w))
      in
      within ~msg:"‖a v_j - w_j v_j‖₂"
        (Nx.add
           (Nx.mul vj (Nx.add_s (Nx.mul_s fa gs) hs))
           (Nx.mul_s (column_norms products) (here n)))
        (column_norms residual))

let test_eig_alone () =
  let finite = f64 [| 2; 2 |] [| 2.; 1.; 1.; 3. |] in
  let w, v = Nx.eig Nx.complex128 finite in
  let bad = f64 [| 2; 2 |] [| 1.; Float.nan; Float.nan; 1. |] in
  let w', v' = Nx.eig Nx.complex128 (Nx.stack [ finite; bad ]) in
  alone ~msg:"w" w w';
  alone ~msg:"v" v v'

let test_eig_raises () =
  invalid_by ~by:"Nx.eig" (fun () ->
      Nx.eig Nx.complex64 (Nx.ones Nx.float32 [| 3 |]));
  invalid_by ~by:"Nx.eig" (fun () ->
      Nx.eig Nx.complex128 (Nx.ones Nx.float64 [| 2; 3 |]));
  invalid_by ~by:"Nx.eig" (fun () ->
      Nx.eig Nx.complex64 (Nx.ones Nx.float16 [| 2; 2 |]));
  invalid_by ~by:"Nx.eig" (fun () ->
      Nx.eig Nx.complex128 (Nx.ones Nx.int32 [| 2; 2 |]))

(* eigvals *)

(* The least singular value of [a - λ I] for each λ of [w], at float64, through
   the real form [[x_r -x_i]; [x_i x_r]] of [a - λ I], whose singular values are
   its own, each twice. *)
let least_singular n a w =
  let lambda = Nx.unsqueeze ~axes:[ -1 ] (Nx.unsqueeze ~axes:[ -1 ] (c128 w)) in
  let x =
    Nx.sub
      (Nx.unsqueeze ~axes:[ -3 ] (c128 a))
      (Nx.mul lambda (Nx.eye Nx.complex128 n))
  in
  let xr = Nx.real Nx.float64 x and xi = Nx.imag Nx.float64 x in
  let row l = Nx.concatenate ~axis:(-1) l in
  let x = Nx.concatenate ~axis:(-2) [ row [ xr; Nx.neg xi ]; row [ xi; xr ] ] in
  Nx.matrix_norm Nx.float64 ~ord:`NegTwo x

(* nx_cpu.mli: w is the spectrum of a + E with ‖E‖_F <= δ = g ‖a‖_F + h, g =
   γ(16n), h = ε(16n), for a matrix balancing leaves alone. So each value has
   σ_min (a - w_j I) <= δ; Σ w is tr (a + E), within √n δ of tr a; and Σ w² is
   tr ((a + E)²), within 2 ‖a‖_F δ + δ² of tr (a²). The checks add float64's svd
   bound of the real form, γ(32n + 100) of its Frobenius norm, and their own
   sums' rounding. *)
let test_eigvals =
  prop "each is an eigenvalue of a matrix within γ(16n) of a, with multiplicity"
    (square_matrices ()) (fun (Case (dt, s, layout, re, im)) ->
      cover_square_shapes s;
      cover_layout layout;
      let n = cols s in
      let re, im = mirrored s re im in
      let a = laid dt layout s re im in
      let (Cplx c) = complex_of dt in
      let w = Nx.eigvals c a in
      equal ~msg:"shape" (array int)
        (Array.append (lead s) [| n |])
        (Nx.shape w);
      let fa = fro a in
      let delta = Nx.add_s (Nx.mul_s fa (g dt (16 * n))) (e dt (16 * n)) in
      let abs_w = Nx.sum ~axes:[ -1 ] (mag w) in
      within ~msg:"Σ w"
        (Nx.add
           (Nx.mul_s delta (Float.sqrt (Float.of_int n)))
           (Nx.mul_s (Nx.add abs_w fa) (here n)))
        (mag (Nx.sub (Nx.sum ~axes:[ -1 ] (c128 w)) (Nx.trace (c128 a))));
      let squares = Nx.sum ~axes:[ -1 ] (Nx.square (mag w)) in
      within ~msg:"Σ w²"
        (Nx.add
           (Nx.add (Nx.mul (Nx.mul_s fa 2.) delta) (Nx.square delta))
           (Nx.mul_s (Nx.add squares (Nx.square fa)) (here n)))
        (mag
           (Nx.sub
              (Nx.sum ~axes:[ -1 ] (Nx.square (c128 w)))
              (Nx.trace (Nx.matmul (c128 a) (c128 a)))));
      if n > 0 then
        let form =
          Nx.add fa
            (Nx.mul_s
               (Nx.max ~axes:[ -1 ] (mag w))
               (Float.sqrt (Float.of_int n)))
        in
        let own =
          Nx.mul_s form (Float.sqrt 2. *. gamma 0x1p-53 ((32 * n) + 100))
        in
        within ~msg:"σ_min (a - w_j I)"
          (Nx.unsqueeze ~axes:[ -1 ] (Nx.add delta own))
          (least_singular n a w))

(* [x]'s values at complex128, each matrix's [n] sorted by real part. *)
let sorted_by_real n x =
  let z = Nx.to_array (c128 x) in
  let by (a : Complex.t) (b : Complex.t) = Float.compare a.re b.re in
  let chunk b = List.sort by (Array.to_list (Array.sub z (b * n) n)) in
  if n = 0 then []
  else List.concat_map chunk (List.init (Array.length z / n) Fun.id)

(* A Hermitian matrix's eigenvalues are real, and each value of a + E lies
   within ‖E‖₂ <= δ of one of them, with multiplicity: sorted, eigvals' and
   eig's lie within 2δ of each other. *)
let test_eigvals_hermitian =
  prop "of a Hermitian matrix, they are eig's w in some order"
    (square_matrices ()) (fun (Case (dt, s, _, re, im)) ->
      cover_square_shapes s;
      let n = cols s in
      let g' = operand dt s re im in
      let h = Nx.add g' (adj g') in
      let (Cplx c) = complex_of dt in
      let delta =
        Array.map
          (fun f ->
            (2. *. ((f *. g dt (16 * n)) +. e dt (16 * n))) +. (here n *. f))
          (Nx.to_array (fro h))
      in
      let got = sorted_by_real n (Nx.eigvals c h)
      and want = sorted_by_real n (fst (Nx.eig c h)) in
      List.iteri
        (fun i ((x : Complex.t), (y : Complex.t)) ->
          at_most
            ~msg:(Printf.sprintf "value %d" i)
            float_exact
            ~than:delta.(i / n)
            (Nx_support.complex_gap x y))
        (List.combine got want))

let test_eigvals_alone () =
  let finite = f64 [| 2; 2 |] [| 2.; 1.; 1.; 3. |] in
  let w = Nx.eigvals Nx.complex128 finite in
  List.iter
    (fun (name, bad) ->
      alone ~msg:name w
        (Nx.eigvals Nx.complex128 (Nx.stack [ finite; f64 [| 2; 2 |] bad ])))
    [
      ("NaN", [| 1.; 0.; Float.nan; 1. |]);
      ("an infinity", [| Float.infinity; 0.; 1.; 1. |]);
    ]

let test_eigvals_raises () =
  invalid_by ~by:"Nx.eigvals" (fun () ->
      Nx.eigvals Nx.complex128 (Nx.ones Nx.float64 [| 2; 3 |]));
  invalid_by ~by:"Nx.eigvals" (fun () ->
      Nx.eigvals Nx.complex64 (Nx.ones Nx.float16 [| 2; 2 |]))

(* eigh *)

(* nx_cpu.mli: with g = γ(16n) and h = ε(16n), ‖a - v diag w vᴴ‖_F <= g ‖a‖_F +
   h and ‖vᴴv - I‖_F <= g. *)
let eigh_bound dt n = (g dt (16 * n), e dt (16 * n))

(* [2 I] in float32, of equal eigenvalues, once failed the order check. *)
let twice_identity =
  Case (D.Float32, [| 2; 2 |], Plain, [| 1.; 0.; 0.; 1. |], Array.make 4 0.)

let test_eigh =
  prop ~examples:[ twice_identity ]
    "v diag w vᴴ is the matrix its lower triangle and diagonal's real parts \
     name, within γ(16n), v orthonormal, w ascending"
    (square_matrices ()) (fun (Case (dt, s, _, re, im)) ->
      cover_square_shapes s;
      let n = cols s in
      let g' = operand dt s re im in
      let x = with_noise (Nx.add g' (adj g')) re im in
      let (Any_float r) = real_of dt in
      let w, v = Nx.eigh r x in
      let h = named x in
      let gs, hs = eigh_bound dt n in
      let vwv = Nx.matmul (columns_by (c128 v) (c128 w)) (adj (c128 v)) in
      let abs =
        Nx.matmul (columns_by (mag v) (mag w)) (Nx.matrix_transpose (mag v))
      in
      within ~msg:"‖a - v diag w vᴴ‖_F"
        (Nx.add
           (Nx.add_s (Nx.mul_s (fro h) gs) hs)
           (Nx.mul_s (fro abs) (here n)))
        (fro (Nx.sub h vwv));
      let abs = Nx.matmul (Nx.matrix_transpose (mag v)) (mag v) in
      within ~msg:"‖vᴴv - I‖_F"
        (Nx.add_s (Nx.mul_s (fro abs) (here n)) gs)
        (fro
           (Nx.sub
              (Nx.matmul (adj (c128 v)) (c128 v))
              (identity Nx.complex128 (lead s) n)));
      ascending ~msg:"ascending" n w;
      (* [xᴴ]'s upper triangle names the matrix [x]'s lower one does. *)
      let wu, vu = Nx.eigh ~triangle:Upper r (adj x) in
      equal ~msg:"Upper's w" (array float_exact) (parts w) (parts wu);
      equal ~msg:"Upper's v" (array float_exact) (parts v) (parts vu))

let test_eigh_alone () =
  let finite = f64 [| 2; 2 |] [| 2.; 1.; 1.; 3. |] in
  let w, v = Nx.eigh Nx.float64 finite in
  let bad = f64 [| 2; 2 |] [| 1.; 0.; Float.nan; 1. |] in
  let w', v' = Nx.eigh Nx.float64 (Nx.stack [ finite; bad ]) in
  alone ~msg:"w" w w';
  alone ~msg:"v" v v';
  let w', _ = Nx.eigh Nx.float64 (f64 [| 2; 2 |] [| 2.; Float.nan; 1.; 3. |]) in
  equal ~msg:"NaN above is not read" (array float_exact) (parts w) (parts w')

let test_eigh_raises () =
  invalid_by ~by:"Nx.eigh" (fun () ->
      Nx.eigh Nx.float64 (Nx.ones Nx.float64 [| 3 |]));
  invalid_by ~by:"Nx.eigh" (fun () ->
      Nx.eigh Nx.float64 (Nx.ones Nx.float64 [| 2; 3 |]));
  invalid_by ~by:"Nx.eigh" (fun () ->
      Nx.eigh Nx.float32 (Nx.ones Nx.float16 [| 2; 2 |]))

(* eigvalsh *)

(* Each value alone, and each of eigh's w, lies within g ‖a‖₂ + h of the exact
   one (nx_cpu.mli): the two within twice that. *)
let test_eigvalsh =
  prop "eigvalsh is eigh's w within its bound, ascending, from the same part"
    (square_matrices ()) (fun (Case (dt, s, _, re, im)) ->
      cover_square_shapes s;
      let n = cols s in
      let g' = operand dt s re im in
      let x = with_noise (Nx.add g' (adj g')) re im in
      let (Any_float r) = real_of dt in
      let w = Nx.eigvalsh r x in
      equal ~msg:"shape" (array int)
        (Array.append (lead s) [| n |])
        (Nx.shape w);
      let gs, hs = eigh_bound dt n in
      within ~msg:"|eigvalsh - w|"
        (Nx.unsqueeze ~axes:[ -1 ]
           (Nx.add_s (Nx.mul_s (fro (named x)) (2. *. gs)) (2. *. hs)))
        (mag (Nx.sub (c128 w) (c128 (fst (Nx.eigh r x)))));
      ascending ~msg:"ascending" n w;
      equal ~msg:"Upper" (array float_exact) (parts w)
        (parts (Nx.eigvalsh ~triangle:Upper r (adj x))))

let test_eigvalsh_alone () =
  let finite = f64 [| 2; 2 |] [| 2.; 1.; 1.; 3. |] in
  let w = Nx.eigvalsh Nx.float64 finite in
  List.iter
    (fun (name, bad) ->
      alone ~msg:name w
        (Nx.eigvalsh Nx.float64 (Nx.stack [ finite; f64 [| 2; 2 |] bad ])))
    [
      ("NaN", [| 1.; 0.; Float.nan; 1. |]);
      ("an infinity", [| Float.infinity; 0.; 1.; 1. |]);
    ];
  equal ~msg:"NaN above is not read" (array float_exact) (parts w)
    (parts
       (Nx.eigvalsh Nx.float64 (f64 [| 2; 2 |] [| 2.; Float.nan; 1.; 3. |])))

let test_eigvalsh_raises () =
  invalid_by ~by:"Nx.eigvalsh" (fun () ->
      Nx.eigvalsh Nx.float64 (Nx.ones Nx.float64 [| 2; 3 |]));
  invalid_by ~by:"Nx.eigvalsh" (fun () ->
      Nx.eigvalsh Nx.float32 (Nx.ones Nx.float16 [| 2; 2 |]))

(* det *)

(* The parity of the permutation [p] of [0 .. n - 1]: [-1.] where it is odd, as
   the interchanges that make it are. *)
let parity p =
  let n = Array.length p and seen = Array.make (Array.length p) false in
  let cycles = ref 0 in
  for i = 0 to n - 1 do
    if not seen.(i) then begin
      incr cycles;
      let j = ref i in
      while not seen.(!j) do
        seen.(!j) <- true;
        j := Int64.to_int p.(!j)
      done
    end
  done;
  if (n - !cycles) mod 2 = 1 then -1. else 1.

(* Each matrix's pivots, the diagonal of lu's u at complex128, and the parity of
   its perm. *)
let pivots (type v s) (a : (v, s, Nx.host) Nx.t) =
  let perm, _, u = Nx.lu a in
  let n = cols (Nx.shape a) in
  let d = Nx.to_array (c128 (Nx.diagonal u)) and p = Nx.to_array perm in
  List.init
    (numel (lead (Nx.shape a)))
    (fun b -> (Array.sub d (b * n) n, parity (Array.sub p (b * n) n)))

(* The product of n pivots in the dtype errs by at most γ(2n) of the product of
   their magnitudes, doubled on complex numbers, and ε(2n) below the least
   normal number. *)
let test_det =
  prop "det is the product of lu's u diagonal, negated once per interchange"
    (square_matrices ()) (fun (Case (dt, s, layout, re, im)) ->
      cover_square_shapes s;
      cover_layout layout;
      let n = cols s in
      let a = laid dt layout s re im in
      let got = Nx.to_array (c128 (Nx.det a)) in
      List.iteri
        (fun b (d, sign) ->
          let want =
            Array.fold_left Complex.mul { Complex.re = sign; im = 0. } d
          in
          let size = Array.fold_left (fun p z -> p *. Complex.norm z) 1. d in
          at_most
            ~msg:(Printf.sprintf "matrix %d" b)
            float_exact
            ~than:(((g dt (2 * n) +. here n) *. size) +. e dt (2 * n))
            (Nx_support.complex_gap got.(b) want))
        (pivots a))

let test_det_edges () =
  let one x = Nx.item [] x in
  equal ~msg:"no row" float_exact 1.
    (one (Nx.det (Nx.zeros Nx.float64 [| 0; 0 |])));
  equal ~msg:"a swap" float_exact (-1.)
    (one (Nx.det (f64 [| 2; 2 |] [| 0.; 1.; 1.; 0. |])));
  (* 1e200 · 1e200 is past the range: slogdet holds it. *)
  equal ~msg:"an overflowing product" float_exact Float.infinity
    (one (Nx.det (f64 [| 2; 2 |] [| 1e200; 0.; 0.; 1e200 |])))

(* A zero determinant is +0 whatever the interchanges and the pivots' signs:
   [[-0]] has none, [[1; 2]; [2; 4]] one. *)
let test_det_zero () =
  let plus_zero ~msg x = equal ~msg float_exact 0. x in
  List.iter
    (fun (msg, s, xs) ->
      let a = f64 s xs in
      plus_zero ~msg (Nx.item [] (Nx.det a));
      plus_zero ~msg:(msg ^ ", slogdet's sign")
        (Nx.item [] (fst (Nx.slogdet Nx.float64 a))))
    [
      ("-0", [| 1; 1 |], [| -0. |]);
      ("a -0 pivot", [| 2; 2 |], [| 1.; 0.; 0.; -0. |]);
      ("one interchange", [| 2; 2 |], [| 1.; 2.; 2.; 4. |]);
    ];
  let z =
    Nx.det
      (Nx.create Nx.complex128 [| 1; 1 |] [| { Complex.re = -0.; im = -0. } |])
  in
  plus_zero ~msg:"complex -0, real part" (Nx.item [] z).re;
  plus_zero ~msg:"complex -0, imaginary part" (Nx.item [] z).im

let test_det_alone () =
  let finite = f64 [| 2; 2 |] [| 2.; 1.; 1.; 3. |] in
  let batch = Nx.stack [ finite; f64 [| 2; 2 |] [| 1.; Float.nan; 0.; 1. |] ] in
  alone ~msg:"det" (Nx.det finite) (Nx.det batch);
  let sign, logabs = Nx.slogdet Nx.float64 finite in
  let sign', logabs' = Nx.slogdet Nx.float64 batch in
  alone ~msg:"slogdet's sign" sign sign';
  alone ~msg:"slogdet's logabs" logabs logabs'

let test_det_raises () =
  invalid_by ~by:"Nx.det" (fun () -> Nx.det (Nx.ones Nx.float64 [| 3 |]));
  invalid_by ~by:"Nx.det" (fun () -> Nx.det (Nx.ones Nx.float64 [| 2; 3 |]));
  invalid_by ~by:"Nx.det" (fun () -> Nx.det (Nx.ones Nx.int32 [| 2; 2 |]))

(* slogdet *)

(* logabs is the sum of the pivots' magnitudes' logarithms and sign the product
   of their signs, negated per interchange (nx.mli). Each logarithm is within 2
   ulps, 4 u of its value ({!Nx.log}), and adds at most u absolutely where it is
   near 0; their sum in the dtype adds γ(n) of the sum of their magnitudes. Each
   complex sign is within 3 u ({!Nx.sign}), their product within γ(2n) more. *)
let test_slogdet =
  prop "sign and logabs are those of lu's pivots" (square_matrices ())
    (fun (Case (dt, s, layout, re, im)) ->
      cover_square_shapes s;
      cover_layout layout;
      let n = cols s in
      let a = laid dt layout s re im in
      let (Any_float r) = real_of dt in
      let sign, logabs = Nx.slogdet r a in
      let sign = Nx.to_array (c128 sign) in
      let logabs = Nx.to_array (Nx.cast Nx.float64 logabs) in
      let u = unit_roundoff dt in
      List.iteri
        (fun b (d, parity) ->
          let msg what = Printf.sprintf "matrix %d: %s" b what in
          if Array.exists (fun (z : Complex.t) -> z.re = 0. && z.im = 0.) d then begin
            equal
              ~msg:(msg "a singular matrix's sign")
              (pair float_exact float_exact)
              (0., 0.) (sign.(b).re, sign.(b).im);
            equal
              ~msg:(msg "a singular matrix's logabs")
              float_exact Float.neg_infinity logabs.(b)
          end
          else begin
            (* A pivot's parts as [(re, im, e)], z = (re + i im) 2^e: below
               2^-900 they are scaled exactly by 2^600, so that a subnormal
               pivot's modulus keeps its precision. *)
            let scaled (z : Complex.t) =
              let big = Float.max (Float.abs z.re) (Float.abs z.im) in
              if big >= 0x1p-900 then (z.re, z.im, 0)
              else (Float.ldexp z.re 600, Float.ldexp z.im 600, -600)
            in
            let unit z =
              let re, im, _ = scaled z in
              let m = Float.hypot re im in
              { Complex.re = re /. m; im = im /. m }
            in
            let log_norm z =
              let re, im, e = scaled z in
              let l = Float.log (Float.hypot re im) in
              if e = 0 then l else l +. (Float.of_int e *. Float.log 2.)
            in
            let want =
              Array.fold_left
                (fun p z -> Complex.mul p (unit z))
                { Complex.re = parity; im = 0. }
                d
            in
            let logs = Array.map log_norm d in
            let sum = Array.fold_left ( +. ) 0. logs in
            let size = Array.fold_left (fun t l -> t +. Float.abs l) 0. logs in
            at_most ~msg:(msg "sign") float_exact
              ~than:((3. *. Float.of_int n *. u) +. g dt (2 * n) +. here n)
              (Nx_support.complex_gap sign.(b) want);
            at_most ~msg:(msg "logabs") float_exact
              ~than:
                ((((4. *. u) +. g dt n +. here n) *. size)
                +. (Float.of_int n *. u))
              (Nx_support.gap logabs.(b) sum)
          end)
        (pivots a))

(* [1e200 I] of order 2 has the determinant 1e400, past float64, and logabs 400
   ln 10; [[3 + 4i]] has logabs ln 5. Each within the bound above. *)
let test_slogdet_edges () =
  let slogdet a = Nx.slogdet Nx.float64 a in
  let one x = Nx.item [] x in
  let sign, logabs = slogdet (f64 [| 2; 2 |] [| 1.; 2.; 2.; 4. |]) in
  equal ~msg:"a singular matrix's sign" float_exact 0. (one sign);
  equal ~msg:"a singular matrix's logabs" float_exact Float.neg_infinity
    (one logabs);
  let sign, logabs =
    slogdet (f64 [| 2; 2 |] [| Float.infinity; 0.; 0.; 0. |])
  in
  equal ~msg:"a zero pivot beside an infinite one: sign" float_exact 0.
    (one sign);
  equal ~msg:"a zero pivot beside an infinite one: logabs" float_exact
    Float.neg_infinity (one logabs);
  let big = 2. *. Float.log 1e200 in
  let sign, logabs = slogdet (f64 [| 2; 2 |] [| 1e200; 0.; 0.; 1e200 |]) in
  equal ~msg:"beyond the range" (float (6. *. 0x1p-53 *. big)) big (one logabs);
  equal ~msg:"beyond the range: sign" float_exact 1. (one sign);
  let _, logabs =
    slogdet
      (Nx.create Nx.complex128 [| 1; 1 |] [| { Complex.re = 3.; im = 4. } |])
  in
  equal ~msg:"a complex pivot"
    (float (5. *. 0x1p-53 *. Float.log 5.))
    (Float.log 5.) (one logabs);
  let sign, logabs = slogdet (Nx.zeros Nx.float64 [| 0; 0 |]) in
  equal ~msg:"no row: sign" float_exact 1. (one sign);
  equal ~msg:"no row: logabs" float_exact 0. (one logabs)

let test_slogdet_raises () =
  invalid_by ~by:"Nx.slogdet" (fun () ->
      Nx.slogdet Nx.float64 (Nx.ones Nx.float64 [| 2; 3 |]));
  invalid_by ~by:"Nx.slogdet" (fun () ->
      Nx.slogdet Nx.float32 (Nx.ones Nx.float16 [| 2; 2 |]))

(* A 1 × 1 complex matrix, its one pivot. *)
type pivot = Pivot : (Complex.t, 'c) D.t * Complex.t -> pivot

let pp_pivot ppf (Pivot (dt, z)) =
  Format.fprintf ppf "%a %h%+hi" D.pp dt z.Complex.re z.im

(* [c (1 + i s)], [c] rounded to [dt]'s parts. *)
let pivot (type c) (dt : (Complex.t, c) D.t) c s =
  let c =
    match dt with
    | D.Complex64 -> Int32.float_of_bits (Int32.bits_of_float c)
    | _ -> c
  in
  Pivot (dt, { Complex.re = c; im = s *. c })

(* The pivot [c (1 ± i)] is not zero, and its modulus [c √2] is past the largest
   float of its dtype. Its sign is [(1 ± i) / √2], within [3 u] ({!Nx.sign}),
   and its logabs [ln c + ln 2 / 2], within 2 ulps of [dt]'s precision
   ({!Nx.log}): nx.mli 3500, 3507-3511. *)
let test_slogdet_overflow =
  cases "a pivot whose modulus overflows keeps a unit sign and its logarithm"
    ~id:(Format.asprintf "%a" pp_pivot)
    [
      pivot D.Complex128 1.5e308 1.;
      pivot D.Complex128 1.5e308 (-1.);
      pivot D.Complex64 3e38 1.;
      pivot D.Complex64 3e38 (-1.);
    ]
    (fun (Pivot (dt, z)) ->
      let u = match dt with D.Complex64 -> 0x1p-24 | _ -> 0x1p-53 in
      let a = Nx.create dt [| 1; 1 |] [| z |] in
      let sign, logabs = Nx.slogdet Nx.float64 a in
      let s = Nx.item [] (Nx.cast Nx.complex128 sign) in
      let half = Float.sqrt 0.5 in
      let part = float (3. *. u) in
      equal ~msg:"sign" (pair part part)
        (half, Float.copy_sign half z.im)
        (s.re, s.im);
      equal ~msg:"logabs"
        (float_rel ~rel:(4. *. u) ~abs:0.)
        (Float.log z.re +. (Float.log 2. /. 2.))
        (Nx.item [] logabs))

(* An infinite pivot is not zero: its sign is the direction its parts point to
   ({!Nx.sign}), and the logarithm of its magnitude is infinity. *)
let test_slogdet_infinite =
  let inf = Float.infinity and half = Float.sqrt 0.5 in
  cases "an infinite pivot has a unit sign and logabs infinity"
    ~id:(fun ((z : Complex.t), _) -> Printf.sprintf "%h%+hi" z.re z.im)
    [
      ({ Complex.re = inf; im = 0. }, (1., 0.));
      ({ Complex.re = -.inf; im = 2. }, (-1., 0.));
      ({ Complex.re = inf; im = -.inf }, (half, -.half));
    ]
    (fun (z, want) ->
      let a = Nx.create Nx.complex128 [| 1; 1 |] [| z |] in
      let sign, logabs = Nx.slogdet Nx.float64 a in
      let s = Nx.item [] sign in
      let part = float (3. *. 0x1p-53) in
      equal ~msg:"sign" (pair part part) want (s.re, s.im);
      equal ~msg:"logabs" float_exact inf (Nx.item [] logabs))

(* solve_triangular *)

(* nx_cpu.mli: (op a + Δ) x = b + δ with |Δ| <= γ(2n) |op a| and |δ_i| <= ε(2n)
   (1 + |(op a)_ii|), so the residual op a x - b is within γ(2n) |op a| |x| +
   ε(2n) (1 + |(op a)_ii|), and the check's own. *)
let triangular_solved ~msg t x b =
  let dt = Nx.dtype t and n = cols (Nx.shape t) in
  let tii = Nx.unsqueeze ~axes:[ -1 ] (mag (Nx.diagonal t)) in
  let bound =
    Nx.add
      (Nx.mul_s (Nx.matmul (mag t) (mag x)) (g dt (2 * n) +. here n))
      (Nx.add
         (Nx.mul_s (Nx.add_s tii 1.) (e dt (2 * n)))
         (Nx.mul_s (mag b) (here n)))
  in
  within ~msg bound (mag (Nx.sub (Nx.matmul (c128 t) (c128 x)) (c128 b)))

let forms =
  List.concat_map
    (fun triangle ->
      List.concat_map
        (fun transpose ->
          List.map (fun unit -> (triangle, transpose, unit)) [ false; true ])
        [ None; Some Nx.Transpose; Some Nx.Adjoint ])
    [ Nx.Lower; Nx.Upper ]

let form_name (triangle, transpose, unit) =
  String.concat ", "
    [
      (match triangle with Nx.Lower -> "lower" | Nx.Upper -> "upper");
      (match transpose with
      | None -> "a"
      | Some Nx.Transpose -> "aᵀ"
      | Some Nx.Adjoint -> "aᴴ");
      (if unit then "unit diagonal" else "its diagonal");
    ]

(* The matrix [a]'s [triangle] names, with ones on its diagonal where [unit]. *)
let triangle_of (type v s) (a : (v, s, Nx.host) Nx.t) triangle unit =
  let s = Nx.shape a in
  match (triangle, unit) with
  | Nx.Lower, false -> Nx.tril a
  | Nx.Upper, false -> Nx.triu a
  | Nx.Lower, true ->
      Nx.add (Nx.tril ~k:(-1) a) (identity (Nx.dtype a) (lead s) (cols s))
  | Nx.Upper, true ->
      Nx.add (Nx.triu ~k:1 a) (identity (Nx.dtype a) (lead s) (cols s))

let test_solve_triangular =
  prop
    "x solves its triangle within γ(2n) under each transpose and unit_diag, \
     reading that triangle alone"
    (square_matrices ()) (fun (Case (dt, s, _, re, im)) ->
      cover_square_shapes s;
      (* Complex values off the diagonal tell aᵀ from aᴴ. *)
      let a = definite (operand dt s re im) in
      let b = operand dt s im re in
      List.iter
        (fun ((triangle, transpose, unit_diag) as form) ->
          let x = Nx.solve_triangular ~triangle ?transpose ~unit_diag a b in
          let t = triangle_of a triangle unit_diag in
          let t =
            match transpose with
            | None -> t
            | Some Nx.Transpose -> Nx.matrix_transpose t
            | Some Nx.Adjoint -> adj t
          in
          triangular_solved ~msg:(form_name form) t x b)
        forms)

(* The first [k] parts of [re] and [im], a vector for each matrix of [s]. *)
let vector_of (type v s) (dt : (v, s) D.t) s re im =
  let k = numel (lead s) * cols s in
  operand dt
    (Array.append (lead s) [| cols s |])
    (Array.sub re 0 k) (Array.sub im 0 k)

let column x = Nx.unsqueeze ~axes:[ -1 ] x
let first x = Nx.slice [ Nx.R (0, 1) ] x

let test_solve_triangular_shapes =
  prop "a vector b gives x of its shape, and a's and b's leading axes broadcast"
    (square_matrices ()) (fun (Case (dt, s, _, re, im)) ->
      cover_square_shapes s;
      let a = definite (operand dt s re im) in
      let v = vector_of dt s im re in
      let x = Nx.solve_triangular a v in
      equal ~msg:"a vector's shape" (array int) (Nx.shape v) (Nx.shape x);
      triangular_solved ~msg:"a vector" (Nx.tril a) (column x) (column v);
      if Array.length s = 3 && s.(0) = 2 then begin
        let b = operand dt s im re in
        let x = Nx.solve_triangular (first a) b in
        equal ~msg:"a's batch of one: shape" (array int) s (Nx.shape x);
        triangular_solved ~msg:"a's batch of one" (Nx.tril (first a)) x b;
        let x = Nx.solve_triangular a (first b) in
        equal ~msg:"b's batch of one: shape" (array int) s (Nx.shape x);
        triangular_solved ~msg:"b's batch of one" (Nx.tril a) x (first b)
      end)

let test_solve_triangular_alone () =
  let finite = f64 [| 2; 2 |] [| 2.; 0.; 1.; 3. |] in
  let zero = f64 [| 2; 2 |] [| 2.; 0.; 1.; 0. |] in
  let b = Nx.ones Nx.float64 [| 2 |] in
  alone ~msg:"a zero on the diagonal"
    (Nx.solve_triangular finite b)
    (Nx.solve_triangular
       (Nx.stack [ finite; zero ])
       (Nx.ones Nx.float64 [| 2; 2 |]));
  equal ~msg:"a zero unit_diag does not read" (array float_exact)
    (parts (Nx.solve_triangular ~unit_diag:true finite b))
    (parts (Nx.solve_triangular ~unit_diag:true zero b))

let test_solve_triangular_raises () =
  let by = "Nx.solve_triangular" and sq = Nx.eye Nx.float64 2 in
  invalid_by ~by (fun () ->
      Nx.solve_triangular
        (Nx.ones Nx.float64 [| 3 |])
        (Nx.ones Nx.float64 [| 3 |]));
  invalid_by ~by (fun () ->
      Nx.solve_triangular
        (Nx.ones Nx.float64 [| 2; 3 |])
        (Nx.ones Nx.float64 [| 2; 1 |]));
  invalid_by ~by (fun () ->
      Nx.solve_triangular
        (Nx.ones Nx.float16 [| 2; 2 |])
        (Nx.ones Nx.float16 [| 2; 1 |]));
  invalid_by ~by (fun () ->
      Nx.solve_triangular sq (Nx.ones Nx.float64 [| 1; 1; 2; 1 |]));
  invalid_by ~by (fun () -> Nx.solve_triangular sq (Nx.ones Nx.float64 [| 3 |]));
  invalid_by ~by (fun () ->
      Nx.solve_triangular
        (Nx.ones Nx.float64 [| 3; 2; 2 |])
        (Nx.ones Nx.float64 [| 2; 2; 1 |]))

(* solve *)

(* lu's P a = L U within γ(2k) and its two triangular solves within γ(2n) each
   (nx_cpu.mli) leave P (a x - b) within γ(6n) |L| |U| |x|, as γ(a) + γ(b) +
   γ(a) γ(b) <= γ(a + b), plus the ε(2n) (1 + |U|max) of each step's elements,
   carried through rows of L of magnitudes at most 1 and x's columns: at most
   ε(6n) (1 + |U|max) (n + 2 + Σ_i |x_ij|). *)
let solved ~msg a b x =
  let b, x = if Nx.ndim b < Nx.ndim a then (column b, column x) else (b, x) in
  let dt = Nx.dtype a and n = cols (Nx.shape a) in
  if n > 0 then begin
    let perm, l, u = Nx.lu a in
    let by_perm y = Nx.take_along_axis ~axis:(-2) (column perm) y in
    let own = Nx.mul_s (Nx.add (Nx.matmul (mag a) (mag x)) (mag b)) (here n) in
    let umax = Nx.max ~axes:[ -2; -1 ] ~keepdims:true (mag u) in
    let xsum = Nx.sum ~axes:[ -2 ] ~keepdims:true (mag x) in
    let eps =
      Nx.mul_s
        (Nx.mul (Nx.add_s umax 1.) (Nx.add_s xsum (Float.of_int (n + 2))))
        (e dt (6 * n))
    in
    let lux = Nx.matmul (Nx.matmul (mag l) (mag u)) (mag x) in
    let bound =
      Nx.add (Nx.add (Nx.mul_s lux (g dt (6 * n))) eps) (by_perm own)
    in
    within ~msg bound
      (mag (by_perm (Nx.sub (Nx.matmul (c128 a) (c128 x)) (c128 b))))
  end

let test_solve =
  prop
    "a x is b within γ(6n), vector or columns, leading axes of a and b \
     broadcast"
    (square_matrices ()) (fun (Case (dt, s, _, re, im)) ->
      cover_square_shapes s;
      let a = conditioned (operand dt s re im) in
      let b = operand dt s im re in
      solved ~msg:"columns" a b (Nx.solve a b);
      let v = vector_of dt s re im in
      let x = Nx.solve a v in
      equal ~msg:"a vector's shape" (array int) (Nx.shape v) (Nx.shape x);
      solved ~msg:"a vector" a v x;
      if Array.length s = 3 && s.(0) = 2 then begin
        solved ~msg:"a's batch of one" (first a) b (Nx.solve (first a) b);
        solved ~msg:"b's batch of one" a (first b) (Nx.solve a (first b))
      end)

let test_solve_alone () =
  let regular = f64 [| 2; 2 |] [| 2.; 0.; 0.; 2. |] in
  let singular = f64 [| 2; 2 |] [| 1.; 2.; 2.; 4. |] in
  let batch = Nx.stack [ regular; singular ] in
  alone ~msg:"solve"
    (Nx.solve regular (Nx.ones Nx.float64 [| 2 |]))
    (Nx.solve batch (Nx.ones Nx.float64 [| 2; 2 |]));
  alone ~msg:"inv" (Nx.inv regular) (Nx.inv batch)

let test_solve_raises () =
  let sq = Nx.eye Nx.float64 2 in
  invalid_by ~by:"Nx.solve" (fun () -> Nx.solve sq (Nx.ones Nx.float64 [| 3 |]));
  invalid_by ~by:"Nx.solve" (fun () ->
      Nx.solve sq (Nx.ones Nx.float64 [| 1; 1; 2; 1 |]));
  invalid_by ~by:"Nx.solve" (fun () ->
      Nx.solve (Nx.ones Nx.float64 [| 2; 3 |]) (Nx.ones Nx.float64 [| 2 |]));
  invalid_by ~by:"Nx.solve" (fun () ->
      Nx.solve (Nx.eye Nx.int32 2) (Nx.ones Nx.int32 [| 2 |]))

(* inv *)

let test_inv =
  prop "a inv a is the identity within solve's bound" (square_matrices ())
    (fun (Case (dt, s, _, re, im)) ->
      cover_square_shapes s;
      let a = conditioned (operand dt s re im) in
      solved ~msg:"a inv a" a (identity dt (lead s) (cols s)) (Nx.inv a))

let test_inv_raises () =
  invalid_by ~by:"Nx.inv" (fun () -> Nx.inv (Nx.ones Nx.int32 [| 2; 2 |]));
  invalid_by ~by:"Nx.inv" (fun () -> Nx.inv (Nx.ones Nx.float64 [| 2; 3 |]));
  invalid_by ~by:"Nx.inv" (fun () -> Nx.inv (Nx.ones Nx.float64 [| 3 |]))

(* lstsq *)

(* Two right-hand sides of [m] rows behind [lead], in [dt]. *)
let rhs (type v s) (dt : (v, s) D.t) lead m : (v, s, Nx.host) Nx.t =
  let s = Array.append lead [| m; 2 |] in
  Nx.cast dt
    (Nx.sin
       (Nx.reshape s (Nx.arange_f Nx.float64 0. (Float.of_int (numel s)) 1.)))

(* With a + E = u diag s vh and E within svd's δ = g ‖a‖_F + h (nx_cpu.mli), x
   solves the least-squares problem of a + E but for its products' rounding, so
   aᴴ (b - a x) = -Eᴴ r + aᴴ E x to first order: each column within 4 δ (‖r_j‖ +
   ‖a‖_F ‖x_j‖), the 4 holding the second order, u's and vh's departure from
   orthonormality (within g) and the products. *)
let test_lstsq_normal =
  prop "aᴴ (b - a x) is zero within svd's bound" (matrices ())
    (fun (Case (dt, s, _, re, im)) ->
      cover_shapes s;
      let a = conditioned (operand dt s re im) in
      let m = rows s and n = cols s in
      let b = rhs dt (lead s) m in
      let (Any_float f) = real_of dt in
      let x, _, _, _ = Nx.lstsq f a b in
      let gs, hs = svd_bound dt s in
      let delta = column (Nx.add_s (Nx.mul_s (fro a) gs) hs) in
      let r = Nx.sub (c128 b) (Nx.matmul (c128 a) (c128 x)) in
      let fa = column (fro a) in
      let bound =
        Nx.mul_s
          (Nx.mul delta (Nx.add (column_norms r) (Nx.mul fa (column_norms x))))
          4.
      in
      let products =
        Nx.matmul
          (Nx.matrix_transpose (mag a))
          (Nx.add (mag b) (Nx.matmul (mag a) (mag x)))
      in
      within ~msg:"‖aᴴ r_j‖"
        (Nx.add bound (Nx.mul_s (column_norms products) (here (m + n))))
        (column_norms (Nx.matmul (adj (c128 a)) r)))

(* lstsq's x is vhᴴ (d (uᴴ b)) and pinv's p b is ((vhᴴ d) uᴴ) b, of one svd's
   factors and reciprocals d: they differ by their products' rounding, within
   γ(2 (m + n) + 4) of |vhᴴ| |d| |uᴴ| |b|. *)
let test_lstsq_least =
  prop "x is pinv's b, the solution of least norm; s is svd's and the rank full"
    (matrices ()) (fun (Case (dt, s, _, re, im)) ->
      cover_shapes s;
      let a = conditioned (operand dt s re im) in
      let m = rows s and n = cols s in
      let b = rhs dt (lead s) m in
      let (Any_float f) = real_of dt in
      let x, _, rank, sv = Nx.lstsq f a b in
      let u, s', vh = Nx.svd f a in
      let d = Nx.cast Nx.float64 (Nx.recip s') in
      let abs =
        Nx.matmul
          (columns_by (Nx.matrix_transpose (mag vh)) d)
          (Nx.matmul (Nx.matrix_transpose (mag u)) (mag b))
      in
      within ~msg:"x - pinv a b"
        (Nx.mul_s abs (g dt ((2 * (m + n)) + 4) +. here (m + n)))
        (mag (Nx.sub (c128 x) (Nx.matmul (c128 (Nx.pinv a)) (c128 b))));
      equal ~msg:"s" (array float_exact) (Nx.to_array s') (Nx.to_array sv);
      Array.iter
        (equal ~msg:"rank" int64 (Int64.of_int (Stdlib.min m n)))
        (Nx.to_array rank))

(* b - a x formed in the dtype is within γ(n + 1) (|b| + |a| |x|) of r, so its
   squared norm within 2 ‖r_j‖ ‖Δr_j‖ + ‖Δr_j‖², and its sum in the dtype adds
   γ(m + 1) (‖r_j‖ + ‖Δr_j‖)². *)
let test_lstsq_residuals =
  prop
    "residuals are the squared norms of b - a x's columns, of b's shape \
     without its rows"
    (matrices ()) (fun (Case (dt, s, _, re, im)) ->
      cover_shapes s;
      let a = conditioned (operand dt s re im) in
      let m = rows s and n = cols s in
      let (Any_float f) = real_of dt in
      let check msg b x residuals =
        let r = Nx.sub (c128 b) (Nx.matmul (c128 a) (c128 x)) in
        let rn = column_norms r in
        let dr =
          column_norms
            (Nx.mul_s
               (Nx.add (mag b) (Nx.matmul (mag a) (mag x)))
               (g dt (n + 1) +. here n))
        in
        let bound =
          Nx.add
            (Nx.add (Nx.mul_s (Nx.mul rn dr) 2.) (Nx.square dr))
            (Nx.mul_s (Nx.square (Nx.add rn dr)) (g dt (m + 1)))
        in
        within ~msg bound
          (Nx.abs (Nx.sub (Nx.cast Nx.float64 residuals) (Nx.square rn)))
      in
      let b = rhs dt (lead s) m in
      let x, residuals, _, _ = Nx.lstsq f a b in
      equal ~msg:"columns' shape" (array int)
        (Array.append (lead s) [| 2 |])
        (Nx.shape residuals);
      check "columns" b x residuals;
      let v =
        Nx.slice (List.init (Array.length s - 1) (fun _ -> Nx.A) @ [ Nx.I 0 ]) b
      in
      let x, residuals, _, _ = Nx.lstsq f a v in
      equal ~msg:"a vector's x" (array int)
        (Array.append (lead s) [| n |])
        (Nx.shape x);
      equal ~msg:"a vector's shape" (array int) (lead s) (Nx.shape residuals);
      check "a vector" (column v) (column x) (column residuals))

let test_lstsq_rank () =
  let lstsq ?rtol a b = Nx.lstsq ?rtol Nx.float64 a b in
  let _, _, rank, _ =
    lstsq
      (f64 [| 3; 2 |] [| 1.; 1.; 2.; 2.; 3.; 3. |])
      (Nx.ones Nx.float64 [| 3 |])
  in
  equal ~msg:"a repeated column" int64 1L (Nx.item [] rank);
  (* 1e-3 is at or below 2e-3 · 1: it counts as zero. *)
  let a = f64 [| 2; 2 |] [| 1.; 0.; 0.; 1e-3 |] in
  let x, _, rank, _ = lstsq ~rtol:2e-3 a (Nx.ones Nx.float64 [| 2 |]) in
  equal ~msg:"rtol's rank" int64 1L (Nx.item [] rank);
  equal ~msg:"rtol's x" (array float_exact) [| 1.; 0. |] (Nx.to_array x)

let test_lstsq_alone () =
  let finite = f64 [| 2; 2 |] [| 2.; 1.; 1.; 3. |] in
  let b = Nx.ones Nx.float64 [| 2 |] in
  let x, residuals, _, s = Nx.lstsq Nx.float64 finite b in
  let batch = Nx.stack [ finite; f64 [| 2; 2 |] [| 1.; Float.nan; 0.; 1. |] ] in
  let x', residuals', rank, s' =
    Nx.lstsq Nx.float64 batch (Nx.ones Nx.float64 [| 2; 2 |])
  in
  alone ~msg:"x" x x';
  alone ~msg:"residuals" residuals residuals';
  alone ~msg:"s" s s';
  equal ~msg:"rank" (array int64) [| 2L; -1L |] (Nx.to_array rank)

(* The default rtol is max m n times the machine epsilon: for a 2 × 2 float64, 2
   ε ≈ 4.4e-16, so 3e-16 counts as zero and 6e-16 does not; for a 3 × 2, 3 ε ≈
   6.7e-16, so 5e-16 counts as zero, which min m n would keep; for a 2 × 2
   float32, 2 · 2^-23 ≈ 2.4e-7. *)
let test_default_rtol () =
  List.iter
    (fun (msg, s, d, want) ->
      let a =
        f64 s
          (Array.init (numel s) (fun k ->
               if k = 0 then 1. else if k = 3 then d else 0.))
      in
      let _, _, rank, _ =
        Nx.lstsq Nx.float64 a (Nx.ones Nx.float64 [| rows s |])
      in
      equal ~msg:(msg ^ ", lstsq") int64 want (Nx.item [] rank);
      equal ~msg:(msg ^ ", matrix_rank") int64 want
        (Nx.item [] (Nx.matrix_rank a));
      (* pinv's second diagonal element times [d]: 1 where [d] is kept. *)
      let p = Nx.item [ 1; 1 ] (Nx.pinv a) in
      equal ~msg:(msg ^ ", pinv") (float 0x1p-51)
        (if want = 2L then 1. else 0.)
        (p *. d))
    [
      ("2 × 2 float64, 3e-16", [| 2; 2 |], 3e-16, 1L);
      ("2 × 2 float64, 6e-16", [| 2; 2 |], 6e-16, 2L);
      ("3 × 2 float64, 5e-16", [| 3; 2 |], 5e-16, 1L);
      ("3 × 2 float64, 8e-16", [| 3; 2 |], 8e-16, 2L);
    ];
  List.iter
    (fun (msg, d, want) ->
      let a = Nx.cast Nx.float32 (f64 [| 2; 2 |] [| 1.; 0.; 0.; d |]) in
      equal ~msg:(msg ^ ", matrix_rank") int64 want
        (Nx.item [] (Nx.matrix_rank a)))
    [ ("2 × 2 float32, 1.5e-7", 1.5e-7, 1L); ("2 × 2 float32, 3e-7", 3e-7, 2L) ]

let test_lstsq_raises () =
  let by = "Nx.lstsq" and r = Nx.ones Nx.float64 [| 2; 3 |] in
  invalid_by ~by (fun () ->
      Nx.lstsq ~rtol:(-1.) Nx.float64 r (Nx.ones Nx.float64 [| 2 |]));
  invalid_by ~by (fun () ->
      Nx.lstsq ~rtol:Float.nan Nx.float64 r (Nx.ones Nx.float64 [| 2 |]));
  invalid_by ~by (fun () ->
      Nx.lstsq Nx.float32
        (Nx.ones Nx.float16 [| 2; 2 |])
        (Nx.ones Nx.float16 [| 2 |]));
  invalid_by ~by (fun () ->
      Nx.lstsq Nx.float64
        (Nx.ones Nx.float64 [| 3 |])
        (Nx.ones Nx.float64 [| 3 |]));
  invalid_by ~by (fun () -> Nx.lstsq Nx.float64 r (Nx.ones Nx.float64 [| 3 |]));
  invalid_by ~by (fun () ->
      Nx.lstsq Nx.float64 r (Nx.ones Nx.float64 [| 1; 1; 2; 1 |]));
  invalid_by ~by (fun () ->
      Nx.lstsq Nx.float64
        (Nx.ones Nx.float64 [| 3; 2; 2 |])
        (Nx.ones Nx.float64 [| 2; 2; 1 |]))

(* pinv *)

(* A matrix's 2-norm at float64, 0 where it has no element. *)
let two_norm x =
  let s = Nx.shape x in
  if Stdlib.min (rows s) (cols s) = 0 then Nx.zeros Nx.float64 (lead s)
  else Nx.matrix_norm Nx.float64 ~ord:`Two (c128 x)

(* pinv a is the pseudoinverse of a + E, E within svd's δ = g ‖a‖_F + h
   (nx_cpu.mli), but for its products' rounding, which moves it as an E of ρ =
   γ(2 (m + n) + 4) ‖a‖_F² ‖p‖_F does. For p the pseudoinverse of a + E, a p a -
   a = E - E p (a + E) - (a + E) p E + E p E, so with κ = ‖a‖₂ ‖p‖₂ each
   condition holds within its first-order term in δ + ρ, times 4 for the second
   order: ‖a p a - a‖_F <= 4 (δ + ρ) (1 + κ)², ‖p a p - p‖_F <= 4 (δ + ρ) ‖p‖₂²
   (1 + κ), and a p and p a are Hermitian within 4 (δ + ρ) ‖p‖₂ (1 + κ). The
   checks add their own products'. *)
let test_pinv =
  prop "the Penrose conditions hold within svd's bound" (matrices ())
    (fun (Case (dt, s, _, re, im)) ->
      cover_shapes s;
      let a = conditioned (operand dt s re im) in
      let m = rows s and n = cols s in
      let p = Nx.pinv a in
      equal ~msg:"shape" (array int)
        (Array.append (lead s) [| n; m |])
        (Nx.shape p);
      let gs, hs = svd_bound dt s in
      let fa = fro a and pn = two_norm p in
      let kappa = Nx.add_s (Nx.mul (two_norm a) pn) 1. in
      let err =
        Nx.mul_s
          (Nx.add
             (Nx.add_s (Nx.mul_s fa gs) hs)
             (Nx.mul_s
                (Nx.mul (Nx.square fa) (fro p))
                (g dt ((2 * (m + n)) + 4))))
          4.
      in
      let za = c128 a and zp = c128 p in
      let ap = Nx.matmul za zp and pa = Nx.matmul zp za in
      let own2 x y =
        Nx.mul_s (fro (Nx.matmul (mag x) (mag y))) (here (m + n))
      in
      let own x y z =
        Nx.mul_s
          (fro (Nx.matmul (Nx.matmul (mag x) (mag y)) (mag z)))
          (here (m + n))
      in
      within ~msg:"a p a - a"
        (Nx.add (Nx.mul err (Nx.square kappa)) (own a p a))
        (fro (Nx.sub (Nx.matmul ap za) za));
      within ~msg:"p a p - p"
        (Nx.add (Nx.mul (Nx.mul err (Nx.square pn)) kappa) (own p a p))
        (fro (Nx.sub (Nx.matmul pa zp) zp));
      let hermitian = Nx.mul (Nx.mul err pn) kappa in
      within ~msg:"(a p)ᴴ - a p"
        (Nx.add hermitian (own2 a p))
        (fro (Nx.sub (adj ap) ap));
      within ~msg:"(p a)ᴴ - p a"
        (Nx.add hermitian (own2 p a))
        (fro (Nx.sub (adj pa) pa)))

(* Both are within √2 ‖p‖₂² (δ + ρ) of the exact pseudoinverse of a full rank
   matrix, δ eigh's and svd's (nx_cpu.mli): within twice the sum of each other,
   times 2 for the second order. *)
let test_pinv_hermitian =
  prop "under ~hermitian it is pinv, of a definite or an indefinite matrix"
    (square_matrices ()) (fun (Case (dt, s, _, re, im)) ->
      cover_square_shapes s;
      let n = cols s in
      cover "indefinite" (n >= 2);
      let a = conditioned (operand dt s re im) in
      let gs, hs = svd_bound dt s and ge, he = eigh_bound dt n in
      let close msg h =
        let p = Nx.pinv h in
        let fh = fro h and pn = two_norm p in
        let delta =
          Nx.add
            (Nx.add_s (Nx.mul_s fh (gs +. ge)) (hs +. he))
            (Nx.mul_s
               (Nx.mul (Nx.square fh) (fro p))
               (2. *. g dt ((4 * n) + 4)))
        in
        within ~msg
          (Nx.mul_s (Nx.mul delta (Nx.square pn)) 4.)
          (fro (Nx.sub (c128 (Nx.pinv ~hermitian:true h)) (c128 p)))
      in
      close "definite" (definite a);
      (* q diag w qᴴ with eigenvalues 1, -2, 3, …: Hermitian, indefinite. *)
      let q, _ = Nx.qr a in
      let w =
        Array.init n (fun i ->
            Float.of_int (if i mod 2 = 0 then i + 1 else -(i + 1)))
      in
      let w = Nx.cast dt (f64 [| n |] w) in
      close "indefinite" (Nx.matmul (columns_by q w) (adj q)))

let test_pinv_edges () =
  let a = f64 [| 2; 2 |] [| 1.; 0.; 0.; 1e-3 |] in
  equal ~msg:"rtol" (array float_exact) [| 1.; 0.; 0.; 0. |]
    (Nx.to_array (Nx.pinv ~rtol:2e-3 a));
  (* Under ~hermitian the lower triangle alone is read. *)
  let lower = f64 [| 2; 2 |] [| 2.; 1.; 1.; 3. |] in
  equal ~msg:"NaN above is not read" (array float_exact)
    (parts (Nx.pinv ~hermitian:true lower))
    (parts
       (Nx.pinv ~hermitian:true (f64 [| 2; 2 |] [| 2.; Float.nan; 1.; 3. |])))

let test_pinv_alone () =
  let finite = f64 [| 2; 2 |] [| 2.; 1.; 1.; 3. |] in
  let batch =
    Nx.stack [ finite; f64 [| 2; 2 |] [| 1.; Float.nan; Float.nan; 1. |] ]
  in
  alone ~msg:"pinv" (Nx.pinv finite) (Nx.pinv batch);
  alone ~msg:"Hermitian pinv"
    (Nx.pinv ~hermitian:true finite)
    (Nx.pinv ~hermitian:true batch)

let test_pinv_raises () =
  let by = "Nx.pinv" in
  invalid_by ~by (fun () ->
      Nx.pinv ~hermitian:true (Nx.ones Nx.float64 [| 2; 3 |]));
  invalid_by ~by (fun () ->
      Nx.pinv ~hermitian:true (Nx.zeros Nx.float64 [| 0; 3 |]));
  invalid_by ~by (fun () -> Nx.pinv ~rtol:(-1.) (Nx.ones Nx.float64 [| 2; 2 |]));
  invalid_by ~by (fun () ->
      Nx.pinv ~rtol:Float.nan (Nx.ones Nx.float64 [| 2; 2 |]));
  invalid_by ~by (fun () -> Nx.pinv (Nx.ones Nx.float16 [| 2; 2 |]));
  invalid_by ~by (fun () -> Nx.pinv (Nx.ones Nx.float64 [| 3 |]))

(* The [m × n] matrix whose diagonal is [d]. *)
let diagonal (type v s) (dt : (v, s) D.t) m n d : (v, s, Nx.host) Nx.t =
  let re =
    Array.init (m * n) (fun k -> if k / n = k mod n then d.(k / n) else 0.)
  in
  operand dt [| m; n |] re (Array.make (m * n) 0.)

(* A singular value counts as zero at or below [rtol] times the greatest,
   whatever the shape. Of [c · diag 1, 2{^-k}, …], [c = 2{^e}], under [rtol =
   0.75 · 2{^-j}] no singular value lies on the cutoff, and a kept [c · 2{^-j}]
   is at or below [c · rtol · max m n] once [max m n ≥ 2]. pinv's error is
   within 4 ‖p‖₂² (g ‖a‖_F + h) of the exact one's, as above, ‖p‖₂ the
   reciprocal of the least kept value: below ‖p‖₂, which a value wrongly kept or
   dropped moves by at least. *)
let law_rtol =
  prop "a given rtol is relative to the greatest singular value"
    (Gen.with_pp
       (fun ppf (D.Any dt, (((m, n), e), j, ks)) ->
         Format.fprintf ppf "%a %dx%d, e %d, j %d, k %s" D.pp dt m n e j
           (String.concat " " (Array.to_list (Array.map string_of_int ks))))
       Gen.(
         pair (of_list fulls)
           (triple
              (pair (pair (int_range 1 4) (int_range 1 4)) (int_range (-4) 4))
              (int_range 2 12)
              (array ~size:(constant 4) (int_range (-1) 2)))))
    (fun (D.Any dt, (((m, n), e), j, ks)) ->
      let r = Stdlib.min m n in
      let rtol = 0.75 *. Float.ldexp 1. (-j) in
      let s =
        Array.init r (fun i ->
            Float.ldexp 1. (e - if i = 0 then 0 else j + ks.(i)))
      in
      let top = Float.ldexp 1. e in
      let kept x = x > rtol *. top in
      let wide = rtol *. top *. Float.of_int (Stdlib.max m n) in
      cover "kept, at or below rtol · max m n"
        (Array.exists (fun x -> kept x && x <= wide) s);
      cover "dropped" (Array.exists (fun x -> not (kept x)) s);
      cover "the greatest is not 1" (e <> 0);
      let a = diagonal dt m n s in
      let p =
        diagonal dt n m (Array.map (fun x -> if kept x then 1. /. x else 0.) s)
      in
      let least =
        Array.fold_left (fun l x -> if kept x then Float.min l x else l) top s
      in
      let fa = Float.sqrt (Array.fold_left (fun t x -> t +. (x *. x)) 0. s) in
      let pinv_bound (gk, hk) =
        Nx.full Nx.float64 [| n; m |]
          ((4. *. ((gk *. fa) +. hk) /. (least *. least))
          +. (g dt ((2 * (m + n)) + 4) /. least))
      in
      let shape = [| m; n |] in
      within ~msg:"pinv"
        (pinv_bound (svd_bound dt shape))
        (mag (Nx.sub (c128 (Nx.pinv ~rtol a)) (c128 p)));
      if m = n then
        within ~msg:"Hermitian pinv"
          (pinv_bound (eigh_bound dt n))
          (mag (Nx.sub (c128 (Nx.pinv ~rtol ~hermitian:true a)) (c128 p)));
      let count =
        Array.fold_left (fun c x -> if kept x then Int64.succ c else c) 0L s
      in
      equal ~msg:"matrix_rank" int64 count (Nx.item [] (Nx.matrix_rank ~rtol a));
      let (Any_float f) = real_of dt in
      let _, _, rank, _ = Nx.lstsq ~rtol f a (Nx.ones dt [| m |]) in
      equal ~msg:"lstsq's rank" int64 count (Nx.item [] rank))

(* vector_norm *)

(* The vector norm [ord] of the magnitudes [m], by its definition. *)
let reference ord (m : float array) =
  let sum f = Array.fold_left (fun a v -> a +. f v) 0. m in
  if ord = Float.infinity then Array.fold_left Float.max m.(0) m
  else if ord = Float.neg_infinity then Array.fold_left Float.min m.(0) m
  else if ord = 0. then sum (fun v -> if v = 0. then 0. else 1.)
  else sum (fun v -> v ** ord) ** (1. /. ord)

(* At the precision it computes at, of unit roundoff u: each magnitude within 3
   ulps ({!Nx.hypot}), exact on a real; each scaled term (m / c)^p within 1 + 4
   ulps ({!Nx.pow}) and |p| times the magnitudes' error; their sum γ(n) more;
   and the root's 4 ulps plus 1 / |p| <= 2 times the sum's: within (2n + 64) u
   relatively, and as much at float64 for the reference. A subnormal magnitude's
   3 ulps are 3 η absolutely, carried through the scaled sum to at most 8 n²
   η. *)
let norm_close dt n =
  let u = unit_roundoff dt +. 0x1p-53 in
  float_rel
    ~rel:(Float.of_int ((2 * n) + 64) *. u)
    ~abs:(8. *. Float.of_int (n * n) *. least_subnormal dt)

let test_vector_norm =
  prop "each order is its definition over the magnitudes" (matrices ())
    (fun (Case (dt, s, layout, re, im)) ->
      cover_shapes s;
      cover_layout layout;
      let (Any_float r) = real_of dt in
      (* Zero elements, which order 0 skips and a negative order meets. *)
      let zeroed v = Array.map2 (fun r x -> if r < -0.5 then 0. else x) re v in
      let x = laid dt layout s (zeroed re) (zeroed im) in
      let n = cols s in
      let m = Nx.to_array (mag x) in
      cover "a zero element" (Array.exists (( = ) 0.) m);
      let lines = numel s / Stdlib.max 1 n in
      let extremes =
        if n > 0 then [ Float.infinity; Float.neg_infinity ] else []
      in
      List.iter
        (fun ord ->
          let y =
            Nx.to_array
              (Nx.cast Nx.float64 (Nx.vector_norm r ~ord ~axes:[ -1 ] x))
          in
          for i = 0 to lines - 1 do
            let want = reference ord (Array.sub m (i * n) n) in
            equal
              ~msg:(Printf.sprintf "ord %g, vector %d" ord i)
              (norm_close dt n) want y.(i)
          done)
        ([ 2.; 1.; 3.; 0.5; 0.; -1.; -2.; -0.5 ] @ extremes);
      equal ~msg:"every axis"
        (norm_close dt (Array.length m))
        (reference 2. m)
        (Nx.item [] (Nx.cast Nx.float64 (Nx.vector_norm r x))))

(* The sums of powers are scaled by the greatest magnitude, or the least for a
   negative order: [[1e200; 1e200]]'s norm is 1e200 √2, and [[1e-200; 1e-200]]'s
   of order -2 is 1e-200 / √2, within the bound above. *)
let test_vector_norm_scaled () =
  let norm ?ord x =
    Nx.item [] (Nx.vector_norm Nx.float64 ?ord (f64 [| 2 |] x))
  in
  equal ~msg:"no overflow" (norm_close D.Float64 2)
    (1e200 *. Float.sqrt 2.)
    (norm [| 1e200; 1e200 |]);
  equal ~msg:"no underflow" (norm_close D.Float64 2)
    (1e-200 /. Float.sqrt 2.)
    (norm ~ord:(-2.) [| 1e-200; 1e-200 |])

let test_vector_norm_edges () =
  let norm ?ord ?axes x = Nx.vector_norm Nx.float64 ?ord ?axes x in
  let one ?ord x = Nx.item [] (norm ?ord (f64 [| Array.length x |] x)) in
  equal ~msg:"no element" float_exact 0. (one [||]);
  equal ~msg:"order 0 of no element" float_exact 0. (one ~ord:0. [||]);
  equal ~msg:"a negative order of no element" float_exact Float.infinity
    (one ~ord:(-1.) [||]);
  equal ~msg:"order 0 counts the nonzero" float_exact 2.
    (one ~ord:0. [| 0.; -2.; 5. |]);
  List.iter
    (fun ord ->
      equal
        ~msg:(Printf.sprintf "NaN at order %g" ord)
        float_exact Float.nan
        (one ~ord [| Float.nan; 0.; 1. |]))
    [ 2.; 1.; 0.; -1.; -2.; 3.; Float.infinity; Float.neg_infinity ];
  let z = Nx.create Nx.complex64 [| 1 |] [| { Complex.re = 3.; im = 4. } |] in
  equal ~msg:"a complex modulus" float_exact 5.
    (Nx.item [] (Nx.vector_norm Nx.float32 z));
  (* An element of NaN and infinite parts has an infinite magnitude. *)
  let z =
    Nx.create Nx.complex64 [| 1 |]
      [| { Complex.re = Float.nan; im = Float.infinity } |]
  in
  equal ~msg:"an infinite magnitude" float_exact Float.infinity
    (Nx.item [] (Nx.vector_norm Nx.float32 z));
  let x = Nx.ones Nx.float32 [| 2; 3; 4 |] in
  equal ~msg:"every axis" (array int) [||]
    (Nx.shape (Nx.vector_norm Nx.float32 x));
  equal ~msg:"every axis's value" (float 1e-6) (Float.sqrt 24.)
    (Nx.item [] (Nx.vector_norm Nx.float32 x));
  let y = Nx.vector_norm Nx.float32 ~axes:[ 0; 2 ] ~keepdims:true x in
  equal ~msg:"two axes kept" (array int) [| 1; 3; 1 |] (Nx.shape y);
  equal ~msg:"two axes" (float 1e-6) (Float.sqrt 8.) (Nx.item [ 0; 1; 0 ] y);
  (* Along no axis each element is a vector of its own. *)
  let x = f64 [| 2; 2 |] [| -3.; 0.; 4.; Float.infinity |] in
  equal ~msg:"along no axis" (array float_exact)
    [| 3.; 0.; 4.; Float.infinity |]
    (Nx.to_array (norm ~axes:[] x));
  equal ~msg:"along no axis, order 0" (array float_exact) [| 1.; 0.; 1.; 1. |]
    (Nx.to_array (norm ~ord:0. ~axes:[] x));
  equal ~msg:"no vector" (array int) [| 0 |]
    (Nx.shape
       (norm ~ord:Float.infinity ~axes:[ 1 ] (Nx.zeros Nx.float64 [| 0; 0 |])))

let test_vector_norm_raises () =
  let by = "Nx.vector_norm"
  and f64 = Nx.float64
  and r = Nx.ones Nx.float64 [| 2; 3 |] in
  invalid_by ~by (fun () -> Nx.vector_norm f64 ~ord:Float.nan r);
  invalid_by ~by (fun () -> Nx.vector_norm f64 ~axes:[ 2 ] r);
  invalid_by ~by (fun () -> Nx.vector_norm f64 ~axes:[ 0; 0 ] r);
  invalid_by ~by (fun () ->
      Nx.vector_norm f64 ~ord:Float.infinity (Nx.zeros Nx.float64 [| 0 |]));
  invalid_by ~by (fun () ->
      Nx.vector_norm f64 ~ord:Float.neg_infinity
        (Nx.zeros Nx.float64 [| 2; 0 |]));
  invalid_by ~by (fun () -> Nx.vector_norm f64 (Nx.ones Nx.int32 [| 3 |]))

(* matrix_norm *)

(* `Fro, `One and `Inf as vector_norm's bound over their elements, lines and
   sums; the singular values within svd's g ‖a‖_F + h, and the float64
   reference's within its own. *)
(* The matrices of an element draw "empty" only through a batch of none,
   about one case in seven: 200 cases reach it. *)
let test_matrix_norm =
  prop ~count:200 "each order is its definition" (matrices ())
    (fun (Case (dt, s, layout, re, im)) ->
      cover_shapes s;
      cover_layout layout;
      let x = laid dt layout s re im in
      let m = rows s and n = cols s in
      assume (m > 0 && n > 0);
      let abs = mag x in
      let (Any_float r) = real_of dt in
      let norm ord =
        Nx.to_array (Nx.cast Nx.float64 (Nx.matrix_norm r ~ord x))
      in
      let near msg close want got =
        equal ~msg (array close) (Nx.to_array want) got
      in
      near "fro"
        (norm_close dt (m * n))
        (Nx.sqrt (Nx.sum ~axes:[ -2; -1 ] (Nx.square abs)))
        (norm `Fro);
      let lines = norm_close dt (Stdlib.max m n) in
      near "one" lines
        (Nx.max ~axes:[ -1 ] (Nx.sum ~axes:[ -2 ] abs))
        (norm `One);
      near "negone" lines
        (Nx.min ~axes:[ -1 ] (Nx.sum ~axes:[ -2 ] abs))
        (norm `NegOne);
      near "inf" lines
        (Nx.max ~axes:[ -1 ] (Nx.sum ~axes:[ -1 ] abs))
        (norm `Inf);
      near "neginf" lines
        (Nx.min ~axes:[ -1 ] (Nx.sum ~axes:[ -1 ] abs))
        (norm `NegInf);
      let sv = Nx.svdvals Nx.float64 (c128 x) in
      let gs, hs = svd_bound dt s and g64, h64 = svd_bound D.Complex128 s in
      let k = Stdlib.min m n in
      let delta = Nx.add_s (Nx.mul_s (fro x) (gs +. g64)) (hs +. h64) in
      let values msg want got k =
        within ~msg
          (Nx.add (Nx.mul_s delta (Float.of_int k)) (Nx.mul_s want (g dt k)))
          (Nx.abs (Nx.sub want (f64 (Nx.shape want) got)))
      in
      values "two" (Nx.max ~axes:[ -1 ] sv) (norm `Two) 1;
      values "negtwo" (Nx.min ~axes:[ -1 ] sv) (norm `NegTwo) 1;
      values "nuc" (Nx.sum ~axes:[ -1 ] sv) (norm `Nuc) k;
      equal ~msg:"default" (array float_exact) (norm `Fro)
        (Nx.to_array (Nx.cast Nx.float64 (Nx.matrix_norm r x)));
      equal ~msg:"keepdims" (array int)
        (Array.append (lead s) [| 1; 1 |])
        (Nx.shape (Nx.matrix_norm r ~keepdims:true x)))

let test_matrix_norm_edges () =
  let e = Nx.zeros Nx.float64 [| 0; 3 |] in
  let one ?ord x = Nx.item [] (Nx.matrix_norm Nx.float64 ?ord x) in
  equal ~msg:"Frobenius of no element" float_exact 0. (one e);
  equal ~msg:"columns of no element" float_exact 0. (one ~ord:`One e);
  equal ~msg:"nuclear of no element" float_exact 0. (one ~ord:`Nuc e);
  equal ~msg:"no matrix" (array int) [| 0 |]
    (Nx.shape
       (Nx.matrix_norm Nx.float64 ~ord:`Inf (Nx.zeros Nx.float64 [| 0; 0; 3 |])));
  equal ~msg:"no matrix's singular value" (array int) [| 0 |]
    (Nx.shape
       (Nx.matrix_norm Nx.float64 ~ord:`Two (Nx.zeros Nx.float64 [| 0; 2; 0 |])));
  let nan = f64 [| 2; 2 |] [| 1.; Float.nan; 0.; 1. |] in
  List.iter
    (fun (name, ord) ->
      equal ~msg:("NaN, " ^ name) float_exact Float.nan (one ~ord nan))
    [
      ("fro", `Fro);
      ("nuc", `Nuc);
      ("one", `One);
      ("two", `Two);
      ("inf", `Inf);
      ("negone", `NegOne);
      ("negtwo", `NegTwo);
      ("neginf", `NegInf);
    ]

let test_matrix_norm_raises () =
  let by = "Nx.matrix_norm" and f64 = Nx.float64 in
  invalid_by ~by (fun () -> Nx.matrix_norm f64 (Nx.ones Nx.float64 [| 3 |]));
  invalid_by ~by (fun () ->
      Nx.matrix_norm f64 ~ord:`Two (Nx.zeros Nx.float64 [| 2; 0 |]));
  invalid_by ~by (fun () ->
      Nx.matrix_norm f64 ~ord:`NegTwo (Nx.zeros Nx.complex64 [| 0; 2 |]));
  invalid_by ~by (fun () ->
      Nx.matrix_norm f64 ~ord:`One (Nx.zeros Nx.float64 [| 2; 0 |]));
  invalid_by ~by (fun () ->
      Nx.matrix_norm f64 ~ord:`NegInf (Nx.zeros Nx.float16 [| 0; 2 |]));
  invalid_by ~by (fun () -> Nx.matrix_norm f64 (Nx.ones Nx.bool [| 2; 2 |]))

(* matrix_rank *)

(* A rank [q] matrix [s v] of [m] rows and [n] columns: [s]'s first [q] rows are
   the identity's, [v]'s rows unit vectors moved by at most [0.1] an element, so
   its [q] singular values other than zero are far above [10{^-3}] of the
   greatest, and its others far below. *)
let law_matrix_rank =
  prop "matrix_rank counts the factors of a product"
    Gen.(
      pair (of_list fulls)
        (pair
           (triple (int_range 0 5) (int_range 0 5) (int_range 0 5))
           (array ~size:(constant 50) (float_range (-1.) 1.))))
    (fun (D.Any dt, ((m, n, q), r)) ->
      let q = Stdlib.min q (Stdlib.min m n) in
      cover "deficient" (q < Stdlib.min m n);
      cover "full" (q = Stdlib.min m n && q > 0);
      let s =
        Array.init (m * q) (fun k ->
            let i = k / q and j = k mod q in
            if i >= q then r.(k mod 50) else if i = j then 1. else 0.)
      in
      let v =
        Array.init (q * n) (fun k ->
            let i = k / n and j = k mod n in
            (if i = j then 1. else 0.) +. (0.1 *. r.((k + 25) mod 50)))
      in
      let a = Nx.to_array (Nx.matmul (f64 [| m; q |] s) (f64 [| q; n |] v)) in
      let a = operand dt [| m; n |] a (Array.make (m * n) 0.) in
      equal ~msg:"rank" int64 (Int64.of_int q)
        (Nx.item [] (Nx.matrix_rank ~rtol:1e-3 a)))

let test_matrix_rank_edges () =
  let ranks ?rtol x =
    Array.map Int64.to_int (Nx.to_array (Nx.matrix_rank ?rtol x))
  in
  let batch =
    Nx.stack
      [
        Nx.eye Nx.float64 3;
        Nx.zeros Nx.float64 [| 3; 3 |];
        f64 [| 3; 3 |] [| 1.; 2.; 3.; 2.; 4.; 6.; 1.; 0.; 0. |];
        f64 [| 3; 3 |] [| 1.; Float.nan; 0.; 0.; 1.; 0.; 0.; 0.; 1. |];
        f64 [| 3; 3 |] [| 1.; 0.; 0.; 0.; Float.infinity; 0.; 0.; 0.; 1. |];
      ]
  in
  equal ~msg:"full, zero, deficient, NaN, an infinity" (array int)
    [| 3; 0; 2; -1; -1 |] (ranks batch);
  equal ~msg:"rtol" (array int) [| 1 |]
    (ranks ~rtol:1e-2
       (Nx.reshape [| 1; 2; 2 |] (f64 [| 2; 2 |] [| 1.; 0.; 0.; 1e-3 |])));
  equal ~msg:"no element" (array int) [| 0; 0 |]
    (ranks (Nx.zeros Nx.float32 [| 2; 0; 3 |]))

(* A narrower float counts at float32's epsilon: past [1 / ε] of the format, its
   own epsilon would leave no singular value above the bound. *)
let test_matrix_rank_narrow () =
  let rank x = Nx.item [] (Nx.matrix_rank x) in
  equal ~msg:"bfloat16, 256" int64 256L (rank (Nx.eye Nx.bfloat16 256));
  equal ~msg:"float8_e4m3fn, 16" int64 16L (rank (Nx.eye Nx.float8_e4m3fn 16));
  equal ~msg:"float4_e2m1fn, 4" int64 4L (rank (Nx.eye Nx.float4_e2m1fn 4))

let test_matrix_rank_raises () =
  let by = "Nx.matrix_rank" in
  invalid_by ~by (fun () -> Nx.matrix_rank (Nx.zeros Nx.float32 [| 3 |]));
  invalid_by ~by (fun () -> Nx.matrix_rank (Nx.zeros Nx.int32 [| 2; 2 |]));
  invalid_by ~by (fun () ->
      Nx.matrix_rank ~rtol:(-1.) (Nx.zeros Nx.float32 [| 2; 2 |]));
  invalid_by ~by (fun () ->
      Nx.matrix_rank ~rtol:Float.nan (Nx.zeros Nx.float32 [| 2; 2 |]))

(* cond *)

(* Under `Two cond divides the greatest of the singular values matrix_norm's
   `Two and `NegTwo take by the least, and under `One and `Inf multiplies
   matrix_norm's norms of a and inv a, each at the same precision: one
   rounding to the dtype from those, and one at float64 for the check. *)
(* As matrix_norm's, about one case in six is empty: 200 cases reach it. *)
let test_cond =
  prop ~count:200 "each p is its definition" (square_matrices ())
    (fun (Case (dt, s, _, re, im)) ->
      cover_square_shapes s;
      assume (cols s > 0);
      cover "complex" (D.is D.Complex dt);
      let x = conditioned (operand dt s re im) in
      let (Any_float r) = real_of dt in
      let norm ord y =
        Nx.to_array (Nx.cast Nx.float64 (Nx.matrix_norm r ~ord y))
      in
      let cond p = Nx.to_array (Nx.cast Nx.float64 (Nx.cond r ~p x)) in
      let close =
        array (float_rel ~rel:(2. *. (unit_roundoff dt +. 0x1p-53)) ~abs:0.)
      in
      equal ~msg:"two" close
        (Array.map2 ( /. ) (norm `Two x) (norm `NegTwo x))
        (cond `Two);
      let inv = Nx.inv x in
      equal ~msg:"one" close
        (Array.map2 ( *. ) (norm `One x) (norm `One inv))
        (cond `One);
      equal ~msg:"inf" close
        (Array.map2 ( *. ) (norm `Inf x) (norm `Inf inv))
        (cond `Inf))

let test_cond_edges () =
  let one ?p a = Nx.item [] (Nx.cond Nx.float64 ?p a) in
  let ps = [ `One; `Two; `Inf ] in
  let d = f64 [| 2; 2 |] [| 2.; 0.; 0.; 0.5 |] in
  List.iter (fun p -> equal ~msg:"diagonal" (float 1e-15) 4. (one ~p d)) ps;
  (* [d] with [2i] for [2]: its magnitudes are [d]'s. *)
  let z =
    Nx.create Nx.complex128 [| 2; 2 |]
      Complex.[| { re = 0.; im = 2. }; zero; zero; { re = 0.5; im = 0. } |]
  in
  List.iter
    (fun p -> equal ~msg:"complex diagonal" (float 1e-15) 4. (one ~p z))
    ps;
  equal ~msg:"wide" (array int) [| 2 |]
    (Nx.shape (Nx.cond Nx.float32 (Nx.ones Nx.float32 [| 2; 2; 3 |])));
  equal ~msg:"no matrix" (array int) [| 0 |]
    (Nx.shape (Nx.cond Nx.float32 (Nx.zeros Nx.float32 [| 0; 2; 0 |])))

let test_cond_singular () =
  let one ?p a = Nx.item [] (Nx.cond Nx.float64 ?p a) in
  equal ~msg:"zero" float_exact Float.infinity
    (one (Nx.zeros Nx.float64 [| 2; 2 |]));
  equal ~msg:"singular, one" float_exact Float.infinity
    (one ~p:`One (f64 [| 2; 2 |] [| 1.; 2.; 2.; 4. |]));
  (* [(1 + i) b] for the singular [b]: its pivot is exactly zero. *)
  let b = f64 [| 2; 2 |] [| 1.; 2.; 2.; 4. |] in
  let singular = Nx.complex Nx.complex64 ~re:b ~im:b in
  List.iter
    (fun p ->
      equal ~msg:"complex singular" float_exact Float.infinity (one ~p singular))
    [ `One; `Inf ];
  equal ~msg:"complex zero" float_exact Float.infinity
    (one (Nx.zeros Nx.complex64 [| 2; 2 |]));
  (* A norm that overflows is no infinity the matrix holds. *)
  equal ~msg:"an overflowed norm" float_exact Float.infinity
    (one ~p:`One (f64 [| 2; 2 |] [| 1e308; 0.; 1e308; 1. |]))

let test_cond_alone () =
  let finite = f64 [| 2; 2 |] [| 2.; 1.; 1.; 3. |] in
  let nan = f64 [| 2; 2 |] [| 1.; Float.nan; 0.; 1. |] in
  List.iter
    (fun (name, p) ->
      alone ~msg:name
        (Nx.cond Nx.float64 ~p finite)
        (Nx.cond Nx.float64 ~p (Nx.stack [ finite; nan ]));
      let z =
        Nx.complex Nx.complex128 ~re:nan ~im:(Nx.zeros Nx.float64 [| 2; 2 |])
      in
      equal ~msg:("complex NaN, " ^ name) float_exact Float.nan
        (Nx.item [] (Nx.cond Nx.float64 ~p z)))
    [ ("one", `One); ("two", `Two); ("inf", `Inf) ]

let test_cond_raises () =
  let by = "Nx.cond" in
  List.iter
    (fun (Nx_kernel.Any_float f) ->
      if not (D.float_format f).infinities then
        invalid_by ~by (fun () -> Nx.cond f (Nx.eye Nx.float32 2)))
    float_formats;
  invalid_by ~by (fun () -> Nx.cond Nx.float32 (Nx.zeros Nx.float32 [| 2; 0 |]));
  invalid_by ~by (fun () ->
      Nx.cond Nx.float32 ~p:`One (Nx.ones Nx.float32 [| 2; 3 |]));
  invalid_by ~by (fun () ->
      Nx.cond Nx.float32 ~p:`Inf (Nx.ones Nx.float32 [| 2; 3 |]));
  invalid_by ~by (fun () -> Nx.cond Nx.float32 (Nx.eye Nx.int32 2));
  invalid_by ~by (fun () -> Nx.cond Nx.float32 (Nx.ones Nx.float32 [| 3 |]))

(* Functions of several operations *)

(* The [by]s a rule receives while [f] runs under it, each once, in order. *)
let observed_names f =
  let names = ref [] in
  let observe _ ~by op =
    if not (List.mem by !names) then names := by :: !names;
    Nx.Prim.eval ~by op
  in
  Nx.Prim.interpret ~name:"test.names" Extent observe (fun _ -> ignore (f ()));
  List.rev !names

let test_composites_named () =
  let a = f64 [| 2; 2 |] [| 2.; 1.; 1.; 3. |]
  and b = f64 [| 2 |] [| 1.; 2. |] in
  let z = Nx.complex Nx.complex64 ~re:a ~im:a in
  let zb = Nx.complex Nx.complex64 ~re:b ~im:b in
  let named name f =
    equal ~msg:name (list string) [ name ] (observed_names f)
  in
  named "Nx.cholesky" (fun () -> Nx.cholesky ~triangle:Upper a);
  named "Nx.lu" (fun () -> Nx.lu a);
  named "Nx.svd" (fun () -> Nx.svd Nx.float32 a);
  named "Nx.svdvals" (fun () -> Nx.svdvals Nx.float32 a);
  named "Nx.eig" (fun () -> Nx.eig Nx.complex64 a);
  named "Nx.eigvals" (fun () -> Nx.eigvals Nx.complex64 a);
  named "Nx.eigh" (fun () -> Nx.eigh ~triangle:Upper Nx.float32 a);
  named "Nx.eigvalsh" (fun () -> Nx.eigvalsh ~triangle:Upper Nx.float32 a);
  named "Nx.det" (fun () -> Nx.det a);
  named "Nx.slogdet" (fun () -> Nx.slogdet Nx.float64 z);
  named "Nx.solve_triangular" (fun () ->
      Nx.solve_triangular ~transpose:Transpose z zb);
  named "Nx.solve" (fun () -> Nx.solve a b);
  named "Nx.inv" (fun () -> Nx.inv a);
  named "Nx.lstsq" (fun () -> Nx.lstsq Nx.float64 a b);
  named "Nx.pinv" (fun () -> Nx.pinv a);
  named "Nx.pinv" (fun () -> Nx.pinv ~hermitian:true a);
  named "Nx.vector_norm" (fun () -> Nx.vector_norm Nx.float32 ~ord:3. z);
  named "Nx.matrix_norm" (fun () -> Nx.matrix_norm Nx.float64 ~ord:`Nuc a);
  let narrow = Nx.cast Nx.bfloat16 a in
  named "Nx.matrix_rank" (fun () -> Nx.matrix_rank narrow);
  List.iter
    (fun p -> named "Nx.cond" (fun () -> Nx.cond Nx.float64 ~p a))
    [ `One; `Two; `Inf ]

(* Prim.linalg *)

(* What a routine's kernel was given beyond its arrays. *)
type routine =
  | Cholesky of Nx_kernel.triangle
  | Lu
  | Qr
  | Svd
  | Eigh
  | Eig
  | Solve_triangular of {
      triangle : Nx_kernel.triangle;
      adjoint : bool;
      unit_diagonal : bool;
    }

(* nx.cpu, whose routines record what they are given and store [k + 1] in every
   element of result [k], so that a result names its slot. A routine's
   destinations are its values, then its vectors where it has them. *)
module Filling = struct
  include Nx_cpu

  let name = "nx.filling"
  let seen : (routine * int list list * int list list) list ref = ref []

  let filled r ops dsts =
    let shape (A.Any a) = Iarray.to_list (L.shape (A.layout a)) in
    seen := (r, List.map shape ops, List.map shape dsts) :: !seen;
    List.iteri
      (fun k (A.Any d) ->
        let dt = A.dtype d in
        let bits = P.bits dt (D.of_float dt (Float.of_int (k + 1))) in
        ignore (Nx_kernels_support.map0 (module Nx_cpu) (Fill bits) ~dst:d))
      dsts;
    A.Done

  let cholesky t ~dst a = filled (Cholesky t) [ A.Any a ] [ A.Any dst ]

  let lu ~lu ~pivots ~perm a =
    filled Lu [ A.Any a ] [ A.Any lu; A.Any pivots; A.Any perm ]

  let qr ~q ~r a = filled Qr [ A.Any a ] [ A.Any q; A.Any r ]

  let svd ~s ?vectors a =
    let vectors =
      match vectors with None -> [] | Some (~u, ~vh) -> [ A.Any u; A.Any vh ]
    in
    filled Svd [ A.Any a ] (A.Any s :: vectors)

  let eigen r ~w v a =
    let v = match v with None -> [] | Some v -> [ A.Any v ] in
    filled r [ A.Any a ] (A.Any w :: v)

  let eigh ~w ?v a = eigen Eigh ~w v a
  let eig ~w ?v a = eigen Eig ~w v a

  let solve_triangular triangle ~adjoint ~unit_diagonal ~dst a b =
    filled
      (Solve_triangular { triangle; adjoint; unit_diagonal })
      [ A.Any a; A.Any b ] [ A.Any dst ]
end

let m = Nx_support.memory

(* nx.cpu without lu, svd, eigh and eig. *)
module Declining = struct
  include Nx_cpu

  let name = "nx.declining"
  let lu ~lu:_ ~pivots:_ ~perm:_ _ = A.Declined
  let svd ~s:_ ?vectors:_ _ = A.Declined
  let eigh ~w:_ ?v:_ _ = A.Declined
  let eig ~w:_ ?v:_ _ = A.Declined
end

module One = (val Nx.devices ~kernels:(module Filling) [ m 0 ])
module Two = (val Nx.devices ~kernels:(module Filling) [ m 0; m 1 ])

let read x =
  A.to_array (Option.get (Nx.Repr.array ~by:"test" (Nx.place Nx.Host.on x)))

let zeros dt s =
  Nx.place One.on (Nx.Repr.of_array Nx.Host.v (A.create Rig.host dt s))

let linalg l = Nx.Prim.eval ~by:"t" (Linalg l)

(* Each result's dtype and shape, as Nx.Prim.results gives its forms. *)
type ('v, 's, 'd) Nx.Prim.payload += Slot : ('v, 's, 'd) Nx.Prim.payload

let forms l =
  Nx.Prim.interpret ~name:"t.forms" Values
    (fun _ ~by:_ _ -> fail "t.forms received an operation")
    (fun i ->
      let out = ref [] in
      ignore
        (Nx.Prim.results ~by:"t"
           (fun k (f : _ Nx.Prim.form) ->
             let s = Iarray.to_array (L.shape f.layout) in
             out := (k, D.name f.dtype, s) :: !out;
             Nx.Prim.traced i f Slot)
           (Linalg l));
      List.rev !out)

let pp_form ppf (k, dt, s) =
  Format.fprintf ppf "%d: %s [%s]" k dt
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))

let form = Testable.make ~pp:pp_form ~equal:( = )

let test_forms () =
  let f32 = zeros D.Float32 and c64 = zeros D.Complex64 in
  let check msg want l = equal ~msg (list form) want (forms l) in
  check "cholesky"
    [ (0, "float32", [| 2; 3; 3 |]) ]
    (Cholesky { triangle = Lower; a = f32 [| 2; 3; 3 |] });
  check "lu"
    [
      (0, "float32", [| 2; 3; 4 |]);
      (1, "int64", [| 2; 3 |]);
      (2, "int64", [| 2; 3 |]);
    ]
    (Lu (f32 [| 2; 3; 4 |]));
  check "qr reduced"
    [ (0, "float32", [| 5; 3 |]); (1, "float32", [| 3; 3 |]) ]
    (Qr { factors = Reduced; a = f32 [| 5; 3 |] });
  check "qr complete"
    [ (0, "float32", [| 5; 5 |]); (1, "float32", [| 5; 3 |]) ]
    (Qr { factors = Complete; a = f32 [| 5; 3 |] });
  check "svd reduced"
    [
      (0, "float32", [| 3 |]);
      (1, "float32", [| 3; 3 |]);
      (2, "float32", [| 3; 5 |]);
    ]
    (Svd { factors = Reduced; dtype = Float32; a = f32 [| 3; 5 |] });
  check "svd of a complex matrix"
    [
      (0, "float32", [| 3 |]);
      (1, "complex64", [| 3; 3 |]);
      (2, "complex64", [| 3; 5 |]);
    ]
    (Svd { factors = Reduced; dtype = Float32; a = c64 [| 3; 5 |] });
  check "svd complete"
    [
      (0, "float32", [| 3 |]);
      (1, "float32", [| 3; 3 |]);
      (2, "float32", [| 5; 5 |]);
    ]
    (Svd { factors = Complete; dtype = Float32; a = f32 [| 3; 5 |] });
  check "singular values"
    [ (0, "float64", [| 3 |]) ]
    (Svd_values { dtype = Float64; a = zeros D.Complex128 [| 3; 5 |] });
  check "eigh"
    [ (0, "float32", [| 4 |]); (1, "complex64", [| 4; 4 |]) ]
    (Eigh { dtype = Float32; a = c64 [| 4; 4 |] });
  check "eigh's values"
    [ (0, "float32", [| 4 |]) ]
    (Eigh_values { dtype = Float32; a = f32 [| 4; 4 |] });
  check "eig"
    [ (0, "complex64", [| 4 |]); (1, "complex64", [| 4; 4 |]) ]
    (Eig { dtype = Complex64; a = c64 [| 4; 4 |] });
  check "eig of a real matrix"
    [ (0, "complex128", [| 4 |]); (1, "complex128", [| 4; 4 |]) ]
    (Eig { dtype = Complex128; a = zeros D.Float64 [| 4; 4 |] });
  check "eig's values"
    [ (0, "complex64", [| 4 |]) ]
    (Eig_values { dtype = Complex64; a = f32 [| 4; 4 |] });
  check "solve"
    [ (0, "float32", [| 2; 3; 4 |]) ]
    (Solve_triangular
       {
         triangle = Upper;
         adjoint = true;
         unit_diagonal = false;
         a = f32 [| 2; 3; 3 |];
         b = f32 [| 2; 3; 4 |];
       })

let test_rules () =
  let raises f = raises_match (Exn.invalid_arg ~sub:"t: ") f in
  Filling.seen := [];
  raises (fun () ->
      linalg (Cholesky { triangle = Lower; a = zeros D.Float32 [| 2; 3 |] }));
  raises (fun () -> linalg (Lu (zeros D.Int32 [| 2; 2 |])));
  raises (fun () ->
      linalg (Eigh { dtype = Float32; a = zeros D.Float16 [| 2; 2 |] }));
  raises (fun () ->
      linalg (Svd_values { dtype = Float64; a = zeros D.Float64 [| 3 |] }));
  (* A real result of another precision than [a]'s. *)
  let into sub f = raises_match (Exn.invalid_arg ~sub:("t: " ^ sub)) f in
  into "Svd of complex64 into float64" (fun () ->
      linalg (Svd_values { dtype = Float64; a = zeros D.Complex64 [| 3; 3 |] }));
  into "Svd of float64 into float32" (fun () ->
      linalg
        (Svd
           {
             factors = Reduced;
             dtype = Float32;
             a = zeros D.Float64 [| 3; 3 |];
           }));
  into "Eigh of complex128 into float32" (fun () ->
      linalg (Eigh { dtype = Float32; a = zeros D.Complex128 [| 2; 2 |] }));
  into "Eigh of float32 into float16" (fun () ->
      linalg (Eigh_values { dtype = Float16; a = zeros D.Float32 [| 2; 2 |] }));
  raises (fun () ->
      linalg (Eig_values { dtype = Complex64; a = zeros D.Float64 [| 2; 2 |] }));
  raises (fun () ->
      linalg
        (Solve_triangular
           {
             triangle = Lower;
             adjoint = false;
             unit_diagonal = false;
             a = zeros D.Float64 [| 3; 3 |];
             b = zeros D.Float64 [| 2; 4 |];
           }));
  equal ~msg:"kernel calls" int 0 (List.length !Filling.seen)

(* Filling stores [k + 1] in every element of result [k]: each result comes back
   from its routine's slot. *)
let test_order () =
  let a = zeros D.Float64 [| 2; 3; 3 |] in
  let lu, pivots, perm = linalg (Lu a) in
  equal ~msg:"lu, pivots, perm"
    (list (array float_exact))
    [ Array.make 18 1.; Array.make 6 2.; Array.make 6 3. ]
    [
      read lu;
      Array.map Int64.to_float (read pivots);
      Array.map Int64.to_float (read perm);
    ];
  let s, u, vh = linalg (Svd { factors = Reduced; dtype = Float64; a }) in
  equal ~msg:"s, u, vh"
    (list (array float_exact))
    [ Array.make 6 1.; Array.make 18 2.; Array.make 18 3. ]
    [ read s; read u; read vh ];
  let q, r =
    linalg (Qr { factors = Complete; a = zeros D.Float64 [| 4; 2 |] })
  in
  equal ~msg:"q, r"
    (list (array float_exact))
    [ Array.make 16 1.; Array.make 8 2. ]
    [ read q; read r ];
  let x =
    linalg
      (Solve_triangular
         {
           triangle = Upper;
           adjoint = false;
           unit_diagonal = true;
           a;
           b = zeros D.Float64 [| 2; 3; 2 |];
         })
  in
  equal ~msg:"solve_triangular" (array float_exact) (Array.make 12 1.) (read x)

(* What each constructor's routine received: its attributes and how many
   destinations, the values alone one, for s or w, and the factorisations their
   vectors too. *)
let test_without_vectors () =
  Filling.seen := [];
  let sq = zeros D.Complex64 [| 3; 3 |] in
  ignore (linalg (Svd_values { dtype = Float32; a = sq }));
  ignore (linalg (Svd { factors = Reduced; dtype = Float32; a = sq }));
  ignore (linalg (Eigh_values { dtype = Float32; a = sq }));
  ignore (linalg (Eigh { dtype = Float32; a = sq }));
  ignore (linalg (Eig_values { dtype = Complex64; a = sq }));
  ignore (linalg (Eig { dtype = Complex64; a = sq }));
  let name = function
    | Svd -> "svd"
    | Eigh -> "eigh"
    | Eig -> "eig"
    | Cholesky _ | Lu | Qr | Solve_triangular _ -> "other"
  in
  equal ~msg:"routines and destinations"
    (list (pair string int))
    [ ("svd", 1); ("svd", 3); ("eigh", 1); ("eigh", 2); ("eig", 1); ("eig", 2) ]
    (List.rev_map
       (fun (r, _, dsts) -> (name r, List.length dsts))
       !Filling.seen)

let test_split () =
  Filling.seen := [];
  let a =
    Nx.place (Two.split ~axis:0)
      (Nx.Repr.of_array Nx.Host.v (A.create Rig.host D.Float32 [| 2; 3; 3 |]))
  in
  let w, v = linalg (Eigh { dtype = Float32; a }) in
  equal ~msg:"each device its matrices"
    (list (list (list int)))
    [ [ [ 1; 3; 3 ] ]; [ [ 1; 3; 3 ] ] ]
    (List.map (fun (_, ops, _) -> ops) !Filling.seen);
  equal ~msg:"w" (array float_exact) (Array.make 6 1.) (read w);
  equal ~msg:"v" (array float_exact) (Array.make 18 2.) (read v)

(* Matrices of no columns factor into an empty [l] and [u] and a permutation
   with elements, which every device of a cut batch computes. *)
let test_split_empty () =
  let module Cpu2 = (val Nx.devices [ m 0; m 1 ]) in
  let a = Nx.zeros D.Float32 [| 4; 3; 0 |] in
  let perm, l, u = Nx.lu (Nx.place (Cpu2.split ~axis:0) a) in
  equal ~msg:"l" (array int) [| 4; 3; 0 |] (Nx.shape l);
  equal ~msg:"u" (array int) [| 4; 0; 0 |] (Nx.shape u);
  equal ~msg:"perm" (array int64)
    (Array.concat (List.init 4 (fun _ -> [| 0L; 1L; 2L |])))
    (Nx.to_array (Nx.place Nx.Host.on perm))

let test_declined () =
  let module Declines = (val Nx.devices ~kernels:(module Declining) [ m 0 ]) in
  let a =
    Nx.place Declines.on
      (Nx.Repr.of_array Nx.Host.v (A.create Rig.host D.Complex64 [| 2; 2 |]))
  in
  match linalg (Eig_values { dtype = D.Complex64; a }) with
  | _ -> fail "a declined factorisation computed"
  | exception Invalid_argument e ->
      List.iter
        (fun sub -> contains ~msg:sub ~sub e)
        [ "t: "; "nx.declining"; "Eig"; "complex64"; "Nx.place" ]

let () =
  exit
    (run "nx linalg"
       [
         group "linear algebra"
           [
             test_rounded_once;
             test
               "a NaN or an infinity in a dtype without it is what cast gives"
               test_unheld;
             test_precision;
             test "a narrow float's norm, rank and condition compute at float32"
               test_narrow;
           ];
         group "cholesky"
           [
             test_cholesky;
             test_cholesky_form;
             test
               "a matrix it fails on is NaN alone, and the upper triangle is \
                not read"
               test_cholesky_alone;
             test
               "raises for fewer than two axes, a matrix not square, or its \
                dtype"
               test_cholesky_raises;
           ];
         group "qr"
           [
             test_qr;
             test "raises for fewer than two axes or its dtype" test_qr_raises;
           ];
         group "lu"
           [
             test_lu;
             test "raises for fewer than two axes or its dtype" test_lu_raises;
           ];
         group "svd"
           [
             test_svd;
             test "a zero singular value is +0" test_svd_zero;
             test
               "a matrix holding NaN or an infinity is NaN alone in all three"
               test_svd_alone;
             test "raises for fewer than two axes or its dtype" test_svd_raises;
           ];
         group "svdvals"
           [
             test_svdvals;
             test "one element, k = 0 and no matrix give their values"
               test_svdvals_edges;
             test "a matrix holding NaN or an infinity is NaN alone"
               test_svdvals_alone;
             test "raises for fewer than two axes or its dtype"
               test_svdvals_raises;
           ];
         group "eig"
           [
             test_eig;
             test "a matrix holding NaN is NaN alone in both" test_eig_alone;
             test
               "raises for fewer than two axes, a matrix not square, or its \
                dtype"
               test_eig_raises;
           ];
         group "eigvals"
           [
             test_eigvals;
             test_eigvals_hermitian;
             test "a matrix holding NaN or an infinity is NaN alone"
               test_eigvals_alone;
             test "raises as eig" test_eigvals_raises;
           ];
         group "eigh"
           [
             test_eigh;
             test "a matrix holding NaN in what it reads is NaN alone"
               test_eigh_alone;
             test
               "raises for fewer than two axes, a matrix not square, or its \
                dtype"
               test_eigh_raises;
           ];
         group "eigvalsh"
           [
             test_eigvalsh;
             test
               "a matrix holding NaN or an infinity in what it reads is NaN \
                alone"
               test_eigvalsh_alone;
             test "raises as eigh" test_eigvalsh_raises;
           ];
         group "det"
           [
             test_det;
             test "a zero determinant is +0" test_det_zero;
             test
               "no row gives 1, a swap -1, and the product overflows where it \
                does"
               test_det_edges;
             test "a matrix holding NaN is NaN alone, in slogdet too"
               test_det_alone;
             test
               "raises for fewer than two axes, a matrix not square, or its \
                dtype"
               test_det_raises;
           ];
         group "slogdet"
           [
             test_slogdet;
             test
               "a singular matrix has sign +0 and logabs neg_infinity; logabs \
                holds determinants beyond the range"
               test_slogdet_edges;
             test_slogdet_overflow;
             test_slogdet_infinite;
             test "raises as det" test_slogdet_raises;
           ];
         group "solve_triangular"
           [
             test_solve_triangular;
             test_solve_triangular_shapes;
             test "a zero on a diagonal it reads makes that matrix's x NaN"
               test_solve_triangular_alone;
             test
               "raises for a, its dtype, b's rank or rows, or leading axes \
                that do not broadcast"
               test_solve_triangular_raises;
           ];
         group "solve"
           [
             test_solve;
             test
               "a pivot of exactly zero makes that matrix's x NaN, and its \
                inverse"
               test_solve_alone;
             test "raises as solve_triangular" test_solve_raises;
           ];
         group "inv"
           [
             test_inv;
             test
               "raises for fewer than two axes, a matrix not square, or its \
                dtype"
               test_inv_raises;
           ];
         group "lstsq"
           [
             test_lstsq_normal;
             test_lstsq_least;
             test_lstsq_residuals;
             test "a repeated column has rank 1, and a given rtol drops a value"
               test_lstsq_rank;
             test "rtol defaults to max m n times the machine epsilon"
               test_default_rtol;
             test "a matrix svd fails on has rank -1 and is NaN alone"
               test_lstsq_alone;
             test "raises for a, its dtype, b, leading axes or rtol"
               test_lstsq_raises;
           ];
         group "pinv"
           [
             test_pinv;
             test_pinv_hermitian;
             law_rtol;
             test
               "a given rtol drops a value, and ~hermitian reads the lower \
                triangle"
               test_pinv_edges;
             test "a matrix the factorisation fails on is NaN alone"
               test_pinv_alone;
             test
               "raises for a, a matrix not square under ~hermitian, its dtype \
                or rtol"
               test_pinv_raises;
           ];
         group "vector_norm"
           [
             test_vector_norm;
             test "the sums of powers neither overflow nor underflow"
               test_vector_norm_scaled;
             test "no element, order 0, NaN, complex moduli, axes and keepdims"
               test_vector_norm_edges;
             test
               "raises for an axis, a NaN ord, an extreme ord over no element, \
                or its dtype"
               test_vector_norm_raises;
           ];
         group "matrix_norm"
           [
             test_matrix_norm;
             test "a matrix of no element, no matrix, and NaN"
               test_matrix_norm_edges;
             test
               "raises for fewer than two axes, an ord a matrix has none of, \
                or its dtype"
               test_matrix_norm_raises;
           ];
         group "matrix_rank"
           [
             law_matrix_rank;
             test "a batch's ranks, -1 where svdvals is NaN, 0 of no element"
               test_matrix_rank_edges;
             test "a narrower float counts at float32's epsilon"
               test_matrix_rank_narrow;
             test "raises for fewer than two axes, its dtype, or rtol"
               test_matrix_rank_raises;
           ];
         group "cond"
           [
             test_cond;
             test "diagonal and complex matrices, and their shapes"
               test_cond_edges;
             test "a singular matrix has an infinite condition number"
               test_cond_singular;
             test "a matrix holding NaN has NaN alone" test_cond_alone;
             test
               "raises for a dtype of no infinity, no element, p and a matrix \
                not square, or its dtype"
               test_cond_raises;
           ];
         group "refusals"
           [
             test
               "every operation of a function is named by the function called"
               test_composites_named;
           ];
         group "Prim.linalg"
           [
             test "results gives each routine Shape's forms" test_forms;
             test "breaking its rule raises naming by, before any kernel"
               test_rules;
             test "results come back in the routine's order" test_order;
             test "a values constructor's routine computes no vectors"
               test_without_vectors;
             test
               "a batch split across devices factors each device's matrices \
                there"
               test_split;
             test
               "a cut batch of matrices of no columns gives each its \
                permutation"
               test_split_empty;
             test "a declined routine raises naming the move" test_declined;
           ];
       ])
