(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Random compositions of nx functions of two float arrays of one shape, their
   operands drawn to break code: zero-size and one-element shapes, views, NaN,
   -0, infinities. *)

open Windtrap

type un =
  | Neg
  | Abs
  | Sin
  | Exp
  | Sqrt
  | Tanh
  | Square
  | Log
  | Sigmoid
  | Cumsum of int
  | Flip of int
  | Sum of int
  | Max of int
  | Mean of int

type bin = Add | Sub | Mul | Div | Maximum | Minimum | Where_less | Pow
type expr = X | Y | K of float | Un of un * expr | Bin of bin * expr * expr

let pp_un ppf = function
  | Neg -> Format.pp_print_string ppf "neg"
  | Abs -> Format.pp_print_string ppf "abs"
  | Sin -> Format.pp_print_string ppf "sin"
  | Exp -> Format.pp_print_string ppf "exp"
  | Sqrt -> Format.pp_print_string ppf "sqrt"
  | Tanh -> Format.pp_print_string ppf "tanh"
  | Square -> Format.pp_print_string ppf "square"
  | Log -> Format.pp_print_string ppf "log"
  | Sigmoid -> Format.pp_print_string ppf "sigmoid"
  | Cumsum a -> Format.fprintf ppf "cumsum~%d" a
  | Flip a -> Format.fprintf ppf "flip~%d" a
  | Sum a -> Format.fprintf ppf "sum~%d" a
  | Max a -> Format.fprintf ppf "max~%d" a
  | Mean a -> Format.fprintf ppf "mean~%d" a

let pp_bin ppf b =
  Format.pp_print_string ppf
    (match b with
    | Add -> "add"
    | Sub -> "sub"
    | Mul -> "mul"
    | Div -> "div"
    | Maximum -> "maximum"
    | Minimum -> "minimum"
    | Where_less -> "where_less"
    | Pow -> "pow")

let rec pp ppf = function
  | X -> Format.pp_print_string ppf "x"
  | Y -> Format.pp_print_string ppf "y"
  | K c -> Format.fprintf ppf "%h" c
  | Un (u, e) -> Format.fprintf ppf "%a(%a)" pp_un u pp e
  | Bin (b, e, e') -> Format.fprintf ppf "%a(%a, %a)" pp_bin b pp e pp e'

let un (type s d) u (x : (float, s, d) Nx.t) : (float, s, d) Nx.t =
  match u with
  | Neg -> Nx.neg x
  | Abs -> Nx.abs x
  | Sin -> Nx.sin x
  | Exp -> Nx.exp x
  | Sqrt -> Nx.sqrt x
  | Tanh -> Nx.tanh x
  | Square -> Nx.square x
  | Log -> Nx.log x
  | Sigmoid -> Nx.sigmoid x
  | Cumsum a -> Nx.cumsum ~axis:a x
  | Flip a -> Nx.flip ~axes:[ a ] x
  | Sum a -> Nx.sum ~axes:[ a ] ~keepdims:true x
  | Max a -> Nx.max ~axes:[ a ] ~keepdims:true x
  | Mean a -> Nx.mean ~axes:[ a ] ~keepdims:true x

let bin (type s d) b (x : (float, s, d) Nx.t) y : (float, s, d) Nx.t =
  match b with
  | Add -> Nx.add x y
  | Sub -> Nx.sub x y
  | Mul -> Nx.mul x y
  | Div -> Nx.div x y
  | Maximum -> Nx.maximum x y
  | Minimum -> Nx.minimum x y
  | Where_less -> Nx.where (Nx.less x y) x y
  | Pow -> Nx.pow x y

let rec eval : type s d.
    (float, s, d) Nx.t -> (float, s, d) Nx.t -> expr -> (float, s, d) Nx.t =
 fun x y -> function
  | X -> x
  | Y -> y
  | K c -> Nx.scalar_like x c
  | Un (u, e) -> un u (eval x y e)
  | Bin (b, e, e') -> bin b (eval x y e) (eval x y e')

let rec size = function
  | X | Y | K _ -> 1
  | Un (_, e) -> 1 + size e
  | Bin (_, e, e') -> 1 + size e + size e'

(* An expression over [rank] axes, of depth at most [depth]. *)
let expr ~rank =
  let open Gen in
  let leaf =
    frequency
      [
        (3, constant X);
        (3, constant Y);
        ( 1,
          map (fun c -> K c) (of_list [ 0.; -0.; 0.5; 2.; -3.; Float.infinity ])
        );
      ]
  in
  let axis = int_range 0 (Int.max 0 (rank - 1)) in
  let unary =
    let plain = [ Neg; Abs; Sin; Exp; Sqrt; Tanh; Square; Log; Sigmoid ] in
    if rank = 0 then of_list plain
    else
      frequency
        [
          (3, of_list plain);
          ( 2,
            let* a = axis in
            of_list [ Cumsum a; Flip a; Sum a; Max a; Mean a ] );
        ]
  in
  let binary =
    of_list [ Add; Sub; Mul; Div; Maximum; Minimum; Where_less; Pow ]
  in
  let rec go depth =
    if depth = 0 then leaf
    else
      frequency
        [
          (1, leaf);
          ( 2,
            let* u = unary in
            let+ e = go (depth - 1) in
            Un (u, e) );
          ( 3,
            let* b = binary in
            let* e = go (depth - 1) in
            let+ e' = go (depth - 1) in
            Bin (b, e, e') );
        ]
  in
  go 5

(* Exact compositions

   A compiled call gives eager's bits for the functions both compute exactly
   (Rune.jit's classes): movements, extremes, comparisons, selections, and
   IEEE 754's correctly rounded arithmetic. [exact e] is [e] with each other
   function of floats replaced by an exact one; [elementwise e] also replaces
   each reduction by a movement, so that no derivative of [e] sums terms, as a
   reduction's transpose or its ties' share does. *)

let rec exact = function
  | (X | Y | K _) as e -> e
  | Un (u, e) ->
      let u =
        match u with
        | Sin | Log -> Neg
        | Exp | Tanh | Sigmoid -> Abs
        | Cumsum a | Sum a | Mean a -> Max a
        | (Neg | Abs | Sqrt | Square | Flip _ | Max _) as u -> u
      in
      Un (u, exact e)
  | Bin (b, e, e') ->
      let b = match b with Pow -> Mul | b -> b in
      Bin (b, exact e, exact e')

let rec elementwise = function
  | (X | Y | K _) as e -> e
  | Un (Max a, e) -> Un (Flip a, elementwise e)
  | Un (u, e) -> Un (u, elementwise e)
  | Bin (b, e, e') -> Bin (b, elementwise e, elementwise e')

(* [exact_any dt e] is the [generic] expression [e] with the functions not
   exact at [dt] replaced: a complex number's magnitude, products, sums and
   extremes along an axis, which leave a NaN's parts unspecified, and a float's
   sums. *)
let rec exact_any : type v s. (v, s) Nx.dtype -> expr -> expr =
 fun dt e ->
  let complex = Nx.Dtype.is Complex dt and float = Nx.Dtype.is Float dt in
  match e with
  | (X | Y | K _) as e -> e
  | Un (u, e) ->
      let u =
        match u with
        | (Abs | Square) when complex -> Neg
        | (Sum a | Cumsum a | Max a) when complex -> Flip a
        | (Sum a | Cumsum a) when float -> Max a
        | u -> u
      in
      Un (u, exact_any dt e)
  | Bin (Mul, e, e') when complex -> Bin (Add, exact_any dt e, exact_any dt e')
  | Bin (b, e, e') -> Bin (b, exact_any dt e, exact_any dt e')

(* Operands *)

let pool =
  [|
    0.;
    -0.;
    1.;
    -1.;
    0.5;
    -2.5;
    3.;
    1e-3;
    1e30;
    -1e30;
    Float.infinity;
    Float.neg_infinity;
    Float.nan;
    7.25;
    -0.125;
  |]

type view = Plain | Flipped | Strided | Broadcast

let pp_view ppf v =
  Format.pp_print_string ppf
    (match v with
    | Plain -> "plain"
    | Flipped -> "flipped"
    | Strided -> "strided"
    | Broadcast -> "broadcast")

let shape =
  let open Gen in
  let* r = int_range 0 3 in
  array ~size:(constant r)
    (frequency [ (1, constant 0); (2, constant 1); (4, int_range 2 4) ])

let numel s = Array.fold_left ( * ) 1 s

let pp_shape ppf s =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))

type operand = { view : view; values : float array }

let operand s =
  let open Gen in
  let* view = of_list [ Plain; Flipped; Strided; Broadcast ] in
  let n =
    match view with
    | Plain | Flipped -> numel s
    | Strided -> numel s * 2
    | Broadcast -> if Array.length s = 0 then 1 else numel s / Int.max 1 s.(0)
  in
  let+ values =
    array ~size:(constant n)
      (frequency
         [
           (3, map (fun i -> pool.(i)) (int_range 0 (Array.length pool - 1)));
           (2, float_range (-4.) 4.);
         ])
  in
  { view; values }

(* The operand of shape [s] at [dt], as a view of the kind drawn. *)
let make (type s) (dt : (float, s) Nx.dtype) s o : (float, s, Nx.host) Nx.t =
  let r = Array.length s in
  match o.view with
  | Plain -> Nx.create dt s o.values
  | Flipped -> Nx.flip (Nx.create dt s o.values)
  | Strided when r = 0 -> Nx.create dt s (Array.sub o.values 0 1)
  | Strided ->
      let s' = Array.copy s in
      s'.(r - 1) <- 2 * s.(r - 1);
      Nx.slice
        (List.init r (fun i -> if i = r - 1 then Nx.Rs (0, s'.(i), 2) else Nx.A))
        (Nx.create dt s' o.values)
  | Broadcast when r = 0 -> Nx.create dt s o.values
  | Broadcast ->
      let s1 = Array.copy s in
      s1.(0) <- 1;
      if s.(0) = 0 then Nx.create dt s [||]
      else Nx.broadcast_to s (Nx.create dt s1 o.values)

(* Results compared bit for bit *)

type 'a outcome = Value of 'a | Raised of string

let outcome f =
  match f () with v -> Value v | exception Invalid_argument m -> Raised m

(* A float's bits, every NaN one code: nx leaves a NaN result's sign and payload
   unspecified. *)
let code f = if Float.is_nan f then Int64.minus_one else Int64.bits_of_float f

let bits (type s d) (x : (float, s, d) Nx.t) =
  (Nx.shape x, Array.map code (Nx.to_array x))

let bits_t =
  Testable.make
    ~pp:(fun ppf (s, b) ->
      Format.fprintf ppf "%a %a" pp_shape s
        (Format.pp_print_list
           ~pp_sep:(fun ppf () -> Format.pp_print_string ppf " ")
           (fun ppf b -> Format.fprintf ppf "%h" (Int64.float_of_bits b)))
        (Array.to_list b))
    ~equal:( = )

let outcome_t t =
  Testable.make
    ~pp:(fun ppf -> function
      | Value v -> Testable.pp t ppf v
      | Raised m -> Format.fprintf ppf "raised %S" m)
    ~equal:(fun a b ->
      match (a, b) with
      | Value a, Value b -> Testable.equal t a b
      | Raised a, Raised b -> String.equal a b
      | _ -> false)

(* Every dtype *)

(* [e] with the functions only floats have replaced by ones every dtype has. *)
let rec generic = function
  | (X | Y | K _) as e -> e
  | Un (u, e) ->
      let u =
        match u with
        | Sin | Exp | Sqrt | Log -> Abs
        | Tanh | Sigmoid -> Square
        | Mean a -> Sum a
        | u -> u
      in
      Un (u, generic e)
  | Bin (b, e, e') ->
      let b = match b with Div | Pow -> Mul | b -> b in
      Bin (b, generic e, generic e')

let un_any (type v s d) u (x : (v, s, d) Nx.t) : (v, s, d) Nx.t =
  match u with
  | Neg -> Nx.neg x
  | Abs -> Nx.abs x
  | Square -> Nx.square x
  | Cumsum a -> Nx.cumsum ~axis:a x
  | Flip a -> Nx.flip ~axes:[ a ] x
  | Sum a -> Nx.sum ~axes:[ a ] ~keepdims:true x
  | Max a -> Nx.max ~axes:[ a ] ~keepdims:true x
  | Sin | Exp | Sqrt | Tanh | Log | Sigmoid | Mean _ ->
      invalid_arg "Compose.un_any: a float function"

let bin_any (type v s d) b (x : (v, s, d) Nx.t) y : (v, s, d) Nx.t =
  match b with
  | Add -> Nx.add x y
  | Sub -> Nx.sub x y
  | Mul -> Nx.mul x y
  | Maximum -> Nx.maximum x y
  | Minimum -> Nx.minimum x y
  | Where_less -> Nx.where (Nx.less x y) x y
  | Div | Pow -> invalid_arg "Compose.bin_any: a float function"

let rec eval_any : type v s d.
    (v, s, d) Nx.t -> (v, s, d) Nx.t -> expr -> (v, s, d) Nx.t =
 fun x y -> function
  | X -> x
  | Y -> y
  | K c -> Nx.scalar_like x (Nx.Dtype.of_float (Nx.dtype x) c)
  | Un (u, e) -> un_any u (eval_any x y e)
  | Bin (b, e, e') -> bin_any b (eval_any x y e) (eval_any x y e')

(* The operand of shape [s] at [dt], its values stored from float64. *)
let make_any (type v s) (dt : (v, s) Nx.dtype) s o : (v, s, Nx.host) Nx.t =
  let r = Array.length s in
  let create s vs = Nx.cast dt (Nx.create Nx.float64 s vs) in
  match o.view with
  | Plain -> create s o.values
  | Flipped -> Nx.flip (create s o.values)
  | Strided when r = 0 -> create s (Array.sub o.values 0 1)
  | Strided ->
      let s' = Array.copy s in
      s'.(r - 1) <- 2 * s.(r - 1);
      Nx.slice
        (List.init r (fun i -> if i = r - 1 then Nx.Rs (0, s'.(i), 2) else Nx.A))
        (create s' o.values)
  | Broadcast when r = 0 -> create s o.values
  | Broadcast ->
      let s1 = Array.copy s in
      s1.(0) <- 1;
      if s.(0) = 0 then create s [||]
      else Nx.broadcast_to s (create s1 o.values)

(* Elements as text that tells every two values apart: a float's bits, a complex
   number's two parts' bits, and other values as printed. *)
let codes (type v s d) (x : (v, s, d) Nx.t) =
  let dt = Nx.dtype x in
  let code : v -> string =
    match Nx.Dtype.kind dt with
    | Float -> fun f -> Printf.sprintf "%Lx" (code f)
    | Complex ->
        fun c -> Printf.sprintf "%Lx,%Lx" (code c.Complex.re) (code c.im)
    | _ -> Format.asprintf "%a" (Nx.Dtype.pp_value dt)
  in
  (Nx.shape x, Array.map code (Nx.to_array x))

let codes_t =
  Testable.make
    ~pp:(fun ppf (s, c) ->
      Format.fprintf ppf "%a [%s]" pp_shape s
        (String.concat " " (Array.to_list c)))
    ~equal:( = )

(* Bits, but for zeros' signs, and with [nan] NaNs the second value has: a
   loop's derivative holds a tangent or cotangent of a carry at every step, a
   zero where a step gives none, whose products are zeros of either sign, and
   NaNs with an infinite partial derivative, where the steps written out have no
   term; and it sums a row's contributions as stacked rows, where the steps
   written out add each row's slice padded with [+0.]. *)
let lenient ~nan =
  let zero x = x = 0L || x = Int64.min_int in
  Testable.make ~pp:(Testable.pp bits_t) ~equal:(fun (s, a) (s', b) ->
      s = s'
      && Array.length a = Array.length b
      && Array.for_all2
           (fun x y ->
             x = y || (nan && y = Int64.minus_one) || (zero x && zero y))
           a b)

(* Whether [e] reads [X]. *)
let rec reads_x = function
  | X -> true
  | Y | K _ -> false
  | Un (_, e) -> reads_x e
  | Bin (_, e, e') -> reads_x e || reads_x e'

(* Whether [e] reads [Y]. *)
let rec reads_y = function
  | Y -> true
  | X | K _ -> false
  | Un (_, e) -> reads_y e
  | Bin (_, e, e') -> reads_y e || reads_y e'

(* Whether [e] reads neither [X] nor [Y]. *)
let constant e =
  let rec reads = function
    | X | Y -> true
    | K _ -> false
    | Un (_, e) -> reads e
    | Bin (_, e, e') -> reads e || reads e'
  in
  not (reads e)
