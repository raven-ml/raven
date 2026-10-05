(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Compiled programs against eager ones. A program is a small expression over
   nx's public operations, drawn with its dtypes and shapes; [Rune.jit] of it,
   on the host, computes eager's bits.

   The programs keep to what the jit's numerics paragraph promises bit for bit.
   Float sums over an axis, transcendental functions and float products are left
   out, since their bits may differ. So are a multiply feeding an add, which the
   compiler may fuse, a division by a captured constant, which it may turn into
   a multiplication, and a multiplication by a reciprocal, which it may turn
   into a division. The sign of a zero that {!Nx.maximum} or {!Nx.minimum}
   computes is the target's, so a program holding one compares zeros up to sign,
   and nothing whose value turns on that sign (a division, a reciprocal) reads
   it. Programs that end in one of the operations left out are compared within
   the rounding the jit's documentation allows them. *)

open Windtrap

(* Programs *)

type dt = F32 | F16 | I32 | Bool | I64

type leaf = {
  dt : dt;
  shape : int array;
  values : float array; (* integers and booleans held exactly *)
  capture : bool; (* a constant of the compiled function, else an argument *)
}

type un =
  | Neg
  | Abs
  | Sign
  | Square
  | Sqrt
  | Recip
  | Floor
  | Ceil
  | Round
  | Trunc
  | Not

type bin = Add | Sub | Mul | Div | Maximum | Minimum | And | Or | Xor
type cmp = Less | Equal
type red = Sum | Prod | Max | Min
type spec = Range of int * int * int | Index of int

type expr =
  | Leaf of leaf
  | Un of un * expr
  | Isnan of expr
  | Bin of bin * expr * expr
  | Cmp of cmp * expr * expr
  | Where of expr * expr * expr
  | Cast of dt * expr
  | Reduce of red * int list * bool * expr
  | Reshape of int array * expr
  | Contiguous of expr
  | Transpose of int list * expr
  | Slice of spec list * expr
  | Broadcast of int array * expr
  | Pad of (int * int) array * float * expr
  | Take of int option * expr * expr
  | Matmul of expr * expr

let is_float = function F32 | F16 -> true | I32 | Bool | I64 -> false
let numel shape = Array.fold_left ( * ) 1 shape

(* The leaves of [e], left to right: the order arguments are passed in. *)
let rec leaves e =
  match e with
  | Leaf l -> [ l ]
  | Un (_, a)
  | Isnan a
  | Cast (_, a)
  | Reduce (_, _, _, a)
  | Reshape (_, a)
  | Contiguous a
  | Transpose (_, a)
  | Slice (_, a)
  | Broadcast (_, a)
  | Pad (_, _, a) ->
      leaves a
  | Bin (_, a, b) | Cmp (_, a, b) | Matmul (a, b) | Take (_, a, b) ->
      leaves a @ leaves b
  | Where (c, a, b) -> leaves c @ leaves a @ leaves b

let rec exists p e =
  p e
  ||
  match e with
  | Leaf _ -> false
  | Un (_, a)
  | Isnan a
  | Cast (_, a)
  | Reduce (_, _, _, a)
  | Reshape (_, a)
  | Contiguous a
  | Transpose (_, a)
  | Slice (_, a)
  | Broadcast (_, a)
  | Pad (_, _, a) ->
      exists p a
  | Bin (_, a, b) | Cmp (_, a, b) | Matmul (a, b) | Take (_, a, b) ->
      exists p a || exists p b
  | Where (c, a, b) -> exists p c || exists p a || exists p b

(* Whether [e] computes a maximum or a minimum, whose zero's sign is the
   target's under a compiled function. *)
let signs_zeros =
  exists (function
    | Bin ((Maximum | Minimum), _, _) | Reduce ((Max | Min), _, _, _) -> true
    | _ -> false)

(* Printing, as the OCaml that builds the program *)

let dtype_name = function
  | F32 -> "Nx.float32"
  | F16 -> "Nx.float16"
  | I32 -> "Nx.int32"
  | Bool -> "Nx.bool"
  | I64 -> "Nx.int64"

let pp_ints ppf a =
  Format.fprintf ppf "[|%s|]"
    (String.concat "; " (Array.to_list (Array.map string_of_int a)))

let pp_float ppf x =
  if Float.is_nan x then Format.pp_print_string ppf "Float.nan"
  else if x = Float.infinity then Format.pp_print_string ppf "Float.infinity"
  else if x = Float.neg_infinity then
    Format.pp_print_string ppf "Float.neg_infinity"
  else if Float.sign_bit x then Format.fprintf ppf "(%h)" x
  else Format.fprintf ppf "%h" x

let pp_value dt ppf x =
  match dt with
  | F32 | F16 -> pp_float ppf x
  | I32 -> Format.fprintf ppf "%.0fl" x
  | I64 -> Format.fprintf ppf "%.0fL" x
  | Bool -> Format.pp_print_bool ppf (x <> 0.)

let un_name dt = function
  | Neg -> "neg"
  | Abs -> "abs"
  | Sign -> "sign"
  | Square -> "square"
  | Sqrt -> "sqrt"
  | Recip -> "recip"
  | Floor -> "floor"
  | Ceil -> "ceil"
  | Round -> "round"
  | Trunc -> "trunc"
  | Not -> if dt = Bool then "logical_not" else "bitwise_not"

let bin_name dt = function
  | Add -> "add"
  | Sub -> "sub"
  | Mul -> "mul"
  | Div -> "div"
  | Maximum -> "maximum"
  | Minimum -> "minimum"
  | And -> if dt = Bool then "logical_and" else "bitwise_and"
  | Or -> if dt = Bool then "logical_or" else "bitwise_or"
  | Xor -> if dt = Bool then "logical_xor" else "bitwise_xor"

let red_name = function
  | Sum -> "sum"
  | Prod -> "prod"
  | Max -> "max"
  | Min -> "min"

(* The dtype [e] computes. *)
let rec dtype_of = function
  | Leaf l -> l.dt
  | Isnan _ | Cmp _ -> Bool
  | Cast (dt, _) -> dt
  | Un (_, a)
  | Reduce (_, _, _, a)
  | Reshape (_, a)
  | Contiguous a
  | Transpose (_, a)
  | Slice (_, a)
  | Broadcast (_, a)
  | Pad (_, _, a)
  | Take (_, _, a)
  | Bin (_, a, _)
  | Where (_, a, _)
  | Matmul (a, _) ->
      dtype_of a

let pp_program ppf e =
  let names = ref [] in
  let args = ref 0 and caps = ref 0 in
  List.iter
    (fun l ->
      let name =
        if l.capture then (
          incr caps;
          Printf.sprintf "c%d" (!caps - 1))
        else (
          incr args;
          Printf.sprintf "a%d" (!args - 1))
      in
      names := (l, name) :: !names;
      Format.fprintf ppf "@[<hov 2>let %s =@ Nx.create %s %a@ [|%a|] in@]@,"
        name (dtype_name l.dt) pp_ints l.shape
        (Format.pp_print_list
           ~pp_sep:(fun ppf () -> Format.fprintf ppf ";@ ")
           (pp_value l.dt))
        (Array.to_list l.values))
    (leaves e);
  let name l = List.assq l !names in
  let rec pp ppf e =
    match e with
    | Leaf l -> Format.pp_print_string ppf (name l)
    | Un (op, a) ->
        Format.fprintf ppf "@[<hov 2>(Nx.%s@ %a)@]"
          (un_name (dtype_of a) op)
          pp a
    | Isnan a -> Format.fprintf ppf "@[<hov 2>(Nx.isnan@ %a)@]" pp a
    | Bin (op, a, b) ->
        Format.fprintf ppf "@[<hov 2>(Nx.%s@ %a@ %a)@]"
          (bin_name (dtype_of a) op)
          pp a pp b
    | Cmp (op, a, b) ->
        Format.fprintf ppf "@[<hov 2>(Nx.%s@ %a@ %a)@]"
          (match op with Less -> "less" | Equal -> "equal")
          pp a pp b
    | Where (c, a, b) ->
        Format.fprintf ppf "@[<hov 2>(Nx.where@ %a@ %a@ %a)@]" pp c pp a pp b
    | Cast (dt, a) ->
        Format.fprintf ppf "@[<hov 2>(Nx.cast %s@ %a)@]" (dtype_name dt) pp a
    | Reduce (op, axes, keepdims, a) ->
        Format.fprintf ppf "@[<hov 2>(Nx.%s ~axes:[%s] ~keepdims:%b@ %a)@]"
          (red_name op)
          (String.concat "; " (List.map string_of_int axes))
          keepdims pp a
    | Reshape (s, a) ->
        Format.fprintf ppf "@[<hov 2>(Nx.reshape %a@ %a)@]" pp_ints s pp a
    | Contiguous a -> Format.fprintf ppf "@[<hov 2>(Nx.contiguous@ %a)@]" pp a
    | Transpose (axes, a) ->
        Format.fprintf ppf "@[<hov 2>(Nx.transpose ~axes:[%s]@ %a)@]"
          (String.concat "; " (List.map string_of_int axes))
          pp a
    | Slice (specs, a) ->
        let spec = function
          | Range (start, stop, step) ->
              Printf.sprintf "Rs (%d, %d, %d)" start stop step
          | Index i -> Printf.sprintf "I (%d)" i
        in
        Format.fprintf ppf "@[<hov 2>(Nx.slice [%s]@ %a)@]"
          (String.concat "; " (List.map spec specs))
          pp a
    | Broadcast (s, a) ->
        Format.fprintf ppf "@[<hov 2>(Nx.broadcast_to %a@ %a)@]" pp_ints s pp a
    | Pad (widths, v, a) ->
        Format.fprintf ppf "@[<hov 2>(Nx.pad [|%s|] %a@ %a)@]"
          (String.concat "; "
             (Array.to_list
                (Array.map (fun (b, a) -> Printf.sprintf "(%d, %d)" b a) widths)))
          (pp_value (dtype_of a))
          v pp a
    | Take (axis, i, a) ->
        Format.fprintf ppf "@[<hov 2>(Nx.take%s ~indices:%a@ %a)@]"
          (match axis with
          | Some ax -> Printf.sprintf " ~axis:%d" ax
          | None -> "")
          pp i pp a
    | Matmul (a, b) ->
        Format.fprintf ppf "@[<hov 2>(Nx.matmul@ %a@ %a)@]" pp a pp b
  in
  Format.fprintf ppf "@[<v>%a@]" pp e

(* Evaluation *)

type poly1 = { f1 : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t }
type poly2 = { f2 : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t }

type to_bool = {
  fb : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> (bool, Nx.bool_elt) Nx.t;
}

let create (l : leaf) : Nx.packed =
  match l.dt with
  | F32 -> P (Nx.create Nx.float32 l.shape l.values)
  | F16 -> P (Nx.create Nx.float16 l.shape l.values)
  | I32 -> P (Nx.create Nx.int32 l.shape (Array.map Int32.of_float l.values))
  | I64 -> P (Nx.create Nx.int64 l.shape (Array.map Int64.of_float l.values))
  | Bool ->
      P (Nx.create Nx.bool l.shape (Array.map (fun x -> x <> 0.) l.values))

let map1 dt g (p : Nx.packed) : Nx.packed =
  match dt with
  | F32 -> P (g.f1 (Nx.unpack Nx.float32 p))
  | F16 -> P (g.f1 (Nx.unpack Nx.float16 p))
  | I32 -> P (g.f1 (Nx.unpack Nx.int32 p))
  | I64 -> P (g.f1 (Nx.unpack Nx.int64 p))
  | Bool -> P (g.f1 (Nx.unpack Nx.bool p))

let map2 dt g (a : Nx.packed) (b : Nx.packed) : Nx.packed =
  match dt with
  | F32 -> P (g.f2 (Nx.unpack Nx.float32 a) (Nx.unpack Nx.float32 b))
  | F16 -> P (g.f2 (Nx.unpack Nx.float16 a) (Nx.unpack Nx.float16 b))
  | I32 -> P (g.f2 (Nx.unpack Nx.int32 a) (Nx.unpack Nx.int32 b))
  | I64 -> P (g.f2 (Nx.unpack Nx.int64 a) (Nx.unpack Nx.int64 b))
  | Bool -> P (g.f2 (Nx.unpack Nx.bool a) (Nx.unpack Nx.bool b))

let compare2 dt g (a : Nx.packed) (b : Nx.packed) : Nx.packed =
  match dt with
  | F32 -> P (g.fb (Nx.unpack Nx.float32 a) (Nx.unpack Nx.float32 b))
  | F16 -> P (g.fb (Nx.unpack Nx.float16 a) (Nx.unpack Nx.float16 b))
  | I32 -> P (g.fb (Nx.unpack Nx.int32 a) (Nx.unpack Nx.int32 b))
  | I64 -> P (g.fb (Nx.unpack Nx.int64 a) (Nx.unpack Nx.int64 b))
  | Bool -> P (g.fb (Nx.unpack Nx.bool a) (Nx.unpack Nx.bool b))

let cast dt (P x : Nx.packed) : Nx.packed =
  match dt with
  | F32 -> P (Nx.cast Nx.float32 x)
  | F16 -> P (Nx.cast Nx.float16 x)
  | I32 -> P (Nx.cast Nx.int32 x)
  | I64 -> P (Nx.cast Nx.int64 x)
  | Bool -> P (Nx.cast Nx.bool x)

let pad widths v dt (p : Nx.packed) : Nx.packed =
  match dt with
  | F32 -> P (Nx.pad widths v (Nx.unpack Nx.float32 p))
  | F16 -> P (Nx.pad widths v (Nx.unpack Nx.float16 p))
  | I32 -> P (Nx.pad widths (Int32.of_float v) (Nx.unpack Nx.int32 p))
  | I64 -> P (Nx.pad widths (Int64.of_float v) (Nx.unpack Nx.int64 p))
  | Bool -> P (Nx.pad widths (v <> 0.) (Nx.unpack Nx.bool p))

let un_op dt op =
  match op with
  | Neg -> { f1 = Nx.neg }
  | Abs -> { f1 = Nx.abs }
  | Sign -> { f1 = Nx.sign }
  | Square -> { f1 = Nx.square }
  | Sqrt -> { f1 = Nx.sqrt }
  | Recip -> { f1 = Nx.recip }
  | Floor -> { f1 = Nx.floor }
  | Ceil -> { f1 = Nx.ceil }
  | Round -> { f1 = Nx.round }
  | Trunc -> { f1 = Nx.trunc }
  | Not ->
      if dt = Bool then { f1 = Nx.logical_not } else { f1 = Nx.bitwise_not }

let bin_op dt op =
  match op with
  | Add -> { f2 = Nx.add }
  | Sub -> { f2 = Nx.sub }
  | Mul -> { f2 = Nx.mul }
  | Div -> { f2 = Nx.div }
  | Maximum -> { f2 = Nx.maximum }
  | Minimum -> { f2 = Nx.minimum }
  | And ->
      if dt = Bool then { f2 = Nx.logical_and } else { f2 = Nx.bitwise_and }
  | Or -> if dt = Bool then { f2 = Nx.logical_or } else { f2 = Nx.bitwise_or }
  | Xor ->
      if dt = Bool then { f2 = Nx.logical_xor } else { f2 = Nx.bitwise_xor }

let red_op axes keepdims = function
  | Sum -> { f1 = (fun x -> Nx.sum ~axes ~keepdims x) }
  | Prod -> { f1 = (fun x -> Nx.prod ~axes ~keepdims x) }
  | Max -> { f1 = (fun x -> Nx.max ~axes ~keepdims x) }
  | Min -> { f1 = (fun x -> Nx.min ~axes ~keepdims x) }

(* [eval value e] is [e] computed with nx, each leaf [l] being [value l]. *)
let rec eval value e : Nx.packed =
  match e with
  | Leaf l -> value l
  | Un (op, a) ->
      let dt = dtype_of a in
      map1 dt (un_op dt op) (eval value a)
  | Isnan a ->
      let (P x) = eval value a in
      P (Nx.isnan x)
  | Bin (op, a, b) ->
      let dt = dtype_of a in
      let x = eval value a in
      let y = eval value b in
      map2 dt (bin_op dt op) x y
  | Cmp (op, a, b) ->
      let x = eval value a in
      let y = eval value b in
      let g =
        match op with Less -> { fb = Nx.less } | Equal -> { fb = Nx.equal }
      in
      compare2 (dtype_of a) g x y
  | Where (c, a, b) ->
      let c' = Nx.unpack Nx.bool (eval value c) in
      let x = eval value a in
      let y = eval value b in
      map2 (dtype_of a) { f2 = (fun x y -> Nx.where c' x y) } x y
  | Cast (dt, a) -> cast dt (eval value a)
  | Reduce (op, axes, keepdims, a) ->
      map1 (dtype_of a) (red_op axes keepdims op) (eval value a)
  | Reshape (s, a) ->
      map1 (dtype_of a) { f1 = (fun x -> Nx.reshape s x) } (eval value a)
  | Contiguous a -> map1 (dtype_of a) { f1 = Nx.contiguous } (eval value a)
  | Transpose (axes, a) ->
      map1 (dtype_of a) { f1 = (fun x -> Nx.transpose ~axes x) } (eval value a)
  | Slice (specs, a) ->
      let specs =
        List.map
          (function
            | Range (start, stop, step) -> Nx.Rs (start, stop, step)
            | Index i -> Nx.I i)
          specs
      in
      map1 (dtype_of a) { f1 = (fun x -> Nx.slice specs x) } (eval value a)
  | Broadcast (s, a) ->
      map1 (dtype_of a) { f1 = (fun x -> Nx.broadcast_to s x) } (eval value a)
  | Pad (widths, v, a) -> pad widths v (dtype_of a) (eval value a)
  | Take (axis, i, a) ->
      let indices = Nx.unpack Nx.int64 (eval value i) in
      map1 (dtype_of a)
        { f1 = (fun x -> Nx.take ?axis ~indices x) }
        (eval value a)
  | Matmul (a, b) ->
      let x = eval value a in
      let y = eval value b in
      map2 (dtype_of a) { f2 = Nx.matmul } x y

(* A list of tensors of any dtypes, as a compiled function's argument and
   result. *)
module Packs = struct
  type _ t = Nx.packed list

  let walk c l =
    Nx.Ptree.Walk.list
      (fun c (Nx.P x : Nx.packed) : Nx.packed -> P (Nx.Ptree.Walk.tensor c x))
      c l
end

let packs = Nx.Ptree.instantiate (module Packs)

(* [eager e] and [compiled e] are [e]'s value. [compiled] passes [e]'s argument
   leaves to one compiled call and closes over its captured ones. *)
let eager e = eval create e

let compiled e =
  let all = leaves e in
  let captured =
    List.map (fun l -> (l, create l)) (List.filter (fun l -> l.capture) all)
  in
  let args = List.filter (fun l -> not l.capture) all in
  let f values =
    let bound = List.combine args values in
    let value l =
      if l.capture then List.assq l captured else List.assq l bound
    in
    [ eval value e ]
  in
  match
    Rune.jit Nx.Ptree.(packs @-> returns packs) f (List.map create args)
  with
  | [ y ] -> y
  | _ -> assert false

(* Values *)

type value = { dtype : dt; dims : int array; elements : float array }

let value dt (P x : Nx.packed) =
  let elements : float array =
    match dt with
    | F32 -> Nx.to_array (Nx.unpack Nx.float32 (P x))
    | F16 -> Nx.to_array (Nx.unpack Nx.float16 (P x))
    | I32 -> Array.map Int32.to_float (Nx.to_array (Nx.unpack Nx.int32 (P x)))
    | I64 -> Array.map Int64.to_float (Nx.to_array (Nx.unpack Nx.int64 (P x)))
    | Bool ->
        Array.map
          (fun b -> if b then 1. else 0.)
          (Nx.to_array (Nx.unpack Nx.bool (P x)))
  in
  { dtype = dt; dims = Nx.shape x; elements }

let pp_result ppf v =
  Format.fprintf ppf "@[<hov 2>%s %a@ [|%a|]@]" (dtype_name v.dtype) pp_ints
    v.dims
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.fprintf ppf ";@ ")
       (pp_value v.dtype))
    (Array.to_list v.elements)

(* Bit for bit, every NaN equal to every NaN, and with [~zeros] [-0.] equal to
   [0.]. *)
let same ~zeros a b =
  let element x y =
    (Float.is_nan x && Float.is_nan y)
    || (zeros && x = 0. && y = 0.)
    || Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y)
  in
  a.dtype = b.dtype && a.dims = b.dims
  && Array.length a.elements = Array.length b.elements
  && Array.for_all2 element a.elements b.elements

(* What a run gives: a value, or the message of the [Invalid_argument] it
   raised. *)
type outcome = Value of value | Refused of string

let outcome dt f =
  match f () with
  | y -> Value (value dt y)
  | exception Invalid_argument m -> Refused m

let pp_outcome ppf = function
  | Value v -> pp_result ppf v
  | Refused m -> Format.fprintf ppf "Invalid_argument %S" m

let outcomes ~zeros =
  Testable.make ~pp:pp_outcome ~equal:(fun a b ->
      match (a, b) with
      | Value a, Value b -> same ~zeros a b
      | Refused a, Refused b -> String.equal a b
      | _ -> false)

(* Generators *)

let range a b = List.init (b - a + 1) (fun i -> a + i)

(* A float16's value from its bits. *)
let of_half bits =
  let sign = if bits land 0x8000 <> 0 then -1. else 1. in
  let e = (bits lsr 10) land 0x1f and m = bits land 0x3ff in
  if e = 0x1f then if m = 0 then sign *. Float.infinity else Float.nan
  else if e = 0 then sign *. Float.ldexp (Float.of_int m) (-24)
  else sign *. Float.ldexp (Float.of_int (m lor 0x400)) (e - 25)

(* Floats of the dtype's own range: any double, which a float32 or float16
   rounds, any bit pattern of the dtype, and its corners. *)
let element = function
  | F32 ->
      Gen.frequency
        [
          (3, Gen.any_float);
          (3, Gen.map Int32.float_of_bits Gen.int32);
          ( 1,
            Gen.of_list
              [
                0x1p-149; -0x1p-149; 0x1p-126; 0x1.fffffcp-127; 0x1.fffffep+127;
              ] );
        ]
  | F16 ->
      Gen.frequency
        [
          (3, Gen.any_float);
          (3, Gen.map of_half (Gen.int_range 0 0xffff));
          (1, Gen.of_list [ 0x1p-24; -0x1p-24; 0x1p-14; 0x1.ff8p-15; 65504. ]);
        ]
  | I32 ->
      Gen.frequency
        [
          (6, Gen.map Float.of_int (Gen.int_range (-9) 9));
          ( 1,
            Gen.map Int32.to_float
              (Gen.of_list
                 [
                   Int32.min_int;
                   Int32.succ Int32.min_int;
                   Int32.max_int;
                   Int32.pred Int32.max_int;
                 ]) );
        ]
  | Bool -> Gen.map (fun b -> if b then 1. else 0.) Gen.bool
  | I64 -> Gen.map Float.of_int (Gen.int_range (-2) 4)

let leaf ~capture dt shape =
  let open Gen in
  let* values = array ~size:(constant (numel shape)) (element dt) in
  let+ capture =
    if capture then frequency [ (7, constant false); (3, constant true) ]
    else constant false
  in
  Leaf { dt; shape; values; capture }

let dim =
  Gen.frequency
    [ (1, Gen.constant 0); (3, Gen.constant 1); (4, Gen.int_range 2 3) ]

let shape = Gen.array ~size:(Gen.int_range 0 3) dim

(* A shape that broadcasts to [s]: some leading axes dropped, some axes of [s]
   made [1]. *)
let broadcasting s =
  let open Gen in
  let r = Array.length s in
  let* drop = int_range 0 r in
  let kept = Array.sub s drop (r - drop) in
  let+ ones =
    array
      ~size:(constant (Array.length kept))
      (frequency [ (3, constant false); (1, constant true) ])
  in
  Array.mapi (fun i d -> if ones.(i) then 1 else d) kept

(* Two shapes that broadcast together to [s], the first or the second [s]
   itself. *)
let operands s =
  let open Gen in
  let* other = broadcasting s in
  let+ swap = bool in
  if swap then (other, s) else (s, other)

let rec factors n p =
  if n <= 1 then []
  else if n mod p = 0 then p :: factors (n / p) p
  else factors n (p + 1)

(* A shape of [n] elements. *)
let of_numel n =
  let open Gen in
  if n = 0 then
    let* r = int_range 1 3 in
    let* dims = array ~size:(constant r) (int_range 0 3) in
    let+ z = int_range 0 (r - 1) in
    Array.mapi (fun i d -> if i = z then 0 else d) dims
  else
    let fs = factors n 2 in
    let* r = int_range (if fs = [] then 0 else 1) 3 in
    let+ slots =
      list
        ~size:(constant (List.length fs))
        (int_range 0 (Stdlib.max 0 (r - 1)))
    in
    let dims = Array.make r 1 in
    List.iter2 (fun f s -> dims.(s) <- dims.(s) * f) fs slots;
    dims

(* A slice of a source shape giving [s], as the source's dimensions and the
   slice's specs: each axis of [s] a strided range, which never ends at a
   negative position, and some indexed axes inserted. *)
let sliced s =
  let open Gen in
  let range d =
    let* step = of_list [ 1; 2; -1; -2 ] in
    let* first = int_range 0 2 in
    let+ extra = int_range 0 2 in
    if d = 0 then (first + extra, Range (first, first, step))
    else if step > 0 then
      let stop = first + ((d - 1) * step) + 1 in
      (stop + extra, Range (first, stop, step))
    else
      (* [first] is the last position read, [start] the first. *)
      let start = first + 1 + ((d - 1) * -step) in
      (start + 1 + extra, Range (start, first, step))
  in
  let* ranges =
    Array.fold_right
      (fun d acc ->
        let* r = range d in
        let+ rest = acc in
        r :: rest)
      s (constant [])
  in
  let* indexed = int_range 0 2 in
  if indexed = 0 then constant ranges
  else
    let* at = int_range 0 (List.length ranges) in
    let* m = int_range 1 3 in
    let+ i = int_range (-m) (m - 1) in
    List.filteri (fun j _ -> j < at) ranges
    @ [ (m, Index i) ]
    @ List.filteri (fun j _ -> j >= at) ranges

(* Programs of a dtype and shape. [no_product] keeps a product off an add's
   operand, and [no_recip] a reciprocal off a product's, through the operations
   a compiler would fuse or rewrite through; [no_minmax] keeps maxima and minima
   out, under an operation that reads a zero's sign; [args_only] makes every
   leaf an argument, under a divisor. *)
type ctx = {
  depth : int;
  no_product : bool;
  no_recip : bool;
  no_minmax : bool;
  args_only : bool;
}

let floats = [ F32; F16 ]
let dtypes = [ F32; F16; I32; Bool ]

let rec program ctx dt s : expr Gen.t =
  let open Gen in
  let here = leaf ~capture:(not ctx.args_only) dt s in
  if ctx.depth = 0 then here
  else
    let sub ?(no_product = false) ?(no_recip = false)
        ?(no_minmax = ctx.no_minmax) ?(args_only = ctx.args_only) dt s =
      program
        { depth = ctx.depth - 1; no_product; no_recip; no_minmax; args_only }
        dt s
    in
    (* An operation a fused multiply-add, or a product by a reciprocal turned
       into a division, would reach through. *)
    let through dt s =
      sub ~no_product:ctx.no_product ~no_recip:ctx.no_recip dt s
    in
    let product_ok = not (ctx.no_product && is_float dt) in
    let unary =
      let ops =
        match dt with
        | F32 | F16 ->
            [ Neg; Abs; Sign; Sqrt; Floor; Ceil; Round; Trunc ]
            @ (if ctx.no_recip then [] else [ Recip ])
            @ if product_ok then [ Square ] else []
        | I32 -> [ Neg; Abs; Sign; Not ] @ if product_ok then [ Square ] else []
        | Bool -> [ Not ]
        | I64 -> []
      in
      let* op = of_list ops in
      match op with
      | Neg -> map (fun a -> Un (Neg, a)) (through dt s)
      | Recip | Sign -> map (fun a -> Un (op, a)) (sub ~no_minmax:true dt s)
      | _ -> map (fun a -> Un (op, a)) (sub dt s)
    in
    let arithmetic =
      let ops =
        match dt with
        | F32 | F16 ->
            [ Add; Sub; Div ]
            @ (if product_ok then [ Mul ] else [])
            @ if ctx.no_minmax then [] else [ Maximum; Minimum ]
        | I32 ->
            [ Add; Sub; Div; And; Or; Xor ]
            @ (if product_ok then [ Mul ] else [])
            @ if ctx.no_minmax then [] else [ Maximum; Minimum ]
        | Bool -> [ And; Or; Xor ]
        | I64 -> []
      in
      let* op = of_list ops in
      let* sa, sb = operands s in
      match op with
      | Add | Sub ->
          let* a = sub ~no_product:true dt sa in
          let+ b = sub ~no_product:true dt sb in
          Bin (op, a, b)
      | Div ->
          let* a = sub dt sa in
          let+ b = sub ~no_minmax:true ~args_only:true dt sb in
          Bin (op, a, b)
      | Mul ->
          let* a = sub ~no_recip:true dt sa in
          let+ b = sub ~no_recip:true dt sb in
          Bin (op, a, b)
      | _ ->
          let* a = sub dt sa in
          let+ b = sub dt sb in
          Bin (op, a, b)
    in
    let comparison =
      let* op = of_list [ Less; Equal ] in
      let* operand = of_list dtypes in
      let* sa, sb = operands s in
      let* a = sub operand sa in
      let+ b = sub operand sb in
      Cmp (op, a, b)
    in
    let isnan =
      let* operand = of_list floats in
      map (fun a -> Isnan a) (sub operand s)
    in
    let where =
      let* sc, sa = operands s in
      let* sb = broadcasting s in
      let* c = sub Bool sc in
      let* a = through dt sa in
      let+ b = through dt sb in
      Where (c, a, b)
    in
    let cast =
      let* source = of_list (List.filter (fun d -> d <> dt) dtypes) in
      map (fun a -> Cast (dt, a)) (sub source s)
    in
    let reduction =
      let ops =
        match dt with
        | F32 | F16 -> if ctx.no_minmax then [] else [ Max; Min ]
        | I32 -> [ Sum; Prod ] @ if ctx.no_minmax then [] else [ Max; Min ]
        | Bool | I64 -> []
      in
      let* op = of_list ops in
      let lowest = match op with Max | Min -> 1 | Sum | Prod -> 0 in
      let ones =
        List.filter (fun i -> s.(i) = 1) (range 0 (Array.length s - 1))
      in
      let* keepdims = if ones = [] then constant false else bool in
      if keepdims then (
        let* axes = subsequence ones in
        let axes = if axes = [] then [ List.hd ones ] else axes in
        let* sizes =
          list ~size:(constant (List.length axes)) (int_range lowest 3)
        in
        let source = Array.copy s in
        List.iter2 (fun a d -> source.(a) <- d) axes sizes;
        map (fun a -> Reduce (op, axes, true, a)) (sub dt source))
      else
        let* k = int_range 1 (Stdlib.min 2 (4 - Array.length s)) in
        let* sizes = list ~size:(constant k) (int_range lowest 3) in
        let* positions =
          list ~size:(constant k) (int_range 0 (Array.length s))
        in
        (* Insert each reduced axis before the kept axis at its position. *)
        let kept = Array.to_list (Array.mapi (fun i d -> (i, d)) s) in
        let inserted = List.combine positions sizes in
        let source, axes, _ =
          List.fold_left
            (fun (acc, axes, n) (i, d) ->
              let here = List.filter (fun (p, _) -> p = i) inserted in
              let acc = acc @ List.map snd here @ [ d ] in
              let axes = axes @ List.mapi (fun j _ -> n + j) here in
              (acc, axes, n + List.length here + 1))
            ([], [], 0) kept
        in
        let tail = List.filter (fun (p, _) -> p = Array.length s) inserted in
        let n = List.length source in
        let source = Array.of_list (source @ List.map snd tail) in
        let axes = axes @ List.mapi (fun j _ -> n + j) tail in
        map (fun a -> Reduce (op, axes, false, a)) (sub dt source)
    in
    let reshape =
      let* source = of_numel (numel s) in
      map (fun a -> Reshape (s, a)) (through dt source)
    in
    let contiguous = map (fun a -> Contiguous a) (through dt s) in
    let transpose =
      let* axes = permutation (range 0 (Array.length s - 1)) in
      let source = Array.make (Array.length s) 0 in
      List.iteri (fun i a -> source.(a) <- s.(i)) axes;
      map (fun a -> Transpose (axes, a)) (through dt source)
    in
    let slice =
      let* specs = sliced s in
      let source = Array.of_list (List.map fst specs) in
      map (fun a -> Slice (List.map snd specs, a)) (through dt source)
    in
    let broadcast =
      let* source = broadcasting s in
      map (fun a -> Broadcast (s, a)) (through dt source)
    in
    let pad =
      let* widths =
        Array.fold_right
          (fun d acc ->
            let* before = int_range 0 (Stdlib.min 2 d) in
            let* after = int_range 0 (Stdlib.min 2 (d - before)) in
            let+ rest = acc in
            (before, after) :: rest)
          s (constant [])
      in
      let widths = Array.of_list widths in
      let source =
        Array.mapi (fun i d -> d - fst widths.(i) - snd widths.(i)) s
      in
      let* v = element dt in
      map (fun a -> Pad (widths, v, a)) (through dt source)
    in
    let take =
      let* axis =
        if Array.length s = 0 then constant None
        else option (int_range 0 (Array.length s - 1))
      in
      match axis with
      | None ->
          let* source = shape in
          let n = numel source in
          let indices = Gen.map Float.of_int (int_range (-2) (n + 1)) in
          let* values = array ~size:(constant (numel s)) indices in
          let* capture = bool in
          let i =
            Leaf
              {
                dt = I64;
                shape = s;
                values;
                capture = capture && not ctx.args_only;
              }
          in
          map (fun a -> Take (None, i, a)) (through dt source)
      | Some ax ->
          (* Indices of any shape, read in C order along the axis. *)
          let* d = int_range 0 3 in
          let source = Array.mapi (fun i x -> if i = ax then d else x) s in
          let indices = Gen.map Float.of_int (int_range (-2) (d + 1)) in
          let* shape = of_numel s.(ax) in
          let* values = array ~size:(constant s.(ax)) indices in
          let* capture = bool in
          let i =
            Leaf
              {
                dt = I64;
                shape;
                values;
                capture = capture && not ctx.args_only;
              }
          in
          map (fun a -> Take (Some ax, i, a)) (through dt source)
    in
    (* An integer product: the contraction of a float one is a sum. *)
    let matmul =
      let* k = int_range 0 3 in
      let* sa, sb =
        match s with
        | [||] -> constant ([| k |], [| k |])
        | [| n |] -> of_list [ ([| k |], [| k; n |]); ([| n; k |], [| k |]) ]
        | [| m; n |] -> constant ([| m; k |], [| k; n |])
        | [| b; m; n |] ->
            of_list
              [ ([| b; m; k |], [| b; k; n |]); ([| m; k |], [| b; k; n |]) ]
        | _ -> constant ([| k |], [| k |])
      in
      let* a = sub dt sa in
      let+ b = sub dt sb in
      Matmul (a, b)
    in
    let some cond w g = if cond then [ (w, g) ] else [] in
    frequency
      (List.concat
         [
           [ (3, here); (2, unary); (3, arithmetic); (2, cast) ];
           [
             (2, reshape);
             (1, contiguous);
             (2, transpose);
             (2, slice);
             (2, broadcast);
           ];
           [ (2, pad); (2, take) ];
           some (dt = Bool) 2 comparison;
           some (dt = Bool) 1 isnan;
           some (dt <> I64) 2 where;
           some
             (Array.length s <= 3
             &&
             match dt with
             | F32 | F16 -> not ctx.no_minmax
             | I32 -> true
             | Bool | I64 -> false)
             2 reduction;
           some (dt = I32 && Array.length s <= 3) 1 matmul;
         ])

let programs =
  let open Gen in
  let* dt = of_list dtypes in
  let* s = shape in
  with_pp pp_program
    (program
       {
         depth = 4;
         no_product = false;
         no_recip = false;
         no_minmax = false;
         args_only = false;
       }
       dt s)

(* The law *)

let family = function
  | Leaf _ -> "a leaf"
  | Un _ -> "a unary operation"
  | Isnan _ -> "isnan"
  | Bin ((Add | Sub | Mul | Div | Maximum | Minimum), _, _) -> "arithmetic"
  | Bin ((And | Or | Xor), _, _) -> "a logical or bitwise operation"
  | Cmp _ -> "a comparison"
  | Where _ -> "where"
  | Cast _ -> "a cast"
  | Reduce _ -> "a reduction"
  | Reshape _ -> "reshape"
  | Contiguous _ -> "contiguous"
  | Transpose _ -> "transpose"
  | Slice _ -> "slice"
  | Broadcast _ -> "broadcast_to"
  | Pad _ -> "pad"
  | Take _ -> "take"
  | Matmul _ -> "matmul"

let families =
  [
    "a unary operation";
    "isnan";
    "arithmetic";
    "a logical or bitwise operation";
    "a comparison";
    "where";
    "a cast";
    "a reduction";
    "reshape";
    "contiguous";
    "transpose";
    "slice";
    "broadcast_to";
    "pad";
    "take";
    "matmul";
  ]

let dims e =
  match eager e with
  | P x -> Some (Nx.shape x)
  | exception Invalid_argument _ -> None

let has_leaf p e = List.exists p (leaves e)

(* Whether a float leaf of [e] holds, at its dtype, a value satisfying [p]. *)
let has_float p e =
  List.exists
    (fun l -> is_float l.dt && Array.exists p (value l.dt (create l)).elements)
    (leaves e)

let subnormal dt x =
  x <> 0. && Float.abs x < match dt with F16 -> 0x1p-14 | _ -> 0x1p-126

let label e =
  List.iter (fun f -> cover f (exists (fun n -> family n = f) e)) families;
  List.iter
    (fun (name, dt) -> cover name (has_leaf (fun l -> l.dt = dt) e))
    [
      ("a float32 leaf", F32);
      ("a float16 leaf", F16);
      ("an int32 leaf", I32);
      ("a bool leaf", Bool);
    ];
  cover "a zero-size leaf" (has_leaf (fun l -> numel l.shape = 0) e);
  cover "a one-element leaf" (has_leaf (fun l -> numel l.shape = 1) e);
  cover "a scalar leaf" (has_leaf (fun l -> l.shape = [||]) e);
  cover "a captured leaf" (has_leaf (fun l -> l.capture) e);
  cover "a broadcast operand"
    (exists
       (function
         | Broadcast (s, a) -> dims a <> Some s
         | Bin (_, a, b) | Cmp (_, a, b) -> dims a <> dims b
         | Where (c, a, b) -> dims c <> dims a || dims a <> dims b
         | _ -> false)
       e);
  cover "a NaN" (has_float Float.is_nan e);
  cover "an infinity" (has_float (fun x -> Float.abs x = Float.infinity) e);
  cover "a negative zero" (has_float (fun x -> x = 0. && Float.sign_bit x) e);
  cover "a subnormal"
    (List.exists
       (fun l ->
         is_float l.dt
         && Array.exists (subnormal l.dt) (value l.dt (create l)).elements)
       (leaves e))

let compiled_is_eager e =
  label e;
  let dt = dtype_of e in
  let zeros = signs_zeros e in
  let expected = outcome dt (fun () -> eager e) in
  classify "eager refuses the program"
    (match expected with Refused _ -> true | Value _ -> false);
  classify "an empty result"
    (match expected with Value v -> v.elements = [||] | Refused _ -> false);
  classify "a float result" (is_float dt);
  equal (outcomes ~zeros) expected (outcome dt (fun () -> compiled e))

(* Rounded programs

   A program whose last operation the jit may round otherwise: a float sum or
   mean over axes or a float product, whose terms it may sum in another
   association, or a transcendental function, an approximation within a few
   units in the last place. Its operands are exact programs, so both runs see
   the same terms. *)

type fn = Exp | Log | Sin | Cos | Tanh | Exp2 | Log2

type rounded =
  | Sum_f of int list * bool * expr
  | Mean_f of int list * bool * expr
  | Prod_f of int list * bool * expr
  | Matmul_f of expr * expr
  | Fn of fn * expr

let fn_name = function
  | Exp -> "exp"
  | Log -> "log"
  | Sin -> "sin"
  | Cos -> "cos"
  | Tanh -> "tanh"
  | Exp2 -> "exp2"
  | Log2 -> "log2"

let rounded_leaves = function
  | Sum_f (_, _, a) | Mean_f (_, _, a) | Prod_f (_, _, a) | Fn (_, a) ->
      leaves a
  | Matmul_f (a, b) -> leaves a @ leaves b

(* The rounded program as an expression: its root is applied to the exact
   operands' values. *)
let root r : Nx.packed list -> Nx.packed =
 fun operands ->
  let floats f p =
    match p with
    | Nx.P x as p -> (
        match Nx.dtype x with
        | Nx.Float32 -> Nx.P (f.f1 (Nx.unpack Nx.float32 p))
        | Nx.Float16 -> Nx.P (f.f1 (Nx.unpack Nx.float16 p))
        | _ -> invalid_arg "a rounded program's operand is a float")
  in
  match (r, operands) with
  | Sum_f (axes, keepdims, _), [ a ] ->
      floats { f1 = (fun x -> Nx.sum ~axes ~keepdims x) } a
  | Prod_f (axes, keepdims, _), [ a ] ->
      floats { f1 = (fun x -> Nx.prod ~axes ~keepdims x) } a
  | Mean_f (axes, keepdims, _), [ a ] ->
      floats { f1 = (fun x -> Nx.mean ~axes ~keepdims x) } a
  | Fn (fn, _), [ a ] ->
      let f =
        match fn with
        | Exp -> { f1 = Nx.exp }
        | Log -> { f1 = Nx.log }
        | Sin -> { f1 = Nx.sin }
        | Cos -> { f1 = Nx.cos }
        | Tanh -> { f1 = Nx.tanh }
        | Exp2 -> { f1 = Nx.exp2 }
        | Log2 -> { f1 = Nx.log2 }
      in
      floats f a
  | Matmul_f _, [ (Nx.P x as a); b ] -> (
      match Nx.dtype x with
      | Nx.Float32 ->
          Nx.P (Nx.matmul (Nx.unpack Nx.float32 a) (Nx.unpack Nx.float32 b))
      | Nx.Float16 ->
          Nx.P (Nx.matmul (Nx.unpack Nx.float16 a) (Nx.unpack Nx.float16 b))
      | _ -> invalid_arg "a rounded program's operand is a float")
  | _ -> invalid_arg "a rounded program's operands"

let operands = function
  | Sum_f (_, _, a) | Mean_f (_, _, a) | Prod_f (_, _, a) | Fn (_, a) -> [ a ]
  | Matmul_f (a, b) -> [ a; b ]

let eval_rounded value r = root r (List.map (eval value) (operands r))

let pp_rounded ppf r =
  let e =
    (* Printed through an expression of the same leaves. *)
    match r with
    | Sum_f (_, _, a) | Mean_f (_, _, a) | Prod_f (_, _, a) | Fn (_, a) -> a
    | Matmul_f (a, b) -> Bin (Mul, a, b)
  in
  pp_program ppf e;
  Format.fprintf ppf "@,(* then %s *)"
    (match r with
    | Sum_f (axes, keepdims, _) ->
        Printf.sprintf "Nx.sum ~axes:[%s] ~keepdims:%b"
          (String.concat "; " (List.map string_of_int axes))
          keepdims
    | Mean_f (axes, keepdims, _) ->
        Printf.sprintf "Nx.mean ~axes:[%s] ~keepdims:%b"
          (String.concat "; " (List.map string_of_int axes))
          keepdims
    | Prod_f (axes, keepdims, _) ->
        Printf.sprintf "Nx.prod ~axes:[%s] ~keepdims:%b"
          (String.concat "; " (List.map string_of_int axes))
          keepdims
    | Matmul_f _ -> "Nx.matmul of the two operands, in place of Nx.mul"
    | Fn (fn, _) -> "Nx." ^ fn_name fn)

let rounded_programs =
  let open Gen in
  let ctx =
    {
      depth = 3;
      no_product = false;
      no_recip = false;
      no_minmax = false;
      args_only = false;
    }
  in
  (* The factors of a float product are no reciprocals. *)
  let factors = { ctx with no_recip = true } in
  let reduction ?(ctx = ctx) root =
    let* dt = of_list floats in
    let* s = shape in
    let* axes = subsequence (range 0 (Array.length s - 1)) in
    let* keepdims = bool in
    let+ a = program ctx dt s in
    root (axes, keepdims, a)
  in
  let matmul =
    let* dt = of_list floats in
    let* m = int_range 1 3 and+ k = int_range 0 4 and+ n = int_range 1 3 in
    let* sa, sb =
      of_list
        [
          ([| m; k |], [| k; n |]);
          ([| k |], [| k; n |]);
          ([| m; k |], [| k |]);
          ([| k |], [| k |]);
          ([| 2; m; k |], [| k; n |]);
        ]
    in
    let* a = program factors dt sa in
    let+ b = program factors dt sb in
    Matmul_f (a, b)
  in
  let fn =
    let* dt = of_list floats in
    let* f = of_list [ Exp; Log; Sin; Cos; Tanh; Exp2; Log2 ] in
    let* s = shape in
    let+ a = program ctx dt s in
    Fn (f, a)
  in
  with_pp pp_rounded
    (frequency
       [
         (2, reduction (fun (x, k, a) -> Sum_f (x, k, a)));
         (1, reduction (fun (x, k, a) -> Mean_f (x, k, a)));
         (1, reduction ~ctx:factors (fun (x, k, a) -> Prod_f (x, k, a)));
         (2, matmul);
         (3, fn);
       ])

(* Units of rounding *)

(* The units in the last place of the result's dtype a compiled transcendental
   function is within. *)
let transcendental_ulps = 4
let unit_roundoff = function F16 -> 0x1p-11 | _ -> 0x1p-24
let largest = function F16 -> 65504. | _ -> 0x1.fffffep+127
let least = function F16 -> 0x1p-24 | _ -> 0x1p-149

(* The spacing of [dt]'s floats at [x]. *)
let ulp dt x =
  let x = Float.abs x in
  if x = 0. then least dt
  else
    let e = snd (Float.frexp x) in
    let p = match dt with F16 -> 11 | _ -> 24 in
    Float.max (least dt) (Float.ldexp 1. (e - p))

let float64 (Nx.P x) = Nx.cast Nx.float64 x
let f64s t = Nx.to_array t

(* What the stated rounding allows of a sum of terms: [count] terms whose exact
   sum is [r] and whose magnitudes sum to [s], summed at float32 in any
   association and rounded once to [dt]. Past [dt]'s greatest float, [total],
   the sum's magnitude before a mean divides it, any result is allowed. *)
let summed ?total dt ~count ~r ~s c e =
  let total = Option.value total ~default:s in
  let close () =
    Float.abs (c -. e)
    <= (2. *. Float.of_int count *. 0x1p-24 *. s)
       +. (2. *. unit_roundoff dt *. Float.max (Float.abs c) (Float.abs e))
  in
  if Float.is_nan r then Float.is_nan c && Float.is_nan e
  else if total > largest dt then true
  else if Float.abs r = Float.infinity then c = r && e = r
  else if s = 0. then
    (* A sum of zeros is [0.], never [-0.]. *)
    c = 0. && e = 0. && (not (Float.sign_bit c)) && not (Float.sign_bit e)
  else close ()

(* What the stated rounding allows of a product of [count] factors whose exact
   product is [r]: in any association, each rounding at most a unit of [dt].
   Where some association's partial product can leave [dt]'s normal range,
   [2^big] above its greatest float or [2^small] below its least normal one, any
   result is allowed. With [~zeros], a factor's zero may have either sign. *)
let multiplied dt ~zeros ~count ~r ~big ~small c e =
  let least_normal = match dt with F16 -> 0x1p-14 | _ -> 0x1p-126 in
  if Float.is_nan r then Float.is_nan c && Float.is_nan e
  else if big >= Float.log2 (largest dt) || small < Float.log2 least_normal then
    true
  else if zeros && r = 0. then c = 0. && e = 0.
  else if Float.abs r = Float.infinity || r = 0. then
    Int64.equal (Int64.bits_of_float c) (Int64.bits_of_float r)
    && Int64.equal (Int64.bits_of_float e) (Int64.bits_of_float r)
  else
    Float.abs (c -. e)
    <= 2. *. Float.of_int (count + 1) *. unit_roundoff dt *. Float.abs r

let within_rounding r =
  cover "a sum" (match r with Sum_f _ -> true | _ -> false);
  cover "a mean" (match r with Mean_f _ -> true | _ -> false);
  cover "a float product over axes"
    (match r with Prod_f _ -> true | _ -> false);
  cover "a float product" (match r with Matmul_f _ -> true | _ -> false);
  List.iter
    (fun f ->
      cover ("Nx." ^ fn_name f) (match r with Fn (g, _) -> f = g | _ -> false))
    [ Exp; Log; Sin; Cos; Tanh; Exp2; Log2 ];
  cover "a float16 operand"
    (List.exists (fun l -> l.dt = F16) (rounded_leaves r));
  let dt = dtype_of (List.hd (operands r)) in
  let all = rounded_leaves r in
  let captured =
    List.map (fun l -> (l, create l)) (List.filter (fun l -> l.capture) all)
  in
  let args = List.filter (fun l -> not l.capture) all in
  let f values =
    let bound = List.combine args values in
    let value l =
      if l.capture then List.assq l captured else List.assq l bound
    in
    [ eval_rounded value r ]
  in
  let compiled () =
    match
      Rune.jit Nx.Ptree.(packs @-> returns packs) f (List.map create args)
    with
    | [ y ] -> y
    | _ -> assert false
  in
  match List.map (eval create) (operands r) with
  | exception Invalid_argument m ->
      equal (outcomes ~zeros:false) (Refused m) (outcome dt compiled)
  | ops -> (
      let e = value dt (root r ops) in
      let c = value dt (compiled ()) in
      equal ~msg:"shape" (array int) e.dims c.dims;
      let n = Array.length e.elements in
      let check i ok =
        if not ok then
          failf "element %d: eager %h, compiled %h, beyond the stated rounding"
            i e.elements.(i) c.elements.(i)
      in
      match (r, ops) with
      | (Sum_f (axes, keepdims, _) | Mean_f (axes, keepdims, _)), [ a ] ->
          let x = float64 a in
          let shape = Nx.shape x in
          let count = List.fold_left (fun n ax -> n * shape.(ax)) 1 axes in
          let rs = f64s (Nx.sum ~axes ~keepdims x) in
          let ss = f64s (Nx.sum ~axes ~keepdims (Nx.abs x)) in
          for i = 0 to n - 1 do
            let ce = c.elements.(i) and ee = e.elements.(i) in
            match r with
            | Mean_f _ ->
                let m = Float.of_int count in
                check i
                  (summed dt ~count ~total:ss.(i)
                     ~r:(rs.(i) /. m)
                     ~s:(ss.(i) /. m)
                     ce ee)
            | _ -> check i (summed dt ~count ~r:rs.(i) ~s:ss.(i) ce ee)
          done
      | Prod_f (axes, keepdims, _), [ a ] ->
          let x = float64 a in
          let shape = Nx.shape x in
          let count = List.fold_left (fun n ax -> n * shape.(ax)) 1 axes in
          let lx = Nx.log2 (Nx.abs x) in
          let part keep =
            f64s
              (Nx.sum ~axes ~keepdims
                 (Nx.where (keep lx) lx (Nx.zeros_like lx)))
          in
          let rs = f64s (Nx.prod ~axes ~keepdims x) in
          let big =
            part (fun l -> Nx.logical_and (Nx.isfinite l) (Nx.greater_s l 0.))
          in
          let small =
            part (fun l -> Nx.logical_and (Nx.isfinite l) (Nx.less_s l 0.))
          in
          for i = 0 to n - 1 do
            check i
              (multiplied dt
                 ~zeros:(List.exists signs_zeros (operands r))
                 ~count ~r:rs.(i) ~big:big.(i) ~small:small.(i) c.elements.(i)
                 e.elements.(i))
          done
      | Matmul_f _, [ a; b ] ->
          let a = float64 a and b = float64 b in
          let k = (Nx.shape a).(Array.length (Nx.shape a) - 1) in
          let rs = f64s (Nx.matmul a b) in
          let ss = f64s (Nx.matmul (Nx.abs a) (Nx.abs b)) in
          for i = 0 to n - 1 do
            check i
              (summed dt ~count:k ~r:rs.(i) ~s:ss.(i) c.elements.(i)
                 e.elements.(i))
          done
      | Fn _, _ ->
          for i = 0 to n - 1 do
            let ce = c.elements.(i) and ee = e.elements.(i) in
            check i
              ((Float.is_nan ce && Float.is_nan ee)
              || ce = ee
              || Float.is_finite ce && Float.is_finite ee
                 && Float.abs (ce -. ee)
                    <= Float.of_int transcendental_ulps
                       *. ulp dt (Float.max (Float.abs ce) (Float.abs ee)))
          done
      | _ -> assert false)

(* Counterexamples the property found, as it shrunk them. *)

let leaf ?(capture = false) dt shape values =
  Leaf { dt; shape; values; capture }

(* A reshape of a broadcast, which no strides can view: a copy. *)
let unviewable_reshape =
  Bin
    ( Add,
      Contiguous (leaf F32 [| 3 |] [| 0.; 0.; 0. |]),
      Slice
        ( [ Range (5, 2, -1) ],
          Reshape
            ( [| 8 |],
              Broadcast ([| 4; 2 |], leaf F32 [| 4; 1 |] [| 0.; 0.; 0.; 0. |])
            ) ) )

let found =
  [
    (* [take] without an axis at scalar indices raised. *)
    Bin
      ( Add,
        Take (None, leaf I64 [||] [| 0. |], leaf F32 [||] [| 0. |]),
        leaf ~capture:true F32 [||] [| 0. |] );
    (* A reshape eager nx refused for its operand's layout. *)
    unviewable_reshape;
    (* A widened float16 padded with a subnormal, read through a slice: the
       float16 load took the pad as its own value, which rounds to 0. *)
    Slice
      ( [ Range (1, 2, 1); Range (0, 2, 1); Range (1, 4, 2) ],
        Pad
          ( [| (1, 0); (0, 0); (0, 2) |],
            0x1p-149,
            Cast
              ( F32,
                leaf F16 [| 1; 2; 3 |] [| 0.; 0x1p-24; 0x1p-24; 0.; 0.; 0. |] )
          ) );
    (* A pad by a value that rounds to [0.] in float32 compared with [0.]: its
       bounds held the value unrounded, which excluded [0.]. *)
    Broadcast
      ( [| 2 |],
        Cmp
          ( Equal,
            Pad
              ( [| (0, 1) |],
                0x0.0000000000001p-1022,
                Where
                  ( leaf Bool [| 1 |] [| 0. |],
                    leaf ~capture:true F32 [| 1 |] [| 0x1.0000003496cdcp+128 |],
                    leaf ~capture:true F32 [| 1 |] [| 0x1.000174ec81b52p+128 |]
                  ) ),
            Take
              ( None,
                leaf I64 [| 1 |] [| 0. |],
                Cast (F32, leaf F16 [| 1; 0 |] [||]) ) ) );
  ]

(* [where (x < 0) x 0], which selects the zero at [x = -0.], the zero a captured
   scalar: a constant of the program. *)
let below_zero =
  let x = leaf F32 [| 3 |] [| -0.; 1.; -1. |] in
  let zero = Broadcast ([| 3 |], leaf ~capture:true F32 [||] [| 0. |]) in
  Where (Cmp (Less, x, zero), x, zero)

(* A sum of a minimum against a zero that folds to a constant, which clang's
   AArch64 backend turned into a minimum instruction keeping [-0.], and whose
   added [+0.] it then dropped. *)
let rounded_found =
  [
    Sum_f
      ( [],
        false,
        Bin
          ( Minimum,
            Take
              ( Some 0,
                leaf I64 [| 1 |] [| 0. |],
                Cast (F32, leaf F16 [| 0 |] [||]) ),
            leaf ~capture:true F32 [| 2 |] [| -0.; 0. |] ) );
  ]

let suite =
  group "a compiled program"
    [
      prop "computes eager's bits" ~count:300 ~examples:found programs
        compiled_is_eager;
      prop "computes eager's values within the stated rounding" ~count:300
        ~examples:rounded_found rounded_programs within_rounding;
      test "reshapes a layout no strides can view as eagerly" (fun () ->
          let dt = dtype_of unviewable_reshape in
          equal (outcomes ~zeros:false)
            (outcome dt (fun () -> eager unviewable_reshape))
            (outcome dt (fun () -> compiled unviewable_reshape)));
      test "selects a zero as eagerly, whatever the other value's sign"
        (fun () ->
          equal (outcomes ~zeros:false)
            (outcome F32 (fun () -> eager below_zero))
            (outcome F32 (fun () -> compiled below_zero)));
    ]

let () = exit (run "Rune.jit programs" [ suite ])
