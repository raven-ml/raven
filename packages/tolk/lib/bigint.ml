(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* An integer that fits an int is that int, unboxed. Any other is a big: its
   two's complement in [bits]-bit limbs, least significant first, the last
   limb's top bit the sign repeated to its left, in the fewest limbs that hold
   it. [Obj.is_int] tells the two apart, which keeps arithmetic on small
   integers free of allocation. *)

type t
type big = int array

external of_small : int -> t = "%identity"
external of_big : big -> t = "%identity"
external to_small : t -> int = "%identity"
external to_big : t -> big = "%identity"

let is_small n = Obj.is_int (Obj.repr n)

exception Overflow

let bits = 30
let mask = (1 lsl bits) - 1
let int_limbs = (Sys.int_size / bits) + 1
let sign_bit limb = limb land (1 lsl (bits - 1)) <> 0
let negative (l : big) = sign_bit l.(Array.length l - 1)
let sign_limb l = if negative l then mask else 0
let limb l i = if i < Array.length l then l.(i) else sign_limb l

let limbs n : big =
  if is_small n then
    let n = to_small n in
    Array.init int_limbs (fun i ->
        (n asr Int.min (bits * i) (Sys.int_size - 1)) land mask)
  else to_big n

(* [l]'s integer, from any array of limbs that holds it with its sign. *)
let normalize (l : big) =
  let sign = sign_limb l in
  let n = ref (Array.length l) in
  while
    !n > 1 && l.(!n - 1) = sign && Bool.equal (sign_bit l.(!n - 2)) (sign <> 0)
  do
    decr n
  done;
  let n = !n in
  let top = if sign = 0 then l.(n - 1) else l.(n - 1) - (1 lsl bits) in
  let fits_int =
    n < int_limbs
    || n = int_limbs
       &&
       let s = top asr (Sys.int_size - 1 - (bits * (n - 1))) in
       s = 0 || s = -1
  in
  if fits_int then (
    let v = ref top in
    for i = n - 2 downto 0 do
      v := (!v lsl bits) lor l.(i)
    done;
    of_small !v)
  else of_big (if n = Array.length l then l else Array.sub l 0 n)

(* Constants *)

let zero = of_small 0
let one = of_small 1
let minus_one = of_small (-1)

(* Predicates and comparisons *)

let sign n =
  if is_small n then Int.compare (to_small n) 0
  else if negative (to_big n) then -1
  else 1

let compare_big a b =
  let na = negative a in
  if na <> negative b then if na then -1 else 1
  else
    let la = Array.length a and lb = Array.length b in
    if la <> lb then if Bool.equal (la > lb) na then -1 else 1
    else
      let i = ref (la - 1) in
      while !i > 0 && a.(!i) = b.(!i) do
        decr i
      done;
      Int.compare a.(!i) b.(!i)

let compare a b =
  match (is_small a, is_small b) with
  | true, true -> Int.compare (to_small a) (to_small b)
  | true, false -> -sign b
  | false, true -> sign a
  | false, false -> compare_big (to_big a) (to_big b)

let equal a b =
  match (is_small a, is_small b) with
  | true, true -> Int.equal (to_small a) (to_small b)
  | false, false -> compare_big (to_big a) (to_big b) = 0
  | _ -> false

let leq a b = compare a b <= 0
let geq a b = compare a b >= 0
let lt a b = compare a b < 0
let gt a b = compare a b > 0
let min a b = if leq a b then a else b
let max a b = if geq a b then a else b

(* Arithmetic *)

let add_limbs a b carry =
  let n = Int.max (Array.length a) (Array.length b) + 1 in
  let r = Array.make n 0 and c = ref carry in
  for i = 0 to n - 1 do
    let s = limb a i + limb b i + !c in
    r.(i) <- s land mask;
    c := s lsr bits
  done;
  normalize r

let complement l = Array.map (fun x -> x lxor mask) l

let add a b =
  if is_small a && is_small b then
    let x = to_small a and y = to_small b in
    let s = x + y in
    if x lxor s land (y lxor s) >= 0 then of_small s
    else add_limbs (limbs a) (limbs b) 0
  else add_limbs (limbs a) (limbs b) 0

let sub a b =
  if is_small a && is_small b then
    let x = to_small a and y = to_small b in
    let s = x - y in
    if x lxor y land (x lxor s) >= 0 then of_small s
    else add_limbs (limbs a) (complement (limbs b)) 1
  else add_limbs (limbs a) (complement (limbs b)) 1

let neg n = sub zero n
let abs n = if sign n < 0 then neg n else n
let succ n = add n one
let pred n = sub n one

(* [n]'s absolute value in limbs read unsigned. *)
let magnitude n = limbs (abs n)

let mul_limbs a b =
  let x = magnitude a and y = magnitude b in
  let nx = Array.length x and ny = Array.length y in
  let r = Array.make (nx + ny + 1) 0 in
  for i = 0 to nx - 1 do
    let c = ref 0 in
    for j = 0 to ny - 1 do
      let s = r.(i + j) + (x.(i) * y.(j)) + !c in
      r.(i + j) <- s land mask;
      c := s lsr bits
    done;
    r.(i + ny) <- !c
  done;
  let p = normalize r in
  if sign a * sign b < 0 then neg p else p

let mul a b =
  if is_small a && is_small b then
    let x = to_small a and y = to_small b in
    let p = x * y in
    if x = 0 || (p / x = y && not (x = -1 && y = min_int)) then of_small p
    else mul_limbs a b
  else mul_limbs a b

(* [x], unsigned limbs, divided by [d], a limb other than zero. *)
let short_div x d =
  let q = Array.make (Array.length x + 1) 0 and r = ref 0 in
  for i = Array.length x - 1 downto 0 do
    let cur = (!r lsl bits) lor x.(i) in
    q.(i) <- cur / d;
    r := cur mod d
  done;
  (normalize q, !r)

let shift_left n k =
  if k < 0 then invalid_arg "Bigint.shift_left: negative shift";
  if is_small n && k < Sys.int_size && (to_small n lsl k) asr k = to_small n
  then of_small (to_small n lsl k)
  else if sign n = 0 then zero
  else
    let l = limbs n and q = k / bits and r = k mod bits in
    normalize
      (Array.init
         (Array.length l + q + 1)
         (fun j ->
           let lo = if j > q then limb l (j - q - 1) lsr (bits - r) else 0 in
           let hi = if j >= q then limb l (j - q) lsl r else 0 in
           lo lor hi land mask))

let numbits n =
  let int_numbits x =
    let k = ref 0 and x = ref x in
    while !x <> 0 do
      incr k;
      x := !x lsr 1
    done;
    !k
  in
  if is_small n && to_small n <> min_int then int_numbits (Int.abs (to_small n))
  else
    let m = magnitude n in
    let i = ref (Array.length m - 1) in
    while m.(!i) = 0 do
      decr i
    done;
    (bits * !i) + int_numbits m.(!i)

(* [x] and [y] non-negative, [y] non-zero. *)
let long_div x y =
  let q = ref zero and r = ref x in
  for i = numbits x - numbits y downto 0 do
    let s = shift_left y i in
    q := shift_left !q 1;
    if geq !r s then (
      r := sub !r s;
      q := succ !q)
  done;
  (!q, !r)

let div_rem a b =
  if sign b = 0 then raise Division_by_zero;
  let x = abs a and y = abs b in
  let q, r =
    if is_small y && to_small y <= mask then
      let q, r = short_div (limbs x) (to_small y) in
      (q, of_small r)
    else long_div x y
  in
  ((if sign a * sign b < 0 then neg q else q), if sign a < 0 then neg r else r)

(* Whether [a] and [b] divide as ints: [min_int / -1] overflows. *)
let divides_small a b =
  is_small a && is_small b
  && to_small b <> 0
  && not (to_small a = min_int && to_small b = -1)

let div a b =
  if divides_small a b then of_small (to_small a / to_small b)
  else fst (div_rem a b)

let rem a b =
  if divides_small a b then of_small (to_small a mod to_small b)
  else snd (div_rem a b)

let fdiv a b =
  if divides_small a b then
    let x = to_small a and y = to_small b in
    let r = x mod y in
    of_small (if r <> 0 && r lxor y < 0 then (x / y) - 1 else x / y)
  else
    let q, r = div_rem a b in
    if sign r <> 0 && sign r <> sign b then pred q else q

let cdiv a b =
  let q, r = div_rem a b in
  if sign r <> 0 && sign r = sign b then succ q else q

let ediv_rem a b =
  let q, r = div_rem a b in
  if sign r >= 0 then (q, r)
  else if sign b > 0 then (pred q, add r b)
  else (succ q, sub r b)

let divisible a b = if sign b = 0 then sign a = 0 else sign (rem a b) = 0
let ediv a b = fst (ediv_rem a b)
let erem a b = snd (ediv_rem a b)

let rec gcd a b =
  if is_small a && is_small b && to_small a <> min_int && to_small b <> min_int
  then (
    let x = ref (Int.abs (to_small a)) and y = ref (Int.abs (to_small b)) in
    while !y <> 0 do
      let r = !x mod !y in
      x := !y;
      y := r
    done;
    of_small !x)
  else if sign b = 0 then abs a
  else gcd b (rem a b)

let pow b e =
  if e < 0 then invalid_arg "Bigint.pow: negative exponent";
  let rec go acc b e =
    if e = 0 then acc
    else
      let acc = if e land 1 = 1 then mul acc b else acc in
      go acc (if e > 1 then mul b b else b) (e lsr 1)
  in
  go one b e

(* Bits *)

let bitwise f a b =
  let a = limbs a and b = limbs b in
  normalize
    (Array.init
       (Int.max (Array.length a) (Array.length b))
       (fun i -> f (limb a i) (limb b i)))

let logand a b =
  if is_small a && is_small b then of_small (to_small a land to_small b)
  else bitwise ( land ) a b

let logor a b =
  if is_small a && is_small b then of_small (to_small a lor to_small b)
  else bitwise ( lor ) a b

let logxor a b =
  if is_small a && is_small b then of_small (to_small a lxor to_small b)
  else bitwise ( lxor ) a b

let lognot n =
  if is_small n then of_small (lnot (to_small n))
  else normalize (complement (to_big n))

let shift_right n k =
  if k < 0 then invalid_arg "Bigint.shift_right: negative shift";
  if is_small n then of_small (to_small n asr Int.min k (Sys.int_size - 1))
  else
    let l = to_big n and q = k / bits and r = k mod bits in
    let len = Array.length l in
    if q >= len then if negative l then minus_one else zero
    else
      normalize
        (Array.init (len - q) (fun j ->
             (limb l (j + q) lsr r)
             lor (limb l (j + q + 1) lsl (bits - r))
             land mask))

let check_field fn off len =
  if off < 0 || len <= 0 then
    invalid_arg
      (Printf.sprintf "Bigint.%s: bits %d to %d are not a field" fn off
         (off + len - 1))

let field n off len = logand (shift_right n off) (pred (shift_left one len))

let extract n off len =
  check_field "extract" off len;
  if is_small n then
    let v = to_small n asr Int.min off (Sys.int_size - 1) in
    if len < Sys.int_size - 1 then of_small (v land ((1 lsl len) - 1))
    else if v >= 0 then of_small v
    else field n off len
  else field n off len

let signed_extract n off len =
  check_field "signed_extract" off len;
  if is_small n && len <= Sys.int_size then
    let v = to_small n asr Int.min off (Sys.int_size - 1) in
    let s = Sys.int_size - len in
    of_small ((v lsl s) asr s)
  else
    let e = extract n off len in
    if sign (shift_right e (len - 1)) <> 0 then sub e (shift_left one len)
    else e

let trailing_zeros n =
  let int_trailing_zeros x =
    let k = ref 0 and x = ref x in
    while !x land 1 = 0 do
      incr k;
      x := !x asr 1
    done;
    !k
  in
  if sign n = 0 then max_int
  else if is_small n then int_trailing_zeros (to_small n)
  else
    let l = to_big n in
    let i = ref 0 in
    while l.(!i) = 0 do
      incr i
    done;
    (bits * !i) + int_trailing_zeros l.(!i)

let popcount n =
  let int_popcount x =
    let k = ref 0 and x = ref x in
    while !x <> 0 do
      k := !k + (!x land 1);
      x := !x lsr 1
    done;
    !k
  in
  if sign n < 0 then raise Overflow
  else if is_small n then int_popcount (to_small n)
  else Array.fold_left (fun k l -> k + int_popcount l) 0 (to_big n)

let sqrt n =
  if sign n < 0 then invalid_arg "Bigint.sqrt: negative argument";
  let rec go x =
    let y = shift_right (add x (div n x)) 1 in
    if geq y x then x else go y
  in
  if sign n = 0 then zero else go (shift_left one ((numbits n + 1) / 2))

(* Conversions *)

let of_int = of_small

let of_int64 x =
  let i = Int64.to_int x in
  if Int64.equal (Int64.of_int i) x then of_small i
  else
    normalize
      (Array.init
         ((64 / bits) + 1)
         (fun k ->
           Int64.(to_int (logand (shift_right x (bits * k)) (of_int mask)))))

let of_int32 x = of_int64 (Int64.of_int32 x)
let of_nativeint x = of_int64 (Int64.of_nativeint x)

let of_int32_unsigned x =
  of_int64 (Int64.logand (Int64.of_int32 x) 0xFFFF_FFFFL)

let of_int64_unsigned x =
  if Int64.compare x 0L >= 0 then of_int64 x
  else add (of_int64 x) (shift_left one 64)

let of_float x =
  if not (Float.is_finite x) then raise Overflow;
  if Float.abs x < Float.ldexp 1. (Sys.int_size - 1) then
    of_small (Float.to_int x)
  else
    let m, e = Float.frexp (Float.abs x) in
    let a =
      shift_left (of_int64 (Int64.of_float (Float.ldexp m 53))) (e - 53)
    in
    if x < 0. then neg a else a

let of_string s =
  let fail () =
    invalid_arg (Printf.sprintf "Bigint.of_string: %S is not an integer" s)
  in
  let n = String.length s in
  let negative, i =
    if n > 0 && s.[0] = '-' then (true, 1)
    else if n > 0 && s.[0] = '+' then (false, 1)
    else (false, 0)
  in
  let base, i =
    if i + 1 < n && s.[i] = '0' then
      match s.[i + 1] with
      | 'x' | 'X' -> (16, i + 2)
      | 'o' | 'O' -> (8, i + 2)
      | 'b' | 'B' -> (2, i + 2)
      | _ -> (10, i)
    else (10, i)
  in
  let digit = function
    | '0' .. '9' as c -> Char.code c - Char.code '0'
    | 'a' .. 'f' as c -> Char.code c - Char.code 'a' + 10
    | 'A' .. 'F' as c -> Char.code c - Char.code 'A' + 10
    | _ -> base
  in
  if i >= n || s.[i] = '_' then fail ();
  let acc = ref zero and b = of_small base in
  for j = i to n - 1 do
    if s.[j] <> '_' then (
      let d = digit s.[j] in
      if d >= base then fail ();
      acc := add (mul !acc b) (of_small d))
  done;
  if negative then neg !acc else !acc

let fits_int = is_small
let int64_min = of_int64 Int64.min_int
let int64_max = of_int64 Int64.max_int
let fits_int64 n = is_small n || (leq int64_min n && leq n int64_max)

(* [n]'s low 64 bits. *)
let low_int64 n =
  if is_small n then Int64.of_int (to_small n)
  else
    let l = to_big n and r = ref 0L in
    for k = 64 / bits downto 0 do
      r := Int64.(logor (shift_left !r bits) (of_int (limb l k)))
    done;
    !r

let to_int n = if is_small n then to_small n else raise Overflow
let to_int64 n = if fits_int64 n then low_int64 n else raise Overflow

let to_int32 n =
  let x = to_int64 n in
  if Int64.equal (Int64.of_int32 (Int64.to_int32 x)) x then Int64.to_int32 x
  else raise Overflow

let to_unsigned width n =
  if sign n >= 0 && numbits n <= width then low_int64 n else raise Overflow

let to_int32_unsigned n = Int64.to_int32 (to_unsigned 32 n)
let to_int64_unsigned n = to_unsigned 64 n

(* Rounding to odd onto 55 bits leaves two bits below a double's 53 for the
   conversion to round to nearest once. *)
let to_float n =
  if is_small n then Float.of_int (to_small n)
  else
    let a = abs n in
    let shift = numbits a - 55 in
    let q = shift_right a shift in
    let q = if trailing_zeros a >= shift then q else logor q one in
    let x = Float.ldexp (Int64.to_float (low_int64 q)) shift in
    if sign n < 0 then -.x else x

let to_string n =
  if is_small n then Int.to_string (to_small n)
  else
    let rec chunks m acc =
      if sign m = 0 then acc
      else
        let q, r = short_div (limbs m) 1_000_000_000 in
        chunks q (r :: acc)
    in
    match chunks (abs n) [] with
    | first :: rest ->
        String.concat ""
          ((if sign n < 0 then "-" else "")
          :: Int.to_string first
          :: List.map (Printf.sprintf "%09d") rest)
    | [] -> assert false

let pp_print ppf n = Format.pp_print_string ppf (to_string n)

(* Operators *)

let ( + ) = add
let ( - ) = sub
let ( * ) = mul
let ( / ) = div
let ( lsl ) = shift_left
let ( = ) = equal
let ( < ) = lt
let ( <= ) = leq
let ( > ) = gt
let ( >= ) = geq
