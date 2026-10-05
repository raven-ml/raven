(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The primes below [small_limit], sieved at initialisation. *)

let small_limit = 1 lsl 16

let small_primes =
  let composite = Bytes.make small_limit '\000' in
  let primes = ref [] in
  for i = 2 to small_limit - 1 do
    if Bytes.get composite i = '\000' then begin
      primes := i :: !primes;
      let j = ref (i * i) in
      while !j < small_limit do
        Bytes.set composite !j '\001';
        j := !j + i
      done
    end
  done;
  Array.of_list (List.rev !primes)

(* Modular arithmetic below 2^62. Operands are below the modulus [m]. *)

let add_mod x y m =
  let d = x - (m - y) in
  if d < 0 then d + m else d

let chunk_bits = 31
let chunk = 1 lsl chunk_bits

(* [mul_small a c m] is [a * c mod m] for [c <= 2^31]. The float quotient [a c /
   m] is within 2^-20 of the true one, so with [q] its rounding, [a c - q m]
   lies within (1/2 + 2^-20) m of zero: it fits an int, and computing it modulo
   2^63 gives it exactly. *)
let mul_small a c m =
  let q = Float.of_int a *. Float.of_int c /. Float.of_int m in
  let q = Float.to_int (Float.round q) in
  let r = (a * c) - (q * m) in
  if r < 0 then r + m else if r >= m then r - m else r

(* [mul_mod a b m] splits [b] into two 31-bit chunks: [a b = (a b1) 2^31 + a
   b0]. *)
let mul_mod a b m =
  if a < chunk && b < chunk then a * b mod m
  else
    let hi = mul_small (mul_small a (b lsr chunk_bits) m) chunk m in
    add_mod hi (mul_small a (b land (chunk - 1)) m) m

let pow_mod x k m =
  let rec loop acc x k =
    if k = 0 then acc
    else
      let acc = if k land 1 = 1 then mul_mod acc x m else acc in
      loop acc (mul_mod x x m) (k lsr 1)
  in
  loop 1 (x mod m) k

let rec gcd a b = if b = 0 then a else gcd b (a mod b)

(* Miller–Rabin with the first twelve primes as bases is exact for every n <
   3.3e24, so for every OCaml int. Requires an odd [n] above 37. *)
let miller_rabin_bases = [ 2; 3; 5; 7; 11; 13; 17; 19; 23; 29; 31; 37 ]

let is_prime n =
  let rec split d s =
    if d land 1 = 0 then split (d lsr 1) (s + 1) else (d, s)
  in
  let d, s = split (n - 1) 0 in
  let witness a =
    let x = pow_mod a d n in
    if x = 1 || x = n - 1 then false
    else begin
      let rec square x i =
        if i = s then true
        else
          let x = mul_mod x x n in
          if x = n - 1 then false else square x (i + 1)
      in
      square x 1
    end
  in
  not (List.exists witness miller_rabin_bases)

(* Pollard's rho with Brent's cycle detection on x^2 + c, from 2, with the
   differences batched [batch] at a time into one gcd. The result divides [n];
   it is [n] when this [c] fails to split it. *)
let batch = 128

let brent n c =
  let f x = add_mod (mul_mod x x n) c n in
  let diff x y = if x > y then x - y else y - x in
  let y = ref 2 and r = ref 1 and q = ref 1 and g = ref 1 in
  let x = ref 2 and ys = ref 2 in
  while !g = 1 do
    x := !y;
    for _ = 1 to !r do
      y := f !y
    done;
    let k = ref 0 in
    while !k < !r && !g = 1 do
      ys := !y;
      for _ = 1 to min batch (!r - !k) do
        y := f !y;
        q := mul_mod !q (diff !x !y) n
      done;
      g := gcd !q n;
      k := !k + batch
    done;
    r := 2 * !r
  done;
  if !g <> n then !g
  else begin
    (* The batch overshot: replay it one step at a time. *)
    let g = ref 1 in
    while !g = 1 do
      ys := f !ys;
      g := gcd (diff !x !ys) n
    done;
    !g
  end

(* [split n] is the prime factors of [n], an odd number with no prime factor
   below [small_limit] and above [small_limit^2], with multiplicity. *)
let rec split n =
  if is_prime n then [ n ]
  else
    let rec find c =
      let d = brent n c in
      if d = n then find (c + 1) else d
    in
    let d = find 1 in
    split d @ split (n / d)

let rec group = function
  | [] -> []
  | p :: rest ->
      let same, rest = List.partition (Int.equal p) rest in
      (p, 1 + List.length same) :: group rest

(* [divide_out n p] is [(k, n / p^k)] with [k] the multiplicity of [p]. *)
let divide_out n p =
  let rec loop n k = if n mod p = 0 then loop (n / p) (k + 1) else (k, n) in
  loop n 0

let factor n =
  if n < 1 then invalid_arg "Prime.factor: below 1";
  let found = ref [] and n = ref n and i = ref 0 in
  let count = Array.length small_primes in
  while !i < count && small_primes.(!i) * small_primes.(!i) <= !n do
    let p = small_primes.(!i) in
    let k, rest = divide_out !n p in
    if k > 0 then found := (p, k) :: !found;
    n := rest;
    incr i
  done;
  let large =
    if !n = 1 then []
    else if !n < small_limit * small_limit then [ (!n, 1) ]
    else group (List.sort Int.compare (split !n))
  in
  List.rev_append !found large

(* Smooth factoring of naturals *)

let smooth_limit = 1 lsl 24
let segment = 1 lsl 16

exception Found of (int * int) list

(* [factor_smooth] divides by the primes below [smooth_limit] while its cofactor
   is wider than an int, and factors the cofactor with [factor] once it is not.
   The primes above [small_limit] come from a sieve of segments of [segment]
   numbers by the primes below 2^12. *)
let factor_smooth n =
  let found = ref [] and n = ref n in
  (* [divide ps] divides [n] by each prime of [ps] that divides it. *)
  let divide ps =
    let rs = Nat.rem_ints !n ps in
    Array.iteri
      (fun i r ->
        if r = 0 then begin
          let p = ps.(i) and k = ref 0 in
          while Nat.rem_int !n p = 0 do
            n := Nat.div_int !n p;
            incr k
          done;
          found := (p, !k) :: !found;
          match Nat.to_int !n with
          | Some c -> raise_notrace (Found (List.rev_append !found (factor c)))
          | None -> ()
        end)
      rs
  in
  try
    (match Nat.to_int !n with
    | Some c -> raise_notrace (Found (factor c))
    | None -> ());
    divide small_primes;
    let composite = Bytes.create segment in
    let primes = Array.make segment 0 in
    let lo = ref small_limit in
    while !lo < smooth_limit do
      Bytes.fill composite 0 segment '\000';
      Array.iter
        (fun p ->
          if p * p < !lo + segment then begin
            let start = max (p * p) ((!lo + p - 1) / p * p) in
            let j = ref (start - !lo) in
            while !j < segment do
              Bytes.unsafe_set composite !j '\001';
              j := !j + p
            done
          end)
        small_primes;
      let count = ref 0 in
      for j = 0 to segment - 1 do
        if Bytes.unsafe_get composite j = '\000' then begin
          primes.(!count) <- !lo + j;
          incr count
        end
      done;
      divide (Array.sub primes 0 !count);
      lo := !lo + segment
    done;
    None
  with Found f -> Some f
