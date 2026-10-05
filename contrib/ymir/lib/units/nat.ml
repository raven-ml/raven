(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A natural is an array of 30-bit limbs, least significant first, with no most
   significant zero limb: zero is the empty array. With 30-bit limbs a limb
   product plus two limbs stays below 2^61, inside OCaml's 63-bit ints. *)

type t = int array

let limb_bits = 30
let base = 1 lsl limb_bits
let mask = base - 1
let zero = [||]
let one = [| 1 |]
let is_zero n = Array.length n = 0

let trim a =
  let len = ref (Array.length a) in
  while !len > 0 && a.(!len - 1) = 0 do
    decr len
  done;
  if !len = Array.length a then a else Array.sub a 0 !len

let of_int n =
  if n < 0 then invalid_arg "Nat.of_int: negative";
  let rec limbs n =
    if n = 0 then [] else (n land mask) :: limbs (n lsr limb_bits)
  in
  Array.of_list (limbs n)

let int_bits n =
  let rec loop n k = if n = 0 then k else loop (n lsr 1) (k + 1) in
  loop n 0

let bit_length n =
  let len = Array.length n in
  if len = 0 then 0 else ((len - 1) * limb_bits) + int_bits n.(len - 1)

let to_int n =
  if bit_length n > Sys.int_size - 1 then None
  else begin
    let v = ref 0 in
    for i = Array.length n - 1 downto 0 do
      v := (!v lsl limb_bits) lor n.(i)
    done;
    Some !v
  end

let to_int64 n =
  let v = ref 0L in
  for i = min (Array.length n) 3 - 1 downto 0 do
    v := Int64.logor (Int64.shift_left !v limb_bits) (Int64.of_int n.(i))
  done;
  !v

let compare m n =
  let lm = Array.length m and ln = Array.length n in
  if lm <> ln then Int.compare lm ln
  else begin
    let i = ref (lm - 1) in
    while !i >= 0 && m.(!i) = n.(!i) do
      decr i
    done;
    if !i < 0 then 0 else Int.compare m.(!i) n.(!i)
  end

let equal m n = compare m n = 0

let low_bits_zero n k =
  let full = k / limb_bits and rest = k mod limb_bits in
  let rec limbs i =
    i >= full || i >= Array.length n || (n.(i) = 0 && limbs (i + 1))
  in
  limbs 0 && (full >= Array.length n || n.(full) land ((1 lsl rest) - 1) = 0)

(* Arithmetic *)

let add m n =
  let m, n = if Array.length m >= Array.length n then (m, n) else (n, m) in
  let lm = Array.length m and ln = Array.length n in
  let r = Array.make (lm + 1) 0 in
  let carry = ref 0 in
  for i = 0 to lm - 1 do
    let s = m.(i) + (if i < ln then n.(i) else 0) + !carry in
    r.(i) <- s land mask;
    carry := s lsr limb_bits
  done;
  r.(lm) <- !carry;
  trim r

let sub m n =
  if compare m n < 0 then invalid_arg "Nat.sub: negative result";
  let ln = Array.length n in
  let r = Array.copy m in
  let borrow = ref 0 in
  for i = 0 to Array.length m - 1 do
    let d = m.(i) - (if i < ln then n.(i) else 0) - !borrow in
    r.(i) <- d land mask;
    borrow := if d < 0 then 1 else 0
  done;
  trim r

let mul m n =
  let lm = Array.length m and ln = Array.length n in
  if lm = 0 || ln = 0 then zero
  else begin
    let r = Array.make (lm + ln) 0 in
    for i = 0 to lm - 1 do
      let mi = m.(i) and carry = ref 0 in
      for j = 0 to ln - 1 do
        let p = (mi * n.(j)) + r.(i + j) + !carry in
        r.(i + j) <- p land mask;
        carry := p lsr limb_bits
      done;
      r.(i + ln) <- !carry
    done;
    trim r
  end

let pow n k =
  if k < 0 then invalid_arg "Nat.pow: negative exponent";
  let rec loop acc n k =
    if k = 0 then acc
    else
      let acc = if k land 1 = 1 then mul acc n else acc in
      if k = 1 then acc else loop acc (mul n n) (k lsr 1)
  in
  loop one n k

let shift_left n k =
  if is_zero n || k = 0 then n
  else begin
    let limbs = k / limb_bits and bits = k mod limb_bits in
    let ln = Array.length n in
    let r = Array.make (ln + limbs + 1) 0 in
    for i = 0 to ln - 1 do
      let v = n.(i) lsl bits in
      r.(i + limbs) <- r.(i + limbs) lor (v land mask);
      r.(i + limbs + 1) <- v lsr limb_bits
    done;
    trim r
  end

let shift_right n k =
  let limbs = k / limb_bits and bits = k mod limb_bits in
  let ln = Array.length n in
  if limbs >= ln then zero
  else begin
    let r = Array.make (ln - limbs) 0 in
    for i = 0 to ln - limbs - 1 do
      let hi = if i + limbs + 1 < ln then n.(i + limbs + 1) else 0 in
      r.(i) <-
        (n.(i + limbs) lsr bits) lor (hi lsl (limb_bits - bits)) land mask
    done;
    trim r
  end

let rem_int n d =
  let r = ref 0 in
  for i = Array.length n - 1 downto 0 do
    r := (!r lsl limb_bits) lor n.(i) mod d
  done;
  !r

(* Four divisors at a time: their remainder chains are independent, so the
   divisions overlap instead of each waiting on the previous one. *)
let rem_ints n ds =
  let count = Array.length ds in
  let r = Array.make count 0 in
  let top = Array.length n - 1 in
  let i = ref 0 in
  while !i + 3 < count do
    let d0 = ds.(!i)
    and d1 = ds.(!i + 1)
    and d2 = ds.(!i + 2)
    and d3 = ds.(!i + 3) in
    let r0 = ref 0 and r1 = ref 0 and r2 = ref 0 and r3 = ref 0 in
    for j = top downto 0 do
      let l = Array.unsafe_get n j in
      r0 := (!r0 lsl limb_bits) lor l mod d0;
      r1 := (!r1 lsl limb_bits) lor l mod d1;
      r2 := (!r2 lsl limb_bits) lor l mod d2;
      r3 := (!r3 lsl limb_bits) lor l mod d3
    done;
    r.(!i) <- !r0;
    r.(!i + 1) <- !r1;
    r.(!i + 2) <- !r2;
    r.(!i + 3) <- !r3;
    i := !i + 4
  done;
  for k = !i to count - 1 do
    r.(k) <- rem_int n ds.(k)
  done;
  r

let div_int n d =
  let q = Array.make (Array.length n) 0 and r = ref 0 in
  for i = Array.length n - 1 downto 0 do
    let cur = (!r lsl limb_bits) lor n.(i) in
    q.(i) <- cur / d;
    r := cur mod d
  done;
  trim q

(* Knuth's algorithm D (TAOCP 4.3.1). The divisor is shifted so that its top
   limb has its high bit set, which makes each estimated quotient limb at most
   two above the true one. *)
let div_long m n =
  let shift = limb_bits - int_bits n.(Array.length n - 1) in
  let v = shift_left n shift in
  let u = Array.make (Array.length m + 1) 0 in
  let m = shift_left m shift in
  Array.blit m 0 u 0 (Array.length m);
  let ln = Array.length v in
  let lq = Array.length u - ln in
  let q = Array.make lq 0 in
  let vtop = v.(ln - 1) and vnext = v.(ln - 2) in
  for j = lq - 1 downto 0 do
    let num = (u.(j + ln) lsl limb_bits) lor u.(j + ln - 1) in
    let qhat = ref (num / vtop) and rhat = ref (num mod vtop) in
    while
      !rhat < base
      && (!qhat >= base
         || !qhat * vnext > (!rhat lsl limb_bits) lor u.(j + ln - 2))
    do
      decr qhat;
      rhat := !rhat + vtop
    done;
    let borrow = ref 0 and carry = ref 0 in
    for i = 0 to ln - 1 do
      let p = (!qhat * v.(i)) + !carry in
      carry := p lsr limb_bits;
      let d = u.(i + j) - (p land mask) - !borrow in
      u.(i + j) <- d land mask;
      borrow := if d < 0 then 1 else 0
    done;
    let d = u.(j + ln) - !carry - !borrow in
    u.(j + ln) <- d land mask;
    if d < 0 then begin
      (* The estimate was one too large: add the divisor back. *)
      decr qhat;
      let carry = ref 0 in
      for i = 0 to ln - 1 do
        let s = u.(i + j) + v.(i) + !carry in
        u.(i + j) <- s land mask;
        carry := s lsr limb_bits
      done;
      u.(j + ln) <- (u.(j + ln) + !carry) land mask
    end;
    q.(j) <- !qhat
  done;
  (trim q, shift_right (trim (Array.sub u 0 ln)) shift)

let div_rem m n =
  if is_zero n then raise Division_by_zero;
  if compare m n < 0 then (zero, m)
  else if Array.length n = 1 then
    let d = n.(0) in
    (div_int m d, of_int (rem_int m d))
  else div_long m n

(* [root_above n k] is above [n^(1/k)] by a relative margin of about 2^-20, from
   a float estimate of [log2 n] whose error is far below the margin. *)
let root_above n k =
  let bits = bit_length n in
  let t = min bits 62 in
  let top = Option.get (to_int (shift_right n (bits - t))) in
  let log2_root =
    (Float.log2 (Float.of_int top) +. Float.of_int (bits - t)) /. Float.of_int k
  in
  let e = max 0 (int_of_float log2_root - 52) in
  let m = Float.pow 2. (log2_root -. Float.of_int e) *. (1. +. 0x1p-20) in
  add (shift_left (of_int (int_of_float (Float.ceil m))) e) one

(* Newton's iteration for the integer root, started above the root, decreases
   strictly until it reaches it. From below the root its first step rises, so a
   seed that fell below falls back to the power of two above the root. *)
let root n k =
  if k < 1 then invalid_arg "Nat.root: degree below 1";
  if k = 1 || is_zero n then n
  else begin
    let bits = bit_length n in
    if bits <= k then one
    else begin
      let step x =
        let xk1 = pow x (k - 1) in
        div_int (add (mul (of_int (k - 1)) x) (fst (div_rem n xk1))) k
      in
      let rec descend x =
        let y = step x in
        if compare y x >= 0 then x else descend y
      in
      let seed = root_above n k in
      if compare (step seed) seed > 0 then
        descend (shift_left one ((bits + k - 1) / k))
      else descend seed
    end
  end

(* Decimal *)

let chunk_digits = 9

let pow10 =
  Array.init (chunk_digits + 1) (fun k ->
      int_of_string ("1" ^ String.make k '0'))

let chunk = pow10.(chunk_digits)

let of_digits s =
  let len = String.length s in
  let acc = ref zero and i = ref 0 in
  let first = len mod chunk_digits in
  let step width =
    let v = int_of_string (String.sub s !i width) in
    acc := add (mul !acc (of_int pow10.(width))) (of_int v);
    i := !i + width
  in
  if first > 0 then step first;
  while !i < len do
    step chunk_digits
  done;
  !acc

let to_string n =
  if is_zero n then "0"
  else begin
    let rec chunks n acc =
      if is_zero n then acc
      else chunks (div_int n chunk) (rem_int n chunk :: acc)
    in
    match chunks n [] with
    | [] -> assert false
    | top :: rest ->
        let b = Buffer.create ((List.length rest * chunk_digits) + 10) in
        Buffer.add_string b (string_of_int top);
        List.iter (fun c -> Buffer.add_string b (Printf.sprintf "%09d" c)) rest;
        Buffer.contents b
  end
