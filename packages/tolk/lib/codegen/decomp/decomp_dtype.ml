(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

open Ops
open Shape

let shl = Transcendental.shl
let shr = Transcendental.shr
let rule = Pattern_matcher.rule
let rule_ctx = Pattern_matcher.rule_ctx
let ops l = Op.Set.of_list l
let uint = Dtype.Uint32

(* Long as two ints *)

let longs = Dtype.[ Int64; Uint64 ]

let l2i_dt : Dtype.t -> Dtype.t = function
  | Int64 -> Int32
  | Uint64 -> Uint32
  | dt -> invalid_arg (Format.asprintf "%a is not a long" Dtype.pp dt)

let is_long dt = List.mem dt longs

let unpack32 v =
  let u = bitcast v uint in
  (O.(u land int 0xFFFF), shr u 16)

let reindex ?(mul = 1) idx off =
  match (op idx, src idx) with
  | Op.Shrink, buf :: i :: _ ->
      if mul <> 1 then invalid_arg "can't reindex SHRINK with mul != 1";
      replace idx ~op:Op.Index ~src:[ buf; O.(i + int off) ]
  | _, buf :: i :: rest ->
      replace idx ~src:(buf :: O.((i * int mul) + int off) :: rest)
  | _ -> invalid_arg "reindex needs a buffer and an index"

let pair = function [ a; b ] -> (a, b) | _ -> invalid_arg "expected two words"

(* Section 4.3.1 of TAOCP. The result is the two words of a long, low first, or
   one value for a comparison or a cast to another type. *)
let rec l2i op dt uops =
  let zero = const ~dtype:dt (`Int Bigint.zero) in
  let bin () =
    match uops with
    | [ a0; a1; b0; b1 ] -> (a0, a1, b0, b1)
    | _ ->
        invalid_arg
          (Printf.sprintf "long %s needs two long operands" (Op.name op))
  in
  let words a0 a1 = [ a0; a1 ] in
  match op with
  | Op.Neg -> l2i Op.Sub dt (zero :: zero :: uops)
  | Op.Cast when is_long dt && not (Dtype.is_float (dtype (List.hd uops))) ->
      (* The high word is the sign extension; unsigned and bool sources zero
         extend. *)
      let x = List.hd uops in
      let lo = cast x (l2i_dt dt) in
      let zero = const_like lo (`Int Bigint.zero) in
      if Dtype.equal (dtype x) Bool || List.mem (dtype x) Dtype.uints then
        words lo zero
      else
        words lo
          (where
             O.(x < const_like x (`Int Bigint.zero))
             (const_like lo (`Int Bigint.minus_one))
             zero)
  | Op.Cast when is_long dt ->
      (* The words of the truncated float's magnitude, its quotient and
         remainder by 2^32, which the float holds exactly and each word holds,
         negated as a long if the float is negative. Converting the float itself
         to a word is undefined past 2^31. *)
      let t = trunc (List.hd uops) in
      let negative = O.(t < int 0) in
      let a = where negative (neg t) t in
      let q = trunc O.(a / int 0x1_0000_0000) in
      let hi = cast q uint and lo = cast O.(a - (q * int 0x1_0000_0000)) uint in
      let n0, n1 = pair (l2i Op.Neg uint [ lo; hi ]) in
      words
        (bitcast (where negative n0 lo) (l2i_dt dt))
        (bitcast (where negative n1 hi) (l2i_dt dt))
  | Op.Cast when Dtype.is_float dt && not (Dtype.equal dt Float64) ->
      let a0, a1 = pair uops in
      [ cast (long_to_float dt a0 a1) dt ]
  | Op.Cast when Dtype.is_float dt ->
      let a0, a1 = pair uops in
      let small =
        O.(
          eq a1 (int 0)
          land (a0 >= int 0)
          lor (eq a1 (int (-1)) land (a0 < int 0)))
      in
      [
        where small (cast a0 dt)
          (cast
             O.((cast a1 dt * int 0x1_0000_0000) + cast (bitcast a0 uint) dt)
             dt);
      ]
  | Op.Cast -> [ cast (bitcast (List.hd uops) uint) dt ]
  | Op.Bitcast ->
      let a0, a1 = pair uops in
      words (bitcast a0 dt) (bitcast a1 dt)
  | Op.Shl ->
      let a0, a1, b0 = shift_operands op uops in
      let a0u = bitcast a0 uint and a1u = bitcast a1 uint in
      let n = cast O.(b0 land int 31) uint in
      let lo = bitcast O.(a0u lsl n) dt in
      let hi =
        bitcast O.((a1u lsl n) lor ((a0u lsr int 1) lsr (int 31 - n))) dt
      in
      let far = O.(b0 >= int 32) in
      words (where far zero lo) (where far lo hi)
  | Op.Shr ->
      let a0, a1, b0 = shift_operands op uops in
      let a0u = bitcast a0 uint and a1u = bitcast a1 uint in
      let n = cast O.(b0 land int 31) uint in
      let lo =
        bitcast O.((a0u lsr n) lor ((a1u lsl int 1) lsl (int 31 - n))) dt
      in
      let hi = O.(a1 lsr (b0 land int 31)) in
      (* The vacated high word: sign bits when signed, else 0. *)
      let fill = if Dtype.equal dt Int32 then O.(a1 lsr int 31) else zero in
      let far = O.(b0 >= int 32) in
      words (where far hi lo) (where far fill hi)
  | Op.Add ->
      let a0, a1, b0, b1 = bin () in
      let low = O.(a0 + b0) in
      words low O.(a1 + b1 + (bitcast low uint < bitcast a0 uint))
  | Op.Sub ->
      let a0, a1, b0, b1 = bin () in
      words O.(a0 - b0) O.(a1 - b1 - (bitcast a0 uint < bitcast b0 uint))
  | Op.Mul ->
      let a0, a1, b0, b1 = bin () in
      let a00, a01 = unpack32 a0 and b00, b01 = unpack32 b0 in
      let cross0 = O.(a00 * b01) and cross1 = O.(a01 * b00) in
      let mid =
        l2i Op.Add dt
          [
            bitcast (shl cross0 16) dt;
            bitcast (shr cross0 16) dt;
            bitcast (shl cross1 16) dt;
            bitcast (shr cross1 16) dt;
          ]
      in
      l2i Op.Add dt
        (mid
        @ [
            bitcast O.(a00 * b00) dt;
            O.(bitcast (a01 * b01) dt + (a0 * b1) + (a1 * b0));
          ])
  | Op.Cdiv | Op.Cmod -> long_division op dt (bin ())
  | Op.Cmplt ->
      let a0, a1, b0, b1 = bin () in
      [ O.((a1 < b1) lor (eq a1 b1 land (bitcast a0 uint < bitcast b0 uint))) ]
  | Op.Cmpeq ->
      let a0, a1, b0, b1 = bin () in
      [ O.(eq a0 b0 land eq a1 b1) ]
  | Op.Cmpne ->
      let a0, a1, b0, b1 = bin () in
      [ O.((a0 <> b0) lor (a1 <> b1)) ]
  | Op.Xor | Op.Or | Op.And ->
      let a0, a1, b0, b1 = bin () in
      words (v op ~src:[ a0; b0 ]) (v op ~src:[ a1; b1 ])
  | Op.Where -> (
      match uops with
      | [ c; a0; a1; b0; b1 ] -> words (where c a0 b0) (where c a1 b1)
      | _ -> invalid_arg "long WHERE needs a condition and two longs")
  | Op.Max ->
      let a0, a1, b0, b1 = bin () in
      l2i Op.Where dt (l2i Op.Cmplt dt uops @ [ b0; b1; a0; a1 ])
  | op ->
      invalid_arg
        (Printf.sprintf "long decomposition of %s unsupported" (Op.name op))

(* A long as a float32 rounded once: to nearest for a float32, and to odd for a
   narrower float, whose own conversion then rounds it once. The top 32 bits of
   its magnitude, with a sticky bit for the rest, hold what a float32 keeps. *)
and long_to_float dt a0 a1 =
  let lo = bitcast a0 uint and hi = bitcast a1 uint in
  let negative, hi, lo =
    if Dtype.equal (dtype a1) Int32 then
      let neg = O.(a1 < int 0) in
      let n0, n1 = pair (l2i Op.Neg uint [ lo; hi ]) in
      (Some neg, where neg n1 hi, where neg n0 lo)
    else (None, hi, lo)
  in
  (* The word holding the leading bit, the word below it, and its weight. *)
  let high = eq hi (int 0) in
  let top = where high lo hi and below = where high (int 0) lo in
  let weight = where high (int 0) (int 32) in
  (* The leading bit's position: the exponent of [top] as a float32, which is
     one too many where the conversion rounded up to a power of two. *)
  let e = O.((bitcast (cast top Float32) uint lsr int 23) - int 127) in
  let e = where O.(e < int 31) e (int 31) in
  let e = where (eq O.(top lsr e) (int 0)) O.(e - int 1) e in
  let bits = O.(e + int 1) in
  let up = O.(int 32 - bits) in
  let m =
    O.(
      (top lsl up)
      lor ((below lsr int 1) lsr (bits - int 1))
      lor cast (below lsl up <> int 0) uint)
  in
  let scale k =
    let bias = int (127 - k) in
    bitcast O.((bits + weight + bias) lsl int 23) Float32
  in
  let f =
    if Dtype.equal dt Float32 then O.(cast m Float32 * scale 32)
    else
      let m24 = O.((m lsr int 8) lor cast (m land int 0xFF <> int 0) uint) in
      O.(cast m24 Float32 * scale 24)
  in
  match negative with Some n -> where n (neg f) f | None -> f

(* A shift's count is a single word. *)
and shift_operands op = function
  | a0 :: a1 :: b0 :: _ -> (a0, a1, b0)
  | _ ->
      invalid_arg
        (Printf.sprintf "long %s needs a long and a count" (Op.name op))

(* TAOCP's Algorithm 4.3.1D could be faster, but it must be parameterised over
   the width of the divisor. *)
and long_division op dt (a0, a1, b0, b1) =
  let zero = const ~dtype:dt (`Int Bigint.zero) in
  let signed = Dtype.equal dt Int32 in
  let negate a0 a1 = pair (l2i Op.Neg uint [ a0; a1 ]) in
  let magnitude w0 w1 =
    let u0 = bitcast w0 uint and u1 = bitcast w1 uint in
    let neg = O.(w1 < zero) in
    let n0, n1 = negate u0 u1 in
    (neg, where neg n0 u0, where neg n1 u1)
  in
  let a_neg, a0, a1, b_neg, b0, b1 =
    if signed then
      let a_neg, a0, a1 = magnitude a0 a1 and b_neg, b0, b1 = magnitude b0 b1 in
      (Some a_neg, a0, a1, Some b_neg, b0, b1)
    else (None, a0, a1, None, b0, b1)
  in
  let z = const ~dtype:uint (`Int Bigint.zero) in
  let q = ref (z, z) and r = ref (z, z) in
  for i = 63 downto 0 do
    let r0, r1 = !r in
    let r0, r1 =
      pair (l2i Op.Shl uint [ r0; r1; const ~dtype:uint (`Int Bigint.one); z ])
    in
    let bit =
      List.hd
        (l2i Op.Shr uint [ a0; a1; const ~dtype:uint (`Int (Bigint.of_int i)); z ])
    in
    let r0 = O.(r0 lor (bit land int 1)) in
    let cond = logical_not (List.hd (l2i Op.Cmplt uint [ r0; r1; b0; b1 ])) in
    let d0, d1 = pair (l2i Op.Sub uint [ r0; r1; b0; b1 ]) in
    let set w = O.(w lor shl (cast cond uint) (i mod 32)) in
    (q := match !q with q0, q1 -> if i < 32 then (set q0, q1) else (q0, set q1));
    r := pair (l2i Op.Where uint [ cond; d0; d1; r0; r1 ])
  done;
  let (q0, q1), (r0, r1) = (!q, !r) in
  match (a_neg, b_neg) with
  | Some a_neg, Some b_neg ->
      let as_dt (w0, w1) = pair (l2i Op.Bitcast dt [ w0; w1 ]) in
      let nq0, nq1 = as_dt (negate q0 q1) in
      let nr0, nr1 = as_dt (negate r0 r1) in
      let q0, q1 = as_dt (q0, q1) and r0, r1 = as_dt (r0, r1) in
      if op = Op.Cmod then [ where a_neg nr0 r0; where a_neg nr1 r1 ]
      else
        let s = O.(a_neg lxor b_neg) in
        [ where s nq0 q0; where s nq1 q1 ]
  | _ -> if op = Op.Cmod then [ r0; r1 ] else [ q0; q1 ]

let l2i_define x =
  (* A variable cannot be decomposed. *)
  match arg x with
  | Param p when addrspace x = Some Dtype.Alu ->
      invalid_arg
        (Printf.sprintf "long decomposition of variable %s unsupported"
           (Option.value p.name ~default:"None"))
  | Param p ->
      v (op x)
        ~arg:
          (Param
             {
               p with
               dtype = l2i_dt p.dtype;
               size = Option.map (fun n -> 2 * n) p.size;
             })
        ?tag:(tag x)
  | _ -> invalid_arg "a long definition needs a parameter argument"

(* The word a node becomes, [0] for the low word and [1] for the high one, and
   the type of that word. *)
let word_tag w dt = Tag.Tuple [ Int w; Dtype dt ]

let word x =
  match tag x with
  | Some (Tuple [ Int w; Dtype _ ]) -> w
  | _ -> invalid_arg "a long node needs the word it becomes"

let lo_hi dt a = [ rtag ~tag:(word_tag 0 dt) a; rtag ~tag:(word_tag 1 dt) a ]

(* Memo of the splits of one pass: both words of a node ask for the same one. *)
module Splits = Hashtbl.Make (struct
  type t = Op.t * Dtype.t * Ops.t list

  let equal (o0, d0, l0) (o1, d1, l1) =
    o0 = o1 && Dtype.equal d0 d1 && List.equal ( == ) l0 l1

  let hash (o, d, l) =
    Hashtbl.hash (Op.to_int o, Dtype.hash d, List.map Ops.hash l)
end)

(* l2i computes on its inputs: the rules split them into words first, and l2i
   recurses on itself. *)
let rec split_l2i ctx op dt uops =
  let key = (op, dt, uops) in
  match Splits.find_opt ctx key with
  | Some words -> words
  | None ->
      let words =
        graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx (sink uops)
          (Before_sources (Lazy.force pm_long_decomp))
        |> src |> l2i op dt
      in
      Splits.replace ctx key words;
      words

and pm_long_decomp =
  lazy
    (let long = Upat.var ~dtype:longs in
     let tagged x f =
       match tag x with None -> None | Some _ -> Some (f (word x))
     in
     let w ws i = List.nth ws i in
     Pattern_matcher.v
       (fun () -> [
         (* The decomposition's own rewrite can mint bare constants: they commit
            at the long sibling's type. *)
         rule (Upat.v ~op:Op.Set.all ~name:"x" ()) (fun m ->
             let x = m "x" in
             Some
               (Uop_weak.commit_weak_consts x
                  (List.find_map
                     (fun s ->
                       if is_long (dtype s) then Some (dtype s) else None)
                     (src x))));
         rule (Upat.v ~op:Op.Set.defines ~dtype:longs ~name:"x" ()) (fun m ->
             Some (l2i_define (m "x")));
         rule (Upat.op Op.Index ~dtype:longs ~name:"x") (fun m ->
             let x = m "x" in
             tagged x (fun w -> replace (reindex ~mul:2 x w) ~tag:None));
         rule
           (Upat.op Op.Store ~name:"st" ~src:[ long "idx"; Upat.var "val" ])
           (fun m ->
             let st = m "st" and idx = m "idx" and value = m "val" in
             if Option.is_some (tag value) then None
             else
               let dt = l2i_dt (dtype idx) in
               let half w =
                 replace st
                   ~src:
                     [
                       rtag ~tag:(word_tag w dt) idx;
                       rtag ~tag:(word_tag w dt) value;
                     ]
               in
               Some (group [ half 0; half 1 ]));
         rule_ctx
           (Upat.v ~op:Op.Set.comparison
              ~perm:[ long "a"; Upat.wild ]
              ~name:"x" ())
           (fun ctx m ->
             let x = m "x" in
             let dt = l2i_dt (dtype (m "a")) in
             Some
               (List.hd
                  (split_l2i ctx (op x) dt (List.concat_map (lo_hi dt) (src x)))));
         rule_ctx
           (Upat.op Op.Cast ~dtype:longs ~src:[ long "a" ] ~name:"x")
           (fun ctx m ->
             let x = m "x" and a = m "a" in
             let words =
               split_l2i ctx Op.Bitcast
                 (l2i_dt (dtype x))
                 (lo_hi (l2i_dt (dtype a)) a)
             in
             Some (w words (word x)));
         (* A constant splits by value; the general cast rule would drop its
            high word. *)
         rule
           (Upat.op Op.Cast ~name:"x"
              ~src:[ Upat.op Op.Const ~name:"c" ]
              ~tag:
                (List.concat_map
                   (fun w -> List.map (word_tag w) Dtype.[ Int32; Uint32 ])
                   [ 0; 1 ]))
           (fun m ->
             let x = m "x" and c = m "c" in
             match (tag x, value c) with
             | Some (Tuple [ Int w; Dtype dt ]), (#Dtype.value as v) ->
                 let n = Bigint.shift_right (Dtype.Value.to_z v) (32 * w) in
                 Some
                   (const ~dtype:dt (Dtype.truncate dt (`Int n) :> Dtype.const))
             | _ -> None);
         rule_ctx
           (Upat.op Op.Cast ~dtype:longs ~src:[ Upat.var "a" ] ~name:"x")
           (fun ctx m ->
             let x = m "x" in
             tagged x (w (split_l2i ctx Op.Cast (dtype x) [ m "a" ])));
         rule_ctx
           (Upat.op Op.Cast ~src:[ long "a" ] ~name:"x")
           (fun ctx m ->
             let x = m "x" and a = m "a" in
             if is_long (dtype x) || Option.is_some (tag a) then None
             else
               Some
                 (List.hd
                    (split_l2i ctx Op.Cast (dtype x)
                       (lo_hi (l2i_dt (dtype a)) a))));
         rule_ctx
           (Upat.v
              ~op:(ops [ Op.Shl; Op.Shr ])
              ~dtype:longs
              ~src:[ Upat.var "a"; Upat.var "b" ]
              ~name:"x" ())
           (fun ctx m ->
             let x = m "x" in
             let dt = l2i_dt (dtype x) in
             tagged x
               (w
                  (split_l2i ctx (op x) dt
                     (lo_hi dt (m "a") @ [ rtag ~tag:(word_tag 0 dt) (m "b") ]))));
         rule_ctx
           (Upat.op Op.Where ~dtype:longs
              ~src:[ Upat.var "c"; Upat.var "a"; Upat.var "b" ]
              ~name:"x")
           (fun ctx m ->
             let x = m "x" in
             let dt = l2i_dt (dtype x) in
             tagged x
               (w
                  (split_l2i ctx Op.Where dt
                     ((m "c" :: lo_hi dt (m "a")) @ lo_hi dt (m "b")))));
         rule_ctx
           (Upat.v
              ~op:
                (Op.Set.union
                   (Op.Set.diff Op.Set.alu
                      (Op.Set.union Op.Set.comparison
                         (ops [ Op.Shl; Op.Shr; Op.Where ])))
                   (ops [ Op.Bitcast ]))
              ~dtype:longs ~name:"x" ())
           (fun ctx m ->
             let x = m "x" in
             let dt = l2i_dt (dtype x) in
             tagged x
               (w
                  (split_l2i ctx (op x) dt (List.concat_map (lo_hi dt) (src x)))));
         rule_ctx
           (Upat.op Op.Load ~dtype:longs ~src:[ Upat.var "idx" ] ~name:"x")
           (fun ctx m ->
             let x = m "x" in
             tagged x (fun w ->
                 let idx =
                   graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx (m "idx")
                     (Before_sources (Lazy.force pm_long_decomp))
                 in
                 load (replace (reindex ~mul:2 idx w) ~tag:None) []));
       ]))

(* Forced as the module initialises, on one domain: the compilers' domains
   would race to force it first, and a lazy value that two domains force at
   once raises. *)
let pm_long_decomp = Lazy.force pm_long_decomp

(* Floats *)

(* The operations that move values of an emulated float without computing:
   a lane, a stack and a selection. *)
let moves = ops [ Op.Stack; Op.Index; Op.Where ]

(* The unsigned integer of a float's width, which holds its bits. *)
let f2f_dt dt =
  match Dtype.bitsize dt with
  | 8 -> Dtype.Uint8
  | 16 -> Uint16
  | 32 -> Uint32
  | 64 -> Uint64
  | _ ->
      invalid_arg
        (Format.asprintf "%a has no unsigned integer of its width" Dtype.pp dt)

(* [v >> s], rounded to nearest, ties to even. *)
let rne v s =
  let low = int ((1 lsl (s - 1)) - 1)
  and q = shr v s
  and half = shr v (s - 1) in
  O.(q + (half land int 1 land ((v land low <> int 0) lor (q land int 1))))

let f2f_clamp ?(sat = true) value dt =
  let e, m = Dtype.finfo dt in
  let max_exp, max_man =
    if List.mem dt Dtype.fp8_fnuz then ((1 lsl e) - 1, (1 lsl m) - 1)
    else if Dtype.equal dt Fp8e4m3 then ((1 lsl e) - 1, (1 lsl m) - 2)
    else ((1 lsl e) - 2, (1 lsl m) - 1)
  in
  (* The greatest finite value, and the least magnitude that rounds past it:
     half an ulp above. *)
  let limit extra =
    const_like value
      (`Float
         (Float.pow 2.
            (float_of_int (max_exp - Transcendental.exponent_bias dt))
         *. (1. +. ((float_of_int max_man +. extra) /. float_of_int (1 lsl m)))
         ))
  in
  let clamped =
    if List.mem dt Dtype.fp8s && sat then
      (* A finite value saturates; an infinity stays one, and becomes the
         format's NaN if it has no infinity. *)
      let mx = limit 0. and inf = const_like value (`Float Float.infinity) in
      where
        O.(eq value inf lor eq value (neg inf))
        value
        (where O.(value < neg mx) (neg mx) (where O.(mx < value) mx value))
    else
      let lim = limit 0.5 and inf = const_like value (`Float Float.infinity) in
      where O.(neg lim < value) (where O.(value < lim) value inf) (neg inf)
  in
  where O.(value <> value) value clamped

(* [x] as a float32 rounded to odd: towards zero, with the last bit set if bits
   were dropped. A narrow float rounded to nearest from it is [x] rounded to
   nearest once. Only a source more precise than a float32 needs it; an integer
   that rounds to the power of two past its type's range cannot be cast back,
   and rounded up. *)
let narrow x to_ =
  let src = dtype x in
  if
    not
      (Dtype.equal to_ Float32
      && List.mem src Dtype.[ Float64; Int32; Uint32; Int64; Uint64 ])
  then cast x to_
  else
    let y = cast x Float32 in
    let or_top, back =
      if Dtype.is_float src then (Fun.id, cast y src)
      else
        let k = Dtype.bitsize src - if Dtype.is_unsigned src then 0 else 1 in
        let edge = Float.ldexp 1. k in
        let top = O.(float edge <= y) in
        let below = float (edge -. Float.ldexp 1. (k - 24)) in
        ((fun c -> O.(top lor c)), cast (where top below y) src)
    in
    let zero = const_like x (`Int Bigint.zero) in
    let away =
      or_top O.((x < back) land (zero < x) lor ((back < x) land (x < zero)))
    in
    let bits = bitcast y Uint32 in
    let truncated = where away O.(bits - int 1) bits in
    bitcast O.(truncated lor cast (or_top (back <> x)) Uint32) Float32

let rec f2f ?(sat = true) v fr to_ =
  let is_narrow dt = List.mem dt Dtype.(Float16 :: Bfloat16 :: fp8s) in
  if
    not
      ((Dtype.equal fr Float32 && is_narrow to_)
      || (Dtype.equal to_ Float32 && is_narrow fr))
  then
    invalid_arg
      (Format.asprintf "unsupported decomp %a -> %a" Dtype.pp fr Dtype.pp to_);
  let fs = Dtype.bitsize fr and fb = Transcendental.exponent_bias fr in
  let fe, fm = Dtype.finfo fr in
  let ts = Dtype.bitsize to_ and tb = Transcendental.exponent_bias to_ in
  let te, tm = Dtype.finfo to_ in
  let ones n = int ((1 lsl n) - 1) and bit n = int (1 lsl n) in
  let tdt = f2f_dt to_ in
  let sign_bit = bit (fs - 1) and magnitude = ones (fs - 1) in
  let exp_ones = int (((1 lsl te) - 1) lsl tm) in
  let rebias = int ((tb - fb) lsl tm) and unbias = int ((fb - tb) lsl tm) in
  if Dtype.equal to_ Float32 then begin
    let sign = shl (cast O.(v land sign_bit) tdt) (ts - fs) in
    let nosign = cast O.(v land magnitude) tdt in
    let exp = shr nosign fm and widened = shl nosign (tm - fm) in
    let norm = O.(widened + rebias) and nan = O.(widened lor exp_ones) in
    (* A subnormal of a format with float32's exponent keeps its bits; any other
       is mantissa * 2^(1 - bias - m), a float32 normal, built from the mantissa
       converted exactly. *)
    let subnormal =
      if fb = tb then widened
      else
        let exponent_shift = int ((fb + fm - 1) lsl tm) in
        let scaled = O.(bitcast (cast nosign Float32) tdt - exponent_shift) in
        where (eq nosign (int 0)) (int 0) scaled
    in
    let finite = where (eq exp (int 0)) subnormal norm in
    if List.mem fr Dtype.fp8_fnuz then
      let fnuz_nan = O.((sign <> int 0) land eq nosign (int 0)) in
      let qnan = int ((((1 lsl te) - 1) lsl tm) lor (1 lsl (tm - 1))) in
      bitcast (where fnuz_nan qnan O.(sign lor finite)) to_
    else
      (* e4m3 has one NaN. *)
      let is_nan =
        if Dtype.equal fr Fp8e4m3 then eq nosign (ones (fm + fe))
        else eq exp (ones fe)
      in
      bitcast O.(sign lor where is_nan nan finite) to_
  end
  else begin
    let v = bitcast (f2f_clamp ~sat (bitcast v fr) to_) (f2f_dt fr) in
    let to_sign = bit (ts - 1) and dropped = fm - tm and shift = fs - ts in
    let sign = cast O.(shr v shift land to_sign) tdt in
    let nosign = O.(v land magnitude) in
    let norm = cast O.(rne nosign dropped - unbias) tdt in
    let exp = O.(shr v fm land ones fe) in
    (* A NaN keeps the top of its payload and becomes quiet, as converting it
       does; an infinity keeps its zero mantissa. *)
    let nan_mantissa =
      if Dtype.equal to_ Fp8e4m3 then ones tm
      else
        let quiet = where O.(v land ones fm <> int 0) (bit (tm - 1)) (int 0) in
        O.(shr nosign dropped land ones tm lor quiet)
    in
    let nan = cast O.(sign lor nan_mantissa lor exp_ones) tdt in
    let is_nan = eq exp (ones fe) in
    (* Below [to_]'s least exponent, the significand shifted right by [k] and
       rounded to nearest even is the subnormal's mantissa; one that rounds up
       to the least normal is its code too. A format with float32's exponent
       rounds its subnormals as its normals. *)
    let finite =
      if fb = tb then norm
      else
        let least = int (1 + fb - tb) and far = int (fm + 2) in
        let first = int (1 + fb - tb + dropped) and implicit = bit fm in
        let sig_ = O.(v land ones fm lor implicit) in
        (* The shift, from 1 to [far], also where the value is normal and the
           result goes unused, so that no count reaches the width. *)
        let underflow = O.(exp < least) and k = O.(first - exp) in
        let k = where underflow (where O.(k < far) k far) (int 1) in
        let q = O.(sig_ lsr k)
        and half = O.((sig_ lsr (k - int 1)) land int 1) in
        let sticky = O.(sig_ land ((int 1 lsl (k - int 1)) - int 1) <> int 0) in
        let sub = cast O.(q + (half land (sticky lor (q land int 1)))) tdt in
        where underflow sub norm
    in
    if List.mem to_ Dtype.fp8_fnuz then
      (* An fnuz format has no negative zero: its code is the NaN. *)
      where is_nan to_sign
        (where (eq finite (int 0)) (int 0) O.(sign lor finite))
    else where is_nan nan O.(sign lor finite)
  end

(* Emulating [fr] as [to_]. *)
and f2f_load x fr to_ =
  let storage_idx = f2f_rewrite (fr, to_) (nth x 0) in
  let rest = List.tl (src x) in
  match max_numel x with
  | 1 -> f2f (load storage_idx rest) fr to_
  | n ->
      v Op.Stack
        ~src:
          (List.init n (fun i -> f2f (load (reindex storage_idx i) rest) fr to_))

and f2f_store st idx value fr to_ =
  let tdt = f2f_dt to_ in
  match max_numel value with
  | 1 -> replace st ~src:[ idx; f2f (bitcast value tdt) to_ fr ]
  | n ->
      group
        (List.init n (fun i ->
             replace st
               ~src:
                 [
                   reindex idx i;
                   f2f (bitcast (index value [ int i ]) tdt) to_ fr;
                 ]))

(* The bits that [x], a value of the emulated float, moves from storage
   without arithmetic: a load, a constant, a selection between such values,
   and stacks and lanes of them. A move keeps every code, a signalling NaN's
   included, where converting through the emulating float would quiet it
  . *)
and moved ((fr, _) as ctx) x =
  let all xs =
    List.fold_right
      (fun x acc ->
        match (x, acc) with Some x, Some xs -> Some (x :: xs) | _ -> None)
      xs (Some [])
  in
  let tdt = f2f_dt fr in
  match (op x, src x) with
  | (Op.Const | Op.Cast), _
    when Dtype.equal (dtype x) fr || Dtype.equal (dtype x) Dtype.Weak_float -> (
      (* A constant the float holds exactly, as its bits. *)
      match value x with
      | #Dtype.value as c when Dtype.equal_const (Dtype.const fr c) c ->
          Some (const ~dtype:tdt (Dtype.bitcast fr tdt c :> Dtype.const))
      | _ -> None
      | exception Invalid_argument _ -> None)
  | _ when not (Dtype.equal (dtype x) fr) -> None
  | Op.Load, [ idx ] -> Some (load (f2f_rewrite ctx idx) [])
  | Op.Where, [ c; a; b ] -> (
      match (moved ctx a, moved ctx b) with
      | Some a, Some b -> Some (where c a b)
      | _ -> None)
  | Op.Stack, xs ->
      Option.map (fun xs -> v Op.Stack ~src:xs) (all (List.map (moved ctx) xs))
  | Op.Index, lanes :: at when addrspace x = Some Dtype.Alu ->
      Option.map (fun lanes -> index lanes at) (moved ctx lanes)
  | _ -> None

(* [x], a value of the emulating float, rounded to the emulated float [fr]: its
   bits encoded as [fr]'s, decoded back. Every emulated node holds a value of
   [fr], so a cast or an operation rounds where nx rounds it. *)
and rounded (fr, to_) x =
  f2f (f2f (bitcast x (f2f_dt to_)) to_ fr) fr to_

and f2f_rewrite ctx x =
  graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx x
    (Before_sources (Lazy.force pm_float_decomp))

and pm_float_decomp =
  lazy
    (let emulated (fr, _) x = Dtype.equal (dtype x) fr in
     let tagged (fr, _) x =
       match tag x with Some (Dtype d) -> Dtype.equal d fr | _ -> false
     in
     let floats = Dtype.floats in
     Pattern_matcher.v
       (fun () -> [
         rule_ctx (Upat.v ~op:Op.Set.defines ~name:"x" ())
           (fun ((fr, _) as ctx) m ->
             let x = m "x" in
             match arg x with
             | Param p when emulated ctx x ->
                 Some
                   (v (op x) ~src:(src x)
                      ~arg:(Param { p with dtype = f2f_dt fr })
                      ~tag:(Dtype fr))
             | _ -> None);
         (* An index into a load or a stack selects a lane of a value already
            converted, which the load rules below own. *)
         rule_ctx
           (Upat.v
              ~op:(ops [ Op.Index; Op.Shrink ])
              ~allow_any_len:true
              ~src:
                [
                  Upat.v
                    ~op:(Op.Set.diff Op.Set.all (ops [ Op.Load; Op.Stack ]))
                    ();
                ]
              ~name:"x" ())
           (fun ((fr, _) as ctx) m ->
             let x = m "x" in
             if not (emulated ctx x) then None
             else
               Some
                 (v (op x)
                    ~src:(f2f_rewrite ctx (nth x 0) :: List.tl (src x))
                    ~arg:(arg x) ~tag:(Dtype fr)));
         rule_ctx (Upat.op Op.Load ~dtype:floats ~name:"x")
           (fun ((fr, to_) as ctx) m ->
             let x = m "x" in
             if emulated ctx x then Some (f2f_load x fr to_) else None);
         (* A bitcast of a load loads the bits. *)
         rule_ctx
           (Upat.op Op.Bitcast ~src:[ Upat.op Op.Load ~name:"ld" ] ~name:"bc")
           (fun ctx m ->
             let ld = m "ld" in
             if not (emulated ctx ld) then None
             else
               Some
                 (bitcast
                    (load (f2f_rewrite ctx (nth ld 0)) (List.tl (src ld)))
                    (dtype (m "bc"))));
         (* A bitcast from the emulating float. *)
         rule_ctx
           (Upat.op Op.Bitcast ~src:[ Upat.var ~dtype:floats "x" ] ~name:"bc")
           (fun (fr, to_) m ->
             let x = m "x" and bc = m "bc" in
             if
               Dtype.equal (dtype x) to_
               && Dtype.bitsize (dtype bc) = Dtype.bitsize fr
             then Some (replace bc ~src:[ f2f (bitcast x (f2f_dt to_)) to_ fr ])
             else None);
         (* A bitcast to the emulated float. *)
         rule_ctx
           (Upat.op Op.Bitcast ~src:[ Upat.var "x" ] ~name:"bc")
           (fun (fr, to_) m ->
             if Dtype.equal (dtype (m "bc")) fr then
               Some (f2f (bitcast (m "x") (f2f_dt fr)) fr to_)
             else None);
         rule_ctx
           (Upat.op Op.Cast ~dtype:floats ~src:[ Upat.var "val" ] ~name:"x")
           (fun ((fr, to_) as ctx) m ->
             if emulated ctx (m "x") then
               Some (rounded ctx (f2f_clamp (narrow (m "val") to_) fr))
             else None);
         rule_ctx
           (Upat.v
              ~op:(Op.Set.union Op.Set.alu (ops [ Op.Stack; Op.Index ]))
              ~dtype:floats ~name:"x" ())
           (fun ((fr, to_) as ctx) m ->
             let x = m "x" in
             if not (emulated ctx x) then None
             else
               let y =
                 v (op x)
                   ~src:
                     (List.map
                        (fun s ->
                          if Dtype.equal (dtype s) fr then cast s to_ else s)
                        (src x))
                   ~arg:(arg x) ?tag:(tag x)
               in
               (* A lane, a stack or a selection moves values already of
                  [fr]; arithmetic rounds its result. *)
               Some
                 (if Op.Set.mem (op x) moves then y else rounded ctx y));
         (* A store of a move stores the bits moved. *)
         rule_ctx
           (Upat.v ~op:(ops [ Op.Store ]) ~allow_any_len:true
              ~src:[ Upat.var "idx"; Upat.var "val" ]
              ~name:"st" ())
           (fun ((fr, _) as ctx) m ->
             let st = m "st" and value = m "val" in
             if not (Dtype.equal (dtype value) fr) then None
             else
               Option.map
                 (fun bits ->
                   replace st
                     ~src:
                       (f2f_rewrite ctx (m "idx")
                       :: bits
                       :: List.tl (List.tl (src st))))
                 (moved ctx value));
         rule_ctx
           (Upat.op Op.Store ~name:"st"
              ~src:
                [ Upat.var "idx"; Upat.op Op.Bitcast ~dtype:floats ~name:"val" ])
           (fun ((fr, _) as ctx) m ->
             let idx = m "idx" and value = m "val" in
             if emulated ctx value && tagged ctx idx then
               Some
                 (replace (m "st")
                    ~src:[ idx; bitcast (nth value 0) (f2f_dt fr) ])
             else None);
         rule_ctx
           (Upat.op Op.Store ~name:"st"
              ~src:
                [
                  Upat.or_casted (Upat.var "idx"); Upat.var ~dtype:floats "val";
                ])
           (fun ((fr, to_) as ctx) m ->
             let idx = m "idx" and value = m "val" in
             if Dtype.equal (dtype value) to_ && tagged ctx idx then
               Some (f2f_store (m "st") idx value fr to_)
             else None);
       ]))

(* Forced as the module initialises, on one domain: the compilers' domains
   would race to force it first, and a lazy value that two domains force at
   once raises. *)
let pm_float_decomp = Lazy.force pm_float_decomp

(* Passes *)

let emulable = Dtype.(fp8s @ [ Bfloat16; Float16; Int64; Uint64 ])

let computes r =
  let among l dt = List.exists (Dtype.equal dt) l in
  let supported = Renderer.supported_dtypes r in
  List.filter (fun dt -> among supported dt || among emulable dt) Dtype.all

let emulates r =
  let named =
    List.filter_map
      (fun s -> Result.to_option (Dtype.of_string s))
      (Setting.value Setting.emulated_dtypes)
  in
  let supported = Renderer.supported_dtypes r in
  (* Unsigned 64-bit integers are emulated with signed ones. *)
  fun dt ->
    let dt = if Dtype.equal dt Uint64 then Dtype.Int64 else dt in
    List.mem dt named || not (List.mem dt supported)

type ctx = { mutable found : Dtype.t list; renderer : Renderer.t }

let ctx renderer = { found = []; renderer }

let do_dtype_decomps ctx sink =
  let sink =
    List.fold_left
      (fun sink fr ->
        let to_ = if Dtype.equal fr Int64 then Dtype.Int32 else Float32 in
        if Setting.value Setting.debug >= 2 then
          Format.eprintf "emulating %a as %a@." Dtype.pp fr Dtype.pp to_;
        if List.mem fr Dtype.floats then
          graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:(fr, to_) sink
            (Before_sources pm_float_decomp)
        else
          graph_rewrite ~calls:Skip ~pass:Fixed_point ~ctx:(Splits.create 64)
            sink (Before_sources pm_long_decomp))
      sink
      (List.sort Dtype.compare (List.filter (emulates ctx.renderer) ctx.found))
  in
  ctx.found <- [];
  sink

let pm_dtype_decomps =
  Pattern_matcher.v
    (fun () -> [
      (* Find the types to decompose. *)
      rule_ctx
        (Upat.v ~op:Op.Set.all ~dtype:emulable ~name:"x" ())
        (fun ctx m ->
          let dt =
            match dtype (m "x") with Uint64 -> Dtype.Int64 | dt -> dt
          in
          if not (List.mem dt ctx.found) then ctx.found <- dt :: ctx.found;
          None);
      rule_ctx (Upat.op Op.Sink ~name:"sink") (fun ctx m ->
          Some (do_dtype_decomps ctx (m "sink")));
    ])
