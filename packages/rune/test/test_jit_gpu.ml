(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.jit on this machine's first GPU: a scatter and an assembly of any
   pieces give the host's eager values, a scatter keeps the bits of each
   element no update lands on, a Set scatter of rows gives eager's bits and
   a numbering of repeated keys eager's counts, a cast between float32 and a
   16-bit float gives eager's bits at every code and tie, as does a cast of
   an integer to a float at the ties of every exponent, a product of narrow
   floats or of MXFP4 weights holds its bound, a narrow float's fma rounds once, a call's check raises at the next wait, so the
   call returns while its work runs, a key split there has the host's words,
   and a program over the GPU and a software device, devices of two targets,
   compiles to eager's values with replicas equal bit for bit, a running
   operation of a prefix is the prefix of its result, attention holds its
   bound against a float64 reference and gives +0 for a sum of zeros, paged
   keys and values give dense ones' bits, a sum
   of a contraction's result holds its bound, exp
   holds its bound in ulps of the correctly rounded result, and a range
   gather and its gradient compile to eager's bits, and an argmax or argmin
   over a length 16 does not divide gives nx's position, a sum and a product
   their exact values, and a first k of a sort gives eager's values and
   positions bit for bit. It skips without a GPU,
   or without the compiler the GPU's kernels need. *)

open Windtrap
open Rune_support
open Witness

let gpu =
  lazy
    (match Rig_gpus.first () with
    | Error _ -> None
    | Ok d -> Some (d, Nx.devices [ d ]))

(* A compile error that names a header the machine lacks is the machine's: the
   test skips. Half floats' kernels read headers no other kernel does. *)
let missing m =
  let has s =
    let n = String.length s in
    let rec at i =
      i + n <= String.length m && (String.sub m i n = s || at (i + 1))
    in
    at 0
  in
  if has "cannot open source file" then
    Some ("a header is missing, set CUDA_PATH: " ^ m)
  else None

let compiles (module S : Nx.Devices) =
  let neg (Nx.Dtype.Any dt) =
    let x = Nx.cast dt (Nx.create Nx.float32 [| 2 |] [| 1.; 2. |]) in
    ignore (Rune.jit' Nx.neg (Nx.place S.on x))
  in
  List.iter
    (fun d ->
      match neg d with
      | () -> ()
      | exception (Tolk.Renderer.Compiler.Compile_error m as e) -> (
          match missing m with
          | Some why -> skip ~reason:why ()
          | None -> raise e))
    Nx.Dtype.[ Any Nx.float32; Any Nx.float16; Any Nx.bfloat16 ]

(* The first GPU, skipping where its renderer cannot be made: without its
   compiler's library, each compiled call raises. *)
let on_gpu () =
  match Lazy.force gpu with
  | None -> skip ~reason:"no GPU" ()
  | Some (d, s) -> (
      match Tolk_engine.renderer d with
      | exception Failure why -> skip ~reason:why ()
      | _ ->
          compiles s;
          (d, s))

(* Whether the kernels of [d] have float64, which Metal's lack. *)
let has_float64 d =
  List.exists (Tolk.Dtype.equal Tolk.Dtype.Float64)
    (Tolk.Renderer.supported_dtypes (Tolk_engine.renderer d))

(* Whether [d] flushes float32's subnormals to zero, as rune.mli's Numerics
   says Metal does. *)
let flushes d =
  String.equal (Tolk_engine.target d).Tolk.Helpers.Target.device "METAL"

(* The scatter law on the GPU. A float64 case is discarded where the GPU's
   kernels lack float64, since no compiled call computes it there. *)
let law c =
  let d, s = on_gpu () in
  let (Nx.Dtype.Any dt) = c.Scatters.dtype in
  if Nx.Dtype.equal dt Nx.float64 then assume (has_float64 d);
  Scatters.law ~flushes:(flushes d) s c

(* The integer cast law on the GPU, a case into float64 or complex128
   discarded where its kernels lack float64. *)
let int_casts c =
  let d, s = on_gpu () in
  if Int_casts.dst64 c then assume (has_float64 d);
  Int_casts.law s c

let unreached c = Unreached.law (snd (on_gpu ())) c

(* A numbering of 4096 keys of 5 values, compiled on the GPU, gives eager's
   ids, first rows, counts and count: each count adds about 800 ones at one
   entry. *)
let repeated_keys () =
  let _, (module S : Nx.Devices) = on_gpu () in
  let key i = Int32.of_int (i.(0) * 7 mod 5) in
  let x = Nx.init Nx.int32 [| 4096 |] key in
  let f x =
    let g = Nx.unique ~size:8 x in
    Nx.concatenate ~axis:0
      [ g.ids; g.first; g.counts; Nx.reshape [| 1 |] g.count ]
  in
  let y = Nx.place Nx.Host.on (Rune.jit' f (Nx.place S.on x)) in
  equal codes_t (codes (f x)) (codes y)

(* The law of Set scatters of rows on the GPU, a float64 case discarded where
   its kernels lack float64. *)
let row_set c =
  let d, s = on_gpu () in
  assume (c.Row_sets.dtype <> Nx.Dtype.Any Nx.float64 || has_float64 d);
  Row_sets.law (Tolk_engine.renderer d) s c
let casts16 dt () = Casts16.law (snd (on_gpu ())) dt

(* The attention law on the GPU, a float64 case discarded where its kernels
   lack float64. *)
let attention c =
  let d, s = on_gpu () in
  if c.Attentions.dtype = Nx.Dtype.Any Nx.float64 then assume (has_float64 d);
  Attentions.law s c

(* The law of paged attention against dense on the GPU. *)
let paged_attention c = Paged_attentions.law (snd (on_gpu ())) c

(* The gradient law on the GPU, compiled or mapped by vmap. *)
let attention_grad ~vmap c =
  let d, s = on_gpu () in
  if c.Attentions.dtype = Nx.Dtype.Any Nx.float64 then assume (has_float64 d);
  Attentions.grad_law s ~vmap c

(* A sum of a weighted contraction's result, with products past those a
   compiled reduction fuses over, holds the contraction's and the sum's bound
   [γ(n) Σ |w| |a| |b|] against a float64 reference, for the [n] floating
   additions and products a term takes: fewer than 65600, so [4e-3]. *)
let sum_of_contraction () =
  let _, (module S : Nx.Devices) = on_gpu () in
  let draw i s = Nx.Rng.normal ~key:(Nx.Rng.key i) Nx.float32 s in
  let a = draw 1 [| 256; 64 |] and b = draw 2 [| 64; 256 |] in
  let w = draw 3 [| 256; 256 |] in
  let p = Nx.Pattern.v "i j, j k -> i k | j" in
  let f (type s d) (acc : (float, s) Nx.dtype) (w : (float, s, d) Nx.t) a b =
    Nx.sum (Nx.mul w (Nx.contract acc p a b))
  in
  let compiled =
    Rune.jit
      Nx.Ptree.(tensor @-> tensor @-> tensor @-> returns tensor)
      (f Nx.float32)
      (Nx.place S.on w) (Nx.place S.on a) (Nx.place S.on b)
  in
  let f64 x = Nx.cast Nx.float64 x in
  let exact = Nx.item [] (f Nx.float64 (f64 w) (f64 a) (f64 b)) in
  let mass =
    Nx.item []
      (Nx.sum
         (Nx.mul (Nx.abs (f64 w))
            (Nx.contract Nx.float64 p (Nx.abs (f64 a)) (Nx.abs (f64 b)))))
  in
  at_most ~msg:"the sum" float_exact ~than:(4e-3 *. mass)
    (Float.abs (Nx.item [] (Nx.place Nx.Host.on compiled) -. exact))

(* A sum over a broadcast of a contraction's result holds the result, which
   tolk stores before the broadcast, and never the broadcast: 256 KiB against
   64 MiB here, which at a user's sizes would not fit the device. The bound
   leaves room for the kernels and the arguments' copies. *)
let sum_of_broadcast () =
  let gpu, (module S : Nx.Devices) = on_gpu () in
  let draw i s = Nx.place S.on (Nx.Rng.normal ~key:(Nx.Rng.key i) Nx.float32 s) in
  let a = draw 1 [| 256; 64 |] and b = draw 2 [| 64; 256 |] in
  let p = Nx.Pattern.v "i j, j k -> i k | j" in
  let f =
    Rune.jit
      Nx.Ptree.(tensor @-> tensor @-> returns tensor)
      (fun a b ->
        Nx.sum (Nx.broadcast_to [| 256; 256; 256 |] (Nx.contract Nx.float32 p a b)))
  in
  let before = Rig.allocated gpu in
  Nx.sync (f a b);
  at_most ~msg:"the bytes the call holds" Testable.int ~than:(16 * 1024 * 1024)
    (Rig.allocated gpu - before)

(* Forward mode of a loss over attention, whose tangent sums the pair loop's
   contractions (8 heads, 64 rows, 128 keys of 64 dimensions) and is summed
   again: compiled in float32 against the host's float64 within [1e-2]
   relative, set by hand as rounding dominates. *)
let attention_jvp () =
  let _, (module S : Nx.Devices) = on_gpu () in
  let draw i s = Nx.Rng.normal ~key:(Nx.Rng.key i) Nx.float32 s in
  let q = draw 1 [| 1; 8; 64; 64 |] and w = draw 2 [| 1; 8; 64; 64 |] in
  let k = draw 3 [| 1; 8; 128; 64 |] and v = draw 4 [| 1; 8; 128; 64 |] in
  let tq = draw 5 (Nx.shape q) and tk = draw 6 (Nx.shape k) in
  let tv = draw 7 (Nx.shape v) in
  let jvp (type a d) (acc : (float, a) Nx.dtype) (w : (float, a, d) Nx.t) xs ts =
    let s = Nx.Ptree.(list tensor) in
    let loss = function
      | [ q; k; v ] ->
          let o, _ = Nx.attention ~acc ~kv:(Nx.Dense { k; v }) q in
          Nx.sum (Nx.mul w o)
      | _ -> invalid_arg "loss"
    in
    snd (Rune.jvp s Nx.Ptree.tensor loss xs ts)
  in
  let on x = Nx.place S.on x and f64 x = Nx.cast Nx.float64 x in
  let compiled =
    Rune.jit
      Nx.Ptree.(list tensor @-> list tensor @-> returns tensor)
      (jvp Nx.float32 (on w))
      [ on q; on k; on v ] [ on tq; on tk; on tv ]
  in
  let exact =
    Nx.item []
      (jvp Nx.float64 (f64 w) [ f64 q; f64 k; f64 v ] [ f64 tq; f64 tk; f64 tv ])
  in
  at_most ~msg:"the tangent" float_exact
    ~than:(1e-2 *. (1. +. Float.abs exact))
    (Float.abs (Nx.item [] (Nx.place Nx.Host.on compiled) -. exact))

(* A program over [x] and a value [y], each [256 x 256] float32. *)
type over = {
  f :
    'd.
    (float, Nx.float32_elt, 'd) Nx.t ->
    (float, Nx.float32_elt, 'd) Nx.t ->
    (float, Nx.float32_elt, 'd) Nx.t;
}

let norm h =
  let ms = Nx.mean ~axes:[ 1 ] ~keepdims:true (Nx.mul h h) in
  Nx.div h (Nx.sqrt (Nx.add_s ms 1e-5))

let close h = Nx.place (Nx.Placement.closed (Nx.placement h)) h

(* Programs whose reductions read [y], with its readers: a weighted sum, a sum
   of two maps of [y], a sum of its squares, a norm of a residual that the
   program reads again, as a transformer's block does, closed or not, and a
   softmax. *)
let reading_programs =
  [
    ("a weighted sum", { f = (fun x y -> Nx.sum (Nx.mul x y)) });
    ( "a sum of two maps",
      { f = (fun _ y -> Nx.sum (Nx.add (Nx.mul_s y 2.) (Nx.mul_s y 3.))) } );
    ("a sum of squares", { f = (fun _ y -> Nx.sum (Nx.mul y y)) });
    ( "a norm of a residual",
      { f = (fun x y -> let h = Nx.add x y in Nx.add (norm h) h) } );
    ( "a norm of a closed residual",
      { f = (fun x y -> let h = close (Nx.add x y) in Nx.add (norm h) h) } );
    ( "a softmax",
      {
        f =
          (fun _ y ->
            let e = Nx.exp (Nx.sub y (Nx.max ~axes:[ 1 ] ~keepdims:true y)) in
            Nx.div e (Nx.sum ~axes:[ 1 ] ~keepdims:true e));
      } );
  ]

(* [y] as a reduction of at least [2{^21}] terms, more than a compiled
   reduction runs inside the loop of another, from operands of these shapes:
   a contraction of [256 x 64 x 256] products, or a sum of [256 x 256 x 64]
   elements along the last axis. *)
type reduction = {
  operands : int array list;
  reduce :
    'd.
    (float, Nx.float32_elt, 'd) Nx.t list -> (float, Nx.float32_elt, 'd) Nx.t;
}

let reductions =
  [
    ( "a contraction",
      {
        operands = [ [| 256; 64 |]; [| 64; 256 |] ];
        reduce =
          (function
          | [ a; b ] ->
              Nx.contract Nx.float32 (Nx.Pattern.v "i j, j k -> i k | j") a b
          | _ -> invalid_arg "a contraction");
      } );
    ( "a sum",
      {
        operands = [ [| 256; 256; 64 |] ];
        reduce =
          (function
          | [ z ] -> Nx.sum ~axes:[ 2 ] z | _ -> invalid_arg "a sum");
      } );
  ]

(* A reduction that a program's reductions read adds one kernel to the
   program, its own: the program compiles to the kernels it compiles over the
   reduction's result passed as an argument, and one more. Fewer would run the
   reduction's terms inside another's loop; more would store its result, or a
   map of it, twice. *)
let one_kernel { operands; reduce } { f } =
  let _, (module S : Nx.Devices) = on_gpu () in
  let draw i s =
    Nx.place S.on (Nx.Rng.normal ~key:(Nx.Rng.key i) Nx.float32 s)
  in
  let x = draw 1 [| 256; 256 |] in
  let zs = List.mapi (fun i s -> draw (i + 2) s) operands in
  let zs_t = Nx.Ptree.(list tensor) in
  let y = Rune.jit Nx.Ptree.(zs_t @-> returns tensor) reduce zs in
  let kernels g =
    List.length (snd (Tolk.Realize.kernels (fun () -> Nx.sync (g ()))))
  in
  let given =
    kernels (fun () ->
        Rune.jit Nx.Ptree.(tensor @-> tensor @-> returns tensor) f x y)
  in
  let reduced =
    kernels (fun () ->
        Rune.jit
          Nx.Ptree.(tensor @-> zs_t @-> returns tensor)
          (fun x zs -> f x (reduce zs))
          x zs)
  in
  equal ~msg:"kernels" Testable.int (given + 1) reduced

(* A reduction that a program returns and another reduction reads is stored
   once, into the result: the program compiles to the kernels of the reading
   reduction alone, and returns the reduction and its sum within [1e-5] of
   each compiled apart, relative to their magnitudes. *)
let returned { operands; reduce } () =
  let _, (module S : Nx.Devices) = on_gpu () in
  let draw i s =
    Nx.place S.on (Nx.Rng.normal ~key:(Nx.Rng.key i) Nx.float32 s)
  in
  let zs = List.mapi (fun i s -> draw (i + 2) s) operands in
  let zs_t = Nx.Ptree.(list tensor) in
  let kernels g =
    List.length (snd (Tolk.Realize.kernels (fun () -> Nx.sync (g ()))))
  in
  let host x = Nx.place Nx.Host.on x in
  let summed () =
    Rune.jit Nx.Ptree.(zs_t @-> returns tensor) (fun zs -> Nx.sum (reduce zs)) zs
  in
  let both () =
    Rune.jit
      Nx.Ptree.(zs_t @-> returns (pair tensor tensor))
      (fun zs ->
        let y = reduce zs in
        (y, Nx.sum y))
      zs
  in
  equal ~msg:"kernels" Testable.int (kernels summed)
    (kernels (fun () -> snd (both ())));
  let y, total = both () in
  let alone = host (Rune.jit Nx.Ptree.(zs_t @-> returns tensor) reduce zs) in
  let off a b =
    Nx.item [] (Nx.max (Nx.abs (Nx.sub a b)))
    /. Float.max 1. (Nx.item [] (Nx.max (Nx.abs b)))
  in
  at_most ~msg:"the reduction" float_exact ~than:1e-5 (off (host y) alone);
  at_most ~msg:"its sum" float_exact ~than:1e-5
    (off (host total) (host (summed ())))

(* A reduction runs inside the one that reads it below [2{^21}] terms, or of
   at most 32 terms an element, which cost little there: a weighted sum of the
   last axis's sums of [z] compiles to the kernels of the weighted sum over
   those sums passed as an argument. *)
let inside shape () =
  let _, (module S : Nx.Devices) = on_gpu () in
  let draw i s =
    Nx.place S.on (Nx.Rng.normal ~key:(Nx.Rng.key i) Nx.float32 s)
  in
  let z = draw 2 shape in
  let x = draw 1 (Array.sub shape 0 2) in
  let t2 = Nx.Ptree.(tensor @-> tensor @-> returns tensor) in
  let weighted x y = Nx.sum (Nx.mul x y) in
  let y =
    Rune.jit Nx.Ptree.(tensor @-> returns tensor) (Nx.sum ~axes:[ 2 ]) z
  in
  let kernels g =
    List.length (snd (Tolk.Realize.kernels (fun () -> Nx.sync (g ()))))
  in
  let given = kernels (fun () -> Rune.jit t2 weighted x y) in
  let summed =
    kernels (fun () ->
        Rune.jit t2 (fun x z -> weighted x (Nx.sum ~axes:[ 2 ] z)) x z)
  in
  equal ~msg:"kernels" Testable.int given summed

let attention_dtypes =
  Nx.Dtype.[ Any Nx.float16; Any Nx.bfloat16; Any Nx.float32; Any Nx.float64 ]

(* The gradient law's examples: the forward's, and its bfloat16 prefill
   with lse's cotangent alone. *)
let grad_examples =
  let ex = Attentions.examples attention_dtypes in
  ex @ [ { (List.nth ex 1) with seed = 1 } ]

(* A case vmap maps: dense keys, no lengths, two lead indices or more, and
   no mask that captures an array along the lead axes. *)
let mappable (c : Attentions.t) =
  let mask = if c.mask = Spans then Attentions.Positions else c.mask in
  { c with page = None; lengths = None; lead = max 2 c.lead; mask }

(* The prefix law on the GPU, a float64 case discarded where its kernels lack
   float64. *)
let prefixes (Prefixes.Case { dt; _ } as c) =
  let d, s = on_gpu () in
  if Nx.Dtype.equal dt Nx.float64 then assume (has_float64 d);
  Prefixes.law s c

(* The assembly law on the GPU, a float64 or complex128 case discarded where
   its kernels lack float64. *)
let assembly c =
  let d, s = on_gpu () in
  let (Nx.Dtype.Any dt) = Assemblies.dtype c in
  if Nx.Dtype.bits dt = 64 && Nx.Dtype.is Float dt then assume (has_float64 d);
  if Nx.Dtype.equal dt Nx.complex128 then assume (has_float64 d);
  Assemblies.law s c

(* The position of a vector's greatest element, compiled, at lengths 16 does
   not divide: under 32768 one reduction, above it one that scheduling splits
   in two, 32896 into 128 runs of 257 and gpt-oss's 201088 into 128 of 1571.
   Every value is exact in bfloat16, the greatest negative in some cases. The
   position is nx.mli's: the first greatest, the first NaN where there is
   one, [+0.] above [-0.]; argmin's of the negated values is the same. *)
type peak = First | Last | Tie | Nan | Zeros | Flat | Negative | Infinite

let peak_name = function
  | First -> "first"
  | Last -> "last"
  | Tie -> "a tie"
  | Nan -> "NaN"
  | Zeros -> "-0 and +0"
  | Flat -> "all equal"
  | Negative -> "negative"
  | Infinite -> "all -inf"

type extreme = Extreme : int * (float, 's) Nx.dtype * peak -> extreme

let extremes =
  List.concat_map
    (fun n ->
      List.concat_map
        (fun peak ->
          [ Extreme (n, Nx.float32, peak); Extreme (n, Nx.bfloat16, peak) ])
        [ First; Last; Tie; Nan; Zeros; Flat; Negative; Infinite ])
    [ 33; 47; 1571; 4099; 128 * 257; 151936; 201088 ]

(* The values of [peak] at length [n], below 0 but those it sets, and the
   position of their greatest. *)
let peak_values n peak =
  let a =
    Array.init n (fun i -> -.Float.of_int ((i * 31 mod 97) + 1) /. 128.)
  in
  let position =
    match peak with
    | First ->
        a.(0) <- 1.;
        0
    | Last ->
        a.(n - 1) <- 1.;
        n - 1
    | Tie ->
        a.(n / 3) <- 1.;
        a.(n - 1) <- 1.;
        n / 3
    | Nan ->
        a.(n / 2) <- 1.;
        a.(n - 2) <- Float.nan;
        a.(n - 1) <- Float.nan;
        n - 2
    | Zeros ->
        Array.fill a 0 n (-0.);
        a.(n - 1) <- 0.;
        n - 1
    | Flat ->
        Array.fill a 0 n 0.5;
        0
    | Negative ->
        a.(n - 1) <- -1. /. 256.;
        n - 1
    | Infinite ->
        Array.fill a 0 n Float.neg_infinity;
        0
  in
  (a, position)

let argmax_extreme (Extreme (n, dt, peak)) =
  let _, (module S) = on_gpu () in
  let a, position = peak_values n peak in
  let x = Nx.cast dt (Nx.create Nx.float32 [| n |] a) in
  let x = Nx.place S.on x in
  let greatest = Rune.jit' (Nx.argmax ~axis:0) x in
  equal ~msg:"argmax" Testable.int64 (Int64.of_int position)
    (Nx.item [] greatest);
  (* Negated, the least is where the greatest was: [-0.] below [+0.]. *)
  let least = Rune.jit' (fun x -> Nx.argmin ~axis:0 (Nx.neg x)) x in
  equal ~msg:"argmin of the negation" Testable.int64 (Int64.of_int position)
    (Nx.item [] least)

(* A float32 sum and product over the same lengths, every term exact: [n]
   ones sum to [n], and ones with a [-1.] last multiply to [-1.]. *)
let folds n =
  let _, (module S) = on_gpu () in
  let ones = Nx.place S.on (Nx.ones Nx.float32 [| n |]) in
  let signed =
    Nx.place S.on
      (Nx.init Nx.float32 [| n |] (fun i -> if i.(0) = n - 1 then -1. else 1.))
  in
  let sum = Rune.jit' (Nx.sum ~axes:[ 0 ]) ones in
  equal ~msg:"sum" Testable.float_exact (Float.of_int n) (Nx.item [] sum);
  let prod = Rune.jit' (Nx.prod ~axes:[ 0 ]) signed in
  equal ~msg:"product" Testable.float_exact (-1.) (Nx.item [] prod)

(* Top k

   A first [k] of a sort keeps keys of at most 32 bits by one of three paths:
   comparing every pair of keys where the whole array's [k n²] comparisons are
   at most [selection_comparisons], else [k] rounds where [k] is at most the
   network's [s(s+1)/2] compare steps for [n <= 2^s], else the network, as wider
   keys always do. The draws reach each path, at slices of 17 to 40 keys and
   enough rows to pass the selection's bound. *)

type top = {
  t_dt : Nx.Dtype.any;
  t_shape : int array;
  t_axis : int;
  t_k : int;
  t_operand : Operand.t;
}

let pp_top ppf c =
  let (Nx.Dtype.Any dt) = c.t_dt in
  Format.fprintf ppf "%s %a axis %d k %d %a [%a]" (Nx.Dtype.name dt)
    Operand.pp_shape c.t_shape c.t_axis c.t_k Operand.pp_view c.t_operand.view
    (Format.pp_print_seq
       ~pp_sep:(fun ppf () -> Format.pp_print_char ppf ' ')
       Operand.pp_value)
    (Array.to_seq c.t_operand.values)

let narrow_keys =
  Nx.Dtype.
    [
      Any Nx.float32;
      Any Nx.float16;
      Any Nx.bfloat16;
      Any Nx.float8_e4m3fn;
      Any Nx.int8;
      Any Nx.uint16;
      Any Nx.int32;
      Any Nx.uint32;
      Any Nx.bool;
    ]

let wide_keys = Nx.Dtype.[ Any Nx.float64; Any Nx.int64; Any Nx.complex64 ]
let selection_comparisons = 1 lsl 16

let rec bit_length n = if n = 0 then 0 else 1 + bit_length (n lsr 1)

(* The network's compare steps over [n] keys. *)
let steps n =
  let s = bit_length (n - 1) in
  s * (s + 1) / 2

type path = Trivial | Selection | Rounds | Past_steps | Wide

let path c =
  let n = c.t_shape.(c.t_axis) and size = Operand.numel c.t_shape in
  if c.t_k = 0 || n <= 1 || size = 0 then Trivial
  else if List.mem c.t_dt wide_keys then Wide
  else if c.t_k * n * size <= selection_comparisons then Selection
  else if c.t_k <= steps n then Rounds
  else Past_steps

(* [laid ~rows n] is a shape of at least [rows] slices of [n] and the slices'
   axis: the last, the first, or the middle of three. *)
let laid ~rows n =
  let open Gen in
  let* rows = int_range rows (rows + 8) in
  one_of
    [
      constant ([| rows; n |], 1);
      constant ([| n; rows |], 0);
      map (fun a -> ([| a; n; (rows + a - 1) / a |], 1)) (int_range 2 4);
    ]

(* The rows that put a first [k] of [n] past the selection's bound. *)
let past_selection ~k n = (selection_comparisons / (k * n * n)) + 1

let top_gen =
  let open Gen in
  let small =
    let* r = int_range 1 3 in
    let* s =
      array ~size:(constant r)
        (frequency [ (1, constant 0); (2, constant 1); (4, int_range 2 4) ])
    in
    let* axis = int_range 0 (r - 1) in
    let* long = option (int_range 9 17) in
    let s =
      match long with
      | None -> s
      | Some n -> Array.mapi (fun i e -> if i = axis then n else e) s
    in
    let* dt = of_list (narrow_keys @ wide_keys) in
    let+ k = int_range 0 s.(axis) in
    (dt, s, axis, k)
  in
  let rounds =
    let* n = int_range 17 40 in
    let* k = int_range 1 (Int.min n (steps n)) in
    let* dt = of_list narrow_keys in
    let+ s, axis = laid ~rows:(past_selection ~k n) n in
    (dt, s, axis, k)
  in
  let past_steps =
    let* n = int_range 17 32 in
    let* k = int_range (steps n + 1) n in
    let* dt = of_list narrow_keys in
    let+ s, axis = laid ~rows:(past_selection ~k n) n in
    (dt, s, axis, k)
  in
  let wide =
    let* n = int_range 2 17 in
    let* k = int_range 1 n in
    let* dt = of_list wide_keys in
    let+ s, axis = laid ~rows:1 n in
    (dt, s, axis, k)
  in
  let* dt, s, axis, k =
    frequency [ (2, small); (2, rounds); (1, past_steps); (1, wide) ]
  in
  let+ o = Operand.draw s in
  { t_dt = dt; t_shape = s; t_axis = axis; t_k = k; t_operand = o }

(* Plain operands cycling through the pool, whose specials then repeat in a
   slice: each path, gpt-oss's router at one and 512 rows among them, rounds
   down to a slice's last key, and the edges of [k] and of the shape. *)
let top_examples =
  let example d s axis k =
    let values =
      Array.init (Operand.numel s) (fun i ->
          Operand.pool.(i mod Array.length Operand.pool))
    in
    {
      t_dt = d;
      t_shape = s;
      t_axis = axis;
      t_k = k;
      t_operand = { Operand.view = Plain; values };
    }
  in
  Nx.Dtype.
    [
      example (Any Nx.float32) [| 1; 32 |] 1 4;
      example (Any Nx.float32) [| 512; 32 |] 1 4;
      example (Any Nx.bfloat16) [| 4; 20; 60 |] 1 3;
      example (Any Nx.int8) [| 70; 10 |] 1 10;
      example (Any Nx.int32) [| 8; 32 |] 1 16;
      example (Any Nx.float64) [| 3; 9 |] 1 3;
      example (Any Nx.complex64) [| 6; 2 |] 0 2;
      example (Any Nx.float16) [| 3; 5 |] 1 0;
      example (Any Nx.bfloat16) [| 2; 17 |] 1 17;
      example (Any Nx.int8) [| 4; 1 |] 1 1;
      example (Any Nx.float32) [| 0; 5 |] 1 2;
      example (Any Nx.uint32) [| 3; 17; 4 |] 1 5;
    ]

(* [raw x] is [x]'s codes, a float's and a complex number's parts read as
   unsigned integers of their width, so that a NaN's sign and payload count. *)
let raw (type v s) (x : (v, s, Nx.host) Nx.t) =
  let dt = Nx.dtype x in
  let width =
    match Nx.Dtype.kind dt with
    | Complex -> Nx.Dtype.bits dt / 2
    | Float -> Nx.Dtype.bits dt
    | Signed | Unsigned | Boolean -> 0
  in
  match width with
  | 0 -> codes x
  | 8 -> codes (Nx.bitcast Nx.uint8 x)
  | 16 -> codes (Nx.bitcast Nx.uint16 x)
  | 32 -> codes (Nx.bitcast Nx.uint32 x)
  | _ -> codes (Nx.bitcast Nx.uint64 x)

(* Whether two elements of a slice of [x] along [axis] are equal in the sort
   order. A complex number's are not looked for. *)
let ties (type v s) (x : (v, s, Nx.host) Nx.t) axis =
  let s = Nx.shape x in
  let n = s.(axis) in
  let inner = Operand.numel (Array.sub s (axis + 1) (Array.length s - axis - 1)) in
  match Nx.Dtype.kind (Nx.dtype x) with
  | Complex -> false
  | Float | Signed | Unsigned | Boolean ->
      let keys = Nx.to_array (Nx.order_key Nx.uint64 x) in
      let seen = Hashtbl.create 64 in
      let tie = ref false in
      Array.iteri
        (fun j key ->
          let at = (j / (n * inner), j mod inner, key) in
          if Hashtbl.mem seen at then tie := true else Hashtbl.add seen at ())
        keys;
      !tie

(* [specials x] is whether [x] holds a NaN, both zeros, an infinity, and an
   integer's least or greatest value. *)
let specials (type v s) (x : (v, s, Nx.host) Nx.t) =
  let dt = Nx.dtype x in
  let xs = Nx.to_array x in
  let has p = Array.exists p xs in
  match Nx.Dtype.kind dt with
  | Float ->
      let zero negative v = v = 0. && Float.sign_bit v = negative in
      ( has Float.is_nan,
        has (zero true) && has (zero false),
        has (fun v -> Float.abs v = Float.infinity),
        false )
  | Signed | Unsigned ->
      let least = Operand.value dt Operand.Least
      and greatest = Operand.value dt Operand.Greatest in
      (false, false, false, has (fun v -> v = least || v = greatest))
  | Complex | Boolean -> (false, false, false, false)

(* [Nx.top_k] compiled on the GPU gives eager's values and positions, bit for
   bit. A float64 case is discarded where the GPU's kernels lack float64. *)
let top_k c =
  let d, (module S) = on_gpu () in
  let (Nx.Dtype.Any dt) = c.t_dt in
  let path = path c and n = c.t_shape.(c.t_axis) in
  let x = Operand.make dt c.t_shape c.t_operand in
  let nan, zeros, infinite, extreme = specials x in
  cover "a first k by selection" (path = Selection);
  cover "a first k by rounds" (path = Rounds);
  cover "k = n by rounds" (path = Rounds && c.t_k = n);
  cover "a first k past the rounds' steps by the network" (path = Past_steps);
  cover "a first k of 64-bit or complex keys by the network" (path = Wide);
  cover "k = 0" (c.t_k = 0);
  cover "k = n" (c.t_k = n && n > 1);
  cover "n = 1" (n = 1);
  cover "zero rows" (n > 0 && Operand.numel c.t_shape = 0);
  cover "a middle axis"
    (c.t_axis > 0 && c.t_axis < Array.length c.t_shape - 1);
  cover "a view" (c.t_operand.view <> Plain);
  cover "equal elements" (ties x c.t_axis);
  cover "NaN" nan;
  cover "-0 and +0" zeros;
  cover "an infinity" infinite;
  cover "an integer's least or greatest" extreme;
  if c.t_dt = Nx.Dtype.Any Nx.float64 then assume (has_float64 d);
  let f x = Nx.top_k ~k:c.t_k ~axis:c.t_axis x in
  let values, positions = f x in
  let values', positions' =
    Rune.jit
      Nx.Ptree.(tensor @-> returns (pair tensor tensor))
      f (Nx.place S.on x)
  in
  let host x = Nx.place Nx.Host.on x in
  equal ~msg:"positions" codes_t (codes positions) (codes (host positions'));
  equal ~msg:"values" codes_t (raw values) (raw (host values'))

(* A stack of 64 rows of 64, each computed on the GPU, compiles to the host's
   eager bits. Doubling is exact, a [-0.] kept. *)
let test_stack () =
  let _, (module S) = on_gpu () in
  let row i =
    Nx.create Nx.float32 [| 64 |]
      (Array.init 64 (fun j ->
           if i = 5 && j = 7 then -0. else sin (Float.of_int ((i * 64) + j))))
  in
  let rows = List.init 64 row in
  let bits x = Nx.to_array (Nx.bitcast Nx.int32 (Nx.place Nx.Host.on x)) in
  let doubled rows = Nx.stack (List.map (fun r -> Nx.mul_s r 2.) rows) in
  let f = Rune.jit Nx.Ptree.(list tensor @-> returns tensor) doubled in
  let before = Rune.evaluations () in
  let y = f (List.map (Nx.place S.on) rows) in
  equal ~msg:"compiled" Testable.int 0 (Rune.evaluations () - before);
  equal (Testable.array Testable.int32) (bits (doubled rows)) (bits y)

(* A compiled [Nx.eigh] on the GPU, which runs every sweep, of a batch of 6 x 6
   float32 matrices that converge apart: a diagonal matrix, which needs no
   sweep, one with NaN in its lower triangle, and uniform and graded ones,
   which need several. The values lie within the bound rune.mli states of the
   host's eager ones, [γ(16 (R + n)) ‖a‖_F] with [R] 8 sweeps of 5 rounds, and
   eager's own [γ(16 n) ‖a‖_F]; the
   failing matrix's alone are NaN, and [‖vᵀ v - I‖_F] is within
   [γ(16 (R + n) (1 + √n))], to the check's own float64 rounding. *)
let test_eigh () =
  let _, (module S) = on_gpu () in
  let n = 6 in
  let entry k i j =
    let i, j = (max i j, min i j) in
    let s = sin (float_of_int ((k * 7919) + (i * n) + j + 1)) in
    match k with
    | 0 -> if i = j then s else 0.
    | 1 -> if i = 2 && j = 1 then Float.nan else s
    | 2 -> s
    | _ -> if i = j then 10. ** float_of_int (-i) else 1e-3 *. s
  in
  let a =
    Nx.create Nx.float32 [| 4; n; n |]
      (Array.init (4 * n * n) (fun e ->
           entry (e / (n * n)) (e / n mod n) (e mod n)))
  in
  let f =
    Rune.jit
      Nx.Ptree.(tensor @-> returns (pair tensor tensor))
      (Nx.eigh Nx.float32)
  in
  let before = Rune.evaluations () in
  let w, v = f (Nx.place S.on a) in
  equal ~msg:"compiled" Testable.int 0 (Rune.evaluations () - before);
  let host x = Nx.to_array (Nx.cast Nx.float64 (Nx.place Nx.Host.on x)) in
  let w = host w and v = host v in
  let eager = Nx.to_array (Nx.cast Nx.float64 (Nx.eigvalsh Nx.float32 a)) in
  let gamma_at u k = k *. u /. (1. -. (k *. u)) in
  let gamma = gamma_at (ldexp 1. (-24))
  and gamma64 = gamma_at (ldexp 1. (-53)) in
  let rounds = 40. and nf = float_of_int n in
  for k = 0 to 3 do
    let norm = ref 0. in
    for i = 0 to n - 1 do
      for j = 0 to n - 1 do
        let x = entry k i j in
        if not (Float.is_nan x) then norm := !norm +. (x *. x)
      done
    done;
    let tol =
      (gamma (16. *. (rounds +. nf)) +. gamma (16. *. nf)) *. sqrt !norm
    in
    let g = gamma (16. *. (rounds +. nf) *. (1. +. sqrt nf)) in
    let orthogonality = ref 0. in
    for i = 0 to n - 1 do
      for j = 0 to n - 1 do
        let s = ref (if i = j then -1. else 0.) in
        for l = 0 to n - 1 do
          let at r c = v.((((k * n) + r) * n) + c) in
          s := !s +. (at l i *. at l j)
        done;
        orthogonality := !orthogonality +. (!s *. !s)
      done
    done;
    for i = 0 to n - 1 do
      let e = (k * n) + i in
      if k = 1 then equal ~msg:"NaN" Testable.bool true (Float.is_nan w.(e))
      else
        less
          ~msg:(Printf.sprintf "matrix %d, value %d" k i)
          Testable.float_exact ~than:(tol +. Float.min_float)
          (Float.abs (w.(e) -. eager.(e)))
    done;
    if k <> 1 then
      less
        ~msg:(Printf.sprintf "matrix %d: vᵀ v = I" k)
        Testable.float_exact ~than:(g +. gamma64 (nf +. 2.))
        (sqrt !orthogonality)
  done

(* A range gather compiles on the GPU, as a gather at its runs' positions,
   to the host's eager bits, and so does its gradient: runs that straddle
   the source's ends, an empty run, overlapping runs, and a start near
   int64's greatest after an end near its least, which reads the axis only
   modulo 2^64. *)
let test_ranges () =
  let _, (module S) = on_gpu () in
  let runs a = Nx.create Nx.int64 [| 2; 4 |] a in
  let starts =
    runs [| -2L; 10L; 10L; 45L; Int64.sub Int64.max_int 1L; 3L; 0L; 0L |]
  and ends =
    runs [| 5L; 5L; 12L; 20L; Int64.add Int64.min_int 2L; 9L; 9L; 15L |]
  in
  let x =
    Nx.create Nx.float32 [| 2; 50; 3 |]
      (Array.init 300 (fun i -> Float.of_int ((i * 7 mod 23) - 11)))
  in
  let w = Nx.create Nx.float32 [| 2; 18; 3 |] (Array.init 108 Float.of_int) in
  let gather (starts, ends) x =
    Nx.Prim.eval ~by:"test" (Gather_ranges { size = 18; starts; ends; x })
  in
  let loss runs w x = Nx.sum (Nx.mul w (gather runs x)) in
  let on v = Nx.place S.on v in
  let placed = (on starts, on ends) in
  let bits y = Nx.to_array (Nx.bitcast Nx.int32 (Nx.place Nx.Host.on y)) in
  let ints = Testable.array Testable.int32 in
  let before = Rune.evaluations () in
  let y = Rune.jit' (gather placed) (on x) in
  let g = Rune.jit' (Rune.grad' (loss placed (on w))) (on x) in
  equal ~msg:"compiled" Testable.int 0 (Rune.evaluations () - before);
  equal ~msg:"gather" ints (bits (gather (starts, ends) x)) (bits y);
  equal ~msg:"gradient" ints
    (bits (Rune.grad' (loss (starts, ends) w) x))
    (bits g)

exception Big of int array * float

(* A step whose work is in flight when the call returns. An iterate's stop is
   read on the host each trip, so the work that stays in flight is the work
   after it: [products] products of [n; n] matrices, after a check of the first
   one, which fails at [0; 0]. They are 4.4 TFLOP: at least 68 ms at the RTX
   5000 Ada's float32 peak of 65 TFLOPS, tens of milliseconds on any current
   GPU, while the host reads the timeline microseconds after the call's last
   submit, so the GPU cannot finish them before the read. Each product by
   [w], whose elements are [1/n], keeps the values [n], exactly. *)
let n = 4096
let products = 32

let test_deferred () =
  let d, (module S) = on_gpu () in
  let step x =
    let x =
      Rune.iterate' ~max:4
        ~until:(fun x -> Nx.greater (Nx.sum x) (Nx.scalar_like (Nx.sum x) 0.))
        (fun x -> Nx.add x (Nx.scalar_like x 1.))
        x
    in
    let y = Nx.matmul x x in
    Nx.check Nx.Ptree.tensor
      (Nx.less y (Nx.scalar_like y (float_of_int n /. 2.)))
      y
      (fun i v -> Big (i, Nx.item [] v));
    let w = Nx.div x (Nx.scalar_like x (float_of_int n)) in
    let h = ref y in
    for _ = 1 to products do
      h := Nx.matmul !h w
    done;
    !h
  in
  let x = Nx.place S.on (Nx.zeros Nx.float32 [| n; n |]) in
  let g = Rune.jit' step in
  let before = Rune.evaluations () in
  (* The first call compiles. *)
  (try Nx.sync (g x) with Big _ -> ());
  let y = g x in
  let signaled = Rig.signaled d and submitted = Rig.submitted d in
  equal ~msg:"compiled" Testable.int 0 (Rune.evaluations () - before);
  less ~msg:"the call returned before its work completed" Testable.int
    ~than:submitted signaled;
  raises ~msg:"the next wait raises the check"
    (Big ([| 0; 0 |], float_of_int n))
    (fun () -> Nx.sync y)

(* Memory of a split program

   A chain of [chain] products of [chain_n] by [chain_n] float32 matrices on
   the GPU, in a program split by an argument on a software device. A product
   reads the one before it, so at most two products live at once beside the
   arguments and the result, and the call holds no more: its plan places
   the products in that much memory. The call returns while its products
   run, 17 GFLOP each, at least 2 ms in all at the RTX 5000 Ada's float32
   peak. The GPU reuses memory a call returns only once the work on it is
   done, so at the call's return it holds, beside what it held before, at
   most the result and the two products. *)
let chain = 8
let chain_n = 2048

let test_split_memory () =
  let gpu, _ = on_gpu () in
  let mem = Result.get_ok (Rig.software "rune-jit-gpu-chain") in
  let (module S) = Nx.devices [ gpu; mem ] in
  let on_gpu = Nx.Placement.device S.v gpu
  and on_mem = Nx.Placement.device S.v mem in
  let x = Nx.place on_gpu (Nx.full Nx.float32 [| chain_n; chain_n |] 1.) in
  let w =
    Nx.place on_gpu
      (Nx.full Nx.float32 [| chain_n; chain_n |] (1. /. float_of_int chain_n))
  in
  let z = Nx.place on_mem (Nx.full Nx.float32 [| 4 |] 1.) in
  let f (x, (w, z)) =
    let y = ref x in
    for _ = 1 to chain do
      y := Nx.matmul !y w
    done;
    (!y, Nx.mul_s z 2.)
  in
  let g =
    Rune.jit
      Nx.Ptree.(pair tensor (pair tensor tensor) @-> returns (pair tensor tensor))
      f
  in
  let settled () =
    for _ = 1 to 4 do
      Gc.full_major ();
      Rig.free_cache gpu
    done;
    Rig.allocated gpu
  in
  (* The first call compiles. *)
  Nx.sync (fst (g (x, (w, z))));
  let before = settled () in
  let evaluated = Rune.evaluations () in
  let y, _ = g (x, (w, z)) in
  let held = Rig.allocated gpu - before in
  let signaled = Rig.signaled gpu and submitted = Rig.submitted gpu in
  equal ~msg:"compiled" int 0 (Rune.evaluations () - evaluated);
  less ~msg:"the call returned before its work completed" int ~than:submitted
    signaled;
  let product = chain_n * chain_n * 4 in
  at_most int ~msg:"the result and two products" ~than:(3 * product) held;
  equal ~msg:"the result" float_exact 1. (Nx.item [ 0; 0 ] (Nx.place Nx.Host.on y))

(* A detour

   A function of host values that computes on the GPU's set, which has no
   kernels, and returns to the host compiles and gives the host's arithmetic. *)

let detour () =
  let _, (module S : Nx.Devices) = on_gpu () in
  let x = Nx.create Nx.float32 [| 3 |] [| 1.5; -2.; 3.25 |] in
  let f x =
    let y = Nx.place S.on x in
    Nx.place Nx.Host.on (Nx.add (Nx.mul y y) y)
  in
  let before = Rune.evaluations () in
  let y = Rune.jit' f x in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  equal bits_t (bits (Nx.add (Nx.mul x x) x)) (bits y)

(* Several targets

   A program over the GPU beside a software device compiles, each device running
   kernels of its own target, and gives the placements the same function gives
   eagerly over two software devices, an exact function's bits and another's
   values within rune.mli's classes. The devices of the first device's
   architecture compute a value whole on several devices and the others
   receive copies, so its replicas are equal bit for bit, whichever device
   comes first. *)

let f32 = Nx.float32

(* A placement: whole on each device, cut along an axis, whole on the [k]th
   device, or, over a 2 x 2 mesh, cut along axis [0] over its first axis and
   whole along its second, or along both, axis [0] over the mesh's first axis
   or its second. *)
type cut = On | Split of int | Dev of int | Rows | Grid of bool

(* A set, and the mesh over it where it has four devices. *)
type 'd set = { s : 'd Nx.Devices.t; mesh : 'd Nx.Mesh.t option }

let set_of (type d) (s : d Nx.Devices.t) =
  let mesh =
    if List.length (Nx.Devices.to_list s) = 4 then
      Some (Nx.Mesh.v s [ ("x", 2); ("y", 2) ])
    else None
  in
  { s; mesh }

let at set = function
  | On -> Nx.Placement.on set.s
  | Split axis -> Nx.Placement.split ~axis set.s
  | Dev k -> Nx.Placement.device set.s (List.nth (Nx.Devices.to_list set.s) k)
  | Rows -> Nx.Placement.mesh (Option.get set.mesh) [ (0, [ "x" ]) ]
  | Grid a ->
      let x, y = if a then ("x", "y") else ("y", "x") in
      Nx.Placement.mesh (Option.get set.mesh) [ (0, [ x ]); (1, [ y ]) ]

(* A value of [shape] on the host, of elements in [-1, 1] that depend on [seed]
   and their index. *)
let value ?(seed = 0) shape =
  let n = Array.fold_left ( * ) 1 shape in
  Nx.create f32 shape
    (Array.init n (fun i -> sin (float_of_int ((seed * 7919) + i + 1))))

let host x = Nx.place Nx.Host.on x
let placement () = Testable.make ~pp:Nx.Placement.pp ~equal:Nx.Placement.equal

(* [γ(k)] at [float32]. *)
let gamma k =
  let u = ldexp 1. (-24) in
  let k = float_of_int k in
  k *. u /. (1. -. (k *. u))

(* [γ(k, 2u)] at [float32], a contraction's bound ({!Nx.contract}). *)
let product k =
  let v = ldexp 1. (-23) in
  let k = float_of_int k in
  k *. v /. (1. -. (k *. v))

(* [jit] within [bound] of [eager] elementwise, NaN where it is NaN. *)
let within ~bound eager jit =
  let e = Nx.to_array (Nx.cast Nx.float64 (host eager)) in
  let j = Nx.to_array (Nx.cast Nx.float64 (host jit)) in
  let b = Nx.to_array (Nx.cast Nx.float64 (host bound)) in
  Array.iteri
    (fun i ei ->
      if Float.is_nan ei then equal ~msg:"NaN" bool true (Float.is_nan j.(i))
      else
        less
          ~msg:(Printf.sprintf "distance at %d" i)
          float_exact
          ~than:(b.(i) +. Float.min_float)
          (Float.abs (j.(i) -. ei)))
    e

(* A function of two values, its arguments on the host with their placements,
   its result's placement, and how its compiled result compares with eager's:
   [close x y eager jit]. *)
type case = {
  name : string;
  f : 'd. (float, Nx.float32_elt, 'd) Nx.t -> (float, Nx.float32_elt, 'd) Nx.t ->
      (float, Nx.float32_elt, 'd) Nx.t;
  x : (float, Nx.float32_elt, Nx.host) Nx.t * cut;
  y : (float, Nx.float32_elt, Nx.host) Nx.t * cut;
  result : cut;
  close :
    (float, Nx.float32_elt, Nx.host) Nx.t ->
    (float, Nx.float32_elt, Nx.host) Nx.t ->
    (float, Nx.float32_elt, Nx.host) Nx.t ->
    (float, Nx.float32_elt, Nx.host) Nx.t ->
    unit;
}

let exact _ _ e j = equal bits_t (bits e) (bits j)

(* A value whose magnitudes span twenty binades, so that two orders of
   addition round apart. *)
let spread shape =
  let n = Array.fold_left ( * ) 1 shape in
  Nx.create f32 shape
    (Array.init n (fun i -> ldexp (sin (float_of_int (i + 1))) (i * 7 mod 20)))

let mass x = Nx.abs (Nx.cast Nx.float64 x)

(* [tanh (3 x)]: on the GPU and eagerly each within 18 ulps of the correctly
   rounded result, an ulp at most [2 u] of it, so within [72 u] of each other,
   relatively, and [73 u] of eager's result. *)
let tanh x = Nx.tanh (Nx.mul_s x 3.)
let tanh_err = 73. *. ldexp 1. (-24)

let cases =
  let add_to carry row = (Nx.add carry row, Nx.zeros_like row) in
  let row_of xs = Nx.zeros_like (Nx.sum ~axes:[ 0 ] xs) in
  let stops_at limit x =
    Rune.iterate' ~max:3
      ~until:(fun c -> Nx.greater_equal_s (Nx.sum c) limit)
      (Nx.add x) (Nx.zeros_like x)
  in
  [
    {
      name = "elementwise over a cut and a whole value";
      f = (fun x y -> Nx.maximum (Nx.mul x y) (Nx.sub x (Nx.neg y)));
      x = (value [| 8; 3 |], Split 0);
      y = (value ~seed:1 [| 8; 3 |], On);
      result = Split 0;
      close = exact;
    };
    {
      name = "elementwise over whole values";
      f = (fun x y -> Nx.maximum (Nx.mul x y) (Nx.sub x (Nx.neg y)));
      x = (value [| 8; 3 |], On);
      y = (value ~seed:1 [| 8; 3 |], On);
      result = On;
      close = exact;
    };
    {
      name = "a tanh of a whole value";
      f = (fun x _ -> tanh x);
      x = (value [| 1024 |], On);
      y = (value ~seed:1 [| 1 |], On);
      result = On;
      close = (fun _ _ e j -> within ~bound:(Nx.mul_s (mass e) tanh_err) e j);
    };
    {
      name = "a sum over the cut axis";
      f = (fun x _ -> Nx.sum ~axes:[ 0 ] x);
      x = (spread [| 9; 7 |], Split 0);
      y = (value ~seed:1 [| 1 |], On);
      result = On;
      close =
        (fun x _ e j ->
          within ~bound:(Nx.mul_s (Nx.sum ~axes:[ 0 ] (mass x)) (2. *. gamma 9))
            e j);
    };
    {
      name = "a matrix product cut along its contracted axis";
      f = Nx.matmul;
      x = (value [| 5; 6 |], Split 1);
      y = (value ~seed:1 [| 6; 3 |], Split 0);
      result = On;
      close =
        (fun x y e j ->
          within
            ~bound:(Nx.mul_s (Nx.matmul (mass x) (mass y)) (2. *. product 7))
            e j);
    };
    {
      name = "a scan over cut rows whose first value the program makes";
      f = (fun xs _ -> fst (Rune.scan' add_to ~init:(row_of xs) xs));
      x = (value [| 3; 8 |], Split 1);
      y = (value ~seed:1 [| 1 |], On);
      result = Split 0;
      close = exact;
    };
    {
      name = "a scan's outputs the program makes, on a device apart";
      f = (fun _ xs -> snd (Rune.scan' add_to ~init:(row_of xs) xs));
      x = (value ~seed:1 [| 1 |], Dev 0);
      y = (value [| 3; 8 |], Dev 1);
      result = Dev 1;
      close = exact;
    };
    {
      name = "a scan of no rows over cut rows";
      f = (fun xs _ -> fst (Rune.scan' add_to ~init:(row_of xs) xs));
      x = (value [| 0; 8 |], Split 1);
      y = (value ~seed:1 [| 1 |], On);
      result = Split 0;
      close = exact;
    };
    {
      name = "an iterate over a cut value whose first value the program makes";
      f = (fun x _ -> stops_at 16. x);
      x = (Nx.ones f32 [| 8 |], Split 0);
      y = (value ~seed:1 [| 1 |], On);
      result = Split 0;
      close = exact;
    };
    {
      name = "an iterate of no steps on a device apart";
      f = (fun _ y -> stops_at (-1.) y);
      x = (value ~seed:1 [| 1 |], Dev 0);
      y = (Nx.ones f32 [| 8 |], Dev 1);
      result = Dev 1;
      close = exact;
    };
    {
      (* As test_jit_devices.ml's gradient: [g = xᵀ (1 - t²)] for
         [t = tanh (x w)]. *)
      name = "the gradient of a loss over a cut batch";
      f = (fun x w -> Rune.grad' (fun w -> Nx.sum (Nx.tanh (Nx.matmul x w))) w);
      x = (value [| 6; 4 |], Split 0);
      y = (value ~seed:1 [| 4; 3 |], On);
      result = On;
      close =
        (fun x w e j ->
          let ax = mass x and u = ldexp 1. (-24) in
          let term =
            Nx.add_s
              (Nx.mul_s (Nx.matmul ax (mass w)) (2. *. product 5))
              ((38. *. u) +. product 7)
          in
          let bound = Nx.mul_s (Nx.matmul (Nx.transpose ax) term) 2. in
          within ~bound e j);
    };
  ]

(* Each device's piece of [y], on the host, and its window. *)
let pieces y =
  let s = Nx.Placement.devices (Nx.placement y) in
  let windows = Nx.Prim.pieces (Nx.placement y) (Nx.shape y) in
  Array.mapi
    (fun j a -> (bits (host (Nx.Repr.of_array s a)), snd windows.(j)))
    (Iarray.to_array (Nx.Repr.shards y))

(* Over a 2 x 2 mesh: a value cut along its first axis over the mesh's first
   axis, which its second repeats, so that each window lies on two devices and
   each fold over that axis closes in two groups. *)
let mesh_cases =
  (* The terms [t] of a sum or a cumulative sum of [n] of [tanh (3 x)]: within
     [(73 u + 2 γ(n)) Σ|t|] of eager's. *)
  let folded fold n x _ e j =
    let terms = Nx.abs (Nx.cast Nx.float64 (tanh x)) in
    within ~bound:(Nx.mul_s (fold terms) (tanh_err +. (2. *. gamma n))) e j
  in
  [
    {
      name = "a tanh of rows";
      f = (fun x _ -> tanh x);
      x = (value [| 256; 4 |], Rows);
      y = (value ~seed:1 [| 1 |], On);
      result = Rows;
      close = (fun _ _ e j -> within ~bound:(Nx.mul_s (mass e) tanh_err) e j);
    };
    {
      name = "a sum over the rows";
      f = (fun x _ -> Nx.sum ~axes:[ 0 ] (tanh x));
      x = (value [| 64; 7 |], Rows);
      y = (value ~seed:1 [| 1 |], On);
      result = On;
      close = folded (Nx.sum ~axes:[ 0 ]) 64;
    };
    {
      (* The positions of the column extremes, distinct values, as floats. *)
      name = "the positions of the extremes over the rows";
      f = (fun x _ -> Nx.cast f32 (Nx.argmax ~axis:0 (Nx.mul_s x 3.)));
      x = (value [| 64; 5 |], Rows);
      y = (value ~seed:1 [| 1 |], On);
      result = On;
      close = exact;
    };
    {
      name = "a cumulative sum along the rows";
      f = (fun x _ -> Nx.cumsum ~axis:0 (tanh x));
      x = (value [| 64; 3 |], Rows);
      y = (value ~seed:1 [| 1 |], On);
      result = Rows;
      close = folded (Nx.cumsum ~axis:0) 64;
    };
  ]

(* [c] compiled over [mixed] against [c] applied eagerly over [memory]. Devices
   that hold one window of the result hold one piece, bit for bit. *)
let check (type d e) (mixed : d set) (memory : e set) c =
  let (x, cx), (y, cy) = (c.x, c.y) in
  let eager = c.f (Nx.place (at memory cx) x) (Nx.place (at memory cy) y) in
  equal ~msg:"eager's placement" (placement ()) (at memory c.result)
    (Nx.placement eager);
  let f = Rune.jit Nx.Ptree.(tensor @-> tensor @-> returns tensor) c.f in
  let before = Rune.evaluations () in
  let jit = f (Nx.place (at mixed cx) x) (Nx.place (at mixed cy) y) in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  equal ~msg:"placement" (placement ()) (at mixed c.result) (Nx.placement jit);
  c.close x y (host eager) (host jit);
  let ps = pieces jit
  and devices = Array.of_list (Nx.Devices.to_list mixed.s) in
  Array.iteri
    (fun j (b, w) ->
      match Array.find_index (fun (_, w') -> w' = w) ps with
      | Some i when i < j ->
          equal
            ~msg:("the replica on " ^ Rig.name devices.(j))
            bits_t (fst ps.(i)) b
      | _ -> ())
    ps

let memory k = Result.get_ok (Rig.software ("rune-jit-gpu-" ^ k))

(* The sets the laws run over: the GPU and software devices, in an order that
   puts the GPU at [first], and as many software devices. *)
let sets =
  Hashtbl.create 4

let mixed_sets ~devices ~first =
  let gpu, _ = on_gpu () in
  match Hashtbl.find_opt sets (devices, first) with
  | Some s -> s
  | None ->
      let ms = List.init devices (fun k -> memory (string_of_int k)) in
      let others = List.tl ms in
      let mixed =
        List.filteri (fun k _ -> k < first) others
        @ [ gpu ]
        @ List.filteri (fun k _ -> k >= first) others
      in
      let s = (Nx.devices mixed, Nx.devices ms) in
      Hashtbl.replace sets (devices, first) s;
      s

let over ~devices ~first c () =
  let mixed, eager = mixed_sets ~devices ~first in
  let (module S : Nx.Devices) = mixed and (module M : Nx.Devices) = eager in
  check (set_of S.v) (set_of M.v) c

(* A law over a set of a GPU beside a software device, or over the set's
   software devices alone. *)
type law = { law : 'd. 'd set -> unit }

(* [l] over the GPU beside a software device, the GPU first and second. *)
let on_mixed { law } =
  List.iter
    (fun first ->
      let mixed, _ = mixed_sets ~devices:2 ~first in
      let (module S : Nx.Devices) = mixed in
      law (set_of S.v))
    [ 0; 1 ]

let no_call_evaluated f =
  let before = Rune.evaluations () in
  let r = f () in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  r

(* An emission over a GPU beside a software device compiles and is offered once,
   with eager's bits. *)
let emission set =
  let x = value [| 4 |] in
  let p = Rune.Probe.make Nx.Ptree.tensor in
  let f x =
    Rune.Probe.emit p (Nx.mul x x);
    Nx.add x x
  in
  let placed = Nx.place (at set (Split 0)) x in
  let r, offers =
    no_call_evaluated (fun () ->
        Rune.Probe.collect p (fun () -> Rune.jit' f placed))
  in
  (match r with
  | Ok y -> equal bits_t (bits (Nx.add x x)) (bits (host y))
  | Error (e, bt) -> Printexc.raise_with_backtrace e bt);
  match offers with
  | [ o ] ->
      (* An offer holds its emits as rows. *)
      equal bits_t (bits (Nx.stack ~axis:0 [ Nx.mul x x ])) (bits (host o))
  | _ -> failf "%d offers" (List.length offers)

(* Constructs compiled over a GPU beside software devices give what the same
   function gives eagerly over software devices of the same number. *)

type 'a agrees = { agree : 'd. 'd set -> jit:bool -> 'a }

let agreeing ?(devices = 2) { agree } t =
  List.iter
    (fun first ->
      let mixed, eager = mixed_sets ~devices ~first in
      let (module S : Nx.Devices) = mixed
      and (module M : Nx.Devices) = eager in
      equal t (agree (set_of M.v) ~jit:false) (agree (set_of S.v) ~jit:true))
    [ 0; 1 ]

let codes_text y = bits (host y)

let compiled_or_eager ~jit f x y =
  let s = Nx.Ptree.(tensor @-> tensor @-> returns tensor) in
  if jit then no_call_evaluated (fun () -> Rune.jit s f x y) else f x y

(* The outcome of [f]: its bits at a wait, or a failure at the wait. *)
let waited f =
  match f () with
  | y -> ( match host y with h -> Ok (bits h) | exception Failure s -> Error s)
  | exception Failure s -> Error s

let outcome_t = Testable.(result bits_t string)

(* A check fails as eagerly, its flags and data wherever they lie. *)
let check_fails ?devices ~flags ~data shape falses () =
  agreeing ?devices
    {
      agree =
        (fun set ~jit ->
          let n = Array.fold_left ( * ) 1 shape in
          let ok =
            Nx.place (at set flags)
              (Nx.create Nx.bool shape
                 (Array.init n (fun i -> not (List.mem i falses))))
          in
          let d = Nx.place (at set data) (value shape) in
          let fail idx v =
            let i =
              String.concat "," (Array.to_list (Array.map string_of_int idx))
            in
            Failure (Printf.sprintf "%s: %h" i (Nx.item [] (host v)))
          in
          let f ok d =
            Nx.check Nx.Ptree.tensor ok d fail;
            Nx.add d d
          in
          waited (fun () -> compiled_or_eager ~jit f ok d));
    }
    outcome_t

(* Exact functions' values under custom rules and remats, and their additions
   to a total, as eagerly. *)
let eight set =
  let x = Nx.create f32 [| 8 |] (Array.init 8 Float.of_int) in
  Nx.place (at set (Split 0)) x

let customs set ~jit =
  let x = eight set in
  let square =
    Rune.custom_vjp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
        (Nx.mul x x, fun ct -> Nx.mul_s ct 3.))
  in
  let s = Nx.Ptree.(tensor @-> returns tensor) in
  let g x _ = Nx.add (Rune.remat s square x) x in
  let d x _ =
    Rune.grad' (fun x -> Nx.sum (Rune.remat s square x)) x
  in
  (codes_text (compiled_or_eager ~jit g x x),
   codes_text (compiled_or_eager ~jit d x x))

let total set ~jit =
  let x = eight set in
  let t = Rune.Total.make () in
  let f x =
    Rune.Total.add t (Nx.mul x x);
    x
  in
  let a, e =
    Rune.Total.collect t ~init:(Nx.zeros_like x) (fun () ->
        if jit then no_call_evaluated (fun () -> Rune.jit' f x) else f x)
  in
  (codes_text a, codes_text e)

(* A switch compiles over a GPU beside software devices: its index on any
   device or whole on each, its branch's value and placement as eagerly
   over software devices, an index out of range failing at a wait. *)
let waited_any f =
  match f () with
  | y -> (
      match host y with
      | h -> Ok (bits h)
      | exception (Failure s | Invalid_argument s) -> Error s)
  | exception (Failure s | Invalid_argument s) -> Error s

let switch_agrees ?devices ~cut:c ~index:ic ~taken shape () =
  agreeing ?devices
    {
      agree =
        (fun set ~jit ->
          let x = Nx.place (at set c) (value shape) in
          let i = Nx.place (at set ic) (Nx.scalar Nx.int32 taken) in
          let f i x =
            Rune.switch Nx.Ptree.tensor i
              [ (fun () -> Nx.neg x); (fun () -> Nx.mul_s x 2.) ]
          in
          waited_any (fun () -> compiled_or_eager ~jit f i x));
    }
    outcome_t

(* A scan holding a switch and a nested switch agree with eager. *)
let switch_scan set ~jit =
  let start = Nx.create f32 [| 6 |] (Array.init 6 Float.of_int) in
  let x = Nx.place (at set (Split 0)) start in
  let xs = Nx.place (at set (Split 1)) (value ~seed:1 [| 3; 6 |]) in
  let i0 = Nx.place (at set On) (Nx.scalar Nx.int32 1l) in
  let step (i, c) r =
    let c' =
      Rune.switch Nx.Ptree.tensor i
        [ (fun () -> Nx.add c r); (fun () -> Nx.mul c r) ]
    in
    ((Nx.sub (Nx.scalar_like i 1l) i, c'), c')
  in
  let f x xs =
    let (_, c), ys =
      Rune.scan Nx.Ptree.(pair tensor tensor) Nx.Ptree.tensor
        Nx.Ptree.tensor step ~init:(i0, x) xs
    in
    Nx.add c (Rune.switch Nx.Ptree.tensor i0
                [ (fun () -> Nx.sum ys ~axes:[ 0 ]); (fun () -> c) ])
  in
  (codes_text (compiled_or_eager ~jit f x xs), ())

(* A declined program over a GPU beside a software device raises at the call,
   naming the decline. *)
let declined set =
  let xs = Nx.place (at set (Split 0)) (value [| 3; 2 |]) in
  let f xs =
    let init = Nx.sum ~axes:[ 0 ] xs in
    fst (Rune.scan' (fun c x -> (Nx.add c x, x)) ~init xs)
  in
  raises_match
    (Exn.invalid_arg ~sub:"the compiler declines this program")
    (fun () -> Rune.jit' f xs)

(* [split k]'s second key under jit at the GPU's set. A GPU set has no
   kernels, so a key there derives only in a program. *)
type second =
  | Second : 'd Nx.Placement.t * ('d Nx.Rng.key -> 'd Nx.Rng.key) -> second

let second =
  lazy
    (let _, (module S) = on_gpu () in
     Second
       ( S.on,
         Rune.jit
           Nx.Ptree.(Nx.Rng.ptree @-> returns Nx.Rng.ptree)
           (fun k -> (Nx.Rng.split k).(1)) ))

let words k = Nx.to_array (Nx.Rng.to_tensor (Nx.Rng.place Nx.Host.on k))

let law_split seed =
  let (Second (p, f)) = Lazy.force second in
  let k = Nx.Rng.key seed in
  equal
    (Testable.array Testable.int32)
    (words (Nx.Rng.split k).(1))
    (words (f (Nx.Rng.place p k)))

(* Matrix units

   A compiled contraction of bfloat16 or float16 operands into float32 holds
   {!Nx.contract}'s bound, [γ(K + 1, 2u) (|init| + Σ|a||b|)], on the matrix
   units the compiler takes for it too. Each row is [c] then [K - 1] equal
   terms [t] against ones, with [t] just under [2^j 2u c] for [j] from [1] to
   [13], up to float16's unit in the last place at [c], and just under one and
   two of float32's. A unit whose sum keeps float32's width errs by less than
   one of its units per addition, so it meets the bound even where it
   truncates, as the rows with [t] near [u c] press it to. A narrower sum, or
   one that drops a term, loses about [t] per addition from [j = 1], twice the
   bound or more: a float16 accumulator errs by 2 to 1700 times it on these
   rows. *)

let units_k = 2880

(* The terms of each row, as multiples of [u c]: fifteen, and the first again,
   so that the rows fill a tensor core's tile of sixteen, which takes no
   product of fifteen rows. *)
let unit_terms =
  let terms =
    (255. /. 256.) :: (255. /. 128.)
    :: List.init 13 (fun j -> 255. /. 256. *. ldexp 1. (j + 2))
  in
  terms @ [ List.hd terms ]

let unit_case (type b) (dt : (float, b) Nx.dtype) c () =
  let _, (module S) = on_gpu () in
  let u = ldexp 1. (-24) in
  let ts = Array.of_list (List.map (fun m -> m *. u *. c) unit_terms) in
  let rows = Array.length ts and cols = 64 in
  let a =
    Nx.init Nx.float32 [| rows; units_k |] (fun i ->
        if i.(1) = 0 then c else ts.(i.(0)))
  in
  equal ~msg:"the terms are exact in the operands' dtype"
    (Testable.array float_exact) (Nx.to_array a)
    (Nx.to_array (Nx.cast Nx.float32 (Nx.cast dt a)));
  let a = Nx.place S.on (Nx.cast dt a) in
  let b = Nx.place S.on (Nx.ones dt [| units_k; cols |]) in
  let p = Nx.Pattern.v "i j, j k -> i k | j" in
  let f a b = Nx.contract Nx.float32 p a b in
  let jit = Rune.jit Nx.Ptree.(tensor @-> tensor @-> returns tensor) f a b in
  let n = float_of_int (units_k + 1) and v = 2. *. u in
  let g = n *. v /. (1. -. (n *. v)) in
  Array.iteri
    (fun i y ->
      let t = ts.(i / cols) in
      let exact = c +. (float_of_int (units_k - 1) *. t) in
      less
        ~msg:(Printf.sprintf "row %d, terms %h, column %d" (i / cols) t
                (i mod cols))
        float_exact ~than:(g *. exact)
        (Float.abs (y -. exact)))
    (Nx.to_array (host jit))

let unit_cases =
  [
    test "bfloat16 into float32" (unit_case Nx.bfloat16 1.);
    test "float16 into float32" (unit_case Nx.float16 2048.);
  ]

(* A product of narrow floats: a [m; k] by [k; n] Nx.matmul compiled on the
   GPU, against the product of the same values computed on the host in
   float64, whose own error is 2^-29 of the bound's. {!Nx.contract}'s bound
   for [k] products summed at float32 is [γ(k + 1, 2u) Σ|a b|]; the result
   then rounds once into its dtype, by a relative [u] of that dtype or, below
   its least normal, half its least subnormal. A GPU's tensor cores take
   extents that 16 divides and refuse the others. *)

type narrow = Narrow : (float, 's) Nx.dtype * float * float -> narrow

let narrows =
  [
    Narrow (Nx.float16, 0x1p-11, 0x1p-25);
    Narrow (Nx.bfloat16, 0x1p-8, 0x1p-134);
  ]

let pp_product ppf (Narrow (dt, _, _), (m, k, n), seed) =
  Format.fprintf ppf "%a [%d; %d] by [%d; %d], seed %d" Nx.Dtype.pp dt m k k n
    seed

let products_gen =
  let extent = Gen.of_list [ 0; 1; 7; 16; 17; 32; 48 ] in
  let tiled = Gen.of_list [ 16; 32; 48 ] in
  let shape =
    Gen.frequency
      [
        (1, Gen.triple extent extent extent); (1, Gen.triple tiled tiled tiled);
      ]
  in
  Gen.with_pp pp_product
    (Gen.triple (Gen.of_list narrows) shape (Gen.int_range 0 1_000_000))

(* Float32 products on a GPU's registers: extents of 64 and 128 on both output
   axes block a thread's outputs 8 by 8, and unroll a sum of 52 by 4; a sum
   of 13 is not unrolled. 40 columns of 256 rows leave 5 runs of 8 outputs,
   which the blocked threads take padded; 40 rows of 512 columns leave 5
   runs of rows, and a workgroup of 16 threads. *)
let padded_products = [ (256, 13, 40); (40, 13, 512) ]

let float32_products_gen =
  let outputs = Gen.of_list [ 1; 13; 64; 128 ] in
  let blocked = Gen.of_list [ 64; 128 ] in
  let shape =
    Gen.frequency
      [
        (1, Gen.triple outputs (Gen.of_list [ 1; 13; 64 ]) outputs);
        (2, Gen.triple blocked (Gen.of_list [ 13; 52 ]) blocked);
        (1, Gen.of_list padded_products);
      ]
  in
  Gen.with_pp pp_product
    (Gen.triple
       (Gen.constant (Narrow (Nx.float32, 0x1p-24, 0x1p-149)))
       shape (Gen.int_range 0 1_000_000))

let within_bound (Narrow (dt, u, tiny), (m, k, n), seed) =
  let _, (module S) = on_gpu () in
  let draw seed s =
    let x = Nx.Rng.uniform ~key:(Nx.Rng.key seed) Nx.float32 s in
    Nx.cast dt (Nx.sub_s (Nx.mul_s x 4.) 2.)
  in
  let a = draw seed [| m; k |] and b = draw (seed + 1) [| k; n |] in
  let before = Rune.evaluations () in
  let y =
    Rune.jit
      Nx.Ptree.(tensor @-> tensor @-> returns tensor)
      Nx.matmul (Nx.place S.on a) (Nx.place S.on b)
  in
  equal ~msg:"compiled" int 0 (Rune.evaluations () - before);
  let y = Nx.to_array (Nx.cast Nx.float64 (host y)) in
  let a = Nx.cast Nx.float64 a and b = Nx.cast Nx.float64 b in
  let exact = Nx.to_array (Nx.matmul a b) in
  let mass = Nx.to_array (Nx.matmul (Nx.abs a) (Nx.abs b)) in
  Array.iteri
    (fun i z ->
      let e = product (k + 1) *. mass.(i) in
      let bound = e +. (u *. (Float.abs z +. e)) +. tiny in
      less
        ~msg:(Printf.sprintf "[%d; %d], exactly %h" (i / n) (i mod n) z)
        float_exact ~than:bound
        (Float.abs (y.(i) -. z)))
    exact

let product_law ((_, (m, k, n), _) as c) =
  cover "extents that 16 divides"
    (m > 0 && k > 0 && n > 0 && m mod 16 = 0 && k mod 16 = 0 && n mod 16 = 0);
  within_bound c

let float32_law ((_, (m, k, n), _) as c) =
  let blocked = m mod 64 = 0 && n mod 64 = 0 && m > 0 && n > 0 in
  cover "outputs that 64 divides, a sum that 4 divides" (blocked && k = 52);
  cover "outputs that 64 divides, a sum of 13" (blocked && k = 13);
  cover "columns padded to the blocked threads" ((m, k, n) = (256, 13, 40));
  cover "rows that leave a blocked workgroup 16 threads"
    ((m, k, n) = (40, 13, 512));
  within_bound c

(* A padded blocked product with a bias of each column, which its kernel adds
   to each output: within the bound of a sum of [k + 1] terms. *)
let test_padded_bias () =
  let _, (module S) = on_gpu () in
  let m = 256 and k = 52 and n = 440 in
  let draw seed s =
    let x = Nx.Rng.uniform ~key:(Nx.Rng.key seed) Nx.float32 s in
    Nx.sub_s (Nx.mul_s x 4.) 2.
  in
  let a = draw 1 [| m; k |] and b = draw 2 [| k; n |] and c = draw 3 [| n |] in
  let y =
    Rune.jit
      Nx.Ptree.(tensor @-> tensor @-> tensor @-> returns tensor)
      (fun a b c -> Nx.add (Nx.matmul a b) c)
      (Nx.place S.on a) (Nx.place S.on b) (Nx.place S.on c)
  in
  let y = Nx.to_array (Nx.cast Nx.float64 (host y)) in
  let d x = Nx.cast Nx.float64 x in
  let exact = Nx.to_array (Nx.add (Nx.matmul (d a) (d b)) (d c)) in
  let mass =
    Nx.to_array
      (Nx.add (Nx.matmul (Nx.abs (d a)) (Nx.abs (d b))) (Nx.abs (d c)))
  in
  let u = 0x1p-24 in
  Array.iteri
    (fun i z ->
      let e = product (k + 2) *. mass.(i) in
      less
        ~msg:(Printf.sprintf "[%d; %d]" (i / n) (i mod n))
        float_exact ~than:(e +. (u *. (Float.abs z +. e)))
        (Float.abs (y.(i) -. z)))
    exact

(* Sixteen products of [1] by [1; t; ...; t], [t = 255 2^-31] just under
   float32's [u], every value a bfloat16: rows of ones by columns of these
   terms, [32; 16] by [16; 32], extents a core takes. A sum that keeps
   float32's width rounds each addition of [t] to [1] at most to its unit in
   the last place, within [Nx.contract]'s bound. *)
let test_sixteen_terms () =
  let _, (module S) = on_gpu () in
  let k = 16 and n = 32 and t = 255. *. ldexp 1. (-31) in
  let a = Nx.ones Nx.bfloat16 [| n; k |] in
  let b =
    Nx.cast Nx.bfloat16
      (Nx.init Nx.float32 [| k; n |] (fun i -> if i.(0) = 0 then 1. else t))
  in
  equal ~msg:"the terms are bfloat16s" float_exact t
    (Nx.item [ 1; 0 ] (Nx.cast Nx.float32 b));
  let y =
    Rune.jit
      Nx.Ptree.(tensor @-> tensor @-> returns tensor)
      (Nx.contract Nx.float32 (Nx.Pattern.v "i j, j k -> i k | j"))
      (Nx.place S.on a) (Nx.place S.on b)
  in
  let exact = 1. +. (float_of_int (k - 1) *. t) in
  Array.iteri
    (fun i y ->
      less
        ~msg:(Printf.sprintf "[%d; %d]" (i / n) (i mod n))
        float_exact ~than:(product (k + 1) *. exact)
        (Float.abs (y -. exact)))
    (Nx.to_array (host y))

(* A dense attention layer's step, as an Earth-observation encoder trains it:
   projections into 4 heads, scores with a bias of each key and the bounds of
   the keys each query sees, none of which reads the heads, dropped
   probabilities, and the heads merged and projected. Its gradient is the
   host's eager one within float32's rounding: the upcasts of the scores' 64
   pixels by 4 heads by 256 queries by 256 keys take the 4 heads whole. *)
let dense_grad (module S : Nx.Devices) ~jit =
  let r = 64 and t = 256 and heads = 4 and dm = 64 in
  let draw i s = Nx.place S.on (Nx.Rng.normal ~key:(Nx.Rng.key i) Nx.float32 s) in
  let x = draw 1 [| r; t; dm |] in
  let w i = draw i [| dm; dm |] in
  let bias = draw 6 [| r; t |] in
  let start = Nx.place S.on (Nx.zeros Nx.int32 [| r; t |]) in
  let stop =
    Nx.place S.on (Nx.full Nx.int32 [| r; t |] (Int32.of_int (t - 1)))
  in
  let keep =
    Nx.place S.on
      (Nx.cast Nx.float32
         (Nx.greater_s
            (Nx.Rng.uniform ~key:(Nx.Rng.key 7) Nx.float32 [| r; heads; t; t |])
            0.1))
  in
  let col =
    Nx.place S.on (Nx.reshape [| 1; 1; 1; t |] (Nx.arange Nx.int32 0 t 1))
  in
  let split w x =
    Nx.swapaxes 1 2 (Nx.reshape [| r; t; heads; dm / heads |] (Nx.matmul x w))
  in
  let loss (x, (wq, wk, wv, wo)) =
    let q = split wq x and k = split wk x and v = split wv x in
    let s =
      Nx.contract Nx.float32 (Nx.Pattern.v "r h q d, r h k d -> r h q k | d") q k
    in
    let s = Nx.add (Nx.mul_s s 0.25) (Nx.reshape [| r; 1; 1; t |] bias) in
    let bound b = Nx.reshape [| r; 1; t; 1 |] b in
    let visible =
      Nx.logical_and (Nx.greater_equal col (bound start)) (Nx.less col (bound stop))
    in
    let p = Nx.mul (Nx.softmax ~where:visible s) keep in
    let o =
      Nx.contract Nx.float32 (Nx.Pattern.v "r h q k, r h k d -> r h q d | k") p v
    in
    Nx.sum (Nx.matmul (Nx.reshape [| r; t; dm |] (Nx.swapaxes 1 2 o)) wo)
  in
  let w4 = Nx.Ptree.(quad tensor tensor tensor tensor) in
  let p = Nx.Ptree.(pair tensor w4) in
  let g = Rune.grad p loss in
  let g = if jit then Rune.jit Nx.Ptree.(p @-> returns p) g else g in
  let gx, (gq, _, _, _) = g (x, (w 2, w 3, w 4, w 5)) in
  (Nx.to_array (host gx), Nx.to_array (host gq))

let test_dense_grad () =
  let _, s = on_gpu () in
  let gx, gq = dense_grad s ~jit:true in
  let ex, eq = dense_grad (module Nx.Host) ~jit:false in
  let close name a e =
    let m = Array.fold_left (fun m v -> Float.max m (Float.abs v)) 0. e in
    let d = ref 0. in
    Array.iteri (fun i v -> d := Float.max !d (Float.abs (v -. e.(i)))) a;
    less ~msg:name float_exact ~than:(1e-4 *. m) !d
  in
  close "x" gx ex;
  close "wq" gq eq

(* A projection of 16 by 256 rows into attention's 4 heads of 110, its output
   laid out by head: the upcasts take the heads whole and the width, which the
   same operands read, takes their second upcast, padded to its threads.
   Within the bound of a sum of [k] terms. *)
let test_heads_projection () =
  let _, (module S) = on_gpu () in
  let b = 16 and t = 256 and k = 52 and h = 4 and d = 110 in
  let draw seed s =
    let x = Nx.Rng.uniform ~key:(Nx.Rng.key seed) Nx.float32 s in
    Nx.sub_s (Nx.mul_s x 4.) 2.
  in
  let x = draw 1 [| b; t; k |] and w = draw 2 [| k; h; d |] in
  let heads = Nx.Pattern.v "b t k, k h d -> b h t d | k" in
  let y =
    Rune.jit
      Nx.Ptree.(tensor @-> tensor @-> returns tensor)
      (Nx.contract Nx.float32 heads)
      (Nx.place S.on x) (Nx.place S.on w)
  in
  let y = Nx.to_array (Nx.cast Nx.float64 (host y)) in
  let f x = Nx.cast Nx.float64 x in
  let exact = Nx.to_array (Nx.contract Nx.float64 heads (f x) (f w)) in
  let mass =
    Nx.to_array (Nx.contract Nx.float64 heads (Nx.abs (f x)) (Nx.abs (f w)))
  in
  let u = 0x1p-24 in
  Array.iteri
    (fun i z ->
      let e = product (k + 1) *. mass.(i) in
      less ~msg:(string_of_int i) float_exact
        ~than:(e +. (u *. (Float.abs z +. e)))
        (Float.abs (y.(i) -. z)))
    exact

(* A routed expert at a prompt of 12 tokens: each token goes through the
   experts its logits rank first, 4 of 32, as gpt-oss-20b's do, each expert
   an MXFP4 projection up, a nonlinearity and an MXFP4 projection down. 48
   positions over 32 segments are each a block of their own. [Nx.top_k] gives
   every id in range, so every position reads its token's own row: each kernel
   that sums a product reads accesses alone, a single reduction, and spreads
   it across a workgroup's threads, as a matrix-vector product streams its
   matrix. The optimisations read are those its program applied, replayed on
   the kernel as the compiler schedules it. The result is the host's eager
   one. *)
let experts = 32
and per_token = 4
and tokens = 12
and width = 512
and hidden = 1024

module K = Tolk.Postrange.Scheduler

(* Whether [ast] sums a product. *)
let sums_product ast ren =
  List.exists
    (fun r ->
      match Tolk.Ops.arg r with
      | Reduce { op = Add; _ } -> Tolk.Ops.op (Tolk.Ops.nth r 0) = Mul
      | _ -> false)
    (K.reduceops (K.v ast ren))

(* The scheduler of [k]'s sink after the passes that precede optimisation,
   with the optimisations [k]'s program applied, through the hook a search
   takes. A kernel that reaches no search never calls the hook, and fails
   here by name. *)
let optimised (k : Tolk.Realize.kernel) =
  let chosen = ref None in
  let replay _ s =
    List.iter
      (fun o ->
        match K.apply_opt s o with
        | Ok _ -> ()
        | Error e -> failf "%s: %a: %s" k.name Tolk.Opt.pp o e)
      k.opts;
    chosen := Some s;
    s
  in
  let ast = k.ast and ren = k.renderer in
  let ast =
    match Tolk.Ops.arg ast with
    | Kernel i -> Tolk.Ops.replace ast ~arg:(Kernel { i with beam = 1 })
    | _ -> failf "%a is not a kernel" Tolk.Ops.pp ast
  in
  ignore (Tolk.Codegen.full_rewrite_to_sink ~beam:replay ast ren);
  match !chosen with
  | Some s -> s
  | None -> failf "%s: no search was handed the kernel to replay" k.name

(* Each expert's [outputs; inputs] matrix of random MXFP4 codes, at scales of
   one, on the GPU. *)
let experts_matrix (type d) (on : d Nx.Placement.t) ~key ~outputs ~inputs =
  let codes =
    Nx.reshape
      [| experts; outputs; inputs / 2 |]
      (Nx.bitcast Nx.uint8
         (Nx.Rng.bits ~key:(Nx.Rng.key key) [| experts; outputs; inputs / 8 |]))
  in
  let scales =
    Nx.copy (Nx.full Nx.uint8 [| experts; outputs; inputs / 32 |] 127)
  in
  Nx_quant.place on (Nx_quant.mxfp4 ~scales codes)

(* The routed expert of [up] and [down]. *)
let routed up down logits x =
  let expert owners rows =
    let apply w x = Nx_quant.apply (Nx_quant.take ~axis:0 owners w) x in
    apply down (Nx.abs (apply up rows))
  in
  let _, ids = Nx.top_k ~k:per_token logits in
  Nx.map_segments ~segments:experts ids expert (Nx.unsqueeze ~axes:[ -2 ] x)

let test_routed_prompt () =
  let _, (module S) = on_gpu () in
  let up = experts_matrix Nx.Host.on ~key:1 ~outputs:hidden ~inputs:width in
  let down = experts_matrix Nx.Host.on ~key:2 ~outputs:width ~inputs:hidden in
  let f =
    Rune.jit
      Nx.Ptree.(tensor @-> tensor @-> returns tensor)
      (routed (Nx_quant.place S.on up) (Nx_quant.place S.on down))
  in
  (* Distinct logits in each row: 13 is prime to 32. *)
  let logits =
    Nx.init Nx.float32 [| tokens; experts |] (fun i ->
        float_of_int (((i.(0) * 7) + (i.(1) * 13)) mod experts))
  in
  let x = Nx.copy (Nx.full Nx.float32 [| tokens; width |] 0.5) in
  let eager = Nx.to_array (routed up down logits x) in
  let y, kernels =
    Tolk.Realize.kernels (fun () ->
        Nx.to_array (host (f (Nx.place S.on logits) (Nx.place S.on x))))
  in
  (* The sums run in another order: each element within 1e-4 of the largest
     result, some 1.6 times float32's bound on a sum of 1024 terms. *)
  let largest = Array.fold_left (fun m v -> Float.max m (Float.abs v)) 0. eager in
  less ~msg:"the largest result" float_exact ~than:largest 0.;
  Array.iteri
    (fun i e ->
      less ~msg:(Printf.sprintf "element %d" i) float_exact
        ~than:(1e-4 *. largest)
        (Float.abs (y.(i) -. e)))
    eager;
  let products =
    List.filter
      (fun (k : Tolk.Realize.kernel) -> sums_product k.ast k.renderer)
      kernels
  in
  equal ~msg:"kernels that sum a product" int 2 (List.length products);
  List.iter
    (fun (k : Tolk.Realize.kernel) ->
      let o = optimised k in
      equal ~msg:(k.name ^ ": reductions") int 1 (List.length (K.reduceops o));
      not_equal ~msg:(k.name ^ ": reduce axes across threads") (list int) []
        (K.axes_of ~reduce:true o [ Local ]))
    products

(* Quantised products

   [Nx_quant.apply] by one row of [x], as a decode step multiplies each
   projection, and by 512, as a prompt does: compiled on the GPU, eagerly and
   compiled on the host, each within {!Nx.contract}'s bound of the float64
   reference product (Quants.reference). *)

type quant = { format : Quants.format; m : int; seed : int }

let quant_k = 512

let pp_quant ppf c =
  Format.fprintf ppf "%s by %d rows, seed %d" (Quants.name c.format) c.m c.seed

let quant_gen =
  let open Gen in
  (let+ format = of_list Quants.formats
   and+ m = of_list [ 1; 512 ]
   and+ seed = int_range 0 1000 in
   { format; m; seed })
  |> with_pp pp_quant

let quant_examples =
  List.concat_map
    (fun format -> List.map (fun m -> { format; m; seed = 3 }) [ 1; 512 ])
    Quants.formats

let quant_law c =
  let _, (module S) = on_gpu () in
  cover (Printf.sprintf "%s by %d rows" (Quants.name c.format) c.m) true;
  let blocks = quant_k / Quants.block_values c.format in
  let w = Quants.weight c.format ~rows:64 ~blocks in
  let x = value ~seed:c.seed [| c.m; quant_k |] in
  let exact, side = Quants.reference x w in
  let near msg y =
    let e = Nx.to_array exact and y = Nx.to_array (host y) in
    let b = Nx.to_array side in
    Array.iteri
      (fun i e ->
        less
          ~msg:(Printf.sprintf "%s at %d" msg i)
          float_exact ~than:(b.(i) +. Float.min_float)
          (Float.abs (y.(i) -. e)))
      e
  in
  let compiled w x = Rune.jit' (Nx_quant.apply w) x in
  near "eager" (Nx_quant.apply w x);
  near "compiled on the host" (compiled w x);
  near "compiled on the GPU"
    (compiled (Nx_quant.place S.on w) (Nx.place S.on x))

(* A Value scale on the first operand, batch axes and both of a Prim Contract
   in the partial form, within the contraction's bound of eager's. *)
let partial_side_a () =
  let _, (module S) = on_gpu () in
  let n = 64 and m = 64 and blocks = 8 and c = 32 in
  let s = [| 2; n; blocks; c |] in
  let codes =
    Nx.cast Nx.int8
      (Nx.mul_s (Nx.sub_s (Nx.Rng.uniform Nx.float32 s) 0.5) 200.)
  in
  let scale =
    Nx.cast Nx.float16
      (Nx.add_s (Nx.mul_s (Nx.Rng.uniform Nx.float32 [| 2; n; blocks; 1 |]) 1.9) 0.1)
  in
  let x = Nx.cast Nx.bfloat16 (Nx.Rng.uniform Nx.float32 [| 2; m; blocks; c |]) in
  let contract ~acc codes scale x =
    Nx.Prim.eval ~by:"partial_side_a"
      (Contract
         {
           batch = Iarray.of_list [ (0, 0) ];
           contracting = Iarray.of_list [ (2, 2); (3, 3) ];
           acc = Nx.Dtype.Any acc;
           out = acc;
           a = codes;
           b = x;
           init = None;
           a_scale = Some (Value scale);
           b_scale = None;
           segments = None;
           sum = false;
         })
  in
  let f c d x = contract ~acc:Nx.float32 c d x in
  let y =
    host
      (Rune.jit
         Nx.Ptree.(tensor @-> tensor @-> tensor @-> returns tensor)
         f (Nx.place S.on codes) (Nx.place S.on scale) (Nx.place S.on x))
  in
  let r = f codes scale x in
  equal ~msg:"shape" (array int) [| 2; n; m |] (Nx.shape y);
  let sum =
    contract ~acc:Nx.float64 (Nx.abs (Nx.cast Nx.float64 codes))
      (Nx.abs (Nx.cast Nx.float64 scale))
      (Nx.abs (Nx.cast Nx.float64 x))
  in
  let products = float_of_int (blocks * c) in
  let u = 0x1p-24 in
  let gamma = (((2. *. products) +. 2.) *. u) /. (1. -. (((2. *. products) +. 2.) *. u)) in
  let y = Nx.to_array y and r = Nx.to_array r and sum = Nx.to_array sum in
  Array.iteri
    (fun i e ->
      less ~msg:(Printf.sprintf "at %d" i) float_exact
        ~than:((2. *. gamma *. sum.(i)) +. Float.min_float)
        (Float.abs (e -. r.(i))))
    y

(* A contraction over two axes whose operand does not read the outer one, its
   co-operand computed: the core's trips are the inner axis, its operand's
   tile is re-copied at each trip of the outer axis, and each output is
   within the contraction's bound of eager's. *)
let core_under_a_reduce () =
  let _, (module S) = on_gpu () in
  let m = 64 and h = 4 and k = 64 and n = 64 in
  let a =
    Nx.broadcast_to [| m; h; k |]
      (Nx.cast Nx.bfloat16 (Nx.Rng.uniform Nx.float32 [| m; 1; k |]))
  in
  let b = Nx.cast Nx.bfloat16 (Nx.Rng.uniform Nx.float32 [| h; n; k |]) in
  let scale = Nx.Rng.uniform Nx.float32 [| h; 1; 1 |] in
  let p = Nx.Pattern.v "i h k, h j k -> i j | h k" in
  let f a b s =
    Nx.contract Nx.float32 p a
      (Nx.cast Nx.bfloat16 (Nx.mul (Nx.cast Nx.float32 b) s))
  in
  let y =
    host
      (Rune.jit
         Nx.Ptree.(tensor @-> tensor @-> tensor @-> returns tensor)
         f (Nx.place S.on a) (Nx.place S.on b) (Nx.place S.on scale))
  in
  let r = f a b scale in
  let sum =
    Nx.contract Nx.float64 p (Nx.abs (Nx.cast Nx.float64 a))
      (Nx.abs (Nx.cast Nx.float64 (Nx.cast Nx.bfloat16 (Nx.mul (Nx.cast Nx.float32 b) scale))))
  in
  let products = float_of_int (h * k) in
  let u = 0x1p-24 in
  let gamma = (((2. *. products) +. 2.) *. u) /. (1. -. (((2. *. products) +. 2.) *. u)) in
  let y = Nx.to_array y and r = Nx.to_array r and sum = Nx.to_array sum in
  Array.iteri
    (fun i e ->
      less ~msg:(Printf.sprintf "at %d" i) float_exact
        ~than:((2. *. gamma *. sum.(i)) +. Float.min_float)
        (Float.abs (e -. r.(i))))
    y

(* A Q8_0 product of 512 rows on the GPU multiplies each block's partial sum by
   its scale ({!Tolk.Heuristic.core_fits}): a NaN scale makes every output that
   reads its block NaN, and a weight of no blocks sums no term. *)

let q8_0_block = 34

let q8_0_blocks rows blocks scale =
  let bytes = rows * blocks * q8_0_block in
  let b =
    Array.init bytes (fun i ->
        if i mod q8_0_block = 0 then fst scale
        else if i mod q8_0_block = 1 then snd scale
        else (i mod 200) + 1)
  in
  Nx_quant.q8_0
    (Nx.create Nx.uint8 [| rows; blocks * q8_0_block |] b)

let q8_0_special =
  let unit_scale = (0x00, 0x3c) in
  let nan_scale = (0x00, 0x7e) in
  [
    test "a NaN scale reaches exactly the outputs that read its block, the \
          others eager's bits"
      (fun () ->
        let _, (module S) = on_gpu () in
        let rows = 64 and blocks = 16 in
        (* Scales of 1/2, 1, 2 and 4 that vary along the rows and the blocks,
           NaN at row 0's block 0 and row 5's block 3; codes from -4 to 4 that
           vary along a row and a block, against integers from -2 to 2 that
           vary along a row and a column. Each sum is exact, so an output read
           with another layout of the codes, the scales or the rows differs
           from eager's. *)
        let nan_block r b = (r = 0 && b = 0) || (r = 5 && b = 3) in
        let high = [| 0x38; 0x3c; 0x40; 0x44 |] in
        let bytes =
          Array.init (rows * blocks * q8_0_block) (fun i ->
              let blk = i / q8_0_block in
              let r = blk / blocks and b = blk mod blocks in
              match i mod q8_0_block with
              | 0 -> if nan_block r b then fst nan_scale else 0x00
              | 1 ->
                  if nan_block r b then snd nan_scale
                  else high.(((r * 3) + b) mod 4)
              | _ -> ((((i * 7) + (blk * 3)) mod 9) - 4) land 0xff)
        in
        let w =
          Nx_quant.q8_0
            (Nx.create Nx.uint8 [| rows; blocks * q8_0_block |] bytes)
        in
        let k = 32 * blocks in
        let x =
          Nx.cast Nx.bfloat16
            (Nx.init Nx.float32 [| 512; k |] (fun ix ->
                 Float.of_int ((((ix.(0) * 2) + (ix.(1) * 3)) mod 5) - 2)))
        in
        let y =
          Nx.to_array
            (host
               (Rune.jit'
                  (Nx_quant.apply (Nx_quant.place S.on w))
                  (Nx.place S.on x)))
        and e = Nx.to_array (Nx_quant.apply w x) in
        Array.iteri
          (fun i v ->
            let o = i mod rows in
            let want = if o = 0 || o = 5 then Float.nan else e.(i) in
            equal ~msg:(Printf.sprintf "output %d" i) float_exact want v)
          y);
    test "a weight of no blocks sums no term" (fun () ->
        let _, (module S) = on_gpu () in
        let w = q8_0_blocks 64 0 unit_scale in
        let x = Nx.full Nx.bfloat16 [| 512; 0 |] 1.0 in
        let y =
          host
            (Rune.jit' (Nx_quant.apply (Nx_quant.place S.on w)) (Nx.place S.on x))
        in
        equal ~msg:"shape" (array int) [| 512; 64 |] (Nx.shape y);
        Array.iter (fun v -> equal ~msg:"+0" float_exact 0. v) (Nx.to_array y));
  ]

(* Exp at rune.mli's bound for a device whose exp2 is within 4 ulps, as
   CUDA's exp2f is within 2 (CUDA C++ Programming Guide, Mathematical
   Functions), against libm's float64 exp rounded once (Exps). Measured over
   every float32 on an RTX 5000 Ada: 3 ulps at most, at 18 operands. Every
   float16 and bfloat16 code is within one ulp. *)
let exp_law codes =
  let d, s = on_gpu () in
  Exps.law s ~ulps:9 ~flushes:(flushes d) codes

let exp16 dt () =
  let d, s = on_gpu () in
  Exps.law16 s ~flushes:(flushes d) dt

(* Multiply-adds of narrow floats: fma at a float narrower than float32,
   compiled on the GPU, is its exact value rounded once, as Exact's integers
   round it. Where the GPU flushes float32's subnormals, rune.mli lets an
   element differ whose operand, product or exact value is one. *)

type fma_dtype = Fma : (float, 's) Nx.dtype -> fma_dtype

let fma_dtypes =
  Nx.Dtype.
    [
      Fma Float16; Fma Bfloat16; Fma Float8_e4m3fn; Fma Float8_e5m2;
      Fma Float4_e2m1fn;
    ]

let narrow_fma (Fma dt, seed) =
  let d, (module S : Nx.Devices) = on_gpu () in
  let f = Exact.format dt in
  let a, b, c = Kernels_cases.Narrow_fma.operands f seed in
  let subnormal x = x <> 0. && Float.abs x < 0x1p-126 in
  let kept i =
    not
      (flushes d
      && List.exists subnormal
           [ a.(i); b.(i); c.(i); a.(i) *. b.(i); Exact.exact_fma a.(i) b.(i) c.(i) ])
  in
  let keep xs = Array.of_list (List.filteri (fun i _ -> kept i) (Array.to_list xs)) in
  let a = keep a and b = keep b and c = keep c in
  let v x = Nx.place S.on (Nx.create dt [| Array.length x |] x) in
  let fma =
    Rune.jit Nx.Ptree.(tensor @-> tensor @-> tensor @-> returns tensor) Nx.fma
  in
  let y = Nx.place Nx.Host.on (fma (v a) (v b) (v c)) in
  let got = Nx.to_array (Nx.cast Nx.float64 y) in
  equal (list string) [] (Kernels_cases.Narrow_fma.misses f a b c got)

let () =
  exit
    (run "Rune.jit on a GPU"
       [
         group "scatters"
           [
             prop ~count:4 ~examples:Scatters.examples
               "a scatter compiles to eager's values"
               (Gen.with_pp Scatters.pp Scatters.gen)
               law;
             Windtrap.cases ~id:(Format.asprintf "%a" Unreached.pp)
               "a scatter keeps the bits of each element no update lands on"
               Unreached.cases unreached;
             test "a numbering of repeated keys compiles to eager's counts"
               repeated_keys;
             prop ~count:8 ~examples:Row_sets.examples
               "a Set scatter of rows compiles to eager's bits"
               (Gen.with_pp Row_sets.pp Row_sets.gen)
               row_set;
           ];
         group "integers"
           [
             prop ~count:20 ~examples:Wraps.examples
               "integer arithmetic at its type's edges gives eager's values"
               (Gen.with_pp Wraps.pp Wraps.gen)
               (fun c -> Wraps.law (snd (on_gpu ())) c);
             prop ~count:20 ~examples:Int_casts.examples
               "an integer cast to a float gives eager's bits"
               (Gen.with_pp Int_casts.pp Int_casts.gen)
               int_casts;
           ];
         group "running operations"
           [
             prop ~count:4 ~examples:Prefixes.examples
               "a running operation's prefix is the prefix of its result"
               (Gen.with_pp Prefixes.pp Prefixes.gen)
               prefixes;
           ];
         Windtrap.cases
           "an argmax or argmin over a length 16 does not divide is nx's \
            position"
           ~id:(fun (Extreme (n, dt, peak)) ->
             Printf.sprintf "%d %s, %s" n (Nx.Dtype.name dt) (peak_name peak))
           extremes argmax_extreme;
         Windtrap.cases
           "a sum and a product over a length 16 does not divide are exact"
           ~id:string_of_int
           [ 33; 47; 1571; 4099; 128 * 257; 151936; 201088 ]
           folds;
         group "top k"
           [
             prop ~count:24 ~examples:top_examples
               "a first k compiles to eager's values and positions"
               (Gen.with_pp pp_top top_gen)
               top_k;
           ];
         group "assemblies"
           [
             test "a stack of computed rows compiles to eager's bits"
               test_stack;
             test "a range gather and its gradient compile to eager's bits"
               test_ranges;
             prop ~count:8 ~examples:Assemblies.examples
               "an assembly compiles to eager's bits"
               (Gen.with_pp Assemblies.pp Assemblies.gen)
               assembly;
           ];
         group "exp"
           [
             prop ~count:24 "exp within 9 ulps at float32" Exps.gen exp_law;
             test "exp of every float16 code within one ulp"
               (exp16 Nx.float16);
             test "exp of every bfloat16 code within one ulp"
               (exp16 Nx.bfloat16);
           ];
         group "multiply-adds"
           [
             Windtrap.cases "fma rounds once at floats narrower than float32"
               ~id:(fun (Fma dt, _) -> Nx.Dtype.name dt)
               (List.map (fun n -> (n, 1)) fma_dtypes)
               narrow_fma;
           ];
         group "16-bit casts"
           [
             test "float16 gives eager's bits" (casts16 Nx.float16);
             test "bfloat16 gives eager's bits" (casts16 Nx.bfloat16);
           ];
         group "products"
           [
             prop ~count:24
               "a float16 or bfloat16 matmul is within its bound of the exact \
                product"
               products_gen product_law;
             prop ~count:24
               ~examples:
                 (List.map
                    (fun s -> (Narrow (Nx.float32, 0x1p-24, 0x1p-149), s, 7))
                    padded_products)
               "a float32 matmul is within its bound of the exact product"
               float32_products_gen float32_law;
             test "sixteen terms under float32's unit hold the bound"
               test_sixteen_terms;
             test
               "a blocked product padded to its threads, plus a bias, is \
                within its bound"
               test_padded_bias;
             test
               "a projection into heads, blocked over their width, is within \
                its bound"
               test_heads_projection;
             test
               "a dense attention step's gradient is the host's within \
                float32's rounding"
               test_dense_grad;
             Windtrap.cases
               "an MXFP4 product is within its bound of the exact one"
               ~id:(Format.asprintf "%a" Mxfp4_products.pp)
               Mxfp4_products.cases
               (fun c ->
                 let d, s = on_gpu () in
                 Mxfp4_products.law ~flushes:(flushes d) s c);
             test
               "a routed MXFP4 expert of 12 tokens streams its matrices across \
                threads"
               test_routed_prompt;
           ];
         (* Each case compiles programs of its own shapes. The examples reach
            every cover, each mask and each dtype, on every run; four draws
            add shapes of their own. *)
         group "attention"
           [
             prop ~count:4
               ~examples:(Attentions.examples attention_dtypes)
               "jit holds nx's bound against a float64 reference"
               (Gen.with_pp Attentions.pp (Attentions.gen attention_dtypes))
               attention;
             prop ~count:4 ~examples:grad_examples
               "its compiled gradient holds rune's bound against a float64 \
                reference"
               (Gen.with_pp Attentions.pp (Attentions.gen attention_dtypes))
               (attention_grad ~vmap:false);
             prop ~count:4
               ~examples:(List.map mappable grad_examples)
               "its gradient mapped by vmap holds rune's bound against a \
                float64 reference"
               (Gen.with_pp Attentions.pp
                  (Gen.map mappable (Attentions.gen attention_dtypes)))
               (attention_grad ~vmap:true);
             (* float32 on the SIMT kernel; the narrow floats on the target's
                tensor core where it has one that chains, else on the SIMT
                kernel. *)
             Windtrap.cases "a sum of zeros is +0 at rows of one run of keys"
               ~id:(fun (Nx.Dtype.Any dt) -> Nx.Dtype.name dt)
               Nx.Dtype.[ Any Nx.float32; Any Nx.bfloat16; Any Nx.float16 ]
               (fun dt -> Attentions.zero_sums (snd (on_gpu ())) dt);
             test "forward mode of a summed loss holds its bound" attention_jvp;
             prop ~count:4 ~examples:Paged_attentions.examples
               "paged keys and values give dense ones' bits"
               (Gen.with_pp Paged_attentions.pp Paged_attentions.gen)
               paged_attention;
           ];
         group "a reduction another reads"
           [
             test "holds its bound" sum_of_contraction;
             test "over a broadcast stores no broadcast" sum_of_broadcast;
             Windtrap.cases "read by a reduction adds one kernel, its own"
               ~id:(fun ((r, _), (p, _)) -> r ^ ", " ^ p)
               (List.concat_map
                  (fun r -> List.map (fun p -> (r, p)) reading_programs)
                  reductions)
               (fun ((_, r), (_, p)) -> one_kernel r p);
             test "of 128 x 128 x 64 terms stays inside the one that reads it"
               (inside [| 128; 128; 64 |]);
             test "of 32 terms an element stays inside the one that reads it"
               (inside [| 256; 256; 32 |]);
             Windtrap.cases "returned too is stored once"
               ~id:(fun (r, _) -> r)
               reductions
               (fun (_, r) -> returned r ());
           ];
         group "linear algebra"
           [
             test "eigh of matrices that converge apart holds its bound"
               test_eigh;
           ];
         group "checks"
           [
             test "a call's check raises at the next wait, not at the call"
               test_deferred;
           ];
         group "memory"
           [
             test
               "a split program holds at its return at most its result and \
                its plan"
               test_split_memory;
           ];
         group "a detour" [ test "onto the GPU's set and back" detour ];
         group "switches"
           [
             test "over a cut value, the index whole on each"
               (switch_agrees ~cut:(Split 0) ~index:On ~taken:1l [| 8 |]);
             test "the index on the first device"
               (switch_agrees ~cut:(Split 0) ~index:(Dev 0) ~taken:0l [| 8 |]);
             test "the index on another device"
               (switch_agrees ~cut:(Split 0) ~index:(Dev 1) ~taken:1l [| 8 |]);
             test "a value on one device apart"
               (switch_agrees ~cut:(Dev 1) ~index:(Dev 0) ~taken:1l [| 8 |]);
             test "an empty value"
               (switch_agrees ~cut:(Split 0) ~index:On ~taken:0l [| 0 |]);
             test "an index out of range"
               (switch_agrees ~cut:(Split 0) ~index:On ~taken:7l [| 8 |]);
             test "a mesh"
               (switch_agrees ~devices:4 ~cut:(Grid true) ~index:(Dev 3)
                  ~taken:1l [| 4; 4 |]);
           ];
         group "emissions and declines over several devices"
           [
             test "an emission compiles, offered once" (fun () ->
                 on_mixed { law = emission });
             test "a declined program raises at the call, named" (fun () ->
                 on_mixed { law = declined });
           ];
         group "checks"
           [
             test "a first false on another device, data cut alike"
               (check_fails ~flags:(Split 0) ~data:(Split 0) [| 8 |] [ 6 ]);
             test "data on one device, a flag on the other"
               (check_fails ~flags:(Split 0) ~data:(Dev 1) [| 6 |] [ 1 ]);
             test "data whole on each device"
               (check_fails ~flags:(Split 0) ~data:On [| 6 |] [ 4 ]);
             test "passing"
               (check_fails ~flags:(Split 0) ~data:(Split 0) [| 6 |] []);
             test "a mesh, first false in C order in another tile"
               (check_fails ~devices:4 ~flags:(Grid true) ~data:(Grid true)
                  [| 4; 4 |] [ 3; 4; 10 ]);
             test "a mesh in the other axis order, data apart"
               (check_fails ~devices:4 ~flags:(Grid false) ~data:(Grid true)
                  [| 4; 4 |] [ 7 ]);
             test "a mesh, data on one device"
               (check_fails ~devices:4 ~flags:(Grid true) ~data:(Dev 2)
                  [| 4; 4 |] [ 7 ]);
           ];
         group "custom rules, remats and totals"
           [
             test "a custom rule, a remat and a gradient" (fun () ->
                 agreeing { agree = customs }
                   (Testable.pair bits_t bits_t));
             test "a switch inside a scan's step and nested" (fun () ->
                 agreeing { agree = switch_scan }
                   (Testable.pair bits_t Testable.unit));
             test "additions to a total" (fun () ->
                 agreeing { agree = total } (Testable.pair bits_t bits_t));
           ];
         group "a GPU beside a software device"
           (List.map
              (fun c -> test c.name (over ~devices:2 ~first:0 c))
              cases);
         group "a software device beside a GPU"
           (List.map
              (fun c -> test c.name (over ~devices:2 ~first:1 c))
              cases);
         group "a mesh of a GPU first and three software devices"
           (List.map
              (fun c -> test c.name (over ~devices:4 ~first:0 c))
              mesh_cases);
         group "a mesh of a software device, a GPU and two software devices"
           (List.map
              (fun c -> test c.name (over ~devices:4 ~first:1 c))
              mesh_cases);
         group "a contraction on matrix units holds its bound" unit_cases;
         group "quantised products"
           ([
              prop ~count:10 ~examples:quant_examples
                "one row and 512 rows hold the bound, eager and compiled"
                quant_gen quant_law;
            ]
           @ q8_0_special
           @ [
               test "a Value scale on the first operand in the partial form"
                 partial_side_a;
               test "a core under a reduce re-reads a tile of each outer trip"
                 core_under_a_reduce;
             ]);
         group "keys"
           [
             prop ~count:20
               ~examples:[ 0; -1; max_int; min_int ]
               "a key split under jit has the eager split's words" Gen.int
               law_split;
           ];
       ])
