(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_array
open Frontend

(* One splittable Threefry-2x32 generator with two front-ends over it: the
   explicit samplers below take a key and are pure functions of it
   (order-independent, transform-safe), and the keyless [rand]/[randn]/… draw a
   fresh subkey from the ambient scope and call the same samplers. A key is a
   [|2|] int32 tensor and a batch of keys adds leading axes; [Nx] makes the type
   private, so only this module builds one, and it coerces back to the tensor
   that flows wherever tensors go — a structure's leaf, a jit input, a mapped
   axis. *)
module Rng = struct
  type key = (int32, int32_elt) t

  let shape_string s =
    String.concat "; " (Array.to_list (Array.map string_of_int s))

  (* A key's type guarantees its last axis holds the two words, so the only
     wrong shape a sampler can be given is a batch: [split_batch] outside the
     [vmap] whose lanes it was built for. *)
  let check_key name k =
    if shape k <> [| 2 |] then
      invalid_arg
        (Printf.sprintf
           "Nx.Rng.%s: expected one key, got a batch of keys of shape [%s]; \
            map over a batch with Rune.vmap"
           name
           (shape_string (shape k)))

  let of_tensor t =
    let s = shape t in
    let r = Array.length s in
    if r = 0 || s.(r - 1) <> 2 then
      invalid_arg
        (Printf.sprintf
           "Nx.Rng.of_tensor: a key is an int32 tensor of shape [2], and a \
            batch of keys one of shape [...; 2]; got shape [%s]"
           (shape_string s));
    t

  let check_shape name shape' =
    if Array.exists (fun d -> d < 0) shape' then
      invalid_arg
        (Printf.sprintf
           "Nx.Rng.%s: invalid shape [%s], dimensions must be non-negative" name
           (shape_string shape'))

  (* The low and high 32-bit words of [seed] as the key's two lanes. *)
  let key ctx seed =
    create ctx Nx_dtype.int32 [| 2 |]
      [| Int32.of_int (seed asr 32); Int32.of_int seed |]

  (* The [n; 2] counters of [n] blocks. Block [k] holds the words [(2k, 2k + 1)
     mod 2^32], which repeat from 2^31 blocks on, so in a draw of more than 2^31
     blocks the second word of block [k = q * 2^31 + r] is xored with [q], read
     from either word's high half: the pair [(2r, (2r + 1) xor q)] gives back
     [r] and [q], and the counters stay distinct up to 2^63 blocks. The term is
     zero below 2^31 blocks, where it is not computed. *)
  let counters ctx n =
    let w = arange ctx Nx_dtype.int64 0 (2 * n) 1 in
    let w =
      if n <= 1 lsl 31 then w
      else
        bitwise_xor w
          (mul (rshift w 32) (bitwise_and w (scalar ctx Nx_dtype.int64 1L)))
    in
    reshape [| n; 2 |] (cast Nx_dtype.int32 w)

  (* One Threefry application over [n] independent blocks: the key broadcast
     across the rows of the counters. The broadcast is a stride-0 view; the
     kernel reads the key through its strides, so materialising it would cost as
     many bytes as the draw. *)
  let blocks name k n =
    check_key name k;
    let kb = broadcast_to [| n; 2 |] (reshape [| 1; 2 |] k) in
    B.threefry kb (counters (Value.context k) n)

  let split ?(n = 2) k =
    if n < 1 then invalid_arg "Nx.Rng.split: n must be at least 1";
    let bits = blocks "split" k n in
    Array.init n (fun i -> contiguous (slice [ I i ] bits))

  (* Row [i] of the batch is [split ~n k]'s key [i]: both are row [i] of one
     Threefry application. *)
  let split_batch ~n k =
    if n < 1 then invalid_arg "Nx.Rng.split_batch: n must be at least 1";
    blocks "split_batch" k n

  let fold_in k data =
    check_key "fold_in" k;
    let ctx = Value.context k in
    let ctr =
      create ctx Nx_dtype.int32 [| 2 |]
        [| Int32.of_int (data asr 32); Int32.of_int data |]
    in
    B.threefry (contiguous k) ctr

  (* [fold_in] of a scalar index tensor rather than a host int. The host form
     counts with [(idx asr 32, idx)], whose high word is the sign of an index
     that fits in 32 bits: the counter is [(0, 1)] scaled by [idx] plus [(1, 0)]
     scaled by that sign. A batched [idx] (under vmap) therefore yields a
     batched, per-lane key, and a traced one a traced key. *)
  let fold_in_tensor k idx =
    check_key "fold_in_tensor" k;
    let ctx = Value.context k in
    let word v = create ctx Nx_dtype.int32 [| 2 |] v in
    let sign =
      neg (cast Nx_dtype.int32 (cmplt idx (scalar ctx Nx_dtype.int32 0l)))
    in
    let ctr =
      add (mul (word [| 0l; 1l |]) idx) (mul (word [| 1l; 0l |]) sign)
    in
    B.threefry (contiguous k) ctr

  (* Significand width of [dtype], the leading bit included. *)
  let significand_bits : type b. (float, b) Nx_dtype.t -> int = function
    | Float8_e5m2 -> 3
    | Float8_e4m3 -> 4
    | BFloat16 -> 8
    | Float16 -> 11
    | Float32 -> 24
    | Float64 -> 53

  (* Parameters are tensors and the draw has their shape. The samplers work at
     float64 for float64 parameters and at float32 for every other float dtype,
     then return at the parameters' own dtype; [at] moves a tensor between the
     two without the copy [cast] makes when nothing changes, and [pair]
     broadcasts two parameters against each other. *)
  let at (type a b) (target : (float, a) Nx_dtype.t) (x : (float, b) t) :
      (float, a) t =
    match Nx_dtype.equal_witness (dtype x) target with
    | Some Equal -> x
    | None -> cast target x

  let pair a b =
    let target = Shape.broadcast (shape a) (shape b) in
    (broadcast_to target a, broadcast_to target b)

  (* [distinct x] is [x] with each broadcast axis, of stride 0, cut to its first
     element: the elements a check must read, one for a broadcast scalar. The
     first failing element of [x] in C order sits at index 0 on such an axis, so
     it has the same index in both. *)
  let distinct x =
    let strides = View.strides (Value.view x) in
    if not (Array.mem 0 strides) then x
    else
      let keep d n = if strides.(d) = 0 && n > 0 then (0, 1) else (0, n) in
      shrink (Array.mapi keep (shape x)) x

  (* A parameter's domain: as written in a refusal, and where a tensor's
     elements lie in it. NaN lies outside each. *)
  type 'a domain = { text : string; inside : 'a -> (bool, Nx_dtype.bool_elt) t }

  let positive =
    {
      text = "(0, inf)";
      inside = (fun x -> logical_and (cmpgt x (scalar_like x 0.0)) (isfinite x));
    }

  let non_negative =
    {
      text = "[0, inf)";
      inside = (fun x -> logical_and (cmpge x (scalar_like x 0.0)) (isfinite x));
    }

  let natural =
    { text = "[0, inf)"; inside = (fun x -> cmpge x (scalar_like x 0l)) }

  let probability =
    {
      text = "[0, 1]";
      inside =
        (fun x ->
          logical_and
            (cmpge x (scalar_like x 0.0))
            (cmple x (scalar_like x 1.0)));
    }

  let not_nan =
    { text = "[-inf, inf]"; inside = (fun x -> logical_not (isnan x)) }

  let below_infinity =
    {
      text = "[-inf, inf)";
      inside = (fun x -> cmplt x (scalar_like x Float.infinity));
    }

  (* A sampler checks each parameter against its domain with one [check] before
     it draws, so a concrete parameter raises at once and a traced one when its
     compiled call returns. The refusal names the first element outside the
     domain by its index and value. *)
  let require sampler name d x =
    let x = distinct x in
    check Ptree.tensor (d.inside x) x (fun i x ->
        let at =
          if Array.length i = 0 then ""
          else Printf.sprintf " at [%s]" (shape_string i)
        in
        Invalid_argument
          (Printf.sprintf "Nx.Rng.%s: %s%s is %s, not in %s" sampler name at
             (to_string x) d.text))

  (* Random bits -> [0, 1): keep the low [p] bits and scale them by 2^-p,
     where [p] is the destination's significand width. Both steps are exact,
     so a draw is one of the 2^p multiples of 2^-p in [0, 1 - 2^-p]: the
     interval is half-open by construction, at every dtype, rather than by a
     rounding accident.

     A Threefry word carries 32 bits, so anything up to float32 takes its [p]
     bits from one word and builds the draw in float32. float64 wants 53 and
     so consumes a whole row, 21 bits of one word and all 32 of the next; both
     parts and their combination are exact in float64. Widening a float32 draw
     instead would have left a double with 24 random bits, which is what this
     used to do. *)
  (* The words every sampler is built from. A Threefry row is two words, so
     [n] words cost ceil (n/2) rows. *)
  let bits k shape =
    check_shape "bits" shape;
    let n = array_prod shape in
    if n = 0 then zeros (Value.context k) Nx_dtype.int32 shape
    else
      reshape shape
        (shrink [| (0, n) |] (flatten (blocks "bits" k ((n + 1) / 2))))

  (* [unit k dtype shape] is a draw in [0, 1) as the expression of the
     generator's bits that makes it: a compiled program fuses it into what reads
     it, and knows its range from the mask and the scale. *)
  let unit (type b) k (dtype : (float, b) Nx_dtype.t) shape : (float, b) t =
    let ctx = Value.context k in
    let n = array_prod shape in
    if n = 0 then zeros ctx dtype shape
    else
      match dtype with
      | Nx_dtype.Float64 ->
          (* One row per draw: [(hi, lo)] contributes 21 + 32 bits. *)
          let words = blocks "uniform" k n in
          let word col =
            reshape [| n |]
              (contiguous (shrink [| (0, n); (col, col + 1) |] words))
          in
          let top =
            cast Nx_dtype.float64
              (bitwise_and (word 0) (scalar ctx Nx_dtype.int32 0x1F_FFFFl))
          in
          let bottom =
            cast Nx_dtype.float64
              (bitwise_and
                 (cast Nx_dtype.int64 (word 1))
                 (scalar ctx Nx_dtype.int64 0xFFFF_FFFFL))
          in
          let u =
            mul
              (add (mul top (scalar ctx Nx_dtype.float64 4294967296.0)) bottom)
              (scalar ctx Nx_dtype.float64 (Float.ldexp 1.0 (-53)))
          in
          reshape shape u
      | _ ->
          let p = significand_bits dtype in
          let mask = scalar ctx Nx_dtype.int32 (Int32.of_int ((1 lsl p) - 1)) in
          let u =
            mul
              (cast Nx_dtype.float32 (bitwise_and (bits k shape) mask))
              (scalar ctx Nx_dtype.float32 (Float.ldexp 1.0 (-p)))
          in
          cast dtype u

  (* A draw is a buffer of its own, as tinygrad's [rand] is: a compiled program
     that reads it more than once, such as a matmul against a drawn matrix,
     would otherwise recompute the generator for every read. *)
  let uniform k dtype shape =
    check_shape "uniform" shape;
    if array_prod shape = 0 then unit k dtype shape
    else copy (unit k dtype shape)

  (* Box-Muller: a radius from one uniform and an angle from another give two
     independent samples, r cos(2 pi u2) and r sin(2 pi u2). Both are kept, so
     [n] samples come from [n] uniforms — themselves ceil (n/2) Threefry rows.

     The transform runs at the destination's precision, not always at float32:
     building a double's normal out of float32 uniforms and widening the result
     leaves a double carrying float32 noise, which is what this used to do. [u1]
     can be exactly 0, so it is floored before the log — at 2^-p, the smallest
     positive value the uniform can take, so the floor rewrites zero and nothing
     else.

     The samples are a buffer of their own, like a uniform draw, where
     tinygrad's [randn] is not: the transform costs a logarithm, a square root
     and a cosine per sample, and a compiled matmul against a drawn matrix would
     pay them again for every row it reads, forty times the matmul itself on the
     CPU. The uniforms the transform reads once are not: fused with it, their
     range is known, so the angle 2 pi u2 is below 2 pi and its sine and cosine
     take the short argument reduction. Read from a buffer, the angle has no
     bound, and the long one runs as well. *)
  let normal (type b) k (dtype : (float, b) Nx_dtype.t) shape : (float, b) t =
    check_shape "normal" shape;
    let ctx = Value.context k in
    let n = array_prod shape in
    if n = 0 then zeros ctx dtype shape
    else
      let pairs = (n + 1) / 2 in
      let box_muller (type c) (compute : (float, c) Nx_dtype.t) =
        let u = unit k compute [| 2; pairs |] in
        let u1 = slice [ I 0 ] u and u2 = slice [ I 1 ] u in
        let smallest =
          scalar ctx compute (Float.ldexp 1.0 (-significand_bits compute))
        in
        let r =
          sqrt (mul (scalar ctx compute (-2.0)) (log (maximum u1 smallest)))
        in
        let angle = mul u2 (scalar ctx compute (2.0 *. Float.pi)) in
        let z = concatenate ~axis:0 [ mul r (cos angle); mul r (sin angle) ] in
        copy (cast dtype (reshape shape (shrink [| (0, n) |] z)))
      in
      match dtype with
      | Nx_dtype.Float64 -> box_muller Nx_dtype.float64
      | _ -> box_muller Nx_dtype.float32

  (* Gumbel(0, 1) by inverse CDF: -log (-log u). The double logarithm has a pole
     at each end of the unit interval. The draw never reaches 1, and the floor
     at 2^-p — the smallest value a uniform can take — moves the single draw
     that would land on the other pole and leaves every other alone. *)
  let gumbel (type b) k (dtype : (float, b) Nx_dtype.t) shape : (float, b) t =
    check_shape "gumbel" shape;
    let ctx = Value.context k in
    let draw (type c) (compute : (float, c) Nx_dtype.t) =
      let u = uniform k compute shape in
      let smallest =
        scalar ctx compute (Float.ldexp 1.0 (-significand_bits compute))
      in
      cast dtype (neg (log (neg (log (maximum u smallest)))))
    in
    match dtype with
    | Nx_dtype.Float64 -> draw Nx_dtype.float64
    | _ -> draw Nx_dtype.float32

  (* Exponential(1) by inverse CDF. Built from [1 - u] rather than [u]: the draw
     can be exactly 0, where a logarithm diverges, but never 1. *)
  let exponential (type b) k (dtype : (float, b) Nx_dtype.t) shape :
      (float, b) t =
    check_shape "exponential" shape;
    let ctx = Value.context k in
    let draw (type c) (compute : (float, c) Nx_dtype.t) =
      let u = uniform k compute shape in
      cast dtype (neg (log (sub (scalar ctx compute 1.0) u)))
    in
    match dtype with
    | Nx_dtype.Float64 -> draw Nx_dtype.float64
    | _ -> draw Nx_dtype.float32

  (* Marsaglia-Tsang (2000). At a concentration of 1 or more, a normal draw is
     squeezed through v = (1 + cx)^3 and accepted against a uniform; acceptance
     exceeds 98% at every concentration and rises with it. Below 1, the identity
     Gamma(a) = Gamma(a + 1) * U^(1/a) shifts into that range.

     The acceptance test is data-dependent and a trace cannot branch on data, so
     a fixed number of attempts is drawn at once and the first acceptance
     selected branchlessly. Eight rounds at 98% leave a failure probability near
     1e-14 per element, and an element failing all eight falls back to the
     distribution's mean. This is the only sampler here that is not exact: the
     price of a shape that does not depend on the draw. *)
  let gamma_rounds = 8

  (* Marsaglia-Tsang at [compute]: a draw of Gamma(boosted, 1), for [boosted]
     the concentration raised past 1 where it is below, and what the shift back
     needs: the concentration, where it is below 1, and the shift's uniform,
     floored so that its power and logarithm stay finite when it draws exactly
     zero. *)
  let marsaglia_tsang (type c) (compute : (float, c) Nx_dtype.t) k concentration
      =
    let ctx = Value.context k in
    let lit v = scalar ctx compute v in
    let tiny = lit (Float.ldexp 1.0 (-significand_bits compute)) in
    let shape = shape concentration in
    let a = at compute concentration in
    let below_one = cmplt a (lit 1.0) in
    let boosted = where below_one (add a (lit 1.0)) a in
    let d = sub boosted (lit (1.0 /. 3.0)) in
    let squeeze = recip (sqrt (mul (lit 9.0) d)) in
    let ks = split ~n:3 k in
    let attempts = Array.append [| gamma_rounds |] shape in
    let x = normal ks.(0) compute attempts in
    let u = uniform ks.(1) compute attempts in
    (* The mean of Gamma(boosted, 1) is [boosted]: the least wrong value for an
       element no round accepted. *)
    let acc = ref boosted in
    let settled = ref (cmpne boosted boosted) in
    for j = 0 to gamma_rounds - 1 do
      let xj = contiguous (slice [ I j ] x) in
      let uj = contiguous (slice [ I j ] u) in
      let t = add (lit 1.0) (mul squeeze xj) in
      let v = mul t (mul t t) in
      (* [v] can be non-positive, where the logarithm is undefined. Floor it so
         the arithmetic stays finite and let [positive] do the rejecting. *)
      let positive = cmpgt v (lit 0.0) in
      let log_v = log (maximum v tiny) in
      let bound =
        add
          (mul (lit 0.5) (mul xj xj))
          (add d (add (neg (mul d v)) (mul d log_v)))
      in
      let accept = logical_and positive (cmplt (log (maximum uj tiny)) bound) in
      let take = logical_and accept (logical_not !settled) in
      acc := where take (mul d v) !acc;
      settled := logical_or !settled accept
    done;
    let boost = maximum (uniform ks.(2) compute shape) tiny in
    (!acc, a, below_one, boost)

  (* Gamma(a) = Gamma(a + 1) * U^(1/a) below 1. *)
  let gamma (type b) k (concentration : (float, b) t) : (float, b) t =
    require "gamma" "concentration" positive concentration;
    let draw (type c) (compute : (float, c) Nx_dtype.t) =
      let acc, a, below_one, boost = marsaglia_tsang compute k concentration in
      at (dtype concentration)
        (where below_one (mul acc (pow boost (recip a))) acc)
    in
    match dtype concentration with
    | Nx_dtype.Float64 -> draw Nx_dtype.float64
    | _ -> draw Nx_dtype.float32

  (* The logarithm of a {!gamma} draw, at [compute]. Below a concentration of
     about 0.03 most float32 draws underflow to zero, where their logarithm, of
     order [log u / a], is still finite: ratios of gammas are formed from it. *)
  let log_gamma compute k concentration =
    let acc, a, below_one, boost = marsaglia_tsang compute k concentration in
    add (log acc) (where below_one (div (log boost) a) (zeros_like a))

  (* Beta(a, b) = G(a) / (G(a) + G(b)) for independent gammas of unit rate,
     formed as [1 / (1 + exp (log G(b) - log G(a)))]: at small concentrations
     both gammas can underflow to zero, where their ratio is lost but the
     difference of their logarithms is not. Both draws inherit {!gamma}'s
     bounded-rejection approximation. *)
  let beta (type b) k (a : (float, b) t) (b : (float, b) t) : (float, b) t =
    require "beta" "a" positive a;
    require "beta" "b" positive b;
    let a, b = pair a b in
    let ks = split k in
    let draw (type c) (compute : (float, c) Nx_dtype.t) =
      let d = sub (log_gamma compute ks.(1) b) (log_gamma compute ks.(0) a) in
      at (dtype a) (recip (add (scalar (Value.context k) compute 1.0) (exp d)))
    in
    match dtype a with
    | Nx_dtype.Float64 -> draw Nx_dtype.float64
    | _ -> draw Nx_dtype.float32

  (* Dirichlet: one gamma per component, normalised across the last axis in log
     space, as {!beta} is, so that every row of the result sums to one however
     small the concentrations. *)
  let dirichlet (type b) k (concentration : (float, b) t) : (float, b) t =
    let s = shape concentration in
    let nd = Array.length s in
    if nd = 0 || s.(nd - 1) < 2 then
      invalid_arg
        "Nx.Rng.dirichlet: concentration needs at least two components on its \
         last axis";
    require "dirichlet" "concentration" positive concentration;
    let axes = [ nd - 1 ] in
    let draw (type c) (compute : (float, c) Nx_dtype.t) =
      let l = log_gamma compute k concentration in
      let e = exp (sub l (max ~axes ~keepdims:true l)) in
      at (dtype concentration) (div e (sum ~axes ~keepdims:true e))
    in
    match dtype concentration with
    | Nx_dtype.Float64 -> draw Nx_dtype.float64
    | _ -> draw Nx_dtype.float32

  (* Von Mises(0, kappa) by Best and Fisher's rejection (1979) from a wrapped
     Cauchy envelope of parameter [rho]: a proposal [z = cos theta] from one
     uniform, accepted against a second through [y = kappa (s - z)], where [s =
     (1 + rho^2) / (2 rho)], with a squeeze [y (2 - y) > v] or the bound [log (y
     / v) + 1 - y >= 0]. Acceptance is lowest as [kappa] grows, near 0.66, so 20
     rounds leave 5e-10 of an element unaccepted; that element takes its last
     proposal, a draw from the envelope.

     The textbook form cancels at both ends: [rho = (tau - sqrt (2 tau)) / (2
     kappa)] with [tau = 1 + sqrt (1 + 4 kappa^2)] loses every digit as [kappa]
     goes to zero, and [z] reaches 1 at large [kappa], where [acos z] is the
     root of a difference. So the draw is written in the uniform's half angle
     [phi = pi (2 u - 1) / 2], with [D = tau + sqrt (2 tau)], [rho = 2 kappa /
     D], [1 - rho = (D - 2 kappa) / D] and [A = D (1 - rho)^2 / 4]:

     denom = (1 - rho)^2 + 4 rho cos^2 phi,

     y = A (1 + rho)^2 / denom,

     theta = 2 asin (|sin phi| (1 - rho) / sqrt denom), signed as phi,

     where [D - 2 kappa = 1 + 1 / (sqrt (1 + 4 kappa^2) + 2 kappa) + sqrt (2
     tau)] has no cancellation. From [kappa = 1] up the constants are formed in
     [e = 1 / (2 kappa)], which keeps them finite up to the largest float: [D /
     (2 kappa) = d = h + e + g] for [h = sqrt (1 + e^2)] and [g = sqrt (2 e (e +
     h))], [rho = 1 / d], [1 - rho = sqrt e w / d] and [A = w^2 / (4 d)], where
     [w = (d - 1) / sqrt e = e sqrt e / (h + 1) + sqrt e + sqrt (2 (e + h))]. At
     [kappa = 0], [rho] is 0, [y] is 1, every proposal is accepted and [theta]
     is [2 phi]: the uniform circle. *)
  let von_mises_rounds = 20

  let wrapped_cauchy kappa =
    let lit v = scalar_like kappa v in
    (* Each regime reads a concentration of its own range, so neither divides by
       zero or overflows where the other is selected: a derivative passes
       through both. *)
    let large = cmpge kappa (lit 1.0) in
    let small_kappa = where large (lit 0.0) kappa in
    let large_kappa = where large kappa (lit 1.0) in
    let two_kappa = mul (lit 2.0) small_kappa in
    let root = sqrt (add (lit 1.0) (mul two_kappa two_kappa)) in
    let tau = add (lit 1.0) root in
    let r = sqrt (mul (lit 2.0) tau) in
    let dd = add tau r in
    let gap = add (add (lit 1.0) (recip (add root two_kappa))) r in
    let e = div (lit 0.5) large_kappa in
    let h = sqrt (add (lit 1.0) (mul e e)) in
    let d = add (add h e) (sqrt (mul (lit 2.0) (mul e (add e h)))) in
    let w =
      add
        (add (div (mul e (sqrt e)) (add h (lit 1.0))) (sqrt e))
        (sqrt (mul (lit 2.0) (add e h)))
    in
    let rho = where large (recip d) (div two_kappa dd) in
    let one_minus_rho = where large (div (mul (sqrt e) w) d) (div gap dd) in
    let a =
      where large
        (div (mul w w) (mul (lit 4.0) d))
        (div (mul gap gap) (mul (lit 4.0) dd))
    in
    (rho, one_minus_rho, a)

  let von_mises (type b) k (concentration : (float, b) t) : (float, b) t =
    require "von_mises" "concentration" non_negative concentration;
    let ctx = Value.context k in
    let shape = shape concentration in
    let draw (type c) (compute : (float, c) Nx_dtype.t) =
      let lit v = scalar ctx compute v in
      let rho, one_minus_rho, a = wrapped_cauchy (at compute concentration) in
      let one_plus_rho = add (lit 1.0) rho in
      let lift = mul a (mul one_plus_rho one_plus_rho) in
      let denom phi =
        let c = cos phi in
        add
          (mul one_minus_rho one_minus_rho)
          (mul (mul (lit 4.0) rho) (mul c c))
      in
      let rounds = von_mises_rounds in
      let draws = unit k compute (Array.append [| 2; rounds |] shape) in
      (* The rounds settle on a half angle; the angle is formed once, from the
         half angle each element took. *)
      let acc = ref (zeros ctx compute shape) in
      let last = ref !acc in
      let settled = ref (cmpne !acc !acc) in
      for j = 0 to rounds - 1 do
        let u = slice [ I 0; I j ] draws in
        let v = slice [ I 1; I j ] draws in
        let phi = mul (sub u (lit 0.5)) (lit Float.pi) in
        let y = div lift (denom phi) in
        let accept =
          logical_or
            (cmpgt (mul y (sub (lit 2.0) y)) v)
            (cmpge (sub (log (div y v)) y) (lit (-1.0)))
        in
        let take = logical_and accept (logical_not !settled) in
        acc := where take phi !acc;
        settled := logical_or !settled accept;
        last := phi
      done;
      let phi = where !settled !acc !last in
      let half = div (mul (abs (sin phi)) one_minus_rho) (sqrt (denom phi)) in
      let theta =
        mul (sign phi) (mul (lit 2.0) (asin (minimum half (lit 1.0))))
      in
      at (dtype concentration) theta
    in
    match dtype concentration with
    | Nx_dtype.Float64 -> draw Nx_dtype.float64
    | _ -> draw Nx_dtype.float32

  (* Log pmfs at integer-valued counts, written so that no term grows with the
     counts: the direct forms cancel terms of size [m log m] to a margin of
     order one, which float32 loses above a mean of ten thousand. With
     Stirling's formula for the factorials they are sums of saddle-point
     deviances [bd0 (x, m) = x log (x/m) + m - x], remainders [stirlerr x] of
     Stirling's series and a logarithm of order one (Loader, 2000).

     Near [x = m] the deviance is itself a cancellation, so there it comes from
     Loader's series in [v = (x - m) / (x + m)], whose terms are all the same
     sign; [deviance x m d] takes [d = x - m] from its caller, who can form it
     without subtracting two large numbers. The remainder's asymptotic series
     holds from 8 up; below that it is read off a shift by eight, where every
     quantity is small. Both regimes of each [where] run everywhere, so the
     arithmetic must stay finite wherever the result is not selected: a count of
     zero, where the deviance is [0 log 0], gets its own branch. *)
  let stirlerr y =
    let lit v = scalar_like y v in
    let stirling_main y =
      add
        (sub (mul (add y (lit 0.5)) (log y)) y)
        (lit (0.5 *. Float.log (2.0 *. Float.pi)))
    in
    let stirling_tail y =
      let y2 = mul y y in
      div
        (sub
           (lit (1.0 /. 12.0))
           (div (sub (lit (1.0 /. 360.0)) (div (lit (1.0 /. 1260.0)) y2)) y2))
        y
    in
    let shifted = add y (lit 8.0) in
    let product = ref (add y (lit 1.0)) in
    for i = 2 to 8 do
      product := mul !product (add y (lit (float_of_int i)))
    done;
    let small =
      sub
        (sub
           (add (stirling_main shifted) (stirling_tail shifted))
           (log !product))
        (stirling_main y)
    in
    where (cmplt y (lit 8.0)) small (stirling_tail y)

  let deviance x m d =
    let lit v = scalar_like x v in
    let s = add x m in
    let v = div d s in
    let v2 = mul v v in
    let acc = ref (mul d v) in
    let term = ref (mul (mul (lit 2.0) x) v) in
    for j = 1 to 6 do
      term := mul !term v2;
      acc := add !acc (div !term (lit (float_of_int ((2 * j) + 1))))
    done;
    let direct = add (sub (mul x (log (div x m))) x) m in
    where (cmplt (abs d) (mul (lit 0.1) s)) !acc direct

  (* The Poisson log pmf: -bd0 (k, rate) - log (2 pi k) / 2 - stirlerr k. *)
  let log_poisson_pmf k rate =
    let lit v = scalar_like k v in
    where
      (cmplt k (lit 0.5))
      (neg rate)
      (sub
         (sub
            (neg (deviance k rate (sub k rate)))
            (mul (lit 0.5) (log (mul (lit (2.0 *. Float.pi)) k))))
         (stirlerr k))

  (* The binomial log pmf of [n] trials of probability [p], with [q = 1 - p]:

     stirlerr n - stirlerr k - stirlerr (n - k)

     - bd0 (k, n p) - bd0 (n - k, n q) - log (2 pi k (n - k) / n) / 2,

     where both deviances take their difference from [n p - k], which no large
     [n] cancels. The counts [0] and [n] have their own branches, the rest is
     read at a count kept strictly inside them. *)
  let log_binomial_pmf k n p q =
    let lit v = scalar_like k v in
    let inside = maximum (minimum k (sub n (lit 1.0))) (lit 1.0) in
    let rest = sub n inside in
    let d = sub inside (mul n p) in
    let saddle =
      sub
        (sub
           (sub (sub (stirlerr n) (stirlerr inside)) (stirlerr rest))
           (deviance inside (mul n p) d))
        (deviance rest (mul n q) (neg d))
    in
    let spread =
      add
        (log (mul (lit (2.0 *. Float.pi)) inside))
        (log1p (neg (div inside n)))
    in
    where
      (cmplt k (lit 0.5))
      (mul n (log1p (neg p)))
      (where
         (cmpgt k (sub n (lit 0.5)))
         (mul n (log p))
         (sub saddle (mul (lit 0.5) spread)))

  (* Two regimes with a fixed round count each, chosen per element, so the shape
     of the computation does not depend on the rate and the rate can be data.
     Both run for every element and [where] picks.

     Below 10, inversion: one uniform against the cumulative pmf, whose terms
     exp (-rate + k log rate - log k!) are formed directly over a leading axis
     of 48 rounds. The mass beyond 48 at rate 10 is 4e-18. A rate of zero makes
     every term NaN, every comparison false and the count 0.

     From 10 up, Hörmann's transformed rejection with squeeze (PTRS): a proposal
     from a scaled logistic hat, accepted by a squeeze test or, failing that, by
     comparing against the log pmf. Acceptance is 0.75 at rate 10 and 0.89 in
     the limit, so 16 rounds leave 2e-10 of an element unaccepted; that element
     takes its last proposal, a draw from the hat with mean near the rate. The
     elements the inversion owns see a benign rate of 1e5 here: the hat's
     constants have poles below rate 1 that would otherwise put infinities into
     the integer cast, which is undefined in C.

     The draw runs at the rate's compute dtype. What bounds float32 is not the
     pmf test but the proposal itself, an integer formed next to the rate: above
     a rate of about 1e5 the float32 spacing there exceeds a hundredth of a
     count and the proposals drift off their bins. *)
  let poisson_inversion_rounds = 48
  let poisson_rejection_rounds = 16

  let poisson (type b) k (rate : (float, b) t) =
    require "poisson" "rate" non_negative rate;
    let ctx = Value.context k in
    let shape = shape rate in
    let draw (type c) (compute : (float, c) Nx_dtype.t) =
      let rate = at compute rate in
      let lit v = scalar ctx compute v in
      let ks = split k in
      let small = logical_not (cmpge rate (lit 10.0)) in
      let inversion =
        let rounds = poisson_inversion_rounds in
        let u = uniform ks.(0) compute shape in
        let along_rounds v =
          reshape
            (Array.append [| rounds |] (Array.make (Array.length shape) 1))
            v
        in
        let count =
          along_rounds (cast compute (arange ctx Nx_dtype.int32 0 rounds 1))
        in
        let log_fact =
          let acc = ref 0.0 in
          Array.init rounds (fun i ->
              if i > 0 then acc := !acc +. Float.log (float_of_int i);
              !acc)
        in
        let log_fact =
          along_rounds (create ctx compute [| rounds |] log_fact)
        in
        let log_pmf = sub (sub (mul count (log rate)) rate) log_fact in
        let cdf = cumsum ~axis:0 (exp log_pmf) in
        sum ~axes:[ 0 ] (cast compute (cmplt cdf u))
      in
      let rejection =
        let rounds = poisson_rejection_rounds in
        let lam = where small (lit 1e5) rate in
        let b = add (lit 0.931) (mul (lit 2.53) (sqrt lam)) in
        let a = add (lit (-0.059)) (mul (lit 0.02483) b) in
        let log_inv_alpha =
          log (add (lit 1.1239) (div (lit 1.1328) (sub b (lit 3.4))))
        in
        let vr = sub (lit 0.9277) (div (lit 3.6224) (sub b (lit 2.0))) in
        let draws =
          uniform ks.(1) compute (Array.append [| 2; rounds |] shape)
        in
        let acc = ref (zeros ctx compute shape) in
        let last = ref !acc in
        let settled = ref (cmpne !acc !acc) in
        for j = 0 to rounds - 1 do
          let u = sub (contiguous (slice [ I 0; I j ] draws)) (lit 0.5) in
          let v = contiguous (slice [ I 1; I j ] draws) in
          let us = sub (lit 0.5) (abs u) in
          let proposal =
            floor
              (add
                 (add (mul (add (div (mul (lit 2.0) a) us) b) u) lam)
                 (lit 0.43))
          in
          (* Brought into int32 before anything is done with it: a [us] of zero
             sends the proposal to infinity and an infinite rate makes it NaN.
             The tests below reject both, but the final cast must never see
             them, and [where] on a comparison sends NaN to zero without leaning
             on how [maximum] treats it. *)
          let count =
            where
              (cmpge proposal (lit 0.0))
              (minimum proposal (lit 2147483647.0))
              (lit 0.0)
          in
          let squeeze = logical_and (cmpge us (lit 0.07)) (cmple v vr) in
          let reject =
            logical_or
              (cmplt proposal (lit 0.0))
              (logical_and (cmplt us (lit 0.013)) (cmpgt v us))
          in
          let lhs =
            sub (add (log v) log_inv_alpha) (log (add (div a (mul us us)) b))
          in
          let accept =
            logical_or squeeze
              (logical_and (logical_not reject)
                 (cmple lhs (log_poisson_pmf count lam)))
          in
          let take = logical_and accept (logical_not !settled) in
          acc := where take count !acc;
          settled := logical_or !settled accept;
          last := count
        done;
        where !settled !acc !last
      in
      cast Nx_dtype.int32 (where small inversion rejection)
    in
    match dtype rate with
    | Nx_dtype.Float64 -> draw Nx_dtype.float64
    | _ -> draw Nx_dtype.float32

  (* Binomial(n, p) counts the failures of [1 - p] where [p] exceeds one half,
     so the samplers below see [p <= 1/2]. As for [poisson], two regimes run for
     every element and [where] picks, by the mean [n p].

     Below 10, inversion over a leading axis of 48 rounds: the terms of the pmf
     follow from [pmf 0 = (1 - p)^n] by the ratios [(n - j) p / ((j + 1) (1 -
     p))], summed in logarithms, so a [p] of zero or a count past [n] gives a
     term of zero rather than a product with an infinity. A uniform scaled by
     the sum of the terms is compared against their running sum, so the count
     never passes [n] or the last round. The mass beyond 48 is below 1e-16.

     From 10 up, Hörmann's transformed rejection (BTRS, 1993): a proposal from a
     scaled logistic hat around the mean, accepted by a squeeze test or, failing
     that, by comparing against the log pmf relative to the mode's. Acceptance
     is least, 0.71, at [n = 20] and [p = 1/2], so 18 rounds leave 2e-10 of an
     element unaccepted; that element takes its last proposal. The elements the
     inversion owns see the benign [n = 1000, p = 1/2] here. *)
  let binomial_inversion_rounds = 48
  let binomial_rejection_rounds = 18

  let binomial (type b) k (n : int32_t) (p : (float, b) t) =
    require "binomial" "n" natural n;
    require "binomial" "p" probability p;
    let ctx = Value.context k in
    let shape = Shape.broadcast (shape n) (shape p) in
    let n = broadcast_to shape n and p = broadcast_to shape p in
    let draw (type c) (compute : (float, c) Nx_dtype.t) =
      let lit v = scalar ctx compute v in
      let nf = cast compute n and p = at compute p in
      let flip = cmpgt p (lit 0.5) in
      let p = where flip (sub (lit 1.0) p) p in
      let small = logical_not (cmpge (mul nf p) (lit 10.0)) in
      let ks = split k in
      let inversion =
        let rounds = binomial_inversion_rounds in
        let along_rounds v =
          reshape
            (Array.append [| rounds |] (Array.make (Array.length shape) 1))
            v
        in
        let j =
          along_rounds (cast compute (arange ctx Nx_dtype.int32 0 rounds 1))
        in
        let log_fact =
          let acc = ref 0.0 in
          Array.init rounds (fun i ->
              if i > 0 then acc := !acc +. Float.log (float_of_int i);
              !acc)
        in
        let log_fact =
          along_rounds (create ctx compute [| rounds |] log_fact)
        in
        (* Round [j] holds [log (n (n - 1) ... (n - j + 1) (p / q)^j)]: the
           logarithms of the ratios [(n - i) p / q], summed over [i < j]. *)
        let ratio =
          add (log (maximum (sub nf j) (lit 0.0))) (sub (log p) (log1p (neg p)))
        in
        let falling =
          concatenate ~axis:0
            [
              zeros ctx compute (Array.append [| 1 |] shape);
              cumsum ~axis:0 (slice [ R (0, rounds - 1) ] ratio);
            ]
        in
        let log_pmf = sub (add (mul nf (log1p (neg p))) falling) log_fact in
        let cdf = cumsum ~axis:0 (exp log_pmf) in
        let total = slice [ I (rounds - 1) ] cdf in
        let u = uniform ks.(0) compute shape in
        sum ~axes:[ 0 ] (cast compute (cmplt cdf (mul u total)))
      in
      let rejection =
        let rounds = binomial_rejection_rounds in
        let nf = where small (lit 1000.0) nf and p = where small (lit 0.5) p in
        let q = sub (lit 1.0) p in
        let mean = mul nf p in
        let spq = sqrt (mul mean q) in
        let b = add (lit 1.15) (mul (lit 2.53) spq) in
        let a =
          add (add (lit (-0.0873)) (mul (lit 0.0248) b)) (mul (lit 0.01) p)
        in
        let c = add mean (lit 0.5) in
        let log_alpha = log (mul (add (lit 2.83) (div (lit 5.1) b)) spq) in
        let vr = sub (lit 0.92) (div (lit 4.2) b) in
        let mode = floor (mul (add nf (lit 1.0)) p) in
        let log_mode = log_binomial_pmf mode nf p q in
        let draws =
          uniform ks.(1) compute (Array.append [| 2; rounds |] shape)
        in
        let acc = ref (zeros ctx compute shape) in
        let last = ref !acc in
        let settled = ref (cmpne !acc !acc) in
        for j = 0 to rounds - 1 do
          let u = sub (contiguous (slice [ I 0; I j ] draws)) (lit 0.5) in
          let v = contiguous (slice [ I 1; I j ] draws) in
          let us = sub (lit 0.5) (abs u) in
          let proposal =
            floor (add (mul (add (div (mul (lit 2.0) a) us) b) u) c)
          in
          (* A [us] of zero sends the proposal to an infinity, which the range
             test rejects; the count stays finite for the cast. *)
          let count =
            where (cmpge proposal (lit 0.0)) (minimum proposal nf) (lit 0.0)
          in
          let inside =
            logical_and (cmpge proposal (lit 0.0)) (cmple proposal nf)
          in
          let squeeze = logical_and (cmpge us (lit 0.07)) (cmple v vr) in
          let lhs =
            sub (add (log v) log_alpha) (log (add (div a (mul us us)) b))
          in
          let accept =
            logical_and inside
              (logical_or squeeze
                 (cmple lhs (sub (log_binomial_pmf count nf p q) log_mode)))
          in
          let take = logical_and accept (logical_not !settled) in
          acc := where take count !acc;
          settled := logical_or !settled accept;
          last := count
        done;
        where !settled !acc !last
      in
      let y = cast Nx_dtype.int32 (where small inversion rejection) in
      where flip (sub n y) y
    in
    match dtype p with
    | Nx_dtype.Float64 -> draw Nx_dtype.float64
    | _ -> draw Nx_dtype.float32

  (* The draw is built and returned in int32, so the range must fit there:
     [Int32.of_int] would otherwise wrap a wide bound into a valid-looking
     narrow one. *)
  let check_range name ~low ~high =
    if low >= high then
      invalid_arg
        (Printf.sprintf "Nx.%s: invalid range, low=%d >= high=%d" name low high);
    let fits v = v >= -0x8000_0000 && v <= 0x7FFF_FFFF in
    if not (fits low && fits high) then
      invalid_arg
        (Printf.sprintf
           "Nx.%s: range [%d, %d) does not fit in int32, the result dtype" name
           low high)

  let randint k ?(low = 0) ~high shape =
    check_range "Rng.randint" ~low ~high;
    let ctx = Value.context k in
    let u = uniform k Nx_dtype.float32 shape in
    (* [u * (high - low)] is non-negative, so the cast's truncation is a floor;
       shifting by [low] afterwards keeps it one, where folding [low] in first
       would truncate towards zero and both drop [low] and double the count of 0
       for a negative [low]. The offset reaches [2 ** 32 - 256] when the range
       spans int32, past what int32 holds, so it is formed in uint32 and shifted
       there, where the sum wraps into the int32 value it names. *)
    let span = scalar ctx Nx_dtype.float32 (float_of_int (high - low)) in
    bitcast Nx_dtype.int32
      (add
         (cast Nx_dtype.uint32 (mul u span))
         (scalar ctx Nx_dtype.uint32 (Int32.of_int low)))

  let bernoulli (type b) k (p : (float, b) t) =
    require "bernoulli" "p" probability p;
    let draw (type c) (compute : (float, c) Nx_dtype.t) =
      cmplt (uniform k compute (shape p)) (at compute p)
    in
    match dtype p with
    | Nx_dtype.Float64 -> draw Nx_dtype.float64
    | _ -> draw Nx_dtype.float32

  (* Inverse-CDF truncation: a standard normal restricted to [lo, hi] is [sqrt 2
     * erfinv u] for [u] uniform over the images of the bounds under [erf]. One
     draw per sample, no acceptance test, so it compiles and its cost does not
     depend on how much mass the interval holds. *)
  let truncated_normal (type b) k (lower : (float, b) t) (upper : (float, b) t)
      : (float, b) t =
    require "truncated_normal" "lower" not_nan lower;
    require "truncated_normal" "upper" not_nan upper;
    let lower, upper = pair lower upper in
    let ctx = Value.context k in
    let target = dtype lower in
    let draw (type c) (compute : (float, c) Nx_dtype.t) =
      let lit v = scalar ctx compute v in
      let lower = at compute lower and upper = at compute upper in
      (* [erfinv] is infinite at +/-1, which infinite bounds would reach; back
         the interval off to the last value before +/-1 so unbounded truncation
         is the untruncated normal rather than an infinity. A wider margin cuts
         the support short of where [erf] resolves it: a margin of [1e-7] stops
         float64 draws at 5.3 standard deviations. *)
      let limit = 1.0 -. Float.ldexp 1.0 (-significand_bits compute) in
      let edge x =
        maximum (lit (-.limit))
          (minimum (lit limit)
             (B.unary Erf (mul x (lit (1.0 /. Float.sqrt 2.0)))))
      in
      let lo = edge lower and hi = edge upper in
      let u = uniform k compute (shape lower) in
      let x =
        mul (lit (Float.sqrt 2.0)) (Special.erfinv (add lo (mul u (sub hi lo))))
      in
      (* The inverse carries seven digits, so a draw next to a bound can land an
         ulp or so past it; the clamp makes the support exact. *)
      minimum (maximum x (minimum lower upper)) (maximum lower upper)
    in
    match target with
    | Nx_dtype.Float64 -> at target (draw Nx_dtype.float64)
    | _ -> at target (draw Nx_dtype.float32)

  (* Order [n] random sort keys. The keys are 64 bits wide, built from a
     Threefry row per element, rather than a [uniform] draw: a uniform carries
     at most 24 significant bits, and at [n = 60_000] — one MNIST epoch — that
     is about 107 expected collisions, each of which [argsort] resolves towards
     the input order. Sixty-four bits put the expected collision count at [3e-8]
     for a million elements. Any bijection of the random bits keeps the ordering
     uniform, so the arithmetic below is free to wrap. *)
  let permutation k n =
    if n <= 0 then invalid_arg "Nx.Rng.permutation: n must be positive";
    let ctx = Value.context k in
    let words = blocks "permutation" k n in
    let word col =
      cast Nx_dtype.int64
        (reshape [| n |]
           (contiguous (shrink [| (0, n); (col, col + 1) |] words)))
    in
    let low_32 = scalar ctx Nx_dtype.int64 0xFFFF_FFFFL in
    let sort_key =
      add
        (mul
           (bitwise_and (word 0) low_32)
           (scalar ctx Nx_dtype.int64 0x1_0000_0000L))
        (bitwise_and (word 1) low_32)
    in
    argsort sort_key ~axis:0 ~descending:false

  let shuffle k x =
    let s = shape x in
    if Array.length s = 0 || s.(0) = 0 then x
    else take ~axis:0 ~indices:(permutation k s.(0)) x

  (* Gumbel-max: adding Gumbel noise to log-probabilities and taking the argmax
     samples from the distribution they describe. The noise is built at least in
     float32, then cast to the logits' dtype — a float16 Gumbel would quantise
     the comparison the argmax turns on. *)
  let categorical (type b) k ?(axis = -1) (logits : (float, b) t) =
    let logits_dtype = dtype logits in
    let logits_shape = shape logits in
    let nd = Array.length logits_shape in
    let axis = if axis < 0 then nd + axis else axis in
    if axis < 0 || axis >= nd then
      invalid_arg
        (Printf.sprintf
           "Nx.Rng.categorical: axis %d out of bounds for %dD tensor" axis nd);
    let noise compute = astype logits_dtype (gumbel k compute logits_shape) in
    let g =
      match logits_dtype with
      | Float64 -> noise Nx_dtype.float64
      | Float32 | Float16 | BFloat16 -> noise Nx_dtype.float32
      | Float8_e4m3 | Float8_e5m2 ->
          invalid_arg "Nx.Rng.categorical: float8 logits are not supported"
    in
    require "categorical" "logits" below_infinity logits;
    argmax (add logits g) ~axis ~keepdims:false

  (* The scope: [next_key] performs [E_next_key]; [with_key] answers it by
     [fold_in root counter] with an incrementing counter — the same [fold_in]
     the explicit path uses, so the two front-ends share one stream. Every
     derived key is therefore a tensor computation on the root, which is what
     lets a scope rooted at a traced or batched key compile and batch like an
     explicit one. [next_root] performs [E_place], which takes a counter and
     answers with a function performing [E_at (scope, counter)]: only the scope
     that took the place answers it, every other passes it outward, so the key
     is that scope's whatever scopes lie between.

     The handler is an effect handler, so it is per-fiber and per-domain: a draw
     on a domain spawned inside a scope does not see it and falls back below. *)
  type scope = unit ref

  type _ Effect.t +=
    | E_next_key : key Effect.t
    | E_place : (unit -> key) Effect.t
    | E_at : scope * int -> key Effect.t

  (* The root is taken at the first draw, in the handler, outside the scope: a
     draw [root] makes comes from the scope around. A key that raises raises at
     the draw, inside the scope, so that the code between the scope and the draw
     unwinds; the next draw takes the root again. *)
  let make_handler root =
    let scope = ref () and counter = ref 0 and taken = ref None in
    let at c =
      let r =
        match !taken with
        | Some r -> r
        | None ->
            let r = root () in
            taken := Some r;
            r
      in
      fold_in r c
    in
    let take () =
      let c = !counter in
      incr counter;
      c
    in
    let open Effect.Deep in
    let answer (type a) (k : (a, _) continuation) (f : unit -> a) =
      match f () with
      | v -> continue k v
      | exception e ->
          let bt = Printexc.get_raw_backtrace () in
          discontinue_with_backtrace k e bt
    in
    {
      retc = Fun.id;
      exnc = raise;
      effc =
        (fun (type a) (eff : a Effect.t) ->
          match eff with
          | E_next_key ->
              Some
                (fun (k : (a, _) continuation) ->
                  answer k (fun () ->
                      let key = at !counter in
                      ignore (take ());
                      key))
          | E_place ->
              Some
                (fun (k : (a, _) continuation) ->
                  let c = take () in
                  continue k (fun () -> Effect.perform (E_at (scope, c))))
          | E_at (s, c) when s == scope ->
              Some (fun (k : (a, _) continuation) -> answer k (fun () -> at c))
          | _ -> None);
    }

  let with_root root f = Effect.Deep.match_with f () (make_handler root)
  let with_key k f = with_root (fun () -> k) f
  let fallback = Domain.DLS.new_key (fun () -> ref None)

  (* Outside any scope, seed from system entropy. [Random.bits] on the default
     state is not an option: OCaml seeds that state deterministically, so
     unscoped draws repeated exactly from run to run — and would have started
     varying the moment some linked library called [Random.self_init].
     Reproducibility is what a scope is for; without one the draws should be
     fresh. *)
  let unscoped ctx =
    let cell = Domain.DLS.get fallback in
    let state =
      match !cell with
      | Some s -> s
      | None ->
          let entropy = Random.State.make_self_init () in
          key ctx (Int64.to_int (Random.State.bits64 entropy))
    in
    let keys = split state in
    cell := Some keys.(0);
    keys.(1)

  let next_key ctx =
    try Effect.perform E_next_key with Effect.Unhandled _ -> unscoped ctx

  (* Outside any scope, the place is one key of the domain's generator, taken at
     the first call. A call after the scope returned finds no scope to answer
     it. *)
  let next_root ctx =
    match Effect.perform E_place with
    | place -> (
        fun () ->
          try place ()
          with Effect.Unhandled _ ->
            invalid_arg
              "Nx.Rng.next_root: the key's scope returned before it was \
               computed")
    | exception Effect.Unhandled _ -> (
        let taken = ref None in
        fun () ->
          match !taken with
          | Some k -> k
          | None ->
              let k = unscoped ctx in
              taken := Some k;
              k)
end

let validate_random_float_params op dtype shape =
  if not (Nx_dtype.is_float dtype) then
    err op
      "dtype %s, not a float type, rand/randn only support Float16, Float32, \
       Float64"
      (Nx_dtype.to_string dtype);
  if Array.exists (fun x -> x < 0) shape then
    err op "invalid shape %s, dimensions must be non-negative"
      (Shape.to_string shape)

(* Sample at [dtype] rather than at float32 and narrow: rounding a float32 draw
   down to a narrower dtype can land on 1. *)
let rand ctx (type b) (dtype : (float, b) Nx_dtype.t) shape =
  validate_random_float_params "rand" dtype shape;
  Rng.uniform (Rng.next_key ctx) dtype shape

let randn ctx (type b) (dtype : (float, b) Nx_dtype.t) shape =
  validate_random_float_params "randn" dtype shape;
  Rng.normal (Rng.next_key ctx) dtype shape

let randint ctx ?(low = 0) ~high shape =
  Rng.check_range "randint" ~low ~high;
  Rng.randint (Rng.next_key ctx) ~low ~high shape

let bernoulli ctx p = Rng.bernoulli (Rng.next_key ctx) p
let permutation ctx n = Rng.permutation (Rng.next_key ctx) n
let shuffle ctx x = Rng.shuffle (Rng.next_key ctx) x

let categorical ctx ?axis logits =
  Rng.categorical (Rng.next_key ctx) ?axis logits

let truncated_normal ctx lower upper =
  Rng.truncated_normal (Rng.next_key ctx) lower upper
