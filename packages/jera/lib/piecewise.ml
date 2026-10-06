(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type ('v, 'b) t = {
  s : 'v Nx.Ptree.t;
  breaks : (float, 'b) Nx.t;
  coefficients : 'v;
  extension : Cheb.extension;
}

let fail fn fmt = Printf.ksprintf (fun m -> invalid_arg (fn ^ ": " ^ m)) fmt
let pieces p = Nx.dim 0 p.breaks - 1

let check_breaks fn what breaks =
  if Nx.ndim breaks <> 1 || Nx.dim 0 breaks < 2 then
    fail fn "%s must be 1-D with at least two elements, got shape %s" what
      (Num.shape (Nx.shape breaks));
  Num.check_increasing fn what breaks

let v s ~breaks c =
  let fn = "Jera.Piecewise.v" in
  check_breaks fn "breaks" breaks;
  let n = Nx.dim 0 breaks - 1 in
  Nx.Ptree.fold s
    (fun path x () ->
      ignore (Num.on_float fn { f = Fun.id } x);
      if Nx.ndim x < 2 || Nx.dim 0 x <> n then
        fail fn
          "%s: shape %s; a leaf has shape [pieces; degree + 1] @ value, with \
           %d pieces"
          (Nx.Ptree.Path.to_string path)
          (Num.shape (Nx.shape x))
          n)
    c ();
  { s; breaks; coefficients = c; extension = Cheb.Bounded }

(* Interpolants *)

type 'b ends =
  [ `Natural | `Not_a_knot | `Clamped of (float, 'b) Nx.t * (float, 'b) Nx.t ]

let samples fn x y =
  check_breaks fn "the knots" x;
  if Nx.ndim y = 0 || Nx.dim 0 y <> Nx.dim 0 x then
    fail fn "the samples have shape %s for %d knots"
      (Num.shape (Nx.shape y))
      (Nx.dim 0 x)

let interpolant breaks c =
  { s = Nx.Ptree.tensor; breaks; coefficients = c; extension = Cheb.Bounded }

let linear x y =
  samples "Jera.Piecewise.linear" x y;
  interpolant x (Spline.linear x y)

let cubic ends x y =
  let fn = "Jera.Piecewise.cubic" in
  samples fn x y;
  (match ends with
  | `Clamped (s0, s1) ->
      let value = Array.sub (Nx.shape y) 1 (Nx.ndim y - 1) in
      if Nx.shape s0 <> value || Nx.shape s1 <> value then
        fail fn
          "the clamped slopes have shapes %s and %s for values of shape %s"
          (Num.shape (Nx.shape s0))
          (Num.shape (Nx.shape s1))
          (Num.shape value)
  | `Natural | `Not_a_knot -> ());
  interpolant x (Spline.cubic ends x y)

let steffen x y =
  samples "Jera.Piecewise.steffen" x y;
  interpolant x (Spline.steffen x y)

let hermite x ~values ~slopes =
  let fn = "Jera.Piecewise.hermite" in
  samples fn x values;
  if Nx.shape slopes <> Nx.shape values then
    fail fn "the slopes have shape %s and the values %s"
      (Num.shape (Nx.shape slopes))
      (Num.shape (Nx.shape values));
  interpolant x (Spline.hermite x values slopes)

(* Fits *)

(* [k / pieces] of the way from [a] to [b], the last break [b] itself. *)
let uniform_breaks ~pieces a b =
  let s =
    Num.constant (Nx.dtype a)
      (Array.init pieces (fun k -> float k /. float pieces))
  in
  Nx.concatenate ~axis:0
    [ Nx.add a (Nx.mul (Nx.sub b a) s); Nx.reshape [| 1 |] b ]

let chebyshev s ~degree ~pieces f a b =
  let fn = "Jera.Piecewise.chebyshev" in
  if degree < 0 then fail fn "degree = %d is negative" degree;
  if pieces < 1 then fail fn "pieces = %d is not positive" pieces;
  if Nx.ndim a <> 0 || Nx.ndim b <> 0 then
    fail fn "the ends must be scalars, got shapes %s and %s"
      (Num.shape (Nx.shape a))
      (Num.shape (Nx.shape b));
  let breaks = uniform_breaks ~pieces a b in
  let lo = Nx.slice [ Nx.R (0, pieces) ] breaks
  and hi = Nx.slice [ Nx.R (1, pieces + 1) ] breaks in
  let mid = Nx.div_s (Nx.add lo hi) 2. and half = Nx.div_s (Nx.sub hi lo) 2. in
  let u =
    Nx.reshape
      [| 1; degree + 1 |]
      (Num.constant (Nx.dtype a) (Cheb.nodes degree))
  in
  let points =
    Nx.add
      (Nx.reshape [| pieces; 1 |] mid)
      (Nx.mul (Nx.reshape [| pieces; 1 |] half) u)
  in
  let values = f points in
  let coefficients =
    Nx.Ptree.map s
      (fun path x ->
        let shape = Nx.shape x in
        if
          Array.length shape < 2
          || shape.(0) <> pieces
          || shape.(1) <> degree + 1
        then
          fail fn "%s: shape %s for points of shape %s"
            (Nx.Ptree.Path.to_string path)
            (Num.shape shape)
            (Num.shape (Nx.shape points));
        Num.on_float fn { f = Cheb.fit } x)
      values
  in
  { s; breaks; coefficients; extension = Cheb.Bounded }

(* Adaptive fits *)

let max_level = 62

let adapt s ~degree ~tol ~budget f a b =
  let fn = "Jera.Piecewise.adapt" in
  if degree < 2 then fail fn "degree = %d is below 2" degree;
  if budget < 1 then fail fn "budget = %d is below 1" budget;
  if Nx.ndim a <> 0 || Nx.ndim b <> 0 then
    fail fn "the ends must be scalars, got shapes %s and %s"
      (Num.shape (Nx.shape a))
      (Num.shape (Nx.shape b));
  let dtype = Nx.dtype a in
  let m = degree + 1 in
  let u = Nx.reshape [| 1; m |] (Num.constant dtype (Cheb.nodes degree)) in
  (* The pieces [index / 2^level, (index + 1) / 2^level] of [0, 1], as their
     ends' fractions. *)
  let fractions level index =
    let scale = Nx.exp2 (Nx.neg (Nx.cast dtype level)) in
    let i = Nx.cast dtype index in
    (Nx.mul i scale, Nx.mul (Nx.add_s i 1.) scale)
  in
  let points a b (t0, t1) =
    let w = Nx.sub b a in
    let x0 = Nx.add a (Nx.mul w t0) and x1 = Nx.add a (Nx.mul w t1) in
    let k = Nx.dim 0 t0 in
    let mid = Nx.div_s (Nx.add x0 x1) 2.
    and half = Nx.div_s (Nx.sub x1 x0) 2. in
    Nx.add (Nx.reshape [| k; 1 |] mid) (Nx.mul (Nx.reshape [| k; 1 |] half) u)
  in
  let fit f pts =
    Nx.Ptree.map s
      (fun path y ->
        let shape = Nx.shape y in
        if Array.length shape < 2 || shape.(0) <> Nx.dim 0 pts || shape.(1) <> m
        then
          fail fn "%s: shape %s for points of shape %s"
            (Nx.Ptree.Path.to_string path)
            (Num.shape shape)
            (Num.shape (Nx.shape pts));
        Num.on_float fn { f = Cheb.fit } y)
      (f pts)
  in
  (* Each piece's tail against [tol]: the root mean square, over its components,
     of the larger of the last two coefficients' magnitudes over the scale of
     its largest. *)
  let tails c =
    let rows =
      Nx.Ptree.fold s
        (fun _ x acc ->
          Num.on_float fn
            {
              f =
                (fun x ->
                  let k = Nx.dim 0 x in
                  let flat = Nx.reshape [| k; m; -1 |] x in
                  let last i = Nx.abs (Nx.get [ i ] (Nx.moveaxis 1 0 flat)) in
                  let e = Nx.maximum (last degree) (last (degree - 1)) in
                  let y = Nx.max ~axes:[ 1 ] (Nx.abs flat) in
                  Nx.cast (Nx.dtype x) (Tolerance.ratio tol ~e ~y));
            }
            x
          |> fun r -> Nx.cast dtype r :: acc)
        c []
    in
    Num.rms_rows (Nx.concatenate ~axis:1 (List.rev rows))
  in
  let search_f x = Nx.Ptree.map s (fun _ y -> Rune.detach y) (f x) in
  let a0 = Rune.detach a and b0 = Rune.detach b in
  let slots = Nx.arange Nx.int32 0 budget 1 in
  let in_use used = Nx.less slots (Nx.broadcast_to [| budget |] used) in
  let worst used ratio =
    Nx.argmax
      (Nx.where (in_use used) ratio (Nx.full_like ratio Float.neg_infinity))
  in
  let pick j v = Nx.take ~axis:0 ~indices:(Nx.reshape [| 1 |] j) v in
  (* The carry: each slot's level, index, tail ratio and coefficients, the slots
     used, the status and the evaluations. *)
  let settle (level, (index, (ratio, (c, (used, (st, n)))))) =
    let any_nan = Nx.any (Nx.logical_and (in_use used) (Nx.isnan ratio)) in
    let st = Elementwise.settle st any_nan Not_finite in
    let met =
      Nx.all
        (Nx.logical_or
           (Nx.logical_not (in_use used))
           (Nx.less_equal_s ratio 1.))
    in
    let st = Elementwise.settle st met Converged in
    let j = worst used ratio in
    let lj = pick j level and ij = pick j index in
    let t0, t1 = fractions lj ij in
    let x0 = Nx.add a0 (Nx.mul (Nx.sub b0 a0) t0)
    and x1 = Nx.add a0 (Nx.mul (Nx.sub b0 a0) t1) in
    let flat =
      Nx.logical_or
        (Nx.greater_equal_s lj (Int32.of_int max_level))
        (Num.adjacent (Nx.minimum x0 x1) (Nx.maximum x0 x1))
    in
    let st = Elementwise.settle st (Nx.reshape [||] flat) Stalled in
    let st =
      Elementwise.settle st
        (Nx.greater_equal_s used (Int32.of_int budget))
        Budget_spent
    in
    (level, (index, (ratio, (c, (used, (st, n))))))
  in
  let step (level, (index, (ratio, (c, (used, (st, n)))))) =
    let j = worst used ratio in
    let lj = pick j level and ij = pick j index in
    let level2 = Nx.concatenate ~axis:0 [ Nx.add_s lj 1l; Nx.add_s lj 1l ] in
    let index2 =
      Nx.concatenate ~axis:0 [ Nx.mul_s ij 2L; Nx.add_s (Nx.mul_s ij 2L) 1L ]
    in
    let c2 = fit search_f (points a0 b0 (fractions level2 index2)) in
    let r2 = tails c2 in
    let left =
      Nx.equal slots (Nx.cast Nx.int32 (Nx.broadcast_to [| budget |] j))
    in
    let right = Nx.equal slots (Nx.broadcast_to [| budget |] used) in
    let put v two =
      let mask m =
        Nx.reshape (Array.append [| budget |] (Array.make (Nx.ndim v - 1) 1)) m
      in
      let child i =
        Nx.broadcast_to (Nx.shape v) (Nx.slice [ Nx.R (i, i + 1) ] two)
      in
      Nx.where (mask left) (child 0) (Nx.where (mask right) (child 1) v)
    in
    let c = Nx.Ptree.map2 s (fun _ v two -> put v two) c c2 in
    settle
      ( put level level2,
        ( put index index2,
          ( put ratio r2,
            (c, (Nx.add_s used 1l, (st, Nx.add_s n (Int32.of_int (2 * m))))) )
        ) )
  in
  let initial =
    let one dt = Nx.zeros dt [| 1 |] in
    let c1 =
      fit search_f (points a0 b0 (fractions (one Nx.int32) (one Nx.int64)))
    in
    let pad v =
      Nx.concatenate ~axis:0
        [
          v;
          Nx.zeros (Nx.dtype v)
            (Array.append
               [| budget - 1 |]
               (Array.sub (Nx.shape v) 1 (Nx.ndim v - 1)));
        ]
    in
    settle
      ( Nx.zeros Nx.int32 [| budget |],
        ( Nx.zeros Nx.int64 [| budget |],
          ( pad (tails c1),
            ( Nx.Ptree.map s (fun _ v -> pad v) c1,
              ( Nx.scalar Nx.int32 1l,
                ( Nx.scalar Nx.int32 Elementwise.running,
                  Nx.scalar Nx.int32 (Int32.of_int m) ) ) ) ) ) )
  in
  let carry =
    Nx.Ptree.(
      pair tensor
        (pair tensor (pair tensor (pair s (pair tensor (pair tensor tensor))))))
  in
  let level, (index, (_, (c, (used, (st, n))))) =
    Rune.iterate carry ~max:budget
      ~until:(fun (_, (_, (_, (_, (_, (st, _)))))) ->
        Nx.logical_not (Elementwise.searching st))
      ~f:step initial
  in
  let ok = Nx.equal_s st (Solution.code Converged) in
  (* The final partition in increasing order, unused slots last as empty pieces
     at b. *)
  let used_mask = in_use used in
  let t0, _ = fractions level index in
  let order =
    Nx.argsort (Nx.where used_mask t0 (Nx.full_like t0 Float.infinity))
  in
  let sorted v = Nx.take ~axis:0 ~indices:order v in
  let level = sorted level
  and index = sorted index
  and live = sorted used_mask in
  let t0, t1 = fractions level index in
  let t0 = Nx.where live t0 (Nx.ones_like t0)
  and t1 = Nx.where live t1 (Nx.ones_like t1) in
  let series a b c =
    let breaks =
      Nx.concatenate ~axis:0
        [ Nx.add a (Nx.mul (Nx.sub b a) t0); Nx.reshape [| 1 |] b ]
    in
    let c =
      Nx.Ptree.map s
        (fun _ x ->
          let mask =
            Nx.reshape
              (Array.append [| budget |] (Array.make (Nx.ndim x - 1) 1))
              live
          in
          Nx.where mask x (Nx.zeros_like x))
        c
    in
    { s; breaks; coefficients = c; extension = Cheb.Bounded }
  in
  (* The answer interpolates the tracked f at the final pieces' points; an
     unused piece's points are inside the range. *)
  let mid = Nx.full_like t0 0.5 in
  let tracked =
    fit f (points a b (Nx.where live t0 mid, Nx.where live t1 mid))
  in
  let best = series a0 b0 (Nx.Ptree.map s (fun _ x -> sorted x) c) in
  let answer = series a b tracked in
  let choose =
    Nx.Ptree.map2 s (fun _ x y ->
        Nx.where (Nx.broadcast_to (Nx.shape x) ok) x y)
  in
  let value =
    {
      answer with
      breaks =
        Nx.where
          (Nx.broadcast_to (Nx.shape answer.breaks) ok)
          answer.breaks best.breaks;
      coefficients = choose answer.coefficients best.coefficients;
    }
  in
  let error =
    {
      best with
      coefficients =
        Nx.Ptree.map s
          (fun _ x ->
            Num.on_float fn
              {
                f =
                  (fun x ->
                    let tail i =
                      Nx.abs (Nx.slice [ Nx.A; Nx.R (i, i + 1) ] x)
                    in
                    Nx.maximum (tail degree) (tail (degree - 1)));
              }
              x)
          best.coefficients;
    }
  in
  Solution.v ~fn
    ~settings:
      (Format.asprintf "degree %d, tol %a, budget %d" degree Tol.pp tol budget)
    ~value ~error ~status:st ~evaluations:n ~facts:[]

(* Evaluation *)

let locate fn p x =
  let x = Nx.reshape [| Nx.numel x |] x in
  Cheb.locate fn p.extension p.breaks x

(* Each leaf's series of piece [i] at [u], both flat, reshaped to [q]. *)
let series fn p q i u =
  Nx.Ptree.map p.s
    (fun _ c ->
      Num.on_float fn
        {
          f =
            (fun c ->
              let g = Nx.take ~axis:0 ~indices:i c in
              let y = Cheb.clenshaw (Nx.cast (Nx.dtype c) u) g in
              let value = Array.sub (Nx.shape c) 2 (Nx.ndim c - 2) in
              Nx.reshape (Array.append q value) y);
        }
        c)
    p.coefficients

let eval p x =
  let fn = "Jera.Piecewise.eval" in
  let i, u = locate fn p x in
  (* A NaN point is NaN whatever the degree: a constant piece ignores u. *)
  let nan = Nx.isnan (Nx.reshape [| Nx.numel x |] x) in
  Nx.Ptree.map p.s
    (fun _ y ->
      Num.on_float fn
        {
          f =
            (fun y ->
              let mask =
                Nx.reshape
                  (Array.append (Nx.shape x)
                     (Array.make (Nx.ndim y - Nx.ndim x) 1))
                  nan
              in
              Nx.where mask (Nx.full_like y Float.nan) y);
        }
        y)
    (series fn p (Nx.shape x) i u)

let eval_at p i u =
  let fn = "Jera.Piecewise.eval_at" in
  if Nx.shape i <> Nx.shape u then
    fail fn "the indices have shape %s and the coordinates %s"
      (Num.shape (Nx.shape i))
      (Num.shape (Nx.shape u));
  let n = pieces p in
  let flat_i = Nx.reshape [| Nx.numel i |] i in
  Nx.check Nx.Ptree.tensor
    (Nx.logical_and
       (Nx.greater_equal_s flat_i 0L)
       (Nx.less_s flat_i (Int64.of_int n)))
    flat_i
    (fun k i ->
      Invalid_argument
        (Printf.sprintf "%s: the index at [%d] is %Ld, not one of the %d pieces"
           fn k.(0) (Nx.item [] i) n));
  series fn p (Nx.shape u) flat_i (Nx.reshape [| Nx.numel u |] u)

(* Calculus *)

let widths p =
  let n = pieces p in
  Nx.sub
    (Nx.slice [ Nx.R (1, n + 1) ] p.breaks)
    (Nx.slice [ Nx.R (0, n) ] p.breaks)

type op = { op : 'c. (float, 'c) Nx.t -> (float, 'c) Nx.t -> (float, 'c) Nx.t }

let calculus fn { op } p =
  let w = widths p in
  let coefficients =
    Nx.Ptree.map p.s
      (fun _ c ->
        Num.on_float fn { f = (fun c -> op (Nx.cast (Nx.dtype c) w) c) } c)
      p.coefficients
  in
  { p with coefficients }

let derivative p =
  calculus "Jera.Piecewise.derivative" { op = Cheb.derivative } p

let integral p = calculus "Jera.Piecewise.integral" { op = Cheb.integral } p

let extend e p =
  {
    p with
    extension = (match e with `Hold -> Cheb.Hold | `Polynomial -> Polynomial);
  }

(* Access *)

let breaks p = p.breaks
let coefficients p = p.coefficients

let ptree (type v b) (s : v Nx.Ptree.t) (_ : (float, b) Nx.dtype) :
    (v, b) t Nx.Ptree.t =
  let module M = struct
    type nonrec _ t = (v, b) t

    let walk c p =
      let open Nx.Ptree.Walk in
      let extension =
        field c "extension"
          (fun c e ->
            case c
              (match (e : Cheb.extension) with
              | Bounded -> "bounded"
              | Hold -> "hold"
              | Polynomial -> "polynomial");
            e)
          p.extension
      in
      let breaks = field c "breaks" tensor p.breaks in
      let coefficients = field c "coefficients" (structure s) p.coefficients in
      { p with breaks; coefficients; extension }
  end in
  Nx.Ptree.instantiate (module M)
