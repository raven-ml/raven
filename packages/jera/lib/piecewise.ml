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

let ptree (type v b) (s : v Nx.Ptree.t) : (v, b) t Nx.Ptree.t =
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
