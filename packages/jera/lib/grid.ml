(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Coefficients of shape [[P_1; …; P_d; D_1; …; D_d] @ value], D_k the series'
   lengths. Construction works in the interleaved layout [[P_1; D_1; …; P_d;
   D_d] @ value], one axis at a time. *)
type 'b t = { breaks : (float, 'b) Nx.t list; coefficients : (float, 'b) Nx.t }

let fail fn fmt = Printf.ksprintf (fun m -> invalid_arg (fn ^ ": " ^ m)) fmt
let dims g = List.length g.breaks

(* The permutation that moves the axes [src] of a tensor of [n] axes to the
   front, in order. *)
let front src n =
  src @ List.filter (fun a -> not (List.mem a src)) (List.init n Fun.id)

let to_front src c = Nx.transpose ~axes:(front src (Nx.ndim c)) c

let of_front src c =
  let perm = front src (Nx.ndim c) in
  let inverse = Array.make (Nx.ndim c) 0 in
  List.iteri (fun i a -> inverse.(a) <- i) perm;
  Nx.transpose ~axes:(Array.to_list inverse) c

(* Apply [f] from samples [[n] @ rest] to pieces [[P; D] @ rest] along each axis
   in turn, then gather the pieces' axes in front of the series'. *)
let build f d values =
  let c = ref values in
  for k = 0 to d - 1 do
    let front = to_front [ 2 * k ] !c in
    c := of_front [ 2 * k; (2 * k) + 1 ] (f k front)
  done;
  let value = List.init (Nx.ndim !c - (2 * d)) (fun i -> (2 * d) + i) in
  Nx.transpose
    ~axes:
      (List.init d (fun k -> 2 * k) @ List.init d (fun k -> (2 * k) + 1) @ value)
    !c

let check_axes fn axes values =
  if axes = [] then fail fn "no axis";
  let d = List.length axes in
  if Nx.ndim values < d then
    fail fn "values of shape %s for %d axes" (Num.shape (Nx.shape values)) d;
  List.iteri
    (fun k x ->
      if Nx.ndim x <> 1 || Nx.dim 0 x < 2 then
        fail fn "axis %d must be 1-D with at least two knots, got shape %s" k
          (Num.shape (Nx.shape x));
      if Nx.dim k values <> Nx.dim 0 x then
        fail fn "axis %d has %d knots and the values %d" k (Nx.dim 0 x)
          (Nx.dim k values);
      Num.check_increasing fn (Printf.sprintf "the knots of axis %d" k) x)
    axes

let interpolant fn spline ~axes values =
  check_axes fn axes values;
  let axes_a = Array.of_list axes in
  {
    breaks = axes;
    coefficients =
      build (fun k y -> spline axes_a.(k) y) (List.length axes) values;
  }

let linear ~axes values =
  interpolant "Jera.Grid.linear" Spline.linear ~axes values

let cubic ends ~axes values =
  let ends = (ends :> _ Spline.ends) in
  interpolant "Jera.Grid.cubic" (Spline.cubic ends) ~axes values

let chebyshev ~degree ~pieces f ~lo ~hi =
  let fn = "Jera.Grid.chebyshev" in
  if degree < 0 then fail fn "degree = %d is negative" degree;
  if pieces < 1 then fail fn "pieces = %d is not positive" pieces;
  if Nx.ndim lo <> 1 || Nx.shape lo <> Nx.shape hi || Nx.dim 0 lo < 1 then
    fail fn "lo and hi must have one shape [d], got %s and %s"
      (Num.shape (Nx.shape lo))
      (Num.shape (Nx.shape hi));
  let d = Nx.dim 0 lo and m = degree + 1 in
  let dtype = Nx.dtype lo in
  let s =
    Num.constant dtype (Array.init pieces (fun k -> float k /. float pieces))
  in
  let u = Num.constant dtype (Cheb.nodes degree) in
  let breaks =
    List.init d (fun k ->
        let a = Nx.get [ k ] lo and b = Nx.get [ k ] hi in
        Nx.concatenate ~axis:0
          [ Nx.add a (Nx.mul (Nx.sub b a) s); Nx.reshape [| 1 |] b ])
  in
  (* Axis k's coordinates, of shape [[P; m]], broadcast at axes 2k and 2k + 1 of
     the interleaved grid [[P; m; …; P; m]]. *)
  let grid = Array.concat (List.init d (fun _ -> [| pieces; m |])) in
  let coordinate k b =
    let lo = Nx.slice [ Nx.R (0, pieces) ] b
    and hi = Nx.slice [ Nx.R (1, pieces + 1) ] b in
    let mid = Nx.div_s (Nx.add lo hi) 2.
    and half = Nx.div_s (Nx.sub hi lo) 2. in
    let x =
      Nx.add
        (Nx.reshape [| pieces; 1 |] mid)
        (Nx.mul (Nx.reshape [| pieces; 1 |] half) (Nx.reshape [| 1; m |] u))
    in
    let shape =
      Array.init (2 * d) (fun a ->
          if a = 2 * k then pieces else if a = (2 * k) + 1 then m else 1)
    in
    Nx.broadcast_to grid (Nx.reshape shape x)
  in
  let points = Nx.stack ~axis:(-1) (List.mapi coordinate breaks) in
  let values = f points in
  if Nx.shape values <> grid then
    fail fn "the function returned shape %s for points of shape %s"
      (Num.shape (Nx.shape values))
      (Num.shape (Nx.shape points));
  let fit k c =
    Cheb.fit (to_front [ 2 * k; (2 * k) + 1 ] c)
    |> of_front [ 2 * k; (2 * k) + 1 ]
  in
  let c = ref values in
  for k = 0 to d - 1 do
    c := fit k !c
  done;
  let coefficients =
    Nx.transpose
      ~axes:(List.init d (fun k -> 2 * k) @ List.init d (fun k -> (2 * k) + 1))
      !c
  in
  { breaks; coefficients }

(* Evaluation *)

let eval g x =
  let fn = "Jera.Grid.eval" in
  let d = dims g in
  let shape = Nx.shape x in
  let rank = Array.length shape in
  if rank = 0 || shape.(rank - 1) <> d then
    fail fn "points of shape %s for a grid of %d axes" (Num.shape shape) d;
  let q = Array.sub shape 0 (rank - 1) in
  let n = Array.fold_left ( * ) 1 q in
  let flat = Nx.reshape [| n; d |] x in
  let located =
    List.mapi
      (fun k b ->
        Cheb.locate fn Cheb.Bounded b (Nx.slice [ Nx.A; Nx.I k ] flat))
      g.breaks
  in
  (* The pieces' axes flattened in C order, then one gather per point. *)
  let index =
    List.fold_left2
      (fun acc b (i, _) ->
        Nx.add (Nx.mul_s acc (Int64.of_int (Nx.dim 0 b - 1))) i)
      (Nx.zeros Nx.int64 [| n |])
      g.breaks located
  in
  let shape = Nx.shape g.coefficients in
  let total = Array.fold_left ( * ) 1 (Array.sub shape 0 d) in
  let c =
    Nx.reshape
      (Array.append [| total |] (Array.sub shape d (Array.length shape - d)))
      g.coefficients
  in
  let y =
    List.fold_left
      (fun y (_, u) -> Cheb.clenshaw u y)
      (Nx.take ~axis:0 ~indices:index c)
      located
  in
  (* A NaN coordinate is NaN whatever the degrees. *)
  let nan = Nx.any ~axes:[ 1 ] (Nx.isnan flat) in
  let value = Array.sub shape (2 * d) (Array.length shape - (2 * d)) in
  let mask =
    Nx.reshape (Array.append [| n |] (Array.make (Array.length value) 1)) nan
  in
  let y = Nx.where mask (Nx.full_like y Float.nan) y in
  Nx.reshape (Array.append q value) y

(* Calculus *)

let along fn op ~axis g =
  let d = dims g in
  if axis < 0 || axis >= d then fail fn "axis = %d is not in [0, %d)" axis d;
  let b = List.nth g.breaks axis in
  let p = Nx.dim 0 b - 1 in
  let widths =
    Nx.sub (Nx.slice [ Nx.R (1, p + 1) ] b) (Nx.slice [ Nx.R (0, p) ] b)
  in
  let axes = [ axis; d + axis ] in
  {
    g with
    coefficients = of_front axes (op widths (to_front axes g.coefficients));
  }

let derivative ~axis g = along "Jera.Grid.derivative" Cheb.derivative ~axis g
let integral ~axis g = along "Jera.Grid.integral" Cheb.integral ~axis g

(* Access *)

let ptree (type b) () : b t Nx.Ptree.t =
  let module M = struct
    type nonrec _ t = b t

    let walk c g =
      let open Nx.Ptree.Walk in
      let breaks = field c "breaks" (list tensor) g.breaks in
      let coefficients = field c "coefficients" tensor g.coefficients in
      { breaks; coefficients }
  end in
  Nx.Ptree.instantiate (module M)
