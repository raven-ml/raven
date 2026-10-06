(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* A parameter's domain check, run on use: [run context unless] checks the
   parameter where [unless] does not hold, its message starting with [context];
   [inside ()] is whether every element is in the domain. *)
type check = {
  run : string -> Nx.bool_t option -> unit;
  inside : unit -> Nx.bool_t;
}

type ('x, 'f) kind =
  | Continuous : ((float, 'f) Nx.t, 'f) kind
  | Counts : (Nx.int32_t, 'f) kind
  | Categories : (Nx.int64_t, 'f) kind
  | Booleans : (Nx.bool_t, 'f) kind

type ('x, 'f) t = {
  family : string;
  descr : string;
  dtype : (float, 'f) Nx.dtype;
  shape : int array;
  checks : check list;
  checked : bool;
  support : unit -> Support.t;
  bounds : unit -> (float, 'f) Nx.t * (float, 'f) Nx.t;
      (* each element's least and greatest value, unbroadcast *)
  factors : 'x -> (float, 'f) Nx.t;
  draw : Nx.Rng.t -> int array -> 'x; (* at a value shape *)
  coords : 'f Bij.t option;
  standardize : 'f Bij.t option;
  quantile : ((float, 'f) Nx.t -> (float, 'f) Nx.t) option;
  membership : ('x -> (float, 'f) Nx.t) option;
  kind : ('x, 'f) kind;
}

(* A combinator applies tensor operations to values whose type only their family
   knows: [apply k f x] applies [f], polymorphic in the tensor's type, to [x],
   of kind [k]. *)
type map = { f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t }

let apply : type x f. (x, f) kind -> map -> x -> x =
 fun kind m x ->
  match kind with
  | Continuous -> m.f x
  | Counts -> m.f x
  | Categories -> m.f x
  | Booleans -> m.f x

let ndim : type x f. (x, f) kind -> x -> int =
 fun kind x ->
  match kind with
  | Continuous -> Nx.ndim x
  | Counts -> Nx.ndim x
  | Categories -> Nx.ndim x
  | Booleans -> Nx.ndim x

(* [select axis i x] is [x] at the indices [i] along [axis], which it
   removes. *)
let select axis i =
  {
    f =
      (fun x ->
        let lead = Array.sub (Nx.shape x) 0 axis in
        let rest = Array.sub (Nx.shape x) (axis + 1) (Nx.ndim x - axis - 1) in
        let shape = Array.concat [ lead; [| 1 |]; rest ] in
        let i =
          Nx.reshape
            (Array.concat [ lead; [| 1 |]; Array.map (fun _ -> 1) rest ])
            i
        in
        Nx.squeeze ~axes:[ axis ]
          (Nx.take_along_axis ~axis ~indices:(Nx.broadcast_to shape i) x));
  }

(* Shapes *)

let shape_string s =
  String.concat "; " (Array.to_list (Array.map string_of_int s))

let broadcast fn shapes =
  let n = List.fold_left (fun n s -> max n (Array.length s)) 0 shapes in
  let out = Array.make n 1 in
  List.iter
    (fun s ->
      let off = n - Array.length s in
      Array.iteri
        (fun i d ->
          let j = off + i in
          if out.(j) = 1 then out.(j) <- d
          else if d <> 1 && d <> out.(j) then
            invalid_argf
              "Norn.Dist.%s: parameters of shapes %s do not broadcast" fn
              (String.concat " and "
                 (List.map (fun s -> "[" ^ shape_string s ^ "]") shapes)))
        s)
    shapes;
  out

let describe family params =
  Printf.sprintf "%s(%s)" family
    (String.concat ", "
       (List.map
          (fun (name, dt, s) ->
            Printf.sprintf "%s: %s [%s]" name dt (shape_string s))
          params))

let param name x = (name, Nx_dtype.to_string (Nx.dtype x), Nx.shape x)

(* Domains *)

let const x v = Nx.scalar (Nx.dtype x) v
let neg_inf x = const x Float.neg_infinity
let positive x = Nx.greater x (const x 0.)
let non_negative x = Nx.greater_equal x (const x 0.)
let finite x = Nx.isfinite x
let below_infinity x = Nx.less x (const x Float.infinity)
let not_nan x = Nx.logical_not (Nx.isnan x)

let check family param domain inside x =
  let run context unless =
    let ok = inside x in
    let ok = match unless with None -> ok | Some u -> Nx.logical_or ok u in
    Nx.check Nx.Ptree.tensor ok x (fun i v ->
        (* [unless] may broadcast [ok] past [x]: [x]'s place is the trailing
           entries. *)
        let n = Nx.ndim x and k = Array.length i in
        let at =
          if n = 0 then ""
          else Printf.sprintf " at [%s]" (shape_string (Array.sub i (k - n) n))
        in
        Invalid_argument
          (Printf.sprintf "%s: %s: %s%s is %s, not in %s" context family param
             at (Nx.to_string v) domain))
  in
  { run; inside = (fun () -> Nx.all (inside x)) }

let in_positive family name x = check family name "(0, inf)" positive x
let in_reals family name x = check family name "(-inf, inf)" finite x

let run_checks ?unless context checks =
  List.iter (fun c -> c.run context unless) checks

let checked context d =
  if not d.checked then run_checks ("Norn.Dist." ^ context) d.checks

(* Arithmetic *)

let half_log_2pi = 0.5 *. Float.log (2. *. Float.pi)
let log_pi = Float.log Float.pi

let softplus x =
  Nx.add (Nx.maximum x (const x 0.)) (Nx.log1p (Nx.exp (Nx.neg (Nx.abs x))))

let log_sigmoid x = Nx.neg (softplus (Nx.neg x))

(* [xlogy x y] is [x log y], [0] where [x] is [0]. *)
let xlogy x y =
  Nx.where (Nx.equal x (const x 0.)) (const x 0.) (Nx.mul x (Nx.log y))

(* [inside ok x safe] is [x] where [ok] holds and [safe] elsewhere: a value
   outside the support enters the density's arithmetic as [safe], so neither its
   value nor its derivative is NaN, and the factor is then set to [-inf]. *)
let inside ok x safe = Nx.where ok x (const x safe)
let outside ok f = Nx.where ok f (neg_inf f)

(* [open_unit k dt s] is uniform on [(0, 1)]: a draw of [Nx.Rng.uniform], a
   multiple of [2^-p], moved by half a step, so neither end is reached. *)
let open_unit k dt s = Nx.add_s (Nx.Rng.uniform k dt s) (Prec.eps dt /. 4.)
let broadcast_to s x = Nx.broadcast_to s x

(* Families *)

(* A continuous value's bounds are its coordinates' image of the reals. *)
let coord_bounds dtype coords () =
  Bijection.bounds coords
    (Nx.scalar dtype Float.neg_infinity, Nx.scalar dtype Float.infinity)

let continuous ~family ~params ~shape ~checks ~support ~factors ~draw ~coords
    ?standardize ?quantile dtype =
  {
    family;
    descr = describe family params;
    dtype;
    shape;
    checks;
    checked = false;
    support;
    bounds = coord_bounds dtype coords;
    factors;
    draw;
    coords = Some coords;
    standardize;
    quantile;
    membership = None;
    kind = Continuous;
  }

let location_scale family ~loc ~scale ~density ~draw ?quantile () =
  let dt = Nx.dtype loc in
  continuous ~family
    ~params:[ param "loc" loc; param "scale" scale ]
    ~shape:(broadcast family [ Nx.shape loc; Nx.shape scale ])
    ~checks:[ in_reals family "loc" loc; in_positive family "scale" scale ]
    ~support:(fun () -> Support.Real)
    ~factors:(fun x ->
      Nx.sub (density (Nx.div (Nx.sub x loc) scale)) (Nx.log scale))
    ~draw:(fun k s -> Nx.add loc (Nx.mul scale (draw k dt s)))
    ~coords:Bij.identity ~standardize:(Bij.affine ~loc ~scale)
    ?quantile:(Option.map (fun q p -> Nx.add loc (Nx.mul scale (q p))) quantile)
    dt

let normal ~loc ~scale =
  location_scale "normal" ~loc ~scale
    ~density:(fun z ->
      Nx.add_s (Nx.mul_s (Nx.square z) (-0.5)) (-.half_log_2pi))
    ~draw:Nx.Rng.normal ~quantile:Nx.ndtri ()

let cauchy_quantile p = Nx.tan (Nx.mul_s (Nx.sub_s p 0.5) Float.pi)

let cauchy ~loc ~scale =
  location_scale "cauchy" ~loc ~scale
    ~density:(fun z -> Nx.add_s (Nx.neg (Nx.log1p (Nx.square z))) (-.log_pi))
    ~draw:(fun k dt s -> cauchy_quantile (open_unit k dt s))
    ~quantile:cauchy_quantile ()

let laplace_quantile p =
  let d = Nx.sub_s p 0.5 in
  Nx.neg (Nx.mul (Nx.sign d) (Nx.log1p (Nx.mul_s (Nx.abs d) (-2.))))

let laplace ~loc ~scale =
  location_scale "laplace" ~loc ~scale
    ~density:(fun z -> Nx.sub_s (Nx.neg (Nx.abs z)) (Float.log 2.))
    ~draw:(fun k dt s -> laplace_quantile (open_unit k dt s))
    ~quantile:laplace_quantile ()

let logistic_quantile p = Nx.sub (Nx.log p) (Nx.log1p (Nx.neg p))

let logistic ~loc ~scale =
  location_scale "logistic" ~loc ~scale
    ~density:(fun z -> Nx.sub (Nx.neg z) (Nx.mul_s (softplus (Nx.neg z)) 2.))
    ~draw:(fun k dt s -> logistic_quantile (open_unit k dt s))
    ~quantile:logistic_quantile ()

let student_t ~df ~loc ~scale =
  let family = "student_t" in
  let dt = Nx.dtype loc in
  let half_df = Nx.mul_s df 0.5 in
  let half_df1 = Nx.add_s half_df 0.5 in
  let normaliser =
    Nx.sub
      (Nx.sub (Nx.lgamma half_df1) (Nx.lgamma half_df))
      (Nx.mul_s (Nx.add_s (Nx.log df) log_pi) 0.5)
  in
  let factors x =
    let z = Nx.div (Nx.sub x loc) scale in
    Nx.sub
      (Nx.sub normaliser (Nx.log scale))
      (Nx.mul half_df1 (Nx.log1p (Nx.div (Nx.square z) df)))
  in
  (* A normal over the square root of a chi-square over its degrees of freedom:
     [z sqrt (df / 2g)], [g] gamma of concentration [df / 2]. *)
  let draw k s =
    let k1, k2 =
      match Nx.Rng.split k with [| a; b |] -> (a, b) | _ -> assert false
    in
    let z = Nx.Rng.normal k1 dt s in
    let g = Nx.Rng.gamma k2 (broadcast_to s half_df) in
    Nx.add loc (Nx.mul scale (Nx.mul z (Nx.sqrt (Nx.div half_df g))))
  in
  continuous ~family
    ~params:[ param "df" df; param "loc" loc; param "scale" scale ]
    ~shape:(broadcast family [ Nx.shape df; Nx.shape loc; Nx.shape scale ])
    ~checks:
      [
        in_positive family "df" df;
        in_reals family "loc" loc;
        in_positive family "scale" scale;
      ]
    ~support:(fun () -> Support.Real)
    ~factors ~draw ~coords:Bij.identity ~standardize:(Bij.affine ~loc ~scale) dt

let lognormal ~loc ~scale =
  let family = "lognormal" in
  let dt = Nx.dtype loc in
  let factors x =
    let ok = positive x in
    let lx = Nx.log (inside ok x 1.) in
    let z = Nx.div (Nx.sub lx loc) scale in
    outside ok
      (Nx.sub
         (Nx.add_s (Nx.mul_s (Nx.square z) (-0.5)) (-.half_log_2pi))
         (Nx.add lx (Nx.log scale)))
  in
  continuous ~family
    ~params:[ param "loc" loc; param "scale" scale ]
    ~shape:(broadcast family [ Nx.shape loc; Nx.shape scale ])
    ~checks:[ in_reals family "loc" loc; in_positive family "scale" scale ]
    ~support:(fun () -> Support.Greater 0.)
    ~factors
    ~draw:(fun k s -> Nx.exp (Nx.add loc (Nx.mul scale (Nx.Rng.normal k dt s))))
    ~coords:Bij.exp
    ~standardize:(Bij.compose Bij.exp (Bij.affine ~loc ~scale))
    ~quantile:(fun p -> Nx.exp (Nx.add loc (Nx.mul scale (Nx.ndtri p))))
    dt

(* A family on [(0, inf)] whose density at [x] is [density x], the value
   replaced by [1] outside before the arithmetic. *)
let positive_family ~family ~params ~shape ~checks ~density ~draw ?quantile
    ~closed dt =
  let factors x =
    let ok = if closed then non_negative x else positive x in
    outside ok (density (inside ok x 1.))
  in
  continuous ~family ~params ~shape ~checks
    ~support:(fun () -> Support.Greater 0.)
    ~factors ~draw ~coords:Bij.exp ?quantile dt

let half_normal ~scale =
  let family = "half_normal" in
  let dt = Nx.dtype scale in
  positive_family ~family ~closed:true
    ~params:[ param "scale" scale ]
    ~shape:(Nx.shape scale)
    ~checks:[ in_positive family "scale" scale ]
    ~density:(fun x ->
      let z = Nx.div x scale in
      Nx.sub
        (Nx.add_s
           (Nx.mul_s (Nx.square z) (-0.5))
           (Float.log 2. -. half_log_2pi))
        (Nx.log scale))
    ~draw:(fun k s -> Nx.mul scale (Nx.abs (Nx.Rng.normal k dt s)))
    ~quantile:(fun p -> Nx.mul scale (Nx.ndtri (Nx.mul_s (Nx.add_s p 1.) 0.5)))
    dt

let half_cauchy ~scale =
  let family = "half_cauchy" in
  let dt = Nx.dtype scale in
  let quantile p = Nx.mul scale (Nx.tan (Nx.mul_s p (Float.pi /. 2.))) in
  positive_family ~family ~closed:true
    ~params:[ param "scale" scale ]
    ~shape:(Nx.shape scale)
    ~checks:[ in_positive family "scale" scale ]
    ~density:(fun x ->
      Nx.sub
        (Nx.add_s
           (Nx.neg (Nx.log1p (Nx.square (Nx.div x scale))))
           (Float.log 2. -. log_pi))
        (Nx.log scale))
    ~draw:(fun k s -> quantile (open_unit k dt s))
    ~quantile dt

let exponential ~rate =
  let family = "exponential" in
  let dt = Nx.dtype rate in
  positive_family ~family ~closed:true
    ~params:[ param "rate" rate ]
    ~shape:(Nx.shape rate)
    ~checks:[ in_positive family "rate" rate ]
    ~density:(fun x -> Nx.sub (Nx.log rate) (Nx.mul rate x))
    ~draw:(fun k s -> Nx.div (Nx.Rng.exponential k dt s) rate)
    ~quantile:(fun p -> Nx.div (Nx.neg (Nx.log1p (Nx.neg p))) rate)
    dt

let gamma ~concentration ~rate =
  let family = "gamma" in
  let dt = Nx.dtype rate in
  let a = concentration in
  positive_family ~family ~closed:false
    ~params:[ param "concentration" a; param "rate" rate ]
    ~shape:(broadcast family [ Nx.shape a; Nx.shape rate ])
    ~checks:
      [ in_positive family "concentration" a; in_positive family "rate" rate ]
    ~density:(fun x ->
      Nx.sub
        (Nx.add
           (Nx.sub (Nx.mul a (Nx.log rate)) (Nx.lgamma a))
           (Nx.mul (Nx.sub_s a 1.) (Nx.log x)))
        (Nx.mul rate x))
    ~draw:(fun k s -> Nx.div (Nx.Rng.gamma k (broadcast_to s a)) rate)
    dt

let inverse_gamma ~concentration ~scale =
  let family = "inverse_gamma" in
  let dt = Nx.dtype scale in
  let a = concentration in
  positive_family ~family ~closed:false
    ~params:[ param "concentration" a; param "scale" scale ]
    ~shape:(broadcast family [ Nx.shape a; Nx.shape scale ])
    ~checks:
      [ in_positive family "concentration" a; in_positive family "scale" scale ]
    ~density:(fun x ->
      Nx.sub
        (Nx.sub
           (Nx.sub (Nx.mul a (Nx.log scale)) (Nx.lgamma a))
           (Nx.mul (Nx.add_s a 1.) (Nx.log x)))
        (Nx.div scale x))
    ~draw:(fun k s -> Nx.div scale (Nx.Rng.gamma k (broadcast_to s a)))
    dt

let beta ~a ~b =
  let family = "beta" in
  let dt = Nx.dtype a in
  let factors x =
    let ok = Nx.logical_and (positive x) (Nx.less x (const x 1.)) in
    let x = inside ok x 0.5 in
    outside ok
      (Nx.sub
         (Nx.add
            (Nx.mul (Nx.sub_s a 1.) (Nx.log x))
            (Nx.mul (Nx.sub_s b 1.) (Nx.log1p (Nx.neg x))))
         (Nx.lbeta a b))
  in
  continuous ~family
    ~params:[ param "a" a; param "b" b ]
    ~shape:(broadcast family [ Nx.shape a; Nx.shape b ])
    ~checks:[ in_positive family "a" a; in_positive family "b" b ]
    ~support:(fun () -> Support.Interval (0., 1.))
    ~factors
    ~draw:(fun k s -> Nx.Rng.beta k (broadcast_to s a) (broadcast_to s b))
    ~coords:(Bij.interval ~low:(Nx.scalar dt 0.) ~high:(Nx.scalar dt 1.))
    dt

let uniform ~low ~high =
  let family = "uniform" in
  let dt = Nx.dtype low in
  let width = Nx.sub high low in
  let factors x =
    let ok = Nx.logical_and (Nx.greater_equal x low) (Nx.less_equal x high) in
    outside ok (Nx.broadcast_to (Nx.shape ok) (Nx.neg (Nx.log width)))
  in
  let coords = Bij.interval ~low ~high in
  continuous ~family
    ~params:[ param "low" low; param "high" high ]
    ~shape:(broadcast family [ Nx.shape low; Nx.shape high ])
    ~checks:
      [
        check family "high - low" "(0, inf)"
          (fun w -> Nx.logical_and (positive w) (finite w))
          width;
      ]
    ~support:(fun () -> Bijection.image coords Support.Real [||])
    ~factors
    ~draw:(fun k s -> Nx.add low (Nx.mul width (Nx.Rng.uniform k dt s)))
    ~coords
    ~quantile:(fun p -> Nx.add low (Nx.mul width p))
    dt

let dirichlet ~concentration =
  let family = "dirichlet" in
  let a = concentration in
  let dt = Nx.dtype a in
  let s = Nx.shape a in
  let n = Array.length s in
  if n = 0 || s.(n - 1) < 2 then
    invalid_argf
      "Norn.Dist.dirichlet: concentration of shape [%s] has fewer than two \
       components"
      (shape_string s);
  let k = s.(n - 1) in
  let last x = Nx.ndim x - 1 in
  let tol = 64. *. float_of_int k *. Prec.eps dt in
  let factors x =
    let x = Nx.broadcast_to (broadcast family [ Nx.shape x; s ]) x in
    let ax = last x in
    let ok =
      Nx.logical_and
        (Nx.all ~axes:[ ax ]
           (Nx.logical_and (positive x) (Nx.less_equal x (const x 1.))))
        (Nx.less_equal
           (Nx.abs (Nx.sub_s (Nx.sum ~axes:[ ax ] x) 1.))
           (const x tol))
    in
    let x =
      Nx.where (Nx.unsqueeze ~axes:[ ax ] ok) x (const x (1. /. float_of_int k))
    in
    let normaliser =
      Nx.sub
        (Nx.lgamma (Nx.sum ~axes:[ n - 1 ] a))
        (Nx.sum ~axes:[ n - 1 ] (Nx.lgamma a))
    in
    outside ok
      (Nx.add normaliser
         (Nx.sum ~axes:[ ax ] (Nx.mul (Nx.sub_s a 1.) (Nx.log x))))
  in
  continuous ~family
    ~params:[ param "concentration" a ]
    ~shape:s
    ~checks:[ in_positive family "concentration" a ]
    ~support:(fun () -> Support.Simplex k)
    ~factors
    ~draw:(fun key shape -> Nx.Rng.dirichlet key (broadcast_to shape a))
    ~coords:Bij.simplex dt

let mvn ~loc ~scale_tril =
  let family = "mvn" in
  let dt = Nx.dtype loc in
  let ls = Nx.shape loc and ms = Nx.shape scale_tril in
  let nl = Array.length ls and nm = Array.length ms in
  if
    nl = 0 || nm < 2 || ms.(nm - 1) <> ms.(nm - 2) || ms.(nm - 1) <> ls.(nl - 1)
  then
    invalid_argf
      "Norn.Dist.mvn: loc of shape [%s] and scale_tril of shape [%s] do not \
       describe vectors of one length"
      (shape_string ls) (shape_string ms);
  let d = ls.(nl - 1) in
  let lower = Nx.tril scale_tril in
  let shape =
    Array.append
      (broadcast family [ Array.sub ls 0 (nl - 1); Array.sub ms 0 (nm - 2) ])
      [| d |]
  in
  let log_diag = Nx.log (Nx.diagonal lower) in
  let factors x =
    let diff = Nx.sub x loc in
    let n = Nx.ndim diff in
    let batch =
      broadcast family
        [ Array.sub (Nx.shape diff) 0 (n - 1); Array.sub ms 0 (nm - 2) ]
    in
    let l = Nx.broadcast_to (Array.append batch [| d; d |]) lower in
    let diff = Nx.broadcast_to (Array.append batch [| d |]) diff in
    let z =
      Nx.solve_triangular l (Nx.unsqueeze ~axes:[ Array.length batch + 1 ] diff)
    in
    let nb = Array.length batch in
    let quad = Nx.sum ~axes:[ nb; nb + 1 ] (Nx.square z) in
    Nx.sub
      (Nx.add_s (Nx.mul_s quad (-0.5)) (-.float_of_int d *. half_log_2pi))
      (Nx.sum ~axes:[ Nx.ndim log_diag - 1 ] log_diag)
  in
  let draw k s =
    let z = Nx.Rng.normal k dt s in
    let n = Array.length s in
    Nx.add loc
      (Nx.squeeze ~axes:[ n ] (Nx.matmul lower (Nx.unsqueeze ~axes:[ n ] z)))
  in
  continuous ~family
    ~params:[ param "loc" loc; param "scale_tril" scale_tril ]
    ~shape
    ~checks:
      [
        in_reals family "loc" loc;
        in_positive family "scale_tril's diagonal" (Nx.diagonal scale_tril);
      ]
    ~support:(fun () -> Support.Real)
    ~factors ~draw ~coords:Bij.identity
    ~standardize:(Bij.affine_tril ~loc ~scale_tril)
    dt

(* Discrete families *)

let discrete ~family ~kind ~params ~shape ~checks ~support ~bounds ~factors
    ~draw dtype =
  {
    family;
    descr = describe family params;
    dtype;
    shape;
    checks;
    checked = false;
    support = (fun () -> support);
    bounds =
      (fun () -> (Nx.scalar dtype (fst bounds), Nx.scalar dtype (snd bounds)));
    factors;
    draw;
    coords = None;
    standardize = None;
    quantile = None;
    membership = None;
    kind;
  }

let bernoulli ~logits =
  let family = "bernoulli" in
  discrete ~family ~kind:Booleans
    ~params:[ param "logits" logits ]
    ~shape:(Nx.shape logits)
    ~checks:[ check family "logits" "[-inf, inf]" not_nan logits ]
    ~support:Support.Boolean ~bounds:(0., 1.)
    ~factors:(fun x ->
      Nx.where x (log_sigmoid logits) (log_sigmoid (Nx.neg logits)))
    ~draw:(fun k s -> Nx.Rng.bernoulli k (Nx.sigmoid (broadcast_to s logits)))
    (Nx.dtype logits)

let poisson ~rate =
  let family = "poisson" in
  discrete ~family ~kind:Counts
    ~params:[ param "rate" rate ]
    ~shape:(Nx.shape rate)
    ~checks:
      [
        check family "rate" "[0, inf)"
          (fun r -> Nx.logical_and (non_negative r) (below_infinity r))
          rate;
      ]
    ~support:(Support.Integers_from 0) ~bounds:(0., Float.infinity)
    ~factors:(fun x ->
      let ok = Nx.greater_equal x (Nx.scalar Nx.int32 0l) in
      let x = Nx.cast (Nx.dtype rate) (Nx.where ok x (Nx.scalar Nx.int32 0l)) in
      outside ok
        (Nx.sub (Nx.sub (xlogy x rate) rate) (Nx.lgamma (Nx.add_s x 1.))))
    ~draw:(fun k s -> Nx.Rng.poisson k (broadcast_to s rate))
    (Nx.dtype rate)

let neg_binomial ~mean ~dispersion =
  let family = "neg_binomial" in
  let r = dispersion in
  discrete ~family ~kind:Counts
    ~params:[ param "mean" mean; param "dispersion" dispersion ]
    ~shape:(broadcast family [ Nx.shape mean; Nx.shape r ])
    ~checks:
      [
        check family "mean" "[0, inf)"
          (fun m -> Nx.logical_and (non_negative m) (below_infinity m))
          mean;
        in_positive family "dispersion" r;
      ]
    ~support:(Support.Integers_from 0) ~bounds:(0., Float.infinity)
    ~factors:(fun x ->
      let ok = Nx.greater_equal x (Nx.scalar Nx.int32 0l) in
      let x = Nx.cast (Nx.dtype mean) (Nx.where ok x (Nx.scalar Nx.int32 0l)) in
      let total = Nx.log (Nx.add r mean) in
      let f =
        Nx.add
          (Nx.sub
             (Nx.sub (Nx.lgamma (Nx.add x r)) (Nx.lgamma r))
             (Nx.lgamma (Nx.add_s x 1.)))
          (Nx.sub
             (Nx.add (Nx.mul r (Nx.sub (Nx.log r) total)) (xlogy x mean))
             (Nx.mul x total))
      in
      outside ok f)
    ~draw:(fun k s ->
      let k1, k2 =
        match Nx.Rng.split k with [| a; b |] -> (a, b) | _ -> assert false
      in
      let rate = Nx.mul (Nx.Rng.gamma k1 (broadcast_to s r)) (Nx.div mean r) in
      Nx.Rng.poisson k2 rate)
    (Nx.dtype mean)

let categorical ~logits =
  let family = "categorical" in
  let s = Nx.shape logits in
  let n = Array.length s in
  if n = 0 || s.(n - 1) = 0 then
    invalid_argf "Norn.Dist.categorical: logits of shape [%s] have no category"
      (shape_string s);
  let k = s.(n - 1) in
  let batch = Array.sub s 0 (n - 1) in
  let factors x =
    let lp = Nx.log_softmax ~axes:[ n - 1 ] logits in
    let b = broadcast family [ Nx.shape x; batch ] in
    let lp = Nx.broadcast_to (Array.append b [| k |]) lp in
    let ok =
      Nx.logical_and
        (Nx.greater_equal x (Nx.scalar Nx.int64 0L))
        (Nx.less x (Nx.scalar Nx.int64 (Int64.of_int k)))
    in
    let i = Nx.unsqueeze ~axes:[ Array.length b ] (Nx.broadcast_to b x) in
    let f =
      Nx.squeeze
        ~axes:[ Array.length b ]
        (Nx.take_along_axis ~axis:(Array.length b) ~indices:i lp)
    in
    outside (Nx.broadcast_to b ok) f
  in
  discrete ~family ~kind:Categories
    ~params:[ param "logits" logits ]
    ~shape:batch
    ~checks:
      [
        check family "logits" "[-inf, inf)"
          (fun l -> Nx.logical_and (not_nan l) (below_infinity l))
          logits;
      ]
    ~support:(Support.Integer_interval (0, k - 1))
    ~bounds:(0., float_of_int (k - 1))
    ~factors
    ~draw:(fun key shape ->
      Nx.Rng.categorical key (broadcast_to (Array.append shape [| k |]) logits))
    (Nx.dtype logits)

(* Combinators *)

let iid s d =
  Array.iter
    (fun n ->
      if n < 0 then
        invalid_argf "Norn.Dist.iid: shape [%s] has a negative length"
          (shape_string s))
    s;
  {
    d with
    descr = Printf.sprintf "iid [%s] %s" (shape_string s) d.descr;
    shape = Array.append s d.shape;
  }

let log_factorial n =
  let s = ref 0. in
  for k = 2 to n do
    s := !s +. Float.log (float_of_int k)
  done;
  !s

let sorted n d =
  if Array.length d.shape <> 0 then
    invalid_argf "Norn.Dist.sorted: %s is not over scalars" d.descr;
  if n < 1 then invalid_argf "Norn.Dist.sorted: n = %d is not positive" n;
  let coords = match d.coords with Some b -> b | None -> assert false in
  let factors x =
    let ax = Nx.ndim x - 1 in
    let m = (Nx.shape x).(ax) in
    let range a b =
      Nx.shrink
        (Array.mapi (fun i l -> if i = ax then (a, b) else (0, l)) (Nx.shape x))
        x
    in
    let ok =
      Nx.all ~axes:[ ax ] (Nx.greater_equal (range 1 m) (range 0 (m - 1)))
    in
    outside ok (Nx.add_s (Nx.sum ~axes:[ ax ] (d.factors x)) (log_factorial n))
  in
  {
    d with
    family = "sorted";
    descr = Printf.sprintf "sorted %d %s" n d.descr;
    shape = [| n |];
    support = (fun () -> Support.Ordered);
    factors;
    draw = (fun k s -> fst (Nx.sort ~axis:(Array.length s - 1) (d.draw k s)));
    coords = Some (Bij.compose coords Bij.ordered);
    bounds = coord_bounds d.dtype (Bij.compose coords Bij.ordered);
    standardize = None;
    quantile = None;
    membership = None;
  }

(* [sum_trailing k x] sums the last [k] axes of [x]. *)
let sum_trailing k x =
  if k <= 0 then x
  else Nx.sum ~axes:(List.init k (fun i -> Nx.ndim x - 1 - i)) x

let transform b d =
  let coords = match d.coords with Some c -> c | None -> assert false in
  let shape = Nx.shape (fst (Bij.forward b (Nx.zeros d.dtype d.shape))) in
  let factors y =
    let u = Bij.inverse b y in
    let _, ld = Bij.forward b u in
    let f = d.factors u in
    let r = Nx.ndim f - Nx.ndim ld in
    let lp = Nx.sub (sum_trailing r f) (sum_trailing (-r) ld) in
    Nx.where (Nx.isnan lp) (neg_inf lp) lp
  in
  {
    d with
    family = "transform";
    descr = Format.asprintf "transform %a %s" Bij.pp b d.descr;
    shape;
    support = (fun () -> Bijection.image b (d.support ()) shape);
    factors;
    draw = (fun k s -> fst (Bij.forward b (d.draw k (Bij.shape b s))));
    coords = Some (Bij.compose b coords);
    bounds = coord_bounds d.dtype (Bij.compose b coords);
    standardize = None;
    quantile = None;
    membership = None;
  }

(* [hull_coords family d] is the bijector onto the support of every component of
   [d] together: the hull of their supports, read on the host. *)
let hull_coords family d =
  match d.coords with
  | None -> None
  | Some _ -> (
      let c v = Nx.scalar d.dtype v in
      match d.support () with
      | Support.Real -> Some Bij.identity
      | Support.Greater 0. -> Some Bij.exp
      | Support.Greater a -> Some (Bij.greater ~low:(c a))
      | Support.Interval (a, b) -> Some (Bij.interval ~low:(c a) ~high:(c b))
      | Support.Simplex _ -> Some Bij.simplex
      | Support.Ordered -> Some Bij.ordered
      | Support.Sum_to_zero -> Some Bij.sum_to_zero
      | Support.Correlation_cholesky _ -> Some Bij.cholesky_corr
      | Support.Integers_from _ | Support.Integer_interval _ | Support.Boolean
        ->
          invalid_argf "Norn.Dist.%s: %s has no coordinates" family d.family)

let mixture ~logits d =
  let family = "mixture" in
  let ls = Nx.shape logits in
  if Array.length ls <> 1 || Array.length d.shape = 0 || d.shape.(0) <> ls.(0)
  then
    invalid_argf
      "Norn.Dist.mixture: logits of shape [%s] do not match %s of shape [%s]"
      (shape_string ls) d.descr (shape_string d.shape);
  let k = ls.(0) in
  let rest = Array.sub d.shape 1 (Array.length d.shape - 1) in
  let hull = hull_coords family d in
  (* Each component's log weight plus its log density of [x], with the component
     axis after [x]'s leading axes. *)
  let joint x =
    let r = ndim d.kind x - Array.length rest in
    let f =
      d.factors (apply d.kind { f = (fun x -> Nx.unsqueeze ~axes:[ r ] x) } x)
    in
    let f = sum_trailing (Nx.ndim f - r - 1) f in
    (r, Nx.add f (Nx.log_softmax ~axes:[ 0 ] logits))
  in
  let factors x =
    let r, j = joint x in
    Nx.logsumexp ~axes:[ r ] j
  in
  let membership x =
    let r, j = joint x in
    Nx.exp (Nx.log_softmax ~axes:[ r ] j)
  in
  let draw key s =
    let k1, k2 =
      match Nx.Rng.split key with [| a; b |] -> (a, b) | _ -> assert false
    in
    let batch = Array.sub s 0 (Array.length s - Array.length rest) in
    let r = Array.length batch in
    let all = d.draw k1 (Array.concat [ batch; [| k |]; rest ]) in
    let c =
      Nx.Rng.categorical k2 (broadcast_to (Array.append batch [| k |]) logits)
    in
    apply d.kind (select r c) all
  in
  {
    d with
    family;
    descr =
      Printf.sprintf "mixture(logits: %s [%d], %s)"
        (Nx_dtype.to_string (Nx.dtype logits))
        k d.descr;
    shape = rest;
    checks =
      d.checks
      @ [
          check family "logits" "[-inf, inf)"
            (fun l -> Nx.logical_and (not_nan l) (below_infinity l))
            logits;
        ];
    factors;
    draw;
    coords = hull;
    bounds =
      (match hull with Some h -> coord_bounds d.dtype h | None -> d.bounds);
    standardize = None;
    quantile = None;
    membership = Some membership;
  }

(* Eliminators *)

let factors d x =
  checked "factors" d;
  d.factors x

let log_density d x =
  checked "log_density" d;
  Nx.sum (d.factors x)

let sample k d =
  checked "sample" d;
  d.draw k d.shape

let quantile d p =
  match d.quantile with
  | None -> invalid_argf "Norn.Dist.quantile: %s has no quantile" d.family
  | Some q ->
      checked "quantile" d;
      q p

let coords d =
  match d.coords with
  | Some b -> b
  | None -> invalid_argf "Norn.Dist.coords: %s is discrete" d.family

let standardize d =
  match d.standardize with
  | Some b -> b
  | None ->
      invalid_argf "Norn.Dist.standardize: %s has no standard form" d.family

let mixture_membership d x =
  match d.membership with
  | Some m ->
      checked "mixture_membership" d;
      m x
  | None ->
      invalid_argf "Norn.Dist.mixture_membership: %s is not a mixture" d.family

let family d = d.family
let kind d = d.kind
let dtype d = d.dtype
let shape d = Array.copy d.shape
let support d = d.support ()

let bounds d =
  let lo, hi = d.bounds () in
  (Nx.broadcast_to d.shape lo, Nx.broadcast_to d.shape hi)

let pp ppf d = Format.pp_print_string ppf d.descr

let check ?unless context d =
  run_checks ?unless context d.checks;
  { d with checked = true }

let valid d =
  List.fold_left
    (fun ok c -> Nx.logical_and ok (c.inside ()))
    (Nx.scalar Nx.bool true) d.checks
