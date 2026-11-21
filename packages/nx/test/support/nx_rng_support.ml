(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

(* A block's elements, past nx.cuda's and nx.amd's tile of 1024 terms, and a
   row sampler's row length. *)
let n = 2048
let row = 8

(* Dtypes and codes *)

type fdt = F : (float, 's) Nx.dtype -> fdt
type dtype = Float16 | Bfloat16 | Float32 | Float64

let fdt = function
  | Float16 -> F Nx.float16
  | Bfloat16 -> F Nx.bfloat16
  | Float32 -> F Nx.float32
  | Float64 -> F Nx.float64

let format (F dt) = Nx.Dtype.float_format dt
let least d = (format d).min_normal *. (format d).epsilon
let greatest d = (format d).max_finite

(* [x]'s predecessor in the dtype, for a power of two [x] it holds. *)
let below_pow2 d x = x *. (1. -. ((format d).epsilon /. 2.))

(* A host value's codes: a float's bits, every NaN one code, an integer's
   value, a boolean's 0 or 1. *)
let nan_code = Int64.min_int

let codes (type v r) (y : (v, r, Nx.host) Nx.t) : int64 array =
  let float_codes bits =
    let v = Nx.to_array (Nx.cast Nx.float64 y) in
    Array.mapi (fun i b -> if Float.is_nan v.(i) then nan_code else b) bits
  in
  let narrow () =
    float_codes (Array.map Int64.of_int (Nx.to_array (Nx.bitcast Nx.int16 y)))
  in
  match Nx.dtype y with
  | Nx.Dtype.Float64 -> float_codes (Nx.to_array (Nx.bitcast Nx.int64 y))
  | Float32 ->
      float_codes
        (Array.map Int64.of_int32 (Nx.to_array (Nx.bitcast Nx.int32 y)))
  | Float16 -> narrow ()
  | Bfloat16 -> narrow ()
  | Int32 -> Array.map Int64.of_int32 (Nx.to_array y)
  | Int64 -> Nx.to_array y
  | Bool -> Array.map (fun b -> if b then 1L else 0L) (Nx.to_array y)
  | _ -> invalid_arg "Nx_rng_support: a draw of another dtype"

(* Parameters

   An element of a parameter is one of its bounds at the dtype, a value that
   every dtype holds, or a point [t] of [[-1, 1]] spread over the dtype's
   positive range on a logarithmic scale and given [t]'s sign. Every block
   opens with its parameters' bounds, bound [i] of each at element [i], and
   draws the rest. *)

type v = Bound of int | Value of float | Spread of float

type param = {
  bounds : fdt -> float list;
  value : float Gen.t;
  top : fdt -> float;  (** The greatest magnitude a spread point reaches. *)
  signed : bool;  (** Whether a spread point keeps a negative sign. *)
}

let resolve p d = function
  | Bound i -> List.nth (p.bounds d) i
  | Value x -> x
  | Spread t ->
      (* Within 1% of the range's ends, which no dtype's rounding passes. *)
      let lo = log (4. *. least d) and hi = log (0.99 *. p.top d) in
      let x = exp (lo +. (Float.abs t *. (hi -. lo))) in
      if p.signed && t < 0. then -.x else x

(* Concentrations. A round of Marsaglia and Tsang's rejection reads a
   concentration in (0, 0.5] or [1, 1.5] at 1 to 1.5, where it accepts at most
   97.31% of proposals: measured over 2·10^6 proposals, 95.2% at 1, 95.8% at
   1.1 and 97.3% at 1.5. *)
let conc_region a = (a > 0. && a <= 0.5) || (a >= 1. && a <= 1.5)

let positive =
  {
    bounds = (fun d -> [ least d; (format d).min_normal; 1.; greatest d ]);
    value =
      Gen.frequency
        [ (1, Gen.float_range 0.01 0.5); (1, Gen.float_range 1. 1.5) ];
    top = greatest;
    signed = false;
  }

(* Best and Fisher's rejection accepts at most 76.6% of proposals from a
   concentration of 2 up: measured over 2·10^6 proposals, 76.6% at 2, 67.5% at
   10 and 65.8% at 1000. *)
let kappa_region k = k >= 2. && k <= 1000.

let non_negative =
  {
    bounds = (fun d -> [ 0.; least d; 1.; greatest d ]);
    value = Gen.float_range 2. 1000.;
    top = greatest;
    signed = false;
  }

let probability =
  {
    bounds =
      (fun d -> [ 0.; 1.; 0.5; least d; 1. -. ((format d).epsilon /. 2.) ]);
    value = Gen.float_range 0. 1.;
    top = (fun _ -> 1.);
    signed = false;
  }

(* Hörmann's transformed rejection accepts at most 89.0% of Poisson proposals
   (measured over 10^6 proposals at rates from 10 to 2·10^9) and 88.6% of
   binomial ones (at means from 10 to 10^9). A Poisson rate's bounds: 0, the
   switch to rejection at 10 and the float below it, the least subnormal and
   the greatest float below 2^31. *)
let rate_region r = r >= 10. && r <= 1e4
let below_2_31 d = Float.min (greatest d) (below_pow2 d 0x1p31)

let rate =
  {
    bounds =
      (fun d ->
        [ 0.; 10.; 10. -. (8. *. (format d).epsilon); least d; below_2_31 d ]);
    value = Gen.float_range 10. 1e4;
    top = below_2_31;
    signed = false;
  }

(* Truncation bounds, as pairs of a lower and an upper bound: the whole line
   both ways, two zeros of each order, the greatest floats, and an interval far
   out in each tail. *)
let lower =
  {
    bounds =
      (fun d ->
        let m = greatest d in
        [ Float.neg_infinity; Float.infinity; 0.; -0.; m; -.m; 40.; -40. ]);
    value = Gen.float_range (-6.) 6.;
    top = greatest;
    signed = true;
  }

let upper =
  {
    lower with
    bounds =
      (fun d ->
        let m = greatest d in
        [ Float.infinity; Float.neg_infinity; -0.; 0.; m; -.m; 41.; -39. ]);
  }

(* Logits. A row whose logits are all [-inf] has no category: such a row takes
   [0.] first. *)
let logit =
  {
    bounds =
      (fun d -> [ Float.neg_infinity; -.greatest d; greatest d; 0.; -0. ]);
    value = Gen.float_range (-5.) 5.;
    top = greatest;
    signed = true;
  }

let bounds_count p = List.length (p.bounds (F Nx.float64))

(* A case's columns: element [i] of each, bound [i] where [i] is below the
   bounds' count, else drawn: a value in 6 of 8, a spread point in 1 and a bound
   in 1. Windtrap repeats an earlier draw with probability 1/8, and may land
   the choice on a corner of its range one time in ten, so a drawn element is
   a fresh value with probability at least 7/8 · 9/10 · 6/8 = 0.59: the 2040
   drawn elements of a block hold fewer than 800 values, the most a cover
   asks, with probability below 1e-30. Every value of a concentration or a
   rate lies where a round rejects, corners included. A truncated normal's
   upper bound is its lower one in 1 of 5. *)
let drawn p =
  Gen.frequency
    [
      (6, Gen.map (fun x -> Value x) p.value);
      (1, Gen.map (fun t -> Spread t) (Gen.float_range (-1.) 1.));
      (1, Gen.map (fun i -> Bound i) (Gen.int_range 0 (bounds_count p - 1)));
    ]

let columns ?(equal_in = 0) ps =
  let open Gen in
  let nb = List.fold_left (fun m p -> max m (bounds_count p)) 0 ps in
  let element =
    let alone =
      List.fold_right
        (fun p acc ->
          let+ v = drawn p and+ vs = acc in
          v :: vs)
        ps (constant [])
    in
    if equal_in = 0 then alone
    else
      frequency
        [
          (equal_in - 1, alone);
          (1, map (fun v -> List.map (fun _ -> v) ps) (drawn (List.hd ps)));
        ]
  in
  let+ rest = array ~size:(constant (n - nb)) element in
  List.mapi
    (fun j p ->
      Array.init n (fun i ->
          if i < nb then Bound (i mod bounds_count p)
          else List.nth rest.(i - nb) j))
    ps
  |> Array.of_list

type case = { seed : int; cols : v array array; ints : int64 array array }

let values c j =
  Array.to_list c.cols.(j)
  |> List.filter_map (function Value x -> Some x | _ -> None)

let count p xs = List.length (List.filter p xs)

let seeded gen =
  Gen.(
    let+ seed = int and+ cols = gen in
    { seed; cols; ints = [||] })

let ints_only = seeded (Gen.constant [||])

(* Runners *)

type ('w, 'r) f0 = { f0 : 'd. 'd Nx.Rng.key -> ('w, 'r, 'd) Nx.t }

type ('v, 's, 'w, 'r) f1 = {
  f1 : 'd. 'd Nx.Rng.key -> ('v, 's, 'd) Nx.t -> ('w, 'r, 'd) Nx.t;
}

type ('v, 's, 'd) t = ('v, 's, 'd) Nx.t

type ('v, 's, 'u, 't, 'w, 'r) f2 = {
  f2 : 'd. 'd Nx.Rng.key -> ('v, 's, 'd) t -> ('u, 't, 'd) t -> ('w, 'r, 'd) t;
}

type runner = {
  run0 : 'w 'r. ('w, 'r) f0 -> Nx.host Nx.Rng.key -> ('w, 'r, Nx.host) Nx.t;
  run1 :
    'v 's 'w 'r.
    ('v, 's, 'w, 'r) f1 ->
    Nx.host Nx.Rng.key ->
    ('v, 's, Nx.host) Nx.t ->
    ('w, 'r, Nx.host) Nx.t;
  run2 :
    'v 's 'u 't 'w 'r.
    ('v, 's, 'u, 't, 'w, 'r) f2 ->
    Nx.host Nx.Rng.key ->
    ('v, 's, Nx.host) Nx.t ->
    ('u, 't, Nx.host) Nx.t ->
    ('w, 'r, Nx.host) Nx.t;
}

let at whole cut =
  let on x = Nx.place (cut (Array.length (Nx.shape x))) x in
  let back y = Nx.place Nx.Host.on y and key k = Nx.Rng.place whole k in
  {
    run0 = (fun { f0 } k -> back (f0 (key k)));
    run1 = (fun { f1 } k x -> back (f1 (key k) (on x)));
    run2 = (fun { f2 } k x y -> back (f2 (key k) (on x) (on y)));
  }

let placed (module S : Nx.Devices) = at S.on (fun _ -> S.on)

(* Draws *)

type shape = Zero | One | Block | Row

(* Where a law draws: its float dtypes and shapes. *)
type at = { dtypes : dtype list; shapes : shape list }

(* An integer sampler draws its 32-bit dtype with [Float32], its 64-bit one
   with [Float64]. *)
let width at d = List.mem d at.dtypes

(* A flat sampler's sizes; [Row] is a row sampler's alone. *)
let flat at = List.filter (fun s -> s <> Row) at.shapes
let elements = function Zero -> 0 | One -> 1 | Block | Row -> n

let shape_name = function
  | Zero -> "a zero-size shape"
  | One -> "one element"
  | Block -> Printf.sprintf "%d elements" n
  | Row -> Printf.sprintf "one row of %d" n

type pair = { label : string; eager : int64 array; subject : int64 array }

(* A sampler's [dtypes] are those it draws at, an integer sampler's widths. *)
type sampler = {
  name : string;
  gen : case Gen.t;
  covers : (string * (case -> bool)) list;
  rows : bool;
  dtypes : dtype list;
  pairs : runner -> at -> case -> pair list;
}

let name s = s.name
let dtypes s = s.dtypes
let floats = [ Float16; Bfloat16; Float32; Float64 ]
let key c = Nx.Rng.key c.seed

let p0 r label f c =
  let k = key c in
  { label; eager = codes (f.f0 k); subject = codes (r.run0 f k) }

let p1 r label f c x =
  let k = key c in
  { label; eager = codes (f.f1 k x); subject = codes (r.run1 f k x) }

let p2 r label f c x y =
  let k = key c in
  { label; eager = codes (f.f2 k x y); subject = codes (r.run2 f k x y) }

let label (F dt) shape =
  Printf.sprintf "%s at %s" (Nx.Dtype.name dt) (shape_name shape)

let tensor dt xs = Nx.cast dt (Nx.create Nx.float64 [| Array.length xs |] xs)

(* Column [j]'s first [m] elements for parameter [p], as [dt] holds them. *)
let column p dt c j m =
  tensor dt (Array.init m (fun i -> resolve p (F dt) c.cols.(j).(i)))

(* Each float dtype at each flat shape. *)
let each at f =
  List.concat_map
    (fun d -> List.map (fun s -> f (fdt d) s) (flat at))
    at.dtypes

(* An integer sampler's draws at its dtype of the width of [d], at each flat
   shape. *)
let integer at d f = if width at d then List.map f (flat at) else []

(* Samplers of a shape alone. *)

type shaped = {
  at :
    's 'd.
    'd Nx.Placement.t ->
    ?key:'d Nx.Rng.key ->
    (float, 's) Nx.dtype ->
    int array ->
    (float, 's, 'd) Nx.t;
}

let shaped name g =
  let pairs r at c =
    each at (fun (F dt as d) shape ->
        let sh = [| elements shape |] in
        let f0 k = g.at (Nx.Rng.placement k) ~key:k dt sh in
        p0 r (label d shape) { f0 } c)
  in
  let gen = Gen.map (fun seed -> { seed; cols = [||]; ints = [||] }) Gen.int in
  { name; gen; covers = []; rows = false; dtypes = floats; pairs }

let uniform = shaped "uniform" { at = Nx.Rng.uniform_at }
let normal = shaped "normal" { at = Nx.Rng.normal_at }
let exponential = shaped "exponential" { at = Nx.Rng.exponential_at }
let gumbel = shaped "gumbel" { at = Nx.Rng.gumbel_at }

let bits =
  let pairs r at c =
    integer at Float32 (fun shape ->
        let sh = [| elements shape |] in
        let f0 k = Nx.Rng.bits_at (Nx.Rng.placement k) ~key:k sh in
        p0 r (shape_name shape) { f0 } c)
  in
  {
    name = "bits";
    gen = ints_only;
    covers = [];
    rows = false;
    dtypes = [ Float32 ];
    pairs;
  }

let permutation =
  let pairs r at c =
    integer at Float64 (fun shape ->
        let m = elements shape in
        let f0 k = Nx.Rng.permutation_at (Nx.Rng.placement k) ~key:k m in
        p0 r (shape_name shape) { f0 } c)
  in
  {
    name = "permutation";
    gen = ints_only;
    covers = [];
    rows = false;
    dtypes = [ Float64 ];
    pairs;
  }

(* A key split into one key at the zero-size shape, two at one element and a
   block's at the block, its words int32. *)
let split_batch =
  let pairs r at c =
    integer at Float32 (fun shape ->
        let m = match shape with Zero -> 1 | One -> 2 | Block | Row -> n in
        let f0 k = Nx.Rng.to_tensor (Nx.Rng.split_batch ~n:m k) in
        p0 r (Printf.sprintf "%d keys" m) { f0 } c)
  in
  {
    name = "split_batch";
    gen = ints_only;
    covers = [];
    rows = false;
    dtypes = [ Float32 ];
    pairs;
  }

(* permute's images of [0, n) at the block, of an index of each bound at one
   element, of none at the zero-size shape. *)
let permute =
  let gen =
    Gen.(
      let+ seed = int
      and+ i = array ~size:(constant (n - 2)) (int_range 0 (n - 1)) in
      let i =
        Array.append [| 0L; Int64.of_int (n - 1) |] (Array.map Int64.of_int i)
      in
      { seed; cols = [||]; ints = [| i |] })
  in
  let pairs r at c =
    integer at Float64 (fun shape ->
        let m = elements shape in
        let i = Nx.create Nx.int64 [| m |] (Array.sub c.ints.(0) 0 m) in
        let size = if shape = Block then n else 1 in
        let i = if shape = One then Nx.zeros Nx.int64 [| 1 |] else i in
        let f1 k i = Nx.Rng.permute ~key:k ~n:size i in
        p1 r (shape_name shape) { f1 } c i)
  in
  {
    name = "permute";
    gen;
    covers = [];
    rows = false;
    dtypes = [ Float64 ];
    pairs;
  }

(* A key folded with indices: int64's extremes, 0 and -1, then drawn. Its
   words are int32. *)
let fold_in_tensor =
  let gen =
    Gen.(
      let+ seed = int and+ i = array ~size:(constant (n - 4)) int64 in
      let i = Array.append [| Int64.min_int; Int64.max_int; 0L; -1L |] i in
      { seed; cols = [||]; ints = [| i |] })
  in
  let pairs r at c =
    integer at Float32 (fun shape ->
        let m = elements shape in
        let i = Nx.create Nx.int64 [| m |] (Array.sub c.ints.(0) 0 m) in
        let f1 k i = Nx.Rng.to_tensor (Nx.Rng.fold_in_tensor k i) in
        p1 r (shape_name shape) { f1 } c i)
  in
  {
    name = "fold_in_tensor";
    gen;
    covers = [];
    rows = false;
    dtypes = [ Float32 ];
    pairs;
  }

(* randint between [low] and [high] at int32 and int64, its pairs opening with
   the whole of int32, the least ranges at both ends and around 0. *)
let randint =
  let lo32 = Int64.of_int32 Int32.min_int
  and hi32 = Int64.of_int32 Int32.max_int in
  let fixed =
    [
      (lo32, hi32);
      (0L, 1L);
      (-1L, 0L);
      (lo32, Int64.succ lo32);
      (Int64.pred hi32, hi32);
    ]
  in
  let gen =
    Gen.(
      let pair =
        frequency
          [
            ( 3,
              let+ low = int_range (-1000) 1000
              and+ r = int_range 1 1_000_000 in
              (Int64.of_int low, Int64.of_int (low + r)) );
            ( 1,
              let+ low = int32_range Int32.min_int (Int32.pred Int32.max_int) in
              let low = Int64.of_int32 low in
              (low, hi32) );
          ]
      in
      let+ seed = int
      and+ drawn = array ~size:(constant (n - List.length fixed)) pair in
      let ps = Array.append (Array.of_list fixed) drawn in
      { seed; cols = [||]; ints = [| Array.map fst ps; Array.map snd ps |] })
  in
  let bounds (type v r) (dt : (v, r) Nx.dtype) (of_i64 : int64 -> v) c shape =
    let m = elements shape in
    let col j =
      Nx.create dt [| m |] (Array.map of_i64 (Array.sub c.ints.(j) 0 m))
    in
    (col 0, col 1)
  in
  let pairs r at c =
    let f2 k low high = Nx.Rng.randint ~key:k ~low high in
    integer at Float32 (fun shape ->
        let l, h = bounds Nx.int32 Int64.to_int32 c shape in
        p2 r ("int32 at " ^ shape_name shape) { f2 } c l h)
    @ integer at Float64 (fun shape ->
        let l, h = bounds Nx.int64 Fun.id c shape in
        p2 r ("int64 at " ^ shape_name shape) { f2 } c l h)
  in
  {
    name = "randint";
    gen;
    covers = [];
    rows = false;
    dtypes = [ Float32; Float64 ];
    pairs;
  }

(* Samplers of float parameters *)

let bernoulli =
  let pairs r at c =
    each at (fun (F dt as d) shape ->
        let p = column probability dt c 0 (elements shape) in
        let f1 k p = Nx.Rng.bernoulli ~key:k p in
        p1 r (label d shape) { f1 } c p)
  in
  {
    name = "bernoulli";
    gen = seeded (columns [ probability ]);
    covers = [ ("p at 0, 1 and the greatest float below 1", fun _ -> true) ];
    rows = false;
    dtypes = floats;
    pairs;
  }

let gamma =
  let pairs r at c =
    each at (fun (F dt as d) shape ->
        let a = column positive dt c 0 (elements shape) in
        let f1 k a = Nx.Rng.gamma ~key:k a in
        p1 r (label d shape) { f1 } c a)
  in
  {
    name = "gamma";
    gen = seeded (columns [ positive ]);
    covers =
      [
        ("a at the least and greatest floats", fun _ -> true);
        ( "a round rejects: 800 concentrations in (0, 0.5] or [1, 1.5]",
          fun c -> count conc_region (values c 0) >= 800 );
      ];
    rows = false;
    dtypes = floats;
    pairs;
  }

let beta =
  let pairs r at c =
    each at (fun (F dt as d) shape ->
        let m = elements shape in
        let a = column positive dt c 0 m and b = column positive dt c 1 m in
        let f2 k a b = Nx.Rng.beta ~key:k a b in
        p2 r (label d shape) { f2 } c a b)
  in
  {
    name = "beta";
    gen = seeded (columns [ positive; positive ]);
    covers =
      [
        ("a and b at the least and greatest floats", fun _ -> true);
        ( "a round rejects: 800 concentrations of a in (0, 0.5] or [1, 1.5]",
          fun c -> count conc_region (values c 0) >= 800 );
      ];
    rows = false;
    dtypes = floats;
    pairs;
  }

let von_mises =
  let pairs r at c =
    each at (fun (F dt as d) shape ->
        let k = column non_negative dt c 0 (elements shape) in
        let f1 key k = Nx.Rng.von_mises ~key k in
        p1 r (label d shape) { f1 } c k)
  in
  {
    name = "von_mises";
    gen = seeded (columns [ non_negative ]);
    covers =
      [
        ("k at 0, the switch at 1 and the greatest float", fun _ -> true);
        ( "a round rejects: 100 concentrations in [2, 1000]",
          fun c -> count kappa_region (values c 0) >= 100 );
      ];
    rows = false;
    dtypes = floats;
    pairs;
  }

let truncated_normal =
  let pairs r at c =
    each at (fun (F dt as d) shape ->
        let m = elements shape in
        let lo = column lower dt c 0 m and hi = column upper dt c 1 m in
        let f2 k lo hi = Nx.Rng.truncated_normal ~key:k lo hi in
        p2 r (label d shape) { f2 } c lo hi)
  in
  {
    name = "truncated_normal";
    gen = seeded (columns ~equal_in:5 [ lower; upper ]);
    covers =
      [
        ("infinite, reversed, equal and far bounds", fun _ -> true);
        ( "equal drawn bounds",
          fun c ->
            let lo = c.cols.(0) and hi = c.cols.(1) in
            let rec go i =
              i < n && ((i >= 8 && lo.(i) = hi.(i)) || go (i + 1))
            in
            go 0 );
      ];
    rows = false;
    dtypes = floats;
    pairs;
  }

let poisson =
  let pairs r at c =
    each at (fun (F dt as d) shape ->
        let x = column rate dt c 0 (elements shape) in
        let f1 k x = Nx.Rng.poisson ~key:k x in
        p1 r (label d shape) { f1 } c x)
  in
  {
    name = "poisson";
    gen = seeded (columns [ rate ]);
    covers =
      [
        ("rate at 0, both sides of 10 and below 2^31", fun _ -> true);
        ( "a round rejects: 200 rates in [10, 10^4]",
          fun c -> count rate_region (values c 0) >= 200 );
      ];
    rows = false;
    dtypes = floats;
    pairs;
  }

(* binomial's trials, int32: 0, 1 and the greatest, then drawn from 1 to 10^6 on
   a logarithmic scale, so that the mean of the rarer outcome against a drawn
   probability lies below the switch to rejection at 10 in about one element in
   six, most of those from 1 up, where inversion sums several terms, and past it
   in about two in three. *)
let binomial =
  let fixed = [ 0L; 1L; Int64.of_int32 Int32.max_int ] in
  let trials =
    Gen.map
      (fun l -> Int64.of_float (Float.exp l))
      (Gen.float_range 0. (Float.log 1e6))
  in
  let gen =
    Gen.(
      let+ seed = int
      and+ cols = columns [ probability ]
      and+ t = array ~size:(constant (n - 3)) trials in
      let t = Array.append (Array.of_list fixed) t in
      { seed; cols; ints = [| t |] })
  in
  (* The drawn elements whose mean of the rarer outcome is in [lo, hi). *)
  let means lo hi c =
    let ps = c.cols.(0) and ts = c.ints.(0) in
    let k = ref 0 in
    Array.iteri
      (fun i p ->
        match p with
        | Value p ->
            let m = Int64.to_float ts.(i) *. Float.min p (1. -. p) in
            if m >= lo && m < hi then incr k
        | Bound _ | Spread _ -> ())
      ps;
    !k
  in
  let pairs r at c =
    each at (fun (F dt as d) shape ->
        let m = elements shape in
        let p = column probability dt c 0 m in
        let t =
          Nx.create Nx.int32 [| m |]
            (Array.map Int64.to_int32 (Array.sub c.ints.(0) 0 m))
        in
        let f2 k t p = Nx.Rng.binomial ~key:k t p in
        p2 r (label d shape) { f2 } c t p)
  in
  {
    name = "binomial";
    gen;
    covers =
      [
        ("n at 0, 1 and the greatest int32; p at 0 and 1", fun _ -> true);
        ( "inversion sums terms: 100 drawn means of the rarer outcome from 1 \
           to 10",
          fun c -> means 1. 10. c >= 100 );
        ( "a round rejects: 200 drawn means of the rarer outcome at 10 or more",
          fun c -> means 10. Float.infinity c >= 200 );
      ];
    rows = false;
    dtypes = floats;
    pairs;
  }

(* Row samplers: rows of [row] elements in a block, one row of one element,
   none, and with [Row] the block as one row. *)
let row_shape = function
  | Zero -> [| 0; row |]
  | One -> [| 1; 1 |]
  | Block -> [| n / row; row |]
  | Row -> [| 1; n |]

let numel s = Array.fold_left ( * ) 1 s

let each_row at f =
  List.concat_map
    (fun d -> List.map (fun s -> f (fdt d) s) at.shapes)
    at.dtypes

let dirichlet =
  let pairs r at c =
    each_row at (fun (F dt as d) shape ->
        let s = row_shape shape in
        let a = Nx.reshape s (column positive dt c 0 (numel s)) in
        let f1 k a = Nx.Rng.dirichlet ~key:k a in
        p1 r (label d shape) { f1 } c a)
  in
  {
    name = "dirichlet";
    gen = seeded (columns [ positive ]);
    covers =
      [
        ("a at the least and greatest floats", fun _ -> true);
        ( "a round rejects: 800 concentrations in (0, 0.5] or [1, 1.5]",
          fun c -> count conc_region (values c 0) >= 800 );
      ];
    rows = true;
    dtypes = floats;
    pairs;
  }

(* A row whose largest logit is not finite takes [0.] at its first. *)
let finite_rows len xs =
  for r = 0 to (Array.length xs / len) - 1 do
    let ok = ref false in
    for j = 0 to len - 1 do
      if Float.is_finite xs.((r * len) + j) then ok := true
    done;
    if not !ok then xs.(r * len) <- 0.
  done;
  xs

let categorical =
  let pairs r at c =
    each_row at (fun (F dt as d) shape ->
        let s = row_shape shape in
        let xs = Array.init (numel s) (fun i -> resolve logit d c.cols.(0).(i)) in
        let l = Nx.reshape s (tensor dt (finite_rows s.(1) xs)) in
        let f1 k l = Nx.Rng.categorical ~key:k l in
        p1 r (label d shape) { f1 } c l)
  in
  {
    name = "categorical";
    gen = seeded (columns [ logit ]);
    covers = [ ("logits of -inf and the greatest floats", fun _ -> true) ];
    rows = true;
    dtypes = floats;
    pairs;
  }

let samplers =
  [
    bits;
    uniform;
    normal;
    exponential;
    gumbel;
    randint;
    bernoulli;
    gamma;
    beta;
    dirichlet;
    von_mises;
    truncated_normal;
    poisson;
    binomial;
    categorical;
    permutation;
    permute;
    split_batch;
    fold_in_tensor;
  ]

let pp_case name ppf c =
  Format.fprintf ppf "%s from Nx.Rng.key %d, %d parameter columns" name c.seed
    (Array.length c.cols + Array.length c.ints)

let gen s = Gen.with_pp (pp_case s.name) s.gen

(* Laws *)

(* The elements where the two draws differ, at most four, then their count. *)
let misses p =
  let ne = Array.length p.eager and nc = Array.length p.subject in
  if ne <> nc then
    [ Printf.sprintf "%s: %d elements on the host, %d here" p.label ne nc ]
  else
    let bad = ref [] and k = ref 0 in
    Array.iteri
      (fun i e ->
        let c = p.subject.(i) in
        if not (Int64.equal e c) then begin
          incr k;
          if !k <= 4 then
            bad :=
              Printf.sprintf "%s: element %d: host %#Lx, here %#Lx" p.label i e
                c
              :: !bad
        end)
      p.eager;
    List.rev !bad
    @ if !k > 4 then [ Printf.sprintf "%s: %d elements differ" p.label !k ]
      else []

let dtype_name d = match fdt d with F dt -> Nx.Dtype.name dt

let law r ~dtypes ~shapes s c =
  let at = { dtypes; shapes } in
  List.iter (fun d -> cover (dtype_name d) true) dtypes;
  List.iter
    (fun shape -> cover (shape_name shape) true)
    (if s.rows then shapes else flat at);
  List.iter (fun (label, f) -> cover label (f c)) s.covers;
  equal (list string) [] (List.concat_map misses (s.pairs r at c))
