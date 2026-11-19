(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module S = Alda.Sos

(* Single sections against long-double recursions with the rounding bound's
   weights, and cascades against scipy.signal.sosfilt, from golden/sos.npz
   (contrib/alda/test/gen/sos.py) *)
let golden = lazy (Nx_io.load_npz "golden/sos.npz")

let entry dt key =
  match Nx_io.Archive.find key (Lazy.force golden) with
  | Some p -> Nx.unpack dt p
  | None -> failf "golden/sos.npz has no %s" key

let floats a = Nx.to_array (Nx.cast Nx.float64 a)

(* Each element's bits, every NaN as one. *)
let bits a =
  Array.map
    (fun v -> if Float.is_nan v then 0x7ff8L else Int64.bits_of_float v)
    (floats a)

let equal_bits want got =
  equal (array int) (Nx.shape want) (Nx.shape got);
  equal (array int64) (bits want) (bits got)

(* The blocked evaluation's error stays within a margin of [u (|h| * m')ₙ], [m']
   the recursion's absolute terms with [|p| + |q|] in place of [|a1|]: measured
   within 1.23 of it on every section and input tried, so a margin of 2. [ratio]
   is the largest error over its weight. *)
let margin = 2.

let ratio ~u want got w =
  let r = ref 0. in
  Array.iteri
    (fun i y ->
      let e = Float.abs (got.(i) -. y) in
      if e > 0. then r := Float.max !r (e /. (u *. w.(i))))
    want;
  !r

let section (type s) name (dt : (float, s) Nx.dtype) u i =
  let key s = Printf.sprintf "sec.%d.%s.%s" i name s in
  let row = Nx.cast dt (entry Nx.float64 (Printf.sprintf "sec.%d.row" i)) in
  let x = entry dt (key "x") in
  let got = floats (Alda.sos (Nx.reshape [| 1; 6 |] row) x) in
  let r =
    ratio ~u
      (floats (entry Nx.float64 (key "y")))
      got
      (floats (entry Nx.float64 (key "w")))
  in
  less float_exact ~than:margin r

let sections32 =
  cases "float32 sections within the blocked bound" ~id:string_of_int
    (List.init 8 Fun.id)
    (section "f32" Nx.float32 (Float.ldexp 1. (-24)))

let sections64 =
  cases "float64 sections within the blocked bound" ~id:string_of_int
    [ 0; 2; 4; 6 ]
    (section "f64" Nx.float64 (Float.ldexp 1. (-53)))

(* Cascades of up to six sections in float64: scipy's sequential recursion and
   the blocked one each round within about 1e-14 of the peak here. *)
let cascades =
  cases "float64 cascades are scipy's sosfilt" ~id:string_of_int
    (List.init 4 Fun.id) (fun j ->
      let key s = Printf.sprintf "cas.%d.%s" j s in
      let want = floats (entry Nx.float64 (key "y")) in
      let got =
        floats
          (Alda.sos (entry Nx.float64 (key "sos")) (entry Nx.float64 (key "x")))
      in
      let peak =
        Array.fold_left (fun m v -> Float.max m (Float.abs v)) 0. want
      in
      equal (array (float (1e-12 *. peak))) want got)

(* Random stable sections from the cookbook's peaking and low-pass forms. *)
let cookbook kind ~f ~q ~g =
  let a = 10. ** (g /. 40.) and w = 2. *. Float.pi *. f /. 48_000. in
  let al = Float.sin w /. (2. *. q) and c = Float.cos w in
  match kind with
  | `Peak ->
      [|
        1. +. (al *. a);
        -2. *. c;
        1. -. (al *. a);
        1. +. (al /. a);
        -2. *. c;
        1. -. (al /. a);
      |]
  | `Low ->
      [|
        (1. -. c) /. 2.; 1. -. c; (1. -. c) /. 2.; 1. +. al; -2. *. c; 1. -. al;
      |]

type case = {
  rows : float array list;
  n : int;
  chunks : int list;
  steady : bool;
  channels : bool;
  seed : int;
}

let pp_case ppf c =
  Format.fprintf ppf "{%d sections; n %d; chunks [%s]; %s; %s; seed %d}"
    (List.length c.rows) c.n
    (String.concat "; " (List.map string_of_int c.chunks))
    (if c.steady then "from steady" else "from rest")
    (if c.channels then "[n; 2] along 0" else "[n]")
    c.seed

let row =
  let open Gen in
  let* kind = of_list [ `Peak; `Low ] in
  let* f = float_range 20. 20_000. and+ q = float_range 0.3 8. in
  let+ g = float_range (-12.) 12. in
  let r = cookbook kind ~f ~q ~g in
  (* a0 other than 1, as the cookbook writes it, scaled *)
  Array.map (fun v -> v *. 1.5) r

(* Chunks at multiples of 4096, empty ones included, then any last one. *)
let case =
  let open Gen in
  with_pp pp_case
  @@
  let* k = int_range 1 3 in
  let* rows = list ~size:(constant k) row in
  let* blocks = list ~size:(int_range 0 4) (int_range 0 2) in
  let* last = frequency [ (1, int_range 0 2); (3, int_range 0 4095) ] in
  let* steady = bool and+ channels = bool in
  let+ seed = int_range 0 1_000_000 in
  let chunks = List.map (fun b -> b * 4096) blocks @ [ last ] in
  { rows; n = List.fold_left ( + ) 0 chunks; chunks; steady; channels; seed }

let cascade dt c =
  Nx.cast dt
    (Nx.create Nx.float64 [| List.length c.rows; 6 |] (Array.concat c.rows))

let signal dt c =
  let r = Random.State.make [| c.seed |] in
  let shape = if c.channels then [| c.n; 2 |] else [| c.n |] in
  Nx.cast dt (Nx.init Nx.float64 shape (fun _ -> Random.State.float r 2. -. 1.))

let axis c = if c.channels then 0 else -1

let piece c x a b =
  if c.channels then Nx.slice [ Nx.R (a, b); Nx.A ] x
  else Nx.slice [ Nx.R (a, b) ] x

let first c x =
  if c.channels then Nx.slice [ Nx.I 0; Nx.A ] x else Nx.slice [ Nx.I 0 ] x

(* The state the stream starts from, and the outputs of each chunk. *)
let streamed dt c x =
  let cs = cascade dt c in
  let s0 = if c.steady && c.n > 0 then S.steady cs (first c x) else S.start in
  let _, outs, _ =
    List.fold_left
      (fun (s, outs, at) k ->
        let s, y = S.feed ~axis:(axis c) cs s (piece c x at (at + k)) in
        (s, y :: outs, at + k))
      (s0, [], 0) c.chunks
  in
  (s0, Nx.concatenate ~axis:(axis c) (List.rev outs))

let chunk_law (type s) (dt : (float, s) Nx.dtype) name =
  prop
    (Printf.sprintf "chunks at multiples of 4096 give one call's bits (%s)" name)
    ~count:30 case (fun c ->
      cover ~share:0.2 "more than a block" (c.n > 4096);
      cover ~share:0.1 "an empty chunk" (List.mem 0 c.chunks);
      cover ~share:0.1 "a last block under three samples"
        (let l = List.nth c.chunks (List.length c.chunks - 1) in
         l > 0 && l < 3);
      let x = signal dt c in
      let s0, got = streamed dt c x in
      let _, whole = S.feed ~axis:(axis c) (cascade dt c) s0 x in
      equal_bits whole got)

(* Other splits round differently, within the evaluation's rounding: here 1e-9
   of the peak at float64. *)
let other_splits =
  prop "other splits agree within rounding" ~count:20 case (fun c ->
      let r = Random.State.make [| c.seed; 1 |] in
      let rec split left =
        if left = 0 then [ 0 ]
        else
          let k = Int.min left (1 + Random.State.int r 700) in
          k :: split (left - k)
      in
      let c = { c with chunks = split c.n } in
      cover ~share:0.3 "chunks off the blocks" (List.length c.chunks > 1);
      let x = signal Nx.float64 c in
      let s0, got = streamed Nx.float64 c x in
      let _, whole = S.feed ~axis:(axis c) (cascade Nx.float64 c) s0 x in
      let whole = floats whole in
      let peak =
        Array.fold_left (fun m v -> Float.max m (Float.abs v)) 1. whole
      in
      equal (array (float (1e-9 *. peak))) whole (floats got))

(* From the steady state, a held input gives the held output. *)
let steady =
  test "a held input from steady stays steady" (fun () ->
      let c =
        Nx.create Nx.float64 [| 2; 6 |]
          (Array.append
             (cookbook `Low ~f:200. ~q:0.7 ~g:0.)
             (cookbook `Peak ~f:1000. ~q:2. ~g:6.))
      in
      let u = Nx.create Nx.float64 [| 3 |] [| 1.; -2.; 0.5 |] in
      let x = Nx.broadcast_to [| 3; 5000 |] (Nx.reshape [| 3; 1 |] u) in
      let _, y = S.feed c (S.steady c u) x in
      let g s =
        let r = Array.sub (Nx.to_array c) (6 * s) 6 in
        (r.(0) +. r.(1) +. r.(2)) /. (r.(3) +. r.(4) +. r.(5))
      in
      let want = Array.map (fun u -> u *. g 0 *. g 1) (Nx.to_array u) in
      let got = Nx.to_array (Nx.slice [ Nx.A; Nx.I 4999 ] y) in
      equal (array (float 1e-12)) want got;
      let spread =
        Nx.item []
          (Nx.max
             (Nx.abs
                (Nx.sub y (Nx.reshape [| 3; 1 |] (Nx.slice [ Nx.A; Nx.I 0 ] y)))))
      in
      less float_exact ~than:1e-12 spread)

(* Stability by the poles' moduli, one row each: poles inside, on and outside
   the unit circle, with a0 of either sign. *)
let stable =
  test "stable is the poles inside the unit circle" (fun () ->
      let rows =
        [|
          (* poles 0.5, 0.5 *) 1.;
          0.;
          0.;
          1.;
          -1.;
          0.25;
          (* same, a0 = -2 *) 1.;
          0.;
          0.;
          -2.;
          2.;
          -0.5;
          (* poles 1, 0.5 *) 1.;
          0.;
          0.;
          1.;
          -1.5;
          0.5;
          (* poles ±1.1i *) 1.;
          0.;
          0.;
          1.;
          0.;
          1.21;
          (* poles -0.9, 0.2 *) 1.;
          0.;
          0.;
          1.;
          0.7;
          -0.18;
        |]
      in
      let c = Nx.create Nx.float64 [| 5; 6 |] rows in
      equal (array bool)
        [| true; true; false; false; true |]
        (Nx.to_array (S.stable c)))

(* Gradients *)

(* [loss c x] is a fixed weighting of [sos c x] summed, so that every output
   counts. *)
let loss w c x = Nx.sum (Nx.mul w (Alda.sos c x))

(* Central differences of [f] at [p] along the coordinates [picks] of [p], every
   one by default. A step of 1e-7 relative: near a double pole the truncation
   error of a step of 1e-6 is already 1e-6 of the derivative. *)
let numeric ?picks f p =
  let a = Nx.to_array p in
  let picks = Option.value picks ~default:(List.init (Array.length a) Fun.id) in
  let h i = 1e-7 *. Float.max 1. (Float.abs a.(i)) in
  Array.of_list
    (List.map
       (fun i ->
         let at d =
           let b = Array.copy a in
           b.(i) <- a.(i) +. d;
           Nx.item [] (f (Nx.create Nx.float64 (Nx.shape p) b))
         in
         (at (h i) -. at (-.h i)) /. (2. *. h i))
       picks)

let close want got =
  let top = Array.fold_left (fun m v -> Float.max m (Float.abs v)) 0. want in
  equal (array (float (1e-6 *. top))) want got

let rows_of l =
  Nx.create Nx.float64 [| List.length l / 6; 6 |] (Array.of_list l)

(* RBJ's peaking section with [Q A = 1/2] has a double pole; so does [z² - 1.875
   z + 0.87890625] exactly, at 15/16. *)
let peaking_double =
  let g = -5.84 in
  let a = 10. ** (g /. 40.) in
  Array.to_list (cookbook `Peak ~f:1000. ~q:(0.5 /. a) ~g)

let gradient_cases =
  [
    ( "two sections",
      peaking_double @ Array.to_list (cookbook `Low ~f:3000. ~q:0.9 ~g:0.) );
    ("an exact double pole", [ 0.25; 0.5; 0.25; 1.; -1.875; 0.87890625 ]);
    ("peaking with Q A = 1/2", peaking_double);
  ]

let gradients =
  cases "reverse and forward derivatives are central differences" ~id:fst
    gradient_cases (fun (_, l) ->
      let c = rows_of l in
      let n = 4096 + 300 in
      let r = Random.State.make [| 7 |] in
      let x =
        Nx.init Nx.float64 [| n |] (fun _ -> Random.State.float r 2. -. 1.)
      in
      let w =
        Nx.init Nx.float64 [| n |] (fun _ -> Random.State.float r 2. -. 1.)
      in
      let gc, gx =
        Rune.grad Nx.Ptree.(pair tensor tensor) (fun (c, x) -> loss w c x) (c, x)
      in
      close (numeric (fun c -> loss w c x) c) (Nx.to_array gc);
      (* a few samples of x, around the block boundary *)
      let picks = [ 0; 1; 4095; 4096; 4097; n - 1 ] in
      let gx = Nx.to_array gx in
      close
        (numeric ~picks (fun x -> loss w c x) x)
        (Array.of_list (List.map (fun i -> gx.(i)) picks));
      (* forward mode along a random direction of c *)
      let dc =
        Nx.init Nx.float64 (Nx.shape c) (fun _ -> Random.State.float r 2. -. 1.)
      in
      let _, d = Rune.jvp' (fun c -> loss w c x) c dc in
      let fc = numeric (fun c -> loss w c x) c in
      let want =
        Array.fold_left ( +. ) 0.
          (Array.mapi (fun i v -> v *. fc.(i)) (Nx.to_array dc))
      in
      equal (float (1e-6 *. Float.abs want)) want (Nx.item [] d))

(* Through a stream: the gradient of a loss of two feeds with respect to the
   coefficients and the starting state. *)
let stream_gradient =
  test "derivatives flow through the state" (fun () ->
      let c = rows_of (Array.to_list (cookbook `Peak ~f:500. ~q:3. ~g:9.)) in
      let r = Random.State.make [| 8 |] in
      let x =
        Nx.init Nx.float64 [| 2; 300 |] (fun _ -> Random.State.float r 2. -. 1.)
      in
      let u = Nx.create Nx.float64 [| 2 |] [| 0.3; -0.7 |] in
      let f (c, u) =
        let s = S.steady c u in
        let s, y1 = S.feed c s (Nx.slice [ Nx.A; Nx.R (0, 200) ] x) in
        let _, y2 = S.feed c s (Nx.slice [ Nx.A; Nx.R (200, 300) ] x) in
        Nx.add (Nx.sum (Nx.square y1)) (Nx.sum (Nx.square y2))
      in
      let gc, gu = Rune.grad Nx.Ptree.(pair tensor tensor) f (c, u) in
      close (numeric (fun c -> f (c, u)) c) (Nx.to_array gc);
      close (numeric (fun u -> f (c, u)) u) (Nx.to_array gu))

(* [df1 rows x] is the cascade's recursion one sample after another at float64,
   sample [i] with the rows [rows i], each divided by its [a0]. *)
let df1 rows x =
  let k = Array.length (rows 0) / 6 in
  let ins = Array.make_matrix k 2 0. and outs = Array.make_matrix k 2 0. in
  Array.mapi
    (fun i xn ->
      let r = rows i in
      let u = ref xn in
      for s = 0 to k - 1 do
        let c j = r.((6 * s) + j) /. r.((6 * s) + 3) in
        let y =
          (c 0 *. !u)
          +. (c 1 *. ins.(s).(0))
          +. (c 2 *. ins.(s).(1))
          -. (c 4 *. outs.(s).(0))
          -. (c 5 *. outs.(s).(1))
        in
        ins.(s) <- [| !u; ins.(s).(0) |];
        outs.(s) <- [| y; outs.(s).(0) |];
        u := y
      done;
      !u)
    x

let rel_l2 want got =
  let e = ref 0. and t = ref 0. in
  Array.iteri
    (fun i w ->
      e := !e +. ((got.(i) -. w) ** 2.);
      t := !t +. (w *. w))
    want;
  Float.sqrt (!e /. !t)

(* An 8th-order Butterworth low pass at 100 Hz and 48 kHz (scipy's [butter(8,
   100, fs=48000, output="sos")]), its rows rounded to float32: its poles lie
   within 1.4e-2 of z = 1, close to each other, where [a1² - 4 a2] cancels.
   Computed with one rounding, the float32 evaluation is within 2.4e-6 of
   float64's recursion on the same rows; computed as a product and a difference,
   2.5e-4. *)
let fused =
  test "the discriminant rounds once" (fun () ->
      let rows =
        [|
          3.256481658001225e-18;
          6.51296331600245e-18;
          3.256481658001225e-18;
          1.0;
          -1.9744801162001697;
          0.9746492891952604;
          1.0;
          2.0;
          1.0;
          1.0;
          -1.9782976529235727;
          0.9784671530043181;
          1.0;
          2.0;
          1.0;
          1.0;
          -1.9853905187431644;
          0.9855606265389961;
          1.0;
          2.0;
          1.0;
          1.0;
          -1.9947347938295836;
          0.9949057022407218;
        |]
      in
      let c = Nx.cast Nx.float32 (Nx.create Nx.float64 [| 4; 6 |] rows) in
      let r = Random.State.make [| 5 |] in
      let x =
        Nx.init Nx.float32
          [| 1 lsl 16 |]
          (fun _ -> Random.State.float r 2. -. 1.)
      in
      let rows32 = floats c in
      let want = df1 (fun _ -> rows32) (floats x) in
      less float_exact ~than:2e-5 (rel_l2 want (floats (Alda.sos c x))))

(* Rows may change between feeds: the next output is the recursion with the new
   rows on the samples the state holds. *)
let changing_rows =
  test "rows that change between feeds" (fun () ->
      let c1 =
        Array.append
          (cookbook `Peak ~f:300. ~q:2. ~g:6.)
          (cookbook `Low ~f:5000. ~q:0.8 ~g:0.)
      in
      let c2 =
        Array.append
          (cookbook `Peak ~f:900. ~q:1. ~g:(-4.))
          (cookbook `Low ~f:3000. ~q:0.6 ~g:0.)
      in
      let r = Random.State.make [| 6 |] in
      let x = Array.init 5000 (fun _ -> Random.State.float r 2. -. 1.) in
      let cut = 1234 in
      let nx a = Nx.create Nx.float64 [| Array.length a |] a in
      let rows a = Nx.create Nx.float64 [| 2; 6 |] a in
      let s, y1 = S.feed (rows c1) S.start (nx (Array.sub x 0 cut)) in
      let _, y2 = S.feed (rows c2) s (nx (Array.sub x cut (5000 - cut))) in
      let want = df1 (fun i -> if i < cut then c1 else c2) x in
      let got = Array.append (floats y1) (floats y2) in
      let peak =
        Array.fold_left (fun m v -> Float.max m (Float.abs v)) 0. want
      in
      equal (array (float (1e-12 *. peak))) want got)

(* Under Rune.jit the cascade computes within rune's numerics: here within 1e-12
   of the eager result's peak, over three blocks. *)
let jit =
  test "under Rune.jit" (fun () ->
      let c =
        rows_of
          (Array.to_list (cookbook `Peak ~f:300. ~q:2. ~g:6.)
          @ Array.to_list (cookbook `Low ~f:5000. ~q:0.8 ~g:0.))
      in
      let r = Random.State.make [| 7 |] in
      let x =
        Nx.init Nx.float64
          [| 2; (2 * 4096) + 500 |]
          (fun _ -> Random.State.float r 2. -. 1.)
      in
      let eager = floats (Alda.sos c x) in
      let jitted = floats (Rune.jit' (fun x -> Alda.sos c x) x) in
      let peak =
        Array.fold_left (fun m v -> Float.max m (Float.abs v)) 0. eager
      in
      equal (array (float (1e-12 *. peak))) eager jitted)

let errors =
  group "misuse"
    [
      test "a cascade of five columns" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Alda.sos") (fun () ->
              Alda.sos
                (Nx.zeros Nx.float64 [| 1; 5 |])
                (Nx.zeros Nx.float64 [| 4 |])));
      test "a cascade of no section" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Alda.sos") (fun () ->
              Alda.sos
                (Nx.zeros Nx.float64 [| 0; 6 |])
                (Nx.zeros Nx.float64 [| 4 |])));
      test "a state of another lead" (fun () ->
          let c = rows_of [ 1.; 0.; 0.; 1.; 0.; 0. ] in
          let s, _ = S.feed c S.start (Nx.zeros Nx.float64 [| 2; 4 |]) in
          raises_match (Exn.invalid_arg ~sub:"Alda.Sos.feed") (fun () ->
              S.feed c s (Nx.zeros Nx.float64 [| 3; 4 |])));
    ]

let () =
  exit
    (run "alda.sos"
       [
         sections32;
         sections64;
         cascades;
         chunk_law Nx.float32 "float32";
         chunk_law Nx.float64 "float64";
         other_splits;
         steady;
         stable;
         gradients;
         stream_gradient;
         fused;
         changing_rows;
         jit;
         errors;
       ])
