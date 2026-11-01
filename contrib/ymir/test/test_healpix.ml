(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* HEALPix: pixels, their centres, corners and neighbours in both schemes as
   healpy gives them, and spherical harmonic transforms as healpy computes
   them, within the bounds the interface states. *)

open Windtrap
open Ymir
module Reference = Ymir_test.Healpix_reference

let ring nside = Healpix.v Frame.icrs ~nside Healpix.Ring
let nested nside = Healpix.v Frame.icrs ~nside Healpix.Nested
let int64s xs = Nx.create Nx.int64 [| Array.length xs |] xs
let vectors vs = Nx.create Nx.float64 [| Array.length vs / 3; 3 |] vs
let icrs v = Direction.of_xyz Frame.icrs v

(* [raw v] is the rows of [v] as a direction, unnormalised. *)
let raw v =
  Nx.Ptree.map (Direction.ptree ())
    (fun _ x -> Nx.cast (Nx.dtype x) v)
    (icrs (Nx.create Nx.float64 [| 3 |] [| 1.; 0.; 0. |]))

let flat3 l = Array.concat (List.map (fun (x, y, z) -> [| x; y; z |]) l)
let pixel h vs = Nx.to_array (Healpix.pixel h (raw (vectors vs)))
let max_order = 29

(* The angle between two unit vectors, within 1e-16 at small angles. *)
let apart (ax, ay, az) (bx, by, bz) =
  let cx = (ay *. bz) -. (az *. by)
  and cy = (az *. bx) -. (ax *. bz)
  and cz = (ax *. by) -. (ay *. bx) in
  Float.atan2
    (Float.sqrt ((cx *. cx) +. (cy *. cy) +. (cz *. cz)))
    ((ax *. bx) +. (ay *. by) +. (az *. bz))

let triple_at a i = (a.(3 * i), a.((3 * i) + 1), a.((3 * i) + 2))

(* The bound on centres and corners the interface states. *)
let centre_tol = 2e-15

(* healpy's pixels *)

let nsides =
  List.sort_uniq compare (List.map (fun (n, _, _, _) -> n) Reference.pixels)

let healpy_pixels =
  group "healpy's pixels"
    [
      cases "each direction gets one of its pixels" ~id:string_of_int nsides
        (fun nside ->
          let rows =
            List.filter (fun (n, _, _, _) -> n = nside) Reference.pixels
          in
          let vs = flat3 (List.map (fun (_, v, _, _) -> v) rows) in
          let check h pick =
            List.iteri
              (fun i got ->
                let row = List.nth rows i in
                match pick row with
                | [ p ] -> equal int64 p got
                | ps -> mem int64 got ps)
              (Array.to_list (pixel h vs))
          in
          check (ring nside) (fun (_, _, r, _) -> r);
          check (nested nside) (fun (_, _, _, n) -> n));
      test "the two schemes name one pixel of each direction" (fun () ->
          List.iter
            (fun nside ->
              let rows =
                List.filter (fun (n, _, _, _) -> n = nside) Reference.pixels
              in
              let v =
                raw (vectors (flat3 (List.map (fun (_, v, _, _) -> v) rows)))
              in
              let r = Healpix.pixel (ring nside) v
              and n = Healpix.pixel (nested nside) v in
              equal (array int64) (Nx.to_array n)
                (Nx.to_array (Healpix.renumber (ring nside) Healpix.Nested r)))
            nsides);
    ]

(* healpy's cells *)

let cell_nsides =
  List.sort_uniq compare
    (List.map (fun (n, _, _, _, _, _, _) -> n) Reference.cells)

let rows_at nside =
  List.filter (fun (n, _, _, _, _, _, _) -> n = nside) Reference.cells

let ring_ids rows =
  int64s (Array.of_list (List.map (fun (_, r, _, _, _, _, _) -> r) rows))

let nested_ids rows =
  int64s (Array.of_list (List.map (fun (_, _, n, _, _, _, _) -> n) rows))

let by_nside name f =
  cases name ~id:string_of_int cell_nsides (fun n -> f n (rows_at n))

let healpy_cells =
  group "healpy's cells"
    [
      by_nside "renumber maps RING to NESTED and back" (fun nside rows ->
          let r = ring_ids rows and n = nested_ids rows in
          equal (array int64) (Nx.to_array n)
            (Nx.to_array (Healpix.renumber (ring nside) Healpix.Nested r));
          equal (array int64) (Nx.to_array r)
            (Nx.to_array (Healpix.renumber (nested nside) Healpix.Ring n)));
      by_nside "centre is pix2vec's within 2e-15 rad" (fun nside rows ->
          let check h ids =
            let c = Nx.to_array (Direction.xyz (Healpix.centre h ids)) in
            List.iteri
              (fun i (_, _, _, v, _, _, _) ->
                at_most float_exact ~than:centre_tol (apart v (triple_at c i)))
              rows
          in
          check (ring nside) (ring_ids rows);
          check (nested nside) (nested_ids rows));
      by_nside "corners are boundaries' within 2e-15 rad" (fun nside rows ->
          let check h ids =
            let c = Nx.to_array (Direction.xyz (Healpix.corners h ids)) in
            List.iteri
              (fun i (_, _, _, _, corners, _, _) ->
                Array.iteri
                  (fun k v ->
                    at_most float_exact ~than:centre_tol
                      (apart v (triple_at c ((4 * i) + k))))
                  corners)
              rows
          in
          check (ring nside) (ring_ids rows);
          check (nested nside) (nested_ids rows));
      by_nside "neighbours are get_all_neighbours'" (fun nside rows ->
          let check h ids pick =
            let got = Nx.to_array (Healpix.neighbours h ids) in
            let want =
              Array.concat (List.map (fun r -> Array.of_list (pick r)) rows)
            in
            equal (array int64) want got
          in
          check (ring nside) (ring_ids rows) (fun (_, _, _, _, _, r, _) -> r);
          check (nested nside) (nested_ids rows) (fun (_, _, _, _, _, _, n) ->
              n));
    ]

(* Laws *)

(* A pixelisation and pixels of it: every order is drawn often enough that
   each run reaches the first and the last. *)
let drawn_pixels =
  Gen.bind (Gen.int_range 0 max_order) (fun order ->
      let nside = 1 lsl order in
      let npix = Int64.of_int (12 * nside * nside) in
      let edge = Gen.of_list [ 0L; 1L; Int64.pred npix; Int64.sub npix 2L ] in
      let any = Gen.int64_range 0L (Int64.pred npix) in
      Gen.map
        (fun ps -> (nside, Array.of_list ps))
        (Gen.list ~size:(Gen.int_range 1 16)
           (Gen.frequency [ (1, edge); (3, any) ])))

let covers nside =
  cover "nside 1" (nside = 1);
  cover "nside 2^29" (nside = 1 lsl max_order)

let both nside = [ ring nside; nested nside ]

let laws =
  group "laws"
    [
      prop ~count:300 "renumber to the other scheme and back is the identity"
        drawn_pixels (fun (nside, ps) ->
          covers nside;
          let p = int64s ps in
          let there = Healpix.renumber (ring nside) Healpix.Nested p in
          equal (array int64) ps
            (Nx.to_array (Healpix.renumber (nested nside) Healpix.Ring there)));
      prop ~count:300 "a pixel's centre lies in it" drawn_pixels
        (fun (nside, ps) ->
          covers nside;
          List.iter
            (fun h ->
              let p = int64s ps in
              equal (array int64) ps
                (Nx.to_array (Healpix.pixel h (Healpix.centre h p))))
            (both nside));
      prop ~count:300 "each pixel is a neighbour of each of its neighbours"
        drawn_pixels (fun (nside, ps) ->
          covers nside;
          List.iter
            (fun h ->
              let nb = Nx.to_array (Healpix.neighbours h (int64s ps)) in
              let present =
                List.filter
                  (fun t -> nb.(t) >= 0L)
                  (List.init (Array.length nb) Fun.id)
              in
              let qs = Array.of_list (List.map (fun t -> nb.(t)) present) in
              let back = Nx.to_array (Healpix.neighbours h (int64s qs)) in
              List.iteri
                (fun k t ->
                  mem int64
                    ps.(t / 8)
                    (Array.to_list (Array.sub back (8 * k) 8)))
                present)
            (both nside));
      prop ~count:300 "a nested pixel / 4 holds it at half the nside"
        (Gen.such_that (fun (n, _) -> n > 1) drawn_pixels)
        (fun (nside, ps) ->
          cover "nside 2^29" (nside = 1 lsl max_order);
          let c = Healpix.centre (nested nside) (int64s ps) in
          equal (array int64)
            (Array.map (fun p -> Int64.div p 4L) ps)
            (Nx.to_array (Healpix.pixel (nested (nside / 2)) c)));
      prop "pixel reads the ray: a power of two times a row"
        (Gen.triple
           (Gen.int_range 0 max_order)
           (Gen.list ~size:(Gen.int_range 1 8)
              (Gen.triple (Gen.float_range (-1.) 1.) (Gen.float_range (-1.) 1.)
                 (Gen.float_range (-1.) 1.)))
           (Gen.of_list [ 0x1p-1000; 0x1p-60; 4.; 0x1p900 ]))
        (fun (order, vs, k) ->
          cover "subnormal-range scale" (k = 0x1p-1000);
          (* A multiple that rounds no component, as the ray's contract
             states it. *)
          let exact c = c *. k /. k = c in
          let vs =
            List.filter
              (fun (x, y, z) ->
                (x <> 0. || y <> 0. || z <> 0.) && exact x && exact y && exact z)
              vs
          in
          let vs = flat3 vs in
          let h = ring (1 lsl order) in
          equal (array int64) (pixel h vs)
            (pixel h (Array.map (fun x -> x *. k) vs)));
      test "a NaN or zero row names no pixel" (fun () ->
          equal (array int64) [| -1L; -1L; 0L |]
            (pixel (ring 4) [| Float.nan; 0.; 1.; 0.; 0.; 0.; 0.; 0.; 1. |]));
      test "an infinite row raises" (fun () ->
          raises
            (Invalid_argument "Healpix.pixel: the vector at [0] is infinite")
            (fun () -> pixel (ring 4) [| Float.infinity; 0.; 0. |]));
      test "an index outside [-1, 12 nside^2) raises" (fun () ->
          raises
            (Invalid_argument
               "Healpix.centre: the pixel at [1] is outside [-1, 48)")
            (fun () -> Healpix.centre (ring 2) (int64s [| 0L; 48L |]));
          raises
            (Invalid_argument
               "Healpix.neighbours: the pixel at [0] is outside [-1, 48)")
            (fun () -> Healpix.neighbours (nested 2) (int64s [| -2L |])));
      test "-1 names no pixel in and out" (fun () ->
          let none = int64s [| -1L |] in
          List.iter
            (fun h ->
              equal (array int64) [| -1L |]
                (Nx.to_array (Healpix.renumber h Healpix.Nested none));
              equal (array int64) [| -1L |]
                (Nx.to_array (Healpix.renumber h Healpix.Ring none));
              equal (array int64) (Array.make 8 (-1L))
                (Nx.to_array (Healpix.neighbours h none));
              let nan = Nx.to_array (Direction.xyz (Healpix.centre h none)) in
              equal int 3
                (Array.length
                   (Array.of_list
                      (List.filter Float.is_nan (Array.to_list nan))));
              let c = Nx.to_array (Direction.xyz (Healpix.corners h none)) in
              equal int 12
                (List.length (List.filter Float.is_nan (Array.to_list c))))
            (both 4));
      test "the centre of a NaN row's pixel is a NaN row" (fun () ->
          let p =
            Healpix.pixel (ring 4) (raw (vectors [| Float.nan; 0.; 1. |]))
          in
          let c = Nx.to_array (Direction.xyz (Healpix.centre (ring 4) p)) in
          equal int 3 (List.length (List.filter Float.is_nan (Array.to_list c))));
      cases "the centre of each neighbour of every pixel is a direction or NaN"
        ~id:(fun (n, s) -> Printf.sprintf "nside %d, %s" n s)
        [
          (1, "ring");
          (2, "ring");
          (4, "ring");
          (1, "nested");
          (2, "nested");
          (4, "nested");
        ]
        (fun (nside, scheme) ->
          let h = if scheme = "ring" then ring nside else nested nside in
          let all = Nx.arange Nx.int64 0 (Healpix.pixels h) 1 in
          let nb = Healpix.neighbours h all in
          let c = Nx.to_array (Direction.xyz (Healpix.centre h nb)) in
          Array.iteri
            (fun i q ->
              let x, y, z = triple_at c i in
              if q < 0L then
                equal (triple bool bool bool) (true, true, true)
                  (Float.is_nan x, Float.is_nan y, Float.is_nan z)
              else
                equal (float 1e-15) 1.
                  (Float.sqrt ((x *. x) +. (y *. y) +. (z *. z))))
            (Nx.to_array nb));
      test "nside must be a power of two to 2^29" (fun () ->
          raises
            (Invalid_argument
               "Healpix.v: nside 3 is not a power of two from 1 to 2^29")
            (fun () -> ring 3);
          raises
            (Invalid_argument
               "Healpix.v: nside 1073741824 is not a power of two from 1 to \
                2^29") (fun () -> ring (1 lsl 30)));
      test "area is pi / (3 nside^2) steradians" (fun () ->
          equal float_exact
            (Float.pi /. (3. *. 1024. *. 1024.))
            (Nx.item []
               (Quantity.value Unit.steradian (Healpix.area (ring 1024)))));
      test "eager and compiled pixels and neighbours agree bit for bit"
        (fun () ->
          let h = nested 1024 in
          let v =
            Nx.reshape [| 100; 3 |]
              (Nx.sin (Nx.mul_s (Nx.arange_f Nx.float64 0. 300. 1.) 1.7))
          in
          let f v = Healpix.pixel h (raw v) in
          equal (array int64) (Nx.to_array (f v)) (Nx.to_array (Rune.jit' f v));
          let p = f v in
          let n p = Healpix.neighbours h p in
          equal (array int64) (Nx.to_array (n p)) (Nx.to_array (Rune.jit' n p)));
      by_nside "compiled centres are pix2vec's within 2e-15 rad"
        (fun nside rows ->
          let g h p = Direction.xyz (Healpix.centre h p) in
          let check h ids =
            let c = Nx.to_array (Rune.jit' (g h) ids) in
            List.iteri
              (fun i (_, _, _, v, _, _, _) ->
                at_most float_exact ~than:centre_tol (apart v (triple_at c i)))
              rows
          in
          check (ring nside) (ring_ids rows);
          check (nested nside) (nested_ids rows));
    ]

(* Harmonics *)

let complex_of (re, im) = Nx.complex Nx.complex128 ~re ~im

let alm lmax (re, im) =
  let shape = [| lmax + 1; lmax + 1 |] in
  complex_of (Nx.create Nx.float64 shape re, Nx.create Nx.float64 shape im)

let map f = Nx.create Nx.float64 [| Array.length f |] f

let re_im z =
  (Nx.to_array (Nx.real Nx.float64 z), Nx.to_array (Nx.imag Nx.float64 z))

(* [rel a b] is the root of the summed squared difference of [a] and [b]
   relative to [b]'s. *)
let rel a b =
  let d = ref 0. and n = ref 0. in
  Array.iteri
    (fun i x ->
      d := !d +. ((x -. b.(i)) ** 2.);
      n := !n +. (b.(i) ** 2.))
    a;
  Float.sqrt (!d /. !n)

let rel_alm a b =
  let ar, ai = re_im a and br, bi = re_im b in
  rel (Array.append ar ai) (Array.append br bi)

let id_harmonic (nside, lmax, _, _, _, _) =
  Printf.sprintf "nside %d, lmax %d" nside lmax

let id_analysis (nside, lmax, _, _, _) =
  Printf.sprintf "nside %d, lmax %d" nside lmax

let healpy_harmonics =
  group "healpy's harmonics"
    [
      cases "synthesis is alm2map's within 1e-14" ~id:id_harmonic
        Reference.harmonics (fun (nside, lmax, a, f, _, _) ->
          let got = Healpix.synthesis (ring nside) (alm lmax a) in
          at_most float_exact ~than:1e-14 (rel (Nx.to_array got) f));
      cases "analysis is map2alm's within 1e-13, with 0 and 3 iterations"
        ~id:id_harmonic Reference.harmonics (fun (nside, lmax, _, f, a0, a3) ->
          let h = ring nside in
          at_most float_exact ~than:1e-13
            (rel_alm
               (Healpix.analysis h ~lmax ~iterations:0 (map f))
               (alm lmax a0));
          at_most float_exact ~than:1e-13
            (rel_alm
               (Healpix.analysis h ~lmax ~iterations:3 (map f))
               (alm lmax a3)));
      cases "analysis of a map drawn per pixel is map2alm's within 1e-13"
        ~id:id_analysis Reference.analyses (fun (nside, lmax, f, a0, a3) ->
          let h = ring nside in
          at_most float_exact ~than:1e-13
            (rel_alm
               (Healpix.analysis h ~lmax ~iterations:0 (map f))
               (alm lmax a0));
          at_most float_exact ~than:1e-13
            (rel_alm
               (Healpix.analysis h ~lmax ~iterations:3 (map f))
               (alm lmax a3)));
    ]

(* [drawn_alm ~band] draws an nside from 4 to 16, an lmax of [band nside],
   and coefficients to it in [-1, 1), real at m = 0 and 0 at m > l. *)
let drawn_alm ~band =
  Gen.bind (Gen.int_range 2 4) (fun order ->
      let nside = 1 lsl order in
      Gen.bind (band nside) (fun lmax ->
          let n = (lmax + 1) * (lmax + 1) in
          Gen.map
            (fun xs ->
              let xs = Array.of_list xs in
              let at l m part =
                if m > l || (m = 0 && part = 1) then 0.
                else xs.((2 * ((l * (lmax + 1)) + m)) + part)
              in
              let part p =
                Array.init n (fun i -> at (i / (lmax + 1)) (i mod (lmax + 1)) p)
              in
              (nside, lmax, part 0, part 1))
            (Gen.list ~size:(Gen.constant (2 * n)) (Gen.float_range (-1.) 1.))))

let at_nside n = Gen.constant n

let power (re, im) lmax =
  let s = ref 0. in
  Array.iteri
    (fun i r ->
      let m = i mod (lmax + 1) in
      let w = if m = 0 then 1. else 2. in
      s := !s +. (w *. ((r *. r) +. (im.(i) *. im.(i)))))
    re;
  !s

(* What a synthesis and an analysis round, below which no iteration
   reaches: float64's precision times about [lmax]. *)
let rounding = 1e-14

(* [rho nside lmax] is the stated largest error of an analysis without
   iterations, from the reference. *)
let rho nside lmax =
  let _, _, r =
    List.find (fun (n, l, _) -> n = nside && l = lmax) Reference.quadrature
  in
  r

let harmonics =
  group "harmonic laws"
    [
      prop ~count:60
        "after k iterations an analysis errs by at most rho^(k + 1), the \
         stated bound"
        (Gen.pair
           (drawn_alm ~band:(fun n -> Gen.of_list [ n; 2 * n ]))
           (Gen.of_list [ 0; 1; 3; 10 ]))
        (fun ((nside, lmax, re, im), k) ->
          cover "nside 4" (nside = 4);
          cover "nside 16" (nside = 16);
          cover "lmax 2 nside" (lmax = 2 * nside);
          cover "10 iterations" (k = 10);
          let h = ring nside and a = alm lmax (re, im) in
          let b =
            Healpix.analysis h ~lmax ~iterations:k (Healpix.synthesis h a)
          in
          let br, bi = re_im b in
          let minus x y = Array.mapi (fun i v -> v -. y.(i)) x in
          let err = power (minus br re, minus bi im) lmax in
          at_most float_exact
            ~than:
              (((rho nside lmax ** float_of_int (k + 1)) *. 1.001) +. rounding)
            (Float.sqrt (err /. power (re, im) lmax)));
      cases "power iteration reaches the stated rho within 2%"
        ~id:(fun (n, l, _) -> Printf.sprintf "nside %d, lmax %d" n l)
        (List.filter (fun (n, _, _) -> n <= 8) Reference.quadrature)
        (fun (nside, lmax, rho) ->
          let h = ring nside in
          let l1 = lmax + 1 in
          let v =
            ref
              (alm lmax
                 ( Array.init (l1 * l1) (fun i ->
                       if i mod l1 > i / l1 then 0.
                       else 1. +. Float.sin (float_of_int i)),
                   Array.init (l1 * l1) (fun i ->
                       if i mod l1 = 0 || i mod l1 > i / l1 then 0.
                       else Float.cos (float_of_int i)) ))
          in
          let ratio = ref 0. in
          for _ = 1 to 60 do
            let q =
              Nx.sub !v
                (Healpix.analysis h ~lmax ~iterations:0 (Healpix.synthesis h !v))
            in
            ratio := Float.sqrt (power (re_im q) lmax /. power (re_im !v) lmax);
            v := q
          done;
          at_most float_exact ~than:(rho *. 1.001) !ratio;
          at_least float_exact ~than:(rho *. 0.98) !ratio);
      prop ~count:40 "Parseval: the map's power comes back within 2 rho^4"
        (drawn_alm ~band:at_nside) (fun (nside, lmax, re, im) ->
          let h = ring nside and a = alm lmax (re, im) in
          let b =
            Healpix.analysis h ~lmax ~iterations:3 (Healpix.synthesis h a)
          in
          let p = power (re, im) lmax and e = rho nside lmax ** 4. in
          at_most float_exact
            ~than:(((2. *. e) +. (e *. e)) *. 1.001 *. p)
            (Float.abs (power (re_im b) lmax -. p)));
      prop ~count:40 "analysis is synthesis' adjoint, scaled by the pixel area"
        (Gen.pair
           (drawn_alm ~band:(fun n -> at_nside (3 * n)))
           (Gen.int_range 0 1000))
        (fun ((nside, lmax, re, im), seed) ->
          let h = ring nside in
          let a = alm lmax (re, im) in
          let f =
            Nx.sin
              (Nx.mul_s
                 (Nx.arange_f Nx.float64 0.
                    (float_of_int (Healpix.pixels h))
                    1.)
                 (float_of_int seed +. 0.37))
          in
          let lhs =
            Nx.item [] (Nx.sum (Nx.mul (Healpix.synthesis h a) f))
            *. 4. *. Float.pi
            /. float_of_int (Healpix.pixels h)
          in
          let br, bi = re_im (Healpix.analysis h ~lmax ~iterations:0 f) in
          let rhs = ref 0. in
          Array.iteri
            (fun i r ->
              let w = if i mod (lmax + 1) = 0 then 1. else 2. in
              rhs := !rhs +. (w *. ((r *. br.(i)) +. (im.(i) *. bi.(i)))))
            re;
          equal (float_rel ~rel:1e-12 ~abs:1e-14) lhs !rhs);
      test "a NESTED map's transforms are its RING map's" (fun () ->
          let nside = 4 and lmax = 8 in
          let n = 12 * nside * nside in
          let f =
            Nx.sin
              (Nx.mul_s (Nx.arange_f Nx.float64 0. (float_of_int n) 1.) 0.71)
          in
          let order =
            Healpix.renumber (nested nside) Healpix.Ring
              (Nx.arange Nx.int64 0 n 1)
          in
          let g = Nx.take order f in
          let a = Healpix.analysis (ring nside) ~lmax ~iterations:2 f in
          at_most float_exact ~than:1e-14
            (rel_alm (Healpix.analysis (nested nside) ~lmax ~iterations:2 g) a);
          at_most float_exact ~than:1e-14
            (rel
               (Nx.to_array (Healpix.synthesis (nested nside) a))
               (Nx.to_array (Nx.take order (Healpix.synthesis (ring nside) a)))));
      test "a batch transforms each map alone" (fun () ->
          let h = ring 2 and lmax = 5 in
          let f = Nx.sin (Nx.mul_s (Nx.arange_f Nx.float64 0. 144. 1.) 0.3) in
          let batch = Nx.reshape [| 3; 48 |] f in
          let a = Healpix.analysis h ~lmax ~iterations:1 batch in
          List.iteri
            (fun i row ->
              let one = Healpix.analysis h ~lmax ~iterations:1 row in
              equal (array float_exact)
                (fst (re_im one))
                (fst (re_im (Nx.slice [ Nx.I i ] a))))
            [
              Nx.slice [ Nx.I 0 ] batch;
              Nx.slice [ Nx.I 1 ] batch;
              Nx.slice [ Nx.I 2 ] batch;
            ]);
      test "eager and compiled agree within 1e-15" (fun () ->
          let h = ring 2 and lmax = 4 in
          let f = Nx.cos (Nx.mul_s (Nx.arange_f Nx.float64 0. 48. 1.) 0.9) in
          let analyse f = Healpix.analysis h ~lmax ~iterations:1 f in
          at_most float_exact ~than:1e-15
            (rel_alm (Rune.jit' analyse f) (analyse f));
          let a = analyse f in
          let synthesise a = Healpix.synthesis h a in
          at_most float_exact ~than:1e-15
            (rel
               (Nx.to_array (Rune.jit' synthesise a))
               (Nx.to_array (synthesise a))));
      test "rune differentiates both transforms" (fun () ->
          let h = ring 2 and lmax = 4 in
          let w = Nx.sin (Nx.arange_f Nx.float64 0. 48. 1.) in
          let f = Nx.cos (Nx.mul_s (Nx.arange_f Nx.float64 0. 48. 1.) 0.9) in
          let a = Healpix.analysis h ~lmax ~iterations:0 f in
          is_ok
            (Rune.check_grads ~tol:1e-8 Nx.Ptree.tensor
               (fun a -> Nx.sum (Nx.mul w (Healpix.synthesis h a)))
               a);
          is_ok
            (Rune.check_grads ~tol:1e-8 Nx.Ptree.tensor
               (fun f ->
                 Nx.sum
                   (Nx.real Nx.float64
                      (Healpix.analysis h ~lmax ~iterations:1 f)))
               f));
      test "a compiled transform's program does not grow with lmax" (fun () ->
          let h = ring 4 in
          let f = Nx.cos (Nx.mul_s (Nx.arange_f Nx.float64 0. 192. 1.) 0.9) in
          let size lmax =
            let p =
              Rune.program Nx.Ptree.tensor Nx.Ptree.tensor
                (fun f -> Healpix.analysis h ~lmax ~iterations:1 f)
                f
            in
            List.length
              (String.split_on_char '\n'
                 (Format.asprintf "%a" Rune.pp_program p))
          in
          equal int (size 4) (size 64));
      test "compiled at nside 128, lmax 256, as eagerly within 1e-14" (fun () ->
          let h = ring 128 in
          let f =
            Nx.sin (Nx.mul_s (Nx.arange_f Nx.float64 0. 196608. 1.) 0.37)
          in
          let analyse f = Healpix.analysis h ~lmax:256 ~iterations:0 f in
          at_most float_exact ~than:1e-14
            (rel_alm (Rune.jit' analyse f) (analyse f)));
      test "rings whose FFT is Bluestein's give real m = 0 coefficients"
        (fun () ->
          (* nside 256's cap ring 251 holds 1004 pixels, whose prime factor
             251 the FFT reaches by Bluestein's algorithm. *)
          let h = ring 256 in
          let f =
            Nx.sin (Nx.mul_s (Nx.arange_f Nx.float64 0. 786432. 1.) 0.37)
          in
          let _, im = re_im (Healpix.analysis h ~lmax:4 ~iterations:0 f) in
          equal (array float_exact) (Array.make 5 0.)
            (Array.init 5 (fun l -> im.(l * 5))));
      test "analysis and synthesis refuse shapes they cannot read" (fun () ->
          let h = ring 2 in
          raises
            (Invalid_argument
               "Healpix.analysis: expected maps of 48 pixels on the last axis, \
                got shape [47]") (fun () ->
              Healpix.analysis h ~lmax:2 ~iterations:0
                (Nx.zeros Nx.float64 [| 47 |]));
          raises (Invalid_argument "Healpix.analysis: lmax -1 is negative")
            (fun () ->
              Healpix.analysis h ~lmax:(-1) ~iterations:0
                (Nx.zeros Nx.float64 [| 48 |]));
          raises
            (Invalid_argument "Healpix.analysis: iterations -1 is negative")
            (fun () ->
              Healpix.analysis h ~lmax:2 ~iterations:(-1)
                (Nx.zeros Nx.float64 [| 48 |]));
          raises
            (Invalid_argument
               "Healpix.synthesis: expected coefficients [...; lmax + 1; lmax \
                + 1], got shape [3; 4]") (fun () ->
              Healpix.synthesis h (Nx.zeros Nx.complex128 [| 3; 4 |])));
    ]

let () =
  exit
    (run "Healpix"
       [ healpy_pixels; healpy_cells; laws; healpy_harmonics; harmonics ])
