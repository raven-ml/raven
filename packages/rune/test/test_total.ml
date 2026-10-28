(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.Total: collect and add, the order additions chain in through every
   construct, and where a compiled call takes a running value. *)

open Windtrap
open Compose
module Total = Rune.Total

let invalid ~sub f = raises_match (Exn.invalid_arg ~substring:sub) f
let f32 = Nx.float32
let v xs = Nx.create f32 [| Array.length xs |] xs
let zero () = Nx.scalar f32 (-0.)

let test_collect () =
  let t = Total.make () in
  let a = Nx.scalar f32 1.5 and b = Nx.scalar f32 (-0.25) in
  let r, sum =
    Total.collect t ~init:(zero ()) (fun () ->
        Total.add t a;
        Total.add t b;
        "done")
  in
  equal string "done" r;
  equal bits_t (bits Nx.(add (add (zero ()) a) b)) (bits sum);
  (* No addition keeps the zero's bits. *)
  let _, sum = Total.collect t ~init:(zero ()) (fun () -> ()) in
  equal bits_t (bits (zero ())) (bits sum);
  (* With no scope open, an addition does nothing: the next collect receives
     none. *)
  Total.add t a;
  let _, sum = Total.collect t ~init:(zero ()) (fun () -> ()) in
  equal ~msg:"no scope" bits_t (bits (zero ())) (bits sum);
  (* A scope of another total lets the additions pass. *)
  let u = Total.make () in
  let ((), inner), outer =
    Total.collect t ~init:(zero ()) (fun () ->
        Total.collect u ~init:(zero ()) (fun () ->
            Total.add t a;
            Total.add u b))
  in
  equal bits_t (bits b) (bits inner);
  equal bits_t (bits a) (bits outer)

(* Two open scopes of one total: the innermost takes the additions. *)
let test_same_total () =
  let t = Total.make () in
  let a = Nx.scalar f32 1.5 and b = Nx.scalar f32 (-0.25) in
  let ((), inner), outer =
    Total.collect t ~init:(zero ()) (fun () ->
        Total.add t a;
        Total.collect t ~init:(zero ()) (fun () -> Total.add t b))
  in
  equal bits_t (bits Nx.(add (zero ()) b)) (bits inner);
  equal bits_t (bits Nx.(add (zero ()) a)) (bits outer)

(* A scope inside a jit sums the function's additions there. *)
let test_scope_in_jit () =
  let t = Total.make () in
  let f x =
    let y, s =
      Total.collect t ~init:(zero ()) (fun () ->
          Total.add t (Nx.sum x);
          Nx.neg x)
    in
    Nx.add y s
  in
  let x = v [| 0.5; 2. |] in
  equal bits_t (bits (f x)) (bits (Rune.jit' f x))

(* An addition counts where it is made, and a traced function's as its construct
   returns: a jit whose function raises adds nothing. *)
let test_raising_jit () =
  let t = Total.make () in
  let f x =
    Total.add t x;
    raise Exit
  in
  let x = Nx.scalar f32 2. in
  let caught g = try ignore (g x) with Exit -> () in
  let (), eager = Total.collect t ~init:(zero ()) (fun () -> caught f) in
  let (), jitted =
    Total.collect t ~init:(zero ()) (fun () -> caught (Rune.jit' f))
  in
  equal bits_t (bits Nx.(add (zero ()) x)) (bits eager);
  equal bits_t (bits (zero ())) (bits jitted)

(* The additions a function makes, collected around it and around its jit, give
   the same total, its zero's sign kept. *)
let additions =
  let open Gen in
  let value = map (fun i -> pool.(i)) (int_range 0 (Array.length pool - 1)) in
  let zero = frequency [ (1, constant (-0.)); (1, constant 0.); (2, value) ] in
  pair zero (list ~size:(int_range 0 4) value)
  |> with_pp (fun ppf (z, cs) ->
      Format.fprintf ppf "zero %h, adds %s" z
        (String.concat " " (List.map (Printf.sprintf "%h") cs)))

let law_additions (z, cs) =
  cover "no addition" (cs = []);
  cover "a -0 zero" (z = 0. && 1. /. z < 0.);
  let t = Total.make () in
  let f x =
    List.iter (fun c -> Total.add t (Nx.mul x (Nx.scalar f32 c))) cs;
    Nx.neg x
  in
  let x = Nx.scalar f32 1. and zero () = Nx.scalar f32 z in
  let y, s = Total.collect t ~init:(zero ()) (fun () -> f x) in
  let y', s' = Total.collect t ~init:(zero ()) (fun () -> Rune.jit' f x) in
  equal bits_t (bits y) (bits y');
  equal bits_t (bits s) (bits s')

let test_shape_refused () =
  let t = Total.make () in
  invalid ~sub:"Rune.Total.add: shape [2], where the total's init has []"
    (fun () ->
      Total.collect t ~init:(zero ()) (fun () -> Total.add t (v [| 1.; 2. |])))

(* A total made at the top level serves every set; each addition lies on its
   collect's [init]'s set, eagerly and traced. *)
module M1 = (val Nx.devices [ Nx_support.memory 1 ])
module M2 = (val Nx.devices [ Nx_support.memory 2 ])

let top : (float, Nx.float32_elt, 'd) Total.t = Total.make ()
let placed x = Format.asprintf "%a" Nx.Placement.pp (Nx.placement x)

let summed (type d) (p : d Nx.Placement.t) ~jit =
  let at x = Nx.scalar_at p f32 x in
  let f () =
    Total.add top (at 1.5);
    Total.add top (at (-0.25))
  in
  let run () =
    if jit then
      ignore
        (Rune.jit'
           (fun x ->
             f ();
             x)
           (at 0.))
    else f ()
  in
  let (), sum = Total.collect top ~init:(at (-0.)) run in
  (bits sum, placed sum)

let test_sets () =
  let want =
    bits Nx.(add (add (zero ()) (scalar f32 1.5)) (scalar f32 (-0.25)))
  in
  List.iter
    (fun jit ->
      let msg = if jit then "jit" else "eager" in
      equal ~msg (pair bits_t string) (want, "m1") (summed M1.on ~jit);
      equal ~msg (pair bits_t string) (want, "m2") (summed M2.on ~jit))
    [ false; true ]

(* Another mint of [M1]'s device list: another set. *)
module M1' = (val Nx.devices [ Nx_support.memory 1 ])

(* An addition on [M2] under a collect on [M1] raises, wherever it is made, and
   reaches no collect around: the one on [M2] keeps its [init]. *)
let test_set_refused () =
  let sub =
    "Rune.Total.add: a value on [m2], where the total's init lies on [m1]"
  in
  let x2 () = Nx.scalar_at M2.on f32 1. in
  let add x = Total.add top x in
  let warm =
    Rune.jit' (fun x ->
        add x;
        x)
  in
  ignore
    (Total.collect top ~init:(Nx.scalar_at M2.on f32 0.) (fun () ->
         warm (x2 ())));
  let made =
    [
      ("eager", fun () -> add (x2 ()));
      ( "jit",
        fun () ->
          ignore
            (Rune.jit'
               (fun x ->
                 add x;
                 x)
               (x2 ())) );
      ("a warm jit", fun () -> ignore (warm (x2 ())));
      ( "scan",
        fun () ->
          ignore
            (Rune.scan'
               ~f:(fun c r ->
                 add r;
                 (c, r))
               ~init:(x2 ())
               (Nx.full_at M2.on f32 [| 3 |] 1.)) );
      ( "vmap",
        fun () ->
          ignore
            (Rune.vmap'
               (fun x ->
                 add x;
                 x)
               (Nx.full_at M2.on f32 [| 2 |] 1.)) );
      ( "grad",
        fun () ->
          ignore
            (Rune.grad'
               (fun x ->
                 add x;
                 x)
               (x2 ())) );
    ]
  in
  List.iter
    (fun (msg, f) ->
      let outer = Nx.scalar_at M2.on f32 (-0.) in
      let (), sum =
        Total.collect top ~init:outer (fun () ->
            raises_match ~msg (Exn.invalid_arg ~substring:sub) (fun () ->
                Total.collect top ~init:(Nx.scalar_at M1.on f32 0.) f))
      in
      equal ~msg bits_t (bits outer) (bits sum))
    made;
  invalid
    ~sub:
      "Rune.Total.add: a value on another set of the devices [m1] than the \
       total's init's" (fun () ->
      Total.collect top ~init:(Nx.scalar_at M1'.on f32 0.) (fun () ->
          add (Nx.scalar_at M1.on f32 1.)))

(* A construct whose delivery to one total raises delivers to none: the
   additions of a construct that raises reach no total. *)
let test_delivery_raises () =
  let refused
      ?(sub = "Rune.Total.add: shape [3], where the total's init has [2]") f =
    invalid ~sub (fun () -> ignore (f ()))
  in
  let t1 = Total.make () and t2 = Total.make () in
  let z n = Nx.zeros f32 [| n |] in
  let (), s1 =
    Total.collect t1 ~init:(z 2) (fun () ->
        refused (fun () ->
            Total.collect t2 ~init:(z 2) (fun () ->
                Rune.scan' ~init:(Nx.zeros f32 [||])
                  ~f:(fun c r ->
                    Total.add t1 (Nx.add (z 2) r);
                    Total.add t2 (Nx.add (z 3) r);
                    (c, c))
                  (Nx.ones f32 [| 4 |]))))
  in
  equal bits_t (bits (z 2)) (bits s1);
  (* Under a map, and in a root's solve. *)
  let (), s1 =
    Total.collect t1 ~init:(z 2) (fun () ->
        refused (fun () ->
            Total.collect t2 ~init:(z 2) (fun () ->
                Rune.vmap'
                  (fun x ->
                    fst
                    @@ Rune.scan' ~init:(Nx.zeros f32 [||])
                         ~f:(fun c r ->
                           Total.add t1 (Nx.add (z 2) r);
                           Total.add t2 (Nx.add (z 3) r);
                           (c, c))
                         x)
                  (Nx.ones f32 [| 2; 4 |]))))
  in
  equal ~msg:"under a map" bits_t (bits (z 2)) (bits s1);
  let (), s1 =
    Total.collect t1 ~init:(z 2) (fun () ->
        refused (fun () ->
            Total.collect t2 ~init:(z 2) (fun () ->
                Rune.vmap'
                  (Rune.iterate' ~max:4
                     ~until:(fun x -> Nx.greater_equal_s x 3.)
                     ~f:(fun x ->
                       Total.add t1 (Nx.add (z 2) x);
                       Total.add t2 (Nx.add (z 3) x);
                       Nx.add_s x 1.))
                  (v [| 0.; 2. |]))))
  in
  equal ~msg:"under a map whose lanes stop apart" bits_t (bits (z 2)) (bits s1);
  let (), s1 =
    Total.collect t1 ~init:(z 2) (fun () ->
        refused (fun () ->
            Total.collect t2 ~init:(z 2) (fun () ->
                Rune.root Nx.Ptree.tensor
                  ~residual:(fun x -> Nx.sub x (v [| 1. |]))
                  (fun () ->
                    Total.add t1 (z 2);
                    Total.add t2 (z 3);
                    v [| 1. |]))))
  in
  equal ~msg:"in a root's solve" bits_t (bits (z 2)) (bits s1);
  (* A compiled call run again at its key traces nothing: its delivery checks
     each total first. *)
  let f =
    Rune.jit' (fun x ->
        Total.add t1 x;
        Total.add t2 x;
        x)
  in
  let x = Nx.ones f32 [| 2 |] in
  let call n =
    Total.collect t1 ~init:(z 2) (fun () ->
        if n = 2 then ignore (Total.collect t2 ~init:(z n) (fun () -> f x))
        else
          refused
            ~sub:"Rune.Total.add: shape [2], where the total's init has [3]"
            (fun () -> Total.collect t2 ~init:(z n) (fun () -> f x)))
  in
  equal ~msg:"a compiled call" bits_t (bits x) (bits (snd (call 2)));
  equal ~msg:"a compiled call refused" bits_t (bits (z 2)) (bits (snd (call 3)))

let test_raises () =
  let t = Total.make () in
  raises Exit (fun () ->
      Total.collect t ~init:(zero ()) (fun () ->
          Total.add t (Nx.scalar f32 1.);
          raise Exit))

(* A scope around a jit receives the function's additions once per call. *)
let test_jit_scope () =
  let t = Total.make () and traces = ref 0 in
  let f x =
    incr traces;
    Total.add t x;
    Nx.neg x
  in
  let g = Rune.jit' f in
  let x = v [| 0.5; 2. |] in
  let calls = Rune.evaluations () in
  let (y, z), sum =
    Total.collect t
      ~init:(v [| 0.; 0. |])
      (fun () ->
        let y = g x in
        (y, g x))
  in
  equal ~msg:"compiled" int 0 (Rune.evaluations () - calls);
  equal ~msg:"one trace" int 1 !traces;
  equal ~msg:"the first call" bits_t (bits (Nx.neg x)) (bits y);
  equal ~msg:"the cached call" bits_t (bits (Nx.neg x)) (bits z);
  equal ~msg:"two additions" bits_t
    (bits (Nx.add (Nx.add (v [| 0.; 0. |]) x) x))
    (bits sum)

(* Where additions chain. A scan's step adds to a total before and after a
   construct it runs, which adds too: the additions reach a collect that already
   holds a running value in the order they are made, each added to the total so
   far, as the same additions written out eagerly do, bit for bit. Values of
   every set and placed ones mix, as constants and computed values do. *)
type inner = Plain | Scan_in | Iterate_in | Remat_in | Custom_in | Jit_in

let pp_inner ppf i =
  Format.pp_print_string ppf
    (match i with
    | Plain -> "nothing"
    | Scan_in -> "a scan"
    | Iterate_in -> "an iterate"
    | Remat_in -> "a remat"
    | Custom_in -> "a custom_jvp"
    | Jit_in -> "a jit")

type chained = {
  zero : float * bool;
  pre : float * bool;
  rows : int;
  inner : inner;
  trips : int;
  adds : (float * bool) array;  (** Before, inside twice, after. *)
  mode : mode;
}

and mode =
  | Eager
  | Jitted
  | Forward
  | Reverse
  | Mapped
  | Jit_forward
  | Jit_reverse
  | Jit_mapped

let pp_mode ppf m =
  Format.pp_print_string ppf
    (match m with
    | Eager -> "eagerly"
    | Jitted -> "under jit"
    | Forward -> "under jvp"
    | Reverse -> "under grad"
    | Mapped -> "under vmap"
    | Jit_forward -> "under jit of jvp"
    | Jit_reverse -> "under jit of grad"
    | Jit_mapped -> "under jit of vmap")

let inners = [ Plain; Scan_in; Iterate_in; Remat_in; Custom_in; Jit_in ]

let modes =
  [
    Eager;
    Jitted;
    Forward;
    Reverse;
    Mapped;
    Jit_forward;
    Jit_reverse;
    Jit_mapped;
  ]

let chained =
  let open Gen in
  let value =
    pair (of_list [ 1e8; -1e8; 12.; 3.; 0.1; 1e-8; -0.; 0.; 7.25 ]) bool
  in
  (let* zero = value and+ pre = value in
   let* rows = int_range 0 3 and+ trips = int_range 0 3 in
   let* inner = of_list inners in
   let* adds = array ~size:(constant 4) value and+ mode = of_list modes in
   constant { zero; pre; rows; inner; trips; adds; mode })
  |> with_pp (fun ppf c ->
      let value ppf (x, placed) =
        Format.fprintf ppf "%g%s" x (if placed then " placed" else "")
      in
      Format.fprintf ppf
        "zero %a, %a first, %d rows, %a of %d trips, %a, adds %a" value c.zero
        value c.pre c.rows pp_inner c.inner c.trips pp_mode c.mode
        (Format.pp_print_list ~pp_sep:Format.pp_print_space value)
        (Array.to_list c.adds))

(* One case of each mode, each inner construct among them, no row, and a scan
   and an iterate of no trip, so that every run reaches each. *)
let chained_examples =
  List.mapi
    (fun k mode ->
      let inner = List.nth inners (k mod List.length inners) in
      let value = ((if k mod 2 = 0 then 0.1 else -0.), k mod 3 = 0) in
      {
        zero = value;
        pre = (12., k mod 2 = 1);
        rows = k mod 4;
        inner;
        trips = (match k with 1 | 2 -> 0 | k -> k mod 4);
        adds = [| (1e8, false); (1e-8, true); (-1e8, false); value |];
        mode;
      })
    modes

let law_chained c =
  List.iter
    (fun i -> cover (Format.asprintf "inside %a" pp_inner i) (c.inner = i))
    inners;
  List.iter
    (fun m -> cover (Format.asprintf "under %a" pp_mode m) (c.mode = m))
    modes;
  cover "no row" (c.rows = 0);
  cover "a scan of no trip" (c.inner = Scan_in && c.trips = 0);
  cover "an iterate of no trip" (c.inner = Iterate_in && c.trips = 0);
  let t = Total.make () in
  (* A placed value is computed; one of every set is a formula. An addition
     scales its value by the carry's sum, one, so that it has a tangent. *)
  let value (x, placed) =
    let s = Nx.scalar f32 x in
    if placed then Nx.place Nx.Host.on s else s
  in
  let add k x = Total.add t (Nx.mul (Nx.sum x) (value c.adds.(k))) in
  let inner_ops carry =
    match c.inner with
    | Plain -> ()
    | Scan_in ->
        ignore
          (Rune.scan' ~init:carry
             ~f:(fun x _ ->
               add 1 x;
               add 2 x;
               (x, x))
             (Nx.zeros f32 [| c.trips; 1 |]))
    | Iterate_in ->
        ignore
          (Rune.iterate
             Nx.Ptree.(pair tensor tensor)
             ~max:c.trips
             ~until:(fun (_, k) ->
               Nx.greater_equal k (Nx.scalar f32 (Float.of_int c.trips)))
             ~f:(fun (x, k) ->
               add 1 x;
               add 2 x;
               (x, Nx.add k (Nx.scalar f32 1.)))
             (carry, Nx.scalar f32 0.))
    | Remat_in ->
        ignore
          (Rune.remat
             Nx.Ptree.(tensor @-> returns tensor)
             (fun x ->
               add 1 x;
               add 2 x;
               x)
             carry)
    | Custom_in ->
        ignore
          (Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor
             (fun x ->
               add 1 x;
               add 2 x;
               (x, Fun.id))
             carry)
    | Jit_in ->
        ignore
          (Rune.jit'
             (fun x ->
               add 1 x;
               add 2 x;
               x)
             carry)
  in
  let inner_trips =
    match c.inner with
    | Plain -> 0
    | Scan_in | Iterate_in -> c.trips
    | Remat_in | Custom_in | Jit_in -> 1
  in
  (* The carry and the collect's total, the collect inside the transformation:
     the additions written out, or made in constructs. *)
  let collected f x =
    let y, s =
      Total.collect t ~init:(value c.zero) (fun () ->
          Total.add t (value c.pre);
          f x)
    in
    (Nx.sum y, s)
  in
  let written =
    collected (fun x ->
        for _ = 1 to c.rows do
          add 0 x;
          for _ = 1 to inner_trips do
            add 1 x;
            add 2 x
          done;
          add 3 x
        done;
        x)
  in
  let run =
    collected (fun init ->
        fst
          (Rune.scan' ~init
             ~f:(fun x _ ->
               add 0 x;
               inner_ops x;
               add 3 x;
               (x, x))
             (Nx.zeros f32 [| c.rows; 1 |])))
  in
  let init = v [| 1. |] and p = Nx.Ptree.(pair tensor tensor) in
  let both (y, s) = (bits y, bits s) in
  let pair = Testable.pair bits_t bits_t in
  let before = Rune.evaluations () in
  (match c.mode with
  | Eager -> equal pair (both (written init)) (both (run init))
  | Jitted ->
      equal pair
        (both (written init))
        (both (Rune.jit Nx.Ptree.(tensor @-> returns p) run init))
  | Forward | Jit_forward ->
      (* The total's tangent too, a collect inside the differentiation receiving
         the additions' tangents. *)
      let jvp f = Rune.jvp Nx.Ptree.tensor p f init init in
      let jvp_run =
        if c.mode = Forward then jvp run
        else
          Rune.jit
            Nx.Ptree.(tensor @-> returns (pair p p))
            (fun x -> Rune.jvp Nx.Ptree.tensor p run x x)
            init
      in
      equal pair (both (fst (jvp written))) (both (fst jvp_run));
      (* A loop's sums hold a tangent at every step once one has, a zero of
         either sign where the additions written out give none. A custom rule's
         additions are its primal's, which forward mode runs on the primals. *)
      if c.inner <> Custom_in then
        let l = lenient ~nan:false in
        equal (Testable.pair l l)
          (both (snd (jvp written)))
          (both (snd jvp_run))
  | Reverse | Jit_reverse ->
      let aux f x =
        let _, g, s =
          Rune.value_and_grad_aux Nx.Ptree.tensor Nx.Ptree.tensor f x
        in
        (g, s)
      in
      let aux_run =
        if c.mode = Reverse then aux run init
        else Rune.jit Nx.Ptree.(tensor @-> returns p) (aux run) init
      in
      equal pair (both (aux written init)) (both aux_run)
  | Mapped | Jit_mapped ->
      (* A collect inside the map: each lane's total. *)
      let lanes = Nx.reshape [| 2; 1 |] (v [| 1.; 2. |]) in
      let map f = Rune.vmap Nx.Ptree.(tensor @-> returns p) f in
      let map_run =
        if c.mode = Mapped then map run lanes
        else Rune.jit Nx.Ptree.(tensor @-> returns p) (map run) lanes
      in
      equal pair (both (map written lanes)) (both map_run));
  (* A jit around a transformation compiles its derived programs. *)
  let compiled = Rune.evaluations () = before in
  List.iter
    (fun m ->
      cover (Format.asprintf "%a compiled" pp_mode m) (c.mode = m && compiled))
    [ Jitted; Jit_forward; Jit_reverse; Jit_mapped ]

(* Under a map, each addition adds its lanes in turn, lane 0 first and an outer
   map's lanes before an inner map's, one addition after another, bit for bit.
   [ordered] makes that order with a loop of scalar additions, and with one lane
   it is what the body makes run without the map. Where it runs, each body adds
   [x] then [1 - x]: once, on each of a scan's two trips (the second over [2x]),
   in a jit or a remat, on each trip of an iterate whose lanes over 2 run two
   trips and the others one, in a switch whose lanes over 2 take a branch adding
   twice, in a root's solve, whose residual's additions are dropped, or in a
   custom rule whose pullback or tangent map also adds [x] where a
   differentiation applies it. Over [1e8; 1; -1e8; 1] from 1, adding lane by
   lane, or a lane's additions from the identity, rounds otherwise. *)
type lane_body =
  | Direct
  | Scan_lane
  | Jit_lane
  | Remat_lane
  | Apart
  | Nested_apart
  | Together
  | Switch_lane
  | Switch_shared
  | Root_lane
  | Pullback
  | Tangent
  | Constant
  | Scan_jit

type lane_mode =
  | Map
  | Jit_of_map
  | Grad_of_map
  | Jvp_of_map
  | Map_of_map
  | Map_of_jit
  | Map_of_grad
  | Map_of_jvp
  | Map_grad_map
  | Grad_of_jit_of_map
  | Map_of_grad_of_jit

type in_turn = {
  start : float;
  lanes : float array;
  body : lane_body;
  lmode : lane_mode;
}

let lane_bodies =
  [
    Direct;
    Scan_lane;
    Jit_lane;
    Remat_lane;
    Apart;
    Nested_apart;
    Together;
    Switch_lane;
    Switch_shared;
    Root_lane;
    Pullback;
    Tangent;
    Constant;
    Scan_jit;
  ]

let lane_modes =
  [
    Map;
    Jit_of_map;
    Grad_of_map;
    Jvp_of_map;
    Map_of_map;
    Map_of_jit;
    Map_of_grad;
    Map_of_jvp;
    Map_grad_map;
    Grad_of_jit_of_map;
    Map_of_grad_of_jit;
  ]

let pp_body ppf b =
  Format.pp_print_string ppf
    (match b with
    | Direct -> "an addition"
    | Scan_lane -> "a scan"
    | Jit_lane -> "a jit"
    | Remat_lane -> "a remat"
    | Apart -> "an iterate stopping apart"
    | Nested_apart -> "a map inside an iterate stopping apart"
    | Together -> "an iterate stopping together"
    | Switch_lane -> "a switch taking branches apart"
    | Switch_shared -> "a switch every lane takes"
    | Root_lane -> "a root"
    | Pullback -> "a custom_vjp rule"
    | Tangent -> "a custom_jvp rule"
    | Constant -> "an addition of a constant"
    | Scan_jit -> "a jit over a constant in a scan")

let pp_lmode ppf m =
  Format.pp_print_string ppf
    (match m with
    | Map -> "vmap"
    | Jit_of_map -> "jit of vmap"
    | Grad_of_map -> "grad of vmap"
    | Jvp_of_map -> "jvp of vmap"
    | Map_of_map -> "vmap of vmap"
    | Map_of_jit -> "vmap of jit"
    | Map_of_grad -> "vmap of grad"
    | Map_of_jvp -> "vmap of jvp"
    | Map_grad_map -> "vmap of grad of vmap"
    | Grad_of_jit_of_map -> "grad of jit of vmap"
    | Map_of_grad_of_jit -> "vmap of grad of jit")

(* A custom_vjp rule has no forward derivative. *)
let valid body lmode =
  match (body, lmode) with
  | Pullback, Jvp_of_map -> Grad_of_map
  | Pullback, Map_of_jvp -> Map_of_grad
  | _ -> lmode

let in_turn =
  let open Gen in
  let value = of_list [ 1e8; -1e8; 1.; 3.; 0.1; -0.; 7.25 ] in
  (let* start = value and+ lanes = array ~size:(int_range 0 6) value in
   let* body = of_list lane_bodies and+ lmode = of_list lane_modes in
   constant { start; lanes; body; lmode = valid body lmode })
  |> with_pp (fun ppf c ->
      Format.fprintf ppf "start %g, lanes [%a], %a under %a" c.start
        (Format.pp_print_list
           ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
           (fun ppf x -> Format.fprintf ppf "%g" x))
        (Array.to_list c.lanes) pp_body c.body pp_lmode c.lmode)

(* Every body under every mode, over lanes that cancel and over one lane, so
   that every run reaches the cases another order would round. *)
let in_turn_examples =
  List.concat_map
    (fun body ->
      List.concat_map
        (fun lmode ->
          let lmode = valid body lmode in
          [
            { start = 1.; lanes = [| 1e8; 1.; -1e8; 1. |]; body; lmode };
            { start = 1.; lanes = [| 1e8; 3.; -1e8; 1. |]; body; lmode };
            { start = 1.; lanes = [| 1e8 |]; body; lmode };
            { start = 1.; lanes = [| -1e8; 3.; 1. |]; body; lmode };
            { start = 1.; lanes = [||]; body; lmode };
          ])
        lane_modes)
    lane_bodies

let s = Nx.scalar f32

let differentiated c =
  match c.lmode with
  | Grad_of_map | Jvp_of_map | Map_of_grad | Map_of_jvp | Map_grad_map
  | Grad_of_jit_of_map | Map_of_grad_of_jit ->
      true
  | Map | Jit_of_map | Map_of_map | Map_of_jit -> false

let reverse c =
  differentiated c && c.lmode <> Jvp_of_map && c.lmode <> Map_of_jvp

(* Each lane's additions, by position in the order the map makes them: [[]]
   where the lane makes none there, several where an inner map's lanes add at
   that position. A switch's branches add one after another, the lowest index
   first. A tangent map's additions are summed from -0, that sum one addition.
   An iterate apart tests its stop, which adds [x / 2], at its init and after
   each trip: those over 2 run two trips, those under 0 none, the others one. *)
let lane_adds c x =
  let one y = [ s y ] and minus y = [ Nx.sub (s 1.) (s y) ] in
  let pair y = [ one y; minus y ] in
  let none = [ []; [] ] in
  let summed y = Nx.add (Nx.add (zero ()) (s y)) (Nx.sub (s 1.) (s y)) in
  let trips = if x > 2. then 2 else if x < 0. then 0 else 1 in
  let trip k ys = if trips >= k then ys else List.map (fun _ -> []) ys in
  let u = one (x /. 2.) in
  match c.body with
  | Direct | Jit_lane | Remat_lane | Root_lane -> pair x
  | Scan_lane -> pair x @ pair (2. *. x)
  | Apart -> (u :: trip 1 (pair x @ [ u ])) @ trip 2 (pair x @ [ u ])
  | Nested_apart ->
      let rows y = [ one y @ one (2. *. y); minus y @ minus (2. *. y) ] in
      trip 1 (rows x) @ trip 2 (rows x)
  | Together | Switch_shared -> pair x @ pair x
  | Switch_lane ->
      if x > 2. then none @ pair x @ pair x else pair x @ none @ none
  | Constant -> [ [ s 1e8 ]; minus x ]
  | Scan_jit -> pair 1e8 @ pair x @ pair 2e8 @ pair x
  | Pullback -> pair x @ if reverse c then pair x else []
  | Tangent -> pair x @ if differentiated c then [ [ summed x ] ] else []

let ordered t rows =
  let count = Array.fold_left (fun m r -> max m (Array.length r)) 0 rows in
  for d = 0 to count - 1 do
    Array.iter
      (fun r -> if d < Array.length r then List.iter (Total.add t) r.(d))
      rows
  done

let rows c lanes = Array.map (fun x -> Array.of_list (lane_adds c x)) lanes

(* Two orders the law tells apart: lane by lane, and each lane's additions
   summed from the identity, the lanes' sums then in turn. *)
let lane_major t c =
  Array.iter
    (fun x -> List.iter (List.iter (Total.add t)) (lane_adds c x))
    c.lanes

let from_identity t c =
  Array.iter
    (fun x ->
      let add r a = List.fold_left Nx.add r a in
      Total.add t (List.fold_left add (zero ()) (lane_adds c x)))
    c.lanes

let law_in_turn c =
  List.iter
    (fun b -> cover (Format.asprintf "%a" pp_body b) (c.body = b))
    lane_bodies;
  List.iter
    (fun m -> cover (Format.asprintf "under %a" pp_lmode m) (c.lmode = m))
    lane_modes;
  let n = Array.length c.lanes in
  cover "no lane" (n = 0);
  cover "one lane" (n = 1);
  let ran k = Array.exists (fun x -> k (x > 2.)) c.lanes in
  cover "lanes held apart"
    ((c.body = Apart || c.body = Nested_apart || c.body = Switch_lane)
    && ran Fun.id && ran not);
  cover "a lane stopping at its init"
    (c.body = Apart && Array.exists (fun x -> x < 0.) c.lanes);
  let t = Total.make () in
  let pair y =
    Total.add t y;
    Total.add t (Nx.sub (s 1.) y)
  in
  let body x =
    match c.body with
    | Direct ->
        pair x;
        x
    | Scan_lane ->
        fst
          (Rune.scan' ~init:x
             ~f:(fun y k ->
               pair (Nx.mul x k);
               (y, y))
             (v [| 1.; 2. |]))
    | Jit_lane ->
        Rune.jit'
          (fun x ->
            pair x;
            x)
          x
    | Remat_lane ->
        Rune.remat
          Nx.Ptree.(tensor @-> returns tensor)
          (fun x ->
            pair x;
            Nx.mul x x)
          x
    | Apart ->
        let trips =
          Nx.where
            (Nx.greater x (s 2.))
            (s 2.)
            (Nx.where (Nx.less x (s 0.)) (s 0.) (s 1.))
        in
        fst
          (Rune.iterate
             Nx.Ptree.(pair tensor tensor)
             ~max:2
             ~until:(fun (_, k) ->
               Total.add t (Nx.div_s x 2.);
               Nx.greater_equal k trips)
             ~f:(fun (y, k) ->
               pair x;
               (y, Nx.add k (s 1.)))
             (x, s 0.))
    | Nested_apart ->
        let trips =
          Nx.where
            (Nx.greater x (s 2.))
            (s 2.)
            (Nx.where (Nx.less x (s 0.)) (s 0.) (s 1.))
        in
        fst
          (Rune.iterate
             Nx.Ptree.(pair tensor tensor)
             ~max:2
             ~until:(fun (_, k) -> Nx.greater_equal k trips)
             ~f:(fun (y, k) ->
               ignore
                 (Rune.vmap'
                    (fun r ->
                      pair (Nx.mul x r);
                      r)
                    (v [| 1.; 2. |]));
               (y, Nx.add k (s 1.)))
             (x, s 0.))
    | Together ->
        fst
          (Rune.iterate
             Nx.Ptree.(pair tensor tensor)
             ~max:2
             ~until:(fun (_, k) -> Nx.greater_equal k (s 2.))
             ~f:(fun (y, k) ->
               pair x;
               (y, Nx.add k (s 1.)))
             (x, s 0.))
    | Switch_shared ->
        Rune.switch Nx.Ptree.tensor (Nx.scalar Nx.int32 1l)
          [
            (fun () -> x);
            (fun () ->
              pair x;
              pair x;
              Nx.neg x);
          ]
    | Constant ->
        Total.add t (s 1e8);
        Total.add t (Nx.sub (s 1.) x);
        x
    | Scan_jit ->
        let g =
          Rune.jit' (fun k ->
              pair (Nx.mul_s k 1e8);
              k)
        in
        fst
          (Rune.scan' ~init:x
             ~f:(fun y k ->
               ignore (g k);
               pair x;
               (y, y))
             (v [| 1.; 2. |]))
    | Switch_lane ->
        let i = Nx.cast Nx.int32 (Nx.greater x (s 2.)) in
        Rune.switch Nx.Ptree.tensor i
          [
            (fun () ->
              pair x;
              x);
            (fun () ->
              pair x;
              pair x;
              Nx.neg x);
          ]
    | Root_lane ->
        Rune.root Nx.Ptree.tensor
          ~residual:(fun y ->
            pair x;
            Nx.sub y x)
          (fun () ->
            pair x;
            Nx.add x (s 0.))
    | Pullback ->
        Rune.custom_vjp Nx.Ptree.tensor Nx.Ptree.tensor
          (fun x ->
            pair x;
            ( x,
              fun ct ->
                pair (Nx.mul ct x);
                ct ))
          x
    | Tangent ->
        Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor
          (fun x ->
            pair x;
            ( Nx.mul x x,
              fun dx ->
                pair x;
                Nx.mul (Nx.mul_s x 2.) dx ))
          x
  in
  (* The transformation inside the map, and the one around it. *)
  let inside f x =
    match c.lmode with
    | Map_of_jit -> Rune.jit' f x
    | Map_of_grad -> Rune.grad' (fun x -> Nx.sum (f x)) x
    | Map_of_jvp -> snd (Rune.jvp' f x x)
    | Map_grad_map -> Rune.grad' (fun x -> Nx.sum (f x)) x
    | Map_of_grad_of_jit -> Rune.grad' (fun x -> Nx.sum (Rune.jit' f x)) x
    | Map | Jit_of_map | Grad_of_map | Jvp_of_map | Map_of_map
    | Grad_of_jit_of_map ->
        f x
  in
  let under f x =
    match c.lmode with
    | Map | Map_of_map | Map_of_jit | Map_of_grad | Map_of_jvp | Map_grad_map
    | Map_of_grad_of_jit ->
        ignore (f x)
    | Grad_of_jit_of_map ->
        ignore (Rune.grad' (fun x -> Nx.sum (Rune.jit' f x)) x)
    | Jit_of_map -> ignore (Rune.jit' f x)
    | Grad_of_map -> ignore (Rune.grad' (fun x -> Nx.sum (f x)) x)
    | Jvp_of_map -> ignore (Rune.jvp' f x x)
  in
  let split = if n >= 2 && n mod 2 = 0 then 2 else 1 in
  let grid () = Nx.reshape [| split; n / split |] (v c.lanes) in
  let mapped () =
    match c.lmode with
    | Map_of_map -> under (Rune.vmap' (Rune.vmap' (inside body))) (grid ())
    | Map_grad_map ->
        (* The backward pass of each outer lane's inner map. *)
        under (Rune.vmap' (inside (Rune.vmap' body))) (grid ())
    | _ -> under (Rune.vmap' (inside body)) (v c.lanes)
  in
  let collected f = bits (snd (Total.collect t ~init:(s c.start) f)) in
  let expected = collected (fun () -> ordered t (rows c c.lanes)) in
  cover "lane by lane rounds otherwise"
    (collected (fun () -> lane_major t c) <> expected);
  cover "a lane's sum from the identity rounds otherwise"
    (collected (fun () -> from_identity t c) <> expected);
  let before = Rune.evaluations () in
  equal bits_t expected (collected mapped);
  let compiled = Rune.evaluations () = before in
  cover "jit of vmap compiled" (c.lmode = Jit_of_map && compiled);
  if n = 1 then
    equal ~msg:"one lane, without the map" bits_t expected
      (collected (fun () -> under (inside body) (s c.lanes.(0))));
  (* A collect between two maps receives the inner map's lanes of its row. *)
  if c.lmode = Map_of_map then begin
    let per_row =
      Array.init split (fun r ->
          let row = Array.sub c.lanes (r * (n / split)) (n / split) in
          collected (fun () -> ordered t (rows c row)))
    in
    let between =
      Rune.vmap'
        (fun row ->
          snd
            (Total.collect t ~init:(s c.start) (fun () ->
                 ignore (Rune.vmap' body row))))
        (grid ())
    in
    let each = Array.concat (Array.to_list (Array.map snd per_row)) in
    equal ~msg:"a collect between the maps" bits_t ([| split |], each)
      (bits between)
  end

(* A jit called again under a collect whose running value lies elsewhere than at
   its first call chains its additions as eagerly: the program takes the running
   value where the call finds it. *)
let test_jit_running () =
  let t = Total.make () in
  let g =
    Rune.jit' (fun x ->
        Total.add t (Nx.scalar f32 12.);
        Total.add t (Nx.place Nx.Host.on (Nx.scalar f32 12.));
        x)
  in
  let x = v [| 1. |] in
  let total f =
    snd
      (Total.collect t ~init:(Nx.scalar f32 1e8) (fun () ->
           f ();
           f ()))
  in
  let twelve () = Nx.scalar f32 12. in
  let eager () =
    Total.add t (twelve ());
    Total.add t (Nx.place Nx.Host.on (twelve ()))
  in
  equal bits_t (bits (total eager)) (bits (total (fun () -> ignore (g x))))

(* One jit called outside a map, under one, and outside again: each call adds in
   its own order, from 1 over [1e8; -1e8] at f32, which gives 0 eagerly and 1
   summed from the identity first. *)
let test_jit_mapped () =
  let t = Total.make () in
  let g =
    Rune.jit' (fun x ->
        Total.add t (Nx.mul_s x 1e8);
        Total.add t (Nx.mul_s x (-1e8));
        x)
  in
  let total f = bits (snd (Total.collect t ~init:(s 1.) f)) in
  let eager =
    total (fun () ->
        Total.add t (s 1e8);
        Total.add t (s (-1e8)))
  in
  let outside () = total (fun () -> ignore (g (s 1.))) in
  let mapped () = total (fun () -> ignore (Rune.vmap' g (v [| 1. |]))) in
  equal ~msg:"eager" bits_t (bits (s 0.)) eager;
  equal ~msg:"outside" bits_t eager (outside ());
  equal ~msg:"under a map" bits_t eager (mapped ());
  equal ~msg:"outside again" bits_t eager (outside ());
  equal ~msg:"under a map again" bits_t eager (mapped ())

(* A remat inside a map, under reverse mode: each lane adds [x] once, so over
   [1; 2] from 0 the total is 3. *)
let test_remat_mapped () =
  let t = Total.make () in
  let body x =
    Rune.grad'
      (fun x ->
        Nx.sum
          (Rune.remat
             Nx.Ptree.(tensor @-> returns tensor)
             (fun x ->
               Total.add t x;
               Nx.mul x x)
             x))
      x
  in
  let _, r =
    Total.collect t ~init:(s 0.) (fun () ->
        ignore (Rune.vmap' body (v [| 1.; 2. |])))
  in
  equal bits_t (bits (s 3.)) (bits r)

(* One jit under two maps, the collect around both and then between them: each
   lane adds [x] once, so over [[1; 2]; [3; 4]] the totals are 10, and 3 and 7
   for the outer lanes. *)
let test_jit_between_maps () =
  let t = Total.make () in
  let g =
    Rune.jit' (fun x ->
        Total.add t x;
        x)
  in
  let xs = Nx.reshape [| 2; 2 |] (v [| 1.; 2.; 3.; 4. |]) in
  let around () =
    snd
      (Total.collect t ~init:(s 0.) (fun () ->
           ignore (Rune.vmap' (Rune.vmap' g) xs)))
  in
  let between () =
    Rune.vmap'
      (fun row ->
        snd (Total.collect t ~init:(s 0.) (fun () -> ignore (Rune.vmap' g row))))
      xs
  in
  equal ~msg:"around" bits_t (bits (s 10.)) (bits (around ()));
  equal ~msg:"between" bits_t (bits (v [| 3.; 7. |])) (bits (between ()));
  equal ~msg:"around again" bits_t (bits (s 10.)) (bits (around ()))

(* A collect inside a differentiation holds its additions' tangents, as the same
   additions written out give them: from an init of no tangent, an addition's
   tangent of -0 stays -0. *)
let test_tangent_sign () =
  let t = Total.make () in
  let collected x =
    snd (Total.collect t ~init:(s 0.) (fun () -> Total.add t x))
  in
  let written x = Nx.add (s 0.) x in
  let tangent f = bits (snd (Rune.jvp' f (s 1.) (s (-0.)))) in
  equal ~msg:"eagerly" bits_t (tangent written) (tangent collected);
  equal ~msg:"under jit" bits_t (tangent written)
    (tangent (Rune.jit' collected))

let () =
  exit
    (run "rune2 total"
       [
         test "collect sums each addition once" test_collect;
         test "the innermost scope of a total" test_same_total;
         test "a scope inside a jit" test_scope_in_jit;
         prop "jit adds what eager adds" additions law_additions;
         test "a jit that raises adds nothing" test_raising_jit;
         test "an addition of another shape raises" test_shape_refused;
         test "a top-level total serves every set" test_sets;
         test "an addition on another set raises" test_set_refused;
         test "an exception passes through collect" test_raises;
         test "a construct whose delivery raises delivers nothing"
           test_delivery_raises;
         test "a jit adds once per call" test_jit_scope;
         prop ~examples:chained_examples
           "additions chain wherever they are made" chained law_chained;
         test "a jit takes the running value where it lies" test_jit_running;
         test "a jit adds in the order of each call's place" test_jit_mapped;
         test "a total's tangent keeps its sign" test_tangent_sign;
         test "a remat under a map adds once under reverse mode"
           test_remat_mapped;
         test "a jit adds once under maps a collect lies between"
           test_jit_between_maps;
         prop ~examples:in_turn_examples "a map's lanes add in turn" in_turn
           law_in_turn;
       ])
