(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.callback: its function runs on values once per execution, in eager
   order, at a scan's rows, an iterate's trips and tests and a switch's taken
   branch, and never at a trace. *)

open Windtrap
module Rune = Rune2
module T = Nx.Ptree

let f64 = Nx.float64
let vec a = Nx.create f64 [| Array.length a |] a
let scalar x = Nx.scalar f64 x
let floats x = Array.to_list (Nx.to_array (Nx.copy x))
let tt = T.(tensor @-> returns tensor)

(* A callback of [g] that logs the values of each argument it receives. *)
let logged g =
  let log = ref [] in
  let cb x =
    Rune.callback tt ~like:x
      (fun v ->
        log := floats v :: !log;
        g v)
      x
  in
  (cb, fun () -> List.rev !log)

let values = Gen.(list ~size:(int_range 0 5) (float_range (-4.) 4.))
let fl = Testable.(list (list float_exact))

(* Eager: once, at the call, its function's result. *)
let law_eager xs =
  let x = vec (Array.of_list xs) in
  cover "no element" (xs = []);
  let cb, log = logged (fun v -> Nx.mul v v) in
  let y = cb x in
  equal ~msg:"result" Testable.(list float_exact) (floats (Nx.mul x x)) (floats y);
  equal ~msg:"calls" fl [ xs ] (log ())

(* A scan's step: once per row, in order; zero rows, never. *)
let law_scan xs =
  let n = List.length xs in
  cover "no row" (n = 0);
  cover "rows" (n > 1);
  let cb, log = logged (fun v -> Nx.sin v) in
  let f c x =
    let y = cb x in
    (Nx.add c y, y)
  in
  let c, ys =
    Rune.scan T.tensor T.tensor T.tensor ~f ~init:(scalar 0.)
      (vec (Array.of_list xs))
  in
  let rows = List.map (fun x -> [ x ]) xs in
  equal ~msg:"calls" fl rows (log ());
  let fold = List.fold_left (fun c x -> c +. sin x) 0. xs in
  equal ~msg:"carry" (Testable.float 1e-12) fold (Nx.item [] c);
  let eager = floats (Nx.sin (vec (Array.of_list xs))) in
  equal ~msg:"outputs" Testable.(list float_exact) eager (floats ys)

(* An iterate: once per trip in the step and once per test in [until]. *)
let law_iterate k =
  cover "no trip" (k = 0);
  let steps, slog = logged (fun v -> Nx.add_s v 1.) in
  let tests, ulog = logged Fun.id in
  let until c = Nx.greater_equal (tests c) (scalar (float_of_int k)) in
  let r = Rune.iterate T.tensor ~max:10 ~until ~f:steps (scalar 0.) in
  equal ~msg:"result" Testable.float_exact (float_of_int k) (Nx.item [] r);
  let upto n = List.init n (fun i -> [ float_of_int i ]) in
  equal ~msg:"steps" fl (upto k) (slog ());
  equal ~msg:"tests" fl (upto (k + 1)) (ulog ())

(* A switch: the taken branch only. *)
let law_switch k =
  let cbs = List.init 3 (fun j -> logged (fun v -> Nx.add_s v (float_of_int j))) in
  let x = scalar 1. in
  let y =
    Rune.switch T.tensor (Nx.scalar Nx.int32 (Int32.of_int k))
      (List.map (fun (cb, _) () -> cb x) cbs)
  in
  equal ~msg:"result" Testable.float_exact (1. +. float_of_int k) (Nx.item [] y);
  List.iteri
    (fun j (_, log) ->
      equal ~msg:(Printf.sprintf "branch %d" j) Testable.int
        (if j = k then 1 else 0)
        (List.length (log ())))
    cbs


(* Under jit: once per execution, never at the trace. *)
let test_jit () =
  let cb, log = logged (fun v -> Nx.exp v) in
  let g = Rune.jit tt (fun x -> Nx.add_s (cb (Nx.mul_s x 2.)) 1.) in
  let y1 = g (scalar 0.5) and y2 = g (scalar 1.) in
  equal ~msg:"calls" fl [ [ 1. ]; [ 2. ] ] (log ());
  equal ~msg:"first" (Testable.float 1e-12) (exp 1. +. 1.) (Nx.item [] y1);
  equal ~msg:"second" (Testable.float 1e-12) (exp 2. +. 1.) (Nx.item [] y2)

(* Under vmap: once per lane, in lane order, on the lane's rows; with [~lanes],
   its function once on every lane. *)
let law_vmap xs =
  let n = List.length xs in
  cover "no lane" (n = 0);
  cover "lanes" (n > 1);
  let x = vec (Array.of_list xs) in
  let cb, log = logged (fun v -> Nx.mul_s v 3.) in
  let y = Rune.vmap tt cb x in
  equal ~msg:"per lane" fl (List.map (fun x -> [ x ]) xs) (log ());
  equal ~msg:"results" Testable.(list float_exact) (floats (Nx.mul_s x 3.))
    (floats y);
  let each = ref 0 and all = ref [] in
  let g v =
    all := floats v :: !all;
    Nx.mul_s v 3.
  in
  let y =
    Rune.vmap tt
      (fun x ->
        Rune.callback ~lanes:g tt ~like:x
          (fun v ->
            incr each;
            v)
          x)
      x
  in
  equal ~msg:"lanes" fl [ xs ] (List.rev !all);
  equal ~msg:"its function" Testable.int 0 !each;
  equal ~msg:"lanes' results" Testable.(list float_exact)
    (floats (Nx.mul_s x 3.)) (floats y);
  (* An argument the map does not hold is the same at every lane. *)
  let cb, log = logged Nx.neg in
  let _ = Rune.vmap tt (fun _ -> cb (scalar 2.)) x in
  equal ~msg:"an argument the map does not hold" fl
    (List.map (fun _ -> [ 2. ]) xs)
    (log ())

(* Nested maps: outer lanes major, per lane and with [~lanes]. *)
let test_nested () =
  let x = Nx.create f64 [| 2; 3 |] [| 0.; 1.; 2.; 3.; 4.; 5. |] in
  let cb, log = logged Nx.neg in
  let y = Rune.vmap tt (Rune.vmap tt cb) x in
  equal ~msg:"per lane" fl
    (List.init 6 (fun k -> [ float_of_int k ]))
    (log ());
  equal ~msg:"results" Testable.(list float_exact) (floats (Nx.neg x))
    (floats y);
  let shapes = ref [] in
  let g v =
    shapes := (Nx.shape v, floats v) :: !shapes;
    Nx.neg v
  in
  let inner x = Rune.callback ~lanes:g tt ~like:x Fun.id x in
  let y = Rune.vmap tt (Rune.vmap tt inner) x in
  equal ~msg:"lanes"
    Testable.(list (pair (array int) (list float_exact)))
    [ ([| 6 |], floats (Nx.reshape [| 6 |] x)) ]
    (List.rev !shapes);
  equal ~msg:"lanes' results" Testable.(list float_exact) (floats (Nx.neg x))
    (floats y)

(* With [~lanes], a held lane receives its donor's rows, and an argument the
   map does not hold is broadcast; a pair of results splits. *)
let test_lanes_more () =
  let rows = ref [] in
  let g (c, b) =
    rows := (floats c, floats b) :: !rows;
    (Nx.add_s c 1., Nx.sum ~axes:[ 1 ] b)
  in
  let pp = T.(pair tensor tensor @-> returns (pair tensor tensor)) in
  let step b c =
    fst (Rune.callback ~lanes:g pp ~like:(c, scalar 0.) (fun _ -> assert false) (c, b))
  in
  let loop k =
    Rune.iterate T.tensor ~max:5
      ~until:(fun c -> Nx.greater_equal c k)
      ~f:(step (vec [| 1.; 2. |]))
      (Nx.zeros_like k)
  in
  let r = Rune.vmap tt loop (vec [| 1.; 2. |]) in
  equal ~msg:"results" Testable.(list float_exact) [ 1.; 2. ] (floats r);
  let fl2 = Testable.(list (pair (list float_exact) (list float_exact))) in
  equal ~msg:"rows" fl2
    [ ([ 0.; 0. ], [ 1.; 2.; 1.; 2. ]); ([ 1.; 1. ], [ 1.; 2.; 1.; 2. ]) ]
    (List.rev !rows)

(* A scan, and an iterate over no lane value, inside a map: each lane runs its
   own calls. *)
let test_map_loops () =
  let cb, log = logged (fun v -> Nx.add_s v 1.) in
  let xs = vec [| 1.; 2. |] in
  let _ =
    Rune.vmap tt
      (fun x ->
        fst
          (Rune.scan T.tensor T.tensor T.tensor
             ~f:(fun c r -> (cb (Nx.add c r), r))
             ~init:x (vec [| 10.; 20. |])))
      xs
  in
  equal ~msg:"a scan" fl [ [ 11. ]; [ 12. ]; [ 32. ]; [ 33. ] ] (log ());
  let cb, log = logged (fun v -> Nx.add_s v 1.) in
  let r =
    Rune.vmap tt
      (fun x ->
        Nx.add x
          (Rune.iterate T.tensor ~max:5
             ~until:(fun c -> Nx.greater_equal c (scalar 2.))
             ~f:cb (scalar 0.)))
      xs
  in
  equal ~msg:"an iterate" Testable.(list float_exact) [ 3.; 4. ] (floats r);
  equal ~msg:"an iterate's calls" Testable.int 4 (List.length (log ()))

(* Lanes whose iterate stops apart: a held lane's step calls nothing. *)
let law_held ks =
  cover "lanes stopping apart" (List.length (List.sort_uniq compare ks) > 1);
  let cb, log = logged (fun v -> Nx.add_s v 1.) in
  let loop k =
    Rune.iterate T.tensor ~max:5
      ~until:(fun c -> Nx.greater_equal c k)
      ~f:cb (Nx.zeros_like k)
  in
  let k = vec (Array.of_list (List.map float_of_int ks)) in
  let r = Rune.vmap tt loop k in
  equal ~msg:"results" Testable.(list float_exact) (floats k) (floats r);
  let top = List.fold_left max 0 ks in
  let expected =
    List.concat
      (List.init top (fun t ->
           List.filter_map
             (fun k -> if k > t then Some [ float_of_int t ] else None)
             ks))
  in
  equal ~msg:"calls" fl expected (log ())

(* Derivatives: a tracked argument reaching a float result raises; a custom
   rule around the call gives the derivative. *)
let test_refusals () =
  let x = scalar 2. in
  let cb x = Rune.callback tt ~like:x Nx.sin x in
  let refused name =
    Invalid_argument
      ("Rune.callback: " ^ name
     ^ " tracks an argument of a call with a float result; a custom rule \
        around the call gives its derivative")
  in
  raises ~msg:"grad" (refused "Rune.grad'") (fun () -> Rune.grad' cb x);
  raises ~msg:"jvp" (refused "Rune.jvp'") (fun () -> Rune.jvp' cb x x);
  raises ~msg:"vjp" (refused "Rune.vjp'") (fun () -> Rune.vjp' cb x);
  raises ~msg:"linearize" (refused "Rune.linearize") (fun () ->
      Rune.linearize T.tensor T.tensor cb x);
  raises ~msg:"linear_transpose" (refused "Rune.linear_transpose") (fun () ->
      Rune.linear_transpose T.tensor T.tensor cb x x);
  (* A value the differentiation tracks, captured, is no argument: the
     function runs outside the differentiation, where using it raises. *)
  let captured x =
    Rune.callback tt ~like:x (fun v -> Nx.mul v x) (scalar 1.)
  in
  raises ~msg:"a captured tracked value"
    (Invalid_argument "Nx.mul: a value of Rune.grad', used outside Rune.grad''s extent") (fun () ->
      Rune.grad' captured x);
  raises ~msg:"in a scan" (refused "Rune.grad'") (fun () ->
      Rune.grad'
        (fun x ->
          fst
            (Rune.scan T.tensor T.tensor T.tensor
               ~f:(fun c r -> (Nx.add c (cb r), r))
               ~init:x (Nx.reshape [| 1 |] x)))
        x);
  (* An untracked argument computes a constant, once. *)
  let cb, log = logged Nx.sin in
  let g = Rune.grad' (fun x -> Nx.mul x (cb (scalar 1.))) x in
  equal ~msg:"untracked" (Testable.float 1e-15) (sin 1.) (Nx.item [] g);
  equal ~msg:"untracked calls" fl [ [ 1. ] ] (log ());
  (* A result with no derivative is no refusal. *)
  let int_like = Nx.scalar Nx.int32 0l in
  let s = T.(tensor @-> returns tensor) in
  let g =
    Rune.grad'
      (fun x ->
        let k = Rune.callback s ~like:int_like (fun v -> Nx.cast Nx.int32 v) x in
        Nx.mul x (Nx.cast f64 k))
      x
  in
  equal ~msg:"an integer result" Testable.float_exact 2. (Nx.item [] g)

(* A cube computed by a callback, whose pullback calls an adjoint through
   another: the pattern of a density whose gradient an external solver
   computes. *)
let cube () =
  let fwd, flog = logged (fun v -> Nx.mul v (Nx.mul v v)) in
  let adj = ref 0 in
  let adjoint (x, ct) =
    incr adj;
    Nx.mul ct (Nx.mul_s (Nx.mul x x) 3.)
  in
  let p = T.(pair tensor tensor @-> returns tensor) in
  let f =
    Rune.custom_vjp T.tensor T.tensor (fun x ->
        (fwd x, fun ct -> Rune.callback p ~like:x adjoint (x, ct)))
  in
  (f, flog, adj)

let test_rule () =
  let f, flog, adj = cube () in
  let x = scalar 1.5 in
  let g = Rune.grad' f x in
  equal ~msg:"gradient" (Testable.float 1e-12) (3. *. 1.5 *. 1.5) (Nx.item [] g);
  equal ~msg:"primal calls" fl [ [ 1.5 ] ] (flog ());
  equal ~msg:"adjoint calls" Testable.int 1 !adj;
  (* Compiled, the rule's primal runs once per execution, its derivative's
     included. *)
  let f, flog, adj = cube () in
  let g = Rune.grad' (Rune.jit' f) x in
  equal ~msg:"compiled gradient" (Testable.float 1e-12) 6.75 (Nx.item [] g);
  equal ~msg:"compiled primal calls" fl [ [ 1.5 ] ] (flog ());
  equal ~msg:"compiled adjoint calls" Testable.int 1 !adj;
  raises ~msg:"second order"
    (Invalid_argument
       "Rune.callback: Rune.grad' tracks an argument of a call with a float \
        result; a custom rule around the call gives its derivative")
    (fun () -> Rune.grad' (Rune.grad' f) x)

(* The rule inside a scan, under grad: once per row forward and once per row
   in the backward pass. *)
let law_rule_scan xs =
  let n = List.length xs in
  cover "no row" (n = 0);
  cover "rows" (n > 1);
  let f, flog, adj = cube () in
  let loss xs =
    fst
      (Rune.scan T.tensor T.tensor T.tensor
         ~f:(fun c r -> (Nx.add c (f r), r))
         ~init:(scalar 0.) xs)
  in
  let g = Rune.grad' loss (vec (Array.of_list xs)) in
  equal ~msg:"gradient"
    Testable.(list (float 1e-12))
    (List.map (fun x -> 3. *. x *. x) xs)
    (floats g);
  equal ~msg:"primal calls" fl (List.map (fun x -> [ x ]) xs) (flog ());
  equal ~msg:"adjoint calls" Testable.int n !adj

(* Under reverse mode a remat keeps a callback's results: nothing runs again
   in the backward pass. *)
let test_remat () =
  let x = scalar 1.5 in
  let cb, log = logged Nx.cos in
  let body x = Nx.mul (Nx.sin x) (cb (scalar 1.)) in
  let g = Rune.grad' (Rune.remat tt body) x in
  equal ~msg:"gradient" (Testable.float 1e-12) (cos 1.5 *. cos 1.) (Nx.item [] g);
  equal ~msg:"calls" fl [ [ 1. ] ] (log ());
  let f, flog, adj = cube () in
  let g = Rune.grad' (Rune.remat tt (fun x -> Nx.sin (f x))) x in
  let c = 1.5 *. 1.5 *. 1.5 in
  equal ~msg:"rule's gradient" (Testable.float 1e-12)
    (cos c *. 3. *. 1.5 *. 1.5) (Nx.item [] g);
  equal ~msg:"rule's primal calls" fl [ [ 1.5 ] ] (flog ());
  equal ~msg:"rule's adjoint calls" Testable.int 1 !adj

(* Memory *)

module Memory = Rune2_support.Memory

(* Under reverse mode a remat keeps its arguments and its callbacks' results
   and computes the rest again, so a pullback holds the result's bytes and the
   callback's. A callback inside a loop the derivative splits keeps the loop's
   values instead, and still runs once. *)
let test_remat_memory () =
  let n = 4096 and f32 = Nx.float32 in
  let tt = T.(tensor @-> returns tensor) in
  let x = Nx.place Memory.On.on (Nx.ones f32 [| n |]) in
  let calls = ref 0 in
  (* A rule whose pullback keeps no array: the remat keeps its result alone. *)
  let twice ~host x =
    if host then
      Rune.callback tt ~like:x
        (fun v ->
          incr calls;
          Nx.mul_s v 2.)
        x
    else Nx.mul_s x 2.
  in
  let double ~host x = (twice ~host x, fun ct -> twice ~host:false ct) in
  (* Its pullback a callback too, which only the backward pass runs. *)
  let both x = (twice ~host:true x, fun ct -> twice ~host:true ct) in
  let body ~host x =
    let y = Rune.custom_vjp T.tensor T.tensor (double ~host) x in
    Nx.mul (Nx.sin y) (Nx.exp y)
  in
  let kept = ref [] in
  let held f =
    let r = Memory.held (fun () -> kept := Rune.vjp' f x :: !kept) in
    ignore (Sys.opaque_identity !kept);
    kept := [];
    r
  in
  let remat ~host = Rune.remat tt (body ~host) in
  equal ~msg:"no callback" Testable.int (4 * n) (held (remat ~host:false));
  equal ~msg:"a callback" Testable.int (2 * 4 * n) (held (remat ~host:true));
  equal ~msg:"calls" Testable.int 2 !calls;
  let pulled x =
    let y = Rune.custom_vjp T.tensor T.tensor both x in
    Nx.mul (Nx.sin y) (Nx.exp y)
  in
  equal ~msg:"a pullback's callback" Testable.int (2 * 4 * n)
    (held (Rune.remat tt pulled));
  let ct = Nx.place Memory.On.on (Nx.ones f32 [| n |]) in
  calls := 0;
  let y, pb = Rune.vjp' (remat ~host:true) x in
  let g = pb ct in
  let y0, pb0 = Rune.vjp' (body ~host:false) x in
  equal ~msg:"result" Testable.(array (float_rel ~rel:1e-6 ~abs:0.))
    (Nx.to_array y0) (Nx.to_array y);
  equal ~msg:"gradient" Testable.(array (float_rel ~rel:1e-6 ~abs:0.))
    (Nx.to_array (pb0 ct)) (Nx.to_array g);
  equal ~msg:"calls with the backward pass" Testable.int 1 !calls;
  calls := 0;
  let loop x =
    fst
      (Rune.scan T.tensor T.tensor T.tensor
         ~f:(fun c r -> (Nx.add c (body ~host:true r), r))
         ~init:(Nx.zeros_like x) (Nx.reshape [| 1; n |] x))
  in
  let g = Rune.grad' (fun x -> Nx.sum (Rune.remat tt loop x)) x in
  let g0 = Rune.grad' (fun x -> Nx.sum (body ~host:false x)) x in
  equal ~msg:"a loop's gradient" Testable.(array (float_rel ~rel:1e-6 ~abs:0.))
    (Nx.to_array g0) (Nx.to_array g);
  equal ~msg:"a loop's calls" Testable.int 1 !calls

(* The results' forms are [like]'s, which is never read. *)
let test_like () =
  let x = vec [| 1.; 2. |] in
  raises (Invalid_argument "Rune.callback: the root: shape [3], where like has [2]")
    (fun () -> ignore (Rune.callback tt ~like:x (fun _ -> vec [| 1.; 2.; 3. |]) x));
  let like = Rune.eval_shape T.tensor T.tensor Nx.neg x in
  let y = Rune.callback tt ~like Nx.neg x in
  equal ~msg:"a stand-in like" Testable.(list float_exact) [ -1.; -2. ]
    (floats y)

(* A call whose results nothing reads still runs: here in a residual of a
   custom rule that no derivative reads. *)
let test_unread () =
  let cb, log = logged Fun.id in
  let rule x =
    let _, r =
      Rune.scan T.tensor T.tensor T.tensor
        ~f:(fun c r -> (c, cb r))
        ~init:x (Nx.reshape [| 1 |] x)
    in
    (Nx.mul_s x 2., fun ct -> Nx.mul ct (Nx.sum r))
  in
  let step c x = (c, Rune.custom_vjp T.tensor T.tensor rule x) in
  let _ =
    Rune.scan T.tensor T.tensor T.tensor ~f:step ~init:(scalar 0.)
      (vec [| 1.; 2.; 3. |])
  in
  equal ~msg:"calls" fl [ [ 1. ]; [ 2. ]; [ 3. ] ] (log ())

(* The function runs outside every transformation: its own collects work, also
   where it runs at a trace. *)
let test_outside () =
  let p = Rune.Probe.make T.tensor in
  let count v =
    let _, l = Rune.Probe.collect p (fun () -> Rune.Probe.emit p v) in
    scalar (float_of_int (List.length l))
  in
  let x = scalar 5. in
  let _, ys =
    Rune.scan T.tensor T.tensor T.tensor
      ~f:(fun c _ -> (c, Rune.callback tt ~like:x count x))
      ~init:x (vec [| 1.; 2. |])
  in
  equal ~msg:"offers" Testable.(list float_exact) [ 1.; 1. ] (floats ys)

exception Mine of int

(* Exceptions pass through unchanged; additions, emissions and draws without a
   key raise. *)
let test_raise () =
  let x = scalar 1. in
  let boom _ = raise (Mine 3) in
  raises ~msg:"eager" (Mine 3) (fun () -> ignore (Rune.callback tt ~like:x boom x));
  raises ~msg:"in a scan" (Mine 3) (fun () ->
      ignore
        (Rune.scan T.tensor T.tensor T.tensor
           ~f:(fun c r -> (c, Rune.callback tt ~like:r boom r))
           ~init:x (vec [| 1.; 2. |])));
  let t = Rune.Total.make () in
  raises (Invalid_argument "Rune.callback: the function adds to a total collected around the call")
    (fun () ->
      ignore
        (Rune.Total.collect t ~init:x (fun () ->
             Rune.callback tt ~like:x
               (fun v ->
                 Rune.Total.add t v;
                 v)
               x)));
  (* A collect inside the function is the function's own. *)
  let y =
    Rune.callback tt ~like:x
      (fun v -> snd (Rune.Total.collect t ~init:x (fun () -> Rune.Total.add t v)))
      x
  in
  equal ~msg:"an inner collect" Testable.float_exact 2. (Nx.item [] y);
  let p = Rune.Probe.make T.tensor in
  let r, _ =
    Rune.Probe.collect p (fun () ->
        Rune.callback tt ~like:x
          (fun v ->
            Rune.Probe.emit p v;
            v)
          x)
  in
  raises
    (Invalid_argument
       "Rune.callback: the function emits to a probe collected around the call")
    (fun () -> match r with Ok _ -> () | Error (e, _) -> raise e);
  raises (Invalid_argument "Nx.Rng.uniform: no key inside Rune.callback, which refuses keyless draws")
    (fun () ->
      ignore (Rune.callback tt ~like:x (fun _ -> Nx.Rng.uniform f64 [||]) x))

let () =
  exit
    (run "rune2 callback"
       [
         group "runs"
           [
             prop "eager, once at the call" values law_eager;
             prop "a scan, once per row" values law_scan;
             prop "an iterate, once per trip and test" Gen.(int_range 0 4)
               law_iterate;
             prop "a switch, its taken branch" Gen.(int_range 0 2) law_switch;
             test "a jit, once per execution" test_jit;
           ];
         group "maps"
           [
             prop "per lane, and with lanes" values law_vmap;
             test "nested maps" test_nested;
             test "lanes held, broadcast and paired" test_lanes_more;
             test "loops inside a map" test_map_loops;
             prop "held lanes call nothing" Gen.(list ~size:(int_range 1 4) (int_range 0 3)) law_held;
           ];
         group "derivatives"
           [
             test "refusals" test_refusals;
             test "a custom rule around it" test_rule;
             prop "a custom rule in a scan, under grad" values law_rule_scan;
             test "a remat keeps the results" test_remat;
             test "a remat keeps the results alone" test_remat_memory;
           ];
         group "interface"
           [
             test "results like like" test_like;
             test "raises" test_raise;
             test "results nothing reads" test_unread;
             test "outside every transformation" test_outside;
           ];
       ])
