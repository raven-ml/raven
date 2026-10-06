(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.scan: a scan is its loop, run where it is written, under every
   transformation, scope and key around it; a compiled function stages it as one
   loop or writes it out. The trusted side is an OCaml loop over the rows, or
   the same computation through Nx.cumsum. *)

open Windtrap

let f64 = Nx.float64
let vec a = Nx.create f64 [| Array.length a |] a
let scalar x = Nx.scalar f64 x
let exact () = Oracle.tensor ()
let close () = Oracle.tensor ~rel:1e-12 ()
let v4 () = vec [| 0.5; -1.2; 2.1; 0.8 |]

let rows () =
  Nx.create f64 [| 2; 4 |] [| 0.5; -1.2; 2.1; 0.8; 1.7; -0.4; 0.9; 0.2 |]

let running_sum xs =
  snd
    (Rune.scan'
       ~f:(fun c x ->
         let c = Nx.add c x in
         (c, c))
       ~init:(scalar 0.) xs)

(* The loop a scan of [f] is: [f] on each row of [xs] in order, the outputs
   stacked. *)
let loop f init xs =
  let n = (Nx.shape xs).(0) in
  let c = ref init and ys = ref [] in
  for i = 0 to n - 1 do
    let c', y = f !c (Nx.get [ i ] xs) in
    c := c';
    ys := y :: !ys
  done;
  (!c, Nx.stack (List.rev !ys))

(* A step from two programs: the next carry is [p (c + x)], the output [q c]. *)
let step p q c x = (Expr.eval p (Nx.add c x), Expr.eval q c)

(* The fold *)

let fold_tests =
  [
    prop "a scan is its loop"
      Gen.(
        let* n = int_range 1 5 in
        triple (pair Expr.gen Expr.gen) Expr.point (Expr.points n))
      (fun ((p, q), init, xs) ->
        let c, ys = Rune.scan' ~f:(step p q) ~init xs in
        let c', ys' = loop (step p q) init xs in
        equal ~msg:"carry" (exact ()) c' c;
        equal ~msg:"outputs" (exact ()) ys' ys);
    test "a running sum is a cumulative sum" (fun () ->
        equal (close ()) (Nx.cumsum (v4 ())) (running_sum (v4 ())));
    test "the result's carry is the last step's" (fun () ->
        let c, _ =
          Rune.scan' ~f:(fun c x -> (Nx.add c x, c)) ~init:(scalar 0.) (v4 ())
        in
        equal (close ()) (scalar (0.5 -. 1.2 +. 2.1 +. 0.8)) c);
    test "a structured carry, rows and outputs" (fun () ->
        let xs = (v4 (), Nx.mul_s (v4 ()) 2.) in
        let (sum, count), ys =
          Rune.scan
            Nx.Ptree.(pair tensor tensor)
            Nx.Ptree.(pair tensor tensor)
            Nx.Ptree.(list tensor)
            ~f:(fun (sum, count) (a, b) ->
              let sum = Nx.add sum (Nx.add a b) in
              ((sum, Nx.add_s count 1.), [ sum; a ]))
            ~init:(scalar 0., scalar 0.)
            xs
        in
        equal ~msg:"sum" (close ())
          (scalar (3. *. (0.5 -. 1.2 +. 2.1 +. 0.8)))
          sum;
        equal ~msg:"count" (exact ()) (scalar 4.) count;
        match ys with
        | [ sums; firsts ] ->
            equal ~msg:"sums" (close ()) (Nx.cumsum (Nx.mul_s (v4 ()) 3.)) sums;
            equal ~msg:"rows" (exact ()) (v4 ()) firsts
        | _ -> fail "the outputs are a list of two tensors");
    test "a fold with nothing to emit returns unit" (fun () ->
        let c, () =
          Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor Nx.Ptree.unit
            ~f:(fun c x -> (Nx.add c x, ()))
            ~init:(scalar 0.) (v4 ())
        in
        equal (close ()) (scalar (0.5 -. 1.2 +. 2.1 +. 0.8)) c);
    test "one row gives outputs of leading length one" (fun () ->
        let c, ys =
          Rune.scan'
            ~f:(fun c x -> (Nx.mul c x, Nx.add c x))
            ~init:(scalar 2.) (vec [| 3. |])
        in
        equal ~msg:"carry" (exact ()) (scalar 6.) c;
        equal ~msg:"outputs" (exact ()) (vec [| 5. |]) ys);
    test "an empty axis is refused" (fun () ->
        raises (Invalid_argument "Rune.scan': xs is empty along the scan axis")
          (fun () -> running_sum (vec [||])));
    test "outputs are stacked in row order" (fun () ->
        let _, ys =
          Rune.scan'
            ~f:(fun c x -> (Nx.add_s c 1., Nx.mul c x))
            ~init:(scalar 0.)
            (vec [| 1.; 1.; 1.; 1. |])
        in
        equal (exact ()) (vec [| 0.; 1.; 2.; 3. |]) ys);
    test "a carry may change its shape between steps" (fun () ->
        let grow c x =
          (Nx.concatenate ~axis:0 [ c; Nx.reshape [| 1 |] x ], ())
        in
        let c, () =
          Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor Nx.Ptree.unit ~f:grow
            ~init:(vec [||]) (v4 ())
        in
        equal (exact ()) (v4 ()) c);
  ]

(* Each transformation runs the step once per row *)

let counting () =
  let runs = ref 0 in
  let f x =
    fst
      (Rune.scan'
         ~f:(fun c x ->
           incr runs;
           (Nx.add c (Nx.mul x x), c))
         ~init:(scalar 0.) x)
  in
  (runs, f)

let runs_tests =
  let under name transform =
    test
      ("under " ^ name ^ " the step runs once per row")
      (fun () ->
        let runs, f = counting () in
        transform f;
        equal int 4 !runs)
  in
  [
    under "no transformation" (fun f -> ignore (f (v4 ())));
    under "jvp" (fun f -> ignore (Rune.jvp' f (v4 ()) (v4 ())));
    under "grad" (fun f -> ignore (Rune.grad' f (v4 ())));
    under "vmap" (fun f ->
        ignore (Rune.vmap' f (Nx.reshape [| 1; 4 |] (v4 ()))));
  ]

(* Refusals *)

let changed_length =
  "Rune.scan: the root: length 2 in the carry the step returned, length 1 in \
   the carry it received"

let grow c x = (c @ [ x ], ())

let scan_growing xs =
  let c, () =
    Rune.scan
      Nx.Ptree.(list tensor)
      Nx.Ptree.tensor Nx.Ptree.unit ~f:grow
      ~init:[ scalar 0. ]
      xs
  in
  Nx.sum (List.hd c)

module Packed = struct
  type _ t = Nx.packed

  let walk c (Nx.P x) = Nx.P (Nx.Ptree.Walk.tensor c x)
end

let refusal_cases =
  [
    ( "a carry of another length",
      changed_length,
      fun () -> ignore (scan_growing (v4 ())) );
    ( "a carry of another dtype",
      "Rune.scan: the root: float32 in the carry the step returned, float64 in \
       the carry it received",
      fun () ->
        ignore
          (Rune.scan
             (Nx.Ptree.instantiate (module Packed))
             Nx.Ptree.tensor Nx.Ptree.unit
             ~f:(fun (Nx.P c) _ -> (Nx.P (Nx.cast Nx.float32 c), ()))
             ~init:(Nx.P (scalar 0.))
             (v4 ())) );
    ( "a carry that loses its option",
      "Rune.scan: the root: None in the carry the step returned, Some in the \
       carry it received",
      fun () ->
        ignore
          (Rune.scan
             Nx.Ptree.(option tensor)
             Nx.Ptree.tensor Nx.Ptree.unit
             ~f:(fun _ _ -> (None, ()))
             ~init:(Some (scalar 0.))
             (v4 ())) );
    ( "outputs that differ from the first step's",
      "Rune.scan: the root: Some in a step's outputs, None in the first step's \
       outputs",
      fun () ->
        ignore
          (Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor
             Nx.Ptree.(option tensor)
             ~f:(fun c x ->
               let c = Nx.add_s c 1. in
               (c, if Nx.item [] c > 1.5 then Some x else None))
             ~init:(scalar 0.) (v4 ())) );
  ]

let refusal_tests =
  [
    group "messages"
      (List.map
         (fun (name, m, f) ->
           test name (fun () -> raises (Invalid_argument m) f))
         refusal_cases);
    group "under transformations"
      (List.map
         (fun (name, f) ->
           test ("a changed carry is refused under " ^ name) (fun () ->
               raises (Invalid_argument changed_length) f))
         [
           ("jvp", fun () -> ignore (Rune.jvp' scan_growing (v4 ()) (v4 ())));
           ("grad", fun () -> ignore (Rune.grad' scan_growing (v4 ())));
           ("vmap", fun () -> ignore (Rune.vmap' scan_growing (rows ())));
         ]);
    test "rows with no tensor are refused" (fun () ->
        raises (Invalid_argument "Rune.scan: xs has no leaf") (fun () ->
            Rune.scan Nx.Ptree.tensor Nx.Ptree.unit Nx.Ptree.unit
              ~f:(fun c () -> (c, ()))
              ~init:(scalar 0.) ()));
    test "a scalar row tensor is refused" (fun () ->
        raises (Invalid_argument "Rune.scan': an xs leaf is a scalar")
          (fun () -> running_sum (scalar 1.)));
    test "row tensors of two leading lengths are refused" (fun () ->
        raises
          (Invalid_argument
             "Rune.scan: the xs leaves differ in their leading length")
          (fun () ->
            Rune.scan Nx.Ptree.tensor
              Nx.Ptree.(pair tensor tensor)
              Nx.Ptree.unit
              ~f:(fun c (a, b) -> (Nx.add c (Nx.add (Nx.sum a) (Nx.sum b)), ()))
              ~init:(scalar 0.)
              (v4 (), vec [| 1.; 2. |])));
  ]

(* The step's exceptions and effects *)

exception Boom

type _ Effect.t += Ask : float Effect.t

let raising_scan x =
  fst (Rune.scan' ~f:(fun _ _ -> raise Boom) ~init:x (rows ()))

(* [guarded x] is the scan, or [3 x] when it raises [Boom]; [finalised] counts
   the runs of the finaliser around it. *)
let guarded finalised x =
  match
    Fun.protect ~finally:(fun () -> incr finalised) (fun () -> raising_scan x)
  with
  | y -> y
  | exception Boom -> Nx.mul_s x 3.

let step_tests =
  let caught name transform expected =
    test ("an exception of the step reaches the scan under " ^ name) (fun () ->
        let finalised = ref 0 in
        equal (close ()) expected (transform (guarded finalised));
        equal ~msg:"finaliser" int 1 !finalised)
  in
  let x = vec [| 0.5; -1.2; 2.1; 0.8 |] in
  [
    caught "no transformation" (fun g -> g x) (Nx.mul_s x 3.);
    caught "jvp" (fun g -> snd (Rune.jvp' g x x)) (Nx.mul_s x 3.);
    caught "grad"
      (fun g -> Rune.grad' (fun x -> Nx.sum (g x)) x)
      (Nx.full f64 [| 4 |] 3.);
    caught "vmap"
      (fun g -> Rune.vmap' g (Nx.reshape [| 1; 4 |] x))
      (Nx.reshape [| 1; 4 |] (Nx.mul_s x 3.));
    test "an effect the step leaves unhandled is the step's" (fun () ->
        let runs = ref 0 in
        let f x =
          fst
            (Rune.scan'
               ~f:(fun c _ ->
                 incr runs;
                 (Nx.add_s c (Effect.perform Ask), c))
               ~init:x (rows ()))
        in
        let g x =
          match f x with
          | y -> y
          | exception Effect.Unhandled Ask -> Nx.mul_s x 3.
        in
        equal (exact ())
          (vec [| 3.; 3.; 3.; 3. |])
          (Rune.grad' (fun x -> Nx.sum (g x)) (v4 ()));
        equal ~msg:"runs" int 1 !runs);
    test "a handler between grad and the scan answers the step's effects"
      (fun () ->
        let answered f =
          Effect.Deep.match_with f ()
            {
              retc = Fun.id;
              exnc = raise;
              effc =
                (fun (type a) (e : a Effect.t) ->
                  match e with
                  | Ask ->
                      Some
                        (fun (k : (a, _) Effect.Deep.continuation) ->
                          Effect.Deep.continue k 2.)
                  | _ -> None);
            }
        in
        let g =
          Rune.grad'
            (fun x ->
              answered (fun () ->
                  Nx.sum
                    (snd
                       (Rune.scan'
                          ~f:(fun c xi -> (c, Nx.mul_s xi (Effect.perform Ask)))
                          ~init:(scalar 0.) x))))
            (v4 ())
        in
        equal (exact ()) (Nx.full f64 [| 4 |] 2.) g);
  ]

(* Folded where it is written *)

let total : (float, Nx.float64_elt) Rune.Total.t = Rune.Total.make ()

(* A scan that adds each row's square to [total], inside a scope opened in the
   transformed function: the scope returns the sum over the rows. *)
let collected x =
  snd
    (Rune.Total.collect total ~zero:(scalar 0.) (fun () ->
         Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor Nx.Ptree.unit
           ~f:(fun c xi ->
             Rune.Total.add total (Nx.mul xi xi);
             (Nx.add c xi, ()))
           ~init:(scalar 0.) x))

let sum_of_squares x = Nx.sum (Nx.mul x x)

(* Draws under a key scope opened in the transformed function, one per row. *)
let drawn x =
  Nx.Rng.with_key (Nx.Rng.key 7) (fun () ->
      snd
        (Rune.scan'
           ~f:(fun c xi -> (Nx.add c xi, Nx.rand f64 [||]))
           ~init:(scalar 0.) x))

let placement_tests =
  [
    test "a scan inside a scope inside jvp counts each row once" (fun () ->
        let t, dt = Rune.jvp' collected (v4 ()) (v4 ()) in
        equal ~msg:"total" (close ()) (sum_of_squares (v4 ())) t;
        equal ~msg:"its tangent" (close ())
          (Nx.mul_s (sum_of_squares (v4 ())) 2.)
          dt);
    test "a scan inside a scope inside grad counts each row once" (fun () ->
        let t = ref (scalar nan) in
        let g =
          Rune.grad'
            (fun x ->
              let s = collected x in
              t := Rune.detach s;
              s)
            (v4 ())
        in
        equal ~msg:"gradient" (close ()) (Nx.mul_s (v4 ()) 2.) g;
        equal ~msg:"total" (close ()) (sum_of_squares (v4 ())) !t);
    test "a scan inside a scope inside vmap counts each lane's rows" (fun () ->
        let per_lane = Rune.vmap' collected (rows ()) in
        let expected = Nx.sum ~axes:[ 1 ] (Nx.mul (rows ()) (rows ())) in
        equal (close ()) expected per_lane);
    test "a scan inside a key scope inside grad draws as its loop" (fun () ->
        let seen = ref (vec [||]) in
        ignore
          (Rune.grad'
             (fun x ->
               seen := Rune.detach (drawn x);
               Nx.sum x)
             (v4 ()));
        equal (exact ()) (drawn (v4 ())) !seen);
    test "a scan inside a key scope inside jvp draws as its loop" (fun () ->
        let y, _ = Rune.jvp' drawn (v4 ()) (v4 ()) in
        equal (exact ()) (drawn (v4 ())) y);
  ]

(* Through the transformations, against Nx.cumsum *)

let squared_sums cumsum xs = Nx.sum (Nx.mul (cumsum xs) (cumsum xs))
let by_scan = squared_sums running_sum
let by_cumsum = squared_sums (fun x -> Nx.cumsum x)

let transformed_tests =
  [
    test "grad of a scan is grad of its primitive" (fun () ->
        equal (close ())
          (Rune.grad' by_cumsum (v4 ()))
          (Rune.grad' by_scan (v4 ())));
    test "jvp of a scan is jvp of its primitive" (fun () ->
        equal (close ())
          (snd (Rune.jvp' by_cumsum (v4 ()) (v4 ())))
          (snd (Rune.jvp' by_scan (v4 ()) (v4 ()))));
    test "vmap of a scan is the cumulative sum of each row" (fun () ->
        equal (close ())
          (Nx.cumsum ~axis:1 (rows ()))
          (Rune.vmap' running_sum (rows ())));
    test "per-row gradients of a scan" (fun () ->
        equal (close ())
          (Rune.vmap' (Rune.grad' by_cumsum) (rows ()))
          (Rune.vmap' (Rune.grad' by_scan) (rows ())));
    test "second derivatives of a scan" (fun () ->
        let second f = Rune.grad' (fun x -> Nx.sum (Rune.grad' f x)) in
        equal (close ()) (second by_cumsum (v4 ())) (second by_scan (v4 ())));
    test "Hessian-vector products of a scan" (fun () ->
        let v = vec [| 1.; 0.; -1.; 0.5 |] in
        let hvp f = snd (Rune.jvp' (Rune.grad' f) (v4 ()) v) in
        equal (close ()) (hvp by_cumsum) (hvp by_scan));
    test "a captured tensor's cotangent sums over the steps" (fun () ->
        (* Σ_i (c + x_i) · w over four rows: d/dw = Σ_i (c_i + x_i), with c_i
           the running sum before row i. *)
        let w_grad =
          Rune.grad'
            (fun w ->
              fst
                (Rune.scan'
                   ~f:(fun c x -> (Nx.add c (Nx.mul x w), c))
                   ~init:(scalar 0.) (v4 ())))
            (scalar 1.)
        in
        equal (close ()) (scalar (0.5 -. 1.2 +. 2.1 +. 0.8)) w_grad);
    test "a row's cotangent comes from its own step" (fun () ->
        (* Output i is x_i · i, so d/dx_i of the outputs' sum is i. *)
        let g =
          Rune.grad'
            (fun x ->
              Nx.sum
                (snd
                   (Rune.scan'
                      ~f:(fun c xi -> (Nx.add_s c 1., Nx.mul xi c))
                      ~init:(scalar 0.) x)))
            (v4 ())
        in
        equal (exact ()) (vec [| 0.; 1.; 2.; 3. |]) g);
  ]

(* Under a compiled function *)

let compiled_tests =
  [
    test "a scan under jit on the host is its loop written out" (fun () ->
        equal (close ()) (running_sum (v4 ())) (Rune.jit' running_sum (v4 ())));
    test "a scan's gradient under jit on the host is the eager one" (fun () ->
        equal (close ())
          (Rune.grad' by_scan (v4 ()))
          (Rune.jit' (Rune.grad' by_scan) (v4 ())));
    test "a scan under jit whose step reduces twice over one length is eager"
      (fun () ->
        let f xs =
          let c = Nx.eye f64 4 in
          Nx.sum
            (snd
               (Rune.scan Nx.Ptree.unit Nx.Ptree.tensor Nx.Ptree.tensor
                  ~f:(fun () x -> ((), Nx.sum (Nx.mul x (Nx.matmul x c))))
                  ~init:() xs))
        in
        equal (close ()) (f (rows ())) (Rune.jit' f (rows ())));
    test
      "a scan under jit reads a copy of a slice that starts off a 16-byte \
       boundary" (fun () ->
        let f x =
          let y = Nx.copy (Nx.shrink [| (1, 8) |] x) in
          fst
            (Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor Nx.Ptree.unit
               ~f:(fun c _ -> (Nx.add c y, ()))
               ~init:(Nx.zeros f64 [| 7 |])
               (Nx.zeros f64 [| 4; 1 |]))
        in
        let x = vec [| 0.; 1.; 2.; 3.; 4.; 5.; 6.; 7. |] in
        equal (close ()) (f x) (Rune.jit' f x));
    test
      "a scan under jit reads the bits of a slice that starts off a 16-byte \
       boundary" (fun () ->
        let f x =
          let y = Nx.bitcast Nx.int64 (Nx.shrink [| (1, 8) |] x) in
          fst
            (Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor Nx.Ptree.unit
               ~f:(fun c _ -> (Nx.add c y, ()))
               ~init:(Nx.zeros Nx.int64 [| 7 |])
               (Nx.zeros f64 [| 4; 1 |]))
        in
        let x = vec [| 0.; 1.; 2.; 3.; 4.; 5.; 6.; 7. |] in
        equal (exact ()) (f x) (Rune.jit' f x));
    test
      "a scan under jit reads rows of a slice that starts off a 16-byte \
       boundary" (fun () ->
        let f x =
          let xs = Nx.reshape [| 4; 2 |] (Nx.shrink [| (1, 9) |] x) in
          fst
            (Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor Nx.Ptree.unit
               ~f:(fun c r -> (Nx.add c r, ()))
               ~init:(Nx.zeros f64 [| 2 |]) xs)
        in
        let x = vec [| 0.; 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |] in
        equal (close ()) (f x) (Rune.jit' f x));
    test "a changed carry is refused under jit" (fun () ->
        raises (Invalid_argument changed_length) (fun () ->
            Rune.jit' scan_growing (v4 ())));
    test "a declined scan under jit (vmap (grad f)) is its loop written out"
      (fun () ->
        let grown x =
          let c, () =
            Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor Nx.Ptree.unit
              ~f:(fun c xi ->
                ( Nx.concatenate ~axis:0 [ c; Nx.reshape [| 1 |] (Nx.mul xi xi) ],
                  () ))
              ~init:(vec [||]) x
          in
          Nx.sum c
        in
        let f = Rune.vmap' (Rune.grad' grown) in
        equal (close ()) (f (rows ())) (Rune.jit' f (rows ())));
    test "a declined scan inside a scope under jit counts each row once"
      (fun () ->
        let grown x =
          snd
            (Rune.Total.collect total ~zero:(scalar 0.) (fun () ->
                 Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor Nx.Ptree.unit
                   ~f:(fun c xi ->
                     Rune.Total.add total (Nx.mul xi xi);
                     (Nx.concatenate ~axis:0 [ c; Nx.reshape [| 1 |] xi ], ()))
                   ~init:(vec [||]) x))
        in
        equal ~msg:"total" (close ()) (grown (v4 ())) (Rune.jit' grown (v4 ()));
        equal ~msg:"its gradient" (close ())
          (Rune.grad' grown (v4 ()))
          (Rune.jit' (Rune.grad' grown) (v4 ())));
    test "a loop in a tangent map under jit (grad f) is written out" (fun () ->
        let g =
          Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
              ( Nx.sin x,
                fun dx ->
                  fst
                    (Rune.scan'
                       ~f:(fun c _ -> (Nx.mul c (Nx.cos x), c))
                       ~init:dx
                       (vec [| 0.; 0. |])) ))
        in
        let f x = Nx.sum (g x) in
        equal (close ())
          (Rune.grad' f (v4 ()))
          (Rune.jit' (Rune.grad' f) (v4 ())));
  ]

(* Memory *)

(* A forward-mode fold of [n] trips whose step keeps a weak pointer to each
   carry it returns: at the last step, the number of those carries, other than
   the two most recent, that a full collection leaves alive. *)
let carries_alive_in_fold n =
  let kept = Weak.create n in
  let step = ref 0 and alive = ref (-1) in
  let x = Nx.ones f64 [| 64 |] in
  ignore
    (Rune.jvp'
       (fun x ->
         fst
           (Rune.scan Nx.Ptree.tensor Nx.Ptree.tensor Nx.Ptree.unit
              ~f:(fun c xi ->
                if !step = n - 1 then begin
                  Gc.full_major ();
                  alive := 0;
                  for i = 0 to n - 3 do
                    if Weak.check kept i then incr alive
                  done
                end;
                let c = Nx.add (Nx.sin c) xi in
                Weak.set kept !step (Some c);
                incr step;
                (c, ()))
              ~init:x (Nx.ones f64 [| n |])))
       x x);
  !alive

let memory_tests =
  [
    test "an eager jvp over a fold keeps no earlier step's carry alive"
      (fun () -> equal int 0 (carries_alive_in_fold 64));
  ]

let () =
  exit
    (run "Rune.scan"
       [
         group "fold" fold_tests;
         group "runs" runs_tests;
         group "refusals" refusal_tests;
         group "step" step_tests;
         group "placement" placement_tests;
         group "transformed" transformed_tests;
         group "compiled" compiled_tests;
         group "memory" memory_tests;
       ])
