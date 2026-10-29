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
         group "interface"
           [
             test "results like like" test_like;
             test "raises" test_raise;
             test "results nothing reads" test_unread;
             test "outside every transformation" test_outside;
           ];
       ])
