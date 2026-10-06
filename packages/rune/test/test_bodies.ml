(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Bodies at the call. A function a construct carries runs where its arguments
   exist: inside the handlers and the total scopes around its call, under every
   transformation, compiled or not. The trusted side is the same function with
   no construct, and the additions with no transformation. The suite reaches
   [Total.discarding], so it links rune_internals. *)

open Windtrap
module Rune = Rune_internals.Rune
module Total = Rune_internals.Total

let f64 = Nx.float64
let vec a = Nx.create f64 [| Array.length a |] a
let scalar x = Nx.scalar f64 x
let close () = Oracle.tensor ~rel:1e-9 ~abs:1e-12 ()
let tot : (float, Nx.float64_elt) Rune.Total.t = Rune.Total.make ()

let collect f =
  Rune.Total.collect tot ~zero:(scalar 0.) (fun () -> ignore (f ())) |> snd

(* A user effect, answered by a handler installed inside the transformed
   function, around a construct's call. *)
type _ Effect.t += Ask : float Effect.t

let answers = ref 0

let answer f =
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
                  incr answers;
                  Effect.Deep.continue k 2.)
          | _ -> None);
    }

let ask () = Effect.perform Ask

(* g x = Σ sin (a xᵢ), [a] asked of the handler, with an addition of Σ xᵢ: the
   function every body computes. *)
let terms x = Nx.sin (Nx.mul_s x (ask ()))

let plain x =
  Rune.Total.add tot (Nx.sum x);
  Nx.sum (terms x)

(* [dplain a x] is [plain]'s derivative, [a] asked at the rule's call: a
   pullback runs after the call returns, outside the handler. *)
let dplain a x = Nx.mul_s (Nx.cos (Nx.mul_s x a)) a

(* Bodies *)

type body =
  | Plain
  | Scan
  | Iterate
  | Remat
  | Custom_jvp
  | Custom_vjp
  | Root
  | Jit

let bodies = [ Plain; Scan; Iterate; Remat; Custom_jvp; Custom_vjp; Root; Jit ]

let body_name = function
  | Plain -> "plain"
  | Scan -> "scan"
  | Iterate -> "iterate"
  | Remat -> "remat"
  | Custom_jvp -> "custom_jvp"
  | Custom_vjp -> "custom_vjp"
  | Root -> "root"
  | Jit -> "jit"

let pair = Nx.Ptree.(pair tensor tensor)

let through = function
  | Plain -> plain
  | Scan ->
      fun x ->
        fst
          (Rune.scan'
             ~f:(fun c xi ->
               Rune.Total.add tot xi;
               (Nx.add c (Nx.sum (terms xi)), c))
             ~init:(scalar 0.) x)
  | Iterate ->
      fun x ->
        let n = (Nx.shape x).(0) in
        let at i =
          Nx.take ~axis:0 ~indices:(Nx.reshape [| 1 |] (Nx.cast Nx.int64 i)) x
        in
        snd
          (Rune.iterate pair ~max:n
             ~until:(fun (i, _) -> Nx.greater_equal_s i (Float.of_int n))
             ~f:(fun (i, acc) ->
               let xi = at i in
               Rune.Total.add tot (Nx.sum xi);
               (Nx.add_s i 1., Nx.add acc (Nx.sum (terms xi))))
             (scalar 0., scalar 0.))
  | Remat -> Rune.remat Nx.Ptree.(tensor @-> returns tensor) plain
  | Custom_jvp ->
      Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
          let a = ask () in
          (plain x, fun dx -> Nx.sum (Nx.mul (dplain a x) dx)))
  | Custom_vjp ->
      Rune.custom_vjp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
          let a = ask () in
          (plain x, fun ct -> Nx.mul ct (dplain a x)))
  | Root ->
      fun x ->
        Rune.root Nx.Ptree.tensor
          ~residual:(fun y -> Nx.sub y (Nx.sum (terms x)))
          (fun () -> plain x)
  | Jit -> Rune.jit' plain

(* Stacks *)

type layer = Grad | Jvp | Vmap | Jit_layer | Collect | Discarding

let layers = [ Grad; Jvp; Vmap; Jit_layer; Collect; Discarding ]

let layer_name = function
  | Grad -> "grad"
  | Jvp -> "jvp"
  | Vmap -> "vmap"
  | Jit_layer -> "jit"
  | Collect -> "collect"
  | Discarding -> "discarding"

let pp_stack ppf (ts, b) =
  Format.fprintf ppf "%s ∘ %s"
    (String.concat " ∘ " (List.map layer_name ts))
    (body_name b)

(* [compose ts f] is [f] through [ts], the first outermost; the handler sits
   innermost, around the body's call. A scope inside adds its total to the one
   around, so it changes no total. *)
let compose ts f =
  let rec go = function
    | [] -> fun x -> answer (fun () -> f x)
    | t :: rest -> (
        let g = go rest in
        match t with
        | Grad -> fun x -> Rune.grad' (fun x -> Nx.sum (g x)) x
        | Jvp -> fun x -> snd (Rune.jvp' g x (Nx.cos x))
        | Vmap -> Rune.vmap' g
        | Jit_layer -> Rune.jit' g
        | Collect ->
            fun x ->
              let y, t =
                Rune.Total.collect tot ~zero:(scalar 0.) (fun () -> g x)
              in
              Rune.Total.add tot t;
              y
        | Discarding -> fun x -> Total.discarding (fun () -> g x))
  in
  go ts

let argument ts =
  let maps = List.length (List.filter (( = ) Vmap) ts) in
  let shape = Array.append (Array.make maps 2) [| 3 |] in
  let n = Array.fold_left ( * ) 1 shape in
  Nx.reshape shape
    (vec (Array.init n (fun i -> 0.3 +. (0.17 *. Float.of_int i))))

(* A custom_vjp rule has no forward derivative. *)
let forward_inside ts =
  List.fold_left
    (fun acc t -> match t with Grad -> Some Grad | Jvp -> Some Jvp | _ -> acc)
    None ts
  = Some Jvp

let stack =
  let open Gen in
  let* n = int_range 1 3 in
  let* ts = list ~size:(constant n) (of_list layers) in
  let+ b = of_list bodies in
  (ts, b)

let valid (ts, b) = not (b = Custom_vjp && forward_inside ts)

let law1 =
  prop ~tags:[ "slow" ] ~count:120
    ~examples:
      [
        ([ Grad; Grad; Collect ], Custom_jvp);
        ([ Collect; Grad; Collect ], Remat);
      ]
    "a body runs at its call under every stack: the handler around the call \
     answers it, the value is the plain function's, and each addition counts \
     once"
    (Gen.with_pp pp_stack (Gen.such_that valid stack))
    (fun (ts, b) ->
      cover "a loop under jit"
        ((b = Scan || b = Iterate) && List.mem Jit_layer ts);
      cover "a rule under jvp" (b = Custom_jvp && List.mem Jvp ts);
      cover "a root under vmap" (b = Root && List.mem Vmap ts);
      let x = argument ts in
      let expected = compose ts plain x in
      answers := 0;
      let total = ref (scalar 0.) in
      let got =
        Rune.Total.collect tot ~zero:(scalar 0.) (fun () ->
            compose ts (through b) x)
      in
      total := snd got;
      greater ~msg:"answers" int ~than:0 !answers;
      equal ~msg:"value" (close ()) expected (fst got);
      let added =
        if List.mem Discarding ts then scalar 0. else Nx.sum (argument ts)
      in
      equal ~msg:"total" (close ()) added !total)

(* The boundary: a construct an answer derives meets each installation once *)

let a = Rune.axis ()

(* [nested ran x] scans the rows of [x], each step scanning its row's elements,
   the inner step counted in [ran]. The carry starts from [x], so that a map
   batches it from the start and no attempt restarts. *)
let nested ran x =
  fst
    (Rune.scan'
       ~f:(fun c row ->
         let inner, _ =
           Rune.scan'
             ~f:(fun d e ->
               incr ran;
               (Nx.add (Nx.mul_s d 0.5) (Nx.sin e), d))
             ~init:c row
         in
         (inner, inner))
       ~init:(Nx.mul_s (Nx.get [ 0; 0 ] x) 0.)
       x)

let lanes_gen =
  Gen.with_pp Nx.pp
    (Gen.map
       (fun l -> Nx.reshape [| 2; 3; 2 |] (vec (Array.of_list l)))
       Gen.(list ~size:(constant 12) (float_range (-2.) 2.)))

let boundary_tests =
  [
    prop
      "a scan in a scan under jit of vmap traces the inner step once and is \
       each lane's"
      lanes_gen (fun xs ->
        let ran = ref 0 in
        let each =
          Nx.stack (List.init 2 (fun i -> nested ran (Nx.get [ i ] xs)))
        in
        ran := 0;
        let got = Rune.jit' (Rune.vmap' (nested ran)) xs in
        equal (close ()) each got;
        equal ~msg:"inner step traces" int 1 !ran);
    prop "grad of a map of a remat whose function scans is each lane's gradient"
      lanes_gen (fun xs ->
        let f w x =
          Rune.remat
            Nx.Ptree.(tensor @-> returns tensor)
            (fun x -> nested (ref 0) (Nx.mul x w))
            x
        in
        let w0 = scalar 0.7 in
        let lane i w = f w (Nx.get [ i ] xs) in
        let expected =
          Nx.add (Rune.grad' (lane 0) w0) (Rune.grad' (lane 1) w0)
        in
        let got = Rune.grad' (fun w -> Nx.sum (Rune.vmap' (f w) xs)) w0 in
        equal (close ()) expected got);
    prop
      "an addition each lane makes in a compiled scan's step counts once per \
       lane"
      lanes_gen (fun xs ->
        let f x =
          fst
            (Rune.scan'
               ~f:(fun c row ->
                 Rune.Total.add tot (Nx.sum row);
                 (Nx.add c (Nx.sum row), c))
               ~init:(scalar 0.) x)
        in
        equal (close ()) (Nx.sum xs)
          (collect (fun () -> Rune.jit' (Rune.vmap' f) xs)));
    prop
      "an addition each lane makes in a masked step counts once per trip it \
       takes"
      lanes_gen (fun xs ->
        (* Lane i halves until below a tenth of its start's size: lanes stop
           apart. *)
        let f x =
          Rune.iterate' ~max:64
            ~until:(fun y -> Nx.less_s (Nx.sum (Nx.abs y)) 0.5)
            ~f:(fun y ->
              Rune.Total.add tot (scalar 1.);
              Nx.mul_s y 0.5)
            x
        in
        let trips x =
          let rec go k y =
            if Nx.item [] (Nx.sum (Nx.abs y)) < 0.5 then k
            else go (k + 1) (Nx.mul_s y 0.5)
          in
          go 0 x
        in
        let flat = Nx.reshape [| 2; 6 |] xs in
        let expected =
          Float.of_int (trips (Nx.get [ 0 ] flat) + trips (Nx.get [ 1 ] flat))
        in
        cover "lanes stop apart"
          (trips (Nx.get [ 0 ] flat) <> trips (Nx.get [ 1 ] flat));
        equal (close ()) (scalar expected)
          (collect (fun () -> Rune.vmap' f flat)));
    prop
      "an addition each lane makes in a compiled call in a masked step counts \
       once per trip it takes"
      lanes_gen (fun xs ->
        let counted =
          Rune.jit' (fun y ->
              Rune.Total.add tot (scalar 1.);
              Nx.mul_s y 0.5)
        in
        let f x =
          Rune.iterate' ~max:64
            ~until:(fun y -> Nx.less_s (Nx.sum (Nx.abs y)) 0.5)
            ~f:counted x
        in
        let trips x =
          let rec go k y =
            if Nx.item [] (Nx.sum (Nx.abs y)) < 0.5 then k
            else go (k + 1) (Nx.mul_s y 0.5)
          in
          go 0 x
        in
        let flat = Nx.reshape [| 2; 6 |] xs in
        let expected =
          Float.of_int (trips (Nx.get [ 0 ] flat) + trips (Nx.get [ 1 ] flat))
        in
        cover "lanes stop apart"
          (trips (Nx.get [ 0 ] flat) <> trips (Nx.get [ 1 ] flat));
        equal (close ()) (scalar expected)
          (collect (fun () -> Rune.vmap' f flat)));
  ]

(* Installations that pass a construct meet its function at their level *)

let rule_adding =
  Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
      Rune.Total.add tot (Nx.sum x);
      (x, Fun.id))

let x3 () = vec [| 0.5; -1.; 2. |]

let passing_tests =
  [
    test "a rule's addition under grad reaches the scope around its call"
      (fun () ->
        let total = ref (scalar 0.) in
        ignore
          (Rune.grad'
             (fun x ->
               let y, t =
                 Rune.Total.collect tot ~zero:(scalar 0.) (fun () ->
                     Nx.sum (rule_adding x))
               in
               total := Rune.detach t;
               y)
             (x3 ()));
        equal (close ()) (Nx.sum (x3 ())) !total);
    test
      "a rule's addition under vmap reaches the scope around its call, lane by \
       lane" (fun () ->
        let xs = Nx.reshape [| 3; 1 |] (x3 ()) in
        let totals =
          Rune.vmap'
            (fun x ->
              snd
                (Rune.Total.collect tot ~zero:(scalar 0.) (fun () ->
                     ignore (rule_adding x))))
            xs
        in
        equal (close ()) (x3 ()) totals);
    test "a rule's addition under a discarding scope inside grad is dropped"
      (fun () ->
        equal (close ()) (scalar 0.)
          (collect (fun () ->
               Rune.grad'
                 (fun x -> Total.discarding (fun () -> Nx.sum (rule_adding x)))
                 (x3 ()))));
    test
      "a rule a map answers inside grad reads a detached value through its \
       closure with no derivative" (fun () ->
        let xs = Nx.reshape [| 3; 1 |] (x3 ()) in
        let g x =
          Rune.grad'
            (fun w ->
              let rule =
                Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
                    let c = Rune.detach (Nx.mul w w) in
                    (Nx.mul x c, fun dx -> Nx.mul dx c))
              in
              Nx.sum (rule x))
            (scalar 1.5)
        in
        equal (close ()) (Nx.zeros f64 [| 3 |]) (Rune.vmap' g xs));
    test
      "grad through a compiled remat of an iterate at an untracked point is \
       eager's" (fun () ->
        let x0 = vec [| 4.; -3. |] in
        let halve x =
          Rune.iterate' ~max:16
            ~until:(fun y -> Nx.less_s (Nx.max (Nx.abs y)) 0.5)
            ~f:(fun y -> Nx.mul_s y 0.5)
            x
        in
        let g w =
          Nx.add w
            (Nx.sum
               (Rune.jit'
                  (Rune.remat Nx.Ptree.(tensor @-> returns tensor) halve)
                  x0))
        in
        equal (close ()) (scalar 1.) (Rune.grad' g (scalar 0.2)));
    test
      "grad through a compiled root whose solve iterates, at an untracked \
       point, is eager's" (fun () ->
        let x0 = scalar 2. in
        let sqrt x =
          Rune.root Nx.Ptree.tensor
            ~residual:(fun y -> Nx.sub (Nx.mul y y) x)
            (fun () ->
              Rune.iterate' ~max:64
                ~until:(fun y ->
                  Nx.less_s (Nx.abs (Nx.sub (Nx.mul y y) x)) 1e-12)
                ~f:(fun y -> Nx.mul_s (Nx.add y (Nx.div x y)) 0.5)
                (scalar 1.))
        in
        let g w = Nx.add w (Rune.jit' sqrt x0) in
        equal (close ()) (scalar 1.) (Rune.grad' g (scalar 0.2)));
    test
      "a residual that gathers the map's lanes inside a compiled scan is \
       refused" (fun () ->
        let xs = Nx.reshape [| 2; 1 |] (vec [| 4.; 9. |]) in
        let sqrt w x =
          Rune.root Nx.Ptree.tensor
            ~residual:(fun y ->
              let s, _ =
                Rune.scan'
                  ~f:(fun c e -> (Nx.add c (Nx.sum (Rune.lanes a e)), c))
                  ~init:(scalar 0.) (Nx.mul y y)
              in
              Nx.sub (Nx.add (Nx.mul y y) (Nx.mul_s s 0.)) (Nx.mul w x))
            (fun () -> Nx.sqrt (Nx.mul w x))
        in
        raises
          (Invalid_argument
             "Rune.root: the residual reads other lanes of the map, so the \
              lanes' systems are not separate") (fun () ->
            Rune.jit'
              (Rune.grad' (fun w -> Nx.sum (Rune.vmap' ~axis:a (sqrt w) xs)))
              (scalar 1.)));
  ]

(* Records: a construct in a step, replayed in the backward pass *)

(* A rule whose tangent map is not its primal code's derivative: twice the
   tangent, where [sin]'s derivative is [cos]. A replay that differentiated the
   rule's code would give [cos]. *)
let doubled =
  Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
      (Nx.sin x, fun dx -> Nx.mul_s dx 2.))

(* [ruled step x] scans [x]'s elements with a step that applies [doubled]. *)
let ruled x =
  fst
    (Rune.scan'
       ~f:(fun c e ->
         let c = Nx.add (Nx.mul_s c 0.5) (doubled (Nx.mul e c)) in
         (c, c))
       ~init:(Nx.add_s (Nx.get [ 0 ] x) 0.2)
       x)

(* Rules for [sin] whose tangent map or pullback reads a value the rule's own
   run computed, [cos x]: a replay at another trip must bind it to that trip's
   value. *)
let sin_jvp =
  Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
      let c = Nx.cos x in
      (Nx.sin x, fun dx -> Nx.mul dx c))

let sin_vjp =
  Rune.custom_vjp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
      let c = Nx.cos x in
      (Nx.sin x, fun ct -> Nx.mul ct c))

(* [stepped rule x] scans [x]'s elements with a step that applies [rule];
   [unrolled] is the same loop in OCaml with [Nx.sin]. *)
let stepped rule x =
  fst
    (Rune.scan'
       ~f:(fun c e ->
         let c = Nx.add (Nx.mul_s c 0.5) (rule (Nx.mul e c)) in
         (c, c))
       ~init:(Nx.add_s (Nx.get [ 0 ] x) 0.2)
       x)

let unrolled x =
  let c = ref (Nx.add_s (Nx.get [ 0 ] x) 0.2) in
  for i = 0 to (Nx.shape x).(0) - 1 do
    c := Nx.add (Nx.mul_s !c 0.5) (Nx.sin (Nx.mul (Nx.get [ i ] x) !c))
  done;
  !c

(* [halving x] iterates [doubled] on a carry that halves until it is small: a
   lane's trips depend on its start. *)
let halving x =
  Rune.iterate' ~max:32
    ~until:(fun y -> Nx.less_s (Nx.sum (Nx.abs y)) 0.1)
    ~f:(fun y -> Nx.mul_s (doubled y) 0.5)
    x

let grad2 f x =
  Rune.grad' (fun x -> Nx.sum (Rune.grad' (fun x -> Nx.sum (f x)) x)) x

(* A root in a step: the square root of [c + 1], by Newton steps. *)
let rooted ?linear_solve x =
  fst
    (Rune.scan'
       ~f:(fun c e ->
         let target = Nx.add_s (Nx.mul (Nx.mul c c) e) 1. in
         let r =
           Rune.root Nx.Ptree.tensor ?linear_solve
             ~residual:(fun y -> Nx.sub (Nx.mul y y) target)
             (fun () ->
               Rune.iterate' ~max:64
                 ~until:(fun y ->
                   Nx.less_s (Nx.abs (Nx.sub (Nx.mul y y) target)) 1e-12)
                 ~f:(fun y -> Nx.mul_s (Nx.add y (Nx.div target y)) 0.5)
                 (Nx.ones_like target))
         in
         let c = Nx.mul_s r 0.5 in
         (c, c))
       ~init:(Nx.add_s (Nx.get [ 0 ] x) 0.3)
       x)

(* Linear solves of a scalar system that apply their operator: once, and under a
   jvp they open. *)
let applying op b = Nx.div b (op (Nx.ones_like b))
let opening op b = Nx.div b (snd (Rune.jvp' op b (Nx.ones_like b)))

(* A step that calls a compiled function and a custom_vjp rule, whose pullback
   is its function's derivative. *)
let squared =
  Rune.custom_vjp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
      (Nx.mul x x, fun ct -> Nx.mul ct (Nx.mul_s x 2.)))

let compiled_sin = Rune.jit' Nx.sin

let calling x =
  fst
    (Rune.scan'
       ~f:(fun c e ->
         let c =
           Nx.add (Nx.mul_s (compiled_sin (Nx.mul c e)) 0.5) (squared e)
         in
         (c, c))
       ~init:(Nx.add_s (Nx.get [ 0 ] x) 0.1)
       x)

(* A step that reads a detached value: [c + detach (c²) e]. *)
let detaching x =
  fst
    (Rune.scan'
       ~f:(fun c e ->
         let c = Nx.sin (Nx.add c (Nx.mul (Rune.detach (Nx.mul c c)) e)) in
         (c, c))
       ~init:(Nx.add_s (Nx.get [ 0 ] x) 0.1)
       x)

let gen_vec =
  Gen.with_pp Nx.pp
    Gen.(
      map
        (fun l -> vec (Array.of_list l))
        (list ~size:(int_range 1 4) (float_range (-1.) 1.)))

let gen_lanes =
  Gen.with_pp Nx.pp
    Gen.(
      map
        (fun l -> Nx.reshape [| 2; 3 |] (vec (Array.of_list l)))
        (list ~size:(constant 6) (float_range (-1.) 1.)))

let grad1 f x = Rune.grad' (fun x -> Nx.sum (f x)) x

(* [held loss x] is the host memory a pullback of [loss] at [x] holds once the
   forward pass returned, after a first, uncounted run: a chain of finalisers
   frees its memory a cycle late, so a fixed number of rounds settles it. *)
let held loss x =
  let allocated () =
    for _ = 1 to 4 do
      Gc.full_major ()
    done;
    Nx_device.Stats.allocated (Nx_device.stats Nx_device.host)
  in
  let measure () =
    let base = allocated () in
    let _, pullback = Rune.vjp' loss x in
    let used = allocated () - base in
    ignore (Sys.opaque_identity (Obj.repr pullback));
    used
  in
  ignore (measure ());
  measure ()

(* Eight blocks of three products each, on a batch of 256 rows. *)
let blocks remat x =
  let w = Nx.mul_s (Nx.eye f64 64) 0.9 in
  let block a =
    Nx.tanh (Nx.matmul (Nx.tanh (Nx.matmul (Nx.tanh (Nx.matmul a w)) w)) w)
  in
  let block =
    if remat then Rune.remat Nx.Ptree.(tensor @-> returns tensor) block
    else block
  in
  let rec go k a = if k = 0 then a else go (k - 1) (block a) in
  Nx.sum (go 8 x)

(* Compiled functions are made once, so that each shape compiles once. *)
let record_tests =
  let rooted_grad = grad1 (fun x -> rooted x) in
  let rooted_jit = Rune.jit' rooted_grad in
  let solves =
    List.map
      (fun (name, linear_solve) ->
        let g = grad1 (rooted ~linear_solve) in
        (name, g, Rune.jit' g))
      [ ("applying", applying); ("opening", opening) ]
  in
  let calling_jit = Rune.jit' (grad1 calling) in
  let calling_mapped_jit = Rune.jit' (Rune.vmap' (grad1 calling)) in
  let detaching_jit = Rune.jit' (grad2 detaching) in
  let ruled_jit = Rune.jit' (grad2 ruled) in
  let bound =
    List.map
      (fun (name, rule) ->
        let g = grad2 (stepped rule) in
        (name, rule, g, Rune.jit' g, Rune.jit' (Rune.vmap' g)))
      [ ("custom_jvp", sin_jvp); ("custom_vjp", sin_vjp) ]
  in
  [
    test "eager grad of a remat keeps no intermediate of its function"
      (fun () ->
        let x = Nx.full f64 [| 256; 64 |] 0.1 in
        let plain = held (blocks false) x and rematted = held (blocks true) x in
        less
          ~msg:(Printf.sprintf "%d bytes with remat, %d without" rematted plain)
          int ~than:(plain / 2) rematted);
    prop "a root in a compiled scan's step under grad is eager's" gen_vec
      (fun x -> equal (close ()) (rooted_grad x) (rooted_jit x));
    prop
      "a root's linear_solve in a step, replayed, applies the replay's \
       operator, directly and under a jvp it opens, compiled and mapped"
      gen_lanes (fun xs ->
        let x = Nx.get [ 0 ] xs in
        let expected = rooted_grad x
        and each =
          Nx.stack (List.init 2 (fun i -> rooted_grad (Nx.get [ i ] xs)))
        in
        List.iter
          (fun (name, g, compiled) ->
            equal ~msg:name (close ()) expected (g x);
            equal ~msg:(name ^ ", compiled") (close ()) expected (compiled x);
            equal ~msg:(name ^ ", mapped") (close ()) each (Rune.vmap' g xs))
          solves);
    prop
      "a compiled call and a custom_vjp rule in a step, compiled under grad \
       and mapped, are eager's"
      gen_lanes (fun xs ->
        let each =
          Nx.stack (List.init 2 (fun i -> grad1 calling (Nx.get [ i ] xs)))
        in
        equal ~msg:"compiled" (close ())
          (grad1 calling (Nx.get [ 0 ] xs))
          (calling_jit (Nx.get [ 0 ] xs));
        equal ~msg:"mapped" (close ()) each (Rune.vmap' (grad1 calling) xs);
        equal ~msg:"mapped, compiled" (close ()) each (calling_mapped_jit xs));
    prop
      "grad of grad through a compiled scan whose step detaches is eager's: \
       the detached value has no derivative at either order"
      gen_vec (fun x -> equal (close ()) (grad2 detaching x) (detaching_jit x));
    prop
      "vmap of grad of an iterate whose step reads each lane's index is each \
       lane's, lanes stopping apart"
      gen_lanes (fun xs ->
        let f lane x =
          Rune.iterate' ~max:64
            ~until:(fun y -> Nx.less_s (Nx.sum (Nx.abs y)) 0.2)
            ~f:(fun y ->
              Nx.mul (Nx.mul_s y 0.5)
                (Nx.add_s (Nx.mul_s (Nx.cast f64 (lane ())) 0.1) 1.))
            x
        in
        let g lane x = Rune.grad' (fun x -> Nx.sum (f lane x)) x in
        let each =
          Nx.stack
            (List.init 2 (fun i ->
                 g
                   (fun () -> Nx.scalar Nx.int32 (Int32.of_int i))
                   (Nx.get [ i ] xs)))
        in
        equal (close ()) each (Rune.vmap' (g (fun () -> Rune.lane_index ())) xs));
    prop
      "vmap of grad of an iterate whose step gathers the lanes, with a shared \
       stop, is the batch's"
      gen_lanes (fun xs ->
        let step mean y = Nx.mul_s (Nx.add y (mean y)) 0.4 in
        let until (_, k) = Nx.greater_equal_s k 3. in
        let f mean x =
          fst
            (Rune.iterate
               Nx.Ptree.(pair tensor tensor)
               ~max:8 ~until
               ~f:(fun (y, k) -> (step mean y, Nx.add_s k 1.))
               (x, scalar 0.))
        in
        let lanes y = Nx.mean ~axes:[ 0 ] (Rune.lanes a y) in
        let batch y =
          Nx.broadcast_to (Nx.shape y) (Nx.mean ~axes:[ 0 ] ~keepdims:true y)
        in
        let expected = Rune.grad' (fun xs -> Nx.sum (f batch xs)) xs in
        equal (close ()) expected
          (Rune.grad' (fun xs -> Nx.sum (Rune.vmap' ~axis:a (f lanes) xs)) xs));
    prop
      "a custom rule in a compiled scan's step under grad of grad applies its \
       tangent map, as eagerly"
      Gen.(
        map
          (fun l -> vec (Array.of_list l))
          (list ~size:(int_range 1 4) (float_range (-1.) 1.)))
      (fun x -> equal (close ()) (grad2 ruled x) (ruled_jit x));
    prop
      "a rule in a scan's step whose tangent map or pullback reads its own \
       run's value is, under grad of grad, the unrolled loop's, eagerly, \
       compiled and mapped"
      gen_lanes (fun xs ->
        let x = Nx.get [ 0 ] xs in
        let expected = grad2 unrolled x
        and each =
          Nx.stack (List.init 2 (fun i -> grad2 unrolled (Nx.get [ i ] xs)))
        in
        List.iter
          (fun (name, _, g, compiled, mapped) ->
            equal ~msg:name (close ()) expected (g x);
            equal ~msg:(name ^ ", compiled") (close ()) expected (compiled x);
            equal ~msg:(name ^ ", mapped") (close ()) each (Rune.vmap' g xs);
            equal
              ~msg:(name ^ ", mapped and compiled")
              (close ()) each (mapped xs))
          bound);
    prop
      "grad of grad of a scan whose step applies such a rule is the central \
       difference of its grad" (Gen.pair gen_lanes gen_lanes) (fun (xs, vs) ->
        let x = Nx.get [ 0 ] xs and v = Nx.get [ 0 ] vs in
        List.iter
          (fun (name, rule, g, _, _) ->
            let grad x = Nx.sum (grad1 (stepped rule) x) in
            equal ~msg:name (float 1e-6)
              (Nx.item [] (Oracle.central ~eps:1e-5 grad x v))
              (Oracle.dot (g x) v))
          bound);
    prop
      "a custom rule in an iterate's step under vmap of grad applies its \
       tangent map, each lane as alone"
      Gen.(
        map
          (fun l -> Nx.reshape [| 2; 2 |] (vec (Array.of_list l)))
          (list ~size:(constant 4) (float_range (-2.) 2.)))
      (fun xs ->
        let g x = Rune.grad' (fun x -> Nx.sum (halving x)) x in
        let each = Nx.stack (List.init 2 (fun i -> g (Nx.get [ i ] xs))) in
        equal (close ()) each (Rune.vmap' g xs));
  ]

(* Total scopes and roots *)

let root_adding x =
  Rune.root Nx.Ptree.tensor
    ~residual:(fun y ->
      Rune.Total.add tot (scalar 100.);
      Nx.sub (Nx.mul y y) x)
    ~linear_solve:(fun op b ->
      Rune.Total.add tot (scalar 1000.);
      Nx.div b (op (Nx.ones_like b)))
    (fun () ->
      Rune.Total.add tot x;
      Nx.sqrt x)

let lane_totals f xs =
  Rune.vmap'
    (fun x ->
      snd (Rune.Total.collect tot ~zero:(scalar 0.) (fun () -> ignore (f x))))
    xs

let root_tests =
  let xs () = vec [| 4.; 9.; 0.25 |] in
  [
    test "a root's solve adds to the scope inside a map, lane by lane"
      (fun () ->
        equal (close ()) (xs ())
          (lane_totals root_adding (Nx.reshape [| 3 |] (xs ()))));
    test "a root's solve adds to the scope inside a compiled map, lane by lane"
      (fun () ->
        equal (close ()) (xs ()) (Rune.jit' (lane_totals root_adding) (xs ())));
    test "the residual's and linear_solve's additions are dropped under grad"
      (fun () ->
        let total = ref (scalar 0.) in
        let g =
          Rune.grad'
            (fun x ->
              let y, t =
                Rune.Total.collect tot ~zero:(scalar 0.) (fun () ->
                    root_adding x)
              in
              total := Rune.detach t;
              y)
            (scalar 4.)
        in
        equal ~msg:"derivative" (close ()) (scalar 0.25) g;
        equal ~msg:"total" (close ()) (scalar 4.) !total);
    test "the solve's additions under grad of a map count once per lane"
      (fun () ->
        equal (close ())
          (Nx.sum (xs ()))
          (collect (fun () ->
               Rune.grad' (fun x -> Nx.sum (Rune.vmap' root_adding x)) (xs ()))));
  ]

(* A handler that computes from a site's value *)

type _ Effect.t +=
  | Sample : Nx.float64_t -> Nx.float64_t Effect.t
  | Site : (Nx.float64_t -> Nx.float64_t) Effect.t

(* [with_sites f] answers [Sample v] by adding [v]'s log density, computed in
   the handler, and [Site] by a function the site applies, which adds it where
   the site is. *)
let with_sites f =
  let density v = Nx.neg (Nx.sum (Nx.mul v v)) in
  Effect.Deep.match_with f ()
    {
      retc = Fun.id;
      exnc = raise;
      effc =
        (fun (type a) (e : a Effect.t) ->
          match e with
          | Sample v ->
              Some
                (fun (k : (a, _) Effect.Deep.continuation) ->
                  Rune.Total.add tot (density v);
                  Effect.Deep.continue k v)
          | Site ->
              Some
                (fun (k : (a, _) Effect.Deep.continuation) ->
                  Effect.Deep.continue k (fun v ->
                      Rune.Total.add tot (density v);
                      v))
          | _ -> None);
    }

let rows () = Nx.create f64 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]

let model site p =
  fst
    (Rune.scan'
       ~f:(fun c x ->
         let c = site (Nx.add c (Nx.mul p x)) in
         (c, c))
       ~init:(Nx.zeros f64 [| 2 |]) (rows ()))

let log_density site p =
  snd
    (Rune.Total.collect tot ~zero:(scalar 0.) (fun () ->
         with_sites (fun () -> model site p)))

let sampled v = Effect.perform (Sample v)
let applied v = (Effect.perform Site) v
let p0 () = vec [| 0.1; 0.2 |]

let handler_tests =
  [
    test
      "a handler that adds a value it computes from a step's site reaches the \
       scope eagerly" (fun () ->
        equal (close ())
          (log_density applied (p0 ()))
          (log_density sampled (p0 ())));
    test
      "a handler that adds a value it computes from a step's site raises under \
       jit" (fun () ->
        raises
          (Invalid_argument
             "Rune.jit: a value computed inside a loop's step escaped it; \
              return it in the carry or add it to a Rune.Total") (fun () ->
            Rune.jit' (log_density sampled) (p0 ())));
    test
      "a handler that answers with a function the site applies adds under jit \
       as eagerly" (fun () ->
        equal (close ())
          (log_density applied (p0 ()))
          (Rune.jit' (log_density applied) (p0 ())));
  ]

(* Law 5: a loop's draws. Step [i] draws from a scope rooted at [Nx.Rng.fold_in
   k i], [k] one key the loop takes at its first draw; a loop whose step draws
   nothing takes none. The trusted side is the loop unrolled in OCaml with those
   scopes, under the same stack. *)

type drawing = Scan_draws | Iterate_draws | Nested_draws

let drawing_name = function
  | Scan_draws -> "scan"
  | Iterate_draws -> "iterate"
  | Nested_draws -> "scan in a scan's step"

(* [scoped x f] runs [f] in a scope whose root depends on [x], so that a jit
   compiles its draws. *)
let scoped x f =
  let zero = Nx.cast Nx.int32 (Nx.mul_s (Nx.sum x) 0.) in
  Nx.Rng.with_key (Nx.Rng.fold_in_tensor (Nx.Rng.key 7) zero) f

let draw draws = if draws then Nx.rand f64 [||] else scalar 0.5
let row x i = Nx.slice [ Nx.I i ] x
let weighed draws c xi = Nx.add c (Nx.mul xi (draw draws))

(* [unrolled draws n step] is [n] steps [step i c] from [c = 0], step [i] in a
   scope rooted at [fold_in k i]. *)
let unrolled draws n step =
  let k = lazy (Nx.Rng.next_key ()) in
  let c = ref (scalar 0.) in
  for i = 0 to n - 1 do
    let run () = step i !c in
    c :=
      if draws then Nx.Rng.with_key (Nx.Rng.fold_in (Lazy.force k) i) run
      else run ()
  done;
  !c

let summed draws x =
  fst (Rune.scan' ~f:(fun c xi -> (weighed draws c xi, c)) ~init:(scalar 0.) x)

let looped kind draws x =
  let n = (Nx.shape x).(0) in
  match kind with
  | Scan_draws -> summed draws x
  | Iterate_draws ->
      let at i =
        Nx.sum
          (Nx.take ~axis:0 ~indices:(Nx.reshape [| 1 |] (Nx.cast Nx.int64 i)) x)
      in
      snd
        (Rune.iterate pair ~max:n
           ~until:(fun (i, _) -> Nx.greater_equal_s i (Float.of_int n))
           ~f:(fun (i, c) -> (Nx.add_s i 1., weighed draws c (at i)))
           (scalar 0., scalar 0.))
  | Nested_draws ->
      fst
        (Rune.scan'
           ~f:(fun c xi ->
             let c = Nx.add c (Nx.mul xi (summed draws x)) in
             (weighed draws c xi, c))
           ~init:(scalar 0.) x)

let spec kind draws x =
  let n = (Nx.shape x).(0) in
  let flat i c = weighed draws c (row x i) in
  match kind with
  | Scan_draws | Iterate_draws -> unrolled draws n flat
  | Nested_draws ->
      unrolled draws n (fun i c ->
          let c = Nx.add c (Nx.mul (row x i) (unrolled draws n flat)) in
          weighed draws c (row x i))

(* The draws around the loop show the key it takes. *)
let around loop x =
  scoped x (fun () ->
      let before = Nx.rand f64 [||] in
      let c = loop x in
      let after = Nx.rand f64 [||] in
      Nx.add c (Nx.mul (Nx.sum x) (Nx.add before after)))

let law5 =
  let gen =
    let open Gen in
    let* n = int_range 0 2 in
    let* ts = list ~size:(constant n) (of_list layers) in
    let* kind = of_list [ Scan_draws; Iterate_draws; Nested_draws ] in
    let+ draws = bool in
    (ts, kind, draws)
  in
  let pp ppf (ts, kind, draws) =
    Format.fprintf ppf "%s ∘ %s%s"
      (String.concat " ∘ " (List.map layer_name ts))
      (drawing_name kind)
      (if draws then "" else ", drawing nothing")
  in
  prop ~count:80
    ~examples:[ ([ Grad; Jit_layer ], Scan_draws, true) ]
    "step i of a loop draws from a scope rooted at fold_in k i, k one key the \
     loop takes at its first draw, under every stack"
    (Gen.with_pp pp gen)
    (fun (ts, kind, draws) ->
      cover "a loop under jit" (List.mem Jit_layer ts);
      cover "a loop under vmap" (List.mem Vmap ts);
      cover "a loop under grad" (List.mem Grad ts);
      cover "a step that draws nothing" (not draws);
      let x = argument ts in
      equal (close ())
        (compose ts (around (spec kind draws)) x)
        (compose ts (around (looped kind draws)) x))

(* A loop's key scope whose root raises: under jit, a constant-key scope's draw
   raises at the first draw of a trip. The step's own cleanup, and that of the
   constructs it entered, runs. *)
let raising_root_tests =
  [
    test
      "a trip's root that raises runs the step's finalisers under jit, inside \
       a construct" (fun () ->
        let cleaned = ref 0 in
        let f x =
          Nx.Rng.with_key (Nx.Rng.key 42) (fun () ->
              Rune.scan'
                ~f:(fun c e ->
                  Fun.protect
                    ~finally:(fun () -> incr cleaned)
                    (fun () ->
                      let g =
                        Rune.grad'
                          (fun e -> Nx.sum (Nx.mul e (Nx.rand f64 [||])))
                          e
                      in
                      (Nx.add c g, c)))
                ~init:(scalar 0.) x)
          |> fst
        in
        raises_match
          (function Rune.Jit_error _ -> true | _ -> false)
          (fun () -> Rune.jit' f (vec [| 1.; 2.; 3. |]));
        equal ~msg:"finalisers run" int 1 !cleaned);
  ]

let () =
  exit
    (run "Rune bodies"
       [
         group "law 1" [ law1 ];
         group "law 5" [ law5 ];
         group "a raising root" raising_root_tests;
         group "the boundary" boundary_tests;
         group "passing installations" passing_tests;
         group "roots in total scopes" root_tests;
         group "records" record_tests;
         group "handlers" handler_tests;
       ])
