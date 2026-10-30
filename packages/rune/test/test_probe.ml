(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune.Probe: what a collect receives, under every construct and
   transformation, against the emits a sequential run makes. *)

open Windtrap
open Compose
module T = Nx.Ptree
module D = Nx.Dtype
module Probe = Rune.Probe

let f64 = Nx.float64
let vec xs = Nx.create f64 [| Array.length xs |] xs

(* Offers *)

(* A probe of a site's tag, which the tests number in the order of their emits,
   beside a value. *)
let tagged () = T.(pair tensor tensor)
let tag k = Nx.scalar Nx.int32 (Int32.of_int k)

(* An offer as its tag, the same in every row, and its value's bits. *)
let seen (type s) ((t, x) : _ * (float, s, _) Nx.t) =
  let tags = Array.map Int32.to_int (Nx.to_array t) in
  let k = if tags = [||] then -1 else tags.(0) in
  if Array.exists (( <> ) k) tags then failf "an offer of two sites: %d" k;
  (k, bits x)

let offers_t = list (pair int bits_t)

(* The offer of the site [k] whose executions gave [rows], in order. *)
let offer k rows = (k, bits (Nx.stack ~axis:0 rows))

(* The outcome of [f ()] and the offers to [p] it made. *)
let collected p f =
  let r, offers = Probe.collect p f in
  let r =
    match r with
    | Ok v -> Value v
    | Error (e, _) -> Raised (Printexc.to_string e)
  in
  (r, List.map seen offers)

let offers p f = snd (collected p f)
let invalid ~msg ~sub f = raises_match ~msg (Exn.invalid_arg ~sub) f

(* Eager emits *)

let test_closed () =
  let p = Probe.make (tagged ()) and q = Probe.make (tagged ()) in
  let x = vec [| 1.; 2. |] in
  Probe.emit p (tag 0, x);
  equal ~msg:"another probe's collect" offers_t []
    (offers q (fun () -> Probe.emit p (tag 0, x)));
  equal ~msg:"a collect after the emit" offers_t [] (offers p ignore)

let test_eager () =
  let p = Probe.make (tagged ()) in
  let x = vec [| 1.; Float.nan |] and s = Nx.scalar f64 (-0.) in
  let e = Nx.zeros f64 [| 0; 2 |] in
  let r, got =
    collected p (fun () ->
        Probe.emit p (tag 0, x);
        Probe.emit p (tag 1, s);
        Probe.emit p (tag 2, e);
        3)
  in
  equal ~msg:"result" (outcome_t int) (Value 3) r;
  equal ~msg:"one row each" offers_t
    [ offer 0 [ x ]; offer 1 [ s ]; offer 2 [ e ] ]
    got

let test_nested () =
  let p = Probe.make (tagged ()) in
  let x = vec [| 1. |] in
  let inner = ref [] in
  let outer =
    offers p (fun () ->
        Probe.emit p (tag 0, x);
        inner := offers p (fun () -> Probe.emit p (tag 1, x));
        Probe.emit p (tag 2, x))
  in
  equal ~msg:"inner" offers_t [ offer 1 [ x ] ] !inner;
  equal ~msg:"outer" offers_t [ offer 0 [ x ]; offer 2 [ x ] ] outer

let test_raise () =
  let p = Probe.make (tagged ()) in
  let x = vec [| 1. |] in
  let r, got =
    collected p (fun () ->
        Probe.emit p (tag 0, x);
        invalid_arg "stop")
  in
  equal ~msg:"outcome" (outcome_t pass) (Raised "Invalid_argument(\"stop\")") r;
  equal ~msg:"offers" offers_t [ offer 0 [ x ] ] got

(* Scans *)

(* Rows of [s] at [dt], [n] of them, a scan over them emitting at two sites each
   step. *)
type scan =
  | Scan : {
      dt : (float, 's) Nx.dtype;
      s : int array;
      n : int;
      xs : operand;
    }
      -> scan

let scans =
  let open Gen in
  (let* s = shape in
   let* n = int_range 0 3 in
   let* xs = operand (Array.append [| n |] s) in
   let+ (D.Any dt) = of_list [ D.Any Nx.float32; Any f64 ] in
   match D.kind dt with Float -> Scan { dt; s; n; xs } | _ -> assert false)
  |> with_pp (fun ppf (Scan c) ->
      Format.fprintf ppf "%a %d rows of %a" Nx.pp_dtype c.dt c.n pp_shape c.s)

(* A step emitting its carry at site 0 and its row's square at site 1. *)
let step p c r =
  Probe.emit p (tag 0, c);
  let y = Nx.mul r r in
  Probe.emit p (tag 1, y);
  (Nx.add c r, y)

let scanned p xs =
  let row = Array.sub (Nx.shape xs) 1 (Nx.ndim xs - 1) in
  let c, ys =
    Rune.scan T.tensor T.tensor T.tensor (step p)
      ~init:(Nx.zeros (Nx.dtype xs) row)
      xs
  in
  Nx.add (Nx.sum c) (Nx.sum ys)

(* The emits of the scan's steps written out. *)
let scan_offers xs =
  let n = (Nx.shape xs).(0) in
  if n = 0 then []
  else
    let row = Array.sub (Nx.shape xs) 1 (Nx.ndim xs - 1) in
    let c = ref (Nx.zeros (Nx.dtype xs) row) and cs = ref [] and ys = ref [] in
    for k = 0 to n - 1 do
      let r = Nx.get [ k ] xs in
      cs := !c :: !cs;
      ys := Nx.mul r r :: !ys;
      c := Nx.add !c r
    done;
    [ offer 0 (List.rev !cs); offer 1 (List.rev !ys) ]

(* Each way of running [f] that runs its primal once. *)
type mode = {
  name : string;
  run :
    's.
    ((float, 's, Nx.host) Nx.t -> (float, 's, Nx.host) Nx.t) ->
    (float, 's, Nx.host) Nx.t ->
    unit;
}

let modes =
  [
    { name = "eager"; run = (fun f x -> ignore (f x)) };
    { name = "jit"; run = (fun f x -> ignore (Rune.jit' f x)) };
    { name = "jvp"; run = (fun f x -> ignore (Rune.jvp' f x x)) };
    { name = "grad"; run = (fun f x -> ignore (Rune.grad' f x)) };
    {
      name = "vjp and its pullback";
      run =
        (fun f x ->
          let y, pb = Rune.vjp' f x in
          ignore (pb (Nx.ones_like y)));
    };
    {
      name = "grad of a remat";
      run =
        (fun f x ->
          ignore (Rune.grad' (Rune.remat T.(tensor @-> returns tensor) f) x));
    };
    {
      name = "jit of grad";
      run = (fun f x -> ignore (Rune.jit' (Rune.grad' f) x));
    };
    {
      name = "grad of jit";
      run = (fun f x -> ignore (Rune.grad' (Rune.jit' f) x));
    };
    {
      name = "a map of one lane";
      run =
        (fun f x ->
          ignore
            (Rune.vmap' f (Nx.reshape (Array.append [| 1 |] (Nx.shape x)) x)));
    };
  ]

let law_scan (Scan c) =
  cover "no row" (c.n = 0);
  cover "several rows" (c.n > 1);
  cover "zero-size rows" (numel c.s = 0 && c.n > 0);
  let xs = make c.dt (Array.append [| c.n |] c.s) c.xs in
  let expected = scan_offers xs in
  List.iter
    (fun m ->
      let p = Probe.make (tagged ()) in
      equal ~msg:m.name offers_t expected
        (offers p (fun () -> m.run (scanned p) xs)))
    modes

let test_scan_raise () =
  let p = Probe.make (tagged ()) in
  let xs = vec [| 1.; 2.; Float.nan; 4. |] in
  let f c r =
    Probe.emit p (tag 0, r);
    Nx.check T.tensor (Nx.logical_not (Nx.isnan r)) r (fun _ _ -> Exit);
    Probe.emit p (tag 1, r);
    (Nx.add c r, r)
  in
  let scan xs =
    snd (Rune.scan T.tensor T.tensor T.tensor f ~init:(Nx.scalar f64 0.) xs)
  in
  let row k = Nx.get [ k ] xs in
  List.iter
    (fun (mode, run) ->
      let r, got = collected p (fun () -> run xs) in
      equal ~msg:mode (outcome_t pass) (Raised "Stdlib.Exit") r;
      equal ~msg:mode offers_t
        [ offer 0 [ row 0; row 1; row 2 ]; offer 1 [ row 0; row 1 ] ]
        got)
    [ ("eager", scan); ("jit", Rune.jit' scan) ]

(* A scan in a scan's step: the inner rows of each outer row, in turn. *)
let test_nested_scan () =
  let p = Probe.make (tagged ()) in
  let xs = Nx.reshape [| 2; 3 |] (Nx.arange_f f64 0. 6. 1.) in
  let inner c r =
    Probe.emit p (tag 1, r);
    (Nx.add c r, r)
  in
  let outer c row =
    Probe.emit p (tag 0, c);
    let c', _ = Rune.scan T.tensor T.tensor T.tensor inner ~init:c row in
    (c', c')
  in
  let scan xs =
    fst (Rune.scan T.tensor T.tensor T.tensor outer ~init:(Nx.scalar f64 0.) xs)
  in
  let e k = Nx.scalar f64 k in
  let expected =
    [ offer 0 [ e 0.; e 3. ]; offer 1 [ e 0.; e 1.; e 2.; e 3.; e 4.; e 5. ] ]
  in
  equal ~msg:"eager" offers_t expected (offers p (fun () -> scan xs));
  let before = Rune.evaluations () in
  equal ~msg:"jit" offers_t expected (offers p (fun () -> Rune.jit' scan xs));
  equal ~msg:"compiled" int 0 (Rune.evaluations () - before)

(* Maps *)

(* Each lane's offers, run one by one, as a map offers them: a site's rows of
   every lane, lane 0's first, the sites in the order of their tags. *)
let lane_major lanes =
  let tags =
    List.sort_uniq Int.compare (List.concat_map (List.map fst) lanes)
  in
  List.map
    (fun k ->
      let rows = List.concat_map (List.filter (fun (j, _) -> j = k)) lanes in
      let shapes = List.map (fun (_, (s, _)) -> s) rows in
      let s = List.hd shapes in
      let n = List.fold_left (fun a s -> a + s.(0)) 0 shapes in
      let s = Array.append [| n |] (Array.sub s 1 (Array.length s - 1)) in
      (k, (s, Array.concat (List.map (fun (_, (_, b)) -> b) rows))))
    tags

(* Each way of running a map [f] of lanes [x]. *)
let map_modes =
  [
    { name = "eager"; run = (fun f x -> ignore (Rune.vmap' f x)) };
    {
      name = "jit of the map";
      run = (fun f x -> ignore (Rune.jit' (Rune.vmap' f) x));
    };
    {
      name = "the map of jit";
      run = (fun f x -> ignore (Rune.vmap' (Rune.jit' f) x));
    };
    {
      name = "grad of the map";
      run =
        (fun f x -> ignore (Rune.grad' (fun x -> Nx.sum (Rune.vmap' f x)) x));
    };
    {
      name = "jvp of the map";
      run = (fun f x -> ignore (Rune.jvp' (Rune.vmap' f) x x));
    };
  ]

(* [f p] mapped over the lanes of [x] in every mode offers each lane's offers,
   lane-major. *)
let mapped ~f x =
  let n = (Nx.shape x).(0) in
  let lane k =
    let p = Probe.make (tagged ()) in
    offers p (fun () -> ignore (f p (Nx.get [ k ] x)))
  in
  let expected = lane_major (List.init n lane) in
  List.iter
    (fun m ->
      let p = Probe.make (tagged ()) in
      let got = offers p (fun () -> m.run (f p) x) in
      let got = List.stable_sort (fun (a, _) (b, _) -> Int.compare a b) got in
      equal ~msg:m.name offers_t expected got)
    map_modes

(* Lanes of two rows each, and of a trip count beside an array. *)
type lanes =
  | Lanes : {
      dt : (float, 's) Nx.dtype;
      s : int array;
      trips : int array;
      max : int;
      rows : operand;
      x : operand;
    }
      -> lanes

let lanes =
  let open Gen in
  (let* s = shape in
   let* n = int_range 0 3 in
   let* trips = array ~size:(constant n) (int_range 0 3) in
   let most = Array.fold_left Int.max 0 trips in
   let* max = of_list [ most; most + 2 ] in
   let* rows = operand (Array.concat [ [| n; 2 |]; s ]) in
   let* x = operand [| n; numel s |] in
   let+ (D.Any dt) = of_list [ D.Any Nx.float32; Any f64 ] in
   match D.kind dt with
   | Float -> Lanes { dt; s; trips; max; rows; x }
   | _ -> assert false)
  |> with_pp (fun ppf (Lanes c) ->
      Format.fprintf ppf "%a %a, trips [%s], max %d" Nx.pp_dtype c.dt pp_shape
        c.s
        (String.concat "; " (Array.to_list (Array.map string_of_int c.trips)))
        c.max)

(* An iterate over a lane [z], its trip count then its array's elements, whose
   step emits the carry at site 0 and its double at site 1. *)
let iterated s ~max p z =
  let m = (Nx.shape z).(0) in
  let t = Nx.get [ 0 ] z and x = Nx.reshape s (Nx.slice [ R (1, m) ] z) in
  let x, _ =
    Rune.iterate
      T.(pair tensor tensor)
      ~max
      ~until:(fun (_, k) -> Nx.greater_equal k t)
      (fun (x, k) ->
        Probe.emit p (tag 0, x);
        let x2 = Nx.add x x in
        Probe.emit p (tag 1, x2);
        (x2, Nx.add_s k 1.))
      (x, Nx.zeros_like t)
  in
  Nx.sum x

let law_lanes (Lanes c) =
  let n = Array.length c.trips in
  cover "no lane" (n = 0);
  cover "lanes stopping apart"
    (n > 1 && Array.exists (( <> ) c.trips.(0)) c.trips);
  cover "a lane with no step" (Array.exists (( = ) 0) c.trips);
  mapped ~f:scanned (make c.dt (Array.concat [ [| n; 2 |]; c.s ]) c.rows);
  let t = Nx.create c.dt [| n; 1 |] (Array.map float_of_int c.trips) in
  let z = Nx.concatenate ~axis:1 [ t; make c.dt [| n; numel c.s |] c.x ] in
  mapped ~f:(iterated c.s ~max:c.max) z

(* A switch over a lane [z], its branch index then its array's elements: branch
   0 emits at site 0, branch 1 at site 1, branch 2 nowhere. *)
let switched s p z =
  let m = (Nx.shape z).(0) in
  let i = Nx.cast Nx.int32 (Nx.get [ 0 ] z) in
  let x = Nx.reshape s (Nx.slice [ R (1, m) ] z) in
  let branch k f () =
    let y = f x in
    if k < 2 then Probe.emit p (tag k, y);
    y
  in
  Nx.sum
    (Rune.switch T.tensor i
       [
         branch 0 (fun x -> Nx.add x x);
         branch 1 (fun x -> Nx.mul x x);
         branch 2 Nx.neg;
       ])

type branches =
  | Branches : {
      dt : (float, 's) Nx.dtype;
      s : int array;
      index : int array;
      x : operand;
    }
      -> branches

let branches =
  let open Gen in
  (let* s = shape in
   let* n = int_range 0 3 in
   let* index = array ~size:(constant n) (int_range 0 2) in
   let* x = operand [| n; numel s |] in
   let+ (D.Any dt) = of_list [ D.Any Nx.float32; Any f64 ] in
   match D.kind dt with
   | Float -> Branches { dt; s; index; x }
   | _ -> assert false)
  |> with_pp (fun ppf (Branches c) ->
      Format.fprintf ppf "%a %a, branches [%s]" Nx.pp_dtype c.dt pp_shape c.s
        (String.concat "; " (Array.to_list (Array.map string_of_int c.index))))

let law_branches (Branches c) =
  let n = Array.length c.index in
  cover "lanes in different branches"
    (n > 1 && Array.exists (( <> ) c.index.(0)) c.index);
  cover "no lane" (n = 0);
  let i = Nx.create c.dt [| n; 1 |] (Array.map float_of_int c.index) in
  let z = Nx.concatenate ~axis:1 [ i; make c.dt [| n; numel c.s |] c.x ] in
  mapped ~f:(switched c.s) z;
  List.iter
    (fun m ->
      for k = 0 to n - 1 do
        let p = Probe.make (tagged ()) in
        let zk = Nx.get [ k ] z in
        let expected = offers p (fun () -> ignore (switched c.s p zk)) in
        let p = Probe.make (tagged ()) in
        equal ~msg:m.name offers_t expected
          (offers p (fun () -> m.run (switched c.s p) zk))
      done)
    modes

(* A map in a scan's step: each row's lanes, row by row. *)
let test_map_in_scan () =
  let p = Probe.make (tagged ()) in
  let xs = Nx.reshape [| 2; 3 |] (Nx.arange_f f64 0. 6. 1.) in
  let step c row =
    let y =
      Rune.vmap'
        (fun x ->
          Probe.emit p (tag 0, x);
          Nx.add x c)
        row
    in
    (Nx.add c (Nx.sum y), y)
  in
  let got =
    offers p (fun () ->
        Rune.scan T.tensor T.tensor T.tensor step ~init:(Nx.scalar f64 0.) xs)
  in
  equal offers_t
    [ offer 0 (List.init 6 (fun k -> Nx.scalar f64 (float_of_int k))) ]
    got

(* Maps in maps: the outer lane major. *)
let test_nested_maps () =
  let p = Probe.make (tagged ()) in
  let x = Nx.reshape [| 2; 3 |] (Nx.arange_f f64 0. 6. 1.) in
  let f =
    Rune.vmap'
      (Rune.vmap' (fun x ->
           Probe.emit p (tag 0, x);
           x))
  in
  let expected =
    [ offer 0 (List.init 6 (fun k -> Nx.scalar f64 (float_of_int k))) ]
  in
  equal ~msg:"eager" offers_t expected (offers p (fun () -> f x));
  equal ~msg:"jit" offers_t expected (offers p (fun () -> Rune.jit' f x))

let test_until () =
  let p = Probe.make (tagged ()) in
  let iterate () =
    Rune.iterate' ~max:3
      ~until:(fun x ->
        Probe.emit p (tag 0, x);
        Nx.greater_s x 2.)
      (fun x -> Nx.add_s x 1.)
      (Nx.scalar f64 0.)
  in
  equal ~msg:"no collect open" bits_t
    (bits (Nx.scalar f64 3.))
    (bits (iterate ()));
  (match collected p iterate with
  | Raised m, [] -> contains ~sub:"Rune.iterate': until emits to a probe" m
  | _ -> fail "an emit in until, a collect open: no raise");
  let in_scan () =
    Rune.iterate' ~max:3
      ~until:(fun x ->
        let c, _ =
          Rune.scan' ~init:x
            (fun c r ->
              Probe.emit p (tag 0, c);
              (Nx.add c r, r))
            (Nx.reshape [| 1 |] x)
        in
        Nx.greater_s c 2.)
      (fun x -> Nx.add_s x 1.)
      (Nx.scalar f64 0.)
  in
  match collected p in_scan with
  | Raised m, [] -> contains ~sub:"until emits to a probe" m
  | _ -> fail "a scan emitting in until, a collect open: no raise"

(* Derivatives *)

(* [x²] whose tangent map emits the tangent at site 1. *)
let square p =
  Rune.custom_jvp T.tensor T.tensor (fun x ->
      Probe.emit p (tag 0, x);
      ( Nx.mul x x,
        fun dx ->
          Probe.emit p (tag 1, dx);
          Nx.mul (Nx.mul_s x 2.) dx ))

let test_tangent () =
  let p = Probe.make (tagged ()) in
  let x = vec [| 3. |] and t = vec [| 1. |] in
  equal ~msg:"forward mode offers the tangent" offers_t
    [ offer 0 [ x ]; offer 1 [ t ] ]
    (offers p (fun () -> Rune.jvp' (square p) x t));
  equal ~msg:"reverse mode, no collect open" bits_t
    (bits (vec [| 6. |]))
    (bits (Rune.grad' (fun x -> Nx.sum (square p x)) x));
  let refused msg f =
    match collected p f with
    | Raised m, _ -> contains ~msg ~sub:"Rune.Probe.emit" m
    | Value _, _ -> failf "%s: no raise" msg
  in
  refused "reverse mode, a collect open" (fun () ->
      Rune.grad' (fun x -> Nx.sum (square p x)) x);
  let stepped x =
    let c, _ =
      Rune.scan' ~init:x
        (fun c r -> (Nx.add (square p c) r, r))
        (Nx.reshape [| 1; 1 |] x)
    in
    Nx.sum c
  in
  equal ~msg:"forward mode in a step" offers_t
    [ offer 0 [ x ]; offer 1 [ t ] ]
    (offers p (fun () -> Rune.jvp' stepped x t));
  refused "reverse mode in a step" (fun () -> Rune.grad' stepped x);
  refused "reverse mode in a compiled step" (fun () ->
      Rune.grad' (Rune.jit' stepped) x)

(* A custom_vjp whose pullback emits at site 1: no backward pass offers. *)
let test_backward () =
  let p = Probe.make (tagged ()) in
  let f =
    Rune.custom_vjp T.tensor T.tensor (fun x ->
        Probe.emit p (tag 0, x);
        ( Nx.mul x x,
          fun ct ->
            Probe.emit p (tag 1, ct);
            Nx.mul (Nx.mul_s x 2.) ct ))
  in
  let x = vec [| 3. |] in
  equal ~msg:"grad" offers_t
    [ offer 0 [ x ] ]
    (offers p (fun () -> Rune.grad' (fun x -> Nx.sum (f x)) x));
  let _, pb = Rune.vjp' f x in
  equal ~msg:"a pullback applied" offers_t []
    (offers p (fun () -> pb (vec [| 1. |])));
  let _, pb = Rune.vjp' (fun x -> scanned p (Nx.reshape [| 1; 1 |] x)) x in
  equal ~msg:"a scan's pullback applied" offers_t []
    (offers p (fun () -> pb (Nx.scalar f64 1.)))

(* A tangent map's emission of a residual, and a linear function's. *)
let test_rules () =
  let p = Probe.make (tagged ()) in
  let f =
    Rune.custom_jvp T.tensor T.tensor (fun x ->
        ( Nx.mul x x,
          fun dx ->
            let s = Nx.mul_s x 2. in
            Probe.emit p (tag 0, s);
            Nx.mul s dx ))
  in
  let x = vec [| 3. |] in
  equal ~msg:"a residual emitted under reverse mode" offers_t
    [ offer 0 [ vec [| 6. |] ] ]
    (offers p (fun () -> Rune.grad' (fun x -> Nx.sum (f x)) x));
  (* The tangent read before the residual is first used. *)
  let g =
    Rune.custom_jvp T.tensor T.tensor (fun x ->
        ( Nx.mul x x,
          fun dx ->
            let t = Nx.mul dx dx in
            let s = Nx.sin x in
            Probe.emit p (tag 0, s);
            Nx.add (Nx.mul (Nx.mul_s x 2.) dx) (Nx.mul s t) ))
  in
  equal ~msg:"a residual emitted after a tangent's use" offers_t
    [ offer 0 [ Nx.sin x ] ]
    (offers p (fun () -> Rune.grad' (fun x -> Nx.sum (g x)) x));
  let lin x =
    Probe.emit p (tag 0, x);
    Nx.mul_s x 2.
  in
  let tr = Rune.linear_transpose T.tensor T.tensor lin x in
  equal ~msg:"a linear transpose" offers_t []
    (offers p (fun () -> tr (vec [| 1. |])));
  let ct =
    Rune.custom_transpose T.tensor T.tensor
      (fun x ->
        Probe.emit p (tag 0, x);
        Nx.mul_s x 2.)
      ~transpose:(fun c -> Nx.mul_s c 2.)
  in
  equal ~msg:"a custom transpose under jvp" offers_t
    [ offer 0 [ x ] ]
    (offers p (fun () -> Rune.jvp' ct x (vec [| 1. |])))

(* A root whose solve and linear solve emit: the solve once, the derivative
   nothing. *)
let test_root () =
  let p = Probe.make (tagged ()) in
  let f a =
    Rune.root T.tensor
      ~linear_solve:(fun op b ->
        Probe.emit p (tag 2, b);
        Nx.div b (op (Nx.ones_like b)))
      ~residual:(fun x ->
        Probe.emit p (tag 1, x);
        Nx.sub (Nx.mul a x) (vec [| 6. |]))
      (fun () ->
        let x = Nx.div (vec [| 6. |]) a in
        Probe.emit p (tag 0, x);
        x)
  in
  let a = vec [| 2. |] in
  let once = [ offer 0 [ vec [| 3. |] ] ] in
  equal ~msg:"eager" offers_t once (offers p (fun () -> f a));
  equal ~msg:"jvp" offers_t once (offers p (fun () -> Rune.jvp' f a a));
  equal ~msg:"grad" offers_t once
    (offers p (fun () -> Rune.grad' (fun a -> Nx.sum (f a)) a))

(* Extents *)

let test_inside () =
  let p = Probe.make (tagged ()) in
  let x = vec [| 1. |] in
  let inner x =
    ignore (Probe.collect p (fun () -> Probe.emit p (tag 0, x)));
    x
  in
  let refused msg f =
    invalid ~msg ~sub:"Rune.Probe.collect: inside" (fun () -> ignore (f ()))
  in
  refused "jit" (fun () -> Rune.jit' inner x);
  refused "a map" (fun () -> Rune.vmap' inner (Nx.reshape [| 1; 1 |] x));
  refused "jvp" (fun () -> Rune.jvp' inner x x);
  refused "grad" (fun () -> Rune.grad' (fun x -> Nx.sum (inner x)) x);
  refused "a scan's step" (fun () ->
      Rune.scan' ~init:x (fun c r -> (inner c, r)) (Nx.reshape [| 1; 1 |] x))

let test_domain () =
  let p = Probe.make (tagged ()) in
  let x = vec [| 1. |] in
  equal offers_t []
    (offers p (fun () ->
         Domain.join (Domain.spawn (fun () -> Probe.emit p (tag 0, x)))))

(* A total and a probe in one compiled step. *)
let test_total () =
  let p = Probe.make (tagged ()) and total = Rune.Total.make () in
  let xs = Nx.reshape [| 3; 1 |] (Nx.arange_f f64 1. 4. 1.) in
  let f xs =
    let c, _ =
      Rune.scan' ~init:(Nx.zeros f64 [| 1 |])
        (fun c r ->
          Rune.Total.add total r;
          Probe.emit p (tag 0, c);
          (Nx.add c r, r))
        xs
    in
    c
  in
  let run f =
    collected p (fun () ->
        let c, t =
          Rune.Total.collect total ~init:(Nx.zeros f64 [| 1 |]) (fun () -> f xs)
        in
        (bits c, bits t))
  in
  equal ~msg:"jit"
    (pair (outcome_t (pair bits_t bits_t)) offers_t)
    (run f)
    (run (Rune.jit' f))

(* Jit *)

let test_jit_keys () =
  let p = Probe.make (tagged ()) in
  let f =
    Rune.jit' (fun x ->
        Probe.emit p (tag 0, x);
        Nx.add x x)
  in
  let x = vec [| 1.; 2. |] in
  let y = bits (Nx.add x x) in
  equal ~msg:"no collect open" bits_t y (bits (f x));
  let r, got = collected p (fun () -> bits (f x)) in
  equal ~msg:"a collect opened since" (outcome_t bits_t) (Value y) r;
  equal ~msg:"its offers" offers_t [ offer 0 [ x ] ] got;
  equal ~msg:"two calls" offers_t
    [ offer 0 [ x ]; offer 0 [ x ] ]
    (offers p (fun () ->
         ignore (f x);
         ignore (f x)));
  equal ~msg:"closed again" bits_t y (bits (f x));
  (* Under maps of several lane counts, with two probes, each call offers what
     the function does eagerly: collects open or closed, lanes repeated, met
     again or new, in turn. *)
  let a = Rune.axis () and q = Probe.make (tagged ()) in
  let g x =
    let s = Nx.sum (Rune.lanes a x) in
    Probe.emit p (tag 1, Nx.add x s);
    Probe.emit q (tag 2, s);
    Nx.mul x s
  in
  let jg = Rune.jit' g in
  let collect_if opened pr f =
    if not opened then (f (), [])
    else
      let r, o = Probe.collect pr f in
      ((match r with Ok v -> v | Error (e, _) -> raise e), List.map seen o)
  in
  let call f (on_p, on_q, n) =
    let xs = Nx.reshape [| n; 2 |] (Nx.arange_f f64 0. (Float.of_int (2 * n)) 1.) in
    let (y, at_p), at_q =
      collect_if on_q q (fun () ->
          collect_if on_p p (fun () -> bits (Rune.vmap' ~axis:a f xs)))
    in
    (y, at_p, at_q)
  in
  List.iter
    (fun ((on_p, on_q, n) as c) ->
      let msg = Printf.sprintf "%d lanes, collects %b %b" n on_p on_q in
      equal ~msg (triple bits_t offers_t offers_t) (call g c) (call jg c))
    [
      (false, false, 2); (true, false, 2); (true, false, 3); (true, false, 2);
      (true, true, 2); (true, false, 2); (true, true, 3); (false, false, 3);
      (false, true, 3); (true, true, 3); (false, false, 2);
    ]

(* A compiled program offers its scan's rows as more results: nx evaluates
   nothing. *)
let test_compiled () =
  let p = Probe.make (tagged ()) in
  let xs = Nx.reshape [| 3; 2 |] (Nx.arange_f f64 0. 6. 1.) in
  let f = Rune.jit' (fun xs -> Nx.add_s (scanned p xs) 1.) in
  let before = Rune.evaluations () in
  let got = offers p (fun () -> f xs) in
  equal ~msg:"offers" offers_t (scan_offers xs) got;
  equal ~msg:"evaluated calls" int 0 (Rune.evaluations () - before)

let () =
  exit
    (run "rune probe"
       [
         group "eager"
           [
             test "an emit with no collect open does nothing" test_closed;
             test "each emit offers one row" test_eager;
             test "the innermost collect receives" test_nested;
             test "a raise keeps the offers" test_raise;
           ];
         group "scans"
           [
             prop "a step offers a row per row, in every mode" scans law_scan;
             test "a raise keeps the rows before" test_scan_raise;
             test "a scan in a step" test_nested_scan;
           ];
         group "maps"
           [
             prop "lanes offer lane-major, each its own steps" lanes law_lanes;
             test "an emit in an iterate's until" test_until;
             prop "lanes offer the branches they take" branches law_branches;
             test "a map in a scan's step" test_map_in_scan;
             test "maps in maps" test_nested_maps;
           ];
         group "derivatives"
           [
             test "a tangent emitted under reverse mode" test_tangent;
             test "backward passes offer nothing" test_backward;
             test "custom rules' residuals and linear functions" test_rules;
             test "a root's derivative offers nothing" test_root;
           ];
         group "extents"
           [
             test "a collect inside a transformation raises" test_inside;
             test "an emit on another domain does nothing" test_domain;
           ];
         group "jit"
           [
             test "a total beside a probe" test_total;
             test "a collect opened or closed retraces" test_jit_keys;
             test "a compiled program offers" test_compiled;
           ];
       ])
