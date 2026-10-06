(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx's special functions compiled, against the goldens nx's suite holds them to
   (nx's golden/special/, written by its gen/special.py), on the host and on the
   Metal device where the machine has one. Metal computes no float64 and flushes
   float32 subnormals, so it is held to the float32 rows whose arguments and
   value are normal. *)

open Windtrap
open Nx_test.Special

let golden name = "../../nx/test/golden/special/" ^ name ^ ".golden"

type u = { u : 'b. (float, 'b) Nx.t -> (float, 'b) Nx.t }

(* [compiled at u] is [u] compiled, its argument placed at [at] and its result
   read back on the host. *)
let compiled at { u } =
  {
    f =
      (fun a ->
        let x = Nx.place at a.(0) in
        Nx.place Nx.Placement.host (Rune.jit' u x));
  }

(* [compiled2 at g] is [g] of two arguments, compiled likewise. *)
type b = { b : 'b. (float, 'b) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t }

let compiled2 at { b } =
  {
    f =
      (fun a ->
        let f = Rune.jit Nx.Ptree.(tensor @-> tensor @-> returns tensor) b in
        Nx.place Nx.Placement.host (f (Nx.place at a.(0)) (Nx.place at a.(1))));
  }

let metal = Result.to_option (Nx_metal.get 0)

let on_devices name ~bound compile =
  let host = check ~bound (golden name) (compile Nx.Placement.host) in
  let metal =
    match metal with
    | Some m ->
        check
          ~keep:(fun r -> not (subnormal r))
          ~dtypes:[ `F32 ] ~bound (golden name)
          (compile (Nx.Placement.on m))
    | None -> [ test "metal" (fun () -> skip ~reason:"no Metal device" ()) ]
  in
  group name [ group "on the host" host; group "on Metal" metal ]

let unary name ~bound u = on_devices name ~bound (fun at -> compiled at u)
let everywhere b _ = b

let by_sign ~positive ~negative args =
  if args.(0) < 0. then negative else positive

let error_function =
  group "error function"
    [
      unary "erf" ~bound:(everywhere (Ulps 2)) { u = Nx.erf };
      unary "erfinv" ~bound:(everywhere (Inverse (4, 8))) { u = Nx.erfinv };
      unary "erfc" ~bound:(everywhere (Ulps 8)) { u = Nx.erfc };
    ]

let normal =
  group "standard normal"
    [
      unary "ndtr" ~bound:(everywhere (Ulps 16)) { u = Nx.ndtr };
      unary "log_ndtr" ~bound:(everywhere (Ulps 32)) { u = Nx.log_ndtr };
      unary "ndtri" ~bound:(everywhere (Inverse (4, 16))) { u = Nx.ndtri };
    ]

let gamma =
  group "gamma"
    [
      unary "lgamma"
        ~bound:(by_sign ~positive:(Near_zeros (16, 16)) ~negative:(Scaled 16))
        { u = Nx.lgamma };
      unary "digamma"
        ~bound:(by_sign ~positive:(Near_zeros (16, 16)) ~negative:(Scaled 16))
        { u = Nx.digamma };
      on_devices "lbeta"
        ~bound:(everywhere (Near_zeros (256, 512)))
        (fun at -> compiled2 at { b = Nx.lbeta });
    ]

let bessel =
  group "modified Bessel"
    [
      unary "i0e" ~bound:(everywhere (Ulps 8)) { u = Nx.i0e };
      unary "i1e" ~bound:(everywhere (Ulps 8)) { u = Nx.i1e };
    ]

(* Derivatives

   rune differentiates each function through the operations nx computes it with.
   Each derivative is held to golden/special_grad/ (written by
   gen/special_grad.py) eagerly and compiled on the host: within 16 times its
   function's bound, and second derivatives within [2^-40] relative at float64
   and [2^-16] at float32. *)

let grad_golden name = "golden/special_grad/" ^ name ^ ".golden"

(* What a derivative's float64 compile costs: an inverse's runs under the [slow]
   tag, which [dune build @packages/rune/test/slow] runs and [runtest] does
   not. *)
type cost = Quick | Slow_at_float64

let compiled_checks cost ~bound file f =
  match cost with
  | Quick -> check ~zeros:`Unsigned ~bound file f
  | Slow_at_float64 ->
      check ~zeros:`Unsigned ~dtypes:[ `F32 ] ~bound file f
      @ check ~zeros:`Unsigned ~dtypes:[ `F64 ] ~tags:[ "slow" ] ~bound file f

(* [d] eagerly and compiled, at both dtypes. *)
let derivative ?(cost = Quick) name ~bound (d : u) =
  let eager = { f = (fun a -> d.u a.(0)) } in
  let compiled = { f = (fun a -> Rune.jit' d.u a.(0)) } in
  group name
    [
      group "eagerly" (check ~zeros:`Unsigned ~bound (grad_golden name) eager);
      group "compiled" (compiled_checks cost ~bound (grad_golden name) compiled);
    ]

let derivative2 name ~bound (d : b) =
  let eager = { f = (fun a -> d.b a.(0) a.(1)) } in
  let compiled =
    {
      f =
        (fun a ->
          Rune.jit
            Nx.Ptree.(tensor @-> tensor @-> returns tensor)
            d.b a.(0) a.(1));
    }
  in
  group name
    [
      group "eagerly" (check ~zeros:`Unsigned ~bound (grad_golden name) eager);
      group "compiled"
        (check ~zeros:`Unsigned ~bound (grad_golden name) compiled);
    ]

let d { u } = { u = (fun x -> Rune.grad' (fun x -> Nx.sum (u x)) x) }
let second = Relative (0x1p-40, 0x1p-16)

(* [digamma]'s derivative, [1/x^2] to leading order, overflows to [+inf] as [x]
   nears 0, from either side. *)
let digamma_near_zero =
  let at (type b) (dt : (float, b) Nx.dtype) xs =
    let x = Nx.create dt [| Array.length xs |] xs in
    let g = d { u = Nx.digamma } in
    let infinite = Array.map (fun _ -> Float.infinity) xs in
    equal ~msg:"eagerly" (array float_exact) infinite
      (Nx.to_array (Nx.cast Nx.float64 (g.u x)));
    equal ~msg:"compiled" (array float_exact) infinite
      (Nx.to_array (Nx.cast Nx.float64 (Rune.jit' g.u x)))
  in
  test "digamma's derivative is +inf beside 0" (fun () ->
      at Nx.float64 [| -1e-300; -1e-160; 1e-300 |];
      at Nx.float32 [| -1e-20; -1e-40; 1e-20 |])

(* [lbeta]'s derivative in [first], then in [second]. *)
let lbeta_d2 first second a b =
  let d1 a b =
    match first with
    | `A -> Rune.grad' (fun a -> Nx.sum (Nx.lbeta a b)) a
    | `B -> Rune.grad' (fun b -> Nx.sum (Nx.lbeta a b)) b
  in
  match second with
  | `A -> Rune.grad' (fun a -> Nx.sum (d1 a b)) a
  | `B -> Rune.grad' (fun b -> Nx.sum (d1 a b)) b

(* [log_ndtr x] is [-ndtr (-x)] to leading order for large [x], so its
   derivatives vanish there, also where [x^2] passes the largest float. *)
let log_ndtr_far =
  let at (type b) (dt : (float, b) Nx.dtype) xs =
    let x = Nx.create dt [| Array.length xs |] xs in
    let g = d (d { u = Nx.log_ndtr }) in
    let zeros = Array.map (fun _ -> 0.) xs in
    equal ~msg:"eagerly" (array float_exact) zeros
      (Nx.to_array (Nx.cast Nx.float64 (Nx.abs (g.u x))));
    equal ~msg:"compiled" (array float_exact) zeros
      (Nx.to_array (Nx.cast Nx.float64 (Nx.abs (Rune.jit' g.u x))))
  in
  test "log_ndtr's second derivative is 0 at large x" (fun () ->
      at Nx.float64 [| 1e100; 1e200; Float.max_float |];
      at Nx.float32 [| 1e10; 1e30; 0x1.fffffep127 |])

(* At [-0], [i0e]'s and [i1e]'s derivatives are their right-hand ones, those at
   [+0]. *)
let bessel_at_zero =
  let at (type b) (dt : (float, b) Nx.dtype) =
    let x = Nx.create dt [| 2 |] [| 0.; -0. |] in
    let check name { u } =
      let both y =
        match Nx.to_array (Nx.cast Nx.float64 y) with
        | [| p; n |] -> (p, n)
        | _ -> assert false
      in
      let p, n = both (u x) in
      equal ~msg:(name ^ " eagerly") float_exact p n;
      let p, n = both (Rune.jit' u x) in
      equal ~msg:(name ^ " compiled") float_exact p n
    in
    check "i0e" (d { u = Nx.i0e });
    check "i1e" (d { u = Nx.i1e })
  in
  test "i0e's and i1e's derivatives at -0 are those at +0" (fun () ->
      at Nx.float64;
      at Nx.float32)

let derivatives =
  group "derivatives"
    [
      digamma_near_zero;
      log_ndtr_far;
      bessel_at_zero;
      derivative "i0e" ~bound:(everywhere (Ulps 128)) (d { u = Nx.i0e });
      derivative "i1e"
        ~bound:(everywhere (Near_zeros (128, 128)))
        (d { u = Nx.i1e });
      derivative "erfc" ~bound:(everywhere (Ulps 128)) (d { u = Nx.erfc });
      derivative "ndtr" ~bound:(everywhere (Ulps 256)) (d { u = Nx.ndtr });
      derivative "log_ndtr" ~bound:(everywhere (Ulps 512))
        (d { u = Nx.log_ndtr });
      derivative ~cost:Slow_at_float64 "erfinv" ~bound:(everywhere (Ulps 64))
        (d { u = Nx.erfinv });
      derivative ~cost:Slow_at_float64 "ndtri" ~bound:(everywhere (Ulps 64))
        (d { u = Nx.ndtri });
      derivative "lgamma"
        ~bound:
          (by_sign ~positive:(Near_zeros (256, 256)) ~negative:(Scaled 256))
        (d { u = Nx.lgamma });
      derivative "digamma"
        ~bound:(by_sign ~positive:(Ulps 256) ~negative:(Scaled 256))
        (d { u = Nx.digamma });
      derivative2 "lbeta_a"
        ~bound:(everywhere (Near_zeros (4096, 8192)))
        { b = (fun a b -> Rune.grad' (fun a -> Nx.sum (Nx.lbeta a b)) a) };
      derivative2 "lbeta_b"
        ~bound:(everywhere (Near_zeros (4096, 8192)))
        { b = (fun a b -> Rune.grad' (fun b -> Nx.sum (Nx.lbeta a b)) b) };
      derivative "lgamma_2" ~bound:(everywhere second) (d (d { u = Nx.lgamma }));
      derivative2 "lbeta_aa" ~bound:(everywhere second)
        { b = (fun a b -> lbeta_d2 `A `A a b) };
      derivative2 "lbeta_bb" ~bound:(everywhere second)
        { b = (fun a b -> lbeta_d2 `B `B a b) };
      derivative2 "lbeta_ab" ~bound:(everywhere second)
        { b = (fun a b -> lbeta_d2 `A `B a b) };
    ]

(* Residuals

   The bytes per element an eager [vjp] of each function keeps for its pullback,
   at float64 over 4096 elements, on a test device that counts what it holds:
   what reverse mode costs through a composition of nx's operations. *)

let device = Nx.Device.cpu 5

let settled () =
  let m = Nx.Device.memory device in
  for _ = 1 to 4 do
    Gc.full_major ();
    Nx_device.synchronize m
  done;
  Nx_device.Stats.allocated (Nx_device.stats m)

let elements = 4096

let residual { u } lo hi =
  let x = Nx.linspace Nx.float64 lo hi elements in
  let x = Nx.place (Nx.Placement.on device) x in
  let before = settled () in
  let y, pullback = Rune.vjp' u x in
  let held = settled () - before in
  ignore (Sys.opaque_identity (y, pullback));
  held / elements

let residuals =
  test "an eager vjp keeps these bytes per element" (fun () ->
      let lbeta = { u = (fun a -> Nx.lbeta a (Nx.full_like a 3.5)) } in
      List.iter
        (fun (name, u, lo, hi) ->
          Printf.printf "%s %d\n" name (residual u lo hi))
        [
          ("erfc", { u = Nx.erfc }, -6., 27.);
          ("ndtr", { u = Nx.ndtr }, -38., 9.);
          ("log_ndtr", { u = Nx.log_ndtr }, -40., 10.);
          ("erfinv", { u = Nx.erfinv }, -1., 1.);
          ("ndtri", { u = Nx.ndtri }, 1e-300, 1.);
          ("lgamma", { u = Nx.lgamma }, -10.5, 30.);
          ("digamma", { u = Nx.digamma }, -10.5, 30.);
          ("lbeta", lbeta, 0.1, 30.);
          ("i0e", { u = Nx.i0e }, -30., 30.);
          ("i1e", { u = Nx.i1e }, -30., 30.);
        ];
      expect (output ())
      @@ __POS_OF__
           {|
        erfc 495
        ndtr 547
        log_ndtr 671
        erfinv 804
        ndtri 1026
        lgamma 730
        digamma 590
        lbeta 1267
        i0e 563
        i1e 564
        |})

let () =
  exit
    (run "rune special"
       [ error_function; normal; gamma; bessel; derivatives; residuals ])
