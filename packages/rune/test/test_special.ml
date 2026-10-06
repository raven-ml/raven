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

(* [compiled3 at g] is [g] of three arguments, compiled likewise. *)
type c = {
  c :
    'b.
    (float, 'b) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t -> (float, 'b) Nx.t;
}

let compiled3 at { c } =
  {
    f =
      (fun a ->
        let f =
          Rune.jit Nx.Ptree.(tensor @-> tensor @-> tensor @-> returns tensor) c
        in
        Nx.place Nx.Placement.host
          (f (Nx.place at a.(0)) (Nx.place at a.(1)) (Nx.place at a.(2))));
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

let in_domain row =
  let a = row.args.(0) in
  not (Float.is_finite a && a > 0x1p20)

(* [on_devices] for the incomplete gamma family, whose bounds hold up to [a =
   2^20]. *)
let binary name ~bound b =
  let compile at = compiled2 at b in
  let host =
    check ~keep:in_domain ~bound (golden name) (compile Nx.Placement.host)
  in
  let metal =
    match metal with
    | Some m ->
        check
          ~keep:(fun r -> in_domain r && not (subnormal r))
          ~dtypes:[ `F32 ] ~bound (golden name)
          (compile (Nx.Placement.on m))
    | None -> [ test "metal" (fun () -> skip ~reason:"no Metal device" ()) ]
  in
  group name [ group "on the host" host; group "on Metal" metal ]

let log_ulps_bound k args = Inverse (4, int_of_float (log_ulps k args.(1)))

let incomplete_gamma =
  group "incomplete gamma"
    [
      binary "gammainc" ~bound:(everywhere (Log_ulps 16)) { b = Nx.gammainc };
      binary "gammaincc" ~bound:(everywhere (Log_ulps 16)) { b = Nx.gammaincc };
      binary "log_gammainc"
        ~bound:(everywhere (Near_zeros (16, 16)))
        { b = Nx.log_gammainc };
      binary "log_gammaincc"
        ~bound:(everywhere (Near_zeros (16, 16)))
        { b = Nx.log_gammaincc };
      binary "gammaincinv" ~bound:(log_ulps_bound 16) { b = Nx.gammaincinv };
      binary "gammainccinv" ~bound:(log_ulps_bound 16) { b = Nx.gammainccinv };
    ]

let incomplete_beta =
  let ternary name ~bound c =
    on_devices name ~bound:(everywhere bound) (fun at -> compiled3 at c)
  in
  group "incomplete beta"
    [
      ternary "betainc" ~bound:(Log_ulps 32) { c = Nx.betainc };
      ternary "log_betainc" ~bound:(Near_zeros (32, 32)) { c = Nx.log_betainc };
    ]

(* Derivatives

   rune differentiates each function through the operations nx computes it with.
   Each derivative is held to golden/special_grad/ (written by
   gen/special_grad.py) eagerly and compiled on the host: within 16 times its
   function's bound, and second derivatives within [2^-40] relative at float64
   and [2^-16] at float32. *)

let grad_golden name = "golden/special_grad/" ^ name ^ ".golden"

(* [d] held eagerly and compiled, at both dtypes. *)
let derivative name ~bound (d : u) =
  let eager = { f = (fun a -> d.u a.(0)) } in
  let compiled = { f = (fun a -> Rune.jit' d.u a.(0)) } in
  group name
    [
      group "eagerly" (check ~zeros:`Unsigned ~bound (grad_golden name) eager);
      group "compiled"
        (check ~zeros:`Unsigned ~bound (grad_golden name) compiled);
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

(* [f]'s derivative in its [i]th argument, held to [grad_golden name] eagerly
   and compiled. *)
let derivative3 name ~bound i { c } =
  let d a b x =
    match i with
    | 0 -> Rune.grad' (fun a -> Nx.sum (c a b x)) a
    | 1 -> Rune.grad' (fun b -> Nx.sum (c a b x)) b
    | _ -> Rune.grad' (fun x -> Nx.sum (c a b x)) x
  in
  let eager = { f = (fun v -> d v.(0) v.(1) v.(2)) } in
  let compiled =
    {
      f =
        (fun v ->
          Rune.jit
            Nx.Ptree.(tensor @-> tensor @-> tensor @-> returns tensor)
            d v.(0) v.(1) v.(2));
    }
  in
  group name
    [
      group "eagerly" (check ~zeros:`Unsigned ~bound (grad_golden name) eager);
      group "compiled"
        (check ~zeros:`Unsigned ~bound (grad_golden name) compiled);
    ]

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

(* The derivative of [b] in its first argument, or its second. *)
let d_first { b } = { b = (fun a x -> Rune.grad' (fun a -> Nx.sum (b a x)) a) }
let d_second { b } = { b = (fun a x -> Rune.grad' (fun x -> Nx.sum (b a x)) x) }

(* In [a], [2^-40 (1 + |log f|)] relative at float64 and [2^-16 (1 + |log f|)]
   at float32, the scale holding [1 + |log f|]; in [x], 16 times the function's
   own bound. *)
let in_a = everywhere (Relative_scaled (0x1p-40, 0x1p-16))
let linear_in_x = everywhere (Inverse (64, 512))

let incomplete_gamma_derivatives =
  group "incomplete gamma derivatives"
    [
      derivative2 "gammainc_a" ~bound:in_a (d_first { b = Nx.gammainc });
      derivative2 "gammainc_x" ~bound:linear_in_x (d_second { b = Nx.gammainc });
      derivative2 "gammaincc_a" ~bound:in_a (d_first { b = Nx.gammaincc });
      derivative2 "gammaincc_x" ~bound:linear_in_x
        (d_second { b = Nx.gammaincc });
      derivative2 "log_gammainc_a" ~bound:in_a (d_first { b = Nx.log_gammainc });
      derivative2 "log_gammainc_x"
        ~bound:(everywhere (Near_zeros (256, 256)))
        (d_second { b = Nx.log_gammainc });
      derivative2 "log_gammaincc_a" ~bound:in_a
        (d_first { b = Nx.log_gammaincc });
      derivative2 "log_gammaincc_x"
        ~bound:(everywhere (Near_zeros (256, 256)))
        (d_second { b = Nx.log_gammaincc });
      derivative2 "gammaincinv_a"
        ~bound:(everywhere (Relative (0x1p-40, 0x1p-16)))
        (d_first { b = Nx.gammaincinv });
      derivative2 "gammaincinv_p"
        ~bound:(fun args ->
          Inverse (64, 16 * int_of_float (log_ulps 16 args.(1))))
        (d_second { b = Nx.gammaincinv });
      derivative2 "gammainccinv_a"
        ~bound:(everywhere (Relative (0x1p-40, 0x1p-16)))
        (d_first { b = Nx.gammainccinv });
      derivative2 "gammainccinv_p"
        ~bound:(fun args ->
          Inverse (64, 16 * int_of_float (log_ulps 16 args.(1))))
        (d_second { b = Nx.gammainccinv });
    ]

let derivatives =
  group "derivatives"
    [
      digamma_near_zero;
      log_ndtr_far;
      bessel_at_zero;
      derivative "i0e" ~bound:(everywhere (Ulps 128)) (d { u = Nx.i0e });
      group "incomplete beta"
        (List.concat_map
           (fun (tail, c) ->
             [
               derivative3 (tail ^ "_x")
                 ~bound:(everywhere (Relative_scaled (0x1p-47, 0x1p-18)))
                 2 c;
               derivative3 (tail ^ "_a") ~bound:in_a 0 c;
               derivative3 (tail ^ "_b") ~bound:in_a 1 c;
             ])
           [
             ("log_betainc", { c = Nx.log_betainc });
             ("log_betaincc", { c = Nx.log_betaincc });
           ]);
      derivative "i1e"
        ~bound:(fun args ->
          let x = Float.abs args.(0) in
          if 1. <= x && x <= 2. then Near_zeros (128, 128) else Ulps 128)
        (d { u = Nx.i1e });
      derivative "erfc" ~bound:(everywhere (Ulps 128)) (d { u = Nx.erfc });
      derivative "ndtr" ~bound:(everywhere (Ulps 256)) (d { u = Nx.ndtr });
      derivative "log_ndtr" ~bound:(everywhere (Ulps 512))
        (d { u = Nx.log_ndtr });
      derivative "erfinv" ~bound:(everywhere (Ulps 64)) (d { u = Nx.erfinv });
      derivative "ndtri" ~bound:(everywhere (Ulps 64)) (d { u = Nx.ndtri });
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
      let full v t = Nx.full_like t v in
      let in_x = { u = (fun x -> Nx.log_betainc (full 2.5 x) (full 7. x) x) } in
      let in_a = { u = (fun a -> Nx.log_betainc a (full 7. a) (full 0.3 a)) } in
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
          ( "gammainc",
            { u = (fun x -> Nx.gammainc (Nx.full_like x 3.5) x) },
            0.1,
            30. );
          ( "gammainc in a",
            { u = (fun a -> Nx.gammainc a (Nx.full_like a 5.)) },
            0.1,
            30. );
          ( "gammaincinv",
            { u = (fun p -> Nx.gammaincinv (Nx.full_like p 3.5) p) },
            1e-6,
            0.999 );
          ( "gammaincinv in a",
            { u = (fun a -> Nx.gammaincinv a (Nx.full_like a 0.3)) },
            0.1,
            30. );
          ("log_betainc in x", in_x, 1e-3, 0.999);
          ("log_betainc in a", in_a, 0.1, 50.);
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
        i0e 579
        i1e 564
        gammainc 4921
        gammainc in a 6963
        gammaincinv 16361
        gammaincinv in a 26355
        log_betainc in x 14385
        log_betainc in a 23947
        |})

let () =
  exit
    (run "rune special"
       [
         error_function;
         normal;
         gamma;
         bessel;
         incomplete_gamma;
         incomplete_beta;
         derivatives;
         incomplete_gamma_derivatives;
         residuals;
       ])
