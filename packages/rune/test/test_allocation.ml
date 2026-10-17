(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The words reverse mode and a compiled call's replay allocate on the minor
   heap over one-element values, whose kernels have almost nothing to do: the
   cost of recording and transposing each operation, and of keying and running a
   program, exactly. Then the device bytes an eager [vjp] keeps for its
   pullback. Counts are deterministic, so each is pinned: a change that lowers
   one updates it here, in the same commit. *)

open Windtrap

(* The fewest words any of ten calls allocates, after a warm-up call: a
   finaliser that happens to run inside one call cannot inflate the count. *)
let words f =
  ignore (Sys.opaque_identity (f ()));
  let fewest = ref max_int in
  for _ = 1 to 10 do
    let before = Gc.minor_words () in
    ignore (Sys.opaque_identity (f ()));
    fewest := min !fewest (int_of_float (Gc.minor_words () -. before))
  done;
  !fewest

let x = Nx.create Nx.float32 [| 1 |] [| 0.5 |]
let w = Nx.create Nx.float32 [| 1; 1 |] [| 0.5 |]
let row = Nx.reshape [| 1; 1 |] x

(* A selection: [x] where it is positive, [0.] elsewhere. *)
let relu x =
  let zero = Nx.scalar_like x 0. in
  Nx.where (Nx.less_equal x zero) zero x

let reverse =
  group "reverse mode"
    [
      test "grad of a product" (fun () ->
          equal int 1257
            (words (fun () -> Rune.grad' (fun x -> Nx.sum (Nx.mul x x)) x)));
      test "grad of a selection" (fun () ->
          equal int 1607
            (words (fun () -> Rune.grad' (fun x -> Nx.sum (relu x)) x)));
      test "grad of a matrix product" (fun () ->
          equal int 1167
            (words (fun () -> Rune.grad' (fun w -> Nx.sum (Nx.matmul row w)) w)));
    ]

(* A compiled call that replays its program: the cost of keying, binding and
   running it, with the settings it keys on read as they are. *)
let neg = Rune.jit' Nx.neg

(* A loop that adds one to [x] until it reaches [limit]: one program, whose
   trip count its arguments decide. *)
let count_to =
  Rune.jit
    Nx.Ptree.(tensor @-> tensor @-> returns tensor)
    (fun x limit ->
      Rune.iterate' ~max:100
        ~until:(fun x -> Nx.all (Nx.less_equal limit x))
        ~f:(fun x -> Nx.add_s x 1.)
        x)

let replay =
  group "jit replay"
    [
      test "neg of one element" (fun () ->
          equal int 643 (words (fun () -> neg x)));
      test "a loop's run allocates the same whatever its trip count" (fun () ->
          let limit v = Nx.create Nx.float32 [| 1 |] [| v |] in
          let one = limit 1.5 and many = limit 64.5 in
          equal int
            (words (fun () -> count_to x one))
            (words (fun () -> count_to x many)));
    ]

(* Residuals

   The bytes per element an eager [vjp] of each function keeps for its pullback,
   at float64 over 4096 elements, on a test device that counts what it holds:
   what reverse mode costs through a composition of nx's operations. Each count
   is read after forced collections, which take time in proportion to the heap,
   so they run in this process, which holds little. *)

let device = Nx.Device.cpu 5

let settled () =
  let m = Nx.Device.memory device in
  for _ = 1 to 4 do
    Gc.full_major ();
    (* An allocation on the device returns its collected memory. *)
    ignore (Rig.Buffer.create m 0);
    Rig.free_cache m
  done;
  Rig.allocated m

let elements = 4096

let residual u lo hi =
  let x = Nx.linspace Nx.float64 lo hi elements in
  let x = Nx.place (Nx.Placement.on device) x in
  let before = settled () in
  let y, pullback = Rune.vjp' u x in
  let held = settled () - before in
  ignore (Sys.opaque_identity (y, pullback));
  held / elements

let residuals =
  test "an eager vjp keeps these bytes per element" (fun () ->
      let full v t = Nx.full_like t v in
      List.iter
        (fun (name, u, lo, hi) ->
          Printf.printf "%s %d\n" name (residual u lo hi))
        [
          ("erfc", Nx.erfc, -6., 27.);
          ("ndtr", Nx.ndtr, -38., 9.);
          ("log_ndtr", Nx.log_ndtr, -40., 10.);
          ("erfinv", Nx.erfinv, -1., 1.);
          ("ndtri", Nx.ndtri, 1e-300, 1.);
          ("lgamma", Nx.lgamma, -10.5, 30.);
          ("digamma", Nx.digamma, -10.5, 30.);
          ("lbeta", (fun a -> Nx.lbeta a (full 3.5 a)), 0.1, 30.);
          ("i0e", Nx.i0e, -30., 30.);
          ("i1e", Nx.i1e, -30., 30.);
          ("gammainc", (fun x -> Nx.gammainc (full 3.5 x) x), 0.1, 30.);
          ("gammainc in a", (fun a -> Nx.gammainc a (full 5. a)), 0.1, 30.);
          ("gammaincinv", (fun p -> Nx.gammaincinv (full 3.5 p) p), 1e-6, 0.999);
          ( "gammaincinv in a",
            (fun a -> Nx.gammaincinv a (full 0.3 a)),
            0.1,
            30. );
          ( "log_betainc in x",
            (fun x -> Nx.log_betainc (full 2.5 x) (full 7. x) x),
            1e-3,
            0.999 );
          ( "log_betainc in a",
            (fun a -> Nx.log_betainc a (full 7. a) (full 0.3 a)),
            0.1,
            50. );
        ];
      expect (output ())
      @@ __POS_OF__
           {|
        erfc 487
        ndtr 556
        log_ndtr 680
        erfinv 813
        ndtri 1027
        lgamma 731
        digamma 599
        lbeta 1276
        i0e 580
        i1e 573
        gammainc 4922
        gammainc in a 6964
        gammaincinv 16362
        gammaincinv in a 26356
        log_betainc in x 11520
        log_betainc in a 19849
        |})

(* Fallbacks

   While backtraces are recorded, the runtime keeps the last exception raised
   reachable. A construct no transformation answers, such as one beside another
   domain's [grad], keeps none of its arguments' bytes once it returns. *)

(* [beside f] is [f ()] while another domain is inside a [grad]. *)
let beside f =
  let inside = Atomic.make false and stop = Atomic.make false in
  let differentiated x =
    Atomic.set inside true;
    while not (Atomic.get stop) do
      Domain.cpu_relax ()
    done;
    Nx.sum x
  in
  let other = Domain.spawn (fun () -> ignore (Rune.grad' differentiated x)) in
  while not (Atomic.get inside) do
    Domain.cpu_relax ()
  done;
  Fun.protect
    ~finally:(fun () ->
      Atomic.set stop true;
      Domain.join other)
    f

(* [detached ()] detaches a fresh value on [device] and drops it. *)
let[@inline never] detached () =
  let x =
    Nx.place (Nx.Placement.on device) (Nx.zeros Nx.float64 [| elements |])
  in
  ignore (Sys.opaque_identity (Rune.detach x))

(* [held_after f] is the bytes [device] holds once [f ()] returns, beyond those
   it held before, after one uncounted run, with backtraces recorded. *)
let held_after f =
  let recording = Printexc.backtrace_status () in
  Printexc.record_backtrace true;
  Fun.protect ~finally:(fun () -> Printexc.record_backtrace recording)
  @@ fun () ->
  f ();
  let before = settled () in
  f ();
  settled () - before

let fallbacks =
  test "a construct beside another domain's grad keeps no argument's bytes"
    (fun () -> equal int 0 (beside (fun () -> held_after detached)))

let () = exit (run "rune allocation" [ reverse; replay; residuals; fallbacks ])
