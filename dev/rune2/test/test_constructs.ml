(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Constructs without programs: detach, lanes, lane_index and totals, eagerly,
   inside a traced step and under jit. *)

open Windtrap
open Compose
module Rune = Rune2
module Total = Rune.Total

let invalid ~sub f = raises_match (Exn.invalid_arg ~substring:sub) f
let f32 = Nx.float32
let v xs = Nx.create f32 [| Array.length xs |] xs

let test_detach () =
  let x = v [| 1.; -0. |] in
  let one_value = Testable.make ~pp:Nx.pp ~equal:( == ) in
  equal ~msg:"x itself" one_value x (Rune.detach x);
  equal bits_t
    (bits (Nx.sqrt x))
    (bits (Rune.jit' (fun x -> Nx.sqrt (Rune.detach x)) x))

(* With no map around, lanes and lane_index raise eagerly and under jit. *)
let test_no_map () =
  let a = Rune.axis () in
  let no_lanes = "Rune.lanes: no map named by the axis is around the call" in
  let no_index =
    "Rune.lane_index: no map named by the axis is around the call"
  in
  invalid ~sub:no_lanes (fun () -> Rune.lanes a (v [| 1. |]));
  invalid ~sub:no_index (fun () -> Rune.lane_index a (v [| 1. |]));
  invalid ~sub:no_lanes (fun () ->
      Rune.jit' (fun x -> Rune.lanes a x) (v [| 1. |]));
  invalid ~sub:no_index (fun () ->
      Rune.jit'
        (fun x -> Nx.add x (Nx.cast f32 (Rune.lane_index a x)))
        (v [| 1. |]))

(* Totals *)

let zero () = Nx.scalar f32 (-0.)

let test_collect () =
  let t = Total.make () in
  let a = Nx.scalar f32 1.5 and b = Nx.scalar f32 (-0.25) in
  let r, sum =
    Total.collect t ~zero:(zero ()) (fun () ->
        Total.add t a;
        Total.add t b;
        "done")
  in
  equal string "done" r;
  equal bits_t (bits Nx.(add (add (zero ()) a) b)) (bits sum);
  (* No addition keeps the zero's bits. *)
  let _, sum = Total.collect t ~zero:(zero ()) (fun () -> ()) in
  equal bits_t (bits (zero ())) (bits sum);
  (* With no scope open, an addition does nothing. *)
  Total.add t a;
  (* A scope of another total lets the additions pass. *)
  let u = Total.make () in
  let ((), inner), outer =
    Total.collect t ~zero:(zero ()) (fun () ->
        Total.collect u ~zero:(zero ()) (fun () ->
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
    Total.collect t ~zero:(zero ()) (fun () ->
        Total.add t a;
        Total.collect t ~zero:(zero ()) (fun () -> Total.add t b))
  in
  equal bits_t (bits Nx.(add (zero ()) b)) (bits inner);
  equal bits_t (bits Nx.(add (zero ()) a)) (bits outer)

(* A scope inside a jit sums the function's additions there. *)
let test_scope_in_jit () =
  let t = Total.make () in
  let f x =
    let y, s =
      Total.collect t ~zero:(zero ()) (fun () ->
          Total.add t (Nx.sum x);
          Nx.sin x)
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
  let (), eager = Total.collect t ~zero:(zero ()) (fun () -> caught f) in
  let (), jitted =
    Total.collect t ~zero:(zero ()) (fun () -> caught (Rune.jit' f))
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
  let y, s = Total.collect t ~zero:(zero ()) (fun () -> f x) in
  let y', s' = Total.collect t ~zero:(zero ()) (fun () -> Rune.jit' f x) in
  equal bits_t (bits y) (bits y');
  equal bits_t (bits s) (bits s')

let test_shape_refused () =
  let t = Total.make () in
  invalid ~sub:"Rune.Total.add: shape [2], where the total's zero has []"
    (fun () ->
      Total.collect t ~zero:(zero ()) (fun () -> Total.add t (v [| 1.; 2. |])))

let test_raises () =
  let t = Total.make () in
  raises Exit (fun () ->
      Total.collect t ~zero:(zero ()) (fun () ->
          Total.add t (Nx.scalar f32 1.);
          raise Exit))

(* A scope around a jit receives the function's additions once per call. *)
let test_jit_scope () =
  let t = Total.make () in
  let f x =
    Total.add t x;
    Nx.sin x
  in
  let x = v [| 0.5; 2. |] in
  let y, sum =
    Total.collect t ~zero:(v [| 0.; 0. |]) (fun () -> Rune.jit' f x)
  in
  equal bits_t (bits (Nx.sin x)) (bits y);
  equal bits_t (bits (Nx.add (v [| 0.; 0. |]) x)) (bits sum)

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

and mode = Eager | Jitted | Forward | Reverse

let pp_mode ppf m =
  Format.pp_print_string ppf
    (match m with
    | Eager -> "eagerly"
    | Jitted -> "under jit"
    | Forward -> "under jvp"
    | Reverse -> "under grad")

let chained =
  let open Gen in
  let value =
    pair (of_list [ 1e8; -1e8; 12.; 3.; 0.1; 1e-8; -0.; 0.; 7.25 ]) bool
  in
  (let* zero = value and+ pre = value in
   let* rows = int_range 0 3 and+ trips = int_range 0 3 in
   let* inner =
     of_list [ Plain; Scan_in; Iterate_in; Remat_in; Custom_in; Jit_in ]
   in
   let* adds = array ~size:(constant 4) value
   and+ mode = of_list [ Eager; Jitted; Forward; Reverse ] in
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

let law_chained c =
  cover "an inner construct" (c.inner <> Plain);
  cover "under a transformation" (c.mode <> Eager);
  let t = Total.make () in
  (* A placed value is computed; one of every set is a formula. *)
  let value (x, placed) =
    let s = Nx.scalar f32 x in
    if placed then Nx.place Nx.Host.on s else s
  in
  let add k = Total.add t (value c.adds.(k)) in
  let inner_ops carry =
    match c.inner with
    | Plain -> ()
    | Scan_in ->
        ignore
          (Rune.scan' ~init:carry
             ~f:(fun x _ ->
               add 1;
               add 2;
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
               add 1;
               add 2;
               (x, Nx.add k (Nx.scalar f32 1.)))
             (carry, Nx.scalar f32 0.))
    | Remat_in ->
        ignore
          (Rune.remat
             Nx.Ptree.(tensor @-> returns tensor)
             (fun x ->
               add 1;
               add 2;
               x)
             carry)
    | Custom_in ->
        ignore
          (Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor
             (fun x ->
               add 1;
               add 2;
               (x, Fun.id))
             carry)
    | Jit_in ->
        ignore
          (Rune.jit'
             (fun x ->
               add 1;
               add 2;
               x)
             carry)
  in
  let inner_trips =
    match c.inner with
    | Plain -> 0
    | Scan_in | Iterate_in -> c.trips
    | Remat_in | Custom_in | Jit_in -> 1
  in
  let collect f = snd (Total.collect t ~zero:(value c.zero) f) in
  let expected =
    collect (fun () ->
        Total.add t (value c.pre);
        for _ = 1 to c.rows do
          add 0;
          for _ = 1 to inner_trips do
            add 1;
            add 2
          done;
          add 3
        done)
  in
  let scanned init =
    fst
      (Rune.scan' ~init
         ~f:(fun x _ ->
           add 0;
           inner_ops x;
           add 3;
           (x, x))
         (Nx.zeros f32 [| c.rows; 1 |]))
  in
  (* The scan's carry and the collect's total, the collect inside the
     transformation. *)
  let run x =
    let y, s =
      Total.collect t ~zero:(value c.zero) (fun () ->
          Total.add t (value c.pre);
          scanned x)
    in
    (Nx.sum y, s)
  in
  let init = v [| 1. |] in
  let traced =
    match c.mode with
    | Eager -> snd (run init)
    | Jitted ->
        snd
          (Rune.jit Nx.Ptree.(tensor @-> returns (pair tensor tensor)) run init)
    | Forward ->
        snd
          (fst
             (Rune.jvp Nx.Ptree.tensor
                Nx.Ptree.(pair tensor tensor)
                run init init))
    | Reverse ->
        let _, _, s =
          Rune.value_and_grad_aux Nx.Ptree.tensor Nx.Ptree.tensor run init
        in
        s
  in
  equal bits_t (bits expected) (bits traced)

let () =
  exit
    (run "rune2 constructs"
       [
         test "detach is its operand" test_detach;
         test "lanes and lane_index raise with no map" test_no_map;
         group "totals"
           [
             test "collect sums each addition once" test_collect;
             test "the innermost scope of a total" test_same_total;
             test "a scope inside a jit" test_scope_in_jit;
             prop "jit adds what eager adds" additions law_additions;
             test "a jit that raises adds nothing" test_raising_jit;
             test "an addition of another shape raises" test_shape_refused;
             test "an exception passes through collect" test_raises;
             test "a jit adds once per call" test_jit_scope;
             prop "additions chain wherever they are made" chained law_chained;
           ];
       ])
