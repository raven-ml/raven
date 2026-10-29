(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What every construct promises of the functions it traces: when they run, what
   they read, what they raise and where their extents end. Beside it, detach,
   lanes, lane_index and totals, eagerly, inside a traced step and under jit. *)

open Windtrap
open Compose
module Rune = Rune2
module Total = Rune.Total
module T = Nx.Ptree

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

let zero () = Nx.scalar f32 (-0.)

(* Tracing, construct by construct

   Each row applies a construct to [x] with [body] as the function it traces,
   or, for a custom rule outside every transformation, runs on values. *)

type f32 = (float, Nx.float32_elt, Nx.host) Nx.t

type construct = {
  label : string;
  entry : string; (* The entry point a message names. *)
  traced : bool; (* Whether [body] runs over stand-ins. *)
  per_key : bool; (* Whether a trace serves later calls, as a jit's does. *)
  make : (f32 -> f32) -> f32 -> f32;
      (* [make body] is bound once and called; a jit compiles at [make]. *)
  eager : (f32 -> f32) -> f32 -> f32; (* What the construct computes. *)
}

let rows = Nx.create f32 [| 3; 2 |] [| 1.; 2.; -0.; 4.; 0.5; 6. |]

(* A carry of a step count beside [x]: three steps. *)
let counted = T.(pair tensor tensor)

let constructs =
  let fold body x =
    List.fold_left
      (fun c k -> body (Nx.add c (Nx.get [ k ] rows)))
      x [ 0; 1; 2 ]
  in
  let thrice body x = body (body (body x)) in
  let rule custom body x =
    custom T.tensor T.tensor (fun x -> (body x, Fun.id)) x
  in
  let on_values entry custom =
    {
      label = entry ^ " on values";
      entry;
      traced = false;
      per_key = false;
      make = rule custom;
      eager = Fun.id;
    }
  in
  let mapped entry custom =
    let make body x =
      Rune.vmap' (rule custom body) (Nx.reshape [| 2; 1 |] x)
      |> Nx.reshape [| 2 |]
    in
    {
      label = entry ^ " in a map";
      entry;
      traced = true;
      per_key = false;
      make;
      eager = Fun.id;
    }
  in
  let at entry make =
    {
      label = entry;
      entry;
      traced = true;
      per_key = false;
      make;
      eager = Fun.id;
    }
  in
  [
    {
      (at "Rune.scan'" (fun body x ->
           fst (Rune.scan' ~init:x ~f:(fun c r -> (body (Nx.add c r), r)) rows)))
      with
      eager = fold;
    };
    {
      (at "Rune.iterate" (fun body x ->
           let until (k, _) = Nx.greater_equal_s k 3. in
           let step (k, y) = (Nx.add_s k 1., body y) in
           snd
             (Rune.iterate counted ~max:4 ~until ~f:step (Nx.scalar f32 0., x))))
      with
      eager = thrice;
    };
    at "Rune.switch" (fun body x ->
        Rune.switch T.tensor (Nx.scalar Nx.int32 0l)
          [ (fun () -> body x); (fun () -> Nx.neg x) ]);
    at "Rune.root" (fun body x ->
        Rune.root T.tensor ~residual:(fun y -> Nx.sub y x) (fun () -> body x));
    at "Rune.remat" (fun body -> Rune.remat T.(tensor @-> returns tensor) body);
    { (at "Rune.jit'" Rune.jit') with per_key = true };
    on_values "Rune.custom_jvp" Rune.custom_jvp;
    on_values "Rune.custom_vjp" Rune.custom_vjp;
    mapped "Rune.custom_jvp" Rune.custom_jvp;
    mapped "Rune.custom_vjp" Rune.custom_vjp;
  ]

let construct_name c = c.label
let x0 = v [| 1.5; -0.25 |]

(* Whether [e] is [Invalid_argument] naming [name], and not an entry point whose
   name extends it, as [Rune.jit'] extends [Rune.jit]. *)
let names name = function
  | Invalid_argument m ->
      let n = String.length name and l = String.length m in
      let ends i =
        i + n = l
        ||
        match m.[i + n] with
        | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '_' | '\'' -> false
        | _ -> true
      in
      let rec at i =
        i + n <= l && ((String.sub m i n = name && ends i) || at (i + 1))
      in
      at 0
  | _ -> false

let read_prefix = "Nx.item: the value is traced by "

(* A construct traces at each call, a jit once per key, and a traced function
   reads a ref's contents at its trace; a rule on values runs at each call. *)
let test_traces c =
  let runs = ref 0 and r = ref 1. in
  let body y =
    incr runs;
    Nx.add_s y !r
  in
  let g = c.make body in
  let first = g x0 in
  r := 2.;
  let second = g x0 in
  let plus a y = Nx.add_s y a in
  equal ~msg:"runs" int (if c.per_key then 1 else 2) !runs;
  equal ~msg:"the first call" bits_t (bits (c.eager (plus 1.) x0)) (bits first);
  let seen = if c.per_key then 1. else 2. in
  equal ~msg:"the ref's contents" bits_t
    (bits (c.eager (plus seen) x0))
    (bits second)

(* A read of a value computed in a traced function raises naming the entry
   point; on values it reads the value. *)
let test_reads c =
  let body y = Nx.add_s y (Nx.item [] (Nx.sum (Nx.neg y))) in
  if c.traced then
    raises_match
      (fun e ->
        (match e with
          | Invalid_argument m -> String.starts_with ~prefix:read_prefix m
          | _ -> false)
        && names c.entry e)
      (fun () -> c.make body x0)
  else equal bits_t (bits (body x0)) (bits (c.make body x0))

(* A draw without a key raises naming the entry point, on values too. *)
let test_keyless c =
  let body y = Nx.add y (Nx.Rng.uniform f32 (Nx.shape y)) in
  raises_match (names c.entry) (fun () -> c.make body x0)

(* An exception the function raises reaches the call; a jit whose trace raised
   traces again. *)
let test_exceptions c =
  raises Exit (fun () -> c.make (fun _ -> raise Exit) x0);
  let raised = ref false in
  let body y =
    if !raised then Nx.neg y
    else begin
      raised := true;
      raise Exit
    end
  in
  let g = c.make body in
  raises Exit (fun () -> g x0);
  equal ~msg:"the next call" bits_t (bits (c.eager Nx.neg x0)) (bits (g x0))

(* Under a transformation, a [try] around the construct inside the transformed
   function sees the exception. *)
let test_caught c =
  let f x =
    try Nx.sum (c.make (fun _ -> raise Exit) x) with Exit -> Nx.sum (Nx.neg x)
  in
  let minus = v [| -1.; -1. |] in
  equal ~msg:"grad" bits_t (bits minus) (bits (Rune.grad' f x0));
  equal ~msg:"jvp" bits_t
    (bits (Nx.scalar f32 (-2.)))
    (bits (snd (Rune.jvp' f x0 (v [| 1.; 1. |]))));
  let xs = Nx.reshape [| 1; 2 |] x0 in
  equal ~msg:"vmap" bits_t
    (bits (Nx.reshape [| 1 |] (Nx.scalar f32 (-1.25))))
    (bits (Rune.vmap' f xs))

(* Extents *)

type _ Effect.t += Pause : unit Effect.t

(* [f ()] in a fiber that pauses once and is resumed on this domain. *)
let in_fiber f =
  let open Effect.Deep in
  try_with
    (fun () ->
      Effect.perform Pause;
      f ())
    ()
    {
      effc =
        (fun (type a) (e : a Effect.t) ->
          match e with
          | Pause -> Some (fun (k : (a, _) continuation) -> continue k ())
          | _ -> None);
    }

let test_fiber () =
  let a = Rune.axis () in
  let xs = v [| 3.; 5. |] in
  let lane x =
    in_fiber (fun () -> Nx.add x (Nx.cast f32 (Rune.lane_index a x)))
  in
  equal ~msg:"a lane's index" bits_t
    (bits (v [| 3.; 6. |]))
    (bits (Rune.vmap' ~axis:a lane xs));
  let t = Total.make () in
  let (), sum =
    Total.collect t ~zero:(zero ()) (fun () ->
        in_fiber (fun () -> Total.add t (Nx.scalar f32 1.5)))
  in
  equal ~msg:"an addition" bits_t (bits (Nx.scalar f32 1.5)) (bits sum)

let test_elsewhere () =
  let a = Rune.axis () in
  let xs = v [| 3.; 5. |] in
  let elsewhere start f =
    let r = ref None in
    start (fun () -> r := Some (try Ok (f ()) with e -> Error e));
    match !r with
    | Some (Ok y) -> y
    | Some (Error e) -> raise e
    | None -> assert false
  in
  let domain f = Domain.join (Domain.spawn f) in
  let thread f = Thread.join (Thread.create f ()) in
  let starts =
    [ ("a domain", fun f -> domain f); ("a systhread", fun f -> thread f) ]
  in
  List.iter
    (fun (msg, start) ->
      invalid ~sub:"Rune.lanes: no map named by the axis is around the call"
        (fun () ->
          Rune.vmap' ~axis:a
            (fun x -> elsewhere start (fun () -> Rune.lanes a x))
            xs);
      let t = Total.make () in
      let (), sum =
        Total.collect t ~zero:(zero ()) (fun () ->
            elsewhere start (fun () -> Total.add t (Nx.scalar f32 1.5)))
      in
      equal ~msg bits_t (bits (zero ())) (bits sum))
    starts

(* Totals *)

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

(* A construct whose delivery to one total raises delivers to none: the
   additions of a construct that raises reach no total. *)
let test_delivery_raises () =
  let refused
      ?(sub = "Rune.Total.add: shape [3], where the total's zero has [2]") f =
    invalid ~sub (fun () -> ignore (f ()))
  in
  let t1 = Total.make () and t2 = Total.make () in
  let z n = Nx.zeros f32 [| n |] in
  let (), s1 =
    Total.collect t1 ~zero:(z 2) (fun () ->
        refused (fun () ->
            Total.collect t2 ~zero:(z 2) (fun () ->
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
    Total.collect t1 ~zero:(z 2) (fun () ->
        refused (fun () ->
            Total.collect t2 ~zero:(z 2) (fun () ->
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
    Total.collect t1 ~zero:(z 2) (fun () ->
        refused (fun () ->
            Total.collect t2 ~zero:(z 2) (fun () ->
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
    Total.collect t1 ~zero:(z 2) (fun () ->
        refused (fun () ->
            Total.collect t2 ~zero:(z 2) (fun () ->
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
    Total.collect t1 ~zero:(z 2) (fun () ->
        if n = 2 then ignore (Total.collect t2 ~zero:(z n) (fun () -> f x))
        else
          refused
            ~sub:"Rune.Total.add: shape [2], where the total's zero has [3]"
            (fun () -> Total.collect t2 ~zero:(z n) (fun () -> f x)))
  in
  equal ~msg:"a compiled call" bits_t (bits x) (bits (snd (call 2)));
  equal ~msg:"a compiled call refused" bits_t (bits (z 2)) (bits (snd (call 3)))

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
   and+ mode =
     of_list
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
   in
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
  cover "under a jit of a transformation"
    (match c.mode with
    | Jit_forward | Jit_reverse | Jit_mapped -> true
    | Eager | Jitted | Forward | Reverse | Mapped -> false);
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
      Total.collect t ~zero:(value c.zero) (fun () ->
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
      (Total.collect t ~zero:(Nx.scalar f32 1e8) (fun () ->
           f ();
           f ()))
  in
  let twelve () = Nx.scalar f32 12. in
  let eager () =
    Total.add t (twelve ());
    Total.add t (Nx.place Nx.Host.on (twelve ()))
  in
  equal bits_t (bits (total eager)) (bits (total (fun () -> ignore (g x))))

let () =
  exit
    (run "rune2 constructs"
       [
         test "detach is its operand" test_detach;
         test "lanes and lane_index raise with no map" test_no_map;
         cases ~name:construct_name "traced once per trace" constructs
           test_traces;
         cases ~name:construct_name "a read of a traced value raises" constructs
           test_reads;
         cases ~name:construct_name "a keyless draw raises" constructs
           test_keyless;
         cases ~name:construct_name "an exception reaches the call" constructs
           test_exceptions;
         cases ~name:construct_name "a try inside a transformation sees it"
           constructs test_caught;
         group "extents"
           [
             test "a fiber resumed on the domain" test_fiber;
             test "a domain or systhread the function starts" test_elsewhere;
           ];
         group "totals"
           [
             test "collect sums each addition once" test_collect;
             test "the innermost scope of a total" test_same_total;
             test "a scope inside a jit" test_scope_in_jit;
             prop "jit adds what eager adds" additions law_additions;
             test "a jit that raises adds nothing" test_raising_jit;
             test "an addition of another shape raises" test_shape_refused;
             test "an exception passes through collect" test_raises;
             test "a construct whose delivery raises delivers nothing"
               test_delivery_raises;
             test "a jit adds once per call" test_jit_scope;
             prop "additions chain wherever they are made" chained law_chained;
             test "a jit takes the running value where it lies" test_jit_running;
           ];
       ])
