(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What every construct promises of the functions it traces: when they run, what
   they read, what they raise and where their extents end. Beside it, detach,
   lanes and lane_index with no map. *)

open Windtrap
open Compose
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
  stages : bool; (* Whether it records operations on captures too. *)
  bodies : int; (* How often one trace runs [body]: once per branch. *)
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
      stages = false;
      bodies = 1;
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
      stages = false;
      bodies = 1;
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
      stages = false;
      bodies = 1;
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
    {
      (at "Rune.switch" (fun body x ->
           Rune.switch T.tensor (Nx.scalar Nx.int32 0l)
             [ (fun () -> body x); (fun () -> body (Nx.neg x)) ]))
      with
      stages = true;
      bodies = 2;
    };
    {
      (at "Rune.root" (fun body x ->
           Rune.root T.tensor ~residual:(fun y -> Nx.sub y x) (fun () -> body x)))
      with
      stages = true;
    };
    {
      (at "Rune.remat" (fun body ->
           Rune.remat T.(tensor @-> returns tensor) body))
      with
      stages = true;
    };
    { (at "Rune.jit'" Rune.jit') with per_key = true; stages = true };
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
  equal ~msg:"runs" int (c.bodies * if c.per_key then 1 else 2) !runs;
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
    raises (Invalid_argument (read_prefix ^ c.entry)) (fun () -> c.make body x0)
  else equal bits_t (bits (body x0)) (bits (c.make body x0))

(* An operation on captures alone computes at the call, so a read of its result
   gives the value, but where the construct stages it. *)
let test_staged c =
  let body y = Nx.add_s y (Nx.item [] (Nx.sum (Nx.neg x0))) in
  if c.stages then
    raises (Invalid_argument (read_prefix ^ c.entry)) (fun () -> c.make body x0)
  else equal bits_t (bits (c.eager body x0)) (bits (c.make body x0))

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

(* Arguments kept

   While backtraces are recorded, the runtime keeps the last exception raised
   reachable, and a construct no transformation answers is answered by catching
   one. Once it returns, the construct keeps none of its arguments. *)

let recording f =
  let was = Printexc.backtrace_status () in
  Printexc.record_backtrace true;
  Fun.protect ~finally:(fun () -> Printexc.record_backtrace was) f

(* Whether [f]'s argument is reachable once [f] returns, a first call made. *)
let kept f =
  ignore (f (v [| 0.5; 2. |]));
  recording @@ fun () ->
  let w = Weak.create 1 in
  let[@inline never] call () =
    let x = v [| 1.5; -0.25 |] in
    Weak.set w 0 (Some x);
    ignore (Sys.opaque_identity (f x))
  in
  call ();
  Gc.full_major ();
  Weak.check w 0

let test_kept_detach () = equal bool false (kept Rune.detach)

(* The constructs outside every transformation: a map answers those in it. *)
let outside =
  List.filter
    (fun c -> not (String.ends_with ~suffix:" in a map" c.label))
    constructs

let test_kept c = equal bool false (kept (c.make Nx.neg))

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
    Total.collect t ~init:(zero ()) (fun () ->
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
        Total.collect t ~init:(zero ()) (fun () ->
            elsewhere start (fun () -> Total.add t (Nx.scalar f32 1.5)))
      in
      equal ~msg bits_t (bits (zero ())) (bits sum))
    starts

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
         cases ~name:construct_name "a read of captures alone raises if staged"
           constructs test_staged;
         cases ~name:construct_name "a keyless draw raises" constructs
           test_keyless;
         cases ~name:construct_name "an exception reaches the call" constructs
           test_exceptions;
         cases ~name:construct_name "a try inside a transformation sees it"
           constructs test_caught;
         test "detach keeps no argument once it returns" test_kept_detach;
         cases ~name:construct_name "keeps no argument once it returns"
           outside test_kept;
         group "extents"
           [
             test "a fiber resumed on the domain" test_fiber;
             test "a domain or systhread the function starts" test_elsewhere;
           ];
       ])
