(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Nx.Prim, in nx.mli's order: which interpretation an operation reaches, the
   extent and the rule; operations as data, their operands, forms and results;
   interpret, traced and twinned values, payloads and owners. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module P = Nx_kernel.Prog
module C = Nx_support.Counting
module Count = (val Nx.devices ~kernels:(module C) [ Nx_support.memory 0 ])

module Count2 =
  (val Nx.devices ~kernels:(module C)
         [ Nx_support.memory 0; Nx_support.memory 1 ])

let invalid ~sub f = raises_match (Exn.invalid_arg ~sub) f
let f32 = Nx.float32
let f64 = Nx.float64
let host_array x = Option.get (Nx.Repr.array ~by:"test" (Nx.place Nx.Host.on x))
let elements x = A.to_array (host_array x)

let vec dt xs =
  Nx.Repr.of_array Nx.Host.v (A.of_array dt [| Array.length xs |] xs)

let on_count dt xs = Nx.place Count.on (vec dt xs)

(* [logged ~name reach log f] interprets [f] with a rule that logs each
   operation it receives and gives traced results tagged with their position. *)
type ('v, 's, 'd) Nx.Prim.payload += Tag : int -> ('v, 's, 'd) Nx.Prim.payload

let tagged i ~by op =
  Nx.Prim.results ~by (fun k form -> Nx.Prim.traced i form (Tag k)) op

let logged ~name reach log f =
  Nx.Prim.interpret ~name reach
    (fun i ~by op ->
      log := Nx.Prim.name op :: !log;
      tagged i ~by op)
    f

(* A value [i] owns, of [x]'s form. *)
let tag i x = Nx.Prim.traced i (Nx.Prim.form x) (Tag 0)
let received log = List.rev !log
let names = list string

(* An effect handler's exception clause that keeps the raise's backtrace. *)
let reraise e = Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())

(* An [Invalid_argument] message naming [by] and the interpretation [name]. *)
let names_both ~by name m =
  starts_with ~prefix:(by ^ ": ") m;
  contains ~sub:name m

let placed_at p = Format.asprintf "%a" Nx.Placement.pp p

(* The rule *)

let x3 () = vec f32 [| 1.; 2.; 3. |]

let test_values_reach () =
  let log = ref [] in
  let x = x3 () in
  logged ~name:"test.values" Values log (fun i ->
      ignore (Nx.add (tag i x) x);
      let y = Nx.add x x in
      equal ~msg:"computed" (array float_exact) [| 2.; 4.; 6. |] (elements y);
      ignore (Nx.zeros f32 [| 2 |]));
  equal ~msg:"received" names [ "Map" ] (received log)

let test_extent_reach () =
  let log = ref [] in
  let x = Nx.place Count.on (x3 ()) in
  C.reset ();
  logged ~name:"test.extent" Extent log (fun _ ->
      ignore (Nx.zeros f32 [| 2 |]);
      ignore (Nx.add x x);
      ignore (Nx.copy x));
  equal ~msg:"received" names [ "Map"; "Map"; "Copy" ] (received log);
  equal ~msg:"kernel calls" int 0 (C.calls ());
  ignore (Nx.add x x);
  equal ~msg:"after its extent" int 1 (C.calls ())

let test_innermost () =
  let outer = ref [] and inner = ref [] in
  let x = x3 () in
  logged ~name:"outer" Values outer (fun i ->
      let t = tag i x in
      logged ~name:"inner" Extent inner (fun _ -> ignore (Nx.add t x)));
  equal ~msg:"an extent started later" names [ "Map" ] (received inner);
  equal ~msg:"the values started earlier" names [] (received outer);
  let outer = ref [] and inner = ref [] in
  logged ~name:"outer" Extent outer (fun _ ->
      logged ~name:"inner" Values inner (fun i ->
          ignore (Nx.add (tag i x) x);
          ignore (Nx.add x x)));
  equal ~msg:"values started later" names [ "Map" ] (received inner);
  equal ~msg:"the extent, for a concrete operation" names [ "Map" ]
    (received outer)

(* An operation on its own value inside its rule raises naming it, as outside
   its extent, also under an extent around it. *)
let test_own_value () =
  let x = x3 () in
  let own i ~by op =
    let (Nx.Prim.Operands xs) = Nx.Prim.operands op in
    let (Nx.Prim.Any x) = List.hd xs in
    ignore (Nx.copy (tag i x));
    Nx.Prim.eval ~by op
  in
  let outside =
    "Nx.copy: a value of test.own, used outside test.own's extent"
  in
  invalid ~sub:outside (fun () ->
      Nx.Prim.interpret ~name:"test.own" Values own (fun i ->
          ignore (Nx.add (tag i x) x)));
  invalid ~sub:outside (fun () ->
      logged ~name:"test.extent" Extent (ref []) (fun _ ->
          Nx.Prim.interpret ~name:"test.own" Values own (fun i ->
              ignore (Nx.add (tag i x) x))))

(* A rule reads an operation as it was called: one handle at two operands,
   as in [add h h], is read as eager execution reads it. A donor beside its
   own handle raises at the rule's operation. *)
let test_donor_beside () =
  let x = x3 () in
  let h = Nx.donate x in
  let under f =
    Nx.Prim.interpret ~name:"test.extent" Extent
      (fun _ ~by op -> Nx.Prim.eval ~by op)
      (fun _ -> f ())
  in
  equal (array float_exact) [| 2.; 4.; 6. |]
    (elements (under (fun () -> Nx.add h h)));
  (* A later use of the handle reads its elements or names the call. *)
  (match elements (Nx.copy h) with
  | xs -> equal ~msg:"the handle" (array float_exact) [| 1.; 2.; 3. |] xs
  | exception Invalid_argument m ->
      equal ~msg:"the handle" string
        "Nx.copy: float32 [3] was donated to Nx.add" m);
  let x = x3 () in
  raises (Invalid_argument "Nx.add: float32 [3] was donated") (fun () ->
      under (fun () -> Nx.add (Nx.donate x) x))

(* The [by]s a rule receives while [f] runs under it, each once, in order. *)
let observed_names f =
  let names = ref [] in
  let observe _ ~by op =
    if not (List.mem by !names) then names := by :: !names;
    Nx.Prim.eval ~by op
  in
  Nx.Prim.interpret ~name:"test.names" Extent observe (fun _ -> ignore (f ()));
  List.rev !names

let square64 () = Nx.create f64 [| 2; 2 |] [| 2.; 0.; 0.; 1. |]

(* Every operation a function makes reaches a rule named by the function the
   user called, whatever functions that one computes through. *)
let test_by_called () =
  let m = square64 () in
  equal ~msg:"trace" names [ "Nx.trace" ] (observed_names (fun () -> Nx.trace m));
  equal ~msg:"inv" names [ "Nx.inv" ] (observed_names (fun () -> Nx.inv m));
  equal ~msg:"cholesky Upper" names [ "Nx.cholesky" ]
    (observed_names (fun () -> Nx.cholesky ~triangle:Upper m));
  equal ~msg:"a call after the composite" names [ "Nx.trace"; "Nx.neg" ]
    (observed_names (fun () -> Nx.neg (Nx.trace m)))

(* A rule's refusal reaches the caller as the rule raised it, from inside a
   composite too; one in the name the rule received names the function the
   user called. *)
let test_refusal_named () =
  let a = square64 () and b = vec f64 [| 1.; 2. |] in
  let lstsq why =
    Nx.Prim.interpret ~name:"test.refuse" Extent
      (fun _ ~by op ->
        match op with
        | Nx.Prim.Linalg _ -> invalid_arg (why by)
        | _ -> Nx.Prim.eval ~by op)
      (fun _ -> ignore (Nx.lstsq f64 a b))
  in
  raises (Invalid_argument "test.refuse: no factorisation") (fun () ->
      lstsq (fun _ -> "test.refuse: no factorisation"));
  raises (Invalid_argument "Nx.lstsq: test.refuse has no factorisation")
    (fun () -> lstsq (fun by -> by ^ ": test.refuse has no factorisation"))

(* A rule is the user's function: a refusal of an nx function it calls reaches
   the caller as raised, from inside a composite too. *)
let test_rule_calls_named () =
  let a = square64 () and b = vec f64 [| 1.; 2. |] in
  let reshaping _ ~by op =
    (match op with
    | Nx.Prim.Linalg _ -> ignore (Nx.reshape [| 3 |] b)
    | _ -> ());
    Nx.Prim.eval ~by op
  in
  invalid ~sub:"Nx.reshape: " (fun () ->
      Nx.Prim.interpret ~name:"test.reshaping" Extent reshaping (fun _ ->
          Nx.lstsq f64 a b))

(* What the function and the rule raise reach the caller with the backtrace
   of their raise. *)
let test_raised_backtrace () =
  let raise_exit () = raise (Sys.opaque_identity Exit) in
  let x = vec f64 [| 1.; 2. |] in
  equal ~msg:"the function" string "test_interp.ml"
    (Nx_support.raised_in (fun () ->
         Nx.Prim.interpret ~name:"test.raise" Values
           (fun _ ~by op -> Nx.Prim.eval ~by op)
           (fun _ -> raise_exit ())));
  equal ~msg:"the rule" string "test_interp.ml"
    (Nx_support.raised_in (fun () ->
         Nx.Prim.interpret ~name:"test.raise" Extent
           (fun _ ~by:_ _ -> raise_exit ())
           (fun _ -> Nx.neg x)))

(* What a rule donates dies at the donation, and the handle it consumes
   names the function the rule called, inside a composite too. *)
let test_rule_donates () =
  (* A copy, so that the value owns its memory. *)
  let own = Nx.copy (vec f64 [| 1.; 2. |]) and spent = ref None in
  Nx.Prim.interpret ~name:"test.donor" Extent
    (fun _ ~by op ->
      if Option.is_none !spent then begin
        let h = Nx.donate own in
        spent := Some h;
        ignore (Nx.neg h)
      end;
      Nx.Prim.eval ~by op)
    (fun _ -> ignore (Nx.trace (square64 ())));
  raises (Invalid_argument "Nx.copy: float64 [2] was donated") (fun () ->
      Nx.copy own);
  invalid ~sub:"Nx.copy: float64 [2] was donated to Nx.neg" (fun () ->
      Nx.copy (Option.get !spent))

(* A rule inside [trace] performs an effect whose handler resumes the rest of
   the call on a second domain, as a work-stealing scheduler does. A plain
   operation on the first domain afterwards is named by its own function: the
   [by] a rule receives, and its handle's death where it computes now. *)
type _ Effect.t += Hop : unit Effect.t

let test_names_after_hop () =
  let hopping f =
    Effect.Deep.match_with f ()
      {
        retc = Fun.id;
        exnc = reraise;
        effc =
          (fun (type a) (e : a Effect.t) ->
            match e with
            | Hop ->
                Some
                  (fun (k : (a, unit) Effect.Deep.continuation) ->
                    Domain.join
                      (Domain.spawn (fun () -> Effect.Deep.continue k ())))
            | _ -> None);
      }
  in
  let hopped = ref false in
  hopping (fun () ->
      Nx.Prim.interpret ~name:"test.hop" Extent
        (fun _ ~by op ->
          if not !hopped then begin
            hopped := true;
            Effect.perform Hop
          end;
          Nx.Prim.eval ~by op)
        (fun _ -> ignore (Nx.trace (square64 ()))));
  satisfies ~msg:"hopped" ~claim:"the rule's call resumed on another domain"
    bool Fun.id !hopped;
  equal ~msg:"a rule's by" names [ "Nx.neg" ]
    (observed_names (fun () -> Nx.neg (Nx.donate (vec f64 [| 1.; 2. |]))));
  let h = Nx.donate (Nx.copy (vec f64 [| 1.; 2. |])) in
  ignore (Nx.neg h);
  invalid ~sub:"Nx.copy: float64 [2] was donated to Nx.neg" (fun () ->
      Nx.copy h)

(* One systhread holds a composite, blocked in its rule, while another of the
   same domain runs a plain operation: the [by] each operation's rule receives
   names its own function. *)
let test_names_two_threads () =
  let m = Mutex.create () and c = Condition.create () in
  let inside = ref false and plain_done = ref false in
  let set flag =
    Mutex.lock m;
    flag := true;
    Condition.broadcast c;
    Mutex.unlock m
  in
  let await flag =
    Mutex.lock m;
    while not !flag do
      Condition.wait c m
    done;
    Mutex.unlock m
  in
  let a = square64 () and composite = ref [] in
  let held =
    Thread.create
      (fun () ->
        let names = ref [] in
        Nx.Prim.interpret ~name:"test.held" Extent
          (fun _ ~by op ->
            if not (List.mem by !names) then names := by :: !names;
            if not !inside then begin
              set inside;
              await plain_done
            end;
            Nx.Prim.eval ~by op)
          (fun _ -> ignore (Nx.trace (Nx.donate a)));
        composite := List.rev !names)
      ()
  in
  await inside;
  let x = vec f64 [| 1.; 2. |] in
  let plain = observed_names (fun () -> Nx.neg (Nx.donate x)) in
  set plain_done;
  Thread.join held;
  equal ~msg:"the plain operation's by" names [ "Nx.neg" ] plain;
  equal ~msg:"the composite's by" names [ "Nx.trace" ] !composite

let test_after_return () =
  let x = x3 () in
  let i, t =
    logged ~name:"test.values" Values (ref []) (fun i -> (i, tag i x))
  in
  invalid ~sub:"Nx.add: a value of test.values, used after test.values returned"
    (fun () -> Nx.add t x);
  let f = Nx.Prim.form t in
  equal ~msg:"its form answers"
    (pair string (array int))
    ("float32", [| 3 |])
    (D.name f.dtype, Iarray.to_array (L.shape f.layout));
  satisfies ~msg:"its owner answers" ~claim:"the interpretation that made it"
    (option pass)
    (function Some o -> o == i | None -> false)
    (Nx.Prim.owner t);
  equal ~msg:"its payload answers" (option int) (Some 0)
    (match Nx.Prim.payload i t with Some (Tag k) -> Some k | _ -> None);
  invalid ~sub:"test.array: a value traced by test.values has no bytes"
    (fun () -> Nx.Repr.array ~by:"test.array" t);
  let leaked = ref None in
  (try
     logged ~name:"test.raised" Values (ref []) (fun i ->
         leaked := Some (tag i x);
         failwith "f raises")
   with Failure _ -> ());
  match Nx.add (Option.get !leaked) x with
  | _ -> fail "a value of an interpretation whose f raised computed"
  | exception Invalid_argument m -> names_both ~by:"Nx.add" "test.raised" m

let test_domains () =
  let x = x3 () in
  let log = ref [] in
  logged ~name:"test.values" Values log (fun i ->
      let t = tag i x in
      let elsewhere () =
        match Nx.add t x with
        | _ -> None
        | exception Invalid_argument m -> Some m
      in
      match Domain.join (Domain.spawn elsewhere) with
      | Some m -> names_both ~by:"Nx.add" "test.values" m
      | None -> fail "a traced value computed on another domain");
  let log = ref [] in
  let y =
    logged ~name:"test.extent" Extent log (fun _ ->
        Domain.join (Domain.spawn (fun () -> Nx.add x x)))
  in
  equal ~msg:"received" names [] (received log);
  equal ~msg:"another domain computes" (array float_exact) [| 2.; 4.; 6. |]
    (elements y)

let test_systhreads () =
  let x = x3 () in
  logged ~name:"test.values" Values (ref []) (fun i ->
      let t = tag i x and got = ref None in
      let elsewhere () =
        got :=
          match Nx.add t x with
          | _ -> None
          | exception Invalid_argument m -> Some m
      in
      Thread.join (Thread.create elsewhere ());
      match !got with
      | Some m -> names_both ~by:"Nx.add" "test.values" m
      | None -> fail "a traced value computed on another systhread");
  let log = ref [] in
  let y =
    logged ~name:"test.extent" Extent log (fun _ ->
        let y = ref None in
        Thread.join (Thread.create (fun () -> y := Some (Nx.add x x)) ());
        Option.get !y)
  in
  equal ~msg:"received" names [] (received log);
  equal ~msg:"another systhread computes" (array float_exact) [| 2.; 4.; 6. |]
    (elements y)

(* A creation made outside an interpretation is a value with bytes at its
   placement: a Values rule reads it as it is, and nothing computes again. *)
let test_creation_values () =
  let c = Nx.zeros_at Count.on f32 [| 3 |] in
  let x = on_count f32 [| 1.; 2.; 3. |] in
  let placed = ref [] in
  let looking i ~by op =
    let (Nx.Prim.Operands xs) = Nx.Prim.operands op in
    List.iter
      (fun (Nx.Prim.Any y) ->
        if Nx.Prim.owner y = None then
          placed := placed_at (Nx.placement y) :: !placed)
      xs;
    tagged i ~by op
  in
  C.reset ();
  Nx.Prim.interpret ~name:"test.values" Values looking (fun i ->
      ignore (Nx.add (tag i x) c));
  equal ~msg:"at its placement" (list string) [ placed_at Count.on ] !placed;
  equal ~msg:"kernel calls" int 0 (C.calls ())

(* The name of an operation, a creation told from a map. *)
let kind (type r) (op : r Nx.Prim.t) =
  match op with Map { loads = [||]; _ } -> "Creation" | _ -> Nx.Prim.name op

(* An Extent rule receives a creation as the map of no loads it is, at its
   placement, which results gives its result; one made outside the extent
   reaches it as the value it is. *)
let test_captured () =
  let outside = Nx.zeros_at Count.on f32 [| 2 |] in
  let log = ref [] and named = ref [] and placed = ref [] in
  let looking i ~by op =
    log := kind op :: !log;
    (match op with
    | Map { loads = [||]; at = Some p; _ } -> named := placed_at p :: !named
    | _ -> ());
    let r = tagged i ~by op in
    Nx.Prim.iter_results
      (fun y -> placed := placed_at (Nx.placement y) :: !placed)
      op r;
    r
  in
  C.reset ();
  Nx.Prim.interpret ~name:"test.extent" Extent looking (fun _ ->
      ignore (Nx.add outside (Nx.zeros_at Count.on f32 [| 2 |])));
  equal ~msg:"received" names [ "Creation"; "Map" ] (received log);
  equal ~msg:"the creation names its placement" (list string)
    [ placed_at Count.on ]
    !named;
  equal ~msg:"results lie there" (list string)
    [ placed_at Count.on; placed_at Count.on ]
    !placed;
  equal ~msg:"kernel calls" int 0 (C.calls ())

(* Forms: results gives the form eager execution gives *)

type case = Case : string * 'r Nx.Prim.t -> case
type kind = Concrete | View | Created

let pp_kind ppf k =
  Format.pp_print_string ppf
    (match k with
    | Concrete -> "concrete"
    | View -> "view"
    | Created -> "created")

let pp_shape ppf s =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))

(* A float32 operand of [shape] on Count's set: an array, a transposed view of
   one, or a creation there. *)
let operand kind shape : (float, D.float32_elt, Count.d) Nx.t =
  let n = Array.fold_left ( * ) 1 shape in
  let data = Array.init n (fun i -> Float.of_int i -. 1.5) in
  match kind with
  | Created -> Nx.zeros_at Count.on f32 shape
  | Concrete ->
      Nx.place Count.on
        (Nx.Repr.of_array Nx.Host.v
           (A.of_array D.Float32 (Iarray.of_array shape) data))
  | View ->
      let r = Array.length shape in
      let rev = Array.init r (fun i -> r - 1 - i) in
      let t = Iarray.of_array (Array.map (fun a -> shape.(a)) rev) in
      let base =
        Nx.place Count.on
          (Nx.Repr.of_array Nx.Host.v (A.of_array D.Float32 t data))
      in
      Nx.Prim.eval ~by:"test" (Move (Permute (Iarray.of_array rev), base))

let add_prog =
  P.v
    ~ins:[| D.Any D.Float32; D.Any D.Float32 |]
    [| In 0; In 1; Op2 (Binary Add, 0, 1) |]
    ~outs:[| 2 |]

(* x + y and (x + y) * y, two results. *)
let chain_prog =
  P.v
    ~ins:[| D.Any D.Float32; D.Any D.Float32 |]
    [| In 0; In 1; Op2 (Binary Add, 0, 1); Op2 (Binary Mul, 2, 1) |]
    ~outs:[| 2; 3 |]

let cases_of shape a b =
  let x = operand a shape and y = operand b shape in
  let r = Array.length shape in
  let numel = Array.fold_left ( * ) 1 shape in
  [
    Case ("copy", Copy x);
    Case ("reverse", Move (Permute (Iarray.init r (fun i -> r - 1 - i)), x));
    Case ("flatten", Move (Reshape [| numel |], x));
    Case
      ( "broadcast",
        Move (Broadcast (Iarray.of_array (Array.append [| 2 |] shape)), x) );
    Case ("int32", Bitcast (Nx.int32, x));
    Case ("bytes", Bitcast (Nx.uint8, x));
    Case ("host", Place (Nx.Host.on, x));
    Case ("count", Place (Count.on, x));
    Case
      ( "add",
        Map
          {
            layout = L.contiguous (Iarray.of_array shape);
            prog = add_prog;
            outs = Nx.Prim.[ f32 ];
            loads = [| Plain x; Plain y |];
            at = None;
          } );
    Case
      ( "chain",
        Map
          {
            layout = L.contiguous (Iarray.of_array shape);
            prog = chain_prog;
            outs = Nx.Prim.[ f32; f32 ];
            loads = [| Plain x; Plain y |];
            at = None;
          } );
  ]

let shapes =
  Gen.of_list ~pp:pp_shape
    [ [||]; [| 0 |]; [| 1 |]; [| 3 |]; [| 2; 3 |]; [| 0; 2 |]; [| 2; 1; 3 |] ]

let kinds = Gen.of_list ~pp:pp_kind [ Concrete; View; Created ]

let forms_case =
  Gen.with_pp
    (fun ppf (shape, a, b, _) ->
      Format.fprintf ppf "%a %a %a" pp_shape shape pp_kind a pp_kind b)
    (Gen.map
       (fun ((shape, a), (b, k)) -> (shape, a, b, k))
       (Gen.pair (Gen.pair shapes kinds) (Gen.pair kinds (Gen.int_range 0 9))))

(* A value of a result, of any dtype and set. *)
type value = V : ('v, 's, 'd) Nx.t -> value

let rec out_values : type d r. (d, r) Nx.Prim.outs -> r -> value list =
 fun outs r ->
  match (outs, r) with
  | [], () -> []
  | _ :: outs, (x, r) -> V x :: out_values outs r

let reduction_values : type d a. (d, a) Nx.Prim.reduction -> a -> value list =
 fun r a ->
  match (r, a) with
  | Monoid _, x -> [ V x ]
  | Logsumexp _, x -> [ V x ]
  | Moments _, (m, v) -> [ V m; V v ]
  | Arg _, (e, i) -> [ V e; V i ]

let rec reductions_values : type d r.
    (d, r) Nx.Prim.reductions -> r -> value list =
 fun rs r ->
  match (rs, r) with
  | [], () -> []
  | x :: rs, (a, r) -> reduction_values x a @ reductions_values rs r

let linalg_values : type d r. (d, r) Nx.Prim.linalg -> r -> value list =
 fun l r ->
  match (l, r) with
  | Cholesky _, x -> [ V x ]
  | Lu _, (lu, pivots, perm) -> [ V lu; V pivots; V perm ]
  | Qr _, (q, r) -> [ V q; V r ]
  | Svd_values _, s -> [ V s ]
  | Svd _, (s, u, vh) -> [ V s; V u; V vh ]
  | Eigh_values _, w -> [ V w ]
  | Eigh _, (w, v) -> [ V w; V v ]
  | Eig_values _, w -> [ V w ]
  | Eig _, (w, v) -> [ V w; V v ]
  | Solve_triangular _, x -> [ V x ]

(* The values of [op]'s result [r], in the order of its type: the reference the
   positions of results and iter_results are read against. *)
let values : type r. r Nx.Prim.t -> r -> value list =
 fun op r ->
  match op with
  | Map { outs; _ } -> out_values outs r
  | Reduce { reductions; _ } -> reductions_values reductions r
  | Scan { reduction; _ } -> reduction_values reduction r
  | Sort _ ->
      let v, p = r in
      [ V v; V p ]
  | Unique _ ->
      let i, f, c, k = r in
      [ V i; V f; V c; V k ]
  | Linalg l -> linalg_values l r
  | Attention _ ->
      let o, l = r in
      [ V o; V l ]
  | Check _ -> []
  | Gather _ -> [ V r ]
  | Gather_ranges _ -> [ V r ]
  | Scatter _ -> [ V r ]
  | Fft (C2c _) -> [ V r ]
  | Fft (R2c _) -> [ V r ]
  | Fft (C2r _) -> [ V r ]
  | Fold _ -> [ V r ]
  | Assemble _ -> [ V r ]
  | Copy _ -> [ V r ]
  | Contract _ -> [ V r ]
  | Move _ -> [ V r ]
  | Bitcast _ -> [ V r ]
  | Place _ -> [ V r ]

let form_of (V x) =
  let f = Nx.Prim.form x in
  Format.asprintf "%s %a at %a" (D.name f.dtype) L.pp f.layout Nx.Placement.pp
    f.placement

(* The forms of [op]'s result traced under a tagging rule, and eagerly. *)
let traced_and_eager_forms op =
  let eager = Nx.Prim.eval ~by:"test" op in
  logged ~name:"test.forms" Values (ref []) (fun i ->
      let traced = tagged i ~by:"test" op in
      (List.map form_of (values op traced), List.map form_of (values op eager)))

let law_eager_forms (shape, a, b, k) =
  cover "a creation" (a = Created || b = Created);
  cover "empty" (Array.mem 0 shape);
  cover "0-d" (shape = [||]);
  let (Case (name, op)) = List.nth (cases_of shape a b) k in
  let traced, eager = traced_and_eager_forms op in
  equal ~msg:name (list string) eager traced

(* A float32 of the coordinate along axis 1 plus load 0: past the rank of a loop
   of one axis. *)
let past_rank =
  P.v ~ins:[| D.Any D.Float32 |]
    [| In 0; Coord 1; Op1 (Cast, D.Any D.Float32, 1); Op2 (Binary Add, 0, 2) |]
    ~outs:[| 3 |]

(* A value of shape [s] holding [xs] on Count's set. *)
let placed dt s xs = Nx.place Count.on (Nx.create dt s xs)

let row i : M.range iarray =
  [| { start = i; count = 1; step = 1 }; { start = 0; count = 3; step = 1 } |]

(* Loops of no loads that name no placement or a misplaced one, and loops of
   loads that name one. *)
let misplaced () =
  let x = operand Concrete [| 3 |] and line = L.contiguous [| 3 |] in
  let one = P.of_exprs ~ins:[||] [| P.E.(any (const f32 1.)) |]
  and load = P.v ~ins:[| D.Any D.Float32 |] [| In 0 |] ~outs:[| 0 |] in
  let sum =
    Nx.Placement.partial (Nx.Mesh.v Count2.v [ ("a", 2) ]) [ "a" ] []
  in
  let reduce name ?(loads : _ Nx.Prim.load iarray = [||]) ?(prog = one) at =
    Case
      ( name,
        Reduce
          {
            layout = line;
            axes = [| 0 |];
            prog;
            reductions = Nx.Prim.[ Monoid (Sum, 0, f32) ];
            loads;
            at;
          } )
  in
  let scan name ?(layout = line) ?(loads : _ Nx.Prim.load iarray = [||])
      ?(prog = one) ?carry at =
    Case
      ( name,
        Scan
          {
            layout;
            axis = 0;
            prog;
            reduction = Monoid (Sum, 0, f32);
            loads;
            carry;
            at;
          } )
  in
  (* A carry whole on both devices, beside a scan cut along the axis it
     keeps, which a carry's route would read where it lies. *)
  let map name ?(loads : _ Nx.Prim.load iarray = [||]) ?(prog = one) at =
    Case (name, Map { layout = line; prog; outs = Nx.Prim.[ f32 ]; loads; at })
  in
  let carry = Nx.Prim.Any (Nx.place Count2.on (Nx.zeros f32 [| 1; 2 |])) in
  [
    map "a map of no loads that names no placement" None;
    map "a map of loads that names a placement" ~prog:load ~loads:[| Plain x |]
      (Some Count.on);
    map "a map of no loads at a cut of an axis it lacks"
      (Some (Count2.split ~axis:1));
    reduce "a reduction of no loads that names no placement" None;
    reduce "a reduction of loads that names a placement" ~prog:load
      ~loads:[| Plain x |] (Some Count.on);
    reduce "a reduction of no loads at a cut of an axis it lacks"
      (Some (Count2.split ~axis:1));
    reduce "a reduction of no loads at a sum" (Some sum);
    scan "a scan of no loads that names no placement" None;
    scan "a scan of loads that names a placement" ~prog:load
      ~loads:[| Plain x |] (Some Count.on);
    scan "a scan of no loads at a cut of an axis it lacks"
      (Some (Count2.split ~axis:1));
    scan "a scan of no loads at a sum" (Some sum);
    scan "a scan of no loads from a carry elsewhere"
      ~layout:(L.contiguous [| 3; 2 |])
      ~carry
      (Some (Count2.split ~axis:1));
  ]

(* Ill-formed operations, each refused naming [by]. *)
let ill_formed () =
  let x = operand Concrete [| 3 |] and y = operand Concrete [| 2 |] in
  let x23 = placed Nx.int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
  let positions s =
    placed Nx.int64 s (Array.make (Array.fold_left ( * ) 1 s) 0L)
  in
  let flags = placed Nx.bool [| 2 |] [| true; false |] in
  let line = L.contiguous [| 3 |] in
  let none = operand Concrete [| 0; 2 |] in
  let load = P.v ~ins:[| D.Any D.Float32 |] [| In 0 |] ~outs:[| 0 |] in
  let extreme name r =
    Case
      ( name,
        Reduce
          {
            layout = L.contiguous [| 0; 2 |];
            axes = [| 0 |];
            prog = load;
            reductions = r;
            loads = [| Plain none |];
            at = None;
          } )
  in
  [
    extreme "a maximum of no term" Nx.Prim.[ Monoid (Max, 0, f32) ];
    extreme "a position of a minimum of no term" Nx.Prim.[ Arg (Min, 0, f32) ];
    Case
      ( "loads of another shape",
        Map
          {
            layout = L.contiguous [| 3 |];
            prog = add_prog;
            outs = Nx.Prim.[ f32 ];
            loads = [| Plain x; Plain y |];
            at = None;
          } );
    Case
      ( "a load of another dtype",
        Map
          {
            layout = L.contiguous [| 3 |];
            prog = add_prog;
            outs = Nx.Prim.[ f32 ];
            loads = [| Plain x; Plain (Nx.cast Nx.int32 x) |];
            at = None;
          } );
    Case ("a permutation of another rank", Move (Permute [| 1; 0 |], x));
    Case ("a reshape of another size", Move (Reshape [| 4 |], x));
    Case ("a wider bitcast of an odd axis", Bitcast (Nx.float64, x));
    Case
      ( "a map's coordinate past its rank",
        Map
          {
            layout = line;
            prog = past_rank;
            outs = Nx.Prim.[ f32 ];
            loads = [| Plain x |];
            at = None;
          } );
    Case
      ( "a reduction's coordinate past its rank",
        Reduce
          {
            layout = line;
            axes = [| 0 |];
            prog = past_rank;
            reductions = Nx.Prim.[ Monoid (Sum, 0, f32) ];
            loads = [| Plain x |];
            at = None;
          } );
    Case
      ( "a scan's coordinate past its rank",
        Scan
          {
            layout = line;
            axis = 0;
            prog = past_rank;
            reduction = Monoid (Sum, 0, f32);
            loads = [| Plain x |];
            carry = None;
            at = None;
          } );
    Case
      ( "a gather's positions of another rank",
        Gather { axis = 0; idx = positions [| 2 |]; x = x23 } );
    Case
      ( "a gather along an axis past the rank",
        Gather { axis = 2; idx = positions [| 2; 3 |]; x = x23 } );
    Case
      ( "a scatter's positions and updates of two shapes",
        Scatter
          {
            combine = Set;
            unique = false;
            axis = 1;
            idx = positions [| 2; 1 |];
            updates = placed Nx.int32 [| 2; 2 |] [| 0l; 0l; 0l; 0l |];
            into = x23;
          } );
    Case
      ( "a scatter that adds bools",
        Scatter
          {
            combine = Add;
            unique = false;
            axis = 0;
            idx = positions [| 2 |];
            updates = flags;
            into = flags;
          } );
    Case
      ( "an assembly's piece of another shape than its region",
        Assemble
          {
            dtype = Nx.int32;
            shape = [| 2; 3 |];
            fill = 0l;
            pieces = [ (row 0, x23) ];
          } );
    Case
      ( "data of another shape",
        Check
          {
            ok = Nx.less x x;
            data = [ y ];
            fail = (fun _ _ -> Failure "fails");
          } );
  ]
  @ misplaced ()

let case_id (Case (name, _)) = name

(* [op] raises naming [by] eagerly, under either reach and from results, before
   any kernel and before results makes a value. *)
let refused (Case (_, op)) =
  C.reset ();
  invalid ~sub:"test.by: " (fun () -> Nx.Prim.eval ~by:"test.by" op);
  List.iter
    (fun reach ->
      invalid ~sub:"test.by: " (fun () ->
          logged ~name:"test" reach (ref []) (fun i ->
              let traced =
                Nx.Prim.map
                  (fun x -> if reach = Nx.Prim.Values then tag i x else x)
                  op
              in
              Nx.Prim.eval ~by:"test.by" traced)))
    [ Nx.Prim.Values; Extent ];
  let made = ref 0 in
  invalid ~sub:"test.by: " (fun () ->
      Nx.Prim.results ~by:"test.by"
        (fun _ _ ->
          incr made;
          failwith "made")
        op);
  equal ~msg:"values results made" int 0 !made;
  equal ~msg:"kernel calls" int 0 (C.calls ())

let refusal (Case (name, op)) =
  match Nx.Prim.eval ~by:"test.by" op with
  | _ -> name ^ ": computed"
  | exception Invalid_argument m -> name ^ ": " ^ m

(* A selection and a constant move bits: each code of a narrow dtype comes
   back unchanged, its infinities, NaNs, [-0], largest finite value and least
   subnormal among them. *)

(* The bytes of elements of [w] bits holding [codes] in order, a sub-byte
   element in the next bits of its byte from the lowest, a wider one in the
   host's byte order. *)
let pack w codes =
  let n = Array.length codes in
  let b = Bytes.make (max 1 (n * w / 8)) '\000' in
  Array.iteri
    (fun j c ->
      match w with
      | 16 -> Bytes.set_uint16_ne b (2 * j) c
      | _ ->
          let p = j * w in
          let byte = Bytes.get_uint8 b (p / 8) in
          Bytes.set_uint8 b (p / 8) (byte lor (c lsl (p mod 8))))
    codes;
  Array.init (Bytes.length b) (Bytes.get_uint8 b)

let of_bytes (type v s) (dt : (v, s) D.t) bytes : (v, s, Nx.host) Nx.t =
  let w = D.bits dt and u = vec Nx.uint8 bytes in
  let n = Array.length bytes * 8 / w in
  if w > 8 then Nx.bitcast dt (Nx.reshape [| n; w / 8 |] u)
  else Nx.reshape [| n |] (Nx.bitcast dt u)

let raw (type v s d) (x : (v, s, d) Nx.t) =
  let w = D.bits (Nx.dtype x) and n = Nx.numel x in
  let x = if w < 8 then Nx.reshape [| n * w / 8; 8 / w |] x else x in
  let b = Nx.bitcast Nx.uint8 x in
  elements (Nx.reshape [| Nx.numel b |] b)

(* Each dtype outside the base set that is not complex, and the codes its
   constants are drawn from: every code up to eight bits; for the 16-bit floats
   the infinities, a quiet NaN, a signaling NaN with a payload, [-0], the
   largest finite value and the least subnormal. *)
let narrow =
  let every w = List.init (1 lsl w) Fun.id in
  [
    (D.Any D.Float16, [ 0x7c00; 0xfc00; 0x7e00; 0x7c01; 0x8000; 0x7bff; 0x0001 ]);
    ( D.Any D.Bfloat16,
      [ 0x7f80; 0xff80; 0x7fc0; 0x7f81; 0x8000; 0x7f7f; 0x0001 ] );
    (D.Any D.Float8_e4m3fn, every 8);
    (D.Any D.Float8_e5m2, every 8);
    (D.Any D.Float4_e2m1fn, every 4);
    (D.Any D.Int4, every 4);
    (D.Any D.Uint4, every 4);
    (D.Any D.Bit, every 1);
  ]

(* The first bytes where [b] differs from [a], as ["k: a -> b"]. *)
let changed a b =
  let out = ref [] in
  Array.iteri
    (fun k x ->
      if x <> b.(k) && List.length !out < 8 then
        out := Printf.sprintf "%d: 0x%02x -> 0x%02x" k x b.(k) :: !out)
    a;
  List.rev !out

let keeps_bits (type v s) (dt : (v, s) D.t) consts =
  let w = D.bits dt in
  let n = max (1 lsl w) (8 / w) in
  let msg s = D.name dt ^ ": " ^ s in
  let same msg a b = equal ~msg (list string) [] (changed a b) in
  let check msg want (f : 'd. 'd Nx.Placement.t -> int array) =
    same msg want (f Nx.Host.on)
  in
  let x =
    of_bytes dt (pack w (Array.init n (fun j -> j land ((1 lsl w) - 1))))
  in
  let all b = vec Nx.bool (Array.make n b) in
  check (msg "where, true") (raw x) (fun on ->
      let x = Nx.place on x in
      raw (Nx.where (Nx.place on (all true)) x (Nx.zeros_like x)));
  check (msg "where, false") (raw x) (fun on ->
      let x = Nx.place on x in
      raw (Nx.where (Nx.place on (all false)) (Nx.zeros_like x) x));
  let m = max 1 (8 / w) in
  List.iter
    (fun c ->
      let b = pack w [| c |] in
      let bits = String.init (D.bytes dt 1) (fun k -> Char.chr b.(k)) in
      check
        (msg (Printf.sprintf "constant 0x%x" c))
        (pack w (Array.make m c))
        (fun on ->
          let y, () =
            Nx.Prim.eval ~by:"test"
              (Map
                 {
                   layout = L.contiguous [| m |];
                   prog = P.of_node ~ins:[||] (Const (D.Any dt, bits));
                   outs = Nx.Prim.[ dt ];
                   loads = [||];
                   at = Some on;
                 })
          in
          raw y))
    consts

let test_keeps_bits () =
  List.iter (fun (D.Any dt, consts) -> keeps_bits dt consts) narrow

(* Operations of each count of results beside [cases_of]'s, each with its
   count from the constructor's doc: reductions of six results, a scan's
   two, a sort's, a numbering's four, a transform's, factorisations of two
   and three, and a check of none. *)
let counted_cases () =
  let x = operand Concrete [| 2; 3 |] and m = operand Concrete [| 3; 3 |] in
  let layout = L.contiguous [| 2; 3 |] in
  let load = P.v ~ins:[| D.Any D.Float32 |] [| In 0 |] ~outs:[| 0 |] in
  [
    ( Case
        ( "reduce",
          Reduce
            {
              layout;
              axes = [| 1 |];
              prog = load;
              reductions =
                Nx.Prim.
                  [
                    Monoid (Sum, 0, f32);
                    Moments (0, f32);
                    Logsumexp (0, f32);
                    Arg (Max, 0, f32);
                  ];
              loads = [| Plain x |];
              at = None;
            } ),
      6 );
    ( Case
        ( "scan",
          Scan
            {
              layout;
              axis = 1;
              prog = load;
              reduction = Moments (0, f32);
              loads = [| Plain x |];
              carry = None;
              at = None;
            } ),
      2 );
    (Case ("sort", Sort { axis = 1; descending = false; k = None; x }), 2);
    ( Case
        ( "unique",
          Unique
            { size = 2; keys = Nx.bitcast Nx.int64 (Nx.cast Nx.float64 x) } ),
      4 );
    ( Case
        ("r2c", Fft (R2c { dtype = Nx.complex64; axes = [| 1 |]; x })),
      1 );
    (Case ("qr", Linalg (Qr { factors = Reduced; a = m })), 2);
    (Case ("eigh", Linalg (Eigh { dtype = f32; a = m })), 2);
    (Case ("eig", Linalg (Eig { dtype = Nx.complex64; a = m })), 2);
    (Case ("lu", Linalg (Lu m)), 3);
    (Case ("svd", Linalg (Svd { factors = Reduced; dtype = f32; a = m })), 3);
    ( Case
        ( "check",
          Check { ok = Nx.less x x; data = []; fail = (fun _ () -> Exit) } ),
      0 );
  ]

(* Each value [results] makes is tagged with the position it is made at;
   iter_results gives them at positions 0, 1, ... in turn. *)
let test_iter_results () =
  let named = function Case ("chain", _) as c -> (c, 2) | c -> (c, 1) in
  List.iter
    (fun (Case (name, op), n) ->
      logged ~name:"test.results" Values (ref []) (fun i ->
          let seen = ref [] in
          let r =
            Nx.Prim.results ~by:"test"
              (fun k form -> Nx.Prim.traced i form (Tag k))
              op
          in
          Nx.Prim.iter_results
            (fun y ->
              match Nx.Prim.payload i y with
              | Some (Tag k) -> seen := k :: !seen
              | _ -> fail "a value results did not make")
            op r;
          equal ~msg:name (list int) (List.init n Fun.id) (List.rev !seen)))
    (List.map named (cases_of [| 2; 3 |] Concrete Concrete) @ counted_cases ())

(* A forward derivative: a Values interpretation whose values hold a primal and
   a tangent, over maps of one node. *)

type ('v, 's, 'd) Nx.Prim.payload +=
  | Dual : {
      primal : ('v, 's, 'd) Nx.t;
      tangent : ('v, 's, 'd) Nx.t;
    }
      -> ('v, 's, 'd) Nx.Prim.payload

let parts i x =
  match Nx.Prim.payload i x with
  | Some (Dual { primal; tangent }) -> (primal, tangent)
  | _ -> (x, Nx.zeros_like x)

let dual i primal tangent =
  Nx.Prim.traced i (Nx.Prim.form primal) (Dual { primal; tangent })

(* [x] at [dt]'s type. *)
let at_dtype (type v s d) (dt : (v, s) D.t) (Nx.Prim.Any x : d Nx.Prim.any) :
    (v, s, d) Nx.t =
  match D.equal_witness (Nx.dtype x) dt with
  | Some Type.Equal -> x
  | None -> failf "expected %s, got %s" (D.name dt) (D.name (Nx.dtype x))

(* The node of a one-node program, if [prog] is one. *)
let one_node (prog : P.t) =
  let k = Iarray.length prog.ins in
  if Iarray.length prog.nodes = k + 1 && prog.outs = [| k |] then
    Some (Iarray.get prog.nodes k)
  else None

(* The tangent of the one node [node] over [loads], whose primal result is
   [y]. *)
let tangent (type v s d) i node (loads : d Nx.Prim.load iarray)
    (y : (v, s, d) Nx.t) : (v, s, d) Nx.t =
  let p j =
    match Iarray.get loads j with
    | Nx.Prim.Plain x -> Nx.Prim.Any (fst (parts i x))
    | Nx.Prim.Padded _ -> fail "test.jvp: a padded load"
  and t j =
    match Iarray.get loads j with
    | Nx.Prim.Plain x -> Nx.Prim.Any (snd (parts i x))
    | Nx.Prim.Padded _ -> fail "test.jvp: a padded load"
  in
  let dt = Nx.dtype y in
  match (node : P.node) with
  | Op2 (Binary Add, 0, 1) -> Nx.add (at_dtype dt (t 0)) (at_dtype dt (t 1))
  | Op2 (Binary Mul, 0, 1) ->
      Nx.add
        (Nx.mul (at_dtype dt (t 0)) (at_dtype dt (p 1)))
        (Nx.mul (at_dtype dt (p 0)) (at_dtype dt (t 1)))
  | Op1 (Cast, _, 0) ->
      let (Nx.Prim.Any tx) = t 0 in
      Nx.cast dt tx
  | Op1 (Copy, _, 0) -> Nx.copy (at_dtype dt (t 0))
  | Op3 (Where, 0, 1, 2) ->
      Nx.where (at_dtype Nx.bool (p 0)) (at_dtype dt (t 1)) (at_dtype dt (t 2))
  | Op2 (Compare _, _, _) | Const _ | Coord _ -> Nx.zeros_like y
  | _ -> fail "test.jvp: a kind it lacks"

let jvp_rule (type r) i ~by (op : r Nx.Prim.t) : r =
  let both f x =
    let p, t = parts i x in
    dual i (f p) (f t)
  in
  let primal op = Nx.Prim.map (fun x -> fst (parts i x)) op in
  match op with
  | Map ({ outs = [ _ ]; prog; loads; _ } as m) -> (
      match one_node prog with
      | Some node ->
          let y, () = Nx.Prim.eval ~by (primal (Map m)) in
          (dual i y (tangent i node loads y), ())
      | None -> fail "test.jvp: a map of several nodes")
  | Map _ -> fail "test.jvp: a map of several outputs"
  | Reduce _ | Scan _ ->
      invalid_arg (by ^ ": test.jvp has no derivative of a reduction")
  | Gather g ->
      (* Linear in [x]: the positions carry no tangent. *)
      let idx = fst (parts i g.idx) in
      both (fun y -> Nx.Prim.eval ~by (Gather { g with idx; x = y })) g.x
  | Gather_ranges g ->
      (* Linear in [x]: the starts and ends carry no tangent. *)
      let starts = fst (parts i g.starts) and ends = fst (parts i g.ends) in
      let gather y =
        Nx.Prim.eval ~by (Gather_ranges { g with starts; ends; x = y })
      in
      both gather g.x
  | Scatter ({ combine = Set | Add; _ } as s) ->
      (* Linear in the updates and the target. *)
      let idx = fst (parts i s.idx) in
      let pu, tu = parts i s.updates and pt, tt = parts i s.into in
      let scatter updates into =
        Nx.Prim.eval ~by (Scatter { s with idx; updates; into })
      in
      dual i (scatter pu pt) (scatter tu tt)
  | Scatter _ ->
      invalid_arg (by ^ ": test.jvp has no derivative of a max or min scatter")
  | Fft (C2c f) ->
      (* Linear in [x]. *)
      both (fun y -> Nx.Prim.eval ~by (Fft (C2c { f with x = y }))) f.x
  | Fft _ ->
      invalid_arg (by ^ ": test.jvp has no derivative of a real transform")
  | Linalg _ -> invalid_arg (by ^ ": test.jvp has no derivative of a routine")
  | Fold f ->
      (* Linear in [x]. *)
      both (fun y -> Nx.Prim.eval ~by (Fold { f with x = y })) f.x
  | Sort s ->
      (* The tangent's elements, gathered at the positions. *)
      let p, t = parts i s.x in
      let v, idx = Nx.Prim.eval ~by (Sort { s with x = p }) in
      (dual i v (Nx.Prim.eval ~by (Gather { axis = s.axis; idx; x = t })), idx)
  | Unique u ->
      (* Integers: no tangent. *)
      Nx.Prim.eval ~by (Unique { u with keys = fst (parts i u.keys) })
  | Assemble a ->
      (* Linear in the pieces; the fill's tangent is zero. *)
      let assemble fill pieces =
        Nx.Prim.eval ~by (Assemble { a with fill; pieces })
      in
      dual i
        (assemble a.fill
           (List.map (fun (r, x) -> (r, fst (parts i x))) a.pieces))
        (assemble (D.zero a.dtype)
           (List.map (fun (r, x) -> (r, snd (parts i x))) a.pieces))
  | Copy x -> both (fun y -> Nx.Prim.eval ~by (Copy y)) x
  | Contract c ->
      (* Bilinear: the tangent is ta · b + a · tb + tinit. *)
      let pa, ta = parts i c.a and pb, tb = parts i c.b in
      let product a b =
        Nx.Prim.eval ~by (Contract { c with a; b; init = None })
      in
      let init = Option.map (fun x -> fst (parts i x)) c.init in
      let y = Nx.Prim.eval ~by (Contract { c with a = pa; b = pb; init }) in
      let t = Nx.add (product ta pb) (product pa tb) in
      let t =
        match c.init with Some x -> Nx.add t (snd (parts i x)) | None -> t
      in
      dual i y t
  | Move (mv, x) -> both (fun y -> Nx.Prim.eval ~by (Move (mv, y))) x
  | Bitcast _ -> invalid_arg (by ^ ": test.jvp has no derivative of a bitcast")
  | Place (q, x) ->
      let p, t = parts i x in
      dual i (Nx.place q p) (Nx.place q t)
  | Check _ -> Nx.Prim.eval ~by (primal op)
  | Attention _ -> invalid_arg (by ^ ": test.jvp has no derivative of attention")

let jvp f x v =
  Nx.Prim.interpret ~name:"test.jvp" Values jvp_rule (fun i ->
      parts i (f (dual i x v)))

(* DeepChain's shape over the seed vocabulary. *)
let chain x =
  let open Nx in
  let y = mul x x in
  let z = add y x in
  let c = less x (zeros_like x) in
  let w = where c (mul z y) (add z (scalar float64 1.)) in
  reshape (shape x) (copy (mul w x))

let law_jvp (xs, vs) =
  let n = min (Array.length xs) (Array.length vs) in
  let xs = Array.sub xs 0 n and vs = Array.sub vs 0 n in
  (* Away from the kink of [less]. *)
  let xs = Array.map (fun x -> if Float.abs x < 0.01 then 0.5 else x) xs in
  let x = vec f64 xs and v = vec f64 vs in
  let y, t = jvp chain x v in
  equal ~msg:"primal" (array float_exact) (elements (chain x)) (elements y);
  let h = 1e-6 in
  let at s =
    elements (chain (vec f64 (Array.mapi (fun k x -> x +. (s *. vs.(k))) xs)))
  in
  let plus = at h and minus = at (-.h) in
  (* [chain] is x^5 + x^4 below 0 and x^3 + x^2 + x from 0, in four roundings:
     with a = |x| and u = eps / 2, an evaluation errs by at most u (5 a^5 + 4
     a^4), and rounding x + hv to a float moves it by |f'| u a. The central
     difference divides both by 2h, which at a = 3, the worst case, gives (1539
     u + 891 u) / h, about 3.4 eps S / h, S the sum of the terms' magnitudes: 8
     eps S / h allows that with room. Its truncation, h^2 |f3| / 6 with |f3| <=
     468 the third derivative for |x| <= 3 and |v| <= 1, is below 1e-10; the
     kink of [less] lies over 0.01 away. *)
  Array.iteri
    (fun k t ->
      let fd = (plus.(k) -. minus.(k)) /. (2. *. h) in
      let a = Float.abs xs.(k) in
      let terms = (a ** 5.) +. (a ** 4.) +. (a ** 3.) +. (a ** 2.) +. a in
      let tol = (8. *. epsilon_float *. terms /. h) +. 1e-9 in
      at_most
        ~msg:(Printf.sprintf "tangent %d: %h against %h" k t fd)
        float_exact ~than:tol
        (Float.abs (t -. fd)))
    (elements t)

(* A staging compiler: an Extent interpretation whose values are nodes of a
   program. Concrete operands are captured, and the program holds their bytes; a
   value of another live interpretation it refuses. *)

type ('v, 's, 'd) Nx.Prim.payload +=
  | Staged : int -> ('v, 's, 'd) Nx.Prim.payload

type program = {
  mutable ops : string list;  (** Newest first. *)
  mutable held : int;  (** Bytes of the captured operands. *)
  mutable params : int;
  mutable next : int;
}

let program () = { ops = []; held = 0; params = 0; next = 0 }

let bytes x =
  List.fold_left
    (fun n a -> n + D.bytes (A.dtype a) (L.numel (A.layout a)))
    0
    (Iarray.to_list (Nx.Repr.shards ~by:"test" x))

let staging prog i ~by op =
  let (Nx.Prim.Operands xs) = Nx.Prim.operands op in
  List.iter
    (fun (Nx.Prim.Any x) ->
      match Nx.Prim.owner x with
      | Some o when o == i -> ()
      | Some _ ->
          invalid_arg
            (by ^ ": test.stage meets a value of another live interpretation")
      | None -> prog.held <- prog.held + bytes x)
    xs;
  prog.ops <- kind op :: prog.ops;
  Nx.Prim.results ~by
    (fun _ form ->
      prog.next <- prog.next + 1;
      Nx.Prim.traced i form (Staged prog.next))
    op

let stage f args =
  let prog = program () in
  let outs =
    Nx.Prim.interpret ~name:"test.stage" Extent
      (fun i ~by op -> staging prog i ~by op)
      (fun i ->
        f
          (List.map
             (fun a ->
               prog.params <- prog.params + 1;
               Nx.Prim.traced i (Nx.Prim.form a) (Staged 0))
             args))
  in
  (outs, prog)

(* On a set whose kernels decline [Fill], a creation stages as on any
   other. *)
module Dec =
  (val Nx.devices
         ~kernels:(module Nx_support.Declining)
         [ Nx_support.memory 1 ])

let test_staging_declining () =
  let x = Nx.place Dec.on (vec f32 [| 1.; 2. |]) in
  let outs, prog =
    stage
      (fun ts ->
        let t = List.hd ts in
        [ Nx.mul t (Nx.zeros_like t) ])
      [ x ]
  in
  equal ~msg:"operations" names [ "Creation"; "Map" ] (List.rev prog.ops);
  equal ~msg:"held bytes" int 0 prog.held;
  equal ~msg:"results are the program's" (list bool) [ true ]
    (List.map (fun y -> Nx.Prim.owner y <> None) outs)

(* A forwarding observer gets values back from the operations it forwards:
   each operation of a chain reaches it once, under another observer too. A
   start made outside both is a value, which reaches neither. *)
let forwarding name log f =
  Nx.Prim.interpret ~name Extent
    (fun _ ~by op ->
      log := kind op :: !log;
      Nx.Prim.eval ~by op)
    f

let doubling n x =
  let y = ref x in
  for _ = 2 to n do
    y := Nx.add !y !y
  done;
  !y

let test_forwarding () =
  let n = 40 in
  let eager = elements (doubling n (Nx.ones f32 [| 2 |])) in
  let nested start =
    let outer = ref [] and inner = ref [] in
    let y =
      forwarding "test.outer" outer (fun _ ->
          forwarding "test.inner" inner (fun _ -> doubling n (start ())))
    in
    equal ~msg:"as eager" (array float_exact) eager (elements y);
    (List.length !inner, List.length !outer)
  in
  let log = ref [] in
  let y =
    forwarding "test.forward" log (fun _ -> doubling n (Nx.ones f32 [| 2 |]))
  in
  equal ~msg:"one observer" int n (List.length !log);
  equal ~msg:"one observer, as eager" (array float_exact) eager (elements y);
  equal ~msg:"nested: inner, outer" (pair int int) (n, n)
    (nested (fun () -> Nx.ones f32 [| 2 |]));
  let c = Nx.ones f32 [| 2 |] in
  equal ~msg:"nested, from a captured start: inner, outer" (pair int int)
    (n - 1, n - 1)
    (nested (fun () -> c))

(* An operation of each kind reaches a forwarding observer as that
   operation, and computes as eager. *)
type every = Every : string * (unit -> ('v, 's, 'd) Nx.t) -> every

let every () =
  let a () = Nx.arange f32 0 6 1 in
  let m () = Nx.reshape [| 2; 3 |] (a ()) in
  let eval op = Nx.Prim.eval ~by:"test" op in
  [
    Every ("Reduce", fun () -> Nx.sum ~axes:[ 1 ] (m ()));
    Every ("Scan", fun () -> Nx.cumsum ~axis:0 (a ()));
    Every
      ( "Sort",
        fun () ->
          snd (eval (Sort { axis = 0; descending = true; k = None; x = a () }))
      );
    Every ("Gather", fun () -> Nx.take (Nx.arange Nx.int64 1 4 1) (a ()));
    Every
      ( "Scatter",
        fun () ->
          Nx.scatter ~axis:0 (Nx.arange Nx.int64 1 3 1) (Nx.full f32 [| 2 |] 5.)
            (a ()) );
    Every ("Assemble", fun () -> Nx.concatenate ~axis:0 [ a (); a () ]);
    Every
      ("Contract", fun () -> Nx.matmul (m ()) (Nx.reshape [| 3; 2 |] (a ())));
    Every
      ( "Fft",
        fun () ->
          eval (Fft (R2c { dtype = Nx.complex64; axes = [| 0 |]; x = a () })) );
    Every ("Bitcast", fun () -> Nx.bitcast Nx.int32 (a ()));
    Every
      ( "Linalg",
        fun () ->
          let m = Nx.add (Nx.eye f32 3) (Nx.eye f32 3) in
          eval (Linalg (Cholesky { triangle = Lower; a = m })) );
    Every ("Move", fun () -> Nx.transpose (m ()));
    Every ("Copy", fun () -> Nx.copy (Nx.transpose (m ())));
  ]

(* The last two operations a log received. *)
let last_two log =
  match !log with b :: a :: _ -> [ a; b ] | _ -> received log

let host_elements (type v s d) (y : (v, s, d) Nx.t) =
  Array.to_list (A.to_array (host_array y))

(* An operation of each kind, copied under a forwarding observer, reaches it as
   that operation and gives eager's bits. *)
let forwards_as_eager (Every (name, made)) =
  let log = ref [] in
  let y = forwarding "test.forward" log (fun _ -> Nx.copy (made ())) in
  let eager = Nx.copy (made ()) in
  equal ~msg:"bits"
    (Nx_support.elements (Nx.dtype y))
    (host_elements eager) (host_elements y);
  equal ~msg:"received" names [ name; "Copy" ] (last_two log)

(* A compiler that stages each operation and runs the program after its extent
   returns: a staged function over captured creations equals the function
   computed eagerly, and staging computes nothing. *)
type ('v, 's, 'd) Nx.Prim.payload +=
  | Later : ('v, 's, 'd) Nx.t Lazy.t -> ('v, 's, 'd) Nx.Prim.payload

let deferring (type r) i ~by (op : r Nx.Prim.t) : r =
  let force (type v s d) (x : (v, s, d) Nx.t) : (v, s, d) Nx.t =
    match Nx.Prim.payload i x with Some (Later y) -> Lazy.force y | _ -> x
  in
  let r = lazy (Nx.Prim.eval ~by (Nx.Prim.map force op)) in
  let later y v = Nx.Prim.traced i (Nx.Prim.form y) (Later v) in
  match op with
  | Map { outs = [ _ ]; _ } ->
      let y, () = tagged i ~by op in
      (later y (lazy (fst (Lazy.force r))), ())
  | Move _ -> later (tagged i ~by op) r
  | _ -> failf "test.later: %s" (Nx.Prim.name op)

(* [f x] staged by [deferring], and the kernel calls made while staging. *)
let staged_later f x =
  let i, y =
    Nx.Prim.interpret ~name:"test.later" Extent deferring (fun i ->
        (i, f (Nx.Prim.traced i (Nx.Prim.form x) (Later (lazy x)))))
  in
  let calls = C.calls () in
  match Nx.Prim.payload i y with
  | Some (Later y) -> (calls, Lazy.force y)
  | _ -> fail "a result outside the program"

(* A captured value on Count's set: one creation, two, or a creation and a
   broadcast. *)
type captured = Full of float | Sum of float * float | Scaled of float * float

let made c =
  let full a = Nx.full_at Count.on f32 [| 3 |] a in
  match c with
  | Full a -> full a
  | Sum (a, b) -> Nx.add (full a) (full b)
  | Scaled (a, b) -> Nx.mul (full a) (Nx.scalar_at Count.on f32 b)

let captured =
  let v = Gen.float_range (-2.) 2. in
  Gen.map
    (fun (k, a, b) ->
      match k with 0 -> Full a | 1 -> Sum (a, b) | _ -> Scaled (a, b))
    (Gen.triple (Gen.int_range 0 2) v v)

(* Steps of the staged function: an operation of the running value and one of
   the captured values. *)
let staged_case =
  Gen.triple
    (Gen.list ~size:(Gen.int_range 1 3) captured)
    (Gen.list ~size:(Gen.int_range 0 6)
       (Gen.pair (Gen.int_range 0 2) (Gen.int_range 0 2)))
    (Gen.array ~size:(Gen.constant 3) (Gen.float_range (-2.) 2.))

let law_staged (cs, steps, xs) =
  let cs = Array.of_list (List.map made cs) in
  let f t =
    List.fold_left
      (fun y (o, j) ->
        let c = cs.(j mod Array.length cs) in
        match o with 0 -> Nx.add y c | 1 -> Nx.mul y c | _ -> Nx.sub c y)
      t steps
  in
  let x = on_count f32 xs in
  C.reset ();
  let calls, y = staged_later f x in
  equal ~msg:"kernel calls while staging" int 0 calls;
  equal ~msg:"as eager" (array float_exact) (elements (f x)) (elements y)

(* Fibers and domains *)

type _ Effect.t += Yield : unit Effect.t

let yield () = Effect.perform Yield

(* Runs [fs] as fibers taking turns at each yield. A fiber of [dropped] is never
   resumed after its first yield. *)
let schedule ?(dropped = []) fs =
  let ready = Queue.create () in
  let fiber name f () =
    Effect.Deep.match_with f ()
      {
        retc = Fun.id;
        exnc = reraise;
        effc =
          (fun (type a) (e : a Effect.t) ->
            match e with
            | Yield ->
                Some
                  (fun (k : (a, unit) Effect.Deep.continuation) ->
                    if not (List.mem name dropped) then
                      Queue.push (fun () -> Effect.Deep.continue k ()) ready)
            | _ -> None);
      }
  in
  List.iter (fun (name, f) -> Queue.push (fiber name f) ready) fs;
  while not (Queue.is_empty ready) do
    (Queue.pop ready) ()
  done

(* A fiber beside an extent's computes; the extent's own fiber is reached. *)
let test_two_fibers () =
  let log = ref [] and beside = ref [||] in
  let x = x3 () in
  schedule
    [
      ( "extent",
        fun () ->
          logged ~name:"test.extent" Extent log (fun _ ->
              yield ();
              ignore (Nx.copy x)) );
      ("beside", fun () -> beside := elements (Nx.add x x));
    ];
  equal ~msg:"received" names [ "Copy" ] (received log);
  equal ~msg:"beside computes" (array float_exact) [| 2.; 4.; 6. |] !beside

(* A continuation dropped inside an extent leaves the count high: later
   operations still compute, and a later extent is still reached. *)
let test_dropped () =
  let x = x3 () in
  schedule ~dropped:[ "dropped" ]
    [
      ( "dropped",
        fun () ->
          logged ~name:"test.dropped" Extent (ref []) (fun _ -> yield ()) );
    ];
  equal ~msg:"computes" (array float_exact) [| 2.; 4.; 6. |]
    (elements (Nx.add x x));
  let log = ref [] in
  logged ~name:"test.extent" Extent log (fun _ -> ignore (Nx.zeros f32 [| 2 |]));
  equal ~msg:"a later extent" names [ "Map" ] (received log)

(* An extent started on this domain and resumed on another still reaches its
   operations there, and each domain's own extents reach theirs. *)
let test_migrated () =
  let log = ref [] in
  let suspended = ref None in
  Effect.Deep.match_with
    (fun () ->
      logged ~name:"test.migrated" Extent log (fun _ ->
          yield ();
          ignore (Nx.zeros f32 [| 1 |])))
    ()
    {
      retc = Fun.id;
      exnc = reraise;
      effc =
        (fun (type a) (e : a Effect.t) ->
          match e with
          | Yield ->
              Some
                (fun (k : (a, unit) Effect.Deep.continuation) ->
                  suspended := Some (fun () -> Effect.Deep.continue k ()))
          | _ -> None);
    };
  let there = ref [] in
  Domain.join
    (Domain.spawn (fun () ->
         (Option.get !suspended) ();
         logged ~name:"test.there" Extent there (fun _ ->
             ignore (Nx.zeros f32 [| 2 |]))));
  equal ~msg:"the migrated extent, from another domain" names [ "Map" ]
    (received log);
  equal ~msg:"an extent on the other domain" names [ "Map" ] (received there);
  let here = ref [] in
  logged ~name:"test.here" Extent here (fun _ -> ignore (Nx.zeros f32 [| 3 |]));
  equal ~msg:"an extent on this domain" names [ "Map" ] (received here)

(* A traced value used on a fiber that a scheduler outside its interpretation
   runs raises, while the interpretation is live. *)
let test_outer_fiber () =
  let x = x3 () in
  let leaked = ref None and used = ref "" in
  schedule
    [
      ( "values",
        fun () ->
          logged ~name:"test.values" Values (ref []) (fun i ->
              leaked := Some (tag i x);
              yield ()) );
      ( "outer",
        fun () ->
          match Nx.add (Option.get !leaked) x with
          | _ -> used := "computed"
          | exception Invalid_argument m -> used := m );
    ];
  names_both ~by:"Nx.add" "test.values" !used

(* A rule that suspends its fiber suspends its extent with it: its yield
   reaches the scheduler around the interpretation, past the one inside the
   extent; a fiber beside the extent runs, and no fiber inside it runs until the
   rule returns. *)
let test_suspended_rule () =
  let events = ref [] and calls = ref 0 in
  let say e = events := e :: !events in
  let rule i ~by op =
    incr calls;
    let n = !calls in
    say (Printf.sprintf "enter %d" n);
    if n = 1 then yield ();
    say (Printf.sprintf "leave %d" n);
    tagged i ~by op
  in
  let x = x3 () in
  schedule
    [
      ( "extent",
        fun () ->
          Nx.Prim.interpret ~name:"test.extent" Extent rule (fun _ ->
              schedule
                [
                  ( "first",
                    fun () ->
                      yield ();
                      ignore (Nx.copy x) );
                  ( "second",
                    fun () ->
                      yield ();
                      ignore (Nx.copy x) );
                ]) );
      ("beside", fun () -> say "beside");
    ];
  equal ~msg:"events" names
    [ "enter 1"; "beside"; "leave 1"; "enter 2"; "leave 2" ]
    (List.rev !events)

(* A rule's exception raises at the operation, inside the extent, whose
   handlers and finalisers see it. *)
let test_rule_raises () =
  let x = x3 () in
  let events = ref [] in
  Nx.Prim.interpret ~name:"test.values" Values
    (fun _ ~by:_ _ -> failwith "the rule")
    (fun i ->
      match
        Fun.protect
          ~finally:(fun () -> events := "finalised" :: !events)
          (fun () -> Nx.add (tag i x) x)
      with
      | _ -> events := "returned" :: !events
      | exception Failure m -> events := ("caught " ^ m) :: !events);
  equal (list string) [ "finalised"; "caught the rule" ] (List.rev !events)

(* Allocation *)

(* Asking a value for its owner and its payload allocates nothing, as their
   interpretations ask for every operand of every operation. *)
let test_queries_words () =
  let x = x3 () in
  logged ~name:"test.p" Values (ref []) (fun p ->
      let t = tag p x in
      let ask y () =
        ignore (Sys.opaque_identity (Nx.Prim.owner y));
        ignore (Sys.opaque_identity (Nx.Prim.payload p y))
      in
      equal ~msg:"a traced value's" float_exact 0. (Nx_support.words (ask t));
      equal ~msg:"an array's" float_exact 0. (Nx_support.words (ask x)))

(* The words an interpretation's dispatch costs one operation on its value,
   beyond what its rule allocates: 56 once 914bee9af7 dropped the closures and
   records no caller kept. *)
let test_dispatch_words () =
  let x = x3 () in
  logged ~name:"test.p" Values (ref []) (fun p ->
      let t = tag p x in
      let copy () = Nx.copy t in
      at_most float_exact ~than:56. (Nx_support.words copy))

(* Memory

   While backtraces are recorded, the runtime keeps the last exception raised
   reachable. Delivering an operation keeps none of its operands' bytes once it
   returns, wherever it computed. *)

let device = Nx_support.memory 2

module Mem = (val Nx.devices [ device ])

(* The bytes [device] holds after four rounds of collection, each of which
   returns to the device what it collected. *)
let settled () =
  for _ = 1 to 4 do
    Gc.full_major ();
    ignore (Rig.Buffer.create device 0);
    Rig.free_cache device
  done;
  Rig.allocated device

(* Reshapes a fresh value on [device] and drops both: a movement makes a view,
   which allocates nothing. *)
let[@inline never] [@local never] moved () =
  let x = Nx.place Mem.on (Nx.zeros f32 [| 1024 |]) in
  ignore (Sys.opaque_identity (Nx.reshape [| 32; 32 |] x))

(* Raises and catches, so the last exception raised holds nothing. *)
let[@inline never] [@local never] cleared () =
  try raise (Sys.opaque_identity Exit) with Exit -> ()

(* The bytes [device] holds once [f ()] returns, beyond those it held before,
   after one uncounted run, with backtraces recorded. The count before follows a
   raise, so what one run of [f] keeps counts whatever the run before kept. *)
let held_after f =
  let recording = Printexc.backtrace_status () in
  Printexc.record_backtrace true;
  Fun.protect ~finally:(fun () -> Printexc.record_backtrace recording)
  @@ fun () ->
  f ();
  cleared ();
  let before = settled () in
  f ();
  settled () - before

let evaluating name f =
  Nx.Prim.interpret ~name Extent (fun _ ~by op -> Nx.Prim.eval ~by op) f

(* [beside f] is [f ()] while another domain is inside an extent. *)
let beside f =
  let inside = Atomic.make false and stop = Atomic.make false in
  let other =
    Domain.spawn (fun () ->
        evaluating "test.other" (fun _ ->
            Atomic.set inside true;
            while not (Atomic.get stop) do
              Domain.cpu_relax ()
            done))
  in
  while not (Atomic.get inside) do
    Domain.cpu_relax ()
  done;
  Fun.protect
    ~finally:(fun () ->
      Atomic.set stop true;
      Domain.join other)
    f

let test_beside_keeps () = equal int 0 (held_after (fun () -> beside moved))

let test_rule_keeps () =
  let within () = evaluating "test.extent" (fun _ -> moved ()) in
  equal int 0 (held_after within)

(* The rule on two domains: each domain's operations go where the rule says,
   whatever the other domain's interpretations do. The reference is the rule's
   statement: who receives each operation, and what eager execution gives. *)

let value =
  abstract "x" ~pp:(fun ppf xs ->
      Format.fprintf ppf "[%s]"
        (String.concat "; " (Array.to_list (Array.map string_of_float xs))))

let leaked = abstract "t"
let make xs = vec f32 xs
let round32 x = Int32.float_of_bits (Int32.bits_of_float x)
let doubled xs = Array.map (fun x -> round32 (x +. x)) xs

let in_extent x =
  let log = ref [] in
  logged ~name:"test.extent" Extent log (fun _ ->
      ignore (Nx.add x x);
      ignore (Nx.zeros f32 [| 2 |]));
  received log

let in_values x =
  let log = ref [] in
  let y =
    logged ~name:"test.values" Values log (fun i ->
        ignore (Nx.add (tag i x) x);
        Nx.add x x)
  in
  (received log, elements y)

let leak x = logged ~name:"test.leaked" Values (ref []) (fun i -> tag i x)

(* An operation on a value of a returned interpretation raises naming the
   function and the interpretation. *)
let refused_leak = "refused, naming Nx.add and test.leaked"

let holds ~sub m =
  let n = String.length sub in
  let rec from i =
    i + n <= String.length m && (String.sub m i n = sub || from (i + 1))
  in
  from 0

let use t =
  match Nx.add t t with
  | _ -> "computed"
  | exception Invalid_argument m ->
      if String.starts_with ~prefix:"Nx.add: " m && holds ~sub:"test.leaked" m
      then refused_leak
      else m

let floats3 =
  Gen.with_pp
    (fun ppf a ->
      Format.fprintf ppf "[%s]"
        (String.concat "; " (Array.to_list (Array.map string_of_float a))))
    (Gen.array ~size:(Gen.int_range 0 4) (Gen.float_range (-4.) 4.))

let rule_commands =
  [
    command "make" (floats3 @-> makes value) (Array.map round32) make;
    command "leak" (value ^-> makes leaked) (fun _ -> ()) leak;
    command "eager"
      (value ^-> returns (array float_exact))
      doubled
      (fun x -> elements (Nx.add x x));
    command "extent"
      (value ^-> returns names)
      (fun _ -> [ "Map"; "Map" ])
      in_extent;
    command "values"
      (value ^-> returns (pair names (array float_exact)))
      (fun xs -> ([ "Map" ], doubled xs))
      in_values;
    command "use" (leaked ^-> returns string) (fun () -> refused_leak) use;
  ]

(* Twins: item of a value an interpretation made with a twin reads the twin
   where the interpretation is around the call, with no extent between
   them; every other read, and item elsewhere, raises as for a traced
   value. *)

let twin i x = Nx.Prim.twinned i x (Tag 0)

let refused_read ~by name =
  invalid ~sub:(by ^ ": the value is traced by " ^ name)

let read_after ~by name =
  invalid
    ~sub:
      (Printf.sprintf "%s: a value of %s, used after %s returned" by name name)

let specials =
  Gen.with_pp
    (fun ppf a ->
      Format.fprintf ppf "[%s]"
        (String.concat "; " (Array.to_list (Array.map Float.to_string a))))
    (Gen.array ~size:(Gen.int_range 1 5)
       (Gen.of_list [ -0.; 0.; Float.nan; Float.infinity; -1.5; 2.25 ]))

let item_bits i x = Int64.bits_of_float (Nx.item i x)

let twins =
  group "twinned"
    [
      prop "item of a twinned value reads its twin" specials (fun xs ->
          cover "-0" (Array.exists (fun v -> 1. /. v = Float.neg_infinity) xs);
          cover "NaN" (Array.exists Float.is_nan xs);
          let x = vec f64 xs in
          logged ~name:"test.twin" Values (ref []) (fun i ->
              let t = twin i x in
              Array.iteri
                (fun k _ ->
                  equal ~msg:"an element" int64 (item_bits [ k ] x)
                    (item_bits [ k ] t))
                xs;
              let s = twin i (Nx.reshape [||] (Nx.slice [ I 0 ] x)) in
              equal ~msg:"a 0-d value" int64 (item_bits [ 0 ] x)
                (item_bits [] s)));
      test "every other read of a twinned value raises" (fun () ->
          logged ~name:"test.twin" Values (ref []) (fun i ->
              let t = twin i (x3 ()) in
              refused_read ~by:"Nx.to_array" "test.twin" (fun () ->
                  Nx.to_array t);
              refused_read ~by:"Nx.to_bigarray" "test.twin" (fun () ->
                  Nx.to_bigarray Bigarray.float32 t);
              refused_read ~by:"Nx.sync" "test.twin" (fun () -> Nx.sync t);
              refused_read ~by:"test.read" "test.twin" (fun () ->
                  Nx.Repr.read ~by:"test.read" t A.to_array);
              invalid ~sub:"test.array: a value traced by test.twin"
                (fun () -> Nx.Repr.array ~by:"test.array" t)));
      test "item after its interpretation returned raises" (fun () ->
          let t =
            logged ~name:"test.twin" Values (ref []) (fun i -> twin i (x3 ()))
          in
          read_after ~by:"Nx.item" "test.twin" (fun () -> Nx.item [ 0 ] t));
      test "item in the interpretation's own rule raises" (fun () ->
          let reads = ref "" in
          let rule i ~by op =
            (let (Nx.Prim.Operands xs) = Nx.Prim.operands op in
             let (Nx.Prim.Any x) = List.hd xs in
             match Nx.item [ 0 ] x with
             | _ -> reads := "read"
             | exception Invalid_argument m -> reads := m);
            tagged i ~by op
          in
          Nx.Prim.interpret ~name:"test.twin" Values rule (fun i ->
              ignore (Nx.add (twin i (x3 ())) (x3 ())));
          equal string "Nx.item: the value is traced by test.twin" !reads);
      test "item on another domain raises" (fun () ->
          logged ~name:"test.twin" Values (ref []) (fun i ->
              let t = twin i (x3 ()) in
              let read () =
                match Nx.item [ 0 ] t with
                | _ -> "read"
                | exception Invalid_argument m -> m
              in
              equal string "Nx.item: the value is traced by test.twin"
                (Domain.join (Domain.spawn read))));
      test "item inside an extent the owner is around raises" (fun () ->
          logged ~name:"test.twin" Values (ref []) (fun i ->
              let t = twin i (x3 ()) in
              logged ~name:"test.stage" Extent (ref []) (fun _ ->
                  refused_read ~by:"Nx.item" "test.twin" (fun () ->
                      Nx.item [ 0 ] t));
              logged ~name:"test.inner" Values (ref []) (fun _ ->
                  equal ~msg:"a values interpretation between" float_exact 1.
                    (Nx.item [ 0 ] t))));
      test "a twin of a twin reads through both while both are around"
        (fun () ->
          let x = x3 () in
          logged ~name:"test.outer" Values (ref []) (fun o ->
              let outer = twin o x in
              let inner =
                logged ~name:"test.inner" Values (ref []) (fun i ->
                    let inner = twin i outer in
                    equal ~msg:"inside both" float_exact 2.
                      (Nx.item [ 1 ] inner);
                    inner)
              in
              read_after ~by:"Nx.item" "test.inner" (fun () ->
                  Nx.item [ 1 ] inner);
              equal ~msg:"the outer twin" float_exact 2.
                (Nx.item [ 1 ] outer)));
      test "a value without a twin raises" (fun () ->
          logged ~name:"test.twin" Values (ref []) (fun i ->
              refused_read ~by:"Nx.item" "test.twin" (fun () ->
                  Nx.item [ 0 ] (tag i (x3 ())))));
      test "an operation on a twinned value reaches its interpretation"
        (fun () ->
          let log = ref [] in
          logged ~name:"test.twin" Values log (fun i ->
              let t = twin i (x3 ()) in
              match Nx.Prim.payload i (Nx.add t t) with
              | Some (Tag k) -> equal ~msg:"its payload" int 0 k
              | _ -> fail "an operation on a twinned value went elsewhere");
          equal names [ "Map" ] (received log));
    ]

(* Placements: a traced result lies where its eager twin does, and a call
   the route refuses raises the same message eagerly and traced. *)

module Mem4 = (val Nx.devices (List.init 4 Nx_support.memory))

let mesh4 = Nx.Mesh.v Mem4.v [ ("a", 2); ("b", 2) ]

type ('v, 's, 'd) Nx.Prim.payload += Stand_in : ('v, 's, 'd) Nx.Prim.payload

(* The placement of [f x y] over stand-ins of [x]'s and [y]'s forms, under an
   Extent interpretation whose rule gives each result the form eager
   execution gives it. *)
let traced_placement f x y =
  Nx.Prim.interpret ~name:"test.forms" Extent
    (fun i ~by op ->
      Nx.Prim.results ~by (fun _ form -> Nx.Prim.traced i form Stand_in) op)
    (fun i ->
      let stand_in v = Nx.Prim.traced i (Nx.Prim.form v) Stand_in in
      Nx.placement (f (stand_in x) (stand_in y)))

(* Where an operand of [[4; 4]] lies over the four devices. *)
type where =
  | On
  | Device of int
  | Split of int
  | Cuts of int * int  (** Axis [x] over mesh axis [a], [y] over [b]. *)
  | Minor of int

let pp_where ppf = function
  | On -> Format.pp_print_string ppf "on"
  | Device k -> Format.fprintf ppf "m%d" k
  | Split a -> Format.fprintf ppf "split %d" a
  | Cuts (x, y) -> Format.fprintf ppf "cuts %d/a %d/b" x y
  | Minor a -> Format.fprintf ppf "cut %d/b" a

let at = function
  | On -> Mem4.on
  | Device k -> Nx.Placement.device Mem4.v (Nx_support.memory k)
  | Split a -> Mem4.split ~axis:a
  | Cuts (x, y) when x = y -> Nx.Placement.mesh mesh4 [ (x, [ "a"; "b" ]) ]
  | Cuts (x, y) -> Nx.Placement.mesh mesh4 [ (x, [ "a" ]); (y, [ "b" ]) ]
  | Minor a -> Nx.Placement.mesh mesh4 [ (a, [ "b" ]) ]

(* Operations of two operands of [[4; 4]], each reading its axes as a kind
   of rule does: elementwise, a contraction, a reduction, a permutation, a
   scan, reshapes and a sort. *)
type binary = Mem4.d Nx.float32_t -> Mem4.d Nx.float32_t -> Mem4.d Nx.float32_t

let binaries : (string * binary) list =
  [
    ("add", Nx.add);
    ("matmul", fun a b -> Nx.matmul a b);
    ( "a sum along 0 beside y",
      fun x y -> Nx.add (Nx.sum ~axes:[ 0 ] ~keepdims:true x) y );
    ("x transposed by y", fun x y -> Nx.mul (Nx.transpose x) y);
    ("a cumsum along 1 by y", fun x y -> Nx.mul (Nx.cumsum ~axis:1 x) y);
    ( "x flattened and back, by y",
      fun x y -> Nx.mul (Nx.reshape [| 4; 4 |] (Nx.reshape [| 16 |] x)) y );
    ("x sorted along 1, by y", fun x y -> Nx.mul (fst (Nx.sort ~axis:1 x)) y);
  ]

let where =
  let open Gen in
  let axis = int_range 0 1 in
  one_of
    [
      constant On;
      map (fun k -> Device k) (int_range 0 3);
      map (fun a -> Split a) axis;
      map (fun (x, y) -> Cuts (x, y)) (pair axis axis);
      map (fun a -> Minor a) axis;
    ]

(* Checks that [f x y] lies where it does traced, or raises one message both
   ways, and gives where it lies eagerly or its refusal. *)
let agrees f x y =
  let outcome g =
    match g () with p -> Ok p | exception Invalid_argument m -> Error m
  in
  let eager = outcome (fun () -> Nx.placement (f x y)) in
  let traced = outcome (fun () -> traced_placement f x y) in
  (match (eager, traced) with
  | Ok p, Ok q ->
      equal ~msg:"where the result lies" (Nx_support.placement ()) p q
  | Error m, Error m' -> equal ~msg:"the refusal" string m m'
  | Ok _, Error m -> fail ("traced alone raises: " ^ m)
  | Error m, Ok _ -> fail ("eager alone raises: " ^ m));
  eager

let cut_result = function
  | Ok p -> not (Nx.Placement.equal p Mem4.on)
  | Error _ -> false

let law_forms (k, wx, wy) =
  let f = snd (List.nth binaries k) in
  let value i = Nx.reshape [| 4; 4 |] (Nx.arange f32 i (i + 16) 1) in
  let x = Nx.place (at wx) (value 0) and y = Nx.place (at wy) (value 16) in
  let eager = agrees f x y in
  cover "a refusal" (Result.is_error eager);
  cover "a cut result" (cut_result eager)

(* Rows of [x], [[4; 4]], each by the matrix of [y], [[4; 4; 4]], that its
   group names, the ids at [wo]: [[0; 2; 2; 3]] put row 0 in group 0, no
   row in group 1, rows 1 and 2 in group 2 and row 3 in group 3. *)
let by_groups wo x y =
  let ids =
    Nx.place (at wo) (Nx.create Nx.int64 [| 4 |] [| 0L; 2L; 2L; 3L |])
  in
  Nx.Prim.eval ~by:"test.by_groups"
    (Contract
       {
         batch = Iarray.of_list [];
         contracting = Iarray.of_list [ (1, 2) ];
         acc = D.Any f32;
         out = f32;
         a = x;
         b = y;
         init = None;
         a_scale = None;
         b_scale = None;
         segments = Some (Rows { ids; a_axis = 0; b_axis = 0 });
         sum = false;
       })

let along_groups = function
  | Split 0 | Minor 0 -> true
  | Cuts (x, y) -> x = 0 || y = 0
  | On | Device _ | Split _ | Minor _ -> false

let law_by_groups (wo, wx, wy) =
  let x = Nx.place (at wx) (Nx.reshape [| 4; 4 |] (Nx.arange f32 0 16 1)) in
  let y =
    Nx.place (at wy) (Nx.reshape [| 4; 4; 4 |] (Nx.arange f32 16 80 1))
  in
  let eager = agrees (by_groups wo) x y in
  cover "a refusal" (Result.is_error eager);
  cover "ids cut" (wo <> On);
  cover "b cut along its groups, computed"
    (along_groups wy && Result.is_ok eager);
  cover "a cut result" (cut_result eager)

(* An operation of [f], resumed on another domain by a scheduler installed
   inside an extent, lies outside it. *)
let test_inner_fiber_elsewhere () =
  let x = x3 () and log = ref [] and copied = ref [||] in
  logged ~name:"test.extent" Extent log (fun _ ->
      Effect.Deep.match_with
        (fun () ->
          yield ();
          copied := elements (Nx.copy x))
        ()
        {
          retc = Fun.id;
          exnc = reraise;
          effc =
            (fun (type a) (e : a Effect.t) ->
              match e with
              | Yield ->
                  Some
                    (fun (k : (a, unit) Effect.Deep.continuation) ->
                      Domain.join
                        (Domain.spawn (fun () -> Effect.Deep.continue k ())))
              | _ -> None);
        });
  equal ~msg:"received" names [] (received log);
  equal ~msg:"computed there" (array float_exact) [| 1.; 2.; 3. |] !copied

(* Operations as data *)

let sig_of (V x) =
  Format.asprintf "%s %a" (D.name (Nx.dtype x)) Nx.pp_shape (Nx.shape x)

let operand_sigs op =
  let (Nx.Prim.Operands xs) = Nx.Prim.operands op in
  List.map (fun (Nx.Prim.Any x) -> sig_of (V x)) xs

(* [x] with a trailing axis of one: the same dtype, a shape of its own. *)
let widened x = Nx.reshape (Array.append (Nx.shape x) [| 1 |]) x

(* An operation and its operands in the order nx.mli states, each of a dtype and
   shape of its own within the operation. *)
type operands_row =
  | Ops : string * (unit -> 'r Nx.Prim.t * value list) -> operands_row

let z dt s = Nx.zeros dt s
let row_id (Ops (name, _)) = name

let operand_rows =
  let line i : M.range iarray =
    [| { start = i; count = 1; step = 1 }; { start = 0; count = 3; step = 1 } |]
  in
  [
    Ops
      ( "a map's loads",
        fun () ->
          let a = z f32 [| 2; 3 |] and b = z Nx.int32 [| 2; 3 |] in
          ( Map
              {
                layout = L.contiguous [| 2; 3 |];
                prog = add_prog;
                outs = Nx.Prim.[ f32 ];
                loads = [| Plain a; Plain b |];
                at = None;
              },
            [ V a; V b ] ) );
    Ops
      ( "a scan's loads, then its carry",
        fun () ->
          let a = z f32 [| 2; 3 |] and c = z f32 [| 2; 1 |] in
          ( Scan
              {
                layout = L.contiguous [| 2; 3 |];
                axis = 1;
                prog = P.v ~ins:[| D.Any D.Float32 |] [| In 0 |] ~outs:[| 0 |];
                reduction = Monoid (Sum, 0, f32);
                loads = [| Plain a |];
                carry = Some (Any c);
                at = None;
              },
            [ V a; V c ] ) );
    Ops
      ( "a gather's positions, then x",
        fun () ->
          let idx = z Nx.int64 [| 2; 1 |] and x = z f32 [| 2; 3 |] in
          (Gather { axis = 1; idx; x }, [ V idx; V x ]) );
    Ops
      ( "a scatter's positions, updates, then target",
        fun () ->
          let idx = z Nx.int64 [| 2; 1 |] and updates = z f32 [| 2; 1 |] in
          let into = z f32 [| 2; 3 |] in
          ( Scatter
              { combine = Set; unique = false; axis = 1; idx; updates; into },
            [ V idx; V updates; V into ] ) );
    Ops
      ( "an assembly's pieces",
        fun () ->
          let p0 = z f32 [| 1; 3 |] and p1 = z f32 [| 1; 2 |] in
          ( Assemble
              {
                dtype = f32;
                shape = [| 2; 3 |];
                fill = 0.;
                pieces = [ (line 0, p0); (line 1, p1) ];
              },
            [ V p0; V p1 ] ) );
    Ops
      ( "a contraction's a, b, init, scales, then ids",
        fun () ->
          let a = z f32 [| 2; 4 |] and b = z f32 [| 4; 3 |] in
          let init = z f32 [| 2; 3 |] and sa = z f32 [| 2; 1 |] in
          let sb = z Nx.uint8 [| 1; 3 |] and ids = z Nx.int64 [| 2 |] in
          ( Contract
              {
                batch = [||];
                contracting = [| (1, 0) |];
                acc = D.Any f32;
                out = f32;
                a;
                b;
                init = Some init;
                a_scale = Some (Value sa);
                b_scale = Some (Exponent sb);
                segments = Some (Rows { ids; a_axis = 0; b_axis = 0 });
                sum = false;
              },
            [ V a; V b; V init; V sa; V sb; V ids ] ) );
    Ops
      ( "a triangular solve's a, then b",
        fun () ->
          let a = z f32 [| 3; 3 |] and b = z f32 [| 3; 2 |] in
          ( Linalg
              (Solve_triangular
                 {
                   triangle = Lower;
                   adjoint = false;
                   unit_diagonal = false;
                   a;
                   b;
                 }),
            [ V a; V b ] ) );
    Ops
      ( "a check's flags, then its data",
        fun () ->
          let ok = z Nx.bool [| 2 |] and d0 = z f32 [| 2 |] in
          let d1 = z Nx.int32 [| 2 |] in
          ( Check { ok; data = [ d0; d1 ]; fail = (fun _ _ -> Exit) },
            [ V ok; V d0; V d1 ] ) );
    Ops
      ( "a transform's operand",
        fun () ->
          let x = z Nx.complex64 [| 4 |] in
          (Fft (C2c { direction = Forward; axes = [| 0 |]; x }), [ V x ]) );
    Ops
      ( "a copy's operand",
        fun () ->
          let x = z f32 [| 2; 3 |] in
          (Copy x, [ V x ]) );
  ]

let in_order (Ops (_, make)) =
  let op, stated = make () in
  equal ~msg:"operands" (list string) (List.map sig_of stated) (operand_sigs op);
  let seen = ref [] in
  Nx.Prim.iteri (fun i x -> seen := (i, sig_of (V x)) :: !seen) op;
  equal ~msg:"iteri"
    (list (pair int string))
    (List.mapi (fun i x -> (i, sig_of x)) stated)
    (List.rev !seen);
  equal ~msg:"map" (list string)
    (List.map (fun (V x) -> sig_of (V (widened x))) stated)
    (operand_sigs (Nx.Prim.map widened op))

let walks_free (Ops (_, make)) =
  let op, _ = make () in
  equal float_exact 0.
    (Nx_support.words (fun () -> Nx.Prim.iteri (fun _ _ -> ()) op))

let kind_rows : (string * P.node * string) list =
  [
    ("a unary operation", Op1 (Unary Exp, D.Any D.Float32, 0), "Exp");
    ("a binary one", Op2 (Binary Add, 0, 1), "Add");
    ("a selection", Op3 (Where, 0, 1, 2), "Where");
    ("an operand", In 0, "In");
    ("a coordinate", Coord 0, "Iota");
    ("a constant", Const (D.Any D.Float32, P.bits D.Float32 1.), "Fill");
  ]

(* A map whose sum node is read twice. *)
let test_pp () =
  let prog =
    P.v
      ~ins:[| D.Any D.Float32; D.Any D.Float32 |]
      [| In 0; In 1; Op2 (Binary Add, 0, 1); Op2 (Binary Mul, 2, 2) |]
      ~outs:[| 3 |]
  in
  let op =
    Nx.Prim.Map
      {
        layout = L.contiguous [| 3 |];
        prog;
        outs = Nx.Prim.[ f32 ];
        loads = [| Plain (x3 ()); Plain (x3 ()) |];
        at = None;
      }
  in
  expect (Format.asprintf "%a" Nx.Prim.pp op)
  @@ __POS_OF__
       {| Map n2 = add(x0, x1); [mul(n2, n2)] (x0: float32 [3] at CPU) (x1: float32 [3] at CPU) |}

let layout_t = Testable.make ~pp:L.pp ~equal:L.equal

let test_form_cut () =
  let x = Nx.place (Count2.split ~axis:0) (vec f32 [| 1.; 2.; 3.; 4. |]) in
  let f = Nx.Prim.form x in
  equal ~msg:"layout" layout_t (L.contiguous [| 4 |]) f.layout;
  equal ~msg:"placement" string
    (placed_at (Nx.placement x))
    (placed_at f.placement)

let test_form_view () =
  let v =
    Nx.transpose (Nx.create f32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |])
  in
  let f = Nx.Prim.form v in
  equal ~msg:"layout" layout_t
    (A.layout (Option.get (Nx.Repr.array ~by:"test" v)))
    f.layout;
  equal ~msg:"dtype" string "float32" (D.name f.dtype)

let pp_reduction ppf (r, k, dt) =
  let name =
    match (r : Nx_kernel.reduction) with
    | Monoid Sum -> "Monoid Sum"
    | Monoid Prod -> "Monoid Prod"
    | Monoid Max -> "Monoid Max"
    | Monoid Min -> "Monoid Min"
    | Logsumexp -> "Logsumexp"
    | Moments -> "Moments"
    | Arg Max -> "Arg Max"
    | Arg Min -> "Arg Min"
  in
  Format.fprintf ppf "%s of output %d, first result %s" name k dt

let reduction_t = Testable.make ~pp:pp_reduction ~equal:( = )
let shown (r, k, D.Any dt) = (r, k, D.name dt)

let test_reduction () =
  equal ~msg:"sum" reduction_t (Monoid Sum, 2, "float32")
    (shown (Nx.Prim.reduction (Monoid (Sum, 2, f32))));
  equal ~msg:"moments" reduction_t (Moments, 1, "float64")
    (shown (Nx.Prim.reduction (Moments (1, f64))));
  equal ~msg:"arg" reduction_t (Arg Max, 0, "int32")
    (shown (Nx.Prim.reduction (Arg (Max, 0, Nx.int32))));
  equal ~msg:"reductions" (list reduction_t)
    [ (Monoid Sum, 2, "float32"); (Arg Min, 0, "int32") ]
    (List.map shown
       (Array.to_list
          (Nx.Prim.reductions
             Nx.Prim.[ Monoid (Sum, 2, f32); Arg (Min, 0, Nx.int32) ])))

let test_outputs () =
  let (Nx.Prim.Outputs (outs, read)) =
    Nx.Prim.outputs [ D.Any D.Float32; D.Any D.Int32 ]
  in
  let prog =
    P.v ~ins:[| D.Any D.Float32 |]
      [| In 0; Op1 (Cast, D.Any D.Float32, 0); Op1 (Cast, D.Any D.Int32, 0) |]
      ~outs:[| 1; 2 |]
  in
  let r =
    Nx.Prim.eval ~by:"test"
      (Map
         {
           layout = L.contiguous [| 3 |];
           prog;
           outs;
           loads = [| Plain (x3 ()) |];
           at = None;
         })
  in
  equal (list string) [ "float32"; "int32" ]
    (List.map (fun (Nx.Prim.Any y) -> D.name (Nx.dtype y)) (read r))

(* Results *)

let counted name =
  if name = "chain" then
    List.find
      (fun (Case (n, _)) -> n = "chain")
      (cases_of [| 2; 3 |] Concrete Concrete)
  else fst (List.find (fun (Case (n, _), _) -> n = name) (counted_cases ()))

let counted_names =
  [
    "chain";
    "reduce";
    "scan";
    "sort";
    "unique";
    "r2c";
    "qr";
    "eigh";
    "eig";
    "lu";
    "svd";
    "check";
  ]

let test_counted_forms name =
  let (Case (_, op)) = counted name in
  let traced, eager = traced_and_eager_forms op in
  equal (list string) eager traced

let test_positions name =
  let (Case (_, op)) = counted name in
  logged ~name:"test.positions" Values (ref []) (fun i ->
      let r = tagged i ~by:"test" op in
      let tags =
        List.map
          (fun (V y) ->
            match Nx.Prim.payload i y with Some (Tag k) -> k | _ -> -1)
          (values op r)
      in
      equal (list int) (List.init (List.length tags) Fun.id) tags)

let placements_case =
  Gen.with_pp
    (fun ppf (k, wx, wy) ->
      Format.fprintf ppf "%s, x at %a, y at %a"
        (fst (List.nth binaries k))
        pp_where wx pp_where wy)
    Gen.(triple (int_range 0 (List.length binaries - 1)) where where)

let groups_case =
  Gen.with_pp
    (fun ppf (wo, wx, wy) ->
      Format.fprintf ppf "ids at %a, a at %a, b at %a" pp_where wo pp_where wx
        pp_where wy)
    Gen.(
      triple
        (frequency [ (3, constant On); (1, constant (Split 0)) ])
        (frequency [ (3, constant On); (1, where) ])
        where)

let () =
  exit
    (run "Nx.Prim"
       [
         group "reach"
           [
             test
               "a Values interpretation receives the operations on its values \
                alone"
               test_values_reach;
             test
               "an Extent interpretation receives every operation of its \
                extent, creations included, and none computes"
               test_extent_reach;
             test
               "the innermost interpretation that reaches an operation \
                receives it"
               test_innermost;
             test
               "an operation reaches a rule as it was called: one handle at \
                two operands, a donor beside its handle refused"
               test_donor_beside;
           ];
         group "extent"
           [
             test "a domain started in an extent lies outside it" test_domains;
             test "a systhread started in an extent lies outside it"
               test_systhreads;
             test
               "a fiber an outer scheduler runs beside an extent lies outside \
                it"
               test_two_fibers;
             test "a fiber an outer scheduler runs refuses a traced value"
               test_outer_fiber;
             test
               "an extent resumed on another domain receives its operations \
                there"
               test_migrated;
             test
               "a fiber a scheduler inside the extent resumes on another \
                domain lies outside it"
               test_inner_fiber_elsewhere;
             test
               "an extent whose continuation is dropped leaves later \
                operations computing and later extents receiving"
               test_dropped;
             stateful ~domains:2
               "operations on two domains reach their own domain's \
                interpretations"
               rule_commands;
           ];
         group "rule"
           [
             test
               "a rule's operations reach the interpretations around its own, \
                each once"
               test_forwarding;
             test "an operation on its own value in its rule raises naming it"
               test_own_value;
             test
               "a rule that suspends its fiber holds its extent and is entered \
                once"
               test_suspended_rule;
             test
               "a rule's exception raises at the operation, inside the extent"
               test_rule_raises;
             test
               "a creation on a set whose kernels decline it reaches an Extent \
                rule"
               test_staging_declining;
             test
               "a Values rule reads a creation made outside as the value it is"
               test_creation_values;
             test
               "an Extent rule receives a creation at its placement, where \
                results puts its result"
               test_captured;
             test "delivering an operation to a rule keeps none of its bytes"
               test_rule_keeps;
             test
               "an operation beside another domain's extent keeps none of its \
                bytes"
               test_beside_keeps;
           ];
         group "Prim.t"
           [
             cases
               "an ill-formed operation raises naming its function, eagerly, \
                under either reach and from results, before any kernel"
               ~id:case_id (ill_formed ()) refused;
             test
               "a loop names where it lies exactly when it has no loads, and \
                its refusals say why" (fun () ->
                 expect (String.concat "\n" (List.map refusal (misplaced ())))
                 @@ __POS_OF__
                      {|
                   a map of no loads that names no placement: test.by: a map of no loads names no placement
                   a map of loads that names a placement: test.by: a map of 1 loads names a placement
                   a map of no loads at a cut of an axis it lacks: test.by: axis 1 is cut and the shape has rank 1
                   a reduction of no loads that names no placement: test.by: a reduction of no loads names no placement
                   a reduction of loads that names a placement: test.by: a reduction of 1 loads names a placement
                   a reduction of no loads at a cut of an axis it lacks: test.by: axis 1 is cut and the shape has rank 1
                   a reduction of no loads at a sum: test.by: a reduction of no loads at partial [m0; m1], a sum
                   a scan of no loads that names no placement: test.by: a scan of no loads names no placement
                   a scan of loads that names a placement: test.by: a scan of 1 loads names a placement
                   a scan of no loads at a cut of an axis it lacks: test.by: axis 1 is cut and the shape has rank 1
                   a scan of no loads at a sum: test.by: a scan of no loads at partial [m0; m1], a sum
                   a scan of no loads from a carry elsewhere: test.by: a carry at on [m0; m1] for a scan of no loads at split ~axis:1 [m0; m1]
                   |});
             test "a selection or constant keeps each narrow code's bits"
               test_keeps_bits;
             prop ~count:100
               "an operation a rule keeps evaluates after its extent to \
                eager's bits, none computing while staged"
               staged_case law_staged;
           ];
         group "kind"
           [
             cases "a node is named by its innermost constructor"
               ~id:(fun (name, _, _) -> name)
               kind_rows
               (fun (_, node, expected) ->
                 equal string expected (Nx.Prim.kind node));
           ];
         group "pp"
           [
             test "an operation prints its program, a node read twice once"
               test_pp;
           ];
         group "operands"
           [
             cases
               "operands are in the stated order, where iteri and map visit \
                them"
               ~id:row_id operand_rows in_order;
             cases "iteri allocates nothing but for a factorisation or a check"
               ~id:row_id
               (List.filter
                  (fun r ->
                    not
                      (List.mem (row_id r)
                         [
                           "a triangular solve's a, then b";
                           "a check's flags, then its data";
                         ]))
                  operand_rows)
               walks_free;
           ];
         group "form"
           [
             test
               "a value cut over devices has the C-contiguous layout of the \
                whole"
               test_form_cut;
             test "a view's form has its layout" test_form_view;
           ];
         group "reduction and outputs"
           [
             test
               "reduction is what a reduction computes, its output and its \
                first result's dtype"
               test_reduction;
             test "outputs are the outs of their dtypes, read in order"
               test_outputs;
           ];
         group "results"
           [
             prop ~count:630 "results gives the forms eager execution gives"
               forms_case law_eager_forms;
             cases "results gives eager's forms at each count of results"
               ~id:Fun.id counted_names test_counted_forms;
             cases "the value made at position k is the result's k-th"
               ~id:Fun.id counted_names test_positions;
             test "iter_results visits a result's values in position order"
               test_iter_results;
             prop
               ~examples:[ (1, Split 1, Split 1) ]
               "a traced result lies where eager's does, and a refusal is \
                eager's message"
               placements_case law_forms;
             prop
               "a traced product of rows by groups lies where eager's does, \
                and a refusal is eager's message"
               groups_case law_by_groups;
           ];
         group "interpret"
           [
             test
               "by is the function the user called, whatever it computes \
                through"
               test_by_called;
             test
               "by names a plain operation after a call resumed on another \
                domain"
               test_names_after_hop;
             test "by names each systhread's own call" test_names_two_threads;
             test "a rule's refusal reaches the caller as raised"
               test_refusal_named;
             test "a rule's operations are named by the functions it calls"
               (fun () ->
                 test_rule_calls_named ();
                 test_rule_donates ());
             test "what the function and the rule raise keep their backtrace"
               test_raised_backtrace;
             cases
               "an operation of each kind a rule evaluates is received as \
                itself and gives eager's bits"
               ~id:(fun (Every (name, _)) -> name)
               (every ()) forwards_as_eager;
             prop ~count:50
               "a derivative over a Values interpretation agrees with central \
                differences"
               (Gen.pair
                  (Gen.array ~size:(Gen.int_range 0 6)
                     (Gen.float_range (-3.) 3.))
                  (Gen.array ~size:(Gen.int_range 0 6)
                     (Gen.float_range (-1.) 1.)))
               law_jvp;
           ];
         group "traced"
           [
             test
               "after its interpretation returned or raised, a traced value \
                refuses operations and its form, payload and owner answer"
               test_after_return;
           ];
         twins;
         group "payload and owner"
           [
             test "owner and payload allocate nothing" test_queries_words;
             test "a traced copy's dispatch allocates at most its budget"
               test_dispatch_words;
           ];
       ])
