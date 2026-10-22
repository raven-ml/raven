(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Interpretations through Nx.Prim: the dispatch rule and its refusals, fibers
   and domains, the forms results gives, and interpreters shaped like nx's
   consumers: a forward derivative, a recording observer and forwarding ones, a
   staging compiler and one that runs its program, a capture of expressions,
   and the staging compiler handing a call to the derivative. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module M = Nx_array.Move
module P = Nx_kernel.Prog
module C = Nx_support.Counting
module Count = (val Nx.devices ~kernels:(module C) [ Nx_support.memory 0 ])

let invalid ~sub f = raises_match (Exn.invalid_arg ~substring:sub) f
let f32 = Nx.float32
let f64 = Nx.float64
let host_array x = Option.get (Nx.Repr.array (Nx.place Nx.Host.on x))
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

(* A rule's own operations on concrete parts compute; on its own values they
   raise naming it, outside its extent, also under an extent around it. *)
let test_running () =
  let x = x3 () in
  let log = ref [] in
  let computing _ ~by op =
    log := Nx.Prim.name op :: !log;
    Nx.Prim.eval ~by op
  in
  let y =
    Nx.Prim.interpret ~name:"test.extent" Extent computing (fun _ -> Nx.add x x)
  in
  equal ~msg:"delivered once" names [ "Map" ] (received log);
  equal ~msg:"computed by the rule" (array float_exact) [| 2.; 4.; 6. |]
    (elements y);
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

(* A rule reads an operation as it was called: the donor beside its own
   handle, as in [add (donate x) x], is read as eager execution reads it, and
   dies once the operation is received. *)
let test_donor_beside () =
  let x = x3 () in
  let y =
    Nx.Prim.interpret ~name:"test.extent" Extent
      (fun _ ~by op -> Nx.Prim.eval ~by op)
      (fun _ -> Nx.add (Nx.donate x) x)
  in
  equal (array float_exact) [| 2.; 4.; 6. |] (elements y);
  invalid ~sub:"Nx.copy: operand 1 was donated to Nx.add" (fun () -> Nx.copy x)

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
      (fun _ -> ignore (Nx.lstsq a b))
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
  let by_reshape = function
    | Invalid_argument m -> String.starts_with ~prefix:"Nx.reshape: " m
    | _ -> false
  in
  raises_match by_reshape (fun () ->
      Nx.Prim.interpret ~name:"test.reshaping" Extent reshaping (fun _ ->
          Nx.lstsq a b))

(* What a rule donates dies naming the function the rule called, inside a
   composite too. *)
let test_rule_donates () =
  let own = vec f64 [| 1.; 2. |] and spent = ref false in
  Nx.Prim.interpret ~name:"test.donor" Extent
    (fun _ ~by op ->
      if not !spent then begin
        spent := true;
        ignore (Nx.neg (Nx.donate own))
      end;
      Nx.Prim.eval ~by op)
    (fun _ -> ignore (Nx.trace (square64 ())));
  invalid ~sub:"Nx.copy: operand 1 was donated to Nx.neg" (fun () ->
      Nx.copy own)

(* A rule inside [trace] performs an effect whose handler resumes the rest of
   the call on a second domain, as a work-stealing scheduler does. A plain
   operation on the first domain afterwards is named by its own function: its
   death, and the [by] a rule receives. *)
type _ Effect.t += Hop : unit Effect.t

let test_names_after_hop () =
  let hopping f =
    Effect.Deep.match_with f ()
      {
        retc = Fun.id;
        exnc = raise;
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
  equal ~msg:"hopped" bool true !hopped;
  let x = vec f64 [| 1.; 2. |] in
  equal ~msg:"a rule's by" names [ "Nx.neg" ]
    (observed_names (fun () -> Nx.neg (Nx.donate x)));
  invalid ~sub:"Nx.copy: operand 1 was donated to Nx.neg" (fun () -> Nx.copy x)

(* One systhread holds a composite, blocked in its rule, while another of the
   same domain runs a plain operation: each operation's death and the [by] its
   rule receives name its own function. *)
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
  equal ~msg:"the composite's by" names [ "Nx.trace" ] !composite;
  invalid ~sub:"Nx.copy: operand 1 was donated to Nx.neg" (fun () -> Nx.copy x);
  invalid ~sub:"Nx.copy: operand 1 was donated to Nx.trace" (fun () ->
      Nx.copy a)

let test_after_return () =
  let x = x3 () in
  let i, t =
    logged ~name:"test.values" Values (ref []) (fun i -> (i, tag i x))
  in
  invalid ~sub:"Nx.add: a value of test.values, used after test.values returned"
    (fun () -> Nx.add t x);
  equal ~msg:"its shape answers" (array int) [| 3 |] (Nx.shape t);
  equal ~msg:"its owner answers" bool true
    (match Nx.Prim.owner t with Some o -> o == i | None -> false);
  equal ~msg:"its payload answers" bool true
    (match Nx.Prim.payload i t with Some (Tag 0) -> true | _ -> false);
  invalid ~sub:"Nx.Repr.array: a value traced by test.values has no bytes"
    (fun () -> Nx.Repr.array t);
  let expansion = ref ignore in
  Nx.Prim.interpret ~name:"test.extent" Extent
    (fun i ~by op ->
      (expansion :=
         fun () -> ignore (Nx.Prim.expand i ~by ~repeat:Nx_support.folding op));
      Nx.Prim.eval ~by op)
    (fun _ -> ignore (Nx.add x x));
  invalid ~sub:"Nx.add: an expansion for test.extent after it returned"
    !expansion

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
      | Some m ->
          contains ~sub:"Nx.add: a value of test.values, used outside" m
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
      | Some m -> contains ~sub:"Nx.add: a value of test.values, used outside" m
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

(* [quiet] is false while an extent lives, seen from any domain; a values
   interpretation leaves it as it was. It may be false with no extent live,
   after a continuation dropped in one, so the test asserts nothing of it
   outside. *)
let test_quiet () =
  let outside = Nx.Prim.quiet () in
  logged ~name:"test.values" Values (ref []) (fun _ ->
      equal ~msg:"inside values" bool outside (Nx.Prim.quiet ()));
  logged ~name:"test.extent" Extent (ref []) (fun _ ->
      equal ~msg:"inside an extent" bool false (Nx.Prim.quiet ());
      equal ~msg:"on another domain" bool false
        (Domain.join (Domain.spawn Nx.Prim.quiet)))

(* A formula reaches a Values rule as it is, computing nothing. *)
let test_constants () =
  let c = Nx.zeros f32 [| 3 |] in
  let x = on_count f32 [| 1.; 2.; 3. |] in
  let placed = ref [] in
  let looking i ~by op =
    let (Nx.Prim.Operands xs) = Nx.Prim.operands op in
    List.iter
      (fun (Nx.Prim.Any y) ->
        if Nx.Prim.owner y = None then
          placed := Option.is_none (Nx.placement y) :: !placed)
      xs;
    tagged i ~by op
  in
  C.reset ();
  Nx.Prim.interpret ~name:"test.values" Values looking (fun i ->
      ignore (Nx.add (tag i x) c));
  equal ~msg:"without a placement" (list bool) [ true ] !placed;
  equal ~msg:"kernel calls" int 0 (C.calls ())

(* The name of an operation, a creation told from a map. *)
let kind (type r) (op : r Nx.Prim.t) =
  match op with Map { loads = [||]; _ } -> "Creation" | _ -> Nx.Prim.name op

(* A formula made outside an Extent reaches its rule as its operations, once
   per operand, and the operation reads the rule's results. *)
let owned_by i y =
  match Nx.Prim.owner y with Some o -> o == i | None -> false

let test_captured () =
  let c = Nx.zeros f32 [| 2 |] in
  let log = ref [] and owned = ref [] in
  let looking i ~by op =
    log := kind op :: !log;
    let (Nx.Prim.Operands xs) = Nx.Prim.operands op in
    List.iter (fun (Nx.Prim.Any y) -> owned := owned_by i y :: !owned) xs;
    tagged i ~by op
  in
  Nx.Prim.interpret ~name:"test.extent" Extent looking (fun _ ->
      ignore (Nx.add c c));
  equal ~msg:"received" names
    [ "Creation"; "Creation"; "Map" ]
    (received log);
  equal ~msg:"the map reads the rule's results" (list bool) [ true; true ]
    !owned

let rule =
  group "rule"
    [
      test "values reach the operations on their values alone" test_values_reach;
      test "an extent reaches every operation inside it, creations included"
        test_extent_reach;
      test "the innermost that reaches an operation receives it" test_innermost;
      test "a rule computes, and refuses its own values" test_running;
      test "a rule reads the donor beside its handle" test_donor_beside;
      test "a rule receives the function the user called" test_by_called;
      test "a rule's refusal keeps its name inside a composite"
        test_refusal_named;
      test "a rule's calls keep their names inside a composite"
        test_rule_calls_named;
      test "what a rule donates dies naming the rule's call" test_rule_donates;
      test "a plain operation after a call resumed on another domain"
        test_names_after_hop;
      test "a plain operation beside another systhread's composite"
        test_names_two_threads;
      test "a value used after its interpretation returned raises"
        test_after_return;
      test "a value refuses another domain; an extent does not reach one"
        test_domains;
      test "a value refuses another systhread; an extent does not reach one"
        test_systhreads;
      test "quiet is false while an extent lives" test_quiet;
      test "a formula reaches a values rule as it is" test_constants;
      test "a captured formula reaches an extent as its operations"
        test_captured;
    ]

(* Forms: results gives the form eager execution gives (Law 3) *)

type case = Case : string * 'r Nx.Prim.t -> case
type kind = Concrete | View | Constant

let pp_kind ppf k =
  Format.pp_print_string ppf
    (match k with
    | Concrete -> "concrete"
    | View -> "view"
    | Constant -> "constant")

let pp_shape ppf s =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))

(* A float32 operand of [shape] on Count's set: an array, a transposed view of
   one, or a constant. *)
let operand kind shape : (float, D.float32_elt, Count.d) Nx.t =
  let n = Array.fold_left ( * ) 1 shape in
  let data = Array.init n (fun i -> Float.of_int i -. 1.5) in
  match kind with
  | Constant -> Nx.zeros f32 shape
  | Concrete ->
      Nx.place Count.on
        (Nx.Repr.of_array Nx.Host.v (A.of_array D.Float32 shape data))
  | View ->
      let r = Array.length shape in
      let rev = Array.init r (fun i -> r - 1 - i) in
      let t = Array.map (fun a -> shape.(a)) rev in
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
            layout = L.contiguous shape;
            prog = add_prog;
            outs = Nx.Prim.[ f32 ];
            loads = [| Plain x; Plain y |];
          } );
    Case
      ( "chain",
        Map
          {
            layout = L.contiguous shape;
            prog = chain_prog;
            outs = Nx.Prim.[ f32; f32 ];
            loads = [| Plain x; Plain y |];
          } );
  ]

let shapes =
  Gen.of_list ~pp:pp_shape
    [ [||]; [| 0 |]; [| 1 |]; [| 3 |]; [| 2; 3 |]; [| 0; 2 |]; [| 2; 1; 3 |] ]

let kinds = Gen.of_list ~pp:pp_kind [ Concrete; View; Constant ]

let forms_case =
  Gen.with_pp
    (fun ppf (shape, a, b, _) ->
      Format.fprintf ppf "%a %a %a" pp_shape shape pp_kind a pp_kind b)
    (Gen.map
       (fun ((shape, a), (b, k)) -> (shape, a, b, k))
       (Gen.pair (Gen.pair shapes kinds) (Gen.pair kinds (Gen.int_range 0 9))))

let same_form ~msg (x : ('v, 's, 'd) Nx.t) (y : ('v, 's, 'd) Nx.t) =
  let f = Nx.Prim.form x and g = Nx.Prim.form y in
  equal ~msg:(msg ^ ": dtype") string (D.name f.dtype) (D.name g.dtype);
  equal ~msg:(msg ^ ": layout")
    (Testable.make ~pp:L.pp ~equal:L.equal)
    f.layout g.layout;
  equal ~msg:(msg ^ ": placement") bool true
    (match (f.placement, g.placement) with
    | Some p, Some q -> Nx.Placement.equal p q
    | None, None -> true
    | Some _, None | None, Some _ -> false)

let rec same_outs : type d r. string -> (d, r) Nx.Prim.outs -> r -> r -> unit =
 fun msg outs a b ->
  match (outs, a, b) with
  | [], (), () -> ()
  | _ :: outs, (x, a), (y, b) ->
      same_form ~msg x y;
      same_outs msg outs a b

let same_reduction : type d a.
    string -> (d, a) Nx.Prim.reduction -> a -> a -> unit =
 fun msg r a b ->
  match (r, a, b) with
  | Monoid _, a, b -> same_form ~msg a b
  | Logsumexp _, a, b -> same_form ~msg a b
  | Moments _, (a, a'), (b, b') ->
      same_form ~msg a b;
      same_form ~msg a' b'
  | Arg _, (a, a'), (b, b') ->
      same_form ~msg a b;
      same_form ~msg a' b'

let rec same_reductions : type d r.
    string -> (d, r) Nx.Prim.reductions -> r -> r -> unit =
 fun msg rs a b ->
  match (rs, a, b) with
  | [], (), () -> ()
  | r :: rs, (x, a), (y, b) ->
      same_reduction msg r x y;
      same_reductions msg rs a b

let same_results : type r. string -> r Nx.Prim.t -> r -> r -> unit =
 fun msg op a b ->
  match op with
  | Map { outs; _ } -> same_outs msg outs a b
  | Reduce { reductions; _ } -> same_reductions msg reductions a b
  | Scan { reduction; _ } -> same_reduction msg reduction a b
  | Gather _ -> same_form ~msg a b
  | Scatter _ -> same_form ~msg a b
  | Sort _ ->
      let (v, p), (v', p') = (a, b) in
      same_form ~msg v v';
      same_form ~msg p p'
  | Fft (C2c _) -> same_form ~msg a b
  | Fft (R2c _) -> same_form ~msg a b
  | Fft (C2r _) -> same_form ~msg a b
  | Linalg _ -> fail "law_forms builds no factorisation"
  | Fold _ -> same_form ~msg a b
  | Assemble _ -> same_form ~msg a b
  | Copy _ -> same_form ~msg a b
  | Contract _ -> same_form ~msg a b
  | Move _ -> same_form ~msg a b
  | Bitcast _ -> same_form ~msg a b
  | Place _ -> same_form ~msg a b
  | Check _ -> ()

let law_forms (shape, a, b, k) =
  cover "constant" (a = Constant || b = Constant);
  cover "empty" (Array.mem 0 shape);
  cover "0-d" (shape = [||]);
  let (Case (name, op)) = List.nth (cases_of shape a b) k in
  let eager = Nx.Prim.eval ~by:"test" op in
  logged ~name:"test.forms" Values (ref []) (fun i ->
      let traced = tagged i ~by:"test" op in
      same_results name op traced eager)

(* Ill-formed operations raise naming [by], eagerly and under either reach,
   before any kernel (Laws 7 and 13). *)
let ill_formed () =
  let x = operand Concrete [| 3 |] and y = operand Concrete [| 2 |] in
  [
    Case
      ( "loads of another shape",
        Map
          {
            layout = L.contiguous [| 3 |];
            prog = add_prog;
            outs = Nx.Prim.[ f32 ];
            loads = [| Plain x; Plain y |];
          } );
    Case
      ( "a load of another dtype",
        Map
          {
            layout = L.contiguous [| 3 |];
            prog = add_prog;
            outs = Nx.Prim.[ f32 ];
            loads = [| Plain x; Plain (Nx.cast Nx.int32 x) |];
          } );
    Case ("a permutation of another rank", Move (Permute [| 1; 0 |], x));
    Case ("a wider bitcast of an odd axis", Bitcast (Nx.float64, x));
    Case
      ( "data of another shape",
        Check
          {
            ok = Nx.less x x;
            data = [ y ];
            fail = (fun _ _ -> Failure "fails");
          } );
  ]

let test_ill_formed () =
  List.iter
    (fun (Case (name, op)) ->
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
      equal ~msg:(name ^ ": kernel calls") int 0 (C.calls ()))
    (ill_formed ())

(* Expansions keep bits. A selection and a constant move bits, so an
   interpretation that expands every operation it can gives each code of a
   narrow dtype back unchanged: its infinities, NaNs, [-0], largest finite
   value and least subnormal among them. *)

let expanding f =
  Nx.Prim.interpret ~name:"test.expand" Extent
    (fun i ~by op ->
      match Nx.Prim.expand i ~by ~repeat:Nx_support.folding op with
      | Some r -> r
      | None -> Nx.Prim.eval ~by op)
    (fun _ -> f ())

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
  let msg s = D.name dt ^ ", expanded: " ^ s in
  let same msg a b = equal ~msg (list string) [] (changed a b) in
  let check msg want (f : 'd. 'd Nx.Placement.t -> int array) =
    same msg want (expanding (fun () -> f Nx.Host.on))
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
                 })
          in
          raw (Nx.place on y)))
    consts

let test_keeps_bits () =
  List.iter (fun (D.Any dt, consts) -> keeps_bits dt consts) narrow

let laws =
  group "laws"
    [
      prop "results gives the forms eager execution gives" forms_case law_forms;
      test "an expanded selection or constant keeps each narrow code's bits"
        test_keeps_bits;
      test
        "an ill-formed operation raises naming its function, before any kernel"
        test_ill_formed;
    ]

(* (a) A forward derivative: a Values interpretation whose values hold a primal
   and a tangent. A map of several nodes it expands into one map per node. *)

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
let expect (type v s d) (dt : (v, s) D.t) (Nx.Prim.Any x : d Nx.Prim.any) :
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
  | Op2 (Binary Add, 0, 1) -> Nx.add (expect dt (t 0)) (expect dt (t 1))
  | Op2 (Binary Mul, 0, 1) ->
      Nx.add
        (Nx.mul (expect dt (t 0)) (expect dt (p 1)))
        (Nx.mul (expect dt (p 0)) (expect dt (t 1)))
  | Op1 (Cast, _, 0) ->
      let (Nx.Prim.Any tx) = t 0 in
      Nx.cast dt tx
  | Op1 (Copy, _, 0) -> Nx.copy (expect dt (t 0))
  | Op3 (Where, 0, 1, 2) ->
      Nx.where (expect Nx.bool (p 0)) (expect dt (t 1)) (expect dt (t 2))
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
      match
        (one_node prog, Nx.Prim.expand i ~by ~repeat:Nx_support.folding op)
      with
      | _, Some r -> r
      | Some node, None ->
          let y, () = Nx.Prim.eval ~by (primal (Map m)) in
          (dual i y (tangent i node loads y), ())
      | None, None -> fail "test.jvp: a map that does not expand")
  | Map _ -> (
      match Nx.Prim.expand i ~by ~repeat:Nx_support.folding op with
      | Some r -> r
      | None -> fail "test.jvp: a map that does not expand")
  | Reduce _ | Scan _ ->
      invalid_arg (by ^ ": test.jvp has no derivative of a reduction")
  | Gather g ->
      (* Linear in [x]: the positions carry no tangent. *)
      let idx = fst (parts i g.idx) in
      both (fun y -> Nx.Prim.eval ~by (Gather { g with idx; x = y })) g.x
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
  Array.iteri
    (fun k t ->
      let fd = (plus.(k) -. minus.(k)) /. (2. *. h) in
      let tol = 1e-4 *. (1. +. Float.abs fd) in
      less
        ~msg:(Printf.sprintf "tangent %d: %h against %h" k t fd)
        float_exact ~than:tol
        (Float.abs (t -. fd)))
    (elements t)

(* (b) A recording observer: an Extent interpretation that records each
   operation and computes it as eager execution would. *)
type recorded = Recorded : 'r Nx.Prim.t * 'r -> recorded

let recording log _ ~by op =
  let r = Nx.Prim.eval ~by op in
  log := Recorded (op, r) :: !log;
  r

let step k x = Nx.(add (mul x x) (scalar float32 (Float.of_int k)))

let test_recording () =
  let x = vec f32 [| 0.5; -1.; 2. |] in
  let trips =
    List.init 3 (fun k ->
        let log = ref [] in
        let y =
          Nx.Prim.interpret ~name:"test.record" Extent
            (fun i ~by op -> recording log i ~by op)
            (fun _ -> step k x)
        in
        equal
          ~msg:(Printf.sprintf "trip %d as eager" k)
          (array float_exact)
          (elements (step k x))
          (elements y);
        List.rev !log)
  in
  let names =
    List.map (List.map (fun (Recorded (op, _)) -> Nx.Prim.name op)) trips
  in
  equal ~msg:"each trip's operations"
    (list (list string))
    (List.init 3 (fun _ -> [ "Map"; "Map"; "Move"; "Map" ]))
    names;
  (* Replaying trip 2's operations gives its results again. *)
  List.iter
    (fun (Recorded (op, r)) ->
      match op with
      | Map { outs = [ _ ]; _ } ->
          let y, () = Nx.Prim.eval ~by:"test.replay" op in
          let y0, () = r in
          let bits x =
            Array.map Int64.bits_of_float (elements (Nx.cast f64 x))
          in
          equal ~msg:"replayed" (array int64) (bits y0) (bits y)
      | _ -> ())
    (List.nth trips 2)

(* (c) A staging compiler: an Extent interpretation whose values are nodes of a
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
    (Iarray.to_list (Option.get (Nx.Repr.shards x)))

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

(* A formula made outside the extent reaches the compiler as its operations:
   the program computes it, and nothing computes now. *)
let test_staging () =
  let w = on_count f32 [| 1.; 2.; 3.; 4. |] in
  let c = Nx.zeros f32 [| 4 |] in
  let x = on_count f32 [| 0.; 1.; 0.; 1. |] in
  C.reset ();
  let outs, prog =
    stage
      (fun ts ->
        let t = List.hd ts in
        let y = Nx.add (Nx.mul t w) c in
        [ y; Nx.add (Nx.zeros f32 [| 4 |]) t ])
      [ x ]
  in
  equal ~msg:"kernel calls" int 0 (C.calls ());
  equal ~msg:"held bytes: the weight" int 16 prog.held;
  equal ~msg:"operations" names
    [ "Map"; "Creation"; "Map"; "Creation"; "Map" ]
    (List.rev prog.ops);
  equal ~msg:"results are the program's" bool true
    (List.for_all (fun y -> Nx.Prim.owner y <> None) outs)

(* On a set whose kernels decline [Fill], a captured formula stages as on any
   other. *)
module Dec =
  (val Nx.devices
         ~kernels:(module Nx_support.Declining)
         [ Nx_support.memory 1 ])

let test_staging_declining () =
  let c = Nx.zeros f32 [| 2 |] in
  let x = Nx.place Dec.on (vec f32 [| 1.; 2. |]) in
  let outs, prog = stage (fun ts -> [ Nx.mul (List.hd ts) c ]) [ x ] in
  equal ~msg:"operations" names [ "Creation"; "Map" ] (List.rev prog.ops);
  equal ~msg:"held bytes" int 0 prog.held;
  equal ~msg:"results are the program's" bool true
    (List.for_all (fun y -> Nx.Prim.owner y <> None) outs)

(* A forwarding observer gets formulas back from the operations it forwards:
   each operation of a chain reaches it once, under another observer too. A
   captured start reaches each once per operand that reads it. *)
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
    (n + 1, n + 1)
    (nested (fun () -> c))

(* A captured formula of every operation a formula node can hold reaches a
   forwarding observer as that operation, and computes as eager. *)
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
    Every ("Move", fun () -> Nx.transpose (m ()));
    Every ("Copy", fun () -> Nx.copy (Nx.transpose (m ())));
  ]

(* The last two operations a log received. *)
let last_two log =
  match !log with b :: a :: _ -> [ a; b ] | _ -> received log

(* CR: Check captured result positions through the existing Tag k payload.
   These cases miss multi-output Map, paired reductions/scans and Linalg
   pairs/triples; swapping same-dtype results can pass. Read each selected
   result through Copy under tagged and check its operand's expected Tag k.
   Keep the numerical forwarding checks; this also covers Linalg without
   requiring an eager kernel. *)
let test_every () =
  List.iter
    (fun (Every (name, made)) ->
      let c = made () in
      let log = ref [] in
      let y = forwarding "test.forward" log (fun _ -> Nx.copy c) in
      let printed x = Nx.to_string (Nx.place Nx.Host.on x) in
      equal ~msg:(name ^ ": as eager") string (printed (Nx.copy c)) (printed y);
      equal ~msg:(name ^ ": received") names [ name; "Copy" ] (last_two log))
    (every ());
  (* No kernel of the host computes a factorisation yet: a rule that computes
     nothing receives one. *)
  let c =
    let a = Nx.add (Nx.eye f32 3) (Nx.eye f32 3) in
    Nx.Prim.eval ~by:"test" (Linalg (Cholesky { triangle = Lower; a }))
  in
  let log = ref [] in
  logged ~name:"test.extent" Extent log (fun _ -> ignore (Nx.copy c));
  equal ~msg:"Linalg: received" names [ "Linalg"; "Copy" ] (last_two log)

(* A compiler that stages each operation and runs the program after its extent
   returns: a staged function over captured formulas equals the function
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

(* A captured formula: one creation, two, or a creation and a broadcast. *)
type captured = Full of float | Sum of float * float | Scaled of float * float

let made c =
  let full a = Nx.full f32 [| 3 |] a in
  match c with
  | Full a -> full a
  | Sum (a, b) -> Nx.add (full a) (full b)
  | Scaled (a, b) -> Nx.mul (full a) (Nx.scalar f32 b)

let captured =
  let v = Gen.float_range (-2.) 2. in
  Gen.map
    (fun (k, a, b) ->
      match k with 0 -> Full a | 1 -> Sum (a, b) | _ -> Scaled (a, b))
    (Gen.triple (Gen.int_range 0 2) v v)

(* Steps of the staged function: an operation of the running value and one of
   the captured formulas. *)
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

(* (d) A capture of expressions, whose literals come from creations. A map of
   several nodes is expanded into one map per node, which reach it again. *)

type expr =
  | Lit of string * int array
  | Col of int array
  | App of string * expr list

let rec pp_expr ppf = function
  | Lit (bits, s) -> Format.fprintf ppf "lit %S %a" bits pp_shape s
  | Col s -> Format.fprintf ppf "col %a" pp_shape s
  | App (f, args) ->
      Format.fprintf ppf "(%s %a)" f
        (Format.pp_print_list ~pp_sep:Format.pp_print_space pp_expr)
        args

let exprs = Testable.make ~pp:pp_expr ~equal:( = )

type ('v, 's, 'd) Nx.Prim.payload += Expr : expr -> ('v, 's, 'd) Nx.Prim.payload

let capturing (type r) i ~by (op : r Nx.Prim.t) : r =
  match Nx.Prim.expand i ~by ~repeat:Nx_support.folding op with
  | Some r -> r
  | None ->
      let (Nx.Prim.Operands xs) = Nx.Prim.operands op in
      let arg (Nx.Prim.Any x) =
        match Nx.Prim.payload i x with
        | Some (Expr e) -> e
        | _ -> Col (Nx.shape x)
      in
      let e =
        match op with
        | Map { loads = [||]; prog; layout; _ } -> (
            match Iarray.get prog.nodes (Iarray.length prog.nodes - 1) with
            | Const (_, bits) -> Lit (bits, L.shape layout)
            | _ -> App ("creation", []))
        | _ -> App (Nx.Prim.name op, List.map arg xs)
      in
      Nx.Prim.results ~by (fun _ form -> Nx.Prim.traced i form (Expr e)) op

let capture f =
  Nx.Prim.interpret ~name:"test.capture" Extent capturing (fun i ->
      List.map
        (fun y ->
          match Nx.Prim.payload i y with
          | Some (Expr e) -> e
          | _ -> fail "a result outside the capture")
        (f ()))

let test_capture () =
  let x = vec f32 [| 1.; 2.; 3. |] in
  let two = P.bits D.Float32 2. in
  equal ~msg:"a literal and a column" (list exprs)
    [ App ("Map", [ App ("Move", [ Lit (two, [||]) ]); Col [| 3 |] ]) ]
    (capture (fun () -> [ Nx.add (Nx.scalar f32 2.) x ]));
  let col = Col [| 3 |] in
  let sum = App ("Map", [ col; col ]) in
  equal ~msg:"a chain, node by node" (list exprs)
    [ sum; App ("Map", [ sum; col ]) ]
    (capture (fun () ->
         let y, (z, ()) =
           Nx.Prim.eval ~by:"test"
             (Map
                {
                  layout = L.contiguous [| 3 |];
                  prog = chain_prog;
                  outs = Nx.Prim.[ f32; f32 ];
                  loads = [| Plain x; Plain x |];
                })
         in
         [ y; z ]))

(* (e) The staging compiler called with arguments of a live derivative hands the
   call to it, which stages the derivative: the program takes each argument's
   primal and tangent. Without the hand-off, staging refuses. *)

let staged_call f args =
  let owned =
    List.find_map
      (fun a ->
        match Nx.Prim.owner a with
        | Some j -> (
            match Nx.Prim.payload j a with Some (Dual _) -> Some j | _ -> None)
        | None -> None)
      args
  in
  match owned with
  | None -> stage f args
  | Some j ->
      let n = List.length args in
      let primals = List.map (fun a -> fst (parts j a)) args in
      let tangents = List.map (fun a -> snd (parts j a)) args in
      stage
        (fun xs ->
          let ps = List.filteri (fun k _ -> k < n) xs in
          let ts = List.filteri (fun k _ -> k >= n) xs in
          Nx.Prim.interpret ~name:"test.jvp" Values jvp_rule (fun i ->
              let ys = f (List.map2 (dual i) ps ts) in
              List.map (fun y -> fst (parts i y)) ys
              @ List.map (fun y -> snd (parts i y)) ys))
        (primals @ tangents)

let test_hand_off () =
  let x = vec f32 [| 1.; 2. |] and v = vec f32 [| 1.; 0. |] in
  let square ts = List.map (fun t -> Nx.mul t t) ts in
  let (outs, prog), refused =
    Nx.Prim.interpret ~name:"test.jvp" Values jvp_rule (fun j ->
        let d = dual j x v in
        let handed = staged_call square [ d ] in
        let refused =
          match stage (fun _ -> square [ d ]) [ x ] with
          | _ -> None
          | exception Invalid_argument m -> Some m
        in
        (handed, refused))
  in
  equal ~msg:"params: a primal and a tangent" int 2 prog.params;
  equal ~msg:"results: a primal and a tangent" int 2 (List.length outs);
  (* x * x, then its tangent t * x + x * t. *)
  equal ~msg:"the derivative, staged" names
    [ "Map"; "Map"; "Map"; "Map" ]
    (List.rev prog.ops);
  equal ~msg:"without the hand-off" (option string)
    (Some "Nx.mul: test.stage meets a value of another live interpretation")
    refused

(* A cast's tangent is the tangent cast. *)
let test_jvp_cast () =
  let x = vec f64 [| 1.5; -2. |] and v = vec f64 [| 0.25; 3. |] in
  let y, t = jvp (fun x -> Nx.cast f32 x) x v in
  equal ~msg:"primal" (array float_exact) [| 1.5; -2. |] (elements y);
  equal ~msg:"tangent" (array float_exact) [| 0.25; 3. |] (elements t)

let interpreters =
  group "interpreters"
    [
      test "a forward derivative casts its tangent" test_jvp_cast;
      prop ~count:50 "a forward derivative agrees with finite differences"
        (Gen.pair
           (Gen.array ~size:(Gen.int_range 0 6) (Gen.float_range (-3.) 3.))
           (Gen.array ~size:(Gen.int_range 0 6) (Gen.float_range (-1.) 1.)))
        law_jvp;
      test "a recording observer computes as eager, and its trips replay"
        test_recording;
      test "a staging compiler computes nothing and stages captured formulas"
        test_staging;
      test "staging a captured formula needs no kernel of the set"
        test_staging_declining;
      test "a forwarding observer receives each operation once"
        test_forwarding;
      test "a captured formula of every operation forwards as eager"
        test_every;
      prop ~count:100 "a staged function over captured formulas equals eager"
        staged_case law_staged;
      test "a capture takes literals from creations" test_capture;
      test "staging hands a call on a derivative's values to it" test_hand_off;
    ]

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
        exnc = raise;
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
      exnc = raise;
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
  contains
    ~sub:"Nx.add: a value of test.values, used outside test.values's extent"
    !used

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
  let finalised = ref false in
  let got =
    Nx.Prim.interpret ~name:"test.values" Values
      (fun _ ~by:_ _ -> failwith "the rule")
      (fun i ->
        match
          Fun.protect
            ~finally:(fun () -> finalised := true)
            (fun () -> Nx.add (tag i x) x)
        with
        | _ -> "returned"
        | exception Failure m -> m)
  in
  equal ~msg:"caught inside" string "the rule" got;
  equal ~msg:"finalised" bool true !finalised

let fibers =
  group "fibers"
    [
      test "a fiber beside an extent computes" test_two_fibers;
      test "a continuation dropped inside an extent changes no result"
        test_dropped;
      test "an extent resumed on another domain still receives" test_migrated;
      test "a fiber an outer scheduler runs refuses a traced value"
        test_outer_fiber;
      test "a suspended rule holds its extent and enters once"
        test_suspended_rule;
      test "a rule's exception raises inside the extent" test_rule_raises;
    ]

(* Nesting: an interpretation that adopts another's values *)

let adopting ~name p log f = logged ~name (Adopting p) log f

(* A rule's answer to an operation, or the message of its refusal. *)
let answer f =
  match f () with
  | _ -> "computed"
  | exception Invalid_argument m -> m

(* An adopting interpretation receives the operations on its parent's values
   as on its own, before the parent, which receives none of them. *)
let test_adopting_reach () =
  let x = x3 () in
  let parent = ref [] and adopted = ref [] in
  logged ~name:"test.p" Values parent (fun p ->
      let t = tag p x in
      adopting ~name:"test.c" p adopted (fun c ->
          ignore (Nx.copy t);
          ignore (Nx.add (tag c x) x);
          ignore (Nx.add x x)));
  equal ~msg:"the adopting one" names [ "Copy"; "Map" ] (received adopted);
  equal ~msg:"the parent" names [] (received parent)

(* Adoption follows the parents: the innermost receives its grandparent's
   values. A returned sibling's value is refused. *)
let test_adopting_chain () =
  let x = x3 () in
  let log = ref [] in
  logged ~name:"test.p" Values (ref []) (fun p ->
      let sibling =
        adopting ~name:"test.sibling" p (ref []) (fun s -> tag s x)
      in
      adopting ~name:"test.c1" p (ref []) (fun c1 ->
          let t1 = tag c1 x in
          adopting ~name:"test.c2" c1 log (fun _ ->
              ignore (Nx.copy (tag p x));
              ignore (Nx.copy t1);
              equal ~msg:"the sibling's value" string
                "Nx.copy: a value of test.sibling, used after test.sibling \
                 returned"
                (answer (fun () -> Nx.copy sibling)))));
  equal ~msg:"the innermost" names [ "Copy"; "Copy" ] (received log)

(* Inside an extent, an adopting interpretation receives an operation on its
   parent's value before the extent, and the operations its rule applies reach
   the extent. *)
let test_adopting_inside_extent () =
  let x = x3 () in
  let extent = ref [] and adopted = ref [] in
  logged ~name:"test.p" Values (ref []) (fun p ->
      let t = tag p x in
      logged ~name:"test.extent" Extent extent (fun _ ->
          Nx.Prim.interpret ~name:"test.c" (Adopting p)
            (fun c ~by op ->
              adopted := Nx.Prim.name op :: !adopted;
              ignore (Nx.neg x);
              tagged c ~by op)
            (fun _ -> ignore (Nx.copy t))));
  equal ~msg:"the adopting one" names [ "Copy" ] (received adopted);
  equal ~msg:"the extent, the rule's operation" names [ "Map" ]
    (received extent)

(* Adopting an interpretation that returned, an extent, or one that is not
   around the caller raises. *)
let test_adopting_refused () =
  let x = x3 () in
  let adopt p =
    answer (fun () ->
        Nx.Prim.interpret ~name:"test.c" (Adopting p) tagged ignore)
  in
  let p = logged ~name:"test.p" Values (ref []) Fun.id in
  equal ~msg:"returned" string "test.c: adopts test.p, which returned"
    (adopt p);
  logged ~name:"test.extent" Extent (ref []) (fun e ->
      equal ~msg:"an extent" string "test.c: adopts test.extent, an Extent"
        (adopt e));
  let outside = "test.c: adopts test.p outside test.p's extent" in
  logged ~name:"test.p" Values (ref []) (fun p ->
      equal ~msg:"another domain" string outside
        (Domain.join (Domain.spawn (fun () -> adopt p))));
  let in_rule = ref "" in
  Nx.Prim.interpret ~name:"test.p" Values
    (fun p ~by op ->
      in_rule := adopt p;
      tagged p ~by op)
    (fun p -> ignore (Nx.copy (tag p x)));
  equal ~msg:"its parent's rule" string outside !in_rule

(* A fiber beside an adopting interpretation's, under a scheduler inside the
   parent, reaches the parent; the adopting fiber's operations, before and
   after it yields, reach the adopting one. *)
let test_adopting_yield () =
  let x = x3 () in
  let parent = ref [] and adopted = ref [] in
  logged ~name:"test.p" Values parent (fun p ->
      let t = tag p x in
      schedule
        [
          ( "adopting",
            fun () ->
              adopting ~name:"test.c" p adopted (fun _ ->
                  ignore (Nx.copy t);
                  yield ();
                  ignore (Nx.copy t)) );
          ("beside", fun () -> ignore (Nx.add t t));
        ]);
  equal ~msg:"the adopting one" names [ "Copy"; "Copy" ] (received adopted);
  equal ~msg:"the parent" names [ "Map" ] (received parent)

(* [f ()] on a fiber that its first yield suspends, then resumed on another
   domain. *)
let migrated f =
  let suspended = ref None in
  Effect.Deep.match_with f ()
    {
      retc = Fun.id;
      exnc = raise;
      effc =
        (fun (type a) (e : a Effect.t) ->
          match e with
          | Yield ->
              Some
                (fun (k : (a, unit) Effect.Deep.continuation) ->
                  suspended := Some (fun () -> Effect.Deep.continue k ()))
          | _ -> None);
    };
  Domain.join (Domain.spawn (Option.get !suspended))

(* An adopting interpretation resumed on another domain without its parent
   refuses there its own values and its parent's. *)
let test_adopting_migrated () =
  let x = x3 () in
  let got = ref [] in
  logged ~name:"test.p" Values (ref []) (fun p ->
      migrated (fun () ->
          adopting ~name:"test.c" p (ref []) (fun c ->
              let own = tag c x in
              yield ();
              got :=
                [
                  answer (fun () -> Nx.copy (tag p x));
                  answer (fun () -> Nx.copy own);
                ])));
  equal ~msg:"refused there" names
    [
      "Nx.copy: a value of test.p, used outside test.p's extent";
      "Nx.copy: a value of test.c, used outside test.c's extent";
    ]
    !got

(* An adopting interpretation resumed on another domain with its parent, and
   an extent between them, still receives its own values and its parent's
   there, before the extent. *)
let test_adopting_migrated_whole () =
  let x = x3 () in
  let parent = ref [] and extent = ref [] and adopted = ref [] in
  migrated (fun () ->
      logged ~name:"test.p" Values parent (fun p ->
          logged ~name:"test.extent" Extent extent (fun _ ->
              adopting ~name:"test.c" p adopted (fun c ->
                  let own = tag c x in
                  yield ();
                  ignore (Nx.copy (tag p x));
                  ignore (Nx.copy own)))));
  equal ~msg:"the adopting one" names [ "Copy"; "Copy" ] (received adopted);
  equal ~msg:"the extent" names [] (received extent);
  equal ~msg:"the parent" names [] (received parent)

(* The words an operation allocates, over [n] runs after one uncounted. *)
let words_per f =
  let n = 1000 in
  f ();
  let before = Gc.minor_words () in
  for _ = 1 to n do
    f ()
  done;
  (Gc.minor_words () -. before) /. Float.of_int n

(* An operation on a parent's value allocates what one on the adopting
   interpretation's own value does: the parent is around by construction. *)
let test_adopting_words () =
  let x = x3 () in
  logged ~name:"test.p" Values (ref []) (fun p ->
      let t = tag p x in
      Nx.Prim.interpret ~name:"test.c" (Adopting p) tagged (fun c ->
          let own = tag c x in
          let copy y () = ignore (Sys.opaque_identity (Nx.copy y)) in
          equal ~msg:"words per operation" float_exact
            (words_per (copy own))
            (words_per (copy t))))

let nesting =
  group "nesting"
    [
      test "an adopting interpretation receives its parent's values"
        test_adopting_reach;
      test "adoption follows the parents" test_adopting_chain;
      test "an adopting interpretation is reached before an extent"
        test_adopting_inside_extent;
      test "adopting a returned, an extent or an absent one raises"
        test_adopting_refused;
      test "a fiber beside an adopting one reaches the parent"
        test_adopting_yield;
      test "an adopting interpretation left by its parent refuses"
        test_adopting_migrated;
      test "an adopting interpretation migrated with its parent receives"
        test_adopting_migrated_whole;
      test "a parent's value costs what an own value does" test_adopting_words;
    ]

(* Loops: an expansion's loop runs through the interpretation's [repeat] *)

(* [repeat] as a fold, counting its trips. *)
let fold trips n step init =
  Nx_support.folding n
    (fun k carry ->
      incr trips;
      step k carry)
    init

let pp_ints ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int a)))

let form_name (Nx.Prim.Any x) =
  let f = Nx.Prim.form x in
  Format.asprintf "%s%a" (D.name f.dtype) pp_ints (L.shape f.layout)

(* The forms [step] gives once traced at stand-ins of its index and of [init]:
   what a stager that holds [step] once sees. *)
let traced_once step init =
  Nx.Prim.interpret ~name:"test.stager" Values tagged (fun s ->
      let stand_in (Nx.Prim.Any x) =
        Nx.Prim.Any (Nx.Prim.traced s (Nx.Prim.form x) (Tag 0))
      in
      let k = Nx.Prim.traced s (Nx.Prim.form (Nx.scalar Nx.int64 0L)) (Tag 0) in
      List.map form_name (step k (List.map stand_in init)))

(* [f ()] with each scatter whose targets may repeat expanded through
   [repeat]. *)
let looping
    (repeat :
      'd.
      int ->
      ((int64, D.int64_elt, 'd) Nx.t -> 'd Nx.Prim.any list ->
       'd Nx.Prim.any list) ->
      'd Nx.Prim.any list ->
      'd Nx.Prim.any list) f =
  Nx.Prim.interpret ~name:"test.loops" Extent
    (fun (type r) i ~by (op : r Nx.Prim.t) : r ->
      match op with
      | Scatter { unique = false; _ } -> (
          match Nx.Prim.expand i ~by ~repeat op with
          | Some r -> r
          | None -> fail "a repeating scatter that does not expand")
      | _ -> Nx.Prim.eval ~by op)
    (fun _ -> f ())

(* A scatter whose targets may repeat loops once per position along its
   axis, through [repeat], to the eager sum. *)
let test_repeat () =
  let into = vec f32 [| 0.; 0. |] in
  let idx = vec Nx.int64 [| 0L; 1L; 0L |] in
  let updates = vec f32 [| 1.; 2.; 3. |] in
  let loops = ref 0 and trips = ref 0 in
  let repeat n step init =
    incr loops;
    fold trips n step init
  in
  let y =
    looping repeat (fun () ->
        Nx.scatter ~combine:Nx.Add ~axis:0 idx updates into)
  in
  equal ~msg:"the sum" (array float_exact) [| 4.; 2. |] (elements y);
  equal ~msg:"loops" int 1 !loops;
  equal ~msg:"trips" int 3 !trips

(* A loop's step keeps its carry's forms, traced once at stand-ins, so that a
   stager may hold it once; folded, it gives the eager result. *)
let test_repeat_forms () =
  let into = Nx.reshape [| 2; 2 |] (vec f32 [| 0.; 0.; 0.; 0. |]) in
  let idx = Nx.reshape [| 2; 2 |] (vec Nx.int64 [| 0L; 1L; 1L; 1L |]) in
  let updates = Nx.reshape [| 2; 2 |] (vec f32 [| 1.; 2.; 3.; 4. |]) in
  let seen = ref [] and trips = ref 0 in
  let repeat n step init =
    seen := (List.map form_name init, traced_once step init) :: !seen;
    fold trips n step init
  in
  let y =
    looping repeat (fun () ->
        Nx.scatter ~combine:Nx.Max ~axis:1 idx updates into)
  in
  equal ~msg:"the result" (array float_exact) [| 1.; 2.; 0.; 4. |]
    (elements y);
  equal ~msg:"forms kept"
    (list (pair (list string) (list string)))
    [ ([ "float32[2; 2]" ], [ "float32[2; 2]" ]) ]
    !seen

let loops =
  group "loops"
    [
      test "a repeating scatter loops through repeat" test_repeat;
      test "a loop's step keeps its carry's forms" test_repeat_forms;
    ]

(* Memory

   While backtraces are recorded, the runtime keeps the last exception raised
   reachable. An operation keeps none of its operands' bytes once it returns,
   wherever it computed. *)

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
let[@inline never] moved () =
  let x = Nx.place Mem.on (Nx.zeros f32 [| 1024 |]) in
  ignore (Sys.opaque_identity (Nx.reshape [| 32; 32 |] x))

(* Raises and catches, so the last exception raised holds nothing. *)
let[@inline never] cleared () =
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

let memory =
  group "memory"
    [
      test "an operation beside another domain's extent keeps no bytes"
        (fun () -> equal int 0 (held_after (fun () -> beside moved)));
      test "an operation a rule applies keeps no bytes" (fun () ->
          let within () = evaluating "test.extent" (fun _ -> moved ()) in
          equal int 0 (held_after within));
    ]

(* The rule on two domains: each domain's operations go where the rule says,
   whatever the other domain's interpretations do (Law 1). The reference is the
   rule's statement: who receives each operation, and what eager execution
   gives. *)

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

let use t =
  match Nx.add t t with
  | _ -> "computed"
  | exception Invalid_argument m ->
      if String.length m > 0 then "refused" else "refused without a reason"

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
    command "use" (leaked ^-> returns string) (fun () -> "refused") use;
  ]

let domains =
  group "domains"
    [
      stateful ~domains:2 "each domain's operations follow the rule"
        rule_commands;
    ]

let () =
  exit
    (run "nx interp"
       [ rule; laws; fibers; nesting; loops; memory; interpreters; domains ])
