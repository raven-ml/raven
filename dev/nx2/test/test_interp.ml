(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Interpretations through Nx.Prim: the dispatch rule and its refusals, fibers
   and domains, the forms results gives, and five interpreters shaped like nx's
   consumers: a forward derivative, a recording observer, a staging compiler, a
   capture of expressions, and the staging compiler handing a call to the
   derivative. *)

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

(* An interpretation that logs each operation it receives and gives traced
   results tagged with their position. *)
type ('v, 's, 'd) Nx.Prim.payload += Tag : int -> ('v, 's, 'd) Nx.Prim.payload

let logging log =
  {
    Nx.Prim.rule =
      (fun i ~by op ->
        log := Nx.Prim.name op :: !log;
        Nx.Prim.results ~by
          { make = (fun k form -> Nx.Prim.traced i form (Tag k)) }
          op);
  }

(* A value [i] owns, of [x]'s form. *)
let tag i x = Nx.Prim.traced i (Nx.Prim.form x) (Tag 0)
let received log = List.rev !log
let names = list string

(* The rule *)

let x3 () = vec f32 [| 1.; 2.; 3. |]

let test_values_reach () =
  let log = ref [] in
  let x = x3 () in
  Nx.Prim.interpret ~name:"test.values" Values (logging log) (fun i ->
      ignore (Nx.add (tag i x) x);
      let y = Nx.add x x in
      equal ~msg:"computed" (array float_exact) [| 2.; 4.; 6. |] (elements y);
      ignore (Nx.zeros f32 [| 2 |]));
  equal ~msg:"received" names [ "Map" ] (received log)

let test_extent_reach () =
  let log = ref [] in
  let x = Nx.place Count.on (x3 ()) in
  C.reset ();
  Nx.Prim.interpret ~name:"test.extent" Extent (logging log) (fun _ ->
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
  Nx.Prim.interpret ~name:"outer" Values (logging outer) (fun i ->
      let t = tag i x in
      Nx.Prim.interpret ~name:"inner" Extent (logging inner) (fun _ ->
          ignore (Nx.add t x)));
  equal ~msg:"an extent started later" names [ "Map" ] (received inner);
  equal ~msg:"the values started earlier" names [] (received outer);
  let outer = ref [] and inner = ref [] in
  Nx.Prim.interpret ~name:"outer" Extent (logging outer) (fun _ ->
      Nx.Prim.interpret ~name:"inner" Values (logging inner) (fun i ->
          ignore (Nx.add (tag i x) x);
          ignore (Nx.add x x)));
  equal ~msg:"values started later" names [ "Map" ] (received inner);
  equal ~msg:"the extent, for a concrete operation" names [ "Map" ]
    (received outer)

(* A rule's own operations on concrete parts compute; on its own values they
   raise naming it. *)
let test_running () =
  let x = x3 () in
  let computing log =
    {
      Nx.Prim.rule =
        (fun _ ~by op ->
          log := Nx.Prim.name op :: !log;
          Nx.Prim.eval ~by op);
    }
  in
  let log = ref [] in
  let y =
    Nx.Prim.interpret ~name:"test.extent" Extent (computing log) (fun _ ->
        Nx.add x x)
  in
  equal ~msg:"delivered once" names [ "Map" ] (received log);
  equal ~msg:"computed by the rule" (array float_exact) [| 2.; 4.; 6. |]
    (elements y);
  let own =
    {
      Nx.Prim.rule =
        (fun i ~by op ->
          let (Nx.Prim.Operands xs) = Nx.Prim.operands op in
          let (Nx.Prim.Any x) = List.hd xs in
          ignore (Nx.copy (tag i x));
          Nx.Prim.eval ~by op);
    }
  in
  invalid ~sub:"Nx.copy: test.own's rule applied an operation to its own value"
    (fun () ->
      Nx.Prim.interpret ~name:"test.own" Values own (fun i ->
          ignore (Nx.add (tag i x) x)))

let test_after_return () =
  let x = x3 () in
  let i, t =
    Nx.Prim.interpret ~name:"test.values" Values
      (logging (ref []))
      (fun i -> (i, tag i x))
  in
  invalid ~sub:"Nx.add: a value of test.values, used after test.values returned"
    (fun () -> Nx.add t x);
  equal ~msg:"its shape answers" (array int) [| 3 |] (Nx.shape t);
  equal ~msg:"its owner answers" bool true
    (match Nx.Prim.owner t with Some o -> o == i | None -> false);
  equal ~msg:"its payload answers" bool true
    (match Nx.Prim.payload i t with Some (Tag 0) -> true | _ -> false);
  invalid ~sub:"Nx.Repr.array: a value traced by test.values has no bytes"
    (fun () -> Nx.Repr.array t)

let test_domains () =
  let x = x3 () in
  let log = ref [] in
  Nx.Prim.interpret ~name:"test.values" Values (logging log) (fun i ->
      let t = tag i x in
      let elsewhere () =
        match Nx.add t x with
        | _ -> None
        | exception Invalid_argument m -> Some m
      in
      match Domain.join (Domain.spawn elsewhere) with
      | Some m ->
          contains ~sub:"Nx.add: a value of test.values, used on another domain"
            m
      | None -> fail "a traced value computed on another domain");
  let log = ref [] in
  let y =
    Nx.Prim.interpret ~name:"test.extent" Extent (logging log) (fun _ ->
        Domain.join (Domain.spawn (fun () -> Nx.add x x)))
  in
  equal ~msg:"received" names [] (received log);
  equal ~msg:"another domain computes" (array float_exact) [| 2.; 4.; 6. |]
    (elements y)

let test_later () =
  let a, b =
    Nx.Prim.interpret ~name:"a" Values
      (logging (ref []))
      (fun a ->
        Nx.Prim.interpret ~name:"b" Values (logging (ref [])) (fun b -> (a, b)))
  in
  equal ~msg:"b later" bool true (Nx.Prim.later b a);
  equal ~msg:"a not later" bool false (Nx.Prim.later a b);
  let c =
    Domain.join
      (Domain.spawn (fun () ->
           Nx.Prim.interpret ~name:"c" Values (logging (ref [])) Fun.id))
  in
  invalid ~sub:"Nx.Prim.later: c and a started on two domains" (fun () ->
      Nx.Prim.later c a)

(* A constant reaches a rule computed where the operation reads it, once. *)
let test_constants () =
  let c = Nx.zeros f32 [| 3 |] in
  let x = on_count f32 [| 1.; 2.; 3. |] in
  let devices = ref [] in
  let array_of y = Option.get (Nx.Repr.array y) in
  let looking =
    {
      Nx.Prim.rule =
        (fun i ~by op ->
          let (Nx.Prim.Operands xs) = Nx.Prim.operands op in
          List.iter
            (fun (Nx.Prim.Any y) ->
              if Nx.Prim.owner y = None then
                devices := Rig.name (A.device (array_of y)) :: !devices)
            xs;
          Nx.Prim.results ~by
            { make = (fun k form -> Nx.Prim.traced i form (Tag k)) }
            op);
    }
  in
  C.reset ();
  Nx.Prim.interpret ~name:"test.values" Values looking (fun i ->
      let t = tag i x in
      ignore (Nx.add t c);
      ignore (Nx.add t c));
  equal ~msg:"computed on the operation's device" names [ "m0"; "m0" ] !devices;
  equal ~msg:"computed once" int 1 (C.calls ())

let rule =
  group "rule"
    [
      test "values reach the operations on their values alone" test_values_reach;
      test "an extent reaches every operation inside it, creations included"
        test_extent_reach;
      test "the innermost by start receives" test_innermost;
      test "a running rule computes, and refuses its own values" test_running;
      test "a value used after its interpretation returned raises"
        test_after_return;
      test "a value refuses another domain; an extent does not reach one"
        test_domains;
      test "later orders by start on one domain" test_later;
      test "a constant reaches a rule computed where it is read, once"
        test_constants;
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
      Nx.Prim.eval ~by:"test" (Move (Permute rev, base))

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
    Case ("reverse", Move (Permute (Array.init r (fun i -> r - 1 - i)), x));
    Case ("flatten", Move (Reshape [| numel |], x));
    Case ("broadcast", Move (Broadcast (Array.append [| 2 |] shape), x));
    Case ("int32", Bitcast (Nx.int32, x));
    Case ("bytes", Bitcast (Nx.uint8, x));
    Case ("host", Place (Nx.Host.on, x));
    Case ("count", Place (Count.on, x));
    Case
      ( "add",
        Map
          {
            shape;
            prog = add_prog;
            outs = Nx.Prim.[ f32 ];
            loads = [| Plain x; Plain y |];
          } );
    Case
      ( "chain",
        Map
          {
            shape;
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
    (Nx.Placement.equal f.placement g.placement)

let rec same_outs : type d r. string -> (d, r) Nx.Prim.outs -> r -> r -> unit =
 fun msg outs a b ->
  match (outs, a, b) with
  | [], (), () -> ()
  | _ :: outs, (x, a), (y, b) ->
      same_form ~msg x y;
      same_outs msg outs a b

let same_results : type r. string -> r Nx.Prim.t -> r -> r -> unit =
 fun msg op a b ->
  match op with
  | Map { outs; _ } -> same_outs msg outs a b
  | Copy _ -> same_form ~msg a b
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
  Nx.Prim.interpret ~name:"test.forms" Values
    (logging (ref []))
    (fun i ->
      let traced =
        Nx.Prim.results ~by:"test"
          { make = (fun k form -> Nx.Prim.traced i form (Tag k)) }
          op
      in
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
            shape = [| 3 |];
            prog = add_prog;
            outs = Nx.Prim.[ f32 ];
            loads = [| Plain x; Plain y |];
          } );
    Case
      ( "a load of another dtype",
        Map
          {
            shape = [| 3 |];
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
            data = [ Any y ];
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
              Nx.Prim.interpret ~name:"test" reach
                (logging (ref []))
                (fun i ->
                  let traced =
                    Nx.Prim.map
                      {
                        map =
                          (fun x ->
                            if reach = Nx.Prim.Values then tag i x else x);
                      }
                      op
                  in
                  Nx.Prim.eval ~by:"test.by" traced)))
        [ Nx.Prim.Values; Extent ];
      equal ~msg:(name ^ ": kernel calls") int 0 (C.calls ()))
    (ill_formed ())

let laws =
  group "laws"
    [
      prop "results gives the forms eager execution gives" forms_case law_forms;
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
let one_node prog =
  let k = Array.length (P.ins prog) in
  if P.length prog = k + 1 && P.outs prog = [| k |] then Some (P.node prog k)
  else None

(* The tangent of the one node [node] over [loads], whose primal result is
   [y]. *)
let tangent (type v s d) i node (loads : d Nx.Prim.load array)
    (y : (v, s, d) Nx.t) : (v, s, d) Nx.t =
  let p j =
    let (Nx.Prim.Plain x) = loads.(j) in
    Nx.Prim.Any (fst (parts i x))
  and t j =
    let (Nx.Prim.Plain x) = loads.(j) in
    Nx.Prim.Any (snd (parts i x))
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

let jvp_rule =
  {
    Nx.Prim.rule =
      (fun (type r) i ~by (op : r Nx.Prim.t) : r ->
        let both f x =
          let p, t = parts i x in
          dual i (f p) (f t)
        in
        let primal = { Nx.Prim.map = (fun x -> fst (parts i x)) } in
        match op with
        | Map ({ outs = [ _ ]; prog; loads; _ } as m) -> (
            match (one_node prog, Nx.Prim.expand i ~by op) with
            | _, Some r -> r
            | Some node, None ->
                let y, () = Nx.Prim.eval ~by (Nx.Prim.map primal (Map m)) in
                (dual i y (tangent i node loads y), ())
            | None, None -> fail "test.jvp: a map that does not expand")
        | Map _ -> (
            match Nx.Prim.expand i ~by op with
            | Some r -> r
            | None -> fail "test.jvp: a map that does not expand")
        | Copy x -> both (fun y -> Nx.Prim.eval ~by (Copy y)) x
        | Move (mv, x) -> both (fun y -> Nx.Prim.eval ~by (Move (mv, y))) x
        | Bitcast _ ->
            invalid_arg (by ^ ": test.jvp has no derivative of a bitcast")
        | Place (q, x) ->
            let p, t = parts i x in
            dual i (Nx.place q p) (Nx.place q t)
        | Check _ -> Nx.Prim.eval ~by (Nx.Prim.map primal op));
  }

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

let recording log =
  {
    Nx.Prim.rule =
      (fun _ ~by op ->
        let r = Nx.Prim.eval ~by op in
        log := Recorded (op, r) :: !log;
        r);
  }

let step k x = Nx.(add (mul x x) (scalar float32 (Float.of_int k)))

let test_recording () =
  let x = vec f32 [| 0.5; -1.; 2. |] in
  let trips =
    List.init 3 (fun k ->
        let log = ref [] in
        let y =
          Nx.Prim.interpret ~name:"test.record" Extent (recording log) (fun _ ->
              step k x)
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
    (Array.to_list (Option.get (Nx.Repr.shards x)))

let staging prog =
  {
    Nx.Prim.rule =
      (fun i ~by op ->
        let (Nx.Prim.Operands xs) = Nx.Prim.operands op in
        List.iter
          (fun (Nx.Prim.Any x) ->
            match Nx.Prim.owner x with
            | Some o when o == i -> ()
            | Some _ ->
                invalid_arg
                  (by
                 ^ ": test.stage meets a value of another live interpretation")
            | None -> prog.held <- prog.held + bytes x)
          xs;
        prog.ops <- Nx.Prim.name op :: prog.ops;
        Nx.Prim.results ~by
          {
            make =
              (fun _ form ->
                prog.next <- prog.next + 1;
                Nx.Prim.traced i form (Staged prog.next));
          }
          op);
  }

let stage f args =
  let prog = program () in
  let outs =
    Nx.Prim.interpret ~name:"test.stage" Extent (staging prog) (fun i ->
        f
          (List.map
             (fun a ->
               prog.params <- prog.params + 1;
               Nx.Prim.traced i (Nx.Prim.form a) (Staged 0))
             args))
  in
  (outs, prog)

let test_staging () =
  let w = on_count f32 [| 1.; 2.; 3.; 4. |] in
  let c = Nx.zeros f32 [| 4 |] in
  let x = on_count f32 [| 0.; 1.; 0.; 1. |] in
  C.reset ();
  let outs, prog =
    stage
      (fun ts ->
        let t = List.hd ts in
        [ Nx.add (Nx.mul t w) c; Nx.add (Nx.zeros f32 [| 4 |]) t ])
      [ x ]
  in
  equal ~msg:"kernel calls: the captured constant, once" int 1 (C.calls ());
  equal ~msg:"held bytes: the weight and the constant" int 32 prog.held;
  equal ~msg:"operations" names
    [ "Map"; "Map"; "Map"; "Map" ]
    (List.rev prog.ops);
  equal ~msg:"results are the program's" bool true
    (List.for_all (fun y -> Nx.Prim.owner y <> None) outs)

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

let capturing =
  {
    Nx.Prim.rule =
      (fun (type r) i ~by (op : r Nx.Prim.t) : r ->
        match Nx.Prim.expand i ~by op with
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
              | Map { loads = [||]; prog; shape; _ } -> (
                  match P.node prog (P.length prog - 1) with
                  | Const (_, bits) -> Lit (bits, shape)
                  | _ -> App ("creation", []))
              | _ -> App (Nx.Prim.name op, List.map arg xs)
            in
            Nx.Prim.results ~by
              { make = (fun _ form -> Nx.Prim.traced i form (Expr e)) }
              op);
  }

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
                  shape = [| 3 |];
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
      test "a staging compiler computes nothing and holds its captures"
        test_staging;
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
          Nx.Prim.interpret ~name:"test.extent" Extent (logging log) (fun _ ->
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
          Nx.Prim.interpret ~name:"test.dropped" Extent
            (logging (ref []))
            (fun _ -> yield ()) );
    ];
  equal ~msg:"computes" (array float_exact) [| 2.; 4.; 6. |]
    (elements (Nx.add x x));
  let log = ref [] in
  Nx.Prim.interpret ~name:"test.extent" Extent (logging log) (fun _ ->
      ignore (Nx.zeros f32 [| 2 |]));
  equal ~msg:"a later extent" names [ "Map" ] (received log)

(* An extent started on this domain and ended on another: each domain's extents
   still reach their creations. *)
let test_migrated () =
  let log = ref [] in
  let suspended = ref None in
  Effect.Deep.match_with
    (fun () ->
      Nx.Prim.interpret ~name:"test.migrated" Extent (logging log) (fun _ ->
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
         Nx.Prim.interpret ~name:"test.there" Extent (logging there) (fun _ ->
             ignore (Nx.zeros f32 [| 2 |]))));
  equal ~msg:"the migrated extent, from another domain" names [] (received log);
  equal ~msg:"an extent on the other domain" names [ "Map" ] (received there);
  let here = ref [] in
  Nx.Prim.interpret ~name:"test.here" Extent (logging here) (fun _ ->
      ignore (Nx.zeros f32 [| 3 |]));
  equal ~msg:"an extent on this domain" names [ "Map" ] (received here)

let fibers =
  group "fibers"
    [
      test "a fiber beside an extent computes" test_two_fibers;
      test "a continuation dropped inside an extent changes no result"
        test_dropped;
      test "an extent that ends on another domain leaves both counts right"
        test_migrated;
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
  Nx.Prim.interpret ~name:"test.extent" Extent (logging log) (fun _ ->
      ignore (Nx.add x x);
      ignore (Nx.zeros f32 [| 2 |]));
  received log

let in_values x =
  let log = ref [] in
  let y =
    Nx.Prim.interpret ~name:"test.values" Values (logging log) (fun i ->
        ignore (Nx.add (tag i x) x);
        Nx.add x x)
  in
  (received log, elements y)

let leak x =
  Nx.Prim.interpret ~name:"test.leaked" Values
    (logging (ref []))
    (fun i -> tag i x)

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

let () = exit (run "nx interp" [ rule; laws; fibers; interpreters; domains ])
