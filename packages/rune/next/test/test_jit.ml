(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Compiled calls. A call computes eager's values over every layout of its
   arguments; traces once per key, and once more for each part of the key that
   changes; gives each result storage of its own; consumes and lends storage as
   its signature says, refusing before any work what it cannot consume; binds
   its captures once; raises before consuming anything; runs from several
   domains; runs where its arguments and captures lie, on the host and on
   devices over the host's memory; and folds a scan inside its trace. *)

open Windtrap
open Nx_test
module Rune = Rune_next.Rune

let floats = tensor float_exact
let close = Oracle.tensor ~rel:1e-5 ~abs:1e-30 ()
let x () = Nx.create Nx.float32 [| 4 |] [| 1.; -2.; 3.; 0.5 |]
let y () = Nx.create Nx.float32 [| 4 |] [| 2.; 0.; -1.; 4. |]
let poly x = Nx.add (Nx.mul x x) x
let host t = Nx.place Nx.Placement.host t
let address t = List.hd (Witness.addresses t)
let consumes = Nx.Ptree.(consumes tensor @@ returns tensor)
let two = Nx.Ptree.(tensor @-> tensor @-> returns tensor)

(* Observing a call *)

(* [profiled f] is [f ()] and the host spans the compiled call recorded
   meanwhile, in order. *)
let profiled f =
  let p = Nx_device.Profile.start () in
  match f () with
  | y ->
      let spans =
        List.filter_map
          (function
            | Nx_device.Profile.Span s
              when String.starts_with ~prefix:"rune.jit: " s.name ->
                Some s.name
            | _ -> None)
          (Nx_device.Profile.stop p)
      in
      (y, spans)
  | exception e ->
      ignore (Nx_device.Profile.stop p);
      raise e

(* [traces f] is the number of traces while [f ()] runs. *)
let traces f =
  let (), spans = profiled f in
  List.length (List.filter (String.equal "rune.jit: trace") spans)

(* [counted f] is [f] and the number of times it ran. *)
let counted f =
  let n = ref 0 in
  ( (fun x ->
      incr n;
      f x),
    n )

let raises_jit_error f =
  raises_match
    (function Rune.Jit_error _ -> true | _ -> false)
    (fun () -> ignore (f ()))

let message f = Oracle.message (fun () -> ignore (f ()))

(* Devices over the host's memory, whose programs are the host's *)

let driver ?(mapping = Some Nx_device.Driver.Identity) name =
  Nx_device.Driver.device ~name ~arch:"test" ~budget:max_int
    (Host_visible { memory = Nx_device.Driver.host_memory; mapping })

let d1, d2, d3, d4 =
  match
    List.map
      (fun n -> Nx.Device.of_runtime (driver n))
      [ "J1"; "J2"; "J3"; "J4" ]
  with
  | [ a; b; c; d ] -> (a, b, c, d)
  | _ -> assert false

let on d = Nx.Placement.device ~backend:Rune.compiled d
let placed d t = Nx.place (on d) t
let stats d = Nx_device.stats (Nx.Device.runtime d)
let bytes_in d = Nx_device.Stats.bytes_in (stats d)
let allocated d = Nx_device.Stats.allocated (stats d)

(* Values *)

(* An operation of one family, by name, over two float32 arguments of one shape;
   [exact] is whether it is computed exactly. *)
type family = {
  name : string;
  exact : bool;
  light : bool;  (** Whether the default run takes it. *)
  apply : Nx.float32_t -> Nx.float32_t -> Nx.float32_t;
}

let families =
  let f ?(exact = true) ?(light = false) name apply =
    { name; exact; light; apply }
  in
  [
    f ~light:true "neg, abs, max" (fun a b -> Nx.maximum (Nx.neg a) (Nx.abs b));
    f "where a less than b" (fun a b -> Nx.where (Nx.less a b) a b);
    f ~exact:false "exp and sin" (fun a b -> Nx.add (Nx.exp a) (Nx.sin b));
    f ~exact:false ~light:true "a sum over the last axis" (fun a b ->
        Nx.add a (Nx.sum ~axes:[ -1 ] ~keepdims:true b));
    f "a maximum over every axis" (fun a b -> Nx.mul a (Nx.max b));
    f ~exact:false "a running sum" (fun a b -> Nx.add a (Nx.cumsum ~axis:0 b));
    f "a transpose made contiguous" (fun a b ->
        Nx.add a (Nx.transpose (Nx.contiguous (Nx.transpose b))));
    f ~light:true "a flip and a pad" (fun a b ->
        Nx.add a
          (Nx.shrink
             (Array.map (fun n -> (1, n + 1)) (Nx.shape b))
             (Nx.pad (Array.map (fun _ -> (1, 1)) (Nx.shape b)) 0. (Nx.flip b))));
    f "a concatenation sliced back" (fun a b ->
        Nx.add a
          (Nx.shrink
             (Array.mapi
                (fun i n -> if i = 0 then (n, 2 * n) else (0, n))
                (Nx.shape b))
             (Nx.concatenate ~axis:0 [ a; b ])));
    f "a cast to int32 and back" (fun a b ->
        Nx.add a (Nx.cast Nx.float32 (Nx.cast Nx.int32 b)));
    f ~exact:false "a product with the transpose" (fun a b ->
        Nx.add a (Nx.matmul (Nx.matmul a (Nx.matrix_transpose b)) b));
  ]

(* Two arguments of one shape under one drawn layout: transposed, flipped, every
   other row, offset, broadcast, in windows. *)
let operands =
  let open Gen in
  let* rows = int_range 1 4 in
  let* cols = int_range 1 4 in
  let value = float_range (-2.) 2. in
  let* a = array ~size:(constant (rows * cols)) value in
  let* b = array ~size:(constant (rows * cols)) value in
  let+ steps = layout in
  let make xs = lay_out steps (Nx.create Nx.float32 [| rows; cols |] xs) in
  (make a, make b)

let laid =
  Gen.with_pp
    (fun ppf (a, b) -> Format.fprintf ppf "%a@ %a" Nx.pp a Nx.pp b)
    operands

(* The laws of values, [count] cases each; [heavy] adds the families the default
   run leaves out, and the tests whose programs take longest to compile. *)
let values ~count ~heavy =
  let law { name; exact; apply; _ } =
    prop ~count name laid (fun (a, b) ->
        match apply a b with
        | expected ->
            equal
              (if exact then floats else close)
              expected (Rune.jit two apply a b)
        | exception Invalid_argument m ->
            (* An empty extreme raises eagerly; compiled, it raises too. *)
            raises_match ~msg:m Exn.invalid_arg (fun () ->
                Rune.jit two apply a b))
  in
  group "values"
    [
      group "one operation per family equals eager"
        (List.map law (List.filter (fun f -> heavy || f.light) families));
      test "a replay reads its new arguments, and an earlier call's again"
        (fun () ->
          let g = Rune.jit' poly in
          equal floats (poly (x ())) (g (x ()));
          equal floats (poly (y ())) (g (y ()));
          equal floats (poly (x ())) (g (x ())));
      test "a structured result equals eager's leaf by leaf" (fun () ->
          let s = Nx.Ptree.(pair tensor (list (option tensor))) in
          let f a = (Nx.neg a, [ Some (poly a); None; Some a ]) in
          equal (Oracle.structure s)
            (f (x ()))
            (Rune.jit Nx.Ptree.(tensor @-> returns s) f (x ())));
      test "64-bit integer constants keep every bit" (fun () ->
          let a = Nx.create Nx.int64 [| 2 |] [| 1L; -1L |] in
          let f a =
            Nx.add a
              (Nx.create Nx.int64 [| 2 |] [| Int64.max_int; Int64.min_int |])
          in
          equal (tensor int64) (f a) (Rune.jit' f a));
      test "integer constants wrap at the operand's width" (fun () ->
          let a = Nx.create Nx.int8 [| 3 |] [| 127; -128; 100 |] in
          let f a = Nx.add (Nx.mul_s a 3) (Nx.full Nx.int8 [| 3 |] 100) in
          equal (tensor int) (f a) (Rune.jit' f a));
      test "float identities hold only where IEEE keeps them" (fun () ->
          let a =
            Nx.create Nx.float32 [| 4 |]
              [| -0.; 0.; Float.infinity; Float.nan |]
          in
          let f a =
            Nx.stack ~axis:0
              [
                Nx.add a (Nx.zeros_like a);
                Nx.div a a;
                Nx.mul a (Nx.zeros_like a);
              ]
          in
          equal floats (f a) (Rune.jit' f a));
      test "a zero-size result is an empty tensor" (fun () ->
          let a = Nx.zeros Nx.float32 [| 0; 3 |] in
          let r = Rune.jit' poly a in
          equal (array int) [| 0; 3 |] (Nx.shape r));
    ]

(* Keys *)

(* A structure that reports an integer. *)
module Windowed = struct
  type 'a t = { n : int; x : 'a }

  let walk c { n; x } =
    let open Nx.Ptree.Walk in
    let n = field c "n" int n in
    let x = field c "x" leaf x in
    { n; x }
end

let windowed : (float, Nx.float32_elt) Nx.t Windowed.t Nx.Ptree.t =
  Nx.Ptree.instantiate (module Windowed)

(* A structure with two cases. *)
module Choice = struct
  type 'a t = Left of 'a | Right of 'a

  let walk c =
    let open Nx.Ptree.Walk in
    function
    | Left x ->
        case c "left";
        Left (leaf c x)
    | Right x ->
        case c "right";
        Right (leaf c x)
end

let choice : (float, Nx.float32_elt) Nx.t Choice.t Nx.Ptree.t =
  Nx.Ptree.instantiate (module Choice)

(* [retraces first other] asserts that [first ()] traces once, [other ()] once
   more, and [other ()] again not at all. *)
let retraces first other =
  equal ~msg:"the first call" int 1 (traces first);
  equal ~msg:"the other key" int 1 (traces other);
  equal ~msg:"the other key again" int 0 (traces other)

(* [shares first other] asserts that [other ()] replays the program of [first
   ()]. *)
let shares first other =
  first ();
  equal ~msg:"traces" int 0 (traces other)

(* [checked g f a] is [g a], checked against eager's [f a]. *)
let checked g f a () = equal close (f a) (g a)
let arange n = Nx.arange_f Nx.float32 0. (float_of_int n) 1.
let grid r c = Nx.reshape [| r; c |] (arange (r * c))

(* Every other element of [t] along its first axis, as a view. *)
let every_other t =
  Nx.squeeze ~axes:[ -1 ] (Nx.sliding_window ~axis:0 ~window:1 ~step:2 t)

let keys =
  let g () = Rune.jit' poly in
  group "keys"
    [
      test "a call with the key of an earlier one replays its program"
        (fun () ->
          let f, ran = counted poly in
          let g = Rune.jit' f in
          ignore (g (x ()));
          equal int 0 (traces (fun () -> ignore (g (y ()))));
          equal int 1 !ran);
      test "another extent retraces once" (fun () ->
          let g = g () in
          retraces (checked g poly (x ())) (checked g poly (arange 5)));
      test "another rank retraces once" (fun () ->
          let g = g () in
          retraces (checked g poly (arange 4)) (checked g poly (grid 2 2)));
      test "strides out of C order retrace once" (fun () ->
          let g = g () in
          retraces
            (checked g poly (grid 2 3))
            (checked g poly (Nx.transpose (grid 3 2))));
      test
        "an argument starting 4 bytes further within 16 bytes of memory \
         retraces once" (fun () ->
          let g = g () in
          let a = arange 12 in
          retraces
            (checked g poly (Nx.slice [ R (0, 4) ] a))
            (checked g poly (Nx.slice [ R (1, 5) ] a)));
      test "an argument starting 16 bytes further shares the program" (fun () ->
          let g = g () in
          let a = arange 12 in
          shares
            (checked g poly (Nx.slice [ R (0, 4) ] a))
            (checked g poly (Nx.slice [ R (4, 8) ] a)));
      test
        "strides out of C order only on axes of one element share the program"
        (fun () ->
          let g = g () in
          shares
            (checked g poly (grid 4 1))
            (checked g poly (Nx.transpose (grid 1 4))));
      test "another reported integer retraces once" (fun () ->
          let f { Windowed.n; x } = Nx.mul_s x (float_of_int n) in
          let g = Rune.jit Nx.Ptree.(windowed @-> returns tensor) f in
          let w n () = { Windowed.n; x = x () } in
          retraces
            (fun () -> equal close (f (w 1 ())) (g (w 1 ())))
            (fun () -> equal close (f (w 2 ())) (g (w 2 ()))));
      slow "another case retraces once" (fun () ->
          let f = function Choice.Left x -> Nx.neg x | Right x -> poly x in
          let g = Rune.jit Nx.Ptree.(choice @-> returns tensor) f in
          retraces
            (fun () -> equal close (f (Left (x ()))) (g (Left (x ()))))
            (fun () -> equal close (f (Right (x ()))) (g (Right (x ())))));
      slow "another list length retraces once" (fun () ->
          let f l = List.fold_left Nx.add (x ()) l in
          let g = Rune.jit Nx.Ptree.(list tensor @-> returns tensor) f in
          retraces
            (fun () -> equal close (f [ y () ]) (g [ y () ]))
            (fun () -> equal close (f [ y (); y () ]) (g [ y (); y () ])));
      slow "an option's presence retraces once" (fun () ->
          let f = function None -> x () | Some a -> poly a in
          let g = Rune.jit Nx.Ptree.(option tensor @-> returns tensor) f in
          retraces
            (fun () -> equal close (f None) (g None))
            (fun () -> equal close (f (Some (y ()))) (g (Some (y ())))));
      slow "a key met again after another replays its first program" (fun () ->
          let g = g () in
          ignore (g (x ()));
          ignore (g (arange 5));
          equal int 0 (traces (fun () -> ignore (g (y ())))));
      slow "two compiled functions of one function keep their own programs"
        (fun () ->
          let g1 = Rune.jit' poly and g2 = Rune.jit' poly in
          equal int 1 (traces (fun () -> ignore (g1 (x ()))));
          equal int 1 (traces (fun () -> ignore (g2 (x ())))));
      test "a change of NOOPT around a call retraces once" (fun () ->
          let g = g () in
          let noopt f =
            Tolk_next.Helpers.context [ B (Tolk_next.Helpers.noopt, true) ] f
          in
          retraces
            (checked g poly (x ()))
            (fun () -> noopt (checked g poly (x ()))));
      slow "a flipped view retraces once" (fun () ->
          let g = g () in
          retraces
            (checked g poly (grid 2 3))
            (checked g poly (Nx.flip (grid 2 3))));
      slow "a broadcast view retraces once" (fun () ->
          let g = g () in
          retraces
            (checked g poly (grid 2 3))
            (checked g poly (Nx.broadcast_to [| 2; 3 |] (grid 1 3))));
      slow "a view skipping elements retraces once" (fun () ->
          let g = g () in
          retraces
            (checked g poly (arange 3))
            (checked g poly (every_other (arange 6))));
      slow "overlapping windows retrace once" (fun () ->
          let g = g () in
          retraces
            (checked g poly (grid 4 2))
            (checked g poly (Nx.sliding_window ~window:2 (arange 5))));
      slow "another device retraces once" (fun () ->
          let g = g () in
          retraces
            (fun () -> equal close (poly (x ())) (host (g (placed d1 (x ())))))
            (fun () -> equal close (poly (x ())) (host (g (placed d2 (x ()))))));
      slow "another backend on one device retraces once" (fun () ->
          let g = g () in
          retraces
            (fun () -> equal close (poly (x ())) (host (g (placed d1 (x ())))))
            (fun () ->
              equal close
                (poly (x ()))
                (host (g (Nx.place (Nx.Placement.device d1) (x ()))))));
      slow "a split value after a replicated one retraces once" (fun () ->
          let g = g () in
          let on p () =
            equal close (poly (x ())) (host (g (Nx.place p (x ()))))
          in
          retraces
            (on (Nx.Placement.replicated ~backend:Rune.compiled [ d1; d2 ]))
            (on
               (Nx.Placement.sharded ~backend:Rune.compiled ~axis:0 [ d1; d2 ])));
    ]

(* Results *)

let distinct a b =
  is_false ~msg:"distinct storage"
    (List.exists2 Nativeint.equal (Witness.addresses a) (Witness.addresses b))

let results =
  group "results"
    [
      test "every result leaf has storage of its own" (fun () ->
          let a, b =
            Rune.jit
              Nx.Ptree.(tensor @-> returns (pair tensor tensor))
              (fun a -> (Nx.neg a, poly a))
              (x ())
          in
          distinct a b);
      test "a result that returns a read argument is a copy" (fun () ->
          let a = x () in
          let r = Rune.jit' Fun.id a in
          equal floats a r;
          distinct a r);
      test "a result that returns a capture is a copy" (fun () ->
          let w = y () in
          let r = Rune.jit' (fun _ -> w) (x ()) in
          equal floats w r;
          distinct w r);
      test "a value at two result leaves comes back as two values" (fun () ->
          let a, b =
            Rune.jit
              Nx.Ptree.(tensor @-> returns (pair tensor tensor))
              (fun a ->
                let r = poly a in
                (r, r))
              (x ())
          in
          equal floats a b;
          distinct a b);
      test "one value passed at two read leaves is read at both" (fun () ->
          let a = x () in
          equal floats (Nx.add a a) (Rune.jit two Nx.add a a));
    ]

(* Consumption *)

let consumed_message a = message (fun () -> Nx.to_array a)

let consumption =
  group "consumption"
    [
      test "a consumed argument raises on read, naming its path" (fun () ->
          let a = x () in
          ignore (Rune.jit consumes (fun a -> Nx.add_s a 1.) a);
          let m = consumed_message a in
          is_true ~msg:m (String.length m > 0);
          raises_match (Exn.invalid_arg ~substring:"consumed at 0") (fun () ->
              ignore (Nx.to_array a)));
      test "a consumed argument keeps its shape and dtype" (fun () ->
          let a = Nx.zeros Nx.float32 [| 2; 3 |] in
          ignore (Rune.jit consumes Nx.neg a);
          equal (array int) [| 2; 3 |] (Nx.shape a);
          is_true (Nx_dtype.equal Nx.float32 (Nx.dtype a)));
      test "a consumed argument raises as an operand and as an argument"
        (fun () ->
          let a = x () in
          ignore (Rune.jit consumes Nx.neg a);
          raises_invalid_arg (fun () -> Nx.add a a);
          raises_invalid_arg (fun () -> Rune.jit' Nx.neg a);
          raises_invalid_arg (fun () -> Rune.jit consumes Nx.neg a));
      test "a view of consumed storage taken before the call raises on read"
        (fun () ->
          let a = x () in
          let v = Nx.slice [ R (0, 2) ] a in
          ignore (Rune.jit consumes Nx.neg a);
          raises_invalid_arg (fun () -> Nx.to_array v));
      test "a consumed slice raises before any work, consuming nothing"
        (fun () ->
          let a = Nx.slice [ R (0, 2) ] (x ()) in
          raises_match
            (Exn.invalid_arg
               ~substring:
                 "0 is consumed and does not cover its whole storage; consume \
                  Nx.copy of it") (fun () -> Rune.jit consumes Nx.neg a);
          equal floats (Nx.slice [ R (0, 2) ] (x ())) a);
      test
        "a consumed leaf that another leaf reaches raises before any work, \
         naming both paths" (fun () ->
          let a = x () in
          let g =
            Rune.jit
              Nx.Ptree.(consumes tensor @@ tensor @-> returns tensor)
              Nx.add
          in
          raises_match
            (Exn.invalid_arg
               ~substring:"0 is consumed and 1 reaches its storage") (fun () ->
              g a a);
          equal floats (x ()) a);
      test "two consumed leaves over one storage raise before any work"
        (fun () ->
          let a = x () in
          let g =
            Rune.jit
              Nx.Ptree.(consumes (pair tensor tensor) @@ returns tensor)
              (fun (a, b) -> Nx.add a b)
          in
          raises_invalid_arg (fun () -> g (a, a));
          equal floats (x ()) a);
      test "a consumed leaf whose storage the function captures raises"
        (fun () ->
          let w = y () in
          let g = Rune.jit consumes (fun a -> Nx.add a w) in
          raises_match
            (Exn.invalid_arg
               ~substring:"0 is consumed and the function captures its storage")
            (fun () -> g w);
          equal floats (y ()) w);
      test "a call that raises while tracing consumes nothing" (fun () ->
          let a = x () in
          raises_jit_error (fun () ->
              Rune.jit consumes
                (fun a -> if Nx.item [ 0 ] a > 0. then a else Nx.neg a)
                a);
          equal floats (x ()) a);
      test "read arguments stay readable after any number of calls" (fun () ->
          let a = x () in
          let g = Rune.jit' poly in
          for _ = 1 to 3 do
            ignore (g a)
          done;
          equal floats (x ()) a);
    ]

(* Lending *)

let state =
  Nx.Ptree.(consumes (pair tensor tensor) @@ returns (pair tensor tensor))

let lending =
  group "lending"
    [
      test "a consumed host argument lends its storage to the result" (fun () ->
          let a = x () in
          let before = address a in
          let r = Rune.jit consumes (fun a -> Nx.add_s a 1.) a in
          equal floats (Nx.add_s (x ()) 1.) r;
          equal nativeint before (address r));
      test "a result takes the consumed leaf it derives from at its own index"
        (fun () ->
          let a = x () and b = y () in
          let ab = (address a, address b) in
          let r1, r2 =
            Rune.jit state (fun (a, b) -> (Nx.add_s b 1., Nx.mul_s a 2.)) (a, b)
          in
          equal (pair nativeint nativeint) ab (address r2, address r1));
      test "an indexed write takes the leaf it writes before any other result"
        (fun () ->
          let a = x () and b = y () in
          let ab = (address a, address b) in
          let r1, r2 =
            Rune.jit state
              (fun (a, _) ->
                ( Nx.full Nx.float32 [| 4 |] 7.,
                  Nx.set [ I 0 ] (Nx.scalar Nx.float32 9.) a ))
              (a, b)
          in
          equal floats (Nx.set [ I 0 ] (Nx.scalar Nx.float32 9.) (x ())) r2;
          equal (pair nativeint nativeint) ab (address r2, address r1));
      test "the other results take the free leaves in walk order" (fun () ->
          let a = x () and b = y () in
          let ab = (address a, address b) in
          let r1, r2 =
            Rune.jit state
              (fun _ ->
                let r1 = Nx.full Nx.float32 [| 4 |] 1. in
                let r2 = Nx.full Nx.float32 [| 4 |] 2. in
                (r1, r2))
              (a, b)
          in
          equal (pair nativeint nativeint) ab (address r1, address r2));
      test "a storage lends to one result" (fun () ->
          let a = x () in
          let before = address a in
          let r1, r2 =
            Rune.jit
              Nx.Ptree.(consumes tensor @@ returns (pair tensor tensor))
              (fun a -> (Nx.add_s a 1., Nx.mul_s a 2.))
              a
          in
          equal nativeint before (address r1);
          distinct r1 r2;
          equal floats (Nx.mul_s (x ()) 2.) r2);
      cases ~name:fst
        "a result that reads its consumed leaf at other indices takes fresh \
         storage"
        [
          ( "left rotation",
            ( (fun a ->
                Nx.concatenate ~axis:0
                  [ Nx.slice [ R (1, 8) ] a; Nx.slice [ R (0, 1) ] a ]),
              [| 2.; 3.; 4.; 5.; 6.; 7.; 8.; 1. |] ) );
          ( "right rotation",
            ( (fun a ->
                Nx.concatenate ~axis:0
                  [ Nx.slice [ R (7, 8) ] a; Nx.slice [ R (0, 7) ] a ]),
              [| 8.; 1.; 2.; 3.; 4.; 5.; 6.; 7. |] ) );
          ( "flip",
            ( (fun a -> Nx.add_s (Nx.flip a) 1.),
              [| 9.; 8.; 7.; 6.; 5.; 4.; 3.; 2. |] ) );
        ]
        (fun (_, (f, expected)) ->
          let a = Nx.arange_f Nx.float32 1. 9. 1. in
          let before = address a in
          let r = Rune.jit consumes f a in
          equal floats (Nx.create Nx.float32 [| 8 |] expected) r;
          is_false (Nativeint.equal before (address r));
          raises_invalid_arg (fun () -> Nx.to_array a));
      test "a result of another dtype does not take the leaf" (fun () ->
          let a = x () in
          let before = address a in
          let r =
            Rune.jit
              Nx.Ptree.(consumes tensor @@ returns tensor)
              (fun a -> Nx.cast Nx.float64 a)
              a
          in
          equal (tensor float_exact) (Nx.cast Nx.float64 (x ())) r;
          is_false (Nativeint.equal before (address r)));
      test "a consumed leaf returned unchanged is lent with no store" (fun () ->
          let a = x () in
          let before = address a in
          let r = Rune.jit consumes Fun.id a in
          equal floats (x ()) r;
          equal nativeint before (address r));
      test "a borrowed consumed argument is copied, and still consumed"
        (fun () ->
          let ba =
            Bigarray.Array1.of_array Bigarray.float32 Bigarray.c_layout
              [| 1.; 2.; 3.; 4. |]
          in
          let a = Nx.of_bigarray (Bigarray.genarray_of_array1 ba) in
          let before = address a in
          let r = Rune.jit consumes (fun a -> Nx.mul_s a 2.) a in
          equal floats (Nx.create Nx.float32 [| 4 |] [| 2.; 4.; 6.; 8. |]) r;
          is_false (Nativeint.equal before (address r));
          raises_invalid_arg (fun () -> Nx.to_array a));
      test
        "a window written at a position read when the call runs reuses the \
         cache" (fun () ->
          let cache = Nx.zeros Nx.float32 [| 4; 3 |] in
          let before = address cache in
          let step =
            Rune.jit
              Nx.Ptree.(consumes tensor @@ tensor @-> tensor @-> returns tensor)
              (fun cache pos row -> Nx.set [ D (pos, 1) ] row cache)
          in
          let row = Nx.ones Nx.float32 [| 1; 3 |] in
          let r = step cache (Nx.scalar Nx.int32 2l) row in
          let expected =
            Nx.set
              [ R (2, 3) ]
              (Nx.ones Nx.float32 [| 1; 3 |])
              (Nx.zeros Nx.float32 [| 4; 3 |])
          in
          equal floats expected r;
          equal nativeint before (address r));
      test "two programs alternating on one consumed state keep its storage"
        (fun () ->
          let a = x () in
          let before = address a in
          let inc = Rune.jit consumes (fun a -> Nx.add_s a 1.)
          and dbl = Rune.jit consumes (fun a -> Nx.mul_s a 2.) in
          let r = ref a in
          for _ = 1 to 3 do
            r := dbl (inc !r)
          done;
          let expected =
            List.fold_left
              (fun a _ -> Nx.mul_s (Nx.add_s a 1.) 2.)
              (x ()) [ 1; 2; 3 ]
          in
          equal floats expected !r;
          equal nativeint before (address !r));
      test "every leaf of a consumed state derived at its own index is lent"
        (fun () ->
          let s = Nx.Ptree.(list tensor) in
          let leaves () =
            List.init 4 (fun i -> Nx.full Nx.float32 [| 3 |] (float_of_int i))
          in
          let ls = leaves () in
          let before = List.map address ls in
          let r =
            Rune.jit
              Nx.Ptree.(consumes s @@ returns s)
              (List.map (fun l -> Nx.add_s l 1.))
              ls
          in
          equal (list nativeint) before (List.map address r));
      test "a loop consuming its state holds two generations of it" (fun () ->
          let n = 1 lsl 16 in
          let step = Rune.jit consumes (fun a -> Nx.add_s a 1.) in
          let s = ref (placed d3 (Nx.zeros Nx.float32 [| n |])) in
          s := step !s;
          let base = allocated d3 in
          for _ = 1 to 20 do
            s := step !s
          done;
          at_most ~msg:"bytes allocated across 20 steps" int ~than:(4 * n)
            (allocated d3 - base);
          equal (tensor float_exact) (Nx.full Nx.float32 [| n |] 21.) (host !s));
    ]

(* Captures *)

let storage_of x =
  match Nx.Repr.v x with
  | Placed p -> Nx.Repr.Placed.storage p
  | Host _ | Traced _ -> fail "expected a placed value"

let captures =
  group "captures"
    [
      test "a captured tensor is a constant of the program" (fun () ->
          let w = y () in
          let g = Rune.jit' (fun a -> Nx.mul a w) in
          equal floats (Nx.mul (x ()) w) (g (x ()));
          equal floats (Nx.mul (y ()) w) (g (y ())));
      test "a capture of one element on a device is bound as a constant"
        (fun () ->
          let w = placed d1 (Nx.scalar Nx.float32 3.) in
          let g = Rune.jit' (fun a -> Nx.mul a w) in
          equal close (Nx.mul_s (x ()) 3.) (host (g (placed d1 (x ()))));
          equal int 0 (Nx.Repr.Storage.pins (storage_of w)));
      test "a capture placed where the call computes is bound, not uploaded"
        (fun () ->
          let w = placed d1 (y ()) in
          let g = Rune.jit' (fun a -> Nx.mul a w) in
          let a = placed d1 (x ()) in
          ignore (g a);
          equal ~msg:"pins" int 1 (Nx.Repr.Storage.pins (storage_of w));
          let before = bytes_in d1 in
          let r = g a in
          equal ~msg:"bytes received" int before (bytes_in d1);
          equal close (Nx.mul (x ()) (y ())) (host r));
      test "a host capture of a call on a device is placed there once"
        (fun () ->
          let w = y () in
          let g = Rune.jit' (fun a -> Nx.mul a w) in
          let a = placed d2 (x ()) in
          ignore (g a);
          let before = bytes_in d2 in
          for _ = 1 to 3 do
            ignore (g a)
          done;
          equal ~msg:"bytes received by later calls" int before (bytes_in d2));
      test "a capture decides the device of a call of host arguments" (fun () ->
          let w = placed d1 (y ()) in
          let r = Rune.jit' (fun a -> Nx.mul a w) (x ()) in
          is_true (Nx.Placement.equal (on d1) (Nx.placement r));
          equal close (Nx.mul (x ()) (y ())) (host r));
      test
        "a host capture another call consumes makes the program raise, naming \
         its path" (fun () ->
          let w = y () in
          let g = Rune.jit' (fun a -> Nx.mul a w) in
          ignore (g (x ()));
          ignore (Rune.jit consumes Nx.neg w);
          raises_match (Exn.invalid_arg ~substring:"consumed at 0") (fun () ->
              g (x ())));
      test "two compiled functions share one captured buffer" (fun () ->
          let w = placed d1 (y ()) in
          let g1 = Rune.jit' (fun a -> Nx.mul a w)
          and g2 = Rune.jit' (fun a -> Nx.add a w) in
          let a = placed d1 (x ()) in
          equal close (Nx.mul (x ()) (y ())) (host (g1 a));
          equal close (Nx.add (x ()) (y ())) (host (g2 a));
          equal ~msg:"pins" int 2 (Nx.Repr.Storage.pins (storage_of w)));
      test "a draw from a key the function captures raises Jit_error" (fun () ->
          raises_jit_error (fun () ->
              Rune.jit'
                (fun a ->
                  Nx.add a
                    (Nx.Rng.with_key (Nx.Rng.key 42) (fun () ->
                         Nx.rand Nx.float32 [| 4 |])))
                (x ())));
      test "a draw from a captured key at counters of the arguments computes"
        (fun () ->
          let key =
            Nx.create Nx.int32 [| 4; 2 |]
              (Array.init 8 (fun i -> Int32.of_int (7 + (i mod 2))))
          in
          let draw c = Nx.Op.eval (Nx.Op.Threefry (key, c)) in
          let c = Nx.create Nx.int32 [| 4; 2 |] (Array.init 8 Int32.of_int) in
          equal (tensor int32) (draw c) (Rune.jit' draw c));
      test "an empty draw from a captured key computes" (fun () ->
          let key = Nx.zeros Nx.int32 [| 0; 2 |] in
          let draw x = Nx.Op.eval (Nx.Op.Threefry (key, Nx.cast Nx.int32 x)) in
          let x = Nx.zeros Nx.float64 [| 0; 2 |] in
          equal (tensor int32) (draw x) (Rune.jit' draw x));
      slow "a draw from a key the function takes draws again at each call"
        (fun () ->
          let g =
            Rune.jit
              Nx.Ptree.(tensor @-> returns tensor)
              (fun k ->
                Nx.Rng.with_key (Nx.Rng.of_tensor k) (fun () ->
                    Nx.rand Nx.float32 [| 4 |]))
          in
          let draw k =
            Nx.Rng.with_key k (fun () -> Nx.rand Nx.float32 [| 4 |])
          in
          let k1 = Nx.Rng.key 1 and k2 = Nx.Rng.key 2 in
          equal floats (draw k1) (g (k1 :> Nx.int32_t));
          equal floats (draw k2) (g (k2 :> Nx.int32_t)));
    ]

(* Errors *)

let twin1 = Nx.Device.of_runtime (driver "TWIN")
let twin2 = Nx.Device.of_runtime (driver "TWIN")

let errors =
  let leaked = ref None in
  let messages =
    [
      ( "operands on two devices",
        fun () ->
          Rune.jit two Nx.add (placed d1 (x ())) (placed d2 (y ())) |> ignore );
      ( "a name met with two devices",
        fun () ->
          Rune.jit
            Nx.Ptree.(tensor @-> tensor @-> returns (pair tensor tensor))
            (fun a b -> (Nx.neg a, Nx.neg b))
            (placed twin1 (x ()))
            (placed twin2 (y ()))
          |> ignore );
      ( "a consumed slice",
        fun () ->
          Rune.jit consumes Nx.neg (Nx.slice [ R (0, 2) ] (x ())) |> ignore );
    ]
  in
  group "errors"
    [
      test "reading a traced value raises Jit_error" (fun () ->
          raises_jit_error (fun () ->
              Rune.jit'
                (fun a -> if Nx.item [ 0 ] a > 0. then a else Nx.neg a)
                (x ())));
      test "an operation no target computes raises Jit_error" (fun () ->
          raises_jit_error (fun () ->
              Rune.jit'
                (fun a -> Nx.real Nx.float32 (Nx.fft (Nx.cast Nx.complex64 a)))
                (x ())));
      test "operands on two devices raise nx's message" (fun () ->
          raises_invalid_arg (List.assoc "operands on two devices" messages));
      test "a name met with two devices raises" (fun () ->
          raises_invalid_arg (List.assoc "a name met with two devices" messages));
      test "a traced value kept after the call raises on read" (fun () ->
          let r =
            Rune.jit'
              (fun a ->
                let t = poly a in
                leaked := Some t;
                t)
              (x ())
          in
          equal close (poly (x ())) r;
          match !leaked with
          | Some t -> raises_invalid_arg (fun () -> Nx.to_array t)
          | None -> fail "the function did not run");
      test "a call that raised traces again at the next call" (fun () ->
          let g = Rune.jit' (fun a -> if Nx.item [ 0 ] a > 0. then a else a) in
          equal int 1 (traces (fun () -> raises_jit_error (fun () -> g (x ()))));
          equal int 1 (traces (fun () -> raises_jit_error (fun () -> g (x ())))));
      test "no message names the compiled call's internals" (fun () ->
          List.iter
            (fun (name, f) ->
              let m = message f in
              List.iter
                (fun word -> not_contains ~msg:name ~sub:word m)
                [ "PARAM"; "UOp"; "slot"; "latch"; "Lower"; "Staged" ])
            messages);
    ]

(* Domains *)

(* [together fs] runs each of [fs] on a domain of its own, all released at once,
   and is their results. *)
let together fs =
  let go = Atomic.make false in
  let ds =
    List.map
      (fun f ->
        Domain.spawn (fun () ->
            while not (Atomic.get go) do
              Domain.cpu_relax ()
            done;
            f ()))
      fs
  in
  Atomic.set go true;
  List.map Domain.join ds

let domains =
  group "domains"
    [
      test "two domains meeting one new key trace it once" (fun () ->
          let g = Rune.jit' poly in
          let rs, spans =
            profiled (fun () ->
                together [ (fun () -> g (x ())); (fun () -> g (x ())) ])
          in
          List.iter (equal close (poly (x ()))) rs;
          equal int 1
            (List.length (List.filter (String.equal "rune.jit: trace") spans)));
      test "two domains replay one program, each reading its own arguments"
        (fun () ->
          let g = Rune.jit' poly in
          ignore (g (x ()));
          let run k () =
            List.init 20 (fun i ->
                let v =
                  Nx.full Nx.float32 [| 4 |] (float_of_int ((k * 100) + i))
                in
                (poly v, g v))
          in
          List.iter
            (List.iter (fun (e, a) -> equal close e a))
            (together [ run 1; run 2 ]));
    ]

(* Transformations *)

let transformations =
  group "transformations"
    [
      test "under grad a compiled function consumes nothing" (fun () ->
          let a = x () in
          let f a = Nx.sum (Rune.jit consumes poly a) in
          ignore (Rune.grad' f a);
          equal floats (x ()) a);
      test "under a transformation a compiled function runs its function"
        (fun () ->
          let f a = Nx.sum (poly a) in
          equal floats (Rune.grad' f (x ())) (Rune.grad' (Rune.jit' f) (x ())));
      test
        "a compiled function called inside another one's trace traces through"
        (fun () ->
          let inner = Rune.jit' Nx.neg in
          let g = Rune.jit' (fun a -> inner (inner a)) in
          equal int 1 (traces (fun () -> equal floats (x ()) (g (x ())))));
      test "a remat's gradient under jit is its function's" (fun () ->
          let g =
            Rune.remat
              Nx.Ptree.(tensor @-> returns tensor)
              (fun x -> Nx.tanh (Nx.mul x x))
          in
          let f x = Nx.sum (g (Nx.mul_s x 2.)) in
          let x = Nx.create Nx.float64 [| 4 |] [| 0.5; -1.; 0.25; 2. |] in
          equal
            (Oracle.tensor ~abs:1e-12 ~rel:1e-9 ())
            (Rune.grad'
               (fun x ->
                 Nx.sum (Nx.tanh (Nx.mul (Nx.mul_s x 2.) (Nx.mul_s x 2.))))
               x)
            (Rune.jit' (Rune.grad' f) x));
    ]

(* Placement and views *)

let placement =
  group "placement"
    [
      test "a call runs where its arguments lie, and leaves its results there"
        (fun () ->
          let r = Rune.jit' poly (placed d1 (x ())) in
          is_true (Nx.Placement.equal (on d1) (Nx.placement r));
          equal close (poly (x ())) (host r));
      test "a host argument of a call on a device is uploaded at each call"
        (fun () ->
          let g = Rune.jit two Nx.mul in
          let a = placed d2 (x ()) in
          ignore (g a (y ()));
          let before = bytes_in d2 in
          ignore (g a (y ()));
          equal ~msg:"bytes received" int (before + 16) (bytes_in d2));
      test "a placed argument feeds a call with no transfer" (fun () ->
          let g = Rune.jit' poly in
          let a = placed d1 (x ()) in
          ignore (g a);
          let before = bytes_in d1 in
          ignore (g a);
          equal int before (bytes_in d1));
      test "a placed view is read where it lies" (fun () ->
          let a = Nx.transpose (placed d1 (grid 2 3)) in
          let before = bytes_in d1 in
          let r = Rune.jit' poly a in
          equal int before (bytes_in d1);
          equal close (poly (Nx.transpose (grid 2 3))) (host r));
      test "a split argument computes on each device, and stays split"
        (fun () ->
          let p =
            Nx.Placement.sharded ~backend:Rune.compiled ~axis:0 [ d1; d2 ]
          in
          let r = Rune.jit' poly (Nx.place p (x ())) in
          is_true (Nx.Placement.equal p (Nx.placement r));
          equal close (poly (x ())) (host r));
      test "a consumed split state is lent on every device" (fun () ->
          let p =
            Nx.Placement.sharded ~backend:Rune.compiled ~axis:0 [ d3; d4 ]
          in
          let a = Nx.place p (x ()) in
          let before = Witness.addresses a in
          let r = Rune.jit consumes (fun a -> Nx.add_s a 1.) a in
          equal (list nativeint) before (Witness.addresses r);
          is_true (Nx.Placement.equal p (Nx.placement r)));
      slow "a product and a sum over four devices equal one device" (fun () ->
          let p =
            Nx.Placement.sharded ~backend:Rune.compiled ~axis:0
              [ d1; d2; d3; d4 ]
          in
          let w =
            Nx.place
              (Nx.Placement.replicated ~backend:Rune.compiled [ d1; d2; d3; d4 ])
              (grid 3 3)
          in
          let f a = Nx.sum ~axes:[ 1 ] (Nx.matmul a w) in
          let a = grid 4 3 in
          let eager = host (f (Nx.place p a)) in
          equal close (Nx.sum ~axes:[ 1 ] (Nx.matmul a (grid 3 3))) eager;
          equal close eager (host (Rune.jit' f (Nx.place p a))));
    ]

(* Scans and remats *)

let scans =
  let cumulative xs =
    Rune.scan'
      ~f:(fun c x -> (Nx.add c x, Nx.mul c x))
      ~init:(Nx.zeros Nx.float32 [| 2 |])
      xs
  in
  group "scans"
    [
      test "a scan folds inside the trace and equals eager" (fun () ->
          let f xs = snd (cumulative xs) in
          equal close (f (grid 3 2)) (Rune.jit' f (grid 3 2)));
      test "a gradient through a scan equals eager's" (fun () ->
          let f xs = Nx.sum (snd (cumulative xs)) in
          equal close
            (Rune.grad' f (grid 3 2))
            (Rune.jit' (Rune.grad' f) (grid 3 2)));
      test "a carry that changes its shape across steps is written out"
        (fun () ->
          let f xs =
            fst
              (Rune.scan'
                 ~f:(fun c x -> (Nx.concatenate ~axis:0 [ c; x ], x))
                 ~init:(Nx.zeros Nx.float32 [| 1 |])
                 xs)
          in
          let xs = Nx.reshape [| 3; 1 |] (arange 3) in
          equal close (f xs) (Rune.jit' f xs));
      test "an empty scan axis raises Rune.scan's message" (fun () ->
          let f xs = snd (cumulative xs) in
          let xs = Nx.zeros Nx.float32 [| 0; 2 |] in
          equal string
            (message (fun () -> f xs))
            (message (fun () -> Rune.jit' f xs)));
      test "a remat under a compiled gradient equals eager's" (fun () ->
          let block a = Nx.tanh (Nx.mul a a) in
          let f a =
            Nx.sum (Rune.remat Nx.Ptree.(tensor @-> returns tensor) block a)
          in
          (* Away from tanh's saturation, where 1 - tanh² is a difference of
             nearly equal numbers. *)
          let a = Nx.create Nx.float32 [| 4 |] [| 0.5; -0.3; 0.8; 0.1 |] in
          equal close (Rune.grad' f a) (Rune.jit' (Rune.grad' f) a));
    ]

let () =
  exit
    (run "Rune_next.Jit"
       [
         values ~count:1 ~heavy:false;
         keys;
         results;
         consumption;
         lending;
         captures;
         errors;
         domains;
         transformations;
         placement;
         scans;
         group ~tags:[ "slow" ] "swept" [ values ~count:25 ~heavy:true ];
       ])
