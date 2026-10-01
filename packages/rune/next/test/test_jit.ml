(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Compiled calls on the host: a call computes eager's values, traces once per
   key, reads views in place, consumes and lends storage, refuses a consumed
   leaf another reaches before any work, and composes with the
   transformations. *)

open Windtrap
open Nx_test
module Rune = Rune_next.Rune

let floats = tensor float_exact
let x () = Nx.create Nx.float32 [| 4 |] [| 1.; -2.; 3.; 0.5 |]
let poly x = Nx.add (Nx.mul x x) x

(* [counted f] is [f] and the number of times it ran. *)
let counted f =
  let n = ref 0 in
  ( (fun x ->
      incr n;
      f x),
    n )

let address t = List.hd (Witness.addresses t)
let consumes = Nx.Ptree.(consumes tensor @@ returns tensor)

let calls =
  group "calls"
    [
      test "a compiled function computes eager's values" (fun () ->
          equal floats (poly (x ())) (Rune.jit' poly (x ())));
      test "a call with the key of an earlier one replays its program"
        (fun () ->
          let f, traces = counted poly in
          let g = Rune.jit' f in
          ignore (g (x ()));
          let y = Nx.create Nx.float32 [| 4 |] [| 2.; 0.; -1.; 4. |] in
          equal floats (poly y) (g y);
          equal int 1 !traces);
      test "a call with another shape traces again" (fun () ->
          let f, traces = counted poly in
          let g = Rune.jit' f in
          ignore (g (x ()));
          let y = Nx.ones Nx.float32 [| 2; 3 |] in
          equal floats (poly y) (g y);
          equal int 2 !traces);
      test "a strided argument is read where it lies" (fun () ->
          let a =
            Nx.transpose
              (Nx.reshape [| 2; 3 |] (Nx.arange_f Nx.float32 0. 6. 1.))
          in
          equal floats (poly a) (Rune.jit' poly a));
      test "a captured tensor is a constant of the program" (fun () ->
          let w = Nx.create Nx.float32 [| 4 |] [| 3.; 1.; 4.; 1. |] in
          let g = Rune.jit' (fun x -> Nx.mul x w) in
          equal floats (Nx.mul (x ()) w) (g (x ())));
      test "a result that returns its argument is a copy" (fun () ->
          let a = x () in
          let y = Rune.jit' Fun.id a in
          equal floats a y;
          is_false (share_memory (storage a) (storage y)));
      test "float16 values compute at float32 and round once" (fun () ->
          let a = Nx.cast Nx.float16 (x ()) in
          equal floats
            (Nx.cast Nx.float32 (poly a))
            (Nx.cast Nx.float32 (Rune.jit' poly a)));
      test "a draw from a key the function captures raises Jit_error" (fun () ->
          raises_match
            (function Rune.Jit_error _ -> true | _ -> false)
            (fun () ->
              ignore
                (Rune.jit'
                   (fun x ->
                     Nx.add x
                       (Nx.Rng.with_key (Nx.Rng.key 42) (fun () ->
                            Nx.rand Nx.float32 [| 4 |])))
                   (x ()))));
      test "reading a traced value raises Jit_error" (fun () ->
          raises_match
            (function Rune.Jit_error _ -> true | _ -> false)
            (fun () ->
              ignore
                (Rune.jit'
                   (fun x -> if Nx.item [ 0 ] x > 0. then x else Nx.neg x)
                   (x ()))));
    ]

let consumption =
  group "consumption"
    [
      test "a consumed host argument lends its storage to the result" (fun () ->
          let a = x () in
          let before = address a in
          let y = Rune.jit consumes (fun a -> Nx.add_s a 1.) a in
          equal floats (Nx.add_s (x ()) 1.) y;
          equal nativeint before (address y));
      test "a consumed argument raises on use, naming its path" (fun () ->
          let a = x () in
          ignore (Rune.jit consumes (fun a -> Nx.add_s a 1.) a);
          raises_match (Exn.invalid_arg ~substring:"consumed at 0") (fun () ->
              ignore (Nx.to_array a)));
      test "a borrowed consumed argument is copied, and still consumed"
        (fun () ->
          let ba =
            Bigarray.Array1.of_array Bigarray.float32 Bigarray.c_layout
              [| 1.; 2.; 3.; 4. |]
          in
          let a = Nx.of_bigarray (Bigarray.genarray_of_array1 ba) in
          let before = address a in
          let y = Rune.jit consumes (fun a -> Nx.mul_s a 2.) a in
          equal floats (Nx.create Nx.float32 [| 4 |] [| 2.; 4.; 6.; 8. |]) y;
          is_false (Nativeint.equal before (address y));
          raises_invalid_arg (fun () -> Nx.to_array a));
      test "a consumed leaf that another leaf reaches raises before any work"
        (fun () ->
          let a = x () in
          let g =
            Rune.jit
              Nx.Ptree.(consumes tensor @@ tensor @-> returns tensor)
              Nx.add
          in
          raises_match
            (Exn.invalid_arg
               ~substring:"0 is consumed and 1 reaches its storage") (fun () ->
              ignore (g a a));
          equal floats (x ()) a);
      test "a consumed slice raises before any work" (fun () ->
          let a = Nx.slice [ R (0, 2) ] (x ()) in
          raises_match
            (Exn.invalid_arg ~substring:"does not cover its whole storage")
            (fun () -> ignore (Rune.jit consumes Nx.neg a));
          equal int 2 (Array.length (Nx.to_array a)));
      (* A result that reads its consumed argument elsewhere than at its own
         index takes fresh storage: writing it over the argument would change
         what its own kernel still reads. *)
      cases ~name:fst
        "a result that reads its consumed argument at other indices is right"
        [
          ( "left rotation",
            fun a ->
              Nx.concatenate ~axis:0
                [ Nx.slice [ R (1, 8) ] a; Nx.slice [ R (0, 1) ] a ] );
          ( "right rotation",
            fun a ->
              Nx.concatenate ~axis:0
                [ Nx.slice [ R (7, 8) ] a; Nx.slice [ R (0, 7) ] a ] );
          ("flip", fun a -> Nx.add_s (Nx.flip a) 1.);
        ]
        (fun (_, f) ->
          let eight () = Nx.arange_f Nx.float32 1. 9. 1. in
          equal floats (f (eight ())) (Rune.jit consumes f (eight ())));
    ]

let compositions =
  group "compositions"
    [
      test "under a transformation a compiled function runs its function"
        (fun () ->
          let f x = Nx.sum (poly x) in
          equal floats (Rune.grad' f (x ())) (Rune.grad' (Rune.jit' f) (x ())));
      test "a gradient computes inside a compiled function" (fun () ->
          let f x = Nx.sum (poly x) in
          equal floats (Rune.grad' f (x ())) (Rune.jit' (Rune.grad' f) (x ())));
      test "a compiled function called inside its own trace traces through"
        (fun () ->
          let inner = Rune.jit' Nx.neg in
          let g = Rune.jit' (fun x -> inner (inner x)) in
          equal floats (x ()) (g (x ())));
      test "a scan no stager takes folds inside the trace" (fun () ->
          let f xs =
            fst
              (Rune.scan'
                 ~f:(fun c x -> (Nx.add c x, c))
                 ~init:(Nx.zeros Nx.float32 [||]) xs)
          in
          equal floats (f (x ())) (Rune.jit' f (x ())));
    ]

let domains =
  group "domains"
    [
      test "two domains replay one program" (fun () ->
          let g = Rune.jit' poly in
          ignore (g (x ()));
          let run k =
            let y = Nx.full Nx.float32 [| 4 |] (Float.of_int k) in
            Nx.to_array (g y)
          in
          let ds =
            List.init 2 (fun k -> Domain.spawn (fun () -> run (k + 1)))
          in
          List.iteri
            (fun k d ->
              let v = Float.of_int (k + 1) in
              equal (array float_exact)
                (Array.make 4 ((v *. v) +. v))
                (Domain.join d))
            ds);
    ]

(* A device over the host's memory, whose programs are the host's. *)
let device =
  Nx.Device.of_runtime
    (Nx_device.Driver.device ~name:"R1" ~arch:"test" ~budget:max_int
       (Host_visible
          { memory = Nx_device.Driver.host_memory; mapping = Some Identity }))

let on_device = Nx.Placement.device device
let placed t = Nx.place on_device t

let bytes_in () =
  Nx_device.Stats.bytes_in (Nx_device.stats (Nx.Device.runtime device))

let devices =
  group "devices"
    [
      test "a call runs where its arguments lie, and leaves its results there"
        (fun () ->
          let y = Rune.jit' poly (placed (x ())) in
          is_true (Nx.Placement.equal on_device (Nx.placement y));
          equal floats (poly (x ())) (Nx.place Nx.Placement.host y));
      test "a capture placed where the call computes is bound, not uploaded"
        (fun () ->
          let w = placed (Nx.create Nx.float32 [| 4 |] [| 3.; 1.; 4.; 1. |]) in
          let g = Rune.jit' (fun x -> Nx.mul x w) in
          let a = placed (x ()) in
          ignore (g a);
          let before = bytes_in () in
          let y = g a in
          equal int before (bytes_in ());
          equal floats
            (Nx.mul (x ()) (Nx.place Nx.Placement.host w))
            (Nx.place Nx.Placement.host y));
      test "a consumed placed argument lends its storage" (fun () ->
          let a = placed (x ()) in
          let before = Witness.addresses a in
          let y = Rune.jit consumes (fun a -> Nx.add_s a 1.) a in
          equal (list nativeint) before (Witness.addresses y);
          equal floats (Nx.add_s (x ()) 1.) (Nx.place Nx.Placement.host y));
    ]

let () =
  exit
    (run "Rune_next.Jit" [ calls; consumption; compositions; domains; devices ])
