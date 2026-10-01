(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Interception: an interpreter receives each operation its extent performs,
   once, and none from elsewhere; its own operations reach the interpretation
   around it; what it raises, the performer sees; and no result depends on
   whether an interpreter is installed anywhere. *)

open Windtrap
module E = Nx.Op

let x = Nx.create Nx.float32 [| 3 |] [| 1.; -2.; 3. |]

(* An interpreter that records the name of each operation it receives. *)
let recording () =
  let names = ref [] in
  let run op =
    names := E.name op :: !names;
    E.eval op
  in
  ({ E.run; claims = (fun _ -> true) }, names)

let names = list string

let extent =
  group "extent"
    [
      test "each operation reaches the interpreter once" (fun () ->
          let i, seen = recording () in
          ignore (E.intercept i (fun () -> Nx.neg (Nx.add x x)));
          equal names [ "neg"; "add" ] !seen);
      test "operations outside the extent do not" (fun () ->
          let i, seen = recording () in
          ignore (E.intercept i (fun () -> Nx.add x x));
          ignore (Nx.neg x);
          equal names [ "add" ] !seen);
      test "the interpreter's operations reach the enclosing one" (fun () ->
          let outer, seen = recording () in
          let run op =
            ignore (Nx.neg x);
            E.eval op
          in
          ignore
            (E.intercept outer (fun () ->
                 E.intercept
                   { run; claims = (fun _ -> true) }
                   (fun () -> Nx.add x x)));
          equal names [ "add"; "neg" ] !seen);
      test "a domain spawned inside the extent is outside it" (fun () ->
          let i, seen = recording () in
          let y =
            E.intercept i (fun () ->
                Domain.join (Domain.spawn (fun () -> Nx.add x x)))
          in
          equal names [] !seen;
          equal (array float_exact) [| 2.; -4.; 6. |] (Nx.to_array y));
    ]

let asking =
  group "intercepted"
    [
      test "is false outside every extent" (fun () ->
          is_false (E.intercepted ()));
      test "is true inside one" (fun () ->
          is_true
            (E.intercept
               { run = E.eval; claims = (fun _ -> true) }
               E.intercepted));
      test "is false inside the only interpreter" (fun () ->
          let answer = ref true in
          let run op =
            answer := E.intercepted ();
            E.eval op
          in
          ignore
            (E.intercept
               { run; claims = (fun _ -> true) }
               (fun () -> Nx.add x x));
          is_false !answer);
    ]

let failing =
  group "exceptions"
    [
      test "one the interpreter raises reaches the performer" (fun () ->
          let run _ = raise Exit in
          is_true
            (E.intercept
               { run; claims = (fun _ -> true) }
               (fun () ->
                 match Nx.add x x with _ -> false | exception Exit -> true)));
      test "one the interpreter's claims raises reaches the performer"
        (fun () ->
          let claims _ = raise Exit in
          is_true
            (E.intercept { run = E.eval; claims } (fun () ->
                 match Nx.add x x with _ -> false | exception Exit -> true)));
      test "one that ends the extent ends the interception" (fun () ->
          (match
             E.intercept
               { run = E.eval; claims = (fun _ -> true) }
               (fun () -> raise Exit)
           with
          | () -> ()
          | exception Exit -> ());
          is_false (E.intercepted ()));
    ]

(* A program over many kinds of operation: elementwise, comparison, selection,
   reduction, scan, product, movement, sorting and conversion. *)
let program () =
  let m = Nx.reshape [| 3; 1 |] x in
  let p = Nx.matmul m (Nx.transpose m) in
  let s = Nx.where (Nx.less p (Nx.zeros_like p)) (Nx.neg p) (Nx.exp p) in
  let c = Nx.cumsum ~axis:1 s in
  let at = Nx.cast Nx.float32 (Nx.argmax x) in
  Nx.concatenate ~axis:0
    [
      Nx.sum ~axes:[ 0 ] c;
      fst (Nx.sort x);
      Nx.broadcast_to [| 3 |] (Nx.reshape [| 1 |] at);
    ]

let unobservable =
  group "the gate"
    [
      test "an identity interpreter changes no result" (fun () ->
          equal (array float_exact)
            (Nx.to_array (program ()))
            (Nx.to_array
               (E.intercept { run = E.eval; claims = (fun _ -> true) } program)));
      test "an interpreter on another domain changes no result" (fun () ->
          let entered = Atomic.make false and finished = Atomic.make false in
          let other =
            Domain.spawn (fun () ->
                E.intercept
                  { run = E.eval; claims = (fun _ -> true) }
                  (fun () ->
                    Atomic.set entered true;
                    while not (Atomic.get finished) do
                      Domain.cpu_relax ()
                    done))
          in
          while not (Atomic.get entered) do
            Domain.cpu_relax ()
          done;
          let y =
            Fun.protect ~finally:(fun () -> Atomic.set finished true) program
          in
          Domain.join other;
          equal (array float_exact) (Nx.to_array (program ())) (Nx.to_array y));
    ]

(* Claims *)

(* An interpreter that records the operations it receives, and claims only
   additions. *)
let claiming_adds () =
  let names = ref [] in
  let run op =
    names := E.name op :: !names;
    E.eval op
  in
  let claims : type r. r E.t -> bool = function
    | Binary (Add, _, _) -> true
    | _ -> false
  in
  ({ E.run; claims }, names)

let add_then_mul () = Nx.mul (Nx.add x x) x

let claiming =
  group "claims"
    [
      test "an operation the interpreter does not claim never reaches it"
        (fun () ->
          let i, seen = claiming_adds () in
          ignore (E.intercept i add_then_mul);
          equal names [ "add" ] !seen);
      test "an unclaimed operation reaches the enclosing interpreter once"
        (fun () ->
          let outer, outer_seen = recording () in
          let inner, _ = claiming_adds () in
          ignore (E.intercept outer (fun () -> E.intercept inner add_then_mul));
          equal names [ "add"; "mul" ] (List.rev !outer_seen));
      test "an operation no interpreter claims is computed" (fun () ->
          let i, _ = claiming_adds () in
          equal (array float_exact)
            (Nx.to_array (add_then_mul ()))
            (Nx.to_array (E.intercept i add_then_mul)));
    ]

(* Results' metadata *)

type case = C : ('a, 'b) Nx.t E.t -> case

let f32 shape =
  Nx.reshape shape
    (Nx.arange_f Nx.float32 0.
       (Float.of_int (Array.fold_left ( * ) 1 shape))
       1.)

let i32 shape values = Nx.create Nx.int32 shape values

(* An operation of each constructor whose result is one value, and each
   movement, over operands of shapes they change. *)
let cases () =
  let x = f32 [| 2; 3; 4 |] in
  let z = Nx.cast Nx.complex64 x in
  let unfold =
    E.Unfold
      {
        kernel_size = [| 2; 2 |];
        stride = [| 1; 1 |];
        dilation = [| 1; 1 |];
        padding = [| (0, 0); (0, 0) |];
        x;
      }
  in
  let windows = E.eval unfold in
  let lower =
    Nx.add (Nx.tril (f32 [| 3; 3 |])) (Nx.mul_s (Nx.eye Nx.float32 3) 10.)
  in
  let spd = Nx.matmul lower (Nx.transpose lower) in
  [
    C (Unary (Sin, x));
    C (Binary (Add, x, x));
    C (Compare (Less, x, x));
    C (Where (Nx.less x x, x, x));
    C (Reduce (Sum, [| 0; 2 |], x));
    C (Scan (Max, 1, x));
    C (Arg_reduce (Argmax, 2, x));
    C (Sort { descending = true; axis = 1; x });
    C (Argsort { descending = false; axis = 0; x });
    C (Pad ([| (1, 2); (0, 0); (3, 0) |], 0., x));
    C (Cat (1, [ x; f32 [| 2; 5; 4 |] ]));
    C (Cat (2, [ x; x; x ]));
    C (Convert (Cast, Nx.int32, x));
    C (Convert (Bitcast, Nx.int32, x));
    C (Threefry (i32 [| 2 |] [| 1l; 2l |], i32 [| 2 |] [| 3l; 4l |]));
    C (Gather (1, i32 [| 2; 2; 4 |] (Array.make 16 1l), x));
    C
      (Scatter
         {
           mode = `Add;
           unique = false;
           axis = 1;
           indices = i32 [| 2; 1; 4 |] (Array.make 8 2l);
           updates = f32 [| 2; 1; 4 |];
           into = x;
         });
    C (Update (x, i32 [| 3 |] [| 0l; 1l; 1l |], f32 [| 2; 2; 3 |]));
    C unfold;
    C
      (Fold
         {
           output_size = [| 3; 4 |];
           kernel_size = [| 2; 2 |];
           stride = [| 1; 1 |];
           dilation = [| 1; 1 |];
           padding = [| (0, 0); (0, 0) |];
           x = windows;
         });
    C (Matmul (x, f32 [| 4; 5 |]));
    C (Matmul (f32 [| 3; 4 |], f32 [| 5; 4; 2 |]));
    C (Fft { inverse = true; axes = [| 0; 2 |]; x = z });
    C (Rfft { dtype = Nx.complex64; axes = [| 2 |]; x });
    C (Irfft { dtype = Nx.float32; axes = [| 2 |]; s = None; x = z });
    C
      (Irfft
         { dtype = Nx.float32; axes = [| 1; 2 |]; s = Some [| 3; 5 |]; x = z });
    C (Contiguous (Nx.transpose x));
    C (Cholesky { upper = true; x = spd });
    C
      (Solve_triangular
         {
           upper = false;
           transpose = false;
           unit_diag = false;
           a = lower;
           b = f32 [| 3; 2 |];
         });
    C (Move (x, Reshape [| 6; 4 |]));
    C (Move (f32 [| 2; 1; 4 |], Expand [| 2; 5; 4 |]));
    C (Move (x, Permute [| 2; 0; 1 |]));
    C (Move (x, Shrink [| (0, 1); (1, 3); (0, 4) |]));
    C (Move (x, Flip [| true; false; true |]));
    C (Move (x, Window { axis = 2; size = 2; step = 1 }));
    C (Place (Nx.Placement.device Nx_test.Devices.d1, x));
  ]

let describing =
  group "Op.shape and Op.dtype"
    [
      test "are the shape and dtype of each operation's result" (fun () ->
          List.iter
            (fun (C op) ->
              let msg = E.name op in
              let r = E.eval op in
              equal ~msg (array int) (Nx.shape r) (E.shape op);
              equal ~msg string
                (Nx_dtype.to_string (Nx.dtype r))
                (Nx_dtype.to_string (E.dtype op)))
            (cases ()));
      test "refuse a concatenation of no value" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              E.shape (Cat (0, ([] : (float, Nx.float32_elt) Nx.t list))));
          raises_match Exn.invalid_arg (fun () ->
              E.dtype (Cat (0, ([] : (float, Nx.float32_elt) Nx.t list)))));
    ]

let () =
  exit
    (run "nx interception"
       [ extent; asking; failing; unobservable; claiming; describing ])
