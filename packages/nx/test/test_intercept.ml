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
let i64 shape values = Nx.create Nx.int64 shape values

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
    C (Convert (Bitcast, Nx.uint16, x));
    C (Convert (Bitcast, Nx.float64, Nx.reshape [| 2; 3; 2; 2 |] x));
    C (Threefry (i32 [| 2 |] [| 1l; 2l |], i32 [| 2 |] [| 3l; 4l |]));
    C (Gather (1, i64 [| 2; 2; 4 |] (Array.make 16 1L), x));
    C
      (Scatter
         {
           mode = `Add;
           unique = false;
           axis = 1;
           indices = i64 [| 2; 1; 4 |] (Array.make 8 2L);
           updates = f32 [| 2; 1; 4 |];
           into = x;
         });
    C
      (Scatter
         {
           mode = `Max;
           unique = true;
           axis = 0;
           indices = i64 [| 1; 3; 4 |] (Array.make 12 1L);
           updates = f32 [| 1; 3; 4 |];
           into = x;
         });
    C (Update (x, i64 [| 3 |] [| 0L; 1L; 1L |], f32 [| 2; 2; 3 |]));
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
    C (Place (Nx.Placement.on Nx_test.Devices.d1, x));
  ]

(* Operations that change their operands' shapes, drawn over shapes with empty
   axes and over scalars wherever the operation takes them. *)

let dim = Gen.int_range 0 4
let shape r = Gen.array ~size:(Gen.constant r) dim
let ranked lo hi = Gen.bind (Gen.int_range lo hi) shape
let zeros s = Nx.zeros Nx.float32 s
let complex s = Nx.zeros Nx.complex64 s

(* Distinct axes of a rank [r] value, in any order, at least [least] of them. *)
let axes ?(least = 0) r =
  let open Gen in
  let* chosen =
    such_that
      (fun l -> List.length l >= least)
      (subsequence (List.init r Fun.id))
  in
  map Array.of_list (permutation chosen)

(* A window over [n] spatial axes: its kernel, stride, dilation and padding, and
   spatial sizes that hold at least one window. *)
let window n =
  let open Gen in
  let+ kernel = array ~size:(constant n) (int_range 1 3)
  and+ stride = array ~size:(constant n) (int_range 1 2)
  and+ dilation = array ~size:(constant n) (int_range 1 2)
  and+ padding = array ~size:(constant n) (pair (int_range 0 1) (int_range 0 1))
  and+ extra = array ~size:(constant n) (int_range 0 3) in
  let sizes =
    Array.init n (fun i ->
        let before, after = padding.(i) in
        Int.max 0
          ((dilation.(i) * (kernel.(i) - 1)) + 1 - before - after + extra.(i)))
  in
  (kernel, stride, dilation, padding, sizes)

(* Every generator of [gs], drawn in order. *)
let rec all = function
  | [] -> Gen.constant []
  | g :: gs ->
      let open Gen in
      let+ x = g and+ xs = all gs in
      x :: xs

let generated =
  let open Gen in
  let pad =
    let* s = ranked 0 3 in
    let+ padding =
      array
        ~size:(constant (Array.length s))
        (pair (int_range 0 2) (int_range 0 2))
    in
    C (Pad (padding, 0., zeros s))
  in
  let cat =
    let* s = ranked 1 3 in
    let* axis = int_range 0 (Array.length s - 1) in
    let+ sizes = list ~size:(int_range 1 3) dim in
    C
      (Cat
         ( axis,
           List.map
             (fun n ->
               zeros (Array.mapi (fun i d -> if i = axis then n else d) s))
             sizes ))
  in
  let reduce =
    let* s = ranked 0 3 in
    let+ axes = axes (Array.length s) in
    C (Reduce (Sum, axes, zeros s))
  in
  let arg_reduce =
    let* s = ranked 1 3 in
    let s = Array.map (Int.max 1) s in
    let+ axis = int_range 0 (Array.length s - 1) in
    C (Arg_reduce (Argmax, axis, zeros s))
  in
  let matmul =
    let* batch = ranked 0 2 in
    let* m, k, n = triple dim dim dim in
    let operand_batch =
      let* kept = int_range 0 (Array.length batch) in
      let+ ones = array ~size:(constant kept) bool in
      Array.mapi
        (fun i one -> if one then 1 else batch.(Array.length batch - kept + i))
        ones
    in
    let+ ba = operand_batch and+ bb = operand_batch in
    C
      (Matmul
         (zeros (Array.append ba [| m; k |]), zeros (Array.append bb [| k; n |])))
  in
  let unfold =
    let* lead = ranked 0 2 in
    let+ kernel_size, stride, dilation, padding, sizes =
      bind (int_range 1 2) window
    in
    C
      (Unfold
         {
           kernel_size;
           stride;
           dilation;
           padding;
           x = zeros (Array.append lead sizes);
         })
  in
  let fold =
    let* lead = ranked 0 1 in
    let+ kernel_size, stride, dilation, padding, output_size =
      bind (int_range 1 2) window
    in
    let windows =
      Array.mapi
        (fun i d ->
          let before, after = padding.(i) in
          (d + before + after - ((dilation.(i) * (kernel_size.(i) - 1)) + 1))
          / stride.(i)
          + 1)
        output_size
    in
    let product = Array.fold_left ( * ) 1 in
    C
      (Fold
         {
           output_size;
           kernel_size;
           stride;
           dilation;
           padding;
           x =
             zeros
               (Array.append lead [| product kernel_size; product windows |]);
         })
  in
  let rfft =
    let* s = ranked 1 3 in
    let s = Array.map (Int.max 1) s in
    let+ axes = axes ~least:1 (Array.length s) in
    C (Rfft { dtype = Nx.complex64; axes; x = zeros s })
  in
  let irfft =
    let* s = ranked 1 3 in
    let s = Array.map (Int.max 2) s in
    let* axes = axes ~least:1 (Array.length s) in
    let+ last = option (int_range 1 6) in
    let sizes =
      Option.map
        (fun n ->
          Array.mapi
            (fun i a -> if i = Array.length axes - 1 then n else s.(a))
            axes)
        last
    in
    C (Irfft { dtype = Nx.float32; axes; s = sizes; x = complex s })
  in
  let move =
    let* s = ranked 0 3 in
    let r = Array.length s in
    let+ m =
      one_of
        [
          constant (E.Reshape [| Array.fold_left ( * ) 1 s |]);
          map
            (fun order -> E.Permute (Array.of_list order))
            (permutation (List.init r Fun.id));
          map
            (fun limits -> E.Shrink (Array.of_list limits))
            (all
               (List.map
                  (fun d ->
                    let* lo = int_range 0 d in
                    let+ hi = int_range lo d in
                    (lo, hi))
                  (Array.to_list s)));
        ]
    in
    C (Move (zeros s, m))
  in
  let window_move =
    let* s = ranked 1 3 in
    let s = Array.map (Int.max 1) s in
    let* axis = int_range 0 (Array.length s - 1) in
    let* size = int_range 1 s.(axis) in
    let+ step = int_range 1 2 in
    C (Move (zeros s, Window { axis; size; step }))
  in
  with_pp
    (fun ppf (C op) -> E.pp ppf op)
    (one_of
       [
         pad;
         cat;
         reduce;
         arg_reduce;
         matmul;
         unfold;
         fold;
         rfft;
         irfft;
         move;
         window_move;
       ])

let describing =
  group "Op.shape and Op.dtype"
    [
      test "are the shape and dtype of each operation's result" (fun () ->
          List.iter (fun (C op) -> Nx_test.described op (E.eval op)) (cases ()));
      prop "are the shape and dtype of each drawn operation's result" generated
        (fun (C op) -> Nx_test.described op (E.eval op));
      test "refuse a concatenation of no value" (fun () ->
          raises_match Exn.invalid_arg (fun () ->
              E.shape (Cat (0, ([] : (float, Nx.float32_elt) Nx.t list))));
          raises_match Exn.invalid_arg (fun () ->
              E.dtype (Cat (0, ([] : (float, Nx.float32_elt) Nx.t list)))));
    ]

(* Reads *)

(* An interpreter that claims only reads and records the name each carries. *)
let naming () =
  let seen = ref [] in
  let run : type r. r E.t -> r =
   fun op ->
    (match op with Read { by; _ } -> seen := by :: !seen | _ -> ());
    E.eval op
  in
  let claims : type r. r E.t -> bool = function Read _ -> true | _ -> false in
  ({ E.run; claims }, seen)

let mask = Nx.create Nx.bool [| 3 |] [| true; false; true |]
let square = Nx.create Nx.float64 [| 2; 2 |] [| 2.; 1.; 1.; 3. |]
let wide = Nx.create Nx.float64 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 7. |]

let reads =
  let discard f () = ignore (f ()) in
  group "reads"
    [
      Windtrap.cases ~name:fst "a read names the function the program called"
        [
          ("Nx.item", discard (fun () -> Nx.item [ 1 ] x));
          ("Nx.to_array", discard (fun () -> Nx.to_array x));
          ("Nx.to_bigarray", discard (fun () -> Nx.to_bigarray x));
          ("Nx.to_string", discard (fun () -> Nx.to_string x));
          ("Nx.pp", fun () -> Nx.pp Format.str_formatter x);
          ("Nx.fold_item", discard (fun () -> Nx.fold_item ( +. ) 0. x));
          ("Nx.map_item", discard (fun () -> Nx.map_item Fun.id x));
          ("Nx.iter_item", fun () -> Nx.iter_item ignore x);
          ("Nx.positions", discard (fun () -> Nx.positions mask));
          ("Nx.unique", discard (fun () -> Nx.unique x));
          ("Nx.compress", discard (fun () -> Nx.compress ~condition:mask x));
          ("Nx.extract", discard (fun () -> Nx.extract ~condition:mask x));
          ("Nx.nonzero", discard (fun () -> Nx.nonzero x));
          ("Nx.argwhere", discard (fun () -> Nx.argwhere x));
          ("Nx.slice", discard (fun () -> Nx.slice [ M mask ] x));
          ( "Nx.set",
            discard (fun () ->
                Nx.set [ M mask ] (Nx.zeros Nx.float32 [| 2 |]) x) );
          ("Nx.matrix_rank", discard (fun () -> Nx.matrix_rank square));
          ("Nx.cond", discard (fun () -> Nx.cond square));
          ("Nx.pinv", discard (fun () -> Nx.pinv square));
          ( "Nx.lstsq",
            discard (fun () -> Nx.lstsq wide (Nx.ones Nx.float64 [| 2 |])) );
          ( "Nx.tensorsolve",
            discard (fun () ->
                Nx.tensorsolve
                  (Nx.zeros Nx.float64 [| 2; 2 |])
                  (Nx.ones Nx.float64 [| 2 |])) );
          ( "Nx.tensorinv",
            discard (fun () -> Nx.tensorinv (Nx.zeros Nx.float64 [| 2; 2 |])) );
        ]
        (fun (expected, f) ->
          let i, seen = naming () in
          E.intercept i f;
          equal names [ expected ] (List.sort_uniq String.compare !seen));
      Windtrap.cases ~name:fst
        "a function whose length depends on values reads it once"
        [
          ("positions of a mask", discard (fun () -> Nx.positions mask));
          ( "positions of counts",
            discard (fun () ->
                Nx.positions (Nx.create Nx.int32 [| 3 |] [| 2l; 0l; 1l |])) );
          ("compress", discard (fun () -> Nx.compress ~condition:mask x));
          ( "compress along an axis",
            discard (fun () ->
                Nx.compress ~axis:1
                  ~condition:(Nx.create Nx.bool [| 3 |] [| false; true; true |])
                  wide) );
          ("extract", discard (fun () -> Nx.extract ~condition:mask x));
          ("nonzero", discard (fun () -> Nx.nonzero wide));
          ("argwhere", discard (fun () -> Nx.argwhere wide));
          ("unique", discard (fun () -> Nx.unique wide));
        ]
        (fun (_, f) ->
          let i, seen = naming () in
          E.intercept i f;
          equal int 1 (List.length !seen));
      test "positions of nothing and nonzero of a scalar read nothing"
        (fun () ->
          let i, seen = naming () in
          E.intercept i (fun () ->
              ignore (Nx.positions (Nx.zeros Nx.bool [| 0 |]));
              ignore (Nx.positions (Nx.zeros Nx.int32 [| 0 |]));
              ignore (Nx.nonzero (Nx.scalar Nx.float32 1.));
              ignore (Nx.unique (Nx.zeros Nx.int32 [| 0 |])));
          equal names [] !seen);
    ]

let () =
  exit
    (run "nx interception"
       [ extent; asking; failing; unobservable; claiming; describing; reads ])
