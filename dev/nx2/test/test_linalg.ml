(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Linalg through Nx.Prim: the shapes and dtypes Nx_kernel.Spec.linalg states
   for each routine's results, its rule before any kernel, the routine and
   operands each constructor gives the kernels and the order their results come
   back in, a split batch, and a library that declines. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module P = Nx_kernel.Prog
module S = Nx_kernel.Spec

(* nx.cpu, whose routines record what they are given and store [k + 1] in every
   element of result [k], so that a result names its slot. *)
module Filling = struct
  include Nx_cpu

  let name = "nx.filling"
  let seen : (S.routine * int list list * int) list ref = ref []

  let linalg s ~dsts ops =
    let shape (A.Any a) = Array.to_list (L.shape (A.layout a)) in
    seen :=
      (S.routine s, Array.to_list (Array.map shape ops), Array.length dsts)
      :: !seen;
    Array.iteri
      (fun k (A.Any d) ->
        let dt = A.dtype d in
        ignore
          (Nx_cpu.apply0
             (Fill (P.bits dt (D.of_float dt (Float.of_int (k + 1)))))
             ~dst:d))
      dsts;
    A.Done
end

let m = Nx_support.memory

module One = (val Nx.devices ~kernels:(module Filling) [ m 0 ])
module Two = (val Nx.devices ~kernels:(module Filling) [ m 0; m 1 ])

let read x = A.to_array (Option.get (Nx.Repr.array (Nx.place Nx.Host.on x)))

let zeros dt s =
  Nx.place One.on (Nx.Repr.of_array Nx.Host.v (A.create Rig.host dt s))

let linalg l = Nx.Prim.eval ~by:"t" (Linalg l)

(* Each result's dtype and shape, as Nx.Prim.results gives its forms. *)
type ('v, 's, 'd) Nx.Prim.payload += Slot : ('v, 's, 'd) Nx.Prim.payload

let forms l =
  Nx.Prim.interpret ~name:"t.forms" Values
    (fun _ ~by:_ _ -> fail "t.forms received an operation")
    (fun i ->
      let out = ref [] in
      ignore
        (Nx.Prim.results ~by:"t"
           (fun k (f : _ Nx.Prim.form) ->
             out := (k, D.name f.dtype, L.shape f.layout) :: !out;
             Nx.Prim.traced i f Slot)
           (Linalg l));
      List.rev !out)

let pp_form ppf (k, dt, s) =
  Format.fprintf ppf "%d: %s [%s]" k dt
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))

let form = Testable.make ~pp:pp_form ~equal:( = )

let test_forms () =
  let f32 = zeros D.Float32 and c64 = zeros D.Complex64 in
  let check msg want l = equal ~msg (list form) want (forms l) in
  check "cholesky"
    [ (0, "float32", [| 2; 3; 3 |]) ]
    (Cholesky { triangle = Lower; a = f32 [| 2; 3; 3 |] });
  check "lu"
    [
      (0, "float32", [| 2; 3; 4 |]);
      (1, "int64", [| 2; 3 |]);
      (2, "int64", [| 2; 3 |]);
    ]
    (Lu (f32 [| 2; 3; 4 |]));
  check "qr reduced"
    [ (0, "float32", [| 5; 3 |]); (1, "float32", [| 3; 3 |]) ]
    (Qr { factors = Reduced; a = f32 [| 5; 3 |] });
  check "qr complete"
    [ (0, "float32", [| 5; 5 |]); (1, "float32", [| 5; 3 |]) ]
    (Qr { factors = Complete; a = f32 [| 5; 3 |] });
  check "svd reduced"
    [
      (0, "float32", [| 3; 3 |]);
      (1, "float32", [| 3 |]);
      (2, "float32", [| 3; 5 |]);
    ]
    (Svd { factors = Reduced; a = f32 [| 3; 5 |] });
  check "svd complete"
    [
      (0, "float32", [| 3; 3 |]);
      (1, "float32", [| 3 |]);
      (2, "float32", [| 5; 5 |]);
    ]
    (Svd { factors = Complete; a = f32 [| 3; 5 |] });
  check "singular values"
    [ (0, "float32", [| 3 |]) ]
    (Svd_values (f32 [| 3; 5 |]));
  check "eigh"
    [ (0, "complex64", [| 4 |]); (1, "complex64", [| 4; 4 |]) ]
    (Eigh (c64 [| 4; 4 |]));
  check "eigh's values"
    [ (0, "float32", [| 4 |]) ]
    (Eigh_values (f32 [| 4; 4 |]));
  check "eig"
    [ (0, "complex64", [| 4 |]); (1, "complex64", [| 4; 4 |]) ]
    (Eig (c64 [| 4; 4 |]));
  check "eig's values"
    [ (0, "complex64", [| 4 |]) ]
    (Eig_values (c64 [| 4; 4 |]));
  check "solve"
    [ (0, "float32", [| 2; 3; 4 |]) ]
    (Solve_triangular
       {
         triangle = Upper;
         transpose = true;
         unit_diagonal = false;
         a = f32 [| 2; 3; 3 |];
         b = f32 [| 2; 3; 4 |];
       })

let test_rules () =
  let raises f = raises_match (Exn.invalid_arg ~substring:"t: ") f in
  Filling.seen := [];
  raises (fun () ->
      linalg (Cholesky { triangle = Lower; a = zeros D.Float32 [| 2; 3 |] }));
  raises (fun () -> linalg (Lu (zeros D.Int32 [| 2; 2 |])));
  raises (fun () -> linalg (Eigh (zeros D.Float16 [| 2; 2 |])));
  raises (fun () -> linalg (Svd_values (zeros D.Float64 [| 3 |])));
  raises (fun () ->
      linalg
        (Solve_triangular
           {
             triangle = Lower;
             transpose = false;
             unit_diagonal = false;
             a = zeros D.Float64 [| 3; 3 |];
             b = zeros D.Float64 [| 2; 4 |];
           }));
  equal ~msg:"kernel calls" int 0 (List.length !Filling.seen)

let test_dispatch () =
  Filling.seen := [];
  let a = zeros D.Float64 [| 2; 3; 3 |] in
  let lu, pivots, perm = linalg (Lu a) in
  equal ~msg:"lu" (array float_exact) (Array.make 18 1.) (read lu);
  equal ~msg:"pivots" (array int64) (Array.make 6 2L) (read pivots);
  equal ~msg:"perm" (array int64) (Array.make 6 3L) (read perm);
  let u, s, vh = linalg (Svd { factors = Reduced; a }) in
  equal ~msg:"u, s, vh in Spec's order"
    (list (array float_exact))
    [ Array.make 18 1.; Array.make 6 2.; Array.make 18 3. ]
    [ read u; read s; read vh ];
  let x =
    linalg
      (Solve_triangular
         {
           triangle = Upper;
           transpose = false;
           unit_diagonal = true;
           a;
           b = zeros D.Float64 [| 2; 3; 2 |];
         })
  in
  equal ~msg:"solve" (array float_exact) (Array.make 12 1.) (read x);
  let routine =
    Testable.make
      ~pp:(fun ppf _ -> Format.pp_print_string ppf "<routine>")
      ~equal:( = )
  in
  equal ~msg:"what the kernels received"
    (list (triple routine (list (list int)) int))
    [
      (S.Lu, [ [ 2; 3; 3 ] ], 3);
      (S.Svd { vectors = Some Reduced }, [ [ 2; 3; 3 ] ], 3);
      ( S.Solve_triangular
          { triangle = Upper; transpose = false; unit_diagonal = true },
        [ [ 2; 3; 3 ]; [ 2; 3; 2 ] ],
        1 );
    ]
    (List.rev !Filling.seen)

let test_split () =
  Filling.seen := [];
  let a =
    Nx.place (Two.split ~axis:0)
      (Nx.Repr.of_array Nx.Host.v (A.create Rig.host D.Float32 [| 2; 3; 3 |]))
  in
  let w, v = linalg (Eigh a) in
  equal ~msg:"each device its matrices"
    (list (list (list int)))
    [ [ [ 1; 3; 3 ] ]; [ [ 1; 3; 3 ] ] ]
    (List.map (fun (_, ops, _) -> ops) !Filling.seen);
  equal ~msg:"w" (array float_exact) (Array.make 6 1.) (read w);
  equal ~msg:"v" (array float_exact) (Array.make 18 2.) (read v)

let test_declined () =
  let a = Nx.Repr.of_array Nx.Host.v (A.create Rig.host D.Float32 [| 2; 2 |]) in
  match linalg (Cholesky { triangle = Lower; a }) with
  | _ -> fail "a declined factorisation computed"
  | exception Invalid_argument e ->
      List.iter
        (fun sub -> contains ~msg:sub ~sub e)
        [
          "t: ";
          "nx.cpu";
          "Cholesky";
          "float32";
          "not available yet";
          "Nx.place";
        ]

let () =
  exit
    (run "nx linalg"
       [
         group "routines"
           [
             test "each routine's results have Spec's shapes" test_forms;
             test "the rule raises before any kernel" test_rules;
             test "each routine reaches the kernels, its results in order"
               test_dispatch;
             test "a split batch factors on each device" test_split;
             test "a declined routine raises naming the move" test_declined;
           ];
       ])
