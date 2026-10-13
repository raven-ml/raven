(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Linalg through Nx.Prim: the shapes and dtypes Nx_kernel.Shape states for
   each routine's results, its rule before any kernel, the routine and
   operands each constructor gives the kernels and the order their results come
   back in, a split batch, and a library that declines. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout
module P = Nx_kernel.Prog
(* What a routine's kernel was given beyond its arrays. *)
type routine =
  | Cholesky of Nx_kernel.triangle
  | Lu
  | Qr
  | Svd
  | Eigh
  | Eig
  | Solve_triangular of {
      triangle : Nx_kernel.triangle;
      adjoint : bool;
      unit_diagonal : bool;
    }

(* nx.cpu, whose routines record what they are given and store [k + 1] in every
   element of result [k], so that a result names its slot. A routine's
   destinations are its values, then its vectors where it has them. *)
module Filling = struct
  include Nx_cpu

  let name = "nx.filling"
  let seen : (routine * int list list * int list list) list ref = ref []

  let filled r ops dsts =
    let shape (A.Any a) = Array.to_list (L.shape (A.layout a)) in
    seen := (r, List.map shape ops, List.map shape dsts) :: !seen;
    List.iteri
      (fun k (A.Any d) ->
        let dt = A.dtype d in
        ignore
          (Nx_cpu.apply0
             (Fill (P.bits dt (D.of_float dt (Float.of_int (k + 1)))))
             ~dst:d))
      dsts;
    A.Done

  let cholesky t ~dst a = filled (Cholesky t) [ A.Any a ] [ A.Any dst ]

  let lu ~lu ~pivots ~perm a =
    filled Lu [ A.Any a ] [ A.Any lu; A.Any pivots; A.Any perm ]

  let qr ~q ~r a = filled Qr [ A.Any a ] [ A.Any q; A.Any r ]

  let svd ~s ?vectors a =
    let vectors =
      match vectors with
      | None -> []
      | Some (~u, ~vh) -> [ A.Any u; A.Any vh ]
    in
    filled Svd [ A.Any a ] (A.Any s :: vectors)

  let eigen r ~w v a =
    let v = match v with None -> [] | Some v -> [ A.Any v ] in
    filled r [ A.Any a ] (A.Any w :: v)

  let eigh ~w ?v a = eigen Eigh ~w v a
  let eig ~w ?v a = eigen Eig ~w v a

  let solve_triangular triangle ~adjoint ~unit_diagonal ~dst a b =
    filled
      (Solve_triangular { triangle; adjoint; unit_diagonal })
      [ A.Any a; A.Any b ] [ A.Any dst ]
end

let m = Nx_support.memory

(* nx.cpu without eig. *)
module Declining = struct
  include Nx_cpu

  let name = "nx.declining"
  let eig ~w:_ ?v:_ _ = A.Declined
end

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
      (0, "float32", [| 3 |]);
      (1, "float32", [| 3; 3 |]);
      (2, "float32", [| 3; 5 |]);
    ]
    (Svd { vectors = Vectors Reduced; a = f32 [| 3; 5 |] });
  check "svd complete"
    [
      (0, "float32", [| 3 |]);
      (1, "float32", [| 3; 3 |]);
      (2, "float32", [| 5; 5 |]);
    ]
    (Svd { vectors = Vectors Complete; a = f32 [| 3; 5 |] });
  check "singular values"
    [ (0, "float32", [| 3 |]) ]
    (Svd { vectors = No_vectors; a = f32 [| 3; 5 |] });
  check "eigh"
    [ (0, "complex64", [| 4 |]); (1, "complex64", [| 4; 4 |]) ]
    (Eigh { vectors = Vectors (); a = c64 [| 4; 4 |] });
  check "eigh's values"
    [ (0, "float32", [| 4 |]) ]
    (Eigh { vectors = No_vectors; a = f32 [| 4; 4 |] });
  check "eig"
    [ (0, "complex64", [| 4 |]); (1, "complex64", [| 4; 4 |]) ]
    (Eig { vectors = Vectors (); dtype = Complex64; a = c64 [| 4; 4 |] });
  check "eig of a real matrix"
    [ (0, "complex128", [| 4 |]); (1, "complex128", [| 4; 4 |]) ]
    (Eig
       {
         vectors = Vectors ();
         dtype = Complex128;
         a = zeros D.Float64 [| 4; 4 |];
       });
  check "eig's values"
    [ (0, "complex64", [| 4 |]) ]
    (Eig { vectors = No_vectors; dtype = Complex64; a = f32 [| 4; 4 |] });
  check "solve"
    [ (0, "float32", [| 2; 3; 4 |]) ]
    (Solve_triangular
       {
         triangle = Upper;
         adjoint = true;
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
  raises (fun () ->
      linalg (Eigh { vectors = Vectors (); a = zeros D.Float16 [| 2; 2 |] }));
  raises (fun () ->
      linalg (Svd { vectors = No_vectors; a = zeros D.Float64 [| 3 |] }));
  raises (fun () ->
      linalg
        (Eig
           {
             vectors = No_vectors;
             dtype = Complex64;
             a = zeros D.Float64 [| 2; 2 |];
           }));
  raises (fun () ->
      linalg
        (Solve_triangular
           {
             triangle = Lower;
             adjoint = false;
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
  let s, (u, vh) = linalg (Svd { vectors = Vectors Reduced; a }) in
  equal ~msg:"s, u, vh in the constructor's order"
    (list (array float_exact))
    [ Array.make 6 1.; Array.make 18 2.; Array.make 18 3. ]
    [ read s; read u; read vh ];
  let tall = zeros D.Float64 [| 4; 2 |] in
  let q, r = linalg (Qr { factors = Complete; a = tall }) in
  equal ~msg:"q, r in the constructor's order"
    (list (array float_exact))
    [ Array.make 16 1.; Array.make 8 2. ]
    [ read q; read r ];
  ignore (linalg (Svd { vectors = No_vectors; a = tall }));
  let sq = zeros D.Complex64 [| 3; 3 |] in
  ignore (linalg (Eigh { vectors = Vectors (); a = sq }));
  ignore (linalg (Eigh { vectors = No_vectors; a = sq }));
  ignore (linalg (Eig { vectors = Vectors (); dtype = Complex64; a = sq }));
  ignore (linalg (Eig { vectors = No_vectors; dtype = Complex64; a = sq }));
  let x =
    linalg
      (Solve_triangular
         {
           triangle = Upper;
           adjoint = false;
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
    (list (triple routine (list (list int)) (list (list int))))
    [
      (Lu, [ [ 2; 3; 3 ] ], [ [ 2; 3; 3 ]; [ 2; 3 ]; [ 2; 3 ] ]);
      (Svd, [ [ 2; 3; 3 ] ], [ [ 2; 3 ]; [ 2; 3; 3 ]; [ 2; 3; 3 ] ]);
      (Qr, [ [ 4; 2 ] ], [ [ 4; 4 ]; [ 4; 2 ] ]);
      (Svd, [ [ 4; 2 ] ], [ [ 2 ] ]);
      (Eigh, [ [ 3; 3 ] ], [ [ 3 ]; [ 3; 3 ] ]);
      (Eigh, [ [ 3; 3 ] ], [ [ 3 ] ]);
      (Eig, [ [ 3; 3 ] ], [ [ 3 ]; [ 3; 3 ] ]);
      (Eig, [ [ 3; 3 ] ], [ [ 3 ] ]);
      ( Solve_triangular
          { triangle = Upper; adjoint = false; unit_diagonal = true },
        [ [ 2; 3; 3 ]; [ 2; 3; 2 ] ],
        [ [ 2; 3; 2 ] ] );
    ]
    (List.rev !Filling.seen)

let test_split () =
  Filling.seen := [];
  let a =
    Nx.place (Two.split ~axis:0)
      (Nx.Repr.of_array Nx.Host.v (A.create Rig.host D.Float32 [| 2; 3; 3 |]))
  in
  let w, v = linalg (Eigh { vectors = Vectors (); a }) in
  equal ~msg:"each device its matrices"
    (list (list (list int)))
    [ [ [ 1; 3; 3 ] ]; [ [ 1; 3; 3 ] ] ]
    (List.map (fun (_, ops, _) -> ops) !Filling.seen);
  equal ~msg:"w" (array float_exact) (Array.make 6 1.) (read w);
  equal ~msg:"v" (array float_exact) (Array.make 18 2.) (read v)

let test_declined () =
  let module Declines = (val Nx.devices ~kernels:(module Declining) [ m 0 ]) in
  let a =
    Nx.place Declines.on
      (Nx.Repr.of_array Nx.Host.v (A.create Rig.host D.Complex64 [| 2; 2 |]))
  in
  match linalg (Eig { vectors = No_vectors; dtype = D.Complex64; a }) with
  | _ -> fail "a declined factorisation computed"
  | exception Invalid_argument e ->
      List.iter
        (fun sub -> contains ~msg:sub ~sub e)
        [
          "t: ";
          "nx.declining";
          "Eig";
          "complex64";
          "not available yet";
          "Nx.place";
        ]

let () =
  exit
    (run "nx linalg"
       [
         group "routines"
           [
             test "each routine's results have Shape's shapes" test_forms;
             test "the rule raises before any kernel" test_rules;
             test "each routine reaches the kernels, its results in order"
               test_dispatch;
             test "a split batch factors on each device" test_split;
             test "a declined routine raises naming the move" test_declined;
           ];
       ])
