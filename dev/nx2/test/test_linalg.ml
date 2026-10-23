(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Linalg through Nx.Prim: the shapes and dtypes Nx_kernel.Shape states for
   each routine's results, its rule before any kernel, the routine and
   operands each constructor gives the kernels and the order their results come
   back in, a split batch, a library that declines, and nx's expansions. *)

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
        let bits = P.bits dt (D.of_float dt (Float.of_int (k + 1))) in
        ignore (Nx_kernels_support.map0 (module Nx_cpu) (Fill bits) ~dst:d))
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

(* nx.cpu without lu, svd, eigh and eig. *)
module Declining = struct
  include Nx_cpu

  let name = "nx.declining"
  let lu ~lu:_ ~pivots:_ ~perm:_ _ = A.Declined
  let svd ~s:_ ?vectors:_ _ = A.Declined
  let eigh ~w:_ ?v:_ _ = A.Declined
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
          "Nx.place";
        ]

(* The functions, through Nx: each law against an identity it states, at the
   four dtypes, over batches with empty and one-element axes. A law holds within
   [tolerance dt] of its operands' scale, where the matrices are well
   conditioned. *)

type full = Full : ('v, 's) D.t -> full

let fulls = [ Full D.Float32; Full D.Float64; Full D.Complex64; Full D.Complex128 ]

let tolerance (type v s) (dt : (v, s) D.t) =
  match dt with D.Float32 | D.Complex64 -> 1e-4 | _ -> 1e-11

let value (type v s) (dt : (v, s) D.t) shape re im : (v, s, Nx.host) Nx.t =
  let f64 a = Nx.create Nx.float64 shape a in
  match dt with
  | D.Float32 -> Nx.cast dt (f64 re)
  | D.Float64 -> f64 re
  | D.Complex64 -> Nx.complex dt ~re:(f64 re) ~im:(f64 im)
  | D.Complex128 -> Nx.complex dt ~re:(f64 re) ~im:(f64 im)
  | _ -> invalid_arg "value"

(* [x]'s magnitudes at float64. *)
let magnitudes (type v s d) (x : (v, s, d) Nx.t) : float array =
  match D.kind (Nx.dtype x) with
  | D.Complex -> Nx.to_array (Nx.magnitude Nx.float64 x)
  | D.Float -> Nx.to_array (Nx.cast Nx.float64 (Nx.abs x))
  | D.Signed | D.Unsigned | D.Boolean -> fail "magnitudes of a non-float"

(* The greatest magnitude, NaN where one is. *)
let worst x =
  Array.fold_left
    (fun w v -> if Float.is_nan v || Float.is_nan w then Float.nan else Float.max w v)
    0. (magnitudes x)

(* [d] below [than], failing on a NaN: [float_exact]'s order puts a NaN
   below every number. *)
let below ~msg ~than d =
  if Float.is_nan d then failf "%s: NaN, not below %h" msg than;
  less ~msg float_exact ~than d

(* [x] within [tolerance] of [y], relative to [y]'s scale and at least 1. *)
let near ?(scale = 1.) ?tol ~msg x y =
  let tol = Option.value tol ~default:(tolerance (Nx.dtype y)) in
  let bound = tol *. scale *. Float.max 1. (worst y) in
  below ~msg ~than:bound (worst (Nx.sub x y))

let all_nan ~msg x =
  equal ~msg bool true (Array.for_all Float.is_nan (magnitudes x))

let adj x = Nx.conjugate (Nx.matrix_transpose x)

let identity dt lead n = Nx.broadcast_to (Array.append lead [| n; n |]) (Nx.eye dt n)

(* [v] scaled along its last axis by [w]: [v · diag w]. *)
let columns_by v w = Nx.mul v (Nx.unsqueeze ~axes:[ -2 ] w)

type case = Case : ('v, 's) D.t * int array * float array * float array -> case

let pp_case ppf (Case (dt, s, _, _)) =
  Format.fprintf ppf "%a [%s]" D.pp dt
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))

(* Matrices of up to 5 rows and columns, square ones where [square], behind a
   batch of none, one or two. *)
let matrices ?(square = false) () =
  Gen.with_pp pp_case
    (let open Gen in
     let* (Full dt) = of_list fulls in
     let* lead = of_list [ [||]; [| 0 |]; [| 1 |]; [| 2 |] ] in
     let* m = int_range 0 5 in
     let* n = if square then constant m else int_range 0 5 in
     let s = Array.append lead [| m; n |] in
     let k = Array.fold_left ( * ) 1 s in
     let+ re = array ~size:(constant k) (float_range (-1.) 1.)
     and+ im = array ~size:(constant k) (float_range (-1.) 1.) in
     Case (dt, s, re, im))

let lead s = Array.sub s 0 (Array.length s - 2)

(* Demands that a law meet each shape the generator draws: wide, tall and
   square matrices (square ones alone where [square]), an empty one, and a
   batch of two. *)
let cover_shapes ?(square = false) s =
  let r = Array.length s in
  let m = s.(r - 2) and n = s.(r - 1) in
  if not square then begin
    cover "wide" (m < n);
    cover "tall" (m > n)
  end;
  cover "square" (m = n);
  cover "empty" (Array.exists (( = ) 0) s);
  cover "a batch of two" (r = 3 && s.(0) = 2)

(* [x] plus [c] on its main diagonal, in its dtype. *)
let shifted c x =
  let s = Nx.shape x in
  let r = Array.length s in
  let eye = Nx.eye ~m:s.(r - 1) Nx.float64 s.(r - 2) in
  Nx.add x (Nx.cast (Nx.dtype x) (Nx.mul_s eye c))

(* [g gᴴ + (n + 1) I]: Hermitian, positive definite and well conditioned. *)
let definite g =
  let n = (Nx.shape g).(Nx.ndim g - 1) in
  shifted (Float.of_int (n + 1)) (Nx.matmul g (adj g))

(* [g] with singular values of at least 1, at most 11, for entries of
   magnitude at most [√2] and up to 5 rows and columns. *)
let conditioned g = shifted 6. g

let test_cholesky =
  prop "cholesky: L Lᴴ and Uᴴ U are a, each from the triangle it names"
    (matrices ~square:true ())
    (fun (Case (dt, s, re, im)) ->
      cover_shapes ~square:true s;
      let a = definite (value dt s re im) in
      (* Noise strictly below the diagonal, which the upper triangle omits. *)
      let noise = Nx.tril ~k:(-1) (value dt s im re) in
      let l = Nx.cholesky (Nx.add a (Nx.matrix_transpose noise)) in
      near ~msg:"L Lᴴ" (Nx.matmul l (adj l)) a;
      equal ~msg:"above" float_exact 0. (worst (Nx.triu ~k:1 l));
      let u = Nx.cholesky ~triangle:Upper (Nx.add a noise) in
      near ~msg:"Uᴴ U" (Nx.matmul (adj u) u) a;
      equal ~msg:"below" float_exact 0. (worst (Nx.tril ~k:(-1) u)))

let test_cholesky_fails () =
  let a = Nx.create Nx.float64 [| 2; 2; 2 |] [| 2.; 1.; 1.; 2.; 1.; 2.; 2.; 1. |] in
  let l = Nx.cholesky a in
  near ~msg:"the definite one" (Nx.matmul (Nx.get [ 0 ] l) (adj (Nx.get [ 0 ] l))) (Nx.get [ 0 ] a);
  all_nan ~msg:"the indefinite one" (Nx.get [ 1 ] l)

let test_qr =
  prop "qr: q r is a, q's columns orthonormal, r upper" (matrices ())
    (fun (Case (dt, s, re, im)) ->
      cover_shapes s;
      let a = value dt s re im in
      let m = s.(Array.length s - 2) and n = s.(Array.length s - 1) in
      List.iter
        (fun (factors, cols) ->
          let q, r = Nx.qr ~factors a in
          equal ~msg:"q" (array int) (Array.append (lead s) [| m; cols |]) (Nx.shape q);
          near ~msg:"q r" (Nx.matmul q r) a;
          near ~msg:"qᴴ q" (Nx.matmul (adj q) q) (identity dt (lead s) cols);
          equal ~msg:"r below" float_exact 0. (worst (Nx.tril ~k:(-1) r)))
        [ (Nx.Reduced, Stdlib.min m n); (Nx.Complete, m) ])

let test_lu =
  prop "lu: l u is a's rows by perm" (matrices ())
    (fun (Case (dt, s, re, im)) ->
      cover_shapes s;
      let a = value dt s re im in
      let perm, l, u = Nx.lu a in
      let rows = Nx.take_along_axis ~axis:(-2) (Nx.unsqueeze ~axes:[ -1 ] perm) a in
      near ~msg:"l u" (Nx.matmul l u) rows;
      equal ~msg:"l above" float_exact 0. (worst (Nx.triu ~k:1 l));
      near ~msg:"l's diagonal" (Nx.diagonal l) (Nx.ones_like (Nx.diagonal l));
      equal ~msg:"u below" float_exact 0. (worst (Nx.tril ~k:(-1) u)))

let test_svd =
  prop "svd: u diag s vh is a, s descending" (matrices ())
    (fun (Case (dt, s, re, im)) ->
      cover_shapes s;
      let a = value dt s re im in
      let m = s.(Array.length s - 2) and n = s.(Array.length s - 1) in
      let k = Stdlib.min m n in
      let u, sv, vh = Nx.svd a in
      near ~msg:"u diag s vh" (Nx.matmul (columns_by u sv) vh) a;
      near ~msg:"uᴴ u" (Nx.matmul (adj u) u) (identity dt (lead s) k);
      near ~msg:"vh vhᴴ" (Nx.matmul vh (adj vh)) (identity dt (lead s) k);
      let imag = Nx.imag Nx.float64 (Nx.cast Nx.complex128 sv) in
      equal ~msg:"s is real" float_exact 0. (worst imag);
      let values = magnitudes sv in
      Array.iteri
        (fun i x ->
          if i mod k > 0 then
            less ~msg:"descending" float_exact ~than:(values.(i - 1) +. Float.min_float) x)
        values;
      let u', sv', vh' = Nx.svd ~vectors:false a in
      equal ~msg:"no u" (array int) (Array.append (lead s) [| m; 0 |]) (Nx.shape u');
      equal ~msg:"no vh" (array int) (Array.append (lead s) [| 0; n |]) (Nx.shape vh');
      near ~msg:"values alone" sv' sv;
      let u, _, vh = Nx.svd ~factors:Complete a in
      near ~msg:"complete u" (Nx.matmul (adj u) u) (identity dt (lead s) m);
      near ~msg:"complete vh" (Nx.matmul vh (adj vh)) (identity dt (lead s) n))

let test_svd_nan () =
  let a = Nx.create Nx.float64 [| 2; 2; 2 |] [| 1.; 2.; 3.; 4.; 1.; Float.nan; 0.; 1. |] in
  let u, s, vh = Nx.svd a in
  List.iter (fun (msg, x) -> all_nan ~msg (Nx.get [ 1 ] x)) [ ("u", u); ("s", s); ("vh", vh) ];
  near ~msg:"the finite one" (Nx.matmul (columns_by (Nx.get [ 0 ] u) (Nx.get [ 0 ] s)) (Nx.get [ 0 ] vh)) (Nx.get [ 0 ] a)

let test_eig =
  prop "eig: a v is v diag w" (matrices ~square:true ())
    (fun (Case (dt, s, re, im)) ->
      cover_shapes ~square:true s;
      let n = s.(Array.length s - 1) in
      let check (type s) (dt : (Complex.t, s) D.t) =
        let a = value dt s re im in
        let w, v = Nx.eig a in
        near ~scale:10. ~msg:"a v" (Nx.matmul a v) (columns_by v w);
        let w', v' = Nx.eig ~vectors:false a in
        equal ~msg:"no v" (array int) (Array.append (lead s) [| n; 0 |]) (Nx.shape v');
        near ~scale:10. ~msg:"values alone" (Nx.prod ~axes:[ -1 ] w') (Nx.prod ~axes:[ -1 ] w)
      in
      match dt with
      | D.Complex64 -> check D.Complex64
      | D.Complex128 -> check D.Complex128
      | _ -> ())

let test_eigh =
  prop "eigh: a v is v diag w, w ascending, one triangle read" (matrices ~square:true ())
    (fun (Case (dt, s, re, im)) ->
      cover_shapes ~square:true s;
      let g = value dt s re im in
      let a = Nx.add g (adj g) in
      let n = s.(Array.length s - 1) in
      let w, v = Nx.eigh a in
      near ~msg:"a v" (Nx.matmul a v) (columns_by v w);
      near ~msg:"vᴴ v" (Nx.matmul (adj v) v) (identity dt (lead s) n);
      let values = Nx.to_array (Nx.real Nx.float64 (Nx.cast Nx.complex128 w)) in
      Array.iteri
        (fun i x ->
          if i mod n > 0 then
            less ~msg:"ascending" float_exact ~than:x (values.(i - 1) -. Float.min_float))
        values;
      let noise = Nx.tril ~k:(-1) (Nx.add g g) in
      let upper = Nx.add (Nx.triu a) noise in
      let lower = Nx.add (Nx.tril a) (Nx.matrix_transpose noise) in
      let wu, _ = Nx.eigh ~triangle:Upper upper and wl, _ = Nx.eigh lower in
      near ~msg:"upper triangle" wu w;
      near ~msg:"lower triangle" wl w)

let test_det =
  prop "det: det (a b) is det a det b; slogdet's sign exp logabs is det"
    (matrices ~square:true ())
    (fun (Case (dt, s, re, im)) ->
      cover_shapes ~square:true s;
      let a = conditioned (value dt s re im) in
      let b = Nx.matrix_transpose a in
      let d = Nx.det a in
      near ~scale:10. ~msg:"det (a b)" (Nx.det (Nx.matmul a b)) (Nx.mul d (Nx.det b));
      let sign, logabs = Nx.slogdet a in
      let e = Nx.exp (Nx.real Nx.float64 (Nx.cast Nx.complex128 logabs)) in
      near ~scale:10. ~msg:"sign exp logabs" (Nx.mul sign (Nx.cast dt e)) d;
      near ~tol:(tolerance dt) ~msg:"|sign|" (Nx.cast Nx.float64 (Nx.magnitude Nx.float64 (Nx.cast Nx.complex128 sign)))
        (Nx.ones Nx.float64 (lead s)))

let test_det_cases () =
  let one x = Nx.item [] x in
  equal ~msg:"no row" float_exact 1. (one (Nx.det (Nx.zeros Nx.float64 [| 0; 0 |])));
  equal ~msg:"a swap" float_exact (-1.)
    (one (Nx.det (Nx.create Nx.float64 [| 2; 2 |] [| 0.; 1.; 1.; 0. |])));
  let sign, logabs = Nx.slogdet (Nx.create Nx.float64 [| 2; 2 |] [| 1.; 2.; 2.; 4. |]) in
  equal ~msg:"singular sign" float_exact 0. (one sign);
  equal ~msg:"singular logabs" float_exact Float.neg_infinity (one logabs);
  let big = Nx.mul_s (Nx.eye Nx.float64 200) 100. in
  let sign, logabs = Nx.slogdet big in
  equal ~msg:"beyond the range" (float 1e-9) (200. *. Float.log 100.) (one logabs);
  equal ~msg:"its sign" float_exact 1. (one sign);
  let _, logabs = Nx.slogdet (Nx.create Nx.complex128 [| 1; 1 |] [| { Complex.re = 3.; im = 4. } |]) in
  equal ~msg:"complex logabs" (float 1e-15) (Float.log 5.) (Nx.item [] logabs).re;
  equal ~msg:"its imaginary part" float_exact 0. (Nx.item [] logabs).im;
  (* A zero determinant is +0 whatever the interchanges and the pivots'
     signs: [[-0]] has none, [[1; 2]; [2; 4]] one. *)
  let plus_zero ~msg x = equal ~msg (pair float_exact bool) (0., false) (x, Float.sign_bit x) in
  List.iter
    (fun (msg, s, xs) ->
      let a = Nx.create Nx.float64 s xs in
      plus_zero ~msg (one (Nx.det a));
      plus_zero ~msg:(msg ^ ", slogdet's sign") (one (fst (Nx.slogdet a))))
    [ ("-0", [| 1; 1 |], [| -0. |]);
      ("a -0 pivot", [| 2; 2 |], [| 1.; 0.; 0.; -0. |]);
      ("one interchange", [| 2; 2 |], [| 1.; 2.; 2.; 4. |]) ];
  let z = Nx.det (Nx.create Nx.complex128 [| 1; 1 |] [| { Complex.re = -0.; im = -0. } |]) in
  plus_zero ~msg:"complex -0, real part" (Nx.item [] z).re;
  plus_zero ~msg:"complex -0, imaginary part" (Nx.item [] z).im;
  let sign, logabs = Nx.slogdet (Nx.create Nx.float64 [| 2; 2 |] [| Float.infinity; 0.; 0.; 0. |]) in
  equal ~msg:"an infinite pivot's sign" float_exact 0. (one sign);
  equal ~msg:"an infinite pivot's logabs" float_exact Float.neg_infinity (one logabs)

(* A batch of a finite matrix and one holding NaN: each function's results
   for the finite one hold, and those of the other are NaN. *)
let test_nan_alone () =
  let a = Nx.create Nx.float64 [| 2; 2; 2 |] [| 2.; 1.; 1.; 3.; 1.; Float.nan; Float.nan; 1. |] in
  let finite = Nx.get [ 0 ] a in
  let w, v = Nx.eigh a in
  near ~msg:"eigh" (Nx.matmul (Nx.get [ 0 ] a) (Nx.get [ 0 ] v)) (columns_by (Nx.get [ 0 ] v) (Nx.get [ 0 ] w));
  all_nan ~msg:"eigh's w" (Nx.get [ 1 ] w);
  all_nan ~msg:"eigh's v" (Nx.get [ 1 ] v);
  let c = Nx.cast Nx.complex128 a in
  let w, v = Nx.eig c in
  near ~msg:"eig" (Nx.matmul (Nx.get [ 0 ] c) (Nx.get [ 0 ] v)) (columns_by (Nx.get [ 0 ] v) (Nx.get [ 0 ] w));
  all_nan ~msg:"eig's w" (Nx.get [ 1 ] w);
  all_nan ~msg:"eig's v" (Nx.get [ 1 ] v);
  let d = Nx.det a in
  equal ~msg:"det" (float 1e-12) 5. (Nx.item [ 0 ] d);
  all_nan ~msg:"det's NaN" (Nx.get [ 1 ] d);
  let sign, logabs = Nx.slogdet a in
  equal ~msg:"slogdet" (float 1e-12) (Float.log 5.) (Nx.item [ 0 ] logabs);
  equal ~msg:"slogdet's sign" float_exact 1. (Nx.item [ 0 ] sign);
  all_nan ~msg:"slogdet's NaN sign" (Nx.get [ 1 ] sign);
  all_nan ~msg:"slogdet's NaN logabs" (Nx.get [ 1 ] logabs);
  let p = Nx.pinv a in
  near ~msg:"pinv" (Nx.get [ 0 ] p) (Nx.inv finite);
  all_nan ~msg:"pinv's NaN" (Nx.get [ 1 ] p);
  let p = Nx.pinv ~hermitian:true a in
  near ~msg:"Hermitian pinv" (Nx.get [ 0 ] p) (Nx.inv finite);
  all_nan ~msg:"Hermitian pinv's NaN" (Nx.get [ 1 ] p)

let test_solve =
  prop "solve: a x is b, vector or columns, leading axes broadcast"
    (matrices ~square:true ())
    (fun (Case (dt, s, re, im)) ->
      cover_shapes ~square:true s;
      let g = value dt s re im in
      let a = conditioned g in
      let r = Array.length s and n = s.(Array.length s - 1) in
      let b = Nx.matrix_transpose g in
      near ~msg:"columns" (Nx.matmul a (Nx.solve a b)) b;
      let v = Nx.slice (List.init (r - 1) (fun _ -> Nx.A) @ [ Nx.I 0 ]) (Nx.pad (Array.init r (fun i -> if i = r - 1 then (0, 1) else (0, 0))) (Nx.item [] (Nx.zeros dt [||])) b) in
      let x = Nx.solve a v in
      equal ~msg:"vector shape" (array int) (Nx.shape v) (Nx.shape x);
      near ~msg:"vector" (Nx.matmul a (Nx.unsqueeze ~axes:[ -1 ] x)) (Nx.unsqueeze ~axes:[ -1 ] v);
      near ~msg:"inv" (Nx.matmul a (Nx.inv a)) (identity dt (lead s) n);
      if r = 3 && s.(0) = 2 then begin
        let b1 = Nx.slice [ Nx.R (0, 1); Nx.A; Nx.A ] b in
        near ~msg:"broadcast" (Nx.matmul a (Nx.solve a b1)) (Nx.broadcast_to s b1)
      end)

let test_solve_singular () =
  let a = Nx.create Nx.float64 [| 2; 2; 2 |] [| 1.; 2.; 2.; 4.; 2.; 0.; 0.; 2. |] in
  let x = Nx.solve a (Nx.ones Nx.float64 [| 2; 2 |]) in
  all_nan ~msg:"singular" (Nx.get [ 0 ] x);
  near ~msg:"regular" (Nx.get [ 1 ] x) (Nx.full Nx.float64 [| 2 |] 0.5);
  all_nan ~msg:"inv" (Nx.get [ 0 ] (Nx.inv a))

let test_solve_triangular =
  prop "solve_triangular: a x is b for each triangle and form" (matrices ~square:true ())
    (fun (Case (dt, s, re, im)) ->
      cover_shapes ~square:true s;
      (* Complex values off the diagonal tell aᵀ from aᴴ. *)
      let a = definite (value dt s re im) in
      let b = value dt s im re in
      List.iter
        (fun (triangle, transpose, unit_diag) ->
          let t = match triangle with Nx.Lower -> Nx.tril a | Nx.Upper -> Nx.triu a in
          let t =
            if not unit_diag then t
            else Nx.add (match triangle with Nx.Lower -> Nx.tril ~k:(-1) a | Nx.Upper -> Nx.triu ~k:1 a)
                   (identity dt (lead s) s.(Array.length s - 1))
          in
          let x = Nx.solve_triangular ~triangle ?transpose ~unit_diag a b in
          let t =
            match transpose with
            | None -> t
            | Some Nx.Transpose -> Nx.matrix_transpose t
            | Some Nx.Adjoint -> adj t
          in
          near ~scale:10. ~msg:"t x" (Nx.matmul t x) b)
        [ (Nx.Lower, None, false); (Nx.Upper, None, false);
          (Nx.Lower, Some Nx.Adjoint, false); (Nx.Upper, Some Nx.Adjoint, true);
          (Nx.Lower, Some Nx.Transpose, false); (Nx.Upper, Some Nx.Transpose, true);
          (Nx.Lower, None, true) ])

let test_lstsq =
  prop "lstsq: aᴴ (b - a x) is zero, x of least norm, residuals" (matrices ())
    (fun (Case (dt, s, re, im)) ->
      cover_shapes s;
      let a = conditioned (value dt s re im) in
      let m = s.(Array.length s - 2) in
      let bs = Array.append (lead s) [| m; 2 |] in
      let k = Array.fold_left ( * ) 1 bs in
      let b = Nx.cast dt (Nx.sin (Nx.reshape bs (Nx.arange_f Nx.float64 0. (Float.of_int k) 1.))) in
      let x, residuals, rank, sv = Nx.lstsq a b in
      let r = Nx.sub b (Nx.matmul a x) in
      let ar = Nx.matmul (adj a) r in
      near ~scale:10. ~msg:"aᴴ r" ar (Nx.zeros_like ar);
      near ~scale:10. ~msg:"least norm" x (Nx.matmul (Nx.pinv a) b);
      let squares = magnitudes (Nx.sum ~axes:[ -2 ] (Nx.mul (Nx.conjugate r) r)) in
      let residuals = magnitudes residuals in
      Array.iteri
        (fun i v ->
          equal ~msg:"residuals" (float (tolerance dt *. 10. *. Float.max 1. v)) v residuals.(i))
        squares;
      let _, s', _ = Nx.svd ~vectors:false a in
      equal ~msg:"s" (array int) (Nx.shape s') (Nx.shape sv);
      let full = Int64.of_int (Stdlib.min m s.(Array.length s - 1)) in
      Array.iter (equal ~msg:"rank" int64 full) (Nx.to_array rank))

let test_lstsq_rank () =
  let a = Nx.create Nx.float64 [| 3; 2 |] [| 1.; 1.; 2.; 2.; 3.; 3. |] in
  let _, _, rank, _ = Nx.lstsq a (Nx.ones Nx.float64 [| 3 |]) in
  equal ~msg:"repeated column" int64 1L (Nx.item [] rank);
  let a = Nx.create Nx.float64 [| 2; 2 |] [| 1.; Float.nan; 0.; 1. |] in
  let x, _, rank, _ = Nx.lstsq a (Nx.ones Nx.float64 [| 2 |]) in
  equal ~msg:"failed rank" int64 (-1L) (Nx.item [] rank);
  all_nan ~msg:"failed x" x;
  (* 1e-3 is at or below 6e-4 · max m n · 1: it counts as zero. *)
  let a = Nx.create Nx.float64 [| 2; 2 |] [| 1.; 0.; 0.; 1e-3 |] in
  let b = Nx.ones Nx.float64 [| 2 |] in
  let x, _, rank, _ = Nx.lstsq ~rtol:6e-4 a b in
  equal ~msg:"rtol's rank" int64 1L (Nx.item [] rank);
  equal ~msg:"rtol's x" (array (float 1e-15)) [| 1.; 0. |] (Nx.to_array x);
  equal ~msg:"pinv's rtol" (array (float 1e-15)) [| 1.; 0.; 0.; 0. |]
    (Nx.to_array (Nx.pinv ~rtol:6e-4 a))

let test_pinv =
  prop "pinv: the Penrose conditions" (matrices ())
    (fun (Case (dt, s, re, im)) ->
      cover_shapes s;
      let a = conditioned (value dt s re im) in
      let p = Nx.pinv a in
      let ap = Nx.matmul a p and pa = Nx.matmul p a in
      near ~scale:10. ~msg:"a p a" (Nx.matmul ap a) a;
      near ~scale:10. ~msg:"p a p" (Nx.matmul pa p) p;
      near ~scale:10. ~msg:"(a p)ᴴ" (adj ap) ap;
      near ~scale:10. ~msg:"(p a)ᴴ" (adj pa) pa;
      let n = s.(Array.length s - 1) in
      if n = s.(Array.length s - 2) then begin
        let h = definite a in
        near ~scale:10. ~msg:"hermitian" (Nx.pinv ~hermitian:true h) (Nx.pinv h);
        (* q diag w qᴴ with eigenvalues 1, -2, 3, …: Hermitian, indefinite. *)
        cover "indefinite" (n >= 2);
        let q, _ = Nx.qr a in
        let w = Array.init n (fun i -> Float.of_int (if i mod 2 = 0 then i + 1 else -(i + 1))) in
        let w = Nx.cast dt (Nx.create Nx.float64 [| n |] w) in
        let h = Nx.matmul (columns_by q w) (adj q) in
        near ~scale:10. ~msg:"hermitian, indefinite" (Nx.pinv ~hermitian:true h) (Nx.pinv h)
      end)

(* [a] equal to [b], or within [tol] of it relative to its scale and at
   least 1. *)
let close tol =
  Testable.make ~pp:Format.pp_print_float ~equal:(fun a b ->
      a = b || Float.abs (a -. b) <= tol *. Float.max 1. (Float.abs b))

(* The vector norm [ord] of the magnitudes [m], by its definition. *)
let reference ord (m : float array) =
  let sum f = Array.fold_left (fun a v -> a +. f v) 0. m in
  if ord = Float.infinity then Array.fold_left Float.max m.(0) m
  else if ord = Float.neg_infinity then Array.fold_left Float.min m.(0) m
  else if ord = 0. then sum (fun v -> if v = 0. then 0. else 1.)
  else sum (fun v -> v ** ord) ** (1. /. ord)

let test_vector_norms =
  prop "vector_norm: each order is its definition" (matrices ())
    (fun (Case (dt, s, re, im)) ->
      cover_shapes s;
      (* Zero elements, which order 0 skips and a negative order meets. *)
      let zeroed v = Array.map2 (fun r x -> if r < -0.5 then 0. else x) re v in
      let x = value dt s (zeroed re) (zeroed im) in
      let n = s.(Array.length s - 1) in
      let rows = Array.fold_left ( * ) 1 (Array.sub s 0 (Array.length s - 1)) in
      let m = magnitudes x in
      cover "a zero element" (Array.exists (( = ) 0.) m);
      let extremes = if n > 0 then [ Float.infinity; Float.neg_infinity ] else [] in
      List.iter
        (fun ord ->
          let y = magnitudes (Nx.vector_norm ~ord ~axes:[ -1 ] x) in
          for i = 0 to rows - 1 do
            let want = reference ord (Array.sub m (i * n) n) in
            equal ~msg:(Printf.sprintf "ord %g" ord) (close (tolerance dt)) want y.(i)
          done)
        ([ 2.; 1.; 3.; 0.5; 0.; -1.; -2.; -0.5 ] @ extremes);
      equal ~msg:"every axis" (close (tolerance dt)) (reference 2. m)
        (magnitudes (Nx.vector_norm x)).(0))

let test_matrix_norms =
  prop "matrix_norm: each order is its definition" (matrices ())
    (fun (Case (dt, s, re, im)) ->
      cover_shapes s;
      let x = value dt s re im in
      let m = s.(Array.length s - 2) and n = s.(Array.length s - 1) in
      assume (m > 0 && n > 0);
      let f64 y = Nx.magnitude Nx.float64 (Nx.cast Nx.complex128 y) in
      let abs = f64 x in
      let norm ord = f64 (Nx.matrix_norm ~ord x) in
      let near = near ~tol:(tolerance dt) in
      near ~msg:"fro" (norm `Fro) (Nx.sqrt (Nx.sum ~axes:[ -2; -1 ] (Nx.square abs)));
      near ~msg:"one" (norm `One) (Nx.max ~axes:[ -1 ] (Nx.sum ~axes:[ -2 ] abs));
      near ~msg:"negone" (norm `NegOne) (Nx.min ~axes:[ -1 ] (Nx.sum ~axes:[ -2 ] abs));
      near ~msg:"inf" (norm `Inf) (Nx.max ~axes:[ -1 ] (Nx.sum ~axes:[ -1 ] abs));
      near ~msg:"neginf" (norm `NegInf) (Nx.min ~axes:[ -1 ] (Nx.sum ~axes:[ -1 ] abs));
      let _, sv, _ = Nx.svd ~vectors:false (Nx.cast Nx.complex128 x) in
      let sv = Nx.real Nx.float64 sv in
      near ~msg:"two" (norm `Two) (Nx.max ~axes:[ -1 ] sv);
      near ~msg:"negtwo" (norm `NegTwo) (Nx.min ~axes:[ -1 ] sv);
      near ~msg:"nuc" (norm `Nuc) (Nx.sum ~axes:[ -1 ] sv);
      near ~msg:"default" (f64 (Nx.matrix_norm x)) (norm `Fro);
      let kept = Nx.matrix_norm ~keepdims:true x in
      equal ~msg:"keepdims" (array int) (Array.append (lead s) [| 1; 1 |]) (Nx.shape kept))

let test_norm_cases () =
  let f64 xs = Nx.create Nx.float64 [| Array.length xs |] xs in
  let one x = Nx.item [] x in
  let v = f64 [| 1e200; 1e200 |] in
  equal ~msg:"no overflow" (float 1e186) (1e200 *. Float.sqrt 2.) (one (Nx.vector_norm v));
  let v = f64 [| 1e-200; 1e-200 |] in
  equal ~msg:"no underflow" (float 1e-214) (1e-200 /. Float.sqrt 2.)
    (one (Nx.vector_norm ~ord:(-2.) v));
  let empty = f64 [||] in
  equal ~msg:"no element" float_exact 0. (one (Nx.vector_norm empty));
  equal ~msg:"order 0 of no element" float_exact 0. (one (Nx.vector_norm ~ord:0. empty));
  equal ~msg:"a negative order of no element" float_exact Float.infinity
    (one (Nx.vector_norm ~ord:(-1.) empty));
  equal ~msg:"order 0" float_exact 2. (one (Nx.vector_norm ~ord:0. (f64 [| 0.; -2.; 5. |])));
  List.iter
    (fun ord ->
      let y = one (Nx.vector_norm ~ord (f64 [| Float.nan; 0.; 1. |])) in
      equal ~msg:(Printf.sprintf "NaN at order %g" ord) bool true (Float.is_nan y))
    [ 2.; 1.; 0.; -1.; -2.; 3.; Float.infinity; Float.neg_infinity ];
  let z = Nx.vector_norm (Nx.create Nx.complex64 [| 1 |] [| { Complex.re = 3.; im = 4. } |]) in
  equal ~msg:"complex" (float 1e-6) 5. (Nx.item [] z).re;
  equal ~msg:"its imaginary part" float_exact 0. (Nx.item [] z).im;
  let x = Nx.ones Nx.float32 [| 2; 3; 4 |] in
  equal ~msg:"every axis" (array int) [||] (Nx.shape (Nx.vector_norm x));
  equal ~msg:"every axis's value" (float 1e-6) (Float.sqrt 24.) (one (Nx.vector_norm x));
  let y = Nx.vector_norm ~axes:[ 0; 2 ] ~keepdims:true x in
  equal ~msg:"two axes kept" (array int) [| 1; 3; 1 |] (Nx.shape y);
  equal ~msg:"two axes" (float 1e-6) (Float.sqrt 8.) (Nx.item [ 0; 1; 0 ] y);
  let e = Nx.zeros Nx.float64 [| 0; 3 |] in
  equal ~msg:"Frobenius of no element" float_exact 0. (one (Nx.matrix_norm e));
  equal ~msg:"columns of no element" float_exact 0. (one (Nx.matrix_norm ~ord:`One e));
  let none = Nx.zeros Nx.float64 [| 0; 0 |] in
  equal ~msg:"no vector" (array int) [| 0 |]
    (Nx.shape (Nx.vector_norm ~ord:Float.infinity ~axes:[ 1 ] none));
  equal ~msg:"no matrix" (array int) [| 0 |]
    (Nx.shape (Nx.matrix_norm ~ord:`Inf (Nx.zeros Nx.float64 [| 0; 0; 3 |])))

(* Narrow floats: a norm computes at float32 and rounds once. *)

type narrow = Narrow : (float, 's) D.t * int array * float array -> narrow

let narrow_dtypes =
  List.filter
    (fun (D.Any dt) -> D.is D.Float dt && D.bits dt < 32)
    D.all

let narrow_case =
  let open Gen in
  with_pp
    (fun ppf (Narrow (dt, s, _)) ->
      Format.fprintf ppf "%a [%s]" D.pp dt
        (String.concat "; " (Array.to_list (Array.map string_of_int s))))
    (let* (D.Any dt) =
       of_list ~pp:(fun ppf (D.Any dt) -> D.pp ppf dt) narrow_dtypes
     in
     match D.kind dt with
     | D.Float ->
         let* s = array ~size:(int_range 2 3) (int_range 0 3) in
         let+ xs =
           array
             ~size:(constant (Array.fold_left ( * ) 1 s))
             (frequency
                [
                  (6, float_range (-8.) 8.);
                  ( 1,
                    of_list
                      [ 0.; -0.; 1e4; -1e4; Float.infinity; Float.nan ] );
                ])
         in
         Narrow (dt, s, xs)
     | _ -> assert false)

(* Each order of [x], at its dtype and at float32 rounded once: the bits of
   each, or [None] where it raises. *)
let law_narrow_norms (Narrow (dt, s, xs)) =
  let x = Nx.create dt s xs in
  let wide = Nx.cast Nx.float32 x in
  let bits y =
    Array.map Int64.bits_of_float (Nx.to_array (Nx.cast Nx.float64 y))
  in
  let outcome f =
    match f () with y -> Some (bits y) | exception Invalid_argument _ -> None
  in
  let same msg (f : 'r. (float, 'r, Nx.host) Nx.t -> (float, 'r, Nx.host) Nx.t)
      =
    let narrow = outcome (fun () -> f x)
    and rounded = outcome (fun () -> Nx.cast dt (f wide)) in
    cover "computed" (Option.is_some narrow);
    equal ~msg (option (array int64)) rounded narrow
  in
  List.iter
    (fun ord ->
      same (Printf.sprintf "vector_norm ord %g" ord) (fun x ->
          Nx.vector_norm ~ord ~axes:[ -1 ] x))
    [ 2.; 1.; 3.; 0.5; 0.; -1.; -2.; Float.infinity; Float.neg_infinity ];
  List.iter
    (fun (name, ord) -> same name (fun x -> Nx.matrix_norm ~ord x))
    [
      ("fro", `Fro); ("nuc", `Nuc); ("one", `One); ("two", `Two);
      ("inf", `Inf); ("negone", `NegOne); ("negtwo", `NegTwo);
      ("neginf", `NegInf);
    ]

let test_narrow_norms =
  prop "a narrow float's norms are float32's, rounded once" narrow_case
    law_narrow_norms

let invalid_by ~by f = raises_match (Exn.invalid_arg ~substring:(by ^ ": ")) f

let test_refusals () =
  let h = Nx.ones Nx.float16 [| 2; 2 |] and r = Nx.ones Nx.float64 [| 2; 3 |] in
  invalid_by ~by:"Nx.cholesky" (fun () -> Nx.cholesky h);
  invalid_by ~by:"Nx.cholesky" (fun () -> Nx.cholesky r);
  invalid_by ~by:"Nx.qr" (fun () -> Nx.qr (Nx.ones Nx.float64 [| 3 |]));
  invalid_by ~by:"Nx.det" (fun () -> Nx.det r);
  invalid_by ~by:"Nx.inv" (fun () -> Nx.inv (Nx.ones Nx.int32 [| 2; 2 |]));
  invalid_by ~by:"Nx.solve" (fun () -> Nx.solve (Nx.eye Nx.float64 2) (Nx.ones Nx.float64 [| 3 |]));
  invalid_by ~by:"Nx.solve" (fun () -> Nx.solve (Nx.eye Nx.float64 2) (Nx.ones Nx.float64 [| 1; 1; 2; 1 |]));
  invalid_by ~by:"Nx.solve_triangular" (fun () ->
      Nx.solve_triangular (Nx.ones Nx.float64 [| 3; 2; 2 |]) (Nx.ones Nx.float64 [| 2; 2; 1 |]));
  invalid_by ~by:"Nx.lstsq" (fun () -> Nx.lstsq ~rtol:(-1.) r (Nx.ones Nx.float64 [| 2 |]));
  invalid_by ~by:"Nx.lstsq" (fun () -> Nx.lstsq ~rtol:Float.nan r (Nx.ones Nx.float64 [| 2 |]));
  invalid_by ~by:"Nx.pinv" (fun () -> Nx.pinv ~hermitian:true r);
  invalid_by ~by:"Nx.pinv" (fun () -> Nx.pinv ~hermitian:true (Nx.zeros Nx.float64 [| 0; 3 |]));
  let i = Nx.ones Nx.int32 [| 2; 2 |] in
  invalid_by ~by:"Nx.vector_norm" (fun () -> Nx.vector_norm i);
  invalid_by ~by:"Nx.vector_norm" (fun () -> Nx.vector_norm ~ord:Float.nan r);
  invalid_by ~by:"Nx.vector_norm" (fun () -> Nx.vector_norm ~axes:[ 2 ] r);
  invalid_by ~by:"Nx.vector_norm" (fun () -> Nx.vector_norm ~axes:[ 0; 0 ] r);
  invalid_by ~by:"Nx.vector_norm" (fun () ->
      Nx.vector_norm ~ord:Float.infinity (Nx.zeros Nx.float64 [| 0 |]));
  invalid_by ~by:"Nx.vector_norm" (fun () ->
      Nx.vector_norm ~ord:Float.neg_infinity (Nx.zeros Nx.float64 [| 2; 0 |]));
  invalid_by ~by:"Nx.matrix_norm" (fun () -> Nx.matrix_norm i);
  invalid_by ~by:"Nx.matrix_norm" (fun () -> Nx.matrix_norm (Nx.ones Nx.float64 [| 3 |]));
  invalid_by ~by:"Nx.matrix_norm" (fun () -> Nx.matrix_norm ~ord:`Two (Nx.zeros Nx.float64 [| 2; 0 |]));
  invalid_by ~by:"Nx.matrix_norm" (fun () -> Nx.matrix_norm ~ord:`One (Nx.zeros Nx.float64 [| 2; 0 |]))

(* nx.cpu without lu, svd, eigh, eig and gathers. *)
module Gatherless = struct
  include Declining

  let gather ~axis:_ ~dst:_ _ _ = A.Declined
end

(* On a library that declines lu, svd, eigh and gathers, a composite's
   refusal names the function called, whatever function it calls inside. *)
let test_composites_named () =
  let module Declines = (val Nx.devices ~kernels:(module Gatherless) [ m 0 ]) in
  let on s = Nx.place Declines.on (Nx.Repr.of_array Nx.Host.v (A.create Rig.host D.Float64 s)) in
  let a = on [| 2; 2 |] in
  invalid_by ~by:"Nx.lstsq" (fun () -> Nx.lstsq a (on [| 2 |]));
  invalid_by ~by:"Nx.pinv" (fun () -> Nx.pinv a);
  invalid_by ~by:"Nx.pinv" (fun () -> Nx.pinv ~hermitian:true a);
  invalid_by ~by:"Nx.det" (fun () -> Nx.det a);
  invalid_by ~by:"Nx.slogdet" (fun () -> Nx.slogdet a);
  invalid_by ~by:"Nx.matrix_norm" (fun () -> Nx.matrix_norm ~ord:`Nuc a);
  invalid_by ~by:"Nx.matrix_norm" (fun () -> Nx.matrix_norm ~ord:`Two a)

(* A donated operand that another operation consumed is dead: a composite
   names itself, whatever function it computes through. *)
let test_dead_named () =
  let dead dt s v =
    let x = Nx.create dt s (Array.make (Array.fold_left ( * ) 1 s) v) in
    ignore (Nx.neg (Nx.donate x));
    x
  in
  let square () = dead Nx.float64 [| 2; 2 |] 1. in
  invalid_by ~by:"Nx.cholesky" (fun () ->
      Nx.cholesky ~triangle:Upper (square ()));
  invalid_by ~by:"Nx.eigh" (fun () -> Nx.eigh ~triangle:Upper (square ()));
  invalid_by ~by:"Nx.solve_triangular" (fun () ->
      Nx.solve_triangular ~transpose:Transpose (Nx.eye Nx.complex64 2)
        (dead Nx.complex64 [| 2; 1 |] Complex.one))

(* Expansions: each routine but eig computed by nx's expansion, within the
   bounds Expand states and in agreement with the eager kernels. *)

(* [f ()] with each routine computed by its expansion. *)
let expanded f =
  Nx_support.expanded
    (fun (type r) (op : r Nx.Prim.t) ->
      match op with Linalg _ -> true | _ -> false)
    f

let back x = Nx.place Nx.Host.on x

(* The bounds Expand states: [u] the parts' unit roundoff, [ν] the most one
   operation errs by, [2u] on floats and [4u] on complex numbers, and [γ(k) = k
   ν / (1 - k ν)]. *)
let unit_of (type v s) (dt : (v, s) D.t) =
  match dt with
  | D.Float32 | D.Complex64 -> 0x1p-24
  | D.Float64 | D.Complex128 -> 0x1p-53
  | _ -> fail "a dtype no routine takes"

let is_complex (type v s) (dt : (v, s) D.t) =
  match D.kind dt with D.Complex -> true | _ -> false

let nu dt = (if is_complex dt then 4. else 2.) *. unit_of dt

let gamma dt k =
  let v = Float.of_int k *. nu dt in
  v /. (1. -. v)

(* A rotation's error, [6u] or [24u], and the trips of each Jacobi. *)
let rho dt = (if is_complex dt then 24. else 6.) *. unit_of dt
let rounds n = if n mod 2 = 0 then n - 1 else n

let digits (type v s) (dt : (v, s) D.t) =
  match dt with D.Float32 | D.Complex64 -> 24 | _ -> 53

let eigh_trips n e =
  if n <= 1 then 0
  else if rounds n = 1 then 1
  else
    int_of_float
      (Float.ceil
         (-2. *. Float.of_int e *. Float.log 2.
         /. Float.log1p (-1. /. Float.of_int (rounds n))))

let compounded e t = ((1. +. e) ** Float.of_int t) -. 1.

(* Qr's [ε] over [k] reflectors of [m] rows: [(1 + τ)^k - 1], [τ = ν + 2γ(8m +
   24)]. *)
let eps_qr dt m k = compounded (nu dt +. (2. *. gamma dt ((8 * m) + 24))) k

(* nx.cpu's own bounds on eigh's and svd's values ({!Nx_cpu}), against which the
   eager values are taken: [γ(16n)] and [γ(16 max m n + 100)] of [u], doubled
   [k] on complex dtypes. *)
let eager_gamma dt k =
  let k = if is_complex dt then 2 * k else k in
  let v = Float.of_int k *. unit_of dt in
  v /. (1. -. v)

(* What a check's own float64 products and sums of [k] terms add, relative to
   the magnitudes they sum. *)
let check k = Float.of_int (k + 2) *. 0x1p-52

(* [x]'s matrices, or vectors, along its last [k] axes: each one's elements in C
   order, at complex128. *)
let parts ?(k = 2) x =
  let s = Nx.shape x in
  let r = Array.length s in
  let per = Array.fold_left ( * ) 1 (Array.sub s (r - k) k) in
  let all = Nx.to_array (Nx.cast Nx.complex128 (back x)) in
  let b = if per = 0 then 0 else Array.length all / per in
  Array.init b (fun i -> Array.sub all (i * per) per)

(* The Euclidean norm of [m], scaled by its greatest magnitude so that no square
   overflows or underflows. *)
let norm (m : Complex.t array) =
  let big = Array.fold_left (fun t z -> Float.max t (Complex.norm z)) 0. m in
  if big = 0. || not (Float.is_finite big) then big
  else
    big
    *. Float.sqrt
         (Array.fold_left
            (fun t z ->
              t +. Complex.norm2 (Complex.div z { Complex.re = big; im = 0. }))
            0. m)

(* Each element of [x] within [bound] of [y]'s, the bounds' arrays as [x]'s
   matrices: [bound i j] for matrix [i], element [j]. *)
let within ~msg bound x y =
  Array.iteri
    (fun i xs ->
      Array.iteri
        (fun j (z : Complex.t) ->
          let d = Complex.norm (Complex.sub z y.(i).(j)) in
          below
            ~msg:(Printf.sprintf "%s: matrix %d, element %d" msg i j)
            ~than:(bound i j +. Float.min_float)
            d)
        xs)
    x

let c128 x = Nx.cast Nx.complex128 (back x)

(* [|x| |y|], its products' magnitudes at float64. *)
let abs_product x y =
  let m z = Nx.cast Nx.complex128 (Nx.magnitude Nx.float64 (c128 z)) in
  Nx.matmul (m x) (m y)

(* The Frobenius norm of each matrix of [a]. *)
let norms a = Array.map norm (parts a)

type kind = Random | Deficient | Graded

let pp_kind ppf k =
  Format.pp_print_string ppf
    (match k with
    | Random -> "random"
    | Deficient -> "rank-deficient"
    | Graded -> "graded")

type drawn =
  | Drawn : ('v, 's) D.t * kind * int array * float array * float array -> drawn

let pp_drawn ppf (Drawn (dt, k, s, _, _)) =
  Format.fprintf ppf "%a %a [%s]" D.pp dt pp_kind k
    (String.concat "; " (Array.to_list (Array.map string_of_int s)))

(* Matrices of up to 4 rows and columns, square ones where [square]: empty, one
   element, square, tall or wide, random, of a lesser rank, or with columns
   graded down by [2^-10] (float32) or [2^-20] each; behind a batch of none,
   two, or one and two. *)
let drawn ?(square = false) () =
  Gen.with_pp pp_drawn
    (let open Gen in
     let* (Full dt) = of_list fulls in
     let* kind = of_list [ Random; Deficient; Graded ] in
     let* lead = of_list [ [||]; [| 2 |]; [| 1; 2 |] ] in
     let* form =
       of_list
         (if square then [ `Empty; `One; `Square ]
          else [ `Empty; `One; `Square; `Tall; `Wide ])
     in
     let* m, n =
       match form with
       | `Empty -> constant (0, 0)
       | `One -> constant (1, 1)
       | `Square -> map (fun n -> (n, n)) (int_range 2 4)
       | `Tall -> pair (int_range 3 4) (int_range 1 2)
       | `Wide -> pair (int_range 1 2) (int_range 3 4)
     in
     let s = Array.append lead [| m; n |] in
     let k = Array.fold_left ( * ) 1 s in
     let+ re = array ~size:(constant k) (float_range (-1.) 1.)
     and+ im = array ~size:(constant k) (float_range (-1.) 1.) in
     Drawn (dt, kind, s, re, im))

let operand (type v s) (dt : (v, s) D.t) kind s re im : (v, s, Nx.host) Nx.t =
  let g = value dt s re im in
  let r = Array.length s in
  let m = s.(r - 2) and n = s.(r - 1) in
  match kind with
  | Random -> g
  | Deficient ->
      let k = Stdlib.max 0 (Stdlib.min m n - 1) in
      let all = List.init (r - 2) (fun _ -> Nx.A) in
      Nx.matmul
        (Nx.slice (all @ [ Nx.A; Nx.R (0, k) ]) g)
        (Nx.slice (all @ [ Nx.R (0, k); Nx.A ]) g)
  | Graded ->
      let step =
        match dt with D.Float32 | D.Complex64 -> 0x1p-10 | _ -> 0x1p-20
      in
      let d =
        Nx.create Nx.float64 [| n |]
          (Array.init n (fun j -> step ** Float.of_int j))
      in
      columns_by g (Nx.cast dt d)

(* Demands each case the expansions must meet. *)
let cover_drawn dt kind s =
  let r = Array.length s in
  let m = s.(r - 2) and n = s.(r - 1) in
  cover "0×0" (m = 0 && n = 0);
  cover "1×1" (m = 1 && n = 1);
  cover "rank-deficient" (kind = Deficient && Stdlib.min m n >= 2);
  cover "ill-conditioned" (kind = Graded && Stdlib.min m n >= 2);
  cover "batched" (r > 2);
  List.iter (fun (Full d) -> cover (D.name d) (D.name d = D.name dt)) fulls

let cover_shape s =
  let r = Array.length s in
  cover "non-square" (s.(r - 2) <> s.(r - 1))

let expansion name ?square law = prop name (drawn ?square ()) law

(* The factorisations' laws, each within its stated bound plus what the check's
   own float64 arithmetic adds. *)

let check_cholesky dt a l =
  let n = (Nx.shape a).(Nx.ndim a - 1) in
  let bound = parts (abs_product l (adj l)) in
  within ~msg:"L Lᴴ"
    (fun i j -> (gamma dt (n + 1) +. check n) *. Complex.norm bound.(i).(j))
    (parts (Nx.matmul (c128 l) (adj (c128 l))))
    (parts (c128 a))

let test_cholesky_expanded =
  expansion "cholesky: |L Lᴴ - a| is within γ(n + 1) |L| |Lᴴ|" ~square:true
    (fun (Drawn (dt, kind, s, re, im)) ->
      cover_drawn dt kind s;
      let a = definite (operand dt kind s re im) in
      let noise = Nx.tril ~k:(-1) (value dt s im re) in
      List.iter
        (fun (triangle, read, l_of) ->
          let l = l_of (expanded (fun () -> Nx.cholesky ~triangle read)) in
          check_cholesky dt a l;
          if kind <> Graded then
            near ~scale:10. ~msg:"eager" l (l_of (Nx.cholesky ~triangle read)))
        [
          (Nx.Lower, Nx.add a (Nx.matrix_transpose noise), Fun.id);
          (Nx.Upper, Nx.add a noise, adj);
        ])

let test_cholesky_expanded_fails () =
  let a =
    Nx.create Nx.float64 [| 2; 2; 2 |] [| 2.; 1.; 1.; 2.; 1.; 2.; 2.; 1. |]
  in
  let l = back (expanded (fun () -> Nx.cholesky a)) in
  near ~msg:"the definite one" (Nx.get [ 0 ] l) (Nx.get [ 0 ] (Nx.cholesky a));
  all_nan ~msg:"the indefinite one" (Nx.get [ 1 ] l)

let test_solve_expanded =
  expansion "solve_triangular: |t x - b| is within γ(n) |t| |x|" ~square:true
    (fun (Drawn (dt, kind, s, re, im)) ->
      cover_drawn dt kind s;
      let n = s.(Array.length s - 1) in
      let a = definite (operand dt kind s re im) in
      let b = value dt s im re in
      List.iter
        (fun (triangle, transpose, unit_diag) ->
          let x =
            expanded (fun () ->
                Nx.solve_triangular ~triangle ?transpose ~unit_diag a b)
          in
          let t =
            match triangle with Nx.Lower -> Nx.tril a | Nx.Upper -> Nx.triu a
          in
          let t =
            if not unit_diag then t
            else
              Nx.add
                (match triangle with
                | Nx.Lower -> Nx.tril ~k:(-1) a
                | Nx.Upper -> Nx.triu ~k:1 a)
                (identity dt (lead s) n)
          in
          let t =
            match transpose with
            | None -> t
            | Some Nx.Transpose -> Nx.matrix_transpose t
            | Some Nx.Adjoint -> adj t
          in
          let bound = parts (abs_product t x) in
          within ~msg:"t x"
            (fun i j -> (gamma dt n +. check n) *. Complex.norm bound.(i).(j))
            (parts (Nx.matmul (c128 t) (c128 x)))
            (parts (c128 b));
          if kind <> Graded then
            near ~scale:10. ~msg:"eager" (back x)
              (Nx.solve_triangular ~triangle ?transpose ~unit_diag a b))
        [
          (Nx.Lower, None, false);
          (Nx.Upper, None, false);
          (Nx.Lower, Some Nx.Adjoint, false);
          (Nx.Upper, Some Nx.Adjoint, true);
          (Nx.Upper, Some Nx.Transpose, false);
          (Nx.Lower, None, true);
        ])

let test_solve_expanded_singular () =
  let a =
    Nx.create Nx.float64 [| 2; 2; 2 |] [| 1.; 0.; 3.; 0.; 2.; 0.; 1.; 2. |]
  in
  let x =
    expanded (fun () ->
        Nx.solve_triangular a (Nx.ones Nx.float64 [| 2; 2; 1 |]))
  in
  all_nan ~msg:"a zero on the diagonal" (Nx.get [ 0 ] x);
  near ~msg:"the regular one" (Nx.get [ 1 ] x)
    (Nx.create Nx.float64 [| 2; 1 |] [| 0.5; 0.25 |])

let test_lu_expanded =
  expansion "lu: |l u - p a| is within γ(k) |l| |u|"
    (fun (Drawn (dt, kind, s, re, im)) ->
      cover_drawn dt kind s;
      cover_shape s;
      let a = operand dt kind s re im in
      let r = Array.length s in
      let k = Stdlib.min s.(r - 2) s.(r - 1) in
      let perm, l, u = expanded (fun () -> Nx.lu a) in
      let rows =
        Nx.take_along_axis ~axis:(-2) (Nx.unsqueeze ~axes:[ -1 ] (back perm)) a
      in
      let bound = parts (abs_product l u) in
      within ~msg:"l u"
        (fun i j -> (gamma dt k +. check k) *. Complex.norm bound.(i).(j))
        (parts (Nx.matmul (c128 l) (c128 u)))
        (parts (c128 rows));
      equal ~msg:"l above" float_exact 0. (worst (Nx.triu ~k:1 (back l)));
      equal ~msg:"u below" float_exact 0. (worst (Nx.tril ~k:(-1) (back u)));
      if kind = Random then begin
        let perm', l', u' = Nx.lu a in
        equal ~msg:"perm" (array int64) (Nx.to_array perm')
          (Nx.to_array (back perm));
        near ~scale:10. ~msg:"eager l" (back l) l';
        near ~scale:10. ~msg:"eager u" (back u) u'
      end)

(* The computed q r is a column by column within (1 + √w (1 + ε)) ε ‖a_j‖, and
   qᴴ q is I within 2ε + ε² elementwise, ε Qr's over min m n reflectors of m
   rows, w q's columns: Expand's statements, plus the check's own float64
   error. *)
let check_qr dt a q r =
  let s = Nx.shape a in
  let rk = Array.length s in
  let m = s.(rk - 2) and n = s.(rk - 1) in
  let w = (Nx.shape q).(rk - 1) in
  let eps = eps_qr dt m (Stdlib.min m n) in
  let columns x = parts ~k:1 (Nx.matrix_transpose x) in
  let want = columns (c128 a) and got = columns (Nx.matmul (c128 q) (c128 r)) in
  Array.iteri
    (fun j c ->
      below
        ~msg:(Printf.sprintf "q r, column %d" j)
        ~than:
          ((((1. +. (Float.sqrt (Float.of_int w) *. (1. +. eps))) *. eps)
           +. check (m * w))
           *. norm c
          +. Float.min_float)
        (norm (Array.map2 Complex.sub got.(j) c)))
    want;
  within ~msg:"qᴴ q"
    (fun _ _ -> (2. *. eps) +. (eps *. eps) +. check m)
    (parts (Nx.matmul (adj (c128 q)) (c128 q)))
    (parts (identity Nx.complex128 (Array.sub s 0 (rk - 2)) w))

let test_qr_expanded =
  expansion "qr: q r is a and q orthonormal within Qr's ε"
    (fun (Drawn (dt, kind, s, re, im)) ->
      cover_drawn dt kind s;
      cover_shape s;
      let a = operand dt kind s re im in
      let r = Array.length s in
      let m = s.(r - 2) and n = s.(r - 1) in
      List.iter
        (fun (factors, cols) ->
          let q, rr = expanded (fun () -> Nx.qr ~factors a) in
          let q = back q and rr = back rr in
          equal ~msg:"q" (array int)
            (Array.append (lead s) [| m; cols |])
            (Nx.shape q);
          check_qr dt a q rr;
          equal ~msg:"r below" float_exact 0. (worst (Nx.tril ~k:(-1) rr));
          if kind = Random then
            near ~scale:10. ~tol:(tolerance dt) ~msg:"eager |r_ii|"
              (Nx.magnitude Nx.float64 (Nx.cast Nx.complex128 (Nx.diagonal rr)))
              (Nx.magnitude Nx.float64
                 (Nx.cast Nx.complex128 (Nx.diagonal (snd (Nx.qr ~factors a))))))
        [ (Nx.Reduced, Stdlib.min m n); (Nx.Complete, m) ])

(* Svd's laws on [a]: values within its bound of eager nx's, plus eager nx's
   own, u diag s vh within its bound of [a], and u's columns and vh's rows
   within ε_qr + ε_J of orthonormal. *)
let check_svd dt a =
  let s = Nx.shape a in
  let r = Array.length s in
  let m = s.(r - 2) and n = s.(r - 1) in
  let k = Stdlib.min m n and mm = Stdlib.max m n in
  let u0 = unit_of dt in
  let floor = (2. *. rho dt) +. gamma dt (k + 2) in
  let tol = (Float.sqrt (Float.of_int k) *. u0) +. floor in
  let tol_alone = (Float.of_int k *. u0) +. floor in
  let eq = eps_qr dt mm k and ej = compounded (rho dt) (30 * rounds k) in
  let eager = eager_gamma dt ((16 * mm) + 100) in
  let fro = norms a in
  let want =
    parts ~k:1
      (let _, s, _ = Nx.svd ~vectors:false a in
       s)
  in
  let values msg tol got =
    within ~msg
      (fun i j ->
        let sigma = Complex.norm want.(i).(j) in
        (((Float.of_int (k - 1) *. tol) +. gamma dt (k + 4)) *. sigma)
        +. ((eq +. ej +. eager) *. fro.(i)))
      (parts ~k:1 got) want
  in
  let u, sv, vh = expanded (fun () -> Nx.svd a) in
  values "s" tol (back sv);
  let _, alone, _ = expanded (fun () -> Nx.svd ~vectors:false a) in
  values "s alone" tol_alone (back alone);
  let ksq = Float.sqrt (Float.of_int (Stdlib.max 1 k)) in
  within ~msg:"u diag s vh"
    (fun i _ ->
      ksq
      *. ((2. *. Float.of_int k *. tol)
         +. (3. *. eq) +. ej
         +. gamma dt (k + 4)
         +. check (mm * k))
      *. fro.(i))
    (parts (Nx.matmul (columns_by (c128 u) (c128 sv)) (c128 vh)))
    (parts (c128 a));
  let ortho msg x cols e =
    within ~msg
      (fun _ _ -> (3. *. e) +. check mm)
      (parts x)
      (parts (identity Nx.complex128 (Array.sub s 0 (r - 2)) cols))
  in
  let e = eq +. ej in
  ortho "uᴴ u" (Nx.matmul (adj (c128 u)) (c128 u)) k e;
  ortho "vh vhᴴ" (Nx.matmul (c128 vh) (adj (c128 vh))) k e;
  let u, _, vh = expanded (fun () -> Nx.svd ~factors:Complete a) in
  ortho "complete uᴴ u" (Nx.matmul (adj (c128 u)) (c128 u)) m e;
  ortho "complete vh vhᴴ" (Nx.matmul (c128 vh) (adj (c128 vh))) n e

let test_svd_expanded =
  expansion "svd: s, u diag s vh and the vectors within Svd's bounds"
    (fun (Drawn (dt, kind, s, re, im)) ->
      cover_drawn dt kind s;
      cover_shape s;
      check_svd dt (operand dt kind s re im))

(* Eigh's laws on Hermitian [a]: w within its bound of eager nx's, plus eager
   nx's own, a v within its bound of v diag w, and v's columns within (1 + ρ)^T
   - 1 of orthonormal. *)
let check_eigh dt a =
  let n = (Nx.shape a).(Nx.ndim a - 1) in
  let t = eigh_trips n (digits dt) and r = rounds n in
  let rh = rho dt and u0 = unit_of dt in
  let fro = norms a in
  let eager = eager_gamma dt (16 * n) in
  let want = parts ~k:1 (fst (Nx.eigh a)) in
  let bound = u0 +. (Float.of_int ((3 * t) + (6 * r)) *. rh) in
  let values msg got =
    within ~msg (fun i _ -> (bound +. eager) *. fro.(i)) (parts ~k:1 got) want
  in
  let w, v = expanded (fun () -> Nx.eigh a) in
  let w = back w and v = back v in
  values "w" w;
  values "w alone" (back (fst (expanded (fun () -> Nx.eigh ~vectors:false a))));
  let residual =
    u0
    +. (Float.of_int ((3 * t) + (6 * r)) *. rh)
    +. (2. *. Float.sqrt (Float.of_int n) *. Float.of_int t *. rh)
  in
  let got = parts (Nx.matmul (c128 a) (c128 v))
  and want = parts (columns_by (c128 v) (c128 w)) in
  Array.iteri
    (fun i g ->
      below
        ~msg:(Printf.sprintf "‖a v - v diag w‖_F, matrix %d" i)
        ~than:(((residual +. check n) *. fro.(i)) +. Float.min_float)
        (norm (Array.map2 Complex.sub g want.(i))))
    got;
  within ~msg:"vᴴ v"
    (fun _ _ -> (3. *. compounded rh t) +. check n)
    (parts (Nx.matmul (adj (c128 v)) (c128 v)))
    (parts (identity Nx.complex128 (lead (Nx.shape a)) n));
  w

let test_eigh_expanded =
  expansion "eigh: w, a v and vᴴ v within Eigh's bounds" ~square:true
    (fun (Drawn (dt, kind, s, re, im)) ->
      cover_drawn dt kind s;
      let g = operand dt kind s re im in
      let a = Nx.add g (adj g) in
      let w = check_eigh dt a in
      let noise = Nx.tril ~k:(-1) (Nx.add g g) in
      let upper = Nx.add (Nx.triu a) noise in
      within ~msg:"upper triangle"
        (fun _ _ -> 0.)
        (parts ~k:1
           (fst (expanded (fun () -> Nx.eigh ~triangle:Upper upper))))
        (parts ~k:1 w))

let test_jacobi_nan () =
  let a =
    Nx.create Nx.float64 [| 2; 2; 2 |]
      [| 2.; 1.; 1.; 3.; 1.; Float.nan; Float.nan; 1. |]
  in
  let w, v = expanded (fun () -> Nx.eigh a) in
  all_nan ~msg:"eigh's w" (Nx.get [ 1 ] (back w));
  all_nan ~msg:"eigh's v" (Nx.get [ 1 ] (back v));
  near ~msg:"eigh's finite one"
    (Nx.get [ 0 ] (back w))
    (Nx.get [ 0 ] (fst (Nx.eigh a)));
  let u, s, vh = expanded (fun () -> Nx.svd a) in
  List.iter
    (fun (msg, x) -> all_nan ~msg (Nx.get [ 1 ] (back x)))
    [ ("u", u); ("s", s); ("vh", vh) ];
  let _, s', _ = Nx.svd a in
  near ~msg:"svd's finite one" (Nx.get [ 0 ] (back s)) (Nx.get [ 0 ] s')

(* A column whose norm is past [u^-1/2] times smaller than another's still
   rotates against it: [ζ²] overflows there, and a rotation of [t = 0] left the
   pair's cosine above [tol] to the last sweep, a NaN. *)
let test_svd_far_columns () =
  let t = Float.ldexp 1. in
  check_svd D.Float64
    (Nx.create Nx.float64 [| 2; 4 |]
       [| t (-102); 0.; -1.34; 0.; 0.; 0.; -.t (-972); 0. |]);
  check_svd D.Float32
    (Nx.create Nx.float32 [| 2; 4 |]
       [| t (-20); 0.; -1.34; 0.; 0.; 0.; -.t (-100); 0. |])

(* Larger matrices: of order 8 and 16, with a spectrum graded over the dtype's
   precision, and scaled by 2^-70 or 2^66 (float32) or 2^-600 or 2^600
   (float64), where elements' squares underflow or overflow. *)

(* A random unitary matrix, [n × n], real unless [complex]: q of a random
   matrix's QR, at complex128. *)
let unitary rng ~complex n =
  let draw () =
    Nx.create Nx.float64 [| n; n |]
      (Array.init (n * n) (fun _ -> Random.State.float rng 2. -. 1.))
  in
  let im = if complex then draw () else Nx.zeros Nx.float64 [| n; n |] in
  fst (Nx.qr (Nx.complex Nx.complex128 ~re:(draw ()) ~im))

(* [p] diag d [q]ᴴ at [dt], [m × n], [d] from 1 down to [2^-g] in steps equal in
   the exponent, times [2^e]. *)
let spectral_matrix (type v s) (dt : (v, s) D.t) rng ~m ~n ~g ~e ~hermitian :
    (v, s, Nx.host) Nx.t =
  let k = Stdlib.min m n in
  let d =
    Array.init k (fun i ->
        let x = Float.ldexp 1. (-(g * i / Stdlib.max 1 (k - 1))) in
        if hermitian && i mod 2 = 1 then -.x else x)
  in
  let complex = is_complex dt in
  let p = unitary rng ~complex m in
  let q = if hermitian then p else unitary rng ~complex n in
  let p = Nx.slice [ Nx.A; Nx.R (0, k) ] p
  and q = Nx.slice [ Nx.A; Nx.R (0, k) ] q in
  let d = Nx.cast Nx.complex128 (Nx.create Nx.float64 [| k |] d) in
  let x = Nx.matmul (columns_by p d) (adj q) in
  let x =
    if hermitian then Nx.div_s (Nx.add x (adj x)) { Complex.re = 2.; im = 0. }
    else x
  in
  let x = Nx.mul_s x { Complex.re = Float.ldexp 1. e; im = 0. } in
  match dt with
  | D.Complex64 -> Nx.cast dt x
  | D.Complex128 -> Nx.cast dt x
  | D.Float32 -> Nx.cast dt (Nx.real Nx.float64 x)
  | D.Float64 -> Nx.real Nx.float64 x
  | _ -> fail "a dtype no routine takes"

(* The orders, gradings and scales each Jacobi and Qr meet. *)
let large
    (f : 'v 's. ('v, 's) D.t -> Random.State.t -> int -> int -> int -> unit) =
  let rng = Random.State.make [| 6 |] in
  List.iter
    (fun (Full dt) ->
      let g = digits dt - 4 in
      let lo, hi = if digits dt = 24 then (-70, 66) else (-600, 600) in
      List.iter
        (fun (n, g, e) -> f dt rng n g e)
        [ (8, g, 0); (16, g, 0); (8, 4, lo); (8, 4, hi) ])
    fulls

(* A column of subnormal elements beside a normal one: its reflector's [β] is
   subnormal, and computed unscaled it rounded far from its value, so the
   reflector was not unitary and Qr and Svd broke their bounds (seed
   s1:0000000000000011's case). *)
let test_subnormal_column () =
  let tiny = Float.ldexp 1. (-1074) in
  let a =
    Nx.create Nx.float64 [| 3; 2 |]
      [| 0.; 0.; -.tiny; 0x1.6a0cb278c0ab2p-43; tiny; -0x1.6a0b0b3d976f4p-42 |]
  in
  List.iter
    (fun factors ->
      let q, r = expanded (fun () -> Nx.qr ~factors a) in
      check_qr D.Float64 a (back q) (back r))
    [ Nx.Reduced; Nx.Complete ];
  check_svd D.Float64 a

(* Well-conditioned matrices of k = 16 and 32 columns or rows converge within
   dgesvj's 30 sweeps: none comes out NaN, and each meets Svd's bounds. *)
let test_svd_converges () =
  let rng = Random.State.make [| 32 |] in
  List.iter
    (fun (Full dt) ->
      List.iter
        (fun k ->
          List.iter
            (fun (m, n) ->
              let a = spectral_matrix dt rng ~m ~n ~g:3 ~e:0 ~hermitian:false in
              let _, s, _ = expanded (fun () -> Nx.svd a) in
              let _, alone, _ = expanded (fun () -> Nx.svd ~vectors:false a) in
              List.iter
                (fun (msg, x) ->
                  equal
                    ~msg:
                      (Printf.sprintf "%s, %s %d x %d: no NaN" msg (D.name dt) m
                         n)
                    bool false
                    (Array.exists Float.is_nan (magnitudes (back x))))
                [ ("with vectors", s); ("values alone", alone) ];
              check_svd dt a)
            [ (k + 3, k); (k, k + 2) ])
        [ 16; 32 ])
    fulls

let test_large_qr () =
  large (fun dt rng n g e ->
      List.iter
        (fun (m, n) ->
          let a = spectral_matrix dt rng ~m ~n ~g ~e ~hermitian:false in
          List.iter
            (fun factors ->
              let q, r = expanded (fun () -> Nx.qr ~factors a) in
              check_qr dt a (back q) (back r))
            [ Nx.Reduced; Nx.Complete ])
        [ (n + 3, n); (n, n + 5) ])

let test_large_svd () =
  large (fun dt rng n g e ->
      check_svd dt (spectral_matrix dt rng ~m:(n + 3) ~n ~g ~e ~hermitian:false);
      check_svd dt
        (spectral_matrix dt rng ~m:n ~n:(n + 2) ~g ~e ~hermitian:false))

let test_large_eigh () =
  large (fun dt rng n g e ->
      ignore
        (check_eigh dt (spectral_matrix dt rng ~m:n ~n ~g ~e ~hermitian:true)))

(* Loops: each routine through an interpretation whose [repeat] folds keeps
   its carry's forms at every trip, as {!Nx.Prim.expand} states. *)

let form_of (Nx.Prim.Any x) =
  let f = Nx.Prim.form x in
  Format.asprintf "%s %a" (D.name f.dtype) L.pp f.layout

(* [f ()] with its routines expanded through a [repeat] that folds and checks
   that each trip gives its carry's forms back. *)
let looping f =
  let repeat n step init =
    let rec go k c =
      if k = n then c
      else
        let c' = step (Nx.scalar Nx.int64 (Int64.of_int k)) c in
        equal ~msg:"a trip's forms" (list string) (List.map form_of c)
          (List.map form_of c');
        go (k + 1) c'
    in
    go 0 init
  in
  Nx.Prim.interpret ~name:"test.loops" Extent
    (fun (type r) i ~by (op : r Nx.Prim.t) : r ->
      match op with
      | Linalg _ -> (
          match Nx.Prim.expand i ~by ~repeat op with
          | Some r -> r
          | None -> Nx.Prim.eval ~by op)
      | _ -> Nx.Prim.eval ~by op)
    (fun _ -> f ())

let test_loops_fold =
  expansion "a folding repeat keeps each trip's forms" ~square:true
    (fun (Drawn (dt, kind, s, re, im)) ->
      cover_drawn dt kind s;
      let g = operand dt kind s re im in
      let a = Nx.add g (adj g) and p = definite (operand dt kind s re im) in
      let each f = ignore (looping f) in
      each (fun () -> Nx.cholesky p);
      each (fun () -> Nx.solve_triangular ~transpose:Adjoint p g);
      each (fun () -> Nx.lu g);
      each (fun () -> Nx.qr g);
      each (fun () -> Nx.eigh a);
      each (fun () -> Nx.svd g))

let () =
  exit
    (run "nx linalg"
       [
         group "expansions"
           [
             test_cholesky_expanded;
             test "cholesky's expansion fails in its matrix alone"
               test_cholesky_expanded_fails;
             test_solve_expanded;
             test "a solve's expansion is NaN for a singular triangle alone"
               test_solve_expanded_singular;
             test_lu_expanded;
             test_qr_expanded;
             test_svd_expanded;
             test_eigh_expanded;
             test "Jacobi's expansions fail in their matrix alone" test_jacobi_nan;
             test "svd rotates a column far below another" test_svd_far_columns;
             test "svd converges at k = 16 and 32" test_svd_converges;
             test "qr and svd beside a subnormal column" test_subnormal_column;
             test "qr at orders 8 and 16, tall and wide, graded and scaled"
               test_large_qr;
             test "svd at orders 8 and 16, graded and scaled" test_large_svd;
             test "eigh at orders 8 and 16, graded and scaled" test_large_eigh;
             test_loops_fold;
           ];
         group "routines"
           [
             test "each routine's results have Shape's shapes" test_forms;
             test "the rule raises before any kernel" test_rules;
             test "each routine reaches the kernels, its results in order"
               test_dispatch;
             test "a split batch factors on each device" test_split;
             test "a declined routine raises naming the move" test_declined;
           ];
         group "factorisations"
           [
             test_cholesky;
             test "cholesky fails in its matrix alone" test_cholesky_fails;
             test_qr;
             test_lu;
             test_svd;
             test "svd fails in its matrix alone" test_svd_nan;
             test_eig;
             test_eigh;
           ];
         group "determinants and solves"
           [
             test_det;
             test "det and slogdet at their edges" test_det_cases;
             test_solve;
             test "solve is NaN for a singular matrix alone" test_solve_singular;
             test_solve_triangular;
             test_lstsq;
             test "lstsq's rank" test_lstsq_rank;
             test_pinv;
             test "a NaN stays in its matrix" test_nan_alone;
           ];
         group "norms"
           [
             test_vector_norms;
             test_matrix_norms;
             test "norms at their edges" test_norm_cases;
             test_narrow_norms;
           ];
         group "refusals"
           [
             test "each refusal names its function" test_refusals;
             test "a composite names its function" test_composites_named;
             test "a dead operand names the function called" test_dead_named;
           ];
       ])
