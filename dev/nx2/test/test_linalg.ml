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
    let shape (A.Any a) = Iarray.to_list (L.shape (A.layout a)) in
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
             let s = Iarray.to_array (L.shape f.layout) in
             out := (k, D.name f.dtype, s) :: !out;
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
    (Svd { factors = Reduced; dtype = Float32; a = f32 [| 3; 5 |] });
  check "svd of a complex matrix"
    [
      (0, "float32", [| 3 |]);
      (1, "complex64", [| 3; 3 |]);
      (2, "complex64", [| 3; 5 |]);
    ]
    (Svd { factors = Reduced; dtype = Float32; a = c64 [| 3; 5 |] });
  check "svd complete"
    [
      (0, "float32", [| 3 |]);
      (1, "float32", [| 3; 3 |]);
      (2, "float32", [| 5; 5 |]);
    ]
    (Svd { factors = Complete; dtype = Float32; a = f32 [| 3; 5 |] });
  check "singular values"
    [ (0, "float64", [| 3 |]) ]
    (Svd_values { dtype = Float64; a = zeros D.Complex128 [| 3; 5 |] });
  check "eigh"
    [ (0, "float32", [| 4 |]); (1, "complex64", [| 4; 4 |]) ]
    (Eigh { dtype = Float32; a = c64 [| 4; 4 |] });
  check "eigh's values"
    [ (0, "float32", [| 4 |]) ]
    (Eigh_values { dtype = Float32; a = f32 [| 4; 4 |] });
  check "eig"
    [ (0, "complex64", [| 4 |]); (1, "complex64", [| 4; 4 |]) ]
    (Eig { dtype = Complex64; a = c64 [| 4; 4 |] });
  check "eig of a real matrix"
    [ (0, "complex128", [| 4 |]); (1, "complex128", [| 4; 4 |]) ]
    (Eig { dtype = Complex128; a = zeros D.Float64 [| 4; 4 |] });
  check "eig's values"
    [ (0, "complex64", [| 4 |]) ]
    (Eig_values { dtype = Complex64; a = f32 [| 4; 4 |] });
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
      linalg (Eigh { dtype = Float32; a = zeros D.Float16 [| 2; 2 |] }));
  raises (fun () ->
      linalg (Svd_values { dtype = Float64; a = zeros D.Float64 [| 3 |] }));
  (* A real result of another precision than [a]'s. *)
  let into sub f =
    raises_match (Exn.invalid_arg ~substring:("t: " ^ sub)) f
  in
  into "Svd of complex64 into float64" (fun () ->
      linalg
        (Svd_values { dtype = Float64; a = zeros D.Complex64 [| 3; 3 |] }));
  into "Svd of float64 into float32" (fun () ->
      linalg
        (Svd
           { factors = Reduced; dtype = Float32; a = zeros D.Float64 [| 3; 3 |] }));
  into "Eigh of complex128 into float32" (fun () ->
      linalg (Eigh { dtype = Float32; a = zeros D.Complex128 [| 2; 2 |] }));
  into "Eigh of float32 into float16" (fun () ->
      linalg (Eigh_values { dtype = Float16; a = zeros D.Float32 [| 2; 2 |] }));
  raises (fun () ->
      linalg
        (Eig_values { dtype = Complex64; a = zeros D.Float64 [| 2; 2 |] }));
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
  let s, u, vh = linalg (Svd { factors = Reduced; dtype = Float64; a }) in
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
  ignore (linalg (Svd_values { dtype = Float64; a = tall }));
  let sq = zeros D.Complex64 [| 3; 3 |] in
  ignore (linalg (Eigh { dtype = Float32; a = sq }));
  ignore (linalg (Eigh_values { dtype = Float32; a = sq }));
  ignore (linalg (Eig { dtype = Complex64; a = sq }));
  ignore (linalg (Eig_values { dtype = Complex64; a = sq }));
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
  let w, v = linalg (Eigh { dtype = Float32; a }) in
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
  match linalg (Eig_values { dtype = D.Complex64; a }) with
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

(* The float of a dtype's precision, the routines' real results'. *)
type real = Real : (float, 'r) D.t -> real

let real_of (type v s) (dt : (v, s) D.t) =
  match dt with D.Float32 | D.Complex64 -> Real D.Float32 | _ -> Real D.Float64

(* The complex dtype of a dtype's precision, eig's results'. *)
type cplx = Cplx : (Complex.t, 'c) D.t -> cplx

let complex_of (type v s) (dt : (v, s) D.t) =
  match dt with
  | D.Float32 | D.Complex64 -> Cplx D.Complex64
  | _ -> Cplx D.Complex128

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
   batch of none, one or two, of [dtypes] (default {!fulls}). *)
let matrices ?(square = false) ?(dtypes = fulls) () =
  Gen.with_pp pp_case
    (let open Gen in
     let* (Full dt) = of_list dtypes in
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
      let (Real r) = real_of dt in
      let u, sv, vh = Nx.svd r a in
      near ~msg:"u diag s vh" (Nx.matmul (columns_by u (Nx.cast dt sv)) vh) a;
      near ~msg:"uᴴ u" (Nx.matmul (adj u) u) (identity dt (lead s) k);
      near ~msg:"vh vhᴴ" (Nx.matmul vh (adj vh)) (identity dt (lead s) k);
      let values = Nx.to_array (Nx.cast Nx.float64 sv) in
      Array.iter (greater ~msg:"not negative, a zero +0" float_exact ~than:(-0.)) values;
      Array.iteri
        (fun i x ->
          if i mod k > 0 then
            less ~msg:"descending" float_exact ~than:(values.(i - 1) +. Float.min_float) x)
        values;
      let u', sv', vh' = Nx.svd ~vectors:false r a in
      equal ~msg:"no u" (array int) (Array.append (lead s) [| m; 0 |]) (Nx.shape u');
      equal ~msg:"no vh" (array int) (Array.append (lead s) [| 0; n |]) (Nx.shape vh');
      near ~msg:"values alone" sv' sv;
      let u, _, vh = Nx.svd ~factors:Complete r a in
      near ~msg:"complete u" (Nx.matmul (adj u) u) (identity dt (lead s) m);
      near ~msg:"complete vh" (Nx.matmul vh (adj vh)) (identity dt (lead s) n))

let test_svd_nan () =
  let a = Nx.create Nx.float64 [| 2; 2; 2 |] [| 1.; 2.; 3.; 4.; 1.; Float.nan; 0.; 1. |] in
  let u, s, vh = Nx.svd Nx.float64 a in
  List.iter (fun (msg, x) -> all_nan ~msg (Nx.get [ 1 ] x)) [ ("u", u); ("s", s); ("vh", vh) ];
  near ~msg:"the finite one" (Nx.matmul (columns_by (Nx.get [ 0 ] u) (Nx.get [ 0 ] s)) (Nx.get [ 0 ] vh)) (Nx.get [ 0 ] a)

(* The least singular value of [a - λ I] for each λ of [w], through the
   real form [[x_r -x_i]; [x_i x_r]] of [a - λ I], whose singular values are
   its own, each twice: small where λ is an eigenvalue of a matrix near
   [a]. *)
let least_singular (type r) (r : (float, r) D.t) n a w =
  let lambda = Nx.unsqueeze ~axes:[ -1 ] (Nx.unsqueeze ~axes:[ -1 ] w) in
  let x = Nx.sub (Nx.unsqueeze ~axes:[ -3 ] a) (Nx.mul lambda (Nx.eye (Nx.dtype a) n)) in
  let xr = Nx.real r x and xi = Nx.imag r x in
  let row l = Nx.concatenate ~axis:(-1) l in
  let x = Nx.concatenate ~axis:(-2) [ row [ xr; Nx.neg xi ]; row [ xi; xr ] ] in
  Nx.matrix_norm r ~ord:`NegTwo x

let test_eig =
  prop "eig: a v is v diag w, of a real or complex a, its values with v or without"
    (matrices ~square:true ())
    (fun (Case (dt, s, re, im)) ->
      cover_shapes ~square:true s;
      let n = s.(Array.length s - 1) in
      let a = value dt s re im in
      let (Cplx c) = complex_of dt in
      let (Real r) = real_of dt in
      let w, v = Nx.eig c a in
      let w', v' = Nx.eig ~vectors:false c a in
      equal ~msg:"no v" (array int) (Array.append (lead s) [| n; 0 |]) (Nx.shape v');
      let a = Nx.cast c a in
      near ~msg:"a v" (Nx.matmul a v) (columns_by v w);
      near ~msg:"‖v_j‖"
        (Nx.vector_norm r ~axes:[ -2 ] v)
        (Nx.ones r (Array.append (lead s) [| n |]));
      if n > 0 then
        List.iter
          (fun (msg, w) ->
            below ~msg:("σ_min (a - w_j I), " ^ msg)
              ~than:(tolerance r *. Float.max 1. (worst a))
              (worst (least_singular r n a w)))
          [ ("with v", w); ("alone", w') ])

let test_eigh =
  prop "eigh: a v is v diag w, w ascending, one triangle read" (matrices ~square:true ())
    (fun (Case (dt, s, re, im)) ->
      cover_shapes ~square:true s;
      let g = value dt s re im in
      let a = Nx.add g (adj g) in
      let n = s.(Array.length s - 1) in
      let (Real r) = real_of dt in
      let w, v = Nx.eigh r a in
      near ~msg:"a v" (Nx.matmul a v) (columns_by v (Nx.cast dt w));
      near ~msg:"vᴴ v" (Nx.matmul (adj v) v) (identity dt (lead s) n);
      let values = Nx.to_array (Nx.cast Nx.float64 w) in
      Array.iteri
        (fun i x ->
          if i mod n > 0 then
            less ~msg:"ascending" float_exact ~than:x (values.(i - 1) -. Float.min_float))
        values;
      let noise = Nx.tril ~k:(-1) (Nx.add g g) in
      let upper = Nx.add (Nx.triu a) noise in
      let lower = Nx.add (Nx.tril a) (Nx.matrix_transpose noise) in
      let wu, _ = Nx.eigh ~triangle:Upper r upper and wl, _ = Nx.eigh r lower in
      near ~msg:"upper triangle" wu w;
      near ~msg:"lower triangle" wl w)

(* Every float format, each a dtype a real result may take. *)
type flt = Flt : (float, 's) D.t -> flt

let float_formats =
  D.
    [
      Flt Float64; Flt Float32; Flt Float16; Flt Bfloat16; Flt Float8_e4m3fn;
      Flt Float8_e5m2; Flt Float4_e2m1fn;
    ]

(* [x]'s parts at float64, which holds every float format's values. *)
let parts (type v s) (x : (v, s, Nx.host) Nx.t) =
  let z = Nx.cast Nx.complex128 x in
  Array.append
    (Nx.to_array (Nx.real Nx.float64 z))
    (Nx.to_array (Nx.imag Nx.float64 z))

let test_rounded_once =
  prop "a result in dt is a's precision's, rounded once to dt"
    (matrices ~square:true ())
    (fun (Case (dt, s, re, im)) ->
      cover_shapes ~square:true s;
      let g = value dt s re im in
      let a = Nx.add g (adj g) in
      let (Real r) = real_of dt in
      let rounded msg want x =
        equal ~msg (array float_exact) (parts want) (parts x)
      in
      let _, s_r, _ = Nx.svd r a in
      let _, values_r, _ = Nx.svd ~vectors:false r a in
      let w_r, _ = Nx.eigh r a in
      let _, l_r = Nx.slogdet r a in
      let _, res_r, _, ls_r = Nx.lstsq r a a in
      List.iter
        (fun (Flt f) ->
          cover "wider" (D.bits f > D.bits r);
          let msg what = what ^ " in " ^ D.name f in
          let _, s', _ = Nx.svd f a in
          rounded (msg "svd's s") (Nx.cast f s_r) s';
          let _, s', _ = Nx.svd ~vectors:false f a in
          rounded (msg "svd's values") (Nx.cast f values_r) s';
          rounded (msg "eigh's w") (Nx.cast f w_r) (fst (Nx.eigh f a));
          rounded (msg "slogdet's logabs") (Nx.cast f l_r) (snd (Nx.slogdet f a));
          let _, res', _, ls' = Nx.lstsq f a a in
          rounded (msg "lstsq's residuals") (Nx.cast f res_r) res';
          rounded (msg "lstsq's s") (Nx.cast f ls_r) ls')
        float_formats;
      let (Cplx c) = complex_of dt in
      let w_c, v_c = Nx.eig c a in
      List.iter
        (fun (Cplx c') ->
          let msg what = what ^ " in " ^ D.name c' in
          let w', v' = Nx.eig c' a in
          rounded (msg "eig's w") (Nx.cast c' w_c) w';
          rounded (msg "eig's v") (Nx.cast c' v_c) v')
        [ Cplx D.Complex64; Cplx D.Complex128 ])

let test_det =
  prop "det: det (a b) is det a det b; slogdet's sign exp logabs is det"
    (matrices ~square:true ())
    (fun (Case (dt, s, re, im)) ->
      cover_shapes ~square:true s;
      let a = conditioned (value dt s re im) in
      let b = Nx.matrix_transpose a in
      let d = Nx.det a in
      near ~scale:10. ~msg:"det (a b)" (Nx.det (Nx.matmul a b)) (Nx.mul d (Nx.det b));
      let (Real r) = real_of dt in
      let sign, logabs = Nx.slogdet r a in
      let e = Nx.exp (Nx.cast Nx.float64 logabs) in
      near ~scale:10. ~msg:"sign exp logabs" (Nx.mul sign (Nx.cast dt e)) d;
      near ~tol:(tolerance dt) ~msg:"|sign|" (Nx.cast Nx.float64 (Nx.magnitude Nx.float64 (Nx.cast Nx.complex128 sign)))
        (Nx.ones Nx.float64 (lead s)))

let test_det_cases () =
  let one x = Nx.item [] x in
  equal ~msg:"no row" float_exact 1. (one (Nx.det (Nx.zeros Nx.float64 [| 0; 0 |])));
  equal ~msg:"a swap" float_exact (-1.)
    (one (Nx.det (Nx.create Nx.float64 [| 2; 2 |] [| 0.; 1.; 1.; 0. |])));
  let slogdet a = Nx.slogdet Nx.float64 a in
  let sign, logabs = slogdet (Nx.create Nx.float64 [| 2; 2 |] [| 1.; 2.; 2.; 4. |]) in
  equal ~msg:"singular sign" float_exact 0. (one sign);
  equal ~msg:"singular logabs" float_exact Float.neg_infinity (one logabs);
  let big = Nx.mul_s (Nx.eye Nx.float64 200) 100. in
  let sign, logabs = slogdet big in
  equal ~msg:"beyond the range" (float 1e-9) (200. *. Float.log 100.) (one logabs);
  equal ~msg:"its sign" float_exact 1. (one sign);
  let _, logabs = slogdet (Nx.create Nx.complex128 [| 1; 1 |] [| { Complex.re = 3.; im = 4. } |]) in
  equal ~msg:"complex logabs" (float 1e-15) (Float.log 5.) (Nx.item [] logabs);
  (* A zero determinant is +0 whatever the interchanges and the pivots'
     signs: [[-0]] has none, [[1; 2]; [2; 4]] one. *)
  let plus_zero ~msg x = equal ~msg (pair float_exact bool) (0., false) (x, Float.sign_bit x) in
  List.iter
    (fun (msg, s, xs) ->
      let a = Nx.create Nx.float64 s xs in
      plus_zero ~msg (one (Nx.det a));
      plus_zero ~msg:(msg ^ ", slogdet's sign") (one (fst (slogdet a))))
    [ ("-0", [| 1; 1 |], [| -0. |]);
      ("a -0 pivot", [| 2; 2 |], [| 1.; 0.; 0.; -0. |]);
      ("one interchange", [| 2; 2 |], [| 1.; 2.; 2.; 4. |]) ];
  let z = Nx.det (Nx.create Nx.complex128 [| 1; 1 |] [| { Complex.re = -0.; im = -0. } |]) in
  plus_zero ~msg:"complex -0, real part" (Nx.item [] z).re;
  plus_zero ~msg:"complex -0, imaginary part" (Nx.item [] z).im;
  let sign, logabs = slogdet (Nx.create Nx.float64 [| 2; 2 |] [| Float.infinity; 0.; 0.; 0. |]) in
  equal ~msg:"an infinite pivot's sign" float_exact 0. (one sign);
  equal ~msg:"an infinite pivot's logabs" float_exact Float.neg_infinity (one logabs)

(* A batch of a finite matrix and one holding NaN: each function's results
   for the finite one hold, and those of the other are NaN. *)
let test_nan_alone () =
  let a = Nx.create Nx.float64 [| 2; 2; 2 |] [| 2.; 1.; 1.; 3.; 1.; Float.nan; Float.nan; 1. |] in
  let finite = Nx.get [ 0 ] a in
  let w, v = Nx.eigh Nx.float64 a in
  near ~msg:"eigh" (Nx.matmul (Nx.get [ 0 ] a) (Nx.get [ 0 ] v)) (columns_by (Nx.get [ 0 ] v) (Nx.get [ 0 ] w));
  all_nan ~msg:"eigh's w" (Nx.get [ 1 ] w);
  all_nan ~msg:"eigh's v" (Nx.get [ 1 ] v);
  let c = Nx.cast Nx.complex128 a in
  let w, v = Nx.eig Nx.complex128 a in
  near ~msg:"eig" (Nx.matmul (Nx.get [ 0 ] c) (Nx.get [ 0 ] v)) (columns_by (Nx.get [ 0 ] v) (Nx.get [ 0 ] w));
  all_nan ~msg:"eig's w" (Nx.get [ 1 ] w);
  all_nan ~msg:"eig's v" (Nx.get [ 1 ] v);
  let d = Nx.det a in
  equal ~msg:"det" (float 1e-12) 5. (Nx.item [ 0 ] d);
  all_nan ~msg:"det's NaN" (Nx.get [ 1 ] d);
  let sign, logabs = Nx.slogdet Nx.float64 a in
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
      let (Real f) = real_of dt in
      let x, residuals, rank, sv = Nx.lstsq f a b in
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
      let _, s', _ = Nx.svd ~vectors:false f a in
      equal ~msg:"s" (array float_exact) (Nx.to_array s') (Nx.to_array sv);
      let full = Int64.of_int (Stdlib.min m s.(Array.length s - 1)) in
      Array.iter (equal ~msg:"rank" int64 full) (Nx.to_array rank))

let test_lstsq_rank () =
  let a = Nx.create Nx.float64 [| 3; 2 |] [| 1.; 1.; 2.; 2.; 3.; 3. |] in
  let lstsq ?rtol a b = Nx.lstsq ?rtol Nx.float64 a b in
  let _, _, rank, _ = lstsq a (Nx.ones Nx.float64 [| 3 |]) in
  equal ~msg:"repeated column" int64 1L (Nx.item [] rank);
  let a = Nx.create Nx.float64 [| 2; 2 |] [| 1.; Float.nan; 0.; 1. |] in
  let x, _, rank, _ = lstsq a (Nx.ones Nx.float64 [| 2 |]) in
  equal ~msg:"failed rank" int64 (-1L) (Nx.item [] rank);
  all_nan ~msg:"failed x" x;
  (* 1e-3 is at or below 6e-4 · max m n · 1: it counts as zero. *)
  let a = Nx.create Nx.float64 [| 2; 2 |] [| 1.; 0.; 0.; 1e-3 |] in
  let b = Nx.ones Nx.float64 [| 2 |] in
  let x, _, rank, _ = lstsq ~rtol:6e-4 a b in
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
  prop "vector_norm: each order is its definition, of the magnitudes"
    (matrices ())
    (fun (Case (dt, s, re, im)) ->
      cover_shapes s;
      let (Real r) = real_of dt in
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
          let y = magnitudes (Nx.vector_norm r ~ord ~axes:[ -1 ] x) in
          for i = 0 to rows - 1 do
            let want = reference ord (Array.sub m (i * n) n) in
            equal ~msg:(Printf.sprintf "ord %g" ord) (close (tolerance dt)) want y.(i)
          done)
        ([ 2.; 1.; 3.; 0.5; 0.; -1.; -2.; -0.5 ] @ extremes);
      equal ~msg:"every axis" (close (tolerance dt)) (reference 2. m)
        (magnitudes (Nx.vector_norm r x)).(0))

let test_matrix_norms =
  prop "matrix_norm: each order is its definition"
    (matrices ())
    (fun (Case (dt, s, re, im)) ->
      cover_shapes s;
      let x = value dt s re im in
      let m = s.(Array.length s - 2) and n = s.(Array.length s - 1) in
      assume (m > 0 && n > 0);
      let f64 y = Nx.magnitude Nx.float64 (Nx.cast Nx.complex128 y) in
      let abs = f64 x in
      let (Real r) = real_of dt in
      let norm ord = Nx.cast Nx.float64 (Nx.matrix_norm r ~ord x) in
      let near = near ~tol:(tolerance dt) in
      near ~msg:"fro" (norm `Fro) (Nx.sqrt (Nx.sum ~axes:[ -2; -1 ] (Nx.square abs)));
      near ~msg:"one" (norm `One) (Nx.max ~axes:[ -1 ] (Nx.sum ~axes:[ -2 ] abs));
      near ~msg:"negone" (norm `NegOne) (Nx.min ~axes:[ -1 ] (Nx.sum ~axes:[ -2 ] abs));
      near ~msg:"inf" (norm `Inf) (Nx.max ~axes:[ -1 ] (Nx.sum ~axes:[ -1 ] abs));
      near ~msg:"neginf" (norm `NegInf) (Nx.min ~axes:[ -1 ] (Nx.sum ~axes:[ -1 ] abs));
      let _, sv, _ = Nx.svd ~vectors:false Nx.float64 (Nx.cast Nx.complex128 x) in
      near ~msg:"two" (norm `Two) (Nx.max ~axes:[ -1 ] sv);
      near ~msg:"negtwo" (norm `NegTwo) (Nx.min ~axes:[ -1 ] sv);
      near ~msg:"nuc" (norm `Nuc) (Nx.sum ~axes:[ -1 ] sv);
      near ~msg:"default" (Nx.cast Nx.float64 (Nx.matrix_norm r x)) (norm `Fro);
      let kept = Nx.matrix_norm r ~keepdims:true x in
      equal ~msg:"keepdims" (array int) (Array.append (lead s) [| 1; 1 |])
        (Nx.shape kept))

let test_norm_cases () =
  let f64 xs = Nx.create Nx.float64 [| Array.length xs |] xs in
  let one x = Nx.item [] x in
  let norm ?ord x = one (Nx.vector_norm Nx.float64 ?ord x) in
  equal ~msg:"no overflow" (float 1e186) (1e200 *. Float.sqrt 2.)
    (norm (f64 [| 1e200; 1e200 |]));
  equal ~msg:"no underflow" (float 1e-214) (1e-200 /. Float.sqrt 2.)
    (norm ~ord:(-2.) (f64 [| 1e-200; 1e-200 |]));
  let empty = f64 [||] in
  equal ~msg:"no element" float_exact 0. (norm empty);
  equal ~msg:"order 0 of no element" float_exact 0. (norm ~ord:0. empty);
  equal ~msg:"a negative order of no element" float_exact Float.infinity
    (norm ~ord:(-1.) empty);
  equal ~msg:"order 0" float_exact 2. (norm ~ord:0. (f64 [| 0.; -2.; 5. |]));
  List.iter
    (fun ord ->
      let y = norm ~ord (f64 [| Float.nan; 0.; 1. |]) in
      equal ~msg:(Printf.sprintf "NaN at order %g" ord) bool true (Float.is_nan y))
    [ 2.; 1.; 0.; -1.; -2.; 3.; Float.infinity; Float.neg_infinity ];
  let z = Nx.create Nx.complex64 [| 1 |] [| { Complex.re = 3.; im = 4. } |] in
  equal ~msg:"complex" float_exact 5. (one (Nx.vector_norm Nx.float32 z));
  (* [1; -1; 1] holds in bfloat16, and its norm √3 at float32 is not
     bfloat16's √3 widened. *)
  let b = Nx.create Nx.bfloat16 [| 3 |] [| 1.; -1.; 1. |] in
  let root3 = Int32.float_of_bits (Int32.bits_of_float (Float.sqrt 3.)) in
  equal ~msg:"bfloat16 at float32" float_exact root3
    (one (Nx.vector_norm Nx.float32 b));
  equal ~msg:"bfloat16 at bfloat16" float_exact 1.734375
    (one (Nx.cast Nx.float32 (Nx.vector_norm Nx.bfloat16 b)));
  let x = Nx.ones Nx.float32 [| 2; 3; 4 |] in
  equal ~msg:"every axis" (array int) [||] (Nx.shape (Nx.vector_norm Nx.float32 x));
  equal ~msg:"every axis's value" (float 1e-6) (Float.sqrt 24.)
    (one (Nx.vector_norm Nx.float32 x));
  let y = Nx.vector_norm Nx.float32 ~axes:[ 0; 2 ] ~keepdims:true x in
  equal ~msg:"two axes kept" (array int) [| 1; 3; 1 |] (Nx.shape y);
  equal ~msg:"two axes" (float 1e-6) (Float.sqrt 8.) (Nx.item [ 0; 1; 0 ] y);
  let e = Nx.zeros Nx.float64 [| 0; 3 |] in
  equal ~msg:"Frobenius of no element" float_exact 0.
    (one (Nx.matrix_norm Nx.float64 e));
  equal ~msg:"columns of no element" float_exact 0.
    (one (Nx.matrix_norm Nx.float64 ~ord:`One e));
  let none = Nx.zeros Nx.float64 [| 0; 0 |] in
  equal ~msg:"no vector" (array int) [| 0 |]
    (Nx.shape (Nx.vector_norm Nx.float64 ~ord:Float.infinity ~axes:[ 1 ] none));
  equal ~msg:"no matrix" (array int) [| 0 |]
    (Nx.shape
       (Nx.matrix_norm Nx.float64 ~ord:`Inf (Nx.zeros Nx.float64 [| 0; 0; 3 |])));
  equal ~msg:"no matrix's singular value" (array int) [| 0 |]
    (Nx.shape
       (Nx.matrix_norm Nx.float64 ~ord:`Two (Nx.zeros Nx.float64 [| 0; 2; 0 |])))

(* Rank and condition *)

(* A rank [q] matrix [s v] of [m] rows and [n] columns: [s]'s first [q] rows
   are the identity's, [v]'s rows unit vectors moved by at most [0.1] an
   element, so its [q] singular values other than zero are far above
   [10{^-3}] of the greatest, and its others far below. *)
let law_matrix_rank =
  prop "matrix_rank counts the factors of a product"
    Gen.(
      pair (of_list fulls)
        (pair
           (triple (int_range 0 5) (int_range 0 5) (int_range 0 5))
           (array ~size:(constant 50) (float_range (-1.) 1.))))
    (fun (Full dt, ((m, n, q), r)) ->
      let q = Stdlib.min q (Stdlib.min m n) in
      cover "deficient" (q < Stdlib.min m n);
      cover "full" (q = Stdlib.min m n && q > 0);
      let s =
        Array.init (m * q) (fun k ->
            let i = k / q and j = k mod q in
            if i >= q then r.(k mod 50) else if i = j then 1. else 0.)
      in
      let v =
        Array.init (q * n) (fun k ->
            let i = k / n and j = k mod n in
            (if i = j then 1. else 0.) +. (0.1 *. r.((k + 25) mod 50)))
      in
      let f64 sh a = Nx.create Nx.float64 sh a in
      let a = Nx.to_array (Nx.matmul (f64 [| m; q |] s) (f64 [| q; n |] v)) in
      let a = value dt [| m; n |] a (Array.make (m * n) 0.) in
      equal ~msg:"rank" int64 (Int64.of_int q) (Nx.item [] (Nx.matrix_rank ~rtol:1e-3 a)))

(* cond is its definition: the greatest singular value over the least, the
   norms of [a] and [inv a] multiplied. *)
let law_cond =
  prop "cond: each norm is its definition"
    (matrices ~square:true ())
    (fun (Case (dt, s, re, im)) ->
      let x = value dt s re im in
      assume (s.(Array.length s - 1) > 0);
      cover "complex" (D.is D.Complex dt);
      let (Real r) = real_of dt in
      let f64 y = Nx.cast Nx.float64 y in
      let norm ord y = f64 (Nx.matrix_norm r ~ord y) in
      assume (Array.for_all (fun v -> v > 1e-3) (Nx.to_array (norm `NegTwo x)));
      let near = near ~tol:(tolerance dt) in
      near ~msg:"two" (f64 (Nx.cond r x)) (Nx.div (norm `Two x) (norm `NegTwo x));
      near ~msg:"one" (f64 (Nx.cond r ~p:`One x)) (Nx.mul (norm `One x) (norm `One (Nx.inv x)));
      near ~msg:"inf" (f64 (Nx.cond r ~p:`Inf x)) (Nx.mul (norm `Inf x) (norm `Inf (Nx.inv x))))

let test_rank_cond_cases () =
  let invalid_by ~by f = raises_match (Exn.invalid_arg ~substring:(by ^ ": ")) f in
  let f64 sh a = Nx.create Nx.float64 sh a in
  let ranks ?rtol x = Array.map Int64.to_int (Nx.to_array (Nx.matrix_rank ?rtol x)) in
  let batch =
    Nx.stack
      [ Nx.eye Nx.float64 3; Nx.zeros Nx.float64 [| 3; 3 |];
        f64 [| 3; 3 |] [| 1.; 2.; 3.; 2.; 4.; 6.; 1.; 0.; 0. |];
        f64 [| 3; 3 |] [| 1.; Float.nan; 0.; 0.; 1.; 0.; 0.; 0.; 1. |] ]
  in
  equal ~msg:"a batch" (array int) [| 3; 0; 2; -1 |] (ranks batch);
  equal ~msg:"rtol" (array int) [| 1 |]
    (ranks ~rtol:1e-2 (Nx.reshape [| 1; 2; 2 |] (f64 [| 2; 2 |] [| 1.; 0.; 0.; 1e-3 |])));
  (* Past [1 / ε] of the format, its own epsilon would leave no singular
     value above the bound. *)
  let eye dt n = Nx.reshape [| 1; n; n |] (Nx.eye dt n) in
  equal ~msg:"bfloat16, 256" (array int) [| 256 |] (ranks (eye Nx.bfloat16 256));
  equal ~msg:"float8_e4m3fn, 16" (array int) [| 16 |] (ranks (eye Nx.float8_e4m3fn 16));
  equal ~msg:"float4_e2m1fn, 4" (array int) [| 4 |] (ranks (eye Nx.float4_e2m1fn 4));
  equal ~msg:"no element" (array int) [| 0; 0 |] (ranks (Nx.zeros Nx.float32 [| 2; 0; 3 |]));
  equal ~msg:"bfloat16" (array int) [| 4 |]
    (ranks (Nx.reshape [| 1; 4; 4 |] (Nx.eye Nx.bfloat16 4)));
  let one x = Nx.item [] x in
  let cond ?p a = one (Nx.cond Nx.float64 ?p a) in
  let ps = [ `One; `Two; `Inf ] in
  let d = f64 [| 2; 2 |] [| 2.; 0.; 0.; 0.5 |] in
  List.iter (fun p -> equal ~msg:"diagonal" (float 1e-12) 4. (cond ~p d)) ps;
  (* [d] with [2i] for [2]: its magnitudes are [d]'s. *)
  let z =
    Nx.create Nx.complex128 [| 2; 2 |]
      Complex.[| { re = 0.; im = 2. }; zero; zero; { re = 0.5; im = 0. } |]
  in
  List.iter (fun p -> equal ~msg:"complex diagonal" (float 1e-12) 4. (cond ~p z)) ps;
  equal ~msg:"identity, bfloat16" float_exact 1.
    (one (Nx.cond Nx.bfloat16 (Nx.eye Nx.bfloat16 3)));
  equal ~msg:"identity, float8_e4m3fn, at float32" float_exact 1.
    (one (Nx.cond Nx.float32 (Nx.eye Nx.float8_e4m3fn 3)));
  equal ~msg:"zero" float_exact Float.infinity (cond (Nx.zeros Nx.float64 [| 2; 2 |]));
  equal ~msg:"singular, one" float_exact Float.infinity
    (cond ~p:`One (f64 [| 2; 2 |] [| 1.; 2.; 2.; 4. |]));
  (* [(1 + i) b] for the singular [b]: its pivot is exactly zero. *)
  let b = f64 [| 2; 2 |] [| 1.; 2.; 2.; 4. |] in
  let singular = Nx.complex Nx.complex64 ~re:b ~im:b in
  List.iter
    (fun p ->
      equal ~msg:"complex singular" float_exact Float.infinity (cond ~p singular))
    [ `One; `Inf ];
  equal ~msg:"complex zero" float_exact Float.infinity
    (cond (Nx.zeros Nx.complex64 [| 2; 2 |]));
  let nan = f64 [| 2; 2 |] [| 1.; Float.nan; 0.; 1. |] in
  List.iter (fun p -> equal ~msg:"NaN" float_exact Float.nan (cond ~p nan)) ps;
  let znan = Nx.complex Nx.complex128 ~re:nan ~im:(Nx.zeros Nx.float64 [| 2; 2 |]) in
  List.iter (fun p -> equal ~msg:"complex NaN" float_exact Float.nan (cond ~p znan)) ps;
  (* A norm that overflows is no infinity the matrix holds. *)
  equal ~msg:"an overflowed norm" float_exact Float.infinity
    (cond ~p:`One (f64 [| 2; 2 |] [| 1e308; 0.; 1e308; 1. |]));
  equal ~msg:"wide" (array int) [| 2 |]
    (Nx.shape (Nx.cond Nx.float32 (Nx.ones Nx.float32 [| 2; 2; 3 |])));
  equal ~msg:"no matrix" (array int) [| 0 |]
    (Nx.shape (Nx.cond Nx.float32 (Nx.zeros Nx.float32 [| 0; 2; 0 |])));
  (* An element of NaN and infinite parts has an infinite magnitude. *)
  let z = Nx.create Nx.complex64 [| 1 |] [| { Complex.re = Float.nan; im = Float.infinity } |] in
  equal ~msg:"an infinite magnitude" float_exact Float.infinity
    (one (Nx.vector_norm Nx.float32 z));
  invalid_by ~by:"Nx.matrix_rank" (fun () -> Nx.matrix_rank (Nx.zeros Nx.float32 [| 3 |]));
  invalid_by ~by:"Nx.matrix_rank" (fun () -> Nx.matrix_rank (Nx.zeros Nx.int32 [| 2; 2 |]));
  invalid_by ~by:"Nx.matrix_rank" (fun () ->
      Nx.matrix_rank ~rtol:(-1.) (Nx.zeros Nx.float32 [| 2; 2 |]));
  List.iter
    (fun (Flt f) ->
      if not (D.float_format f).infinities then
        invalid_by ~by:"Nx.cond" (fun () -> Nx.cond f (Nx.eye Nx.float32 2)))
    float_formats;
  invalid_by ~by:"Nx.cond" (fun () -> Nx.cond Nx.float32 (Nx.zeros Nx.float32 [| 2; 0 |]));
  invalid_by ~by:"Nx.cond" (fun () ->
      Nx.cond Nx.float32 ~p:`One (Nx.ones Nx.float32 [| 2; 3 |]));
  invalid_by ~by:"Nx.cond" (fun () -> Nx.cond Nx.float32 (Nx.eye Nx.int32 2))

(* Precision: a norm computes at the wider of its operand's precision and
   its dtype's, then rounds once to its dtype. *)

type operand = Operand : ('v, 's) D.t -> operand

let pp_operand ppf (Operand dt) = D.pp ppf dt

(* Every float format and both complex ones. *)
let operands =
  List.filter_map
    (fun (D.Any dt) ->
      if D.is D.Float dt || D.is D.Complex dt then Some (Operand dt) else None)
    D.all

type precise =
  | Precise :
      ('v, 's) D.t * (float, 'r) D.t * int array * float array * float array
      -> precise

let precise_case =
  let open Gen in
  with_pp
    (fun ppf (Precise (dt, f, s, _, _)) ->
      Format.fprintf ppf "%a [%s] in %a" D.pp dt
        (String.concat "; " (Array.to_list (Array.map string_of_int s)))
        D.pp f)
    (let* (Operand dt) = of_list ~pp:pp_operand operands in
     let* (Flt f) = of_list float_formats in
     let* s = array ~size:(int_range 2 3) (int_range 0 3) in
     let part =
       array
         ~size:(constant (Array.fold_left ( * ) 1 s))
         (frequency
            [
              (6, float_range (-8.) 8.);
              (1, of_list [ 0.; -0.; 1e4; -1e4; Float.infinity; Float.nan ]);
            ])
     in
     let+ re = part and+ im = part in
     Precise (dt, f, s, re, im))

(* A value of [dt] of shape [s], of real parts [re] and imaginary parts [im]
   where [dt] is complex, each rounded to [dt]. *)
let operand (type v s) (dt : (v, s) D.t) s re im : (v, s, Nx.host) Nx.t =
  let f64 a = Nx.create Nx.float64 s a in
  match D.kind dt with
  | D.Float -> Nx.cast dt (f64 re)
  | D.Complex -> Nx.complex dt ~re:(f64 re) ~im:(f64 im)
  | D.Signed | D.Unsigned | D.Boolean -> invalid_arg "operand"

type wide = Wide : ('v, 's, Nx.host) Nx.t -> wide

(* [x] cast to the dtype of its kind of [float64]'s precision where
   [double], else of [float32]'s. *)
let widened (type v s) double (x : (v, s, Nx.host) Nx.t) =
  match (D.kind (Nx.dtype x), double) with
  | D.Complex, true -> Wide (Nx.cast Nx.complex128 x)
  | D.Complex, false -> Wide (Nx.cast Nx.complex64 x)
  | _, true -> Wide (Nx.cast Nx.float64 x)
  | _, false -> Wide (Nx.cast Nx.float32 x)

(* Each norm of [x] in [f], and the same of [x] cast to the wider precision
   [w] rounded once to [f]: the bits of each, or [None] where it raises. *)
let law_precision (Precise (dt, f, s, re, im)) =
  let x = operand dt s re im in
  let operand64 = (D.is D.Float dt && D.bits dt = 64) || D.bits dt = 128 in
  let double = operand64 || D.bits f = 64 in
  let (Real w) = if double then Real D.Float64 else Real D.Float32 in
  let (Wide wide) = widened double x in
  cover "complex" (D.is D.Complex dt);
  cover "a narrow operand" (D.is D.Float dt && D.bits dt < 32);
  cover "a dtype wider than the operand" (D.bits f = 64 && not operand64);
  cover "a dtype narrower than its computation" (D.bits f < D.bits w);
  let bits y =
    Array.map Int64.bits_of_float (Nx.to_array (Nx.cast Nx.float64 y))
  in
  let outcome g =
    match g () with y -> Some (bits y) | exception Invalid_argument _ -> None
  in
  let same msg
      (g :
        'r 'v 's.
        (float, 'r) D.t -> ('v, 's, Nx.host) Nx.t -> (float, 'r, Nx.host) Nx.t)
      =
    let direct = outcome (fun () -> g f x)
    and rounded = outcome (fun () -> Nx.cast f (g w wide)) in
    cover "computed" (Option.is_some direct);
    equal ~msg (option (array int64)) rounded direct
  in
  List.iter
    (fun ord ->
      same (Printf.sprintf "vector_norm ord %g" ord) (fun d x ->
          Nx.vector_norm d ~ord ~axes:[ -1 ] x))
    [ 2.; 1.; 3.; 0.5; 0.; -1.; -2.; Float.infinity; Float.neg_infinity ];
  List.iter
    (fun (name, ord) -> same name (fun d x -> Nx.matrix_norm d ~ord x))
    [
      ("fro", `Fro); ("nuc", `Nuc); ("one", `One); ("two", `Two);
      ("inf", `Inf); ("negone", `NegOne); ("negtwo", `NegTwo);
      ("neginf", `NegInf);
    ];
  if (D.float_format f).infinities then
    List.iter
      (fun (name, p) -> same name (fun d x -> Nx.cond d ~p x))
      [ ("cond one", `One); ("cond two", `Two); ("cond inf", `Inf) ]

let test_precision =
  prop "a norm is its value at the wider precision, rounded once"
    precise_case law_precision

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
  invalid_by ~by:"Nx.lstsq" (fun () -> Nx.lstsq ~rtol:(-1.) Nx.float64 r (Nx.ones Nx.float64 [| 2 |]));
  invalid_by ~by:"Nx.lstsq" (fun () -> Nx.lstsq ~rtol:Float.nan Nx.float64 r (Nx.ones Nx.float64 [| 2 |]));
  invalid_by ~by:"Nx.svd" (fun () -> Nx.svd Nx.float32 h);
  invalid_by ~by:"Nx.svd" (fun () -> Nx.svd Nx.float64 (Nx.ones Nx.float64 [| 3 |]));
  invalid_by ~by:"Nx.eigh" (fun () -> Nx.eigh Nx.float32 h);
  invalid_by ~by:"Nx.eigh" (fun () -> Nx.eigh Nx.float64 r);
  invalid_by ~by:"Nx.eig" (fun () -> Nx.eig Nx.complex64 h);
  invalid_by ~by:"Nx.eig" (fun () -> Nx.eig Nx.complex128 (Nx.ones Nx.int32 [| 2; 2 |]));
  invalid_by ~by:"Nx.eig" (fun () -> Nx.eig Nx.complex128 r);
  invalid_by ~by:"Nx.slogdet" (fun () -> Nx.slogdet Nx.float32 h);
  invalid_by ~by:"Nx.lstsq" (fun () -> Nx.lstsq Nx.float32 h (Nx.ones Nx.float16 [| 2 |]));
  invalid_by ~by:"Nx.pinv" (fun () -> Nx.pinv ~hermitian:true r);
  invalid_by ~by:"Nx.pinv" (fun () -> Nx.pinv ~hermitian:true (Nx.zeros Nx.float64 [| 0; 3 |]));
  let f64 = Nx.float64 in
  invalid_by ~by:"Nx.vector_norm" (fun () -> Nx.vector_norm f64 ~ord:Float.nan r);
  invalid_by ~by:"Nx.vector_norm" (fun () -> Nx.vector_norm f64 ~axes:[ 2 ] r);
  invalid_by ~by:"Nx.vector_norm" (fun () -> Nx.vector_norm f64 ~axes:[ 0; 0 ] r);
  invalid_by ~by:"Nx.vector_norm" (fun () ->
      Nx.vector_norm f64 ~ord:Float.infinity (Nx.zeros Nx.float64 [| 0 |]));
  invalid_by ~by:"Nx.vector_norm" (fun () ->
      Nx.vector_norm f64 ~ord:Float.neg_infinity (Nx.zeros Nx.float64 [| 2; 0 |]));
  invalid_by ~by:"Nx.vector_norm" (fun () -> Nx.vector_norm f64 (Nx.ones Nx.int32 [| 3 |]));
  invalid_by ~by:"Nx.matrix_norm" (fun () -> Nx.matrix_norm f64 (Nx.ones Nx.float64 [| 3 |]));
  invalid_by ~by:"Nx.matrix_norm" (fun () ->
      Nx.matrix_norm f64 ~ord:`Two (Nx.zeros Nx.float64 [| 2; 0 |]));
  invalid_by ~by:"Nx.matrix_norm" (fun () ->
      Nx.matrix_norm f64 ~ord:`NegTwo (Nx.zeros Nx.complex64 [| 0; 2 |]));
  invalid_by ~by:"Nx.matrix_norm" (fun () ->
      Nx.matrix_norm f64 ~ord:`One (Nx.zeros Nx.float64 [| 2; 0 |]));
  invalid_by ~by:"Nx.matrix_norm" (fun () ->
      Nx.matrix_norm f64 ~ord:`NegInf (Nx.zeros Nx.float16 [| 0; 2 |]));
  invalid_by ~by:"Nx.matrix_norm" (fun () ->
      Nx.matrix_norm f64 (Nx.ones Nx.bool [| 2; 2 |]))

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
  invalid_by ~by:"Nx.lstsq" (fun () -> Nx.lstsq Nx.float64 a (on [| 2 |]));
  invalid_by ~by:"Nx.svd" (fun () -> Nx.svd Nx.float32 a);
  invalid_by ~by:"Nx.eigh" (fun () -> Nx.eigh Nx.float32 a);
  invalid_by ~by:"Nx.eig" (fun () -> Nx.eig Nx.complex64 a);
  invalid_by ~by:"Nx.pinv" (fun () -> Nx.pinv a);
  invalid_by ~by:"Nx.pinv" (fun () -> Nx.pinv ~hermitian:true a);
  invalid_by ~by:"Nx.det" (fun () -> Nx.det a);
  invalid_by ~by:"Nx.slogdet" (fun () -> Nx.slogdet Nx.float64 a);
  invalid_by ~by:"Nx.matrix_norm" (fun () -> Nx.matrix_norm Nx.float64 ~ord:`Nuc a);
  invalid_by ~by:"Nx.matrix_norm" (fun () -> Nx.matrix_norm Nx.float64 ~ord:`Two a);
  invalid_by ~by:"Nx.cond" (fun () -> Nx.cond Nx.float64 a)

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
  invalid_by ~by:"Nx.eigh" (fun () ->
      Nx.eigh ~triangle:Upper Nx.float64 (square ()));
  invalid_by ~by:"Nx.solve_triangular" (fun () ->
      Nx.solve_triangular ~transpose:Transpose (Nx.eye Nx.complex64 2)
        (dead Nx.complex64 [| 2; 1 |] Complex.one))

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
             test_rounded_once;
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
             test_precision;
             law_matrix_rank;
             law_cond;
             test "rank and condition at their edges" test_rank_cond_cases;
           ];
         group "refusals"
           [
             test "each refusal names its function" test_refusals;
             test "a composite names its function" test_composites_named;
             test "a dead operand names the function called" test_dead_named;
           ];
       ])
