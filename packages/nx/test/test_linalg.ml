(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Products against their definitions, written as loops over the reference;
   factorizations and solvers by the identities that define them. *)

open Windtrap
open Nx_test

let pp_float ppf x = Format.fprintf ppf "%.17g" x
let near = tensor (close ~rel:1e-9 ~abs:1e-9 ())
let near_ref = Ref.witness (close ~rel:1e-9 ~abs:1e-9 ())

(* Well scaled: a matrix of subnormals and huge ratios is a question of the
   solvers' scaling, which these identities do not ask. *)
let entry =
  Gen.map (fun x -> Float.round (x *. 1e6) /. 1e6) (Gen.float_range (-1.) 1.)

let dim = Gen.int_range 1 5
let batch = Gen.of_list ~pp:pp_shape [ [||]; [| 2 |]; [| 2; 1 |]; [| 1; 3 |] ]

let matrix ?(batch = Gen.constant ~pp:pp_shape [||]) m n =
  let open Gen in
  let* b = batch in
  let shape = Array.append b [| m; n |] in
  let+ xs = array ~size:(constant (Ref.numel shape)) entry in
  Nx.create Nx.float64 shape xs

let t = Nx.matrix_transpose
let ( *@ ) = Nx.matmul

(* The identity of the last two axes of [a]. *)
let identity_like a =
  let s = Nx.shape a in
  let n = s.(Array.length s - 1) in
  Nx.broadcast_to
    (Array.append (Array.sub s 0 (Array.length s - 2)) [| n; n |])
    (Nx.eye Nx.float64 n)

(* Well conditioned: dominated by its diagonal. *)
let square ?(batch = batch) n =
  Gen.map
    (fun a -> Nx.add a (Nx.mul_s (identity_like a) (Float.of_int (n + 2))))
    (matrix ~batch n n)

let plain = Gen.constant ~pp:pp_shape [||]

let spd n =
  Gen.map
    (fun a -> Nx.add (t a *@ a) (Nx.mul_s (identity_like a) 1.))
    (matrix ~batch n n)

let symmetric n =
  Gen.map (fun a -> Nx.mul_s (Nx.add a (t a)) 0.5) (matrix ~batch n n)

let sized f = Gen.bind dim f

(* Reference products *)

let matmul_ref (a : float Ref.t) (b : float Ref.t) =
  let a1 = Ref.ndim a = 1 and b1 = Ref.ndim b = 1 in
  let a = if a1 then Ref.reshape [| 1; a.shape.(0) |] a else a in
  let b = if b1 then Ref.reshape [| b.shape.(0); 1 |] b else b in
  let ra = Ref.ndim a and rb = Ref.ndim b in
  let m = a.shape.(ra - 2) and k = a.shape.(ra - 1) and n = b.shape.(rb - 1) in
  if b.shape.(rb - 2) <> k then invalid_arg "matmul";
  let batch =
    Ref.broadcast_shapes
      (Array.sub a.shape 0 (ra - 2))
      (Array.sub b.shape 0 (rb - 2))
  in
  let nb = Array.length batch in
  let a = Ref.broadcast_to (Array.append batch [| m; k |]) a in
  let b = Ref.broadcast_to (Array.append batch [| k; n |]) b in
  let out =
    Ref.init
      (Array.append batch [| m; n |])
      (fun idx ->
        let s = ref 0. in
        for p = 0 to k - 1 do
          let ia = Array.copy idx and ib = Array.copy idx in
          ia.(nb + 1) <- p;
          ib.(nb) <- p;
          s := !s +. (Ref.get a ia *. Ref.get b ib)
        done;
        !s)
  in
  let out = if a1 then Ref.squeeze ~axes:[ nb ] out else out in
  if b1 then Ref.squeeze ~axes:[ Ref.ndim out - 1 ] out else out

(* Einstein summation by brute force: every output position sums, over every
   value of the letters the output drops, the product of the operands. [...]
   stands for the same trailing letters in every operand. *)
let einsum_ref spec (ops : float Ref.t array) =
  let inputs, output =
    match String.index_opt spec '-' with
    | Some i ->
        ( String.sub spec 0 i,
          Some (String.sub spec (i + 2) (String.length spec - i - 2)) )
    | None -> (spec, None)
  in
  let inputs = Array.of_list (String.split_on_char ',' inputs) in
  let letters s =
    List.filter (fun c -> c <> '.') (List.of_seq (String.to_seq s))
  in
  let ell_rank =
    Array.fold_left max 0
      (Array.mapi
         (fun i s ->
           if String.length s >= 3 && String.contains s '.' then
             Ref.ndim ops.(i) - List.length (letters s)
           else 0)
         inputs)
  in
  let ell = List.init ell_rank (fun i -> Char.chr (Char.code 'A' + i)) in
  let expand s rank =
    let named = letters s in
    if String.contains s '.' then
      let k = rank - List.length named in
      let pre = List.filteri (fun i _ -> i >= ell_rank - k) ell in
      let idx = String.index s '.' in
      letters (String.sub s 0 idx)
      @ pre
      @ letters (String.sub s (idx + 3) (String.length s - idx - 3))
    else named
  in
  let subs = Array.mapi (fun i s -> expand s (Ref.ndim ops.(i))) inputs in
  let size = Hashtbl.create 8 in
  Array.iteri
    (fun i l ->
      List.iteri (fun d c -> Hashtbl.replace size c ops.(i).shape.(d)) l)
    subs;
  let all = List.sort_uniq compare (List.concat (Array.to_list subs)) in
  let out =
    match output with
    | Some o -> expand o (ell_rank + List.length (letters o))
    | None ->
        let once c =
          List.length (List.filter (( = ) c) (List.concat (Array.to_list subs)))
          = 1
        in
        ell
        @ List.filter
            (fun c -> once c && not (List.mem c ell))
            (List.sort compare all)
  in
  let summed = List.filter (fun c -> not (List.mem c out)) all in
  let dims l = Array.of_list (List.map (Hashtbl.find size) l) in
  Ref.init (dims out) (fun oi ->
      let value = Hashtbl.create 8 in
      List.iteri (fun d c -> Hashtbl.replace value c oi.(d)) out;
      let s = ref 0. in
      let sd = dims summed in
      for k = 0 to Ref.numel sd - 1 do
        let si = Ref.unravel sd k in
        List.iteri (fun d c -> Hashtbl.replace value c si.(d)) summed;
        s :=
          !s
          +. Array.fold_left ( *. ) 1.
               (Array.mapi
                  (fun i l ->
                    Ref.get ops.(i)
                      (Array.of_list (List.map (Hashtbl.find value) l)))
                  subs)
      done;
      !s)

let products =
  let pair_shapes =
    Gen.of_list
      ~pp:(fun ppf (a, b) -> Format.fprintf ppf "%a @ %a" pp_shape a pp_shape b)
      [
        ([| 3 |], [| 3 |]);
        ([| 3 |], [| 3; 2 |]);
        ([| 2; 3 |], [| 3 |]);
        ([| 2; 3 |], [| 3; 4 |]);
        ([| 2; 2; 3 |], [| 3; 4 |]);
        ([| 3; 4 |], [| 2; 4; 1 |]);
        ([| 2; 1; 2; 3 |], [| 3; 3; 2 |]);
        ([| 0; 3 |], [| 3; 2 |]);
        ([| 2; 0 |], [| 0; 3 |]);
      ]
  in
  let operands =
    Gen.bind pair_shapes (fun (sa, sb) ->
        let draw s =
          Gen.map (Nx.create Nx.float64 s)
            (Gen.array ~size:(Gen.constant (Ref.numel s)) entry)
        in
        Gen.pair (draw sa) (draw sb))
  in
  group "products"
    [
      prop
        "matmul broadcasts its batch axes and treats vectors as rows and \
         columns"
        operands (fun (a, b) ->
          equal near_ref
            (matmul_ref (Ref.of_nx a) (Ref.of_nx b))
            (Ref.of_nx (Nx.matmul a b)));
      prop "matmul reads every layout"
        (Gen.pair (sized (fun n -> matrix n n)) layout)
        (fun (a, steps) ->
          let v = lay_out steps a in
          assume (Nx.ndim v = 2 && Nx.dim 0 v = Nx.dim 1 v);
          equal near (Nx.contiguous v *@ Nx.contiguous v) (v *@ v));
      test "matmul refuses scalars and mismatched inner axes" (fun () ->
          raises_invalid_arg (fun () ->
              Nx.matmul (Nx.scalar Nx.float64 1.) (Nx.ones Nx.float64 [| 2 |]));
          raises_invalid_arg (fun () ->
              Nx.matmul
                (Nx.ones Nx.float64 [| 2; 3 |])
                (Nx.ones Nx.float64 [| 2; 3 |])));
      prop
        "dot contracts the last axis of a with the second-to-last of b, \
         keeping both batches"
        (Gen.pair
           (matrix ~batch:(Gen.constant ~pp:pp_shape [| 2 |]) 2 3)
           (matrix ~batch:(Gen.constant ~pp:pp_shape [| 4 |]) 3 2))
        (fun (a, b) ->
          equal near_ref
            (einsum_ref "aij,bjk->aibk" [| Ref.of_nx a; Ref.of_nx b |])
            (Ref.of_nx (Nx.dot a b)));
      prop "vdot, vecdot, inner, outer and kron are their sums of products"
        (Gen.pair (matrix 2 3) (matrix 2 3))
        (fun (a, b) ->
          let ra = Ref.of_nx a and rb = Ref.of_nx b in
          equal near_ref
            (einsum_ref "ij,ij->" [| ra; rb |])
            (Ref.of_nx (Nx.vdot a b));
          equal near_ref
            (einsum_ref "ij,ij->i" [| ra; rb |])
            (Ref.of_nx (Nx.vecdot a b));
          equal near_ref
            (einsum_ref "ij,kj->ik" [| ra; rb |])
            (Ref.of_nx (Nx.inner a b));
          equal near_ref
            (einsum_ref "i,j->ij"
               [| Ref.reshape [| 6 |] ra; Ref.reshape [| 6 |] rb |])
            (Ref.of_nx (Nx.outer a b));
          equal near_ref
            (Ref.reshape [| 4; 9 |]
               (Ref.transpose ~axes:[ 0; 2; 1; 3 ]
                  (einsum_ref "ij,kl->ijkl" [| ra; rb |])))
            (Ref.of_nx (Nx.kron a b)));
      prop "tensordot contracts the named axes"
        (Gen.pair
           (matrix ~batch:(Gen.constant ~pp:pp_shape [| 2 |]) 3 4)
           (matrix 4 2))
        (fun (a, b) ->
          equal near_ref
            (einsum_ref "aij,jk->aik" [| Ref.of_nx a; Ref.of_nx b |])
            (Ref.of_nx (Nx.tensordot ~axes:([ 2 ], [ 0 ]) a b)));
      prop "multi_dot is the chained product, and matrix_power repeated product"
        (sized (fun n ->
             Gen.triple (matrix n n) (matrix n n) (square ~batch:plain n)))
        (fun (a, b, c) ->
          equal near (a *@ b *@ c) (Nx.multi_dot [| a; b; c |]);
          equal near (identity_like c) (Nx.matrix_power c 0);
          equal near (c *@ c *@ c) (Nx.matrix_power c 3);
          equal near (Nx.inv (c *@ c)) (Nx.matrix_power c (-2)));
      prop "cross is the cross product of three-vectors"
        (Gen.pair (matrix 2 3) (matrix 2 3))
        (fun (a, b) ->
          let ra = Ref.of_nx a and rb = Ref.of_nx b in
          let c =
            Ref.init [| 2; 3 |] (fun i ->
                let g r j = Ref.get r [| i.(0); j |] in
                let j = i.(1) in
                let u = (j + 1) mod 3 and v = (j + 2) mod 3 in
                (g ra u *. g rb v) -. (g ra v *. g rb u))
          in
          equal near_ref c (Ref.of_nx (Nx.cross a b)));
      prop
        "diagonal, trace and matrix_transpose read and swap the last two axes"
        (Gen.pair
           (sized (fun m -> sized (fun n -> matrix ~batch m n)))
           (Gen.int_range (-3) 3))
        (fun (a, offset) ->
          let r = Ref.of_nx a in
          let n = Ref.ndim r in
          equal near_ref
            (Ref.transpose
               ~axes:
                 (List.init n (fun i ->
                      if i = n - 2 then n - 1
                      else if i = n - 1 then n - 2
                      else i))
               r)
            (Ref.of_nx (t a));
          let m, k = (r.shape.(n - 2), r.shape.(n - 1)) in
          let len =
            Int.max 0
              (if offset >= 0 then Int.min m (k - offset)
               else Int.min (m + offset) k)
          in
          let d =
            Ref.init
              (Array.append (Array.sub r.shape 0 (n - 2)) [| len |])
              (fun i ->
                let b = Array.sub i 0 (n - 2) and p = i.(n - 2) in
                let row, col =
                  if offset >= 0 then (p, p + offset) else (p - offset, p)
                in
                Ref.get r (Array.append b [| row; col |]))
          in
          equal near_ref d
            (Ref.of_nx (Nx.diagonal ~offset ~axis1:(n - 2) ~axis2:(n - 1) a));
          if n = 2 then
            equal near_ref
              (Ref.reduce ~axes:[ 0 ] ( +. ) 0. d)
              (Ref.of_nx (Nx.trace ~offset a)));
      test "the products refuse mismatched axes" (fun () ->
          let v n = Nx.ones Nx.float64 [| n |] in
          raises_invalid_arg (fun () -> Nx.vdot (v 2) (v 3));
          raises_invalid_arg (fun () -> Nx.inner (v 2) (v 3));
          raises_invalid_arg (fun () -> Nx.dot (v 2) (v 3));
          raises_invalid_arg (fun () -> Nx.cross (v 2) (v 2));
          raises_invalid_arg (fun () ->
              Nx.tensordot ~axes:([ 0 ], [ 0 ]) (v 2) (v 3));
          raises_invalid_arg (fun () -> Nx.multi_dot [||]);
          raises_invalid_arg (fun () -> Nx.trace (v 2)));
    ]

let einsums =
  let specs =
    [
      "i,i->";
      "ij,jk->ik";
      "ij->ji";
      "i,j->ij";
      "ij->";
      "ii->i";
      "ii";
      "...ii->...i";
      "abnb->an";
      "aaab->ab";
      "abiib->ab";
      "abnb,bn->an";
      "...ij,...jk->...ik";
      "i,jk->jki";
      "ij,klj->kli";
      "abc,bd->dac";
      "ab,bc,cd->ad";
      "i,i->i";
      "ij,ij->";
      "ijk,k->ij";
      "ab,b->a";
      "...i,...i->...";
      "i...->...i";
      "ij,j->i";
      "ab,cd->";
      "ij,kj->";
      "ab,cd->ac";
      "ij->i";
      "ij->j";
    ]
  in
  (* The shapes of a spec's operands, from one size per letter and per position
     of [...]. *)
  let operands spec =
    let open Gen in
    let* sizes = array ~size:(constant 26) (int_range 1 3) in
    let+ ell = array ~size:(int_range 0 2) (int_range 1 3) in
    let inputs =
      match String.index_opt spec '-' with
      | Some i -> String.sub spec 0 i
      | None -> spec
    in
    Array.of_list
      (List.map
         (fun s ->
           let shape =
             List.concat_map
               (fun c ->
                 if c = '.' then [] else [ sizes.(Char.code c - Char.code 'a') ])
               (List.of_seq (String.to_seq s))
           in
           let shape =
             if String.contains s '.' then
               let i = String.index s '.' in
               let pre =
                 List.length
                   (List.filter (( <> ) '.')
                      (List.of_seq (String.to_seq (String.sub s 0 i))))
               in
               List.filteri (fun k _ -> k < pre) shape
               @ Array.to_list ell
               @ List.filteri (fun k _ -> k >= pre) shape
             else shape
           in
           let shape = Array.of_list shape in
           Nx.init Nx.float64 shape (fun i ->
               Float.of_int (Ref.ravel shape i mod 7) -. 3.))
         (String.split_on_char ',' inputs))
  in
  group "einsum"
    [
      prop
        "einsum sums, over the letters its output drops, the product of its \
         operands"
        (Gen.bind (Gen.of_list ~pp:Format.pp_print_string specs) (fun spec ->
             Gen.map (fun ops -> (spec, ops)) (operands spec)))
        (fun (spec, ops) ->
          equal near_ref
            (einsum_ref spec (Array.map Ref.of_nx ops))
            (Ref.of_nx (Nx.einsum spec ops)));
      cases "einsum refuses a malformed subscript" ~name:Fun.id
        [ "ij,jk,kl->il"; "ij->ik"; "ij,jk->iik"; "ij,jk->i-k" ] (fun spec ->
          raises_invalid_arg (fun () ->
              Nx.einsum spec
                [|
                  Nx.ones Nx.float64 [| 2; 2 |]; Nx.ones Nx.float64 [| 2; 2 |];
                |]));
    ]

(* Factorizations reconstruct their matrix and have the structure they
   promise. *)

let is_upper r =
  let r = Ref.of_nx r in
  let n = Ref.ndim r in
  Array.for_all Fun.id
    (Array.mapi
       (fun k v ->
         let i = Ref.unravel r.shape k in
         v = 0. || i.(n - 1) >= i.(n - 2))
       r.data)

let factorizations =
  group "factorizations"
    [
      prop "cholesky gives L with L Lᵀ = a, reading only the lower triangle"
        (sized spd) (fun a ->
          let l = Nx.cholesky a in
          equal near a (l *@ t l);
          is_true ~msg:"L is lower-triangular" (is_upper (t l));
          equal near l
            (Nx.cholesky
               (Nx.add (Nx.tril a) (Nx.triu ~k:1 (Nx.full_like a 7.))));
          let u = Nx.cholesky ~upper:true a in
          equal near a (t u *@ u));
      test "cholesky refuses a matrix that is not positive definite" (fun () ->
          raises_match
            (function
              | Nx.Linalg_error { kind = `Not_positive_definite; _ } -> true
              | _ -> false)
            (fun () -> Nx.cholesky (Nx.neg (Nx.eye Nx.float64 2))));
      prop "qr gives an orthonormal Q and an upper-triangular R with Q R = a"
        (Gen.pair
           (sized (fun m -> sized (fun n -> matrix ~batch m n)))
           Gen.bool)
        (fun (a, complete) ->
          let mode = if complete then `Complete else `Reduced in
          let q, r = Nx.qr ~mode a in
          equal near a (q *@ r);
          equal near (identity_like q) (t q *@ q);
          is_true ~msg:"R is upper-triangular" (is_upper r));
      prop
        "svd gives orthonormal U and Vh and descending S with U diag S Vh = a"
        (sized (fun m -> sized (fun n -> matrix ~batch m n)))
        (fun a ->
          let u, s, vh = Nx.svd a in
          equal near a (Nx.mul u (Nx.unsqueeze ~axes:[ -2 ] s) *@ vh);
          equal near (identity_like u) (t u *@ u);
          equal near (identity_like (t vh)) (vh *@ t vh);
          equal ~msg:"S descends" near
            (fst (Nx.sort ~descending:true ~axis:(-1) s))
            s;
          equal ~msg:"S is not negative" (tensor bool)
            (Nx.ones_like (Nx.greater_equal_s s 0.))
            (Nx.greater_equal_s s 0.);
          equal near s (Nx.svdvals a));
      prop
        "eigh gives ascending eigenvalues and orthonormal vectors with a V = V \
         diag w"
        (sized symmetric) (fun a ->
          let w, v = Nx.eigh a in
          equal near (a *@ v) (Nx.mul v (Nx.unsqueeze ~axes:[ -2 ] w));
          equal near (identity_like v) (t v *@ v);
          equal near w (Nx.eigvalsh a));
      prop "eig gives complex pairs with a V = V diag w"
        (sized (fun n -> matrix n n))
        (fun a ->
          let w, v = Nx.eig a in
          let ac = Nx.cast Nx.complex128 a in
          let c = close ~rel:1e-8 ~abs:1e-8 () in
          equal
            (tensor
               (Testable.contramap
                  (fun (z : Complex.t) -> (z.re, z.im))
                  (pair c c)))
            (Nx.matmul ac v)
            (Nx.mul v (Nx.unsqueeze ~axes:[ 0 ] w)));
      test "the factorizations refuse integers and non-square matrices"
        (fun () ->
          raises_invalid_arg (fun () -> Nx.qr (Nx.ones Nx.int32 [| 2; 2 |]));
          raises_invalid_arg (fun () ->
              Nx.cholesky (Nx.ones Nx.float64 [| 2; 3 |]));
          raises_invalid_arg (fun () -> Nx.eigh (Nx.ones Nx.float64 [| 2; 3 |])));
    ]

let invariants =
  group "norms and invariants"
    [
      xfail ~reason:"det computes through slogdet, which rounds to float32"
        (prop "det is multiplicative at float64"
           (sized (fun n -> Gen.pair (square n) (square n)))
           (fun (a, b) ->
             equal near (Nx.mul (Nx.det a) (Nx.det b)) (Nx.det (a *@ b))));
      xfail
        ~reason:"det takes its sign from R alone and drops the sign of det Q"
        (test "det of an odd permutation is -1" (fun () ->
             let swap =
               Nx.create Nx.float64 [| 3; 3 |]
                 [| 0.; 1.; 0.; 1.; 0.; 0.; 0.; 0.; 1. |]
             in
             equal (close ~rel:0. ()) (-1.) (Nx.item [] (Nx.det swap))));
      prop "slogdet is det's sign and log magnitude, at float32"
        (sized (fun n -> Gen.pair (square n) (square n)))
        (fun (a, _) ->
          (* slogdet gives float32, whatever the dtype. *)
          let sign, logabs = Nx.slogdet a in
          equal
            (tensor (close ~rel:1e-6 ~abs:1e-6 ()))
            (Nx.det a)
            (Nx.mul (Nx.cast Nx.float64 sign)
               (Nx.exp (Nx.cast Nx.float64 logabs))));
      prop "vector norms are their sums of powers" (matrix 1 5) (fun v ->
          let x = Array.map Float.abs (Nx.to_array v) in
          let sum f = Array.fold_left (fun s e -> s +. f e) 0. x in
          let check ord expected =
            equal (close ~rel:1e-9 ()) expected
              (Nx.item [] (Nx.norm ~ord (Nx.reshape [| 5 |] v)))
          in
          check `One (sum Fun.id);
          check `Two (Float.sqrt (sum (fun e -> e *. e)));
          check `Inf (Array.fold_left Float.max 0. x);
          check (`P 3.) (Float.cbrt (sum (fun e -> e *. e *. e))));
      prop "matrix norms are their row, column and singular-value sums"
        (sized (fun m -> sized (fun n -> matrix m n)))
        (fun a ->
          let r = Ref.map Float.abs (Ref.of_nx a) in
          let s = Nx.to_array (Nx.svdvals a) in
          let check ord expected =
            equal
              (close ~rel:1e-9 ~abs:1e-12 ())
              expected
              (Nx.item [] (Nx.norm ~ord a))
          in
          let most l = Array.fold_left Float.max 0. l.Ref.data in
          check `One (most (Ref.reduce ~axes:[ 0 ] ( +. ) 0. r));
          check `Inf (most (Ref.reduce ~axes:[ 1 ] ( +. ) 0. r));
          check `Fro
            (Float.sqrt (Array.fold_left (fun t e -> t +. (e *. e)) 0. r.data));
          check `Two s.(0);
          check `Nuc (Array.fold_left ( +. ) 0. s));
      prop "cond is the norm of a times the norm of its inverse"
        (sized (square ~batch:plain))
        (fun a ->
          equal (close ~rel:1e-8 ())
            (Nx.item []
               (Nx.mul (Nx.norm ~ord:`One a) (Nx.norm ~ord:`One (Nx.inv a))))
            (Nx.item [] (Nx.cond ~p:`One a)));
      prop "matrix_rank counts the independent columns of a product of rank k"
        (sized (fun m ->
             sized (fun n ->
                 Gen.bind
                   (Gen.int_range 0 (Int.min m n))
                   (fun k ->
                     Gen.map
                       (fun (l, r) -> (k, l, r))
                       (Gen.pair (matrix m k) (matrix k n))))))
        (fun (k, l, r) ->
          (* A dominant diagonal makes each factor of rank k. *)
          let full x =
            Nx.add x
              (Nx.mul_s (Nx.eye ~m:(Nx.dim 1 x) Nx.float64 (Nx.dim 0 x)) 3.)
          in
          equal int k (Nx.matrix_rank (full l *@ full r)));
    ]

let solvers =
  group "solvers"
    [
      prop
        "solve gives x with a x = b, for stacked right-hand sides sharing a's \
         batch"
        (sized (fun n ->
             Gen.bind batch (fun bt ->
                 let batch = Gen.constant ~pp:pp_shape bt in
                 Gen.pair (square ~batch n) (matrix ~batch n 2))))
        (fun (a, b) -> equal near b (a *@ Nx.solve a b));
      prop
        "solve takes a right-hand side of one fewer axis as a stack of vectors"
        (sized (fun n -> Gen.pair (square ~batch:plain n) (matrix 1 n)))
        (fun (a, b) ->
          let v = Nx.reshape [| Nx.dim 1 b |] b in
          equal near v (Nx.matmul a (Nx.solve a v)));
      prop "inv gives the inverse" (sized square) (fun a ->
          equal near (identity_like a) (a *@ Nx.inv a));
      test "solve and inv refuse a singular matrix" (fun () ->
          let singular = Nx.create Nx.float64 [| 2; 2 |] [| 1.; 2.; 2.; 4. |] in
          let is_singular = function
            | Nx.Linalg_error { kind = `Singular; _ } -> true
            | _ -> false
          in
          raises_match is_singular (fun () ->
              Nx.solve singular (Nx.ones Nx.float64 [| 2 |]));
          raises_match is_singular (fun () -> Nx.inv singular);
          raises_match is_singular (fun () -> Nx.matrix_power singular (-1)));
      prop "solve_triangular solves with the named triangle only"
        (sized (fun n ->
             Gen.bind batch (fun bt ->
                 let batch = Gen.constant ~pp:pp_shape bt in
                 Gen.triple (square ~batch n) (matrix ~batch n 2)
                   (Gen.pair Gen.bool Gen.bool))))
        (fun (a, b, (upper, transpose)) ->
          let tri = if upper then Nx.triu a else Nx.tril a in
          let noisy =
            Nx.add tri
              (if upper then Nx.tril ~k:(-1) (Nx.full_like a 9.)
               else Nx.triu ~k:1 (Nx.full_like a 9.))
          in
          let solved a x = equal near b (a *@ x) in
          solved
            (if transpose then t tri else tri)
            (Nx.solve_triangular ~upper ~transpose noisy b);
          let unit =
            Nx.add (Nx.sub tri (Nx.mul tri (identity_like a))) (identity_like a)
          in
          solved unit (Nx.solve_triangular ~upper ~unit_diag:true noisy b));
      prop "lstsq gives x with residual orthogonal to the columns, and the rank"
        (Gen.pair (Gen.int_range 1 3) (Gen.int_range 0 3))
        (fun (n, extra) ->
          let m = n + extra in
          let a =
            Nx.init Nx.float64 [| m; n |] (fun i ->
                Float.cos (Float.of_int ((5 * i.(0)) + (11 * i.(1)) + 1))
                +. if i.(0) = i.(1) then 3. else 0.)
          in
          let b =
            Nx.init Nx.float64 [| m; 2 |] (fun i ->
                Float.of_int ((i.(0) * 2) + i.(1)))
          in
          let x, _, rank, _ = Nx.lstsq a b in
          equal int n rank;
          equal near (Nx.zeros Nx.float64 [| n; 2 |]) (t a *@ Nx.sub (a *@ x) b));
      prop "pinv meets the four Moore-Penrose conditions"
        (sized (fun m -> sized (fun n -> matrix m n)))
        (fun a ->
          let p = Nx.pinv a in
          equal near a (a *@ p *@ a);
          equal near p (p *@ a *@ p);
          equal near (t (a *@ p)) (a *@ p);
          equal near (t (p *@ a)) (p *@ a));
      test
        "tensorsolve inverts tensordot, and tensorinv is inv of the matricised \
         tensor" (fun () ->
          let m =
            Nx.add (Nx.eye Nx.float64 6)
              (Nx.mul_s (Nx.ones Nx.float64 [| 6; 6 |]) 0.1)
          in
          let a = Nx.reshape [| 2; 3; 6 |] m in
          let b =
            Nx.init Nx.float64 [| 2; 3 |] (fun i ->
                Float.of_int (i.(0) + (3 * i.(1))))
          in
          equal near b
            (Nx.tensordot ~axes:([ 2 ], [ 0 ]) a (Nx.tensorsolve a b));
          equal near
            (Nx.reshape [| 6; 2; 3 |] (Nx.inv m))
            (Nx.tensorinv ~ind:2 a));
    ]

(* float16 and bfloat16 products widen to float32 and round once. *)
let narrow =
  group "narrow floats"
    [
      prop "a float16 or bfloat16 matmul is float32's, rounded once"
        (Gen.pair (matrix 3 5) (matrix 5 2))
        (fun (a, b) ->
          let once dt =
            Nx.cast dt
              (Nx.matmul
                 (Nx.cast Nx.float32 (Nx.cast dt a))
                 (Nx.cast Nx.float32 (Nx.cast dt b)))
          in
          equal
            (tensor (close ~rel:0. ()))
            (once Nx.float16)
            (Nx.matmul (Nx.cast Nx.float16 a) (Nx.cast Nx.float16 b));
          equal
            (tensor (close ~rel:0. ()))
            (once Nx.bfloat16)
            (Nx.matmul (Nx.cast Nx.bfloat16 a) (Nx.cast Nx.bfloat16 b)));
    ]

let () =
  exit
    (run "nx linalg"
       [ products; einsums; factorizations; invariants; solvers; narrow ])
