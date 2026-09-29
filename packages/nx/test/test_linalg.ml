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
          let c = Nx.contiguous v in
          equal near (c *@ t c) (v *@ t v));
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

(* Complex tensors compare by their parts. *)
let near_complex =
  let c = close ~rel:1e-9 ~abs:1e-9 () in
  tensor (Testable.contramap (fun (z : Complex.t) -> (z.re, z.im)) (pair c c))

let complex_matrix ?(batch = Gen.constant ~pp:pp_shape [||]) m n =
  Gen.bind batch (fun bt ->
      let batch = Gen.constant ~pp:pp_shape bt in
      Gen.map
        (fun (re, im) -> Nx.complex Nx.complex128 ~re ~im)
        (Gen.pair (matrix ~batch m n) (matrix ~batch m n)))

(* P is a permutation matrix: zeros and ones, one 1 in each row and column. *)
(* Every lane of [perm] orders its rows once each. *)
let is_permutation perm =
  let r = Ref.of_nx perm in
  let n = Ref.ndim r in
  let m = r.shape.(n - 1) in
  let sorted =
    Ref.along ~axis:(n - 1) ~length:m
      (fun l ->
        let l = Array.copy l in
        Array.sort compare l;
        l)
      r
  in
  Array.for_all
    (fun k -> Int32.to_int sorted.data.(k) = k mod m)
    (Array.init (Array.length r.data) Fun.id)

(* The rows of [a] in the order [perm], lane by lane. *)
let rows perm a =
  Nx.take_along_axis ~axis:(-2)
    ~indices:(Nx.broadcast_to (Nx.shape a) (Nx.unsqueeze ~axes:[ -1 ] perm))
    a

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
      prop "cholesky of a Hermitian matrix gives L with L Lᴴ = a" (sized spd)
        (fun a ->
          let z =
            Nx.complex Nx.complex128 ~re:a
              ~im:
                (Nx.sub
                   (Nx.triu ~k:1 (Nx.mul_s a 0.1))
                   (t (Nx.triu ~k:1 (Nx.mul_s a 0.1))))
          in
          let l = Nx.cholesky z in
          let c = close ~rel:1e-9 ~abs:1e-9 () in
          equal
            (tensor
               (Testable.contramap
                  (fun (w : Complex.t) -> (w.re, w.im))
                  (pair c c)))
            z
            (l *@ Nx.conjugate (t l)));
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
        "lu gives a row order, a unit lower-triangular l with entries of \
         magnitude at most 1, and an upper-triangular u whose product is a's \
         rows in that order"
        (sized (fun m -> sized (fun n -> matrix ~batch m n)))
        (fun a ->
          let p, l, u = Nx.lu a in
          equal near (rows p a) (l *@ u);
          let p32, l32, u32 = Nx.lu (Nx.cast Nx.float32 a) in
          equal ~msg:"at float32"
            (tensor (close ~rel:1e-5 ~abs:1e-5 ()))
            (rows p32 (Nx.cast Nx.float32 a))
            (Nx.matmul l32 u32);
          is_true ~msg:"the order is a permutation" (is_permutation p);
          is_true ~msg:"L is lower-triangular" (is_upper (t l));
          equal ~msg:"L has a unit diagonal" near
            (Nx.ones_like (Nx.diagonal l))
            (Nx.diagonal l);
          is_true ~msg:"L's entries are at most 1 in magnitude"
            (Array.for_all
               (fun x -> Float.abs x <= 1.)
               (Nx.to_array (Nx.contiguous l)));
          is_true ~msg:"U is upper-triangular" (is_upper u));
      prop "lu reads every layout"
        (Gen.pair (sized (fun m -> sized (fun n -> matrix ~batch m n))) layout)
        (fun (a, steps) ->
          let v = lay_out steps a in
          assume (Nx.ndim v >= 2);
          let same x y =
            let p, l, u = Nx.lu x and p', l', u' = Nx.lu y in
            let exact = tensor (close ~rel:0. ()) in
            equal (tensor int32) p p';
            equal exact l l';
            equal exact u u'
          in
          same (Nx.contiguous v) v;
          let v32 = Nx.cast Nx.float32 v in
          same (Nx.contiguous v32) v32);
      prop "lu of a complex matrix gives l u = a's rows in its order"
        (sized (fun m -> sized (fun n -> complex_matrix ~batch m n)))
        (fun z ->
          let p, l, u = Nx.lu z in
          equal near_complex (rows p z) (Nx.matmul l u));
      test "lu keeps a zero pivot of a singular matrix on U's diagonal"
        (fun () ->
          let a =
            Nx.create Nx.float64 [| 3; 3 |]
              [| 1.; 2.; 3.; 2.; 4.; 6.; 1.; 0.; 1. |]
          in
          let p, l, u = Nx.lu a in
          equal near (rows p a) (l *@ u);
          equal (close ~rel:0. ~abs:1e-15 ()) 0. (Nx.item [ 2; 2 ] u));
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
      prop
        "eigh and eigvalsh read only the triangle uplo names, and its diagonal"
        (Gen.pair (sized symmetric) Gen.bool)
        (fun (a, upper) ->
          let junk = Nx.full_like a 9. in
          let kept, uplo =
            if upper then (Nx.add (Nx.triu a) (Nx.tril ~k:(-1) junk), `U)
            else (Nx.add (Nx.tril a) (Nx.triu ~k:1 junk), `L)
          in
          let w, v = Nx.eigh ~uplo kept in
          equal near (a *@ v) (Nx.mul v (Nx.unsqueeze ~axes:[ -2 ] w));
          equal near w (Nx.eigvalsh ~uplo kept));
      prop "eigh of a complex Hermitian matrix has real w and a V = V diag w"
        (sized (fun n -> complex_matrix ~batch n n))
        (fun z ->
          let h =
            Nx.mul_s
              (Nx.add z (Nx.conjugate (t z)))
              Complex.{ re = 0.5; im = 0. }
          in
          let w, v = Nx.eigh h in
          equal near_complex (h *@ v)
            (Nx.mul v (Nx.unsqueeze ~axes:[ -2 ] (Nx.cast Nx.complex128 w)));
          equal near_complex
            (Nx.cast Nx.complex128 (identity_like v))
            (Nx.conjugate (t v) *@ v);
          equal near w (Nx.eigvalsh h));
      test "qr without a mode is the reduced factorization" (fun () ->
          let q, r = Nx.qr (Nx.ones Nx.float64 [| 5; 3 |]) in
          equal (array int) [| 5; 3 |] (Nx.shape q);
          equal (array int) [| 3; 3 |] (Nx.shape r));
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
          raises_invalid_arg (fun () -> Nx.lu (Nx.ones Nx.int32 [| 2; 2 |]));
          raises_invalid_arg (fun () -> Nx.lu (Nx.ones Nx.float64 [| 2 |]));
          raises_invalid_arg (fun () ->
              Nx.cholesky (Nx.ones Nx.float64 [| 2; 3 |]));
          raises_invalid_arg (fun () -> Nx.eigh (Nx.ones Nx.float64 [| 2; 3 |])));
    ]

let invariants =
  group "norms and invariants"
    [
      prop "det is multiplicative at float64"
        (sized (fun n ->
             Gen.bind batch (fun bt ->
                 let batch = Gen.constant ~pp:pp_shape bt in
                 Gen.pair (square ~batch n) (square ~batch n))))
        (fun (a, b) ->
          equal near (Nx.mul (Nx.det a) (Nx.det b)) (Nx.det (a *@ b)));
      test "det of every permutation matrix up to size 4 is its parity"
        (fun () ->
          let rec permutations = function
            | [] -> [ [] ]
            | l ->
                List.concat_map
                  (fun x ->
                    List.map
                      (fun p -> x :: p)
                      (permutations (List.filter (( <> ) x) l)))
                  l
          in
          let parity p =
            let p = Array.of_list p in
            let inversions = ref 0 in
            Array.iteri
              (fun i x ->
                for j = i + 1 to Array.length p - 1 do
                  if p.(j) < x then incr inversions
                done)
              p;
            if !inversions mod 2 = 0 then 1. else -1.
          in
          List.iter
            (fun n ->
              List.iter
                (fun p ->
                  let m =
                    Nx.init Nx.float64 [| n; n |] (fun i ->
                        if List.nth p i.(0) = i.(1) then 1. else 0.)
                  in
                  equal (close ~rel:0. ()) (parity p) (Nx.item [] (Nx.det m)))
                (permutations (List.init n Fun.id)))
            [ 1; 2; 3; 4 ]);
      test "det of an empty matrix is 1" (fun () ->
          equal (close ~rel:0. ()) 1.
            (Nx.item [] (Nx.det (Nx.zeros Nx.float64 [| 0; 0 |]))));
      prop "slogdet's sign times the exponential of its log is det, at float64"
        (sized square) (fun a ->
          let sign, logabs = Nx.slogdet a in
          equal near (Nx.det a) (Nx.mul sign (Nx.exp logabs)));
      prop
        "slogdet of a complex matrix is a sign of modulus 1 and det's log \
         magnitude"
        (sized (fun n ->
             Gen.map
               (fun z ->
                 Nx.add z
                   (Nx.mul_s (Nx.eye Nx.complex128 n)
                      { Complex.re = Float.of_int (n + 2); im = 0. }))
               (complex_matrix n n)))
        (fun z ->
          let sign, logabs = Nx.slogdet z in
          equal (close ~rel:1e-12 ()) 1. (Complex.norm (Nx.item [] sign));
          equal near_complex (Nx.det z)
            (Nx.mul sign (Nx.cast Nx.complex128 (Nx.exp logabs))));
      test "slogdet of a singular matrix is a zero sign and a log of -inf"
        (fun () ->
          let sign, logabs =
            Nx.slogdet (Nx.create Nx.float64 [| 2; 2 |] [| 1.; 2.; 2.; 4. |])
          in
          equal (close ~rel:0. ()) 0. (Nx.item [] sign);
          equal (close ~rel:0. ()) Float.neg_infinity (Nx.item [] logabs));
      test "slogdet holds a determinant beyond float64's range" (fun () ->
          let _, logabs = Nx.slogdet (Nx.mul_s (Nx.eye Nx.float64 400) 10.) in
          equal (close ~rel:1e-12 ())
            (400. *. Float.log 10.)
            (Nx.item [] logabs));
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
      prop "solve gives x with a x = b for a complex system"
        (sized (fun n ->
             Gen.pair
               (Gen.map
                  (fun z ->
                    Nx.add z
                      (Nx.mul_s (Nx.eye Nx.complex128 n)
                         { Complex.re = Float.of_int (n + 2); im = 0. }))
                  (complex_matrix n n))
               (complex_matrix n 2)))
        (fun (a, b) -> equal near_complex b (Nx.matmul a (Nx.solve a b)));
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
      prop "a float16 det is float32's, rounded once" (sized square) (fun a ->
          let a16 = Nx.cast Nx.float16 a in
          equal
            (tensor (close ~rel:0. ()))
            (Nx.cast Nx.float16 (Nx.det (Nx.cast Nx.float32 a16)))
            (Nx.det a16));
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

(* Factorizations at scale: the identities above over orders past each
   algorithm's crossover (divide and conquer past 25, blocked panels past 32, 64
   and 128) and over every dtype a factorization takes. A matrix is built at
   complex128 and rounded to its dtype; residuals are relative Frobenius norms
   over the whole batch, measured at complex128. *)

type fdtype =
  | F : {
      name : string;
      dtype : ('a, 'b) Nx.dtype;
      complex : bool;
      compute : float;
      storage : float;
    }
      -> fdtype

(* [compute] and [storage] are the unit roundoffs the factorization runs at and
   the factors are stored at: float16 factors at float32 and rounds once. *)
let fdtypes =
  let u32 = ldexp 1. (-24) and u64 = ldexp 1. (-53) in
  [
    F
      {
        name = "float64";
        dtype = Nx.float64;
        complex = false;
        compute = u64;
        storage = u64;
      };
    F
      {
        name = "float32";
        dtype = Nx.float32;
        complex = false;
        compute = u32;
        storage = u32;
      };
    F
      {
        name = "float16";
        dtype = Nx.float16;
        complex = false;
        compute = u32;
        storage = ldexp 1. (-11);
      };
    F
      {
        name = "complex128";
        dtype = Nx.complex128;
        complex = true;
        compute = u64;
        storage = u64;
      };
    F
      {
        name = "complex64";
        dtype = Nx.complex64;
        complex = true;
        compute = u32;
        storage = u32;
      };
  ]

(* The residual a backward-stable factorization of order [n] meets. *)
let bound ?(c = 64.) (F d) n =
  (c *. float_of_int n *. d.compute) +. (16. *. d.storage)

let small ~msg limit value =
  if not (value <= limit) then failf "%s: %g exceeds %g" msg value limit

let c128 x = Nx.cast Nx.complex128 x
let adjoint x = Nx.conjugate (t x)

let fro x =
  Array.fold_left (fun s z -> s +. Complex.norm2 z) 0. (Nx.to_array (c128 x))
  |> Float.sqrt

let rel ~expected actual =
  fro (Nx.sub (c128 expected) (c128 actual)) /. fro expected

(* ‖QᴴQ - I‖ relative to ‖I‖, over the columns of [q]. *)
let orthonormality q =
  let g = adjoint (c128 q) *@ c128 q in
  rel ~expected:(c128 (identity_like g)) g

let real_part z = Nx.cast Nx.float64 z

(* A matrix whose parts are [entries] at complex128, its imaginary part zero
   unless [complex]. *)
let entries ?(seed = 0) shape =
  Nx.init Nx.float64 shape (fun i ->
      let k = Ref.ravel shape i in
      Float.sin (float_of_int (((k * 13) + seed) mod 251))
      +. (0.1 *. Float.cos (float_of_int (k + seed))))

let parts ?(seed = 0) ~complex shape =
  Nx.complex Nx.complex128 ~re:(entries ~seed shape)
    ~im:
      (if complex then Nx.mul_s (entries ~seed:(seed + 7) shape) 0.3
       else Nx.zeros Nx.float64 shape)

let scalar re = { Complex.re; im = 0. }

let hermitian ~complex shape =
  let a = parts ~complex shape in
  Nx.mul_s (Nx.add a (adjoint a)) (scalar 0.5)

let positive ~complex shape =
  let a = parts ~complex shape in
  let n = shape.(Array.length shape - 1) in
  Nx.add
    (a *@ adjoint a)
    (Nx.mul_s (c128 (identity_like a)) (scalar (float_of_int n)))

let is_zero x = equal (close ~rel:0. ()) 0. (fro x)

let check_eigh ?(c = 64.) (F d as fd) a =
  let n = Nx.dim (-1) a in
  let a = Nx.cast d.dtype a in
  let w, v = Nx.eigh a in
  let v = c128 v in
  small ~msg:"a = V diag(w) Vᴴ" (bound ~c fd n)
    (rel ~expected:a
       (Nx.mul v (Nx.unsqueeze ~axes:[ -2 ] (c128 w)) *@ adjoint v));
  small ~msg:"V is orthonormal" (bound ~c fd n) (orthonormality v);
  equal ~msg:"w ascends" (tensor float_exact) (fst (Nx.sort w)) w;
  small ~msg:"eigvalsh is eigh's w" (bound ~c fd n)
    (rel ~expected:w (Nx.eigvalsh a))

let check_svd ?(full = false) (F d as fd) a =
  let a = Nx.cast d.dtype a in
  let m = Nx.dim (-2) a and n = Nx.dim (-1) a in
  let k = Int.min m n in
  let u, s, vh = Nx.svd ~full_matrices:full a in
  equal ~msg:"U's shape" (array int)
    [| m; (if full then m else k) |]
    (Nx.shape u);
  equal ~msg:"Vh's shape" (array int)
    [| (if full then n else k); n |]
    (Nx.shape vh);
  let uk = c128 (Nx.slice [ A; R (0, k) ] u)
  and vk = c128 (Nx.slice [ R (0, k); A ] vh) in
  small ~msg:"a = U diag(S) Vh"
    (bound fd (Int.max m n))
    (rel ~expected:a (Nx.mul uk (Nx.unsqueeze ~axes:[ -2 ] (c128 s)) *@ vk));
  small ~msg:"U is orthonormal" (bound fd m) (orthonormality u);
  small ~msg:"Vh's rows are orthonormal" (bound fd n)
    (orthonormality (adjoint vh));
  equal ~msg:"S descends" (tensor float_exact)
    (fst (Nx.sort ~descending:true s))
    s;
  equal ~msg:"S is not negative" (tensor bool)
    (Nx.ones_like (Nx.greater_equal_s s 0.))
    (Nx.greater_equal_s s 0.);
  small ~msg:"svdvals is svd's S" (bound fd k) (rel ~expected:s (Nx.svdvals a))

let check_qr ~mode (F d as fd) a =
  let a = Nx.cast d.dtype a in
  let m = Nx.dim (-2) a and n = Nx.dim (-1) a in
  let q, r = Nx.qr ~mode a in
  small ~msg:"a = Q R" (bound fd (Int.max m n)) (rel ~expected:a (q *@ r));
  small ~msg:"Q is orthonormal" (bound fd m) (orthonormality q);
  is_zero (Nx.tril ~k:(-1) (c128 r))

let check_cholesky ~upper (F d as fd) a =
  let n = Nx.dim (-1) a in
  let a = Nx.cast d.dtype a in
  let f = c128 (Nx.cholesky ~upper a) in
  small ~msg:"a is the factor times its adjoint" (bound fd n)
    (rel ~expected:a (if upper then adjoint f *@ f else f *@ adjoint f));
  is_zero (if upper then Nx.tril ~k:(-1) f else Nx.triu ~k:1 f)

(* A triangle of [a] with a dominant diagonal, the other triangle noise the
   solver must not read. *)
let check_solve_triangular ~upper ~transpose ~unit_diag (F d as fd) n nrhs =
  let a =
    Nx.add
      (Nx.mul_s (parts ~complex:d.complex [| n; n |]) (scalar 0.3))
      (Nx.mul_s (c128 (Nx.eye Nx.float64 n)) (scalar 1.5))
  in
  let b = parts ~seed:3 ~complex:d.complex [| n; nrhs |] in
  let tri = if upper then Nx.triu a else Nx.tril a in
  let tri =
    if unit_diag then
      Nx.add
        (Nx.sub tri (Nx.mul tri (c128 (Nx.eye Nx.float64 n))))
        (c128 (Nx.eye Nx.float64 n))
    else tri
  in
  let tri = c128 (Nx.cast d.dtype tri) and b = Nx.cast d.dtype b in
  let x =
    c128
      (Nx.solve_triangular ~upper ~transpose ~unit_diag (Nx.cast d.dtype a) b)
  in
  let op = if transpose then adjoint tri else tri in
  small ~msg:"op(a) x = b, relative to ‖op(a)‖ ‖x‖" (bound fd n)
    (fro (Nx.sub (op *@ x) (c128 b)) /. (fro op *. fro x))

let fname (F d) = d.name
let dtype_named name = List.find (fun fd -> fname fd = name) fdtypes
let pp_order ppf n = Format.fprintf ppf "order %d" n
let pp_dims ppf (m, n) = Format.fprintf ppf "%dx%d" m n

(* [check] over every dtype and each of [sizes], and each of [large] in a group
   tagged slow. *)
let over_dtypes ?(large = []) ~pp title sizes check =
  let rows sizes =
    List.concat_map (fun s -> List.map (fun fd -> (fd, s)) fdtypes) sizes
  in
  let name (fd, s) = Format.asprintf "%s, %a" (fname fd) pp s in
  let run (fd, s) = check fd s in
  group title
    (cases ~name "past each crossover" (rows sizes) run
    ::
    (if large = [] then []
     else [ cases ~tags:[ "slow" ] ~name "at large orders" (rows large) run ]))

let at_scale =
  group "factorizations at scale"
    [
      over_dtypes ~pp:pp_order
        "eigh: a = V diag(w) Vᴴ with orthonormal V, and eigvalsh gives w"
        [ 1; 2; 5; 16; 26; 33; 40; 64 ] ~large:[ 100; 129; 257 ]
        (fun (F d as fd) n ->
          check_eigh fd (hermitian ~complex:d.complex [| n; n |]));
      over_dtypes ~pp:pp_dims
        "svd: a = U diag(S) Vh with orthonormal factors and S descending"
        [
          (1, 1);
          (3, 5);
          (5, 3);
          (5, 5);
          (4, 6);
          (8, 8);
          (16, 9);
          (26, 26);
          (33, 33);
          (40, 40);
          (40, 64);
          (64, 40);
          (53, 53);
        ]
        ~large:[ (129, 96); (96, 129); (129, 129); (200, 200) ]
        (fun (F d as fd) (m, n) ->
          let a = parts ~complex:d.complex [| m; n |] in
          check_svd fd a;
          check_svd ~full:true fd a);
      over_dtypes ~pp:pp_dims
        "qr: a = Q R with orthonormal Q and upper-triangular R"
        [ (3, 5); (5, 3); (40, 30); (64, 100); (100, 64) ]
        ~large:[ (129, 129); (150, 140); (140, 150); (2100, 160) ]
        (fun (F d as fd) (m, n) ->
          let a = parts ~complex:d.complex [| m; n |] in
          check_qr ~mode:`Reduced fd a;
          check_qr ~mode:`Complete fd a);
      over_dtypes ~pp:pp_order
        "cholesky: a is the factor times its adjoint, the factor triangular"
        [ 1; 2; 33; 65 ] ~large:[ 128; 200 ] (fun (F d as fd) n ->
          let a = positive ~complex:d.complex [| n; n |] in
          check_cholesky ~upper:false fd a;
          check_cholesky ~upper:true fd a);
      over_dtypes ~pp:pp_dims
        "solve_triangular: op(a) x = b from the named triangle, for every \
         option"
        [ (5, 3); (40, 3); (63, 3); (65, 3) ]
        ~large:[ (128, 3); (192, 256) ]
        (fun fd (n, nrhs) ->
          List.iter
            (fun (upper, transpose, unit_diag) ->
              check_solve_triangular ~upper ~transpose ~unit_diag fd n nrhs)
            [
              (false, false, false);
              (true, false, false);
              (false, true, false);
              (true, true, false);
              (false, false, true);
              (true, true, true);
            ]);
      test "a batch factors each matrix alone" (fun () ->
          let fd = List.hd fdtypes in
          check_eigh fd (hermitian ~complex:false [| 6; 7; 7 |]);
          check_cholesky ~upper:false fd
            (positive ~complex:false [| 9; 40; 40 |]);
          check_qr ~mode:`Reduced fd (parts ~complex:false [| 4; 9; 6 |]);
          let a = parts ~complex:false [| 3; 40; 30 |] in
          let u, s, vh = Nx.svd a in
          small ~msg:"a = U diag(S) Vh" (bound fd 40)
            (rel ~expected:a
               (Nx.mul (c128 u) (Nx.unsqueeze ~axes:[ -2 ] (c128 s)) *@ c128 vh)));
      test
        "cholesky of a batch with one matrix not positive definite raises, \
         whichever worker meets it" (fun () ->
          let a =
            Nx.set [ I 13; I 0; I 0 ]
              (Nx.scalar Nx.float64 (-1.))
              (real_part (positive ~complex:false [| 20; 6; 6 |]))
          in
          raises_match
            (function
              | Nx.Linalg_error { kind = `Not_positive_definite; _ } -> true
              | _ -> false)
            (fun () -> Nx.cholesky a));
      test
        "solve_triangular of a batch with one zero pivot raises `Singular, \
         whichever worker meets it" (fun () ->
          let a =
            Nx.set [ I 7; I 2; I 2 ] (Nx.scalar Nx.float64 0.)
              (Nx.add
                 (Nx.mul_s (Nx.ones Nx.float64 [| 20; 5; 5 |]) 0.3)
                 (Nx.mul_s
                    (identity_like (Nx.zeros Nx.float64 [| 20; 5; 5 |]))
                    2.))
          in
          raises_match
            (function
              | Nx.Linalg_error { kind = `Singular; _ } -> true | _ -> false)
            (fun () -> Nx.solve_triangular a (Nx.ones Nx.float64 [| 20; 5 |])));
      test "svd of a rank-deficient complex64 matrix factors it" (fun () ->
          let n = 30 in
          let b =
            Nx.init Nx.float64 [| n; n |] (fun i ->
                if i.(0) < n / 2 && i.(1) < n / 2 then 1.
                else if i.(0) = i.(1) then 2.
                else 0.)
          in
          check_svd (dtype_named "complex64")
            (Nx.complex Nx.complex128 ~re:b ~im:(Nx.mul_s b 0.5)));
      cases
        ~name:(fun (name, s) -> Printf.sprintf "%s scaled by %g" name s)
        "a matrix of extreme magnitude has the factors of what it holds, \
         unscaled, scaled"
        [
          ("float64", 1e-170);
          ("float64", 1e-300);
          ("float64", 1e-310);
          ("float64", 1e170);
          ("float64", 1e300);
          ("float32", 1e-20);
          ("float32", 1e-30);
          ("float32", 1e20);
          ("float32", 1e30);
        ]
        (fun (name, s) ->
          let (F d as fd) = dtype_named name in
          let a = entries [| 6; 5 |] in
          let h = real_part (hermitian ~complex:false [| 6; 6 |]) in
          let scaled x = Nx.cast d.dtype (Nx.mul_s x s) in
          let unscale x = Nx.div_s (Nx.cast Nx.float64 x) s in
          (* Scaling rounds, to fewer bits where the entries turn subnormal: the
             reference is the matrix the factorization is given. *)
          let held x = Nx.cast d.dtype (unscale (scaled x)) in
          small ~msg:"svdvals" (bound fd 6)
            (rel
               ~expected:(Nx.svdvals (held a))
               (unscale (Nx.svdvals (scaled a))));
          small ~msg:"qr's R" (bound fd 6)
            (rel
               ~expected:(snd (Nx.qr (held a)))
               (unscale (snd (Nx.qr (scaled a)))));
          small ~msg:"eigvalsh" (bound fd 6)
            (rel
               ~expected:(Nx.eigvalsh (held h))
               (unscale (Nx.eigvalsh (scaled h))));
          small ~msg:"eigh's w" (bound fd 6)
            (rel
               ~expected:(Nx.eigvalsh (held h))
               (unscale (fst (Nx.eigh (scaled h))))));
      cases
        ~name:(Format.asprintf "%a" pp_shape)
        "an empty matrix or an empty batch factors into empty factors"
        [ [| 0; 0 |]; [| 0; 3; 3 |] ]
        (fun shape ->
          let a = Nx.zeros Nx.float64 shape in
          let batch = Array.sub shape 0 (Array.length shape - 2) in
          let n = shape.(Array.length shape - 1) in
          let sq = Array.append batch [| n; n |]
          and vec = Array.append batch [| n |] in
          let w, v = Nx.eigh a in
          equal (array int) vec (Nx.shape w);
          equal (array int) sq (Nx.shape v);
          equal (array int) vec (Nx.shape (Nx.eigvalsh a));
          let q, r = Nx.qr a in
          equal (array int) sq (Nx.shape q);
          equal (array int) sq (Nx.shape r);
          let u, s, vh = Nx.svd a in
          equal (array int) sq (Nx.shape u);
          equal (array int) vec (Nx.shape s);
          equal (array int) sq (Nx.shape vh);
          equal (array int) sq (Nx.shape (Nx.cholesky a));
          equal (array int) vec
            (Nx.shape (Nx.solve_triangular a (Nx.zeros Nx.float64 vec))));
    ]

(* The singular values np.linalg.svd gives the bidiagonal fixtures below. *)

let bd_clus40_s =
  [|
    1.0000000000004552;
    1.0000000000004277;
    1.0000000000003986;
    1.000000000000378;
    1.0000000000003681;
    1.0000000000003455;
    1.0000000000003382;
    1.0000000000003162;
    1.0000000000003086;
    1.000000000000307;
    1.000000000000286;
    1.0000000000002782;
    1.0000000000002756;
    1.0000000000002558;
    1.0000000000002482;
    1.0000000000002456;
    1.0000000000002258;
    1.0000000000002183;
    1.0000000000002156;
    1.0000000000001958;
    1.0000000000001883;
    1.0000000000001856;
    1.0000000000001659;
    1.0000000000001585;
    1.000000000000156;
    1.000000000000136;
    1.0000000000001286;
    1.0000000000001261;
    1.0000000000001057;
    1.000000000000098;
    1.0000000000000957;
    1.0000000000000755;
    1.0000000000000657;
    1.0000000000000475;
    1.000000000000036;
    1.000000000000026;
    1.0000000000000062;
    0.9999999999999889;
    0.999999999999975;
    0.999999999999944;
  |]

let bd_grad40_s =
  [|
    100000022.90322469;
    38881551.80308503;
    15117750.706156598;
    5878016.072274924;
    2285463.864134994;
    888623.8162743407;
    345510.7294592218;
    134339.93325988983;
    52233.45074266833;
    20309.17620904739;
    7896.522868499733;
    3070.29062975785;
    1193.7766417144355;
    464.1588833612773;
    180.4721766827174;
    70.17038286703837;
    27.283333764867706;
    10.608183551394484;
    4.124626382901348;
    1.6037187437513274;
    0.6235507341273916;
    0.24244620170823314;
    0.0942668455117885;
    0.036652412370796264;
    0.014251026703029992;
    0.005541020330009493;
    0.002154434690031882;
    0.0008376776400682923;
    0.0003257020655659783;
    0.00012663801734674022;
    4.923882631706742e-05;
    1.9144819761699567e-05;
    7.443803013251697e-06;
    2.8942661247167537e-06;
    1.1253355826007646e-06;
    4.3754793750741814e-07;
    1.7012542798525891e-07;
    6.614740641230145e-08;
    2.57191380905934e-08;
    9.999997709677007e-09;
  |]

let bd_zero33_s =
  [|
    1.6839902448903397;
    1.6126256353992048;
    1.5943055651751838;
    1.5850523995755001;
    1.5366302774637621;
    1.444177351153566;
    1.3671706449867962;
    1.3384978042100268;
    1.2581061847874977;
    1.2467485554070765;
    1.214284996995534;
    1.1740301046398476;
    1.1470415351593835;
    1.102589253529079;
    1.101940121030122;
    1.0629781329062375;
    1.009396599291299;
    1.0016612179567417;
    0.7662832160350473;
    0.705225469734421;
    0.6561439274729741;
    0.5483791568402014;
    0.5318729597193121;
    0.46464434623681933;
    0.4103381307826223;
    0.2422633969559378;
    0.19755340972195773;
    0.17433822959296214;
    0.17165146097483558;
    0.09472915460929923;
    0.033210166904989645;
    0.0;
    0.0;
  |]

let bd_wilk41_s =
  [|
    20.231765920239;
    20.231765920239;
    19.036585297985386;
    19.036585297985383;
    18.01504909703887;
    18.01504909703887;
    17.014744407790822;
    17.01474440779082;
    16.015636893688484;
    16.015636893688473;
    15.016680555438686;
    15.016680555438679;
    14.017874218411217;
    14.017874218411217;
    13.019252094705747;
    13.01925209470574;
    12.020860444516636;
    12.020860444516632;
    11.02276246662289;
    11.022762466622888;
    10.025046836465645;
    10.025046836465645;
    9.027842013151702;
    9.027842013151702;
    8.03134143610338;
    8.031341436103379;
    7.035850721802344;
    7.035850721802342;
    6.041883198534905;
    6.041883198534904;
    5.050373826697282;
    5.05037382669728;
    4.063229009991975;
    4.063229009991974;
    3.085057131288046;
    3.0850571312880453;
    2.1308656387760907;
    2.1308656387760907;
    1.2834973309298923;
    1.283497330929892;
    0.0;
  |]

(* Small matrices np.linalg.svd factored. *)

let svd_graded5 =
  [|
    32.386797746303365;
    16.827033870464344;
    14.584698473625766;
    -18.850569830179445;
    40.42840547068902;
    0.046132882854762004;
    0.02344450153202795;
    0.015648132918435656;
    -0.024303383986049882;
    0.043136264240973894;
    -12.129897333388689;
    -5.889762551323407;
    -5.79014615560565;
    6.926071545200442;
    -14.853552436553978;
    -27.77607326992134;
    -14.616090675580534;
    -12.360592663421496;
    16.2264304812836;
    -34.79871712862178;
    31.88425186254641;
    15.835254878984497;
    14.940170870726424;
    -18.3212109624654;
    39.29466835322517;
  |]

let svd_clustered4 =
  [|
    1.0319022220349705;
    -1.1692421800252757;
    -1.2519467991343354;
    -0.02301362382118173;
    1.4605164703680484;
    1.1209106287001123;
    0.16983450111505743;
    -0.7556567362558355;
    -0.4701649161878702;
    0.1385554176909072;
    -0.48267787450226246;
    -1.0258956403995578;
    -0.5414448548498132;
    -0.4068288671037338;
    -0.049613307019892874;
    -1.5079502874438235;
  |]

let svd_tall63 =
  [|
    0.4155026142980106;
    -0.9235446567132176;
    -0.19602731230796547;
    -0.5907698188631366;
    -0.29971123732782745;
    1.296885192726673;
    1.5295796333931557;
    0.6694181934096611;
    0.5487451197783295;
    0.6766289895558858;
    -0.012242186606443108;
    -0.07566346148706628;
    -0.6736451873760739;
    -0.05586745005007389;
    2.2599469866262614;
    0.8690393292538257;
    -0.3421170234268632;
    -0.4719266521335216;
  |]

let svd_dyn5 =
  [|
    12539723.02952069;
    -17505815.700353518;
    8917634.289808983;
    -16937631.63217063;
    4742540.701164636;
    -2762955.81560511;
    3857177.694615324;
    -1964885.2312229422;
    3731986.3108871873;
    -1045011.2446501966;
    -31291444.481013276;
    43683751.27652291;
    -22252930.88896055;
    42265915.521155044;
    -11834427.366774203;
    17868284.075847495;
    -24944618.11789646;
    12707031.17294191;
    -24134993.23686137;
    6757697.663380319;
    -19519069.286400933;
    27249173.575499415;
    -13880995.406145057;
    26364750.557379056;
    -7382084.710354824;
  |]

let svd_dyn5_s =
  [|
    99999999.99999999;
    100.00000000017037;
    1.000000000739783;
    9.999692044246688e-05;
    1.0152727663761473e-08;
  |]

let svd_clus6 =
  [|
    1.3762482976891157;
    -1.211578303678782;
    1.7029830637032164;
    1.1969969710698656;
    -0.23525908046506158;
    -0.3296615614040424;
    -0.6749225211531032;
    -0.46187176340362057;
    -0.16028257965153886;
    0.5914655454141511;
    -1.750989547527381;
    1.5334592779542342;
    2.0410556265995954;
    -0.2994038720024088;
    -1.3925101322350015;
    -1.0694069124241379;
    -1.0384666443579424;
    0.2936203371031439;
    -0.25987274966933854;
    -0.6227662353143555;
    -1.453093833319683;
    0.8234424169539848;
    0.5980158161545885;
    0.4190279860305053;
    0.9063803901859939;
    -0.7608715199431577;
    -0.17622785291524268;
    0.578868492851017;
    0.21671273669207655;
    -0.10349301644334462;
    -0.6191436082567583;
    -0.35719544629461536;
    0.22807085774378452;
    0.6553843800842688;
    -0.9070567424045388;
    0.8710376317212473;
  |]

let svd_clus6_s =
  [|
    2.999999999999999;
    2.9999999999999987;
    2.999999999999998;
    1.9999999999999998;
    1.0000000006201816e-07;
    9.999999990522373e-08;
  |]

let svd_rank54 =
  [|
    -2.1696103122284724;
    3.9831908107256426;
    2.3661310821741566;
    2.9319476599936753;
    -0.4973397612223236;
    1.2107073289003105;
    0.7855771904017829;
    2.3244441993188394;
    0.020612554164313977;
    1.8714810832971354;
    0.9543147010704648;
    -0.027118102233273104;
    3.596196166404184;
    1.3673206175009527;
    0.44047011478031906;
    0.32674110459701894;
    -1.2435439739734906;
    0.8178173438705062;
    0.38894380501552506;
    -2.275394443284693;
  |]

let svd_rank54_s =
  [|
    6.61055515551338;
    4.151029831382346;
    2.938020265275789;
    9.106250838955923e-17;
  |]

(* Small matrices with np.linalg.svd singular values: each within [rel] of it,
   with an absolute floor of [abs] times the largest where numpy's own values
   sit at the rounding floor of a ‖a‖ ε solver. *)
let svd_oracles =
  [
    ( "[[1, 1], [0, 1]], φ and 1/φ",
      2,
      2,
      [| 1.; 1.; 0.; 1. |],
      [| 1.618033988749895; 0.6180339887498949 |],
      1e-12,
      0. );
    ("a column [3; 4]", 2, 1, [| 3.; 4. |], [| 5. |], 1e-12, 0.);
    ("a row [3, 4, 0]", 1, 3, [| 3.; 4.; 0. |], [| 5. |], 1e-12, 0.);
    ( "graded from 1e2 to 1e-6",
      5,
      5,
      svd_graded5,
      [| 100.; 1.; 0.01; 1e-4; 1e-6 |],
      1e-7,
      0. );
    ("clustered at 2", 4, 4, svd_clustered4, [| 2.; 2.; 2.; 0.5 |], 1e-9, 0.);
    ( "tall",
      6,
      3,
      svd_tall63,
      [| 2.895986520101071; 1.911860947586696; 1.163883857289863 |],
      1e-9,
      0. );
    ("clustered at 3 with a 1e-7 pair", 6, 6, svd_clus6, svd_clus6_s, 1e-8, 0.);
    ("spanning 1e8 to 1e-8", 5, 5, svd_dyn5, svd_dyn5_s, 1e-9, 1e-13);
    ("of rank 3", 5, 4, svd_rank54, svd_rank54_s, 1e-9, 1e-13);
  ]

(* Matrices of chosen numerical structure, the spectra they stress in the
   solvers' deflation, splitting and sign handling. *)

let tridiagonal d e =
  let n = Array.length d in
  Nx.init Nx.float64 [| n; n |] (fun i ->
      if i.(0) = i.(1) then d.(i.(0))
      else if i.(0) = i.(1) + 1 then e.(i.(1))
      else if i.(1) = i.(0) + 1 then e.(i.(0))
      else 0.)

let upper_bidiagonal d e =
  let n = Array.length d in
  Nx.init Nx.float64 [| n; n |] (fun i ->
      if i.(0) = i.(1) then d.(i.(0))
      else if i.(1) = i.(0) + 1 then e.(i.(0))
      else 0.)

(* Symmetric matrices for eigh's divide and conquer: three levels of equal
   diagonal under a 1e-6 perturbation (clusters), Wilkinson blocks glued by a
   1e-7 coupling (near-degenerate pairs across the tear), a diagonal graded from
   1e-8 to 1e8, and a smooth and a random dense matrix. *)
let symmetric_fixtures =
  let sym n f =
    Nx.init Nx.float64 [| n; n |] (fun i ->
        f (Int.max i.(0) i.(1)) (Int.min i.(0) i.(1)))
  in
  [
    ( "clustered",
      2048.,
      fun n ->
        sym n (fun i j ->
            if i = j then
              if i < n / 2 then 1. else if i < 3 * n / 4 then 2. else 3.
            else 1e-6 *. Float.sin (float_of_int ((i * n) + j))) );
    ( "glued Wilkinson",
      2048.,
      fun n ->
        let half = (n - 1) / 2 in
        tridiagonal
          (Array.init n (fun i ->
               Float.abs (float_of_int ((i mod (half + 1)) - (half / 2)))))
          (Array.init (n - 1) (fun i -> if i = (n / 2) - 1 then 1e-7 else 1.))
    );
    ( "graded",
      64.,
      fun n ->
        let d =
          Array.init n (fun i ->
              10. ** ((16. *. (float_of_int i /. float_of_int (n - 1))) -. 8.))
        in
        tridiagonal d
          (Array.init (n - 1) (fun i ->
               1e-3 *. Float.sqrt (Float.abs (d.(i) *. d.(i + 1))))) );
    ( "smooth",
      64.,
      fun n ->
        sym n (fun i j ->
            if i = j then Float.sin (float_of_int i *. 1.3)
            else 0.3 *. Float.cos (float_of_int ((i * 7) + j))) );
    ( "random",
      64.,
      fun n ->
        let seed = ref 20260717 in
        let next () =
          seed := ((!seed * 1103515245) + 12345) land 0x7fffffff;
          (float_of_int !seed /. 2147483648.) -. 0.5
        in
        let lower = Array.init (n * n) (fun _ -> next ()) in
        sym n (fun i j -> lower.((i * n) + j)) );
  ]

(* Upper bidiagonals with offline np.linalg.svd singular values: near-equal d
   with tiny e (maximal deflation), d graded from 1e-8 to 1e8, exact zeros in d
   and e (splits and zero singular values), and the Wilkinson-like |i - 20|
   diagonal (paired singular values and a zero). *)
let bidiagonal_fixtures =
  [
    ( "clustered",
      upper_bidiagonal
        (Array.init 40 (fun i -> 1. +. (1e-14 *. float_of_int i)))
        (Array.init 39 (fun i ->
             1e-13 *. (0.5 +. (float_of_int (i * 7 mod 3) /. 3.)))),
      bd_clus40_s );
    ( "graded",
      (let d =
         Array.init 40 (fun i -> 10. ** ((16. *. float_of_int i /. 39.) -. 8.))
       in
       upper_bidiagonal d
         (Array.init 39 (fun i -> 1e-3 *. Float.sqrt (d.(i) *. d.(i + 1))))),
      bd_grad40_s );
    ( "with zeros",
      (let d =
         Array.init 33 (fun i ->
             Float.sin (float_of_int (i * 13 mod 251))
             +. (0.1 *. Float.cos (float_of_int i)))
       and e =
         Array.init 32 (fun i ->
             Float.sin (float_of_int (i * 17 mod 241))
             +. (0.1 *. Float.cos (float_of_int (2 * i))))
       in
       d.(11) <- 0.;
       d.(22) <- 0.;
       e.(16) <- 0.;
       upper_bidiagonal d e),
      bd_zero33_s );
    ( "Wilkinson-like",
      upper_bidiagonal
        (Array.init 41 (fun i -> float_of_int (abs (i - 20))))
        (Array.make 40 1.),
      bd_wilk41_s );
  ]

(* Each computed eigenvalue matched to its nearest expected one, each expected
   one used once: the largest distance. *)
let spectrum_distance expected got =
  let used = Array.make (Array.length expected) false in
  Array.fold_left
    (fun worst (z : Complex.t) ->
      let best = ref (-1) in
      Array.iteri
        (fun i e ->
          if
            (not used.(i))
            && (!best < 0
               || Complex.norm (Complex.sub e z)
                  < Complex.norm (Complex.sub expected.(!best) z))
          then best := i)
        expected;
      used.(!best) <- true;
      Float.max worst (Complex.norm (Complex.sub expected.(!best) z)))
    0. got

(* The largest ‖a v - λ v‖ / (‖a‖ ‖v‖) over the eigenpairs. *)
let pair_residual a w v =
  let a = c128 a and v = c128 v and w = c128 w in
  let r = Nx.sub (a *@ v) (Nx.mul v (Nx.unsqueeze ~axes:[ -2 ] w)) in
  let n = Nx.dim (-1) v in
  List.fold_left
    (fun worst j ->
      let col x = Nx.slice [ A; I j ] x in
      Float.max worst (fro (col r) /. (fro a *. fro (col v))))
    0. (List.init n Fun.id)

let c re im = { Complex.re; im }

(* Spectra derived by hand from characteristic polynomials or construction. *)
let spectra =
  let real ?(narrow = true) ~tol name rows expected =
    (name, `Real (rows : float array array), expected, tol, narrow)
  and cplx ?(narrow = true) ~tol name rows expected =
    (name, `Complex (rows : Complex.t array array), expected, tol, narrow)
  in
  let r2 = Float.sqrt 2. and s7 = Float.sqrt 7. /. 2. in
  let disc = Complex.sqrt (c 5. 4.) in
  [
    real ~tol:1e-9 "a rotation by a right angle has ±i"
      [| [| 0.; -1. |]; [| 1.; 0. |] |]
      [| c 0. 1.; c 0. (-1.) |];
    real ~tol:1e-9 "a Jordan block has its eigenvalue twice"
      [| [| 2.; 1. |]; [| 0.; 2. |] |]
      [| c 2. 0.; c 2. 0. |];
    real ~tol:1e-9 "the defective [[1, 1], [0, 1]] has 1 twice"
      [| [| 1.; 1. |]; [| 0.; 1. |] |]
      [| c 1. 0.; c 1. 0. |];
    real ~tol:1e-5
      "a companion of (x-1)(x-2)(x-3) scaled by diag(1e8, 1, 1e-8) keeps 1, 2, \
       3"
      [| [| 0.; 1e8; 0. |]; [| 0.; 0.; 1e8 |]; [| 6e-16; -1.1e-7; 6. |] |]
      [| c 1. 0.; c 2. 0.; c 3. 0. |];
    real ~narrow:false ~tol:1e-6
      "a companion of (x-0.999)(x-1)(x-1.001) keeps the cluster"
      [| [| 0.; 1.; 0. |]; [| 0.; 0.; 1. |]; [| 0.999999; -2.999999; 3. |] |]
      [| c 0.999 0.; c 1. 0.; c 1.001 0. |];
    real ~tol:1e-9 "a rotation by π/4 scaled by 2 has √2 ± i√2"
      [| [| r2; -.r2 |]; [| r2; r2 |] |]
      [| c r2 r2; c r2 (-.r2) |];
    real ~tol:1e-9 "a companion of (x-2)(x²+1) has 2 and ±i"
      [| [| 0.; 1.; 0. |]; [| 0.; 0.; 1. |]; [| 2.; -1.; 2. |] |]
      [| c 2. 0.; c 0. 1.; c 0. (-1.) |];
    real ~tol:1e-9 "[[2,-1,0],[1,3,-1],[0,1,2]] has 2 and 2.5 ± i√7/2"
      [| [| 2.; -1.; 0. |]; [| 1.; 3.; -1. |]; [| 0.; 1.; 2. |] |]
      [| c 2. 0.; c 2.5 s7; c 2.5 (-.s7) |];
    real ~tol:1e-12 "a 1x1 matrix is its eigenvalue" [| [| 7. |] |]
      [| c 7. 0. |];
    cplx ~tol:1e-9 "a complex triangular matrix has its diagonal"
      [|
        [| c 1. 1.; c 2. 0.; c 0. 0. |];
        [| c 0. 0.; c 3. (-1.); c 1. 0. |];
        [| c 0. 0.; c 0. 0.; c 2. 0. |];
      |]
      [| c 1. 1.; c 3. (-1.); c 2. 0. |];
    cplx ~tol:1e-9 "[[1, i], [i, 1]] has 1 ± i"
      [| [| c 1. 0.; c 0. 1. |]; [| c 0. 1.; c 1. 0. |] |]
      [| c 1. 1.; c 1. (-1.) |];
    cplx ~tol:1e-9
      "[[0,1,1],[i,0,1],[1,1,0]], whose reduction takes a complex reflector, \
       has -1 and (1 ± √(5+4i))/2"
      [|
        [| c 0. 0.; c 1. 0.; c 1. 0. |];
        [| c 0. 1.; c 0. 0.; c 1. 0. |];
        [| c 1. 0.; c 1. 0.; c 0. 0. |];
      |]
      [|
        c (-1.) 0.;
        Complex.mul (c 0.5 0.) (Complex.add Complex.one disc);
        Complex.mul (c 0.5 0.) (Complex.sub Complex.one disc);
      |];
  ]

let matrix_of_rows dtype rows =
  let n = Array.length rows in
  Nx.create dtype [| n; n |] (Array.concat (Array.to_list rows))

let eigs =
  let spectrum (name, rows, expected, tol, narrow) =
    let check a tol =
      let w, v = Nx.eig a in
      let got = Nx.to_array w in
      let scale =
        Array.fold_left (fun m z -> Float.max m (Complex.norm z)) 1. expected
      in
      small ~msg:"the eigenvalues are the spectrum" (tol *. scale)
        (spectrum_distance expected got);
      small ~msg:"each pair has a v = λ v" tol (pair_residual a w v);
      small ~msg:"eigvals is eig's values" (tol *. scale)
        (spectrum_distance got (Nx.to_array (Nx.eigvals a)))
    in
    test name (fun () ->
        match rows with
        | `Real rows ->
            check (matrix_of_rows Nx.float64 rows) tol;
            if narrow then check (matrix_of_rows Nx.float32 rows) 1e-4
        | `Complex rows ->
            check (matrix_of_rows Nx.complex128 rows) tol;
            if narrow then check (matrix_of_rows Nx.complex64 rows) 1e-4)
  in
  let real_rows k =
    match List.nth spectra k with
    | _, `Real rows, expected, _, _ -> (rows, expected)
    | _ -> invalid_arg "real_rows"
  in
  group "eig"
    (List.map spectrum spectra
    @ [
        test "a batch has each matrix's spectrum" (fun () ->
            let r0, e0 = real_rows 6 and r1, e1 = real_rows 7 in
            let a =
              Nx.stack ~axis:0
                [ matrix_of_rows Nx.float64 r0; matrix_of_rows Nx.float64 r1 ]
            in
            let w, v = Nx.eig a in
            List.iteri
              (fun b expected ->
                let wb = Nx.slice [ I b ] w and vb = Nx.slice [ I b ] v in
                small ~msg:"the spectrum" 1e-6
                  (spectrum_distance expected (Nx.to_array wb));
                small ~msg:"a v = λ v" 1e-6
                  (pair_residual (Nx.slice [ I b ] a) wb vb))
              [ e0; e1 ]);
        cases
          ~name:(fun (name, _) -> name)
          "a dense matrix has a v = λ v and eigvals' values"
          [
            ("real, order 40", Nx.cast Nx.complex128 (entries [| 40; 40 |]));
            ("complex, order 24", parts ~complex:true [| 24; 24 |]);
            ("complex, order 64", parts ~complex:true [| 64; 64 |]);
          ]
          (fun (_, a) ->
            let w, v = Nx.eig a in
            small ~msg:"a v = λ v" 1e-9 (pair_residual a w v);
            small ~msg:"eigvals is eig's values" 1e-9
              (spectrum_distance (Nx.to_array w) (Nx.to_array (Nx.eigvals a))));
        cases
          ~name:(fun n -> Printf.sprintf "order %d" n)
          "a Wilkinson tridiagonal, its eigenvalues in near-equal pairs, \
           converges as real and as complex input"
          [ 20; 40; 80 ]
          (fun n ->
            let a =
              tridiagonal
                (Array.init n (fun i -> float_of_int (abs ((n / 2) - i))))
                (Array.make (n - 1) 1.)
            in
            let w, v = Nx.eig a in
            small ~msg:"real input" 1e-6 (pair_residual a w v);
            let z = Nx.cast Nx.complex128 a in
            let wz, vz = Nx.eig z in
            small ~msg:"complex input" 1e-6 (pair_residual z wz vz);
            small ~msg:"one spectrum" 1e-5
              (spectrum_distance (Nx.to_array w) (Nx.to_array wz)));
        cases
          ~name:(fun s -> Printf.sprintf "scaled by %g" s)
          "a matrix of extreme magnitude has the spectrum of its scaled copy"
          [ 1e-170; 1e-300; 1e-310; 1e170; 1e300 ]
          (fun s ->
            let rows, expected = real_rows 7 in
            let a = Nx.mul_s (matrix_of_rows Nx.float64 rows) s in
            let scaled w =
              Array.map (fun z -> Complex.div z (c s 0.)) (Nx.to_array w)
            in
            small ~msg:"real" 1e-6
              (spectrum_distance expected (scaled (Nx.eigvals a)));
            small ~msg:"complex" 1e-6
              (spectrum_distance expected
                 (scaled (Nx.eigvals (Nx.cast Nx.complex128 a)))));
      ])

let structures =
  group "numerical structure"
    [
      cases
        ~name:(fun ((name, _, _), n) -> Printf.sprintf "%s, order %d" name n)
        "eigh of a symmetric fixture reconstructs it with orthonormal vectors"
        (List.concat_map
           (fun f -> List.map (fun n -> (f, n)) [ 26; 33; 50; 64; 100; 128 ])
           symmetric_fixtures)
        (fun ((_, c, build), n) ->
          check_eigh ~c (List.hd fdtypes) (Nx.cast Nx.complex128 (build n)));
      cases ~tags:[ "slow" ]
        ~name:(fun ((name, _, _), n) -> Printf.sprintf "%s, order %d" name n)
        "eigh of a large symmetric fixture reconstructs it"
        (List.concat_map
           (fun f -> List.map (fun n -> (f, n)) [ 200; 500 ])
           symmetric_fixtures)
        (fun ((_, c, build), n) ->
          check_eigh ~c (List.hd fdtypes) (Nx.cast Nx.complex128 (build n)));
      cases
        ~name:(fun (name, _, _, _, _, _, _) -> name)
        "svd of a small fixture has numpy's singular values" svd_oracles
        (fun (_, m, n, xs, expected, rel_tol, abs_tol) ->
          let a = Nx.create Nx.float64 [| m; n |] xs in
          check_svd (List.hd fdtypes) (Nx.cast Nx.complex128 a);
          check_svd ~full:true (List.hd fdtypes) (Nx.cast Nx.complex128 a);
          let s = Nx.to_array (Nx.svdvals a) in
          Array.iteri
            (fun i e ->
              small
                ~msg:(Printf.sprintf "S.(%d)" i)
                ((rel_tol *. Float.abs e) +. (abs_tol *. expected.(0)))
                (Float.abs (s.(i) -. e)))
            expected);
      cases
        ~name:(fun (name, _, _) -> name)
        "svd of a bidiagonal fixture has numpy's singular values"
        bidiagonal_fixtures
        (fun (_, a, expected) ->
          check_svd (List.hd fdtypes) (Nx.cast Nx.complex128 a);
          let s = Nx.to_array (Nx.svdvals a) in
          Array.iteri
            (fun i e ->
              small
                ~msg:(Printf.sprintf "S.(%d)" i)
                (1e-9 *. (Float.abs e +. 1.))
                (Float.abs (s.(i) -. e)))
            expected);
      cases ~name:fst
        "svd diagonalizes a matrix whose bidiagonal sweeps backward: Uᵀ a V is \
         diagonal"
        [
          ("a 3x3 witness", (3, 3, [| -1.; 0.; 1.; 0.; -2.; 1.; 0.; 0.; 1. |]));
          ( "a 4x3 matrix",
            (4, 3, [| 1.; 2.; -1.; 0.; 3.; 1.; 2.; -2.; 1.; 1.; 0.; 4. |]) );
          ( "the bidiagonal 1, 3, 5",
            (3, 3, [| 1.; 2.; 0.; 0.; 3.; 2.; 0.; 0.; 5. |]) );
          ( "the bidiagonal 2, 3, 4",
            (3, 3, [| 2.; 1.; 0.; 0.; 3.; 1.; 0.; 0.; 4. |]) );
        ]
        (fun (_, (m, n, xs)) ->
          let a = Nx.create Nx.float64 [| m; n |] xs in
          let u, s, vh = Nx.svd a in
          let d = t u *@ a *@ t vh in
          small ~msg:"a = U diag(S) Vh" 1e-12
            (rel ~expected:a (Nx.mul u (Nx.unsqueeze ~axes:[ -2 ] s) *@ vh));
          small ~msg:"Uᵀ a V off its diagonal" 1e-12
            (fro (Nx.sub d (Nx.mul d (identity_like d))) /. fro a));
      test "svd gives +0 for the zero singular value of [[1, 1], [0, -0]]"
        (fun () ->
          List.iter
            (fun s ->
              let s = Nx.to_array s in
              equal (close ~rel:1e-6 ()) (Float.sqrt 2.) s.(0);
              equal float_exact 0. s.(1))
            [
              Nx.svdvals (Nx.create Nx.float64 [| 2; 2 |] [| 1.; 1.; 0.; -0. |]);
              Nx.cast Nx.float64
                (Nx.svdvals
                   (Nx.create Nx.float32 [| 2; 2 |] [| 1.; 1.; 0.; -0. |]));
            ]);
    ]

(* Matmul over every dtype, at shapes on both sides of each change of method: a
   few outputs, a single row or column, fewer rows than a register tile, a
   blocked product past 48³, row blocks past 256, contractions past 2048 and
   past 65536, and the products the platform library takes past 64³. Each
   operand is contiguous, transposed, or every other column of a wider matrix.
   The reference is the sum of products of the operands' values: exact and
   wrapped for integers, and for floats within the rounding nx.mli allows, k u
   Σ|a b| at the accumulation's unit roundoff u, then one rounding to the
   result's dtype. *)

type mm_dtype =
  | Mf : {
      name : string;
      dtype : (float, 'b) Nx.dtype;
      acc : float;
      storage : float;
      floor : float;
    }
      -> mm_dtype
  | Mi : int_dtype -> mm_dtype
  | Mc : {
      name : string;
      dtype : (Complex.t, 'b) Nx.dtype;
      unit : float;
    }
      -> mm_dtype

let mm_dtypes =
  let u32 = ldexp 1. (-24) in
  let float name dtype acc storage floor =
    Mf { name; dtype; acc; storage; floor }
  in
  [
    float "float64" Nx.float64 (ldexp 1. (-53)) (ldexp 1. (-53))
      (ldexp 1. (-1074));
    float "float32" Nx.float32 u32 u32 (ldexp 1. (-149));
    float "float16" Nx.float16 u32 (ldexp 1. (-11)) (ldexp 1. (-24));
    float "bfloat16" Nx.bfloat16 u32 (ldexp 1. (-8)) (ldexp 1. (-133));
    float "float8_e4m3" Nx.float8_e4m3 u32 (ldexp 1. (-4)) (ldexp 1. (-9));
    float "float8_e5m2" Nx.float8_e5m2 u32 (ldexp 1. (-3)) (ldexp 1. (-16));
    Mc { name = "complex128"; dtype = Nx.complex128; unit = ldexp 1. (-53) };
    Mc { name = "complex64"; dtype = Nx.complex64; unit = u32 };
  ]
  @ List.map (fun d -> Mi d) int_dtypes

let mm_name = function
  | Mf d -> d.name
  | Mc d -> d.name
  | Mi (Int_dtype d) -> d.name

type mm_layout = Contiguous | Transposed | Strided

(* The [r x c] matrix of row-major [xs] held in [layout]. *)
let held dtype layout r c xs =
  match layout with
  | Contiguous -> Nx.create dtype [| r; c |] xs
  | Transposed ->
      Nx.matrix_transpose
        (Nx.create dtype [| c; r |]
           (Array.init (r * c) (fun i -> xs.((i mod r * c) + (i / r)))))
  | Strided when c = 0 -> Nx.create dtype [| r; c |] xs
  | Strided ->
      let wide =
        Nx.create dtype
          [| r; 2 * c |]
          (Array.init
             (r * 2 * c)
             (fun i ->
               let row = i / (2 * c) and col = i mod (2 * c) in
               xs.((row * c) + (col / 2))))
      in
      Nx.squeeze ~axes:[ -1 ] (Nx.sliding_window ~axis:1 ~window:1 ~step:2 wide)

let layouts =
  [ (Contiguous, Contiguous); (Transposed, Strided); (Strided, Transposed) ]

let check_matmul md (m, k, n) (la, lb) =
  let fail_at i j fmt =
    Format.kasprintf
      (fun s ->
        failf "%s %dx%dx%d, output (%d, %d): %s" (mm_name md) m k n i j s)
      fmt
  in
  match md with
  | Mf d ->
      (* The operands' values are those of the dtype: rounded once. *)
      let values len =
        Nx.to_array
          (Nx.cast Nx.float64
             (Nx.cast d.dtype
                (Nx.create Nx.float64 [| len |]
                   (Array.init len (fun _ -> Random.float 2. -. 1.)))))
      in
      let xa = values (m * k) and xb = values (k * n) in
      let c =
        Nx.to_array
          (Nx.cast Nx.float64
             (Nx.matmul (held d.dtype la m k xa) (held d.dtype lb k n xb)))
      in
      for i = 0 to m - 1 do
        for j = 0 to n - 1 do
          let s = ref 0. and mag = ref 0. in
          for p = 0 to k - 1 do
            let t = xa.((i * k) + p) *. xb.((p * n) + j) in
            s := !s +. t;
            mag := !mag +. Float.abs t
          done;
          let limit =
            (1.01 *. float_of_int k *. d.acc *. !mag)
            +. (d.storage *. Float.abs !s)
            +. d.floor
          in
          let got = c.((i * n) + j) in
          if not (Float.abs (got -. !s) <= limit) then
            fail_at i j "got %.17g, the sum of products is %.17g (bound %g)" got
              !s limit
        done
      done
  | Mc d ->
      let values len =
        Nx.to_array
          (Nx.cast Nx.complex128
             (Nx.cast d.dtype
                (Nx.create Nx.complex128 [| len |]
                   (Array.init len (fun _ ->
                        {
                          Complex.re = Random.float 2. -. 1.;
                          im = Random.float 2. -. 1.;
                        })))))
      in
      let xa = values (m * k) and xb = values (k * n) in
      let c =
        Nx.to_array
          (Nx.cast Nx.complex128
             (Nx.matmul (held d.dtype la m k xa) (held d.dtype lb k n xb)))
      in
      for i = 0 to m - 1 do
        for j = 0 to n - 1 do
          let s = ref Complex.zero and mag = ref 0. in
          for p = 0 to k - 1 do
            let t = Complex.mul xa.((i * k) + p) xb.((p * n) + j) in
            s := Complex.add !s t;
            mag := !mag +. Complex.norm t
          done;
          let limit =
            (4.04 *. float_of_int (k + 1) *. d.unit *. !mag)
            +. (d.unit *. Complex.norm !s)
          in
          let got = c.((i * n) + j) in
          if not (Complex.norm (Complex.sub got !s) <= limit) then
            fail_at i j "got %g%+gi, the sum of products is %g%+gi (bound %g)"
              got.re got.im !s.re !s.im limit
        done
      done
  | Mi (Int_dtype d) ->
      let value () =
        let r = Random.int64 Int64.max_int in
        wrap ~bits:d.bits ~signed:d.signed
          (if Random.bool () then r
           else Int64.sub (Int64.of_int (Random.int 19)) 9L)
      in
      let xa = Array.init (m * k) (fun _ -> value ())
      and xb = Array.init (k * n) (fun _ -> value ()) in
      let of_i64 xs = Array.map d.of_i64 xs in
      let c =
        Nx.to_array
          (Nx.matmul
             (held d.dtype la m k (of_i64 xa))
             (held d.dtype lb k n (of_i64 xb)))
      in
      for i = 0 to m - 1 do
        for j = 0 to n - 1 do
          let s = ref 0L in
          for p = 0 to k - 1 do
            s := Int64.add !s (Int64.mul xa.((i * k) + p) xb.((p * n) + j))
          done;
          let expected = wrap ~bits:d.bits ~signed:d.signed !s in
          let got = d.to_i64 c.((i * n) + j) in
          if got <> expected then
            fail_at i j "got %Ld, expected %Ld" got expected
        done
      done

let pp_mnk ppf (m, k, n) = Format.fprintf ppf "%dx%d times %dx%d" m k k n

let routes =
  let rows shapes =
    List.concat_map (fun s -> List.map (fun md -> (md, s)) mm_dtypes) shapes
  in
  let name (md, s) = Format.asprintf "%s, %a" (mm_name md) pp_mnk s in
  let run (md, s) = List.iter (check_matmul md s) layouts in
  group "matmul routes"
    [
      cases ~name
        "matmul is the sum of products at every dtype, route and layout"
        (rows
           [
             (1, 1, 1);
             (0, 3, 2);
             (2, 0, 3);
             (1, 17, 1);
             (3, 16, 5);
             (2, 300, 2);
             (4, 16, 70);
             (1, 33, 40);
             (40, 33, 1);
             (8, 8, 8);
             (9, 31, 13);
             (4, 300, 64);
             (70, 70, 70);
             (33, 65, 65);
             (65, 65, 65);
           ])
        run;
      cases ~tags:[ "slow" ] ~name
        "matmul is the sum of products at large shapes"
        (rows
           [
             (16, 2100, 16);
             (300, 64, 40);
             (1, 300, 2100);
             (2100, 300, 1);
             (3, 70001, 5);
             (1, 196613, 1);
             (1025, 33, 1);
             (1, 65573, 50);
             (50, 65573, 1);
             (40, 64, 300);
             (300, 64, 300);
             (300, 2100, 40);
             (1300, 260, 130);
           ])
        run;
      test "matmul refuses bool and 4-bit operands, as arithmetic does"
        (fun () ->
          raises_invalid_arg (fun () ->
              Nx.matmul
                (Nx.ones Nx.bool [| 2; 2 |])
                (Nx.ones Nx.bool [| 2; 2 |]));
          raises_invalid_arg (fun () ->
              Nx.matmul
                (Nx.zeros Nx.int4 [| 2; 2 |])
                (Nx.zeros Nx.int4 [| 2; 2 |])));
      test "matmul gives the same bits on every call" (fun () ->
          let a =
            Nx.init Nx.float32 [| 3; 200003 |] (fun i ->
                Float.sin (float_of_int (i.(0) + (3 * i.(1)))))
          and b =
            Nx.init Nx.float32 [| 200003; 5 |] (fun i ->
                Float.cos (float_of_int (i.(0) + (7 * i.(1)))))
          in
          let bits x = Array.map Int32.bits_of_float (Nx.to_array x) in
          equal (array int32) (bits (Nx.matmul a b)) (bits (Nx.matmul a b)));
    ]

let () =
  exit
    (run "nx linalg"
       [
         products;
         einsums;
         factorizations;
         invariants;
         solvers;
         narrow;
         at_scale;
         eigs;
         structures;
         routes;
       ])
