(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* nx's operations on a device, computed by each backend: each computing
   operation, on operands placed on a device of the backend, gives the host's
   result bit for bit, placed on that device, for every dtype and layout. *)

open Windtrap
open Nx_test

(* How a case receives its operands: as they are, or placed. *)
type at = { at : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t }

let host = { at = Fun.id }
let f64 shape xs = Nx.create Nx.float64 shape xs
let unit_interval = f64 [| 2; 3 |] [| 0.1; 0.25; 0.4; 0.55; 0.7; 0.85 |]
let signed = f64 [| 2; 3 |] [| -1.5; 0.5; 2.5; -0.25; 3.; -2. |]
let ints = Nx.create Nx.int32 [| 2; 3 |] [| 5l; -3l; 12l; 7l; 0l; -9l |]
let ints' = Nx.create Nx.int32 [| 2; 3 |] [| 2l; 5l; 3l; -4l; 1l; 6l |]
let square = f64 [| 3; 3 |] [| 4.; 1.; 2.; 1.; 5.; 3.; 2.; 3.; 6. |]
let wide = f64 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 7. |]
let rows = Nx.create Nx.int64 [| 2; 2 |] [| 2L; 0L; 1L; 2L |]
let complex = Nx.cast Nx.complex128 signed
let images = f64 [| 1; 4; 4 |] (Array.init 16 float_of_int)
let patches = f64 [| 1; 4; 9 |] (Array.init 36 float_of_int)
let unary name f = (name, fun { at } -> [ Nx.P (f (at unit_interval)) ])

let binary name f =
  (name, fun { at } -> [ Nx.P (f (at signed) (at unit_interval)) ])

let integer name f = (name, fun { at } -> [ Nx.P (f (at ints) (at ints')) ])

let cases : (string * (at -> Nx.packed list)) list =
  [
    unary "neg" Nx.neg;
    unary "recip" Nx.recip;
    unary "abs" Nx.abs;
    unary "sqrt" Nx.sqrt;
    unary "sign" Nx.sign;
    unary "exp" Nx.exp;
    unary "log" Nx.log;
    unary "sin" Nx.sin;
    unary "cos" Nx.cos;
    unary "tan" Nx.tan;
    unary "asin" Nx.asin;
    unary "acos" Nx.acos;
    unary "atan" Nx.atan;
    unary "sinh" Nx.sinh;
    unary "cosh" Nx.cosh;
    unary "tanh" Nx.tanh;
    unary "trunc" Nx.trunc;
    unary "ceil" Nx.ceil;
    unary "floor" Nx.floor;
    unary "round" Nx.round;
    unary "erf" Nx.erf;
    binary "add" Nx.add;
    binary "sub" Nx.sub;
    binary "mul" Nx.mul;
    binary "div" Nx.div;
    binary "pow" (fun a b -> Nx.pow (Nx.abs a) b);
    binary "atan2" Nx.atan2;
    binary "maximum" Nx.maximum;
    binary "minimum" Nx.minimum;
    binary "mod" Nx.mod_;
    integer "integer div" Nx.div;
    integer "bitwise_and" Nx.bitwise_and;
    integer "bitwise_or" Nx.bitwise_or;
    integer "bitwise_xor" Nx.bitwise_xor;
    binary "equal" Nx.equal;
    binary "not_equal" Nx.not_equal;
    binary "less" Nx.less;
    binary "less_equal" Nx.less_equal;
    ( "where",
      fun { at } ->
        [
          Nx.P
            (Nx.where (at (Nx.less signed unit_interval)) (at signed) (at wide));
        ] );
    unary "sum" (fun x -> Nx.sum ~axes:[ 1 ] x);
    unary "prod" (fun x -> Nx.prod ~axes:[ 0 ] x);
    unary "max" (fun x -> Nx.max ~axes:[ 1 ] x);
    unary "min" (fun x -> Nx.min x);
    unary "cumsum" (fun x -> Nx.cumsum ~axis:1 x);
    unary "cumprod" (fun x -> Nx.cumprod ~axis:0 x);
    unary "cummax" (fun x -> Nx.cummax ~axis:1 x);
    unary "cummin" (fun x -> Nx.cummin ~axis:1 x);
    ("argmax", fun { at } -> [ Nx.P (Nx.argmax ~axis:1 (at signed)) ]);
    ("argmin", fun { at } -> [ Nx.P (Nx.argmin ~axis:0 (at signed)) ]);
    ( "sort",
      fun { at } ->
        let s, i = Nx.sort ~axis:1 (at signed) in
        [ Nx.P s; Nx.P i ] );
    ("argsort", fun { at } -> [ Nx.P (Nx.argsort ~descending:true (at signed)) ]);
    unary "pad" (Nx.pad [| (1, 0); (0, 2) |] 9.);
    ( "concatenate",
      fun { at } -> [ Nx.P (Nx.concatenate ~axis:0 [ at signed; at wide ]) ] );
    ("cast", fun { at } -> [ Nx.P (Nx.cast Nx.float32 (at signed)) ]);
    ("bitcast", fun { at } -> [ Nx.P (Nx.bitcast Nx.int64 (at signed)) ]);
    ("bitcast int32", fun { at } -> [ Nx.P (Nx.bitcast Nx.float32 (at ints)) ]);
    ( "bitcast narrower",
      fun { at } -> [ Nx.P (Nx.bitcast Nx.uint16 (at signed)) ] );
    ( "bitcast of a view",
      fun { at } ->
        [ Nx.P (Nx.bitcast Nx.int64 (Nx.slice [ I 1; R (1, 3) ] (at signed))) ]
    );
    ( "bitcast narrower of a view",
      fun { at } ->
        [ Nx.P (Nx.bitcast Nx.uint16 (Nx.slice [ A; R (1, 3) ] (at signed))) ]
    );
    ( "bitcast wider",
      fun { at } ->
        [ Nx.P (Nx.bitcast Nx.complex128 (at (Nx.reshape [| 3; 2 |] signed))) ]
    );
    ( "a host scalar joining a placed operand",
      fun { at } -> [ Nx.P (Nx.add (at signed) (Nx.scalar Nx.float64 2.)) ] );
    ( "threefry",
      fun { at } ->
        let key = Nx.Rng.of_tensor (at (Nx.Rng.key 7 :> Nx.int32_t)) in
        [ Nx.P (Nx.Rng.uniform key Nx.float32 [| 5 |]) ] );
    ( "take",
      fun { at } -> [ Nx.P (Nx.take ~axis:1 ~indices:(at rows) (at signed)) ] );
    ( "scatter",
      fun { at } ->
        [
          Nx.P
            (Nx.scatter ~axis:1 ~indices:(at rows)
               ~values:(at (Nx.slice [ R (0, 2); R (0, 2) ] wide))
               (at signed));
        ] );
    ( "scatter add",
      fun { at } ->
        [
          Nx.P
            (Nx.scatter ~mode:`Add ~axis:1 ~indices:(at rows)
               ~values:(at (Nx.slice [ R (0, 2); R (0, 2) ] wide))
               (at signed));
        ] );
    ( "scatter max",
      fun { at } ->
        [
          Nx.P
            (Nx.scatter ~mode:`Max ~axis:1 ~indices:(at rows)
               ~values:(at (Nx.slice [ R (0, 2); R (0, 2) ] wide))
               (at signed));
        ] );
    ( "scatter min",
      fun { at } ->
        [
          Nx.P
            (Nx.scatter ~mode:`Min ~axis:1 ~indices:(at rows)
               ~values:(at (Nx.slice [ R (0, 2); R (0, 2) ] wide))
               (at signed));
        ] );
    ( "reduce_segments add",
      fun { at } ->
        [
          Nx.P
            (Nx.reduce_segments `Add ~segments:3
               (at (Nx.create Nx.int64 [| 2 |] [| 2L; 0L |]))
               (at signed));
        ] );
    ( "set",
      fun { at } ->
        [
          Nx.P
            (Nx.set
               [ I 1; R (1, 3) ]
               (at (Nx.slice [ I 0; R (0, 2) ] wide))
               (at signed));
        ] );
    ( "extract_patches",
      fun { at } ->
        [
          Nx.P
            (Nx.extract_patches ~kernel_size:[| 2; 2 |] ~stride:[| 1; 1 |]
               ~dilation:[| 1; 1 |]
               ~padding:[| (0, 0); (0, 0) |]
               (at images));
        ] );
    ( "combine_patches",
      fun { at } ->
        [
          Nx.P
            (Nx.combine_patches ~output_size:[| 4; 4 |] ~kernel_size:[| 2; 2 |]
               ~stride:[| 1; 1 |] ~dilation:[| 1; 1 |]
               ~padding:[| (0, 0); (0, 0) |]
               (at patches));
        ] );
    ("matmul", fun { at } -> [ Nx.P (Nx.matmul (at wide) (at square)) ]);
    ("fftn", fun { at } -> [ Nx.P (Nx.fftn (at complex)) ]);
    ("ifftn", fun { at } -> [ Nx.P (Nx.ifftn ~axes:[ 1 ] (at complex)) ]);
    ("rfftn", fun { at } -> [ Nx.P (Nx.rfftn Nx.complex128 (at signed)) ]);
    ( "irfftn",
      fun { at } -> [ Nx.P (Nx.irfftn Nx.float64 ~axes:[ 1 ] (at complex)) ] );
    ( "irfftn to a given size",
      fun { at } ->
        [ Nx.P (Nx.irfftn Nx.float64 ~axes:[ 1 ] ~s:[ 6 ] (at complex)) ] );
    ("copy", fun { at } -> [ Nx.P (Nx.copy (at (Nx.transpose signed))) ]);
    ("cholesky", fun { at } -> [ Nx.P (Nx.cholesky (at square)) ]);
    ( "cholesky upper",
      fun { at } -> [ Nx.P (Nx.cholesky ~upper:true (at square)) ] );
    ( "qr",
      fun { at } ->
        let q, r = Nx.qr (at wide) in
        [ Nx.P q; Nx.P r ] );
    ( "qr complete",
      fun { at } ->
        let q, r = Nx.qr ~mode:`Complete (at wide) in
        [ Nx.P q; Nx.P r ] );
    ( "lu",
      fun { at } ->
        let perm, l, u = Nx.lu (at square) in
        [ Nx.P perm; Nx.P l; Nx.P u ] );
    ( "svd",
      fun { at } ->
        let u, s, vt = Nx.svd (at wide) in
        [ Nx.P u; Nx.P s; Nx.P vt ] );
    ( "svd full",
      fun { at } ->
        let u, s, vt = Nx.svd ~full_matrices:true (at wide) in
        [ Nx.P u; Nx.P s; Nx.P vt ] );
    ("svdvals", fun { at } -> [ Nx.P (Nx.svdvals (at wide)) ]);
    ("eigvals", fun { at } -> [ Nx.P (Nx.eigvals (at square)) ]);
    ( "eig",
      fun { at } ->
        let w, v = Nx.eig (at square) in
        [ Nx.P w; Nx.P v ] );
    ( "eigh",
      fun { at } ->
        let w, v = Nx.eigh (at square) in
        [ Nx.P w; Nx.P v ] );
    ("eigvalsh", fun { at } -> [ Nx.P (Nx.eigvalsh (at square)) ]);
    ( "solve_triangular",
      fun { at } ->
        [
          Nx.P
            (Nx.solve_triangular (at (Nx.triu square)) (at (Nx.transpose wide)));
        ] );
    ( "solve_triangular of a vector, transposed",
      fun { at } ->
        [
          Nx.P
            (Nx.solve_triangular ~upper:true ~transpose:true
               (at (Nx.triu square))
               (at (Nx.slice [ I 0 ] wide)));
        ] );
  ]

(* Operations of every dtype, through the kernels that read their operands'
   layout: a copy, a comparison, a selection and a concatenation. *)
let generic : (string * (Nx.packed -> Nx.packed)) list =
  [
    ("copy", fun (Nx.P x) -> Nx.P (Nx.copy x));
    ("equal", fun (Nx.P x) -> Nx.P (Nx.equal x x));
    ("where", fun (Nx.P x) -> Nx.P (Nx.where (Nx.equal x x) x x));
    ( "concatenate",
      fun (Nx.P x) ->
        let flat = Nx.reshape [| -1 |] x in
        Nx.P (Nx.concatenate ~axis:0 [ flat; flat ]) );
  ]

(* Each backend computes the cases on a device it computes on: nx.cpu on a test
   memory, and a backend paired with the host's memory, whose values are views
   of host values. A backend added to raven joins this list. *)
let backends =
  [
    ("nx.cpu on a test memory", Devices.d1);
    ( "a backend paired with the host's memory",
      Nx.Device.with_backend (module Renamed) Nx.Device.host );
  ]

let conformance (backend, d) =
  let p = Nx.Placement.on d in
  let placed = { at = (fun x -> Nx.place p x) } in
  let placed_result (Nx.P y) =
    equal ~msg:"placed on the device" Devices.placement p (Nx.placement y);
    Nx.P (Nx.place Nx.Placement.host y)
  in
  let every_dtype (Stored.Case c) =
    prop
      (c.name ^ " values of every layout, each operation's host result")
      (Gen.pair c.tensors
         (Gen.of_list
            ~pp:(fun ppf (n, _) -> Format.pp_print_string ppf n)
            generic))
      (fun (x, (_, f)) ->
        cover "zero-size" (Nx.numel x = 0);
        cover "strided" (not (Nx.is_c_contiguous x));
        equal Stored.packed (f (Nx.P x))
          (placed_result (f (Nx.P (Nx.place p x)))))
  in
  group backend
    [
      group "cases"
        (List.map
           (fun (name, f) ->
             test name (fun () ->
                 List.iter2
                   (fun expected got ->
                     equal Stored.packed expected (placed_result got))
                   (f host) (f placed)))
           cases);
      group "every dtype" (List.map every_dtype Stored.every);
    ]

let () = exit (run "nx conformance" (List.map conformance backends))
