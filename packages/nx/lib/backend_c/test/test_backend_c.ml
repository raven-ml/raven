(* C binding invariants that cannot be stated through the backend-neutral Nx
   contract. *)

open Windtrap
module B = Nx_backend
module F = Nx_core.Make_frontend (B)
module Dtype = Nx_core.Dtype

external dtype_tag : ('a, 'b) Nx_buffer.t -> int = "caml_nx_c_dtype_tag"

let ctx = B.create_context ()

let fexact =
  testable ~pp:(fun ppf x -> Format.fprintf ppf "%g" x) ~equal:( = ) ()

let tests =
  group "binding-abi"
    [
      test "t-field-order" (fun () ->
          (* C reads buffer/shape/strides/offset at record slots 0-3. An
             offset, non-contiguous view makes a slot mismatch observable. *)
          let base =
            F.create ctx F.float64 [| 3; 4 |]
              (Array.init 12 (fun i -> float_of_int i))
          in
          let input = B.shrink base [| (0, 2); (1, 4) |] in
          let expected = [| -1.; -2.; -3.; -5.; -6.; -7. |] in
          equal ~msg:"neg over strided offset view" (array fexact) expected
            (F.to_array (B.neg input)));
      test "kind-to-tag" (fun () ->
          List.iter
            (fun (Dtype.Pack dt as packed) ->
              let buffer = Nx_buffer.create dt 1 in
              equal
                ~msg:(Printf.sprintf "%s tag"
                        (Dtype.Packed.to_string packed))
                int (Dtype.Packed.tag packed) (dtype_tag buffer))
            Dtype.Packed.all);
      test "linalg-error-translation" (fun () ->
          let raises_linalg kind thunk =
            match thunk () with
            | _ -> fail "expected Linalg_error, got a normal result"
            | exception Nx_core.Backend_intf.Linalg_error error ->
                equal ~msg:"kind" bool true (error.kind = kind)
            | exception exn ->
                fail
                  ("expected Linalg_error, got " ^ Printexc.to_string exn)
          in
          let not_positive_definite =
            F.create ctx F.float64 [| 2; 2 |] [| 1.; 2.; 2.; 1. |]
          in
          raises_linalg `Not_positive_definite (fun () ->
              B.cholesky ~upper:false not_positive_definite);
          let singular =
            F.create ctx F.float64 [| 2; 2 |] [| 0.; 0.; 1.; 2. |]
          in
          let rhs = F.create ctx F.float64 [| 2; 1 |] [| 1.; 2. |] in
          raises_linalg `Singular (fun () ->
              B.triangular_solve ~upper:false ~transpose:false ~unit_diag:false
                singular rhs));
    ]

let () = Windtrap.run "nx C backend ABI" [ tests ]
