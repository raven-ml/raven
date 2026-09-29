(* C binding invariants that cannot be stated through the backend-neutral Nx
   contract. *)

open Windtrap
module B = Nx_backend
module F = Nx_core.Make_frontend (B)

let ctx = B.create_context ()

(* The bytes of two elements, the class bits and whether the row is a signed
   integer, as the engine sees the dtype of a tensor. *)
external dtype_facts : ('a, 'b) B.t -> int * int * bool
  = "caml_nx_c_dtype_facts"

(* nx_c.h's NX_C_CLASS_ bits *)
let class_int = 0x01
let class_float = 0x02
let class_complex = 0x04
let class_bool = 0x08
let class_packed = 0x10

type dtype = Dtype : ('a, 'b) Nx_dtype.t -> dtype

let fexact =
  Testable.make ~pp:(fun ppf x -> Format.fprintf ppf "%g" x) ~equal:( = )

let tests =
  group "binding-abi"
    [
      test "t-field-order" (fun () ->
          (* C reads buffer/shape/strides/offset/dtype at record slots 0-4. An
             offset, non-contiguous view makes a slot mismatch observable. *)
          let base =
            F.create ctx F.float64 [| 3; 4 |]
              (Array.init 12 (fun i -> float_of_int i))
          in
          let input = B.shrink base [| (0, 2); (1, 4) |] in
          let expected = [| -1.; -2.; -3.; -5.; -6.; -7. |] in
          equal ~msg:"neg over strided offset view" (array fexact) expected
            (F.to_array (B.neg input)));
      test "dtype-rows" (fun () ->
          List.iter
            (fun (Dtype dt) ->
              let bits = Nx_dtype.Scalar.(bitsize (of_dtype dt)) in
              let flag b c = if b then c else 0 in
              let expected =
                ( 2 * bits / 8,
                  flag (Nx_dtype.is_int dt) class_int
                  lor flag (Nx_dtype.is_float dt) class_float
                  lor flag (Nx_dtype.is_complex dt) class_complex
                  lor flag (Nx_dtype.equal dt Nx_dtype.bool) class_bool
                  lor flag (bits = 4) class_packed,
                  Nx_dtype.is_int dt && not (Nx_dtype.is_uint dt) )
              in
              equal ~msg:(Nx_dtype.to_string dt) (triple int int bool) expected
                (dtype_facts (B.buffer ctx dt [| 1 |])))
            Nx_dtype.
              [
                Dtype float16;
                Dtype float32;
                Dtype float64;
                Dtype bfloat16;
                Dtype float8_e4m3;
                Dtype float8_e5m2;
                Dtype int4;
                Dtype uint4;
                Dtype int8;
                Dtype uint8;
                Dtype int16;
                Dtype uint16;
                Dtype int32;
                Dtype uint32;
                Dtype int64;
                Dtype uint64;
                Dtype complex64;
                Dtype complex128;
                Dtype bool;
              ]);
      test "linalg-error-translation" (fun () ->
          let raises_linalg kind thunk =
            match thunk () with
            | _ -> fail "expected Linalg_error, got a normal result"
            | exception Nx_core.Backend_intf.Linalg_error error ->
                equal ~msg:"kind" bool true (error.kind = kind)
            | exception exn ->
                fail ("expected Linalg_error, got " ^ Printexc.to_string exn)
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
              B.solve_triangular ~upper:false ~transpose:false ~unit_diag:false
                singular rhs));
    ]

let () = exit (Windtrap.run "nx C backend ABI" [ tests ])
