(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type unary =
  | Neg
  | Recip
  | Abs
  | Sqrt
  | Sign
  | Exp
  | Log
  | Sin
  | Cos
  | Tan
  | Asin
  | Acos
  | Atan
  | Sinh
  | Cosh
  | Tanh
  | Trunc
  | Ceil
  | Floor
  | Round
  | Erf

type binary =
  | Add
  | Sub
  | Mul
  | Fdiv
  | Idiv
  | Mod
  | Pow
  | Atan2
  | Maximum
  | Minimum
  | And
  | Or
  | Xor

type compare = Equal | Not_equal | Less | Less_equal
type reduce = Sum | Prod | Max | Min
type arg_reduce = Argmax | Argmin

exception Refused of string

exception
  Linalg_error of {
    op : string;
    kind : [ `Not_positive_definite | `Singular | `No_convergence ];
  }

let () =
  Printexc.register_printer (function
    | Refused reason -> Some (Printf.sprintf "Nx_backend.Refused(%S)" reason)
    | Linalg_error { op; kind } ->
        let detail =
          match kind with
          | `Not_positive_definite -> "matrix is not positive-definite"
          | `Singular -> "matrix is singular"
          | `No_convergence -> "algorithm failed to converge"
        in
        Some (Printf.sprintf "Nx.Linalg_error(%s): %s" op detail)
    | _ -> None)

type int32_array = (int32, Nx_dtype.int32_elt) Nx_array.t

module type S = sig
  val name : string
  val runs_on : Nx_device.t -> bool
  val unary : unary -> ('a, 'b) Nx_array.t -> dst:('a, 'b) Nx_array.t -> unit

  val binary :
    binary ->
    ('a, 'b) Nx_array.t ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit

  val compare :
    compare ->
    ('a, 'b) Nx_array.t ->
    ('a, 'b) Nx_array.t ->
    dst:(bool, Nx_dtype.bool_elt) Nx_array.t ->
    unit

  val where :
    (bool, Nx_dtype.bool_elt) Nx_array.t ->
    ('a, 'b) Nx_array.t ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit

  val cast : ('a, 'b) Nx_array.t -> dst:('c, 'd) Nx_array.t -> unit
  val threefry : int32_array -> int32_array -> dst:int32_array -> unit

  val reduce :
    reduce ->
    axes:int array ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit

  val scan :
    reduce -> axis:int -> ('a, 'b) Nx_array.t -> dst:('a, 'b) Nx_array.t -> unit

  val arg_reduce :
    arg_reduce -> axis:int -> ('a, 'b) Nx_array.t -> dst:int32_array -> unit

  val sort :
    descending:bool ->
    axis:int ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit

  val argsort :
    descending:bool ->
    axis:int ->
    ('a, 'b) Nx_array.t ->
    dst:int32_array ->
    unit

  val pad :
    (int * int) array ->
    'a ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit

  val cat :
    axis:int -> ('a, 'b) Nx_array.t list -> dst:('a, 'b) Nx_array.t -> unit

  val contiguous : ('a, 'b) Nx_array.t -> dst:('a, 'b) Nx_array.t -> unit

  val gather :
    axis:int ->
    int32_array ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit

  val scatter :
    mode:[ `Set | `Add ] ->
    unique:bool ->
    axis:int ->
    indices:int32_array ->
    updates:('a, 'b) Nx_array.t ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit

  val update :
    ('a, 'b) Nx_array.t ->
    starts:int32_array ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit

  val unfold :
    kernel_size:int array ->
    stride:int array ->
    dilation:int array ->
    padding:(int * int) array ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit

  val fold :
    output_size:int array ->
    kernel_size:int array ->
    stride:int array ->
    dilation:int array ->
    padding:(int * int) array ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit

  val matmul :
    ('a, 'b) Nx_array.t ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit

  val fft :
    inverse:bool ->
    axes:int array ->
    (Complex.t, 'b) Nx_array.t ->
    dst:(Complex.t, 'b) Nx_array.t ->
    unit

  val rfft :
    axes:int array ->
    (float, 'b) Nx_array.t ->
    dst:(Complex.t, 'c) Nx_array.t ->
    unit

  val irfft :
    axes:int array ->
    s:int array option ->
    (Complex.t, 'b) Nx_array.t ->
    dst:(float, 'c) Nx_array.t ->
    unit

  val cholesky :
    upper:bool -> ('a, 'b) Nx_array.t -> dst:('a, 'b) Nx_array.t -> unit

  val qr :
    reduced:bool ->
    ('a, 'b) Nx_array.t ->
    q:('a, 'b) Nx_array.t ->
    r:('a, 'b) Nx_array.t ->
    unit

  val lu :
    ('a, 'b) Nx_array.t ->
    lu:('a, 'b) Nx_array.t ->
    pivots:int32_array ->
    perm:int32_array ->
    unit

  val svd :
    ('a, 'b) Nx_array.t ->
    u:('a, 'b) Nx_array.t ->
    s:(float, Nx_dtype.float64_elt) Nx_array.t ->
    vt:('a, 'b) Nx_array.t ->
    unit

  val eig :
    ('a, 'b) Nx_array.t ->
    values:(Complex.t, Nx_dtype.complex64_elt) Nx_array.t ->
    vectors:(Complex.t, Nx_dtype.complex64_elt) Nx_array.t option ->
    unit

  val eigh :
    ('a, 'b) Nx_array.t ->
    values:(float, Nx_dtype.float64_elt) Nx_array.t ->
    vectors:('a, 'b) Nx_array.t option ->
    unit

  val solve_triangular :
    upper:bool ->
    transpose:bool ->
    unit_diag:bool ->
    ('a, 'b) Nx_array.t ->
    ('a, 'b) Nx_array.t ->
    dst:('a, 'b) Nx_array.t ->
    unit
end

type t = { kernels : (module S) }

let make kernels = { kernels }
let kernels b = b.kernels
let name { kernels = (module B) } = B.name
let runs_on { kernels = (module B) } d = B.runs_on d
let equal = ( == )
