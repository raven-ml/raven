(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Kernels = Kernels
module View = Nx_core.View
module E = Nx_effect

(* A value's storage: the kernels' tensor, whose view nx keeps in the value. *)
type E.storage += Arrays : ('a, 'b) Kernels.t -> E.storage

let name = "nx-oxcaml"
let context = lazy (Kernels.create_context ())
let refuse op why = raise (E.Backend.Refused (name ^ ": " ^ op ^ ": " ^ why))

(* The dtypes the kernels hold in arrays. *)
let holds : type a b. (a, b) Nx_dtype.t -> bool = function
  | Nx_dtype.Float64 | Nx_dtype.Float32 | Nx_dtype.Int8 | Nx_dtype.Int16
  | Nx_dtype.Int32 | Nx_dtype.Int64 | Nx_dtype.Bool ->
      true
  | _ -> false

let check op dtype =
  if not (holds dtype) then refuse op ("no " ^ Nx_dtype.to_string dtype)

(* The placement of an operation's result: its placed operands'. A host operand
   would be a silent copy, so it raises as operands of two placements do. *)
let at op xs =
  let p =
    List.find_map
      (fun (E.P x) -> match x with E.Placed r -> Some r.r_placement | _ -> None)
      xs
  in
  let p = match p with Some p -> p | None -> E.Placement.host in
  List.iter
    (fun (E.P x) ->
      match x with
      | E.Host _ -> E.mixed op p E.Placement.host
      | E.Traced _ -> E.outside_trace ()
      | E.Placed _ -> ())
    xs;
  p

(* The kernels' tensor of an operand: its arrays under its view. A value nx
   holds itself is its one element, broadcast. *)
let project : type a b. string -> (a, b) E.t -> (a, b) Kernels.t =
 fun op x ->
  match x with
  | E.Placed r -> (
      check op r.r_dtype;
      match E.Cell.state r.r_cell with
      | Live (Arrays k) -> (
          match Nx_dtype.equal_witness (Kernels.dtype k) r.r_dtype with
          | Some Type.Equal -> { k with Kernels.view = r.r_view }
          | None -> invalid_arg (op ^ ": nx-oxcaml storage of another dtype"))
      | Live (E.Held _) ->
          Kernels.reshape
            (Kernels.from_host (Lazy.force context) r.r_dtype
               (E.read_elements r))
            (View.shape r.r_view)
      | Live _ -> invalid_arg (op ^ ": storage nx-oxcaml does not hold")
      | Consumed k -> E.consumed k)
  | E.Host _ -> invalid_arg (op ^ ": a host value is not nx-oxcaml's")
  | E.Traced _ -> E.outside_trace ()

(* A result of the kernels as a value at [p], its own storage. *)
let inject p k =
  let v = Kernels.view k in
  E.placed p (Kernels.dtype k) v
    (E.cell ~placement:p ~length:(View.numel v) (Arrays k))

let unary op f x =
  let p = at op [ E.P x ] in
  inject p (f (project op x))

let binary op f a b =
  let p = at op [ E.P a; E.P b ] in
  inject p (f (project op a) (project op b))

let creation op p dtype make =
  check op dtype;
  inject p (make (Lazy.force context))

module Backend = struct
  let name = name
  let runs_on = E.Device.is_host

  let place p x =
    let dtype = E.dtype x in
    check "place" dtype;
    let shape = View.shape (E.view x) in
    let elements = E.elements x in
    inject p
      (Kernels.reshape (Kernels.from_host (Lazy.force context) dtype elements) shape)

  let to_host x = Kernels.to_host (Kernels.contiguous (project "to_host" x))

  let buffer p dtype shape =
    creation "buffer" p dtype (fun c -> Kernels.buffer c dtype shape)

  let full p dtype shape value =
    creation "full" p dtype (fun c -> Kernels.full c dtype shape value)

  let from_host p dtype buffer =
    creation "from_host" p dtype (fun c -> Kernels.from_host c dtype buffer)

  let add a b = binary "add" Kernels.add a b
  let sub a b = binary "sub" Kernels.sub a b
  let mul a b = binary "mul" Kernels.mul a b
  let fdiv a b = binary "div" Kernels.fdiv a b
  let idiv a b = binary "div" Kernels.idiv a b
  let mod_ a b = binary "mod" Kernels.mod_ a b
  let pow a b = binary "pow" Kernels.pow a b
  let atan2 a b = binary "atan2" Kernels.atan2 a b
  let cmpeq a b = binary "equal" Kernels.cmpeq a b
  let cmpne a b = binary "not_equal" Kernels.cmpne a b
  let cmplt a b = binary "less" Kernels.cmplt a b
  let cmple a b = binary "less_equal" Kernels.cmple a b
  let max a b = binary "max" Kernels.max a b
  let min a b = binary "min" Kernels.min a b
  let xor a b = binary "xor" Kernels.xor a b
  let or_ a b = binary "or" Kernels.or_ a b
  let and_ a b = binary "and" Kernels.and_ a b
  let neg x = unary "neg" Kernels.neg x
  let recip x = unary "recip" Kernels.recip x
  let abs x = unary "abs" Kernels.abs x
  let sqrt x = unary "sqrt" Kernels.sqrt x
  let sign x = unary "sign" Kernels.sign x
  let exp x = unary "exp" Kernels.exp x
  let log x = unary "log" Kernels.log x
  let sin x = unary "sin" Kernels.sin x
  let cos x = unary "cos" Kernels.cos x
  let tan x = unary "tan" Kernels.tan x
  let asin x = unary "asin" Kernels.asin x
  let acos x = unary "acos" Kernels.acos x
  let atan x = unary "atan" Kernels.atan x
  let sinh x = unary "sinh" Kernels.sinh x
  let cosh x = unary "cosh" Kernels.cosh x
  let tanh x = unary "tanh" Kernels.tanh x
  let trunc x = unary "trunc" Kernels.trunc x
  let ceil x = unary "ceil" Kernels.ceil x
  let floor x = unary "floor" Kernels.floor x
  let round x = unary "round" Kernels.round x
  let erf x = unary "erf" Kernels.erf x

  let where c a b =
    let p = at "where" [ E.P c; E.P a; E.P b ] in
    inject p
      (Kernels.where (project "where" c) (project "where" a)
         (project "where" b))

  let reduce ~op ~axes x = unary "reduce" (Kernels.reduce ~op ~axes) x
  let argmax ~axis ~keepdims x = unary "argmax" (Kernels.argmax ~axis ~keepdims) x
  let argmin ~axis ~keepdims x = unary "argmin" (Kernels.argmin ~axis ~keepdims) x

  let associative_scan ~axis ~op x =
    unary "associative_scan" (Kernels.associative_scan ~axis ~op) x

  let sort ~axis ~descending x = unary "sort" (Kernels.sort ~axis ~descending) x

  let argsort ~axis ~descending x =
    unary "argsort" (Kernels.argsort ~axis ~descending) x

  (* A movement is nx's view arithmetic over the same arrays. *)
  let expand = E.Backend.Host.expand
  let reshape = E.Backend.Host.reshape
  let permute = E.Backend.Host.permute
  let shrink = E.Backend.Host.shrink
  let flip = E.Backend.Host.flip
  let sliding_window = E.Backend.Host.sliding_window

  let pad x padding value =
    unary "pad" (fun k -> Kernels.pad k padding value) x

  let cat xs ~axis =
    let p = at "concatenate" (List.map (fun x -> E.P x) xs) in
    inject p (Kernels.cat (List.map (project "concatenate") xs) ~axis)

  let cast ~dtype x =
    check "cast" dtype;
    unary "cast" (Kernels.cast ~dtype) x

  let bitcast ~dtype x =
    check "bitcast" dtype;
    unary "bitcast" (Kernels.bitcast ~dtype) x

  (* A value whose view covers its arrays is already contiguous. *)
  let contiguous x =
    match x with
    | E.Placed r when E.covers r -> E.Backend.Host.contiguous x
    | _ -> unary "contiguous" Kernels.contiguous x

  let copy x = unary "copy" Kernels.copy x
  let threefry key ctr = binary "threefry" Kernels.threefry key ctr
  let gather data indices ~axis =
    binary "take" (fun d i -> Kernels.gather d i ~axis) data indices

  let scatter ~mode ~unique_indices data ~indices ~updates ~axis =
    let op = "scatter" in
    let p = at op [ E.P data; E.P indices; E.P updates ] in
    inject p
      (Kernels.scatter ~mode ~unique_indices (project op data)
         ~indices:(project op indices) ~updates:(project op updates) ~axis)

  let update x ~starts v =
    let op = "set" in
    let p = at op [ E.P x; E.P starts; E.P v ] in
    inject p
      (Kernels.update (project op x) ~starts:(project op starts) (project op v))

  let unfold x ~kernel_size ~stride ~dilation ~padding =
    unary "unfold"
      (fun k -> Kernels.unfold k ~kernel_size ~stride ~dilation ~padding)
      x

  let fold x ~output_size ~kernel_size ~stride ~dilation ~padding =
    unary "fold"
      (fun k -> Kernels.fold k ~output_size ~kernel_size ~stride ~dilation ~padding)
      x

  let matmul a b = binary "matmul" Kernels.matmul a b

  (* The kernels have no Fourier transform and no linear algebra beyond the
     product. *)
  let fft _ ~axes:_ = refuse "fft" "not implemented"
  let ifft _ ~axes:_ = refuse "ifft" "not implemented"
  let rfft _ ~dtype:_ ~axes:_ = refuse "rfft" "not implemented"
  let irfft ?s:_ _ ~dtype:_ ~axes:_ = refuse "irfft" "not implemented"
  let cholesky ~upper:_ _ = refuse "cholesky" "not implemented"
  let qr ~reduced:_ _ = refuse "qr" "not implemented"
  let lu _ = refuse "lu" "not implemented"
  let svd ~full_matrices:_ _ = refuse "svd" "not implemented"
  let eigvals _ = refuse "eigvals" "not implemented"
  let eig _ = refuse "eig" "not implemented"
  let eigvalsh _ = refuse "eigvalsh" "not implemented"
  let eigh _ = refuse "eigh" "not implemented"

  let solve_triangular ~upper:_ ~transpose:_ ~unit_diag:_ _ _ =
    refuse "solve_triangular" "not implemented"
end

let backend : E.Backend.t = (module Backend)
