(* XLA Compiler for Rune

   This module provides an alternative JIT compilation path using XLA
   (Accelerated Linear Algebra). It is completely independent from the existing
   custom JIT infrastructure. *)

(* A unique identifier for symbolic variables in the graph *)
module Var = struct
  type t = int

  let counter = ref 0

  let fresh () =
    incr counter;
    !counter

  let equal = Int.equal
  let hash = Fun.id

  module Map = Map.Make (Int)

  module Table = Hashtbl.Make (struct
    type nonrec t = t

    let equal = equal
    let hash = hash
  end)
end

(* An atom can be a variable or a concrete tensor value (constant) *)
type atom =
  | Var of Var.t
  | Const of Xla.Literal.t (* Store XLA literal directly *)

(* Shape and dtype info for variables *)
type var_info = { shape : int array; dtype : Nx_core.Dtype.packed }

(* Define unary operation types *)
type unary_op = Neg | Sin | Sqrt | Exp2 | Log2 | Recip

(* Define binary operation types *)
type binary_op = Add | Mul | Div | Pow | Max | Mod | And | Or | Xor | Idiv

(* Define reduction operation types *)
type reduction_op = Sum

(* The operation types we support *)
type op =
  | Unary of unary_op * atom
  | Binary of binary_op * atom * atom
  | Reduce of reduction_op * atom * int array (* op, input, axes *)
  | Transpose of atom * int array (* input, permutation *)
  | Reshape of atom * int array (* input, new_shape *)
  | Expand of atom * int array (* input, new_shape *)
  | Pad of atom * (int * int) array * atom (* input, padding, fill_value *)
  | Shrink of atom * (int * int) array (* input, limits *)
  | Flip of atom * bool array (* input, axes_to_flip *)
  | Cat of atom list * int (* tensors, axis *)
  | Where of atom * atom * atom (* condition, if_true, if_false *)
  | Cast of atom * Nx_core.Dtype.packed (* input, target_dtype *)
  | Comparison of [ `Lt | `Ne ] * atom * atom (* op, a, b *)
  | Matmul of atom * atom (* a, b *)
  | Gather of atom * atom * int (* data, indices, axis *)
  | Scatter of atom * atom * atom * int * [ `Set | `Add ]
    (* data_template, indices, updates, axis, mode *)
  | Conv2d of
      atom
      * atom
      * int array
      * (int * int) array
      * int array (* input, kernel, strides, padding, dilation *)
  | Fold of atom * int array * int array * int array * (int * int) array
(* input, output_size, kernel_size, strides, padding *)

(* The graph itself *)
type expression = {
  inputs : Var.t list;
  outputs : Var.t list;
  equations : (Var.t * op) list;
  var_info : var_info Var.Map.t;
}

(* Create a hashtable module for Symbolic_id *)
module Symbolic_id_table = Hashtbl.Make (struct
  type t = Nx_rune.Symbolic_id.t

  let equal = Nx_rune.Symbolic_id.equal
  let hash = Nx_rune.Symbolic_id.hash
end)

(* Tracer state *)
type tracer_state = {
  mutable equations : (Var.t * op) list;
  mutable var_info : var_info Var.Map.t;
  sym_to_var : Var.t Symbolic_id_table.t;
}

let get_packed_dtype : type a b. (a, b) Nx_core.Dtype.t -> Nx_core.Dtype.packed
    =
 fun dtype -> Nx_core.Dtype.Pack dtype

let get_var_info (tensor : ('a, 'b) Nx_rune.t) : var_info =
  let view = Nx_rune.view tensor in
  let shape = Nx_core.View.shape view in
  let dtype = Nx_rune.dtype tensor in
  { shape; dtype = get_packed_dtype dtype }

(* Convert a tensor to XLA literal - this extracts data from the tensor *)
let tensor_to_literal (type a b) (tensor : (a, b) Nx_rune.t) : Xla.Literal.t =
  match tensor with
  | Nx_rune.Ocaml_tensor native_t ->
      (* For OCaml backend, the buffer is the bigarray *)
      let arr = Nx_native.data native_t in
      Xla.Literal.of_bigarray
        (Obj.magic arr : (_, _, Bigarray.c_layout) Bigarray.Genarray.t)
  | Nx_rune.C_tensor c_t ->
      (* For C backend, similar approach *)
      let arr = Nx_c.data c_t in
      Xla.Literal.of_bigarray
        (Obj.magic arr : (_, _, Bigarray.c_layout) Bigarray.Genarray.t)
  | Nx_rune.Metal_tensor metal_t ->
      (* For Metal backend, get the data from GPU *)
      let arr = Rune_metal.data metal_t in
      Xla.Literal.of_bigarray
        (Obj.magic arr : (_, _, Bigarray.c_layout) Bigarray.Genarray.t)
  | Nx_rune.Symbolic_tensor _ ->
      failwith "XLA: Cannot convert symbolic tensor to literal"

(* Convert a tensor to an atom, registering it in the tracer state if needed *)
let to_atom (state : tracer_state) (tensor : ('a, 'b) Nx_rune.t) : atom =
  match tensor with
  | Nx_rune.Symbolic_tensor { id; dtype = _; shape = _ } -> (
      match Symbolic_id_table.find_opt state.sym_to_var id with
      | Some var -> Var var
      | None ->
          (* This shouldn't happen in normal tracing *)
          failwith "Untracked symbolic tensor in XLA tracer")
  | _ ->
      (* It's a concrete tensor, convert to XLA literal *)
      let literal = tensor_to_literal tensor in
      Const literal

(* Create a new symbolic tensor from a variable *)
let symbolic_from_var (state : tracer_state) (var : Var.t)
    (dtype : ('a, 'b) Nx_core.Dtype.t) (shape : int array) : ('a, 'b) Nx_rune.t
    =
  let sym_id = Nx_rune.Symbolic_id.fresh () in
  Symbolic_id_table.add state.sym_to_var sym_id var;
  state.var_info <-
    Var.Map.add var
      (get_var_info (Nx_rune.Symbolic_tensor { id = sym_id; dtype; shape }))
      state.var_info;
  Nx_rune.Symbolic_tensor { id = sym_id; dtype; shape }

(* Add an equation and return the output symbolic tensor *)
let add_equation (state : tracer_state) (op : op) ~dtype ~shape :
    ('a, 'b) Nx_rune.t =
  let out_var = Var.fresh () in
  state.equations <- (out_var, op) :: state.equations;
  symbolic_from_var state out_var dtype shape

(* Build expression graph by tracing the function *)
let build_expr (f : ('a, 'b) Nx_rune.t -> ('c, 'd) Nx_rune.t)
    (initial_input : ('a, 'b) Nx_rune.t) : expression =
  let state =
    {
      equations = [];
      var_info = Var.Map.empty;
      sym_to_var = Symbolic_id_table.create 16;
    }
  in

  (* Create input variable and symbolic tensor *)
  let input_var = Var.fresh () in
  let input_shape = Nx_core.View.shape (Nx_rune.view initial_input) in
  let input_dtype = Nx_rune.dtype initial_input in
  let input_symbolic =
    symbolic_from_var state input_var input_dtype input_shape
  in

  (* Run the function with effect handler to trace operations *)
  let run_tracer input_tensor =
    let handler =
      let open Effect.Deep in
      {
        retc = (fun result -> result);
        exnc = raise;
        effc =
          (fun (type a) (eff : a Effect.t) ->
            match eff with
            | Nx_rune.E_add { a; b } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let a_atom = to_atom state a in
                    let b_atom = to_atom state b in
                    let out =
                      add_equation state
                        (Binary (Add, a_atom, b_atom))
                        ~dtype:(Nx_rune.dtype a)
                        ~shape:(Nx_core.View.shape (Nx_rune.view a))
                    in
                    continue k out)
            | Nx_rune.E_mul { a; b } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let a_atom = to_atom state a in
                    let b_atom = to_atom state b in
                    let out =
                      add_equation state
                        (Binary (Mul, a_atom, b_atom))
                        ~dtype:(Nx_rune.dtype a)
                        ~shape:(Nx_core.View.shape (Nx_rune.view a))
                    in
                    continue k out)
            | Nx_rune.E_fdiv { a; b } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let a_atom = to_atom state a in
                    let b_atom = to_atom state b in
                    let out =
                      add_equation state
                        (Binary (Div, a_atom, b_atom))
                        ~dtype:(Nx_rune.dtype a)
                        ~shape:(Nx_core.View.shape (Nx_rune.view a))
                    in
                    continue k out)
            | Nx_rune.E_neg { t_in } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let in_atom = to_atom state t_in in
                    let out =
                      add_equation state
                        (Unary (Neg, in_atom))
                        ~dtype:(Nx_rune.dtype t_in)
                        ~shape:(Nx_core.View.shape (Nx_rune.view t_in))
                    in
                    continue k out)
            | Nx_rune.E_sin { t_in } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let in_atom = to_atom state t_in in
                    let out =
                      add_equation state
                        (Unary (Sin, in_atom))
                        ~dtype:(Nx_rune.dtype t_in)
                        ~shape:(Nx_core.View.shape (Nx_rune.view t_in))
                    in
                    continue k out)
            | Nx_rune.E_sqrt { t_in } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let in_atom = to_atom state t_in in
                    let out =
                      add_equation state
                        (Unary (Sqrt, in_atom))
                        ~dtype:(Nx_rune.dtype t_in)
                        ~shape:(Nx_core.View.shape (Nx_rune.view t_in))
                    in
                    continue k out)
            | Nx_rune.E_exp2 { t_in } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let in_atom = to_atom state t_in in
                    let out =
                      add_equation state
                        (Unary (Exp2, in_atom))
                        ~dtype:(Nx_rune.dtype t_in)
                        ~shape:(Nx_core.View.shape (Nx_rune.view t_in))
                    in
                    continue k out)
            | Nx_rune.E_log2 { t_in } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let in_atom = to_atom state t_in in
                    let out =
                      add_equation state
                        (Unary (Log2, in_atom))
                        ~dtype:(Nx_rune.dtype t_in)
                        ~shape:(Nx_core.View.shape (Nx_rune.view t_in))
                    in
                    continue k out)
            | Nx_rune.E_reduce_sum { t_in; axes; keepdims } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let in_atom = to_atom state t_in in
                    let result_shape =
                      if keepdims then (
                        let shape =
                          Array.copy (Nx_core.View.shape (Nx_rune.view t_in))
                        in
                        Array.iter (fun axis -> shape.(axis) <- 1) axes;
                        shape)
                      else
                        let shape =
                          Array.to_list (Nx_core.View.shape (Nx_rune.view t_in))
                        in
                        let axes_list = Array.to_list axes in
                        List.filteri
                          (fun i _ -> not (List.mem i axes_list))
                          shape
                        |> Array.of_list
                    in
                    let out =
                      add_equation state
                        (Reduce (Sum, in_atom, axes))
                        ~dtype:(Nx_rune.dtype t_in) ~shape:result_shape
                    in
                    continue k out)
            | Nx_rune.E_reshape { t_in; new_shape } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let in_atom = to_atom state t_in in
                    let out =
                      add_equation state
                        (Reshape (in_atom, new_shape))
                        ~dtype:(Nx_rune.dtype t_in) ~shape:new_shape
                    in
                    continue k out)
            | Nx_rune.E_permute { t_in; axes } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let in_atom = to_atom state t_in in
                    let old_shape = Nx_core.View.shape (Nx_rune.view t_in) in
                    let new_shape =
                      Array.init (Array.length axes) (fun i ->
                          old_shape.(axes.(i)))
                    in
                    let out =
                      add_equation state
                        (Transpose (in_atom, axes))
                        ~dtype:(Nx_rune.dtype t_in) ~shape:new_shape
                    in
                    continue k out)
            | Nx_rune.E_pow { a; b } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let a_atom = to_atom state a in
                    let b_atom = to_atom state b in
                    let out =
                      add_equation state
                        (Binary (Pow, a_atom, b_atom))
                        ~dtype:(Nx_rune.dtype a)
                        ~shape:(Nx_core.View.shape (Nx_rune.view a))
                    in
                    continue k out)
            | Nx_rune.E_max { a; b } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let a_atom = to_atom state a in
                    let b_atom = to_atom state b in
                    let out =
                      add_equation state
                        (Binary (Max, a_atom, b_atom))
                        ~dtype:(Nx_rune.dtype a)
                        ~shape:(Nx_core.View.shape (Nx_rune.view a))
                    in
                    continue k out)
            | Nx_rune.E_mod { a; b } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let a_atom = to_atom state a in
                    let b_atom = to_atom state b in
                    let out =
                      add_equation state
                        (Binary (Mod, a_atom, b_atom))
                        ~dtype:(Nx_rune.dtype a)
                        ~shape:(Nx_core.View.shape (Nx_rune.view a))
                    in
                    continue k out)
            | Nx_rune.E_idiv { a; b } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let a_atom = to_atom state a in
                    let b_atom = to_atom state b in
                    let out =
                      add_equation state
                        (Binary (Idiv, a_atom, b_atom))
                        ~dtype:(Nx_rune.dtype a)
                        ~shape:(Nx_core.View.shape (Nx_rune.view a))
                    in
                    continue k out)
            | Nx_rune.E_and { a; b } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let a_atom = to_atom state a in
                    let b_atom = to_atom state b in
                    let out =
                      add_equation state
                        (Binary (And, a_atom, b_atom))
                        ~dtype:(Nx_rune.dtype a)
                        ~shape:(Nx_core.View.shape (Nx_rune.view a))
                    in
                    continue k out)
            | Nx_rune.E_or { a; b } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let a_atom = to_atom state a in
                    let b_atom = to_atom state b in
                    let out =
                      add_equation state
                        (Binary (Or, a_atom, b_atom))
                        ~dtype:(Nx_rune.dtype a)
                        ~shape:(Nx_core.View.shape (Nx_rune.view a))
                    in
                    continue k out)
            | Nx_rune.E_xor { a; b } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let a_atom = to_atom state a in
                    let b_atom = to_atom state b in
                    let out =
                      add_equation state
                        (Binary (Xor, a_atom, b_atom))
                        ~dtype:(Nx_rune.dtype a)
                        ~shape:(Nx_core.View.shape (Nx_rune.view a))
                    in
                    continue k out)
            | Nx_rune.E_recip { t_in } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let in_atom = to_atom state t_in in
                    let out =
                      add_equation state
                        (Unary (Recip, in_atom))
                        ~dtype:(Nx_rune.dtype t_in)
                        ~shape:(Nx_core.View.shape (Nx_rune.view t_in))
                    in
                    continue k out)
            | Nx_rune.E_cmplt { a; b } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let a_atom = to_atom state a in
                    let b_atom = to_atom state b in
                    let out =
                      add_equation state
                        (Comparison (`Lt, a_atom, b_atom))
                        ~dtype:Nx_core.Dtype.uint8
                        ~shape:(Nx_core.View.shape (Nx_rune.view a))
                    in
                    continue k out)
            | Nx_rune.E_cmpne { a; b } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let a_atom = to_atom state a in
                    let b_atom = to_atom state b in
                    let out =
                      add_equation state
                        (Comparison (`Ne, a_atom, b_atom))
                        ~dtype:Nx_core.Dtype.uint8
                        ~shape:(Nx_core.View.shape (Nx_rune.view a))
                    in
                    continue k out)
            | Nx_rune.E_expand { t_in; new_target_shape } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let in_atom = to_atom state t_in in
                    let out =
                      add_equation state
                        (Expand (in_atom, new_target_shape))
                        ~dtype:(Nx_rune.dtype t_in) ~shape:new_target_shape
                    in
                    continue k out)
            | Nx_rune.E_pad { t_in; padding_config; fill_value } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let in_atom = to_atom state t_in in
                    let fill_atom =
                      Const
                        (tensor_to_literal
                           (Nx_rune.Ocaml_tensor
                              (Nx_native.op_const_scalar
                                 (Nx_native.create_context ())
                                 fill_value (Nx_rune.dtype t_in))))
                    in
                    let old_shape = Nx_core.View.shape (Nx_rune.view t_in) in
                    let new_shape =
                      Array.mapi
                        (fun i dim ->
                          let pb, pa = padding_config.(i) in
                          dim + pb + pa)
                        old_shape
                    in
                    let out =
                      add_equation state
                        (Pad (in_atom, padding_config, fill_atom))
                        ~dtype:(Nx_rune.dtype t_in) ~shape:new_shape
                    in
                    continue k out)
            | Nx_rune.E_shrink { t_in; limits } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let in_atom = to_atom state t_in in
                    let old_shape = Nx_core.View.shape (Nx_rune.view t_in) in
                    let new_shape =
                      Array.mapi
                        (fun i _ ->
                          let start, stop = limits.(i) in
                          stop - start)
                        old_shape
                    in
                    let out =
                      add_equation state
                        (Shrink (in_atom, limits))
                        ~dtype:(Nx_rune.dtype t_in) ~shape:new_shape
                    in
                    continue k out)
            | Nx_rune.E_flip { t_in; dims_to_flip } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let in_atom = to_atom state t_in in
                    let out =
                      add_equation state
                        (Flip (in_atom, dims_to_flip))
                        ~dtype:(Nx_rune.dtype t_in)
                        ~shape:(Nx_core.View.shape (Nx_rune.view t_in))
                    in
                    continue k out)
            | Nx_rune.E_where { condition; if_true; if_false } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let cond_atom = to_atom state condition in
                    let true_atom = to_atom state if_true in
                    let false_atom = to_atom state if_false in
                    let out =
                      add_equation state
                        (Where (cond_atom, true_atom, false_atom))
                        ~dtype:(Nx_rune.dtype if_true)
                        ~shape:(Nx_core.View.shape (Nx_rune.view if_true))
                    in
                    continue k out)
            | Nx_rune.E_cast { t_in; target_dtype } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let in_atom = to_atom state t_in in
                    let out =
                      add_equation state
                        (Cast (in_atom, Nx_core.Dtype.Pack target_dtype))
                        ~dtype:target_dtype
                        ~shape:(Nx_core.View.shape (Nx_rune.view t_in))
                    in
                    continue k out)
            | Nx_rune.E_cat { t_list; axis } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let atoms = List.map (to_atom state) t_list in
                    let shapes =
                      List.map
                        (fun t -> Nx_core.View.shape (Nx_rune.view t))
                        t_list
                    in
                    let first_shape = List.hd shapes in
                    let axis =
                      if axis < 0 then axis + Array.length first_shape else axis
                    in
                    let new_shape = Array.copy first_shape in
                    new_shape.(axis) <-
                      List.fold_left
                        (fun acc shape -> acc + shape.(axis))
                        0 shapes;
                    let out =
                      add_equation state
                        (Cat (atoms, axis))
                        ~dtype:(Nx_rune.dtype (List.hd t_list))
                        ~shape:new_shape
                    in
                    continue k out)
            | Nx_rune.E_matmul { a; b } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let a_atom = to_atom state a in
                    let b_atom = to_atom state b in
                    (* Matmul output shape calculation *)
                    let a_shape = Nx_core.View.shape (Nx_rune.view a) in
                    let b_shape = Nx_core.View.shape (Nx_rune.view b) in
                    let out_shape =
                      match (Array.length a_shape, Array.length b_shape) with
                      | 2, 2 -> [| a_shape.(0); b_shape.(1) |]
                      | _ ->
                          failwith
                            "XLA: Matmul currently only supports 2D tensors"
                    in
                    let out =
                      add_equation state
                        (Matmul (a_atom, b_atom))
                        ~dtype:(Nx_rune.dtype a) ~shape:out_shape
                    in
                    continue k out)
            | Nx_rune.E_gather { data; indices; axis } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let data_atom = to_atom state data in
                    let indices_atom = to_atom state indices in
                    (* Output shape is indices shape + data shape without axis
                       dimension *)
                    let data_shape = Nx_core.View.shape (Nx_rune.view data) in
                    let indices_shape =
                      Nx_core.View.shape (Nx_rune.view indices)
                    in
                    let out_shape =
                      let data_dims =
                        Array.to_list data_shape
                        |> List.filteri (fun i _ -> i <> axis)
                      in
                      Array.append indices_shape (Array.of_list data_dims)
                    in
                    let out =
                      add_equation state
                        (Gather (data_atom, indices_atom, axis))
                        ~dtype:(Nx_rune.dtype data) ~shape:out_shape
                    in
                    continue k out)
            | Nx_rune.E_scatter { data_template; indices; updates; axis } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let data_atom = to_atom state data_template in
                    let indices_atom = to_atom state indices in
                    let updates_atom = to_atom state updates in
                    (* Output shape is same as data_template *)
                    let out_shape =
                      Nx_core.View.shape (Nx_rune.view data_template)
                    in
                    (* Default to Set mode for now - proper implementation would
                       track the mode *)
                    let out =
                      add_equation state
                        (Scatter
                           (data_atom, indices_atom, updates_atom, axis, `Set))
                        ~dtype:(Nx_rune.dtype data_template)
                        ~shape:out_shape
                    in
                    continue k out)
            | Nx_rune.E_unfold { t_in; kernel_size; stride; dilation; padding }
              ->
                Some
                  (fun (k : (a, _) continuation) ->
                    (* For unfold, we can implement it using conv2d with an identity kernel *)
                    (* This is a simplified approach - real unfold would need more complex handling *)
                    let in_atom = to_atom state t_in in
                    let in_shape = Nx_core.View.shape (Nx_rune.view t_in) in
                    (* Create identity kernel for unfold *)
                    let kernel_h, kernel_w =
                      (kernel_size.(0), kernel_size.(1))
                    in
                    let channels = in_shape.(1) in
                    (* Create a kernel that extracts patches *)
                    let kernel_atom = Const (Xla.Literal.create_r0_f32 1.0) in
                    (* Simplified - would need proper kernel *)

                    (* Calculate output shape *)
                    let batch = in_shape.(0) in
                    let pad_h_before, pad_h_after = padding.(0) in
                    let pad_w_before, pad_w_after = padding.(1) in
                    let h_out =
                      (in_shape.(2) + pad_h_before + pad_h_after
                      - (dilation.(0) * (kernel_h - 1))
                      - 1)
                      / stride.(0)
                      + 1
                    in
                    let w_out =
                      (in_shape.(3) + pad_w_before + pad_w_after
                      - (dilation.(1) * (kernel_w - 1))
                      - 1)
                      / stride.(1)
                      + 1
                    in
                    let out_shape =
                      [| batch; channels * kernel_h * kernel_w; h_out; w_out |]
                    in

                    let out =
                      add_equation state
                        (Conv2d (in_atom, kernel_atom, stride, padding, dilation))
                        ~dtype:(Nx_rune.dtype t_in) ~shape:out_shape
                    in
                    continue k out)
            | Nx_rune.E_fold
                {
                  t_in;
                  output_size;
                  kernel_size;
                  stride;
                  dilation = _;
                  padding;
                } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    let in_atom = to_atom state t_in in
                    let in_shape = Nx_core.View.shape (Nx_rune.view t_in) in
                    (* Fold is the inverse of unfold - it reconstructs patches into an image *)
                    (* For now, we'll use a simplified implementation *)
                    let batch = in_shape.(0) in
                    let channels_x_kh_x_kw = in_shape.(1) in
                    let h_out, w_out = (output_size.(0), output_size.(1)) in
                    let out_shape =
                      [|
                        batch;
                        channels_x_kh_x_kw / (kernel_size.(0) * kernel_size.(1));
                        h_out;
                        w_out;
                      |]
                    in

                    let out =
                      add_equation state
                        (Fold
                           (in_atom, output_size, kernel_size, stride, padding))
                        ~dtype:(Nx_rune.dtype t_in) ~shape:out_shape
                    in
                    continue k out)
            | Nx_rune.E_copy { t_in } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    (* Copy is just identity in XLA - the data will be copied if
                       needed *)
                    let in_atom = to_atom state t_in in
                    let out =
                      add_equation state
                        (Reshape
                           (in_atom, Nx_core.View.shape (Nx_rune.view t_in)))
                        (* Identity reshape *)
                        ~dtype:(Nx_rune.dtype t_in)
                        ~shape:(Nx_core.View.shape (Nx_rune.view t_in))
                    in
                    continue k out)
            | Nx_rune.E_const_scalar { context = _; value; dtype } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    (* Create a scalar constant *)
                    let literal =
                      match dtype with
                      | Nx_core.Dtype.Float32 ->
                          Xla.Literal.create_r0_f32 (Obj.magic value : float)
                      | Nx_core.Dtype.Float64 ->
                          Xla.Literal.create_r0_f32 (Obj.magic value : float)
                          (* TODO: r0_f64 *)
                      | Nx_core.Dtype.Int32 ->
                          Xla.Literal.create_r0_f32
                            (Int32.to_float (Obj.magic value : int32))
                      | _ ->
                          failwith "XLA: Unsupported dtype for scalar constant"
                    in
                    let out_var = Var.fresh () in
                    state.equations <-
                      (out_var, Cast (Const literal, get_packed_dtype dtype))
                      :: state.equations;
                    symbolic_from_var state out_var dtype [||] |> continue k)
            | Nx_rune.E_const_array { context = _; array } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    (* Create array constant from bigarray *)
                    (* Convert Array1 to Genarray *)
                    let dims = [| Bigarray.Array1.dim array |] in
                    let genarray =
                      Bigarray.reshape (Bigarray.genarray_of_array1 array) dims
                    in
                    let literal = Xla.Literal.of_bigarray genarray in
                    (* We need to infer dtype from the array kind *)
                    let dtype_pack =
                      match Bigarray.Array1.kind array with
                      | Bigarray.Float32 -> Nx_core.Dtype.Pack Float32
                      | Bigarray.Float64 -> Pack Float64
                      | Bigarray.Int32 -> Pack Int32
                      | Bigarray.Int64 -> Pack Int64
                      | Bigarray.Int8_signed -> Pack Int8
                      | Bigarray.Int8_unsigned -> Pack UInt8
                      | _ -> failwith "XLA: Unsupported array kind"
                    in
                    let out_var = Var.fresh () in
                    (* The array is already 1D, so use its dimension *)
                    let shape = dims in
                    state.equations <-
                      (out_var, Reshape (Const literal, shape))
                      :: state.equations;
                    (* We need to create a symbolic tensor with the correct type *)
                    (* Since we don't know the exact type at compile time, we need to use the effect's continuation type *)
                    state.var_info <-
                      Var.Map.add out_var
                        { shape; dtype = dtype_pack }
                        state.var_info;
                    let sym_id = Nx_rune.Symbolic_id.fresh () in
                    Symbolic_id_table.add state.sym_to_var sym_id out_var;
                    (* Return a symbolic tensor - the type will be unified
                       through the continuation *)
                    let result =
                      Nx_rune.Symbolic_tensor
                        { id = sym_id; dtype = Obj.magic (); shape }
                    in
                    continue k (Obj.magic result))
            | Nx_rune.E_contiguous { t_in } ->
                Some
                  (fun (k : (a, _) continuation) ->
                    (* Contiguous is just identity in XLA - data is always
                       contiguous *)
                    let in_atom = to_atom state t_in in
                    let out =
                      add_equation state
                        (Reshape
                           (in_atom, Nx_core.View.shape (Nx_rune.view t_in)))
                        (* Identity reshape *)
                        ~dtype:(Nx_rune.dtype t_in)
                        ~shape:(Nx_core.View.shape (Nx_rune.view t_in))
                    in
                    continue k out)
            | Nx_rune.E_assign { dst = _; src = _ } ->
                Some
                  (fun (_k : (a, _) continuation) ->
                    (* Assignment is not supported in XLA - fail with meaningful
                       error *)
                    failwith
                      "XLA: In-place assignment (E_assign) is not supported in \
                       JIT compilation. Use functional operations instead.")
            | _ ->
                None (* Let other effects fall through to the default handler *));
      }
    in
    Effect.Deep.match_with f input_tensor handler
  in

  (* Start tracing *)
  let output_symbolic = run_tracer input_symbolic in

  (* Extract output variable *)
  let output_var =
    match output_symbolic with
    | Nx_rune.Symbolic_tensor { id; _ } -> (
        match Symbolic_id_table.find_opt state.sym_to_var id with
        | Some var -> var
        | None -> failwith "Output tensor not tracked in XLA tracer")
    | _ ->
        failwith "XLA JIT function must return a tensor derived from its input"
  in

  {
    inputs = [ input_var ];
    outputs = [ output_var ];
    equations = List.rev state.equations;
    var_info = state.var_info;
  }

(* Packed tensor type for dynamic handling *)
type packed_tensor = Packed_tensor : ('a, 'b) Nx_rune.t -> packed_tensor

(* Compile expression to XLA executable *)
let compile_expr (expr : expression) (device : Nx_rune.device_type) :
    packed_tensor list -> packed_tensor list =
  let builder = Xla.Builder.create "rune_xla" in
  let xla_client =
    match device with
    | Nx_rune.Metal -> (
        (* Use GPU client for Metal device *)
        try Xla.Client.gpu ()
        with _ ->
          (* Fall back to CPU if GPU is not available *)
          Printf.eprintf
            "XLA: GPU client creation failed, falling back to CPU\n";
          Xla.Client.cpu ())
    | Nx_rune.Ocaml | Nx_rune.C -> Xla.Client.cpu ()
  in

  (* Map from our variables to XLA ops *)
  let var_to_op = Var.Table.create 16 in

  (* Helper to get XLA op for an atom *)
  let get_atom_op atom =
    match atom with
    | Var v -> (
        match Var.Table.find_opt var_to_op v with
        | Some op -> op
        | None ->
            failwith (Printf.sprintf "Unbound variable %d in XLA compiler" v))
    | Const literal ->
        (* Create XLA constant from literal *)
        Xla.Builder.constant builder literal
  in

  (* Create XLA parameters for each input *)
  List.iteri
    (fun i input_var ->
      let var_info = Var.Map.find input_var expr.var_info in
      let shape = Xla.Shape.create var_info.shape in
      let param_op =
        Xla.Builder.parameter builder i shape (Printf.sprintf "param%d" i)
      in
      Var.Table.add var_to_op input_var param_op)
    expr.inputs;

  (* Walk the equations and build the XLA graph *)
  List.iter
    (fun (out_var, op) ->
      let xla_op =
        match op with
        | Binary (bop, atom_a, atom_b) -> (
            let a = get_atom_op atom_a in
            let b = get_atom_op atom_b in
            match bop with
            | Add -> Xla.Builder.add builder a b
            | Mul -> Xla.Builder.multiply builder a b
            | Div -> Xla.Builder.divide builder a b
            | Max -> Xla.Builder.max builder a b
            | Mod -> Xla.Builder.remainder builder a b
            | Pow -> Xla.Builder.pow builder a b
            | And -> Xla.Builder.and_ builder a b
            | Or -> Xla.Builder.or_ builder a b
            | Xor -> Xla.Builder.xor builder a b
            | Idiv ->
                (* Integer division - for now just use regular division TODO:
                   properly handle integer division with truncation *)
                Xla.Builder.divide builder a b)
        | Unary (uop, atom_in) -> (
            let x = get_atom_op atom_in in
            match uop with
            | Neg -> Xla.Builder.neg builder x
            | Sin -> Xla.Builder.sin builder x
            | Sqrt -> Xla.Builder.sqrt builder x
            | Exp2 ->
                (* exp2(x) = exp(x * ln(2)) *)
                let ln2 =
                  Xla.Builder.constant builder
                    (Xla.Literal.create_r0_f32 0.693147)
                in
                Xla.Builder.exp builder (Xla.Builder.multiply builder x ln2)
            | Log2 ->
                (* log2(x) = log(x) / ln(2) *)
                let ln2 =
                  Xla.Builder.constant builder
                    (Xla.Literal.create_r0_f32 0.693147)
                in
                Xla.Builder.divide builder (Xla.Builder.log builder x) ln2
            | Recip ->
                (* recip(x) = 1.0 / x *)
                let one =
                  Xla.Builder.constant builder (Xla.Literal.create_r0_f32 1.0)
                in
                Xla.Builder.divide builder one x)
        | Reduce (rop, atom_in, axes) -> (
            let x = get_atom_op atom_in in
            match rop with
            | Sum ->
                Xla.Builder.reduce_sum builder x ~dims:axes ~keep_dims:false)
        | Transpose (atom_in, perm) ->
            let x = get_atom_op atom_in in
            Xla.Builder.transpose builder x perm
        | Reshape (atom_in, new_shape) ->
            let x = get_atom_op atom_in in
            Xla.Builder.reshape builder x new_shape
        | Expand (atom_in, new_shape) ->
            let x = get_atom_op atom_in in
            Xla.Builder.broadcast builder x new_shape
        | Comparison (op, atom_a, atom_b) -> (
            let a = get_atom_op atom_a in
            let b = get_atom_op atom_b in
            match op with
            | `Lt -> Xla.Builder.lt builder a b
            | `Ne -> Xla.Builder.ne builder a b)
        | Where (cond_atom, true_atom, false_atom) ->
            let cond = get_atom_op cond_atom in
            let t = get_atom_op true_atom in
            let f = get_atom_op false_atom in
            Xla.Builder.select builder cond t f
        | Cast (atom_in, target_dtype) ->
            let x = get_atom_op atom_in in
            let target_type =
              match target_dtype with
              | Pack Float32 -> Xla.Element_type.F32
              | Pack Float64 -> Xla.Element_type.F64
              | Pack Int32 -> Xla.Element_type.S32
              | Pack Int64 -> Xla.Element_type.S64
              | Pack Int8 -> Xla.Element_type.S8
              | Pack UInt8 -> Xla.Element_type.U8
              | Pack Int16 -> Xla.Element_type.S16
              | Pack UInt16 -> Xla.Element_type.U16
              | _ -> failwith "XLA: Unsupported dtype for cast"
            in
            Xla.Builder.convert_element_type builder x target_type
        | Pad (atom_in, padding_config, fill_atom) ->
            let x = get_atom_op atom_in in
            let fill = get_atom_op fill_atom in
            (* Convert padding config to XLA format *)
            let rank = Array.length padding_config in
            let low_padding = Array.make rank 0 in
            let high_padding = Array.make rank 0 in
            let interior_padding = Array.make rank 0 in
            Array.iteri
              (fun i (low, high) ->
                low_padding.(i) <- low;
                high_padding.(i) <- high)
              padding_config;
            Xla.Builder.pad builder x fill ~low_padding ~high_padding
              ~interior_padding
        | Shrink (atom_in, limits) ->
            let x = get_atom_op atom_in in
            let starts = Array.map fst limits in
            let limits_array = Array.map snd limits in
            let strides = Array.make (Array.length limits) 1 in
            Xla.Builder.slice builder x ~start_indices:starts
              ~limit_indices:limits_array ~strides
        | Flip (atom_in, dims_to_flip) ->
            let x = get_atom_op atom_in in
            let rank = Array.length dims_to_flip in
            let axes = ref [] in
            for i = 0 to rank - 1 do
              if dims_to_flip.(i) then axes := i :: !axes
            done;
            let axes_array = Array.of_list (List.rev !axes) in
            Xla.Builder.reverse builder x axes_array
        | Cat (atoms, axis) ->
            let ops = List.map get_atom_op atoms in
            let ops_array = Array.of_list ops in
            Xla.Builder.concatenate builder ops_array axis
        | Matmul (atom_a, atom_b) ->
            let a = get_atom_op atom_a in
            let b = get_atom_op atom_b in
            Xla.Builder.dot builder a b
        | Gather (data_atom, indices_atom, axis) ->
            let data = get_atom_op data_atom in
            let indices = get_atom_op indices_atom in
            Xla.Builder.gather builder data indices ~axis
        | Conv2d (input_atom, kernel_atom, strides, padding, dilation) ->
            let input = get_atom_op input_atom in
            let kernel = get_atom_op kernel_atom in
            Xla.Builder.conv2d builder input kernel ~strides ~padding ~dilation
              ()
        | Scatter (data_atom, indices_atom, updates_atom, axis, mode) ->
            let data = get_atom_op data_atom in
            let indices = get_atom_op indices_atom in
            let updates = get_atom_op updates_atom in
            (* Create appropriate update computation based on mode *)
            let var_info = Var.Map.find out_var expr.var_info in
            let _element_type =
              match var_info.dtype with
              | Pack Float32 -> Xla.Element_type.F32
              | Pack Float64 -> Xla.Element_type.F64
              | Pack Int32 -> Xla.Element_type.S32
              | Pack Int64 -> Xla.Element_type.S64
              | _ -> Xla.Element_type.F32 (* Default fallback *)
            in
            let update_computation =
              match mode with
              | `Add ->
                  (* Create add computation *)
                  let temp_builder = Xla.Builder.create "scatter_add" in
                  let scalar_shape = Xla.Shape.create [||] in
                  let param0 =
                    Xla.Builder.parameter temp_builder 0 scalar_shape "x"
                  in
                  let param1 =
                    Xla.Builder.parameter temp_builder 1 scalar_shape "y"
                  in
                  let sum = Xla.Builder.add temp_builder param0 param1 in
                  Xla.Builder.build temp_builder sum
              | `Set ->
                  (* For set mode, create a computation that returns the update
                     value *)
                  let temp_builder = Xla.Builder.create "scatter_set" in
                  let scalar_shape = Xla.Shape.create [||] in
                  let _param0 =
                    Xla.Builder.parameter temp_builder 0 scalar_shape "old"
                  in
                  let param1 =
                    Xla.Builder.parameter temp_builder 1 scalar_shape "new"
                  in
                  Xla.Builder.build temp_builder param1
            in
            Xla.Builder.scatter builder data indices updates ~axis
              ~update_computation
        | Fold (input_atom, output_size, _kernel_size, _strides, _padding) ->
            (* Fold is the inverse of unfold - it reassembles patches back into an image *)
            (* This is a complex operation that requires careful handling of overlapping regions *)
            (* For now, we implement a simplified version using reshape and transpose *)
            let input = get_atom_op input_atom in
            let var_info = Var.Map.find out_var expr.var_info in

            (* Expected input shape: [batch, channels * kh * kw, h_patches, w_patches] *)
            (* Expected output shape: [batch, channels, h_out, w_out] *)
            let batch = var_info.shape.(0) in
            let channels = var_info.shape.(1) in
            let h_out = output_size.(0) in
            let w_out = output_size.(1) in

            (* For a proper fold implementation, we would need to: *)
            (* 1. Reshape to separate kernel dimensions *)
            (* 2. Transpose to get patches in the right order *)
            (* 3. Use scatter-add to accumulate overlapping patches *)
            (* 4. Handle padding and stride effects *)

            (* Simplified implementation: just reshape to output size *)
            let output_shape = [| batch; channels; h_out; w_out |] in
            Xla.Builder.reshape builder input output_shape
      in
      Var.Table.add var_to_op out_var xla_op)
    expr.equations;

  (* Get output ops and build computation *)
  let output_ops =
    List.map
      (fun v ->
        match Var.Table.find_opt var_to_op v with
        | Some op -> op
        | None -> failwith (Printf.sprintf "Output variable %d not found" v))
      expr.outputs
  in

  let root =
    match output_ops with
    | [ single ] -> single
    | _ -> failwith "XLA: Multiple outputs not yet supported"
  in

  let computation = Xla.Builder.build builder root in
  let executable = Xla.Computation.compile xla_client computation in

  (* Return the execution function *)
  fun (inputs : packed_tensor list) ->
    (* Get the context from the first input to maintain device consistency *)
    let ctx =
      match inputs with
      | [] -> failwith "XLA: No inputs provided"
      | Packed_tensor first :: _ -> Nx_rune.context first
    in

    (* Convert inputs to XLA literals *)
    let input_literals =
      List.map (fun (Packed_tensor t) -> tensor_to_literal t) inputs
    in

    (* Execute *)
    let result_literals = Xla.Computation.execute executable input_literals in

    (* Convert results back to Nx_rune tensors *)
    List.map2
      (fun output_var literal ->
        let var_info = Var.Map.find output_var expr.var_info in
        (* We need to handle each dtype case separately due to existential
           types *)
        match var_info.dtype with
        | Pack Float32 ->
            let arr = Xla.Literal.to_bigarray literal Bigarray.Float32 in
            (* Convert genarray to array1 by reshaping to flat array *)
            let size = Array.fold_left ( * ) 1 (Bigarray.Genarray.dims arr) in
            let arr1 = Bigarray.reshape_1 arr size in
            let tensor = Nx_rune.op_const_array ctx arr1 in
            (* Now reshape back to original shape *)
            let tensor = Nx_rune.op_reshape tensor var_info.shape in
            Packed_tensor tensor
        | Pack Float64 ->
            let arr = Xla.Literal.to_bigarray literal Bigarray.Float64 in
            let size = Array.fold_left ( * ) 1 (Bigarray.Genarray.dims arr) in
            let arr1 = Bigarray.reshape_1 arr size in
            let tensor = Nx_rune.op_const_array ctx arr1 in
            let tensor = Nx_rune.op_reshape tensor var_info.shape in
            Packed_tensor tensor
        | Pack Int32 ->
            let arr = Xla.Literal.to_bigarray literal Bigarray.Int32 in
            let size = Array.fold_left ( * ) 1 (Bigarray.Genarray.dims arr) in
            let arr1 = Bigarray.reshape_1 arr size in
            let tensor = Nx_rune.op_const_array ctx arr1 in
            let tensor = Nx_rune.op_reshape tensor var_info.shape in
            Packed_tensor tensor
        | Pack Int64 ->
            let arr = Xla.Literal.to_bigarray literal Bigarray.Int64 in
            let size = Array.fold_left ( * ) 1 (Bigarray.Genarray.dims arr) in
            let arr1 = Bigarray.reshape_1 arr size in
            let tensor = Nx_rune.op_const_array ctx arr1 in
            let tensor = Nx_rune.op_reshape tensor var_info.shape in
            Packed_tensor tensor
        | Pack Int8 ->
            let arr = Xla.Literal.to_bigarray literal Bigarray.Int8_signed in
            let size = Array.fold_left ( * ) 1 (Bigarray.Genarray.dims arr) in
            let arr1 = Bigarray.reshape_1 arr size in
            let tensor = Nx_rune.op_const_array ctx arr1 in
            let tensor = Nx_rune.op_reshape tensor var_info.shape in
            Packed_tensor tensor
        | Pack UInt8 ->
            let arr = Xla.Literal.to_bigarray literal Bigarray.Int8_unsigned in
            let size = Array.fold_left ( * ) 1 (Bigarray.Genarray.dims arr) in
            let arr1 = Bigarray.reshape_1 arr size in
            let tensor = Nx_rune.op_const_array ctx arr1 in
            let tensor = Nx_rune.op_reshape tensor var_info.shape in
            Packed_tensor tensor
        | Pack _ -> failwith "XLA: Unsupported dtype for output")
      expr.outputs result_literals

(* Get device type from tensor *)
let get_device_type (tensor : ('a, 'b) Nx_rune.t) : Nx_rune.device_type =
  match tensor with
  | Nx_rune.Ocaml_tensor _ -> Nx_rune.Ocaml
  | Nx_rune.C_tensor _ -> Nx_rune.C
  | Nx_rune.Metal_tensor _ -> Nx_rune.Metal
  | Nx_rune.Symbolic_tensor _ -> Nx_rune.Ocaml (* Default for symbolic *)

(* Main JIT entry point *)
let jit (f : ('a, 'b) Nx_rune.t -> ('c, 'd) Nx_rune.t)
    (input : ('a, 'b) Nx_rune.t) : ('c, 'd) Nx_rune.t =
  (* Detect device from input *)
  let device = get_device_type input in

  (* Build expression graph *)
  let expr = build_expr f input in

  (* Compile to XLA *)
  let compiled_fn = compile_expr expr device in

  (* Execute immediately and return the result *)
  match compiled_fn [ Packed_tensor input ] with
  | [ Packed_tensor result ] ->
      (* We need to cast the result to the correct type *)
      Obj.magic result
  | _ -> failwith "XLA JIT: Expected single output"
