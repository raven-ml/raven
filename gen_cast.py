import collections

# --- Configuration ---

DTYPES_INFO = collections.OrderedDict([
    ("f16", {"ocaml_type": "float", "ba_elt": "float16_elt", "dtype_constr": "Float16"}),
    ("f32", {"ocaml_type": "float", "ba_elt": "float32_elt", "dtype_constr": "Float32"}),
    ("f64", {"ocaml_type": "float", "ba_elt": "float64_elt", "dtype_constr": "Float64"}),
    ("i8",  {"ocaml_type": "int", "ba_elt": "int8_signed_elt", "dtype_constr": "Int8"}),
    ("u8",  {"ocaml_type": "int", "ba_elt": "int8_unsigned_elt", "dtype_constr": "UInt8"}),
    ("i16", {"ocaml_type": "int", "ba_elt": "int16_signed_elt", "dtype_constr": "Int16"}),
    ("u16", {"ocaml_type": "int", "ba_elt": "int16_unsigned_elt", "dtype_constr": "UInt16"}),
    ("i32", {"ocaml_type": "int32", "ba_elt": "int32_elt", "dtype_constr": "Int32"}),
    ("i64", {"ocaml_type": "int64", "ba_elt": "int64_elt", "dtype_constr": "Int64"}),
    ("c32", {"ocaml_type": "Complex.t", "ba_elt": "complex32_elt", "dtype_constr": "Complex32"}),
    ("c64", {"ocaml_type": "Complex.t", "ba_elt": "complex64_elt", "dtype_constr": "Complex64"}),
    ("int", {"ocaml_type": "int", "ba_elt": "nativeint_elt", "dtype_constr": "Int"}), # As per user sketch
    ("nativeint", {"ocaml_type": "nativeint", "ba_elt": "nativeint_elt", "dtype_constr": "NativeInt"}), # As per user sketch
])

CONVERSIONS = {
    ("float", "float"): lambda v: v,
    ("float", "int"): lambda v: f"int_of_float {v}",
    ("float", "int32"): lambda v: f"Int32.of_float {v}",
    ("float", "int64"): lambda v: f"Int64.of_float {v}",
    ("float", "Complex.t"): lambda v: f"{{ Complex.re = {v}; im = 0.0 }}",
    ("float", "nativeint"): lambda v: f"Nativeint.of_float {v}",

    ("int", "float"): lambda v: f"float_of_int {v}",
    ("int", "int"): lambda v: v,
    ("int", "int32"): lambda v: f"Int32.of_int {v}",
    ("int", "int64"): lambda v: f"Int64.of_int {v}",
    ("int", "Complex.t"): lambda v: f"{{ Complex.re = float_of_int {v}; im = 0.0 }}",
    ("int", "nativeint"): lambda v: f"Nativeint.of_int {v}",

    ("int32", "float"): lambda v: f"Int32.to_float {v}",
    ("int32", "int"): lambda v: f"Int32.to_int {v}",
    ("int32", "int32"): lambda v: v,
    ("int32", "int64"): lambda v: f"Int64.of_int32 {v}",
    ("int32", "Complex.t"): lambda v: f"{{ Complex.re = Int32.to_float {v}; im = 0.0 }}",
    ("int32", "nativeint"): lambda v: f"Nativeint.of_int32 {v}",

    ("int64", "float"): lambda v: f"Int64.to_float {v}",
    ("int64", "int"): lambda v: f"Int64.to_int {v}",
    ("int64", "int32"): lambda v: f"Int64.to_int32 {v}",
    ("int64", "int64"): lambda v: v,
    ("int64", "Complex.t"): lambda v: f"{{ Complex.re = Int64.to_float {v}; im = 0.0 }}",
    ("int64", "nativeint"): lambda v: f"Nativeint.of_int64 {v}",

    ("Complex.t", "float"): lambda v: f"{v}.Complex.re",
    ("Complex.t", "int"): lambda v: f"int_of_float {v}.Complex.re",
    ("Complex.t", "int32"): lambda v: f"Int32.of_float {v}.Complex.re",
    ("Complex.t", "int64"): lambda v: f"Int64.of_float {v}.Complex.re",
    ("Complex.t", "Complex.t"): lambda v: v,
    ("Complex.t", "nativeint"): lambda v: f"Nativeint.of_float {v}.Complex.re",

    ("nativeint", "float"): lambda v: f"Nativeint.to_float {v}",
    ("nativeint", "int"): lambda v: f"Nativeint.to_int {v}",
    ("nativeint", "int32"): lambda v: f"Nativeint.to_int32 {v}",
    ("nativeint", "int64"): lambda v: f"Nativeint.to_int64 {v}",
    ("nativeint", "Complex.t"): lambda v: f"{{ Complex.re = Nativeint.to_float {v}; im = 0.0 }}",
    ("nativeint", "nativeint"): lambda v: v,
}

# --- Code Generation for Specific Cast Functions ---

def generate_ocaml_cast_function(src_key, dst_key, src_info, dst_info, conversion_func_str_lambda):
    header = f"""let cast_{src_key}_to_{dst_key} (src : ({src_info['ocaml_type']}, {src_info['ba_elt']}) t)
    (dst : ({dst_info['ocaml_type']}, {dst_info['ba_elt']}) t) start_idx end_idx =
  let src_buf, dst_buf = (buffer src, buffer dst) in"""

    conv_str0 = conversion_func_str_lambda("src_val0")
    conv_str1 = conversion_func_str_lambda("src_val1")
    conv_str2 = conversion_func_str_lambda("src_val2")
    conv_str3 = conversion_func_str_lambda("src_val3")
    conv_single_str = conversion_func_str_lambda("src_val")

    contiguous_branch = f"""  if is_contiguous src then (
    let i = ref start_idx in
    while !i + 3 < end_idx do
      let i0 = !i + 0 and i1 = !i + 1 and i2 = !i + 2 and i3 = !i + 3 in
      let src_val0 = Array1.unsafe_get src_buf (offset src + i0) in
      let src_val1 = Array1.unsafe_get src_buf (offset src + i1) in
      let src_val2 = Array1.unsafe_get src_buf (offset src + i2) in
      let src_val3 = Array1.unsafe_get src_buf (offset src + i3) in
      Array1.unsafe_set dst_buf i0 {conv_str0};
      Array1.unsafe_set dst_buf i1 {conv_str1};
      Array1.unsafe_set dst_buf i2 {conv_str2};
      Array1.unsafe_set dst_buf i3 {conv_str3};
      i := !i + 4
    done;
    while !i < end_idx do
      let idx = !i in
      let src_val = Array1.unsafe_get src_buf (offset src + idx) in
      Array1.unsafe_set dst_buf idx {conv_single_str};
      incr i
    done)"""

    non_contiguous_branch = f"""  else
    for k = start_idx to end_idx - 1 do
      let md_index = offset_to_index_contig k (shape dst) in
      let src_lin = index_to_offset md_index (strides src) in
      let src_val = Array1.unsafe_get src_buf (offset src + src_lin) in
      Array1.unsafe_set dst_buf k {conv_single_str}
    done"""
    return f"{header}\n{contiguous_branch}\n{non_contiguous_branch}\n"

# --- Code Generation for cast_kernel ---

def generate_cast_kernel_function(dtypes_info_ordered_dict):
    lines = []
    lines.append("""let cast_kernel (type a b c d) (src_dtype : (a, b) Dtype.t)
    (dst_dtype : (c, d) Dtype.t)
    : ((a,b) t -> (c,d) t -> int -> int -> unit) =
  match (src_dtype, dst_dtype) with""")

    type_keys_list = list(dtypes_info_ordered_dict.keys())

    for src_key in type_keys_list:
        src_constr = dtypes_info_ordered_dict[src_key]["dtype_constr"]
        lines.append(f"  (* {src_constr} Source *)")
        for dst_key in type_keys_list:
            dst_constr = dtypes_info_ordered_dict[dst_key]["dtype_constr"]
            
            if src_key == dst_key: # Identity DType case
                lines.append(f"  | {src_constr}, {dst_constr} ->")
                lines.append(f"      (* Identity DType: {src_constr} to {dst_constr}. *)")
                lines.append(f"      (* Specific kernel cast_{src_key}_to_{dst_key} was not generated. *)")
                lines.append(f"      (* This should be handled by a copy operation or a generic identity kernel. *)")
                lines.append(f"      fun _ _ _ _ -> failwith (\"Internal: Identity cast kernel for \" ^ Dtype.to_string src_dtype ^ \" should be pre-empted or use a generic copy.\")")
            else: # Non-identity DType case, the specific function cast_{src_key}_to_{dst_key} was generated
                function_to_call = f"cast_{src_key}_to_{dst_key}"
                lines.append(f"  | {src_constr}, {dst_constr} -> Obj.magic {function_to_call}")
                # To match user's sketch formatting for non-identity, which wrapped it in a lambda:
                # lines.append(f"  | {src_constr}, {dst_constr} ->")
                # lines.append(f"      fun s d start_idx end_idx -> Obj.magic ({function_to_call} s d start_idx end_idx)")
                # However, Obj.magic function_name is more direct if function_name already has the right signature.
                # The specific cast functions have type: (src_t) -> (dst_t) -> int -> int -> unit
                # The cast_kernel expects to return: (gen_src_t) -> (gen_dst_t) -> int -> int -> unit
                # So Obj.magic on the function name itself is correct.
    
    lines.append("""  | _s, _d ->
      failwith
        ("cast_kernel: BUG or Incomplete - unsupported dtype combination from " ^ Dtype.to_string src_dtype
       ^ " to " ^ Dtype.to_string dst_dtype)""")
    lines.append("\n")
    return "\n".join(lines)

# --- Main Script Execution ---

if __name__ == "__main__":
    # --- Generate Specific Cast Functions ---
    generated_specific_functions = []
    type_keys_list = list(DTYPES_INFO.keys())

    for src_t_key in type_keys_list:
        for dst_t_key in type_keys_list:
            if src_t_key == dst_t_key: # Skip identity functions
                # print(f"(* Specific cast function: Skipping identity cast: cast_{src_t_key}_to_{dst_t_key} *)")
                continue

            current_src_info = DTYPES_INFO[src_t_key]
            current_dst_info = DTYPES_INFO[dst_t_key]
            
            ocaml_type_conversion_key = (current_src_info["ocaml_type"], current_dst_info["ocaml_type"])
            
            if ocaml_type_conversion_key not in CONVERSIONS:
                raise ValueError(
                    f"Fatal: Missing OCaml type conversion logic for {ocaml_type_conversion_key}. "
                    f"Needed for cast_{src_t_key}_to_{dst_t_key} (OCaml src: {current_src_info['ocaml_type']}, OCaml dst: {current_dst_info['ocaml_type']})."
                )

            current_conversion_lambda = CONVERSIONS[ocaml_type_conversion_key]
            
            ocaml_code = generate_ocaml_cast_function(
                src_t_key, dst_t_key,
                current_src_info, current_dst_info,
                current_conversion_lambda
            )
            generated_specific_functions.append(ocaml_code)
    
    # --- Output ---
    print("(* BEGIN GENERATED OCAML CODE *)")
    print("(* Assumed open statements: *)")
    print("(* open Bigarray *)")
    print("(* module Dtype = Nx_core.Dtype *)")
    print("(* open Nx_core.View *)")
    print("(* open Internal *)")
    print("(* (* Complex module may also be needed *) *)\n")

    print("(* Specific (non-identity) Casting Functions *)")
    print("\n\n".join(generated_specific_functions))
    print(f"\n(* Total specific (non-identity) cast functions generated: {len(generated_specific_functions)} *)\n")

    print("\n(* Generated cast_kernel Dispatch Function *)")
    cast_kernel_code = generate_cast_kernel_function(DTYPES_INFO)
    print(cast_kernel_code)
    
    print("(* END GENERATED OCAML CODE *)")