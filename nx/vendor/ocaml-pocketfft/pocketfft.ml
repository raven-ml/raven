(** PocketFFT bindings *)

external c2c_f32 :
     shape:int array
  -> stride_in:int array
  -> stride_out:int array
  -> axes:int array
  -> forward:bool
  -> fct:float
  -> data_in:(Complex.t, Bigarray.complex32_elt, Bigarray.c_layout) Bigarray.Array1.t
  -> data_out:(Complex.t, Bigarray.complex32_elt, Bigarray.c_layout) Bigarray.Array1.t
  -> nthreads:int
  -> unit = "caml_pocketfft_c2c_f32_bytecode" "caml_pocketfft_c2c_f32"
[@@noalloc]

external r2c_f32 :
     shape_in:int array
  -> stride_in:int array
  -> stride_out:int array
  -> axes:int array
  -> forward:bool
  -> fct:float
  -> data_in:(float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t
  -> data_out:(Complex.t, Bigarray.complex32_elt, Bigarray.c_layout) Bigarray.Array1.t
  -> nthreads:int
  -> unit = "caml_pocketfft_r2c_f32_bytecode" "caml_pocketfft_r2c_f32"
[@@noalloc]

external c2r_f32 :
     shape_out:int array
  -> stride_in:int array
  -> stride_out:int array
  -> axes:int array
  -> forward:bool
  -> fct:float
  -> data_in:(Complex.t, Bigarray.complex32_elt, Bigarray.c_layout) Bigarray.Array1.t
  -> data_out:(float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t
  -> nthreads:int
  -> unit = "caml_pocketfft_c2r_f32_bytecode" "caml_pocketfft_c2r_f32"
[@@noalloc]

external dct_f32 :
     shape:int array
  -> stride_in:int array
  -> stride_out:int array
  -> axes:int array
  -> dct_type:int
  -> ortho:bool
  -> fct:float
  -> data_in:(float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t
  -> data_out:(float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t
  -> nthreads:int
  -> unit = "caml_pocketfft_dct_f32_bytecode" "caml_pocketfft_dct_f32"
[@@noalloc]

external dst_f32 :
     shape:int array
  -> stride_in:int array
  -> stride_out:int array
  -> axes:int array
  -> dct_type:int
  -> ortho:bool
  -> fct:float
  -> data_in:(float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t
  -> data_out:(float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t
  -> nthreads:int
  -> unit = "caml_pocketfft_dst_f32_bytecode" "caml_pocketfft_dst_f32"
[@@noalloc]

external c2c_f64 :
     shape:int array
  -> stride_in:int array
  -> stride_out:int array
  -> axes:int array
  -> forward:bool
  -> fct:float
  -> data_in:(Complex.t, Bigarray.complex64_elt, Bigarray.c_layout) Bigarray.Array1.t
  -> data_out:(Complex.t, Bigarray.complex64_elt, Bigarray.c_layout) Bigarray.Array1.t
  -> nthreads:int
  -> unit = "caml_pocketfft_c2c_f64_bytecode" "caml_pocketfft_c2c_f64"
[@@noalloc]

external r2c_f64 :
     shape_in:int array
  -> stride_in:int array
  -> stride_out:int array
  -> axes:int array
  -> forward:bool
  -> fct:float
  -> data_in:(float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t
  -> data_out:(Complex.t, Bigarray.complex64_elt, Bigarray.c_layout) Bigarray.Array1.t
  -> nthreads:int
  -> unit = "caml_pocketfft_r2c_f64_bytecode" "caml_pocketfft_r2c_f64"
[@@noalloc]

external c2r_f64 :
     shape_out:int array
  -> stride_in:int array
  -> stride_out:int array
  -> axes:int array
  -> forward:bool
  -> fct:float
  -> data_in:(Complex.t, Bigarray.complex64_elt, Bigarray.c_layout) Bigarray.Array1.t
  -> data_out:(float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t
  -> nthreads:int
  -> unit = "caml_pocketfft_c2r_f64_bytecode" "caml_pocketfft_c2r_f64"
[@@noalloc]

external dct_f64 :
     shape:int array
  -> stride_in:int array
  -> stride_out:int array
  -> axes:int array
  -> dct_type:int
  -> ortho:bool
  -> fct:float
  -> data_in:(float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t
  -> data_out:(float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t
  -> nthreads:int
  -> unit = "caml_pocketfft_dct_f64_bytecode" "caml_pocketfft_dct_f64"
[@@noalloc]

external dst_f64 :
     shape:int array
  -> stride_in:int array
  -> stride_out:int array
  -> axes:int array
  -> dct_type:int
  -> ortho:bool
  -> fct:float
  -> data_in:(float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t
  -> data_out:(float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t
  -> nthreads:int
  -> unit = "caml_pocketfft_dst_f64_bytecode" "caml_pocketfft_dst_f64"
[@@noalloc]
