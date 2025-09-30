/*****************************************************************************/
/*                                                                           */
/*                                                                           */
/*  OCaml PocketFFT Bindings                                                 */
/*                                                                           */
/*                                                                           */
/*  Licensed under the Apache License, Version 2.0 (the "License");          */
/*  you may not use this file except in compliance with the License.         */
/*  You may obtain a copy of the License at                                  */
/*                                                                           */
/*    http://www.apache.org/licenses/LICENSE-2.0                             */
/*                                                                           */
/*  Unless required by applicable law or agreed to in writing, software      */
/*  distributed under the License is distributed on an "AS IS" BASIS,        */
/*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. */
/*  See the License for the specific language governing permissions and      */
/*  limitations under the License.                                           */
/*                                                                           */
/*****************************************************************************/

#include "pocketfft/pocketfft_hdronly.h"

extern "C" {
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
}

#if defined(__GNUC__) || defined(__clang__)
#define POCKETFFT_RESTRICT __restrict__
#define POCKETFFT_INLINE inline
#define POCKETFFT_HOT __attribute__((hot))
#define POCKETFFT_CONST __attribute__((const))
#define POCKETFFT_PURE __attribute__((pure))
#elif defined(_MSC_VER)
#define POCKETFFT_RESTRICT __restrict
#define POCKETFFT_INLINE __forceinline
#define POCKETFFT_HOT
#define POCKETFFT_CONST
#define POCKETFFT_PURE
#else
#define POCKETFFT_RESTRICT
#define POCKETFFT_INLINE inline
#define POCKETFFT_HOT
#define POCKETFFT_CONST
#define POCKETFFT_PURE
#endif

#if defined(__GNUC__) || defined(__clang__)
#define ASSUME_ALIGNED(ptr, alignment) __builtin_assume_aligned(ptr, alignment)
#elif defined(__cpp_lib_assume_aligned)
#define ASSUME_ALIGNED(ptr, alignment) std::assume_aligned<alignment>(ptr)
#else
#define ASSUME_ALIGNED(ptr, alignment) (ptr)
#endif

#define EXTRACT_SHAPE_STACK(v_shape, shape_var)                                \
    size_t shape_var##_len = Wosize_val(v_shape);                              \
    std::vector<size_t> shape_var##_data(shape_var##_len);                     \
    for (size_t i = 0; i < shape_var##_len; i++) {                             \
        shape_var##_data[i] = Long_val(Field(v_shape, i));                     \
    }                                                                          \
    pocketfft::shape_t shape_var(shape_var##_data.begin(),                     \
                                 shape_var##_data.end());

#define EXTRACT_STRIDE_STACK(v_stride, stride_var)                             \
    size_t stride_var##_len = Wosize_val(v_stride);                            \
    std::vector<ptrdiff_t> stride_var##_data(stride_var##_len);                \
    for (size_t i = 0; i < stride_var##_len; i++) {                            \
        stride_var##_data[i] = Long_val(Field(v_stride, i));                   \
    }                                                                          \
    pocketfft::stride_t stride_var(stride_var##_data.begin(),                  \
                                   stride_var##_data.end());

extern "C" {

// Float32 Complex-to-Complex FFT
POCKETFFT_HOT POCKETFFT_INLINE value
caml_pocketfft_c2c_f32(value v_shape, value v_stride_in, value v_stride_out,
                       value v_axes, value v_forward, value v_fct,
                       value v_data_in, value v_data_out, value v_nthreads) {
    try {
        EXTRACT_SHAPE_STACK(v_shape, shape);
        EXTRACT_STRIDE_STACK(v_stride_in, stride_in);
        EXTRACT_STRIDE_STACK(v_stride_out, stride_out);
        EXTRACT_SHAPE_STACK(v_axes, axes);

        bool forward = Bool_val(v_forward);
        float fct = Double_val(v_fct);
        size_t nthreads = Long_val(v_nthreads);

        auto* POCKETFFT_RESTRICT data_in = static_cast<std::complex<float>*>(
            ASSUME_ALIGNED(Caml_ba_data_val(v_data_in), 32));
        auto* POCKETFFT_RESTRICT data_out = static_cast<std::complex<float>*>(
            ASSUME_ALIGNED(Caml_ba_data_val(v_data_out), 32));

        caml_release_runtime_system();
        pocketfft::c2c(shape, stride_in, stride_out, axes, forward, data_in,
                       data_out, fct, nthreads);
        caml_acquire_runtime_system();

    } catch (const std::exception& e) {
        caml_acquire_runtime_system();
        caml_failwith(e.what());
    }
    return Val_unit;
}

value caml_pocketfft_c2c_f32_bytecode(value* argv, int argn) {
    return caml_pocketfft_c2c_f32(argv[0], argv[1], argv[2], argv[3], argv[4],
                                  argv[5], argv[6], argv[7], argv[8]);
}

// Float32 Real-to-Complex FFT
POCKETFFT_HOT POCKETFFT_INLINE value
caml_pocketfft_r2c_f32(value v_shape_in, value v_stride_in, value v_stride_out,
                       value v_axes, value v_forward, value v_fct,
                       value v_data_in, value v_data_out, value v_nthreads) {
    try {
        EXTRACT_SHAPE_STACK(v_shape_in, shape_in);
        EXTRACT_STRIDE_STACK(v_stride_in, stride_in);
        EXTRACT_STRIDE_STACK(v_stride_out, stride_out);
        EXTRACT_SHAPE_STACK(v_axes, axes);

        bool forward = Bool_val(v_forward);
        float fct = Double_val(v_fct);
        size_t nthreads = Long_val(v_nthreads);

        auto* POCKETFFT_RESTRICT data_in = static_cast<float*>(
            ASSUME_ALIGNED(Caml_ba_data_val(v_data_in), 32));
        auto* POCKETFFT_RESTRICT data_out = static_cast<std::complex<float>*>(
            ASSUME_ALIGNED(Caml_ba_data_val(v_data_out), 32));

        caml_release_runtime_system();
        pocketfft::r2c(shape_in, stride_in, stride_out, axes, forward, data_in,
                       data_out, fct, nthreads);
        caml_acquire_runtime_system();

    } catch (const std::exception& e) {
        caml_acquire_runtime_system();
        caml_failwith(e.what());
    }
    return Val_unit;
}

value caml_pocketfft_r2c_f32_bytecode(value* argv, int argn) {
    return caml_pocketfft_r2c_f32(argv[0], argv[1], argv[2], argv[3], argv[4],
                                  argv[5], argv[6], argv[7], argv[8]);
}

// Float32 Complex-to-Real FFT
POCKETFFT_HOT POCKETFFT_INLINE value
caml_pocketfft_c2r_f32(value v_shape_out, value v_stride_in, value v_stride_out,
                       value v_axes, value v_forward, value v_fct,
                       value v_data_in, value v_data_out, value v_nthreads) {
    try {
        EXTRACT_SHAPE_STACK(v_shape_out, shape_out);
        EXTRACT_STRIDE_STACK(v_stride_in, stride_in);
        EXTRACT_STRIDE_STACK(v_stride_out, stride_out);
        EXTRACT_SHAPE_STACK(v_axes, axes);

        bool forward = Bool_val(v_forward);
        float fct = Double_val(v_fct);
        size_t nthreads = Long_val(v_nthreads);

        auto* POCKETFFT_RESTRICT data_in = static_cast<std::complex<float>*>(
            ASSUME_ALIGNED(Caml_ba_data_val(v_data_in), 32));
        auto* POCKETFFT_RESTRICT data_out = static_cast<float*>(
            ASSUME_ALIGNED(Caml_ba_data_val(v_data_out), 32));

        caml_release_runtime_system();
        pocketfft::c2r(shape_out, stride_in, stride_out, axes, forward, data_in,
                       data_out, fct, nthreads);
        caml_acquire_runtime_system();

    } catch (const std::exception& e) {
        caml_acquire_runtime_system();
        caml_failwith(e.what());
    }
    return Val_unit;
}

value caml_pocketfft_c2r_f32_bytecode(value* argv, int argn) {
    return caml_pocketfft_c2r_f32(argv[0], argv[1], argv[2], argv[3], argv[4],
                                  argv[5], argv[6], argv[7], argv[8]);
}

// Float32 DCT
POCKETFFT_HOT POCKETFFT_INLINE value caml_pocketfft_dct_f32(
    value v_shape, value v_stride_in, value v_stride_out, value v_axes,
    value v_dct_type, value v_ortho, value v_fct, value v_data_in,
    value v_data_out, value v_nthreads) {
    try {
        EXTRACT_SHAPE_STACK(v_shape, shape);
        EXTRACT_STRIDE_STACK(v_stride_in, stride_in);
        EXTRACT_STRIDE_STACK(v_stride_out, stride_out);
        EXTRACT_SHAPE_STACK(v_axes, axes);

        int dct_type = Long_val(v_dct_type);
        bool ortho = Bool_val(v_ortho);
        float fct = Double_val(v_fct);
        size_t nthreads = Long_val(v_nthreads);

        auto* POCKETFFT_RESTRICT data_in = static_cast<float*>(
            ASSUME_ALIGNED(Caml_ba_data_val(v_data_in), 32));
        auto* POCKETFFT_RESTRICT data_out = static_cast<float*>(
            ASSUME_ALIGNED(Caml_ba_data_val(v_data_out), 32));

        caml_release_runtime_system();
        pocketfft::dct(shape, stride_in, stride_out, axes, dct_type, data_in,
                       data_out, fct, ortho, nthreads);
        caml_acquire_runtime_system();

    } catch (const std::exception& e) {
        caml_acquire_runtime_system();
        caml_failwith(e.what());
    }
    return Val_unit;
}

value caml_pocketfft_dct_f32_bytecode(value* argv, int argn) {
    return caml_pocketfft_dct_f32(argv[0], argv[1], argv[2], argv[3], argv[4],
                                  argv[5], argv[6], argv[7], argv[8], argv[9]);
}

// Float32 DST
POCKETFFT_HOT POCKETFFT_INLINE value caml_pocketfft_dst_f32(
    value v_shape, value v_stride_in, value v_stride_out, value v_axes,
    value v_dct_type, value v_ortho, value v_fct, value v_data_in,
    value v_data_out, value v_nthreads) {
    try {
        EXTRACT_SHAPE_STACK(v_shape, shape);
        EXTRACT_STRIDE_STACK(v_stride_in, stride_in);
        EXTRACT_STRIDE_STACK(v_stride_out, stride_out);
        EXTRACT_SHAPE_STACK(v_axes, axes);

        int dct_type = Long_val(v_dct_type);
        bool ortho = Bool_val(v_ortho);
        float fct = Double_val(v_fct);
        size_t nthreads = Long_val(v_nthreads);

        auto* POCKETFFT_RESTRICT data_in = static_cast<float*>(
            ASSUME_ALIGNED(Caml_ba_data_val(v_data_in), 32));
        auto* POCKETFFT_RESTRICT data_out = static_cast<float*>(
            ASSUME_ALIGNED(Caml_ba_data_val(v_data_out), 32));

        caml_release_runtime_system();
        pocketfft::dst(shape, stride_in, stride_out, axes, dct_type, data_in,
                       data_out, fct, ortho, nthreads);
        caml_acquire_runtime_system();

    } catch (const std::exception& e) {
        caml_acquire_runtime_system();
        caml_failwith(e.what());
    }
    return Val_unit;
}

value caml_pocketfft_dst_f32_bytecode(value* argv, int argn) {
    return caml_pocketfft_dst_f32(argv[0], argv[1], argv[2], argv[3], argv[4],
                                  argv[5], argv[6], argv[7], argv[8], argv[9]);
}

// Float64 Complex-to-Complex FFT
POCKETFFT_HOT POCKETFFT_INLINE value
caml_pocketfft_c2c_f64(value v_shape, value v_stride_in, value v_stride_out,
                       value v_axes, value v_forward, value v_fct,
                       value v_data_in, value v_data_out, value v_nthreads) {
    try {
        EXTRACT_SHAPE_STACK(v_shape, shape);
        EXTRACT_STRIDE_STACK(v_stride_in, stride_in);
        EXTRACT_STRIDE_STACK(v_stride_out, stride_out);
        EXTRACT_SHAPE_STACK(v_axes, axes);

        bool forward = Bool_val(v_forward);
        double fct = Double_val(v_fct);
        size_t nthreads = Long_val(v_nthreads);

        auto* POCKETFFT_RESTRICT data_in = static_cast<std::complex<double>*>(
            ASSUME_ALIGNED(Caml_ba_data_val(v_data_in), 32));
        auto* POCKETFFT_RESTRICT data_out = static_cast<std::complex<double>*>(
            ASSUME_ALIGNED(Caml_ba_data_val(v_data_out), 32));

        caml_release_runtime_system();
        pocketfft::c2c(shape, stride_in, stride_out, axes, forward, data_in,
                       data_out, fct, nthreads);
        caml_acquire_runtime_system();

    } catch (const std::exception& e) {
        caml_acquire_runtime_system();
        caml_failwith(e.what());
    }
    return Val_unit;
}

value caml_pocketfft_c2c_f64_bytecode(value* argv, int argn) {
    return caml_pocketfft_c2c_f64(argv[0], argv[1], argv[2], argv[3], argv[4],
                                  argv[5], argv[6], argv[7], argv[8]);
}

// Float64 Real-to-Complex FFT
POCKETFFT_HOT POCKETFFT_INLINE value
caml_pocketfft_r2c_f64(value v_shape_in, value v_stride_in, value v_stride_out,
                       value v_axes, value v_forward, value v_fct,
                       value v_data_in, value v_data_out, value v_nthreads) {
    try {
        EXTRACT_SHAPE_STACK(v_shape_in, shape_in);
        EXTRACT_STRIDE_STACK(v_stride_in, stride_in);
        EXTRACT_STRIDE_STACK(v_stride_out, stride_out);
        EXTRACT_SHAPE_STACK(v_axes, axes);

        bool forward = Bool_val(v_forward);
        double fct = Double_val(v_fct);
        size_t nthreads = Long_val(v_nthreads);

        auto* POCKETFFT_RESTRICT data_in = static_cast<double*>(
            ASSUME_ALIGNED(Caml_ba_data_val(v_data_in), 32));
        auto* POCKETFFT_RESTRICT data_out = static_cast<std::complex<double>*>(
            ASSUME_ALIGNED(Caml_ba_data_val(v_data_out), 32));

        caml_release_runtime_system();
        pocketfft::r2c(shape_in, stride_in, stride_out, axes, forward, data_in,
                       data_out, fct, nthreads);
        caml_acquire_runtime_system();

    } catch (const std::exception& e) {
        caml_acquire_runtime_system();
        caml_failwith(e.what());
    }
    return Val_unit;
}

value caml_pocketfft_r2c_f64_bytecode(value* argv, int argn) {
    return caml_pocketfft_r2c_f64(argv[0], argv[1], argv[2], argv[3], argv[4],
                                  argv[5], argv[6], argv[7], argv[8]);
}

// Float64 Complex-to-Real FFT
POCKETFFT_HOT POCKETFFT_INLINE value
caml_pocketfft_c2r_f64(value v_shape_out, value v_stride_in, value v_stride_out,
                       value v_axes, value v_forward, value v_fct,
                       value v_data_in, value v_data_out, value v_nthreads) {
    try {
        EXTRACT_SHAPE_STACK(v_shape_out, shape_out);
        EXTRACT_STRIDE_STACK(v_stride_in, stride_in);
        EXTRACT_STRIDE_STACK(v_stride_out, stride_out);
        EXTRACT_SHAPE_STACK(v_axes, axes);

        bool forward = Bool_val(v_forward);
        double fct = Double_val(v_fct);
        size_t nthreads = Long_val(v_nthreads);

        auto* POCKETFFT_RESTRICT data_in = static_cast<std::complex<double>*>(
            ASSUME_ALIGNED(Caml_ba_data_val(v_data_in), 32));
        auto* POCKETFFT_RESTRICT data_out = static_cast<double*>(
            ASSUME_ALIGNED(Caml_ba_data_val(v_data_out), 32));

        caml_release_runtime_system();
        pocketfft::c2r(shape_out, stride_in, stride_out, axes, forward, data_in,
                       data_out, fct, nthreads);
        caml_acquire_runtime_system();

    } catch (const std::exception& e) {
        caml_acquire_runtime_system();
        caml_failwith(e.what());
    }
    return Val_unit;
}

value caml_pocketfft_c2r_f64_bytecode(value* argv, int argn) {
    return caml_pocketfft_c2r_f64(argv[0], argv[1], argv[2], argv[3], argv[4],
                                  argv[5], argv[6], argv[7], argv[8]);
}

// Float64 DCT
POCKETFFT_HOT POCKETFFT_INLINE value caml_pocketfft_dct_f64(
    value v_shape, value v_stride_in, value v_stride_out, value v_axes,
    value v_dct_type, value v_ortho, value v_fct, value v_data_in,
    value v_data_out, value v_nthreads) {
    try {
        EXTRACT_SHAPE_STACK(v_shape, shape);
        EXTRACT_STRIDE_STACK(v_stride_in, stride_in);
        EXTRACT_STRIDE_STACK(v_stride_out, stride_out);
        EXTRACT_SHAPE_STACK(v_axes, axes);

        int dct_type = Long_val(v_dct_type);
        bool ortho = Bool_val(v_ortho);
        double fct = Double_val(v_fct);
        size_t nthreads = Long_val(v_nthreads);

        auto* POCKETFFT_RESTRICT data_in = static_cast<double*>(
            ASSUME_ALIGNED(Caml_ba_data_val(v_data_in), 32));
        auto* POCKETFFT_RESTRICT data_out = static_cast<double*>(
            ASSUME_ALIGNED(Caml_ba_data_val(v_data_out), 32));

        caml_release_runtime_system();
        pocketfft::dct(shape, stride_in, stride_out, axes, dct_type, data_in,
                       data_out, fct, ortho, nthreads);
        caml_acquire_runtime_system();

    } catch (const std::exception& e) {
        caml_acquire_runtime_system();
        caml_failwith(e.what());
    }
    return Val_unit;
}

value caml_pocketfft_dct_f64_bytecode(value* argv, int argn) {
    return caml_pocketfft_dct_f64(argv[0], argv[1], argv[2], argv[3], argv[4],
                                  argv[5], argv[6], argv[7], argv[8], argv[9]);
}

// Float64 DST
POCKETFFT_HOT POCKETFFT_INLINE value caml_pocketfft_dst_f64(
    value v_shape, value v_stride_in, value v_stride_out, value v_axes,
    value v_dct_type, value v_ortho, value v_fct, value v_data_in,
    value v_data_out, value v_nthreads) {
    try {
        EXTRACT_SHAPE_STACK(v_shape, shape);
        EXTRACT_STRIDE_STACK(v_stride_in, stride_in);
        EXTRACT_STRIDE_STACK(v_stride_out, stride_out);
        EXTRACT_SHAPE_STACK(v_axes, axes);

        int dct_type = Long_val(v_dct_type);
        bool ortho = Bool_val(v_ortho);
        double fct = Double_val(v_fct);
        size_t nthreads = Long_val(v_nthreads);

        auto* POCKETFFT_RESTRICT data_in = static_cast<double*>(
            ASSUME_ALIGNED(Caml_ba_data_val(v_data_in), 32));
        auto* POCKETFFT_RESTRICT data_out = static_cast<double*>(
            ASSUME_ALIGNED(Caml_ba_data_val(v_data_out), 32));

        caml_release_runtime_system();
        pocketfft::dst(shape, stride_in, stride_out, axes, dct_type, data_in,
                       data_out, fct, ortho, nthreads);
        caml_acquire_runtime_system();

    } catch (const std::exception& e) {
        caml_acquire_runtime_system();
        caml_failwith(e.what());
    }
    return Val_unit;
}

value caml_pocketfft_dst_f64_bytecode(value* argv, int argn) {
    return caml_pocketfft_dst_f64(argv[0], argv[1], argv[2], argv[3], argv[4],
                                  argv[5], argv[6], argv[7], argv[8], argv[9]);
}
}
