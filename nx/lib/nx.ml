module F = Nx_core.Make_frontend (Nx_c)
include F

(* Re-export extended type aliases *)
type bfloat16_t = (float, Bigarray_ext.bfloat16_elt) t
type bool_t = (bool, Bigarray_ext.bool_elt) t
type int4_t = (int, Bigarray_ext.int4_signed_elt) t
type uint4_t = (int, Bigarray_ext.int4_unsigned_elt) t
type float8_e4m3_t = (float, Bigarray_ext.float8_e4m3_elt) t
type float8_e5m2_t = (float, Bigarray_ext.float8_e5m2_elt) t
type complex16_t = (Complex.t, Bigarray_ext.complex16_elt) t
type qint8_t = (int, Bigarray_ext.qint8_elt) t
type quint8_t = (int, Bigarray_ext.quint8_elt) t

(* Re-export extended dtype value constructors *)
let bfloat16 = Nx_core.Dtype.bfloat16
let bool = Nx_core.Dtype.bool
let int4 = Nx_core.Dtype.int4
let uint4 = Nx_core.Dtype.uint4
let float8_e4m3 = Nx_core.Dtype.float8_e4m3
let float8_e5m2 = Nx_core.Dtype.float8_e5m2
let complex16 = Nx_core.Dtype.complex16
let qint8 = Nx_core.Dtype.qint8
let quint8 = Nx_core.Dtype.quint8

(* ───── Overriding functions with default context ───── *)

let context = Lazy.from_fun Nx_c.create_context
let create dtype shape arr = F.create (Lazy.force context) dtype shape arr
let init dtype shape f = F.init (Lazy.force context) dtype shape f
let empty dtype shape = F.empty (Lazy.force context) dtype shape
let full dtype shape value = F.full (Lazy.force context) dtype shape value
let ones dtype shape = F.ones (Lazy.force context) dtype shape
let zeros dtype shape = F.zeros (Lazy.force context) dtype shape
let scalar dtype v = F.scalar (Lazy.force context) dtype v
let eye ?m ?k dtype n = F.eye (Lazy.force context) ?m ?k dtype n
let identity dtype n = F.identity (Lazy.force context) dtype n

let arange dtype start stop step =
  F.arange (Lazy.force context) dtype start stop step

let arange_f dtype start stop step =
  F.arange_f (Lazy.force context) dtype start stop step

let linspace dtype ?endpoint start stop num =
  F.linspace (Lazy.force context) dtype ?endpoint start stop num

let logspace dtype ?endpoint ?base start stop num =
  F.logspace (Lazy.force context) dtype ?endpoint ?base start stop num

let geomspace dtype ?endpoint start stop num =
  F.geomspace (Lazy.force context) dtype ?endpoint start stop num

let of_bigarray ba = F.of_bigarray (Lazy.force context) ba
let rand dtype ?seed shape = F.rand (Lazy.force context) dtype ?seed shape
let randn dtype ?seed shape = F.randn (Lazy.force context) dtype ?seed shape

let randint dtype ?seed ?high shape low =
  F.randint (Lazy.force context) dtype ?seed ?high shape low

(* ───── FFT ───── *)

let fftfreq ?d n = F.fftfreq (Lazy.force context) ?d n
let rfftfreq ?d n = F.rfftfreq (Lazy.force context) ?d n
