(* Internal invariants that the backend-neutral contract cannot express. *)

open Bigarray
open Windtrap
module Buf = Nx_buffer

type ('a, 'b) ffi = {
  buffer : ('a, 'b, c_layout) Genarray.t;
  shape : int array;
  strides : int array;
  offset : int;
}

external sort :
  (float, float64_elt) ffi -> (float, float64_elt) ffi -> int -> bool -> unit
  = "caml_nx_c_sort"

external argsort :
  (int32, int32_elt) ffi -> (float, float64_elt) ffi -> int -> bool -> unit
  = "caml_nx_c_argsort"

external cast_convert_selfcheck : unit -> int
  = "caml_nx_c_cast_convert_selfcheck_test"

let row_major shape =
  let n = Array.length shape in
  let strides = Array.make n 1 in
  for i = n - 2 downto 0 do
    strides.(i) <- strides.(i + 1) * shape.(i + 1)
  done;
  strides

let ffi ?(offset = 0) ?strides ?shape buffer =
  let len = Buf.length buffer in
  let shape = Option.value shape ~default:[| len |] in
  let strides = Option.value strides ~default:(row_major shape) in
  {
    buffer = Buf.to_genarray buffer [| Int.max 1 len |];
    shape;
    strides;
    offset;
  }

let buffer kind values =
  let result = Buf.create kind (Array.length values) in
  Array.iteri (Buf.set result) values;
  result

let test_cast_converters () =
  equal ~msg:"f16/bf16 converters match Nx_buffer over the exhaustive corpus"
    int 0
    (cast_convert_selfcheck ())

let test_sort_rejects_aliased_output () =
  let input = buffer Buf.float64 [| 3.; 1.; 2.; 4. |] in
  let output = Buf.create Buf.float64 4 in
  raises ~msg:"zero-stride output"
    (Invalid_argument "sort: output has a broadcast (zero) stride") (fun () ->
      sort
        (ffi ~shape:[| 2; 2 |] ~strides:[| 0; 1 |] output)
        (ffi ~shape:[| 2; 2 |] ~strides:[| 2; 1 |] input)
        1 false)

let test_sort_worker_scratch () =
  (* Many independent rows force the worker-indexed scratch path. Value sort and
     argsort use different scratch layouts, so both are checked. *)
  let rows = 4096 and cols = 128 in
  let shape = [| rows; cols |] in
  let strides = row_major shape in
  Random.init 777;
  let values = Array.init (rows * cols) (fun _ -> Random.float 1.) in
  let input = buffer Buf.float64 values in
  let sorted = Buf.create Buf.float64 (rows * cols) in
  sort (ffi ~shape ~strides sorted) (ffi ~shape ~strides input) 1 false;
  for row = 0 to rows - 1 do
    let base = row * cols in
    for col = 1 to cols - 1 do
      is_true ~msg:"each worker-owned row is sorted"
        (Buf.get sorted (base + col - 1) <= Buf.get sorted (base + col))
    done
  done;
  let indices = Buf.create Buf.int32 (rows * cols) in
  argsort (ffi ~shape ~strides indices) (ffi ~shape ~strides input) 1 false;
  let seen = Array.make cols false in
  for row = 0 to rows - 1 do
    Array.fill seen 0 cols false;
    let base = row * cols in
    let previous = ref Float.neg_infinity in
    for col = 0 to cols - 1 do
      let index = Int32.to_int (Buf.get indices (base + col)) in
      is_true ~msg:"argsort index in range" (index >= 0 && index < cols);
      is_false ~msg:"argsort row is a permutation" seen.(index);
      seen.(index) <- true;
      let value = values.(base + index) in
      is_true ~msg:"argsort gather is sorted" (!previous <= value);
      previous := value
    done
  done

let tests =
  group "kernel-invariants"
    [
      test "cast converter equivalence" test_cast_converters;
      test "sort rejects aliased output" test_sort_rejects_aliased_output;
      test "sort worker scratch is isolated" test_sort_worker_scratch;
    ]

let () = Windtrap.run "nx C backend kernels" [ tests ]
