(** Safetensors - Simple, safe way to store and distribute tensors

    This is an OCaml port of the safetensors format, providing efficient
    serialization and deserialization of tensor data with metadata support. *)

(** {1 Error Handling} *)

type safetensor_error =
  | Invalid_header of string  (** Invalid UTF-8 in header *)
  | Invalid_header_start  (** Invalid start character in header *)
  | Invalid_header_deserialization of string  (** JSON parse error *)
  | Header_too_large  (** Header exceeds maximum size *)
  | Header_too_small  (** Header is too small *)
  | Invalid_header_length  (** Invalid header length *)
  | Tensor_not_found of string  (** Tensor with given name not found *)
  | Tensor_invalid_info  (** Invalid shape, dtype, or offset for tensor *)
  | Invalid_offset of string  (** Invalid offset for tensor *)
  | Io_error of string  (** I/O error during file operations *)
  | Json_error of string  (** JSON processing error *)
  | Invalid_tensor_view of string * int list * int
      (** Invalid tensor view creation *)
  | Metadata_incomplete_buffer
      (** Incomplete metadata, file not fully covered *)
  | Validation_overflow  (** Overflow computing buffer size *)
  | Misaligned_slice
      (** Slice not aligned to byte boundary for sub-byte dtypes *)

val string_of_error : safetensor_error -> string
(** Convert an error to a human-readable string *)

(** {1 Data Types} *)

type dtype =
  | BOOL
  | F4  (** 4-bit float *)
  | F6_E2M3  (** 6-bit float with 2 exponent bits, 3 mantissa bits *)
  | F6_E3M2  (** 6-bit float with 3 exponent bits, 2 mantissa bits *)
  | U8  (** Unsigned 8-bit integer *)
  | I8  (** Signed 8-bit integer *)
  | F8_E5M2  (** 8-bit float with 5 exponent bits, 2 mantissa bits *)
  | F8_E4M3  (** 8-bit float with 4 exponent bits, 3 mantissa bits *)
  | F8_E8M0  (** 8-bit float with 8 exponent bits, no mantissa *)
  | I16  (** Signed 16-bit integer *)
  | U16  (** Unsigned 16-bit integer *)
  | F16  (** Half-precision float *)
  | BF16  (** Brain float 16 *)
  | I32  (** Signed 32-bit integer *)
  | U32  (** Unsigned 32-bit integer *)
  | F32  (** Single-precision float *)
  | F64  (** Double-precision float *)
  | I64  (** Signed 64-bit integer *)
  | U64  (** Unsigned 64-bit integer *)

val dtype_to_string : dtype -> string
(** Convert a dtype to its string representation *)

val dtype_of_string : string -> dtype option
(** Parse a dtype from its string representation *)

val bitsize : dtype -> int
(** Get the size in bits of one element of this dtype *)

(** {1 Tensor Views} *)

type tensor_view = {
  dtype : dtype;
  shape : int list;
  data : string;  (** Backing buffer *)
  offset : int;  (** Byte offset into data *)
  length : int;  (** Number of bytes *)
}
(** A view into tensor data without copying *)

val tensor_view_new :
  dtype:dtype ->
  shape:int list ->
  data:string ->
  (tensor_view, safetensor_error) result
(** Create a new tensor view. The data must exactly match the expected size
    based on dtype and shape. *)

(** {1 SafeTensors Container} *)

type tensor_info = { dtype : dtype; shape : int list; data_offsets : int * int }

type metadata = {
  metadata_kv : (string * string) list option;
  tensors : tensor_info array;
  index_map : (string, int) Hashtbl.t;
}
(** Metadata structure containing tensor information and optional key-value
    pairs *)

type safetensors = { metadata : metadata; data : string }
(** The main container holding tensor metadata and data *)

val deserialize : string -> (safetensors, safetensor_error) result
(** Deserialize a safetensors buffer into a container. The buffer should contain
    the complete safetensors file content. *)

val serialize :
  (string * tensor_view) list ->
  (string * string) list option ->
  (string, safetensor_error) result
(** Serialize a list of named tensors with optional metadata key-value pairs
    into a safetensors buffer. *)

val serialize_to_file :
  (string * tensor_view) list ->
  (string * string) list option ->
  string ->
  (unit, safetensor_error) result
(** Serialize tensors directly to a file. This is more memory-efficient than
    [serialize] for large tensors. *)

(** {1 Tensor Access} *)

val tensor : safetensors -> string -> (tensor_view, safetensor_error) result
(** Get a specific tensor by name from the container *)

val tensors : safetensors -> (string * tensor_view) list
(** Get all tensors from the container as a list of name-view pairs *)

val iter : safetensors -> (string * tensor_view) list
(** Iterate over tensors in offset order (the order they appear in the file) *)

val names : safetensors -> string list
(** Get the names of all tensors in the container *)

val len : safetensors -> int
(** Get the number of tensors in the container *)

val is_empty : safetensors -> bool
(** Check if the container has no tensors *)

(** {1 Slicing} *)

type bound = Unbounded | Excluded of int | Included of int
type tensor_indexer = Select of int | Narrow of bound * bound

type invalid_slice =
  | Too_many_slices
  | Slice_out_of_range of { dim_index : int; asked : int; dim_size : int }
  | Misaligned_slices

type slice_iterator = {
  view : tensor_view;
  mutable indices : (int * int) list;
  newshape : int list;
}

(** Slicing support for lazy iteration over tensor data *)
module Slice : sig
  type t = slice_iterator
  (** Slice iterator *)

  type error = invalid_slice

  type index = tensor_indexer
  (** Indexing specification for a dimension *)

  val select : int -> index
  (** Select a single index from a dimension *)

  val ( // ) : bound -> bound -> index
  (** Create a range slice with start and stop bounds *)

  val unbounded : bound
  (** Unbounded slice bound *)

  val included : int -> bound
  (** Inclusive bound *)

  val excluded : int -> bound
  (** Exclusive bound *)

  val make : tensor_view -> index list -> (t, error) result
  (** Create a slice iterator from a tensor view and slice specifications *)

  val next : t -> string option
  (** Get the next chunk of data from the slice iterator *)

  val remaining_byte_len : t -> int
  (** Get the total number of bytes remaining in the iterator *)

  val newshape : t -> int list
  (** Get the shape of the sliced tensor *)
end

(** {1 Low-level Functions} *)

val read_u64_le : string -> int -> int64
(** Read a little-endian 64-bit integer from a string at the given offset *)

val write_u64_le : Bytes.t -> int -> int64 -> unit
(** Write a little-endian 64-bit integer to a byte buffer at the given offset *)
