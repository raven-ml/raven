#include <caml/mlvalues.h>
#include <caml/bigarray.h>
#include <caml/memory.h>
#include <caml/fail.h>
#include <unistd.h>
#include <string.h>

/* Write bigarray data to file descriptor */
CAMLprim value caml_npy_write_bigarray(value vfd, value vba) {
  CAMLparam2(vfd, vba);
  /* Unix.file_descr is represented as an integer in OCaml */
  int fd = Int_val(vfd);
  struct caml_ba_array *ba = Caml_ba_array_val(vba);
  
  /* Calculate total size in bytes */
  uintnat size = caml_ba_byte_size(ba);
  
  /* Write the data */
  ssize_t written = write(fd, ba->data, size);
  
  if (written < 0) {
    caml_failwith("Failed to write bigarray to file");
  }
  if ((size_t)written != size) {
    caml_failwith("Incomplete write of bigarray to file");
  }
  
  CAMLreturn(Val_unit);
}

/* Read file data into bigarray */
CAMLprim value caml_npy_read_bigarray(value vfd, value vba) {
  CAMLparam2(vfd, vba);
  int fd = Int_val(vfd);
  struct caml_ba_array *ba = Caml_ba_array_val(vba);
  
  /* Calculate total size in bytes */
  uintnat size = caml_ba_byte_size(ba);
  
  /* Read the data */
  ssize_t bytes_read = read(fd, ba->data, size);
  
  if (bytes_read < 0) {
    caml_failwith("Failed to read file into bigarray");
  }
  if ((size_t)bytes_read != size) {
    caml_failwith("Incomplete read of file into bigarray");
  }
  
  CAMLreturn(Val_unit);
}