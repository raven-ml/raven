#include <assert.h>
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/threads.h>
#include "hdf5_caml.h"

void h5d_finalize(value v)
{
  if (!Hid_closed(v))
    H5Dclose(Hid_val(v));
  Hid_closed(v) = true;
}

static struct custom_operations h5d_ops = {
  "hdf5.h5d",
  h5d_finalize,
  custom_compare_default,
  custom_compare_ext_default,
  custom_hash_default,
  custom_serialize_default,
  custom_deserialize_default,
#ifdef custom_fixed_length_default
  custom_fixed_length_default,
#endif
};

static value alloc_h5d(hid_t id)
{
  value v;
  raise_if_fail(id);
  v = caml_alloc_custom(&h5d_ops, sizeof(hid_t) + sizeof(bool), 0, 1);
  Hid_val(v) = id;
  Hid_closed(v) = false;
  return v;
}

H5D_layout_t H5D_layout_val(value layout)
{
  switch (Int_val(layout))
  {
    case  0: return H5D_COMPACT;
    case  1: return H5D_CONTIGUOUS;
    case  2: return H5D_CHUNKED;
    case  3: return H5D_NLAYOUTS;
    default: caml_failwith("unrecognized H5D.Layout.t");
  }
}

value Val_h5d_layout(H5D_layout_t layout)
{
  switch (layout)
  {
    case H5D_LAYOUT_ERROR: fail();
    case H5D_COMPACT:      return Val_int(0);
    case H5D_CONTIGUOUS:   return Val_int(1);
    case H5D_CHUNKED:      return Val_int(2);
    case H5D_NLAYOUTS:     return Val_int(3);
    default: caml_failwith("unrecognized H5D_layout_t");
  }
}

H5D_space_status_t H5D_space_status_val(value v)
{
  switch (Int_val(v))
  {
    case  0: return H5D_SPACE_STATUS_NOT_ALLOCATED;
    case  1: return H5D_SPACE_STATUS_PART_ALLOCATED;
    case  2: return H5D_SPACE_STATUS_ALLOCATED;
    default: caml_failwith("unrecognized H5d.Space_status.t");
  }
}

value Val_h5d_space_status(H5D_space_status_t s)
{
  switch (s)
  {
    case H5D_SPACE_STATUS_ERROR:          fail();
    case H5D_SPACE_STATUS_NOT_ALLOCATED:  return Val_int(0);
    case H5D_SPACE_STATUS_PART_ALLOCATED: return Val_int(1);
    case H5D_SPACE_STATUS_ALLOCATED:      return Val_int(2);
    default: caml_failwith("unrecognized H5D_space_status_t");
  }
}

value hdf5_h5d_create(value loc_v, value name_v, value dtype_v, value lcpl_v,
    value dcpl_v, value dapl_v, value space_v)
{
  CAMLparam5(loc_v, name_v, dtype_v, lcpl_v, dcpl_v);
  CAMLxparam2(dapl_v, space_v);

  CAMLreturn(alloc_h5d(H5Dcreate2(
    Hid_val(loc_v),
    String_val(name_v),
    Hid_val(dtype_v),
    Hid_val(space_v),
    H5P_opt_val(lcpl_v),
    H5P_opt_val(dcpl_v),
    H5P_opt_val(dapl_v))));
}

value hdf5_h5d_create_bytecode(value *argv, int argn)
{
  assert(argn == 7);
  return hdf5_h5d_create(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]);
}

value hdf5_h5d_create_anon(value loc_v, value type_v, value dcpl_v, value dapl_v,
    value space_v)
{
  CAMLparam5(loc_v, type_v, dcpl_v, dapl_v, space_v);

  CAMLreturn(alloc_h5d(H5Dcreate_anon(
    Hid_val(loc_v),
    Hid_val(type_v),
    Hid_val(space_v),
    H5P_opt_val(dcpl_v),
    H5P_opt_val(dapl_v))));
}

value hdf5_h5d_open(value loc_v, value dapl_v, value name_v)
{
  CAMLparam3(loc_v, dapl_v, name_v);
  CAMLreturn(alloc_h5d(H5Dopen2(
    Hid_val(loc_v),
    String_val(name_v),
    H5P_opt_val(dapl_v))));
}

void hdf5_h5d_close(value dataset_v)
{
  CAMLparam1(dataset_v);
  raise_if_fail(H5Dclose(Hid_val(dataset_v)));
  Hid_closed(dataset_v) = true;
  CAMLreturn0;
}

value hdf5_h5d_get_space(value dataset_v)
{
  CAMLparam1(dataset_v);
  CAMLreturn(alloc_h5s(H5Dget_space(Hid_val(dataset_v))));
}

value hdf5_h5d_get_space_status(value dset_v)
{
  CAMLparam1(dset_v);
  H5D_space_status_t s;
  raise_if_fail(H5Dget_space_status(Hid_val(dset_v), &s));
  CAMLreturn(Val_h5d_space_status(s));
}

value hdf5_h5d_get_type(value dataset_v)
{
  CAMLparam1(dataset_v);
  CAMLreturn(alloc_h5t(H5Dget_type(Hid_val(dataset_v))));
}

value hdf5_h5d_get_create_plist(value dataset_v)
{
  CAMLparam1(dataset_v);
  CAMLreturn(alloc_h5p(H5Dget_create_plist(Hid_val(dataset_v))));
}

void hdf5_h5d_read(value dataset_v, value mem_type_v, value mem_space_v,
  value file_space_v, value xfer_plist_v, value buf_v)
{
  CAMLparam5(dataset_v, mem_type_v, mem_space_v, file_space_v, xfer_plist_v);
  CAMLxparam1(buf_v);
  herr_t err;

#if defined H5_HAVE_THREADSAFE
  caml_release_runtime_system();
#endif
  err = H5Dread(
    Hid_val(dataset_v),
    Hid_val(mem_type_v),
    Hid_val(mem_space_v),
    Hid_val(file_space_v),
    H5P_opt_val(xfer_plist_v),
    (void*) buf_v);
#if defined H5_HAVE_THREADSAFE
  caml_acquire_runtime_system();
#endif
  raise_if_fail(err);

  CAMLreturn0;
}

void hdf5_h5d_read_bytecode(value *argv, int argn)
{
  assert(argn == 6);
  hdf5_h5d_read(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]);
}

void hdf5_h5d_read_bigarray(value dataset_v, value mem_type_v, value mem_space_v,
  value file_space_v, value xfer_plist_v, value buf_v)
{
  CAMLparam5(dataset_v, mem_type_v, mem_space_v, file_space_v, xfer_plist_v);
  CAMLxparam1(buf_v);
  herr_t err;

#if defined H5_HAVE_THREADSAFE
  caml_release_runtime_system();
#endif
  err = H5Dread(
    Hid_val(dataset_v),
    Hid_val(mem_type_v),
    Hid_val(mem_space_v),
    Hid_val(file_space_v),
    H5P_opt_val(xfer_plist_v),
    Caml_ba_data_val(buf_v));
#if defined H5_HAVE_THREADSAFE
  caml_acquire_runtime_system();
#endif
  raise_if_fail(err);

  CAMLreturn0;
}

void hdf5_h5d_read_bigarray_bytecode(value *argv, int argn)
{
  assert(argn == 6);
  hdf5_h5d_read_bigarray(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]);
}

void hdf5_h5d_write(value dataset_v, value mem_type_v, value mem_space_v,
  value file_space_v, value xfer_plist_v, value buf_v)
{
  CAMLparam5(dataset_v, mem_type_v, mem_space_v, file_space_v, xfer_plist_v);
  CAMLxparam1(buf_v);
  herr_t err;

#if defined H5_HAVE_THREADSAFE
  caml_release_runtime_system();
#endif
  err = H5Dwrite(
    Hid_val(dataset_v),
    Hid_val(mem_type_v),
    Hid_val(mem_space_v),
    Hid_val(file_space_v),
    H5P_opt_val(xfer_plist_v),
    (void*) buf_v);
#if defined H5_HAVE_THREADSAFE
  caml_acquire_runtime_system();
#endif
  raise_if_fail(err);

  CAMLreturn0;
}

void hdf5_h5d_write_bytecode(value *argv, int argn)
{
  assert(argn == 6);
  hdf5_h5d_write(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]);
}

void hdf5_h5d_write_bigarray(value dataset_v, value mem_type_v, value mem_space_v,
  value file_space_v, value xfer_plist_v, value buf_v)
{
  CAMLparam5(dataset_v, mem_type_v, mem_space_v, file_space_v,
    xfer_plist_v);
  CAMLxparam1(buf_v);
  herr_t err;

#if defined H5_HAVE_THREADSAFE
  caml_release_runtime_system();
#endif
  err = H5Dwrite(
    Hid_val(dataset_v),
    Hid_val(mem_type_v),
    Hid_val(mem_space_v),
    Hid_val(file_space_v),
    H5P_opt_val(xfer_plist_v),
    Caml_ba_data_val(buf_v));
#if defined H5_HAVE_THREADSAFE
  caml_acquire_runtime_system();
#endif
  raise_if_fail(err);

  CAMLreturn0;
}

void hdf5_h5d_write_bigarray_bytecode(value *argv, int argn)
{
  assert(argn == 6);
  hdf5_h5d_write_bigarray(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]);
}

void hdf5_h5d_set_extent(value dset_v, value size_v)
{
  CAMLparam2(dset_v, size_v);
  hsize_t *size;
  herr_t err;

  (void) hsize_t_array_val(size_v, &size);
  if (size == NULL)
    caml_raise_out_of_memory();
  err = H5Dset_extent(Hid_val(dset_v), size);
  free(size);
  raise_if_fail(err);

  CAMLreturn0;
}

void hdf5_h5d_vlen_reclaim(value type_v, value space_v, value plist_v, value buf_v)
{
  CAMLparam4(type_v, space_v, plist_v, buf_v);
  raise_if_fail(
    H5Dvlen_reclaim(Hid_val(type_v), Hid_val(space_v), Hid_val(plist_v), (void*) buf_v));
  CAMLreturn0;
}
