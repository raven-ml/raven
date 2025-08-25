#include <assert.h>
#include <caml/alloc.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include "hdf5_caml.h"

void h5o_finalize(value v)
{
  if (!Hid_closed(v))
    H5Oclose(Hid_val(v));
  Hid_closed(v) = true;
}

static struct custom_operations h5o_ops = {
  "hdf5.h5o",
  h5o_finalize,
  custom_compare_default,
  custom_compare_ext_default,
  custom_hash_default,
  custom_serialize_default,
  custom_deserialize_default,
#ifdef custom_fixed_length_default
  custom_fixed_length_default,
#endif
};

static value alloc_h5o(hid_t id, bool close)
{
  value v;
  raise_if_fail(id);
  v = caml_alloc_custom(&h5o_ops, sizeof(hid_t) + sizeof(bool), 0, 1);
  Hid_val(v) = id;
  Hid_closed(v) = !close;
  return v;
}

unsigned H5O_copy_val(value v)
{
  CAMLparam0();
  CAMLlocal1(h);
  unsigned flags = 0;

  while (Is_block(v))
  {
    assert(Tag_val(v) == 0);
    h = Field(v, 0);
    v = Field(v, 1);
    assert(Is_long(h));
    switch (Long_val(h))
    {
      case 0: flags |= H5O_COPY_SHALLOW_HIERARCHY_FLAG; break;
      case 1: flags |= H5O_COPY_EXPAND_SOFT_LINK_FLAG ; break;
      case 2: flags |= H5O_COPY_EXPAND_EXT_LINK_FLAG  ; break;
      case 3: flags |= H5O_COPY_EXPAND_REFERENCE_FLAG ; break;
      case 4: flags |= H5O_COPY_WITHOUT_ATTR_FLAG     ; break;
      case 5: flags |= H5O_COPY_PRESERVE_NULL_FLAG    ; break;
      case 6: flags |= H5O_COPY_ALL                   ; break;
      default: caml_failwith("unrecognized Copy.t");
    }
  }
  assert(Int_val(v) == 0);
  CAMLreturnT(unsigned, flags);
}

value val_h5o_copy(unsigned f)
{
  CAMLparam0();
  CAMLlocal3(h, v, t);

  h = Val_unit;
  while (f != 0)
  {
    if (f & H5O_COPY_ALL)
    {
      v = Val_int(6);
      f ^= H5O_COPY_ALL;
    }
    else if (f & H5O_COPY_SHALLOW_HIERARCHY_FLAG)
    {
      v = Val_int(0);
      f ^= H5O_COPY_SHALLOW_HIERARCHY_FLAG;
    }
    else if (f & H5O_COPY_EXPAND_SOFT_LINK_FLAG)
    {
      v = Val_int(1);
      f ^= H5O_COPY_EXPAND_SOFT_LINK_FLAG;
    }
    else if (f & H5O_COPY_EXPAND_EXT_LINK_FLAG)
    {
      v = Val_int(2);
      f ^= H5O_COPY_EXPAND_EXT_LINK_FLAG;
    }
    else if (f & H5O_COPY_EXPAND_REFERENCE_FLAG)
    {
      v = Val_int(3);
      f ^= H5O_COPY_EXPAND_REFERENCE_FLAG;
    }
    else if (f & H5O_COPY_WITHOUT_ATTR_FLAG)
    {
      v = Val_int(4);
      f ^= H5O_COPY_WITHOUT_ATTR_FLAG;
    }
    else if (f & H5O_COPY_PRESERVE_NULL_FLAG)
    {
      v = Val_int(5);
      f ^= H5O_COPY_PRESERVE_NULL_FLAG;
    }
    else caml_failwith("unrecognized Copy.t");
    t = h;
    h = caml_alloc(2, 0);
    Store_field(h, 0, v);
    Store_field(h, 1, t);
  }
  CAMLreturn(h);
}

value Val_h5o_copy(unsigned v)
{
  switch (v)
  {
    case H5O_COPY_SHALLOW_HIERARCHY_FLAG: return 0;
    case H5O_COPY_EXPAND_SOFT_LINK_FLAG : return 1;
    case H5O_COPY_EXPAND_EXT_LINK_FLAG  : return 2;
    case H5O_COPY_EXPAND_REFERENCE_FLAG : return 3;
    case H5O_COPY_WITHOUT_ATTR_FLAG     : return 4;
    case H5O_COPY_PRESERVE_NULL_FLAG    : return 5;
    case H5O_COPY_ALL                   : return 6;
    default: caml_failwith("unrecognized Copy.t");
  }
}

H5O_type_t H5O_type_val(value type)
{
  switch (Int_val(type))
  {
    case 0: return H5O_TYPE_GROUP;
    case 1: return H5O_TYPE_DATASET;
    case 2: return H5O_TYPE_NAMED_DATATYPE;
    case 3: return H5O_TYPE_NTYPES;
    default: caml_failwith("unrecognized H5O_type_t");
  }
}

value Val_h5o_type(H5O_type_t type)
{
  switch (type)
  {
    case H5O_TYPE_UNKNOWN:        fail();
    case H5O_TYPE_GROUP:          return Val_int(0);
    case H5O_TYPE_DATASET:        return Val_int(1);
    case H5O_TYPE_NAMED_DATATYPE: return Val_int(2);
    case H5O_TYPE_NTYPES:         return Val_int(3);
    default: caml_failwith("unrecognized H5O_type_t");
  }
}

H5O_hdr_info_t H5O_hdr_info_val(value hdr_info_v)
{
  CAMLparam1(hdr_info_v);
  CAMLlocal2(space_v, mesg_v);
  H5O_hdr_info_t hdr_info;
  space_v = Field(hdr_info_v, 4);
  mesg_v  = Field(hdr_info_v, 5);
  hdr_info.version = Int_val(Field(hdr_info_v, 0));
  hdr_info.nmesgs  = Int_val(Field(hdr_info_v, 1));
  hdr_info.nchunks = Int_val(Field(hdr_info_v, 2));
  hdr_info.flags   = Int_val(Field(hdr_info_v, 3));
  hdr_info.space.total = Int_val(Field(space_v, 0));
  hdr_info.space.meta  = Int_val(Field(space_v, 1));
  hdr_info.space.mesg  = Int_val(Field(space_v, 2));
  hdr_info.space.free  = Int_val(Field(space_v, 3));
  hdr_info.mesg.present = Int_val(Field(mesg_v, 0));
  hdr_info.mesg.shared  = Int_val(Field(mesg_v, 1));
  CAMLreturnT(H5O_hdr_info_t, hdr_info);
}

value Val_h5o_hdr_info(const H5O_hdr_info_t *hdr_info)
{
  CAMLparam0();
  CAMLlocal3(hdr_info_v, space_v, mesg_v);
  hdr_info_v = caml_alloc_tuple(6);
  space_v    = caml_alloc_tuple(4);
  mesg_v     = caml_alloc_tuple(2);
  Store_field(hdr_info_v, 0, Val_int(hdr_info->version));
  Store_field(hdr_info_v, 1, Val_int(hdr_info->nmesgs));
  Store_field(hdr_info_v, 2, Val_int(hdr_info->nchunks));
  Store_field(hdr_info_v, 3, Val_int(hdr_info->flags));
  Store_field(hdr_info_v, 4, space_v);
  Store_field(hdr_info_v, 5, mesg_v);
  Store_field(space_v, 0, Val_int(hdr_info->space.total));
  Store_field(space_v, 1, Val_int(hdr_info->space.meta));
  Store_field(space_v, 2, Val_int(hdr_info->space.mesg));
  Store_field(space_v, 3, Val_int(hdr_info->space.free));
  Store_field(mesg_v, 0, Val_int(hdr_info->mesg.present));
  Store_field(mesg_v, 1, Val_int(hdr_info->mesg.shared));
  CAMLreturn(hdr_info_v);
}

H5O_info_t H5O_info_val(value info_v)
{
  CAMLparam1(info_v);
  CAMLlocal1(meta_size_v);
  H5O_info_t info;
  meta_size_v         = Field(info_v, 10);
  info.fileno         = Int_val(Field(info_v, 0));
  info.addr           = Hid_val(Field(info_v, 1));
  info.type           = H5O_type_val(Field(info_v, 2));
  info.rc             = Int_val(Field(info_v, 3));
  info.atime          = Int64_val(Field(info_v, 4));
  info.mtime          = Int64_val(Field(info_v, 5));
  info.ctime          = Int64_val(Field(info_v, 6));
  info.btime          = Int64_val(Field(info_v, 7));
  info.num_attrs      = Int_val(Field(info_v, 8));
  info.hdr            = H5O_hdr_info_val(Field(info_v, 9));
  info.meta_size.obj  = H5_ih_info_val(Field(meta_size_v, 0));
  info.meta_size.attr = H5_ih_info_val(Field(meta_size_v, 1));
  CAMLreturnT(H5O_info_t, info);
}

value Val_h5o_info(const H5O_info_t *info)
{
  CAMLparam0();
  CAMLlocal2(info_v, meta_size_v);
  info_v      = caml_alloc_tuple(10);
  meta_size_v = caml_alloc_tuple(2);
  Store_field(info_v, 0, Val_int(info->fileno));
  Store_field(info_v, 1, info->addr);
  Store_field(info_v, 2, Val_h5o_type(info->type));
  Store_field(info_v, 3, Val_int(info->rc));
  Store_field(info_v, 4, caml_copy_int64(info->atime));
  Store_field(info_v, 5, caml_copy_int64(info->mtime));
  Store_field(info_v, 6, caml_copy_int64(info->ctime));
  Store_field(info_v, 7, caml_copy_int64(info->btime));
  Store_field(info_v, 8, Val_int(info->num_attrs));
  Store_field(info_v, 9, Val_h5o_hdr_info(&info->hdr));
  Store_field(info_v, 10, meta_size_v);
  Store_field(meta_size_v, 0, Val_h5_ih_info(info->meta_size.obj));
  Store_field(meta_size_v, 1, Val_h5_ih_info(info->meta_size.attr));
  CAMLreturn(info_v);
}

value hdf5_h5o_open(value loc_v, value lapl_v, value name_v)
{
  CAMLparam3(loc_v, lapl_v, name_v);
  CAMLreturn(alloc_h5o(H5Oopen(Hid_val(loc_v), String_val(name_v),
    H5P_opt_val(lapl_v)), true));
}

void hdf5_h5o_close(value object_v)
{
  CAMLparam1(object_v);
  raise_if_fail(H5Oclose(Hid_val(object_v)));
  Hid_closed(object_v) = true;
  CAMLreturn0;
}

void hdf5_h5o_copy(value src_loc_v, value src_name_v, value dst_loc_v, value ocpypl_v,
  value lcpl_v, value dst_name_v)
{
  CAMLparam5(src_loc_v, src_name_v, dst_loc_v, ocpypl_v, lcpl_v);
  CAMLxparam1(dst_name_v);
  raise_if_fail(H5Ocopy(Hid_val(src_loc_v), String_val(src_name_v), Hid_val(dst_loc_v),
    String_val(dst_name_v), H5P_opt_val(ocpypl_v), H5P_opt_val(lcpl_v)));
  CAMLreturn0;
}

void hdf5_h5o_copy_bytecode(value *argv, int argn)
{
  assert(argn == 6);
  hdf5_h5o_copy(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]);
}

void hdf5_h5o_set_comment(value object_v, value comment_v)
{
  CAMLparam2(object_v, comment_v);
  raise_if_fail(H5Oset_comment(Hid_val(object_v), String_val(comment_v)));
  CAMLreturn0;
}

value hdf5_h5o_get_info(value object_v)
{
  CAMLparam1(object_v);
  H5O_info_t object_info;
  raise_if_fail(H5Oget_info(Hid_val(object_v), &object_info));
  CAMLreturn(Val_h5o_info(&object_info));
}

value hdf5_h5o_get_info_by_name(value loc_v, value lapl_v, value object_name_v)
{
  CAMLparam3(loc_v, lapl_v, object_name_v);
  H5O_info_t object_info;
  raise_if_fail(H5Oget_info_by_name(Hid_val(loc_v), String_val(object_name_v),
    &object_info, H5P_opt_val(lapl_v)));
  CAMLreturn(Val_h5o_info(&object_info));
}
