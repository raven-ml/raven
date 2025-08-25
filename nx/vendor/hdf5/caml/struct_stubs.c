#include <assert.h>
#include <memory.h>
#include <stdbool.h>
#include <stdio.h>
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/intext.h>
#include <caml/memory.h>
#include "hdf5_caml.h"

value hdf5_caml_struct_bigstring_of_string(value s_v)
{
  CAMLparam1(s_v);
  CAMLlocal1(b_v);
  long dim;
  void *data;

  dim = caml_string_length(s_v);
  b_v =
    caml_ba_alloc_dims(
      CAML_BA_CHAR | CAML_BA_C_LAYOUT | CAML_BA_MANAGED, 1, NULL, Bosize_val(s_v) + 1);
  data = Caml_ba_data_val(b_v);
  memcpy(data, String_val(s_v), dim);
  /* HDF5 requires strings to be null terminated */
  ((char*) data)[dim] = '\0';

  CAMLreturn(b_v);
}

value hdf5_caml_struct_bigstring_to_string(value b_v)
{
  CAMLparam1(b_v);
  CAMLreturn(caml_copy_string(Caml_ba_data_val(b_v)));
}

value hdf5_caml_struct_array_char_of_string(value s_v)
{
  CAMLparam1(s_v);
  CAMLlocal1(b_v);
  long dim;
  void *data;

  dim = caml_string_length(s_v);
  b_v =
    caml_ba_alloc_dims(CAML_BA_CHAR | CAML_BA_C_LAYOUT | CAML_BA_MANAGED, 1, NULL, dim);
  data = Caml_ba_data_val(b_v);
  memcpy(data, String_val(s_v), dim);

  CAMLreturn(b_v);
}

value hdf5_caml_struct_array_char_to_string(value b_v)
{
  CAMLparam1(b_v);
  CAMLlocal1(s_v);
  struct caml_ba_array *b;
  size_t len;

  b = Caml_ba_array_val(b_v);
  len = b->dim[0];
  s_v = caml_alloc_string(len);
  memcpy((void*) String_val(s_v), b->data, len);
  CAMLreturn(s_v);
}

/* [Mem.t] and [Ptr.t] support marshalling with sharing.  Each call to [Marshal.to_*] is
   assigned a new [serialize_count].  Serializing [Mem.t] and [Ptr.t] stores the
   underlying array of structs only if it wasn't already serialized for the current
   [serialize_count].  Otherwise, it stores the reference to the one already serialized.

   Deserialization [Mem.t] and [Ptr.t] has to deserialize the underlying array of structs
   or use one already deserialized. */

void hdf5_mem_finalize(value);
void hdf5_mem_serialize(value, uintnat*, uintnat*);
uintnat hdf5_mem_deserialize(void*);

static struct custom_operations hdf5_mem_ops = {
  "hdf5.struct.mem",
  hdf5_mem_finalize,
  custom_compare_default,
  custom_hash_default,
  hdf5_mem_serialize,
  hdf5_mem_deserialize,
  custom_compare_ext_default,
#ifdef custom_fixed_length_default
  custom_fixed_length_default,
#endif
};

enum type {
  /* Includes Int, Int64, Float64 and String from [type.mli] */
  simple,
  bigstring,
  array
};

struct hdf5_caml_field {
  /* The size of the field */
  size_t size;
  /* Indicates whether the field has a variable length */
  enum type type;
  /* The size of an element of the array if [type] is variable */
  size_t element_size;
  /* The proxies for variable length field */
  struct caml_ba_proxy **proxies;
};

struct hdf5_caml_mem {
  size_t refcount;
  void *data;
  /* The following fields are stored as value since they're mostly used from the OCaml
     code.  They are always integers so it's not a problem that they're not tracked by the
     GC. */
  /* The number of elements that [data] is allocated for. */
  value capacity;
  /* The number of elements allowed to be accessed in [data].  It is less or equal than
     [capacity]. */
  value nmemb;
  /* The size of a single element stored in [data]. */
  value size;
  /* Contains the last [serialize_count] for which this struct was serialized. */
  unsigned long serialize_count;
  /* The ID of this struct in the latest serialization. */
  unsigned long serialize_id;
  /* The number of table fields. */
  value nfields;
  /* The field data. */
  struct hdf5_caml_field *fields;
  /* Indicates whether bigstring and array fields are present. */
  bool has_varlen;
};

#define Mem_val(v) (*((struct hdf5_caml_mem**) Data_custom_val(v)))

#define CAML_HDF5_MAX_MEMORY (1024 * 1024 * 1024)

/* The current amount of memory allocated. */
static size_t memory_allocated = 0;

void caml_hdf5_raise_out_of_memory(void)
{
  char msg[256];
  snprintf(msg, sizeof(msg), "Out of memory: %ld", memory_allocated);
  caml_failwith(msg);
}

value hdf5_caml_struct_mem_create(
  value capacity_v, value size_v, value field_sizes_v, value types_v)
{
  CAMLparam4(capacity_v, size_v, field_sizes_v, types_v);
  CAMLlocal2(type_v, v);
  void *data;
  size_t capacity, size, nfields, i, element_size;
  struct hdf5_caml_mem *mem;
  struct hdf5_caml_field *fields;
  enum type type;

  capacity = Long_val(capacity_v);
  size = Long_val(size_v);
  nfields = Wosize_val(field_sizes_v);

  data = malloc(capacity * size);
  if (data == NULL)
    caml_hdf5_raise_out_of_memory();

  mem = malloc(sizeof(struct hdf5_caml_mem));
  if (mem == NULL)
  {
    free(data);
    caml_hdf5_raise_out_of_memory();
  }

  fields = malloc(nfields * sizeof(struct hdf5_caml_field));
  if (nfields > 0 && fields == NULL)
  {
    free(data);
    free(mem);
    caml_hdf5_raise_out_of_memory();
  }

  memory_allocated += capacity * size;
  mem->data = data;
  mem->refcount = 1;
  mem->capacity = capacity_v;
  mem->nmemb = capacity_v;
  mem->size = size_v;
  mem->serialize_count = 0;
  mem->serialize_id = 0;
  mem->nfields = Val_long(nfields);
  mem->fields = fields;
  mem->has_varlen = false;

  for (i = 0; i < nfields; i++)
  {
    type_v = Field(types_v, i);
    if (Is_long(type_v))
      switch (Long_val(type_v))
      {
        case 0:
        case 1:
        case 2:
          type = simple;
          element_size = 1;
          break;
        case 3:
          type = bigstring;
          element_size = 1;
          break;
        case 4:
          type = array;
          element_size = 4;
          break;
        case 5:
          type = array;
          element_size = 8;
          break;
        case 6:
        case 7:
          type = array;
          element_size = 1;
          break;
        case 8:
        case 9:
          type = array;
          element_size = 2;
          break;
        case 10:
          type = array;
          element_size = 4;
          break;
        case 11:
        case 12:
        case 13:
          type = array;
          element_size = 8;
          break;
        case 14:
          type = array;
          element_size = 1;
          break;
        default: caml_failwith("Unexpected field type");
      }
    else
      switch (Tag_val(type_v))
      {
        case 0:
          type = simple;
          element_size = 1;
          break;
        default: caml_failwith("Unexpected field type");
      };
    fields[i].size         = Long_val(Field(field_sizes_v, i));
    fields[i].type         = type;
    fields[i].element_size = element_size;
    fields[i].proxies      = NULL;
    mem->has_varlen |= (type != simple);
  }
  /* Variable fields need to be initialized to NULL since they will be freed on finalize
    */
  if (mem->has_varlen)
    memset(data, 0, capacity * size);

  v =
    caml_alloc_custom(
      &hdf5_mem_ops, sizeof(struct hdf5_caml_mem*), capacity * size,
      CAML_HDF5_MAX_MEMORY);
  Mem_val(v) = mem;

  CAMLreturn(v);
}

value hdf5_caml_struct_mem_of_mem(struct hdf5_caml_mem *mem)
{
  CAMLparam0();
  CAMLlocal1(v);
  v = caml_alloc_custom(&hdf5_mem_ops, sizeof(struct hdf5_caml_mem*), 0, 1);
  Mem_val(v) = mem;
  mem->refcount++;
  CAMLreturn(v);
}

void mem_free(struct hdf5_caml_mem *mem)
{
  struct caml_ba_proxy **proxies, *proxy;
  char *data;
  size_t i, j, nfields, offset, size, capacity;
  hvl_t *hvl;

  if (--mem->refcount == 0)
  {
    size = Long_val(mem->size);
    offset = 0;
    nfields = Long_val(mem->nfields);
    capacity = Long_val(mem->capacity);
    for (i = 0; i < nfields; i++)
    {
      switch (mem->fields[i].type)
      {
        case simple: break;
        case bigstring:
          data = (char*) mem->data + offset;
          proxies = mem->fields[i].proxies;
          if (proxies == NULL)
            for (j = 0; j < capacity; j++)
            {
              free(*((void**) data));
              data += size;
            }
          else
          {
            for (j = 0; j < capacity; j++)
            {
              proxy = proxies[j];
              if (proxy == NULL || --proxy->refcount == 0)
              {
                free(*((void**) data));
                free(proxy);
              }
              data += size;
            }
            free(proxies);
          }
          break;
        case array:
          data = (char*) mem->data + offset;
          proxies = mem->fields[i].proxies;
          if (proxies == NULL)
            for (j = 0; j < capacity; j++)
            {
              hvl = (hvl_t*) data;
              free(hvl->p);
              data += size;
            }
          else
          {
            for (j = 0; j < capacity; j++)
            {
              hvl = (hvl_t*) data;
              proxy = proxies[j];
              if (proxy == NULL || --proxy->refcount == 0)
              {
                free(hvl->p);
                free(proxy);
              }
              data += size;
            }
            free(proxies);
          }
          break;
      }
      offset += (mem->fields[i].size + 7) / 8 * 8;
    }
    memory_allocated -= capacity * size;
    free(mem->fields);
    free(mem->data);
    free(mem);
  }
}

void hdf5_mem_finalize(value v)
{
  mem_free(Mem_val(v));
}

static unsigned long serialize_count = 0;
static unsigned long serialize_id = 0;

void serialize_mem(struct hdf5_caml_mem *mem)
{
  size_t capacity, nmemb, size, nfields, offset, len, i, j;
  char *s, *data;
  hvl_t *hvl;

  /* Check whether this [hdf5_caml_mem] was already serialized in this serialization
     session.  If so, only store its ID.  Otherwise assign it a new ID and store the whole
     structure. */
  if (mem->serialize_count == serialize_count)
  {
    caml_serialize_int_8(mem->serialize_id);
  }
  else
  {
    mem->serialize_count = serialize_count;
    mem->serialize_id = serialize_id++;
    capacity = Long_val(mem->capacity);
    nmemb    = Long_val(mem->nmemb);
    size     = Long_val(mem->size);
    nfields  = Long_val(mem->nfields);
    caml_serialize_int_8(mem->serialize_id);
    caml_serialize_int_8(capacity);
    caml_serialize_int_8(nmemb);
    caml_serialize_int_8(size);
    caml_serialize_int_8(nfields);
    for (i = 0; i < nfields; i++)
    {
      caml_serialize_int_8(mem->fields[i].size);
      caml_serialize_int_1(mem->fields[i].type);
      caml_serialize_int_2(mem->fields[i].element_size);
    }
    caml_serialize_block_1(mem->data, nmemb * size);
    offset = 0;
    for (i = 0; i < nfields; i++)
    {
      switch (mem->fields[i].type)
      {
        case simple: break;
        case bigstring:
          data = (char*) mem->data + offset;
          for (j = 0; j < capacity; j++)
          {
            s = *((char**) data);
            if (s == NULL)
              caml_serialize_int_8(0);
            else
            {
              len = strnlen(s, MAX_BIGSTRING_LEN);
              caml_serialize_int_8(len + 1);
              caml_serialize_block_1(s, len);
            }
            data += size;
          };
          break;
        case array:
          data = (char*) mem->data + offset;
          for (j = 0; j < capacity; j++)
          {
            hvl = (hvl_t*) data;
            if (hvl->p == NULL)
              caml_serialize_int_8(0);
            else
            {
              caml_serialize_int_8(hvl->len + 1);
              caml_serialize_block_1(hvl->p, hvl->len * mem->fields[i].element_size);
            }
            data += size;
          };
          break;
      }
      offset += mem->fields[i].size;
    }
  }
}

void hdf5_mem_serialize(value v, uintnat *wsize_32, uintnat *wsize_64)
{
  serialize_mem(Mem_val(v));
  *wsize_32 = sizeof(struct hdf5_caml_mem*);
  *wsize_64 = sizeof(struct hdf5_caml_mem*);
}

static struct hdf5_caml_mem **mems = NULL;
static size_t mems_capacity = 0;
static unsigned long deserialize_id = 0;

struct hdf5_caml_mem* deserialize_mem(void)
{
  struct hdf5_caml_mem *mem;
  unsigned long id;
  size_t capacity, nmemb, size, nfields, offset, len, i, j;
  struct hdf5_caml_field *fields;
  char *s, *data;
  hvl_t *hvl;

  id = caml_deserialize_uint_8();
  /* If we have already deserialized this [hdf5_caml_mem] just fetch it.  IDs come in
     sequence, so comparing it to the largest deserialized ID checks whether it was
     already deserialized. */
  if (id < deserialize_id)
    mem = mems[id];
  else
  {
    deserialize_id = id + 1;
    if (id >= mems_capacity)
    {
      mems_capacity = id * 2;
      mems = realloc(mems, mems_capacity * sizeof(struct hdf5_caml_mem*));
      if (mems == NULL)
        caml_hdf5_raise_out_of_memory();
    }
    capacity = caml_deserialize_uint_8();
    nmemb    = caml_deserialize_uint_8();
    size     = caml_deserialize_uint_8();
    nfields  = caml_deserialize_uint_8();
    mem = malloc(sizeof(struct hdf5_caml_mem));
    if (mem == NULL)
      caml_hdf5_raise_out_of_memory();
    fields = malloc(nfields * sizeof(struct hdf5_caml_field));
    if (nfields > 0 && fields == NULL)
    {
      free(mem);
      caml_hdf5_raise_out_of_memory();
    }
    data = malloc(capacity * size);
    if (data == NULL)
    {
      free(mem);
      free(fields);
      caml_hdf5_raise_out_of_memory();
    }
    for (i = 0; i < nfields; i++)
    {
      fields[i].size         = caml_deserialize_uint_8();
      fields[i].type         = caml_deserialize_uint_1();
      fields[i].element_size = caml_deserialize_uint_2();
      fields[i].proxies      = NULL;
    }
    memory_allocated += capacity * size;
    caml_deserialize_block_1(data, nmemb * size);
    mem->refcount = 0;
    mem->data = data;
    mem->capacity = Val_long(capacity);
    mem->nmemb = Val_long(nmemb);
    mem->size = Val_long(size);
    mem->serialize_count = 0;
    mem->serialize_id = 0;
    mem->nfields = Val_long(nfields);
    mem->fields = fields;
    mem->has_varlen = false;
    mems[id] = mem;
    offset = 0;
    for (i = 0; i < nfields; i++)
    {
      switch (fields[i].type)
      {
        case simple: break;
        case bigstring:
          data = (char*) mem->data + offset;
          for (j = 0; j < capacity; j++)
          {
            len = caml_deserialize_uint_8();
            if (len == 0)
              *((void**) data) = NULL;
            else
            {
              s = malloc(len);
              if (s == NULL)
                /* There's a memory leak here but it's highly unlikely it will matter */
                caml_hdf5_raise_out_of_memory();
              caml_deserialize_block_1(s, len - 1);
              s[len - 1] = '\0';
              *((char**) data) = s;
            };
            data += size;
          };
          mem->has_varlen = true;
          break;
        case array:
          data = (char*) mem->data + offset;
          for (j = 0; j < capacity; j++)
          {
            hvl = (hvl_t*) data;
            len = caml_deserialize_uint_8();
            if (len == 0)
            {
              hvl->len = 0;
              hvl->p = NULL;
            }
            else
            {
              len--;
              hvl->len = len;
              hvl->p = malloc(len * fields[i].element_size);
              if (hvl->p == NULL)
                /* There's a memory leak here but it's highly unlikely it will matter */
                caml_hdf5_raise_out_of_memory();
              caml_deserialize_block_1(hvl->p, len * fields[i].element_size);
            };
            data += size;
          };
          mem->has_varlen = true;
          break;
      }
      offset += fields[i].size;
    }
  }

  return mem;
}

uintnat hdf5_mem_deserialize(void *dst)
{
  struct hdf5_caml_mem *mem;

  mem = deserialize_mem();
  mem->refcount++;
  *((struct hdf5_caml_mem**) dst) = mem;
  return sizeof(struct hdf5_caml_mem*);
}

void hdf5_caml_struct_mem_write_buf(
  struct hdf5_caml_mem *mem, value buf_v, value dst_pos_v)
{
  memcpy(
    (char*) Caml_ba_data_val(buf_v) + Long_val(dst_pos_v),
    mem->data,
    Long_val(mem->nmemb) * Long_val(mem->size));
}

void hdf5_caml_struct_mem_read_buf(
  struct hdf5_caml_mem *mem, value buf_v, value dst_pos_v)
{
  memcpy(
    mem->data,
    (char*) Caml_ba_data_val(buf_v) + Long_val(dst_pos_v),
    Long_val(mem->nmemb) * Long_val(mem->size));
}

void hdf5_caml_struct_mem_realloc(value t_v, value capacity_v)
{
  CAMLparam2(t_v, capacity_v);
  struct hdf5_caml_mem *mem;
  void *data;
  size_t i, old_capacity, capacity, size, nfields;
  struct caml_ba_proxy **proxies;

  mem = Mem_val(t_v);
  old_capacity = Long_val(mem->capacity);
  capacity = Long_val(capacity_v);
  size = Long_val(mem->size);
  nfields = Long_val(mem->nfields);
  data = realloc(mem->data, capacity * size);
  if (data == NULL)
    caml_hdf5_raise_out_of_memory();
  if (mem->has_varlen)
    bzero((char*) data + old_capacity * size, (capacity - old_capacity) * size);
  mem->data = data;
  for (i = 0; i < nfields; i++)
  {
    proxies = mem->fields[i].proxies;
    if (proxies != NULL)
    {
      proxies = realloc(proxies, capacity * sizeof(struct caml_ba_proxy*));
      if (proxies == NULL)
        caml_hdf5_raise_out_of_memory();
      bzero(
        proxies + old_capacity,
        (capacity - old_capacity) * sizeof(struct caml_ba_proxy*));
      mem->fields[i].proxies = proxies;
    }
  }
  mem->capacity = capacity_v;
  CAMLreturn0;
}

void hdf5_caml_struct_mem_blit(
  value src_v, value src_pos_v, value dst_v, value dst_pos_v, value len_v)
{
  CAMLparam3(src_v, dst_v, len_v);
  struct hdf5_caml_mem *src, *dst;
  size_t src_pos, dst_pos, len, nfields, size, offset, dim, i, j;
  struct caml_ba_proxy **src_proxies, **dst_proxies, *proxy;
  char *data;
  enum type type;
  hvl_t *hvl;

  src     = Mem_val(src_v);
  dst     = Mem_val(dst_v);
  src_pos = Long_val(src_pos_v);
  dst_pos = Long_val(dst_pos_v);
  len     = Long_val(len_v);
  size    = Long_val(src->size);
  nfields = Long_val(src->nfields);

  memcpy(
    (char*) dst->data + dst_pos * size,
    (char*) src->data + src_pos * size,
    len * size);

  offset = 0;
  for (i = 0; i < nfields; i++)
  {
    type = src->fields[i].type;
    if (type != simple)
    {
      src_proxies = src->fields[i].proxies;
      if (src_proxies == NULL)
      {
        src_proxies = calloc(Long_val(src->capacity), sizeof(struct caml_ba_proxy*));
        if (src_proxies == NULL)
          caml_hdf5_raise_out_of_memory();
        src->fields[i].proxies = src_proxies;
      }
      dst_proxies = dst->fields[i].proxies;
      if (dst_proxies == NULL)
      {
        dst_proxies = calloc(Long_val(dst->capacity), sizeof(struct caml_ba_proxy*));
        if (dst_proxies == NULL)
          caml_hdf5_raise_out_of_memory();
        dst->fields[i].proxies = dst_proxies;
      }
      switch (type)
      {
        case simple: break;
        case bigstring:
          data = (char*) src->data + offset;
          for (j = 0; j < len; j++)
          {
            proxy = src_proxies[src_pos + j];
            if (proxy == NULL)
            {
              proxy = malloc(sizeof(struct caml_ba_proxy));
              if (proxy == NULL)
                caml_hdf5_raise_out_of_memory();
              proxy->refcount = 2;
              proxy->data = *((void**) data);
              dim = strnlen(proxy->data, MAX_BIGSTRING_LEN);
              if (dim == MAX_BIGSTRING_LEN && ((char*) proxy->data)[dim] != '\0')
                caml_failwith("The given C_string bigger than 1 Gb");
              proxy->size = dim;
              src_proxies[src_pos + j] = proxy;
            }
            else
              ++proxy->refcount;
            dst_proxies[dst_pos + j] = proxy;
            data += size;
          };
          break;
        case array:
          data = (char*) src->data + offset;
          for (j = 0; j < len; j++)
          {
            proxy = src_proxies[src_pos + j];
            hvl = (hvl_t*) data;
            if (proxy == NULL)
            {
              proxy = malloc(sizeof(struct caml_ba_proxy));
              if (proxy == NULL)
                caml_hdf5_raise_out_of_memory();
              proxy->refcount = 2;
              proxy->data = hvl->p;
              proxy->size = hvl->len * src->fields[i].element_size;
              src_proxies[src_pos + j] = proxy;
            }
            else
              ++proxy->refcount;
            dst_proxies[dst_pos + j] = proxy;
            data += size;
          };
          break;
      }
    }
    offset += src->fields[i].size;
  }

  CAMLreturn0;
}

value hdf5_caml_struct_mem_field(value mem_v, value i_v)
{
  CAMLparam2(mem_v, i_v);
  CAMLlocal1(res_v);
  struct hdf5_caml_mem *mem;
  long i;

  mem = Mem_val(mem_v);
  i = Long_val(i_v);

  if (i < 0 || i >= Long_val(mem->nfields))
    caml_invalid_argument("index out of bounds");

  res_v = caml_alloc_tuple(3);
  Store_field(res_v, 0, Val_long(mem->fields[i].size));
  Store_field(res_v, 1, Val_int((unsigned int) mem->fields[i].type));
  Store_field(res_v, 2, Val_int((unsigned int) mem->fields[i].element_size));
  CAMLreturn(res_v);
}

void hdf5_ptr_finalize(value v);
void hdf5_ptr_serialize(value, uintnat*, uintnat*);
uintnat hdf5_ptr_deserialize(void*);

static struct custom_operations hdf5_ptr_ops = {
  "hdf5.struct.ptr",
  hdf5_ptr_finalize,
  custom_compare_default,
  custom_hash_default,
  hdf5_ptr_serialize,
  hdf5_ptr_deserialize,
  custom_compare_ext_default,
#ifdef custom_fixed_length_default
  custom_fixed_length_default,
#endif
};

struct hdf5_caml_ptr {
  void *ptr;
  struct hdf5_caml_mem *mem;
  value pos;
};

#define Ptr_val(v) ((struct hdf5_caml_ptr*) Data_custom_val(v))

value hdf5_caml_struct_ptr_create(value mem_v, value pos_v)
{
  CAMLparam2(mem_v, pos_v);
  CAMLlocal1(v);
  struct hdf5_caml_mem *mem;
  struct hdf5_caml_ptr *ptr;

  mem = Mem_val(mem_v);
  v = caml_alloc_custom(&hdf5_ptr_ops, sizeof(struct hdf5_caml_ptr), 0, 1);
  ptr = Ptr_val(v);
  ptr->ptr = (char*) mem->data + Long_val(pos_v) * Long_val(mem->size);
  ptr->mem = mem;
  ptr->pos = pos_v;
  mem->refcount++;

  CAMLreturn(v);
}

void hdf5_ptr_finalize(value v)
{
  mem_free(Ptr_val(v)->mem);
}

void hdf5_ptr_serialize(value v, uintnat *wsize_32, uintnat *wsize_64)
{
  struct hdf5_caml_ptr *ptr;

  ptr = Ptr_val(v);
  caml_serialize_int_8(Long_val(ptr->pos));
  serialize_mem(ptr->mem);
  *wsize_32 = sizeof(struct hdf5_caml_ptr);
  *wsize_64 = sizeof(struct hdf5_caml_ptr);
}

uintnat hdf5_ptr_deserialize(void *dst)
{
  struct hdf5_caml_mem *mem;
  struct hdf5_caml_ptr *ptr;
  unsigned long pos;

  ptr = dst;
  pos = caml_deserialize_sint_8();
  mem = deserialize_mem();
  mem->refcount++;
  ptr->ptr = (char*) mem->data + pos * Long_val(mem->size);
  ptr->mem = mem;
  ptr->pos = Val_long(pos);

  return sizeof(struct hdf5_caml_ptr);
}

value hdf5_caml_struct_ptr_get_bigstring(
  void *ptr, struct hdf5_caml_mem *mem, long bo, value column_v, value row_v)
{
  CAMLparam2(column_v, row_v);
  CAMLlocal1(res_v);
  size_t column, row, dim;
  struct caml_ba_proxy **proxies, *proxy;
  char *data;

  column = Long_val(column_v);
  row    = Long_val(row_v);
  data   = *((char**) ((char*) ptr + bo - 1));

  proxies = mem->fields[column].proxies;
  if (proxies == NULL)
  {
    proxies = calloc(Long_val(mem->capacity), sizeof(struct caml_ba_proxy*));
    if (proxies == NULL)
      caml_hdf5_raise_out_of_memory();
    mem->fields[column].proxies = proxies;
  }
  proxy = proxies[row];
  if (proxy == NULL)
  {
    proxy = malloc(sizeof(struct caml_ba_proxy));
    if (proxy == NULL)
      caml_hdf5_raise_out_of_memory();
    proxies[row] = proxy;
    proxy->refcount = 2;
    proxy->data = data;
    dim = strnlen(data, MAX_BIGSTRING_LEN);
    if (dim == MAX_BIGSTRING_LEN && data[dim] != '\0')
      caml_failwith("The given C_string bigger than 1 Gb");
    proxy->size = dim;
  }
  else
  {
    ++proxy->refcount;
    dim = proxy->size;
  }
  res_v =
    caml_ba_alloc_dims(CAML_BA_CHAR | CAML_BA_C_LAYOUT | CAML_BA_MANAGED, 1, data, dim);
  Caml_ba_array_val(res_v)->proxy = proxy;
  CAMLreturn(res_v);
}

void hdf5_caml_struct_ptr_set_bigstring(
  void *ptr, struct hdf5_caml_mem *mem, long bo, value column_v, value row_v,
  value bigstring_v)
{
  CAMLparam2(column_v, row_v);
  size_t column, row;
  struct caml_ba_array *bigstring;
  struct caml_ba_proxy **proxies, *proxy;
  void *data;

  column    = Long_val(column_v);
  row       = Long_val(row_v);
  data      = (char*) ptr + bo - 1;
  bigstring = Caml_ba_array_val(bigstring_v);

  proxies = mem->fields[column].proxies;
  if (proxies == NULL)
  {
    proxies = calloc(Long_val(mem->capacity), sizeof(struct caml_ba_proxy*));
    if (proxies == NULL)
      caml_hdf5_raise_out_of_memory();
    mem->fields[column].proxies = proxies;
  }
  proxy = proxies[row];
  if (proxy != NULL && --proxy->refcount == 0)
  {
    free(*((void**) data));
    free(proxy);
  }
  proxy = bigstring->proxy;
  if (proxy == NULL)
  {
    proxy = malloc(sizeof(struct caml_ba_proxy));
    if (proxy == NULL)
      caml_hdf5_raise_out_of_memory();
    proxy->refcount = 2;
    proxy->data = bigstring->data;
    proxy->size = bigstring->dim[0];
    bigstring->proxy = proxy;
  }
  else
    ++proxy->refcount;

  proxies[row] = proxy;
  *((void**) data) = proxy->data;

  CAMLreturn0;
}

void hdf5_caml_struct_ptr_set_bigstring_bytecode(value *argv, int argn)
{
  assert(argn == 6);
  hdf5_caml_struct_ptr_set_bigstring(
    (void*) argv[0], (struct hdf5_caml_mem*) argv[1], (long) argv[2], argv[3], argv[4],
    argv[5]);
}

value hdf5_caml_struct_ptr_get_array(
  void *ptr, struct hdf5_caml_mem *mem, long bo, value column_v, value row_v)
{
  CAMLparam2(column_v, row_v);
  CAMLlocal1(res_v);
  size_t column, row;
  struct caml_ba_proxy **proxies, *proxy;
  hvl_t *hvl;

  column = Long_val(column_v);
  row    = Long_val(row_v);
  hvl    = (hvl_t*) ((char*) ptr + bo - 1);

  proxies = mem->fields[column].proxies;
  if (proxies == NULL)
  {
    proxies = calloc(Long_val(mem->capacity), sizeof(struct caml_ba_proxy*));
    if (proxies == NULL)
      caml_hdf5_raise_out_of_memory();
    mem->fields[column].proxies = proxies;
  }
  proxy = proxies[row];
  if (proxy == NULL)
  {
    proxy = malloc(sizeof(struct caml_ba_proxy));
    if (proxy == NULL)
      caml_hdf5_raise_out_of_memory();
    proxies[row] = proxy;
    proxy->refcount = 2;
    proxy->data = hvl->p;
    proxy->size = hvl->len * mem->fields[column].element_size;
  }
  else
    ++proxy->refcount;
  res_v =
    caml_ba_alloc_dims(
      CAML_BA_FLOAT64 | CAML_BA_C_LAYOUT | CAML_BA_MANAGED, 1, hvl->p, hvl->len);
  Caml_ba_array_val(res_v)->proxy = proxy;
  CAMLreturn(res_v);
}

void hdf5_caml_struct_ptr_set_array(
  void *ptr, struct hdf5_caml_mem *mem, long bo, value column_v, value row_v,
  value bigarray_v)
{
  CAMLparam2(column_v, row_v);
  size_t column, row, len;
  struct caml_ba_array *bigarray;
  struct caml_ba_proxy **proxies, *proxy;
  hvl_t *hvl;
  void *data;

  column   = Long_val(column_v);
  row      = Long_val(row_v);
  hvl      = (hvl_t*) ((char*) ptr + bo - 1);
  bigarray = Caml_ba_array_val(bigarray_v);
  data     = bigarray->data;
  len      = bigarray->dim[0];

  proxies = mem->fields[column].proxies;
  if (proxies == NULL)
  {
    proxies = calloc(Long_val(mem->capacity), sizeof(struct caml_ba_proxy*));
    if (proxies == NULL)
      caml_hdf5_raise_out_of_memory();
    mem->fields[column].proxies = proxies;
  }
  proxy = proxies[row];
  if (proxy != NULL && --proxy->refcount == 0)
  {
    free(proxy->data);
    free(proxy);
  }
  proxy = bigarray->proxy;
  if (proxy == NULL)
  {
    proxy = malloc(sizeof(struct caml_ba_proxy));
    if (proxy == NULL)
      caml_hdf5_raise_out_of_memory();
    proxy->refcount = 2;
    proxy->data = data;
    proxy->size = len * mem->fields[column].element_size;
    bigarray->proxy = proxy;
  }
  else
    ++proxy->refcount;

  proxies[row] = proxy;
  hvl->p   = data;
  hvl->len = len;

  CAMLreturn0;
}

void hdf5_caml_struct_ptr_set_array_bytecode(value *argv, int argn)
{
  assert(argn == 6);
  hdf5_caml_struct_ptr_set_array(
    (void*) argv[0], (struct hdf5_caml_mem*) argv[1], (long) argv[2], argv[3], argv[4],
    argv[5]);
}

herr_t c_s1_to_c_s1_variable(hid_t src_id, hid_t dst_id, H5T_cdata_t *cdata,
  size_t nelmts, size_t buf_stride, size_t bkg_stride, void *buf, void *bkg,
  hid_t dset_xfer_plist)
{
  void *src_buf, *dst_buf;
  size_t maxlen, len, i;
  char *s;

  (void) bkg_stride;
  (void) bkg;
  (void) dset_xfer_plist;
  switch (cdata->command)
  {
    case H5T_CONV_INIT:
      return H5Tis_variable_str(dst_id) ? 0 : -1;
    case H5T_CONV_CONV:
      maxlen = H5Tget_size(src_id);
      if (buf_stride <= sizeof(char*))
      {
        src_buf = (char*) buf + nelmts * buf_stride;
        dst_buf = (char*) buf + nelmts * sizeof(char*);
        for (i = 0; i < nelmts; i++)
        {
          src_buf = (char*) src_buf - buf_stride;
          dst_buf = (char*) dst_buf - buf_stride;
          len = strnlen(src_buf, maxlen);
          s = malloc(len + 1);
          memcpy(s, src_buf, len);
          s[len] = '\0';
          *((char**) dst_buf) = s;
        }
      }
      else
      {
        src_buf = buf;
        dst_buf = buf;
        for (i = 0; i < nelmts; i++)
        {
          len = strnlen(src_buf, maxlen);
          s = malloc(len + 1);
          memcpy(s, src_buf, len);
          s[len] = '\0';
          *((char**) dst_buf) = s;
          src_buf = (char*) src_buf + buf_stride;
          dst_buf = (char*) dst_buf + buf_stride;
        }
      }
      return 0;
    case H5T_CONV_FREE: return 0;
    default: return -1;
  }
}

void register_conversion_functions(void)
{
  hid_t datatype;
  datatype = H5Tcopy(H5T_C_S1);
  H5Tset_size(datatype, H5T_VARIABLE);
  H5Tregister(H5T_PERS_SOFT, "c_s1_to_c_s1_variable", H5T_C_S1, datatype,
    c_s1_to_c_s1_variable);
  H5Tclose(datatype);
}

static bool initialized = false;

void hdf5_caml_struct_initialize(void)
{
  if (!initialized)
  {
    initialized = true;
    caml_register_custom_operations(&hdf5_mem_ops);
    caml_register_custom_operations(&hdf5_ptr_ops);
    register_conversion_functions();
  }
}

void hdf5_caml_struct_reset_serialize(void)
{
  hdf5_caml_struct_initialize();
  serialize_count++;
  serialize_id = 0;
}

void hdf5_caml_struct_reset_deserialize(void)
{
  CAMLparam0();
  hdf5_caml_struct_initialize();
  if (mems_capacity == 0)
  {
    mems_capacity = 256;
    mems = malloc(mems_capacity * sizeof(struct hdf5_caml_mem*));
    if (mems == NULL)
      caml_hdf5_raise_out_of_memory();
  }
  deserialize_id = 0;
  CAMLreturn0;
}
