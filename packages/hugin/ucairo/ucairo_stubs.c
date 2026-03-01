/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/callback.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include <cairo.h>
#include <cairo-pdf.h>
#include <string.h>

/* --------------------------------------------------------------------------
   Custom blocks
   -------------------------------------------------------------------------- */

/* Context (cairo_t *) */

#define Context_val(v) (*(cairo_t **)Data_custom_val(v))

static void finalize_context(value v)
{
  cairo_t *cr = Context_val(v);
  if (cr != NULL) { cairo_destroy(cr); Context_val(v) = NULL; }
}

static struct custom_operations context_ops = {
  .identifier  = "ucairo.context",
  .finalize    = finalize_context,
  .compare     = custom_compare_default,
  .hash        = custom_hash_default,
  .serialize   = custom_serialize_default,
  .deserialize = custom_deserialize_default,
  .compare_ext = custom_compare_ext_default,
};

static value alloc_context(cairo_t *cr)
{
  value v = caml_alloc_custom(&context_ops, sizeof(cairo_t *), 0, 1);
  Context_val(v) = cr;
  return v;
}

/* Surface (cairo_surface_t *) */

#define Surface_val(v) (*(cairo_surface_t **)Data_custom_val(v))

static void finalize_surface(value v)
{
  cairo_surface_t *s = Surface_val(v);
  if (s != NULL) { cairo_surface_destroy(s); Surface_val(v) = NULL; }
}

static struct custom_operations surface_ops = {
  .identifier  = "ucairo.surface",
  .finalize    = finalize_surface,
  .compare     = custom_compare_default,
  .hash        = custom_hash_default,
  .serialize   = custom_serialize_default,
  .deserialize = custom_deserialize_default,
  .compare_ext = custom_compare_ext_default,
};

static value alloc_surface(cairo_surface_t *s)
{
  value v = caml_alloc_custom(&surface_ops, sizeof(cairo_surface_t *), 0, 1);
  Surface_val(v) = s;
  return v;
}

/* --------------------------------------------------------------------------
   Helpers
   -------------------------------------------------------------------------- */

static inline cairo_t *check_context(value v, const char *fn)
{
  cairo_t *cr = Context_val(v);
  if (cr == NULL) caml_invalid_argument(fn);
  return cr;
}

static inline cairo_surface_t *check_surface(value v, const char *fn)
{
  cairo_surface_t *s = Surface_val(v);
  if (s == NULL) caml_invalid_argument(fn);
  return s;
}

/* --------------------------------------------------------------------------
   Context creation
   -------------------------------------------------------------------------- */

CAMLprim value caml_ucairo_create(value vsurf)
{
  CAMLparam1(vsurf);
  cairo_surface_t *s = check_surface(vsurf, "Ucairo.create: destroyed surface");
  cairo_t *cr = cairo_create(s);
  if (cairo_status(cr) != CAIRO_STATUS_SUCCESS) {
    const char *msg = cairo_status_to_string(cairo_status(cr));
    cairo_destroy(cr);
    caml_failwith(msg);
  }
  CAMLreturn(alloc_context(cr));
}

/* --------------------------------------------------------------------------
   State
   -------------------------------------------------------------------------- */

CAMLprim value caml_ucairo_save(value vcr)
{
  CAMLparam1(vcr);
  cairo_save(check_context(vcr, "Ucairo.save: destroyed context"));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_ucairo_restore(value vcr)
{
  CAMLparam1(vcr);
  cairo_restore(check_context(vcr, "Ucairo.restore: destroyed context"));
  CAMLreturn(Val_unit);
}

/* --------------------------------------------------------------------------
   Transformations
   -------------------------------------------------------------------------- */

CAMLprim value caml_ucairo_translate(value vcr, value vtx, value vty)
{
  CAMLparam3(vcr, vtx, vty);
  cairo_translate(check_context(vcr, "Ucairo.translate: destroyed context"),
                  Double_val(vtx), Double_val(vty));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_ucairo_scale(value vcr, value vsx, value vsy)
{
  CAMLparam3(vcr, vsx, vsy);
  cairo_scale(check_context(vcr, "Ucairo.scale: destroyed context"),
              Double_val(vsx), Double_val(vsy));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_ucairo_rotate(value vcr, value vangle)
{
  CAMLparam2(vcr, vangle);
  cairo_rotate(check_context(vcr, "Ucairo.rotate: destroyed context"),
               Double_val(vangle));
  CAMLreturn(Val_unit);
}

/* --------------------------------------------------------------------------
   Source
   -------------------------------------------------------------------------- */

CAMLprim value caml_ucairo_set_source_rgba(value vcr, value vr, value vg,
                                           value vb, value va)
{
  CAMLparam5(vcr, vr, vg, vb, va);
  cairo_set_source_rgba(check_context(vcr, "Ucairo.set_source_rgba: destroyed context"),
                        Double_val(vr), Double_val(vg),
                        Double_val(vb), Double_val(va));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_ucairo_set_source_surface(value vcr, value vsurf,
                                              value vx, value vy)
{
  CAMLparam4(vcr, vsurf, vx, vy);
  cairo_set_source_surface(
    check_context(vcr, "Ucairo.set_source_surface: destroyed context"),
    check_surface(vsurf, "Ucairo.set_source_surface: destroyed surface"),
    Double_val(vx), Double_val(vy));
  CAMLreturn(Val_unit);
}

/* --------------------------------------------------------------------------
   Stroke and fill parameters
   -------------------------------------------------------------------------- */

CAMLprim value caml_ucairo_set_line_width(value vcr, value vw)
{
  CAMLparam2(vcr, vw);
  cairo_set_line_width(check_context(vcr, "Ucairo.set_line_width: destroyed context"),
                       Double_val(vw));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_ucairo_set_line_cap(value vcr, value vcap)
{
  CAMLparam2(vcr, vcap);
  static const cairo_line_cap_t caps[] = {
    CAIRO_LINE_CAP_BUTT, CAIRO_LINE_CAP_ROUND, CAIRO_LINE_CAP_SQUARE
  };
  cairo_set_line_cap(check_context(vcr, "Ucairo.set_line_cap: destroyed context"),
                     caps[Int_val(vcap)]);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_ucairo_set_line_join(value vcr, value vjoin)
{
  CAMLparam2(vcr, vjoin);
  static const cairo_line_join_t joins[] = {
    CAIRO_LINE_JOIN_MITER, CAIRO_LINE_JOIN_ROUND, CAIRO_LINE_JOIN_BEVEL
  };
  cairo_set_line_join(check_context(vcr, "Ucairo.set_line_join: destroyed context"),
                      joins[Int_val(vjoin)]);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_ucairo_set_dash(value vcr, value varr)
{
  CAMLparam2(vcr, varr);
  cairo_t *cr = check_context(vcr, "Ucairo.set_dash: destroyed context");
  int n = Wosize_val(varr) / Double_wosize;
  if (n == 0) {
    cairo_set_dash(cr, NULL, 0, 0.0);
  } else {
    double stack_buf[64];
    double *dashes = n <= 64 ? stack_buf
                             : caml_stat_alloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
      dashes[i] = Double_field(varr, i);
    cairo_set_dash(cr, dashes, n, 0.0);
    if (dashes != stack_buf) caml_stat_free(dashes);
  }
  CAMLreturn(Val_unit);
}

CAMLprim value caml_ucairo_set_antialias(value vcr, value vaa)
{
  CAMLparam2(vcr, vaa);
  static const cairo_antialias_t aa[] = {
    CAIRO_ANTIALIAS_DEFAULT, CAIRO_ANTIALIAS_NONE,
    CAIRO_ANTIALIAS_GRAY, CAIRO_ANTIALIAS_SUBPIXEL
  };
  cairo_set_antialias(check_context(vcr, "Ucairo.set_antialias: destroyed context"),
                      aa[Int_val(vaa)]);
  CAMLreturn(Val_unit);
}

/* --------------------------------------------------------------------------
   Font
   -------------------------------------------------------------------------- */

CAMLprim value caml_ucairo_select_font_face(value vcr, value vfamily,
                                            value vweight)
{
  CAMLparam3(vcr, vfamily, vweight);
  static const cairo_font_weight_t weights[] = {
    CAIRO_FONT_WEIGHT_NORMAL, CAIRO_FONT_WEIGHT_BOLD
  };
  cairo_select_font_face(
    check_context(vcr, "Ucairo.select_font_face: destroyed context"),
    String_val(vfamily), CAIRO_FONT_SLANT_NORMAL, weights[Int_val(vweight)]);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_ucairo_set_font_size(value vcr, value vsize)
{
  CAMLparam2(vcr, vsize);
  cairo_set_font_size(check_context(vcr, "Ucairo.set_font_size: destroyed context"),
                      Double_val(vsize));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_ucairo_text_extents(value vcr, value vstr)
{
  CAMLparam2(vcr, vstr);
  CAMLlocal1(result);
  cairo_t *cr = check_context(vcr, "Ucairo.text_extents: destroyed context");
  cairo_text_extents_t ext;
  cairo_text_extents(cr, String_val(vstr), &ext);
  result = caml_alloc(6 * Double_wosize, Double_array_tag);
  Store_double_field(result, 0, ext.x_bearing);
  Store_double_field(result, 1, ext.y_bearing);
  Store_double_field(result, 2, ext.width);
  Store_double_field(result, 3, ext.height);
  Store_double_field(result, 4, ext.x_advance);
  Store_double_field(result, 5, ext.y_advance);
  CAMLreturn(result);
}

CAMLprim value caml_ucairo_show_text(value vcr, value vstr)
{
  CAMLparam2(vcr, vstr);
  cairo_show_text(check_context(vcr, "Ucairo.show_text: destroyed context"),
                  String_val(vstr));
  CAMLreturn(Val_unit);
}

/* --------------------------------------------------------------------------
   Path operations
   -------------------------------------------------------------------------- */

CAMLprim value caml_ucairo_move_to(value vcr, value vx, value vy)
{
  CAMLparam3(vcr, vx, vy);
  cairo_move_to(check_context(vcr, "Ucairo.move_to: destroyed context"),
                Double_val(vx), Double_val(vy));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_ucairo_line_to(value vcr, value vx, value vy)
{
  CAMLparam3(vcr, vx, vy);
  cairo_line_to(check_context(vcr, "Ucairo.line_to: destroyed context"),
                Double_val(vx), Double_val(vy));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_ucairo_arc_native(value vcr, value vxc, value vyc,
                                      value vr, value va1, value va2)
{
  CAMLparam5(vcr, vxc, vyc, vr, va1);
  CAMLxparam1(va2);
  cairo_arc(check_context(vcr, "Ucairo.arc: destroyed context"),
            Double_val(vxc), Double_val(vyc),
            Double_val(vr), Double_val(va1), Double_val(va2));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_ucairo_arc_bytecode(value *argv, int argc)
{
  (void)argc;
  return caml_ucairo_arc_native(argv[0], argv[1], argv[2],
                                argv[3], argv[4], argv[5]);
}

CAMLprim value caml_ucairo_rectangle(value vcr, value vx, value vy,
                                     value vw, value vh)
{
  CAMLparam5(vcr, vx, vy, vw, vh);
  cairo_rectangle(check_context(vcr, "Ucairo.rectangle: destroyed context"),
                  Double_val(vx), Double_val(vy),
                  Double_val(vw), Double_val(vh));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_ucairo_path_close(value vcr)
{
  CAMLparam1(vcr);
  cairo_close_path(check_context(vcr, "Ucairo.Path.close: destroyed context"));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_ucairo_path_clear(value vcr)
{
  CAMLparam1(vcr);
  cairo_new_path(check_context(vcr, "Ucairo.Path.clear: destroyed context"));
  CAMLreturn(Val_unit);
}

/* --------------------------------------------------------------------------
   Drawing operations
   -------------------------------------------------------------------------- */

CAMLprim value caml_ucairo_fill(value vcr)
{
  CAMLparam1(vcr);
  cairo_fill(check_context(vcr, "Ucairo.fill: destroyed context"));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_ucairo_fill_preserve(value vcr)
{
  CAMLparam1(vcr);
  cairo_fill_preserve(check_context(vcr, "Ucairo.fill_preserve: destroyed context"));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_ucairo_stroke(value vcr)
{
  CAMLparam1(vcr);
  cairo_stroke(check_context(vcr, "Ucairo.stroke: destroyed context"));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_ucairo_paint(value vcr)
{
  CAMLparam1(vcr);
  cairo_paint(check_context(vcr, "Ucairo.paint: destroyed context"));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_ucairo_clip(value vcr)
{
  CAMLparam1(vcr);
  cairo_clip(check_context(vcr, "Ucairo.clip: destroyed context"));
  CAMLreturn(Val_unit);
}

/* --------------------------------------------------------------------------
   Surface operations
   -------------------------------------------------------------------------- */

CAMLprim value caml_ucairo_surface_finish(value vsurf)
{
  CAMLparam1(vsurf);
  cairo_surface_t *s = Surface_val(vsurf);
  if (s != NULL) cairo_surface_finish(s);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_ucairo_surface_flush(value vsurf)
{
  CAMLparam1(vsurf);
  cairo_surface_flush(check_surface(vsurf, "Ucairo.Surface.flush: destroyed surface"));
  CAMLreturn(Val_unit);
}

/* --------------------------------------------------------------------------
   Image surface
   -------------------------------------------------------------------------- */

CAMLprim value caml_ucairo_image_create(value vw, value vh)
{
  CAMLparam2(vw, vh);
  cairo_surface_t *s = cairo_image_surface_create(
    CAIRO_FORMAT_ARGB32, Int_val(vw), Int_val(vh));
  if (cairo_surface_status(s) != CAIRO_STATUS_SUCCESS) {
    const char *msg = cairo_status_to_string(cairo_surface_status(s));
    cairo_surface_destroy(s);
    caml_failwith(msg);
  }
  CAMLreturn(alloc_surface(s));
}

CAMLprim value caml_ucairo_image_create_for_data8(value vdata, value vw,
                                                  value vh, value vstride)
{
  CAMLparam4(vdata, vw, vh, vstride);
  unsigned char *data = (unsigned char *)Caml_ba_data_val(vdata);
  cairo_surface_t *s = cairo_image_surface_create_for_data(
    data, CAIRO_FORMAT_ARGB32, Int_val(vw), Int_val(vh), Int_val(vstride));
  if (cairo_surface_status(s) != CAIRO_STATUS_SUCCESS) {
    const char *msg = cairo_status_to_string(cairo_surface_status(s));
    cairo_surface_destroy(s);
    caml_failwith(msg);
  }
  CAMLreturn(alloc_surface(s));
}

CAMLprim value caml_ucairo_image_stride_for_width(value vw)
{
  return Val_int(cairo_format_stride_for_width(CAIRO_FORMAT_ARGB32, Int_val(vw)));
}

/* --------------------------------------------------------------------------
   PDF surface
   -------------------------------------------------------------------------- */

CAMLprim value caml_ucairo_pdf_create(value vfilename, value vw, value vh)
{
  CAMLparam3(vfilename, vw, vh);
  cairo_surface_t *s = cairo_pdf_surface_create(
    String_val(vfilename), Double_val(vw), Double_val(vh));
  if (cairo_surface_status(s) != CAIRO_STATUS_SUCCESS) {
    const char *msg = cairo_status_to_string(cairo_surface_status(s));
    cairo_surface_destroy(s);
    caml_failwith(msg);
  }
  CAMLreturn(alloc_surface(s));
}

/* --------------------------------------------------------------------------
   PNG output
   -------------------------------------------------------------------------- */

CAMLprim value caml_ucairo_png_write(value vsurf, value vfilename)
{
  CAMLparam2(vsurf, vfilename);
  cairo_surface_t *s = check_surface(vsurf, "Ucairo.Png.write: destroyed surface");
  cairo_status_t st = cairo_surface_write_to_png(s, String_val(vfilename));
  if (st != CAIRO_STATUS_SUCCESS)
    caml_failwith(cairo_status_to_string(st));
  CAMLreturn(Val_unit);
}

static cairo_status_t
png_write_func(void *closure, const unsigned char *data, unsigned int length)
{
  CAMLparam0();
  CAMLlocal2(vstr, r);
  vstr = caml_alloc_string(length);
  memcpy(Bytes_val(vstr), data, length);
  /* closure points to CAMLparam-rooted vcallback in the caller frame;
     re-read after allocation so we get the post-GC value. */
  r = caml_callback_exn(*(value *)closure, vstr);
  if (Is_exception_result(r))
    CAMLreturnT(cairo_status_t, CAIRO_STATUS_WRITE_ERROR);
  CAMLreturnT(cairo_status_t, CAIRO_STATUS_SUCCESS);
}

CAMLprim value caml_ucairo_png_write_to_stream(value vsurf, value vcallback)
{
  CAMLparam2(vsurf, vcallback);
  cairo_surface_t *s = check_surface(vsurf,
    "Ucairo.Png.write_to_stream: destroyed surface");
  /* vcallback is rooted by CAMLparam2; we pass its address as the closure
     so the callback can retrieve it. This is safe because
     cairo_surface_write_to_png_stream calls png_write_func synchronously. */
  cairo_status_t st = cairo_surface_write_to_png_stream(
    s, png_write_func, &vcallback);
  if (st != CAIRO_STATUS_SUCCESS)
    caml_failwith(cairo_status_to_string(st));
  CAMLreturn(Val_unit);
}
