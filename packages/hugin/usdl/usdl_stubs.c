/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
#include <string.h>

#ifdef __APPLE__
#include <SDL2/SDL.h>
#else
#include <SDL.h>
#endif

/* Window */

#define Window_val(v) (*(SDL_Window **)Data_custom_val(v))

static void finalize_window(value v)
{
  SDL_Window *w = Window_val(v);
  if (w != NULL) { SDL_DestroyWindow(w); Window_val(v) = NULL; }
}

static struct custom_operations window_ops = {
  .identifier  = "usdl.window",
  .finalize    = finalize_window,
  .compare     = custom_compare_default,
  .hash        = custom_hash_default,
  .serialize   = custom_serialize_default,
  .deserialize = custom_deserialize_default,
  .compare_ext = custom_compare_ext_default,
};

static value alloc_window(SDL_Window *w)
{
  value v = caml_alloc_custom(&window_ops, sizeof(SDL_Window *), 0, 1);
  Window_val(v) = w;
  return v;
}

/* Renderer */

#define Renderer_val(v) (*(SDL_Renderer **)Data_custom_val(v))

static void finalize_renderer(value v)
{
  SDL_Renderer *r = Renderer_val(v);
  if (r != NULL) { SDL_DestroyRenderer(r); Renderer_val(v) = NULL; }
}

static struct custom_operations renderer_ops = {
  .identifier  = "usdl.renderer",
  .finalize    = finalize_renderer,
  .compare     = custom_compare_default,
  .hash        = custom_hash_default,
  .serialize   = custom_serialize_default,
  .deserialize = custom_deserialize_default,
  .compare_ext = custom_compare_ext_default,
};

static value alloc_renderer(SDL_Renderer *r)
{
  value v = caml_alloc_custom(&renderer_ops, sizeof(SDL_Renderer *), 0, 1);
  Renderer_val(v) = r;
  return v;
}

/* Surface */

#define Surface_val(v) (*(SDL_Surface **)Data_custom_val(v))

static void finalize_surface(value v)
{
  SDL_Surface *s = Surface_val(v);
  if (s != NULL) { SDL_FreeSurface(s); Surface_val(v) = NULL; }
}

static struct custom_operations surface_ops = {
  .identifier  = "usdl.surface",
  .finalize    = finalize_surface,
  .compare     = custom_compare_default,
  .hash        = custom_hash_default,
  .serialize   = custom_serialize_default,
  .deserialize = custom_deserialize_default,
  .compare_ext = custom_compare_ext_default,
};

static value alloc_surface(SDL_Surface *s)
{
  value v = caml_alloc_custom(&surface_ops, sizeof(SDL_Surface *), 0, 1);
  Surface_val(v) = s;
  return v;
}

/* Texture */

#define Texture_val(v) (*(SDL_Texture **)Data_custom_val(v))

static void finalize_texture(value v)
{
  SDL_Texture *t = Texture_val(v);
  if (t != NULL) { SDL_DestroyTexture(t); Texture_val(v) = NULL; }
}

static struct custom_operations texture_ops = {
  .identifier  = "usdl.texture",
  .finalize    = finalize_texture,
  .compare     = custom_compare_default,
  .hash        = custom_hash_default,
  .serialize   = custom_serialize_default,
  .deserialize = custom_deserialize_default,
  .compare_ext = custom_compare_ext_default,
};

static value alloc_texture(SDL_Texture *t)
{
  value v = caml_alloc_custom(&texture_ops, sizeof(SDL_Texture *), 0, 1);
  Texture_val(v) = t;
  return v;
}

/* Event — stored inline in custom block, no heap allocation */

#define Event_val(v) ((SDL_Event *)Data_custom_val(v))

static struct custom_operations event_ops = {
  .identifier  = "usdl.event",
  .finalize    = custom_finalize_default,
  .compare     = custom_compare_default,
  .hash        = custom_hash_default,
  .serialize   = custom_serialize_default,
  .deserialize = custom_deserialize_default,
  .compare_ext = custom_compare_ext_default,
};

/* Init / quit */

CAMLprim value caml_usdl_init(value vunit)
{
  CAMLparam1(vunit);
  if (SDL_Init(SDL_INIT_VIDEO) < 0)
    caml_failwith(SDL_GetError());
  SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "1");
  CAMLreturn(Val_unit);
}

CAMLprim value caml_usdl_quit(value vunit)
{
  CAMLparam1(vunit);
  SDL_Quit();
  CAMLreturn(Val_unit);
}

/* Window */

CAMLprim value caml_usdl_window_create(value vtitle, value vw, value vh)
{
  CAMLparam3(vtitle, vw, vh);
  Uint32 flags = SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE |
                 SDL_WINDOW_ALLOW_HIGHDPI;
  SDL_Window *win = SDL_CreateWindow(
    String_val(vtitle),
    SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
    Int_val(vw), Int_val(vh), flags);
  if (win == NULL) caml_failwith(SDL_GetError());
  CAMLreturn(alloc_window(win));
}

CAMLprim value caml_usdl_window_destroy(value vwin)
{
  CAMLparam1(vwin);
  SDL_Window *w = Window_val(vwin);
  if (w != NULL) { SDL_DestroyWindow(w); Window_val(vwin) = NULL; }
  CAMLreturn(Val_unit);
}

/* Renderer */

CAMLprim value caml_usdl_renderer_create(value vwin)
{
  CAMLparam1(vwin);
  SDL_Window *win = Window_val(vwin);
  if (win == NULL) caml_invalid_argument("Usdl.Renderer.create: destroyed window");
  Uint32 flags = SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC;
  SDL_Renderer *ren = SDL_CreateRenderer(win, -1, flags);
  if (ren == NULL) caml_failwith(SDL_GetError());
  CAMLreturn(alloc_renderer(ren));
}

CAMLprim value caml_usdl_renderer_output_size(value vren)
{
  CAMLparam1(vren);
  CAMLlocal1(result);
  SDL_Renderer *ren = Renderer_val(vren);
  if (ren == NULL) caml_invalid_argument("Usdl.Renderer.output_size: destroyed renderer");
  int w, h;
  if (SDL_GetRendererOutputSize(ren, &w, &h) < 0)
    caml_failwith(SDL_GetError());
  result = caml_alloc_tuple(2);
  Store_field(result, 0, Val_int(w));
  Store_field(result, 1, Val_int(h));
  CAMLreturn(result);
}

CAMLprim value caml_usdl_renderer_clear(value vren)
{
  CAMLparam1(vren);
  SDL_Renderer *ren = Renderer_val(vren);
  if (ren == NULL) caml_invalid_argument("Usdl.Renderer.clear: destroyed renderer");
  if (SDL_RenderClear(ren) < 0) caml_failwith(SDL_GetError());
  CAMLreturn(Val_unit);
}

CAMLprim value caml_usdl_renderer_copy(value vren, value vtex)
{
  CAMLparam2(vren, vtex);
  SDL_Renderer *ren = Renderer_val(vren);
  SDL_Texture *tex = Texture_val(vtex);
  if (ren == NULL || tex == NULL)
    caml_invalid_argument("Usdl.Renderer.copy: destroyed handle");
  if (SDL_RenderCopy(ren, tex, NULL, NULL) < 0)
    caml_failwith(SDL_GetError());
  CAMLreturn(Val_unit);
}

CAMLprim value caml_usdl_renderer_present(value vren)
{
  CAMLparam1(vren);
  SDL_Renderer *ren = Renderer_val(vren);
  if (ren == NULL) caml_invalid_argument("Usdl.Renderer.present: destroyed renderer");
  SDL_RenderPresent(ren);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_usdl_renderer_destroy(value vren)
{
  CAMLparam1(vren);
  SDL_Renderer *r = Renderer_val(vren);
  if (r != NULL) { SDL_DestroyRenderer(r); Renderer_val(vren) = NULL; }
  CAMLreturn(Val_unit);
}

/* Surface */

CAMLprim value caml_usdl_surface_create_argb8888(value vw, value vh)
{
  CAMLparam2(vw, vh);
  SDL_Surface *s = SDL_CreateRGBSurfaceWithFormat(
    0, Int_val(vw), Int_val(vh), 32, SDL_PIXELFORMAT_ARGB8888);
  if (s == NULL) caml_failwith(SDL_GetError());
  CAMLreturn(alloc_surface(s));
}

CAMLprim value caml_usdl_surface_destroy(value vsurf)
{
  CAMLparam1(vsurf);
  SDL_Surface *s = Surface_val(vsurf);
  if (s != NULL) { SDL_FreeSurface(s); Surface_val(vsurf) = NULL; }
  CAMLreturn(Val_unit);
}

CAMLprim value caml_usdl_surface_pitch(value vsurf)
{
  CAMLparam1(vsurf);
  SDL_Surface *s = Surface_val(vsurf);
  if (s == NULL) caml_invalid_argument("Usdl.Surface.pitch: destroyed surface");
  CAMLreturn(Val_int(s->pitch));
}

CAMLprim value caml_usdl_surface_pixels(value vsurf)
{
  CAMLparam1(vsurf);
  SDL_Surface *s = Surface_val(vsurf);
  if (s == NULL) caml_invalid_argument("Usdl.Surface.pixels: destroyed surface");
  if (s->pixels == NULL) caml_failwith("Usdl.Surface.pixels: NULL pixels");
  CAMLreturn(caml_ba_alloc_dims(
    CAML_BA_UINT8 | CAML_BA_C_LAYOUT | CAML_BA_EXTERNAL,
    1, s->pixels, (intnat)s->h * s->pitch));
}

/* Texture */

CAMLprim value caml_usdl_texture_of_surface(value vren, value vsurf)
{
  CAMLparam2(vren, vsurf);
  SDL_Renderer *ren = Renderer_val(vren);
  SDL_Surface *s = Surface_val(vsurf);
  if (ren == NULL || s == NULL)
    caml_invalid_argument("Usdl.Texture.of_surface: destroyed handle");
  SDL_Texture *tex = SDL_CreateTextureFromSurface(ren, s);
  if (tex == NULL) caml_failwith(SDL_GetError());
  CAMLreturn(alloc_texture(tex));
}

CAMLprim value caml_usdl_texture_destroy(value vtex)
{
  CAMLparam1(vtex);
  SDL_Texture *t = Texture_val(vtex);
  if (t != NULL) { SDL_DestroyTexture(t); Texture_val(vtex) = NULL; }
  CAMLreturn(Val_unit);
}

/* Event */

CAMLprim value caml_usdl_event_create(value vunit)
{
  CAMLparam1(vunit);
  value v = caml_alloc_custom(&event_ops, sizeof(SDL_Event), 0, 1);
  memset(Event_val(v), 0, sizeof(SDL_Event));
  CAMLreturn(v);
}

CAMLprim value caml_usdl_event_wait(value vev)
{
  CAMLparam1(vev);
  SDL_Event ev;
  caml_release_runtime_system();
  int ret = SDL_WaitEvent(&ev);
  caml_acquire_runtime_system();
  if (ret == 1) memcpy(Event_val(vev), &ev, sizeof(SDL_Event));
  CAMLreturn(Val_bool(ret == 1));
}

CAMLprim value caml_usdl_event_type(value vev)
{
  return Val_int(Event_val(vev)->type);
}

CAMLprim value caml_usdl_event_window_id(value vev)
{
  SDL_Event *ev = Event_val(vev);
  if (ev->type == SDL_WINDOWEVENT)
    return Val_int(ev->window.event);
  return Val_int(-1);
}

CAMLprim value caml_usdl_event_keycode(value vev)
{
  SDL_Event *ev = Event_val(vev);
  if (ev->type == SDL_KEYDOWN || ev->type == SDL_KEYUP)
    return Val_int(ev->key.keysym.sym);
  return Val_int(-1);
}
