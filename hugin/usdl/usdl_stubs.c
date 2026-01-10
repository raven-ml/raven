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

// It's crucial to include SDL.h *after* stdint.h which might be included by
// caml headers
#ifdef __APPLE__
#include <SDL2/SDL.h>
#else
#include <SDL.h>
#endif
#include <stdint.h>  // Ensure stdint types are available
#include <stdio.h>   // For printf debugging if needed

// --- Custom Block Helpers for SDL Pointers ---

// Macro to get the SDL_Window* back (with null check)
#define Window_val(v) (*((SDL_Window**)Data_custom_val(v)))

static void finalize_sdl_window(value v) {
  /* The window pointer lives in the custom block payload. Using Field here
     would read the custom_ops pointer instead of the SDL_Window*, so always
     go through Window_val/Data_custom_val. */
  SDL_Window* win = Window_val(v);
  if (win != NULL) {
    SDL_DestroyWindow(win);
    *((SDL_Window**)Data_custom_val(v)) = NULL;
  }
}

static struct custom_operations sdl_window_ops = {
    "mini_sdl.window",          finalize_sdl_window,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default};

// Helper to wrap an SDL_Window*
static value Val_sdl_window(SDL_Window* win) {
  if (win == NULL)
    return caml_copy_nativeint(
        0);  // Or maybe return None? Let's stick to Option for creation
  value v = caml_alloc_custom(&sdl_window_ops, sizeof(SDL_Window*), 0, 1);
  memcpy(Data_custom_val(v), &win, sizeof(SDL_Window*));
  return v;
}

// --- Similar setup for Renderer ---
static void finalize_sdl_renderer(value v) {
  SDL_Renderer* ren = *((SDL_Renderer**)Data_custom_val(v));
  if (ren != NULL) {
    // printf("Finalizing SDL_Renderer %p\n", ren);
    SDL_DestroyRenderer(ren);
  }
}
static struct custom_operations sdl_renderer_ops = {
    "mini_sdl.renderer", finalize_sdl_renderer, /* rest default */};
static value Val_sdl_renderer(SDL_Renderer* ren) {
  if (ren == NULL) return caml_copy_nativeint(0);
  value v = caml_alloc_custom(&sdl_renderer_ops, sizeof(SDL_Renderer*), 0, 1);
  memcpy(Data_custom_val(v), &ren, sizeof(SDL_Renderer*));
  return v;
}
#define Renderer_val(v) (*((SDL_Renderer**)Data_custom_val(v)))

// --- Similar setup for Surface ---
static void finalize_sdl_surface(value v) {
  SDL_Surface* surf = *((SDL_Surface**)Data_custom_val(v));
  if (surf != NULL) {
    // printf("Finalizing SDL_Surface %p\n", surf);
    SDL_FreeSurface(surf);
  }
}
static struct custom_operations sdl_surface_ops = {
    "mini_sdl.surface", finalize_sdl_surface, /* rest default */};
static value Val_sdl_surface(SDL_Surface* surf) {
  if (surf == NULL) return caml_copy_nativeint(0);
  value v = caml_alloc_custom(&sdl_surface_ops, sizeof(SDL_Surface*), 0, 1);
  memcpy(Data_custom_val(v), &surf, sizeof(SDL_Surface*));
  return v;
}
#define Surface_val(v) (*((SDL_Surface**)Data_custom_val(v)))

// --- Similar setup for Texture ---
static void finalize_sdl_texture(value v) {
  SDL_Texture* tex = *((SDL_Texture**)Data_custom_val(v));
  if (tex != NULL) {
    // printf("Finalizing SDL_Texture %p\n", tex);
    SDL_DestroyTexture(tex);
  }
}
static struct custom_operations sdl_texture_ops = {
    "mini_sdl.texture", finalize_sdl_texture, /* rest default */};
static value Val_sdl_texture(SDL_Texture* tex) {
  if (tex == NULL) return caml_copy_nativeint(0);
  value v = caml_alloc_custom(&sdl_texture_ops, sizeof(SDL_Texture*), 0, 1);
  memcpy(Data_custom_val(v), &tex, sizeof(SDL_Texture*));
  return v;
}
#define Texture_val(v) (*((SDL_Texture**)Data_custom_val(v)))

// --- Event Storage ---
// We store a pointer to an SDL_Event struct allocated on the C heap.
// The OCaml 'Event.t' holds this pointer.
static void finalize_sdl_event(value v) {
  SDL_Event* event_ptr = *((SDL_Event**)Data_custom_val(v));
  if (event_ptr != NULL) {
    // printf("Finalizing SDL_Event storage %p\n", event_ptr);
    free(event_ptr);  // Free the allocated memory
  }
}
static struct custom_operations sdl_event_ops = {
    "mini_sdl.event", finalize_sdl_event, /* rest default */};

// Allocates memory to hold one SDL_Event struct
CAMLprim value caml_sdl_alloc_event_storage(value unit) {
  CAMLparam1(unit);
  SDL_Event* event_ptr = malloc(sizeof(SDL_Event));
  if (event_ptr == NULL) {
    caml_failwith("Failed to allocate memory for SDL_Event");
  }
  // No need to initialize it here, SDL_WaitEvent will fill it.
  value v = caml_alloc_custom(&sdl_event_ops, sizeof(SDL_Event*), 0, 1);
  memcpy(Data_custom_val(v), &event_ptr, sizeof(SDL_Event*));
  CAMLreturn(v);
}
#define Event_val(v) (*((SDL_Event**)Data_custom_val(v)))

// --- SDL Function Stubs ---

CAMLprim value caml_sdl_init(value flags) {
  CAMLparam1(flags);
  // Clear error string before potentially failing call
  SDL_ClearError();
  int ret = SDL_Init(Int_val(flags));
  CAMLreturn(Val_int(ret));  // Return 0 on success, < 0 on error
}

CAMLprim value caml_sdl_quit(value unit) {
  CAMLparam1(unit);
  SDL_Quit();
  CAMLreturn(Val_unit);
}

CAMLprim value caml_sdl_get_error(value unit) {
  CAMLparam1(unit);
  CAMLreturn(caml_copy_string(SDL_GetError()));
}

CAMLprim value caml_sdl_set_hint(value name, value value_str) {
  CAMLparam2(name, value_str);
  SDL_bool ret = SDL_SetHint(String_val(name), String_val(value_str));
  CAMLreturn(Val_bool(ret == SDL_TRUE));
}

CAMLprim value caml_sdl_create_window(value title, value w, value h,
                                      value flags) {
  CAMLparam4(title, w, h, flags);
  CAMLlocal1(result);  // For the Option type

  SDL_ClearError();
  SDL_Window* win = SDL_CreateWindow(
      String_val(title),
      SDL_WINDOWPOS_UNDEFINED,  // Use undefined position
      SDL_WINDOWPOS_UNDEFINED, Int_val(w), Int_val(h), Int_val(flags));

  if (win == NULL) {
    result = Val_none;  // None
  } else {
    result = caml_alloc(1, 0);  // Some(v)
    Store_field(result, 0, Val_sdl_window(win));
  }
  CAMLreturn(result);
}

CAMLprim value caml_sdl_destroy_window(value win_val) {
  CAMLparam1(win_val);
  SDL_Window* win = Window_val(win_val);
  if (win != NULL) {
    SDL_DestroyWindow(win);
    // Nullify the pointer in the custom block to prevent double free by
    // finalizer
    *((SDL_Window**)Data_custom_val(win_val)) = NULL;
  }
  CAMLreturn(Val_unit);
}

CAMLprim value caml_sdl_create_renderer(value win_val, value flags) {
  CAMLparam2(win_val, flags);
  CAMLlocal1(result);
  SDL_Window* win = Window_val(win_val);
  if (win == NULL) caml_failwith("caml_sdl_create_renderer: Invalid window");

  SDL_ClearError();
  SDL_Renderer* ren = SDL_CreateRenderer(
      win, -1, Int_val(flags));  // Use first available driver (-1)

  if (ren == NULL) {
    result = Val_none;
  } else {
    result = caml_alloc(1, 0);  // Some(v)
    Store_field(result, 0, Val_sdl_renderer(ren));
  }
  CAMLreturn(result);
}

CAMLprim value caml_sdl_destroy_renderer(value ren_val) {
  CAMLparam1(ren_val);
  SDL_Renderer* ren = Renderer_val(ren_val);
  if (ren != NULL) {
    SDL_DestroyRenderer(ren);
    *((SDL_Renderer**)Data_custom_val(ren_val)) = NULL;
  }
  CAMLreturn(Val_unit);
}

CAMLprim value caml_sdl_get_renderer_output_size(value ren_val) {
  CAMLparam1(ren_val);
  CAMLlocal1(result);
  SDL_Renderer* ren = Renderer_val(ren_val);
  if (ren == NULL)
    caml_failwith("caml_sdl_get_renderer_output_size: Invalid renderer");

  int w, h;
  SDL_ClearError();
  if (SDL_GetRendererOutputSize(ren, &w, &h) == 0) {
    // Success: return Some((w, h))
    value pair = caml_alloc_tuple(2);
    Store_field(pair, 0, Val_int(w));
    Store_field(pair, 1, Val_int(h));
    result = caml_alloc(1, 0);  // Some tag
    Store_field(result, 0, pair);
  } else {
    // Failure: return None
    result = Val_none;
  }
  CAMLreturn(result);
}

CAMLprim value caml_sdl_render_clear(value ren_val) {
  CAMLparam1(ren_val);
  SDL_Renderer* ren = Renderer_val(ren_val);
  if (ren == NULL) caml_failwith("caml_sdl_render_clear: Invalid renderer");
  SDL_ClearError();
  CAMLreturn(Val_int(SDL_RenderClear(ren)));  // 0 on success, < 0 on error
}

CAMLprim value caml_sdl_render_copy(value ren_val, value tex_val) {
  CAMLparam2(ren_val, tex_val);
  SDL_Renderer* ren = Renderer_val(ren_val);
  SDL_Texture* tex = Texture_val(tex_val);
  if (ren == NULL || tex == NULL)
    caml_failwith("caml_sdl_render_copy: Invalid renderer or texture");
  SDL_ClearError();
  // Copy the entire texture to the entire renderer
  CAMLreturn(Val_int(
      SDL_RenderCopy(ren, tex, NULL, NULL)));  // 0 on success, < 0 on error
}

CAMLprim value caml_sdl_render_present(value ren_val) {
  CAMLparam1(ren_val);
  SDL_Renderer* ren = Renderer_val(ren_val);
  if (ren == NULL) caml_failwith("caml_sdl_render_present: Invalid renderer");
  SDL_RenderPresent(ren);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_sdl_create_rgb_surface_with_format(value w, value h,
                                                       value depth,
                                                       value format) {
  CAMLparam4(w, h, depth, format);
  CAMLlocal1(result);

  SDL_ClearError();
  // SDL_CreateRGBSurfaceWithFormat requires Uint32 format
  SDL_Surface* surf = SDL_CreateRGBSurfaceWithFormat(
      0,  // flags, usually 0
      Int_val(w), Int_val(h), Int_val(depth),
      (Uint32)Int32_val(format)  // Convert OCaml int32 to C Uint32
  );

  if (surf == NULL) {
    result = Val_none;
  } else {
    result = caml_alloc(1, 0);  // Some(v)
    Store_field(result, 0, Val_sdl_surface(surf));
  }
  CAMLreturn(result);
}

CAMLprim value caml_sdl_free_surface(value surf_val) {
  CAMLparam1(surf_val);
  SDL_Surface* surf = Surface_val(surf_val);
  if (surf != NULL) {
    SDL_FreeSurface(surf);
    *((SDL_Surface**)Data_custom_val(surf_val)) = NULL;
  }
  CAMLreturn(Val_unit);
}

CAMLprim value caml_sdl_get_surface_pitch(value surf_val) {
  CAMLparam1(surf_val);
  SDL_Surface* surf = Surface_val(surf_val);
  if (surf == NULL)
    caml_failwith("caml_sdl_get_surface_pitch: Invalid surface");
  CAMLreturn(Val_int(surf->pitch));
}

CAMLprim value caml_sdl_get_surface_pixels(value surf_val) {
  CAMLparam1(surf_val);
  SDL_Surface* surf = Surface_val(surf_val);
  if (surf == NULL)
    caml_failwith("caml_sdl_get_surface_pixels: Invalid surface");
  if (surf->pixels == NULL)
    caml_failwith("caml_sdl_get_surface_pixels: Surface has NULL pixels");

  // Calculate total size in bytes: height * pitch
  // Note: pitch is the length of a row in *bytes*.
  long size = (long)surf->h * surf->pitch;
  if (size <= 0)
    caml_failwith(
        "caml_sdl_get_surface_pixels: Invalid surface dimensions or pitch");

  // Create a Bigarray view directly onto the SDL surface pixels.
  // WARNING: The lifetime of this Bigarray is tied to the SDL_Surface.
  // The OCaml code MUST ensure the SDL_Surface is not freed while this array is
  // live. Our Cairo usage pattern (create_for_data8) should be okay as Cairo
  // copies or references it immediately, and we flush before SDL operations.
  // The dimensions reflect a flat byte array.
  intnat dims[] = {size};
  // CAML_BA_UINT8 corresponds to int8_unsigned_elt
  // CAML_BA_C_LAYOUT is standard C layout
  // CAML_BA_EXTERNAL means the data is managed elsewhere (by SDL).
  // CAML_BA_MANAGED would mean the GC manages it, which isn't true here.
  value ba =
      caml_ba_alloc_dims(CAML_BA_UINT8 | CAML_BA_C_LAYOUT | CAML_BA_EXTERNAL, 1,
                         surf->pixels, dims);

  CAMLreturn(ba);
}

CAMLprim value caml_sdl_create_texture_from_surface(value ren_val,
                                                    value surf_val) {
  CAMLparam2(ren_val, surf_val);
  CAMLlocal1(result);
  SDL_Renderer* ren = Renderer_val(ren_val);
  SDL_Surface* surf = Surface_val(surf_val);
  if (ren == NULL || surf == NULL)
    caml_failwith(
        "caml_sdl_create_texture_from_surface: Invalid renderer or surface");

  SDL_ClearError();
  SDL_Texture* tex = SDL_CreateTextureFromSurface(ren, surf);

  if (tex == NULL) {
    result = Val_none;
  } else {
    result = caml_alloc(1, 0);  // Some(v)
    Store_field(result, 0, Val_sdl_texture(tex));
  }
  CAMLreturn(result);
}

CAMLprim value caml_sdl_destroy_texture(value tex_val) {
  CAMLparam1(tex_val);
  SDL_Texture* tex = Texture_val(tex_val);
  if (tex != NULL) {
    SDL_DestroyTexture(tex);
    *((SDL_Texture**)Data_custom_val(tex_val)) = NULL;
  }
  CAMLreturn(Val_unit);
}

// --- Event Stubs ---

CAMLprim value caml_sdl_wait_event(value event_v) {
  CAMLparam1(event_v);
  SDL_Event* event_ptr = Event_val(event_v);
  if (event_ptr == NULL)
    caml_failwith("caml_sdl_wait_event: Invalid event storage");

  SDL_ClearError();
  int ret = SDL_WaitEvent(
      event_ptr);  // Fills the event structure pointed to by event_ptr

  if (ret == 1) {
    // Success, check if it was SDL_QUIT
    if (event_ptr->type == SDL_QUIT) {
      CAMLreturn(Val_int(0));  // Special code 0 for quit
    } else {
      CAMLreturn(Val_int(1));  // Normal event
    }
  } else {
    // Error occurred (SDL_WaitEvent returns 0 on error, but we map quit to 0)
    // We'll return -1 for actual errors.
    // Check if an error message is actually set. SDL_WaitEvent might return 0
    // without setting an error if event processing was interrupted.
    const char* err = SDL_GetError();
    if (err != NULL && *err != '\0') {
      CAMLreturn(Val_int(-1));  // Error
    } else {
      // Not quit, not error (maybe interrupted?) - Treat as error for
      // simplicity
      CAMLreturn(Val_int(-1));
    }
  }
}

CAMLprim value caml_sdl_get_event_type(value event_v) {
  CAMLparam1(event_v);
  SDL_Event* event_ptr = Event_val(event_v);
  if (event_ptr == NULL)
    caml_failwith("caml_sdl_get_event_type: Invalid event storage");
  CAMLreturn(Val_int(event_ptr->type));
}

// Only call this if type is SDL_WINDOWEVENT
CAMLprim value caml_sdl_get_window_event_id(value event_v) {
  CAMLparam1(event_v);
  SDL_Event* event_ptr = Event_val(event_v);
  if (event_ptr == NULL)
    caml_failwith("caml_sdl_get_window_event_id: Invalid event storage");

  // Add type check for safety, though OCaml code should ensure this
  if (event_ptr->type == SDL_WINDOWEVENT) {
    CAMLreturn(
        Val_int(event_ptr->window.event));  // The 'event' field holds the ID
  } else {
    // Return a value indicating it's not a window event, e.g., -1
    CAMLreturn(Val_int(-1));
  }
}

CAMLprim value caml_sdl_get_event_keycode(value event_v) {
  CAMLparam1(event_v);
  SDL_Event* event_ptr = Event_val(event_v);
  if (event_ptr == NULL)
    caml_failwith("caml_sdl_get_event_keycode: Invalid event storage");

  if (event_ptr->type == SDL_KEYDOWN || event_ptr->type == SDL_KEYUP) {
    CAMLreturn(Val_int(event_ptr->key.keysym.sym));
  } else {
    CAMLreturn(Val_int(-1));
  }
}
