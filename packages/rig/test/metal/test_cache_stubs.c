/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* What test_cache asks of the system: the process's bundle, the user's
   cache directory, a Metal device made outside rig, and a file's lock as
   Metal's shader cache takes it. Objective-C on macOS; elsewhere no Metal
   test runs and nothing here is called. */

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/mlvalues.h>

#ifdef __APPLE__

#include <limits.h>
#include <sys/file.h>
#include <unistd.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

/* The main bundle's identifier, "" for none. */
value rig_metal_cache_test_bundle(value unit) {
  (void)unit;
  @autoreleasepool {
    NSString *id = NSBundle.mainBundle.bundleIdentifier;
    return caml_copy_string(id ? id.UTF8String : "");
  }
}

/* The user's cache directory, which ends with a slash, "" if the system
   gives none. */
value rig_metal_cache_test_user_cache(value unit) {
  (void)unit;
  char dir[PATH_MAX];
  size_t n = confstr(_CS_DARWIN_USER_CACHE_DIR, dir, sizeof dir);
  return caml_copy_string(n > 0 && n <= sizeof dir ? dir : "");
}

/* Makes a Metal device, which the process keeps until it exits: Metal has
   then made a device before rig opens one. */
value rig_metal_cache_test_device(value unit) {
  (void)unit;
  static id<MTLDevice> device;
  if (device == nil) device = MTLCreateSystemDefaultDevice();
  return Val_bool(device != nil);
}

/* Whether the process took an exclusive lock of [v_fd]'s file with flock,
   without waiting, as Metal's shader cache locks its files. */
value rig_metal_cache_test_flock(value v_fd) {
  return Val_bool(flock(Int_val(v_fd), LOCK_EX | LOCK_NB) == 0);
}

#else

value rig_metal_cache_test_bundle(value unit) {
  (void)unit;
  return caml_copy_string("");
}

value rig_metal_cache_test_user_cache(value unit) {
  (void)unit;
  return caml_copy_string("");
}

value rig_metal_cache_test_device(value unit) {
  (void)unit;
  return Val_false;
}

value rig_metal_cache_test_flock(value v_fd) {
  (void)v_fd;
  return Val_false;
}

#endif
