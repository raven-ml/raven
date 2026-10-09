/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The suite's bundle identifier, so that Metal keeps the pipelines it
   compiles in a cache of the suite's own, as the Metal suite's does
   (test/metal/rig_metal_test_bundle.c). */

#define _GNU_SOURCE

#if defined(__APPLE__)
__attribute__((used, section("__TEXT,__info_plist")))
static const char rig_conformance_info_plist[] =
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" "
    "\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
    "<plist version=\"1.0\">\n"
    "<dict>\n"
    "  <key>CFBundleIdentifier</key>\n"
    "  <string>rig-conformance-test</string>\n"
    "</dict>\n"
    "</plist>\n";
#endif
