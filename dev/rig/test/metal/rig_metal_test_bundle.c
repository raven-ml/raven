/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The suite's bundle identifier, in the property list a macOS executable
   carries in its __TEXT,__info_plist section. With it, Metal keeps the
   pipelines the suite compiles in a cache of the suite's own, under the
   user's cache directory, instead of the cache every process without a
   bundle shares. That shared cache grows with every program that compiles
   Metal code, and opening it under AddressSanitizer takes minutes:
   dev/rig/doc/testing.md, "Sanitizers". */

#define _GNU_SOURCE

#if defined(__APPLE__)
__attribute__((used, section("__TEXT,__info_plist")))
static const char rig_metal_test_info_plist[] =
    "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
    "<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" "
    "\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n"
    "<plist version=\"1.0\">\n"
    "<dict>\n"
    "  <key>CFBundleIdentifier</key>\n"
    "  <string>rig-metal-test</string>\n"
    "</dict>\n"
    "</plist>\n";
#endif
