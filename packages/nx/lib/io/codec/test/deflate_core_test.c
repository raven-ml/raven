/*--------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  --------------------------------------------------------------------------*/

#include "nx_io_codec.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *message) {
  fprintf(stderr, "deflate_core_test: %s\n", message);
  exit(1);
}

static uint32_t crc32(const uint8_t *data, size_t len) {
  uint32_t crc = 0xffffffffu;
  for (size_t i = 0; i < len; i++) {
    crc ^= data[i];
    for (unsigned bit = 0; bit < 8; bit++)
      crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
  }
  return crc ^ 0xffffffffu;
}

static void round_trip(const uint8_t *data, size_t len, int dynamic) {
  uint8_t *compressed = NULL;
  uint32_t encoded_crc = 0;
  nx_io_result encoded =
      nx_io_deflate_raw(NULL, 0, data, len, -1, &compressed, &encoded_crc);
  if (encoded.status != NX_IO_OK)
    fail("encoder rejected a valid input");
  if (encoded.input_offset != len || encoded.output_size == 0)
    fail("encoder returned inconsistent sizes");
  if (encoded_crc != crc32(data, len))
    fail("encoder returned an incorrect CRC-32");
  if (dynamic && ((compressed[0] >> 1) & 3u) != 2u)
    fail("compressible input did not select a dynamic Huffman block");

  uint8_t *decoded = malloc(len == 0 ? 1 : len);
  if (decoded == NULL)
    fail("allocation failed");
  uint32_t decoded_crc = 0;
  nx_io_result result = nx_io_inflate_raw(compressed, encoded.output_size, 0,
                                          decoded, len, 0, -1, &decoded_crc);
  if (result.status != NX_IO_OK || result.output_size != len ||
      result.input_offset != encoded.output_size)
    fail("decoder rejected encoder output");
  if (decoded_crc != encoded_crc || memcmp(data, decoded, len) != 0)
    fail("round trip changed the input");

  if (encoded.output_size > 1) {
    result = nx_io_inflate_raw(compressed, encoded.output_size - 1, 0, decoded,
                               len, 0, -1, &decoded_crc);
    if (result.status == NX_IO_OK)
      fail("decoder accepted a truncated stream");
  }
  free(decoded);
  free(compressed);
}

static uint32_t random_word(uint32_t *state) {
  uint32_t value = *state;
  value ^= value << 13;
  value ^= value >> 17;
  value ^= value << 5;
  *state = value;
  return value;
}

static void malformed_inputs(void) {
  uint8_t input[512];
  uint8_t output[4096];
  uint32_t state = 0x4e58494fu;
  uint32_t crc;
  for (unsigned iteration = 0; iteration < 20000; iteration++) {
    size_t len = random_word(&state) % sizeof(input);
    for (size_t i = 0; i < len; i++)
      input[i] = (uint8_t)random_word(&state);
    (void)nx_io_inflate_raw(input, len, 0, output, sizeof(output), 0, -1, &crc);
  }
}

int main(void) {
  static const uint8_t empty[1] = {0};
  round_trip(empty, 0, 0);

  uint8_t repeated[70000];
  memset(repeated, 0x5a, sizeof(repeated));
  round_trip(repeated, sizeof(repeated), 1);

  uint8_t structured[3 * 65535 + 17];
  for (size_t i = 0; i < sizeof(structured); i++)
    structured[i] = (uint8_t)((i * 17 + i / 97) & 0xffu);
  round_trip(structured, sizeof(structured), 1);

  uint8_t random[2 * 65535 + 1];
  uint32_t state = 0x52415645u;
  for (size_t i = 0; i < sizeof(random); i++)
    random[i] = (uint8_t)random_word(&state);
  round_trip(random, sizeof(random), 0);

  malformed_inputs();
  return 0;
}
