/*--------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  --------------------------------------------------------------------------*/

#define _POSIX_C_SOURCE 200809L
#define NX_IO_CODEC_NO_OCAML 1

#include "../nx_io_jpeg.c"
#include "../nx_io_png.c"

#include <stdio.h>

static void fail(const char *message) {
  fprintf(stderr, "image_core_test: %s\n", message);
  exit(1);
}

static uint8_t *encoded_file(FILE *file, size_t *len) {
  int fd = fileno(file);
  off_t end = lseek(fd, 0, SEEK_END);
  if (end <= 0 || (uintmax_t)end > SIZE_MAX)
    fail("invalid encoded file size");
  *len = (size_t)end;
  uint8_t *data = malloc(*len);
  if (data == NULL)
    fail("allocation failed");
  if (lseek(fd, 0, SEEK_SET) < 0)
    fail("seek failed");
  size_t offset = 0;
  while (offset < *len) {
    ssize_t count = read(fd, data + offset, *len - offset);
    if (count <= 0)
      fail("read failed");
    offset += (size_t)count;
  }
  return data;
}

static uint8_t *encode_png_fixture(const uint8_t *pixels, size_t width,
                                   size_t height, size_t *len) {
  FILE *file = tmpfile();
  if (file == NULL)
    fail("tmpfile failed");
  png_status status =
      encode_png(fileno(file), pixels, width * height * 3, width, height, 3);
  if (status != PNG_OK)
    fail("PNG encoder rejected valid pixels");
  uint8_t *data = encoded_file(file, len);
  fclose(file);
  return data;
}

static uint8_t *encode_jpeg_fixture(const uint8_t *pixels, size_t width,
                                    size_t height, size_t *len) {
  FILE *file = tmpfile();
  if (file == NULL)
    fail("tmpfile failed");
  jpeg_status status =
      encode_jpeg(fileno(file), pixels, width * height * 3, width, height, 3);
  if (status != JPEG_OK)
    fail("JPEG encoder rejected valid pixels");
  uint8_t *data = encoded_file(file, len);
  fclose(file);
  return data;
}

static uint32_t random_word(uint32_t *state) {
  uint32_t value = *state;
  value ^= value << 13;
  value ^= value >> 17;
  value ^= value << 5;
  *state = value;
  return value;
}

static void mutate_images(uint8_t *png, size_t png_len, uint8_t *jpeg,
                          size_t jpeg_len, uint8_t *output, size_t output_len) {
  for (size_t len = 0; len < png_len; len++)
    (void)decode_png(png, len, output, output_len, 0);
  for (size_t len = 0; len < jpeg_len; len++)
    (void)decode_jpeg(jpeg, len, output, output_len, 0);

  for (size_t i = 0; i < png_len; i++) {
    png[i] ^= 1;
    (void)decode_png(png, png_len, output, output_len, 0);
    png[i] ^= 1;
  }
  for (size_t i = 0; i < jpeg_len; i++) {
    jpeg[i] ^= 1;
    (void)decode_jpeg(jpeg, jpeg_len, output, output_len, 0);
    jpeg[i] ^= 1;
  }

  uint8_t random[512];
  uint32_t state = 0x494d4147u;
  for (unsigned iteration = 0; iteration < 10000; iteration++) {
    size_t len = random_word(&state) % sizeof(random);
    for (size_t i = 0; i < len; i++)
      random[i] = (uint8_t)random_word(&state);
    (void)decode_png(random, len, output, output_len, 0);
    (void)decode_jpeg(random, len, output, output_len, 0);
  }
}

int main(void) {
  const size_t width = 17;
  const size_t height = 13;
  const size_t output_len = width * height * 3;
  uint8_t pixels[17 * 13 * 3];
  for (size_t i = 0; i < sizeof(pixels); i++)
    pixels[i] = (uint8_t)((i * 37 + i / 11) & 0xffu);

  size_t png_len;
  uint8_t *png = encode_png_fixture(pixels, width, height, &png_len);
  uint8_t output[17 * 13 * 3];
  if (decode_png(png, png_len, output, output_len, 0) != PNG_OK ||
      memcmp(pixels, output, output_len) != 0)
    fail("PNG round trip changed pixels");

  size_t jpeg_len;
  uint8_t *jpeg = encode_jpeg_fixture(pixels, width, height, &jpeg_len);
  size_t decoded_width;
  size_t decoded_height;
  if (probe_jpeg(jpeg, jpeg_len, &decoded_width, &decoded_height) != JPEG_OK ||
      decoded_width != width || decoded_height != height ||
      decode_jpeg(jpeg, jpeg_len, output, output_len, 0) != JPEG_OK)
    fail("JPEG round trip failed");

  mutate_images(png, png_len, jpeg, jpeg_len, output, output_len);
  free(jpeg);
  free(png);
  return 0;
}
