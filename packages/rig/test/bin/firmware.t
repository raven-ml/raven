rig firmware puts each image of a driver's list in a directory, once its
download has the image's pinned digest. support/curl plays
linux-firmware's server from the directory served: here it serves NV's
GSP image with bytes of its own, and no other image.

  $ . support/env.sh
  $ mkdir -p served/nvidia/ga102/gsp
  $ echo other >served/nvidia/ga102/gsp/gsp-570.144.bin

A download with another digest is refused, and one that fails is said
after curl's own line. An image that fails does not stop the others, and
nothing is written for any of them.

  $ rig firmware nv fw
  curl: (22) The requested URL returned error: 404
  rig: nvidia/ad102/gsp/booter_load-570.144.bin: curl exited with status 22
  curl: (22) The requested URL returned error: 404
  rig: nvidia/ad102/gsp/booter_unload-570.144.bin: curl exited with status 22
  curl: (22) The requested URL returned error: 404
  rig: nvidia/ad102/gsp/bootloader-570.144.bin: curl exited with status 22
  curl: (22) The requested URL returned error: 404
  rig: nvidia/ga102/gsp/booter_load-570.144.bin: curl exited with status 22
  curl: (22) The requested URL returned error: 404
  rig: nvidia/ga102/gsp/booter_unload-570.144.bin: curl exited with status 22
  curl: (22) The requested URL returned error: 404
  rig: nvidia/ga102/gsp/bootloader-570.144.bin: curl exited with status 22
  rig: nvidia/ga102/gsp/gsp-570.144.bin: the download's digest is b22206e1e4cb2d881a7284d716a9665fb2f6400ff179c8c6ea33903dbd377d29; the pin is 877edf9f772d262d1ea72b9a6ebe52643e60a37c1d312f6456ebbf7f03feb03f
  curl: (22) The requested URL returned error: 404
  rig: nvidia/gb202/gsp/bootloader-570.144.bin: curl exited with status 22
  curl: (22) The requested URL returned error: 404
  rig: nvidia/gb202/gsp/fmc-570.144.bin: curl exited with status 22
  [123]
  $ test -e fw
  [1]

A file in an image's place with another digest is no image: it is
fetched again, and left as it was when the download is refused.

  $ mkdir -p fw/nvidia/ga102/gsp
  $ echo old >fw/nvidia/ga102/gsp/gsp-570.144.bin
  $ rig firmware nv fw 2>&1 | grep gsp-570
  rig: nvidia/ga102/gsp/gsp-570.144.bin: the download's digest is b22206e1e4cb2d881a7284d716a9665fb2f6400ff179c8c6ea33903dbd377d29; the pin is 877edf9f772d262d1ea72b9a6ebe52643e60a37c1d312f6456ebbf7f03feb03f
  $ ls fw/nvidia/ga102/gsp
  gsp-570.144.bin
  $ cat fw/nvidia/ga102/gsp/gsp-570.144.bin
  old

rig firmware downloads with curl, found through PATH. Without it, it says
so once and fetches nothing.

  $ rig=$(command -v rig)
  $ PATH=$PWD/nowhere "$rig" firmware amd fw
  rig: curl is not on PATH; rig firmware downloads with it
  [123]
  $ test -e fw/amdgpu
  [1]

--help after the arguments is the page, as for every command.

  $ rig firmware amd fw --help | head -1
  NAME

The same downloader takes a small list of synthetic images here. Its
first and last images are missing, so every run below fails while still
processing the second image. Its pin is the BLAKE2b-256 digest of the text
below.

  $ rm -r fw
  $ mkdir -p served/test
  $ printf 'rig test fixture\n' >expected
  $ cp expected served/test/image.bin
  $ umask 022
  $ support/firmware_test.exe fw 2>/dev/null
  fetched test/image.bin
  [123]
  $ cmp expected fw/test/image.bin
  $ ls fw/test
  image.bin
  $ ls -l fw/test/image.bin | cut -c1-10
  -rw-r--r--

An image DIR holds is kept and not downloaded again: the server no
longer has it.

  $ rm served/test/image.bin
  $ support/firmware_test.exe fw 2>/dev/null
  kept test/image.bin
  [123]

A file in its place with another digest is replaced by a download with
the pinned one.

  $ cp expected served/test/image.bin
  $ echo old >fw/test/image.bin
  $ support/firmware_test.exe fw 2>/dev/null
  fetched test/image.bin
  [123]
  $ cmp expected fw/test/image.bin

A download that breaks is not written: here curl fails after half the
image.

  $ rm -r fw
  $ head -c 8 expected >served/test/image.bin
  $ echo 18 >served/test/image.bin.status
  $ support/firmware_test.exe fw 2>&1 | grep test/image.bin
  rig: test/image.bin: curl exited with status 18
  $ test -e fw
  [1]

A write that fails is said and does not stop the later image: here fw/test
is a file. Nothing is left of the image.

  $ cp expected served/test/image.bin
  $ rm served/test/image.bin.status
  $ mkdir fw
  $ touch fw/test
  $ support/firmware_test.exe fw 2>err
  [123]
  $ grep test/image.bin err
  rig: test/image.bin: fw/test/image.bin.part: Not a directory
  $ grep -c 'status 22$' err
  2
  $ ls fw
  test

With no DIR, the images go to rig's firmware directory,
$XDG_DATA_HOME/rig/firmware, created if missing.

  $ XDG_DATA_HOME=$PWD/data support/firmware_test.exe 2>/dev/null
  fetched test/image.bin
  [123]
  $ cmp expected data/rig/firmware/test/image.bin

Where XDG_DATA_HOME is unset, the directory is
~/.local/share/rig/firmware.

  $ env -u XDG_DATA_HOME HOME=$PWD/home support/firmware_test.exe 2>/dev/null
  fetched test/image.bin
  [123]
  $ cmp expected home/.local/share/rig/firmware/test/image.bin

Without HOME either, there is no directory to fill, and nothing is
fetched.

  $ env -u XDG_DATA_HOME -u HOME support/firmware_test.exe
  rig: no DIR, and neither XDG_DATA_HOME nor HOME is an absolute path
  [123]

rig firmware takes no DIR the same way.

  $ env -u XDG_DATA_HOME -u HOME rig firmware nv
  rig: no DIR, and neither XDG_DATA_HOME nor HOME is an absolute path
  [123]
  $ XDG_DATA_HOME=$PWD/data rig firmware nv 2>&1 | grep gsp-570
  rig: nvidia/ga102/gsp/gsp-570.144.bin: the download's digest is b22206e1e4cb2d881a7284d716a9665fb2f6400ff179c8c6ea33903dbd377d29; the pin is 877edf9f772d262d1ea72b9a6ebe52643e60a37c1d312f6456ebbf7f03feb03f
