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
  rig: nvidia/ad102/gsp/bootloader-570.144.bin: curl exited with status 22
  curl: (22) The requested URL returned error: 404
  rig: nvidia/ga102/gsp/booter_load-570.144.bin: curl exited with status 22
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

The AMD list holds the one image the tests carry,
fixtures/smu_13_0_14.bin, AMD's own, whose digest is its pin. The other
images are not served, so every run below fails, and its standard output
is the subject.

  $ rm -r fw
  $ mkdir -p served/amdgpu
  $ cat fixtures/smu_13_0_14.bin >served/amdgpu/smu_13_0_14.bin
  $ umask 022
  $ rig firmware amd fw 2>/dev/null
  fetched amdgpu/smu_13_0_14.bin
  [123]
  $ cmp fixtures/smu_13_0_14.bin fw/amdgpu/smu_13_0_14.bin
  $ ls fw/amdgpu
  smu_13_0_14.bin
  $ ls -l fw/amdgpu/smu_13_0_14.bin | cut -c1-10
  -rw-r--r--

An image DIR holds is kept and not downloaded again: the server no
longer has it.

  $ rm served/amdgpu/smu_13_0_14.bin
  $ rig firmware amd fw 2>/dev/null
  kept amdgpu/smu_13_0_14.bin
  [123]

A file in its place with another digest is replaced by a download with
the pinned one.

  $ cat fixtures/smu_13_0_14.bin >served/amdgpu/smu_13_0_14.bin
  $ echo old >fw/amdgpu/smu_13_0_14.bin
  $ rig firmware amd fw 2>/dev/null
  fetched amdgpu/smu_13_0_14.bin
  [123]
  $ cmp fixtures/smu_13_0_14.bin fw/amdgpu/smu_13_0_14.bin

A download that breaks is not written: here curl fails after half the
image.

  $ rm -r fw
  $ head -c 710 fixtures/smu_13_0_14.bin >served/amdgpu/smu_13_0_14.bin
  $ echo 18 >served/amdgpu/smu_13_0_14.bin.status
  $ rig firmware amd fw 2>&1 | grep smu_13_0_14
  rig: amdgpu/smu_13_0_14.bin: curl exited with status 18
  $ test -e fw
  [1]

A write that fails is said, and does not stop the others: here fw/amdgpu
is a file. Nothing is left of the image.

  $ cat fixtures/smu_13_0_14.bin >served/amdgpu/smu_13_0_14.bin
  $ rm served/amdgpu/smu_13_0_14.bin.status
  $ mkdir fw
  $ touch fw/amdgpu
  $ rig firmware amd fw 2>err
  [123]
  $ grep smu_13_0_14 err
  rig: amdgpu/smu_13_0_14.bin: fw/amdgpu/smu_13_0_14.bin.part: Not a directory
  $ grep -c 'status 22$' err
  44
  $ ls fw
  amdgpu
