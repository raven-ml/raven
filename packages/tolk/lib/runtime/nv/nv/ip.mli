(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Firmware and boot images for the driver-less NVIDIA boot chain.

    Bringing an NVIDIA GPU up without a kernel driver means feeding the
    security-processor boot chain a set of prepared images: the VBIOS
    ucode that reserves the frame-buffer resident tables, the falcon
    bootloaders, the chain-of-trust firmware, and the GSP image with its
    page hierarchy. This module is the parsing and patching layer that
    turns raw firmware files and the on-board VBIOS ROM into those images
    and the offsets that drive them onto the hardware.

    Every function here is a pure transform over byte images (or, for the
    VBIOS window, over an injected register read). Loading the images onto
    the device — allocating boot memory, programming falcon registers, and
    running the boot handshakes — builds on top of the values these
    functions produce. *)

(** {1:firmware Firmware files} *)

module Firmware : sig
  val fetch : ?dir:string -> chip_dir:string -> string -> bytes
  (** [fetch ~chip_dir name] loads and verifies the firmware file [name]
      for chip directory [chip_dir]. The file is read from
      [dir/chip_dir/gsp/name], where [dir] defaults to the [NV_FW_PATH]
      environment variable or [/lib/firmware/nvidia]; a [.zst]-compressed
      copy is accepted and decoded through the system [zstd]. Its SHA-256
      is checked against the pinned digest for [(name, chip_dir)].

      Raises [Failure] when no digest is pinned for the file, when neither
      the plain nor the compressed file is found (naming both searched
      paths, the expected digest, and the pinned upstream URL), or when the
      digest does not match. *)
end

(** {1:falcon Falcon boot images} *)

module Flcn : sig
  type desc = {
    imem_load_size : int;  (** Bytes of instruction memory to load. *)
    imem_phys_base : int;  (** Physical base of the instruction memory. *)
    imem_virt_base : int;  (** Virtual base of the instruction memory. *)
    dmem_phys_base : int;  (** Physical base of the data memory. *)
    dmem_load_size : int;  (** Bytes of data memory to load. *)
    pkc_data_offset : int;  (** Offset of the signature region in the image. *)
    engine_id_mask : int;  (** Engine-id mask the ucode is bound to. *)
    ucode_id : int;  (** Ucode id the boot ROM checks. *)
    stored_size : int;  (** Stored image size, before 256-byte rounding. *)
    interface_offset : int;
        (** Offset of the application-interface header within the image. *)
  }
  (** The FWSEC ucode descriptor, holding the geometry a falcon boot needs
      to load and verify the image. *)

  type ucode = {
    desc : desc;  (** The parsed descriptor. *)
    frts_offset : int;
        (** Byte offset in VRAM reserved for the frame-buffer resident
            tables. *)
    frts_image : bytes;  (** The patched, boot-ready FWSEC image. *)
  }
  (** The prepared FWSEC ucode. *)

  val read_vbios : read32:(int -> int) -> bytes
  (** [read_vbios ~read32] reads the 1 MiB VBIOS ROM out of the register
      window at byte offset [0x300000] into a flat byte image. [read32 a]
      is the unsigned 32-bit value at byte address [a] in the register
      aperture. *)

  val prep_ucode : rom:bytes -> vram_size:int -> ucode
  (** [prep_ucode ~rom ~vram_size] parses the FWSEC ucode out of the VBIOS
      image [rom] and patches it for boot. [vram_size] is the device's
      memory size in bytes, which fixes the reserved-tables offset.

      Raises [Failure] if the VBIOS structures are malformed or the
      production FWSEC descriptor is absent, and [Invalid_argument] on a
      read past the end of [rom]. *)

  type booter = {
    image : bytes;  (** The patched bootloader image. *)
    data_off : int;  (** Byte offset of the data segment in the image. *)
    data_sz : int;  (** Size of the data segment. *)
    code_off : int;  (** Byte offset of the code segment in the image. *)
    code_sz : int;  (** Size of the code segment. *)
  }
  (** A prepared heavy-secured bootloader. *)

  val prep_booter : blob:bytes -> booter
  (** [prep_booter ~blob] parses the heavy-secured bootloader firmware
      [blob] and splices its production signature into the boot image,
      returning the image and its code and data spans.

      Raises [Invalid_argument] on a read past the end of [blob]. *)
end

(** {1:cot Chain-of-trust boot image} *)

module Flcn_cot : sig
  type fmc = {
    image : bytes;  (** The bootable firmware image. *)
    hash : int array;  (** Image hash, as 32-bit words. *)
    signature : int array;  (** Image signature, as 32-bit words. *)
    public_key : int array;  (** Verification public key, as 32-bit words. *)
  }
  (** A chain-of-trust firmware image and its verification blobs. *)

  val init_fmc_image : blob:bytes -> fmc
  (** [init_fmc_image ~blob] splits the chain-of-trust firmware ELF [blob]
      into its bootable image and the hash, signature and public-key blobs.

      Raises [Failure] if a required section is missing, and
      [Invalid_argument] if [blob] is not a valid ELF object. *)
end

(** {1:gsp GSP firmware image} *)

module Gsp : sig
  type radix3 = {
    npages : int array;
        (** Page count at each of the four levels, deepest last. *)
    offsets : int array;
        (** Byte offset of each level within the hierarchy region. *)
    image_off : int;  (** Byte offset where the image begins. *)
  }
  (** The layout of a three-level page hierarchy over the GSP image. *)

  val radix3 : image_len:int -> radix3
  (** [radix3 ~image_len] computes the page-hierarchy layout for a GSP
      image of [image_len] bytes: the deepest level covers the image's
      4 KiB pages, and each level above holds one 8-byte pointer per page
      of the level below. The layout is enough to size and fill the
      hierarchy region once its pages have physical addresses. *)

  type split = {
    image : bytes;  (** The GSP firmware image. *)
    signature : bytes;  (** The per-chip signature section. *)
  }
  (** The GSP image split from its firmware ELF. *)

  val split_gsp_image : blob:bytes -> chip_name:string -> split
  (** [split_gsp_image ~blob ~chip_name] extracts the firmware image and
      the signature section for [chip_name] from the GSP firmware ELF
      [blob]. The signature section is named for the chip family (the first
      four characters of [chip_name], lowercased).

      Raises [Failure] if a required section is missing, and
      [Invalid_argument] if [blob] is not a valid ELF object or
      [chip_name] is shorter than four characters. *)

  type bootloader = {
    image : bytes;  (** The RISC-V bootloader image. *)
    monitor_code_offset : int;  (** Offset of the monitor code segment. *)
    monitor_data_offset : int;  (** Offset of the monitor data segment. *)
    manifest_offset : int;  (** Offset of the boot manifest. *)
  }
  (** A parsed RISC-V bootloader container. *)

  val init_boot_binary_image : blob:bytes -> bootloader
  (** [init_boot_binary_image ~blob] parses the RISC-V bootloader firmware
      [blob], returning the bootloader image and the monitor code, data and
      manifest offsets its descriptor names.

      Raises [Invalid_argument] on a read past the end of [blob]. *)
end
