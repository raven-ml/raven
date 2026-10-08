# Fixtures

`sys/bus/pci/devices/` is a machine's PCI functions as Linux shows them,
each a directory holding its `vendor` and `class` files, written by hand in
the format of Linux's `Documentation/ABI/testing/sysfs-bus-pci`.

- `0000:01:00.0`: an NVIDIA VGA controller (class 0x030000).
- `0000:01:00.1`: its audio function (0x040300).
- `0000:02:00.0`: an AMD VGA controller.
- `0000:03:00.0`: an NVIDIA function whose `class` file is missing.
- `0000:0a:00.0`: an NVIDIA 3D controller (0x030200).
- `ffff:00:00.0` and `10000:00:01.0`: NVIDIA VGA controllers in domains of
  four and five digits, whose order by number is the reverse of their order
  as strings.
