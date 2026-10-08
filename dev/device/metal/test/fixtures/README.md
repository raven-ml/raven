# Metal fixtures

`fill.metallib` holds the kernels of `fill.metal`, whose source says what
each does. Made in this directory on macOS 26.3.1 with Xcode 26.3's Metal
toolchain (`metal` 32023.864):

```
xcrun -sdk macosx metal -c fill.metal -o fill.air
xcrun -sdk macosx metallib fill.air -o fill.metallib
rm fill.air
```
