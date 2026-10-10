# Metal fixtures

`fill.metallib` holds the kernels of `fill.metal`, `launch.metallib`
those of `launch.metal`, `vertex.metallib` the vertex function of
`vertex.metal`, and `threadgroup.metallib` the kernels of
`threadgroup.metal`; each source says what its functions do.
Made in this directory on macOS 26.3.1 with Xcode 26.3's Metal toolchain
(`metal` 32023.864), for each `<name>`:

```
xcrun -sdk macosx metal -c <name>.metal -o <name>.air
xcrun -sdk macosx metallib <name>.air -o <name>.metallib
rm <name>.air
```
