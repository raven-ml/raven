# Claims

This page shows a function that writes its result over its argument, and a
caller that keeps the argument. The rules are in the Claims section of
[`rig.mli`](../lib/rig.mli), which defines read claims, keeps, memory
outside the claims and takes, and says who may take memory and how a taker
writes.

## The story

An array operation whose input dies with the call can save an allocation by
writing its result over that input's memory. That is safe only if nothing
else reads the memory, and claims record what does. A function takes the
inputs it writes over with `Buffer.take`, which succeeds when each buffer
spans its memory, the memory is inside the claims and admits writes, and no
read claim or keep holds it. Every buffer made over the memory before is
then dead, and the function writes through the buffers the take returns. A
caller that still needs the buffer keeps it across the call
(`Buffer.keep`), and the function writes elsewhere.

Three things in the Claims section are easy to miss. Work already submitted
on the memory holds no claim, since the stamps order a write in place after
it. A take is one step: no call sees the memory between its check and the
death of the old buffers, and an input whose buffers lie on several devices
is taken whole or not at all. A prepared submission keeps its fixed memory
and the buffers its parts name for its life. It claims the buffers a submit
passes only while the submit runs, so a take between two submits can kill
one of them.

## Example

This function adds one, modulo 256, to every byte of its argument, which it
takes when it can:

```ocaml
# open Rig;;
# let incr b =
    let n = Buffer.length b in
    let taken = Option.map List.hd (Buffer.take ~why:"donated to incr" [ b ]) in
    let src = Option.value taken ~default:b in
    let bytes = Bytes.create n in
    Buffer.blit_to_bytes src 0 bytes 0 n;
    let up ch = Char.chr ((Char.code ch + 1) land 255) in
    let s = String.map up (Bytes.to_string bytes) in
    let out = match taken with Some b -> b | None -> Buffer.create host n in
    Buffer.blit_from_string s 0 out 0 n;
    out;;
val incr : Rig.Buffer.t -> Rig.Buffer.t = <fun>
# let x = Buffer.of_string "abc" in
  ignore (incr x);
  Buffer.dead x;;
- : string option = Some "donated to incr"
# let x = Buffer.of_string "abc" in
  let k = Buffer.keep [ x ] in
  ignore (incr x);
  Buffer.release k;
  Buffer.dead x;;
- : string option = None
```

The first call writes over `x`'s memory, and `x` is dead from then on. In the
second, the caller's keep refuses the take, so `incr` writes a new buffer and
`x` stays live.
