# Claims

This page shows a function that writes its result over its argument, and a
caller that keeps the argument. The rules are in the Claims section of
[`rig.mli`](../lib/rig.mli), which defines donation, read claims, exclusive
memory and consumption, and says who may donate and how a donor writes.

## The story

An array operation whose input dies with the call can save an allocation by
writing its result over that input's memory. That is safe only if nothing
else reads the memory, and claims record what does. A function takes its
inputs with `Claim.with_`, or its one input with `Claim.take`. A buffer it
donates is exclusive when it spans its memory, the memory is inside the
claims and admits writes, and no other claim holds it. The function then
consumes it, so every buffer made over the memory before is dead. A caller
that still needs the buffer holds a read claim across the call, and the
function writes elsewhere.

Three things in the Claims section are easy to miss. Work already submitted
on the memory holds no claim, since the stamps order a write in place after
it. A function consumes an input it holds exclusive before it touches its
bytes. Until then every call that touches them refuses the memory, a copy
the host runs included. The consumed buffer then works with every call,
`Rig.submit` included. A prepared submission claims its buffers only while
a submit runs, so a donation between two submits can kill one of them.

## Example

This function adds one, modulo 256, to every byte of its argument, which it
donates:

```ocaml
# open Rig;;
# let incr b =
    Claim.with_ ~read:[] ~donate:[ b ] @@ fun c ->
    let n = Buffer.length b in
    let in_place = Claim.exclusive c b in
    let b = if in_place then Claim.consume c ~why:"donated to incr" b else b in
    let bytes = Bytes.create n in
    Buffer.blit_to_bytes b 0 bytes 0 n;
    let up ch = Char.chr ((Char.code ch + 1) land 255) in
    let s = String.map up (Bytes.to_string bytes) in
    let out = if in_place then b else Buffer.create host n in
    Buffer.blit_from_string s 0 out 0 n;
    out;;
val incr : Rig.Buffer.t -> Rig.Buffer.t = <fun>
# let x = Buffer.of_string "abc" in
  ignore (incr x);
  Buffer.dead x;;
- : string option = Some "donated to incr"
# let x = Buffer.of_string "abc" in
  Buffer.read x Bigarray.char (fun _ -> ignore (incr x));
  Buffer.dead x;;
- : string option = None
```

The first call writes over `x`'s memory, and `x` is dead from then on. In the
second, the read claim of the caller's `Buffer.read` keeps the memory from
being exclusive, so `incr` writes a new buffer and `x` stays live.
