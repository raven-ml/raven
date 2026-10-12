# Host program fixtures

The suite's and the bench's host code is `<f>_x86_64.o` and
`<f>_aarch64.o` for each `<f>` of `blocks`, `deep`, `writable`, `undefined`,
`data_entry`, `empty`, `scale`, `many`, `linked` and `got`. `gen.sh` makes
each from `<f>.c`, which says what it does. `got`'s call of an external
function goes through a word on x86_64 too. `linked_x86_64_windows.o`
follows x86_64 Windows' convention for process functions.
