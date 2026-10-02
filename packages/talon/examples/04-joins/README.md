# `04-joins`

Join trials with subjects on `subject = id` with each kind of join, then
assert that every trial has exactly one subject.

```bash
dune exec ./main.exe
```

- `Join.eq "subject" "id"` matches rows whose keys are the same; the key
  appears once, as `subject`.
- `Inner`, `Left` and `Full` keep pairs; `Semi` and `Anti` keep the left rows
  with and without a match.
- `~each_left:One` turns the missing subject `s3` into a failed run:

```
join ~on:(eq "subject" "id") ~each_left:One: row 3: the left row whose "subject" is "s3" matches 0 rows, not one.
```
