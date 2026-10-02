# `06-mistakes`

The two ways a query fails.

```bash
dune exec ./main.exe
```

A verb checks its arguments against its input's schema when it is applied,
and raises one `Invalid_argument` that lists every problem:

```
aggregate: 3 problems
  ~by: no column "carier". Did you mean "carrier"?
  "mean_delay" := mean dep_dly
    no column "dep_dly". Did you mean "dep_delay"?
  "late" := mean carrier
    Col.float reads float16, float32 or float64, but "carrier" is string.
  input (3 columns): carrier string, origin string, dep_delay int64
```

A failure in the data is an `Error` value from `Query.run`, naming the plan
step and the row of its input:

```
derive ["delay" := cast int8 dep_delay]: row 1: cannot cast 130 to int8.
```
