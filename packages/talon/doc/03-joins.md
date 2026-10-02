# Joins

`left |> Query.join ~on right` pairs the rows of two queries that a condition
matches. This page covers the conditions, the kinds of join, the columns of
the result, and assertions on the number of matches.

The examples join reaction-time trials with a table of subjects. Subject
`s3` has no entry, and `s4` has no trials:

```ocaml
open Talon

let trials =
  Query.of_table
    (Talon.v
       [
         ("subject", Column.v Type.string [| "s1"; "s1"; "s2"; "s3"; "s3" |]);
         ("trial", Column.v Type.int16 [| 1; 2; 1; 1; 2 |]);
         ("rt", Column.v Type.float64 [| 0.412; 0.388; 0.501; 0.450; 0.433 |]);
       ])

let subjects =
  Query.of_table
    (Talon.v
       [
         ("id", Column.v Type.string [| "s1"; "s2"; "s4" |]);
         ("age", Column.v Type.uint8 [| 24; 31; 27 |]);
       ])
```

## Conditions

A condition is a value of `Join.cond`. Atoms name a left column, then a
right column:

| Condition | Matches |
|---|---|
| `Join.keys [ "a"; "b" ]` | rows whose columns `a` and `b` are the same keys on both sides |
| `Join.eq "subject" "id"` | rows whose left `subject` and right `id` are the same key |
| `Join.position` | row i of the left with row i of the right |
| `Join.all` | every pair of rows |

Equality atoms combine with `&&`, written inside `Join.( … )`:
`Join.(keys [ "ticker" ] && eq "day" "date")`. A combination that no
algorithm runs, such as `position` with another atom, raises when it is
built.

Keys match by identity: the same values in the total order that sorting
uses, where NaN equals NaN and `-0.` equals `0.`. A null key is one more key,
so null keys on both sides match each other. The two columns of an `eq` must
meet at a common type, as operands of an expression do.

## Kinds

`~kind` says which rows the join keeps. It defaults to `Inner`.

```ocaml
let on = Join.eq "subject" "id"
let inner = Query.(trials |> join ~on subjects)
let left = Query.(trials |> join ~kind:Left ~on subjects)
let full = Query.(trials |> join ~kind:Full ~on subjects)
let semi = Query.(trials |> join ~kind:Semi ~on subjects)
let anti = Query.(trials |> join ~kind:Anti ~on subjects)
```

`Inner` keeps the matched pairs. `Left` also keeps each left row without a
match, its right columns null:

```text
table 5 rows × 4 columns
 subject  trial  rt        age
 string   int16  float64   uint8
 s1           1  0.412000     24
 s1           2  0.388000     24
 s2           1  0.501000     31
 s3           1  0.450000      ∅
 s3           2  0.433000      ∅
```

`Full` then appends each right row without a match, its left columns null
except the key, which takes the right row's value:

```text
table 6 rows × 4 columns
 subject  trial  rt        age
 string   int16  float64   uint8
 s1           1  0.412000     24
 s1           2  0.388000     24
 s2           1  0.501000     31
 s3           1  0.450000      ∅
 s3           2  0.433000      ∅
 s4           ∅         ∅     27
```

`Semi` keeps each left row that has a match, once, and `Anti` each left row
that has none. Both keep the left columns only:

```text
table 2 rows × 3 columns
 subject  trial  rt
 string   int16  float64
 s3           1  0.450000
 s3           2  0.433000
```

A right join is a left join with the arguments swapped.

## Columns and Order

The result has the left columns, then the right columns except the right
side of each equality atom: the key of `eq "subject" "id"` appears once, as
`subject`. In a `Full` join that key has the common type of both sides. A
name on both sides is a problem, and talon adds no suffixes: rename one side
first with `Kit.rename`.

Rows come in the left's order, each left row followed by its matches in the
right's order. An equality join reads both inputs before it returns a row and
runs in time proportional to the rows of both inputs plus the matches.

## Asserting Matches

`~each_left` and `~each_right` state how many matches each row of a side must
have: `Any`, the default, `At_most_one`, `One` or `At_least_one`. A row with
another number fails the run and names the side, the key and the count. A
lookup table that must hold every key once is `~each_left:One`:

```ocaml
let checked = Query.(trials |> join ~on ~each_left:One subjects)
```

```text
join ~on:(eq "subject" "id") ~each_left:One: row 3: the left row whose "subject" is "s3" matches 0 rows, not one.
```

Without the assertion, the inner join drops `s3`'s trials and nothing says
so. With `~each_right:At_most_one`, a lookup table with a duplicated key fails
instead of duplicating left rows.

## Problems

A join checks its condition against both schemas when it is applied. A
column missing on its side is a problem, reported with the columns of both
inputs:

```text
join: 1 problem
  right: no column "subject". The columns are "id" and "age".
  left (3 columns): subject string, trial int16, rt float64
  right (2 columns): id string, age uint8
```

The example [04-joins](../examples/04-joins/README.md) runs every join on this
page.
