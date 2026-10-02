# `09-many-columns`

Work on many columns at once with selectors, and use the compositions of
`Kit`.

```bash
dune exec ./main.exe
```

- `Query.append` puts two tables with the same columns one after the other.
- `Sel` chooses columns by name, prefix, suffix or kind. `Expr.keep` keeps
  them, and `Expr.across` applies one function to each.
- `Kit.null_count`, `Kit.describe` and `Kit.value_counts` summarise a query.
- `Kit.drop`, `Kit.rename`, `Kit.distinct` and `Kit.top_k` reshape it.
- `Kit.categorize` gives a text column the dictionary of its values, and
  `Kit.one_hot` splits a categorical column into one boolean column per
  category:

```
table 6 rows × 5 columns
 site    ph       temp     grade_a  grade_b
 string  float64  float32  bool     bool
 north   7.10000  12.5000  true     false
 north         ∅  13.0000  false    true
 north   6.80000  11.7500  true     false
 south   7.40000  18.0000  true     false
 south   7.20000  17.5000  false    true
 south   7.40000  18.0000  true     false
```
