type ('a, 'd) t = {
  p : (('a[@ptree.walk f]), 'd) Foo.t;
  q : ('a, ('d[@ptree.skip])) Foo.t;
}
[@@deriving ptree]
