# tolk D151 golden=38d7a69c860906dff9035fb914f91278

extern void callback();
__attribute__((min_vector_width(512))) void call_stack() {
  unsigned int buf0[1];
  ((void(*)(unsigned int*))(callback))((unsigned int*)((buf0+0)));
}

