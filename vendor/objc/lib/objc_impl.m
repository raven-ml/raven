#include <stdio.h>  // For NULL
#include "objc_impl.h"

#import <objc/runtime.h>
#import <objc/message.h>

ObjC_Class_t objc_impl_getClass(const char *className) {
  // No @autoreleasepool needed here as objc_getClass doesn't create objects
  // that would typically be autoreleased in this context.
  if (className == NULL) {
    return NULL;
  }
  // objc_getClass returns Class. Cast to ObjC_Class_t (void*) without ownership transfer.
  return (__bridge ObjC_Class_t)objc_getClass(className);
}

ObjC_Sel_t objc_impl_registerName(const char *selectorName) {
  if (selectorName == NULL) {
    return NULL; // NULL is (void*)0, perfectly fine for ObjC_Sel_t
  }
  SEL actual_sel = sel_registerName(selectorName);
  if (actual_sel == NULL) { // sel_registerName can return NULL on error
      return NULL;
  }
  return actual_sel;
}

const char* objc_impl_getNameFromSelector(ObjC_Sel_t selector_as_void_ptr) {
  if (selector_as_void_ptr == NULL) {
    return "";
  }
  return sel_getName(selector_as_void_ptr);
}

const char* objc_impl_getNameFromClass(ObjC_Class_t aClass) {
  if (aClass == NULL) {
    return ""; // Or NULL
  }
  // Cast ObjC_Class_t (void*) to Class for API call, no ownership transfer.
  return class_getName((__bridge Class)aClass);
}

ObjC_Class_t objc_impl_getClassOfObject(ObjC_Id_t object) {
  // object_getClass(nil) is valid and returns Nil (0).
  // Cast ObjC_Id_t (void*) to id, and result Class to ObjC_Class_t (void*).
  // No ownership transfer in either direction.
  return (__bridge ObjC_Class_t)object_getClass((__bridge id)object);
}