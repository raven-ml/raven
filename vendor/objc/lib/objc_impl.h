#ifndef OBJC_IMPL_H
#define OBJC_IMPL_H

typedef void* ObjC_Id_t;
typedef void* ObjC_Sel_t;
typedef void* ObjC_Class_t;

ObjC_Class_t objc_impl_getClass(const char* className);

ObjC_Sel_t objc_impl_registerName(const char* selectorName);

const char* objc_impl_getNameFromSelector(ObjC_Sel_t selector);

const char* objc_impl_getNameFromClass(ObjC_Class_t aClass);

ObjC_Class_t objc_impl_getClassOfObject(ObjC_Id_t object);

#endif  // OBJC_IMPL_H