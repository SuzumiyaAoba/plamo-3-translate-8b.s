#include <stdint.h>
#include <stdio.h>
#include <string.h>
struct tensor {const uint16_t *data;uint64_t bytes,rank,dims[4],reserved;};
extern int plamo_find_tensor(const void*,uint64_t,const char*,struct tensor*);
int main(void) {
 const char *h="{\"__metadata__\":{\"format\":\"pt\"},\"a\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]},\"b\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[2,6]}}";
 unsigned char file[512]={0}; uint64_t n=strlen(h); memcpy(file,&n,8);memcpy(file+8,h,n);
 uint16_t data[]={0x3f80,0x4000,0x4040};memcpy(file+8+n,data,6);
 struct tensor t;
 if(!plamo_find_tensor(file,8+n+6,"b",&t)||t.bytes!=4||t.rank!=1||t.dims[0]!=2||t.data[0]!=0x4000) return 1;
 if(!plamo_find_tensor(file,8+n+6,"a",&t)||t.data[0]!=0x3f80) return 2;
 if(plamo_find_tensor(file,8+n+6,"ab",&t)||t.data) return 3;
 if(plamo_find_tensor(file,8+n+5,"b",&t)||t.data) return 4;
 puts("safetensors lookup: PASS (metadata, names, direct weight pointer, bounds)");
}
