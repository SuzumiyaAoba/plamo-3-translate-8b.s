#include <stdint.h>
#include <stdio.h>
#include <string.h>
struct tensor {const char *next; uint64_t offset,bytes,rank,dims[4];};
extern int plamo_tensor_descriptor(const char*,const char*,uint64_t,struct tensor*);
int main(void) {
 const char *cases[]={
 "{\"dtype\":\"BF16\",\"shape\":[2,3],\"data_offsets\":[4,16]}",
 "{\"data_offsets\":[0,2],\"shape\":[],\"dtype\":\"BF16\"}",
 "{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,4]}",
 "{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,3]}",
 "{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[4,2]}",
 "{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,100]}",
 "{\"dtype\":\"BF16\",\"shape\":[2]}",
 "{\"dtype\":\"BF16\",\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]}"};
 for(unsigned i=0;i<sizeof(cases)/sizeof(*cases);i++) {
  struct tensor t; int ok=plamo_tensor_descriptor(cases[i],cases[i]+strlen(cases[i]),16,&t);
  if(ok!=(i<2)) {fprintf(stderr,"case %u failed\n",i);return 1;}
  if(!ok && (t.next||t.bytes||t.rank)) return 2;
  if(i==0 && (t.offset!=4||t.bytes!=12||t.rank!=2||t.dims[0]!=2||t.dims[1]!=3)) return 3;
 }
 puts("BF16 tensor descriptor: PASS");
}
