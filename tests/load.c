#include <stdint.h>
#include <stdio.h>
#include <string.h>
struct view {const char *header; uint64_t header_size; const void *data; uint64_t data_size;};
struct number {const char *next; uint64_t value;};
extern int plamo_safetensors_view(const void*,uint64_t,struct view*);
extern int plamo_parse_u64(const char*,const char*,struct number*);
int main(void) {
 unsigned char file[20]={8}; memcpy(file+8,"{}      ",8);
 struct view v;
 if(!plamo_safetensors_view(file,20,&v)||v.header_size!=8||v.data_size!=4||v.data!=file+16) return 1;
 file[0]=255;
 if(plamo_safetensors_view(file,20,&v)||v.header||v.data) return 2;
 if(plamo_safetensors_view(file,7,&v)) return 3;
 const char *cases[]={"0","123,","18446744073709551615","18446744073709551616","01","-1",""};
 uint64_t values[]={0,123,UINT64_MAX};
 for(int i=0;i<7;i++) {
  struct number n;
  int ok=plamo_parse_u64(cases[i],cases[i]+strlen(cases[i]),&n);
  if(ok!=(i<3)) return 10+i;
  if(ok && n.value!=values[i]) return 20+i;
  if(!ok && (n.next||n.value)) return 30+i;
 }
 puts("safetensors envelope / uint64: PASS");
}
