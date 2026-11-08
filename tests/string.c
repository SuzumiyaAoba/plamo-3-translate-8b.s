#include <stdint.h>
#include <string.h>
#include <stdio.h>
extern uint64_t plamo_json_string(const char*,const char*,uint32_t*,uint64_t);
int main(void){
 const char *s="日\\u672c\\n\\ud83d\\ude00\\\"";uint32_t out[16],expected[]={0x65e5,0x672c,10,0x1f600,34};
 if(plamo_json_string(s,s+strlen(s),out,16)!=5)return 1;
 for(int i=0;i<5;i++)if(out[i]!=expected[i])return 2;
 const char *bad[]={"\\ud800","\\udc00","\\ud800\\u0041","\\q","\\u12x4","\"","\n"};
 for(unsigned i=0;i<sizeof(bad)/sizeof(*bad);i++)if(plamo_json_string(bad[i],bad[i]+strlen(bad[i]),out,16)!=UINT64_MAX)return 3;
 if(plamo_json_string(s,s+strlen(s),out,2)!=UINT64_MAX)return 4;
 puts("JSON string decode: PASS (UTF-8, escapes, surrogate pairs, capacity)");
}
