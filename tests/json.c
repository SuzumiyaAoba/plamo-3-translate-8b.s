#include <stdint.h>
#include <string.h>
#include <stdio.h>
struct token {const char *next,*start; uint64_t length,value;};
extern int plamo_json_next(const char*,const char*,struct token*);
int main(void) {
 const char *s=" {\"dtype\":\"BF16\",\"shape\":[2,3]} ";
 int expected[]={123,256,58,256,44,256,58,91,257,44,257,93,125,0};
 const char *p=s,*end=s+strlen(s); struct token t;
 for(unsigned i=0;i<sizeof(expected)/sizeof(*expected);i++) {
  int k=plamo_json_next(p,end,&t); if(k!=expected[i]) return 1;
  if(k==257 && t.value!=2 && t.value!=3) return 2;
  p=t.next;
 }
 const char *bad[]={"\"abc","\"\\q\"","\"\\u12xz\"","01","18446744073709551616","2x","\"a\nb\""};
 for(unsigned i=0;i<sizeof(bad)/sizeof(*bad);i++)
  if(plamo_json_next(bad[i],bad[i]+strlen(bad[i]),&t)!=-1) return 3;
 const char *good="\"a\\\"\\u1234\"";
 if(plamo_json_next(good,good+strlen(good),&t)!=256) return 4;
 puts("JSON lexer: PASS (tensor metadata, escapes, malformed input, overflow)");
}
