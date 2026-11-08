#include <stdint.h>
#include <stdio.h>
#include <string.h>
struct scalar {const char *next;uint64_t cp;};
extern int plamo_utf8_next(const char*,const char*,struct scalar*);
extern uint64_t plamo_byte_fallback(uint64_t,const uint32_t*,uint32_t*);
int main(void){
 const char *s="A日本語😀";uint64_t cp[]={65,0x65e5,0x672c,0x8a9e,0x1f600};const char *p=s,*end=s+strlen(s);struct scalar r;
 for(int i=0;i<5;i++){if(!plamo_utf8_next(p,end,&r)||r.cp!=cp[i])return 1;p=r.next;}
 if(p!=end||plamo_utf8_next(p,end,&r))return 2;
 const char *bad[]={"\xc0\x80","\xed\xa0\x80","\xf4\x90\x80\x80","\xe3\x81","\x80","\xe0\x80\x80","\xf0\x80\x80\x80"};
 for(unsigned i=0;i<sizeof(bad)/sizeof(*bad);i++)if(plamo_utf8_next(bad[i],bad[i]+strlen(bad[i]),&r)||r.next||r.cp)return 3;
 uint32_t table[256],ids[4];for(unsigned i=0;i<256;i++)table[i]=1000+i;
 const uint32_t boundary[]={0,0x7f,0x80,0x7ff,0x800,0xd7ff,0xe000,0xffff,0x10000,0x10ffff};
 for(unsigned i=0;i<sizeof(boundary)/sizeof(*boundary);i++){
  unsigned n=plamo_byte_fallback(boundary[i],table,ids);char encoded[4];
  if(!n||n>4)return 4;
  for(unsigned j=0;j<n;j++)encoded[j]=(char)(ids[j]-1000);
  if(!plamo_utf8_next(encoded,encoded+n,&r)||r.cp!=boundary[i]||r.next!=encoded+n)return 5;
 }
 ids[0]=42;if(plamo_byte_fallback(0xd800,table,ids)||ids[0]!=42)return 6;
 if(plamo_byte_fallback(0x110000,table,ids)||ids[0]!=42)return 7;
 // Every valid scalar must round-trip, including NUL and supplementary planes.
 for(uint32_t cp=0;cp<=0x10ffff;cp++){
  if(cp>=0xd800&&cp<=0xdfff)continue;
  unsigned n=plamo_byte_fallback(cp,table,ids);char b[4];
  for(unsigned j=0;j<n;j++)b[j]=(char)(ids[j]-1000);
  if(!plamo_utf8_next(b,b+n,&r)||r.cp!=cp)return 8;
 }
 puts("UTF-8: PASS (Japanese, supplementary planes, malformed sequences, all-scalar byte fallback round-trip)");
}
