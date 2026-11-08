#include <stdint.h>
#include <stdio.h>
#include <string.h>
extern uint64_t plamo_decode_token(const char*,const char*,uint64_t,char*,uint64_t,uint32_t*,uint64_t);
extern uint64_t plamo_translation_prompt(const char*,const char*,const char*,char*,uint64_t);
int main(void){
 const char *v="[\"日本語😀\", -1.0, \"NORMAL\"]\n[\"<0xFF>\", 0.0, \"BYTE\"]\n";char out[1024];uint32_t scratch[100];
 uint64_t n=plamo_decode_token(v,v+strlen(v),0,out,sizeof out,scratch,100);
 const char *expected="日本語😀";if(n!=strlen(expected)||memcmp(out,expected,n))return 1;
 if(plamo_decode_token(v,v+strlen(v),1,out,sizeof out,scratch,100)!=1||(unsigned char)out[0]!=255)return 2;
 if(plamo_decode_token(v,v+strlen(v),2,out,sizeof out,scratch,100)!=UINT64_MAX)return 3;
 if(plamo_decode_token(v,v+strlen(v),0,out,1,scratch,100)!=UINT64_MAX)return 4;
 const char *prompt="<|plamo:bos|><|plamo:tag|>dataset<|plamo:msg|>translation<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>en<|plamo:msg|>Hello.<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>";
 n=plamo_translation_prompt("Hello.","en","ja",out,sizeof out);
 if(n!=strlen(prompt)||strcmp(out,prompt))return 5;
 if(plamo_translation_prompt("Hello.","en","ja",out,3)!=UINT64_MAX)return 6;
 puts("token decode / translation prompt: PASS");
}
