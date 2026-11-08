#include <stdint.h>
#include <stdio.h>
#include <string.h>
struct row{uint64_t length;int64_t score;uint64_t type;};
extern int plamo_vocab_row(const char*,const char*,uint32_t*,uint64_t,struct row*);
int main(void){
 uint32_t cp[1024];struct row r;
 const char *s="[\"日本語\", -12.3456, \"NORMAL\"]\n";
 if(!plamo_vocab_row(s,s+strlen(s),cp,1024,&r)||r.length!=3||r.score!=-123456||r.type||cp[0]!=0x65e5)return 1;
 const char *bad="[\"x\", -1, \"INVALID\"]";if(plamo_vocab_row(bad,bad+strlen(bad),cp,1024,&r)||r.length)return 2;
 FILE *f=fopen("models/plamo/tokenizer.jsonl","r");if(f){char line[16384];unsigned n=0;while(fgets(line,sizeof line,f)){if(!plamo_vocab_row(line,line+strlen(line),cp,1024,&r)){fprintf(stderr,"row %u\n",n);return 3;}n++;}fclose(f);if(n!=107520)return 4;printf("official vocabulary rows: PASS (%u)\n",n);}
 puts("vocabulary row parser: PASS");
}
