#include <stdint.h>
#include <stdio.h>
#include <string.h>
struct score {const char *next;int64_t value;};
extern int plamo_parse_score(const char*,const char*,struct score*);
int main(void){
 const char *v[]={"-13.8468,","0.0]","-99.993","12","-0.1","01","1.23456","-","1e2","1.","1001"};
 int64_t expected[]={-138468,0,-999930,120000,-1000};
 for(unsigned i=0;i<sizeof(v)/sizeof(*v);i++){struct score s;int ok=plamo_parse_score(v[i],v[i]+strlen(v[i]),&s);if(ok!=(i<5)||(ok&&s.value!=expected[i])||(!ok&&(s.next||s.value)))return 1;}
 puts("vocabulary score parser: PASS");
}
