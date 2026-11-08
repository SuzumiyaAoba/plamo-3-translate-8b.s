#include "inference.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
extern int plamo_model_open(void);
extern struct model plamo_model;
extern uint64_t plamo_argmax(const float*,uint64_t);
#ifdef PLAMO_IO_POLICY
extern uint64_t plamo_io_stats[4];
#endif
static const char *dump_directory;
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static uint64_t hash(const void *p,size_t n){const unsigned char *a=p;uint64_t h=14695981039346656037ULL;for(size_t i=0;i<n;i++){h^=a[i];h*=1099511628211ULL;}return h;}
static void dump(const char *phase,const char *name,const void *data,size_t bytes){
 if(!dump_directory)return;char path[4096];int n=snprintf(path,sizeof path,"%s/%s-%s.bin",dump_directory,phase,name);if(n<0||(size_t)n>=sizeof path)exit(70);
 FILE *f=fopen(path,"wb");if(!f||fwrite(data,1,bytes,f)!=bytes||fclose(f))exit(71);
}
static void report(const char *phase,double elapsed,struct model *m){
 if(strcmp(phase,"open")){dump(phase,"logits",m->logits,m->vocab*4);dump(phase,"state",m->state,m->hidden*4);}
 struct rusage u;getrusage(RUSAGE_SELF,&u);uint64_t bytes=0,reads=0,advice=0,errors=0;
#ifdef PLAMO_IO_POLICY
 bytes=plamo_io_stats[0];reads=plamo_io_stats[1];advice=plamo_io_stats[2];errors=plamo_io_stats[3];
#endif
 printf("{\"phase\":\"%s\",\"seconds\":%.9f,\"token\":%llu,\"logits_hash\":\"%016llx\",\"state_hash\":\"%016llx\",\"major_faults\":%ld,\"max_rss_bytes\":%ld,\"staged_bytes\":%llu,\"read_calls\":%llu,\"advice_calls\":%llu,\"read_failures\":%llu}\n",phase,elapsed,(unsigned long long)plamo_argmax(m->logits,m->vocab),(unsigned long long)hash(m->logits,m->vocab*4),(unsigned long long)hash(m->state,m->hidden*4),u.ru_majflt,u.ru_maxrss,(unsigned long long)bytes,(unsigned long long)reads,(unsigned long long)advice,(unsigned long long)errors);fflush(stdout);
}
#ifdef PLAMO_IO_POLICY
extern uint64_t plamo_io_policy;
#endif
int main(int argc,char **argv){
 if(argc>2)dump_directory=argv[2];
#ifdef PLAMO_IO_POLICY
 if(argc>1){plamo_io_policy=strtoull(argv[1],0,10);if(plamo_io_policy>2)return 64;}
#else
 (void)argc;(void)argv;
#endif
 double start=now();if(!plamo_model_open())return 1;struct model *m=&plamo_model;report("open",now()-start,m);
 uint32_t ids[]={1,16,48472,19,22125,16,2225,17,1943,18,1109,19,7272,46,16,4160,17,1943,18,1385,19};size_t n=sizeof ids/sizeof *ids;
 float *states=calloc(n*m->hidden,4);if(!states)return 2;
 start=now();if(!plamo_prefill(m,ids,n,states))return 3;report("prefill_21",now()-start,m);
 uint64_t token=plamo_argmax(m->logits,m->vocab);if(token!=51871)return 4;
 start=now();if(!plamo_forward(m,token))return 5;report("decode",now()-start,m);
 if(plamo_argmax(m->logits,m->vocab)!=16)return 6;free(states);return 0;
}
