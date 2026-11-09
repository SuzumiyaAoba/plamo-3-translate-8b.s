#ifdef PLAMO_RUNTIME_BACKEND
#include "../runtime/engine.h"
static struct plamo_engine *engine;
#else
#include "inference.h"
#endif
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
 printf("{");
#ifdef PLAMO_RUNTIME_BACKEND
 struct plamo_engine_info info;plamo_engine_get_info(engine,&info);
 printf("\"weight_storage\":\"%s\",\"cached_storage_bytes\":%llu,\"packed_weight_bytes\":%llu,",info.gpu.weight_storage[0]?info.gpu.weight_storage:"bf16",(unsigned long long)info.cache.storage_bytes,(unsigned long long)info.gpu.packed_weight_bytes);
 printf("\"command_buffers\":%llu,\"fused_mlps\":%llu,\"fused_decodes\":%llu,\"fused_prefills\":%llu,",(unsigned long long)info.gpu.command_buffers,(unsigned long long)info.gpu.fused_mlps,(unsigned long long)info.gpu.fused_decodes,(unsigned long long)info.gpu.fused_prefills);
 printf("\"backend\":\"%s\",\"precision\":\"%s\",\"gpu_dispatches\":%llu,\"gpu_errors\":%llu,\"cached_weight_bytes\":%llu,\"cache_budget_bytes\":%llu,\"gpu_allocated_bytes\":%llu,\"residency_sets\":%s,",info.backend,info.gpu.precision[0]?info.gpu.precision:"fp32",(unsigned long long)info.gpu.dispatches,(unsigned long long)info.gpu.errors,(unsigned long long)info.cache.cached_bytes,(unsigned long long)info.cache.budget_bytes,(unsigned long long)info.gpu.allocated_bytes,info.gpu.residency_sets?"true":"false");
#endif
 printf("\"phase\":\"%s\",\"seconds\":%.9f,\"token\":%llu,\"logits_hash\":\"%016llx\",\"state_hash\":\"%016llx\",\"major_faults\":%ld,\"max_rss_bytes\":%ld,\"staged_bytes\":%llu,\"read_calls\":%llu,\"advice_calls\":%llu,\"read_failures\":%llu}\n",phase,elapsed,(unsigned long long)plamo_argmax(m->logits,m->vocab),(unsigned long long)hash(m->logits,m->vocab*4),(unsigned long long)hash(m->state,m->hidden*4),u.ru_majflt,u.ru_maxrss,(unsigned long long)bytes,(unsigned long long)reads,(unsigned long long)advice,(unsigned long long)errors);fflush(stdout);
}
#ifdef PLAMO_IO_POLICY
extern uint64_t plamo_io_policy;
#endif
int main(int argc,char **argv){
 if(argc>2)dump_directory=argv[2];
#ifdef PLAMO_IO_POLICY
#ifndef PLAMO_RUNTIME_BACKEND
 if(argc>1){plamo_io_policy=strtoull(argv[1],0,10);if(plamo_io_policy>2)return 64;}
#endif
#else
 (void)argc;(void)argv;
#endif
 double start=now();
#ifdef PLAMO_RUNTIME_BACKEND
 if(argc<2)return 64;
 enum plamo_backend backend;
 if(!strcmp(argv[1],"cpu"))backend=PLAMO_BACKEND_CPU;
 else if(!strcmp(argv[1],"metal"))backend=PLAMO_BACKEND_METAL;
 else if(!strcmp(argv[1],"metal-fp32"))backend=PLAMO_BACKEND_METAL_FP32;
 else return 64;
 enum plamo_memory_mode memory=PLAMO_MEMORY_AUTO;
 if(argc>3){if(!strcmp(argv[3],"resident"))memory=PLAMO_MEMORY_RESIDENT;else if(!strcmp(argv[3],"stream"))memory=PLAMO_MEMORY_STREAM;else if(strcmp(argv[3],"auto"))return 64;}
 struct plamo_engine_info info;char error[1024];
 engine=plamo_engine_open(memory,0,4,backend,&info,error,sizeof error);
 if(!engine){fprintf(stderr,"%s\n",error);return 1;}
 if(!plamo_engine_prepare(engine,error,sizeof error)){fprintf(stderr,"%s\n",error);return 1;}
#else
 if(!plamo_model_open())return 1;
#endif
 struct model *m=&plamo_model;report("open",now()-start,m);
 uint32_t fixed_ids[]={1,16,48472,19,22125,16,2225,17,1943,18,1109,19,7272,46,16,4160,17,1943,18,1385,19};
 const uint32_t *ids=fixed_ids;size_t n=sizeof fixed_ids/sizeof *fixed_ids;int fixed_probe=1;
#ifdef PLAMO_RUNTIME_BACKEND
 if(argc>4){uint64_t count=plamo_engine_tokenize(engine,argv[4],argc>5?(unsigned)strtoul(argv[5],NULL,10):1,&ids,error,sizeof error);if(!count||count==UINT64_MAX)return 7;n=(size_t)count;fixed_probe=!strcmp(argv[4],"Hello.")&&(argc<=5||!strcmp(argv[5],"1"));}
#endif
 char phase[64];snprintf(phase,sizeof phase,"prefill_%zu",n);
 float *states=calloc(n*m->hidden,4);if(!states)return 2;
 start=now();if(!plamo_prefill(m,ids,n,states))return 3;report(phase,now()-start,m);
 uint64_t token=plamo_argmax(m->logits,m->vocab);if(fixed_probe&&token!=51871)return 4;
 start=now();if(!plamo_forward(m,token))return 5;report("decode",now()-start,m);
 if(fixed_probe&&plamo_argmax(m->logits,m->vocab)!=16)return 6;free(states);return 0;
}
