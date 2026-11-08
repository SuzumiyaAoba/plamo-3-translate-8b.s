#include <stdint.h>
#include <stdio.h>
#include <math.h>
extern int plamo_model_open(void);
extern int plamo_embedding(const void*,uint64_t,uint64_t,uint64_t,float*);
extern int plamo_layer(void*,float);
struct model{void *embedding;uint64_t vocab,hidden;float *state;void **layers;uint64_t count;float *scales;};
extern struct model plamo_model;
int main(void){if(!plamo_model_open())return 1;struct model *m=&plamo_model;if(!plamo_embedding(m->embedding,7272,m->vocab,m->hidden,m->state))return 2;
 for(uint64_t i=0;i<m->count;i++){if(!plamo_layer(m->layers[i],m->scales[i]))return 3;double rms=0;for(uint64_t j=0;j<m->hidden;j++)rms+=(double)m->state[j]*m->state[j];printf("layer %llu rms %.9g\n",(unsigned long long)i,sqrt(rms/m->hidden));fflush(stdout);char path[100];snprintf(path,sizeof path,"build/asm_layer_%llu.bin",(unsigned long long)i);FILE *f=fopen(path,"wb");fwrite(m->state,sizeof(float),m->hidden,f);fclose(f);}
}
