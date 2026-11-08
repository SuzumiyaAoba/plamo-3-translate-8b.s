#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <limits.h>
struct mapping {void *base;uint64_t bytes;};
extern struct mapping plamo_weight_mappings[4];
extern int64_t plamo_weight_fds[4];
extern int plamo_model_open(void);
int main(void){
 char root[PATH_MAX];if(!getcwd(root,sizeof root))return 1;char dir[]="/tmp/plamo-retry-XXXXXX";if(!mkdtemp(dir))return 2;
 char models[PATH_MAX],folder[PATH_MAX],paths[2][PATH_MAX];snprintf(models,sizeof models,"%s/models",dir);snprintf(folder,sizeof folder,"%s/models/plamo",dir);if(mkdir(models,0700)||mkdir(folder,0700))return 3;
 for(int i=0;i<2;i++){char source[PATH_MAX];snprintf(source,sizeof source,"%s/models/plamo/model-%05d-of-00004.safetensors",root,i+1);snprintf(paths[i],sizeof paths[i],"%s/model-%05d-of-00004.safetensors",folder,i+1);if(symlink(source,paths[i]))return 4;}
 if(chdir(dir)||plamo_model_open())return 5;
 for(int i=0;i<4;i++)if(plamo_weight_fds[i]!=-1||plamo_weight_mappings[i].base||plamo_weight_mappings[i].bytes)return 6;
 if(chdir(root)||!plamo_model_open())return 7;
 for(int i=0;i<4;i++)if(plamo_weight_fds[i]<0||!plamo_weight_mappings[i].base)return 8;
 for(int i=0;i<2;i++)unlink(paths[i]);rmdir(folder);rmdir(models);rmdir(dir);
 puts("model open retry: PASS (partial failure releases mappings/descriptors, retry succeeds)");
}
