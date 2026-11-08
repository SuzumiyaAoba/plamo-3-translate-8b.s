#include <math.h>
#include <stdint.h>
#include <stdio.h>
extern void plamo_rope_table(const float*,uint64_t,uint64_t,float*,float*,float);
extern const float plamo_rope_local[64],plamo_rope_global[64],plamo_rope_local_scale,plamo_rope_global_scale;
int main(void){
 float c[64],s[64];uint64_t positions[]={0,1,2048,4096,262143};
 for(int mode=0;mode<2;mode++){const float *f=mode?plamo_rope_global:plamo_rope_local;float scale=mode?plamo_rope_global_scale:plamo_rope_local_scale;
 for(int p=0;p<5;p++){plamo_rope_table(f,positions[p],64,c,s,scale);for(int j=0;j<64;j++){float a=(float)positions[p]*f[j];if(fabs(c[j]-cos(a)*scale)>4e-7||fabs(s[j]-sin(a)*scale)>4e-7)return 1;}}
 for(int j=1;j<64;j++)if(!(f[j]<f[j-1]&&f[j]>0))return 2;
 }
 puts("RoPE coefficient tables: PASS (local/YaRN, positions through 262143)");
}
