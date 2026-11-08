#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
struct descriptor{const float *input;const uint16_t *weight,*qn,*kn;float *out;uint64_t hidden,qh,kh,dim;};
extern void plamo_qkv(const struct descriptor*);
static float decode(uint16_t x){uint32_t b=(uint32_t)x<<16;float f;memcpy(&f,&b,4);return f;}
int main(void){
 enum{H=5,QH=4,KH=2,D=3,R=(QH+2*KH)*D};float input[H],out[R];uint16_t w[R*H],qn[D]={0x3f00,0xbf00,0},kn[D]={0,0x3f80,0xbf00};
 uint16_t vals[]={0xbf80,0,0x3f00,0x3f80};for(int i=0;i<R*H;i++)w[i]=vals[(i+i/7)%4];for(int i=0;i<H;i++)input[i]=(i-2)*0.2f;
 double ref[R];for(int i=0;i<R;i++){ref[i]=0;for(int j=0;j<H;j++)ref[i]+=decode(w[i*H+j])*input[j];}
 for(int head=0;head<QH+KH;head++){double var=0;for(int j=0;j<D;j++)var+=ref[head*D+j]*ref[head*D+j];const uint16_t *nw=head<QH?qn:kn;for(int j=0;j<D;j++)ref[head*D+j]=ref[head*D+j]/sqrt(var/D+1e-6)*(1+decode(nw[j]));}
 struct descriptor d={input,w,qn,kn,out,H,QH,KH,D};plamo_qkv(&d);
 for(int i=0;i<R;i++)if(fabs(out[i]-ref[i])>2e-6){fprintf(stderr,"element %d\n",i);return 1;}
 puts("QKV projection + head RMSNorm: PASS (Q/K normalized, V unchanged)");
}
