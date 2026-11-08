#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
struct mapping {void *base;uint64_t bytes;};
extern struct mapping plamo_weight_mappings[4];
extern int64_t plamo_weight_fds[4];
extern uint64_t plamo_io_policy,plamo_io_stats[4];
extern const void *plamo_weight_acquire(const void*,uint64_t);
int main(void){
 const size_t n=1024*1024+128;unsigned char *expected=malloc(n);for(size_t i=0;i<n;i++)expected[i]=(unsigned char)(i*17+i/7);
 FILE *f=tmpfile();if(!f||fwrite(expected,1,n,f)!=n||fflush(f))return 1;int fd=fileno(f);void *m=mmap(0,n,PROT_READ,MAP_PRIVATE,fd,0);if(m==MAP_FAILED)return 2;
 plamo_weight_mappings[0]=(struct mapping){m,n};plamo_weight_fds[0]=fd;plamo_io_policy=2;
 const void *p=plamo_weight_acquire(m,1024*1024);if(p==m||memcmp(p,expected,1024*1024))return 3;
 p=plamo_weight_acquire((char*)m+37,1024*1024);if(p==(char*)m+37||memcmp(p,expected+37,1024*1024))return 4;
 if(plamo_io_stats[0]!=2*1024*1024||plamo_io_stats[1]!=2)return 5;
 plamo_weight_fds[0]=-1;if(plamo_weight_acquire(m,1024*1024)!=m)return 6;
 plamo_io_policy=0;if(plamo_weight_acquire(m,1024*1024)!=m)return 7;
 plamo_io_policy=1;if(plamo_weight_acquire(m,1024*1024)!=m)return 8;
 plamo_io_policy=2;p=plamo_weight_acquire(expected,1024*1024);if(p!=expected)return 9;
 munmap(m,n);fclose(f);free(expected);
 puts("weight I/O: PASS (original-byte staging, unaligned offsets, policy/fd/non-model fallback)");
}
