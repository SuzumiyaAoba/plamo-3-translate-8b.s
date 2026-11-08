#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/resource.h>
extern void plamo_bf16_matvec(const uint16_t*,const float*,float*,size_t,size_t);
extern uint64_t plamo_io_policy;
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(int argc,char **argv){plamo_io_policy=0;if(argc!=5)return 1;int fd=open(argv[1],O_RDONLY);if(fd<0)return 2;off_t offset=strtoull(argv[2],0,10);size_t rows=strtoull(argv[3],0,10),cols=strtoull(argv[4],0,10),bytes=rows*cols*2;off_t length=lseek(fd,0,SEEK_END);float *x=calloc(cols,4),*a=calloc(rows,4),*b=calloc(rows,4);for(size_t i=0;i<cols;i++)x[i]=(int)(i%17)-8;
 void *mapping=mmap(0,length,PROT_READ,MAP_PRIVATE,fd,0);const uint16_t *w=(const void*)((const char*)mapping+offset);double start=now();plamo_bf16_matvec(w,x,a,rows,cols);double mm=now()-start;start=now();plamo_bf16_matvec(w,x,b,rows,cols);double warm=now()-start;if(memcmp(a,b,rows*4))return 3;
 munmap(mapping,length);mapping=mmap(0,length,PROT_READ,MAP_PRIVATE,fd,0);w=(const void*)((const char*)mapping+offset);uintptr_t page=((uintptr_t)w)&~(uintptr_t)(getpagesize()-1);start=now();int advice=madvise((void*)page,bytes+(uintptr_t)w-page,MADV_WILLNEED);plamo_bf16_matvec(w,x,b,rows,cols);double advised=now()-start;if(memcmp(a,b,rows*4))return 4;
 void *buffer=mmap(0,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);start=now();size_t done=0;while(done<bytes){ssize_t n=pread(fd,(char*)buffer+done,bytes-done,offset+done);if(n<=0)return 5;done+=n;}double readtime=now()-start;start=now();plamo_bf16_matvec(buffer,x,b,rows,cols);double compute=now()-start;if(memcmp(a,b,rows*4))return 6;
 printf("%zu MiB weights: mmap first %.3f s, warm %.3f s, WILLNEED+compute %.3f s (advice=%d), pread %.3f s + compute %.3f s\n",bytes>>20,mm,warm,advised,advice,readtime,compute);
 munmap(mapping,length);munmap(buffer,bytes);close(fd);free(x);free(a);free(b);
}
