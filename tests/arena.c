#include <stdint.h>
#include <stdio.h>
extern void *plamo_allocate(uint64_t);
extern void plamo_unmap_file(void*,uint64_t);
extern void plamo_arena_patch(void*,const void*);
int main(void){uint64_t n=5ULL*1024*1024*1024;unsigned char *p=plamo_allocate(n);if(!p||p[0]||p[n-1])return 1;p[0]=1;p[n-1]=2;void *slot=0;uintptr_t table[]={(uintptr_t)&slot,n-1,0,0};plamo_arena_patch(p,table);if(slot!=p+n-1||*(unsigned char*)slot!=2)return 2;plamo_unmap_file(p,n);if(plamo_allocate(0))return 3;puts("dynamic arena: PASS (5 GiB virtual mapping, sparse access, pointer patches)");}
