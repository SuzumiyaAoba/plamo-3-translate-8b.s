#include <stddef.h>
#include <stdio.h>
#include <string.h>
extern void *plamo_map_file_test(const char *,size_t *);
extern void plamo_unmap_file(void *,size_t);
extern int plamo_write_all(int,const void*,size_t);
int main(void) {
    size_t n=99;
    void *p=plamo_map_file_test("docs/config.json",&n);
    if(!p || n<100 || memcmp(p,"{\n",2)) return 1;
    plamo_unmap_file(p,n);
    p=plamo_map_file_test("/this/path/does/not/exist",&n);
    if(p || n) return 2;
    p=plamo_map_file_test("/dev/null",&n);
    if(p || n) return 3;
    plamo_unmap_file(0,0);
    if(!plamo_write_all(-1,0,0)||plamo_write_all(-1,"x",1))return 4;
    puts("Darwin mmap: PASS (file contents, missing file, empty file, cleanup)");
}
