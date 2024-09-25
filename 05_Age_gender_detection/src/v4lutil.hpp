#include <iostream>
#include <fcntl.h>              // For open()
#include <unistd.h>             // For close()
#include <sys/ioctl.h>          // For ioctl()
#include <linux/videodev2.h>    // For V4L2 device
#include <cstring>              // For memset()
#include <errno.h>              // For error handling
#include <sys/mman.h>           // For mmap()
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <getopt.h>             /* getopt_long() */

#include <fcntl.h>              /* low-level i/o */
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <stdint.h>
#include <linux/videodev2.h>
#include <vector>
#include <memory>
struct Buffer
{
    uint32_t length = 0;
    void * start;
};

class V4LUtil
{
    public:
    V4LUtil(std::string device, int width, int height, int numBuffers);
    void Start();
    void Stop();
    std::shared_ptr<std::vector<char>> ReadFrame();
    int mWidth;
    int mHeight;
    std::string mDevice;


    private:
    std::vector<Buffer> buffers;
    int fd;
    int xioctl(int fh, int request, void *arg);
    


};