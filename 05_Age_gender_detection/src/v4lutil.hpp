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
#include <opencv2/opencv.hpp>
struct Buffer
{
    uint32_t length = 0;
    void * start;
};
// Designed to make 
class V4L_ZeroCopyFB
{
    public:
    V4L_ZeroCopyFB(void *pointer, int width, int height, int fd, v4l2_buffer v4lBuffer, __u32 pixelFormat);
    cv::Mat fb;
    // On the destructor we will free the buffer
    ~V4L_ZeroCopyFB();
    const __u32 mPixelFormat;

    private:
    v4l2_buffer v4l;
    int fd;
};

class V4LUtil
{
    public:
    V4LUtil(std::string device, int width, int height, int numBuffers,__u32 pixelFormat);
    void Start();
    void Stop();
    std::shared_ptr<V4L_ZeroCopyFB> ReadFrame();
    int mWidth;
    int mHeight;
    std::string mDevice;
    const __u32 mPixelFormat;


    private:
    std::vector<Buffer> buffers;
    int fd;
    __u32 pixelFormat;
    


};