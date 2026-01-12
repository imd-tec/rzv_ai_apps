#include "v4lutil.hpp"
#include <opencv2/opencv.hpp>
#include "define.h"
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#define BUFFER_MODE V4L2_MEMORY_DMABUF
int ExportDMABufFromSystem(int dma_heap_fd, size_t size);
int xioctl(int fh, int request, void *arg)
{
        int r;

        do {
                r = ioctl(fh, request, arg);
        } while (-1 == r && EINTR == errno);
    return r;
}

// Request buffers using the DMABUF method
int request_buffers(int fd, int buffer_count) {
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = buffer_count;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_DMABUF;  // DMA-Buf Memory

    if (ioctl(fd, VIDIOC_REQBUFS, &req) < 0) {
        perror("Requesting buffers failed");
        return -1;
    }
    return req.count;
}


int ExportDMABufFromSystem(int dma_heap_fd, size_t size)

{
    struct dma_heap_allocation_data alloc_data;
    size_t buffer_size = size;

    // Prepare allocation data
    memset(&alloc_data, 0, sizeof(alloc_data));
    alloc_data.len = size; // Set the buffer size
    alloc_data.fd_flags = O_RDWR | O_CLOEXEC;  // permissions for the memory to be allocated; // No special flags for now

    // Allocate a DMA-BUF
    int dma_buf_fd = ioctl(dma_heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc_data);
    if (dma_buf_fd < 0) {
        std::cout << "Failed to allocate DMA buffer " <<  dma_buf_fd << std::endl;
        return dma_buf_fd;
    }
    return alloc_data.fd;
}



V4LUtil::V4LUtil(std::string device, int width, int height, int numBuffers,__u32 pixelFormat) : mDevice(device), 
    mWidth(width), mHeight(height), mPixelFormat(pixelFormat)
{
    std::cout << "Opening V4L device: " << device << std::endl;
    fd = open(device.c_str(), O_RDWR);
    if (fd == -1) {
        std::cerr << "Error opening device: " << strerror(errno) << std::endl;
        std::cerr << "Device path: " << device << std::endl;
    }

    // Query device capabilities
    v4l2_capability cap;
    if (xioctl(fd, VIDIOC_QUERYCAP, &cap) == -1) {
        std::cerr << "Error querying capabilities for device " <<device<<  ": " << strerror(errno) << std::endl;
        close(fd);

    }

    v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    // Try multiplanar first
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    fmt.fmt.pix_mp.width = width;
    fmt.fmt.pix_mp.height = height;
    fmt.fmt.pix_mp.pixelformat = pixelFormat;
    fmt.fmt.pix_mp.field = V4L2_FIELD_INTERLACED;
    fmt.fmt.pix_mp.num_planes = 2; // Only valid for multiplanar
    if (xioctl(fd, VIDIOC_S_FMT, &fmt) == 0) {
        this->is_multiplanar = true;
        std::cout << "Device supports multiplanar format." << std::endl;
    } else {
        // Fallback to single-planar
        memset(&fmt, 0, sizeof(fmt));
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = width;
        fmt.fmt.pix.height = height;
        fmt.fmt.pix.pixelformat = pixelFormat;
        fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
        if (xioctl(fd, VIDIOC_S_FMT, &fmt) == -1) {
            std::cerr << "Error setting format: " << strerror(errno) << " for device: " << device << " with pixel format: " << pixelFormat << std::endl;
            close(fd);
        }
        this->is_multiplanar = false;
    }
    this->dmaBufFd = open("/dev/dma_heap/linux,cma@58000000", O_RDWR);
    if (dmaBufFd < 0) {
        std::cout << "Failed to open frameBuffer DMA-Heap" << std::endl;
    }

    #if 0 
    // Request buffer
    v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = numBuffers;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (xioctl(fd, VIDIOC_REQBUFS, &req) == -1) {
        std::cerr << "Error requesting buffer: " << strerror(errno) << std::endl;
        close(fd);
    }

    if (req.count < 1) {
        std::cerr << "Insufficient buffer memory" << std::endl;
        close(fd);
    }

    // Map the buffer
    v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = 0;

    if (xioctl(fd, VIDIOC_QUERYBUF, &buf) == -1) {
        std::cerr << "Error querying buffer: " << strerror(errno) << std::endl;
        close(fd);
    }
  
    // Allocate and map buffers
    this->buffers = std::vector<Buffer>(numBuffers);
    for (int i = 0; i < numBuffers; ++i) {
        v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_DMABUF;
        buf.index = i;
        buf.m.fd = ExportDMABufFromSystem(dma_buf_heap,1920*1080*3);

        if (xioctl(fd, VIDIOC_QUERYBUF, &buf) == -1) {
            std::cerr << "Error querying buffer: " << strerror(errno) << std::endl;
            close(fd);
        }

        buffers[i].length = buf.length;
        buffers[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, buf.m.fd , buf.m.offset);

        if (buffers[i].start == MAP_FAILED) {
            std::cerr << "Error mapping buffer " << i << ": " << strerror(errno) << std::endl;
            close(fd);
        }
    }
    #else

    this->buffers = std::vector<Buffer>(numBuffers);
    #define SIZE 1920*1080*3
    if (this->is_multiplanar) {
        std::cout << "Using multiplanar buffers" << std::endl;
        struct v4l2_requestbuffers reqbuf;
        memset(&reqbuf, 0, sizeof(reqbuf));
        reqbuf.count = numBuffers;
        reqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        reqbuf.memory = V4L2_MEMORY_DMABUF;
        if (ioctl(fd, VIDIOC_REQBUFS, &reqbuf) < 0) {
            std::cout << "Failed to request multiplanar buffers" << std::endl;
        }
        for (int i = 0; i < numBuffers; ++i) {
            auto dmaBuf = ExportDMABufFromSystem(this->dmaBufFd, SIZE);
            if (dmaBuf > 0) {
                buffers[i].DMABufFD = dmaBuf;
                buffers[i].start = mmap(NULL, SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, buffers[i].DMABufFD, 0);
                buffers[i].length = SIZE;
                std::cout << "Mapped multiplanar dma buf with size " << SIZE << " At fd: " << buffers[i].DMABufFD << std::endl;
            } else {
                std::cout << "Couldn't map multiplanar DMA buffer:  " << dmaBuf << std::endl;
            }
        }
    } else {
        struct v4l2_requestbuffers reqbuf;
        memset(&reqbuf, 0, sizeof(reqbuf));
        reqbuf.count = numBuffers;
        reqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        reqbuf.memory = V4L2_MEMORY_DMABUF;
        if (ioctl(fd, VIDIOC_REQBUFS, &reqbuf) < 0) {
            std::cout << "Failed to request buffers " << std::endl;
        }
        for (int i = 0; i < numBuffers; ++i) {
            auto dmaBuf = ExportDMABufFromSystem(this->dmaBufFd, SIZE);
            if (dmaBuf > 0) {
                buffers[i].DMABufFD = dmaBuf;
                buffers[i].start = mmap(NULL, SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, buffers[i].DMABufFD, 0);
                buffers[i].length = SIZE;
                std::cout << "Mapped dma buf with size " << SIZE << " At fd: " << buffers[i].DMABufFD << std::endl;
            } else {
                std::cout << "Couldn't map DMA buffer:  " << dmaBuf << std::endl;
            }
        }
    }
    
    #endif

}
 


void V4LUtil::Start()
{
    for (int i = 0; i < buffers.size(); ++i) {
        if (this->is_multiplanar) {
            v4l2_buffer buf;
            memset(&buf, 0, sizeof(buf));
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
            buf.memory = V4L2_MEMORY_DMABUF;
            buf.index = i;
            memset(buffers[i].planes, 0, sizeof(buffers[i].planes));
            buf.m.planes = buffers[i].planes;
            buf.length = 1; // For most YUV420, 1 or 2 planes
            buffers[i].planes[0].m.fd = buffers[i].DMABufFD;
            //std::cout << "Queueing multiplanar buffer for index: " << i << "  Wtith plane pointer: " << buffers[i].planes << " FD: " << buffers[i].DMABufFD << std::endl;
            if (xioctl(fd, VIDIOC_QBUF, &buf) == -1) {
                std::cerr << "Error queueing multiplanar buffer " << i << ": " << strerror(errno) << std::endl;
            }
        } else {
            v4l2_buffer buf;
            memset(&buf, 0, sizeof(buf));
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_DMABUF;
            buf.index = i;
            buf.m.fd = buffers[i].DMABufFD;
            std::cout << "Queueing buffer " << buffers[i].DMABufFD << std::endl;
            if (xioctl(fd, VIDIOC_QBUF, &buf) == -1) {
                std::cerr << "Error queueing buffer " << i << ": " << strerror(errno) << std::endl;
            }
        }
    }

    v4l2_buf_type type = is_multiplanar ? V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE : V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd, VIDIOC_STREAMON, &type) == -1) {
        std::cerr << "Error starting stream: " << strerror(errno) << std::endl;
        close(fd);
    }
}

V4LUtil::~V4LUtil()
{

    for (int i = 0; i < buffers.size(); ++i) {
        close(buffers[i].DMABufFD);
    }
    close(fd);
}
void V4LUtil::Stop()
{
     
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
     // Stop streaming
    if (xioctl(fd, VIDIOC_STREAMOFF, &type) == -1) {
        std::cerr << "Error stopping stream: " << strerror(errno) << std::endl;
    }

    // Unmap and close
    for (int i = 0; i < buffers.size(); ++i) {
        munmap(buffers[i].start, buffers[i].length);
    }

    close(fd);
}

std::shared_ptr<V4L_ZeroCopyFB>  V4LUtil::ReadFrame()

{
    // Detect if multiplanar
   
    v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    auto  planes = std::make_shared<std::array<v4l2_plane, 1>>();
    memset(planes->data(), 0, sizeof(planes));
    if (this->is_multiplanar) {
        v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        buf.memory = V4L2_MEMORY_DMABUF;
        buf.m.planes = planes->data();
        buf.length = 1;
        if (xioctl(fd, VIDIOC_DQBUF, &buf) == -1) {
            return NULL;
        }
        //std::cout << "Dequeued multiplanar buffer index: " << buf.index << " With plane pointer: " << buf.m.planes << std::endl;
        auto &frame_buffer = buffers[buf.index];
        std::shared_ptr<V4L_ZeroCopyFB> fb = std::make_shared<V4L_ZeroCopyFB>(frame_buffer.start, mWidth, mHeight, fd, buf, mPixelFormat, planes);
        return fb;
    } else {
        v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_DMABUF;
        if (xioctl(fd, VIDIOC_DQBUF, &buf) == -1) {
            return NULL;
        }
        auto &frame_buffer = buffers[buf.index];
        std::shared_ptr<V4L_ZeroCopyFB> fb = std::make_shared<V4L_ZeroCopyFB>(frame_buffer.start, mWidth, mHeight, fd, buf, mPixelFormat, planes);
        return fb;
    }

}

V4L_ZeroCopyFB::V4L_ZeroCopyFB(void *pointer, int width, int height, int fd, v4l2_buffer v4lBuffer, __u32 pixelFormat, std::shared_ptr<std::array<v4l2_plane, 1>> &planes_ptr) : mPixelFormat(pixelFormat), planes_ptr(planes_ptr)
{
    int dataSize = CV_8UC3;
    if(pixelFormat == V4L2_PIX_FMT_BGR24 )
        dataSize = CV_8UC3;
    else if(pixelFormat == V4L2_PIX_FMT_RGB24 )
        dataSize = CV_8UC3;
    else if(pixelFormat ==  V4L2_PIX_FMT_YUYV)
        dataSize = CV_8UC2;
    this->fb = cv::Mat(cv::Size(width, height), dataSize, pointer, cv::Mat::AUTO_STEP);
    this->fd = fd;
    this->v4l = v4lBuffer;
}
 V4L_ZeroCopyFB::V4L_ZeroCopyFB(cv::Mat &fb) : mPixelFormat(0)
 {
    this->fb = fb;
    this->fd = 0;
 }
 V4L_ZeroCopyFB::~V4L_ZeroCopyFB()

 {

    //printf("Free buffer is at %p \n",this->fb.ptr() );
    if(fd)
    {
        if (xioctl(fd, VIDIOC_QBUF, &this->v4l) == -1) {
        }
    }
 }