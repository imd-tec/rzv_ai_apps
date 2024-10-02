#include "v4lutil.hpp"
#include <opencv2/opencv.hpp>
#include "define.h"
int xioctl(int fh, int request, void *arg)
{
        int r;

        do {
                r = ioctl(fh, request, arg);
        } while (-1 == r && EINTR == errno);
    return r;
}
V4LUtil::V4LUtil(std::string device, int width, int height, int numBuffers,__u32 pixelFormat) : mDevice(device), 
    mWidth(width), mHeight(height), mPixelFormat(pixelFormat)
{
    std::cout << "Opening V4L device: " << device << std::endl;
    fd = open(device.c_str(), O_RDWR);
    if (fd == -1) {
        std::cerr << "Error opening device: " << strerror(errno) << std::endl;
    }

    // Query device capabilities
    v4l2_capability cap;
    if (xioctl(fd, VIDIOC_QUERYCAP, &cap) == -1) {
        std::cerr << "Error querying capabilities: " << strerror(errno) << std::endl;
        close(fd);

    }
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        std::cerr << "Device does not support video capture" << std::endl;
        close(fd);
    }

    // Set the capture format
    v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width;    // Set the width
    fmt.fmt.pix.height = height;   // Set the height
    fmt.fmt.pix.pixelformat = pixelFormat;  // RGB format
    fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;

    if (xioctl(fd, VIDIOC_S_FMT, &fmt) == -1) {
        std::cerr << "Error setting format: " << strerror(errno) << std::endl;
        close(fd);
    }

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
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (xioctl(fd, VIDIOC_QUERYBUF, &buf) == -1) {
            std::cerr << "Error querying buffer: " << strerror(errno) << std::endl;
            close(fd);
        }

        buffers[i].length = buf.length;
        buffers[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);

        if (buffers[i].start == MAP_FAILED) {
            std::cerr << "Error mapping buffer " << i << ": " << strerror(errno) << std::endl;
            close(fd);
        }
    }

}
 


void V4LUtil::Start()
{
    // Queue all buffers
    for (int i = 0; i < buffers.size(); ++i) {
        v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (xioctl(fd, VIDIOC_QBUF, &buf) == -1) {
            std::cerr << "Error queueing buffer " << i << ": " << strerror(errno) << std::endl;
            close(fd);
        }
    }

    // Start streaming
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd, VIDIOC_STREAMON, &type) == -1) {
        std::cerr << "Error starting stream: " << strerror(errno) << std::endl;
        close(fd);
    }
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
    v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (xioctl(fd, VIDIOC_DQBUF, &buf) == -1) {
        std::cerr << "Error dequeueing buffer: " << strerror(errno) << std::endl;
        return NULL;

    }
    auto &frame_buffer = buffers[buf.index];

    // auto fb = std::make_shared<std::vector<char>>(buf.bytesused);
    // memcpy(fb->data(),buffer.start,buffer.length);
    std::shared_ptr<V4L_ZeroCopyFB>  fb = std::make_shared<V4L_ZeroCopyFB>(frame_buffer.start,mWidth,mHeight,fd,buf,mPixelFormat);
    //std::cout << "(" << buf.index << ")Captured frame for device" <<  this->mDevice << " size: " << buf.bytesused << " bytes" << std::endl;
    // Requeue the buffer
    return fb;

}

V4L_ZeroCopyFB::V4L_ZeroCopyFB(void *pointer, int width, int height, int fd, v4l2_buffer v4lBuffer, __u32 pixelFormat ) : mPixelFormat(pixelFormat)
{
    int dataSize = CV_8UC3;
    if(pixelFormat == V4L2_PIX_FMT_BGR24 )
        dataSize = CV_8UC3;
    else if(pixelFormat ==  V4L2_PIX_FMT_YUYV)
        dataSize = CV_8UC2;
    this->fb = cv::Mat(cv::Size(width, height), dataSize, pointer, cv::Mat::AUTO_STEP);
    this->fd = fd;
    this->v4l = v4lBuffer;
}

 V4L_ZeroCopyFB::~V4L_ZeroCopyFB()

 {
    //printf("Free buffer is at %p \n",this->fb.ptr() );
    if (xioctl(fd, VIDIOC_QBUF, &this->v4l) == -1) {
        std::cerr << "Error requeueing buffer: " << strerror(errno) << std::endl;
    }
 }